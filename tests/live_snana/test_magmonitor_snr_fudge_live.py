"""MAGMONITOR_SNR under an active error fudge, against live snlc_sim.exe.

This is the case the bundled magmonitor_snr fixture cannot reach. SNANA builds
the monitor SNR from the variance BEFORE any error fudge (sqsig_noZ,
snlc_sim.c:27854, used at :27887) and only afterwards rescales it by
SQSIG_FINAL_DATA/SQSIG_CALC_DATA (:28193). jax folds FUDGESCALE_FLUXERR and the
SIMLIB FLUXERR_ADD into its own sqsig_no_z earlier than SNANA does
(engine.py:322-342), so the monitor has to be built from snapshots taken before
that block.

Using the post-fudge values instead still matches SNANA on any fudge-free input
— measured 8e-07 — while disagreeing by 8-34% here, so only this test pins it.

Requires --live-snana (SNANA installed + $SNDATA_ROOT).
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

from jax_snlc_sim.photometry import _fudge_fluxerr_arrays

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for extra in ("tests/parity", "."):
    p = str(PROJECT_ROOT / extra)
    if p not in sys.path:
        sys.path.insert(0, p)

from jax_snlc_core.snana_files import load_snana_dump  # noqa: E402
from replay_snana_dump import build_override_parameters  # noqa: E402
from jax_snlc_sim import run as sim_run  # noqa: E402
from jax_snlc_sim.output import save_snana_outputs

_COL = "SIM_SNRMAG22"
_GENVERSION = "JAX_LIVE_MAGMON_FUDGE"
# same float32 round-off floor as the bundled fixture; the defect this guards
# against is 5 orders of magnitude larger
_REL_TOL = 1.0e-5


def _monitor_by_epoch(head_path, phot_path):
    """{(snid, band, rank-of-MJD-in-band): SIM_SNRMAG22}."""
    with fits.open(head_path) as h:
        hd = h[1].data
        snid = np.asarray(hd["SNID"]).astype(int)
        lo = np.asarray(hd["PTROBS_MIN"]).astype(int) - 1
        hi = np.asarray(hd["PTROBS_MAX"]).astype(int)
    with fits.open(phot_path) as p:
        pd = p[1].data
        mjd = np.asarray(pd["MJD"])
        band = np.array([str(b).strip() for b in pd["BAND"]])
        mon = np.asarray(pd[_COL])
    out = {}
    for k in range(len(snid)):
        sl = slice(lo[k], hi[k])
        m, b, v = mjd[sl], band[sl], mon[sl]
        good = m > 0
        for bb in set(b[good]):
            sel = good & (b == bb)
            for rank, idx in enumerate(np.where(sel)[0][np.argsort(m[sel])]):
                out[(int(snid[k]), bb, rank)] = float(v[idx])
    return out


def test_magmonitor_snr_with_error_fudge_live(tmp_path):
    src = (PROJECT_ROOT / "tests" / "fixtures" / "snana" / "magmonitor_snr"
           / "input.input").read_text()
    s = re.sub(r"GENVERSION:\s+\S+", f"GENVERSION:  {_GENVERSION}", src)
    s += "\nFUDGESCALE_FLUXERR: 1.4\n"
    inp = tmp_path / "input.input"
    inp.write_text(s)

    env = os.environ.copy()
    r = subprocess.run(["snlc_sim.exe", str(inp)], cwd=tmp_path, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]

    out = None
    for cand in (Path(env.get("SNANA_LSST_SIM", "")) / _GENVERSION,
                 Path(env.get("SNDATA_ROOT", "")) / "SIM" / _GENVERSION):
        if cand.is_dir():
            out = cand
    assert out is not None, f"no SNANA output for {_GENVERSION}"

    head = next(out.glob("*HEAD.FITS*"))
    phot = next(out.glob("*PHOT.FITS*"))
    dump = load_snana_dump(str(next(out.glob("*.DUMP"))), dedup_cid=True)
    with fits.open(head) as h:
        head_cols = {n: np.asarray(h[1].data[n]) for n in h[1].data.names}

    config = dict(n_sne=dump.size)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    # the fudge must actually be on, or this test degenerates to the bundled one
    fudge1, _ = _fudge_fluxerr_arrays(resources)
    assert fudge1 is not None and float(np.max(np.asarray(fudge1))) > 1.0, \
        "FUDGESCALE_FLUXERR did not reach the noise model"

    override = build_override_parameters(dump.columns, simulation, resources,
                                         head_columns=head_cols)
    results, sel, spectra_cache = sim_run.run_simulation(sim, override_parameters=override)
    save_snana_outputs(sim, results, "MF", sel, spectra_by_sn=spectra_cache, output_dir=str(tmp_path))

    snana = _monitor_by_epoch(head, phot)
    jax_dir = tmp_path / "MF"
    jax_out = _monitor_by_epoch(next(jax_dir.glob("MF*HEAD*")),
                                next(jax_dir.glob("MF*PHOT*")))
    common = sorted(set(snana) & set(jax_out))
    assert len(common) > 200, f"only {len(common)} epochs matched"

    a = np.array([snana[k] for k in common])
    b = np.array([jax_out[k] for k in common])
    rel = np.abs(b - a) / a
    assert rel.max() < _REL_TOL, (
        f"worst SIM_SNRMAG22 disagreement under FUDGESCALE_FLUXERR "
        f"{rel.max():.2e} (median {np.median(rel):.2e}) over {len(common)} "
        f"epochs -- monitor built from post-fudge variances?")
