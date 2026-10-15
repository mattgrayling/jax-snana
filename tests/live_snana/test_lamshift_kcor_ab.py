"""FUDGESHIFT_LAM vs SNANA's kcor.exe FILTER_LAMSHIFT (A/B).

SNANA's sim-side FUDGESHIFT_LAM key is dead (TMPOFF_LAMFILT is never read);
its supported route bakes the shift into the kcor via
``kcor.exe <input> FILTER_LAMSHIFT <band> <A>``. This test generates a
reference and a z+100A kcor pair from the same recipe, simulates with SNANA
using the shifted kcor, and replays with JAX using the reference kcor plus
our FUDGESHIFT_LAM key — the two shift routes must agree at the replay floor.
Requires --live-snana (kcor.exe + snlc_sim.exe on PATH).
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for extra in ("tests/parity", "."):
    p = str(PROJECT_ROOT / extra)
    if p not in sys.path:
        sys.path.insert(0, p)

from jax_snlc_core.io.snana_dump_reader import load_snana_dump  # noqa: E402
from replay_snana_dump import build_override_parameters  # noqa: E402
from jax_snlc_sim import run as sim_run  # noqa: E402
from _replay_helpers import load_snana_phot, load_jax_phot  # noqa: E402

KCOR_RECIPE = Path(os.path.expandvars(
    "$SNDATA_ROOT/kcor/LSST/baseline_1.9/kcor_LSST.input"))
GENVERSION = "JAX_LIVE_LAMX_AB"


def test_lamshift_kcor_ab(tmp_path):
    if not KCOR_RECIPE.is_file():
        pytest.skip("LSST baseline_1.9 kcor recipe not available")
    if shutil.which("kcor.exe") is None:
        pytest.skip("kcor.exe not on PATH")

    # SNANA path buffers overflow on deep paths; keep a short workdir.
    work = Path(os.path.expandvars("$SNDATA_ROOT")) / "SIM" / "_lamx_work"
    work.mkdir(parents=True, exist_ok=True)
    recipe = KCOR_RECIPE.read_text()
    (work / "kcor_ref.input").write_text(
        re.sub(r"OUTFILE:.*", "OUTFILE: kcor_ref.fits", recipe))
    (work / "kcor_shift.input").write_text(
        re.sub(r"OUTFILE:.*", "OUTFILE: kcor_shift.fits", recipe))
    for cmd in (["kcor.exe", "kcor_ref.input"],
                ["kcor.exe", "kcor_shift.input", "FILTER_LAMSHIFT", "z", "100"]):
        r = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-1500:]

    base = (PROJECT_ROOT / "tests" / "fixtures" / "snana"
            / "fudge_knobs_replay" / "input.input").read_text()
    for key in ("FUDGESHIFT_ZPT_FILTER:  r 0.30",
                "FUDGESCALE_FLUXERR_FILTER:  g 1.30",
                "FUDGESCALE_FLUXERR2_FILTER:  i 1.20"):
        base = base.replace(key + "\n", "")
    base = re.sub(r"GENVERSION:\s+\S+", f"GENVERSION:  {GENVERSION}", base)

    snana_inp = re.sub(r"KCOR_FILE:\s+\S+",
                       f"KCOR_FILE:  {work}/kcor_shift.fits", base)
    (work / "snana.input").write_text(snana_inp)
    env = os.environ.copy()
    r = subprocess.run(["snlc_sim.exe", str(work / "snana.input")],
                       cwd=work, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-1500:]
    sim_out = Path(os.path.expandvars("$SNDATA_ROOT")) / "SIM" / GENVERSION

    jax_inp = re.sub(r"KCOR_FILE:\s+\S+",
                     f"KCOR_FILE:  {work}/kcor_ref.fits", base)
    jax_inp = jax_inp.replace("GENMODEL:",
                              "FUDGESHIFT_LAM_FILTER:  z 100.0\nGENMODEL:")
    jax_file = tmp_path / "jax.input"
    jax_file.write_text(jax_inp)

    dump = load_snana_dump(str(next(sim_out.glob("*.DUMP"))), dedup_cid=True)
    with fits.open(next(sim_out.glob("*HEAD.FITS*"))) as h:
        head_cols = {n: h[1].data[n] for n in h[1].data.names}
    config = dict(n_sne=dump.size)
    sim = sim_run.prepare(str(jax_file), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    assert simulation.instrument.fudgeshift_lam.get('z') == 100.0
    override = build_override_parameters(dump.columns, simulation, resources,
                                         head_columns=head_cols)

    results, sel, spectra_cache = sim_run.run_simulation(sim, override_parameters=override)
    sim_run.save_snana_outputs(sim, results, "LX", sel, spectra_cache=spectra_cache, output_dir=str(tmp_path))

    snana_phot = load_snana_phot(next(sim_out.glob("*HEAD.FITS*")),
                                 next(sim_out.glob("*PHOT.FITS*")))
    jax_phot = load_jax_phot(tmp_path / "LX", "LX")

    per_band = {}
    for key in set(snana_phot) & set(jax_phot):
        _, band, _ = key
        sn_mag, sn_flux, sn_err, _ = snana_phot[key]
        jx_mag = jax_phot[key][0]
        if sn_mag < 50 and jx_mag < 50 and sn_err > 0 and sn_flux / sn_err >= 3:
            per_band.setdefault(band.split('-')[-1], []).append(abs(jx_mag - sn_mag))

    z = np.array(per_band.get('z', []))
    assert len(z) > 100, f"too few z epochs ({len(z)})"
    assert float(np.median(z)) < 0.005, f"z median |dmag| {np.median(z):.4f}"
    others = np.concatenate([np.array(v) for b, v in per_band.items() if b != 'z'])
    assert float(np.median(others)) < 0.005
