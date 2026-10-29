"""SALT3 GENMODEL_EXTRAP_LATETIME (FLAM) SIM_MAGOBS parity, late epochs only.

The general SALT3 replay path is pinned by test_salt3_k21_replay; this fixture
extends GENRANGE_TREST to +100d with the FLAM late-time file (which SNANA
implements for SALT in genmag_SALT2.c), and the comparison is restricted to
rest-frame epochs beyond EXTRAP_DAYMIN (45d) — the epochs served by the FLAM
extrapolation. This also pins the shared LatetimeExtrapIa machinery (file
parsing, per-wavelength tau interpolation, decay factor).
"""

import numpy as np
from _replay_helpers import load_jax_phot, load_snana_phot, slice_replay
from astropy.io import fits
from replay_snana_dump import build_override_parameters

from jax_snlc_core.snana_files import load_snana_dump
from jax_snlc_sim import run as sim_run
from jax_snlc_sim.output import save_snana_outputs

_DAYMIN = 45.0  # EXTRAP_DAYMIN in EXTRAP_LATETIME_SNIa.DAT


def test_salt3_latetime_flam_simmagobs_parity(snana_fixture, tmp_path):
    """JAX replay reproduces SNANA's SIM_MAGOBS beyond the FLAM daymin."""
    fx = snana_fixture("salt3_latetime_replay")

    dump = load_snana_dump(str(fx.dump), dedup_cid=True)
    with fits.open(fx.head_fits) as hdul:
        head_cols = {n: hdul[1].data[n] for n in hdul[1].data.names}

    # 300 events; guard below is ~80% of the resulting yield so a silent
    # loss of matched epochs fails instead of shrinking the sample
    dump_cols, head_cols = slice_replay(dump.columns, head_cols, 300)
    config = dict(n_sne=len(dump_cols['CID']))
    sim = sim_run.prepare(str(fx.input_file), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    override = build_override_parameters(dump_cols, simulation, resources,
                                         head_columns=head_cols)

    results, sel, spectra_cache = sim_run.run_simulation(sim, override_parameters=override)
    save_snana_outputs(sim, results, "LATETIME_REPLAY", sel, spectra_by_sn=spectra_cache, output_dir=str(tmp_path))

    snana_phot = load_snana_phot(fx.head_fits, fx.phot_fits)
    jax_phot = load_jax_phot(tmp_path / "LATETIME_REPLAY", "LATETIME_REPLAY")

    # Per-SN peak/redshift from the dump, for rest-frame phase computation.
    cids = np.asarray(dump.columns["CID"]).astype(int)
    peakmjd = {int(c): float(p) for c, p
               in zip(cids, np.asarray(dump.columns["PEAKMJD"]))}
    genz = {int(c): float(z) for c, z
            in zip(cids, np.asarray(dump.columns["GENZ"]))}

    diffs = []
    common = set(snana_phot) & set(jax_phot)
    assert len(common) > 0, "no overlapping (SNID, MJD, BAND) keys"
    for key in common:
        snid = key[0]
        sn_mag, sn_flux, sn_err, mjd = snana_phot[key]
        trest = (mjd - peakmjd[snid]) / (1.0 + genz[snid])
        if trest <= _DAYMIN:
            continue
        jx_mag = jax_phot[key][0]
        if sn_err <= 0 or sn_flux / sn_err < 3.0:
            continue
        if sn_mag >= 50.0 or jx_mag >= 50.0:
            continue
        diffs.append(jx_mag - sn_mag)

    assert len(diffs) > 550, (
        f"too few S/N≥3 late-time (Trest>{_DAYMIN}d) epochs to compare "
        f"({len(diffs)})")
    abs_diffs = np.abs(np.array(diffs))
    median = float(np.median(abs_diffs))
    p95 = float(np.percentile(abs_diffs, 95))

    assert median < 0.005, f"late-time median |Δmag| = {median:.4f} > 0.005"
    assert p95 < 0.05, f"late-time 95th percentile |Δmag| = {p95:.4f} > 0.05"
