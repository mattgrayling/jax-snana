"""BayeSN W22 SIM_MAGOBS parity: JAX replay vs bundled SNANA PHOT.FITS."""


import numpy as np
from _replay_helpers import load_jax_phot, load_snana_phot, slice_replay
from astropy.io import fits
from replay_snana_dump import build_override_parameters

from jax_snlc_core.snana_files import load_snana_dump
from jax_snlc_sim import run as sim_run
from jax_snlc_sim.output import save_snana_outputs


def test_bayesn_w22_simmagobs_parity(snana_fixture, tmp_path):
    """JAX replay reproduces SNANA's SIM_MAGOBS for BayeSN W22."""
    fx = snana_fixture("bayesn_w22")

    # Replay: pull SNANA truth params + run JAX with them.
    dump = load_snana_dump(str(fx.dump), dedup_cid=True)
    with fits.open(fx.head_fits) as hdul:
        head_cols = {n: hdul[1].data[n] for n in hdul[1].data.names}

    n_total = dump.size
    # 300 events; guard below is ~80% of the resulting yield so a silent
    # loss of matched epochs fails instead of shrinking the sample
    dump_cols, head_cols = slice_replay(dump.columns, head_cols, 300)
    config = dict(n_sne=len(dump_cols['CID']))
    sim = sim_run.prepare(str(fx.input_file), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    override = build_override_parameters(dump_cols, simulation, resources, head_columns=head_cols)

    results, sel, spectra_cache = sim_run.run_simulation(sim, override_parameters=override)
    save_snana_outputs(sim, results, "BAYESN_REPLAY", sel, spectra_by_sn=spectra_cache, output_dir=str(tmp_path))

    snana_phot = load_snana_phot(fx.head_fits, fx.phot_fits)
    jax_phot = load_jax_phot(tmp_path / "BAYESN_REPLAY", "BAYESN_REPLAY")

    diffs = []
    common = set(snana_phot) & set(jax_phot)
    assert len(common) > 0, "no overlapping (SNID, MJD, BAND) keys"
    for key in common:
        sn_mag, sn_flux, sn_err, _ = snana_phot[key]
        jx_mag, jx_flux = jax_phot[key][0], jax_phot[key][1]
        if sn_err <= 0:
            continue
        if sn_flux / sn_err < 3.0:
            continue
        if sn_mag >= 50.0 or jx_mag >= 50.0:
            continue
        diffs.append(jx_mag - sn_mag)

    assert len(diffs) > 800, f"too few S/N≥3 epochs to compare ({len(diffs)})"
    diffs = np.array(diffs)
    abs_diffs = np.abs(diffs)
    median = float(np.median(abs_diffs))
    p95 = float(np.percentile(abs_diffs, 95))

    assert median < 0.005, f"median |Δmag| = {median:.4f} > 0.005"
    assert p95 < 0.05, f"95th percentile |Δmag| = {p95:.4f} > 0.05"
