"""HOSTLIB SN2GAL_Z (MSKOPT bit 16) with BayeSN.

Two complementary checks on the same fixture:
- SIM_MAGOBS replay parity: SNANA's dumped GENZ/MU are the post-swap values;
  replaying them pins the final-values → flux mapping under this config.
- Ds/mu consistency on JAX's OWN sampling flow: SN2GAL_Z recomputes mu
  after sampling, and shift_amplitude must carry that into the SED's Ds
  (this is the assertion that fails without the fix — the replay path
  bypasses the hook, so it cannot catch a stale Ds).
"""

import numpy as np
from _replay_helpers import load_jax_phot, load_snana_phot, slice_replay
from astropy.io import fits
from replay_snana_dump import build_override_parameters

from jax_snlc_core.snana_files import load_snana_dump
from jax_snlc_sim import run as sim_run
from jax_snlc_sim.output import save_snana_outputs


def test_sn2galz_bayesn_simmagobs_parity(snana_fixture, tmp_path):
    """JAX replay reproduces SNANA's SIM_MAGOBS under HOSTLIB SN2GAL_Z."""
    fx = snana_fixture("hostlib_sn2galz_bayesn")

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
    override = build_override_parameters(dump_cols, simulation, resources, head_columns=head_cols)

    results, sel, spectra_cache = sim_run.run_simulation(sim, override_parameters=override)
    save_snana_outputs(sim, results, "SN2GALZ_REPLAY", sel, spectra_by_sn=spectra_cache, output_dir=str(tmp_path))

    snana_phot = load_snana_phot(fx.head_fits, fx.phot_fits)
    jax_phot = load_jax_phot(tmp_path / "SN2GALZ_REPLAY", "SN2GALZ_REPLAY")

    diffs = []
    common = set(snana_phot) & set(jax_phot)
    assert len(common) > 0, "no overlapping (SNID, MJD, BAND) keys"
    for key in common:
        sn_mag, sn_flux, sn_err, _ = snana_phot[key]
        jx_mag = jax_phot[key][0]
        if sn_err <= 0 or sn_flux / sn_err < 3.0:
            continue
        if sn_mag >= 50.0 or jx_mag >= 50.0:
            continue
        diffs.append(jx_mag - sn_mag)

    assert len(diffs) > 650, f"too few S/N≥3 epochs to compare ({len(diffs)})"
    abs_diffs = np.abs(np.array(diffs))
    assert float(np.median(abs_diffs)) < 0.005
    assert float(np.percentile(abs_diffs, 95)) < 0.05


def test_sn2galz_bayesn_ds_tracks_the_distance(snana_fixture, tmp_path):
    """JAX's own flow: after the SN2GAL_Z distance recompute, the SED's Ds must
    equal mu + deltam for every SN (shift_amplitude reached the flux)."""
    fx = snana_fixture("hostlib_sn2galz_bayesn")
    config = dict(n_sne=200,
                                 random_seed=99887)
    sim = sim_run.prepare(str(fx.input_file), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    results, _, _ = sim_run.run_simulation(sim)
    p = results['parameters']

    ds = np.asarray(p['Ds'])
    deltam = np.asarray(p['deltam'])
    mu = np.asarray(p['mu'])
    np.testing.assert_allclose(ds - deltam, mu, rtol=0, atol=1e-5)


def test_sn2galz_host_redshifts_match_snanas_own_draw(snana_fixture):
    """Under SN2GAL_Z the event's redshift is the chosen galaxy's, so the
    generated redshift distribution is SNANA's own weighted galaxy draw over
    the library. jax builds that draw once as a marginal CDF over the
    catalogue; nothing else in the suite compares it with SNANA. The two
    RNGs differ, so this compares the distribution, not the event."""
    fx = snana_fixture("hostlib_sn2galz_bayesn")
    z_snana = np.asarray(
        load_snana_dump(str(fx.dump), dedup_cid=True).columns['GENZ'],
        dtype=np.float64)

    config = dict(n_sne=len(z_snana), random_seed=2029)
    sim = sim_run.prepare(str(fx.input_file), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    assert resources.hostlib.catalogue.marginal_cdf is not None, (
        "HOSTLIB_MSKOPT bit 16 must build the marginal CDF this pins")
    results, _, _ = sim_run.run_simulation(sim)
    z_jax = np.asarray(results['parameters']['sim_redshift_cmb'],
                       dtype=np.float64)

    assert abs(z_jax.mean() - z_snana.mean()) < 0.01
    for q in (0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98):
        qs, qj = float(np.quantile(z_snana, q)), float(np.quantile(z_jax, q))
        assert abs(qs - qj) < 0.015, (q, qs, qj)
