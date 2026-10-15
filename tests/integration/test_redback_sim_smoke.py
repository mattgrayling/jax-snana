"""End-to-end simulation with a redback-jax Arnett GENMODEL."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("redback_jax")

from jax_snlc_sim import run as sim_run

_F = "$JAX_TEST_FIXTURE_ROOT"


def _write_input(out: Path) -> None:
    out.write_text(f"""\
GENVERSION: REDBACK_SMOKE
NGENTOT_LC: 100
FORMAT_MASK: 32
GENTYPE: 10
SNTYPES_Ia: 10 110
GENMODEL: REDBACK.arnett_bolometric
GENPEAK_f_nickel: 0.1
GENSIGMA_f_nickel: 0.02 0.02
GENRANGE_f_nickel: 0.01 0.5
GENPEAK_mej: 2.0
GENSIGMA_mej: 0.5 0.5
GENRANGE_mej: 0.3 10.0
GENPEAK_vej: 9000.0
GENSIGMA_vej: 1000 1000
GENRANGE_vej: 3000 20000
GENPEAK_kappa: 0.1
GENPEAK_kappa_gamma: 0.03
GENPEAK_temperature_floor: 4000.0
OMEGA_MATTER: 0.315
OMEGA_LAMBDA: 0.685
w0_LAMBDA: -1.0
wa_LAMBDA: 0
GENFILTERS: ugrizY
KCOR_FILE: {_F}/kcor/JAX_TEST_LSST.fits
SIMLIB_FILE: {_F}/simlib/JAX_TEST_LSST.SIMLIB
GENRANGE_REDSHIFT: 0.02 0.15
GENRANGE_PEAKMJD: 60540 61460
GENSIGMA_REDSHIFT: 0.001
GENRANGE_TREST: -20 50
SOLID_ANGLE: 0.015
OPT_MWEBV: 0
DNDZ: POWERLAW 2.5E-5 1.5
RANSEED: 4321
""")


def _run(tmp_path, batch_size=None):
    inp = tmp_path / "input.input"
    _write_input(inp)
    # the tests read every event's light curve as an (event, epoch) grid
    kwargs = dict(n_sne=100, random_seed=4321, keep_rejected_events=True)
    if batch_size is not None:
        kwargs['batch_size'] = batch_size
    cfg = dict(**kwargs)
    sim = sim_run.prepare(str(inp), **cfg)
    resources = sim
    results, selection, _ = sim_run.run_simulation(sim)
    return resources, results, selection


@pytest.fixture(scope="module")
def arnett_run(tmp_path_factory):
    """The default 100-SN Arnett run, shared; read-only in every consumer."""
    return _run(tmp_path_factory.mktemp("arnett"))


def test_arnett_sim_end_to_end(arnett_run):
    resources, results, selection = arnett_run
    assert int(selection['n_passed']) > 0

    # Explosion anchoring: peak-SED evaluation at phase 0 is the explosion
    # epoch (zero flux), so SIM_PEAKMAG is sentinel-valued for redback.
    peakmags = np.asarray(results['peakmags'])
    assert not np.any(np.isfinite(peakmags) & (peakmags < 90))

    # Real light-curve epochs carry genuine magnitudes.
    mags = np.asarray(results['magnitudes'])
    valid = np.isfinite(mags) & (mags < 90)
    assert np.any(valid)
    assert np.all((mags[valid] > 10) & (mags[valid] < 40))

    header = resources.model.get_snana_header_params(
        {k: np.asarray(v) for k, v in results['parameters'].items()})
    assert 'SIM_REDBACK_mej' in header


def test_epoch_mag_matches_hand_integration(arnett_run):
    """Engine epoch magnitudes equal an engine-free AB synthetic-photometry
    integration of the model SED through the raw kcor filter curves."""
    import jax.numpy as jnp

    resources, results, _ = arnett_run
    p = results['parameters']
    mags = np.asarray(results['magnitudes'])
    obs = results['observation_schedule']
    filter_names = list(resources.photometry.transmissions.keys())
    model = resources.model
    kcor = resources.kcor

    z = np.asarray(p['sim_redshift_helio'])
    phases = np.asarray(obs['phases'])  # rest-frame, explosion-relative
    fidx = np.asarray(obs['filter_indices'])

    # First SN/epoch with a real magnitude in g/r/i, 10-40d after explosion.
    checked = 0
    for i in range(mags.shape[0]):
        per_sn = {k: jnp.asarray(v)[i] for k, v in p.items()
                  if k in ('f_nickel', 'mej', 'vej', 'kappa', 'kappa_gamma',
                           'temperature_floor', 'mu')}
        lam_rest = jnp.geomspace(1000.0, 25000.0, 4000)
        static_data = {'wavelength_grid': lam_rest}
        for j in range(mags.shape[1]):
            band = filter_names[int(fidx[i, j])]
            phase = phases[i, j]
            if (mags[i, j] > 90 or not np.isfinite(mags[i, j])
                    or band not in ('LSST-g', 'LSST-r', 'LSST-i')
                    or not (10.0 < phase < 40.0)):
                continue
            sed_rest = np.asarray(model.generate_sed(float(phase), per_sn, static_data))
            lam_obs = np.asarray(lam_rest) * (1 + z[i])
            flam_obs = sed_rest / (1 + z[i])
            C = 2.99792458e18
            wave, trans = kcor.get_filter_transmission(band)
            T = np.interp(lam_obs, np.asarray(wave), np.asarray(trans),
                          left=0.0, right=0.0)
            num = np.trapezoid(flam_obs * T * lam_obs, lam_obs)
            den = np.trapezoid(3631e-23 * C / lam_obs**2 * T * lam_obs, lam_obs)
            mag_hand = -2.5 * np.log10(num / den)
            assert abs(mag_hand - mags[i, j]) < 0.01, (
                f"SN{i} {band} phase={phase:.1f}: hand={mag_hand:.4f} "
                f"engine={mags[i, j]:.4f}")
            checked += 1
        if checked >= 3:
            break
    assert checked >= 3


def test_arnett_params_batch_invariant(tmp_path, arnett_run):
    _, r1, _ = arnett_run
    _, r2, _ = _run(tmp_path, batch_size=23)
    for key in ('mej', 'f_nickel', 'vej', 'kappa', 'kappa_gamma'):
        np.testing.assert_array_equal(
            np.asarray(r1['parameters'][key]),
            np.asarray(r2['parameters'][key]), err_msg=key)
    assert 'lbol_grid' not in r1['parameters'] and 't_peak' not in r1['parameters']


def test_mjd_explode_pins_explosion(tmp_path):
    """MJD_EXPLODE pins the reference MJD to the explosion time for redback.
    OPTMASK_T0SHIFT_EXPLODE is not needed — redback SEDs are already
    explosion-anchored (phase 0 = explosion)."""
    inp = tmp_path / "input.input"
    _write_input(inp)
    inp.write_text(inp.read_text() + "MJD_EXPLODE: 61000.0\n")
    cfg = dict(n_sne=50,
                              random_seed=4321)
    sim = sim_run.prepare(str(inp), **cfg)
    resources = sim
    r, _, _ = sim_run.run_simulation(sim)
    assert np.allclose(np.asarray(r['parameters']['peakmjd_true']), 61000.0)
