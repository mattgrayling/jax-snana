"""Observer-frame dust attenuation in the spectrum generation path."""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_snlc_core.dust import load_dust_law
from jax_snlc_core.spectrograph import SpectrographCalibration
from jax_snlc_sim.hostlib.spectra import HostSpecTable, HostSpectra
from jax_snlc_sim.spectra import (_apply_mw_extinction, _host_bins_to_smeared,
                                  _seds_to_smeared, mw_transmission)


def calibration(lam_min, lam_max):
    n = len(lam_min)
    return SpectrographCalibration(
        instrument_name='TEST', magref=jnp.array([20., 25.]),
        n_lam=n, n_texpose=1, lam_min=lam_min, lam_max=lam_max,
        lam_avg=(lam_min + lam_max) / 2., lam_bin=lam_max - lam_min,
        lam_sigma=jnp.zeros(n), texpose_list=jnp.array([100.]),
        log_texpose_list=jnp.array([2.]), zp=jnp.zeros((n, 1)),
        sqsigsky=jnp.zeros((n, 1)), zp_sedmodel=jnp.zeros(n))


@pytest.mark.parametrize('law_name', ['FITZ99_EXACT', 'SOMM25', 'GORD16'])
def test_observer_wavelength_extinction(law_name):
    rest_wave = (np.linspace(1000., 4500., 61) if law_name == 'SOMM25'
                 else np.linspace(3000., 9000., 61))
    wave = rest_wave[None, :] * np.array([1., 2.])[:, None]
    ebv = np.array([0.1, 0.3])[:, None]
    law = load_dust_law(law_name)
    if law_name == 'GORD16':
        law.set_params([4.2, 0.35])
    law.prepare_wavelength_grid(rest_wave)
    original = np.asarray(law.extinction_curve(rest_wave, 3.1, av=0.31))
    actual = mw_transmission(wave, ebv, {'mw_dust_law': law, 'rv_mw': 3.1})
    expected = []
    for w, e in zip(wave, ebv[:, 0]):
        reference = load_dust_law(law_name)
        if law_name == 'GORD16':
            reference.set_params([4.2, 0.35])
        reference.prepare_wavelength_grid(w)
        expected.append(reference.apply_extinction(1., jnp.asarray(w), 3.1 * e, 3.1))
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
    np.testing.assert_array_equal(
        law.extinction_curve(rest_wave, 3.1, av=0.31), original)
    # Zero extinction and the unconfigured-law path both preserve the SED.
    static = {'mw_dust_law': law, 'rv_mw': 3.1}
    np.testing.assert_array_equal(mw_transmission(wave, np.zeros_like(ebv), static), 1.)
    sed = jnp.ones_like(wave)
    np.testing.assert_array_equal(
        _apply_mw_extinction(sed, jnp.array([1., 2.]), ebv[:, 0], rest_wave, {}), sed)


def test_redshifted_spectrum_extinction():
    """The binned spectrum agrees with direct integration beyond the rest grid."""
    wave = np.linspace(3000., 9000., 61)
    lam_min = np.array([8000., 10000., 12000.])
    lam_max = lam_min + 400.
    calib = calibration(lam_min, lam_max)

    class FlatSED:
        def generate_sed(self, phase, params, static_data):
            return jnp.ones_like(static_data['wavelength_grid']) * params['scale']

    law = load_dust_law('FITZ99_EXACT')
    law.prepare_wavelength_grid(wave)
    static = {
        'model_instance': FlatSED(), 'wavelength_grid': jnp.asarray(wave),
        'host_dust_law': load_dust_law('OFF'), 'mw_dust_law': law,
        'rv_mw': 3.1, 'fluxcal_scale_mag_offset': 1.,
    }
    opt = {'zero_neg_flam': False, 'lamspike_idx': -1, 'smearing_matrix': jnp.eye(3)}
    flux, _, smeared = _seds_to_smeared(
        {'scale': jnp.ones(1)}, jnp.zeros(1), jnp.full(1, 3.1), jnp.full(1, 0.1),
        jnp.zeros(1), jnp.full(1, 2.), jnp.asarray(wave), calib, opt, static)
    reference = load_dust_law('FITZ99_EXACT')
    reference.prepare_wavelength_grid(wave * 2.)
    transmission = np.asarray(reference.apply_extinction(1., wave * 2., 0.31, 3.1))
    expected = []
    for lo, hi in zip(lam_min / 2., lam_max / 2.):
        use = (wave >= lo) & (wave <= hi)
        expected.append(np.trapezoid(transmission[use], wave[use]) / (hi - lo) / 2.)
    np.testing.assert_allclose(flux[0], expected, rtol=1e-12)
    np.testing.assert_allclose(smeared, flux, rtol=1e-12)


@pytest.mark.parametrize('law_name', ['FITZ99_EXACT', 'SOMM25'])
def test_host_spectrum_extinction(law_name):
    wave = np.linspace(1000., 9000., 81)
    lam_min = np.array([4000., 6000., 8500.])
    if law_name == 'FITZ99_EXACT':
        lam_min[-1] = 12000.
    lam_max = lam_min + 100.
    calib = calibration(lam_min, lam_max)
    table = HostSpecTable(
        mode='SPECBASIS', wave=wave, wave_min=wave - 50., wave_max=wave + 50.,
        flam=np.ones((1, len(wave))), index=np.array([0]),
        flam_scale=1., flam_scale_powz1=0.)
    law = load_dust_law(law_name)
    law.prepare_wavelength_grid(wave)
    ebv = np.array([0.1, 0.3])
    static = {'mw_dust_law': law, 'rv_mw': 3.1}
    opt = {'lamspike_idx': -1, 'smearing_matrix': jnp.eye(3)}
    flux, _, smeared = _host_bins_to_smeared(
        HostSpectra(table, np.ones((2, 1))), np.arange(2), np.ones(2), ebv,
        np.full(2, 2.), calib, opt, static)
    reference = load_dust_law(law_name)
    reference.prepare_wavelength_grid(calib.lam_avg)
    expected = reference.apply_extinction(
        1., jnp.asarray(calib.lam_avg), 3.1 * ebv[:, None], 3.1)
    np.testing.assert_allclose(flux, expected, rtol=1e-12)
    np.testing.assert_allclose(smeared, flux, rtol=1e-12)


@pytest.mark.parametrize('scatter_name', ['COH', 'C11'])
@pytest.mark.parametrize('grey_offset', [0., .4])
def test_spectral_scatter_and_mw_truth(data_root, scatter_name, grey_offset):
    """Scheduled spectra use the selected events' existing scatter and true dust."""
    import jax
    from jax_snlc_core.models import build_model
    from jax_snlc_sim.scatter import build_smear_model
    from jax_snlc_sim.spectra import generate_simlib_spectra, _sed_to_bin_flux

    model = build_model(str(data_root / 'models/SALT3.K21'), {})
    wave = model.wavelength_grid
    smear = build_smear_model(scatter_name)
    law = load_dust_law('FITZ99_EXACT')
    law.prepare_wavelength_grid(np.asarray(wave))
    static = {
        'model_instance': model, 'wavelength_grid': wave,
        'host_dust_law': load_dust_law('OFF'), 'mw_dust_law': law,
        'rv_mw': 3.1, 'fluxcal_scale_mag_offset': model.fluxcal_scale_mag_offset,
        'gensmear': smear,
    }
    grey_factors = 10.**(-.4 * grey_offset * np.arange(3))
    if grey_offset:
        static['sn_flux_factor'] = grey_factors
    # Event 1 is excluded; event 2 must retain its own already-scaled draws.
    randoms = np.array([0., 9., 2.])[:, None] * np.ones((3, smear.n_randoms))
    params = {name: np.full(3, value) for name, value in {
        'sim_redshift_cmb': .1, 'sim_redshift_helio': .1,
        'peakmjd_true': 60000., 'av': 0., 'rv': 3.1,
        'mwebv': .2, 'sim_mwebv': 0., 'x0': 1.e-5, 'x1': 0., 'c': 0.,
    }.items()}
    params.update(cid=np.arange(3), sn_global_idx=np.arange(3),
                  _gensmear_randoms=randoms)
    lam_min = np.array([4000., 6000., 8000.])
    calib = calibration(lam_min, lam_min + 200.)
    spectra, _ = generate_simlib_spectra(
        calib, params, [np.array([60005.5])] * 3, [np.array([20.])] * 3,
        static, 0., (-20., 50.), jax.random.PRNGKey(3),
        np.array([True, False, True]), {'optmask': 128})
    assert set(spectra) == {0, 2}
    sed = model.generate_sed(5., {'x0': 1.e-5, 'x1': 0., 'c': 0.}, static)
    for event in (0, 2):
        # SIM_MWEBV is zero: a nonzero measured map value must not redden it.
        factor = 10.**(-.4 * smear.delta_mag(jnp.asarray(randoms[event]), wave))
        expected = _sed_to_bin_flux(sed * factor, wave,
                                    calib.lam_min / 1.1, calib.lam_max / 1.1)
        expected = expected / 1.1 * model.fluxcal_scale_mag_offset * grey_factors[event]
        np.testing.assert_allclose(spectra[event][0].sim_flam, expected, rtol=1e-10)
        np.testing.assert_allclose(spectra[event][0].flam, expected, rtol=1e-10)
