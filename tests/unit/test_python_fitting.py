"""The public fitting lifecycle, using installed resources and real kernels."""

import numpy as np
import jax.numpy as jnp
import pytest
from astropy.io import fits

from jax_snlc_core.io.fits_path import resolve_fits_path
from jax_snlc_fit import LightCurveFitter


def test_repeated_fit_keeps_model_and_previous_predictions():
    fitter = LightCurveFitter.from_model(
        'SALT3.K21', batch_size=2, n_fit_iterations=1, maxiter_round1=2)
    model = fitter.model
    calibration = fitter.calibration
    times = np.linspace(59990., 60035., 12)
    flux = 100. * np.exp(-0.5 * ((times - 60005.) / 15.) ** 2)
    first = fitter.fit(mjd=times, flux=flux, flux_err=np.full(12, 5.),
                       band='g_PS1', redshift=0.1)
    prediction = fitter.predict(first, times, 'g_PS1')
    assert first.n_sne == 1
    assert 'mB' in first.to_table().colnames
    assert fitter.epoch_lengths == [12]
    assert fitter.batch_size == 2

    second = fitter.fit(
        photometry={'snid': ['SN2'] * 8, 'mjd': times[:8], 'flux': flux[:8] / 10.,
                    'flux_err': np.full(8, .5), 'band': ['r_PS1'] * 8},
        objects={'snid': ['SN2'], 'redshift': [.1]}, zp=25.)
    assert second.context[0][0]['batch'].zp_fluxcal == 25.
    np.testing.assert_allclose(second.context[0][0]['batch'].fluxcals[0], flux[:8] / 10.)
    assert fitter.model is model
    assert fitter.calibration is calibration
    assert fitter.epoch_lengths == [12]
    assert fitter.filters == ['r_PS1']
    repeated = fitter.predict(first, times, 'g_PS1')
    np.testing.assert_allclose(repeated['flux'], prediction['flux'], rtol=1e-10)

    fitter.reset_shape()
    assert fitter.epoch_lengths == []
    assert fitter.n_z is None
    assert fitter.batch_size == 2
    assert fitter.model is model
    assert fitter.calibration is calibration


def test_redback_prior_support_and_rejected_bounds():
    pytest.importorskip("redback_jax")
    fitter = LightCurveFitter.from_model(
        'REDBACK.arnett_bolometric', batch_size=1,
        priors={'mej': {'distribution': 'uniform', 'lower': '1e0', 'upper': '2e0'}},
        bounds={'mej': (-100., 1000000.)})
    physical = jnp.ones((1, len(fitter.param_names)))
    with pytest.warns(UserWarning, match='Rejected bounds'):
        lower, upper = fitter._bounds(physical, jnp.array([0.]))
    index = fitter.param_names.index('mej')
    np.testing.assert_allclose(lower[:, index], 1.)
    np.testing.assert_allclose(upper[:, index], 2.)


def test_bayesn_bundled_model_can_be_prepared():
    fitter = LightCurveFitter.from_model('BAYESN.W22', batch_size=1)
    assert 'THETA' in fitter.param_names
    assert 'g_PS1' in fitter.calibration.get_filter_names()


def test_bayesn_arrays_return_model_distance_columns(snana_fixture, data_root):
    fixture = snana_fixture('bayesn_fit_dense')
    with fits.open(fixture.head_fits) as hdus:
        head = hdus[1].data[0]
        start, stop = int(head['PTROBS_MIN']) - 1, int(head['PTROBS_MAX'])
        redshift, peak = float(head['REDSHIFT_HELIO']), float(head['PEAKMJD'])
    with fits.open(fixture.phot_fits) as hdus:
        phot = hdus[1].data[start:stop].copy()
    fitter = LightCurveFitter.from_model(
        'BAYESN.W22', calibration=data_root / 'kcor/JAX_TEST_LSST.fits',
        batch_size=1, n_fit_iterations=1, maxiter_round1=2, opt_mwebv=0)
    result = fitter.fit(
        mjd=phot['MJD'].astype(float), flux=phot['FLUXCAL'].astype(float),
        flux_err=phot['FLUXCALERR'].astype(float),
        band=np.char.strip(phot['BAND'].astype(str)), redshift=redshift,
        metadata={'peakmjd': peak}, band_map={band: 'LSST-' + band for band in 'ugrizY'})
    table = result.to_table()
    assert result.n_sne == 1
    assert 'MU_LCFIT' in table.colnames
    assert 'Ds' not in table.colnames
    assert np.isfinite(table['MU_LCFIT'][0])
    np.testing.assert_array_equal(result.take([0]).to_table()['MU_LCFIT'], table['MU_LCFIT'])


def test_full_filter_names_control_independent_bands():
    from jax_snlc_fit.fitter import shift_map_with_poly
    from jax_snlc_core.photometry.filter_system import filter_values
    from jax_snlc_core.utils.error_handling import SNANAError

    names = ['g_PS1', 'r_PS1']
    fitter = LightCurveFitter.from_model(
        'SALT3.K21', batch_size=1, filtlist_fit=names,
        fitwin_trest_filter={'g_PS1': (-10., 20.), 'r_PS1': (-5., 30.)},
        fudge_fiterr_passbands=['g_PS1'], fudge_mag_coverr={'r_PS1': .02})
    lo, hi = fitter.config.fit.fitwin_trest_bounds(names)
    np.testing.assert_array_equal(lo, [-10., -5.])
    np.testing.assert_array_equal(hi, [20., 30.])
    assert fitter.fudge_fiterr_passband_mask == (True, False)
    np.testing.assert_allclose(fitter.fudge_mag_coverr_r, [0., 1. - 10. ** (-.4 * .02)])
    shifts = shift_map_with_poly({'g_PS1': .1}, (0., 1., 0.),
                                 {'g_PS1': 4500., 'r_PS1': 6200.})
    np.testing.assert_allclose(filter_values(shifts, names, 0.), [.55, .62])
    assert filter_values({'g': .1}, ['LSST-g', 'LSST-r'], 0.) == [.1, 0.]
    with pytest.raises(SNANAError, match='Ambiguous filter label'):
        filter_values({'1': .1}, names, 0.)


def test_full_filter_names_reach_photometry_corrections():
    from jax_snlc_fit.data_loader import batch_from_columns
    from jax_snlc_fit.batch import snrmax_band_mask

    names = {0: 'g_PS1', 1: 'r_PS1'}
    head = dict(SNID=['A'], PTROBS_MIN=[1], PTROBS_MAX=[4],
                REDSHIFT_FINAL=[.1], MWEBV=[0.], PEAKMJD=[60000.])
    phot = dict(MJD=[59990., 60000., 60010., 60020.],
                BAND=['g_PS1', 'r_PS1', 'g_PS1', 'r_PS1'],
                FLUXCAL=[10., 20., 30., 40.], FLUXCALERR=[1., 2., 3., 4.])
    batch = batch_from_columns(head, phot, is_sim=False, zp_fluxcal=27.5,
        sim_mw_colour_law=None, filter_map={name: i for i, name in names.items()},
        magobs_shift_zp={'g_PS1': .1, 'r_PS1': -.2})
    factors = 10. ** (-.4 * np.array([.1, -.2, .1, -.2]))
    np.testing.assert_allclose(batch.fluxcals[0], np.asarray(phot['FLUXCAL']) * factors)
    np.testing.assert_allclose(batch.fluxcalerrs[0], np.asarray(phot['FLUXCALERR']) * factors)
    np.testing.assert_array_equal(snrmax_band_mask(names, ['r_PS1']), [False, True])
    np.testing.assert_array_equal(snrmax_band_mask({0: 'LSST-g', 1: 'LSST-r'}, 'r'), [False, True])


def test_empty_observations_report_every_object():
    fitter = LightCurveFitter.from_model('SALT3.K21', batch_size=1)
    result = fitter.fit(mjd=[], flux=[], flux_err=[], band='g_PS1', redshift=.1)
    table = result.to_table()
    assert len(table) == 1
    assert table['status'][0] == 'rejected'
    assert table['mB'].mask[0]
    assert len(result.take([]).to_table()) == 0


@pytest.mark.parametrize('zero_point', [25., 27.5, 31.4])
def test_flux_units_share_the_calibration_zero_point(zero_point):
    from jax_snlc_fit.python import _flux_factor

    fitter = LightCurveFitter.from_model('SALT3.K21', batch_size=1)
    bands = np.array(['g_PS1', 'r_PS1'])
    jy = _flux_factor('Jy', None, bands, fitter.calibration,
                     fitter.model.wavelength_grid, zp_fluxcal=zero_point)
    microjy = _flux_factor('uJy', None, bands, fitter.calibration,
                          fitter.model.wavelength_grid, zp_fluxcal=zero_point)
    np.testing.assert_allclose(jy, microjy * 1e6, rtol=1e-12)
    # A 20th-magnitude AB source, including the catalogue's calibration offsets.
    flux_jy = 10. ** (-.4 * (20. + 48.6)) / 1e-23
    offsets = np.array([fitter.calibration.filter_primary_mag[name] for name in bands])
    expected = 10. ** (.4 * (zero_point - 20.)) * 10. ** (-.4 * offsets)
    np.testing.assert_allclose(jy * flux_jy, expected, rtol=1e-12)
    np.testing.assert_allclose(_flux_factor('fluxcal', 25., bands,
        fitter.calibration, fitter.model.wavelength_grid,
        zp_fluxcal=zero_point), 10. ** (.4 * (zero_point - 25.)))


@pytest.mark.parametrize('tables', [False, True])
def test_default_zero_point_matches_explicit_simulation_zero_point(tables):
    fitter = LightCurveFitter('SALT3.K21', batch_size=1,
                             n_fit_iterations=1, lfixpar_all=True,
                             initial={'x0': .001})
    times = np.linspace(59990., 60030., 8)
    flux = np.full(8, 100.)
    errors = np.full(8, 5.)
    if tables:
        data = dict(photometry=dict(snid=['A'] * 8, mjd=times,
                    flux=flux, flux_err=errors, band=['g_PS1'] * 8,
                    redshift=[.1] * 8, peakmjd=[60000.] * 8))
    else:
        data = dict(mjd=times, flux=flux, flux_err=errors, band='g_PS1',
                    redshift=.1, metadata={'peakmjd': 60000.})
    implicit = fitter.fit(**data, progress=False)
    explicit = fitter.fit(**data, zp=31.4, progress=False)
    assert implicit.n_sne == explicit.n_sne == 1
    assert implicit.context[0][0]['batch'].zp_fluxcal == 31.4
    np.testing.assert_allclose(implicit.context[0][0]['batch'].fluxcals[0], flux)
    np.testing.assert_allclose(implicit.chi_squared, explicit.chi_squared)
    first = fitter.predict(implicit, times, 'g_PS1')
    second = fitter.predict(explicit, times, 'g_PS1')
    np.testing.assert_allclose(first['flux'], second['flux'])
    np.testing.assert_allclose(first['model_flux_err'], second['model_flux_err'])
    scale = 10. ** (.4 * (25. - 31.4))
    if tables:
        rescaled = dict(photometry={**data['photometry'],
                                   'flux': flux * scale, 'flux_err': errors * scale})
    else:
        rescaled = {**data, 'flux': flux * scale, 'flux_err': errors * scale}
    other = fitter.fit(**rescaled, zp=25., progress=False)
    assert other.context[0][0]['batch'].zp_fluxcal == 25.
    np.testing.assert_allclose(other.context[0][0]['batch'].fluxcals[0], flux * scale)
    np.testing.assert_allclose(other.chi_squared, implicit.chi_squared, rtol=1e-10)
    prediction = fitter.predict(other, times, 'g_PS1')
    np.testing.assert_allclose(prediction['flux'], first['flux'] * scale, rtol=1e-10)


def test_repeated_photoz_fit_grows_search_and_matches_fresh_fitter(snana_fixture, data_root):
    fixture = snana_fixture('salt3_fit_photoz')
    with fits.open(fixture.head_fits) as hdus:
        head = hdus[1].data[0]
        start, stop = int(head['PTROBS_MIN']) - 1, int(head['PTROBS_MAX'])
        redshift, peak = float(head['REDSHIFT_HELIO']), float(head['PEAKMJD'])
    with fits.open(fixture.phot_fits) as hdus:
        phot = hdus[1].data[start:stop].copy()
    settings = dict(calibration=data_root / 'kcor/JAX_TEST_LSST.fits',
        batch_size=1, opt_photoz=2, photoz_bound=(.01, .6), n_zgrid=80,
        coarse_grid={'x1': {'nbin': 1, 'range': [0., 0.]},
                     'c': {'nbin': 1, 'range': [0., 0.]}},
        n_fit_iterations=1, maxiter_round1=1, opt_mwebv=0,
        use_model_magerr=False)
    data = dict(mjd=phot['MJD'].astype(float), flux=phot['FLUXCAL'].astype(float),
        flux_err=phot['FLUXCALERR'].astype(float),
        band=np.char.strip(phot['BAND'].astype(str)), redshift=redshift,
        band_map={band: 'LSST-' + band for band in 'ugrizY'})
    fitter = LightCurveFitter.from_model('SALT3.K21', **settings)
    fitter.fit(**data, metadata={'peakmjd': peak, 'redshift_err': .0001})
    narrow_size = fitter.n_z
    wide = fitter.fit(**data, metadata={'peakmjd': peak, 'redshift_err': .1})
    assert fitter.n_z > narrow_size
    fresh = LightCurveFitter.from_model('SALT3.K21', **settings)
    expected = fresh.fit(**data, metadata={'peakmjd': peak, 'redshift_err': .1})
    assert fresh.n_z == fitter.n_z
    for name in wide.params:
        np.testing.assert_allclose(wide.params[name], expected.params[name], rtol=1e-10)
    np.testing.assert_allclose(wide.chi_squared, expected.chi_squared, rtol=1e-10)
    fitter.reset_shape()
    assert fitter.n_z is None and fitter.epoch_lengths == []


def test_object_tables_keep_order_and_include_unobserved_objects():
    fitter = LightCurveFitter.from_model('SALT3.K21', batch_size=2,
        n_fit_iterations=1, lfixpar_all=True, initial={'x0': .001})
    times = np.linspace(59990., 60030., 8)
    photometry = dict(snid=['A'] * 8 + ['B'] * 8,
        mjd=np.tile(times, 2), band=['g_PS1'] * 8 + ['r_PS1'] * 8,
        flux=np.r_[np.full(8, 100.), np.full(8, 10.)],
        flux_err=np.r_[np.full(8, 5.), np.full(8, .5)])
    objects = dict(snid=['B', 'empty', 'A'], redshift=[.1, .1, .2],
                   peakmjd=[60000.] * 3, zp=[25., 27.5, 27.5])
    result = fitter.fit(photometry=photometry, objects=objects)
    table = result.to_table()
    assert list(table['snid']) == ['B', 'empty', 'A']
    assert result.n_sne == 2
    assert table['status'][1] == 'rejected' and table['mB'].mask[1]
    assert table['reason_description'][1] == 'No usable observations'
    assert list(result.snids) == ['B', 'A']
    for context, row in result.context:
        assert context['batch'].zp_fluxcal == 25.
        np.testing.assert_allclose(context['batch'].fluxcals[row], 10.)
    empty = fitter.fit(mjd=[], flux=[], flux_err=[], band='g_PS1', redshift=.1, snid='none')
    for parts, snids, rejected in (
            ([result, empty], ['B', 'empty', 'A', 'none'], [1, 3]),
            ([empty, result], ['none', 'B', 'empty', 'A'], [0, 2])):
        combined = type(result).concatenate(parts)
        assert list(combined.to_table()['snid']) == snids
        assert list(combined.to_table()['reason_description'][rejected]) == [
            'No usable observations', 'No usable observations']
        for name, flags in result.error_flags.items():
            np.testing.assert_array_equal(combined.error_flags[name], flags)


def test_redback_array_fit_evaluates_user_prior():
    pytest.importorskip("redback_jax")
    fitter = LightCurveFitter.from_model('REDBACK.arnett_bolometric',
        batch_size=1, n_fit_iterations=1, lfixpar_all=True,
        redback_tgrid_nbin=128, initial={'mej': 1.5},
        priors={'mej': {'distribution': 'uniform', 'lower': 1., 'upper': 2.}})
    result = fitter.fit(mjd=np.linspace(60005., 60030., 8), flux=np.full(8, 100.),
        flux_err=np.full(8, 10.), band='g_PS1', redshift=.05,
        metadata={'peakmjd': 60000.})
    assert result.n_sne == 1
    np.testing.assert_allclose(result.params['mej'], 1.5)
    assert np.isfinite(result.chi_squared).all()
    assert np.isfinite(result.prior_chi2).all()


def test_configuration_retains_explicit_calibration(data_root):
    calibration = resolve_fits_path(data_root / 'kcor/JAX_TEST_LSST.fits')
    fitter = LightCurveFitter.from_model('BAYESN.W22', calibration=calibration, batch_size=1)
    config = fitter.configuration()
    assert config.data.kcor_file == str(calibration)
    recreated = LightCurveFitter.from_model(config.model.fitmodel_name, config=config)
    assert recreated.calibration.kcor_file_path == fitter.calibration.kcor_file_path
    assert recreated.configuration() == config


@pytest.mark.parametrize('tables', [False, True])
@pytest.mark.parametrize('coordinates', [False, True])
def test_python_redshift_is_heliocentric(tables, coordinates):
    from jax_snlc_fit.python import _batch_from_tables, _input_tables
    from jax_snlc_fit.run import cut_windows, read_shared_files

    fitter = LightCurveFitter.from_model('SALT3.K21', batch_size=1, filtlist_fit=['g_PS1'])
    cfg = fitter.configuration()
    metadata = {'ra': 0., 'dec': 0.} if coordinates else {}
    kwargs = dict(mjd=[60000., 60010.], flux=[100., 80.], flux_err=[5., 5.],
                  band='g_PS1', redshift=.05, mwebv=0., snid='SN1', flux_unit='fluxcal',
                  zp=None, redshift_cmb=.06, metadata=metadata,
                  photometry=None, objects=None, column_map=None)
    if tables:
        kwargs.update(mjd=None, flux=None, flux_err=None, band=None, redshift=None,
                      redshift_cmb=None, metadata=None,
                      photometry=dict(snid=['SN1']*2, mjd=[60000., 60010.],
                                      flux=[100., 80.], flux_err=[5., 5.], band=['g_PS1']*2),
                      objects={k: [v] for k,v in dict(snid='SN1', redshift=.05,
                                                     redshift_cmb=.06, **metadata).items()})
    obs, obj = _input_tables(**kwargs)
    _, epoch_cuts, _ = cut_windows(cfg)
    batch = _batch_from_tables(obs, obj, fitter, cfg, epoch_cuts,
                               read_shared_files(cfg), zp_fluxcal=31.4)
    np.testing.assert_allclose(batch.redshifts, [.05], rtol=1.e-6)


def test_python_fit_applies_calibration_shifts():
    fitter = LightCurveFitter.from_model('SALT3.K21', batch_size=1, n_fit_iterations=1,
        maxiter_round1=2, magobs_shift_zp={'g_PS1': 1.})
    times = np.linspace(59990., 60035., 12)
    flux = 10. * np.exp(-.5 * ((times - 60005.) / 15.)**2)
    result = fitter.fit(mjd=times, flux=flux, flux_err=np.full(12, .5),
                        band='g_PS1', redshift=.1, zp=25.)
    assert result.n_sne == 1
    context = result.context[0][0]
    assert context['batch'].zp_fluxcal == 25.
    np.testing.assert_allclose(context['batch'].fluxcals[0], flux * 10.**(-.4))
    np.testing.assert_allclose(context['batch'].fluxcalerrs[0], .5 * 10.**(-.4))


@pytest.mark.parametrize('kind', ['pandas', 'astropy', 'mapping'])
def test_single_table_matches_two_tables(kind):
    import pandas as pd
    from astropy.table import Table

    fitter = LightCurveFitter('SALT3.K21', batch_size=2,
                             n_fit_iterations=1, lfixpar_all=True,
                             initial={'x0': .001})
    times = np.linspace(59990., 60030., 8)
    photometry = pd.DataFrame(dict(
        snid=['B', 'A'] * 8, mjd=np.repeat(times, 2),
        band=['g_PS1', 'r_PS1'] * 8, flux=[100., 10.] * 8,
        flux_err=[5., .5] * 8))
    objects = pd.DataFrame(dict(snid=['B', 'A'], redshift=[.1, .2],
                                mwebv=[.01, .02], peakmjd=[60000.] * 2,
                                zp=[27.5, 25.]))
    combined = photometry.merge(objects, on='snid', how='left', sort=False)
    original = combined.copy(deep=True)
    if kind == 'astropy':
        supplied = Table.from_pandas(combined)
    elif kind == 'mapping':
        supplied = combined.to_dict('list')
    else:
        supplied = combined
    one = fitter.fit(supplied, progress=False)
    two = fitter.fit(photometry, objects=objects, progress=False)
    assert list(one.to_table()['snid']) == ['B', 'A']
    for name in one.params:
        np.testing.assert_array_equal(one.params[name], two.params[name])
    np.testing.assert_array_equal(one.chi_squared, two.chi_squared)
    np.testing.assert_array_equal(one.param_covariances, two.param_covariances)
    pd.testing.assert_frame_equal(combined, original)
    for result in (one, two):
        assert result.n_sne == 2
        assert np.isfinite(fitter.predict(result, times, 'g_PS1', snid='B')['flux']).all()


@pytest.mark.parametrize('column,values', [
    ('redshift', [.1, .2]), ('mwebv', [.01, .02]),
    ('peakmjd', [60000., 60001.]), ('photoz_err', [.1, .2]),
    ('flux_unit', ['Jy', 'uJy']), ('zp', [25., 27.5]),
    ('redshift', [.1, np.nan]),
])
def test_single_table_rejects_inconsistent_object_properties(column, values):
    from jax_snlc_core.utils.error_handling import SNANAError
    fitter = LightCurveFitter('SALT3.K21', batch_size=1)
    data = dict(snid=['SN1', 'SN1'], mjd=[60000., 60001.],
                band=['g_PS1'] * 2, flux=[1., 2.], flux_err=[.1] * 2,
                redshift=[.1] * 2)
    data[column] = values
    with pytest.raises(SNANAError, match=f"Inconsistent '{column}'.*SN1"):
        fitter.fit(data)


def test_single_table_column_mapping_and_required_metadata():
    import pandas as pd
    from jax_snlc_fit.python import _input_tables
    from jax_snlc_core.utils.error_handling import SNANAError
    data = pd.DataFrame(dict(id=['B', 'A', 'B'], time=[1., 2., 3.],
                             measured=[3., 4., 5.], error=[.1] * 3,
                             filter=['g_PS1'] * 3, z=[.1, .2, .1]))
    obs, objects = _input_tables(
        photometry=data, objects=None, mjd=None, flux=None, flux_err=None,
        band=None, redshift=None, redshift_cmb=None, metadata=None,
        mwebv=0., snid='0', flux_unit='uJy', zp=None,
        column_map={'snid': 'id', 'mjd': 'time', 'flux': 'measured',
                    'flux_err': 'error', 'band': 'filter', 'redshift': 'z'})
    assert list(objects.snid) == ['B', 'A']
    assert list(objects.redshift) == [.1, .2]
    assert 'redshift' not in obs
    assert 'z' in data and 'snid' not in data
    fitter = LightCurveFitter('SALT3.K21', batch_size=1)
    with pytest.raises(SNANAError, match='redshift is required'):
        fitter.fit(obs)
    with pytest.raises(SNANAError, match='Missing input columns: snid'):
        fitter.fit(obs.drop(columns='snid'))


def test_python_selection_is_explicit_and_can_exclude_every_object():
    from jax_snlc_core.utils.error_handling import SNANAError
    data = dict(mjd=np.arange(60000., 60012.), flux=np.full(12, 10.),
                flux_err=np.ones(12), band='g_PS1', redshift=.1)
    fitter = LightCurveFitter('SALT3.K21', batch_size=1,
                             cutwin={'SNRMAX': (1.e6, 1.e9)})
    with pytest.raises(SNANAError, match='apply_selection=True'):
        fitter.fit(**data)
    result = fitter.fit(**data, apply_selection=True, progress=False)
    assert result.n_sne == 0
    assert list(result.to_table()['status']) == ['rejected']
    assert fitter.epoch_lengths == []


def test_predictions_do_not_depend_on_sample_acceptance(monkeypatch):
    times = np.linspace(59990., 60030., 12)
    data = dict(mjd=times, flux=np.full(12, 10.), flux_err=np.ones(12),
                band='g_PS1', redshift=.1)
    fitter = LightCurveFitter('SALT3.K21', batch_size=1, lfixpar_all=True,
                             initial={'x0': .001}, fitwin={'PROB': (.999999, 1.)})
    result = fitter.fit(**data, apply_selection=True, progress=False)
    assert result.n_sne == 1
    assert result.to_table()['status'][0] == 'rejected'
    narrow_window = fitter.static_data['band_window']
    # A prediction in a wider band must not alter the fitter's prepared window.
    predicted = fitter.predict(result, times, 'c_ATLAS')
    assert np.isfinite(predicted['flux']).all()
    assert fitter.static_data['band_window'] == narrow_window
    with monkeypatch.context() as patch:
        patch.setattr('jax_snlc_fit.fitter.fit_band_window', lambda model, photometry: None)
        full_grid = fitter.predict(result, times, 'c_ATLAS')
    np.testing.assert_allclose(predicted['flux'], full_grid['flux'], rtol=1e-12)
    np.testing.assert_allclose(predicted['model_flux_err'], full_grid['model_flux_err'], rtol=1e-12)


def test_python_constructor_checks_photoz_restframe_cuts():
    from jax_snlc_core.utils.error_handling import SNANAError
    with pytest.raises(SNANAError, match='ABORT_ON_CUTWIN_TREST'):
        LightCurveFitter('SALT3.K21', batch_size=1, opt_photoz=1,
                        inistp={'PHOTOZ': 1.}, cutwin={'TRESTMIN': (-10., 0.)})


def test_fit_without_observations_in_model_support_is_rejected():
    from jax_snlc_fit.batch import ERRFLAG_FITPREP_NOEPOCHS
    fitter = LightCurveFitter('SALT3.K21', batch_size=1,
                             n_fit_iterations=1, maxiter_round1=2)
    result = fitter.fit(mjd=np.linspace(59990., 60030., 12),
                        flux=np.full(12, 10.), flux_err=np.ones(12),
                        band='g_PS1', redshift=4., progress=False)
    assert not np.asarray(result.observation_mask).any()
    table = result.to_table()
    assert table['status'][0] == 'rejected'
    assert table['reason'][0] == ERRFLAG_FITPREP_NOEPOCHS


def test_flux_conversion_prepares_bands_once_for_mixed_objects(monkeypatch):
    import pandas as pd
    from jax_snlc_fit import python as interface
    fitter = LightCurveFitter('SALT3.K21', batch_size=1)
    bands = np.array(['g_PS1', 'r_PS1', 'g_PS1', 'r_PS1', 'g_PS1', 'g_PS1'])
    objects = pd.DataFrame({'snid': ['A', 'B', 'C', 'D'],
                            'flux_unit': ['Jy', 'uJy', 'fluxcal', 'Jy'],
                            'zp': [np.nan, np.nan, 25., np.nan]})
    obs = pd.DataFrame({'snid': ['B', 'A', 'C', 'B', 'D', 'A'],
                        'band': bands, 'flux': np.ones(6), 'flux_err': np.full(6, .1)})
    expected = np.empty(6)
    for row in objects.to_dict('records'):
        mask = obs.snid == row['snid']
        expected[mask] = interface._flux_factor(row['flux_unit'],
            None if np.isnan(row['zp']) else row['zp'], bands[mask],
            fitter.calibration, fitter.model.wavelength_grid, zp_fluxcal=25.)
    calls = []
    load = interface.load_photometry_resources
    def recorded(*args):
        calls.append(args[2])
        return load(*args)
    monkeypatch.setattr(interface, 'load_photometry_resources', recorded)
    units, zero_point = interface._normalise_flux(obs, objects, fitter)
    assert zero_point == 25.
    assert calls == [['g_PS1', 'r_PS1']]
    np.testing.assert_allclose(obs.flux, expected)
    np.testing.assert_allclose(obs.flux_err, expected * .1)
    assert units == {'A': ('Jy', None), 'B': ('uJy', None), 'C': ('fluxcal', 25.), 'D': ('Jy', None)}


def test_epoch_rejection_uses_prepared_band_window():
    fitter = LightCurveFitter('SALT3.K21', batch_size=1, n_fit_iterations=2,
                             lfixpar_all=True, initial={'x0': .001}, signoise_reject=5.)
    flux = np.full(12, 10.)
    flux[5] = 0.
    result = fitter.fit(mjd=np.linspace(59995., 60020., 12), flux=flux,
                        flux_err=np.ones(12), band='g_PS1', redshift=.1, progress=False)
    mask = np.asarray(result.observation_mask)[0]
    assert not mask[5]
    assert mask.sum() == 11
