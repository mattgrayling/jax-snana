"""Direct calibration inputs compared with recorded SNANA FITS output."""

import shutil

import jax.numpy as jnp
import numpy as np
import pytest

from jax_snlc_core.io.kcor_reader import KcorReader
from jax_snlc_core.photometry.resources import load_photometry_resources, resolve_filtlist
from jax_snlc_core.utils.error_handling import SNANAError


@pytest.fixture
def calibration_dir(tmp_path, data_root):
    destination = tmp_path / 'calibration'
    shutil.copytree(data_root / 'kcor' / 'direct_input', destination)
    return destination


@pytest.mark.parametrize('stem', [
    'calibration', 'calibration_sed', 'calibration_cut',
    'calibration_duplicate', 'calibration_spectrograph', 'calibration_rebin',
    'calibration_spectrograph_duplicate',
    'calibration_oob', 'calibration_oob_cut', 'calibration_oob_sed',
])
def test_input_matches_snana_generated_fits(calibration_dir, monkeypatch, stem):
    # Every resource is explicit and relative to the input; no SNANA installation
    # or current-working-directory dependency is needed to read the calibration.
    monkeypatch.delenv('SNDATA_ROOT', raising=False)
    monkeypatch.chdir(calibration_dir.parent)
    readers = [KcorReader(calibration_dir / filename)
               for filename in (f'{stem}.input', f'{stem}.fits')]
    direct, recorded = readers
    assert direct.get_filter_names() == recorded.get_filter_names()
    assert direct.filter_names == recorded.filter_names
    assert resolve_filtlist('griz', direct) == ['Test-g', 'Test-r', 'Test-i', 'Test-z']
    assert direct.syn_filters == recorded.syn_filters
    assert direct.filter_primary == recorded.filter_primary
    assert direct.filter_survey == recorded.filter_survey
    for name in direct.filter_names:
        for field in ('wavelength', 'transmission'):
            np.testing.assert_allclose(direct.filter_data[name][field],
                                       recorded.filter_data[name][field], rtol=1e-7)
        np.testing.assert_allclose(direct.filter_primary_mag[name],
                                   recorded.filter_primary_mag[name], atol=5e-7, rtol=0)
        np.testing.assert_allclose(direct.filter_zpoff_snpot[name],
                                   recorded.filter_zpoff_snpot[name], atol=1e-9, rtol=0)
    if direct.spectrograph is not None:
        assert recorded.spectrograph is not None
        for field in ('lam_min', 'lam_max', 'lam_sigma', 'texpose_list',
                      'zp', 'sqsigsky', 'islam_extend'):
            np.testing.assert_allclose(getattr(direct.spectrograph, field),
                                       getattr(recorded.spectrograph, field), rtol=5e-7)
    for name in ('AB', 'BD17'):
        active = np.logical_or.reduce([
            np.asarray(recorded.filter_data[band]['transmission']) > 0
            for band in recorded.filter_names])
        for a, b in zip(direct.primary_seds[name], recorded.primary_seds[name]):
            # The reference SED outside every filter's support is unused.
            np.testing.assert_allclose(np.asarray(a)[active], np.asarray(b)[active], rtol=1e-7)
    resources = [load_photometry_resources(reader, jnp.linspace(2500., 9500., 300),
                                         reader.filter_names) for reader in readers]
    for field in ('filter_ref_fluxes', 'filter_ref_fluxes_snana',
                  'common_trans_grid', 'transmissions_stacked'):
        np.testing.assert_allclose(getattr(resources[0], field),
                                   getattr(resources[1], field), rtol=5e-7, atol=1e-12)


def test_zero_point_override_and_disabled_offsets(calibration_dir):
    path = calibration_dir / 'calibration.input'
    (calibration_dir / 'offsets.dat').write_text('g .12 r -.13 i .14 z -.15\n')
    original = path.read_text()
    path.write_text(original.replace('FILTPATH: .', 'ZPOFF_FILE: offsets.dat\nFILTPATH: .'))
    reader = KcorReader(path)
    assert reader.filter_zpoff_snpot == dict(zip(reader.filter_names, (.12, -.13, .14, -.15)))
    path.write_text('READ_ZPOFF: 0\n' + original)
    assert set(KcorReader(path).filter_zpoff_snpot.values()) == {0.}


def test_data_tree_and_environment_paths(calibration_dir, tmp_path, monkeypatch):
    root = tmp_path / 'sndata'
    (root / 'filters').mkdir(parents=True)
    shutil.copytree(calibration_dir, root / 'filters' / 'survey')
    (root / 'standards').mkdir()
    shutil.copy(calibration_dir / 'ab.dat', root / 'standards' / 'flatnu.dat')
    monkeypatch.setenv('SNDATA_ROOT', str(root))
    monkeypatch.setenv('CUSTOM_FILTERS', str(root / 'filters' / 'survey'))
    path = tmp_path / 'survey.input'
    template = 'MAGSYSTEM: AB FILTSYSTEM: COUNT SURVEY: DES\nFILTPATH: {}\nFILTER: Test-g g.dat 0.0\n'
    for directory in ('survey', '$CUSTOM_FILTERS'):
        path.write_text(template.format(directory))
        reader = KcorReader(path)
        assert reader.filter_zpoff_snpot['Test-g'] == .02
        assert reader.get_filter_primary('Test-g') == 'AB'


@pytest.mark.parametrize('setting, message', [
    ('FILTER_OOB: u 3000 9000 .001', 'band.*missing'),
    ('SYN_FILTER: Test-u 3000 4000 0', 'requires SPECTROGRAPH'),
    ('FILTSYSTEM: UNKNOWN', 'unknown FILTSYSTEM'),
    ('LAMBDA_RANGE: 5000 4000', 'LAMBDA_RANGE'),
    ('LEAKAGE_CUT: -1', 'LEAKAGE_CUT'),
    ('ZPOFF_FILE: missing.dat\nFILTPATH: .', 'resource not found'),
])
def test_unsupported_or_invalid_calibration_fails(calibration_dir, setting, message):
    path = calibration_dir / 'calibration.input'
    path.write_text(path.read_text() + '\n' + setting + '\n')
    with pytest.raises(SNANAError, match=message):
        KcorReader(path)


@pytest.mark.parametrize('magnitude', ['1.2.3', 'nan', '0.1+TYPO', '1++2'])
def test_invalid_primary_magnitude_fails(calibration_dir, magnitude):
    path = calibration_dir / 'calibration.input'
    path.write_text(path.read_text().replace('0.0+.003-.001', magnitude))
    with pytest.raises(SNANAError, match='Invalid calibration magnitude'):
        KcorReader(path)


def test_missing_standard_is_not_replaced_by_ab(calibration_dir):
    path = calibration_dir / 'calibration.input'
    path.write_text(path.read_text().replace('BD17_SED: bd17.dat\n', ''))
    with pytest.raises(SNANAError, match='missing or invalid standard'):
        KcorReader(path)


def test_input_leakage_cut_is_applied_before_grid_interpolation(calibration_dir):
    path = calibration_dir / 'calibration.input'
    path.write_text(path.read_text() + '\nLEAKAGE_CUT: .1\n')
    reader = KcorReader(path)
    wave, transmission = reader.get_filter_transmission('Test-g')
    positive = np.asarray(wave)[np.asarray(transmission) > 0]
    np.testing.assert_array_equal(positive, np.arange(3830., 5171., 10.))


def test_input_leakage_cut_normalises_peak(calibration_dir):
    curve = calibration_dir / 'g.dat'
    wave, transmission = np.loadtxt(curve, unpack=True)
    np.savetxt(curve, np.column_stack((wave, .5 * transmission)))
    path = calibration_dir / 'calibration.input'
    path.write_text(path.read_text() + '\nLEAKAGE_CUT: .1\n')
    assert np.max(KcorReader(path).filter_data['Test-g']['transmission']) == 1.


def test_gzipped_input_and_reference_spectrum(calibration_dir):
    import gzip
    path = calibration_dir / 'calibration.input'
    with gzip.open(str(path) + '.gz', 'wt') as stream:
        stream.write(path.read_text())
    path.unlink()
    standard = calibration_dir / 'ab.dat'
    with gzip.open(str(standard) + '.gz', 'wt') as stream:
        stream.write(standard.read_text())
    standard.unlink()
    reader = KcorReader(path)
    assert reader.get_filter_names() == ['Test-g', 'Test-r', 'Test-i', 'Test-z']


@pytest.mark.parametrize('model,column', [('SALT3.K21', 'mB'), ('BAYESN.W22', 'MU_LCFIT')])
@pytest.mark.parametrize('stem', ['calibration', 'calibration_oob'])
def test_python_fits_use_direct_calibration(calibration_dir, model, column, monkeypatch, stem):
    from jax_snlc_fit import LightCurveFitter

    monkeypatch.delenv('SNDATA_ROOT', raising=False)
    times = np.linspace(59990., 60035., 12)
    flux = 100. * np.exp(-.5 * ((times - 60005.) / 15.)**2)
    values = []
    for suffix in ('input', 'fits'):
        fitter = LightCurveFitter(model, calibration=calibration_dir / f'{stem}.{suffix}',
                                 batch_size=1, n_fit_iterations=1, maxiter_round1=2)
        result = fitter.fit(mjd=times, flux=flux, flux_err=np.full(12, 5.),
                            band='Test-g', redshift=.1)
        values.append(float(result.to_table()[column][0]))
    assert np.all(np.isfinite(values))
    np.testing.assert_allclose(values[0], values[1], atol=2e-5, rtol=0)


def test_text_calibration_has_no_embedded_spectrograph(calibration_dir):
    reader = KcorReader(calibration_dir / 'calibration.input')
    assert reader.spectrograph is None


def test_shifted_duplicates_are_selected_by_full_name(calibration_dir):
    reader = KcorReader(calibration_dir / 'calibration_duplicate.input')
    assert reader.get_filter_names() == ['Test-g', 'Test-r', 'Test-i', 'Test-z']
    assert reader.get_filter_names(include_duplicates=True) == reader.filter_names
    assert resolve_filtlist(['*Test-g'], reader) == ['*Test-g']
    resource = load_photometry_resources(reader, jnp.linspace(3000., 9000., 300), ['*Test-g'])
    assert resource.filter_names == ['*Test-g']


@pytest.mark.parametrize('ratio', [0., .001])
@pytest.mark.parametrize('system,cut', [('COUNT', 0.), ('ENERGY', 0.), ('COUNT', .01)])
def test_out_of_band_matches_an_explicit_transmission_curve(calibration_dir, ratio, system, cut):
    path = calibration_dir / 'calibration.input'
    original = path.read_text()
    original = original.replace('FILTSYSTEM: COUNT', f'FILTSYSTEM: {system}', 1)
    original += f'\nLEAKAGE_CUT: {cut}\n'
    path.write_text(original + f'\nFILTER_OOB: g 3000 9000 {ratio}\n')
    generated = KcorReader(path)
    wave, transmission = np.loadtxt(calibration_dir / 'g.dat', unpack=True)
    if ratio:
        extended = np.arange(3000., 9001., 25.)
        transmission = np.interp(extended, wave, transmission, left=0., right=0.) + ratio * transmission.max()
        wave = extended
    np.savetxt(calibration_dir / 'g_with_leakage.dat', np.column_stack((wave, transmission)))
    path.write_text(original.replace('g.dat', 'g_with_leakage.dat'))
    explicit = KcorReader(path)
    np.testing.assert_array_equal(generated.filter_data['Test-g']['transmission'],
                                  explicit.filter_data['Test-g']['transmission'])
    np.testing.assert_array_equal(generated.primary_seds['AB'][1], explicit.primary_seds['AB'][1])


def test_out_of_band_applies_to_nominal_filters_before_duplication(calibration_dir):
    path = calibration_dir / 'calibration_duplicate.input'
    nominal = KcorReader(path)
    path.write_text(path.read_text() + '\nFILTER_OOB: g 3000 9000 .001\n')
    leakage = KcorReader(path)
    np.testing.assert_array_equal(nominal.filter_data['*Test-g']['transmission'],
                                  leakage.filter_data['*Test-g']['transmission'])
    assert not np.array_equal(nominal.filter_data['Test-g']['transmission'],
                              leakage.filter_data['Test-g']['transmission'])
