"""The script entry point runs the existing survey simulator unchanged."""

from contextlib import chdir
from pathlib import Path
import re

import jax
import numpy as np
import pytest

from jax_snlc_core.logging import SNANAError
from jax_snlc_sim import parse_input, prepare, simulate
from jax_snlc_sim.run import run_simulation


@pytest.mark.parametrize("fixture", ["salt3_fit_select", "bayesn_w22"])
def test_simulate_matches_prepared_run(snana_fixture, fixture, tmp_path):
    source = snana_fixture(fixture).input_file
    numerical = dict(n_sne=2, batch_size=2, random_seed=42)
    compiled = prepare(source, **numerical)
    expected = run_simulation(compiled)

    with chdir(tmp_path):
        actual = simulate(source, ngentot_lc=2, batch_size=2, random_seed=42)
    expected_arrays, expected_tree = jax.tree.flatten(expected)
    actual_arrays, actual_tree = jax.tree.flatten(actual)
    assert actual_tree == expected_tree
    for a, b in zip(actual_arrays, expected_arrays):
        np.testing.assert_array_equal(a, b)
    assert not list(tmp_path.iterdir()), "simulate must not write a data version"


def test_simulate_setup_only_reports_no_generation(snana_fixture):
    config = parse_input(snana_fixture("salt3_fit_select").input_file,
                         {"init_only": True, "ngentot_lc": 2, "batch_size": 2})
    with pytest.raises(SNANAError, match="setup-only mode"):
        simulate(config)


def test_simulate_invalid_keyword_settings_fail(snana_fixture):
    source = snana_fixture("salt3_fit_select").input_file
    with pytest.raises(SNANAError, match="Unknown"):
        simulate(source, batch_szie=2)
    with pytest.raises(SNANAError, match="RANSEED"):
        simulate(genmodel="SALT3.K21", genfilters="griz", simlib_file="survey.SIMLIB",
                 kcor_file="survey.fits", ngentot_lc=2, redshift_range=[.1, .2],
                 peakmjd_range=[60000, 60010], trest_range=[-10, 40])


@pytest.mark.parametrize('stem', ['calibration', 'calibration_spectrograph', 'calibration_oob'])
def test_direct_calibration_matches_fits_in_survey_simulation(data_root, tmp_path, stem):
    source = Path(__file__).parents[2] / 'examples/quickstart/sim_bayesn.input'
    calibration = data_root / 'kcor/direct_input'
    settings = {}
    if stem == 'calibration_spectrograph':
        simlib = tmp_path / 'spectrograph.SIMLIB'
        text = (data_root / 'simlib/JAX_TEST_DES.SIMLIB').read_text()
        text = re.sub(r'NOBS:\s*(\d+)', lambda m: f'NOBS: {int(m[1]) + 1}', text)
        text = text.replace('END_LIBID:', 'SPECTROGRAPH: 56250.0 1800.0\nEND_LIBID:')
        simlib.write_text(text)
        settings = {'simlib_file': str(simlib), 'use_simlib_spectrograph': True,
                    'peakmjd_range': [56240., 56260.], 'redshift_range': [.01, .02]}
    samples = [simulate(source, kcor_file=str(calibration / f'{stem}.{suffix}'),
                        ngentot_lc=2, batch_size=2, apply_cutwin_opt=0,
                        double_precision=True, **settings)
               for suffix in ('input', 'fits')]
    if stem == 'calibration_spectrograph':
        result = samples[0][0]
        bands = np.asarray(result['observation_schedule']['filter_indices'])
        for band in (4, 5):
            observed = bands == band
            assert observed.any(), 'synthetic bands must contribute observations'
            assert np.all(np.isfinite(np.asarray(result['fluxcals'])[observed]))
    # Noisy flux can cross zero. Scale calibration differences by the true
    # flux or measurement uncertainty, whichever is larger.
    direct_results = dict(samples[0][0])
    recorded_results = dict(samples[1][0])
    direct_flux = direct_results.pop('fluxcals_noisy')
    recorded_flux = recorded_results.pop('fluxcals_noisy')
    flux_scale = np.maximum(np.abs(recorded_results['fluxcals']),
                            recorded_results['fluxcalerrs'])
    np.testing.assert_allclose(
        (direct_flux - recorded_flux) / flux_scale,
        np.where(np.isnan(recorded_flux), np.nan, 0.0), atol=2e-6, rtol=0)
    direct, direct_tree = jax.tree.flatten((direct_results, samples[0][1]))
    recorded, recorded_tree = jax.tree.flatten((recorded_results, samples[1][1]))
    assert direct_tree == recorded_tree
    for a, b in zip(direct, recorded):
        np.testing.assert_allclose(a, b, rtol=2e-6, atol=1e-5)

    direct_spectra, recorded_spectra = samples[0][2], samples[1][2]
    assert direct_spectra.keys() == recorded_spectra.keys()
    if stem == 'calibration_spectrograph':
        assert any(direct_spectra.values()), 'spectra must be generated'
    for snid in direct_spectra:
        assert len(direct_spectra[snid]) == len(recorded_spectra[snid])
        for a, b in zip(direct_spectra[snid], recorded_spectra[snid]):
            assert (a.snid, a.instrument, a.n_lam_total, a.is_host) == (
                b.snid, b.instrument, b.n_lam_total, b.is_host)
            (direct, direct_tree), (recorded, recorded_tree) = [
                jax.tree.flatten((
                    s.mjd, s.texpose, s.lam_min, s.lam_max, s.flam, s.flamerr,
                    s.sim_flam, s.snr_compute, s.lammin_snr, s.lammax_snr,
                    s.trest, s.tobs, s.sim_genmag, s.synmag, s.scale_host_contam,
                    s.warp, s.sim_flam_full))
                for s in (a, b)]
            assert direct_tree == recorded_tree
            for actual, expected in zip(direct, recorded):
                np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-5)
