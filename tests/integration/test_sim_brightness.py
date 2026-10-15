"""Brightness corrections at observed, peak and template epochs."""

from dataclasses import replace

import jax
import numpy as np
import pytest

from jax_snlc_core.photometry.filter_system import filter_band
from jax_snlc_sim import run


@pytest.fixture(scope='module')
def prepared():
    config = dict(n_sne=1, batch_size=1, random_seed=3)
    sim = run.prepare('tests/fixtures/snana/salt3_k21_replay/input.input', **config)
    simulation = None if sim is None else sim.config
    resources = sim
    baseline, _, _ = run.run_simulation(sim)
    return simulation, resources, config, baseline


def test_peak_band_corrections(prepared):
    simulation, resources, config, baseline = prepared
    bands = [filter_band(name) for name in resources.photometry.filter_names]
    offsets = np.arange(1, len(bands) + 1) * 0.1
    sigma = 0.15
    model = replace(simulation.model,
                    genmag_off_model=dict(zip(bands, offsets)),
                    genmag_smear_filter={band: sigma for band in bands})
    changed = replace(simulation, model=model)
    sim = run.prepare(changed)
    result, _, _ = run.run_simulation(sim)
    expected = offsets + sigma * result['parameters']['_magsmear_filter_ran'][0]
    peak = np.asarray(baseline['peakmags'][0])
    valid_peak = (peak > -2) & (peak < 49)
    assert valid_peak.any()
    np.testing.assert_allclose(
        (result['peakmags'][0] - peak)[valid_peak], expected[valid_peak], atol=3e-5)
    filt = np.asarray(result['observation_schedule']['filter_indices'])[0]
    mags = np.asarray(baseline['magnitudes'])[0]
    valid = np.isfinite(mags) & (mags > -2) & (mags < 49)
    assert valid.any()
    np.testing.assert_allclose(
        (result['magnitudes'][0] - mags)[valid], expected[filt[valid]], atol=3e-5)


def test_template_brightness(prepared):
    simulation, resources, config, baseline = prepared
    # A template at peak has nonzero source flux, so its normalisation matters.
    peak = float(baseline['parameters']['peakmjd_true'][0])
    template = {filter_band(name): peak for name in resources.filter_names}
    simulation = replace(simulation, survey=replace(simulation.survey, mjd_template=template))
    sim = run.prepare(simulation)
    baseline, _, _ = run.run_simulation(sim)
    changed = replace(simulation, model=replace(simulation.model, mushift=(1., 1.)))
    sim = run.prepare(changed)
    result, _, _ = run.run_simulation(sim)
    for name in ('x0', 'x1', 'c', 'peakmjd_true'):
        np.testing.assert_array_equal(result['parameters'][name], baseline['parameters'][name])
    flux = np.asarray(baseline['fluxcals'])
    valid = np.isfinite(flux)
    assert valid.any() and np.any(flux[valid] < 0)
    np.testing.assert_allclose(result['fluxcals'][valid], flux[valid] * 10.**(-0.4),
                               rtol=2e-5, atol=2e-4)


@pytest.fixture(scope='module')
def spectral_baseline():
    config = dict(n_sne=1, batch_size=1, random_seed=3)
    sim = run.prepare('tests/fixtures/snana/spectro_optmask_replay/input.input', **config)
    simulation = None if sim is None else sim.config
    resources = sim
    result, _, _ = run.run_simulation(sim)
    spectra, _ = run.compute_spectra_for_mask(
        sim, result, np.array([True]),
        key=jax.random.PRNGKey(3))
    assert spectra
    return simulation, resources, config, result, spectra


@pytest.mark.parametrize('settings', [
    {'genmag_off_global': .5}, {'genmag_smear': .2},
    {'mushift': (.3, .3)}, {'mag_offset': .4},
])
def test_spectral_brightness(spectral_baseline, settings):
    simulation, resources, config, baseline, baseline_spectra = spectral_baseline
    changed = replace(simulation, model=replace(simulation.model, **settings))
    sim = run.prepare(changed)
    result, _, _ = run.run_simulation(sim)
    spectra, _ = run.compute_spectra_for_mask(
        sim, result, np.array([True]),
        key=jax.random.PRNGKey(3))
    for name in ('x0', 'x1', 'c', 'peakmjd_true'):
        np.testing.assert_array_equal(result['parameters'][name], baseline['parameters'][name])
    mags = np.asarray(baseline['magnitudes'])
    valid = np.isfinite(mags) & (mags > -2) & (mags < 49)
    shift = np.asarray(result['magnitudes'] - mags)[valid]
    assert shift.size and abs(shift[0]) > .01
    np.testing.assert_allclose(shift, shift[0], atol=1e-10)
    assert spectra.keys() == baseline_spectra.keys()
    for cid, original in baseline_spectra.items():
        assert len(spectra[cid]) == len(original) > 0
        for before, after in zip(original, spectra[cid]):
            assert np.any(before.sim_flam > 0)
            np.testing.assert_allclose(
                after.sim_flam, before.sim_flam * 10.**(-.4 * shift[0]),
                rtol=1e-10, atol=1e-25)


def test_spectral_event_offsets(spectral_baseline):
    simulation, resources, config, baseline, baseline_spectra = spectral_baseline
    params = dict(baseline['parameters'])
    offsets = {'magoff': .11, 'magsmear': .23, 'lensdmu': -.07, 'snmagshift': .19}
    for name, offset in offsets.items():
        params[name] = np.asarray(params.get(name, np.zeros(1))) + offset
    result = dict(baseline, parameters=params)
    sim = run.prepare(simulation)
    spectra, _ = run.compute_spectra_for_mask(
        sim, result, np.array([True]),
        key=jax.random.PRNGKey(3))
    expected = 10.**(-.4 * sum(offsets.values()))
    assert spectra.keys() == baseline_spectra.keys()
    for cid, original in baseline_spectra.items():
        assert len(spectra[cid]) == len(original) > 0
        for before, after in zip(original, spectra[cid]):
            np.testing.assert_allclose(after.sim_flam, before.sim_flam * expected,
                                       rtol=1e-10, atol=1e-25)
