"""Model values can change without reusing stale data or compiling again."""

from copy import deepcopy

import jax
import numpy as np
import pytest

from jax_pipeline import Fit
from jax_snlc_fit import LightCurveFitter


def test_pipeline_model_variants_share_compilation():
    template = LightCurveFitter.from_model(
        'SALT3.K21', batch_size=1, n_fit_iterations=1, maxiter_round1=2)
    models = [template.model, deepcopy(template.model)]
    models[1].m0 = models[1].m0 * 1.02
    models[1].fluxcal_scale_mag_offset *= 1.01
    cells = [(0, 0), (1, 0)]
    configs = {cell: deepcopy(template.config) for cell in cells}
    groups = [([cell], model) for cell, model in zip(cells, models)]
    files = {cell: dict(kcor=template.calibration, shared={}) for cell in cells}
    shape = dict(epoch_lengths=[], batch_size=1, n_z=None, band_window=None)
    prepared = Fit._fitters(configs, groups, files, shape)
    fitters = [prepared[cell]['fitter'] for cell in cells]
    times = np.linspace(59990., 60035., 12)
    observations = dict(
        mjd=times, flux=100. * np.exp(-.5 * ((times - 60005.) / 15.)**2),
        flux_err=np.full(12, 5.), band='g_PS1', redshift=.1, progress=False)
    results = []
    for fitter in (fitters[0], fitters[1], fitters[0]):
        results.append(fitter.fit(**observations))
    for name in ('_run_round', '_run_adjust', '_run_errors', '_run_weights'):
        first, second = (getattr(f, name) for f in fitters)
        assert first is second
        assert first._cache_size() == (0 if name == '_run_weights' else 1)
    np.testing.assert_array_equal(results[0].params['x0'], results[2].params['x0'])
    assert not np.array_equal(results[0].params['x0'], results[1].params['x0'])

    # A fitter's later settings cannot mutate another fitter's compiled code.
    original_round = fitters[0]._run_round
    fitters[1].config.fit.maxiter_round1 += 1
    fitters[1]._prepare_kernels()
    assert fitters[1]._run_round is not original_round
    assert fitters[0]._run_round is original_round
    assert original_round._cache_size() == 1


def test_model_arrays_are_dynamic_but_structure_is_static():
    from jax_snlc_core.models import SALTModel

    model = LightCurveFitter.from_model('SALT3.K21').model
    changed = deepcopy(model)
    changed.m0 = model.m0 * 1.02
    evaluate = jax.jit(lambda m: m.m0.sum())
    first = evaluate(model)
    second = evaluate(changed)
    np.testing.assert_allclose(second, 1.02 * first, rtol=1e-6)
    assert evaluate._cache_size() == 1
    changed.opt_salt2fit = 1
    evaluate(changed)
    assert evaluate._cache_size() == 2
    restored = jax.tree.unflatten(jax.tree.structure(model), jax.tree.leaves(model))
    assert isinstance(restored, SALTModel)
    np.testing.assert_array_equal(restored.m0, model.m0)


@pytest.mark.parametrize("model_name", [
    "SALT3.K21", "BAYESN.W22", "REDBACK.arnett_bolometric",
])
def test_bundled_model_phase_derivatives_with_dynamic_arrays(data_root, model_name):
    import jax.numpy as jnp
    from jax_snlc_core.models import build_model
    from jax_snlc_fit.fit_spec import init_params

    if model_name.startswith("REDBACK."):
        pytest.importorskip("redback_jax")
        path = model_name
    else:
        path = data_root / "models" / model_name
    model = build_model(str(path), {}, n_wave_points=32)
    initial = init_params(model, 1, jnp.array([0.]), jnp.array([35.]))
    vector = jnp.array([initial[name][0] for name in model.get_fit_parameters()])
    params = model.sed_params_from_fit(vector, .1)

    def flux(m, phase):
        return m.generate_sed(phase, params, dict(
            wavelength_grid=m.wavelength_grid, latetime_extrap=True)).sum() * 1e15

    value_and_derivatives = lambda m, phase: (
        flux(m, phase), jax.grad(flux, argnums=1)(m, phase),
        jax.hessian(flux, argnums=1)(m, phase))
    expected = value_and_derivatives(model, 10.)
    actual = jax.jit(value_and_derivatives)(model, 10.)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_different_preparations_do_not_share_fit_functions():
    template = LightCurveFitter.from_model('SALT3.K21', batch_size=1)
    cache = {}
    shape = dict(epoch_lengths=[], batch_size=1, n_z=None, band_window=None)

    def prepare(model):
        return LightCurveFitter._from_prepared(
            template.config, model, template.calibration, shape, kernel_cache=cache)

    original = prepare(template.model)
    identical = prepare(deepcopy(template.model))
    assert identical._run_round is original._run_round
    for attribute, value in (
        ('prior_mjdsig', 10.),
        ('wavelength_grid', template.model.wavelength_grid + 1.),
        ('host_colorlaw', 'CCM89'),
    ):
        changed = deepcopy(template.model)
        setattr(changed, attribute, value)
        assert prepare(changed)._run_round is not original._run_round


def test_bayesn_photoz_uses_prepared_model_data(snana_fixture):
    from _fit_helpers import build_fit_model, build_fitter, fit_config, load_batch, run_fit

    fixture = snana_fixture('bayesn_fit_dense')
    config = fit_config(
        fixture.root / 'fit.nml', dofit_photoz=True, opt_photoz=2,
        prior_zerrscale=1., photoz_bound=(.01, .4), batch_size=1)
    batch = load_batch(config, fixture.head_fits, fixture.phot_fits, max_fit=1)
    model = build_fit_model(config)
    result = run_fit(build_fitter(config, model, batch), batch)
    for values in result.params.values():
        assert np.all(np.isfinite(values))
    np.testing.assert_allclose(result.params['zPHOT'], batch.redshifts, atol=.01)
