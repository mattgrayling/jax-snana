"""RedbackModel: signature introspection, configure validation, SED sanity."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("redback_jax")

from jax_snlc_core.logging import SNANAError  # noqa: E402
from jax_snlc_core.models.redback import (DL10PC_CM, RedbackModel,  # noqa: E402
                                          log10_blackbody_flux_density,
                                          smooth_temperature_floor)
from jax_snlc_core.utils import BOLTZMANN_ERG_K, LIGHT_A, PLANCK  # noqa: E402
from jax_snlc_fit.parameters import init_params, snana_parameter_aliases  # noqa: E402
from jax_snlc_sim.distributions import GenGaussAsym  # noqa: E402
from jax_snlc_sim.population import Population  # noqa: E402


_ARNETT_DISTS = {
        'f_nickel': GenGaussAsym(peak=0.1, sigma=(0.02, 0.02), range=(0.01, 0.5)),
        'mej': GenGaussAsym(peak=2.0, sigma=(0.5, 0.5), range=(0.3, 10.0)),
        'vej': GenGaussAsym(peak=9000.0, sigma=(1000.0, 1000.0), range=(3000.0, 20000.0)),
        'kappa': GenGaussAsym(peak=0.1, range=(0.1, 0.1)),
        'kappa_gamma': GenGaussAsym(peak=0.03, range=(0.03, 0.03)),
        'temperature_floor': GenGaussAsym(peak=4000.0, range=(4000.0, 4000.0)),
}


def _populations(dists):
    return {name: Population(distributions=(d,), shifts=((),))
            for name, d in dists.items()}


ARNETT_CONFIG = {
    'genrange_trest': (-20.0, 50.0),
    'populations': _populations(_ARNETT_DISTS),
}


def _arnett():
    model = RedbackModel.load('REDBACK.arnett_bolometric', ARNETT_CONFIG)
    model.configure_generation(ARNETT_CONFIG, 0, 0.0)
    return model


def _sample(model, n_sne, seed=0):
    """Draw the population with the real distribution sampler and add
    what the model derives, merged with the distance the way the simulator
    does, so the result is what generate_sed is actually handed."""
    from jax_snlc_sim.distributions import sample_gengauss_asym

    dists = _ARNETT_DISTS
    base = jax.random.PRNGKey(seed)

    def draw(key_name):
        import zlib
        keys = jax.random.split(
            jax.random.fold_in(base, zlib.crc32(key_name.encode())), n_sne)
        return jax.vmap(lambda key: sample_gengauss_asym(key, dists[key_name]))(keys)

    params = {'mu': jnp.full(n_sne, 38.0),
              **{name: draw(name) for name in dists}}
    return {**params, **model.derived_parameters(params, None)}


def _sed(model, params, i, phase, wavelength_grid):
    static_data = {'wavelength_grid': wavelength_grid}
    per_sn = {k: v[i] for k, v in params.items()}
    return model.generate_sed(phase, per_sn, static_data)


def test_arnett_signature_introspection():
    model = RedbackModel.load('REDBACK.arnett_bolometric', {})
    assert list(model._param_names) == ['f_nickel', 'mej', 'vej', 'kappa',
                                  'kappa_gamma', 'temperature_floor']


def test_unsupported_models_rejected():
    with pytest.raises(SNANAError, match="nn/eta"):
        RedbackModel.load('REDBACK.csm_interaction_bolometric', {})
    with pytest.raises(SNANAError, match="engine owns cosmology"):
        RedbackModel.load('REDBACK.arnett_with_features_cosmology', {})
    with pytest.raises(SNANAError, match="spectra wrapper"):
        RedbackModel.load('REDBACK.arnett_spectra', {})


def test_population_is_the_signature():
    """The sim input declares the population from GENMODEL alone: the
    bolometric function's required parameters, the photosphere's, then the
    optional ones."""
    structured = RedbackModel.structured_input(
        {'GENMODEL': 'REDBACK.arnett_bolometric'})
    names = list(structured['population_parameters'])
    model = RedbackModel.load('REDBACK.arnett_bolometric', {})
    assert names == list(model._param_names) + list(model._optional)
    assert names[:2] == ['f_nickel', 'mej']
    assert RedbackModel.population_parameters == ()


def test_tgrid_defaults_follow_the_caller_phase_window():
    """Unset TMAX covers the caller's rest-frame window plus tail; with no
    window (the fitter) the grid ends at 200 days."""
    windowed = RedbackModel.load('REDBACK.arnett_bolometric',
                            {'genrange_trest': (-20.0, 50.0)})
    assert windowed.lbol_time_grid[-1] == pytest.approx(200.0)
    unwindowed = RedbackModel.load('REDBACK.arnett_bolometric', {})
    np.testing.assert_allclose(unwindowed.lbol_time_grid,
                               np.geomspace(0.02, 200.0, 200))


def test_build_model_delivers_tgrid_to_evaluation():
    """A non-default REDBACK_TGRID_* configuration reaches the evaluable
    model through the shared construction path, so simulation and fitting
    evaluate the same physical numerical model."""
    from jax_snlc_core.models import build_model
    model = build_model(
        'REDBACK.arnett_bolometric',
        {'REDBACK_TGRID_TMIN': 0.05, 'REDBACK_TGRID_TMAX': 300.0,
         'REDBACK_TGRID_NBIN': 256})
    assert len(model.lbol_time_grid) == 256
    assert model.lbol_time_grid[0] == pytest.approx(0.05)
    assert model.lbol_time_grid[-1] == pytest.approx(300.0)
    assert model.get_wavelength_range() == (1000.0, 60000.0)
    assert float(model.wavelength_grid[0]) == pytest.approx(1000.0)
    assert float(model.wavelength_grid[-1]) == pytest.approx(60000.0)


def test_sample_and_generate_sed():
    model = _arnett()
    n_sne = 4
    params = _sample(model, n_sne)
    assert 'lbol_grid' not in params and 't_peak' not in params

    # Phases are explosion-relative: t=0 is the explosion (zero flux).
    lam = jnp.geomspace(1000.0, 20000.0, 200)
    sed_near_peak = np.asarray(_sed(model, params, 0, 20.0, lam))
    sed_late = np.asarray(_sed(model, params, 0, 80.0, lam))
    sed_explosion = np.asarray(_sed(model, params, 0, 0.0, lam))
    sed_pre = np.asarray(_sed(model, params, 0, -100.0, lam))
    assert np.all(sed_near_peak >= 0) and np.any(sed_near_peak > 0)
    assert sed_near_peak.sum() > sed_late.sum()
    assert np.all(sed_explosion == 0.0)
    assert np.all(sed_pre == 0.0)

    # distance enters as 10^(-0.4*mu), so 2.5*log10(2) brighter doubles the SED
    params2 = dict(params)
    params2['mu'] = params['mu'] - 2.5 * np.log10(2.0)
    sed_x2 = np.asarray(_sed(model, params2, 0, 20.0, lam))
    np.testing.assert_allclose(sed_x2, 2.0 * sed_near_peak, rtol=1e-6)


def test_shock_cooling_smoke():
    model = RedbackModel.load('REDBACK.shock_cooling_bolometric', {})
    dists = {
        'log10_mass': GenGaussAsym(peak=-1.0, range=(-1.0, -1.0)),
        'log10_radius': GenGaussAsym(peak=13.0, range=(13.0, 13.0)),
        'log10_energy': GenGaussAsym(peak=51.0, range=(51.0, 51.0)),
        'nn': GenGaussAsym(peak=10.0, range=(10.0, 10.0)),
        'delta': GenGaussAsym(peak=1.1, range=(1.1, 1.1)),
        'kappa': GenGaussAsym(peak=0.34, range=(0.34, 0.34)),
        'vej': GenGaussAsym(peak=9000.0, range=(9000.0, 9000.0)),
        'temperature_floor': GenGaussAsym(peak=4000.0, range=(4000.0, 4000.0)),
    }
    structured = RedbackModel.structured_input(
        {'GENMODEL': 'REDBACK.shock_cooling_bolometric'})
    assert set(structured['population_parameters']) == set(dists)
    assert set(model._param_names) | set(model._optional) == set(dists)


def test_header_params_float32_no_grid():
    model = _arnett()
    params = _sample(model, 2)
    header = model.get_snana_header_params(
        {k: np.asarray(v) for k, v in params.items()})
    assert set(header) == {f'SIM_REDBACK_{n}'
                           for n in [*model._param_names, *model._optional]}
    assert all(v.dtype == np.float32 for v in header.values())


# ---------------------------------------------------------------------------
# Fitting interface
# ---------------------------------------------------------------------------

_PHYS = ['f_nickel', 'mej', 'vej', 'kappa', 'kappa_gamma', 'temperature_floor']


def test_fittable_set_declares_mu_pinned():
    """mu is always declared, sitting after t0, and carries a zero default step
    so it is held at the cosmological distance modulus unless INISTP_mu frees
    it. Distance is degenerate with the luminosity scale, so free-by-default
    would be unfittable."""
    model = RedbackModel.load('REDBACK.arnett_bolometric', {})
    assert list(model.get_fit_parameters()) == ['PKMJD', 'mu'] + _PHYS
    assert model.get_fit_parameters()['mu']['inistp'] == 0.0

    # and is the only one pinned: a step of zero holds a parameter, and every
    # other declared step is non-zero, which frees it
    declared = model.get_fit_parameters()
    assert [n for n, spec in declared.items()
            if spec.get('inistp') == 0.0] == ['mu']


def test_mu_is_seeded_from_the_caller_not_a_model_cosmology():
    """The model must not compute distances: whatever mu_ref the caller supplies
    is what mu is seeded at, so H0_REF etc. reach the fit unmodified."""
    z = jnp.array([0.01, 0.05, 0.1, 0.3])
    model = RedbackModel.load('REDBACK.arnett_bolometric', {})
    mu_ref = jnp.array([33.0, 36.5, 38.2, 41.0])
    seeded = init_params(model, len(z), jnp.zeros(len(z)), mu_ref)['mu']
    assert np.allclose(np.asarray(seeded), np.asarray(mu_ref), rtol=0, atol=0.0)


def test_bounds_init_scale_completeness():
    model = RedbackModel.load('REDBACK.arnett_bolometric', {})
    declared = model.get_fit_parameters()
    names = list(declared)
    bounds = {n: spec['bounds'] if 'bounds' in spec
              else spec['bounds_about_peak_estimate']
              for n, spec in declared.items()}
    scales = {n: spec['scale'] for n, spec in declared.items()}
    init = init_params(model, 3, jnp.full(3, 60000.0), jnp.full(3, 36.5))

    assert set(bounds) == set(names) == set(scales)
    assert set(init) == set(names)
    for name in names:
        lo, hi = bounds[name]
        assert lo < hi and scales[name] > 0
        if name in _PHYS:  # t0/mu inits are per-SN, not from the table
            val = float(np.asarray(init[name])[0])
            assert lo < val < hi


def test_build_init_params_t0_passthrough():
    """t0 init IS the data's t0 estimate: the sim's PEAKMJD is the explosion
    epoch for redback, so no rise-time shift is applied."""
    model = RedbackModel.load('REDBACK.arnett_bolometric', {})
    z = jnp.array([0.05, 0.2])
    t0_est = jnp.array([60000.0, 60010.0])
    init = init_params(model, 2, t0_est, jnp.full(2, 36.5))
    np.testing.assert_array_equal(np.asarray(init['PKMJD']), np.asarray(t0_est))


@pytest.mark.parametrize("temperature", [1000., 5000.])
def test_blackbody_wien_gradient(temperature):
    frequency = jnp.asarray(LIGHT_A / np.array([1000., 5000., 11000.]), dtype=jnp.float32)
    temp = jnp.asarray(temperature, dtype=jnp.float32)
    def logflux(t):
        return log10_blackbody_flux_density(t, 15., DL10PC_CM, frequency)
    gradient = jax.jacfwd(logflux)(temp)
    x = (PLANCK / BOLTZMANN_ERG_K) * np.asarray(frequency, dtype=float) / temperature
    expected = x / (temperature * -np.expm1(-x) * np.log(10.))
    np.testing.assert_allclose(gradient, expected, rtol=2e-6)
    assert np.isfinite(jax.grad(lambda t: jnp.sum(10. ** logflux(t)))(temp))


def test_generate_sed_fit_gradient_flow():
    """Gradients flow through the fit path: dict built from a traced array."""
    model = RedbackModel.load('REDBACK.arnett_bolometric', {})
    static = {'wavelength_grid': jnp.linspace(2000.0, 11000.0, 500)}
    names = list(model.get_fit_parameters())[1:]
    # mu leads the physics parameters; distance reaches the fit path through it.
    params_arr = jnp.array([36.0, 0.15, 3.0, 9000.0, 0.1, 0.03, 5000.0])
    phases = jnp.array([5.0, 15.0, 30.0, 60.0, 100.0])

    def loss(p):
        params = {name: p[i] for i, name in enumerate(names)}
        seds = jax.vmap(lambda ph: model.generate_sed(ph, params, static))(phases)
        return jnp.sum(seds)

    grad = jax.grad(loss)(params_arr)
    assert np.all(np.isfinite(np.asarray(grad)))
    assert np.all(np.asarray(grad) != 0.0)  # every fitted param influences some phase
    hess_diag = jnp.diag(jax.hessian(loss)(params_arr))
    assert np.all(np.isfinite(np.asarray(hess_diag)))


def test_sim_and_fit_carry_the_same_distance_parameter():
    """Simulating and fitting used to disagree: the sampler stored a precomputed
    ``distance_scale`` and the fit stored ``mu``, so generate_sed carried a
    branch to accept either. Both now read ``mu``, so there is one encoding and
    no branch: the distance modulus itself is the model's distance parameter,
    a magnitude, so a distance update adds to it."""
    model = RedbackModel.load('REDBACK.arnett_bolometric', {})
    sampled = _sample(model, n_sne=2)
    assert 'mu' in sampled and 'distance_scale' not in sampled
    assert 'mu' in model.get_fit_parameters()
    assert model.distance_param == 'mu'
    params = {'mu': jnp.array([38.0]), 'other': jnp.ones(1)}
    shifted = model.shift_amplitude(params, 0.5)
    assert float(shifted['mu'][0]) == 38.5 and shifted['other'] is params['other']


def test_kilonova_fitting_rejected():
    model = RedbackModel.load('REDBACK.metzger_kilonova_bolometric', {})
    with pytest.raises(SNANAError, match="not yet been validated"):
        model.get_fit_parameters()


def test_inistp_names_resolve_against_the_model():
    """Pinning is generic now: INIVAL_/INISTP_ suffixes resolve case-insensitively
    against the model's own parameters, and an unknown one aborts naming them."""
    from jax_snlc_fit.config import FitConfig, Model
    from jax_snlc_fit.parameters import resolve_parameter_keys

    model = RedbackModel.load('REDBACK.arnett_bolometric', {})
    names = list(model.get_fit_parameters())

    cfg = FitConfig(model=Model(inival={'KAPPA': 0.07},
                                inistp={'KAPPA': 0.0}))
    resolved = resolve_parameter_keys(cfg.model, names,
                                      snana_parameter_aliases(model))
    assert resolved['inival'] == {'kappa': 0.07}
    assert resolved['inistp'] == {'kappa': 0.0}

    cfg = FitConfig(model=Model(inival={'bogus': 1.0}))
    with pytest.raises(SNANAError, match="bogus"):
        resolve_parameter_keys(cfg.model, names,
                               snana_parameter_aliases(model))


def test_every_fit_parameter_declares_a_truth_column():
    """USESIM_* needs a SIM_ column per parameter. Redback discovers its
    parameter set from the transient function's signature, so the mapping has
    to be generated rather than listed — a new redback model must not silently
    lose USESIM support."""
    model = RedbackModel.load('REDBACK.arnett_bolometric', {})
    declared = model.get_fit_parameters()
    missing = [n for n, spec in declared.items() if 'sim_truth' not in spec]
    assert not missing, f"no truth column for {missing}"
    assert declared['mu']['sim_truth'] == 'SIM_DLMU'
    assert declared['mej']['sim_truth'] == 'SIM_REDBACK_mej'


def test_non_finite_flux_aborts_with_context():
    """A parameter set that makes the model's flux non-finite aborts after
    the photometry with the model name, the event and its drawn parameters,
    instead of failing every cut as a very dark transient."""
    from types import SimpleNamespace
    from jax_snlc_sim.photometry import check_finite_flux

    model = _arnett()
    # the run's population list, which names the drawn parameters
    model.configured_population = tuple((name, ()) for name in model._param_names)
    n, max_obs = 4, 3
    params = {name: jnp.full(n, 1.0) for name in model._param_names}
    params['mej'] = jnp.array([1.0, jnp.nan, 1.0, 1.0])
    fluxcals = jnp.ones((n, max_obs)).at[1, 2].set(jnp.nan)
    results = {'parameters': params, 'fluxcals': fluxcals,
               'filter_z_valid': jnp.ones((n, max_obs), dtype=bool)}
    schedule = {'filter_indices': jnp.zeros((n, max_obs), dtype=jnp.int32)}
    with pytest.raises(SNANAError, match="non-finite flux for event 11 .*mej=nan"):
        check_finite_flux(SimpleNamespace(model=model), results,
                          schedule, global_start_idx=10, n_events=n)

    # an epoch that is not observed, or whose filter is invalid at the
    # event's redshift, is NaN by design
    schedule['filter_indices'] = schedule['filter_indices'].at[1, 2].set(-1)
    check_finite_flux(SimpleNamespace(model=model), results, schedule, 10, n)
    schedule['filter_indices'] = jnp.zeros((n, max_obs), dtype=jnp.int32)
    results['filter_z_valid'] = results['filter_z_valid'].at[1, 2].set(False)
    check_finite_flux(SimpleNamespace(model=model), results, schedule, 10, n)


@pytest.mark.parametrize("registry_name", [
    "metzger_kilonova_bolometric", "magnetar_boosted_kilonova_bolometric",
])
def test_optional_defaults_in_simulation(registry_name):
    """Unconfigured optional parameters still belong to each simulated event,
    so the simulator's population checks and search-efficiency inputs work."""
    name = f"REDBACK.{registry_name}"
    config = RedbackModel.structured_input({'GENMODEL': name})
    model = RedbackModel.load(name, config)
    # the builder's step: the population the input declares
    model.configured_population = tuple((n, ()) for n in config['population_parameters'])
    configured = dict.fromkeys(model._param_names, 1.0)
    model.configure_generation({'populations': configured}, 2, 0.0)

    def sample(values):
        params = {'mu': jnp.full(2, 38.0),
                  **{name: jnp.full(2, value) for name, value in values.items()}}
        return {**params, **model.derived_parameters(params, None)}

    params = sample(configured)
    variables = model.sn_variables(params)
    truth = model.get_snana_header_params(params)
    for name, default in model._optional.items():
        np.testing.assert_allclose(variables[name], [default, default])
        np.testing.assert_allclose(truth[f'SIM_REDBACK_{name}'], [default, default])
    # A user-supplied population still supersedes the engine's default.
    configured['vmax'] = 0.8
    model.configure_generation({'populations': configured}, 2, 0.0)
    np.testing.assert_allclose(model.sn_variables(sample(configured))['vmax'],
                               [0.8, 0.8])


@pytest.mark.parametrize("registry_name,engine_params", [
    ("shocked_cocoon_bolometric", dict(
        mej=0.1, vej=0.1, eta=3.0, tshock=100.0, shocked_fraction=0.5,
        cos_theta_cocoon=0.5, kappa=0.2)),
    ("metzger_kilonova_bolometric", dict(
        mej=0.05, vej=0.1, beta=3.0, kappa=1.0)),
    ("magnetar_boosted_kilonova_bolometric", dict(
        mej=0.05, vej=0.1, beta=3.0, kappa=1.0,
        p0=2.0, bp=1.0, mass_ns=1.4, theta_pb=1.0)),
    ("arnett_bolometric", dict(
        f_nickel=0.15, mej=3.0, vej=29979.2458, kappa=0.1, kappa_gamma=0.03)),
])
def test_velocity_units_in_sed(registry_name, engine_params):
    """Real engines receive native velocity units; their photospheres expand
    at the corresponding physical speed of 0.1c."""
    model = RedbackModel.load(f"REDBACK.{registry_name}", {})
    tgrid = jnp.asarray(model.lbol_time_grid)
    index = int(np.argmin(np.abs(np.asarray(tgrid) - 2.0)))
    phase = tgrid[index]
    # At a grid knot, the SED must use the engine's luminosity exactly.
    log_luminosity = model._bolo_fn(tgrid, **engine_params)[index]
    lam = jnp.array([4000.0, 6000.0, 9000.0])
    sed = model.generate_sed(phase, {
        **engine_params, 'temperature_floor': 100.0, 'mu': 35.0,
    }, {'wavelength_grid': lam})
    radius = 29979.2458 * 1.e5 * float(phase) * 86400.0
    temperature = (10.0 ** float(log_luminosity)
                   / (4.0 * np.pi * radius ** 2 * 5.6704e-5)) ** 0.25
    assert temperature > 200.0  # Safely on the expanding photosphere branch.
    wave = np.asarray(lam)
    frequency = LIGHT_A / wave
    expected = (2.0 * np.pi * PLANCK * frequency ** 3 / (LIGHT_A * 1.e-8) ** 2
                / np.expm1(PLANCK * frequency / (BOLTZMANN_ERG_K * temperature))
                * (radius / 3.0856775814913673e19) ** 2
                * LIGHT_A / wave ** 2 * 10.0 ** (-0.4 * 35.0))
    np.testing.assert_allclose(sed, expected, rtol=2e-5)


def test_cocoon_fit_velocity_units():
    """The fitter's initial physical vector must feed a subluminal cocoon
    engine and a photosphere expanding at that same velocity."""
    model = RedbackModel.load('REDBACK.shocked_cocoon_bolometric', {})
    specs = model.get_fit_parameters()
    lo, hi = specs['vej']['bounds']
    velocity = specs['vej']['init']
    assert 0.0 < lo < velocity < hi < 1.0
    names = list(specs)
    vector = jnp.asarray([specs[n].get('init', 0.0) for n in names])
    params = model.sed_params_from_fit(vector, 0.1)
    assert float(params['vej']) == pytest.approx(velocity)
    from jax_snlc_core.utils import LIGHT_KM
    arnett = RedbackModel.load('REDBACK.arnett_bolometric', {}).get_fit_parameters()['vej']
    np.testing.assert_allclose(np.array([lo, hi]) * LIGHT_KM, arnett['bounds'])
    assert velocity * LIGHT_KM == pytest.approx(arnett['init'])
    assert specs['vej']['scale'] * LIGHT_KM == pytest.approx(arnett['scale'])


def _direct_arnett_sed(model, phase, params, lam):
    """Evaluate the upstream engine at the requested time, without our grid."""
    luminosity = model._bolo_fn(
        jnp.stack([jnp.asarray(phase), jnp.asarray(model.lbol_time_grid[-1])]),
        **{k: params[k] for k in model._sig_params})[0]
    return _photosphere_sed(model, phase, params, lam, luminosity)


def _photosphere_sed(model, phase, params, lam, log_luminosity):
    """Convert bolometric luminosity to a spectrum using the model photosphere."""
    temperature, radius = model._photosphere_fn(
        time=phase, log10_luminosity=log_luminosity, vej=params['vej'],
        temperature_floor=params['temperature_floor'])
    temperature, radius = smooth_temperature_floor(
        phase, temperature, radius, params['vej'], params['temperature_floor'],
        model.redback_photosphere_smooth_width)
    log_flux = log10_blackbody_flux_density(
        temperature, radius, DL10PC_CM, LIGHT_A / lam)
    return 10.0 ** log_flux * LIGHT_A / lam ** 2 * 10.0 ** (-0.4 * params['mu'])


def _interpolation_parameters():
    return dict(f_nickel=.15, mej=3., vej=9000., kappa=.1,
                kappa_gamma=.03, temperature_floor=4000., mu=38.)


def test_arnett_interpolation_preserves_knots_and_endpoint_behavior():
    model = RedbackModel.load('REDBACK.arnett_bolometric', {})
    params = _interpolation_parameters()
    lam = jnp.array([3500., 5000., 8000.])
    sed = lambda t: model.generate_sed(t, params, {'wavelength_grid': lam})
    for i in [5, 100, 150, 180, 199]:
        t = model.lbol_time_grid[i]
        np.testing.assert_allclose(sed(t), _direct_arnett_sed(model, t, params, lam),
                                   rtol=1e-9, atol=0.)
    for t in [-1., 0., model.lbol_time_grid[0]]:
        np.testing.assert_array_equal(sed(t), jnp.zeros_like(lam))
    assert np.all(np.asarray(sed(model.lbol_time_grid[0] * 1.001)) > 0)
    np.testing.assert_allclose(sed(400.), sed(200.), rtol=1e-12)
    np.testing.assert_array_equal(jax.jacfwd(sed)(400.), jnp.zeros_like(lam))


def test_arnett_phase_derivative_is_continuous_at_interior_grid_knots():
    model = RedbackModel.load('REDBACK.arnett_bolometric', {})
    params = _interpolation_parameters()
    def log_flux(log_time):
        return jnp.log(model.generate_sed(jnp.exp(log_time), params,
                                          {'wavelength_grid': jnp.array([5000.])})[0])
    derivative = jax.jit(jax.grad(log_flux))
    # Rising phases well before the independent temperature-floor transition.
    knots = jnp.log(jnp.asarray(model.lbol_time_grid[jnp.array([110, 120, 130, 140])]))
    left = jax.vmap(derivative)(knots - 1e-7)
    right = jax.vmap(derivative)(knots + 1e-7)
    np.testing.assert_allclose(left, right, rtol=1e-5, atol=1e-5)


def test_arnett_flux_and_phase_derivative_match_direct_bolometric_evaluation():
    """Interpolation preserves the upstream flux and first phase derivative."""
    model = RedbackModel.load('REDBACK.arnett_bolometric', {})
    params = _interpolation_parameters()
    lam = jnp.array([3500., 5000., 8000.])
    def interpolated(x):
        return jnp.log(model.generate_sed(jnp.exp(x), params,
                                          {'wavelength_grid': lam}))
    def direct(x):
        return jnp.log(_direct_arnett_sed(model, jnp.exp(x), params, lam))
    # Compare away from grid endpoints and the separate photosphere kink.
    phases = jnp.array([1.3, 4.3, 12.3, 25.3, 100.3])
    xs = jnp.log(phases)
    for order, tolerance in [(0, 1e-5), (1, 5e-4)]:
        f, g = interpolated, direct
        for _ in range(order):
            f, g = jax.jacfwd(f), jax.jacfwd(g)
        got = jax.jit(jax.vmap(f))(xs)
        expected = jax.jit(jax.vmap(g))(xs)
        np.testing.assert_allclose(got, expected, rtol=0., atol=tolerance)


def test_arnett_phase_curvature_matches_smooth_diffusion_integral():
    """Check curvature against converged quadrature of analytic Ni/Co heating.

    The upstream engine linearly interpolates heating inside its integral;
    differentiating that interpolation twice is not a curvature reference.
    """
    model = RedbackModel.load('REDBACK.arnett_bolometric', {})
    params = _interpolation_parameters()
    lam = jnp.array([3500., 5000., 8000.])
    xs = jnp.log(jnp.array([1.3, 4.3, 12.3, 25.3, 100.3]))
    # Match the engine's physical constants; timescales are in days squared.
    mass = params['mej'] * 1.989e33
    velocity = params['vej'] * 1.e5
    diffusion_time_squared = (2 * params['kappa'] * mass
                              / (13.7 * 2.998e10 * velocity) / 86400.**2)
    trapping_time_squared = (3 * params['kappa_gamma'] * mass
                             / (4 * np.pi * velocity**2) / 86400.**2)

    def reference(log_time, nodes, weights):
        time = jnp.exp(log_time)
        integration_times = time * nodes
        heating = params['f_nickel'] * params['mej'] * (
            6.45e43 * jnp.exp(-integration_times / 8.8)
            + 1.45e43 * jnp.exp(-integration_times / 111.3))
        integrand = (heating * integration_times
                     * jnp.exp((integration_times**2 - time**2) / diffusion_time_squared))
        integral = time * jnp.sum(weights * integrand)
        luminosity = (2 * integral / diffusion_time_squared
                      * -jnp.expm1(-trapping_time_squared / time**2))
        return jnp.log(_photosphere_sed(
            model, time, params, lam, jnp.log10(luminosity)))

    def interpolated(log_time):
        return jnp.log(model.generate_sed(jnp.exp(log_time), params,
                                          {'wavelength_grid': lam}))

    reference_curvature = jax.jit(jax.vmap(
        jax.jacfwd(jax.jacfwd(reference)), in_axes=(0, None, None)))
    estimates = []
    for size in (128, 256):
        nodes, weights = np.polynomial.legendre.leggauss(size)
        estimates.append(reference_curvature(xs, (nodes + 1) / 2, weights / 2))
    np.testing.assert_allclose(estimates[0], estimates[1], rtol=0., atol=1e-8)
    got = jax.jit(jax.vmap(jax.jacfwd(jax.jacfwd(interpolated))))(xs)
    np.testing.assert_allclose(got, estimates[1], rtol=0., atol=.1)


def test_arnett_physics_gradient_includes_the_luminosity_slopes():
    model = RedbackModel.load('REDBACK.arnett_bolometric', {})
    params = _interpolation_parameters()
    lam = jnp.array([5000.])
    def flux(mej):
        p = {**params, 'mej': mej}
        return jnp.log(model.generate_sed(12.3, p, {'wavelength_grid': lam})[0])
    def direct(mej):
        p = {**params, 'mej': mej}
        return jnp.log(_direct_arnett_sed(model, 12.3, p, lam)[0])
    gradient = jax.grad(flux)(3.)
    finite_difference = (flux(3. + 1e-4) - flux(3. - 1e-4)) / 2e-4
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-6)
    np.testing.assert_allclose(gradient, jax.grad(direct)(3.), rtol=.01, atol=.001)


@pytest.mark.parametrize('halfwidth', [0.001, 0.003, 0.01])
def test_smooth_photosphere_preserves_branches_energy_and_c2(halfwidth):
    floor = 4000.
    time, velocity = 10., 9000.
    log_rej = jnp.log(velocity * 1.e5 * 86400. * time)
    width = np.log1p(halfwidth)

    def original(x):
        log_t = jnp.log(floor) + jnp.maximum(x, 0.)
        log_r = log_rej + 2 * jnp.minimum(x, 0.)
        return jnp.exp(log_t), log_r / jnp.log(10.)

    def smoothed(x):
        temperature, radius = original(x)
        return smooth_temperature_floor(time, temperature, radius,
                                        velocity, floor, halfwidth)

    def log_outputs(x):
        temperature, radius = smoothed(x)
        return jnp.stack([jnp.log(temperature), radius * jnp.log(10.)])

    for x in [-2 * width, 2 * width]:
        np.testing.assert_array_equal(smoothed(x), original(x))
    xs = jnp.linspace(-2 * width, 2 * width, 101)
    temperature, radius = jax.vmap(smoothed)(xs)
    assert np.all(np.asarray(temperature) >= floor * (1 - 1.e-14))
    old_t, old_r = jax.vmap(original)(xs)
    np.testing.assert_allclose(2 * radius * jnp.log(10.) + 4 * jnp.log(temperature),
        2 * old_r * jnp.log(10.) + 4 * jnp.log(old_t), rtol=0., atol=1.e-12)
    for order in [0, 1, 2]:
        f = log_outputs
        for _ in range(order):
            f = jax.jacfwd(f)
        for edge in [-width, 0., width]:
            eps = width * 1.e-8
            np.testing.assert_allclose(f(edge - eps), f(edge + eps),
                                       rtol=1.e-5, atol=1.e-3)


def test_smooth_photosphere_zero_width_and_input_validation():
    np.testing.assert_array_equal(
        smooth_temperature_floor(1., 5000., 14., 9000., 4000., 0.), (5000., 14.))
    for width in [-.01, np.nan, np.inf]:
        with pytest.raises(SNANAError, match='finite and nonnegative'):
            RedbackModel.load('REDBACK.arnett_bolometric',
                         {'REDBACK_PHOTOSPHERE_SMOOTH_WIDTH': width})
    model = RedbackModel.load('REDBACK.arnett_bolometric',
                        {'REDBACK_PHOTOSPHERE_SMOOTH_WIDTH': .003})
    assert model.redback_photosphere_smooth_width == .003
    default = RedbackModel.load('REDBACK.arnett_bolometric', {})
    assert default.redback_photosphere_smooth_width == .003
    sharp = RedbackModel.load('REDBACK.arnett_bolometric',
                        {'REDBACK_PHOTOSPHERE_SMOOTH_WIDTH': 0.})
    assert sharp.redback_photosphere_smooth_width == 0.
