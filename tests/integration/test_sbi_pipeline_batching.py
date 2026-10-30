"""The SBI pipeline from build_sbi_pipeline: population hyperparameters, samples,
custom and conditional functions, batching, compilation and gradients."""

import dataclasses
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.extend.core import ClosedJaxpr, Jaxpr

from jax_snlc_core.models.salt import compute_expected_x0_scale
from jax_snlc_core.logging import SNANAError
from jax_snlc_sim import run as sim_run
from jax_snlc_core.random import root_key


_KCOR = "$JAX_TEST_FIXTURE_ROOT/kcor/JAX_TEST_LSST.fits"
_SIMLIB = "$JAX_TEST_FIXTURE_ROOT/simlib/JAX_TEST_LSST.SIMLIB"
_MODEL = "$JAX_TEST_FIXTURE_ROOT/models/SALT3.K21"


def _write_input(out: Path) -> None:
    out.write_text(f"""\
GENVERSION:  JAX_TEST_SBI_CHUNK
NGENTOT_LC:  100
FORMAT_MASK:  32
GENTYPE:  1
SNTYPES_Ia:  1  101
CIDRAN_MIN:  0
CIDRAN_MAX:  3000000
GENMODEL:  {_MODEL}
GENPEAK_SALT2x1:    0.0
GENRANGE_SALT2x1:  -2.0  2.0
GENSIGMA_SALT2x1:   1.0  1.0
GENPEAK_SALT2c:     0.0
GENRANGE_SALT2c:   -0.2  0.3
GENSIGMA_SALT2c:    0.05  0.05
GENPEAK_SALT2ALPHA: 0.14
GENPEAK_SALT2BETA:  3.1
OMEGA_MATTER:  0.3150
OMEGA_LAMBDA:  0.6850
w0_LAMBDA:  -1.0000
wa_LAMBDA:  0
GENFILTERS:  ugrizY
KCOR_FILE:  {_KCOR}
SIMLIB_FILE:  {_SIMLIB}
GENRANGE_REDSHIFT:  0.05  0.40
GENRANGE_MJD:  60500.0  61500.0
GENRANGE_PEAKMJD:  60540.0  61460.0
GENSIGMA_REDSHIFT:  0.001
GENRANGE_TREST: -20.0 50.0
SOLID_ANGLE: 0.015
OPT_MWEBV: 0
DNDZ: POWERLAW 2.5E-5 1.5
RANSEED:  12345
""")


def _build_sim(tmp_path: Path, n_sne: int):
    """A small SALT simulator with a fixed seed."""
    inp = tmp_path / "input.input"
    _write_input(inp)
    return sim_run.prepare(str(inp), n_sne=n_sne, batch_size=n_sne, random_seed=12345)


class _Records(logging.Handler):
    """The log records a logger emits while attached."""

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _assert_close(a, b, msg):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind in 'fc':
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6, err_msg=msg)
    else:
        np.testing.assert_array_equal(a, b, err_msg=msg)


def test_sampled_streams_batch_invariant(tmp_path):
    """VPEC, lensing, phase-correlation and MWEBV draws are keyed per SN by
    name, so sampling is identical for any batch size."""
    inp = tmp_path / "input.input"
    _write_input(inp)
    inp.write_text(inp.read_text().replace("OPT_MWEBV: 0\n", "")
                   + "GENSIGMA_VPEC: 300\n"
                     "WEAKLENS_DSIGMADZ: 0.05\n"
                     "GENMAG_SMEAR_ADDPHASECOR: 0.02 30\n"
                     "OPT_MWEBV: 1\nGENSIGMA_MWEBV_RATIO: 0.16\n"
                     "GENSIGMA_MWEBV: 0.01\n")

    def run(batch_size):
        kwargs = dict(n_sne=40, random_seed=12345)
        if batch_size is not None:
            kwargs['batch_size'] = batch_size
        simulator = sim_run.prepare(str(inp), **kwargs)
        results, _, _ = sim_run.run_simulation(simulator)
        return results['parameters']

    p_un, p_ch = run(40), run(13)
    for key in ('vpec_true', 'vpec', 'lensdmu', 'mwebv', 'sim_mwebv'):
        np.testing.assert_array_equal(
            np.asarray(p_un[key]), np.asarray(p_ch[key]), err_msg=key)
    # The phase-correlated grid is the one draw that passes through a matrix
    # product, L @ z against the Cholesky factor. Batches sample separately,
    # so that contraction runs over a different leading dimension and XLA is
    # free to reorder the summation; the draw is the same, its last ulp is
    # not. Compared to float64 tolerance rather than bit-exactly, since bit
    # exactness across array shapes is not something XLA promises.
    np.testing.assert_allclose(
        np.asarray(p_un['_phasecor_grid']), np.asarray(p_ch['_phasecor_grid']),
        rtol=1e-12, atol=0.0, err_msg='_phasecor_grid')
    assert np.any(np.asarray(p_un['vpec_true']) != 0)
    assert np.any(np.asarray(p_un['lensdmu']) != 0)
    assert np.any(np.asarray(p_un['_phasecor_grid']) != 0)
    assert np.any(np.asarray(p_un['sim_mwebv']) != np.asarray(p_un['mwebv']))


def test_batched_output_matches_unbatched(tmp_path):
    """Every output is the same whatever the batch size."""
    simulator = _build_sim(tmp_path, n_sne=73)

    # the batch size is a compile product, so vary it there
    unbatched = dataclasses.replace(simulator, batch_size=100)
    key = root_key(simulator.config.run.random_seed)
    r_un, s_un = sim_run.build_sbi_pipeline(unbatched)(key)

    batched = dataclasses.replace(simulator, batch_size=20)
    r_ch, s_ch = sim_run.build_sbi_pipeline(batched)(key)

    for label, tree_un, tree_ch in [("results", r_un, r_ch), ("selection", s_un, s_ch)]:
        for k, v in tree_un.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    if not isinstance(v2, (str, list)):
                        _assert_close(v2, tree_ch[k][k2], f"{label}.{k}.{k2}")
            elif not isinstance(v, (str, list)):
                _assert_close(v, tree_ch[k], f"{label}.{k}")


def test_pipeline_fn_traces_under_jit(tmp_path):
    """The whole SBI pipeline is a function of its traced inputs, so a caller
    can fuse it into its own jit."""
    simulator = _build_sim(tmp_path, n_sne=40)
    fn = sim_run.build_sbi_pipeline(dataclasses.replace(simulator, batch_size=20))
    key = root_key(simulator.config.run.random_seed)
    r_t, s_t = jax.jit(fn)(key)
    r_e, s_e = fn(key)
    for label, t, e in (("results", r_t, r_e), ("selection", s_t, s_e)):
        for k, v in t.items():
            if isinstance(v, dict) or isinstance(v, (str, list)):
                continue
            _assert_close(v, e[k], f"{label}.{k}")


def _hostlib_pipeline(snana_fixture, n_sne=40, batch_size=20):
    """The HOSTLIB fixture's simulator and its pipeline."""
    fx = snana_fixture("hostlib_e2e")
    simulator = sim_run.prepare(str(fx.input_file), n_sne=n_sne,
                                batch_size=batch_size, random_seed=122459,
                                keep_rejected_events=True)
    return simulator, sim_run.build_sbi_pipeline(simulator)


def test_pipeline_draws_hosts_from_the_key(snana_fixture):
    """With a HOSTLIB the pipeline gives every event a host and its light;
    the same key gives the same survey, another key other galaxies and
    other noise, and neither recompiles."""
    _, pipeline_fn = _hostlib_pipeline(snana_fixture)
    r1, _ = pipeline_fn(jax.random.PRNGKey(1))
    assert np.any(np.asarray(r1['fluxgal_pe']) > 0)
    assert 'host_LOGMASS' in r1['parameters']

    logger = logging.getLogger('jax._src.interpreters.pxla')
    handler = _Records()
    with jax.log_compiles():
        logger.addHandler(handler)
        try:
            again, _ = pipeline_fn(jax.random.PRNGKey(1))
            other, _ = pipeline_fn(jax.random.PRNGKey(2))
        finally:
            logger.removeHandler(handler)
    assert not [r for r in handler.records if 'Compiling' in r.getMessage()]

    for name in ('host_galaxy_index', 'x1', 'peakmjd'):
        np.testing.assert_array_equal(np.asarray(again['parameters'][name]),
                                      np.asarray(r1['parameters'][name]))
    np.testing.assert_array_equal(np.asarray(again['fluxcals_noisy']),
                                  np.asarray(r1['fluxcals_noisy']))
    assert np.any(np.asarray(other['parameters']['host_galaxy_index'])
                  != np.asarray(r1['parameters']['host_galaxy_index']))


def test_pipeline_matches_the_ordinary_run(snana_fixture):
    """The pipeline and the ordinary run share one forward model: under the
    run's own key they give the same events, hosts and selection, and the
    same values to the last bit. XLA compiles the batch as a loop body a
    little differently from the batch alone, so a float may differ in its
    last place."""
    simulator, pipeline_fn = _hostlib_pipeline(snana_fixture)
    results, selection, _ = sim_run.run_simulation(simulator)
    r, s = pipeline_fn(root_key(simulator.config.run.random_seed))

    def same(ours, theirs, name):
        ours, theirs = np.asarray(ours), np.asarray(theirs)
        if ours.dtype.kind in 'fc':
            np.testing.assert_allclose(ours, theirs, rtol=1e-6, atol=0.0,
                                       equal_nan=True, err_msg=name)
        else:
            np.testing.assert_array_equal(ours, theirs, err_msg=name)

    for name, value in results['parameters'].items():
        if name in r['parameters']:
            same(r['parameters'][name], value, name)
    same(s['passing_mask'], selection['passing_mask'], 'passing_mask')
    n_obs = np.asarray(results['observation_schedule']['n_obs_per_sn'])
    for i in range(len(n_obs)):
        same(r['fluxcals_noisy'][i, :n_obs[i]],
             results['fluxcals_noisy'][i, :n_obs[i]], 'fluxcals_noisy')


def test_pipeline_vmaps_over_keys(snana_fixture):
    """Surveys for several keys at once are the surveys for each key."""
    _, pipeline_fn = _hostlib_pipeline(snana_fixture, n_sne=20)
    keys = jax.random.split(jax.random.PRNGKey(3), 2)
    batched, _ = jax.vmap(pipeline_fn)(keys)
    for k in range(2):
        single, _ = pipeline_fn(keys[k])
        for name in ('host_galaxy_index', 'x1', 'sim_redshift_cmb'):
            np.testing.assert_array_equal(
                np.asarray(batched['parameters'][name][k]),
                np.asarray(single['parameters'][name]), err_msg=name)
        np.testing.assert_allclose(np.asarray(batched['fluxcals_noisy'][k]),
                                   np.asarray(single['fluxcals_noisy']),
                                   rtol=1e-5, equal_nan=True)


def _compiled_constants(jaxpr):
    """Every constant the compiled functions called in ``jaxpr`` close
    over, at any depth."""
    for eqn in jaxpr.eqns:
        for param in eqn.params.values():
            for sub in (param if isinstance(param, (list, tuple)) else [param]):
                if isinstance(sub, ClosedJaxpr):
                    yield from sub.consts
                    yield from _compiled_constants(sub.jaxpr)
                elif isinstance(sub, Jaxpr):
                    yield from _compiled_constants(sub)


def test_the_run_tables_are_inputs_not_constants(snana_fixture):
    """The HOSTLIB catalogue, the SIMLIB schedule and the season bounds
    scale with the run, so the compiled batch and the pipeline take them as
    arguments; a table closed over would be copied into every compiled
    program."""
    simulator, pipeline_fn = _hostlib_pipeline(snana_fixture)
    tables = sim_run.batch_tables(simulator)
    leaves = jax.tree.leaves(tables)
    state = sim_run.initial_batch_state(simulator, simulator.n_sne)
    key = jax.random.PRNGKey(5)

    def baked(fn, *args):
        constants = list(_compiled_constants(jax.make_jaxpr(fn)(*args).jaxpr))
        return [i for i, leaf in enumerate(leaves)
                if any(c is leaf for c in constants)]

    assert baked(lambda simulator, k: sim_run.simulate_batch(
        simulator, k, 0, simulator.batch_size, state), simulator, key) == []
    assert baked(pipeline_fn, key) == []
    # The check sees a table when the simulator is deliberately closed over.
    closed = jax.jit(lambda k: sim_run.simulate_batch(
        simulator, k, 0, simulator.batch_size, state))
    assert baked(closed, key)


def test_zero_width_population_plateau(tmp_path):
    """A plateau with no Gaussian tails draws uniformly across its range."""
    inp = tmp_path / "input.input"
    _write_input(inp)
    with inp.open("a") as stream:
        stream.write("GENPEAKRANGE_SALT2ALPHA: 0.1 0.2\n"
                     "GENRANGE_SALT2ALPHA: 0.1 0.2\n")
    simulator = sim_run.prepare(str(inp), n_sne=32, batch_size=16)
    results, _ = sim_run.build_sbi_pipeline(simulator)(
        jax.random.PRNGKey(0))
    alpha = np.asarray(results['parameters']['SALT2alpha'])
    assert np.all((alpha >= .1) & (alpha <= .2))
    assert alpha.std() > .015


def test_magnitude_offset_gradients(tmp_path):
    """A magnitude offset scales every flux by 10^(-0.4 offset), with the
    matching gradient."""
    inp = tmp_path / "input.input"
    _write_input(inp)
    simulator = sim_run.prepare(str(inp), n_sne=32, batch_size=16)
    pipeline_fn = sim_run.build_sbi_pipeline(
        simulator, samplers={'GENMAG_OFF_GLOBAL': lambda key, offset: offset})
    key = jax.random.PRNGKey(0)

    def fluxes(offset):
        params = {'GENMAG_OFF_GLOBAL': {'offset': offset}}
        return pipeline_fn(key, params=params)[0]['fluxcals']

    plain = np.asarray(fluxes(0.0))
    index = np.unravel_index(np.argmax(np.where(np.isfinite(plain), plain, -np.inf)),
                             plain.shape)
    value, gradient = jax.jit(jax.value_and_grad(lambda o: fluxes(o)[index]))(0.03)
    expected = plain[index] * 10.0 ** (-0.4 * 0.03)
    np.testing.assert_allclose(value, expected, rtol=1e-5)
    np.testing.assert_allclose(gradient, -0.4 * np.log(10) * expected, rtol=1e-5)


@pytest.fixture(scope='module')
def simulator(tmp_path_factory):
    """A small, padded SALT survey with continuous shape and colour populations."""
    path = tmp_path_factory.mktemp('partial_params') / 'sim.input'
    _write_input(path)
    return sim_run.prepare(str(path), n_sne=7, batch_size=4)


@pytest.mark.parametrize('input_form', ['simulator', 'config'])
def test_exact_samples_update_derived_amplitudes(simulator, input_form):
    """Replacing colour retains placement and recomputes SALT amplitudes."""
    inputs = simulator if input_form == 'simulator' else simulator.config
    pipeline = sim_run.build_sbi_pipeline(inputs)
    key = jax.random.PRNGKey(15)
    plain = pipeline(key)[0]['parameters']
    colors = jnp.linspace(-.1, .2, simulator.n_sne)
    results, _ = pipeline(key, params={'SALT2c': colors})
    actual = results['parameters']
    np.testing.assert_array_equal(actual['c'], colors)
    for name in ('x1', 'sim_redshift_cmb'):
        np.testing.assert_array_equal(actual[name], plain[name])
    expected = compute_expected_x0_scale(actual['mu'], actual['SALT2alpha'],
                                        actual['SALT2beta'], actual['x1'], colors)
    np.testing.assert_allclose(actual['x0'], expected, rtol=1e-6)


def test_filter_names_follow_genfilters(simulator):
    """The filter indices refer to the bands named by GENFILTERS."""
    pipeline = sim_run.build_sbi_pipeline(simulator)
    assert [name[-1] for name in pipeline.filter_names] == list('ugrizY')


def test_numerical_hyperparameters_reuse_compilation_and_have_gradients(simulator):
    """The mean is traced once; colour and amplitude gradients stay consistent."""
    pipeline = sim_run.build_sbi_pipeline(simulator)
    key = jax.random.PRNGKey(15)
    traces = []

    @jax.jit
    def mean_amplitude(mean):
        traces.append(True)
        hyperparameters = {'c': {'peak': mean, 'sigma': (.05, .05)}}
        parameters = pipeline(key, params=hyperparameters)[0]['parameters']
        return parameters['x0'].mean()

    first = mean_amplitude(jnp.array(0.))
    second = mean_amplitude(jnp.array(.03))
    assert len(traces) == 1
    assert first != second
    at, step = jnp.array(.01), 1e-4
    gradient = jax.grad(mean_amplitude)(at)
    finite = (mean_amplitude(at + step) - mean_amplitude(at - step)) / (2 * step)
    assert np.isfinite(gradient)
    np.testing.assert_allclose(gradient, finite, rtol=2e-3)


def test_custom_sampler_and_exact_samples_are_batch_invariant(simulator):
    """User samplers draw scalars; the simulator owns event keys and batching."""
    def draw_color(key, center, width):
        return center + width * jax.random.uniform(key, minval=-1., maxval=1.)

    key = jax.random.PRNGKey(28)
    given = {'c': {'center': .02, 'width': .04}, 'x1': jnp.arange(7.) / 4. - 1.}
    outputs = []
    for size in (3, 7, 10):
        pipeline = sim_run.build_sbi_pipeline(dataclasses.replace(simulator, batch_size=size),
                                              samplers={'SALT2c': draw_color})
        outputs.append(pipeline(key, params=given)[0]['parameters'])
        exact = pipeline(key, params={'c': jnp.zeros(7)})[0]['parameters']['c']
        np.testing.assert_array_equal(exact, 0.)
    for parameters in outputs[1:]:
        for name in ('c', 'x1', 'x0'):
            np.testing.assert_allclose(parameters[name], outputs[0][name], rtol=1e-6)


@pytest.mark.parametrize('params, message', [
    ({'SIM_THETA': jnp.zeros(7)}, 'Unknown or unconfigured population'),
    ({'c': jnp.zeros(6)}, 'one sample per event'),
    ({'c': jnp.zeros(8)}, 'one sample per event'),
    ({'c': {'ngrid': 10}}, 'cannot be changed'),
    ({'c': {'sigma': .1}}, 'needs shape'),
    ({'c': jnp.zeros(7), 'SALT2c': jnp.zeros(7)}, 'more than once'),
    ({'GENMAG_OFF_GLOBAL': {'step': .1}}, 'per-event values or the keyword arguments'),
])
def test_invalid_parameter_forms_are_named(simulator, params, message):
    """Reject ambiguous names, wrong lengths and structural changes before compiling."""
    with pytest.raises(SNANAError, match=message):
        sim_run.build_sbi_pipeline(simulator)(jax.random.PRNGKey(0), params=params)


@pytest.mark.parametrize('functions, message', [
    ({'samplers': {'c': lambda key: 0.},
      'conditional_hyperparameters': {'c': lambda event: {}}}, 'Multiple population functions'),
    ({'conditional_hyperparameters': {'GENMAG_OFF_GLOBAL': lambda event: {}}},
     'takes a sampler'),
])
def test_invalid_population_functions_are_named(simulator, functions, message):
    """Conflicting functions are refused when building."""
    with pytest.raises(SNANAError, match=message):
        sim_run.build_sbi_pipeline(simulator, **functions)


def test_per_event_magnitude_offsets(simulator):
    """GENMAG_OFF_GLOBAL takes one value per event without a registered sampler."""
    offsets = jnp.linspace(-.1, .1, simulator.n_sne)
    results, _ = sim_run.build_sbi_pipeline(simulator)(
        jax.random.PRNGKey(0), params={'GENMAG_OFF_GLOBAL': offsets})
    np.testing.assert_allclose(results['parameters']['genmag_off'], offsets)


def test_conditional_sampler_reads_earlier_draws_and_redshift(simulator):
    """A conditional sampler reads earlier draws and the redshift from the event."""
    def draw_colour(key, event, slope):
        return slope * event['x1'] + event['sim_redshift_cmb']

    pipeline = sim_run.build_sbi_pipeline(simulator, conditional_samplers={'c': draw_colour})
    parameters = pipeline(jax.random.PRNGKey(0), params={'c': {'slope': .02}})[0]['parameters']
    np.testing.assert_allclose(parameters['c'],
                               .02 * parameters['x1'] + parameters['sim_redshift_cmb'],
                               rtol=1e-6)


def test_bayesn_theta_hyperparameters_are_differentiable(snana_fixture):
    """BayeSN samples theta before constructing its wavelength-dependent flux."""
    pipeline = sim_run.build_sbi_pipeline(
        str(snana_fixture('bayesn_w22').input_file), n_sne=4, batch_size=2)
    key = jax.random.PRNGKey(17)

    def total_flux(mean):
        results = pipeline(key, params={'THETA': {'peak': mean}})[0]
        return jnp.nansum(results['fluxcals'])

    value, gradient = jax.jit(jax.value_and_grad(total_flux))(jnp.array(.1))
    assert np.isfinite(value) and np.isfinite(gradient)
    step = .001
    numerical = (total_flux(.1 + step) - total_flux(.1 - step)) / (2 * step)
    np.testing.assert_allclose(gradient, numerical, rtol=3e-3)


@pytest.mark.parametrize('conditional', [False, True])
def test_new_hyperparameters_keep_the_configured_variation(tmp_path, conditional):
    """A configured redshift variation applies on top of new hyperparameters,
    whether they are passed directly or returned by a conditional function."""
    path = tmp_path / 'sim.input'
    _write_input(path)
    with path.open('a') as stream:
        stream.write("\nZVARIATION_POLY: GENPEAK_SALT2c 0.0,0.1\n")
    if conditional:
        pipeline = sim_run.build_sbi_pipeline(
            str(path), n_sne=4, batch_size=2,
            conditional_hyperparameters={
                'c': lambda event, mean: {'peak': mean, 'sigma': (0., 0.)}})
        given = {'mean': .02}
    else:
        pipeline = sim_run.build_sbi_pipeline(str(path), n_sne=4, batch_size=2)
        given = {'peak': .02, 'sigma': (0., 0.)}
    parameters = pipeline(jax.random.PRNGKey(0), params={'c': given})[0]['parameters']
    np.testing.assert_allclose(parameters['c'], .02 + .1 * parameters['sim_redshift_cmb'])


@pytest.mark.parametrize('conditional', [False, True])
def test_call_time_hyperparameters_update_population_weights(tmp_path, conditional):
    """Reweighting uses the same mean and width supplied for the colour draw."""
    path = tmp_path / 'sim.input'
    _write_input(path)
    with path.open('a') as stream:
        stream.write('\nGENPDF_EXPON_REWGT: 2.\n')
    simulator = sim_run.prepare(str(path), n_sne=4, batch_size=2)
    if conditional:
        pipeline = sim_run.build_sbi_pipeline(simulator, conditional_hyperparameters={
            'c': lambda event, mean, width: {'peak': mean, 'sigma': (width, width)}})
        given = {'mean': .02, 'width': .04}
    else:
        pipeline = sim_run.build_sbi_pipeline(simulator)
        given = {'peak': .02, 'sigma': (.04, .04)}
    results, _ = pipeline(jax.random.PRNGKey(9), params={'c': given})
    parameters = results['parameters']
    expected = np.exp(.5 * (np.asarray(parameters['x1'])**2
                           + ((np.asarray(parameters['c']) - .02) / .04)**2))
    np.testing.assert_allclose(parameters['wgt_population'], expected, rtol=1e-6)


@pytest.mark.parametrize('custom', [False, True])
def test_replacement_draw_rejects_configured_reweighting(tmp_path, custom):
    """A replacement draw must not silently cancel configured population weights."""
    path = tmp_path / 'sim.input'
    _write_input(path)
    with path.open('a') as stream:
        stream.write('\nGENPDF_EXPON_REWGT: 2.\n')

    def draw_color(key, center):
        return center + .01 * jax.random.normal(key)

    pipeline = sim_run.build_sbi_pipeline(
        str(path), n_sne=4, batch_size=2,
        samplers={'c': draw_color} if custom else None)
    with pytest.raises(SNANAError, match='no density for population reweighting'):
        pipeline(jax.random.PRNGKey(9),
                 params={'c': {'center': .02} if custom else jnp.full(4, .02)})


def _host_dust_input(path, snana_fixture):
    """The BayeSN test input with a HOSTLIB and host dust populations."""
    path.write_text(snana_fixture('bayesn_w22').input_file.read_text() + '''
HOSTLIB_FILE: $JAX_TEST_FIXTURE_ROOT/hostlib/JAX_TEST.HOSTLIB
HOSTLIB_MSKOPT: 258
GENRANGE_REDSHIFT: 0.05 0.20
HOSTLIB_DZTOL: 0.0,0.05,0
GENTAU_AV: 0.2
GENRANGE_AV: 0.0 5.0
GENPEAK_RV: 3.0
GENSIGMA_RV: 0.5 0.5
GENRANGE_RV: 1.2 6.0
''')
    return path


def test_conditional_functions_read_the_event(tmp_path, snana_fixture):
    """Hyperparameter functions and samplers are evaluated at each event's
    host mass, redshift and dust."""
    path = _host_dust_input(tmp_path / 'conditional.input', snana_fixture)

    def rv_hyperparameters(event, low, high):
        return {'peak': jnp.where(event['host_LOGMASS'] > 10., high, low), 'sigma': (0., 0.)}

    def draw_theta(key, event, mean, slope):
        return mean + slope * (event['host_LOGMASS'] - 10.)

    def draw_av(key, event, scale):
        return scale * event['rv']

    def magnitude(key, event, step, slope):
        return (jnp.where(event['host_LOGMASS'] > 10., -step, 0.)
                + slope * event['sim_redshift_cmb'] + event['rv'])

    pipeline = sim_run.build_sbi_pipeline(
        str(path), n_sne=8, batch_size=3,
        conditional_hyperparameters={'RV': rv_hyperparameters},
        conditional_samplers={'THETA': draw_theta, 'AV': draw_av,
                              'GENMAG_OFF_GLOBAL': magnitude})
    parameters = pipeline(jax.random.PRNGKey(23), params={
        'RV': {'low': 2., 'high': 3.},
        'THETA': {'mean': .1, 'slope': .2},
        'AV': {'scale': .1},
        'GENMAG_OFF_GLOBAL': {'step': .1, 'slope': .2}})[0]['parameters']
    mass = np.asarray(parameters['host_LOGMASS'])
    redshift = np.asarray(parameters['sim_redshift_cmb'])
    rv = np.where(mass > 10., 3., 2.)
    assert np.any(mass > 10.) and np.any(mass <= 10.)
    np.testing.assert_allclose(parameters['rv'], rv, rtol=1e-6)
    np.testing.assert_allclose(parameters['av'], .1 * rv, rtol=1e-6)
    np.testing.assert_allclose(parameters['theta'], .1 + .2 * (mass - 10.), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(parameters['genmag_off'],
                               np.where(mass > 10., -.1, 0.) + .2 * redshift + rv,
                               rtol=1e-5, atol=1e-6)


def test_conditional_arguments_vmap_and_differentiate(tmp_path, snana_fixture):
    """Keyword arguments vary under vmap without recompiling, and carry gradients."""
    path = _host_dust_input(tmp_path / 'conditional.input', snana_fixture)

    def av_hyperparameters(event, low, high):
        return {'exp_tau': jnp.where(event['host_LOGMASS'] > 10., high, low)}

    def draw_theta(key, event, mean, slope, width):
        return (mean + slope * (event['host_LOGMASS'] - 10.)
                + width * jax.random.truncated_normal(key, -3., 3.))

    pipeline = sim_run.build_sbi_pipeline(
        str(path), n_sne=8, batch_size=3,
        conditional_hyperparameters={'AV': av_hyperparameters},
        conditional_samplers={'THETA': draw_theta})
    key = jax.random.PRNGKey(23)
    traces = []

    @jax.jit
    def surveys(means):
        traces.append(True)

        def one(mean):
            parameters = pipeline(key, params={
                'AV': {'low': .3 + mean, 'high': .5 + mean},
                'THETA': {'mean': mean, 'slope': .2, 'width': .4}})[0]['parameters']
            return jnp.stack([parameters['theta'].mean(), parameters['av'].mean()])
        return jax.vmap(one)(means)

    means = jnp.linspace(-.2, .2, 16)
    first, second = surveys(means), surveys(means + .1)
    np.testing.assert_allclose((second - first)[:, 0], .1, atol=1.e-6)
    assert np.all(np.asarray(second[:, 1] > first[:, 1]))
    assert len(traces) == 1
    np.testing.assert_allclose(jax.grad(lambda x: surveys(x)[:, 0].sum())(means), 1., atol=1.e-6)
    assert np.all(np.isfinite(jax.grad(lambda x: surveys(x)[:, 1].sum())(means)))


def test_conditional_samplers_use_earlier_draws(simulator):
    """Conditional samplers computed from x1 feed the derived amplitude and the
    magnitude offset."""
    pipeline = sim_run.build_sbi_pipeline(simulator, conditional_samplers={
        'SALT2alpha': lambda key, event, slope: .14 + slope * event['x1'],
        'GENMAG_OFF_GLOBAL': lambda key, event, scale: scale * event['x1']})
    parameters = pipeline(jax.random.PRNGKey(4), params={
        'SALT2alpha': {'slope': .01}, 'GENMAG_OFF_GLOBAL': {'scale': .1}})[0]['parameters']
    np.testing.assert_allclose(parameters['SALT2alpha'], .14 + .01 * parameters['x1'])
    np.testing.assert_allclose(parameters['genmag_off'], .1 * parameters['x1'], rtol=1e-6)
    expected = compute_expected_x0_scale(parameters['mu'], parameters['SALT2alpha'],
                                        parameters['SALT2beta'], parameters['x1'], parameters['c'])
    np.testing.assert_allclose(parameters['x0'], expected, rtol=1.e-6)


def test_call_time_ranges_widen_fixed_populations(tmp_path):
    """A fixed input RV and bounded AV become new populations without rebuilding."""
    path = tmp_path / 'ranges.input'
    _write_input(path)
    path.write_text(path.read_text() + '\nGENPEAK_RV: 2.61\nGENRANGE_RV: 2.61 2.61\n'
                    'GENTAU_AV: 0.2\nGENRANGE_AV: 0.0 0.5\n')
    pipeline = sim_run.build_sbi_pipeline(str(path), n_sne=32, batch_size=13)
    traces = []
    @jax.jit
    def draw(rv_bounds, av_bounds):
        traces.append(True)
        return pipeline(jax.random.PRNGKey(23), params={
            'RV': {'peak': 3., 'sigma': (.4, .4), 'range': rv_bounds},
            'AV': {'exp_tau': .6, 'range': av_bounds}})[0]['parameters']
    first = draw(jnp.array([1.2, 6.]), jnp.array([0., 20.]))
    second = draw(jnp.array([3., 4.]), jnp.array([1., 2.]))
    fixed = draw(jnp.array([2.7, 2.7]), jnp.array([.1, .1]))
    assert len(traces) == 1
    assert np.asarray(first['rv']).std() > .1
    assert np.any(np.asarray(first['av']) > .5)
    assert np.all((second['rv'] >= 3.) & (second['rv'] <= 4.))
    assert np.all((second['av'] >= 1.) & (second['av'] <= 2.))
    np.testing.assert_array_equal(fixed['rv'], 2.7)
    np.testing.assert_array_equal(fixed['av'], .1)


def test_conditional_ranges_follow_the_mean(simulator):
    """Per-event support can move with the conditional mean and width."""
    def color_hyperparameters(event, intercept, slope, width):
        mean = intercept + slope * event['x1']
        return {'peak': mean, 'sigma': (width, width),
                'range': (mean - 3. * width, mean + 3. * width)}
    pipeline = sim_run.build_sbi_pipeline(
        simulator, conditional_hyperparameters={'c': color_hyperparameters})
    key = jax.random.PRNGKey(0)
    @jax.jit
    def draw(intercept):
        return pipeline(key, params={'c': {'intercept': intercept, 'slope': .1,
                                           'width': .02}})[0]['parameters']
    first, second = draw(jnp.array(.5)), draw(jnp.array(.6))
    mean = .5 + .1 * first['x1']
    assert np.all(np.abs(np.asarray(first['c'] - mean)) <= .06)
    np.testing.assert_allclose(second['c'] - first['c'], .1, atol=1.e-6)
    np.testing.assert_allclose(jax.grad(lambda mean: draw(mean)['c'].mean())(jnp.array(.5)), 1.)
