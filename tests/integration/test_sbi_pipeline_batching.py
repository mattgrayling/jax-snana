"""build_sbi_pipeline's pipeline_fn batches internally without changing the result."""

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
from jax_snlc_core.snana_files import MAG_ZEROFLUX
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


def _build_sim_and_params(tmp_path: Path, n_sne: int):
    """The simulator and a survey's own parameters, to supply back to the
    pipeline as a caller's."""
    inp = tmp_path / "input.input"
    _write_input(inp)
    simulator = sim_run.prepare(str(inp), n_sne=n_sne, batch_size=n_sne,
                                random_seed=12345)
    results, _ = sim_run.build_sbi_pipeline(simulator)(
        root_key(simulator.config.run.random_seed))
    return simulator, results['parameters']


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
    simulator, params = _build_sim_and_params(tmp_path, n_sne=73)

    # the batch size is a compile product, so vary it there
    unbatched = dataclasses.replace(simulator, batch_size=100)
    key = root_key(simulator.config.run.random_seed)
    r_un, s_un = sim_run.build_sbi_pipeline(unbatched)(key, events=params)

    batched = dataclasses.replace(simulator, batch_size=20)
    r_ch, s_ch = sim_run.build_sbi_pipeline(batched)(key, events=params)

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
    simulator, params = _build_sim_and_params(tmp_path, n_sne=40)
    fn = sim_run.build_sbi_pipeline(dataclasses.replace(simulator, batch_size=20))
    key = root_key(simulator.config.run.random_seed)
    r_t, s_t = jax.jit(fn)(key, events=params)
    r_e, s_e = fn(key, events=params)
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


def test_complete_events_preserve_host_dependent_offsets(snana_fixture):
    """Complete events retain their supplied hosts and magnitude offsets."""
    _, pipeline_fn = _hostlib_pipeline(snana_fixture)
    key = jax.random.PRNGKey(4)

    generated, _ = pipeline_fn(key)
    params = generated['parameters']
    plain, _ = pipeline_fn(key, events=params)
    step = jnp.where(params['host_LOGMASS'] > 10.0, 0.5, 0.0)
    stepped, _ = pipeline_fn(key, events={**params, 'magoff': params['magoff'] + step})
    massive = np.asarray(plain['parameters']['host_LOGMASS']) > 10.0
    assert massive.any() and (~massive).any()
    np.testing.assert_allclose(
        np.asarray(stepped['parameters']['magoff'])
        - np.asarray(plain['parameters']['magoff']),
        np.where(massive, 0.5, 0.0), atol=1e-6)
    dmag = (np.asarray(stepped['magnitudes'])
            - np.asarray(plain['magnitudes']))
    # an epoch without flux carries the null magnitude in both runs
    finite = np.isfinite(dmag) & (np.asarray(plain['magnitudes'])
                                  != MAG_ZEROFLUX)
    np.testing.assert_allclose(dmag[finite & massive[:, None]], 0.5, atol=1e-4)
    np.testing.assert_allclose(dmag[finite & ~massive[:, None]], 0.0, atol=1e-4)


def test_zero_width_population_plateau(tmp_path):
    inp = tmp_path / "input.input"
    _write_input(inp)
    with inp.open("a") as stream:
        stream.write("GENPEAKRANGE_SALT2ALPHA: 0.1 0.2\n"
                     "GENRANGE_SALT2ALPHA: 0.1 0.2\n")
    cfg = dict(n_sne=32, batch_size=16)
    simulator = sim_run.prepare(str(inp), **cfg)
    results, _ = sim_run.build_sbi_pipeline(simulator)(
        jax.random.PRNGKey(0))
    alpha = np.asarray(results['parameters']['SALT2alpha'])
    assert np.all((alpha >= .1) & (alpha <= .2))
    assert alpha.std() > .015


def test_complete_event_flux_gradients_with_padded_epochs(tmp_path):
    """Offsets remain differentiable at zero, including when they cancel."""
    inp = tmp_path / "input.input"
    _write_input(inp)
    cfg = dict(n_sne=32, batch_size=16)
    simulator = sim_run.prepare(str(inp), **cfg)
    pipeline_fn = sim_run.build_sbi_pipeline(simulator)
    key = jax.random.PRNGKey(0)

    events = pipeline_fn(key)[0]['parameters']

    def adjusted(offsets):
        supplied = {**events,
                    'magoff': jnp.full_like(events['magoff'], offsets[0]),
                    'lensdmu': jnp.full_like(events['lensdmu'], offsets[1]),
                    'snmagshift': jnp.full_like(events['magoff'], offsets[2]),
                    'magsmear': jnp.zeros_like(events['magsmear']),
                    'mushift': jnp.zeros_like(events['mushift'])}
        return pipeline_fn(key, events=supplied)[0]

    plain = adjusted(jnp.zeros(3))
    flux = np.asarray(plain['fluxcals'])
    magnitude = np.asarray(plain['magnitudes'])
    padded = np.asarray(plain['observation_schedule']['filter_indices']) < 0
    candidates = (np.isfinite(flux) & (magnitude > -1) & (magnitude < 49)
                  & padded.any(axis=1)[:, None])
    assert candidates.any()
    index = np.unravel_index(np.argmax(np.where(candidates, flux, -np.inf)),
                            flux.shape)
    peak_magnitudes = np.asarray(plain['peakmags'])[index[0]]
    valid_peak = (peak_magnitudes > -1) & (peak_magnitudes < 49)
    assert valid_peak.any()
    peak_index = (index[0], int(np.flatnonzero(valid_peak)[0]))

    def observables(offsets):
        results = adjusted(offsets)
        return jnp.stack([results['fluxcals'][index],
                          results['magnitudes'][index],
                          results['peakmags'][peak_index],
                          results['fluxcals_noisy'][index],
                          results['fluxcalerrs'][index],
                          results['snrs'][index]])

    @jax.jit
    def values_and_gradients(offsets):
        return observables(offsets), jax.jacrev(observables)(offsets)

    for offsets in ([0., 0., 0.], [0.03, 0.01, 0.], [-0.03, -0.01, 0.],
                    [0.03, -0.03, 0.], [0., 0., 0.02], [0., 0., -0.02],
                    [0.03, 0., -0.03]):
        value, gradient = values_and_gradients(jnp.array(offsets))
        total = sum(offsets)
        expected = [flux[index] * 10.0 ** (-0.4 * total),
                    magnitude[index] + total, peak_magnitudes[peak_index[1]] + total]
        np.testing.assert_allclose(value[:3], expected, rtol=1e-5,
                                   err_msg=f"offsets={offsets}")
        expected_gradient = np.ones((3, 3))
        expected_gradient[0] = -0.4 * np.log(10) * expected[0]
        np.testing.assert_allclose(gradient[:3], expected_gradient, rtol=1e-5,
                                   err_msg=f"offsets={offsets}")
        step = 1e-4
        shift = jnp.array([step, 0., 0.])
        finite_difference = (observables(jnp.array(offsets) + shift)
                             - observables(jnp.array(offsets) - shift)) / (2 * step)
        np.testing.assert_allclose(gradient[3:, 0], finite_difference[3:], rtol=1e-5)


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


def test_numerical_settings_reuse_compilation_and_have_gradients(simulator):
    """The mean is traced once; colour and amplitude gradients stay consistent."""
    pipeline = sim_run.build_sbi_pipeline(simulator)
    key = jax.random.PRNGKey(15)
    traces = []

    @jax.jit
    def mean_amplitude(mean):
        traces.append(True)
        parameters = pipeline(key, params={'c': {'peak': mean, 'sigma': (.05, .05)}})[0]['parameters']
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
    settings = {'c': {'center': .02, 'width': .04}, 'x1': jnp.arange(7.) / 4. - 1.}
    outputs = []
    for size in (3, 7, 10):
        pipeline = sim_run.build_sbi_pipeline(dataclasses.replace(simulator, batch_size=size),
                                              samplers={'SALT2c': draw_color})
        outputs.append(pipeline(key, params=settings)[0]['parameters'])
        exact = pipeline(key, params={'c': jnp.zeros(7)})[0]['parameters']['c']
        np.testing.assert_array_equal(exact, 0.)
    for parameters in outputs[1:]:
        for name in ('c', 'x1', 'x0'):
            np.testing.assert_allclose(parameters[name], outputs[0][name], rtol=1e-6)


@pytest.mark.parametrize('params, message', [
    ({'SIM_THETA': jnp.zeros(7)}, 'Unknown or unconfigured population'),
    ({'c': jnp.zeros(6)}, 'one sample per event'),
    ({'c': jnp.zeros(8)}, 'one sample per event'),
    ({'c': {'ngrid': 10}}, 'unsupported'),
    ({'c': {'sigma': .1}}, 'needs shape'),
    ({'c': jnp.zeros(7), 'SALT2c': jnp.zeros(7)}, 'more than once'),
])
def test_invalid_parameter_forms_are_named(simulator, params, message):
    """Reject ambiguous names, wrong lengths and structural changes before compiling."""
    with pytest.raises(SNANAError, match=message):
        sim_run.build_sbi_pipeline(simulator)(jax.random.PRNGKey(0), params=params)


def test_bayesn_theta_settings_are_differentiable(snana_fixture):
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


def test_call_time_mean_includes_redshift_variation(tmp_path):
    """The call-time mean and configured redshift shift compose at the draw."""
    path = tmp_path / 'sim.input'
    _write_input(path)
    with path.open('a') as stream:
        stream.write("\nZVARIATION_POLY: GENPEAK_SALT2c 0.0,0.1\n")
    pipeline = sim_run.build_sbi_pipeline(str(path), n_sne=4, batch_size=2)
    results, _ = pipeline(jax.random.PRNGKey(0),
                                  params={'c': {'peak': .02, 'sigma': (0., 0.)}})
    parameters = results['parameters']
    np.testing.assert_allclose(parameters['c'], .02 + .1 * parameters['sim_redshift_cmb'])


def test_exact_samples_replace_conditional_genpdf(tmp_path):
    """Exact samples replace a prepared conditional population draw."""
    from tests.integration.test_genpdf_wgtmap_hostfirst import (
        _write_input as write_host_input,
        _write_logmass_only_wgtmap,
        _write_genpdf_c_logmass,
    )

    genpdf = tmp_path / 'color.dat'
    wgtmap = tmp_path / 'host.wgtmap'
    path = tmp_path / 'sim.input'
    _write_genpdf_c_logmass(genpdf)
    _write_logmass_only_wgtmap(wgtmap)
    write_host_input(path, hostlib=None, wgtmap=wgtmap, genpdf=genpdf)
    simulator = sim_run.prepare(str(path), n_sne=40, batch_size=20)
    pipeline = sim_run.build_sbi_pipeline(simulator)
    key = jax.random.PRNGKey(0)
    results, _ = pipeline(key, params={'c': jnp.zeros(40)})
    np.testing.assert_array_equal(results['parameters']['c'], 0.)


@pytest.mark.parametrize('conditional', [False, True])
def test_call_time_settings_update_population_weights(tmp_path, conditional):
    """Reweighting uses the same mean and width supplied for the colour draw."""
    path = tmp_path / 'sim.input'
    _write_input(path)
    with path.open('a') as stream:
        stream.write('\nGENPDF_EXPON_REWGT: 2.\n')
    simulator = sim_run.prepare(str(path), n_sne=4, batch_size=2)
    if conditional:
        pipeline = sim_run.build_sbi_pipeline(simulator, conditional_settings={
            'c': lambda mean, width: {'peak': mean, 'sigma': (width, width)}})
        settings = {'mean': .02, 'width': .04}
    else:
        pipeline = sim_run.build_sbi_pipeline(simulator)
        settings = {'peak': .02, 'sigma': (.04, .04)}
    results, _ = pipeline(jax.random.PRNGKey(9), params={'c': settings})
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


def test_conditional_populations_compile_and_batch(tmp_path, snana_fixture):
    """Host-conditioned dust, relative-support draws and offsets compose under vmap."""
    path = tmp_path / 'conditional.input'
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
GENSIGMA_VPEC: 300
''')

    def av_settings(mass, low, high):
        return {'exp_tau': jnp.where(mass > 10., high, low)}

    def rv_settings(mass, low, high, width):
        return {'peak': jnp.where(mass > 10., high, low), 'sigma': (width, width)}

    def draw_theta(key, mass, mean, slope, width):
        return mean + slope * (mass - 10.) + width * jax.random.truncated_normal(key, -3., 3.)

    def magnitude(mass, zhel, offset, step, slope):
        return offset + jnp.where(mass > 10., .5, -.5) * (step + slope * zhel)

    options = dict(
        conditional_settings={'AV': (av_settings, ['HOST.LOGMASS']),
                              'RV': (rv_settings, ['HOST.LOGMASS'])},
        samplers={'THETA': (draw_theta, ['HOST.LOGMASS'])},
        relations={'GENMAG_OFF_GLOBAL': (magnitude, ['HOST.LOGMASS', 'REDSHIFT_HELIO'])})
    coefficients = {'AV': {'low': .1, 'high': .3},
                    'RV': {'low': 1.1, 'high': 3., 'width': .001},
                    'THETA': {'mean': .1, 'slope': .2, 'width': .4},
                    'GENMAG_OFF_GLOBAL': {'offset': .03, 'step': .1, 'slope': .2}}
    pipeline = sim_run.build_sbi_pipeline(str(path), n_sne=8, batch_size=3, **options)
    key = jax.random.PRNGKey(23)
    parameters = pipeline(key, params=coefficients)[0]['parameters']
    mass = parameters['host_LOGMASS']
    assert np.any(mass > 10.) and np.any(mass <= 10.)
    assert np.all(np.asarray(parameters['rv']) >= 1.2)
    np.testing.assert_allclose(parameters['genmag_off'],
                               magnitude(mass, parameters['sim_redshift_helio'], **coefficients['GENMAG_OFF_GLOBAL']))
    theta_mean = .1 + .2 * (mass - 10.)
    assert np.all(np.abs(np.asarray(parameters['theta'] - theta_mean)) <= 1.2)
    for name in ('av', 'rv'):
        np.testing.assert_array_equal(parameters[name], parameters['sim_' + name])

    unbatched = sim_run.build_sbi_pipeline(str(path), n_sne=8, batch_size=8, **options)
    other = unbatched(key, params=coefficients)[0]['parameters']
    for name in ('av', 'rv', 'theta', 'genmag_off'):
        np.testing.assert_allclose(parameters[name], other[name], rtol=1.e-6)

    traces = []
    @jax.jit
    def surveys(means):
        traces.append(True)
        def one(mean):
            supplied = {**coefficients,
                        'THETA': {**coefficients['THETA'], 'mean': mean},
                        'AV': {'low': .3 + mean, 'high': .5 + mean},
                        'GENMAG_OFF_GLOBAL': {**coefficients['GENMAG_OFF_GLOBAL'], 'offset': mean}}
            generated = pipeline(key, params=supplied)[0]['parameters']
            return jnp.stack([generated[name].mean() for name in ('theta', 'av', 'genmag_off')])
        return jax.vmap(one)(means)
    means = jnp.linspace(-.2, .2, 16)
    first, second = surveys(means), surveys(means + .1)
    np.testing.assert_allclose((second - first)[:, (0, 2)], .1, atol=1.e-6)
    assert np.all(np.asarray(second[:, 1] > first[:, 1]))
    assert len(traces) == 1
    np.testing.assert_allclose(jax.grad(lambda x: surveys(x)[:, 0].sum())(means), 1., atol=1.e-6)
    assert np.all(np.isfinite(jax.grad(lambda x: surveys(x)[:, 1].sum())(means)))


def test_conditional_settings_conflict_with_variations(tmp_path):
    """A conditional field cannot silently accumulate a configured variation."""
    path = tmp_path / 'variation.input'
    _write_input(path)
    path.write_text(path.read_text() + '\nZVARIATION_POLY: GENPEAK_SALT2c 0.0,0.1\n')
    pipeline = sim_run.build_sbi_pipeline(
        str(path), n_sne=4, batch_size=2,
        conditional_settings={'c': lambda mean: {'peak': mean}})
    with pytest.raises(SNANAError, match='both set c.peak'):
        pipeline(jax.random.PRNGKey(0), params={'c': {'mean': .02}})


def test_population_relations_use_earlier_draws(simulator):
    """A deterministic model parameter is computed before derived amplitudes."""
    pipeline = sim_run.build_sbi_pipeline(
        simulator, relations={'SALT2alpha': (lambda x1, slope: .14 + slope * x1, ['x1'])})
    parameters = pipeline(jax.random.PRNGKey(4), params={'SALT2alpha': {'slope': .01}})[0]['parameters']
    np.testing.assert_allclose(parameters['SALT2alpha'], .14 + .01 * parameters['x1'])
    expected = compute_expected_x0_scale(parameters['mu'], parameters['SALT2alpha'],
                                        parameters['SALT2beta'], parameters['x1'], parameters['c'])
    np.testing.assert_allclose(parameters['x0'], expected, rtol=1.e-6)


def test_conditional_settings_keep_other_variations(tmp_path):
    """Replacing the width leaves the independently configured mean variation."""
    path = tmp_path / 'variation.input'
    _write_input(path)
    path.write_text(path.read_text() + '\nZVARIATION_POLY: GENPEAK_SALT2c 0.0,0.1\n')
    pipeline = sim_run.build_sbi_pipeline(
        str(path), n_sne=4, batch_size=2,
        conditional_settings={'c': lambda width: {'sigma': (width, width)}})
    parameters = pipeline(jax.random.PRNGKey(0), params={'c': {'width': 0.}})[0]['parameters']
    np.testing.assert_allclose(parameters['c'], .1 * parameters['sim_redshift_cmb'])


def test_population_dependencies_require_existing_order(simulator):
    """Dependencies cannot refer to a later draw or an absent host catalogue."""
    with pytest.raises(SNANAError, match='not drawn before'):
        sim_run.build_sbi_pipeline(simulator, relations={'x1': (lambda color: color, ['c'])})
    with pytest.raises(SNANAError, match='unavailable host column'):
        sim_run.build_sbi_pipeline(
            simulator, relations={'x1': (lambda mass: mass - 10., ['HOST.LOGMASS'])})


def test_population_dependencies_preserve_precision(tmp_path):
    """Dust and magnitude relations see the exact supplied model parameters."""
    path = tmp_path / 'precision.input'
    _write_input(path)
    path.write_text(path.read_text() + "\nGENPEAK_RV: 3.1\nGENRANGE_RV: 1.2 6.0\n"
                    "GENSIGMA_RV: 0.2 0.2\nGENTAU_AV: 0.2\nGENRANGE_AV: 0.0 5.0\n")
    pipeline = sim_run.build_sbi_pipeline(
        str(path), n_sne=4, batch_size=2,
        relations={'RV': (lambda x1: x1 + 2., ['x1']),
                   'GENMAG_OFF_GLOBAL': (lambda x1: x1, ['x1'])})
    values = jnp.full(4, 1.00000001, dtype=jnp.float64)
    parameters = jax.jit(pipeline)(jax.random.PRNGKey(0), params={'x1': values})[0]['parameters']
    assert parameters['rv'].dtype == values.dtype
    assert parameters['genmag_off'].dtype == values.dtype
    np.testing.assert_array_equal(parameters['rv'], values + 2.)
    np.testing.assert_array_equal(parameters['genmag_off'], values)


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
    def color_settings(shape, intercept, slope, width):
        mean = intercept + slope * shape
        return {'peak': mean, 'sigma': (width, width),
                'range': (mean - 3. * width, mean + 3. * width)}
    pipeline = sim_run.build_sbi_pipeline(
        simulator, conditional_settings={'c': (color_settings, ['x1'])})
    key = jax.random.PRNGKey(0)
    @jax.jit
    def draw(intercept):
        return pipeline(key, params={'c': {'intercept': intercept, 'slope': .1, 'width': .02}})[0]['parameters']
    first, second = draw(jnp.array(.5)), draw(jnp.array(.6))
    mean = .5 + .1 * first['x1']
    assert np.all(np.abs(np.asarray(first['c'] - mean)) <= .06)
    np.testing.assert_allclose(second['c'] - first['c'], .1, atol=1.e-6)
    np.testing.assert_allclose(jax.grad(lambda mean: draw(mean)['c'].mean())(jnp.array(.5)), 1.)
