"""build_sbi_pipeline's pipeline_fn batches internally without changing the result."""

import dataclasses
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.extend.core import ClosedJaxpr, Jaxpr

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
    results, _, _ = sim_run.build_sbi_pipeline(simulator)(
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
    r_un, s_un, _ = sim_run.build_sbi_pipeline(unbatched)(key, params)

    batched = dataclasses.replace(simulator, batch_size=20)
    r_ch, s_ch, _ = sim_run.build_sbi_pipeline(batched)(key, params)

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
    r_t, s_t, _ = jax.jit(fn)(key, params)
    r_e, s_e, _ = fn(key, params)
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
    r1, _, _ = pipeline_fn(jax.random.PRNGKey(1))
    assert np.any(np.asarray(r1['fluxgal_pe']) > 0)
    assert 'host_LOGMASS' in r1['parameters']

    logger = logging.getLogger('jax._src.interpreters.pxla')
    handler = _Records()
    with jax.log_compiles():
        logger.addHandler(handler)
        try:
            again, _, _ = pipeline_fn(jax.random.PRNGKey(1))
            other, _, _ = pipeline_fn(jax.random.PRNGKey(2))
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
    r, s, _ = pipeline_fn(root_key(simulator.config.run.random_seed))

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
    batched, _, _ = jax.vmap(pipeline_fn)(keys)
    for k in range(2):
        single, _, _ = pipeline_fn(keys[k])
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


def test_adjust_sees_the_hosts_before_the_photometry(snana_fixture):
    """``adjust`` gets each batch's parameters with its hosts, and what it
    returns is what the light curves are simulated from."""
    _, pipeline_fn = _hostlib_pipeline(snana_fixture)
    key = jax.random.PRNGKey(4)

    def mass_step(key, params):
        step = jnp.where(params['host_LOGMASS'] > 10.0, 0.5, 0.0)
        return {**params, 'magoff': params['magoff'] + step}

    plain, _, _ = pipeline_fn(key)
    stepped, _, _ = pipeline_fn(key, adjust=mass_step)
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
    results, _, _ = sim_run.build_sbi_pipeline(simulator)(
        jax.random.PRNGKey(0))
    alpha = np.asarray(results['parameters']['SALT2alpha'])
    assert np.all((alpha >= .1) & (alpha <= .2))
    assert alpha.std() > .015


def test_adjust_flux_gradients_with_padded_epochs(tmp_path):
    """Offsets remain differentiable at zero, including when they cancel."""
    inp = tmp_path / "input.input"
    _write_input(inp)
    cfg = dict(n_sne=32, batch_size=16)
    simulator = sim_run.prepare(str(inp), **cfg)
    pipeline_fn = sim_run.build_sbi_pipeline(simulator)
    key = jax.random.PRNGKey(0)

    def adjusted(offsets):
        def adjust(key, params):
            return {**params,
                    'magoff': jnp.full_like(params['magoff'], offsets[0]),
                    'lensdmu': jnp.full_like(params['lensdmu'], offsets[1]),
                    'snmagshift': jnp.full_like(params['magoff'], offsets[2]),
                    'magsmear': jnp.zeros_like(params['magsmear']),
                    'mushift': jnp.zeros_like(params['mushift'])}

        return pipeline_fn(key, adjust=adjust)[0]

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


def test_finish_batch_refuses_what_the_ordinary_run_would(snana_fixture):
    """The pipeline does not abort; finish_batch raises on its results and
    record what the ordinary run raises, here an absurd flux."""
    simulator, pipeline_fn = _hostlib_pipeline(snana_fixture)

    def absurd(key, params):
        return {**params, 'x0': params['x0'] * 1.0e8}

    results, _, record = pipeline_fn(jax.random.PRNGKey(5))
    sim_run.finish_batch(simulator, results, record, 0, simulator.n_sne)
    results, _, record = pipeline_fn(jax.random.PRNGKey(5), adjust=absurd)
    with pytest.raises(SNANAError, match="Crazy flux"):
        sim_run.finish_batch(simulator, results, record, 0, simulator.n_sne)
