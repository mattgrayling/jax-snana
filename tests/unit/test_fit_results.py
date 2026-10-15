"""Unit tests for ``BatchFitResult`` postprocessing helpers."""

import jax.numpy as jnp
import numpy as np

from jax_snlc_fit.uncertainties import errors_from_chi2
from jax_snlc_fit.fit_results import BatchFitResult


def _make_result(n=4, photoz=False) -> BatchFitResult:
    return BatchFitResult(
        snids=np.arange(n).astype(str),
        params={"x1": jnp.linspace(-1.0, 1.0, n), "c": jnp.linspace(-0.1, 0.1, n)},
        param_errors={"x1": jnp.full(n, 0.1), "c": jnp.full(n, 0.05)},
        param_covariances=jnp.broadcast_to(jnp.eye(2) * 0.01, (n, 2, 2)),
        cov_output_params=["x1", "c"],
        chi_squared=jnp.linspace(5.0, 25.0, n),
        ndof=jnp.full(n, 10),
        prior_chi2=jnp.linspace(0.0, 1.0, n),
        observation_mask=jnp.ones((n, 3), dtype=bool),
        hessian_cond=jnp.linspace(2.0, 100.0, n),
        hessian_positive=jnp.ones(n, dtype=bool),
        edm=jnp.linspace(1e-3, 1.0, n),
        z_prior=((jnp.linspace(0.1, 0.4, n), jnp.full(n, 0.02))
                 if photoz else None),
        iter1_verdicts={},
    )


def test_take_keeps_every_field_in_step():
    r = _make_result(n=5, photoz=True)
    sl = r.take(np.arange(1, 4))
    assert sl.n_sne == 3
    np.testing.assert_allclose(np.asarray(sl.hessian_cond), np.asarray(r.hessian_cond[1:4]))
    np.testing.assert_allclose(np.asarray(sl.z_prior[0]), np.asarray(r.z_prior[0][1:4]))
    assert sl.param_covariances.shape == (3, 2, 2)
    assert sl.cov_output_params == ["x1", "c"]


def test_concatenate_combines_two_batches():
    a, b = _make_result(n=3), _make_result(n=4)
    out = BatchFitResult.concatenate([a, b])
    assert out.n_sne == 7
    np.testing.assert_array_equal(out.snids, np.concatenate([a.snids, b.snids]))
    np.testing.assert_allclose(
        np.asarray(out.chi_squared),
        np.concatenate([np.asarray(a.chi_squared), np.asarray(b.chi_squared)]))
    assert out.param_covariances.shape == (7, 2, 2)
    assert out.edm.shape == (7,)
    assert out.z_prior is None


def test_concatenate_joins_the_photoz_prior():
    a, b = _make_result(n=3, photoz=True), _make_result(n=2, photoz=True)
    out = BatchFitResult.concatenate([a, b])
    for i in range(2):
        np.testing.assert_allclose(
            np.asarray(out.z_prior[i]),
            np.concatenate([np.asarray(a.z_prior[i]), np.asarray(b.z_prior[i])]))


def test_fix_covar_lcfit_repairs_only_degenerate_events():
    """OPT_COVAR_LCFIT floors the eigenvalues of COV(amplitude, shape, colour)
    in magnitude space, where the amplitude axis carries d(mag)/d(amplitude)
    (snlc_fit.F90:2587). The block is every fitted parameter but the time of
    maximum and any the model defers. A healthy matrix must come back
    untouched, and a model with no amplitude must be a no-op rather than an
    error.

    Reachable in SNANA: a single-band SALT fit leaves colour unconstrained and
    trips the floor on 3 of 13 events; a six-band fit trips it on 0 of 40.
    """
    from jax_snlc_fit.uncertainties import fix_covar_lcfit

    class Model:
        t0_name = "PKMJD"
        distance_param = "x0"

        def get_fittable_parameter_names(self):
            return ["PKMJD", "x0", "x1", "c"]

        def get_fit_parameters(self):
            return {n: {} for n in self.get_fittable_parameter_names()}

        def amplitude_jacobian(self, amp):
            return -2.5 / (amp * np.log(10.0))

    n = 2
    healthy = np.diag([1e-12, 0.04, 0.0025])
    degenerate = healthy.copy()
    # shape/colour correlation past sqrt(0.04*0.0025) = 0.01, so an eigenvalue
    # of the magnitude-space matrix goes negative and the floor binds
    degenerate[1, 2] = degenerate[2, 1] = 0.012
    cov = np.zeros((n, 4, 4))
    for i, m in enumerate((healthy, degenerate)):
        cov[np.ix_([i], [1, 2, 3], [1, 2, 3])] = m

    res = BatchFitResult(
        snids=np.arange(n).astype(str),
        params={"PKMJD": jnp.zeros(n), "x0": jnp.full(n, 1e-5),
                "x1": jnp.zeros(n), "c": jnp.zeros(n)},
        param_errors={"PKMJD": jnp.full(n, 1.0), "x0": jnp.full(n, 1e-6),
                      "x1": jnp.full(n, 0.2), "c": jnp.full(n, 0.05)},
        param_covariances=jnp.asarray(cov),
        cov_output_params=["PKMJD", "x0", "x1", "c"],
        chi_squared=jnp.full(n, 20.0), ndof=jnp.full(n, 10),
        prior_chi2=jnp.zeros(n), observation_mask=jnp.ones((n, 1), dtype=bool),
        hessian_cond=jnp.ones(n),
        hessian_positive=jnp.ones(n, dtype=bool),
        edm=jnp.zeros(n),
        z_prior=None, iter1_verdicts={},
    )

    out = fix_covar_lcfit(res, Model(), opt=1)
    for name in ("x0", "x1", "c"):
        before, after = np.asarray(res.param_errors[name]), np.asarray(out.param_errors[name])
        assert np.isclose(after[0], before[0]), f"{name}: healthy event was altered"
    assert not np.allclose(np.asarray(out.param_errors["c"])[1],
                           np.asarray(res.param_errors["c"])[1]), \
        "degenerate event should have been repaired"

    class NoDistanceParam(Model):
        distance_param = None
    assert fix_covar_lcfit(res, NoDistanceParam(), opt=1) is res


def test_errors_and_edm_are_formed_on_the_free_subspace():
    """A held parameter has no error, no covariance with anything, and its
    gradient plays no part in the EDM, as MINUIT reports for a fixed one."""
    curvature = jnp.array([[4.0, 1.0, 0.5], [1.0, 9.0, 0.2], [0.5, 0.2, 1.0]])
    optimum = jnp.array([1.0, 2.0, 3.0])
    # the held parameter (index 1) sits well off its own optimum
    point = optimum + jnp.array([0.1, 0.5, -0.2])

    def chi2(q):
        return (q - optimum) @ curvature @ (q - optimum)

    errors, cov, cond, edm, positive = errors_from_chi2(
        chi2, point, jnp.ones(3), jnp.array([0, 2]))

    free = np.ix_([0, 2], [0, 2])
    expected = np.linalg.inv(np.asarray(curvature)[free])   # half the Hessian is A
    np.testing.assert_allclose(np.asarray(cov)[free], expected, rtol=1e-6)
    assert np.all(np.asarray(cov)[1, :] == 0) and np.all(np.asarray(cov)[:, 1] == 0)
    assert float(errors[1]) == 0.0
    half_gradient = np.asarray(curvature @ (point - optimum))[[0, 2]]
    np.testing.assert_allclose(float(edm), half_gradient @ expected @ half_gradient, rtol=1e-6)



def test_hessian_diagnostic():
    """A ridge or absolute errors must not hide a flat or negative direction."""
    for curvature, expected in (([2., 3.], True), ([2., -3.], False),
                                ([2., 0.], False)):
        chi2 = lambda p: jnp.sum(jnp.asarray(curvature) * p**2)
        *_, positive = errors_from_chi2(
            chi2, jnp.zeros(2), jnp.ones(2), jnp.arange(2))
        assert bool(positive) == expected
        # A held negative/flat direction does not invalidate the free fit.
        *_, positive = errors_from_chi2(
            chi2, jnp.zeros(2), jnp.ones(2), jnp.array([0]))
        assert bool(positive)


def test_take_negative_indices_preserves_matching_outcomes():
    import pandas as pd
    result = _make_result(n=2)
    result.outcomes = pd.DataFrame({'fit_index': [0, -1, 1], 'snid': ['0', 'rejected', '1']})
    selected = result.take([-1])
    assert selected.snids.tolist() == ['1']
    assert selected.outcomes.snid.tolist() == ['1']
    assert selected.outcomes.fit_index.tolist() == [0]


def test_shared_postprocessing_applies_error_floor_and_covariance_repair():
    from types import SimpleNamespace
    from jax_snlc_fit.config import FitConfig
    from jax_snlc_fit.run import errmax_bad, finish_fit
    from jax_snlc_fit.uncertainties import fix_covar_lcfit

    class Model:
        distance_param = 'x1'

        def get_fit_parameters(self):
            return {'x1': {'errmax_bad': .2, 'inistp': .3}, 'c': {}}

        def amplitude_jacobian(self, value):
            return np.ones_like(value)

    model = Model()
    fitter = SimpleNamespace(model=model, fit_parameters=model.get_fit_parameters(), held_params=())
    config = FitConfig()
    config.fit.opt_covar_lcfit = 2
    result = _make_result(2)
    result.param_covariances = np.array(result.param_covariances)
    result.param_covariances[1, 0, 1] = result.param_covariances[1, 1, 0] = .1
    expected = fix_covar_lcfit(errmax_bad(config, fitter.fit_parameters, (), result, {}), model, 2)
    actual, keep, codes, rows = finish_fit(fitter, config, None, result, {}, apply_selection=False)
    np.testing.assert_allclose(actual.param_errors['x1'][0], .3)
    assert actual.param_errors['x1'][1] > .3
    assert not np.array_equal(actual.param_covariances[1], result.param_covariances[1])
    np.testing.assert_array_equal(actual.param_covariances, expected.param_covariances)
    for name in expected.param_errors:
        np.testing.assert_array_equal(actual.param_errors[name], expected.param_errors[name])
    assert keep.all() and not codes.any() and not rows


def test_concatenate_empty_results_preserves_verdicts_in_either_order():
    result, empty = _make_result(2), _make_result(0)
    result.iter1_verdicts = {'PROB': jnp.array([True, False])}
    for parts in ([result, empty], [empty, result]):
        combined = BatchFitResult.concatenate(parts)
        np.testing.assert_array_equal(combined.iter1_verdicts['PROB'], [True, False])
    assert BatchFitResult.concatenate([empty, empty]).iter1_verdicts == {}


def test_concatenate_rejects_inconsistent_nonempty_verdicts():
    import pytest
    from jax_snlc_core.utils import SNANAError
    first, second = _make_result(2), _make_result(1)
    first.iter1_verdicts = {'PROB': jnp.array([True, False])}
    with pytest.raises(SNANAError, match='different first-round verdicts'):
        BatchFitResult.concatenate([first, second])


def test_shared_verdict_reaches_python_and_file_outputs(monkeypatch):
    from types import SimpleNamespace
    import pandas as pd
    from jax_snlc_fit.config import FitConfig
    from jax_snlc_fit.run import finish_fit, select
    from jax_snlc_fit.python import _outcomes
    from jax_snlc_fit.batch import ERRFLAG_MNFIT_NAN, ERRFLAG_FITPREP_NOEPOCHS, ERRFLAG_FIT_CUTS
    result = _make_result(5)
    result.params['x1'] = np.array([0., np.nan, np.inf, 0., 0.])
    result.observation_mask = np.array(result.observation_mask)
    result.observation_mask[3] = False
    fitter = SimpleNamespace(fit_parameters={'x1': {}, 'c': {}}, held_params=())
    monkeypatch.setattr('jax_snlc_fit.run.post_fit_cuts', lambda *a: (
        np.array([True, True, True, True, False]), np.zeros(5, dtype=int), []))
    for apply_selection in (False, True):
        actual, keep, codes, _ = finish_fit(fitter, FitConfig(), None, result, {},
                                           apply_selection=apply_selection, fitwins={})
        np.testing.assert_array_equal(keep, [True, False, False, False, not apply_selection])
        np.testing.assert_array_equal(codes, [0, ERRFLAG_MNFIT_NAN, ERRFLAG_MNFIT_NAN,
            ERRFLAG_FITPREP_NOEPOCHS, ERRFLAG_FIT_CUTS if apply_selection else 0])
        outcomes = _outcomes(pd.DataFrame({'snid': result.snids}), actual, [], keep, codes)
        np.testing.assert_array_equal(outcomes.reason, codes)
        np.testing.assert_array_equal(outcomes.status == 'accepted', keep)
        _, selected, _ = select(result, actual, None, keep, codes)
        np.testing.assert_array_equal(selected.snids, result.snids[keep])


def test_reorder_profiles():
    result = _make_result(4)
    result.minos_errors = {n: jnp.arange(8.).reshape(4, 2) for n in result.param_names}
    result.minos_status = {n: jnp.arange(8).reshape(4, 2) % 4 for n in result.param_names}
    combined = BatchFitResult.concatenate([result.take([3, 1]), result.take([0, 2])])
    order = jnp.array([3, 1, 0, 2])
    for name in result.param_names:
        np.testing.assert_array_equal(combined.minos_errors[name],
                                      result.minos_errors[name][order])
        np.testing.assert_array_equal(combined.minos_status[name],
                                      result.minos_status[name][order])


def test_minos_bad_errors():
    from jax_snlc_fit.config_parser import build
    from jax_snlc_fit.run import errmax_bad

    result = _make_result(2)
    specs = {'x1': {'errmax_bad': .2, 'inistp': .5}}
    config = build({'MINOS': 'all'})
    flagged = errmax_bad(config, specs, [], result, {})
    np.testing.assert_allclose(flagged.param_errors['x1'], .5)
    np.testing.assert_array_equal(flagged.error_flags['x1'], [6, 6])
    np.testing.assert_array_equal(flagged.param_errors['c'], result.param_errors['c'])
    legacy = errmax_bad(build({'MINOS': 'none'}), specs, [], result, {})
    np.testing.assert_allclose(legacy.param_errors['x1'], .5)


def test_minos_asymmetric_floor():
    from jax_snlc_fit.config_parser import build
    from jax_snlc_fit.run import errmax_bad

    result = _make_result(2)
    result.minos_errors = {'x1': jnp.array([[.01, .3], [.01, .02]]),
                           'c': jnp.full((2, 2), .05)}
    result.minos_status = {n: jnp.zeros((2, 2), dtype=int) for n in result.param_names}
    result.param_errors['x1'] = jnp.mean(result.minos_errors['x1'], axis=1)
    specs = {'x1': {'errmax_bad': .2, 'inistp': .5}}
    out = errmax_bad(build({'MINOS': 'all'}), specs, [], result, {})
    np.testing.assert_allclose(out.param_errors['x1'][0], .155)
    np.testing.assert_allclose(out.param_errors['x1'][1], .5)
    np.testing.assert_array_equal(out.error_flags['x1'], [1, 6])
    reordered = BatchFitResult.concatenate([out.take([1]), out.take([0])])
    np.testing.assert_array_equal(reordered.error_flags['x1'], [6, 1])
    np.testing.assert_array_equal(reordered.minos_status['x1'], [[0, 0], [0, 0]])


def test_concatenate_rejects_mixed_uncertainty_metadata():
    import pytest
    from jax_snlc_core.utils.error_handling import SNANAError

    for field in ('minos_errors', 'minos_status', 'error_flags'):
        first, second = _make_result(2), _make_result(2)
        shape = (2,) if field == 'error_flags' else (2, 2)
        setattr(second, field, {name: jnp.zeros(shape) for name in second.param_names})
        for results in ([first, second], [second, first]):
            with pytest.raises(SNANAError, match=field):
                BatchFitResult.concatenate(results)
