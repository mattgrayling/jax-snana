"""The fit bounds come from the parameter tables, not from hardcoded literals,
and the covariance survives a parameter the likelihood ignores."""

import logging

import numpy as np
import pytest

from jax_salt2mu.fitter import _fit_covariance, _nuisance_errors, fit_bounds, next_sigint
from jax_snlc_core.models import SALTModel
from jax_snlc_core.utils.error_handling import SNANAError


def test_joint_sigint_bounds_from_config():
    lo, hi = fit_bounds(SALTModel, ["alpha0"], 3, -30.0, "joint", 0.05, 1.5)
    # sigint is the last floated parameter in joint mode
    assert (float(lo[-1]), float(hi[-1])) == (0.05, 1.5)
    # secant mode has no sigint entry at all
    lo2, _ = fit_bounds(SALTModel, ["alpha0"], 3, -30.0, "snana_secant", 0.05, 1.5)
    assert len(lo2) == len(lo) - 1


def test_bbc_own_parameters_take_their_bounds_from_the_shared_table():
    """scalePCC and the H11 coefficients are declared like a model's nuisance
    parameters, so fit_bounds needs no branch for them (prep_fitpar,
    SALT2mu.c:22368, :22373)."""
    lo, hi = fit_bounds(SALTModel, ["alpha0", "scalePCC", "H11mucc2"], 0, -30.0,
                        "snana_secant", 0.05, 1.5)
    assert (float(lo[1]), float(hi[1])) == (-0.10, 15.0)
    assert (float(lo[2]), float(hi[2])) == (-30.0, 30.0)


def test_next_sigint_aborts_on_a_runaway_trial():
    """next_covFitPar aborts above 100 rather than running the next inner fit at
    an arbitrary sigma_int (SALT2mu.c:23351). A near-flat slope is the documented
    degenerate regime that produces one."""
    # chi2red moved 1e-6 over a 0.01 step: slope 1e-4, so the secant asks for
    # sigint - 0.5/1e-4
    with pytest.raises(SNANAError, match="Crazy parval_next"):
        next_sigint([1.500001, 1.5], [0.10, 0.11], 0.01)
    # the same history with a slope that lands in range returns normally
    nxt, _ = next_sigint([2.0, 1.5], [0.10, 0.11], 0.01)
    assert 0.0 <= nxt <= 100.0


def test_flat_parameter_preserves_other_errors(caplog):
    """A floated parameter the likelihood does not depend on (scalePCC with a
    0/1 classifier) has an all-zero Hessian row. Its error is undefined and
    named; the others come from the inverse without it, as MINUIT carries on
    without a flat direction, rather than every error going NaN."""
    nf, nzbin = 3, 2
    H = np.diag([8.0, 0.0, 2.0, 4.0, 4.0])
    H[0, 2] = H[2, 0] = 1.0
    nerr, m0err, cov_m0 = _fit_covariance(H, nf, nzbin, np.array([True, True]), "secant")
    curved = np.ix_([0, 2, 3, 4], [0, 2, 3, 4])
    expect = np.sqrt(np.diag(2.0 * np.linalg.inv(H[curved])))
    np.testing.assert_allclose(nerr[[0, 2]], expect[:2])
    assert np.isnan(nerr[1])
    np.testing.assert_allclose(m0err, expect[2:])
    np.testing.assert_allclose(np.diag(cov_m0), expect[2:] ** 2)
    with caplog.at_level(logging.WARNING):
        errs = _nuisance_errors(nerr, ["alpha0", "scalePCC", "beta0"])
    assert np.isnan(errs["scalePCC"]) and np.isfinite(errs["alpha0"])
    assert "does not depend on scalePCC" in caplog.text


@pytest.mark.parametrize('hessian', [
    np.diag([-2., 2.]),
    2 * np.linalg.inv(np.array([[1., 2.], [2., 1.]])),
    np.ones((2, 2)),
    np.diag([np.nan, 2.]),
    np.diag([np.inf, 2.]),
])
def test_invalid_curvature_rejects_uncertainties(hessian):
    with pytest.raises(SNANAError, match='Hessian'):
        _fit_covariance(hessian, 1, 1, np.array([True]), 'secant')


def test_unfloated_bin_does_not_require_positive_curvature():
    nerr, m0err, cov = _fit_covariance(
        np.diag([2., 4., 0.]), 1, 2, np.array([True, False]), 'secant')
    np.testing.assert_allclose(nerr, [1.])
    np.testing.assert_allclose(m0err[0], np.sqrt(.5))
    assert np.isnan(m0err[1])
    np.testing.assert_array_equal(cov, [[.5, 0.], [0., 1.e5]])


def test_joint_covariance_retains_scatter_correlations():
    hessian = np.array([[4., 1., .5], [1., 3., .8], [.5, .8, 2.]])
    expected = 2 * np.linalg.inv(hessian)
    nerr, m0err, cov = _fit_covariance(
        hessian, 1, 1, np.array([True]), 'joint')
    np.testing.assert_allclose(nerr, np.sqrt(expected.diagonal()[:1]))
    np.testing.assert_allclose(m0err, np.sqrt(expected.diagonal()[1:2]))
    np.testing.assert_allclose(cov, expected[1:2, 1:2])


def test_all_flat_parameters_have_undefined_errors():
    nerr, m0err, cov = _fit_covariance(
        np.zeros((2, 2)), 1, 1, np.array([True]), 'secant')
    assert np.isnan(nerr).all()
    assert np.isnan(m0err).all()
    assert np.isnan(cov).all()


def test_projected_gradient_only_removes_blocked_descent():
    from jax_salt2mu.fitter import _gradient_diagnostics
    # Outward descent at lower/upper bounds; inward descent at the same bounds;
    # a free parameter; and a parameter fixed by equal bounds.
    p = np.array([0., 1., 0., 1., .5, .5])
    gradient = np.array([100., -200., -3., 4., 2., 500.])
    lo = np.array([0., 0., 0., 0., 0., .5])
    hi = np.array([1., 1., 1., 1., 1., .5])
    diagnostics = _gradient_diagnostics(p, gradient, (lo, hi))
    assert diagnostics == dict(max_gradient=500., max_projected_gradient=4.)
