"""Likelihood normalization, posterior weights and population scatter."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_salt2mu.ccprior import H11PriorInputs, CHI2_BEAMS_NOPROB
from jax_salt2mu.config import BBCConfig
from jax_salt2mu.map_kernels import estimate_intrinsic_scatter
from jax_salt2mu.fitter import FitSpec, _chi2_ia
from jax_salt2mu.likelihood import LikelihoodInputs, event_chi2, make_loss
from jax_salt2mu.output import _muerr_renorm
from jax_salt2mu.run import _chi2max_keep
from jax_snlc_core.models import SALTModel


def _inputs(residuals):
    residuals = jnp.asarray(residuals)
    n = len(residuals)
    return LikelihoodInputs(
        lcpar={"mB": 40.0 + residuals, "x1": jnp.zeros(n), "c": jnp.zeros(n)},
        z=jnp.full(n, .1), logmass=jnp.full(n, 10.),
        covmat_fit=jnp.tile(jnp.diag(jnp.array([.01, 0., 0.])), (n, 1, 1)),
        mumodel=jnp.full(n, 40.), izbin=jnp.zeros(n, dtype=int),
        muerr_vpec_sq=jnp.zeros(n), valid=jnp.ones(n),
        muerrsq_last=jnp.full(n, .01),
        h11prior=H11PriorInputs(jnp.full(n, .1), jnp.full(n, .01), jnp.ones(n)))


def _spec(mode="joint", with_log=False):
    nuisance = {name: values[0] for name, values in SALTModel.nuisance_spec.items()}
    nuisance.update(scalePCC=1., H11mucc0=0., H11mucc1=0., H11mucc2=0.,
                    H11sigcc0=1., H11sigcc1=0., H11sigcc2=0.)
    return FitSpec(SALTModel, (), tuple(nuisance.items()), 1, mode, with_log)


@pytest.mark.parametrize("with_log", [False, True])
def test_joint_ia_limit(with_log):
    inputs, spec = _inputs([.2, -.2]), _spec(with_log=with_log)
    pure = make_loss(spec, inputs._replace(h11prior=None))
    beams = make_loss(spec, inputs)
    for sigint in (.1, .3, 1., 2.):
        point = jnp.array([0., sigint])
        # BEAMS's constant probability scaling must not alter the fit.
        assert float(beams(point) - pure(point)) == pytest.approx(-4 * np.log(.15))
        np.testing.assert_allclose(jax.grad(beams)(point), jax.grad(pure)(point), atol=1e-12)
        np.testing.assert_allclose(jax.hessian(beams)(point), jax.hessian(pure)(point), atol=1e-10)


@pytest.mark.parametrize("mode", ["snana_secant", "joint"])
def test_zero_density_weights(mode):
    inputs, spec = _inputs([4., .05]), _spec(mode)
    nuisance = dict(spec.defaults)
    probcc, chi2 = event_chi2(spec, inputs, nuisance, jnp.array([4., .05]),
                             jnp.full(2, .01), inputs.muerrsq_last)
    np.testing.assert_array_equal(probcc, [0., 0.])
    assert chi2[0] == CHI2_BEAMS_NOPROB
    total, dof = _chi2_ia(spec, jnp.array([0.]), inputs, 0., 0)
    assert float(total) == pytest.approx(.25)
    assert float(dof) == 1.
    loss = make_loss(spec, inputs)
    objective = (loss if mode == "joint"
                 else lambda p: loss(p, 0., inputs.muerrsq_last))
    point = jnp.array([0., 0.]) if mode == "joint" else jnp.array([0.])
    assert np.isfinite(objective(point))
    assert np.all(np.isfinite(jax.grad(objective)(point)))
    assert np.all(np.isfinite(jax.hessian(objective)(point)))


def test_zero_density_cut_and_errors():
    inputs, spec = _inputs([4., .05]), _spec("snana_secant")
    cfg = BBCConfig(nuisance_val=dict(spec.defaults), sigint=0.)
    keep, _ = _chi2max_keep(spec, inputs, cfg, 0., np.full(2, 100.))
    np.testing.assert_array_equal(keep, [False, True])
    errors = _muerr_renorm(np.zeros(3), np.array([CHI2_BEAMS_NOPROB, 0., CHI2_BEAMS_NOPROB]),
                           np.full(3, .1), np.array([0, 0, 1]), np.array([.2, .2]), 2)
    # The rejected object contributes zero weight; so does an entirely invalid bin.
    assert np.isinf(errors[[0, 2]]).all()
    assert errors[1] == pytest.approx(.2)
    assert np.sum(1 / errors[:2] ** 2) == pytest.approx(1 / .2**2)


def test_population_weighted_scatter():
    residuals = np.linspace(-1, 1, 100)
    weights = np.full(100, .001)
    weights[0], weights[-1] = 1., .2
    variance = np.average((residuals - np.average(residuals, weights=weights))**2,
                          weights=weights)
    expected = np.sqrt(variance - .01)
    result, _, _ = estimate_intrinsic_scatter(
        residuals, np.full(100, .01), weights, np.zeros(100, dtype=int),
        np.ones(100, dtype=bool), ncell=1)
    assert result[0] == pytest.approx(expected, abs=5e-5)
