"""Compressed BBC curvature must recover all blocks of the dense Hessian."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_salt2mu.derivatives import loss_derivatives
from jax_snlc_core.utils.newton_lm import _grad_and_hess, newton_lm


@pytest.mark.parametrize('nzbin', [1, 2, 7])
@pytest.mark.parametrize('interpolate', [False, True])
@pytest.mark.parametrize('joint', [False, True])
@pytest.mark.parametrize('nuisance', [0, 3])
def test_full_curvature(nzbin, interpolate, joint, nuisance):
    rng = np.random.default_rng(731)
    # Repeated and empty bins; extrapolation and zero interpolation fractions.
    cell = jnp.asarray(rng.integers(0, nzbin, 31))
    cell = jnp.where(cell == 3, 2, cell)
    neighbour = jnp.clip(cell + jnp.where(jnp.arange(31) % 2, 1, -1), 0, nzbin - 1)
    fraction = jnp.linspace(-.3, 1.2, 31) if interpolate else jnp.zeros(31)
    x = jnp.asarray(rng.normal(size=(31, nuisance)))
    observed = jnp.asarray(rng.normal(size=31))
    valid = jnp.asarray(np.arange(31) % 6 != 0)

    def loss(p):
        globals_, offsets = p[:nuisance], p[nuisance:nuisance + nzbin]
        m0 = offsets[cell] + fraction * (offsets[neighbour] - offsets[cell])
        predicted = jnp.sin(x @ globals_) - m0
        variance = .3 + (x @ globals_)**2
        if joint:
            variance += p[-1]**2
        # A shared, nonlinear nuisance-only quantity mimics a CC population
        # calculation without creating any coupling between M0 bins.
        prior = jnp.exp(-jnp.sum(globals_**2))
        term = -2 * jnp.log(jnp.exp(-.5 * (observed - predicted)**2 / variance)
                            / jnp.sqrt(variance) + .1 * prior)
        return jnp.sum(valid * term)

    p = jnp.asarray(rng.normal(size=nuisance + nzbin + int(joint)) * .1)
    for with_value in (False, True):
        actual = jax.jit(partial(loss_derivatives, loss=loss, n_nuisance=nuisance,
                                 nzbin=nzbin, interpolate=interpolate,
                                 with_value=with_value))(p)
        expected = _grad_and_hess(p, loss, with_value)
        for a, b in zip(actual, expected):
            if b is None:
                assert a is None
            else:
                np.testing.assert_allclose(a, b, rtol=2e-12, atol=2e-12)


def test_derivative_callback_preserves_newton_steps():
    def loss(p):
        return jnp.sum((p[1:] - jnp.sin(p[0]))**2) + (p[0] - .2)**2

    def derivatives(p, closed, with_value):
        return loss_derivatives(p, closed, n_nuisance=1, nzbin=4,
                                interpolate=False, with_value=with_value)

    p = jnp.asarray([.7, -.4, .8, .1, -.2])
    lo, hi = jnp.full(5, -1.), jnp.full(5, 1.)
    # Compare the original algorithm at several intermediate step budgets.
    for steps in (1, 2, 5, 60):
        ordinary = newton_lm(p, lo, hi, loss, steps)
        structured = newton_lm(p, lo, hi, loss, steps, derivative_fn=derivatives)
        for a, b in zip(ordinary, structured):
            np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-12)
