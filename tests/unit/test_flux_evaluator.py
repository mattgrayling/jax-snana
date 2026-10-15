"""Faint model predictions stay continuous in the fitting objective."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_snlc_fit.flux_evaluator import fit_flux


@pytest.mark.parametrize('zp', [27.5, 30.0])
def test_faint_positive_flux_and_objective_derivatives(zp):
    old_cutoff = 10.0 ** (-0.4 * (40.0 - zp))
    flux = jnp.asarray(old_cutoff * np.array([.001, .999, 1., 1.001]))
    np.testing.assert_array_equal(fit_flux(flux, zp), flux)

    # A negative observed flux exposed the jump in the original objective.
    loss = lambda f: (fit_flux(f, zp) + .5) ** 2
    gradient = jax.vmap(jax.grad(loss))(flux)
    curvature = jax.vmap(jax.grad(jax.grad(loss)))(flux)
    np.testing.assert_allclose(gradient, 2 * (flux + .5), rtol=1e-6)
    np.testing.assert_allclose(curvature, 2., rtol=1e-6)


def test_nonpositive_nonfinite_and_bright_predictions():
    zp = 27.5
    bright_limit = 10.0 ** (-0.4 * (5.0 - zp))
    flux = jnp.asarray([-1., 0., np.nan, -np.inf, np.inf,
                        bright_limit, 2 * bright_limit])
    np.testing.assert_array_equal(fit_flux(flux, zp), 0.)
