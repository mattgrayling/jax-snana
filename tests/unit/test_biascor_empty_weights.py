"""Coefficient interpolation retains its node dimensions with no events."""
import jax
import jax.numpy as jnp

from jax_salt2mu.biascor_grid import coeff_weights


def test_empty_coefficient_weights_under_jit():
    weights = jax.jit(lambda: coeff_weights(
        (0.14, 3.1), (jnp.array([0.13, 0.15]), jnp.array([3.])),
        (0.02, 1.), 0))()
    assert weights.shape == (0, 2)
