"""Representative numerical paths with JAX's default single precision active."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from jax_snlc_core.cosmology import flat_cpl_comoving_integral
from jax_snlc_core.interpolation import compute_catmullrom_slopes, hermite_interp_1d
from jax_snlc_core.optimiser import newton_lm
from jax_snlc_core.utils import configure_precision


@pytest.fixture(autouse=True)
def single_precision():
    previous = jax.config.x64_enabled
    configure_precision(False)
    try:
        yield
    finally:
        configure_precision(previous)


def test_interpolated_phase_value_and_derivative():
    grid = jnp.arange(-5., 6.)
    luminosity = 10. + 2. * grid + .25 * grid ** 2
    slopes = compute_catmullrom_slopes(grid, luminosity)
    evaluate = jax.jit(jax.value_and_grad(
        lambda phase: hermite_interp_1d(phase, grid, luminosity, slopes)))
    for phase in (-2.001, -2., -1.999, .37):
        value, gradient = evaluate(jnp.asarray(phase))
        assert value.dtype == gradient.dtype == jnp.float32
        np.testing.assert_allclose(value, 10. + 2. * phase + .25 * phase ** 2, rtol=2e-6)
        np.testing.assert_allclose(gradient, 2. + .5 * phase, rtol=2e-5)


def test_batched_bounded_newton_fit():
    targets = jnp.array([[.2, .8], [1.5, .3], [-.5, .7]])
    def fit(target):
        return newton_lm(jnp.array([.5, .5]), jnp.zeros(2), jnp.ones(2),
                         lambda p: jnp.sum((p - target) ** 2), 20)
    params, loss, gradient = jax.jit(jax.vmap(fit))(targets)
    assert params.dtype == loss.dtype == gradient.dtype == jnp.float32
    np.testing.assert_allclose(params, np.clip(targets, 0., 1.), atol=2e-5)
    np.testing.assert_allclose(loss, np.sum((np.clip(targets, 0., 1.) - targets)**2,
                                          axis=1), atol=1e-6)


@pytest.mark.parametrize("radiation", [False, True])
def test_cosmology_integral_matches_independent_quadrature(radiation):
    for z in (.01, .5, 2.):
        actual = jax.jit(flat_cpl_comoving_integral)(
            jnp.asarray(z), .3, -1., 0., radiation=radiation)
        omega_r = 0.9e-4 if radiation else 0.0
        expected = quad(lambda x: 1 / np.sqrt(
            (1 - omega_r) * (.3 * (1+x)**3 + .7) + omega_r * (1+x)**4), 0., z)[0]
        assert actual.dtype == jnp.float32
        np.testing.assert_allclose(actual, expected, rtol=2e-6)
