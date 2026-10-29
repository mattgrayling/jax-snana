"""Observed band integrals retain the derivatives of direct quadrature."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_snlc_core.interpolation import compute_catmullrom_slopes, hermite_interp_1d_select
from jax_snlc_core.models.salt import _make_band_integrator


@pytest.mark.parametrize('photoz', [False, True])
@pytest.mark.parametrize('explicit_weight_derivative', [False, True])
@pytest.mark.parametrize('point', [(.08, .15, .2), (.08, 0., 0.)])
def test_observed_band_derivatives(photoz, explicit_weight_derivative, point):
    rng = np.random.default_rng(18)
    grid = jnp.array([-12., -3., 0., 8., 20.])
    m0, m1 = (jnp.asarray(rng.normal(size=(5, 17))) for _ in range(2))
    boundary = jnp.asarray(rng.normal(size=(2, 4, 17)))
    basis = jnp.linspace(-1., 2., 17)
    weights = jnp.asarray(rng.uniform(.1, 1., size=(3, 17)))
    weight_slope = jnp.asarray(rng.normal(size=(3, 17))) * .1
    times = jnp.array([-30., -12., -3., 0., .2, .2, 8., 20., 40.])
    filters = jnp.array([0, 1, 2, 0, 0, 0, 1, 2, 2])
    integrate = _make_band_integrator(m0, m1, boundary, basis, .03, grid)

    def evaluate(p, custom):
        colour, peak, z = p
        z = z if photoz else .1
        bw = weights + z * weight_slope + z**2 * weights * .02
        bwz = weight_slope + z * weights * .04
        phases = (times - peak) / (1. + z)
        if custom:
            values = integrate(colour, bw, z,
                               bwz if explicit_weight_derivative else None,
                               phases, filters)
        else:
            weighted = bw * jnp.exp((colour - .03) * basis)
            surfaces = (m0 @ weighted.T, m1 @ weighted.T)
            values = tuple(hermite_interp_1d_select(
                phases, grid, surface, compute_catmullrom_slopes(grid, surface), filters)
                for surface in surfaces)
            edges = jnp.einsum('scw,fw->scf', boundary, weighted)
            side = jnp.where(phases < grid[0], 0, 1)
            values = (*values, edges[side, :, filters])
        return jnp.concatenate([value.ravel() for value in values])

    point = jnp.array(point)
    actual = lambda p: evaluate(p, True)
    expected = lambda p: evaluate(p, False)
    np.testing.assert_allclose(actual(point), expected(point), rtol=2e-14, atol=2e-14)
    for derivative in (jax.jacfwd, jax.jacrev,
                       lambda f: jax.jacfwd(jax.jacrev(f))):
        np.testing.assert_allclose(derivative(actual)(point), derivative(expected)(point),
                                   rtol=2e-12, atol=2e-12)
