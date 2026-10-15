"""Unit tests for the cubic Hermite interpolation utilities."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import CubicHermiteSpline

from jax_snlc_core.utils.interp import (
    compute_catmullrom_slopes,
    hermite_grid_slopes,
    hermite_interp_1d,
    hermite_interp_1d_batch,
    hermite_interp_1d_select,
    hermite_interp_grid,
    upsample_grid_linear,
)


# ---------------------------------------------------------------------------
# compute_catmullrom_slopes
# ---------------------------------------------------------------------------

def test_catmullrom_slopes_2d_per_column():
    """For 2D y = (n, m), slopes are computed independently per column."""
    x = jnp.linspace(0.0, 4.0, 5)
    y_2d = jnp.stack([
        x,         # slope 1
        2.0 * x,   # slope 2
        x ** 2,    # slope varies
    ], axis=1)
    slopes = np.asarray(compute_catmullrom_slopes(x, y_2d))
    assert slopes.shape == (5, 3)
    np.testing.assert_allclose(slopes[:, 0], np.ones(5), atol=1e-6)
    np.testing.assert_allclose(slopes[:, 1], 2.0 * np.ones(5), atol=1e-6)
    # Quadratic: slope at x=k is 2k (interior central differences match exactly).
    expected_quad = 2.0 * np.asarray(x)
    np.testing.assert_allclose(slopes[1:-1, 2], expected_quad[1:-1], atol=1e-5)


# ---------------------------------------------------------------------------
# hermite_interp_1d
# ---------------------------------------------------------------------------

def test_hermite_interp_1d_recovers_grid_points():
    """At grid points, interpolation must return the grid values exactly."""
    xp = jnp.linspace(0.0, 4.0, 5)
    fp = jnp.array([1.0, 4.0, 9.0, 16.0, 25.0])
    slopes = compute_catmullrom_slopes(xp, fp)
    out = np.asarray(hermite_interp_1d(xp, xp, fp, slopes))
    np.testing.assert_allclose(out, np.asarray(fp), atol=1e-6)


def test_hermite_interp_1d_clamps_out_of_range():
    """Queries outside [xp[0], xp[-1]] clamp to the boundary value."""
    xp = jnp.linspace(0.0, 4.0, 5)
    fp = jnp.array([1.0, 4.0, 9.0, 16.0, 25.0])
    slopes = compute_catmullrom_slopes(xp, fp)
    below = float(hermite_interp_1d(jnp.array(-10.0), xp, fp, slopes))
    above = float(hermite_interp_1d(jnp.array(+10.0), xp, fp, slopes))
    assert below == pytest.approx(1.0, abs=1e-6)
    assert above == pytest.approx(25.0, abs=1e-6)


def test_hermite_interp_1d_recovers_linear_function_exactly():
    """A cubic Hermite with linear slopes recovers a linear function exactly."""
    xp = jnp.linspace(0.0, 4.0, 5)
    fp = 3.0 * xp - 1.0
    slopes = compute_catmullrom_slopes(xp, fp)
    queries = jnp.linspace(0.5, 3.5, 7)
    out = np.asarray(hermite_interp_1d(queries, xp, fp, slopes))
    expected = 3.0 * np.asarray(queries) - 1.0
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_dense_hermite_grid_preserves_derivatives():
    xp = jnp.linspace(3000., 10000., 27503)
    fp = jnp.sin(xp / 1000.)
    slopes = compute_catmullrom_slopes(xp, fp)
    wave = jnp.linspace(3500., 8500., 500)
    reference = CubicHermiteSpline(np.asarray(xp), np.asarray(fp), np.asarray(slopes))

    def interpolate(wavelength):
        return hermite_interp_1d(wavelength, xp, fp, slopes)

    functions = (interpolate, jax.grad(interpolate), jax.grad(jax.grad(interpolate)))
    for order, function in enumerate(functions):
        actual = jax.jit(jax.vmap(function))(wave)
        expected = reference(np.asarray(wave), nu=order)
        np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-10)


# ---------------------------------------------------------------------------
# hermite_interp_1d_batch
# ---------------------------------------------------------------------------

def test_hermite_interp_1d_batch_matches_per_column_scalar():
    """Batch interp on (n_grid, n_cols) matches per-column scalar interp."""
    xp = jnp.linspace(0.0, 4.0, 5)
    fp_2d = jnp.stack([xp, xp ** 2, jnp.sin(xp)], axis=1)
    slopes_2d = compute_catmullrom_slopes(xp, fp_2d)
    queries = jnp.array([0.3, 1.7, 2.9, 3.5])

    batch = np.asarray(
        hermite_interp_1d_batch(queries, xp, fp_2d, slopes_2d)
    )
    assert batch.shape == (queries.shape[0], 3)
    for col in range(3):
        per_col = np.asarray(hermite_interp_1d(
            queries, xp, fp_2d[:, col], slopes_2d[:, col]
        ))
        np.testing.assert_allclose(batch[:, col], per_col, atol=1e-6)


# ---------------------------------------------------------------------------
# hermite_interp_1d_select
# ---------------------------------------------------------------------------

def test_hermite_interp_1d_select_matches_batch_then_take():
    """select is equivalent to batch followed by per-row column gather."""
    xp = jnp.linspace(0.0, 4.0, 5)
    fp_2d = jnp.stack([xp, xp ** 2, jnp.cos(xp)], axis=1)
    slopes_2d = compute_catmullrom_slopes(xp, fp_2d)
    queries = jnp.array([0.5, 1.5, 2.5, 3.5])
    cols = jnp.array([2, 0, 1, 2])
    sel = np.asarray(hermite_interp_1d_select(queries, xp, fp_2d, slopes_2d, cols))
    batch = np.asarray(hermite_interp_1d_batch(queries, xp, fp_2d, slopes_2d))
    via_batch = batch[np.arange(4), np.asarray(cols)]
    np.testing.assert_allclose(sel, via_batch, atol=1e-6)


# ---------------------------------------------------------------------------
# hermite_interp_grid, upsample_grid_linear
# ---------------------------------------------------------------------------

def test_hermite_grid_recovers_corner_values_and_is_c1_across_grid_lines():
    """The bicubic Hermite surface passes through the samples and, unlike a
    bilinear one, its derivatives are continuous across grid lines."""
    x_grid = jnp.linspace(0.0, 4.0, 5)
    y_grid = jnp.linspace(0.0, 6.0, 4)
    values = jnp.sin(x_grid)[:, None] * jnp.cos(y_grid)[None, :] + 0.1 * x_grid[:, None]
    slopes = hermite_grid_slopes(x_grid, y_grid, values)
    X, Y = np.meshgrid(np.asarray(x_grid), np.asarray(y_grid), indexing="ij")
    out = hermite_interp_grid(jnp.asarray(X.ravel()), jnp.asarray(Y.ravel()),
                              x_grid, y_grid, values, slopes)
    np.testing.assert_allclose(np.asarray(out), np.asarray(values).ravel(), atol=1e-6)

    f = lambda x, y: hermite_interp_grid(jnp.array([x]), jnp.array([y]),
                                         x_grid, y_grid, values, slopes)[0]
    dfdx, dfdy = jax.grad(f, 0), jax.grad(f, 1)
    eps = 1e-4
    assert abs(float(dfdx(2.0 - eps, 2.5)) - float(dfdx(2.0 + eps, 2.5))) < 1e-3
    assert abs(float(dfdy(1.3, 2.0 - eps)) - float(dfdy(1.3, 2.0 + eps))) < 1e-3


def test_upsampled_grid_is_the_linear_surface_with_rounded_corners():
    """The fine grid reads the bilinear surface, so a cubic Hermite through it
    reproduces linear interpolation inside every original cell and rounds
    each corner over one fine cell: on a step of 4 between samples the
    excursion past the samples is a few per cent of the step at a factor of
    4 and halves at a factor of 8, where a cubic through the samples
    themselves swings past them by far more."""
    x_grid = jnp.arange(6.0)
    y_grid = jnp.arange(4.0)
    values = jnp.where(x_grid[:, None] >= 3.0, 5.0, 1.0) + 0.1 * y_grid[None, :]
    x = jnp.linspace(0.0, 5.0, 501); y = jnp.full_like(x, 1.5)

    def excursion(factor):
        xf, yf, fine = upsample_grid_linear(x_grid, y_grid, values, factor)
        np.testing.assert_allclose(np.asarray(fine[::factor, ::factor]), np.asarray(values))
        out = hermite_interp_grid(x, y, xf, yf, fine, hermite_grid_slopes(xf, yf, fine))
        if factor > 1:
            inside = (x > 2.3) & (x < 2.7)
            np.testing.assert_allclose(np.asarray(out[inside]),
                                       1.15 + 4.0 * (np.asarray(x[inside]) - 2.0), atol=1e-6)
        return max(1.15 - float(out.min()), float(out.max()) - 5.15) / 4.0

    plain, four, eight = excursion(1), excursion(4), excursion(8)
    assert four < plain / 3.5 and eight < four / 1.9
