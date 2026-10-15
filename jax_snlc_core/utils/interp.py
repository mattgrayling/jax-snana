"""Cubic Hermite (Catmull-Rom) interpolation utilities, used across models
(SALT, BayeSN) and the fitter (z-grid band weight interpolation).
"""

import jax.numpy as jnp
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def invkd_irr(knots: np.ndarray) -> np.ndarray:
    """K^-1 D for a natural cubic spline on irregular knots (``invKD_irr``,
    genmag_BAYESN.c:1128), padded with a zero row at each end.

    Built once at setup, in double precision whatever frame the run is in, as
    SNANA builds it; the dust laws and BayeSN both spline through it.
    """
    x = np.asarray(knots, dtype=np.float64)
    n = len(x)
    K = np.zeros((n - 2, n - 2))
    D = np.zeros((n - 2, n))

    K[0, 0:2] = [(x[2] - x[0]) / 3, (x[2] - x[1]) / 6]
    K[-1, -2:n - 2] = [(x[n - 2] - x[n - 3]) / 6, (x[n - 1] - x[n - 3]) / 3]

    for j in np.arange(2, n - 2):
        row = j - 1
        K[row, row - 1:row + 2] = [
            (x[j] - x[j - 1]) / 6,
            (x[j + 1] - x[j - 1]) / 3,
            (x[j + 1] - x[j]) / 6
        ]
    for j in np.arange(1, n - 1):
        row = j - 1
        D[row, row:row + 3] = [
            1.0 / (x[j] - x[j - 1]),
            -(1.0 / (x[j + 1] - x[j]) + 1.0 / (x[j] - x[j - 1])),
            1.0 / (x[j + 1] - x[j])
        ]

    M = np.zeros((n, n))
    M[1:-1, :] = np.linalg.solve(K, D)
    return M


def compute_catmullrom_slopes(grid_x, grid_y):
    """Compute Catmull-Rom slopes (central differences) along axis 0.

    For a 1D grid_y: returns shape (n,).
    For a 2D grid_y of shape (n, m): returns shape (n, m).

    Interior points use ``(y[i+1] - y[i-1]) / (x[i+1] - x[i-1])``, boundaries
    one-sided differences. ``grid_x`` is the monotonic grid coordinates,
    shape (n,); ``grid_y`` is the values on the grid, shape (n,) or (n, m);
    either may be numpy or jax. Returns the slopes as a jnp array, the same
    shape as ``grid_y``.
    """
    grid_x = jnp.asarray(grid_x)
    grid_y = jnp.asarray(grid_y)

    if grid_y.ndim == 1:
        dx = grid_x[2:] - grid_x[:-2]
        dy = grid_y[2:] - grid_y[:-2]
        interior = dy / dx
        s0 = (grid_y[1] - grid_y[0]) / (grid_x[1] - grid_x[0])
        sn = (grid_y[-1] - grid_y[-2]) / (grid_x[-1] - grid_x[-2])
        slopes = jnp.concatenate([s0[None], interior, sn[None]])
    else:
        dx = (grid_x[2:] - grid_x[:-2])[:, None]
        dy = grid_y[2:] - grid_y[:-2]
        interior = dy / dx
        s0 = (grid_y[1] - grid_y[0]) / (grid_x[1] - grid_x[0])
        sn = (grid_y[-1] - grid_y[-2]) / (grid_x[-1] - grid_x[-2])
        slopes = jnp.concatenate([s0[None, :], interior, sn[None, :]], axis=0)

    return slopes


def hermite_weights(x, xp):
    """The cubic Hermite basis at ``x`` on grid ``xp``.

    Returns the lower-cell index, the cell width and the four basis
    polynomials (h00, h10, h01, h11), so a caller only has to gather its own
    values and slopes at ``idx`` and ``idx + 1``. Queries outside the grid are
    held at its ends.
    """
    n = xp.shape[0]
    x_clamped = jnp.clip(x, xp[0], xp[-1])
    idx = jnp.clip(jnp.searchsorted(xp, x_clamped, side='right',
                                    method='scan_unrolled') - 1, 0, n - 2)
    h = xp[idx + 1] - xp[idx] + 1e-12
    t = (x_clamped - xp[idx]) / h
    t2 = t * t
    t3 = t2 * t
    basis = (2 * t3 - 3 * t2 + 1, t3 - 2 * t2 + t,
             -2 * t3 + 3 * t2, t3 - t2)
    return idx, h, basis


def hermite_interp_1d(x, xp, fp, slopes):
    """Cubic Hermite interpolation of ``fp`` (n,) with Catmull-Rom ``slopes``.

    Forward pass and AD gradients share the basis, so the derivatives are
    exact at all orders. ``x`` may be any shape.
    """
    idx, h, (h00, h10, h01, h11) = hermite_weights(x, xp)
    return (h00 * fp[idx] + h10 * h * slopes[idx]
            + h01 * fp[idx + 1] + h11 * h * slopes[idx + 1])


def hermite_interp_1d_batch(x, xp, fp_2d, slopes_2d):
    """Cubic Hermite over every column of a surface at once.

    ``fp_2d`` and ``slopes_2d`` are (n_grid, n_cols) and ``x`` is (n_query,);
    the result is (n_query, n_cols). Taking all columns together avoids a vmap
    over columns, which would replicate the grid per column.
    """
    idx, h, basis = hermite_weights(x, xp)
    h00, h10, h01, h11 = (b[:, None] for b in basis)
    h_col = h[:, None]
    return (h00 * fp_2d[idx] + h10 * h_col * slopes_2d[idx]
            + h01 * fp_2d[idx + 1] + h11 * h_col * slopes_2d[idx + 1])


def hermite_interp_1d_select(x, xp, fp_2d, slopes_2d, col_indices):
    """:func:`hermite_interp_1d_batch` taking one column per query point, so
    the (n_query, n_cols) intermediate never exists; returns (n_query,)."""
    idx, h, (h00, h10, h01, h11) = hermite_weights(x, xp)
    return (h00 * fp_2d[idx, col_indices]
            + h10 * h * slopes_2d[idx, col_indices]
            + h01 * fp_2d[idx + 1, col_indices]
            + h11 * h * slopes_2d[idx + 1, col_indices])


def upsample_grid_linear(x_grid, y_grid, values, factor):
    """``values`` (nx, ny) on ``factor`` times finer grids along both axes,
    the new samples read off the bilinear surface, so the fine grid is the
    original's linear interpolant. Returns ``(x_fine, y_fine, values_fine)``.
    A cubic Hermite through the fine grid then reproduces the linear surface
    inside every original cell and rounds its corners over one fine cell,
    which is how a map that is not resolved on its own grid is read smoothly
    without the overshoot of a cubic through its samples."""
    x_grid, y_grid, values = (np.asarray(a, dtype=np.float64) for a in (x_grid, y_grid, values))
    x_fine = np.linspace(x_grid[0], x_grid[-1], (len(x_grid) - 1) * factor + 1)
    y_fine = np.linspace(y_grid[0], y_grid[-1], (len(y_grid) - 1) * factor + 1)
    fine = RegularGridInterpolator((x_grid, y_grid), values)(
        np.stack(np.meshgrid(x_fine, y_fine, indexing='ij'), axis=-1))
    return jnp.asarray(x_fine), jnp.asarray(y_fine), jnp.asarray(fine)


def hermite_grid_slopes(x_grid, y_grid, values):
    """The Catmull-Rom slopes of a surface ``values`` (nx, ny) along each
    axis and the cross derivative, for `hermite_interp_grid`: precomputed
    once per surface. Returns ``(dx, dy, dxy)``, each (nx, ny)."""
    dx = compute_catmullrom_slopes(x_grid, values)
    dy = compute_catmullrom_slopes(y_grid, jnp.asarray(values).T).T
    dxy = compute_catmullrom_slopes(y_grid, dx.T).T
    return dx, dy, dxy


def hermite_interp_grid(x, y, x_grid, y_grid, values, slopes):
    """Bicubic Hermite interpolation of ``values`` (nx, ny) at ``(x, y)``,
    each (n,), with the surface's ``slopes`` from `hermite_grid_slopes`: the
    tensor product of the 1-D cubic Hermite bases, so the surface and its
    first derivatives are continuous across grid lines, where a bilinear
    surface's derivatives jump. Queries outside the grid are held at its
    ends, as `hermite_weights` holds them."""
    dx, dy, dxy = slopes
    ix, hx, (a00, a10, a01, a11) = hermite_weights(x, x_grid)
    iy, hy, (b00, b10, b01, b11) = hermite_weights(y, y_grid)
    ix1, iy1 = ix + 1, iy + 1

    def corner(i, j, ax_v, ax_d, by_v, by_d):
        return (ax_v * by_v * values[i, j]
                + ax_d * hx * by_v * dx[i, j]
                + ax_v * by_d * hy * dy[i, j]
                + ax_d * hx * by_d * hy * dxy[i, j])
    return (corner(ix, iy, a00, a10, b00, b10)
            + corner(ix1, iy, a01, a11, b00, b10)
            + corner(ix, iy1, a00, a10, b01, b11)
            + corner(ix1, iy1, a01, a11, b01, b11))


def parse_genpoly(raw):
    """Parse a SNANA GENPOLY into terms, each a float or a (lo, hi) range
    (SNANA 'lo:hi', drawn per SN). Returns [] if unset.

    read_genpoly (sntools.c:11586) accepts 'a0,a1,a2' and the older
    space-separated 'a0 a1 a2'; the latter reaches here already split into a
    list, so commas and whitespace are both treated as separators.
    """
    if raw is None:
        return []
    tokens = raw if isinstance(raw, (list, tuple)) else [raw]
    terms = []
    for token in tokens:
        for part in str(token).replace(',', ' ').split():
            if ':' in part:
                lo, hi = part.split(':')
                terms.append((float(lo), float(hi)))
            else:
                terms.append(float(part))
    return terms


def eval_genpoly(coeffs, x):
    """SNANA's eval_GENPOLY: c0 + c1*x + c2*x^2 + ... (ascending coeffs).

    Horner form over plain arithmetic, so it evaluates NumPy float64 and
    traced JAX values alike, preserving the caller's dtype. ``coeffs`` needs
    a static length (a sequence, or an array iterated at trace time).
    """
    out = None
    for c in reversed(list(coeffs)):
        out = c if out is None else out * x + c
    return x * 0 if out is None else out + x * 0


def eval_genpoly_rows(coeffs, x):
    """eval_GENPOLY with one coefficient row per element of ``x``.

    ``coeffs`` is (n, n_coeffs) ascending per row (zero-padded rows are
    exact, since zero coefficients contribute nothing); ``x`` is (n,).
    """
    out = None
    n_coeffs = coeffs.shape[1]
    for i in range(n_coeffs - 1, -1, -1):
        c = coeffs[:, i]
        out = c if out is None else out * x + c
    return x * 0 if out is None else out
