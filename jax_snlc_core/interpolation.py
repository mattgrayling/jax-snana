"""Interpolation: natural cubic splines (the K^-1 D matrix and the spline
coefficients BayeSN and the dust laws use), cubic Hermite (Catmull-Rom)
interpolation in one dimension, SNANA's linear interpolation, linear
upsampling of a grid and cubic Hermite interpolation on it in two
dimensions, multilinear interpolation on N-D grids, and the natural and monotone splines through batches of knots.

The models (SALT, BayeSN) and the fitter's redshift grid of band weights use
the Hermite forms; the fitter's redshift PDFs and host-mass grids use the
batched splines.
"""

import itertools

import jax
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


def _spline_coefficients_at(x, knots, invKD, extrap_mode):
    """The natural cubic spline's coefficients at one point ``x``; see
    :func:`spline_coefficients`."""
    n_knots = len(knots)

    q = jnp.searchsorted(knots, x, side='right') - 1
    q = jnp.clip(q, 0, n_knots - 2)
    h = knots[q + 1] - knots[q]
    a = (knots[q + 1] - x) / h
    b = 1.0 - a
    a3 = a * a * a
    b3 = b * b * b
    h2 = h * h
    c = ((a3 - a) / 6.0) * h2
    d = ((b3 - b) / 6.0) * h2
    idx = jnp.arange(n_knots)
    J_interp = (jnp.where(idx == q, a, 0.0) + jnp.where(idx == q + 1, b, 0.0)
                + c * invKD[q, :] + d * invKD[q + 1, :])

    h_last = knots[-1] - knots[-2]
    a_lin_h = (knots[-1] - x) / h_last
    b_lin_h = 1.0 - a_lin_h
    f_lin_h = (x - knots[-1]) * h_last / 6.0
    J_lin_high = jnp.zeros(n_knots)
    J_lin_high = J_lin_high.at[-2].set(a_lin_h)
    J_lin_high = J_lin_high.at[-1].set(b_lin_h)
    J_lin_high = J_lin_high + f_lin_h * invKD[-2, :]

    xu = 2.0 * knots[-1] - knots[-2]
    dx_h = xu - knots[-1]
    t_h = (x - knots[-1]) / dx_h
    r_h = dx_h / (knots[-1] - knots[-2])
    t2_h = t_h * t_h
    t3_h = t2_h * t_h
    h0_decay = 2.0 * t3_h - 3.0 * t2_h + 1.0
    h0_h = jnp.where(extrap_mode == 1, h0_decay, 1.0)
    h1_h = t3_h - 2.0 * t2_h + t_h
    f_h = dx_h * (knots[-1] - knots[-2]) / 6.0

    J_herm_high = jnp.zeros(n_knots)
    J_herm_high = J_herm_high.at[-1].set(h0_h + h1_h * r_h)
    J_herm_high = J_herm_high.at[-2].set(-h1_h * r_h)
    J_herm_high = J_herm_high + h1_h * f_h * invKD[-2, :]

    beyond_high_val = jnp.where(extrap_mode == 2, 1.0, 0.0)
    J_beyond_high = jnp.zeros(n_knots).at[-1].set(beyond_high_val)
    J_hermite_high = jnp.where(x < xu, J_herm_high, J_beyond_high)
    J_extrap_high = jnp.where(extrap_mode == 0, J_lin_high, J_hermite_high)

    h_first = knots[1] - knots[0]
    b_lin_l = (x - knots[0]) / h_first
    a_lin_l = 1.0 - b_lin_l
    f_lin_l = (x - knots[0]) * h_first / 6.0
    J_lin_low = jnp.zeros(n_knots)
    J_lin_low = J_lin_low.at[0].set(a_lin_l)
    J_lin_low = J_lin_low.at[1].set(b_lin_l)
    J_lin_low = J_lin_low - f_lin_l * invKD[1, :]

    xl = 2.0 * knots[0] - knots[1]
    dx_l = knots[0] - xl
    t_l = (x - xl) / dx_l
    r_l = dx_l / (knots[1] - knots[0])
    t2_l = t_l * t_l
    t3_l = t2_l * t_l
    h0_rise = -2.0 * t3_l + 3.0 * t2_l
    h0_l = jnp.where(extrap_mode == 1, h0_rise, 1.0)
    h1_l = t3_l - t2_l
    f_l = (knots[1] - knots[0]) * dx_l / 6.0

    J_herm_low = jnp.zeros(n_knots)
    J_herm_low = J_herm_low.at[0].set(h0_l - h1_l * r_l)
    J_herm_low = J_herm_low.at[1].set(h1_l * r_l)
    J_herm_low = J_herm_low + h1_l * f_l * invKD[1, :]

    beyond_low_val = jnp.where(extrap_mode == 2, 1.0, 0.0)
    J_beyond_low = jnp.zeros(n_knots).at[0].set(beyond_low_val)
    J_hermite_low = jnp.where(x > xl, J_herm_low, J_beyond_low)
    J_extrap_low = jnp.where(extrap_mode == 0, J_lin_low, J_hermite_low)

    return jnp.where(x > knots[-1], J_extrap_high,
                     jnp.where(x < knots[0], J_extrap_low, J_interp))


def spline_coefficients(eval_points, knots, invKD, extrap_mode):
    """The matrix J, (n_points, n_knots), with J @ y the natural cubic spline
    through (``knots``, y) at ``eval_points``, as BayeSN's spline_coeffs_irr
    builds it (genmag_BAYESN.c:1181-1310). ``invKD`` is
    :func:`invkd_irr` of the knots.

    Inside the knots every ``extrap_mode`` is the same. Beyond them mode 0
    extrapolates linearly, as the dust laws' splines do, 1 decays to zero
    and 2 holds a constant, each by a cubic Hermite over one knot spacing.
    """
    return jax.vmap(lambda x: _spline_coefficients_at(x, knots, invKD, extrap_mode))(
        eval_points)


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
    """Cubic Hermite interpolation of ``fp`` with Catmull-Rom ``slopes``.

    ``fp`` is (n,) for an ``x`` of any shape, or (n, m) for a scalar ``x``,
    giving the (m,) row at ``x``. Forward pass and AD gradients share the
    basis, so the derivatives are exact at all orders.
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


def linear_bin(x, xp):
    """The bin of ``x`` on the non-decreasing grid ``xp`` and the fraction of
    the way across it, ``(i, frac)``: the lowest ``i`` with
    ``xp[i] <= x <= xp[i + 1]``, as SNANA's quickBinSearch finds it for
    interp_1DFUN (sntools.c), so ``x`` at a grid value the grid repeats takes
    the bin below the repeat. Outside the grid, where interp_1DFUN aborts,
    ``x`` is held to the grid's ends."""
    x = jnp.clip(x, xp[0], xp[-1])
    i = jnp.clip(jnp.searchsorted(xp, x, side='left') - 1, 0, xp.shape[0] - 2)
    return i, (x - xp[i]) / (xp[i + 1] - xp[i])


def linear_between(fun0, fun1, frac):
    """interp_1DFUN's linear interpolation, ``fun0 + frac (fun1 - fun0)``,
    between the values at the two ends of a :func:`linear_bin` bin."""
    return fun0 + frac * (fun1 - fun0)


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


def spline_knots(x, y):
    """The strictly increasing knots of a batch of splines, their spacing
    and the secant slopes between them, ``(x, h, secant)``: ``x`` [n,
    n_knots], ``y`` [n_knots] or [n, n_knots]. What the slope rules and the
    evaluation read."""
    x = np.asarray(x, dtype=np.float64)
    y = np.broadcast_to(np.asarray(y, dtype=np.float64), x.shape)
    h = np.diff(x, axis=1)
    return x, h, np.diff(y, axis=1) / h


def natural_spline_slopes(knots):
    """The natural cubic spline's first derivative at every knot: its
    second derivatives at the interior knots solve a tridiagonal system with
    zero at both ends (GSL's cspline)."""
    x, h, slope = knots
    n, n_knots = x.shape
    k = n_knots - 2
    system = np.zeros((n, k, k))
    rows = np.arange(k)
    system[:, rows, rows] = 2.0 * (h[:, :-1] + h[:, 1:])
    system[:, rows[:-1], rows[1:]] = h[:, 1:-1]
    system[:, rows[1:], rows[:-1]] = h[:, 1:-1]
    second = np.zeros((n, n_knots))
    second[:, 1:-1] = np.linalg.solve(
        system, 6.0 * np.diff(slope, axis=1)[..., None])[..., 0]
    left = slope - h * (2.0 * second[:, :-1] + second[:, 1:]) / 6.0
    last = slope[:, -1:] + h[:, -1:] * (second[:, -2:-1] + 2.0 * second[:, -1:]) / 6.0
    return np.concatenate([left, last], axis=1)


def steffen_slopes(knots):
    """Steffen's monotone first derivative at every knot (GSL's steffen,
    Steffen 1990): the interior slope is the parabola's through the three
    knots, capped at twice the smaller neighbouring secant and zero where
    the secants change sign, and the end slopes are the end secants."""
    x, h, slope = knots
    parabola = ((slope[:, :-1] * h[:, 1:] + slope[:, 1:] * h[:, :-1])
                / (h[:, :-1] + h[:, 1:]))
    interior = ((np.sign(slope[:, :-1]) + np.sign(slope[:, 1:]))
                * np.minimum(np.abs(slope[:, :-1]),
                             np.minimum(np.abs(slope[:, 1:]),
                                        0.5 * np.abs(parabola))))
    return np.concatenate([slope[:, :1], interior, slope[:, -1:]], axis=1)


def hermite_spline_derivative(knots, knot_slopes, n_fine):
    """The derivative of the piecewise cubic through the ``knots`` with
    first derivative ``knot_slopes`` at them, on ``n_fine`` evenly spaced
    points over each row's knots. Returns ``(points, derivative)``, each
    [n, n_fine]."""
    x, h, slope = knots
    n, n_knots = x.shape
    points = np.linspace(x[:, 0], x[:, -1], n_fine, axis=1)
    step = (x[:, -1] - x[:, 0]) / (n_fine - 1)
    first = np.ceil((x[:, 1:-1] - x[:, :1]) / step[:, None]).astype(int)
    edges = np.concatenate([np.zeros((n, 1), int), np.clip(first, 0, n_fine),
                            np.full((n, 1), n_fine)], axis=1)
    counts = np.diff(edges, axis=1).ravel()               # [n * (n_knots-1)]

    def per_point(per_interval):
        return np.repeat(per_interval.ravel(), counts).reshape(n, n_fine)

    d_lo, d_hi = knot_slopes[:, :-1], knot_slopes[:, 1:]
    # y = d + c t + b t^2 + a t^3 on each interval, t from its left knot
    a = per_point((d_lo + d_hi - 2.0 * slope) / h ** 2)
    b = per_point((3.0 * slope - 2.0 * d_lo - d_hi) / h)
    t = points - per_point(x[:, :-1])
    return points, per_point(d_lo) + t * (2.0 * b + 3.0 * a * t)


def hermite_spline_at(knots, values, slopes, q):
    """The piecewise cubic through ``values`` at the ``knots`` with first
    derivative ``slopes`` there, at one point per row, and 0 outside the
    knots, as eval_spline returns (sntools_spline.c:266)."""
    x = knots[0]
    n_knots = x.shape[1]
    i = np.clip((x <= q[:, None]).sum(axis=1) - 1, 0, n_knots - 2)
    r = np.arange(len(q))
    h = x[r, i + 1] - x[r, i]
    t = (q - x[r, i]) / h
    t2, t3 = t * t, t * t * t
    cubic = ((2 * t3 - 3 * t2 + 1) * values[r, i]
             + (t3 - 2 * t2 + t) * h * slopes[r, i]
             + (3 * t2 - 2 * t3) * values[r, i + 1]
             + (t3 - t2) * h * slopes[r, i + 1])
    return np.where((q >= x[:, 0]) & (q <= x[:, -1]), cubic, 0.0)


def interpolate_corners(values, lower_indices, fractions, map_index=None):
    """Sum the corner values of each query's grid cell with multilinear weights.

    ``lower_indices`` and ``fractions`` contain one query-shaped array per
    axis: the lower knot index and the fraction towards the upper knot.
    With ``map_index``, ``values`` has a leading map axis and each query
    selects its own map. Knot indices are clipped at the grid edges.
    """
    shape = values.shape if map_index is None else values.shape[1:]
    result = 0.0
    for corner in itertools.product((0, 1), repeat=len(shape)):
        indices = [] if map_index is None else [map_index]
        weight = None
        for index, fraction, size, upper in zip(lower_indices, fractions, shape, corner):
            axis_weight = fraction if upper else 1.0 - fraction
            weight = axis_weight if weight is None else weight * axis_weight
            indices.append(jnp.clip(index + upper, 0, size - 1))
        result = result + weight * values[tuple(indices)]
    return result


def interp_nd_regular(points, grid_axes_list, grid_values, opt_extrap):
    """Multilinear interpolation on per-dimension axes.

    ``points`` is (n_points, n_dims), ``grid_axes_list`` one (n_grid_i,)
    array per dimension, ``grid_values`` the N-D array of values on the
    grid. ``opt_extrap`` is 0 to return 0 out-of-range or 1 to clip to the
    edge. Returns the (n_points,) interpolated values.
    """
    n_points = points.shape[0]
    n_dims = len(grid_axes_list)

    # a one-knot 1-D map is its value everywhere (interp_GRIDMAP, sntools_gridmap.c:538-545)
    if n_dims == 1 and grid_axes_list[0].shape[0] == 1:
        return jnp.full(n_points, grid_values[0])

    in_range = jnp.ones(n_points, dtype=bool)
    if opt_extrap == 0:
        for d in range(n_dims):
            ax = grid_axes_list[d]
            in_range = in_range & (points[:, d] >= ax[0]) & (points[:, d] <= ax[-1])

    if n_dims == 1:
        result = jnp.interp(points[:, 0], grid_axes_list[0], grid_values)
        if opt_extrap == 0:
            result = jnp.where(in_range, result, 0.0)
        return result

    lower_indices = []
    fractions = []
    for d in range(n_dims):
        ax = grid_axes_list[d]
        n_ax = len(ax)
        x = jnp.clip(points[:, d], ax[0], ax[-1])
        idx = jnp.clip(jnp.searchsorted(ax, x, side='right') - 1, 0, n_ax - 2)
        dx = ax[idx + 1] - ax[idx]
        safe_dx = jnp.where(dx > 0, dx, 1.0)
        frac = jnp.where(dx > 0, (x - ax[idx]) / safe_dx, 0.0)
        lower_indices.append(idx)
        fractions.append(frac)

    result = interpolate_corners(grid_values, lower_indices, fractions)

    if opt_extrap == 0:
        result = jnp.where(in_range, result, 0.0)
    return result
