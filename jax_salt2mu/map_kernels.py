"""JAX numerical kernels for BBC correction maps.

Array contents, selection masks and scientific bin counts are dynamic. Buffer
dimensions and algorithm choices are static. Event rows and redshift grids use
capacities chosen from the known requests.
"""
from functools import partial, wraps
from itertools import product
from math import prod

import numpy as np

import jax
import jax.numpy as jnp

from .scatter import estimate
from .biascor_grid import below_centroid


def map_bounds(bins, z_capacity=None):
    """Numerical bin edges with optional spare capacity on the redshift axis."""
    bounds = []
    for axis, b in enumerate(bins):
        capacity = z_capacity if axis == 0 and z_capacity is not None else b.nbin
        if capacity < b.nbin:
            raise ValueError("redshift capacity is smaller than the scientific grid")
        bounds.append((np.pad(b.lo, (0, capacity - b.nbin), constant_values=np.inf),
                       np.pad(b.hi, (0, capacity - b.nbin), constant_values=np.inf),
                       b.binsize, b.nbin))
    return tuple(bounds)


def pad_map_rows(array, capacity, fill=0):
    """Pad event arrays only when a sample needs spare numerical capacity."""
    array = np.asarray(array)
    extra = capacity - len(array)
    if extra < 0:
        raise ValueError("bias-correction row capacity is too small")
    if extra == 0:
        return array
    return np.pad(array, [(0, extra), *([(0, 0)] * (array.ndim - 1))],
                  constant_values=fill)


def pad_map_cells(array, dims, z_axis, z_capacity, fill):
    """Pad flattened map cells along redshift, preserving coefficient-node order."""
    array = np.asarray(array)
    extra = z_capacity - dims[z_axis]
    if extra < 0:
        raise ValueError("redshift capacity is smaller than the scientific grid")
    if extra == 0:
        return array
    widths = [(0, 0)] * (len(dims) + array.ndim - 1)
    widths[z_axis] = (0, extra)
    shape = (*dims, *array.shape[1:])
    padded = np.pad(array.reshape(shape), widths, constant_values=fill)
    return padded.reshape(-1, *array.shape[1:])


def trim_map_cells(array, dims, z_axis, z_count):
    """Return only scientific cells from a map's padded numerical buffer."""
    array = np.asarray(array)
    selection = [slice(None)] * len(dims)
    selection[z_axis] = slice(z_count)
    return array.reshape(*dims, *array.shape[1:])[tuple(selection)].reshape(
        -1, *array.shape[1:]).copy()


def index(v, b, clamp=False):
    """Inclusive bin edges, lower-bin ties and the final-edge 1e-10 allowance."""
    lo, hi, _, count = b
    hi = hi.at[count - 1].add(1e-10)
    i = jnp.searchsorted(hi, v, side='left')
    s = jnp.minimum(i, count - 1)
    i = jnp.where((i < count) & (v >= lo[s]) & (v <= hi[s]), i, -9)
    if clamp:
        i = jnp.where(v < (lo[0] + hi[0]) / 2, 0, i)
        i = jnp.where(v > (lo[count - 1] + hi[count - 1] - 1e-10) / 2, count - 1, i)
    return i


def cell_sum(j, x, n):
    return jnp.bincount(j, weights=x, length=n)


@partial(jax.jit, static_argnames=('dims', 'active'))
def lookup(values, indices, bounds, averages, valmap, counts, node,
           min_per_cell, zmax, *, dims, active):
    """Interpolate bias at each event's coefficient node, with coverage status.

    Widths expand to neighbouring centroids. Weights multiply in the reference
    order (z, model coordinates, mass); coefficient nodes never mix.
    """
    code = jnp.where(jnp.all(jnp.stack([i >= 0 for i in indices]), axis=0), 1, -1)
    counts_per_axis = tuple(b[3] for b in bounds)
    centres = [jnp.clip(i, 0, n - 1) for i, n in zip(indices, counts_per_axis)]
    strides = tuple(prod(dims[i + 1:]) for i in range(len(dims)))

    def ravel(coords):
        prefix = node if coords[0].ndim == 1 else node[:, None]
        return prefix * prod(dims) + sum(c * s for c, s in zip(coords, strides))
    centre_cells = ravel(centres)
    choices = []
    valid = []
    upper = []
    for v, c, avg, n, on, bound in zip(values, centres, averages, counts_per_axis, active, bounds):
        below = below_centroid(v, avg[centre_cells], counts[centre_cells], bound[2], jnp)
        lo = c - below.astype(jnp.int64)
        candidate = lo[:, None] + jnp.arange(2) if on else c[:, None]
        inside = ((candidate >= 0) & (candidate < n) if on
                  else jnp.ones(candidate.shape, dtype=bool))
        choices.append(candidate)
        valid.append(inside)
        upper.append(jnp.minimum(lo + 1, n - 1))
    corners = jnp.array(list(product(*(range(2 if on else 1) for on in active))))
    nbr = [c[:, corners[:, d]] for d, c in enumerate(choices)]
    inside = jnp.all(jnp.stack([v[:, corners[:, d]]
                               for d, v in enumerate(valid)]), axis=0)
    inside &= (code == 1)[:, None]
    cells = ravel([jnp.clip(i, 0, n - 1) for i, n in zip(nbr, counts_per_axis)])
    enough = counts[cells] >= min_per_cell
    defined = jnp.all(jnp.stack([avg[cells] <= 9000 for avg in averages]), axis=0)
    spread = inside & enough & defined
    widths = []
    for (_, _, width, size), avg, neighbour, centre, on in zip(
            bounds, averages, nbr, centres, active):
        if not on:
            code = jnp.where((code == 1) & ((width == 0) | (width > 9000)), -2, code)
            widths.append(jnp.full_like(values[0], width))
            continue
        dist = jnp.where(spread & (neighbour != centre[:, None]),
                         jnp.abs(avg[centre_cells, None] - avg[cells]), -jnp.inf)
        width = jnp.where(size > 1, jnp.maximum(width, jnp.max(dist, axis=1)), width)
        code = jnp.where((code == 1) & ((width == 0) | (width > 9000)), -2, code)
        widths.append(width)
    use = inside & enough & (code == 1)[:, None]
    factors = []
    for v, avg, width, on in zip(values, averages, widths, active):
        distance = (jnp.abs(v[:, None] - avg[cells]) /
                    jnp.where(width > 0, width, 1.0)[:, None])
        factors.append(1.0 - distance if on else jnp.ones_like(distance))
    weights = factors[0]
    for factor in (*factors[2:], factors[1]):
        weights = weights * factor
    weights = jnp.where(use, jnp.maximum(weights, 0.0), 0.0)
    total = weights.sum(axis=1)
    nsum = jnp.where(use, counts[cells], 0).sum(axis=1)
    centre_used = (use & (cells == centre_cells[:, None])).any(axis=1)
    upper_used = (use & (nbr[0] == upper[0][:, None])).any(axis=1)
    capacity = jnp.prod(jnp.array([jnp.where(n > 1, 2, 1) if on else 1
                                   for n, on in zip(counts_per_axis, active)]))
    minimum = jnp.minimum(3, capacity)
    for bad, why in (
            (use.sum(axis=1) < minimum, -9), (~centre_used, -10),
            (~upper_used, -11),
            (~((nsum >= 10) | ((nsum >= 2) & (values[0] > zmax))), -12)):
        code = jnp.where((code == 1) & bad, why, code)
    val = ((weights[:, :, None] * valmap[cells]).sum(axis=1) /
           jnp.where(total > 0, total, 1.0)[:, None])
    invalid = (code == 1) & ((total <= 1e-09) | jnp.any(jnp.abs(val) > 20.0, axis=1))
    return (val, code, invalid)


@partial(jax.jit, static_argnames=('dims', 'active'))
def fit_map(coords, bounds, node, coeff_valid, v, covfit, bias, pop,
            siggrid, mask, nbr, nbrvalid, sigma, *, dims, active):
    """Bin, locate centroids, apply Gaussian weights and accumulate bias moments."""
    ncell = len(nbr)
    spatial = int(prod(dims))
    ix = tuple(index(x, b) for x, b in zip(coords, bounds))
    valid = mask & coeff_valid & jnp.all(jnp.stack([i >= 0 for i in ix]), axis=0)
    j = node * spatial
    for d, i in enumerate(ix):
        j = j + jnp.maximum(i, 0) * int(prod(dims[d + 1:]))
    j = jnp.where(valid, j, 0)
    coords = tuple(jnp.where(valid, x, 0.) for x in coords)
    bias = jnp.where(valid[:, None], bias, 0.0)
    sig = siggrid[node]
    w = jnp.where(valid, pop / (covfit + sig * sig * v[:, 0] ** 2), 0.0)
    averages = centroids(j, coords, w, ncell=ncell)
    sqd = sum(((x - avg[j]) / b[2]) ** 2
              for d, (x, avg, b) in enumerate(zip(coords, averages, bounds))
              if d != 1 or (dims[1] > 1 and active[1]))
    w *= jnp.where(sigma <= 10., jnp.exp(-.5 * sqd / (sigma * sigma)), 1.)
    w = jnp.where(valid, w, 0.0)
    wp = jnp.where(valid, pop, 0.0)
    val, err, rms, count = fit_moments(j, bias, w, wp, valid, nbr, nbrvalid)
    return val, err, rms, count, averages


@jax.jit
def fit_moments(j, bias, w, wp, valid, nbr, nbrvalid):
    """Weighted bias and population scatter, borrowing only spatial neighbours.

    The mean uses inverse-variance weights. Scatter uses population weights;
    its error divides by the centre population even when neighbours contribute.
    """
    ncell = len(nbr)
    count = cell_sum(j, valid.astype(jnp.int64), ncell)
    population = cell_sum(j, wp, ncell)
    weights = cell_sum(j, w, ncell)

    def hood(x):
        return jnp.where(count < 10,
                         jnp.where(nbrvalid, x[nbr], 0.).sum(axis=1), x)
    ptotal = hood(population)

    def parameter(b):
        p = hood(cell_sum(j, wp * b, ncell))
        q = hood(cell_sum(j, wp * b * b, ncell))
        rms = jnp.sqrt(jnp.abs(q / ptotal - (p / ptotal) ** 2))
        err = rms / jnp.sqrt(population)
        return (jnp.where(count > 0, cell_sum(j, w * b, ncell) / weights, 999.),
                jnp.where(count > 0, jnp.where(err != 0, err, 1e-12), 999.),
                jnp.where(count > 0, rms, 999.))
    val, err, rms = jax.vmap(parameter, in_axes=1, out_axes=1)(bias)
    return val, err, rms, count


@partial(jax.jit, static_argnames=('dims', 'mdims', 'active',
                                  'do_mad', 'do_add', 'add_mad'))
def error_map(coords, mcoords, bounds, mbounds, node, coeff_valid, bcoef,
              covfit, mures, pop, siggrid, mask, averages, val, count, zmax,
              mpc, *, dims, mdims, active, do_mad, do_add, add_mad):
    """Accumulate pull variance/MAD and optional signed intrinsic covariance."""
    ncell = len(count)
    spatial = prod(dims)
    ix = tuple(index(x, b) for x, b in zip(coords, bounds))
    sig = siggrid[node]
    iv, code, invalid = lookup(coords, ix, bounds, averages, val, count,
                               node, mpc, zmax, dims=dims, active=active)
    mix = tuple(index(x, b, clamp=d > 0)
                for d, (x, b) in enumerate(zip(mcoords, mbounds)))
    mn = int(ncell // spatial * prod(mdims))
    mj = node * int(prod(mdims))
    for d, i in enumerate(mix):
        mj = mj + jnp.maximum(i, 0) * int(prod(mdims[d + 1:]))
    keep = (mask & coeff_valid & (code == 1) &
            jnp.all(jnp.stack([i >= 0 for i in mix]), axis=0))
    mj = jnp.where(keep, mj, 0)
    md = mures - jnp.sum(bcoef * iv, axis=1)
    md = jnp.where(keep, md, 0.0)
    covfit = jnp.where(keep, covfit, 1.0)
    mcoords = tuple(jnp.where(keep, x, 0.) for x in mcoords)
    pull = md / jnp.sqrt(covfit if do_add else covfit + sig * sig)
    mw = jnp.where(keep, pop, 0.0)
    total = cell_sum(mj, mw, mn)
    number = cell_sum(mj, keep.astype(jnp.int64), mn)
    use = number >= 5
    if do_mad:
        # SNANA's population-weighted median uses event-count parity to
        # decide whether to average two adjacent sorted absolute pulls.
        jj = jnp.where(keep, mj, mn)
        # Carry weights through the stable sort: tied pulls retain the same
        # cumulative weights without sorting and gathering an index array.
        _, ps, sorted_weights = jax.lax.sort((jj, jnp.abs(pull), mw), num_keys=2)
        end = jnp.cumsum(number)
        start = end - number
        cum = jnp.concatenate((jnp.zeros(1), jnp.cumsum(sorted_weights)))
        k0 = jnp.searchsorted(cum, 0.5 * (cum[start] + cum[end])) - 1
        k1 = jnp.where((end - start) % 2 == 1, k0, k0 + 1)
        scale = (1.48 * 0.5 * (ps[k0] + ps[k1])) ** 2
        badmedian = jnp.any(use & (k1 >= end))
    else:
        mean = cell_sum(mj, mw * pull, mn) / total
        scale = cell_sum(mj, mw * pull * pull, mn) / total - mean**2
        badmedian = False
    scale = jnp.where(use, scale, 1.0)
    mav = tuple(jnp.where(use, cell_sum(mj, mw * x, mn) / total, 9999.)
                for x in mcoords)
    if do_add:
        scatter, diagnostics, _ = estimate(
            md, covfit, jnp.ones_like(md), mj, keep, ncell=mn,
            use_mad=add_mad, minimum=5)
        scatter = jnp.where(use & (scatter == 0.), 1e-12, scatter)
        added = scatter * jnp.abs(scatter)
        badadd = use & (diagnostics[0] | diagnostics[3])
    else:
        added = jnp.zeros(mn)
        badadd = jnp.zeros(mn, bool)
    errors = (jnp.any(invalid & mask & coeff_valid), badmedian,
              jnp.any(badadd), jnp.any(mask & (mix[0] < 0)))
    return scale, use, mav, added, errors


@partial(jax.jit, static_argnames=("dims", "spatial_dims"))
def neighbours(dims, spatial_dims, real_dims=None):
    """Spatial neighbours of every cell, without mixing coefficient nodes."""
    offsets = jnp.zeros((3 ** len(spatial_dims), len(dims)), dtype=jnp.int64)
    offsets = offsets.at[:, jnp.array(spatial_dims)].set(
        jnp.array(list(product((-1, 0, 1), repeat=len(spatial_dims)))))
    shape = jnp.array(dims)
    coords = jnp.stack(jnp.unravel_index(jnp.arange(prod(dims)), dims), axis=1)
    real_shape = shape if real_dims is None else jnp.asarray(real_dims)
    centres_valid = jnp.all(coords < real_shape, axis=1)
    coords = coords[:, None, :] + offsets
    valid = centres_valid[:, None] & jnp.all((coords >= 0) & (coords < real_shape), axis=2)
    strides = jnp.array([prod(dims[i+1:]) for i in range(len(dims))])
    cells = jnp.sum(jnp.clip(coords, 0, shape-1) * strides, axis=2)
    return cells, valid


@partial(jax.jit, static_argnames=("ncell",))
def centroids(cell, coords, weights, *, ncell):
    """Weighted cell positions, with SNANA's empty-cell sentinel."""
    total = cell_sum(cell, weights, ncell)
    return tuple(jnp.where(total > 0, cell_sum(cell, weights*x, ncell) /
                           jnp.where(total > 0, total, 1.), 9999.) for x in coords)


def map_precision():
    """Keep map statistics in float64, independently of the fitter's precision.

    NumPy map construction used float64 even in a single-precision fit. A local
    context preserves that accuracy without changing the caller's JAX setting.
    The experimental spelling supports the package's older JAX releases.
    """
    if hasattr(jax, "enable_x64"):
        return jax.enable_x64(True)
    from jax.experimental import enable_x64
    return enable_x64()


def in_map_precision(function):
    """Scope a host map operation, including its transfers, to float64."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        with map_precision():
            return function(*args, **kwargs)
    return wrapped


@jax.jit
def redshift_centroids(z, v0, covariance, population, node, scatter, valid, lo, hi):
    """Global correction-weighted redshifts for interpolated M0 bins."""
    cell = index(z, (lo, hi, 1., len(lo)))
    valid &= cell >= 0
    cell = jnp.where(valid, cell, 0)
    variance = covariance + scatter[node]**2 * v0**2
    weights = jnp.where(valid, population / variance, 0.)
    z = jnp.where(valid, z, 0.)
    total = cell_sum(cell, weights, len(lo))
    mean = cell_sum(cell, weights * z, len(lo)) / jnp.where(total > 0, total, 1.)
    return jnp.where(total > 0, mean, .5 * (lo + hi))
