"""biasCor map: per-cell fit-parameter bias (SNANA makeMap_fitPar_biasCor).

For each cell of the (coefficients, redshift, host mass, model coordinates) grid, the map
holds the weighted-mean bias ``VAL = <fit - sim>`` per LC parameter, its error and
RMS. Cells with fewer than ``MINPERCELL_BIASCOR_RMS`` events borrow their RMS from
the 3x3 spatial neighbourhood. Built once per IDSAMPLE with JAX.
"""

from dataclasses import dataclass
from itertools import product
from typing import Tuple

import numpy as np

from .map_kernels import in_map_precision

from jax_snlc_core.utils.error_handling import SNANAError

from .cellgrid import BinInfo, CellGrid
from .biascor_grid import below_centroid

MINPERCELL_BIASCOR_RMS = 10  # SALT2mu.c:449 (used at :12564)


_UNDEFINED = 9999.0  # cell-average sentinel for empty cells


@dataclass(frozen=True)
class FitParBias:
    val: np.ndarray        # [ncell, n_lcpar]  weighted-mean bias
    err: np.ndarray        # [ncell, n_lcpar]
    rms: np.ndarray        # [ncell, n_lcpar]
    npercell: np.ndarray   # [ncell] int


@in_map_precision
def cell_averages(grid: CellGrid, j1d: np.ndarray, coords, weights: np.ndarray):
    """Per-cell centroid of each coordinate, weighted by ``weights`` (SNANA's
    ``WGT = WGT_pop/muerrsq``; makeMap_binavg_biasCor:13932). ``_UNDEFINED`` where
    the cell is empty. ``coords`` are the per-event spatial values (z, logmass, ...)."""
    from .map_kernels import centroids
    return [np.array(x) for x in centroids(j1d, tuple(coords), weights, ncell=grid.ncell)]


@in_map_precision
def build_fitparbias(grid: CellGrid, j1d: np.ndarray, biasval: np.ndarray,
                     wgt: np.ndarray, wgt_pop: np.ndarray,
                     spatial_dims) -> FitParBias:
    """Accumulate bias moments with the same JAX kernel used by map construction."""
    from .map_kernels import fit_moments, neighbours
    cells, inside = neighbours(grid.dims, tuple(spatial_dims))
    arrays = fit_moments(j1d, biasval, wgt, wgt_pop,
                         np.ones(len(j1d), dtype=bool), cells, inside)
    return FitParBias(*(np.array(x) for x in arrays))


NCELL_INTERP_MIN = 3    # SALT2mu.c:15558
BIASCOR_MINSUM = 10     # SALT2mu.c:563
_MAXBIAS = 20.0         # |bias| > this aborts (:15647)


@dataclass
class BiasCorMap:
    grid: CellGrid
    bininfo: Tuple[BinInfo, ...]
    fitparbias: FitParBias
    avg: Tuple[np.ndarray, ...]  # centroids in spatial grid order
    axis_names: Tuple[str, ...] = ()
    n_coeff: int = 0
    interp_logmass: bool = True
    zmax: float = 1.0


def fitparbias_events(bcmap: BiasCorMap, coeff_idx: Tuple[int, ...],
                      z, mass, lcpar, min_per_cell: int):
    """Interpolate at one coefficient node using the map's named coordinates.

    Return value, error, RMS, contributing event count, and status arrays.
    Status is 1 on success; -1 for range, -2 for width, -9 for too few cells,
    -10 for missing centre, -11 for missing upper-z coverage, -12 for population.

    Visit one corner at a time so temporary storage scales with event count,
    rather than event count times the number of interpolation corners. The
    first pass determines widths; the second accumulates weighted statistics.
    """
    bins = bcmap.bininfo[bcmap.n_coeff:]
    values = tuple(np.asarray(v, dtype=float) for v in
                   (z, mass, *(lcpar[k] for k in bcmap.axis_names)))
    n = len(values[0])
    indices = [b.index(v) for b, v in zip(bins, values)]
    code = np.where(np.all([i >= 0 for i in indices], axis=0), 1, -1)
    centres = [np.clip(i, 0, b.nbin - 1) for i, b in zip(indices, bins)]
    dims = bcmap.grid.dims[bcmap.n_coeff:]
    strides = [int(np.prod(dims[i + 1:])) for i in range(len(dims))]
    node = (int(np.ravel_multi_index(coeff_idx, bcmap.grid.dims[:bcmap.n_coeff]))
            if bcmap.n_coeff else 0)
    offset = node * int(np.prod(dims))

    def ravel(coords):
        return offset + sum(stride * coord for stride, coord in zip(strides, coords))

    centre_cells = ravel(centres)
    active = [True, bcmap.interp_logmass] + [True] * len(bcmap.axis_names)
    centre_averages = [avg[centre_cells] for avg in bcmap.avg]
    lower = [c - below_centroid(v, avg, bcmap.fitparbias.npercell[centre_cells],
                                b.binsize).astype(np.int64)
             for c, v, avg, b in zip(centres, values, centre_averages, bins)]
    upper_z = np.minimum(lower[0] + 1, bins[0].nbin - 1)
    npc = bcmap.fitparbias.npercell

    def neighbours():
        for offsets in product(*(range(2) if on else range(1) for on in active)):
            coords = [lo + delta if on else c
                      for lo, delta, on, c in zip(lower, offsets, active, centres)]
            inside = code == 1
            for coord, b in zip(coords, bins):
                inside = inside & (coord >= 0) & (coord < b.nbin)
            cells = ravel([np.clip(coord, 0, b.nbin - 1)
                           for coord, b in zip(coords, bins)])
            use = inside & (npc[cells] >= min_per_cell)
            averages = [avg[cells] for avg in bcmap.avg]
            yield coords, cells, use, averages

    widths = [np.full(n, b.binsize) for b in bins]
    for coords, cells, use, averages in neighbours():
        defined = np.logical_and.reduce([avg <= 9000 for avg in averages])
        for j, (coord, centre, avg, centre_avg) in enumerate(
                zip(coords, centres, averages, centre_averages)):
            distance = np.where(use & defined & (coord != centre),
                                np.abs(centre_avg - avg), -np.inf)
            widths[j] = np.maximum(widths[j], distance)
    for width in widths:
        code = np.where((code == 1) & ((width == 0) | (width > 9000)), -2, code)

    sum_weight = np.zeros(n)
    count = np.zeros(n, dtype=np.int64)
    nsum = np.zeros(n, dtype=np.int64)
    centre_used = np.zeros(n, dtype=bool)
    upper_used = np.zeros(n, dtype=bool)
    val = np.zeros((n, bcmap.fitparbias.val.shape[1]))
    variance = np.zeros_like(val)
    inverse_error = np.zeros_like(val)
    for coords, cells, use, averages in neighbours():
        # Match the reference order: redshift, model coordinates, mass.
        weight = np.ones(n)
        for j in (0, *range(2, len(values)), 1):
            if active[j]:
                distance = np.abs(values[j] - averages[j]) / np.where(widths[j] > 0, widths[j], 1.)
                weight *= 1. - distance
        weight = np.where(use, np.maximum(weight, 0.), 0.)
        sum_weight += weight
        count += use
        nsum += np.where(use, npc[cells], 0)
        centre_used |= use & (cells == centre_cells)
        upper_used |= use & (coords[0] == upper_z)
        w = weight[:, None]
        val += w * bcmap.fitparbias.val[cells]
        rms = bcmap.fitparbias.rms[cells]
        err = bcmap.fitparbias.err[cells]
        variance += w * rms * rms
        with np.errstate(invalid="ignore", divide="ignore"):
            inverse_error += np.where(err > 0, w / (err * err), 0.)

    capacity = np.prod([2 if on and b.nbin > 1 else 1 for b, on in zip(bins, active)])
    minimum = min(NCELL_INTERP_MIN, capacity)
    for bad, why in ((count < minimum, -9), (~centre_used, -10), (~upper_used, -11),
                     (~((nsum >= BIASCOR_MINSUM) |
                        ((nsum >= 2) & (values[0] > bcmap.zmax))), -12)):
        code = np.where((code == 1) & bad, why, code)
    ok = code == 1
    if np.any(ok & (sum_weight <= 1.e-9)):
        raise SNANAError("biasCor SUM_WGT <= 0", function_name="fitparbias_events")
    denom = np.where(sum_weight > 0, sum_weight, 1.)[:, None]
    val /= denom
    rms = np.sqrt(variance / denom)
    with np.errstate(invalid="ignore", divide="ignore"):
        err = np.sqrt((1. / inverse_error) / denom)
    if np.any(np.abs(val[ok]) > _MAXBIAS):
        raise SNANAError(f"biasCor |bias| > {_MAXBIAS}", function_name="fitparbias_events")
    return val, err, rms, nsum, code
