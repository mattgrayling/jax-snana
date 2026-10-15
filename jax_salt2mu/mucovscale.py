"""muCOVscale map + application (SNANA makeMap_sigmu_biasCor:12205,
get_muCOVcorr:14836).

Scales each data event's mu-error by the biasCor sample's pull variance per
(coefficients, redshift, host mass, declared model coordinates) cell. Applied as
``muerrsq -> (muerrsq - vpec^2) * muCOVscale + vpec^2``.
"""

from dataclasses import dataclass
from itertools import product
from typing import Optional, Tuple

import numpy as np

from .map_kernels import in_map_precision

from jax_snlc_core.utils.error_handling import SNANAError

from .biascor_grid import bininfo_from_range
from .cellgrid import BinInfo, CellGrid
from .config import MASK_BIASCOR_MAD, MASK_BIASCOR_MUCOVADD, lcpar_windows, map_bins
from .distances import bias_coef, bias_parameters

MINPERCELL_MUCOVSCALE = 5   # SALT2mu.c:450


@dataclass
class MuCovScaleMap:
    grid: CellGrid
    bininfo: Tuple[BinInfo, ...]
    scale: np.ndarray          # [ncell] MUCOVSCALE (1.0 where unused)
    use: np.ndarray            # [ncell] bool
    avg: Tuple[np.ndarray, ...]  # centroids in spatial grid order
    axis_names: Tuple[str, ...] = ()
    n_coeff: int = 0
    interp_logmass: bool = False
    covadd: Optional[np.ndarray] = None  # [ncell] MUCOVADD sigint*|sigint| (None if off)


def build_mucovscale_grid(coeff_bins, bz, bm, model, config):
    """Error-map coordinates are independent of the bias-map coordinates."""
    windows = lcpar_windows(model, config)
    axes = [bininfo_from_range(*windows[name],
                              (windows[name][1] - windows[name][0]) / count, name)
            for name, count in map_bins(model, config, "mucovscale_bins").items()]
    bins = (*coeff_bins, bz, bm, *axes)
    return CellGrid(bins=bins), bins


@in_map_precision
def build_mucovscale_map(biascor_tv, events, fitpar_map, coeff_bins, sig_grid,
                         event_mask, bz, bm, model, config,
                         min_per_cell, *, z_capacity=None, row_capacity=None) -> MuCovScaleMap:
    """MUCOVSCALE = Var(pull) per cell, pull = (muresid - muBias)/muErr over the
    biasCor events selected by ``event_mask``."""
    from . import map_kernels

    grid, bins = build_mucovscale_grid(coeff_bins, bz, bm, model, config)
    nc = len(model.biascor_coeffs)
    names = tuple(map_bins(model, config, "mucovscale_bins"))
    coords = (biascor_tv.zhd, biascor_tv.logmass,
              *(biascor_tv.lcpar[name] for name in fitpar_map.axis_names))
    mcoords = (biascor_tv.zhd, biascor_tv.logmass,
               *(biascor_tv.lcpar[name] for name in names))
    z_capacity = bz.nbin if z_capacity is None else z_capacity
    row_capacity = len(event_mask) if row_capacity is None else row_capacity
    extra = row_capacity - len(event_mask)
    if extra < 0:
        raise ValueError("bias-correction row capacity is too small")
    if row_capacity == 0:
        return MuCovScaleMap(
            grid, bins, np.ones(grid.ncell), np.zeros(grid.ncell, dtype=bool),
            tuple(np.full(grid.ncell, 9999.) for _ in mcoords), names, nc,
            config.interp_biascor_logmass > 0,
            np.zeros(grid.ncell) if config.opt_biascor & MASK_BIASCOR_MUCOVADD else None)
    bounds = map_kernels.map_bounds(fitpar_map.bininfo[nc:], z_capacity)
    mbounds = map_kernels.map_bounds(bins[nc:], z_capacity)
    dims = (*fitpar_map.grid.dims[:nc], z_capacity, *fitpar_map.grid.dims[nc + 1:])
    mdims = (*grid.dims[:nc], z_capacity, *grid.dims[nc + 1:])
    averages = tuple(map_kernels.pad_map_cells(x, fitpar_map.grid.dims, nc, z_capacity, 9999.)
                     for x in fitpar_map.avg)
    values = map_kernels.pad_map_cells(fitpar_map.fitparbias.val, fitpar_map.grid.dims,
                                      nc, z_capacity, 999.)
    counts = map_kernels.pad_map_cells(fitpar_map.fitparbias.npercell, fitpar_map.grid.dims,
                                      nc, z_capacity, 0)
    valid_coeff = np.ones(len(event_mask), dtype=bool)
    for index in events.coeff_idx:
        valid_coeff &= index >= 0
    scale, use, avg, added, errors = map_kernels.error_map(
        tuple(map_kernels.pad_map_rows(x, row_capacity) for x in coords),
        tuple(map_kernels.pad_map_rows(x, row_capacity) for x in mcoords), bounds, mbounds,
        map_kernels.pad_map_rows(events.node, row_capacity),
        map_kernels.pad_map_rows(valid_coeff, row_capacity),
        map_kernels.pad_map_rows(bias_coef(events.v, bias_parameters(model, config)), row_capacity),
        map_kernels.pad_map_rows(events.mucov_fit, row_capacity, 1.),
        map_kernels.pad_map_rows(events.mures, row_capacity),
        map_kernels.pad_map_rows(biascor_tv.sim_truth["SIM_WGT_POP"], row_capacity), sig_grid,
        map_kernels.pad_map_rows(event_mask, row_capacity), averages, values, counts,
        fitpar_map.zmax, min_per_cell, dims=dims[nc:], mdims=mdims[nc:],
        active=(True, fitpar_map.interp_logmass, *([True] * len(fitpar_map.axis_names))),
        do_mad=bool(config.opt_biascor & MASK_BIASCOR_MAD),
        do_add=bool(config.opt_biascor & MASK_BIASCOR_MUCOVADD),
        add_mad=not config.restore_sn_unite)
    for failed, message in zip(errors, (
            "Invalid bias interpolation weights or value",
            "muCOVscale weighted median undefined",
            "ONE NOT CONTAINED by rmsPull array",
            "biasCor event is outside the muCOVscale z grid")):
        if bool(failed):
            raise SNANAError(message, function_name="build_mucovscale_map")
    return MuCovScaleMap(
        grid, bins, map_kernels.trim_map_cells(scale, mdims, nc, bz.nbin),
        map_kernels.trim_map_cells(use, mdims, nc, bz.nbin),
        tuple(map_kernels.trim_map_cells(x, mdims, nc, bz.nbin) for x in avg),
        names, nc, config.interp_biascor_logmass > 0,
        map_kernels.trim_map_cells(added, mdims, nc, bz.nbin)
        if config.opt_biascor & MASK_BIASCOR_MUCOVADD else None)


def mucovscale_events(m: MuCovScaleMap, coeff_idx: Tuple[int, ...], z, mass, lcpar):
    """Interpolate scale and additive correction over named nearest neighbours.

    Return (scale, covadd, ok). Invalid events have no usable correction.
    """
    values = tuple(np.asarray(v, dtype=float) for v in
                   (z, mass, *(lcpar[name] for name in m.axis_names)))
    bins = m.bininfo[m.n_coeff:]
    indices = [b.index(v, opt=2 if j else 0)
               for j, (b, v) in enumerate(zip(bins, values))]
    offsets = list(product(*[(-1, 0, 1) if j != 1 or m.interp_logmass else (0,)
                             for j in range(len(bins))]))
    offsets = np.asarray(offsets)
    neighbours = [idx[:, None] + offsets[:, j] for j, idx in enumerate(indices)]
    inside = np.logical_and.reduce([(idx >= 0) & (idx < b.nbin)
                                    for idx, b in zip(neighbours, bins)])
    prefix = tuple(np.full(neighbours[0].shape, q) for q in coeff_idx)
    cells = np.ravel_multi_index(
        (*prefix, *(np.clip(idx, 0, b.nbin - 1) for idx, b in zip(neighbours, bins))),
        m.grid.dims)
    used = inside & m.use[cells]
    centre = used[:, np.all(offsets == 0, axis=1)].any(axis=1)
    distances = [np.abs(v[:, None] - avg[cells]) / b.binsize
                 if j != 1 or m.interp_logmass else np.zeros_like(cells, dtype=float)
                 for j, (v, avg, b) in enumerate(zip(values, m.avg, bins))]
    weight = np.ones_like(cells, dtype=float)
    within = used.copy()
    for distance in distances:
        within &= distance <= 0.99
        weight *= 1.0 - distance
    weight = np.where(within, weight, 0.0)
    sum_wgt = weight.sum(axis=1)
    ok = (indices[0] >= 0) & centre & (sum_wgt > 0.01)
    denom = np.where(ok, sum_wgt, 1.0)
    scale = (weight * m.scale[cells]).sum(axis=1) / denom
    covadd = ((weight * m.covadd[cells]).sum(axis=1) / denom
              if m.covadd is not None else None)
    return scale, covadd, ok
