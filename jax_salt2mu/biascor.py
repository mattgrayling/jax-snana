"""Construct and apply simulation-derived bias corrections for each BBC sample."""

from dataclasses import dataclass, replace
from itertools import product
from typing import Optional, Tuple

import jax
import numpy as np

from jax_snlc_core.logging import SNANAError

from . import map_kernels
from .binning import (BinInfo, CellGrid, below_centroid, biascor_gammadm_offset,
                      bininfo_from_range,
                      build_biascor_cellgrid, build_coeff_bins, coeff_node,
                      make_zbins, sample_z_bininfo)
from .config import (MASK_BIASCOR_MAD, MASK_BIASCOR_MUCOVADD, lcpar_windows, map_bins,
                     MASK_BIASCOR_1DZAVG, MASK_BIASCOR_CELLMAP, MASK_BIASCOR_MUCOVSCALE,
                     M0FITFLAG_ZBINS_INTERP, output_path, parval)
from .distances import (bias_coef, bias_parameters, bias_values, mu_at_sim, mu_coef,
                        mu_variance_scaled, muerr_vpec_sq, sim_mu_coef)
from .samples import assign_biascor, sample_binsize
from .tablevar import CUTBIT_zBIASCOR, CUTBIT_BIASCOR, write_passcuts_list


@dataclass(frozen=True)
class FitParBias:
    val: np.ndarray        # [ncell, n_lcpar]  weighted-mean bias
    err: np.ndarray        # [ncell, n_lcpar]
    rms: np.ndarray        # [ncell, n_lcpar]
    npercell: np.ndarray   # [ncell] int


MINEVT_SIGINT_COMPUTE = 50  # SALT2mu.c:14220
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


@dataclass
class BiasCorMaps:
    samples: list                  # BiasCorSample per IDSAMPLE
    maps: list                     # BiasCorMap per IDSAMPLE
    mucov_maps: list               # MuCovScaleMap per IDSAMPLE (None if MUCOVSCALE off)
    zM0: np.ndarray = None         # weighted mean z per z bin, for uM0=2
    version_photometry: str = ""   # the biasCor input's header key
    sim_cosmology: dict = None     # the biasCor simulation's, from its header
    biascor_counts: tuple = ()     # total followed by counts in sample order
    biascor_nevt_total: int = None  # 1D path: NSN_PASSCUTS, not the cutmask sum
    ndim: int = 5                  # NSIM(<ndim>D-BIASCOR)
    coeff_nbin: tuple = ()         # (N_alpha, N_beta, ...) grid shape
    sigint_biascor: float = 0.0    # sigInt derived from the biasCor
    gammadm_offset: float = 0.0    # the biasCor mass-step axis midpoint (:15335)
    dust_flag: bool = False        # a BS21 dust-model biasCor (SALT2mu.c:965)


@dataclass
class BiasCorEvents:
    """Per-event quantities over the whole biasCor table, computed once and
    read by every map built from it."""
    coeff_idx: list              # each coefficient axis's bin per event
    node: np.ndarray             # flat coefficient node per event
    v: np.ndarray                # d(mu)/d(lcpar) at the SIM coefficients, [n, k]
    mucov_fit: np.ndarray        # mu variance at sigInt=0, peculiar velocity included
    mures: np.ndarray            # muresid_biasCor at the SIM coefficients
    biasval: np.ndarray          # fitted less true bias parameters, [n, k]

    def take(self, rows) -> "BiasCorEvents":
        return BiasCorEvents([idx[rows] for idx in self.coeff_idx], self.node[rows],
                             self.v[rows], self.mucov_fit[rows], self.mures[rows],
                             self.biasval[rows])


def biascor_events(biascor_tv, coeff_bins, model, config,
                   gammadm_offset) -> BiasCorEvents:
    """The quantities every biasCor map reads, over every biasCor event."""
    sim = biascor_tv.sim_truth
    coeffs = model.biascor_coeffs
    n = biascor_tv.zhd.shape[0]
    coeff_idx = [ax.index(sim[sc]) for ax, (_, sc, _) in zip(coeff_bins, coeffs)]
    v = sim_mu_coef(model, sim, n, config._u_flags)
    mucov_fit = (mu_variance_scaled(v, biascor_tv.covmat_fit, biascor_tv.zhd,
                                    config.lensing_zpar)
                 + muerr_vpec_sq(biascor_tv.zhd, biascor_tv.zmuerr))
    mures = (mu_at_sim(model, biascor_tv, sim, coeffs, config, gammadm_offset)
             - model.m0_reference - biascor_tv.mumodel)
    fitted, true = bias_values(model, biascor_tv, config)
    biasval = np.column_stack([fitted[nm] - true[nm]
                               for nm in bias_parameters(model, config)])
    return BiasCorEvents(coeff_idx=coeff_idx,
                         node=coeff_node(coeff_idx, coeff_bins, n), v=v,
                         mucov_fit=mucov_fit, mures=mures, biasval=biasval)


def _apply_biascor_grid_cut(tv, idsample, samples, model, config):
    """outside_biasCor_grid (SALT2mu.c:19239): a biasCor event outside any
    per-event axis of its sample's biasCor grid fails that axis's bit,
    using model compatibility bits or the generic bias-correction bit. Only
    events with no other cut are checked (:18967) and the first failing axis wins."""
    z = np.asarray(tv.zhd)
    zlo = np.full(z.shape, -np.inf)
    zhi = np.full(z.shape, np.inf)
    for s in samples:
        b_z = sample_z_bininfo(sample_binsize(s, config, model))
        m = idsample == s.idsample
        zlo[m], zhi[m] = b_z.lo[0], b_z.hi[-1]

    cand = tv.cutmask
    bad_any = (z < zlo) | (z > zhi)
    tv.cutbits[cand & bad_any] |= 1 << CUTBIT_zBIASCOR
    grid_bits = {name: pair[1] for name, pair in model.bbc_cut_bits.items()}
    for name, (vmin, vmax, _) in model.biascor_binning.items():
        vals = np.asarray(tv.lcpar[name])
        bad = (vals < vmin) | (vals > vmax)
        bit = grid_bits.get(name, CUTBIT_BIASCOR)
        tv.cutbits[cand & ~bad_any & bad] |= 1 << bit
        bad_any = bad_any | bad


def build_biascor_partitions(biascor_tv, partitions, config, model, dust_flag):
    """Build known partitions together, sharing event quantities and equal maps."""
    coeff_bins = build_coeff_bins(config, model, biascor_tv.sim_truth)
    gdm_offset = biascor_gammadm_offset(model, coeff_bins, config)
    events = biascor_events(biascor_tv, coeff_bins, model, config, gdm_offset)
    nc = len(coeff_bins)
    selected = set(config.idsample_select)
    do_mucov = bool(config.opt_biascor & MASK_BIASCOR_MUCOVSCALE)
    variables, scatter, requests = {}, {}, {}
    for key, samples in partitions.items():
        tv = replace(biascor_tv, cutbits=biascor_tv.cutbits.copy())
        tv.idsample = assign_biascor(tv, samples, config, model)
        if config.opt_biascor & MASK_BIASCOR_CELLMAP:
            _apply_biascor_grid_cut(tv, tv.idsample, samples, model, config)
        in_select = np.isin(tv.idsample, list(selected)) if selected else None
        sig_grid = sigint_coeff_grid(tv, events, coeff_bins, config, event_mask=in_select)
        variables[key], scatter[key] = tv, sig_grid
        for sample in samples:
            rows = np.flatnonzero(tv.idsample == sample.idsample)
            mask = tv.cutmask[rows]
            if selected and sample.idsample not in selected:
                mask = np.zeros_like(mask)
            requests[key, sample.idsample] = sample, rows, mask, sig_grid

    capacities = {}
    for sample, rows, _, _ in requests.values():
        family = (sample.surveygroup, sample.fieldgroup, sample.is_photoz, sample.stringopt)
        z_count = sample_z_bininfo(sample_binsize(sample, config, model)).nbin
        previous = capacities.get(family, (0, 0))
        capacities[family] = max(previous[0], z_count), max(previous[1], len(rows))

    groups = []
    for key, (sample, rows, mask, sig_grid) in requests.items():
        group = next((keys for keys in groups
                      if requests[keys[0]][0] == sample
                      and np.array_equal(requests[keys[0]][1], rows)
                      and np.array_equal(requests[keys[0]][2], mask)
                      and np.array_equal(requests[keys[0]][3], sig_grid)), None)
        if group is None:
            groups.append([key])
        else:
            group.append(key)

    built = {}
    for keys in groups:
        key = keys[0]
        sample, rows, mask, sig_grid = requests[key]
        grid, bins = build_biascor_cellgrid(config, model, coeff_bins,
                                            sample_binsize(sample, config, model))
        tv, sample_events = variables[key[0]].take(rows), events.take(rows)
        family = (sample.surveygroup, sample.fieldgroup, sample.is_photoz, sample.stringopt)
        z_capacity, row_capacity = capacities[family]
        fm = _build_one_map(tv, sample_events, model, grid, bins, sig_grid,
                            mask, config.zmax, config, z_capacity=z_capacity, row_capacity=row_capacity)
        mc = None
        if do_mucov:
            mc = build_mucovscale_map(
                tv, sample_events, fm, coeff_bins, sig_grid, mask,
                bins[nc], bins[nc + 1], model, config,
                config.min_per_cell_biascor, z_capacity=z_capacity, row_capacity=row_capacity)
        for request in keys:
            built[request] = fm, mc

    result = {}
    for key, samples in partitions.items():
        tv, sig_grid = variables[key], scatter[key]
        zM0 = (calc_zM0_biascor(tv, events, sig_grid, make_zbins(config), config)
               if config.uM0 == M0FITFLAG_ZBINS_INTERP else None)
        result[key] = BiasCorMaps(
            samples=samples,
            maps=[built[key, sample.idsample][0] for sample in samples],
            mucov_maps=[built[key, sample.idsample][1] for sample in samples],
            zM0=zM0, dust_flag=dust_flag,
            biascor_counts=sample_counts(tv.idsample, tv.cutmask, samples),
            coeff_nbin=tuple(b.nbin for b in coeff_bins),
            sigint_biascor=float(np.mean(sig_grid)), gammadm_offset=gdm_offset,
            version_photometry=tv.version_photometry, sim_cosmology=tv.sim_cosmology)
        if config.write_biascor:
            write_passcuts_list(output_path(config, "_BIASCOR.FITRES"), "BIASCOR", tv, tv.idsample)
    return result


def sample_counts(idsample, keep, samples):
    """Passing total and per-sample counts, without retaining event arrays."""
    return (int(keep.sum()),
            *(int((keep & (idsample == sample.idsample)).sum()) for sample in samples))


@jax.enable_x64(True)
def calc_zM0_biascor(biascor_tv, events, sig_grid, zbins, config):
    """Weighted mean z per z bin over the biasCor (calc_zM0_biasCor, SALT2mu.c:14774).

    WGT is WGT_biasCor(opt=1), the population weight over muerrsq at COVFIT+COVINT+ZMUERR
    (:12750). Bins with no biasCor fall back to the bin centre (:14820). Global across
    IDSAMPLEs, matching the single INFO_BIASCOR.zM0 array.
    """

    keep = biascor_tv.cutmask & np.all([i >= 0 for i in events.coeff_idx], axis=0)

    return np.array(map_kernels.redshift_centroids(
        biascor_tv.zhd, events.v[:, 0], events.mucov_fit,
        biascor_tv.sim_truth["SIM_WGT_POP"], events.node, sig_grid, keep,
        zbins.lo, zbins.hi))


@jax.enable_x64(True)
def _build_one_map(biascor_tv, events, model, grid, bins, sig_grid, event_mask,
                   zmax, config, *, z_capacity=None, row_capacity=None):
    """Accumulate one IDSAMPLE's FITPARBIAS map (makeMap_fitPar_biasCor)."""
    nc = len(model.biascor_coeffs)
    names = tuple(model.biascor_binning)
    coords = (biascor_tv.zhd, biascor_tv.logmass,
              *(biascor_tv.lcpar[name] for name in names))
    spatial = bins[nc:]
    z_capacity = spatial[0].nbin if z_capacity is None else z_capacity
    row_capacity = len(event_mask) if row_capacity is None else row_capacity
    extra = row_capacity - len(event_mask)
    if extra < 0:
        raise ValueError("bias-correction row capacity is too small")
    bounds = map_kernels.map_bounds(spatial, z_capacity)
    dims = (*grid.dims[:nc], z_capacity, *grid.dims[nc + 1:])
    neighbours, inside = map_kernels.neighbours(
        dims, tuple(range(nc, len(bins))), grid.dims)
    valid_coeff = np.ones(len(event_mask), dtype=bool)
    for index in events.coeff_idx:
        valid_coeff &= index >= 0
    arrays = map_kernels.fit_map(
        tuple(map_kernels.pad_map_rows(x, row_capacity) for x in coords), bounds,
        map_kernels.pad_map_rows(events.node, row_capacity),
        map_kernels.pad_map_rows(valid_coeff, row_capacity),
        map_kernels.pad_map_rows(events.v, row_capacity),
        map_kernels.pad_map_rows(events.mucov_fit, row_capacity, 1.),
        map_kernels.pad_map_rows(events.biasval, row_capacity),
        map_kernels.pad_map_rows(biascor_tv.sim_truth["SIM_WGT_POP"], row_capacity), sig_grid,
        map_kernels.pad_map_rows(event_mask, row_capacity), neighbours, inside, config.sigma_cell_biascor,
        dims=dims[nc:],
        active=(True, config.interp_biascor_logmass > 0, *([True] * len(names))))
    val, err, rms, count, avg = arrays
    # Maps cross the host boundary once; fitting consumes their interpolated values.
    return BiasCorMap(
        grid=grid, bininfo=bins,
        fitparbias=FitParBias(*(map_kernels.trim_map_cells(x, dims, nc, spatial[0].nbin)
                                 for x in (val, err, rms, count))),
        avg=tuple(map_kernels.trim_map_cells(x, dims, nc, spatial[0].nbin) for x in avg),
        axis_names=names,
        n_coeff=nc, zmax=zmax, interp_logmass=config.interp_biascor_logmass > 0)


def _coeff_nodes(bcmap):
    return list(product(*[range(bcmap.bininfo[j].nbin) for j in range(bcmap.n_coeff)]))


def precompute_fitparbias_nodes(data_tv, maps, min_per_cell: int,
                                idsample=None):
    """Per event, the LC-parameter bias at every coefficient node and an all-nodes
    valid mask (storeBias_data), ``[n, N_node, k]``. ``idsample`` overrides
    ``data_tv.idsample`` (e.g. for CC-prior events)."""
    m0 = maps.maps[0]
    nodes = _coeff_nodes(m0)
    N, k = len(nodes), m0.fitparbias.val.shape[1]
    n = data_tv.size
    if idsample is None:
        idsample = data_tv.idsample
    z, mass = data_tv.zhd, data_tv.logmass

    val = np.zeros((n, N, k))
    err = np.zeros((n, N, k))
    nevt = np.zeros((n, N), dtype=np.int64)
    ok = np.zeros(n, dtype=bool)
    for isamp in np.unique(idsample):
        if isamp < 0:                    # excluded from every IDSAMPLE
            continue
        sel = np.nonzero(idsample == isamp)[0]
        bcmap = maps.maps[isamp]
        allok = np.ones(sel.shape, dtype=bool)
        for inode, node in enumerate(nodes):
            v, e, _, ns, code = fitparbias_events(
                bcmap, node, z[sel], mass[sel],
                {name: data_tv.lcpar[name][sel] for name in bcmap.axis_names}, min_per_cell)
            good = code == 1
            rows = sel[good]
            val[rows, inode] = v[good]
            err[rows, inode] = e[good]
            nevt[rows, inode] = ns[good]
            allok &= good
        ok[sel] = allok
    return val, nevt, err, ok


def precompute_mucovscale(data_tv, maps):
    """Per data event, muCOVscale (and muCOVADD, if built) at every coefficient node
    (get_muCOVcorr) and an all-nodes valid mask, ``[n, N_node]``."""
    m0 = maps.mucov_maps[0]
    nodes = _coeff_nodes(m0)
    N, n = len(nodes), data_tv.size
    z, mass = data_tv.zhd, data_tv.logmass
    has_covadd = m0.covadd is not None
    val = np.ones((n, N))
    cadd = np.zeros((n, N)) if has_covadd else None
    ok = np.zeros(n, dtype=bool)
    for isamp in np.unique(data_tv.idsample):
        if isamp < 0:                    # excluded from every IDSAMPLE
            continue
        sel = np.nonzero(data_tv.idsample == isamp)[0]
        mc = maps.mucov_maps[isamp]
        allok = np.ones(sel.shape, dtype=bool)
        for inode, node in enumerate(nodes):
            scale, ca, good = mucovscale_events(
                mc, node, z[sel], mass[sel],
                {name: data_tv.lcpar[name][sel] for name in mc.axis_names})
            rows = sel[good]
            val[rows, inode] = scale[good]
            if has_covadd:
                cadd[rows, inode] = ca[good]
            allok &= good
        ok[sel] = allok
    return val, ok, cadd


def precompute_biascor(data_tv, maps, min_per_cell):
    """All cell-map corrections before data-specific exemptions and cuts."""
    val, nevt, err, ok = precompute_fitparbias_nodes(data_tv, maps, min_per_cell)
    if maps.mucov_maps[0] is None:
        scale, added = np.ones(val.shape[:2]), None
    else:
        scale, cov_ok, added = precompute_mucovscale(data_tv, maps)
        ok = ok & cov_ok
    return val, nevt, err, scale, added, ok


def biascor_1d_grids(biascor_tv, samples, config, model):
    """Each sample's frozen 1D distance bias from the biasCor, on the data's
    partition ``samples``: ``(zavg, mubias, zlo, zhi)`` over its populated z
    bins, or None for a sample with no bias (unselected or unpopulated). Also
    the biasCor events' IDSAMPLE, their cutmask and how many were in range.
    Nothing here reads a data event, so cells sharing settings, biasCor and
    partition share it; :func:`biascor_1d_data` applies it to one cell's data."""

    # muBias is frozen at the fit's initial parval (SALT2mu.c:22354)
    nuis = {n: parval(config, n) for n in model.nuisance_spec}
    m0ref = model.m0_reference
    sigmb = config.sigint

    use_avg = (config.opt_biascor & MASK_BIASCOR_1DZAVG) > 0

    bc_idsample = assign_biascor(biascor_tv, samples, config, model)

    names = model.lcpar_names
    lcpar = {n: biascor_tv.lcpar[n] for n in names}
    host = {"z": biascor_tv.zhd, "gamma_var": biascor_tv.logmass}
    mu_fit = np.asarray(model.compute_mu(
        lcpar, nuis, host, config._u_flags, apply_mass_step=False)) - m0ref
    bias = mu_fit - np.asarray(biascor_tv.mumodel)      # mu_fit - SIM_MUz

    if use_avg:
        wgt_all = np.ones(biascor_tv.size)
    else:
        v = np.tile(mu_coef(model, nuis, config._u_flags), (biascor_tv.size, 1))
        muerrsq = (mu_variance_scaled(v, biascor_tv.covmat_fit, biascor_tv.zhd,
                                      config.lensing_zpar,
                                      sigint_sq=sigmb * sigmb)
                   + muerr_vpec_sq(biascor_tv.zhd, biascor_tv.zmuerr))
        wgt_all = 1.0 / muerrsq

    # each sample bins on its own RANGE_REDSHIFT / BINSIZE_REDSHIFT (:11243)
    binsize = [sample_binsize(s, config, model) for s in samples]
    zc, snr = biascor_tv.zhd, biascor_tv.snrmax
    select = set(config.idsample_select)

    in_range = np.zeros(biascor_tv.size, dtype=bool)
    for s, bs in zip(samples, binsize):
        zlo0, zhi0 = bs.zrange
        in_range |= ((bc_idsample == s.idsample) & (zc >= zlo0) & (zc <= zhi0))
    in_range &= biascor_tv.cutmask

    # the high-SNR reference is global, not per-sample (SALT2mu.c:12160, :12221)
    hisnr = in_range & (snr > config.snrmin_sigint_biascor)
    w_all = np.where(in_range, wgt_all, 0.0)
    sw_hi = w_all[hisnr].sum()
    mu_hisnr = (w_all[hisnr] * bias[hisnr]).sum() / sw_hi if sw_hi > 0 else 0.0

    grids = []
    for s, bs in zip(samples, binsize):
        # DOFLAG_SELECT=0: the sample's data are cut, not corrected (:12259)
        if select and s.idsample not in select:
            grids.append(None)
            continue
        zlo0, zhi0 = bs.zrange
        zbin = bs.zbin
        nzbin = int(np.ceil((zhi0 - zlo0) / zbin))
        bmask = in_range & (bc_idsample == s.idsample)
        iz = np.clip(((zc - zlo0) / zbin).astype(int), 0, nzbin - 1)
        w = np.where(bmask, wgt_all, 0.0)
        sumwgt = np.bincount(iz, weights=w, minlength=nzbin)
        summu = np.bincount(iz, weights=w * bias, minlength=nzbin)

        pop = sumwgt > 1e-9
        mubias = np.zeros(nzbin)
        mubias[pop] = summu[pop] / sumwgt[pop] - mu_hisnr
        izs = np.nonzero(pop)[0]
        if izs.size == 0:
            grids.append(None)
            continue
        izmin, izmax = izs[0], izs[-1]
        zavg = zlo0 + zbin * (np.arange(nzbin) + 0.5)
        grids.append((zavg[izmin:izmax + 1], mubias[izmin:izmax + 1],
                      zlo0 + zbin * izmin, zlo0 + zbin * (izmax + 1)))

    return grids, bc_idsample, biascor_tv.cutmask, int(in_range.sum())


def biascor_1d_data(grids, samples, data_tv, data_idsample):
    """Return ``(muBias_zinterp, accept)`` for one cell's data events: each
    event's bias interpolated on its sample's grid (:func:`biascor_1d_grids`),
    and the mask of events with one."""
    muBias_data = np.zeros(data_tv.size)
    accept = np.zeros(data_tv.size, dtype=bool)
    zd = data_tv.zhd
    for s, grid in zip(samples, grids):
        if grid is None:
            continue
        zavg, mubias, zlo, zhi = grid
        inrange = (data_idsample == s.idsample) & (zd >= zlo) & (zd <= zhi)
        muBias_data[inrange] = np.interp(zd[inrange], zavg, mubias)
        accept |= inrange
    return muBias_data, accept


@jax.enable_x64(True)
def sigint_coeff_grid(biascor_tv, events, coeff_bins, config,
                      event_mask=None) -> np.ndarray:
    """sigInt per coefficient node from the SNR>snrmin subset, flat ``[N_node]``. A
    fixed ``sigint_biascor`` (>=0) overrides the computation. ``event_mask`` narrows
    the biasCor events considered, e.g. to the selected IDSAMPLEs."""
    n_node = int(np.prod([b.nbin for b in coeff_bins])) if coeff_bins else 1
    if config.sigint_biascor >= 0.0:
        return np.full(n_node, config.sigint_biascor)

    tv = biascor_tv
    # selection without the biasCor z cut, keeping the global z cut, so
    # high-z-only fits still find SNR>snrmin events (SALT2mu.c:14326-14335)
    passing = (tv.cutbits & ~(1 << CUTBIT_zBIASCOR)) == 0
    # SNR membership depends on the correction catalogue and configuration,
    # not the data realisation. Project this fixed population once; keep cuts
    # as a mask so the iterative kernel does not scan millions of low-SNR rows.
    rows = np.flatnonzero(tv.snrmax > config.snrmin_sigint_biascor)
    sel = passing[rows]
    if event_mask is not None:
        sel = sel & event_mask[rows]
    if not len(rows):
        raise SNANAError("No biasCor events above snrmin_sigint_biascor; "
                         f"sigInt needs at least {MINEVT_SIGINT_COMPUTE}",
                         function_name="sigint_coeff_grid")

    result, diagnostics, counts = map_kernels.estimate_intrinsic_scatter(
        events.mures[rows], events.mucov_fit[rows], tv.sim_truth["SIM_WGT_POP"][rows],
        events.node[rows], sel, ncell=n_node, minimum=MINEVT_SIGINT_COMPUTE)
    counts = np.asarray(counts)
    for i in np.flatnonzero(counts < MINEVT_SIGINT_COMPUTE):
        raise SNANAError(
            f"coefficient node {i} has {counts[i]} biasCor events above "
            f"snrmin_sigint_biascor={config.snrmin_sigint_biascor}; "
            f"sigInt needs at least {MINEVT_SIGINT_COMPUTE}",
            function_name="sigint_coeff_grid")
    for condition, message in zip(diagnostics, (
            "Invalid MUCOV in intrinsic-scatter sample",
            "Cannot compute sigint because STD < AVG_MUERR",
            "Cannot compute sigint because sig trial < -0.300000",
            "ONE NOT CONTAINED by rmsPull array")):
        if np.any(np.asarray(condition)):
            raise SNANAError(message, function_name="sigint_coeff_grid")
    return np.array(result)


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
            for name, count in map_bins(model.mucovscale_bins, config.mucovscale_bins).items()]
    bins = (*coeff_bins, bz, bm, *axes)
    return CellGrid(bins=bins), bins


@jax.enable_x64(True)
def build_mucovscale_map(biascor_tv, events, fitpar_map, coeff_bins, sig_grid,
                         event_mask, bz, bm, model, config,
                         min_per_cell, *, z_capacity=None, row_capacity=None) -> MuCovScaleMap:
    """MUCOVSCALE = Var(pull) per cell, pull = (muresid - muBias)/muErr over the
    biasCor events selected by ``event_mask``."""
    grid, bins = build_mucovscale_grid(coeff_bins, bz, bm, model, config)
    nc = len(model.biascor_coeffs)
    names = tuple(map_bins(model.mucovscale_bins, config.mucovscale_bins))
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
