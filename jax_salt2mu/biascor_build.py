"""One biasCor map per IDSAMPLE (SNANA prepare_biasCor, SALT2mu.c:11411).

Partitions the samples, builds the coefficient grid they share, and fills a
BiasCorMap (and muCOVscale map) for each. Reading a bias back out of a map is
biascor_map's job.
"""

from dataclasses import dataclass, replace

import numpy as np

from .map_kernels import in_map_precision

from .biascor_map import BiasCorMap, FitParBias
from . import map_kernels
from .biascor_grid import (biascor_gammadm_offset, build_biascor_cellgrid,
                           build_coeff_bins, coeff_node, sample_z_bininfo)
from .covint import sigint_coeff_grid
from .distances import (bias_parameters, bias_values, mu_at_sim, mu_variance_scaled,
                        muerr_vpec_sq, sim_mu_coef)
from .config import (MASK_BIASCOR_CELLMAP, MASK_BIASCOR_MUCOVSCALE,
                     M0FITFLAG_ZBINS_INTERP)
from .samples import assign_biascor, sample_binsize
from .tablevar import CUTBIT_zBIASCOR, CUTBIT_BIASCOR
from .mucovscale import build_mucovscale_map
from .zbins import make_zbins
from .output import output_path, write_passcuts_list


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


def build_biascor_maps(biascor_tv, samples, config, model, dust_flag: bool) -> BiasCorMaps:
    """Build one partition through the batch producer."""
    return build_biascor_partitions(biascor_tv, {None: samples}, config, model, dust_flag)[None]


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


@in_map_precision
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


@in_map_precision
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
