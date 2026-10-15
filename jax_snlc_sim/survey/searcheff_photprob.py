"""SEARCHEFF PHOTPROB: per-observation photometric classification probability.

A CDF-based probability per observation from SNR, surface brightness, host
galaxy magnitude and/or redshift; REDUCED_CORR correlates the draws across
observations through one shared covariance (sntools_trigger.c:2679). The
MAPNAME_PHOTPROB blocks are read by
:mod:`jax_snlc_sim.survey.searcheff_pipeline`, which reads the whole
SEARCHEFF_PIPELINE_EFF_FILE in one pass, and turned into a PhotoprobMap by
``build_photprob_map`` here.

SNANA reference: sntools_trigger.c lines 465-691 (reader), 2605-2897 (evaluator)
"""

from dataclasses import dataclass, replace
from typing import List, Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm as jax_norm

from jax_snlc_core.photometry.filter_system import filter_band
from jax_snlc_core.utils.error_handling import SNANAError
from jax_snlc_sim.survey.searcheff_map import (field_name_mask,
                                               interp_nd_regular,
                                               split_fieldlist)


# Variable type indices matching SNANA sntools_trigger.h:96-101
IVARABS_SNR = 0
IVARABS_LOGSNR = 1
IVARABS_SBMAG = 2
IVARABS_REDSHIFT = 3
IVARABS_GALMAG = 4

VARNAME_TO_IVARABS = {
    'SNR': IVARABS_SNR,
    'LOGSNR': IVARABS_LOGSNR,
    'SBMAG': IVARABS_SBMAG,
    'GALMAG': IVARABS_GALMAG,
    'REDSHIFT': IVARABS_REDSHIFT,
}


@dataclass
class PhotoprobMap:
    """One PHOTPROB map from SEARCHEFF_PIPELINE_EFF_FILE."""
    name: str
    varnames: List[str]
    ivarabs: List[int]            # absolute variable type indices
    nvar_map: int
    nfun_cdf: int
    grid_axes: List[np.ndarray]   # unique values per dimension
    grid_cdf: np.ndarray          # (*axis_sizes, nfun_cdf), cumulative CDF weights
    cdf_bins: np.ndarray          # [0, 1/K, ..., 1], length K+1
    valmin: np.ndarray            # per-variable clamp range
    valmax: np.ndarray
    fieldlist: str = 'ALL'
    filterlist: str = 'ALL'
    require_detection: bool = False
    cutval: float = -999.0        # reject obs with PHOTPROB <= cutval
    reduced_corr: float = 0.0     # inter-observation correlation
    # per SIMLIB entry, from `fieldlist`; None for 'ALL'
    fieldlist_mask: Optional[np.ndarray] = None
    filter_mask: Optional[np.ndarray] = None


def build_photprob_map(info: dict, data_rows: list) -> PhotoprobMap:
    """Build a PhotoprobMap from parsed data.

    The grid is filled by each row's coordinates rather than assuming row
    order; a duplicate coordinate, a missing cell, a negative weight or a
    zero-mass weight row is malformed input and fatal.
    """
    name = info['name']
    nvar_map = info['nvar_map']
    nfun_cdf = info['nfun_cdf']
    data = np.array(data_rows)
    var_data = data[:, :nvar_map]
    wgt_data = data[:, nvar_map:]

    if np.any(wgt_data < 0.0):
        raise SNANAError(
            f"PHOTPROB map {name!r} has negative PHOTPROB_WGTLIST entries.",
            function_name='build_photprob_map')
    row_sums = wgt_data.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise SNANAError(
            f"PHOTPROB map {name!r} has ROW(s) whose PHOTPROB_WGTLIST sums to "
            f"zero; the CDF is undefined.",
            function_name='build_photprob_map')
    # Convert weights to cumulative CDF (SNANA LOAD_PHOTPROB_CDF)
    cumulative = np.cumsum(wgt_data, axis=1) / row_sums

    # Track VALMIN/VALMAX per variable (sntools_trigger.c:631-636)
    valmin = var_data.min(axis=0)
    valmax = var_data.max(axis=0)

    grid_axes = [np.unique(var_data[:, d]) for d in range(nvar_map)]
    shape = tuple(len(ax) for ax in grid_axes)
    idx_per_dim = tuple(
        np.searchsorted(grid_axes[d], var_data[:, d]) for d in range(nvar_map))
    filled = np.zeros(shape, dtype=bool)
    filled[idx_per_dim] = True
    flat_cell = np.ravel_multi_index(idx_per_dim, shape)
    if len(np.unique(flat_cell)) < len(data_rows):
        raise SNANAError(
            f"PHOTPROB map {name!r} defines the same grid coordinate more "
            f"than once.",
            function_name='build_photprob_map')
    if not filled.all():
        raise SNANAError(
            f"PHOTPROB map {name!r} is missing "
            f"{int(filled.size - filled.sum())} grid cell(s) of the "
            f"{shape} outer-product grid.",
            function_name='build_photprob_map')

    grid_cdf = np.zeros(shape + (nfun_cdf,), dtype=np.float64)
    grid_cdf[idx_per_dim] = cumulative
    cdf_bins = np.linspace(0.0, 1.0, nfun_cdf + 1)

    return PhotoprobMap(
        name=info['name'],
        varnames=info['varnames'],
        ivarabs=info['ivarabs'],
        nvar_map=nvar_map,
        nfun_cdf=nfun_cdf,
        grid_axes=grid_axes,
        grid_cdf=grid_cdf,
        cdf_bins=cdf_bins,
        fieldlist=info['fieldlist'],
        filterlist=info['filterlist'],
        require_detection=info['require_detection'],
        cutval=info['cutval'],
        reduced_corr=info['reduced_corr'],
        valmin=valmin,
        valmax=valmax,
    )


def resolve_photprob_fieldlist_indices(maps, simlib_field_names):
    """Resolve each PHOTPROB map's fieldlist string to a per-SIMLIB-entry bool mask.

    fieldlist format: 'ALL', or '+' / ',' separated list (e.g. 'X1+X2,C3').
    Populates `fieldlist_mask` in place; left as None when fieldlist='ALL'.
    The per-SN `field_idx` returned by the scheduler indexes into this mask
    at call time: `fieldlist_mask[field_idx]` gives the per-SN match.

    Every SIMLIB entry whose FIELD is listed matches, as the per-event
    MATCH_SEARCHEFF_FIELD does (sntools_trigger.c:2576-2579). Aborts on a
    field name the SIMLIB does not carry.
    """
    for m in maps:
        if m.fieldlist == 'ALL' or not m.fieldlist:
            m.fieldlist_mask = None
            continue
        m.fieldlist_mask = field_name_mask(split_fieldlist(m.fieldlist),
                                           simlib_field_names,
                                           'PHOTPROB map FIELDLIST')


def prepare_photprob(maps, filter_names):
    """Pre-resolve filter/field matching for JIT-compiled PHOTPROB evaluation.

    Called ONCE at setup time (Python). ``maps`` is the List[PhotoprobMap]
    from read_pipeline_eff_file(), with `fieldlist_mask` resolved by
    resolve_photprob_fieldlist_indices; ``filter_names`` is the list of
    filter names (e.g., ['LSST-u', ...]). Returns prepared maps with JAX arrays.

    Raises `SNANAError` when more than one map matches a (field, filter)
    observation (setObs_for_PHOTPROB, sntools_trigger.c:2589-2594).
    """
    filt_chars = [filter_band(fn) for fn in filter_names]

    prepared = []
    for m in maps:
        if m.filterlist == 'ALL':
            filter_mask = np.ones(len(filter_names), dtype=bool)
        else:
            filter_mask = np.zeros(len(filter_names), dtype=bool)
            for fi, fc in enumerate(filt_chars):
                if fc in m.filterlist:
                    filter_mask[fi] = True

        prepared.append(replace(
            m,
            grid_axes=[jnp.asarray(ax, dtype=jnp.float32) for ax in m.grid_axes],
            grid_cdf=jnp.asarray(m.grid_cdf, dtype=jnp.float32),
            cdf_bins=jnp.asarray(m.cdf_bins, dtype=jnp.float32),
            filter_mask=jnp.asarray(filter_mask),
            valmin=jnp.asarray(m.valmin, dtype=jnp.float32),
            valmax=jnp.asarray(m.valmax, dtype=jnp.float32),
            fieldlist_mask=(None if m.fieldlist_mask is None
                            else jnp.asarray(m.fieldlist_mask)),
        ))

    # setObs_for_PHOTPROB NMATCH>1 aborts (sntools_trigger.c:2589-2594)
    n_entries = max((len(m.fieldlist_mask) for m in maps
                     if m.fieldlist_mask is not None), default=1)
    n_match = np.zeros((n_entries, len(filter_names)), dtype=np.int32)
    for m, pm in zip(maps, prepared):
        fmask = np.asarray(pm.filter_mask)
        emask = (np.asarray(m.fieldlist_mask)
                 if m.fieldlist_mask is not None
                 else np.ones(n_entries, dtype=bool))
        n_match += emask[:, None] & fmask[None, :]
    if np.any(n_match > 1):
        e, f = np.argwhere(n_match > 1)[0]
        raise SNANAError(
            f"{int(n_match[e, f])} PHOTPROB maps match SIMLIB field entry "
            f"{int(e)}, filter {filter_names[int(f)]!r}; SNANA aborts on multiple "
            f"matches.",
            function_name='prepare_photprob')

    return prepared


def evaluate_photprob(prepared_maps, snr_2d, filter_indices_2d,
                      detection_mask_2d, valid_mask_2d, redshift, sbmag_2d,
                      galmag_2d, field_match_per_map, rng_key, sn_idx):
    """Evaluate prepared PHOTPROB maps (the output of prepare_photprob)
    against batched per-obs inputs.

    Per-observation CDF interpolation + inverse CDF sampling. Operates on
    2D (n_sne, max_obs) arrays with vmap. ``field_match_per_map`` is a list
    of (n_sne,) bool per map, pre-computed from FIELDLIST strings.
    ``sbmag_2d`` and ``galmag_2d`` are (n_sne, n_filters) host surface
    brightness and galaxy magnitudes, 99 where unavailable; ``sn_idx`` is
    the per-SN index the random draws fold in.

    Returns ``(photprob_2d, photprob_pass_2d)``: ``photprob_2d`` is
    (n_sne, max_obs) float, -9.0 where not evaluated; ``photprob_pass_2d``
    is (n_sne, max_obs) bool.
    """
    n_sne, max_obs = snr_2d.shape
    photprob_2d = jnp.full((n_sne, max_obs), -9.0, dtype=jnp.float32)
    photprob_pass_2d = jnp.ones((n_sne, max_obs), dtype=bool)

    # disjoint: prepare_photprob allows one map per (field, filter)
    applicable_per_map = []
    for pm, field_match in zip(prepared_maps, field_match_per_map):
        applicable = valid_mask_2d
        filt_match = pm.filter_mask[jnp.maximum(filter_indices_2d, 0)]
        filt_match = filt_match & (filter_indices_2d >= 0)
        applicable = applicable & filt_match & field_match[:, None]
        if pm.require_detection:
            applicable = applicable & detection_mask_2d
        applicable_per_map.append(applicable)

    # shared-normal form of setRan_for_PHOTPROB's covariance (sntools_trigger.c:2679)
    corr_2d = jnp.zeros((n_sne, max_obs), dtype=jnp.float32)
    for pm, applicable in zip(prepared_maps, applicable_per_map):
        corr_2d = jnp.where(applicable, jnp.float32(pm.reduced_corr),
                            corr_2d)
    sn_keys = jax.vmap(lambda i: jax.random.fold_in(rng_key, i))(sn_idx)

    def draw_one_sn(key):
        k_common, k_indep = jax.random.split(key)
        eps_common = jax.random.normal(k_common)
        eps_indep = jax.random.normal(k_indep, (max_obs,))
        return eps_common, eps_indep

    eps_common, eps_indep = jax.vmap(draw_one_sn)(sn_keys)
    z_corr = (jnp.sqrt(1.0 - corr_2d) * eps_indep
              + jnp.sqrt(corr_2d) * eps_common[:, None])
    uniforms_2d = jax_norm.cdf(z_corr)

    for imap, pm in enumerate(prepared_maps):
        applicable = applicable_per_map[imap]
        n_vars = pm.nvar_map
        obs_points = jnp.zeros((n_sne, max_obs, n_vars), dtype=jnp.float32)
        for d, ivarabs in enumerate(pm.ivarabs):
            if ivarabs == IVARABS_SNR:
                vals = jnp.maximum(snr_2d, 0.1)
            elif ivarabs == IVARABS_LOGSNR:
                vals = jnp.log10(jnp.maximum(snr_2d, 0.1))
            elif ivarabs == IVARABS_SBMAG:
                safe_fi = jnp.maximum(filter_indices_2d, 0)
                vals = sbmag_2d[jnp.arange(n_sne)[:, None], safe_fi]
            elif ivarabs == IVARABS_GALMAG:
                safe_fi = jnp.maximum(filter_indices_2d, 0)
                vals = galmag_2d[jnp.arange(n_sne)[:, None], safe_fi]
            else:  # IVARABS_REDSHIFT
                vals = jnp.broadcast_to(redshift[:, None], (n_sne, max_obs))
            obs_points = obs_points.at[:, :, d].set(vals)

        for d in range(n_vars):
            safety = 1e-9 * (pm.valmax[d] - pm.valmin[d])
            obs_points = obs_points.at[:, :, d].set(
                jnp.clip(obs_points[:, :, d],
                         pm.valmin[d] + safety, pm.valmax[d] - safety))

        flat_points = obs_points.reshape(-1, n_vars)
        cdf_values = jnp.zeros((n_sne * max_obs, pm.nfun_cdf + 1))
        for k in range(pm.nfun_cdf):
            grid_k = pm.grid_cdf[..., k]
            cdf_k = jnp.clip(interp_nd_regular(flat_points, pm.grid_axes,
                                               grid_k, opt_extrap=0), 0.0, 1.0)
            cdf_values = cdf_values.at[:, k + 1].set(cdf_k)
        cdf_values_2d = cdf_values.reshape(n_sne, max_obs, pm.nfun_cdf + 1)

        cdf_bins = pm.cdf_bins

        def inv_cdf_one_obs(uniform, cdf_row):
            return jnp.interp(uniform, cdf_row, cdf_bins)

        photprob_vals = jax.vmap(jax.vmap(inv_cdf_one_obs))(uniforms_2d, cdf_values_2d)
        photprob_2d = jnp.where(applicable, photprob_vals, photprob_2d)

        # PHOTPROB <= CUTVAL fails (sntools_trigger.c:2238-2241)
        fail = applicable & (pm.cutval > -900) & (photprob_vals <= pm.cutval)
        photprob_pass_2d = jnp.where(fail, False, photprob_pass_2d)

    return photprob_2d, photprob_pass_2d


jax.tree_util.register_dataclass(
    PhotoprobMap,
    data_fields=[
        'grid_axes',
        'grid_cdf',
        'cdf_bins',
        'valmin',
        'valmax',
        'cutval',
        'reduced_corr',
        'fieldlist_mask',
        'filter_mask',
    ],
    meta_fields=[
        'name',
        'varnames',
        'ivarabs',
        'nvar_map',
        'nfun_cdf',
        'fieldlist',
        'filterlist',
        'require_detection',
    ],
)
