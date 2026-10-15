"""HOSTLIB file reader for JAX-SNANA.

This module parses SNANA HOSTLIB files containing galaxy catalogues with
properties like redshift, photometry, morphology, and host-specific
information.
"""

import logging
import time
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from typing import Dict, Optional, List, Tuple
from io import StringIO
from jax_snlc_core.io.fits_path import open_text, resolve_fits_path
from jax_snlc_core.io.host_properties import (HOSTGAL_PROPERTY_BASENAMES,
                                              HOST_COORDINATE_NAMES,
                                              canonical_property_name)
from jax_snlc_core.utils import MXBIN_HOSTGALz_QUANTILE, SNANAError, snana_log as slog
from .cache_manager import cache_paths, has_valid_cache, resolve_cache_dir, save_to_cache
from .sersic_sampling import SERSIC_INDEX_MIN, SERSIC_INDEX_MAX

logger = logging.getLogger(__name__)


SIMULATION_ESSENTIAL_COLUMNS = [
    'GALID', 'ZTRUE', *HOST_COORDINATE_NAMES,
    'LOGMASS', 'LOGSFR',
    'SQRADIUS', 'ELLIPTICITY',
]

# three log-z pointer bins of DZPTR_HOSTLIB = 0.01 (sntools_host.h:85, .c:3564)
DLOGZ_SAFETY = 3.0 * 0.01
# a catalogue mag error below 0 or above 9 is read as 9 (MAGERR_UNDEFINED, sntools.h:127)
MAGERR_UNDEFINED = 9.0

# Maximum number of Sersic profile components (sntools_host.h MXSERSIC_HOSTLIB)
MXSERSIC_HOSTLIB = 9

# All possible Sersic component columns (Sersic_names, sntools_host.c:1183-1194)
SERSIC_COMPONENT_COLUMNS = [
    f'{p}{j}_Sersic' for j in range(MXSERSIC_HOSTLIB) for p in 'abwn'
] + ['a_rot']

# Photo-z quantile columns: ZPHOT_Q<percent> names the percentile, QZPHOT<i>
# the index in a uniform set (PREFIX_QZPHOT, sndata.h:131; manual 5030-5047)
ZPHOT_Q_PREFIX = 'ZPHOT_Q'
QZPHOT_PREFIX = 'QZPHOT'
# the eleven decades SNANA forces under ZPHOT_QGAUSS when the catalogue
# declares no quantiles (sntools_host.c:4584)
DECADE_QUANTILES = [(f'{ZPHOT_Q_PREFIX}{p:03d}', float(p)) for p in range(0, 101, 10)]


def _quantile_column(name: str) -> bool:
    for prefix in (ZPHOT_Q_PREFIX, QZPHOT_PREFIX):
        if name.startswith(prefix) and name[len(prefix):].isdigit():
            return True
    return False


def zphot_quantile_columns(varnames: List[str]) -> List[Tuple[str, float]]:
    """The HOSTLIB's photo-z quantile columns as (column, percentile) pairs
    in percentile order: ZPHOT_Qnnn carries its percentile in the name,
    QZPHOTnn its index among nn+1 uniformly spaced percentiles. A file
    naming both conventions is refused, since SNANA counts both, as is one
    naming more quantiles than the output vector holds (sndata.h:185)."""
    legacy = [v for v in varnames if v.startswith(ZPHOT_Q_PREFIX) and _quantile_column(v)]
    indexed = [v for v in varnames if v.startswith(QZPHOT_PREFIX) and _quantile_column(v)]
    if legacy and indexed:
        raise SNANAError(
            f"the HOSTLIB names its photo-z quantiles both as {ZPHOT_Q_PREFIX}* "
            f"and as {QZPHOT_PREFIX}*; use one convention",
            function_name="zphot_quantile_columns")
    if len(legacy + indexed) > MXBIN_HOSTGALz_QUANTILE:
        raise SNANAError(
            f"{len(legacy + indexed)} photo-z quantile columns in the HOSTLIB "
            f"exceed MXBIN_HOSTGALz_QUANTILE = {MXBIN_HOSTGALz_QUANTILE}, the "
            f"quantile vector's width in the output",
            function_name="zphot_quantile_columns")
    if indexed:
        indexed.sort(key=lambda v: int(v[len(QZPHOT_PREFIX):]))
        span = max(len(indexed) - 1, 1)
        return [(v, 100.0 * i / span) for i, v in enumerate(indexed)]
    return sorted(((v, float(v[len(ZPHOT_Q_PREFIX):])) for v in legacy),
                  key=lambda pair: pair[1])


def zphot_columns(varname_zphot: str, varnames) -> Tuple[str, Optional[str]]:
    """The HOSTLIB columns holding ZPHOT and ZPHOT_ERR. The photo-z is the
    column HOSTLIB_VARNAME_ZPHOT names, and its error the first present of
    that name with ERR or _ERR appended, or ZPHOTERR
    (checkAlternateVarNames_HOSTLIB, sntools_host.c:2988-2996); None where
    the file carries no error column."""
    err = next((c for c in (varname_zphot + 'ERR', varname_zphot + '_ERR', 'ZPHOTERR')
                if c in varnames), None)
    return varname_zphot, err


def discover_output_only_columns(varnames: List[str]) -> List[str]:
    """Output-only columns present in this HOSTLIB, loaded on demand.

    Discovered from the actual VARNAMES rather than a fixed list: the
    photo-z quantiles (sntools_host.c:2977-2986), ZPHOT/ZPHOT_ERR and the
    P50/IQR summaries, [band]_obs / [band]_obs_err photometry for whatever
    bands exist (HOSTLIB_SUFFIX_MAGOBS, sntools_host.c:2757-2766), and OBJID2.
    """
    named = {'P50', 'IQR', 'ZPHOT', 'ZPHOT_ERR', 'OBJID2'}
    cols = []
    for v in varnames:
        if (_quantile_column(v)
                or v in named
                or v.endswith('_obs') or v.endswith('_obs_err')):
            cols.append(v)
    return cols


def _read_header_values(filepath: str, keys) -> Dict[str, float]:
    """The values of the header keys named in ``keys`` (``'name:'``), found
    anywhere in the header word stream as SNANA scans it
    (parse_Sersic_n_fixed, sntools_host.c:3064-3093); the first GAL: row
    ends the search."""
    found = {}
    with open_text(filepath) as f:
        for line in f:
            if line.startswith('GAL:'):
                break
            parts = line.split()
            for k, part in enumerate(parts[:-1]):
                if part in keys:
                    found[part] = float(parts[k + 1])
    return found


def _read_sersic_fixed_n(filepath: str) -> Dict[int, float]:
    """Fixed Sersic indices declared as header keys 'n{j}_Sersic: <value>'."""
    keys = {f'n{j}_Sersic:': j for j in range(MXSERSIC_HOSTLIB)}
    return {keys[k]: v for k, v in _read_header_values(filepath, keys).items()}


def _read_vpec_err(filepath: str):
    """The header's VPEC_ERR, under either spelling (sntools_host.c:2518),
    or None."""
    found = _read_header_values(filepath, ('VPEC_ERR:', 'VPECERR:'))
    return next(iter(found.values()), None)


def _fixsersic_n(fixsersic, fixed_n):
    """FIXSERSIC's n overrides header-fixed indices for components without an
    n column (sntools_host.c:5309-5310)."""
    if fixsersic is None or fixsersic[2] <= -998.0:
        return fixed_n
    return {j: fixsersic[2] for j in fixed_n}


def apply_fixsersic_columns(fixsersic, columns, scale_sersic_size=1.0):
    """HOSTLIB_FIXSERSIC writes the fixed values back into the catalogue
    columns, as SNANA writes them into VALUE_ZSORTED (get_Sersic_info,
    sntools_host.c:5303-5330), so every consumer of catalogue values
    (profiles, STOREPAR, SIMGEN_DUMP) reports the override. SCALE_SERSIC_SIZE
    then scales the half-light sizes, as SNANA scales SERSIC->a/b for the
    same struct (sntools_host.c:5328-5329), so the position, DDLR and GALMAG
    paths share one set of sizes. Weight columns are untouched. Mutates and
    returns ``columns``.
    """
    if fixsersic is not None:
        fa, fb, fn, fang = fixsersic
        for name, arr in columns.items():
            m = re.match(r'([abn])\d+_Sersic$', name)
            if m:
                fix = {'a': fa if fa > 0.0 else None,
                       'b': fb if fb > 0.0 else None,
                       'n': fn if fn > -998.0 else None}[m.group(1)]
                if fix is not None:
                    columns[name] = np.full_like(np.asarray(arr), fix)
            elif name == 'a_rot' and fang > -998.0:
                columns[name] = np.full_like(np.asarray(arr), fang)
    # a FIXSERSIC size is scaled too (sntools_host.c:5303-5329)
    if scale_sersic_size != 1.0:
        for name, arr in columns.items():
            if re.match(r'[ab]\d+_Sersic$', name):
                columns[name] = np.asarray(arr) * scale_sersic_size
    return columns


def build_sersic_profiles(sorted_column_data: Dict[str, np.ndarray],
                          varnames: List[str],
                          fixed_n: Dict[int, float]) -> Optional[Dict]:
    """Discover, validate, and pack the HOSTLIB's Sersic components.

    Mirrors init_Sersic_VARNAMES (sntools_host.c:5004-5110) and
    get_Sersic_info (sntools_host.c:5263-5436), vectorised over the whole
    catalogue at build time:

    - components j with both a{j}_Sersic and b{j}_Sersic columns are packed;
      a partial a/b pair is fatal
    - every component needs its index from an n{j}_Sersic column or a fixed
      header value, but not both
    - NPROF-1 weights may be declared, with the last inferred as
      1 - sum(others); fewer is fatal
    - weight sums off unity beyond 1e-3 are fatal; smaller truncation slop
      is renormalised away
    - a_rot is required alongside any profile (GEN_SNHOST_POS aborts
      without it, sntools_host.c:7630-7636)

    Returns a dict with 'n_prof', 'a', 'b', 'n', 'w' (each (ngal, n_prof)
    float32, w normalised to sum 1) and 'a_rot' (ngal,), or None when the
    HOSTLIB declares no profile.
    """
    fnam = "build_sersic_profiles"
    comps = []
    for j in range(MXSERSIC_HOSTLIB):
        a_name, b_name = f'a{j}_Sersic', f'b{j}_Sersic'
        has_a, has_b = a_name in varnames, b_name in varnames
        if has_a != has_b:
            present, absent = (a_name, b_name) if has_a else (b_name, a_name)
            raise SNANAError(
                f"{present} is defined but {absent} is NOT. Check VARNAMES in "
                f"HOSTLIB.",
                function_name=fnam)
        if has_a:
            comps.append(j)

    if not comps:
        # A fixed index without a matching a/b pair is fatal
        # (sntools_host.c:5073-5077).
        if fixed_n:
            raise SNANAError(
                f"{len(fixed_n)} FIXED Sersic indices defined (n#_Sersic) but 0 "
                f"matched a,b profiles.",
                function_name=fnam)
        return None

    unmatched_fix = [j for j in fixed_n if j not in comps]
    if unmatched_fix:
        raise SNANAError(
            f"FIXED Sersic indices n{unmatched_fix}_Sersic have no matching a,b "
            f"profiles.",
            function_name=fnam)

    n_prof = len(comps)
    a = np.stack([np.asarray(sorted_column_data[f'a{j}_Sersic'],
                             dtype=np.float32) for j in comps], axis=1)
    b = np.stack([np.asarray(sorted_column_data[f'b{j}_Sersic'],
                             dtype=np.float32) for j in comps], axis=1)
    ngal = a.shape[0]

    # Sersic index per component: column or fixed value, never both
    # (sntools_host.c:5081-5110)
    n_cols = []
    for j in comps:
        n_name = f'n{j}_Sersic'
        has_col, has_fix = n_name in varnames, j in fixed_n
        if has_col and has_fix:
            raise SNANAError(
                f"Sersic index '{n_name}' is defined twice (column and fixed header "
                f"value). Only 1 definition allowed.", function_name=fnam)
        if not has_col and not has_fix:
            raise SNANAError(
                f"Sersic index '{n_name}' is NOT defined for a{j}_Sersic/b{j}_Sersic. "
                f"Check VARNAMES keys in HOSTLIB.", function_name=fnam)
        n_cols.append(np.asarray(sorted_column_data[n_name], dtype=np.float32)
                      if has_col
                      else np.full(ngal, fixed_n[j], dtype=np.float32))
    n_arr = np.stack(n_cols, axis=1)

    if 'a_rot' not in varnames:
        raise SNANAError(
            "HOSTLIB defines Sersic profiles but no a_rot column; SN positions "
            "cannot be generated.", function_name=fnam)
    a_rot = np.asarray(sorted_column_data['a_rot'], dtype=np.float32)

    oob_n = (n_arr < SERSIC_INDEX_MIN) | (n_arr > SERSIC_INDEX_MAX)
    if np.any(oob_n):
        bad = n_arr[oob_n]
        raise SNANAError(
            f"Sersic index outside valid range "
            f"({SERSIC_INDEX_MIN:.2f}-{SERSIC_INDEX_MAX:.2f}) for "
            f"{int(oob_n.any(axis=1).sum())} galaxies (first bad n = "
            f"{bad[0]:.4f}); get_Sersic_info aborts.", function_name=fnam)

    zero_size = np.all((a < 1.0e-9) | (b < 1.0e-9), axis=1)
    if np.any(zero_size):
        raise SNANAError(
            f"Profile size is zero for {int(zero_size.sum())} galaxies (all "
            f"components have a or b < 1e-9); get_Sersic_info aborts.",
            function_name=fnam)

    # Weights: NPROF or NPROF-1 declared, the missing one inferred
    # (sntools_host.c:5350-5396)
    w_present = [f'w{j}_Sersic' in varnames for j in comps]
    nwgt = sum(w_present)
    if nwgt < n_prof - 1:
        raise SNANAError(
            f"Inadequate Sersic weights: {n_prof} Sersic terms, but only {nwgt} "
            f"weights defined.",
            function_name=fnam)
    w_cols = []
    for j, present in zip(comps, w_present):
        w_cols.append(np.asarray(sorted_column_data[f'w{j}_Sersic'],
                                 dtype=np.float32) if present else None)
    if nwgt == n_prof - 1:
        declared = sum(c for c in w_cols if c is not None)
        declared = declared if nwgt > 0 else np.zeros(ngal, dtype=np.float32)
        j_nowgt = w_present.index(False)
        w_cols[j_nowgt] = 1.0 - declared
    w = np.stack(w_cols, axis=1).astype(np.float32)

    if np.any(w < 0.0):
        raise SNANAError(
            f"Negative Sersic weight for {int((w < 0).any(axis=1).sum())} "
            f"galaxies. Check values of w#_Sersic (declared weights exceed "
            f"1 where the last weight is inferred as 1 - sum).",
            function_name=fnam)

    wtot = w.sum(axis=1)
    bad_sum = np.abs(wtot - 1.0) > 1.0e-3
    if np.any(bad_sum):
        first = int(np.argmax(bad_sum))
        raise SNANAError(
            f"Sum of Sersic weights = {wtot[first]:.6f} (ne 1) for "
            f"{int(bad_sum.sum())} galaxies. Check values of w1, w2, etc.",
            function_name=fnam)
    # Renormalise HOSTLIB truncation slop (sntools_host.c:5427-5434)
    w = (w / wtot[:, None]).astype(np.float32)

    slog.line(logger, f"Sersic profiles: {n_prof} component(s) "
              f"(j = {', '.join(str(j) for j in comps)})")
    return {
        'n_prof': n_prof,
        'component_ids': comps,
        'a': a,
        'b': b,
        'n': n_arr,
        'w': w,
        'a_rot': a_rot,
    }


def neighbour_offsets(ra_gal: np.ndarray, dec_gal: np.ndarray,
                      nbr_array: np.ndarray) -> np.ndarray:
    """Each neighbour's RA/DEC offset from the galaxy whose neighbour it is.

    Differenced in float64 so the DDLR geometry subtracts small numbers
    rather than two absolute sky coordinates, which in float32 lose ~0.1
    arcsec to cancellation at RA ~ 300 deg. Absent neighbours (index < 0)
    take a zero offset.

    Returns (2, ngal, M+1) float32: RA offsets then DEC offsets.
    """
    ra = np.asarray(ra_gal, dtype=np.float64)
    dec = np.asarray(dec_gal, dtype=np.float64)
    nbr = np.asarray(nbr_array)
    safe = np.clip(nbr, 0, len(ra) - 1)
    present = nbr >= 0
    return np.stack([
        np.where(present, ra[safe] - ra[:, None], 0.0),
        np.where(present, dec[safe] - dec[:, None], 0.0),
    ]).astype(np.float32)


def _remap_nbr_array_to_sorted(nbr_array: np.ndarray, sorted_indices: np.ndarray) -> np.ndarray:
    """Convert nbr_array from original file order to z-sorted space.

    The raw ``nbr_array`` (ngal, max_neighbours), -1 padded, stores original
    row numbers in original file order; ``sorted_indices`` is the argsort
    output mapping sorted_pos -> original_pos. After z-sorting, both the
    rows (to match sorted galaxy order) and the neighbour-index values (to
    point to sorted positions) need remapping.

    Returns the remapped array in sorted order with sorted-space neighbour
    indices.
    """
    ngal = len(sorted_indices)
    inv_sort = np.empty(ngal, dtype=np.int32)
    inv_sort[sorted_indices] = np.arange(ngal, dtype=np.int32)
    valid = nbr_array >= 0
    remapped = np.where(valid, inv_sort[np.clip(nbr_array, 0, ngal - 1)], -1)
    return remapped[sorted_indices]


@dataclass(frozen=True)
class HostlibTable:
    """One HOSTLIB as read: the requested columns of every galaxy the read
    cuts keep, in redshift order, with their neighbour lists and Sersic
    profiles. The build consumes it and nothing holds it afterwards."""
    columns: Dict[str, np.ndarray]    # integer columns int64, the rest float32
    neighbours: np.ndarray            # (ngal, max_nbr) row indices, -1 padded
    sersic_profiles: Optional[Dict]
    z_min: float                      # the file's own ZTRUE over the kept rows
    z_max: float
    vpec_max: Optional[float]         # the largest |VPEC|
    vpec_err: Optional[float]

    @property
    def ngal(self) -> int:
        return len(self.columns['ZTRUE'])


def essential_columns() -> List[str]:
    """The columns every simulation with a HOSTLIB reads when the file
    carries them: the essential set, every Sersic component, the a_DLR/b_DLR
    override of the Sersic sizes (sntools_host.c:8559-8562) and every host
    property under each of its spellings (GEN_SNHOST_PROPERTY,
    sntools_host.c:2699-2721)."""
    columns = [*SIMULATION_ESSENTIAL_COLUMNS, *SERSIC_COMPONENT_COLUMNS,
               'a_DLR', 'b_DLR']
    for base in HOSTGAL_PROPERTY_BASENAMES:
        columns.extend([base, f'{base}_TRUE', f'{base}_OBS', f'{base}_ERR'])
    return columns


def hostlib_declared(filepath: str) -> List[str]:
    """Every column the HOSTLIB carries, under its property name, from the
    VARNAMES header; the neighbour list is not a column."""
    return list(dict.fromkeys(canonical_property_name(v)
                              for v in _read_varnames(filepath)
                              if v != 'NBR_LIST'))


def read_hostlib(filepath: str,
                 columns: List[str],
                 genrange_ra: Tuple[float, float] = (-999.0, 999.0),
                 genrange_dec: Tuple[float, float] = (-999.0, 999.0),
                 genrange_redshift: Optional[Tuple[float, float]] = None,
                 maxread: int = 80_000_000,
                 snr_store=(None, ''),
                 varname_zphot: str = "ZPHOT",
                 swap_zphot: bool = False,
                 use_cache: bool = True,
                 cache_dir: Optional[str] = None,
                 fixsersic=None,
                 scale_sersic_size: float = 1.0) -> HostlibTable:
    """Read the HOSTLIB's ``columns`` (those the file carries) for every
    galaxy the read cuts keep, sorted by redshift.

    ``columns`` are named as the simulation names them; HOSTLIB_VARNAME_ZPHOT
    says which file column is ZPHOT and its error
    (checkAlternateVarNames_HOSTLIB, sntools_host.c:2988-2996). ``snr_store``
    is the HOSTLIB_SNR_STORE ladder (or None) with the GENFILTERS string whose
    bands it is applied over. ``swap_zphot`` is HOSTLIB_MSKOPT bit 128 (SWAPZPHOT): draw and sort
    the catalogue on ZPHOT instead of ZTRUE. FIXSERSIC and SCALE_SERSIC_SIZE
    are written into the Sersic columns. The file is parsed once and kept
    as a Parquet cache in ``cache_dir`` (None picks the platform location),
    unless ``use_cache`` is False.
    """
    filepath = resolve_fits_path(filepath, required=True)
    logger.info(f"Reading HOSTLIB file: {filepath}")
    t_start = time.time()
    column_data, neighbours = _read_columns(filepath, columns, varname_zphot,
                                            use_cache, cache_dir)
    ztrue_file = column_data['ZTRUE']
    column_data = _swap_ztrue_for_zphot(column_data, swap_zphot)
    order, sorted_columns = _sort_by_redshift(column_data)
    neighbours = _remap_nbr_array_to_sorted(neighbours, order)
    order, sorted_columns, neighbours = _apply_read_selection(
        sorted_columns, order, neighbours, ztrue_file[order], genrange_ra,
        genrange_dec, genrange_redshift, maxread, snr_store)
    apply_fixsersic_columns(fixsersic, sorted_columns, scale_sersic_size)
    sersic_profiles = build_sersic_profiles(
        sorted_columns, list(sorted_columns),
        _fixsersic_n(fixsersic, _read_sersic_fixed_n(filepath)))
    kept_ztrue = ztrue_file[order]
    slog.line(logger, f"Loaded {len(order):,} galaxies, {len(sorted_columns)} "
              f"columns, in {time.time() - t_start:.2f}s")
    return HostlibTable(
        columns=sorted_columns,
        neighbours=neighbours,
        sersic_profiles=sersic_profiles,
        z_min=float(np.min(kept_ztrue)),
        z_max=float(np.max(kept_ztrue)),
        vpec_max=_vpec_max(sorted_columns),
        vpec_err=_read_vpec_err(filepath),
    )


def _read_columns(filepath, columns, varname_zphot, use_cache, cache_dir):
    """The requested columns the file carries, in file order and under the
    requested names, and the neighbour array in file order. The cache holds
    every column of the parsed file, so a run reading other columns later
    reads them from it."""
    if not use_cache:
        data, neighbours = _parse_hostlib(filepath)
        names = _file_columns(columns, varname_zphot, data)
        return {name: data[spelt] for name, spelt in names.items()}, neighbours
    cache_dir = resolve_cache_dir(cache_dir)
    if not has_valid_cache(cache_dir, filepath):
        slog.line(logger, "No valid cache found, parsing from source")
        data, neighbours = _parse_hostlib(filepath)
        save_to_cache(cache_dir, filepath, pd.DataFrame(data), neighbours)
    cache_path, nbr_path = cache_paths(cache_dir, filepath)
    names = _file_columns(columns, varname_zphot,
                          pq.read_schema(cache_path).names)
    frame = pd.read_parquet(cache_path,
                            columns=list(dict.fromkeys(names.values())))
    return ({name: frame[spelt].values for name, spelt in names.items()},
            np.load(nbr_path))


def _parse_hostlib(filepath):
    """Every column of the HOSTLIB text file under its property name, in
    file order, and the NBR_LIST as a padded neighbour array."""
    varnames = _read_varnames(filepath)
    column_data = _read_galaxy_data(filepath, varnames)
    ngal = len(column_data[varnames[0]])
    if 'NBR_LIST' in varnames:
        neighbours = _parse_and_pad_nbr_list(column_data.pop('NBR_LIST'))
    else:
        neighbours = np.full((ngal, 1), -1, dtype=np.int32)
    column_data = _canonical_properties(column_data, filepath)
    slog.line(logger, f"Parsed {ngal:,} galaxies from {filepath}")
    return column_data, neighbours


def _file_columns(columns, varname_zphot, available):
    """The file column read for each requested column the file carries,
    keyed by the requested name; ZPHOT and ZPHOT_ERR follow
    HOSTLIB_VARNAME_ZPHOT."""
    zphot, zphot_err = zphot_columns(varname_zphot, available)
    alias = {'ZPHOT': zphot, 'ZPHOT_ERR': zphot_err}
    names = {}
    for name in dict.fromkeys(canonical_property_name(c) for c in columns):
        spelt = alias.get(name) or name
        if spelt in available:
            names[name] = spelt
    return names


def _snr_store_mask(sorted_column_data, thresholds, genfilters):
    """HOSTLIB_SNR_STORE: the i-th brightest band's signal-to-noise must
    reach the i-th threshold, over the GENFILTERS bands with both a mag and
    an error column (snr_store_HOSTLIB, sntools_host.c:8226; the ladder test
    is passCuts_snr_HOSTLIB, :8290)."""
    err_cols = [f'{band}_obs_err' for band in genfilters
                if f'{band}_obs' in sorted_column_data
                and f'{band}_obs_err' in sorted_column_data]
    if len(err_cols) < len(thresholds):
        raise SNANAError(
            f"NBAND = {len(err_cols)} but NBAND_SNR_CUT=={len(thresholds)}. Check "
            f"<BAND>_OBS in HOSTLIB, and sim-input HOSTLIB_SNR_DETECT / "
            f"HOSTLIB_SNR_STORE.",
            function_name='_snr_store_mask')
    errs = np.stack([np.asarray(sorted_column_data[c], dtype=np.float64)
                     for c in err_cols], axis=1)
    errs = np.where((errs < 0.0) | (errs > MAGERR_UNDEFINED), MAGERR_UNDEFINED, errs)
    with np.errstate(divide='ignore'):
        snr = np.where(errs > 0.0, (2.5 / np.log(10.0)) / errs, 0.0)
    ranked = -np.sort(-snr, axis=1)[:, :len(thresholds)]
    return np.all(ranked >= np.asarray(thresholds, dtype=np.float64)[None, :],
                  axis=1)


def _apply_read_selection(sorted_column_data, sorted_indices, nbr_array,
                          ztrue_file, genrange_ra, genrange_dec,
                          genrange_redshift, maxread, snr_store):
    """HOSTLIB_MAXREAD (first N file rows, sntools_host.c:3176) and the cuts
    of passCuts_HOSTLIB (sntools_host.c:3351) on the z-sorted catalogue: the
    GENRANGE_RA/DEC box, GENRANGE_REDSHIFT widened by 0.01 and three
    log-z pointer bins each side (sntools_host.c:3147-3156), and the
    HOSTLIB_SNR_STORE ladder, so a galaxy the generation can never reach is
    neither stored nor weighted.
    ``ztrue_file`` is the file's own ZTRUE in sorted order: SNANA applies the
    window to the row as read (:3179), before SWAPZPHOT overwrites ZTRUE
    with ZPHOT (:3228)."""
    ngal = len(sorted_indices)
    keep = np.ones(ngal, dtype=bool)

    thresholds, genfilters = snr_store
    if thresholds is not None:
        keep &= _snr_store_mask(sorted_column_data, thresholds, genfilters)

    if maxread < ngal:
        keep &= (sorted_indices < maxread)

    if genrange_redshift is not None:
        # a NaN edge from log10 of a non-positive bound cuts nothing, as in SNANA
        with np.errstate(divide='ignore', invalid='ignore'):
            zwin = (10.0 ** (np.log10(genrange_redshift[0] - 0.01) - DLOGZ_SAFETY),
                    10.0 ** (np.log10(genrange_redshift[1] + 0.01) + DLOGZ_SAFETY))
        keep &= ~(ztrue_file < zwin[0]) & ~(ztrue_file > zwin[1])

    ra_lo, ra_hi = float(genrange_ra[0]), float(genrange_ra[1])
    dec_lo, dec_hi = float(genrange_dec[0]), float(genrange_dec[1])
    ra_default = (ra_lo == -999.0 and ra_hi == 999.0)
    dec_default = (dec_lo == -999.0 and dec_hi == 999.0)
    if (not ra_default or not dec_default) and \
            'RA_GAL' in sorted_column_data and 'DEC_GAL' in sorted_column_data:
        ra = sorted_column_data['RA_GAL']
        dec = sorted_column_data['DEC_GAL']
        in_ra = ((ra > ra_lo - 1e-4) & (ra < ra_hi + 1e-4)) | \
                ((ra + 360.0 > ra_lo - 1e-4) & (ra + 360.0 < ra_hi + 1e-4))
        in_dec = (dec > dec_lo - 1e-4) & (dec < dec_hi + 1e-4)
        keep &= in_ra & in_dec

    if keep.all():
        return sorted_indices, sorted_column_data, nbr_array

    new_ngal = int(keep.sum())
    new_map = np.full(ngal, -1, dtype=np.int64)
    new_map[keep] = np.arange(new_ngal)

    f_column_data = {k: v[keep] for k, v in sorted_column_data.items()}
    f_sorted_indices = sorted_indices[keep]
    f_nbr = nbr_array[keep]
    f_nbr = np.where(f_nbr >= 0, new_map[np.clip(f_nbr, 0, ngal - 1)], -1).astype(nbr_array.dtype)

    slog.line(logger, f"Read-time cut: {ngal:,} -> {new_ngal:,} galaxies "
              f"(MAXREAD/GENRANGE_RA/GENRANGE_DEC/GENRANGE_REDSHIFT/SNR_STORE)")
    return f_sorted_indices, f_column_data, f_nbr


def _canonical_properties(column_data, filepath):
    """The columns under their in-memory property names, as SNANA reads a
    HOSTLIB (sntools_host.c:2481); a file spelling one property both ways
    is refused, as its VARNAMES would then repeat a name there."""
    renamed = {}
    spelt = {}
    for name, values in column_data.items():
        canonical = canonical_property_name(name)
        if canonical in renamed:
            raise SNANAError(f"{filepath} defines both {spelt[canonical]} and "
                             f"{name}, which name one quantity",
                             function_name="read_hostlib")
        renamed[canonical] = values
        spelt[canonical] = name
    return renamed


def _read_varnames(filepath: str) -> List[str]:
    """Extract column names from the VARNAMES: line in the header.

    Returns the list of column names for the HOSTLIB file at ``filepath``.
    Raises `SNANAError` if no VARNAMES line is found.
    """
    with open_text(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('VARNAMES:'):
                return line.split()[1:]

    raise SNANAError(f"VARNAMES line not found in {filepath}")


def _read_galaxy_data(filepath: str,
                      varnames: List[str]) -> Dict[str, np.ndarray]:
    """Read galaxy data from ``filepath`` using pandas.

    Returns a dict mapping each of ``varnames`` to a numpy array.
    """
    pandas_columns = ['prefix'] + varnames
    gal_lines = []
    with open_text(filepath) as f:
        for line in f:
            if line.startswith('GAL:'):
                gal_lines.append(line)
    data_str = ''.join(gal_lines)

    df = pd.read_csv(
        StringIO(data_str),
        sep=r'\s+',
        names=pandas_columns,
        comment='#',
    )
    df = df.drop('prefix', axis=1)
    column_data = {col: df[col].values for col in varnames}

    return column_data


def _swap_ztrue_for_zphot(column_data: Dict[str, np.ndarray], swap: bool):
    """HOSTLIB_MSKOPT bit 128 (SWAPZPHOT): the catalogue is sorted, searched
    and drawn on ZPHOT.

    SNANA writes ZPHOT over ZTRUE as it reads each galaxy row and sorts
    afterwards (sntools_host.c:3228-3231, :193), so the redshift-window
    search runs on the swapped column. HOSTLIB.ZMIN/ZMAX keep the file's own
    ZTRUE, which the read loop accumulates before the swap (:3201-3204).
    """
    if not swap:
        return column_data
    missing = [c for c in ('ZPHOT', 'ZPHOT_ERR') if c not in column_data]
    if missing:
        raise SNANAError(
            f"HOSTLIB_MSKOPT bit 128 (SWAPZPHOT) needs the "
            f"{' and '.join(missing)} column(s), which this HOSTLIB does not "
            f"carry; add them or drop the bit.",
            function_name="read_hostlib")
    return {**column_data, 'ZTRUE': column_data['ZPHOT']}


def _vpec_max(column_data: Dict[str, np.ndarray]):
    """The largest peculiar velocity USEVPEC can hand an event, or None
    without a VPEC column."""
    if 'VPEC' not in column_data:
        return None
    return float(np.max(np.abs(column_data['VPEC'])))


def _sort_by_redshift(column_data: Dict[str, np.ndarray]) -> tuple:
    """The columns in ZTRUE order: integer columns as int64, since GALIDs
    exceed 2^31, and the rest as float32.

    Returns ``(order, sorted_columns)``, ``order`` the file row of each
    sorted galaxy.
    """
    order = np.argsort(column_data['ZTRUE'])
    return order, {
        name: (values[order].astype(np.int64)
               if np.issubdtype(values.dtype, np.integer)
               else values[order].astype(np.float32))
        for name, values in column_data.items()}


def _parse_and_pad_nbr_list(nbr_list_series: np.ndarray) -> np.ndarray:
    """Parse NBR_LIST column from strings to padded 2D array.

    Converts comma-separated neighbour indices into a rectangular JAX-friendly
    array with -1 padding for galaxies with fewer neighbours. ``nbr_list_series``
    holds the NBR_LIST strings. Returns a 2D NumPy array (ngal, max_neighbours)
    with -1 padding.
    """
    neighbour_lists = []
    for val in nbr_list_series:
        if pd.isna(val) or val == '-1' or val == '':
            neighbour_lists.append([])
        else:
            neighbour_lists.append([int(x.strip()) for x in str(val).split(',')])

    max_neighbours = max((len(n) for n in neighbour_lists), default=0)

    if max_neighbours == 0:
        nbr_array = np.full((len(neighbour_lists), 1), -1, dtype=np.int32)
    else:
        nbr_array = np.full((len(neighbour_lists), max_neighbours), -1, dtype=np.int32)
        for i, n_list in enumerate(neighbour_lists):
            if len(n_list) > 0:
                nbr_array[i, :len(n_list)] = n_list

    return nbr_array
