"""Load and prepare host galaxy catalogues for simulation.

This module parses SNANA HOSTLIB files containing galaxy catalogues with
properties like redshift, photometry, morphology, and host-specific
information.
"""


import hashlib
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import jax
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from platformdirs import user_cache_dir

from jax_snlc_core import logging as log, snana_files
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import (
    HOST_COORDINATE_NAMES, HOSTGAL_PROPERTY_BASENAMES, canonical_property_name,
    open_text, resolve_fits_path)
from jax_snlc_sim.hostlib.photoz import is_zphot_quantile, zphot_columns
from jax_snlc_sim.hostlib.sersic import (
    MXSERSIC_HOSTLIB, SERSIC_COMPONENT_COLUMNS, apply_fixsersic_columns,
    build_sersic_profiles)


logger = logging.getLogger(__name__)

_HOSTLIB_SOURCE_HASH = hashlib.sha256(
    Path(__file__).read_bytes() + Path(snana_files.__file__).read_bytes()).hexdigest()
_SOURCE_HASH_META_KEY = b'jax_snana_source_hash'


SIMULATION_ESSENTIAL_COLUMNS = [
    'GALID', 'ZTRUE', *HOST_COORDINATE_NAMES,
    'LOGMASS', 'LOGSFR',
    'SQRADIUS', 'ELLIPTICITY',
]

# three log-z pointer bins of DZPTR_HOSTLIB = 0.01 (sntools_host.h:85, .c:3564)
DLOGZ_SAFETY = 3.0 * 0.01
# a catalogue mag error below 0 or above 9 is read as 9 (MAGERR_UNDEFINED, sntools.h:127)
MAGERR_UNDEFINED = 9.0

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
        if (is_zphot_quantile(v)
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
    fixed_n = _read_sersic_fixed_n(filepath)
    # FIXSERSIC overrides fixed header indices for components without an n column.
    if fixsersic is not None and fixsersic[2] > -998.0:
        fixed_n = {j: fixsersic[2] for j in fixed_n}
    sersic_profiles = build_sersic_profiles(sorted_columns, list(sorted_columns), fixed_n)
    kept_ztrue = ztrue_file[order]
    log.line(logger, f"Loaded {len(order):,} galaxies, {len(sorted_columns)} "
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
        log.line(logger, "No valid cache found, parsing from source")
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
    frame = snana_files.read_fitres(filepath, row_keys=('GAL:',)).df
    column_data = {name: frame[name].to_numpy() for name in frame.columns}
    ngal = len(frame)
    if 'NBR_LIST' in column_data:
        neighbours = _parse_and_pad_nbr_list(column_data.pop('NBR_LIST'))
    else:
        neighbours = np.full((ngal, 1), -1, dtype=np.int32)
    column_data = _canonical_properties(column_data, filepath)
    log.line(logger, f"Parsed {ngal:,} galaxies from {filepath}")
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

    log.line(logger, f"Read-time cut: {ngal:,} -> {new_ngal:,} galaxies "
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


def resolve_cache_dir(cache_dir: Optional[str] = None) -> str:
    """The cache directory: the caller's choice, else the platform location."""
    if cache_dir is not None:
        return cache_dir
    return str(Path(user_cache_dir("jax_snana")) / "hostlib_cache")


def cache_paths(cache_dir: str, source_filepath: str) -> Tuple[str, str]:
    """The ``(parquet_path, numpy_path)`` cache file paths for a source
    HOSTLIB file."""
    abs_path = os.path.abspath(source_filepath)
    hasher = hashlib.sha256(abs_path.encode('utf-8'))
    hash_str = hasher.hexdigest()[:16]
    basename_clean = Path(source_filepath).name.replace(
        '.gz', '').replace('.HOSTLIB', '')

    parquet_filename = f"{basename_clean}_{hash_str}.parquet"
    numpy_filename = f"{basename_clean}_{hash_str}_nbr.npy"

    cache_dir_path = Path(cache_dir)
    return (str(cache_dir_path / parquet_filename),
            str(cache_dir_path / numpy_filename))


def has_valid_cache(cache_dir: str, source_filepath: str) -> bool:
    """True when a valid cache exists for ``source_filepath``: both parquet
    and numpy files exist, the cache is newer than the source, and the
    parquet's embedded source hash matches the reader that would write it
    now."""
    if not Path(source_filepath).exists():
        return False

    parquet_path, numpy_path = cache_paths(cache_dir, source_filepath)
    parquet_path, numpy_path = Path(parquet_path), Path(numpy_path)
    if not (parquet_path.exists() and numpy_path.exists()):
        return False

    source_mtime = Path(source_filepath).stat().st_mtime
    cache_mtime = parquet_path.stat().st_mtime
    if cache_mtime <= source_mtime:
        return False

    schema_meta = pq.read_schema(parquet_path).metadata or {}
    return schema_meta.get(_SOURCE_HASH_META_KEY) == _HOSTLIB_SOURCE_HASH.encode()


def save_to_cache(cache_dir: str, source_filepath: str, df: pd.DataFrame,
                  nbr_array: np.ndarray):
    """Write the parsed HOSTLIB columns and neighbour array to the cache.

    Both files are written under a temporary name in the cache directory and
    renamed into place, so a reader that arrives while another process is
    still writing sees either the previous file or none, never a half-written
    one. Two workers running the same input on a cold cache is the ordinary
    case, not a rare one.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    parquet_path, numpy_path = cache_paths(cache_dir, source_filepath)

    logger.info(f"Saving to cache: {parquet_path}")
    t0 = time.time()

    table = pa.Table.from_pandas(df, preserve_index=False)
    schema_meta = dict(table.schema.metadata or {})
    schema_meta[_SOURCE_HASH_META_KEY] = _HOSTLIB_SOURCE_HASH.encode()
    table = table.replace_schema_metadata(schema_meta)

    _write_then_rename(parquet_path,
                       lambda path: pq.write_table(table, path,
                                                   compression='snappy'))
    _write_then_rename(numpy_path, lambda path: np.save(path, nbr_array))

    log.line(logger, f"Saved {len(df):,} galaxies to cache in "
             f"{time.time() - t0:.2f}s")


def _write_then_rename(path: str, write) -> None:
    """Write through a uniquely named neighbour of ``path`` and rename it
    into place, which is atomic on one filesystem."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


class HostCatalogue(NamedTuple):
    """Every per-galaxy array the forward model reads, galaxies in ZTRUE
    order. A compiled batch takes it as an argument rather than holding it
    as a constant, since a real HOSTLIB holds millions of galaxies. A field
    is None where the input has no use for it.

    ``wgtsum`` and ``marginal_cdf`` hold one block of WGTMAP SN-variable
    bins per GROUPID set, the first block the whole catalogue's;
    ``marginal_cdf`` repeats those blocks at the DNDZ_PEC1A rate when there
    is one, then adds a block per distinct (GROUPID set, SIMLIB entry
    redshift window).
    """
    redshifts: jax.Array                      # (ngal,) ZTRUE
    wgtsum: jax.Array                         # (n_sets * n_bins, ngal)
    marginal_cdf: Optional[jax.Array]         # wgtsum's blocks, again at the DNDZ_PEC1A rate, then the entry-window blocks; SN2GAL_Z
    weights: Optional[jax.Array]              # (n_bins, ngal), GALWGT
    snmagshift: Optional[jax.Array]           # (ngal,) or (n_bins, ngal)
    columns: jax.Array                        # (n_vars, ngal), HostSelector.varnames
    sersic: Optional[Dict[str, jax.Array]]    # a, b, n, w (n_comp, ngal); a_rot (ngal,)
    neighbours: jax.Array                     # (ngal, max_nbr), -1 padded
    neighbour_offsets: jax.Array              # (2, ngal, max_nbr), degrees
    detect_columns: jax.Array                 # (n_bands, ngal)
    detect_errs: jax.Array                    # (n_bands, ngal)
    extra_columns: Dict[str, jax.Array]       # deferred columns by name
