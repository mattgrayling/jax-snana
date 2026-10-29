"""SIMLIB observing conditions: reading, preparation and scheduling.

A SIMLIB is a global header followed by one LIBID entry per pointing, each
carrying its own header and one ``S:`` line per observation. The reader
returns a :class:`SimlibData` holding the entries as :class:`SimlibField`
objects and, alongside them, the same per-observation columns concatenated
into flat arrays the JAX scheduler indexes directly. Parsed entries are
cached in HDF5 in the SIMLIB cache directory. A source-file or module
change rebuilds the cache; each read returns fresh arrays for preparation.
"""

import hashlib
import json
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import h5py
import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core import logging as log
from jax_snlc_core.cosmology import eq_to_gal
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.noise import noise_equivalent_aperture as _compute_noise_equivalent_aperture
from jax_snlc_core.snana_files import (check_file_docana, load_survey_ids,
                                       open_text, parse_prescales, resolve_fits_path)
from jax_snlc_core.spectrograph import synthetic_band_zptpsfsky
from jax_snlc_core.random import stream_key
from jax_snlc_sim.simulation import run_key

logger = logging.getLogger(__name__)

FWHM_SIGMA_RATIO = 2.3548    # sntools.h:99
TWOPI = 2.0 * np.pi

_READER_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

# snlc_sim.h:70,72
TGAP_SEASON_SIMLIB = 90.0
MXSEASON_SIMLIB = 30


def _compute_seasons(mjds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Segment sorted MJDs into (start, end) seasons padded to MXSEASON_SIMLIB."""
    starts = np.full(MXSEASON_SIMLIB, np.inf, dtype=np.float32)
    ends = np.full(MXSEASON_SIMLIB, -np.inf, dtype=np.float32)
    if mjds.size == 0:
        return starts, ends
    gaps = np.diff(mjds) > TGAP_SEASON_SIMLIB
    seg_starts = np.concatenate(([0], np.flatnonzero(gaps) + 1))
    seg_ends = np.concatenate((np.flatnonzero(gaps), [mjds.size - 1]))
    n_seasons = len(seg_starts)
    if n_seasons >= MXSEASON_SIMLIB:
        raise SNANAError(
            f"Invalid NSEASON={n_seasons} exceeds bound of {MXSEASON_SIMLIB}.",
            function_name='_compute_seasons')
    starts[:n_seasons] = mjds[seg_starts]
    ends[:n_seasons] = mjds[seg_ends]
    return starts, ends


SIMLIB_OBSERVATION_KEYS = (
    'mjds', 'filter_indices', 'skysigs', 'psf1s', 'psf2s', 'psf_ratios',
    'neas', 'zptavgs', 'zpterrs', 'gains', 'readnoises', 'nexposes',
    'template_skysigs', 'template_readnoises', 'template_zpts',
    'append_photflags',
)

# NEA is derived from the PSF triple, not parsed
_PER_OBS_DTYPES = {k: np.float32 for k in SIMLIB_OBSERVATION_KEYS}
_PER_OBS_DTYPES['filter_indices'] = np.int8
_PER_OBS_DTYPES['append_photflags'] = np.int32
_PARSED_OBS_KEYS = tuple(k for k in SIMLIB_OBSERVATION_KEYS if k != 'neas')

# Per-entry header settings, cached as one JSON string per entry.
_ENTRY_KEYS = ('header_values', 'gengauss', 'cutwin_redshift',
               'genrange_redshift', 'redshift_keys', 'subsurvey')

# the header keys that each set the entry's redshift window; the last one read wins
_REDSHIFT_KEYS = ('REDSHIFT', 'DISTANCE', 'GENRANGE_REDSHIFT')
# an entry's redshift window redraws z only above this lower edge
ZMIN_SIMLIB_REGEN = 1.0e-5


def _entry_from_json(text) -> dict:
    entry = json.loads(text)
    entry['gengauss'] = {k: tuple(v) for k, v in entry['gengauss'].items()}
    entry['cutwin_redshift'] = tuple(entry['cutwin_redshift'])
    entry['genrange_redshift'] = tuple(entry['genrange_redshift'])
    entry['redshift_keys'] = tuple(entry['redshift_keys'])
    return entry


def header_column(fields, key: str) -> np.ndarray:
    """One header value per entry under ``key``, NaN where the entry has none."""
    return np.array([float(f.header_values.get(key, 'nan')) for f in fields])


@dataclass
class FluxErrCorMap:
    """SIMLIB header FLUXERR_COR: per-filter FLUXCALERR scale vs log10(SNR),
    interpolated at each observation's SNR when enabled."""
    enabled: bool = False
    log10snr: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    scale: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))


def _build_fluxerr_add(line, filter_mapping):
    n_filters = len(filter_mapping)
    out = np.zeros(n_filters, dtype=np.float32)
    if line is None:
        return out
    filt_chars, values = line
    for c, val in zip(filt_chars, values):
        if c in filter_mapping:
            out[filter_mapping[c]] = val
    return out


def _build_fluxerr_cor_map(rows, filter_mapping):
    if not rows:
        return FluxErrCorMap()
    for filt_chars, log10snr, scales in rows:
        for c, val in zip(filt_chars, scales):
            if val < 0.1 or val > 10.0:
                raise SNANAError(
                    f"FLUXERR_COR: scale {val} for filter {c!r} at log10SNR={log10snr} is "
                    f"outside [0.1, 10.0]. SNANA aborts at SIMLIB load on this.",
                    function_name="_build_fluxerr_cor_map")

    per_filt: Dict[str, list] = {}
    for filt_chars, log10snr, scales in rows:
        for c, val in zip(filt_chars, scales):
            per_filt.setdefault(c, []).append((log10snr, float(val)))

    union = np.array(sorted({r[1] for r in rows}), dtype=np.float32)
    n_filters = len(filter_mapping)
    scale = np.ones((n_filters, len(union)), dtype=np.float32)
    for c, entries in per_filt.items():
        if c not in filter_mapping:
            continue
        entries.sort(key=lambda e: e[0])
        xs = np.array([e[0] for e in entries], dtype=np.float32)
        ys = np.array([e[1] for e in entries], dtype=np.float32)
        scale[filter_mapping[c]] = np.interp(union, xs, ys)
    return FluxErrCorMap(enabled=True, log10snr=union, scale=scale)

def _parse_idexpt_token(token: str) -> Tuple[int, int]:
    """
    Extract exposure ID and NEXPOSE from tokens like '2662*4' or
    ``ID[*NEXPOSE][(DETNUM[,XPIX,YPIX])]`` (parse_SIMLIB_IDEXPT).
    Returns (exposure_id, nexpose): each read as sscanf's %d reads it, -9
    and 1 when unreadable. A parenthesised part with other than 1 or 3
    arguments aborts.
    """
    match = re.fullmatch(r'([^(]*)(?:\(([^)]*)\))?', token)
    if match is None or (match.group(2) is not None
                         and len(match.group(2).split(',')) not in (1, 3)):
        raise SNANAError(
            f"Invalid IDEXPT '{token}': expected 1 or 3 comma-separated args "
            f"in parentheses", function_name="_parse_idexpt_token")
    id_str, _, nexpose_str = match.group(1).partition('*')
    return _scan_int(id_str, -9), _scan_int(nexpose_str, 1)


def _scan_int(text: str, unread: int) -> int:
    """The leading integer of ``text`` as sscanf's %d reads it, or ``unread``."""
    match = re.match(r'\s*([+-]?\d+)', text)
    return int(match.group(1)) if match else unread


@dataclass
class SimlibField:
    """Complete field from SIMLIB with all observations."""
    libid: int
    ra: float
    dec: float
    mwebv: float         # Milky Way extinction
    pixsize: float       # pixel scale (arcsec)
    nobs: int
    mjds: np.ndarray
    filter_indices: np.ndarray  # index into the SIMLIB FILTERS header order
    skysigs: np.ndarray
    psf1s: np.ndarray
    psf2s: np.ndarray
    psf_ratios: np.ndarray
    neas: np.ndarray
    zptavgs: np.ndarray
    zpterrs: np.ndarray
    gains: np.ndarray
    readnoises: np.ndarray
    nexposes: np.ndarray
    template_skysigs: np.ndarray
    template_readnoises: np.ndarray
    template_zpts: np.ndarray
    # APPEND: <flag> before the row: ORed into PHOTFLAG, and the row stays
    # after the sorted ones (snlc_sim.c:19745, :21719, :25633)
    append_photflags: np.ndarray
    season_starts: np.ndarray = field(default_factory=lambda: np.full(MXSEASON_SIMLIB, np.inf, dtype=np.float32))
    season_ends: np.ndarray = field(default_factory=lambda: np.full(MXSEASON_SIMLIB, -np.inf, dtype=np.float32))
    field_name: str = "FIELD_NONAME"  # SIMLIB FIELD: name (e.g., 'C1', 'DDF')
    groupid_hostlib: tuple = ()  # GROUPID_HOSTLIB values (empty = no constraint)
    # SPECTROGRAPH: rows (MJD, TEXPOSE), used under USE_SIMLIB_SPECTROGRAPH
    spec_mjds: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    spec_texposes: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    # TAKE_SPECTRUM header lines, used under USE_SIMLIB_TAKE_SPECTRUM
    take_spectrum: tuple = ()
    # single-valued header keys as written, for the USE_SIMLIB_* overrides
    header_values: dict = field(default_factory=dict)
    # GENRANGE_<name>/GENSIGMA_<name> as (lo, hi, sigma) (snlc_sim.c:22173-22192)
    gengauss: dict = field(default_factory=dict)
    # NaN when absent (snlc_sim.c:20144-20156, :22358)
    cutwin_redshift: tuple = (float('nan'), float('nan'))
    genrange_redshift: tuple = (float('nan'), float('nan'))
    redshift_keys: tuple = ()
    subsurvey: str = ''


def _resolve_simlib_path(path):
    """The SIMLIB that exists on disk for ``path``, plain or gzipped.

    SNANA's open_TEXTgz treats a plain and a gzipped copy of the same SIMLIB as
    an error rather than picking one, since which is current is ambiguous.
    """
    path = Path(path)
    if path.exists() and path.with_name(path.name + '.gz').exists():
        raise SNANAError(
            f"Found both {path} and {path}.gz. Remove one; which is current "
            f"is ambiguous.", function_name="_resolve_simlib_path")
    resolved = resolve_fits_path(str(path))
    if not Path(resolved).exists():
        raise SNANAError(f"SIMLIB file not found: {path}",
                         function_name="_resolve_simlib_path")
    return resolved


def _header_key_runs(line):
    """``(key, values)`` for each ``KEY:`` token on a header line, the values
    being the tokens up to the next key (SNANA walks the word list the same
    way, snlc_sim.c:19690-19758)."""
    runs = []
    for token in line.split():
        if token.endswith(':') and len(token) > 1:
            runs.append((token[:-1], []))
        elif runs:
            runs[-1][1].append(token)
    return runs


def _apply_header_key(field_data, key, words):
    """One entry header key onto the entry (snlc_sim.c:19700-19758,
    :22126-22192); a key with no handler is kept by name as a single value
    for the USE_SIMLIB_* overrides."""
    if key in _REDSHIFT_KEYS:
        field_data['redshift_keys'] = (*field_data.get('redshift_keys', ()), key)
    if key == 'RA':
        field_data['ra'] = float(words[0])
    elif key in ('DEC', 'DECL'):
        field_data['dec'] = float(words[0])
    elif key == 'NOBS':
        field_data['nobs'] = int(words[0])
    elif key == 'MWEBV':
        field_data['mwebv'] = float(words[0])
    elif key == 'PIXSIZE':
        field_data['pixsize'] = float(words[0])
    elif key == 'FIELD':
        field_data['field_name'] = words[0]
    elif key in ('GROUPID_HOSTLIB', 'HOSTLIB_GROUPID'):
        # either spelling (snlc_sim.c:22249); comma- or space-separated
        field_data['groupid_hostlib'] = tuple(
            int(x) for x in ' '.join(words).replace(',', ' ').split())
    elif key in ('CUTWIN_REDSHIFT', 'REDSHIFT_RANGE'):
        field_data['cutwin_redshift'] = (float(words[0]), float(words[1]))
    elif key == 'GENRANGE_REDSHIFT':
        field_data['genrange_redshift'] = (float(words[0]), float(words[1]))
    elif key.startswith(('GENRANGE_', 'GENSIGMA_')):
        name = key.split('_', 1)[1].replace('PKMJD', 'PEAKMJD')
        gengauss = field_data.setdefault('gengauss', {})
        lo, hi, sigma = gengauss.get(name, (float('nan'), float('nan'), 0.0))
        if key.startswith('GENRANGE_'):
            lo, hi = float(words[0]), float(words[1])
        else:
            sigma = float(words[0])
        gengauss[name] = (lo, hi, sigma)
    elif key == 'SUBSURVEY':
        field_data['subsurvey'] = words[0]
    elif len(words) == 1:
        field_data.setdefault('header_values', {})[key] = words[0]


def _sort_simlib_field(field):
    """Sort observations and spectra, updating observation count and seasons."""
    # Appended rows follow the MJD-sorted ones, in file order (SIMLIB_sortbyMJD,
    # snlc_sim.c:21719).
    appended = field.append_photflags > 0
    order = np.lexsort((np.where(appended, np.arange(appended.size), field.mjds),
                        appended))
    for name in SIMLIB_OBSERVATION_KEYS:
        setattr(field, name, getattr(field, name)[order])
    field.season_starts, field.season_ends = _compute_seasons(field.mjds)
    field.nobs = len(field.mjds)  # header NOBS over-counts (SPECTROGRAPH rows)
    spec_order = np.argsort(field.spec_mjds, kind='stable')
    field.spec_mjds = field.spec_mjds[spec_order]
    field.spec_texposes = field.spec_texposes[spec_order]
    return field


@dataclass
class SimlibData:
    """A parsed SIMLIB: its fields, the filter set, and the survey-level knobs."""

    simlib_file: str
    fields: List[SimlibField] = field(default_factory=list)
    # Built from the SIMLIB's FILTERS header, not from the config.
    filter_mapping: Dict[str, int] = field(default_factory=dict)
    filter_names: List[str] = field(default_factory=list)
    survey_name: str = ''
    # SUBSURVEY_LIST: in the global header (snlc_sim.c:18433)
    subsurvey_list: tuple = ()
    # SIMLIB global saturation overrides; None means the SIMLIB did not set one
    # (snlc_sim.c:18479-18481).
    global_npe_pixel_saturate: Optional[int] = None
    global_photflag_saturate: Optional[int] = None
    fluxerr_cor: FluxErrCorMap = field(default_factory=FluxErrCorMap)
    fluxerr_add: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32))
    # PSF_UNIT: NEA_* (SNDATA.NEA_PSF_UNIT)
    nea_psf_unit: bool = False
    template_texpose_spectrograph: float = 0.0
    flat_field_index: Optional[Dict] = None
    flat_obs_data: Optional[Dict] = None
    # set by the builder: SIMLIB_FIELDLIST prescales
    field_prescales: Optional[np.ndarray] = None


def read_simlib(simlib_file: str, require_docana: bool = True,
                cache_dir: Optional[Path] = None) -> SimlibData:
    """Read a SIMLIB file, via the HDF5 cache when it is newer than the source."""
    data = SimlibData(simlib_file=_resolve_simlib_path(simlib_file))
    check_file_docana(data.simlib_file, require_docana)
    cache_dir = Path(cache_dir or default_cache_dir())
    if not _load_from_cache(data, cache_dir):
        _load_simlib(data)
        _save_to_cache(data, cache_dir)
    rebuild_flat_observations(data)
    return data

def default_cache_dir() -> Path:
    """``<repo_root>/simlib_cache``, where the HDF5 caches live by default."""
    return Path(__file__).parent.parent / 'simlib_cache'


def _get_cache_path(data, cache_dir: Path) -> str:
    """The HDF5 cache file for this SIMLIB, named after its absolute path so
    that identically-named files in different directories do not collide."""
    simlib_path = Path(data.simlib_file)
    safe_path = str(simlib_path.absolute()).replace('/', '_').replace('\\', '_')
    return str(cache_dir / f"{safe_path}.h5")


def _load_from_cache(data, cache_dir: Path) -> bool:
    """Restore a SimlibData from its HDF5 cache; False if the file is
    missing, stale or unreadable."""
    cache_path = Path(_get_cache_path(data, cache_dir))
    if (not cache_path.exists()
            or cache_path.stat().st_mtime < Path(data.simlib_file).stat().st_mtime):
        return False

    try:
        logger.info(f"Loading from binary cache: {cache_path.name}")
        start_time = time.time()

        with h5py.File(cache_path, 'r') as f:
            if f.attrs.get('reader_hash') != _READER_HASH:
                return False
            n_fields = f.attrs['n_fields']

            filter_order = f.attrs['filter_order']
            data.filter_mapping = {c: i for i, c in enumerate(filter_order)}
            data.filter_names = list(filter_order)
            data.survey_name = f.attrs['survey_name']
            data.nea_psf_unit = bool(f.attrs['nea_psf_unit'])
            data.template_texpose_spectrograph = float(
                f.attrs['template_texpose_spectrograph'])
            npe_sat = int(f.attrs['global_npe_pixel_saturate'])
            photflag_sat = int(f.attrs['global_photflag_saturate'])
            data.global_npe_pixel_saturate = None if npe_sat < 0 else npe_sat
            data.global_photflag_saturate = (None if photflag_sat < 0
                                             else photflag_sat)

            all_spec_mjds = f['spectrograph/spec_mjds'][:]
            all_spec_texposes = f['spectrograph/spec_texposes'][:]
            field_spec_count = f['spectrograph/field_spec_count'][:]
            field_spec_start = np.concatenate([[0], np.cumsum(field_spec_count)])
            raw_take = f['spectrograph/take_spectrum'][:]
            all_take_spectrum = [t.decode() if isinstance(t, bytes) else str(t)
                                 for t in raw_take]
            field_take_count = f['spectrograph/field_take_count'][:]
            field_take_start = np.concatenate([[0], np.cumsum(field_take_count)])

            obs_columns = {key: f[f'observations/{key}'][:]
                           for key in SIMLIB_OBSERVATION_KEYS}

            field_obs_start_idx = f['fields/obs_start_idx'][:]
            field_obs_count = f['fields/obs_count'][:]
            field_libids = f['fields/libids'][:]
            field_ras = f['fields/ras'][:]
            field_decs = f['fields/decs'][:]
            field_mwebvs = f['fields/mwebvs'][:]
            field_pixsizes = f['fields/pixsizes'][:]
            entry_headers = [_entry_from_json(s) for s in f['fields/entry_header'][:]]
            data.subsurvey_list = tuple(json.loads(f.attrs['subsurvey_list']))

            field_groupid_strs = [s.decode() if isinstance(s, bytes) else s
                                  for s in f['fields/groupid_hostlib'][:]]
            field_name_strs = [s.decode() if isinstance(s, bytes) else s
                               for s in f['fields/field_names'][:]]

            if 'fluxerr_cor' in f:
                data.fluxerr_cor = FluxErrCorMap(
                    enabled=True,
                    log10snr=f['fluxerr_cor/log10snr'][:],
                    scale=f['fluxerr_cor/scale'][:],
                )
            if 'fluxerr_add' in f:
                data.fluxerr_add = f['fluxerr_add'][:]
            else:
                data.fluxerr_add = np.zeros(
                    len(data.filter_names), dtype=np.float32)

        for i in range(n_fields):
            start_idx = field_obs_start_idx[i]
            count = field_obs_count[i]
            end_idx = start_idx + count

            gid_str = field_groupid_strs[i]
            gid_tuple = tuple(int(x) for x in gid_str.split(',') if x.strip()) if gid_str else ()

            field = _sort_simlib_field(SimlibField(
                libid=int(field_libids[i]),
                ra=float(field_ras[i]),
                dec=float(field_decs[i]),
                mwebv=float(field_mwebvs[i]),
                pixsize=float(field_pixsizes[i]),
                nobs=int(count),
                **{key: col[start_idx:end_idx]
                   for key, col in obs_columns.items()},
                groupid_hostlib=gid_tuple,
                field_name=field_name_strs[i],
                spec_mjds=all_spec_mjds[field_spec_start[i]:field_spec_start[i + 1]],
                take_spectrum=tuple(
                    all_take_spectrum[field_take_start[i]:field_take_start[i + 1]]),
                spec_texposes=all_spec_texposes[field_spec_start[i]:field_spec_start[i + 1]],
                **entry_headers[i],
            ))
            data.fields.append(field)

        elapsed = time.time() - start_time
        total_obs = sum(len(field.mjds) for field in data.fields)
        log.line(logger, f"Loaded {n_fields} fields ({total_obs} obs) from cache "
                 f"in {elapsed:.3f}s")

        return True

    except OSError as e:
        logger.warning(f"Cache load failed: {e}")
        logger.warning("  Falling back to text parsing...")
        return False


def _save_to_cache(data, cache_dir: Path):
    """Write a parsed SimlibData to its HDF5 cache: the per-observation
    columns concatenated into one flat array each, plus the per-entry index
    that slices them."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(_get_cache_path(data, cache_dir))

    try:
        logger.info(f"Saving binary cache: {cache_path.name}")
        preprocess_start = time.time()

        obs_lists = {key: [] for key in SIMLIB_OBSERVATION_KEYS}
        all_spec_mjds = []
        all_take_spectrum = []
        all_spec_texposes = []

        field_obs_start_idx = []
        field_obs_count = []
        field_spec_count = []
        field_take_count = []
        field_libids = []
        field_ras = []
        field_decs = []
        field_mwebvs = []
        field_pixsizes = []
        entry_headers = []

        current_start_idx = 0
        for fld in data.fields:
            n_obs = len(fld.mjds)

            for key in SIMLIB_OBSERVATION_KEYS:
                obs_lists[key].append(getattr(fld, key))
            all_spec_mjds.append(fld.spec_mjds)
            all_spec_texposes.append(fld.spec_texposes)
            all_take_spectrum.extend(fld.take_spectrum)

            field_obs_start_idx.append(current_start_idx)
            field_obs_count.append(n_obs)
            field_spec_count.append(len(fld.spec_mjds))
            field_take_count.append(len(fld.take_spectrum))
            field_libids.append(fld.libid)
            field_ras.append(fld.ra)
            field_decs.append(fld.dec)
            field_mwebvs.append(fld.mwebv)
            field_pixsizes.append(fld.pixsize)
            entry_headers.append(json.dumps({k: getattr(fld, k) for k in _ENTRY_KEYS}))

            current_start_idx += n_obs

        obs_columns = {key: np.concatenate(cols)
                       for key, cols in obs_lists.items()}
        all_spec_mjds = (np.concatenate(all_spec_mjds) if all_spec_mjds
                         else np.empty(0, dtype=np.float64))
        all_spec_texposes = (np.concatenate(all_spec_texposes) if all_spec_texposes
                             else np.empty(0, dtype=np.float64))

        field_obs_start_idx = np.array(field_obs_start_idx, dtype=np.int32)
        field_obs_count = np.array(field_obs_count, dtype=np.int32)
        field_spec_count = np.array(field_spec_count, dtype=np.int32)
        field_take_count = np.array(field_take_count, dtype=np.int32)
        field_libids = np.array(field_libids, dtype=np.int32)
        # float64, the precision of the text parse
        field_ras = np.array(field_ras, dtype=np.float64)
        field_decs = np.array(field_decs, dtype=np.float64)
        field_mwebvs = np.array(field_mwebvs, dtype=np.float64)
        field_pixsizes = np.array(field_pixsizes, dtype=np.float64)

        # Publish only complete files; concurrent readers keep their old snapshot.
        with tempfile.TemporaryDirectory(prefix='.simlib-', dir=cache_dir) as temp_dir:
            temp_path = Path(temp_dir) / 'cache.h5'
            with h5py.File(temp_path, 'w') as f:
                f.attrs['reader_hash'] = _READER_HASH
                f.attrs['simlib_file'] = data.simlib_file
                f.attrs['n_fields'] = len(data.fields)
                f.attrs['n_obs_total'] = len(obs_columns['mjds'])
                f.attrs['max_obs_per_field'] = int(np.max(field_obs_count))
                f.attrs['filter_order'] = ''.join(data.filter_names)
                f.attrs['survey_name'] = data.survey_name
                f.attrs['nea_psf_unit'] = bool(data.nea_psf_unit)
                f.attrs['template_texpose_spectrograph'] = float(
                    data.template_texpose_spectrograph)
                # -1 stores None
                f.attrs['global_npe_pixel_saturate'] = (
                    -1 if data.global_npe_pixel_saturate is None
                    else int(data.global_npe_pixel_saturate))
                f.attrs['global_photflag_saturate'] = (
                    -1 if data.global_photflag_saturate is None
                    else int(data.global_photflag_saturate))

                obs_grp = f.create_group('observations')
                field_grp = f.create_group('fields')

                for key, column in obs_columns.items():
                    obs_grp.create_dataset(key, data=column, compression='gzip',
                                           compression_opts=4)

                spec_grp = f.create_group('spectrograph')
                spec_grp.create_dataset('spec_mjds', data=all_spec_mjds)
                spec_grp.create_dataset('spec_texposes', data=all_spec_texposes)
                spec_grp.create_dataset('field_spec_count', data=field_spec_count)

                spec_grp.create_dataset(
                    'take_spectrum',
                    data=np.array(all_take_spectrum, dtype=object),
                    dtype=h5py.special_dtype(vlen=str))
                spec_grp.create_dataset('field_take_count', data=field_take_count)

                field_grp.create_dataset('obs_start_idx', data=field_obs_start_idx)
                field_grp.create_dataset('obs_count', data=field_obs_count)
                field_grp.create_dataset('libids', data=field_libids)
                field_grp.create_dataset('ras', data=field_ras)
                field_grp.create_dataset('decs', data=field_decs)
                field_grp.create_dataset('mwebvs', data=field_mwebvs)
                field_grp.create_dataset('pixsizes', data=field_pixsizes)
                dt = h5py.string_dtype()
                field_grp.create_dataset('entry_header', data=entry_headers, dtype=dt)
                f.attrs['subsurvey_list'] = json.dumps(list(data.subsurvey_list))

                groupid_strs = [
                    ','.join(str(g) for g in field.groupid_hostlib)
                    for field in data.fields
                ]
                field_grp.create_dataset('groupid_hostlib', data=groupid_strs, dtype=dt)

                field_name_strs = [f.field_name for f in data.fields]
                field_grp.create_dataset('field_names', data=field_name_strs, dtype=dt)

                if data.fluxerr_cor.enabled:
                    fc_grp = f.create_group('fluxerr_cor')
                    fc_grp.create_dataset('log10snr', data=data.fluxerr_cor.log10snr)
                    fc_grp.create_dataset('scale', data=data.fluxerr_cor.scale)

                if np.any(data.fluxerr_add != 0):
                    f.create_dataset('fluxerr_add', data=data.fluxerr_add)

            temp_path.replace(cache_path)

        elapsed = time.time() - preprocess_start
        cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
        log.line(logger, f"Saved cache in {elapsed:.3f}s ({cache_size_mb:.1f} MB)")

    except OSError as e:
        logger.warning(f"Cache save failed: {e}")


def _load_simlib(data):
    """Load and parse SIMLIB file."""
    logger.info(f"Loading SIMLIB file: {data.simlib_file}")
    start_time = time.time()

    with open_text(data.simlib_file) as f:
        content = f.read()

    lines = content.split('\n')

    current_field_data = None
    obs_lists = None
    previous_mwebv = 0.0
    psf_unit = "PIXEL_SIGMA"
    skysig_unit = "ADU_PER_SQPIXEL"
    global_pixsize = 0.0
    global_npe_pixel_saturate = None
    global_photflag_saturate = None
    filter_order = ""
    template_by_field = {}
    current_field_name = global_field = "FIELD_NONAME"
    in_global_header = True
    use_nea_unit = False
    fluxerr_cor_rows = []
    fluxerr_add_line = None

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        if line.startswith('LIBID:'):
            if current_field_data is not None:
                _create_field_from_lists(data, current_field_data, obs_lists)
                previous_mwebv = current_field_data['mwebv']

            parts = line.split()
            libid = int(parts[1])
            # RA/DEC 9999 fail GENRANGE_RA/DEC; MWEBV carries over; FIELD
            # starts as the global header's (init_SIMLIB_HEADER)
            current_field_data = {'libid': libid, 'ra': 9999.0, 'dec': 9999.0,
                                  'mwebv': previous_mwebv, 'pixsize': 0.0,
                                  'field_name': global_field}
            current_append_photflag = 0
            obs_lists = {key: [] for key in _PARSED_OBS_KEYS}
            obs_lists.update(spec_mjds=[], spec_texposes=[], take_spectrum=[])
            for key, words in _header_key_runs(line)[1:]:
                _apply_header_key(current_field_data, key, words)
            current_field_name = current_field_data['field_name']
            continue

        # SURVEY:/FILTERS: may share a line with other keys
        if 'SURVEY:' in line or 'FILTERS:' in line:
            tokens = line.split()
            for i, tok in enumerate(tokens):
                if tok == 'SURVEY:' and i + 1 < len(tokens):
                    data.survey_name = tokens[i + 1]
                elif tok == 'FILTERS:' and i + 1 < len(tokens):
                    filter_order = tokens[i + 1]
                    data.filter_mapping = {c: i for i, c in enumerate(filter_order)}
                    data.filter_names = list(filter_order)

        # a FIELD in the global header, which ends at BEGIN, is every entry's
        # default (snlc_sim.c:18436, :18529, :21898)
        if in_global_header and current_field_data is None:
            tokens = line.split()
            head = tokens[:tokens.index('BEGIN')] if 'BEGIN' in tokens else tokens
            if 'FIELD:' in head:
                global_field = head[head.index('FIELD:') + 1]
            in_global_header = 'BEGIN' not in tokens

        if line.startswith('PSF_UNIT:'):
            parts = line.replace(':', ' ').split()
            unit_idx = parts.index('PSF_UNIT') + 1 if 'PSF_UNIT' in parts else 1
            psf_unit = parts[unit_idx]
            if psf_unit not in ('PIXEL_SIGMA', 'ARCSEC_FWHM', 'NEA_PIXEL', 'NEA_ARCSECSQ'):
                raise SNANAError(f"Invalid PSF_UNIT '{psf_unit}' in SIMLIB header")
            use_nea_unit = psf_unit in ('NEA_PIXEL', 'NEA_ARCSECSQ')
            data.nea_psf_unit = use_nea_unit
            continue

        if line.startswith('SKYSIG_UNIT:'):
            parts = line.replace(':', ' ').split()
            unit_idx = parts.index('SKYSIG_UNIT') + 1 if 'SKYSIG_UNIT' in parts else 1
            skysig_unit = parts[unit_idx]
            if skysig_unit not in ('ADU_PER_SQPIXEL', 'ADU_PER_SQARCSEC'):
                raise SNANAError(f"Invalid SKYSIG_UNIT '{skysig_unit}' in SIMLIB header")
            continue

        if line.startswith('PIXSIZE:'):
            parts = line.replace(':', ' ').split()
            pix_idx = parts.index('PIXSIZE') + 1 if 'PIXSIZE' in parts else 1
            global_pixsize = float(parts[pix_idx])
            continue

        if line.startswith('SUBSURVEY_LIST:'):
            data.subsurvey_list = tuple(line.split(':', 1)[1].replace(',', ' ').split())
            continue

        if line.startswith('TEXPOSE('):
            raise SNANAError(
                f"SIMLIB header key {line.split(':', 1)[0]} is not implemented: SNANA "
                f"writes these per-band exposure times to the PHOT TEXPOSE column and "
                f"feeds them to NONLINEARITY; jax writes TEXPOSE 0.",
                function_name="parse_simlib")

        if line.startswith('NPE_PIXEL_SATURATE:'):
            parts = line.replace(':', ' ').split()
            idx = parts.index('NPE_PIXEL_SATURATE') + 1
            global_npe_pixel_saturate = int(parts[idx])
            continue

        if line.startswith('PHOTFLAG_SATURATE:'):
            parts = line.replace(':', ' ').split()
            idx = parts.index('PHOTFLAG_SATURATE') + 1
            global_photflag_saturate = int(parts[idx])
            continue

        if line.startswith('FLUXERR_COR:'):
            parts = line.split()
            filt_chars = parts[1]
            log10snr = float(parts[2])
            scales = np.array([float(x) for x in parts[3:3 + len(filt_chars)]],
                              dtype=np.float32)
            fluxerr_cor_rows.append((filt_chars, log10snr, scales))
            continue

        if line.startswith('FLUXERR_ADD:'):
            parts = line.split()
            filt_chars = parts[1]
            values = [float(x) for x in parts[2:2 + len(filt_chars)]]
            fluxerr_add_line = (filt_chars, values)
            continue

        # entry header keys, from the token stream as SNANA walks it: any key
        # may share a line with any other (snlc_sim.c:19690-19758)
        if (current_field_data is not None and ':' in line
                and not line.startswith(('S:', 'TEMPLATE_', 'SPECTROGRAPH:',
                                         'TAKE_SPECTRUM', 'WARP_SPECTRUM',
                                         'END_LIBID:'))):
            for key, words in _header_key_runs(line):
                if key == 'APPEND':
                    # the flag for every row that follows in this entry
                    current_append_photflag = int(words[0])
                else:
                    _apply_header_key(current_field_data, key, words)
            current_field_name = current_field_data['field_name']
            continue

        if line.startswith('TEMPLATE_'):
            if not filter_order:
                raise SNANAError("SIMLIB TEMPLATE_* encountered before FILTERS")
            parts = line.replace(':', ' ').split()
            key = parts[0]
            values = parts[1:]
            if key == 'TEMPLATE_TEXPOSE_SPECTROGRAPH':
                # single global value (SNANA SIMLIB_TEMPLATE, snlc_sim.c:19769)
                data.template_texpose_spectrograph = float(values[0])
                continue
            if len(values) < len(filter_order):
                continue  # not a per-filter template key
            if current_field_name not in template_by_field:
                n_filt = len(data.filter_mapping)
                template_by_field[current_field_name] = {
                    'skysig': np.zeros(n_filt, dtype=np.float32),
                    'readnoise': np.zeros(n_filt, dtype=np.float32),
                    'zpt': np.zeros(n_filt, dtype=np.float32),
                }
            target = template_by_field[current_field_name]
            for ifilt, fchar in enumerate(filter_order):
                if fchar not in data.filter_mapping:
                    raise SNANAError(f"Unknown filter '{fchar}' in SIMLIB FILTERS")
                idx = data.filter_mapping[fchar]
                val = float(values[ifilt])
                if key == 'TEMPLATE_SKYSIG':
                    target['skysig'][idx] = val
                elif key == 'TEMPLATE_CCDSIG':
                    target['readnoise'][idx] = val
                elif key == 'TEMPLATE_ZPT':
                    target['zpt'][idx] = val
            continue

        if line.startswith('S:') and obs_lists is not None:
            parts = line.split()
            min_cols = 11 if use_nea_unit else 13
            if len(parts) < min_cols:
                raise SNANAError("SIMLIB S: line missing required columns")
            idx = 1
            mjd = float(parts[idx]); idx += 1
            _, nexpose = _parse_idexpt_token(parts[idx]); idx += 1
            filter_char = parts[idx]; idx += 1
            gain = float(parts[idx]); idx += 1
            readnoise = float(parts[idx]); idx += 1
            skysig = float(parts[idx]); idx += 1

            pixsize = current_field_data['pixsize']
            if pixsize <= 0.0:
                pixsize = global_pixsize
            if pixsize <= 0.0:
                raise SNANAError("SIMLIB missing PIXSIZE for SKYSIG/PSF conversions")

            if use_nea_unit:
                nea = float(parts[idx]); idx += 1
                if psf_unit == 'NEA_ARCSECSQ':
                    nea /= (pixsize * pixsize)
                psf1 = np.sqrt(nea / (2.0 * TWOPI))
                psf2 = 0.0
                psf_ratio = 0.0
            else:
                psf1 = float(parts[idx]); idx += 1
                psf2 = float(parts[idx]); idx += 1
                psf_ratio = float(parts[idx]); idx += 1
                if psf_unit == 'ARCSEC_FWHM':
                    psf1 /= (pixsize * FWHM_SIGMA_RATIO)
                    psf2 /= (pixsize * FWHM_SIGMA_RATIO)
            zptavg = float(parts[idx]); idx += 1
            zpterr = float(parts[idx]); idx += 1
            _ = float(parts[idx]); idx += 1

            if skysig_unit == 'ADU_PER_SQARCSEC':
                skysig *= pixsize

            template_values = template_by_field.get(current_field_name)
            if template_values is None:
                template_values = template_by_field.get("ALLFIELDS")
            if template_values is None:
                template_skysig = 0.0
                template_readnoise = 0.0
                template_zpt = 0.0
            else:
                if filter_char not in data.filter_mapping:
                    raise SNANAError(f"Unknown filter '{filter_char}' in SIMLIB S: line")
                fidx = data.filter_mapping[filter_char]
                template_skysig = float(template_values['skysig'][fidx])
                template_readnoise = float(template_values['readnoise'][fidx])
                template_zpt = float(template_values['zpt'][fidx])

            obs_lists['mjds'].append(mjd)
            obs_lists['filter_indices'].append(data.filter_mapping.get(filter_char, -1))
            obs_lists['gains'].append(gain)
            obs_lists['readnoises'].append(readnoise)
            obs_lists['nexposes'].append(nexpose)
            obs_lists['skysigs'].append(skysig)
            obs_lists['psf1s'].append(psf1)
            obs_lists['psf2s'].append(psf2)
            obs_lists['psf_ratios'].append(psf_ratio)
            obs_lists['zptavgs'].append(zptavg)
            obs_lists['zpterrs'].append(zpterr)
            obs_lists['template_skysigs'].append(template_skysig)
            obs_lists['template_readnoises'].append(template_readnoise)
            obs_lists['template_zpts'].append(template_zpt)
            obs_lists['append_photflags'].append(current_append_photflag)

        elif line.startswith('SPECTROGRAPH:') and obs_lists is not None:
            parts = line.split()
            obs_lists['spec_mjds'].append(float(parts[1]))
            obs_lists['spec_texposes'].append(float(parts[2]))

        elif (line.startswith('TAKE_SPECTRUM') or line.startswith('WARP_SPECTRUM')) \
                and obs_lists is not None:
            # raw text, parsed by the consumer (parse_input_TAKE_SPECTRUM, snlc_sim.c:22269)
            obs_lists['take_spectrum'].append(line.strip())

        elif line.startswith('END_LIBID:'):
            continue

    if current_field_data is not None:
        _create_field_from_lists(data, current_field_data, obs_lists)

    data.global_npe_pixel_saturate = global_npe_pixel_saturate
    data.global_photflag_saturate = global_photflag_saturate
    data.fluxerr_cor = _build_fluxerr_cor_map(
        fluxerr_cor_rows, data.filter_mapping)
    data.fluxerr_add = _build_fluxerr_add(
        fluxerr_add_line, data.filter_mapping)

    elapsed = time.time() - start_time
    log.line(logger, f"Loaded {len(data.fields)} fields from SIMLIB in {elapsed:.3f}s")

    total_obs = sum(len(field.mjds) for field in data.fields)
    log.line(logger, f"Total observations: {total_obs}")

    # Count observations per filter in one pass over the flat filter-index array
    n_filt = len(data.filter_names)
    all_indices = np.concatenate([f.filter_indices for f in data.fields]) if data.fields else np.array([], dtype=np.int8)
    valid = all_indices[all_indices >= 0]
    filter_counts = np.bincount(valid, minlength=n_filt)[:n_filt]

    log.line(logger, "Observations per filter:")
    for i, count in enumerate(filter_counts):
        if count > 0:
            log.line(logger, f"{data.filter_names[i]}: {count}", indent=2)


def rebuild_flat_observations(data) -> None:
    """Rebuild flat_obs_data and flat_field_index from data.fields."""
    flat_obs = {}
    for k in SIMLIB_OBSERVATION_KEYS:
        arrs = [getattr(f, k) for f in data.fields]
        flat_obs[k] = np.concatenate(arrs) if arrs else np.array([])
    obs_counts = np.array([f.nobs for f in data.fields], dtype=np.int32)
    if len(obs_counts):
        obs_starts = np.concatenate(
            ([0], np.cumsum(obs_counts)[:-1])).astype(np.int32)
    else:
        obs_starts = np.array([], dtype=np.int32)
    flat_idx = {
        'obs_start_idx': obs_starts,
        'obs_count': obs_counts,
        'libids': np.array([f.libid for f in data.fields], dtype=np.int32),
        'ras': np.array([f.ra for f in data.fields], dtype=np.float32),
        'decs': np.array([f.dec for f in data.fields], dtype=np.float32),
        'mwebvs': np.array([f.mwebv for f in data.fields], dtype=np.float32),
        'pixsizes': jnp.asarray([f.pixsize for f in data.fields]),
    }
    data.flat_obs_data = flat_obs
    data.flat_field_index = flat_idx


def _create_field_from_lists(data, field_data: Dict, obs_lists: Dict):
    """Create a SimlibField from the lists one LIBID's S: lines filled."""
    columns = {key: np.array(obs_lists[key], dtype=_PER_OBS_DTYPES[key])
               for key in _PARSED_OBS_KEYS}
    columns['neas'] = _compute_noise_equivalent_aperture(
        columns['psf1s'], columns['psf2s'], columns['psf_ratios'])

    field = _sort_simlib_field(SimlibField(
        libid=field_data['libid'],
        ra=field_data['ra'],
        dec=field_data['dec'],
        mwebv=field_data['mwebv'],
        pixsize=field_data['pixsize'],
        nobs=len(columns['mjds']),
        **columns,
        field_name=field_data['field_name'],
        groupid_hostlib=field_data.get('groupid_hostlib', ()),
        spec_mjds=np.array(obs_lists['spec_mjds'], dtype=np.float64),
        spec_texposes=np.array(obs_lists['spec_texposes'], dtype=np.float64),
        take_spectrum=tuple(obs_lists['take_spectrum']),
        **{k: field_data[k] for k in _ENTRY_KEYS if k in field_data},
    ))
    data.fields.append(field)


FLOAT_PROPERTY_NAMES = [
    'skysig', 'psf1', 'psf2', 'psf_ratio', 'nea',
    'zptavg', 'zpterr', 'gain', 'readnoise', 'nexpose',
    'template_skysig', 'template_readnoise', 'template_zpt',
    'append_photflag',
]

FLAT_OBS_KEYS = {
    'mjd': 'mjds', 'filter_idx': 'filter_indices',
    'skysig': 'skysigs', 'psf1': 'psf1s', 'psf2': 'psf2s',
    'psf_ratio': 'psf_ratios', 'nea': 'neas',
    'zptavg': 'zptavgs', 'zpterr': 'zpterrs',
    'gain': 'gains', 'readnoise': 'readnoises', 'nexpose': 'nexposes',
    'template_skysig': 'template_skysigs',
    'template_readnoise': 'template_readnoises',
    'template_zpt': 'template_zpts',
    'append_photflag': 'append_photflags',
}


def build_jax_field_cache(simlib, filter_translation) -> Dict[str, jnp.ndarray]:
    """Build padded per-field observation arrays from a SimlibData.

    ``simlib`` has flat_obs_data and flat_field_index. ``filter_translation``
    is an array mapping each SIMLIB filter index to the canonical
    (GENFILTERS-restricted) index, ``-1`` for bands not in GENFILTERS.
    Observations of dropped bands are pushed out of MJD range so the
    scheduler excludes them; kept bands are reindexed.

    Returns a dict of JAX arrays for use with schedule_all.
    """
    flat = simlib.flat_obs_data
    flat_idx = simlib.flat_field_index
    n_fields = len(simlib.fields)
    max_obs_per_field = int(np.max(flat_idx['obs_count']))
    trans = np.asarray(filter_translation, dtype=np.int32)

    mjds = np.full((n_fields, max_obs_per_field), np.inf, dtype=np.float64)
    filter_idx = np.full((n_fields, max_obs_per_field), -1, dtype=np.int32)
    float_stack = np.full((n_fields, max_obs_per_field, len(FLOAT_PROPERTY_NAMES)),
                          np.nan, dtype=np.float32)

    for i in range(n_fields):
        start = flat_idx['obs_start_idx'][i]
        count = flat_idx['obs_count'][i]
        end = start + count
        canon = trans[flat['filter_indices'][start:end]]
        mjds[i, :count] = np.where(canon < 0, np.inf, flat['mjds'][start:end])
        filter_idx[i, :count] = canon
        for j, prop in enumerate(FLOAT_PROPERTY_NAMES):
            float_stack[i, :count, j] = flat[FLAT_OBS_KEYS[prop]][start:end]

    season_starts = np.stack([f.season_starts for f in simlib.fields]) \
        if n_fields else np.zeros((0, MXSEASON_SIMLIB), dtype=np.float32)
    season_ends = np.stack([f.season_ends for f in simlib.fields]) \
        if n_fields else np.zeros((0, MXSEASON_SIMLIB), dtype=np.float32)

    return {
        'mjds': jnp.array(mjds),
        'filter_idx': jnp.array(filter_idx),
        'float_stack': jnp.array(float_stack),
        # each entry's first index into flat_obs_data
        'obs_start': jnp.array(flat_idx['obs_start_idx'], dtype=jnp.int32),
        'meta_libid': jnp.array(flat_idx['libids'], dtype=jnp.int32),
        'season_starts': jnp.array(season_starts, dtype=jnp.float32),
        'season_ends': jnp.array(season_ends, dtype=jnp.float32),
        'n_fields': n_fields,
        'max_obs_per_field': max_obs_per_field,
    }


def max_obs_per_event(field_cache: Dict, window_days: float,
                      mskopt: int) -> int:
    """The most epochs one event can take from any SIMLIB entry.

    An event is given the epochs of a single entry that fall inside its
    observation window, so the longest schedule a run can produce is the
    densest ``window_days`` stretch of any entry. SIMLIB_MSKOPT bit 256
    replaces the window by the whole entry and bit 128 by the season holding
    the peak, matching the branches of `schedule_all`. Every other
    scheduler cut (SIMLIB_NSKIPMJD, GENRANGE_MJD, the out-of-survey test)
    only removes epochs, so the count stands as an upper bound.
    """
    mjds = np.asarray(field_cache['mjds'])
    real = np.asarray(field_cache['filter_idx']) >= 0
    if mskopt & 256:
        return int(np.max(real.sum(axis=1), initial=0))
    if mskopt & 128:
        starts = np.asarray(field_cache['season_starts'])
        ends = np.asarray(field_cache['season_ends'])
        widest = 0
        for j in range(starts.shape[1]):
            in_season = real & (mjds >= starts[:, j:j + 1]) \
                & (mjds <= ends[:, j:j + 1])
            widest = max(widest, int(np.max(in_season.sum(axis=1), initial=0)))
        return widest
    widest = 0
    for row, keep in zip(mjds, real):
        epochs = np.sort(row[keep])
        if epochs.size == 0:
            continue
        last = np.searchsorted(epochs, epochs + window_days, side='right')
        widest = max(widest, int(np.max(last - np.arange(epochs.size))))
    return widest


# in schedule_one_sn's argument order
_FIELD_CACHE_ARRAYS = (
    'mjds', 'filter_idx', 'float_stack', 'obs_start',
    'meta_libid',
    'season_starts', 'season_ends',
)


def scheduler_tables(field_cache: Dict) -> Dict:
    """The field-cache arrays the scheduler reads, which it takes as an
    argument: closing over them would make them constants of every program
    that schedules, and at survey scale the float stack alone is hundreds of
    megabytes."""
    return {k: field_cache[k] for k in _FIELD_CACHE_ARRAYS}


def schedule_one_sn(field_idx, mjd_min, mjd_max, peak_mjd, redshift_helio,
                    field_mjds, field_filters, field_float_stack,
                    field_obs_start, field_meta_libid, field_season_starts,
                    field_season_ends, nskip_ps, nskip_tmin, *, max_obs,
                    mskopt, tobs_pad):
    max_fobs = field_mjds.shape[1]
    use_entire_survey = bool(mskopt & 256)
    use_entire_season = bool(mskopt & 128) and not use_entire_survey
    fmjds = field_mjds[field_idx]
    if use_entire_survey:
        mask = field_filters[field_idx] >= 0
    elif use_entire_season:
        starts = field_season_starts[field_idx]
        ends = field_season_ends[field_idx]
        in_season = (starts <= peak_mjd) & (peak_mjd <= ends)
        any_match = jnp.any(in_season)
        iseason = jnp.argmax(in_season)
        season_start = jnp.where(any_match, starts[iseason], jnp.inf)
        season_end = jnp.where(any_match, ends[iseason], -jnp.inf)
        mask = (fmjds >= season_start) & (fmjds <= season_end)
    else:
        mask = (fmjds >= mjd_min) & (fmjds <= mjd_max)

    # outside the surviving seasons: no epochs (set_SIMLIB_MJDrange, snlc_sim.c:21411)
    survey_min = jnp.min(field_season_starts[field_idx])
    survey_max = jnp.max(field_season_ends[field_idx])
    outside = ((mjd_max < survey_min + tobs_pad)
               | (mjd_min > survey_max - tobs_pad))
    mask = mask & ~outside

    # SIMLIB_NSKIPMJD after the window cut (keep_SIMLIB_MJD, snlc_sim.c:21250)
    if nskip_ps.shape[0] > 0:
        # heliocentric Trest; SNANA uses zCMB
        trest = (fmjds - peak_mjd) / (1.0 + redshift_helio)
        changed = jnp.concatenate([
            jnp.array([False]), (fmjds[1:] - fmjds[:-1]) > 1.0e-6])
        nmjd = jnp.cumsum((mask & changed).astype(jnp.int32))
        prescale = jnp.ones_like(nmjd)
        for ips in range(nskip_ps.shape[0]):
            prescale = jnp.where(trest >= nskip_tmin[ips],
                                 nskip_ps[ips], prescale)
        mask = mask & (nmjd % prescale == 0)

    # top_k compacts kept epochs first in MJD order: jnp.where(mask)[0] has no static length
    selection_order = mask.astype(jnp.float32) * max_fobs - jnp.arange(max_fobs)
    _, indices = lax.top_k(selection_order, max_obs)
    valid = mask[indices]

    raw_mjds = fmjds[indices]
    out_mjds = jnp.where(valid, raw_mjds, jnp.nan)
    out_filters = jnp.where(valid, field_filters[field_idx, indices], -1)
    out_floats = jnp.where(valid[:, None],
                           field_float_stack[field_idx, indices, :], jnp.nan)
    out_index = jnp.where(valid, field_obs_start[field_idx] + indices, -1)
    n_valid = jnp.sum(valid)
    libid = field_meta_libid[field_idx]

    return (out_mjds, out_filters, out_floats, out_index,
            valid, n_valid, libid)

@jax.jit(static_argnames=('max_obs', 'mskopt'))
def schedule_all(tables, field_indices, mjd_mins, mjd_maxs, peak_mjds,
                 redshifts_helio, *, max_obs, mskopt, tobs_pad):
    """Schedule a batch using explicit SIMLIB arrays and numerical bounds."""
    def event(field_idx, mjd_min, mjd_max, peak_mjd, redshift):
        return schedule_one_sn(
            field_idx, mjd_min, mjd_max, peak_mjd, redshift,
            *(tables[k] for k in _FIELD_CACHE_ARRAYS),
            tables['nskip_prescale'], tables['nskip_trestmin'],
            max_obs=max_obs, mskopt=mskopt, tobs_pad=tobs_pad)
    return jax.vmap(event)(field_indices, mjd_mins, mjd_maxs, peak_mjds,
                           redshifts_helio)


def survey_id(name: str) -> int:
    """SURVEY.DEF integer ID for ``name`` (get_IDSURVEY, snlc_sim.c:18574).

    Includes surveys with no observatory coordinates. SNANA aborts on an unmatched name
    (SIMLIB_prepGlobalHeader), so this raises.
    """
    name_to_id = load_survey_ids()[0]
    if name not in name_to_id:
        raise SNANAError(
            f"Invalid 'SURVEY: {name}' in SIMLIB header; check valid SURVEY "
            f"names in $SNDATA_ROOT/SURVEY.DEF")
    return name_to_id[name]


def resolve_survey(header_name: str, override: str) -> tuple:
    """(name, IDSURVEY): the SIMLIB header SURVEY after the SIMLIB_SURVEY
    override. SNANA applies the override while reading the SIMLIB header
    (snlc_sim.c:18428) and takes the ID from the overridden name (:18576), so
    both come from one resolution.
    """
    name = override or header_name
    return name, survey_id(name)


def survey_from_simlib_file(simlib_file: str, override: str) -> tuple:
    """:func:`resolve_survey` for callers that run before the SIMLIB is
    loaded. Opens through the gzip-aware SIMLIB path resolution, so a
    gzipped SIMLIB reads the same as in the main reader."""
    header_name = None
    with open_text(resolve_fits_path(os.path.expandvars(simlib_file))) as fh:
        for line in fh:
            if line.startswith('SURVEY:'):
                header_name = line.split()[1]
                break
    return resolve_survey(header_name, override)


def prepare_season_boundaries(resources):
    """Stack the per-field season ranges as padded JAX arrays.

    Seasons come from the SIMLIB reader's `_compute_seasons`
    (TGAP_SEASON_SIMLIB=90, snlc_sim.h:72) with SIMLIB_MINSEASON pruning
    already applied by the builder, matching SNANA, which removes short
    seasons before computing DTSEASON_PEAK (snlc_sim.c:21538-21559).
    Pruned or padded slots carry +inf/-inf and are never selected.
    ``resources`` is the built LightCurveSimulator, carrying .fields.

    Returns ``(season_starts, season_ends)``, each (n_fields,
    MXSEASON_SIMLIB) jnp.float32.
    """
    starts = np.stack([f.season_starts for f in resources.fields])
    ends = np.stack([f.season_ends for f in resources.fields])
    return jnp.array(starts, dtype=jnp.float32), jnp.array(ends, dtype=jnp.float32)


def compute_dtseason_peak(peak_mjds, field_indices, season_starts,
                          season_ends):
    """DTSEASON_PEAK inside JIT: the season-edge distance of smallest
    magnitude, with SNANA's sign convention: positive if in season
    (snlc_sim.c:21542-21559). Per season the candidates are
    PEAKMJD - MJD_MIN and MJD_MAX - PEAKMJD; the one with the smallest ``|DT|`` wins,
    first candidate on ties as in SNANA's strict `<` update loop. A field
    with no surviving season keeps SNANA's DT_MIN init 999999.9.

    ``season_starts``/``season_ends`` are (n_fields, n_seasons), from
    prepare_season_boundaries. Returns the (n_sne,) signed DTSEASON_PEAK.
    """
    starts = season_starts[field_indices]
    ends = season_ends[field_indices]
    valid = jnp.isfinite(starts)
    dt_start = peak_mjds[:, None] - starts
    dt_end = ends - peak_mjds[:, None]
    # (start, end) interleaved per season, SNANA's candidate order
    cands = jnp.stack([dt_start, dt_end], axis=-1).reshape(starts.shape[0], -1)
    cand_valid = jnp.repeat(valid, 2, axis=1)
    abs_cands = jnp.where(cand_valid, jnp.abs(cands), jnp.inf)
    min_idx = jnp.argmin(abs_cands, axis=1)
    dtseason = cands[jnp.arange(cands.shape[0]), min_idx]
    return jnp.where(jnp.any(cand_valid, axis=1), dtseason, 999999.9)


def simlib_redshift_windows(simlib, z_range):
    """The entries grouped by distinct CUTWIN_REDSHIFT window for the entry
    draw, or None when no entry has one. The windows must between them cover
    the generated range, or SNANA exhausts the library.
    """
    z_lo, z_hi = z_range
    windows = np.array([f.cutwin_redshift for f in simlib.fields])
    if np.isnan(windows).all():
        return None
    windows = np.where(np.isnan(windows), [[-np.inf, np.inf]], windows)
    order = np.lexsort((windows[:, 1], windows[:, 0]))
    distinct, counts = np.unique(windows[order], axis=0, return_counts=True)
    covered = z_lo
    for lo, hi in distinct:
        if lo <= covered:
            covered = max(covered, hi)
    if covered < z_hi:
        raise SNANAError(f"SIMLIB CUTWIN_REDSHIFT windows do not cover "
                         f"GENRANGE_REDSHIFT {z_range}", function_name="build_simulation")
    starts = np.concatenate(([0], np.cumsum(counts)))
    return {'lo': distinct[:, 0], 'hi': distinct[:, 1], 'start': starts,
            'count': counts, 'order': order}


def _prune_short_seasons(field, min_season: float):
    """SIMLIB_MINSEASON: drop seasons shorter than `min_season` from the entry's
    season bookkeeping, leaving its epochs in place (remove_short_SIMLIB_SEASON,
    snlc_sim.c:21584). The entry stays in the library even when no season
    survives: SNANA leaves the LIBID in the cycle with an empty MJD range
    (:21649), and the scheduler rejects its events.
    """
    if min_season <= 0.0:
        return field
    starts, ends = field.season_starts, field.season_ends
    short = np.isfinite(starts) & ((ends - starts) < min_season)
    if short.any():
        field.season_starts = np.where(short, np.inf, starts)
        field.season_ends = np.where(short, -np.inf, ends)
    return field


def apply_zpt_shift(simlib: SimlibData, shifts: Dict[str, float]) -> None:
    """FUDGESHIFT_ZPT: add per-filter shifts to every epoch's SIMLIB ZPT,
    search and template alike (snlc_sim.c get_SIMLIB_SCALES, :20835)."""
    per_idx = np.array([shifts.get(name, 0.0) for name in simlib.filter_names],
                       dtype=np.float32)
    for f in simlib.fields:
        d = per_idx[f.filter_indices]
        f.zptavgs = f.zptavgs + d
        # SNANA shifts TEMPLATE_ZPT unconditionally for fields carrying a
        # template block (fields without one keep their sentinel).
        if np.any(f.template_skysigs > 0.0) or np.any(f.template_zpts != 0.0):
            f.template_zpts = f.template_zpts + d


def apply_noise_scales(simlib: SimlibData, psf: float, sky: float,
                        read: float, template: float) -> None:
    """FUDGESCALE_PSF / NOISE_SKY / NOISE_READ / NOISE_TEMPLATE: scale every
    epoch's SIMLIB noise terms (snlc_sim.c:20777-20783, :20827-20832).
    Template noise carries the product of the search scale and the template
    scale; NEA scales as psf^2."""
    for field in simlib.fields:
        if sky != 1.0:
            field.skysigs = field.skysigs * sky
        if read != 1.0:
            field.readnoises = field.readnoises * read
        if psf != 1.0:
            field.psf1s = field.psf1s * psf
            field.psf2s = field.psf2s * psf
            field.neas = field.neas * (psf * psf)
        if sky * template != 1.0:
            field.template_skysigs = field.template_skysigs * (sky * template)
        if read * template != 1.0:
            field.template_readnoises = field.template_readnoises * (read * template)


def apply_ignore_template_noise(simlib: SimlibData) -> None:
    """SIMLIB_MSKOPT bit 16 (IGNORE_TEMPLATENOISE): SNANA's reader returns
    before storing any correlated template-noise key (snlc_sim.c:19288-19289),
    so no template variance ever reaches an error term. Zeroing the parsed
    template arrays reproduces that at the same boundary."""
    for f in simlib.fields:
        f.template_skysigs = np.zeros_like(f.template_skysigs)
        f.template_readnoises = np.zeros_like(f.template_readnoises)
        f.template_zpts = np.zeros_like(f.template_zpts)


def validate_simlib_observations(simlib) -> None:
    """One construction-time validation pass over every parsed SIMLIB
    observation.

    Mirrors SNANA's reader sanity ranges (snlc_sim.c:20573-20580), the fatal
    per-epoch observing-condition bounds in gen_fluxNoise_calc
    (snlc_sim.c:27753-27764) and its template-ZPT requirement
    (snlc_sim.c:27808-27813). The noise kernels consume these arrays
    unguarded, so every observation is checked here, once.
    """
    flat = simlib.flat_obs_data
    if len(flat.get('mjds', ())) == 0:
        return
    starts = np.asarray(simlib.flat_field_index['obs_start_idx'])
    libids = np.asarray(simlib.flat_field_index['libids'])
    mjds = np.asarray(flat['mjds'])

    def _check(name, values, bad_mask, rule):
        idx = np.flatnonzero(bad_mask)
        if idx.size == 0:
            return
        i = int(idx[0])
        entry = int(np.searchsorted(starts, i, side='right') - 1)
        raise SNANAError(
            f"SIMLIB {name}={values[i]} (LIBID {libids[entry]}, "
            f"MJD {mjds[i]:.3f}) violates {rule}",
            function_name="validate_simlib_observations",
        )

    zptavg = np.asarray(flat['zptavgs'])
    zpterr = np.asarray(flat['zpterrs'])
    skysig = np.asarray(flat['skysigs'])
    psf1 = np.asarray(flat['psf1s'])
    psf2 = np.asarray(flat['psf2s'])
    psf_ratio = np.asarray(flat['psf_ratios'])
    nea = np.asarray(flat['neas'])
    gain = np.asarray(flat['gains'])
    readnoise = np.asarray(flat['readnoises'])
    nexpose = np.asarray(flat['nexposes'])

    _check('ZPTAVG', zptavg,
           ~np.isfinite(zptavg) | (zptavg < 10.0) | (zptavg > 50.0),
           'ZPTAVG in [10, 50]')
    _check('ZPTERR', zpterr,
           ~np.isfinite(zpterr) | (zpterr < 0.0) | (zpterr > 5.0),
           'ZPTERR in [0, 5]')
    _check('SKYSIG', skysig,
           ~np.isfinite(skysig) | (skysig < 1.0e-4) | (skysig > 1.0e5),
           'SKYSIG in [1e-4, 1e5]')
    _check('PSFSIG1', psf1,
           ~np.isfinite(psf1) | (psf1 < 1.0e-4) | (psf1 > 30.0),
           'PSFSIG1 in [1e-4, 30] pixels')
    _check('PSFSIG2', psf2,
           ~np.isfinite(psf2) | (psf2 < 0.0) | (psf2 > 30.0),
           'PSFSIG2 in [0, 30] pixels')
    _check('PSFRATIO', psf_ratio,
           ~np.isfinite(psf_ratio) | (psf_ratio < 0.0) | (psf_ratio > 1.0),
           'PSFRATIO in [0, 1]')
    _check('NEA', nea, ~np.isfinite(nea) | (nea <= 0.0),
           'NEA > 0 pixel^2')
    _check('GAIN', gain, ~np.isfinite(gain) | (gain <= 0.0),
           'CCD gain > 0 (converts flux to photoelectrons)')
    _check('READNOISE', readnoise,
           ~np.isfinite(readnoise) | (readnoise < 0.0), 'READNOISE >= 0')
    _check('NEXPOSE', nexpose, ~np.isfinite(nexpose) | (nexpose < 1.0),
           'NEXPOSE >= 1 (saturation divides per-exposure flux by it)')

    t_sky = np.asarray(flat['template_skysigs'])
    t_read = np.asarray(flat['template_readnoises'])
    t_zpt = np.asarray(flat['template_zpts'])
    _check('TEMPLATE_SKYSIG', t_sky,
           ~np.isfinite(t_sky) | (t_sky < 0.0),
           'TEMPLATE_SKYSIG >= 0')
    _check('TEMPLATE_READNOISE', t_read,
           ~np.isfinite(t_read) | (t_read < 0.0),
           'TEMPLATE_CCDSIG >= 0')
    _check('TEMPLATE_ZPT', t_zpt,
           (t_sky > 0.0) & (~np.isfinite(t_zpt) | (t_zpt < 10.0)),
           'TEMPLATE_ZPT >= 10 where template noise is present')


def apply_noise_scales_per_filter(simlib: SimlibData, sky_scale: Dict[str, float],
                                   read_scale: Dict[str, float]) -> None:
    """EXPOSURE_TIME per-band sky/read noise scaling (snlc_sim.c:20778, :20827-20832);
    template sky/read carry the same per-band scale."""
    sky = np.array([sky_scale.get(name, 1.0) for name in simlib.filter_names],
                   dtype=np.float32)
    read = np.array([read_scale.get(name, 1.0) for name in simlib.filter_names],
                    dtype=np.float32)
    for f in simlib.fields:
        si = sky[f.filter_indices]
        ri = read[f.filter_indices]
        f.skysigs = f.skysigs * si
        f.template_skysigs = f.template_skysigs * si
        f.readnoises = f.readnoises * ri
        f.template_readnoises = f.template_readnoises * ri


def _field_prescale(field_name: str, prescales) -> int:
    """Prescale for a field, or -1 when it is not listed (SNANA get_string_dict
    with OPT_DICT=1: a **partial** name match, first hit wins)."""
    if prescales is None:
        return 1
    for name, value in prescales.items():
        if name in field_name:
            return int(value)
    return -1


def spell_entry_headers(simlib, spellings) -> None:
    """Key each entry header's population values by parameter name, as an
    entry may spell a parameter any way the input may."""
    for f in simlib.fields:
        f.header_values = {spellings.get(k.lower(), k): v
                           for k, v in f.header_values.items()}
        f.gengauss = {spellings.get(k.lower(), k): v
                      for k, v in f.gengauss.items()}


class InputGenranges(NamedTuple):
    """The input ranges a SIMLIB entry's own redshift, peak-MJD and
    population-parameter windows must overlap, and each entry's DISTANCE as
    a redshift (NaN where it has none or USE_SIMLIB_DISTANCE is off),
    aligned with the unfiltered entries."""
    redshift: Tuple[float, float]
    peakmjd: Tuple[float, float]
    population: Dict[str, Tuple[float, float]]
    distance_redshift: np.ndarray


def _clip_to_genrange(window, genrange):
    """check_SIMLIB_GENRANGE (snlc_sim.c:22288): ``window`` clipped onto
    ``genrange``, or None when they do not overlap. The comparisons are
    strict, so a single value on either edge fails."""
    lo, hi = window
    if lo >= genrange[1] or hi <= genrange[0]:
        return None
    return max(lo, genrange[0]), min(hi, genrange[1])


def _entry_redshift_window(f, s, distance_z):
    """The window set by the last of the entry's REDSHIFT, DISTANCE and
    GENRANGE_REDSHIFT keys that applies, a value as a zero-width window
    (parse_SIMLIB_GENRANGES, snlc_sim.c:22138-22165)."""
    window = (np.nan, np.nan)
    for key in f.redshift_keys:
        if key == 'REDSHIFT' and s.use_simlib_redshift:
            window = (float(f.header_values['REDSHIFT']),) * 2
        elif key == 'DISTANCE' and np.isfinite(distance_z):
            window = (float(distance_z),) * 2
        elif key == 'GENRANGE_REDSHIFT':
            window = f.genrange_redshift
    return window


def _entry_genranges_pass(f, s, genranges, distance_z) -> bool:
    """Whether the entry's redshift and peak-MJD windows overlap the input
    ranges, clipping the windows it draws from in place. The window checked
    is the one the draw uses: the last of the entry's REDSHIFT, DISTANCE
    (each under its USE_SIMLIB flag) and GENRANGE_REDSHIFT keys, a value
    as a zero-width window; a PEAKMJD value under
    USE_SIMLIB_PEAKMJD, else GENRANGE_PEAKMJD, which is read whatever the
    flag (parse_SIMLIB_GENRANGES, snlc_sim.c:22135-22185). SNANA skips an
    entry that fails and reads the next (:20152-20180)."""
    window = _entry_redshift_window(f, s, distance_z)
    if window[0] > 0.0:
        window = _clip_to_genrange(window, genranges.redshift)
        if window is None:
            return False
    f.genrange_redshift = window

    t_value = float(f.header_values.get('PEAKMJD', 'nan')) if s.use_simlib_peakmjd else np.nan
    if np.isfinite(t_value):
        if _clip_to_genrange((t_value, t_value), genranges.peakmjd) is None:
            return False
    elif 'PEAKMJD' in f.gengauss:
        lo, hi, sigma = f.gengauss['PEAKMJD']
        clipped = _clip_to_genrange((lo, hi), genranges.peakmjd)
        if clipped is None:
            return False
        f.gengauss = {**f.gengauss, 'PEAKMJD': (*clipped, sigma)}

    # USE_SIMLIB_SALT2: a population parameter the entry sets, as a value or
    # a GENRANGE, against the input's range for it (:20184-20205)
    if s.use_simlib_salt2:
        for name, genrange in genranges.population.items():
            value = float(f.header_values.get(name, 'nan'))
            if np.isfinite(value):
                if _clip_to_genrange((value, value), genrange) is None:
                    return False
            elif name in f.gengauss:
                lo, hi, sigma = f.gengauss[name]
                clipped = _clip_to_genrange((lo, hi), genrange)
                if clipped is None:
                    return False
                f.gengauss = {**f.gengauss, name: (*clipped, sigma)}
    return True


def apply_simlib_filters(simlib: SimlibData, s, genranges: InputGenranges) -> None:
    """Mutate simlib.fields in-place per the SIMLIB_* knobs and the input
    ranges, and rebuild flat structures."""
    prescales = (None if str(s.simlib_fieldlist).upper() == "ALL"
                 else parse_prescales(s.simlib_fieldlist))
    surviving = []
    for f, distance_z in zip(simlib.fields, genranges.distance_redshift):
        pruned = _prune_short_seasons(f, s.simlib_minseason)
        if pruned.nobs < s.simlib_minobs or pruned.nobs > s.simlib_maxobs:
            continue
        if s.simlib_idstart > 0 and pruned.libid < s.simlib_idstart:
            continue
        if s.simlib_idlock > 0 and pruned.libid != s.simlib_idlock:
            continue
        if pruned.libid in s.simlib_idskip:
            continue
        if s.simlib_prescale > 1 and pruned.libid % s.simlib_prescale != 0:
            continue
        if not (s.genrange_ra[0] <= pruned.ra <= s.genrange_ra[1]):
            continue
        if not (s.genrange_decl[0] <= pruned.dec <= s.genrange_decl[1]):
            continue
        if s.genrange_b != (-90.0, 90.0):
            _, b = eq_to_gal(pruned.ra, pruned.dec)
            if not (s.genrange_b[0] <= float(b) <= s.genrange_b[1]):
                continue
        if not _entry_genranges_pass(pruned, s, genranges, distance_z):
            continue
        surviving.append(pruned)
    if not surviving:
        raise SNANAError(
            f"no SIMLIB entry survives the SIMLIB_* cuts and the input ranges "
            f"GENRANGE_REDSHIFT {genranges.redshift} and GENRANGE_PEAKMJD "
            f"{genranges.peakmjd}", function_name="apply_simlib_filters")
    if s.simlib_mjd_shift != 0.0:
        for f in surviving:
            f.mjds = f.mjds + s.simlib_mjd_shift
            f.season_starts = f.season_starts + s.simlib_mjd_shift
            f.season_ends = f.season_ends + s.simlib_mjd_shift
    _apply_genrange_mjd_exclude(s.genrange_mjd_exclude, surviving)
    simlib.fields = surviving
    # SIMLIB_FIELDLIST rejects the event rather than the entry (SKIP_SIMLIB_FIELD,
    # snlc_sim.c:22587), so the cycle keeps every LIBID and the test is per-SN.
    simlib.field_prescales = np.array(
        [_field_prescale(f.field_name, prescales) for f in surviving],
        dtype=np.int64)
    rebuild_flat_observations(simlib)


def _apply_genrange_mjd_exclude(window, fields) -> None:
    """Drop observations inside GENRANGE_MJD_EXCLUDE (snlc_sim.c:19824)."""
    if window is None or window[0] >= window[1]:
        return
    lo, hi = float(window[0]), float(window[1])
    for f in fields:
        keep = ~((f.mjds > lo) & (f.mjds < hi))
        if keep.all():
            continue
        for k in SIMLIB_OBSERVATION_KEYS:
            setattr(f, k, getattr(f, k)[keep])
        f.nobs = len(f.mjds)
        f.season_starts, f.season_ends = _compute_seasons(f.mjds)


def apply_forceval_psf(simlib, value: float) -> None:
    """SNANA FORCEVAL_PSF (snlc_sim.c:19864): overwrite every per-obs PSF sigma
    (PSFSIG1) with a constant."""
    for field in simlib.fields:
        field.psf1s[:] = value


def prepare_field_data(simlib, population_names):
    """Numerical entry headers used by the batch; prepare them once."""
    fields = simlib.fields
    names = set(population_names) | {'PEAKMJD'}
    names.update(name for field in fields for name in field.header_values)
    gengauss_names = {name for field in fields for name in field.gengauss}
    redshift_windows = np.array([field.genrange_redshift for field in fields])
    if not (redshift_windows[:, 0] > 1.0e-5).any():
        redshift_windows = None
    libids = np.array([field.libid for field in fields], dtype=np.int64)
    prescales = simlib.field_prescales
    return {
        'ra': np.array([field.ra for field in fields], dtype=np.float32),
        'dec': np.array([field.dec for field in fields], dtype=np.float32),
        'keep': (prescales > 0) & (libids % np.maximum(prescales, 1) == 0),
        'redshift_windows': redshift_windows,
        'headers': {name: header_column(fields, name) for name in sorted(names)},
        'gengauss': {
            name: np.array([field.gengauss.get(name, (np.nan, np.nan, 0.0))
                            for field in fields]) for name in sorted(gengauss_names)},
    }


def simlib_job_start(config, n_fields: int) -> int:
    """LIBID offset giving this submit_batch_jobs job its own SIMLIB slice.

    SIMLIB_findStart (snlc_sim.c:19092-19121): job k starts
    ``(JOBID_SHIFT-1) * NLIBID/NJOBTOT`` entries in, so N parallel jobs read
    disjoint slices instead of all starting at the first entry. The GENTYPE term
    staggers which job reads which slice across models. Applied only for a
    batched run over a pool of more than 100 entries, and never together with
    SIMLIB_IDSTART, which sets its own start.

    ``n_fields`` is the post-pruning count; SNANA uses the raw NLIBID header,
    since it slices before reading. Only the union of the jobs matters, and that
    is the same either way. A property of the job rather than of one
    simulation, so it draws off the run's root key (:func:`run_key`) whatever
    key a simulation is given.
    """
    run = config.run
    njobtot, jobid = run.njobtot, run.jobid
    if njobtot <= 0 or n_fields <= 100 or config.survey.simlib_idstart > 0:
        return 0

    gentype = config.rates.gentype
    shift = (jobid + gentype - 1) % njobtot + 1 if gentype > 0 else jobid
    slice_len = n_fields // njobtot

    # Spread the start within the slice when the job cannot consume it all.
    spare = slice_len - run.ngentot_lc
    if run.ngentot_lc > 0 and spare > 5:
        key = stream_key(run_key(config), 'SIMLIB_JOB_START')
        spare = int(jax.random.randint(key, (), 0, spare))
    else:
        spare = 0
    return (shift - 1) * slice_len + spare


def simlib_field_index(key, global_indices, n_fields: int,
                       nrepeat: int, maxranstart: int, batch_start: int = 0):
    """Per-SN SIMLIB entry index from the global SN index.

    With no start option the cycle begins at the first LIBID
    (SIMLIB_findStart, snlc_sim.c:19139-19147); MAXRANSTART draws a random
    start off the root ``key``, and IDSTART/batch handling arrives via
    ``batch_start``.
    """
    if maxranstart > 0:
        bound = min(maxranstart, n_fields)
        start = jax.random.randint(stream_key(key, 'SIMLIB_MAXRANSTART'), (),
                                   0, bound)
    else:
        start = 0
    start = start + batch_start
    return (start + jnp.asarray(global_indices) // nrepeat) % n_fields


def draw_simlib_entry(windows, redshifts, keys):
    """An entry per event among those whose CUTWIN_REDSHIFT window holds the
    redshift: a window with probability proportional to its entry count,
    then an entry within it."""
    lo, hi = jnp.asarray(windows['lo']), jnp.asarray(windows['hi'])
    count = jnp.asarray(windows['count'], dtype=jnp.float32)
    accept = (redshifts[:, None] >= lo) & (redshifts[:, None] <= hi)
    cumulative = jnp.cumsum(jnp.where(accept, count, 0.0), axis=1)
    u = jax.vmap(lambda k: jax.random.uniform(k, shape=(2,)))(keys)
    window = jnp.argmax(cumulative > u[:, :1] * cumulative[:, -1:], axis=1)
    within = jnp.floor(u[:, 1] * count[window]).astype(jnp.int32)
    return jnp.asarray(windows['order'])[jnp.asarray(windows['start'])[window] + within]


def schedule_observations(simulator, tables, field_indices,
                          peak_mjds, redshifts_helio) -> Dict:
    """Schedule each event on the prepared SIMLIB observation tables.

    The epochs are windowed on the drawn peak and redshift, as SNANA reads
    the SIMLIB before the host can change either (set_SIMLIB_MJDrange,
    snlc_sim.c:21300); ``redshifts_helio`` sets the window because it is an
    observer-frame Doppler effect. Phases come later, from the final values
    (:func:`epoch_phases`).
    """
    trest_min, trest_max = simulator.phase_range
    pad = simulator.survey.genrange_tobs_pad
    mjd_mins = peak_mjds + trest_min * (1.0 + redshifts_helio) - pad
    mjd_maxs = peak_mjds + trest_max * (1.0 + redshifts_helio) + pad

    genrange_mjd = simulator.rates.genrange_mjd
    if genrange_mjd is not None:
        mjd_mins = jnp.maximum(mjd_mins, genrange_mjd[0])
        mjd_maxs = jnp.minimum(mjd_maxs, genrange_mjd[1])

    (mjds, filter_indices, float_stack, simlib_obs_idx, valid, n_valid,
     libid) = schedule_all(
        tables, field_indices, mjd_mins, mjd_maxs, peak_mjds,
        redshifts_helio, max_obs=simulator.max_obs,
        mskopt=simulator.survey.simlib_mskopt,
        tobs_pad=simulator.survey.genrange_tobs_pad)

    obs = {
        'mjds': mjds, 'filter_indices': filter_indices,
        # what a finished run keeps of each epoch (`simlib_epochs`)
        'simlib_obs_idx': simlib_obs_idx,
        'n_obs_per_sn': n_valid,
        'libid': libid,
    }
    for i, prop in enumerate(FLOAT_PROPERTY_NAMES):
        obs[FLAT_OBS_KEYS[prop]] = float_stack[:, :, i]
    return obs


def epoch_phases(obs_schedule, params):
    """Rest-frame phase of every epoch, from the final peak MJD and zHEL
    once the host has had its say over both (LOAD_TOBS, snlc_sim.c:13005);
    NaN on padded epochs."""
    return ((obs_schedule['mjds'] - params['peakmjd'][:, None])
            / (1.0 + params['sim_redshift_helio'][:, None]))


def append_spectrograph_photometry(simlib, calib, kcor, exposure_scale) -> None:
    """Append synthetic-band observations at each SIMLIB SPECTROGRAPH epoch so they
    flow through the normal photometry pipeline (SIMLIB_addCadence_SPECTROGRAPH)."""

    if not kcor.syn_filters:
        return
    texpose_t = simlib.template_texpose_spectrograph
    for name in kcor.syn_filters:  # register bands at the end of the shared order
        simlib.filter_mapping.setdefault(name, len(simlib.filter_names))
        if name not in simlib.filter_names:
            simlib.filter_names.append(name)

    for i, fld in enumerate(simlib.fields):
        if fld.spec_mjds.size == 0:
            continue
        rows = []
        for name, (lammin, lammax) in kcor.syn_filters.items():
            fidx = simlib.filter_mapping[name]
            for mjd, tex in zip(fld.spec_mjds, fld.spec_texposes):
                zpt, psf, sks, skt, nea = synthetic_band_zptpsfsky(
                    calib, lammin, lammax, tex * exposure_scale, texpose_t, fld.pixsize)
                rows.append((mjd, fidx, sks, psf, 0., 0., nea, zpt, 0., 1., 0., 1., skt, 0., zpt, 0))
        new = np.array(rows).T
        obs = {k: np.concatenate([getattr(fld, k), new[j]]).astype(getattr(fld, k).dtype)
               for j, k in enumerate(SIMLIB_OBSERVATION_KEYS)}
        simlib.fields[i] = _sort_simlib_field(replace(fld, **obs))

    rebuild_flat_observations(simlib)
