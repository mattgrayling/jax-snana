"""Reader for the SNANA SIMLIB: the survey's observing conditions.

A SIMLIB is a global header followed by one LIBID entry per pointing, each
carrying its own header and one ``S:`` line per observation. The reader
returns a :class:`SimlibData` holding the entries as :class:`SimlibField`
objects and, alongside them, the same per-observation columns concatenated
into flat arrays the JAX scheduler indexes directly. Parsed entries are
cached in HDF5 beside the SIMLIB, keyed on this file's own hash so a change
to the reader rebuilds them.
"""

import logging
import numpy as np
import jax
import jax.lax as lax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import re
from pathlib import Path
import time
import h5py
import hashlib
import json

from jax_snlc_core.io.docana import check_file_docana
from jax_snlc_core.io.fits_path import open_text, resolve_fits_path
from jax_snlc_core.photometry.noise import (
    noise_equivalent_aperture as _compute_noise_equivalent_aperture)
from jax_snlc_core.utils import snana_log as slog
from jax_snlc_core.utils.error_handling import SNANAError

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


_PER_OBS_KEYS = (
    'mjds', 'filter_indices', 'skysigs', 'psf1s', 'psf2s', 'psf_ratios',
    'neas', 'zptavgs', 'zpterrs', 'gains', 'readnoises', 'nexposes',
    'template_skysigs', 'template_readnoises', 'template_zpts',
    'append_photflags',
)

# NEA is derived from the PSF triple, not parsed
_PER_OBS_DTYPES = {k: np.float32 for k in _PER_OBS_KEYS}
_PER_OBS_DTYPES['filter_indices'] = np.int8
_PER_OBS_DTYPES['append_photflags'] = np.int32
_PARSED_OBS_KEYS = tuple(k for k in _PER_OBS_KEYS if k != 'neas')

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


def _build_simlib_field(**kwargs):
    # appended rows follow the MJD-sorted ones, in file order (SIMLIB_sortbyMJD,
    # snlc_sim.c:21719)
    appended = kwargs['append_photflags'] > 0
    order = np.lexsort((np.where(appended, np.arange(appended.size), kwargs['mjds']),
                        appended))
    for k in _PER_OBS_KEYS:
        kwargs[k] = kwargs[k][order]
    kwargs['season_starts'], kwargs['season_ends'] = _compute_seasons(kwargs['mjds'])
    kwargs['nobs'] = len(kwargs['mjds'])  # header NOBS over-counts (SPECTROGRAPH rows)
    spec_order = np.argsort(kwargs['spec_mjds'], kind='stable')
    kwargs['spec_mjds'] = kwargs['spec_mjds'][spec_order]
    kwargs['spec_texposes'] = kwargs['spec_texposes'][spec_order]
    return SimlibField(**kwargs)


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
    # set by the builder: SIMLIB_FIELDLIST prescales, spectrograph bands appended
    field_prescales: Optional[np.ndarray] = None
    spectro_applied: bool = False


def read_simlib(simlib_file: str, require_docana: bool = True,
                cache_dir: Optional[Path] = None) -> SimlibData:
    """Read a SIMLIB file, via the HDF5 cache when it is newer than the source."""
    data = SimlibData(simlib_file=_resolve_simlib_path(simlib_file))
    check_file_docana(data.simlib_file, require_docana)
    _load_simlib_with_cache(data, Path(cache_dir or default_cache_dir()))
    return data

def default_cache_dir() -> Path:
    """``<repo_root>/simlib_cache``, where the HDF5 caches live by default."""
    return Path(__file__).parent.parent.parent / 'simlib_cache'


def _get_cache_path(data, cache_dir: Path) -> str:
    """The HDF5 cache file for this SIMLIB, named after its absolute path so
    that identically-named files in different directories do not collide."""
    simlib_path = Path(data.simlib_file)
    safe_path = str(simlib_path.absolute()).replace('/', '_').replace('\\', '_')
    return str(cache_dir / f"{safe_path}.h5")


def _is_cache_valid(data, cache_dir: Path) -> bool:
    """Check if cache exists and is newer than source SIMLIB file."""
    cache_path = Path(_get_cache_path(data, cache_dir))

    if not cache_path.exists():
        return False

    source_mtime = Path(data.simlib_file).stat().st_mtime
    cache_mtime = cache_path.stat().st_mtime
    if cache_mtime < source_mtime:
        return False

    try:
        with h5py.File(cache_path, 'r') as f:
            return f.attrs.get('reader_hash') == _READER_HASH
    except OSError:
        return False


def _load_from_cache(data, cache_dir: Path) -> bool:
    """Restore a SimlibData from its HDF5 cache; False if the file is
    unreadable."""
    cache_path = Path(_get_cache_path(data, cache_dir))

    try:
        logger.info(f"Loading from binary cache: {cache_path.name}")
        start_time = time.time()

        with h5py.File(cache_path, 'r') as f:
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
                           for key in _PER_OBS_KEYS}

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

            field = _build_simlib_field(
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
            )
            data.fields.append(field)

        elapsed = time.time() - start_time
        total_obs = sum(len(field.mjds) for field in data.fields)
        slog.line(logger, f"Loaded {n_fields} fields ({total_obs} obs) from cache in {elapsed:.3f}s")

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

        obs_lists = {key: [] for key in _PER_OBS_KEYS}
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

            for key in _PER_OBS_KEYS:
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

        with h5py.File(cache_path, 'w') as f:
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

        elapsed = time.time() - preprocess_start
        cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
        slog.line(logger, f"Saved cache in {elapsed:.3f}s ({cache_size_mb:.1f} MB)")

    except OSError as e:
        logger.warning(f"Cache save failed: {e}")
        cache_path.unlink(missing_ok=True)


def _load_simlib_with_cache(data, cache_dir: Path):
    """Load SIMLIB with automatic caching."""
    if not (_is_cache_valid(data, cache_dir) and _load_from_cache(data, cache_dir)):
        _load_simlib(data)
        _save_to_cache(data, cache_dir)
    _rebuild_flat_structures(data)


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
    slog.line(logger, f"Loaded {len(data.fields)} fields from SIMLIB in {elapsed:.3f}s")

    total_obs = sum(len(field.mjds) for field in data.fields)
    slog.line(logger, f"Total observations: {total_obs}")

    # Count observations per filter in one pass over the flat filter-index array
    n_filt = len(data.filter_names)
    all_indices = np.concatenate([f.filter_indices for f in data.fields]) if data.fields else np.array([], dtype=np.int8)
    valid = all_indices[all_indices >= 0]
    filter_counts = np.bincount(valid, minlength=n_filt)[:n_filt]

    slog.line(logger, "Observations per filter:")
    for i, count in enumerate(filter_counts):
        if count > 0:
            slog.line(logger, f"{data.filter_names[i]}: {count}", indent=2)


def _rebuild_flat_structures(data) -> None:
    """Rebuild flat_obs_data and flat_field_index from data.fields."""
    flat_obs = {}
    for k in _PER_OBS_KEYS:
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

    field = _build_simlib_field(
        libid=field_data['libid'],
        ra=field_data['ra'],
        dec=field_data['dec'],
        mwebv=field_data['mwebv'],
        pixsize=field_data['pixsize'],
        **columns,
        field_name=field_data['field_name'],
        groupid_hostlib=field_data.get('groupid_hostlib', ()),
        spec_mjds=np.array(obs_lists['spec_mjds'], dtype=np.float64),
        spec_texposes=np.array(obs_lists['spec_texposes'], dtype=np.float64),
        take_spectrum=tuple(obs_lists['take_spectrum']),
        **{k: field_data[k] for k in _ENTRY_KEYS if k in field_data},
    )
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

    Returns a dict of JAX arrays for use with make_jax_scheduler.
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
    the peak, matching the branches of `make_jax_scheduler`. Every other
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


def make_jax_scheduler(field_cache, max_obs, mskopt, nskipmjd, tobs_pad):
    """Convenience scheduler backed by the shared module-level function."""
    def schedule(tables, field_indices, mjd_mins, mjd_maxs, peak_mjds, redshifts):
        tables = {**tables,
                  'nskip_prescale': jnp.asarray([e[0] for e in nskipmjd], dtype=jnp.int32),
                  'nskip_trestmin': jnp.asarray([e[1] for e in nskipmjd], dtype=jnp.float32)}
        return schedule_all(tables, field_indices, mjd_mins, mjd_maxs,
                            peak_mjds, redshifts, max_obs=max_obs,
                            mskopt=mskopt, tobs_pad=tobs_pad)
    return schedule
