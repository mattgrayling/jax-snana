"""FITS serialisation of the HEAD, PHOT and SPEC tables."""

import logging
import shutil
from pathlib import Path
from typing import Dict

import numpy as np
from isal import igzip_threaded

from jax_snlc_core.io.snana_table import SNANA_VERSION_STRING
from jax_snlc_core.photometry.filter_system import filter_band
from jax_snlc_core.utils import snana_log as slog

from jax_snlc_core.utils.error_handling import SNANAError

from jax_snlc_sim.output.tables import (
    MXCHAR_CCID,
    filters_string, mw_option_keys, sim_model_name, subsurvey_flag)

# threads for the output gzip
GZIP_THREADS = 8

logger = logging.getLogger(__name__)

# SNANA's format-version stamp; 29 is the revision gating the detector
# columns on DETINFO_FLAG (SNFITSIO_CODE_IVERSION, sntools_dataformat_fits.c:1120).
_CODE_IVERSION = 29

# A FITS file is a whole number of 2880-byte blocks, header and data alike.
_BLOCK = 2880
_CARD = 80
# longest quoted string one card holds
_MAX_STRING = 68
# Keywords beyond this go on a HIERARCH card.
_SHORT_KEYWORD = 8

# WRITE_MASK bits (sndata.h:51-61)
_WRITE_MASK_SIM_SNANA = 4
_WRITE_MASK_SIM_SNRMON = 16
_WRITE_MASK_SIM_MODELPAR = 32
_WRITE_MASK_COMPACT = 64
_WRITE_MASK_SPECTRA = 128
_WRITE_MASK_ATMOS = 256
_WRITE_MASK_SED_TRUE = 1024
_WRITE_MASK_COMPACT_NOFLUXCAL = 4096

_PACKAGE_PATH = str(Path(__file__).resolve().parents[2])

# the row closing each event's epochs and each spectrum's bins
# (SNFITSIO_EOE_MARKER, sntools_dataformat_fits.c:1509-1516, :2895)
_EOE_MARKER = -777.0
_PHOT_MARKER = {'MJD': _EOE_MARKER, 'FLUXCAL': _EOE_MARKER,
                'FLUXCALERR': _EOE_MARKER, 'BAND': b'-', 'FIELD': b'XXXX'}

# numpy dtype -> (TFORM code, the comment SNANA's own files carry).
_TFORM = {
    'f8': ('D', '8-byte DOUBLE'), 'f4': ('E', '4-byte REAL'),
    'i8': ('K', '8-byte INTEGER'), 'i4': ('J', '4-byte INTEGER'),
    'i2': ('I', '2-byte INTEGER'), 'i1': ('B', '1-byte INTEGER'),
    'u1': ('B', '1-byte INTEGER'), 'b1': ('L', '1-byte LOGICAL'),
}


def _card(key, value, comment=''):
    """One 80-character header card.

    A keyword beyond the eight characters FITS allows goes on a HIERARCH card,
    which is what SNANA's own headers carry for names like WRITE_MASK_HEAD. A
    string value too long for one card needs CONTINUE cards, which
    ``_string_cards`` produces; truncating it silently here would corrupt the
    header, so it aborts instead.
    """
    if isinstance(value, str):
        if len(value) > _MAX_STRING:
            raise SNANAError(
                f"FITS header value for {key} is {len(value)} characters, "
                f"beyond the {_MAX_STRING} a single card holds",
                function_name="_card")
        text = f"'{value:<8}'"
    elif isinstance(value, bool):
        text = 'T' if value else 'F'
    elif isinstance(value, (int, np.integer)):
        text = str(int(value))
    else:
        text = f"{float(value):.8G}"

    if len(key) > _SHORT_KEYWORD:
        body = f"HIERARCH {key} = {text}"
    elif isinstance(value, str):
        body = f"{key:<8}= " + text.ljust(20)
    else:
        body = f"{key:<8}= {text:>20}"
    card = f"{body} / {comment}" if comment else body
    if len(card) > _CARD:
        raise SNANAError(
            f"FITS card for {key} needs {len(card)} characters, beyond the "
            f"{_CARD} a card holds", function_name="_card")
    return card.ljust(_CARD).encode('ascii')


def _string_cards(key, value):
    """The cards for a string value: one, or a run of CONTINUE cards for a
    value beyond one card, the long-string convention the LONGSTRN key
    declares and SNANA writes its file paths with."""
    head = (f"HIERARCH {key} = " if len(key) > _SHORT_KEYWORD
            else f"{key:<8}= ")
    room = _CARD - len(head) - 2
    if len(value) <= room:
        return [_card(key, value)]
    cards, rest = [], value
    while rest:
        chunk, rest = rest[:room - 1], rest[room - 1:]
        text = f"'{chunk}&'" if rest else f"'{chunk}'"
        cards.append(f"{head}{text}".ljust(_CARD).encode('ascii'))
        head, room = 'CONTINUE  ', _CARD - 12
    return cards


def _write_header(fh, cards):
    """Write header cards, the END card, and the padding to a block."""
    block = b''.join(cards) + b'END'.ljust(_CARD)
    fh.write(block.ljust(-(-len(block) // _BLOCK) * _BLOCK))


def _write_primary(fh, keywords):
    """The primary HDU: a header and no data, as every SNANA data file has."""
    cards = [_card('SIMPLE', True, 'conforms to FITS standard'),
             _card('BITPIX', 8, 'array data type'),
             _card('NAXIS', 0, 'number of array dimensions'),
             _card('EXTEND', True)]
    for key, value in keywords.items():
        cards += (_string_cards(key, value) if isinstance(value, str)
                  else [_card(key, value)])
    _write_header(fh, cards)


def primary_keywords(config, resources, kcor_filter_names, file_prefix, model,
                     snrmon_colname, version_name, zp_fluxcal, columns, flags,
                     writes_spectra, writes_sed_true, head):
    """The primary header SNANA writes to each of its data files
    (wr_snfitsio_create, sntools_dataformat_fits.c:1123-1412). The
    private-variable count goes to HEAD alone, and a BLINDTEST file ends at
    DATATYPE, as data does.
    """
    keys = {
        'CODE_IVERSION': _CODE_IVERSION,
        'SNANA_PATH': _PACKAGE_PATH,
        'SNANA_VERSION': SNANA_VERSION_STRING,
        'SURVEY': resources.survey_name,
        'SUBSURVEY_FLAG': int(subsurvey_flag(resources)),
        'ZP_FLUXCAL': zp_fluxcal,
        # USE_SIMLIB_IDEXPT detector columns (snlc_sim.c:7425)
        'DETINFO_FLAG': 0,
        'ATMOS_FLAG': int(flags['atmos']),
        'PHOTFLAG_DETECT': resources.searcheff_photflag_detect,
        'PHOTFLAG_TRIGGER': resources.searcheff_photflag_trigger,
        'FILTERS': filters_string(kcor_filter_names),
        'VERSION': version_name,
        'PHOTFILE': f'{file_prefix}_PHOT.FITS',
    }
    if writes_spectra:
        keys['SPECFILE'] = f'{file_prefix}_SPEC.FITS'
    if head:
        keys['NPRIVATE'] = 0
    if flags['blindtest']:
        keys['DATATYPE'] = 'DATA'
        return keys

    write_mask = _WRITE_MASK_SIM_SNANA
    if snrmon_colname:
        write_mask |= _WRITE_MASK_SIM_SNRMON
    if config.output.wrflag_modelpar:
        write_mask |= _WRITE_MASK_SIM_MODELPAR
    if flags['compact']:
        write_mask |= _WRITE_MASK_COMPACT
        if config.instrument.smearflag_flux == 0:
            write_mask |= _WRITE_MASK_COMPACT_NOFLUXCAL
    if flags['atmos']:
        write_mask |= _WRITE_MASK_ATMOS
    if writes_spectra:
        write_mask |= _WRITE_MASK_SPECTRA
    if writes_sed_true:
        write_mask |= _WRITE_MASK_SED_TRUE

    keys.update({
        'DATATYPE': 'SIM_SNANA',
        'LONGSTRN': 'OGIP 1.0',
        'SIMLIB_FILE': config.survey.simlib_file,
        'CALIB_FILE': config.survey.kcor_file,
        'SIMLIB_MSKOPT': config.survey.simlib_mskopt,
        'HOSTLIB_FILE': config.host.hostlib_file or 'NONE',
        'SIM_MODEL_INDEX': model.sim_model_index,
        **mw_option_keys(config),
        'SIMOPT_FLUXERR': 0,     # FUDGEOPT_FLUXERR is refused at parse
    })
    if resources.calibration is not None:
        keys['SPECTROGRAPH_INSTRUMENT'] = resources.calibration.instrument_name
    if model.fits_param_block and config.output.wrflag_modelpar and model.dump_columns:
        block = model.fits_param_block
        keys[f'{block}_MODEL'] = sim_model_name(config, model)
        keys[f'{block}_NPAR'] = len(model.dump_columns)
        for i, name in enumerate(model.dump_columns):
            keys[f'{block}_PAR{i:02d}'] = name
    hostlib_pars = [c for c in columns if c.startswith('SIM_HOSTLIB(')]
    if hostlib_pars:
        keys['SIM_HOSTLIB_NPAR'] = len(hostlib_pars)
        for i, name in enumerate(hostlib_pars):
            keys[f'SIM_HOSTLIB_PAR{i:02d}'] = name
    keys.update({
        'SIM_NSUBSAMPLE_MARK': config.output.nsubsample_mark,
        'SIM_VARNAME_SNRMON': snrmon_colname or '',
        'SIM_SL_FLAG': 0,
        'SIM_BIASCOR_MASK': 0,
        'WRITE_MASK_HEAD': write_mask,
        'WRITE_MASK_PHOT': write_mask,
    })
    if writes_spectra:
        keys['WRITE_MASK_SPEC'] = write_mask
    return keys


def _record_dtype(columns):
    """The big-endian record dtype FITS stores the columns as.

    Column order is the dict's, which is the order the header declares. Byte
    strings pass through untouched; a Unicode column would have to be encoded
    per element, which is the single most expensive thing this module can do,
    so it is refused rather than done quietly.
    """
    dtype = []
    for name, col in columns.items():
        if col.dtype.kind == 'U':
            raise SNANAError(
                f"column {name} is Unicode; FITS stores bytes, and encoding "
                f"per element here would cost more than the rest of the write",
                function_name="_record_dtype")
        base = (col.dtype.newbyteorder('>') if col.dtype.kind in 'fiu'
                else col.dtype)
        # a 2-D column is one vector field per row (TFORM '20E')
        dtype.append((name, base, col.shape[1:]) if col.ndim > 1 else (name, base))
    return np.dtype(dtype)


# rows interleaved into the row-major table at a time
_ROWS_PER_WRITE = 1 << 20


def _write_bintable(fh, columns, extname, group_sizes=None, marker=None):
    """A BINTABLE extension holding ``columns``, written straight to ``fh``.

    The TFORM repeat count and the blank TUNIT match what SNANA's own writer
    emits. With ``group_sizes``, a marker row follows each group of that many
    rows, as SNANA closes each event's epochs and each spectrum's bins: the
    columns ``marker`` names take its values, and every other column is zero.
    Classifiers such as SuperNNova split the light curves at those rows.
    """
    dtype = _record_dtype(columns)
    n_source = len(next(iter(columns.values())))
    marker_rows = (np.zeros(0, dtype=np.int64) if group_sizes is None
                   else np.cumsum(group_sizes) + np.arange(len(group_sizes)))
    n_rows = n_source + len(marker_rows)
    cards = [_card('XTENSION', 'BINTABLE', 'binary table extension'),
             _card('BITPIX', 8, '8-bit bytes'),
             _card('NAXIS', 2, '2-dimensional binary table'),
             _card('NAXIS1', dtype.itemsize, 'width of table in bytes'),
             _card('NAXIS2', n_rows, 'number of rows in table'),
             _card('PCOUNT', 0, 'size of special data area'),
             _card('GCOUNT', 1, 'one data group (required keyword)'),
             _card('TFIELDS', len(dtype.names), 'number of fields in each row')]
    for i, name in enumerate(dtype.names, start=1):
        dt = dtype[name]
        repeat = int(np.prod(dt.shape)) if dt.shape else 1
        dt = dt.base
        if dt.kind == 'S':
            form, desc = f'{dt.itemsize}A', 'ASCII Character'
        else:
            code, desc = _TFORM[dt.str[1:]]
            form = f'{repeat}{code}'
        cards.append(_card(f'TTYPE{i}', name, f'label for field {i:>3}'))
        cards.append(_card(f'TFORM{i}', form, f'data format of field: {desc}'))
        cards.append(_card(f'TUNIT{i}', '', 'physical unit of field'))
    cards.append(_card('EXTNAME', extname, 'name of this binary table extension'))
    _write_header(fh, cards)
    for start in range(0, n_rows, _ROWS_PER_WRITE):
        stop = min(start + _ROWS_PER_WRITE, n_rows)
        rec = np.zeros(stop - start, dtype=dtype)
        is_marker = np.isin(np.arange(start, stop), marker_rows)
        # the source rows of the chunk's other rows are consecutive
        first = start - int(np.searchsorted(marker_rows, start))
        last = first + int((~is_marker).sum())
        for name, col in columns.items():
            rec[name][~is_marker] = col[first:last]
        for name, value in (marker or {}).items():
            rec[name][is_marker] = value
        rec.tofile(fh)
    remainder = -(n_rows * dtype.itemsize) % _BLOCK
    if remainder:
        fh.write(b'\0' * remainder)


def write_head_fits(keywords, file_prefix, output_dir: str, columns) -> Path:
    """Write HEAD.FITS: the run's metadata, then one row per event, its
    PTROBS stepping over the marker rows ``write_phot_fits`` adds."""
    output_file = Path(output_dir) / f'{file_prefix}_HEAD.FITS'
    events = np.arange(len(columns['PTROBS_MIN']), dtype=np.int32)
    columns = {**columns, 'PTROBS_MIN': columns['PTROBS_MIN'] + events,
               'PTROBS_MAX': columns['PTROBS_MAX'] + events}
    with open(output_file, 'wb') as fh:
        _write_primary(fh, keywords)
        _write_bintable(fh, columns, 'Header')
    slog.line(logger, f"wrote {output_file}", indent=2)
    return output_file


def write_phot_fits(keywords, file_prefix, output_dir: str, columns,
                    obs_counts) -> Path:
    """Write PHOT.FITS: every epoch of every event, in one table, each
    event's ``obs_counts`` epochs closed by a marker row."""
    output_file = Path(output_dir) / f'{file_prefix}_PHOT.FITS'
    marker = {k: v for k, v in _PHOT_MARKER.items() if k in columns}
    with open(output_file, 'wb') as fh:
        _write_primary(fh, keywords)
        _write_bintable(fh, columns, 'Photometry', obs_counts, marker)
    slog.line(logger, f"wrote {output_file}", indent=2)
    return output_file


def write_spec_fits(config, keywords, kcor_filter_names, file_prefix,
                    output_dir: str, spectra_by_sn: Dict, to_written_cid) -> Path:
    """Write SPEC FITS file with SPECTRO_HEADER and SPECTRO_FLUX tables.

    Matches SNANA SPEC FITS format: SPECTRO_HEADER has one row per spectrum
    (SNID, MJD, TEXPOSE, etc.), SPECTRO_FLUX one row per wavelength bin, and
    PTRSPEC_MIN/MAX index into SPECTRO_FLUX.

    ``spectra_by_sn`` maps the internal CID to a list of SpectrumResult;
    ``to_written_cid`` maps internal CIDs to the ones written (FORMAT_MASK
    CIDRAN). Returns the path to the written file.
    """
    output_file = Path(output_dir) / f'{file_prefix}_SPEC.FITS'

    # WRSPEC_PRESCALE tests the internal CID (apply_prescale_GENSPEC, snlc_sim.c:21855)
    ps = config.output.wrspec_prescale
    all_spectra = []
    for snid in sorted(spectra_by_sn.keys()):
        if ps > 1 and int(snid) % ps != 0:
            continue
        all_spectra.extend(spectra_by_sn[snid])

    n_spec = len(all_spectra)

    snid_arr = np.array(
        [str(int(c)) for c in to_written_cid([s.snid for s in all_spectra])],
        dtype=f'S{MXCHAR_CCID}')
    mjd_arr = np.array([s.mjd for s in all_spectra], dtype=np.float64)
    texpose_arr = np.array([s.texpose for s in all_spectra], dtype=np.float64)
    nbin_arr = np.array([len(s.lam_min) for s in all_spectra], dtype=np.int16)
    snr_compute_arr = np.array([s.snr_compute for s in all_spectra], dtype=np.float32)
    contam_arr = np.array([s.scale_host_contam for s in all_spectra], dtype=np.float32)
    lammin_snr_arr = np.array([s.lammin_snr for s in all_spectra], dtype=np.float32)
    lammax_snr_arr = np.array([s.lammax_snr for s in all_spectra], dtype=np.float32)

    # SED-true reduced format: grid as header scalars, SIM_FLAM the only flux column
    sed_true = bool(config.effective_spectrograph_optmask & 128)
    instrument_arr = np.array([s.instrument for s in all_spectra], dtype='S20')
    lammin_hdr = np.array([s.lam_min[0] for s in all_spectra], dtype=np.float32)
    lammax_hdr = np.array([s.lam_max[-1] for s in all_spectra], dtype=np.float32)
    lambin_hdr = np.array([s.lam_max[0] - s.lam_min[0] for s in all_spectra],
                          dtype=np.float32)
    synmag_bands = [filter_band(n) for n in kcor_filter_names]
    synmag_cols = {
        f'SIM_SYNMAG_{b}': np.array([(s.synmag or {}).get(b, 99.0) for s in all_spectra],
                                    dtype=np.float32)
        for b in synmag_bands}

    # 1-indexed row pointers into SPECTRO_FLUX, past each spectrum's marker row
    ptrspec_min = np.zeros(n_spec, dtype=np.int32)
    ptrspec_max = np.zeros(n_spec, dtype=np.int32)
    ptr = 1
    for i, spec in enumerate(all_spectra):
        n_bins = len(spec.lam_min)
        ptrspec_min[i] = ptr
        ptrspec_max[i] = ptr + n_bins - 1
        ptr += n_bins + 1

    total_flux_rows = sum(len(s.lam_min) for s in all_spectra)
    lammin_flux = np.zeros(total_flux_rows, dtype=np.float32)
    lammax_flux = np.zeros(total_flux_rows, dtype=np.float32)
    flam_flux = np.zeros(total_flux_rows, dtype=np.float32)
    flamerr_flux = np.zeros(total_flux_rows, dtype=np.float32)
    sim_flam_flux = np.zeros(total_flux_rows, dtype=np.float32)

    row = 0
    for spec in all_spectra:
        n_bins = len(spec.lam_min)
        lammin_flux[row:row + n_bins] = spec.lam_min
        lammax_flux[row:row + n_bins] = spec.lam_max
        flam_flux[row:row + n_bins] = spec.flam
        flamerr_flux[row:row + n_bins] = spec.flamerr
        sim_flam_flux[row:row + n_bins] = spec.sim_flam
        row += n_bins

    if sed_true:
        header_cols = {
            'SNID': snid_arr, 'MJD': mjd_arr, 'Texpose': texpose_arr,
            'INSTRUMENT': instrument_arr, 'LAMMIN': lammin_hdr,
            'LAMMAX': lammax_hdr, 'LAMBIN': lambin_hdr, 'NBIN_LAM': nbin_arr,
            'PTRSPEC_MIN': ptrspec_min, 'PTRSPEC_MAX': ptrspec_max,
            **synmag_cols,
        }
        flux_cols = {'SIM_FLAM': sim_flam_flux}
    else:
        header_cols = {
            'SNID': snid_arr, 'MJD': mjd_arr, 'Texpose': texpose_arr,
            'INSTRUMENT': instrument_arr,
            'SNR_COMPUTE': snr_compute_arr,
            'LAMMIN_SNR': lammin_snr_arr, 'LAMMAX_SNR': lammax_snr_arr,
            'SCALE_HOST_CONTAM': contam_arr,
            'NBIN_LAM': nbin_arr, 'PTRSPEC_MIN': ptrspec_min,
            'PTRSPEC_MAX': ptrspec_max,
            **synmag_cols,
        }
        flux_cols = {
            'LAMMIN': lammin_flux, 'LAMMAX': lammax_flux,
            'FLAM': flam_flux, 'FLAMERR': flamerr_flux,
            'SIM_FLAM': sim_flam_flux,
        }

    with open(output_file, 'wb') as fh:
        _write_primary(fh, keywords)
        _write_bintable(fh, header_cols, 'SPECTRO_HEADER')
        _write_bintable(fh, flux_cols, 'SPECTRO_FLUX', nbin_arr,
                        dict.fromkeys(flux_cols, _EOE_MARKER))

    slog.line(logger, f"SPEC.FITS: {n_spec} spectra, {total_flux_rows} flux rows -> {output_file}",
              indent=2)
    return output_file


def gzip_output_fits(output_dir: str) -> None:
    """gzip every ``*.FITS`` in the version dir (SNANA ``gzip *.FITS``),
    replacing each ``X.FITS`` with ``X.FITS.gz``. Streamed for large files.
    The ``.LIST``/``.README`` keep referencing the logical ``.FITS`` names, as
    SNANA does; the reader resolves ``.FITS`` -> ``.FITS.gz`` transparently.
    Call after *all* FITS (HEAD/PHOT/SPEC) are written.

    isal writes the same gzip format across threads. Its levels run 0 to 3,
    a scale of its own, and 3 is its strongest; SNANA runs ``gzip *.FITS``
    with no level flag (sntools_dataformat_fits.c:3140), so zlib's default 6.
    """
    for path in Path(output_dir).glob("*.FITS"):
        gz_path = path.with_name(path.name + '.gz')
        with open(path, 'rb') as fin, \
                igzip_threaded.open(gz_path, 'wb', compresslevel=3,
                                    threads=GZIP_THREADS) as fout:
            shutil.copyfileobj(fin, fout)
        path.unlink()
