"""SNANA's file formats, where more than one program reads or writes them.

- The values SNANA stores for an unset quantity.
- Finding and opening files: ``$SNDATA_ROOT``, the ``.gz`` spelling SNANA
  falls back to as CFITSIO does, FITS tables, the values that mean "no
  file", and the DOCUMENTATION block SNANA requires at the top of its data
  and config files so a published result can be traced back to its inputs.
- Two-column tables (rd2columnFile), SURVEY.DEF, and the cosmology recorded
  in a simulation's README.
- VARNAMES text tables (FITRES, M0DIF, SIMGEN_DUMP and the rest), read and
  written, with the header lines that name the SNANA version, the program
  and the jax version that wrote them.
- Host-galaxy column names and host slots, WGTMAP weight grids, and
  create_covariance's matrix files.

A model's own formats live with the model in :mod:`jax_snlc_core.models`,
and a format only one program reads lives in that program's package.
"""

import csv
import functools
import gzip
import io
import logging
import mmap
import os
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
from astropy.io import fits
from astropy.table import Table
from jax.scipy.ndimage import map_coordinates

from jax_snlc_core import __version__, logging as log
from jax_snlc_core.logging import SNANAError, version_string

logger = logging.getLogger(__name__)

# unset measured value (init_SNDATA_EVENT)
SNANA_NULLFLOAT = -9.0

# the magnitude of zero flux, as before explosion (sntools.h:124)
MAG_ZEROFLUX = 99.0

# a magnitude nothing defines, such as a missing host's (sntools.h:126)
MAG_UNDEFINED = 666.0

# A host property no catalogue supplied (HOSTLIB_PROPERTY_UNDEFINED, sntools.h:131).
HOSTGAL_PROPERTY_NULL = -9999.0


def sndata_root() -> Path:
    """$SNDATA_ROOT, which SURVEY.DEF, the models and the maps live under.
    SNANA reads it from the environment alone."""
    try:
        return Path(os.environ["SNDATA_ROOT"])
    except KeyError:
        raise SNANAError("$SNDATA_ROOT is not set", function_name="sndata_root") from None


def resolve_fits_path(path, required=False):
    """Return the file that exists on disk for ``path``, plain or ``.gz``.

    SNANA gzips its output but references the plain name, and CFITSIO opens
    the ``.gz`` when given the base name; astropy does not, so every SNANA
    file is found through here. Prefers the plain file when both exist.
    With neither present, ``required`` aborts naming both spellings;
    otherwise the original path passes through so the caller's own open
    raises.
    """
    path = Path(path)
    if path.exists():
        return str(path)
    gz_path = path.with_name(path.name + ".gz")
    if gz_path.exists():
        return str(gz_path)
    if required:
        raise SNANAError(f"file not found: {path} (also checked {path}.gz)",
                         function_name="resolve_fits_path")
    return str(path)


def read_fits_table(path):
    """Read a SNANA primary header and decoded Astropy table in native byte order.

    Read the table once into a buffer, in bounded chunks even for gzip files.
    Astropy decodes that buffer, including logical flags, scaling and shapes;
    ordinary numeric columns remain views rather than whole-table copies.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    try:
        with opener(path, 'rb') as stream:
            primary = fits.Header.fromfile(stream)
            if primary['NAXIS'] and all(primary[f'NAXIS{i}'] > 0
                                        for i in range(1, primary['NAXIS'] + 1)):
                raise ValueError("SNANA requires an empty primary HDU")
            header = fits.Header.fromfile(stream)
            if header['XTENSION'] != 'BINTABLE':
                raise ValueError("SNANA requires a binary table after the primary HDU")
            encoded = header.tostring().encode('ascii')
            size = header['NAXIS1'] * header['NAXIS2'] + header['PCOUNT']
            buffer = mmap.mmap(-1, len(encoded) + size)
            buffer[:len(encoded)] = encoded
            view = memoryview(buffer)
            offset = len(encoded)
            while offset < len(buffer):
                count = stream.readinto(view[offset:offset + (1 << 20)])
                if not count:
                    raise ValueError("file ends before its table is complete")
                offset += count
        hdu = fits.BinTableHDU.fromstring(buffer, character_as_bytes=True, uint=True)
        table = Table(hdu.data, copy=False)
        # Unconverted columns view these records; Astropy's converted columns
        # already have native byte order. Swap row blocks to stay in cache.
        records = hdu.data.view(np.ndarray)
        columns = [column for column in table.itercols() if not column.dtype.isnative]
        block_rows = max(1, (1 << 20) // max(1, header['NAXIS1']))
        for start in range(0, len(table), block_rows):
            records[start:start + block_rows].byteswap(inplace=True)
        for column in columns:
            column.dtype = column.dtype.newbyteorder('=')
        return primary, table
    except (OSError, ValueError, TypeError) as exc:
        raise SNANAError(f"Cannot read {path}: {exc}",
                         function_name="read_fits_table") from exc


def open_text(path):
    """Open a SNANA text file, accepting its plain name or ``.gz`` spelling."""
    path = Path(resolve_fits_path(path))
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path, "rt")


def is_ignorefile(filename: Optional[str]) -> bool:
    """True when a configured file value means "no file".

    Mirrors SNANA's IGNOREFILE (sntools.c:4832-4853): None, the empty
    string, whitespace only, or NULL/NONE/BLANK in any case.
    """
    if filename is None:
        return True
    stripped = str(filename).strip()
    return stripped == '' or stripped.upper() in ('NONE', 'NULL', 'BLANK')


# sntools.h:45-46
KEYNAME_DOCANA = 'DOCUMENTATION:'
KEYNAME_DOCANA_END = 'DOCUMENTATION_END:'


def skip_docana(lines: Iterable[str]) -> Iterator[str]:
    """``lines`` with the DOCUMENTATION block removed.

    A line whose first whitespace-separated token is exactly
    ``DOCUMENTATION:`` opens the block and ``DOCUMENTATION_END:`` closes it;
    both keys and every line between them are dropped, as rd2columnFile does
    (sntools.c:7297-7303).
    """
    in_block = False
    for line in lines:
        tokens = line.split()
        if tokens and tokens[0] == KEYNAME_DOCANA:
            in_block = True
        elif tokens and tokens[0] == KEYNAME_DOCANA_END:
            in_block = False
        elif not in_block:
            yield line


def check_file_docana(path, require: bool = True, header_lines=None) -> None:
    """Abort (or warn) when ``path`` does not begin with ``DOCUMENTATION:``.

    ``header_lines`` selects the metadata check: the key may appear among
    the words of the first few lines. Otherwise it must be the first word.

    SNANA opens these files through snana_openTextFile, whose
    check_openFile_docana reads one fscanf("%s") token: the first
    whitespace-delimited word of the file, after any leading blank lines.
    """
    with open_text(path) as fh:
        if header_lines is None:
            first = next((words for words in map(str.split, fh) if words), [])
            found = bool(first) and first[0] == KEYNAME_DOCANA
        else:
            found = any(KEYNAME_DOCANA in fh.readline().split('#')[0].split()
                        for _ in range(header_lines))

    if found:
        return

    location = ('as its first word' if header_lines is None
                else f'in its first {header_lines} lines')
    detail = (f"{path} must contain '{KEYNAME_DOCANA}' {location}. See DOCANA examples "
              f"with: grep -R DOCUMENTATION_END $SNDATA_ROOT")
    if require:
        raise SNANAError(f"Missing DOCUMENTATION block. {detail}",
                         function_name="check_file_docana")
    log.warn(logger, "DOCANA", detail)


# sntools.c:7341 commentchar()
COMMENT_CHARS = '#!%@'


def read_two_column_file(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read the first two columns of an ASCII table, plain or gzipped, as
    SNANA's rd2columnFile does (sntools.c:7242-7330).

    A line of two characters or fewer, a line whose first word starts with
    ``#``, ``!``, ``%`` or ``@``, and the DOCUMENTATION block are skipped;
    columns beyond the second are ignored. A data line without two numbers,
    which rd2columnFile has no defined reading of, aborts.
    """
    col1, col2 = [], []
    with open_text(path) as fh:
        for line in skip_docana(fh):
            tokens = line.split()
            if len(line) <= 2 or not tokens or tokens[0][0] in COMMENT_CHARS:
                continue
            try:
                col1.append(float(tokens[0]))
                col2.append(float(tokens[1]))
            except (IndexError, ValueError):
                raise SNANAError(f"{path}: '{line.rstrip()}' is not two numbers",
                                 function_name="read_two_column_file") from None
    return np.array(col1), np.array(col2)


def _survey_def_path(path: Optional[str]) -> str:
    """``path``, or $SNDATA_ROOT's SURVEY.DEF."""
    if path is None:
        return str(sndata_root() / 'SURVEY.DEF')
    return path


def _survey_def_rows(path: Optional[str], prefix: str) -> Iterator[Sequence[str]]:
    """Tokenised SURVEY.DEF rows carrying ``prefix``, comments stripped.

    ``SURVEY:`` rows name a survey, its integer ID (GENLC.IDSURVEY) and
    optionally a ``geo:`` token; ``FIELD:`` rows name a field and its survey.
    The readers built on this return mappings and leave what an unknown
    survey means to their callers.
    """
    path = _survey_def_path(path)
    if not Path(path).is_file():
        raise SNANAError(f"Unable to open file with mode = rt. Check '{path}'",
                         function_name="_survey_def_rows")
    with open_text(path) as fh:
        for line in fh:
            toks = line.split('#', 1)[0].split()
            if toks and toks[0] == prefix:
                yield toks


def load_survey_ids(
        path: Optional[str] = None) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Parse SURVEY.DEF into (name->id, id->name); $SNDATA_ROOT's by default.

    BBC asks for this several times per fit, so the parse is cached on the
    resolved path: a caller that changes $SNDATA_ROOT gets the new file.
    Callers read the mappings and do not modify them.
    """
    return _load_survey_ids(_survey_def_path(path))


@functools.lru_cache(maxsize=None)
def _load_survey_ids(path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    name_to_id: Dict[str, int] = {}
    id_to_name: Dict[int, str] = {}
    for toks in _survey_def_rows(path, 'SURVEY:'):
        if len(toks) < 3:
            continue
        name, sid = toks[1], int(toks[2])
        # A repeated name takes the lowest ID (get_IDSURVEY, sntools.c:3712).
        if sid < name_to_id.get(name, sid + 1):
            name_to_id[name] = sid
        id_to_name[sid] = name
    return name_to_id, id_to_name


def load_survey_geo(
        path: Optional[str] = None) -> Dict[str, Tuple[float, float, float]]:
    """{survey_name: (latitude, longitude, altitude)} for the surveys whose
    ``SURVEY:`` line carries a ``geo:`` token; the rest are absent."""
    out: Dict[str, Tuple[float, float, float]] = {}
    for toks in _survey_def_rows(path, 'SURVEY:'):
        if len(toks) < 3:
            continue
        geo = next((t for t in toks[3:] if t.startswith('geo:')), None)
        if geo is None:
            continue
        coords = geo[len('geo:'):].split(',')
        out[toks[1]] = (float(coords[0]), float(coords[1]),
                        float(coords[2]) if len(coords) >= 3 else 0.0)
    return out


def load_survey_fields(name: str, path: Optional[str] = None) -> List[str]:
    """``name``'s FIELD names in file order (SNANA builds each event's
    FIELDLIST in this order, snana.F90:19117)."""
    return [toks[1] for toks in _survey_def_rows(path, 'FIELD:')
            if len(toks) >= 4 and toks[3] == name]


# KEYLIST_COSPAR_SIM (create_covariance.py:287)
SIM_COSMOLOGY_KEYS = ("OMEGA_MATTER", "OMEGA_LAMBDA", "w0_LAMBDA", "wa_LAMBDA",
                      "MUSHIFT")

# a batch job's merged README labels each part's keys; a single run's does not
_INPUT_KEY_BLOCKS = ("INPUT_KEYS_SNIaMODEL00", "INPUT_KEYS")


def parse_mushift(text: str) -> tuple[float, float]:
    """MUSHIFT: ``<value>`` or ``<lo>:<hi>``; a bare value gives lo == hi."""
    lo, _, hi = str(text).strip().partition(':')
    return (float(lo), float(hi or lo))


def read_simulation_cosmology(readme: Path) -> dict[str, str]:
    """The README's cosmology keys as written, or {} for a version that is not
    a simulation.

    The light-curve fit copies them into its FITRES header as ``SIM_<key>``
    lines, and BBC copies the bias-correction simulation's into its outputs as
    ``SIM_<key>(BIASCOR)``, so create_covariance reads the cosmology with the
    distances instead of looking the simulation up by name (get_cospar_sim,
    create_covariance.py:2514).

    MUSHIFT appears only when the simulation set it (sntools_sim_readme.c:848),
    so its absence means no shift.
    """
    with open_text(str(readme)) as stream:
        # BaseLoader keeps every value as its text; the default loader reads
        # some lo:hi ranges as sexagesimal numbers
        docana = yaml.load(stream, Loader=yaml.BaseLoader)["DOCUMENTATION"]
    keys = next((docana[name] for name in _INPUT_KEY_BLOCKS if name in docana), {})
    if "OMEGA_MATTER" not in keys:
        return {}
    return {key: keys.get(key, "0") if key == "MUSHIFT" else keys[key]
            for key in SIM_COSMOLOGY_KEYS}


def fitopt_muopt_name(cell: tuple[int, int]) -> str:
    """The SNANA file or directory stem for a FITOPT/MUOPT grid cell."""
    fitopt, muopt = cell
    return f"FITOPT{fitopt:03d}_MUOPT{muopt:03d}"


def fitopt_muopt_indices(name: str) -> tuple[int, int]:
    """Read the indices from a FITOPT003_MUOPT002 stem."""
    fit, _, mu = name.partition('_')
    return int(fit.replace('FITOPT', '')), int(mu.replace('MUOPT', ''))


_ROWKEY = "_ROWKEY"

# the row keys of a table's data rows (validRowKey_TEXT, sntools_output_text.c:1386)
TABLE_ROW_KEYS = ("SN:", "ROW:", "GAL:", "OBS:")


@dataclass(frozen=True)
class FitresTable:
    """A parsed FITRES/M0DIF table.

    ``df`` holds the columns named by ``VARNAMES:`` (dtypes
    inferred per column, with CID/SNID kept as strings); ``meta`` holds the
    ``# KEY: value`` header block
    above ``VARNAMES:``, each value as the string the file wrote, so a number
    is converted by the caller. Lines below ``VARNAMES:`` written as
    ``KEY = value`` are not header lines and are not in ``meta``.
    """

    #: Table data with the columns named by ``VARNAMES:``.
    df: pd.DataFrame
    #: The column names, in the order ``VARNAMES:`` gave them.
    varnames: List[str]
    #: The ``# KEY: value`` header block, values as strings.
    meta: Dict[str, str]
    #: Each source file's own ``VARNAMES:``, in file order, for a table read
    #: from several files whose columns were unioned; a single-file read has
    #: one entry.
    varnames_by_file: List[List[str]] = dc_field(default_factory=list)
    #: The row key (one of :data:`TABLE_ROW_KEYS`) opening each data row, kept
    #: only by a single-file ``as_text`` read so the rows can be written out
    #: again as they came; None otherwise.
    rowkey: Optional[np.ndarray] = None

    @property
    def size(self) -> int:
        """The number of rows."""
        return len(self.df)

    def has(self, col: str) -> bool:
        """Whether the table carries a column of this name."""
        return col in self.df.columns

    def get(self, col: str, default=None) -> np.ndarray:
        """The column as a numpy array, or ``default`` where the table has no
        such column, so a reader can take an optional column in one step."""
        if col in self.df.columns:
            return self.df[col].to_numpy()
        return default


def _parse_fitres_header_line(s: str, meta: Dict[str, str]) -> None:
    body = s[1:].strip()
    if ":" in body:
        key, val = body.split(":", 1)
        meta[key.strip()] = val.strip()


def read_fitres_header(path: str) -> Dict[str, str]:
    """The ``# KEY: value`` comment header of a FITRES/M0DIF file, read up to
    the ``VARNAMES:`` line without parsing the table."""
    meta: Dict[str, str] = {}
    with open_text(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("VARNAMES:"):
                break
            if s.startswith("#"):
                _parse_fitres_header_line(s, meta)
    return meta


def read_fitres(path: str, as_text: bool = False, *, row_keys=TABLE_ROW_KEYS) -> FitresTable:
    """Parse one FITRES/M0DIF file (``path``, optionally gzipped) into a
    :class:`FitresTable`.

    Column dtypes are kept: string columns such as ``FIELD``, ``BANDLIST`` and
    a non-numeric ``CID`` stay strings. ``as_text`` keeps every value as its
    original token and records the row key for concatenation (``cat_only``).

    As SNANA's SNTABLE_READ_EXEC_TEXT reads a table (sntools_output_text.c:
    1255-1282), only lines starting with one of :data:`TABLE_ROW_KEYS` are
    rows. ``row_keys`` restricts the accepted prefixes for a specific format.
    Values past the last VARNAMES column are ignored, and a row with
    fewer values than VARNAMES aborts.
    """
    meta: Dict[str, str] = {}
    varnames: Optional[List[str]] = None

    with open_text(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("VARNAMES:"):
                varnames = s.split()[1:]
                break
            if s.startswith("#"):
                _parse_fitres_header_line(s, meta)

        if varnames is None:
            raise SNANAError(f"No VARNAMES line found in {path}",
                             function_name="read_fitres")
        if not varnames:
            raise SNANAError(
                f"No column names found in VARNAMES for {path}",
                function_name="read_fitres",
            )

        rows = io.StringIO()
        for line_number, line in enumerate(f, start=1):
            words = line.split('#', 1)[0].split()
            if not words or words[0] not in row_keys:
                continue
            if len(words) < len(varnames) + 1:
                raise SNANAError(
                    f"{path}: row at line {line_number} after VARNAMES holds fewer "
                    "values than VARNAMES names", function_name="read_fitres")
            rows.write(line)

    rows.seek(0)
    df = pd.read_csv(
        rows, sep=r"\s+", comment="#", header=None, names=[_ROWKEY] + varnames,
        usecols=range(len(varnames) + 1),
        quoting=csv.QUOTE_NONE, dtype=str if as_text else None,
        # IDs remain literal tokens, including leading zeros and "NA".
        converters=None if as_text else {
            name: str for name in ('CID', 'SNID') if name in varnames},
        keep_default_na=not as_text,
    )
    rowkeys = df.pop(_ROWKEY)
    rowkey = rowkeys.to_numpy() if as_text else None
    return FitresTable(df=df, varnames=varnames, meta=meta,
                       varnames_by_file=[varnames], rowkey=rowkey)


def read_fitres_multi(paths: Sequence[str], as_text: bool = False) -> FitresTable:
    """Read several FITRES files and concatenate their rows (SNANA ``datafile=a,b``).

    Columns are unioned across files; a column absent from one file is filled
    with NaN (or None under ``as_text``). ``meta`` is taken from the first file.
    """
    return concat_tables([read_fitres(p, as_text=as_text) for p in paths])


def concat_tables(tables: Sequence[FitresTable]) -> FitresTable:
    """One table from several, rows in order (SNANA ``datafile=a,b``).

    Columns are unioned; a column absent from one table is filled with NaN.
    ``meta`` is taken from the first table.
    """
    df = pd.concat([t.df for t in tables], ignore_index=True)
    return FitresTable(df=df, varnames=list(df.columns), meta=tables[0].meta,
                       varnames_by_file=[v for t in tables for v in t.varnames_by_file])


@dataclass
class SnanaDumpData:
    """Container for parsed SNANA SIMGEN_DUMP columns."""

    columns: Dict[str, np.ndarray]

    @property
    def size(self) -> int:
        """Number of entries in the dump."""
        if not self.columns:
            return 0
        first_key = next(iter(self.columns))
        return len(self.columns[first_key])

    def get(self, key: str, default=None):
        return self.columns.get(key, default)


def _float_or_nan(token) -> float:
    """A dump value as a float, NaN where it is not a number."""
    try:
        return float(token)
    except ValueError:
        return np.nan


def load_snana_dump(dump_path: str, dedup_cid: bool = False) -> SnanaDumpData:
    """Load a SNANA SIMGEN_DUMP file (``dump_path``) into a
    :class:`SnanaDumpData` of float arrays, one per column, NaN where a value
    is not a number. The rows are the table's ``SN:`` rows.

    SNANA reuses CIDs when a SN is prefiltered (NEPOCH < CUTWIN_NEPOCH) and
    regenerated, and SIMGEN_DUMPALL writes both entries; when ``dedup_cid``
    is True, only the entry with the highest CUTMASK per CID is kept.
    """
    table = read_fitres(dump_path, as_text=True)
    rows = table.rowkey == 'SN:'
    columns = {name: np.array([_float_or_nan(token) for token in table.df[name][rows]])
               for name in table.varnames}

    if dedup_cid and 'CID' in columns and 'CUTMASK' in columns:
        cids = columns['CID'].astype(int)
        cutmasks = columns['CUTMASK'].astype(int)
        best_row, tied = {}, set()
        for i, (cid, cm) in enumerate(zip(cids, cutmasks)):
            if cid not in best_row or cm > cutmasks[best_row[cid]]:
                best_row[cid] = i
                tied.discard(cid)
            elif cm == cutmasks[best_row[cid]]:
                tied.add(cid)
        # rows tied at CUTMASK 0 are all rejects, so any one stands for them
        ambiguous = sorted(c for c in tied if cutmasks[best_row[c]] > 0)
        if ambiguous:
            cid = ambiguous[0]
            raise SNANAError(
                f"CID {cid} appears twice with CUTMASK {cutmasks[best_row[cid]]}, "
                "so neither row is the event: the sim reused the CID. Give the "
                "events distinct CIDs (CIDRAN_MIN/CIDRAN_MAX) and re-run the "
                "dump.", function_name='load_snana_dump')
        keep_indices = sorted(best_row.values())
        if len(keep_indices) < len(cids):
            n_removed = len(cids) - len(keep_indices)
            log.line(logger, f"De-duplicated {n_removed} CID-reuse entries "
                     f"({len(cids)} -> {len(keep_indices)} rows)")
            columns = {name: arr[keep_indices] for name, arr in columns.items()}

    return SnanaDumpData(columns=columns)


DISTRIBUTION = "jax-snana"

# must parse as >= 11.02 for the VPEC sign (correct_sign_vpec_data)
SNANA_VERSION_STRING = "v12_02d-12-g74dfb2c5-JAX"


def snana_version_comment():
    """The SNANA_VERSION header line (sntools_output_text.c): the one value
    every jax output carries, which readers parse as a release."""
    return f"SNANA_VERSION: {SNANA_VERSION_STRING}"


def program_comment(program):
    """The PROGRAM header line SNANA's tables carry after SNANA_VERSION
    (sntools_output_text.c:721, SALT2mu.c:23555), naming the script that
    wrote the file."""
    return f"PROGRAM:       {program}"


def jax_version_comment():
    """The line that says which jax wrote the file, beside SNANA_VERSION."""
    return f"JAX_VERSION: {DISTRIBUTION} {__version__} {version_string()}"


def snana_value_strings(name, values):
    """Format a float column as SNANA's text writer does.

    The choice is per value, not per column, so one column can hold both
    ``-9`` and ``2.29797e-05`` (sntools_output_text.c:467-495).
    """
    v = np.asarray(values, dtype=float)
    if "MJD" in name:
        return np.char.mod("%.4f", v)
    if name[:1] == "z":
        return np.char.mod("%.5f", v)

    finite = np.isfinite(v)
    truncated = np.where(finite, v, 0.0).astype(np.int64)
    fmt = np.full(v.shape, "%.5e", dtype=object)
    fmt[v > 1.0e10] = "%.4e"
    fmt[(v > 0.1) & (v <= 1.0e10)] = "%.5f"
    # an exactly integral value prints as an integer, whatever its magnitude
    integral = finite & (v - truncated == 0.0)
    fmt[integral] = "%d"
    # coordinates outrank the tests above, but only in their own range
    if "RA" in name or "DEC" in name:
        fmt[np.abs(v) < 400] = "%.6f"
    out = np.empty(v.shape, dtype=object)
    for spec in np.unique(fmt):
        rows = fmt == spec
        src = truncated if spec == "%d" else v
        out[rows] = [spec % x for x in src[rows]]
    return out.astype(str)


def write_varnames_table(output_path, varnames, columns, fmt, style='key',
                         rowkey='SN:', header_comments=(),
                         blank_separator=True, post_varnames_comments=()):
    """Write an SNANA text table (sntools_output_text.c:600-761 formats):
    FITRES (``SN:``), M0DIF (``ROW:``), SNANA, OUTLIER, LCPLOT and the BBC
    diagnostic dumps.

    ``columns`` maps varname -> array; ``fmt`` maps varname -> printf format.
    A column it leaves out, or maps to None, is written as SNANA's own text
    writer would: strings as they are, numbers by the per-value rule.
    Styles: key (# comments + VARNAMES: + rowkey rows), csv, col, none.
    """
    if style == 'none':
        return
    str_cols = []
    for name in varnames:
        data = np.asarray(columns[name])
        spec = fmt.get(name)
        if spec == '%d':
            str_cols.append(data.astype(np.int64).astype(str))
        elif spec == '%s' or (spec is None and data.dtype.kind in 'OUS'):
            str_cols.append(data.astype(str))
        elif spec is None:
            str_cols.append(snana_value_strings(name, data))
        else:
            str_cols.append(np.char.mod(spec, data))

    sep = ',' if style == 'csv' else ' '
    # explicit dtype: an empty table must stay a str array
    lines = np.array([sep.join(row) for row in zip(*str_cols)],
                     dtype=np.str_) if str_cols else np.empty(0, dtype=np.str_)

    with open(output_path, 'w') as f:
        if style == 'key':
            for line in header_comments:
                # None is a truly blank separator line; "" is a bare "# "
                f.write("\n" if line is None else f"# {line}\n")
            if header_comments and blank_separator:
                f.write("#\n")
            f.write(f"VARNAMES: {' '.join(varnames)}\n")
            # after VARNAMES, as write_fitres_driver
            for line in post_varnames_comments:
                f.write("\n" if line is None else f"# {line}\n")
            lines = np.char.add(f'{rowkey} ', lines)
        elif style == 'csv':
            f.write(','.join(varnames) + '\n')
        f.write('\n'.join(lines))
        f.write('\n')
    log.line(logger, f"Wrote {len(lines)} rows to {output_path}")


# host slots HOSTGAL, HOSTGAL2, HOSTGAL3
MXHOSTGAL = 3
HOSTGAL_COORD_NULL = -999.0

HOSTGAL_SHAPE_NULL = 999.0

HOSTGAL_MAG_NULL = 999.0

HOSTGAL_SB_NULL = -999.0

HOSTGAL_CONFUSION_NULL = -99.0

HOSTGAL_NO_HOST_NULL = {
    'HOSTGAL_MAG': HOSTGAL_MAG_NULL, 'HOSTGAL_MAGERR': HOSTGAL_MAG_NULL,
    'HOSTGAL_RA': HOSTGAL_COORD_NULL, 'HOSTGAL_DEC': HOSTGAL_COORD_NULL,
    'HOSTGAL_SNSEP': SNANA_NULLFLOAT, 'HOSTGAL_DDLR': SNANA_NULLFLOAT,
    'HOSTGAL_SPECZ': SNANA_NULLFLOAT, 'HOSTGAL_SPECZ_ERR': SNANA_NULLFLOAT,
    'HOSTGAL_PHOTOZ': SNANA_NULLFLOAT, 'HOSTGAL_PHOTOZ_ERR': SNANA_NULLFLOAT,
    'HOSTGAL_CONFUSION': HOSTGAL_CONFUSION_NULL,
    'HOSTGAL_OBJID': 0, 'HOSTGAL_OBJID2': 0, 'HOSTGAL_OBJID_UNIQUE': 0,
    'HOSTGAL_FLAG': 0,
    'HOSTGAL_LOGSFR': HOSTGAL_PROPERTY_NULL,
    'HOSTGAL_LOGSFR_ERR': HOSTGAL_PROPERTY_NULL,
    'HOSTGAL_COLOR': HOSTGAL_PROPERTY_NULL,
    'HOSTGAL_COLOR_ERR': HOSTGAL_PROPERTY_NULL,
    'HOSTGAL_ELLIPTICITY': HOSTGAL_PROPERTY_NULL,
    'HOSTGAL_SQRADIUS': HOSTGAL_PROPERTY_NULL,
}

HOSTGAL_PREFIXES = ('HOSTGAL',) + tuple(f'HOSTGAL{k}' for k in range(2, MXHOSTGAL + 1))
# width of each slot's photo-z quantile vector in a data file (sndata.h:185)
MXBIN_HOSTGALz_QUANTILE = 20
# width of each slot's LOGMASS(z) grid in a data file (sndata.h:184)
MXBIN_HOSTGALz = 40

#: sndata.h:84-87
HOSTGAL_PROPERTY_BASENAMES = ('LOGMASS', 'LOGSFR', 'LOGsSFR', 'COLOR')

#: sntools_host.h:116-121, each to the spelling kept in memory
HOST_COORDINATE_NAMES = {'RA': 'RA_GAL', 'RA_HOST': 'RA_GAL', 'RA_GAL': 'RA_GAL',
                         'DEC': 'DEC_GAL', 'DEC_HOST': 'DEC_GAL',
                         'DEC_GAL': 'DEC_GAL'}


def hostgal_null_key(col, bands):
    """Column name with any trailing band stripped, for sentinel lookup."""
    for b in bands:
        if col.endswith(f'_{b}'):
            return col[: -(len(b) + 1)]
    return col


def canonical_property_name(name: str) -> str:
    """The in-memory spelling of a host column: ``LOGMASS_TRUE`` and
    ``LOGMASS`` are both ``LOGMASS``, ``RA`` and ``RA_HOST`` are ``RA_GAL``;
    any other name is itself.

    SNANA rewrites each property basename to ``<BASE>_TRUE`` wherever it reads
    a name: the HOSTLIB header, a WGTMAP's variables and ``HOSTLIB_STOREPAR``
    (checkAlternateVarNames_HOSTLIB, sntools_host.c:3029, applied at :2481,
    :1333, :1098). It reads the host coordinates under ``RA``, ``RA_HOST`` or
    ``RA_GAL``, and likewise for DEC (sntools_host.c:2740-2750). Every name
    read from an input goes through here first."""
    if name.endswith('_TRUE') and name[:-5] in HOSTGAL_PROPERTY_BASENAMES:
        return name[:-5]
    return HOST_COORDINATE_NAMES.get(name, name)


@dataclass
class WgtMapData:
    """Parsed WGTMAP data: the N-dimensional weight grid, its axes, and
    which dimensions are SN variables (e.g. x1, c) rather than catalogue
    columns."""
    ndim: int
    varnames: List[str]
    nbins: np.ndarray  # Shape: (ndim,)
    ranges: np.ndarray  # Shape: (ndim, 2)
    wgt: jnp.ndarray
    snmagshift: Optional[jnp.ndarray]
    opt_extrap: int  # 0 = abort on out-of-bounds, 1 = clamp to edge
    snvar_names: List[str]
    snvar_indices: List[int]


def _read_wgtmap_header(source):
    """Consume the header and return dimension and function names.

    Columns before WGT are dimensions; WGT and subsequent columns are
    functions (read_WGTMAP, sntools_wgtmap.c:78-92).
    """
    for line in source:
        words = line.split('#', 1)[0].split()
        if words and words[0] in ('VARNAMES:', 'VARNAMES_WGTMAP:'):
            names = words[1:]
            if 'WGT' not in names or names[0] == 'WGT':
                raise SNANAError(f"Invalid VARNAMES_WGTMAP in {source.name}",
                                 function_name='_read_wgtmap_header')
            iwgt = names.index('WGT')
            return [canonical_property_name(n) for n in names[:iwgt]], names[iwgt:]
    raise SNANAError(f"VARNAMES_WGTMAP line not found in {source.name}",
                     function_name='_read_wgtmap_header')


def _parse_wgtmap(filepath: str) -> Tuple[List[str], List[str], np.ndarray]:
    """Read dimension names, function names and grid rows in one pass."""
    rows = []
    with open_text(filepath) as source:
        varnames, fun_names = _read_wgtmap_header(source)
        for line in source:
            words = line.split('#', 1)[0].split()
            if words and words[0] == 'WGT:':
                rows.append([float(x) for x in words[1:]])

    data = np.array(rows, dtype=np.float64)
    ncol = len(varnames) + len(fun_names)
    if data.ndim != 2 or data.shape[1] != ncol:
        raise SNANAError(
            f"WGTMAP rows must have {ncol} values ({len(varnames)} dimensions + "
            f"{len(fun_names)} functions); got "
            f"{data.shape[1] if data.ndim == 2 else 'ragged rows'}",
            function_name="_parse_wgtmap")
    return varnames, fun_names, data


def read_wgtmap_varnames(filepath: str) -> List[str]:
    """Read only the WGTMAP dimension names, without loading its grid."""
    with open_text(resolve_fits_path(filepath, required=True)) as source:
        varnames, _ = _read_wgtmap_header(source)
    return varnames


def read_wgtmap(filepath: str, sn_variables, opt_extrap: int = 0) -> WgtMapData:
    """Read and parse a WGTMAP file (``filepath``, .gz or plain text) into a
    :class:`WgtMapData` (read_WGTMAP, sntools_wgtmap.c): the weight grid of
    HOSTLIB galaxy selection and SIMSED surface selection.

    ``sn_variables`` maps each lower-cased spelling of a per-SN quantity,
    rather than a catalogue column, to the name the caller stores it under;
    a dimension it spells takes that name. The caller derives it from its
    model (checkSNvar_HOSTLIB_WGTMAP, sntools_host.c:2868).
    ``opt_extrap`` is 0 to abort if any galaxy is out of bounds (SNANA
    default) or 1 to clamp out-of-bounds values to the grid edge.
    """
    filepath = resolve_fits_path(filepath, required=True)
    logger.info(f"Reading WGTMAP file: {filepath}")

    varnames, fun_names, data = _parse_wgtmap(filepath)
    ndim = len(varnames)
    log.line(logger, f"Dimensions: {ndim} ({', '.join(varnames)})")
    log.line(logger, f"Loaded {len(data)} grid points")

    # uniform bins, tolerance 1e-6, as read_GRIDMAP
    axes = [np.unique(data[:, i]) for i in range(ndim)]
    for name, ax in zip(varnames, axes):
        d = np.diff(ax)
        if len(d) and np.any(np.abs(d - d[0]) / d > 1.0e-6):
            raise SNANAError(
                f"non-uniform bin in WGTMAP dimension '{name}'",
                function_name="read_wgtmap")

    nbins = np.array([len(ax) for ax in axes], dtype=np.int32)
    ranges = np.array([[ax[0], ax[-1]] for ax in axes])

    # every cell exactly once, in any row order
    idx = tuple(np.searchsorted(axes[i], data[:, i]) for i in range(ndim))
    counts = np.zeros(tuple(nbins), dtype=np.int64)
    np.add.at(counts, idx, 1)
    if np.any(counts != 1):
        cell = np.argwhere(counts != 1)[0]
        coords = ', '.join(f"{v}={axes[i][cell[i]]:g}"
                           for i, v in enumerate(varnames))
        raise SNANAError(
            f"WGTMAP cell at {coords} appears "
            f"{int(counts[tuple(cell)])} times; every grid cell must be "
            f"tabulated exactly once.", function_name="read_wgtmap")

    wgt_grid = np.zeros(tuple(nbins))
    wgt_grid[idx] = data[:, ndim]

    # an all-zero SNMAGSHIFT means unused
    snmagshift_grid = None
    if 'SNMAGSHIFT' in fun_names:
        vals = data[:, ndim + fun_names.index('SNMAGSHIFT')]
        if not np.allclose(vals, 0.0):
            snmagshift_grid = np.zeros(tuple(nbins))
            snmagshift_grid[idx] = vals
            snmagshift_grid = jnp.array(snmagshift_grid, dtype=jnp.float32)

    # float64 against underflow
    wgt_grid = jnp.array(wgt_grid, dtype=jnp.float64)
    log.line(logger, f"Grid shape: {' x '.join(map(str, nbins))}, weight range: "
             f"{float(jnp.min(wgt_grid)):.3e} to {float(jnp.max(wgt_grid)):.3e}")

    varnames = [sn_variables.get(var.lower(), var) for var in varnames]
    snvar_indices = [i for i, var in enumerate(varnames)
                     if var.lower() in sn_variables]
    snvar_names = [varnames[i] for i in snvar_indices]
    if snvar_names:
        log.line(logger, f"Detected SN variables: {snvar_names}")

    return WgtMapData(
        ndim=ndim,
        varnames=varnames,
        nbins=nbins,
        ranges=ranges,
        wgt=wgt_grid,
        snmagshift=snmagshift_grid,
        opt_extrap=opt_extrap,
        snvar_names=snvar_names,
        snvar_indices=snvar_indices,
    )


@jax.jit
def interpolate_wgtmap(
    wgt_grid: jnp.ndarray,
    grid_ranges: jnp.ndarray,  # (ndim, 2)
    grid_nbins: jnp.ndarray,   # (ndim,)
    galaxy_properties: jnp.ndarray  # (n_galaxies, ndim)
) -> jnp.ndarray:
    """N-dimensional weight interpolation via map_coordinates, vectorised
    over all galaxies at once, returning the interpolated weight per galaxy,
    shape (n_galaxies,).

    ``wgt_grid`` is the N-dimensional weight map; ``grid_ranges`` the
    min/max values for each dimension, shape (ndim, 2); ``grid_nbins`` the
    number of bins for each dimension, shape (ndim,); ``galaxy_properties``
    the physical properties for each galaxy, shape (n_gal, ndim), e.g.
    columns [LOGMASS, LOG_SFR] or [LOGMASS, LOG_SFR, x1].
    """
    # a single-bin axis maps to index 0
    min_ranges = grid_ranges[:, 0]
    span = grid_ranges[:, 1] - min_ranges
    scale = jnp.where(grid_nbins > 1,
                      (grid_nbins - 1) / jnp.where(span > 0.0, span, 1.0),
                      0.0)
    coords = (galaxy_properties - min_ranges) * scale
    return map_coordinates(wgt_grid, coordinates=coords.T, order=1,
                           mode='nearest')


def expand_upper_triangle(values, n: int) -> np.ndarray:
    """Rebuild the full symmetric matrix from its packed upper triangle."""
    values = np.asarray(values, dtype=np.float64)
    expected = n * (n + 1) // 2
    if values.size != expected:
        raise SNANAError(
            f"covariance has {values.size} values, not the {expected} of an "
            f"upper triangle for nsn={n}",
            function_name="expand_upper_triangle",
        )
    full = np.zeros((n, n))
    full[np.triu_indices(n)] = values
    return full + np.triu(full, k=1).T


def read_covariance(path) -> np.ndarray:
    """Read a covariance matrix from NPZ or SNANA text, optionally gzipped."""
    if Path(path).suffix.lower() == ".npz":
        return read_covariance_npz(path)
    return read_covariance_text(path)


def read_covariance_npz(path) -> np.ndarray:
    """Read a dense ``cov`` matrix or a packed upper triangle with ``nsn``.

    Packed values need the explicit row count: their length can also be a
    perfect square (for nsn = 1, 8, 49, 288), so inferring a shape is ambiguous.
    """
    with np.load(path) as payload:
        if "cov" not in payload:
            raise SNANAError(f"{path} has no 'cov' array",
                             function_name="read_covariance_npz")
        matrix = np.asarray(payload["cov"], dtype=np.float64)
        if matrix.ndim == 2:
            return matrix
        if "nsn" not in payload:
            raise SNANAError(
                f"{path} holds a 1-D covariance but no 'nsn'; its order cannot be recovered",
                function_name="read_covariance_npz")
        n = int(np.asarray(payload["nsn"]).ravel()[0])
    return expand_upper_triangle(matrix, n)


def write_covariance_npz(path, cov) -> None:
    """Write a matrix in create_covariance's NPZ form."""
    cov = np.asarray(cov)
    np.savez(path, nsn=[len(cov)],
             cov=cov[np.triu_indices_from(cov)].astype(np.float32))


def read_covariance_text(path) -> np.ndarray:
    """Read SNANA's text covariance: the dimension, then the flattened square
    matrix, one value per line, optionally gzipped.

    Only the first token of a line is read, as ``read_mucov_text`` does
    (wfit.c:2011), so the per-element labels that ``--label_cov_rows`` appends
    are ignored.
    """
    values = []
    with open_text(path) as stream:
        for line in stream:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                values.append(float(stripped.split()[0]))
    if not values:
        raise SNANAError(
            f"empty covariance file: {path}",
            function_name="read_covariance_text",
        )
    size = int(values[0])
    flat = np.asarray(values[1:], dtype=np.float64)
    if flat.size != size * size:
        raise SNANAError(
            f"{path} contains {flat.size} values; expected {size * size}",
            function_name="read_covariance_text",
        )
    return flat.reshape(size, size)


def validate_factorized_covariance(covariance):
    """Validate and normalise the factor arrays at input/output boundaries."""
    if not {"diag", "U"} <= covariance.keys():
        raise SNANAError("factorized covariance requires diag and U")
    diag = np.asarray(covariance["diag"], dtype=np.float64)
    vectors = np.asarray(covariance["U"], dtype=np.float64)
    labels = tuple(covariance.get("vec_labels", ()))
    if diag.ndim != 1 or vectors.ndim != 2 or vectors.shape[0] != diag.size:
        raise SNANAError("factorized covariance requires diag (N,) and U (N,K)")
    if not np.isfinite(diag).all() or (diag <= 0).any():
        raise SNANAError("factorized covariance statistical variances must be finite and positive")
    if not np.isfinite(vectors).all():
        raise SNANAError("factorized covariance U contains non-finite values")
    if labels and len(labels) != vectors.shape[1]:
        raise SNANAError("factorized covariance labels must match U columns")
    return {"diag": diag, "U": vectors, "vec_labels": labels}


def dense_factorized_covariance(covariance):
    """The dense matrix diag(D) + U U.T, for an output that asks for it."""
    return np.diag(covariance["diag"]) + covariance["U"] @ covariance["U"].T


def read_factorized_covariance(path, *, size=None):
    """Read a covariance stored as C = diag(D) + U U.T: ``nsn``, ``diag``,
    ``U`` and optional ``vec_labels``, checked against ``size`` rows."""
    with np.load(path, allow_pickle=False) as payload:
        if not {"nsn", "diag", "U"} <= set(payload.files):
            raise SNANAError(f"{path}: factorized covariance needs nsn, diag and U")
        nsn = payload["nsn"]
        if nsn.size != 1 or not np.issubdtype(nsn.dtype, np.integer):
            raise SNANAError(f"{path}: nsn must contain one integer")
        result = validate_factorized_covariance({
            "diag": payload["diag"], "U": payload["U"],
            "vec_labels": tuple(payload["vec_labels"].astype(str)) if "vec_labels" in payload else ()})
        if int(nsn.item()) != result["diag"].size or (size is not None and size != result["diag"].size):
            raise SNANAError(f"{path}: factorized covariance size does not match HD rows/nsn")
    return result


def write_factorized_covariance(path, covariance):
    """Write a covariance as C = diag(D) + U U.T, the inverse of
    :func:`read_factorized_covariance`."""
    covariance = validate_factorized_covariance(covariance)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, nsn=np.array([covariance["diag"].size]), diag=covariance["diag"],
             U=covariance["U"], vec_labels=np.asarray(covariance["vec_labels"], dtype=str))


def parse_prescales(raw) -> Dict[str, float]:
    """Parse a SNANA prescale dict, e.g. `C3/4.3+X3/2.2` (sntools.c
    parse_string_prescales). '+' is the default separator; a comma anywhere in
    the string switches to comma-separated. An entry with no `/value` takes
    prescale 1.0 (sntools.c:1002)."""
    if raw is None:
        return {}
    text = str(raw).strip()
    if is_ignorefile(text):
        return {}
    sep = ',' if ',' in text else '+'
    out: Dict[str, float] = {}
    for item in text.split(sep):
        item = item.strip()
        if not item:
            continue
        name, slash, value = item.partition('/')
        out[name.strip()] = float(value) if slash else 1.0
    return out
