"""SNANA's file formats, where more than one program reads or writes them.

- The values SNANA stores for an unset quantity.
- Finding and opening files: ``$SNDATA_ROOT``, the ``.gz`` spelling SNANA
  falls back to as CFITSIO does, FITS tables, the values that mean "no
  file", and the DOCUMENTATION block SNANA requires at the top of its data
  and config files so a published result can be traced back to its inputs.
- Reading SNANA's text files: their comments, words and ``KEY:`` values,
  map rows on a grid (GRIDMAP), two-column tables (rd2columnFile) and
  three-column SEDs (rd_sedFlux).
- SURVEY.DEF, and the cosmology recorded in a simulation's README.
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
import os
import re
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

from jax_snlc_core import logging as log
from jax_snlc_core.logging import SNANAError

logger = logging.getLogger(__name__)

# unset measured value (init_SNDATA_EVENT)
SNANA_NULLFLOAT = -9.0

# the magnitude of zero flux, as before explosion (sntools.h:125)
MAG_ZEROFLUX = 99.0

# a magnitude nothing defines, such as a missing host's (sntools.h:127)
MAG_UNDEFINED = 666.0

# A host property no catalogue supplied (HOSTLIB_PROPERTY_UNDEFINED, sntools.h:132).
HOSTGAL_PROPERTY_NULL = -9999.0


def sndata_root() -> Path:
    """$SNDATA_ROOT, which SURVEY.DEF, the models and the maps live under.
    SNANA reads it from the environment alone."""
    try:
        return Path(os.environ["SNDATA_ROOT"])
    except KeyError:
        raise SNANAError("$SNDATA_ROOT is not set", function_name="sndata_root") from None


def resolve_fits_path(path, required=False, plain_first=False):
    """Return the file that exists on disk for ``path``, plain or ``.gz``.

    SNANA gzips its output but references the plain name, and CFITSIO opens
    the ``.gz`` when given the base name; astropy does not, so every SNANA
    file is found through here. Both present is an error, as in SNANA's
    open_TEXTgz and rd_snfitsio_check_gzip, since which is current is
    ambiguous; ``plain_first`` takes the plain file instead, as CFITSIO
    does where SNANA opens a file with it directly (the kcor file and the
    SFD maps). With neither present, ``required`` aborts naming both
    spellings; otherwise the original path passes through so the caller's
    own open raises.
    """
    path = Path(path)
    gz_path = path.with_name(path.name + ".gz")
    if path.exists() and gz_path.exists() and not plain_first:
        raise SNANAError(f"found both {path} and {gz_path}; remove one",
                         function_name="resolve_fits_path")
    if path.exists():
        return str(path)
    if gz_path.exists():
        return str(gz_path)
    if required:
        raise SNANAError(f"file not found: {path} (also checked {path}.gz)",
                         function_name="resolve_fits_path")
    return str(path)


def read_fits_table(path):
    """The primary header and the binary table of a SNANA FITS file, plain
    or ``.gz``, as an Astropy table in the machine's byte order.

    astropy parses the two headers; the rows stream from the file into one
    buffer that is byte-swapped in place, so the table is held once and
    never copied. A ``TSCALn``/``TZEROn`` column, such as an unsigned
    integer, is scaled as CFITSIO scales it for SNANA.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        primary = fits.Header.fromfile(stream)
        header = fits.Header.fromfile(stream)
        fields = range(1, header["TFIELDS"] + 1)
        columns = fits.ColDefs([fits.Column(name=header[f"TTYPE{i}"],
                                            format=header[f"TFORM{i}"])
                                for i in fields])
        stored = columns.dtype.newbyteorder(">")
        if stored.itemsize != header["NAXIS1"]:
            raise SNANAError(f"{path}: table rows are {header['NAXIS1']} bytes, but "
                             f"their columns hold {stored.itemsize}",
                             function_name="read_fits_table")
        rows = bytearray(header["NAXIS2"] * stored.itemsize)
        view = memoryview(rows)
        done = 0
        # A gzip stream inflates each read through a temporary of the size
        # asked for, so read a piece at a time.
        while done < len(rows):
            got = stream.readinto(view[done:done + (1 << 20)])
            if not got:
                raise SNANAError(f"{path}: file ends {len(rows) - done} "
                                 "bytes short of its table",
                                 function_name="read_fits_table")
            done += got
    table = np.frombuffer(rows, stored)
    # Keep a block in cache while NumPy visits its individual columns.
    block_rows = max(1, (1 << 20) // stored.itemsize)
    for start in range(0, len(table), block_rows):
        table[start:start + block_rows].byteswap(inplace=True)
    table = Table(table.view(stored.newbyteorder("=")), copy=False)
    for i in fields:
        if f"TSCAL{i}" in header or f"TZERO{i}" in header:
            name = header[f"TTYPE{i}"]
            table[name] = (table[name] * np.asarray(header.get(f"TSCAL{i}", 1))
                           + np.asarray(header.get(f"TZERO{i}", 0)))
    return primary, table


def open_text(path):
    """Open a SNANA text file, accepting its plain name or ``.gz`` spelling."""
    path = Path(resolve_fits_path(path))
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path, "rt")


def is_ignorefile(filename: Optional[str]) -> bool:
    """True when a configured file value means "no file".

    Mirrors SNANA's IGNOREFILE (sntools.c:4829-4850): None, the empty
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
    (sntools.c:6992-6998).
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


# sntools.c:7036 commentchar()
COMMENT_CHARS = '#!%@'


def line_words(line: str) -> List[str]:
    """The words of a line of SNANA text before its first comment word, one
    starting with any of :data:`COMMENT_CHARS`. A comment character inside a
    word does not start a comment."""
    words = line.split()
    for i, word in enumerate(words):
        if word[0] in COMMENT_CHARS:
            return words[:i]
    return words


def text_lines(path) -> Iterator[List[str]]:
    """The words of each line of the text file ``path``, plain or ``.gz``,
    leaving out comments, blank lines and the DOCUMENTATION block."""
    with open_text(path) as fh:
        for line in skip_docana(fh):
            words = line_words(line)
            if words:
                yield words


def key_runs(path) -> Iterator[Tuple[str, List[str]]]:
    """Each ``KEY:`` of the key-value text file ``path`` with the words after
    it, up to the next key, as ``(KEY, words)``. SNANA reads these files as
    one stream of words, so a line may hold several keys and a key's values
    may run onto the next line."""
    return word_runs(path, (word for words in text_lines(path) for word in words))


def gridmap_segments(path, row_key: str) -> Iterator[List[Tuple[str, List[str]]]]:
    """:func:`key_runs` of ``path`` in segments, each ended by the blank line
    that ends a map's ``row_key`` rows; a comment line does not
    (read_GRIDMAP, sntools_gridmap.c:116-123)."""
    lines, after_row = [], False
    with open_text(path) as fh:
        for line in skip_docana(fh):
            if line.split():
                words = line_words(line)
                lines.append(words)
                after_row = after_row or words[:1] == [row_key]
            elif after_row:
                yield list(word_runs(path, (word for words in lines for word in words)))
                lines, after_row = [], False
    yield list(word_runs(path, (word for words in lines for word in words)))


def header_words(lines: Iterator[List[str]], stop: str) -> Iterator[str]:
    """The words of the lines ``lines`` up to the word ``stop``, which ends
    a file's header; the lines are consumed up to and including the one
    holding it. SNANA reads such headers as one stream of words."""
    for words in lines:
        if stop in words:
            yield from words[:words.index(stop)]
            return
        yield from words


def word_runs(path, words: Iterable[str]) -> Iterator[Tuple[str, List[str]]]:
    """Each ``KEY:`` among ``words``, read from the file ``path``, with the
    words after it, up to the next key, as ``(KEY, words)``. Words before the
    first key belong to no key; SNANA passes over them, and so does this,
    with a warning."""
    key, values = None, []
    for word in words:
        if word.endswith(':'):
            if key is None and values:
                log.warn(logger, "TEXT", f"{path}: '{' '.join(values)}' comes before "
                         f"any key and is not read")
            elif key is not None:
                yield key, values
            key, values = word[:-1], []
        else:
            values.append(word)
    if key is not None:
        yield key, values


def leading_values(path, key: str, values: List[str], count: int) -> List[str]:
    """The first ``count`` of the words ``values`` that follow ``key`` in the
    file ``path``. Where SNANA reads a file as a stream of words, each key
    takes the words it needs and the reader passes over the rest, such as a
    note after a value; too few words is an error."""
    if len(values) < count:
        raise SNANAError(f"{path}: {key}: takes {count} value(s), not "
                         f"'{' '.join(values)}'", function_name="leading_values")
    return values[:count]


def exact_values(path, key: str, values: List[str], count: int) -> List[str]:
    """``values`` when ``key`` of the file ``path`` has exactly ``count`` of
    them, so a short or long entry is an error rather than misread."""
    if len(values) != count:
        raise SNANAError(f"{path}: {key}: takes {count} value(s), not "
                         f"'{' '.join(values)}'", function_name="exact_values")
    return values


def split_names(text: str) -> List[str]:
    """The names in a list such as VARNAMES or PARNAMES, separated by commas,
    blanks or both (store_PARSE_WORDS, sntools.c:2643)."""
    return [name for name in re.split(r'[\s,]+', text) if name]


def gridmap(rows, varnames: Sequence[str], where: str
            ) -> Tuple[List[np.ndarray], np.ndarray]:
    """A map given as rows of its variables' values followed by its
    functions' values, as SNANA's GRIDMAP holds one (init_interp_GRIDMAP,
    sntools_gridmap.c:307-479): the knots of each of ``varnames``, which
    must be uniformly spaced, and the functions at every grid point, shape
    (*knots, n_functions). Every grid point needs exactly one row, in any
    order. ``where`` names the map in an error. Steps differing by more
    than 1e-6 of their size are not uniform (read_GRIDMAP,
    sntools_gridmap.c:162-167)."""
    data = np.asarray(rows, dtype=np.float64)
    if len(data) == 0:
        raise SNANAError(f"{where} has no rows", function_name="gridmap")
    ndim = len(varnames)
    axes = [np.unique(data[:, d]) for d in range(ndim)]
    for name, axis in zip(varnames, axes):
        steps = np.diff(axis)
        if np.any(np.abs(steps - steps[:1]) > 1.0e-6 * steps):
            raise SNANAError(f"{where}: {name} is not uniformly spaced",
                             function_name="gridmap")
    shape = tuple(len(axis) for axis in axes)
    cells = tuple(np.searchsorted(axis, data[:, d]) for d, axis in enumerate(axes))
    if len(data) != np.prod(shape) or len(np.unique(np.ravel_multi_index(cells, shape))) != len(data):
        raise SNANAError(f"{where}: {len(data)} rows for a {shape} grid; every grid "
                         f"point needs exactly one row", function_name="gridmap")
    values = np.empty((*shape, data.shape[1] - ndim))
    values[cells] = data[:, ndim:]
    return axes, values


def read_two_column_file(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read the first two columns of an ASCII table, plain or gzipped, as
    SNANA's rd2columnFile does (sntools.c:6937-7026); columns beyond the
    second are ignored, and a line without two numbers aborts."""
    col1, col2 = [], []
    for words in text_lines(path):
        try:
            col1.append(float(words[0]))
            col2.append(float(words[1]))
        except (IndexError, ValueError):
            raise SNANAError(f"{path}: '{' '.join(words)}' is not two numbers",
                             function_name="read_two_column_file") from None
    return np.array(col1), np.array(col2)


# rd_sedFlux's tolerances on uniform day and wavelength steps
# (sntools.c:8121-8122)
SED_DAYSTEP_TOL = 0.5e-3
SED_LAMSTEP_TOL = 0.01


def read_sed_file(filepath: str, day_range: Tuple[float, float],
                  wavelength_range: Tuple[float, float],
                  t0shift_peakmag: bool = True, *, uniform_days: bool
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a 3-column SED file (phase, wavelength, flux) at ``filepath``
    (``.SED`` or ``.SED.gz``), as SNANA's rd_sedFlux does for NON1ASED and
    SIMSED templates, SALT surfaces and error maps, and the calibration's SN
    SED, skipping blank lines, comments and the DOCUMENTATION block. A data
    line whose first three words are not numbers is an error.

    Rows outside ``day_range`` or ``wavelength_range`` (both inclusive) are
    excluded, and NaN fluxes are read as zero, with a warning. Phases must
    be in increasing order, with the same wavelengths in the same order at
    each phase. With ``uniform_days``, phase spacing must be constant within
    ``SED_DAYSTEP_TOL``. Non-uniform wavelength spacing produces a warning.

    When ``t0shift_peakmag`` (SNANA's default-on ``OPTMASK_T0SHIFT_PEAKMAG``,
    disabled by ``OPTFLAG_T0SHIFT_PEAKMAG: 0`` in a SIMSED SED.INFO), the
    phase grid is shifted so that peak bolometric flux occurs at phase=0.

    Returns ``(phases, wavelengths, flux_grid)``: phases is the 1-D array of
    unique phase values (days), wavelengths the 1-D array of unique
    wavelength values (Angstroms), and flux_grid the 2-D array of shape
    (n_phases, n_wavelengths).
    """
    filepath = str(filepath)

    data_rows = []
    for words in text_lines(filepath):
        try:
            data_rows.append((float(words[0]), float(words[1]), float(words[2])))
        except (IndexError, ValueError):
            raise SNANAError(f"{filepath}: '{' '.join(words)}' is not three numbers",
                             function_name="read_sed_file") from None

    if not data_rows:
        raise SNANAError(f"No valid SED data found in {filepath}",
                         function_name="read_sed_file")

    data = np.array(data_rows)
    inside = ((data[:, 0] >= day_range[0]) & (data[:, 0] <= day_range[1])
              & (data[:, 1] >= wavelength_range[0])
              & (data[:, 1] <= wavelength_range[1]))
    data = data[inside]
    if not len(data):
        raise SNANAError(
            f"No SED rows of {filepath} inside days {day_range} and "
            f"wavelengths {wavelength_range}", function_name="read_sed_file")
    raw_phases = data[:, 0]
    raw_wavelengths = data[:, 1]
    nan = np.isnan(data[:, 2])
    if nan.any():
        log.warn(logger, "SED", f"{filepath}: {int(nan.sum())} NaN fluxes are read as zero")
    raw_flux = np.where(nan, 0.0, data[:, 2])

    # days never decrease (sntools.c:8272-8276)
    decreasing = np.flatnonzero(np.diff(raw_phases) < 0)
    if decreasing.size:
        i = decreasing[0]
        raise SNANAError(
            f"{filepath}: day {raw_phases[i + 1]:f} follows day "
            f"{raw_phases[i]:f}; a new day must increment",
            function_name="read_sed_file")
    # every day has as many wavelengths as the first (:8675-8690)
    phases, per_day = np.unique(raw_phases, return_counts=True)
    if np.any(per_day != per_day[0]):
        k = int(np.argmax(per_day != per_day[0]))
        raise SNANAError(
            f"{filepath}: {per_day[k]} wavelengths at day {phases[k]:.2f} but "
            f"{per_day[0]} at day {phases[0]:.2f}; the wavelength grid must "
            f"be the same for each day", function_name="read_sed_file")
    # ... and lists the first day's wavelengths, in order (:8612-8624)
    n_phases, n_wavelengths = len(phases), int(per_day[0])
    lam_grid = raw_wavelengths.reshape(n_phases, n_wavelengths)
    wavelengths = lam_grid[0]
    # a falling wavelength restarts the list at its first entry (:8583)
    falling = np.flatnonzero(np.diff(wavelengths) < 0)
    if falling.size:
        j = falling[0] + 1
        raise SNANAError(
            f"{filepath}: found LAM={wavelengths[j]:.1f} at day "
            f"{phases[0]:f} but expected LAM={wavelengths[0]:.2f}",
            function_name="read_sed_file")
    mismatched = np.argwhere(lam_grid != wavelengths)
    if mismatched.size:
        k, j = mismatched[0]
        raise SNANAError(
            f"{filepath}: found LAM={lam_grid[k, j]:.1f} at day "
            f"{phases[k]:f} but expected LAM={wavelengths[j]:.2f}",
            function_name="read_sed_file")
    # uniform day steps unless the model allows otherwise (:8549-8560)
    day_step = np.diff(phases)
    step_change = np.abs(np.diff(day_step)) > SED_DAYSTEP_TOL
    if uniform_days and step_change.any():
        k = int(np.argmax(step_change)) + 1
        raise SNANAError(
            f"{filepath}: day step {day_step[k]:e} after {day_step[k - 1]:e} "
            f"at day {phases[k + 1]:.4f}; the day bins must be uniform",
            function_name="read_sed_file")
    # a non-uniform wavelength step only warns (:8595-8606)
    if np.any(np.abs(np.diff(np.diff(wavelengths))) > SED_LAMSTEP_TOL):
        log.warn(logger, 'SED', f"{filepath}: non-uniform wavelength step "
                 f"(rd_sedFlux warns and reads on)")

    # phase varies slowest, wavelength fastest
    flux_grid = raw_flux.reshape(n_phases, n_wavelengths)

    # T0shiftPeak_SEDMODEL: peak bolometric flux at day 0
    if t0shift_peakmag:
        bolometric = flux_grid.sum(axis=1)
        peak_idx = np.argmax(bolometric)
        phases = phases - phases[peak_idx]

    return phases, wavelengths, flux_grid


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
    for toks in text_lines(path):
        if toks[0] == prefix:
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
        # A repeated name takes the lowest ID (get_IDSURVEY, sntools.c:3709).
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
    FIELDLIST in this order, snana.F90:19209), upper-cased, leaving out a
    field whose IDFIELD is 0 or less, such as ``INT``, which says how a
    survey's fields are written rather than naming one (snana.F90:12455-12477).
    A '+', which joins the fields of an overlap, is not allowed in a name."""
    fields = []
    for toks in _survey_def_rows(path, 'FIELD:'):
        if len(toks) >= 4 and toks[3] == name and int(toks[2]) > 0:
            if '+' in toks[1]:
                raise SNANAError(f"SURVEY.DEF field {toks[1]} contains '+', which joins "
                                 f"the fields of an overlap", function_name="load_survey_fields")
            fields.append(toks[1].upper())
    return fields


# KEYLIST_COSPAR_SIM (create_covariance.py:293)
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
    create_covariance.py:2655).

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

# the row keys of a table's data rows (validRowKey_TEXT, sntools_output_text.c:1430)
TABLE_ROW_KEYS = ("SN:", "ROW:", "GAL:", "OBS:")

# Columns read as strings: these names in any case, and any name containing
# one of the parts (ICAST_for_textVar, sntools_output_text.c:1473-1502).
STRING_COLUMNS = frozenset((
    "CID", "SNID", "CCID", "GALID", "ROW", "STARID", "FIELD", "BAND", "BANDLIST",
    "NAME_TRANSIENT", "NAME_IAUC", "IAUC", "CATALOG", "VERSION", "PARNAME", "OBJID",
    "SIM_TYPE"))
STRING_COLUMN_PARTS = ("CCID", "ROW", "GALID", "OBJID")


def is_string_column(name: str) -> bool:
    """Whether a text table's column ``name`` holds strings."""
    return (name.upper() in STRING_COLUMNS
            or any(part in name for part in STRING_COLUMN_PARTS))


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


def read_fitres(path: str, as_text: bool = False, *, row_keys=TABLE_ROW_KEYS,
                string_column=is_string_column) -> FitresTable:
    """Parse one FITRES/M0DIF file (``path``, optionally gzipped) into a
    :class:`FitresTable`.

    The columns ``string_column`` accepts stay literal strings, such as
    ``FIELD 01`` and an ``NA`` CID; the rest take inferred dtypes.
    ``as_text`` keeps every value as its original token and records the row
    key for concatenation (``cat_only``).

    As SNANA's SNTABLE_READ_EXEC_TEXT reads a table (sntools_output_text.c:
    1255-1282), only lines starting with one of :data:`TABLE_ROW_KEYS` are
    rows. ``row_keys`` and ``string_column`` follow a format with its own
    reader, such as a HOSTLIB. Values past the last VARNAMES column are
    ignored, and a row with fewer values than VARNAMES aborts. A ``*`` in a
    string first column repeats the row above's value.
    """
    meta: Dict[str, str] = {}
    varnames: Optional[List[str]] = None

    with open_text(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("VARNAMES:"):
                varnames = split_names(s[len("VARNAMES:"):])
                break
            if s.startswith("#"):
                _parse_fitres_header_line(s, meta)

        if varnames is None:
            raise SNANAError(f"No VARNAMES line found in {path}",
                             function_name="read_fitres")

        rows = io.StringIO()
        for line_number, line in enumerate(f, start=1):
            words = line_words(line)
            if not words or words[0] not in row_keys:
                continue
            if len(words) < len(varnames) + 1:
                raise SNANAError(
                    f"{path}: row at line {line_number} after VARNAMES holds fewer "
                    "values than VARNAMES names", function_name="read_fitres")
            rows.write(" ".join(words) + "\n")

    rows.seek(0)
    df = pd.read_csv(
        rows, sep=" ", header=None, names=[_ROWKEY] + varnames, quoting=csv.QUOTE_NONE,
        usecols=range(len(varnames) + 1), dtype=str if as_text else None,
        converters=None if as_text else {
            name: str for name in varnames if string_column(name)},
        keep_default_na=not as_text,
    )
    rowkeys = df.pop(_ROWKEY)
    rowkey = rowkeys.to_numpy() if as_text else None
    first = varnames[0]
    if string_column(first):
        # a '*' repeats the row above's value (sntools_output_text.c:1355-1361)
        if len(df) and df[first].iloc[0] == "*":
            raise SNANAError(f"{path}: the first row's {first} is '*', but no row "
                             f"above it gives a value to repeat", function_name="read_fitres")
        df[first] = df[first].mask(df[first] == "*").ffill()
    return FitresTable(df=df, varnames=varnames, meta=meta,
                       varnames_by_file=[varnames], rowkey=rowkey)


# VARNAME_TABLE_IDLIST, sntools_output.h:88
TABLE_ID_COLUMNS = ("ROW", "CID", "CCID", "SNID", "GALID", "HOSTGAL_OBJID")


def table_ids(path, table: FitresTable) -> List[str]:
    """The names of the rows of ``table``, read from ``path``: its first
    column of :data:`TABLE_ID_COLUMNS` (SNTABLE_AUTOSTORE_INIT,
    sntools_output.cpp:2253)."""
    id_columns = [name for name in TABLE_ID_COLUMNS if table.has(name)]
    if not id_columns:
        raise SNANAError(f"{path} has none of the columns {TABLE_ID_COLUMNS} to name its rows",
                         function_name="table_ids")
    return [str(cid) for cid in table.df[id_columns[0]]]


def read_cid_list(path) -> Tuple[List[str], Optional[FitresTable]]:
    """The CIDs of a CID list file, as SNANA reads one (match_cidlist_init,
    sntools.c:260-413): a table under VARNAMES, read as :func:`read_fitres`
    reads it and named by :func:`table_ids`, or else every word of the file,
    a word holding commas split at them. Returns the CIDs and the table, or
    None for a plain list."""
    if any(words[0] == 'VARNAMES:' for words in text_lines(path)):
        table = read_fitres(path)
        return table_ids(path, table), table
    return [cid for words in text_lines(path) for cid in split_names(' '.join(words))], None


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


# must parse as >= 11.02 for the VPEC sign (correct_sign_vpec_data)
SNANA_VERSION_STRING = "v12_02f-JAX"


def snana_version_comment():
    """The SNANA_VERSION header line (sntools_output_text.c): the one value
    every jax output carries, which readers parse as a release."""
    return f"SNANA_VERSION: {SNANA_VERSION_STRING}"


def program_comment(program):
    """The PROGRAM header line SALT2mu's tables carry after SNANA_VERSION
    (SALT2mu.c:23569), naming the script that wrote the file."""
    return f"PROGRAM:       {program}"


def snana_value_strings(name, values):
    """Format a float column as SNANA's text writer does.

    The choice is per value, not per column, so one column can hold both
    ``-9`` and ``2.29797e-05`` (sntools_output_text.c:478-506).
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
                         sntable_frame=True, post_varnames_comments=()):
    """Write an SNANA text table (sntools_output_text.c:612-785 formats):
    FITRES (``SN:``), M0DIF (``ROW:``), SNANA, OUTLIER, LCPLOT and the BBC
    diagnostic dumps.

    ``columns`` maps varname -> array; ``fmt`` maps varname -> printf format.
    A column it leaves out, or maps to None, is written as SNANA's own text
    writer would: strings as they are, numbers by the per-value rule.
    Styles: key (# comments + VARNAMES: + rowkey rows), csv, col, none.
    ``header_comments`` start every style; with ``sntable_frame`` they are
    followed by a blank line, and a key table has ``#`` after its VARNAMES,
    as SNANA's table writer frames them (SNTABLE_WRITE_HEADER_TEXT,
    sntools_output_text.c:714-760).
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

    framed = bool(header_comments) and sntable_frame
    with open(output_path, 'w') as f:
        for line in header_comments:
            # None is a truly blank separator line; "" is a bare "# "
            f.write("\n" if line is None else f"# {line}\n")
        if framed:
            f.write("\n")
        if style == 'key':
            f.write(f"VARNAMES: {' '.join(varnames)}\n")
            if framed:
                f.write("#\n")
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
# the most photo-z quantile columns SNANA reads from a HOSTLIB (sndata.h:187)
MXBIN_HOSTGALz_QUANTILE = 20

#: sndata.h:87-90
HOSTGAL_PROPERTY_BASENAMES = ('LOGMASS', 'LOGSFR', 'LOGsSFR', 'COLOR')

#: sntools_host.h:123-128, each to the spelling kept in memory
HOST_COORDINATE_NAMES = {'RA': 'RA_GAL', 'RA_HOST': 'RA_GAL', 'RA_GAL': 'RA_GAL',
                         'DEC': 'DEC_GAL', 'DEC_HOST': 'DEC_GAL',
                         'DEC_GAL': 'DEC_GAL'}

#: sntools_host.c:3020-3044: every other spelling of a HOSTLIB column, to the
#: one kept in memory. ZTRUE_CMB is the redshift in the CMB frame, which
#: read_hostlib turns into the heliocentric ZTRUE.
HOST_ALTERNATE_NAMES = {**HOST_COORDINATE_NAMES, 'REDSHIFT': 'ZTRUE',
                        'ZTRUE_CMB': 'ZTRUE', 'VPECERR': 'VPEC_ERR'}


def hostgal_null_key(col):
    """Column name with any trailing band stripped, for sentinel lookup;
    SNANA bands are single characters."""
    stem, _, band = col.rpartition('_')
    return stem if len(band) == 1 and stem in HOSTGAL_NO_HOST_NULL else col


def canonical_property_name(name: str) -> str:
    """The in-memory spelling of a host column: ``LOGMASS_TRUE`` and
    ``LOGMASS`` are both ``LOGMASS``, ``RA`` and ``RA_HOST`` are ``RA_GAL``,
    ``REDSHIFT`` is ``ZTRUE``; any other name is itself.

    SNANA rewrites each name it reads wherever it reads one: the HOSTLIB
    header, a WGTMAP's variables and ``HOSTLIB_STOREPAR``
    (checkAlternateVarNames_HOSTLIB, sntools_host.c:2981, applied at :2492,
    :1344, :1109). It reads the host coordinates under ``RA``, ``RA_HOST`` or
    ``RA_GAL``, and likewise for DEC (sntools_host.c:2758-2768). Every name
    read from an input goes through here first."""
    if name.endswith('_TRUE') and name[:-5] in HOSTGAL_PROPERTY_BASENAMES:
        return name[:-5]
    return HOST_ALTERNATE_NAMES.get(name, name)


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


def _read_wgtmap_header(filepath, lines):
    """Consume the header from the word ``lines`` of ``filepath`` and return
    dimension and function names.

    Columns before WGT are dimensions; WGT and subsequent columns are
    functions (read_WGTMAP, sntools_wgtmap.c:78-92).
    """
    for words in lines:
        if words[0] in ('VARNAMES:', 'VARNAMES_WGTMAP:'):
            names = split_names(' '.join(words[1:]))
            if 'WGT' not in names or names[0] == 'WGT':
                raise SNANAError(f"Invalid VARNAMES_WGTMAP in {filepath}",
                                 function_name='_read_wgtmap_header')
            iwgt = names.index('WGT')
            return [canonical_property_name(n) for n in names[:iwgt]], names[iwgt:]
    raise SNANAError(f"VARNAMES_WGTMAP line not found in {filepath}",
                     function_name='_read_wgtmap_header')


def _parse_wgtmap(filepath: str) -> Tuple[List[str], List[str], List[List[float]]]:
    """Read dimension names, function names and grid rows in one pass."""
    lines = text_lines(filepath)
    varnames, fun_names = _read_wgtmap_header(filepath, lines)
    ncol = len(varnames) + len(fun_names)
    rows = [[float(x) for x in exact_values(filepath, 'WGT', words[1:], ncol)]
            for words in lines if words[0] == 'WGT:']
    return varnames, fun_names, rows


def read_wgtmap_varnames(filepath: str) -> List[str]:
    """Read only the WGTMAP dimension names, without loading its grid."""
    filepath = resolve_fits_path(filepath, required=True)
    varnames, _ = _read_wgtmap_header(filepath, text_lines(filepath))
    return varnames


def read_wgtmap(filepath: str, sn_variables, opt_extrap: int = 0) -> WgtMapData:
    """Read and parse a WGTMAP file (``filepath``, .gz or plain text) into a
    :class:`WgtMapData` (read_WGTMAP, sntools_wgtmap.c): the weight grid of
    HOSTLIB galaxy selection and SIMSED surface selection.

    ``sn_variables`` maps each lower-cased spelling of a per-SN quantity,
    rather than a catalogue column, to the name the caller stores it under;
    a dimension it spells takes that name. The caller derives it from its
    model (checkSNvar_HOSTLIB_WGTMAP, sntools_host.c:2886).
    ``opt_extrap`` is 0 to abort if any galaxy is out of bounds (SNANA
    default) or 1 to clamp out-of-bounds values to the grid edge.
    """
    filepath = resolve_fits_path(filepath, required=True)
    logger.info(f"Reading WGTMAP file: {filepath}")

    varnames, fun_names, data = _parse_wgtmap(filepath)
    ndim = len(varnames)
    log.line(logger, f"Dimensions: {ndim} ({', '.join(varnames)})")
    log.line(logger, f"Loaded {len(data)} grid points")

    axes, functions = gridmap(data, varnames, f"WGTMAP {filepath}")
    nbins = np.array([len(ax) for ax in axes], dtype=np.int32)
    ranges = np.array([[ax[0], ax[-1]] for ax in axes])
    wgt_grid = functions[..., 0]

    # an all-zero SNMAGSHIFT means unused
    snmagshift_grid = None
    if 'SNMAGSHIFT' in fun_names:
        snmagshift = functions[..., fun_names.index('SNMAGSHIFT')]
        if not np.allclose(snmagshift, 0.0):
            snmagshift_grid = jnp.array(snmagshift, dtype=jnp.float32)

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
    (wfit.c:2078), so the per-element labels that ``--label_cov_rows`` appends
    are ignored.
    """
    values = [float(words[0]) for words in text_lines(path)]
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
    prescale 1.0 (sntools.c:1001)."""
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
