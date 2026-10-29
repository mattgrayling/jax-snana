"""
SEARCHEFF map reader and evaluator for SPECID and zHOST efficiency maps.

Handles the SNANA SEARCHEFF file format with:

- Multiple maps per file, each a VARNAMES line and its rows
- N-D uniform grid interpolation
- FIELD, REQUIRE, PEAKMJD_RANGE per map
- OR multi-map combination (SNANA's BOOLEAN_AND is initialised 0 and
  never set, sntools_trigger.c:1108-1109)
- Special file values: NONE/ONE → EFF=1, ZERO → EFF=0

Variables supported: REDSHIFT, PEAKMJD, DTSEASON_PEAK, DTPEAK,
peak magnitudes (single char like 'r', 'i'), colours ('g-r', 'r-i'),
LOGMASS, HOSTMAG_<filt>, SBMAG_<filt>, SALT2mB, SALT2x1, SALT2c,
and arbitrary HOSTLIB column names (r_obs_auto, obs_gr_auto, etc.)
"""

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core import logging as log
from jax_snlc_core.filters import filter_band
from jax_snlc_core.interpolation import interp_nd_regular
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import (COMMENT_CHARS, check_file_docana,
                                       is_ignorefile, open_text, resolve_fits_path,
                                       sndata_root)

logger = logging.getLogger(__name__)


FLAG_NORMAL = 0
FLAG_ALL_PASS = 1   # NONE or ONE
FLAG_ALL_FAIL = 2   # ZERO

# gen_searcheff_map round-off window: below 0 -> 0, above 1 -> 0.99999
EFF_ROUNDOFF_MIN = -0.001
EFF_ROUNDOFF_MAX = 1.0003

# APPLY_SEARCHEFF_OPT bits (sntools_trigger.h:84-87)
APPLYMASK_SEARCHEFF_PIPELINE = 1
APPLYMASK_SEARCHEFF_SPECID = 2
APPLYMASK_SEARCHEFF_zHOST = 4
APPLYMASK_SEARCHEFF_zSPEC = 8

MXMAP_SEARCHEFF_MAP = 50      # sntools_trigger.h:50
MXROW_SEARCHEFF_MAP = 30000   # sntools_trigger.h:51
MXDIM_1DINDEX = 10            # sntools.h:285

# the efficiency column of either map type (assign_MAP_VARNAME, sntools_trigger.c:1507-1517)
EFF_COLUMN_NAMES = ('SPECEFF', 'HOSTEFF')

# row key and input key per map type (read_searcheff_map, sntools_trigger.c:1281-1294)
_MAPTYPE_KEYS = {'SPEC': ('SPECEFF:', 'SEARCHEFF_SPEC_FILE'),
                 'zHOST': ('HOSTEFF:', 'SEARCHEFF_zHOST_FILE')}

# map attributes until a FIELD/PEAKMJD/REQUIRE key sets them (sntools_trigger.c:1341-1342)
_FIELDLIST_ALL = 'ALL'
_PEAKMJD_RANGE_DEFAULT = (10000.0, 90000.0)

_C_WHITESPACE = ' \t\n\v\f\r'  # what fscanf("%s") skips
_INT_PREFIX = re.compile(r'[+-]?\d+')
_FLOAT_PREFIX = re.compile(r'[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?')


@dataclass
class SearchEffMap:
    """One efficiency map (one VARNAMES block) from a SEARCHEFF file."""
    varnames: List[str]          # excluding the SPECEFF/HOSTEFF column
    grid_axes: List[np.ndarray]  # the uniform knots along each dimension
    grid_values: np.ndarray
    require: bool = False        # REQUIRE: 1 -> independent random draw
    # FIELD/FIELDLIST string, matched by substring; 'ALL' matches every field
    fieldlist: str = _FIELDLIST_ALL
    peakmjd_range: Optional[Tuple[float, float]] = None
    opt_extrap: int = 0          # 0 = zero out of range, >0 = clip to edge
    # an event matching no map aborts (1) or gets EFF=0 (0) (sntools_trigger.c:1356, :2984)
    opt_fieldmatch_require: int = 1
    # per SIMLIB entry, from `fieldlist` by `resolve_fieldlist_indices`
    fieldlist_mask: Optional[np.ndarray] = None


class _TextStream:
    """A text file read as SNANA reads it: word by word with fscanf("%s"),
    and line by line with fgets."""

    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def word(self) -> Optional[str]:
        """The next whitespace-separated word, or None at the end of the file."""
        text, n, i = self.text, len(self.text), self.pos
        while i < n and text[i] in _C_WHITESPACE:
            i += 1
        j = i
        while j < n and text[j] not in _C_WHITESPACE:
            j += 1
        self.pos = j
        return text[i:j] if j > i else None

    def line(self, size: int) -> str:
        """fgets(LINE, size): the rest of the line up to and including its
        newline, at most ``size - 1`` characters; '' at the end of the file."""
        start = self.pos
        limit = min(len(self.text), start + size - 1)
        newline = self.text.find('\n', start, limit)
        self.pos = newline + 1 if newline >= 0 else limit
        return self.text[start:self.pos]


def _parse_words(line: str, comma: bool) -> List[str]:
    """The words of ``line`` as store_PARSE_WORDS splits a string and
    get_PARSE_WORD trims each word (store_PARSE_WORDS, splitString2 and
    trim_blank_spaces, sntools.c:2646, :6888, :6713).

    Words are separated by blanks, or by commas when ``comma`` is set and the
    line holds one. A tab does not separate words. Each word ends at its
    first blank, newline or carriage return."""
    sep = ',' if comma and ',' in line else ' '
    words = []
    for token in line.split(sep):
        if not token or token[0] == '\n':
            continue
        words.append(re.split(r'[ \n\r]', token.lstrip(' '), maxsplit=1)[0])
    return words


def _read_number(stream: _TextStream, pattern, key: str, path: Path) -> str:
    """The leading number of the word after ``key``, as readint and
    readdouble take it (sntools.c:11650-11680, :11744-11775)."""
    word = stream.word()
    match = pattern.match(word or '')
    if match is None:
        raise SNANAError(f"Could not read a number for {key} from '{word}' "
                         f"in {path}", function_name='read_searcheff_maps')
    return match.group(0)


def resolve_searcheff_file(name: str, key_file: str, path_user_input: str
                           ) -> Path:
    """A SEARCHEFF file (the pipeline, logic, SPEC or zHOST file): ``name``
    as given, else under each PATH_USER_INPUT directory, else under
    $SNDATA_ROOT/models/searcheff, each plain or ``.gz`` (PATH_SEARCHEFF,
    sntools_trigger.c:105; snana_openTextFile and open_TEXTgz, sntools.c).
    A file present both plain and gzipped aborts, as open_TEXTgz does.
    ``key_file`` is the input key, for the error."""
    given = Path(os.path.expandvars(name))
    dirs = [Path(os.path.expandvars(d)) for d in path_user_input.split()]
    dirs.append(sndata_root() / 'models' / 'searcheff')
    for candidate in [given] + [directory / given for directory in dirs]:
        gzipped = candidate.with_name(candidate.name + '.gz')
        if candidate.exists() and gzipped.exists():
            raise SNANAError(
                f"{key_file}: found both {candidate} and {gzipped}; "
                f"remove one", function_name='open_TEXTgz')
        found = Path(resolve_fits_path(candidate))
        if found.exists():
            return found
    raise SNANAError(
        f"{key_file} {name} not found, as given or under "
        f"{', '.join(str(d) for d in dirs)}",
        function_name='snana_openTextFile')


def read_searcheff_maps(filepath: str, maptype: str, path_user_input: str
                        ) -> Tuple[List[SearchEffMap], int]:
    """Read a SEARCHEFF_SPEC_FILE (``maptype`` 'SPEC') or a
    SEARCHEFF_zHOST_FILE ('zHOST') as read_searcheff_map does
    (sntools_trigger.c:1238-1470).

    NONE (or another IGNOREFILE spelling) and ONE give every event EFF=1,
    ZERO gives EFF=0 (:1297-1316). Otherwise the file, its DOCANA block
    included, is read word by word. Only a ``VARNAMES:`` word opens a map,
    whose rows follow it, keyed ``SPECEFF:`` or ``HOSTEFF:`` by map type
    (`_read_map_rows`). Every other word that is not a key is passed over,
    so rows with no VARNAMES line above them are never read, and a file
    with no map gives every event EFF=1 (gen_searcheff_map, :2898-2908).

    ``FIELD:``/``FIELDLIST:``, ``PEAKMJD:``/``PEAKMJD_RANGE:`` and
    ``REQUIRE:`` apply to the next map and reset after it (:1361-1368,
    :1413-1415). ``OPT_EXTRAP:`` applies to the maps after it (:1396), and
    ``OPT_FIELDMATCH_REQUIRE:`` to the whole file, its last value winning
    (:1356-1357, :2864).

    ``path_user_input`` is PATH_USER_INPUT, searched for a file not found
    as given (`resolve_searcheff_file`).

    Returns ``(maps, flag)``: flag is FLAG_ALL_PASS for NONE and ONE,
    FLAG_ALL_FAIL for ZERO, and FLAG_NORMAL for a file, which may hold no
    map.
    """
    key_row, key_file = _MAPTYPE_KEYS[maptype]
    if is_ignorefile(filepath) or filepath == 'ONE':
        return [], FLAG_ALL_PASS
    if filepath == 'ZERO':
        return [], FLAG_ALL_FAIL

    path = resolve_searcheff_file(filepath, key_file, path_user_input)
    # DOCANA required whatever REQUIRE_DOCANA says (snlc_sim.c:1297)
    check_file_docana(path)
    with open_text(path) as f:
        stream = _TextStream(f.read())

    maps = []
    opt_extrap = 0
    opt_fieldmatch_require = 1
    fieldlist, peakmjd_range, require = (
        _FIELDLIST_ALL, _PEAKMJD_RANGE_DEFAULT, 0)
    while (word := stream.word()) is not None:
        if word[0] in COMMENT_CHARS:
            stream.line(100)
            continue
        if word == 'NVAR:':
            logger.warning(f"Should remove obsolete NVAR key from {path}")
        elif word == 'OPT_EXTRAP:':
            opt_extrap = int(_read_number(stream, _INT_PREFIX, word, path))
        elif word == 'OPT_FIELDMATCH_REQUIRE:':
            opt_fieldmatch_require = int(
                _read_number(stream, _INT_PREFIX, word, path))
        elif word in ('FIELD:', 'FIELDLIST:'):
            fieldlist = stream.word()
            if fieldlist is None:
                break
        elif word in ('PEAKMJD:', 'PEAKMJD_RANGE:'):
            peakmjd_range = (
                float(_read_number(stream, _FLOAT_PREFIX, word, path)),
                float(_read_number(stream, _FLOAT_PREFIX, word, path)))
        elif word == 'REQUIRE:':
            require = int(_read_number(stream, _INT_PREFIX, word, path))
        # maps past the limit are passed over (:1373)
        elif word == 'VARNAMES:' and len(maps) < MXMAP_SEARCHEFF_MAP:
            names = _parse_words(stream.line(100), comma=True)
            maps.append(_read_map(stream, names, key_row, key_file, path,
                                  opt_extrap, fieldlist, peakmjd_range,
                                  require))
            fieldlist, peakmjd_range, require = (
                _FIELDLIST_ALL, _PEAKMJD_RANGE_DEFAULT, 0)

    for m in maps:
        m.opt_fieldmatch_require = opt_fieldmatch_require
    return maps, FLAG_NORMAL


def _read_map(stream, names, key_row, key_file, path, opt_extrap, fieldlist,
              peakmjd_range, require) -> SearchEffMap:
    """One map: the VARNAMES ``names`` and the rows after them.

    The efficiency is the column named SPECEFF or HOSTEFF, in either file
    (assign_MAP_VARNAME, sntools_trigger.c:1507-1517). It must be the last
    column: read_GRIDMAP takes the last column as the function of the others
    (:1397-1400), and an efficiency column anywhere else is a map variable
    SNANA cannot evaluate (LOAD_SEARCHEFF_VAR, :3125-3128).
    """
    where = f"{key_file} {path}, map 'VARNAMES: {' '.join(names)}'"
    eff_columns = [i for i, name in enumerate(names) if name in EFF_COLUMN_NAMES]
    if not eff_columns:
        raise SNANAError(f"{where}: no efficiency column; name it SPECEFF or "
                         f"HOSTEFF", function_name='read_searcheff_maps')
    if eff_columns != [len(names) - 1]:
        raise SNANAError(f"{where}: the efficiency column "
                         f"{names[eff_columns[0]]} must be the last column",
                         function_name='read_searcheff_maps')
    ndim = len(names) - 1
    # init_1DINDEX, sntools_gridmap.c:825-829
    if not 1 <= ndim < MXDIM_1DINDEX:
        raise SNANAError(f"{where}: {ndim} map variables; a map takes 1 to "
                         f"{MXDIM_1DINDEX - 1}",
                         function_name='read_searcheff_maps')
    # interp_GRIDMAP has no rule for a negative OPT_EXTRAP (sntools_gridmap.c:579-581)
    if opt_extrap < 0:
        raise SNANAError(f"{key_file} {path}: OPT_EXTRAP: {opt_extrap} is not "
                         f"supported; 0 gives zero outside a map, 1 its edge "
                         f"value", function_name='read_searcheff_maps')
    rows = _read_map_rows(stream, len(names), key_row, where)
    axes, values = _uniform_grid(np.asarray(rows), names[:-1], where)
    return SearchEffMap(varnames=names[:-1], grid_axes=axes,
                        grid_values=values, require=bool(require),
                        fieldlist=fieldlist, peakmjd_range=peakmjd_range,
                        opt_extrap=opt_extrap)


def _read_map_rows(stream, nvartot, key_row, where) -> List[List[float]]:
    """The rows of one map, read line by line as read_GRIDMAP reads them
    (sntools_gridmap.c:23-240).

    A line starting ``key_row`` holds the ``nvartot`` values, then nothing
    or a comment starting with '#'. Any other line is passed over, but ten
    in a row, or twenty lines without a row, abort. The first blank line
    after a row ends the map. Along each map variable, every positive step
    between successive rows must equal the variable's previous positive
    step to 1e-6, 1-D maps included.
    """
    rows = []
    n_line = n_skip = n_bad = 0
    first_bad = None
    ndim = nvartot - 1
    last, step = [None] * ndim, [None] * ndim
    while True:
        words = _parse_words(stream.line(200), comma=False)
        n_line += 1
        if n_line > 20 and not rows:
            raise SNANAError(f"{where}: no '{key_row}' row in the 20 lines "
                             f"after VARNAMES", function_name='read_GRIDMAP')
        if not words:
            if rows:
                break
            continue
        # the empty stop key (sntools_trigger.c:1256, sntools_gridmap.c:130)
        if words[0] == '':
            break
        if words[0] != key_row:
            n_skip += 1
            if n_skip >= 10:
                raise SNANAError(
                    f"{where}: 10 lines in a row that are not '{key_row}' "
                    f"rows; a blank line ends a map",
                    function_name='read_GRIDMAP')
            continue
        n_skip = 0
        n_values = len(words) - 1
        if n_values < nvartot or (n_values > nvartot
                                  and not words[nvartot + 1].startswith('#')):
            raise SNANAError(
                f"{where}: expected {nvartot} values after '{key_row}', "
                f"found {n_values} in row {len(rows)}",
                function_name='read_GRIDMAP')
        values = []
        for word in words[1:nvartot + 1]:
            match = _FLOAT_PREFIX.match(word)
            if match is None:
                raise SNANAError(f"{where}: '{word}' in row {len(rows)} is not "
                                 f"a number", function_name='read_GRIDMAP')
            values.append(float(match.group(0)))
        # uniform-bin check (sntools_gridmap.c:163-177)
        for d in range(ndim):
            if last[d] is not None:
                dif = values[d] - last[d]
                if dif > 0.0:
                    if step[d] is not None and abs((dif - step[d]) / dif) > 1.0e-6:
                        n_bad += 1
                        first_bad = len(rows) if first_bad is None else first_bad
                    step[d] = dif
            last[d] = values[d]
        rows.append(values)
        if len(rows) >= MXROW_SEARCHEFF_MAP:
            raise SNANAError(f"{where}: {len(rows)} rows reach the limit of "
                             f"{MXROW_SEARCHEFF_MAP}",
                             function_name='read_GRIDMAP')
    if not rows:
        raise SNANAError(f"{where}: no '{key_row}' rows",
                         function_name='read_GRIDMAP')
    if n_bad:
        raise SNANAError(
            f"{where}: {n_bad} non-uniform bin(s), the first at row "
            f"{first_bad}; every map variable needs uniform steps",
            function_name='read_GRIDMAP')
    return rows


def _uniform_grid(data: np.ndarray, varnames: List[str], where: str
                  ) -> Tuple[List[np.ndarray], np.ndarray]:
    """The knots of each map variable and the efficiency at every grid
    point, as init_interp_GRIDMAP builds them (sntools_gridmap.c:307-479).

    A variable's bin is its last positive step between successive rows,
    and its number of knots the range over that bin, which must divide the
    range to 1e-4. Each row lands on the knot just below its value; a later
    row on the same grid point replaces an earlier one. A grid the rows do
    not fill aborts, as does a variable of a map with several variables that
    takes one value only, since interp_GRIDMAP cannot interpolate across it
    (sntools_gridmap.c:649-661).
    """
    ndim = len(varnames)
    eff = data[:, ndim]
    axes, cells = [], []
    for d, name in enumerate(varnames):
        col = data[:, d]
        lo, hi = col.min(), col.max()
        span = hi - lo
        if span <= 1.0e-12:
            if ndim > 1:
                raise SNANAError(f"{where}: {name} takes a single value",
                                 function_name='init_interp_GRIDMAP')
            axes.append(np.array([lo]))
            cells.append(np.zeros(len(col), dtype=int))
            continue
        steps = np.diff(col)
        steps = steps[steps > 0.0]
        if steps.size == 0:
            raise SNANAError(f"{where}: {name} never increases from one row "
                             f"to the next", function_name='init_interp_GRIDMAP')
        bin_size = steps[-1]
        n_bin = int((span + 0.001 * bin_size) / bin_size) + 1
        if abs((n_bin - 1) * bin_size / span - 1.0) > 1.0e-4:
            raise SNANAError(
                f"{where}: non-uniform binning in {name}: its range {span:g} "
                f"is not a whole number of bins of {bin_size:g}",
                function_name='init_interp_GRIDMAP')
        cell = np.floor((col - lo) * (1.0 + 1.0e-9) / bin_size).astype(int)
        if cell.max() >= n_bin:
            raise SNANAError(f"{where}: {name} does not lie on a uniform grid",
                             function_name='init_interp_GRIDMAP')
        axis = lo + bin_size * np.arange(n_bin)
        axis[-1] = hi
        axes.append(axis)
        cells.append(cell)

    shape = tuple(len(axis) for axis in axes)
    flat = np.ravel_multi_index(tuple(cells), shape)
    # the last row on a grid point wins (INVMAP, sntools_gridmap.c:477)
    _, from_end = np.unique(flat[::-1], return_index=True)
    keep = len(flat) - 1 - from_end
    values = np.full(int(np.prod(shape)), np.nan)
    values[flat[keep]] = eff[keep]
    n_filled = len(keep)
    if n_filled < values.size:
        raise SNANAError(f"{where}: the rows fill {n_filled} of the "
                         f"{values.size} grid points",
                         function_name='init_interp_GRIDMAP')
    return axes, values.reshape(shape)


def read_specid_zhost(config, spellings):
    """``(specid_maps, specid_flag, zhost_maps, zhost_flag)`` from the
    SEARCHEFF SPEC and zHOST files, after checking that APPLY_SEARCHEFF_OPT
    asks only for stages they can pass (init_SEARCHEFF,
    sntools_trigger.c:119-125). A map variable spelling a population
    parameter, in any of the ways ``spellings`` holds, is named as the
    simulation names it."""
    sc = config.survey
    loaded = {}
    for maptype, named in (('SPEC', sc.searcheff_spec_eff_file),
                           ('zHOST', sc.searcheff_zhost_eff_file)):
        maps, flag = read_searcheff_maps(named, maptype,
                                         config.run.path_user_input)
        for m in maps:
            m.varnames = [spellings.get(v.lower(), v) for v in m.varnames]
        if flag == FLAG_NORMAL:
            log.line(logger, f"Loaded SEARCHEFF_{maptype} from "
                     f"{Path(named).name} ({len(maps)} maps)")
        loaded[maptype] = (maps, flag)
    (specid_maps, specid_flag), (zhost_maps, zhost_flag) = (
        loaded['SPEC'], loaded['zHOST'])
    check_applymask_searcheff(sc.apply_searcheff_opt, specid_maps,
                              specid_flag, zhost_maps, zhost_flag)
    return specid_maps, specid_flag, zhost_maps, zhost_flag


def check_applymask_searcheff(apply_opt: int, specid_maps, specid_flag: int,
                              zhost_maps, zhost_flag: int) -> None:
    """Abort when APPLY_SEARCHEFF_OPT requires a stage the SPEC and zHOST
    maps cannot pass (check_APPLYMASK_SEARCHEFF, sntools_trigger.c:133-194).

    A map type counts as present when its file holds a map or its key is
    NONE or ONE (NONZERO_SEARCHEFF, :1297-1309, :1410), so ZERO, or a file
    with no map, cannot pass. The zHOST stage also needs SEARCHEFF_SPEC_FILE
    to be ZERO, NONE, ONE or a file with a map (LSPECID, :148).
    """
    nonzero_specid = specid_flag == FLAG_ALL_PASS or bool(specid_maps)
    nonzero_zhost = zhost_flag == FLAG_ALL_PASS or bool(zhost_maps)
    lspecid = specid_flag != FLAG_NORMAL or bool(specid_maps)

    allowed = APPLYMASK_SEARCHEFF_PIPELINE
    if nonzero_specid:
        allowed |= APPLYMASK_SEARCHEFF_SPECID | APPLYMASK_SEARCHEFF_zSPEC
    if nonzero_zhost and lspecid:
        allowed |= APPLYMASK_SEARCHEFF_zHOST | APPLYMASK_SEARCHEFF_zSPEC

    if (apply_opt & allowed) != apply_opt:
        raise SNANAError(
            f"Invalid APPLY_SEARCHEFF_OPT: {apply_opt}; the simulation can "
            f"produce SEARCHEFF mask {allowed}. Bit "
            f"{APPLYMASK_SEARCHEFF_PIPELINE} is pipeline detection; "
            f"{APPLYMASK_SEARCHEFF_SPECID}, spectroscopic confirmation, needs "
            f"a SEARCHEFF_SPEC_FILE with a map, or NONE or ONE; "
            f"{APPLYMASK_SEARCHEFF_zHOST}, a host redshift, needs the same of "
            f"SEARCHEFF_zHOST_FILE and a SEARCHEFF_SPEC_FILE that is not an "
            f"empty map file; {APPLYMASK_SEARCHEFF_zSPEC}, a redshift from "
            f"either, needs either.",
            function_name='check_APPLYMASK_SEARCHEFF')


def split_fieldlist(field_str: str) -> List[str]:
    """Field names from a FIELDLIST string; SNANA separates them by + or ,."""
    return [f.strip() for f in str(field_str).replace(',', '+').split('+') if f.strip()]


def field_name_mask(fieldlist, simlib_field_names, context: str) -> np.ndarray:
    """Bool mask over SIMLIB field entries for `fieldlist`, aborting on a name
    the SIMLIB does not define rather than matching nothing."""
    unknown = [fn for fn in fieldlist if fn not in set(simlib_field_names)]
    if unknown:
        raise SNANAError(
            f"{context} names fields not present in SIMLIB: {unknown}. "
            f"Available SIMLIB field names: {sorted(set(simlib_field_names))}")
    return np.isin(np.asarray(simlib_field_names), list(fieldlist))


def resolve_fieldlist_indices(maps, simlib_field_names):
    """Resolve each map's FIELDLIST string to a per-SIMLIB-entry bool mask.

    A SIMLIB field matches when its name is a substring of the FIELDLIST
    string, and 'ALL' matches every field (MATCH_SEARCHEFF_FIELD,
    sntools_trigger.c:3266-3306). ``maps`` (efficiency or PHOTPROB maps) is
    mutated in place: each gets `fieldlist_mask` populated, None for 'ALL'.
    ``simlib_field_names`` is one str per SIMLIB field entry; the per-SN
    `field_idx` returned by the scheduler indexes into this list, so
    `fieldlist_mask[field_idx]` gives the per-SN match.
    """
    for m in maps:
        m.fieldlist_mask = (
            None if m.fieldlist == _FIELDLIST_ALL
            else np.array([name in m.fieldlist for name in simlib_field_names]))


def is_filter_variable(vname: str) -> bool:
    """Whether a map variable is a band magnitude or colour rather than a HOSTLIB column (assign_MAP_VARNAME_FILTERS)."""
    vu = vname.upper()
    return (len(vname) == 1 or '+' in vname or '-' in vname
            or vname.endswith('_obs')
            or vu.startswith(('PEAKMAG_', 'HOSTMAG_', 'SBMAG_')))


def _is_magshift_var(vname: str) -> bool:
    """True when MAGSHIFT_<MAPTYPE>EFF applies to this variable.

    Mirrors SNANA's FLAG_EFFMAP_MAG classification
    (sntools_trigger.c:1748,3171+): single-char filter mags, PEAKMAG_<f>,
    HOSTMAG_<f>. Excludes:
    - 3-char colours 'a-b' (SNANA skips MAGSHIFT in the colour branch)
    - SBMAG_<f> (SNANA passes MAGSHIFT=0 for SBMAG, line 3116)
    - arbitrary HOSTLIB passthrough columns (e.g. 'r_obs_auto') which
      aren't flagged IS_MAG at SNANA map parse time.
    """
    if (len(vname) == 3 and vname[1] == '-'
            and vname[0].isalpha() and vname[2].isalpha()):
        return False
    vu = vname.upper()
    if vu.startswith('SBMAG_'):
        return False
    if vu.startswith('HOSTMAG_') or vu.startswith('PEAKMAG_'):
        return True
    if len(vname) == 1 and vname.isalpha():
        return True
    if '+' in vname:
        parts = [p for p in vname.split('+') if p]
        if parts:
            return all(_is_magshift_var(p) for p in parts)
    return False


def _resolve_peakmag(filt_char: str, sn_vars: Dict) -> Optional[jnp.ndarray]:
    """Resolve a single-char filter peak magnitude from `sn_vars`.

    Returns a JAX array or None if no matching key is found. Lookups are on
    static dict keys, so this is trace-safe when called from inside @jax.jit.
    """
    for key in (f'peakmag_{filt_char}', f'PEAKMAG_{filt_char}'):
        if key in sn_vars:
            return jnp.asarray(sn_vars[key])
    prefix = 'peakmag_'
    for key, val in sn_vars.items():
        if key.startswith(prefix) and '-' in key[len(prefix):]:
            if filter_band(key) == filt_char:
                return jnp.asarray(val)
    return None


def _resolve_variable(varname: str, sn_vars: Dict,
                      zp_fluxcal: float) -> jnp.ndarray:
    """Resolve a SEARCHEFF map variable into a per-SN JAX array.

    Supports direct lookup, multi-filter OR, colour (X-Y), single-char peakmag,
    HOSTMAG_<filt>, SBMAG_<filt>, and host_<name>. All ops are jnp primitives
    so the result traces correctly when called from inside `@jax.jit`.
    """
    vname_upper = varname.upper()

    if varname in sn_vars:
        return jnp.asarray(sn_vars[varname])
    if vname_upper in sn_vars:
        return jnp.asarray(sn_vars[vname_upper])

    standard_map = {
        'REDSHIFT': 'sim_redshift_cmb',
        'PEAKMJD': 'peakmjd',
        'DTSEASON_PEAK': 'dtseason_peak',
        'DTPEAK': 'dtpeak',
        'LOGMASS': 'host_LOGMASS',
        'SALT2MB': 'SALT2mB',
    }
    if vname_upper in standard_map:
        key = standard_map[vname_upper]
        if key in sn_vars:
            return jnp.asarray(sn_vars[key])

    # mean of the valid (0 < mag < 40) mags, else -9 (sntools_trigger.c:3171-3189)
    if '+' in varname:
        parts = [p for p in varname.split('+') if p]
        if parts:
            mags = [_resolve_variable(p, sn_vars, zp_fluxcal) for p in parts]
            stack = jnp.stack(mags, axis=0)
            valid = (stack > 0) & (stack < 40)
            n_valid = valid.sum(axis=0)
            return jnp.where(
                n_valid > 0,
                jnp.where(valid, stack, 0.0).sum(axis=0) / jnp.maximum(n_valid, 1),
                -9.0,
            )

    if '-' in varname and len(varname.split('-')) == 2:
        a, b = varname.split('-')
        mag_a = _resolve_peakmag(a, sn_vars)
        mag_b = _resolve_peakmag(b, sn_vars)
        if mag_a is not None and mag_b is not None:
            return mag_a - mag_b

    if len(varname) == 1 and varname.isalpha():
        mag = _resolve_peakmag(varname, sn_vars)
        if mag is not None:
            return mag

    if vname_upper.startswith('HOSTMAG_'):
        filt_char = varname[len('HOSTMAG_'):]
        for key in sn_vars:
            if key.startswith('host_') and key.endswith('_obs') and filt_char in key:
                return jnp.asarray(sn_vars[key])
        for key_pattern in (f'host_MAG_{filt_char}', f'host_{filt_char}_obs'):
            if key_pattern in sn_vars:
                return jnp.asarray(sn_vars[key_pattern])

    if vname_upper.startswith('SBMAG_'):
        filt_char = varname[len('SBMAG_'):]
        for key in sn_vars:
            if key.startswith('host_SB_FLUXCAL') and filt_char in key:
                fluxcal = jnp.asarray(sn_vars[key])
                safe_flux = jnp.where(fluxcal > 0, fluxcal, 1e-30)
                return -2.5 * jnp.log10(safe_flux) + zp_fluxcal

    host_key = f'host_{varname}'
    if host_key in sn_vars:
        return jnp.asarray(sn_vars[host_key])

    varname_lower = varname.lower()
    for key in sn_vars:
        if key.lower() == varname_lower or key.lower() == f'host_{varname_lower}':
            return jnp.asarray(sn_vars[key])

    raise SNANAError(
        f"SEARCHEFF variable '{varname}' is unknown: neither a per-event "
        f"quantity nor a HOSTLIB column. Known: {sorted(sn_vars.keys())}",
        function_name='_resolve_variable')


def prepare_searcheff(maps, sn_vars, zp_fluxcal: float, var_shifts: Dict,
                      magshift: float):
    """Resolve per-SN inputs (points, field/PEAKMJD match) for JIT evaluation.

    Pure JAX: all operations are jnp primitives, so this function traces
    cleanly when called from inside `@jax.jit`. String lookups happen on
    static map metadata at Python time (map.varnames, fieldlist_mask, etc.),
    so these don't enter the trace.

    Each of ``maps`` must have `fieldlist_mask` populated by
    `resolve_fieldlist_indices` (unless its fieldlist is None). ``sn_vars``
    is the per-SN variable dict; it must include 'field_idx' (int32), which
    also sets the number of SNe, and 'PEAKMJD'. ``var_shifts`` is a
    ``{varname: shift}`` added to resolved values (SNANA
    sntools_trigger.c:2952); ``magshift`` a scalar added to mag-type
    variables (sntools_trigger.c:3186).

    Returns ``(prepared, n_matches_per_sn)``: ``prepared`` is a list of
    per-map dicts (points/grid_axes/grid_values/opt_extrap/applies_to/
    require) consumed by `evaluate_searcheff`; ``n_matches_per_sn`` is
    (n_sne,) jnp.int32, the number of maps that apply to each SN, used by
    the Python wrapper to enforce OPT_FIELDMATCH_REQUIRE post-JIT (raises
    only outside the trace).
    """
    field_idx = sn_vars['field_idx']
    peakmjds = sn_vars['PEAKMJD']
    n_sne = field_idx.shape[0]

    prepared = []
    n_matches_per_sn = jnp.zeros(n_sne, dtype=jnp.int32)
    for m in maps:
        if m.fieldlist_mask is not None:
            field_match = jnp.asarray(m.fieldlist_mask)[field_idx]
        else:
            field_match = jnp.ones(n_sne, dtype=bool)

        # inclusive both ends (sntools_trigger.c:2936)
        if m.peakmjd_range is not None:
            mjd_lo, mjd_hi = m.peakmjd_range
            mjd_match = (peakmjds >= mjd_lo) & (peakmjds <= mjd_hi)
        else:
            mjd_match = jnp.ones(n_sne, dtype=bool)

        applies_to = field_match & mjd_match
        n_matches_per_sn = n_matches_per_sn + applies_to.astype(jnp.int32)

        resolved = []
        for vname in m.varnames:
            arr = _resolve_variable(vname, sn_vars, zp_fluxcal).astype(jnp.float32)
            if vname in var_shifts:
                arr = arr + jnp.float32(var_shifts[vname])
            if magshift != 0.0 and _is_magshift_var(vname):
                arr = arr + jnp.float32(magshift)
            resolved.append(arr)
        points = jnp.stack(resolved, axis=1)

        prepared.append({
            'points': points,
            'grid_axes': [jnp.asarray(ax, dtype=jnp.float32) for ax in m.grid_axes],
            'grid_values': jnp.asarray(m.grid_values, dtype=jnp.float32),
            'opt_extrap': m.opt_extrap,
            'applies_to': applies_to,
            'require': m.require,
        })

    return prepared, n_matches_per_sn


def eff_outside_window(eff):
    """Per SN, whether an EFF left by `evaluate_searcheff` is one gen_searcheff_map aborts on."""
    return (eff < 0.0) | (eff > EFF_ROUNDOFF_MAX)


def evaluate_searcheff(prepared_maps, flag, n_sne, rng_key, sn_idx,
                       spec_scale=1.0, require_key=None):
    """Evaluate prepared SEARCHEFF maps on per-SN inputs. JIT-compatible.

    Every matched map, REQUIRE included, folds into the OR combination
    EFF = 1 - prod(1 - eff_imap); a REQUIRE map additionally gets its own
    independent draw against its eff (sntools_trigger.c:2971-2978). An SN
    matched by no map has every eff zeroed, so EFF=0; the caller aborts on
    such an SN unless the map file sets OPT_FIELDMATCH_REQUIRE: 0. A
    combined EFF just outside [0, 1], within the round-off window
    (EFF_ROUNDOFF_MIN, EFF_ROUNDOFF_MAX), is set to 0 or 0.99999 as in
    gen_searcheff_map (sntools_trigger.c:3011-3019); one beyond the window
    is left for the caller to abort on (`eff_outside_window`).

    ``prepared_maps`` is the output of prepare_searcheff; ``flag`` is
    FLAG_NORMAL / FLAG_ALL_PASS / FLAG_ALL_FAIL; ``sn_idx`` the (n_sne,)
    per-SN index each random draw folds in; ``spec_scale`` a
    multiplicative scale on efficiency, applied after the round-off
    window and clipped to [0, 1]. ``require_key``, where given, replaces
    the key the REQUIRE draws come from, so that two map sets can share
    them as SNANA's SPECID and zHOST maps do under REFAC_SEARCHEFF_MAP 1
    (snlc_sim.c:24771). Returns ``(eff, passed, unscaled_eff)``, each
    (n_sne,) jnp arrays, ``unscaled_eff`` being the EFF before
    ``spec_scale``.
    """
    if flag == FLAG_ALL_PASS:
        return jnp.ones(n_sne), jnp.ones(n_sne, dtype=bool), jnp.ones(n_sne)
    if flag == FLAG_ALL_FAIL:
        return (jnp.zeros(n_sne), jnp.zeros(n_sne, dtype=bool),
                jnp.zeros(n_sne))

    all_effs = []
    require_maps = []
    for imap, pm in enumerate(prepared_maps):
        eff = interp_nd_regular(
            pm['points'], pm['grid_axes'], pm['grid_values'],
            opt_extrap=pm['opt_extrap'])
        eff = jnp.where(pm['applies_to'], eff, 0.0)
        all_effs.append(eff)
        if pm['require']:
            require_maps.append((imap, eff, pm['applies_to']))

    prob_all_fail = jnp.ones(n_sne)
    for eff in all_effs:
        prob_all_fail = prob_all_fail * (1.0 - eff)
    combined_eff = 1.0 - prob_all_fail
    combined_eff = jnp.where(
        (combined_eff < 0.0) & (combined_eff > EFF_ROUNDOFF_MIN), 0.0,
        jnp.where((combined_eff > 1.0) & (combined_eff < EFF_ROUNDOFF_MAX),
                  0.99999, combined_eff))
    unscaled_eff = combined_eff

    combined_eff = jnp.where(
        spec_scale == 1.0, combined_eff,
        jnp.clip(combined_eff * spec_scale, 0.0, 1.0))

    key1, key2 = jax.random.split(rng_key)
    if require_key is not None:
        key2 = require_key
    keys1 = jax.vmap(lambda i: jax.random.fold_in(key1, i))(sn_idx)
    rand_combined = jax.vmap(lambda k: jax.random.uniform(k))(keys1)
    passed = rand_combined < combined_eff

    # a REQUIRE draw per map index, only where it matches (sntools_trigger.c:2976)
    for imap, eff, applies_to in require_maps:
        key_req = jax.random.fold_in(key2, imap)
        keys_req = jax.vmap(lambda k: jax.random.fold_in(key_req, k))(sn_idx)
        rand_req = jax.vmap(lambda k: jax.random.uniform(k))(keys_req)
        passed = passed & (~applies_to | (rand_req < eff))

    return combined_eff, passed, unscaled_eff


jax.tree_util.register_dataclass(
    SearchEffMap,
    data_fields=[
        'grid_axes',
        'grid_values',
        'fieldlist_mask',
        'peakmjd_range',
    ],
    meta_fields=[
        'varnames',
        'require',
        'fieldlist',
        'opt_extrap',
        'opt_fieldmatch_require',
    ],
)
