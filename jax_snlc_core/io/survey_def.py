"""SURVEY.DEF readers, shared by the sim, the fitter and SALT2mu.

The file carries several record types on its own line prefixes: ``SURVEY:``
lines name a survey, its integer ID (SNANA GENLC.IDSURVEY) and optionally a
``geo:`` token, while ``FIELD:`` lines name a field and the survey it belongs
to. One tokeniser serves them all, so the file's comment and whitespace rules
are written once. Callers differ in what an unknown survey means, so these
return the mappings and leave that policy to them.
"""

import functools
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from jax_snlc_core.io.fits_path import open_text
from jax_snlc_core.utils import SNANAError, sndata_root


def _survey_def_path(path: Optional[str]) -> str:
    """``path``, or $SNDATA_ROOT's SURVEY.DEF."""
    if path is None:
        return str(sndata_root() / 'SURVEY.DEF')
    return path


def _rows(path: Optional[str], prefix: str) -> Iterator[Sequence[str]]:
    """Tokenised SURVEY.DEF rows carrying ``prefix``, comments stripped."""
    path = _survey_def_path(path)
    if not Path(path).is_file():
        raise SNANAError(f"Unable to open file with mode = rt. Check '{path}'",
                         function_name="read_SURVEYDEF")
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
    for toks in _rows(path, 'SURVEY:'):
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
    for toks in _rows(path, 'SURVEY:'):
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
    return [toks[1] for toks in _rows(path, 'FIELD:')
            if len(toks) >= 4 and toks[3] == name]
