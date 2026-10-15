"""Reader for SEARCHEFF_PIPELINE_EFF_FILE.

The file carries three kinds of block, and SNANA reads them in one pass
(init_SEARCHEFF, sntools_trigger.c:330-334): the detection curves that give
a per-observation efficiency against SNR, |SNR| or observed magnitude
(readMap_SEARCHEFF_DETECT, :358-423), the PHOTPROB maps
(readMap_SEARCHEFF_PHOTPROB), and the PHOTFLAG_DETECT / PHOTFLAG_TRIGGER
bits, each of which the file supplies only while the input file has left it
at zero (:321-325).
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from jax_snlc_core.io.docana import check_file_docana, skip_docana
from jax_snlc_core.io.fits_path import open_text
from jax_snlc_core.utils.error_handling import SNANAError
from jax_snlc_sim.survey.searcheff_photprob import (
    PhotoprobMap, VARNAME_TO_IVARABS, build_photprob_map)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectionCurve:
    """One detection map: an efficiency read against SNR, |SNR| (``use_abs``)
    or observed magnitude (``is_mag``), for the bands in ``filterlist`` and
    the fields in ``fieldlist`` ('ALL' matches every entry)."""

    filterlist: str
    fieldlist: str
    vals: np.ndarray
    effs: np.ndarray
    use_abs: bool
    is_mag: bool


@dataclass(frozen=True)
class PipelineEffFile:
    """What one SEARCHEFF_PIPELINE_EFF_FILE holds."""

    curves: List[DetectionCurve]
    photprob_maps: List[PhotoprobMap]
    photflag_detect: int
    photflag_trigger: int


def read_pipeline_eff_file(filepath: str, photflag_detect: int,
                           photflag_trigger: int) -> PipelineEffFile:
    """Read the file once. ``photflag_detect``/``photflag_trigger`` are the
    input file's own values, which the file fills in only while they are 0."""
    expanded = Path(os.path.expandvars(filepath))
    check_file_docana(expanded)

    curves: List[DetectionCurve] = []
    photprob_maps: List[PhotoprobMap] = []
    detect_map = None
    named_detect = False    # inside MAPNAME_DETECT ... ENDMAP
    photprob = None
    photprob_rows: List[List[float]] = []

    def flush_detect():
        nonlocal detect_map
        if detect_map is not None and detect_map['vals']:
            curves.append(DetectionCurve(
                filterlist=detect_map['filterlist'],
                fieldlist=detect_map['fieldlist'],
                vals=np.array(detect_map['vals']),
                effs=np.array(detect_map['effs']),
                use_abs=detect_map['use_abs'],
                is_mag=detect_map['is_mag']))
        detect_map = None

    with open_text(expanded) as f:
        for raw_line in skip_docana(f):
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            key = parts[0]

            if key == 'PHOTFLAG_DETECT:' and not photflag_detect:
                photflag_detect = int(parts[1])
                continue
            if key == 'PHOTFLAG_TRIGGER:' and not photflag_trigger:
                photflag_trigger = int(parts[1])
                continue
            if key in ('PHOTFLAG_DETECT:', 'PHOTFLAG_TRIGGER:'):
                continue

            if key == 'MAPNAME_PHOTPROB:':
                flush_detect()
                photprob = {
                    'name': parts[1] if len(parts) > 1 else 'UNNAMED',
                    'fieldlist': 'ALL', 'filterlist': 'ALL', 'nvar_tot': 0,
                    'require_detection': False, 'cutval': -999.0,
                    'reduced_corr': 0.0, 'varnames': [], 'ivarabs': [],
                    'nvar_map': 0, 'nfun_cdf': 0,
                }
                photprob_rows = []
                continue

            if key == 'ENDMAP:':
                if photprob is not None:
                    photprob_maps.append(build_photprob_map(photprob,
                                                            photprob_rows))
                    photprob, photprob_rows = None, []
                else:
                    flush_detect()
                    named_detect = False
                continue

            if photprob is not None:
                _read_photprob_line(photprob, photprob_rows, key, parts)
                continue

            if key == 'MAPNAME_DETECT:':
                flush_detect()
                detect_map = _new_detect_map()
                named_detect = True
                continue
            if key in ('FILTER:', 'BAND:'):
                if not (named_detect and detect_map is not None):
                    flush_detect()
                    detect_map = _new_detect_map()
                detect_map['filterlist'] = parts[1]
                continue
            if key == 'FIELD:' and detect_map is not None:
                detect_map['fieldlist'] = parts[1]
                continue
            if detect_map is not None and len(parts) >= 3:
                if key == 'ABS(SNR):':
                    detect_map['use_abs'] = True
                elif key == 'MAG:':
                    detect_map['is_mag'] = True
                elif key != 'SNR:':
                    continue
                detect_map['vals'].append(float(parts[1]))
                detect_map['effs'].append(float(parts[2]))
    flush_detect()

    return PipelineEffFile(curves=curves, photprob_maps=photprob_maps,
                           photflag_detect=photflag_detect,
                           photflag_trigger=photflag_trigger)


def _new_detect_map() -> dict:
    return {'filterlist': '', 'fieldlist': 'ALL', 'vals': [], 'effs': [],
            'use_abs': False, 'is_mag': False}


def _read_photprob_line(photprob: dict, rows: List[List[float]], key: str,
                        parts: List[str]) -> None:
    """One line inside a MAPNAME_PHOTPROB block."""
    if key in ('FIELD:', 'FIELDLIST:'):
        photprob['fieldlist'] = parts[1] if len(parts) > 1 else 'ALL'
    elif key in ('FILTER:', 'BAND:'):
        photprob['filterlist'] = parts[1] if len(parts) > 1 else 'ALL'
    elif key == 'NVAR:':
        photprob['nvar_tot'] = int(parts[1])
    elif key == 'REQUIRE_DETECTION:':
        photprob['require_detection'] = int(parts[1]) != 0
    elif key == 'CUTVAL:':
        photprob['cutval'] = float(parts[1])
    elif key == 'REDUCED_CORR:':
        photprob['reduced_corr'] = float(parts[1])
    elif key == 'VARNAMES:':
        varnames = []
        for token in parts[1:]:
            if token == 'PHOTPROB_WGTLIST':
                break
            if token.upper() not in VARNAME_TO_IVARABS:
                raise SNANAError(
                    f"Invalid PHOTPROB variable '{token}'. "
                    f"Valid: {list(VARNAME_TO_IVARABS.keys())}",
                    function_name='read_pipeline_eff_file')
            varnames.append(token.upper())
        photprob['varnames'] = varnames
        photprob['ivarabs'] = [VARNAME_TO_IVARABS[v] for v in varnames]
        photprob['nvar_map'] = len(varnames)
        photprob['nfun_cdf'] = photprob['nvar_tot'] - photprob['nvar_map']
    elif key == 'ROW:':
        rows.append([float(x) for x in parts[1:]])
