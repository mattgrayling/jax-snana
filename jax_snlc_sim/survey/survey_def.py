"""The simulator's view of SURVEY.DEF: survey identity and observatory site.

The file is read by ``jax_snlc_core.io.survey_def``; this module turns what it
returns into the identity and the site the simulation needs.

Every ``SURVEY:`` line carries a name and an integer ID, which together are the
identity of the survey being simulated (SNANA GENLC.SURVEY_NAME and
GENLC.IDSURVEY). A line may additionally carry a ``geo:lat,lon,alt`` token that
sets telescope latitude/longitude (deg) and altitude (m) for the DCR subsystem,
mirroring SNANA read_SURVEYDEF / parse_geoSURVEYDEF (sntools.c); site
temperature, pressure and precipitable water vapour follow altitude via the same
formulae SNANA uses (sntools.c:3692).
"""

from jax_snlc_core.utils.error_handling import SNANAError
import math
import os
from dataclasses import dataclass
from typing import Dict, Optional

from jax_snlc_core.io.fits_path import open_text, resolve_fits_path
from jax_snlc_core.io.survey_def import load_survey_geo, load_survey_ids


@dataclass
class SiteGeo:
    """Observatory geo-coordinates and altitude-derived atmosphere params."""
    geo_lat: float          # degrees
    geo_lon: float          # degrees
    geo_alt: float          # metres
    temperature: float      # Celsius
    pressure: float         # mm Hg
    pwv: float              # mm Hg


def _pressure_atmos(alt_m: float) -> float:
    """Atmospheric pressure (mm Hg) at altitude alt_m (pressure_atmos, sntools.c:135)."""
    return 760.0 * (1.0 - alt_m * 2.25577e-5) ** 5.25588


def _site_params(alt_m: float) -> tuple:
    """T (C), BP (mm Hg), PWV (mm Hg) from altitude (sntools.c:3692)."""
    temperature = 20.0 - 0.006 * alt_m
    pressure = _pressure_atmos(alt_m)
    pwv = 3.0 * math.exp(-alt_m / 2000.0)
    return temperature, pressure, pwv


def parse_survey_def(path: Optional[str] = None) -> Dict[str, SiteGeo]:
    """Parse ``$SNDATA_ROOT/SURVEY.DEF`` into a {survey_name: SiteGeo} map.

    Only surveys carrying a ``geo:`` token are included; the atmosphere the
    site sees is derived from its altitude.
    """
    out: Dict[str, SiteGeo] = {}
    for name, (lat, lon, alt) in load_survey_geo(path).items():
        out[name] = SiteGeo(lat, lon, alt, *_site_params(alt))
    return out


def survey_id(name: str) -> int:
    """SURVEY.DEF integer ID for ``name`` (get_IDSURVEY, snlc_sim.c:18574).

    Covers every survey, unlike :func:`parse_survey_def`, which keeps only those
    carrying a ``geo:`` token. SNANA aborts on an unmatched name
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
