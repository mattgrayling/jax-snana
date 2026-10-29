"""Fit input: an SNANA namelist or YAML file, command-line overrides, or
Python settings, turned into one checked FitConfig.

The SNANA route takes three steps, in order:

    read_fit_input(path) -> the input's keys, aliases resolved
    apply_overrides      -> raw -> raw, replaying command-line tokens as input lines
    build(raw)           -> a FitConfig, in one interpretation pass

`parse_fit_input_file` runs them through `build_file_config`, which also
reports keys the build never read. A YAML input takes named settings and
goes through `build_config` instead.

&SNLCINP and &FITINP share no key names, so both sections flatten into one dict.
YAML inputs use the same section names and flatten identically.

`build` resolves paths and values but reads no data files: `known_keys`
runs it on an empty input to learn which tokens are keys.

Python and command-line settings go through `build_config` directly. Every
route produces the same FitConfig, and model construction and scientific
validation do not depend on which syntax supplied it.
"""

import argparse
import copy
import dataclasses
import logging
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import (Any, Dict, List, Optional, Sequence, Tuple, Union, get_origin,
                    get_type_hints)

import yaml

from jax_snlc_core import logging as log
from jax_snlc_core.dust import OPT_MWCOLORLAW_NAMES
from jax_snlc_core.input_settings import (UsedTrackingDict, add_settings, bootstrap_settings,
                                          build_dataclass, convert, dataclass_settings,
                                          derived_changes, mapping_tokens, merge_settings,
                                          parsed_settings, supplied_values, value_type)
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.models import (MODEL_REGISTRY, infer_model_type_from_path, model_input_keys,
                                  parse_model_input_value)
from jax_snlc_core.models.registry import model_input_declarations
from jax_snlc_core.peak_time import (OPTMASK_SETPKMJD_BAZIN, OPTMASK_SETPKMJD_FLUXMAX,
                                     OPTMASK_SETPKMJD_FLUXMAX2, OPTMASK_SETPKMJD_FLUXMAX3,
                                     OPTMASK_SETPKMJD_NOABORT, OPTMASK_SETPKMJD_TRIGGER)
from jax_snlc_core.snana_files import is_ignorefile
from jax_snlc_core.utils import CUTVAL_OPEN

from .config import (PRIOR_ZERRSCALE_UNSET, SN_CUT_BAND_PREFIXES, SN_CUT_NAMES, EarlyLC, Fit,
                     FitConfig, RequireEpochs)
from .parameters import parse_prior

RECORD_CLASSES = get_type_hints(FitConfig)
FIELD_RECORD = {name: record for record, owner in RECORD_CLASSES.items()
                for name in get_type_hints(owner)}

logger = logging.getLogger(__name__)


# Key families read by iterating the dict, not by lookup.
_OPEN_KEY_PREFIXES = ('CUTWIN_', 'INIVAL_', 'INISTP_', 'FITWIN_')

# (canonical, *alternates): EQUIVALENCE pairs (snlc_fit.F90:747-748) and SNANA spellings
_KEY_ALIAS_GROUPS = (
    ('FITWIN_MJD', 'MJD_REJECT'),                    # EQUIVALENCE
    ('FITWIN_TREST', 'TREST_REJECT'),                # EQUIVALENCE
    ('OPT_COVAR_FLUX', 'OPT_COVAR'),                 # snlc_fit.F90:570
    ('OPT_COVAR_MWXTERR', 'OPT_XTMW_ERR'),           # snlc_fit.F90:573
    ('PRIOR_SHAPE_RANGE', 'PRIOR_LUMIPAR_RANGE'),
    ('PRIOR_SHAPE_SIGMA', 'PRIOR_LUMIPAR_SIGMA'),
    ('PRIOR_AV_EXP', 'PRIOR_AVEXP'),
    ('PRIOR_AV_WGT', 'PRIOR_AVWGT'),
    ('PRIOR_AV_RES', 'PRIOR_AVRES'),
    ('OPT_SETPKMJD', 'OPT_SETPEAKMJD'),
    # ORed into one flag (snana.F90:17789)
    ('USE_SNHOST_ZPHOT', 'USE_HOSTGAL_PHOTOZ'),
)
_KEY_ALIASES = {alt: canon for canon, *alts in _KEY_ALIAS_GROUPS for alt in alts}
for _owner in RECORD_CLASSES.values():
    for _name, _aliases in getattr(_owner, 'input_names', {}).items():
        for _alias in _aliases:
            _KEY_ALIASES[_alias] = _name.upper()

# any filename is_ignorefile treats as none
_IGNORE_FILENAMES = object()

def _refused(reason, *keys, **inactive):
    """Unsupported keys, with optional values that leave them inactive."""
    return {**dict.fromkeys(keys, (reason, None)),
            **{key: (reason, value) for key, value in inactive.items()}}


_REJECTED_OPTIONS = {
    **_refused("use MINOS='z' or MINOS='all' to select profile uncertainties",
        'MINOS_METHOD', 'MINOS_PARAMETERS', USE_MINOS=False),
    **_refused('appending user-defined columns to output tables is not supported',
        'SNTABLE_APPEND_VARNAME', 'SNTABLE_APPEND_VALUE'),
    **_refused('remapping filter labels in output tables is not supported',
        'SNTABLE_FILTER_REMAP'),
    **_refused('ROOT output is not supported',
        ROOTFILE_OUT=_IGNORE_FILENAMES),
    **_refused('MARZ output is not supported',
        MARZFILE_OUT=_IGNORE_FILENAMES),
    **_refused('user-tag files are not supported',
        USERTAGS_FILE=_IGNORE_FILENAMES),
    **_refused('data reformatting is not supported by snlc_fit',
        'VERSION_REFORMAT_FITS', 'VERSION_REFORMAT_TEXT', 'OPT_REFORMAT_FITS',
        'OPT_REFORMAT_TEXT', 'OPT_REFORMAT_SPECTRA', 'REFORMAT_KEYS',
        'ZP_FLUXCAL_REFORMAT'),
    **_refused('SNANA-specific data-structure configuration is not supported',
        'REFAC_DATA_FLAG'),
    **_refused('writing a SIMLIB from light-curve data is not supported',
        'SIMLIB_OUT', 'SIMLIB_OUTFILE', 'SIMLIB_ZPERR_LIST', 'OPT_SIMLIB_OUT',
        'SIMLIB_OUT_TMINFIX'),
    **_refused('SNANA plotting and contour controls are not supported',
        'NCCID_PLOT', 'SNCCID_PLOT', 'MJDPERIOD_PLOT', 'MJDSHIFT_PLOT',
        'DTOBS_MODEL_PLOT', 'SCALE_PLOTCHI2', 'CONTOUR_LIST'),
    **_refused('MINUIT configuration is not supported',
        'MINUIT_PRINT_LEVEL', 'MINOS_ERR_DLMAG', 'MINOS_ERR_REQALL0',
        'MINOS_ERR_REQALL1', 'OPT_MNSTAT_COV', USE_MIGRAD=False,
        USE_SIMPLEX=False),
    **_refused('marginalised parameter-PDF grids are not supported',
        'MAX_INTEGPDF', 'FITWIN_NOMARG_PROB', NGRID_PDF=0.0, NSIGMA_PDF=4.0,
        VARLIST_PDF='ALL'),
    **_refused('MCMC photometric-redshift fitting is not supported',
        'NEVAL_MCMC_PHOTOZ', 'PARLIST_MCMC_PHOTOZ'),
    **_refused('SNANA diagnostic output and tracing controls are not supported',
        'DEBUG_FLAG', 'LTRACE', 'NFIT_VERBOSE', 'FORCE_STDOUT_BATCH', 'OPT_YAML',
        'LDMP_SNANA_VERSION', 'LDMP_SATURATE', 'LDMP_PHOTOZ_DROPFILTER', 'LTEST_KCOR',
        'LTEST_INTERP', 'LTEST_MAG', 'LTEST_U3BAND', 'CCID_DUMP_CHI2_MATRIX',
        'FILTLIST_DMPFCN', 'TREST_DMPFUN', 'DUMP_STRING', 'MNFIT_PKMJD_LOGFILE',
        'SNMJD_LIST_FILE', 'SNMJD_OUT_FILE', 'OUT_EPOCH_IGNORE_FILE', LDMP_SNFAIL=False,
        FILTLIST_DMPFUN=''),
    **_refused('these diagnostic columns and files are not produced',
        'TREST_PEAKRENORM', 'OPTSIM_LCWIDTH', 'LPROB_TRUEFLUX', 'SALT2_DICTFILE'),
    **_refused('host-extinction grids are not supported',
        OPT_SNXT=1.0),
    **_refused('per-event filter-response updates are not supported',
        'FILTER_UPDATE_PATH'),
    **_refused('Landolt-system transformations are not supported',
        'LANDOLT_COLOR_SHIFT', OPT_LANDOLT=0.0),
    **_refused('the U-band BX–B correction is not supported',
        'UCOR_BXB'),
    **_refused('the K-correction uncertainty prescription is not supported',
        'OPT_KCORERR'),
    **_refused('K-corrections from colour-warped or stretched templates are not supported',
        LKCOR_AVWARP=False, LKCOR_STRETCH=False),
    **_refused('the rest-frame magnitude fitting mode',
        'FILTLIST_FITRESTMAG', 'LUMIFIX_FITRESTMAG', 'SHAPEFIX_FITRESTMAG',
        'COLORFIX_FITRESTMAG', 'LAMEXTRAP_FITRESTMAG', 'DLAMTOL_FITRESTMAG',
        'MAGREST_SHIFT_PRIMARY', 'MAGREST_SHIFT_ZP'),
    **_refused('the per-band refit consistency test',
        'FILTER_FITMAGDIF'),
    **_refused('switches that restore earlier SNANA calculations are not supported',
        'RESTORE_DES5YR', 'RESTORE_MWEBV_ERR_BUG', 'RESTORE_OVERRIDE_ZBUG'),
    **_refused('restarting fits with parameter shifts scaled by the preceding errors is not supported',
        'INIVAL_SHIFT_ERRFRAC'),
    **_refused('use HEADER_OVERRIDE_FILE',
        'VPEC_FILE'),
    **_refused('model smoothing for rest-frame filter transformations is not supported',
        'LAMREST_MODEL_SMOOTH'),
    **_refused('repeating fits in response to changes in parameter errors is not supported',
        'FRACERRDIF_REPEATFIT'),
    **_refused('priors on a second light-curve shape parameter are not supported',
        'PRIOR_SHAPE2_RANGE', 'PRIOR_SHAPE2_SIGMA'),
    **_refused('DOPRIOR_DLMAG follows from PRIOR_MUERRSCALE; set that instead',
        'DOPRIOR_DLMAG'),
    **_refused('MLCS2k2 fitting is not supported',
        'PRIOR_DELTA_PROFILE'),
    **_refused('correcting input photometry for Milky Way extinction is not supported; extinction is applied to the model',
        USE_MWCOR=False),
    **_refused('apply spectroscopic or host-redshift selection when preparing the input sample',
        LSIM_SEARCH_SPEC=False, LSIM_SEARCH_ZHOST=False),
    **_refused('MLCS2k2 fitting is not supported',
        'FITCOVAR_FILE'),
    **_refused('rescaling model correlation coefficients is not supported',
        FUDGE_COVAR=0.0),
    **_refused('selection by multi-season activity is not supported',
        'MULTISEASON_OPTMASK', 'MULTISEASON_TGAP', 'MULTISEASON_NREJECT_OUTLIER',
        'MULTISEASON_CHI2RED_ACTIVE'),
    **_refused('configuring these error-handling policies is not supported',
        'ABORT_ON_BADAVWARP', 'ABORT_ON_BADFILTER', 'ABORT_ON_BADKCOR',
        'ABORT_ON_BADSURVEY', 'ABORT_ON_BADZ', 'ABORT_ON_MARGPDF0'),
}

# Reasons are also read by the documentation generator.
_REJECTED_KEYS = {key: reason for key, (reason, _) in _REJECTED_OPTIONS.items()}

# Keys SNANA itself gives no behaviour.
_INERT_KEYS = frozenset({
    # only extends SURVEY_FILTERS (snana.F90:12530)
    'NONSURVEY_FILTERS',
    # the model interpolation option, written but never read upstream
    # (snana.F90:5697, snlc_fit.F90:7186, :11534)
    'INTERP_OPT',
    # gate commented out (snana.F90:22414)
    'ABORT_ON_DUPLMJD',
    # dead upstream: declared, defaulted, parsed, never read
    'ADD_SALT2ERR_FROMZ', 'FILTLIST_LAMSHIFT',
    # only switches on a DETNUM column (snana.F90:23310)
    'DETNUM_LIST',
    # fills ifiltdef_ignore_rest, which nothing ever reads
    'NEARFILT_IGNORE_REST',
    # parsed but never read by SNANA
    'OVERRIDE_RESTLAM_BOUNDARY',
    # never read (snlc_fit.F90:6966)
    'OPT_NEARFILT',
    # declared, initialised and never read anywhere in SNANA's sources
    'EPCUT_PSFSIG', 'EPCUT_SKYSIG',
    # only renumbers an LCPLOT index column
    'JOBSPLIT_EXTERNAL',
    # declared, never read in the fitter
    'STRETCH_TEMPLATE_FILE',
})


# a comma is a list separator only if an even number of quotes follows it
_LIST_COMMA = re.compile(r""",(?=(?:[^'"]*['"][^'"]*['"])*[^'"]*$)""")


def _parse_value(value_str: str) -> Any:
    """A key's value: a scalar, or a list when it has several comma-separated
    tokens. Quotes stripped, $ENV expanded. A blank-separated value stays one
    string here; each key's reader splits it where the key takes a list."""
    value_str = value_str.strip().rstrip(",")
    parts = _LIST_COMMA.split(value_str)
    if len(parts) > 1:
        return [os.path.expandvars(p.strip().strip("'\"")) for p in parts
                if p.strip()]
    if len(value_str) >= 2 and value_str[0] in ("'", '"') \
            and value_str[-1] == value_str[0]:
        return os.path.expandvars(value_str[1:-1])
    return os.path.expandvars(value_str)


def _strip_comment(line: str) -> str:
    """Drop a trailing ``!`` comment, leaving one inside quotes alone:
    PRIVATE_CUTWIN_STRING's veto spelling is ``VARNAME != VALUE``."""
    quote = ""
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "'\"":
            quote = ch
        elif ch == "!":
            return line[:i].strip()
    return line.strip()


def _read_line(line: str, raw: UsedTrackingDict, section: Optional[str]) -> Optional[str]:
    """Consume one namelist line into `raw`; return the section now open.

    Shared with `apply_overrides`, so a command-line override is parsed exactly
    like a line from the file.
    """
    line = _strip_comment(line)
    if not line:
        return section

    m = re.match(r"&(\w+)", line)
    if m:
        name = m.group(1).upper()
        return None if name == "END" else name
    if line.upper() == "&END":
        return None

    if section is not None and "=" in line:
        key, _, val = line.partition("=")
        key = key.strip().upper()
        raw[_KEY_ALIASES.get(key, key)] = _parse_value(val)
    return section


def _flatten_sections(node: Any) -> Dict[str, Any]:
    """Uppercase every key, expand $ENV in strings, and drop the section level."""
    if isinstance(node, dict):
        flat: Dict[str, Any] = {}
        for key, value in node.items():
            key = key.upper()
            if isinstance(value, dict) and key in ("SNLCINP", "FITINP"):
                flat.update(_flatten_sections(value))
            else:
                flat[_KEY_ALIASES.get(key, key)] = _flatten_sections(value)
        return flat
    if isinstance(node, list):
        return [_flatten_sections(v) for v in node]
    if isinstance(node, str):
        return os.path.expandvars(node)
    return node


def _snana_names(settings):
    """SNANA spellings contain capitals, including suffixes such as INIVAL_x1."""
    return bool(settings) and all(
        isinstance(key, str) and key != key.lower() for key in settings)


def read_fit_input(input_file: Union[str, Path]):
    """Read once and distinguish namelists, SNANA YAML and flat YAML.

    Namelists begin with a section header. YAML uses SNANA syntax when it
    contains SNLCINP/FITINP sections or SNANA-style capitalised input names.
    """
    path = Path(input_file)
    if not path.is_file():
        raise SNANAError(f"Input file not found: {input_file}",
                         function_name="read_fit_input")
    text = path.read_text()
    first_line = next((line.strip() for line in text.splitlines()
                       if line.strip() and not line.lstrip().startswith(('!', '#'))), '')
    if re.fullmatch(r'&(?:SNLCINP|FITINP)\s*(?:!.*)?', first_line, re.IGNORECASE):
        raw = UsedTrackingDict()
        section = None
        for line in text.splitlines():
            section = _read_line(line, raw, section)
        return 'snana', raw

    try:
        supplied = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SNANAError(f"Invalid YAML in {input_file}: {exc}") from exc
    if not isinstance(supplied, dict) or not all(isinstance(key, str) for key in supplied):
        raise SNANAError(f"Fit input {input_file} must contain a mapping of setting names to values")
    sectioned = any(key.upper() in ('SNLCINP', 'FITINP') for key in supplied)
    if sectioned or _snana_names(supplied):
        return 'snana', UsedTrackingDict(_flatten_sections(supplied))
    return 'yaml', supplied


def read_keys(input_file: Union[str, Path]) -> UsedTrackingDict:
    """Read SNANA keys with aliases resolved and both sections flattened."""
    _, settings = read_fit_input(input_file)
    return UsedTrackingDict(_flatten_sections(settings))


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return str(value).split()


def _flt(raw, key, into, name=None):
    if key in raw:
        into[name or key.lower()] = float(raw[key])


def _int(raw, key, into, name=None):
    if key in raw:
        into[name or key.lower()] = int(float(raw[key]))


def _opt_vmax(raw, into) -> None:
    """MAGLIM_VMAX / OPT_VMAX. Bit 0 is refused: SNANA indexes
    PEAKMAG_OBS_MODEL by Vmax list position where PEAKMAG_CALC fills it by
    survey-filter index (snlc_fit.F90:15240 against :14738). SNANA's other two
    bits change no output, bit 1 being tested only in VMAX_INI's banner
    (:15057) and bit 7 being verbosity, so the value is kept with no effect."""
    if "MAGLIM_VMAX" in raw:
        into["maglim_vmax"] = _filt_map(
            raw["MAGLIM_VMAX"], str(raw.get("FILTLIST_FIT", "") or ""))
    opt = int(float(raw.get("OPT_VMAX", 0)))
    if opt & 1:
        raise SNANAError(
            "OPT_VMAX bit 0 reads PEAKMAG_OBS_MODEL by Vmax list position from an "
            "array filled by survey-filter index; the two agree only "
            "when MAGLIM_VMAX names the leading survey filters in order.",
            function_name="_opt_vmax")
    into["opt_vmax"] = opt


def _opt_sncid_list(raw, into) -> None:
    """OPT_SNCID_LIST bits (snana.F90:6013), which FLAG_USE_SAME_EVENTS
    overrides when positive (:6290); bit 2 needs per-SN Gaussian priors in
    every model, so it is refused."""
    opt = int(float(raw.get("OPT_SNCID_LIST", 0)))
    if int(float(raw.get("FLAG_USE_SAME_EVENTS", 0))) > 0:
        opt = int(float(raw["FLAG_USE_SAME_EVENTS"]))
    if opt & 4:
        raise SNANAError(
            "OPT_SNCID_LIST bit 2 adds a per-CID Gaussian prior on every fit "
            "parameter, which needs model-side prior_data in all three models.",
            function_name="_opt_sncid_list")
    if opt & ~3:
        raise SNANAError(f"Unsupported OPT_SNCID_LIST = {opt}; bits 0 and 1 "
                         f"are implemented.", function_name="_opt_sncid_list")
    into["opt_sncid_list"] = opt


def _allow_negflux(raw, into) -> None:
    """ALLOW_NEGFLUX_SALT2 defaults true, which is what jax does; SNANA's manual
    marks = F "test only" (snana_manual.tex:11892)."""
    if "ALLOW_NEGFLUX_SALT2" not in raw:
        return
    if not convert(raw["ALLOW_NEGFLUX_SALT2"], bool, "ALLOW_NEGFLUX_SALT2"):
        raise SNANAError(
            "ALLOW_NEGFLUX_SALT2 = F clamps the SED per wavelength inside the "
            "band integral, which SNANA documents as test only. jax implements "
            "the default (negatives allowed).",
            function_name="_allow_negflux")


def _model_key_values(raw, model_type) -> Dict[str, Any]:
    """The supplied keys among those the model declares, coerced by the
    declared types with `parse_model_input_value`.

    Without FITMODEL_NAME the model is unknown, so the union of every
    registered class's keys is accepted; build_model sets only the actual
    class's keys.
    """
    classes = ([MODEL_REGISTRY[model_type]] if model_type in MODEL_REGISTRY
               else MODEL_REGISTRY.values())
    keys = {}
    for cls in classes:
        keys.update(model_input_keys(cls))
    values = {}
    for key, spec in keys.items():
        if key in raw:
            values[key] = parse_model_input_value(key, spec, raw[key])
    return values


def _str(raw, key, into, name=None):
    if key in raw:
        into[name or key.lower()] = str(raw[key])


def _file(raw, key, into, as_list=False):
    """A file key; an IGNOREFILE value (NONE, NULL, blank) sets it to None."""
    if key not in raw:
        return
    value = raw[key]
    if is_ignorefile(" ".join(map(str, value)) if isinstance(value, list)
                     else str(value)):
        into[key.lower()] = None
    elif as_list:
        _str_list(raw, key, into)
    else:
        _str(raw, key, into)


def _filt(raw, key, into):
    """A SNANA filter string, mapped to {band: value} against FILTLIST_FIT."""
    if key in raw:
        into[key.lower()] = _filt_map(
            raw[key], str(raw.get("FILTLIST_FIT", "") or ""))


def _triplets(raw, key, cast, into, name=None):
    """A flat list read as (a, b, c) groups, as SNANA's triplet keys are."""
    if key not in raw:
        return
    toks = _as_list(raw[key])
    if len(toks) % 3:
        raise SNANAError(
            f"{key} needs triplets; got {len(toks)} words.", function_name=key)
    into[name or key.lower()] = tuple(
        tuple(c(t) for c, t in zip(cast, toks[i:i + 3]))
        for i in range(0, len(toks), 3))


def _bool(raw, key, into, name=None):
    """``raw[key]`` read as a Fortran logical."""
    if key in raw:
        into[name or key.lower()] = convert(raw[key], bool, key)


def _pair(raw, key, into, name=None):
    if key in raw:
        vals = _as_list(raw[key])
        into[name or key.lower()] = (float(vals[0]), float(vals[1]))


# PRIOR_ suffixes that are not fit parameters
_PRIOR_NOT_A_PARAMETER = frozenset({'SHAPE2', 'DMU', 'DELTA'})

# PRIOR_<p>_<SUFFIX>: (form, coefficients, value kind)
_PRIOR_KEY_FORMS = {
    'RANGE': ('flat_rolloff',  ('lo', 'hi'), 'pair'),
    'SIGMA': ('flat_rolloff',  ('sigma',),   'float'),
    'EXP':   ('exp_convolved', ('tau',),     'list'),
    'WGT':   ('exp_convolved', ('wgt',),     'list'),
    'RES':   ('exp_convolved', ('sigma',),   'float'),
}


def _prior_key_parts(key):
    """Recognise the parameter and coefficient in a SNANA prior key."""
    if key.startswith('PRIOR_'):
        parameter, _, coefficient = key[len('PRIOR_'):].rpartition('_')
        if parameter and coefficient in _PRIOR_KEY_FORMS:
            return parameter, coefficient
    return None


def _prior_windows(raw, kw) -> None:
    """PRIOR_<p>_<SUFFIX> as ``{parameter: {form: coefficients}}``.

    The suffix names a parameter, so PRIOR_x1_RANGE and PRIOR_SHAPE_RANGE both
    reach SALT's x1, the second only because SHAPE is a SNANA spelling in the
    same table INIVAL_ and FITWIN_ resolve through.
    :func:`resolve_parameter_keys` rewrites the suffix as the parameter's own
    name once the model is known, so nothing downstream sees a generic one.
    """
    # OPT_PRIOR_AV = 0 disables the AV prior (snlc_fit.F90:7165)
    skip = () if int(float(raw.get("OPT_PRIOR_AV", 1))) > 0 else ('AV',)
    priors: dict = {}
    for key in list(raw):
        parts = _prior_key_parts(key)
        if parts is None:
            continue
        suffix, name = parts
        if suffix in _PRIOR_NOT_A_PARAMETER or suffix in skip:
            continue
        form, coefficients, kind = _PRIOR_KEY_FORMS[name]
        read: dict = {}
        if kind == 'pair':
            _pair(raw, key, read, name=key)
        elif kind == 'float':
            _flt(raw, key, read, name=key)
        elif key in raw and isinstance(raw[key], list):
            read[key] = [float(v) for v in raw[key]]
        if key not in read:
            continue
        target = priors.setdefault(suffix, {}).setdefault(form, {})
        if kind == 'pair':
            target[coefficients[0]], target[coefficients[1]] = read[key]
        else:
            target[coefficients[0]] = read[key]
    if priors:
        kw["priors"] = priors


def _filt_window(raw, key, into, name=None):
    """A per-filter window string as {band: (lo, hi)}.

    ``'gr -10 50  iz 0 50'``: glued band characters then the two edges, as
    SETCUTS_FITWIN_TREST reads it (snlc_fit.F90:2790-2812).
    """
    if key not in raw:
        return
    toks = _as_list(raw[key])
    if len(toks) % 3:
        raise SNANAError(
            f"{key} needs (filters, min, max) triplets; got {len(toks)} "
            f"words.", function_name=key)
    out = {}
    for bands, lo, hi in zip(toks[::3], toks[1::3], toks[2::3]):
        for band in str(bands):
            out[band] = (float(lo), float(hi))
    into[name or key.lower()] = out


def _tuple(raw, key, into, name=None):
    if key in raw:
        into[name or key.lower()] = tuple(float(v) for v in _as_list(raw[key]))


def _cospar(raw, key, into):
    """A cosmology reference: value, or value and uncertainty.

    SNANA declares these as 2-element arrays but inputs commonly set only the
    value, so a single number keeps the default uncertainty.
    """
    if key in raw:
        vals = _as_list(raw[key])
        name = key.lower()
        err = float(vals[1]) if len(vals) > 1 else getattr(Fit, name)[1]
        into[name] = (float(vals[0]), err)


def _int_list(raw, key, into, name=None):
    # exact: a Rubin CID has 18 digits, past a float's 16
    if key in raw:
        into[name or key.lower()] = [int(v) for v in _as_list(raw[key])]


def _until_sentinel(values, floor):
    """The leading run of a SNANA list, which ends at its first entry below
    ``floor`` rather than skipping it."""
    out = []
    for value in values:
        if value < floor:
            break
        out.append(value)
    return out


def _str_list(raw, key, into, name=None, split_words=False):
    if key in raw:
        v = raw[key]
        vals = v if isinstance(v, list) else [v]
        if split_words:
            vals = [word for value in vals for word in str(value).split()]
        into[name or key.lower()] = [str(x).strip().strip("'") for x in vals]


def _filt_map(value: Any, all_bands: str = "") -> Dict[str, float]:
    """A SNANA filter-string as {band: value}; a bare scalar keys ''.

    Bands may be glued ('gri 0.01'), and 'ALL' names every fit band
    (snana.F90:14327-14331).
    """
    toks = value if isinstance(value, list) else str(value).split()
    if len(toks) == 1:
        return {"": float(toks[0])}
    out = {}
    for band, v in zip(toks[::2], toks[1::2]):
        if str(band).upper() == "ALL":
            if not all_bands:
                raise SNANAError(
                    f"'ALL' in filter string '{value}' needs the fit band "
                    f"list", function_name="_filt_map")
            names = all_bands
        else:
            names = str(band)
        for b in names:
            out[b] = float(v)
    return out


# SNANA NML key -> apply_fluxcal_fudges argument name
_FUDGE_FLUXCAL_KEYS = {
    "FUDGE_FLUXCAL_OFFSET": "offset",
    "FUDGE_FLUXCAL_ERROR": "err_add",
    "FUDGE_FLUXCAL_ERRPIX": "err_pix",
    "FUDGE_FLUXERR_SCALE": "err_scale",
    "FUDGE_MAG_ERROR": "mag_err",
    "SIM_FUDGE_MAG_ERROR": "sim_mag_err",
}

# CUTWIN names that mask epochs (SNANA LSIG)
_EPOCH_CUTS = frozenset({
    "mjd", "mjd_exclude", "zperr", "photprob", "psf", "zpadu", "zpnpe",
    "tobs", "trest", "lamobs", "lamrest", "errtest", "snr_nodetect",
})


# SNTABLE_LIST tables jax does not write
_UNSUPPORTED_TABLES = {
    'prior': 'MINUIT-grid product; no jax analog',
    'pdf': 'MINUIT-grid product; no jax analog',
    'specplot': 'fitter has no spectra handling',
    'modelspec': 'fitter has no spectra handling',
    'spectra': 'fitter has no spectra handling',
}
_TEXT_FORMATS = ('key', 'csv', 'col', 'none')


def parse_sntable_list(value):
    """Parse the SNTABLE_LIST string (snana.F90:6540-6740).

    Returns the Output fields this string sets, and only those: ``tables``
    ({name: format}), whichever of the flags
    ``host_text``/``host2_text``/``nosimvar``/``nozphot``/``nonuisance``/
    ``iauc`` and the
    knobs ``cutmask``/``mxlc``/``outlier_nsig``/``outlier_ftrue``/
    ``outlier_sbmag`` it names, and ``unsupported`` (warning strings). The
    defaults for the rest live on the record.
    """
    out = {'tables': {}, 'unsupported': []}
    value = str(value or '').strip()
    if value == 'DEFAULT':
        value = 'SNANA FITRES'
    if not value or value.upper().startswith('NONE'):
        return out

    for word in value.split():
        m = re.fullmatch(r'([\w+]+)(?:\((.*)\))?', word.lower())
        if m is None:
            raise SNANAError(f"Cannot parse SNTABLE_LIST entry '{word}'")
        tbname, opts = m.group(1), (m.group(2) or '')

        # pseudo-options (snana.F90:6656-6660)
        if tbname.startswith('nosimvar'):
            out['nosimvar'] = True
            continue
        if tbname.startswith('nozphot'):
            out['nozphot'] = True
            continue
        # jax-only: drop nuisance parameters from the FITRES
        if tbname.startswith('nonuisance'):
            out['nonuisance'] = True
            continue

        # prefix-matched table names (snana.F90:6597-6666)
        if tbname.startswith('snana'):
            name = 'snana'
            if tbname != 'snana':
                out['unsupported'].append(
                    f"'{tbname}': epoch/sim_magobs blocks are ROOT-only (no TEXT effect)")
        elif tbname.startswith('fitres'):
            name = 'fitres'
            if tbname != 'fitres':
                out['unsupported'].append(
                    f"'{tbname}': residual block is ROOT-only (no TEXT effect)")
        elif tbname.startswith('outlier'):
            name = 'outlier'
        elif tbname.startswith('lcplot') or tbname.startswith('lcphot'):
            name = 'lcplot'
        else:
            for known, reason in _UNSUPPORTED_TABLES.items():
                if tbname.startswith(known):
                    out['unsupported'].append(f"'{tbname}': {reason}")
                    name = None
                    break
            else:
                raise SNANAError(
                    f"Unrecognized SNTABLE_LIST table name '{word}'")
            if name is None:
                continue

        out['tables'][name] = 'key'
        for opt in filter(None, (o.strip() for o in opts.split(','))):
            if opt.startswith('text:'):
                arg = opt[5:]
                # host2 before host: SNANA's if-chain order (snana.F90:6678)
                if arg.startswith('host2'):
                    out['host2_text'] = True
                elif arg.startswith('host'):
                    out['host_text'] = True
                elif arg.startswith('errcalc'):
                    out['unsupported'].append(
                        "'text:errcalc': no-op in current SNANA (flag never read)")
                elif arg in _TEXT_FORMATS:
                    out['tables'][name] = arg
                else:
                    raise SNANAError(f"Invalid TEXT_FORMAT = '{arg}'")
            elif opt.startswith('cutmask:') and name == 'snana':
                out['cutmask'] = int(opt[8:])
            elif opt.startswith('mxlc:') and name == 'lcplot':
                out['mxlc'] = int(opt[5:])
            elif opt.startswith('nsig:') and name == 'outlier':
                out['outlier_nsig'] = float(opt[5:])
            elif opt.startswith('ftrue:') and name == 'outlier':
                out['outlier_ftrue'] = float(opt[6:])
            elif opt.startswith('sbmag:') and name == 'outlier':
                out['outlier_sbmag'] = float(opt[6:])
            elif opt.startswith('iauc'):
                out['iauc'] = True
            elif opt.startswith('ps:'):
                out['unsupported'].append(
                    f"'{opt}': no effect on SNANA TEXT tables (verified)")
            else:
                out['unsupported'].append(f"'{opt}': unrecognized option")
    return out


def build(raw: Dict[str, Any], *, settings=None) -> FitConfig:
    """Interpret the raw keys into a FitConfig, in one pass.

    Cheap and side-effect free: it resolves paths and values but reads no data
    files, because `known_keys` calls it a second time.
    """
    kw: Dict[str, Any] = {}

    # jax-only keys
    if "PHOTOMETRY_COLUMNS" in raw:
        toks = str(raw["PHOTOMETRY_COLUMNS"]).split()
        kw["photometry_columns"] = dict(
            toks[i:i + 2] for i in range(0, len(toks), 2))

    # SNANA forces E(B-V) only above -0.001 (snana.F90:17209); its default,
    # -9, leaves it off
    if "MWEBV_FORCE" in raw:
        force = float(raw["MWEBV_FORCE"])
        kw["mwebv_force"] = force if force > -0.001 else None
    _int_list(raw, "SNTYPE_LIST", kw)
    # a numeric CID list ends at its first entry below 1 (DO WHILE
    # SNCID_LIST(i) >= 1, snana.F90:11888)
    for key in ("SNCID_LIST", "SNCID_IGNORE"):
        if key in raw:
            kw[key.lower()] = _until_sentinel([int(v) for v in _as_list(raw[key])], 1)
    _str_list(raw, "SNCCID_LIST", kw, split_words=True)
    _str_list(raw, "SNFIELD_LIST", kw, split_words=True)
    _str_list(raw, "SNCCID_IGNORE", kw, split_words=True)
    # ends at the first negative entry (snana.F90:5915)
    if "SNTYPE_IGNORE" in raw:
        kw["sntype_ignore"] = _until_sentinel(
            [int(v) for v in _as_list(raw["SNTYPE_IGNORE"])], 0)
    _prior_windows(raw, kw)
    _filt(raw, "MAGOBS_SHIFT_ZP", kw)
    if "MAGOBS_SHIFT_ZP_PARAMS" in raw:
        vals = [float(v) for v in _as_list(raw["MAGOBS_SHIFT_ZP_PARAMS"])]
        kw["magobs_shift_zp_params"] = tuple(vals + [0.0] * (3 - len(vals)))[:3]
    _filt(raw, "MAGOBS_SHIFT_PRIMARY", kw)
    if "MAGOBS_SHIFT_PRIMARY_PARAMS" in raw:
        vals = [float(v) for v in _as_list(raw["MAGOBS_SHIFT_PRIMARY_PARAMS"])]
        kw["magobs_shift_primary_params"] = tuple(
            vals + [0.0] * (3 - len(vals)))[:3]
    _filt(raw, "FILTER_LAMSHIFT", kw)
    if "FILTER_REPLACE" in raw:
        toks = str(raw["FILTER_REPLACE"]).split()
        if len(toks) != 3 or toks[1] != "->":
            raise SNANAError(
                f"FILTER_REPLACE must read 'LIST1 -> LIST2'; got "
                f"'{raw['FILTER_REPLACE']}'.", function_name="FILTER_REPLACE")
        if len(toks[0]) != len(toks[2]):
            raise SNANAError(
                f"FILTER_REPLACE lists differ in length: "
                f"N({toks[0]}) != N({toks[2]}).",
                function_name="FILTER_REPLACE")
        kw["filter_replace"] = dict(zip(toks[0], toks[2]))
    _pair(raw, "PEAKMJD_SANITY", kw)
    if "PHOTFLAG_MSKREJ" in raw:
        kw["photflag_mskrej"] = tuple(
            int(float(v)) for v in _as_list(raw["PHOTFLAG_MSKREJ"]))
    if "PHOTFLAG_BITLIST_REJECT" in raw:
        bits = [int(float(v)) for v in
                _as_list(raw["PHOTFLAG_BITLIST_REJECT"])]
        mask = 0
        for bit in _until_sentinel(bits, 0):
            mask |= 1 << bit
        # replaces element 1 of PHOTFLAG_MSKREJ (snana.F90:5856-5861)
        rest = tuple(kw.get("photflag_mskrej", ())[1:])
        kw["photflag_mskrej"] = (mask,) + rest

    _filt(raw, "EPCUT_SNRMIN", kw)
    _bool(raw, "USE_MODEL_MAGERR", kw)
    # SCALE_COVAR <= 0 only disables the model error (snlc_fit.F90:7161)
    if "SCALE_COVAR" in raw and float(raw["SCALE_COVAR"]) <= 0.0:
        kw["use_model_magerr"] = False

    cutwin: Dict[str, Tuple[float, float]] = {}
    epoch_cutwin: Dict[str, Tuple[float, float]] = {}
    for key in list(raw):
        if not key.startswith("CUTWIN_"):
            continue
        name = key[len("CUTWIN_"):].lower()
        if not (name in _EPOCH_CUTS or name in SN_CUT_NAMES
                or name.startswith(SN_CUT_BAND_PREFIXES)):
            raise SNANAError(f"{key} names no quantity the fit cuts on",
                             function_name="CUTWIN")
        vals = _as_list(raw[key])
        # upper edge defaults open (snana.F90:11732-11733)
        try:
            window = (float(vals[0]),
                      float(vals[1]) if len(vals) > 1 else CUTVAL_OPEN)
        except ValueError:
            raise SNANAError(f"{key} takes a numeric window 'lo hi', not "
                             f"{raw[key]!r}; a per-band threshold is SNCUT_<X>",
                             function_name="CUTWIN") from None
        target = epoch_cutwin if name in _EPOCH_CUTS else cutwin
        target[name] = window
    # ZPADU/ZPNPE exclusive; lower edge <= 0 unsets (snana.F90:5942, :5931)
    zp_set = [n for n in ("zpadu", "zpnpe")
              if epoch_cutwin.get(n, (0.0,))[0] > 0.0]
    if len(zp_set) > 1:
        raise SNANAError(
            "Cannot specify both CUTWIN_ZPADU and CUTWIN_ZPNPE.",
            function_name="CUTWIN_ZP")
    for name in ("zpadu", "zpnpe"):
        if name in epoch_cutwin and name not in zp_set:
            del epoch_cutwin[name]

    _read_require_epochs(raw, kw)
    _read_earlylc(raw, kw)

    if "PRIVATE_VARNAME_READLIST" in raw:
        kw["private_varname_readlist"] = tuple(
            str(v) for v in _as_list(raw["PRIVATE_VARNAME_READLIST"]))
    _priv, _veto = {}, {}
    for key in [k for k in raw if k.startswith("PRIVATE_CUTWIN_STRING")]:
        # first three words of each element (snana.F90:5085-5093)
        toks = [str(t) for t in _as_list(raw[key])]
        if len(toks) < 3:
            raise SNANAError(
                f"{key} needs 'VARNAME CUTMIN CUTMAX' or 'VARNAME != VALUE'; "
                f"got {len(toks)} words.",
                function_name="PRIVATE_CUTWIN_STRING")
        name, second, third = toks[0], toks[1], toks[2]
        if second == "!=":
            _veto[name] = float(third)
        else:
            _priv[name] = (float(second), float(third))
    if _priv:
        kw["private_cutwin"] = _priv
    if _veto:
        kw["private_veto"] = _veto

    _triplets(raw, "SNCUT_NOBS_TREST",
              (lambda v: int(float(v)), float, float), kw)

    _triplets(raw, "SIMVAR_CUTWIN_STRING", (str, float, float), kw, "_simvar")
    if "_simvar" in kw:
        kw["simvar_cutwin"] = {n: (lo, hi) for n, lo, hi in kw.pop("_simvar")}

    if "JOBSPLIT" in raw:
        vals = [int(float(v)) for v in _as_list(raw["JOBSPLIT"])]
        kw["jobsplit"] = (vals[0], vals[1])
    if "SIM_TEMPLATE_INDEX_LIST" in raw:
        # stops at the first non-positive entry (PARSE_INTLIST)
        kw["sim_template_index_list"] = _until_sentinel(
            [int(float(v)) for v in _as_list(raw["SIM_TEMPLATE_INDEX_LIST"])], 1)

    # SNCUT_<X> sets the CUTWIN lower edge (snana.F90:1441)
    fit_bands = str(raw.get("FILTLIST_FIT", "") or "")
    for key, name in (("SNCUT_SNRMAX", "snrmax"),
                      ("SNCUT_HOST_SBFLUX", "sbflux")):
        if key in raw:
            for band, lo in _filt_map(raw[key], fit_bands).items():
                cutwin[f"{name}_{band.lower()}"] = (lo, CUTVAL_OPEN)

    if cutwin:
        kw["cutwin"] = cutwin
    if epoch_cutwin:
        kw["epoch_cutwin"] = epoch_cutwin

    for key in ("MAGCOR_FILE", "SIM_MAGCOR_FILE", "NONLINEARITY_FILE",
                "EPOCH_IGNORE_FILE",
                "FUDGE_HOSTNOISE_FILE", "FLUXERRMODEL_FILE",
                "SIM_FLUXERRMODEL_FILE", "HEADER_OVERRIDE_DIR",
                "SIM_HEADER_OVERRIDE_DIR"):
        _file(raw, key, kw)
    for key in ("HEADER_OVERRIDE_FILE", "SIM_HEADER_OVERRIDE_FILE"):
        _file(raw, key, kw, as_list=True)

    fudge_fluxcal = {name: _filt_map(raw[key])
                     for key, name in _FUDGE_FLUXCAL_KEYS.items() if key in raw}
    if fudge_fluxcal:
        kw["fudge_fluxcal"] = fudge_fluxcal
    _tuple(raw, "FUDGE_FITERR_TREST", kw)
    _tuple(raw, "FUDGE_FITERR_RESTLAM", kw)
    if "FUDGE_MAG_COVERR" in raw:
        kw["fudge_mag_coverr"] = _filt_map(raw["FUDGE_MAG_COVERR"])

    _int(raw, "OPT_MWCOLORLAW", kw)
    _flt(raw, "RV_MWCOLORLAW", kw)
    # SNANA's unset values
    if kw.get("opt_mwcolorlaw", 0) <= -998:
        kw["opt_mwcolorlaw"] = None
    if kw.get("rv_mwcolorlaw", 0.0) <= -0.01:
        kw["rv_mwcolorlaw"] = None
    if (kw.get("opt_mwcolorlaw") is not None
            and kw["opt_mwcolorlaw"] not in OPT_MWCOLORLAW_NAMES):
        raise SNANAError(
            f"Invalid OPT_MWCOLORLAW = {kw['opt_mwcolorlaw']}; valid codes: "
            f"{', '.join(str(c) for c in sorted(OPT_MWCOLORLAW_NAMES))}.",
            function_name="OPT_MWCOLORLAW")
    if "PARLIST_MWCOLORLAW" in raw:
        kw["parlist_mwcolorlaw"] = tuple(
            float(v) for v in _as_list(raw["PARLIST_MWCOLORLAW"]))

    kw["sntable"] = parse_sntable_list(str(raw.get("SNTABLE_LIST", "DEFAULT")))
    # MXLC_PLOT loses to mxlc:N (snana.F90:6692).
    if "mxlc" not in kw["sntable"] and "MXLC_PLOT" in raw:
        kw["sntable"]["mxlc"] = int(float(raw["MXLC_PLOT"]))

    usesim = [generic for key, generic in _USESIM_PARAM_KEYS.items()
              if key in raw and convert(raw[key], bool, key)]
    if usesim:
        kw["usesim_params"] = usesim
    _allow_negflux(raw, kw)
    _opt_sncid_list(raw, kw)
    _opt_vmax(raw, kw)

    for key in ("H0_REF", "OMAT_REF", "OLAM_REF", "W0_REF", "WA_REF"):
        _cospar(raw, key, kw)

    _int(raw, "NFIT_ITERATION", kw, "n_fit_iterations")
    _pair(raw, "RESTLAMBDA_FITRANGE", kw)
    _pair(raw, "FITWIN_TREST", kw)
    _filt_window(raw, "FITWIN_TREST_FILTER", kw, "fitwin_trest_filter")
    _pair(raw, "FITWIN_MJD", kw)
    _read_fitwin(raw, kw)

    # OPT_COVAR_FLUX=2 unsupported; _warn_unimplemented_values warns
    if "OPT_COVAR_FLUX" in raw:
        kw["use_offdiag_covariance"] = int(float(raw["OPT_COVAR_FLUX"])) != 0

    # SNANA applies no prior for a scale of 9 or more (snlc_fit.F90:4910);
    # namelists write 100 or 1000 to switch it off
    if "PRIOR_MUERRSCALE" in raw:
        scale = float(raw["PRIOR_MUERRSCALE"])
        kw["prior_muerrscale"] = scale if scale < 9.0 else None
    _pair(raw, "PHOTOZ_BOUND", kw)
    _pair(raw, "PHOTOZ_ITER1_LAMRANGE", kw)
    # coarse-grid colour and shape axes (snlc_fit.F90:606-617)
    coarse_grid = {}
    for slot, range_key, nbin_key in (
            ('COLOR', 'RANGEC_COURSE_PHOTOZ', 'NBINC_COURSE_PHOTOZ'),
            ('SHAPE', 'RANGES_COURSE_PHOTOZ', 'NBINS_COURSE_PHOTOZ')):
        axis = {}
        _pair(raw, range_key, axis, 'range')
        _int(raw, nbin_key, axis, 'nbin')
        if axis:
            coarse_grid[slot] = axis
    kw["coarse_grid"] = coarse_grid
    _int(raw, "NBINZ_COURSE_PHOTOZ", kw, "nbinz_course")
    _pair(raw, "RANGEZ_COURSE_PHOTOZ", kw, "rangez_course")
    _int(raw, "NBINT_COURSE_PHOTOZ", kw, "nbint_course")
    _pair(raw, "RANGET_COURSE_PHOTOZ", kw, "ranget_course")
    _int(raw, "SCALE_NBIN_COURSE_PHOTOZ", kw, "scale_nbin_course")

    _int(raw, 'OPT_CHI2_SIGMA', kw)
    if 'DOCHI2_SIGMA' in raw and convert(raw['DOCHI2_SIGMA'], bool, 'DOCHI2_SIGMA'):
        kw['opt_chi2_sigma'] = 1

    _read_inival_inistp(raw, kw)

    model_type = selected_model(merge_fit_settings(kw, settings or {}))
    if model_type is not None and model_type not in MODEL_REGISTRY:
        raise SNANAError(
            f"Unknown model type '{model_type}'. Available: "
            f"{', '.join(MODEL_REGISTRY)}.", function_name="MODEL_TYPE")
    model_values = _model_key_values(raw, model_type)

    table_options = kw.pop("sntable")
    table_warnings = table_options.pop("unsupported")
    kw.update(table_options)

    # Ordinary new fields need no corresponding edit to this compatibility
    # reader. Special SNANA structures above retain their own interpretation.
    types, _, _ = declarations(model_type)
    for name, annotation in types.items():
        key = name.upper()
        if key in raw and name not in kw:
            value = raw[key]
            # a Fortran namelist separates an array's values by blanks too
            if (isinstance(value, str)
                    and get_origin(value_type(annotation)) in (list, tuple)):
                value = value.split()
            kw[name] = convert(value, annotation, key)
    config = build_config(merge_fit_settings(kw, settings or {}))
    config.model._model_key_values = {**model_values, **config.model._model_key_values}
    config.output._unsupported = table_warnings
    return config


# USESIM_<x> -> generic parameter (snlc_fit.F90:1135-1140)
_USESIM_PARAM_KEYS = {
    'USESIM_PEAKMJD': 'PEAKMJD',
    'USESIM_SHAPE': 'SHAPE',
    'USESIM_LUMIPAR': 'LUMIPAR',
    'USESIM_COLOR': 'COLOR',
    'USESIM_AV': 'AV',
    'USESIM_RV': 'RV',
    'USESIM_DLMAG': 'DLMAG',
    'USESIM_REDSHIFT': 'PHOTOZ',
}


# FITWIN_ suffixes read elsewhere in build(), as the epoch cuts they are
_FITWIN_EPOCH = frozenset({'TREST', 'TREST_FILTER', 'MJD'})

# Mid-fit windows jax evaluates on the round-1 fit rather than the finished one
_FITWIN_ITER1 = {'PROB_ITER1': 'fitwin_prob_iter1',
                 'CHI2RED_INI': 'fitwin_chi2red_ini',
                 'CHI2RED_INI2': 'fitwin_chi2red_ini2'}

_FITWIN_REFUSED = {
    'TREST_FILT': "set FITWIN_TREST or FITWIN_TREST_FILTER for rest-frame phase cuts",
    'CHI2SIGMA': 'selection on a separate model-uncertainty likelihood term is not supported',
}

# FITWIN arrays are initialised wide, so an input may set one edge alone
# (CUTINI, snlc_fit.F90:6763-6764).
_FITWIN_OPEN = 9.0e8


def _fitwin_pair(value) -> Tuple[float, float]:
    """Both edges of a FITWIN window, filling an unset one from CUTINI."""
    vals = _as_list(value)
    return (float(vals[0]),
            float(vals[1]) if len(vals) > 1 else _FITWIN_OPEN)


def _read_require_epochs(raw: UsedTrackingDict, kw: dict) -> None:
    """REQUIRE_EPOCHS_STRING, in its four-word and keyword forms
    (PARSE_REQUIRE_EPOCHS, snana.F90:16466)."""
    if "REQUIRE_EPOCHS_STRING" not in raw:
        return
    toks = [str(t) for t in _as_list(raw["REQUIRE_EPOCHS_STRING"])]
    req = {"on": False}
    if len(toks) == 4:
        req.update(on=True, tranges=tuple(float(t) for t in toks[1:4]))
    else:
        i = 0
        while i < len(toks):
            w = toks[i].upper()
            if w == "FILTERS":
                req["on"] = True; i += 2
            elif w == "SNRMIN":
                req["snrmin"] = float(toks[i + 1]); i += 2
            elif w in ("TOBS_RANGES", "TREST_RANGES"):
                req["rest_frame"] = w.startswith("TREST")
                req["tranges"] = tuple(float(t) for t in toks[i + 1:i + 4])
                i += 4
            else:
                raise SNANAError(
                    f"REQUIRE_EPOCHS_STRING: unrecognised word '{toks[i]}'; "
                    f"expected FILTERS, SNRMIN, TOBS_RANGES or TREST_RANGES.",
                    function_name="REQUIRE_EPOCHS_STRING")
    if req.pop("on"):
        kw["require_epochs"] = req


def _read_earlylc(raw: UsedTrackingDict, kw: dict) -> None:
    """EARLYLC_STRING keyword/value pairs (PARSE_EARLYLC_STRING,
    snana.F90:16252)."""
    if "EARLYLC_STRING" not in raw:
        return
    toks = [str(t) for t in _as_list(raw["EARLYLC_STRING"])]
    early = {}
    types, _ = dataclass_settings(EarlyLC)
    names = {name.upper(): name for name in types}
    i = 0
    while i < len(toks):
        name = names.get(toks[i].upper())
        if name is None or i + 1 >= len(toks):
            raise SNANAError(
                f"EARLYLC_STRING: unrecognised word '{toks[i]}'; expected "
                f"one of {', '.join(sorted(names))}, each followed by a "
                f"value.", function_name="EARLYLC_STRING")
        early[name] = convert(toks[i + 1], types[name], name)
        i += 2
    kw["earlylc"] = early


def _read_fitwin(raw: UsedTrackingDict, kw: dict) -> None:
    """Collect the FITWIN_ windows applied to the finished fit.

    The suffix names a FITRES column (or one of its _ERR / _ERRz1 forms), which
    only the model can resolve, so it is carried raw and resolved by
    :func:`~jax_snlc_fit.parameters.resolve_fitwin_windows`.
    """
    windows = {}
    iter1 = {}
    for key in [k for k in raw if k.startswith("FITWIN_")]:
        suffix = key[len("FITWIN_"):]
        if suffix in _FITWIN_EPOCH:
            continue
        if suffix in _FITWIN_REFUSED:
            raise SNANAError(_FITWIN_REFUSED[suffix], function_name=key)
        if suffix in _FITWIN_ITER1:
            kw[_FITWIN_ITER1[suffix]] = _fitwin_pair(raw[key])
        elif suffix in ("SHAPE_ITER1", "COLOR_ITER1"):
            iter1[suffix[:-len("_ITER1")]] = _fitwin_pair(raw[key])
        else:
            windows[suffix] = _fitwin_pair(raw[key])
    if windows:
        kw["fitwin"] = windows
    if iter1:
        kw["fitwin_iter1"] = iter1


def _read_inival_inistp(raw: UsedTrackingDict, kw: dict) -> None:
    """Split the family: named settings first, the rest name parameters."""
    inival, inistp, grids = {}, {}, {}
    for key in [k for k in raw if k.startswith(("INIVAL_", "INISTP_"))]:
        if key in raw.used or key in _INERT_KEYS:
            continue
        prefix, _, suffix = key.partition("_")
        if suffix.startswith("GRIDSEARCH_"):
            grids[suffix[len("GRIDSEARCH_"):]] = tuple(
                float(v) for v in _as_list(raw[key]))
        elif prefix == "INIVAL":
            inival[suffix] = float(raw[key])
        else:
            inistp[suffix] = float(raw[key])

    for name, value in (("inival", inival), ("inistp", inistp),
                        ("inival_gridsearch", grids)):
        if value:
            kw[name] = value


# warnings belong here: build() runs twice
def _warn_unimplemented_values(raw: Dict[str, Any]) -> None:
    """Warn about recognised settings whose value jax does not implement."""
    if "OPT_COVAR_FLUX" in raw and int(float(raw["OPT_COVAR_FLUX"])) == 2:
        log.warn(logger, "CONFIG",
                 "OPT_COVAR_FLUX=2 (recompute covariance inside the fit loop) "
                 "is not implemented; using the frozen covariance of =1")


def _requests_nothing(key, value) -> bool:
    """Whether a refused key carries the value at which it asks for nothing."""
    _, inert = _REJECTED_OPTIONS[key]
    if inert is None:
        return False
    text = str(_as_list(value)[0] if isinstance(value, list) else value).strip()
    if isinstance(inert, bool):
        return convert(text, bool, key) is inert
    if isinstance(inert, str):
        return text.strip("'\"") == inert
    if inert is _IGNORE_FILENAMES:
        return is_ignorefile(text.strip("'\""))
    try:
        return inert(float(text)) if callable(inert) else float(text) == inert
    except ValueError:
        return False


def _check_rejected_keys(raw) -> None:
    """Abort on input keys jax refuses, unless the value switches them off."""
    found = []
    for key in sorted(set(raw) & set(_REJECTED_KEYS)):
        if not _requests_nothing(key, raw[key]):
            found.append(key)
    if found:
        raise SNANAError(
            "Unsupported SNANA input key(s):\n"
            + "\n".join(f"  {k}: {_REJECTED_KEYS[k]}" for k in found)
            + "\nThe SNANA differences section of the documentation has the detail.",
            function_name="_check_rejected_keys")


def _warn_unrecognized_keys(raw: UsedTrackingDict, typed_keys: set) -> None:
    """Log one warning listing input keys the build never read."""
    unknown = sorted(
        k for k in typed_keys
        if k not in raw.used and k not in _INERT_KEYS
        and not any(k.startswith(p) for p in _OPEN_KEY_PREFIXES))
    if unknown:
        log.warn(logger, "CONFIG",
                 f"Unrecognized SNANA input key(s) ignored: {', '.join(unknown)}")


def known_keys() -> set:
    """The keys this input recognises, for classifying command-line tokens.

    A token is a key iff the build reads it. Aliases and rejected keys join them
    so an override naming one is grouped, then refused by name rather than
    reported as an unknown key.
    """
    # Empty input discovers reads without validating values that an override
    # may replace. Model declarations supply their keys without loading assets.
    probe = UsedTrackingDict()
    build(probe)
    return (set(probe.used) | set(_KEY_ALIASES) | set(_REJECTED_KEYS)
            | set(_INERT_KEYS))


def _is_override_key(token: str, known: set) -> bool:
    """Whether a command-line token names a key rather than a value."""
    return (token in known or token.startswith(_OPEN_KEY_PREFIXES)
            or _prior_key_parts(token) is not None)


def group_override_tokens(tokens: Sequence[str],
                          known: set) -> List[Tuple[str, List[str]]]:
    """Split command-line tokens into (key, values) pairs.

    Values run until the next token that is itself a key, so multi-token values
    (`FITWIN_TREST -15 45`) group correctly.
    """
    groups: List[Tuple[str, List[str]]] = []
    i = 0
    while i < len(tokens):
        key = tokens[i].upper()
        if "=" in key:
            key, _, inline = key.partition("=")
            key = _KEY_ALIASES.get(key, key)
            if not _is_override_key(key, known):
                raise SNANAError(f"Unknown command-line override key '{key}'.",
                                 function_name="group_override_tokens")
            groups.append((key, [tokens[i].partition("=")[2]]))
            i += 1
            continue
        key = _KEY_ALIASES.get(key, key)
        if not _is_override_key(key, known):
            raise SNANAError(f"Unknown command-line override key '{key}'.",
                             function_name="group_override_tokens")
        j = i + 1
        while j < len(tokens) and not _is_override_key(
                tokens[j].upper().partition("=")[0], known):
            j += 1
        values = list(tokens[i + 1:j])
        if not values:
            raise SNANAError(f"Command-line override '{key}' has no value.",
                             function_name="group_override_tokens")
        groups.append((key, values))
        i = j
    return groups


def apply_overrides(raw: UsedTrackingDict,
                    groups: Sequence[Tuple[str, List[str]]]) -> List[str]:
    """Replay each override through the input-file line parser."""
    lines = []
    for key, values in groups:
        line = f"{key} = {' '.join(str(v) for v in values)}"
        if "!" in line:
            raise SNANAError(
                f"Command-line override '{line}' contains '!', which the input "
                f"parser treats as a comment.",
                function_name="apply_overrides")
        _read_line(line, raw, section="OVERRIDE")
        lines.append(line)
    return lines


def _split_command_tokens(tokens, settings, input_format):
    """Command-line ``tokens``, named options and SNANA-style overrides in
    any mix, as (SNANA-style tokens, named settings). The named options are
    parsed for the model they select: a named ``--model-type`` or
    ``--fitmodel-name``, else the input's own, with its SNANA-style overrides
    applied. A YAML input takes named options only."""
    types, owners, _ = declarations()
    selectors, snana_tokens = bootstrap_settings(
        tokens, types, owners, *model_input_declarations('fit'), readers=INPUT_READERS)
    selection = UsedTrackingDict(settings)
    if input_format == 'snana' and snana_tokens:
        apply_overrides(selection, group_override_tokens(snana_tokens, known_keys()))
    model_key = 'FITMODEL_NAME' if input_format == 'snana' else 'fitmodel_name'
    type_key = 'MODEL_TYPE' if input_format == 'snana' else 'model_type'
    model_type = selectors.get('model_type') or selected_model({
        'fitmodel_name': selectors.get('fitmodel_name') or selection.get(model_key),
        'model_type': None if selectors.get('fitmodel_name') else selection.get(type_key)})
    parser = argument_parser(model_type)
    args, snana_tokens = parser.parse_known_args(tokens)
    if snana_tokens and (input_format == 'yaml'
                         or any(token.startswith('--') for token in snana_tokens)):
        parser.error('unrecognized arguments: ' + ' '.join(snana_tokens))
    types, _, _ = declarations(model_type)
    return snana_tokens, parsed_settings(args, types, readers=INPUT_READERS)


def build_file_config(settings, input_format, overrides=()):
    """Apply overrides to already-read input, then build its configuration.
    ``overrides`` are command-line tokens (:func:`_split_command_tokens`), or
    a mapping of SNANA keys or of named settings."""
    snana_mapping = (input_format == 'snana' and isinstance(overrides, Mapping)
                     and _snana_names(overrides))
    if isinstance(overrides, Mapping):
        snana_tokens, named = (), ({} if snana_mapping else overrides)
    else:
        snana_tokens, named = _split_command_tokens(overrides, settings, input_format)
    if input_format == 'yaml':
        return build_config(merge_fit_settings(settings, named))

    raw = UsedTrackingDict(settings)
    if snana_mapping:
        supplied = _flatten_sections(dict(overrides))
        known = known_keys()
        unknown = [key for key in supplied if not _is_override_key(key, known)]
        if unknown:
            raise SNANAError(f"Unknown SNANA override keys: {', '.join(unknown)}")
        raw.update(supplied)
    elif snana_tokens:
        apply_overrides(raw, group_override_tokens(snana_tokens, known_keys()))
    typed_keys = set(raw)
    _check_rejected_keys(raw)
    cfg = build(raw, settings=merge_fit_settings(named))
    _warn_unrecognized_keys(raw, typed_keys)
    _warn_unimplemented_values(raw)
    return cfg


def parse_fit_input_file(input_file: Union[str, Path],
                         overrides: Union[Sequence[str], Mapping[str, Any]] = ()) -> FitConfig:
    """Read fit settings and apply command-line or mapping overrides."""
    input_format, settings = read_fit_input(input_file)
    return build_file_config(settings, input_format, overrides)


def declarations(model_type=None):
    """Find configurable fields on their owners, including the selected model."""
    types, owners, groups = {}, {}, {}
    for group, owner in get_type_hints(FitConfig).items():
        for name, annotation in dataclass_settings(owner)[0].items():
            if name.startswith('_'):
                continue
            if name in types:
                raise SNANAError(f"Input {name!r} is declared by more than one component")
            types[name], owners[name], groups[name] = annotation, owner, group
    if model_type:
        if model_type not in MODEL_REGISTRY:
            raise SNANAError(f"Unknown model type {model_type!r}; choose from {tuple(MODEL_REGISTRY)}")
        owner = MODEL_REGISTRY[model_type]
        for spec in model_input_keys(owner, 'fit').values():
            name = spec.attribute
            if name in types:
                # Existing shared model settings have one configuration owner.
                if types[name] != spec.kind:
                    raise SNANAError(f"Conflicting declarations for {name}")
                continue
            types[name] = get_type_hints(owner, include_extras=True)[name]
            owners[name], groups[name] = owner, '_model_key_values'
    return types, owners, groups


def selected_model(settings):
    """Select a model class without opening its trained assets."""
    return settings.get('model_type') or infer_model_type_from_path(
        os.path.expandvars(settings.get('fitmodel_name') or ''))


def configuration_values(config, *, supplied=True):
    """Return detached flat inputs, including model-owned settings."""
    values = {}
    for field in dataclasses.fields(config):
        record = getattr(config, field.name)
        values.update({name: copy.deepcopy(getattr(record, name))
                       for name in dataclass_settings(type(record))[0]})
    model_values = config.model._model_key_values
    for name, value in model_values.items():
        values[name.lower()] = value
    if supplied:
        values = supplied_values(values, config.fit._derived_settings)
    return values


def supplied_configuration(config):
    """Copy a config with linked-rule changes restored to their supplied values."""
    result = copy.deepcopy(config)
    for group in dataclasses.fields(result):
        record = getattr(result, group.name)
        values = supplied_values(dataclasses.asdict(record), result.fit._derived_settings)
        for name, value in values.items():
            setattr(record, name, value)
    result.fit._derived_settings = {}
    return result


def merge_fit_settings(*sources):
    """Merge fit inputs, matching cut quantities regardless of letter case.

    Normalize each source before merging so later overrides replace the same
    quantity even when its spelling differs between Python, YAML and SNANA.
    """
    normalized = []
    for source in sources:
        values = dict(source)
        for name in ('cutwin', 'epoch_cutwin'):
            if isinstance(values.get(name), Mapping):
                values[name] = {str(key).lower(): value
                                for key, value in values[name].items()}
        normalized.append(values)
    return merge_settings(*normalized)


def build_config(settings=None, *, config=None):
    """Build a typed configuration from supplied inputs without loading files."""
    if settings is not None and not isinstance(settings, Mapping):
        raise SNANAError('Fit settings must be a mapping')
    values = merge_fit_settings(configuration_values(config) if config else {}, settings or {})
    types, _, groups = declarations(selected_model(values))
    unknown = values.keys() - types.keys()
    if unknown:
        raise SNANAError(f"Unknown fit settings: {', '.join(sorted(unknown))}")
    records = {name: {} for name in get_type_hints(FitConfig)}
    if config is not None:
        for group, owner in get_type_hints(FitConfig).items():
            record = getattr(config, group)
            records[group].update({field.name: copy.deepcopy(getattr(record, field.name))
                                   for field in dataclasses.fields(owner)
                                   if field.name.startswith('_')})
    model_values = {}
    for name, value in values.items():
        value = convert(value, types[name], name)
        if groups[name] == '_model_key_values':
            model_values[name.upper()] = value
        else:
            records[groups[name]][name] = value
    if model_values:
        records['model']['_model_key_values'] = model_values
    result = FitConfig(**{group: owner(**records[group])
                          for group, owner in get_type_hints(FitConfig).items()})
    original = configuration_values(result, supplied=False)
    resolve_settings(result)
    resolved = configuration_values(result, supplied=False)
    result.fit._derived_settings = derived_changes(original, resolved)
    return result


def resolve_settings(config):
    """Apply the fitting rules shared by all input formats to a new config."""
    for name, owner in (('earlylc', EarlyLC), ('require_epochs', RequireEpochs)):
        values = getattr(config.selection, name)
        if values is not None:
            setattr(config.selection, name, dataclasses.asdict(build_dataclass(owner, values)))
    fit, model, photoz = config.fit, config.model, config.photoz
    if fit.minos is not None:
        fit.minos = fit.minos.lower()
        if fit.minos in ('none', 'null'):
            fit.minos = None
    if fit.minos not in (None, 'z', 'all'):
        raise SNANAError("minos must be None, 'z', or 'all'")
    if "lcplot" not in config.output.tables:
        config.output.mxlc = 0
    for name in ('batch_size', 'num_nobs_bins', 'n_fit_iterations'):
        if getattr(fit, name) < 1:
            raise SNANAError(f'{name} must be positive')
    if photoz.scale_nbin_course <= 0:
        raise SNANAError('scale_nbin_course must be positive')
    if photoz.dofit_photoz and photoz.opt_photoz == 0:
        photoz.opt_photoz = 1
    if photoz.opt_photoz > 0:
        photoz.dofit_photoz = True
    if photoz.opt_photoz & 8:
        photoz.opt_photoz |= 4
    if photoz.opt_photoz & 2 and photoz.prior_zerrscale == PRIOR_ZERRSCALE_UNSET:
        photoz.prior_zerrscale = 1.0
    gaussian_mu_prior = photoz.prior_muerrscale is not None
    flat_mu_prior = photoz.prior_dmu_range is not None
    if gaussian_mu_prior and flat_mu_prior:
        raise SNANAError('PRIOR_MUERRSCALE and PRIOR_DMU_RANGE are both distance '
                         'priors; set one of them')
    if (gaussian_mu_prior or flat_mu_prior) and not photoz.dofit_photoz:
        raise SNANAError('the distance priors PRIOR_MUERRSCALE and PRIOR_DMU_RANGE '
                         'apply only to photometric-redshift fits')
    if photoz.dofit_photoz and any(name.upper() == 'DLMAG' and step == 0
                                   for name, step in model.inistp.items()):
        raise SNANAError(
            "DOFIT_PHOTOZ with INISTP_DLMAG=0 requests a cosmology-constrained "
            "amplitude, which is not implemented. Use a free amplitude.",
            function_name="resolve_settings")
    if fit.opt_chi2_sigma not in (0, 1):
        raise SNANAError('opt_chi2_sigma must be 0 or 1')
    if photoz.dofit_photoz:
        fit.opt_chi2_sigma = 1
    if not model.use_model_magerr:
        model.fudge_modelerr_scale = 0.0
        fit.use_offdiag_covariance = False
    if model.lfixpar_all or model.lfixpar_adjust:
        fit.maxiter_round1 = fit.maxiter_round2 = 0
        if model.lfixpar_all:
            fit.skip_grid_search = True
    unknown_bits = config.data.opt_setpkmjd & ~(
        OPTMASK_SETPKMJD_BAZIN | OPTMASK_SETPKMJD_NOABORT | OPTMASK_SETPKMJD_FLUXMAX
        | OPTMASK_SETPKMJD_FLUXMAX2 | OPTMASK_SETPKMJD_FLUXMAX3 | OPTMASK_SETPKMJD_TRIGGER)
    if unknown_bits:
        logging.getLogger(__name__).warning(
            "OPT_SETPKMJD=%s selects unimplemented bits %s; implemented: "
            "1 (Bazin09), 4 (no abort), 8 (maximum flux), 16 (Fmax clump), "
            "32 (Fmax clump weighted by log10 S/N), 64 (MJD_TRIGGER)",
            config.data.opt_setpkmjd, unknown_bits)
    if config.data.has_setpkmjd_trigger() and config.data.photflag_trigger == 0:
        raise SNANAError('photflag_trigger is required when opt_setpkmjd selects MJD_TRIGGER')


def read_cut_window_tokens(tokens):
    """Read a CLI cut window using the same quantity names as SNANA inputs."""
    return {str(name).lower(): value for name, value in mapping_tokens(tokens).items()}


def read_prior_tokens(tokens):
    """Read --prior PARAM DISTRIBUTION coefficient=value ... ."""
    if len(tokens) == 1:
        values = yaml.safe_load(tokens[0])
        if isinstance(values, dict):
            for prior in values.values():
                if prior is not None:
                    parse_prior(prior)
            return values
    if len(tokens) < 2:
        raise SNANAError('--prior needs a parameter and a distribution')
    name, form, *coefficients = tokens
    if form.lower() in ('none', 'null'):
        if coefficients:
            raise SNANAError('Removing a prior takes no coefficients')
        return {name: None}
    values = {'distribution': form}
    for coefficient in coefficients:
        key, sep, value = coefficient.partition('=')
        if not sep or key in values:
            raise SNANAError(f'Invalid prior coefficient {coefficient!r}')
        values[key] = yaml.safe_load(value)
    parse_prior(values)
    return {name: values}


#: the command-line readers of the settings whose tokens are not KEY=VALUE pairs
INPUT_READERS = {'cutwin': read_cut_window_tokens, 'epoch_cutwin': read_cut_window_tokens,
                 'priors': read_prior_tokens}


def argument_parser(model_type=None, *, input_file=False, add_help=True):
    """Build argparse from component/model declarations, without model I/O."""
    parser = argparse.ArgumentParser(description='Fit supernova light curves.',
                                     allow_abbrev=False, add_help=add_help)
    if input_file:
        parser.add_argument('input_file', nargs='?')
    parser.add_argument('--verbose', '-v', action='store_true')
    types, owners, _ = declarations(model_type)
    add_settings(parser, types, owners)
    return parser


def command_line_input(argv):
    """The input file ``argv`` names, whether it asks for verbose logging,
    and its other tokens, the overrides :func:`parse_fit_input_file` takes."""
    types, owners, _ = declarations()
    selectors, remaining = bootstrap_settings(
        argv, types, owners, *model_input_declarations('fit'), readers=INPUT_READERS)
    initial, _ = argument_parser(input_file=True, add_help=False).parse_known_args(remaining)
    if initial.input_file is None:
        parser = argument_parser(selectors.get('model_type'), input_file=True)
        parser.parse_known_args(argv)  # --help lists the options and exits here
        parser.error('an input file is required')
    tokens = list(argv)
    tokens.remove(initial.input_file)
    return initial.input_file, initial.verbose, tokens


def command_settings(tokens, file_settings):
    """Resolve selectors first, then parse every selected-model option."""
    selectors = argument_parser(add_help=False)
    selected, _ = selectors.parse_known_args(tokens)
    model_type = selected_model(merge_settings(file_settings, vars(selected)))
    parser = argument_parser(model_type)
    args = parser.parse_args(tokens)
    types, _, _ = declarations(model_type)
    return parsed_settings(args, types, readers=INPUT_READERS)


def fitted_model_type(config):
    """The fitted model's type, from MODEL_TYPE or FITMODEL_NAME."""
    if config.model.fitmodel_name is None:
        raise SNANAError(
            "FITMODEL_NAME is required in the input file",
            function_name="FITMODEL_NAME")
    model_type = selected_model({'model_type': config.model.model_type,
                                 'fitmodel_name': config.model.fitmodel_name})
    if model_type is None:
        raise SNANAError(
            f"Cannot infer model type from FITMODEL_NAME "
            f"'{config.model.fitmodel_name}'. Set MODEL_TYPE "
            f"({', '.join(MODEL_REGISTRY)}).",
            function_name="FITMODEL_NAME")
    return model_type
