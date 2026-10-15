"""Two-stage parser for SALT2mu/BBC input.

Stage 1 tokenises the input file and command-line ``key=value`` overrides without
interpreting anything, so ``model_lcfit`` and ``double_precision`` can be read
before the rest. Stage 2 translates supplied settings with the model class
in hand, resolving its cut shorthands, its own BBC keys and its
``p<N>``/``u<N>`` slots. Defaults and scientific rules are applied after
the input file and override layers have been merged.

Numbered ``p``/``u``/``s`` slots are a wire format confined to this module: they
are translated to parameter names against ``BBC_NUISANCE_SPEC`` and the
slots the model's ``nuisance_spec`` declares, and never appear above the parser.
"""

import logging
import re
from pathlib import Path
from collections.abc import Mapping
from dataclasses import fields
from typing import Dict, List, Optional, Tuple

from jax_snlc_core.utils import snana_log as slog
from jax_snlc_core.utils.config_tracking import refused
from jax_snlc_core.utils.error_handling import SNANAError
from jax_snlc_core.utils.precision import DEFAULT_DOUBLE_PRECISION
from jax_snlc_core.utils.input_settings import merge_settings

from jax_snlc_core.models import (model_input_keys, parse_model_input_value,
                                  standardised_model)
from .config import BBCConfig, BBC_NUISANCE_SPEC
from jax_snlc_core.io.survey_def import load_survey_ids

logger = logging.getLogger(__name__)

_OPT_CCPRIOR_DEFAULT = 5      # GAUSS+UPDATE, SALT2mu.c:489

# slot -> name for the reference cosmology and sigint (FITPARNAMES_DEFAULT, SALT2mu.c:1446)
_SLOT_NAMES: Dict[int, str] = {
    9: "Omega_L", 10: "Omega_k", 11: "w0", 12: "wa", 14: "sigint",
}

# Live synonyms: both spellings appear in SNANA's ppar; accept either.
_ALIASES: Dict[str, str] = {
    "file": "datafile",
    "bins": "nzbin",
    "zVARNAME": "varname_z",
    "sigmb_biascor": "sigint_biascor",
    "fieldsplit_biascor": "fieldgroup_biascor",
    "fieldGroup_biascor": "fieldgroup_biascor",
    "fieldList_biascor": "fieldlist_biascor",
    "surveyGroup_biascor": "surveygroup_biascor",
    "simfile_bias": "simfile_biascor",
    # all set INPUTS.sigmB; SNANA's own sigint= is a dead key (SALT2mu.c:19972)
    "sig1": "sigint", "sigmb": "sigint", "sigmB": "sigint",
    "sig1fit": "fitflag_sigmb", "sig1tol": "redchi2_tol",
    "prescale_sim1a": "prescale_simIa",
    "catfile_out": "cat_file_out",
    "dumpflag_nobiascor": "ndump_nobiascor",
    "simdata_prescale": "prescale_simdata",
    "sntype": "sntype_select",
}

# the keys whose spelling differs from the BBCConfig field they set
_FIELD_KEYS: Dict[str, str] = {
    "varname_pia": "varname_pIa", "prescale_simia": "prescale_simIa",
    "select_trueia": "select_trueIa",
}

# BBCConfig fields set by their own code below, or by no key at all; every
# other field is its key's value read as the field's type
_BESPOKE = frozenset({
    "model_lcfit", "double_precision", "_u_flags", "_model_inputs", "_unblind",
    "cutwin", "cutwin_disable", "lcpar_range", "chi2max_fitwgt0",
    "mucovscale_bins", "ccprior_bins",
    "chi2max_surveys", "nuisance_val", "nuisance_float", "nuisance_onebin",
    "omega_l", "omega_k", "w0", "wa", "powzbin", "znhalf",
    "blindpar", "force_pia", "use_h11", "nmax", "prescale_biascor",
    "zwin_vpec_check", "surveylist_nobiascor", "type_list_probcc0",
    "idsurvey_list_probcc0", "cat_only", "cat_prescale",
})

_COSMO_ALIASES: Dict[str, str] = {
    "OL": "omega_l", "Ok": "omega_k", "w": "w0", "wa": "wa",
}

# p<N> for slots 9-12 sets the reference cosmology, not nuisance_val
_COSMO_SLOTS: Dict[str, str] = {
    "Omega_L": "omega_l", "Omega_k": "omega_k", "w0": "w0", "wa": "wa",
}

_P_SLOT = re.compile(r"^p(\d+)$")
_U_SLOT = re.compile(r"^u(\d+)$")
_S_SLOT = re.compile(r"^s(\d+)$")


def parse_bbc_input(path: Optional[str], overrides=(), *, settings=None) -> BBCConfig:
    """Read a file, SNANA overrides and typed settings, in that order.

    ``overrides`` uses SALT2mu keys and values, either as command-line tokens
    or a mapping. ``settings`` uses the typed Python/YAML field names.
    """
    return build_bbc_config(path, [(overrides, settings or {})])


def _override_tokens(overrides):
    """Encode a SNANA mapping using the same assignment syntax as its CLI."""
    if not isinstance(overrides, Mapping):
        return list(overrides)

    def scalar(value):
        return str(int(value)) if isinstance(value, bool) else str(value)

    tokens = []
    for key, value in overrides.items():
        cut = key.split()[0] == 'CUTWIN'
        if isinstance(value, Mapping):
            raise SNANAError(f'{key}: SNANA overrides do not take mappings; use settings')
        if isinstance(value, (list, tuple)):
            value = (' ' if cut else ',').join(scalar(item) for item in value)
        else:
            value = scalar(value)
        tokens.append(f'{key} {value}' if cut else f'{key}={value}')
    return tokens


def build_bbc_config(path, layers):
    """Merge ordered SNANA/typed input layers, then prepare one configuration.

    A pipeline supplies a nominal layer followed by a MUOPT layer. Each
    layer retains its input syntax; values never determine which reader runs.
    """
    from .inputs import build_config, command_settings, merge_named_settings
    from jax_snlc_core.utils.input_settings import read_settings

    text = '' if path is None else Path(path).read_text()
    text = re.split(r'(?m)^\s*#END_YAML\s*$', text, maxsplit=1)[-1]
    is_yaml = any(re.match(r'^[a-zA-Z_][\w]*\s*:', line) for line in text.splitlines())
    values = read_settings(path, text=text) if is_yaml else {}
    file_raw, file_cuts, file_off, file_repeated = _read_words(
        [] if is_yaml else _input_words(text))
    model_name = values.get('model_lcfit', file_raw.get('model_lcfit', BBCConfig.model_lcfit))
    parsed_layers = []
    for overrides, settings in layers:
        tokens = _override_tokens(overrides)
        selectors = {'model_lcfit': model_name}
        selectors.update(word.split('=', 1) for word in tokens if word.startswith('model_lcfit='))
        named, tokens = command_settings(tokens, merge_settings(selectors, settings))
        named = merge_settings(named, settings)
        raw, cuts, off, repeated = _read_words([word for token in tokens for word in token.split()])
        model_name = named.get('model_lcfit', raw.get('model_lcfit', model_name))
        parsed_layers.append((raw, cuts, off, repeated, named))
    model = standardised_model(model_name)
    values['model_lcfit'] = model_name
    if is_yaml:
        values = merge_named_settings({}, values)
    else:
        values = _read_snana_settings(list(file_raw.items()), file_cuts, file_off,
                                      file_repeated, set(), model, values)
    for raw, cuts, off, repeated, named in parsed_layers:
        values = _read_snana_settings(list(raw.items()), cuts, off, repeated,
                                      set(raw), model, values)
        values = merge_named_settings(values, named)
    flags = values.pop('_u_flags', {})
    return build_config(values, _snana_flags=flags)


def _double_precision(raw: Dict[str, str]) -> bool:
    """The floating-point frame the input asks for. A jax-only key."""
    val = raw.get("double_precision", "")
    if not val:
        return DEFAULT_DOUBLE_PRECISION
    try:
        return bool(int(val))
    except ValueError:
        raise SNANAError(f"double_precision={val} is not a valid int",
                         function_name="parse_bbc_input") from None


def _input_words(text: str) -> List[str]:
    """The input file's whitespace-separated words, comments stripped."""
    return [w for line in text.splitlines() for w in line.split("#", 1)[0].split()]


def _read_words(words: List[str]):
    """Stage 1: words -> (key=value dict, CUTWIN list, CUTWIN NONE flag, keys
    given more than once). No validation. ``CUTWIN var lo hi`` takes four words
    and a later CUTWIN on the same variable replaces the earlier
    (parseLine_SELECT_VAR, SALT2mu.c:20983, :21128); ``CUTWIN NONE`` disables
    the cuts (:21012). The qualified forms ``CUTWIN(...)``,
    ``CUTWIN_IDSAMPLE(...)`` and ``CUTWIN_SURVEY(...)`` are not implemented."""
    raw: Dict[str, str] = {}
    cutwins: List[Tuple[str, float, float]] = []
    disable = False
    assigned = set()
    repeated = set()
    i = 0
    while i < len(words):
        word = words[i]
        if word == "CUTWIN":
            if words[i + 1:i + 2] == ["NONE"]:
                disable = True
                i += 2
                continue
            if len(words) < i + 4:
                raise SNANAError("CUTWIN needs a variable and two limits: "
                                 f"'{' '.join(words[i:])}'",
                                 function_name="parse_bbc_input")
            var, lo, hi = words[i + 1:i + 4]
            cutwins = [c for c in cutwins if c[0] != var]
            cutwins.append((var, float(lo), float(hi)))
            i += 4
            continue
        if word.startswith("CUTWIN"):
            raise SNANAError(f"{word} is not implemented; only 'CUTWIN var lo hi' "
                             "and 'CUTWIN NONE' are",
                             function_name="parse_bbc_input")
        # a word with no '=' records an empty value, which take() treats as unset
        key, eq, val = word.partition("=")
        if eq:
            if key in assigned:
                repeated.add(key)
            assigned.add(key)
        raw[key] = val
        i += 1
    return raw, cutwins, disable, repeated


def _read_snana_settings(items, cutwins, cutwin_disable, repeated,
                         cli_keys, model, values):
    """Apply supplied SNANA values without adding defaults or resolving rules."""
    cfg = merge_settings(values)
    if cutwins:
        replaced = {cut[0] for cut in cutwins}
        cfg['cutwin'] = [cut for cut in cfg.get('cutwin', []) if cut[0] not in replaced] + list(cutwins)
    if cutwin_disable:
        cfg['cutwin_disable'] = True

    nuis_names = set(model.nuisance_spec)
    slot_names = {**_SLOT_NAMES,
                  **{slot: name for name, (_, _, _, slot)
                     in BBC_NUISANCE_SPEC.items()},
                  **{slot: name for name, (_, _, _, slot)
                     in model.nuisance_spec.items()}}

    # refuse before aliasing, so the refusal names the key as typed
    keyvals: Dict[str, str] = {}
    for key, val in items:
        if key in _REJECTED_KEYS:
            raise SNANAError(f"{key}: {_REJECTED_KEYS[key]}",
                             function_name="parse_bbc_input")
        keyvals[_ALIASES.get(key, key)] = val

    if 'chi2max' in keyvals and cfg.get('chi2max_surveys'):
        raise SNANAError('Cannot define chi2max after chi2max(SURVEYLIST)',
                         function_name='parse_bbc_input')

    used_names = set()
    valueless = set()

    def take(name, cast=str, default=None):
        val = keyvals.get(name, "")
        if not val:
            if name in keyvals:
                used_names.add(name)
                valueless.add(name)
            return default
        used_names.add(name)
        if cast is str:
            # remove_quote (sntools.c:6590) drops every quote, not just a surrounding pair.
            val = val.replace("'", "").replace('"', "")
        try:
            return cast(val)
        except ValueError:
            raise SNANAError(f"{name}={val} is not a valid {cast.__name__}",
                             function_name="parse_bbc_input") from None

    used_names.add("model_lcfit")
    used_names.add("double_precision")
    if 'double_precision' in keyvals:
        cfg['double_precision'] = _double_precision(keyvals)
    if _split_list(take("datafile_override", default="")):
        raise SNANAError("datafile_override is not implemented",
                         function_name="parse_bbc_input")

    for f in fields(BBCConfig):
        if f.name.startswith("_") or f.name in _BESPOKE:
            continue
        cast, convert = _READ_AS[f.type]
        value = take(_FIELD_KEYS.get(f.name, f.name), cast)
        if value is not None:
            cfg[f.name] = convert(value) if convert else value

    pz = _split_list(take("powzbin", str, ""))
    if pz:
        cfg['powzbin'] = float(pz[0])
        # Each SNANA powzbin assignment replaces both components.
        cfg['znhalf'] = float(pz[1]) if len(pz) > 1 else BBCConfig.znhalf
    # blindpar is indexed per fit-parameter slot; SNANA accepts only 9 (OL) and 11
    # (w0), and aborts on a bare blindpar= (parse_blindpar, SALT2mu.c:20556).
    for slot in (9, 11):
        amp_phase = _split_list(take(f"blindpar{slot}", str, ""))
        if amp_phase:
            cfg.setdefault('blindpar', {})[slot] = (float(amp_phase[0]), float(amp_phase[1]))

    if 'surveylist_nobiascor' in keyvals:
        cfg['surveylist_nobiascor'] = [v for v in _split_list(
            take("surveylist_nobiascor", str, "")) if v.upper() != "NONE"]

    # only GAUSS+UPDATE (SALT2mu.c:484-489)
    opt_ccprior = take("opt_ccprior", int, 0)
    if opt_ccprior not in (0, _OPT_CCPRIOR_DEFAULT):
        raise SNANAError(
            f"opt_ccprior={opt_ccprior} is not implemented; only "
            f"{_OPT_CCPRIOR_DEFAULT} (GAUSS+UPDATE) is supported",
            function_name="parse_bbc_input")
    zwin = _floats(take("zwin_vpec_check", str, ""))
    if zwin:
        if len(zwin) != 2:
            raise SNANAError(
                f"zwin_vpec_check takes zmin,zmax; got {len(zwin)} value(s)",
                function_name="parse_bbc_input")
        cfg['zwin_vpec_check'] = (zwin[0], zwin[1])
    if "force_pIa" in keyvals:                      # 'perfect' -> -1 (SALT2mu.c:19722)
        used_names.add("force_pIa")
        fp = keyvals["force_pIa"].strip()
        cfg['force_pia'] = -1.0 if fp.lower() == "perfect" else float(fp)
    # force_probcc0: spec-confirmed SNe -> pIa=1 (SALT2mu.c:19731/19734/19736)
    if 'type_list_probcc0' in keyvals:
        cfg['type_list_probcc0'] = [int(t) for t in _split_list(take("type_list_probcc0", default=""))]
    ids = _split_list(take("idsurvey_list_probcc0", default=""))
    if ids:
        name_to_id, _ = load_survey_ids()
        cfg['idsurvey_list_probcc0'] = [
            int(s) if s.lstrip("-").isdigit() else name_to_id[s] for s in ids]

    _resolve_chi2max_options(keyvals, cfg, used_names)
    nmax = take("nmax", str, "")
    if nmax:
        if "(" in nmax:
            raise SNANAError(
                "per-survey/field nmax is not implemented; only a bare total is",
                function_name="parse_bbc_input")
        parts = _split_list(nmax)
        if len(parts) != 1:
            raise SNANAError(f"cannot parse nmax={nmax}",
                             function_name="parse_bbc_input")
        cfg['nmax'] = int(parts[0])
    ps = take("prescale_biascor", str, "")
    if ps:
        parts = _split_list(ps)
        if len(parts) != 2:
            raise SNANAError("prescale_biascor needs <subset>,<prescale>",
                             function_name="parse_bbc_input")
        subset, prescale = int(parts[0]), int(parts[1])
        if subset >= prescale:                       # SALT2mu.c:20616
            raise SNANAError(
                f"prescale_biascor subset={subset} must be < prescale={prescale}",
                function_name="parse_bbc_input")
        cfg['prescale_biascor'] = (subset, prescale)

    for alias, attr in _COSMO_ALIASES.items():
        if alias in keyvals:
            used_names.add(alias)
            cfg[attr] = float(keyvals[alias])
    _resolve_nuisance(keyvals, cfg, nuis_names, slot_names, used_names)
    _resolve_named_coordinates(items, cfg, model, used_names)
    cfg.update(_model_key_values(keyvals, model, used_names))

    cfg.setdefault('_u_flags', {}).update({k: int(v) for k, v in keyvals.items() if _U_SLOT.match(k)})

    _check_repeated_and_cli_keys(repeated, cli_keys, used_names, valueless)
    _warn_unrecognized(keyvals, used_names)
    if valueless:
        slog.warn(logger, "BBC", "SALT2mu key(s) given with no value, ignored: "
                  + ", ".join(sorted(valueless)))
    return cfg


def _resolve_chi2max_options(keyvals, cfg, used_names) -> None:
    """chi2max(FITWGT0)= deweights instead of rejecting; chi2max(SURVEYLIST)=
    sets per-survey thresholds, with unlisted surveys keeping the global value
    (parse_chi2max, SALT2mu.c:20428-20536)."""
    keys = list(keyvals)
    for key in keys:
        if not (key.startswith("chi2max(") and key.endswith(")")):
            continue
        used_names.add(key)
        opt = key[len("chi2max("):-1]
        val = float(keyvals[key])
        if opt == "FITWGT0":
            cfg['chi2max'] = val
            cfg['chi2max_fitwgt0'] = True
            continue
        surveys = [s.strip() for s in opt.split(",")]
        if "FITWGT0" in surveys:
            raise SNANAError(
                "Cannot mix FITWGT0 option with SURVEY",
                function_name="parse_bbc_input")
        if "chi2max" in keys and keys.index("chi2max") > keys.index(key):
            raise SNANAError(
                "Cannot define chi2max after chi2max(SURVEYLIST)",
                function_name="parse_bbc_input")
        for survey in surveys:
            cfg.setdefault('chi2max_surveys', {})[survey] = val


def _model_key_values(keyvals, model, used_names) -> Dict[str, object]:
    """The supplied values of the model's BBC keys, by attribute name, typed
    as the model declares them. SALT2mu keys are lowercase, so the match is
    case-insensitive."""
    given = {key.lower(): key for key in keyvals}
    values = {}
    for key, spec in model_input_keys(model, consumer='bbc').items():
        if key.lower() in given:
            used_names.add(given[key.lower()])
            values[spec.attribute] = parse_model_input_value(
                key, spec, keyvals[given[key.lower()]])
    return values


def _check_slot(key: str, name, settable) -> None:
    """A slot key must name a parameter this fit has: SNANA reads any p<N>
    into parval[N], where an unused slot does nothing, silently."""
    if name is None:
        raise SNANAError(f"{key}: no fit parameter has number {key[1:]} for "
                         f"this model", function_name="parse_bbc_input")
    if name not in settable and name not in _COSMO_SLOTS:
        raise SNANAError(f"{key}: {name} is not a parameter this fit sets",
                         function_name="parse_bbc_input")


# the spellings SALT2mu still reads for the renamed slopes (SALT2mu.c:19960-19977)
_NUISANCE_ALIASES = {"beta1": "dbeta_dz", "alphaHost": "dalpha_dlogm",
                     "betaHost": "dbeta_dlogm"}


def _resolve_nuisance(keyvals, cfg, nuis_names, slot_names, used_names) -> None:
    settable = nuis_names | set(BBC_NUISANCE_SPEC)
    for key, val in keyvals.items():
        name = None
        m = _P_SLOT.match(key)
        if m:
            name = slot_names.get(int(m.group(1)))
            used_names.add(key)
            _check_slot(key, name, settable)
        elif key in settable or _NUISANCE_ALIASES.get(key) in settable:
            # SNANA also accepts the core params by name (scalePCC -> parval[13],
            # SALT2mu.c:19970), not only through the p<N> slots.
            name = _NUISANCE_ALIASES.get(key, key)
            used_names.add(key)
        if name is None:
            continue
        if name in _COSMO_SLOTS:
            cfg[_COSMO_SLOTS[name]] = float(val)
        elif name in settable:
            cfg.setdefault('nuisance_val', {})[name] = float(val)
        # names not owned by this model (e.g. gamma0 for BayeSN) fall through to
        # the unrecognized-key check, which names the model.

    for key, val in keyvals.items():
        m = _U_SLOT.match(key)
        if not m:
            continue
        name = slot_names.get(int(m.group(1)))
        used_names.add(key)
        _check_slot(key, name, settable)
        try:
            flag = int(val)
        except ValueError:
            raise SNANAError(f"{key}={val} is not a valid int",
                             function_name="parse_bbc_input") from None
        if name in _COSMO_SLOTS and flag != 0:
            raise SNANAError(
                f"{key}=1 floats {name}, but jax holds the reference cosmology fixed",
                function_name="parse_bbc_input")
        if name in settable:
            cfg.setdefault('nuisance_float', {})[name] = flag != 0
            if flag == 0:
                cfg.setdefault('nuisance_onebin', {}).pop(name, None)
            else:
                cfg.setdefault('nuisance_onebin', {})[name] = flag == 3
    # s<N> step sizes have no Newton-LM analogue; recognised, ignored.
    for key in keyvals:
        if _S_SLOT.match(key):
            used_names.add(key)


def _resolve_named_coordinates(items, cfg, model, used_names):
    """Resolve file then CLI assignments by named target, including aliases."""
    for key, value in items:
        # These structural bounds were already read as BBCConfig fields.
        if key in ("logmass_min", "logmass_max"):
            continue
        try:
            if key in model.cut_aliases:
                var, side = model.cut_aliases[key]
                prev = cfg.get('lcpar_range', {}).get(var, (None, None))
                val = float(value)
                cfg.setdefault('lcpar_range', {})[var] = (val, prev[1]) if side == "lo" else (prev[0], val)
            elif key.startswith("lcpar_range."):
                var = key.split(".", 1)[1]
                bounds = tuple(float(v) for v in value.split(","))
                if not var or len(bounds) != 2:
                    raise ValueError()
                cfg.setdefault('lcpar_range', {})[var] = bounds
            else:
                target = getattr(model, "bbc_input_aliases", {}).get(key)
                if target is None and key.startswith(("mucovscale_bins.", "ccprior_bins.")):
                    target = key.split(".", 1)
                if target is None:
                    continue
                field, var = target
                if not var:
                    raise ValueError()
                cfg.setdefault(field, {})[var] = int(value)
        except ValueError:
            raise SNANAError(f"Invalid named BBC setting {key}={value}",
                             function_name="parse_bbc_input") from None
        used_names.add(key)


# refused keys, matched exactly
_REJECTED_KEYS = {
    **refused('set intrinsic scatter with sig1= (or sigmB=)',
              "sigint"),
    **refused('different standardisation slopes above and below a stretch break are not supported',
              "alpha1"),
    **refused('intrinsic stretch/colour scatter and its correlations with magnitude scatter are not supported',
              "sigx1", "sigc", "sig2", "sig3", "xi01", "xi0c", "xi1c"),
    **refused('use p7 and u7 for the host-mass split',
              "host_logmass_split"),
    **refused('use zpecerr for the peculiar-velocity uncertainty in redshift units', "pecv"),
    **refused("only the numbered forms blindpar9 and blindpar11 are supported",
              "blindpar"),
    # Event synchronisation prepares pass-1 acceptance without fitting. If
    # opt_chi2max >= 2 is implemented, that pass must fit before selecting.
    **refused('reapplying chi2max selection with recalculated event chi-squares is not supported',
              "opt_chi2max"),
    # ZPOLY_*: SALT2mu uses whichever row was read last (SALT2mu.c:9276)
    **refused('redshift- and survey-dependent intrinsic-scatter covariance is not supported',
              "ZPOLY_sigmB", "ZPOLY_sigx1", "ZPOLY_sigc", "ZPOLY_xi01",
              "ZPOLY_xi0c", "ZPOLY_xi1c"),
    **refused('split-sample execution is configured through the Pippin interface, not BBC inputs',
              "NSPLITRAN", "JOBID_SPLITRAN"),
    **refused('SNANA diagnostic output and tracing controls are not supported',
              "NDUMPLOG", "debug_flag", "debug_malloc", "debug_mucovscale",
              "snid_mucovdump", "cidlist_debug_biascor"),
    **refused('the SNANA population-reweighting subprocess interface is not supported',
              "SUBPROCESS_FILES", "SUBPROCESS_HELP", "SUBPROCESS_ISEED",
              "SUBPROCESS_OPTMASK", "SUBPROCESS_SIMREF_FILE",
              "SUBPROCESS_CID_REWGT_DUMP", "SUBPROCESS_NEVT_SIM_PRESCALE",
              "SUBPROCESS_STDOUT_CLOBBER", "SUBPROCESS_VARNAMES_GENPDF"),
    **refused('switches that restore earlier SNANA calculations are not supported',
              "restore_bug_mucovadd", "restore_bug2_mucovadd",
              "restore_bug_mucovscale", "restore_bug_mumodel_zhel",
              "restore_bug_muzerr", "restore_bug_sigint0",
              "restore_bug_zmax_biascor", "restore_des5yr", "restore_sigz"),
    **refused('MINUIT profile uncertainties are not supported in BBC',
              "minos", "minos2"),
    **refused('manual CPU-thread control through BBC inputs is not supported',
              "nthread"),
    **refused('selecting and rewriting a table without standardisation is not supported',
              "cutwin_only"),
}


def _check_repeated_and_cli_keys(repeated, cli_keys, used_names, valueless) -> None:
    """Abort on a key repeated within the file or within the command line
    (uniqueOverlap), and on a command-line key nothing reads
    (override_parFile). chi2max keys may repeat."""
    def used(key):
        return _ALIASES.get(key, key) in used_names

    dup = sorted(k for k in repeated if used(k) and not k.startswith("chi2max"))
    if dup:
        raise SNANAError(f"key(s) given more than once: {', '.join(dup)}",
                         function_name="parse_bbc_input")
    invalid = sorted(k for k in cli_keys
                     if not used(k) or _ALIASES.get(k, k) in valueless)
    if invalid:
        raise SNANAError(f"Invalid command-line override: {', '.join(invalid)}",
                         function_name="parse_bbc_input")


def _warn_unrecognized(keyvals, used_names) -> None:
    """A key SALT2mu does not know is a typo, and an input file carrying one
    should still run, so it warns; the keys jax refuses were refused as
    the input spelt them, before aliasing."""
    unknown = sorted(k for k in keyvals if k not in used_names)
    if unknown:
        slog.warn(logger, "BBC", f"Unrecognized SALT2mu key(s) ignored: {', '.join(unknown)}")


def _split_list(value: str) -> List[str]:
    if not value:
        return []
    return [v for v in value.split(",") if v]


def _split_idlist(value: str) -> List[int]:
    """parse_select_IDLIST (SALT2mu.c:20703): comma- or plus-separated integer IDs;
    mixing the two separators aborts. An empty list selects everything."""
    if not value:
        return []
    if "," in value and "+" in value:
        raise SNANAError(f"cannot mix ',' and '+' separators in '{value}'",
                         function_name="parse_bbc_input")
    sep = "+" if "+" in value else ","
    return [int(v) for v in value.split(sep) if v]


def _floats(value: str) -> List[float]:
    return [float(v) for v in _split_list(value)]


def _float_tuple(value: str) -> Tuple[float, ...]:
    return tuple(_floats(value))


# per field type, the cast take() applies and what then makes it the field's value
_READ_AS = {
    int: (int, None), float: (float, None), str: (str, None), bool: (int, bool),
    List[str]: (str, _split_list), List[float]: (str, _floats),
    List[int]: (str, _split_idlist), Tuple[float, ...]: (str, _float_tuple),
}
