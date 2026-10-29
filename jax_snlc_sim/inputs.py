"""Simulation inputs: Python settings, YAML, SNANA files and command-line overrides.

Three steps, in order:

    read_keys(path)   -> the input's keys, verbatim: includes, repeats, aliases
    build(raw)        -> a SimulationConfig, in one interpretation pass
    prep_ransystpar(cfg) -> the RANSYSTPAR draws, before the counts use them

`parse_snana_input_file` runs all three, with `validate` (the checks SNANA
applies after command-line overrides) after build. The generated counts
need the built resources, so `rate.resolve_counts` runs later, during resource setup.
Defaults live on the dataclass fields and nowhere else: a key the input did
not supply is simply omitted from the constructor call, so the field's own
default stands.
"""

import argparse
import logging
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, get_type_hints

import numpy as np
import yaml

from jax_snlc_core import logging as log
from jax_snlc_core.snana_files import (
    canonical_property_name,
    is_ignorefile,
    parse_mushift,
    parse_prescales,
    resolve_fits_path,
)
from jax_snlc_core.dust import OPT_MWCOLORLAW_NAMES
from jax_snlc_core.input_settings import (UsedTrackingDict, add_settings,
                                          bootstrap_settings as parse_bootstrap,
                                          build_dataclass, convert, dataclass_settings,
                                          merge_settings, parsed_settings, refused, value_type)
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.models import (MODEL_REGISTRY, SimulatedModel, infer_model_type_from_path,
                                  model_input_keys, parse_model_input_value)
from jax_snlc_core.models.registry import model_input_declarations
from jax_snlc_sim.distributions import (GenExpHalfGauss, GenGaussAsym, flat_distribution,
                                        validate_gaussian)
from jax_snlc_sim.genpdf import load_all_genpdf
from jax_snlc_sim.population import (HOST_DUST_PARAMETERS, Population, configured_distributions,
                                     declared_population, population_spellings, targets)
from jax_snlc_sim.rate import RateModifiers, neutral_rate_mods
from jax_snlc_sim.systematics import pick_wildcard_file, prep_ransystpar
from jax_snlc_sim.simulation import (HOSTLIB_MSKOPT_REWRITE_BITS,
                                     Cosmology, Cuts, EpochsTimeCut, Host, Instrument, Lensing,
                                     Model, MWDust, NumericalSettings, Output,
                                     Rates, Run, SimulationConfig, Spectra,
                                     SubPopulation, Survey, field_default)
from jax_snlc_sim.spectra import (HOST_CONTAM_MIN, parse_take_spectrum_entry,
                                  prepare_spectrum_requests)
from jax_snlc_sim.zvariation import load_variation, load_zvariation, spelled_by_name


def declarations(model_type=None):
    types, owners, groups = {}, {}, {}
    groups_by_owner = get_type_hints(SimulationConfig)
    # Scientific settings own shared names; execution overrides use them too.
    for group in sorted(groups_by_owner, key=lambda name: name == 'numerical'):
        owner = groups_by_owner[group]
        declared, _ = dataclass_settings(owner)
        for name, annotation in declared.items():
            if name in types:
                if value_type(types[name]) != value_type(annotation):
                    raise SNANAError(f'Simulation setting {name} has conflicting declarations')
                continue
            types[name], owners[name], groups[name] = annotation, owner, group
    if model_type:
        if model_type not in MODEL_REGISTRY:
            raise SNANAError(f'Unknown simulation model type {model_type!r}')
        model = MODEL_REGISTRY[model_type]
        for spec in model_input_keys(model, 'sim').values():
            name = spec.attribute
            if name not in types:
                types[name] = get_type_hints(model, include_extras=True)[name]
                owners[name], groups[name] = model, None
    return types, owners, groups


def translated_settings(settings, raw=None):
    """Merge shared inputs, resolve model selection, then read model-owned inputs.

    Random-systematic file choices must precede model declaration discovery
    and population construction, whichever input syntax supplied them.
    """
    raw = {} if raw is None else raw
    direct = {}
    common_types, common_owners, common_groups = declarations()
    canonical = _key_aliases()

    def translate(name, value, types, owners, groups):
        value = convert(value, types[name], name)
        # Typed switches carry a value; SNANA presence flags may also carry
        # column lists, which must survive a switch override.
        if groups[name] == 'numerical' or (groups[name] is not None and isinstance(value, bool)):
            direct[name] = value
            return
        aliases = getattr(owners[name], 'input_names', {})
        keys = aliases.get(name, [name.upper()])
        # An explicit override replaces every SNANA spelling of this setting.
        for key in keys:
            raw.pop(canonical.get(key, key), None)
        key = canonical.get(keys[0], keys[0])
        if name == 'ransystpar':
            if not value:
                for key in list(raw):
                    if key.startswith('RANSYSTPAR_'):
                        del raw[key]
            raw.update({'RANSYSTPAR_' + key: item for key, item in value.items()})
        elif key in FILTER_ROW_KEYS:
            raw[f'{key}_ROWS'] = list(value.items())
        elif key == 'GENMAG_SMEAR_FILTER':
            raw[key] = list(value.items())
        elif name in aliases:
            raw[key] = value
        elif groups[name] is None or isinstance(value, (str, int, float, bool)):
            raw[name.upper()] = value
        elif name in ('populations', 'subpops'):
            raw[name.upper()] = value
        else:
            direct[name] = value

    for name, value in settings.items():
        if name in common_types:
            translate(name, value, common_types, common_owners, common_groups)
    _resolve_ransystpar_wildcards(raw)
    model_type = infer_model_type_from_path(raw.get('GENMODEL', ''))
    if model_type not in MODEL_REGISTRY:
        raise SNANAError(f'genmodel must name a registered simulation model; got {raw.get("GENMODEL")!r}')
    types, owners, groups = declarations(model_type)
    unknown = set(settings) - set(types)
    if unknown:
        raise SNANAError(f'Unknown simulation setting(s): {sorted(unknown)}')
    for name, value in settings.items():
        if name not in common_types:
            translate(name, value, types, owners, groups)
    if 'populations' in settings:
        model = MODEL_REGISTRY[model_type]
        raw['POPULATIONS'] = configured_distributions(
            settings['populations'], model, model.structured_input(raw))
    return raw, direct, groups


def build_config(settings):
    """Construct and validate a simulation without loading survey resources."""
    raw, direct, groups = translated_settings(settings)
    grouped = {}
    for name, value in direct.items():
        grouped.setdefault(groups[name], {})[name] = value
    cfg = build_validated(raw, settings=grouped)
    prep_ransystpar(cfg)
    return cfg


def read_input(path):
    """Recognise lowercase YAML mappings; SNANA keeps its own reader."""
    text = Path(path).read_text()
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError as error:
        if any(re.match(r'^\s*[a-z_][a-z0-9_]*\s*:', line) for line in text.splitlines()):
            raise SNANAError(f'Cannot read {path}: {error}') from error
        value = None
    if isinstance(value, dict) and value and all(isinstance(k, str) and k == k.lower() for k in value):
        return dict(value)
    return None


def parse_input(path, overrides=None):
    """Read a file or build Python settings through the same validation."""
    if path is None or isinstance(path, Mapping):
        if overrides is not None and not isinstance(overrides, Mapping):
            raise SNANAError('Python simulation inputs require named settings')
        return build_config(merge_settings(path or {}, overrides or {}))
    settings = read_input(path)
    if settings is None:
        if isinstance(overrides, dict) and all(k == k.lower() for k in overrides):
            return parse_snana_input_file(path, settings=overrides)
        return parse_snana_input_file(path, overrides or ())
    if overrides is not None and not isinstance(overrides, dict):
        extra, snana = command_settings(overrides, settings)
        if snana:
            raise SNANAError(f'Simulation inputs need --options: {snana}')
    else:
        extra = overrides or {}
    cfg = build_config(merge_settings(settings, extra))
    cfg.run._input_file = str(path)
    effective = merge_settings(settings, extra)
    cfg.run._input_lines = [(name, yaml.safe_dump({name: value}, default_flow_style=False, sort_keys=False).rstrip())
                            for name, value in effective.items()]
    return cfg


def command_settings(tokens, settings):
    types, owners, _ = declarations()
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    add_settings(parser, types, owners)
    initial, _ = parser.parse_known_args(tokens)
    selection = merge_settings(settings, parsed_settings(initial, types, readers={}))
    if selection.get('ransystpar'):
        raw, _, _ = translated_settings(selection)
        model_type = infer_model_type_from_path(raw['GENMODEL'])
    else:
        model_type = infer_model_type_from_path(selection.get('genmodel', ''))
    types, owners, _ = declarations(model_type)
    parser = argparse.ArgumentParser(description='Simulate supernova light curves.', allow_abbrev=False)
    add_settings(parser, types, owners)
    supplied, snana = parser.parse_known_args(tokens)
    return parsed_settings(supplied, types, readers={}), snana


def bootstrap_settings(tokens):
    """Read shared and model options before locating the input file."""
    types, owners, _ = declarations()
    return parse_bootstrap(tokens, types, owners, *model_input_declarations('sim'), readers={})


def _resolve_input_and_overrides(args: argparse.Namespace):
    """The input file and the override tokens, whichever CLI form was used.

    With --input, argparse puts the first override token in the positional slot.
    """
    if args.input:
        overrides = list(args.overrides)
        if args.input_file:
            overrides.insert(0, args.input_file)
        return args.input, overrides
    return args.input_file, list(args.overrides)


def parse_cli_args(argv=None) -> argparse.Namespace:
    settings, remaining = bootstrap_settings(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False,
        description="Run the JAX vectorised SNANA simulation pipeline. Needs "
                    "SNDATA_ROOT (kcor files, models, SIMLIBs) from the "
                    "environment; writes "
                    "the data version to ./<GENVERSION>/ in the working "
                    "directory."
    )
    parser.add_argument(
        "input_file", nargs="?", default=None,
        help="Path to the SNANA .input configuration file (e.g., INP_SIMGEN_*.input).",
    )
    parser.add_argument(
        "--input", "-i", default=None,
        help="The input file, as a flag instead of a positional argument.",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="GENVERSION name for the outputs (defaults to GENVERSION from the input file).",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=None,
        help="Alias for --batch-size.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed override (defaults to RANSEED in the input file).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging.",
    )
    args, args.overrides = parser.parse_known_args(remaining)
    args.settings = settings
    return args


def command_line_config(args) -> SimulationConfig:
    """Resolve the CLI input, model options and aliases into simulation settings."""
    input_arg, overrides = _resolve_input_and_overrides(args)
    if not input_arg and any(flag in overrides for flag in ("-h", "--help")):
        command_settings(overrides, args.settings)
    if not input_arg:
        raise SNANAError("No input file: pass it as the first argument, or with --input.")
    snana_input = Path(input_arg).expanduser().resolve()
    if not snana_input.is_file():
        raise SNANAError(f"Input file not found: {snana_input}")

    file_settings = read_input(snana_input)
    if file_settings is None:
        selectors = {'genmodel': read_keys(snana_input).get('GENMODEL', '')}
    else:
        selectors = file_settings
    named_values, overrides = command_settings(overrides, merge_settings(selectors, args.settings))
    named_values = merge_settings(named_values, args.settings)
    aliases = {key: value for key, value in {
        'random_seed': args.seed, 'batch_size': args.chunk_size,
        'genversion': args.output}.items() if value is not None}
    named_values = merge_settings(named_values, aliases)
    if file_settings is not None:
        if overrides:
            raise SNANAError('YAML simulation inputs take named --options')
        snana_config = parse_input(snana_input, named_values)
    else:
        snana_config = parse_snana_input_file(snana_input, overrides, settings=named_values)
    return snana_config


# OPT_MWEBV: which EBV source to use (MWgaldust.h:43-46). 0 off, 1 FILE
# (SIMLIB or data header), 2 SFD98, 3 Schlafly11+PS2013.
OPT_MWEBV_CODES = frozenset({0, 1, 2, 3})

logger = logging.getLogger(__name__)


# key families consumed by prefix
_OPEN_KEY_PREFIXES = (
    'CUTWIN_',
    'RANSYSTPAR_',
    'SEARCHEFF_SPEC_SHIFT(',
    'SEARCHEFF_zHOST_SHIFT(',
    'GENMAG_SMEAR_SCALE(',
    'ATMOSPHERE_',
)

# Repeated global/per-band SNANA input records.
FILTER_ROW_KEYS = (
    'EXPOSURE_TIME_FILTER', 'MJD_TEMPLATE_FILTER',
    'FUDGESHIFT_ZPT_FILTER', 'FUDGESCALE_FLUXERR_FILTER',
    'FUDGESCALE_FLUXERR2_FILTER', 'FUDGESHIFT_LAM_FILTER',
    'FUDGE_MAGERR_FILTER', 'FUDGE_ZPTERR_FILTER',
)

# (canonical, *alternates); alternates are rewritten on the way in
_KEY_ALIAS_GROUPS = (
    ('w0_LAMBDA', 'W0_LAMBDA'),
    ('wa_LAMBDA', 'Wa_LAMBDA'),
    ('KCOR_FILE', 'CALIB_FILE'),
    ('GENRANGE_DECL', 'GENRANGE_DEC'),
    ('HOSTLIB_GENRANGE_DEC', 'HOSTLIB_GENRANGE_DECL'),
    ('GENRANGE_b', 'GENRANGE_GLAT'),
    ('GENSIGMA_PEAKMJD', 'GENSIGMA_SEARCH_PEAKMJD'),
    ('GENMODEL_MSKOPT', 'GENMODEL_OPTMASK'),
    ('OPT_SETPKMJD', 'OPT_SETPEAKMJD'),
    ('PATH_NON1ASED', 'PATH_NONIASED'),
    ('NGEN_SEASON', 'NGEN_UNIT'),
    ('HOSTLIB_SCALE_SERSIC_SIZE', 'HOSTLIB_SCALE_SERSIC'),
    ('SEARCHEFF_SPEC_FILE', 'SEARCHEFF_SPECID_FILE', 'SEARCHEFF_SPECEFF_FILE'),
    ('SEARCHEFF_SPEC_SCALE', 'SEARCHEFF_SPECEFF_SCALE'),
    ('HOSTLIB_STOREPAR', 'HOSTLIB_STOREVAR'),
    ('SEARCHEFF_PIPELINE_EFF_FILE', 'SEARCHEFF_PIPELINE_FILE'),
    ('EXTINC_HOSTGAL', 'COLORLAW_HOSTGAL'),
    ('WV07_REWGT_EXPAV', 'GENAV_REWGT_EXPAV'),
    ('GENAV_WV07', 'WV07_GENAV_FLAG'),
)
_KEY_ALIASES = {alt: canon for canon, *alts in _KEY_ALIAS_GROUPS for alt in alts}


def _key_aliases():
    return {**_KEY_ALIASES,
            **{alt: canon for cls in dict.fromkeys(MODEL_REGISTRY.values())
               if issubclass(cls, SimulatedModel)
               for canon, *alts in cls.INPUT_KEY_ALIASES for alt in alts}}


# keys jax refuses
_REJECTED_KEYS = {
    **refused('the simulation uses a 60-day window and SNR cuts of 5 and 3; select the peak-estimation method with OPT_SETPKMJD',
        'MJDWIN_SETPKMJD', 'SNRCUT_SETPKMJD', 'SNRCUT_BACKUP'),
    **refused('scaling the model error map to generate per-passband scatter is not supported',
        'GENMODEL_ERRSCALE', 'GENMODEL_ERRSCALE_OPT', 'GENMODEL_ERRSCALE_CORRELATION'),
    **refused('the HOSTNOISE_FILE noise prescription is not supported',
        'HOSTNOISE_FILE'),
    **refused('the DES-specific flux-error rescaling prescription is not supported',
        'FUDGEOPT_FLUXERR'),
    **refused('user-defined magnitude-smearing functions are not supported',
        'GENMAG_SMEAR_USRFUN'),
    **refused("overrides a scatter model's internal parameters; "
              "not implemented",
        'GENMAG_SMEARPAR_OVERRIDE'),
    **refused('free-form model argument strings are not supported',
        'GENMODEL_ARGLIST'),
    **refused('rescaling each event to a target peak signal-to-noise ratio is not supported',
        'FUDGE_SNRMAX', 'FUDGE2_SNRMAX'),
    **refused('use OPT_MWEBV',
        'EXTINC_MILKYWAY'),
    **refused('use RV_MWCOLORLAW',
        'RVMW'),
    **refused('use FORMAT_MASK',
        'GEN_SNDATA_SIM'),
    **refused('this switch is not supported',
        'SPECTYPE', 'SMEARFLAG_HOSTGAL'),
    **refused('stopping generation at a target uncertainty on the simulation efficiency is not supported',
        'EFFERR_STOPGEN'),
    **refused('using SIMLIB MAGOBS values in place of model photometry is not supported',
        'USE_SIMLIB_MAGOBS'),
    **refused('an observer-frame phase grid for grid simulations is not supported',
        'GENRANGE_TOBS'),
    **refused('selecting hosts by their normalised photometric-redshift residual is not supported',
        'HOSTLIB_GENRANGE_NSIGZ'),
    **refused('template-based stretch modifications are not supported',
        'STRETCH_TEMPLATE_FILE'),
    **refused('fixing the uniform random draws used for intrinsic scatter is not supported',
        'GENSMEAR_RANFLAT_FIX', 'GENSMEAR_RANFlat_FIX'),
    **refused('flux non-linearity maps are supported in fitting, but not simulation',
        'NONLINEARITY_FILE'),
    **refused('rewriting HOSTLIB catalogues is not supported',
        '+HOSTMAGS', '+HOSTNBR', '+HOSTAPPEND', '+HOSTSELECT',
        'NNBR_WRITE_MAX', 'SEPNBR_MAX', 'MAX_HOSTLIB_SELECT'),
    **refused('LCLIB event rates are not supported',
        'DNDB_SCALE'),
    **refused('diagnostic dumps for selected galaxies, events or phases are not supported',
        'HOSTLIB_GALID_DUMP', 'GENRANGE_DMPEVENT', 'GENRANGE_DMPTREST'),
    **refused('grid simulations are not supported',
        'GRID_FORMAT', 'NGRID_COLORLAW', 'NGRID_COLORPAR', 'NGRID_LOGZ',
        'NGRID_SHAPEPAR', 'NGRID_LUMIPAR', 'NGRID_TREST',),
    **refused('a submit_batch_jobs configuration key, not a simulation setting',
        'BATCH_INFO', 'CONFIG',),
    **refused('switches that restore earlier SNANA calculations are not supported',
        'RESTORE_BUGS_DES3YR', 'RESTORE_BUG_ZHEL', 'RESTORE_DES5YR',
        'RESTORE_WRONG_VPEC',),
    **refused('an output field; use SNTYPE_Ia or SNTYPES_Ia to set the simulated type',
        'SNTYPE',),
    **refused('SNANA diagnostic controls are not supported',
        'DEBUG_FLAG',),
    **refused('SNANA interactive and tracing controls are not supported',
        'CLEARPROMPT', 'TRACE_MAIN', 'UNIT_TEST',),
    **refused('configures LCLIB, which is not implemented',
        'LCLIB_CUTWIN', 'LCLIB_DEBUG_DAYSCALE', 'LCLIB_DEBUG_TOBS_OFFSET',
        'LCLIB_DEBUG_ZERO_TEMPLATE_FLUX', 'LCLIB_DEBUG_FORCE_NREPEAT',),
    **refused("forces each event's host to its SIMLIB entry's GALID; "
              "not implemented",
        'USE_SIMLIB_GALID',),
    **refused('strongly-lensed supernova simulations are not supported',
        'STRONGLENS_FILE',),
    **refused('selection by lensed-image separation relative to the PSF is not supported',
        'NPSFSIGMA_MINSEP_DETECT',),
    **refused('evaluates the model on a coarse grid and interpolates; not implemented',
        'TGRIDSTEP_MODEL_INTERP',),
    **refused('manual configuration of SNANA random streams and random-list positions is not supported',
        'NSTREAM_RAN', 'RANLIST_START_GENSMEAR',),
    # rest-frame-only Tshift (snlc_sim.c:29663)
    **dict.fromkeys(
        (prefix + family
         for prefix in ('GENPEAK_', 'GENMEAN_', 'GENPEAKRANGE_', 'GENSIGMA_',
                        'GENSKEW_', 'GENRANGE_', 'GENGRID_', 'GENPROB2_',
                        'GENPEAK2_', 'GENSIGMA2_')
         for family in ('RISETIME_SHIFT', 'FALLTIME_SHIFT')),
        'independent shifts of rise and fall times are not supported'),
}

_REQUIRED_KEYS = ('SIMLIB_FILE', 'GENMODEL', 'GENFILTERS', 'KCOR_FILE', 'RANSEED')

# CUTWIN_EPOCHS_TRANGE[1] init: no upper bound on TIME_ABOVE_SNRMIN
# (snlc_sim.c:1378)
CUTWIN_EPOCHS_TRANGE_MAX_DEFAULT = field_default(EpochsTimeCut, 'time_above_max')

_COUNT_PREFIXED_KEYS = ('SIMGEN_DUMP', 'SIMGEN_DUMPALL')

# what a bad input value raises: a wrong cast, a short list, or a failed check
_BAD_INPUT = (TypeError, ValueError, IndexError, SNANAError)

# SNANA aborts on these in a sim-input file (snlc_sim.c:3595, :3952, :4966).
_ARGV_ONLY_KEYS = ('SIMGEN_DUMPADD', 'HOSTLIB_MSKOPT_ADD',
                   'NON1A_INDEX_FORCE', 'NONIA_INDEX_FORCE',
                   'REQUIRE_DOCANA')

# Keys SNANA accepts only in a sim-input file, never on the command line
# (parse_input_NON1ASED, snlc_sim.c:3481); each model declares its own.
def _file_only_keys():
    return {key for cls in dict.fromkeys(MODEL_REGISTRY.values())
            if issubclass(cls, SimulatedModel)
            for key in cls.FILE_ONLY_INPUT_KEYS}

# AV-family keys whose presence activates the host-AV profile
_AV_PROFILE_KEYS = ('GENRANGE_AV', 'GENTAU_AV', 'GENSIG_AV', 'GENSIGMA_AV',
                    'GENGAUPEAK_AV', 'GENRATIO_AV', 'GENRATIO_AV0')

_HOST_DUST_KEYS = ('GENRANGE_AV', 'GENTAU_AV', 'GENSIGMA_AV', 'GENRANGE_RV',
                   'GENMEAN_RV', 'GENSIGMA_RV', 'GENAV_WV07', 'WV07_REWGT_EXPAV')

# HOSTLIB knobs that describe a host the run does not have without a catalog.
_HOSTLIB_KEYS_NEEDING_FILE = (
    'HOSTLIB_ABMAG_OFFSET', 'HOSTLIB_ABMAG_FORCE', 'HOSTLIB_FIXRAN_WGT',
    'HOSTLIB_FIXRAN_RADIUS', 'HOSTLIB_FIXRAN_PHI', 'HOSTLIB_MNINTFLUX_SNPOS',
    'HOSTLIB_MXINTFLUX_SNPOS', 'HOSTLIB_SNR_STORE',
    'HOSTLIB_SCALE_SERSIC_SIZE', 'HOSTLIB_FIXSERSIC', 'HOSTLIB_GENRANGE_RA',
    'HOSTLIB_GENRANGE_DEC', 'HOSTLIB_MAXREAD', 'HOSTLIB_VARNAME_ZPHOT',
    'HOSTLIB_GALID_UNIQUE',
)

_UNSUPPORTED_SIMLIB_MSKOPT = {2: "REPEAT_UNTIL_ACCEPT", 512: "IDEAL_GRID"}


def _require(raw: dict, key: str, function_name: str) -> Any:
    """Fetch `key` or raise: SNANA aborts when it is missing, and silently
    defaulting hides a misconfigured input behind a plausible-but-wrong run."""
    if key not in raw:
        raise SNANAError(f"Required key '{key}' missing from SNANA input file",
                         function_name=function_name)
    return raw[key]


def _require_range(raw) -> Tuple[float, float]:
    """GENRANGE_REDSHIFT: a zmin/zmax pair, with no invented fallback."""
    parsed = _parse_range_tuple(raw)
    if parsed is None:
        raise SNANAError(
            f"GENRANGE_REDSHIFT: {raw!r} is not a zmin/zmax pair.",
            function_name="build")
    return parsed


def _pair(raw) -> Tuple[float, float]:
    """A `lo hi` key value as a float pair."""
    vals = raw if isinstance(raw, (list, tuple)) else str(raw).split()
    return (float(vals[0]), float(vals[1]))


def _parse_range_tuple(raw) -> Optional[Tuple[float, float]]:
    """Extract a 2-tuple of floats from a SNANA range-style list value.

    SNANA range keys (GENRANGE_REDSHIFT, GENRANGE_PEAKMJD, GENRANGE_TREST, …)
    arrive as lists of ≥2 elements; the first two are (lo, hi). Returns
    None otherwise: callers decide whether to raise or skip.
    """
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return (float(raw[0]), float(raw[1]))
    return None


def _parse_comma_list(value, cast=float) -> List[Any]:
    """SNANA comma-separated list value -> list of ``cast``.

    Empty and whitespace-only tokens are dropped, so a trailing comma parses.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):     # already split by the reader
        return [cast(v) for v in value]
    return [cast(tok) for tok in str(value).split(',') if tok.strip()]


def _parse_simlib_nskipmjd(raw) -> Tuple[Tuple[int, int], ...]:
    """Parse SIMLIB_NSKIPMJD, e.g. `2(50),4(100)` (snlc_sim.c:6137).

    Returns ((prescale, trestmin), ...). TRESTMIN defaults to -999999 when the
    parenthesised part is absent; SNANA aborts unless TRESTMIN increases.
    """
    if raw is None:
        return ()
    text = str(raw).strip()
    if is_ignorefile(text):
        return ()
    out: List[Tuple[int, int]] = []
    for part in text.split(','):
        part = part.strip()
        if not part:
            continue
        m = re.match(r'^(-?\d+)\s*(?:\(\s*(-?\d+)\s*\))?$', part)
        if not m:
            raise SNANAError(
                f"SIMLIB_NSKIPMJD entry {part!r} is not PRESCALE or "
                f"PRESCALE(TRESTMIN) with integer values.",
                function_name="_parse_simlib_nskipmjd")
        prescale = int(m.group(1))
        trestmin = int(m.group(2)) if m.group(2) is not None else -999999
        if out and trestmin <= out[-1][1]:
            raise SNANAError(
                f"SIMLIB_NSKIPMJD TRESTMIN={trestmin} does not increase "
                f"(previous {out[-1][1]}); check {text!r}.",
                function_name="_parse_simlib_nskipmjd")
        out.append((prescale, trestmin))
    return tuple(out)


def _parse_warp_spectrum(raw) -> List[float]:
    """Polynomial coefficients from typed input or SNANA LAMPOLY(...) syntax."""
    if isinstance(raw, (list, tuple)):
        return [float(c) for c in raw]
    text = str(raw).strip()
    head, _, rest = text.partition('(')
    if head != 'LAMPOLY' or not rest.endswith(')'):
        raise SNANAError(
            f"{text} is an invalid WARP_SPECTRUM argument; expected "
            f"LAMPOLY(a0,a1,...).", function_name="_parse_warp_spectrum")
    return [float(c) for c in rest[:-1].split(',')]


def _parse_cutwin_filter_triplets(raw: dict, key: str, cast=int) -> List[Tuple[Any, Any, str]]:
    """Parse repeatable `KEY: lo hi filters` triplets (snlc_sim.c:4363-4370).

    The reader accumulates occurrences as a list-of-lists.
    """
    out = []
    for t in raw.get(key) or []:
        if not isinstance(t, (list, tuple)) or len(t) < 3:
            raise SNANAError(
                f"{key} expects 'lo hi filters', got {t!r}",
                function_name="_parse_cutwin_filter_triplets")
        out.append((cast(t[0]), cast(t[1]), str(t[2])))
    return out


def _site_sigmas(raw: Dict[str, Any]) -> Dict[str, float]:
    """ATMOSPHERE_SITE_SIGMAS: one key giving temperature, barometric
    pressure and precipitable-water sigmas (snlc_sim.c:5474)."""
    vals = _parse_comma_list(raw.get('ATMOSPHERE_SITE_SIGMAS'))
    names = ('sigma_site_temp', 'sigma_site_bp', 'sigma_site_pwv')
    return {n: float(v) for n, v in zip(names, vals)}


def _wv07_profile(raw: Dict[str, Any]) -> int:
    """GENAV_WV07, which WV07_REWGT_EXPAV alone also turns on for any value
    above -1e-9, including the meaningful 0 (snlc_sim.c:7947-7948)."""
    rewgt = raw.get('WV07_REWGT_EXPAV')
    if rewgt is not None:
        rewgt = float(rewgt[0] if isinstance(rewgt, list) else rewgt)
        if rewgt > -1.0e-9:
            return 1
    return int(raw.get('GENAV_WV07') or 0)


def _parse_single_value(value: str) -> Union[int, float, str, bool]:
    """Parse a single whitespace-delimited token into its natural type."""
    if value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _parse_value(value_str: str) -> Any:
    """Parse a key's value: a scalar, or a list when it has several tokens."""
    value_str = value_str.strip()
    if value_str.startswith('$'):
        return value_str
    if '/' in value_str and ('.' in value_str or value_str.endswith('/')):
        return value_str
    parts = value_str.split()
    if len(parts) == 1:
        return _parse_single_value(parts[0])
    return [_parse_single_value(part) for part in parts]


def read_keys(input_file: Union[str, Path],
              exclude: Sequence[str] = ()) -> UsedTrackingDict:
    """The input file's keys, verbatim.

    Resolves INPUT_FILE_INCLUDE recursively, rewrites alternate spellings to
    their canonical key, and pools the repeatable general keys (TAKE_SPECTRUM,
    DNDZ, …) under their own name; every other key keeps its occurrence
    history, which `raw.all()` exposes. No value is interpreted beyond its
    type.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise SNANAError(f"Input file not found: {input_file}",
                         function_name="read_keys")

    raw = UsedTrackingDict()
    _read_lines(input_path, raw, input_path.parent, exclude=exclude)
    return raw


def _strip_comment(line: str) -> str:
    """One line's content, without its trailing comment. Empty when the line
    carries nothing but whitespace or a comment."""
    return line.split('#', 1)[0].strip()


def _read_lines(path: Path, raw: UsedTrackingDict, input_dir: Path,
                exclude: Sequence[str] = ()) -> None:
    """Read one input file into `raw`, a key per `KEY: VALUE` line.

    SNANA reads words, not lines (parse_input_key_driver walks WORDS[N]), so a
    key's values may run on beneath it, as NON1A_KEYS and SIMGEN_DUMP do with
    their names. A line naming no key therefore continues the one above it.
    """
    pending = None
    with open(path, 'r') as f:
        for line in f:
            line = _strip_comment(line)
            if not line:
                continue
            if ':' in line:
                if pending is not None:
                    _read_line(pending, raw, input_dir, exclude=exclude)
                pending = line
            elif pending is None:
                raise SNANAError(
                    f"{path.name}: '{line}' names no key and follows no key, "
                    f"so there is nothing for it to continue.",
                    function_name="_read_lines")
            else:
                pending = f"{pending} {line}"
    if pending is not None:
        _read_line(pending, raw, input_dir, exclude=exclude)


def _redshift_range(values: Any, raw: UsedTrackingDict, source: str) -> Any:
    """GENRANGE_REDSHIFT, where a negative limit means "keep the current one"
    and only a command-line override may say so (snlc_sim.c:2160).

    The other limit is then applied only if it narrows the range.
    """
    parsed = _parse_range_tuple(values)
    if parsed is None:
        return values
    zmin, zmax = parsed
    if zmin >= 0.0 and zmax >= 0.0:
        return [zmin, zmax]
    if source != 'arg':
        raise SNANAError(
            f"Invalid GENRANGE_REDSHIFT: {zmin} {zmax}. A negative value is "
            f"allowed only for a command-line override.",
            function_name="_read_line")

    current = list(raw.get('GENRANGE_REDSHIFT',
                           field_default(Rates, 'redshift_range')))
    if zmin > 0.0 and zmin > current[0]:
        current[0] = zmin
    if zmax > 0.0 and zmax < current[1]:
        current[1] = zmax
    return current


def _read_line(line: str, raw: UsedTrackingDict, input_dir: Path,
               source: str = 'file', exclude: Sequence[str] = ()) -> None:
    """Read one line into `raw`. `input_dir` resolves relative include paths."""
    line = _strip_comment(line)
    if not line or ':' not in line:
        return
    if source == 'file':
        name = line.split(':', 1)[0].strip()
        if name in _ARGV_ONLY_KEYS:
            raise SNANAError(
                f"'{name}' is a command-line override only; remove it from the "
                f"sim-input file.", function_name="_read_line")

    take_spec = re.match(r'(TAKE_SPECTRUM(?:\([^)]*\))?)\s*:\s*(.*)', line)
    if take_spec:
        # NONE discards the entries read so far (snlc_sim.c:5608)
        raw.record_line('TAKE_SPECTRUM', line)
        if take_spec.group(2).strip().upper() == 'NONE':
            raw['TAKE_SPECTRUM'] = []
            return
        entries = raw.setdefault('TAKE_SPECTRUM', [])
        entries.append(parse_take_spectrum_entry(
            take_spec.group(1).strip(), take_spec.group(2).strip()))
        return
    snr_scale = re.match(r'(HOSTLIB_SNR_SCALE\([^)]+\))\s*:\s*(.*)', line)
    solid_angle = re.match(r'SOLID_ANGLE\(([^)]*)\)\s*:\s*(.*)', line)
    if snr_scale:
        key, value = snr_scale.group(1).strip(), snr_scale.group(2).strip()
    elif solid_angle:
        # SOLID_ANGLE(FIELDLIST) names the fields (snlc_sim.c:4498-4530)
        if raw.get('SIMLIB_FIELDLIST') not in (None, 'ALL'):
            raise SNANAError(
                "Cannot declare FIELDLIST in two places: check the "
                "SOLID_ANGLE and SIMLIB_FIELDLIST keys.",
                function_name="_read_line")
        raw['SIMLIB_FIELDLIST'] = solid_angle.group(1).strip()
        key, value = 'SOLID_ANGLE', solid_angle.group(2).strip()
    else:
        key, value = line.split(':', 1)
        key = _key_aliases().get(key.strip(), key.strip())
        value = value.strip()

    if source == 'arg' and key in _file_only_keys():
        raise SNANAError(
            f"'{key}' belongs in the sim-input file; SNANA does not read it "
            f"from the command line.", function_name="_read_line")

    if key == 'EXCLUDE_INCLUDE_FILE':
        return
    if key in ('INPUT_FILE_INCLUDE', 'INPUT_INCLUDE_FILE'):
        inc_path = Path(os.path.expandvars(value))
        if not inc_path.is_absolute() and not inc_path.exists():
            candidate = input_dir / inc_path
            if candidate.exists():
                inc_path = candidate
        # substring match, as SNANA does (strstr, snlc_sim.c:654)
        if any(pat in str(inc_path) for pat in exclude):
            return
        _read_lines(inc_path, raw, input_dir, exclude=exclude)
        return

    raw.record_line(key, line)
    parsed_value = _parse_value(value)

    # a command-line nominal rate model replaces the file's (snlc_sim.c:2712)
    if source == 'arg' and key in ('DNDZ', 'DNDZ_FILE'):
        raw.pop('DNDZ', None)
        raw.pop('DNDZ_FILE', None)

    # SIMSED_REDCOR/COV must name a parameter pair (snlc_sim.c:6330)
    if (key.startswith(('SIMSED_REDCOR', 'SIMSED_COV'))
            and re.fullmatch(r'SIMSED_(REDCOR|COV)\((\w+),(\w+)\)', key) is None):
        raise SNANAError(
            f"{key} does not name a parameter pair. Both keys take "
            f"exactly two comma-separated SIMSED parameters and no "
            f"spaces, e.g. 'SIMSED_REDCOR(stretch,color): -0.656'.",
            function_name="_read_line")

    if key == 'ZVARIATION_POLY':
        parts = value.split()
        raw.setdefault(key, []).append((parts[0], ' '.join(parts[1:])))

    elif key == 'VARIATION':
        raw.setdefault(key, []).append(value)

    elif key == 'SUBPOP':
        raw.setdefault(key, []).append(value)

    # valid_DNDZ_KEY: the nominal rate, or the peculiar-Ia one
    elif key in ('DNDZ', 'DNDZ_NON1A', 'DNDZ_NONIA', 'DNDZ_PEC1A', 'DNDZ_PECIA'):
        group = 'DNDZ_PEC1A' if key in ('DNDZ_PEC1A', 'DNDZ_PECIA') else 'DNDZ'
        raw.setdefault(group, []).append(parsed_value)

    elif key == 'GENRANGE_REDSHIFT':
        raw[key] = _redshift_range(parsed_value, raw, source)

    elif key.startswith('HOSTLIB_SNR_SCALE'):
        match = re.match(r'HOSTLIB_SNR_SCALE\(([^:]+):([^)]+)\)', key)
        if match:
            raw.setdefault('HOSTLIB_SNR_SCALE', []).append(
                (float(match.group(1)), float(match.group(2)),
                 float(parsed_value)))

    elif key in ('SIMLIB_IDSKIP', 'GENMAG_SMEAR_FILTER',
                 'CUTWIN_NOBS_SATURATE', 'CUTWIN_NOBS_NOSATURATE',
                 'CUTWIN_PEAKMAG_BYFIELD', 'CUTWIN_SNRMAX'):
        raw.setdefault(key, []).append(parsed_value)

    elif key in FILTER_ROW_KEYS:
        # each occurrence applies to its bands (snlc_sim.c:3648-3741)
        raw.setdefault(f'{key}_ROWS', []).append(parsed_value)

    else:
        raw.add_occurrence(key, parsed_value)


def _resolve_ransystpar_wildcards(raw: dict) -> None:
    """RANSYSTPAR_*_WILDCARD picks a file, which parse-time derivations
    (model_type, GENPDF load) then read."""
    pairs = (('RANSYSTPAR_GENMODEL_WILDCARD', 'GENMODEL'),
             ('RANSYSTPAR_GENPDF_FILE_WILDCARD', 'GENPDF_FILE'))
    if not any(key in raw for key, _ in pairs):
        return
    seed_syst = int(raw.get('RANSYSTPAR_RANSEED_SYST', 0))
    seed = seed_syst if seed_syst > 0 else int(raw.get('RANSEED', 0))
    jobid = int(raw.get('JOBID', field_default(Run, 'jobid')))
    njobtot = int(raw.get('NJOBTOT', field_default(Run, 'njobtot')))
    for key, target in pairs:
        if key in raw:
            raw[target] = pick_wildcard_file(
                str(raw[key]), seed, key[len('RANSYSTPAR_'):],
                jobid=jobid, njobtot=njobtot)


def _apply_genperfect(raw: dict) -> None:
    """GENPERFECT: ideal-simulation bitmask (genperfect_override, snlc_sim.c:8582).

    Overrides the raw keys to ideal values before anything reads them. Bit 0
    (=1) sets all bits (MASK=63). Bits: 1 exposure/ZP, 2 mag-smear, 3 MW dust,
    4 host AV, 5 HOSTLIB; the redshift/vpec/peakmjd/RV pins are applied
    whenever GENPERFECT>0.
    """
    gp = int(raw.get('GENPERFECT', 0))
    if gp == 0:
        return
    if gp < 0:
        raise SNANAError(
            "GENPERFECT<0 lists the bit options and exits in SNANA; use "
            "GENPERFECT=1 (all) or a bit sum of 2(exposure) 4(smear) 8(MWdust) "
            "16(hostAV) 32(HOSTLIB).", function_name="_apply_genperfect")
    mask = 63 if (gp & 1) else gp
    if mask & 2:
        raw['SMEARFLAG_ZEROPT'] = 0
        raw['EXPOSURE_TIME'] = 1.0e5
        # per-filter exposures scale by 1e5 too (snlc_sim.c:8725-8731)
        rows = raw.get('EXPOSURE_TIME_FILTER_ROWS')
        if rows:
            scaled_rows = []
            for row in rows:
                parts = ([str(x) for x in row]
                         if isinstance(row, (list, tuple))
                         else str(row).split())
                scaled = ','.join(str(float(v) * 1.0e5)
                                  for v in parts[1].split(','))
                scaled_rows.append([parts[0], scaled])
            raw['EXPOSURE_TIME_FILTER_ROWS'] = scaled_rows
    if mask & 4:
        raw['GENMAG_SMEAR'] = 0.0
        raw['GENMAG_SMEAR_MODELNAME'] = None
        raw['WEAKLENS_PROBMAP_FILE'] = None
        raw['LENSING_PROBMAP_FILE'] = None
        # SNANA also zeroes NFILT_SMEAR, killing GENMAG_SMEAR_FILTER
        # (snlc_sim.c:8776-8782)
        if 'GENMAG_SMEAR_FILTER' in raw:
            del raw['GENMAG_SMEAR_FILTER']
    if mask & 8:
        raw['OPT_MWEBV'] = 0
    if mask & 16:
        # zeroes an AV profile without activating one (snlc_sim.c:8735-8754)
        if any(k in raw for k in _AV_PROFILE_KEYS):
            raw['GENRANGE_AV'] = [0.0, 0.0]
            raw['GENTAU_AV'] = 0.0
            raw['GENSIGMA_AV'] = 0.0
        named = raw.get('POPULATIONS', {})
        if 'AV' in named:
            named['AV'] = replace(named['AV'], range=(0., 0.), exp_tau=0., sigma=0.)
    if mask & 32:
        raw['HOSTLIB_FILE'] = None
    raw['GENSIGMA_REDSHIFT'] = 1.0e-5
    raw['GENSIGMA_VPEC'] = 0.0
    raw['GENSIGMA_PEAKMJD'] = 0.1
    raw['OPT_SETPKMJD'] = 0
    raw['GENSIGMA_RV'] = 0.0
    named = raw.get('POPULATIONS', {})
    if 'RV' in named:
        named['RV'] = replace(named['RV'], sigma=(0., 0.))


def _apply_genpop_asymgauss(raw: dict) -> None:
    """GENPOP_ASYMGAUSS: <file> <model>: read the named MODEL block's SALT2
    x1/c GENGAUSS keys and override them in the input (snlc_sim.c:6437).

    The file format itself speaks SALT2 (its blocks carry SALT2x1/SALT2c
    populations), and SNANA applies it whatever the GENMODEL, so the names
    here are constants of the format, not model dispatch.
    """
    value = raw.get('GENPOP_ASYMGAUSS')
    if not value:
        return
    parts = value if isinstance(value, list) else str(value).split()
    if len(parts) < 2:
        raise SNANAError("GENPOP_ASYMGAUSS requires '<file> <model>'.",
                         function_name="_apply_genpop_asymgauss")
    fpath = Path(os.path.expandvars(str(parts[0]))).expanduser()
    model = str(parts[1])
    found_model = found_end = False
    block = {}
    for line in fpath.read_text().splitlines():
        toks = line.split()
        if not toks or toks[0].startswith('#'):
            continue
        if toks[0] == 'MODEL_NAME:':
            found_model = len(toks) > 1 and toks[1] == model
            continue
        if not found_model:
            continue
        if toks[0] == 'MODEL_END:':
            found_end = True
            break
        key = toks[0].rstrip(':')
        if 'SALT' not in key:
            continue
        if key.startswith(('GENPEAK_', 'GENMEAN_')):
            block[key] = float(toks[1])
        else:
            block[key] = [float(v) for v in toks[1:3]]
    if not (found_model and found_end and any('SALT2x1' in k for k in block)
            and any('SALT2c' in k for k in block)):
        raise SNANAError(
            f"GENPOP_ASYMGAUSS model '{model}' missing SALT2x1/SALT2c block "
            f"in {fpath}.", function_name="_apply_genpop_asymgauss")
    raw.update(block)


def _omit(value: Any) -> bool:
    """Whether a value is absent, and so must not reach the constructor.

    Absent means None, or a collection the parser materialised for a key nobody
    set. Passing it anyway would shadow the field's own default with a
    stand-in, which is the drift this layer exists to prevent.
    """
    return value is None or (isinstance(value, (list, tuple, dict, set))
                             and not value)


def _flag(value: Any) -> bool:
    """A SNANA on/off flag, which the input writes as 0 or 1."""
    return bool(int(value))


def _supplied(raw: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    """Constructor kwargs for the keys the input actually supplied.

    `spec` maps field name -> (key, cast); `key` may be a tuple of accepted
    spellings, and `cast` may be None where the reader already produced the
    value the field wants. A field is omitted, so its own default applies,
    when the input did not supply it: absent, None, or an empty collection.
    """
    out = {}
    for fld, (key, cast) in spec.items():
        val = None
        for k in ((key,) if isinstance(key, str) else key):
            val = raw.get(k)
            if val is not None:
                break
        if _omit(val):
            continue
        try:
            out[fld] = val if cast is None else cast(val)
        except (TypeError, ValueError):
            raise SNANAError(f"{k}: cannot read {val!r}",
                             function_name="build") from None
    return out


def resolve_input_path(path, run) -> str:
    """Resolve a SNANA input-resource path, honouring PATH_USER_INPUT.

    Expands env vars; if the file is not found and PATH_USER_INPUT is set, look
    for it (by basename) in that directory (SNANA snlc_sim.c:7230/18405/22896
    prepends PATH_USER_INPUT to the kcor/simlib/zvariation search path).
    """
    expanded = os.path.expandvars(str(path))
    if not expanded:
        return expanded
    p = Path(expanded)
    base = run.path_user_input
    if p.name and base and not p.exists():
        candidate = Path(os.path.expandvars(base)) / p.name
        if candidate.exists():
            return str(candidate)
    return expanded


def _computed(**values: Any) -> Dict[str, Any]:
    """Constructor kwargs for values the build derives rather than reads.

    Same rule as `_supplied`: nothing derived means nothing passed, so an empty
    stand-in can never shadow the field's own default.
    """
    return {k: v for k, v in values.items() if not _omit(v)}


def _cosmology(raw: dict) -> Cosmology:
    anisotropy = str(raw.get('ANISOTROPY_MODELNAME') or '').strip()
    if anisotropy and anisotropy.upper() != 'V04':
        # V04 is SNANA's only model (sntools_cosmology.c:588)
        raise SNANAError(
            f"Unknown ANISOTROPY_MODELNAME '{anisotropy}'; SNANA defines "
            f"only V04.", function_name="_cosmology")
    return Cosmology(**_supplied(raw, {
        'omega_matter':         ('OMEGA_MATTER', float),
        'omega_lambda':         ('OMEGA_LAMBDA', float),
        'h0':      ('H0', float),
        'w0':                   (Cosmology.input_names['w0'], float),
        'wa':                   (Cosmology.input_names['wa'], float),
        'hzfun_file':           (Cosmology.input_names['hzfun_file'], str),
        'anisotropy_modelname': ('ANISOTROPY_MODELNAME', str),
        'gensigma_vpec':        ('GENSIGMA_VPEC', float),
        'vpec_err':             ('VPEC_ERR', float),
        'vel_cmbapex':          ('VEL_CMBAPEX', float),
        'vpec_force':           ('VPEC_FORCE', float),
    }))


def _survey(raw: dict, genfilters: str) -> Survey:
    _require(raw, 'SIMLIB_FILE', "_survey")
    supplied = _supplied(raw, {
        'simlib_file':          ('SIMLIB_FILE', str),
        'leakage_cut':          ('LEAKAGE_CUT', float),
        'survey_override':      (Survey.input_names['survey_override'], str),
        'simlib_minseason':     ('SIMLIB_MINSEASON', float),
        'simlib_minobs':        ('SIMLIB_MINOBS', int),
        'simlib_maxobs':        ('SIMLIB_MAXOBS', int),
        'simlib_idlock':        ('SIMLIB_IDLOCK', int),
        'simlib_idstart':       ('SIMLIB_IDSTART', int),
        'simlib_idskip':        ('SIMLIB_IDSKIP',
                                 lambda v: tuple(int(x) for x in v)),
        'simlib_prescale':      ('SIMLIB_PRESCALE', int),
        'simlib_mjd_shift':     ('SIMLIB_MJD_SHIFT', float),
        'simlib_fieldlist':     ('SIMLIB_FIELDLIST', str),
        'simlib_nrepeat':       ('SIMLIB_NREPEAT', int),
        'simlib_maxranstart':   ('SIMLIB_MAXRANSTART', int),
        'simlib_mskopt':        ('SIMLIB_MSKOPT', int),
        'simlib_nskipmjd':      ('SIMLIB_NSKIPMJD', _parse_simlib_nskipmjd),
        'genrange_mjd_exclude': ('GENRANGE_MJD_EXCLUDE', _parse_range_tuple),
        'genrange_ra':          ('GENRANGE_RA', _pair),
        'genrange_decl':        ('GENRANGE_DECL', _pair),
        'genrange_b':           (Survey.input_names['genrange_b'], _pair),
        'newmjd_dif':           ('NEWMJD_DIF', float),
        'use_simlib_redshift':  ('USE_SIMLIB_REDSHIFT', int),
        'use_simlib_peakmjd':   ('USE_SIMLIB_PEAKMJD', int),
        'use_simlib_salt2':     ('USE_SIMLIB_SALT2', int),
        'use_simlib_distance':  ('USE_SIMLIB_DISTANCE', int),
        'use_simlib_groupid':   ('USE_SIMLIB_GROUPID', _flag),
        'use_simlib_spectrograph': ('USE_SIMLIB_SPECTROGRAPH', _flag),
        'use_simlib_take_spectrum': (Survey.input_names['use_simlib_take_spectrum'], _flag),
        'forceval_psf':         ('FORCEVAL_PSF', float),
        'simlib_dump':          ('SIMLIB_DUMP', int),
        'exposure_time_global': (Survey.input_names['exposure_time_global'], float),
        'exposure_time_mskopt': ('EXPOSURE_TIME_MSKOPT', int),
        'searcheff_apply_detect_single': (Survey.input_names['searcheff_apply_detect_single'], _flag),
        'searcheff_minobs':              (Survey.input_names['searcheff_minobs'], int),
        'searcheff_pipeline_eff_file':   ('SEARCHEFF_PIPELINE_EFF_FILE', str),
        'searcheff_pipeline_logic_file': ('SEARCHEFF_PIPELINE_LOGIC_FILE', str),
        'searcheff_spec_eff_file':       (Survey.input_names['searcheff_spec_eff_file'], str),
        'searcheff_zhost_eff_file':      (Survey.input_names['searcheff_zhost_eff_file'], str),
        'searcheff_spec_scale':          ('SEARCHEFF_SPEC_SCALE', float),
        'searcheff_spec_magshift':       (Survey.input_names['searcheff_spec_magshift'], float),
        'searcheff_zhost_magshift':      (Survey.input_names['searcheff_zhost_magshift'], float),
        'searcheff_photflag_detect':     (Survey.input_names['searcheff_photflag_detect'], int),
        'searcheff_photflag_trigger':    (Survey.input_names['searcheff_photflag_trigger'], int),
        'searcheff_map_refac':           (Survey.input_names['searcheff_map_refac'], int),
    })
    if supplied.get('searcheff_map_refac',
                    field_default(Survey, 'searcheff_map_refac')) not in (1, 2):
        raise SNANAError(
            f"REFAC_SEARCHEFF_MAP = {supplied['searcheff_map_refac']}: only the "
            "refactored SPECID and zHOST maps exist (1, or 2 for an independent "
            "zHOST random); the legacy readers were removed from SNANA",
            function_name="_survey")

    # FLUXERRMODEL sets SIMLIB_MSKOPT bit 32, IGNORE_FLUXERR_COR (snlc_sim.c:18928-18935)
    default_mskopt = field_default(Survey, 'simlib_mskopt')
    fluxerr_path = raw.get('FLUXERRMODEL_FILE')
    if not is_ignorefile(fluxerr_path):
        supplied['simlib_mskopt'] = supplied.get('simlib_mskopt', default_mskopt) | 32
    mskopt = supplied.get('simlib_mskopt', default_mskopt)
    for bit, label in _UNSUPPORTED_SIMLIB_MSKOPT.items():
        if mskopt & bit:
            raise SNANAError(
                f"SIMLIB_MSKOPT bit {bit} ({label}) is not supported "
                f"in jax_SNANA.", function_name="_survey")

    # SEARCHEFF apply fanout (APPLY_SEARCHEFF_OPT bits) + per-variable shifts
    fanout: Dict[str, Any] = {}
    if 'APPLY_SEARCHEFF_OPT' in raw:
        mask = int(raw['APPLY_SEARCHEFF_OPT'])
        fanout = {'_searcheff_apply': bool(mask), 'apply_searcheff_opt': mask,
                  '_searcheff_apply_pipeline': bool(mask & 1),
                  '_searcheff_apply_spec': bool(mask & 2)}
    shifts: Dict[str, Dict[str, float]] = {
        'searcheff_spec_shifts': {}, 'searcheff_zhost_shifts': {}}
    for fld, prefix in (('searcheff_spec_shifts', 'SEARCHEFF_SPEC_SHIFT('),
                        ('searcheff_zhost_shifts', 'SEARCHEFF_zHOST_SHIFT(')):
        for key, val in raw.items():
            if key.startswith(prefix) and key.endswith(')'):
                shifts[fld][key[len(prefix):-1]] = float(val)

    exposure_time = _key_plus_filter(raw, 'EXPOSURE_TIME', 1.0, genfilters)
    mjd_template = _key_plus_filter(raw, 'MJD_TEMPLATE', 0.0, genfilters)
    return Survey(
        genfilters=genfilters,
        kcor_file=_require(raw, 'KCOR_FILE', "_survey"),
        **({'exposure_time': exposure_time} if exposure_time else {}),
        **({'mjd_template': mjd_template} if mjd_template else {}),
        **fanout,
        **{k: v for k, v in shifts.items() if v},
        **supplied)


def _gal_detect(raw, key: str, descending: bool) -> Optional[List[float]]:
    """HOSTLIB_SNR_DETECT / _MAG_DETECT: a sorted threshold ladder
    (parse_input_HOSTLIB_GAL_DETECT)."""
    value = raw.get(key)
    if value is None:
        return None
    if isinstance(value, (int, float)):
        values = [float(value)]
    elif isinstance(value, list):
        values = [float(x) for x in value]
    else:
        values = _parse_comma_list(value)
    values.sort(reverse=descending)
    return values


def _host(raw: dict) -> Host:
    if 'HOSTLIB_FILE' not in raw:
        used_without_file = [k for k in _HOSTLIB_KEYS_NEEDING_FILE if k in raw]
        if used_without_file:
            raise SNANAError(
                f"HOSTLIB position/magnitude/selection knobs require "
                f"HOSTLIB_FILE: {', '.join(used_without_file)}",
                function_name="_host")
    # host dust is on only once a key names it (sentinel -9, snlc_sim.c:889)
    dust = {}
    if any(k in raw for k in _HOST_DUST_KEYS):
        dust = dict(
            dust_enabled=True,
            genav_wv07=_wv07_profile(raw),
            **_supplied(raw, {
                'av_range':         (Host.input_names['av_range'], _pair),
                'tau_av':           (Host.input_names['tau_av'], float),
                'wv07_rewgt_expav': ('WV07_REWGT_EXPAV', float),
            }))
    mskopt = (int(raw['HOSTLIB_MSKOPT']) if 'HOSTLIB_MSKOPT' in raw
              else field_default(Host, 'mskopt'))
    for bit, option in HOSTLIB_MSKOPT_REWRITE_BITS.items():
        if mskopt & bit:
            raise SNANAError(
                f"HOSTLIB_MSKOPT bit {bit} is {option}, which rewrites the "
                "HOSTLIB rather than simulating; not implemented",
                function_name="_host")
    return Host(
        snr_store=_gal_detect(raw, 'HOSTLIB_SNR_STORE', descending=True),
        snr_detect=_gal_detect(raw, 'HOSTLIB_SNR_DETECT', descending=True),
        mag_detect=_gal_detect(raw, 'HOSTLIB_MAG_DETECT', descending=False),
        **dust,
        **_supplied(raw, {
            'hostlib_file':            ('HOSTLIB_FILE', _file_or_none),
            'wgtmap_file':             (Host.input_names['wgtmap_file'], str),
            'wgtmap_extrap':           (Host.input_names['wgtmap_extrap'], int),
            'dztol':                   (Host.input_names['dztol'], str),
            'mskopt':                  (Host.input_names['mskopt'], int),
            'sbradius':                (Host.input_names['sbradius'], float),
            'scale_property_err':      (Host.input_names['scale_property_err'], str),
            'scale_logmass_err':       (Host.input_names['scale_logmass_err'], None),
            'maxddlr':                 (Host.input_names['maxddlr'], float),
            'maxddlr2':                (Host.input_names['maxddlr2'], float),
            'mindaysep_samegal':       (Host.input_names['mindaysep_samegal'], int),
            'galid_force':             (Host.input_names['galid_force'], int),
            'smear_sersic':            (Host.input_names['smear_sersic'], float),
            'galid_priority':          (Host.input_names['galid_priority'], str),
            'abmag_offset':            (Host.input_names['abmag_offset'], float),
            'abmag_force':             (Host.input_names['abmag_force'], float),
            'fixran_wgt':              (Host.input_names['fixran_wgt'], float),
            'fixran_radius':           (Host.input_names['fixran_radius'], float),
            'fixran_phi':              (Host.input_names['fixran_phi'], float),
            'mnintflux_snpos':         (Host.input_names['mnintflux_snpos'], float),
            'mxintflux_snpos':         (Host.input_names['mxintflux_snpos'], float),
            'scale_sersic_size':       (Host.input_names['scale_sersic_size'], float),
            'fixsersic':               (Host.input_names['fixsersic'],
                                        lambda v: tuple(float(x) for x in v)),
            'hostlib_genrange_ra':             ('HOSTLIB_GENRANGE_RA', _pair),
            'hostlib_genrange_dec':            ('HOSTLIB_GENRANGE_DEC', _pair),
            'hostlib_maxread':                 ('HOSTLIB_MAXREAD', int),
            'varname_zphot':           (Host.input_names['varname_zphot'], str),
            'galid_null':              (Host.input_names['galid_null'], int),
            'specbasis_file':          (Host.input_names['specbasis_file'], str),
            'specdata_file':           (Host.input_names['specdata_file'], str),
            'galid_unique':            (Host.input_names['galid_unique'], int),
            'nrepeat_galid_snpos':     (Host.input_names['nrepeat_galid_snpos'], int),
            'wronghost_file':          ('WRONGHOST_FILE', str),
            'biascor_salt2gamma_grid': ('BIASCOR_SALT2GAMMA_GRID', str),
            'snr_scale':               (Host.input_names['snr_scale'], None),
            'genzphot_fudgepar':       (Host.input_names['genzphot_fudgepar'], None),
            'genzphot_fudgemap':       (Host.input_names['genzphot_fudgemap'], None),
            'genzphot_bias':           (Host.input_names['genzphot_bias'], _parse_float_list),
            'zphoteff_file':           (Host.input_names['zphoteff_file'], None),
            # a property under either spelling (sntools_host.c:1098)
            'storepar':                (Host.input_names['storepar'],
                                        lambda v: [canonical_property_name(n)
                                                   for n in _parse_comma_list(v, str)]),
            'mxradius_random_shift':   ('MXRADIUS_RANDOM_SHIFT', float),
        }))


def _file_or_none(value) -> Optional[str]:
    """A file key's value, or None where it names no file (NONE, NULL,
    BLANK or empty, SNANA's IGNOREFILE)."""
    return None if is_ignorefile(value) else str(value)


def _parse_float_list(raw) -> List[float]:
    """Parse a SNANA-style list-or-whitespace-separated string of floats."""
    if isinstance(raw, (list, tuple)):
        return [float(x) for x in raw]
    return [float(x) for x in str(raw).split()]


def _mw_parlist(raw) -> List[float]:
    """PARLIST_MWCOLORLAW: floats, space- or comma-separated. A scalar would
    silently wrap to a one-element list and mask a malformed input."""
    if isinstance(raw, str):
        return _parse_comma_list(raw)
    if isinstance(raw, (list, tuple)):
        return [float(v) for v in raw]
    raise SNANAError(
        f"PARLIST_MWCOLORLAW must be a comma- or space-separated list "
        f"of floats (got {type(raw).__name__}: {raw!r}).",
        function_name="_mw_dust")


def _mw_dust(raw: dict) -> MWDust:
    """MW dust; defaults match SNANA init at snlc_sim.c:968-986."""
    if 'OPT_MWCOLORLAW' in raw and raw['OPT_MWCOLORLAW'] not in OPT_MWCOLORLAW_NAMES:
        raise SNANAError(
            f"Invalid OPT_MWCOLORLAW = {raw['OPT_MWCOLORLAW']}. Valid: "
            f"{', '.join(str(c) for c in sorted(OPT_MWCOLORLAW_NAMES))}.",
            function_name="_mw_dust")
    if 'OPT_MWEBV' in raw and int(raw['OPT_MWEBV']) not in OPT_MWEBV_CODES:
        raise SNANAError(
            f"Invalid OPT_MWEBV = {raw['OPT_MWEBV']}. Valid: 0 off, 1 FILE, 2 "
            f"SFD98, 3 Schlafly11+PS2013.",
            function_name="_mw_dust")
    dust_law = OPT_MWCOLORLAW_NAMES.get(raw.get('OPT_MWCOLORLAW'))
    dust = MWDust(
        **({'dust_law': dust_law} if dust_law else {}),
        **_supplied(raw, {
            'genrange':    (MWDust.input_names['genrange'], _parse_range_tuple),
            'rv':          (MWDust.input_names['rv'], float),
            'parlist':     (MWDust.input_names['parlist'], _mw_parlist),
            'opt_mwebv':   ('OPT_MWEBV', int),
            'sigma_ratio': (MWDust.input_names['sigma_ratio'], float),
            'sigma':       (MWDust.input_names['sigma'], float),
            # SNANA accepts the FUDGE* spellings too (snlc_sim.c:5189-5195)
            'shift':       (MWDust.input_names['shift'], float),
            'scale':       (MWDust.input_names['scale'], float),
            'applyflag_mwebv': ('APPLYFLAG_MWEBV', int),
        }))
    if dust.opt_mwebv == 1 and dust.sigma_ratio < 0.0:
        raise SNANAError(
            "Undefined GENSIGMA_MWEBV_RATIO (remove ancient default of 0.16). "
            "Check sim-inputs.", function_name="_mw_dust")
    return dust


def _instrument(raw: dict, genfilters: str) -> Instrument:
    # Only MONITORCOV(128) and REQUIRE_DOCANA(1024) are defined
    # (sntools_fluxErrModels.h:27,29); neither changes photometry.
    optmask = int(raw.get('FLUXERRMODEL_OPTMASK', 0))
    unknown_bits = optmask & ~(128 | 1024)
    if unknown_bits:
        # bit 256 (MASK_DUMP_MAP_FLUXERRMODEL, sntools_fluxErrModels.h:28)
        # dumps the map, and crashes SNANA itself
        dump_bit = (" Bit 256 is SNANA's map dump, which segfaults there."
                    if unknown_bits & 256 else "")
        raise SNANAError(
            f"FLUXERRMODEL_OPTMASK={optmask} sets unsupported bit(s) "
            f"{unknown_bits}; only 128 (MONITORCOV) and 1024 "
            f"(REQUIRE_DOCANA) are implemented.{dump_bit}",
            function_name="_fluxerrmodel")
    kpf = {
        fld: _key_plus_filter(raw, base, default, genfilters)
        for fld, base, default in (
            ('fudgeshift_zpt', 'FUDGESHIFT_ZPT', 0.0),
            ('fudgescale_fluxerr', 'FUDGESCALE_FLUXERR', 1.0),
            ('fudgescale_fluxerr2', 'FUDGESCALE_FLUXERR2', 1.0),
            ('fudgeshift_lam', 'FUDGESHIFT_LAM', 0.0),
            ('fudge_magerr', 'FUDGE_MAGERR', 0.0),
            ('fudge_zpterr', 'FUDGE_ZPTERR', 0.0),
        )
    }
    off_zp = _per_filter_offsets(raw, 'GENMAG_OFF_ZP', genfilters)
    return Instrument(
        **{k: v for k, v in kpf.items() if v},
        **({'genmag_off_zp': off_zp} if off_zp else {}),
        **_site_sigmas(raw),
        **_supplied(raw, {
        'atmos_optmask':     (Instrument.input_names['atmos_optmask'], int),
        'atmos_sedstar_file': (Instrument.input_names['atmos_sedstar_file'], str),
        'coordres_floor':    (Instrument.input_names['coordres_floor'], float),
        'dcr_coordres_poly': (Instrument.input_names['dcr_coordres_poly'], _parse_comma_list),
        'dcr_magshift_poly': (Instrument.input_names['dcr_magshift_poly'], _parse_comma_list),
        'npe_pixel_saturate': ('NPE_PIXEL_SATURATE', int),
        'photflag_saturate':  ('PHOTFLAG_SATURATE', int),
        'zp_fluxcal':        ('ZP_FLUXCAL', float),
        'smearflag_flux':    ('SMEARFLAG_FLUX', int),
        'smearflag_zeropt':  ('SMEARFLAG_ZEROPT', int),
        'photflag_snrmax':   ('PHOTFLAG_SNRMAX', int),
        'photflag_nearpeak': ('PHOTFLAG_NEARPEAK', int),
        'fudgescale_psf':    ('FUDGESCALE_PSF', float),
        'fudgescale_noise_sky':  (Instrument.input_names['fudgescale_noise_sky'], float),
        'fudgescale_noise_read': (Instrument.input_names['fudgescale_noise_read'], float),
        'fudgescale_noise_template': ('FUDGESCALE_NOISE_TEMPLATE', float),
        'fluxerr_file':   (Instrument.input_names['fluxerr_file'], None),
        'fluxerr_redcov':         (Instrument.input_names['fluxerr_redcov'], None),
        'fluxerr_snrmin_redcov':  (Instrument.input_names['fluxerr_snrmin_redcov'], float),
        # SNANA spells this one FLUXERRMAP_, not FLUXERRMODEL_
        'fluxerr_ignore_dataerr': (Instrument.input_names['fluxerr_ignore_dataerr'], None),
    }))


def _model_input_values(raw: dict, model_type: str) -> Dict[str, Any]:
    """The supplied keys among those the selected model class declares.

    A simulation builds the model but does not fit it, so the keys the model
    marks FIT_ONLY are not among them.
    """
    keys = model_input_keys(MODEL_REGISTRY[model_type], consumer='sim')
    values = {}
    for key, spec in keys.items():
        if key in raw:
            values[key] = parse_model_input_value(key, spec, raw[key])
    return values


def _model(raw: dict, genfilters: str, genmodel: Any, model_type: str,
           populations: Dict[str, Any], structured_config: Dict[str, Any],
           subpops: Tuple[Any, ...], offset_shifts: Dict[str, Any],
           ebv_igm_shift: Any, input_genranges: Dict[str, Tuple[float, float]]
           ) -> Model:
    return Model(
        genmodel=genmodel,
        **({'subpops': subpops} if subpops else {}),
        **_computed(
            _model_input_values=_model_input_values(raw, model_type),
            populations=populations,
            _offset_shifts=offset_shifts,
            _ebv_igm_shift=ebv_igm_shift,
            _input_genranges=input_genranges,
            _structured_config=structured_config,
            salt2beta_cpoly=parse_genpoly(raw.get('SALT2BETA_cPOLY')),
            genmag_off_model=_per_filter_offsets(raw, 'GENMAG_OFF_MODEL', genfilters),
            genmag_smear_filter=_genmag_smear_filter(raw),
            genmag_smear_scale_poly=_genmag_smear_scale_poly(
                raw, MODEL_REGISTRY[model_type], structured_config),
        ),
        **_supplied(raw, {
            'genmodel_mskopt':          ('GENMODEL_MSKOPT', int),
            'mag_offset':              ('MAG_OFFSET', float),
            'mushift':                  ('MUSHIFT', parse_mushift),
            'genmag_smear':             ('GENMAG_SMEAR',
                                         _genmag_smear_sigma),
            'genmag_smear_scale':       ('GENMAG_SMEAR_SCALE', float),
            'genmag_smear_modelname':   ('GENMAG_SMEAR_MODELNAME',
                                         lambda v: str(v).strip()),
            'genmag_smear_addphasecor': ('GENMAG_SMEAR_ADDPHASECOR',
                                         lambda v: list(_pair(v))),
            'sigmaclip_magsmear':       ('SIGMACLIP_MAGSMEAR', _pair),
            'genmag_off_non1a':         ('GENMAG_OFF_NON1A', float),
            'genmag_off_global':        ('GENMAG_OFF_GLOBAL', float),
            'gensmear_rangauss_fix':    (Model.input_names['gensmear_rangauss_fix'], float),
            'genpdf_optmask':           ('GENPDF_OPTMASK', int),
            'zvariation_file':          ('ZVARIATION_FILE', str),
            'zvariation_poly':          ('ZVARIATION_POLY', list),
            'mjd_explode':              ('MJD_EXPLODE', float),
        }))


def _spectra(raw: dict) -> Spectra:
    return Spectra(**_supplied(raw, {
        'spectrograph_file':        ('SPECTROGRAPH_FILE', str),
        'spectrograph_optmask':     ('SPECTROGRAPH_OPTMASK', int),
        'spectrograph_scale_texpose': ('SPECTROGRAPH_SCALE_TEXPOSE', float),
        'spectrograph_scale_snr':   ('SPECTROGRAPH_SCALE_SNR', str),
        'take_spectrum':            ('TAKE_SPECTRUM', list),
        'warp_spectrum_lampoly':    (Spectra.input_names['warp_spectrum_lampoly'], _parse_warp_spectrum),
        'take_spectrum_template_texpose_scale': (
            'TAKE_SPECTRUM_TEMPLATE_TEXPOSE_SCALE', float),
        'take_spectrum_hostfrac':   ('TAKE_SPECTRUM_HOSTFRAC', float),
        'take_spectrum_hostsnfrac': ('TAKE_SPECTRUM_HOSTSNFRAC', float),
        'take_spectrum_prescale':   ('TAKE_SPECTRUM_PRESCALE',
                                     parse_prescales),
        'take_spectrum_dumpcid':    ('TAKE_SPECTRUM_DUMPCID', int),
        'lambin_sed_true':          ('LAMBIN_SED_TRUE', float),
        'lamrange_sed_true':        ('LAMRANGE_SED_TRUE', _parse_range_tuple),
        'verify_sed_true':          ('VERIFY_SED_TRUE', int),
    }))


def _lensing(raw: dict) -> Lensing:
    """SNANA accepts both WEAKLENS_ and LENSING_ prefixes
    (snlc_sim.c:4215-4226); defaults from INPUTS_WEAKLENS init at :1047-1050."""
    return Lensing(**_supplied(raw, {
        'probmap_file': (Lensing.input_names['probmap_file'],
                         None),
        'dmuscale':     (Lensing.input_names['dmuscale'], float),
        'dmuerr_frac':  (Lensing.input_names['dmuerr_frac'], float),
        'dsigmadz':     (Lensing.input_names['dsigmadz'], float),
    }))


# CUTWIN keys parsed specially; everything else CUTWIN_<VAR> is a range.
_SPECIAL_CUTWIN_KEYS = {
    'CUTWIN_NEPOCH', 'CUTWIN_SNRMAX', 'CUTWIN_MJDDIF', 'CUTWIN_NOBSDIF',
    'CUTWIN_EPOCHS_SNRMIN', 'CUTWIN_TRESTMIN', 'CUTWIN_TRESTMAX',
    'CUTWIN_NOBS_SATURATE', 'CUTWIN_NOBS_NOSATURATE',
    'CUTWIN_TGAPMAX', 'CUTWIN_T0GAPMAX',
    'CUTWIN_PEAKMAG', 'CUTWIN_PEAKMAG_ALL', 'CUTWIN_PEAKMAG_BYFIELD',
    'CUTWIN_REDSHIFT',  # aliased to CUTWIN_REDSHIFT_TRUE below
}


def _cuts(raw: dict) -> Cuts:
    """CUTWIN_* selection cuts + EPCUTWIN_* per-epoch windows.

    Malformed values for recognised cut keys abort: a silently dropped cut
    would pass every event the cut was meant to reject.
    """
    named: Dict[str, Any] = {}

    def _cut_range(key):
        if key not in raw:
            return None
        try:
            parsed = _parse_range_tuple(raw[key])
        except ValueError:
            parsed = None
        if parsed is None:
            raise SNANAError(
                f"{key} expects two numbers, got {raw[key]!r}",
                function_name="_cuts") from None
        return parsed

    if 'CUTWIN_NEPOCH' in raw:
        params = raw['CUTWIN_NEPOCH']
        if isinstance(params, list):
            if len(params) >= 1:
                named['nepoch_min'] = int(params[0])
            if len(params) >= 2:
                named['nepoch_snr_threshold'] = float(params[1])
        else:
            named['nepoch_min'] = int(params)

    # CUTWIN_SNRMAX entries, AND-combined (snlc_sim.c:4338-4346, :7741-7760)
    snrmax_cuts = []
    for row in raw.get('CUTWIN_SNRMAX') or []:
        if not isinstance(row, (list, tuple)) or len(row) < 5:
            raise SNANAError(
                f"CUTWIN_SNRMAX expects 'snrmax filters logic trest_lo "
                f"trest_hi', got {row!r}", function_name="_cuts")
        snrmax, filters, logic = float(row[0]), str(row[1]), str(row[2])
        if logic.upper() == 'AND':
            nfilt = len(filters)
        elif logic.upper() == 'OR':
            nfilt = 1
        else:
            try:
                nfilt = int(logic)
            except ValueError:
                raise SNANAError(
                    f"CUTWIN_SNRMAX logic '{logic}' is not AND, OR or an "
                    f"integer", function_name="_cuts")
            if nfilt < 1 or nfilt > len(filters):
                raise SNANAError(
                    f"Invalid NFILT_CUT={nfilt} for FILTERS='{filters}' in "
                    f"CUTWIN_SNRMAX", function_name="_cuts")
        snrmax_cuts.append(
            (snrmax, filters, nfilt, float(row[3]), float(row[4])))
    if len(snrmax_cuts) > 5:
        raise SNANAError(
            f"{len(snrmax_cuts)} CUTWIN_SNRMAX entries; SNANA allows at "
            f"most 5 (MXCUTWIN_SNRMAX)", function_name="_cuts")
    if snrmax_cuts:
        named['snrmax_cuts'] = snrmax_cuts

    for fld, key, cast in (('mjddif_min', 'CUTWIN_MJDDIF', float),
                           ('nobsdiff_min', 'CUTWIN_NOBSDIF', int)):
        if key in raw:
            params = raw[key]
            named[fld] = cast(params[0] if isinstance(params, list) else params)

    if 'CUTWIN_EPOCHS_SNRMIN' in raw:
        params = raw['CUTWIN_EPOCHS_SNRMIN']
        if isinstance(params, list):
            named['epochs_time_cut'] = {
                'snr_min': float(params[0]) if len(params) >= 1 else 0.0,
                # TRANGE[1] (snlc_sim.c:24295-24297)
                'time_above_max': (float(params[1])
                                   if len(params) >= 2
                                   else CUTWIN_EPOCHS_TRANGE_MAX_DEFAULT),
                'filters': str(params[2]) if len(params) >= 3 else '',
            }
        else:
            named['epochs_time_cut'] = {
                'snr_min': float(params),
                'time_above_max': CUTWIN_EPOCHS_TRANGE_MAX_DEFAULT,
                'filters': ''}

    for fld, key in (('trestmin_range', 'CUTWIN_TRESTMIN'),
                     ('trestmax_range', 'CUTWIN_TRESTMAX')):
        parsed = _cut_range(key)
        if parsed is not None:
            named[fld] = parsed

    # per-filter saturated-obs count windows (snlc_sim.c:24467-24477)
    for fld, key in (('saturate_ranges', 'CUTWIN_NOBS_SATURATE'),
                     ('nosaturate_ranges', 'CUTWIN_NOBS_NOSATURATE')):
        triplets = _parse_cutwin_filter_triplets(raw, key)
        if triplets:
            named[fld] = triplets

    for fld, key in (('tgapmax_range', 'CUTWIN_TGAPMAX'),
                     ('t0gapmax_range', 'CUTWIN_T0GAPMAX'),
                     ('peakmag_range', 'CUTWIN_PEAKMAG'),
                     ('peakmag_all_range', 'CUTWIN_PEAKMAG_ALL')):
        parsed = _cut_range(key)
        if parsed is not None:
            named[fld] = parsed

    byfield = _parse_cutwin_filter_triplets(raw, 'CUTWIN_PEAKMAG_BYFIELD', cast=float)
    if byfield:
        named['peakmag_byfield'] = byfield

    # any other CUTWIN_<VAR> is a range on that variable; CUTWIN_REDSHIFT is
    # CUTWIN_REDSHIFT_TRUE (snlc_sim.c:4301)
    generic: Dict[str, Tuple[float, float]] = {}
    if 'CUTWIN_REDSHIFT' in raw and 'CUTWIN_REDSHIFT_TRUE' not in raw:
        parsed = _cut_range('CUTWIN_REDSHIFT')
        if parsed is not None:
            generic['redshift_true'] = parsed
    for key in list(raw):
        if key.startswith('CUTWIN_') and key not in _SPECIAL_CUTWIN_KEYS:
            parsed = _cut_range(key)
            if parsed is not None:
                generic[key[len('CUTWIN_'):].lower()] = parsed
    # MWEBV has its own field; the rest stay in the open-ended map
    if 'mwebv' in generic:
        named['mwebv_range'] = generic.pop('mwebv')

    return Cuts(
        **named,
        **_supplied(raw, {
            'apply_cutwin_opt': ('APPLY_CUTWIN_OPT', int),
            'epcutwin_lamrest': ('EPCUTWIN_LAMREST', _pair),
            'epcutwin_snrmin':  ('EPCUTWIN_SNRMIN', _pair),
        }),
        **_computed(generic_ranges=generic))


def _rates(raw: dict, model_type: str, dndz_entries_pec1a) -> Rates:
    """Rate model + generation ranges.

    OPT_SETPKMJD default 16 = OPTMASK_SETPKMJD_FLUXMAX2 (Fmax-clump method;
    snlc_sim.c:851, sntools.h:413). GENSIGMA_PEAKMJD>0 selects the older
    Gaussian smear instead, which forces OPT_SETPKMJD=0 (snlc_sim.c:10105)."""
    supplied = _supplied(raw, {
        'redshift_range':      (Rates.input_names['redshift_range'], _require_range),
        'peakmjd_range':       (Rates.input_names['peakmjd_range'], _parse_range_tuple),
        'peakmjd_exclude':     (Rates.input_names['peakmjd_exclude'], _parse_range_tuple),
        'trest_range':         (Rates.input_names['trest_range'], _parse_range_tuple),
        'genrange_mjd':        ('GENRANGE_MJD', _parse_range_tuple),
        'solid_angle':         ('SOLID_ANGLE', float),
        'gensigma_redshift':   ('GENSIGMA_REDSHIFT', float),
        'genbias_redshift':    ('GENBIAS_REDSHIFT', float),
        'opt_setpkmjd':        ('OPT_SETPKMJD', int),
        'gensigma_peakmjd':    ('GENSIGMA_PEAKMJD', float),
    })
    return Rates(
        **supplied,
        **_computed(
            dndz_entries=_dndz_entries(raw),
            dndz_entries_pec1a=dndz_entries_pec1a,
            dndz_rate_mods=_dndz_rate_mods(raw, model_type),
        ),
        **_gentypes(raw),
        **_sntypes(raw))


def _run(raw: dict) -> Run:
    return Run(
        random_seed=int(_require(raw, 'RANSEED', "_run")),
        # Valueless flag, so presence is the value (snlc_sim.c:1803).
        readme_dumpflag='README' in raw,
        **_computed(
            ransystpar={k[len('RANSYSTPAR_'):]: v for k, v in raw.items()
                        if k.startswith('RANSYSTPAR_')},
            # DASHBOARD takes no value; presence turns it on (snlc_sim.c:1799).
            dashboard='DASHBOARD' in raw,
        ),
        **_supplied(raw, {
            'ngen_lc':               ('NGEN_LC', int),
            'ngentot_lc':            ('NGENTOT_LC', int),
            'ngen_season':           ('NGEN_SEASON', float),
            'ngen_scale':            ('NGEN_SCALE', float),
            'ngen_scale_non1a':      (Run.input_names['ngen_scale_non1a'], float),
            'ngen_lc_maxgen_factor': ('NGEN_LC_MAXGEN_FACTOR', int),
            'init_only':             ('INIT_ONLY', int),
            'skip_check_crazyflux':  ('SKIP_CHECK_CRAZYFLUX', int),
            'require_docana':        ('REQUIRE_DOCANA', int),
            'jobid':                 ('JOBID', int),
            'njobtot':               ('NJOBTOT', int),
            'path_user_input':       ('PATH_USER_INPUT', str),
            'double_precision':      ('DOUBLE_PRECISION', _flag),
        }))


def _simgen_dump_columns(raw: dict, key: str) -> List[str]:
    """The column list shared by SIMGEN_DUMP and SIMGEN_DUMPALL
    (snlc_sim.c:4957-4976).

    `KEY: N col1 col2 …` declares its own count; the comma-separated form
    carries none (snlc_sim.c:5046). SNANA appends FLAG_ACCEPT to every
    user-specified list (snlc_sim.c:5061-5065).
    """
    value = raw[key]
    cols: List[str] = []
    if isinstance(value, str) and ',' in value:
        cols = _parse_comma_list(value, str)
    elif isinstance(value, list) and len(value) >= 2:
        try:
            n_requested = int(value[0])
        except (TypeError, ValueError) as exc:
            raise SNANAError(
                f"{key} first token must be an integer column count (got "
                f"{value[0]!r}); SNANA format is '{key}: N col1 col2 ...'.",
                function_name="_output") from exc
        cols = [str(x) for x in value[1:1 + n_requested]]
        if len(cols) != n_requested:
            raise SNANAError(
                f"{key} declares {n_requested} columns but only {len(cols)} "
                f"column names follow.", function_name="_output")
    if cols and 'FLAG_ACCEPT' not in cols:
        cols.append('FLAG_ACCEPT')
    return cols


def _output(raw: dict) -> Output:
    dump: Dict[str, Any] = {}
    for key in ('SIMGEN_DUMP', 'SIMGEN_DUMPALL'):
        if key not in raw:
            continue
        cols = _simgen_dump_columns(raw, key)
        if cols:
            dump['simgen_dumpall_columns'] = cols
        dump['simgen_dumpall' if key == 'SIMGEN_DUMPALL' else 'simgen_dump'] = True
    if 'SIMGEN_DUMPADD' in raw:
        added = [c for c in str(raw['SIMGEN_DUMPADD']).split(',') if c]
        cols = list(dump.get('simgen_dumpall_columns', []))
        dump['simgen_dumpall_columns'] = cols + [c for c in added if c not in cols]
    return Output(
        **dump,
        **_supplied(raw, {
            'genversion':            ('GENVERSION', str),
            # where the GENVERSION folder is written (prep_simpath,
            # snlc_sim.c:8551); the working directory without it
            'path_sndata_sim':       ('PATH_SNDATA_SIM',
                                      lambda v: os.path.expandvars(str(v))),
            'genprefix':             ('GENPREFIX', str),
            'gzip_data_files':       ('GZIP_DATA_FILES', _flag),
            'prescale_simgen_dump':  (Output.input_names['prescale_simgen_dump'], int),
            'wrspec_prescale':       ('WRSPEC_PRESCALE', int),
            'simgen_dump_noise':     ('SIMGEN_DUMP_NOISE', int),
            'simgen_dump_rate':      ('SIMGEN_DUMP_RATE', int),
            'simgen_dump_mwcl':      ('SIMGEN_DUMP_MWCL', int),
            'simgen_dump_trainsalt': ('SIMGEN_DUMP_TRAINSALT', int),
            'wrflag_yaml_file':      ('WRFLAG_YAML_FILE', int),
            'wrflag_modelpar':       ('WRFLAG_MODELPAR', int),
            'format_mask':           ('FORMAT_MASK', int),
            'nsubsample_mark':       ('NSUBSAMPLE_MARK', int),
            'append_snid_sedindex':  ('APPEND_SNID_SEDINDEX', int),
            'magmonitor_snr':        ('MAGMONITOR_SNR', int),
            'cidran_min':            ('CIDRAN_MIN', int),
            'cidran_max':            ('CIDRAN_MAX', int),
            'cidoff':                ('CIDOFF', int),
            'cidran_skiplist':       ('CIDRAN_SKIPLIST',
                                      lambda v: tuple(_parse_comma_list(v, int))),
            'time_start':            ('TIME_START',
                                      lambda v: ' '.join(map(str, v))
                                      if isinstance(v, list) else str(v)),
        }))


def _dndz_entries(raw: dict, group: str = 'DNDZ') -> List[Any]:
    """The rate models of one population, in the order SNANA accumulates them.

    A population carries a single model name (parse_input_RATEPAR,
    snlc_sim.c:2778): a second, different name aborts, and a repeat of the same
    name replaces it, except POWERLAW2, whose repeats are the z-ranges of one
    multi-range model (snlc_sim.c:2816).
    """
    values: List[Any] = [v.split() if isinstance(v, str) else list(v)
                         for v in raw.get(group, [])]
    if group == 'DNDZ':
        dndz_file = raw.get('DNDZ_FILE')
        if dndz_file:
            values.append(['FILE', str(dndz_file)])

    entries: List[Any] = []
    name = ''
    for value in values:
        incoming = str(value[0])
        if name and name != incoming:
            raise SNANAError(
                f"cannot mix multiple rate models with {group}: "
                f"{name} and {incoming} (pick one)",
                function_name="_dndz_entries")
        # stored without a CC_S15/PISN multiplier, compared with it (:2905)
        name = next((m for m in ('CC_S15', 'PISN_PLK12') if m in incoming),
                    incoming)
        if name == 'POWERLAW2':
            entries.append(value)
        else:
            entries = [value]
    return entries


def _dndz_rate_mods(raw: dict, model_type: str) -> Dict[str, Any]:
    """DNDZ_* reweights and scales, as applied by SNANA's genz_wgt.

    DNDZ_SCALE is `<scale_Ia> <scale_NON1A>`; SNANA picks slot [1] when
    `NON1A_MODELFLAG > 0 || INDEX_GENMODEL == MODEL_SIMSED` (snlc_sim.c:17844).
    DNDZ_SCALE_NON1A/_NONIA set slot [1] alone. Resolved here, where the model
    type is known, so the rate path applies a single factor.
    """
    mods = neutral_rate_mods()
    raw_scale = raw.get('DNDZ_SCALE')
    scale_pair = [mods['scale'], mods['scale']]
    if raw_scale is not None:
        parts = (list(raw_scale) if isinstance(raw_scale, (list, tuple))
                 else str(raw_scale).split())
        scale_pair = [float(parts[0]),
                      float(parts[1]) if len(parts) > 1 else float(parts[0])]
    for key in ('DNDZ_SCALE_NON1A', 'DNDZ_SCALE_NONIA'):
        if key in raw:
            scale_pair[1] = float(raw[key])

    slot = 1 if MODEL_REGISTRY[model_type].uses_nonia_rates else 0
    allscale = raw.get('DNDZ_ALLSCALE')
    mods['scale'] = scale_pair[slot] * (mods['scale'] if allscale is None
                                        else float(allscale))
    zexp = raw.get('DNDZ_ZEXP_REWGT')
    if zexp is not None:
        mods['zexp'] = float(zexp)
    for fld, key in (('zpoly', 'DNDZ_ZPOLY_REWGT'),
                     ('z1poly', 'DNDZ_Z1POLY_REWGT')):
        if key in raw:
            mods[fld] = parse_genpoly(raw[key])
    return mods


# the RV GENGAUSS init (snlc_sim.c:884-887)
_RV_GENGAUSS_INIT = {'peak': 3.1, 'range': (1.0, 5.0)}

# the GENGAUSS key family (sntools_genGauss_asym.c:510-606)
_GENGAUSS_KEY_STEMS = ('GENPEAK_', 'GENMEAN_', 'GENPEAKRANGE_', 'GENSIGMA_',
                       'GENSKEW_', 'GENRANGE_', 'GENGRID_', 'GENPROB2_',
                       'GENPEAK2_', 'GENSIGMA2_')


def _parse_gengauss_asym(raw: dict, prefix: str, init: Dict[str, Any]
                         ) -> Optional[GenGaussAsym]:
    """A full GENGAUSS_ASYM parameter set for one SNANA parameter suffix.

    Reads GENMEAN/GENPEAK, GENSIGMA, GENRANGE, GENPEAKRANGE, GENSKEW,
    GENPROB2, GENPEAK2, GENSIGMA2, GENGRID for e.g. "RV" or a model's
    declared parameters. Any key in the family activates the population;
    missing keys keep the supplied init values, either SNANA's per-parameter
    defaults or a shared distribution inherited by a SUBPOP. Returns None
    when no family key is present.
    """
    if not any(f'{stem}{prefix}' in raw for stem in _GENGAUSS_KEY_STEMS):
        return None

    def _as_pair(key, default):
        if key not in raw:
            return default
        value = raw[key]
        if isinstance(value, list) and len(value) >= 2:
            return (float(value[0]), float(value[1]))
        scalar = float(value[0] if isinstance(value, list) else value)
        return (scalar, scalar)

    peak = None
    for key in (f'GENPEAK_{prefix}', f'GENMEAN_{prefix}'):
        if key in raw:
            peak = float(raw[key])
            break
    if peak is None:
        peak = init.get('peak', 0.0)
        peak_given = False
    else:
        peak_given = True

    sigma = _as_pair(f'GENSIGMA_{prefix}', init.get('sigma', (0.0, 0.0)))
    # without GENRANGE or an init range, a given peak is a delta
    if peak_given:
        default_range = init.get('range', (peak, peak))
    else:
        default_range = init.get('range', (0.0, 0.0))
    gen_range = _as_pair(f'GENRANGE_{prefix}', default_range)
    # GENPEAK sets PEAKRANGE (sntools_genGauss_asym.c:532-538)
    if f'GENPEAKRANGE_{prefix}' in raw:
        peakrange = _parse_range_tuple(raw[f'GENPEAKRANGE_{prefix}'])
    elif peak_given:
        peakrange = (peak, peak)
    else:
        peakrange = init.get('peakrange', (0.0, 0.0))
    skew = _as_pair(f'GENSKEW_{prefix}', init.get('skew', (0.0, 0.0)))
    sigma2 = _as_pair(f'GENSIGMA2_{prefix}', init.get('sigma2', (0.0, 0.0)))
    ngrid = int(raw.get(f'GENGRID_{prefix}', init.get('ngrid', 0)))

    distribution = GenGaussAsym(
        peak=peak, sigma=sigma, range=gen_range, peakrange=peakrange, skew=skew,
        prob2=float(raw.get(f'GENPROB2_{prefix}', init.get('prob2', 0.0))),
        peak2=float(raw.get(f'GENPEAK2_{prefix}', init.get('peak2', 0.0))),
        sigma2=sigma2, ngrid=ngrid,
        prob_expon_rewgt=init.get('prob_expon_rewgt', 1.0))

    validate_gaussian(distribution)
    return distribution


def _spell_population_keys(raw: dict, spellings: Dict[str, str]) -> None:
    """Rename the input's GENGAUSS keys to spell their parameter by name."""
    for key in list(raw):
        renamed = spelled_by_name(key, _GENGAUSS_KEY_STEMS, spellings)
        if renamed == key:
            continue
        if renamed in raw.keys():
            raise SNANAError(
                f"{key} and {renamed} set the same population key; give one.",
                function_name="_spell_population_keys")
        # read before removing, so the input's own spelling counts as used
        value = raw[key]
        del raw[key]
        raw[renamed] = value


def _scan_gengauss(raw: dict, distributions: Dict[str, Any],
                   population_names: set, inits: Dict[str, Dict[str, Any]]
                   ) -> None:
    """Add a GenGaussAsym for every configured GENGAUSS parameter.

    A GENPEAK_*/GENMEAN_* key declares a population for any suffix. For
    `population_names`, the names SNANA routes through parse_input_GENGAUSS
    (snlc_sim.c:2303-2378), ANY family key activates the population with the
    SNANA init values filling the rest. The AV/EBV dust profiles are
    excluded: they are ExpHalfGauss families (snlc_sim.c:2379-2390).
    """
    suffixes = set()
    for key in list(raw):
        for prefix in ('GENPEAK_', 'GENMEAN_'):
            if key.startswith(prefix) and key[len(prefix):]:
                suffixes.add(key[len(prefix):])
    for name in population_names:
        if any(f'{stem}{name}' in raw for stem in _GENGAUSS_KEY_STEMS):
            suffixes.add(name)

    for suffix in suffixes - set(_DUST_PROFILE_NAMES):
        dist = _parse_gengauss_asym(raw, suffix, inits.get(suffix, {}))
        if dist is not None:
            distributions[suffix] = dist


def _parse_exp_halfgauss(raw: dict,
                         varnames: Tuple[str, ...],
                         init: Optional[Dict[str, Any]] = None
                         ) -> Optional[GenExpHalfGauss]:
    """The exponential + half-Gaussian dust profile for one VARNAME family.

    Keys, per parse_input_EXP_HALFGAUSS
    (sntools_genExpHalfGauss.c:347-406): GENGAUPEAK_<V> (Gaussian-core
    peak), GENSIG_<V>/GENSIGMA_<V>, GENTAU_<V>,
    GENRATIO_<V>/GENRATIO_<V>0 (plus back-compatible EBV0_HOST), and
    GENRANGE_<V>. Both alias VARNAMEs write one profile
    (snlc_sim.c:2379-2390); any key activates it. Missing values keep
    SNANA's -9 init (init_GEN_EXP_HALFGAUSS; RATIO inits to 0), under
    which a non-positive peak means the positive half-normal and a -9
    width disables that component. A configured profile without a range is
    malformed: SNANA would generate from the (-9, -9) init range.
    A SUBPOP may instead supply its shared distribution as init values.
    """
    init = init or {}

    def _scalar(keys, default):
        for key in keys:
            if key in raw:
                value = raw[key]
                return float(value[0] if isinstance(value, list) else value)
        return default

    peak_keys = [f'GENGAUPEAK_{v}' for v in varnames]
    sig_keys = [k for v in varnames for k in (f'GENSIG_{v}', f'GENSIGMA_{v}')]
    tau_keys = [f'GENTAU_{v}' for v in varnames]
    ratio_keys = [k for v in varnames
                  for k in (f'GENRATIO_{v}', f'GENRATIO_{v}0')]
    if 'EBV_HOST' in varnames:
        ratio_keys.append('EBV0_HOST')
    range_keys = [f'GENRANGE_{v}' for v in varnames]

    if not any(k in raw
               for k in peak_keys + sig_keys + tau_keys + ratio_keys
               + range_keys):
        return None

    gen_range = init.get('range')
    for key in range_keys:
        if key in raw:
            gen_range = _parse_range_tuple(raw[key])
            if gen_range is None:
                raise SNANAError(
                    f"{key} expects two values, got {raw[key]!r}",
                    function_name="_parse_exp_halfgauss")
            break
    if gen_range is None:
        # the (-9, -9) init range draws only the -9 sentinel: no host dust
        return None

    return GenExpHalfGauss(exp_tau=_scalar(tau_keys, init.get('exp_tau', -9.0)),
                           peak=_scalar(peak_keys, init.get('peak', -9.0)),
                           sigma=_scalar(sig_keys, init.get('sigma', -9.0)),
                           ratio=_scalar(ratio_keys, init.get('ratio', 0.0)),
                           range=gen_range,
                           prob_expon_rewgt=init.get('prob_expon_rewgt', 1.0))


_DUST_KEY_BY_NAME = {name: p.key for p in HOST_DUST_PARAMETERS
                     for name in p.input_names}
# the spellings the dust profiles own, which the GenGauss scan leaves alone
_DUST_PROFILE_NAMES = tuple(name for p in HOST_DUST_PARAMETERS
                            if p.exp_halfgauss for name in p.input_names)

def _parse_genpdf_flat(raw) -> List[Tuple[str, Tuple[float, float]]]:
    """Parse GENPDF_FLAT = VAR(min:max),VAR2(min:max) (prep_GENPDF_FLAT,
    snlc_sim.c:8055). Range is mandatory; returns (jax_varname, (lo, hi))."""
    text = (''.join(raw) if isinstance(raw, (list, tuple))
            else str(raw)).replace(' ', '')
    out = []
    for tok in (t for t in text.split(',') if t):
        m = re.fullmatch(r'([A-Za-z0-9_]+)\(([^:]+):([^:]+)\)', tok)
        if m is None:
            raise SNANAError(
                f"GENPDF_FLAT term '{tok}' must be VAR(min:max) with a range.",
                function_name="_parse_genpdf_flat")
        var = _DUST_KEY_BY_NAME.get(m.group(1), m.group(1))
        out.append((var, (float(m.group(2)), float(m.group(3)))))
    return out


def _input_genranges(raw: dict, model_cls, structured_config: Dict[str, Any]
                     ) -> Dict[str, Tuple[float, float]]:
    """Each of the model's population parameters' GENRANGE as the input
    gives it, else the default range the model declares
    (INPUTS.GENGAUSS_SALT2c.RANGE and kin, snlc_sim.c:895-909), whatever
    draws the parameter."""
    named = raw.get('POPULATIONS', {})
    names, _, defaults = declared_population(model_cls, structured_config)
    ranges = {}
    for name in names:
        given = _parse_range_tuple(raw.get(f'GENRANGE_{name}'))
        if name in named:
            given = named[name].range
        ranges[name] = (given if given is not None
                        else defaults.get(name, {}).get('range', (0.0, 0.0)))
    return ranges


def _spellings(model_cls, structured_config):
    """Every spelling of the running model's population parameters,
    lower-cased, mapped to its name."""
    names, aliases, _ = declared_population(model_cls, structured_config)
    return population_spellings(names, aliases)


def _registered_population(model_cls, structured_config):
    """The population parameters the input may give keys for, with SNANA's
    defaults for each: the running model's by name, and every other
    registered model's under each of its spellings, since SNANA reads every
    model's keys whatever the GENMODEL."""
    names, _, defaults = declared_population(model_cls, structured_config)
    population_names = set(names)
    inits = {name: defaults[name] for name in names if name in defaults}
    for cls in dict.fromkeys(MODEL_REGISTRY.values()):
        if cls is model_cls or not issubclass(cls, SimulatedModel):
            continue
        for name in cls.population_parameters:
            for spelling in (name, *cls.population_aliases.get(name, ())):
                population_names.add(spelling)
                if name in cls.population_defaults:
                    inits.setdefault(spelling, cls.population_defaults[name])
    return population_names, inits


def _build_distributions(raw: dict, model_cls, structured_config: Dict[str, Any]
                         ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """The per-parameter population distributions and the GENPDF maps.

    GENPDF_FILE may carry embedded GENPEAK_*/GENSIGMA_* keys; like its maps,
    they override the sim input for the parameters they cover (SNANA turns
    them into maps, which win at generation time; init_genPDF_from_GenGauss,
    sntools_genPDF.c:301-312).
    """
    spellings = _spellings(model_cls, structured_config)
    population_names, inits = _registered_population(model_cls,
                                                     structured_config)
    population_names |= {p.key for p in HOST_DUST_PARAMETERS
                         if not p.exp_halfgauss}
    inits['RV'] = _RV_GENGAUSS_INIT

    genpdf_maps: Dict[str, Any] = {}
    prob_expon_rewgt = float(raw.get('GENPDF_EXPON_REWGT', 1.0))

    if 'GENPDF_FILE' in raw:
        genpdf_path = Path(resolve_fits_path(
            os.path.expandvars(str(raw['GENPDF_FILE']))))
        if genpdf_path.exists():
            genpdf_maps, embedded = load_all_genpdf(
                str(genpdf_path), prob_expon_rewgt=prob_expon_rewgt)
            genpdf_maps = _genpdf_maps_by_parameter(genpdf_maps,
                                                    population_names,
                                                    spellings)
            if 'GENMAG_SMEAR' in embedded and 'GENMAG_SMEAR' in raw:
                raise SNANAError(
                    "Cannot define GENMAG_SMEAR in two input files; check "
                    "sim-input, INCLUDE files, and GENPDF_FILE",
                    function_name="init_modelSmear")
            _spell_population_keys(embedded, spellings)
            for key, val_str in embedded.items():
                logger.info("GENPDF_FILE embedded key %s: %s", key, val_str)
                parts = val_str.split()
                try:
                    raw[key] = ([float(p) for p in parts] if len(parts) > 1
                                else float(parts[0]))
                except ValueError:
                    raw[key] = val_str
        else:
            raise SNANAError(
                f"GENPDF_FILE not found: {genpdf_path}. Generating without the "
                f"requested population PDFs would use the wrong distributions.",
                function_name="_build_distributions")

    distributions: Dict[str, Any] = {}
    _scan_gengauss(raw, distributions, population_names, inits)

    # Dust parameters using an exponential-plus-half-Gaussian distribution.
    exp_halfgauss_keys = set()
    for parameter in HOST_DUST_PARAMETERS:
        if parameter.exp_halfgauss:
            exp_halfgauss_keys.add(parameter.key)
            distributions[parameter.key] = _parse_exp_halfgauss(
                raw, parameter.input_names)

    if prob_expon_rewgt != 1.0:
        for param, dist in distributions.items():
            if dist is not None:
                distributions[param] = replace(
                    dist, prob_expon_rewgt=prob_expon_rewgt)

    # GENPDF_FLAT implies GENPDF_IGNORE (snlc_sim.c:8020, sntools_genPDF.c:267)
    ignore_vars = set()
    raw_ignore = raw.get('GENPDF_IGNORE')
    if raw_ignore:
        toks = raw_ignore if isinstance(raw_ignore, (list, tuple)) else [raw_ignore]
        for tok in toks:
            ignore_vars.update(v for v in str(tok).replace(',', ' ').split() if v)
    raw_flat = raw.get('GENPDF_FLAT')
    if raw_flat:
        for var, (lo, hi) in _parse_genpdf_flat(raw_flat):
            var = spellings.get(var.lower(), var)
            ignore_vars.add(var)
            supported = population_names | exp_halfgauss_keys
            if var not in supported:
                raise SNANAError(
                    f"GENPDF_FLAT cannot flatten '{var}'; supported: "
                    f"{', '.join(sorted(supported))}.",
                    function_name="_build_distributions")
            distributions[var] = flat_distribution((lo, hi), dust=var in exp_halfgauss_keys)
    if ignore_vars:
        # matched against the file's spelling (sntools_genPDF.c:264-269)
        genpdf_maps = {k: v for k, v in genpdf_maps.items()
                       if k not in ignore_vars and v.varname not in ignore_vars}

    if raw.get('POPULATIONS'):
        named = raw['POPULATIONS']
        overlap = set(named) & set(genpdf_maps)
        if overlap:
            raise SNANAError(f'Population distributions and GENPDF both specify {sorted(overlap)}')
        distributions.update(named)
    return ({k: v for k, v in distributions.items() if v is not None},
            genpdf_maps)


# the dust profile's own key stems, beyond the GENGAUSS ones it shares
# (parse_input_EXP_HALFGAUSS, sntools_genExpHalfGauss.c:347-406)
_DUST_KEY_STEMS = ('GENTAU_', 'GENSIG_', 'GENRATIO_', 'GENGAUPEAK_',
                   'GENEXPTAU_')


def _subpop_key_parameter(key: str, population_names) -> Optional[str]:
    """The parameter a class-scoped key sets, empty for GENMAG_OFF_GLOBAL,
    None for a key no class may set.

    Both halves of the key are checked. A prefix alone would admit
    GENRANGE_REDSHIFT and every misspelt suffix, which the class would then
    hold and nothing would read.
    """
    if key == 'GENMAG_OFF_GLOBAL':
        return ''
    for stem in _GENGAUSS_KEY_STEMS + _DUST_KEY_STEMS:
        name = key[len(stem):]
        if key.startswith(stem) and name in population_names:
            return _DUST_KEY_BY_NAME.get(name, name)
    return None


def _build_subpops(raw: dict, model_cls, structured_config: Dict[str, Any],
                   genpdf_maps: Dict[str, Any], distributions: Dict[str, Any]) -> List[Any]:
    """The SUBPOP classes, and per class the population its keys describe.

    Each class's distributions come from this module's own builder run over
    the input's keys overlaid with the class's, so a key a class does not set
    keeps the shared value. Named shared distributions inherit class
    coefficients through the same family readers.
    """
    if raw.get('SUBPOPS'):
        classes = []
        seen = set()
        for subpop in raw['SUBPOPS']:
            if subpop.genmag_off_global is None:
                subpop = replace(subpop, genmag_off_global=float(raw.get('GENMAG_OFF_GLOBAL', field_default(Model, 'genmag_off_global'))))
            if subpop.name in seen or not np.isfinite(subpop.weight) or subpop.weight < 0:
                raise SNANAError('SUBPOP names must be unique and weights finite and non-negative')
            seen.add(subpop.name)
            shared = dict(distributions)
            own = configured_distributions(subpop.populations, model_cls, structured_config)
            if set(own) & set(genpdf_maps):
                raise SNANAError('SUBPOP distributions cannot replace GENPDF parameters')
            shared.update(own)
            classes.append((subpop, shared, {}))
        if not sum(item[0].weight for item in classes) > 0:
            raise SNANAError('SUBPOP weights must have a positive total')
        return classes
    lines = raw.get('SUBPOP') or []
    if not lines:
        return []

    spellings = _spellings(model_cls, structured_config)
    population_names, _ = _registered_population(model_cls, structured_config)
    population_names |= {p.key for p in HOST_DUST_PARAMETERS}
    population_names |= {name for p in HOST_DUST_PARAMETERS
                         for name in p.input_names}

    weights: Dict[str, float] = {}
    overrides: Dict[str, Dict[str, Any]] = {}
    set_parameters: Dict[str, set] = {}
    variations: Dict[str, List[str]] = {}
    order: List[str] = []
    for text in lines:
        tokens = str(text).split()
        if len(tokens) < 3:
            raise SNANAError(
                f"SUBPOP takes '<class> <key> <values>', got '{text}'",
                function_name="_build_subpops")
        name, key, rest = (tokens[0],
                           spelled_by_name(tokens[1], _GENGAUSS_KEY_STEMS, spellings),
                           tokens[2:])
        if name not in order:
            order.append(name)
        if key == 'VARIATION':
            variations.setdefault(name, []).append(' '.join(rest))
            continue
        if key == 'WGT':
            if len(rest) != 1:
                raise SNANAError(
                    f"SUBPOP {name} WGT takes one number, got '{text}'",
                    function_name="_build_subpops")
            if name in weights:
                raise SNANAError(f"SUBPOP {name} sets WGT twice",
                                 function_name="_build_subpops")
            try:
                weights[name] = float(rest[0])
            except ValueError:
                raise SNANAError(
                    f"SUBPOP {name} WGT is not a number: '{rest[0]}'",
                    function_name="_build_subpops") from None
            continue
        parameter = _subpop_key_parameter(key, population_names)
        if parameter is None:
            raise SNANAError(
                f"SUBPOP cannot set '{key}': a class may set its population "
                f"parameters, GENMAG_OFF_GLOBAL, WGT, and VARIATION on those. "
                f"Anything else describes a different run, not a population.",
                function_name="_build_subpops")
        if key in overrides.get(name, {}):
            raise SNANAError(f"SUBPOP {name} sets {key} twice",
                             function_name="_build_subpops")
        overrides.setdefault(name, {})[key] = _parse_value(' '.join(rest))
        if parameter:
            set_parameters.setdefault(name, set()).add(parameter)

    missing = [n for n in order if n not in weights]
    if missing:
        raise SNANAError(
            f"SUBPOP class(es) {', '.join(missing)} have no WGT. Every class "
            f"needs one; the weights are relative, so they need not sum to 1.",
            function_name="_build_subpops")

    for name in order:
        for parameter in sorted(set_parameters.get(name, ())):
            if parameter in genpdf_maps:
                raise SNANAError(
                    f"SUBPOP {name} sets {parameter}, which a GENPDF map also "
                    f"samples. The map belongs to the run, so it would draw "
                    f"every class alike; state the population one way or the "
                    f"other.", function_name="_build_subpops")

    classes = []
    for name in order:
        merged = dict(raw)
        merged.update(overrides.get(name, {}))
        if raw.get('POPULATIONS'):
            named = dict(raw['POPULATIONS'])
            for parameter in set_parameters.get(name, ()):
                if parameter not in named:
                    continue
                inherited = named[parameter]
                if isinstance(inherited, GenExpHalfGauss):
                    varnames = next(p.input_names for p in HOST_DUST_PARAMETERS
                                    if p.key == parameter)
                    updated = _parse_exp_halfgauss(
                        overrides[name], varnames, asdict(inherited))
                else:
                    updated = _parse_gengauss_asym(
                        overrides[name], parameter, asdict(inherited))
                if updated is not None:
                    named[parameter] = updated
            merged['POPULATIONS'] = named
        dists, _ = _build_distributions(merged, model_cls, structured_config)
        scoped = load_variation(
            variations.get(name, []), spellings,
            'population_parameters' in structured_config,
            _file_or_none(raw.get('HOSTLIB_FILE')), scope=name,
            extra_targets=('WGT',))
        set_keys = overrides.get(name, {})
        subpop = SubPopulation(
            name=name,
            weight=weights[name],
            genmag_off_global=float(
                set_keys.get('GENMAG_OFF_GLOBAL',
                             raw.get('GENMAG_OFF_GLOBAL',
                                     field_default(Model,
                                                   'genmag_off_global')))))
        classes.append((subpop, dists, scoped))
    return classes


def _shifts_on(param_name, zvariation, *variations):
    """The shift entries that move this parameter, run-level and class-scoped.

    A target names the component it moves, so the parameter it belongs to is
    read from the component table rather than by splitting the name.
    """
    entries = []
    for target in sorted(targets(param_name)):
        if target in zvariation:
            entries.append(zvariation[target])
        for variation in variations:
            entries.extend(variation.get(target, ()))
    return tuple(entries)


def _build_populations(distributions, genpdf_maps, zvariation, variation,
                       classes):
    """One record per parameter, and the shifts on the targets the run applies
    rather than draws.

    A class contributes its own distribution only where it differs from the
    shared one, so a parameter every class agrees on keeps a single
    distribution and is drawn once rather than once per class. A target no
    parameter claims is one the simulator applies directly, GENMAG_OFF_GLOBAL
    or a class's WGT, and its entries are kept per class for the site that
    applies them.
    """
    per_class = [dists for _, dists, _ in classes] or [distributions]
    for dists in [distributions, *per_class]:
        configured = [p.key for p in HOST_DUST_PARAMETERS
                      if p.exp_halfgauss and dists.get(p.key) is not None]
        if len(configured) > 1:
            raise SNANAError(
                f"Cannot specify more than one host dust distribution: "
                f"{', '.join(configured)} are configured. Use one or the other.",
                function_name="_build_populations")
    scoped = [var for _, _, var in classes] or [{}]
    names = set(distributions) | set(genpdf_maps)
    names.update(name for d in per_class for name in d)

    populations = {}
    claimed = set()
    for name in sorted(names):
        claimed.update(targets(name))
        if name in genpdf_maps:
            if _shifts_on(name, zvariation, variation, *scoped):
                raise SNANAError(
                    f"A GENPDF map samples {name}, and a ZVARIATION or "
                    f"VARIATION also shifts it. The map is the tabulated form "
                    f"of the same statement, so state it one way or the other.",
                    function_name="_build_populations")
            populations[name] = Population(genpdf=genpdf_maps[name])
            continue
        if any(name not in d for d in per_class):
            raise SNANAError(
                f"{name} is configured for some SUBPOP classes but not "
                f"others, so the run cannot draw it for every event. Declare "
                f"the unqualified keys as well, giving the classes that do "
                f"not set it a shared value.",
                function_name="_build_populations")
        dists = [d[name] for d in per_class]
        scoped_here = any(t in var for var in scoped for t in targets(name))
        if all(d == dists[0] for d in dists) and not scoped_here:
            dists, per_scope = dists[:1], [{}]
        else:
            per_scope = scoped
        populations[name] = Population(
            distributions=tuple(dists),
            shifts=tuple(_shifts_on(name, zvariation, variation, s)
                         for _, s in zip(dists, per_scope)))

    applied = {}
    for target in sorted(set(zvariation) | set(variation)
                         | {t for var in scoped for t in var}):
        if target in claimed:
            continue
        base = ([zvariation[target]] if target in zvariation else []
                ) + list(variation.get(target, ()))
        applied[target] = tuple(tuple(base + list(var.get(target, ())))
                                for var in scoped)
    return populations, applied


def _genpdf_maps_by_parameter(maps, population_names, spellings
                              ) -> Dict[str, Any]:
    """Key each GENPDF map by the parameter it samples.

    A map's first VARNAME is the parameter, under any of its spellings, or
    LOG<parameter> for a map tabulated in log10 (IMAP_GENPDF,
    sntools_genPDF.c:965-986). A name nothing draws aborts, as SNANA's
    check against the model's own list does (checkVarGen_genPDF, :572)
    after its two named mistakes, SALT2 parameters without their prefix
    and output columns with a SIM_ one (checkAbort_VARNAME_GENPDF, :547).
    """
    accepted = population_names | set(_DUST_KEY_BY_NAME)

    def parameter(spelling):
        spelling = spellings.get(spelling.lower(), spelling)
        if spelling not in accepted:
            return None
        return _DUST_KEY_BY_NAME.get(spelling, spelling)

    by_parameter = {}
    for name, genpdf in maps.items():
        if parameter(name) is not None:
            # the exact match wins over the LOG prefix
            by_parameter[parameter(name)] = replace(genpdf, is_log=False)
        elif name.startswith('LOG') and parameter(name[3:]) is not None:
            by_parameter[parameter(name[3:])] = genpdf
        else:
            raise SNANAError(
                f"GENPDF map '{name}' names no population parameter of any "
                f"model, so nothing would draw from it. Parameters: "
                f"{', '.join(sorted(accepted))}",
                function_name="_build_distributions")
    return by_parameter


def _per_filter_offsets(raw: dict, key: str, genfilters: str) -> Dict[str, float]:
    """A space-separated float per GENFILTERS band (GENMAG_OFF_ZP / _MODEL)."""
    value = raw.get(key)
    if value is None:
        return {}
    values = (value if isinstance(value, (list, tuple)) else str(value).split())
    if len(values) != len(genfilters):
        raise SNANAError(
            f"{key} has {len(values)} values; expected {len(genfilters)} for "
            f"GENFILTERS='{genfilters}'", function_name="_per_filter_offsets")
    return {c: float(v) for c, v in zip(genfilters, values)}


def _key_plus_filter(raw: dict, base: str, default: float,
                     genfilters: str) -> Dict[str, float]:
    """SNANA KEY_PLUS_FILTER syntax (parse_input_KEY_PLUS_FILTER,
    snlc_sim.c:3648-3741): the bare key broadcasts to every band, and each
    KEY_FILTER occurrence sets its named bands, applied cumulatively. The
    per-band comma pairing activates only when the filter token itself
    contains commas; otherwise one value serves every named band. Nothing
    set gives an empty map, which reads the same to every consumer as an
    all-neutral one. SNANA applies bare and _FILTER keys in file order; the
    bare key is applied first here, which matches the global-then-per-band
    layout every known input uses."""
    rows = raw.get(f'{base}_FILTER_ROWS', [])
    if base not in raw and not rows:
        return {}
    out = {c: default for c in genfilters}
    value = raw.get(base)
    if value is not None:
        for c in genfilters:
            out[c] = float(value)
    for row in rows:
        parts = ([str(x) for x in row] if isinstance(row, (list, tuple))
                 else str(row).split())
        band_token = parts[0]
        bands = band_token.replace(',', '')
        if ',' in band_token:
            values = parts[1].split(',')
            if len(values) != len(bands):
                raise SNANAError(
                    f"{base}_FILTER: {len(bands)} bands but "
                    f"{len(values)} values",
                    function_name="_key_plus_filter")
        else:
            # one value for every named band (snlc_sim.c:3697-3703)
            values = [parts[1].split(',')[0]] * len(bands)
        for c, v in zip(bands, values):
            out[c] = float(v)
    return out


def _genmag_smear_sigma(value) -> float:
    """GENMAG_SMEAR as one coherent sigma.

    SNANA reads the key with split2floats on a comma (snlc_sim.c:5272) and
    draws an asymmetric coherent smear when the two sigmas differ
    (:14586-14593); jax implements the symmetric case only.
    """
    text = str(value).strip()
    if ',' in text:
        raise SNANAError(
            f"GENMAG_SMEAR: {text} asks for the asymmetric siglo,sighi form, "
            f"which jax does not implement. Give one sigma.",
            function_name="_genmag_smear_sigma")
    return float(text)


def _genmag_smear_filter(raw: dict) -> Dict[str, float]:
    """GENMAG_SMEAR_FILTER: repeatable '<filterlist> <sigma>' (snlc_sim.c:5306)."""
    out: Dict[str, float] = {}
    for row in raw.get('GENMAG_SMEAR_FILTER', []):
        toks = row if isinstance(row, (list, tuple)) else str(row).split()
        sigma = float(toks[1])
        for band in str(toks[0]).replace(',', ''):
            out[band] = sigma
    return out


def _genmag_smear_scale_poly(raw: dict, model_cls, structured_config
                             ) -> Optional[Tuple[str, Tuple[float, ...]]]:
    """GENMAG_SMEAR_SCALE(VAR): scale = eval_poly(VAR) (snlc_sim.c:5333,
    sntools_genSmear.c:742). VAR may be any spelling of a population
    parameter, which the sampler reads by name."""
    spellings = _spellings(model_cls, structured_config)
    for key, value in raw.items():
        if key.startswith('GENMAG_SMEAR_SCALE(') and key.endswith(')'):
            var = key[len('GENMAG_SMEAR_SCALE('):-1]
            var = spellings.get(var.lower(), var)
            return (var, tuple(float(x) for x in
                               str(value).replace(',', ' ').split()))
    return None


def _check_rejected_keys(typed_keys: set) -> None:
    """Abort on input keys jax refuses rather than silently dropping."""
    found = sorted(typed_keys & set(_REJECTED_KEYS))
    if found:
        raise SNANAError(
            "Unsupported SNANA input key(s):\n"
            + "\n".join(f"  {k}: {_REJECTED_KEYS[k]}" for k in found)
            + "\nThe SNANA differences section of the documentation has the detail.",
            function_name="_check_rejected_keys")


def _check_misnamed_simsed_params(typed_keys: set) -> None:
    """Abort on GEN*_SIMSED_PARAM(<name>), which sets nothing in either engine.

    A SIMSED distribution is declared by naming the parameter in the key
    itself: "SIMSED_PARAM: stretch" then GENPEAK_stretch. Writing the literal
    words SIMSED_PARAM in place of the parameter name reads as a range but
    sets no distribution; SNANA drops it without a word for any GENMODEL
    (snlc_sim.c:2250 routes every *SIMSED* key into parse_input_SIMSED,
    where nothing matches).
    """
    found = sorted(k for k in typed_keys
                   if k.startswith('GEN') and 'SIMSED_PARAM(' in k)
    if found:
        raise SNANAError(
            "SIMSED parameter key(s) name no parameter: " + ", ".join(found)
            + ". Put the parameter name in the key, 'SIMSED_PARAM: stretch',"
            " then 'GENRANGE_stretch: 0.65 1.25'. As written the range is"
            " silently dropped and every event uses grid entry 0.",
            function_name="_check_misnamed_simsed_params")


def _check_extinc_hostgal(raw: dict) -> None:
    """EXTINC_HOSTGAL selects OPT_SNXT: CCM89, or SJPAR which SNANA itself
    rejects as unavailable (genmag_extinction.c:44-46); anything else is
    fatal (snlc_sim.c:6833-6836)."""
    value = raw.get('EXTINC_HOSTGAL')
    if value is None or str(value).strip() == 'CCM89':
        return
    value = str(value).strip()
    detail = ("SNANA rejects it too: 'not yet available in refactored C code'"
              if value == 'SJPAR' else "SNANA accepts only CCM89 or SJPAR")
    raise SNANAError(f"EXTINC_HOSTGAL: {value} is not supported: {detail}.",
                     function_name="_check_extinc_hostgal")


def _check_gensource(raw: dict) -> None:
    """Only GENSOURCE: RANDOM is supported; GRID is a different product.

    SNANA branches on GENSOURCE (snlc_sim.c:118-138): RANDOM simulates a
    survey, GRID calls init_GRIDsource and writes a model grid for PSNID
    photometric templates instead of light curves, and anything else is a
    fatal abort. jax implements only RANDOM, so an unhandled value must not
    quietly produce a survey the user did not ask for.
    """
    value = raw.get('GENSOURCE')
    if value is None or str(value).strip().upper() == 'RANDOM':
        return
    detail = ("GRID writes a model grid for PSNID photometric templates "
              "(NGRID_* / GRID_FORMAT keys), not simulated light curves"
              if str(value).strip().upper() == 'GRID'
              else "SNANA accepts only RANDOM or GRID")
    raise SNANAError(
        f"GENSOURCE: {value} is not supported: jax implements RANDOM only. "
        f"{detail}.", function_name="_check_gensource")


def _check_explosion_anchor(raw: dict, model_type: str) -> None:
    """MJD_EXPLODE / OPTMASK_T0SHIFT_EXPLODE anchor the light curve to
    explosion (manual §6301). NON1ASED/SIMSED reanchor the SED day grid;
    redback is already explosion-anchored (OPTMASK no-op)."""
    anchor = MODEL_REGISTRY[model_type].mjd_explode_anchor
    if anchor not in ('none', 'optmask', 'native'):
        raise SNANAError(
            f"{MODEL_REGISTRY[model_type].__name__} declares "
            f"mjd_explode_anchor={anchor!r}; the states are 'none', "
            f"'optmask' and 'native'.",
            function_name="_check_explosion_anchor")
    if raw.get('MJD_EXPLODE') is None:
        return
    if anchor == 'none':
        supported = ', '.join(sorted(
            name for name, cls in MODEL_REGISTRY.items()
            if issubclass(cls, SimulatedModel)
            and cls.mjd_explode_anchor != 'none'))
        raise SNANAError(
            f"MJD_EXPLODE is not supported for GENMODEL type '{model_type}' "
            f"({supported} only).",
            function_name="_check_explosion_anchor")
    if anchor == 'optmask' and int(raw.get('OPTMASK_T0SHIFT_EXPLODE') or 0) < 1:
        raise SNANAError(
            f"MJD_EXPLODE requires OPTMASK_T0SHIFT_EXPLODE>=1 for GENMODEL "
            f"type '{model_type}' (SNANA has no valid DAY=0=explosion "
            f"default for it).",
            function_name="_check_explosion_anchor")


def _check_spectra_sources(cfg: SimulationConfig, raw: dict) -> None:
    """TAKE_SPECTRUM comes from the SIMLIB header or the sim-input file."""
    if 'USE_SIMLIB_TAKE_SPECTRUM' in raw and 'USE_SIMLIB_SPECTRA' in raw:
        # USE_SIMLIB_SPECTRA is the pre-Aug-2025 spelling (snlc_sim.c:3833)
        raise SNANAError(
            "USE_SIMLIB_TAKE_SPECTRUM and USE_SIMLIB_SPECTRA are aliases; "
            "set only one.", function_name="_check_spectra_sources")
    if not cfg.survey.use_simlib_take_spectrum:
        return
    # the manual documents an abort; the code lets the SIMLIB win (:21912)
    if cfg.spectra.take_spectrum:
        raise SNANAError(
            "TAKE_SPECTRUM keys are taken from either the SIMLIB header or "
            "the sim-input file, not both; remove one.",
            function_name="_check_spectra_sources")
    # both SIMLIB spectra modes at once makes GENSPEC_MJD_ORDER take its
    # NOBS_SPECTROGRAPH branch (:10909), silently disabling near-peak
    if cfg.survey.use_simlib_spectrograph:
        raise SNANAError(
            "USE_SIMLIB_TAKE_SPECTRUM and USE_SIMLIB_SPECTROGRAPH cannot "
            "both be set.", function_name="_check_spectra_sources")


def _check_scalar_keys(raw: dict) -> None:
    """Aborts that need one key each and nothing else."""
    if 'GENRANGE_PEAKMAG' in raw:
        raise SNANAError(
            "GENRANGE_PEAKMAG is a per-filter peakmag bounds check that "
            "requires rejection sampling (forward-model output). Not "
            "implemented in jax_SNANA.", function_name="_check_scalar_keys")

    # GENMAG_SMEAR_MSKOPT bit 32, SNANA's only bit, is output-neutral
    # (sntools_genSmear.c:458)
    try:
        smear_mskopt = int(raw.get('GENMAG_SMEAR_MSKOPT', 0))
    except ValueError:
        raise SNANAError(
            f"GENMAG_SMEAR_MSKOPT={raw['GENMAG_SMEAR_MSKOPT']!r} is not an "
            f"integer bitmask.", function_name="_check_scalar_keys") from None
    if smear_mskopt & ~32:
        raise SNANAError(
            f"GENMAG_SMEAR_MSKOPT={smear_mskopt} sets unsupported bit(s) "
            f"{smear_mskopt & ~32}; only 32 (no-repeat) is defined in SNANA.",
            function_name="_check_scalar_keys")

    for a, b in (('FUDGESCALE_NOISE_SKY', 'FUDGESCALE_SKYNOISE'),
                 ('FUDGESCALE_NOISE_READ', 'FUDGESCALE_READNOISE')):
        if a in raw and b in raw:
            raise SNANAError(f"{a} and {b} are aliases; set only one",
                             function_name="_check_scalar_keys")


def _warn_unrecognized_keys(raw: UsedTrackingDict, typed_keys: set) -> None:
    """Log one warning listing input keys the build never read."""
    unknown = sorted(
        k for k in typed_keys
        if k not in raw.used
        and k not in _REJECTED_KEYS
        and not any(k.startswith(p) for p in _OPEN_KEY_PREFIXES))
    if unknown:
        log.warn(logger, "CONFIG",
                 f"Unrecognized SNANA input key(s) ignored: {', '.join(unknown)}")


# GENMODELs that set magnitudes directly (snlc_sim.c:9463-9479)
_UNIMPLEMENTED_GENMODELS = {
    'FIXMAG': "every epoch gets one fixed magnitude",
    'RANMAG': "each observation gets a magnitude drawn flat in a range",
    'SIMLIB': "magnitudes are read from the SIMLIB itself",
    'LCLIB': "light curves are replayed from a library of recurring sources",
}


def _check_genmodel_implemented(genmodel: Any) -> None:
    """Reject the GENMODELs that set magnitudes directly, rather than
    mis-parsing them.

    Their values carry trailing numbers, so the multi-token form reaches path
    handling as a list and dies in posixpath; the bare `SIMLIB` form parses to
    an empty model_type and is accepted silently.
    """
    name = str(genmodel[0] if isinstance(genmodel, list) else genmodel).split()[0]
    detail = _UNIMPLEMENTED_GENMODELS.get(name.upper())
    if detail:
        raise SNANAError(
            f"GENMODEL: {name} is not implemented: {detail}, bypassing the "
            f"SED-to-filter path jax generates through.",
            function_name="_check_genmodel_implemented")


def build(raw: Dict[str, Any], *, settings=None) -> SimulationConfig:
    """Interpret the input's keys into a SimulationConfig, in one pass."""
    if not isinstance(raw, UsedTrackingDict):
        raw = UsedTrackingDict(raw)
    settings = settings or {}

    def complete(group, record):
        # Each owner receives every supplied value before dependent preparation.
        # Fields without specialised SNANA syntax need no compatibility reader.
        types, _ = dataclass_settings(type(record))
        updates = {}
        aliases = getattr(type(record), 'input_names', {})
        for name, annotation in types.items():
            keys = [name.upper(), *aliases.get(name, ())]
            # A specialised reader owns all aliases of a value it consumed.
            if any(key in raw.used and key in raw.keys() for key in keys):
                continue
            supplied = [key for key in keys if key in raw.keys()]
            if supplied:
                key = supplied[-1]
                updates[name] = convert(raw[key], annotation, key)
        updates.update(settings.get(group, {}))
        return replace(record, **updates)

    genmodel = _require(raw, 'GENMODEL', "build")
    _check_genmodel_implemented(genmodel)
    genfilters = str(_require(raw, 'GENFILTERS', "build"))
    model_type = infer_model_type_from_path(genmodel) or ''
    if model_type not in MODEL_REGISTRY:
        raise SNANAError(
            f"GENMODEL: {genmodel} matches no registered model "
            f"({', '.join(sorted(MODEL_REGISTRY))}).",
            function_name="build")
    model_cls = MODEL_REGISTRY[model_type]
    if not issubclass(model_cls, SimulatedModel):
        raise SNANAError(
            f"GENMODEL: {genmodel} cannot be simulated: "
            f"{model_cls.__name__} does not implement the SimulatedModel role.",
            function_name="build")

    # GENPOP_ASYMGAUSS overrides the SALT2 x1/c population (snlc_sim.c:6437)
    _apply_genpop_asymgauss(raw)

    # general sections first, so known_keys records them before a model error
    run = complete('run', _run(raw))
    cosmology = complete('cosmology', _cosmology(raw))
    survey = complete('survey', _survey(raw, genfilters))
    host = complete('host', _host(raw))
    mw_dust = complete('mw_dust', _mw_dust(raw))
    instrument = complete('instrument', _instrument(raw, genfilters))
    lensing = complete('lensing', _lensing(raw))
    spectra = complete('spectra', _spectra(raw))
    spectra.take_spectrum = prepare_spectrum_requests(spectra.take_spectrum)
    cuts = complete('cuts', _cuts(raw))
    output = complete('output', _output(raw))

    rates = complete('rates', _rates(raw, model_type, _dndz_entries(raw, 'DNDZ_PEC1A')))
    rates.dndz_rate_mods = asdict(build_dataclass(RateModifiers, rates.dndz_rate_mods))
    rates.dndz_entries = _dndz_entries({'DNDZ': rates.dndz_entries})
    rates.dndz_entries_pec1a = _dndz_entries({'DNDZ_PEC1A': rates.dndz_entries_pec1a}, 'DNDZ_PEC1A')
    if rates.dndz_entries_pec1a and not model_cls.non1a_templates:
        raise SNANAError("DNDZ_PEC1A key allowed only for NON1A model.", function_name="build")
    if rates.gensigma_peakmjd > 1.0e-9:
        rates.opt_setpkmjd = 0
    peak_keys = ['PEAKMJD_RANGE', *Rates.input_names['peakmjd_range']]
    if not any(key in raw for key in peak_keys) and 'peakmjd_range' not in settings.get('rates', {}):
        if raw.get('MJD_EXPLODE') is not None:
            anchor = float(raw['MJD_EXPLODE'])
            rates.peakmjd_range = (anchor, anchor)
        elif survey.simlib_dump > 0 and rates.genrange_mjd:
            rates.peakmjd_range = rates.genrange_mjd
    if cuts.epochs_time_cut is not None:
        cuts.epochs_time_cut = asdict(build_dataclass(EpochsTimeCut, cuts.epochs_time_cut))
    if len(cuts.snrmax_cuts) > 5 or any(not 1 <= n <= len(filters)
                                     for _, filters, n, _, _ in cuts.snrmax_cuts):
        raise SNANAError('snrmax_cuts requires at most five cuts and valid filter counts')

    structured_config = model_cls.structured_input(raw)
    spellings = _spellings(model_cls, structured_config)
    names_from_input = 'population_parameters' in structured_config
    _spell_population_keys(raw, spellings)
    # before a GENPDF file's embedded keys join raw
    input_genranges = _input_genranges(raw, model_cls, structured_config)
    distributions, genpdf_maps = _build_distributions(raw, model_cls,
                                                      structured_config)
    classes = _build_subpops(raw, model_cls, structured_config, genpdf_maps,
                             distributions)
    # Report the same run-level dust profile that supplies the population draws.
    # WV07 has its own range and scale, independent of the AV population record.
    if 'AV' in distributions:
        host.dust_enabled = True
        if not host.genav_wv07 or 'GENRANGE_AV' not in raw:
            host.av_range = distributions['AV'].range
        if not host.genav_wv07:
            host.tau_av = distributions['AV'].exp_tau
    model = complete('model', _model(raw, genfilters, genmodel, model_type, {},
                     structured_config, tuple(sp for sp, _, _ in classes),
                     {}, None, input_genranges))
    zvar_file = model.zvariation_file
    zvariation = load_zvariation(
        model.zvariation_poly or [],
        resolve_input_path(zvar_file, run) if zvar_file else '',
        spellings, names_from_input, rates.redshift_range)
    variation = load_variation(raw.get('VARIATION') or [], spellings,
                               names_from_input, host.hostlib_file)
    populations, offset_shifts = _build_populations(
        distributions, genpdf_maps, zvariation, variation, classes)

    cfg = SimulationConfig(
        run=run,
        rates=rates,
        cosmology=cosmology,
        model=replace(model, populations=populations, _offset_shifts=offset_shifts,
                      _ebv_igm_shift=zvariation.get('EBV_IGM')),
        numerical=complete('numerical', NumericalSettings()),
        survey=survey,
        host=host,
        mw_dust=mw_dust,
        instrument=instrument,
        lensing=lensing,
        spectra=spectra,
        cuts=cuts,
        output=output,
    )

    return cfg


def validate(cfg: SimulationConfig, raw: Dict[str, Any]) -> None:
    """Reject configurations SNANA rejects.

    Separate from `build` because SNANA validates only after command-line
    overrides are applied (snlc_sim.c:680-683), so an override can supply or
    repair whatever a check would otherwise reject.

    Every check runs before any of them aborts, so a user with several bad keys
    sees them all at once, and `known_keys` can run this pass purely to
    record the keys the checks read.
    """
    checks = (
        lambda: _check_scalar_keys(raw),
        lambda: _check_extinc_hostgal(raw),
        lambda: _check_gensource(raw),
        lambda: _check_spectra_sources(cfg, raw),
        lambda: _check_explosion_anchor(raw, cfg.model.model_type),
        lambda: _check_redshift_range(cfg),
        lambda: _check_genranges(cfg),
        lambda: _check_host_spectra(cfg),
        lambda: _check_batch_split(cfg),
        lambda: _check_subpop_simlib_conflict(cfg),
    )
    failures = []
    for check in checks:
        try:
            check()
        except SNANAError as exc:
            failures.append(exc)
    if len(failures) == 1:
        raise failures[0]
    if failures:
        joined = "\n   ".join(exc.message for exc in failures)
        raise SNANAError(f"{len(failures)} invalid inputs:\n   {joined}",
                         function_name="validate")


def _check_subpop_simlib_conflict(cfg: SimulationConfig) -> None:
    """Refuse a population statement a SIMLIB entry header would overwrite.

    USE_SIMLIB_SALT2 takes the parameter from the entry header for the events
    drawn on it (_simlib_header_override), so a class or a shift on it would
    collapse silently.
    """
    if not cfg.survey.use_simlib_salt2:
        return
    named = sorted(name for name, pop in cfg.model.populations.items()
                   if pop.is_shaped())
    if named:
        raise SNANAError(
            f"USE_SIMLIB_SALT2 takes {', '.join(named)} from the SIMLIB entry "
            f"header, which would discard the population this run states for "
            f"it. Drop the SUBPOP or VARIATION, or USE_SIMLIB_SALT2.",
            function_name="_check_subpop_simlib_conflict")


def _check_batch_split(cfg: SimulationConfig) -> None:
    """SIMLIB_IDLOCK pins one LIBID, so it cannot also be sliced across batch
    jobs (SIMLIB_findStart, snlc_sim.c:19150). IDSTART + JOBID is allowed."""
    if cfg.run.njobtot > 0 and cfg.survey.simlib_idlock > 0:
        raise SNANAError(
            f"SIMLIB_IDLOCK {cfg.survey.simlib_idlock} cannot combine with batch mode "
            f"(NJOBTOT {cfg.run.njobtot}): one locks a single LIBID, the other "
            f"splits the library across jobs.", function_name="_check_batch_split")


def _check_host_spectra(cfg: SimulationConfig) -> None:
    """Validate the host-spectrum keys.

    The files they name are read by `hostlib.spectra.load_host_spec_table`, which
    loads the single table SNANA stores rather than opening both.
    """
    # SNANA segfaults here without a HOSTLIB
    if (any(e['epoch_frame'] == 'HOST' for e in cfg.spectra.take_spectrum)
            and not (cfg.host.specbasis_file or cfg.host.specdata_file)):
        raise SNANAError(
            "Cannot generate host spectrum without spec basis or data. "
            "Check HOSTLIB_SPECBASIS[DATA]_FILE key in sim-input.",
            function_name="_check_host_spectra")

    hostfrac, hostsnfrac = cfg.spectra.take_spectrum_hostfrac, cfg.spectra.take_spectrum_hostsnfrac
    if hostfrac > HOST_CONTAM_MIN and hostsnfrac > HOST_CONTAM_MIN:
        raise SNANAError(
            f"Cannot use both TAKE_SPECTRUM_HOSTFRAC ({hostfrac}) and "
            f"TAKE_SPECTRUM_HOSTSNFRAC ({hostsnfrac}). Pick one.",
            function_name="_check_host_spectra")

    # IMJD_HOST stays -9 without a HOST epoch (snlc_sim.c:9749)
    if (max(hostfrac, hostsnfrac) > HOST_CONTAM_MIN
            and not any(e['epoch_frame'] == 'HOST' for e in cfg.spectra.take_spectrum)):
        raise SNANAError(
            "Host contamination needs a host spectrum to add. Check for a "
            "TAKE_SPECTRUM: HOST key in sim-input.",
            function_name="_check_host_spectra")


def _check_redshift_range(cfg: SimulationConfig) -> None:
    """zmin must be positive (snlc_sim.c:27000)."""
    zmin, zmax = cfg.rates.redshift_range
    if zmin <= 0.0:
        raise SNANAError(
            f"Invalid GENRANGE_REDSHIFT zmin={zmin} (must be > 0); "
            f"zmax={zmax}. Set GENRANGE_REDSHIFT in the sim-input file.",
            function_name="build")


def _check_genranges(cfg: SimulationConfig) -> None:
    """SNANA inits GENRANGE_TREST and GENRANGE_PEAKMJD to (0, 0)
    (snlc_sim.c:841-842, 862-863). A degenerate TREST range is fatal at
    init (:7310-7316); a degenerate PEAKMJD range matches no SIMLIB epoch
    and aborts ("GENRANGE_PEAKMJD and SIMLIB probably don't overlap",
    :567-569); both are rejected here with the missing key named."""
    trest_lo, trest_hi = cfg.rates.trest_range
    if trest_lo >= trest_hi:
        raise SNANAError(
            f"Invalid GENRANGE_TREST: {trest_lo} {trest_hi}. "
            f"Set GENRANGE_TREST in the sim-input file.",
            function_name="_check_genranges")
    pk_lo, pk_hi = cfg.rates.peakmjd_range
    if pk_lo >= pk_hi and cfg.model.mjd_explode is None:
        raise SNANAError(
            f"Invalid GENRANGE_PEAKMJD: {pk_lo} {pk_hi}. Set "
            f"GENRANGE_PEAKMJD (or MJD_EXPLODE) in the sim-input file.",
            function_name="_check_genranges")


def _gentypes(raw: dict) -> Dict[str, int]:
    """GENTYPES gives the spec/phot pair; GENTYPE gives spec alone and implies
    spec+100 (snlc_sim.c:1918-1926). GENTYPE_SPEC inits to -9 and only
    overrides when > 0 (snlc_sim.c:814)."""
    gentypes = raw.get('GENTYPES')
    if gentypes is not None:
        return {'gentype': int(gentypes[0]), 'gentype_phot': int(gentypes[1])}
    return _supplied(raw, {'gentype': ('GENTYPE', int)})


def _sntypes(raw: dict) -> Dict[str, int]:
    """SNTYPE_Ia sets both types from one value; SNTYPES_Ia gives them
    separately (snlc_sim.c:1909-1917). OFFSET_TYPE_PHOT = 100."""
    pair = raw.get('SNTYPES_Ia')
    if pair is not None:
        return {'sntype_spec': int(pair[0]), 'sntype_phot': int(pair[1])}
    one = raw.get('SNTYPE_Ia')
    if one is not None:
        return {'sntype_spec': int(one), 'sntype_phot': int(one) + 100}
    return {}


def is_key(token: str, known: set) -> bool:
    """Whether a command-line token names a key rather than a value.

    The parser consumes `_OPEN_KEY_PREFIXES` families by iterating the dict, so
    their members are never looked up by name and never recorded as used. They
    are still keys SNANA's own parser matches, so the test answers for the
    family rather than for an enumeration nobody maintains.
    """
    return token in known or token.startswith(_OPEN_KEY_PREFIXES)


def build_validated(raw, *, settings=None):
    """Apply simulation-wide input rules before building either input format."""
    _apply_genperfect(raw)
    cfg = build(raw, settings=settings)
    validate(cfg, raw)
    return cfg


def _record_key_lookups(probe: UsedTrackingDict) -> bool:
    """Run the build for the key names it looks up, discarding its verdict.
    False when it could not finish, so it read only some of them."""
    try:
        build_validated(probe)
    except _BAD_INPUT:
        return False
    return True


def known_keys(raw: UsedTrackingDict, tokens: Sequence[str]) -> set:
    """The key names this input recognises, for classifying command-line tokens.

    A token is a key iff the build reads it. Aliases, ignored and rejected keys
    are added because `used` records canonical names the build consumed.
    `validate` runs too, for the keys only its checks read.

    A value the file got wrong stops the build before it has read every key,
    and a short set reads a later override key as a value of the one before
    it. So it reads again with the overrides applied, which gets past the bad
    value; `used` only grows.
    """
    probe = UsedTrackingDict(raw)
    for key in _REQUIRED_KEYS:
        if key not in probe and key in tokens:
            probe[key] = tokens[tokens.index(key) + 1]
    if not _record_key_lookups(probe):
        file_keys = set(raw)
        probe.update({token: tokens[i + 1]
                      for i, token in enumerate(tokens[:-1])
                      if token in file_keys})
        _record_key_lookups(probe)
    types, owners, _ = declarations()
    ordinary = {name.upper() for name in types}
    for name, owner in owners.items():
        ordinary.update(getattr(owner, 'input_names', {}).get(name, ()))
    return (ordinary | set(probe.used) | set(_key_aliases())
            | set(_REJECTED_KEYS) | set(_ARGV_ONLY_KEYS))


def _override_values(tokens: Sequence[str], start: int, key: str,
                     known: set) -> Tuple[List[str], int]:
    """The values belonging to `key`, and the index just past them.

    A count-prefixed key takes exactly the number of values it declares, so a
    value that happens to share a key's name is not mistaken for the next key.
    """
    if (key in _COUNT_PREFIXED_KEYS and start < len(tokens)
            and tokens[start].isdigit()):
        count = int(tokens[start])
        end = start + 1 + count
        if end > len(tokens):
            raise SNANAError(
                f"'{key}' declares {count} values but only "
                f"{len(tokens) - start - 1} follow.",
                function_name="group_override_tokens")
        return list(tokens[start:end]), end

    end = start
    while end < len(tokens) and not is_key(tokens[end], known):
        end += 1
    return list(tokens[start:end]), end


def group_override_tokens(tokens: Sequence[str],
                          known: set) -> List[Tuple[str, List[str]]]:
    """Split command-line tokens into (key, values) pairs."""
    groups: List[Tuple[str, List[str]]] = []
    i = 0
    while i < len(tokens):
        key = tokens[i]
        # a refused flag such as +HOSTMAGS takes no value in SNANA, so it is
        # refused here before its missing value is
        if key in _REJECTED_KEYS:
            _check_rejected_keys({key})
        if not is_key(key, known):
            raise SNANAError(f"Unknown command-line override key '{key}'.",
                             function_name="group_override_tokens")
        values, i = _override_values(tokens, i + 1, key, known)
        if not values:
            raise SNANAError(f"Command-line override '{key}' has no value.",
                             function_name="group_override_tokens")
        groups.append((key, values))
    return groups


def apply_overrides(raw: UsedTrackingDict,
                    groups: Sequence[Tuple[str, List[str]]],
                    input_dir: Path) -> List[str]:
    """Replay each override through the input-file line parser."""
    lines = []
    for key, values in groups:
        line = f"{key}: {' '.join(str(v) for v in values)}"
        if '#' in line:
            raise SNANAError(
                f"Command-line override '{line}' contains '#', which the input "
                f"format reads as a comment.", function_name="apply_overrides")
        _read_line(line, raw, input_dir, source='arg')
        lines.append(line)
    return lines


def _apply_hostlib_mskopt_add(raw: UsedTrackingDict) -> None:
    """OR HOSTLIB_MSKOPT_ADD into HOSTLIB_MSKOPT (snlc_sim.c:3947-3961)."""
    if 'HOSTLIB_MSKOPT_ADD' not in raw:
        return
    default = field_default(Host, 'mskopt')
    current = int(raw.get('HOSTLIB_MSKOPT', default))
    if current == default:
        raise SNANAError(
            "HOSTLIB_MSKOPT_ADD is forbidden when HOSTLIB_MSKOPT is 0; set "
            "HOSTLIB_MSKOPT or drop the override.",
            function_name="_apply_hostlib_mskopt_add")
    raw['HOSTLIB_MSKOPT'] = current | int(raw['HOSTLIB_MSKOPT_ADD'])


_ARGV_INCLUDE_KEYS = ('INPUT_FILE_INCLUDE', 'INPUT_INCLUDE_FILE',
                      'EXCLUDE_INCLUDE_FILE')


def _argv_include_directives(
        tokens: Sequence[str]) -> Tuple[List[str], List[str], List[str]]:
    """EXCLUDE patterns, extra INCLUDE files, and the tokens left over.

    Read before the primary file, since jax expands includes as it meets them
    while SNANA collects them first (snlc_sim.c:634).
    """
    exclude: List[str] = []
    includes: List[str] = []
    rest: List[str] = []
    remaining = iter(tokens)
    for token in remaining:
        if token not in _ARGV_INCLUDE_KEYS:
            rest.append(token)
            continue
        filename = next(remaining, None)
        if filename is None:
            raise SNANAError(f"'{token}' needs a filename after it.",
                             function_name="_argv_include_directives")
        (exclude if token == 'EXCLUDE_INCLUDE_FILE' else includes).append(filename)
    return exclude, includes, rest


def parse_snana_input_file(input_file: Union[str, Path],
                           overrides: Sequence[str] = (), *, settings=None) -> SimulationConfig:
    """Read a SNANA sim-input file into a SimulationConfig.

    `overrides` are command-line tokens (`NGENTOT_LC 500 …`), applied after the
    file and its includes, before GENPERFECT: SNANA's order (snlc_sim.c:680).
    """
    if isinstance(overrides, Mapping):
        overrides = [str(part) for key, value in overrides.items()
                     for part in (key, *(value if isinstance(value, (list, tuple)) else [value]))]
    exclude, argv_includes, overrides = _argv_include_directives(overrides)
    raw = read_keys(input_file, exclude=exclude)
    for path in argv_includes:
        _read_lines(Path(os.path.expandvars(path)), raw,
                    Path(input_file).parent, exclude=exclude)
    override_lines: List[str] = []
    if overrides:
        override_lines = apply_overrides(
            raw,
            group_override_tokens(overrides, known_keys(raw, overrides)),
            Path(input_file).parent)
        _apply_hostlib_mskopt_add(raw)

    direct, groups = {}, {}
    if settings:
        raw, direct, groups = translated_settings(settings, raw)
    else:
        _resolve_ransystpar_wildcards(raw)
    typed_keys = set(raw)
    _check_rejected_keys(typed_keys)
    _check_misnamed_simsed_params(typed_keys)

    try:
        grouped = {}
        for name, value in direct.items():
            grouped.setdefault(groups[name], {})[name] = value
        cfg = build_validated(raw, settings=grouped)
        # Before the rate model, so the counts use the shifts (snlc_sim.c:122).
        prep_ransystpar(cfg)
    except _BAD_INPUT as exc:
        # a mistyped override key surfaces as a bad value of the key before it
        detail = exc.message if isinstance(exc, SNANAError) else str(exc)
        overridden = [line.split(':', 1)[0] for line in override_lines]
        if not any(re.search(rf"\b{re.escape(k)}\b", detail) for k in overridden):
            raise
        applied = "\n  ".join(override_lines)
        raise SNANAError(
            f"{detail}\n\nwhile applying these command-line overrides:\n  "
            f"{applied}\nIs one of the values a mistyped key?",
            function_name="parse_snana_input_file") from exc
    cfg.run._input_file = str(Path(input_file))
    cfg.run._input_lines = raw.lines()

    # a command-line-only key nothing consumed is fatal, as in SNANA's
    # check_argv (sntools.c:11358-11383)
    for key in _ARGV_ONLY_KEYS:
        if key in raw.keys() and key not in raw.used:
            raise SNANAError(
                f"Command-line key {key} is not used by this GENMODEL.",
                function_name="parse_snana_input_file")

    _warn_unrecognized_keys(raw, typed_keys)
    return cfg


def parse_genpoly(raw):
    """Parse a SNANA GENPOLY into terms, each a float or a (lo, hi) range
    (SNANA 'lo:hi', drawn per SN). Returns [] if unset.

    read_genpoly (sntools.c:11586) accepts 'a0,a1,a2' and the older
    space-separated 'a0 a1 a2'; the latter reaches here already split into a
    list, so commas and whitespace are both treated as separators.
    """
    if raw is None:
        return []
    tokens = raw if isinstance(raw, (list, tuple)) else [raw]
    terms = []
    for token in tokens:
        for part in str(token).replace(',', ' ').split():
            if ':' in part:
                lo, hi = part.split(':')
                terms.append((float(lo), float(hi)))
            else:
                terms.append(float(part))
    return terms
