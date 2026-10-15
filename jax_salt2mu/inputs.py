"""Typed BBC settings for Python, YAML and generated command-line options."""

import argparse
from dataclasses import asdict
import copy
from typing import get_type_hints

from jax_snlc_core.utils.input_settings import bootstrap_settings as parse_bootstrap
from jax_snlc_core.models import model_input_keys, model_input_declarations, standardised_model
from jax_snlc_core.utils import SNANAError
from jax_snlc_core.utils.input_settings import (
    add_settings, build_dataclass, convert, dataclass_settings, merge_settings,
    parsed_settings, configuration_settings, derived_changes)
from .config import BBCConfig, BBC_NUISANCE_SPEC, validate_coordinates, resolve_input_settings


def declarations(model_name):
    types, owners = dataclass_settings(BBCConfig)
    model = standardised_model(model_name)
    for spec in model_input_keys(model, 'bbc').values():
        name = spec.attribute
        if name not in types:
            types[name] = get_type_hints(model, include_extras=True)[name]
            owners[name] = model
    return types, owners


def merge_named_settings(values, settings):
    """Merge supplied fields, replacing SNANA modes only when explicitly changed."""
    if any(name.startswith('_') for name in settings):
        raise SNANAError('Private BBC fields are not input settings')
    result = merge_settings(values, settings)
    model = standardised_model(result.get('model_lcfit', BBCConfig.model_lcfit))
    specs = {**BBC_NUISANCE_SPEC, **model.nuisance_spec}
    flags = result.get('_u_flags', {})
    for field in ('nuisance_float', 'nuisance_onebin'):
        if field not in settings:
            continue
        names = settings[field] or values.get(field, {})
        for name in names:
            if name in specs:
                flags.pop(f'u{specs[name][3]}', None)
    return result


def build_config(settings, config=None, *, _snana_flags=None):
    """Build named BBC settings without loading FITRES or bias-correction data."""
    base = configuration_settings(BBCConfig() if config is None else config)
    values = merge_settings(base, settings)
    model = standardised_model(values.get('model_lcfit', BBCConfig.model_lcfit))
    types, owners = declarations(values.get('model_lcfit', BBCConfig.model_lcfit))
    unknown = set(values) - set(types)
    if unknown:
        raise SNANAError(f'Unknown BBC setting(s): {sorted(unknown)}')
    core, model_values = {}, {}
    for name, value in values.items():
        converted = convert(value, types[name], name)
        (core if owners[name] is BBCConfig else model_values)[name] = converted
    cfg = build_dataclass(BBCConfig, core)
    if config is not None:
        cfg._model_inputs = copy.deepcopy(config._model_inputs)
        cfg._u_flags = copy.deepcopy(config._u_flags)
        cfg._unblind = copy.deepcopy(config._unblind)
    cfg._model_inputs.update(model_values)
    original = {k: v for k, v in asdict(cfg).items() if not k.startswith('_')}
    allowed = set(model.nuisance_spec) | set(BBC_NUISANCE_SPEC)
    for mapping in (cfg.nuisance_val, cfg.nuisance_float, cfg.nuisance_onebin):
        unknown = set(mapping) - allowed
        if unknown:
            raise SNANAError(f'{cfg.model_lcfit} has no nuisance parameters {sorted(unknown)}')
    specs = {**BBC_NUISANCE_SPEC, **model.nuisance_spec}
    # Unrelated overrides must preserve SNANA modes such as u13=2.
    changed = set(settings.get('nuisance_float', {})) | set(settings.get('nuisance_onebin', {}))
    if config is not None:
        changed.update(name for name, value in config.nuisance_onebin.items()
                       if base['nuisance_onebin'].get(name) != value)
    for field in ('nuisance_float', 'nuisance_onebin'):
        if field in settings and not settings[field] and config is not None:
            changed.update(getattr(config, field))
    for name in changed:
        floated = cfg.nuisance_float.get(name, False)
        onebin = cfg.nuisance_onebin.get(name, not floated)
        onebin = onebin if floated else True
        cfg.nuisance_onebin[name] = onebin
        slot = specs[name][3]
        cfg._u_flags[f'u{slot}'] = (3 if onebin else 1) if floated else 0
    if _snana_flags is not None:
        cfg._u_flags.update(_snana_flags)
    declared = {s.attribute for s in model_input_keys(model, 'bbc').values()}
    if set(cfg._model_inputs) - declared:
        raise SNANAError(f'Unknown {cfg.model_lcfit} model settings: {sorted(set(cfg._model_inputs) - declared)}')
    resolve_input_settings(cfg)
    validate_coordinates(model, cfg)
    resolved = {k: v for k, v in asdict(cfg).items() if not k.startswith("_")}
    cfg._derived_settings = derived_changes(original, resolved)
    return cfg


def command_settings(tokens, settings):
    parser = argparse.ArgumentParser(allow_abbrev=False, add_help=False)
    types, owners = dataclass_settings(BBCConfig)
    add_settings(parser, types, owners)
    selectors, _ = parser.parse_known_args(tokens)
    model_name = vars(selectors).get('model_lcfit', settings.get('model_lcfit', BBCConfig.model_lcfit))
    types, owners = declarations(model_name)
    parser = argparse.ArgumentParser(description='Fit bias-corrected supernova distances.', allow_abbrev=False)
    add_settings(parser, types, owners)
    supplied, snana = parser.parse_known_args(tokens)
    return parsed_settings(supplied, types, owners), snana


def bootstrap_settings(tokens):
    """Read shared and model options before locating the input file."""
    types, owners = dataclass_settings(BBCConfig)
    return parse_bootstrap(tokens, types, owners, *model_input_declarations('bbc'))
