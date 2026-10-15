"""Simulation inputs using the simulator's own configuration fields."""

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import get_type_hints

import yaml

from jax_snlc_core.utils.input_settings import bootstrap_settings as parse_bootstrap
from jax_snlc_core.models import (MODEL_REGISTRY, infer_model_type_from_path,
                                  model_input_keys, model_input_declarations)
from jax_snlc_core.utils import SNANAError
from jax_snlc_core.utils.input_settings import (
    add_settings, convert, dataclass_settings, merge_settings, parsed_settings, value_type)
from .simulation import SimulationConfig


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
    from .config_parser import FILTER_ROW_KEYS, _key_aliases, _resolve_ransystpar_wildcards
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
        from .population import configured_distributions
        model = MODEL_REGISTRY[model_type]
        raw['POPULATIONS'] = configured_distributions(
            settings['populations'], model, model.structured_input(raw))
    return raw, direct, groups


def build_config(settings):
    """Construct and validate a simulation without loading survey resources."""
    from .config_parser import build_validated
    from .ransystpar import prep_ransystpar
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
        import re
        if any(re.match(r'^\s*[a-z_][a-z0-9_]*\s*:', line) for line in text.splitlines()):
            raise SNANAError(f'Cannot read {path}: {error}') from error
        value = None
    if isinstance(value, dict) and value and all(isinstance(k, str) and k == k.lower() for k in value):
        return dict(value)
    return None


def parse_input(path, overrides=None):
    """Read a file or build Python settings through the same validation."""
    from .config_parser import parse_snana_input_file
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
    selection = merge_settings(settings, parsed_settings(initial, types, owners))
    if selection.get('ransystpar'):
        raw, _, _ = translated_settings(selection)
        model_type = infer_model_type_from_path(raw['GENMODEL'])
    else:
        model_type = infer_model_type_from_path(selection.get('genmodel', ''))
    types, owners, _ = declarations(model_type)
    parser = argparse.ArgumentParser(description='Simulate supernova light curves.', allow_abbrev=False)
    add_settings(parser, types, owners)
    supplied, snana = parser.parse_known_args(tokens)
    return parsed_settings(supplied, types, owners), snana


def bootstrap_settings(tokens):
    """Read shared and model options before locating the input file."""
    types, owners, _ = declarations()
    return parse_bootstrap(tokens, types, owners, *model_input_declarations('sim'))
