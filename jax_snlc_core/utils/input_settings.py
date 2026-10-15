"""Typed settings shared by Python, YAML and generated command-line options.

Owners declare their inputs with annotations. Special input forms belong to
ordinary reader functions on that owner; the machinery here does not know
about models or scientific parameter names.
"""

import argparse
import copy
import dataclasses
import inspect
import re
from collections.abc import Mapping
from types import UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin, get_type_hints

import yaml

from .error_handling import SNANAError


def supplied_values(values, changes):
    """Undo linked-rule changes that the caller has not subsequently edited.

    Restore mappings entry by entry so editing one entry does not retain
    unrelated derived entries as if they had been supplied explicitly.
    """
    result = copy.deepcopy(values)
    for name, (original, resolved) in changes.items():
        if name not in result:
            continue
        current = result[name]
        if isinstance(current, Mapping) and isinstance(original, Mapping) and isinstance(resolved, Mapping):
            for key in resolved:
                if key in current and current[key] == resolved[key]:
                    if key in original:
                        current[key] = copy.deepcopy(original[key])
                    else:
                        del current[key]
        elif current == resolved:
            result[name] = copy.deepcopy(original)
    return result


def derived_changes(original, resolved):
    """Record values changed by preparation, separately from supplied inputs."""
    return {name: (copy.deepcopy(value), copy.deepcopy(resolved[name]))
            for name, value in original.items() if value != resolved[name]}


def configuration_settings(config):
    """Recover public inputs from a config, retaining edits to resolved values."""
    values = {name: value for name, value in dataclasses.asdict(config).items()
              if not name.startswith('_')}
    return supplied_values(values, config._derived_settings)


def value_type(annotation):
    """Unwrap optional and annotated types for command-line construction."""
    if get_origin(annotation) is Annotated:
        return value_type(get_args(annotation)[0])
    if get_origin(annotation) in (Union, UnionType):
        types = [t for t in get_args(annotation) if t is not type(None)]
        if len(types) == 1:
            return value_type(types[0])
    return annotation


def convert(value, annotation, name):
    """Convert one supplied setting, reporting invalid values by input name."""
    origin, args = get_origin(annotation), get_args(annotation)
    if dataclasses.is_dataclass(annotation):
        if isinstance(value, annotation):
            return copy.deepcopy(value)
        return build_dataclass(annotation, value)
    if annotation is Any:
        return copy.deepcopy(value)
    if origin is Annotated:
        return convert(value, args[0], name)
    if origin in (Union, UnionType):
        if value is None and type(None) in args:
            return None
        alternatives = [kind for kind in args if kind is not type(None)]
        if len(alternatives) == 1:
            return convert(value, alternatives[0], name)
        for kind in args:
            if kind is type(None):
                continue
            try:
                return convert(value, kind, name)
            except SNANAError:
                pass
        raise SNANAError(f"{name}: invalid value {value!r}")
    if value is None:
        raise SNANAError(f"{name} cannot be null")
    if origin is Literal:
        if value not in args:
            raise SNANAError(f"{name} must be one of {args}; got {value!r}")
        return value
    if origin is dict or annotation is dict:
        if not isinstance(value, Mapping):
            raise SNANAError(f"{name} must be a mapping")
        kt, vt = args or (Any, Any)
        return {convert(k, kt, name): convert(v, vt, f"{name}.{k}")
                for k, v in value.items()}
    if origin in (list, tuple) or annotation in (list, tuple):
        if not isinstance(value, (list, tuple)):
            raise SNANAError(f"{name} must be a list")
        kind = origin or annotation
        if kind is tuple and args and args[-1] is not Ellipsis:
            if len(value) != len(args):
                raise SNANAError(f"{name} needs {len(args)} values")
            return tuple(convert(v, t, name) for v, t in zip(value, args))
        converted = [convert(v, args[0] if args else Any, name) for v in value]
        return kind(converted)
    if annotation is bool:
        if isinstance(value, bool):
            return value
        if str(value).lower() in ('true', 't', '.true.', '1'):
            return True
        if str(value).lower() in ('false', 'f', '.false.', '0'):
            return False
        raise SNANAError(f"{name} must be true or false")
    if annotation is str:
        if not isinstance(value, str):
            raise SNANAError(f"{name} must be a string")
        return value
    try:
        converted = annotation(value)
        if annotation is int and float(value) != converted:
            raise ValueError('not an integer')
        return converted
    except (TypeError, ValueError, OverflowError) as error:
        raise SNANAError(f"{name}: invalid value {value!r}") from error


def merge_settings(*sources):
    """Merge settings without mutating sources; mappings merge by entry.

    An empty mapping resets that setting. Entries themselves replace whole
    values, so a changed prior never inherits coefficients of another form.
    """
    result = {}
    for source in sources:
        for name, value in source.items():
            if isinstance(value, Mapping) and value:
                previous = result.get(name, {})
                value = {**previous, **value} if isinstance(previous, Mapping) else value
            result[name] = copy.deepcopy(value)
    return result


def field_help(owner):
    """Read the owner's existing #: field documentation for argparse help."""
    result, comments = {}, []
    try:
        source = inspect.getsource(owner)
    except (OSError, TypeError):
        return result
    for line in source.splitlines():
        text = line.strip()
        if text.startswith('#:'):
            comments.append(text[2:].strip())
            continue
        match = re.match(r'(\w+)\s*:', text)
        if match:
            if '#:' in text:
                comments.append(text.split('#:', 1)[1].strip())
            if comments:
                result[match[1]] = ' '.join(comments)
        comments = []
    return result


def mapping_tokens(tokens):
    """Read either a YAML mapping or one named mapping entry from argv."""
    if len(tokens) == 1:
        value = yaml.safe_load(tokens[0])
        if isinstance(value, Mapping):
            return value
        raise SNANAError('Expected a mapping or a name followed by its value')
    values = [yaml.safe_load(token) for token in tokens[1:]]
    return {tokens[0]: values[0] if len(values) == 1 else values}


def add_settings(parser, declarations, owners):
    """Add options from {name: annotation} and {name: owning class}.

    Dictionaries are collected as repeatable entries and converted after
    parsing. This keeps argparse actions generic and custom readers local.
    """
    docs = {owner: field_help(owner) for owner in set(owners.values())}
    for name, ann in declarations.items():
        owner = owners[name]
        kind = value_type(ann)
        origin, args = get_origin(kind), get_args(kind)
        options = dict(dest=name, default=argparse.SUPPRESS,
                       help=docs[owner].get(name, '').replace('%', '%%'))
        flag = getattr(owner, 'input_flags', {}).get(name, name.replace('_', '-'))
        if kind is Any or dataclasses.is_dataclass(kind):
            options["type"] = yaml.safe_load
        elif kind is bool:
            options['action'] = argparse.BooleanOptionalAction
        elif origin is dict or kind is dict:
            options.update(action='append', nargs='+')
        elif origin in (tuple, list) or kind in (tuple, list):
            options['nargs'] = (len(args) if origin is tuple and args
                                and args[-1] is not Ellipsis else '+')
        elif origin is Literal:
            options.update(type=type(args[0]), choices=args)
        elif origin in (Union, UnionType):
            if any(get_origin(t) is list for t in args):
                options['nargs'] = '+'
            else:
                options['type'] = str
        else:
            options['type'] = kind
        parser.add_argument('--' + flag, **options)


def parsed_settings(namespace, declarations, owners):
    """Convert argparse's supplied values through the same typed readers."""
    result = {}
    for name, raw in vars(namespace).items():
        if name not in declarations:
            continue
        annotation = declarations[name]
        kind = value_type(annotation)
        if get_origin(kind) is dict or kind is dict:
            reader = getattr(owners[name], 'input_readers', {}).get(name, mapping_tokens)
            value = {}
            for tokens in raw:
                try:
                    entry = reader(tokens)
                except yaml.YAMLError as error:
                    raise SNANAError(f'{name}: invalid YAML value: {error}') from error
                value = merge_settings({'entry': value}, {'entry': entry})['entry']
        else:
            value = raw
            origin, args = get_origin(kind), get_args(kind)
            if origin in (Union, UnionType) and str in args and isinstance(raw, list) and len(raw) == 1:
                value = raw[0]
            if origin in (list, tuple) and args and (
                    args[0] is Any or dataclasses.is_dataclass(args[0])
                    or get_origin(args[0]) in (dict, tuple, list)):
                try:
                    value = [yaml.safe_load(item) if isinstance(item, str) else item for item in raw]
                except yaml.YAMLError as error:
                    raise SNANAError(f'{name}: invalid YAML value: {error}') from error
        result[name] = convert(value, annotation, name)
    return result


def dataclass_settings(owner):
    """Public constructor fields are the complete set of ordinary inputs."""
    hints = get_type_hints(owner)
    types = {f.name: hints[f.name] for f in dataclasses.fields(owner)
             if f.init and not f.name.startswith('_')}
    return types, dict.fromkeys(types, owner)


def build_dataclass(owner, settings):
    """Construct a detached typed config, rejecting unknown supplied fields."""
    if not isinstance(settings, Mapping):
        raise SNANAError(f'{owner.__name__} settings must be a mapping')
    types, _ = dataclass_settings(owner)
    unknown = set(settings) - set(types)
    if unknown:
        raise SNANAError(f'Unknown {owner.__name__} setting(s): {sorted(unknown)}')
    values = {name: convert(value, types[name], name)
              for name, value in settings.items()}
    try:
        return owner(**values)
    except TypeError as error:
        raise SNANAError(f'{owner.__name__}: {error}') from error


def settings_parser(owner, description=None, *, add_help=True):
    """Generate every ordinary option from its owning dataclass."""
    parser = argparse.ArgumentParser(description=description, allow_abbrev=False,
                                     add_help=add_help)
    types, owners = dataclass_settings(owner)
    add_settings(parser, types, owners)
    return parser


def read_settings(path, *, text=None):
    """Read a YAML mapping without guessing defaults for bad inputs."""
    from pathlib import Path
    try:
        value = yaml.safe_load(Path(path).read_text() if text is None else text)
    except (OSError, yaml.YAMLError) as error:
        raise SNANAError(f'Cannot read {path}: {error}') from error
    if not isinstance(value, Mapping):
        raise SNANAError(f'{path} must contain a YAML mapping')
    return dict(value)


def bootstrap_settings(tokens, types, owners, model_types, model_owners):
    """Consume option values before locating a positional input filename."""
    for name, annotation in model_types.items():
        if name in types and value_type(types[name]) != value_type(annotation):
            raise SNANAError(f'Conflicting input declarations for {name}')
    combined = {**model_types, **types}
    owners = {**model_owners, **owners}
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    add_settings(parser, combined, owners)
    supplied, remaining = parser.parse_known_args(tokens)
    return parsed_settings(supplied, combined, owners), remaining
