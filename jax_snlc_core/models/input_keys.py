"""How a model declares its input keys.

A model lists its keys as annotated attributes marked :data:`FIT_ONLY`,
:data:`SIM_ONLY` or :data:`BBC_ONLY`; the fit, simulation and BBC parsers
read them through :func:`model_input_keys` and
:func:`parse_model_input_value`.
"""

from dataclasses import fields, is_dataclass
from types import UnionType
from typing import (Annotated, Literal, NamedTuple, Optional, Tuple, Union,
                    get_args, get_origin, get_type_hints)

import jax

from jax_snlc_core.input_settings import convert
from jax_snlc_core.logging import SNANAError


# markers for a key only one input accepts: an unmarked key is the SED's,
# read by the simulation and the fit alike; a BBC key must be marked
FIT_ONLY = 'FIT_ONLY'
SIM_ONLY = 'SIM_ONLY'
BBC_ONLY = 'BBC_ONLY'

INPUT_TYPES = (int, float, bool, str)


class ModelInputKey(NamedTuple):
    """What a model's annotated attribute declares about its input key."""
    attribute: str
    kind: type
    allowed: Optional[Tuple] = None
    fit_only: bool = False
    sim_only: bool = False
    bbc_only: bool = False


def _annotation_spec(attr, ann):
    """Read an attribute's annotation as the key it declares.

    `Annotated[..., FIT_ONLY]`, `Annotated[..., SIM_ONLY]` and
    `Annotated[..., BBC_ONLY]` mark a key only one input accepts,
    `Literal[...]` names the values allowed, and either spelling of an
    optional is unwrapped.
    """
    fit_only = sim_only = bbc_only = False
    if get_origin(ann) is Annotated:
        ann, *markers = get_args(ann)
        fit_only = FIT_ONLY in markers
        sim_only = SIM_ONLY in markers
        bbc_only = BBC_ONLY in markers
    if get_origin(ann) in (Union, UnionType):
        args = [a for a in get_args(ann) if a is not type(None)]
        if len(args) == 1:
            ann = args[0]
    allowed = None
    if get_origin(ann) is Literal:
        allowed = get_args(ann)
        ann = type(allowed[0])
    if ann not in INPUT_TYPES:
        raise SNANAError(
            f"Model attribute '{attr}' is annotated {ann}, but an input key "
            f"carries one value, so it must be annotated "
            f"{', '.join(t.__name__ for t in INPUT_TYPES)}. An attribute with "
            f"no annotation is a model constant, settable by no input.",
            function_name="_annotation_spec")
    return ModelInputKey(attr, ann, allowed, fit_only, sim_only, bbc_only)


def _offered(spec, consumer):
    """Whether ``consumer``'s input accepts a declared key."""
    if consumer is None:
        return True
    if consumer == 'bbc':
        return spec.bbc_only
    return not spec.bbc_only and not (spec.fit_only if consumer == 'sim'
                                      else spec.sim_only)


def model_input_keys(model_cls, consumer='fit'):
    """The input-file keys a model class declares, as
    {KEY: :class:`ModelInputKey`}.

    A lowercase scalar annotation declares an input-file setting under its
    upper-cased name. A ``jax.Array`` annotation declares prepared numerical
    data, not an input-file setting. Dataclass dictionary and tuple
    fields likewise describe prepared state rather than scalar input keys.
    Private attributes and dataclass fields excluded from construction or
    marked ``metadata={"input": False}`` are not input settings either.
    Unannotated attributes are model constants.
    ``consumer`` picks the input asking: 'sim' the simulation's, which the
    keys marked FIT_ONLY do not configure, 'fit' the fitter's, which the keys
    marked SIM_ONLY do not, 'bbc' BBC's, which is the keys marked BBC_ONLY
    alone, and None every key the model declares.
    """
    model_fields = {f.name: f for klass in reversed(model_cls.__mro__)
                    if is_dataclass(klass) for f in fields(klass)}
    non_input_fields = {name for name, f in model_fields.items()
                        if not f.init or f.metadata.get("input") is False}
    keys = {}
    for attr, ann in get_type_hints(model_cls, include_extras=True).items():
        if (ann is jax.Array or attr.startswith("_") or attr in non_input_fields
                or (is_dataclass(model_cls) and ann in (dict, tuple))):
            continue
        if attr.islower():
            spec = _annotation_spec(attr, ann)
            if _offered(spec, consumer):
                keys[attr.upper()] = spec
    return keys


def parse_model_input_value(key, spec, value):
    """One input-file value, typed and checked as the model declares it.

    Both parsers read a model key here, so it means the same thing wherever
    it is set: SNANA writes a logical as a Fortran T/F and lets an integer
    key be written as a float.
    """
    if spec.kind is bool:
        typed = convert(str(value).strip(), bool, key)
    elif spec.kind is int:
        typed = int(float(value))
    elif spec.kind is float:
        typed = float(value)
    else:
        typed = str(value)
    if spec.allowed is not None and typed not in spec.allowed:
        raise SNANAError(
            f"{key} = {value} is not one of "
            f"{', '.join(str(a) for a in spec.allowed)}.",
            function_name="parse_model_input_value")
    return typed
