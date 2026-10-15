"""Supernova model implementations in JAX.

Every production model inherits from :class:`base.SNModel` and is registered in
:data:`MODEL_REGISTRY` by its SNANA model-type key. :func:`build_model` is the
single construction path used by the simulator (photometry and spectra) and the
fitter: it resolves the model type and builds the registered class from the
input's model keys, which the model reads for itself, so a model is
fully built when its loader returns. Implement a new model in its own
module, declare ``GENMODEL_ALIASES``, and call :func:`register_model`.
"""

import logging
import os
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, get_type_hints
from weakref import WeakSet

import jax

from jax_snlc_core.utils.error_handling import SNANAError

from .base import (BBC_ONLY, FIT_ONLY, INPUT_TYPES, SIM_ONLY, ModelInputKey,
                   SNModel, model_input_keys, parse_model_input_value)
from .fitted import FittedModel
from .simulated import SimulatedModel
from .standardised import StandardisedModel
from .salt_model import SALTModel
from .bayesn_model import BayeSNModel
from .non1ased_model import NON1ASEDModel
from .simsed_model import SIMSEDModel
from .redback_model import RedbackModel

__all__ = [
    "SALTModel", "BayeSNModel", "NON1ASEDModel", "SIMSEDModel", "RedbackModel",
    "SNModel", "SimulatedModel", "FittedModel", "StandardisedModel",
    "MODEL_REGISTRY", "register_model", "infer_model_type_from_path", "metadata_files",
    "standardised_model",
    "resolve_model_type", "build_model", "ModelInputKey", "model_input_keys",
    "parse_model_input_value", "FIT_ONLY", "SIM_ONLY", "BBC_ONLY",
    "INPUT_TYPES",
]

logger = logging.getLogger(__name__)

_BUILTIN_MODELS = {
    'bayesn': BayeSNModel,
    'salt': SALTModel,
    'non1ased': NON1ASEDModel,
    'simsed': SIMSEDModel,
    'redback': RedbackModel,
}

MODEL_REGISTRY = {}
_JAX_MODEL_TYPES = WeakSet()


def register_model(name: str, model_cls, *, replace: bool = False) -> None:
    """Register a model for subsequent simulations and fits in this process.

    ``name`` is its lowercase model-type key. The class declares its path
    prefixes in ``GENMODEL_ALIASES``. Registration can follow imports of any
    consumer; existing model instances are unchanged. ``replace=True`` permits
    redefining a registered model, for example while developing in a notebook.

    Explicit dataclasses also register their constructor fields with JAX.
    Fields with ``metadata={"static": True}`` select the compiled program;
    other fields are numerical inputs. ``init=False`` fields remain host-side
    preparation state. Ordinary model classes need no changes.
    """
    if not isinstance(name, str) or not name.isidentifier() or name != name.lower():
        raise SNANAError("model name must be a lowercase identifier",
                         function_name="register_model")
    if not isinstance(model_cls, type) or not issubclass(model_cls, SNModel):
        raise SNANAError("registered model must inherit SNModel",
                         function_name="register_model")
    if name in MODEL_REGISTRY and MODEL_REGISTRY[name] is not model_cls and not replace:
        raise SNANAError(f"model {name!r} is already registered; use replace=True to redefine it",
                         function_name="register_model")
    aliases = model_cls.GENMODEL_ALIASES
    if not isinstance(aliases, tuple) or not aliases or any(
            not isinstance(alias, str) or not alias for alias in aliases):
        raise SNANAError("GENMODEL_ALIASES must be a non-empty tuple of path prefixes",
                         function_name="register_model")
    for key, other in MODEL_REGISTRY.items():
        if key == name or other is model_cls:
            continue
        for alias in aliases:
            for existing in other.GENMODEL_ALIASES:
                a, b = alias.lower(), existing.lower()
                if a.startswith(b) or b.startswith(a):
                    raise SNANAError(
                        f"model alias {alias!r} overlaps {existing!r} registered by {key!r}",
                        function_name="register_model")
    # Only an explicitly declared dataclass opts in. Ordinary subclasses of
    # the model roles retain their existing construction and JIT behaviour.
    if ("__dataclass_fields__" in vars(model_cls)
            and model_cls not in _JAX_MODEL_TYPES):
        constructor_fields = [f for f in fields(model_cls) if f.init]
        jax.tree_util.register_dataclass(
            model_cls,
            data_fields=[f.name for f in constructor_fields
                         if not f.metadata.get("static", False)],
            meta_fields=[f.name for f in constructor_fields
                         if f.metadata.get("static", False)],
        )
        _JAX_MODEL_TYPES.add(model_cls)
    MODEL_REGISTRY[name] = model_cls


def standardised_model(model_lcfit: str):
    """The class BBC's MODEL_LCFIT names, which must play the
    :class:`StandardisedModel` role, or a fatal error naming those that do."""
    model_type = infer_model_type_from_path(model_lcfit)
    cls = MODEL_REGISTRY.get(model_type)
    if cls is None or not issubclass(cls, StandardisedModel):
        playing = sorted(key for key, c in MODEL_REGISTRY.items()
                         if issubclass(c, StandardisedModel))
        raise SNANAError(
            f"MODEL_LCFIT '{model_lcfit}' names no model BBC can standardise; "
            f"expecting one of {playing}", function_name="standardised_model")
    return cls


def metadata_files(model_cls, model_path: str) -> Tuple[str, ...]:
    """The model directory's metadata files, by the patterns the class
    declares in ``METADATA_FILES``."""
    root = Path(os.path.expandvars(model_path)).expanduser()
    return tuple(str(p) for pattern in getattr(model_cls, 'METADATA_FILES', ())
                 for p in sorted(root.glob(pattern)))


def infer_model_type_from_path(model_path: str):
    """Infer the MODEL_REGISTRY key from a GENMODEL path, or None if no alias matches.

    Mirrors SNANA's GENMODEL dispatch (snlc_sim.c GENMODEL_NAME + strncmp):
    the lowercased GENMODEL basename must start with one of the model's
    ``GENMODEL_ALIASES``.
    """
    expanded = Path(os.path.expandvars(model_path)).expanduser()
    lowered = expanded.name.lower()
    for key, cls in MODEL_REGISTRY.items():
        if lowered.startswith(tuple(alias.lower() for alias in cls.GENMODEL_ALIASES)):
            return key
    return None


def resolve_model_type(model_path: str, hint: Optional[str] = None) -> str:
    """The MODEL_REGISTRY key for ``model_path``, or a fatal error.

    The path decides (:func:`infer_model_type_from_path`); ``hint`` is a
    caller-supplied fallback used, with a warning, only when the path does
    not match any registered alias.
    """
    if model_path:
        model_type = infer_model_type_from_path(model_path)
        if model_type is not None:
            return model_type
    if hint:
        logger.warning(
            f"Falling back to model_type hint '{hint}' (GENMODEL='{model_path}')")
        return hint
    raise SNANAError(
        "Cannot determine model type: GENMODEL path is missing or unsupported.",
        function_name="resolve_model_type",
    )


def build_model(model_path: str, config: Dict[str, Any], *,
                n_wave_points: int = 500,
                wavelength_range: Optional[Tuple[float, float]] = None,
                model_type: Optional[str] = None):
    """Construct one ready-to-use model: resolve the type, expand the path,
    and build the registered class, which reads ``config`` for itself.

    Shared by simulator photometry, simulator spectra and the fitter, so the
    same input always yields the same evaluable model. Sampling setup
    (``configure_generation``) and fitter optimisation setup stay with their
    callers, after this returns.

    The model carries what it chose: ``wavelength_grid`` is the working grid,
    the support cut to ``wavelength_range``, the rest-frame interval the
    caller will evaluate (None for the whole support), at ``n_wave_points``;
    ``get_wavelength_range()`` is the support itself, and
    ``filter_center_range`` the window a filter centre may sit in, ``None``
    where that is the whole support.
    """
    if model_type is None:
        model_type = resolve_model_type(model_path)
    if model_type not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise SNANAError(
            f"Unknown model_type: '{model_type}'. Available models: {available}",
            function_name="build_model")
    expanded = str(Path(os.path.expandvars(model_path)).expanduser())
    return MODEL_REGISTRY[model_type].load(
        expanded, config, n_wave_points, wavelength_range)


def model_input_declarations(purpose):
    """Discover option types across registered models without loading assets."""
    types, owners = {}, {}
    for model in MODEL_REGISTRY.values():
        hints = get_type_hints(model, include_extras=True)
        for spec in model_input_keys(model, purpose).values():
            name, annotation = spec.attribute, hints[spec.attribute]
            if name in types and types[name] != annotation:
                raise SNANAError(f'Conflicting model input declarations for {name}')
            types[name], owners[name] = annotation, model
    return types, owners


for _name, _model in _BUILTIN_MODELS.items():
    register_model(_name, _model)
