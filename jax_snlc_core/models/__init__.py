"""The SED models, and the API the programs use to find and build them.

:mod:`.sed_model` holds :class:`SNModel`, what every model is, and the
tabulated-SED tools the models share. :mod:`.registry` finds and builds a
model, :mod:`.roles` holds what each program asks of a model, and
:mod:`.input_keys` how a model declares its input keys. ``__all__`` lists
what the programs import from the package.
"""

from .bayesn import BayeSNModel
from .input_keys import (BBC_ONLY, FIT_ONLY, INPUT_TYPES, SIM_ONLY, ModelInputKey,
                         model_input_keys, parse_model_input_value)
from .non1a import NON1ASEDModel
from .redback import RedbackModel
from .registry import (MODEL_REGISTRY, build_model, infer_model_type_from_path, metadata_files,
                       register_model, resolve_model_type, standardised_model)
from .roles import FittedModel, SimulatedModel, StandardisedModel
from .salt import SALTModel
from .sed_model import SNModel
from .simsed import SIMSEDModel

__all__ = [
    "SALTModel", "BayeSNModel", "NON1ASEDModel", "SIMSEDModel", "RedbackModel",
    "SNModel", "SimulatedModel", "FittedModel", "StandardisedModel",
    "MODEL_REGISTRY", "register_model", "infer_model_type_from_path", "metadata_files",
    "standardised_model",
    "resolve_model_type", "build_model", "ModelInputKey", "model_input_keys",
    "parse_model_input_value", "FIT_ONLY", "SIM_ONLY", "BBC_ONLY",
    "INPUT_TYPES",
]
