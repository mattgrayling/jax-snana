"""The model-key ownership gate.

A model's input keys are its lowercase annotated class attributes
(`model_input_keys`); the general fit package must not know any of them by
name. This is what keeps adding a model down to one class file: the moment a
model key is parsed, stored or copied in general code, these tests fail.
"""

import re
from pathlib import Path

from jax_snlc_core.models import (MODEL_REGISTRY, FittedModel, SNModel,
                                  SimulatedModel, StandardisedModel,
                                  model_input_keys)
from jax_snlc_fit.config_parser import RECORD_CLASSES


class _EveryRole(SNModel, SimulatedModel, FittedModel):
    """The generic keys are what the base and the roles declare."""

REPO = Path(__file__).resolve().parents[2]
FIT_SOURCES = sorted(Path(REPO, 'jax_snlc_fit').glob('*.py')) + [
    REPO / 'snlc_fit.py']
SIM_SOURCES = sorted(Path(REPO, 'jax_snlc_sim').rglob('*.py')) + [
    REPO / 'snlc_sim.py']
BBC_SOURCES = sorted(Path(REPO, 'jax_salt2mu').glob('*.py')) + [
    REPO / 'salt2mu.py']

# SNANA's generic keys, declared on the base class and the roles and so
# shared by every model rather than owned by one, as the fitter's input
# sees them.
GENERIC_KEYS = set(model_input_keys(_EveryRole))


def _model_keys():
    keys = {}
    for cls in dict.fromkeys(MODEL_REGISTRY.values()):
        for key in model_input_keys(cls):
            if key not in GENERIC_KEYS:
                keys.setdefault(key, cls.__name__)
    return keys


def test_no_model_key_is_named_in_the_general_fit_code():
    offenders = []
    for path in FIT_SOURCES:
        source = path.read_text()
        for key, owner in _model_keys().items():
            if re.search(rf"['\"]{key}['\"]", source):
                offenders.append(f"{path.name}: {key} ({owner})")
    assert not offenders, offenders


def test_no_model_module_is_imported_by_the_fit_package():
    model_modules = {cls.__module__.rsplit('.', 1)[-1]
                     for cls in MODEL_REGISTRY.values()}
    offenders = []
    for path in FIT_SOURCES:
        source = path.read_text()
        for mod in model_modules:
            if f'models.{mod}' in source:
                offenders.append(f"{path.name}: imports {mod}")
    assert not offenders, offenders


def test_no_config_field_shadows_a_model_attribute():
    attrs = {spec.attribute for cls in dict.fromkeys(MODEL_REGISTRY.values())
             for spec in model_input_keys(cls).values()}
    # FitConfig carries the six records, so the names a model key could
    # shadow are the records' own fields
    fields = {name for cls in RECORD_CLASSES.values()
              for name in cls.__dataclass_fields__}
    assert {'fitmodel_name', 'opt_mwebv'} <= fields, fields
    assert not attrs & fields, attrs & fields


def _sim_model_keys():
    keys = {}
    generic = set(model_input_keys(_EveryRole, consumer='sim'))
    for cls in dict.fromkeys(MODEL_REGISTRY.values()):
        if not issubclass(cls, SimulatedModel):
            continue
        for key in model_input_keys(cls, consumer='sim'):
            if key not in generic:
                keys.setdefault(key, cls.__name__)
        for name in cls.population_parameters:
            for spelling in (name, *cls.population_aliases.get(name, ())):
                keys.setdefault(spelling, cls.__name__)
        for key in cls.FILE_ONLY_INPUT_KEYS:
            keys.setdefault(key, cls.__name__)
        for group in cls.INPUT_KEY_ALIASES:
            for key in group:
                keys.setdefault(key, cls.__name__)
    return keys


# Model-key spellings general sim code may carry as part of an SNANA file
# format rather than as model dispatch, each under a user ruling.
_FORMAT_LITERALS = {
    # the SIMLIB header format carries SALT2 parameters (USE_SIMLIB_SALT2)
    'sampler.py': {'SALT2x1', 'SALT2c'},
    # the central command-line-only key table; the SALT2BETA_cPOLY key SNANA
    # parses for every GENMODEL (snlc_sim.c:2368); the GENPOP_ASYMGAUSS file
    # format, whose blocks carry SALT2 populations (snlc_sim.c:6437)
    'config_parser.py': {'NON1A_INDEX_FORCE', 'NONIA_INDEX_FORCE',
                         'SALT2BETA_cPOLY', 'SALT2x1', 'SALT2c'},
    # the GENPDF_FILE format's own VARNAMES spelling, in the examples of
    # what the reader accepts; the reader itself reads whatever the file names
    'genpdf_reader.py': {'SALT2c'},
    # the SEARCHEFF map-file format's own variable spellings
    'searcheff_map.py': {'SALT2x1', 'SALT2c'},
    # the SIMLIB header format carries SALT2 parameters (snlc_sim.c:22099)
    'simlib_reader.py': {'SALT2x1', 'SALT2c'},
}


def test_no_model_key_is_named_in_the_general_sim_code():
    offenders = []
    for path in SIM_SOURCES:
        source = path.read_text()
        allowed = _FORMAT_LITERALS.get(path.name, set())
        for key, owner in _sim_model_keys().items():
            if key in allowed:
                continue
            if re.search(rf"['\"]{key}['\"]", source):
                offenders.append(f"{path.name}: {key} ({owner})")
    assert not offenders, offenders


def test_no_model_type_literal_in_the_general_sim_code():
    """`if model_type == 'x'` in general code is the other face of a named
    key: it makes adding a model an edit outside the model class. General
    code branches on a trait the model declares, never on its name."""
    offenders = []
    for path in SIM_SOURCES:
        source = path.read_text()
        for key in MODEL_REGISTRY:
            hits = len(re.findall(rf"['\"]{key}['\"]", source))
            if hits:
                offenders.append(f"{path.relative_to(REPO)}: '{key}' x{hits}")
    assert not offenders, offenders


def test_no_model_module_is_imported_by_the_sim_package():
    model_modules = {cls.__module__.rsplit('.', 1)[-1]
                     for cls in MODEL_REGISTRY.values()}
    offenders = []
    for path in SIM_SOURCES:
        source = path.read_text()
        for mod in model_modules:
            if f'models.{mod}' in source:
                offenders.append(f"{path.name}: imports {mod}")
    assert not offenders, offenders


def _bbc_model_keys():
    keys = {}
    for cls in dict.fromkeys(MODEL_REGISTRY.values()):
        if issubclass(cls, StandardisedModel):
            for key in model_input_keys(cls, consumer='bbc'):
                keys.setdefault(key, cls.__name__)
    return keys


def test_no_model_key_is_named_in_the_general_bbc_code():
    """SALT2mu keys are lowercase, so the spelling to catch is the model's own."""
    offenders = []
    for path in BBC_SOURCES:
        source = path.read_text()
        for key, owner in _bbc_model_keys().items():
            if re.search(rf"['\"]{key}['\"]", source, re.IGNORECASE):
                offenders.append(f"{path.name}: {key} ({owner})")
    assert not offenders, offenders
