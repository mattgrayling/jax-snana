"""Models registered after consumer imports reach parsers and output tables."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pytest

# Import consumers before registering anything, as in an active notebook.
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.models import (MODEL_REGISTRY, FittedModel, SimulatedModel, SNModel,
                                  infer_model_type_from_path, register_model)
from jax_snlc_fit.config_parser import parse_fit_input_file
from jax_snlc_fit.output import snana_sim_block
from jax_snlc_sim.inputs import parse_snana_input_file


class RuntimeModel(SNModel, SimulatedModel, FittedModel):
    GENMODEL_ALIASES = ('Runtime',)
    runtime_option: float = 2.
    INPUT_KEY_ALIASES = (('GENPEAK_RUNTIMEWIDTH', 'WIDTH_PEAK'),)
    FILE_ONLY_INPUT_KEYS = ('RUNTIME_OPTION',)
    population_parameters = ('RUNTIMEWIDTH',)
    SNANA_TABLE_SIM_COLUMNS = (('SIM_WIDTH', 'SIM_RUNTIMEWIDTH'),)


@pytest.fixture(autouse=True)
def restore_registry():
    original = dict(MODEL_REGISTRY)
    yield
    MODEL_REGISTRY.clear()
    MODEL_REGISTRY.update(original)


def test_late_registration(tmp_path):
    register_model('runtime', RuntimeModel)
    assert infer_model_type_from_path('/models/RUNTIME.v1') == 'runtime'
    source = Path('tests/fixtures/snana/atmosphere_replay/input.input')
    inp = tmp_path / 'sim.input'
    inp.write_text(source.read_text() + '\nGENMODEL: runtime\n'
                   'RUNTIME_OPTION: 3\nGENRANGE_RUNTIMEWIDTH: 1 9\n')
    config = parse_snana_input_file(inp, overrides=['WIDTH_PEAK', '4'])
    assert config.model._model_input_values['RUNTIME_OPTION'] == 3.
    assert config.model.populations['RUNTIMEWIDTH'].distributions[0].peak == 4.
    flat = parse_snana_input_file(inp, overrides=['GENPDF_FLAT', 'RUNTIMEWIDTH(2:7)'])
    assert flat.model.populations['RUNTIMEWIDTH'].distributions[0].range == (2., 7.)
    with pytest.raises(SNANAError, match='belongs in the sim-input file'):
        parse_snana_input_file(inp, overrides=['RUNTIME_OPTION', '4'])

    fit = tmp_path / 'fit.nml'
    fit.write_text("&FITINP\n FITMODEL_NAME='runtime'\n RUNTIME_OPTION=5\n &END\n")
    fitted = parse_fit_input_file(fit)
    assert fitted.model._model_key_values['RUNTIME_OPTION'] == 5.
    columns = snana_sim_block({'SIM_RUNTIMEWIDTH': np.array([4.])}, False)
    assert list(columns) == ['SIM_WIDTH']
    np.testing.assert_array_equal(columns['SIM_WIDTH'], [4.])


def test_registration_collisions():
    register_model('runtime', RuntimeModel)
    register_model('runtime', RuntimeModel)

    class Revised(RuntimeModel):
        runtime_option: float = 6.

    with pytest.raises(SNANAError, match='already registered'):
        register_model('runtime', Revised)
    register_model('runtime', Revised, replace=True)
    assert MODEL_REGISTRY['runtime'] is Revised

    class Ambiguous(RuntimeModel):
        GENMODEL_ALIASES = ('salt3_custom',)

    with pytest.raises(SNANAError, match='overlaps'):
        register_model('ambiguous', Ambiguous)
    assert 'ambiguous' not in MODEL_REGISTRY


def test_documented_model_simulates_from_its_declaration_alone(tmp_path):
    """The adding-a-model example declares its one population parameter and
    no sampling code: the simulator draws it from the input's distribution,
    stores it under its name and records it as SIM_WIDTH."""
    from jax_snlc_sim import run as sim_run

    doc = Path('docs/adding_a_model.md').read_text()
    exec(doc.split('```python\n', 1)[1].split('```', 1)[0], {})
    keys = next(block.split('```', 1)[0] for block in doc.split('```text\n')[1:]
                if 'GENMODEL: PULSE' in block.split('```', 1)[0])
    inp = tmp_path / 'pulse.input'
    inp.write_text(
        "GENVERSION: PULSE\n"
        "NGENTOT_LC: 100\n"
        "GENFILTERS: griz\n"
        "KCOR_FILE: tests/fixtures/data/kcor/JAX_TEST_DES.fits\n"
        "SIMLIB_FILE: tests/fixtures/data/simlib/JAX_TEST_DES_RFIELD.SIMLIB\n"
        "GENRANGE_REDSHIFT: 0.05 0.3\n"
        "DNDZ: POWERLAW 2.5E-5 1.5\n"
        "GENRANGE_MJD: 56150 56365\n"
        "GENRANGE_PEAKMJD: 56150 56365\n"
        "GENRANGE_TREST: -20 50\n"
        "OPT_MWEBV: 0\n"
        "FORMAT_MASK: 32\n"
        "RANSEED: 12345\n" + keys)
    simulation = sim_run.prepare(str(inp), n_sne=20, random_seed=3)
    results, _, _ = sim_run.run_simulation(simulation)
    width = np.asarray(results['parameters']['width'])
    assert np.all((width >= 5.0) & (width <= 25.0)) and np.std(width) > 0.0
    header = simulation.model.get_snana_header_params({'width': width})
    assert list(header) == ['SIM_WIDTH']



def test_postponed_annotations_preserve_input_markers():
    """Resolving annotations retains consumer restrictions and allowed values."""
    from jax_snlc_core.models.input_keys import model_input_keys

    class ExtendedRuntime(RuntimeModel):
        runtime_option: int = 3
        fit_choice: Annotated[Literal['a', 'b'], 'FIT_ONLY'] = 'a'
        sim_scale: Annotated[float | None, 'SIM_ONLY'] = None
        bbc_scale: Annotated[float, 'BBC_ONLY'] = 1.

    fit = model_input_keys(ExtendedRuntime, 'fit')
    sim = model_input_keys(ExtendedRuntime, 'sim')
    bbc = model_input_keys(ExtendedRuntime, 'bbc')
    assert fit['RUNTIME_OPTION'].kind is int
    assert fit['FIT_CHOICE'].allowed == ('a', 'b')
    assert 'SIM_SCALE' not in fit and 'BBC_SCALE' not in fit
    assert sim['SIM_SCALE'].kind is float
    assert 'FIT_CHOICE' not in sim and 'BBC_SCALE' not in sim
    assert set(bbc) == {'BBC_SCALE'}



def test_postponed_dataclass_model_annotations():
    """A dataclass extension exposes its new option without exposing model state."""
    from jax_snlc_core.models import SALTModel, model_input_keys

    @dataclass(eq=False)
    class ExtendedSALT(SALTModel):
        extra_setting: float = 1.

    keys = model_input_keys(ExtendedSALT)
    assert keys['EXTRA_SETTING'].kind is float
    assert set(keys) == set(model_input_keys(SALTModel)) | {'EXTRA_SETTING'}


def test_subclass_input_markers_replace_inherited_markers():
    """Consumer restrictions apply to the most-derived declaration of a key."""
    from jax_snlc_core.models.input_keys import model_input_keys

    class Parent(RuntimeModel):
        option: Annotated[float, 'FIT_ONLY'] = 1.

    class Child(Parent):
        option: Annotated[int, 'SIM_ONLY'] = 2

    assert 'OPTION' in model_input_keys(Parent, 'fit')
    assert 'OPTION' not in model_input_keys(Child, 'fit')
    assert model_input_keys(Child, 'sim')['OPTION'].kind is int
    assert 'OPTION' not in model_input_keys(Child, 'bbc')


def test_subclass_dataclass_field_replaces_inherited_input_exclusion():
    """A subclass can expose a field that its parent excluded from input."""
    from jax_snlc_core.models.input_keys import model_input_keys

    @dataclass
    class Parent:
        option: float = field(default=1., metadata={'input': False})

    @dataclass
    class Child(Parent):
        option: float = 2.

    assert 'OPTION' not in model_input_keys(Parent)
    assert 'OPTION' in model_input_keys(Child)
