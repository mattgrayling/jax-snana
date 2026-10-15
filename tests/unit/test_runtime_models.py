"""Models registered after consumer imports reach parsers and output tables."""

from pathlib import Path

import numpy as np
import pytest

# Import consumers before registering anything, as in an active notebook.
import jax_pipeline
from jax_snlc_fit.config_parser import parse_fit_input_file
from jax_snlc_fit.sntable import snana_sim_block
from jax_snlc_sim.config_parser import parse_snana_input_file
from jax_snlc_core.models import (
    FittedModel, MODEL_REGISTRY, SNModel, SimulatedModel,
    infer_model_type_from_path, register_model,
)
from jax_snlc_core.utils import SNANAError


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
