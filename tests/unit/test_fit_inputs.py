"""The same named settings reach the fitter from Python, YAML and argv."""

from pathlib import Path

import pytest

from jax_snlc_core.logging import SNANAError
from jax_snlc_fit.config import Fit, FitConfig
from jax_snlc_fit.config_parser import (build, build_config, command_line_input, command_settings,
                                        declarations,
                                        parse_fit_input_file, read_fit_input)


def test_python_yaml_and_cli_settings_agree(tmp_path):
    values = {'fitmodel_name': 'REDBACK.arnett_bolometric', 'batch_size': 8,
              'fitwin_trest': [-100., 45.], 'redback_tgrid_nbin': 300,
              'fixed': {'mej': 3.}, 'use_model_magerr': False}
    path = tmp_path / 'fit.input'
    import yaml
    path.write_text(yaml.safe_dump(values))
    python = build_config(values)
    file = parse_fit_input_file(path)
    cli = build_config(command_settings([
        '--fitmodel-name', 'REDBACK.arnett_bolometric', '--batch-size', '8',
        '--fitwin-trest', '-100', '45', '--redback-tgrid-nbin', '300',
        '--fixed', 'mej', '3', '--no-use-model-magerr'], {}))
    assert python == file == cli
    assert cli.fit.fitwin_trest == (-100., 45.)
    assert cli.model._model_key_values == {'REDBACK_TGRID_NBIN': 300}


def test_yaml_value_does_not_change_file_format(tmp_path):
    path = tmp_path / 'fit.input'
    path.write_text('output_name: |\n  NAME = value\n  &FITINP\nbatch_size: 8\n')
    config = parse_fit_input_file(path)
    assert config.output.output_name == 'NAME = value\n&FITINP\n'
    assert config.fit.batch_size == 8


@pytest.mark.parametrize('text', ['batch_size: [', '- batch_size', '12: value'])
def test_malformed_yaml_reports_input_error(tmp_path, text):
    path = tmp_path / 'fit.input'
    path.write_text(text)
    with pytest.raises(SNANAError):
        read_fit_input(path)


def test_namelist_is_recognised_after_comments(tmp_path):
    path = tmp_path / 'fit.input'
    path.write_text('! fit settings\n\n&FITINP ! controls\n BATCH_SIZE = 8\n&END\n')
    config = parse_fit_input_file(path)
    assert config.fit.batch_size == 8


@pytest.mark.parametrize('settings', [
    {'dofit_photoz': True},
    {'opt_photoz': 2},
    {'opt_photoz': 8},
    {'use_model_magerr': False},
    {'lfixpar_all': True},
    {'lfixpar_adjust': True},
])
def test_linked_settings_match_snana(settings):
    named = build_config(settings)
    legacy = build({key.upper(): value for key, value in settings.items()})
    assert named.fit == legacy.fit
    assert named.photoz == legacy.photoz
    assert named.model == legacy.model


def test_override_is_applied_before_value_validation(tmp_path):
    path = tmp_path / 'fit.input'
    path.write_text('&FITINP\n NFIT_ITERATION = 0\n&END\n')
    config = parse_fit_input_file(path, ['NFIT_ITERATION', '2'])
    assert config.fit.n_fit_iterations == 2


def test_model_storage_is_not_a_user_setting():
    types, _, _ = declarations('redback')
    assert 'redback_tgrid_nbin' in types
    assert 'model_key_values' not in types
    assert not any(name.startswith('_') for name in types)


def test_prior_override_does_not_need_a_matching_key_in_the_file(tmp_path):
    path = tmp_path / 'fit.input'
    path.write_text('&FITINP\n FITMODEL_NAME = SALT3.K21\n&END\n')
    config = parse_fit_input_file(path, ['PRIOR_x1_RANGE', '-2', '2'])
    assert config.model.priors['X1']['flat_rolloff'] == {'lo': -2., 'hi': 2.}


@pytest.mark.parametrize('text', [
    'fitmodel_name: SALT3.K21\nbatch_size: 8\n',
    '&FITINP\n FITMODEL_NAME = SALT3.K21\n BATCH_SIZE = 8\n&END\n',
])
def test_cli_resolves_model_override(tmp_path, text):
    """CLI model overrides select the model's named options before parsing them."""
    from jax_snlc_fit.config_parser import fitted_model_type

    path = tmp_path / 'fit.input'
    path.write_text(text)
    input_file, _, overrides = command_line_input([
        str(path), '--fitmodel-name', 'REDBACK.arnett_bolometric',
        '--redback-tgrid-nbin', '128', '--batch-size', '2'])
    config = parse_fit_input_file(input_file, overrides)
    assert fitted_model_type(config) == 'redback'
    assert config.fit.batch_size == 2
    assert config.model._model_key_values['REDBACK_TGRID_NBIN'] == 128


def test_mapping_overrides_preserve_other_entries_and_caller():
    config = build_config({'fixed': {'mej': 3., 'kappa': 0.1},
                           'priors': {'mej': {'distribution': 'normal',
                                             'mean': 3., 'sigma': 0.5}}})
    changed = build_config({'fixed': {'mej': None},
                           'priors': {'mej': {'distribution': 'uniform',
                                             'lower': 1., 'upper': 5.}}}, config=config)
    assert config.model.fixed['mej'] == 3.
    assert changed.model.fixed == {'mej': None, 'kappa': 0.1}
    assert 'sigma' not in changed.model.priors['mej']
    assert build_config({'priors': {}}, config=changed).model.priors == {}


@pytest.mark.parametrize('values', [
    {'batch_sze': 4}, {'batch_size': 1.5}, {'use_model_magerr': 'perhaps'},
    {'fitwin_trest': [1.]}, {'redback_tgrid_nbin': 100},
])
def test_invalid_named_settings_fail(values):
    with pytest.raises(SNANAError):
        build_config(values)


def test_new_field_needs_no_parser_registration():
    """The shared settings parser discovers real fields on an extended owner."""
    from dataclasses import dataclass, field
    from jax_snlc_core.input_settings import build_dataclass, settings_parser

    @dataclass
    class ExtendedFit(Fit):
        example_setting: int = 1
        derived_value: int = field(default=0, init=False)

    args = settings_parser(ExtendedFit).parse_args(['--example-setting', '12'])
    assert vars(args) == {'example_setting': 12}
    assert build_dataclass(ExtendedFit, vars(args)).example_setting == 12
    with pytest.raises(SNANAError, match='Unknown ExtendedFit setting'):
        build_dataclass(ExtendedFit, {'derived_value': 5})


def test_omitted_precision_and_explicit_false_are_distinct():
    assert build_config().fit.double_precision is None
    assert build_config({'double_precision': False}).fit.double_precision is False


def test_config_copy_is_detached():
    original = FitConfig()
    copied = build_config(config=original)
    copied.model.initial['x1'] = 1.
    assert original.model.initial == {}


def test_prior_cli_replaces_one_distribution():
    values = command_settings(['--prior', 'mej', 'normal', 'mean=3', 'sigma=.5',
                               '--prior', 'kappa', 'none'], {})
    assert values == {'priors': {'mej': {'distribution': 'normal', 'mean': 3, 'sigma': .5},
                                'kappa': None}}


def test_prior_cli_can_reset_to_model_defaults():
    values = command_settings(['--prior', 'mej', 'normal', 'mean=3', 'sigma=.5',
                               '--prior', '{}'], {})
    assert values == {'priors': {}}


@pytest.mark.parametrize('prior', [
    {'distribution': 'normal', 'mean': 0, 'sigma': 0},
    {'distribution': 'loguniform', 'lower': 0, 'upper': 1},
    {'distribution': 'uniform', 'lower': 2, 'upper': 1},
    {'distribution': 'normal', 'mean': 0, 'sigm': 1},
])
def test_prior_domains_fail(prior):
    from jax_snlc_fit.parameters import parse_prior
    with pytest.raises(SNANAError):
        parse_prior(prior)


def test_public_time_prior_uses_absolute_mjd():
    import jax.numpy as jnp
    import numpy as np
    from jax_snlc_fit.parameters import per_sn, prior_chi2
    data = per_sn(jnp.array([0.]), jnp.array([35.]), jnp.array([60000.]))
    prior = (('gauss', 0, {'centre': 60002., 'sigma': 2., 'public_time': True}),)
    np.testing.assert_allclose(prior_chi2(jnp.array([0.]), prior, data[0]), 1.)


def test_negative_cli_values_use_standard_argparse_syntax():
    config = build_config(command_settings([
        '--mwebv-shift=-1e-2', '--fixed', '{x1: -1e0}',
        '--fitwin-trest', '-100', '45'], {}))
    assert config.data.mwebv_shift == -0.01
    assert config.model.fixed == {'x1': -1.0}
    assert config.fit.fitwin_trest == (-100., 45.)


def test_cli_shared_options_can_precede_input_file(tmp_path):
    path = tmp_path / 'fit.yaml'
    path.write_text('fitmodel_name: SALT3.K21\n')
    input_file, _, overrides = command_line_input(['--batch-size', '2', str(path)])
    config = parse_fit_input_file(input_file, overrides)
    assert config.fit.batch_size == 2


def test_named_overrides_reject_unused_positional_values():
    with pytest.raises(SystemExit) as error:
        command_settings(['BATCH_SZE', '4'], {})
    assert error.value.code == 2
    with pytest.raises(SNANAError, match='invalid YAML'):
        command_settings(['--fixed', '{x1: [}'], {})


@pytest.mark.parametrize('overrides', [
    ['--use-model-magerr'], {'use_model_magerr': True},
])
def test_named_override_precedes_snana_derived_settings(tmp_path, overrides):
    path = tmp_path / 'fit.nml'
    path.write_text('&FITINP\n USE_MODEL_MAGERR = F\n&END\n')
    config = parse_fit_input_file(path, overrides)
    assert config.model.use_model_magerr
    assert config.model.fudge_modelerr_scale == 1.
    assert config.fit.use_offdiag_covariance


def test_named_override_replaces_invalid_snana_range_before_validation(tmp_path):
    path = tmp_path / 'fit.nml'
    path.write_text('&FITINP\n NFIT_ITERATION = 0\n BATCH_SIZE = -2\n&END\n')
    config = parse_fit_input_file(path, ['--n-fit-iterations', '2', '--batch-size', '4'])
    assert config.fit.n_fit_iterations == 2
    assert config.fit.batch_size == 4


@pytest.mark.parametrize('text, overrides, fitopts', [
    ('fitmodel_name: SALT3.K21\nbatch_size: 8\n',
     {'batch_size': 2, 'fixed': {'x1': .5}, 'fitwin_trest': (-15., 45.)},
     {'narrow': {'priors': {'c': {'distribution': 'normal', 'mean': 0., 'sigma': .1}}}}),
    ('&FITINP\n FITMODEL_NAME = SALT3.K21\n BATCH_SIZE = 8\n&END\n',
     {'BATCH_SIZE': 2, 'FITWIN_TREST': (-15., 45.)},
     {'narrow': {'PRIOR_c_SIGMA': .1}}),
])
def test_pipeline_fitter_keeps_typed_override_mappings(tmp_path, text, overrides, fitopts):
    from jax_pipeline import Fit
    path = tmp_path / 'fit.input'
    path.write_text(text)
    stage = Fit('FIT', path, overrides=overrides, fitopts=fitopts)
    baseline = stage.configuration()
    varied = stage.configuration((1, 0))
    assert baseline.fit.batch_size == varied.fit.batch_size == 2
    assert baseline.fit.fitwin_trest == (-15., 45.)
    assert varied.model.priors
    assert not baseline.model.priors
    if 'fixed' in overrides:
        assert baseline.model.fixed == {'x1': .5}
        assert varied.model.priors['c']['distribution'] == 'normal'


def test_unknown_snana_mapping_override_is_rejected(tmp_path):
    path = tmp_path / 'fit.nml'
    path.write_text('&FITINP\n&END\n')
    with pytest.raises(SNANAError, match='BATCH_SZE'):
        parse_fit_input_file(path, {'BATCH_SZE': 4})


def test_pipeline_variation_preserves_other_prior_entries(tmp_path):
    from jax_pipeline import Fit
    path = tmp_path / 'fit.yaml'
    path.write_text('fitmodel_name: SALT3.K21\n')
    shape = {'distribution': 'normal', 'mean': 0., 'sigma': 1.}
    colour = {'distribution': 'normal', 'mean': 0., 'sigma': .1}
    stage = Fit('FIT', path, overrides={'priors': {'x1': shape}},
                fitopts={'colour': {'priors': {'c': colour}}})
    assert stage.configuration((1, 0)).model.priors == {'x1': shape, 'c': colour}
    assert stage.overrides == {'priors': {'x1': shape}}


def test_snana_yaml_accepts_mixed_case_parameter_suffixes(tmp_path):
    path = tmp_path / 'fit.yaml'
    path.write_text('FITMODEL_NAME: SALT3.K21\nINIVAL_x1: 0.5\n')
    config = parse_fit_input_file(path)
    assert config.model.inival == {'X1': .5}


def test_named_and_snana_defaults_agree():
    from dataclasses import asdict
    assert asdict(build({})) == asdict(build_config())
    assert build_config({'tables': {}}).output.tables == {}
    assert build_config({'tables': {'lcplot': 'key'}}).output.mxlc == 5
    assert build({}, settings={'tables': {'lcplot': 'key'}}).output.mxlc == 5


def test_model_options_can_precede_file_with_explicit_model(tmp_path):
    path = tmp_path / 'fit.yaml'
    path.write_text('fitmodel_name: BAYESN.W22\n')
    input_file, _, overrides = command_line_input([
        '--fitmodel-name', 'SALT3.K21', '--opt-salt2fit', '1', str(path)])
    config = parse_fit_input_file(input_file, overrides)
    assert config.model._model_key_values['OPT_SALT2FIT'] == 1


@pytest.mark.parametrize('setting, quantity', [
    ('cutwin', 'SNRMAX'), ('epoch_cutwin', 'MJD')])
def test_cut_quantity_case_and_override_precedence(tmp_path, setting, quantity):
    """Case changes must not create duplicate cuts across input layers."""
    flag = '--' + setting.replace('_', '-')
    lower, mixed = quantity.lower(), quantity.title()
    base = build_config({setting: {quantity: (1., 20.)}})
    updated = build_config({setting: {mixed: (2., 30.)}}, config=base)
    assert getattr(updated.selection, setting) == {lower: (2., 30.)}
    assert getattr(base.selection, setting) == {lower: (1., 20.)}

    # A repeated CLI spelling must win even if that spelling appeared earlier.
    tokens = [flag, quantity, '1', '20', flag, lower, '2', '30',
              flag, quantity, '3', '40']
    cli = command_settings(tokens, {})
    assert getattr(build_config(cli).selection, setting) == {lower: (3., 40.)}

    path = tmp_path / 'fit.yaml'
    path.write_text(f'{setting}: {{{quantity}: [1, 20]}}\n')
    file = parse_fit_input_file(path, tokens)
    assert getattr(file.selection, setting) == {lower: (3., 40.)}
    file = parse_fit_input_file(path, {setting: {mixed: (4., 50.)}})
    assert getattr(file.selection, setting) == {lower: (4., 50.)}

    native = tmp_path / 'fit.nml'
    native.write_text(f'&SNLCINP\n CUTWIN_{quantity} = 1, 20\n&END\n')
    file = parse_fit_input_file(native, {setting: {mixed: (4., 50.)}})
    assert getattr(file.selection, setting) == {lower: (4., 50.)}
    assert getattr(build_config({setting: {}}, config=base).selection, setting) == {}


def test_cut_quantity_normalization_preserves_column_and_band_names():
    config = build_config({'private_cutwin': {'PRIVATE(Foo)': (1., 20.)},
                           'simvar_cutwin': {'x1': (-2., 2.)},
                           'epcut_snrmin': {'ZTF_g': 3.}})
    assert config.selection.private_cutwin == {'PRIVATE(Foo)': (1., 20.)}
    assert config.selection.simvar_cutwin == {'x1': (-2., 2.)}
    assert config.selection.epcut_snrmin == {'ZTF_g': 3.}


@pytest.mark.parametrize('selector', [
    ['FITMODEL_NAME', 'REDBACK.arnett_bolometric'],
    ['FITMODEL_NAME=REDBACK.arnett_bolometric'],
])
def test_cli_snana_model_override_discovers_named_model_options(tmp_path, selector):
    path = tmp_path / 'fit.nml'
    path.write_text('&FITINP\n FITMODEL_NAME = SALT3.K21\n&END\n')
    input_file, _, overrides = command_line_input([
        str(path), *selector, '--redback-tgrid-nbin', '128'])
    config = parse_fit_input_file(input_file, overrides)
    assert config.model.fitmodel_name == 'REDBACK.arnett_bolometric'
    assert config.model._model_key_values['REDBACK_TGRID_NBIN'] == 128
