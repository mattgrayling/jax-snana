"""Regressions for configuration equivalence and direct-fitting constraints."""

from dataclasses import asdict, fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from jax_snlc_core.logging import SNANAError


def test_bbc_output_override_preserves_special_nuisance_mode():
    from jax_salt2mu.config import parse_bbc_input
    from jax_salt2mu.config import build_config

    original = parse_bbc_input(None, ['u13=2'])
    changed = parse_bbc_input(None, ['u13=2', '--prefix', 'renamed'])
    assert changed._u_flags == original._u_flags == {'u13': 2}
    expected = asdict(original)
    expected['prefix'] = 'renamed'
    assert asdict(changed) == expected
    assert asdict(build_config({'prefix': 'renamed'}, original)) == expected
    assert original.prefix != 'renamed'


@pytest.mark.parametrize('onebin,flag', [(False, 1), (True, 3)])
def test_bbc_named_floating_parameters_preserve_correction_bins(onebin, flag):
    from jax_salt2mu.config import parse_bbc_input
    from jax_salt2mu.config import build_config
    from jax_salt2mu.binning import build_coeff_bins
    from jax_snlc_core.models import standardised_model

    settings = {'nuisance_float': {'alpha0': True, 'beta0': True}}
    if onebin:
        settings['nuisance_onebin'] = {'alpha0': True, 'beta0': True}
    named = build_config(settings)
    snana = parse_bbc_input(None, [f'u1={flag}', f'u2={flag}'])
    # Provenance distinguishes an omitted one-bin option from an explicit one.
    assert {k: v for k, v in asdict(named).items() if k != '_derived_settings'} == {
        k: v for k, v in asdict(snana).items() if k != '_derived_settings'}
    truth = {'SIM_alpha': np.array([.12, .16, .12, .16]),
             'SIM_beta': np.array([2.5, 2.5, 3.5, 3.5]),
             'SIM_gammaDM': np.zeros(4)}
    model = standardised_model('SALT2')
    assert [b.nbin for b in build_coeff_bins(named, model, truth)] == ([1, 1, 1] if onebin else [2, 2, 1])
    held = build_config({'nuisance_float': {'alpha0': False}}, named)
    assert held._u_flags['u1'] == 0 and held.nuisance_onebin['alpha0']
    reset = build_config({'nuisance_float': {}}, named)
    assert reset._u_flags == {'u1': 0, 'u2': 0}
    assert all(reset.nuisance_onebin.values())


@pytest.mark.parametrize('setting', ['initial', 'fixed'])
@pytest.mark.parametrize('value', [99., float('nan')])
def test_explicit_fit_values_must_respect_model_limits(setting, value):
    from jax_snlc_fit import LightCurveFitter

    fitter = LightCurveFitter(
        'SALT3.K21', batch_size=1, n_fit_iterations=1, maxiter_round1=0,
        **{setting: {'x1': value}})
    times = np.linspace(59990., 60035., 12)
    with pytest.raises(SNANAError, match='Explicit value for x1'):
        fitter.fit(mjd=times, flux=100 * np.exp(-.5 * ((times-60005)/15)**2),
                   flux_err=np.full(12, 5.), band='g_PS1', redshift=.1)


def test_covariance_bin_options_agree_across_input_routes(tmp_path):
    from create_covariance import config_from_args
    from jax_create_cov.config import build_config
    from jax_pipeline import Covariance

    settings = {'input_dir': '.', 'version': 'DATA', 'outdir': 'output'}
    path = tmp_path / 'cov.yaml'
    path.write_text(yaml.safe_dump(settings))
    named = config_from_args([str(path), '--nbin-x1', '2', '--nbin-c', '3'])
    snana = config_from_args([str(path), '--nbin_x1', '2', '--nbin_c', '3'])
    # Route-specific derivation provenance may differ; public input settings
    # and their resolved values must agree across all routes.
    def input_settings(config):
        return {f.name: getattr(config, f.name) for f in fields(config) if f.init}

    assert input_settings(named) == input_settings(snana)
    assert named.mode == 'rebin'
    settings.update(nbin_x1=2, nbin_c=3)
    assert input_settings(build_config(settings)) == input_settings(named)
    path.write_text(yaml.safe_dump(settings))
    distances = SimpleNamespace(name='distances', systematics=lambda: ((), ()), grid_cells=lambda: [(0, 0)])
    cfg, _, _ = Covariance('cov', distances, input_file=path).configuration()
    assert (cfg.mode, cfg.nbin_x1, cfg.nbin_c) == ('rebin', 2, 3)


@pytest.mark.parametrize('position', ['before', 'after', 'explicit'])
def test_simulation_options_can_precede_input_file(tmp_path, position):
    """Named model options resolve through the CLI parser in each argument order."""
    pytest.importorskip("redback_jax")
    from jax_snlc_sim.inputs import parse_cli_args, command_line_config

    path = tmp_path/'sim.yaml'
    path.write_text(yaml.safe_dump(dict(
        genmodel='REDBACK.arnett_bolometric', genfilters='griz',
        simlib_file='survey.simlib', kcor_file='survey.fits', random_seed=42,
        redshift_range=[.1, .2], trest_range=[-10, 40],
        peakmjd_range=[60000, 60010], sigma_ratio=0., ngentot_lc=2)))
    options = ['--ngentot-lc', '3', '--batch-size', '2', '--redback-tgrid-nbin', '128']
    argv = (options + [str(path)] if position == 'before' else
            [str(path)] + options if position == 'after' else options + ['--input', str(path)])
    cfg = command_line_config(parse_cli_args(argv))
    assert cfg.run.ngentot_lc == 3 and cfg.numerical.batch_size == 2
    assert cfg.model._model_input_values['REDBACK_TGRID_NBIN'] == 128


@pytest.mark.parametrize('position', ['before', 'after', 'explicit', 'no_file'])
def test_bbc_options_can_precede_input_file(tmp_path, position):
    from salt2mu import _parse_cli_args
    from jax_salt2mu.config import parse_bbc_input

    path = tmp_path/'bbc.yaml'
    path.write_text('model_lcfit: BAYESN\n')
    options = ['--nzbin', '4']
    argv = (options + [str(path)] if position == 'before' else
            [str(path)] + options if position == 'after' else
            options + ['--input', str(path)] if position == 'explicit' else options)
    filename, tokens, _, _, settings = _parse_cli_args(argv)
    cfg = parse_bbc_input(filename, tokens, settings=settings)
    assert filename == (None if position == 'no_file' else str(path))
    assert cfg.nzbin == 4


def test_reused_fit_configuration_restores_linked_settings():
    from jax_snlc_fit.config_parser import build_config

    original = build_config({'use_model_magerr': False, 'dofit_photoz': True,
                             'lfixpar_all': True, 'fudge_modelerr_scale': .7})
    changed = build_config({'use_model_magerr': True, 'dofit_photoz': False,
                            'lfixpar_all': False}, config=original)
    expected = build_config({'fudge_modelerr_scale': .7})
    assert changed == expected
    assert original.model.fudge_modelerr_scale == 0
    # Direct edits to a detached configuration remain explicit inputs.
    original.model.fudge_modelerr_scale = .4
    changed = build_config({'use_model_magerr': True}, config=original)
    assert changed.model.fudge_modelerr_scale == .4


def test_redundant_bbc_float_override_keeps_explicit_onebin():
    from jax_salt2mu.config import parse_bbc_input

    original = parse_bbc_input(None, ['u1=3'])
    changed = parse_bbc_input(None, ['u1=3', '--nuisance-float', 'alpha0', 'true'])
    assert asdict(original) == asdict(changed)


@pytest.mark.parametrize('syntax', ['named', 'snana'])
def test_covariance_overrides_precede_resource_loading(tmp_path, syntax):
    from create_covariance import config_from_args

    scales = tmp_path/'scales.yaml'
    scales.write_text('CALIB: 0.5\n')
    settings = dict(input_dir='.', version='DATA', outdir='output',
                    sys_scale_file=str(tmp_path/'missing.yaml'))
    if syntax == 'snana':
        settings = {key.upper(): value for key, value in settings.items()}
    path = tmp_path/'cov.yaml'
    path.write_text(yaml.safe_dump(settings))
    cfg = config_from_args([str(path), '--sys-scale-file', str(scales)])
    assert cfg.sys_scales == {'CALIB': .5}


def test_covariance_combines_file_and_cli_bin_counts(tmp_path):
    from create_covariance import config_from_args

    path = tmp_path/'cov.yaml'
    path.write_text('input_dir: .\nversion: DATA\noutdir: output\nnbin_x1: 2\n')
    cfg = config_from_args([str(path), '--nbin-c', '3'])
    assert (cfg.mode, cfg.nbin_x1, cfg.nbin_c) == ('rebin', 2, 3)
    path.write_text(path.read_text() + 'nbin_c: 3\n')
    with pytest.raises(SNANAError, match='conflict'):
        config_from_args([str(path), '--mode', 'unbinned'])
    cfg = config_from_args([str(path), '--unbinned'])
    assert cfg.mode == 'unbinned' and cfg.nbin_x1 == cfg.nbin_c == 0


def test_cli_filter_shorthand_matches_yaml():
    from jax_snlc_fit.config_parser import command_settings
    from jax_snlc_fit.calibration import resolve_filtlist

    calibration = SimpleNamespace(filter_names=['LSST-g', 'LSST-r'],
        header={'NFILTERS': 2, 'FILT001': 'LSST-g', 'FILT002': 'LSST-r'})
    for tokens, expected in [(['gr'], ['LSST-g', 'LSST-r']),
                             (['LSST-g'], ['LSST-g']),
                             (['LSST-g', 'LSST-r'], ['LSST-g', 'LSST-r'])]:
        values = command_settings(['--filtlist-fit', *tokens], {})
        assert resolve_filtlist(values['filtlist_fit'], calibration) == expected


def test_fitter_model_option_precedes_file_selected_model(tmp_path):
    """The real command parser resolves model options before loading settings."""
    from jax_snlc_fit.config_parser import command_line_input, parse_fit_input_file

    path = tmp_path / 'fit.yaml'
    path.write_text('model_type: redback\nfitmodel_name: REDBACK.arnett_bolometric\n')
    input_file, _, overrides = command_line_input(['--redback-tgrid-nbin', '128', str(path)])
    config = parse_fit_input_file(input_file, overrides)
    assert config.model._model_key_values['REDBACK_TGRID_NBIN'] == 128


def test_cli_help_includes_input_and_literal_percent(capsys):
    from create_covariance import config_from_args
    from jax_wfit.config import parse_command

    with pytest.raises(SystemExit) as exit:
        parse_command(['--help'])
    assert exit.value.code == 0
    help_text = ' '.join(capsys.readouterr().out.split())
    assert '68.3% interval' in help_text and '--config' in help_text
    with pytest.raises(SystemExit) as exit:
        config_from_args(['--help'])
    assert exit.value.code == 0
    help_text = capsys.readouterr().out
    assert 'positional arguments:' in help_text and 'input YAML' in help_text


def test_bundled_model_configuration_roundtrip():
    from jax_snlc_fit import LightCurveFitter

    original = LightCurveFitter('SALT3.K21', batch_size=1, use_model_magerr=False)
    changed = LightCurveFitter('SALT3.K21', config=original.configuration(),
                                         use_model_magerr=True)
    assert changed.config.model.fudge_modelerr_scale == 1
    assert changed.config.fit.use_offdiag_covariance
    explicit = original.configuration()
    explicit.model.fudge_modelerr_scale = 0
    held = LightCurveFitter('SALT3.K21', config=explicit, use_model_magerr=True)
    assert held.config.model.fudge_modelerr_scale == 0
    with pytest.raises(SNANAError, match='different models'):
        LightCurveFitter('BAYESN.W22', config=original.configuration())


def test_partial_selection_settings_use_shared_snana_defaults(tmp_path):
    from jax_snlc_fit.config_parser import build_config
    from jax_snlc_fit.config_parser import parse_fit_input_file

    named = build_config({'earlylc': {'maxobs': 5}, 'require_epochs': {'tranges': [-10, 0, 10]}})
    assert named.selection.earlylc['photmask'] == 0
    assert named.selection.require_epochs['snrmin'] == 0.
    path = tmp_path/'fit.nml'
    path.write_text("&SNLCINP\n EARLYLC_STRING='MAXOBS 5'\n REQUIRE_EPOCHS_STRING='ALL -10 0 10'\n/\n&FITINP\n/\n")
    snana = parse_fit_input_file(path)
    assert snana.selection.earlylc == named.selection.earlylc
    assert snana.selection.require_epochs == named.selection.require_epochs
    for name in ('earlylc', 'require_epochs'):
        with pytest.raises(SNANAError, match='unknown|Unknown'):
            build_config({name: {'typo': 1}})


def test_simulation_structured_settings_are_prepared_before_consumption():
    from jax_snlc_sim.inputs import build_config
    from jax_snlc_sim.inputs import build

    settings = dict(genmodel='SALT3.K21', genfilters='griz', simlib_file='survey.simlib',
                    kcor_file='survey.fits', random_seed=42, redshift_range=[.1, .2],
                    trest_range=[-10, 40], peakmjd_range=[60000, 60010], ngentot_lc=2, sigma_ratio=0.)
    cfg = build_config({**settings, 'dndz_rate_mods': {'scale': 2.},
                        'epochs_time_cut': {'snr_min': 5.},
                        'zvariation_poly': [['GENPEAK_SALT2x1', '0,1']],
                        'populations': {'SALT2x1': {'distribution': 'normal', 'peak': 0., 'sigma': 1., 'range': [-3., 3.]}}})
    assert cfg.rates.dndz_rate_mods == {'scale': 2., 'zexp': 0., 'zpoly': [], 'z1poly': []}
    assert cfg.cuts.epochs_time_cut == {'snr_min': 5., 'time_above_max': 99999., 'filters': ''}
    raw = dict(GENMODEL='SALT3.K21', GENFILTERS='griz', RANSEED=42,
               SIMLIB_FILE='survey.simlib', KCOR_FILE='survey.fits',
               GENPEAK_SALT2x1=0., GENSIGMA_SALT2x1=[1., 1.], GENRANGE_SALT2x1=[-3., 3.],
               GENSIGMA_MWEBV_RATIO=0., GENRANGE_REDSHIFT=[.1, .2], ZVARIATION_POLY=[('GENPEAK_SALT2x1', '0,1')])
    expected = build(raw)
    assert cfg.model.populations['x1'].is_shifted()
    assert repr(cfg.model.populations['x1']) == repr(expected.model.populations['x1'])
    assert build_config({**settings, 'dndz_rate_mods': {}}).rates.dndz_rate_mods['zexp'] == 0.
    for extra, match in [({'dndz_entries_pec1a': ['POWERLAW 2e-5 1.5']}, 'NON1A'),
                         ({'model_type': 'bayesn'}, 'Unknown simulation'),
                         ({'dndz_rate_mods': {'typo': 2}}, 'Unknown'),
                         ({'epochs_time_cut': {'typo': 2}}, 'Unknown')]:
        with pytest.raises(SNANAError, match=match):
            build_config({**settings, **extra})
    assert cfg.model.model_type == 'salt'


def test_bbc_configuration_reuse_restores_supplied_settings():
    from jax_salt2mu.config import build_config
    from jax_salt2mu.config import parse_bbc_input

    for cfg in (build_config({'muref_zpoly': [1., 2.], 'chi2max': 25.}),
                parse_bbc_input(None, ['muref_zpoly=1,2', 'chi2max=25'])):
        assert cfg.chi2max == 1.e9
        changed = build_config({'muref_zpoly': []}, cfg)
        assert changed.chi2max == 25.
        assert cfg.chi2max == 1.e9
    floated = build_config({'nuisance_float': {'alpha0': True},
                            'nuisance_onebin': {'alpha0': False}})
    held = build_config({'nuisance_float': {'alpha0': False}}, floated)
    refloated = build_config({'nuisance_float': {'alpha0': True}}, held)
    assert [c._u_flags['u1'] for c in (floated, held, refloated)] == [1, 0, 1]
    assert build_config({'prefix': 'again'}, floated)._u_flags['u1'] == 1


def test_covariance_merges_all_supplied_settings_before_resolution(tmp_path):
    from jax_create_cov.config import build_config
    from jax_pipeline import Covariance

    scales = tmp_path/'scales.yaml'
    scales.write_text('CALIB: 1.0\nDUST: 0.5\n')
    config = build_config({'sys_scale_file': str(scales)}, {'sys_scales': {'CALIB': 2.}})
    assert config.sys_scales == {'CALIB': 2., 'DUST': .5}
    path = tmp_path/'cov.yaml'
    path.write_text('nbin_x1: 2\n')
    distances = SimpleNamespace(name='distances', systematics=lambda: ((), ()), grid_cells=lambda: [(0, 0)])
    cfg, _, _ = Covariance('cov', distances, input_file=path, nbin_c=3, mode='rebin').configuration()
    assert (cfg.mode, cfg.nbin_x1, cfg.nbin_c) == ('rebin', 2, 3)


def test_spectrum_requests_share_preparation_and_validation():
    from jax_snlc_sim.inputs import build_config
    from jax_snlc_sim.spectra import parse_take_spectrum_entry, prepare_spectrum_requests

    settings = dict(genmodel='SALT3.K21', genfilters='griz', simlib_file='survey.simlib',
                    kcor_file='survey.fits', random_seed=42, redshift_range=[.1, .2],
                    trest_range=[-10, 40], peakmjd_range=[60000, 60010], ngentot_lc=2,
                    sigma_ratio=0., spectrograph_file='instrument.fits')
    request = dict(epoch_frame='REST', epoch_range=[-5, 20], opt_texpose=1,
                   zpoly_texpose=[3600, 0])
    named = build_config({**settings, 'take_spectrum': [request]})
    assert named.spectra.take_spectrum == prepare_spectrum_requests([parse_take_spectrum_entry(
        'TAKE_SPECTRUM', 'TREST(-5:20) TEXPOSE_ZPOLY(3600,0)')])
    assert 'wgt' not in request
    with pytest.raises(SNANAError, match='expected SNR_ZPOLY'):
        build_config({**settings, 'take_spectrum': [{'epoch_frame': 'REST'}]})
    with pytest.raises(SNANAError, match='Unknown SpectrumRequest'):
        build_config({**settings, 'take_spectrum': [{**request, 'typo': 1}]})
    with pytest.raises(SNANAError, match='Unknown SpectrumRequest'):
        build_config({**settings, 'take_spectrum': [{**request, 'prescale_key': 0}]})
    with pytest.raises(SNANAError, match='positive'):
        prepare_spectrum_requests([parse_take_spectrum_entry(
            'TAKE_SPECTRUM', 'MJD(60000:60100:0) TEXPOSE_ZPOLY(100)')])


def test_structured_simulation_cli_lists_match_python():
    from jax_snlc_sim.inputs import build_config, command_settings

    base = dict(genmodel='SALT3.K21', genfilters='griz', simlib_file='survey.simlib',
                kcor_file='survey.fits', random_seed=42, redshift_range=[.1, .2],
                trest_range=[-10, 40], peakmjd_range=[60000, 60010], ngentot_lc=2,
                sigma_ratio=0., spectrograph_file='instrument.fits')
    rows = {'take_spectrum': [{'epoch_frame': 'REST', 'epoch_range': [-5, 20],
                              'opt_texpose': 1, 'zpoly_texpose': [3600.]}],
            'dndz_entries': [['POWERLAW', 2.e-5, 1.5]],
            'salt2beta_cpoly': [[2., 3.]]}
    tokens = [token for name, values in rows.items()
              for token in ['--' + name.replace('_', '-'), yaml.safe_dump(values[0], default_flow_style=True).strip()]]
    settings, remaining = command_settings(tokens, base)
    assert not remaining
    cli = build_config({**base, **settings})
    python = build_config({**base, **rows})
    assert asdict(cli) == asdict(python)
    assert cli.dndz_params() == python.dndz_params()
    with pytest.raises(SNANAError, match='invalid YAML'):
        command_settings(['--take-spectrum', '{epoch_frame:'], base)


@pytest.mark.parametrize('overrides', [
    {'DNDZ': 'POWERLAW 2e-5 1.5'},
    {'GENMAG_SMEAR_FILTER': 'g 0.1'},
    {'CUTWIN_SNRMAX': '10 gr OR -10 30'},
    {'GENRANGE_REDSHIFT': [.05, .2]},
])
def test_pipeline_snana_overrides_use_the_line_readers(overrides):
    from jax_pipeline import Sim
    from jax_snlc_sim.inputs import parse_snana_input_file

    path = Path('tests/fixtures/snana/salt3_k21_replay/input.input')
    tokens = [str(item) for name, value in overrides.items()
              for item in (name, *(value if isinstance(value, list) else [value]))]
    expected = parse_snana_input_file(path, tokens)
    actual = Sim('simulation', path, overrides=overrides).configuration()
    assert asdict(actual) == asdict(expected)


def test_pipeline_snana_include_and_exclude_overrides(tmp_path):
    from jax_pipeline import Sim
    from jax_snlc_sim.inputs import parse_snana_input_file

    included = tmp_path/'included.input'
    included.write_text('MAG_OFFSET: 0.2\n')
    replacement = tmp_path/'replacement.input'
    replacement.write_text('MAG_OFFSET: 0.4\n')
    path = tmp_path/'sim.input'
    path.write_text(Path('tests/fixtures/snana/salt3_k21_replay/input.input').read_text()
                    + f'\nINPUT_FILE_INCLUDE: {included}\n')
    overrides = {'EXCLUDE_INCLUDE_FILE': str(included),
                 'INPUT_FILE_INCLUDE': str(replacement), 'NGENTOT_LC': 3}
    tokens = [str(item) for pair in overrides.items() for item in pair]
    expected = parse_snana_input_file(path, tokens)
    actual = Sim('simulation', path, overrides=overrides).configuration()
    assert asdict(actual) == asdict(expected)
    assert actual.model.mag_offset == .4


def test_file_spectrum_requests_preserve_shared_prescale_draws(tmp_path):
    import jax
    from jax_snlc_sim.inputs import parse_snana_input_file
    from jax_snlc_sim.spectra import spectrum_prescales

    path = tmp_path/'sim.input'
    path.write_text(Path('tests/fixtures/snana/spectrum_prescale_replay/input.input').read_text()
                    + '\nTAKE_SPECTRUM: NONE\n'
                    + 'TAKE_SPECTRUM(ALL/2): MJD(60000:60002:1) TEXPOSE_ZPOLY(1000)\n'
                    + 'TAKE_SPECTRUM(ALL/2): TREST(5:5) TEXPOSE_ZPOLY(1000)\n')
    cfg = parse_snana_input_file(path)
    assert [entry['prescale_key'] for entry in cfg.spectra.take_spectrum] == [0, 0, 0, 3]
    named = parse_snana_input_file(path, settings={'take_spectrum': [
        dict(field='ALL', wgt=.5, epoch_frame='MJD', epoch_range=[60000, 60002],
             epoch_step=1., opt_texpose=1, zpoly_texpose=[1000.]),
        dict(field='ALL', wgt=.5, epoch_frame='REST', epoch_range=[5, 5],
             opt_texpose=1, zpoly_texpose=[1000.])]})
    assert named.spectra.take_spectrum == cfg.spectra.take_spectrum
    results = {'parameters': {'field_idx': np.zeros(32, dtype=int),
                             'sn_global_idx': np.arange(32), 'cid': np.arange(32)}}
    simulator = SimpleNamespace(config=cfg, fields=[SimpleNamespace(field_name='ALL')])
    keep = spectrum_prescales(simulator, results, key=jax.random.PRNGKey(42))
    assert keep.any() and not keep.all()
    np.testing.assert_array_equal(keep[:, 0], keep[:, 1])
    np.testing.assert_array_equal(keep[:, 0], keep[:, 2])
    np.testing.assert_array_equal(keep, spectrum_prescales(
        SimpleNamespace(config=named, fields=simulator.fields), results,
        key=jax.random.PRNGKey(42)))
    assert np.any(keep[:, 0] != keep[:, 3])


@pytest.mark.parametrize('named', [False, True])
def test_spectrum_source_conflicts_use_completed_configuration(tmp_path, named):
    from jax_snlc_sim.inputs import parse_input

    base = dict(genmodel='SALT3.K21', genfilters='griz', simlib_file='survey.simlib',
                kcor_file='survey.fits', random_seed=42, redshift_range=[.1, .2],
                trest_range=[-10, 40], peakmjd_range=[60000, 60010], ngentot_lc=2,
                sigma_ratio=0., spectrograph_file='instrument.fits',
                use_simlib_take_spectrum=True)
    path = tmp_path/'sim.input'
    path.write_text(yaml.safe_dump(base) if named else
                    Path('tests/fixtures/snana/spectrum_prescale_replay/input.input').read_text()
                    + '\nTAKE_SPECTRUM: NONE\nUSE_SIMLIB_TAKE_SPECTRUM: 1\n')
    request = {'epoch_frame': 'REST', 'epoch_range': [0, 0],
               'opt_texpose': 1, 'zpoly_texpose': [1000.]}
    with pytest.raises(SNANAError, match='not both'):
        parse_input(path, {'take_spectrum': [request]})
    assert parse_input(path, {'use_simlib_take_spectrum': False,
                             'take_spectrum': [request]}).spectra.take_spectrum


@pytest.mark.parametrize('parameter,name,dust', [('SALT2x1', 'x1', False),
                                                  ('AV', 'AV', True)])
@pytest.mark.parametrize('bounds', [(0., 2.), (1., 1.)])
def test_uniform_populations_preserve_snana_sampling(parameter, name, dust, bounds):
    import jax
    from jax_snlc_core.models import MODEL_REGISTRY
    from jax_snlc_sim.inputs import _build_distributions
    from jax_snlc_sim.distributions import sample_exp_halfgauss, sample_gengauss_asym
    from jax_snlc_sim.population import distribution_from_settings

    lo, hi = bounds
    snana, _ = _build_distributions(
        {'GENPDF_FLAT': f'{parameter}({lo}:{hi})'}, MODEL_REGISTRY['salt'], {})
    named = distribution_from_settings({'distribution': 'uniform', 'range': bounds}, dust=dust)
    assert named == snana[name]
    keys = jax.random.split(jax.random.PRNGKey(14), 32)
    sample = sample_exp_halfgauss if dust else sample_gengauss_asym
    np.testing.assert_array_equal(jax.vmap(lambda key: sample(key, named))(keys),
                                  jax.vmap(lambda key: sample(key, snana[name]))(keys))


@pytest.mark.parametrize('perfect', [0, 16])
def test_named_dust_replaces_snana_profile_before_perfect_mode(tmp_path, perfect):
    import jax
    from jax_snlc_sim.inputs import parse_snana_input_file
    from jax_snlc_sim.distributions import sample_exp_halfgauss

    path = tmp_path/'sim.input'
    base = Path('tests/fixtures/snana/spectrum_prescale_replay/input.input').read_text()
    path.write_text(base + f'\nGENRANGE_AV: 0 1\nGENTAU_AV: 0.1\nGENPERFECT: {perfect}\n')
    values = {'AV': {'distribution': 'exponential', 'range': [0, 2], 'exp_tau': .4}}
    named = parse_snana_input_file(path, settings={'populations': values})
    snana = parse_snana_input_file(path, ['GENRANGE_AV', '0', '2', 'GENTAU_AV', '.4'])
    keys = jax.random.split(jax.random.PRNGKey(17), 32)
    named_draws = jax.vmap(lambda key: sample_exp_halfgauss(
        key, named.model.populations['AV'].distributions[0]))(keys)
    snana_draws = jax.vmap(lambda key: sample_exp_halfgauss(
        key, snana.model.populations['AV'].distributions[0]))(keys)
    np.testing.assert_array_equal(named_draws, snana_draws)
    assert named.host.av_range == snana.host.av_range
    assert named.host.tau_av == snana.host.tau_av
    assert values['AV']['range'] == [0, 2]


@pytest.mark.parametrize('dust', [False, True])
def test_native_population_records_use_shared_validation(dust):
    from jax_snlc_sim.distributions import GenExpHalfGauss, GenGaussAsym
    from jax_snlc_sim.population import distribution_from_settings

    record_type = GenExpHalfGauss if dust else GenGaussAsym
    supplied = record_type(range=(0., 2.))
    converted = distribution_from_settings(supplied, dust=dust)
    assert converted == supplied and converted is not supplied
    for bounds in ((2., 0.), (0., np.inf)):
        with pytest.raises(SNANAError, match='Invalid population range'):
            distribution_from_settings(record_type(range=bounds), dust=dust)
    width = {'exp_tau': -1.} if dust else {'sigma': (-1., 1.)}
    with pytest.raises(SNANAError, match='widths'):
        distribution_from_settings(record_type(range=(0., 2.), **width), dust=dust)


@pytest.mark.parametrize('explicit_range', [False, True])
def test_wv07_uses_supplied_av_range(tmp_path, explicit_range):
    import jax
    from jax_snlc_sim.inputs import parse_snana_input_file
    from jax_snlc_sim.sampler import _sample_wv07_av

    path = tmp_path/'sim.input'
    base = Path('tests/fixtures/snana/spectrum_prescale_replay/input.input').read_text()
    path.write_text(base + '\nGENAV_WV07: 1\n' +
                    ('GENRANGE_AV: 0 1\n' if explicit_range else ''))
    named = parse_snana_input_file(path, settings={'populations': {
        'AV': {'distribution': 'exponential', 'range': [0, 2], 'exp_tau': .7}}})
    expected_range = (0., 1.) if explicit_range else (0., 2.)
    snana = parse_snana_input_file(path, ['GENRANGE_AV', *map(str, expected_range)])
    assert named.host.av_range == expected_range
    keys = jax.random.split(jax.random.PRNGKey(17), 32)
    samples = _sample_wv07_av(named.host, keys)
    assert np.isfinite(samples).all()
    np.testing.assert_array_equal(samples, _sample_wv07_av(snana.host, keys))


@pytest.mark.parametrize('values,tokens,field', [
    ({'idsurvey_select': '1,2'}, ['idsurvey_select=1,2'], 'idsurvey_select'),
    ({'zbinuser': '0.01,0.1,0.2'}, ['zbinuser=0.01,0.1,0.2'], 'zbinuser'),
    ({'simfile_ccprior': 'H11'}, ['simfile_ccprior=H11'], 'use_h11'),
])
def test_bbc_snana_mapping_strings_keep_their_syntax(values, tokens, field):
    from jax_salt2mu.config import parse_bbc_input

    expected = parse_bbc_input(None, tokens)
    for other in ({}, {'u1': 1}):
        actual = parse_bbc_input(None, {**values, **other})
        assert getattr(actual, field) == getattr(expected, field)
    actual = parse_bbc_input(None, values, settings={'nuisance_val': {'alpha0': .14}})
    assert getattr(actual, field) == getattr(expected, field)
    assert parse_bbc_input(None, {'idsurvey_select': [1, 2]}).idsurvey_select == [1, 2]


def test_bbc_batch_header_is_separate_from_scientific_inputs(tmp_path):
    from jax_salt2mu.config import parse_bbc_input

    path = tmp_path/'bbc.input'
    path.write_text('CONFIG:\n  BATCH_INFO: sbatch example 1\n#END_YAML\nnzbin=10\n')
    assert asdict(parse_bbc_input(path)) == asdict(parse_bbc_input(None, ['nzbin=10']))
    path.write_text('nzbin: [\n')
    with pytest.raises(SNANAError, match='Cannot read'):
        parse_bbc_input(path)


def test_covariance_override_aliases_resolve_before_merging():
    from jax_create_cov.config import build_config

    assert build_config({'write_mask_cov': 3}, {'WRITE_MASK_COV': 2}).write_mask_cov == 2
    assert build_config({'WRITE_MASK_COV': 3}, {'write_mask_cov': 2}).write_mask_cov == 2


@pytest.mark.parametrize('subpop', [False, True])
def test_population_dust_conflicts_are_checked_after_merging(tmp_path, subpop):
    from jax_snlc_sim.inputs import parse_input

    path = tmp_path/'sim.input'
    path.write_text(Path('tests/fixtures/snana/salt3_k21_replay/input.input').read_text()
                    + '\nGENRANGE_AV: 0 1\nGENTAU_AV: 0.1\n')
    dust = {'distribution': 'exponential', 'exp_tau': .1, 'range': [0, 1]}
    settings = ({'subpops': [{'name': 'dusty', 'weight': 1., 'populations': {'EBV_HOST': dust}}]}
                if subpop else {'populations': {'EBV_HOST': dust}})
    with pytest.raises(SNANAError, match='more than one host dust distribution'):
        parse_input(path, settings)


def test_simulator_reports_conditional_population_draw_failure():
    from jax_snlc_sim.population import Population
    from jax_snlc_sim.run import validate_batch
    from jax_snlc_sim.simulation import Model, SimulationConfig

    genpdf = SimpleNamespace(cond_varnames=('LOGMASS',),
                             cond_grids=(np.array([9., 11.]),))
    simulator = SimpleNamespace(
        config=SimulationConfig(),
        model_settings=Model(populations={'SALT2c': Population(genpdf=genpdf)}))
    results = {
        'parameters': {'cid': np.array([1]), 'prefilter_pass': np.array([True]),
                       'prefilter_reason_code': np.array([0])},
        'observation_schedule': {},
    }
    with pytest.raises(SNANAError, match="Event 7: GENPDF for SALT2c.*LOGMASS"):
        validate_batch(simulator, results, {'draw_input_failure': np.array([1])},
                       start_index=7, n_events=1)
