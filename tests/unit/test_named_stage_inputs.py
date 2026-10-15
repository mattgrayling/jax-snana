"""Input routes agree before executing the scientific stages."""
from dataclasses import asdict
import pytest
import yaml

from jax_snlc_core.utils import SNANAError
from jax_snlc_sim.inputs import build_config as simulation_config


def simulation_settings():
    return dict(genmodel='SALT3.K21', genfilters='griz', simlib_file='survey.simlib',
                kcor_file='survey.fits', random_seed=42, redshift_range=[.1, .2],
                trest_range=[-10, 40], peakmjd_range=[60000, 60010],
                sigma_ratio=0., ngentot_lc=2)


def test_simulation_named_and_legacy_settings_agree():
    from jax_snlc_sim.config_parser import build, validate
    raw = dict(GENMODEL='SALT3.K21', GENFILTERS='griz', SIMLIB_FILE='survey.simlib',
               KCOR_FILE='survey.fits', RANSEED=42, GENRANGE_REDSHIFT=[.1,.2],
               GENRANGE_TREST=[-10,40], GENRANGE_PEAKMJD=[60000,60010],
               GENSIGMA_MWEBV_RATIO=0., NGENTOT_LC=2)
    legacy = build(raw)
    validate(legacy, raw)
    assert asdict(simulation_config(simulation_settings())) == asdict(legacy)


def test_populations_and_subpops_are_distinct():
    from jax_snlc_sim.distributions import GenGaussAsym
    settings = simulation_settings()
    settings['populations'] = {'SALT2x1': dict(distribution='normal', peak=0., sigma=1., range=[-3,3])}
    settings['subpops'] = [dict(name='young', weight=2., populations={
        'SALT2x1': dict(distribution='normal', peak=1., sigma=.5, range=[-3,3])}),
        dict(name='old', weight=1.)]
    config = simulation_config(settings)
    draws = config.model.populations['x1'].distributions
    assert len(draws) == 2
    assert all(isinstance(d, GenGaussAsym) for d in draws)
    assert [d.peak for d in draws] == [1., 0.]
    assert [s.weight for s in config.model.subpops] == [2., 1.]
    assert settings['populations']['SALT2x1']['sigma'] == 1.


def test_simulation_missing_required_inputs_fail():
    values = simulation_settings()
    del values['random_seed']
    with pytest.raises(SNANAError, match='RANSEED'):
        simulation_config(values)
    with pytest.raises(SNANAError, match='Unknown simulation'):
        simulation_config({**simulation_settings(), 'batch_szie': 4})


def test_simulation_cli_generated_and_numerical_controls():
    from jax_snlc_sim.inputs import command_settings
    values = simulation_settings()
    extra, legacy = command_settings(['--batch-size', '7', '--hostlib-maxread', '50'], values)
    assert not legacy
    # HOSTLIB settings require an explicit catalogue.
    cfg = simulation_config({**values, **extra, 'hostlib_file': 'hosts.txt'})
    assert cfg.numerical.batch_size == 7
    assert cfg.host.hostlib_maxread == 50


def test_bbc_named_parameters_match_snana_slots(tmp_path):
    from jax_salt2mu.config_parser import parse_bbc_input
    from jax_salt2mu.inputs import build_config
    legacy = parse_bbc_input(None, ['model_lcfit=SALT2', 'p1=0.14', 'u1=1'])
    named = build_config({'model_lcfit':'SALT2', 'nuisance_val':{'alpha0':.14},
                           'nuisance_float':{'alpha0':True}, 'nuisance_onebin':{'alpha0':False}})
    assert asdict(legacy) == asdict(named)
    path = tmp_path / 'bbc.yaml'
    path.write_text(yaml.safe_dump({'model_lcfit':'BAYESN','nzbin':5}))
    assert parse_bbc_input(path, settings={'nzbin':8}).nzbin == 8
    with pytest.raises(SNANAError, match='nuisance'):
        build_config({'model_lcfit':'BAYESN', 'nuisance_val':{'alpha':.1}})


def test_covariance_named_selectors_and_cli(tmp_path):
    from jax_create_cov.config_parser import build_config, apply_filter
    from create_covariance import config_from_args
    named = build_config({'covopts':[{'label':'calibration', 'fitopt_filter':{'include':'zpt', 'exclude':'u'}, 'scale':2.}], 'subtract_vpec':True})
    assert named.mode == 'unbinned'
    assert apply_filter('zpt_g', named.covopts[0].fitopt_filter)
    assert not apply_filter('zpt_u', named.covopts[0].fitopt_filter)
    path=tmp_path/'cov.yaml'
    path.write_text('input_dir: .\nversion: DATA\noutdir: output\nsystematic_hd: true\n')
    config=config_from_args([str(path),'--write-mask-cov','7','--no-label-cov-rows'])
    assert config.write_mask_cov == 7 and config.systematic_hd
    with pytest.raises(SNANAError, match='Unknown'):
        build_config({'write_mask_cvo':7})


def test_wfit_generated_options_preserve_false_and_precision(tmp_path):
    from jax_wfit.config_parser import parse_command
    path=tmp_path/'wfit.yaml'
    path.write_text('hubble_diagram: data.txt\nmarg: true\ndouble_precision: true\n')
    cfg, _ = parse_command(['--config',str(path),'--no-marg','--grid-chunk-size','12'])
    assert cfg.marg is False and cfg.double_precision is True
    assert cfg.grid_chunk_size == 12
    legacy, _ = parse_command(['data.txt','-ommin','0.1','-double_precision','1'])
    named, _ = parse_command(['data.txt','--ommin','0.1','--double-precision'])
    assert asdict(legacy) == asdict(named)


def test_simulation_cli_reads_named_options(tmp_path, monkeypatch):
    import snlc_sim
    values = simulation_settings()
    path = tmp_path / 'sim.yaml'
    path.write_text(yaml.safe_dump(values))
    seen = []
    monkeypatch.setattr(snlc_sim, 'configure_precision', lambda value: None)
    monkeypatch.setattr(snlc_sim.sim_run, 'prepare', lambda cfg: seen.append(cfg))
    snlc_sim.main([str(path), '--ngentot-lc', '5', '--batch-size', '3'])
    assert seen[0].run.ngentot_lc == 5
    assert seen[0].numerical.batch_size == 3


def test_simulation_linked_settings_resolve_after_named_overrides(tmp_path):
    from jax_snlc_sim.inputs import parse_input
    values = simulation_settings()
    values.update(gensigma_peakmjd=1., opt_setpkmjd=16)
    assert simulation_config(values).rates.opt_setpkmjd == 0
    path = tmp_path / 'sim.input'
    path.write_text('GENMODEL: SALT3.K21\nGENFILTERS: griz\nSIMLIB_FILE: survey.simlib\nKCOR_FILE: survey.fits\nRANSEED: 42\nGENRANGE_REDSHIFT: .1 .2\nGENRANGE_TREST: -10 40\nGENRANGE_PEAKMJD: 60000 60010\nGENSIGMA_MWEBV_RATIO: 0\nNGENTOT_LC: 2\n')
    cfg = parse_input(path, {'gensigma_peakmjd':1., 'opt_setpkmjd':16})
    assert cfg.rates.opt_setpkmjd == 0


def test_named_simulation_seed_and_numerical_settings_reach_pipeline(tmp_path):
    from jax_pipeline import Sim
    values=simulation_settings()
    values['batch_size']=3
    path=tmp_path/'sim.yaml'
    path.write_text(yaml.safe_dump(values))
    cfg=Sim('simulation',path,overrides={'random_seed':91, 'ngentot_lc':4}).configuration()
    assert cfg.run.random_seed == 91
    assert cfg.run.ngentot_lc == 4
    assert cfg.numerical.batch_size == 3


def test_covariance_pipeline_retains_file_settings_and_explicit_false(tmp_path):
    from types import SimpleNamespace
    from jax_pipeline import Covariance
    source=SimpleNamespace(name='distances',systematics=lambda:(('zpt_g',),()),grid_cells=lambda:[(0,0),(1,0)])
    path=tmp_path/'cov.yaml'
    path.write_text('mode: unbinned\nsystematic_hd: true\nlabel_cov_rows: true\nmxsize_test_posdef: 40\ncovopts:\n  - label: calibration\n    fitopt_filter: {include: zpt}\n')
    cfg,_,_=Covariance('cov',source,input_file=path,systematic_hd=False).configuration()
    assert cfg.mode=='unbinned' and not cfg.systematic_hd
    assert cfg.label_cov_rows and cfg.mxsize_test_posdef==40
    assert cfg.covopts[0].label=='calibration'


def test_simulation_detection_mask_is_resolved():
    cfg=simulation_config({**simulation_settings(),'apply_searcheff_opt':3})
    assert cfg.survey._searcheff_apply
    assert cfg.survey._searcheff_apply_pipeline and cfg.survey._searcheff_apply_spec


def test_named_dust_and_subpopulation_offsets_match_shared_values():
    cfg=simulation_config({**simulation_settings(),'genmag_off_global':.1,
        'populations':{'AV':{'distribution':'exponential','exp_tau':.2,'range':[0,2]}},
        'subpops':[{'name':'one','weight':1.}]})
    assert cfg.host.dust_enabled and cfg.host.tau_av==.2
    assert cfg.model.subpops[0].genmag_off_global==.1


@pytest.mark.parametrize('settings,tokens', [
    ({'vpecerr': 250.}, ['vpecerr=250']),
    ({'simfile_ccprior': ['H11']}, ['simfile_ccprior=H11']),
    ({'muref_zpoly': [0., 1.]}, ['muref_zpoly=0,1']),
    ({'fixpar_all': True}, ['fixpar_all=1']),
])
def test_bbc_linked_settings_agree(settings, tokens):
    from jax_salt2mu.inputs import build_config
    from jax_salt2mu.config_parser import parse_bbc_input
    assert asdict(build_config(settings)) == asdict(parse_bbc_input(None, tokens))


@pytest.mark.parametrize('settings', [
    {'nzbin': 5, 'nlogzbin': 5}, {'uM0': 42},
    {'sigint_mode': 'unknown'}, {'cutmask_write': 42},
])
def test_bbc_invalid_input_combinations_fail(settings):
    from jax_salt2mu.inputs import build_config
    with pytest.raises(SNANAError):
        build_config(settings)


def test_bbc_named_overrides_precede_validation(tmp_path):
    from jax_salt2mu.config_parser import parse_bbc_input
    path = tmp_path / 'bbc.input'
    path.write_text('uM0=42 model_lcfit=BAYESN\n')
    assert parse_bbc_input(path, ['--uM0', '1']).uM0 == 1
    assert parse_bbc_input(path, {'uM0': 1}).uM0 == 1


def test_bbc_pipeline_merges_input_layers_before_preparation(tmp_path):
    from jax_pipeline import BBC

    path = tmp_path / 'bbc.input'
    path.write_text('p1=0.1\nu13=2\nx1min=-3\nx1max=3\nchi2max=25\n')
    stage = BBC('distances', data=[], input_file=path,
                overrides={'p1': .12, 'x1min': -2},
                settings={'nuisance_val': {'alpha0': .14},
                          'muref_zpoly': [0., 1.]},
                muopts={'changed': {'p1': .16, 'x1max': 2}},
                muopt_settings={'changed': {'muref_zpoly': []},
                                'typed': {'nuisance_val': {'alpha0': .18}}})
    original = stage.configuration()
    changed = stage.configuration(1)
    typed = stage.configuration(2)
    assert original.nuisance_val['alpha0'] == .14
    assert original.chi2max == 1.e9
    assert changed.nuisance_val['alpha0'] == .16
    assert changed.lcpar_range['x1'] == (-2., 2.)
    assert changed.chi2max == 25.
    assert typed.nuisance_val['alpha0'] == .18
    assert original._u_flags['u13'] == changed._u_flags['u13'] == typed._u_flags['u13'] == 2
    assert stage.settings['muref_zpoly'] == [0., 1.]


def test_bbc_pipeline_typed_muopt_retains_stage_dependencies():
    from jax_pipeline import BBC, Fit

    original = Fit('original', fitres='original.FITRES')
    alternate = Fit('alternate', fitres='alternate.FITRES')
    stage = BBC('distances', data=original,
                settings={'nuisance_val': {'alpha0': .14}},
                muopt_settings={'alternate': {'datafile': alternate,
                                             'nuisance_val': {'alpha0': .16}}})
    assert stage.inputs() == {'data:original': original, 'data:alternate': alternate}
    assert stage.systematics() == ((), ('alternate',))
    assert stage.configuration(1).nuisance_val['alpha0'] == .16


def test_bbc_supplied_translation_preserves_defaults_and_modes():
    from jax_salt2mu.config_parser import parse_bbc_input
    from jax_salt2mu.inputs import build_config

    raw = parse_bbc_input(None, {'u13': 2, 'idsurvey_select': '1,2'},
                          settings={'prefix': 'out'})
    assert raw._u_flags['u13'] == 2
    assert raw.idsurvey_select == [1, 2]
    assert asdict(build_config({'prefix': 'out'})) == asdict(parse_bbc_input(None, settings={'prefix': 'out'}))
    changed = parse_bbc_input(None, {'u13': 2},
                              settings={'nuisance_float': {'scalePCC': False}})
    assert changed._u_flags['u13'] == 0
    with pytest.raises(SNANAError, match='use settings'):
        parse_bbc_input(None, {'nuisance_val': {'alpha0': .14}})


def test_bbc_muopt_can_disable_true_ia_selection_before_rules(tmp_path):
    from jax_pipeline import BBC
    from jax_salt2mu.config_parser import parse_bbc_input

    path = tmp_path / 'bbc.input'
    path.write_text('select_trueIa=1\nprescale_simcc=2\n'
                    'simfile_ccprior=prior.FITRES\nvarname_pIa=PROB_IA\n')
    stage = BBC('distances', data=[], input_file=path,
                muopts={'all': {'select_trueIa': 0}},
                muopt_settings={'typed': {'select_trueia': False}})
    assert stage.configuration().prescale_simcc == 999999.
    for index in (1, 2):
        result = stage.configuration(index)
        assert result.prescale_simcc == 2.
        assert result.simfile_ccprior == ['prior.FITRES']
        assert result.varname_pia == 'PROB_IA'
        assert result.nuisance_float == {}
    expected = parse_bbc_input(path, {'select_trueIa': 0})
    assert asdict(stage.configuration(1)) == asdict(expected)


def test_bbc_releasing_snana_held_parameter_restores_coefficient_bins(tmp_path):
    from jax_pipeline import BBC
    from jax_salt2mu.config_parser import parse_bbc_input
    from jax_salt2mu.inputs import build_config

    path = tmp_path / 'bbc.input'
    path.write_text('u1=0\nu13=2\n')
    floating = {'nuisance_float': {'alpha0': True}}
    stage = BBC('distances', data=[], input_file=path,
                muopt_settings={'float': floating})
    held = stage.configuration()
    assert held._u_flags['u1'] == 0 and held.nuisance_onebin['alpha0']
    for cfg in (stage.configuration(1), build_config(floating, held),
                parse_bbc_input(path, settings=floating)):
        assert cfg._u_flags['u1'] == 1
        assert not cfg.nuisance_onebin['alpha0']
        assert cfg._u_flags['u13'] == 2
    explicit_onebin = parse_bbc_input(None, {'u1': 3})
    assert build_config(floating, explicit_onebin)._u_flags['u1'] == 3


def test_bbc_powzbin_override_replaces_the_complete_snana_option(tmp_path):
    from jax_pipeline import BBC
    from jax_salt2mu.config_parser import parse_bbc_input

    path = tmp_path / 'bbc.input'
    path.write_text('powzbin=3,0.4\n')
    stage = BBC('distances', data=[], input_file=path,
                muopts={'uniform_transition_off': {'powzbin': 2}})
    assert stage.configuration().znhalf == .4
    expected = parse_bbc_input(None, {'powzbin': 2})
    assert asdict(stage.configuration(1)) == asdict(expected)


def test_bbc_yaml_cannot_supply_internal_slot_flags(tmp_path):
    from jax_salt2mu.config_parser import parse_bbc_input

    path = tmp_path / 'bbc.yaml'
    path.write_text('model_lcfit: SALT2\n_u_flags: {u13: 2}\n')
    with pytest.raises(SNANAError, match='Private BBC'):
        parse_bbc_input(path)


def test_bbc_mapping_settings_preserve_defaults_and_explicit_resets(tmp_path):
    from jax_salt2mu.config import BBCConfig
    from jax_salt2mu.config_parser import parse_bbc_input
    from jax_salt2mu.inputs import build_config

    supplied = {'blindpar': {9: (.1, 4522.)}}
    named = build_config(supplied)
    snana = parse_bbc_input(None, {'blindpar9': '.1,4522'})
    path = tmp_path / 'bbc.yaml'
    path.write_text(yaml.safe_dump(supplied))
    assert asdict(named) == asdict(snana) == asdict(parse_bbc_input(path))
    assert named.blindpar[11] == BBCConfig().blindpar[11]
    assert build_config({'blindpar': {}}).blindpar == {}
    assert build_config({'blindpar': {}}, named).blindpar == {}
    assert supplied == {'blindpar': {9: (.1, 4522.)}}


def test_bbc_snana_global_cut_preserves_order_across_input_layers(tmp_path):
    from jax_pipeline import BBC
    from jax_salt2mu.config_parser import parse_bbc_input

    path = tmp_path / 'bbc.input'
    path.write_text('chi2max(DES)=12\n')
    with pytest.raises(SNANAError, match='Cannot define chi2max after'):
        parse_bbc_input(path, {'chi2max': 16})
    stage = BBC('distances', data=[], input_file=path,
                muopts={'global': {'chi2max': 16}})
    with pytest.raises(SNANAError, match='Cannot define chi2max after'):
        stage.configuration(1)
    typed = parse_bbc_input(path, settings={'chi2max': 16})
    assert typed.chi2max == 16 and typed.chi2max_surveys == {'DES': 12}


def test_genperfect_override_uses_the_same_preparation_as_file_input(tmp_path):
    from jax_snlc_sim.config_parser import parse_snana_input_file
    text = ('GENMODEL: SALT3.K21\nGENFILTERS: griz\nSIMLIB_FILE: survey.simlib\n'
            'KCOR_FILE: survey.fits\nRANSEED: 42\nGENRANGE_REDSHIFT: .1 .2\n'
            'GENRANGE_TREST: -10 40\nGENRANGE_PEAKMJD: 60000 60010\n'
            'GENSIGMA_MWEBV_RATIO: 0\nNGENTOT_LC: 2\n')
    path = tmp_path / 'simulation.input'
    path.write_text(text)
    overridden = parse_snana_input_file(path, ['GENPERFECT', '16'])
    path.write_text(text + 'GENPERFECT: 16\n')
    in_file = parse_snana_input_file(path)
    a, b = asdict(overridden), asdict(in_file)
    for values in (a, b):
        values['run'].pop('_input_lines')
    assert a == b


def test_simulation_cli_aliases_share_configuration(tmp_path, data_root, monkeypatch):
    import snlc_sim
    from dataclasses import asdict
    settings = simulation_settings()
    settings.update(genmodel=str(data_root / 'models/SALT3.K21'),
                    kcor_file=str(data_root / 'kcor/JAX_TEST_LSST.fits'),
                    simlib_file=str(data_root / 'simlib/JAX_TEST_DES.SIMLIB'),
                    ransystpar={'SIGSCALE_MWEBV': .1})
    path = tmp_path / 'sim.yaml'
    path.write_text(yaml.safe_dump(settings))
    prepared = []
    def prepare(simulation):
        prepared.append(asdict(simulation))
        return None
    monkeypatch.setattr(snlc_sim.sim_run, 'prepare', prepare)
    monkeypatch.setattr(snlc_sim, 'configure_precision', lambda value: None)
    snlc_sim.main([str(path), '--seed', '7', '--chunk-size', '3', '--output', 'test'])
    snlc_sim.main([str(path), '--random-seed', '7', '--batch-size', '3', '--genversion', 'test'])
    assert prepared[0] == prepared[1]
    assert prepared[0]['run']['random_seed'] == 7


@pytest.mark.parametrize('systematic_seed,generation_seed', [(0, 0), (7, 0), (7, 91)])
def test_prepare_seed_is_resolved_before_random_systematics(
        tmp_path, data_root, monkeypatch, systematic_seed, generation_seed):
    from dataclasses import asdict
    from jax_snlc_sim import run
    settings = simulation_settings()
    settings.update(genmodel=str(data_root / 'models/SALT3.K21'),
                    kcor_file=str(data_root / 'kcor/JAX_TEST_LSST.fits'),
                    simlib_file=str(data_root / 'simlib/JAX_TEST_DES.SIMLIB'),
                    simlib_dump=0,
                    ransystpar={'SIGSHIFT_OMEGA_MATTER': .01,
                                'RANSEED_SYST': systematic_seed,
                                'RANSEED_GEN': generation_seed})
    prepared_configs = []
    monkeypatch.setattr(run, 'run_simlib_dump',
                        lambda config: prepared_configs.append(config) or [])
    assert run.prepare({**settings, 'random_seed': 123}) is None
    expected = prepared_configs[-1]
    assert run.prepare(settings, random_seed=123) is None
    supplied = prepared_configs[-1]
    assert asdict(supplied) == asdict(expected)
    path = tmp_path / 'simulation.yaml'
    path.write_text(yaml.safe_dump(settings))
    assert run.prepare(path, random_seed=123) is None
    from_file = prepared_configs[-1]
    assert asdict(from_file.cosmology) == asdict(expected.cosmology)
    assert from_file.run.random_seed == expected.run.random_seed == (generation_seed or 123)
    assert settings['random_seed'] == 42
    assert run.prepare(supplied) is None
    assert asdict(prepared_configs[-1]) == asdict(supplied)
    with pytest.raises(SNANAError, match='Apply simulation settings'):
        run.prepare(supplied, random_seed=456)


@pytest.mark.parametrize('flag', ['dashboard', 'simgen_dump', 'simgen_dumpall'])
@pytest.mark.parametrize('enabled', [False, True])
def test_explicit_simulation_flags(flag, enabled, tmp_path):
    from jax_snlc_sim.inputs import parse_input, command_settings
    settings = simulation_settings()
    path = tmp_path / 'sim.input'
    columns = '' if flag == 'dashboard' else '2 CID ZCMB'
    path.write_text(_simulation_snana_text() + f'{flag.upper()}: {columns}\n')
    option = '--' + ('' if enabled else 'no-') + flag.replace('_', '-')
    overrides, unused = command_settings([option], settings)
    assert not unused
    for cfg in (simulation_config({**settings, flag: enabled}),
                parse_input(path, overrides)):
        owner = cfg.run if flag == 'dashboard' else cfg.output
        assert getattr(owner, flag) is enabled
    if flag != 'dashboard':
        assert owner.simgen_dumpall_columns == ['CID', 'ZCMB', 'FLAG_ACCEPT']


def test_named_override_replaces_all_snana_aliases(tmp_path):
    from jax_snlc_sim.inputs import parse_input
    path = tmp_path / 'sim.input'
    path.write_text(_simulation_snana_text() + 'FUDGESCALE_MWEBV: 1.1\nGENSCALE_MWEBV: 1.05\n')
    assert parse_input(path, {'scale': 1.2}).mw_dust.scale == 1.2


@pytest.mark.parametrize('genperfect', [0, 2, 4, 6])
def test_per_band_settings_share_preparation(genperfect, tmp_path):
    from jax_snlc_sim.inputs import parse_input
    settings = simulation_settings()
    maps = dict(exposure_time_global=3., exposure_time={'g': 2.},
                mjd_template={'r': 59000.}, fudgescale_fluxerr={'i': 1.3},
                genmag_smear_filter={'g': .1})
    named = simulation_config({**settings, **maps})
    path = tmp_path / 'sim.input'
    text = _simulation_snana_text() + f'GENPERFECT: {genperfect}\n'
    path.write_text(text)
    overridden = parse_input(path, maps)
    path.write_text(text + 'EXPOSURE_TIME: 3\nEXPOSURE_TIME_FILTER: g 2\n'
                    'MJD_TEMPLATE_FILTER: r 59000\nFUDGESCALE_FLUXERR_FILTER: i 1.3\n'
                    'GENMAG_SMEAR_FILTER: g .1\n')
    legacy = parse_input(path)
    configs = [overridden, legacy] + ([named] if genperfect == 0 else [])
    exposure = dict(g=2e5, r=1e5, i=1e5, z=1e5) if genperfect & 2 else dict(g=2., r=3., i=3., z=3.)
    for cfg in configs:
        assert cfg.survey.exposure_time == exposure
        assert cfg.survey.mjd_template == dict(g=0., r=59000., i=0., z=0.)
        assert cfg.instrument.fudgescale_fluxerr == dict(g=1., r=1., i=1.3, z=1.)
        assert cfg.model.genmag_smear_filter == ({} if genperfect & 4 else {'g': .1})


def _simulation_snana_text():
    return ('GENMODEL: SALT3.K21\nGENFILTERS: griz\nSIMLIB_FILE: survey.simlib\n'
            'KCOR_FILE: survey.fits\nRANSEED: 42\nGENRANGE_REDSHIFT: .1 .2\n'
            'GENRANGE_TREST: -10 40\nGENRANGE_PEAKMJD: 60000 60010\n'
            'GENSIGMA_MWEBV_RATIO: 0\nNGENTOT_LC: 2\n')


@pytest.mark.parametrize('name,neutral', [
    ('fudgeshift_zpt', 0.), ('fudgescale_fluxerr', 1.),
    ('fudgescale_fluxerr2', 1.), ('fudgeshift_lam', 0.),
    ('fudge_magerr', 0.), ('fudge_zpterr', 0.),
])
def test_typed_band_overrides_preserve_global_setting(name, neutral, tmp_path):
    from jax_snlc_sim.inputs import parse_input
    path = tmp_path / 'sim.input'
    path.write_text(_simulation_snana_text() + f'{name.upper()}: 1.2\n'
                    f'{name.upper()}_FILTER: r 1.4\n')
    config = parse_input(path, {name: {'g': 1.3}})
    assert getattr(config.instrument, name) == dict(g=1.3, r=1.2, i=1.2, z=1.2)
    config = simulation_config({**simulation_settings(), name: {'g': 1.3}})
    assert getattr(config.instrument, name) == dict(g=1.3, r=neutral, i=neutral, z=neutral)


def test_exposure_settings_do_not_depend_on_mapping_order():
    for overrides in (dict(exposure_time={'g': 2.}, exposure_time_global=3.),
                      dict(exposure_time_global=3., exposure_time={'g': 2.})):
        config = simulation_config({**simulation_settings(), **overrides})
        assert config.survey.exposure_time == dict(g=2., r=3., i=3., z=3.)


def test_spectrum_polynomial_agrees_across_input_formats(tmp_path):
    from jax_snlc_sim.inputs import parse_input, command_settings
    settings = {**simulation_settings(), 'warp_spectrum_lampoly': [1., .01]}
    yaml_path = tmp_path / 'sim.yaml'
    yaml_path.write_text(yaml.safe_dump(settings))
    snana_path = tmp_path / 'sim.input'
    snana_path.write_text(_simulation_snana_text() + 'WARP_SPECTRUM: LAMPOLY(1,.01)\n')
    options, unused = command_settings(['--warp-spectrum-lampoly', '1', '.01'], simulation_settings())
    assert not unused
    configs = [simulation_config(settings), parse_input(yaml_path), parse_input(snana_path),
               simulation_config({**simulation_settings(), **options}),
               parse_input(snana_path, {'warp_spectrum_lampoly': []})]
    for config in configs[:-1]:
        assert config.spectra.warp_spectrum_lampoly == [1., .01]
    assert configs[-1].spectra.warp_spectrum_lampoly == []


@pytest.mark.parametrize('input_format', ['yaml', 'snana'])
def test_simulation_readmes_round_trip_individually_and_combined(tmp_path, input_format):
    from types import SimpleNamespace
    from jax_snlc_sim.inputs import parse_input
    from jax_snlc_sim.output.driver import _write_readme_file, write_combined_readme

    settings = {**simulation_settings(),
        'genprefix': 'SN: INPUT_KEYS and GENTYPE_TO_NAME # provenance',
        'populations': {'SALT2x1': {'distribution': 'normal', 'peak': 0.,
                                   'sigma': 1., 'range': [-3., 3.]}},
        'subpops': [{'name': 'young', 'weight': 2., 'populations': {
            'SALT2x1': {'distribution': 'normal', 'peak': 1.,
                        'sigma': .5, 'range': [-3., 3.]}}},
                    {'name': 'old', 'weight': 1.}],
    }
    path = tmp_path / f'simulation.{input_format}'
    path.write_text(yaml.safe_dump(settings) if input_format == 'yaml'
                    else _simulation_snana_text() + 'FUDGESCALE_FLUXERR: 1.3\n')
    config = parse_input(path, {'batch_size': 7} if input_format == 'yaml' else None)
    resources = SimpleNamespace(survey_name='TEST', photometry=SimpleNamespace(
        filter_names=tuple('griz'),
        transmissions={band: None for band in 'griz'},
        filter_mean_wavelengths=[4800., 6200., 7500., 8700.]))
    closing = ('  GENTYPE_TO_NAME:\n    1: Ia SALT3\n'
               '  OUTPUT_SUMMARY:\n    NGENLC_TOT: 2\n    NGENLC_WRITE: 2\n'
               '    NGENSPEC_WRITE: 0\n    CPU_MINUTES: 0.1\n')
    misc = tmp_path / 'misc'
    misc.mkdir()
    parts = []
    for index in range(2):
        name = f'part{index}'
        readme = _write_readme_file(config, resources, 1, True, 2, str(path),
                                    'combined', name, str(misc), closing)
        document = yaml.safe_load(readme.read_text())['DOCUMENTATION']
        keys = document['INPUT_KEYS']
        if input_format == 'yaml':
            for key, value in {**settings, 'batch_size': 7}.items():
                assert keys[key] == value
        else:
            assert keys['FUDGESCALE_FLUXERR'] == 1.3
        parts.append((str(path), name))
    combined = write_combined_readme(tmp_path, 'combined', parts)
    document = yaml.safe_load(combined.read_text())['DOCUMENTATION']
    for index in range(2):
        merged = document[f'INPUT_KEYS_SNIaMODEL{index:02d}']
        assert 'NGENTOT_LC' not in merged
        if input_format == 'yaml':
            for key, value in {**settings, 'batch_size': 7}.items():
                assert merged[key] == value
        else:
            assert merged['FUDGESCALE_FLUXERR'] == 1.3
    assert document['FILTERS'] == dict(g=4800., r=6200., i=7500., z=8700.)
    assert document['STAT_SUMMARY'][0].split()[:3] == ['TOTAL', '4', '4']
