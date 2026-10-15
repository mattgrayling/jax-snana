"""Exercise public input and execution contracts independently of numerical parity."""

from dataclasses import asdict

import jax
import numpy as np
import pandas as pd
import pytest
import yaml

from jax_create_cov import compute_covariance
from jax_create_cov.config import CreateCovConfig
from jax_create_cov.config_parser import build_config as covariance_config
from jax_salt2mu import fit_distances
from jax_salt2mu.inputs import build_config as distance_config
from jax_snlc_core.utils import SNANAError
from jax_snlc_fit import LightCurveFitter
from jax_snlc_sim import parse_input, prepare, simulate
from jax_wfit import fit_cosmology, WFITConfig
from jax_wfit.config_parser import build as cosmology_config


@pytest.fixture
def active_precision(active):
    previous = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', active)
    try:
        yield
    finally:
        jax.config.update('jax_enable_x64', previous)


def covariance_inputs():
    nominal = pd.DataFrame({'ROW': [1, 2], 'zHD': [.1, .2], 'MU': [38., 40.],
                            'MUREF': [38., 40.], 'MUERR': [.1, .1]})
    shifted = nominal.copy()
    shifted['MU'] += [.02, -.03]
    return dict(diagrams={'FITOPT000_MUOPT000': nominal, 'FITOPT001_MUOPT000': shifted},
                fitopt_scales={0: ('DEFAULT', 1.), 1: ('zpt', 1.)},
                muopt_scales={0: ('DEFAULT', 1.)})


@pytest.mark.parametrize('changes, expected', [
    ({'subtract_vpec': True}, 'unbinned'),
    ({'rebin_bins': {}}, 'rebin'),
])
def test_covariance_inferred_mode_survives_config_overrides(changes, expected):
    base = covariance_config({})
    before = asdict(base)
    assert covariance_config(base, changes).mode == expected
    assert covariance_config({}, changes).mode == expected
    assert asdict(base) == before
    assert covariance_config(covariance_config(base, changes)).mode == expected


def test_covariance_explicit_mode_and_caller_edits_are_preserved():
    with pytest.raises(SNANAError, match='conflict'):
        covariance_config(CreateCovConfig(mode='binned'), {'rebin_bins': {}})
    edited = covariance_config({})
    edited.mode = 'unbinned'
    assert covariance_config(edited).mode == 'unbinned'
    automatic = covariance_config({'subtract_vpec': True})
    assert covariance_config(automatic, {'subtract_vpec': False}).mode == 'binned'


def test_scale_file_override_replaces_values(tmp_path):
    first, second = tmp_path / 'first.yml', tmp_path / 'second.yml'
    first.write_text('zpt: 2\nother: 5\n')
    second.write_text('zpt: 3\nother: 6\n')
    config = covariance_config({'sys_scale_file': first, 'sys_scales': {'explicit': 4}})
    config.sys_scales['other'] = 7
    result = covariance_config(config, {'sys_scale_file': second})
    assert result.sys_scales == {'zpt': 3., 'other': 7., 'explicit': 4.}
    assert covariance_config(result, {'sys_scale_file': None}).sys_scales == {
        'other': 7., 'explicit': 4.}
    assert config.sys_scales['zpt'] == 2.


def test_cosmology_method_is_inferred_again_after_overrides():
    base = cosmology_config({})
    before = asdict(base)
    grid = cosmology_config(base, {'outfile_chi2grid': 'grid.txt'})
    assert grid.method == 'grid'
    assert cosmology_config(grid).method == 'grid'
    assert cosmology_config(grid, {'outfile_chi2grid': None}).method == 'laplace'
    assert asdict(base) == before
    explicit = cosmology_config(WFITConfig(method='laplace'), {'outfile_chi2grid': 'grid.txt'})
    with pytest.raises(SNANAError, match='grid output'):
        explicit.validate()
    base.method = 'nuts'
    assert cosmology_config(base).method == 'nuts'


@pytest.mark.parametrize('stage', ['distance', 'covariance', 'cosmology'])
@pytest.mark.parametrize('form', ['keywords', 'mapping', 'yaml', 'config'])
def test_analysis_input_forms_run_without_writing(stage, form, repo_root, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    if stage == 'distance':
        function, builder = fit_distances, distance_config
        values = dict(datafile=[str(repo_root / 'tests/fixtures/snana/bbc_des_beams/DATA.FITRES')],
                      nzbin=5, zmin=.1, zmax=1.)
        changes = {'nzbin': 10}
        kwargs = {}
    elif stage == 'covariance':
        function, builder = compute_covariance, covariance_config
        values, kwargs = {'sys_scales': {'zpt': 1.}}, covariance_inputs()
        changes = {'sys_scales': {'zpt': 2.}}
    else:
        function, builder = fit_cosmology, cosmology_config
        values = dict(hubble_diagram=str(repo_root / 'tests/fixtures/wfit/SALT2mu.M0DIF'),
                      model='wcdm')
        changes = {'model': 'lcdm'}
        kwargs = {}
    before = yaml.safe_dump(values)
    if form == 'keywords':
        result = function(**{**values, **changes}, **kwargs)
    else:
        inputs = values
        if form == 'yaml':
            inputs = tmp_path / 'input.yml'
            inputs.write_text(before)
        elif form == 'config':
            inputs = builder(values)
        result = function(inputs, **changes, **kwargs)
    if stage == 'distance':
        assert result.fit.nsnfit > 0
        assert len(result.hubble_diagram) == 10
    elif stage == 'covariance':
        np.testing.assert_allclose(result.covariances['ALL']['matrix'],
                                   [[.0016, -.0024], [-.0024, .0036]])
    else:
        assert np.isfinite(result[0]['point_estimate']['omega_m'])
        assert set(result[0]['point_estimate']) == {'omega_m'}
    assert yaml.safe_dump(values) == before
    assert {p.name for p in tmp_path.iterdir()} == ({'input.yml'} if form == 'yaml' else set())


@pytest.mark.parametrize('stage', ['simulation', 'fitter', 'distance', 'covariance', 'cosmology'])
@pytest.mark.parametrize('active', [False, True])
def test_conflicting_precision_fails_before_execution(stage, active, active_precision,
                                                      repo_root, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    request = not active
    with pytest.raises(SNANAError, match='active JAX precision'):
        if stage == 'simulation':
            simulate(repo_root / 'tests/fixtures/snana/salt3_fit_select/input.input',
                     double_precision=request)
        elif stage == 'fitter':
            LightCurveFitter('SALT3.K21', double_precision=request)
        elif stage == 'distance':
            fit_distances(datafile=[str(repo_root / 'tests/fixtures/snana/bbc_des_beams/DATA.FITRES')],
                          double_precision=request)
        elif stage == 'covariance':
            compute_covariance(**covariance_inputs(), double_precision=request)
        else:
            fit_cosmology(hubble_diagram=str(repo_root / 'tests/fixtures/wfit/SALT2mu.M0DIF'),
                           double_precision=request)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('active', [False, True])
@pytest.mark.parametrize('precision_setting', [None, 'matching'])
def test_covariance_omitted_and_matching_precision_preserve_process(active, precision_setting,
                                                                   active_precision):
    settings = {} if precision_setting is None else {'double_precision': active}
    result = compute_covariance(**covariance_inputs(), **settings)
    assert len(result.hubble_diagram) == 2
    assert jax.config.jax_enable_x64 == active


@pytest.mark.parametrize('settings', [
    {'simlib_dump': 1}, {'dashboard': True}, {'init_only': 1},
    {'simgen_dump': True, 'simgen_dumpall_columns': []},
])
def test_simulation_setup_modes_fail_before_resources_or_files(settings, snana_fixture,
                                                               tmp_path, monkeypatch):
    source = snana_fixture('salt3_fit_select').input_file
    config = parse_input(source, settings)
    before = asdict(config)
    monkeypatch.chdir(tmp_path)

    def unexpected_build(*args):
        pytest.fail('Rejected modes must not load simulation resources')

    monkeypatch.setattr('jax_snlc_sim.run.build_simulation', unexpected_build)
    with pytest.raises(SNANAError, match='setup-only mode'):
        simulate(config)
    assert not list(tmp_path.iterdir())
    assert asdict(config) == before


def test_prepare_still_runs_requested_simlib_dump(snana_fixture, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    simulator = prepare(snana_fixture('salt3_fit_select').input_file,
                                      simlib_dump=1)
    resources = simulator
    assert resources is None and simulator is None
    assert len(list(tmp_path.glob('SIMLIB_DUMP_AVG_*.TEXT'))) == 1


@pytest.mark.parametrize('stage', ['simulation', 'fitter', 'distance', 'cosmology'])
@pytest.mark.parametrize('precision_setting', [None, False])
@pytest.mark.parametrize('active', [False])
def test_python_stages_run_in_single_precision(stage, precision_setting, active,
                                               active_precision, repo_root, tmp_path,
                                               monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = {} if precision_setting is None else {'double_precision': precision_setting}
    if stage == 'simulation':
        results, selection, _ = simulate(
            repo_root / 'tests/fixtures/snana/salt3_fit_select/input.input',
            ngentot_lc=2, batch_size=2, **settings)
        observed = np.asarray(selection['valid_mask'])
        assert np.isfinite(np.asarray(results['fluxcals'])[observed]).all()
    elif stage == 'fitter':
        fitter = LightCurveFitter('SALT3.K21', batch_size=1, n_fit_iterations=1,
                                  maxiter_round1=4, **settings)
        times = np.linspace(59990., 60035., 12)
        flux = 100. * np.exp(-.5 * ((times - 60005.) / 15.) ** 2)
        result = fitter.fit(mjd=times, flux=flux, flux_err=np.full(12, 5.),
                             band='g_PS1', redshift=.1)
        assert result.n_sne == 1 and np.isfinite(result.chi_squared).all()
    elif stage == 'distance':
        result = fit_distances(
            datafile=[str(repo_root / 'tests/fixtures/snana/bbc_des_beams/DATA.FITRES')],
            nzbin=10, zmin=.1, zmax=1., **settings)
        assert result.fit.nsnfit > 0 and np.isfinite(result.output.mu).all()
    else:
        result = fit_cosmology(
            hubble_diagram=str(repo_root / 'tests/fixtures/wfit/SALT2mu.M0DIF'),
            model='lcdm', **settings)
        assert np.isfinite(result[0]['point_estimate']['omega_m'])
    assert jax.config.jax_enable_x64 is False
    assert not list(tmp_path.iterdir())
