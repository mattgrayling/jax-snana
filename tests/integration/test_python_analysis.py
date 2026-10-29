"""Script entry points preserve the established analysis and control file writing."""

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
import yaml

from jax_create_cov import compute_covariance
from jax_create_cov.config import load_config as covariance_config
from jax_salt2mu import fit_distances
from jax_salt2mu.config import parse_bbc_input
from jax_salt2mu.run import run_bbc_fit
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import read_fitres
from jax_wfit import WFITConfig, fit_cosmology
from jax_wfit.hubble import load_hubble_diagrams
from jax_wfit.run import run_wfit


@pytest.fixture(scope='module')
def bbc_reference(repo_root):
    data = repo_root / 'tests/fixtures/snana/bbc_des_beams/DATA.FITRES'
    config = parse_bbc_input(None, overrides=[f'datafile={data}', 'nzbin=10',
                                             'zmin=0.1', 'zmax=1.0', 'u1=1', 'u2=1'])
    return data, config, run_bbc_fit(config)


def test_python_distance_fit_matches_existing_and_does_not_write(bbc_reference, tmp_path):
    path, config, expected = bbc_reference
    table = read_fitres(path)
    actual = fit_distances(data=table.df, nzbin=10, zmin=.1, zmax=1.,
                           nuisance_float={'alpha0': True, 'beta0': True},
                           nuisance_onebin={'alpha0': False, 'beta0': False},
                           write_chi2grid=2, prefix=str(tmp_path / 'BBC'))
    for name in expected[0].nuisance:
        assert actual.fit.nuisance[name] == expected[0].nuisance[name]
    np.testing.assert_array_equal(actual.accepted_mask, expected[4])
    np.testing.assert_array_equal(actual.output.mu, expected[1].mu)
    assert 'MUDIF' in actual.hubble_diagram
    assert not list(tmp_path.iterdir())


def test_python_distance_fit_writing_is_explicit(bbc_reference, tmp_path):
    _, config, _ = bbc_reference
    before = asdict(config)
    result = fit_distances(config, write=True, prefix=str(tmp_path / 'BBC'),
                           write_chi2grid=2, write_yaml=True)
    assert result.fit.nsnfit > 0
    assert {'BBC.FITRES', 'BBC.M0DIF', 'BBC.COV', 'BBC.YAML', 'BBC.CHI2GRID'} <= {
        p.name for p in tmp_path.iterdir()}
    assert asdict(config) == before


def test_distance_table_file_conflict_is_rejected(bbc_reference):
    data, config, _ = bbc_reference
    with pytest.raises(SNANAError, match='file and a data table'):
        fit_distances(config, data=read_fitres(data))


def test_python_bayesn_distance_fit_matches_existing(repo_root):
    root = repo_root / 'tests/fixtures/snana'
    config = parse_bbc_input(None, overrides=[
        'model_lcfit=BAYESN', 'opt_biascor=144', 'sigint_biascor=0.10',
        'nzbin=4', 'zmin=0.05', 'zmax=0.21', 'nbin_logmass=1',
        'snrmin_sigint_biascor=6.0', 'p9=0.7', 'p11=-1', 'blindflag=0',
        f'datafile={root}/bbc_des_bayesn_data/FITRES.TEXT',
        f'simfile_biascor={root}/bbc_des_bayesn_biascor/FITRES.TEXT.gz'])
    expected = run_bbc_fit(config)
    actual = fit_distances(config)
    np.testing.assert_array_equal(actual.output.mu, expected[1].mu)
    np.testing.assert_array_equal(actual.output.m0dif, expected[1].m0dif)
    assert actual.fit.nuisance == expected[0].nuisance


def covariance_grid():
    nominal = pd.DataFrame({'ROW': [1, 2], 'zHD': [.1, .2], 'MU': [38., 40.],
                            'MUREF': [38., 40.], 'MUERR': [.1, .1]})
    shifted = nominal.copy()
    shifted['MU'] += [.02, -.03]
    return {'FITOPT000_MUOPT000': nominal, 'FITOPT001_MUOPT000': shifted}


def test_python_covariance_scales_and_writing(tmp_path):
    grid = covariance_grid()
    original = {name: frame.copy() for name, frame in grid.items()}
    options = dict(diagrams=grid, fitopt_scales={0: ('DEFAULT', 1.), 1: ('zpt', 1.)},
                   muopt_scales={0: ('DEFAULT', 1.)}, isdata_real=0,
                   cosmology={'omega_matter': .315, 'omega_lambda': .685},
                   outdir=tmp_path / 'covariance')
    result = compute_covariance(**options)
    scaled = compute_covariance(**options, sys_scales={'zpt': 2.})
    np.testing.assert_allclose(scaled.covariances['ALL']['matrix'], result.covariances['ALL']['matrix'] * 4)
    assert not list(tmp_path.iterdir())
    compute_covariance(**options, write=True)
    assert (tmp_path / 'covariance/INFO.YML').is_file()
    assert yaml.safe_load((tmp_path / 'covariance/INFO.YML').read_text())['BBC_DIR'] is None
    for name in grid:
        pd.testing.assert_frame_equal(grid[name], original[name])


def test_python_covariance_ignores_unselected_cells():
    grid = covariance_grid()
    grid['FITOPT000_MUOPT001'] = pd.DataFrame({'irrelevant': [1]})
    result = compute_covariance(diagrams=grid, muopt=0,
                                 fitopt_scales={0: ('DEFAULT', 1.), 1: ('zpt', 1.)},
                                 muopt_scales={0: ('DEFAULT', 1.), 1: ('other', 1.)})
    assert len(result.hubble_diagram) == 2


def test_python_file_covariance_preserves_results_and_suppresses_outputs(repo_root, tmp_path):
    source = repo_root / 'tests/fixtures/snana/create_cov/input.yml'
    config = covariance_config(source)
    config.outdir = tmp_path / 'unused'
    before = asdict(config)
    result = compute_covariance(config)
    assert result.covariances and len(result.hubble_diagram)
    assert result.version_photometry == compute_covariance(
        config, write=True, outdir=tmp_path / 'written').version_photometry
    assert result.version_photometry
    assert {p.name for p in tmp_path.iterdir()} == {'written'}
    assert asdict(config) == before


@pytest.fixture(scope='module')
def cosmology_reference(repo_root):
    config = WFITConfig(hubble_diagram=str(repo_root / 'tests/fixtures/wfit/SALT2mu.M0DIF'),
                         model='lcdm')
    return config, run_wfit(config, write=False)


def test_python_cosmology_file_and_memory_match(cosmology_reference, tmp_path):
    config, expected = cosmology_reference
    before = asdict(config)
    file_result = fit_cosmology(config, cospar_yaml=str(tmp_path / 'unused.yaml'))
    memory_result = fit_cosmology(diagrams=load_hubble_diagrams(config), model='lcdm')
    assert file_result[0]['point_estimate'] == expected[0]['point_estimate']
    assert memory_result[0]['point_estimate'] == expected[0]['point_estimate']
    assert not list(tmp_path.iterdir())
    assert asdict(config) == before


def test_python_cosmology_yaml_overrides_and_writing(cosmology_reference, tmp_path):
    config, _ = cosmology_reference
    source = tmp_path / 'settings.yaml'
    source.write_text(yaml.safe_dump({'hubble_diagram': config.hubble_diagram,
                                      'model': 'wcdm'}))
    result = fit_cosmology(source, model='lcdm', write=True,
                           cospar_yaml=str(tmp_path / 'fit.yaml'),
                           outfile_cospar=str(tmp_path / 'fit.cospar'))
    assert set(result[0]['point_estimate']) == {'omega_m'}
    assert (tmp_path / 'fit.yaml').is_file() and (tmp_path / 'fit.cospar').is_file()


@pytest.mark.parametrize('function', [fit_distances, compute_covariance, fit_cosmology])
def test_python_analysis_unknown_and_missing_inputs_fail(function):
    with pytest.raises(SNANAError, match='Unknown|unknown'):
        function(batch_szie=2)
    with pytest.raises(SNANAError, match='Supply|hubble_diagram is required'):
        function()


def test_python_covariance_missing_metadata_fails():
    with pytest.raises(SNANAError, match='fitopt_scales and muopt_scales'):
        compute_covariance(diagrams=covariance_grid())
    with pytest.raises(SNANAError, match='outdir'):
        compute_covariance(write=True)


def test_python_cosmology_conflicts_and_empty_inputs_fail(cosmology_reference):
    config, _ = cosmology_reference
    diagrams = load_hubble_diagrams(config)
    with pytest.raises(SNANAError, match='not both'):
        fit_cosmology(config, diagrams=diagrams)
    with pytest.raises(SNANAError, match='At least one'):
        fit_cosmology(diagrams=[])
    with pytest.raises(SNANAError, match='hubble_diagram is required'):
        fit_cosmology(hubble_diagram=[])
    with pytest.raises(SNANAError, match='cospar_yaml and outfile_cospar'):
        fit_cosmology(diagrams=diagrams, write=True)
