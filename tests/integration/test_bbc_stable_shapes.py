"""Known BBC grids share numerical capacities without changing their science."""
from dataclasses import replace

from jax_snlc_core.io.fits_path import open_text

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_salt2mu import map_kernels, preparation
from jax_salt2mu.config import BBCConfig
from jax_salt2mu.fitter import _fit, _fit_derivatives, _chi2_ia, pad_ccprior, padded
from jax_salt2mu.run import prepare_cells, fit_spec, finish_grid
from jax_salt2mu.tablevar import EVENT_TYPE_DATA, EVENT_TYPE_BIASCOR, EVENT_TYPE_CCPRIOR
from jax_snlc_core.io.fitres_reader import read_fitres
from jax_snlc_core.models import standardised_model
from jax_snlc_core.utils.error_handling import SNANAError


@pytest.mark.parametrize('model,opt', [('SALT2', 112), ('SALT2', 240),
                                      ('SALT2', 4208), ('BAYESN', 176)])
def test_map_padding_preserves_values_and_compilation(model, opt):
    if model == 'BAYESN':
        data = read_fitres('tests/fixtures/snana/bbc_des_bayesn_data/FITRES.TEXT')
        biascor = read_fitres('tests/fixtures/snana/bbc_des_bayesn_biascor/FITRES.TEXT.gz')
        cfg = BBCConfig(model_lcfit=model, nzbin=4, zmin=.05, zmax=.21, opt_biascor=opt,
                        sigint_biascor=.1, nbin_logmass=1, snrmin_sigint_biascor=6.,
                        surveygroup_biascor='DES(zbin=0.04)')
        upper = .13
    else:
        root = 'tests/fixtures/snana/bbc_des_beams'
        data, biascor = (read_fitres(f'{root}/{name}.FITRES') for name in ('DATA', 'BIASCOR'))
        cfg = BBCConfig(nzbin=10, zmin=.1, zmax=1., opt_biascor=opt,
                        min_per_cell_biascor=3, snrmin_sigint_biascor=60.,
                        surveygroup_biascor='DES(zbin=0.15:x1bin=2.0:cbin=0.2)')
        upper = .5
    shorter = replace(data, df=data.df.loc[data.df.zHD < upper].reset_index(drop=True))
    tables = [{EVENT_TYPE_DATA: table, EVENT_TYPE_BIASCOR: biascor} for table in (data, shorter)]
    cells = preparation.read_cells({key: (cfg, source) for key, source in enumerate(tables)})
    before_fit = map_kernels.fit_map._cache_size()
    before_error = map_kernels.error_map._cache_size()
    maps = preparation.prepare_biascor_maps(cells)
    assert map_kernels.fit_map._cache_size() - before_fit <= 1
    assert map_kernels.error_map._cache_size() - before_error <= 1
    assert maps[0].maps[0].grid.dims != maps[1].maps[0].grid.dims
    for key in cells:
        reference = preparation.prepare_biascor_maps({key: cells[key]})[key]
        for actual, expected in zip(maps[key].maps, reference.maps):
            assert actual.grid.dims == expected.grid.dims
            np.testing.assert_array_equal(actual.fitparbias.npercell, expected.fitparbias.npercell)
            for name in ('val', 'err', 'rms'):
                np.testing.assert_allclose(getattr(actual.fitparbias, name),
                                           getattr(expected.fitparbias, name), rtol=1e-12, atol=1e-12)
            for a, b in zip(actual.avg, expected.avg):
                np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
        for actual, expected in zip(maps[key].mucov_maps, reference.mucov_maps):
            if actual is None:
                assert expected is None
                continue
            np.testing.assert_array_equal(actual.use, expected.use)
            np.testing.assert_allclose(actual.scale, expected.scale, rtol=1e-12, atol=1e-12)
            if actual.covadd is not None:
                np.testing.assert_allclose(actual.covadd, expected.covadd, rtol=1e-12, atol=1e-12)
        numeric = preparation.prepare_data({key: cells[key]})
        correction = preparation.interpolate_corrections({key: cells[key]}, numeric, maps)
        independent = preparation.interpolate_corrections({key: cells[key]}, numeric, {key: reference})
        a = prepare_cells({key: cells[key]}, numeric, maps, correction, {key: None})[key]
        b = prepare_cells({key: cells[key]}, numeric, {key: reference}, independent, {key: None})[key]
        np.testing.assert_array_equal(a.accept, b.accept)
        np.testing.assert_array_equal(a.tv.cutbits, b.tv.cutbits)


def test_fit_padding_preserves_results_and_compilation():
    from jax_salt2mu.results import _output_parts
    data = read_fitres('tests/fixtures/snana/bbc_des_beams/DATA.FITRES')
    cfg = BBCConfig(nzbin=4, zmin=.1, zmax=1., sigint_fix=(.1,))
    tables = [data, replace(data, df=data.df.iloc[::2].reset_index(drop=True))]
    cells = preparation.read_cells({key: (cfg, {EVENT_TYPE_DATA: table})
                                    for key, table in enumerate(tables)})
    cosmology_widths = {standardised_model('SALT2'): data.size}
    numeric = preparation.prepare_data(cells, cosmology_widths=cosmology_widths)
    preparation.format_data(cells, numeric)
    maps = {key: None for key in cells}
    prepared = prepare_cells(cells, numeric, maps, maps, maps)
    assert prepared[0].spec == prepared[1].spec
    spec = fit_spec(cells[0][0], prepared[0].model)
    assert spec == prepared[0].spec
    width = max(pair[0].size for pair in numeric.values())
    first = finish_grid({0: prepared[0]}, fit_widths={spec: width}, write=False)[0]
    caches = [kernel._cache_size() for kernel in (_fit, _fit_derivatives, _chi2_ia, _output_parts)]
    second = finish_grid({1: prepared[1]}, fit_widths={spec: width}, write=False)[1]
    assert caches == [kernel._cache_size() for kernel in (_fit, _fit_derivatives, _chi2_ia, _output_parts)]
    reference = finish_grid({1: prepared[1]}, write=False)[1]
    np.testing.assert_allclose(second[1].mu, reference[1].mu, rtol=0., atol=1e-11)
    np.testing.assert_allclose(second[0].m0, reference[0].m0, rtol=0., atol=1e-11)
    assert first[0].nsnfit > second[0].nsnfit
    with pytest.raises(SNANAError, match='capacity'):
        padded(prepared[0].inputs, 1)


@pytest.fixture(scope='module')
def cc_cell():
    from jax_salt2mu.run import prepare_cell
    root = 'tests/fixtures/snana/bbc_des_beams'
    return prepare_cell(BBCConfig(
        datafile=[f'{root}/CONTAM_DATA.FITRES'], simfile_biascor=[f'{root}/BIASCOR.FITRES'],
        simfile_ccprior=[f'{root}/CCPRIOR.FITRES'], nzbin=10, zmin=.1, zmax=1.,
        opt_biascor=112, min_per_cell_biascor=3, snrmin_sigint_biascor=60.,
        surveygroup_biascor='DES(zbin=0.15:x1bin=2.0:cbin=0.2)', blindflag=0,
        nuisance_float={'alpha0': True, 'beta0': True},
        varname_pia='PROBIA', typeIa_ccprior=999, nuisance_val={'scalePCC': 1.}))


@pytest.mark.parametrize('empty', [False, True])
@pytest.mark.parametrize('joint', [False, True])
def test_cc_prior_padding_preserves_value_gradient_and_hessian(cc_cell, empty, joint):
    base = cc_cell.inputs
    prior = base.ccprior
    if empty:
        prior = prior._replace(
            cc_z=prior.cc_z[:0], cc_logmass=prior.cc_logmass[:0],
            cc_mumodel=prior.cc_mumodel[:0], cc_fitparbias_nodes=prior.cc_fitparbias_nodes[:0],
            cc_cell=prior.cc_cell[:0], cc_valid=prior.cc_valid[:0],
            cc_lcpar={k: v[:0] for k, v in prior.cc_lcpar.items()})
    inp = base._replace(ccprior=prior)
    extra = inp._replace(ccprior=pad_ccprior(prior, len(prior.cc_z) + 19, base.ccprior))
    spec = replace(cc_cell.spec, mode='joint' if joint else 'snana_secant')
    point = jnp.array([dict(spec.defaults)[n] for n in spec.floated]
                      + [cc_cell.model.m0_reference] * spec.nzbin + ([.1] if joint else []))

    def evaluate(settings, inputs):
        loss = settings.closed(inputs, .1, inputs.muerrsq_last)
        return loss(point), jax.grad(loss)(point), jax.hessian(loss)(point)

    expected = evaluate(spec, padded(inp, len(inp.z)))
    actual = evaluate(replace(spec, cc_ncell=spec.cc_ncell + 17), padded(extra, len(inp.z) + 23))
    for a, b in zip(actual, expected):
        assert np.isfinite(a).all() and np.isfinite(b).all()
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-9)


def test_numeric_preparation_and_cosmology_padding():
    from jax_snlc_core.physics.cosmology import distance_modulus
    data = read_fitres('tests/fixtures/snana/bbc_des_beams/DATA.FITRES')
    cfg = BBCConfig(nzbin=4, zmin=.1, zmax=1.)
    inputs = [data, replace(data, df=data.df.iloc[::2].reset_index(drop=True))]
    for table in inputs:
        cells = preparation.read_cells({0: (cfg, {EVENT_TYPE_DATA: table})})
        numeric = preparation.prepare_data(cells, cosmology_widths={standardised_model('SALT2'): data.size})
        assert not numeric[0][0].passthrough
        if table is data:
            compiled = distance_modulus._cache_size()
        else:
            assert distance_modulus._cache_size() == compiled
        ordinary = preparation.prepare_data(cells)
        np.testing.assert_array_equal(numeric[0][0].mumodel, ordinary[0][0].mumodel)
        preparation.format_data(cells, numeric)
        assert 'CID' in numeric[0][0].passthrough
        assert not numeric[0][0].passthrough['CID'].flags.writeable


def test_prepared_cc_priors_share_capacity_with_an_empty_partition(cc_cell):
    root = 'tests/fixtures/snana/bbc_des_beams'
    tables = {kind: read_fitres(f'{root}/{name}.FITRES') for kind, name in
              ((EVENT_TYPE_DATA, 'CONTAM_DATA'), (EVENT_TYPE_BIASCOR, 'BIASCOR'),
               (EVENT_TYPE_CCPRIOR, 'CCPRIOR'))}
    config = replace(cc_cell.config, datafile=[], simfile_biascor=[], simfile_ccprior=[])
    prior = tables[EVENT_TYPE_CCPRIOR]
    outside = replace(prior, df=prior.df.assign(zHD=2., zCMB=2.))
    cells = preparation.read_cells({0: (config, tables),
                                   1: (config, {**tables, EVENT_TYPE_CCPRIOR: outside})})
    maps = preparation.prepare_biascor_maps(cells)
    priors = preparation.prepare_ccpriors(cells, maps)
    assert len(priors[0][0].cc_z) > 0
    assert len(priors[0][0].cc_z) == len(priors[1][0].cc_z)
    assert priors[0][1] == priors[1][1]
    assert np.sum(priors[1][0].cc_valid) == 0
    assert priors[1][2][0] == 0
    for array in jax.tree.leaves(priors[1][0]):
        assert np.isfinite(array).all()


def test_sync_row_bound_includes_duplicate_rows_and_independent_fitopts():
    from jax_pipeline.synchronise import synchronised_row_counts
    data = read_fitres('tests/fixtures/snana/bbc_des_beams/DATA.FITRES')
    data = replace(data, df=data.df.iloc[:100].reset_index(drop=True))
    from jax_snlc_core.io.fitres_reader import concat_tables
    data = concat_tables([data, replace(data, df=data.df.iloc[:1].copy())])
    config = BBCConfig(nzbin=4, zmin=.1, zmax=1., force_realdata=True,
                       blindflag=0, iflag_duplicate=2)
    cells = preparation.read_cells({cell: (config, {EVENT_TYPE_DATA: data})
                                    for cell in ((0, 0), (0, 1), (1, 0), (2, 0))})
    numeric = preparation.prepare_data(cells)
    counts = synchronised_row_counts(cells, numeric, ['[independent] NOREJECT', '[shared]'])
    ordinary = int(preparation.select_data(cells[0, 0][0], cells[0, 0][1], numeric[0, 0], None).cutmask.sum())
    assert counts[0, 0] == ordinary + 1
    assert counts[0, 1] == counts[2, 0] == counts[0, 0]
    assert counts[1, 0] == ordinary


@pytest.mark.parametrize('event_sync', [False, True])
def test_realisations_share_compilation(tmp_path, event_sync):
    from pathlib import Path
    import shutil
    import yaml
    from jax_pipeline import BBC, Fit, run
    source = Path('tests/fixtures/snana/bbc_des_beams/DATA.FITRES')
    table = read_fitres(str(source))
    names = set(table.df.loc[table.df.zHD < .5, 'CID'].astype(str))
    shorter = tmp_path / 'shorter.FITRES'
    with open_text(source) as stream:
        fixture_text = stream.read()
    shorter.write_text(''.join(line for line in fixture_text.splitlines(keepends=True)
                               if not line.startswith('SN:') or line.split()[1] in names))
    correction = Fit('biascor', fitres=Path('tests/fixtures/snana/bbc_des_beams/BIASCOR.FITRES'))
    options = {'nzbin': 4, 'zmin': .1, 'zmax': 1., 'u1': 1, 'u2': 1,
               'opt_biascor': 112, 'min_per_cell_biascor': 3, 'snrmin_sigint_biascor': 60.,
               'surveygroup_biascor': 'DES(zbin=0.15:x1bin=2.0:cbin=0.2)'}
    stages = []
    for i, path in enumerate((source, shorter)):
        version = tmp_path / f'imported{i}' / 'VERSION'
        version.mkdir(parents=True)
        with open_text(path) as source, (version / 'FITOPT000.FITRES').open("w") as target:
            shutil.copyfileobj(source, target)
        (version.parent / 'SUBMIT.INFO').write_text(yaml.safe_dump({
            'FITOPT_LIST': [['FITOPT000', None, None]], 'OPT_SNCID_LIST': int(event_sync)}))
        stages.append(BBC(f'bbc{i}', data=Fit(f'data{i}', fitres=version),
                          biascor=correction, overrides=options))
    kernels = (map_kernels.fit_map, map_kernels.error_map, _fit, _fit_derivatives)
    before = [kernel._cache_size() for kernel in kernels]
    outputs = run(*stages, outdir=tmp_path / 'pipeline')
    after = [kernel._cache_size() for kernel in kernels]
    assert all(b - a <= 1 for a, b in zip(before, after))
    assert len(outputs['bbc0']['nominal']['FITRES'].df) > len(outputs['bbc1']['nominal']['FITRES'].df)
    for stage in stages:
        assert (tmp_path / 'pipeline' / stage.name / 'BBC_ACCEPT_SUMMARY.LIST').exists() == event_sync
