"""Shared preparation preserves independent cell selections and source changes."""

from _bbc_helpers import prepare_cell
from dataclasses import replace

import numpy as np
import pytest

from jax_salt2mu import preparation, run as driver
from jax_salt2mu.config import BBCConfig
from jax_salt2mu.tablevar import EVENT_TYPE_BIASCOR, EVENT_TYPE_CCPRIOR, EVENT_TYPE_DATA
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import read_fitres


@pytest.mark.parametrize('opt', [112, 4208])
def test_selection_reuses_numerical_work_with_independent_cuts(tmp_path, opt):
    root = 'tests/fixtures/snana/bbc_des_beams'
    tables = {EVENT_TYPE_DATA: read_fitres(f'{root}/DATA.FITRES'),
              EVENT_TYPE_BIASCOR: read_fitres(f'{root}/BIASCOR.FITRES')}
    config = BBCConfig(nzbin=10, zmin=.1, zmax=1., opt_biascor=opt,
                       min_per_cell_biascor=3, snrmin_sigint_biascor=60.,
                       surveygroup_biascor='DES(zbin=0.15:x1bin=2.0:cbin=0.2)')
    cells = preparation.read_cells({0: (config, tables)})
    maps = preparation.prepare_biascor_maps(cells)
    priors = preparation.prepare_ccpriors(cells, maps)
    data = preparation.prepare_data(cells)
    preparation.format_data(cells, data)
    biascor = preparation.interpolate_corrections(cells, data, maps)
    nominal = driver.prepare_cells(cells, data, maps, biascor, priors)[0]
    original_bits = nominal.tv.cutbits.copy()
    selected = tmp_path / 'selected.list'
    selected.write_text('\n'.join(nominal.tv.name[nominal.accept][::2]) + '\n')
    cfg, sources, text = cells[0]
    selected_cells = {0: (replace(cfg, cid_select_file=[str(selected)], cutwin_disable=True), sources, text)}
    reused = driver.prepare_cells(selected_cells, data, maps, biascor, priors)[0]
    independent = prepare_cell(replace(config, cid_select_file=[str(selected)], cutwin_disable=True), tables)
    assert reused.maps is nominal.maps
    assert reused.tv is not nominal.tv
    assert reused.tv.covmat_fit is nominal.tv.covmat_fit
    assert reused.tv.passthrough is not nominal.tv.passthrough
    assert reused.tv.passthrough['CID'] is nominal.tv.passthrough['CID']
    assert not reused.tv.passthrough['CID'].flags.writeable
    assert reused.accept.sum() < nominal.accept.sum()
    np.testing.assert_array_equal(nominal.tv.cutbits, original_bits)
    np.testing.assert_array_equal(reused.accept, independent.accept)
    import jax
    shared, shared_tree = jax.tree_util.tree_flatten(reused.inputs)
    fresh, fresh_tree = jax.tree_util.tree_flatten(independent.inputs)
    assert shared_tree == fresh_tree
    for a, b in zip(shared, fresh):
        np.testing.assert_array_equal(a, b)


def test_data_files_share_one_source():
    config = BBCConfig(datafile=['tests/fixtures/snana/bbc_des_beams/DATA.FITRES'],
                       nzbin=10, zmin=.1, zmax=1.)
    cells = preparation.read_cells({0: (config, None), 1: (replace(config, cutwin_disable=True), None)})
    assert cells[0][1][EVENT_TYPE_DATA] is cells[1][1][EVENT_TYPE_DATA]
    data = preparation.prepare_data(cells)
    preparation.format_data(cells, data)
    selected = [preparation.select_data(cfg, tables, data[key], None)
                for key, (cfg, tables, _) in cells.items()]
    assert selected[0].passthrough['CID'] is selected[1].passthrough['CID']
    assert not np.shares_memory(selected[0].cutbits, selected[1].cutbits)


@pytest.mark.parametrize('change', [
    {'redchi2_tol': .01}, {'chi2max': 20.}, {'nmax': 500},
    {'sigint_fix': (.12,)}, {'prefix': 'different', 'write_yaml': True},
    {'zphot_shift': .01}, {'surveylist_nobiascor': ['DES']},
])
def test_data_and_fit_changes_preserve_map_reuse_and_fresh_results(change):
    root = 'tests/fixtures/snana/bbc_des_beams'
    tables = {EVENT_TYPE_DATA: read_fitres(f'{root}/DATA.FITRES'),
              EVENT_TYPE_BIASCOR: read_fitres(f'{root}/BIASCOR.FITRES')}
    config = BBCConfig(nzbin=10, zmin=.1, zmax=1., opt_biascor=112,
                       min_per_cell_biascor=3, snrmin_sigint_biascor=60.,
                       surveygroup_biascor='DES(zbin=0.15:x1bin=2.0:cbin=0.2)')
    changed = replace(config, **change)
    cells = preparation.read_cells({0: (config, tables), 1: (changed, tables)})
    maps = preparation.prepare_biascor_maps(cells)
    assert maps[0] is maps[1]
    data = preparation.prepare_data(cells)
    preparation.format_data(cells, data)
    nodes = preparation.interpolate_corrections(cells, data, maps)
    prepared = driver.prepare_cells(cells, data, maps, nodes, {0: None, 1: None})[1]
    independent = prepare_cell(changed, tables)
    np.testing.assert_array_equal(prepared.accept, independent.accept)
    np.testing.assert_array_equal(prepared.tv.cutbits, independent.tv.cutbits)
    import jax
    a, atree = jax.tree_util.tree_flatten(prepared.inputs)
    b, btree = jax.tree_util.tree_flatten(independent.inputs)
    assert atree == btree
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize('change', [
    {'sigint_biascor': .15}, {'prescale_biascor': (0, 2)},
    {'lensing_zpar': .07}, {'zpecerr': .001}, {'nzbin': 12},
    {'lcpar_range': {'c': (-.2, .2)}}, {'nuisance_val': {'alpha0': .16}},
    {'nuisance_float': {'gamma0': True}}, {'nuisance_onebin': {'alpha0': False}},
])
def test_correction_settings_remain_dependencies(change):
    config = BBCConfig()
    assert preparation.correction_config(config) != preparation.correction_config(replace(config, **change))


def test_diagnostic_prefix_and_precision_remain_dependencies():
    import jax
    config = BBCConfig(write_biascor=True, prefix='first')
    assert preparation.correction_config(config) != preparation.correction_config(replace(config, prefix='second'))
    with jax.enable_x64(True):
        double = preparation.correction_config(config)
    with jax.enable_x64(False):
        single = preparation.correction_config(config)
    assert double != single


@pytest.mark.parametrize('prior', ['none', 'fixed', 'floated'])
def test_supplied_tables_match_file_inputs_without_changing_filenames(prior):
    import jax

    root = 'tests/fixtures/snana/bbc_des_beams'
    config = BBCConfig(nzbin=10, zmin=.1, zmax=1., min_per_cell_biascor=3,
                       snrmin_sigint_biascor=60.,
                       surveygroup_biascor='DES(zbin=0.15:x1bin=2.0:cbin=0.2)',
                       nuisance_float={'alpha0': True, 'beta0': True})
    if prior != 'none':
        config = replace(config, varname_pia='PROBIA', typeIa_ccprior=999,
                         nuisance_val={'scalePCC': 1.})
    if prior == 'floated':
        config = replace(config, nuisance_float={**config.nuisance_float, 'scalePCC': True})
    data_path = f'{root}/{"DATA" if prior == "none" else "CONTAM_DATA"}.FITRES'
    biascor_path = f'{root}/BIASCOR.FITRES'
    ccprior_path = f'{root}/CCPRIOR.FITRES'
    files = replace(config, datafile=[data_path], simfile_biascor=[biascor_path],
                    simfile_ccprior=[ccprior_path] if prior != 'none' else [])
    tables = {EVENT_TYPE_DATA: read_fitres(data_path),
              EVENT_TYPE_BIASCOR: read_fitres(biascor_path)}
    if prior != 'none':
        tables[EVENT_TYPE_CCPRIOR] = read_fitres(ccprior_path)
    supplied = prepare_cell(config, tables)
    from_files = prepare_cell(files)
    assert supplied.config.datafile == []
    assert supplied.config.simfile_biascor == []
    assert supplied.config.simfile_ccprior == []
    assert config.opt_biascor == 0
    assert supplied.config.opt_biascor == from_files.config.opt_biascor
    assert supplied.spec == from_files.spec
    np.testing.assert_array_equal(supplied.accept, from_files.accept)
    np.testing.assert_array_equal(supplied.tv.cutbits, from_files.tv.cutbits)
    a, atree = jax.tree_util.tree_flatten(supplied.inputs)
    b, btree = jax.tree_util.tree_flatten(from_files.inputs)
    assert atree == btree
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_fixed_zero_cc_scale_ignores_supplied_prior_without_biascor():
    config = BBCConfig(nzbin=10, zmin=.1, zmax=1.,
                       nuisance_float={'scalePCC': False}, nuisance_val={'scalePCC': 0.})
    tables = {EVENT_TYPE_DATA: read_fitres('tests/fixtures/snana/bbc_des_beams/DATA.FITRES'),
              EVENT_TYPE_CCPRIOR: object()}
    cell = prepare_cell(config, tables)
    assert cell.inputs.ccprior is None
    assert 'scalePCC' not in cell.spec.floated
    assert cell.config.simfile_ccprior == []
    assert EVENT_TYPE_CCPRIOR in tables


@pytest.mark.parametrize('opt,with_biascor,with_ccprior,message', [
    (8, True, False, 'opt_biascor bit 8'),
    (1, True, True, 'cannot use a 1D biasCor'),
    (0, False, True, 'CC prior reads its bias from the 5D'),
])
def test_supplied_corrections_are_validated_before_reading_data(
        opt, with_biascor, with_ccprior, message):
    config = BBCConfig(datafile=['must-not-read.FITRES'], nzbin=10, zmin=.1, zmax=1.,
                       opt_biascor=opt, nuisance_val={'scalePCC': 1.})
    tables = {}
    if with_biascor:
        tables[EVENT_TYPE_BIASCOR] = object()
    if with_ccprior:
        tables[EVENT_TYPE_CCPRIOR] = object()
    with pytest.raises(SNANAError, match=message):
        preparation.read_cells({0: (config, tables)})


def test_maps_release_converted_source_arrays():
    import gc
    import weakref
    from jax_salt2mu.biascor import build_biascor_partitions
    from jax_salt2mu.config import resolve
    from jax_salt2mu.samples import partition_idsamples
    from _bbc_helpers import read_tablevar
    from jax_snlc_core.models import SALTModel

    path = 'tests/fixtures/snana/bbc_des_beams/BIASCOR.FITRES'
    config = resolve(BBCConfig(nzbin=10, zmin=.1, zmax=1., opt_biascor=112,
                     simfile_biascor=[path], min_per_cell_biascor=3,
                     snrmin_sigint_biascor=60.,
                     surveygroup_biascor='DES(zbin=0.15:x1bin=2.0:cbin=0.2)'), SALTModel)
    tv = read_tablevar([path], EVENT_TYPE_BIASCOR, config, SALTModel)
    _, samples = partition_idsamples(tv.idsurvey, tv.field, tv.zhd, tv.is_photoz, config)
    references = [weakref.ref(value) for value in (tv, tv.zhd, tv.covmat_fit)]
    maps = build_biascor_partitions(tv, {0: samples}, config, SALTModel, False)[0]
    del tv
    gc.collect()
    assert all(reference() is None for reference in references)
    assert maps.biascor_counts[0] > 0
    assert maps.maps[0].fitparbias.val.size > 0


def test_reported_elapsed_time_includes_shared_preparation():
    import time
    root = 'tests/fixtures/snana/bbc_des_beams'
    tables = {EVENT_TYPE_DATA: read_fitres(f'{root}/DATA.FITRES'),
              EVENT_TYPE_BIASCOR: read_fitres(f'{root}/BIASCOR.FITRES')}
    config = BBCConfig(nzbin=10, zmin=.1, zmax=1., opt_biascor=112,
                       min_per_cell_biascor=3, snrmin_sigint_biascor=60.,
                       surveygroup_biascor='DES(zbin=0.15:x1bin=2.0:cbin=0.2)',
                       nuisance_float={'alpha0': True, 'beta0': True})
    started = time.time()
    _, output, _, _, _ = driver.run_bbc_fit(config, tables, write=False)
    elapsed = time.time() - started
    assert elapsed - .02 <= output.yaml.cpu_minutes * 60. <= elapsed
