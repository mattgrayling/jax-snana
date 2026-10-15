"""Acceptance-only synchronisation, final-fit parity and cache reuse."""
from pathlib import Path
from jax_snlc_core.io.fits_path import open_text

import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

from jax_pipeline import BBC, Fit, run
from jax_salt2mu import parse_bbc_input
from jax_salt2mu.run import run_bbc_fit, run_prepared_grid
from jax_salt2mu.tablevar import EVENT_TYPE_DATA
from jax_snlc_core.io.fitres_reader import read_fitres


def _two_pass_reference(grid, maps, priors, fitopts, muopts, outdir, *, t_start=None):
    """Original synchronisation path: fit pass 1 before taking its acceptance."""
    from jax_pipeline.synchronise import _selected, _write_selection

    def events(fitted):
        _, output, _, tv, accept = fitted
        fields = tv.field if tv.field is not None else np.full(tv.size, 'NULL')
        identities = zip(tv.name[accept], tv.idsurvey[accept], fields[accept])
        return {(str(cid), int(survey), str(field)): int(zbin)
                for (cid, survey, field), zbin in zip(identities, output.izbin)}

    initial = run_prepared_grid({(0, 0): grid[(0, 0)]}, maps, priors)
    reference = events(initial[(0, 0)])
    reference_file = Path(outdir) / 'REFERENCE.LIST'
    _write_selection(reference_file, reference)

    def independent(cell):
        return cell[0] > 0 and 'NOREJECT' in fitopts[cell[0] - 1]

    others = {cell: _selected(value, reference_file) if cell[0] and not independent(cell)
              else value for cell, value in grid.items() if cell != (0, 0)}
    if others:
        initial.update(run_prepared_grid(others, maps, priors))
    common = set(reference)
    for cell, fitted in initial.items():
        exempt = cell[1] > 0 and 'NOREJECT' in muopts[cell[1] - 1]
        if not independent(cell) and not exempt:
            common.intersection_update(events(fitted))
    selected = Path(outdir) / 'BBC_ACCEPT_SUMMARY.LIST'
    _write_selection(selected, {event: zbin for event, zbin in reference.items() if event in common})
    return run_prepared_grid({cell: value if independent(cell) else _selected(value, selected)
                         for cell, value in grid.items()}, maps, priors)


@pytest.mark.parametrize('model,source,cut,chi2max', [
    ('SALT2', 'bbc_des_beams/DATA.FITRES', 'CUTWIN c -1 0', None),
    ('SALT2', 'bbc_des_beams/DATA.FITRES', 'CUTWIN c -1 0', 8.),
    ('BAYESN', 'bbc_des_bayesn_data/FITRES.TEXT', 'CUTWIN THETA -10 0', None),
])
def test_common_sample_matches_two_pass_fits_and_loads_from_cache(
        tmp_path, model, source, cut, chi2max):
    version = tmp_path / 'imported' / 'VERSION'
    version.mkdir(parents=True)
    data = Path('tests/fixtures/snana') / source
    with open_text(data) as source, (version / 'FITOPT000.FITRES').open("w") as target:
        shutil.copyfileobj(source, target)
    (version.parent / 'SUBMIT.INFO').write_text(yaml.safe_dump({
        'FITOPT_LIST': [['FITOPT000', None, None]], 'OPT_SNCID_LIST': 1}))
    fit = Fit('fit', fitres=version)
    keys = {'model_lcfit': model, 'nzbin': 4, 'zmin': 0.1, 'zmax': 1., 'blindflag': 0}
    if model == 'SALT2':
        keys.update(u1=1, u2=1)
    else:
        keys.update(zmin=0.05, zmax=0.21)
    if chi2max is not None:
        keys['chi2max'] = chi2max
    _, column, lo, hi = cut.split()
    bbc = BBC('bbc', data=fit, overrides=keys, muopts={'cut': {f'CUTWIN {column}': f'{lo} {hi}'}})
    root = tmp_path / 'run'
    outputs = run(bbc, outdir=root)
    results = outputs['bbc']
    first = read_fitres(str(root / 'bbc/initial/FITOPT000_MUOPT000/ACCEPT.LIST'))
    selected = root / 'bbc/BBC_ACCEPT_SUMMARY.LIST'
    common = read_fitres(str(selected))
    assert 0 < len(common.df) < len(first.df)
    for output in results.values():
        assert set(output['FITRES'].df.CID) == set(common.df.CID)
    assert not list((root / 'bbc/initial').rglob('*.FITRES'))
    assert not list((root / 'bbc/initial').rglob('*.M0DIF'))
    # Compare acceptance and serialized outputs with two fitted passes.
    from jax_pipeline.stages import prepare_bbc_stages
    from jax_pipeline.systematics import dirname
    from jax_salt2mu.run import write_outputs
    legacy_root = tmp_path / 'two_pass'
    legacy_dir = legacy_root / 'bbc'
    legacy_dir.mkdir(parents=True)
    inputs = {role: outputs[used.name] for role, used in bbc.inputs().items()}
    cells, maps, priors, fit_widths, cosmology_widths, started = prepare_bbc_stages({bbc: (inputs, legacy_dir)})
    grid = {cell: cells['bbc', cell] for cell in bbc.grid_cells()}
    grid_maps = {cell: maps['bbc', cell] for cell in grid}
    grid_priors = {cell: priors['bbc', cell] for cell in grid}
    fitted = _two_pass_reference(grid, grid_maps, grid_priors, *bbc.systematics(), legacy_dir)
    for cell, values in fitted.items():
        cell_dir = legacy_dir / dirname(cell)
        cell_dir.mkdir(parents=True)
        write_outputs(cell_dir / 'bbc.FITRES', *values)
    legacy = bbc.load(inputs, legacy_dir)
    assert selected.read_bytes() == (legacy_root / 'bbc/BBC_ACCEPT_SUMMARY.LIST').read_bytes()
    for cell in results:
        for table in ('FITRES', 'M0DIF'):
            pd.testing.assert_frame_equal(results[cell][table].df, legacy[cell][table].df,
                                          check_exact=False, rtol=0., atol=5.1e-6)
    # Independently run BBC with the common-event list and SNANA's CUTWIN NONE.
    config = parse_bbc_input(None, overrides=[f'{k}={v}' for k,v in keys.items()] +
                             [f'cid_select_file={selected}', 'CUTWIN NONE'])
    direct = run_bbc_fit(config, {EVENT_TYPE_DATA: read_fitres(str(data))})[1]
    np.testing.assert_allclose(results['nominal']['M0DIF'].df.MUDIF, direct.m0dif, atol=5.1e-6, rtol=0)
    again = run(bbc, outdir=root)['bbc']
    assert yaml.safe_load((root / 'MANIFEST.yml').read_text())['stages']['bbc']['loaded']
    np.testing.assert_array_equal(results['nominal']['M0DIF'].df.MUDIF, again['nominal']['M0DIF'].df.MUDIF)
    # The inherited setting participates in caching even though SUBMIT.INFO
    # lives alongside, rather than inside, the imported fit version.
    info = version.parent / 'SUBMIT.INFO'
    metadata = yaml.safe_load(info.read_text())
    metadata['OPT_SNCID_LIST'] = 0
    info.write_text(yaml.safe_dump(metadata))
    independent = run(bbc, outdir=root)['bbc']
    assert not yaml.safe_load((root / 'MANIFEST.yml').read_text())['stages']['bbc']['loaded']
    assert len(independent['nominal']['FITRES'].df) == len(first.df)



def test_fitopt_sync_is_separate_for_each_realisation_and_split(tmp_path):
    from jax_snlc_core.io.snana_table import write_varnames_table
    table = read_fitres('tests/fixtures/snana/bbc_des_beams/DATA.FITRES').df
    stages = []
    for realisation in (1, 2):
        version = tmp_path / f'import{realisation}' / 'VERSION'
        version.mkdir(parents=True)
        for i, shift in enumerate((0., 0.03)):
            frame = table.assign(c=table.c + shift + (realisation - 1) * 0.04)
            write_varnames_table(str(version / f'FITOPT{i:03d}.FITRES'), list(frame),
                                 {c: frame[c].to_numpy() for c in frame}, fmt={})
        (version.parent / 'SUBMIT.INFO').write_text(yaml.safe_dump({
            'FITOPT_LIST': [['FITOPT000', None, None], ['FITOPT001', 'CAL', None]],
            'FLAG_USE_SAME_EVENTS': 1}))
        source = Fit(f'fit{realisation}', fitres=version)
        for partition in (1, 2):
            stages.append(BBC(f'bbc{realisation}_split{partition}', data=source,
                realisation=realisation, split_count=2, split_index=partition,
                fitopt_muopt=['0+0'],
                overrides={'nzbin': 4, 'zmin': 0.1, 'zmax': 1., 'u1': 1, 'u2': 1, 'blindflag': 0},
                muopts={'cut': {'CUTWIN c': '-1 0'}}))
    root = tmp_path / 'run'
    results = run(*stages, outdir=root)
    selections = []
    for stage in stages:
        cells = results[stage.name]
        identities = [set(output['FITRES'].df.CID) for output in cells.values()]
        assert identities[0] and all(ids == identities[0] for ids in identities)
        assert all(int(cid) % 2 + 1 == stage.split_index for cid in identities[0])
        selections.append(identities[0])
        initial = root / stage.name / 'initial'
        nominal = read_fitres(str(initial / 'FITOPT000_MUOPT000/ACCEPT.LIST')).df
        variant = read_fitres(str(initial / 'FITOPT001_MUOPT000/ACCEPT.LIST')).df
        assert set(variant.CID) <= set(nominal.CID)
    assert not selections[0] & selections[1]
    assert not selections[2] & selections[3]
    assert selections[0] != selections[2]
