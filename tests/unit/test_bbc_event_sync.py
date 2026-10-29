"""Identity matching and SNANA's two-pass selection rules."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from jax_pipeline.synchronise import fit_prepared_common_sample
from jax_salt2mu.preparation import read_cells, prepare_biascor_maps, prepare_ccpriors
from jax_salt2mu.config import BBCConfig
from jax_salt2mu.tablevar import EVENT_TYPE_DATA, match_cid_selection, read_cid_selection
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import read_fitres


def test_selection_distinguishes_surveys_and_fields_and_preserves_bins(tmp_path):
    selected = tmp_path / "accepted.list"
    selected.write_text("VARNAMES: CID IDSURVEY FIELD IZBIN\nSN: 001 10 A 2\nSN: 001 20 A 3\n")
    tv = SimpleNamespace(name=np.array(['001'] * 4), idsurvey=np.array([10, 10, 20, 30]),
                         field=np.array(['A', 'B', 'A', 'A']))
    keep, bins = match_cid_selection(tv, [selected])
    assert keep.tolist() == [True, False, True, False]
    assert bins.tolist() == [2, -9, 3, -9]
    selected.write_text('001\n')
    assert match_cid_selection(tv, [selected])[0].all()
    selected.write_text('VARNAMES: CID IDSURVEY\nSN: 001 10\n')
    assert match_cid_selection(tv, [selected])[0].tolist() == [True, True, False, False]


def _prepared(ids):
    # izbin is already restricted to accepted rows; it is not a full-table array.
    tv = SimpleNamespace(name=np.array(['rejected', *ids]),
                         idsurvey=np.ones(len(ids) + 1, dtype=int), field=None)
    return SimpleNamespace(tv=tv, accept=np.array([False, *([True] * len(ids))]),
                           inputs=SimpleNamespace(izbin=np.arange(len(ids))))


def test_two_pass_rules_and_noreject_exceptions(tmp_path):
    data = read_fitres('tests/fixtures/snana/bbc_des_beams/DATA.FITRES')
    frame = data.df.iloc[np.repeat(np.arange(4), 3)].copy()
    frame['CID'] = np.arange(1, 13).astype(str)
    frame['SYNCCUT'] = np.repeat(np.arange(4), 3)
    data = replace(data, df=frame)
    config = BBCConfig(nzbin=1, zmin=.1, zmax=1., fitflag_sigmb=0,
                       cutwin=[('SYNCCUT', 0., 2.)])
    grid = {
        (0, 0): (config, {EVENT_TYPE_DATA: data}),
        (1, 0): (config, {EVENT_TYPE_DATA: replace(data, df=frame.loc[frame.SYNCCUT != 2])}),
        (0, 1): (replace(config, cutwin=[('SYNCCUT', 1., 2.)]), {EVENT_TYPE_DATA: data}),
        (2, 0): (replace(config, cutwin=[('SYNCCUT', 2., 2.)]), {EVENT_TYPE_DATA: data}),
        (0, 2): (replace(config, cutwin=[('SYNCCUT', 0., 0.)]), {EVENT_TYPE_DATA: data}),
    }
    cells = read_cells(grid)
    maps = prepare_biascor_maps(cells)
    priors = prepare_ccpriors(cells, maps)
    result = fit_prepared_common_sample(cells, maps, priors, ['CAL', 'NOREJECT_CAL'], ['CUT', 'NOREJECT_CUT'], tmp_path)
    accepted = {(0, 0): [str(i) for i in range(1, 10)],
                (1, 0): [str(i) for i in range(1, 7)],
                (0, 1): [str(i) for i in range(4, 10)],
                (2, 0): ['7', '8', '9'], (0, 2): ['1', '2', '3']}
    assert set(result) == set(grid)
    for cell, expected in accepted.items():
        path = tmp_path / f'initial/FITOPT{cell[0]:03d}_MUOPT{cell[1]:03d}/ACCEPT.LIST'
        events = read_cid_selection([path])
        assert [cid for cid, _, _ in events] == expected
        assert set(events.values()) == {0}
        _, output, _, tv, accept = result[cell]
        assert tv.name[accept].tolist() == (['7', '8', '9'] if cell == (2, 0) else ['4', '5', '6'])
        assert output.izbin.tolist() == [0, 0, 0]
    assert read_cid_selection([tmp_path / 'BBC_ACCEPT_SUMMARY.LIST']) == {
        (cid, 10, 'VOID'): 0 for cid in ('4', '5', '6')}
    assert config.cid_select_file == [] and not config.cutwin_disable
    assert (tmp_path / 'initial/FITOPT000_MUOPT000/ACCEPT.LIST').read_bytes() == (
        tmp_path / 'initial/REFERENCE.LIST').read_bytes()
    assert not list((tmp_path / 'initial').rglob('*.FITRES'))
    assert not list((tmp_path / 'initial').rglob('*.M0DIF'))


def test_empty_common_sample_fails_before_fitting(tmp_path):
    data = read_fitres('tests/fixtures/snana/bbc_des_beams/DATA.FITRES')
    config = BBCConfig(nzbin=1, zmin=.1, zmax=1., fitflag_sigmb=0)
    grid = {cell: (config, {EVENT_TYPE_DATA: replace(data, df=data.df.iloc[[row]])})
            for cell, row in (((0, 0), 0), ((0, 1), 1))}
    cells = read_cells(grid)
    maps = prepare_biascor_maps(cells)
    priors = prepare_ccpriors(cells, maps)
    with pytest.raises(SNANAError, match='no events survive'):
        fit_prepared_common_sample(cells, maps, priors, [], ['CUT'], tmp_path)
    assert (tmp_path / 'initial/FITOPT000_MUOPT000/ACCEPT.LIST').is_file()
    assert (tmp_path / 'initial/FITOPT000_MUOPT001/ACCEPT.LIST').is_file()
    assert not (tmp_path / 'BBC_ACCEPT_SUMMARY.LIST').exists()


def test_prepared_events_reject_duplicate_identities():
    from jax_pipeline.synchronise import _events
    with pytest.raises(SNANAError, match='unique CID/IDSURVEY/FIELD'):
        _events(_prepared(['1', '1']))


@pytest.mark.parametrize('value', [-1, 1.5, float('inf'), 'invalid'])
def test_invalid_sync_flags_are_rejected(value):
    from jax_pipeline.stages import sync_setting
    with pytest.raises(SNANAError, match='non-negative integer'):
        sync_setting({'FLAG_USE_SAME_EVENTS': value})
