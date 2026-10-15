"""Grouping numerical data must preserve conversion and local selection."""
from copy import deepcopy
from dataclasses import fields, replace
import numpy as np
import pytest

from jax_salt2mu.config import BBCConfig
from jax_salt2mu.preparation import read_cells, prepare_data, format_data, select_data
from jax_salt2mu.tablevar import EVENT_TYPE_DATA, tablevar_from
from jax_snlc_core.io.fitres_reader import read_fitres
from jax_snlc_core.models import standardised_model


@pytest.fixture
def source():
    table = read_fitres('tests/fixtures/snana/bbc_des_beams/BIASCOR.FITRES')
    frame = table.df.iloc[:256].copy()
    frame['OPT_PHOTOZ'] = np.arange(len(frame)) % 2
    frame['zHELERR'] = np.linspace(0., .1, len(frame))
    return replace(table, df=frame)


@pytest.mark.parametrize('change', [
    {'zpecerr': .001}, {'omega_l': .65}, {'w0': -.9}, {'wa': .2},
    {'radiation': False}, {'omega_k': .03}, {'h0': 68.}, {'muref_zpoly': [40., 2.]},
    {'varname_z': 'SIM_ZCMB'}, {'varname_gamma': 'SIM_DLMAG'},
    {'uzsim': True}, {'force_realdata': True}, {'zphot_shift': .01},
    {'zspec_errmax_idsample': .005}, {'varname_pia': 'SIM_WGT_POP'},
    {'zmin': .3}, {'cutwin_disable': True}, {'lcpar_range': {'c': (-.1, .1)}},
    {'lcpar_range': {'SNRMAX1': (0., 1.e5)}},
    {'_model_inputs': {'maxerr_abort_x0': 1., 'maxerr_abort_x1': 1., 'maxerr_abort_c': 1.}},
])
def test_changed_settings_match_independent_reading(source, change):
    config = BBCConfig(nzbin=10, zmin=.1, zmax=1.)
    cells = read_cells({0: (config, {EVENT_TYPE_DATA: source}),
                        1: (replace(config, **change), {EVENT_TYPE_DATA: source})})
    data = prepare_data(cells)
    format_data(cells, data)
    cfg, tables, _ = cells[1]
    shared = select_data(cfg, tables, data[1], None)
    independent = tablevar_from(source, EVENT_TYPE_DATA, cfg, standardised_model(cfg.model_lcfit))
    for f in fields(shared):
        a, b = getattr(shared, f.name), getattr(independent, f.name)
        if isinstance(a, dict):
            assert a.keys() == b.keys()
            for name in a:
                np.testing.assert_array_equal(a[name], b[name])
        else:
            np.testing.assert_array_equal(a, b)


def test_selection_changes_share_arrays_without_sharing_cutbits(source):
    config = BBCConfig(nzbin=10, zmin=.1, zmax=1.)
    cells = read_cells({0: (config, {EVENT_TYPE_DATA: source}),
                        1: (replace(config, zmin=.4), {EVENT_TYPE_DATA: source})})
    data = prepare_data(cells)
    format_data(cells, data)
    first = select_data(cells[0][0], cells[0][1], data[0], None)
    second = select_data(cells[1][0], cells[1][1], data[1], None)
    assert first.covmat_fit is second.covmat_fit
    assert not first.covmat_fit.flags.writeable
    assert first.passthrough['CID'] is second.passthrough['CID']
    original = second.cutbits.copy()
    first.cutbits[:] = 123
    first.lcpar.clear()
    np.testing.assert_array_equal(second.cutbits, original)
    assert second.lcpar


def test_duplicate_merging_preserves_original_cuts_and_input_arrays(source):
    frame = source.df.iloc[:20].drop(columns=[c for c in source.df if c.startswith('SIM_')]).copy()
    frame.loc[frame.index[1], 'CID'] = frame.iloc[0].CID
    frame.loc[frame.index[1], 'c'] = .8
    source = replace(source, df=frame, varnames=list(frame), varnames_by_file=[list(frame)])
    original = frame.copy(deep=True)
    config = BBCConfig(nzbin=10, zmin=.1, zmax=1., iflag_duplicate=2,
                       cutwin=[('c', -.2, .2)])
    cells = read_cells({0: (config, {EVENT_TYPE_DATA: source}),
                        1: (replace(config, cutwin_disable=True), {EVENT_TYPE_DATA: source})})
    data = prepare_data(cells)
    format_data(cells, data)
    assert data[0][1] is data[1][1]
    for key, (cfg, tables, _) in cells.items():
        shared = select_data(cfg, tables, data[key], None)
        independent = tablevar_from(source, EVENT_TYPE_DATA, cfg, standardised_model('SALT2'))
        np.testing.assert_array_equal(shared.cutbits, independent.cutbits)
        np.testing.assert_array_equal(shared.covmat_fit, independent.covmat_fit)
        for name in shared.lcpar:
            np.testing.assert_array_equal(shared.lcpar[name], independent.lcpar[name])
    import pandas as pd
    pd.testing.assert_frame_equal(frame, original, check_exact=True)
