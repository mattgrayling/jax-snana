"""Shared preparation agrees with independently loading each FITOPT."""

import dataclasses

import numpy as np
import pytest

from jax_pipeline import Fit
from jax_snlc_core.utils import group_equal
from jax_snlc_fit import run
from jax_snlc_fit.fitter import fit_band_window


def assert_prepared_equal(actual, expected):
    if dataclasses.is_dataclass(expected):
        assert_prepared_equal(vars(actual), vars(expected))
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_prepared_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_prepared_equal(left, right)
    elif expected is None:
        assert actual is None
    else:
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('synchronise, snana_output, peak_option', [(False, False, 0), (True, True, 16)])
def test_shared_preparation_matches_independent_loading(
        repo_root, tmp_path, synchronise, snana_output, peak_option):
    fixture = repo_root / 'tests/fixtures/snana/salt3_fit_select'
    directories = []
    for name in ('first', 'second'):
        directory = tmp_path / name
        directory.mkdir()
        for table in ('HEAD', 'PHOT'):
            (directory / f'{name}_{table}.FITS.gz').symlink_to(fixture / f'{table}.FITS.gz')
        directories.append(directory)
    stage = Fit('prepare', fixture / 'fit.nml',
        overrides={'BATCH_SIZE': 8, 'OPT_SNCID_LIST': int(synchronise),
                   'OPT_SETPKMJD': peak_option,
                   'SNTABLE_LIST': 'SNANA FITRES' if snana_output else 'FITRES'},
        fitopts={'flux': {'MAGOBS_SHIFT_ZP': 'g .02'},
                 'dust': {'MWEBV_SCALE': 1.1},
                 'redshift': {'REDSHIFT_FINAL_SHIFT': .001},
                 'bands': {'FILTLIST_FIT': 'gri'}})
    configs, models = stage._groups()
    files = stage._files(configs)
    cells = [dict(config=run.resolved_config(configs[key], model), model=model,
                  **files[key]) for keys, model in models for key in keys]
    # Separate model objects with the same data requirements share measured arrays.
    cells.insert(1, {**cells[0], 'model': dataclasses.replace(cells[0]['model'])})
    shape, prepared = run.prepare_datasets(cells, directories)
    needs, windows = [], []
    for group in group_equal({i: (cell['config'], id(cell['model']))
                             for i, cell in enumerate(cells)}):
        i = group[0]
        cell = cells[i]
        for directory in directories:
            expected = run.load(run._at_version(cell['config'], directory),
                cell['model'], cell['kcor'], [directory], cell['shared'],
                keep_preselection=synchronise)
            assert_prepared_equal(prepared[i][directory][:3], expected[:3])
            needs.append(run.batch_needs(cell['config'], expected[0]))
        windows.append(fit_band_window(cell['model'], expected[0].photometry))
    assert shape == run.shape_for(cells[0]['config'], needs, windows)
    first, duplicate = (prepared[i][directories[0]][0] for i in (0, 1))
    assert first is not duplicate
    assert first.fluxcals is duplicate.fluxcals
    assert first.photometry is not duplicate.photometry
    reverse_shape, reverse = run.prepare_datasets(cells[::-1], directories[::-1])
    assert reverse_shape == shape
    for actual, expected in zip(reverse[::-1], prepared):
        for directory in directories:
            assert_prepared_equal(actual[directory][:3], expected[directory][:3])
