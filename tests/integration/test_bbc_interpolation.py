"""Changed coordinates, partitions or maps must not share interpolation."""
from dataclasses import replace

import numpy as np
import pytest

from jax_salt2mu import preparation
from jax_salt2mu.config import BBCConfig
from jax_salt2mu.tablevar import EVENT_TYPE_BIASCOR, EVENT_TYPE_DATA
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import read_fitres


@pytest.mark.parametrize('change', ['coordinates', 'assignment', 'minimum', 'maps'])
def test_interpolation_groups_follow_their_actual_inputs(change):
    root = 'tests/fixtures/snana/bbc_des_beams'
    tables = {EVENT_TYPE_DATA: read_fitres(f'{root}/DATA.FITRES'),
              EVENT_TYPE_BIASCOR: read_fitres(f'{root}/BIASCOR.FITRES')}
    config = BBCConfig(nzbin=10, zmin=.1, zmax=1., opt_biascor=4208,
                       min_per_cell_biascor=3, snrmin_sigint_biascor=60.,
                       surveygroup_biascor='DES(zbin=0.15:x1bin=2.0:cbin=0.2)')
    cells = preparation.read_cells({0: (config, tables), 1: (config, tables)})
    maps = preparation.prepare_biascor_maps(cells)
    data = preparation.prepare_data(cells)
    shared = preparation.interpolate_corrections(cells, data, maps)
    assert shared[0] is shared[1]
    cfg, sources, text = cells[1]
    if change == 'coordinates':
        base, merged = data[1]
        altered = replace(merged, zhd=merged.zhd + .02)
        data[1] = base, altered
    elif change == 'assignment':
        base, merged = data[1]
        altered = replace(merged, idsurvey=np.full(merged.size, 1))
        data[1] = base, altered
        with pytest.raises(SNANAError, match='no data for requested SURVEY'):
            preparation.interpolate_corrections(cells, data, maps)
        return
    elif change == 'minimum':
        cells[1] = replace(cfg, min_per_cell_biascor=100000), sources, text
    else:
        first = maps[1].maps[0]
        changed = replace(first, fitparbias=replace(first.fitparbias, val=first.fitparbias.val + .1))
        maps[1] = replace(maps[1], maps=[changed, *maps[1].maps[1:]])
    changed = preparation.interpolate_corrections(cells, data, maps)
    assert changed[0] is not changed[1]
    independent = preparation.interpolate_corrections({1: cells[1]}, data, maps)[1]
    for name, value in changed[1].items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, independent[name])
