"""Partial map reuse must be equivalent to rebuilding the complete partition."""
from dataclasses import replace

import numpy as np
import pytest

from jax_salt2mu.biascor_build import build_biascor_partitions, build_biascor_maps
from jax_salt2mu.config import BBCConfig, resolve
from jax_salt2mu.samples import partition_idsamples
from jax_salt2mu.tablevar import EVENT_TYPE_BIASCOR, read_tablevar
from jax_snlc_core.models import SALTModel


def _assert_maps_equal(left, right):
    for a, b in zip(left.maps, right.maps):
        for name in ('val', 'err', 'rms', 'npercell'):
            np.testing.assert_array_equal(getattr(a.fitparbias, name),
                                          getattr(b.fitparbias, name))
        for x, y in zip(a.avg, b.avg):
            np.testing.assert_array_equal(x, y)
    for a, b in zip(left.mucov_maps, right.mucov_maps):
        for name, value in vars(a).items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, getattr(b, name))
    assert left.biascor_counts == right.biascor_counts
    assert left.sigint_biascor == right.sigint_biascor


@pytest.mark.parametrize('zmin', [.25, .4])
def test_grouped_maps_match_independent_builds(zmin):
    cfg = resolve(BBCConfig(nzbin=10, zmin=.1, zmax=1., opt_biascor=112,
        min_per_cell_biascor=3, snrmin_sigint_biascor=60.,
        surveygroup_biascor='DES(zbin=0.15:x1bin=2.0:cbin=0.2)'), SALTModel)
    tv = read_tablevar(['tests/fixtures/snana/bbc_des_beams/BIASCOR.FITRES'],
                       EVENT_TYPE_BIASCOR, cfg, SALTModel)
    tv.idsurvey = np.where(np.arange(tv.size) % 2, 10, 11)
    _, samples = partition_idsamples(tv.idsurvey, tv.field, tv.zhd, tv.is_photoz, cfg)
    changed = [replace(samples[0], zrange=(zmin, 1.)), samples[1]]
    grouped = build_biascor_partitions(tv, {0: samples, 1: changed}, cfg, SALTModel, False)
    independent = build_biascor_maps(tv, changed, cfg, SALTModel, False)
    _assert_maps_equal(grouped[1], independent)
    assert grouped[0].maps[0] is not grouped[1].maps[0]
    assert grouped[0].maps[1] is grouped[1].maps[1]
    assert grouped[0].mucov_maps[1] is grouped[1].mucov_maps[1]
