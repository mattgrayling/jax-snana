"""SIMLIB HDF5 cache round-trip.

Most other tests run with a warm SIMLIB cache, so the cache-miss codepath
(text parse + cache write) is unhit in normal runs. One test covers the
full cycle: parse from text, write the cache, reload from it, and require
the two readers to agree on their complete per-field state — a cache that
silently drops any parsed attribute fails here. Also checks mtime-based
staleness invalidation.
"""

import os
import shutil
from dataclasses import fields as dataclass_fields
from pathlib import Path

import numpy as np
import pytest

from jax_snlc_sim.survey.simlib_reader import (
    SimlibField, _get_cache_path, _is_cache_valid, read_simlib,
)


@pytest.fixture
def cache_dir(tmp_path):
    """A cache rooted in ``tmp_path`` so the first read forces a real parse
    rather than using the repo cache."""
    return tmp_path / "simlib_cache"


def test_simlib_cache_roundtrip(lsst_simlib_path, cache_dir, tmp_path):
    local_simlib = tmp_path / "LSST_TEST_LOCAL.SIMLIB"
    shutil.copy(lsst_simlib_path, local_simlib)

    # cache miss: text parse + cache write
    parsed = read_simlib(str(local_simlib), cache_dir=cache_dir)
    assert len(list(cache_dir.glob("*.h5"))) == 1
    assert parsed.fields and parsed.filter_names

    # cache hit: full reader state must survive the round trip
    cached = read_simlib(str(local_simlib), cache_dir=cache_dir)
    assert cached.filter_names == parsed.filter_names
    assert cached.survey_name == parsed.survey_name
    assert (cached.template_texpose_spectrograph
            == parsed.template_texpose_spectrograph)
    # the SIMLIB header's own saturation values: SNANA reads them on every
    # job (snlc_sim.c:18479-18481), so a cache that drops them turns
    # saturation off from the second run onwards
    assert parsed.global_npe_pixel_saturate is not None
    assert parsed.global_photflag_saturate is not None
    assert (cached.global_npe_pixel_saturate
            == parsed.global_npe_pixel_saturate)
    assert cached.global_photflag_saturate == parsed.global_photflag_saturate
    assert len(cached.fields) == len(parsed.fields)
    for f_parsed, f_cached in zip(parsed.fields, cached.fields):
        for fld in dataclass_fields(SimlibField):
            a = getattr(f_parsed, fld.name)
            b = getattr(f_cached, fld.name)
            if isinstance(a, (np.ndarray, tuple, float)):
                # NaN marks an absent window and must survive as NaN
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b),
                                              err_msg=fld.name)
            else:
                assert a == b, fld.name

    # staleness: bumping the source mtime invalidates the cache
    reader = read_simlib(str(local_simlib), cache_dir=cache_dir)
    cache_path = Path(_get_cache_path(reader, cache_dir))
    assert _is_cache_valid(reader, cache_dir)
    new_mtime = cache_path.stat().st_mtime + 100.0
    os.utime(local_simlib, (new_mtime, new_mtime))
    assert _is_cache_valid(reader, cache_dir) is False
