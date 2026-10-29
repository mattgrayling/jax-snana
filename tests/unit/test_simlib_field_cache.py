"""Unit tests for the JAX field cache over the toy SIMLIB."""

import numpy as np

from jax_snlc_sim.simlib import build_jax_field_cache, max_obs_per_event, read_simlib


def _field_cache(reader):
    """The field cache with every SIMLIB band kept at its own index."""
    return build_jax_field_cache(
        reader, filter_translation=np.arange(len(reader.filter_names)))


def test_build_jax_field_cache_values_match_reader(lsst_simlib_path):
    """Per-field MJD column in the cache matches SimlibField.mjds."""
    reader = read_simlib(str(lsst_simlib_path))
    cache = _field_cache(reader)
    for i, fld in enumerate(reader.fields):
        cache_mjds = np.asarray(cache["mjds"])[i, :fld.nobs]
        np.testing.assert_allclose(cache_mjds, fld.mjds, rtol=1e-6)


def test_max_obs_per_event_counts_the_densest_window(lsst_simlib_path):
    """The padding width is the epoch count of the densest window, and it
    saturates at the entry length once the window covers the entry."""
    reader = read_simlib(str(lsst_simlib_path))
    cache = _field_cache(reader)
    spacing = float(np.diff(np.asarray(reader.fields[0].mjds)).min())

    assert max_obs_per_event(cache, 0.0, mskopt=0) == 1
    assert max_obs_per_event(cache, 3.5 * spacing, mskopt=0) == 4
    assert (max_obs_per_event(cache, np.inf, mskopt=0)
            == cache["max_obs_per_field"])


def test_max_obs_per_event_mskopt_replaces_the_window(lsst_simlib_path):
    """SIMLIB_MSKOPT bit 256 takes the whole entry and bit 128 the season
    holding the peak, so neither reads the window; the toy entries run
    without a season gap."""
    reader = read_simlib(str(lsst_simlib_path))
    cache = _field_cache(reader)

    assert max_obs_per_event(cache, 1.0, mskopt=256) == 50
    assert max_obs_per_event(cache, 1.0, mskopt=128) == 50
