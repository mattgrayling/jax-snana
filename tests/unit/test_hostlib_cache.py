"""Unit tests for the HOSTLIB cache round-trip behaviour."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from jax_snlc_sim.hostlib.catalogue import (
    _SOURCE_HASH_META_KEY, cache_paths, has_valid_cache, save_to_cache,
)


def _touch_newer(path, seconds=10):
    """Age a file forward; cache validation compares raw getmtime floats."""
    t = os.path.getmtime(path) + seconds
    os.utime(path, (t, t))


def test_save_then_load_roundtrip(tmp_path):
    """What save_to_cache writes reads back as the same DataFrame and array."""
    cache_dir = tmp_path / "cache"
    src = tmp_path / "fake.HOSTLIB"
    src.write_text("# placeholder\n")
    cache = str(cache_dir)

    df = pd.DataFrame({
        "GALID": np.arange(50, dtype=np.int64),
        "ZTRUE": np.linspace(0.05, 0.5, 50),
        "LOGMASS": np.linspace(8.0, 12.0, 50),
    })
    nbr_array = np.full((50, 4), -1, dtype=np.int32)
    nbr_array[:, 0] = np.arange(50)

    save_to_cache(cache, str(src), df, nbr_array)
    assert has_valid_cache(cache, str(src))

    parquet_path, numpy_path = cache_paths(cache, str(src))
    pd.testing.assert_frame_equal(pd.read_parquet(parquet_path), df)
    np.testing.assert_array_equal(np.load(numpy_path), nbr_array)


def test_cache_invalidated_when_source_newer(tmp_path):
    """Cache is invalid if the source HOSTLIB has been touched after caching."""
    
    cache_dir = tmp_path / "cache"
    src = tmp_path / "fake.HOSTLIB"
    src.write_text("# v1\n")
    cache = str(cache_dir)

    df = pd.DataFrame({"GALID": np.arange(3, dtype=np.int64)})
    nbr_array = np.full((3, 1), -1, dtype=np.int32)
    save_to_cache(cache, str(src), df, nbr_array)
    assert has_valid_cache(cache, str(src))

    src.write_text("# v2 — newer\n")
    _touch_newer(src)
    assert has_valid_cache(cache, str(src)) is False


def test_cache_from_other_reader_source_is_invalid(tmp_path):
    """A parquet cache stamped by another reader source is rejected."""
    cache_dir = tmp_path / "cache"
    src = tmp_path / "fake.HOSTLIB"
    src.write_text("# placeholder\n")
    cache = str(cache_dir)

    df = pd.DataFrame({"GALID": np.arange(3, dtype=np.int64)})
    nbr_array = np.full((3, 1), -1, dtype=np.int32)
    save_to_cache(cache, str(src), df, nbr_array)
    assert has_valid_cache(cache, str(src))

    # Rewrite the parquet as another reader source would have stamped it
    parquet_path, _ = cache_paths(cache, str(src))
    table = pq.read_table(parquet_path)
    schema_meta = dict(table.schema.metadata)
    schema_meta[_SOURCE_HASH_META_KEY] = b'0' * 64
    pq.write_table(table.replace_schema_metadata(schema_meta), parquet_path)
    _touch_newer(parquet_path)
    assert has_valid_cache(cache, str(src)) is False
def test_a_failed_cache_write_leaves_nothing_behind(tmp_path):
    """The cache is written through a temporary neighbour and renamed, so a
    write that dies leaves the previous file (or none) rather than the
    zero-byte Parquet the next reader trips over. Two pytest workers on a
    cold cache is the ordinary case, not a rare one."""
    from jax_snlc_sim.hostlib.catalogue import _write_then_rename

    target = tmp_path / "x.parquet"

    def die_half_written(path):
        Path(path).write_bytes(b"half a file")
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        _write_then_rename(str(target), die_half_written)
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_a_successful_cache_write_renames_into_place(tmp_path):
    """The happy path leaves the file and no temporary neighbours."""
    from jax_snlc_sim.hostlib.catalogue import _write_then_rename

    target = tmp_path / "y.parquet"
    _write_then_rename(str(target), lambda path: Path(path).write_bytes(b"ok"))
    assert target.read_bytes() == b"ok"
    assert [p.name for p in tmp_path.iterdir()] == ["y.parquet"]
