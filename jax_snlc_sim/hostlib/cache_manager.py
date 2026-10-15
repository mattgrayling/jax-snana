"""
Cache management for HOSTLIB files.

Provides fast Parquet-based caching to avoid re-parsing large HOSTLIB files.
First load parses the slow ASCII/gzip format, subsequent loads use fast binary cache.
"""

import hashlib
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from platformdirs import user_cache_dir

from jax_snlc_core.utils import snana_log as slog

logger = logging.getLogger(__name__)

APP_NAME = "jax_snana"
CACHE_SUBDIR = "hostlib_cache"

_MODULE_DIR = Path(__file__).parent


def _source_hash(*paths: Path) -> str:
    """SHA-256 of the sources that decide what a cache file holds."""
    hasher = hashlib.sha256()
    for path in paths:
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


_HOSTLIB_SOURCE_HASH = _source_hash(
    Path(__file__),
    _MODULE_DIR / 'hostlib_reader.py',
)
_SOURCE_HASH_META_KEY = b'jax_snana_source_hash'


def resolve_cache_dir(cache_dir: Optional[str] = None) -> str:
    """The cache directory: the caller's choice, else the platform location."""
    if cache_dir is not None:
        return cache_dir
    return str(Path(user_cache_dir(APP_NAME)) / CACHE_SUBDIR)


def cache_paths(cache_dir: str, source_filepath: str) -> Tuple[str, str]:
    """The ``(parquet_path, numpy_path)`` cache file paths for a source
    HOSTLIB file."""
    abs_path = os.path.abspath(source_filepath)
    hasher = hashlib.sha256(abs_path.encode('utf-8'))
    hash_str = hasher.hexdigest()[:16]
    basename_clean = Path(source_filepath).name.replace(
        '.gz', '').replace('.HOSTLIB', '')

    parquet_filename = f"{basename_clean}_{hash_str}.parquet"
    numpy_filename = f"{basename_clean}_{hash_str}_nbr.npy"

    cache_dir_path = Path(cache_dir)
    return (str(cache_dir_path / parquet_filename),
            str(cache_dir_path / numpy_filename))


def has_valid_cache(cache_dir: str, source_filepath: str) -> bool:
    """True when a valid cache exists for ``source_filepath``: both parquet
    and numpy files exist, the cache is newer than the source, and the
    parquet's embedded source hash matches the reader that would write it
    now."""
    if not Path(source_filepath).exists():
        return False

    parquet_path, numpy_path = cache_paths(cache_dir, source_filepath)
    parquet_path, numpy_path = Path(parquet_path), Path(numpy_path)
    if not (parquet_path.exists() and numpy_path.exists()):
        return False

    source_mtime = Path(source_filepath).stat().st_mtime
    cache_mtime = parquet_path.stat().st_mtime
    if cache_mtime <= source_mtime:
        return False

    schema_meta = pq.read_schema(parquet_path).metadata or {}
    return schema_meta.get(_SOURCE_HASH_META_KEY) == _HOSTLIB_SOURCE_HASH.encode()


def save_to_cache(cache_dir: str, source_filepath: str, df: pd.DataFrame,
                  nbr_array: np.ndarray):
    """Write the parsed HOSTLIB columns and neighbour array to the cache.

    Both files are written under a temporary name in the cache directory and
    renamed into place, so a reader that arrives while another process is
    still writing sees either the previous file or none, never a half-written
    one. Two workers running the same input on a cold cache is the ordinary
    case, not a rare one.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    parquet_path, numpy_path = cache_paths(cache_dir, source_filepath)

    logger.info(f"Saving to cache: {parquet_path}")
    t0 = time.time()

    table = pa.Table.from_pandas(df, preserve_index=False)
    schema_meta = dict(table.schema.metadata or {})
    schema_meta[_SOURCE_HASH_META_KEY] = _HOSTLIB_SOURCE_HASH.encode()
    table = table.replace_schema_metadata(schema_meta)

    _write_then_rename(parquet_path,
                       lambda path: pq.write_table(table, path,
                                                   compression='snappy'))
    _write_then_rename(numpy_path, lambda path: np.save(path, nbr_array))

    slog.line(logger, f"Saved {len(df):,} galaxies to cache in "
              f"{time.time() - t0:.2f}s")


def _write_then_rename(path: str, write) -> None:
    """Write through a uniquely named neighbour of ``path`` and rename it
    into place, which is atomic on one filesystem."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        # np.save appends .npy unless the name already ends in it
        target = tmp + '.npy' if path.suffix == '.npy' else tmp
        write(target)
        os.replace(target, path)
    finally:
        for leftover in (Path(tmp), Path(tmp + '.npy')):
            if leftover.exists():
                leftover.unlink()

