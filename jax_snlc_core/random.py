"""Reproducible named random streams and per-event keys.

Keys depend on the seed, stream name and event index, independent of batching.
"""

import zlib

import jax
import jax.numpy as jnp
import numpy as np


def _tag(name: str) -> np.uint32:
    return np.uint32(zlib.crc32(name.encode()))


def root_key(random_seed):
    """The root key for an integer random seed."""
    return jax.random.PRNGKey(int(random_seed))


def stream_key(key, name: str):
    """The base key of a named random stream off the root ``key``."""
    return jax.random.fold_in(key, _tag(name))


def event_keys(key, name: str, global_indices):
    """Per-event keys for a named stream off the root ``key``, at arbitrary
    global event indices: the event index folded in first, then the name."""
    event_roots = jax.vmap(lambda idx: jax.random.fold_in(key, idx))(
        jnp.asarray(global_indices))
    return jax.vmap(lambda k: jax.random.fold_in(k, _tag(name)))(event_roots)


def batch_keys(key, name: str, n_sne: int, global_start_idx: int):
    """Per-event keys for a named stream over a contiguous run of events."""
    return event_keys(key, name, jnp.arange(n_sne) + global_start_idx)
