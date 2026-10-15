"""Named random streams for the simulator.

Every draw in a run derives from one root key. An ordinary run's is `run_key`,
save the random CID list, which is drawn off `root_key` of the RANSEED alone;
the compiled pipeline (`build_sbi_pipeline`) takes its root key from the
caller. A stream folds its name into the root, and a per-SN draw folds the
SN's global event index in as well, so a quantity depends on the SN and not
on the batch it was generated in, and two streams never share a sequence.
"""

import zlib

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_sim.simulation import FORMAT_MASK_CIDRAN


def _tag(name: str) -> np.uint32:
    return np.uint32(zlib.crc32(name.encode()))


def root_key(random_seed):
    """The root key of a run whose randomness comes from RANSEED."""
    return jax.random.PRNGKey(int(random_seed))


def run_key(simulation):
    """The root key of an ordinary run's events: RANSEED's, with CIDOFF folded
    in under random CIDs (FORMAT_MASK 16), where SNANA's CID draw takes
    CIDOFF and more randoms off the stream before the first event. The CID
    draw itself stays on `root_key`, so every job sharing a seed draws the
    same CID list."""
    key = root_key(simulation.run.random_seed)
    if simulation.output.format_mask & FORMAT_MASK_CIDRAN and simulation.output.cidoff:
        key = jax.random.fold_in(key, simulation.output.cidoff)
    return key


def stream_key(key, name: str):
    """The base key of a named random stream off the root ``key``."""
    return jax.random.fold_in(key, _tag(name))


def event_keys(key, name: str, global_indices):
    """Per-SN keys for a named stream off the root ``key``, at arbitrary
    global event indices: the SN index folded in first, then the name."""
    sn_keys = jax.vmap(lambda idx: jax.random.fold_in(key, idx))(
        jnp.asarray(global_indices))
    return jax.vmap(lambda k: jax.random.fold_in(k, _tag(name)))(sn_keys)


def batch_keys(key, name: str, n_sne: int, global_start_idx: int):
    """Per-SN keys for a named stream over a contiguous run of events."""
    return event_keys(key, name, jnp.arange(n_sne) + global_start_idx)
