"""Unit tests for the Sersic profile helpers."""

import jax.numpy as jnp
import numpy as np

from jax_snlc_sim.hostlib.sersic_sampling import (
    compute_sersic_bn, load_sersic_bn_table)

# File-backed lookup loaded once for the tests (build-path load, HOST-12)
_BN_N_LOOKUP, _BN_LOOKUP = load_sersic_bn_table()


def _bn(ns):
    return np.asarray(
        compute_sersic_bn(jnp.asarray(ns), _BN_N_LOOKUP, _BN_LOOKUP))


def test_sersic_bn_exponential():
    """n=1 (exponential disk) → b_n ≈ 1.678."""
    assert abs(float(_bn([1.0])[0]) - 1.678) < 0.01


def test_sersic_bn_de_vaucouleurs():
    """n=4 (de Vaucouleurs) → b_n ≈ 7.669."""
    assert abs(float(_bn([4.0])[0]) - 7.669) < 0.01


def test_sersic_bn_monotonic():
    """b_n is monotonically increasing in n for n>0.36."""
    ns = np.linspace(0.5, 6.0, 50)
    assert np.all(np.diff(_bn(ns)) > 0)


