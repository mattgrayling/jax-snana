"""SNANA covariance NPZ packing.

create_covariance stores only the upper triangle, with the row count in a
separate ``nsn`` key. Reading it as a flattened square usually raises, but at
N = 1, 8, 49, 288 the triangle length is itself a perfect square and the reshape
silently returns a wrong-sized matrix.
"""

import numpy as np
import pytest

from jax_snlc_core.snana_files import (expand_upper_triangle, read_covariance_npz,
                                       write_covariance_npz)


def _symmetric(n, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(n, n))
    return m + m.T


@pytest.mark.parametrize("n", [1, 3, 8, 10, 49, 288])
def test_triangle_round_trips(n):
    cov = _symmetric(n)
    expanded = expand_upper_triangle(cov[np.triu_indices(n)], n)

    assert expanded.shape == (n, n)
    np.testing.assert_allclose(expanded, cov)


def test_npz_round_trip(tmp_path):
    cov = _symmetric(9)
    path = tmp_path / "covtot_inv_000.npz"
    write_covariance_npz(path, cov)

    with np.load(path) as payload:
        assert sorted(payload.files) == ["cov", "nsn"]
        assert payload["cov"].dtype == np.float32
        assert payload["cov"].size == 9 * 10 // 2

    # float32 storage is the precision ceiling, as it is for SNANA's own files
    np.testing.assert_allclose(read_covariance_npz(path), cov, rtol=0, atol=1e-6)
