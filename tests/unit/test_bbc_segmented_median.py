"""Exact cell medians, including ties, masks and exceptional floating values."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_salt2mu.map_kernels import _median


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('case', ['random', 'ties', 'constant', 'adjacent', 'wide'])
def test_cell_medians_match_sorted_order_statistics(dtype, case):
    rng = np.random.default_rng(715)
    sizes = np.array([0, 1, 2, 3, 4, 17, 64, 999, 1000, 0])
    cells = np.repeat(np.arange(len(sizes)), sizes)
    values = rng.normal(size=len(cells)).astype(dtype)
    if case == 'ties':
        values = np.round(values, 1)
    elif case == 'constant':
        values[:] = 0.
    elif case == 'adjacent':
        values = rng.choice([dtype(1), np.nextafter(dtype(1), dtype(2))], len(cells))
    elif case == 'wide':
        span = 30 if dtype == np.float32 else 290
        values = np.power(dtype(10), rng.uniform(-span, span, len(cells))).astype(dtype)
    expected = []
    for i, size in enumerate(sizes):
        ordered = np.sort(values[cells == i])
        expected.append(dtype(.5) * (ordered[(size - 1) // 2] + ordered[size // 2])
                        if size else dtype(0))
    # Excluded events deliberately contain non-finite values.
    cells = np.r_[cells, [len(sizes)] * 3]
    values = np.r_[values, np.array([np.nan, np.inf, -np.inf], dtype=dtype)]
    order = rng.permutation(len(cells))
    with jax.enable_x64(True):
        actual = _median(values[order], cells[order], sizes)
    np.testing.assert_array_equal(actual, np.asarray(expected, dtype=dtype))


@pytest.mark.parametrize('values', [
    [1., 2., np.nan], [1., np.inf], [-np.inf, 1., 2.],
    [-np.inf, np.inf], [np.nan], [1., 1., 2., np.nan],
])
def test_nonfinite_values_preserve_sort_semantics(values):
    values = np.asarray(values)
    with np.errstate(invalid='ignore'):
        ordered = np.sort(values)
        expected = .5 * (ordered[(len(values) - 1) // 2] + ordered[len(values) // 2])
    with jax.enable_x64(True):
        actual = _median(values, np.zeros(len(values), dtype=int), np.array([len(values), 0]))
    np.testing.assert_array_equal(actual, [expected, 0.])


def test_empty_input_has_empty_cell_medians():
    np.testing.assert_array_equal(
        _median(jnp.zeros(0), jnp.zeros(0, dtype=int), jnp.zeros(3, dtype=int)), 0.)


def test_subnormal_values_match_existing_jax_arithmetic():
    with jax.enable_x64(True):
        values = jnp.asarray([0., np.nextafter(0., 1.), np.nextafter(0., 1.), 1.])
        ordered = jnp.sort(values)
        expected = .5 * (ordered[1] + ordered[2])
        actual = _median(values, jnp.zeros(4, dtype=int), jnp.array([4]))
        np.testing.assert_array_equal(actual, [expected])
