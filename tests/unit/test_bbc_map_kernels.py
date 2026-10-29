"""Device map selection must preserve values without changing JIT signatures."""
import jax
import numpy as np
import pytest

from jax_salt2mu import map_kernels
from _bbc_helpers import uniform_bininfo
from jax_salt2mu.map_kernels import estimate_intrinsic_scatter as estimate


def _fit(mask, *, poisoned=False, zlimits=(0., 1.)):
    n = len(mask)
    bins = (uniform_bininfo(*zlimits, 2), uniform_bininfo(-1., 1., 1))
    bounds = tuple((b.lo, b.hi, b.binsize, b.nbin) for b in bins)
    z = np.linspace(.1, .9, n)
    bias = (z + .2)[:, None]
    cov = np.full(n, .04)
    if poisoned:
        z[~mask] = np.nan
        bias[~mask] = np.nan
        cov[~mask] = np.nan
    neighbours, inside = map_kernels.neighbours((2, 1), (0, 1))
    return map_kernels.fit_map(
        (z, np.zeros(n)), bounds, np.zeros(n, dtype=np.int64),
        np.ones(n, dtype=bool), np.ones((n, 1)), cov, bias, np.ones(n),
        np.array([.1]), mask, neighbours, inside, 100., dims=(2, 1),
        active=(True, False))


def test_selection_changes_reuse_compilation_and_ignore_invalid_rows():
    mask = np.ones(60, dtype=bool)
    jax.block_until_ready(_fit(mask))
    compiled = map_kernels.fit_map._cache_size()
    mask[::3] = False
    clean = jax.tree.map(np.asarray, _fit(mask))
    poisoned = jax.tree.map(np.asarray, _fit(mask, poisoned=True))
    assert map_kernels.fit_map._cache_size() == compiled
    for left, right in zip(jax.tree.leaves(clean), jax.tree.leaves(poisoned)):
        np.testing.assert_array_equal(left, right)
    z = np.linspace(.1, .9, len(mask))
    for cell, population in enumerate((mask & (z <= .5), mask & (z > .5))):
        assert clean[3][cell] == population.sum()
        np.testing.assert_allclose(clean[0][cell, 0], np.mean(z[population] + .2))
    shifted = jax.tree.map(np.asarray, _fit(mask, zlimits=(.2, 1.2)))
    assert map_kernels.fit_map._cache_size() == compiled
    assert shifted[3].sum() == np.sum(mask & (z >= .2))
    empty = jax.tree.map(np.asarray, _fit(np.zeros(60, dtype=bool), poisoned=True))
    assert map_kernels.fit_map._cache_size() == compiled
    np.testing.assert_array_equal(empty[3], 0)
    np.testing.assert_array_equal(empty[0], 999.)
    for centroid in empty[4]:
        np.testing.assert_array_equal(centroid, 9999.)


def test_empty_event_array():
    arrays = jax.tree.map(np.asarray, _fit(np.zeros(0, dtype=bool)))
    np.testing.assert_array_equal(arrays[3], 0)
    np.testing.assert_array_equal(arrays[0], 999.)


@pytest.mark.parametrize('mad', [False, True])
def test_segmented_scatter_matches_independent_cells_with_masked_nan(mad):
    rng = np.random.default_rng(147)
    residual = rng.normal(0., .2, 150)
    covariance = rng.uniform(.005, .03, 150)
    weights = rng.uniform(.5, 1.5, 150)
    cell = np.arange(150) % 3
    valid = np.arange(150) % 7 != 0
    residual[~valid] = np.nan
    covariance[~valid] = np.nan
    result, errors, count = estimate(residual, covariance, weights, cell, valid,
                                     ncell=4, use_mad=mad, minimum=5)
    for k in range(3):
        selected = valid & (cell == k)
        single, _, _ = estimate(residual[selected], covariance[selected], weights[selected],
                                 np.zeros(selected.sum(), dtype=np.int64),
                                 np.ones(selected.sum(), dtype=bool), ncell=1, use_mad=mad)
        np.testing.assert_allclose(result[k], single[0], atol=2e-14, rtol=0.)
        assert count[k] == selected.sum()
    assert count[3] == 0 and result[3] == 0.
    assert not np.asarray(errors[0]).any()
    assert not np.asarray(errors[3]).any()


def test_centroid_roundoff_does_not_change_neighbour_choice():
    import jax.numpy as jnp
    from jax_salt2mu.binning import below_centroid

    value = np.array([.2, -.3, 10.])
    count = np.array([1, 100, 10000])
    width = .1
    for direction in (-np.inf, np.inf):
        centroid = np.nextafter(value, direction)
        np.testing.assert_array_equal(below_centroid(value, centroid, count, width), False)
        np.testing.assert_array_equal(
            below_centroid(jnp.asarray(value), jnp.asarray(centroid),
                           jnp.asarray(count), width, jnp), False)
    np.testing.assert_array_equal(below_centroid(value - .001, value, count, width), True)
    np.testing.assert_array_equal(below_centroid(value + .001, value, count, width), False)


def test_error_map_selection_reuses_compilation():
    n = 60
    bins = (uniform_bininfo(0., 1., 2), uniform_bininfo(-1., 1., 1))
    bounds = tuple((b.lo, b.hi, b.binsize, b.nbin) for b in bins)
    coords = (np.linspace(.1, .9, n), np.zeros(n))
    val, _, _, count, averages = _fit(np.ones(n, dtype=bool))

    def run(mask):
        return map_kernels.error_map(
            coords, coords, bounds, bounds, np.zeros(n, dtype=np.int64),
            np.ones(n, dtype=bool), np.ones((n, 1)), np.full(n, .01),
            np.tile([-.2, .2], n // 2), np.ones(n), np.array([.1]),
            mask, averages, val, count, 1., 3, dims=(2, 1), mdims=(2, 1),
            active=(True, False), do_mad=True, do_add=True, add_mad=True)

    jax.block_until_ready(run(np.ones(n, dtype=bool)))
    compiled = map_kernels.error_map._cache_size()
    jax.block_until_ready(run(np.arange(n) % 3 != 0))
    empty = jax.tree.map(np.asarray, run(np.zeros(n, dtype=bool)))
    assert map_kernels.error_map._cache_size() == compiled
    np.testing.assert_array_equal(empty[0], 1.)
    np.testing.assert_array_equal(empty[1], False)
    np.testing.assert_array_equal(empty[3], 0.)
    assert not any(empty[4])


def test_host_map_entry_preserves_float64_in_single_precision_fit():
    from types import SimpleNamespace
    from jax_salt2mu.biascor import BiasCorEvents, _build_one_map
    from jax_salt2mu.binning import CellGrid
    from jax_salt2mu.config import BBCConfig

    bins = (uniform_bininfo(0., 1., 1), uniform_bininfo(-1., 1., 1))
    grid = CellGrid(bins)
    bias = np.tile([1., 1. + 1.e-9], 10)[:, None]
    events = BiasCorEvents([], np.zeros(20, dtype=int), np.ones((20, 1)),
                           np.ones(20), np.zeros(20), bias)
    table = SimpleNamespace(zhd=np.full(20, .5), logmass=np.zeros(20),
                            sim_truth={"SIM_WGT_POP": np.ones(20)}, lcpar={})
    model = SimpleNamespace(biascor_coeffs=(), biascor_binning={})
    with jax.enable_x64(False):
        result = _build_one_map(table, events, model, grid, bins, np.array([0.]),
                                np.ones(20, dtype=bool), 1., BBCConfig())
        assert not jax.config.jax_enable_x64
    assert result.fitparbias.val.dtype == np.float64
    np.testing.assert_allclose(result.fitparbias.val[0, 0], np.mean(bias),
                               atol=1.e-15, rtol=0.)



@pytest.mark.parametrize('clamp', [False, True])
def test_device_bins_preserve_inclusive_edges_and_nan(clamp):
    from jax_salt2mu.binning import BinInfo

    bins = BinInfo(3, np.array([0., .2, .5]), np.array([.2, .4, .8]),
                   np.array([.1, .3, .65]), .2)
    values = np.array([-1., 0., .2, np.nextafter(.2, np.inf), .4, .45,
                       .5, .8, .8 + .5e-10, .8 + 2e-10, np.nan])
    result = jax.jit(map_kernels.index, static_argnames=('clamp',))(
        values, (bins.lo, bins.hi, bins.binsize, bins.nbin), clamp=clamp)
    np.testing.assert_array_equal(result, bins.index(values, opt=2 if clamp else 0))


def test_scatter_population_weight_normalization():
    rng = np.random.default_rng(13)
    residual = rng.normal(0., .2, 100)
    covariance = np.full(100, .01)
    weights = rng.uniform(.5, 1.5, 100)
    cell = np.zeros(100, dtype=np.int64)
    valid = np.ones(100, dtype=bool)
    ordinary = estimate(residual, covariance, weights, cell, valid, ncell=1)[0]
    large = estimate(residual, covariance, weights * 1.e200, cell, valid, ncell=1)[0]
    np.testing.assert_allclose(ordinary, large, rtol=0., atol=2e-14)


@pytest.mark.parametrize('count', [1, 3])
@pytest.mark.parametrize('clamp', [False, True])
def test_padded_bin_edges_use_true_last_bin(count, clamp):
    bins = uniform_bininfo(.1, .7, count)
    values = np.array([0., .1, *bins.hi, .7 + .5e-10, .7 + 2e-10, .9, np.nan])
    result = jax.jit(map_kernels.index, static_argnames=('clamp',))(
        values, map_kernels.map_bounds((bins,), 5)[0], clamp=clamp)
    np.testing.assert_array_equal(result, bins.index(values, opt=2 if clamp else 0))


def test_padded_neighbours_exclude_spare_cells():
    dims = (2, 5, 2)
    for count in (1, 3):
        real = (2, count, 2)
        cells, valid = jax.tree.map(np.asarray, map_kernels.neighbours(dims, (1, 2), real))
        for centre in range(np.prod(dims)):
            position = np.unravel_index(centre, dims)
            if position[1] >= count:
                assert not valid[centre].any()
            for neighbour in cells[centre, valid[centre]]:
                coordinate = np.unravel_index(neighbour, dims)
                assert coordinate[0] == position[0]
                assert coordinate[1] < count
