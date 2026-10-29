"""Structural tests for the biasCor FITPARBIAS accumulation.

Exact SNANA parity is checked at the fixture stage; here we verify the weighted
mean recovers an injected per-cell bias and that under-populated cells borrow.
"""

import numpy as np

from jax_salt2mu.biascor import FitParBias
from jax_salt2mu.map_kernels import MINPERCELL_BIASCOR_RMS, fit_moments, neighbours
from jax_salt2mu.binning import CellGrid
from _bbc_helpers import uniform_bininfo


def _grid(nz, nc):
    return CellGrid((uniform_bininfo(0.0, 1.0, nz), uniform_bininfo(0.0, 1.0, nc)))


def _moments(grid, cells, bias, weights, population, spatial_dims):
    """Run the production accumulation kernel on explicitly assigned cells."""
    nearby, inside = neighbours(grid.dims, tuple(spatial_dims))
    values = fit_moments(cells, bias, weights, population,
                        np.ones(len(cells), dtype=bool), nearby, inside)
    return FitParBias(*(np.asarray(value) for value in values))


def test_weighted_mean_bias_recovered():
    grid = _grid(3, 3)  # ncell 9, spatial dims [0,1]
    # cell 4: 12 events, bias 0.5; cell 0: 2 events, bias 0.2 and 0.6 -> mean 0.4.
    j1d = np.array([4] * 12 + [0, 0])
    bias = np.array([[0.5]] * 12 + [[0.2], [0.6]])
    wgt = np.ones(14)
    fpb = _moments(grid, j1d, bias, wgt, wgt, spatial_dims=[0, 1])

    assert fpb.val.shape == (9, 1)
    np.testing.assert_allclose(fpb.val[4, 0], 0.5)
    np.testing.assert_allclose(fpb.val[0, 0], 0.4)
    # empty cells carry SNANA's 999 sentinel
    assert fpb.val[8, 0] == 999.0
    np.testing.assert_array_equal(fpb.npercell[[0, 4, 8]], [2, 12, 0])


def test_low_count_cell_borrows_rms():
    grid = _grid(3, 3)
    # Populous centre (>=MINPERCELL) with spread; a single-event neighbour borrows.
    centre = [4] * MINPERCELL_BIASCOR_RMS
    j1d = np.array(centre + [1])
    rng = np.linspace(0.0, 1.0, MINPERCELL_BIASCOR_RMS)
    bias = np.array([[v] for v in rng] + [[0.5]])
    wgt = np.ones(len(j1d))
    fpb = _moments(grid, j1d, bias, wgt, wgt, spatial_dims=[0, 1])

    # cell 1 has one event (<MINPERCELL) -> borrows the populous neighbour 4,
    # so its RMS is nonzero despite a single own event.
    assert fpb.npercell[1] == 1
    assert fpb.rms[1, 0] > 0.0
    # cell 4 uses its own spread.
    assert fpb.rms[4, 0] > 0.0


def test_neighbour_population_weights_keep_coefficient_nodes_separate():
    grid = _grid(2, 3)  # coefficient node, spatial coordinate
    cells = np.array([0, 1, 1, 3, 4])
    bias = np.array([0., 2., 4., 100., 200.])[:, None]
    population = np.array([1., 2., 1., 1., 1.])
    result = _moments(grid, cells, bias, np.ones(5), population, [1])
    # Node 0 has population mean 2 and variance 2; node 1 must not contribute.
    np.testing.assert_allclose(result.rms[:2, 0], np.sqrt(2.))
    np.testing.assert_allclose(result.err[:2, 0], np.sqrt(2. / np.array([1., 3.])))
    assert result.val[1, 0] == 3.  # fit weights, rather than population weights
    assert result.val[2, 0] == 999.
