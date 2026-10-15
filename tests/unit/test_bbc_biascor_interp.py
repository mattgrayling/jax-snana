"""Tests for the scalar get_fitParBias oracle (biasCor interpolation)."""

import numpy as np
import pytest

from jax_salt2mu.biascor_map import FitParBias
from jax_salt2mu.biascor_map import BiasCorMap, fitparbias_events
from jax_salt2mu.cellgrid import BinInfo, CellGrid, uniform_bininfo


def _const_map(bias=0.5, npercell=20):
    # dims (a,b,g,z,m,s,c) = (1,1,1,3,1,3,3); no logmass interpolation.
    single = BinInfo(nbin=1, lo=np.array([-9.0]), hi=np.array([9.0]),
                     avg=np.array([0.0]), binsize=18.0)
    biz = uniform_bininfo(0.0, 0.6, 3)
    bim = BinInfo(nbin=1, lo=np.array([-20.0]), hi=np.array([20.0]),
                  avg=np.array([10.0]), binsize=40.0)
    bis = uniform_bininfo(-3.0, 3.0, 3)
    bic = uniform_bininfo(-0.3, 0.3, 3)
    grid = CellGrid((single, single, single, biz, bim, bis, bic))

    ncell = grid.ncell
    fpb = FitParBias(
        val=np.full((ncell, 1), bias), err=np.full((ncell, 1), 0.1),
        rms=np.full((ncell, 1), 0.05), npercell=np.full(ncell, npercell))

    # per-cell centroids at the bin centres
    avg = {"z": np.zeros(ncell), "m": np.zeros(ncell),
           "x1": np.zeros(ncell), "c": np.zeros(ncell)}
    for j in range(ncell):
        _, _, _, iz, im, is_, ic = np.unravel_index(j, grid.dims)
        avg["z"][j] = biz.avg[iz]
        avg["m"][j] = bim.avg[im]
        avg["x1"][j] = bis.avg[is_]
        avg["c"][j] = bic.avg[ic]
    return BiasCorMap(grid=grid, bininfo=grid.bins, fitparbias=fpb, avg=tuple(avg.values()), axis_names=("x1", "c"),
                      n_coeff=3, interp_logmass=False, zmax=1.0)


def _one(m, z, mass, s, c, min_per_cell=3):
    """One event through the vectorised API, unpacked."""
    val, err, rms, nsum, code = fitparbias_events(
        m, (0, 0, 0), np.array([z]), np.array([mass]), {"x1": np.array([s]),
                               "c": np.array([c])}, min_per_cell)
    return val[0], err[0], rms[0], int(nsum[0]), int(code[0])


def test_constant_bias_recovered():
    m = _const_map(bias=0.5)
    val, err, rms, nsum, code = _one(m, z=0.3, mass=10.0, s=0.0, c=0.0)
    assert code == 1
    np.testing.assert_allclose(val[0], 0.5)     # uniform map -> exact bias
    np.testing.assert_allclose(rms[0], 0.05)


def test_out_of_range_error_codes():
    m = _const_map()
    # Any coordinate outside the map is rejected.
    assert _one(m, 5.0, 10.0, 0.0, 0.0)[-1] == -1
    assert _one(m, 0.3, 10.0, 0.0, 9.0)[-1] == -1


def test_sparse_cells_rejected():
    # all cells below min_per_cell -> no usable cells -> rejection (negative code)
    m = _const_map(npercell=2)
    code = _one(m, 0.3, 10.0, 0.0, 0.0)[-1]
    assert code < 0


@pytest.mark.parametrize("node", [0, 1])
def test_linear_field_with_all_spatial_axes_and_multiple_coefficient_nodes(node):
    bins = (uniform_bininfo(0., 1., 2), *(uniform_bininfo(0., 1., 3) for _ in range(4)))
    grid = CellGrid(bins)
    indices = np.array(np.unravel_index(np.arange(grid.ncell), grid.dims))
    averages = tuple(b.avg[i] for b, i in zip(bins[1:], indices[1:]))
    slope = np.arange(1., 5.)
    field = 2 * indices[0] + slope @ np.array(averages)
    fpb = FitParBias(np.column_stack((field, -.25 * field)),
                    np.full((grid.ncell, 2), .01), np.full((grid.ncell, 2), .1),
                    np.full(grid.ncell, 20))
    m = BiasCorMap(grid, bins, fpb, averages, axis_names=("stretch", "colour"), n_coeff=1)
    values = np.random.default_rng(38).uniform(.2, .8, (4, 25))
    val, err, rms, nsum, code = fitparbias_events(
        m, (node,), values[0], values[1], {"stretch": values[2], "colour": values[3]}, 3)
    expected = 2 * node + slope @ values
    np.testing.assert_allclose(val, np.column_stack((expected, -.25 * expected)), atol=3e-15)
    np.testing.assert_allclose(err, .01, atol=1e-16)
    np.testing.assert_allclose(rms, .1, atol=1e-15)
    np.testing.assert_array_equal(nsum, 16 * 20)
    np.testing.assert_array_equal(code, 1)


def test_empty_interpolation_retains_output_shapes():
    empty = np.empty(0)
    result = fitparbias_events(_const_map(), (0, 0, 0), empty, empty,
                              {"x1": empty, "c": empty}, 3)
    assert [x.shape for x in result] == [(0, 1), (0, 1), (0, 1), (0,), (0,)]
