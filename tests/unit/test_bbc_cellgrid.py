"""Unit tests for the CellGrid binning primitive (J1D, neighbours, scatter)."""

import numpy as np
import pytest

from jax_salt2mu.binning import coeff_bininfo, BinInfo, CellGrid
from _bbc_helpers import uniform_bininfo
from jax_snlc_core.logging import SNANAError


def _grid(*nbins):
    return CellGrid(tuple(uniform_bininfo(0.0, 1.0, n) for n in nbins))


def test_bininfo_index_semantics():
    b = uniform_bininfo(0.0, 1.0, 4)  # edges 0,.25,.5,.75,1
    np.testing.assert_array_equal(b.index(np.array([0.1, 0.3, 0.6, 0.9])), [0, 1, 2, 3])
    # out of range -> -9; clamp with opt=2
    np.testing.assert_array_equal(b.index(np.array([-0.1, 1.5])), [-9, -9])
    np.testing.assert_array_equal(b.index(np.array([-0.1, 1.5]), opt=2), [0, 3])
    # last bin closed at hi
    assert b.index(np.array([1.0]))[0] == 3


@pytest.mark.parametrize("dtype", [int, float])
def test_bin_edges_gaps_and_nonfinite_values(dtype):
    b = BinInfo(3, np.array([0, 1, 3], dtype=dtype), np.array([1, 2, 4], dtype=dtype),
                np.array([.5, 1.5, 3.5]), 1.)
    x = np.array([0., 1., np.nextafter(1., np.inf), 2., 2.5, 3.,
                  4. + 1e-10, np.nextafter(4. + 1e-10, np.inf),
                  np.nan, -np.inf, np.inf])
    np.testing.assert_array_equal(b.index(x), [0, 0, 1, 1, -9, 2, 2, -9, -9, -9, -9])
    np.testing.assert_array_equal(b.index(x, opt=2), [0, 0, 1, 1, -9, 2, 2, 2, -9, 0, 2])


def test_j1d_is_c_order():
    g = _grid(2, 3, 4)  # dims (2,3,4), ncell 24
    assert g.ncell == 24
    # C-order: j1d(ia,ib,ic) = ia*12 + ib*4 + ic
    j = g.j1d((np.array([1]), np.array([2]), np.array([3])))
    assert int(j[0]) == 1 * 12 + 2 * 4 + 3 == 23
    # round-trips through unravel
    ia, ib, ic = g.unravel(np.array([23]))
    assert (int(ia[0]), int(ib[0]), int(ic[0])) == (1, 2, 3)


def test_neighbours_interior_edge_and_centre_once():
    g = _grid(5, 5)
    interior = g.j1d((np.array([2]), np.array([2])))[0]
    nbr = g.neighbours(interior)
    assert len(nbr) == 9                       # 3x3 interior
    assert list(nbr).count(interior) == 1      # centre included exactly once

    corner = g.j1d((np.array([0]), np.array([0])))[0]
    assert len(g.neighbours(corner)) == 4      # 2x2 at the corner


def test_neighbours_vary_subset():
    g = _grid(5, 5)
    interior = g.j1d((np.array([2]), np.array([2])))[0]
    # vary only dim 0 -> 3 neighbours along that axis
    assert len(g.neighbours(interior, vary_dims=[0])) == 3


def test_scatter_bincount():
    g = _grid(3)
    j1d = np.array([0, 0, 1, 2])
    vals = np.array([10.0, 20.0, 5.0, 7.0])
    wgt = np.array([1.0, 1.0, 2.0, 1.0])
    sum_wv, sum_w = g.scatter(j1d, vals, wgt)
    np.testing.assert_allclose(sum_wv, [30.0, 10.0, 7.0])
    np.testing.assert_allclose(sum_w, [2.0, 2.0, 1.0])
    np.testing.assert_array_equal(g.count(j1d), [2, 1, 1])


def test_a_continuous_coefficient_takes_one_bin_under_u3():
    """A biasCor drawn with a per-SN beta has thousands of distinct SIM_beta
    values; u2=3 forces one bin over them, u2=1 refuses the grid."""
    beta = np.linspace(1.3, 2.8, 200)

    forced = coeff_bininfo(beta, False, True, "beta0")
    assert forced.nbin == 1 and forced.lo[0] < beta.min() and forced.hi[0] > beta.max()
    with pytest.raises(SNANAError, match="exceed the 2-value grid bound"):
        coeff_bininfo(beta, False, False, "beta0")


def test_the_grid_is_read_from_true_ia_rows_only():
    """SALT2mu skips NONIA_INDEX != 0 when counting distinct values
    (:15480), so contaminants in the biasCor do not widen the grid."""
    alpha = np.array([0.14, 0.16, 0.14, 0.16, 0.12, 0.18, 0.20])
    is_ia = np.array([1, 1, 1, 1, 0, 0, 0], dtype=bool)

    bins = coeff_bininfo(alpha, False, False, "alpha0", is_ia)

    assert bins.nbin == 2
    np.testing.assert_allclose(bins.avg, [0.14, 0.16])
