"""Unit tests for the JIT-compiled WGTMAP weight interpolator."""

import jax.numpy as jnp
import numpy as np

from jax_snlc_core.snana_files import interpolate_wgtmap


def test_linear_ramp_recovers_linear_weight():
    """Bilinear grid linear in axis 0 → interpolated weights track that axis."""
    n0, n1 = 5, 3
    grid_ranges = jnp.array([[0.0, 4.0], [0.0, 2.0]])
    # weight = x for x in [0,4]
    grid_x = np.linspace(0.0, 4.0, n0)
    grid = jnp.asarray(np.broadcast_to(grid_x[:, None], (n0, n1)).copy())
    grid_nbins = jnp.array([n0, n1])
    props = jnp.array([
        [0.0, 1.0],
        [2.0, 1.0],
        [4.0, 1.0],
        [3.0, 0.5],
    ])
    w = np.asarray(interpolate_wgtmap(grid, grid_ranges, grid_nbins, props))
    np.testing.assert_allclose(w, [0.0, 2.0, 4.0, 3.0], atol=1e-4)


def test_galaxies_outside_grid_clamp_to_edges():
    """Galaxy properties outside the grid range get edge weights (no extrapolation)."""
    wgt_grid = jnp.array([[1.0, 1.0], [9.0, 9.0]])
    grid_ranges = jnp.array([[0.0, 1.0], [0.0, 1.0]])
    grid_nbins = jnp.array([2, 2])
    props = jnp.array([
        [-1.0, 0.5],   # below in axis 0
        [2.0, 0.5],    # above in axis 0
        [0.5, 5.0],    # above in axis 1
    ])
    w = np.asarray(interpolate_wgtmap(wgt_grid, grid_ranges, grid_nbins, props))
    assert np.all(np.isfinite(w))
    assert abs(w[0] - 1.0) < 1e-3
    assert abs(w[1] - 9.0) < 1e-3


# ----------------------------------------------------------------------------
# _interpolate_weights_with_snvar — multi-bin SN-var path
# ----------------------------------------------------------------------------

from jax_snlc_sim.hostlib.host_selection_weights import (_interpolate_weights_with_snvar)


def test_interpolate_with_snvar_returns_per_bin_weights():
    """For a (LOGMASS × x1) grid, output shape is (n_snvar_bins, n_galaxies)."""
    # 3 LOGMASS bins × 2 x1 bins
    wgt_grid = jnp.array([
        [1.0, 9.0],   # LOGMASS=8: x1_bin0=1, x1_bin1=9
        [2.0, 8.0],   # LOGMASS=10
        [3.0, 7.0],   # LOGMASS=12
    ])
    grid_ranges = jnp.array([[8.0, 12.0], [-1.0, 1.0]])
    grid_nbins = jnp.array([3, 2])
    # 3 galaxies with 1 galaxy variable, and 2 SN bins, in float32 like the
    # HOSTLIB columns and the SN-variable grid
    galaxy_props = jnp.array([[8.0], [10.0], [12.0]], dtype=jnp.float32)
    snvar_bin_centers = jnp.array([[-1.0], [+1.0]], dtype=jnp.float32)
    snvar_indices = jnp.array([1])
    galaxy_indices = jnp.array([0])

    out = np.asarray(_interpolate_weights_with_snvar(
        wgt_grid, grid_ranges, grid_nbins, galaxy_props,
        snvar_bin_centers, snvar_indices, galaxy_indices,
    ))
    assert out.shape == (2, 3)
    # Bin 0 (x1=-1): per-galaxy weights are 1, 2, 3.
    np.testing.assert_allclose(out[0], [1.0, 2.0, 3.0], atol=1e-4)
    # Bin 1 (x1=+1): 9, 8, 7.
    np.testing.assert_allclose(out[1], [9.0, 8.0, 7.0], atol=1e-4)
