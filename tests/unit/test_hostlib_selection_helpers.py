"""Unit tests for HOSTLIB selection pure helpers — DZTOL polynomial and SN-var binning."""

import jax.numpy as jnp
import numpy as np

from jax_snlc_sim.hostlib.hostlib_selection import (
    compute_dztol,
    compute_snvar_bin_index_1d,
    compute_snvar_bin_index_1d_vectorized,
    compute_snvar_bin_index_nd,
)


def test_compute_dztol_quadratic():
    """Three coefficients evaluate the full polynomial."""
    z = jnp.array([0.5])
    # 0.001 + 0.02*0.5 + 0.04*0.25 = 0.001 + 0.01 + 0.01 = 0.021
    dztol = np.asarray(compute_dztol(z, jnp.array([0.001, 0.02, 0.04])))
    np.testing.assert_allclose(dztol, [0.021], atol=1e-6)


def test_snvar_bin_index_1d_half_bin_snap_at_edges():
    """The bin snap covers the half-bin beyond each edge centre; values
    further out are rejected fatally at the call boundary
    (_compute_snvar_bin_indices, sntools_host.c:1655-1660), so the kernel
    itself applies no clipping."""
    centres = jnp.linspace(-3.0, 3.0, 7)
    assert int(compute_snvar_bin_index_1d(jnp.float32(-3.0), centres)) == 0
    assert int(compute_snvar_bin_index_1d(jnp.float32(+3.0), centres)) == 6
    # In-range values within half a bin of the edge centre snap to it
    assert int(compute_snvar_bin_index_1d(jnp.float32(-2.6), centres)) == 0
    assert int(compute_snvar_bin_index_1d(jnp.float32(+2.6), centres)) == 6


def test_snvar_bin_index_1d_vectorised():
    """Batched bin assignment matches per-element call."""
    centres = jnp.linspace(0.0, 1.0, 5)
    vals = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
    idx = np.asarray(compute_snvar_bin_index_1d_vectorized(vals, centres))
    np.testing.assert_array_equal(idx, [0, 1, 2, 3, 4])


def test_snvar_bin_index_nd_2d_flat_index():
    """Flat index for 2-D bin = ix * n_y + iy (row-major)."""
    x_centres = jnp.array([0.0, 1.0, 2.0])  # 3 bins
    y_centres = jnp.array([-1.0, 1.0])      # 2 bins
    # SN at (x=1.0, y=1.0) → bin (1, 1) → flat = 1*2 + 1 = 3
    flat = int(np.asarray(compute_snvar_bin_index_nd(
        [jnp.array([1.0]), jnp.array([1.0])],
        [x_centres, y_centres],
    )).item())
    assert flat == 3


def test_snvar_value_outside_wgtmap_range_is_flagged():
    """An event's WGTMAP SN-variable value outside [VALMIN, VALMAX] is flagged
    for the run to abort on (getBin_SNVAR_HOSTLIB_WGTMAP,
    sntools_host.c:1655-1660) instead of silently taking the edge bin's
    weights; the edges themselves are inside."""
    from types import SimpleNamespace
    from jax_snlc_sim.hostlib.hostlib_selection import HostSelector

    sel = SimpleNamespace(
        has_snvars=True,
        snvar_names=['x1', 'c'],
        snvar_bin_centers=[jnp.linspace(-3.0, 3.0, 7),
                           jnp.linspace(-0.3, 0.5, 5)],
    )
    outside = HostSelector.snvar_outside(
        sel, {'x1': jnp.array([0.0, 3.5, 3.0, 3.5]),
              'c': jnp.array([0.0, 0.0, 0.9, 0.9])}, 4)
    np.testing.assert_array_equal(np.asarray(outside), [0, 1, 2, 1])
