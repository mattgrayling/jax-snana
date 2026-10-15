"""Unit tests for the SFD98 dust-map coordinate helpers (no FITS needed)."""

import numpy as np

from jax_snlc_core.io.sfd_reader import SFDDustMap


def test_radec_to_galactic_north_pole_round():
    """Equatorial north pole should map near galactic l/b = (123, 27)."""
    l, b = SFDDustMap._radec_to_galactic(np.array([0.0]), np.array([90.0]))
    # Galactic NCP location is well known.
    assert 120.0 < float(l[0]) < 125.0
    assert 26.0 < float(b[0]) < 29.0


def test_lambert_projection_round_pixel_centre():
    """At galactic l=0, b=90 (NGP), pixel maps near image centre."""
    px, py = SFDDustMap._lambert_lb2fpix(
        l_deg=0.0, b_deg=90.0, nsgp=1.0, scale=2048.0,
        crpix1=2048.0, crpix2=2048.0, crval1=0.0, crval2=0.0,
    )
    # NGP maps to (~2047, ~2047) — image centre minus 1.
    assert abs(px - 2047) < 5
    assert abs(py - 2047) < 5


def test_bilinear_interp_returns_corner_value():
    """When (px, py) is exactly on a corner, bilinear returns that pixel."""
    data = np.arange(16, dtype=np.float32).reshape(4, 4)  # ny=4, nx=4
    val = SFDDustMap._bilinear_interp(data, np.array([2.0]), np.array([1.0]))
    # At (px=2, py=1) → exactly data[1, 2] = 1*4 + 2 = 6.
    np.testing.assert_allclose(val, [6.0])


def test_bilinear_interp_midpoint_average():
    """At the midpoint between four cells, bilinear returns the mean of the four."""
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    val = SFDDustMap._bilinear_interp(data, np.array([1.5]), np.array([1.5]))
    # At (px=1.5, py=1.5): cells (1,1), (1,2), (2,1), (2,2) → values 5, 6, 9, 10.
    np.testing.assert_allclose(val, [(5 + 6 + 9 + 10) / 4.0])
