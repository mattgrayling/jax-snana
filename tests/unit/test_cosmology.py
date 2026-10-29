"""Unit tests for the cosmology distance-modulus calculator."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_snlc_core.cosmology import (_comoving_integral, distance_modulus,
                                     distance_modulus_to_redshift, eq_to_gal,
                                     hubble_parameter, load_hz_map,
                                     transverse_comoving_distance)
from jax_snlc_core.logging import SNANAError


HUBBLE_DISTANCE_MPC = 2.998e5 / 70.0  # c/H0 with H0=70


def test_angular_diameter_distance_peaks():
    """D_A = D_M/(1+z) turns over at intermediate z (max angular size of
    standard rulers), a non-trivial property of the D_M integral."""
    da = [float(transverse_comoving_distance(
              z, 0.3, 0.0, 0.7, HUBBLE_DISTANCE_MPC)) / (1.0 + z)
          for z in [0.1, 0.5, 1.0, 1.5, 2.0, 3.0]]
    max_idx = max(range(len(da)), key=lambda i: da[i])
    assert 0 < max_idx < len(da) - 1


def test_load_hz_map_rejects_table_short_of_z_max(tmp_path):
    """SNANA's Hzfun_interp delegates to interp_1DFUN, which aborts outside
    the table (sntools_cosmology.c:512-522); a table ending below the
    configured redshift maximum is fatal at load."""
    f = tmp_path / "HZFUN.DAT"
    f.write_text("0.0 70.0\n0.5 80.0\n1.0 95.0\n")
    z, e = load_hz_map(str(f), 70.0, z_max=0.9)
    assert float(z[-1]) == 1.0
    with pytest.raises(SNANAError, match="ends at z"):
        load_hz_map(str(f), 70.0, z_max=1.5)


def test_hz_map_reads_a_repeated_redshift_from_the_lower_bin():
    """Hzfun_interp is interp_1DFUN (sntools_cosmology.c:513-522), which
    takes the lowest bin holding z, so at a redshift the table repeats E(z)
    is the end of the bin below, and between nodes it is linear."""
    hz_map = (jnp.asarray([0.0, 0.5, 0.5, 1.0]), jnp.asarray([1.0, 2.0, 3.0, 4.0]))
    e = [float(hubble_parameter(z, 0.3, 0.0, 0.7, hz_map=hz_map)) for z in (0.25, 0.5, 0.75)]
    assert e == [1.5, 2.0, 3.5]


def test_transverse_distance_flat_threshold_matches_snana():
    """|1 - OM - OL| <= 1e-5 takes the flat branch, beyond it the sinh
    correction (Hzinv_integral, sntools_cosmology.c:386-391). The values are
    SNANA's Hzinv_integral at z=1, H0=70, compiled and run; the correction
    just past the threshold is ~1e-6, so the tolerance tells the branches
    apart."""
    dh = 299792.458 / 70.0
    for omega_k, snana in ((0.0, 3303.8287844352),
                           (1.0e-5, 3303.8188637674),
                           (2.0e-5, 3303.8154967874)):
        d = float(transverse_comoving_distance(
            1.0, 0.3, omega_k, 0.7 - omega_k, dh, radiation=False))
        assert d == pytest.approx(snana, rel=4e-7), omega_k


def test_mu_to_redshift_round_trip_including_high_z():
    """The mu -> zCMB inversion recovers redshifts across the full range,
    with no upper bound (SNANA's zcmb_dLmag_invert iterates unbounded; the
    old interpolation grid silently pinned everything above z = 3)."""
    z_true = np.array([0.01, 0.5, 2.9, 3.7, 6.0])
    mus = np.array([
        float(distance_modulus(z, z, 0.3, 0.7, HUBBLE_DISTANCE_MPC))
        for z in z_true
    ])
    z_rec = np.asarray(distance_modulus_to_redshift(
        mus, 0.3, 0.7, HUBBLE_DISTANCE_MPC))
    assert np.allclose(z_rec, z_true, rtol=1e-3)


def test_mu_to_redshift_scalar_input():
    """A scalar mu returns a scalar-shaped redshift."""
    mu = float(distance_modulus(0.3, 0.3, 0.3, 0.7, HUBBLE_DISTANCE_MPC))
    z = distance_modulus_to_redshift(mu, 0.3, 0.7, HUBBLE_DISTANCE_MPC)
    assert np.shape(z) == ()
    assert abs(float(z) - 0.3) < 1e-3


def test_hz_map_integral_tracks_a_structured_table():
    """An HzFUN table with structure below a fixed quadrature's node spacing:
    the integral must follow SNANA's dense midpoint rule (Hzinv_integral,
    sntools_cosmology.c:352-397), checked against the exact integral of the
    same piecewise-linear table."""
    zt = np.linspace(0.0, 4.0, 401)
    et = 1.0 + 0.6 * zt + 0.05 * np.sin(40.0 * zt)
    hz_map = (jnp.asarray(zt), jnp.asarray(et))

    def exact(z):
        z0, z1 = zt[:-1], zt[1:]
        e0, e1 = et[:-1], et[1:]
        hi = np.minimum(z1, z)
        f = np.clip((hi - z0) / (z1 - z0), 0.0, 1.0)
        eh = e0 + f * (e1 - e0)
        b = (e1 - e0) / (z1 - z0)
        linear = np.abs(b) > 1e-12
        seg = np.where(linear,
                       np.log(eh / e0) / np.where(linear, b, 1.0),
                       np.clip(hi - z0, 0.0, None) / e0)
        return float(np.sum(np.where(f > 0.0, seg, 0.0)))

    for z in (0.5, 1.7, 3.0):
        got = float(_comoving_integral(jnp.asarray(z), 0.3, 0.0, 0.7,
                                       hz_map=hz_map))
        ref = exact(z)
        assert abs(got - ref) / ref < 5e-5, (z, got, ref)


def test_eq_to_gal_matches_snana_slaeqgal():
    """Galactic coordinates as SNANA's slaEqgal gives them (sntools.c:4409),
    the rotation SNANA uses for the CMB dipole and the dust map; the
    values are slaEqgal's own output for these positions."""
    snana = np.array([(0.0, 0.0, 96.337269649876, -60.188551749437),
                      (150.1, 2.2, 236.813654194253, 42.102645909687),
                      (52.6, -27.9, 223.614399451310, -54.902734147399),
                      (335.0, -60.0, 330.381477351645, -48.264381386836)])
    l, b = eq_to_gal(snana[:, 0], snana[:, 1])
    np.testing.assert_allclose(l, snana[:, 2], rtol=0, atol=1e-8)
    np.testing.assert_allclose(b, snana[:, 3], rtol=0, atol=1e-8)


def test_eq_to_gal_after_precision_changes():
    """Scalar and compiled vector rotations use the current precision."""
    compiled = jax.jit(eq_to_gal)
    for double in (True, False, True):
        with jax.enable_x64(double):
            dtype = np.float64 if double else np.float32
            tolerance = 1e-8 if double else 2e-5
            for ra, dec in [(0., 0.), (np.array([0.]), np.array([0.]))]:
                for rotate in (eq_to_gal, compiled):
                    longitude, latitude = rotate(ra, dec)
                    assert longitude.dtype == latitude.dtype == dtype
                    np.testing.assert_allclose(longitude, 96.337269649876,
                                               rtol=0, atol=tolerance)
                    np.testing.assert_allclose(latitude, -60.188551749437,
                                               rtol=0, atol=tolerance)
