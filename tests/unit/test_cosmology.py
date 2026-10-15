"""Unit tests for the cosmology distance-modulus calculator."""

import numpy as np
import pytest

from jax_snlc_core.physics.cosmology import (
    transverse_comoving_distance,
    distance_modulus,
    distance_modulus_to_redshift,
)


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
    from jax_snlc_core.physics.cosmology import load_hz_map
    from jax_snlc_core.utils.error_handling import SNANAError
    f = tmp_path / "HZFUN.DAT"
    f.write_text("0.0 70.0\n0.5 80.0\n1.0 95.0\n")
    z, e = load_hz_map(str(f), 70.0, z_max=0.9)
    assert float(z[-1]) == 1.0
    with pytest.raises(SNANAError, match="ends at z"):
        load_hz_map(str(f), 70.0, z_max=1.5)


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
    from jax_snlc_core.physics.cosmology import _comoving_integral
    import jax.numpy as jnp
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
