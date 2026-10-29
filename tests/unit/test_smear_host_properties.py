"""Unit tests for ``_smear_host_properties``.

Host-property smearing reads the HOSTLIB settings and a root random key.
These tests use the normal simulation configuration and explicit property arrays.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_snlc_sim.hostlib.association import _parse_property_err_scales, _smear_host_properties
from jax_snlc_core.random import root_key
from jax_snlc_sim.simulation import SimulationConfig


# ---------------------------------------------------------------------------
# _parse_property_err_scales
# ---------------------------------------------------------------------------

def test_parse_property_err_scales_legacy_logmass_only():
    cfg = SimulationConfig()
    cfg.host.scale_logmass_err = 0.5
    out = _parse_property_err_scales(cfg.host)
    assert out == {"LOGMASS": 0.5}


def test_parse_property_err_scales_value_paren_name_form():
    """SNANA accepts ``0.5(LOGMASS),2.0(LOGSFR)``."""
    cfg = SimulationConfig()
    cfg.host.scale_property_err = "0.5(LOGMASS),2.0(LOGSFR)"
    out = _parse_property_err_scales(cfg.host)
    assert out == {"LOGMASS": 0.5, "LOGSFR": 2.0}


def test_parse_property_err_scales_legacy_and_new_keys_collide():
    """Specifying both legacy and new keys is a hard error."""
    from jax_snlc_core.logging import SNANAError
    cfg = SimulationConfig()
    cfg.host.scale_logmass_err = 0.5
    cfg.host.scale_property_err = "1.0(LOGMASS)"
    with pytest.raises(SNANAError, match="Cannot specify both"):
        _parse_property_err_scales(cfg.host)


# ---------------------------------------------------------------------------
# _smear_host_properties
# ---------------------------------------------------------------------------

def _smear(cfg, host_properties, n_sne=4, global_start_idx=0):
    _smear_host_properties(
        cfg.host, host_properties,
        np.arange(n_sne) + global_start_idx,
        key=root_key(cfg.run.random_seed))
    return host_properties


def test_smear_keeps_obs_when_already_present():
    """If LOGMASS_OBS exists, it's used as-is (no perturbation), but the
    ERR is still scaled by HOSTLIB_SCALE_PROPERTY_ERR if specified."""
    cfg = SimulationConfig()
    cfg.host.scale_property_err = "2.0(LOGMASS)"
    host = {
        "LOGMASS": jnp.array([10.0, 11.0, 9.5, 10.2]),
        "LOGMASS_OBS": jnp.array([10.05, 10.95, 9.55, 10.18]),
        "LOGMASS_ERR": jnp.array([0.1, 0.1, 0.1, 0.1]),
    }
    out = _smear(cfg, host)
    np.testing.assert_array_equal(
        np.asarray(out["LOGMASS_OBS"]),
        np.asarray(host["LOGMASS_OBS"]),
    )
    # ERR scaled ×2.
    np.testing.assert_allclose(
        np.asarray(out["LOGMASS_ERR"]),
        2.0 * np.asarray([0.1, 0.1, 0.1, 0.1]),
        rtol=1e-6,
    )


def test_smear_synthesises_obs_from_true_plus_err():
    """LOGMASS + LOGMASS_ERR but no _OBS → _OBS = LOGMASS + G*ERR with G a
    truncated normal on [-3, 3] (getRan_GaussClip rejection-samples,
    sntools.c:7870-7878)."""
    cfg = SimulationConfig()
    cfg.run.random_seed = 42
    n_sne = 50
    true_vals = jnp.full((n_sne,), 10.0)
    err_vals = jnp.full((n_sne,), 0.1)
    host = {"LOGMASS": true_vals, "LOGMASS_ERR": err_vals}
    out = _smear(cfg, host, n_sne=n_sne)
    assert "LOGMASS_OBS" in out
    deltas = np.asarray(out["LOGMASS_OBS"]) - 10.0
    # Each delta = clipped_normal * 0.1 → bounded by ±3 * 0.1 = ±0.3.
    assert np.max(np.abs(deltas)) <= 0.3 + 1e-5
    # Sample mean should be near zero (Gaussian draws).
    assert abs(float(np.mean(deltas))) < 0.05


def test_smear_obs_falls_back_to_true_when_no_err():
    """LOGMASS only, no _ERR and no _OBS → _OBS = LOGMASS (no smearing applied)."""
    cfg = SimulationConfig()
    host = {"LOGMASS": jnp.array([10.0, 11.0, 9.5])}
    out = _smear(cfg, host, n_sne=3)
    np.testing.assert_array_equal(
        np.asarray(out["LOGMASS_OBS"]),
        np.asarray(host["LOGMASS"]),
    )


def test_smear_does_not_derive_logssfr():
    """SNANA never derives LOGsSFR from LOGSFR - LOGMASS; each property
    comes from its own HOSTLIB columns (GEN_SNHOST_PROPERTY,
    sntools_host.c:7495-7542)."""
    cfg = SimulationConfig()
    host = {
        "LOGSFR": jnp.array([0.0, 0.5]),
        "LOGSFR_ERR": jnp.array([0.3, 0.4]),
        "LOGMASS": jnp.array([10.0, 11.0]),
        "LOGMASS_ERR": jnp.array([0.4, 0.3]),
    }
    out = _smear(cfg, host, n_sne=2)
    assert "LOGsSFR" not in out
    assert "LOGsSFR_ERR" not in out


def test_smear_applies_per_property_err_scale():
    """LOGSFR scale=3 triples the scatter while preserving the random draws."""
    base_cfg = SimulationConfig()
    base_cfg.run.random_seed = 7
    scaled_cfg = SimulationConfig()
    scaled_cfg.run.random_seed = 7  # same seed → same Gaussian draws
    scaled_cfg.host.scale_property_err = "3.0(LOGSFR)"

    n_sne = 200
    host_base = {
        "LOGSFR": jnp.zeros((n_sne,)),
        "LOGSFR_ERR": jnp.full((n_sne,), 0.1),
    }
    host_scaled = {
        "LOGSFR": jnp.zeros((n_sne,)),
        "LOGSFR_ERR": jnp.full((n_sne,), 0.1),
    }
    base_out = _smear(base_cfg, host_base, n_sne=n_sne)
    scaled_out = _smear(scaled_cfg, host_scaled, n_sne=n_sne)

    # Same seed + same true → same Gaussian draws, so OBS_scaled ≈ 3 × OBS_base.
    np.testing.assert_allclose(
        np.asarray(scaled_out["LOGSFR_OBS"]),
        3.0 * np.asarray(base_out["LOGSFR_OBS"]),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(scaled_out["LOGSFR_ERR"]),
        3.0 * np.asarray(base_out["LOGSFR_ERR"]),
        rtol=1e-6,
    )


def test_smear_truncated_normal_has_no_atoms_at_bounds():
    """getRan_GaussClip redraws until the value lies inside [-3, 3]
    (sntools.c:7870-7878): a truncated normal, not a clipped one, so no
    draw lands exactly on the +-3 sigma boundary."""
    cfg = SimulationConfig()
    cfg.run.random_seed = 7
    n_sne = 20000
    host = {"LOGMASS": jnp.full((n_sne,), 10.0),
            "LOGMASS_ERR": jnp.full((n_sne,), 1.0)}
    out = _smear(cfg, host, n_sne=n_sne)
    deltas = np.asarray(out["LOGMASS_OBS"]) - 10.0
    assert np.max(np.abs(deltas)) <= 3.0
    assert int(np.sum(np.abs(deltas) == 3.0)) == 0
