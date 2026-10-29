"""Unit tests for the SED-extrapolation helpers (rise + decay)."""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_snlc_core.logging import SNANAError
from jax_snlc_core.models.sed_model import LatetimeExtrapIa, modelflux_extrap


def test_extrap_at_phase_edge_returns_flux_edge():
    """At phase = phase_edge the extrapolation matches the input flux."""
    flux_edge = jnp.array([1.0, 2.0, 3.0])
    flux_slope = jnp.array([0.05, 0.10, 0.15])
    out = np.asarray(modelflux_extrap(
        phase=jnp.float32(30.0), phase_edge=jnp.float32(30.0),
        flux_edge=flux_edge, flux_slope=flux_slope,
    ))
    np.testing.assert_allclose(out, np.asarray(flux_edge), rtol=1e-4)


def test_extrap_post_max_decay_decreases():
    """For phase > phase_edge with positive slope, flux decreases monotonically."""
    flux_edge = jnp.array([1.0])
    flux_slope = jnp.array([0.05])  # mag/day decay
    f30 = float(modelflux_extrap(jnp.float32(30.0), jnp.float32(30.0),
                                  flux_edge, flux_slope)[0])
    f50 = float(modelflux_extrap(jnp.float32(50.0), jnp.float32(30.0),
                                  flux_edge, flux_slope)[0])
    f100 = float(modelflux_extrap(jnp.float32(100.0), jnp.float32(30.0),
                                   flux_edge, flux_slope)[0])
    assert f50 < f30
    assert f100 < f50
    # Decay never goes below zero, never above edge.
    assert f100 > 0.0


def test_extrap_pre_max_rise_zero_at_t0():
    """Fireball model: flux at T0 = -20 d is exactly zero."""
    flux_edge = jnp.array([1.0])
    flux_slope = jnp.array([0.10])
    out = float(modelflux_extrap(jnp.float32(-20.0), jnp.float32(0.0),
                                  flux_edge, flux_slope)[0])
    assert abs(out) < 1e-3


def test_extrap_rise_below_explosion_clamped_to_zero():
    """Phases below T0=-20 are zero by the rise-branch step."""
    flux_edge = jnp.array([1.0])
    flux_slope = jnp.array([0.10])
    out = float(modelflux_extrap(jnp.float32(-30.0), jnp.float32(0.0),
                                  flux_edge, flux_slope)[0])
    assert out == 0.0


# ----------------------------------------------------------------------------
# LatetimeExtrapIa parser
# ----------------------------------------------------------------------------


def _write_latetime_file(tmp_path, lines):
    p = tmp_path / "latetime.dat"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def test_latetime_parses_minimal_file(tmp_path):
    path = _write_latetime_file(tmp_path, [
        "# header comment",
        "EXTRAP_DAYMIN: 45.0",
        "EXTRAP_PARLIST: 3500.0  30.0  60.0  0.05",
        "EXTRAP_PARLIST: 5000.0  35.0  80.0  0.10",
        "EXTRAP_PARLIST: 8000.0  40.0  90.0  0.15",
    ])
    extrap = LatetimeExtrapIa(path)
    assert extrap.daymin == 45.0
    assert extrap.nlambin == 3
    np.testing.assert_allclose(extrap.lam_bins, [3500.0, 5000.0, 8000.0])
    np.testing.assert_allclose(extrap.tau1, [30.0, 35.0, 40.0])
    np.testing.assert_allclose(extrap.tau2, [60.0, 80.0, 90.0])
    np.testing.assert_allclose(extrap.expratio, [0.05, 0.10, 0.15])


def test_latetime_raises_on_tau2_less_than_tau1(tmp_path):
    path = _write_latetime_file(tmp_path, [
        "EXTRAP_DAYMIN: 45.0",
        "EXTRAP_PARLIST: 5000.0  60.0  30.0  0.05",
    ])
    with pytest.raises(SNANAError):
        LatetimeExtrapIa(path)


def test_latetime_raises_on_too_small_daymin(tmp_path):
    path = _write_latetime_file(tmp_path, [
        "EXTRAP_DAYMIN: 5.0",
        "EXTRAP_PARLIST: 5000.0  30.0  60.0  0.05",
    ])
    with pytest.raises(SNANAError):
        LatetimeExtrapIa(path)


def test_latetime_strips_inline_comments(tmp_path):
    path = _write_latetime_file(tmp_path, [
        "EXTRAP_DAYMIN: 45.0",
        "EXTRAP_PARLIST: 5000.0  30.0  60.0  0.05  # inline comment",
    ])
    extrap = LatetimeExtrapIa(path)
    assert extrap.nlambin == 1




def test_rise_branch_gradient_is_finite_at_the_explosion_edge():
    """SALT3's phase grid starts at the explosion epoch T0 = -20d, so
    phase_edge - T0 is zero and the fireball cubic's denominator vanishes.
    SNANA returns zero flux below T0 (genmag_extrap.c:74) and the forward
    value already matched, but reverse-mode AD sent 0/0 back through the
    untaken branch and the whole gradient became NaN, freezing the event for
    the rest of the round.
    """
    import jax

    def flux(phase, edge, f_edge, slope):
        return jnp.sum(modelflux_extrap(phase, edge, f_edge, slope))

    f_edge = jnp.array([1.0, 2.0, 3.0])
    slope = jnp.array([0.05, 0.10, 0.15])
    # an epoch before the grid, whose edge sits exactly at T0
    val = flux(jnp.float64(-20.1), jnp.float64(-20.0), f_edge, slope)
    grad = jax.grad(flux, argnums=(0, 2, 3))(
        jnp.float64(-20.1), jnp.float64(-20.0), f_edge, slope)
    assert float(val) == 0.0, "below T0 SNANA returns zero flux"
    for g in grad:
        assert np.all(np.isfinite(np.asarray(g))), "gradient must stay finite"

    # an edge above T0 is untouched: same values and same finite gradients
    v2 = flux(jnp.float64(-18.0), jnp.float64(-17.0), f_edge, slope)
    g2 = jax.grad(flux, argnums=0)(
        jnp.float64(-18.0), jnp.float64(-17.0), f_edge, slope)
    assert np.isfinite(float(v2)) and np.isfinite(float(g2))
