"""Unit tests for the asymmetric photo-z error solver."""

import numpy as np

from jax_snlc_sim.hostlib.zphot_generator import (
    NRAND_ZPHOT,
    GenZphot,
    generate_zphot_from_calc,
    zphoterr_asym,
)


def test_zphoterr_asym_rms_tracks_input():
    """The output RMS should approximate the requested zphoterr."""
    ztrue = np.full(20, 0.3)
    zphoterr = np.linspace(0.01, 0.10, 20)
    _, _, _, rms = zphoterr_asym(ztrue, zphoterr)
    # The asymmetric solver targets variance = zphoterr^2; allow ~30%
    # tolerance because the bifurcated-Gaussian variance constraint shifts
    # things at small z/zerr ratios.
    rel_err = np.abs(rms - zphoterr) / zphoterr
    assert float(np.median(rel_err)) < 0.3


def test_generated_zphot_mean_above_ztrue_for_asymmetric():
    """At small z/sigma the asymmetric solver pushes sigma_hi above sigma_lo
    (zphoterr_asym), so the generated ZPHOT mean lies above ztrue.

    Uses the engine's own draw (GEN_SNHOST_ZPHOT_from_CALC via the truncated
    bifurcated inverse CDF, sntools_host.c:6916-6952).
    """
    n = 5000
    rng = np.random.default_rng(7)
    ztrue = np.full(n, 0.05)
    uniforms = rng.uniform(size=(n, NRAND_ZPHOT))
    model = GenZphot(fudgepar=np.array([0.05, 0.0, 0.0, 0.0, 0.0]),
                     fudgemap_z=None, fudgemap_rms=None, bias=None,
                     outlier_range=(0.0, 1.0))
    zphot, _ = generate_zphot_from_calc(ztrue, model, uniforms, ztrue)
    assert float(np.mean(zphot)) > 0.05
