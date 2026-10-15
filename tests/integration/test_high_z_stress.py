"""High-z SALT3 stress: sim at z=1.4 must produce finite SIM_MAGOBS for the
restframe-fittable epochs (no NaN/inf from K-correction edge / SED extrapolation)."""

import numpy as np
import pytest

from jax_snlc_sim import run as sim_run


def test_salt3_high_z_no_nan(tmp_path, fixture_root):
    """SALT3 sim at GENRANGE_REDSHIFT 1.0–1.4 produces finite per-epoch fluxes
    and SIM_MAGOBS for the in-range epochs."""
    base_input = (
        fixture_root / "snana" / "salt3_k21_replay" / "input.input"
    ).read_text()
    high_z_input = base_input.replace(
        "GENRANGE_REDSHIFT:  0.0200  0.4000",
        "GENRANGE_REDSHIFT:  1.0000  1.4000",
    ).replace(
        "GENVERSION:  JAX_TEST_SALT3_K21_REPLAY",
        "GENVERSION:  JAX_TEST_SALT3_HIGHZ",
    ).replace(
        "NGENTOT_LC:  1500",
        "NGENTOT_LC:  100",
    )
    inp = tmp_path / "highz.input"
    inp.write_text(high_z_input)

    cfg = dict(n_sne=100, random_seed=1)
    sim = sim_run.prepare(str(inp), **cfg)
    results, _, _ = sim_run.run_simulation(sim)

    mags = np.asarray(results["magnitudes"])
    sched = results["observation_schedule"]
    valid = np.asarray(sched.get("valid", np.ones_like(mags, dtype=bool)))

    # SED-out-of-range epochs carry sentinel mag (>=50). The remaining epochs
    # are the K-correction-active set we actually care about being finite.
    in_range = valid & (mags < 50.0)
    assert int(in_range.sum()) > 50, (
        f"too few in-range high-z epochs ({int(in_range.sum())}) — fixture "
        "may be misconfigured"
    )

    fluxcal = np.asarray(results["fluxcals_noisy"])[in_range]
    fluxcalerr = np.asarray(results["fluxcalerrs"])[in_range]
    mag_in = mags[in_range]
    n_bad = int((~np.isfinite(fluxcal)).sum() + (~np.isfinite(fluxcalerr)).sum()
                + (~np.isfinite(mag_in)).sum())
    assert n_bad == 0, (
        f"{n_bad} NaN/inf in in-range high-z flux/err/mag — K-correction "
        "produced bad values"
    )
    assert (np.abs(fluxcal) < 1e8).all(), "flux blew up beyond 1e8"
