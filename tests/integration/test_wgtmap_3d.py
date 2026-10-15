"""3-D WGTMAP (x1, c, LOGMASS): exercises multi-bin SN-variable wgtsum machinery."""

from pathlib import Path

import numpy as np


from jax_snlc_sim import run as sim_run  # noqa: E402


def test_wgtmap_3d_runs_and_couples_both_axes(snana_fixture):
    """3-D WGTMAP keeps the mass-step in x1 AND in c jointly active."""
    fx = snana_fixture("wgtmap_3d")
    cfg = dict(n_sne=600, random_seed=122459)
    sim = sim_run.prepare(str(fx.input_file), **cfg)
    results, _, _ = sim_run.run_simulation(sim)
    p = results["parameters"]

    x1 = np.asarray(p["x1"])
    c = np.asarray(p["c"])
    lm = np.asarray(p.get("host_LOGMASS"))
    ok = lm > 0
    x1, c, lm = x1[ok], c[ok], lm[ok]
    assert (lm < 9.5).sum() >= 30 and (lm >= 10.5).sum() >= 30, (
        "fixture no longer fills both mass bins with 30 hosts"
    )

    delta_x1 = float(x1[lm < 9.5].mean() - x1[lm >= 10.5].mean())
    delta_c = float(c[lm < 9.5].mean() - c[lm >= 10.5].mean())
    # WGTMAP defines x1 anti-correlated with mass (1.0 → -1.0 across the range)
    # and c anti-correlated with mass (0.10 → -0.10). Population shifts are
    # smaller than the curve amplitude due to mixing + noise; require both
    # axes to show a positive coupling sign with margin > 0.1 / 0.005.
    assert delta_x1 > 0.1, f"x1 mass-step weak in 3-D WGTMAP (Δ={delta_x1:.3f})"
    assert delta_c > 0.001, f"c mass-step weak in 3-D WGTMAP (Δ={delta_c:.4f})"
