"""Coverage for DNDZ POWERLAW2 (cosmic-rate variant) and a 2nd WGTMAP axis."""

from pathlib import Path

import numpy as np


from jax_snlc_sim import run as sim_run  # noqa: E402


def test_wgtmap_couples_color_to_logmass(snana_fixture):
    """c×LOGMASS WGTMAP: low-mass hosts get higher mean SALT2 c than high-mass."""
    fx = snana_fixture("wgtmap_color")
    config = dict(n_sne=500, random_seed=122459)
    sim = sim_run.prepare(str(fx.input_file), **config)
    results, _, _ = sim_run.run_simulation(sim)
    p = results["parameters"]
    c = np.asarray(p["c"])
    lm = np.asarray(p.get("host_LOGMASS"))
    ok = lm > 0
    c, lm = c[ok], lm[ok]
    assert (lm < 9.5).sum() >= 20 and (lm >= 10.5).sum() >= 20, (
        "wgtmap_color no longer fills both mass bins with 20 hosts"
    )
    delta = float(c[lm < 9.5].mean() - c[lm >= 10.5].mean())
    # WGTMAP has c_pref(LOGMASS=8) = +0.10, c_pref(LOGMASS=12) = -0.10. The
    # GENRANGE_SALT2c=[-0.2, +0.3] window plus the narrow c sigma compresses
    # the population effect: empirical Δ across (low, mid, high) bins is
    # ~0.016. Anything > 0.01 confirms the WGTMAP is taking effect; a
    # regression that breaks coupling would push Δ to ~0.
    assert delta > 0.01, (
        f"c not coupled to LOGMASS via WGTMAP: mean(low)-mean(high) = {delta:.3f}"
    )
