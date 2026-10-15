"""HOSTLIB + WGTMAP host-first sampling: SN x1 must follow host LOGMASS.

The bundled WGTMAP couples SALT2 ``x1`` to host ``LOGMASS`` such that low-mass
hosts prefer x1≈+1 and high-mass hosts prefer x1≈-1 (mock mass-step). This
test asserts that JAX's host-first sampler produces a population where
``mean(x1 | low-mass)`` exceeds ``mean(x1 | high-mass)`` by at least 1.0.

Pins the wiring of HOSTLIB_WGTMAP_FILE → host-first PDF restriction → x1 draw.
"""

from pathlib import Path

import numpy as np


from jax_snlc_sim import run as sim_run  # noqa: E402


def test_hostlib_wgtmap_mass_step(snana_fixture):
    """JAX-sampled x1 splits by host LOGMASS as the WGTMAP prescribes."""
    fx = snana_fixture("hostlib_wgtmap_x1mass")

    config = dict(
        n_sne=500, random_seed=122459,
    )
    sim = sim_run.prepare(str(fx.input_file), **config)
    results, _, _ = sim_run.run_simulation(sim)
    params = results["parameters"]

    x1 = np.asarray(params["x1"])
    logmass = np.asarray(params.get("host_LOGMASS"))
    assigned = logmass > 0
    x1 = x1[assigned]
    logmass = logmass[assigned]
    assert len(x1) >= 300, f"too few host-assigned SNe ({len(x1)})"

    low_mask = logmass < 9.5
    high_mask = logmass >= 10.5
    assert int(low_mask.sum()) >= 50, f"low-mass bin underpopulated ({int(low_mask.sum())})"
    assert int(high_mask.sum()) >= 50, f"high-mass bin underpopulated ({int(high_mask.sum())})"

    mean_low = float(x1[low_mask].mean())
    mean_high = float(x1[high_mask].mean())
    delta = mean_low - mean_high

    # WGTMAP defines x1_pref = 1 - 0.5*(LOGMASS-8). At LOGMASS=9 → x1=+0.5,
    # at LOGMASS=11 → x1=-0.5. Per-Gaussian width 0.8. Population means in
    # the (<9.5) and (>=10.5) bins should land near +0.7 and -0.5 → Δ≈1.2,
    # reduced from the WGTMAP's nominal 1.0 spread by population mixing.
    assert delta > 1.0, (
        f"x1 not coupled to host LOGMASS: mean(low)={mean_low:.3f}, "
        f"mean(high)={mean_high:.3f}, Δ={delta:.3f}"
    )
