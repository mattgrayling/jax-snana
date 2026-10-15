"""Cross-check WFIT's approximate and sampled backends against its exact grid."""

from pathlib import Path

import pytest

from jax_wfit.config import WFITConfig
from jax_wfit.run import run_wfit


#: Effective sample size required of a NUTS run, as a fraction of its draws.
#: Both configurations here return about 8%; the sampler is seeded, so the
#: number is deterministic and a floor just under it is not flaky.
_ESS_FRACTION = 0.05


def _config(repo_root: Path, method: str) -> WFITConfig:
    config = WFITConfig()
    config.hubble_diagram = str(
        repo_root / "tests" / "fixtures" / "wfit" / "SALT2mu.M0DIF"
    )
    config.method = method
    config.cmb = True
    config.dompri = 0.9
    return config


def test_laplace_and_nuts_agree_with_cmb_constrained_grid(repo_root):
    """Each backend against its like-for-like grid quantity: NUTS means
    against the grid's marginal means, the laplace mode against the grid's
    own -minchi2 mode (within a grid spacing), sigmas across the board."""
    grid = run_wfit(_config(repo_root, "grid"), write=False)[0]

    mode_config = _config(repo_root, "grid")
    mode_config.marg = False
    grid_mode = run_wfit(mode_config, write=False)[0]

    laplace = run_wfit(_config(repo_root, "laplace"), write=False)[0]

    nuts_config = _config(repo_root, "nuts")
    nuts_config.nuts_chains = 4
    nuts_config.nuts_warmup = 300
    nuts_config.nuts_samples = 500
    nuts = run_wfit(nuts_config, write=False)[0]

    assert laplace["diagnostics"]["success"]
    assert nuts["diagnostics"]["success"]
    assert nuts["diagnostics"]["divergences"] == 0
    assert max(nuts["diagnostics"]["rhat"].values()) < 1.01
    assert min(nuts["diagnostics"]["ess"].values()) > (
        _ESS_FRACTION * nuts_config.nuts_chains * nuts_config.nuts_samples
    )

    assert (laplace["likelihood"]["total_chi2"]
            <= grid_mode["likelihood"]["total_chi2"] + 1e-9)
    for name in ("omega_m", "w0"):
        grid_summary = grid["summaries"][name]

        lo, hi = mode_config.bounds(name)
        spacing = (hi - lo) / (mode_config.steps(name) - 1)
        assert abs(laplace["point_estimate"][name]
                   - grid_mode["point_estimate"][name]) < spacing

        nuts_summary = nuts["summaries"][name]
        shift = abs(nuts_summary["mean"]
                    - grid_summary["mean"]) / grid_summary["std"]
        assert shift < 0.08
        assert nuts_summary["std"] == pytest.approx(
            grid_summary["std"], rel=0.10)
        assert laplace["summaries"][name]["std"] == pytest.approx(
            grid_summary["std"], rel=0.10)


def test_default_nuts_agrees_with_broad_grid_posterior(repo_root):
    grid_config = WFITConfig()
    grid_config.hubble_diagram = str(
        repo_root / "tests" / "fixtures" / "wfit" / "SALT2mu.M0DIF"
    )
    grid_config.method = "grid"
    grid = run_wfit(grid_config, write=False)[0]

    nuts_config = WFITConfig()
    nuts_config.hubble_diagram = grid_config.hubble_diagram
    nuts_config.method = "nuts"
    nuts = run_wfit(nuts_config, write=False)[0]

    assert nuts["diagnostics"]["success"]
    assert nuts["diagnostics"]["divergences"] == 0
    assert min(nuts["diagnostics"]["ess"].values()) > (
        _ESS_FRACTION * nuts_config.nuts_chains * nuts_config.nuts_samples
    )
    for name in ("omega_m", "w0"):
        grid_summary = grid["summaries"][name]
        nuts_summary = nuts["summaries"][name]
        mean_shift = abs(nuts_summary["mean"] - grid_summary["mean"])
        mcse = (
            nuts_summary["std"]
            / nuts["diagnostics"]["ess"][name] ** 0.5
        )
        assert mean_shift < 3.0 * mcse
        assert nuts_summary["std"] == pytest.approx(grid_summary["std"], rel=0.10)
