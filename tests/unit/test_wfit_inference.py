"""Frozen WFIT grid parity and fast-backend smoke tests."""

from pathlib import Path

import numpy as np
import pytest

from jax_wfit.config import WFITConfig
from jax_wfit.run import run_wfit


def _fixture(repo_root: Path) -> str:
    return str(repo_root / "tests" / "fixtures" / "wfit" / "SALT2mu.M0DIF")


@pytest.mark.parametrize("radiation,expected", [
    (False, (-0.84238064, 0.23971898, 0.30211167, 0.12246771, -0.88574094)),
    (True, (-0.84240149, 0.23960549, 0.30203541, 0.12243586, -0.88575608)),
])
def test_grid_result_is_frozen_against_optimiser_drift(repo_root, radiation, expected):
    """jax's own grid numbers, pinned so a change to the grid or the likelihood
    shows up. The comparison against SNANA's `wfit.exe` on this fixture is
    tests/parity/test_wfit_frozen.py::test_cospar_matches_frozen_snana.
    """
    config = WFITConfig()
    config.hubble_diagram = _fixture(repo_root)
    config.method = "grid"
    config.radiation = radiation
    result = run_wfit(config, write=False)[0]

    assert result["point_estimate"]["w0"] == pytest.approx(expected[0], abs=2.0e-7)
    assert result["point_estimate"]["omega_m"] == pytest.approx(expected[1], abs=2.0e-7)
    assert result["summaries"]["w0"]["std"] == pytest.approx(expected[2], abs=2.0e-7)
    assert result["summaries"]["omega_m"]["std"] == pytest.approx(expected[3], abs=2.0e-7)
    assert result["correlation"][0, 1] == pytest.approx(expected[4], abs=2.0e-7)
    assert result["ndof_hd"] == 7


def test_laplace_smoke(repo_root):
    config = WFITConfig()
    config.hubble_diagram = _fixture(repo_root)
    result = run_wfit(config, write=False)[0]
    assert result["diagnostics"]["success"]
    assert set(result["point_estimate"]) == {"omega_m", "w0"}
    assert result["diagnostics"]["newton_step_over_error"] < 1.0e-3
    assert all(value > 0 for value in result["diagnostics"]["precision_eigenvalues"])
    # the chi2 minimum value is well conditioned even though this fixture's
    # posterior is a flat banana; the position is checked against the grid's
    # own scan in test_laplace_centre_matches_grid_minchi2
    assert result["likelihood"]["total_chi2"] == pytest.approx(16.60254, abs=1e-3)


def test_laplace_centre_matches_grid_minchi2(repo_root):
    """The optimiser must beat every grid node on chi2 and sit within one
    grid spacing of the node the grid scan selects."""
    lap_config = WFITConfig()
    lap_config.hubble_diagram = _fixture(repo_root)
    lap = run_wfit(lap_config, write=False)[0]

    grid_config = WFITConfig()
    grid_config.hubble_diagram = _fixture(repo_root)
    grid_config.method = "grid"
    grid_config.marg = False
    grid = run_wfit(grid_config, write=False)[0]

    assert (lap["likelihood"]["total_chi2"]
            <= grid["likelihood"]["total_chi2"] + 1e-9)
    for name in ("omega_m", "w0"):
        lo, hi = grid_config.bounds(name)
        spacing = (hi - lo) / (grid_config.steps(name) - 1)
        assert abs(lap["point_estimate"][name]
                   - grid["point_estimate"][name]) < spacing


def test_laplace_matches_grid_marginals_when_gaussian(repo_root):
    """With a CMB prior the posterior is near-Gaussian, so the optimiser's
    mode and Hessian sigma must agree with the grid's marginalised mean and
    width."""
    def run(method):
        config = WFITConfig()
        config.hubble_diagram = _fixture(repo_root)
        config.method = method
        config.cmb = True
        config.ompri_enabled = True
        config.dompri = 0.9
        return run_wfit(config, write=False)[0]

    lap, grid = run("laplace"), run("grid")
    for name in ("omega_m", "w0"):
        sigma = grid["summaries"][name]["sigma"]
        assert abs(lap["summaries"][name]["value"]
                   - grid["summaries"][name]["value"]) < 0.2 * sigma
        assert abs(lap["summaries"][name]["sigma"] / sigma - 1.0) < 0.05


@pytest.mark.parametrize("model", ("lcdm", "w0wa"))
def test_laplace_other_models_smoke(repo_root, model):
    config = WFITConfig()
    config.hubble_diagram = _fixture(repo_root)
    config.model = model
    if model == "w0wa":
        config.cmb = True
        config.dompri = 0.9
    result = run_wfit(config, write=False)[0]

    expected = {"omega_m"} if model == "lcdm" else {"omega_m", "w0", "wa"}
    assert result["diagnostics"]["success"]
    assert set(result["point_estimate"]) == expected
    assert result["diagnostics"]["newton_step_over_error"] < 1.0e-3
    assert all(value > 0 for value in result["diagnostics"]["precision_eigenvalues"])


def test_grid_padded_realisations_match_size_one_fits(tmp_path, repo_root):
    original = Path(_fixture(repo_root))
    shifted = tmp_path / "SHIFTED.M0DIF"
    shifted_lines = original.read_text().replace(
        "0.00423", "0.40423",
    ).splitlines()
    shifted.write_text("\n".join(shifted_lines[:-1]) + "\n")
    paths = (str(original), str(shifted))

    config = WFITConfig()
    config.hubble_diagram = list(paths)
    config.method = "grid"
    config.omsteps = 31
    config.wsteps = 41

    together = run_wfit(config, write=False)
    separately = []
    for path in paths:
        config.hubble_diagram = path
        separately.append(run_wfit(config, write=False)[0])

    assert len(together) == 2
    assert [
        result["ndof_hd"] + 1 + 2 for result in together
    ] == [10, 9]
    for combined, single in zip(together, separately):
        assert combined["point_estimate"] == pytest.approx(
            single["point_estimate"], abs=1.0e-13,
        )
        np.testing.assert_allclose(
            combined["covariance"], single["covariance"], atol=1.0e-13,
        )
    assert (
        abs(
            together[0]["point_estimate"]["w0"]
            - together[1]["point_estimate"]["w0"]
        )
        > 1.0e-3
    )


def test_nuts_realisations_have_independent_states_and_keys(repo_root):
    path = _fixture(repo_root)
    config = WFITConfig()
    config.hubble_diagram = [path, path]
    config.method = "nuts"
    config.nuts_chains = 2
    config.nuts_warmup = 20
    config.nuts_samples = 20
    config.ompri = 0.3
    config.dompri = 0.02
    config.perturb_priors = True
    config.prior_perturbation_seed = 86420

    results = run_wfit(config, write=False)

    assert len(results) == 2
    assert (
        results[0]["prior_targets"]["omega_m"]
        != results[1]["prior_targets"]["omega_m"]
    )
    for result in results:
        assert result["samples"]["omega_m"].shape == (2, 20)
        assert result["samples"]["w0"].shape == (2, 20)
    assert not np.array_equal(
        results[0]["samples"]["omega_m"],
        results[1]["samples"]["omega_m"],
    )
    config.hubble_diagram = path
    size_one = run_wfit(config, write=False)[0]
    for name in ("omega_m", "w0"):
        np.testing.assert_allclose(
            results[0]["samples"][name],
            size_one["samples"][name],
            rtol=2.0e-9,
            atol=5.0e-10,
        )


def test_nuts_accepts_padded_realisations(tmp_path, repo_root):
    path = Path(_fixture(repo_root))
    shorter = tmp_path / "SHORTER.M0DIF"
    shorter.write_text("\n".join(
        line
        for line in path.read_text().splitlines()
        if not line.startswith("ROW: 5 ")
    ) + "\n")
    config = WFITConfig()
    config.hubble_diagram = [str(path), str(shorter)]
    config.method = "nuts"
    config.nuts_chains = 2
    config.nuts_warmup = 10
    config.nuts_samples = 10

    results = run_wfit(config, write=False)

    assert [result["ndof_hd"] for result in results] == [7, 6]
    for result in results:
        assert result["samples"]["omega_m"].shape == (2, 10)
        assert result["samples"]["w0"].shape == (2, 10)


def test_sig_type_selects_the_reported_uncertainty(repo_root):
    """-sig_type picks which posterior width every consumer reports.

    "68" is SNANA's default (wfit.c:792) and averages the two half-widths of the
    68.3% interval about the mean, which reduces to half the q16-q84 width.
    "std" is SNANA's legacy option. Both are always computed; only the choice of
    which one `sigma` carries changes.
    """
    def run(sig_type):
        config = WFITConfig()
        config.hubble_diagram = _fixture(repo_root)
        config.method = "grid"
        config.sig_type = sig_type
        return run_wfit(config, write=False)[0]["summaries"]["omega_m"]

    default, legacy = run("68"), run("std")
    assert default["sigma"] == pytest.approx(
        0.5 * (default["q84"] - default["q16"]))
    assert legacy["sigma"] == pytest.approx(legacy["std"])
    # both statistics survive either way, so nothing is lost by the choice
    assert default["std"] == pytest.approx(legacy["std"])
    assert default["sigma"] != pytest.approx(legacy["sigma"])
