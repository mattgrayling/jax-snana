"""jax_wfit against frozen wfit.exe output.

The live_snana suite runs wfit.exe directly and is the stronger check, but it is
opt-in and never runs in CI. These compare the same cospar quantities against
references committed by tests/fixtures/wfit/build.py, so the comparison survives
without SNANA installed.

JAX matches SNANA's radiation convention for SN distances and priors.
Remaining differences include numerical integration and output rounding.
"""

import os

import pytest

from jax_wfit.config import WFITConfig
from jax_wfit.run import run_wfit

_FIXTURE = os.path.join("tests", "fixtures", "wfit")

#: Allow numerical integration and output-rounding differences.
_TOLERANCE = {
    "w": 1.0e-3,
    "om": 1.0e-3,
    "wsig_marg": 1.0e-3,
    "omsig_marg": 1.0e-3,
    "rho_wom": 2.0e-3,
    "chi2": 0.5,
}

#: Both codes omit radiation from CMB R, but SNANA uses approximately 1000
#: midpoint bins in scale factor. JAX uses Gauss--Legendre in log(1+z).
#: Retain the fixed-target CMB integration tolerance.
_CASE_TOLERANCE = {"binned_cmb": {"w": 2.0e-3, "rho_wom": 3.0e-3}}

_CASES = {
    "binned": ("SALT2mu.M0DIF", {}),
    "binned_lcdm": ("SALT2mu.M0DIF", {"model": "lcdm"}),
    "wide": ("wide.M0DIF", {}),
    "wide_lcdm": ("wide.M0DIF", {"model": "lcdm"}),
    "wide_ompri": ("wide.M0DIF", {"ompri": 0.30, "dompri": 0.02}),
    "wide_zcut": ("wide.M0DIF", {"zmin": 0.10, "zmax": 0.90}),
    "wide_snrms": ("wide.M0DIF", {"snrms": 0.05}),
    "binned_cmb": ("SALT2mu.M0DIF", {"cmb": True}),
    "binned_bao_desi": ("SALT2mu.M0DIF", {"bao": "DESI_2024"}),
    "three_row": ("three_row.M0DIF", {}),
}


def _reference(name: str) -> dict[str, str]:
    path = os.path.join(_FIXTURE, f"ref_{name}.cospar")
    with open(path) as stream:
        lines = [line for line in stream if line.strip()]
    return dict(zip(lines[0].lstrip("#").split(), lines[1].split()))


def _fit(repo_root, diagram: str, overrides: dict):
    config = WFITConfig()
    config.hubble_diagram = str(repo_root / _FIXTURE / diagram)
    config.method = "grid"
    for key, value in overrides.items():
        setattr(config, key, value)
    return run_wfit(config, write=False)[0]


@pytest.mark.parametrize("name", sorted(_CASES))
def test_cospar_matches_frozen_snana(name, repo_root):
    diagram, overrides = _CASES[name]
    result = _fit(repo_root, diagram, overrides)
    reference = _reference(name)

    observed = {
        "om": result["summaries"]["omega_m"]["value"],
        "omsig_marg": result["summaries"]["omega_m"]["sigma"],
        "chi2": float(result["likelihood"]["total_chi2"]),
    }
    if overrides.get("model") != "lcdm":
        observed["w"] = result["summaries"]["w0"]["value"]
        observed["wsig_marg"] = result["summaries"]["w0"]["sigma"]
        observed["rho_wom"] = float(result["correlation"][0, 1])

    tolerance = {**_TOLERANCE, **_CASE_TOLERANCE.get(name, {})}
    for key, value in observed.items():
        assert value == pytest.approx(
            float(reference[key]), abs=tolerance[key],
        ), f"{name}: {key}"

    if name == "binned":
        # This fixture distinguishes the reported 68% interval from std.
        summary = result["summaries"]["omega_m"]
        assert abs(summary["std"] - float(reference["omsig_marg"])) > 0.01

    # jax subtracts only the parameters it fits, so flat LCDM is N-2 where
    # WFIT uses N-3 unconditionally
    expected_ndof = int(reference["Ndof"])
    if overrides.get("model") == "lcdm":
        expected_ndof += 1
    assert result["ndof_hd"] == expected_ndof


@pytest.mark.parametrize("method", ["laplace", "nuts"])
def test_sample_based_methods_agree_with_the_pinned_grid(method, repo_root):
    """The grid is tied to SNANA above; this ties the other two to the grid.

    SNANA has no Laplace or NUTS equivalent, so cross-method agreement is the
    only numerical check available for jax's default method. Without it,
    laplace is covered by a smoke test that would pass on a wrong cosmology.
    """
    config = WFITConfig()
    config.hubble_diagram = str(repo_root / _FIXTURE / "wide.M0DIF")
    config.method = "grid"
    grid = run_wfit(config, write=False)[0]["summaries"]

    config.method = method
    config.nuts_warmup = config.nuts_samples = 400
    other = run_wfit(config, write=False)[0]["summaries"]

    for name, reference in grid.items():
        offset = abs(other[name]["value"] - reference["value"])
        assert offset < 0.2 * reference["sigma"], (
            f"{method} {name} differs from the grid by "
            f"{offset / reference['sigma']:.2f} sigma"
        )
