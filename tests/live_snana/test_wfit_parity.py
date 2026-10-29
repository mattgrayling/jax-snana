"""Live parity checks against the current SNANA ``wfit.exe``."""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from jax_snlc_core.cosmology import distance_modulus
from jax_snlc_core.utils import LIGHT_KM
from jax_wfit.blinding import blinding_statuses, parameter_random_integers
from jax_wfit.config import WFITConfig
from jax_wfit.hubble import load_hubble_diagrams
from jax_wfit.output import ensemble_summary
from jax_wfit.run import run_wfit


def _run_snana_wfit(
    tmp_path: Path,
    arguments: list[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run wfit.exe with exact chi-square evaluation."""
    executable = shutil.which("wfit.exe")
    if executable is None:
        pytest.skip("wfit.exe not on PATH")
    return subprocess.run(
        [executable, *arguments, "-speed_flag_chi2", "0"],
        check=check,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _read_cospar(path: Path) -> dict[str, float]:
    lines = [
        line.strip() for line in path.read_text().splitlines() if line.strip()
    ]
    names = lines[0].lstrip("#").split()
    values = lines[1].split()
    return {
        name: float(value)
        for name, value in zip(names, values)
        if name not in {"label", "method"}
    }


def _read_first_table_row(path: Path) -> dict[str, float]:
    lines = path.read_text().splitlines()
    names = next(
        line.split()[1:] for line in lines if line.startswith("VARNAMES:")
    )
    values = next(
        line.split()[1:] for line in lines if line.startswith("ROW:")
    )
    return {
        name: float(value)
        for name, value in zip(names, values)
        if name not in {"NAME_COSPAR"}
    }


def _sigma68(result, name: str) -> float:
    summary = result["summaries"][name]
    return 0.5 * (summary["q84"] - summary["q16"])


def _write_covariance(path: Path, matrix: np.ndarray) -> None:
    # SNANA read_mucov_text consumes one matrix element from each line.
    lines = [str(matrix.shape[0])]
    lines.extend(f"{value:.14e}" for value in matrix.ravel())
    path.write_text("\n".join(lines) + "\n")


def _read_covariance(path: Path) -> np.ndarray:
    values = path.read_text().split()
    size = int(values[0])
    return np.asarray(values[1:], dtype=np.float64).reshape(size, size)


def _write_unbinned_hubble_diagram(path: Path, size: int = 315) -> None:
    redshift = np.linspace(0.02, 1.10, size)
    mu_model = np.asarray(distance_modulus(
        redshift, redshift, 0.3, 0.7, LIGHT_KM / 70.0, -1.0,
    ))
    residual = 0.04 * np.sin(31.0 * redshift) + 0.015 * np.cos(
        7.0 * redshift
    )
    muerr = 0.08 + 0.02 * redshift
    lines = ["VARNAMES: CID zHD zHDERR MU MUERR"]
    lines.extend(
        f"SN: SN{index:04d} {z:.8f} 0.001 {mu:.8f} {sigma:.8f}"
        for index, (z, mu, sigma) in enumerate(
            zip(redshift, mu_model + residual, muerr), start=1,
        )
    )
    path.write_text("\n".join(lines) + "\n")


def _write_hdibc_hubble_diagram(
    directory: Path,
    omega_m_biascor: float,
    mu_shift_sign: float,
    redshift_offset: float = 0.0,
    mushift: float = 0.0,
    size: int = 20,
) -> tuple[Path, Path]:
    directory.mkdir()
    redshift = np.linspace(0.05, 1.0, size)
    redshift = redshift + redshift_offset * np.sin(
        np.arange(redshift.size),
    )
    mu_model = np.asarray(distance_modulus(
        redshift, redshift, 0.3, 0.7, LIGHT_KM / 70.0, -1.0,
    ))
    residual = (
        0.025 * np.sin(19.0 * redshift)
        + mu_shift_sign * 0.015 * redshift
    )
    hd = directory / "HD.FITRES"
    lines = ["VARNAMES: CID zHD MU MUERR"]
    lines.extend(
        f"SN: SN{index:03d} {z:.8f} {mu:.10f} 0.08"
        for index, (z, mu) in enumerate(
            zip(redshift, mu_model + residual), start=1,
        )
    )
    hd.write_text("\n".join(lines) + "\n")
    info = directory / "INFO.YML"
    info.write_text(
        f"MUSHIFT: {mushift}\n"
        f"OMEGA_MATTER: {omega_m_biascor}\n"
        f"OMEGA_LAMBDA: {1.0 - omega_m_biascor}\n"
        "w0_LAMBDA: -1.0\n"
        "wa_LAMBDA: 0.0\n"
    )
    return hd, info


def _read_residual_cids(path: Path) -> list[str]:
    return [
        line.split()[1]
        for line in path.read_text().splitlines()
        if line.startswith("SN:")
    ]


def _read_residuals(path: Path) -> dict[str, np.ndarray]:
    lines = path.read_text().splitlines()
    names = next(
        line.split()[1:] for line in lines if line.startswith("VARNAMES:")
    )
    rows = [
        line.split()[1:] for line in lines if line.startswith("SN:")
    ]
    columns = dict(zip(names, zip(*rows)))
    return {
        name: np.asarray(values, dtype=np.float64)
        for name, values in columns.items()
        if name != "CID"
    }


@pytest.mark.parametrize("prior", ("none", "cmb", "DESI_2024", "SDSS4_2020"))
def test_wfit_grid_against_live_snana(tmp_path, repo_root, prior):
    fixture = repo_root / "tests" / "fixtures" / "wfit" / "SALT2mu.M0DIF"
    hd = tmp_path / "SALT2mu.M0DIF"
    hd.write_bytes(fixture.read_bytes())
    cospar = tmp_path / "SNANA.cospar"
    command = [
        str(hd),
        "-outfile_cospar",
        str(cospar),
    ]
    if prior == "cmb":
        command.append("-cmb")
    elif prior in {"DESI_2024", "SDSS4_2020"}:
        bao_root = Path(os.environ["SNDATA_ROOT"]) / "models" / "BAO"
        if not (bao_root / f"{prior}_mean.dat").exists():
            pytest.skip(f"{prior} BAO resources are unavailable")
        command.extend(["-bao", prior])
    _run_snana_wfit(tmp_path, command)
    reference = _read_cospar(cospar)

    config = WFITConfig()
    config.hubble_diagram = str(hd)
    config.method = "grid"
    if prior == "cmb":
        config.cmb = True
        config.dompri = 0.9
    elif prior in {"DESI_2024", "SDSS4_2020"}:
        config.bao = prior
        config.ompri_enabled = False
    result = run_wfit(config, write=False)[0]

    assert result["point_estimate"]["w0"] == pytest.approx(
        reference["w"], abs=0.25 * 0.01,
    )
    assert result["point_estimate"]["omega_m"] == pytest.approx(
        reference["om"], abs=0.25 * 0.0125,
    )
    for name, reference_name in (("w0", "wsig_marg"), ("omega_m", "omsig_marg")):
        assert _sigma68(result, name) == pytest.approx(
            reference[reference_name], rel=0.01,
        )


@pytest.mark.parametrize(
    "model,legacy_option,reference_names",
    [
        ("lcdm", "-lcdm", {"omega_m": ("om", "omsig_marg")}),
        (
            "w0wa",
            "-wa",
            {
                "omega_m": ("om", "omsig_marg"),
                "w0": ("w0", "w0sig_marg"),
                "wa": ("wa", "wasig_marg"),
            },
        ),
    ],
)
def test_wfit_grid_models_against_live_snana(
    tmp_path, repo_root, model, legacy_option, reference_names,
):
    fixture = repo_root / "tests" / "fixtures" / "wfit" / "SALT2mu.M0DIF"
    cospar = tmp_path / f"SNANA_{model}.cospar"
    _run_snana_wfit(
        tmp_path,
        [
            str(fixture),
            legacy_option,
            "-outfile_cospar",
            str(cospar),
        ],
    )
    reference = _read_cospar(cospar)

    config = WFITConfig()
    config.hubble_diagram = str(fixture)
    config.model = model
    config.method = "grid"
    result = run_wfit(config, write=False)[0]

    step = {"omega_m": 0.0125, "w0": 0.01, "wa": 0.08}
    for name, (mean_name, uncertainty_name) in reference_names.items():
        assert result["point_estimate"][name] == pytest.approx(
            reference[mean_name], abs=0.25 * step[name],
        )
        assert _sigma68(result, name) == pytest.approx(
            reference[uncertainty_name], rel=0.01,
        )

    if model == "lcdm":
        assert result["ndof_hd"] == 8
        assert int(reference["Ndof"]) == 7  # Known SNANA set_Ndof bug.
    else:
        assert result["ndof_hd"] == int(reference["Ndof"])


@pytest.mark.parametrize("covariance_kind", ("systematics", "total_inverse"))
def test_wfit_covariance_against_live_snana(
    tmp_path, repo_root, covariance_kind,
):
    fixture = repo_root / "tests" / "fixtures" / "wfit" / "SALT2mu.M0DIF"
    muerr = np.asarray(
        [
            0.03616,
            0.02459,
            0.01562,
            0.01826,
            0.01642,
            0.01803,
            0.02282,
            0.02701,
            0.03275,
            0.15219,
        ]
    )
    size = muerr.size
    systematics = np.eye(size) * 1.0e-4
    adjacent = np.arange(size - 1)
    systematics[adjacent, adjacent + 1] = 3.0e-5
    systematics[adjacent + 1, adjacent] = 3.0e-5
    if covariance_kind == "systematics":
        matrix = systematics
        option = "-mucovsys_file"
    else:
        matrix = np.linalg.inv(systematics + np.diag(muerr**2))
        option = "-mucovtot_inv_file"
    covariance = tmp_path / f"{covariance_kind}.COV"
    _write_covariance(covariance, matrix)
    cospar = tmp_path / f"SNANA_{covariance_kind}.cospar"
    _run_snana_wfit(
        tmp_path,
        [
            str(fixture),
            option,
            str(covariance),
            "-outfile_cospar",
            str(cospar),
        ],
    )
    reference = _read_cospar(cospar)

    config = WFITConfig()
    config.hubble_diagram = str(fixture)
    config.covariance = str(covariance)
    config.covariance_kind = covariance_kind
    config.method = "grid"
    result = run_wfit(config, write=False)[0]

    assert result["point_estimate"]["w0"] == pytest.approx(
        reference["w"], abs=0.25 * 0.01,
    )
    assert result["point_estimate"]["omega_m"] == pytest.approx(
        reference["om"], abs=0.25 * 0.0125,
    )
    assert _sigma68(result, "w0") == pytest.approx(
        reference["wsig_marg"], rel=0.01,
    )
    assert _sigma68(result, "omega_m") == pytest.approx(
        reference["omsig_marg"], rel=0.01,
    )


def test_wfit_precision_export_against_live_snana(
    tmp_path, repo_root,
):
    fixture = repo_root / "tests" / "fixtures" / "wfit" / "SALT2mu.M0DIF"
    size = 10
    systematics = np.eye(size) * 1.0e-4
    adjacent = np.arange(size - 1)
    systematics[adjacent, adjacent + 1] = 3.0e-5
    systematics[adjacent + 1, adjacent] = 3.0e-5
    covariance = tmp_path / "systematics.COV"
    _write_covariance(covariance, systematics)
    snana_precision = tmp_path / "SNANA_PRECISION.COV"
    completed = _run_snana_wfit(
        tmp_path,
        [
            str(fixture),
            "-mucovsys_file",
            str(covariance),
            "-outfile_mucovtot_inv",
            str(snana_precision),
        ],
        check=False,
    )
    # SNANA writes the matrix and then calls debugexit(), which returns 1.
    assert completed.returncode == 1
    assert snana_precision.exists()

    jax_precision = tmp_path / "JAX_PRECISION.COV"
    config = WFITConfig()
    config.hubble_diagram = str(fixture)
    config.covariance = str(covariance)
    config.covariance_kind = "systematics"
    config.model = "lcdm"
    config.method = "grid"
    config.omsteps = 11
    config.cospar_yaml = str(tmp_path / "JAX_PRECISION.yaml")
    config.outfile_cospar = str(tmp_path / "JAX_PRECISION.cospar")
    config.outfile_mucovtot_inv = str(jax_precision)
    run_wfit(config)

    np.testing.assert_allclose(
        _read_covariance(jax_precision),
        _read_covariance(snana_precision),
        rtol=5.0e-8,
    )


def test_wfit_idealized_uncertainty_polynomial_against_live_snana(
    tmp_path, repo_root,
):
    fixture = repo_root / "tests" / "fixtures" / "wfit" / "SALT2mu.M0DIF"
    snana_residuals = tmp_path / "SNANA_IDEAL.RESID"
    snana_cospar = tmp_path / "SNANA_IDEAL.cospar"
    _run_snana_wfit(
        tmp_path,
        [
            str(fixture),
            "-lcdm",
            "-muerr_ideal",
            "0.08,0.04",
            "-omsteps",
            "11",
            "-outfile_resid",
            str(snana_residuals),
            "-outfile_cospar",
            str(snana_cospar),
        ],
    )

    config = WFITConfig()
    config.hubble_diagram = str(fixture)
    config.om_sim = 0.3
    config.w0_sim = -1.0
    config.wa_sim = 0.0
    config.muerr_ideal = True
    config.muerr_ideal_polynomial = (0.08, 0.04)
    data = load_hubble_diagrams(config)
    reference = _read_residuals(snana_residuals)

    np.testing.assert_allclose(
        data[0].muerr,
        reference["mu_sig"],
        atol=6.0e-5,
    )


def test_wfit_unbinned_hubble_diagram_against_live_snana(tmp_path):
    hubble_diagram = tmp_path / "UNBINNED.FITRES"
    _write_unbinned_hubble_diagram(hubble_diagram)
    cospar = tmp_path / "SNANA_unbinned.cospar"
    _run_snana_wfit(
        tmp_path,
        [
            str(hubble_diagram),
            "-omsteps",
            "21",
            "-wsteps",
            "41",
            "-outfile_cospar",
            str(cospar),
        ],
    )
    reference = _read_cospar(cospar)

    config = WFITConfig()
    config.hubble_diagram = str(hubble_diagram)
    config.method = "grid"
    config.omsteps = 21
    config.wsteps = 41
    result = run_wfit(config, write=False)[0]

    assert result["ndof_hd"] == 312
    assert result["ndof_hd"] == int(reference["Ndof"])
    assert result["point_estimate"]["w0"] == pytest.approx(
        reference["w"], abs=0.25 * 0.05,
    )
    assert result["point_estimate"]["omega_m"] == pytest.approx(
        reference["om"], abs=0.25 * 0.05,
    )
    assert _sigma68(result, "w0") == pytest.approx(
        reference["wsig_marg"], rel=0.01,
    )
    assert _sigma68(result, "omega_m") == pytest.approx(
        reference["omsig_marg"], rel=0.01,
    )


@pytest.mark.parametrize("option", ("z_cuts", "muerr_force", "snrms"))
def test_wfit_likelihood_controls_against_live_snana(
    tmp_path, repo_root, option,
):
    fixture = repo_root / "tests" / "fixtures" / "wfit" / "SALT2mu.M0DIF"
    config = WFITConfig()
    config.hubble_diagram = str(fixture)
    config.method = "grid"
    command_options: list[str]
    if option == "z_cuts":
        command_options = ["-zmin", "0.28", "-zmax", "0.82"]
        config.zmin = 0.28
        config.zmax = 0.82
    elif option == "muerr_force":
        command_options = ["-muerr_force", "0.05"]
        config.muerr_force = 0.05
    else:
        command_options = ["-snrms", "0.03"]
        config.snrms = 0.03
    cospar = tmp_path / f"SNANA_{option}.cospar"
    _run_snana_wfit(
        tmp_path,
        [
            str(fixture),
            *command_options,
            "-outfile_cospar",
            str(cospar),
        ],
    )
    reference = _read_cospar(cospar)
    result = run_wfit(config, write=False)[0]

    assert result["ndof_hd"] == int(reference["Ndof"])
    assert result["point_estimate"]["w0"] == pytest.approx(
        reference["w"], abs=0.25 * 0.01,
    )
    assert result["point_estimate"]["omega_m"] == pytest.approx(
        reference["om"], abs=0.25 * 0.0125,
    )
    assert _sigma68(result, "w0") == pytest.approx(
        reference["wsig_marg"], rel=0.01,
    )
    assert _sigma68(result, "omega_m") == pytest.approx(
        reference["omsig_marg"], rel=0.01,
    )


def test_wfit_residual_output_against_live_snana(tmp_path, repo_root):
    fixture = repo_root / "tests" / "fixtures" / "wfit" / "SALT2mu.M0DIF"
    snana_residuals = tmp_path / "SNANA_RESID.FITRES"
    snana_cospar = tmp_path / "SNANA_RESID.cospar"
    _run_snana_wfit(
        tmp_path,
        [
            str(fixture),
            "-outfile_resid",
            str(snana_residuals),
            "-outfile_cospar",
            str(snana_cospar),
        ],
    )

    jax_residuals = tmp_path / "JAX_RESID.FITRES"
    config = WFITConfig()
    config.hubble_diagram = str(fixture)
    config.method = "grid"
    config.cospar_yaml = str(tmp_path / "JAX_RESID.yaml")
    config.outfile_cospar = str(tmp_path / "JAX_RESID.cospar")
    config.outfile_resid = str(jax_residuals)
    run_wfit(config)

    reference = _read_residuals(snana_residuals)
    actual = _read_residuals(jax_residuals)
    np.testing.assert_allclose(actual["zHD"], reference["zHD"], atol=6.0e-5)
    for name in ("mu_obs", "mu_model", "mu_res"):
        np.testing.assert_allclose(actual[name], reference[name], atol=6.0e-4)
    np.testing.assert_allclose(actual["mu_sig"], reference["mu_sig"], atol=6.0e-5)
    np.testing.assert_allclose(
        actual["chi2_diag"], reference["chi2_diag"], atol=6.0e-2,
    )


@pytest.mark.parametrize("covariance_count", (0, 1, 2))
def test_wfit_hdibc_against_live_snana(tmp_path, covariance_count):
    hd0, info0 = _write_hdibc_hubble_diagram(
        tmp_path / "BIASCOR0", 0.2, 1.0, mushift=0.03,
    )
    hd1, info1 = _write_hdibc_hubble_diagram(
        tmp_path / "BIASCOR1",
        0.4,
        -1.0,
        redshift_offset=2.0e-4,
        mushift=-0.02,
    )
    cospar = tmp_path / "SNANA_HDIBC.cospar"
    snana_residuals = tmp_path / "SNANA_HDIBC.RESID"
    command = [
        f"{hd0},{hd1}",
        "-lcdm",
        "-omsteps",
        "41",
        "-outfile_cospar",
        str(cospar),
        "-outfile_resid",
        str(snana_residuals),
    ]
    covariance_paths = None
    if covariance_count:
        size = 20
        covariance_paths = [
            tmp_path / "BIASCOR0" / "COVSYS.txt",
            tmp_path / "BIASCOR1" / "COVSYS.txt",
        ]
        for index, path in enumerate(covariance_paths):
            covariance = np.eye(size) * (1.0e-4 + index * 2.0e-5)
            adjacent = np.arange(size - 1)
            covariance[adjacent, adjacent + 1] = 2.0e-5
            covariance[adjacent + 1, adjacent] = 2.0e-5
            _write_covariance(path, covariance)
        selected_paths = covariance_paths[:covariance_count]
        command.extend([
            "-mucovsys_file",
            ",".join(str(path) for path in selected_paths),
        ])
    _run_snana_wfit(tmp_path, command)
    reference = _read_cospar(cospar)

    config = WFITConfig()
    config.hubble_diagram = [str(hd0), str(hd1)]
    config.model = "lcdm"
    config.hdibc = True
    config.hdibc_info = [str(info0), str(info1)]
    if covariance_paths is not None:
        selected_paths = covariance_paths[:covariance_count]
        config.covariance = (
            str(selected_paths[0])
            if covariance_count == 1
            else [str(path) for path in selected_paths]
        )
        config.covariance_kind = "systematics"
    config.method = "grid"
    config.omsteps = 41
    jax_residuals = tmp_path / "JAX_HDIBC.RESID"
    config.cospar_yaml = str(tmp_path / "JAX_HDIBC.yaml")
    config.outfile_resid = str(jax_residuals)
    result = run_wfit(config)[0]

    assert result["point_estimate"]["omega_m"] == pytest.approx(
        reference["om"], abs=0.25 * 0.025,
    )
    assert _sigma68(result, "omega_m") == pytest.approx(
        reference["omsig_marg"], rel=0.02,
    )
    reference_residuals = _read_residuals(snana_residuals)
    actual_residuals = _read_residuals(jax_residuals)
    for actual_name, reference_name, tolerance in (
        ("zHD", "zHD", 6.0e-5),
        ("mu_obs", "mu_obs", 6.0e-4),
        ("mu_model", "mu_model", 6.0e-4),
        ("mu_res", "mu_res", 6.0e-4),
        ("f_interp_hdibc", "f_interp_hdibc", 2.0e-3),
        ("mu_obs0_hdibc", "mu_obs0_hdibc", 6.0e-4),
        ("mu_obs1_hdibc", "mu_obs1_hdibc", 6.0e-4),
    ):
        np.testing.assert_allclose(
            actual_residuals[actual_name],
            reference_residuals[reference_name],
            atol=tolerance,
        )


@pytest.mark.parametrize("covariance_count", (1, 2))
def test_wfit_hdibc_inverse_covariance_against_live_snana(
    tmp_path, covariance_count,
):
    hd0, info0 = _write_hdibc_hubble_diagram(
        tmp_path / "BIASCOR0", 0.2, 1.0, mushift=0.03,
    )
    hd1, info1 = _write_hdibc_hubble_diagram(
        tmp_path / "BIASCOR1", 0.4, -1.0, mushift=-0.02,
    )
    precision_paths = [
        tmp_path / "BIASCOR0" / "COVTOT_INV.txt",
        tmp_path / "BIASCOR1" / "COVTOT_INV.txt",
    ]
    size = 20
    for index, path in enumerate(precision_paths):
        covariance = np.eye(size) * (
            0.08**2 + 1.0e-4 + index * 2.0e-5
        )
        adjacent = np.arange(size - 1)
        covariance[adjacent, adjacent + 1] = 2.0e-5
        covariance[adjacent + 1, adjacent] = 2.0e-5
        _write_covariance(path, np.linalg.inv(covariance))
    selected_paths = precision_paths[:covariance_count]
    cospar = tmp_path / "SNANA_HDIBC_INV.cospar"
    _run_snana_wfit(
        tmp_path,
        [
            f"{hd0},{hd1}",
            "-lcdm",
            "-omsteps",
            "41",
            "-mucovtot_inv_file",
            ",".join(str(path) for path in selected_paths),
            "-outfile_cospar",
            str(cospar),
        ],
    )
    reference = _read_cospar(cospar)

    config = WFITConfig()
    config.hubble_diagram = [str(hd0), str(hd1)]
    config.covariance = (
        str(selected_paths[0])
        if covariance_count == 1
        else [str(path) for path in selected_paths]
    )
    config.covariance_kind = "total_inverse"
    config.model = "lcdm"
    config.hdibc = True
    config.hdibc_info = [str(info0), str(info1)]
    config.method = "grid"
    config.omsteps = 41
    result = run_wfit(config, write=False)[0]

    assert result["point_estimate"]["omega_m"] == pytest.approx(
        reference["om"], abs=0.25 * 0.025,
    )
    assert _sigma68(result, "omega_m") == pytest.approx(
        reference["omsig_marg"], rel=0.02,
    )


def test_wfit_hdibc_cid_synchronization_against_live_snana(tmp_path):
    hd0, info0 = _write_hdibc_hubble_diagram(
        tmp_path / "BIASCOR0", 0.2, 1.0,
    )
    hd1, info1 = _write_hdibc_hubble_diagram(
        tmp_path / "BIASCOR1", 0.4, -1.0,
    )
    lines = hd1.read_text().splitlines()
    rows = lines[1:]
    retained = [
        row for index, row in enumerate(rows)
        if index not in {3, 7}
    ]
    extras = [
        rows[0].replace("SN001", "EXTRA001"),
        rows[-1].replace("SN020", "EXTRA002"),
    ]
    hd1.write_text("\n".join([lines[0], *retained, *extras]) + "\n")

    snana_cospar = tmp_path / "SNANA_SYNC.cospar"
    snana_residuals = tmp_path / "SNANA_SYNC.RESID"
    _run_snana_wfit(
        tmp_path,
        [
            f"{hd0},{hd1}",
            "-lcdm",
            "-omsteps",
            "41",
            "-outfile_cospar",
            str(snana_cospar),
            "-outfile_resid",
            str(snana_residuals),
        ],
    )
    reference = _read_cospar(snana_cospar)

    config = WFITConfig()
    config.hubble_diagram = [str(hd0), str(hd1)]
    config.model = "lcdm"
    config.hdibc = True
    config.hdibc_info = [str(info0), str(info1)]
    config.method = "grid"
    config.omsteps = 41
    jax_residuals = tmp_path / "JAX_SYNC.RESID"
    config.cospar_yaml = str(tmp_path / "JAX_SYNC.yaml")
    config.outfile_resid = str(jax_residuals)
    result = run_wfit(config)[0]

    assert result["ndof_hd"] == 16
    assert result["point_estimate"]["omega_m"] == pytest.approx(
        reference["om"], abs=0.25 * 0.025,
    )
    assert _sigma68(result, "omega_m") == pytest.approx(
        reference["omsig_marg"], rel=0.02,
    )
    assert _read_residual_cids(jax_residuals) == _read_residual_cids(
        snana_residuals,
    )


def test_wfit_hdibc_unbinned_redshifts_against_live_snana(tmp_path):
    hd0, info0 = _write_hdibc_hubble_diagram(
        tmp_path / "BIASCOR0", 0.2, 1.0, size=125,
    )
    hd1, info1 = _write_hdibc_hubble_diagram(
        tmp_path / "BIASCOR1",
        0.4,
        -1.0,
        redshift_offset=2.0e-4,
        size=125,
    )
    cospar = tmp_path / "SNANA_UNBINNED_HDIBC.cospar"
    _run_snana_wfit(
        tmp_path,
        [
            f"{hd0},{hd1}",
            "-lcdm",
            "-omsteps",
            "41",
            "-outfile_cospar",
            str(cospar),
        ],
    )
    reference = _read_cospar(cospar)

    config = WFITConfig()
    config.hubble_diagram = [str(hd0), str(hd1)]
    config.model = "lcdm"
    config.hdibc = True
    config.hdibc_info = [str(info0), str(info1)]
    config.method = "grid"
    config.omsteps = 41
    result = run_wfit(config, write=False)[0]

    assert result["ndof_hd"] == 123
    assert result["point_estimate"]["omega_m"] == pytest.approx(
        reference["om"], abs=0.25 * 0.025,
    )
    assert _sigma68(result, "omega_m") == pytest.approx(
        reference["omsig_marg"], rel=0.02,
    )


@pytest.mark.parametrize(
    "model,model_options,steps,reference_names",
    [
        (
            "wcdm",
            [],
            (21, 41, 1),
            {"omega_m": ("om", "omsig_marg"), "w0": ("w", "wsig_marg")},
        ),
        (
            "w0wa",
            ["-wa"],
            (11, 21, 11),
            {
                "omega_m": ("om", "omsig_marg"),
                "w0": ("w0", "w0sig_marg"),
                "wa": ("wa", "wasig_marg"),
            },
        ),
    ],
)
def test_wfit_hdibc_models_against_live_snana(
    tmp_path, model, model_options, steps, reference_names,
):
    hd0, info0 = _write_hdibc_hubble_diagram(
        tmp_path / "BIASCOR0", 0.2, 1.0, mushift=0.03,
    )
    hd1, info1 = _write_hdibc_hubble_diagram(
        tmp_path / "BIASCOR1", 0.4, -1.0, mushift=-0.02,
    )
    omega_steps, w0_steps, wa_steps = steps
    cospar = tmp_path / f"SNANA_HDIBC_{model}.cospar"
    _run_snana_wfit(
        tmp_path,
        [
            f"{hd0},{hd1}",
            *model_options,
            "-cmb",
            "-omsteps",
            str(omega_steps),
            "-wsteps",
            str(w0_steps),
            "-wasteps",
            str(wa_steps),
            "-outfile_cospar",
            str(cospar),
        ],
    )
    reference = _read_cospar(cospar)

    config = WFITConfig()
    config.hubble_diagram = [str(hd0), str(hd1)]
    config.model = model
    config.hdibc = True
    config.hdibc_info = [str(info0), str(info1)]
    config.cmb = True
    config.dompri = 0.9
    config.method = "grid"
    config.omsteps = omega_steps
    config.wsteps = w0_steps
    config.wasteps = wa_steps
    result = run_wfit(config, write=False)[0]

    grid_step = {
        "omega_m": 1.0 / (omega_steps - 1),
        "w0": 2.0 / (w0_steps - 1),
        "wa": 8.0 / (wa_steps - 1) if wa_steps > 1 else 0.0,
    }
    for name, (value_name, sigma_name) in reference_names.items():
        assert result["point_estimate"][name] == pytest.approx(
            reference[value_name], abs=0.25 * grid_step[name],
        )
        assert _sigma68(result, name) == pytest.approx(
            reference[sigma_name], rel=0.03,
        )


@pytest.mark.parametrize("prior", ("cmb_sim", "bao_sim"))
def test_wfit_simulation_centered_priors_against_live_snana(
    tmp_path, repo_root, prior,
):
    fixture = repo_root / "tests" / "fixtures" / "wfit" / "SALT2mu.M0DIF"
    cospar = tmp_path / f"SNANA_{prior}.cospar"
    command = [
        str(fixture),
        "-om_sim",
        "0.3",
        "-w0_sim",
        "-0.9",
        "-wa_sim",
        "0.2",
        "-outfile_cospar",
        str(cospar),
    ]
    if prior == "cmb_sim":
        command.append("-cmb_sim")
    else:
        bao_root = Path(os.environ["SNDATA_ROOT"]) / "models" / "BAO"
        if not (bao_root / "DESI_2024_mean.dat").exists():
            pytest.skip("DESI_2024 BAO resources are unavailable")
        command.extend(["-bao_sim", "DESI_2024"])
    _run_snana_wfit(tmp_path, command)
    reference = _read_cospar(cospar)

    config = WFITConfig()
    config.hubble_diagram = str(fixture)
    config.om_sim = 0.3
    config.w0_sim = -0.9
    config.wa_sim = 0.2
    config.method = "grid"
    if prior == "cmb_sim":
        config.cmb = True
        config.cmb_sim = True
        config.dompri = 0.9
    else:
        config.bao = "DESI_2024"
        config.bao_sim = "DESI_2024"
        config.ompri_enabled = False
    result = run_wfit(config, write=False)[0]

    assert result["point_estimate"]["w0"] == pytest.approx(
        reference["w"], abs=0.25 * 0.01,
    )
    assert result["point_estimate"]["omega_m"] == pytest.approx(
        reference["om"], abs=0.25 * 0.0125,
    )
    assert _sigma68(result, "w0") == pytest.approx(
        reference["wsig_marg"], rel=0.02,
    )
    assert _sigma68(result, "omega_m") == pytest.approx(
        reference["omsig_marg"], rel=0.02,
    )


def test_wfit_blinding_against_live_snana(tmp_path, repo_root):
    fixture = repo_root / "tests" / "fixtures" / "wfit" / "SALT2mu.M0DIF"
    snana_cospar = tmp_path / "SNANA_BLIND.cospar"
    snana_grid = tmp_path / "SNANA_BLIND.grid"
    snana_prob1d = tmp_path / "SNANA_BLIND.prob1d"
    _run_snana_wfit(
        tmp_path,
        [
            str(fixture),
            "-blind",
            "-blind_seed",
            "48901",
            "-omsteps",
            "21",
            "-wsteps",
            "41",
            "-weightmin",
            "0",
            "-outfile_cospar",
            str(snana_cospar),
            "-outfile_chi2grid",
            str(snana_grid),
            "-outfile_prob1d",
            str(snana_prob1d),
        ],
    )

    config = WFITConfig()
    config.hubble_diagram = str(fixture)
    config.method = "grid"
    config.omsteps = 21
    config.wsteps = 41
    config.blind_mode = "blind"
    config.blind_seed = 48901
    config.cospar_yaml = str(tmp_path / "JAX_BLIND.yaml")
    config.outfile_cospar = str(tmp_path / "JAX_BLIND.cospar")
    config.outfile_chi2grid = str(tmp_path / "JAX_BLIND.grid")
    config.outfile_prob1d = str(tmp_path / "JAX_BLIND.prob1d")
    run_wfit(config)

    reference = _read_cospar(snana_cospar)
    actual = _read_cospar(Path(config.outfile_cospar))
    expected_random = parameter_random_integers(config)
    assert int(reference["wran"]) == expected_random["w0"]
    assert int(reference["omran"]) == expected_random["omega_m"]
    assert actual["w"] == pytest.approx(reference["w"], abs=0.25 * 0.05)
    assert actual["om"] == pytest.approx(reference["om"], abs=0.25 * 0.05)

    reference_grid = _read_first_table_row(snana_grid)
    actual_grid = _read_first_table_row(Path(config.outfile_chi2grid))
    assert actual_grid["w0"] == pytest.approx(reference_grid["w0"])
    assert actual_grid["omm"] == pytest.approx(reference_grid["omm"])

    reference_prob1d = _read_first_table_row(snana_prob1d)
    actual_prob1d = _read_first_table_row(Path(config.outfile_prob1d))
    assert actual_prob1d["VAL"] == pytest.approx(reference_prob1d["VAL"])


def test_wfit_w0wa_blinding_sequence_against_live_snana(
    tmp_path, repo_root,
):
    fixture = repo_root / "tests" / "fixtures" / "wfit" / "SALT2mu.M0DIF"
    cospar = tmp_path / "SNANA_W0WA_BLIND.cospar"
    _run_snana_wfit(
        tmp_path,
        [
            str(fixture),
            "-wa",
            "-blind",
            "-blind_seed",
            "48901",
            "-omsteps",
            "11",
            "-wsteps",
            "21",
            "-wasteps",
            "11",
            "-outfile_cospar",
            str(cospar),
        ],
    )
    reference = _read_cospar(cospar)
    config = WFITConfig()
    config.model = "w0wa"
    expected = parameter_random_integers(config)
    assert int(reference["w0ran"]) == expected["w0"]
    assert int(reference["waran"]) == expected["wa"]
    assert int(reference["omran"]) == expected["omega_m"]


@pytest.mark.parametrize("is_data_real", (0, 1))
def test_wfit_automatic_blinding_against_live_snana(
    tmp_path, repo_root, is_data_real,
):
    fixture = (
        repo_root / "tests" / "fixtures" / "wfit" / "SALT2mu.M0DIF"
    ).read_text()
    hd = tmp_path / "AUTO_BLIND.M0DIF"
    hd.write_text(f"# ISDATA_REAL: {is_data_real}\n" + fixture)
    cospar = tmp_path / "SNANA_AUTO_BLIND.cospar"
    _run_snana_wfit(
        tmp_path,
        [
            str(hd),
            "-lcdm",
            "-blind_auto",
            "-omsteps",
            "21",
            "-outfile_cospar",
            str(cospar),
        ],
    )
    reference = _read_cospar(cospar)

    config = WFITConfig()
    config.hubble_diagram = str(hd)
    config.model = "lcdm"
    config.method = "grid"
    config.omsteps = 21
    config.blind_mode = "auto"
    run_wfit(config, write=False)

    diagrams = load_hubble_diagrams(config)
    assert blinding_statuses(config, diagrams) == (bool(is_data_real),)
    expected_random = (
        parameter_random_integers(config)["omega_m"]
        if is_data_real else 0
    )
    assert int(reference["omran"]) == expected_random


@pytest.mark.parametrize("use_covariance", (False, True))
def test_padded_realisations_and_ensemble_against_live_snana(
    tmp_path, use_covariance,
):
    sizes = (37, 53)
    hubble_diagrams = []
    covariance_files = []
    truth_files = []
    references = []
    for index, size in enumerate(sizes):
        hd = tmp_path / f"HD_{index}.FITRES"
        _write_unbinned_hubble_diagram(hd, size=size)
        hubble_diagrams.append(str(hd))
        info = tmp_path / f"INFO_{index}.YML"
        info.write_text(
            "MUSHIFT: 0.0\n"
            "OMEGA_MATTER: 0.3\n"
            "OMEGA_LAMBDA: 0.7\n"
            "w0_LAMBDA: -1.0\n"
            "wa_LAMBDA: 0.0\n"
        )
        truth_files.append(str(info))
        covariance_options = []
        if use_covariance:
            covariance = np.eye(size) * (1.0e-4 + index * 2.0e-5)
            adjacent = np.arange(size - 1)
            covariance[adjacent, adjacent + 1] = 2.0e-5
            covariance[adjacent + 1, adjacent] = 2.0e-5
            covariance_path = tmp_path / f"COVSYS_{index}.txt"
            _write_covariance(covariance_path, covariance)
            covariance_files.append(str(covariance_path))
            covariance_options = [
                "-mucovsys_file",
                str(covariance_path),
            ]
        cospar = tmp_path / f"SNANA_{index}.cospar"
        _run_snana_wfit(
            tmp_path,
            [
                str(hd),
                "-omsteps",
                "21",
                "-wsteps",
                "41",
                "-sig_type",
                "std",
                *covariance_options,
                "-outfile_cospar",
                str(cospar),
            ],
        )
        references.append(_read_cospar(cospar))

    config = WFITConfig()
    config.sig_type = "std"     # as wfit.exe was told above
    config.hubble_diagram = hubble_diagrams
    config.truth_info = truth_files
    if use_covariance:
        config.covariance = covariance_files
        config.covariance_kind = "systematics"
    config.method = "grid"
    config.omsteps = 21
    config.wsteps = 41
    diagrams = load_hubble_diagrams(config)
    results = run_wfit(config, write=False)

    assert tuple(result["ndof_hd"] for result in results) == (34, 50)
    for result, reference in zip(results, references):
        assert result["point_estimate"]["w0"] == pytest.approx(
            reference["w"], abs=0.25 * 0.05,
        )
        assert result["point_estimate"]["omega_m"] == pytest.approx(
            reference["om"], abs=0.25 * 0.05,
        )
        assert result["summaries"]["w0"]["std"] == pytest.approx(
            reference["wsig_marg"], rel=0.02,
        )
        assert result["summaries"]["omega_m"]["std"] == pytest.approx(
            reference["omsig_marg"], rel=0.02,
        )

    for name, value_key, sigma_key, truth, step in (
        ("w0", "w", "wsig_marg", -1.0, 0.05),
        ("omega_m", "om", "omsig_marg", 0.3, 0.05),
    ):
        biases = np.asarray([
            reference[value_key] - truth for reference in references
        ])
        pulls = np.asarray([
            bias / reference[sigma_key]
            for bias, reference in zip(biases, references)
        ])
        ensemble = ensemble_summary(results, diagrams, config)[name]
        expected_bias_std = np.std(biases, ddof=1)
        assert ensemble["mean_bias"] == pytest.approx(
            np.mean(biases), abs=0.25 * step,
        )
        assert ensemble["bias_std"] == pytest.approx(
            expected_bias_std, abs=0.5 * step,
        )
        assert ensemble["standard_error"] == pytest.approx(
            expected_bias_std / np.sqrt(len(sizes)), abs=0.5 * step,
        )
        assert ensemble["pull_mean"] == pytest.approx(
            np.mean(pulls), abs=0.05,
        )
        assert ensemble["pull_std"] == pytest.approx(
            np.std(pulls, ddof=1), abs=0.10,
        )
