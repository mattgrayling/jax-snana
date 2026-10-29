"""WFIT prior resources, configuration, and output tests."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from jax_snlc_core.logging import SNANAError
from jax_wfit.blinding import parameter_offsets, parameter_random_integers
from jax_wfit.config import WFITConfig, build, build_parser, cli_keys, load_config, read_keys
from jax_wfit.hubble import load_hubble_diagrams, stack_arrays
from jax_wfit.likelihood import evaluate_likelihood
from jax_wfit.priors import load_bao_data, prepare_priors
from jax_wfit.run import run_wfit


def _fixture(repo_root: Path) -> str:
    return str(repo_root / "tests" / "fixtures" / "wfit" / "SALT2mu.M0DIF")


def _read_matrix(path: Path) -> np.ndarray:
    values = path.read_text().split()
    size = int(values[0])
    return np.asarray(values[1:], dtype=np.float64).reshape(size, size)


def test_flat_yaml_configuration(tmp_path):
    path = tmp_path / "wfit.yaml"
    path.write_text(
        "hubble_diagram: HD.M0DIF\n"
        "model: lcdm\n"
        "ommin: 0.1\n"
        "ommax: 0.6\n"
        "method: grid\n"
        "omsteps: 31\n"
    )
    config = load_config(path)
    assert config.model == "lcdm"
    assert config.bounds("omega_m") == (0.1, 0.6)
    assert config.omsteps == 31


def test_unknown_named_yaml_key_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("hubble_diagram: HD\nphysics_default: hidden\n")
    with pytest.raises(SNANAError, match="physics_default"):
        load_config(path)


def test_prior_flags_follow_snana_and_yaml_composes(tmp_path):
    flags = build(cli_keys(
        build_parser().parse_args(["HD", "-bao", "DESI_2024", "-cmb"])
    ))
    assert not flags.omega_m_prior_applies()  # -bao replaces it
    assert flags.bao == "DESI_2024"
    assert flags.cmb

    # -cmb on its own applies WFIT's weak sigma(Omega_m)=0.9 preset.
    preset = build(cli_keys(build_parser().parse_args(["HD", "-cmb"])))
    assert cli_keys(build_parser().parse_args(["HD", "-h0", "68.5"]))["h0"] == 68.5
    assert preset.cmb
    assert preset.ompri_enabled
    assert preset.dompri == 0.9

    path = tmp_path / "composed.yaml"
    path.write_text(
        "hubble_diagram: HD\n"
        "ompri_enabled: true\n"
        "dompri: 0.03\n"
        "bao: DESI_2024\n"
        "cmb: true\n"
    )
    composed = load_config(path)
    assert composed.ompri_enabled
    assert composed.dompri == 0.03
    assert composed.bao == "DESI_2024"
    assert composed.cmb


def _bao_files(tmp_path: Path) -> Path:
    """A two-row DESI_2024 resource pair, in the layout SNANA lays it out in."""
    root = tmp_path / "SNDATA_ROOT" / "models" / "BAO"
    root.mkdir(parents=True)
    (root / "DESI_2024_mean.dat").write_text(
        "0.5 10.0 DM_over_rs\n"
        "0.5 20.0 DH_over_rs\n"
    )
    (root / "DESI_2024_cov.dat").write_text("1.0 0.0\n0.0 4.0\n")
    return root


def test_bao_planck_covariance_term(tmp_path):
    config = WFITConfig()
    config.bao = "DESI_2024"
    data = load_bao_data(config, _bao_files(tmp_path))
    # SNANA tacks the Planck H0 and r_d errors onto the BAO covariance rather
    # than floating them (init_bao_prior, wfit.c:2628-2631 and :2738), using
    # H0 = 67.4 +- 0.5 and r_d = 147.05 +- 0.30 (sntools.h:61-64). For this
    # two-row file that gives (0.5/67.4)^2 + (0.30/147.05)^2 = 5.919472e-5
    # times the outer product of the means [10, 20].
    expected = np.array([[1.005919472278, 0.011838944555],
                         [0.011838944555, 4.023677889110]])
    np.testing.assert_allclose(
        np.linalg.inv(np.asarray(data["bao_precision"])), expected, rtol=1e-6)


def test_simulation_centred_cmb_prior_is_zero_at_truth(repo_root):
    config = WFITConfig()
    config.hubble_diagram = _fixture(repo_root)
    config.om_sim = 0.3
    config.w0_sim = -0.9
    config.wa_sim = 0.2
    config.model = "w0wa"
    config.cmb = True
    config.cmb_sim = True
    data = load_hubble_diagrams(config)
    priors = prepare_priors(config, [d.truth for d in data], None)

    components = evaluate_likelihood(
        {"omega_m": 0.3, "w0": -0.9, "wa": 0.2},
        *stack_arrays(data)[:4],
        priors["omega_m_mean"],
        priors["cmb_r"],
        config,
        priors,
    )

    assert float(components["cmb_chi2"][0]) == pytest.approx(0.0, abs=1.0e-20)


@pytest.mark.parametrize("radiation", [False, True])
def test_simulation_centred_bao_prior_is_zero_at_truth(tmp_path, repo_root, radiation):
    root = _bao_files(tmp_path)
    config = WFITConfig()
    config.hubble_diagram = _fixture(repo_root)
    config.om_sim = 0.3
    config.w0_sim = -1.0
    config.wa_sim = 0.0
    config.radiation = radiation
    config.bao = "DESI_2024"
    config.bao_sim = "DESI_2024"
    config.rd_prior = config.bao_rd
    data = load_hubble_diagrams(config)
    priors = prepare_priors(config, [d.truth for d in data], root)

    components = evaluate_likelihood(
        {"omega_m": 0.3, "w0": -1.0},
        *stack_arrays(data)[:4],
        priors["omega_m_mean"],
        priors["cmb_r"],
        config,
        priors,
    )

    assert float(components["bao_chi2"]) == pytest.approx(0.0, abs=1.0e-20)


def test_prior_perturbations_share_one_draw_and_reproduce():
    """The omm and Rcmb shifts come from the same per-realisation gaussian
    draw (SNANA re-seeds per call, get_prior_ranshift wfit.c:2548), differ
    across realisations, and repeat.

    cmb_sim is set because that is the only mode SNANA perturbs Rcmb in
    (wfit.c:2489); without it the Rcmb draw would not be exercised at all.
    """
    truths = [{"omega_m": 0.3, "w0": -1.0, "wa": 0.0}] * 3

    def priors_for(perturb):
        config = WFITConfig()
        config.ompri = 0.3
        config.dompri = 0.02
        config.cmb = True
        config.cmb_sim = True
        config.sigma_Rcmb = 0.019
        config.perturb_priors = perturb
        config.prior_perturbation_seed = 13579
        return prepare_priors(config, truths, None)

    priors = priors_for(True)
    repeated = priors_for(True)
    unperturbed = priors_for(False)

    omega_pull = (
        np.asarray(priors["omega_m_mean"])
        - np.asarray(unperturbed["omega_m_mean"])
    ) / 0.02
    cmb_pull = (
        np.asarray(priors["cmb_r"]) - np.asarray(unperturbed["cmb_r"])
    ) / 0.019
    np.testing.assert_allclose(omega_pull, cmb_pull, rtol=1e-6)
    np.testing.assert_array_equal(
        repeated["omega_m_mean"], priors["omega_m_mean"],
    )
    np.testing.assert_array_equal(repeated["cmb_r"], priors["cmb_r"])
    assert len(np.unique(np.asarray(priors["omega_m_mean"]))) == 3
    assert len(np.unique(np.asarray(priors["cmb_r"]))) == 3
    assert not np.array_equal(priors["cmb_r"], unperturbed["cmb_r"])


def test_prior_perturbation_targets_are_used_per_realisation(repo_root):
    fixture = _fixture(repo_root)
    config = WFITConfig()
    config.hubble_diagram = [fixture, fixture]
    config.model = "lcdm"
    config.method = "grid"
    config.omsteps = 101
    config.ompri = 0.3
    config.dompri = 0.005
    config.perturb_priors = True
    config.prior_perturbation_seed = 97531

    results = run_wfit(config, write=False)

    assert (
        results[0]["prior_targets"]["omega_m"]
        != results[1]["prior_targets"]["omega_m"]
    )
    assert (
        results[0]["summaries"]["omega_m"]["mean"]
        != results[1]["summaries"]["omega_m"]["mean"]
    )


def test_all_optional_outputs(tmp_path, repo_root):
    config = WFITConfig()
    config.hubble_diagram = _fixture(repo_root)
    config.model = "lcdm"
    config.method = "grid"
    config.omsteps = 21
    config.cospar_yaml = str(tmp_path / "result.yaml")
    config.outfile_resid = str(tmp_path / "residuals.fitres")
    config.outfile_cospar = str(tmp_path / "result.cospar")
    config.outfile_chi2grid = str(tmp_path / "grid.fitres")
    config.outfile_prob1d = str(tmp_path / "prob.fitres")
    config.output_covariance = str(tmp_path / "covariance.txt")
    config.outfile_mucovtot_inv = str(tmp_path / "precision.txt")
    result = run_wfit(config)[0]
    assert result["diagnostics"]["success"]
    payload = yaml.safe_load((tmp_path / "result.yaml").read_text())
    assert payload["assumptions"]["radiation"] is True
    assert payload["assumptions"]["cmb_radiation"] is False
    assert payload["assumptions"]["bao_dh_radiation"] is False
    assert payload["fit"]["ndof_hd"] == 8
    assert "VARNAMES:" in (tmp_path / "residuals.fitres").read_text()
    cospar = (tmp_path / "result.cospar").read_text()
    # the wfit.c column set (write_output_cospar, wfit.c:5162-5277)
    for column in ("rho_wom", "FoM", "Ndof", "sigint", "omran", "label"):
        assert column in cospar
    assert "weight" in (tmp_path / "grid.fitres").read_text()
    assert "CDF" in (tmp_path / "prob.fitres").read_text()
    covariance = _read_matrix(tmp_path / "covariance.txt")
    precision = _read_matrix(tmp_path / "precision.txt")
    np.testing.assert_allclose(covariance @ precision, np.eye(10))


def test_cospar_fixed_param_columns_follow_the_marg_mode(tmp_path, repo_root):
    """The fixed w under lcdm gets wsig_lo/wsig_up in minchi2 mode and
    wsig_marg in marg mode, like any floated parameter (write_output_cospar,
    wfit.c:5168-5180); omm alone always writes sig_marg (:5214-5219)."""
    columns = {}
    for marg in (False, True):
        config = WFITConfig()
        config.hubble_diagram = _fixture(repo_root)
        config.model = "lcdm"
        config.method = "grid"
        config.omsteps = 11
        config.marg = marg
        config.cospar_yaml = str(tmp_path / f"m{marg}.yaml")
        config.outfile_cospar = str(tmp_path / f"m{marg}.cospar")
        run_wfit(config)
        header = Path(config.outfile_cospar).read_text().splitlines()[0]
        columns[marg] = header.split()
    assert {"wsig_lo", "wsig_up"} <= set(columns[False])
    assert "wsig_marg" not in columns[False]
    assert "wsig_marg" in columns[True]
    assert "omsig_marg" in columns[False] and "omsig_marg" in columns[True]


def test_cospar_written_by_default_with_cpu_minutes(tmp_path, repo_root):
    """wfit writes the cospar unconditionally, named from the HD input when
    no -outfile_cospar is given (write_output_cospar, wfit.c:5151-5153), and
    the YAML summary carries CPU_MINUTES for the batch aggregator
    (wfit.c:5314)."""
    hd = tmp_path / "HD.M0DIF"
    hd.write_text(Path(_fixture(repo_root)).read_text())
    config = WFITConfig()
    config.hubble_diagram = str(hd)
    config.model = "lcdm"
    config.method = "grid"
    config.omsteps = 21
    run_wfit(config)

    cospar = (tmp_path / "HD.M0DIF.cospar").read_text().splitlines()
    assert cospar[0].startswith("# ") and "om" in cospar[0].split()
    assert len(cospar[1].split()) == len(cospar[0].split()) - 1
    payload = yaml.safe_load((tmp_path / "HD.M0DIF.wfit.yaml").read_text())
    assert payload["CPU_MINUTES"] > 0.0


def test_multi_realisation_outputs_are_unambiguous(tmp_path, repo_root):
    fixture = _fixture(repo_root)
    config = WFITConfig()
    config.hubble_diagram = [fixture, fixture]
    config.model = "lcdm"
    config.method = "grid"
    config.omsteps = 11
    config.cospar_yaml = str(tmp_path / "result.yaml")
    config.outfile_resid = str(tmp_path / "residuals.fitres")
    config.outfile_cospar = str(tmp_path / "result.cospar")
    config.output_covariance = str(tmp_path / "covariance.txt")
    config.outfile_mucovtot_inv = str(tmp_path / "precision.txt")

    result = run_wfit(config)

    assert len(result) == 2
    payload = yaml.safe_load((tmp_path / "result.yaml").read_text())
    assert payload["n_realisations"] == 2
    assert len(payload["realisations"]) == 2
    assert "configuration" in payload
    assert "configuration" not in payload["realisations"][0]
    for index in range(2):
        assert payload["realisations"][index]["input"][
            "realisation_index"
        ] == index
        assert (tmp_path / f"residuals.r{index:03d}.fitres").exists()
        assert (tmp_path / f"result.r{index:03d}.cospar").exists()
        assert (tmp_path / f"covariance.r{index:03d}.txt").exists()
        assert (tmp_path / f"precision.r{index:03d}.txt").exists()


def test_cli_configures_requested_diagnostic_features():
    config = WFITConfig()
    args = build_parser().parse_args([
        "HD",
        "-muerr_ideal",
        "0.08,0.04",
        "--prior-perturbation-seed",
        "33",
        "--output-covariance",
        "cov.txt",
        "-outfile_mucovtot_inv",
        "precision.txt",
    ])

    config = build(cli_keys(args))

    assert config.muerr_ideal
    assert config.muerr_ideal_polynomial == (0.08, 0.04)
    assert config.perturb_priors
    assert config.prior_perturbation_seed == 33
    assert config.output_covariance == "cov.txt"
    assert config.outfile_mucovtot_inv == "precision.txt"


def test_one_truth_file_serves_every_realisation_on_the_command_line():
    """One --truth-info path is shared by every Hubble diagram, as the
    input file's string form is; several must match the diagrams."""
    one = build(cli_keys(build_parser().parse_args(
        ["a.M0DIF", "b.M0DIF", "--truth-info", "INFO.YML"])))
    assert one.truth_info == "INFO.YML"
    two = build(cli_keys(build_parser().parse_args(
        ["a.M0DIF", "b.M0DIF", "--truth-info", "A.YML", "B.YML"])))
    assert two.truth_info == ["A.YML", "B.YML"]


def test_info_yml_truth_produces_ensemble_bias_summary(tmp_path, repo_root):
    info = tmp_path / "INFO.YML"
    info.write_text(
        "MUSHIFT: 0.0\n"
        "OMEGA_MATTER: 0.3\n"
        "OMEGA_LAMBDA: 0.7\n"
        "w0_LAMBDA: -1.0\n"
        "wa_LAMBDA: 0.0\n"
    )
    fixture = _fixture(repo_root)
    config = WFITConfig()
    config.hubble_diagram = [fixture, fixture]
    config.truth_info = str(info)
    config.model = "lcdm"
    config.method = "grid"
    config.omsteps = 21
    config.cospar_yaml = str(tmp_path / "result.yaml")
    config.outfile_cospar = str(tmp_path / "result.cospar")

    results = run_wfit(config)

    recovered = results[0]["summaries"]["omega_m"]["mean"]
    payload = yaml.safe_load((tmp_path / "result.yaml").read_text())
    ensemble = payload["ensemble"]["omega_m"]
    assert ensemble["count"] == 2
    assert ensemble["mean_bias"] == pytest.approx(recovered - 0.3)
    assert ensemble["bias_std"] == pytest.approx(0.0, abs=1.0e-15)
    assert ensemble["standard_error"] == pytest.approx(0.0, abs=1.0e-15)
    expected_pull = (
        (recovered - 0.3)
        / results[0]["summaries"]["omega_m"]["sigma"]
    )
    assert ensemble["pull_mean"] == pytest.approx(expected_pull)
    assert ensemble["pull_std"] == pytest.approx(0.0, abs=1.0e-15)
    expected_coverage = float(
        results[0]["summaries"]["omega_m"]["q16"]
        <= 0.3
        <= results[0]["summaries"]["omega_m"]["q84"]
    )
    assert ensemble["coverage_68"] == expected_coverage
    payload = yaml.safe_load((tmp_path / "result.yaml").read_text())
    assert payload["ensemble"]["omega_m"]["count"] == 2
    assert payload["realisations"][0]["input"]["simulation_truth"][
        "omega_m"
    ] == 0.3


def test_explicit_blinding_shifts_parameter_outputs(tmp_path, repo_root):
    config = WFITConfig()
    config.hubble_diagram = _fixture(repo_root)
    config.model = "lcdm"
    config.method = "grid"
    config.omsteps = 11
    config.blind_mode = "blind"
    config.cospar_yaml = str(tmp_path / "result.yaml")
    config.outfile_cospar = str(tmp_path / "result.cospar")
    config.outfile_chi2grid = str(tmp_path / "grid.fitres")
    config.outfile_prob1d = str(tmp_path / "prob.fitres")

    result = run_wfit(config)[0]

    offset = parameter_offsets(config)["omega_m"]
    payload = yaml.safe_load((tmp_path / "result.yaml").read_text())
    assert payload["blinded"] is True
    assert payload["configuration"]["blind_seed"] == "REDACTED"
    assert payload["parameters"]["omega_m"]["mean"] == pytest.approx(
        result["summaries"]["omega_m"]["mean"] + offset,
    )
    cospar_lines = (tmp_path / "result.cospar").read_text().splitlines()
    om_index = cospar_lines[0].split()[1:].index("om")
    cospar_value = float(cospar_lines[1].split()[om_index])
    assert cospar_value == pytest.approx(
        result["summaries"]["omega_m"]["mean"] + offset, abs=5.0e-6,
    )
    grid_first = next(
        line for line in (tmp_path / "grid.fitres").read_text().splitlines()
        if line.startswith("ROW:")
    )
    random_integer = parameter_random_integers(config)["omega_m"]
    assert float(grid_first.split()[2]) == pytest.approx(
        random_integer, abs=0.5)
    prob_first = (tmp_path / "prob.fitres").read_text().splitlines()[1]
    assert float(prob_first.split()[4]) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "model,expected",
    [
        ("lcdm", {"omega_m": 382717}),
        ("wcdm", {"w0": 382717, "omega_m": 329377}),
        (
            "w0wa",
            {"w0": 382717, "wa": 329377, "omega_m": 855860},
        ),
    ],
)
def test_blinding_integers_match_darwin_snana(model, expected):
    config = WFITConfig()
    config.model = model
    assert parameter_random_integers(config) == expected


def test_auto_blinding_uses_isdata_real_header(tmp_path, repo_root):
    fixture = Path(_fixture(repo_root)).read_text()
    real = tmp_path / "REAL.M0DIF"
    simulation = tmp_path / "SIM.M0DIF"
    real.write_text("# ISDATA_REAL: 1\n" + fixture)
    simulation.write_text("# ISDATA_REAL: 0\n" + fixture)
    config = WFITConfig()
    config.hubble_diagram = [str(real), str(simulation)]
    config.model = "lcdm"
    config.method = "grid"
    config.omsteps = 11
    config.blind_mode = "auto"
    config.cospar_yaml = str(tmp_path / "result.yaml")

    results = run_wfit(config)

    assert [result["blinded"] for result in results] == [True, False]
    payload = yaml.safe_load((tmp_path / "result.yaml").read_text())
    assert [
        item["blinded"] for item in payload["realisations"]
    ] == [True, False]


def test_blinding_refuses_a_parameter_without_an_offset():
    """A missing offset must raise, not silently write an unblinded value."""
    from jax_wfit.blinding import apply_blinding

    config = WFITConfig()
    config.model = "lcdm"
    with pytest.raises(KeyError):
        apply_blinding({"omega_m": 0.3, "w0": -1.0}, config, blinded=True)
    unblinded = apply_blinding({"omega_m": 0.3, "w0": -1.0}, config, False)
    assert unblinded == {"omega_m": 0.3, "w0": -1.0}


def test_prior_perturbation_only_moves_an_informative_prior():
    """SNANA perturbs the Omega_m prior only when dompri < 1 (wfit.c:2408)."""

    def centre(dompri, **kwargs):
        config = WFITConfig()
        config.perturb_priors = True
        config.dompri = dompri
        for key, value in kwargs.items():
            setattr(config, key, value)
        return np.asarray(prepare_priors(config, [None], None)["omega_m_mean"])[0]

    assert centre(99.0) == pytest.approx(WFITConfig().ompri)
    assert centre(0.03) != pytest.approx(WFITConfig().ompri)


def test_rcmb_perturbation_requires_a_simulation_centred_prior():
    """SNANA shifts Rcmb only in the use_cmb == 2 branch (wfit.c:2489)."""

    truth = {"omega_m": 0.3, "w0": -1.0, "wa": 0.0}

    def target(cmb_sim):
        config = WFITConfig()
        config.perturb_priors = True
        config.cmb = True
        config.cmb_sim = cmb_sim
        config.dompri = 0.9
        return np.asarray(prepare_priors(config, [truth], None)["cmb_r"])[0]

    # without cmb_sim the target is the fixed Rcmb, unperturbed
    assert target(False) == pytest.approx(WFITConfig().Rcmb)
    # with it, the target is recomputed from truth and then perturbed
    assert target(True) != pytest.approx(target(False))


def test_snana_aliases_reach_their_fields():
    """wfit.exe's own spellings, including the ones argparse used to swallow."""
    args = build_parser().parse_args([
        "HD", "-cospar", "c.txt", "-resid", "r.txt",
        "-mucov_file", "m.txt", "-ranseed_Rcmb", "7",
    ])
    config = build(cli_keys(args))
    assert config.outfile_cospar == "c.txt"
    assert config.outfile_resid == "r.txt"
    assert config.covariance == "m.txt"
    assert config.covariance_kind == "systematics"
    assert config.prior_perturbation_seed == 7
    assert config.perturb_priors


def test_abbreviated_flags_are_refused():
    """-cospar used to be an abbreviation of -cospar_yaml; exact match only."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["HD", "-cospar_ya", "x.yaml"])


def test_unimplemented_options_abort_with_a_reason():
    """Unimplemented keys fail loudly, as the sim and SALT2mu parsers do.

    The command line refuses while it is translated, so a caller that only
    translates flags refuses too, and the YAML refuses in build.
    """
    for flag in ("-csv", "-refit", "-weightmin", "-ndump_mucov"):
        with pytest.raises(SNANAError, match="not supported"):
            cli_keys(build_parser().parse_args(["HD", flag, "1"]))


def test_a_yaml_naming_an_unimplemented_option_aborts_with_the_same_reason(tmp_path):
    """The refusal is on the key, not on the route that named it."""
    path = tmp_path / "wfit.yaml"
    path.write_text("hubble_diagram: HD\nndump_mucov: 3\n")

    with pytest.raises(SNANAError, match="ndump_mucov:.*diagnostic covariance dumps") as yaml_error:
        load_config(path)
    with pytest.raises(SNANAError) as cli_error:
        cli_keys(build_parser().parse_args(["HD", "-ndump_mucov", "3"]))
    assert str(yaml_error.value) == str(cli_error.value)


def test_ranseed_rcmb_zero_disables_the_perturbation():
    config = build(cli_keys(build_parser().parse_args(["HD", "-ranseed_Rcmb", "0"])))
    assert not config.perturb_priors


def test_marg_and_minchi2_select_the_reported_estimate(repo_root):
    """Both estimates stay available; -marg picks which one is reported."""
    def run(extra):
        config = build(cli_keys(build_parser().parse_args(
            [_fixture(repo_root), "--method", "grid", "-omsteps", "21"] + extra
        )))
        return run_wfit(config, write=False)[0]["summaries"]["omega_m"]

    default, best_fit = run(["-marg"]), run(["-minchi2"])
    assert default["value"] == pytest.approx(default["mean"])
    assert best_fit["value"] == pytest.approx(best_fit["center"])
    assert default["value"] != pytest.approx(best_fit["value"])


def test_input_file_method_survives_a_grid_output_flag(tmp_path):
    """The grid implication fires only when no source named a method."""
    path = tmp_path / "wfit.yaml"
    path.write_text("hubble_diagram: HD\nmethod: nuts\n")
    raw = read_keys(path)
    raw.update(cli_keys(build_parser().parse_args(["-outfile_chi2grid", "g.txt"])))
    assert build(raw).method == "nuts"
    assert build(cli_keys(
        build_parser().parse_args(["HD", "-outfile_chi2grid", "g.txt"])
    )).method == "grid"


def test_an_unset_flag_does_not_overwrite_the_input_file(tmp_path):
    path = tmp_path / "wfit.yaml"
    path.write_text("hubble_diagram: HD\nhdibc: true\nperturb_priors: true\n")
    raw = read_keys(path)
    raw.update(cli_keys(build_parser().parse_args(["-zmin", "0.02"])))
    config = build(raw)
    assert config.hdibc
    assert config.perturb_priors
    assert config.zmin == 0.02


def test_every_output_reports_the_same_cosmology(repo_root):
    """The residuals file reads point_estimate; the cospar file reads the
    summary. Under laplace those used to be the mode and the sample mean."""
    config = WFITConfig()
    config.hubble_diagram = _fixture(repo_root)
    config.laplace_posterior_samples = 2000
    result = run_wfit(config, write=False)[0]
    for name, summary in result["summaries"].items():
        assert result["point_estimate"][name] == pytest.approx(
            summary["value"], abs=1.0e-12,
        )


def test_hd_derived_seed_is_stable_per_dataset(repo_root, tmp_path):
    """-ranseed_Rcmb 1 seeds from the data, so it repeats for one HD."""
    from jax_wfit.run import hd_seed

    def seed_for(path):
        config = WFITConfig()
        config.hubble_diagram = str(path)
        config.resolve_paths()
        return hd_seed(load_hubble_diagrams(config))

    original = Path(_fixture(repo_root))
    shortened = tmp_path / "short.M0DIF"
    lines = original.read_text().splitlines(keepends=True)
    shortened.write_text("".join(
        line for line in lines if not line.startswith("ROW: 10 ")
    ))

    assert seed_for(original) == seed_for(original)
    assert seed_for(original) != seed_for(shortened)



def _resid_columns(text: str) -> dict[str, np.ndarray]:
    """A residual file's numeric columns, by VARNAME."""
    lines = text.splitlines()
    varnames = next(l.split()[1:] for l in lines if l.startswith("VARNAMES:"))
    rows = [l.split()[1:] for l in lines if l.startswith("SN:")]
    return {name: np.array([float(row[i]) for row in rows])
            for i, name in enumerate(varnames) if name != "CID"}


def _hdibc_run(repo_root: Path, tmp_path: Path, **overrides) -> WFITConfig:
    """The config of the wfit.exe run recorded as hdibc/ref_plain."""
    hdibc = repo_root / "tests" / "fixtures" / "wfit" / "hdibc"
    config = WFITConfig()
    config.hubble_diagram = [str(hdibc / "HD0" / "HD.FITRES"),
                             str(hdibc / "HD1" / "HD.FITRES")]
    config.hdibc = True
    config.hdibc_info = [str(hdibc / "HD0" / "INFO.YML"),
                         str(hdibc / "HD1" / "INFO.YML")]
    config.model = "lcdm"
    config.ompri, config.dompri = 0.30, 0.05
    config.method = "grid"
    config.omsteps = 201
    config.cospar_yaml = str(tmp_path / "out.yaml")
    config.outfile_cospar = str(tmp_path / "out.cospar")
    config.outfile_resid = str(tmp_path / "out.resid")
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def test_the_residual_file_reports_snana_s_simulated_cosmology(
    repo_root, tmp_path,
):
    """mu_sim and its header line come from COSPAR_SIM, which defaults to
    OMEGA_MATTER_DEFAULT = 0.315 (wfit.c:773, :1272; sntools.h:53).

    The reference is the wfit.exe residual file recorded in
    tests/fixtures/wfit/hdibc/README.md, run with no -om_sim.
    """
    hdibc = repo_root / "tests" / "fixtures" / "wfit" / "hdibc"
    reference = (hdibc / "ref_plain.resid").read_text()
    config = _hdibc_run(repo_root, tmp_path)
    run_wfit(config)
    written = Path(config.outfile_resid).read_text()

    header = "# mu_sim   = distance computed from { OM, ODE, w0, wa } = "
    assert ([l for l in written.splitlines() if l.startswith(header)]
            == [l for l in reference.splitlines() if l.startswith(header)])
    # wfit's codist is a trapezoid where jax integrates by Gauss-Legendre, so
    # the column drifts by up to 2e-4 of a magnitude by z = 1.1; a 0.3 rather
    # than 0.315 Omega_m would move it by 0.011 at that redshift
    np.testing.assert_allclose(_resid_columns(written)["mu_sim"],
                               _resid_columns(reference)["mu_sim"],
                               atol=3.0e-4)


def test_a_blinded_residual_file_does_not_show_the_unblinded_model(
    repo_root, tmp_path,
):
    """SNANA writes the residual file at the blinded cosmology.

    write_output_resid builds its cpar from WORKSPACE.omm_final
    (wfit.c:5345-5348), which wfit_final has already offset by sin(ran)
    (wfit.c:3922-3935), so mu_model must not be the model at the fitted point.
    """
    from jax_wfit.output import write_residuals

    config = WFITConfig()
    config.hubble_diagram = _fixture(repo_root)
    config.model = "lcdm"
    config.method = "grid"
    config.omsteps = 41
    config.cospar_yaml = str(tmp_path / "plain.yaml")
    config.outfile_cospar = str(tmp_path / "plain.cospar")
    config.outfile_resid = str(tmp_path / "plain.resid")
    plain = run_wfit(config)[0]
    diagram = load_hubble_diagrams(config)[0]

    blinded_config = WFITConfig(**{**config.to_dict(),
                                   "blind_mode": "blind",
                                   "cospar_yaml": str(tmp_path / "b.yaml"),
                                   "outfile_cospar": str(tmp_path / "b.cospar"),
                                   "outfile_resid": str(tmp_path / "b.resid")})
    blinded = run_wfit(blinded_config)[0]

    plain_columns = _resid_columns(Path(config.outfile_resid).read_text())
    blinded_columns = _resid_columns(
        Path(blinded_config.outfile_resid).read_text())
    unblinded_model = plain_columns["mu_model"]
    blinded_model = blinded_columns["mu_model"]
    assert np.all(blinded_model != unblinded_model)

    # both files are internally consistent: get_chi2_fit is rerun at the
    # cosmology written, so its muresid_avg leaves the weighted mean residual
    # at zero (wfit.c:3952-3956). The tolerance is the 1e-4 the file prints:
    # Csum also carries the Gaussian H0-prior term (wfit.c:4390), so the offset
    # is not exactly the inverse-variance weighted mean and about 1e-5 is left.
    # Subtracting the unblinded offset from the blinded model leaves 0.47 mag.
    for columns in (plain_columns, blinded_columns):
        weight = 1.0 / columns["mu_sig"] ** 2
        weighted_mean = float(
            (weight * columns["mu_res"]).sum() / weight.sum())
        assert weighted_mean == pytest.approx(0.0, abs=1.0e-4)

    # and it is the fitted point plus the blinding offset, which is what the
    # cospar file already reports
    offset = parameter_offsets(blinded_config)["omega_m"]
    shifted = tmp_path / "shifted.resid"
    write_residuals(str(shifted), {**plain, "blinded": False,
                                   "point_estimate": {
                                       "omega_m": (plain["point_estimate"]
                                                   ["omega_m"] + offset)}},
                    config, diagram)
    np.testing.assert_allclose(
        blinded_model, _resid_columns(shifted.read_text())["mu_model"],
        atol=5.0e-5)
    assert blinded["blinded"] and not plain["blinded"]


def test_supplied_diagrams_take_the_redshift_window(repo_root):
    """A diagram handed over in memory is cut as the file route cuts one it
    reads (wfit.c:1539), so zmin and zmax mean the same on both routes."""
    config = WFITConfig()
    config.hubble_diagram = _fixture(repo_root)
    config.model = "lcdm"
    diagrams = load_hubble_diagrams(config)
    z = np.asarray(diagrams[0].redshift)
    # drop the lowest bins only, so the fit stays away from its bounds
    cut = WFITConfig(**{**config.to_dict(), "zmin": float(np.sort(z)[2])})

    cut.hubble_diagram = _fixture(repo_root)
    from_file = load_hubble_diagrams(cut)[0]
    supplied = run_wfit(cut, write=False, diagrams=diagrams)
    np.testing.assert_array_equal(
        np.asarray(run_wfit(cut, write=False,
                            diagrams=[from_file])[0]["point_estimate"]["omega_m"]),
        np.asarray(supplied[0]["point_estimate"]["omega_m"]))
    assert (np.asarray(from_file.redshift) >= cut.zmin).all()
    assert from_file.redshift.shape[0] < z.shape[0]


def test_supplied_diagrams_refuse_the_options_that_read_a_file_column(
    repo_root,
):
    """muerr_force, varname_muerr and muerr_ideal read a column of the file;
    a diagram in memory has none, so they abort rather than pass silently."""
    config = WFITConfig()
    config.hubble_diagram = _fixture(repo_root)
    diagrams = load_hubble_diagrams(config)
    for name, value in (("muerr_force", 0.2), ("varname_muerr", "MUERR"),
                        ("muerr_ideal", True)):
        refused = WFITConfig(**{**config.to_dict(), name: value})
        with pytest.raises(SNANAError, match=name):
            run_wfit(refused, write=False, diagrams=diagrams)
    assert run_wfit(config, write=False, diagrams=diagrams)


def test_a_bao_prior_replaces_the_gaussian_omega_m_term(tmp_path, repo_root):
    """get_chi2_priors takes the BAO branch or the Omega_m branch, never both
    (wfit.c:4446-4460); the YAML route used to add the second."""
    root = _bao_files(tmp_path)
    config = WFITConfig()
    config.hubble_diagram = _fixture(repo_root)
    config.bao = "DESI_2024"
    config.ompri, config.dompri = 0.9, 0.01   # a term impossible to miss
    data = load_hubble_diagrams(config)
    priors = prepare_priors(config, [d.truth for d in data], root)

    components = evaluate_likelihood(
        {"omega_m": 0.3, "w0": -1.0}, *stack_arrays(data)[:4],
        priors["omega_m_mean"], priors["cmb_r"], config, priors,
    )
    assert float(components["omega_m_chi2"]) == 0.0
    assert float(components["bao_chi2"]) > 0.0
    assert not config.omega_m_prior_applies()


def test_isdata_real_minus_one_means_simulated(tmp_path, repo_root):
    """create_covariance writes -1 where the BBC header carried no key;
    SNANA reads ISDATA_REAL = (ITMP > 0), so it unblinds (wfit.c:1791)."""
    fixture = Path(_fixture(repo_root)).read_text()
    unknown = tmp_path / "UNKNOWN.M0DIF"
    unknown.write_text("# ISDATA_REAL: -1\n" + fixture)
    config = WFITConfig()
    config.hubble_diagram = str(unknown)
    config.model = "lcdm"
    config.method = "grid"
    config.omsteps = 11
    config.blind_mode = "auto"
    config.cospar_yaml = str(tmp_path / "out.yaml")
    config.outfile_cospar = str(tmp_path / "out.cospar")

    assert run_wfit(config)[0]["blinded"] is False


def test_an_impossible_correlation_aborts_rather_than_clamping(
    repo_root, tmp_path,
):
    """wfit_FoM aborts on |rho| > 1 (wfit.c:4011-4015). Clamping the root's
    argument at zero would report a finite FoM for a broken covariance."""
    from jax_wfit.output import write_cospar

    config = WFITConfig()
    config.hubble_diagram = _fixture(repo_root)
    config.method = "grid"
    config.omsteps, config.wsteps = 11, 11
    config.cospar_yaml = str(tmp_path / "out.yaml")
    config.outfile_cospar = str(tmp_path / "out.cospar")
    result = run_wfit(config, write=False)[0]
    write_cospar(str(tmp_path / "good.cospar"), result, config)

    broken = dict(result, correlation=np.array([[1.0, 1.5], [1.5, 1.0]]))
    with pytest.raises(SNANAError, match="invalid rho"):
        write_cospar(str(tmp_path / "bad.cospar"), broken, config)


def test_truth_files_keep_diagram_order_and_require_matching_count(tmp_path):
    from jax_wfit.hubble import load_simulation_truths

    paths = []
    for omega_m in (0.25, 0.35):
        path = tmp_path / f"truth_{omega_m}.yml"
        path.write_text(f"MUSHIFT: 0.0\nOMEGA_MATTER: {omega_m}\n"
                        f"OMEGA_LAMBDA: {1 - omega_m}\n"
                        "w0_LAMBDA: -1.0\nwa_LAMBDA: 0.0\n")
        paths.append(str(path))
    config = WFITConfig(truth_info=paths)
    truths = load_simulation_truths(config, 2)
    assert [truth["omega_m"] for truth in truths] == [0.25, 0.35]
    with pytest.raises(SNANAError, match="number of truth INFO.YML files"):
        load_simulation_truths(config, 1)
    assert load_simulation_truths(WFITConfig(), 2) == (None, None)
