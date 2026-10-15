"""The covariance step feeding a cosmology fit.

This is what the port exists for. Two routes are covered: the in-memory one,
where the covariance never touches disk, and the file one, where wfit discovers
what it has been given by reading INFO.YML -- the path a real analysis takes.
"""

import gzip
import os
import shutil
from dataclasses import replace

import numpy as np
import pytest
import yaml

from jax_create_cov import build_covariance, create_covariance, load_config
from jax_create_cov import hubble, inputs, output
from jax_wfit.blinding import blinding_statuses
from jax_wfit.config import WFITConfig
from jax_wfit.data import (hubble_diagram_from_covariance,
                           load_hubble_diagrams, stack_arrays)
from jax_wfit.inference import fit_laplace
from jax_wfit.priors import prepare_priors
from jax_wfit.run import run_wfit
from jax_snlc_core.io.fitres_reader import read_fitres
from jax_snlc_core.utils import SNANAError

_FIXTURE = os.path.join("tests", "fixtures", "snana", "create_cov")


def _fit(diagrams):
    config = WFITConfig()
    config.model = "wcdm"
    # the fixture HD alone prefers w0 below the fit range, which the
    # optimiser reports as a bound-pinned abort; the CMB prior keeps the
    # minimum interior, as a real analysis of this diagram would
    config.cmb = True
    config.ompri_enabled = True
    config.dompri = 0.9
    priors = prepare_priors(config, [d.truth for d in diagrams], None)
    results, _ = fit_laplace(diagrams, stack_arrays(diagrams), config, priors,
                             blinding_statuses(config, diagrams))
    return float(results[0]["summaries"]["w0"]["median"])


@pytest.fixture(scope="module")
def cov_result():
    config = load_config(os.path.join(_FIXTURE, "input.yml"))
    files = inputs.grid_files(config)
    submit_info = inputs.read_submit_info(config.input_dir)
    diagrams = {name: hubble.load_hubble_diagram(path, config.mode)
                for name, path in files.items()}
    return build_covariance(
        diagrams,
        inputs.sys_scales("FITOPT", submit_info, config.sys_scales),
        inputs.sys_scales("MUOPT", submit_info, config.sys_scales),
        config.covopts, cosmology=config.cosmology, isdata_real=0,
    )


def test_covariance_is_built_without_touching_disk(cov_result, tmp_path):
    config = load_config(os.path.join(_FIXTURE, "input.yml"))
    config.outdir = None
    result = create_covariance(config)

    assert result.covariances and result.hubble_diagram is not None
    assert not list(tmp_path.iterdir())


def test_systematics_inflate_the_distance_errors(cov_result):
    covsys = {label: entry["matrix"] for label, entry in cov_result.covariances.items()}
    all_diag = np.diagonal(covsys["ALL"])

    assert np.all(np.diagonal(covsys["NOSYS"]) == 0), "NOSYS is stat only"
    assert all_diag.max() > 0
    np.testing.assert_array_equal(cov_result.muerr_sys, np.sqrt(all_diag))


def test_in_memory_covariance_fits(cov_result):
    """No disk at all: the precision matrix goes straight into the fitter."""
    diagram = hubble_diagram_from_covariance(
        cov_result.hubble_diagram, cov_result.covtot_inv("ALL"),
        cov_result.isdata_real, source="<memory>")
    assert -3.0 < _fit([diagram]) < 0.0


@pytest.mark.parametrize("mode", ["binned", "unbinned"])
def test_in_memory_diagram_matches_the_file_route(mode, tmp_path):
    """The Hubble diagram read back from the outputs, as the pipeline's
    Covariance stage holds it, gives the diagram wfit reads from the same
    files: the binned one keys rows by ROW, the unbinned one by CID."""
    config = load_config(os.path.join(_FIXTURE, "input.yml"))
    config.outdir, config.mode = tmp_path, mode
    create_covariance(config)
    info = yaml.safe_load((tmp_path / output.INFO_YML_FILENAME).read_text())
    _, _, covtot_inv_name = info["COVOPTS"][0].split()

    wfit = WFITConfig()
    wfit.hubble_diagram = str(tmp_path / info["HD"])
    wfit.covariance = str(tmp_path / covtot_inv_name)
    wfit.covariance_kind = "total_inverse"
    from_file = load_hubble_diagrams(wfit)[0]
    held = hubble_diagram_from_covariance(
        read_fitres(wfit.hubble_diagram).df, np.asarray(from_file.precision),
        info["ISDATA_REAL"], source="<memory>")

    np.testing.assert_array_equal(held.cid, from_file.cid)
    for name in ("redshift", "mu", "muerr"):
        np.testing.assert_array_equal(np.asarray(getattr(held, name)),
                                      np.asarray(getattr(from_file, name)))


def test_run_wfit_validates_the_config_on_the_diagrams_path(cov_result):
    """A config handed diagrams in memory is checked as one naming files is."""
    diagram = hubble_diagram_from_covariance(
        cov_result.hubble_diagram, cov_result.covtot_inv("ALL"),
        cov_result.isdata_real, source="<memory>")
    config = WFITConfig()
    config.zmin, config.zmax = 1.0, 0.5

    with pytest.raises(SNANAError, match="zmin must not exceed zmax"):
        run_wfit(config, write=False, diagrams=[diagram])


def test_snrms_on_an_in_memory_inverse_covariance_aborts(cov_result):
    """snrms cannot be added to an already-inverted covariance, in memory as
    from a file."""
    diagram = hubble_diagram_from_covariance(
        cov_result.hubble_diagram, cov_result.covtot_inv("ALL"),
        cov_result.isdata_real, source="<memory>")
    config = WFITConfig()
    config.snrms = 0.1

    with pytest.raises(SNANAError, match="snrms cannot modify"):
        run_wfit(config, write=False, diagrams=[diagram])

def test_create_covariance_reads_an_uncompressed_grid(tmp_path):
    """SALT2mu.exe writes plain FITRES files; only a batch job gzips them."""
    shutil.copytree(os.path.join(_FIXTURE, "bbc"), tmp_path / "bbc")
    packed = tmp_path / "bbc" / "OUTPUT_BBC" / "FITOPT000_MUOPT000.FITRES.gz"
    with gzip.open(packed, "rb") as src, open(packed.with_suffix(""), "wb") as dst:
        shutil.copyfileobj(src, dst)
    packed.unlink()
    config = replace(load_config(os.path.join(_FIXTURE, "input.yml")),
                     input_dir=tmp_path / "bbc", outdir=tmp_path / "out")

    result = create_covariance(config)
    info = yaml.safe_load((tmp_path / "out" / output.INFO_YML_FILENAME).read_text())

    assert result.version_photometry
    assert info["BBC_INFO_FILE"] == str(tmp_path / "bbc" / "SUBMIT.INFO")


def test_wfit_discovers_the_outputs_through_info_yml(tmp_path):
    """INFO.YML names the Hubble diagram and maps each covopt to its file, which
    is how a file-based consumer finds them without being told out of band."""
    config = load_config(os.path.join(_FIXTURE, "input.yml"))
    config.outdir = tmp_path
    create_covariance(config)

    info = yaml.safe_load((tmp_path / output.INFO_YML_FILENAME).read_text())
    assert info["ISDATA_REAL"] == 0
    assert info["SIZE_HD"] > 0

    # "<label> <covsys file> <covtot_inv file>", None where not written
    label, covsys_name, covtot_inv_name = info["COVOPTS"][0].split()
    assert label == "ALL" and covsys_name == "None"

    wfit = WFITConfig()
    wfit.model = "wcdm"
    wfit.hubble_diagram = str(tmp_path / info["HD"])
    wfit.covariance = str(tmp_path / covtot_inv_name)
    wfit.covariance_kind = "total_inverse"
    diagrams = load_hubble_diagrams(wfit)

    assert diagrams[0].n_rows == info["SIZE_HD"]
    assert -3.0 < _fit(diagrams) < 0.0


def test_info_yml_reports_the_files_that_were_written(tmp_path):
    """COVOPTS must name only files that exist, or discovery is worse than none."""
    config = load_config(os.path.join(_FIXTURE, "input.yml"))
    config.outdir = tmp_path
    config.write_mask_cov = 1 | 2
    create_covariance(config)

    info = yaml.safe_load((tmp_path / output.INFO_YML_FILENAME).read_text())
    for entry in info["COVOPTS"].values():
        _, covsys_name, covtot_inv_name = entry.split()
        for name in (covsys_name, covtot_inv_name):
            assert name != "None", "both products were requested"
            assert (tmp_path / name).is_file(), f"{name} is named but absent"


@pytest.mark.parametrize("prior", ["cmb", "bao"])
@pytest.mark.parametrize("truth_source", ["explicit", "file", "files"])
def test_pipeline_simulation_priors_match_cli(tmp_path, monkeypatch, prior, truth_source):
    """Both entry points fit the same written HD and covariance with truth."""
    from jax_pipeline import Cosmology, Covariance, Fit
    from wfit import main

    covariance_config = replace(load_config(os.path.join(_FIXTURE, "input.yml")),
                                outdir=tmp_path / "cov")
    create_covariance(covariance_config)
    covariance = Covariance("covariance", distances=Fit("unused", fitres="unused.FITRES"))
    held = covariance.load({}, covariance_config.outdir)
    info = yaml.safe_load((covariance_config.outdir / "INFO.YML").read_text())
    labels = tuple(held["covariances"])[:2]
    settings = {"method": "grid", "omsteps": 11, "wsteps": 15,
                "double_precision": True, "ompri": 0.315, "dompri": 0.9}
    args = ["-double_precision", "1", "--method", "grid",
            "-omsteps", "11", "-wsteps", "15", "-ompri", "0.315", "-dompri", "0.9"]
    if prior == "cmb":
        settings.update(cmb=True, cmb_sim=True, sigma_Rcmb=0.006)
        args += ["--cmb", "--cmb-sim", "--sigma-Rcmb", "0.006"]
    else:
        root = tmp_path / "sndata" / "models" / "BAO"
        root.mkdir(parents=True)
        (root / "DESI_2024_mean.dat").write_text("0.5 13.0 DM_over_rd\n")
        (root / "DESI_2024_cov.dat").write_text("0.04\n")
        monkeypatch.setenv("SNDATA_ROOT", str(root.parent.parent))
        settings.update(bao="DESI_2024", bao_sim="DESI_2024")
        args += ["--bao", "DESI_2024", "--bao-sim", "DESI_2024"]
    if truth_source == "explicit":
        settings.update(om_sim=0.315, w0_sim=-1.0, wa_sim=0.0)
        args += ["--om-sim", "0.315", "-w0_sim", "-1.0", "-wa_sim", "0.0"]
    else:
        truth_paths = []
        for i in range(len(labels) if truth_source == "files" else 1):
            path = tmp_path / f"truth{i}.yml"
            path.write_text("MUSHIFT: 0.0\nOMEGA_MATTER: 0.315\n"
                            "OMEGA_LAMBDA: 0.685\nw0_LAMBDA: -1.0\nwa_LAMBDA: 0.0\n")
            truth_paths.append(str(path))
        settings["truth_info"] = truth_paths if truth_source == "files" else truth_paths[0]
        args += ["--truth-info", *truth_paths]
    stage = Cosmology("cosmology", covariance=covariance, covopts=labels, overrides=settings)
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    actual = stage.run({"covariance": held}, stage_dir)

    # Run the real CLI entry point, including argument parsing and file loading.
    inverse_files = {entry.split()[0]: entry.split()[2]
                     for entry in info["COVOPTS"].values()}
    hd = str(covariance_config.outdir / info["HD"])
    cli_input = tmp_path / "cli_input.yaml"
    cli_input.write_text(yaml.safe_dump({
        "covariance_kind": "total_inverse",
        "covariance": [str(covariance_config.outdir / inverse_files[label]) for label in labels],
    }))
    cli_yaml = tmp_path / "cli.yaml"
    assert main([*([hd] * len(labels)), "--config", str(cli_input),
                 *args, "-cospar_yaml", str(cli_yaml),
                 "-outfile_cospar", str(tmp_path / "cli.cospar")]) == 0
    expected = yaml.safe_load(cli_yaml.read_text())["realisations"]
    for label, result in zip(labels, expected):
        assert actual[label]["parameters"] == result["parameters"]
        assert actual[label]["input"]["simulation_truth"] == result["input"]["simulation_truth"]


@pytest.mark.parametrize("in_input_file", [False, True])
@pytest.mark.parametrize("multiple", [False, True])
def test_cosmology_cache_tracks_truth_contents(tmp_path, in_input_file, multiple):
    from jax_pipeline import Cosmology, Covariance, Fit
    from jax_pipeline.run import _fields, stage_hash

    paths = [tmp_path / "truth0.yml", tmp_path / "truth1.yml"]
    for path in paths:
        path.write_text("OMEGA_MATTER: 0.315\n")
    settings = {"truth_info": [str(p) for p in paths] if multiple else str(paths[0])}
    covariance = Covariance("covariance", distances=Fit("unused", fitres="unused.FITRES"))
    if in_input_file:
        input_file = tmp_path / "wfit.yml"
        input_file.write_text(yaml.safe_dump(settings))
        stage = Cosmology("cosmology", covariance=covariance, input_file=input_file)
    else:
        stage = Cosmology("cosmology", covariance=covariance, overrides=settings)
    hashes = {id(covariance): "unchanged"}
    before = stage_hash(_fields(stage, hashes))
    paths[1 if multiple else 0].write_text("OMEGA_MATTER: 0.3\n")
    assert stage_hash(_fields(stage, hashes)) != before
    stage.overrides = {"truth_info": None}
    assert stage.cache_files() == ()
