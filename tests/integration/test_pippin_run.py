"""A Pippin config run end to end, and again with every stage loaded.

The config is small and written by the test over this repository's DES-like
fixture inputs, so it needs no Pippin checkout; it exercises the sections
that have stages here, through a fitted cosmology.
"""

from jax_snlc_core.io.fits_path import open_text

import os
from pathlib import Path

import pytest
import yaml
from astropy.io import fits

from jax_pipeline.pippin import run_pippin

_FIXTURES = Path("tests/fixtures/snana").resolve()
_DATA = _FIXTURES / "bbc_des_data"
_BIASCOR = _FIXTURES / "bbc_des_biascor"



@pytest.fixture(scope="module")
def config(tmp_path_factory):
    inputs = tmp_path_factory.mktemp("pippin_inputs")
    (inputs / "bbc.input").write_text(
        "nzbin=8\nzmin=0.1\nzmax=1.0\nu1=1\nu2=1\nmin_per_cell_biascor=3\n"
        "surveygroup_biascor=DES(zbin=0.15:x1bin=2.0:cbin=0.2)\n")
    text = {
        "SIM": {
            "DATA": {"IA": {"BASE": str(_DATA / "input.input"), "NGENTOT_LC": 500},
                     "GLOBAL": {"RANSEED_REPEAT": "1 11"}},
            "BIAS": {"IA": {"BASE": str(_BIASCOR / "input.input"), "NGENTOT_LC": 2500},
                     "GLOBAL": {"RANSEED_REPEAT": "1 12"}},
        },
        "LCFIT": {"D": {"BASE": str(_DATA / "fit.nml"),
                        "FITOPTS": ["/zp_g/ MAGOBS_SHIFT_ZP 'g .01'"]}},
        "BIASCOR": {"BBC": {"BASE": str(inputs / "bbc.input"), "DATA": ["D_DATA"],
                            "SIMFILE_BIASCOR": ["D_BIAS"],
                            "MUOPTS": {"CHI2": {"OPTS": {"chi2max": 16}}}}},
        "CREATE_COV": {"COV": {"OPTS": {"COVOPTS": ["[NOSYS] [=DEFAULT,=DEFAULT]"]}}},
        "COSMOFIT": {"WFIT": {"W": {"OPTS": {"WFITOPTS": ["/om/ -lcdm -ompri 0.31 -dompri 0.01"]}}}},
    }
    path = inputs / "analysis.yml"
    path.write_text(yaml.safe_dump(text))
    return path


def test_a_pippin_config_runs_to_a_cosmology_and_then_loads(config, tmp_path):
    results = run_pippin(config, tmp_path)

    assert set(results) == {"DATA", "BIAS", "LCFIT_D", "BBC", "COV", "WFIT_W_om"}
    assert set(results["LCFIT_D"]) == {"DATA", "BIAS"}
    assert set(results["BBC"]) == {"nominal", "zp_g", "CHI2", ("zp_g", "CHI2")}
    assert list(results["WFIT_W_om"]) == ["ALL", "NOSYS"]
    fit = results["WFIT_W_om"]["ALL"]
    assert "omega_m" in fit["point_estimate"]
    written = os.path.getmtime(tmp_path / "BBC" / "FITOPT000_MUOPT000" / "BBC.M0DIF")

    again = run_pippin(config, tmp_path)
    manifest = yaml.safe_load((tmp_path / "MANIFEST.yml").read_text())

    assert all(entry["loaded"] for entry in manifest["stages"].values())
    assert os.path.getmtime(tmp_path / "BBC" / "FITOPT000_MUOPT000" / "BBC.M0DIF") == written
    assert again["WFIT_W_om"]["ALL"]["point_estimate"] == fit["point_estimate"]


@pytest.mark.parametrize("format_mask", [48, 32])
def test_repeat_jobs_simulate_different_events(tmp_path, format_mask):
    """Two RANSEED_REPEAT jobs, with random CIDs (48) and without (32), draw
    different events: no CID or truth value is shared between them. The
    random-CID pool holds 124 CIDs and each job writes about 40 of its 60
    events, so two independently drawn CID lists would share about 13 CIDs
    and miss each other with probability below 1e-6."""
    config = tmp_path / "analysis.yml"
    config.write_text(yaml.safe_dump({"SIM": {"REP": {
        "IA": {"BASE": str(_DATA / "input.input"), "NGENTOT_LC": 60},
        "GLOBAL": {"RANSEED_REPEAT": "2 11", "FORMAT_MASK": format_mask,
                   "CIDRAN_MAX": 125}}}}))

    run_pippin(config, tmp_path / "run")

    heads = sorted((tmp_path / "run").rglob("*_HEAD.FITS*"))
    assert len(heads) == 2
    first, second = (fits.getdata(head) for head in heads)
    assert len(first) >= 30 and len(second) >= 30
    assert not set(first["SNID"]) & set(second["SNID"])
    for column in ("SIM_PEAKMJD", "SIM_REDSHIFT_CMB", "SIM_SALT2x1", "SIM_SALT2c"):
        assert not set(first[column]) & set(second[column]), column


def test_ranseed_change_runs_separate_reproducible_simulations(tmp_path):
    from jax_pipeline import Sim, run
    from jax_pipeline.pippin import read
    config = tmp_path / "sim.yml"
    config.write_text(yaml.safe_dump({"SIM": {"S": {
        "IA": {"BASE": str(_DATA / "input.input"), "NGENTOT_LC": 20},
        "GLOBAL": {"RANSEED_CHANGE": "2 8123", "FORMAT_MASK": 32}}}}))
    stages = [s for s in read(config) if isinstance(s, Sim)]
    run(*stages, outdir=tmp_path / "run")
    heads = [next((tmp_path / "run" / s.name).rglob("*_HEAD.FITS*")) for s in stages]
    tables = [fits.getdata(path) for path in heads]
    assert all(0 < len(table) <= 20 for table in tables)
    assert not set(tables[0]["SIM_REDSHIFT_CMB"]) & set(tables[1]["SIM_REDSHIFT_CMB"])
    run(*stages, outdir=tmp_path / "run")
    manifest = yaml.safe_load((tmp_path / "run" / "MANIFEST.yml").read_text())
    assert all(entry["loaded"] for entry in manifest["stages"].values())


@pytest.mark.parametrize("reuse_group", [False, True])
def test_realisations_and_partitions_run_through_cosmology(tmp_path, reuse_group):
    """Reuse fitted tables to exercise four separate cosmology analyses."""
    import shutil
    from jax_pipeline import Cosmology, run
    from jax_pipeline.pippin import read
    external = []
    for i in (1, 2):
        task = tmp_path / ("D_S" if reuse_group else f"D_S-{i:04d}")
        version = task / "output" / (f"VERSION-{i:04d}" if reuse_group else "VERSION")
        version.mkdir(parents=True)
        with open_text(_FIXTURES / "bbc_des_beams" / "DATA.FITRES") as source, (version / "FITOPT000.FITRES").open("w") as target:
            shutil.copyfileobj(source, target)
        (task / "output" / "SUBMIT.INFO").write_text(yaml.safe_dump({
            "FITOPT_LIST": [["FITOPT000", None, None]]}))
        if str(task) not in external:
            external.append(str(task))
    bbc_file = tmp_path / "bbc.input"
    bbc_file.write_text("nzbin=4\nzmin=0.1\nzmax=1.0\nu1=1\nu2=1\n")
    config = tmp_path / "analysis.yml"
    config.write_text(yaml.safe_dump({
        "SIM": {"S": {"IA": {"BASE": str(_DATA / "input.input")},
                      "GLOBAL": {"RANSEED_CHANGE": "2 123"}}},
        "LCFIT": {"D": {"BASE": str(_DATA / "fit.nml"), "EXTERNAL_DIRS": external}},
        "BIASCOR": {"B": {"BASE": str(bbc_file), "DATA": ["D_S"],
                            "OPTS": {"NSPLITRAN": 2}}},
        "CREATE_COV": {"C": {}},
        "COSMOFIT": {"WFIT": {"W": {"OPTS": {"WFITOPTS": ["/lcdm/ -lcdm -ompri 0.31 -dompri 0.01"]}}}},
    }))
    if reuse_group:
        # Imported photometry records establish the realisation names; the
        # cosmology graph reads only the already-fitted tables.
        sim_task = tmp_path / "saved_sim"
        sim_task.mkdir()
        folders = []
        for i in (1, 2):
            folder = sim_task / f"VERSION-{i:04d}"
            folder.mkdir()
            (folder / "VERSION.LIST").write_text("")
            folders.append(str(folder))
        (sim_task / "config.yml").write_text(yaml.safe_dump({"OUTPUT": {"sim_folders": folders}}))
        saved = yaml.safe_load(config.read_text())
        saved["SIM"] = {"S": {"EXTERNAL": str(sim_task)}}
        saved["LCFIT"]["D"].pop("BASE")
        config.write_text(yaml.safe_dump(saved))
    stages = [s for s in read(config) if isinstance(s, Cosmology)]
    assert len(stages) == 4
    # The imported legacy tables lack reference cosmology metadata. Supply
    # the BBC default explicitly through the Python stage configuration.
    for stage in stages:
        stage.covariance.overrides = {**stage.covariance.overrides,
            "COSMOLOGY": {"OMEGA_MATTER": 0.3, "OMEGA_LAMBDA": 0.7,
                          "w0_LAMBDA": -1.0, "wa_LAMBDA": 0.0}}
    results = run(*stages, outdir=tmp_path / "run")
    for stage in stages:
        assert "omega_m" in results[stage.name]["ALL"]["point_estimate"]
    import pandas as pd
    summary_path = tmp_path / "run" / "COSMOLOGY_SUMMARY.csv"
    summary = pd.read_csv(summary_path)
    assert len(summary) == 4
    assert set(zip(summary.realisation, summary.split_index)) == {(1, 1), (1, 2), (2, 1), (2, 2)}
    for stage in stages:
        row = summary.loc[summary.cosmology == stage.name].iloc[0]
        assert row.omega_m == pytest.approx(results[stage.name]["ALL"]["point_estimate"]["omega_m"])
    ids = [set(results[f"B-0001-SPLIT{i:03d}"]["nominal"]["FITRES"].df.CID) for i in (1, 2)]
    assert ids[0] and ids[1] and not ids[0] & ids[1]
    run(*stages, outdir=tmp_path / "run")
    pd.testing.assert_frame_equal(summary, pd.read_csv(summary_path))
    manifest = yaml.safe_load((tmp_path / "run" / "MANIFEST.yml").read_text())
    assert all(entry["loaded"] for entry in manifest["stages"].values())
