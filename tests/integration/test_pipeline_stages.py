"""The built-in stages, simulating and fitting for real.

One small simulation feeding one fit pins what the stages promise: that a
fit reads the version the simulation wrote, that every stage writes its own
SNANA outputs, that a FITOPT runs a second fit whose answer differs from the
nominal, that FITOPTs changing only the data share the nominal's fitter, that
one fit of several versions compiles once, that a second run loads every
stage instead of rerunning it, and that several inputs make one version
whose events do not collide.
"""

import collections
import logging
import os
import re
import subprocess
import sys

import jax
import pytest
import yaml

from jax_pipeline import Fit, Sim, run
from jax_snlc_core.logging import SNANAError
from jax_snlc_fit import loader, run as fit_run
from jax_snlc_fit.fitter import prepare

_NOMINAL = "nominal"


@pytest.fixture(scope="module")
def fixture_dir(repo_root):
    return repo_root / "tests" / "fixtures" / "snana" / "salt3_fit_select"


@pytest.fixture(scope="module")
def simulated_and_fitted(fixture_dir, tmp_path_factory):
    """A two-stage analysis, run once for the tests that read it."""
    outdir = tmp_path_factory.mktemp("pipeline")
    sim = Sim("sim", fixture_dir / "input.input",
              overrides={"NGENTOT_LC": 60, "RANSEED": 7})
    fit = Fit("fit", fixture_dir / "fit.nml", datasets=[sim],
              fitopts={"zpt_g": {"MAGOBS_SHIFT_ZP": "g .02"}}, overrides={"BATCH_SIZE": 4})
    return outdir, sim, fit, run(fit, outdir=outdir)


def test_a_fitopt_changes_the_answer(simulated_and_fitted):
    """A zero-point shift the fit is told about must move the fitted
    brightness, or the override never reached the fit."""
    _, _, _, results = simulated_and_fitted
    nominal = results["fit"]["sim"][_NOMINAL].df
    shifted = results["fit"]["sim"]["zpt_g"].df

    assert list(nominal["CID"]) == list(shifted["CID"])
    assert not nominal["mB"].equals(shifted["mB"])


def test_every_stage_writes_its_own_outputs(simulated_and_fitted):
    outdir, _, _, results = simulated_and_fitted

    assert results["sim"] == outdir / "sim"
    assert (outdir / "sim" / "sim_HEAD.FITS.gz").is_file()
    assert (outdir / "sim" / "sim.LIST").is_file()
    for cell in ("FITOPT000_MUOPT000", "FITOPT001_MUOPT000"):
        assert (outdir / "fit" / cell / "sim.FITRES.TEXT").is_file()
    assert (outdir / "fit" / "DONE.yml").is_file()
    assert (outdir / "MANIFEST.yml").is_file()


def test_the_manifest_names_the_systematics(simulated_and_fitted):
    outdir, _, _, _ = simulated_and_fitted
    manifest = yaml.safe_load((outdir / "MANIFEST.yml").read_text())
    fit_entry = manifest["stages"]["fit"]

    assert fit_entry["inputs"] == {"data0": "sim"}
    assert fit_entry["fitopts"] == {
        "zpt_g": {"MAGOBS_SHIFT_ZP": "g .02"}}
    assert manifest["stages"]["sim"]["overrides"] == [
        {"NGENTOT_LC": 60, "RANSEED": 7}]


def test_a_second_run_loads_every_stage(fixture_dir, simulated_and_fitted):
    """Nothing changed, so nothing is rerun and the answer is the same; a Fit
    asked to rerun fits again, to the same answer."""
    outdir, _, _, results = simulated_and_fitted
    written = os.path.getmtime(outdir / "fit" / "FITOPT000_MUOPT000" / "sim.FITRES.TEXT")
    sim = Sim("sim", fixture_dir / "input.input",
              overrides={"NGENTOT_LC": 60, "RANSEED": 7})
    fit = Fit("fit", fixture_dir / "fit.nml", datasets=[sim],
              fitopts={"zpt_g": {"MAGOBS_SHIFT_ZP": "g .02"}}, overrides={"BATCH_SIZE": 4})

    again = run(fit, outdir=outdir)
    manifest = yaml.safe_load((outdir / "MANIFEST.yml").read_text())

    assert all(entry["loaded"] for entry in manifest["stages"].values())
    assert os.path.getmtime(outdir / "fit" / "FITOPT000_MUOPT000" / "sim.FITRES.TEXT") == written
    for cell in results["fit"]["sim"]:
        assert again["fit"]["sim"][cell].df.equals(results["fit"]["sim"][cell].df)

    refit = run(fit, outdir=outdir, refresh=["fit"])
    manifest = yaml.safe_load((outdir / "MANIFEST.yml").read_text())
    assert manifest["stages"]["sim"]["loaded"]
    assert not manifest["stages"]["fit"]["loaded"]
    for cell in results["fit"]["sim"]:
        assert refit["fit"]["sim"][cell].df.equals(results["fit"]["sim"][cell].df)


def test_opt_sncid_list_fits_the_nominal_events_in_every_fitopt(fixture_dir, tmp_path):
    """With OPT_SNCID_LIST, a FITOPT that tightens a cut still fits exactly
    the nominal's events, as submit_batch_jobs arranges through
    SNCID_LIST_FILE; a NOREJECT label keeps its own cuts."""
    sim = Sim("sim", fixture_dir / "input.input",
              overrides={"NGENTOT_LC": 300, "RANSEED": 7})
    fit = Fit("fit", fixture_dir / "fit.nml", datasets=[sim],
              overrides={"BATCH_SIZE": 4, "OPT_SNCID_LIST": 1},
              fitopts={"tight": {"CUTWIN_REDSHIFT": "0.30 0.40"},
                       "tight_NOREJECT": {"CUTWIN_REDSHIFT": "0.30 0.40"}})

    tables = run(fit, outdir=tmp_path)["fit"]["sim"]
    nominal = set(tables[_NOMINAL].df["CID"])

    assert set(tables["tight"].df["CID"]) == nominal
    assert set(tables["tight_NOREJECT"].df["CID"]) < nominal


# the fitter's compiled kernels, as JAX names them in its compile log
_KERNELS = ("_round", "_errors", "_frozen_weights", "_pre_fit_adjust",
            "_coarse_grid", "_reject_epochs", "_flux")


def _compiled(caplog, fit):
    """The fitter kernels JAX compiles while ``fit()`` runs."""
    caplog.clear()
    with jax.log_compiles(), caplog.at_level(logging.WARNING):
        fit()
    return [kernel for kernel in _KERNELS for record in caplog.records
            if f"Compiling jit({kernel})" in record.getMessage()]


def test_one_fit_fits_several_realisations(fixture_dir, tmp_path, caplog):
    """Two realisations fitted by one Fit, sized over both as the namelist
    loads them, so its longest light curve is the longer of theirs: the
    pair compiles the fitter kernels exactly as often as one realisation
    alone, so the second compiles nothing."""
    sims = [Sim(f"sim{seed}", fixture_dir / "input.input",
                overrides={"NGENTOT_LC": 40, "RANSEED": seed}) for seed in (7, 8)]
    nml = fixture_dir / "fit.nml"
    both = Fit("both", nml, datasets=sims, overrides={"BATCH_SIZE": 4})
    alone = [Fit(f"alone{i}", nml, datasets=[sim], overrides={"BATCH_SIZE": 4})
             for i, sim in enumerate(sims)]
    run(*sims, outdir=tmp_path)

    compiled_alone = _compiled(caplog, lambda: run(alone[0], outdir=tmp_path))
    compiled_both = _compiled(caplog, lambda: run(both, outdir=tmp_path))
    results = run(*alone, outdir=tmp_path)

    assert compiled_alone != []
    assert compiled_both == compiled_alone
    longest = [yaml.safe_load((tmp_path / stage.name / "SHAPE.yml")
                              .read_text())["epoch_lengths"][-1]
               for stage in (both, *alone)]
    assert longest[1] != longest[2]
    assert longest[0] == max(longest[1:])
    assert len(results["alone0"]["sim7"][_NOMINAL].df) > 0


def test_an_interrupted_fit_resumes_with_the_versions_left(fixture_dir, tmp_path, caplog):
    """A Fit interrupted while fitting its second version keeps the first
    version's tables when run again, and fits only the second, compiling the
    fitter kernels once, to the tables an uninterrupted run writes. Asked
    to rerun, it starts again."""
    sims = [Sim(f"sim{seed}", fixture_dir / "input.input",
                overrides={"NGENTOT_LC": 40, "RANSEED": seed}) for seed in (7, 8)]
    fit = Fit("fit", fixture_dir / "fit.nml", datasets=sims, overrides={"BATCH_SIZE": 4},
              fitopts={"zpt_g": {"MAGOBS_SHIFT_ZP": "g .02"}})
    run(*sims, outdir=tmp_path)
    full = {}
    compiled = _compiled(caplog, lambda: full.update(run(fit, outdir=tmp_path)["fit"]))

    # what an interruption while fitting sim8 leaves behind
    stage_dir = tmp_path / "fit"
    (stage_dir / "DONE.yml").unlink()
    (stage_dir / "FINISHED.yml").write_text(yaml.safe_dump(["sim7"]))
    for table in stage_dir.rglob("sim8.*"):
        table.unlink()
    kept = {path: os.path.getmtime(path) for path in stage_dir.rglob("sim7.*")}

    resumed = {}
    assert _compiled(caplog, lambda: resumed.update(run(fit, outdir=tmp_path)["fit"])) == compiled
    assert {path: os.path.getmtime(path) for path in kept} == kept
    for version, tables in full.items():
        for cell, table in tables.items():
            assert resumed[version][cell].df.equals(table.df)

    run(fit, outdir=tmp_path, refresh=["fit"])
    assert all(os.path.getmtime(path) != written for path, written in kept.items())


def test_a_version_the_fit_does_not_fit_is_refused(fixture_dir):
    sims = [Sim(f"sim{seed}", fixture_dir / "input.input",
                overrides={"NGENTOT_LC": 40, "RANSEED": seed})
            for seed in (7, 8)]
    fit = Fit("fit", fixture_dir / "fit.nml", datasets=sims[:1])
    with pytest.raises(SNANAError, match="does not fit 'sim8'"):
        fit.version(sims[1])


def test_fitopts_changing_only_the_data_share_the_nominal_fitter(
        fixture_dir, tmp_path, caplog):
    """A FITOPT that changes only how the data are read, calibrated or cut
    is fitted by the nominal's fitter, compiling nothing the nominal alone
    does not, and writes what a fitter built for it alone writes; one that
    changes the fit bands has a fitter of its own."""
    sim = Sim("sim", fixture_dir / "input.input",
              overrides={"NGENTOT_LC": 300, "RANSEED": 7})
    nml = fixture_dir / "fit.nml"
    nominal = Fit("nominal", nml, datasets=[sim], overrides={"BATCH_SIZE": 4})
    shared = Fit("shared", nml, datasets=[sim], fitopts={
        "zpt_g": {"MAGOBS_SHIFT_ZP": "g .02"},
        "lam_g": {"FILTER_LAMSHIFT": "g 10"},
        "primary_g": {"MAGOBS_SHIFT_PRIMARY": "g .02"},
        "zcut": {"CUTWIN_REDSHIFT": "0.05 0.30"}}, overrides={"BATCH_SIZE": 4})
    alone = Fit("alone", nml, datasets=[sim],
                overrides={"BATCH_SIZE": 4, "FILTER_LAMSHIFT": "g 10"},
                fitopts={"gri": {"FILTLIST_FIT": "gri"}})
    run(sim, outdir=tmp_path)

    compiled = _compiled(caplog, lambda: run(nominal, outdir=tmp_path))
    assert compiled != []
    results = {}
    assert _compiled(caplog, lambda: results.update(run(shared, outdir=tmp_path))) == compiled
    assert collections.Counter(_compiled(
        caplog, lambda: results.update(run(alone, outdir=tmp_path)))) == {
            kernel: 2 * count for kernel, count in collections.Counter(compiled).items()}

    # Compare the shared and independent wavelength-shifted fits at the same
    # shapes. The unshifted nominal can need a smaller integration window.
    shapes = [yaml.safe_load((tmp_path / name / "SHAPE.yml").read_text())
              for name in ("shared", "alone")]
    assert shapes[0] == shapes[1]
    tables = results["shared"]["sim"]
    assert len(tables["lam_g"].df) >= 10
    assert results["alone"]["sim"][_NOMINAL].df.equals(tables["lam_g"].df)


def test_a_fitopt_adding_fit_bands_sizes_the_fitter_with_them(
        fixture_dir, tmp_path):
    """The loader keeps only the fit bands' epochs, so a FITOPT that widens
    FILTLIST_FIT is sized on its own light curves, as long as a nominal fit
    in those bands, not on the narrower nominal's."""
    sim = Sim("sim", fixture_dir / "input.input",
              overrides={"NGENTOT_LC": 40, "RANSEED": 7})
    nml = fixture_dir / "fit.nml"
    stages = [Fit("griz", nml, datasets=[sim], overrides={"BATCH_SIZE": 4}),
              Fit("g", nml, datasets=[sim], overrides={"BATCH_SIZE": 4, "FILTLIST_FIT": "g"}),
              Fit("widened", nml, datasets=[sim],
                  overrides={"BATCH_SIZE": 4, "FILTLIST_FIT": "g"},
                  fitopts={"griz": {"FILTLIST_FIT": "griz"}})]
    run(*stages, outdir=tmp_path)
    longest = {stage.name: yaml.safe_load((tmp_path / stage.name / "SHAPE.yml")
                                          .read_text())["epoch_lengths"][-1]
               for stage in stages}

    assert longest["g"] < longest["griz"]
    assert longest["widened"] == longest["griz"]


def test_several_inputs_make_one_version(fixture_dir, tmp_path):
    """Two inputs written into one version, as a Pippin SIM group is: the
    fit reads both HEAD files and no CID is shared between them. Each run's
    README goes under misc/ and the version's own README merges them, as
    SNANA's batch layer writes it: summed statistics, every run's types
    labelled by its model, and every run's input keys. The fit records the
    cosmology the merged README gives."""
    cosmology = {"OMEGA_MATTER": 0.2812, "OMEGA_LAMBDA": 0.7188}
    sim = Sim("mix", [fixture_dir / "input.input", fixture_dir / "input.input"],
              overrides=[{"NGENTOT_LC": 30, "RANSEED": 7, **cosmology},
                         {"NGENTOT_LC": 30, "RANSEED": 8, **cosmology}])
    fit = Fit("fit", fixture_dir / "fit.nml", datasets=[sim], overrides={"BATCH_SIZE": 4})

    results = run(fit, outdir=tmp_path)

    listed = (tmp_path / "mix" / "mix.LIST").read_text().split()
    assert listed == ["mix_input_0_HEAD.FITS", "mix_input_1_HEAD.FITS"]
    runs = [yaml.safe_load((tmp_path / "mix" / "misc" / f"mix_input_{i}.README").read_text())
            ["DOCUMENTATION"] for i in (0, 1)]
    merged = yaml.safe_load((tmp_path / "mix" / "mix.README").read_text())["DOCUMENTATION"]
    assert merged["GENVERSION"] == "mix"
    written = sum(run["OUTPUT_SUMMARY"]["NGENLC_WRITE"] for run in runs)
    assert merged["STAT_SUMMARY"][0].split()[:3] == [
        "TOTAL", str(sum(run["OUTPUT_SUMMARY"]["NGENLC_TOT"] for run in runs)), str(written)]
    # both runs model a Type Ia, so the later one's label is on the shared type
    assert {g: name.split() for g, name in merged["GENTYPE_TO_NAME"].items()} == {
        g: [*name.split(), "SNIaMODEL01"] for g, name in runs[1]["GENTYPE_TO_NAME"].items()}
    assert merged["FILTERS"] == runs[0]["FILTERS"]
    assert {"INPUT_KEYS_SNIaMODEL00", "INPUT_KEYS_SNIaMODEL01"} <= set(merged)
    assert "NGENTOT_LC" not in (tmp_path / "mix" / "mix.README").read_text()
    meta = results["fit"]["mix"][_NOMINAL].meta
    assert {key: float(meta[f"SIM_{key}"]) for key in cosmology} == cosmology
    assert (float(meta["SIM_w0_LAMBDA"]), float(meta["SIM_wa_LAMBDA"])) == (-1.0, 0.0)
    assert meta["SIM_MUSHIFT"] == "0", "an unset MUSHIFT is no shift"
    cids = results["fit"]["mix"][_NOMINAL].df["CID"].astype(int)
    assert cids.is_unique
    assert len(cids) > 0
    assert cids.max() >= 30, "the second input's CIDs were not offset"


def test_quickstart_cosmology_analysis(repo_root, tmp_path):
    """Run the cosmology guide's example with its supplied survey inputs."""
    script = (repo_root / "examples" / "quickstart" / "analysis.py").read_text()
    redirected = script.replace('outdir="run"', f"outdir={str(tmp_path)!r}")
    assert redirected != script, "the script no longer writes into 'run'"

    result = subprocess.run([sys.executable, "-c", redirected],
                            capture_output=True, text=True,
                            cwd=repo_root, timeout=1800)

    assert result.returncode == 0, (
        f"the quickstart exited {result.returncode}\n"
        f"stdout (tail):\n{result.stdout[-1500:]}\n"
        f"stderr (tail):\n{result.stderr[-1500:]}")
    printed = re.search(r"^   ALL  omega_m = +(\S+) \+- (\S+)$", result.stdout,
                        re.MULTILINE)
    assert printed, result.stdout[-1500:]
    omega_m, spread = float(printed[1]), float(printed[2])
    assert abs(omega_m - 0.315) < 0.5 * 0.007, printed[0]
    assert spread == pytest.approx(0.007, abs=0.0005), printed[0]
    fitres = tmp_path / "fit" / "FITOPT000_MUOPT000" / "sim.FITRES.TEXT"
    assert sum(line.startswith("SN:")
               for line in fitres.read_text().splitlines()) > 0

    written = ["MANIFEST.yml",
               "sim/sim.LIST", "sim/sim_HEAD.FITS.gz", "sim/sim_PHOT.FITS.gz",
               "sim/sim.README", "sim/sim.DUMP",
               "covariance/hubble_diagram.txt", "covariance/covtot_factorized_000.npz",
               "covariance/INFO.YML",
               "cosmology/cosmology.wfit.yaml", "cosmology/cosmology.cospar"]
    written += [f"{stage}/{cell}/{table}"
                for stage, table in (("fit", "sim.FITRES.TEXT"),
                                     ("fit", "sim.SNANA.TEXT"),
                                     ("distances", "distances.M0DIF"),
                                     ("distances", "distances.FITRES"))
                for cell in ("FITOPT000_MUOPT000", "FITOPT001_MUOPT000")]
    written += [f"{stage}/DONE.yml" for stage in
                ("sim", "fit", "distances", "covariance", "cosmology")]
    for name in written:
        assert (tmp_path / name).is_file(), f"the page names {name}, which is absent"


def test_a_set_up_and_stop_input_generates_nothing(fixture_dir, tmp_path):
    """INIT_ONLY is one of SNANA's set-up-and-stop modes: prepare carries it
    out and returns no simulator, so the stage writes no version."""
    sim = Sim("sim", fixture_dir / "input.input",
              overrides={"NGENTOT_LC": 20, "RANSEED": 3, "INIT_ONLY": 1})

    outdir = run(sim, outdir=tmp_path)["sim"]

    assert not list(outdir.glob("*HEAD.FITS*"))
    assert not list(outdir.glob("*.LIST"))


@pytest.mark.parametrize('snana_output', [False, True])
def test_retained_data_reselection_matches_a_fresh_load(
        fixture_dir, tmp_path, snana_output):
    """CID synchronization on the retained data writes the same tables, pre-fit
    flags included, as fitting a fresh load of the same CIDs."""
    import dataclasses

    data = tmp_path / 'sample'
    data.mkdir()
    for table in ('HEAD', 'PHOT'):
        (data / f'sample_{table}.FITS.gz').symlink_to(fixture_dir / f'{table}.FITS.gz')
    (data / 'sample.LIST').write_text('sample_HEAD.FITS\n')
    stage = Fit('fit', fixture_dir / 'fit.nml',
                overrides={'BATCH_SIZE': 4, 'OPT_SNCID_LIST': 1,
                           'SNTABLE_LIST': 'SNANA FITRES' if snana_output else 'FITRES'})
    configs, groups = stage._groups()
    files = stage._files(configs)[(0, 0)]
    model = groups[0][1]
    config = fit_run._at_version(configs[(0, 0)], data)
    retained = loader.load(config, model, files['kcor'], [data], files['shared'],
                            keep_preselection=True)
    assert retained[0].n_sne > 1
    config = dataclasses.replace(config, selection=dataclasses.replace(
        config.selection, sncid_list=list(retained[0].snids[:2])))
    expected = loader.load(config, model, files['kcor'], [data], files['shared'])
    fitter = prepare(config, model, files['kcor'], expected[0])

    def tables(prepared_data, prefix):
        prefixed = dataclasses.replace(config, output=dataclasses.replace(
            config.output, textfile_prefix=str(tmp_path / prefix)))
        fit_run.fit_dataset(fitter, prefixed, files['kcor'], files['shared'],
                            prepared_data=prepared_data)
        return {path.name.removeprefix(prefix): [
                    line for line in path.read_text().splitlines() if 'CPU' not in line]
                for path in tmp_path.glob(f'{prefix}.*')}

    reselected = tables(retained, 'reselected')
    assert reselected.keys() == ({'.FITRES.TEXT', '.SNANA.TEXT'} if snana_output
                                 else {'.FITRES.TEXT'})
    assert reselected == tables(None, 'fresh')


def test_combined_yaml_simulations_preserve_prefix_and_id_overrides(tmp_path, fixture_root):
    """Combined YAML runs preserve explicit prefixes and CID offsets in real output."""
    from astropy.table import Table

    data = fixture_root / 'data'
    settings = dict(genmodel=str(data / 'models/SALT3.K21'), genfilters='ugrizY',
                    simlib_file=str(data / 'simlib/JAX_TEST_LSST_DENSE.SIMLIB'),
                    kcor_file=str(data / 'kcor/JAX_TEST_LSST.fits'), random_seed=42,
                    redshift_range=[.1, .2], trest_range=[-10, 40],
                    peakmjd_range=[60540, 61460], sigma_ratio=0., ngentot_lc=2,
                    format_mask=32, opt_mwebv=0,
                    dndz_entries=[['POWERLAW', 2.5e-5, 1.5]],
                    populations={
                        'SALT2x1': {'peak': 0., 'range': [0., 0.]},
                        'SALT2c': {'peak': 0., 'range': [0., 0.]},
                        'SALT2ALPHA': {'peak': .14, 'range': [.14, .14]},
                        'SALT2BETA': {'peak': 3.1, 'range': [3.1, 3.1]}})
    path = tmp_path / 'sim.yaml'
    path.write_text(yaml.safe_dump(settings))
    overrides = [{}, {'genprefix': 'custom', 'cidoff': 12}]
    stage = Sim('combined', [path, path], overrides=overrides, batch_size=2)
    output = stage.run({}, tmp_path / 'combined')

    listed = (output / 'combined.LIST').read_text().split()
    assert listed == ['combined_sim_0_HEAD.FITS', 'custom_HEAD.FITS']
    custom = Table.read(output / 'custom_HEAD.FITS.gz', hdu=1)
    assert sorted(int(cid) for cid in custom['SNID']) == [13, 14]
    assert overrides == [{}, {'genprefix': 'custom', 'cidoff': 12}]
