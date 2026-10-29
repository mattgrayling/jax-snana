"""The distance, covariance and cosmology stages, on real tables.

A distance fit needs a few hundred events to say anything, which is a large
simulation. What these stages promise does not depend on where the table came
from, so fitted tables from the fixtures stand in for the fit above them, and
the chain from there to a fitted cosmology runs for real.
"""


import os
from contextlib import chdir

import numpy as np
import pytest
import yaml

from jax_pipeline import BBC, Cosmology, Covariance, Fit, run
from jax_salt2mu import parse_bbc_input, run_bbc
from jax_salt2mu.tablevar import EVENT_TYPE_BIASCOR, EVENT_TYPE_CCPRIOR, EVENT_TYPE_DATA
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import (dense_factorized_covariance, open_text, read_fitres,
                                       read_fitres_multi, write_varnames_table)

_DIR = os.path.abspath(os.path.join("tests", "fixtures", "snana", "bbc_des_beams"))
_DATA = os.path.join(_DIR, "DATA.FITRES")
_SPLIT_A = os.path.join(_DIR, "DATA_SPLIT_A.FITRES")
_SPLIT_B = os.path.join(_DIR, "DATA_SPLIT_B.FITRES")
_BIASCOR = os.path.join(_DIR, "BIASCOR.FITRES")
_GROUP = "DES(zbin=0.15:x1bin=2.0:cbin=0.2)"
_NOMINAL = "nominal"
_KEYS = {"nzbin": 10, "zmin": 0.1, "zmax": 1.0, "u1": 1, "u2": 1}
# SNANA fitted these tables, so they record no simulated cosmology; this is
# the one SNANA's create_covariance falls back to for them
_COSMOLOGY = {"OMEGA_MATTER": 0.315, "OMEGA_LAMBDA": 0.685}


def test_realisations_share_correction_maps_within_one_run(tmp_path, caplog):
    correction = Fit("biascor", fitres=_BIASCOR)
    keys = {**_KEYS, "opt_biascor": 112, "min_per_cell_biascor": 3,
            "snrmin_sigint_biascor": 60., "surveygroup_biascor": _GROUP}
    stages = [BBC(f"distances{i}", data=Fit(f"data{i}", fitres=_DATA),
                  biascor=correction, overrides=keys) for i in range(2)]
    caplog.set_level('INFO', logger='jax_salt2mu.preparation')
    outputs = run(*stages, outdir=tmp_path / "first")
    map_timings = [record for record in caplog.records
                   if 'seconds' in record.message and 'biascor_maps' in record.message]
    assert len(map_timings) == 1
    a, b = (outputs[f"distances{i}"]["nominal"]["M0DIF"].df for i in range(2))
    np.testing.assert_array_equal(a["MUDIF"], b["MUDIF"])

    caplog.clear()
    run(*stages, outdir=tmp_path / "first")
    assert not caplog.records
    run(*stages, outdir=tmp_path / "first", refresh='distances1')
    assert sum('seconds' in r.message and 'biascor_maps' in r.message for r in caplog.records) == 1
    caplog.clear()
    run(*stages, outdir=tmp_path / "second")
    assert sum('seconds' in r.message and 'biascor_maps' in r.message for r in caplog.records) == 1


def test_grouped_bbc_failure_preserves_completed_grid_and_resumes(tmp_path):
    data = Fit('data', fitres=_DATA)
    first = BBC('distances0', data=data, overrides=_KEYS)
    second = BBC('distances1', data=data, overrides={**_KEYS, 'CUTWIN c': '100 200'})
    with pytest.raises(SNANAError, match='no z bin'):
        run(first, second, outdir=tmp_path)
    done = tmp_path / 'distances0' / 'DONE.yml'
    completed = done.read_bytes()
    assert not (tmp_path / 'distances1' / 'DONE.yml').exists()
    manifest = yaml.safe_load((tmp_path / 'MANIFEST.yml').read_text())
    assert manifest['stages']['distances0']['status'] == 'completed'
    assert manifest['stages']['distances1']['status'] == 'failed'
    second = BBC('distances1', data=data, overrides=_KEYS)
    outputs = run(first, second, outdir=tmp_path)
    assert done.read_bytes() == completed
    np.testing.assert_array_equal(outputs['distances0']['nominal']['M0DIF'].df['MUDIF'],
                                  outputs['distances1']['nominal']['M0DIF'].df['MUDIF'])


def test_future_grid_preparation_error_identifies_cell_and_shared_phase(tmp_path):
    data = Fit('data', fitres=_DATA)
    biascor = Fit('biascor', fitres=_BIASCOR)
    first = BBC('distances0', data=data, biascor=biascor, overrides=_KEYS)
    second = BBC('distances1', data=data, biascor=biascor,
                 overrides={**_KEYS, 'opt_biascor': 8})

    with pytest.raises(SNANAError, match=r'distances1.*\(0, 0\).*opt_biascor bit 8'):
        run(first, second, outdir=tmp_path)
    manifest = yaml.safe_load((tmp_path / 'MANIFEST.yml').read_text())
    assert manifest['current_stage'] == 'shared BBC preparation'
    assert 'distances1' in manifest['error']
    for name in ('distances0', 'distances1'):
        assert manifest['stages'][name]['status'] == 'failed'
        assert not (tmp_path / name / 'DONE.yml').exists()


def test_future_grid_parser_error_is_attributed_to_that_stage(tmp_path):
    data = Fit('data', fitres=_DATA)
    first = BBC('distances0', data=data, overrides=_KEYS)
    second = BBC('distances1', data=data, overrides={**_KEYS, 'opt_chi2max': 2})
    with pytest.raises(SNANAError, match=r'distances1.*\(0, 0\).*opt_chi2max'):
        run(first, second, outdir=tmp_path)
    manifest = yaml.safe_load((tmp_path / 'MANIFEST.yml').read_text())
    assert manifest['current_stage'] == 'distances1'
    assert manifest['stages']['distances1']['status'] == 'failed'
    assert manifest['stages']['distances0']['status'] == 'preparing'
    assert not (tmp_path / 'distances0' / 'DONE.yml').exists()



def _chain(outdir):
    fitted = Fit("fitted", fitres=_DATA)
    bbc = BBC("distances", data=fitted, overrides=_KEYS,
              muopts={"tight_chi2": {"chi2max": 16}})
    covariance = Covariance("covariance", distances=bbc,
                            covopts=["[NOSYS] [=DEFAULT,=DEFAULT]"],
                            overrides={"WRITE_MASK_COV": 7, "COSMOLOGY": _COSMOLOGY})
    fit = Cosmology("cosmology", covariance=covariance, covopts=("ALL", "NOSYS"),
                    overrides={"model": "lcdm", "ompri": 0.315, "dompri": 0.007})
    return fit, run(fit, outdir=outdir)


@pytest.fixture(scope="module")
def distances(tmp_path_factory):
    """Distance fits, their covariance, and a cosmology fit of it."""
    outdir = tmp_path_factory.mktemp("distances")
    _, results = _chain(outdir)
    return outdir, results


def _nfit(cell_output) -> int:
    return int(cell_output["M0DIF"].df["NFIT"].sum())


def _direct_bbc(prefix, keys, tables) -> None:
    """Run the BBC program on these keys and tables, writing at ``prefix``.

    The same call a BBC stage cell makes, so the two outputs compare exactly.
    """
    overrides = [f"{k}={v}" for k, v in keys.items()] + [f"prefix={prefix}"]
    run_bbc(parse_bbc_input(None, overrides=overrides), tables=tables)


def _written_mudif(prefix, keys, tables):
    """A direct distance fit's MUDIF as its M0DIF file holds it, which is
    what a stage hands on, so the two compare exactly."""
    _direct_bbc(prefix, keys, tables)
    return read_fitres(f"{prefix}.M0DIF").df["MUDIF"]


def test_a_muopt_changes_the_distances(distances):
    """A tighter chi-square cut keeps fewer events, or the override never
    reached the distance fit."""
    _, results = distances
    grid = results["distances"]

    assert _nfit(grid["tight_chi2"]) < _nfit(grid[_NOMINAL])


def test_every_cell_writes_its_own_outputs(distances):
    outdir, _ = distances

    for cell in ("FITOPT000_MUOPT000", "FITOPT000_MUOPT001"):
        assert (outdir / "distances" / cell / "distances.M0DIF").is_file()
        assert (outdir / "distances" / cell / "distances.FITRES").is_file()
        assert (outdir / "distances" / cell / "distances.COV").is_file()
    assert (outdir / "covariance" / "INFO.YML").is_file()
    assert (outdir / "cosmology" / "cosmology.wfit.yaml").is_file()


def _warnings(path) -> list[str]:
    return [line for line in path.read_text().splitlines() if "WARNING" in line]


def test_a_cell_is_what_salt2mu_writes(distances, tmp_path):
    """The stage runs the BBC program rather than a writer of its own, so a
    cell holds the same files, with the same warnings in them, as salt2mu run
    on the same keys and table."""
    outdir, _ = distances
    cell = outdir / "distances" / "FITOPT000_MUOPT000"

    _direct_bbc(tmp_path / "direct", _KEYS,
                {EVENT_TYPE_DATA: read_fitres_multi([_DATA])})

    assert ({path.suffix for path in cell.iterdir()}
            == {path.suffix for path in tmp_path.iterdir()})
    for suffix in (".M0DIF", ".FITRES"):
        assert _warnings(cell / f"distances{suffix}") == \
            _warnings(tmp_path / f"direct{suffix}")
    assert _warnings(cell / "distances.M0DIF"), \
        "no warning in either file, so this proves nothing about the warnings"


def test_a_simulated_sample_is_reported_as_such(distances):
    """The distance fit's headers come from the config the fit ran with, so
    a simulation is unblinded, and the covariance carries ISDATA_REAL on."""
    outdir, results = distances

    m0dif = (outdir / "distances" / "FITOPT000_MUOPT000" / "distances.M0DIF").read_text()
    assert "ISDATA_REAL:   0" in m0dif and "BLINDED" not in m0dif
    assert results["covariance"]["isdata_real"] == 0
    info = yaml.safe_load((outdir / "covariance" / "INFO.YML").read_text())
    assert info["ISDATA_REAL"] == 0 and info["BBC_INFO_FILE"] is None
    # the nominal FITRES header's version keys, as the command line writes them
    assert "VERSION_PHOTOMETRY(DATA)" in info
    # WRITE_MASK_COV: 7 from the stage's overrides names a covsys file per covopt
    assert all(entry.split()[1] != "None" for entry in info["COVOPTS"].values())


def test_the_covariance_is_built_from_the_grid(distances):
    _, results = distances
    held = results["covariance"]

    assert set(held["covariances"]) == {"ALL", "NOSYS"}
    assert not np.array_equal(_inverse(held), _inverse(held, "NOSYS"))
    assert len(held["hubble_diagram"]) == 10


def test_the_cosmology_fits_every_covopt(distances):
    _, results = distances
    fits = results["cosmology"]

    assert list(fits) == ["ALL", "NOSYS"]
    for result in fits.values():
        assert set(result["point_estimate"]) <= {"omega_m", "w0", "wa"}
        assert "std" in result["parameters"]["omega_m"]


def test_a_second_run_loads_the_whole_chain(distances, caplog):
    """Every stage is loaded from its files and hands on the same result."""
    outdir, results = distances

    caplog.set_level("INFO", logger="jax_pipeline.run")
    _, again = _chain(outdir)
    assert f"Outputs: {outdir}" in caplog.text
    assert f"Run manifest: {outdir / 'MANIFEST.yml'}" in caplog.text
    assert f"Cosmology summary: {outdir / 'COSMOLOGY_SUMMARY.csv'}" in caplog.text
    assert (outdir / 'COSMOLOGY_SUMMARY.csv').is_file()

    for cell in results["distances"]:
        np.testing.assert_array_equal(
            again["distances"][cell]["M0DIF"].df["MUDIF"],
            results["distances"][cell]["M0DIF"].df["MUDIF"])
    assert again["cosmology"]["ALL"]["point_estimate"] == \
        results["cosmology"]["ALL"]["point_estimate"]


def test_the_cosmology_writes_only_in_its_own_directory(tmp_path):
    """wfit names two outputs after the Hubble diagram it read and there is
    no file here to name them after; they must land in the stage directory,
    not where the run was launched."""
    elsewhere = tmp_path / "launched_from"
    elsewhere.mkdir()
    with chdir(elsewhere):
        _chain(tmp_path / "run")

    assert list(elsewhere.iterdir()) == []


def test_two_surveys_are_concatenated_as_salt2mu_does(tmp_path):
    """A list of data stages is one distance fit over their union, equal to
    SALT2mu's datafile=a,b on the same two files."""
    bbc = BBC("distances", data=[Fit("a", fitres=_SPLIT_A), Fit("b", fitres=_SPLIT_B)],
              overrides=_KEYS)

    results = run(bbc, outdir=tmp_path)
    direct = _written_mudif(
        tmp_path / "direct", _KEYS,
        {EVENT_TYPE_DATA: read_fitres_multi([_SPLIT_A, _SPLIT_B])})

    np.testing.assert_array_equal(results["distances"][_NOMINAL]["M0DIF"].df["MUDIF"],
                                  direct)


def _survey_fits(directory, source, labels, shifts):
    """A survey's fitted tables as SNANA's batch fit leaves them: a version
    directory whose FITOPT000 is the source table and each further FITOPT
    the same table with the colour shifted, and beside it the SUBMIT.INFO
    labelling them."""
    version = directory / "VERSION"
    version.mkdir(parents=True)
    table = read_fitres(source).df
    for i, shift in enumerate((0.0, *shifts)):
        df = table.assign(c=table["c"] + shift)
        write_varnames_table(str(version / f"FITOPT{i:03d}.FITRES"), list(df),
                             {name: df[name].to_numpy() for name in df}, fmt={})
    (directory / "SUBMIT.INFO").write_text(yaml.safe_dump({"FITOPT_LIST": [
        [f"FITOPT{i:03d}", label, None]
        for i, label in enumerate((None, *labels))]}))
    return Fit(directory.name, fitres=version)


def test_each_survey_brings_its_own_systematics(tmp_path):
    """The FITOPT axis is the union over surveys: a shared label refits both,
    a survey's own label refits it alone with the other at its nominal."""
    a = _survey_fits(tmp_path / "a", _SPLIT_A, ("both", "a_shift"), (0.03, 0.02))
    b = _survey_fits(tmp_path / "b", _SPLIT_B, ("both", "b_shift"), (0.03, 0.01))
    bbc = BBC("distances", data=[a, b], overrides=_KEYS)

    results = run(bbc, outdir=tmp_path / "run")
    grid = results["distances"]

    assert bbc.systematics() == (("both", "a_shift", "b_shift"), ())
    assert set(grid) == {_NOMINAL, "both", "a_shift", "b_shift"}
    for label, (a_file, b_file) in {"both": (1, 1), "a_shift": (2, 0), "b_shift": (0, 2)}.items():
        direct = _written_mudif(
            tmp_path / f"{label}", _KEYS,
            {EVENT_TYPE_DATA: read_fitres_multi([
                str(tmp_path / "a" / "VERSION" / f"FITOPT{a_file:03d}.FITRES"),
                str(tmp_path / "b" / "VERSION" / f"FITOPT{b_file:03d}.FITRES")])})
        np.testing.assert_array_equal(grid[label]["M0DIF"].df["MUDIF"], direct)
    assert not np.array_equal(grid["a_shift"]["M0DIF"].df["MUDIF"],
                              grid["b_shift"]["M0DIF"].df["MUDIF"])


def test_swapped_in_data_may_not_bring_its_own_systematics(tmp_path):
    """A MUOPT's data is the nominal data reclassified, so a FITOPT the
    nominal lacks is a mistake, not a new column of the grid."""
    other = _survey_fits(tmp_path / "other", _SPLIT_A, ("extra",), (0.02,))
    bbc = BBC("distances", data=Fit("a", fitres=_SPLIT_A), overrides=_KEYS,
              muopts={"swap": {"datafile": other}})

    with pytest.raises(SNANAError, match="FITOPTs the nominal data lacks"):
        bbc.systematics()


def test_a_muopt_may_swap_the_biascor_stage(bbc_biascor_variants, tmp_path):
    """A MUOPT naming another bias-correction stage consumes it in place of
    the nominal, as Pippin's MUOPT names another simulation; the cell must
    equal a direct fit on that table, and the two corrections must differ."""
    mass_biascor = str(bbc_biascor_variants / "BIASCOR_MASS.FITRES")
    keys = {**_KEYS, "min_per_cell_biascor": 3, "surveygroup_biascor": _GROUP}
    bbc = BBC("distances", data=Fit("fitted", fitres=_DATA),
              biascor=Fit("biascor", fitres=_BIASCOR), overrides=keys,
              muopts={"mass": {"simfile_biascor": Fit("biascor_mass", fitres=mass_biascor)}})

    results = run(bbc, outdir=tmp_path)
    direct = _written_mudif(
        tmp_path / "direct", keys,
        {EVENT_TYPE_DATA: read_fitres_multi([_DATA]),
         EVENT_TYPE_BIASCOR: read_fitres_multi([mass_biascor])})
    grid = results["distances"]

    assert set(bbc.inputs()) == {"data:fitted", "biascor:biascor", "biascor:biascor_mass"}
    np.testing.assert_array_equal(grid["mass"]["M0DIF"].df["MUDIF"],
                                  direct)
    assert not np.array_equal(grid["mass"]["M0DIF"].df["MUDIF"],
                              grid[_NOMINAL]["M0DIF"].df["MUDIF"]), \
        "the two bias corrections agree, so this proves nothing about the swap"


def test_an_unbinned_covariance_reads_the_fitres_not_the_m0dif(tmp_path):
    """Binned reads the M0DIF, unbinned and rebin the FITRES, as
    get_hubble_diagrams selects (create_covariance.py:713-717)."""
    fitted = Fit("fitted", fitres=_DATA)
    bbc = BBC("distances", data=fitted, overrides=_KEYS)
    unbinned = Covariance("covariance", distances=bbc, mode="unbinned",
                          overrides={"COSMOLOGY": _COSMOLOGY})
    results = run(unbinned, outdir=tmp_path)

    diagram = results["covariance"]["hubble_diagram"]
    events = results["distances"][_NOMINAL]["FITRES"].df
    assert "CID" in diagram.columns
    assert len(diagram) == len(events)
    assert sorted(diagram["CID"].astype(str)) == sorted(events["CID"].astype(str))


def test_the_covariance_stage_refuses_what_it_cannot_honour(tmp_path):
    """The stage applies its own covopts, scales and mode, so a key of the
    input file it would drop must abort rather than reach INFO.YML unapplied.

    Rebinning requires an explicit coordinate mapping.
    """
    scales = tmp_path / "scales.yml"
    scales.write_text("SOMESYS: 2.0\n")
    dropped = tmp_path / "dropped.yml"
    dropped.write_text(f"SYS_SCALE_FILE: {scales}\n")
    fitted = Fit("fitted", fitres=_DATA)
    bbc = BBC("distances", data=fitted, overrides=_KEYS)
    stage = Covariance("covariance", distances=bbc, input_file=dropped)
    with pytest.raises(SNANAError, match="cannot honour sys_scale_file"):
        stage.run({"distances": {}}, tmp_path)

    rebin = Covariance("covariance", distances=bbc, mode="rebin")
    with pytest.raises(SNANAError, match="rebin needs REBIN_BINS"):
        rebin.run({"distances": {}}, tmp_path)

    # overrides are input-file keys, refused as the file's own would be
    refused = Covariance("covariance", distances=bbc, overrides={"CALIBRATORS": ["x"]})
    with pytest.raises(SNANAError, match="CALIBRATORS.*unbinned"):
        refused.run({"distances": {}}, tmp_path)

    with pytest.raises(SNANAError, match="create_covariance builds one of"):
        Covariance("covariance", distances=bbc, mode="unbined")


def test_exported_bbc_grid_builds_the_same_covariance(distances, tmp_path):
    from jax_create_cov import create_covariance, load_config

    run_dir, results = distances
    bbc = BBC("distances", data=Fit("fitted", fitres=_DATA), overrides=_KEYS,
              muopts={"tight_chi2": {"chi2max": 16}})
    exported = tmp_path / "exported"
    bbc.export_covariance_inputs(run_dir, exported)
    config_path = tmp_path / "covariance.yml"
    config_path.write_text(yaml.safe_dump({
        "INPUT_DIR": str(exported), "VERSION": "distances",
        "COSMOLOGY": _COSMOLOGY, "COVOPTS": ["[NOSYS] [=DEFAULT,=DEFAULT]"],
    }))
    actual = create_covariance(load_config(config_path))
    reference = results["covariance"]["hubble_diagram"]
    np.testing.assert_allclose(actual.hubble_diagram["MU"], reference["MU"])
    for label, entry in actual.covariances.items():
        cov = entry["matrix"]
        total = cov + np.diag(actual.hubble_diagram["MUERR"].to_numpy() ** 2)
        np.testing.assert_allclose(
            np.linalg.inv(total), _inverse(results["covariance"], label),
            rtol=1e-6, atol=1e-6)
    with pytest.raises(SNANAError, match="already exists"):
        bbc.export_covariance_inputs(run_dir, exported)


@pytest.mark.parametrize("named", [True, False])
def test_rebinned_pipeline_matches_export(distances, tmp_path, named):
    from dataclasses import replace
    from jax_create_cov import create_covariance, load_config

    run_dir, results = distances
    bbc = BBC("distances", data=Fit("fitted", fitres=_DATA), overrides=_KEYS,
              muopts={"tight_chi2": {"chi2max": 16}})
    bins = {"x1": 2, "c": 3}
    options = {"rebin_bins": bins} if named else {"nbin_x1": 2, "nbin_c": 3}
    stage = Covariance("rebin", distances=bbc, mode="rebin",
                       overrides={"COSMOLOGY": _COSMOLOGY}, **options)
    outdir = tmp_path / 'rebin'
    outdir.mkdir()
    actual = stage.run({'distances': results['distances']}, outdir)
    exported = tmp_path / 'export'
    bbc.export_covariance_inputs(run_dir, exported)
    source = tmp_path / 'cov.yml'
    source.write_text(yaml.safe_dump({
        'INPUT_DIR': str(exported), 'VERSION': 'distances', 'REBIN_BINS': bins,
        'COSMOLOGY': _COSMOLOGY}, sort_keys=False))
    config = load_config(source)
    if not named:
        config = replace(config, rebin_bins=None, nbin_x1=2, nbin_c=3)
    expected = create_covariance(config)
    assert actual['hubble_diagram']['ROW'].tolist() == expected.hubble_diagram['ROW'].tolist()
    np.testing.assert_allclose(actual['hubble_diagram']['MU'], expected.hubble_diagram['MU'], atol=5e-5)
    np.testing.assert_allclose(_inverse(actual), expected.covtot_inv(), rtol=1e-6)


def test_the_covariance_takes_the_biascor_simulations_cosmology(tmp_path):
    """The light-curve fit records a simulation's cosmology in its FITRES
    header, BBC carries the bias-correction sample's into its outputs, and
    the covariance builds its reference distances from it. The same values
    given as COSMOLOGY build the same covariance; different ones abort."""
    with open_text(_BIASCOR) as stream:
        biascor_text = stream.read()
    recorded = tmp_path / "BIASCOR.FITRES"
    recorded.write_text("# SIM_OMEGA_MATTER: 0.2812\n# SIM_OMEGA_LAMBDA: 0.7188\n"
                        "# SIM_w0_LAMBDA: -1.0000\n# SIM_wa_LAMBDA: 0\n"
                        "# SIM_MUSHIFT: 0\n" + biascor_text)
    keys = {**_KEYS, "min_per_cell_biascor": 3, "surveygroup_biascor": _GROUP}
    bbc = BBC("distances", data=Fit("fitted", fitres=_DATA),
              biascor=Fit("biascor", fitres=recorded), overrides=keys)
    results = run(Covariance("covariance", distances=bbc), outdir=tmp_path / "run")

    m0dif = (tmp_path / "run" / "distances" / "FITOPT000_MUOPT000"
             / "distances.M0DIF").read_text()
    assert "SIM_OMEGA_MATTER(BIASCOR):" in m0dif
    info = yaml.safe_load((tmp_path / "run" / "covariance" / "INFO.YML").read_text())
    assert info["COSPAR_BIASCOR"] == {"OMEGA_MATTER": 0.2812, "OMEGA_LAMBDA": 0.7188,
                                      "w0_LAMBDA": -1.0, "wa_LAMBDA": 0.0, "MUSHIFT": 0.0}

    def given(cosmology, outdir):
        outdir.mkdir()
        return Covariance("covariance", distances=bbc,
                          overrides={"COSMOLOGY": cosmology}).run(
            {"distances": results["distances"]}, outdir)

    same = given({"OMEGA_MATTER": 0.2812, "OMEGA_LAMBDA": 0.7188}, tmp_path / "same")
    np.testing.assert_array_equal(same["hubble_diagram"]["MUSIM"],
                                  results["covariance"]["hubble_diagram"]["MUSIM"])
    with pytest.raises(SNANAError, match="disagrees"):
        given(_COSMOLOGY, tmp_path / "other")


def test_a_covariance_needing_a_cosmology_without_one_aborts(distances, tmp_path):
    """A simulated sample's MUSIM needs the reference distances, and SNANA's
    tables record no cosmology: nothing is substituted for it."""
    _, results = distances
    bbc = BBC("distances", data=Fit("fitted", fitres=_DATA), overrides=_KEYS,
              muopts={"tight_chi2": {"chi2max": 16}})
    with pytest.raises(SNANAError, match="set COSMOLOGY"):
        Covariance("covariance", distances=bbc).run(
            {"distances": results["distances"]}, tmp_path)


def test_correction_fitopt_selection_and_replacement(bbc_biascor_variants, tmp_path):
    """Selected and replaced corrections agree with direct BBC fits."""
    mass_biascor = str(bbc_biascor_variants / "BIASCOR_MASS.FITRES")
    data = _survey_fits(tmp_path / "data", _DATA, ("CAL",), (0.02,))
    correction = _survey_fits(tmp_path / "correction", _BIASCOR, ("ALT",), (0.04,))
    keys = {**_KEYS, "min_per_cell_biascor": 3, "surveygroup_biascor": _GROUP}
    bbc = BBC("distances", data=data, biascor=correction,
              biascor_fitopts=["ALT"], replace_biascor={"CAL": mass_biascor},
              overrides=keys)
    results = run(bbc, outdir=tmp_path / "run")["distances"]
    for label, number, correction_file in (
        ("nominal", 0, str(tmp_path / "correction" / "VERSION" / "FITOPT001.FITRES")),
        ("CAL", 1, mass_biascor),
    ):
        direct = _written_mudif(tmp_path / f"direct_{label}", keys, {
            EVENT_TYPE_DATA: read_fitres_multi([str(tmp_path / "data" / "VERSION" / f"FITOPT{number:03d}.FITRES")]),
            EVENT_TYPE_BIASCOR: read_fitres_multi([correction_file]),
        })
        np.testing.assert_array_equal(results[label]["M0DIF"].df["MUDIF"], direct)


@pytest.mark.parametrize("role,event_type", [("biascor", EVENT_TYPE_BIASCOR), ("ccprior", EVENT_TYPE_CCPRIOR)])
def test_muopt_correction_selection_takes_precedence(bbc_biascor_variants, tmp_path, role, event_type):
    mass_biascor = str(bbc_biascor_variants / "BIASCOR_MASS.FITRES")
    data = _survey_fits(tmp_path / "data", _DATA, ("CAL",), (0.02,))
    correction = _survey_fits(tmp_path / "correction", _BIASCOR, ("ALT",), (0.04,))
    bbc = BBC("distances", data=data, **{
        role: correction, f"{role}_fitopts": ["ALT"],
        f"replace_{role}": {"CAL": mass_biascor},
        f"muopt_{role}_fitopts": {"swap": ["ALT"]},
    }, overrides=_KEYS, muopts={"swap": {f"simfile_{role}": correction}})
    table = read_fitres(_BIASCOR)
    alt = read_fitres(str(tmp_path / "correction" / "VERSION" / "FITOPT001.FITRES"))
    inputs = {"data:data": {"nominal": read_fitres(_DATA), "CAL": read_fitres(_DATA)},
              f"{role}:correction": {"nominal": table, "ALT": alt}}
    _, actual = bbc._cell(inputs, (1, 1), {})
    np.testing.assert_array_equal(actual[event_type][0].df["c"], alt.df["c"])


def test_sparse_grid_runs_only_requested_cells_and_exports(tmp_path):
    from jax_create_cov import create_covariance, load_config
    data = _survey_fits(tmp_path / "data", _DATA, ("CAL",), (0.02,))
    bbc = BBC("distances", data=data, overrides=_KEYS,
              muopts={"tight": {"chi2max": 16}}, fitopt_muopt=["0+0"])
    stage = Covariance("covariance", distances=bbc, overrides={"COSMOLOGY": _COSMOLOGY})
    results = run(stage, outdir=tmp_path / "run")
    assert set(results["distances"]) == {"nominal", "CAL", "tight"}
    assert not (tmp_path / "run" / "distances" / "FITOPT001_MUOPT001").exists()
    bbc.export_covariance_inputs(tmp_path / "run", tmp_path / "export")
    source = tmp_path / "cov.yml"
    source.write_text(yaml.safe_dump({"INPUT_DIR": str(tmp_path / "export"), "VERSION": "distances",
                                      "COSMOLOGY": _COSMOLOGY}))
    expected = create_covariance(load_config(source))
    np.testing.assert_allclose(_inverse(results["covariance"]), expected.covtot_inv(), rtol=1e-6)
    # A requested cross-combination must include its reference for covariance.
    bbc.fitopt_muopt = ["0&0", "1&1"]
    with pytest.raises(SNANAError, match="reference cells"):
        stage.run({"distances": {}}, tmp_path)


def test_partitions_are_disjoint_complete_and_keep_corrections(tmp_path):
    data = Fit("data", fitres=_DATA)
    correction = Fit("correction", fitres=_BIASCOR)
    input_data = read_fitres(_DATA)
    inputs = {"data:data": {"nominal": input_data},
              "biascor:correction": {"nominal": read_fitres(_BIASCOR)}}
    sets = []
    for index in (1, 2, 3):
        bbc = BBC(f"split{index}", data=data, biascor=correction, overrides=_KEYS,
                  split_count=3, split_index=index)
        _, tables = bbc._cell(inputs, (0, 0), {})
        ids = set(tables[EVENT_TYPE_DATA].df.CID.astype(str))
        assert all(int(cid) % 3 + 1 == index for cid in ids)
        assert len(tables[EVENT_TYPE_BIASCOR][0].df) == len(inputs["biascor:correction"]["nominal"].df)
        sets.append(ids)
    assert set.union(*sets) == set(input_data.df.CID.astype(str))
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])


def _inverse(held, label="ALL"):
    entry = held["covariances"][label]
    if entry["kind"] == "factorized":
        return np.linalg.inv(dense_factorized_covariance(entry))
    return entry["matrix"]


def test_muopt_reordering_invalidates_cached_labels(tmp_path):
    """Reordering MUOPTs reruns BBC while preserving each named cut's results."""
    fit = Fit("fit", fitres=_DATA)
    original = BBC("bbc", data=fit, overrides=_KEYS,
                   muopts={"tight": {"chi2max": 9}, "loose": {"chi2max": 25}})
    outdir = tmp_path / "run"
    before = run(original, outdir=outdir)["bbc"]
    assert _nfit(before["tight"]) < _nfit(before["loose"])
    run(original, outdir=outdir)
    manifest = yaml.safe_load((outdir / "MANIFEST.yml").read_text())
    assert manifest["stages"]["bbc"]["loaded"]

    reordered = BBC("bbc", data=fit, overrides=_KEYS,
                    muopts=dict(reversed(original.muopts.items())))
    after = run(reordered, outdir=outdir)["bbc"]
    for label in ("nominal", "tight", "loose"):
        for column in ("MUDIF", "NFIT"):
            np.testing.assert_array_equal(after[label]["M0DIF"].df[column],
                                          before[label]["M0DIF"].df[column])
        np.testing.assert_array_equal(after[label]["FITRES"].df["CID"],
                                      before[label]["FITRES"].df["CID"])
    manifest = yaml.safe_load((outdir / "MANIFEST.yml").read_text())
    assert manifest["stages"]["fit"]["loaded"]
    assert not manifest["stages"]["bbc"]["loaded"]
