"""Classification stages, on a contaminated table with truth.

The BEAMS fixture's contaminated table carries a fabricated probability and no
truth, so the tests build their own contaminated table from the fixture's
Type Ia data and its core-collapse prior sample, both of which carry
SIM_TEMPLATE_INDEX. The toy classifier then has truth to read, and every
promise of the classification path is pinned: the column values, the join
onto every fitted table, the distance fit reading it, a cut on it reaching
the bias-correction sample, a MUOPT swapping classifiers, training handing a
model to classifying, and a second run loading it all.

SuperNNova runs where it is installed: trained on a jax simulation of Type Ia
and core-collapse events, it must read jax's files as it reads SNANA's, and
its probabilities must carry through to the distance fit.
"""

import importlib.util
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from astropy.io import fits
from astropy.table import Table

from jax_pipeline import BBC, Classify, Fit, Sim, Train, run
from jax_pipeline.classify import Perfect
from jax_pipeline.supernnova import SuperNNova
from jax_salt2mu import parse_bbc_input, run_bbc_fit
from jax_salt2mu.output import write_m0dif
from jax_salt2mu.tablevar import EVENT_TYPE_BIASCOR, EVENT_TYPE_CCPRIOR, EVENT_TYPE_DATA
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import read_fitres, write_varnames_table
from jax_snlc_sim.output.fits import write_head_fits, write_phot_fits

_DIR = os.path.abspath(os.path.join("tests", "fixtures", "snana", "bbc_des_beams"))
_DATA = os.path.join(_DIR, "DATA.FITRES")
_CCPRIOR = os.path.join(_DIR, "CCPRIOR.FITRES")
_BIASCOR = os.path.join(_DIR, "BIASCOR.FITRES")
_NOMINAL = "nominal"
_COLUMN = "PROB_PERFECT"
_SNANA = os.path.abspath(os.path.join("tests", "fixtures", "snana"))
#: the DES Type Ia and K10 core-collapse inputs, on one cadence and calibration
_IA_INPUT = os.path.join(_SNANA, "bbc_des_data", "input.input")
_CC_INPUT = os.path.join(_SNANA, "gentype_map_k10", "input.input")
_FIT_NML = os.path.join(_SNANA, "bbc_des_data", "fit.nml")
#: DES-SN5YR's nominal SuperNNova model, trained on bands named g, i, r, z
_DES_MODEL = Path("tests/fixtures/data/classifiers/SNNTRAINV19_z_TRAINDES_V19/model.pt").resolve()
_DES_BANDS = ("DES-g", "DES-i", "DES-r", "DES-z")
needs_supernnova = pytest.mark.skipif(importlib.util.find_spec("supernnova") is None,
                                      reason="SuperNNova is not installed")
#: the BEAMS fit of the fixture's own parity test, with the classifier column
#: named as this pipeline names it
_KEYS = {"nzbin": 10, "zmin": 0.1, "zmax": 1.0, "u1": 1, "u2": 1,
         "opt_biascor": 112, "min_per_cell_biascor": 3,
         "snrmin_sigint_biascor": 60.0,
         "surveygroup_biascor": "DES(zbin=0.15:x1bin=2.0:cbin=0.2)",
         "typeIa_ccprior": 999, "blindflag": 0, "varname_pIa": _COLUMN,
         # a fixed scalePCC of 0 is SALT2mu's own parval and turns the CC prior
         # off (prep_input, SALT2mu.c:22117), which would leave the classifier's
         # probabilities with nothing to weight
         "p13": 1}



@pytest.fixture(scope="module")
def contaminated(tmp_path_factory):
    """The Ia data with the core-collapse sample mixed in, CIDs kept apart,
    over the columns both carry."""
    ia = read_fitres(_DATA).df
    cc = read_fitres(_CCPRIOR).df
    cc["CID"] = (cc["CID"].astype(int) + 100_000).astype(str)
    mixed = pd.concat([ia[cc.columns], cc], ignore_index=True)
    path = tmp_path_factory.mktemp("contaminated") / "CONTAM.FITRES"
    write_varnames_table(str(path), list(mixed),
                         {name: mixed[name].to_numpy() for name in mixed}, fmt={})
    return path


def _with_truth_column(table, prob_ia, prob_cc):
    """The classifier's column added by hand, for a direct fit to compare to."""
    values = np.where(table.df["SIM_TEMPLATE_INDEX"].astype(int) == 0, prob_ia, prob_cc)
    table.df[_COLUMN] = values
    return table


@pytest.mark.parametrize("prob_ia, prob_cc", [(0.9, 0.1),
                                           (0.99999, 1.234567890123456e-8)])
def test_classification_probabilities(contaminated, tmp_path, prob_ia, prob_cc):
    fitted = Fit("fitted", fitres=contaminated)
    classified = Classify("classified", Perfect(prob_ia=prob_ia, prob_cc=prob_cc), fit=fitted)

    results = run(classified, outdir=tmp_path)
    df = results["classified"][_NOMINAL].df

    truth = df["SIM_TEMPLATE_INDEX"].astype(int) == 0
    assert truth.any() and (~truth).any()
    np.testing.assert_array_equal(df[_COLUMN], np.where(truth, prob_ia, prob_cc))
    assert (tmp_path / "classified" / "predictions.csv").is_file()
    assert _COLUMN in results["classified"][_NOMINAL].varnames_by_file[0]


def test_the_distance_fit_reads_the_joined_column(contaminated, tmp_path):
    """A BEAMS fit through the pipeline equals one on tables carrying the same
    column by hand, so the column reached the distance fit."""
    fitted = Fit("fitted", fitres=contaminated)
    classified = Classify("classified", Perfect(prob_ia=0.9, prob_cc=0.1), fit=fitted)
    bbc = BBC("distances", data=classified, biascor=Fit("biascor", fitres=_BIASCOR),
              ccprior=Fit("ccprior", fitres=_CCPRIOR), overrides=_KEYS)

    results = run(bbc, outdir=tmp_path)
    result, output, zbins, _, _ = run_bbc_fit(
        parse_bbc_input(None, overrides=[f"{k}={v}" for k, v in _KEYS.items()]),
        tables={EVENT_TYPE_DATA: _with_truth_column(read_fitres(str(contaminated)), 0.9, 0.1),
                EVENT_TYPE_BIASCOR: read_fitres(_BIASCOR),
                EVENT_TYPE_CCPRIOR: read_fitres(_CCPRIOR)})
    write_m0dif(tmp_path / "direct.M0DIF", zbins, output, result, output.config)

    np.testing.assert_array_equal(results["distances"][_NOMINAL]["M0DIF"].df["MUDIF"],
                                  read_fitres(str(tmp_path / "direct.M0DIF")).df["MUDIF"])


def test_a_cut_on_the_column_reaches_every_table(contaminated, tmp_path):
    """SALT2mu applies a CUTWIN to the bias-correction sample too, so that
    sample must carry the column; classified on both, the cut runs and
    removes the contaminants."""
    fitted = Fit("fitted", fitres=contaminated)
    toy = Perfect(prob_ia=0.9, prob_cc=0.1)
    data = Classify("classified", toy, fit=fitted)
    biascor = Classify("biascor_classified", toy, fit=Fit("biascor", fitres=_BIASCOR))
    ccprior = Classify("ccprior_classified", toy, fit=Fit("ccprior", fitres=_CCPRIOR))
    cut = BBC("cut", data=data, biascor=biascor, ccprior=ccprior,
              overrides={**_KEYS, f"CUTWIN {_COLUMN}": "0.5 1.0"})
    uncut = BBC("uncut", data=data, biascor=biascor, ccprior=ccprior, overrides=_KEYS)

    results = run(cut, uncut, outdir=tmp_path)

    def nfit(name):
        return int(results[name][_NOMINAL]["M0DIF"].df["NFIT"].sum())

    assert nfit("cut") < nfit("uncut")


def test_a_muopt_swaps_the_classifier(contaminated, tmp_path):
    fitted = Fit("fitted", fitres=contaminated)
    sharp = Classify("sharp", Perfect(prob_ia=0.99, prob_cc=0.01), fit=fitted)
    vague = Classify("vague", Perfect(prob_ia=0.6, prob_cc=0.4), fit=fitted)
    bbc = BBC("distances", data=sharp, biascor=Fit("biascor", fitres=_BIASCOR),
              ccprior=Fit("ccprior", fitres=_CCPRIOR), overrides=_KEYS,
              muopts={"vague": {"datafile": vague}})

    results = run(bbc, outdir=tmp_path)
    grid = results["distances"]

    assert set(bbc.inputs()) == {"data:sharp", "data:vague", "biascor:biascor", "ccprior:ccprior"}
    assert not np.array_equal(grid["vague"]["M0DIF"].df["MUDIF"],
                              grid[_NOMINAL]["M0DIF"].df["MUDIF"])


def test_training_hands_its_model_to_the_classifier(contaminated, tmp_path):
    """The model a training stage wrote is what the classifier uses."""
    fitted = Fit("fitted", fitres=contaminated)
    trained = Train("trained", Perfect(prob_ia=0.7, prob_cc=0.3), fit=fitted)
    classified = Classify("classified", Perfect(), fit=fitted, model=trained)

    results = run(classified, outdir=tmp_path)
    df = results["classified"][_NOMINAL].df

    assert results["trained"] == tmp_path / "trained" / "perfect.model"
    assert set(df[_COLUMN].unique()) == {0.7, 0.3}


def test_predictions_made_elsewhere_are_joined_as_classify_joins_them(contaminated,
                                                                     tmp_path):
    """Predictions already on disk, here a Pippin predictions.csv keyed by
    SNID, join onto every fitted table as a classifier's own do, an event
    with no prediction getting 0."""
    fitted = Fit("fitted", fitres=contaminated)
    df = read_fitres(str(contaminated)).df
    pd.DataFrame({"SNID": df["CID"].astype(str)[1:],
                  "PROB_PIPPIN": np.linspace(0, 1, len(df) - 1)}).to_csv(
        tmp_path / "predictions.csv", index=False, float_format="%0.4f")
    classified = Classify("classified", fit=fitted, predictions=tmp_path / "predictions.csv")

    joined = run(classified, outdir=tmp_path / "run")["classified"][_NOMINAL].df

    expected = np.concatenate([[0.0], np.round(np.linspace(0, 1, len(df) - 1), 4)])
    np.testing.assert_allclose(joined["PROB_PIPPIN"], expected)


def test_a_second_run_loads_the_predictions(contaminated, tmp_path):
    fitted = Fit("fitted", fitres=contaminated)
    classified = Classify("classified", Perfect(prob_ia=0.99999, prob_cc=1e-8), fit=fitted)

    once = run(classified, outdir=tmp_path)
    again = run(classified, outdir=tmp_path)
    manifest = yaml.safe_load((tmp_path / "MANIFEST.yml").read_text())

    assert manifest["stages"]["classified"]["loaded"] is True
    pd.testing.assert_frame_equal(again["classified"][_NOMINAL].df,
                                  once["classified"][_NOMINAL].df)


def test_supernnova_types_come_from_the_version_readme(tmp_path):
    """The types are read from the version's README, a merged one included,
    each with its photometric twin 100 above and ``Ia`` for the Type Ia
    lines; without the README or sntypes training refuses before it starts,
    since SuperNNova would otherwise take a built-in list."""
    version = tmp_path / "V"
    version.mkdir()
    snn = SuperNNova(filters=("DES-g",))
    with pytest.raises(SNANAError, match="sntypes"):
        snn.train(None, version, tmp_path / "trained")

    (version / "V.README").write_text(
        "DOCUMENTATION:\n"
        "  GENTYPE_TO_NAME:   # GENTYPE-integer    (non)Ia   transient-Name     FITS-prefix\n"
        "    1:  Ia       SALT3.K21                      SNIaMODEL00  \n"
        "    33:  nonIa    Ibc+Ic                         NONIaMODEL01  \n"
        "DOCUMENTATION_END:\n")
    assert snn._sntypes(version) == {"1": "Ia", "101": "Ia",
                                     "33": "Ibc+Ic", "133": "Ibc+Ic"}


def test_supernnova_refuses_bands_its_model_was_not_trained_on(tmp_path):
    """SuperNNova gives every event NaN, with no error, when the light
    curves' bands differ from its model's, so the classifier refuses first,
    naming both."""
    version = tmp_path / "V"
    version.mkdir()
    for name in ("HEAD.FITS", "PHOT.FITS"):
        shutil.copy(Path(_SNANA) / "gentype_map_k10" / f"{name}.gz",
                    version / f"V_{name}.gz")

    with pytest.raises(SNANAError, match=r"\['g', 'i', 'r', 'z'\].*\['DES-g', 'DES-i', 'DES-r', "
                                         r"'DES-z'\].*band_map"):
        SuperNNova(filters=_DES_BANDS).predict(None, version, _DES_MODEL)


@pytest.fixture(scope="module")
def supernnova(tmp_path_factory):
    """SuperNNova trained on a jax simulation of Type Ia and core-collapse
    events written into one version; returns the run's directory, the
    classifier and its training stage, so a test running in that directory
    loads them."""
    return _trained(tmp_path_factory.mktemp("supernnova"), "vanilla")


@pytest.fixture(scope="module")
def supernnova_variational(tmp_path_factory):
    """The same, SuperNNova's variational network, which samples its
    weights and reports a median and a spread."""
    return _trained(tmp_path_factory.mktemp("supernnova_variational"), "variational")


def _trained(outdir, variant):
    snn = SuperNNova(filters=("DES-g", "DES-i", "DES-r", "DES-z"), variant=variant,
                     nb_epoch=3)
    training = Sim("training", [_IA_INPUT, _CC_INPUT],
                   overrides=[{"NGENTOT_LC": 400, "RANSEED": 11},
                              {"NGENTOT_LC": 3000, "RANSEED": 12}])
    trained = Train("trained", snn, photometry=training)
    run(trained, outdir=outdir)
    return outdir, snn, trained


def _as_written(column):
    """A column as jax's writer takes it: astropy reads text as Unicode, which
    FITS stores as bytes."""
    column = np.asarray(column)
    return np.char.encode(column, "ascii") if column.dtype.kind == "U" else column


def _without_markers(head_path, phot_path):
    """A SNANA version's HEAD and PHOT columns without its marker rows, as
    jax's writer takes them."""
    head = fits.getdata(head_path, 1)
    phot = fits.getdata(phot_path, 1)
    first, last = head["PTROBS_MIN"] - 1, head["PTROBS_MAX"]
    rows = np.concatenate([np.arange(a, b) for a, b in zip(first, last)])
    counts = last - first
    ends = np.cumsum(counts)
    columns = {name: _as_written(head[name]) for name in head.columns.names}
    columns["PTROBS_MIN"] = (ends - counts + 1).astype(np.int32)
    columns["PTROBS_MAX"] = ends.astype(np.int32)
    return (columns, {name: _as_written(phot[name])[rows] for name in phot.columns.names},
            counts)


@needs_supernnova
def test_supernnova_reads_jax_files_as_it_reads_snana_files(supernnova, tmp_path):
    """SNANA's own events, rewritten by jax's FITS writer, get the same
    probabilities as SNANA's files: SuperNNova finds each light curve by the
    marker rows, so the layout must be SNANA's row for row."""
    _, snn, trained = supernnova
    model = trained.load({}, supernnova[0] / "trained")
    fixture = Path(_SNANA) / "gentype_map_k10"
    snana = tmp_path / "snana"
    snana.mkdir()
    for name in ("HEAD.FITS.gz", "PHOT.FITS.gz"):
        shutil.copy(fixture / name, snana / f"V_{name}")
    shutil.copy(fixture / "SIM.README", snana / "V.README")
    rewritten = tmp_path / "jax"
    rewritten.mkdir()
    head, phot, counts = _without_markers(snana / "V_HEAD.FITS.gz", snana / "V_PHOT.FITS.gz")
    primary = fits.getheader(snana / "V_HEAD.FITS.gz")
    keywords = {key: primary[key]
                for key in ("DATATYPE", "SURVEY", "SIMOPT_MWCOLORLAW", "SIM_MWRV")}
    write_head_fits(keywords, "V", str(rewritten), head)
    write_phot_fits(keywords, "V", str(rewritten), phot, counts)
    shutil.copy(fixture / "SIM.README", rewritten / "V.README")

    by_snana = snn.predict(None, snana, model).sort_index()
    by_jax = snn.predict(None, rewritten, model).sort_index()

    assert len(by_snana) == len(head["SNID"])
    pd.testing.assert_series_equal(by_jax, by_snana)


@needs_supernnova
@pytest.mark.parametrize("trained_snn", ["supernnova", "supernnova_variational"])
def test_supernnova_classifies_in_process_as_its_own_commands_do(trained_snn, request,
                                                                tmp_path):
    """Classifying in this process prepares the light curves itself; the
    probabilities equal those of SuperNNova's own make_data and validate_rnn
    commands, the path Pippin's jobs take, event by event. A variational
    network draws its weights afresh for each of its 50 samples, and the two
    paths draw them in a different order, so its median and spread agree with
    SuperNNova's aggregated ones within their sampling error."""
    outdir, snn, trained = request.getfixturevalue(trained_snn)
    model = trained.load({}, outdir / "trained")
    version = outdir / "training"
    dump = tmp_path / "dump"
    sntypes = snn._sntypes(version)
    snn._snn("make_data", [*snn._data_args(None, version, dump, sntypes), "--data_testing"],
             tmp_path)
    snn._snn("validate_rnn", [*snn._model_args(dump, sntypes), "--model_files", str(model)],
             tmp_path)
    if snn.variant == "vanilla":
        (written,) = (dump / "models").glob("*/PRED_*.pickle")
        columns = {"all_class0": snn.column}
    else:
        (written,) = (dump / "models").glob("*/PRED_*_aggregated.pickle")
        columns = {"all_class0_median": snn.column, "all_class0_std": f"{snn.column}_ERR"}
    reference = pd.read_pickle(written)
    by_commands = pd.DataFrame({name: reference[source].to_numpy()
                                for source, name in columns.items()},
                               index=reference["SNID"].astype(str).str.strip().to_numpy())

    in_process = pd.DataFrame(snn.predict(None, version, model)).sort_index()
    by_commands = by_commands.sort_index()
    if snn.variant == "vanilla":
        pd.testing.assert_frame_equal(in_process, by_commands)
        return
    pd.testing.assert_index_equal(in_process.index, by_commands.index)
    spread = by_commands[f"{snn.column}_ERR"]
    # five standard errors of the difference of two independent estimates
    # from 50 samples: sqrt(2) times 1.25 spread / sqrt(50) for a median, and
    # sqrt(2) times spread / sqrt(98) for a standard deviation
    assert ((in_process[snn.column] - by_commands[snn.column]).abs()
            < 5 * np.sqrt(2) * 1.25 * spread / np.sqrt(50)).all()
    assert ((in_process[f"{snn.column}_ERR"] - spread).abs()
            < 5 * np.sqrt(2) * spread / np.sqrt(98)).all()


@needs_supernnova
def test_a_model_trained_on_other_band_names_classifies_through_band_map(supernnova):
    """DES-SN5YR's own model, trained on bands named g, i, r, z, classifies
    jax's DES simulation, whose bands are DES-g and so on, once band_map
    renames them: every event gets a probability, the Type Ia ones near 1
    and the core-collapse ones near 0."""
    outdir, _, _ = supernnova
    version = outdir / "training"
    snn = SuperNNova(filters=_DES_BANDS,
                     band_map={band: band[-1] for band in _DES_BANDS})

    probability = snn.predict(None, version, _DES_MODEL)
    templates = pd.concat([pd.DataFrame({
        "SNID": np.char.strip(np.asarray(fits.getdata(head, 1)["SNID"], dtype=str)),
        "TEMPLATE": fits.getdata(head, 1)["SIM_TEMPLATE_INDEX"]})
        for head in sorted(version.glob("*HEAD.FITS*"))]).set_index("SNID")["TEMPLATE"]
    is_ia = templates.reindex(probability.index) == 0

    assert is_ia.any() and (~is_ia).any()
    assert probability[is_ia].median() > 0.9
    assert probability[~is_ia].median() < 0.1


@needs_supernnova
def test_supernnova_probabilities_reach_the_distance_fit(supernnova):
    """A second simulation, fitted and classified by the trained model: every
    fitted event gets a probability, the Type Ia ones higher on average, and
    the BEAMS fit runs on that column."""
    outdir, snn, trained = supernnova
    data = Sim("data", [_IA_INPUT, _CC_INPUT],
               overrides=[{"NGENTOT_LC": 3000, "RANSEED": 21},
                          {"NGENTOT_LC": 3000, "RANSEED": 22}])
    classified = Classify("classified", snn, fit=Fit("fit", _FIT_NML, datasets=[data]),
                          model=trained, photometry=data)
    bbc = BBC("distances", data=classified, biascor=Fit("biascor", fitres=_BIASCOR),
              ccprior=Fit("ccprior", fitres=_CCPRIOR),
              overrides={**_KEYS, "varname_pIa": snn.column})

    results = run(bbc, outdir=outdir)
    df = results["classified"][_NOMINAL].df
    predicted = pd.read_csv(outdir / "classified" / "predictions.csv", dtype={"CID": str})
    is_ia = df["SIM_TEMPLATE_INDEX"].astype(int) == 0

    assert set(df["CID"].astype(str)) <= set(predicted["CID"])
    assert df[snn.column].between(0, 1).all()
    assert df.loc[is_ia, snn.column].mean() > df.loc[~is_ia, snn.column].mean()
    assert results["distances"][_NOMINAL]["M0DIF"].df["NFIT"].sum() > 0



def test_external_predictions_preserve_padded_cids(tmp_path):
    fitted = tmp_path / "ids.FITRES"
    fitted.write_text("VARNAMES: CID IDSURVEY MU\n"
                      "SN: 00123 10 40\nSN: 123 10 41\nSN: 00456 10 42\n")
    predictions = tmp_path / "predictions.csv"
    predictions.write_text("CID,PROB_IA\n00456,0.1\n00123,0.95\n123,0.25\n")
    stage = Classify("classified", fit=Fit("fit", fitres=fitted),
                     predictions=predictions)
    outdir = tmp_path / "run"
    for _ in range(2):
        result = run(stage, outdir=outdir)["classified"][_NOMINAL].df
        assert result["CID"].tolist() == ["00123", "123", "00456"]
        np.testing.assert_allclose(result["PROB_IA"], [0.95, 0.25, 0.1])


def test_replacing_predictions_keeps_column_names_unique(tmp_path):
    """Replacing a probability and adding another preserves metadata on run and reload."""
    fitted = tmp_path / 'input.FITRES'
    fitted.write_text('VARNAMES: CID PROB_IA\nSN: 1 .2\nSN: 2 .3\n')
    predictions = tmp_path / 'predictions.csv'
    predictions.write_text('CID,PROB_IA,PROB_OTHER\n1,.9,.1\n')
    stage = Classify('classified', fit=Fit('fit', fitres=fitted), predictions=predictions)
    for _ in range(2):
        table = run(stage, outdir=tmp_path / 'run')['classified'][_NOMINAL]
        assert table.varnames == ['CID', 'PROB_IA', 'PROB_OTHER']
        assert table.varnames_by_file == [table.varnames]
        assert list(table.df.columns) == table.varnames
        np.testing.assert_allclose(table.df['PROB_IA'], [.9, 0.])
        np.testing.assert_allclose(table.df['PROB_OTHER'], [.1, 0.])


@pytest.mark.parametrize('head_gz,phot_gz', [(False, True), (True, False)])
def test_classifier_reads_listed_fits_and_text(tmp_path, head_gz, phot_gz):
    """Mixed compression and TEXT produce identical selected classifier epochs."""
    fits_dir = tmp_path / 'fits'
    text_dir = tmp_path / 'text'
    fits_dir.mkdir()
    text_dir.mkdir()
    head = Table({'SNID': ['1'], 'PTROBS_MIN': [1], 'PTROBS_MAX': [3],
                  'PEAKMJD': [60001.], 'REDSHIFT_FINAL': [0.2],
                  'REDSHIFT_FINAL_ERR': [0.001]})
    phot = Table({'MJD': [60000., 60001., 60002.], 'BAND': ['g', 'g', 'r'],
                  'FLUXCAL': [10., 11., 12.], 'FLUXCALERR': [1., 1., 1.],
                  'PHOTFLAG': [0, 8, 0]})
    primary = fits.PrimaryHDU()
    primary.header['DATATYPE'] = 'DATA'
    primary.header['SURVEY'] = 'DES'
    head_path = fits_dir / ('selected_HEAD.FITS' + ('.gz' if head_gz else ''))
    fits.HDUList([primary, fits.BinTableHDU(head)]).writeto(head_path)
    phot.write(fits_dir / ('selected_PHOT.FITS' + ('.gz' if phot_gz else '')))
    # An unlisted HEAD must not be opened or require a PHOT partner.
    (fits_dir / 'unlisted_HEAD.FITS').write_text('not part of this version')
    (fits_dir / 'version.LIST').write_text('selected_HEAD.FITS\n')
    (text_dir / 'version.LIST').write_text('event.DAT\n')
    (text_dir / 'event.DAT').write_text(
        'SURVEY: DES\nSNID: 1\nFILTERS: gr\nPEAKMJD: 60001\n'
        'REDSHIFT_FINAL: 0.2 +- 0.001\n'
        'VARLIST: MJD FLT FLUXCAL FLUXCALERR PHOTFLAG\n'
        'OBS: 60000 g 10 1 0\nOBS: 60001 g 11 1 8\nOBS: 60002 r 12 1 0\n')
    classifier = SuperNNova(filters=('g',), band_map={'g': 'DES-g'})
    actual = classifier._light_curves(fits_dir)
    pd.testing.assert_frame_equal(actual, classifier._light_curves(text_dir))
    assert actual['SNID'].tolist() == ['1']
    assert actual['FLT'].tolist() == ['DES-g']
    np.testing.assert_allclose(actual['FLUXCAL'], [10.])
    np.testing.assert_allclose(actual['HOSTGAL_SPECZ'], [0.2])
