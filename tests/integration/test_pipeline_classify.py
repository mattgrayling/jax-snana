"""Classification stages, on a contaminated table with truth.

The BEAMS fixture's contaminated table carries a fabricated probability and no
truth, so the tests build their own contaminated table from the fixture's
Type Ia data and its core-collapse prior sample. The fixtures have no TYPE
column, so the tests add one, as SNANA's fit writes it, from
SIM_TEMPLATE_INDEX. The Perfect classifier then has truth to read, and every
promise of the classification path is pinned: the column values, the join
onto every fitted table, the distance fit reading it, a cut on it reaching
the bias-correction sample, a MUOPT choosing another merged classifier,
training handing a model to classifying, and a second run loading it all.

SuperNNova runs where it is installed: a model trained on a jax simulation
classifies it, and DES-SN5YR's model classifies jax's DES light curves, its
probabilities carrying through to the distance fit.
"""

import importlib.util
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from astropy.io import fits
from astropy.table import Table

from jax_pipeline import BBC, Classify, Fit, Merge, Sim, Train, run
from jax_pipeline.classify import Perfect
from jax_pipeline.supernnova import SuperNNova
from jax_salt2mu import parse_bbc_input, run_bbc_fit
from jax_salt2mu.output import write_m0dif
from jax_salt2mu.tablevar import EVENT_TYPE_BIASCOR, EVENT_TYPE_CCPRIOR, EVENT_TYPE_DATA
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import read_fitres, write_varnames_table

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
         # off (prep_input, SALT2mu.c:22131), which would leave the classifier's
         # probabilities with nothing to weight
         "p13": 1}



#: the types the tables are given: Type Ia, and core collapse
_IA_TYPE, _CC_TYPE = 1, 20


def _typed(df: pd.DataFrame) -> pd.DataFrame:
    """The table with TYPE, the event type SNANA's fit writes, made from the
    simulation truth: Type Ia for SIM_TEMPLATE_INDEX 0, core collapse otherwise."""
    return df.assign(TYPE=np.where(df["SIM_TEMPLATE_INDEX"].astype(int) == 0, _IA_TYPE, _CC_TYPE))


def _write_fitres(df: pd.DataFrame, path: Path) -> Path:
    """The table written to ``path`` as a FITRES file."""
    columns = {name: df[name].to_numpy() for name in df}
    write_varnames_table(str(path), list(df), columns, fmt={})
    return path


@pytest.fixture(scope="module")
def contaminated(tmp_path_factory):
    """The Ia data with the core-collapse sample mixed in, CIDs kept apart,
    over the columns both carry, with their types."""
    ia = read_fitres(_DATA).df
    cc = read_fitres(_CCPRIOR).df
    cc["CID"] = (cc["CID"].astype(int) + 100_000).astype(str)
    mixed = pd.concat([ia[cc.columns], cc], ignore_index=True)
    return _write_fitres(_typed(mixed), tmp_path_factory.mktemp("contaminated") / "CONTAM.FITRES")


@pytest.fixture(scope="module")
def typed_corrections(tmp_path_factory):
    """The bias-correction and core-collapse samples with their types, so
    Perfect can classify them too."""
    directory = tmp_path_factory.mktemp("typed")
    biascor = _write_fitres(_typed(read_fitres(_BIASCOR).df), directory / "biascor.FITRES")
    ccprior = _write_fitres(_typed(read_fitres(_CCPRIOR).df), directory / "ccprior.FITRES")
    return {"biascor": biascor, "ccprior": ccprior}


def _with_truth_column(table, prob_ia, prob_cc):
    """The classifier's column added by hand, for a direct fit to compare to."""
    table.df[_COLUMN] = np.where(table.df["TYPE"].astype(int) == _IA_TYPE, prob_ia, prob_cc)
    return table


@pytest.mark.parametrize("prob_ia, prob_cc", [(0.9, 0.1),
                                           (0.99999, 1.234567890123456e-8)])
def test_classification_probabilities(contaminated, tmp_path, prob_ia, prob_cc):
    fitted = Fit("fitted", fitres=contaminated)
    classified = Classify("classified", Perfect(ia_types=(_IA_TYPE,), prob_ia=prob_ia, prob_cc=prob_cc),
                          fit=fitted)

    results = run(classified, outdir=tmp_path)
    df = results["classified"][_NOMINAL].df

    truth = df["TYPE"].astype(int) == _IA_TYPE
    assert truth.any() and (~truth).any()
    np.testing.assert_array_equal(df[_COLUMN], np.where(truth, prob_ia, prob_cc))
    assert (tmp_path / "classified" / "predictions.csv").is_file()
    assert _COLUMN in results["classified"][_NOMINAL].varnames_by_file[0]


def test_the_distance_fit_reads_the_joined_column(contaminated, tmp_path):
    """A BEAMS fit through the pipeline equals one on tables carrying the same
    column by hand, so the column reached the distance fit."""
    fitted = Fit("fitted", fitres=contaminated)
    classified = Classify("classified", Perfect(ia_types=(_IA_TYPE,), prob_ia=0.9, prob_cc=0.1),
                          fit=fitted)
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


def test_a_cut_on_the_column_reaches_every_table(contaminated, typed_corrections, tmp_path):
    """SALT2mu applies a CUTWIN to the bias-correction sample too, so that
    sample must carry the column; classified on both, the cut runs and
    removes the contaminants. A cut on varname_pIa cuts on that column
    (SALT2mu.c:8362-8365)."""
    fitted = Fit("fitted", fitres=contaminated)
    perfect = Perfect(ia_types=(_IA_TYPE,), prob_ia=0.9, prob_cc=0.1)
    data = Classify("classified", perfect, fit=fitted)
    biascor = Classify("biascor_classified", perfect, fit=Fit("biascor", fitres=typed_corrections["biascor"]))
    ccprior = Classify("ccprior_classified", perfect, fit=Fit("ccprior", fitres=typed_corrections["ccprior"]))
    cut = BBC("cut", data=data, biascor=biascor, ccprior=ccprior,
              overrides={**_KEYS, f"CUTWIN {_COLUMN}": "0.5 1.0"})
    aliased = BBC("aliased", data=data, biascor=biascor, ccprior=ccprior,
                  overrides={**_KEYS, "CUTWIN varname_pIa": "0.5 1.0"})
    uncut = BBC("uncut", data=data, biascor=biascor, ccprior=ccprior, overrides=_KEYS)

    results = run(cut, aliased, uncut, outdir=tmp_path)

    def nfit(name):
        return int(results[name][_NOMINAL]["M0DIF"].df["NFIT"].sum())

    assert nfit("cut") < nfit("uncut")
    assert nfit("aliased") == nfit("cut")


def test_a_muopt_chooses_another_merged_classifier(contaminated, tmp_path):
    """A merge of two classifications of one fit carries both columns, so a
    MUOPT chooses the other classifier with varname_pIa alone, and fits the
    distances that classifier gives on its own."""
    fitted = Fit("fitted", fitres=contaminated)
    sharp = Classify("sharp", Perfect(ia_types=(_IA_TYPE,), prob_ia=0.99, prob_cc=0.01,
                                      column="PROB_SHARP"), fit=fitted)
    vague = Classify("vague", Perfect(ia_types=(_IA_TYPE,), prob_ia=0.6, prob_cc=0.4,
                                      column="PROB_VAGUE"), fit=fitted)
    biascor = Fit("biascor", fitres=_BIASCOR)
    ccprior = Fit("ccprior", fitres=_CCPRIOR)
    merged = BBC("merged", data=Merge("merge", [sharp, vague]), biascor=biascor, ccprior=ccprior,
                 overrides={**_KEYS, "varname_pIa": "PROB_SHARP"},
                 muopts={"vague": {"varname_pIa": "PROB_VAGUE"}})
    alone = BBC("alone", data=vague, biascor=biascor, ccprior=ccprior,
                overrides={**_KEYS, "varname_pIa": "PROB_VAGUE"})

    results = run(merged, alone, outdir=tmp_path)

    np.testing.assert_array_equal(results["merged"]["vague"]["M0DIF"].df["MUDIF"],
                                  results["alone"][_NOMINAL]["M0DIF"].df["MUDIF"])
    assert not np.array_equal(results["merged"]["vague"]["M0DIF"].df["MUDIF"],
                              results["merged"][_NOMINAL]["M0DIF"].df["MUDIF"])


@dataclass(frozen=True)
class _ConstantClassifier:
    """Trains by saving one probability, which it then gives every event."""

    probability: float
    column: str = _COLUMN

    def train(self, fit, photometry, outdir) -> Path:
        model = Path(outdir) / "constant.model"
        model.write_text(str(self.probability))
        return model

    def predict(self, fit, photometry, model) -> pd.Series:
        return pd.Series(float(Path(model).read_text()),
                         index=fit.df["CID"].astype(str), name=self.column)


def test_training_hands_its_model_to_the_classifier(contaminated, tmp_path):
    """The model a training stage wrote is what the classifier uses."""
    fitted = Fit("fitted", fitres=contaminated)
    trained = Train("trained", _ConstantClassifier(0.7), fit=fitted)
    classified = Classify("classified", _ConstantClassifier(0.0), fit=fitted, model=trained)

    results = run(classified, outdir=tmp_path)
    df = results["classified"][_NOMINAL].df

    assert results["trained"] == tmp_path / "trained" / "constant.model"
    assert set(df[_COLUMN].unique()) == {0.7}


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
    classified = Classify("classified", Perfect(ia_types=(_IA_TYPE,), prob_ia=0.99999, prob_cc=1e-8),
                          fit=fitted)

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


@needs_supernnova
def test_supernnova_matches_bands_by_their_last_letter(tmp_path):
    """DES-SN5YR's model, trained on bands named g, i, r, z, classifies light
    curves labelled DES-g and so on, selected by LIST_FILTERS g, i, r, z,
    exactly as when band_map renames them."""
    version = tmp_path / "V"
    version.mkdir()
    # under the simulation's names, which its HEAD names its PHOT file by
    for name in ("HEAD.FITS", "PHOT.FITS"):
        shutil.copy(Path(_SNANA) / "gentype_map_k10" / f"{name}.gz",
                    version / f"JAX_TEST_GENTYPE_MAP_{name}.gz")
    (version / "V.LIST").write_text("JAX_TEST_GENTYPE_MAP_HEAD.FITS\n")
    renamed = SuperNNova(filters=_DES_BANDS, band_map={band: band[-1] for band in _DES_BANDS})
    by_band_map = renamed.predict(None, version, _DES_MODEL)

    by_letter = SuperNNova(filters=("g", "i", "r", "z")).predict(None, version, _DES_MODEL)

    pd.testing.assert_series_equal(by_letter, by_band_map)


@needs_supernnova
def test_supernnova_trains_a_model_it_predicts_with(tmp_path):
    """A model trained on a jax simulation of Type Ia and core-collapse
    events gives every event of that simulation a probability."""
    snn = SuperNNova(filters=_DES_BANDS, nb_epoch=3)
    training = Sim("training", [_IA_INPUT, _CC_INPUT],
                   overrides=[{"NGENTOT_LC": 400, "RANSEED": 11},
                              {"NGENTOT_LC": 3000, "RANSEED": 12}])
    trained = Train("trained", snn, photometry=training)
    run(trained, outdir=tmp_path)
    version = tmp_path / "training"
    snids = {snid.strip() for head in version.glob("*HEAD.FITS*")
             for snid in np.asarray(fits.getdata(head, 1)["SNID"], dtype=str)}

    probability = snn.predict(None, version, trained.load({}, tmp_path / "trained"))

    assert set(probability.index) == snids
    assert probability.between(0, 1).all()


@needs_supernnova
def test_supernnova_probabilities_reach_the_distance_fit(tmp_path):
    """A simulation, fitted and classified by DES-SN5YR's model: every fitted
    event gets a probability, the Type Ia ones near 1 and the core-collapse
    ones near 0, and the BEAMS fit runs on that column."""
    snn = SuperNNova(filters=_DES_BANDS)
    data = Sim("data", [_IA_INPUT, _CC_INPUT],
               overrides=[{"NGENTOT_LC": 3000, "RANSEED": 21},
                          {"NGENTOT_LC": 3000, "RANSEED": 22}])
    classified = Classify("classified", snn, fit=Fit("fit", _FIT_NML, datasets=[data]),
                          model=_DES_MODEL, photometry=data)
    bbc = BBC("distances", data=classified, biascor=Fit("biascor", fitres=_BIASCOR),
              ccprior=Fit("ccprior", fitres=_CCPRIOR),
              overrides={**_KEYS, "varname_pIa": snn.column})

    results = run(bbc, outdir=tmp_path)
    df = results["classified"][_NOMINAL].df
    predicted = pd.read_csv(tmp_path / "classified" / "predictions.csv", dtype={"CID": str})
    is_ia = df["SIM_TEMPLATE_INDEX"].astype(int) == 0

    assert set(df["CID"].astype(str)) <= set(predicted["CID"])
    assert df[snn.column].between(0, 1).all()
    assert df.loc[is_ia, snn.column].median() > 0.9
    assert df.loc[~is_ia, snn.column].median() < 0.1
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


def test_predictions_add_columns_the_fit_lacks(tmp_path):
    """Imported predictions add their columns, kept on run and reload; a
    column the fitted table already has is refused."""
    fitted = tmp_path / 'input.FITRES'
    fitted.write_text('VARNAMES: CID PROB_IA\nSN: 1 .2\nSN: 2 .3\n')
    predictions = tmp_path / 'predictions.csv'
    predictions.write_text('CID,PROB_OTHER\n1,.1\n')
    stage = Classify('classified', fit=Fit('fit', fitres=fitted), predictions=predictions)
    for _ in range(2):
        table = run(stage, outdir=tmp_path / 'run')['classified'][_NOMINAL]
        assert table.varnames == ['CID', 'PROB_IA', 'PROB_OTHER']
        assert table.varnames_by_file == [table.varnames]
        assert list(table.df.columns) == table.varnames
        np.testing.assert_allclose(table.df['PROB_IA'], [.2, .3])
        np.testing.assert_allclose(table.df['PROB_OTHER'], [.1, 0.])
    predictions.write_text('CID,PROB_IA\n1,.9\n')
    with pytest.raises(SNANAError, match=r"already has \['PROB_IA'\]"):
        run(stage, outdir=tmp_path / 'replaced')


def test_a_merge_joins_each_classifications_columns(tmp_path):
    """A merge gives one fit's tables the columns of each of its
    classifications, and refuses a classification of another fit."""
    fitres = tmp_path / 'fit.FITRES'
    fitres.write_text('VARNAMES: CID zHD\nSN: 10 0.1\nSN: 11 0.2\n')
    first_predictions = tmp_path / 'first.csv'
    first_predictions.write_text('CID,PROB_A\n10,0.7\n')
    second_predictions = tmp_path / 'second.csv'
    second_predictions.write_text('CID,PROB_B\n11,0.4\n')
    fit = Fit('fit', fitres=fitres)
    first = Classify('first', fit=fit, predictions=first_predictions)
    second = Classify('second', fit=fit, predictions=second_predictions)

    table = run(Merge('merged', [first, second]), outdir=tmp_path / 'run')['merged'][_NOMINAL]

    assert table.varnames == ['CID', 'zHD', 'PROB_A', 'PROB_B']
    assert list(table.df['PROB_A']) == [0.7, 0.0]
    assert list(table.df['PROB_B']) == [0.0, 0.4]
    other = Classify('other', fit=Fit('other_fit', fitres=fitres), predictions=first_predictions)
    with pytest.raises(SNANAError, match='other classifies other_fit, not fit'):
        Merge('merged', [first, other])


@pytest.mark.parametrize('head_gz,phot_gz', [(False, True), (True, False)])
def test_classifier_reads_listed_fits_and_text(tmp_path, head_gz, phot_gz):
    """Mixed compression and TEXT produce identical selected classifier epochs."""
    fits_dir = tmp_path / 'fits'
    text_dir = tmp_path / 'text'
    fits_dir.mkdir()
    text_dir.mkdir()
    head = Table({'SNID': ['1'], 'PTROBS_MIN': [1], 'PTROBS_MAX': [3],
                  'RA': [10.], 'DEC': [-5.], 'PEAKMJD': [60001.], 'REDSHIFT_FINAL': [0.2],
                  'REDSHIFT_FINAL_ERR': [0.001]})
    phot = Table({'MJD': [60000., 60001., 60002.], 'BAND': ['g', 'g', 'r'],
                  'FLUXCAL': [10., 11., 12.], 'FLUXCALERR': [1., 1., 1.],
                  'PHOTFLAG': [0, 8, 0]})
    primary = fits.PrimaryHDU()
    primary.header['DATATYPE'] = 'DATA'
    primary.header['SURVEY'] = 'DES'
    primary.header['PHOTFILE'] = 'selected_PHOT.FITS'
    head_path = fits_dir / ('selected_HEAD.FITS' + ('.gz' if head_gz else ''))
    fits.HDUList([primary, fits.BinTableHDU(head)]).writeto(head_path)
    phot.write(fits_dir / ('selected_PHOT.FITS' + ('.gz' if phot_gz else '')))
    # An unlisted HEAD must not be opened or require a PHOT partner.
    (fits_dir / 'unlisted_HEAD.FITS').write_text('not part of this version')
    (fits_dir / 'fits.LIST').write_text('selected_HEAD.FITS\n')
    (text_dir / 'text.LIST').write_text('event.DAT\n')
    (text_dir / 'event.DAT').write_text(
        'SURVEY: DES\nSNID: 1\nFILTERS: gr\nRA: 10\nDEC: -5\nPEAKMJD: 60001\n'
        'REDSHIFT_FINAL: 0.2 +- 0.001\nNOBS: 3\n'
        'VARLIST: MJD FLT FLUXCAL FLUXCALERR PHOTFLAG\n'
        'OBS: 60000 g 10 1 0\nOBS: 60001 g 11 1 8\nOBS: 60002 r 12 1 0\n')
    classifier = SuperNNova(filters=('g',), band_map={'g': 'DES-g'})
    actual = classifier._light_curves(fits_dir)
    pd.testing.assert_frame_equal(actual, classifier._light_curves(text_dir))
    assert actual['SNID'].tolist() == ['1']
    assert actual['FLT'].tolist() == ['DES-g']
    np.testing.assert_allclose(actual['FLUXCAL'], [10.])
    np.testing.assert_allclose(actual['HOSTGAL_SPECZ'], [0.2])
    # a filter selects bands by their last letter
    by_letter = SuperNNova(filters=('DES-g',))._light_curves(fits_dir)
    assert by_letter['FLT'].tolist() == ['g']
