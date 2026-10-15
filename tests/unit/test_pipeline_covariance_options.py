"""Numerical checks of covariance stage options on a small distance sample."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from jax_snlc_core.io.factorized_covariance import dense_factorized_covariance
from jax_pipeline import Covariance
from jax_snlc_core.io.fitres_reader import FitresTable


def test_external_covariance_calibrators_and_systematic_outputs(tmp_path):
    # Deliberately reverse the external covariance's event order.
    path = tmp_path / "external.npz"
    np.savez(path, CID=np.array(["B", "A"]), IDSURVEY=np.array([1, 1]),
             MU_COV=np.array([0.04, 0.01, 0.09]))
    frame = pd.DataFrame({"CID": ["A", "B"], "IDSURVEY": [1, 1], "FIELD": ["X", "X"],
                          "zHD": [0.1, 0.2], "zHEL": [0.1, 0.2],
                          "MUMODEL": [38., 40.], "MU": [38., 40.], "MUERR": [0.1, 0.2]})
    def cell(df):
        table = FitresTable(df, list(df), {"ISDATA_REAL": "1"})
        return {"FITRES": table, "M0DIF": table}
    inputs = {"distances": {"nominal": cell(frame),
                            "VPEC": cell(frame.assign(MU=frame.MU + [0.1, 0.2]))}}
    source = SimpleNamespace(name="distances", systematics=lambda: (("VPEC",), ()), grid_cells=lambda: [(0, 0), (1, 0)])
    stage = Covariance("cov", distances=source, mode="unbinned",
                       extra_covs=[("external", 0.5, path)], calibrators=["A"],
                       systematic_hd=True)
    out = tmp_path / "out"
    out.mkdir()
    actual = stage.run(inputs, out)
    external = 0.5**2 * np.array([[0.09, 0.01], [0.01, 0.04]])
    systematic = np.array([[0., 0.], [0., 0.2**2]])
    expected = external + systematic + np.diag([0.1**2, 0.2**2])
    np.testing.assert_allclose(_inverse(actual), np.linalg.inv(expected), rtol=1e-6)
    assert (out / "hubble_diagram.txt").is_file()
    assert (out / "FITOPT001_MUOPT000_hubble_diagram.txt").is_file()


@pytest.mark.parametrize("via", ["argument", "overrides", "input_file"])
def test_changing_external_matrix_reruns_covariance(tmp_path, via):
    from dataclasses import dataclass
    import yaml
    from jax_pipeline import run
    from jax_pipeline.stages import Stage

    @dataclass(eq=False)
    class Distances(Stage):
        name: str = "distances"

        def inputs(self):
            return {}

        def run(self, inputs, outdir):
            return self.load(inputs, outdir)

        def load(self, inputs, outdir):
            df = pd.DataFrame({"CID": ["A"], "IDSURVEY": [1], "FIELD": ["X"],
                               "zHD": [0.1], "zHEL": [0.1], "MUMODEL": [38.],
                               "MU": [38.], "MUERR": [0.1]})
            table = FitresTable(df, list(df), {"ISDATA_REAL": "1"})
            return {"nominal": {"FITRES": table, "M0DIF": table}}

    extra = tmp_path / "extra.npz"
    def write(value):
        np.savez(extra, CID=np.array(["A"]), IDSURVEY=np.array([1]), MU_COV=[value])
    write(0.04)
    keys = {"EXTRA_COVS": [f"external 1.0 {extra}"]}
    config = tmp_path / "cov.yml"
    config.write_text(yaml.safe_dump(keys))
    options = ({"extra_covs": [("external", 1., extra)]} if via == "argument" else
               {"overrides": keys} if via == "overrides" else {"input_file": config})
    stage = Covariance("cov", distances=Distances(), mode="unbinned", **options)
    out = tmp_path / "run"
    first = run(stage, outdir=out)
    run(stage, outdir=out)
    assert yaml.safe_load((out / "MANIFEST.yml").read_text())["stages"]["cov"]["loaded"]
    write(0.09)
    changed = run(stage, outdir=out)
    manifest = yaml.safe_load((out / "MANIFEST.yml").read_text())["stages"]
    assert manifest["distances"]["loaded"] and not manifest["cov"]["loaded"]
    np.testing.assert_allclose(first["cov"]["covariances"]["ALL"]["matrix"], [[20.]])
    np.testing.assert_allclose(changed["cov"]["covariances"]["ALL"]["matrix"], [[10.]])


@pytest.mark.parametrize("renorm,velocity,error", [
    (False, 0.06, None), (True, 0.06, None),
    (False, 0.2, "positive variance"), (False, float("nan"), "finite errors"),
])
def test_velocity_subtraction(tmp_path, renorm, velocity, error):
    from jax_snlc_core.utils.error_handling import SNANAError
    df = pd.DataFrame({"CID": ["A"], "IDSURVEY": [1], "FIELD": ["X"],
                       "zHD": [0.1], "zHEL": [0.1], "MUMODEL": [38.],
                       "MU": [38.], "MUERR": [0.1], "MUERR_VPEC": [velocity]})
    if renorm:
        df["MUERR_RENORM"] = 0.15
    table = FitresTable(df, list(df), {"ISDATA_REAL": "1"})
    source = SimpleNamespace(name="distances", systematics=lambda: ((), ()), grid_cells=lambda: [(0, 0)])
    stage = Covariance("cov", distances=source, mode="unbinned", subtract_vpec=True)
    inputs = {"distances": {"nominal": {"FITRES": table, "M0DIF": table}}}
    if error:
        with pytest.raises(SNANAError, match=error):
            stage.run(inputs, tmp_path)
    else:
        actual = stage.run(inputs, tmp_path)
        np.testing.assert_allclose(_inverse(actual), [[1 / ((0.1**2 - velocity**2) * (1.5**2 if renorm else 1))]])


@pytest.mark.parametrize('mode', ['unbinned', 'rebin'])
def test_renormalised_velocity_subtraction_matches_file_interface(tmp_path, mode):
    import yaml
    from jax_create_cov import create_covariance, load_config
    from jax_snlc_core.io.snana_table import write_varnames_table
    df = pd.DataFrame({"CID": ["A", "B"], "IDSURVEY": [1, 1], "FIELD": ["X", "X"],
                       "zHD": [0.1, 0.2], "zHEL": [0.1, 0.2], "MUMODEL": [38., 40.],
                       "MU": [38., 40.], "MUERR": [0.1, 0.2],
                       "IZBIN": [0, 1], "MUERR_RENORM": [0.15, 0.4], "MUERR_VPEC": [0.06, 0.1]})
    table = FitresTable(df, list(df), {"ISDATA_REAL": "1"})
    source = SimpleNamespace(name="distances", systematics=lambda: ((), ()),
                             grid_cells=lambda: [(0, 0)])
    stage = Covariance("cov", distances=source, mode=mode, rebin_bins={} if mode == "rebin" else None, subtract_vpec=True,
                       overrides={"COSMOLOGY": {"OMEGA_MATTER": .315, "OMEGA_LAMBDA": .685}})
    out = tmp_path / "stage"
    out.mkdir()
    actual = stage.run({"distances": {"nominal": {"FITRES": table, "M0DIF": table}}}, out)
    version = tmp_path / "export" / "distances"
    version.mkdir(parents=True)
    (version.parent / "SUBMIT.INFO").write_text(yaml.safe_dump({
        "FITOPT_OUT_LIST": [["FITOPT000", "DEFAULT", ""]],
        "MUOPT_OUT_LIST": [["MUOPT000", "DEFAULT", ""]]}))
    write_varnames_table(str(version / "FITOPT000_MUOPT000.FITRES"), list(df),
                         {name: df[name].to_numpy() for name in df}, fmt={},
                         header_comments=["ISDATA_REAL: 1"])
    config = tmp_path / "cov.yml"
    config.write_text(yaml.safe_dump({"INPUT_DIR": str(version.parent),
                                      "VERSION": "distances", "SUBTRACT_VPEC": True,
                                      "COSMOLOGY": {"OMEGA_MATTER": .315, "OMEGA_LAMBDA": .685},
                                      "mode": mode, "rebin_bins": {} if mode == "rebin" else None}))
    expected = create_covariance(load_config(config))
    np.testing.assert_allclose(_inverse(actual), expected.covtot_inv(), rtol=1e-6)
    variance = np.array([0.15**2 * (1 - 0.06**2 / 0.1**2), 0.4**2 * (1 - 0.1**2 / 0.2**2)])
    if mode == "unbinned":
        np.testing.assert_allclose(np.diag(_inverse(actual)), 1 / variance, rtol=1e-6)
    np.testing.assert_array_equal(df.MUERR_RENORM, [0.15, 0.4])


@pytest.mark.parametrize("flags", ["--write_mask_cov 7 --write_format_cov text",
                                  "--write_mask_cov 7 --label_cov_rows"])
def test_pippin_covariance_writer_flags_roundtrip(tmp_path, flags):
    from jax_pipeline.pippin import _cov_extra_options
    frame = pd.DataFrame({"CID": ["A", "B"], "IDSURVEY": [1, 1], "FIELD": ["X", "X"],
                          "zHD": [0.1, 0.2], "zHEL": [0.1, 0.2],
                          "MUMODEL": [38., 40.], "MU": [38., 40.], "MUERR": [0.1, 0.2]})
    table = FitresTable(frame, list(frame), {"ISDATA_REAL": "1"})
    source = SimpleNamespace(name="distances", systematics=lambda: ((), ()), grid_cells=lambda: [(0, 0)])
    stage = Covariance("cov", distances=source, mode="unbinned", **_cov_extra_options("C", flags))
    result = stage.run({"distances": {"nominal": {"FITRES": table, "M0DIF": table}}}, tmp_path)
    np.testing.assert_allclose(result["covariances"]["ALL"]["matrix"], np.diag([100., 25.]), rtol=1e-6)
    assert len(list(tmp_path.glob("*.gz"))) == 3


@pytest.mark.parametrize('settings,invalid_excluded', [
    ({'nosys': True}, False), ({'muopt': 1}, False),
    ({'muopt': 1}, True), ({'muopt': 9}, False)])
def test_covariance_grid_options_match_file_interface(tmp_path, settings, invalid_excluded):
    import yaml
    from jax_create_cov import create_covariance
    from jax_create_cov.config_parser import build_config
    from jax_pipeline.systematics import key, dirname
    from jax_snlc_core.io.snana_table import write_varnames_table
    from jax_snlc_core.utils import SNANAError

    fitopts, muopts = ('shift',), ('alternate',)
    cells = [(0, 0), (1, 0), (0, 1), (1, 1)]
    source = SimpleNamespace(name='distances', systematics=lambda: (fitopts, muopts),
                             grid_cells=lambda: cells)
    version = tmp_path / 'export' / 'distances'
    version.mkdir(parents=True)
    (version.parent / 'SUBMIT.INFO').write_text(yaml.safe_dump({
        'FITOPT_OUT_LIST': [['FITOPT000', '', 'DEFAULT'], ['FITOPT001', '', 'shift']],
        'MUOPT_OUT_LIST': [['MUOPT000', 'DEFAULT'], ['MUOPT001', 'alternate']]}))
    tables = {}
    for cell, offset in zip(cells, (0., .2, .3, .7)):
        df = pd.DataFrame(dict(ROW=['A', 'B'], zHD=[.1, .2], MU=np.array([38., 40.])+offset,
                               MUREF=[38., 40.], MUERR=[.1, .2]))
        if invalid_excluded and cell[1] != settings['muopt']:
            df = pd.DataFrame({'INVALID': [1]})
        table = FitresTable(df, list(df), {'ISDATA_REAL': '1'})
        tables[key(cell, fitopts, muopts)] = {'M0DIF': table, 'FITRES': table}
        write_varnames_table(str(version / (dirname(cell)+'.M0DIF')), list(df),
            {name: df[name].to_numpy() for name in df}, fmt={},
            header_comments=['ISDATA_REAL: 1'])
    stage = Covariance('cov', source, overrides=settings)
    cfg = build_config({'input_dir': str(version.parent), 'version': 'distances', **settings})
    out = tmp_path / 'stage'
    out.mkdir()
    if settings.get('muopt') == 9:
        with pytest.raises(SNANAError, match='MUOPT009'):
            create_covariance(cfg)
        with pytest.raises(SNANAError, match='MUOPT009'):
            stage.run({'distances': tables}, out)
        return
    expected = create_covariance(cfg)
    actual = stage.run({'distances': tables}, out)
    for label, entry in expected.covariances.items():
        cov = entry["matrix"]
        np.testing.assert_allclose(_inverse(actual, label), expected.covtot_inv(label), rtol=1e-6)
        np.testing.assert_allclose(np.diag(cov), .04 if settings.get('nosys') else .16, atol=1e-12)


def _inverse(held, label="ALL"):
    entry = held["covariances"][label]
    if entry["kind"] == "factorized":
        return np.linalg.inv(dense_factorized_covariance(entry))
    return entry["matrix"]
