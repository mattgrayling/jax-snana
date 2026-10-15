"""SALT3.K21 fitter tests.

``test_salt3_fitter_vs_snana`` is the parity gate: JAX recovered (t0, x1, c)
match SNANA's ``snlc_fit.exe`` per-SN within a fraction of the SNANA fit error.
Different optimisers (JAX = LM/L-BFGS, SNANA = MINUIT) so exact agreement isn't
expected, but well-converged fits should agree within their error bars. The
remaining tests cover load-time fudges, selection, epoch quality and priors.
"""

import gzip
import dataclasses
import functools
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from astropy.io import fits
from astropy.table import Table

from _fit_helpers import (build_fit_model, build_fitter, compare_lcplots,
                          fit_config, fit_fixture, fitter_photometry, for_fit,
                          load_batch, override, parse_fitres, peak_start,
                          regenerate_lcplot, run_fit)

from jax_snlc_core.io.fits_path import open_text, resolve_fits_path
from jax_snlc_core.io.kcor_reader import KcorReader
from jax_snlc_core.physics.cosmology import zhelio_to_zcmb
from jax_snlc_core.models import FittedModel, SNModel
from jax_snlc_core.models.fitted import flux_error_from_fracerr
from jax_snlc_core.photometry.noise import ZEROPOINT_FLUXCAL_SNANA_ORIG
from jax_snlc_fit.fit_spec import (error_surface_overrides, init_params,
                                   snana_parameter_aliases)
from jax_snlc_sim.simulation import Instrument
from jax_snlc_core.utils.error_handling import SNANAError
from jax_snlc_fit.batch import apply_snana_selection, select_events
from jax_snlc_fit.data_loader import (RUBIN_ZP_NJY, batch_from_columns,
                                      load_and_batch_snana_fits,
                                      load_and_batch_snana_text,
                                      model_sim_columns,
                                      read_epoch_ignore_file, read_magcor_file,
                                      read_rubin_hats)
from jax_snlc_fit import LightCurveFitter
from jax_snlc_fit.data_loader import resolve_mw_colour_law
from jax_snlc_fit.fitter import (fit_band_window, fit_photometry,
                                 fiterr_windowed, fitwin_trest_edges,
                                 maxfrac_floor_errs)
from jax_snlc_core.models import build_model
from jax_snlc_fit.photoz import coarse_grid_axes
from jax_snlc_fit.config import FitConfig, Model
from jax_snlc_fit.config_parser import FIELD_RECORD, RECORD_CLASSES
from jax_snlc_fit.run import (build_selection, drop_unfittable, ideal_fit,
                              sncid_list)


_LSST_BANDS = ["LSST-u", "LSST-g", "LSST-r", "LSST-i", "LSST-z", "LSST-Y"]


def _lsst_fitter(model, model_path, kcor_path, **flat):
    """A fitter of ``model`` in the six LSST bands, configured by the flat
    FitConfig fields ``flat`` beyond them, and its photometry."""
    config = override(FitConfig(model=Model(fitmodel_name=str(model_path))),
                      filtlist_fit='ugrizY', **flat)
    kcor = KcorReader(str(kcor_path))
    photometry = {**fit_photometry(config, model, kcor),
                  'zp_fluxcal': Instrument.zp_fluxcal}
    shape = {'epoch_lengths': [4], 'batch_size': 2,
             'band_window': fit_band_window(model, photometry), 'n_z': None}
    return LightCurveFitter._from_prepared(config, model, kcor, shape), photometry


def _stage_version(tmp_path, version, fixture_root):
    """A fixture's FITS laid out as a data version snlc_fit.py can read,
    under ``tmp_path`` as PRIVATE_DATA_PATH."""
    vdir = tmp_path / version
    vdir.mkdir()
    for tail in ("HEAD.FITS", "PHOT.FITS"):
        (vdir / f"{version}_{tail}").write_bytes(
            gzip.decompress((fixture_root / (tail + ".gz")).read_bytes()))
    (vdir / f"{version}.LIST").write_text(f"{version}_HEAD.FITS\n")


def _snlc_fit_rows(nml, tmp_path, repo_root, data_root, output, *overrides):
    """snlc_fit.py run with ``nml`` on the version staged under ``tmp_path``,
    with any ``KEY value`` overrides; its FITRES rows."""
    env = os.environ.copy()
    env["JAX_TEST_FIXTURE_ROOT"] = str(data_root)
    env["JAX_TEST_FIT_SIM_DIR"] = str(tmp_path)
    out = tmp_path / output
    r = subprocess.run(
        [sys.executable, str(repo_root / "snlc_fit.py"), str(nml), "BATCH_SIZE", "4",
         *overrides, "OUTPUT_NAME", str(out)],
        capture_output=True, text=True, env=env, timeout=1800)
    assert r.returncode == 0, r.stdout[-1200:] + r.stderr[-1200:]
    return parse_fitres(out)


@pytest.fixture(scope="module")
def salt3_model_path(data_root):
    return data_root / "models" / "SALT3.K21"


def _mk_config(**flat):
    """A FitConfig from flat field names, routed by the parser's own map."""
    groups = {}
    for name, value in flat.items():
        groups.setdefault(FIELD_RECORD[name], {})[name] = value
    return FitConfig(**{rec: RECORD_CLASSES[rec](**kw)
                        for rec, kw in groups.items()})


@pytest.fixture(scope="module")
def salt3_fit_results(snana_fixture):
    """Run the SALT3 fit on the salt3_fit_dense fixture once for both
    parity tests. Each fit is ~14s of JIT + optimisation."""
    fx = snana_fixture("salt3_fit_dense")
    results, batch = fit_fixture(fx.root, max_fit=80, batch_size=80)
    return fx, results, batch


def test_salt3_fitter_vs_snana(salt3_fit_results):
    """JAX recovers the same (t0, x1, c) as SNANA's snlc_fit.exe within fit error.

    Per-SN tolerance: ``|JAX_param - SNANA_param| < 3 * sqrt(SNANA_err^2 + JAX_err^2)``.
    A 3σ gate is loose enough to absorb honest optimiser disagreement and tight
    enough to catch JAX landing in a different fit minimum.
    """
    fx, results, _ = salt3_fit_results
    fitres_path = fx.root / "FITRES.TEXT"
    assert fitres_path.is_file(), (
        "salt3_fit_dense fixture missing FITRES.TEXT — regenerate with "
        "`python tests/fixtures/regenerate.py --only salt3_fit_dense`"
    )
    _assert_pull_parity(results, fitres_path)


def test_fitchi2_and_ndof_are_snanas(salt3_fit_results):
    """FITCHI2 is the chi2 MINUIT minimised, prior included (FITCHI2_STORE(1),
    snana.F90:28083), while NDOF is the fitted epochs less the floated
    parameters, the prior not counted (XNFITDOF(0), snlc_fit.F90:13132-13140,
    written at :13253); the prior's degree of freedom enters only FITPROB
    (NDOF_STORE(1), snana.F90:28108). Each round's t0 prior is centred on the
    last one's fit, so it is rarely above its 0.01 threshold at the end."""
    fx, results, _ = salt3_fit_results
    snana = parse_fitres(fx.root / "FITRES.TEXT")
    ndof = np.asarray(results.ndof)
    chi2 = np.asarray(results.chi_squared)
    compared = equal = 0
    for i, snid in enumerate(np.asarray(results.snids)):
        s = snana.get(int(snid))
        if s is None or s.get("ERRFLAG_FIT", 0) != 0:
            continue
        compared += 1
        assert abs(chi2[i] - s["FITCHI2"]) < 0.05 * max(s["FITCHI2"], 1.0), (
            f"CID {snid}: FITCHI2 {chi2[i]:.3f} vs SNANA {s['FITCHI2']:.3f}")
        equal += int(ndof[i]) == int(s["NDOF"])
    assert compared >= 30 and equal >= 0.95 * compared


def test_fitres_writes_a_string_cid(salt3_fit_results, tmp_path):
    """SNANA holds the CID as a string (SNLC_CCID) and its text writer emits it
    as one, so a data version whose events carry names rather than numbers
    writes them out as it read them. The OUTLIER table already did."""
    from jax_snlc_core.io.snana_table import write_varnames_table
    from jax_snlc_fit.sntable import build_fitres_table

    fx, results, batch = salt3_fit_results
    named = dataclasses.replace(results.take(np.arange(2)),
                                snids=np.array(["sn2011fe", "0123"]))
    z = np.asarray(batch.redshifts)[:2]
    model = build_fit_model(fit_config(fx.root / "fit.nml"))
    table, fmt, comments = build_fitres_table(
        named, z, np.full(2, 34.0), model=model)
    path = tmp_path / "FITRES.TEXT"
    write_varnames_table(path, table.varnames,
                         {c: table.df[c].values for c in table.varnames},
                         fmt, header_comments=comments)
    cids = [line.split()[1] for line in path.read_text().splitlines()
            if line.startswith("SN:")]
    assert cids == ["sn2011fe", "0123"]


def test_opt_setpkmjd_fmax_clump_vs_snana(snana_fixture, repo_root, data_root,
                                          tmp_path):
    """OPT_SETPKMJD = 20 picks an observed epoch, so PKMJDINI must equal
    SNANA's exactly. The estimate sees every band, u and Y included though
    FILTLIST_FIT drops them, and only the epochs CUTWIN_TREST lets SNANA
    read."""
    fx = snana_fixture("salt3_fit_fmaxclump")
    sim = snana_fixture("salt3_fit_dense")
    _stage_version(tmp_path, "JAX_TEST_SALT3_FIT_DENSE", sim.root)

    jax_rows = _snlc_fit_rows(fx.root / "fit.nml", tmp_path, repo_root,
                              data_root, "FMAXCLUMP.FITRES.TEXT")
    snana_rows = parse_fitres(fx.root / "FITRES.TEXT")
    shared = sorted(set(jax_rows) & set(snana_rows))
    assert len(shared) >= 95, f"too few cross-matched fits ({len(shared)})"

    differ = {c: jax_rows[c]["PKMJDINI"] - snana_rows[c]["PKMJDINI"]
              for c in shared
              if abs(jax_rows[c]["PKMJDINI"] - snana_rows[c]["PKMJDINI"]) > 1e-3}
    assert not differ, f"PKMJDINI differs from SNANA's: {differ}"


def test_selection_keys_change_the_row_set(snana_fixture, repo_root,
                                           data_root, tmp_path):
    """Keys whose whole effect is which events reach the FITRES and what the
    per-event columns say, driven through snlc_fit.py against the same
    fixture with and without each.

    SNCCID_IGNORE and SNCID_IGNORE_FILE drop the named events
    (snana.F90:11872), MXEVT_CUTS caps the events passing cuts (:2895),
    SNRMAX_FILTERS narrows the bands SNRMAX ranks (:17454), and
    LFIXPAR_ADJUST fixes every parameter so the fit takes no step (:1979).
    """
    fx = snana_fixture("salt3_fit_dense")
    _stage_version(tmp_path, "JAX_TEST_SALT3_FIT_DENSE", fx.root)

    def rows(name, *overrides):
        return _snlc_fit_rows(fx.root / "fit.nml", tmp_path, repo_root,
                              data_root, name, *overrides)

    base = rows("BASE.FITRES.TEXT")
    assert len(base) == 100
    dropped = sorted(base)[:3]

    ignored = rows("IGNORE.FITRES.TEXT", "SNCCID_IGNORE",
                   ",".join(str(c) for c in dropped))
    assert set(base) - set(ignored) == set(dropped)

    ignore_file = tmp_path / "ignore.list"
    ignore_file.write_text("\n".join(str(c) for c in dropped) + "\n")
    from_file = rows("IGNOREFILE.FITRES.TEXT", "SNCID_IGNORE_FILE",
                     str(ignore_file))
    assert set(from_file) == set(ignored)

    capped = rows("MXEVT.FITRES.TEXT", "MXEVT_CUTS", "20")
    assert len(capped) == 20

    # SNRMAX ranks the bands SNRMAX_FILTERS names, so narrowing it to one band
    # makes SNRMAX1 that band's own maximum and leaves SNRMAX2 at the sentinel
    one_band = rows("SNRFILT.FITRES.TEXT", "SNRMAX_FILTERS", "r")
    assert all(one_band[c]["SNRMAX1"] <= base[c]["SNRMAX1"] + 1e-6
               for c in one_band)
    assert all(one_band[c]["SNRMAX2"] == -9.0 for c in one_band)
    assert any(one_band[c]["SNRMAX1"] < base[c]["SNRMAX1"] - 1e-3
               for c in one_band)

    # LFIXPAR_ADJUST fixes every parameter, so each keeps its initial value:
    # x1 and c never leave the model's declared start, and FITINI_ADJUST
    # returns before it could move PKMJD off SNANA's own PKMJDINI (:9553)
    fixed = rows("LFIXPAR.FITRES.TEXT", "LFIXPAR_ADJUST", "T")
    assert {round(fixed[c]["x1"], 6) for c in fixed} == {0.001}
    assert {round(fixed[c]["c"], 6) for c in fixed} == {0.01}
    snana = parse_fitres(fx.root / "FITRES.TEXT")
    assert all(abs(fixed[c]["PKMJD"] - snana[c]["PKMJDINI"]) < 2e-3
               for c in fixed if c in snana)

    # PEAKMJD_SANITY windows the fitted peak against the initial estimate
    # (snlc_fit.F90:12813-12828); an impossible window keeps nobody
    assert rows("PKSANE.FITRES.TEXT", "PEAKMJD_SANITY", "1,2") == {}


def test_fit_matches_snana_without_a_covariance_matrix(snana_fixture):
    """Parity for OPT_COVAR_FLUX = OPT_COVAR_MWXTERR = 0, where SNANA takes
    USE_DIAG_ONLY and recomputes sigma at every call rather than freezing it
    (snlc_fit.F90:10572). No other fixture covers that configuration."""
    fx = snana_fixture("salt3_fit_diagcov")
    results, _ = fit_fixture(fx.root, max_fit=None)
    _assert_pull_parity(results, fx.root / "FITRES.TEXT")


def test_fudge_modelerr_scale_vs_snana(snana_fixture, salt3_fit_results):
    """FUDGE_MODELERR_SCALE on the covariance path (OPT_COVAR_FLUX = 1), where
    the model error enters through the off-diagonal blocks and not the
    diagonal (snlc_fit.F90:10762). The fixture is the dense sim refitted with
    the scale, so per event the ratio of scaled to unscaled errors and
    chi-square is the fudge's own effect, free of the minimiser difference
    between the two codes."""
    fx = snana_fixture("salt3_fit_modelerr")
    dense_data = snana_fixture("salt3_fit_dense")
    dense_fx, dense_results, _ = salt3_fit_results
    results, _ = fit_fixture(fx.root, data_dir=dense_data.root, max_fit=80)
    _assert_pull_parity(results, fx.root / "FITRES.TEXT")

    scaled = parse_fitres(fx.root / "FITRES.TEXT")
    plain = parse_fitres(dense_fx.root / "FITRES.TEXT")
    j_scaled = {int(s): i for i, s in enumerate(np.asarray(results.snids))}
    j_plain = {int(s): i for i, s in enumerate(np.asarray(dense_results.snids))}
    cids = [c for c in scaled
            if c in plain and c in j_scaled and c in j_plain
            and scaled[c]["ERRFLAG_FIT"] == 0 and plain[c]["ERRFLAG_FIT"] == 0]
    assert len(cids) >= 30, f"too few cross-matched fits ({len(cids)})"
    assert all(scaled[c]["SIM_x1"] == plain[c]["SIM_x1"] for c in cids), \
        "the two fixtures must share one sim"

    def ratios(jax_values, snana_col):
        s = np.array([scaled[c][snana_col] / plain[c][snana_col] for c in cids])
        j = np.array([float(jax_values(results)[j_scaled[c]])
                      / float(jax_values(dense_results)[j_plain[c]])
                      for c in cids])
        return s, j

    for jax_name, snana_col in (("x1", "x1ERR"), ("c", "cERR")):
        s, j = ratios(lambda r: np.asarray(r.param_errors[jax_name]), snana_col)
        assert np.median(s) > 1.2, f"{snana_col}: the fudge must move the errors"
        n_off = int((np.abs(j - s) > 0.15 * s).sum())
        assert n_off <= 2, (
            f"{snana_col} ratio scaled/plain: {n_off}/{len(cids)} events "
            f"off SNANA's by more than 15 percent")
    s, j = ratios(lambda r: np.asarray(r.chi_squared), "FITCHI2")
    assert np.median(s) < 0.8, "the fudge must lower the chi-square"
    assert np.median(j) == pytest.approx(np.median(s), rel=0.005), (
        f"FITCHI2 ratio scaled/plain: jax median {np.median(j):.3f} vs "
        f"SNANA {np.median(s):.3f}")
    n_off = int((np.abs(j / s - 1.0) > 0.05).sum())
    assert n_off <= 3, (
        f"FITCHI2 ratio scaled/plain: {n_off}/{len(cids)} events off SNANA's "
        "by more than 5 percent")


def test_salt3_fitter_data_fudges(snana_fixture):
    """Load-time FUDGE parity: JAX's post-fudge per-epoch FLUXCAL/FLUXCALERR
    equal SNANA's EXEC_FUDGE_FLUXCAL output (OUTLIER table) to float precision.
    Exercises FLUXCAL_OFFSET, FLUXERR_SCALE, MAG_ERROR, the PSF-coupled
    FLUXCAL_ERRPIX and (sim-only) SIM_FUDGE_MAG_ERROR in one combo — no fit, so
    this measures the fudge math."""
    fudge = snana_fixture("salt3_fit_fudge")
    dense_data = snana_fixture("salt3_fit_dense")
    allband = lambda v: {"": v}
    filter_map = {b: i for i, b in enumerate(_LSST_BANDS)}
    inv = {i: b.split("-")[-1] for b, i in filter_map.items()}
    batch = load_and_batch_snana_fits(
        str(dense_data.head_fits), str(dense_data.phot_fits),
        filter_map=filter_map,
        fit_bands=_LSST_BANDS,
        fudge_flux={"offset": allband(3.0), "err_scale": allband(1.4),
                    "mag_err": allband(0.02), "err_pix": allband(0.5),
                    "sim_mag_err": allband(0.03)},
    ).slice(0, 80)
    mjd = np.asarray(batch.reference_mjds)[:, None] + np.asarray(batch.mjd_offsets)
    fi, mask = np.asarray(batch.filter_indices), np.asarray(batch.observation_mask)
    fc, fe = np.asarray(batch.fluxcals), np.asarray(batch.fluxcalerrs)
    snids = np.asarray(batch.snids)
    jax = {(int(snids[i]), round(float(mjd[i, j]), 2), inv[int(fi[i, j])]):
           (fc[i, j], fe[i, j])
           for i in range(batch.n_sne) for j in range(batch.max_obs) if mask[i, j]}

    sn_flux = _outlier_map(fudge.root, "FLUXCAL_DATA")
    sn_err = _outlier_map(fudge.root, "FLUXCAL_ERR_DATA")
    dflux = [abs(jax[k][0] - v) for k, v in sn_flux.items() if k in jax]
    derr = [abs(jax[k][1] - v) for k, v in sn_err.items() if k in jax]
    assert len(dflux) >= 1500, f"too few matched epochs ({len(dflux)})"
    # float32 noise on a flux scales with the flux, so the bound set at
    # ZP_FLUXCAL 27.5 is carried to the file's zeropoint
    tol = 1e-3 * 10 ** (0.4 * (batch.zp_fluxcal - ZEROPOINT_FLUXCAL_SNANA_ORIG))
    assert max(dflux) < tol, f"flux max|Δ|={max(dflux):.2e}"
    assert max(derr) < tol, f"err max|Δ|={max(derr):.2e}"


def test_fudgeall_iter1_maxfrac_floor():
    """FUDGEALL_ITER1_MAXFRAC adds (maxfrac * the SN's peak flux over every
    band) in quadrature to the round-1 data error over unmasked epochs
    (snlc_fit.F90:10988); maxfrac=0 is identity."""
    flux = jnp.array([[10.0, 100.0, 50.0, 0.0]])       # peak over both bands: 100
    err = jnp.array([[2.0, 3.0, 4.0, 9.0]])
    mask = jnp.array([[True, True, True, False]])       # last epoch masked out
    out = np.asarray(maxfrac_floor_errs(0.1, flux, err, mask))[0]
    expect = np.sqrt(np.array([2, 3, 4, 9.0])**2 + (0.1 * np.array([100, 100, 100, 0.0]))**2)
    np.testing.assert_allclose(out, expect, rtol=1e-6)
    assert maxfrac_floor_errs(0.0, flux, err, mask) is err


def _outlier_map(root, column):
    """SNANA OUTLIER table -> ``{(CID, MJD, band): value}`` for one column."""
    names, out = None, {}
    with open_text(root / "OUTLIER.TEXT") as stream:
        fixture_text = stream.read()
    for line in fixture_text.splitlines():
        words = line.split()
        if line.startswith("VARNAMES:"):
            names = words[1:]
        elif line.startswith("SN:") and names:
            row = dict(zip(names, words[1:]))
            out[(int(row["CID"]), round(float(row["MJD"]), 2), row["BAND"])] = \
                float(row[column])
    return out


def test_salt3_fitter_magcor(snana_fixture, data_root):
    """MAGCOR_FILE parity: JAX's post-correction per-epoch FLUXCAL equals SNANA's
    (OUTLIER table) to float precision. Fixture is sim data, so the SNANA path is
    SIM_MAGCOR_FILE."""
    mc = snana_fixture("salt3_fit_magcor")
    dense_data = snana_fixture("salt3_fit_dense")
    fmap = {b: i for i, b in enumerate(_LSST_BANDS)}
    inv = {i: b.split("-")[-1] for b, i in fmap.items()}
    batch = load_and_batch_snana_fits(
        str(dense_data.head_fits), str(dense_data.phot_fits),
        filter_map=fmap,
        fit_bands=_LSST_BANDS,
        sim_magcor_map=read_magcor_file(
            str(data_root / "magcor" / "MAGCOR.DAT"),
            "JAX_TEST_SALT3_FIT_DENSE")).slice(0, 80)
    mjd = np.asarray(batch.reference_mjds)[:, None] + np.asarray(batch.mjd_offsets)
    fi, m = np.asarray(batch.filter_indices), np.asarray(batch.observation_mask)
    fc, snids = np.asarray(batch.fluxcals), np.asarray(batch.snids)
    jax = {(int(snids[i]), round(float(mjd[i, j]), 2), inv[int(fi[i, j])]): fc[i, j]
           for i in range(batch.n_sne) for j in range(batch.max_obs) if m[i, j]}
    diffs = [abs(jax[k] - v)
             for k, v in _outlier_map(mc.root, "FLUXCAL_DATA").items() if k in jax]
    assert len(diffs) >= 1500, f"too few matched epochs ({len(diffs)})"
    tol = 1e-3 * 10 ** (0.4 * (batch.zp_fluxcal - ZEROPOINT_FLUXCAL_SNANA_ORIG))
    assert max(diffs) < tol, f"flux max|Δ|={max(diffs):.2e}"


def test_magcor_file_version_and_size_checks(tmp_path, snana_fixture):
    """VERSION_ADD names the versions already carrying the correction, so
    adding to one or subtracting from any other is refused (ISTAT_MAGCOR_ADD,
    snana.F90:16750-16760), and a correction beyond 0.2 mag aborts when it
    is applied (snana.F90:16934)."""
    path = tmp_path / "MAGCOR.DAT"
    path.write_text("VERSION_ADD: DONE\nVARNAMES: ROW MAGCOR\n"
                    "ROW: 1-61324.615-i 0.30\n")
    with pytest.raises(SNANAError, match="already in"):
        read_magcor_file(str(path), "DONE")
    with pytest.raises(SNANAError, match="not in"):
        read_magcor_file("-" + str(path), "OTHER")
    assert read_magcor_file("-" + str(path), "DONE") == {("1", 61324.615, "i"): -0.30}

    # a newer SNANA writes a per-epoch wavelength shift beside MAGCOR
    wavecor = tmp_path / "WAVECOR.DAT"
    wavecor.write_text("VARNAMES: ROW MAGCOR WAVECOR\n"
                       "ROW: 1-61324.615-i 0.01 5.0\n")
    with pytest.raises(SNANAError, match="WAVECOR"):
        read_magcor_file(str(wavecor), "OTHER")

    dense_data = snana_fixture("salt3_fit_dense")
    with pytest.raises(SNANAError, match="exceeds 0.2 mag"):
        load_and_batch_snana_fits(
            str(dense_data.head_fits), str(dense_data.phot_fits),
            filter_map={b: i for i, b in enumerate(_LSST_BANDS)},
            fit_bands=_LSST_BANDS,
            sim_magcor_map=read_magcor_file(str(path), "OTHER"))


def test_epoch_ignore_file_drops_the_listed_epochs(snana_fixture,
                                                   salt3_fit_results):
    """The bundle's <version>.IGNORE, which the fitter's default
    EPOCH_IGNORE_FILE reads beside the data (snana.F90:5583, :13040-13049),
    drops its epochs before the fit. The fixture is the dense sim refitted by
    SNANA with the two brightest epochs of its first three events ignored:
    both codes lose exactly those degrees of freedom and still agree."""
    fx = snana_fixture("salt3_fit_ignore")
    dense_data = snana_fixture("salt3_fit_dense")
    dense_fx, dense_results, _ = salt3_fit_results
    ignored, _ = read_epoch_ignore_file(
        str(fx.root / "JAX_TEST_SALT3_FIT_DENSE.IGNORE"))
    assert len(ignored) == 6
    results, _ = fit_fixture(fx.root, data_dir=dense_data.root, max_fit=80,
                             epoch_ignore=ignored)
    _assert_pull_parity(results, fx.root / "FITRES.TEXT")

    snana_ignore = parse_fitres(fx.root / "FITRES.TEXT")
    snana_dense = parse_fitres(dense_fx.root / "FITRES.TEXT")
    ndof = dict(zip(np.asarray(results.snids).astype(int),
                    np.asarray(results.ndof)))
    ndof_dense = dict(zip(np.asarray(dense_results.snids).astype(int),
                          np.asarray(dense_results.ndof)))
    for cid in (1, 2, 3):
        assert snana_dense[cid]["NDOF"] - snana_ignore[cid]["NDOF"] == 2
        assert ndof_dense[cid] - ndof[cid] == 2


@pytest.mark.parametrize("fixture", ["salt3_fit_pinned", "salt3_fit_pinned_narrow"])
def test_first_iteration_model_error_vs_snana(snana_fixture, fixture):
    """SNANA's first-iteration SALT3 model error reads the variance surface at
    x1 = 0 (SALT2xx1, snlc_fit.F90:13792) but divides it by the training flux
    at the trial x1 (Finteg_errPar, genmag_SALT2.c:2905), and USRFUN fits a
    model outside 5 < mag < 40 as zero flux with zero error
    (snlc_fit.F90:5896-5908). The fixtures are the dense sim refitted in one
    iteration with x1 and c pinned, broad and blue (x1 = 1.5, c = -0.1) and
    narrow and red (x1 = -1.5, c = 0.1, where the extrapolated model of nearly
    every event is negative before explosion). At each event's fitted PKMJD
    and x0 the per-epoch model error must be the OUTLIER table's to a hundredth
    of the epoch's measurement error at the 99th percentile over the sample,
    and the chi2 over every fitted epoch its FITCHI2, which is where the
    zero-flux epochs count. The model flux differs from SNANA's by up to a
    tenth of the measurement error on a few per cent of epochs (the SED's
    phase interpolation), which the flux comparison allows; the error maps
    are read smoothly where SNANA reads them linearly, which rounds their
    corners and is what the percentile leaves room for."""
    fx = snana_fixture(fixture)
    dense = snana_fixture("salt3_fit_dense")
    config = fit_config(fx.root / "fit.nml")
    batch = load_batch(config, dense.root / "HEAD.FITS", dense.root / "PHOT.FITS")
    model = build_fit_model(config)
    fitter = build_fitter(config, model, batch)
    snana = parse_fitres(fx.root / "FITRES.TEXT")
    flux_ref = _outlier_map(fx.root, "FLUXCAL_FIT")
    err_ref = _outlier_map(fx.root, "FLUXCAL_ERR_FIT")
    aliases = snana_parameter_aliases(model)
    pinned = {aliases.get(k, k): v for k, v in config.model.inival.items()}
    assert set(pinned) == set(fitter.held_params) == {"x1", "c"}
    overrides = error_surface_overrides(model.get_fit_parameters(), first_round=True)
    band = {i: b.split("-")[-1] for i, b in enumerate(_LSST_BANDS)}
    names = list(fitter.param_names)
    photometry = fitter_photometry(fitter, batch)

    n_matched = 0
    error_pulls = []
    for i, sid in enumerate(np.asarray(batch.snids).astype(int)):
        # an event MINUIT could not fit (a crazy error, so SNANA skips it)
        # has no row to compare against
        if int(sid) not in snana:
            continue
        s = snana[int(sid)]
        z = float(batch.redshifts[i])
        ref = float(batch.reference_mjds[i])
        values = {"PKMJD": s["PKMJD"] - ref, "x0": s["x0"], **pinned}
        bw, bw_err = fitter.band_weights(
            photometry, z, jnp.ones(len(photometry['common_filter_wave_grid'])))
        mjd, fi = jnp.asarray(batch.mjd_offsets[i]), jnp.asarray(batch.filter_indices[i])
        mask = jnp.asarray(batch.observation_mask[i])
        params = jnp.array([values[n] for n in names])
        flux = fitter.model_flux(fitter.model, photometry, params, mjd, fi, mask, bw, z)
        err = fitter.model_flux_error(flux, fitter.model_fracerr(
            fitter.model, photometry, params, mjd, fi, mask, bw, bw_err, z,
            error_overrides=overrides), mask)
        flux, err = np.asarray(flux), np.asarray(err)
        fitted = np.asarray(mask) & np.asarray(fitter.restlambda_mask(
            photometry, fi[None, :], jnp.array([z])))[0]
        assert fitted.sum() == s["NDOF"] + len(pinned), sid

        fc, fce = np.asarray(batch.fluxcals[i]), np.asarray(batch.fluxcalerrs[i])
        for j in np.flatnonzero(fitted):
            key = (int(sid), round(float(mjd[j]) + ref, 2), band[int(fi[j])])
            if key not in flux_ref:
                continue
            n_matched += 1
            assert abs(flux[j] - flux_ref[key]) <= 0.1 * fce[j], (
                f"model flux {key}: jax {flux[j]} snana {flux_ref[key]} +- {fce[j]}")
            error_pulls.append(abs(err[j] - err_ref[key]) / fce[j])
        chi2 = np.sum(np.where(fitted, (fc - flux) ** 2 / (fce ** 2 + err ** 2), 0.0))
        chi2 += ((s["PKMJD"] - s["PKMJDINI"]) / model.prior_mjdsig) ** 2
        assert np.isclose(chi2, s["FITCHI2"], rtol=1e-2), (sid, chi2, s["FITCHI2"])
    assert n_matched >= 2400
    assert np.percentile(error_pulls, 99) <= 0.01, np.percentile(error_pulls, 99)


@pytest.mark.parametrize("fixture,nlfile,rtol", [
    ("salt3_fit_nonlin_nea", "NONLIN_NEA.DAT", 1e-5),
    # PER_PIX: JAX uses np.pi, SNANA a truncated 3.14159 -> <=0.06% pixel-grid
    # divergence.
    ("salt3_fit_nonlin_pix", "NONLIN_PIX.DAT", 1e-3),
])
def test_salt3_fitter_nonlinearity(snana_fixture, data_root, fixture, nlfile, rtol):
    """NONLINEARITY_FILE parity (PER_NEA and PER_PIX modes): JAX's
    post-non-linearity per-epoch FLUXCAL matches SNANA's (OUTLIER)."""
    from jax_snlc_fit.nonlinearity import read_nonlinearity_file
    nl = snana_fixture(fixture)
    dense_data = snana_fixture("salt3_fit_dense")
    fmap = {b: i for i, b in enumerate(_LSST_BANDS)}
    inv = {i: b.split("-")[-1] for b, i in fmap.items()}
    batch = load_and_batch_snana_fits(
        str(dense_data.head_fits), str(dense_data.phot_fits),
        filter_map=fmap,
        fit_bands=_LSST_BANDS,
        nonlin_model=read_nonlinearity_file(
            str(data_root / "nonlin" / nlfile))).slice(0, 80)
    mjd = np.asarray(batch.reference_mjds)[:, None] + np.asarray(batch.mjd_offsets)
    fi, m = np.asarray(batch.filter_indices), np.asarray(batch.observation_mask)
    fc, snids = np.asarray(batch.fluxcals), np.asarray(batch.snids)
    jax = {(int(snids[i]), round(float(mjd[i, j]), 2), inv[int(fi[i, j])]): fc[i, j]
           for i in range(batch.n_sne) for j in range(batch.max_obs) if m[i, j]}
    diffs = [abs(jax[k] - v) / max(abs(v), 1.0)
             for k, v in _outlier_map(nl.root, "FLUXCAL_DATA").items() if k in jax]
    assert len(diffs) >= 1500, f"too few matched epochs ({len(diffs)})"
    assert max(diffs) < rtol, f"flux max rel|Δ|={max(diffs):.2e}"


def test_salt3_fitter_fluxerrmodel(snana_fixture, data_root):
    """FLUXERRMODEL_FILE parity: JAX's post-model per-epoch FLUXCAL_ERR_DATA
    equals SNANA's (OUTLIER) to float precision. Fixture is sim data, so the
    SNANA path is SIM_FLUXERRMODEL_FILE."""
    from jax_snlc_core.photometry.fluxerr_model import read_fluxerr_model_file
    fe = snana_fixture("salt3_fit_fluxerr")
    dense_data = snana_fixture("salt3_fit_dense")
    model = read_fluxerr_model_file(
        str(data_root / "fluxerrmodel" / "JAX_TEST_FLAT.DAT"))
    fmap = {b: i for i, b in enumerate(_LSST_BANDS)}
    inv = {i: b.split("-")[-1] for b, i in fmap.items()}
    batch = load_and_batch_snana_fits(
        str(dense_data.head_fits), str(dense_data.phot_fits),
        filter_map=fmap,
        fit_bands=_LSST_BANDS, sim_fluxerr_model=model).slice(0, 80)
    mjd = np.asarray(batch.reference_mjds)[:, None] + np.asarray(batch.mjd_offsets)
    fi, m = np.asarray(batch.filter_indices), np.asarray(batch.observation_mask)
    fce, snids = np.asarray(batch.fluxcalerrs), np.asarray(batch.snids)
    jax = {(int(snids[i]), round(float(mjd[i, j]), 2), inv[int(fi[i, j])]): fce[i, j]
           for i in range(batch.n_sne) for j in range(batch.max_obs) if m[i, j]}
    diffs = [abs(jax[k] - v)
             for k, v in _outlier_map(fe.root, "FLUXCAL_ERR_DATA").items() if k in jax]
    assert len(diffs) >= 1500, f"too few matched epochs ({len(diffs)})"
    assert max(diffs) < 1e-3, f"err max|Δ|={max(diffs):.2e}"


def test_salt3_fitter_hostnoise(snana_fixture, data_root):
    """FUDGE_HOSTNOISE_FILE parity: JAX's post-model per-epoch FLUXCAL_ERR_DATA
    equals SNANA's (OUTLIER) to float precision. Runs on a HOSTLIB sim, which has
    populated host surface brightness."""
    from jax_snlc_fit.host_noise import read_hostnoise_file
    hn = snana_fixture("salt3_fit_hostnoise")
    maps = read_hostnoise_file(str(data_root / "hostnoise" / "HOSTNOISE.DAT"))
    fmap = {b: i for i, b in enumerate(_LSST_BANDS)}
    inv = {i: b.split("-")[-1] for b, i in fmap.items()}
    batch = load_and_batch_snana_fits(
        str(hn.root / "HEAD.FITS"), str(hn.root / "PHOT.FITS"),
        filter_map=fmap,
        fit_bands=_LSST_BANDS, hostnoise_maps=maps).slice(0, 300)
    mjd = np.asarray(batch.reference_mjds)[:, None] + np.asarray(batch.mjd_offsets)
    fi, m = np.asarray(batch.filter_indices), np.asarray(batch.observation_mask)
    fce, snids = np.asarray(batch.fluxcalerrs), np.asarray(batch.snids)
    jax = {(int(snids[i]), round(float(mjd[i, j]), 2), inv[int(fi[i, j])]): fce[i, j]
           for i in range(batch.n_sne) for j in range(batch.max_obs) if m[i, j]}
    diffs = [abs(jax[k] - v)
             for k, v in _outlier_map(hn.root, "FLUXCAL_ERR_DATA").items() if k in jax]
    assert len(diffs) >= 300, f"too few matched epochs ({len(diffs)})"
    assert max(diffs) < 1e-3, f"err max|Δ|={max(diffs):.2e}"


def test_salt3_fitter_errfudge_combined(snana_fixture, data_root):
    """Combined error-fudge parity: FLUXERRMODEL + FUDGE_FLUXERR_SCALE +
    FUDGE_MAG_ERROR in one run match SNANA's quadrature sum of ERR1-derived
    contributions (a multiplicative chain would scale the additive term too)."""
    from jax_snlc_core.photometry.fluxerr_model import read_fluxerr_model_file
    cb = snana_fixture("salt3_fit_errcombine")
    dense_data = snana_fixture("salt3_fit_dense")
    fe = read_fluxerr_model_file(
        str(data_root / "fluxerrmodel" / "JAX_TEST_FLAT.DAT"))
    ab = lambda v: {"": v}
    fmap = {b: i for i, b in enumerate(_LSST_BANDS)}
    inv = {i: b.split("-")[-1] for b, i in fmap.items()}
    batch = load_and_batch_snana_fits(
        str(dense_data.head_fits), str(dense_data.phot_fits),
        filter_map=fmap,
        fit_bands=_LSST_BANDS, sim_fluxerr_model=fe,
        fudge_flux={"err_scale": ab(1.3), "mag_err": ab(0.02)}).slice(0, 80)
    mjd = np.asarray(batch.reference_mjds)[:, None] + np.asarray(batch.mjd_offsets)
    fi, m = np.asarray(batch.filter_indices), np.asarray(batch.observation_mask)
    fce, snids = np.asarray(batch.fluxcalerrs), np.asarray(batch.snids)
    jax = {(int(snids[i]), round(float(mjd[i, j]), 2), inv[int(fi[i, j])]): fce[i, j]
           for i in range(batch.n_sne) for j in range(batch.max_obs) if m[i, j]}
    diffs = [abs(jax[k] - v)
             for k, v in _outlier_map(cb.root, "FLUXCAL_ERR_DATA").items() if k in jax]
    assert len(diffs) >= 1500, f"too few matched epochs ({len(diffs)})"
    assert max(diffs) < 1e-3, f"err max|Δ|={max(diffs):.2e}"


def test_fiterr_windowed_gating():
    """Windowed FUDGE_FITERR floor fires only where TREST, rest-lambda,
    passband and iteration windows all pass; off or past max-iter are no-ops."""
    cfg = _mk_config(fudge_fiterr_maxfrac=0.1, fudge_fiterr_maxiter=5,
                     fudge_fiterr_trest=(-5.0, 5.0, 999.0, 999.0),
                     fudge_fiterr_restlam=(3000.0, 6000.0, 90000.0, 90000.0))
    pbm = (True, False)
    fmw = jnp.array([4000.0, 8000.0])     # filter 0 in lambda window, filter 1 out
    t0, z = jnp.array([0.0]), jnp.array([0.0])
    mjd = jnp.array([[0.0, 3.0, 20.0]])   # trest 0(in), 3(in), 20(out)
    fi = jnp.array([[0, 1, 0]])           # filter 0(mask on), 1(mask off), 0
    out = np.asarray(fiterr_windowed(cfg, pbm, fmw, t0, z, mjd, fi, 1))[0]
    np.testing.assert_allclose(out, [0.1, 0.0, 0.0])
    assert fiterr_windowed(FitConfig(), pbm, fmw, t0, z, mjd, fi, 1) == 0.0
    assert fiterr_windowed(cfg, pbm, fmw, t0, z, mjd, fi, 6) == 0.0


def test_fudge_magerr_model_zero_removes_the_model_covariance(snana_fixture):
    """Under OPT_COVAR_FLUX the frozen covariance takes the model's own error,
    which FUDGE_MAGERR_MODEL = 0 removes (snlc_fit.F90:10691): each epoch is
    weighted by its measurement error alone and the colour-dispersion block
    is empty."""
    dense = snana_fixture("salt3_fit_dense")
    config = fit_config(dense.root / "fit.nml", fudge_magerr_model=0.0)
    batch = load_batch(config, dense.root / "HEAD.FITS",
                       dense.root / "PHOT.FITS", max_fit=3)
    fitter = build_fitter(config, build_fit_model(config), batch)
    attached = for_fit(fitter, batch)
    photometry, n = attached.photometry, batch.n_sne
    npar = len(fitter.param_names)
    bw = jax.vmap(functools.partial(fitter.band_weights, photometry))(
        batch.redshifts, attached.mw_att_obs)
    _, d_inv, V, _ = fitter._run_weights(
        fitter.model, photometry, None, jnp.zeros((n, npar)), jnp.ones((n, npar)),
        batch.mjd_offsets, batch.filter_indices, batch.observation_mask,
        batch.fluxcals, batch.fluxcalerrs, batch.redshifts, bw,
        fitter._colour_fraction(fitter.model, photometry, batch.filter_indices,
                                batch.redshifts, batch.observation_mask),
        attached.mw_att_obs, attached.mwxt_magerrs)
    mask = np.asarray(batch.observation_mask)
    np.testing.assert_allclose(np.asarray(d_inv)[mask],
                               1.0 / np.asarray(batch.fluxcalerrs)[mask] ** 2,
                               rtol=1e-12)
    assert not np.any(np.asarray(V)[:, :, :fitter.n_filters])


def test_fudge_mag_coverr_woodbury_block(snana_fixture):
    """FUDGE_MAG_COVERR adds a Woodbury V block, after the colour
    dispersion's, equal to r_b * F_i (data flux), so its V Vᵀ is SNANA's
    COV[i,j] = r_b^2 F_i F_j per band. OPT_COVAR_MWXTERR contributes a further
    single column, zero here because MWEBV_ERR is zero."""
    dense = snana_fixture("salt3_fit_dense")
    config = fit_config(dense.root / "fit.nml", fudge_mag_coverr={'': 0.05})
    batch = load_batch(config, dense.root / "HEAD.FITS",
                       dense.root / "PHOT.FITS", max_fit=3)
    r = 1.0 - 10.0 ** (-0.4 * 0.05)
    fitter = build_fitter(config, build_fit_model(config), batch)
    n, nf, npar = batch.n_sne, fitter.n_filters, len(fitter.param_names)
    attached = for_fit(fitter, batch)
    photometry = attached.photometry
    bw = jax.vmap(functools.partial(fitter.band_weights, photometry))(
        batch.redshifts, attached.mw_att_obs)
    _, _, V, _ = fitter._run_weights(
        fitter.model, photometry, None, jnp.zeros((n, npar)), jnp.ones((n, npar)),
        batch.mjd_offsets, batch.filter_indices, batch.observation_mask,
        batch.fluxcals, batch.fluxcalerrs, batch.redshifts, bw,
        fitter._colour_fraction(fitter.model, photometry, batch.filter_indices,
                                batch.redshifts, batch.observation_mask),
        attached.mw_att_obs, attached.mwxt_magerrs)
    V = np.asarray(V)
    # colour-disp block + coverr block + the single MW-extinction column
    assert V.shape[2] == 2 * nf + 1
    assert not np.any(V[:, :, -1])
    got = np.take_along_axis(
        V[:, :, nf:2 * nf],
        np.asarray(batch.filter_indices)[:, :, None], axis=2)[:, :, 0]
    exp = r * np.asarray(batch.fluxcals) * np.asarray(batch.observation_mask)
    np.testing.assert_allclose(got, exp, rtol=1e-6)


def _assert_pull_parity(results, fitres_path, min_compared=30):
    snana_rows = parse_fitres(fitres_path)

    snids = np.asarray(results.snids)
    pulls = {"PKMJD": [], "x1": [], "c": []}
    n_compared = 0
    for i, snid in enumerate(snids):
        cid = int(snid)
        if cid not in snana_rows:
            continue
        s = snana_rows[cid]
        # SNANA convergence flag (ERRFLAG_FIT == 0); skip failed SNANA fits.
        if s.get("ERRFLAG_FIT", 0) != 0:
            continue
        skip = False
        per_param_pulls = {}
        for jax_name, snana_val_col, snana_err_col in [
            ("PKMJD", "PKMJD", "PKMJDERR"),
            ("x1", "x1", "x1ERR"),
            ("c",  "c",  "cERR"),
        ]:
            jx = float(np.asarray(results.params[jax_name])[i])
            jx_err = float(np.asarray(results.param_errors[jax_name])[i])
            sv = s[snana_val_col]
            se = s[snana_err_col]
            sigma = float(np.sqrt(jx_err ** 2 + se ** 2))
            if not np.isfinite(sigma) or sigma <= 0 or not np.isfinite(jx):
                skip = True
                break
            per_param_pulls[jax_name] = (jx - sv) / sigma
        if skip:
            continue
        for k, v in per_param_pulls.items():
            pulls[k].append(v)
        n_compared += 1

    assert n_compared >= min_compared, f"too few cross-matched fits ({n_compared})"

    # 3σ per-SN gate. For Gaussian pulls the expected |pull|>3 fraction is
    # 0.27% — for ~70 SNe that's <1 outlier. Allow at most 2 to absorb edge
    # cases (param near a hard bound, optimiser landing in nearby minimum).
    for name, vals in pulls.items():
        vals_arr = np.asarray(vals)
        n_outliers = int((np.abs(vals_arr) > 3.0).sum())
        assert n_outliers <= 2, (
            f"{name}: {n_outliers}/{n_compared} SNe with |pull| > 3 (max 2)"
        )


_DES_BANDS = ["DES-g", "DES-r", "DES-i", "DES-z"]


def test_salt3_fitter_selection(snana_fixture):
    """Pre-fit selection parity on the salt3_fit_select fixture (DES bands,
    SURVEY.DEF-registered fields): PHOTFLAG_MSKREJ 1024 (saturated epochs),
    CUTWIN_REDSHIFT/SNRMAX windows, SNTYPE_LIST, SNFIELD_LIST (7 of 9
    fields). Gates: jax's accepted-SN count equals SNANA's post-cut count
    (75 of 237 candidates, from the snlc_fit log; SNANA then fits 63 after
    its fit-stage rejections), every SNANA-fitted CID was accepted by jax,
    and fitted params agree at the 3-sigma pull gate.
    """
    fx = snana_fixture("salt3_fit_select")
    config = fit_config(fx.root / "fit.nml")
    selection = build_selection(config, dict(config.selection.cutwin),
                                sncid_list(config))

    batch = select_events(load_batch(
        config, fx.root / "HEAD.FITS", fx.root / "PHOT.FITS",
        photflag_mskrej=config.selection.photflag_mskrej,
        snfield_list=config.selection.snfield_list,
    ), snana_selection=selection, ladder=[],
        max_process=config.selection.mxevt_process,
        max_fit=config.selection.mxevt_cuts)
    assert batch.n_sne == 75, f"jax accepted {batch.n_sne}, SNANA accepted 75"

    snana_rows = parse_fitres(fx.root / "FITRES.TEXT")
    jax_cids = set(np.asarray(batch.snids).astype(int).tolist())
    assert set(snana_rows).issubset(jax_cids), (
        sorted(set(snana_rows) - jax_cids))

    fitter = build_fitter(config, build_fit_model(config), batch)
    _assert_pull_parity(run_fit(fitter, batch), fx.root / "FITRES.TEXT",
                        min_compared=15)


@pytest.mark.parametrize("extra_windows,expected", [
    ({"nfilt_snrmax": (3.0, 99.0)}, 55),
    ({"snrmax2": (20.0, 1.0e8), "nfilt_snrmax2": (2.0, 99.0)}, 59),
])
def test_salt3_fitter_nfilt_snrmax(snana_fixture, extra_windows, expected):
    """CUTWIN_NFILT_SNRMAX(2) parity: count of bands whose SNRMAX clears
    CUTWIN_SNRMAX(2)'s lower edge (snana.F90:17458). SNANA snlc_fit CUTSTAT
    anchors on salt3_fit_select with CUTWIN_SNRMAX 5,1e8."""
    fx = snana_fixture("salt3_fit_select")
    config = fit_config(fx.root / "fit.nml")
    batch = load_batch(
        config, fx.root / "HEAD.FITS", fx.root / "PHOT.FITS",
        photflag_mskrej=config.selection.photflag_mskrej,
        snfield_list=config.selection.snfield_list)
    with fits.open(resolve_fits_path(fx.root / "HEAD.FITS")) as h:
        hd = Table(h[1].data)
    s2t = {int(s.strip()): int(t) for s, t in
           zip(np.array(hd['SNID'], dtype='U'), np.array(hd['SNTYPE']))}
    st = np.array([s2t[c] for c in np.asarray(batch.snids).astype(int)])
    keep, _ = apply_snana_selection(
        batch, head_sntypes=st,
        windows={**config.selection.cutwin, **extra_windows},
        sntype_list=list(config.selection.sntype_list), ladder=[])
    assert int(keep.sum()) == expected, (
        f"jax accepted {int(keep.sum())}, SNANA {expected}")


def test_fit_uses_the_heliocentric_redshift(snana_fixture, tmp_path):
    """SNANA fits with zHELIO and falls back to zCMB only where it is
    undefined (SNLC_REDSHIFT, snana.F90:4256 and :4292) -- the wavelength
    stretch and time dilation follow the observed redshift. zCMB itself comes
    from REDSHIFT_FINAL, or REDSHIFT_CMB where that is undefined (:4277).
    """
    fx = snana_fixture("salt3_fit_select")
    fmap = {b: i for i, b in enumerate(_DES_BANDS)}

    def load(head):
        return load_and_batch_snana_fits(
            str(head), str(fx.root / "PHOT.FITS"), fit_bands=_DES_BANDS,
            filter_map=fmap)

    with fits.open(resolve_fits_path(fx.root / "HEAD.FITS")) as h:
        phdr = h[0].header.copy()
        columns = [(c.name, c.format, np.array(h[1].data[c.name]))
                   for c in h[1].data.columns]
    by_name = dict((n, a) for n, _, a in columns)
    zhel, zfin = by_name['REDSHIFT_HELIO'], by_name['REDSHIFT_FINAL']
    assert not np.allclose(zhel, zfin), "fixture cannot distinguish the frames"

    np.testing.assert_allclose(
        np.asarray(load(fx.root / "HEAD.FITS").redshifts), zhel, rtol=1e-12)

    def rewritten(**edits):
        """The fixture HEAD with these columns set to a constant, adding any
        the file does not already carry."""
        out = [fits.Column(name=n, format=f,
                           array=(np.full(len(a), edits[n], dtype=float)
                                  if n in edits else a))
               for n, f, a in columns]
        out += [fits.Column(name=n, format='1E',
                            array=np.full(len(zhel), v, dtype=float))
                for n, v in edits.items() if n not in by_name]
        p = tmp_path / f"{'_'.join(edits)}_HEAD.FITS"
        fits.HDUList([fits.PrimaryHDU(header=phdr),
                      fits.BinTableHDU.from_columns(out)]).writeto(p)
        return p

    # zHELIO undefined -> zCMB transformed into the heliocentric frame, not
    # zCMB itself, so pushing the answer back to the CMB frame must return
    # REDSHIFT_FINAL. The dipole is ~1e-3 in z, far above the tolerance.
    b = load(rewritten(REDSHIFT_HELIO=-9.0))
    back = zhelio_to_zcmb(np.asarray(b.redshifts),
                              np.asarray(b.ras), np.asarray(b.decs))
    np.testing.assert_allclose(np.asarray(back), zfin, rtol=1e-6)
    assert np.abs(np.asarray(b.redshifts) - zfin).max() > 1e-4

    # ... and REDSHIFT_CMB when REDSHIFT_FINAL is undefined too
    b = load(rewritten(REDSHIFT_HELIO=-9.0, REDSHIFT_FINAL=-9.0,
                       REDSHIFT_CMB=0.25))
    back = zhelio_to_zcmb(np.asarray(b.redshifts),
                              np.asarray(b.ras), np.asarray(b.decs))
    np.testing.assert_allclose(np.asarray(back), 0.25, rtol=1e-6)


def test_fitwin_trest_filter_is_per_band():
    """SNANA fills every filter's Trest window with the global FITWIN_TREST and
    then overwrites the bands FITWIN_TREST_FILTER names, testing only the
    per-filter array (SETCUTS_FITWIN_TREST, snlc_fit.F90:2774-2812, applied at
    :4486). With no band named, every filter keeps the global window, so the
    scalar path and the per-band path agree.
    """
    from jax_snlc_fit.config_parser import build, read_keys

    def cfg(extra=""):
        p = Path(tempfile.mkdtemp()) / "fit.nml"
        p.write_text("&SNLCINP\n  FILTLIST_FIT = 'griz'\n"
                     "  FITWIN_TREST = -15.0, 45.0\n" + extra + "&END\n")
        return build(read_keys(p))

    filters = ["DES-g", "DES-r", "DES-i", "DES-z"]
    # 'gr' takes -10..50; i and z keep the global -15..45
    c = cfg("  FITWIN_TREST_FILTER = 'gr -10.0 50.0'\n")
    assert c.fit.fitwin_trest_filter == {'g': (-10.0, 50.0),
                                               'r': (-10.0, 50.0)}
    lo, hi = c.fit.fitwin_trest_bounds(filters)
    np.testing.assert_allclose(lo, [-10.0, -10.0, -15.0, -15.0])
    np.testing.assert_allclose(hi, [50.0, 50.0, 45.0, 45.0])

    # every epoch gets its own band's edges
    bands = jnp.array([[0, 1, 2, 3]])
    elo, ehi = fitwin_trest_edges(c, bands, filters)
    np.testing.assert_allclose(np.asarray(elo), [[-10.0, -10.0, -15.0, -15.0]])
    np.testing.assert_allclose(np.asarray(ehi), [[50.0, 50.0, 45.0, 45.0]])

    # unset: the scalar window stands, so behaviour is unchanged
    assert cfg().fit.fitwin_trest_bounds(filters) is None
    assert fitwin_trest_edges(cfg(), bands, filters) == (-15.0, 45.0)

    # SNANA aborts on a string that is not (filters, min, max) triplets
    with pytest.raises(SNANAError, match="triplets"):
        cfg("  FITWIN_TREST_FILTER = 'gr -10.0'\n")


def test_fit_prep_epoch_guard(snana_fixture):
    """SNANA skips an event at fit prep, not at the cuts, when it has nothing
    to fit (NFITDATA <= 0, snlc_fit.F90:10610) or fewer epochs than
    CUTWIN_NEPOCH's lower bound (:10630). CUTWIN_NEPOCH is never initialised in
    SNANA, so unset means a floor of one epoch and only the empty events go.

    The events go to the fit driver's rejected set rather than the cut set:
    they passed the SNLC cuts, so the SNANA table reports them CUTFLAG 1. Only
    the empty case can abort, and only under ABORT_ON_NOEPOCHS.
    """
    fx = snana_fixture("salt3_fit_select")
    batch = load_and_batch_snana_fits(
        str(fx.root / "HEAD.FITS"), str(fx.root / "PHOT.FITS"),
        fit_bands=_DES_BANDS, photflag_mskrej=(1024,),
        snfield_list=["C3", "E1", "S1", "S2", "C1", "C2", "X1"],
        filter_map={b: i for i, b in enumerate(_DES_BANDS)})
    nobs = np.asarray(batch.n_obs_per_sn)
    assert (nobs == 0).any(), "fixture no longer has empty events to guard"

    snids = np.asarray(batch.snids)
    opts = dict(abort_on_noepochs=False, quantile_prior=False,
                abort_on_missing_qzphot=True)
    kept, empty, thin, badq = drop_unfittable(batch, None, **opts)
    assert kept.n_sne == int((nobs >= 1).sum())
    assert set(empty.tolist()) == set(snids[nobs < 1].tolist())
    assert thin.size == 0, "with no CUTWIN_NEPOCH nothing is merely thin"
    assert badq.size == 0, "the quantile drop is off unless asked for"

    # an explicit CUTWIN_NEPOCH raises the floor to its lower bound, and the
    # events it adds carry ERRFLAG_FITPREP_NEPOCH rather than _NOEPOCHS
    kept, empty, thin, badq = drop_unfittable(batch, (5.0, 99.0), **opts)
    assert kept.n_sne == int((nobs >= 5).sum())
    assert set(empty.tolist()) == set(snids[nobs < 1].tolist())
    assert set(thin.tolist()) == set(snids[(nobs >= 1) & (nobs < 5)].tolist())
    assert thin.size, "fixture no longer has thin events to separate"

    # ABORT_ON_NOEPOCHS is SNANA's default and aborts on the empty events
    with pytest.raises(SNANAError, match="Nothing to fit"):
        drop_unfittable(batch, None, **{**opts, 'abort_on_noepochs': True})


def test_a_host_zpdf_the_spline_cannot_take_drops_only_that_event(snana_fixture):
    """init_spline requires every quantile pair to increase and returns an
    error rather than aborting (sntools_spline.c:198-210); FITPAR_PREP turns
    that into ERRFLAG_FITPREP_QUANTILES and fits nothing
    (snlc_fit.F90:8540-8543). A hostless event, NBIN 0, aborts the run
    unless ABORT_ON_MISSING_QZPHOT = F, which skips it too (FITINI_PHOTOZ,
    :8468-8478).
    """
    import dataclasses
    fx = snana_fixture("salt3_fit_dense")
    batch = load_and_batch_snana_fits(
        str(fx.root / "HEAD.FITS"), str(fx.root / "PHOT.FITS"),
        fit_bands=_LSST_BANDS,
        filter_map={b: i for i, b in enumerate(_LSST_BANDS)}).slice(0, 6)
    q = np.tile(np.linspace(0.1, 0.5, 11), (batch.n_sne, 1))
    nbin = np.full(batch.n_sne, 11)
    q[2] = -9.0                       # hostless
    nbin[2] = 0
    q[4, 5:] = q[4, 4]                # a flat run, still not increasing
    batch = dataclasses.replace(batch, zpdf_quantiles=q, zpdf_nbin=nbin)
    snids = np.asarray(batch.snids)

    prior = dict(abort_on_noepochs=False, quantile_prior=True,
                 abort_on_missing_qzphot=True)
    with pytest.raises(SNANAError, match=f"Missing zPDF quantiles .* CID={snids[2]}"):
        drop_unfittable(batch, None, **prior)
    kept, empty, thin, badq = drop_unfittable(
        batch, None, **{**prior, 'abort_on_missing_qzphot': False})
    assert set(badq.tolist()) == {snids[2], snids[4]}
    assert kept.n_sne == batch.n_sne - 2
    assert empty.size == 0 and thin.size == 0

    # the same batch is untouched when the quantile prior is not in use
    kept, _, _, badq = drop_unfittable(
        batch, None, **{**prior, 'quantile_prior': False})
    assert badq.size == 0 and kept.n_sne == batch.n_sne

    # asking for the prior with no quantiles at all aborts, as SNANA does
    # (snlc_fit.F90:8531-8535)
    with pytest.raises(SNANAError, match=f"Missing zPDF quantiles .* CID={snids[0]}"):
        drop_unfittable(dataclasses.replace(batch, zpdf_quantiles=None),
                        None, **prior)


def test_loading_leaves_the_caller_filter_map_alone(snana_fixture):
    """Bands the caller did not number get numbered into the loader's own copy.
    Growing the caller's dict instead would silently change how their next load
    numbers its bands, and the resulting batches would disagree on what an
    index means."""
    fx = snana_fixture("salt3_fit_select")
    fmap = {"DES-g": 0}
    batch = load_and_batch_snana_fits(
        str(fx.root / "HEAD.FITS"), str(fx.root / "PHOT.FITS"),
        fit_bands=_DES_BANDS, filter_map=fmap)

    assert fmap == {"DES-g": 0}, f"caller's mapping was mutated to {fmap}"
    # the numbering it did reaches the caller on the batch instead
    assert set(batch.filter_names.values()) >= set(_DES_BANDS)


def test_jobsplit_partitions_the_loaded_events(snana_fixture):
    """JOBSPLIT strides the raw file index (DO isn = IJOB, NSN_VERS, NJOBTOT,
    snana.F90:2893), so events outside a job's share are never read at all.
    The jobs therefore partition the event set exactly -- including the events
    the cuts reject, which are the ones the SNANA table still reports.
    """
    fx = snana_fixture("salt3_fit_select")

    def load(jobsplit):
        return load_and_batch_snana_fits(
            str(fx.root / "HEAD.FITS"), str(fx.root / "PHOT.FITS"),
            fit_bands=_DES_BANDS, jobsplit=jobsplit,
            filter_map={b: i for i, b in enumerate(_DES_BANDS)})

    everything = set(np.asarray(load((1, 1)).snids).astype(int).tolist())
    shares = [set(np.asarray(load((i, 3)).snids).astype(int).tolist())
              for i in (1, 2, 3)]
    assert set().union(*shares) == everything
    assert sum(len(s) for s in shares) == len(everything), "jobs overlap"


def _parse_snana_table(path: Path) -> dict[int, dict[str, str]]:
    """Return ``{cid: {col: raw string}}`` for an SNANA-table TEXT file."""
    names, rows = None, {}
    with open_text(path) as stream:
        fixture_text = stream.read()
    for line in fixture_text.splitlines():
        if line.startswith("VARNAMES:"):
            names = line.split()[1:]
        elif line.startswith("SN:") and names:
            row = dict(zip(names, line.split()[1:]))
            rows[int(row["CID"])] = row
    return rows


def test_snana_table_redshift_columns_vs_snana(snana_fixture, repo_root,
                                               data_root, tmp_path):
    """SNANA-table redshift columns are built from corrected values.

    Every HEAD event gets a row, cut or not, and SNANA fills that row after
    SET_REDSHIFTS has run (SNRECON -> snana.F90:17162, before SET_CUTMASK;
    rejected events fill at :9308). So REDSHIFT_FINAL_SHIFT and
    VPEC_ERR_OVERRIDE reach zHEL/zCMB/zHD/zHDERR/VPECERR for every row. The
    fixture shifts z by +0.01 and overrides VPEC_ERR 300 -> 150, both of which
    a table built from an uncorrected second load would miss.
    """
    import os
    import subprocess
    import sys
    fx = snana_fixture("salt3_fit_zshift")
    vdir = tmp_path / "JAX_TEST_SALT3_FIT_ZSHIFT"
    vdir.mkdir()
    for kind in ("HEAD", "PHOT"):
        (vdir / f"JAX_TEST_SALT3_FIT_ZSHIFT_{kind}.FITS").write_bytes(
            gzip.decompress((fx.root / f"{kind}.FITS.gz").read_bytes()))
    (vdir / "JAX_TEST_SALT3_FIT_ZSHIFT.LIST").write_text(
        "JAX_TEST_SALT3_FIT_ZSHIFT_HEAD.FITS\n")

    env = os.environ.copy()
    env["JAX_TEST_FIXTURE_ROOT"] = str(data_root)
    env["JAX_TEST_FIT_SIM_DIR"] = str(tmp_path)
    out = tmp_path / "OUT.FITRES.TEXT"
    result = subprocess.run(
        [sys.executable, str(repo_root / "snlc_fit.py"),
         str(fx.root / "fit.nml"), "BATCH_SIZE", "4", "OUTPUT_NAME", str(out)],
        capture_output=True, text=True, env=env, timeout=900)
    assert result.returncode == 0, result.stderr[-2000:]

    jax_rows = _parse_snana_table(
        tmp_path / "JAX_TEST_SALT3_FIT_ZSHIFT.SNANA.TEXT")
    ref_rows = _parse_snana_table(fx.root / "SNANA.TEXT")
    assert set(ref_rows) == set(jax_rows), (
        f"jax-only {sorted(set(jax_rows) - set(ref_rows))[:10]}, "
        f"snana-only {sorted(set(ref_rows) - set(jax_rows))[:10]}")

    # These columns are written with five decimals, so a one-in-last-place
    # disagreement is the printed representation rather than the value. A wrong
    # value shows up as most rows moving: deriving zCMB from an already-shifted
    # zHEL, rather than shifting it as SET_zSHIFT does, moved 276 of 300.
    def last_place(value: str) -> int:
        """The printed value in integer units of the fifth decimal."""
        return round(float(value) * 1e5)

    for col in ("zHEL", "zHELERR", "zCMB", "zCMBERR", "zHD", "zHDERR",
                "VPEC", "VPECERR"):
        deltas = {cid: abs(last_place(jax_rows[cid][col]) - last_place(ref[col]))
                  for cid, ref in ref_rows.items()}
        worst_cid = max(deltas, key=deltas.get)
        moved = sum(1 for d in deltas.values() if d)
        assert deltas[worst_cid] <= 1, (
            f"{col}: differs by {deltas[worst_cid]} printed units at "
            f"CID {worst_cid}")
        assert moved <= 0.02 * len(deltas), (
            f"{col}: {moved}/{len(deltas)} rows differ, so this is a value "
            f"error rather than rounding")


def test_salt3_fitter_nfilt_snrmax_fit_stage(snana_fixture, repo_root, data_root,
                                             tmp_path):
    """Post-fit NFILT_SNRMAX recount over fitted epochs (snlc_fit.F90:13038).
    FITWIN_TREST 20-45 pushes SNRMAX epochs out of the fit window, so the
    fit-stage count drops below the stage-1 count. SNANA anchors (snlc_fit
    run on this fixture with the modified nml): CID 244 passes stage-1
    nf>=2 and fits OK but fails the recount; the pinned CIDs survive to
    the FITRES.
    """
    import os
    import subprocess
    import sys
    fx = snana_fixture("salt3_fit_select")
    vdir = tmp_path / "JAX_TEST_SALT3_FIT_SELECT"
    vdir.mkdir()
    (vdir / "JAX_TEST_SALT3_FIT_SELECT_HEAD.FITS").write_bytes(
        gzip.decompress((fx.root / "HEAD.FITS.gz").read_bytes()))
    (vdir / "JAX_TEST_SALT3_FIT_SELECT_PHOT.FITS").write_bytes(
        gzip.decompress((fx.root / "PHOT.FITS.gz").read_bytes()))
    (vdir / "JAX_TEST_SALT3_FIT_SELECT.LIST").write_text(
        "JAX_TEST_SALT3_FIT_SELECT_HEAD.FITS\n")
    nml = (fx.root / "fit.nml").read_text()
    nml = nml.replace("FITWIN_TREST        = -15.0, 45.0",
                      "FITWIN_TREST        = 20.0, 45.0")
    # This fixture keeps post-peak epochs only, which leaves t0 poorly
    # determined, and that is exactly where SNANA's errors are under-stated
    # (MINIMIZE never calls HESSE). Every
    # cut calibrated against a fitted error therefore selects different events
    # in the two codes here, for reasons unrelated to the SNRMAX recount under
    # test: CHECK_CRAZYFITERR, and SNANA's two hard-wired FITWIN error windows
    # (snlc_fit.F90:6835), which are widened rather than left at their default.
    nml = nml.replace("CUTWIN_SNRMAX      = 5.0, 1.0E8",
                      "CUTWIN_SNRMAX      = 5.0, 1.0E8\n"
                      "  CUTWIN_NFILT_SNRMAX = 2, 99\n"
                      "  CHECK_CRAZYFITERR = F\n"
                      "  FITWIN_COV_OFFDIAG = -1.0E8, 1.0E8\n"
                      "  FITWIN_MB_ERR = 0.0, 1.0E8")
    (tmp_path / "fit_bite.nml").write_text(nml)

    env = os.environ.copy()
    env["JAX_TEST_FIXTURE_ROOT"] = str(data_root)
    env["JAX_TEST_FIT_SIM_DIR"] = str(tmp_path)
    out = tmp_path / "OUT.FITRES.TEXT"
    result = subprocess.run(
        [sys.executable, str(repo_root / "snlc_fit.py"),
         str(tmp_path / "fit_bite.nml"), "BATCH_SIZE", "4",
         "MAX_FIT", "15", "OUTPUT_NAME", str(out)],
        capture_output=True, text=True, env=env, timeout=600)
    assert result.returncode == 0, result.stderr[-800:]

    # the first 15 post-cut SNe span CIDs 26-287, including 244
    jax_cids = set(parse_fitres(out))
    snana_kept = {61, 84, 116, 148, 166, 183, 237, 245}
    assert snana_kept <= jax_cids, sorted(snana_kept - jax_cids)
    assert 244 not in jax_cids


def test_nfilt_snrmax_counts_the_fudged_snr(snana_fixture, repo_root,
                                            data_root, tmp_path):
    """A band counts toward NFILT_SNRMAX by its SNR over the fudged error.

    SNANA accumulates SNRMAX_FIT from SNR_FIT, the flux over the measured
    error with the MAXFRAC fudge in quadrature, so a deweighted passband
    stops counting (snlc_fit.F90:4215, :13277). Counting the raw SNR instead
    keeps bands SNANA has already deweighted out of the sample.

    With FUDGEALL_MAXFRAC 0.15 and CUTWIN_NFILT_SNRMAX 3, snlc_fit.exe on
    this fixture keeps 16 events (the fudge scales the light curve's global
    peak since v12_02d, so every band but the brightest is deweighted hard)
    and drops CID 1488; without the fudge it keeps all 54. The two
    error-calibrated cuts are switched off because this fixture keeps
    post-peak epochs, where SNANA's own fit errors are unreliable (Bug 7),
    which would otherwise move events for reasons unrelated to the recount.
    """
    import os
    import subprocess
    import sys
    fx = snana_fixture("salt3_fit_select")
    vdir = tmp_path / "JAX_TEST_SALT3_FIT_SELECT"
    vdir.mkdir()
    for kind in ("HEAD", "PHOT"):
        (vdir / f"JAX_TEST_SALT3_FIT_SELECT_{kind}.FITS").write_bytes(
            gzip.decompress((fx.root / f"{kind}.FITS.gz").read_bytes()))
    (vdir / "JAX_TEST_SALT3_FIT_SELECT.LIST").write_text(
        "JAX_TEST_SALT3_FIT_SELECT_HEAD.FITS\n")
    nml = (fx.root / "fit.nml").read_text()
    nml = nml.replace("CUTWIN_SNRMAX      = 5.0, 1.0E8",
                      "CUTWIN_SNRMAX      = 5.0, 1.0E8\n"
                      "  CUTWIN_NFILT_SNRMAX = 3, 99")
    # the three fit-stage keys belong to &FITINP, which snlc_fit.exe enforces
    nml = nml.replace("PRIOR_MJDSIG        = 20.0",
                      "PRIOR_MJDSIG        = 20.0\n"
                      "  FUDGEALL_MAXFRAC    = 0.15\n"
                      "  CHECK_CRAZYFITERR = F\n"
                      "  FITWIN_COV_OFFDIAG = -1.0E8, 1.0E8\n"
                      "  FITWIN_MB_ERR = 0.0, 1.0E8")
    (tmp_path / "fudged.nml").write_text(nml)

    env = os.environ.copy()
    env["JAX_TEST_FIXTURE_ROOT"] = str(data_root)
    env["JAX_TEST_FIT_SIM_DIR"] = str(tmp_path)
    out = tmp_path / "OUT.FITRES.TEXT"
    result = subprocess.run(
        [sys.executable, str(repo_root / "snlc_fit.py"),
         str(tmp_path / "fudged.nml"), "BATCH_SIZE", "4", "OUTPUT_NAME", str(out)],
        capture_output=True, text=True, env=env, timeout=600)
    assert result.returncode == 0, result.stderr[-800:]

    jax_cids = set(parse_fitres(out))
    assert 1488 not in jax_cids, (
        "CID 1488 survives, so the recount is reading the unfudged SNR")
    assert len(jax_cids) == 16, f"jax kept {len(jax_cids)}, SNANA 16"


def test_salt3_fitter_real_data_load(snana_fixture, tmp_path):
    """Loader real-data path: a DATATYPE=DATA file with the SIM_ columns
    stripped must load identically to the sim (same CIDs/redshift/flux, peak
    from PEAKMJD) but with sim_truth=None (is_sim=False; redshift comes from
    REDSHIFT_FINAL). Derived at runtime from salt3_fit_select — the fit itself
    is on identical photometry and already covered by test_salt3_fitter_selection."""
    fx = snana_fixture("salt3_fit_select")
    for kind in ("HEAD", "PHOT"):
        with fits.open(resolve_fits_path(fx.root / f"{kind}.FITS")) as h:
            keep = [c for c in h[1].data.columns if not c.name.startswith("SIM")]
            phdr = h[0].header.copy()
            phdr["DATATYPE"] = "DATA"
            fits.HDUList([
                fits.PrimaryHDU(header=phdr),
                fits.BinTableHDU.from_columns(fits.ColDefs(keep),
                                              header=h[1].header),
            ]).writeto(str(tmp_path / f"{kind}.FITS"))

    kw = dict(fit_bands=_DES_BANDS, photflag_mskrej=(1024,),
              snfield_list=["C3", "E1", "S1", "S2", "C1", "C2", "X1"])
    fmap = {b: i for i, b in enumerate(_DES_BANDS)}
    real = load_and_batch_snana_fits(
        str(tmp_path / "HEAD.FITS"), str(tmp_path / "PHOT.FITS"),
        filter_map=dict(fmap), **kw)
    sim = load_and_batch_snana_fits(
        str(fx.root / "HEAD.FITS"), str(fx.root / "PHOT.FITS"),
        filter_map=dict(fmap), **kw)

    assert real.sim_truth is None and sim.sim_truth is not None
    assert real.peakmjd_offsets is not None
    assert np.array_equal(np.sort(np.asarray(real.snids).astype(int)),
                          np.sort(np.asarray(sim.snids).astype(int)))
    assert np.allclose(np.sort(np.asarray(real.redshifts)),
                       np.sort(np.asarray(sim.redshifts)))
    assert np.allclose(float(np.asarray(real.fluxcals).sum()),
                       float(np.asarray(sim.fluxcals).sum()))


def test_salt3_fitter_text_loader(snana_fixture):
    """TEXT-format loader (VERSION.LIST + per-SN .DAT) must produce the same
    LightCurveBatch as the FITS loader for the same SNe — flux to TEXT's ~6
    sig-fig precision. salt3_fit_text is 6 SNe from the salt3_fit_select
    realisation written with FORMAT_MASK=2."""
    text_fx = snana_fixture("salt3_fit_text")
    fits_fx = snana_fixture("salt3_fit_select")
    fmap = {b: i for i, b in enumerate(_DES_BANDS)}
    kw = dict(fit_bands=_DES_BANDS, model_sim_columns=('SIM_SALT2x1',))
    text = load_and_batch_snana_text(
        str(text_fx.root), filter_map=dict(fmap), **kw)
    fits_b = load_and_batch_snana_fits(
        str(fits_fx.root / "HEAD.FITS"), str(fits_fx.root / "PHOT.FITS"),
        filter_map=dict(fmap), **kw)

    tcid = np.asarray(text.snids).astype(int)
    fcid = np.asarray(fits_b.snids).astype(int)
    fsub = fits_b.filter_by_mask(np.isin(fcid, tcid))
    assert set(tcid.tolist()) == set(np.asarray(fsub.snids).astype(int).tolist())

    ti = np.argsort(tcid)
    fi = np.argsort(np.asarray(fsub.snids).astype(int))
    assert np.allclose(np.asarray(text.redshifts)[ti],
                       np.asarray(fsub.redshifts)[fi], atol=1e-4)
    # SIM_ truth parsed from the .DAT header matches the FITS sim_truth
    assert np.allclose(np.asarray(text.sim_truth["SIM_SALT2x1"])[ti],
                       np.asarray(fsub.sim_truth["SIM_SALT2x1"])[fi], atol=1e-3)
    assert np.array_equal(np.asarray(text.observation_mask).sum(1)[ti],
                          np.asarray(fsub.observation_mask).sum(1)[fi])

    def sorted_flux(b, order):
        fl = np.asarray(b.fluxcals)
        ok = np.asarray(b.observation_mask)
        return [np.sort(fl[i][ok[i]]) for i in order]
    for a, c in zip(sorted_flux(text, ti), sorted_flux(fsub, fi)):
        assert np.allclose(a, c, rtol=1e-4, atol=1e-2)

    # scalar HOSTGAL* header keys land in host_data on both loaders
    assert np.allclose(
        np.asarray(text.host_data['HOSTGAL_NMATCH'], dtype=float)[ti],
        np.asarray(fsub.host_data['HOSTGAL_NMATCH'], dtype=float)[fi])


def test_rubin_hats_loader(snana_fixture, tmp_path):
    """A Rubin HATS catalogue loads as the same batch as the FITS it was
    written from: each event a DiaObject whose forced light curve is a struct
    of lists, fluxes in nJy, bands as bare letters, which name the kcor's
    filters by their last character as INTFILTER does (sntools.c:10330)."""
    fx = snana_fixture("salt3_fit_select")
    # FITS is big-endian, which pyarrow does not take
    head, phot = ({name: np.asarray(table[name]).astype(
        table[name].dtype.newbyteorder('=')) for name in table.names}
        for table in (fits.getdata(fx.head_fits, 1),
                      fits.getdata(fx.phot_fits, 1)))
    # the fixture's FLUXCAL is on its own zero point, nJy on Rubin's
    to_njy = 10.0 ** (0.4 * (RUBIN_ZP_NJY - fits.getheader(
        fx.head_fits)['ZP_FLUXCAL']))
    spans = [np.arange(lo - 1, hi) for lo, hi in
             zip(head['PTROBS_MIN'], head['PTROBS_MAX'])]

    def lists(values):
        return pa.array([np.asarray(values)[rows] for rows in spans])

    no_flag = [np.zeros(len(rows), dtype=bool) for rows in spans]
    lightcurve = pa.StructArray.from_arrays(
        [lists(phot['MJD']), lists([b[-1] for b in phot['BAND']]),
         lists(phot['FLUXCAL'] * np.float64(to_njy)),
         lists(phot['FLUXCALERR'] * np.float64(to_njy)),
         pa.array(no_flag), pa.array(no_flag)],
        names=['midpointMjdTai', 'band', 'psfDiffFlux', 'psfDiffFluxErr',
               'psfDiffFlux_flag', 'invalidPsfFlag'])
    catalogue = tmp_path / "subsample.hats"
    (catalogue / "dataset" / "Norder=0" / "Dir=0").mkdir(parents=True)
    (catalogue / "hats.properties").write_text("obs_collection=dia_object_lc\n")
    pq.write_table(pa.table({
        'diaObjectId': head['SNID'].astype(np.int64), 'ra': head['RA'],
        'dec': head['DEC'], 'diaObjectForcedSource': lightcurve}),
        catalogue / "dataset" / "Norder=0" / "Dir=0" / "Npix=0.parquet")

    kw = dict(fit_bands=_DES_BANDS,
              filter_map={b: i for i, b in enumerate(_DES_BANDS)})
    head_rubin, phot_rubin, header = read_rubin_hats(catalogue, {})
    rubin = batch_from_columns(head_rubin, phot_rubin, **header, **kw)
    snana = load_and_batch_snana_fits(str(fx.head_fits), str(fx.phot_fits),
                                      **kw)

    assert np.array_equal(np.asarray(rubin.snids), np.asarray(snana.snids))
    assert rubin.zp_fluxcal == RUBIN_ZP_NJY
    ok = np.asarray(snana.observation_mask)
    assert np.array_equal(np.asarray(rubin.observation_mask), ok)
    for name in ('mjd_offsets', 'filter_indices'):
        assert np.array_equal(np.asarray(getattr(rubin, name))[ok],
                              np.asarray(getattr(snana, name))[ok]), name
    # FLUXCAL on Rubin's zero point, whatever the fixture's
    to_rubin = 10.0 ** (0.4 * (RUBIN_ZP_NJY - snana.zp_fluxcal))
    for name in ('fluxcals', 'fluxcalerrs'):
        assert np.allclose(np.asarray(getattr(rubin, name))[ok],
                           np.asarray(getattr(snana, name))[ok] * to_rubin,
                           rtol=1e-12, atol=0.0), name


def test_salt3_fitter_text_redshift_fallback(snana_fixture, tmp_path):
    """Real TEXT files may carry only REDSHIFT_HELIO; the loader must not create
    a spurious REDSHIFT_FINAL that defeats the builder's fallback to HELIO."""
    fx = snana_fixture("salt3_fit_text")
    for dat in fx.root.glob("*.DAT"):
        lines = [ln for ln in dat.read_text().splitlines()
                 if not ln.startswith("REDSHIFT_FINAL:")]
        (tmp_path / dat.name).write_text("\n".join(lines))
    (tmp_path / "V.LIST").write_text(
        "\n".join(p.name for p in tmp_path.glob("*.DAT")))
    batch = load_and_batch_snana_text(
        str(tmp_path), fit_bands=_DES_BANDS,
        filter_map={b: i for i, b in enumerate(_DES_BANDS)})
    assert np.isfinite(np.asarray(batch.redshifts)).all()


def test_salt3_fitter_setpkmjd_clump(snana_fixture):
    """OPT_SETPKMJD=16 (Fmax-clump) fit-side peak estimate. Loading with the
    setpkmjd knob overrides the header PEAKMJD with the shared Fmax-clump
    estimator; the per-SN peak must reproduce SNANA's PKMJDINI (from a
    snlc_fit run with OPT_SETPKMJD=16, MJDWIN=60, SNRCUT=5 on this fixture)."""
    fx = snana_fixture("salt3_fit_select")
    batch = load_and_batch_snana_fits(
        str(fx.root / "HEAD.FITS"), str(fx.root / "PHOT.FITS"),
        filter_map={b: i for i, b in enumerate(_DES_BANDS)},
        fit_bands=_DES_BANDS, photflag_mskrej=(1024,),
        snfield_list=["C3", "E1", "S1", "S2", "C1", "C2", "X1"],
        setpkmjd={"fmax_clump": True, "bazin": False, "mjdwin": 60.0,
                  "snrcut": 5.0, "snrcut_backup": 3.0,
                  "num_nobs_bins": 1, "min_bin_gain": 0.05,
                  "batch_size": 1000})
    cids = np.asarray(batch.snids).astype(int)
    peak = np.asarray(batch.peakmjd_offsets) + np.asarray(batch.reference_mjds)
    cid2peak = {int(c): float(p) for c, p in zip(cids, peak)}
    # SNANA PKMJDINI anchors (snlc_fit OPT_SETPKMJD=16 on this RANSEED)
    for cid, pkmjdini in ((26, 56332.0703), (166, 56275.1719),
                          (611, 56180.1953)):
        assert abs(cid2peak[cid] - pkmjdini) < 0.05, (
            f"CID {cid}: jax clump {cid2peak[cid]:.3f} vs SNANA {pkmjdini:.3f}")


def _des_filter_lamavg(data_root):
    return KcorReader(str(data_root / "kcor" / "JAX_TEST_DES.fits")
                      ).filter_lamavg(_DES_BANDS)


def _epoch_cut_parity(fx, epoch_windows, expected_count, snrmax_anchors,
                      epoch_snrmin=None, filter_lamavg=None):
    """Load an epoch-quality CUTWIN cut and check jax reproduces SNANA's
    accepted count and per-SN SNRMAX (SNANA-table anchors from snlc_fit)."""
    batch = load_and_batch_snana_fits(
        str(fx.root / "HEAD.FITS"), str(fx.root / "PHOT.FITS"),
        filter_map={b: i for i, b in enumerate(_DES_BANDS)},
        fit_bands=_DES_BANDS, photflag_mskrej=(1024,),
        snfield_list=["C3", "E1", "S1", "S2", "C1", "C2", "X1"],
        epoch_windows=epoch_windows, epoch_snrmin=epoch_snrmin,
        filter_lamavg=filter_lamavg)
    with fits.open(resolve_fits_path(fx.root / "HEAD.FITS")) as h:
        hd = Table(h[1].data)
    s2t = {int(s.strip()): int(t) for s, t in
           zip(np.array(hd['SNID'], dtype='U'), np.array(hd['SNTYPE']))}
    cids = np.asarray(batch.snids).astype(int)
    st = np.array([s2t[c] for c in cids])
    keep, _ = apply_snana_selection(
        batch, head_sntypes=st,
        windows={"redshift": (0.05, 0.40), "snrmax": (5.0, 1.0e8)},
        sntype_list=[0, 1], ladder=[])
    assert int(keep.sum()) == expected_count, (
        f"jax accepted {int(keep.sum())}, SNANA {expected_count}")

    obs_ok = np.asarray(batch.observation_mask)
    err = np.asarray(batch.fluxcalerrs)
    snr = np.where(obs_ok & (err > 0),
                   np.asarray(batch.fluxcals) / np.where(err > 0, err, 1.0), -9.0)
    cid2snrmax = {int(c): float(s) for c, s in zip(cids, snr.max(axis=1))}
    for cid, snrmax in snrmax_anchors:
        assert abs(cid2snrmax[cid] - snrmax) < 0.05, (
            f"CID {cid}: jax SNRMAX {cid2snrmax[cid]:.3f} vs SNANA {snrmax:.3f}")


@pytest.mark.parametrize("fixture,epoch_windows,count,anchors", [
    # SNANA snlc_fit accepted counts + SNANA-table SNRMAX on the named fixture
    ("salt3_fit_select", {"mjd": (56150.0, 56300.0)}, 58,
     ((166, 115.59476), (611, 119.23141))),
    ("salt3_fit_select", {"mjd_exclude": (56180.0, 56260.0)}, 69,
     ((26, 302.62454),)),
    ("salt3_fit_select", {"psf": (0.0, 1.2)}, 72,
     ((26, 302.62454), (237, 200.41476))),
    ("salt3_fit_select", {"tobs": (-12.0, 40.0)}, 68, ()),
    ("salt3_fit_select", {"trest": (-10.0, 30.0)}, 67, ()),
    ("salt3_fit_select", {"errtest": (0.8, 1.2)}, 75,
     ((26, 8.27740), (148, 38.10913))),
    ("salt3_fit_quality", {"zperr": (0.0, 0.03)}, 73,
     ((26, 301.57227), (476, 59.64759))),
    ("salt3_fit_quality", {"photprob": (0.5, 1.0)}, 74,
     ((26, 301.57227), (237, 195.92963))),
])
def test_salt3_fitter_epoch_quality_cuts(snana_fixture, fixture, epoch_windows,
                                         count, anchors):
    """Epoch-quality CUTWIN cuts (SNANA LSIG per-epoch masks): dropping epochs
    that fail the cut must reproduce SNANA's accepted count and per-SN SNRMAX
    exactly, since the mask feeds both the SNRMAX cut variable and the fit."""
    _epoch_cut_parity(snana_fixture(fixture), epoch_windows, count, anchors)


def test_salt3_fitter_epcut_snrmin(snana_fixture):
    """EPCUT_SNRMIN per-filter S/N floor (snana.F90:17390): 'griz 8' rejects
    epochs below S/N 8 in every band; SNANA accepts 71."""
    _epoch_cut_parity(snana_fixture("salt3_fit_select"), None, 71, (),
                      epoch_snrmin={"g": 8.0, "r": 8.0, "i": 8.0, "z": 8.0})


@pytest.mark.parametrize("epoch_windows,count,anchors", [
    ({"lamobs": (7000.0, 10000.0)}, 61, ((26, 117.45387), (237, 78.50481))),
    ({"lamrest": (5000.0, 8000.0)}, 69, ((26, 122.04475), (237, 91.59395))),
])
def test_salt3_fitter_lam_cuts(snana_fixture, data_root, epoch_windows,
                               count, anchors):
    """CUTWIN_LAMOBS/LAMREST filter-wavelength epoch cuts (SNANA LLAM): drop
    bands whose (rest-frame) mean wavelength falls outside the window."""
    _epoch_cut_parity(snana_fixture("salt3_fit_select"), epoch_windows, count,
                      anchors, filter_lamavg=_des_filter_lamavg(data_root))


def test_salt3_fitter_prior_mjdsig(snana_fixture):
    """PRIOR_MJDSIG=5 fixture: same photometry as salt3_fit_dense (same
    sim seed), refit by SNANA with the tighter t0 prior. jax matches at
    the same 3-sigma pull gate, and the knob's teeth are guarded by the
    SNANA-side A/B: the tighter prior must shrink reported PKMJD errors."""
    fx = snana_fixture("salt3_fit_mjdsig")
    dense_data = snana_fixture("salt3_fit_dense")
    results, _ = fit_fixture(fx.root, data_dir=dense_data.root, max_fit=80)
    _assert_pull_parity(results, fx.root / "FITRES.TEXT")

    dense = snana_fixture("salt3_fit_dense")
    rows5 = parse_fitres(fx.root / "FITRES.TEXT")
    rows20 = parse_fitres(dense.root / "FITRES.TEXT")
    common = sorted(set(rows5) & set(rows20))
    assert len(common) > 30
    err5 = np.array([rows5[c]["PKMJDERR"] for c in common])
    err20 = np.array([rows20[c]["PKMJDERR"] for c in common])
    # dense-cadence t0 is data-dominated (errors ~0.3 d << prior width),
    # so the prior bites only weakly-constrained SNe — require it to
    # visibly shrink at least one, and never inflate the population
    assert float(np.min(err5 / err20)) < 0.8
    assert float(np.median(err5 / err20)) <= 1.0



def test_salt_evaluation_agrees_with_the_generic_implementation(snana_fixture):
    """SALT overrides ``evaluate`` and ``evaluate_redshift_grid`` because pre-integrating
    M0/M1 through the coloured filters beats generating an SED per observation,
    and because its flux is linear in shape so a whole grid shares one band
    integral. The base implementations compute the same quantities the slow,
    obvious way, so running both on one model instance is an independent check
    of SALT's algebra and the only coverage the generic route has.
    """
    fx = snana_fixture("salt3_fit_dense")
    config = fit_config(fx.root / "fit.nml")
    batch = load_batch(config, fx.root / "HEAD.FITS", fx.root / "PHOT.FITS",
                       max_fit=1)
    model = build_fit_model(config)
    fitter = build_fitter(config, model, batch)
    photometry = fitter_photometry(fitter, batch)
    static_data = {**fitter.static_data, **photometry}
    no_att = jnp.ones(len(photometry['common_filter_wave_grid']))

    mjds = jnp.asarray(batch.mjd_offsets)[0]
    fi = jnp.asarray(batch.filter_indices)[0]
    mask = jnp.asarray(batch.observation_mask)[0]
    params = model.params_at_distance(
        jnp.array([float(jnp.asarray(batch.peakmjd_offsets)[0]),
                   0.7, -0.05, 0.0]),
        config.fit.distance_modulus(jnp.asarray(batch.redshifts))[0],
        names=fitter.param_names)

    z = jnp.asarray(batch.redshifts)[:1]
    bw, _ = jax.vmap(functools.partial(fitter.band_weights, photometry),
                     in_axes=(0, None))(z, no_att)
    sed_params = model.sed_params_from_fit(params, z[0])
    flux_fast = model.evaluate(sed_params, mjds, fi, mask, bw[0], static_data)
    flux_slow = SNModel.evaluate(model, sed_params, mjds, fi, mask, bw[0],
                                 static_data)
    np.testing.assert_allclose(flux_fast, flux_slow, rtol=1e-12,
                               err_msg="flux")

    # The error snake scales the flux, so the two flux paths must carry
    # through to the same error as well.
    frac = model.compute_raw_fracerr(sed_params, mjds, fi, mask, bw[0],
                                     static_data, band_weights_err=bw[0])
    np.testing.assert_allclose(
        flux_error_from_fracerr(flux_fast, frac),
        flux_error_from_fracerr(flux_slow, frac),
        rtol=1e-12, err_msg="err")

    # the grid: one call must lay out [n_z, *axes, n_obs] the way a plain loop
    # over the same points does, each point's brightness set from its own
    # distance, shape and colour
    axes = coarse_grid_axes(config, model.get_fit_parameters())
    z_grid = jnp.linspace(0.2, 0.5, 4)
    mu_grid = config.fit.distance_modulus(z_grid)
    bw_grid, _ = jax.vmap(functools.partial(fitter.band_weights, photometry),
                          in_axes=(0, None))(z_grid, no_att)
    lengths = [len(v) for v in axes.values()]

    for overrides in (None, {"x1": 0.0}):
        fast = model.evaluate_redshift_grid(
            axes, params, mu_grid, z_grid, bw_grid, bw_grid, mjds, fi, mask,
            static_data, error_overrides=overrides)
        slow = FittedModel.evaluate_redshift_grid(
            model, axes, params, mu_grid, z_grid, bw_grid, bw_grid, mjds, fi,
            mask, static_data, error_overrides=overrides)
        for got, ref, name in zip(fast, slow, ("flux", "err")):
            assert got.shape == (4, *lengths, mjds.shape[0])
            np.testing.assert_allclose(got, ref, rtol=1e-12,
                                       err_msg=f"{name} {overrides}")


def test_vmax_solves_for_the_redshift_of_the_magnitude_limit(snana_fixture):
    """MAGLIM_VMAX: z_Vmax is the redshift at which the fitted SN reaches
    MAGLIM (VMAX_ANA, snlc_fit.F90:15170), so re-evaluating the model there
    must return the limit; vol_Vmax is the comoving volume out to it. SNANA
    writes these to its ROOT SNFIT block and never to the text tables, so the
    defining property is what a bundled fixture can check."""
    from jax_snlc_fit.vmax import (compute_vmax, dvdz_integral,
                                   MAGDIF_CONVERGE, Z_NULL)

    fx = snana_fixture("salt3_fit_dense")
    config = fit_config(fx.root / "fit.nml")
    batch = load_batch(config, fx.root / "HEAD.FITS", fx.root / "PHOT.FITS",
                       max_fit=30)
    model = build_fit_model(config)
    # the zero point vmax works in comes from the loaded data (snlc_fit.py:1009)
    fitter = build_fitter(config, model, batch)
    results = run_fit(fitter, batch)

    mjds = np.asarray(batch.reference_mjds)[:, None] + np.asarray(batch.mjd_offsets)
    params = {k: np.asarray(v) for k, v in results.params.items()}
    attached = for_fit(fitter, batch)
    model_flux, _ = fitter.evaluate_flux(
        params, mjds, np.asarray(batch.redshifts),
        np.asarray(batch.filter_indices), np.asarray(batch.observation_mask),
        reference_mjds=np.asarray(batch.reference_mjds),
        mw_att_obs=attached.mw_att_obs, photometry=attached.photometry)
    model_flux = np.asarray(model_flux)
    fit_mask = np.asarray(batch.observation_mask)

    # SNANA moves a SN in distance by scaling x0 (snlc_fit.F90:15294). Pinned
    # here because the z_Vmax check below iterates with this same function and
    # so cannot see an error in it.
    v = np.array([2.0e-5])
    np.testing.assert_allclose(model.shift_amplitude({"x0": v}, 0.5)["x0"],
                               v * 10.0 ** (-0.4 * 0.5), rtol=1e-12)

    maglim = {"r": 23.0}
    cospar = (config.fit.h0_ref[0], config.fit.omat_ref[0], config.fit.olam_ref[0],
              config.fit.w0_ref[0], config.fit.wa_ref[0])
    out = compute_vmax(fitter, results, attached, fit_mask, model_flux,
                       maglim, cospar)

    zmax = np.asarray(out["z_Vmax_r"])
    solved = zmax > Z_NULL
    assert solved.sum() >= 10, f"too few SNe with a Vmax solution ({solved.sum()})"

    # the defining property: at z_Vmax the peak magnitude in-band is MAGLIM
    zp = batch.zp_fluxcal
    band = [i for i, nm in batch.filter_names.items()
            if nm.rsplit("-", 1)[-1] == "r"]
    in_band = fit_mask & np.isin(np.asarray(batch.filter_indices), band)
    peak = np.argmax(np.where(in_band, model_flux, -9.9e8), axis=1)
    rows = np.arange(batch.n_sne)
    mu0 = np.asarray(config.fit.distance_modulus(np.asarray(batch.redshifts)))
    mu1 = np.asarray(config.fit.distance_modulus(np.where(solved, zmax, 0.1)))
    moved = model.shift_amplitude({k: v.copy() for k, v in params.items()},
                                  mu1 - mu0)
    flux_at_zmax, _ = fitter.evaluate_flux(
        moved, mjds[rows, peak][:, None], np.where(solved, zmax, 0.1),
        np.asarray(batch.filter_indices)[rows, peak][:, None],
        np.ones((batch.n_sne, 1), dtype=bool),
        reference_mjds=np.asarray(batch.reference_mjds),
        mw_att_obs=attached.mw_att_obs, photometry=attached.photometry)
    mag_at_zmax = zp - 2.5 * np.log10(
        np.maximum(np.asarray(flux_at_zmax)[:, 0], 1e-30))
    worst = np.max(np.abs(mag_at_zmax[solved] - 23.0))
    assert worst <= MAGDIF_CONVERGE * 1.5, (
        f"z_Vmax does not put the SN at MAGLIM: worst |mag - 23.0| = {worst:.5f}")

    # vol_Vmax is the all-sky comoving volume out to that redshift (:152)
    np.testing.assert_allclose(
        np.asarray(out["vol_Vmax_r"])[solved],
        4.0 * np.pi * dvdz_integral(zmax[solved], cospar), rtol=1e-10)


def test_pre_fit_adjust_reverts_an_adjustment_that_made_the_chi2_worse(
        snana_fixture):
    """CHI8END worse than CHI8INI switches every starting value back to the
    one it came in with, the grid search's peak time excepted
    (snlc_fit.F90:9818-9831).

    Seeded at its own converged fit, the grid cannot beat the minimum and the
    amplitude rescale can only move away from it, so the revert fires for
    every event: round 1 starts from exactly the vector handed in. Seeded from
    cosmology, as the driver normally does, the rescale stands and x0 moves,
    so the assertion below separates the two.
    """
    fx = snana_fixture("salt3_fit_dense")
    config = fit_config(fx.root / "fit.nml")
    batch = load_batch(config, fx.root / "HEAD.FITS", fx.root / "PHOT.FITS",
                       max_fit=20)
    fitter = build_fitter(config, build_fit_model(config), batch)
    batch = for_fit(fitter, batch)
    photometry, att = batch.photometry, batch.mw_att_obs
    bw = jax.vmap(functools.partial(fitter.band_weights, photometry))(
        batch.redshifts, att)
    scales = fitter._scales(batch, None)
    mask = batch.observation_mask & fitter.restlambda_mask(
        photometry, batch.filter_indices, batch.redshifts)

    def adjusted(start):
        """FITINI_ADJUST's output from ``start``, and ``start``, physical."""
        params = fitter.stack(start, batch.reference_mjds) / scales
        out, _, _ = fitter._run_adjust(
            fitter.model, photometry, None, params, scales, batch.mjd_offsets,
            batch.filter_indices, mask, batch.fluxcals, batch.fluxcalerrs,
            batch.redshifts, bw, batch.n_obs_per_sn, None, att)
        return np.asarray(out * scales), np.asarray(params * scales)

    nominal, cosmology = adjusted(peak_start(fitter, batch))

    # from the converged answer, which no adjustment can improve on
    converged = dict(fitter.fit_sample(
        batch, peak_start(fitter, batch))[0].params)
    reverted, seeded = adjusted(converged)
    np.testing.assert_allclose(reverted, seeded, rtol=1e-12, atol=0.0)
    # the amplitude rescale is live: from cosmology it does move x0
    x0 = fitter.param_names.index('x0')
    assert not np.allclose(nominal[:, x0], cosmology[:, x0])

    # and the fit re-seeded at the reverted start is still SNANA's
    refit, _ = fitter.fit_sample(batch, converged)
    _assert_pull_parity(refit, fx.root / "FITRES.TEXT", min_compared=15)


def test_restlambda_usefit_narrows_to_the_model_range(
        salt3_model_path, lsst_kcor_path):
    """RESTLAMBDA_USEFIT is the user range intersected with the model's own
    (snlc_fit.F90:1421-1422), so a band the model cannot reach is dropped
    from the fit rather than modelled by extrapolating past its support."""
    model = build_model(str(salt3_model_path), {}, model_type="salt")
    fitter, _ = _lsst_fitter(model, salt3_model_path, lsst_kcor_path,
                             restlambda_fitrange=(0.0, 25000.0))
    model_lo, model_hi = fitter.static_data['model_wave_range']
    assert fitter.restlambda_usefit == (model_lo, model_hi)

    narrowed, _ = _lsst_fitter(model, salt3_model_path, lsst_kcor_path,
                               restlambda_fitrange=(3000.0, 8000.0))
    assert narrowed.restlambda_usefit == (max(model_lo, 3000.0),
                                          min(model_hi, 8000.0))


def test_simfit_ideal_refits_a_prescaled_subset_on_true_flux(snana_fixture):
    """SIMFIT_IDEAL_PRESCALE runs a second fit on the noise-free flux for
    every PS-th event (snlc_fit.F90:15808), reporting its parameters as
    _IDEAL columns.

    The pass is given the resolved bands, so the per-band FUDGE_FITERR array
    it indexes exists, and the USESIM pins the nominal fit holds, so a pinned
    parameter comes back at the truth it was seeded from. A subset wider than
    one batch still fits.
    """
    fx = snana_fixture("salt3_fit_dense")
    config = fit_config(fx.root / "fit.nml", n_fit_iterations=1,
                        maxiter_round1=15, maxiter_round2=10,
                        simfit_ideal_prescale=2, batch_size=4,
                        fudge_fiterr_maxfrac=0.02,
                        fudge_fiterr_passbands='gri', usesim_params=['x1'])
    model = build_fit_model(config)
    batch = select_events(load_batch(
        config, fx.root / "HEAD.FITS", fx.root / "PHOT.FITS", max_fit=24,
        keep_obs_extras=True, model_sim_columns=model_sim_columns(model)),
        {}, [], config.selection.mxevt_process, config.selection.mxevt_cuts)
    fitter = build_fitter(config, model, batch)

    columns = ideal_fit(fitter, fitter.config, for_fit(fitter, batch))

    assert {'x1_IDEAL', 'c_IDEAL', 'PKMJD_IDEAL'} <= set(columns)
    x1 = columns['x1_IDEAL']
    refitted = x1 != 999.0
    # the prescale takes a subset, and more than one batch of 4
    assert 4 < refitted.sum() < batch.n_sne

    # x1 was pinned by USESIM and seeded at its truth, so the ideal fit
    # reports exactly the simulated value
    np.testing.assert_allclose(
        x1[refitted],
        np.asarray(batch.sim_truth['SIM_SALT2x1'], dtype=float)[refitted],
        atol=1e-6)


def test_the_fit_records_which_round1_window_rejected_each_sn(snana_fixture):
    """A round-1 window that no event can satisfy rejects every event, and
    the result says which window did it, under that window's own name. Only
    the windows the input configured are reported; the SHAPE window resolves
    to SALT's x1 through the parameter's declared legacy keys."""
    fx = snana_fixture("salt3_fit_dense")
    kw = dict(max_fit=3)

    shape_only, _ = fit_fixture(fx.root, fitwin_iter1={'SHAPE': (5.0, 6.0)},
                                **kw)
    assert set(shape_only.iter1_verdicts) == {'SHAPE'}
    assert not np.any(np.asarray(shape_only.iter1_verdicts['SHAPE']))

    both, _ = fit_fixture(fx.root, fitwin_iter1={'SHAPE': (5.0, 6.0),
                                                 'COLOR': (5.0, 6.0)}, **kw)
    assert set(both.iter1_verdicts) == {'SHAPE', 'COLOR'}
    assert not np.any(np.asarray(both.iter1_verdicts['COLOR']))

    prob, _ = fit_fixture(fx.root, fitwin_prob_iter1=(1.1, 2.0), **kw)
    assert set(prob.iter1_verdicts) == {'PROB'}
    assert not np.any(np.asarray(prob.iter1_verdicts['PROB']))

    adjust, _ = fit_fixture(fx.root, fitwin_chi2red_ini=(0.0, 1e-12),
                            fitwin_iter1={'SHAPE': (5.0, 6.0)}, **kw)
    assert set(adjust.iter1_verdicts) == {'SHAPE', 'CHI2RED_INI'}
    assert not np.any(np.asarray(adjust.iter1_verdicts['CHI2RED_INI']))

    passes, _ = fit_fixture(fx.root, **kw)
    assert passes.iter1_verdicts == {}


def test_mw_extinction_uses_each_event_s_av(snana_fixture):
    """Somm25's A(lam)/A(V) shape reads AV, so SNANA rebuilds the MW
    extinction table with each event's AV = RV * E(B-V)
    (fill_TABLE_MWXT_SEDMODEL, genmag_SEDtools.c:2580-2645): two sightlines
    get two curve shapes, not one curve scaled twice. A law whose shape does
    not read AV still resolves to one curve, so its arithmetic is unchanged.
    """
    fx = snana_fixture("salt3_fit_dense")
    config = fit_config(fx.root / "fit.nml")
    batch = select_events(load_batch(
        config, fx.root / "HEAD.FITS", fx.root / "PHOT.FITS", max_fit=2,
        keep_obs_extras=True), {}, [], config.selection.mxevt_process,
        config.selection.mxevt_cuts)
    ebv = np.array([0.02, 0.40])
    batch = dataclasses.replace(batch, mwebvs=ebv)

    def axav(opt_mwcolorlaw, filtlist, parlist=()):
        """A(lam)/A(V) per SN, read back off the attenuation the fitter built."""
        narrowed = fit_config(fx.root / "fit.nml", opt_mwebv=1,
                              opt_mwcolorlaw=opt_mwcolorlaw,
                              filtlist_fit=filtlist, parlist_mwcolorlaw=parlist)
        fitter = build_fitter(narrowed, build_fit_model(narrowed), batch)
        att = np.asarray(for_fit(fitter, batch).mw_att_obs)
        _, rv = resolve_mw_colour_law(narrowed.data, batch.sim_mw_colour_law)
        return -np.log10(att) / (0.4 * rv * ebv[:, None])

    # Somm25 ends at 9134 A, short of z and Y: a law that does not cover the
    # observer grid aborts as SNANA does (MWgaldust.c:688-694, :1163)
    with pytest.raises(SNANAError, match="only valid"):
        axav(225, 'ugrizY')
    somm25 = axav(225, 'ugri')
    assert not np.allclose(somm25[0], somm25[1], rtol=1e-3)

    ccm89 = axav(89, 'ugrizY')
    np.testing.assert_allclose(ccm89[0], ccm89[1], rtol=1e-4)

    # Somm26 takes its four curve parameters from PARLIST_MWCOLORLAW
    # (MWgaldust.c:562-568), ends at 10000 A short of Y, and reads AV only
    # as a scale
    with pytest.raises(SNANAError, match="requires PARLIST_MWCOLORLAW"):
        axav(226, 'ugriz')
    with pytest.raises(SNANAError, match="only valid"):
        axav(226, 'ugrizY', (1.0, 1.0, 0.5, 1.5))
    somm26 = axav(226, 'ugriz', (1.0, 1.0, 0.5, 1.5))
    np.testing.assert_allclose(somm26[0], somm26[1], rtol=1e-4)
    assert not np.allclose(somm26[0], axav(89, 'ugriz')[0], rtol=1e-2)


def test_mw_extinction_error_is_the_difference_snana_takes(snana_fixture):
    """The Milky Way extinction error is XT(E(B-V) + error) - XT(E(B-V))
    (snana.F90:15624-15629). Somm25's curve changes shape with AV, so only
    that difference telescopes: error(E, e) + error(0, E) = error(0, E + e).
    An event with no E(B-V) carries the extinction of its error."""
    fx = snana_fixture("salt3_fit_dense")
    config = fit_config(fx.root / "fit.nml", opt_mwebv=1,
                        opt_mwcolorlaw=225, filtlist_fit='ugri')
    batch = load_batch(config, fx.root / "HEAD.FITS", fx.root / "PHOT.FITS",
                       max_fit=1)
    fitter = build_fitter(config, build_fit_model(config), batch)

    def error(ebv, ebv_err):
        return np.asarray(for_fit(fitter, dataclasses.replace(
            batch, mwebvs=np.array([ebv]), mwebv_errs=np.array([ebv_err])))
            .mwxt_magerrs)[0]

    assert np.all(error(0.0, 0.05) > 0.0)
    np.testing.assert_allclose(error(0.1, 0.05) + error(0.0, 0.1),
                               error(0.0, 0.15), rtol=1e-6)


def test_unset_mw_colour_law_follows_the_simulation(snana_fixture, tmp_path):
    """With OPT_MWCOLORLAW and RV_MWCOLORLAW unset, SNANA fits a simulation
    with the law and R_V its header records and real data with 99 and 3.1; a
    value the user gives wins over both (INIT_GALextinct).

    The fixture is an SNANA simulation made with Somm25; a copy records
    R_V = 2.5, and a second copy is marked as real data.
    """
    fx = snana_fixture("mwdust_somm25")
    nml = snana_fixture("salt3_fit_mwebv").root / "fit.nml"

    def header_copy(name, **cards):
        path = tmp_path / name
        with fits.open(fx.head_fits) as hdul:
            hdul[0].header.update(cards)
            hdul.writeto(path)
        return path

    sim = header_copy("SIM_HEAD.FITS", SIM_MWRV=2.5)
    data = header_copy("DATA_HEAD.FITS", DATATYPE="DATA")

    def fitted(head, **keys):
        """The fit's (law, R_V) and the attenuation it builds."""
        config = fit_config(nml, filtlist_fit='ugri', **keys)
        batch = load_batch(config, head, fx.root / "PHOT.FITS", max_fit=3)
        assert np.all(np.asarray(batch.mwebvs) > 0.0)
        fitter = build_fitter(config, build_fit_model(config), batch)
        return (resolve_mw_colour_law(config.data, batch.sim_mw_colour_law),
                np.asarray(for_fit(fitter, batch).mw_att_obs))

    law, att = fitted(sim)
    assert law == (225, 2.5)
    _, att_given = fitted(sim, opt_mwcolorlaw=225, rv_mwcolorlaw=2.5)
    np.testing.assert_array_equal(att, att_given)
    _, att_default = fitted(sim, opt_mwcolorlaw=99, rv_mwcolorlaw=3.1)
    assert not np.allclose(att, att_default, rtol=1e-3)

    assert fitted(data)[0] == (99, 3.1)
    assert fitted(sim, rv_mwcolorlaw=3.3)[0] == (225, 3.3)
    assert fitted(sim, opt_mwcolorlaw=89)[0] == (89, 2.5)


def test_opt_mwcolorlaw_zero_applies_no_mw_extinction(snana_fixture, tmp_path):
    """OPT_MWCOLORLAW 0 is "No Extinction" (MWgaldust.h:27): with a non-zero
    MWEBV and MWEBV_ERR the attenuation is 1 and the extinction error 0,
    whether the law is set in the namelist or read from a simulation header."""
    fx = snana_fixture("mwdust_opt1")
    nml = snana_fixture("salt3_fit_mwebv").root / "fit.nml"
    off_head = tmp_path / "OFF_HEAD.FITS"
    with fits.open(fx.head_fits) as hdul:
        hdul[0].header['SIMOPT_MWCOLORLAW'] = 0
        hdul.writeto(off_head)

    def attenuation(head, **keys):
        config = fit_config(nml, filtlist_fit='ugri', **keys)
        batch = load_batch(config, head, fx.root / "PHOT.FITS", max_fit=3)
        assert np.all(np.asarray(batch.mwebvs) > 0.0)
        assert np.all(np.asarray(batch.mwebv_errs) > 0.0)
        fitter = build_fitter(config, build_fit_model(config), batch)
        attached = for_fit(fitter, batch)
        law, _ = resolve_mw_colour_law(config.data, batch.sim_mw_colour_law)
        return (law, np.asarray(attached.mw_att_obs),
                np.asarray(attached.mwxt_magerrs))

    for head, keys in ((fx.root / "HEAD.FITS", {'opt_mwcolorlaw': 0}),
                       (off_head, {})):
        law, att, magerrs = attenuation(head, **keys)
        assert law == 0
        np.testing.assert_array_equal(att, 1.0)
        np.testing.assert_array_equal(magerrs, 0.0)
    law, att, _ = attenuation(fx.root / "HEAD.FITS")
    assert law == 99
    assert np.any(att < 0.99)


def assert_analytic_derivatives(f, x, name):
    """``f``'s first and second derivatives against central differences of it.

    A model supplying its own differentiation rule is never differentiated
    automatically, so nothing downstream notices a rule that drops a term:
    a direction can return exactly zero and every fit still converges to
    somewhere plausible. Central differences of the primal never enter the
    rule, so they pin the gradient; central differences of that gradient pin
    the Hessian against the gradient, which is the second level of the rule.

    ``f`` maps a one-dimensional vector to a scalar and ``x`` is the point.
    Every element is held to its own relative tolerance, above a floor of
    float64 rounding over the step. The models are linear interpolants in
    redshift, so the difference must stay within a few kinks: its error
    grows linearly with the step, and 1e-5 keeps it below tolerance.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    grad_f = jax.grad(f)
    grad = np.asarray(grad_f(x))
    hess = np.asarray(jax.jacfwd(grad_f)(x))
    scale = np.maximum(np.abs(np.asarray(x)), 1.0)
    steps = 1.0e-5 * scale
    rounding = np.finfo(np.float64).eps * abs(float(f(x)))
    assert np.abs(grad).max() > 0.0, f"{name}: the gradient is identically zero"

    for i in range(x.size):
        e = jnp.zeros_like(x).at[i].set(steps[i])
        num = float(f(x + e) - f(x - e)) / (2.0 * steps[i])
        assert abs(grad[i] - num) <= (1.0e-4 * max(abs(grad[i]), abs(num))
                                      + rounding / steps[i]), (
            f"{name}: d/d{i} analytic {grad[i]:.8g} vs numeric {num:.8g}")
        hnum = (np.asarray(grad_f(x + e)) - np.asarray(grad_f(x - e))) \
            / (2.0 * steps[i])
        tol = (1.0e-3 * np.maximum(np.abs(hess[i]), np.abs(hnum))
               + rounding / (scale * steps[i]))
        worst = int(np.argmax(np.abs(hess[i] - hnum) - tol))
        assert np.all(np.abs(hess[i] - hnum) <= tol), (
            f"{name}: d2/d{i}d{worst} analytic {hess[i][worst]:.8g} vs "
            f"numeric {hnum[worst]:.8g}")


def one_sn_flux(model, fitter, batch, sn, probe_z=False):
    """A scalar function of one SN's parameter vector, through the model's
    own evaluate; with ``probe_z`` the redshift is the vector's last entry
    and the band weights follow it, as the photo-z loss makes them."""
    mjds = jnp.asarray(batch.mjd_offsets[sn])
    fi = jnp.asarray(batch.filter_indices[sn])
    mask = jnp.asarray(batch.observation_mask[sn])
    z0 = float(batch.redshifts[sn])
    # a per-epoch weight so no epoch's contribution cancels another's
    probe = jnp.asarray(np.linspace(0.3, 1.7, mjds.size))
    photometry = fitter_photometry(fitter, batch)
    z_weights = fitter.z_grid_band_weights(photometry) if probe_z else None
    static_data = {**fitter.static_data, **photometry}
    # a unit attenuation row: no MW extinction
    no_att = jnp.ones_like(photometry['common_filter_wave_grid'])
    fixed = fitter.band_weights(photometry, z0, no_att)

    def scalar(v):
        if probe_z:
            z = v[-1]
            def weights_dz(q):
                return jax.jvp(
                    lambda r: fitter.interpolated_band_weights(
                        photometry, z_weights, r, no_att),
                    (q,), (1.0,))
            ((bw, _bwe), (bwz, _bwez)), (_, (bwzz, _)) = jax.jvp(
                weights_dz, (z,), (1.0,))
            extra = dict(band_weights_dz=bwz, band_weights_dz2=bwzz)
            params = v[:-1]
        else:
            z, bw = z0, fixed[0]
            extra, params = {}, v
        flux = model.evaluate(model.sed_params_from_fit(params, z), mjds, fi,
                              mask, bw, static_data, **extra)
        return jnp.sum(flux * probe)

    return scalar


def test_salt_colour_law_derivatives_match_the_flux(snana_fixture):
    """SALT's pre-integrated band flux differentiates its own flux.

    The colour integrator supplies its own rule so that the wavelength axis
    closes before the parameter axis opens, at both derivative levels. The
    redshift case is the sharp one: the band weights' redshift derivative is
    handed to the rule rather than differentiated through, so it is carried
    by a term of its own that nothing downstream would miss.
    """
    fx = snana_fixture("salt3_fit_dense")
    base = fit_config(fx.root / "fit.nml")
    batch = load_batch(base, fx.head_fits, fx.phot_fits, max_fit=4)
    model = build_fit_model(base)
    z = jnp.asarray([float(batch.redshifts[0])])
    t0 = (float(batch.peakmjd_offsets[0])
          if batch.peakmjd_offsets is not None else 0.0)

    for probe_z in (False, True):
        config = fit_config(fx.root / "fit.nml",
                            **({"dofit_photoz": True} if probe_z else {}))
        fitter = build_fitter(config, model, batch)
        init = init_params(model, 1, jnp.asarray([t0]),
                           config.fit.distance_modulus(z))
        v = jnp.stack([jnp.asarray(init[n])[0]
                       for n in fitter.model_param_names]).astype(jnp.float64)
        if probe_z:
            v = jnp.concatenate([v, z.astype(jnp.float64)])
        assert_analytic_derivatives(
            one_sn_flux(model, fitter, batch, 0, probe_z=probe_z), v,
            f"SALT band flux{' with redshift' if probe_z else ''}")


def test_a_held_colour_has_no_covariance_and_a_free_subspace_edm(snana_fixture):
    """INISTP_c = 0 pins the colour: its errors and covariances are zero,
    and the EDM judges convergence on the parameters that moved, so the
    held fit converges as the free one does rather than reporting the
    pinned direction's gradient as a distance to the minimum."""
    fx = snana_fixture("salt3_fit_dense")
    config = fit_config(fx.root / "fit.nml")
    config.model.inistp['c'] = 0.0
    batch = load_batch(config, fx.root / "HEAD.FITS", fx.root / "PHOT.FITS", 40)
    fitter = build_fitter(config, build_fit_model(config), batch)
    results = run_fit(fitter, batch)

    ic = list(fitter.param_names).index('c')
    cov = np.asarray(results.param_covariances)
    assert np.all(cov[:, ic, :] == 0) and np.all(cov[:, :, ic] == 0)
    assert np.all(np.asarray(results.param_errors['c']) == 0)
    edm = np.asarray(results.edm)
    assert np.isfinite(edm).all() and np.mean(edm < 1e-3) > 0.8


def test_lcplot_rebuilt_under_usesim_trueflux(snana_fixture, tmp_path):
    """Under USESIM_TRUEFLUX the fit plots and flags the simulated true flux
    (snana.F90:17341), and an LCPLOT rebuilt from its FITRES reads the data
    the same way."""
    fx = snana_fixture("salt3_fit_dense")
    overrides = ("USESIM_TRUEFLUX", "T", "NFIT_ITERATION", "1",
                 "SNTABLE_LIST", "FITRES LCPLOT", "MXLC_PLOT", "8",
                 "MAX_FIT", "8")
    fit_rows, rebuilt, _ = regenerate_lcplot(fx.root, tmp_path, *overrides)
    diff = compare_lcplots(fit_rows, rebuilt)
    assert diff["missing"] == 0 and diff["data_changed"] == 0, diff
    assert diff["flags"] == [], diff
