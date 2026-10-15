"""End-to-end BBC fit parity vs SALT2mu.exe (converged 1x1-grid fixture).

Companion to test_bbc_biascor (map-only): runs the full run_bbc_fit driver and
checks alpha/beta/sigint and the per-z-bin M0DIF against the SALT2mu.exe
reference. Config is SNANA's default fitflag_sigmb=0 -> single fit at sigint=0.

Everything runs on the bbc_des_beams fixture set.
"""

from jax_snlc_core.io.fits_path import open_text

import re

import jax.numpy as jnp
import numpy as np
import pytest

from jax_salt2mu.tablevar import force_probcc0
from jax_snlc_core.physics.constants import LIGHT_KM
from jax_salt2mu.config import BBCConfig, M0FITFLAG_CONSTANT, resolve
from jax_salt2mu.likelihood import h11_dPdmu
from jax_salt2mu.output import write_cov, write_fitres, write_m0dif, write_yaml
from jax_salt2mu.results import YamlSummary
from jax_salt2mu.run import run_bbc_fit
from jax_salt2mu.tablevar import EVENT_TYPE_DATA, read_tablevar
from jax_snlc_core.io.fitres_reader import read_fitres
from jax_snlc_core.models import SALTModel
from jax_snlc_core.utils.error_handling import SNANAError


# The biasCor-variant cases run on the DES fixtures: same keys, a sample whose
# light curves are properly sampled. See bbc_des_beams/README.md -- with sigint
# fitted it recovers its own SIM_alpha to 0.3%, where the LSST set is 17% out.
_DES = "tests/fixtures/snana/bbc_des_beams"
_DES_NSNFIT = 1474
_DES_GROUP = "DES(zbin=0.15:x1bin=2.0:cbin=0.2)"


def _cfg(fitflag_sigmb, opt_biascor=112, surveygroup=None):
    cfg = BBCConfig(
        radiation=False,
        datafile=[f"{_DES}/DATA.FITRES"],
        nzbin=10, zmin=0.1, zmax=1.0,
        simfile_biascor=[f"{_DES}/BIASCOR.FITRES"], opt_biascor=opt_biascor,
        min_per_cell_biascor=3, snrmin_sigint_biascor=60.0,
        surveygroup_biascor=_DES_GROUP if surveygroup is None else surveygroup,
        fitflag_sigmb=fitflag_sigmb)
    cfg.nuisance_float["alpha0"] = True
    cfg.nuisance_float["beta0"] = True
    cfg.nuisance_onebin.update(alpha0=False, beta0=False)
    return cfg


def _des_cfg(fitflag_sigmb, opt_biascor=112, surveygroup=_DES_GROUP):
    return _cfg(fitflag_sigmb, opt_biascor, surveygroup)


def _des_check(cfg, m0dif_ref, alpha0, beta0, nsnfit=_DES_NSNFIT, tol=2e-5):
    """Run the fit, check it against the reference, and return its output."""
    result, output, _, _, _ = run_bbc_fit(cfg)
    assert int(output.nfit.sum()) == nsnfit
    assert abs(result.nuisance["alpha0"] - alpha0) < 2e-5
    assert abs(result.nuisance["beta0"] - beta0) < 1e-4
    ref = read_fitres(f"{_DES}/{m0dif_ref}")
    np.testing.assert_array_equal(output.nfit, np.asarray(ref.get("NFIT")))
    assert np.max(np.abs(output.m0dif - np.asarray(ref.get("MUDIF")))) < tol
    assert np.max(np.abs(output.muref - np.asarray(ref.get("MUREF")))) < tol
    return output


def test_bbc_fit_matches_salt2mu():
    # fitflag_sigmb=0 (SNANA default): single fit at sigint=0, no log term.
    _des_check(_des_cfg(0), "SALT2mu.M0DIF", alpha0=0.18189, beta0=3.79911)


def test_bbc_fit_mad_matches_salt2mu():
    """MAD (opt_biascor bit 2048, SALT2mu.c:13394): MUCOVSCALE = (1.48*median|pull|)^2
    instead of Var(pull). alpha0=0.18135 (vs 0.18189 without MAD)."""
    _des_check(_des_cfg(0, opt_biascor=112 | 2048), "SALT2mu_mad.M0DIF",
               alpha0=0.18135, beta0=3.80337)


def test_bbc_fit_gamma0_matches_salt2mu(bbc_biascor_variants):
    """Float the host mass-step gamma0. DATA_MASS carries a logmass spread [8,12] about
    logmass_cen=10 with no injected step, so gamma0 recovers ~0 -- SNANA reports
    -0.00143 +- 0.00637 -- exercising the fermi-step + fit path."""
    cfg = _des_cfg(0)
    cfg.datafile = [f"{_DES}/DATA_MASS.FITRES"]
    cfg.simfile_biascor = [f"{bbc_biascor_variants}/BIASCOR_MASS.FITRES"]
    cfg.nuisance_float["gamma0"] = True
    cfg.nuisance_onebin["gamma0"] = False
    cfg.nuisance_val.update(logmass_cen=10.0, logmass_tau=0.01)
    result, output, _, _, _ = run_bbc_fit(cfg)
    assert int(output.nfit.sum()) == _DES_NSNFIT
    assert abs(result.nuisance["gamma0"] + 0.00143) < 2e-5
    ref = read_fitres(f"{_DES}/SALT2mu_gamma.M0DIF")
    np.testing.assert_array_equal(output.nfit, np.asarray(ref.get("NFIT")))
    assert np.max(np.abs(output.m0dif - np.asarray(ref.get("MUDIF")))) < 2e-5


def test_bbc_fit_mucovadd_needs_the_mucovscale_bit():
    """Only the muCOVscale bit builds the map (SALT2mu.c:11532), so opt_biascor with
    muCOVADD but not muCOVSCALE applies neither correction -- the same answer as a
    plain 5D biasCor, not the muCOVADD one. The column is still written, because the
    writer gates on the bit alone (:24733)."""
    cfg = _des_cfg(0, opt_biascor=112 & ~32 | 4096)
    output = _des_check(cfg, "SALT2mu_covadd_nomucov.M0DIF",
                        alpha0=0.17952, beta0=3.78224)
    ref = read_fitres(f"{_DES}/SALT2mu_covadd_nomucov.FITRES")
    assert np.all(np.asarray(ref.get("biasCor_muCOVADD")) == 0.0)
    assert np.all(output.biascor["biasCor_muCOVADD"] == 0.0)


def test_bbc_fit_mucovadd_matches_salt2mu():
    """muCOVADD (opt_biascor bit 4096, forces MAD): per-cell additive sigInt in place
    of the multiplicative muCOVscale where the pull is underdispersed. alpha0=0.17828."""
    result, output, _, tv, accept = run_bbc_fit(_des_cfg(0, opt_biascor=112 | 4096))
    cids = [str(c) for c in tv.name[accept]]
    assert int(output.nfit.sum()) == _DES_NSNFIT
    assert abs(result.nuisance["alpha0"] - 0.17828) < 2e-5
    ref = read_fitres(f"{_DES}/SALT2mu_covadd.M0DIF")
    np.testing.assert_array_equal(output.nfit, np.asarray(ref.get("NFIT")))
    assert np.max(np.abs(output.m0dif - np.asarray(ref.get("MUDIF")))) < 2e-5

    # biasCor_muCOVADD/muCOVSCALE report the blends from before the fit's
    # additive-or-multiplicative choice, rerun with the writer's own condition.
    per = read_fitres(f"{_DES}/SALT2mu_covadd.FITRES")
    rc = [str(c) for c in per.get("CID")]
    for col, tol in (("biasCor_muCOVADD", 1e-4), ("biasCor_muCOVSCALE", 2e-3)):
        rmap = dict(zip(rc, np.asarray(per.get(col))))
        got = output.biascor[col]
        d = [got[i] - rmap[c] for i, c in enumerate(cids) if c in rmap]
        assert np.max(np.abs(d)) <= tol + 1e-9, col
    # the column is printed to four places, so a blend below 5e-5 reads as 0
    nz = np.asarray(per.get("biasCor_muCOVADD")) != 0
    assert int(nz.sum()) == int((np.round(output.biascor["biasCor_muCOVADD"], 4) != 0).sum())


def test_bbc_fit_bias_on_mu_matches_salt2mu():
    """opt_biascor bit 128: one bias per cell, on mu, from each biasCor event's
    own SIM alpha/beta (compute_more_TABLEVAR, SALT2mu.c:9149-9161), with
    coefficient 1 in the fit. The biasCor residual behind sigInt(biasCor) and
    muCOVscale then takes the input alpha/beta, 0 here with no p1/p2, which is
    why sigInt(5D-BIASCOR) is 0.319 against 0.083 for the parameter bias."""
    result, output, _, tv, accept = run_bbc_fit(_des_cfg(0, opt_biascor=112 | 128))
    assert int(output.nfit.sum()) == 1475       # one more valid cell than 112
    assert abs(result.nuisance["alpha0"] - 0.1791) < 1e-4
    assert abs(result.nuisance["beta0"] - 3.6725) < 1e-3
    assert abs(output.biascor_summary[3] - 0.319) < 1e-3
    ref = read_fitres(f"{_DES}/SALT2mu_mu.M0DIF")
    np.testing.assert_array_equal(output.nfit, np.asarray(ref.get("NFIT")))
    assert np.max(np.abs(output.m0dif - np.asarray(ref.get("MUDIF")))) < 2e-5

    per = read_fitres(f"{_DES}/SALT2mu_mu.FITRES")
    assert "biasCor_mB" not in per.varnames and "biasCor_mB" not in output.biascor
    rc = [str(c) for c in per.get("CID")]
    cids = [str(c) for c in tv.name[accept]]
    for col, tol in (("biasCor_mu", 1e-4), ("biasCor_muCOVSCALE", 2e-3)):
        rmap = dict(zip(rc, np.asarray(per.get(col))))
        got = output.biascor[col]
        d = [got[i] - rmap[c] for i, c in enumerate(cids) if c in rmap]
        assert np.max(np.abs(d)) <= tol + 1e-9, col


@pytest.mark.parametrize("fitflag_sigmb,chi2max,ref,nsnfit,alpha0,beta0", [
    (0, 16, "SALT2mu_chi2max16.M0DIF", 887, 0.13880, 2.83822),
    (0, 8, "SALT2mu_chi2max8.M0DIF", 698, 0.10076, 2.17912),
    # a 5D biasCor forces fitflag_sigmb=1 -> 2, and the cut reads the same per-event
    # chi2 the fit stores, log(sigma) term included (SALT2mu.c:11445, :5622)
    (1, 16, "SALT2mu_chi2max_sigmb.M0DIF", 887, 0.07326, 1.78314),
    # the log term divides by the mu-variance of the pass before it, which the
    # pass itself stores (DO_H0marg runs the FCN at iflag=3, SALT2mu.c:5024,
    # :5660). Dividing by the LC-fit mB variance instead leaves eight events on
    # the wrong side of this ceiling, and none of the one above.
    (1, 12, "SALT2mu_chi2max_sigmb12.M0DIF", 801, 0.06262, 1.62827),
])
def test_bbc_fit_chi2max_matches_salt2mu(fitflag_sigmb, chi2max, ref, nsnfit,
                                         alpha0, beta0):
    """chi2max pre-fit cut with 5D biasCor vs SALT2mu.exe: survivors and M0DIF."""
    cfg = _des_cfg(fitflag_sigmb)
    cfg.chi2max = chi2max
    _des_check(cfg, ref, alpha0, beta0, nsnfit=nsnfit)


def test_bbc_fit_log_sigma_chi2_column():
    """The stored per-event CHI2 carries the log(sigma) term once a 5D biasCor forces
    fitflag_sigmb to 2 (SALT2mu.c:5506), so it is not mures^2/muerrsq."""
    cfg = _des_cfg(1)
    cfg.chi2max = 16.0
    _, output, _, tv, accept = run_bbc_fit(cfg)
    ref = read_fitres(f"{_DES}/SALT2mu_chi2max_sigmb.FITRES")
    rmap = dict(zip([str(c) for c in ref.get("CID")], np.asarray(ref.get("CHI2"))))
    cids = [str(c) for c in tv.name[accept]]
    d = np.array([output.chi2[i] - rmap[c] for i, c in enumerate(cids)])
    assert np.max(np.abs(d)) < 1e-2
    # without the log term the same events sit up to ~0.2 away
    assert np.max(np.abs(output.chi2 - output.mupull ** 2)) > 0.1


def test_bbc_fit_per_event_columns():
    """Per-event MU/MURES/MUERR (bias-corrected, muCOVscale-scaled) vs the
    reference FITRES."""
    cfg = _des_cfg(0)
    _, output, _, tv, accept = run_bbc_fit(cfg)
    cids = [str(c) for c in tv.name[accept]]

    ref = read_fitres(f"{_DES}/SALT2mu.FITRES")
    rc = [str(c) for c in ref.get("CID")]
    for col, arr in (("MU", output.mu), ("MURES", output.mures),
                     ("MUERR", output.muerr)):
        rmap = dict(zip(rc, np.asarray(ref.get(col))))
        d = np.array([arr[i] - rmap[c] for i, c in enumerate(cids) if c in rmap])
        assert np.max(np.abs(d)) < 5e-4, f"{col} max|d|={np.max(np.abs(d)):.2e}"
    # with no log term and no CC prior each event's CHI2 is its pull squared
    # (chi2evt = sqmures/muerrsq, SALT2mu.c:5500)
    np.testing.assert_allclose(output.chi2, output.mupull ** 2, rtol=1e-5)


def _run_and_write(cfg, tmp_path):
    result, output, zbins, tv, accept = run_bbc_fit(cfg)
    pt = {"CID": tv.name[accept], "IDSURVEY": tv.idsurvey[accept],
          "zHD": tv.zhd[accept]}
    for n, a in tv.lcpar.items():
        pt[n] = np.asarray(a)[accept]
    prefix = str(tmp_path / "JAX")
    write_m0dif(f"{prefix}.M0DIF", zbins, output, result, cfg)
    write_fitres(f"{prefix}.FITRES", output, zbins, result, cfg, passthrough=pt)
    write_cov(f"{prefix}.COV", result.cov_m0)
    write_yaml(f"{prefix}.YAML", YamlSummary(
        nevt={"DATA": result.nsnfit},
        nreject_cutwin=int((~tv.cutmask).sum()),
        nreject_biascor=int((tv.cutmask & ~accept).sum())), result, cfg)
    return prefix, result, output


def _read_cov(path):
    with open_text(path) as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    n = int(lines[0])
    vals = np.array([float(x) for x in lines[1:]])
    return vals.reshape(n, n)


def test_bbc_output_files_match_salt2mu(tmp_path):
    """The CLI writers: M0DIF (zHD/MUDIF/MUREF), FITRES BBC columns + summary,
    COV (diagonal + off-diagonal correlations vs SNANA's reduced matrix), YAML."""
    prefix, result, output = _run_and_write(_des_cfg(0), tmp_path)

    # --- M0DIF: zHD, MUDIF, MUDIFERR, MUREF, NFIT vs SNANA ---
    m0 = read_fitres(f"{prefix}.M0DIF")
    ref = read_fitres(f"{_DES}/SALT2mu.M0DIF")
    for col, tol in (("zHD", 1e-3), ("MUDIF", 1e-3), ("MUDIFERR", 1e-3),
                     ("MUREF", 1e-3), ("NFIT", 0)):
        d = np.abs(np.asarray(m0.get(col)) - np.asarray(ref.get(col)))
        assert np.max(d) <= tol + 1e-9, f"M0DIF {col} max|d|={np.max(d):.2e}"

    # --- FITRES: BBC columns by CID vs SNANA ---
    fr = read_fitres(f"{prefix}.FITRES")
    rf = read_fitres(f"{_DES}/SALT2mu.FITRES")
    cid = [str(c) for c in fr.get("CID")]

    # the SALT2mu-appended block: same columns in the same order as SNANA
    # (define_varnames_append, SALT2mu.c:24671), starting at CUTMASK
    jcols, rcols = list(fr.df.columns), list(rf.df.columns)
    assert jcols[jcols.index("CUTMASK"):] == rcols[rcols.index("CUTMASK"):]
    for col, tol in (("MU", 5e-4), ("MUMODEL", 5e-4), ("MUERR", 5e-4),
                     ("MUERR_RAW", 5e-4), ("MURES", 5e-4), ("CHI2", 1.1e-2),
                     ("biasCor_mu", 5e-3), ("biasCor_muCOVSCALE", 5e-3),
                     ("biasCor_nevt", 0), ("biasCorErr_mu", 1e-3),
                     ("IZBIN", 0), ("IDSAMPLE", 0)):
        rmap = dict(zip([str(c) for c in rf.get("CID")], np.asarray(rf.get(col))))
        jmap = dict(zip(cid, np.asarray(fr.get(col))))
        d = np.array([jmap[c] - rmap[c] for c in cid if c in rmap])
        assert np.max(np.abs(d)) <= tol + 1e-9, f"FITRES {col} max|d|={np.max(np.abs(d)):.2e}"

    # --- summary/header scalars ---
    assert abs(result.nuisance["alpha0"] - 0.18189) < 2e-5
    assert abs(result.nuisance["beta0"] - 3.79911) < 2e-5
    assert result.nsnfit == _DES_NSNFIT

    # SNANA prints only the per-bin differences `m0_nn-<M0avg>`
    with open_text(f"{_DES}/SALT2mu.FITRES") as f:
        offsets = [float(ln.split("=")[1].split()[0]) for ln in f
                   if ln.startswith("#  m0_")]
    assert np.max(np.abs((result.m0 - output.m0avg) - np.array(offsets))) < 2e-5

    # --- COV: diagonal == MUDIFERR^2, and m0 off-diagonal correlations vs the
    #     reduced correlation matrix printed in the SNANA FITRES header ---
    cov = _read_cov(f"{prefix}.COV")
    diag = np.sqrt(np.diag(cov))
    assert np.max(np.abs(diag - np.asarray(ref.get("MUDIFERR")))) < 2e-5
    with open_text(f"{_DES}/SALT2mu.FITRES") as f:
        snana_rows = [[float(x) for x in ln.split()[2:]]
                      for ln in f if ln.startswith("# m0_0")]
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    for k, row in enumerate(snana_rows):          # row = [.., alpha,beta, m0 corrs]
        for j, c_ref in enumerate(row[2:]):       # drop alpha0/beta0 columns
            assert abs(corr[k, j] - c_ref) < 0.02, f"corr[{k},{j}]"

    # --- YAML coverage fields ---
    # blank separators and the BBCFIT_RESULTS list are part of SNANA's format
    ymap = dict(ln.split(":", 1) for ln in open_text(f"{prefix}.YAML")
                if ":" in ln and not ln.startswith("  -"))
    assert int(ymap["NEVT_DATA"]) == _DES_NSNFIT
    assert int(ymap["ABORT_IF_ZERO"]) == _DES_NSNFIT


def test_bbc_fit_secant_sigmb_matches_salt2mu():
    # fitflag_sigmb=1 -> forced 2 for 5D biasCor: secant + log term, 5 iters,
    # STOP_COV0 returns the fit at sigint=0.
    # reference generated with minos=1: SALT2mu.exe's default MIGRAD-only
    # covariance puts MUDIFERR below its own floor once the secant iterates
    #, so only MUDIF/NFIT are pinned here.
    _des_check(_des_cfg(1), "SALT2mu_sigmb.M0DIF", alpha0=0.15775, beta0=3.36942)


@pytest.mark.parametrize("opt,sigmb,ref,alpha0,beta0,tol", [
    (1, 0, "SALT2mu_1dzavg.M0DIF", 0.15966, 3.50419, 2e-3),  # 1DZAVG (WGT=1)
    (2, 0, "SALT2mu_1dzwgt.M0DIF", 0.16219, 3.54689, 5e-3),  # 1DZWGT (WGT=1/muerrsq)
    # a 1D map rewrites fitflag_sigmb=2 down to 1, so no log(sigma) term (:10634)
    (1, 2, "SALT2mu_1d_nolog.M0DIF", 0.14975, 3.19200, 2e-3),
])
def test_bbc_fit_1d_biascor(opt, sigmb, ref, alpha0, beta0, tol):
    """1D z-interpolated biasCor (prepare_biasCor_zinterp): frozen per-event
    bias at alpha=beta=0, folded into mumodel. No cell map, so no event is
    rejected for lacking a biasCor and N is the full in-box sample."""
    _des_check(_des_cfg(sigmb, opt_biascor=opt, surveygroup=""), ref,
               alpha0=alpha0, beta0=beta0, nsnfit=1624, tol=tol)


def _ccprior_cfg(datafile="CONTAM_DATA.FITRES", **cc_kw):
    # CONTAM_DATA has no TYPE column (reads -9); typeIa_ccprior=999 avoids
    # force_probcc0 marking every event spec-confirmed (the default -9 would).
    cc_kw.setdefault("typeIa_ccprior", 999)
    cfg = BBCConfig(
        radiation=False,
        datafile=[f"{_DES}/{datafile}"],
        nzbin=10, zmin=0.1, zmax=1.0,
        simfile_biascor=[f"{_DES}/BIASCOR.FITRES"], opt_biascor=112,
        min_per_cell_biascor=3, snrmin_sigint_biascor=60.0,
        surveygroup_biascor=_DES_GROUP,
        simfile_ccprior=[f"{_DES}/CCPRIOR.FITRES"], **cc_kw)
    # CONTAM_* carry no SIM_TEMPLATE_INDEX, so SNANA reads them as real data and
    # would blind the reference cosmology; the references were made unblinded.
    cfg.blindflag = 0
    # the references were made with p13=1; SNANA's own parval[13] is 0
    cfg.nuisance_val["scalePCC"] = 1.0
    cfg.nuisance_float["alpha0"] = True
    cfg.nuisance_float["beta0"] = True
    cfg.nuisance_onebin.update(alpha0=False, beta0=False)
    return cfg


def _des_ccprior_cfg(datafile="CONTAM_DATA.FITRES", **cc_kw):
    return _ccprior_cfg(datafile, **cc_kw)


def test_ccprior_rejects_failed_fits():
    import pandas as pd
    from jax_salt2mu.ccprior import build_ccprior, ccprior_data
    from jax_salt2mu.run import prepare_cell
    from jax_salt2mu.samples import assign_to_samples
    from jax_salt2mu.tablevar import EVENT_TYPE_CCPRIOR, tablevar_from
    from jax_snlc_core.io.fitres_reader import FitresTable

    cell = prepare_cell(_des_ccprior_cfg(varname_pia="PROBIA"))
    table = read_fitres(f"{_DES}/CCPRIOR.FITRES")
    failed = table.df.iloc[:1].copy()
    failed.loc[:, ["x0", "x1", "c"]] = -9.
    frame = pd.concat([table.df, failed], ignore_index=True)
    tv = tablevar_from(FitresTable(frame, table.varnames, table.meta),
                       EVENT_TYPE_CCPRIOR, cell.config, cell.model)
    assert not tv.cutmask[-1] and np.isnan(tv.lcpar["mB"][-1])
    cc_idsample = assign_to_samples(tv.idsurvey, tv.field, tv.is_photoz,
                                    cell.maps.samples, cell.config)
    prior, ncell = build_ccprior(tv, cc_idsample, cell.maps, cell.model, cell.config)
    prior = ccprior_data(prior, cell.tv, cell.accept, cell.maps,
                         cell.model, cell.config)
    assert ncell == cell.spec.cc_ncell
    reference = cell.inputs.ccprior
    for name in cell.model.lcpar_names:
        np.testing.assert_array_equal(prior.cc_lcpar[name], reference.cc_lcpar[name])
    for name in ("cc_z", "cc_logmass", "cc_mumodel", "cc_fitparbias_nodes",
                 "cc_cell", "cc_valid", "data_cell", "pIa"):
        np.testing.assert_array_equal(getattr(prior, name), getattr(reference, name))


def test_bbc_fit_ccprior_beams_perevent():
    """Per-event-pIa BEAMS parity (opt_ccprior=GAUSS+UPDATE) on contaminated data.

    1775 clean Ia (PROBIA~0.95) + 120 injected CC contaminants (PROBIA~0.05) with a
    real ``varname_pIa`` classifier column -- a well-conditioned fit (unlike a flat
    force_pIa), so all 10 M0DIF bins match. The SALT2mu.exe reference needs
    ``typeIa_ccprior=999`` (else SNTYPE=-9 == its default force-sets pIa=1 and
    disables BEAMS) and ``blindflag=0`` (else the reference cosmology is blinded,
    tilting MUMODEL/M0DIF); see the fixture README.
    """
    _, output, _, tv, accept = run_bbc_fit(_des_ccprior_cfg(varname_pia="PROBIA"))

    assert int(output.nfit.sum()) == 1565
    # SNANA: alpha0=0.18370, beta0=3.83335
    ref = read_fitres(f"{_DES}/SALT2mu_contam.M0DIF")
    np.testing.assert_array_equal(output.nfit, np.asarray(ref.get("NFIT")))
    assert np.max(np.abs(output.m0dif - np.asarray(ref.get("MUDIF")))) < 2e-5

    # per-event PROBCC_BEAMS vs SNANA, matched by CID -- all bins, no exclusions
    per = read_fitres(f"{_DES}/SALT2mu_contam.FITRES")
    rp = dict(zip([str(c) for c in per.get("CID")], np.asarray(per.get("PROBCC_BEAMS"))))
    cids = [str(c) for c in tv.name[accept]]
    d = np.abs(np.array([output.probcc[i] - rp[c] for i, c in enumerate(cids)]))
    assert np.max(d) < 5e-4
    assert np.median(d) < 1e-6


def test_chi2max_fitwgt0_deweights_instead_of_rejecting():
    """chi2max(FITWGT0) keeps every event and sets MUERR to 8888.8 for the
    outliers (SALT2mu.c:3491, :6185)."""
    cfg = _des_cfg(0)
    cfg.chi2max = 8.0
    cfg.chi2max_fitwgt0 = True
    result, output, _, tv, accept = run_bbc_fit(cfg)

    ref = read_fitres(f"{_DES}/SALT2mu_chi2max_fitwgt0.M0DIF")
    assert int(output.nfit.sum()) == _DES_NSNFIT
    np.testing.assert_array_equal(output.nfit, np.asarray(ref.get("NFIT")))
    assert np.max(np.abs(output.m0dif - np.asarray(ref.get("MUDIF")))) < 2e-5

    per = read_fitres(f"{_DES}/SALT2mu_chi2max_fitwgt0.FITRES")
    rm = dict(zip([str(c) for c in per.get("CID")], np.asarray(per.get("MUERR"))))
    ref_dw = {c for c, e in rm.items() if e > 8000.0}
    jax_dw = {str(c) for c, e in zip(tv.name[accept], output.muerr) if e > 8000.0}
    assert jax_dw == ref_dw and len(jax_dw) > 0


def test_chi2max_per_survey_thresholds(bbc_biascor_variants):
    """chi2max(SURVEYLIST) cuts each listed survey at its own threshold while
    the bare key covers the rest (SALT2mu.c:20510)."""
    cfg = _cfg(0, surveygroup=f"{_DES_GROUP},SDSS(zbin=0.15:x1bin=2.0:cbin=0.2)")
    cfg.datafile = [f"{_DES}/DATA_SDSS.FITRES"]
    cfg.simfile_biascor = [f"{bbc_biascor_variants}/BIASCOR_SDSS.FITRES"]
    cfg.chi2max = 16.0
    cfg.chi2max_surveys = {"SDSS": 6.0}
    result, output, _, _, _ = run_bbc_fit(cfg)

    ref = read_fitres(f"{_DES}/SALT2mu_chi2max_survey.M0DIF")
    np.testing.assert_array_equal(output.nfit, np.asarray(ref.get("NFIT")))
    assert np.max(np.abs(output.m0dif - np.asarray(ref.get("MUDIF")))) < 2e-5


def test_u13_bayes_factor_scalepcc():
    """u13=2 floats scalePCC as the H11 Eq. 4 Bayes factor,
    PTOT_Ia = P*s / (1 + (s-1)*P) (SALT2mu.c:5530-5535), reported as
    scalePIa."""
    cfg = _des_ccprior_cfg(varname_pia="PROBIA")
    cfg.nuisance_float["scalePCC"] = True
    cfg._u_flags = {"u13": 2}
    from jax_salt2mu.inputs import build_config
    cfg = build_config({"prefix": "renamed"}, cfg)
    assert cfg._u_flags == {"u13": 2}
    result, output, _, tv, accept = run_bbc_fit(cfg)

    ref = read_fitres(f"{_DES}/SALT2mu_u13_bayes.M0DIF")
    np.testing.assert_array_equal(output.nfit, np.asarray(ref.get("NFIT")))
    assert np.max(np.abs(output.m0dif - np.asarray(ref.get("MUDIF")))) < 2e-5
    # SNANA: scalePIa = 6.68657 +- 1.39222
    assert abs(result.nuisance["scalePCC"] - 6.68657) < 2e-3

    per = read_fitres(f"{_DES}/SALT2mu_u13_bayes.FITRES")
    rp = dict(zip([str(c) for c in per.get("CID")], np.asarray(per.get("PROBCC_BEAMS"))))
    cids = [str(c) for c in tv.name[accept]]
    d = np.abs(np.array([output.probcc[i] - rp[c] for i, c in enumerate(cids)]))
    assert np.max(d) < 5e-4
    assert np.median(d) < 1e-6

    # BEAMS replaces the Ia chi2 with -2ln(Prob_Ia+Prob_CC) and adds MUERR_RENORM,
    # which rescales muerr so sum(1/err^2) per z bin reproduces 1/M0DIFERR^2.
    rmap = dict(zip([str(c) for c in per.get("CID")], np.asarray(per.get("CHI2_BEAMS"))))
    d = np.abs(np.array([output.chi2[i] - rmap[c] for i, c in enumerate(cids)]))
    assert np.max(d) <= 1e-2 + 1e-9, f"CHI2_BEAMS max|d|={np.max(d):.2e}"   # %.2f grid
    assert np.median(d) <= 5e-3

    # spans 0.05 to 447 (it carries 1/sqrt(pIa)), so the bound is relative, floored
    # at the reference's %.5f print grid
    rmap = dict(zip([str(c) for c in per.get("CID")], np.asarray(per.get("MUERR_RENORM"))))
    ref = np.array([rmap[c] for c in cids])
    d = np.abs(np.asarray(output.muerr_renorm) - ref)
    assert np.all(d <= np.maximum(2e-3 * np.abs(ref), 2.5e-5)), \
        f"MUERR_RENORM max rel|d|={np.max(d / np.abs(ref)):.2e}"


def test_biascor_unmatched_idsample_is_cut(bbc_biascor_variants):
    """BiasCor events matching no data IDSAMPLE (photo-z rows against
    spec-z-only data) fail the IDSAMPLE cut (SALT2mu.c:18957-18958), so
    sigInt and the maps are built from the matched half only."""
    cfg = _des_cfg(0)
    cfg.simfile_biascor = [f"{bbc_biascor_variants}/BIASCOR_PHOTOZ.FITRES"]
    cfg.zspec_errmax_idsample = 0.05
    _des_check(cfg, "SALT2mu_biascor_stray.M0DIF",
               alpha0=0.18271, beta0=3.80745, nsnfit=1472)


@pytest.mark.parametrize("key,value,ref,nsnfit,alpha0,beta0", [
    # exact field names: the overlap C3+X3 and E2 match no group, carry
    # FIELDGROUP NONE, and join the survey's first field sample through the
    # SURVEYGROUP match (get_IDSAMPLE, SALT2mu.c:11215-11216)
    ("fieldlist_biascor", "C3,X3", "SALT2mu_fieldlist", 1474, 0.17663, 3.78604),
    # |C3| exact and X3 by substring: C3+X3 goes to X3, E2 to DES(C3)
    ("fieldgroup_biascor", "|C3|,X3", "SALT2mu_fieldgroup_exact", 1473,
     0.18154, 3.79516),
])
def test_bbc_fit_field_samples_match_salt2mu(bbc_biascor_variants, key, value, ref, nsnfit, alpha0, beta0):
    """Field samples on FIELD cycling C3, X3, C3+X3, E2: every event's
    IDSAMPLE is SNANA's, and the two-sample fit follows. The field groups
    carry no options, so each sample's binning is the survey group's."""
    cfg = _des_cfg(0)
    cfg.datafile = [f"{_DES}/DATA_FIELD.FITRES"]
    cfg.simfile_biascor = [f"{bbc_biascor_variants}/BIASCOR_FIELD.FITRES"]
    setattr(cfg, key, value)
    _, output, _, tv, accept = run_bbc_fit(cfg)
    per = read_fitres(f"{_DES}/{ref}.FITRES")
    rmap = dict(zip([str(c) for c in per.get("CID")], np.asarray(per.get("IDSAMPLE"))))
    cids = [str(c) for c in tv.name[accept]]
    jax_ids = {c: int(output.idsample[i]) for i, c in enumerate(cids)}
    assert {c: jax_ids[c] for c in rmap} == {c: int(v) for c, v in rmap.items()}
    _des_check(cfg, f"{ref}.M0DIF", alpha0=alpha0, beta0=beta0, nsnfit=nsnfit)


def _grab_contam(path):
    """The contamination block's (value, err) pairs from a written output."""
    text = open_text(path).read()
    vals = {}
    for key in ("SUM_PROBIa", "SUM_PROBCC", "CONTAM_DATA", "CONTAM_TRUE"):
        m = re.search(
            rf"{key}[:=\s]+(-?(?:[0-9.eE+-]+|nan|inf))"
            rf"(?:\s+(-?(?:[0-9.eE+-]+|nan|inf)))?", text)
        if m:
            vals[key] = (float(m.group(1)),
                         float(m.group(2)) if m.group(2) else None)
    return vals


def test_contam_true_lines_match_salt2mu(tmp_path):
    """Sim data under a CC prior with a spec-typed subset: SUM_PROBIa drops
    the spec-confirmed Ia and CONTAM_TRUE appears in the fit summary
    (write_fitres_misc, SALT2mu.c:24968-24986), and the YAML carries
    CONTAM_DATA plus CONTAM_TRUE with its Poisson error (write_yaml_info,
    :23775-23801)."""
    cfg = _des_ccprior_cfg(datafile="CONTAM_SIMTYPE.FITRES",
                           varname_pia="PROBIA", typeIa_ccprior=1)
    result, output, zbins, tv, accept = run_bbc_fit(cfg)
    assert result.nsnfit == 1565            # same fitted sample as SNANA
    assert output.nsnspec_ia > 0            # the spec subset is populated

    write_fitres(str(tmp_path / "J.FITRES"), output, zbins, result, cfg,
                 passthrough={"CID": tv.name[accept]})
    write_yaml(str(tmp_path / "J.YAML"), output.yaml, result, cfg)

    ref = _grab_contam(f"{_DES}/SALT2mu_contam_true.FITRES")
    got = _grab_contam(str(tmp_path / "J.FITRES"))
    assert set(got) == {"SUM_PROBIa", "SUM_PROBCC", "CONTAM_DATA",
                        "CONTAM_TRUE"}
    # the probability sums track the fit; without the spec subtraction
    # SUM_PROBIa would be ~437 higher
    assert abs(got["SUM_PROBIa"][0] - ref["SUM_PROBIa"][0]) < 0.05
    assert abs(got["SUM_PROBCC"][0] - ref["SUM_PROBCC"][0]) < 0.05
    assert abs(got["CONTAM_DATA"][0] - ref["CONTAM_DATA"][0]) < 5e-5
    # pure integer counts: exact
    assert got["CONTAM_TRUE"][0] == ref["CONTAM_TRUE"][0]

    ref_y = _grab_contam(f"{_DES}/SALT2mu_contam_true.YAML")
    got_y = _grab_contam(str(tmp_path / "J.YAML"))
    assert abs(got_y["CONTAM_DATA"][0] - ref_y["CONTAM_DATA"][0]) < 5e-5
    assert got_y["CONTAM_TRUE"] == ref_y["CONTAM_TRUE"]


def test_contam_all_spec_completes_with_nan(tmp_path):
    """With every TYPE spec-confirmed the photo-classified subset is empty:
    the run completes and the ratios print as SALT2mu.exe's C division does,
    nan for 0/0 (CONTAM_DATA) and inf for true CC over zero (CONTAM_TRUE)."""
    cfg = _des_ccprior_cfg(datafile="CONTAM_SIMTYPE.FITRES",
                           varname_pia="PROBIA", typeIa_ccprior=1)
    cfg.type_list_probcc0 = [0, 101, 102]
    result, output, zbins, tv, accept = run_bbc_fit(cfg)
    assert output.nsnspec_ia == result.nsnfit    # the subset really is empty

    write_fitres(str(tmp_path / "J.FITRES"), output, zbins, result, cfg,
                 passthrough={"CID": tv.name[accept]})
    write_yaml(str(tmp_path / "J.YAML"), output.yaml, result, cfg)

    for got_path, ref_path in (
            (str(tmp_path / "J.FITRES"), f"{_DES}/SALT2mu_contam_allspec.FITRES"),
            (str(tmp_path / "J.YAML"), f"{_DES}/SALT2mu_contam_allspec.YAML")):
        got, ref = _grab_contam(got_path), _grab_contam(ref_path)
        assert np.isnan(got["CONTAM_DATA"][0]) and np.isnan(ref["CONTAM_DATA"][0])
        assert got["CONTAM_TRUE"][0] == ref["CONTAM_TRUE"][0] == np.inf


@pytest.mark.parametrize("datafile,ref,cc_kw,forced", [
    # the absent-TYPE trap: CONTAM_DATA has no TYPE column, so TYPE reads -9 ==
    # typeIa_ccprior's own default and every event is forced, disabling BEAMS
    ("CONTAM_DATA.FITRES", "SALT2mu_contam_specConf",
     dict(typeIa_ccprior=-9), "all"),
    # the same all-forced result reached by survey instead of type
    ("CONTAM_MIXED.FITRES", "SALT2mu_contam_idsurvey",
     dict(idsurvey_list_probcc0=[10]), "all"),
    # a genuine mix: TYPE=101 via typeIa_ccprior and TYPE=102 via type_list_probcc0
    # are forced, the TYPE=1 Ia keep their classifier value
    ("CONTAM_MIXED.FITRES", "SALT2mu_contam_mixed",
     dict(typeIa_ccprior=101, type_list_probcc0=[102]), "contaminants"),
])
def testforce_probcc0_mask_matches_salt2mu(datafile, ref, cc_kw, forced):
    """Which events force_probcc0 pins to pIa=1, per event against SALT2mu.exe.

    The mask is a pure function of SNTYPE/IDSURVEY and the three keys
    (SALT2mu.c:22772), so this reads it directly rather than through a fit: on this
    sample the all-forced configurations put every contaminant in at full weight and
    drive beta into its bound, and the resulting fit says nothing about the mask.
    SNANA exposes the mask as PROBCC_BEAMS = 0 exactly.
    """

    cfg = resolve(_des_ccprior_cfg(datafile, varname_pia="PROBIA", **cc_kw), SALTModel)
    tv = read_tablevar(cfg.datafile, EVENT_TYPE_DATA, cfg, SALTModel)
    mask = force_probcc0(tv, cfg)
    got = {str(c) for c, f in zip(tv.name, mask) if f}

    per = read_fitres(f"{_DES}/{ref}.FITRES")
    cids = np.asarray(per.get("CID")).astype(str)
    want = {c for c, p in zip(cids, np.asarray(per.get("PROBCC_BEAMS"))) if p == 0.0}

    assert want, f"{ref}: SNANA forced nothing, so the config tests nothing"
    if forced == "all":
        assert len(want) == len(cids)
    else:
        assert all(int(c) >= 100000 for c in want)      # only the contaminants
    # every event SNANA forced is one jax forces; jax may also flag events SNANA
    # cut before writing the FITRES, so compare on the reference's own rows
    assert want <= got, f"{ref}: jax missed {sorted(want - got)[:5]}"
    kept = set(cids) - want
    assert not (kept & got), f"{ref}: jax forced {sorted(kept & got)[:5]} SNANA did not"


def test_bbc_fit_ccprior_force_pia_runs():
    """The constant-pIa wiring (force_pia instead of a classifier column) drives the
    same BEAMS loss; smoke-check it converges to a finite fit on the same data."""
    result, output, _, _, _ = run_bbc_fit(_des_ccprior_cfg(force_pia=0.9))
    assert int(output.nfit.sum()) == 1565
    assert np.all(np.isfinite(output.m0dif))
    assert 0.0 < result.nuisance["alpha0"] < 0.5


def test_force_pia_spec_confirmed_count_needs_varname_pia():
    """NSNSPEC_IA counts events forced to pIa=1, which SNANA does only under a
    varname_pIa (REQUIRE_pIa, SALT2mu.c:9301). With force_pIa alone and a data
    table whose SNTYPE (-9, no TYPE column) equals typeIa_ccprior's default,
    nothing is spec-confirmed, so the SUM_PROBIa / CONTAM_DATA header values
    stay a probability sum and its ratio."""
    _, output, _, _, _ = run_bbc_fit(_des_ccprior_cfg(force_pia=0.9,
                                                      typeIa_ccprior=-9))
    assert output.nsnspec_ia == 0
    assert output.sum_probia >= 0.0
    assert 0.0 <= output.contam_data <= 1.0


def test_varname_pia_names_an_absent_column():
    """A data table without the varname_pIa column aborts naming it, unless a
    probcc0 list forces every event to pIa=1 (SALT2mu.c:9516-9557)."""
    cfg = resolve(_des_ccprior_cfg(varname_pia="NOSUCH"), SALTModel)
    with pytest.raises(SNANAError, match="Missing column varname_pIa='NOSUCH'"):
        read_tablevar(cfg.datafile, EVENT_TYPE_DATA, cfg, SALTModel)
    cfg = resolve(_des_ccprior_cfg(varname_pia="NOSUCH", idsurvey_list_probcc0=[10]),
                  SALTModel)
    tv = read_tablevar(cfg.datafile, EVENT_TYPE_DATA, cfg, SALTModel)
    assert tv.pIa is None


def test_no_fittable_zbin_aborts():
    """Zero events in every z bin abort as setup_zbins_fit does
    (SALT2mu.c:3133), before anything downstream sees an empty sample."""
    cfg = BBCConfig(datafile=[f"{_DES}/DATA.FITRES"], nzbin=5, zmin=2.0, zmax=3.0)
    with pytest.raises(SNANAError, match="no z bin with at least 1 SN"):
        run_bbc_fit(cfg)


def test_bbc_fit_h11_prob_matches_salt2mu():
    """Per-event H11 (Hlozek 2011) CC probability vs SALT2mu.exe (prob_CCprior_H11,
    SALT2mu.c:18311): the analytic prior models the CC population as a poly(z) Gaussian,
    dPdmu_CC = (1/sqrt(2pi)/sigCC) exp(-(dmu-CCpoly)^2 / 2 sqsigCC). Validated against
    SNANA's instrumented per-event values at the fit's starting polynomial (the two codes
    compute an identical objective). The end-to-end M0DIF is checked only for sane running
    (test_bbc_fit_h11_runs) -- a strong joint-fit parity test needs a fixture that
    constrains the 6-param polynomial, which the available contaminated fixtures don't."""
    path = f"{_DES}/SALT2mu_h11_prob.txt"
    with open_text(path) as fh:
        header = fh.readline()
    mucc = [float(x) for x in re.search(r"mucc=\[([^\]]+)\]", header).group(1).split(",")]
    sigcc = [float(x) for x in re.search(r"sigcc=\[([^\]]+)\]", header).group(1).split(",")]
    z, mures, muerrsq_raw, dpdmu_ref = np.loadtxt(path, skiprows=2, unpack=True)

    class _H:
        pass
    h = _H()
    h.z, h.muerrsq_raw = jnp.asarray(z), jnp.asarray(muerrsq_raw)
    nuis = {"H11mucc0": mucc[0], "H11mucc1": mucc[1], "H11mucc2": mucc[2],
            "H11sigcc0": sigcc[0], "H11sigcc1": sigcc[1], "H11sigcc2": sigcc[2]}
    dpdmu_jax = np.asarray(h11_dPdmu(h, nuis, jnp.asarray(mures)))
    np.testing.assert_allclose(dpdmu_jax, dpdmu_ref, rtol=1e-4)


def test_bbc_fit_h11_runs():
    """H11 analytic CC prior (simfile_ccprior=H11) runs end-to-end and yields finite,
    sane output with the CC prior active (scalePCC floated). The per-event CC probability
    is parity-tested against SALT2mu.exe above; M0DIF isn't asserted here because no
    available fixture constrains the 6-param CC polynomial tightly enough for a clean
    joint-fit comparison."""
    cfg = BBCConfig(
        datafile=[f"{_DES}/CONTAM_DATA.FITRES"],
        nzbin=10, zmin=0.1, zmax=1.0,
        simfile_biascor=[f"{_DES}/BIASCOR.FITRES"], opt_biascor=112,
        min_per_cell_biascor=3, snrmin_sigint_biascor=60.0,
        surveygroup_biascor=_DES_GROUP,
        use_h11=True, varname_pia="PROBIA", typeIa_ccprior=999)
    cfg.nuisance_float.update(alpha0=True, beta0=True)
    cfg.nuisance_onebin.update(alpha0=False, beta0=False)
    # u13: without it scalePCC is held at its parval of 0, which turns the prior
    # off in SALT2mu as well (prep_input, SALT2mu.c:22117)
    cfg.nuisance_float["scalePCC"] = True
    # A broad CC population starts around a 1.6-mag offset and 1.5-mag width,
    # decreasing with redshift. The generic near-zero width starting point
    # sends this smoke-test fixture to coefficient bounds before convergence.
    # All six polynomial coefficients remain free, including quadratic terms.
    cfg.nuisance_val.update(alpha0=.18, beta0=3.8, scalePCC=.15,
                            H11mucc0=1.6, H11mucc1=-.8, H11mucc2=0.,
                            H11sigcc0=1.5, H11sigcc1=-1.1, H11sigcc2=0.)
    result, output, _, _, _ = run_bbc_fit(cfg)
    assert result.max_projected_gradient < 1e-3
    assert output.probcc is not None
    assert int(output.nfit.sum()) == 1565
    assert np.all(np.isfinite(output.m0dif))
    # the fit has to move: a Prob_SUM of ~1e-183 used to overflow the Hessian and
    # leave every parameter at its start with M0DIF zero, which "finite" accepted
    assert np.all(np.isfinite(result.m0err))
    assert np.any(np.abs(output.m0dif) > 1e-6)
    assert abs(result.nuisance["alpha0"] - 0.14) > 1e-6
    assert 0.0 < result.nuisance["alpha0"] < 0.5


def test_bbc_fit_4d_logmass(bbc_biascor_variants):
    """4D (DES5YR-style host-mass biasCor): cell (z, logmass, x1, c). BIASCOR carries
    an injected uniform logmass [8,12]; nbin_logmass=2 + interp_biascor_logmass turn
    it into a real cell axis (M0DIF shifts ~8% vs logmass-off). Data is DATA_MASS.
    alpha/beta and per-z-bin M0DIF vs SALT2mu.exe."""
    cfg = BBCConfig(
        radiation=False,
        datafile=[f"{_DES}/DATA_MASS.FITRES"],
        nzbin=10, zmin=0.1, zmax=1.0,
        simfile_biascor=[f"{bbc_biascor_variants}/BIASCOR_MASS.FITRES"], opt_biascor=112,
        min_per_cell_biascor=3, snrmin_sigint_biascor=60.0,
        surveygroup_biascor=_DES_GROUP,
        nbin_logmass=2, interp_biascor_logmass=1,
        logmass_min=8.0, logmass_max=12.0)
    cfg.nuisance_float.update(alpha0=True, beta0=True)
    cfg.nuisance_onebin.update(alpha0=False, beta0=False)
    result, output, _, _, _ = run_bbc_fit(cfg)

    assert int(output.nfit.sum()) == 1480
    assert abs(result.nuisance["alpha0"] - 0.18047) < 2e-5
    assert abs(result.nuisance["beta0"] - 3.80147) < 2e-4

    ref = read_fitres(f"{_DES}/SALT2mu_4d.M0DIF")
    np.testing.assert_array_equal(output.nfit, np.asarray(ref.get("NFIT")))
    assert np.max(np.abs(output.m0dif - np.asarray(ref.get("MUDIF")))) < 2e-5


def test_dmubin_ccprior_matches_salt2mu():
    """dmubin_ccprior sets the CC-prior dmu bin width. The window is fixed at
    [-3,+4] with nbin=int(7/width) (SALT2mu.c:17290), so 1.5 stops the axis at +3.0
    and the CC events above it stop feeding DMUAVG/DMURMS (:17449).

    The effect is only 3.6e-4 in MUDIF, so this asserts against the default
    reference too -- otherwise a tolerance loose enough to pass would also pass if
    jax ignored the key entirely.
    """
    cfg = _des_ccprior_cfg(varname_pia="PROBIA")
    cfg.dmubin_ccprior = 1.5
    _, output, _, _, _ = run_bbc_fit(cfg)

    ref = np.asarray(read_fitres(f"{_DES}/SALT2mu_dmubin.M0DIF").get("MUDIF"))
    assert np.max(np.abs(output.m0dif - ref)) < 2e-5

    default = np.asarray(read_fitres(f"{_DES}/SALT2mu_contam.M0DIF").get("MUDIF"))
    assert np.max(np.abs(ref - default)) > 2e-4, "the key would be untested"


def test_um0_boxes_m0_in_a_narrow_window():
    """uM0=0 narrows each per-z-bin M0 to M0_DEFAULT +/-0.001 (SALT2mu.c:2634) but
    leaves it floating -- ISFLOAT_z depends only on min_per_zbin (:3101) -- so the
    bins settle on the walls of the box rather than at M0_DEFAULT.

    alpha/beta are not asserted: boxed 100x tighter than their own step, the M0
    parameters leave the objective near-degenerate, and jax settles at
    beta0=3.71498 against SALT2mu's 3.70940. M0DIF and NFIT are unaffected.
    """
    cfg = _des_cfg(0)
    cfg.uM0 = M0FITFLAG_CONSTANT
    _, output, _, _, _ = run_bbc_fit(cfg)

    ref = read_fitres(f"{_DES}/SALT2mu_um0_fixed.M0DIF")
    assert int(output.nfit.sum()) == _DES_NSNFIT
    np.testing.assert_array_equal(output.nfit, np.asarray(ref.get("NFIT")))
    assert np.max(np.abs(output.m0dif - np.asarray(ref.get("MUDIF")))) < 2e-5
    assert np.max(np.abs(output.muref - np.asarray(ref.get("MUREF")))) < 1e-4


@pytest.mark.parametrize("key,value,ref,nsnfit,beta0", [
    ("cmin", -0.2, "SALT2mu_cmin", 1433, 3.88232),
    ("cmax", 0.2, "SALT2mu_cmax", 1445, 3.80361),
    ("x1min", -2.0, "SALT2mu_x1min", 1339, 3.82427),
    ("x1max", 2.0, "SALT2mu_x1max", 1288, 3.90893),
])
def test_cut_shorthands_set_the_biascor_binning(key, value, ref, nsnfit, beta0):
    """cmin/cmax/x1min/x1max set the biasCor cell ranges, not only the cut
    (SALT2mu.c:6987, :17297), so narrowing one moves which cell an event lands in.

    Taking the ranges from default_cuts() instead selected the right events and
    binned them over the unnarrowed window: cmin was 4.4e-3 out in MUDIF and cmax
    3.1e-3, on keys already advertised as supported.
    """
    cfg = _des_cfg(0)
    is_min = key.endswith("min")
    var = "c" if key.startswith("c") else "x1"
    cfg.lcpar_range[var] = (value, None) if is_min else (None, value)

    result, output, _, _, _ = run_bbc_fit(cfg)
    r = read_fitres(f"{_DES}/{ref}.M0DIF")
    assert int(output.nfit.sum()) == nsnfit
    np.testing.assert_array_equal(output.nfit, np.asarray(r.get("NFIT")))
    assert abs(result.nuisance["beta0"] - beta0) < 1e-4
    assert np.max(np.abs(output.m0dif - np.asarray(r.get("MUDIF")))) < 2e-5


def test_biascor_grid_cut_matches_salt2mu():
    """Windows wider than the biasCor grid (x1 +-4, c -0.3..0.5) put events
    between window and grid, where outside_biasCor_grid rejects biasCor
    events per axis (SALT2mu.c:19239) and the sigInt subset drops only the
    z bit (:14326-14335)."""
    cfg = _des_cfg(0)
    for var, lo, hi in (("x1", -6.0, 6.0), ("c", -1.0, 1.0)):
        cfg.lcpar_range[var] = (lo, hi)

    result, output, _, _, _ = run_bbc_fit(cfg)
    r = read_fitres(f"{_DES}/SALT2mu_gridcut.M0DIF")
    ref_fitres = read_fitres(f"{_DES}/SALT2mu_gridcut.FITRES")
    assert int(output.nfit.sum()) == ref_fitres.df.shape[0]
    np.testing.assert_array_equal(output.nfit, np.asarray(r.get("NFIT")))
    assert np.max(np.abs(output.m0dif - np.asarray(r.get("MUDIF")))) < 2e-5


@pytest.mark.parametrize("key,value,ref,nsnfit,beta0", [
    ("x1min", -2.0, "SALT2mu_x1min_cc", 1402, 3.85375),
    ("x1max", 2.0, "SALT2mu_x1max_cc", 1353, 3.97643),
])
def test_x1_shorthand_sets_the_ccprior_shape_axis(key, value, ref, nsnfit, beta0):
    """The x1 pair reaches the binning only through the CC-prior shape axis
    (SALT2mu.c:17303) -- muCOVscale collapses its shape axis to one bin and takes
    the range from biascor_lcpar_binning. Without a CC prior the x1 keys look
    correct whether or not lcpar_windows is wired up, which is why this exists:
    the bug moved MUDIF by 1.1e-3 (x1min) and 1.8e-3 (x1max) only along this path.
    """
    cfg = _des_ccprior_cfg(varname_pia="PROBIA")
    cfg.ccprior_bins["x1"] = 3
    is_min = key.endswith("min")
    cfg.lcpar_range["x1"] = (value, None) if is_min else (None, value)

    result, output, _, _, _ = run_bbc_fit(cfg)
    r = read_fitres(f"{_DES}/{ref}.M0DIF")
    assert int(output.nfit.sum()) == nsnfit
    np.testing.assert_array_equal(output.nfit, np.asarray(r.get("NFIT")))
    assert abs(result.nuisance["beta0"] - beta0) < 1e-4
    assert np.max(np.abs(output.m0dif - np.asarray(r.get("MUDIF")))) < 2e-5


def test_lensing_zpar_matches_salt2mu():
    """lensing_zpar is added inside fcn_muerrsq (SALT2mu.c:6224), so it scales with
    muCOVscale on the data side (:5468) and enters every biasCor muerrsq. Having it
    outside the scale, and absent from the biasCor, put MUDIF 2.4e-3 out."""
    cfg = _des_cfg(0)
    cfg.lensing_zpar = 0.055
    # the widened errors let one more event past the biasCor cut, in both codes
    _des_check(cfg, "SALT2mu_lensing.M0DIF", alpha0=0.18037, beta0=3.75367,
               nsnfit=1474)


@pytest.mark.parametrize("setup,match", [
    # init_sigInt_biasCor_SNRCUT aborts below 50 high-SNR events in a node (:14220)
    (lambda c: setattr(c, "snrmin_sigint_biascor", 1.0e6),
     "sigInt needs at least 50"),
    # a bin wider than its range leaves no bins; SNANA aborts in IBINFUN instead
    (lambda c: setattr(c, "surveygroup_biascor",
                       "DES(zbin=0.15:x1bin=100.0:cbin=0.2)"),
     "leaves no bins"),
    # varname_z is a required read-variable
    (lambda c: setattr(c, "varname_z", "NOSUCHCOLUMN"),
     "is not in the input table"),
])
def test_bbc_fit_aborts_where_salt2mu_aborts(setup, match):
    """Configs SALT2mu.exe rejects outright; each of these used to reach a bare
    numpy/JAX traceback instead of an abort."""
    cfg = _des_cfg(0)
    setup(cfg)
    with pytest.raises(SNANAError, match=match):
        run_bbc_fit(cfg)


@pytest.mark.parametrize("biascor,force_pia,datafile,match", [
    # every CC event fails the valid-biasCor cut without a cell map, which empties
    # the table and aborts the job (print_eventStats, SALT2mu.c:18671)
    (0, None, "CONTAM_DATA.FITRES", "all CCPRIOR events fail cuts"),
    # prepare_CCprior rejects a 1D biasCor outright
    (1, None, "CONTAM_DATA.FITRES", "1D biasCor with the CC likelihood"),
    # force_pIa=perfect reads SIM_TEMPLATE_INDEX, which real data has not got (:9263)
    (112, -1.0, "DATA_REAL.FITRES", "only on simulated data"),
])
def test_bbc_ccprior_aborts_where_salt2mu_aborts(biascor, force_pia, datafile, match):
    cfg = (_des_cfg(0, opt_biascor=biascor, surveygroup="" if biascor == 1 else None)
           if biascor else
           BBCConfig(radiation=False, datafile=[f"{_DES}/{datafile}"],
                     nzbin=10, zmin=0.1, zmax=1.0))
    cfg.datafile = [f"{_DES}/{datafile}"]
    cfg.simfile_ccprior = [f"{_DES}/CCPRIOR.FITRES"]
    cfg.typeIa_ccprior = 999
    # a fixed scalePCC of 0 would turn the CC prior off before any of these
    # aborts could fire, in SALT2mu as well (prep_input, SALT2mu.c:22117)
    cfg.nuisance_val["scalePCC"] = 1.0
    if force_pia is None:
        cfg.varname_pia = "PROBIA"
    else:
        # DATA_REAL has no PROBIA column; naming one would abort on that
        # first, as SALT2mu's store_input_varnames does (:6894 before :6911)
        cfg.force_pia = force_pia
    with pytest.raises(SNANAError, match=match):
        run_bbc_fit(cfg)


def test_bbc_fit_diagnostic_muerr_columns():
    """MUERR_RAW is the LC-fit variance alone and MUERR_VPEC the peculiar-velocity
    term alone: neither carries sigint or muCOVscale (SALT2mu.c:5642). BASE has
    VPECERR=0 and a secant that lands on sigint=0, so vpecerr plus chi2max is what
    makes both terms non-zero at once."""
    cfg = _des_cfg(1)
    cfg.zpecerr = 250.0 / LIGHT_KM
    cfg.chi2max = 16.0
    result, output, _, tv, accept = run_bbc_fit(cfg)
    # both terms have to be live or the columns collapse onto MUERR and prove nothing
    assert output.muerr_vpec.max() > 0.0
    assert result.sigint > 0.0

    ref = read_fitres(f"{_DES}/SALT2mu_vpec_sigmb.FITRES")
    rmap = {c: i for i, c in enumerate(str(x) for x in ref.get("CID"))}
    idx = np.array([rmap[str(c)] for c in tv.name[accept]])
    for col, got in (("MUERR", output.muerr), ("MUERR_RAW", output.muerr_raw),
                     ("MUERR_VPEC", output.muerr_vpec)):
        d = np.asarray(got) - np.asarray(ref.get(col))[idx]
        assert np.max(np.abs(d)) < 6e-5, f"{col} max|d|={np.max(np.abs(d)):.2e}"


def test_a_fixed_scalepcc_of_zero_turns_the_cc_prior_off():
    """prep_input drops the CC prior when scalePCC is held at its parval of 0
    (SALT2mu.c:22117), which is what an input naming simfile_ccprior but no p13 or
    u13 asks for. The appended block is then the plain one: CHI2 rather than
    CHI2_BEAMS, and no PROBCC_BEAMS or MUERR_RENORM.
    """
    cfg = _des_ccprior_cfg(varname_pia="PROBIA")
    del cfg.nuisance_val["scalePCC"]        # the helper sets p13=1 for the rest
    _, output, _, _, _ = run_bbc_fit(cfg)
    assert output.probcc is None

    ref = read_fitres(f"{_DES}/SALT2mu_ccprior_scalepcc0.FITRES", as_text=True)
    appended = ref.varnames[ref.varnames.index("CUTMASK"):]
    assert "CHI2" in appended and "CHI2_BEAMS" not in appended
    assert "PROBCC_BEAMS" not in appended and "MUERR_RENORM" not in appended
    r = read_fitres(f"{_DES}/SALT2mu_ccprior_scalepcc0.M0DIF")
    np.testing.assert_array_equal(output.nfit, np.asarray(r.get("NFIT")))

    # dropping the key and having it dropped are the same fit. MUDIF is not
    # compared against SNANA here: with the contaminants at full weight this
    # config sits on beta0's upper bound in both programs (the reference reads
    # 6.0000 +- 0.0000), and neither minimiser's approach to a bound is the
    # other's.
    dropped = _des_ccprior_cfg(varname_pia="PROBIA")
    del dropped.nuisance_val["scalePCC"]
    dropped.simfile_ccprior = []
    _, plain, _, _, _ = run_bbc_fit(dropped)
    np.testing.assert_array_equal(output.m0dif, plain.m0dif)
