"""End-to-end BBC fit parity vs SALT2mu.exe for a BayeSN (MU-based) biasCor.

Runs run_bbc_fit on a jax-fit BayeSN data sample with an independent jax-fit
biasCor sim (MASK_BIASCOR_MU, opt_biascor=144) and checks the per-z-bin M0DIF
and NFIT against the SALT2mu.exe reference. See
tests/fixtures/snana/bbc_des_bayesn_data/README.md.
"""


import numpy as np
import pytest

from jax_snlc_core.io.fitres_reader import read_fitres
from jax_snlc_core.utils.error_handling import SNANAError
from jax_salt2mu.config import BBCConfig
from jax_salt2mu.run import run_bbc_fit

_DATA_DIR = "tests/fixtures/snana/bbc_des_bayesn_data"
_BC_DIR = "tests/fixtures/snana/bbc_des_bayesn_biascor"
_M0DIF = f"{_DATA_DIR}/SALT2mu_bayesn.M0DIF"
_NSNFIT = 381


def _cfg(opt_biascor=144):
    return BBCConfig(
        radiation=False,
        datafile=[f"{_DATA_DIR}/FITRES.TEXT"], model_lcfit="BAYESN",
        nzbin=4, zmin=0.05, zmax=0.21,
        simfile_biascor=[f"{_BC_DIR}/FITRES.TEXT.gz"], opt_biascor=opt_biascor,
        sigint_biascor=0.10, nbin_logmass=1, snrmin_sigint_biascor=6.0)


def test_bbc_bayesn_matches_salt2mu():
    result, output, _, _, _ = run_bbc_fit(_cfg())

    assert int(output.nfit.sum()) == _NSNFIT

    ref = read_fitres(_M0DIF)
    np.testing.assert_array_equal(output.nfit, np.asarray(ref.get("NFIT")))
    assert np.max(np.abs(output.m0dif - np.asarray(ref.get("MUDIF")))) < 1e-4


def test_bbc_bayesn_1dzwgt():
    """1DZWGT weights the z-interpolated bias by 1/muerrsq -- the only 1D path that
    touches the mu-variance code, on a model with one LC parameter rather than three."""
    _, output, _, _, _ = run_bbc_fit(_cfg(opt_biascor=2))

    ref = read_fitres(f"{_DATA_DIR}/SALT2mu_bayesn_1dzwgt.M0DIF")
    assert int(output.nfit.sum()) == 460
    np.testing.assert_array_equal(output.nfit, np.asarray(ref.get("NFIT")))
    assert np.max(np.abs(output.m0dif - np.asarray(ref.get("MUDIF")))) < 1e-4


def test_bbc_bayesn_rejects_mucovadd():
    """muCOVADD tunes the error per alpha/beta node, so SALT2mu refuses it outright
    for a model that has no standardisation coefficients (SALT2mu.c:11332)."""
    with pytest.raises(SNANAError, match="standardisation coefficients"):
        run_bbc_fit(_cfg(opt_biascor=144 | 4096))


def test_bbc_bayesn_chi2max():
    """applyCut_chi2max on a model with no fitted nuisance parameters: the cut runs
    off whatever exec_mnparm clamped into bounds (SALT2mu.c:2621), which for BayeSN
    is nothing at all."""
    cfg = _cfg()
    cfg.chi2max = 0.5
    _, output, _, _, _ = run_bbc_fit(cfg)

    ref = read_fitres(f"{_DATA_DIR}/SALT2mu_bayesn_chi2max.M0DIF")
    assert int(output.nfit.sum()) == 363
    np.testing.assert_array_equal(output.nfit, np.asarray(ref.get("NFIT")))
    assert np.max(np.abs(output.m0dif - np.asarray(ref.get("MUDIF")))) < 1e-4


@pytest.mark.parametrize("var,lo,hi,ref,nsnfit", [
    ("AV", None, 0.4, "SALT2mu_bayesn_avmax", 363),
    ("THETA", -1.0, None, "SALT2mu_bayesn_thetamin", 356),
])
def test_cut_shorthands_set_the_biascor_binning(var, lo, hi, ref, nsnfit):
    """avmin/avmax/thetamin/thetamax are BayeSN's half of the range inputs that set
    both the cut and the biasCor cell ranges (SALT2mu.c:18816-18823, :18876-18880).

    The fixture includes events above avmax=0.4 that pass the default AV range;
    the explicit limit must affect both selection and bias-correction binning.
    """
    cfg = _cfg()
    cfg.cutwin.append((var, -1.0e12 if lo is None else lo,
                              1.0e12 if hi is None else hi))
    cfg.lcpar_range[var] = (lo, hi)

    _, output, _, _, _ = run_bbc_fit(cfg)
    r = read_fitres(f"{_DATA_DIR}/{ref}.M0DIF")
    assert int(output.nfit.sum()) == nsnfit
    np.testing.assert_array_equal(output.nfit, np.asarray(r.get("NFIT")))
    assert np.max(np.abs(output.m0dif - np.asarray(r.get("MUDIF")))) < 1e-4


@pytest.mark.parametrize("avmax,ref,nsnfit", [
    (None, "SALT2mu_bayesn_mucovscale", 381),
    (0.4, "SALT2mu_bayesn_mucovscale_avmax", 363),
])
def test_avmax_sets_the_mucovscale_colour_axis(avmax, ref, nsnfit):
    """muCOVscale bins its colour axis on the range inputs (SALT2mu.c:6987), which
    for BayeSN are avmin/avmax. opt_biascor=176 sets MASK_BIASCOR_MUCOVSCALE, which
    the 144 used elsewhere here does not, so this is the config that reaches them.
    """
    cfg = _cfg(opt_biascor=176)
    if avmax is not None:
        cfg.cutwin.append(("AV", -1.0e12, avmax))
        cfg.lcpar_range["AV"] = (None, avmax)

    _, output, _, _, _ = run_bbc_fit(cfg)
    r = read_fitres(f"{_DATA_DIR}/{ref}.M0DIF")
    assert int(output.nfit.sum()) == nsnfit
    np.testing.assert_array_equal(output.nfit, np.asarray(r.get("NFIT")))
    assert np.max(np.abs(output.m0dif - np.asarray(r.get("MUDIF")))) < 1e-4
