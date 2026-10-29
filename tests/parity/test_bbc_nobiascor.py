"""Parity vs SALT2mu.exe on the core distance fit (no biasCor).

See tests/fixtures/snana/bbc_des_beams/refs.py for the reference config.
1624 SNe pass the default |x1|<3, |c|<0.3 cuts; SNANA fit alpha0=0.15939,
beta0=3.50105, sigint=0.
"""


import numpy as np
import pytest

from jax_salt2mu.config import parse_bbc_input
from jax_salt2mu.run import run_bbc_fit
from jax_snlc_core.snana_files import read_fitres

_DIR = "tests/fixtures/snana/bbc_des_beams"
_DATA = f"{_DIR}/DATA.FITRES"
_REF = f"{_DIR}/SALT2mu_plain.M0DIF"

_SNANA_ALPHA0 = 0.15939
_SNANA_BETA0 = 3.50105


@pytest.mark.parametrize("chi2max,nsnfit,alpha0,beta0", [
    (16, 1094, 0.12108, 2.63224),
    (9, 924, 0.09892, 2.19269),
    (4, 682, 0.06390, 1.63686),
])
def test_chi2max_cut_matches_salt2mu(chi2max, nsnfit, alpha0, beta0):
    """applyCut_chi2max global pre-fit cut vs SALT2mu.exe: the number surviving
    the cut and the resulting alpha0/beta0. Default parval alpha=beta=0 is clamped
    to the SNANA fit bounds (0.02, 1.0) for the cut's residual/muerr."""
    cfg = parse_bbc_input(None, overrides=[
        "radiation=0",
        f"datafile={_DATA}", "nzbin=10", "zmin=0.1", "zmax=1.0", "u1=1", "u2=1",
        f"chi2max={chi2max}"])
    res, out, _, _, _ = run_bbc_fit(cfg)
    assert int(out.nfit.sum()) == nsnfit
    assert abs(res.nuisance["alpha0"] - alpha0) < 5e-4
    assert abs(res.nuisance["beta0"] - beta0) < 2e-4


@pytest.mark.parametrize("p1,p2,nsnfit", [
    ("0.08", "2.0", 1491),  # both parval inside bounds -> no clamp
    ("0.10", "0.5", 1415),  # beta 0.5 clamped to 1.0, alpha 0.10 unclamped
])
def test_chi2max_parval_clamp_matches_salt2mu(p1, p2, nsnfit):
    """The chi2max cut clamps the parval init per-axis into the SNANA fit bounds:
    survivors match SALT2mu.exe for interior and one-axis-clamped parvals."""
    cfg = parse_bbc_input(None, overrides=[
        "radiation=0",
        f"datafile={_DATA}", "nzbin=10", "zmin=0.1", "zmax=1.0", "u1=1", "u2=1",
        f"p1={p1}", f"p2={p2}", "chi2max=16"])
    _, out, _, _, _ = run_bbc_fit(cfg)
    assert int(out.nfit.sum()) == nsnfit


def test_core_fit_matches_salt2mu():
    cfg = parse_bbc_input(None, overrides=[
        "radiation=0",
        f"datafile={_DATA}", "nzbin=10", "zmin=0.1", "zmax=1.0", "u1=1", "u2=1"])
    res, out, zbins, tv, _ = run_bbc_fit(cfg)

    # default |x1|<3, |c|<0.3 cuts select the same sample SNANA fits.
    assert int(tv.cutmask.sum()) == int(read_fitres(_REF).get("NFIT").sum())

    ref = read_fitres(_REF)
    np.testing.assert_allclose(out.muref, ref.get("MUREF"), atol=1e-3)
    np.testing.assert_allclose(out.m0dif, ref.get("MUDIF"), atol=1e-3)
    np.testing.assert_array_equal(out.nfit, ref.get("NFIT").astype(int))
    assert abs(res.nuisance["alpha0"] - _SNANA_ALPHA0) < 5e-4
    assert abs(res.nuisance["beta0"] - _SNANA_BETA0) < 2e-4
