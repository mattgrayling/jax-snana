"""Parity vs SALT2mu.exe for uM0=2 (interpolated M0) and muref_zpoly.

uM0=2 makes M0 linear between z-bin knots instead of flat per bin (fcn_M0,
SALT2mu.c:5826); the knots are the biasCor's weighted mean z per bin
(calc_zM0_biasCor, :14774), so the mode needs a biasCor and aborts without one.
muref_zpoly replaces the LCDM distance in cosmodl (:25755) and force-sets uM0=2.

References are on the bbc_des_beams 5D config; see that fixture's refs.py.
"""


import numpy as np
import pytest

from jax_salt2mu.config_parser import parse_bbc_input
from jax_salt2mu.run import run_bbc_fit
from jax_snlc_core.io.fitres_reader import read_fitres
from jax_snlc_core.utils.error_handling import SNANAError


_DIR = "tests/fixtures/snana/bbc_des_beams"
# SNANA BBC reference files omit radiation.
_BASE = ["radiation=0",
         "nzbin=10", "zmin=0.1", "zmax=1.0", "u1=1", "u2=1", "p9=0.7", "p11=-1",
         "blindflag=0", f"datafile={_DIR}/DATA.FITRES",
         f"simfile_biascor={_DIR}/BIASCOR.FITRES", "opt_biascor=112",
         "surveygroup_biascor=DES(zbin=0.15:x1bin=2.0:cbin=0.2)",
         "min_per_cell_biascor=3", "snrmin_sigint_biascor=60"]
# a cubic through the reference cosmology over the fitted z range
_ZPOLY = "muref_zpoly=36.732550,19.401037,-20.992814,9.015726"



def _run(overrides):
    _, out, _, _, _ = run_bbc_fit(parse_bbc_input(None, overrides=_BASE + overrides))
    return out


def _assert_m0dif(out, reference, tol=1e-4):
    r = read_fitres(f"{_DIR}/{reference}")
    assert np.max(np.abs(out.m0dif - np.asarray(r.get("MUDIF")))) < tol
    assert np.max(np.abs(out.muref - np.asarray(r.get("MUREF")))) < tol
    # zHD inverts the same reference mu that mumodel used; under muref_zpoly that
    # is the polynomial, not LCDM
    assert np.max(np.abs(out.zbin_z - np.asarray(r.get("zHD")))) < 1e-4


def test_um0_interp_matches_salt2mu():
    """SNANA alpha0=0.23621, beta0=3.40284 (vs 0.24052 at the default uM0=1)."""
    _assert_m0dif(_run(["uM0=2"]), "SALT2mu_um0_interp.M0DIF")


def test_um0_interp_differs_from_flat():
    """The two references differ, so the parity test above would still pass if the
    interpolation were silently ignored -- this is what rules that out."""
    flat = read_fitres(f"{_DIR}/SALT2mu.M0DIF")
    interp = read_fitres(f"{_DIR}/SALT2mu_um0_interp.M0DIF")
    moved = np.max(np.abs(np.asarray(flat.get("MUDIF"))
                          - np.asarray(interp.get("MUDIF"))))
    assert moved > 1e-3


def test_um0_interp_needs_a_biascor():
    """fcn_M0 aborts without simfile_biascor (SALT2mu.c:5828): the knots are the
    biasCor's weighted mean z per bin, and there is nothing to interpolate over."""
    config = parse_bbc_input(None, overrides=[
        f"datafile={_DIR}/DATA.FITRES", "nzbin=10", "zmin=0.1", "zmax=1.0", "uM0=2"])
    with pytest.raises(SNANAError, match="simfile_biascor"):
        run_bbc_fit(config)


def test_muref_zpoly_matches_salt2mu():
    """SNANA alpha0=0.23648, beta0=3.39971."""
    _assert_m0dif(_run([_ZPOLY]), "SALT2mu_muref_zpoly.M0DIF")


def test_muref_zpoly_forces_interp_and_disables_chi2max():
    """prep_input (SALT2mu.c:22203) sets uM0=2 and iflag_chi2max=0 whenever the
    polynomial is given, whatever the user asked for."""
    config = parse_bbc_input(None, overrides=_BASE + [_ZPOLY, "uM0=1", "chi2max=16"])
    assert config.uM0 == 2
    assert config.chi2max > 1e8


def test_crazy_m0_errors_matches_salt2mu():
    """crazy_M0_errors (SALT2mu.c:4811) flags z bins whose fitted M0 error is below
    0.67*sqrt(<muCOVscale>)*0.1/sqrt(N). On this config SNANA reports two:
    'iz=0 M0ERR=0.01430 ERRMIN_COMPUTE=0.10/sqrt(18)=0.02452' and
    'iz=1 M0ERR=0.00854 ERRMIN_COMPUTE=0.10/sqrt(64)=0.01288'.

    SNANA retries the fit only at >=3 flagged bins, so NFIT_EXEC stays 1 here and
    the references in this fixture are all single-fit.
    """
    out = _run([])
    assert list(np.flatnonzero(out.crazy_m0)) == [0, 1]


def test_crazy_check_skipped_under_fixpar_all():
    """fixpar_all returns before the check (SALT2mu.c:4845): nothing is fitted, so
    an M0 error below the floor carries no information."""
    out = _run(["fixpar_all=1"])
    assert not out.crazy_m0.any()
