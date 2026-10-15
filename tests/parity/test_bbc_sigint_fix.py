"""Parity vs SALT2mu.exe for sigint_fix (per-IDSAMPLE fixed intrinsic scatter).

sigint_fix pins sigma_int and stops the sigint iteration (STOP_COVFIX,
SALT2mu.c:4742). The two halves are gated differently: the iteration always stops,
but the scatter only reaches the covariance when a biasCor supplied IDSAMPLEs *and*
the iteration would have run -- parse_sigint_fix (:10151) fires after the initial
covariance is built, so only the iteration's rebuild picks the values up.

References are SALT2mu.exe on the 5D biasCor config; see the fixture README.
"""

from jax_snlc_core.io.fits_path import open_text

import functools

import numpy as np
import pytest

from jax_salt2mu.config_parser import parse_bbc_input
from jax_salt2mu.fitter import MAXFITITER
from jax_salt2mu.output import write_m0_csv
from jax_salt2mu.run import run_bbc_fit
from jax_snlc_core.io.fitres_reader import read_fitres
from jax_snlc_core.utils.error_handling import SNANAError


_DIR = "tests/fixtures/snana/bbc_des_beams"
# SNANA BBC reference files omit radiation.
_BASE = ["radiation=0",
         "nzbin=10", "zmin=0.1", "zmax=1.0", "u1=1", "u2=1", "p9=0.7", "p11=-1",
         "blindflag=0", f"datafile={_DIR}/DATA.FITRES",
         f"simfile_biascor={_DIR}/BIASCOR.FITRES"]
_CELL = ["opt_biascor=112", "min_per_cell_biascor=3", "snrmin_sigint_biascor=60",
         "surveygroup_biascor=DES(zbin=0.15:x1bin=2.0:cbin=0.2)"]

_RUN_CACHE: dict[tuple, object] = {}


def _run(overrides):
    """A fit is deterministic in its overrides, and several tests share configs."""
    key = tuple(overrides)
    if key not in _RUN_CACHE:
        _, out, _, _, _ = run_bbc_fit(
            parse_bbc_input(None, overrides=_BASE + list(overrides)))
        _RUN_CACHE[key] = out
    return _RUN_CACHE[key]


def test_sigint_fix_matches_salt2mu():
    """fitflag_sigmb=1 + sigint_fix=0.11: one fit at the pinned scatter instead of
    the 5-iteration secant. SNANA alpha0=0.1380, beta0=2.9168."""
    out = _run(_CELL + ["fitflag_sigmb=1", "sigint_fix=0.11"])
    r = read_fitres(f"{_DIR}/SALT2mu_sigint_fix.M0DIF")
    np.testing.assert_array_equal(out.nfit, np.asarray(r.get("NFIT")))
    assert np.max(np.abs(out.m0dif - np.asarray(r.get("MUDIF")))) < 2e-5
    assert np.max(np.abs(out.muref - np.asarray(r.get("MUREF")))) < 2e-5


def test_secant_iterates_like_salt2mu():
    """The secant actually iterating to a converged sigma_int, which every other case
    here stops after one step by pinning sigint.

    Stops on STOP_TOL at alpha0=0.14121, beta0=2.97758, sigint=0.0941 in 4 iterations.
    The default sigint_step1=0.01 resolves the slope too coarsely and oscillates back
    to the floor, ending on STOP_COV0 (SALT2mu.c:4739), so the reference uses 0.05.
    """
    out = _run(_CELL + ["fitflag_sigmb=1", "sigint_step1=0.05"])
    r = read_fitres(f"{_DIR}/SALT2mu_secant.M0DIF")
    np.testing.assert_array_equal(out.nfit, np.asarray(r.get("NFIT")))
    # the last bin holds one event (MUDIFERR 0.24), where MINUIT's EDM stop
    # leaves the M0 a few 1e-5 short of the minimum Newton-LM reaches; the
    # populated bins agree to 5e-6
    slack = 2e-5 + 1e-4 * np.asarray(r.get("MUDIFERR"))
    assert np.all(np.abs(out.m0dif - np.asarray(r.get("MUDIF"))) < slack)
    assert np.max(np.abs(out.muref - np.asarray(r.get("MUREF")))) < 2e-5
    # MUDIFERR is deliberately not compared: SNANA's is invalid once the secant
    # iterates -- see test_secant_m0diferr_respects_conditional_floor.


def _conditional_m0err(out):
    """1/sqrt(sum 1/muerr^2) per z bin: the M0 error with alpha and beta held fixed.

    Marginalising over the floated alpha/beta can only widen an error, so this is a
    hard floor on M0DIFERR for any correct covariance.
    """
    floor = np.full(len(out.m0diferr), np.nan)
    for b in np.unique(out.izbin):
        m = out.izbin == b
        floor[b] = 1.0 / np.sqrt(np.sum(1.0 / out.muerr[m] ** 2))
    return floor


@pytest.mark.parametrize("extra", [
    ["fitflag_sigmb=0"],                            # single fit
    ["fitflag_sigmb=1", "sigint_step1=0.05"],       # secant converges to sigint>0
    ["fitflag_sigmb=1"],                            # secant collapses to sigint=0
], ids=["single", "secant_converged", "secant_to_zero"])
def test_secant_m0diferr_respects_conditional_floor(extra):
    """M0DIFERR cannot fall below the fixed-alpha/beta error built from the same
    per-event MUERR, however the sigint loop terminates.

    No SNANA reference, deliberately: this is an internal consistency bound, and
    SALT2mu.exe violates it. On the last case -- the secant walking back to the
    sigint floor and stopping on STOP_COV0 -- SNANA reports 0.80-0.90 of its own
    floor in all 10 bins, and `minos=1` returns a byte-identical file, so neither
    of its error paths recovers. That is why the
    secant references pin MUDIF and MUREF but never MUDIFERR.
    """
    out = _run(_CELL + extra)
    floor = _conditional_m0err(out)
    ok = np.isfinite(floor)
    ratio = out.m0diferr[ok] / floor[ok]
    assert np.all(ratio >= 1.0 - 1e-6), f"below the floor: {ratio}"
    # and not absurdly above it -- alpha/beta marginalisation is a few percent here
    assert np.all(ratio < 1.10), f"implausibly inflated: {ratio}"


def test_sigint_fix_inert_without_iteration():
    """With fitflag_sigmb=0 there is no iteration, so SNANA never applies the fixed
    scatter and the result is the plain baseline. Measuring only this case is what
    previously led to the wrong conclusion that the key does nothing at all."""
    out = _run(_CELL + ["fitflag_sigmb=0", "sigint_fix=0.11"])
    r = read_fitres(f"{_DIR}/SALT2mu.M0DIF")
    assert np.max(np.abs(out.m0dif - np.asarray(r.get("MUDIF")))) < 2e-5


def test_sigint_fix_with_mucovadd():
    """Under muCOVADD (opt_biascor bit 4096) the global scatter is the
    per-cell additive one and stays zero (makeMap_sigmu_biasCor,
    SALT2mu.c:13101), so sigint_fix only stops the iteration and its value
    never enters the covariance: SNANA's fit is the same at 0.11 and 0.30
    (alpha0=0.1450 either way, against 0.1380 for 0.11 without the bit).
    Until v12_02d the value did reach the covariance, through the
    crazy-M0 repeat of the fit alone, which jax does not port."""
    covadd = ["opt_biascor=4208", "min_per_cell_biascor=3",
              "snrmin_sigint_biascor=60",
              "surveygroup_biascor=DES(zbin=0.15:x1bin=2.0:cbin=0.2)",
              "fitflag_sigmb=1"]
    out = _run(covadd + ["sigint_fix=0.11"])
    r = read_fitres(f"{_DIR}/SALT2mu_sigint_fix_covadd.M0DIF")
    assert np.max(np.abs(out.m0dif - np.asarray(r.get("MUDIF")))) < 2e-5
    other = _run(covadd + ["sigint_fix=0.30"])
    np.testing.assert_array_equal(other.m0dif, out.m0dif)


def test_sigint_fix_value_inert_with_1d_biascor():
    """A 1D biasCor rebuilds no cell map, so the pinned value never reaches the
    covariance and only the iteration stop survives -- different values give the same
    fit. jax previously applied the value here, diverging from SNANA."""
    ref = np.asarray(read_fitres(f"{_DIR}/SALT2mu_sigint_fix_1d.M0DIF").get("MUDIF"))
    low = _run(["opt_biascor=1", "fitflag_sigmb=1", "sigint_fix=0.11"])
    high = _run(["opt_biascor=1", "fitflag_sigmb=1", "sigint_fix=0.40"])
    np.testing.assert_allclose(low.m0dif, high.m0dif, atol=1e-6)
    assert np.max(np.abs(low.m0dif - ref)) < 2e-5


def _split_survey(src, dst, second_id=15):
    """Every other SN row -> IDSURVEY=PS1MD, giving two IDSAMPLEs. Derived at test
    time rather than committed: the biasCor copy would be another 3 MB."""
    out, ivar, n = [], None, 0
    for line in open_text(src):
        w = line.split()
        if line.startswith("VARNAMES:"):
            ivar = w.index("IDSURVEY")
        elif line.startswith("SN:") and ivar is not None:
            n += 1
            if n % 2 == 0:
                w[ivar] = str(second_id)
                line = "  ".join(w) + "\n"
        out.append(line)
    dst.write_text("".join(out))
    return str(dst)


@functools.lru_cache(maxsize=None)
def _run_order(bbc_biascor_variants, first, second, sigint_fix):
    data = f"{_DIR}/DATA_SDSS.FITRES"
    biascor = f"{bbc_biascor_variants}/BIASCOR_SDSS.FITRES"
    binning = "(zbin=0.15:x1bin=2.0:cbin=0.2)"
    ov = [k for k in _BASE if not k.startswith(("datafile=", "simfile_biascor="))]
    ov += [f"datafile={data}", f"simfile_biascor={biascor}", "opt_biascor=112",
           f"surveygroup_biascor={first}{binning},{second}{binning}",
           "fitflag_sigmb=1", f"sigint_fix={sigint_fix}"]
    _, out, _, _, _ = run_bbc_fit(parse_bbc_input(None, overrides=ov))
    return out


def test_sigint_fix_wrong_count_aborts(bbc_biascor_variants):
    """SNANA aborts when the value count != N_IDSAMPLE (parse_sigint_fix:19918)."""
    with pytest.raises(SNANAError, match="IDSAMPLE"):
        _run_order(bbc_biascor_variants, "DES", "SDSS", "0.11,0.2,0.3")


@pytest.mark.parametrize("first,second,ref", [
    ("DES", "SDSS", "des_first"),
    ("SDSS", "DES", "sdss_first"),
])
def test_idsample_order_follows_surveygroup(bbc_biascor_variants, first, second, ref):
    """IDSAMPLE index follows the order surveys appear in surveygroup_biascor, not
    IDSURVEY, file order or alphabetical -- SDSS is IDSURVEY=1 yet lands at IDSAMPLE=1
    when listed second. sigint_fix=0.11,0.30 is indexed by that order, so reversing the
    spec changes the fit (0.1000 vs 0.1099). Without this the earlier LSST/PS1MD test
    could not distinguish the ordering rules, which all agreed by coincidence. Also
    covers the per-IDSAMPLE list form: the same 0.11,0.30 maps to different samples."""
    out = _run_order(bbc_biascor_variants, first, second, "0.11,0.30")
    r = read_fitres(f"{_DIR}/SALT2mu_sigint_fix_order_{ref}.M0DIF")
    np.testing.assert_array_equal(out.nfit, np.asarray(r.get("NFIT")))
    assert np.max(np.abs(out.m0dif - np.asarray(r.get("MUDIF")))) < 2e-5


def test_single_value_expands_across_idsamples(bbc_biascor_variants):
    """One value applies to every IDSAMPLE (parse_sigint_fix:19942), so it must equal
    the explicit list of that value repeated. Needs >1 IDSAMPLE: with a single sample
    the expansion is a no-op, which is why the 5D and order tests miss it."""
    one = _run_order(bbc_biascor_variants, "DES", "SDSS", "0.11")
    both = _run_order(bbc_biascor_variants, "DES", "SDSS", "0.11,0.11")
    np.testing.assert_allclose(one.m0dif, both.m0dif, atol=1e-9)
    # and it must differ from a genuinely per-sample list, else the test is vacuous
    split = _run_order(bbc_biascor_variants, "DES", "SDSS", "0.11,0.30")
    assert np.max(np.abs(one.m0dif - split.m0dif)) > 1e-3


def test_write_csv_binned_hubble_diagram(tmp_path):
    """write_csv emits the binned HD for CosmoMC (write_M0_csv, SALT2mu.c:23962):
    MU is the reference distance at the reported z plus M0DIF, not the weighted MUREF
    column of the M0DIF file."""
    cfg = parse_bbc_input(None, overrides=_BASE + _CELL + ["fitflag_sigmb=0"])
    _, out, zb, _, _ = run_bbc_fit(cfg)
    path = tmp_path / "out.CSV"
    write_m0_csv(str(path), zb, out, cfg)
    got = [l.split() for l in path.read_text().splitlines() if l.startswith("BIN")]
    ref = [l.split() for l in open_text(f"{_DIR}/SALT2mu_write_csv.CSV") if l.startswith("BIN")]
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        assert g[0] == r[0]
        assert abs(float(g[1]) - float(r[1])) < 1e-4     # z
        assert abs(float(g[4]) - float(r[4])) < 5e-4     # mu
        assert abs(float(g[5]) - float(r[5])) < 1e-3     # muerr


def test_sigma_cell_biascor():
    """sigma_cell_biascor Gaussian-weights each biasCor event by its normalised
    distance from the cell centroid (WGT_biasCor, SALT2mu.c:12753). The centroids
    themselves come from the flat 1/muerrsq weight, so the two passes are ordered.
    Default 50 (>10) is a flat weight; 0.5 moves alpha0 0.18189 -> 0.18198."""
    cfg = parse_bbc_input(None, overrides=_BASE + _CELL + ["sigma_cell_biascor=0.5"])
    res, out, _, _, _ = run_bbc_fit(cfg)
    r = read_fitres(f"{_DIR}/SALT2mu_sigma_cell.M0DIF")
    assert abs(res.nuisance["alpha0"] - 0.18198) < 2e-5
    assert np.max(np.abs(out.m0dif - np.asarray(r.get("MUDIF")))) < 2e-5


def test_write_chi2grid(tmp_path):
    """write_chi2grid scans alpha over [0.10,0.20] and beta over [2.0,4.0], each row
    varying one with the other at its fitted value (SALT2mu.c:24094)."""
    cfg = parse_bbc_input(None, overrides=_BASE + _CELL + [
        "write_chi2grid=5", f"prefix={tmp_path}/grid"])
    run_bbc_fit(cfg)
    got = [l.split() for l in open_text(f"{tmp_path}/grid.CHI2GRID") if l.startswith("ROW:")]
    ref = [l.split() for l in open_text(f"{_DIR}/SALT2mu_chi2grid.CHI2GRID")
           if l.startswith("ROW:")]
    assert len(got) == len(ref) == 5
    for g, r in zip(got, ref):
        assert abs(float(g[2]) - float(r[2])) < 1e-6      # alpha
        assert abs(float(g[3]) - float(r[3])) < 1e-6      # beta
        for k in (4, 5):                                   # chi2_alpha, chi2_beta
            assert abs(float(g[k]) - float(r[k])) / float(r[k]) < 2e-5


def test_the_secant_reports_the_sigint_it_last_fitted():
    """prepNextFit evaluates the stop conditions and returns before
    next_covFitPar (SALT2mu.c:4738), so the reported sigma_int is always the
    one the last inner fit ran with. With a tolerance nothing can meet, this
    sample's scatter is driven to zero (STOP_COV0) before MAXFITITER, and zero
    is reported."""
    cfg = parse_bbc_input(None, overrides=_BASE + _CELL + ["fitflag_sigmb=1"])
    cfg.redchi2_tol = 1.0e-9
    result, _, _, _, _ = run_bbc_fit(cfg)
    assert 1 < result.niter < MAXFITITER
    assert result.sigint == 0.0
