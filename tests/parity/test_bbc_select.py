"""Parity vs SALT2mu.exe for the selection and z-binning keys (no biasCor).

Each reference is SALT2mu.exe on an existing fixture with one selection key set;
see tests/fixtures/snana/bbc_des_beams/README.md.
"""

import logging

import numpy as np
import pytest

from jax_salt2mu.config import resolve
from jax_salt2mu.config import parse_bbc_input
from jax_salt2mu.run import run_bbc_fit
from _bbc_helpers import prepare_cell
from jax_salt2mu.tablevar import CUTBIT_BIASCOR, CUTBIT_CHI2, EVENT_TYPE_DATA
from _bbc_helpers import read_tablevar
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.models import SALTModel
from jax_snlc_core.snana_files import open_text, read_fitres


_DIR = "tests/fixtures/snana/bbc_des_beams"
# SNANA BBC reference files omit radiation.
_BASE = ["radiation=0",
         "nzbin=10", "zmin=0.1", "zmax=1.0", "u1=1", "u2=1", "p9=0.7", "p11=-1",
         "blindflag=0"]


@pytest.mark.parametrize("datafile,key,ref,nsnfit", [
    ("CONTAM_MIXED.FITRES", "sntype_select=1", "SALT2mu_sntype.M0DIF", 1624),
    ("DATA.FITRES", "prescale_simdata=2", "SALT2mu_ps_simdata.M0DIF", 807),
    ("DATA.FITRES", "prescale_simIa=2", "SALT2mu_ps_simia.M0DIF", 807),
    # z-cheat: zHD <- SIM_ZCMB; no event crosses the z cut on this sample
    ("DATA.FITRES", "uzsim=1", "SALT2mu_uzsim.M0DIF", 1624),
    # CUTBIT_BADERR: an LC-fit error <= the threshold is cut (SALT2mu.c:18885)
    ("DATA.FITRES", "maxerr_abort_x1=0.5", "SALT2mu_maxerr_x1.M0DIF", 1084),
    ("DATA.FITRES", "maxerr_abort_c=0.05", "SALT2mu_maxerr_c.M0DIF", 749),
    ("DATA.FITRES", "nmax=200", "SALT2mu_nmax.M0DIF", 200),
    ("DATA.FITRES", "powzbin=2.0", "SALT2mu_powzbin.M0DIF", 1624),
    ("DATA.FITRES", "vpecerr=250", "SALT2mu_vpecerr.M0DIF", 1624),
    # a generic CUTWIN reads any column of the table (SALT2mu.c:8359, :8809):
    # SNRMAX1 is not otherwise read, PROBIA is DES-SN5YR's BEAMS selection
    ("DATA.FITRES", "CUTWIN SNRMAX1 20 1000", "SALT2mu_cutwin_snrmax.M0DIF", 545),
    ("CONTAM_DATA.FITRES", "CUTWIN PROBIA 0.5 1.0", "SALT2mu_cutwin_probia.M0DIF",
     1624),
    # CUTWIN NONE lifts the x1 cut beside it and the model's default windows,
    # keeping the z cut (setbit_CUTMASK, SALT2mu.c:19016)
    ("DATA.FITRES", "x1min=-2.0 CUTWIN NONE", "SALT2mu_cutwin_none.M0DIF", 1775),
    # nlogzbin/zbinuser replace nzbin entirely; _BASE's nzbin=10 is dropped below.
    ("DATA.FITRES", "nlogzbin=8", "SALT2mu_nlogzbin.M0DIF", 1624),
    ("DATA.FITRES", "zbinuser=0.1,0.3,0.6,1.0", "SALT2mu_zbinuser.M0DIF", 1624),
    # edges inside the data's span: they overwrite zmin/zmax (SALT2mu.c:2897), so the
    # z cut follows the edges rather than the zmin/zmax keys also given here
    ("DATA.FITRES", "zbinuser=0.2,0.4,0.6", "SALT2mu_zbinuser_narrow.M0DIF", 680),
])
def test_selection_key_matches_salt2mu(datafile, key, ref, nsnfit):
    refpath = f"{_DIR}/{ref}"
    base = _BASE
    if key.startswith(("nlogzbin=", "zbinuser=")):
        # SALT2mu.c:21951 aborts unless exactly one binning form is given.
        base = [k for k in base if not k.startswith("nzbin=")]
    cfg = parse_bbc_input(None, overrides=[f"datafile={_DIR}/{datafile}", key] + base)
    _, out, _, _, _ = run_bbc_fit(cfg)

    r = read_fitres(refpath)
    assert int(out.nfit.sum()) == nsnfit
    np.testing.assert_array_equal(out.nfit, np.asarray(r.get("NFIT")))
    assert np.max(np.abs(out.m0dif - np.asarray(r.get("MUDIF")))) < 2e-5


def test_min_per_zbin_matches_salt2mu():
    """min_per_zbin=40 empties the first z bin: setup_zbins_fit sets CUTBIT_MINBIN
    on its 19 events (SALT2mu.c:3127), so it is not floated and writes MUDIFERR=999.

    NFIT is SNANA's NEVT_zFIT, counted before that pass (:3091), so one SALT2mu exec
    writes 19 there. This reference reads 0 because that run repeated its fit ("3 fit
    M0ERR had crazy values --> fit repeated" in its own header) and the second exec
    re-counted with the bits already set. jax never repeats a fit, so it writes the
    single-exec number; see the crazy_M0_errors decision in
    notes/SALT2MU_KEY_TRIAGE.md and the proposed ledger entry in
    notes/review_2026-09-11/impl_bbc.md.
    """
    cfg = parse_bbc_input(None, overrides=[f"datafile={_DIR}/DATA.FITRES",
                                           "min_per_zbin=40"] + _BASE)
    _, out, _, _, _ = run_bbc_fit(cfg)

    r = read_fitres(f"{_DIR}/SALT2mu_min_per_zbin.M0DIF")
    fitted = np.asarray(r.get("MUDIFERR")) < 999.0
    assert not fitted[0] and fitted[1:].all()
    np.testing.assert_array_equal(out.m0diferr < 999.0, fitted)
    np.testing.assert_array_equal(out.nfit[fitted], np.asarray(r.get("NFIT"))[fitted])
    assert np.max(np.abs(out.m0dif[fitted]
                         - np.asarray(r.get("MUDIF"))[fitted])) < 2e-5
    # the emptied bin: 19 events cut, which SNANA's second exec re-counted as 0
    assert out.nfit[0] == 19 and int(np.asarray(r.get("NFIT"))[0]) == 0
    assert int(out.nfit[fitted].sum()) == 1605


def test_prescale_simcc_on_mixed_templates():
    """prescale_simcc halves the SIM_TEMPLATE_INDEX != 0 events of DATA_MIXTMPL
    (1624 -> 1212), with MUDIF comparable because the sample is real Ia photometry."""
    cfg = parse_bbc_input(None, overrides=[f"datafile={_DIR}/DATA_MIXTMPL.FITRES",
                                           "prescale_simcc=2"] + _BASE)
    _, out, _, _, _ = run_bbc_fit(cfg)
    r = read_fitres(f"{_DIR}/SALT2mu_ps_simcc_mixed.M0DIF")
    assert int(out.nfit.sum()) == 1212
    np.testing.assert_array_equal(out.nfit, np.asarray(r.get("NFIT")))
    assert np.max(np.abs(out.m0dif - np.asarray(r.get("MUDIF")))) < 2e-5


@pytest.mark.parametrize("flag,ref,nsnfit", [(0, "ignore", 1641), (2, "merge", 1624)])
def test_iflag_duplicate(flag, ref, nsnfit):
    """iflag_duplicate 0 keeps duplicate CIDs as separate events; 2 merges each set by
    inverse-covariance weight and cuts the rest (CUTBIT_DUPL). force_realdata is needed
    because SNANA only checks duplicates for real data (SALT2mu.c:2270)."""
    cfg = parse_bbc_input(None, overrides=[f"datafile={_DIR}/DATA_DUP.FITRES",
                                           "force_realdata=1",
                                           f"iflag_duplicate={flag}"] + _BASE)
    _, out, _, _, _ = run_bbc_fit(cfg)
    r = read_fitres(f"{_DIR}/SALT2mu_dupl_{ref}.M0DIF")
    assert int(out.nfit.sum()) == nsnfit
    np.testing.assert_array_equal(out.nfit, np.asarray(r.get("NFIT")))
    assert np.max(np.abs(out.m0dif - np.asarray(r.get("MUDIF")))) < 2e-5


def test_iflag_duplicate_aborts():
    """The SNANA default (=1) rejects duplicate CIDs outright."""
    cfg = parse_bbc_input(None, overrides=[f"datafile={_DIR}/DATA_DUP.FITRES",
                                           "force_realdata=1"] + _BASE)
    assert cfg.iflag_duplicate == 1          # SNANA default is ABORT
    with pytest.raises(SNANAError, match="duplicate"):
        run_bbc_fit(cfg)


def test_cid_reject_file():
    """cid_reject_file drops the 100 listed CIDs (1624 -> 1532)."""
    cfg = parse_bbc_input(None, overrides=[f"datafile={_DIR}/DATA.FITRES",
                                           f"cid_reject_file={_DIR}/cid_reject.list"] + _BASE)
    _, out, _, _, _ = run_bbc_fit(cfg)
    r = read_fitres(f"{_DIR}/SALT2mu_cid_reject.M0DIF")
    assert int(out.nfit.sum()) == 1532
    np.testing.assert_array_equal(out.nfit, np.asarray(r.get("NFIT")))
    assert np.max(np.abs(out.m0dif - np.asarray(r.get("MUDIF")))) < 2e-5


def test_cid_select_file():
    """cid_select_file keeps only the 1700 listed CIDs (1624 -> 1553)."""
    cfg = parse_bbc_input(None, overrides=[f"datafile={_DIR}/DATA.FITRES",
                                           f"cid_select_file={_DIR}/cid_select.list"] + _BASE)
    _, out, _, _, _ = run_bbc_fit(cfg)
    r = read_fitres(f"{_DIR}/SALT2mu_cid_select.M0DIF")
    assert int(out.nfit.sum()) == 1553
    np.testing.assert_array_equal(out.nfit, np.asarray(r.get("NFIT")))
    assert np.max(np.abs(out.m0dif - np.asarray(r.get("MUDIF")))) < 2e-5


def test_izbin_from_bare_cid_list_contributes_nothing():
    """izbin_from_cid_file on a legacy bare CID list (no VARNAMES header):
    parse_cidFile_data accepts the format and the IZBIN request contributes
    nothing, so the run completes with the plain cid_select result."""
    cfg = parse_bbc_input(None, overrides=[
        f"datafile={_DIR}/DATA.FITRES",
        f"cid_select_file={_DIR}/cid_select.list",
        "izbin_from_cid_file=1"] + _BASE)
    _, out, _, _, _ = run_bbc_fit(cfg)
    r = read_fitres(f"{_DIR}/SALT2mu_cid_select.M0DIF")
    np.testing.assert_array_equal(out.nfit, np.asarray(r.get("NFIT")))
    assert np.max(np.abs(out.m0dif - np.asarray(r.get("MUDIF")))) < 2e-5


def test_fieldlist_substring_match():
    """select_FIELD (SALT2mu.c:20797) matches by substring: VOID keeps every event in
    this fixture, an absent field keeps none."""
    keep = parse_bbc_input(None, overrides=[f"datafile={_DIR}/DATA.FITRES",
                                            "fieldlist=VOID"] + _BASE)
    _, out, _, _, _ = run_bbc_fit(keep)
    assert int(out.nfit.sum()) == 1624
    none_cfg = resolve(parse_bbc_input(None, overrides=[
        f"datafile={_DIR}/DATA.FITRES", "fieldlist=NOSUCHFIELD"] + _BASE), SALTModel)
    tv = read_tablevar(none_cfg.datafile, EVENT_TYPE_DATA, none_cfg, SALTModel)
    assert tv.n_pass == 0


def test_zrate_scale_file():
    """Deterministic z-dependent downsampling (accept_zrate_scale, SALT2mu.c:22503):
    the last three digits of the CID are the draw, kept when <= the interpolated
    scale(z). A flat scale of 0.5 keeps 816 of 1624 -- not 198, because the last-digit
    distribution is not uniform, which is exactly why it must match SNANA event for
    event rather than statistically."""
    table = f"{_DIR}/zrate.txt"
    cfg = parse_bbc_input(None, overrides=[f"datafile={_DIR}/DATA.FITRES",
                                           f"zrate_scale_file={table}"] + _BASE)
    _, out, _, _, _ = run_bbc_fit(cfg)
    r = read_fitres(f"{_DIR}/SALT2mu_zrate_scale.M0DIF")
    assert int(out.nfit.sum()) == 816
    np.testing.assert_array_equal(out.nfit, np.asarray(r.get("NFIT")))
    assert np.max(np.abs(out.m0dif - np.asarray(r.get("MUDIF")))) < 2e-5


@pytest.fixture(scope="module")
def two_survey_minority(tmp_path_factory):
    """DATA/BIASCOR with every 10th event switched to SDSS, so the second survey is a
    minority and no z bin is made up entirely of it."""
    def split(src, dst):
        out, ivar, n = [], None, 0
        for line in open_text(src):
            w = line.split()
            if line.startswith("VARNAMES:"):
                ivar = w.index("IDSURVEY")
            elif line.startswith("SN:") and ivar is not None:
                n += 1
                if n % 10 == 0:
                    w[ivar] = "1"
                    line = "  ".join(w) + "\n"
            out.append(line)
        dst.write_text("".join(out))
        return str(dst)
    d = tmp_path_factory.mktemp("minority")
    return (split(f"{_DIR}/DATA.FITRES", d / "DATA_SD10.FITRES"),
            split(f"{_DIR}/BIASCOR.FITRES", d / "BIASCOR_SD10.FITRES"))


def test_surveylist_nobiascor_exempts_its_events(two_survey_minority):
    """surveylist_nobiascor drops the biasCor requirement for the listed surveys
    (DOFLAG_BIASCOR=0, SALT2mu.c:10119): their events stop being cut for CUTBIT_BIASCOR.

    Asserted jax-side rather than against SALT2mu.exe: SNANA aborts in calc_zM0_data
    with muerr=0 whenever this key is exercised on a synthetic split -- tried the full
    50/50 split, opt_biascor without MUCOVSCALE, and this 10% minority. No reference is
    obtainable, so this pins the documented behaviour instead of a reference number."""
    data, biascor = two_survey_minority
    base = [k for k in _BASE if not k.startswith(("datafile=", "nzbin="))]
    base += [f"datafile={data}", f"simfile_biascor={biascor}", "nzbin=10",
             "opt_biascor=112",
             "surveygroup_biascor=DES(zbin=0.15:x1bin=2.0:cbin=0.2),"
             "SDSS(zbin=0.15:x1bin=2.0:cbin=0.2)"]
    _, plain, _, tv_plain, _ = run_bbc_fit(parse_bbc_input(None, overrides=base))
    _, exempt, _, tv_exempt, _ = run_bbc_fit(
        parse_bbc_input(None, overrides=base + ["surveylist_nobiascor=SDSS"]))
    assert tv_exempt.n_fail(CUTBIT_BIASCOR) < tv_plain.n_fail(CUTBIT_BIASCOR)
    assert int(exempt.nfit.sum()) > int(plain.nfit.sum())


def test_bcor_loss_counts_match_snana(caplog):
    """BCOR_LOSS and ndump_nobiascor count rejections among events passing the other
    cuts, not among all events. SNANA prints "Rejected 150 (of 1624) DATA events with
    no 5D biasCor (DES)" for this config; an earlier version of this reporting counted
    against the pre-cut total, which the identical fitted result would never expose."""
    cfg = parse_bbc_input(None, overrides=[
        f"datafile={_DIR}/DATA.FITRES", f"simfile_biascor={_DIR}/BIASCOR.FITRES",
        "opt_biascor=112", "surveygroup_biascor=DES(zbin=0.15:x1bin=2.0:cbin=0.2)",
        "ndump_nobiascor=2"] + _BASE)
    with caplog.at_level(logging.WARNING):
        run_bbc_fit(cfg)
    loss = [r.message for r in caplog.records if "BCOR_LOSS" in r.message]
    assert loss and "150 of 1624" in loss[0]
    assert len([r for r in caplog.records if "no valid biasCor for CID" in r.message]) == 2
    # BBC holds the model class, not an instance, so the advice names the class
    advice = [r.message for r in caplog.records if "min_per_cell_biascor" in r.message]
    assert advice and "SALTModel.biascor_binning" in advice[0]


def test_force_realdata_makes_a_sim_blind():
    """force_realdata ignores the SIM columns (SALT2mu.c:8825), so a simulated input
    counts as real data -- which turns blinding on, since sims are otherwise exempt.
    Run without blindflag=0 so the default BLINDMASK_FIXPAR applies."""
    base = [k for k in _BASE if not k.startswith("blindflag=")]
    cfg = parse_bbc_input(None, overrides=[f"datafile={_DIR}/DATA.FITRES",
                                           "force_realdata=1"] + base)
    _, out, _, _, _ = run_bbc_fit(cfg)
    r = read_fitres(f"{_DIR}/SALT2mu_force_realdata.M0DIF")
    np.testing.assert_array_equal(out.nfit, np.asarray(r.get("NFIT")))
    assert np.max(np.abs(out.m0dif - np.asarray(r.get("MUDIF")))) < 2e-5


def test_uave0_reports_mu_against_nominal_m0():
    """uave=0 reports MU against M0_DEFAULT instead of the fitted average M0
    (SNANA SNMAG0, SALT2mu.c:24382) -- a constant offset on every event. Compared on
    the FITRES MU column, since MUDIF is unaffected."""
    cfg = parse_bbc_input(None, overrides=[f"datafile={_DIR}/DATA.FITRES",
                                           "uave=0"] + _BASE)
    _, out, _, _, _ = run_bbc_fit(cfg)
    ref = read_fitres(f"{_DIR}/SALT2mu_uave0.FITRES")
    assert np.max(np.abs(out.mu - np.asarray(ref.get("MU")))) < 2e-5


def test_fixpar_all_computes_mu_from_initial_values():
    """fixpar_all turns BBC into a distance calculator (SALT2mu.c:21939): no nuisance
    floats, one z bin, no sigint iteration, and M0 clamped to the +/-0.001 uM0-constant
    window (:2634). Nuisances freeze at SNANA's parval[] defaults -- 0 for alpha0/beta0
    -- not at the model's nuisance_spec values, so MU is the comparison that matters."""
    cfg = parse_bbc_input(None, overrides=[f"datafile={_DIR}/DATA.FITRES",
                                           "fixpar_all=1"] + _BASE)
    res, out, _, _, _ = run_bbc_fit(cfg)
    assert (res.nuisance["alpha0"], res.nuisance["beta0"]) == (0.0, 0.0)
    assert len(out.nfit) == 1 and out.nfit[0] == 1624
    ref = read_fitres(f"{_DIR}/SALT2mu_fixpar_all.FITRES")
    assert np.max(np.abs(out.mu - np.asarray(ref.get("MU")))) < 2e-5


# A holds the first half of DATA plus PROB_TEST, B the rest without SNRSUM: one
# column missing from a subset, one matching the default append pattern.
_SPLIT = (f"{_DIR}/DATA_SPLIT_A.FITRES", f"{_DIR}/DATA_SPLIT_B.FITRES")


def _passthrough(split, key):
    cfg = resolve(parse_bbc_input(None, overrides=[
        f"datafile={split[0]},{split[1]}", f"append_varname_missing={key}"] + _BASE),
        SALTModel)
    return read_tablevar(cfg.datafile, EVENT_TYPE_DATA, cfg, SALTModel)


def test_append_varname_missing_columns():
    """Output column list for several append_varname_missing patterns: columns in
    every datafile, then the named ones (trailing * = substring)."""
    for line in open_text(f"{_DIR}/SALT2mu_append_varname.VARNAMES"):
        key, *expect = line.split()
        assert list(_passthrough(_SPLIT, key).passthrough) == expect, key


def test_append_varname_missing_values():
    """Every passthrough column matches SALT2mu.exe token-for-token, with "-9.0"
    where the column is absent from one of the datafiles.

    SNANA re-emits the input words rather than reformatting them
    (write_fitres_line, SALT2mu.c:24501), so the comparison is on the text: a
    numeric check would pass while silently dropping the input's precision.
    """
    tv = _passthrough(_SPLIT, "PROB*")
    ref = read_fitres(f"{_DIR}/SALT2mu_append_varname.FITRES", as_text=True)
    cols = list(ref.df.columns)[:list(ref.df.columns).index("CUTMASK")]
    assert list(tv.passthrough) == cols

    keep = {c: i for i, c in enumerate(np.asarray(ref.get("CID")).astype(str))}
    mine = {c: i for i, c in enumerate(tv.name) if c in keep}
    assert len(mine) == ref.size
    for col in cols:
        got, want = tv.passthrough[col], ref.get(col)
        assert all(got[i] == want[keep[c]] for c, i in mine.items()), col

    prob = np.asarray(tv.passthrough["PROB_TEST"], dtype=float)
    assert (prob == -9.0).sum() == tv.size - tv.size // 2   # B is the larger half


@pytest.mark.parametrize("idsample,ref,nsnfit", [
    (0, "SALT2mu_idsample0.M0DIF", 724),
    (1, "SALT2mu_idsample1.M0DIF", 743),
])
def test_idsample_select_matches_salt2mu(bbc_biascor_variants, idsample, ref, nsnfit):
    """An unselected IDSAMPLE gets no biasCor (prep_IDSAMPLE_SELECT, SALT2mu.c:20646),
    so its events fail the biasCor cut. SNANA's explicit idsample cut (:18927) is
    guarded by idsample >= 0 and never fires for data, where the label is not yet set.
    """
    group = ("SDSS(zbin=0.15:x1bin=2.0:cbin=0.2),"
             "DES(zbin=0.15:x1bin=2.0:cbin=0.2)")
    cfg = parse_bbc_input(None, overrides=[
        f"datafile={_DIR}/DATA_SDSS.FITRES",
        f"simfile_biascor={bbc_biascor_variants}/BIASCOR_SDSS.FITRES", "opt_biascor=112",
        "min_per_cell_biascor=3", "snrmin_sigint_biascor=60",
        f"surveygroup_biascor={group}", f"idsample_select={idsample}"] + _BASE)
    _, out, _, _, _ = run_bbc_fit(cfg)
    r = read_fitres(f"{_DIR}/{ref}")
    assert int(out.nfit.sum()) == nsnfit
    np.testing.assert_array_equal(out.nfit, np.asarray(r.get("NFIT")))
    assert np.max(np.abs(out.m0dif - np.asarray(r.get("MUDIF")))) < 2e-5


def test_idsurvey_select_excludes_the_survey_from_the_partition(bbc_biascor_variants):
    """prepare_IDSAMPLE_biasCor skips an excluded survey before it can become an
    IDSAMPLE (SALT2mu.c:10011-10014), so on this two-survey file DES alone is
    IDSAMPLE 0. Counting the excluded survey as a sample of its own would sort
    SDSS (IDSURVEY=1) below DES (10) and label every DES event IDSAMPLE 1.
    """
    cfg = parse_bbc_input(None, overrides=[
        f"datafile={_DIR}/DATA_SDSS.FITRES",
        f"simfile_biascor={bbc_biascor_variants}/BIASCOR_SDSS.FITRES", "opt_biascor=112",
        "min_per_cell_biascor=3", "snrmin_sigint_biascor=60",
        "idsurvey_select=10"] + _BASE)
    _, out, _, tv, accept = run_bbc_fit(cfg)

    r = read_fitres(f"{_DIR}/SALT2mu_idsurvey_select.M0DIF")
    np.testing.assert_array_equal(out.nfit, np.asarray(r.get("NFIT")))
    assert np.max(np.abs(out.m0dif - np.asarray(r.get("MUDIF")))) < 2e-5

    per = read_fitres(f"{_DIR}/SALT2mu_idsurvey_select.FITRES")
    want = dict(zip([str(c) for c in per.get("CID")],
                    np.asarray(per.get("IDSAMPLE"), dtype=int)))
    got = dict(zip([str(c) for c in tv.name[accept]],
                   np.asarray(out.idsample, dtype=int)))
    assert got == want


def test_chi2max_pass_starts_logmass_cen_at_parval():
    """applyCut_chi2max evaluates the FCN at INPUTS.parval, whose logmass_cen is
    10 (set_defaults, SALT2mu.c:6708), so leaving p7 out cuts as p7=10 does."""
    keys = [f"datafile={_DIR}/DATA_MASS.FITRES", "p5=0.1", "p8=0.01",
            "chi2max=4"] + _BASE
    unset = prepare_cell(parse_bbc_input(None, overrides=keys))
    given = prepare_cell(parse_bbc_input(None, overrides=keys + ["p7=10"]))
    assert (unset.tv.cutbits & (1 << CUTBIT_CHI2)).any()
    np.testing.assert_array_equal(unset.accept, given.accept)


def test_logmass_window_cuts_the_data():
    """logmass_min/logmass_max cut on the host mass as well as sizing the biasCor
    logmass grid (set_CUTMASK, SALT2mu.c:18872). With no biasCor there is nothing
    else to reject an out-of-window host, so the window is the whole difference.
    """
    keys = [f"datafile={_DIR}/DATA_MASS.FITRES"] + _BASE
    wide = run_bbc_fit(parse_bbc_input(None, overrides=keys))[1]
    _, out, _, _, _ = run_bbc_fit(parse_bbc_input(
        None, overrides=keys + ["logmass_min=9.0", "logmass_max=11.0"]))

    assert int(out.nfit.sum()) < int(wide.nfit.sum())
    r = read_fitres(f"{_DIR}/SALT2mu_logmass_cut.M0DIF")
    np.testing.assert_array_equal(out.nfit, np.asarray(r.get("NFIT")))
    assert np.max(np.abs(out.m0dif - np.asarray(r.get("MUDIF")))) < 2e-5
