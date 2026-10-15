"""Unit tests for the two-stage SALT2mu/BBC config parser."""

import logging

import numpy as np
import pytest

from jax_salt2mu.ccprior import dmu_window_max
from jax_salt2mu.config import BBCConfig, MASK_BIASCOR_SAMPLE, resolve
from jax_salt2mu.config_parser import parse_bbc_input
from jax_salt2mu.run import check_vpec_sign, run_bbc_fit
from jax_salt2mu.samples import partition_idsamples
from jax_salt2mu.tablevar import EVENT_TYPE_DATA, read_tablevar
from jax_snlc_core.models import BayeSNModel, SALTModel
from jax_salt2mu.distances import bias_parameters
from jax_snlc_core.utils.error_handling import SNANAError


def _write(tmp_path, text):
    p = tmp_path / "bbc.input"
    p.write_text(text)
    return str(p)


def test_override_beats_file_and_default_model(tmp_path):
    p = _write(tmp_path, "nzbin=20\nzmax=1.0\n")
    cfg = parse_bbc_input(p, overrides=["nzbin=5"])
    assert cfg.model_lcfit == "SALT2"          # default
    assert cfg.nzbin == 5                # CLI beats file
    assert cfg.zmax == 1.0


def test_key_and_alias_resolution(tmp_path):
    cfg = parse_bbc_input(_write(tmp_path, "bins=30\nOL=0.68\nw=-0.95\n"))
    assert cfg.nzbin == 30               # bins -> nzbin alias
    assert cfg.omega_l == 0.68          # OL -> omega_l cosmology alias
    assert cfg.w0 == -0.95


def test_nuisance_slots_and_names(tmp_path):
    """p<N>/u<N> slots resolve through the model's nuisance_spec."""
    cfg = parse_bbc_input(_write(tmp_path, "p1=0.15\nbeta0=3.1\nu1=1\nu5=0\n"))
    assert cfg.nuisance_val["alpha0"] == 0.15   # p1 slot -> name
    assert cfg.nuisance_val["beta0"] == 3.1     # named form
    assert cfg.nuisance_float["alpha0"] is True
    assert cfg.nuisance_float["gamma0"] is False


def test_u_is_three_valued_for_the_biascor_bins(tmp_path):
    """u<N>=3 floats a parameter and forces one biasCor bin in it, which is
    what a per-SN continuous beta needs (SALT2mu.c:16393); 0 holds it with
    one bin, 1 floats it on the SIM grid."""
    cfg = parse_bbc_input(_write(tmp_path, "nzbin=10\nu1=1\nu2=3\nu5=0\n"))
    assert cfg.nuisance_float["beta0"] and cfg.nuisance_onebin["beta0"]
    assert cfg.nuisance_float["alpha0"] and not cfg.nuisance_onebin["alpha0"]
    assert not cfg.nuisance_float["gamma0"] and cfg.nuisance_onebin["gamma0"]


def test_a_slot_naming_no_fit_parameter_is_refused(tmp_path):
    """SNANA reads any p<N> or u<N> into a slot and an unused one does
    nothing; here it aborts, naming the slot or the parameter."""
    with pytest.raises(SNANAError, match="no fit parameter has number 99"):
        parse_bbc_input(_write(tmp_path, "nzbin=10\np99=1\n"))
    with pytest.raises(SNANAError, match="sigint is not a parameter this fit sets"):
        parse_bbc_input(_write(tmp_path, "nzbin=10\nu14=1\n"))


def test_cosmology_slots_set_the_reference_cosmology(tmp_path):
    """Slots 9-12 are the reference cosmology, not nuisance parameters, so they
    land on the config.

    Deliberately uses values other than the defaults: p9/p11 were silently dropped
    for a long time, and every fixture passes p9=0.7 p11=-1, which are exactly the
    defaults -- so the keys did nothing while every parity test passed.
    """
    cfg = parse_bbc_input(_write(tmp_path, "p9=0.6\np11=-0.8\n"))
    assert (cfg.omega_l, cfg.w0) == (0.6, -0.8)
    assert "Omega_L" not in cfg.nuisance_val    # not a nuisance parameter

    # the named aliases reach the same fields
    named = parse_bbc_input(_write(tmp_path, "OL=0.6\nw=-0.8\n"))
    assert (named.omega_l, named.w0) == (0.6, -0.8)


@pytest.mark.parametrize("key", ["u9", "u10", "u11", "u12"])
def test_floating_the_cosmology_aborts(tmp_path, key):
    """jax holds the reference cosmology fixed, so a request to float it must
    abort rather than be quietly ignored."""
    with pytest.raises(SNANAError, match="fixed"):
        parse_bbc_input(_write(tmp_path, f"{key}=1\n"))
    # 0 is "do not float", which is what jax does anyway
    parse_bbc_input(_write(tmp_path, f"{key}=0\n"))


def test_cut_resolution(tmp_path):
    cfg = parse_bbc_input(_write(tmp_path, "x1min=-3.0\ncmax=0.3\nCUTWIN zHD 0.02 0.6\n"))
    # model shorthands override the model window; generic CUTWIN stays generic
    assert cfg.lcpar_range["x1"] == (-3.0, None)
    assert cfg.lcpar_range["c"] == (None, 0.3)
    windows = {v: (lo, hi) for v, lo, hi in cfg.cutwin}
    assert windows["zHD"] == (0.02, 0.6)
    assert "x1" not in windows and "c" not in windows


def test_cutwin_forms(tmp_path):
    """A CUTWIN is four words wherever it comes from, a later one on the same
    variable replaces the earlier (parseLine_SELECT_VAR, SALT2mu.c:21128), and
    CUTWIN NONE is a two-word switch (:21012); the qualified forms are refused."""
    cfg = parse_bbc_input(_write(tmp_path, "CUTWIN zHD 0.02 0.6\n"),
                          overrides=["CUTWIN zHD 0.05 0.5", "CUTWIN SNRMAX1 20 1000"])
    assert cfg.cutwin == [("zHD", 0.05, 0.5), ("SNRMAX1", 20.0, 1000.0)]
    assert not cfg.cutwin_disable

    cfg = parse_bbc_input(_write(tmp_path, "CUTWIN NONE\nx1min=-2.0\n"))
    assert cfg.cutwin_disable and cfg.cutwin == []

    for form in ("CUTWIN(FITWGT0) SNRMAX1 20 1000", "CUTWIN_IDSAMPLE(0) zHD 0.1 0.6",
                 "CUTWIN_SURVEY(DES) zHD 0.1 0.6"):
        with pytest.raises(SNANAError, match="not implemented"):
            parse_bbc_input(_write(tmp_path, form + "\n"))
    with pytest.raises(SNANAError, match="two limits"):
        parse_bbc_input(_write(tmp_path, "CUTWIN zHD 0.02\n"))


def test_two_stage_model_specific_cuts(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    # thetamin is a BayeSN shorthand: it resolves only because model_lcfit is
    # read in stage 1, before cut resolution in stage 2.
    cfg = parse_bbc_input(_write(tmp_path, "model_lcfit=BAYESN\nthetamin=-2.0\nx1min=-3.0\n"))
    assert cfg.lcpar_range["THETA"] == (-2.0, None)
    # a SALT shorthand is unknown under BayeSN and warned, not silently applied.
    assert "x1" not in cfg.lcpar_range
    assert any("x1min" in r.message for r in caplog.records)


def test_wired_keys_land_without_warning(tmp_path, caplog):
    """The biasCor/CC-prior/error-model keys the engine reads are parsed onto the
    config and none is flagged unrecognized. surveygroup_biascor carries internal
    '=' and '()' that must survive tokenization."""
    caplog.set_level(logging.WARNING)
    cfg = parse_bbc_input(_write(tmp_path,
        "surveygroup_biascor=LSST(zbin=0.15:x1bin=2.0:cbin=0.2)\n"
        "fieldgroup_biascor=DEEP\n"
        "min_per_cell_biascor=5 snrmin_sigint_biascor=80 sigint_biascor=0.1\n"
        "nbinc_mucovscale=4 nbin_logmass=2 interp_biascor_logmass=1\n"
        "varname_pIa=PROBIA force_pIa=perfect\n"
        "nzbin_ccprior=8 nsbin_ccprior=1 ncbin_ccprior=2\n"
        "typeIa_ccprior=120 type_list_probcc0=119,20 idsurvey_list_probcc0=12,53\n"
        "sigint_step1=0.02 lensing_zpar=0.055\n"
        "p13=1.2 u13=1\n"))
    b = cc = e = cfg
    assert b.surveygroup_biascor == "LSST(zbin=0.15:x1bin=2.0:cbin=0.2)"
    assert b.fieldgroup_biascor == "DEEP"
    assert (b.min_per_cell_biascor, b.snrmin_sigint_biascor, b.sigint_biascor) == (5, 80.0, 0.1)
    assert (b.mucovscale_bins["c"], b.nbin_logmass, b.interp_biascor_logmass) == (4, 2, 1)
    assert cc.varname_pia == "PROBIA" and cc.force_pia == -1.0     # 'perfect'
    assert (cc.nzbin_ccprior, cc.ccprior_bins["x1"], cc.ccprior_bins["c"]) == (8, 1, 2)
    assert cc.typeIa_ccprior == 120
    assert cc.type_list_probcc0 == [119, 20]
    assert cc.idsurvey_list_probcc0 == [12, 53]
    assert (e.sigint_step1, e.lensing_zpar) == (0.02, 0.055)
    assert cfg.nuisance_val["scalePCC"] == 1.2             # p13 core param
    assert cfg.nuisance_float["scalePCC"] is True          # u13
    assert not any("nrecognized" in r.message for r in caplog.records)


def test_h11_ccprior(tmp_path):
    """simfile_ccprior=H11 selects the analytic prior (no sim file); the H11 poly
    slots 17-22 resolve to their names."""
    cfg = parse_bbc_input(_write(tmp_path, "simfile_ccprior=H11\np17=0.02\nu20=1\n"))
    assert cfg.use_h11 is True
    assert cfg.simfile_ccprior == []
    assert cfg.nuisance_val["H11mucc0"] == 0.02
    assert cfg.nuisance_float["H11sigcc0"] is True


def test_sigint_aliases(tmp_path):
    """sig1 (== SNANA sigmB) maps to errmodel.sigint. Its siblings sig2/sig3 alias
    sigx1/sigc, which SALT2mu parses and never reads, so they abort instead."""
    assert parse_bbc_input(_write(tmp_path, "sig1=0.1\n")).sigint == 0.1
    with pytest.raises(SNANAError, match="intrinsic stretch/colour scatter"):
        parse_bbc_input(_write(tmp_path, "sig1=0.1\nsig2=0.2\n"))
    # SNANA's sigint= feeds a fit slot it pins and never reads; here it would
    # silently have set the scatter
    with pytest.raises(SNANAError, match="sigint: .*sig1="):
        parse_bbc_input(_write(tmp_path, "sigint=0.1\n"))


@pytest.mark.parametrize("key,value", [
    ("uM0", 3),              # not a SNANA option at all
    ("opt_ccprior", 2),      # INTERP
    ("opt_ccprior", 128),    # LEGACY
    ("datafile_override", "x.FITRES"),
])
def test_unimplemented_keys_raise(tmp_path, key, value):
    """Keys whose non-default values jax does not implement must abort rather than
    silently fall back to the default and change the fit."""
    with pytest.raises(SNANAError):
        parse_bbc_input(_write(tmp_path, f"{key}={value}\n"))


@pytest.mark.parametrize("text", ["uM0=0\n", "uM0=1\n", "uM0=2\n",
                                  "opt_ccprior=5\n", ""])
def test_supported_values_accepted(tmp_path, text):
    parse_bbc_input(_write(tmp_path, text))


def test_sigint_mode_keys(tmp_path):
    """The jax-only sigint mode and its joint-mode bounds are configurable."""
    cfg = parse_bbc_input(_write(
        tmp_path, "sigint_mode=joint\nsigint_floor=0.05\nsigint_ceiling=1.5\n"))
    assert cfg.sigint_mode == "joint"
    assert (cfg.sigint_floor, cfg.sigint_ceiling) == (0.05, 1.5)
    with pytest.raises(SNANAError):
        parse_bbc_input(_write(tmp_path, "sigint_mode=nonsense\n"))


def test_valueless_key_does_not_become_one(tmp_path, caplog):
    """A truncated line ("nzbin" where "nzbin=10" was meant) must not be read as
    nzbin=1: the key keeps its default and the line is reported."""
    with caplog.at_level(logging.WARNING):
        cfg = parse_bbc_input(_write(tmp_path, "zmax=0.9\nnzbin\n"))
    assert cfg.nzbin == 0            # default, not 1
    assert cfg.zmax == 0.9           # the well-formed line still applies
    assert "no value" in caplog.text and "nzbin" in caplog.text


def test_valueless_and_unknown_reported_separately(tmp_path, caplog):
    """A known key given no value is malformed, not unrecognized; the two get
    distinct messages so an input file can be debugged."""
    with caplog.at_level(logging.WARNING):
        parse_bbc_input(_write(tmp_path, "prefix=\nnot_a_real_key=3\n"))
    unrecognized = [r.message for r in caplog.records if "Unrecognized" in r.message]
    valueless = [r.message for r in caplog.records if "no value" in r.message]
    assert any("not_a_real_key" in m for m in unrecognized)
    assert not any("prefix" in m for m in unrecognized)
    assert any("prefix" in m for m in valueless)


def test_blindpar_slots(tmp_path):
    """blindpar is indexed per fit-parameter slot; unset slots keep SNANA's
    defaults (0.06/23434 for Omega_L, 0.20/8430 for w0)."""
    cfg = parse_bbc_input(_write(tmp_path, "blindpar9=0.1,4522\n"))
    assert cfg.blindpar[9] == (0.1, 4522.0)
    assert cfg.blindpar[11] == (0.20, 8430.0)


def test_quoted_string_values(tmp_path):
    """SNANA strips quotes from string-valued keys (remove_quote), so a quoted
    surveygroup_biascor must parse to the same value as an unquoted one."""
    quoted = parse_bbc_input(_write(
        tmp_path, "surveygroup_biascor='LSST(zbin=0.15:x1bin=2.0)'\n"))
    plain = parse_bbc_input(_write(
        tmp_path, "surveygroup_biascor=LSST(zbin=0.15:x1bin=2.0)\n"))
    assert quoted.surveygroup_biascor == plain.surveygroup_biascor
    assert "'" not in quoted.surveygroup_biascor


@pytest.mark.parametrize("text,key", [
    ("nzbin='10'\n", "nzbin"),      # quotes are not stripped for numeric keys
    ("nzbin=abc\n", "nzbin"),
    ("zmin=x.y\n", "zmin"),
])
def test_uncastable_value_aborts_naming_the_key(tmp_path, text, key):
    """SNANA's sscanf leaves the default and says nothing; jax aborts instead, so a
    malformed value cannot silently change the fit."""
    with pytest.raises(SNANAError) as exc:
        parse_bbc_input(_write(tmp_path, text))
    assert key in str(exc.value) and "not a valid" in str(exc.value)


def test_sigint_fix_parses_to_tuple(tmp_path):
    """One value or a comma list, both landing as floats."""
    assert parse_bbc_input(_write(tmp_path, "sigint_fix=0.11\n")).sigint_fix \
        == (0.11,)
    assert parse_bbc_input(
        _write(tmp_path, "sigint_fix=0.106,0.988,0.904\n")).sigint_fix \
        == (0.106, 0.988, 0.904)
    assert parse_bbc_input(_write(tmp_path, "")).sigint_fix == ()


def test_sigmb_aliases_sigint(tmp_path):
    """sig1/sigmb/sigmB all set INPUTS.sigmB in SNANA, which jax calls sigint."""
    for key in ("sig1", "sigmb", "sigmB"):
        assert parse_bbc_input(_write(tmp_path, f"{key}=0.11\n")).sigint == 0.11


@pytest.mark.parametrize("key,legacy,later,expected", [
    ("cat_file_out", "catfile_out", "b.fitres", "b.fitres"),
    ("ndump_nobiascor", "dumpflag_nobiascor", "4", 4),
    ("prescale_simdata", "simdata_prescale", "5", 5.0),
    ("sntype_select", "sntype", "101", [101]),
])
def test_two_spellings_of_one_key_take_the_later(tmp_path, key, legacy, later,
                                                 expected):
    """SALT2mu reads both spellings into one INPUTS variable, item by item, so
    the one given later wins (SALT2mu.c:19473, :20062, :20090, :19902)."""
    for first, second in ((key, legacy), (legacy, key)):
        cfg = parse_bbc_input(_write(
            tmp_path, f"nzbin=10\n{first}=3\n{second}={later}\n"))
        assert getattr(cfg, key) == expected


def test_command_line_beats_both_spellings_in_the_file(tmp_path):
    """SALT2mu reads the command line after the file (override_parFile,
    SALT2mu.c:19380), so a command-line value of the earlier file spelling
    still wins (:19473, :19477)."""
    path = _write(tmp_path, "nzbin=10\ncatfile_out=f1\ncat_file_out=f2\n")
    assert parse_bbc_input(path, ["catfile_out=cli"]).cat_file_out == "cli"
    assert parse_bbc_input(path, ["cat_file_out=cli"]).cat_file_out == "cli"


def test_survey_group_has_two_spellings(tmp_path):
    """surveygroup_biascor= and surveyGroup_biascor= set one key (SALT2mu.c:19666-19667)."""
    group = "LSST(zbin=0.15:x1bin=2.0:cbin=0.2)"
    for key in ("surveygroup_biascor", "surveyGroup_biascor"):
        cfg = parse_bbc_input(_write(tmp_path, f"nzbin=10\n{key}={group}\n"))
        assert cfg.surveygroup_biascor == group
        assert parse_bbc_input(None, ["nzbin=10", f"{key}={group}"]).surveygroup_biascor == group


def test_photoz_idsample_keys(tmp_path):
    """zspec_errmax_idsample gates the zSPEC/zPHOT split; zphot_shift offsets the
    photo-z subset. Both default to 0, which is "no split, no shift"."""
    cfg = parse_bbc_input(
        _write(tmp_path, "zspec_errmax_idsample=0.002\nzphot_shift=0.01\n"))
    assert cfg.zspec_errmax_idsample == 0.002
    assert cfg.zphot_shift == 0.01
    empty = parse_bbc_input(_write(tmp_path, ""))
    assert empty.zspec_errmax_idsample == 0.0
    assert empty.zphot_shift == 0.0


def test_zbin_forms_are_mutually_exclusive(tmp_path):
    """Exactly one of nzbin/nlogzbin/zbinuser (SALT2mu.c:21951)."""
    for text in ("nzbin=10\nnlogzbin=8\n", "nzbin=10\nzbinuser=0.1,0.5,1.0\n"):
        with pytest.raises(SNANAError, match="redshift bins"):
            parse_bbc_input(_write(tmp_path, text))


def test_varname_gamma_and_scalepcc_by_name(tmp_path):
    """varname_gamma picks the mass-step column; scalePCC is settable by name as
    well as via the p13 slot (SALT2mu.c:19970)."""
    cfg = parse_bbc_input(_write(tmp_path, "nzbin=10\nvarname_gamma=HOSTGAL_LOGMASS\n"
                                           "scalePCC=0.5\n"))
    assert cfg.varname_gamma == "HOSTGAL_LOGMASS"
    assert cfg.nuisance_val["scalePCC"] == 0.5


def test_scalar_knob_keys(tmp_path):
    """Scalar knobs that reach the fit: the fixed secant slope, the z-cheat, and
    forcing sim input to count as real data."""
    cfg = parse_bbc_input(_write(tmp_path, "nzbin=10\ndchi2red_dsigint=2.5\nuzsim=1\n"
                                           "force_realdata=1\nuave=0\n"))
    assert cfg.dchi2red_dsigint == 2.5
    assert cfg.uzsim and cfg.force_realdata
    assert cfg.uave is False


def test_select_trueIa_disables_cc_machinery(tmp_path):
    """prep_input_trueIa (SALT2mu.c:22539): reject every CC event via the >9999
    prescale, and drop the CC prior/classifier with it."""
    cfg = parse_bbc_input(_write(tmp_path, "nzbin=10\nselect_trueIa=1\n"
                                           "varname_pIa=PROBIA\n"))
    assert cfg.prescale_simcc > 9999.0
    assert cfg.varname_pia == "" and not cfg.simfile_ccprior


def test_biascor_survey_keys(tmp_path):
    """surveylist_nobiascor exempts surveys from needing a biasCor; NONE is SNANA's
    'unset' sentinel. End-to-end parity is untested: no fixture gives an exempted
    survey a valid error budget, and SNANA itself aborts in calc_zM0_data on the
    synthetic two-survey split."""
    cfg = parse_bbc_input(_write(tmp_path, "nzbin=10\nsurveylist_nobiascor=SDSS,PS1MD\n"
                                           "surveygroup_biascor_abortflag=1\n"
                                           "frac_warn_nobiascor=0.10\n"))
    assert cfg.surveylist_nobiascor == ["SDSS", "PS1MD"]
    assert cfg.surveygroup_biascor_abortflag is True
    assert cfg.frac_warn_nobiascor == 0.10
    assert parse_bbc_input(_write(tmp_path, "nzbin=10\nsurveylist_nobiascor=NONE\n")
                           ).surveylist_nobiascor == []


def test_prescale_sim1a_alias_and_model_keys(tmp_path):
    """prescale_sim1a is SNANA's older spelling of prescale_simIa. The model's own
    BBC keys land in model_inputs as given (x1 and c thresholds are covered
    end-to-end in test_bbc_select)."""
    cfg = parse_bbc_input(_write(tmp_path, "nzbin=10\nprescale_sim1a=3\n"
                                           "maxerr_abort_x0=1e-8\n"))
    assert cfg.prescale_simia == 3.0
    assert cfg._model_inputs == {"maxerr_abort_x0": 1e-8}


def test_diagnostic_keys_are_wired(tmp_path):
    """check_duplicates_biascor and ndump_nobiascor reach the config; both drive
    reporting only, so there is no fit-level assertion to make."""
    cfg = parse_bbc_input(_write(tmp_path, "nzbin=10\ncheck_duplicates_biascor=1\n"
                                           "ndump_nobiascor=5\n"))
    assert cfg.check_duplicates_biascor is True
    assert cfg.ndump_nobiascor == 5
    # dumpflag_nobiascor is the older spelling of the same knob
    assert parse_bbc_input(_write(tmp_path, "nzbin=10\ndumpflag_nobiascor=4\n")
                           ).ndump_nobiascor == 4


def test_surveygroup_abortflag_rejects_empty_group():
    """surveygroup_biascor_abortflag aborts when a requested survey has no data
    (SALT2mu.c:10479). Without the flag the empty group is simply absent."""
    cfg = BBCConfig()
    cfg.opt_biascor = MASK_BIASCOR_SAMPLE
    cfg.surveygroup_biascor = "LSST,PS1MD"     # no PS1MD events below
    ids, z = np.array([12] * 10), np.full(10, 0.3)
    cfg.surveygroup_biascor_abortflag = True
    with pytest.raises(SNANAError, match="PS1MD"):
        partition_idsamples(ids, None, z, np.zeros(10, bool), cfg)
    cfg.surveygroup_biascor_abortflag = False
    _, table = partition_idsamples(ids, None, z, np.zeros(10, bool), cfg)
    assert [t.surveygroup for t in table] == ["LSST"]


@pytest.mark.parametrize("key,reason", [
    # never read by SALT2mu either -- setting it does nothing there or here
    ("sigc=0.1", "intrinsic stretch/colour scatter"),
    ("sig2=0.1", "intrinsic stretch/colour scatter"),
    ("xi01=0.5", "intrinsic stretch/colour scatter"),
    # SALT2mu refuses these itself
    ("pecv=300", "zpecerr"),
    ("host_logmass_split=10", "p7"),
    # live in SALT2mu, unimplemented here
    ("ZPOLY_sigmB=10 0.1 0 0 0", "intrinsic-scatter covariance"),
    ("minos=1", "MINUIT profile uncertainties"),
    ("restore_sigz=1", "restore earlier SNANA calculations"),
    ("cutwin_only=1", "rewriting a table without standardisation"),
    ("blindpar=1", "blindpar9"),
    ("opt_chi2max=2", "reapplying chi2max"),
    ("opt_chi2max=3", "reapplying chi2max"),
])
def test_unimplemented_keys_abort(tmp_path, key, reason):
    """A SALT2mu key jax does not act on aborts rather than being silently
    dropped, from the file or from the command line."""
    with pytest.raises(SNANAError, match=reason):
        parse_bbc_input(_write(tmp_path, f"nzbin=10\n{key}\n"))
    with pytest.raises(SNANAError, match=reason):
        parse_bbc_input(_write(tmp_path, "nzbin=10\n"), [key])


@pytest.mark.parametrize("key", ["blindpar9=0.1,4522", "u20=1", "sig1=0.1"])
def test_supported_keys_are_not_caught_by_the_reject_list(tmp_path, key):
    """The reject list matches whole keys: "blindpar9" contains "blindpar" and
    "H11sigcc0" (u20) contains "sigc", and both are supported."""
    parse_bbc_input(_write(tmp_path, f"nzbin=10\n{key}\n"))


def test_unknown_key_still_only_warns(tmp_path, caplog):
    """A key SALT2mu does not know is a typo; an input file carrying one still runs."""
    with caplog.at_level(logging.WARNING):
        parse_bbc_input(_write(tmp_path, "nzbin=10\nnosuchkey=3\n"))
    assert any("nosuchkey" in r.message for r in caplog.records)


def test_unknown_command_line_key_aborts(tmp_path):
    """override_parFile (SALT2mu.c:19428) aborts on a command-line item ppar
    does not recognise."""
    path = _write(tmp_path, "nzbin=10\n")
    for bad in (["nosuchkey=3"], ["typo"]):
        with pytest.raises(SNANAError, match="Invalid command-line override"):
            parse_bbc_input(path, overrides=bad)
    cfg = parse_bbc_input(path, overrides=["CUTWIN", "x1", "-2", "2"])
    assert ("x1", -2.0, 2.0) in cfg.cutwin


def test_repeated_key_aborts(tmp_path):
    """uniqueOverlap aborts on a key repeated within the file or within the
    command line (SALT2mu.c:19334, :19402); the command line may still
    override the file, and chi2max keys may repeat (:19810)."""
    with pytest.raises(SNANAError, match="more than once: nzbin"):
        parse_bbc_input(_write(tmp_path, "nzbin=10\nnzbin=12\n"))
    path = _write(tmp_path, "nzbin=10\n")
    with pytest.raises(SNANAError, match="more than once: zmax"):
        parse_bbc_input(path, overrides=["zmax=0.9", "zmax=1.0"])
    assert parse_bbc_input(path, overrides=["nzbin=12"]).nzbin == 12
    assert parse_bbc_input(_write(tmp_path, "nzbin=10\nchi2max=9\nchi2max=16\n")
                           ).chi2max == 16


def test_dmubin_ccprior_truncates_the_dmu_window():
    """The CC-prior dmu axis spans [-3,+4] with nbin=int(7/width) (SALT2mu.c:17290),
    so a width that does not divide 7 leaves the remainder uncovered -- and those
    events out of DMUAVG/DMURMS, which is the whole effect of this key."""
    assert dmu_window_max(0.5) == pytest.approx(4.0)      # SNANA's default: exact
    assert dmu_window_max(0.25) == pytest.approx(4.0)
    assert dmu_window_max(0.6) == pytest.approx(3.6)      # 11 bins, top 0.4 dropped


_VPEC_DIR = "tests/fixtures/snana/bbc_vpec_sign"


def _vpec_reference():
    """SALT2mu.exe's own check_vpec_sign output: exit code and the two RMS values."""
    rows = {}
    for line in open(f"{_VPEC_DIR}/SALT2mu_vpec_sign.txt"):
        if line.startswith("#"):
            continue
        name, code, nominal, flip = line.split()
        rows[name] = (int(code), float(nominal), float(flip))
    return rows


def _run_vpec(name, extra=()):
    # These guard fixtures have x1=c=0 throughout; they cannot constrain the
    # brightness corrections alpha/beta. Fit only the bin offsets here.
    ov = [f"datafile={_VPEC_DIR}/{name}.FITRES", "nzbin=2", "zmin=0.015",
          "zmax=0.05", "u1=0", "u2=0", *extra]
    return run_bbc_fit(parse_bbc_input(None, overrides=ov))


@pytest.mark.parametrize("name", ["vpec_ok", "vpec_bad"])
def test_vpec_sign_rms_matches_salt2mu(name):
    """The two RMS values check_vpec_sign measures, against SALT2mu.exe's own.

    Compared at the 4 decimals SNANA prints. Both tables hold the same supernovae;
    only zHD differs, by the sign of the peculiar-velocity correction.
    """

    cfg = resolve(parse_bbc_input(None, overrides=[
        f"datafile={_VPEC_DIR}/{name}.FITRES", "nzbin=2", "zmin=0.015",
        "zmax=0.05", "u1=1", "u2=1"]), SALTModel)
    tv = read_tablevar(cfg.datafile, EVENT_TYPE_DATA, cfg, SALTModel)

    code, want_nominal, want_flip = _vpec_reference()[name]
    if code == 0:
        nsn, nominal, flip = check_vpec_sign(tv, cfg, SALTModel)
    else:
        with pytest.raises(SNANAError, match="sign-flip of vpec"):
            check_vpec_sign(tv, cfg, SALTModel)
        return
    assert nsn == 120
    assert nominal == pytest.approx(want_nominal, abs=5e-5)
    assert flip == pytest.approx(want_flip, abs=5e-5)


def test_vpec_sign_guard_accepts_consistent_data():
    """check_vpec_sign (SALT2mu.c:4131) runs by default and passes clean input."""
    _run_vpec("vpec_ok")


def test_vpec_sign_guard_aborts_on_inverted_vpec():
    """SALT2mu exits 12 on this table; jax raises rather than fitting biased
    low-z distances."""
    with pytest.raises(SNANAError, match="sign-flip of vpec"):
        _run_vpec("vpec_bad")


def test_vpec_sign_guard_is_disabled_by_a_zero_window():
    """zwin_vpec_check=0,0 switches the check off (SALT2mu.c:4160)."""
    _run_vpec("vpec_bad", extra=["zwin_vpec_check=0,0"])


def test_vpec_sign_guard_needs_70_events():
    """Below SNANA's 70-event floor (:4236) the RMS is not measured at all, so a
    narrow window returns silently even on inverted data."""
    _run_vpec("vpec_bad", extra=["zwin_vpec_check=0.02,0.021"])


_BC = "tests/fixtures/snana/bbc_des_beams"
_OPT_KEYS = [f"datafile={_BC}/DATA.FITRES", f"simfile_biascor={_BC}/BIASCOR.FITRES",
             "nzbin=10", "zmin=0.1", "zmax=1.0"]


def _resolved(overrides):
    """The config as run_bbc_fit sees it: opt_biascor is validated in resolve,
    where SNANA validates it (prepare_biasCor)."""
    return resolve(parse_bbc_input(None, overrides=overrides), SALTModel)


@pytest.mark.parametrize("bit,phrase", [
    (8, "test option"),                 # 1D biasCor with the 5D grid cut
    (256, "scale_covint"),              # empirical 3x3 COVINT
    (512, "per IDSAMPLE"),              # sigint(biasCor) per IDSAMPLE
    (1024, "CUTBIT_BIASCOR"),           # keep events with no valid biasCor
])
def test_an_unimplemented_opt_biascor_bit_aborts_with_its_reason(bit, phrase):
    """These bits change the fit in SNANA (print_biascor_options,
    SALT2mu.c:11642) and do nothing here, which the rejected-key table cannot
    catch because it covers key names, not key values."""
    with pytest.raises(SNANAError, match=phrase):
        _resolved(_OPT_KEYS + [f"opt_biascor={16 + bit}"])


def test_opt_biascor_refuses_a_1d_and_5d_mixture():
    """prepare_biasCor aborts on "Cannot do 1D and 5D biasCor" (SALT2mu.c:11414)."""
    with pytest.raises(SNANAError, match="1D and 5D"):
        _resolved(_OPT_KEYS + ["opt_biascor=17"])


def test_opt_biascor_refuses_neither_1d_nor_5d():
    """With no dimensionality bit SNANA aborts "Invalid opt_biascor" (:11408)."""
    with pytest.raises(SNANAError, match="neither a 1D nor a 5D"):
        _resolved(_OPT_KEYS + ["opt_biascor=32"])


def test_opt_biascor_ignores_a_bit_salt2mu_does_not_define():
    """prepare_biasCor tests the bits it knows and no others, so a bit above the
    defined set changes nothing there and must change nothing here."""
    plain = _resolved(_OPT_KEYS + ["opt_biascor=112"])
    extra = _resolved(_OPT_KEYS + ["opt_biascor=8304"])   # 112 + 8192
    assert extra.opt_biascor == plain.opt_biascor | 8192


def test_bias_on_mu_names_one_bias_parameter():
    """Bit 128 puts the bias on mu alone (ILCPAR_MIN/MAX, SALT2mu.c:11315);
    BayeSN's only LC parameter is already the distance, so its bias vector is
    the same either way."""
    assert bias_parameters(SALTModel, _resolved(_OPT_KEYS + ["opt_biascor=144"])) == ("mu",)
    assert bias_parameters(SALTModel, _resolved(_OPT_KEYS + ["opt_biascor=112"])) == ("mB", "x1", "c")
    assert bias_parameters(BayeSNModel, _resolved(_OPT_KEYS + ["opt_biascor=144"])) == ("MU",)


def test_alpha1_is_refused_because_its_meaning_changed_upstream():
    """alpha1= has pointed at different slots in different SALT2mu releases,
    so it is refused; dalpha_dz=, p3= and u3= set the redshift slope, and the
    older beta1=/alphaHost=/betaHost= spellings still reach their slots
    (SALT2mu.c:19960-19977)."""
    with pytest.raises(SNANAError, match="alpha1"):
        parse_bbc_input(None, overrides=["nzbin=4", "zmin=0.1", "zmax=1.0",
                                         "alpha1=0.1"])
    cfg = parse_bbc_input(None, overrides=["nzbin=4", "zmin=0.1", "zmax=1.0",
                                           "p3=0.1", "u3=1", "beta1=0.2",
                                           "alphaHost=0.03"])
    assert cfg.nuisance_val["dalpha_dz"] == 0.1
    assert cfg.nuisance_float["dalpha_dz"]
    assert cfg.nuisance_val["dbeta_dz"] == 0.2
    assert cfg.nuisance_val["dalpha_dlogm"] == 0.03
