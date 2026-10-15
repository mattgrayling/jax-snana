#!/usr/bin/env python
"""Generate the SALT2mu.exe references for this fixture, one run per CONFIGS entry.

    python refs.py [--check] [NAME ...]

Rerun after build.py. Runs take ~1 s each. SALT2mu.exe runs from a short temp dir:
a prefix of 116+ characters overruns `char PREFIX[100]` (SALT2mu.c:1390) onto
`frac_warn_nobiasCor`.
"""


import argparse
import gzip
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from build import BIASCOR_VARIANTS, build

HERE = Path(__file__).resolve().parent

# keys passed as argv instead of in the input file, for SALT2mu's one-per-source limits
ARGV = "__argv__"

BASE = {
    "datafile": "DATA.FITRES",
    "simfile_biascor": "BIASCOR.FITRES",
    "opt_biascor": "112",
    "min_per_cell_biascor": "3",
    "snrmin_sigint_biascor": "60",
    "surveygroup_biascor": "DES(zbin=0.15:x1bin=2.0:cbin=0.2)",
    "nzbin": "10", "zmin": "0.1", "zmax": "1.0",
    "u1": "1", "u2": "1", "p9": "0.7", "p11": "-1",
}

# blindflag: CONTAM_* have no SIM columns, so SALT2mu blinds them by default.
# typeIa_ccprior=999: the absent TYPE column reads -9 == the key's default, which
# would force pIa=1 everywhere (SALT2mu.c:22789). p13 is scalePCC's value, 0 by
# default, which turns the CC prior off.
BEAMS = {**BASE, "datafile": "CONTAM_DATA.FITRES",
         "simfile_ccprior": "CCPRIOR.FITRES", "opt_ccprior": "5",
         "varname_pIa": "PROBIA", "p13": "1",
         "typeIa_ccprior": "999", "blindflag": "0"}

# selection / blinding / z-binning keys run with no biasCor, matching the test files
NOBC = {k: v for k, v in BASE.items()
        if k not in ("simfile_biascor", "opt_biascor", "min_per_cell_biascor",
                     "snrmin_sigint_biascor", "surveygroup_biascor")}
NOBC["blindflag"] = "0"


def _plain(**keys):
    """The bare fit test_bbc_nobiascor runs: no biasCor. p9/p11 are given because
    SNANA defaults them to 0 while jax defaults to the reference cosmology."""
    return {"datafile": "DATA.FITRES", "nzbin": "10", "zmin": "0.1", "zmax": "1.0",
            "u1": "1", "u2": "1", "p9": "0.7", "p11": "-1", **keys}


def _sel(datafile="DATA.FITRES", drop_nzbin=False, **keys):
    cfg = {**NOBC, "datafile": datafile, **keys}
    if drop_nzbin:
        del cfg["nzbin"]
    return cfg


CONFIGS = {
    "SALT2mu":            (BASE, ["M0DIF", "FITRES", "COV"]),
    "SALT2mu_mad":        ({**BASE, "opt_biascor": "2160"}, ["M0DIF"]),
    # spec-z-only data with a half-photo-z biasCor: the photo-z rows match
    # the surveygroup but no data IDSAMPLE (get_IDSAMPLE returns -9,
    # SALT2mu.c:11252) and fail the IDSAMPLE cut (:18957-18958), keeping
    # them out of sigInt and zM0
    "SALT2mu_biascor_stray": ({**BASE,
                               "simfile_biascor": "BIASCOR_PHOTOZ.FITRES",
                               "zspec_errmax_idsample": "0.05",
                               "blindflag": "0"},
                              ["M0DIF", "FITRES"]),
    "SALT2mu_covadd":     ({**BASE, "opt_biascor": "4208"}, ["M0DIF", "FITRES"]),
    # muCOVADD without the muCOVscale bit: only that bit builds the map
    # (SALT2mu.c:11532), so neither correction is applied
    "SALT2mu_covadd_nomucov": ({**BASE, "opt_biascor": "4176"},
                               ["M0DIF", "FITRES"]),
    "SALT2mu_sigmb":      ({**BASE, "fitflag_sigmb": "1"}, ["M0DIF"]),
    "SALT2mu_secant":     ({**BASE, "fitflag_sigmb": "1", "sigint_step1": "0.05"},
                           ["M0DIF"]),
    "SALT2mu_1dzavg":     ({**BASE, "opt_biascor": "1", "surveygroup_biascor": ""},
                           ["M0DIF"]),
    "SALT2mu_1dzwgt":     ({**BASE, "opt_biascor": "2", "surveygroup_biascor": ""},
                           ["M0DIF"]),
    # a 1D biasCor rewrites fitflag_sigmb=2 down to 1, dropping the log(sigma) term
    # the key asks for (SALT2mu.c:11445)
    "SALT2mu_1d_nolog":   ({**BASE, "opt_biascor": "1", "surveygroup_biascor": "",
                            "fitflag_sigmb": "2"}, ["M0DIF"]),
    # cmin/cmax/x1min/x1max set the biasCor cell ranges as well as the cut
    # (SALT2mu.c:6987 muCOVscale colour axis, :17297 CC-prior axes), so narrowing
    # one moves which cell an event lands in. The colour pair shows up without a CC
    # prior; the x1 pair only through the CC-prior shape axis, hence the last entry.
    "SALT2mu_cmin":       ({**BASE, "cmin": "-0.2"}, ["M0DIF"]),
    "SALT2mu_cmax":       ({**BASE, "cmax": "0.2"}, ["M0DIF"]),
    # windows wider than the biasCor grid (x1 +-4, c -0.3..0.5), so events
    # between window and grid face outside_biasCor_grid (SALT2mu.c:19239)
    "SALT2mu_gridcut":    ({**BASE, "x1min": "-6", "x1max": "6",
                            "cmin": "-1", "cmax": "1"}, ["M0DIF", "FITRES"]),
    # u13=2 floats scalePIa as the H11 Eq. 4 Bayes factor (SALT2mu.c:5530)
    "SALT2mu_u13_bayes":  ({**BEAMS, "u13": "2"}, ["M0DIF", "FITRES"]),
    # sim data with truth and a spec-typed subset: CONTAM_TRUE plus the
    # NSNSPEC_IA subtraction from SUM_PROBIa (SALT2mu.c:24968-24986)
    "SALT2mu_contam_true": ({**BEAMS, "datafile": "CONTAM_SIMTYPE.FITRES",
                             "typeIa_ccprior": "1", "write_yaml": "1"},
                            ["M0DIF", "FITRES", "YAML"]),
    # every TYPE in the file is spec-confirmed, so the contamination ratios
    # are 0/0 and print as nan while the run completes
    "SALT2mu_contam_allspec": ({**BEAMS, "datafile": "CONTAM_SIMTYPE.FITRES",
                                "typeIa_ccprior": "1",
                                "type_list_probcc0": "0,101,102",
                                "write_yaml": "1"},
                               ["M0DIF", "FITRES", "YAML"]),
    # deweight (MUERR -> 8888.8) instead of reject (SALT2mu.c:3491)
    "SALT2mu_chi2max_fitwgt0": ({**BASE, "chi2max(FITWGT0)": "8"},
                                ["M0DIF", "FITRES"]),
    # per-survey threshold on the two-survey sample; the bare key before the
    # survey form seeds the unlisted surveys (SALT2mu.c:20510)
    "SALT2mu_chi2max_survey": (
        {**BASE, "datafile": "DATA_SDSS.FITRES",
         "simfile_biascor": "BIASCOR_SDSS.FITRES",
         "surveygroup_biascor": "DES(zbin=0.15:x1bin=2.0:cbin=0.2),SDSS(zbin=0.15:x1bin=2.0:cbin=0.2)",
         "chi2max": "16", "chi2max(SDSS)": "6"}, ["M0DIF"]),
    # field samples on the C3, X3, C3+X3, E2 cycle: exact field names, where the
    # overlap C3+X3 and E2 match no group and join the survey's first field
    # sample (get_IDSAMPLE on SURVEYGROUP, SALT2mu.c:11215-11216); and the
    # substring form with |C3| exact, where C3+X3 goes to X3
    "SALT2mu_fieldlist": (
        {**BASE, "datafile": "DATA_FIELD.FITRES",
         "simfile_biascor": "BIASCOR_FIELD.FITRES",
         "fieldlist_biascor": "C3,X3"}, ["M0DIF", "FITRES"]),
    "SALT2mu_fieldgroup_exact": (
        {**BASE, "datafile": "DATA_FIELD.FITRES",
         "simfile_biascor": "BIASCOR_FIELD.FITRES",
         "fieldgroup_biascor": "|C3|,X3"}, ["M0DIF", "FITRES"]),
    "SALT2mu_x1min":      ({**BASE, "x1min": "-2.0"}, ["M0DIF"]),
    "SALT2mu_x1max":      ({**BASE, "x1max": "2.0"}, ["M0DIF"]),
    "SALT2mu_x1min_cc":   ({**BEAMS, "nsbin_ccprior": "3", "x1min": "-2.0"},
                           ["M0DIF"]),
    "SALT2mu_x1max_cc":   ({**BEAMS, "nsbin_ccprior": "3", "x1max": "2.0"},
                           ["M0DIF"]),
    # uM0=0 pins every per-z-bin M0 to M0_DEFAULT +/-0.001 (:2499) while alpha and
    # beta still float -- unlike fixpar_all, which also fixes those and forces one bin.
    "SALT2mu_um0_fixed":  ({**BASE, "uM0": "0"}, ["M0DIF"]),
    # lensing_zpar enters fcn_muerrsq (:5379), so it is inside muCOVscale on
    # the data side and inside every biasCor muerrsq
    "SALT2mu_lensing":  ({**BASE, "lensing_zpar": "0.055"}, ["M0DIF"]),
    "SALT2mu_chi2max16":  ({**BASE, "chi2max": "16"}, ["M0DIF"]),
    # bit 128: the bias on mu at each biasCor event's own SIM alpha/beta, one
    # map entry per cell instead of three (SALT2mu.c:9149, :11339-11340)
    "SALT2mu_mu":         ({**BASE, "opt_biascor": "240"}, ["M0DIF", "FITRES"]),
    "SALT2mu_chi2max8":   ({**BASE, "chi2max": "8"}, ["M0DIF"]),
    # the only config here whose sigint secant lands away from zero, and the only one
    # where the chi2max cut reads the log(sigma) chi2 a 5D biasCor forces
    # fitflag_sigmb=1 into (SALT2mu.c:11445, :5506, :3471)
    "SALT2mu_chi2max_sigmb": ({**BASE, "chi2max": "16", "fitflag_sigmb": "1"},
                              ["M0DIF", "FITRES"]),
    # the same log(sigma) chi2 at a ceiling events sit against: the reference the
    # term divides by is the mu-variance of the pass before (:9468, refreshed
    # under DO_H0marg, :5024), and reading the LC-fit mB variance in its place
    # moves eight events across this ceiling where it moves none across 16
    "SALT2mu_chi2max_sigmb12": ({**BASE, "chi2max": "12", "fitflag_sigmb": "1"},
                                ["M0DIF"]),
    # MUERR_RAW and MUERR_VPEC are diagnostics carrying neither sigint nor
    # muCOVscale (SALT2mu.c:5642). BASE leaves VPECERR at 0 and its secant lands on
    # sigint=0, so this is the one config where both terms are non-zero at once.
    "SALT2mu_vpec_sigmb": ({**BASE, "vpecerr": "250", "fitflag_sigmb": "1",
                            "chi2max": "16"}, ["M0DIF", "FITRES"]),
    "SALT2mu_contam":     (BEAMS, ["M0DIF", "FITRES"]),
    # 1.5 does not divide the fixed 7-mag dmu span, so the axis stops at +3.0 and the
    # CC events above it drop out of DMUAVG/DMURMS (SALT2mu.c:17290, :17449). The
    # effect is small -- 3.6e-4 in MUDIF -- so the test tolerance has to sit under it.
    "SALT2mu_dmubin":     ({**BEAMS, "dmubin_ccprior": "1.5"}, ["M0DIF"]),
    # test_bbc_um0_interp
    "SALT2mu_um0_interp":  ({**BASE, "uM0": "2"}, ["M0DIF"]),
    "SALT2mu_muref_zpoly": ({**BASE, "muref_zpoly":
                             "36.732550,19.401037,-20.992814,9.015726"}, ["M0DIF"]),

    # test_bbc_blind -- blindflag is dropped from BASE by each entry's own value
    "SALT2mu_blind_fixpar":   ({**BASE, "blindflag": "66"}, ["M0DIF"]),
    "SALT2mu_blind_muz":      ({**BASE, "blindflag": "65"}, ["M0DIF"]),
    "SALT2mu_blind_par9":     ({**BASE, "blindflag": "66",
                                "blindpar9": "0.1,4522"}, ["M0DIF"]),
    # checkStringUnique permits one 'blindpar' key per source, so the second goes
    # on the command line
    "SALT2mu_blind_par_both": ({**BASE, "blindflag": "66", "blindpar9": "0.1,4522",
                                ARGV: ["blindpar11=0.15,207"]}, ["M0DIF"]),
    "SALT2mu_blind_real":     ({**BASE, "datafile": "DATA_REAL.FITRES",
                                "blindflag": "2"}, ["M0DIF"]),

    # test_bbc_select -- selection and z-binning keys, no biasCor
    "SALT2mu_nobc":          (_sel(), ["M0DIF", "FITRES"]),
    "SALT2mu_sntype":        (_sel("CONTAM_MIXED.FITRES", sntype_select="1"), ["M0DIF"]),
    "SALT2mu_ps_simdata":    (_sel(prescale_simdata="2"), ["M0DIF"]),
    "SALT2mu_ps_simcc":      (_sel("DATA_REAL.FITRES", prescale_simcc="2"), ["M0DIF"]),
    "SALT2mu_ps_simia":      (_sel(prescale_simIa="2"), ["M0DIF"]),
    "SALT2mu_ps_simcc_mixed": (_sel("DATA_MIXTMPL.FITRES", prescale_simcc="2"), ["M0DIF"]),
    "SALT2mu_uzsim":         (_sel(uzsim="1"), ["M0DIF"]),
    "SALT2mu_maxerr_x1":     (_sel(maxerr_abort_x1="0.5"), ["M0DIF"]),
    "SALT2mu_maxerr_c":      (_sel(maxerr_abort_c="0.05"), ["M0DIF"]),
    "SALT2mu_nmax":          (_sel(nmax="200"), ["M0DIF"]),
    "SALT2mu_powzbin":       (_sel(powzbin="2.0"), ["M0DIF"]),
    "SALT2mu_min_per_zbin":  (_sel(min_per_zbin="40"), ["M0DIF"]),
    "SALT2mu_vpecerr":       (_sel(vpecerr="250"), ["M0DIF"]),
    # a generic CUTWIN on a column BBC does not otherwise read; SALT2mu takes the
    # four words from the command line (override_parFile, SALT2mu.c:19402)
    "SALT2mu_cutwin_snrmax": (_sel(**{ARGV: ["CUTWIN", "SNRMAX1", "20", "1000"]}),
                              ["M0DIF"]),
    "SALT2mu_cutwin_probia": (_sel("CONTAM_DATA.FITRES",
                                   **{ARGV: ["CUTWIN", "PROBIA", "0.5", "1.0"]}),
                              ["M0DIF"]),
    # CUTWIN NONE lifts the x1 cut given beside it
    "SALT2mu_cutwin_none":   (_sel(x1min="-2.0", **{ARGV: ["CUTWIN", "NONE"]}),
                              ["M0DIF"]),
    "SALT2mu_nlogzbin":      (_sel(drop_nzbin=True, nlogzbin="8"), ["M0DIF"]),
    "SALT2mu_zbinuser":      (_sel(drop_nzbin=True,
                                   zbinuser="0.1,0.3,0.6,1.0"), ["M0DIF"]),
    # edges inside the data's z span: setup_BININFO_userz overwrites zmin/zmax with
    # them (SALT2mu.c:2897), so the cut follows the edges and NOT the zmin/zmax keys
    "SALT2mu_zbinuser_narrow": (_sel(drop_nzbin=True,
                                     zbinuser="0.2,0.4,0.6"), ["M0DIF"]),
    "SALT2mu_dupl_ignore":   (_sel("DATA_DUP.FITRES", force_realdata="1",
                                   iflag_duplicate="0"), ["M0DIF"]),
    "SALT2mu_dupl_merge":    (_sel("DATA_DUP.FITRES", force_realdata="1",
                                   iflag_duplicate="2"), ["M0DIF"]),
    "SALT2mu_cid_reject":    (_sel(cid_reject_file="cid_reject.list"), ["M0DIF"]),
    "SALT2mu_cid_select":    (_sel(cid_select_file="cid_select.list"), ["M0DIF"]),
    "SALT2mu_uave0":         (_sel(uave="0"), ["FITRES"]),
    "SALT2mu_zrate_scale":   (_sel(zrate_scale_file="zrate.txt"), ["M0DIF"]),
    "SALT2mu_force_realdata": ({k: v for k, v in _sel(force_realdata="1").items()
                                if k != "blindflag"}, ["M0DIF"]),
    "SALT2mu_fixpar_all":    (_sel(fixpar_all="1"), ["FITRES"]),

    # test_bbc_photoz_idsample -- zSPEC/zPHOT IDSAMPLE split
    "SALT2mu_photoz":       ({**BASE, "datafile": "DATA_PHOTOZ.FITRES",
                              "simfile_biascor": "BIASCOR_PHOTOZ.FITRES",
                              "zspec_errmax_idsample": "0.05", "blindflag": "0"},
                             ["M0DIF", "FITRES"]),
    "SALT2mu_zphot_shift":  ({**BASE, "datafile": "DATA_PHOTOZ.FITRES",
                              "simfile_biascor": "BIASCOR_PHOTOZ.FITRES",
                              "zspec_errmax_idsample": "0.05",
                              "zphot_shift": "0.01", "blindflag": "0"},
                             ["M0DIF", "FITRES"]),
    "SALT2mu_1d_photoz":    ({**BASE, "datafile": "DATA_PHOTOZ.FITRES",
                              "simfile_biascor": "BIASCOR_PHOTOZ.FITRES",
                              "opt_biascor": "65", "surveygroup_biascor": "",
                              "zspec_errmax_idsample": "0.05", "blindflag": "0"},
                             ["M0DIF"]),

    # test_bbc_fit force_probcc0: the per-event mask, read off PROBCC_BEAMS. The
    # fit itself runs into the beta bound on this sample and the tests do not use it.
    # the absent-TYPE trap: CONTAM_DATA has no TYPE column, so TYPE reads -9 ==
    # typeIa_ccprior's own default and every event is forced, disabling BEAMS
    "SALT2mu_contam_specConf": ({**BEAMS, "typeIa_ccprior": "-9"}, ["FITRES"]),
    "SALT2mu_contam_idsurvey": ({**BEAMS, "datafile": "CONTAM_MIXED.FITRES",
                                 "idsurvey_list_probcc0": "10"}, ["FITRES"]),
    "SALT2mu_contam_mixed":    ({**BEAMS, "datafile": "CONTAM_MIXED.FITRES",
                                 "typeIa_ccprior": "101",
                                 "type_list_probcc0": "102"}, ["FITRES"]),

    # test_bbc_fit mass-step cases
    "SALT2mu_gamma": ({**BASE, "datafile": "DATA_MASS.FITRES",
                       "simfile_biascor": "BIASCOR_MASS.FITRES", "u5": "1",
                       "p7": "10", "p8": "0.01"}, ["M0DIF"]),
    "SALT2mu_4d":    ({**BASE, "datafile": "DATA_MASS.FITRES",
                       "simfile_biascor": "BIASCOR_MASS.FITRES", "nbin_logmass": "2",
                       "interp_biascor_logmass": "1", "logmass_min": "8",
                       "logmass_max": "12"}, ["M0DIF"]),

    # test_bbc_sigint_fix
    "SALT2mu_sigint_fix":       ({**BASE, "fitflag_sigmb": "1",
                                  "sigint_fix": "0.11"}, ["M0DIF"]),
    "SALT2mu_sigint_fix_covadd": ({**BASE, "opt_biascor": "4208",
                                   "fitflag_sigmb": "1", "sigint_fix": "0.11"},
                                  ["M0DIF"]),
    "SALT2mu_sigint_fix_1d":    ({**BASE, "opt_biascor": "1",
                                  "surveygroup_biascor": "", "fitflag_sigmb": "1",
                                  "sigint_fix": "0.11"}, ["M0DIF"]),
    "SALT2mu_sigma_cell":       ({**BASE, "sigma_cell_biascor": "0.5"}, ["M0DIF"]),
    "SALT2mu_write_csv":        ({**BASE, "fitflag_sigmb": "0", "write_csv": "1"},
                                 ["CSV"]),
    "SALT2mu_chi2grid":         ({**BASE, "write_chi2grid": "5"}, ["CHI2GRID"]),
    "SALT2mu_sigint_fix_order_des_first": (
        {**BASE, "datafile": "DATA_SDSS.FITRES",
         "simfile_biascor": "BIASCOR_SDSS.FITRES",
         "surveygroup_biascor": "DES(zbin=0.15:x1bin=2.0:cbin=0.2),SDSS(zbin=0.15:x1bin=2.0:cbin=0.2)",
         "fitflag_sigmb": "1", "sigint_fix": "0.11,0.30"}, ["M0DIF"]),
    "SALT2mu_sigint_fix_order_sdss_first": (
        {**BASE, "datafile": "DATA_SDSS.FITRES",
         "simfile_biascor": "BIASCOR_SDSS.FITRES",
         "surveygroup_biascor": "SDSS(zbin=0.15:x1bin=2.0:cbin=0.2),DES(zbin=0.15:x1bin=2.0:cbin=0.2)",
         "fitflag_sigmb": "1", "sigint_fix": "0.11,0.30"}, ["M0DIF"]),

    # idsample_select: an unselected sample gets no biasCor (prep_IDSAMPLE_SELECT,
    # :19735), so its events fail the biasCor cut rather than an explicit one
    "SALT2mu_idsample0": ({**BASE, "datafile": "DATA_SDSS.FITRES",
                           "simfile_biascor": "BIASCOR_SDSS.FITRES",
                           "surveygroup_biascor": "SDSS(zbin=0.15:x1bin=2.0:cbin=0.2),DES(zbin=0.15:x1bin=2.0:cbin=0.2)",
                           "idsample_select": "0"}, ["M0DIF"]),
    "SALT2mu_idsample1": ({**BASE, "datafile": "DATA_SDSS.FITRES",
                           "simfile_biascor": "BIASCOR_SDSS.FITRES",
                           "surveygroup_biascor": "SDSS(zbin=0.15:x1bin=2.0:cbin=0.2),DES(zbin=0.15:x1bin=2.0:cbin=0.2)",
                           "idsample_select": "1"}, ["M0DIF"]),

    # test_bbc_nobiascor -- no biasCor and no cosmology overrides
    "SALT2mu_plain":             (_plain(), ["M0DIF"]),
    "SALT2mu_plain_chi2max16":   (_plain(chi2max="16"), ["M0DIF"]),
    "SALT2mu_plain_chi2max9":    (_plain(chi2max="9"), ["M0DIF"]),
    "SALT2mu_plain_chi2max4":    (_plain(chi2max="4"), ["M0DIF"]),
    "SALT2mu_plain_parval_in":   (_plain(p1="0.08", p2="2.0", chi2max="16"), ["M0DIF"]),
    "SALT2mu_plain_parval_clamp": (_plain(p1="0.10", p2="0.5", chi2max="16"), ["M0DIF"]),

    # scalePCC fixed at its parval of 0 turns the CC prior off (prep_input,
    # :21235), so the appended block is the plain one: no CHI2_BEAMS,
    # PROBCC_BEAMS or MUERR_RENORM
    "SALT2mu_ccprior_scalepcc0": ({k: v for k, v in BEAMS.items() if k != "p13"},
                                  ["M0DIF", "FITRES"]),

    # MUz blinding is added and the empty bins are zeroed afterwards, so an
    # unfilled bin writes MUDIF=0 next to MUDIFERR=999 (M0dif_calc, :24291)
    "SALT2mu_blind_muz_emptybin": (_plain(blindflag="65", nzbin="14",
                                          zmax="1.4"), ["M0DIF"]),

    # logmass_min/logmass_max cut the data as well as sizing the biasCor logmass
    # grid (set_CUTMASK, :17987). No biasCor here, so nothing else can cut them.
    "SALT2mu_logmass_cut": (_sel("DATA_MASS.FITRES", logmass_min="9.0",
                                 logmass_max="11.0"), ["M0DIF"]),

    # cutmask_write=-9: the FITRES header and fit-results block with no SN rows,
    # and no M0DIF/CSV/COV at all (write_fitres_driver :23246, write_M0_fitres
    # :22701, write_M0_csv :22821, write_M0_cov :22882)
    "SALT2mu_cutmask_write_m9": ({**BASE, "cutmask_write": "-9"}, ["FITRES"]),

    # idsurvey_select excludes IDSURVEY=1 (SDSS) before the IDSAMPLE partition
    # (prepare_IDSAMPLE_biasCor :9075), so DES alone is IDSAMPLE 0
    "SALT2mu_idsurvey_select": ({**BASE, "datafile": "DATA_SDSS.FITRES",
                                 "simfile_biascor": "BIASCOR_SDSS.FITRES",
                                 "surveygroup_biascor": "",
                                 "idsurvey_select": "10"}, ["M0DIF", "FITRES"]),

    # the CC-prior reference distance comes from COSPAR_UNBLIND while the data's
    # comes from the blinded cosmology (malloc_PROB_CCprior :16965)
    "SALT2mu_blind_ccprior": ({**BEAMS, "blindflag": "2"}, ["M0DIF"]),

    # force_probcc0 stays on bbc_biascor_fit: forcing the contaminants to full
    # weight runs alpha0/beta0 into their bounds on this sample.
}


def run(name, keys, keep, workdir: Path):
    inp = workdir / f"{name}.inp"
    argv = keys.get(ARGV, [])
    lines = [f"prefix={name}"]
    for k, v in keys.items():
        if v == "" or k == ARGV:
            continue
        if k == "datafile":
            datafiles = []
            for filename in v.split(","):
                if filename.endswith((".FITRES", ".list", ".txt")):
                    path = workdir / filename
                    filename = str(path if path.is_file() else HERE / filename)
                datafiles.append(filename)
            lines.append(f"{k}={','.join(datafiles)}")
        elif v.endswith((".FITRES", ".list", ".txt")):
            path = workdir / v
            lines.append(f"{k}={path if path.is_file() else HERE / v}")
        else:
            lines.append(f"{k}={v}")
    inp.write_text("\n".join(lines) + "\n")
    log = subprocess.run(["SALT2mu.exe", inp.name] + list(argv), cwd=workdir,
                         capture_output=True, text=True)
    out = {}
    for ext in keep:
        path = workdir / f"{name}.{ext}"
        if not path.is_file():
            sys.exit(f"{name}: SALT2mu.exe produced no .{ext}\n"
                     + log.stdout[-3000:] + log.stderr[-2000:])
        # drop SALT2mu's wall-clock stamps so reruns are byte-stable
        out[f"{name}.{ext}"] = "".join(
            ln for ln in path.read_text().splitlines(keepends=True)
            if not ln.startswith(("# CPU:", "CPU_MINUTES:")))
    return out


APPEND_PATTERNS = ["PROB*", "SNRSUM", "SNRSUM,PROB*", "NOSUCHVAR"]


def append_varname(workdir: Path):
    """One run per append_varname_missing pattern; .VARNAMES records the column list
    each produced, .FITRES keeps the PROB* run for the value comparison."""
    split = "DATA_SPLIT_A.FITRES,DATA_SPLIT_B.FITRES"
    lines, fitres = [], None
    for i, pattern in enumerate(APPEND_PATTERNS):
        name = f"av{i}"
        out = run(name, _sel(split, append_varname_missing=pattern), ["FITRES"], workdir)
        text = out[f"{name}.FITRES"]
        cols = next(ln.split()[1:] for ln in text.splitlines()
                    if ln.startswith("VARNAMES:"))
        lines.append(" ".join([pattern] + cols[:cols.index("CUTMASK")]))
        if pattern == "PROB*":
            fitres = text
    return {"SALT2mu_append_varname.VARNAMES": "\n".join(lines) + "\n",
            "SALT2mu_append_varname.FITRES": fitres}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report what would change without writing")
    ap.add_argument("names", nargs="*", help="subset of CONFIGS to run")
    args = ap.parse_args()

    if not shutil.which("SALT2mu.exe"):
        sys.exit("SALT2mu.exe is not on $PATH")
    names = args.names or list(CONFIGS)
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        # SALT2mu's bias-correction header reader needs plain text.
        for compressed in HERE.glob("*.FITRES.gz"):
            with gzip.open(compressed, "rb") as source, (Path(tmp) / compressed.stem).open("wb") as target:
                shutil.copyfileobj(source, target)
        tables, _ = build()
        for name in BIASCOR_VARIANTS:
            (Path(tmp) / name).write_text(tables[name])
        outputs = {}
        if not args.names:
            outputs.update(append_varname(Path(tmp)))
        for name in names:
            keys, keep = CONFIGS[name]
            outputs.update(run(name, keys, keep, Path(tmp)))
        changed = False
        for fname, text in outputs.items():
            path = HERE / fname
            stored = path if path.exists() else path.with_name(path.name + ".gz")
            same = False
            if stored.exists():
                opener = gzip.open if stored.suffix == ".gz" else open
                with opener(stored, "rt") as stream:
                    same = stream.read() == text
            changed |= not same
            print(f"  {fname}: {'unchanged' if same else 'WOULD CHANGE'}")
            if not args.check:
                if len(text.encode()) >= 100_000:
                    path.with_name(path.name + ".gz").write_bytes(
                        gzip.compress(text.encode(), compresslevel=6, mtime=0))
                    path.unlink(missing_ok=True)
                else:
                    path.write_text(text)
                    path.with_name(path.name + ".gz").unlink(missing_ok=True)
    if args.check and changed:
        sys.exit(1)
    if not args.check:
        print("written")


if __name__ == "__main__":
    main()
