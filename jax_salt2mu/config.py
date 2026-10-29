"""Typed configuration for the BBC/SALT2mu fit.

One flat :class:`BBCConfig` whose fields are SALT2mu's own input keys, grouped by
comment. Model-specific cut fields, nuisance parameters and keys are deliberately
absent: cuts are resolved through the model's ``cut_aliases`` onto a generic
CUTWIN list, nuisance values onto its ``nuisance_spec``, and the model's own BBC
keys into ``model_inputs``, so adding a distance model needs no config change.

:func:`resolve` is the one place SALT2mu's setup-time rewrites are applied.
"""

import argparse
import copy
import logging
import re
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, get_type_hints

from jax_snlc_core import logging as log
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.models import model_input_keys, parse_model_input_value, standardised_model
from jax_snlc_core.models.registry import model_input_declarations
from jax_snlc_core.snana_files import load_survey_ids
from jax_snlc_core.utils import DEFAULT_DOUBLE_PRECISION, LIGHT_KM
from jax_snlc_core.input_settings import (add_settings, bootstrap_settings as parse_bootstrap,
                                          build_dataclass, configuration_settings, convert,
                                          dataclass_settings, derived_changes, merge_settings,
                                          parsed_settings, read_settings, refused)

from .blinding import BLINDMASK_FIXPAR, BLINDPAR_DEFAULT

logger = logging.getLogger(__name__)

_OPT_CCPRIOR_DEFAULT = 5      # GAUSS+UPDATE, SALT2mu.c:489

# slot -> name for the reference cosmology and sigint (FITPARNAMES_DEFAULT, SALT2mu.c:1446)
_SLOT_NAMES: Dict[int, str] = {
    9: "Omega_L", 10: "Omega_k", 11: "w0", 12: "wa", 14: "sigint",
}

# Live synonyms: both spellings appear in SNANA's ppar; accept either.
_ALIASES: Dict[str, str] = {
    "file": "datafile",
    "bins": "nzbin",
    "zVARNAME": "varname_z",
    "sigmb_biascor": "sigint_biascor",
    "fieldsplit_biascor": "fieldgroup_biascor",
    "fieldGroup_biascor": "fieldgroup_biascor",
    "fieldList_biascor": "fieldlist_biascor",
    "surveyGroup_biascor": "surveygroup_biascor",
    "simfile_bias": "simfile_biascor",
    # all set INPUTS.sigmB; SNANA's own sigint= is a dead key (SALT2mu.c:19972)
    "sig1": "sigint", "sigmb": "sigint", "sigmB": "sigint",
    "sig1fit": "fitflag_sigmb", "sig1tol": "redchi2_tol",
    "prescale_sim1a": "prescale_simIa",
    "catfile_out": "cat_file_out",
    "dumpflag_nobiascor": "ndump_nobiascor",
    "simdata_prescale": "prescale_simdata",
    "sntype": "sntype_select",
}

# the keys whose spelling differs from the BBCConfig field they set
_FIELD_KEYS: Dict[str, str] = {
    "varname_pia": "varname_pIa", "prescale_simia": "prescale_simIa",
    "select_trueia": "select_trueIa",
}

# BBCConfig fields set by their own code below, or by no key at all; every
# other field is its key's value read as the field's type
_BESPOKE = frozenset({
    "model_lcfit", "double_precision", "_u_flags", "_model_inputs", "_unblind",
    "cutwin", "cutwin_disable", "lcpar_range", "chi2max_fitwgt0",
    "mucovscale_bins", "ccprior_bins",
    "chi2max_surveys", "nuisance_val", "nuisance_float", "nuisance_onebin",
    "omega_l", "omega_k", "w0", "wa", "powzbin", "znhalf",
    "blindpar", "force_pia", "use_h11", "nmax", "prescale_biascor",
    "zwin_vpec_check", "surveylist_nobiascor", "type_list_probcc0",
    "idsurvey_list_probcc0", "cat_only", "cat_prescale",
})

_COSMO_ALIASES: Dict[str, str] = {
    "OL": "omega_l", "Ok": "omega_k", "w": "w0", "wa": "wa",
}

# p<N> for slots 9-12 sets the reference cosmology, not nuisance_val
_COSMO_SLOTS: Dict[str, str] = {
    "Omega_L": "omega_l", "Omega_k": "omega_k", "w0": "w0", "wa": "wa",
}

_P_SLOT = re.compile(r"^p(\d+)$")
_U_SLOT = re.compile(r"^u(\d+)$")
_S_SLOT = re.compile(r"^s(\d+)$")


# opt_biascor bits, mirroring SALT2mu.c:463-476.
MASK_BIASCOR_1DZAVG = 1       # interp MUBIAS vs z (1D), WGT=1
MASK_BIASCOR_1DZWGT = 2       # idem, WGT=1/muerr^2
MASK_BIASCOR_1D5DCUT = 8      # 1DZWGT but with the 5D grid cut applied
MASK_BIASCOR_CELLMAP = 16     # SNANA MASK_BIASCOR_5D; dimensionality is data-driven
MASK_BIASCOR_MUCOVSCALE = 32
MASK_BIASCOR_SAMPLE = 64
MASK_BIASCOR_MU = 128         # bias on MU instead of the LC parameters (bias_parameters)
MASK_BIASCOR_COVINT = 256     # empirical 3x3 COVINT, scale_covint replacing sigint
MASK_BIASCOR_SIGINT_SAMPLE = 512   # sigint(biasCor) per IDSAMPLE
MASK_BIASCOR_noCUT = 1024     # keep data events with no valid biasCor
MASK_BIASCOR_MAD = 2048       # median|pull| instead of RMS in COV scale
MASK_BIASCOR_MUCOVADD = 4096  # per-cell additive sigint (forces MAD)
MASK_BIASCOR_DEFAULT = (MASK_BIASCOR_CELLMAP | MASK_BIASCOR_MUCOVSCALE
                        | MASK_BIASCOR_SAMPLE)

# opt_biascor bits SNANA acts on that jax does not (print_biascor_options, SALT2mu.c:11642)
OPT_BIASCOR_REFUSED = {
    MASK_BIASCOR_1D5DCUT:
        "a 1D biasCor with the 5D grid cut applied, which SNANA itself labels "
        "a test option; use the 1D bits alone, or 16 for the 5D map",
    MASK_BIASCOR_COVINT:
        "the empirical 3x3 biasCor COVINT, which makes the iterated parameter "
        "scale_covint instead of sigint; jax keeps SNANA's diagonal COVINT "
        "and iterates sigint",
    MASK_BIASCOR_SIGINT_SAMPLE:
        "sigint(biasCor) per IDSAMPLE; jax computes one sigint for every "
        "sample",
    MASK_BIASCOR_noCUT:
        "keeping data events with no valid biasCor; jax always sets "
        "CUTBIT_BIASCOR for them",
}

# the CC-prior z binsize nzbin_ccprior=0 asks for (SALT2mu.c:17276)
CCPRIOR_ZBIN = 0.10

# M0 bound half-width about M0_DEFAULT (SALT2mu.c:2627-2634)
M0_HALFWIDTH = 5.0
M0_HALFWIDTH_ZPOLY = 10.0     # muref_zpoly stands in for the cosmology
M0_HALFWIDTH_FIXED = 0.001    # uM0=0

# uM0 options (SALT2mu.c:585-587)
M0FITFLAG_CONSTANT = 0
M0FITFLAG_ZBINS_FLAT = 1
M0FITFLAG_ZBINS_INTERP = 2


# name -> (start, lo, hi, slot): SNANA parval (:6701) and MINUIT limits (:22368-22378)
BBC_NUISANCE_SPEC = {
    "scalePCC": (0.0, -0.10, 15.0, 13),
    "H11mucc0": (0.0, -9.0, 9.0, 17),
    "H11mucc1": (0.1, -9.0, 9.0, 18),
    "H11mucc2": (0.1, -30.0, 30.0, 19),
    "H11sigcc0": (0.1, 0.01, 2.0, 20),
    "H11sigcc1": (0.0, -9.0, 9.0, 21),
    "H11sigcc2": (0.0, -9.0, 9.0, 22),
}

# the six Hlozek-2011 polynomial coefficients, which SNANA names by slot rather
# than by spelling (IPAR_H11, NPAR_H11_TOT, SALT2mu.c:6697)
IPAR_H11, NPAR_H11_TOT = 17, 6
H11_PARAMS = tuple(n for n, spec in BBC_NUISANCE_SPEC.items()
                   if IPAR_H11 <= spec[3] < IPAR_H11 + NPAR_H11_TOT)

# SNANA's INPUTS.parval[] starting values (SALT2mu.c:6701): 0 except these and BBC's own
SNANA_PARVAL_DEFAULTS = {"logmass_cen": 10.0, "sigint": 0.10,
                         **{n: spec[0] for n, spec in BBC_NUISANCE_SPEC.items()}}


def parval(config: "BBCConfig", name: str) -> float:
    """SNANA's INPUTS.parval for one parameter: the p<N> value, else its default."""
    return config.nuisance_val.get(name, SNANA_PARVAL_DEFAULTS.get(name, 0.0))


def use_gamma0(model, config) -> bool:
    """SNANA's USE_GAMMA0 flag (prep_input_gamma, SALT2mu.c:22820-22828).

    True when the mass-step coefficient is floated or its input value is
    non-zero. With it off, get_gammadm_host returns before computing either the
    step or its offset (:16776), so the whole term drops out.
    """
    name = model.gammadm_coeff
    if name is None:
        return False
    return bool(config.nuisance_float.get(name, False)
                or abs(parval(config, name)) > 1.0e-8)


@dataclass
class BBCConfig:
    """Everything the bias-correction fit reads from its SALT2mu input file.

    Field names are SALT2mu's own input keys, so a key in the input file and an
    attribute here are spelled the same. Keys which SALT2mu accepts but which
    have no effect there, and those not implemented here, are listed in
    ``_REJECTED_KEYS`` and are refused with a reason rather than
    accepted silently.

    Model-specific cut fields, nuisance parameters and keys are deliberately
    absent. Cuts resolve through the model's ``cut_aliases`` onto the generic
    ``cutwin`` list, nuisance values onto its ``nuisance_spec``, and the model's
    own keys into ``model_inputs``, so adding a distance model requires no
    change here.
    """

    # Input
    #: which distance model produced the light-curve fits.
    model_lcfit: str = "SALT2"
    datafile: List[str] = field(default_factory=list)  #: FITRES file or files of fitted light curves. Required.
    #: SNANA u<N>; whether numbered parameter N is fitted: ``u1=1`` fits it
    #: and ``u1=0`` holds it. The numbered parameters are tabulated at the
    #: foot of this page. u13, u15 and u16 have a further meaning, also
    #: described there.
    _u_flags: Dict[str, int] = field(default_factory=dict)
    _derived_settings: Dict[str, Any] = field(default_factory=dict)
    # Set by the keys the model declares
    #: the values of the keys the distance model adds to this fit, which are
    #: listed on the Model settings page.
    _model_inputs: Dict[str, Any] = field(default_factory=dict)
    # Precision
    #: Double or single precision. Omit to use the active precision in Python;
    #: command-line runs default to single. Not in SNANA.
    double_precision: bool | None = None

    # Redshift binning
    nzbin: int = 0  #: number of equal-width redshift bins.
    nlogzbin: int = 0  #: number of bins equally spaced in log z, used instead of ``nzbin``.
    powzbin: float = 0.0  #: bin widths proportional to (1+z) raised to this power, rather than uniform.
    #: SNANA powzbin; second value in ``powzbin=power,znhalf``, the redshift
    #: above which bins revert to uniform width.
    znhalf: float = 0.0
    zbinuser: List[float] = field(default_factory=list)  #: explicit ascending bin edges, which also set ``nzbin``, ``zmin`` and ``zmax``.
    #: Minimum fitted redshift. Required unless zbinuser specifies the bin
    #: edges.
    zmin: float = -0.01
    #: Maximum fitted redshift. Required unless zbinuser specifies the bin edges.
    zmax: float = 9.99
    min_per_zbin: int = 1  #: redshift bins holding fewer events than this are dropped from the fit.
    varname_z: str = "zHD"  #: column holding the redshift each event is fitted at.

    # Event selection
    #: column holding the host variable the mass step is measured against.
    #: A value below -990 is read as not measured.
    varname_gamma: str = "HOST_LOGMASS"
    #: SNANA CUTWIN; a selection window on one FITRES column, written as the
    #: form ``CUTWIN <column> <low> <high>``, one line per window. A second
    #: window on the same column replaces the first. The
    #: model's own shorthands, such as ``x1min``, add to the same list.
    cutwin: List[Tuple[str, float, float]] = field(default_factory=list)
    #: SNANA CUTWIN; ``CUTWIN NONE`` disables data parameter/host windows,
    #: chi-squared cuts and sparse-redshift-bin rejection. Redshift limits,
    #: CID lists, simulation prescales, covariance-quality checks and valid
    #: bias corrections remain required. BIASCOR and contamination simulations
    #: retain their cuts.
    cutwin_disable: bool = False
    #: SNANA <parameter>min/<parameter>max; the model's window shorthands:
    #: ``x1min``, ``x1max``, ``cmin`` and ``cmax`` for SALT, and
    #: ``thetamin``, ``thetamax``, ``avmin`` and ``avmax`` for BayeSN. Each
    #: moves one edge of the window on that light-curve parameter. The
    #: defaults come from the model. Unlike a generic ``CUTWIN``, these windows
    #: also bound any matching error-map and contaminant-prior coordinates.
    lcpar_range: Dict[str, Tuple[Optional[float], Optional[float]]] = \
        field(default_factory=dict)
    chi2max: float = 1.0e9  #: reject events whose chi-squared about the Hubble diagram exceeds this.
    #: SNANA chi2max(FITWGT0); deweight the events above the ceiling, e.g.
    #: ``chi2max(FITWGT0)=16``, instead of rejecting them.
    chi2max_fitwgt0: bool = False
    #: SNANA chi2max(<SURVEYLIST>); a ceiling for the surveys named in the
    #: brackets, e.g. ``chi2max(DES,LOWZ)=16``. The names are those in
    #: SURVEY.DEF.
    chi2max_surveys: Dict[str, float] = field(default_factory=dict)
    prescale_simdata: float = 1.0  #: keep one simulated data event in this many.
    prescale_simcc: float = 1.0  #: keep one simulated core-collapse event in this many.
    prescale_simia: float = 1.0  #: SNANA prescale_simIa; keep one simulated type Ia event in this many.
    uzsim: bool = False  #: fit against the true simulated redshift rather than the measured one. A diagnostic, not for analysis.
    iflag_duplicate: int = 1  #: treatment of repeated CIDs: 0 retains duplicate rows, 1 aborts, 2 merges them by inverse-covariance weighting.
    fieldlist: List[str] = field(default_factory=list)  #: keep only events whose FIELD contains one of these strings.
    cid_select_file: List[str] = field(default_factory=list)  #: file or files listing the only CIDs to keep.
    cid_reject_file: List[str] = field(default_factory=list)  #: file or files listing CIDs to discard.
    #: Two-column text file of redshift and retained fraction for simulated
    #: type Ia events. Fractions are linearly interpolated at SIM_ZCMB.
    #: Blank lines and comments beginning with # are allowed.
    zrate_scale_file: str = ""
    izbin_from_cid_file: bool = False  #: take each event's redshift bin from an IZBIN column in ``cid_select_file`` rather than from its redshift.
    force_realdata: bool = False  #: ignore the SIM columns, so that simulated input is treated as real data.
    #: SNANA select_trueIa; retain simulated type Ia events and disable contamination modelling.
    select_trueia: bool = False
    nmax: Optional[int] = None  #: cap on the total number of events fitted. SNANA's per-survey and per-field forms are not implemented.
    sntype_select: List[int] = field(default_factory=list)  #: keep only these SNTYPE values.
    idsurvey_select: List[int] = field(default_factory=list)  #: keep only these survey IDs.
    idsample_select: List[int] = field(default_factory=list)  #: keep only these sample indices.

    # Reference cosmology
    h0: float = 70.0  #: Hubble constant of the fixed reference cosmology, in km/s/Mpc.
    #: SNANA p9/OL; dark energy density of the reference cosmology, which is
    #: never fitted.
    omega_l: float = 0.7
    #: SNANA p10/Ok; curvature density of the reference cosmology.
    omega_k: float = 0.0
    #: SNANA p11/w; dark energy equation of state of the reference cosmology.
    w0: float = -1.0
    #: SNANA p12/wa; CPL coefficient in w(a) = w0 + wa(1 - a) for the
    #: reference cosmology.
    wa: float = 0.0
    #: Include radiation in reference distances; set to False for SNANA BBC parity.
    radiation: bool = True
    # Not set by an input key
    #: the reference cosmology before blinding offset it, kept so that blinding
    #: is applied once.
    _unblind: Dict[str, float] = field(default_factory=dict)
    # Fitted parameters
    fixpar_all: bool = False  #: freeze every nuisance parameter at its starting value and compute distances from those. M0 still floats.
    #: SNANA p<N>; the starting value of numbered parameter N, e.g.
    #: ``p1=0.14``. The model-specific and shared distance-fit
    #: parameters are tabulated in the parameter tables below. Numbers 9 to 12
    #: are the fixed reference cosmology, each with a row in the section above.
    nuisance_val: Dict[str, float] = field(default_factory=dict)
    #: SNANA u<N>; whether a coefficient is fitted. SALT standardisation
    #: coefficients held without a supplied value are zero, except
    #: logmass_cen=10. Fitted coefficients use the model's starting values.
    nuisance_float: Dict[str, bool] = field(default_factory=dict)
    #: SNANA u<N>; 0 or 3 gives the bias-correction map one bin in that
    #: parameter, and 3 also fits it.
    nuisance_onebin: Dict[str, bool] = field(default_factory=dict)
    uM0: int = 1  #: 0 holds M0 constant, 1 floats a flat M0 in each redshift bin, 2 interpolates between bin knots.
    #: ascending coefficients of a polynomial in redshift that replaces the
    #: reference distance modulus. For testing only.
    muref_zpoly: Tuple[float, ...] = ()

    # Bias corrections
    simfile_biascor: List[str] = field(default_factory=list)  #: simulation file or files, fitted through the same light-curve fitter, from which the bias is measured.
    #: sum of the bits selecting how the correction is built and applied,
    #: tabulated under :ref:`bbc-biascor-options`. With a ``simfile_biascor``,
    #: zero is read as 112, the cell map (16), error scaling (32) and a map
    #: per sample (64). Without one, zero means no correction.
    opt_biascor: int = 0
    #: Retain one of several disjoint BIASCOR subsets, as
    #: ``prescale_biascor=subset,prescale``. For example, ``0,10`` keeps
    #: every tenth row starting with the first; ``1,10`` starts with the
    #: second. Python and YAML use a pair of integers.
    prescale_biascor: Tuple[int, int] = (0, 1)
    #: intrinsic scatter assumed for the bias-correction simulation. Negative
    #: derives it from the simulation, which needs fifty events above
    #: ``snrmin_sigint_biascor`` for each combination of coefficients.
    sigint_biascor: float = -9.0
    snrmin_sigint_biascor: float = 60.0  #: minimum signal-to-noise for the high-quality subset used to derive that scatter.
    #: With additive cell scatter (opt_biascor bit 4096), 1 uses RMS pulls
    #: instead of 1.48 * median(abs(pull)) when solving for the additional
    #: intrinsic scatter.
    restore_sn_unite: int = 0
    min_per_cell_biascor: int = 3  #: cells holding fewer simulated events than this are treated as having no valid correction.
    #: width of the Gaussian weight given to each simulated event by its
    #: normalised distance from the cell centre. Above 10 the weight is flat.
    sigma_cell_biascor: float = 50.0
    check_duplicates_biascor: bool = False  #: report simulated events sharing redshift and bias-map coordinates. A diagnostic.
    ndump_nobiascor: int = 0  #: list up to this many data events which found no valid bias correction.
    #: SNANA mucovscale_bins.<parameter>; bin counts for the distance-error
    #: map, as ``mucovscale_bins.c=3``. Zero omits a coordinate.
    mucovscale_bins: Dict[str, int] = field(default_factory=dict)
    nbin_logmass: int = 1  #: number of host-mass bins in the BIASCOR map.
    interp_biascor_logmass: int = 1  #: interpolate the correction between host-mass bins rather than taking the containing bin.
    logmass_min: float = -40.0  #: lower edge of the host-mass range covered by the map.
    logmass_max: float = 40.0  #: upper edge of the host-mass range covered by the map.
    #: Survey groups for separate correction maps, with commas separating
    #: groups and + joining surveys, e.g. ``CFA3S+CFA3K,CSP,DES``.
    surveygroup_biascor: str = ""
    surveylist_nobiascor: List[str] = field(default_factory=list)  #: surveys fitted without requiring a bias correction at all.
    surveygroup_biascor_abortflag: bool = True  #: abort when a requested survey group matches no data, rather than continuing without it.
    frac_warn_nobiascor: float = 0.02  #: warn when more than this fraction of events is rejected because no valid bias correction is available.
    #: fields to combine into one sample, in the same form as
    #: ``surveygroup_biascor``. A name matches any FIELD containing it, while a
    #: name in vertical bars, e.g. ``|C3|``, matches that FIELD exactly.
    fieldgroup_biascor: str = ""
    #: exact FIELD names, one sample each. An overlap such as ``C3+X3`` is a
    #: field of its own. Cannot be combined with ``fieldgroup_biascor``.
    fieldlist_biascor: str = ""
    #: splits each sample into spectroscopic and photometric redshift subsets
    #: at this redshift error. It also sets whether the redshift prior
    #: columns are read.
    zspec_errmax_idsample: float = 0.0

    # Core-collapse prior
    #: simulation of core-collapse supernovae, from which their distance
    #: residuals are measured. Needs a ``simfile_biascor`` cell map.
    simfile_ccprior: List[str] = field(default_factory=list)
    #: SNANA simfile_ccprior; ``H11`` in place of a file selects the analytic
    #: Hlozek et al. (2012) Gaussian residual model, with quadratic
    #: redshift dependence in its mean and intrinsic width. Enable it with
    #: ``u13=1`` or a non-zero ``p13``; otherwise setup disables it. Once
    #: enabled, all six coefficients and ``scalePCC`` float, unless
    #: ``fixpar_all`` holds all parameters fixed. No CC simulation is needed.
    use_h11: bool = False
    #: redshift bins in the core-collapse residual map; 0 asks for bins 0.1 wide.
    nzbin_ccprior: int = 0
    #: SNANA ccprior_bins.<parameter>; bin counts for the simulated
    #: contaminant prior, as ``ccprior_bins.x1=2``. Zero omits a coordinate.
    ccprior_bins: Dict[str, int] = field(default_factory=dict)
    #: Distance-residual bin width in magnitudes for the simulated
    #: contaminant prior, over a range of -3 to +4 magnitudes.
    dmubin_ccprior: float = 0.5
    #: SNANA varname_pIa; column holding each event's probability of being a
    #: type Ia.
    varname_pia: str = ""
    #: SNANA force_pIa; give every event this probability instead of reading
    #: a column. Set -1 to use simulated types; in a SALT2mu input file,
    #: ``force_pIa=perfect`` is equivalent.
    force_pia: Optional[float] = None
    #: SNTYPE of a spectroscopically confirmed type Ia, given probability 1.
    #: Read from the ``TYPE`` column, which is -9 where absent, the same as
    #: this default, so a table without it has every event confirmed.
    typeIa_ccprior: int = -9
    type_list_probcc0: List[int] = field(default_factory=list)  #: additional SNTYPE values to treat as confirmed type Ia.
    #: Survey IDs whose events are treated as confirmed type Ia. SALT2mu
    #: inputs also accept survey names in place of IDs.
    idsurvey_list_probcc0: List[int] = field(default_factory=list)

    # Error model and intrinsic scatter
    #: 0 fixes the supplied intrinsic scatter; 1 iterates until reduced
    #: chi-squared reaches one; 2 also includes the changing variance
    #: normalisation in the likelihood. With BIASCOR, enabled iteration
    #: uses 2 for multidimensional corrections and 1 for redshift-only ones.
    #: sigint_mode=joint instead fits scatter directly.
    fitflag_sigmb: int = 0
    #: SNANA sig1/sigmb/sigmB; starting value of the intrinsic scatter, and
    #: its value when it is not fitted.
    sigint: float = 0.0
    #: one fixed intrinsic scatter per sample, in sample order, or a single
    #: value for every sample. It stops the scatter iteration after the first
    #: fit; per-sample applicability is described under :ref:`bbc-fixed-scatter`.
    sigint_fix: Tuple[float, ...] = ()
    redchi2_tol: float = 0.02  #: how close the reduced chi-squared must come to one for the iteration to stop.
    sigint_step1: float = 0.01  #: size of the first step taken by that iteration.
    dchi2red_dsigint: float = 0.0  #: fixed gradient of reduced chi-squared against intrinsic scatter, replacing the measured secant slope when non-zero.
    #: ``snana_secant`` reproduces SNANA's iteration. ``joint``, not in SNANA,
    #: floats the intrinsic scatter within the fit.
    sigint_mode: str = "snana_secant"
    sigint_floor: float = 1.0e-3  #: lower bound on the intrinsic scatter under ``sigint_mode=joint``.
    sigint_ceiling: float = 2.0  #: upper bound on the intrinsic scatter under ``sigint_mode=joint``.
    lensing_zpar: float = 0.0  #: weak-lensing distance scatter, sigma_lens = lensing_zpar * z, in magnitudes.
    zpecerr: float = 0.0  #: redshift uncertainty from peculiar velocities, given as a redshift.
    zwin_vpec_check: Tuple[float, float] = (0.01, 0.05)  #: redshift window over which the sign of the peculiar-velocity correction is checked. Set both to zero to disable the check.
    vpecerr: float = 0.0  #: peculiar-velocity uncertainty in km/s, as an alternative to ``zpecerr``.
    zphot_shift: float = 0.0  #: systematic redshift shift applied to photometric-redshift events after the model distance is computed.

    # Output
    #: Output filename prefix, e.g. ``<prefix>.M0DIF``. Required for
    #: command-line runs. Pipeline stages determine their own output filenames.
    prefix: str = "NONE"
    write_yaml: bool = False  #: write the summary YAML of counts, fitted parameters and their errors.
    write_biascor: bool = False  #: write the identifiers of bias-correction events passing selection.
    write_ccprior: bool = False  #: write the identifiers of contaminant-prior events passing selection.
    write_csv: bool = False  #: write the binned Hubble diagram as CSV for CosmoMC.
    write_chi2grid: int = 0  #: scan the chi-squared over this many points in each standardisation coefficient and write the grid.
    cutmask_write: int = 0  #: 0 writes accepted events to FITRES. With -9, FITRES contains only the header and fit summary, and M0DIF, COV and CSV are omitted. Only these two values are supported.
    append_varname_missing: str = "PROB*"  #: columns to keep when they are present in only some of the input files, as a comma-separated list of patterns.
    #: SNANA cat_only; written in place of the input file, e.g.
    #: ``salt2mu cat_only datafile=<files> cat_file_out=<file>``, it joins the
    #: input files and exits without cutting or fitting.
    cat_only: bool = False
    #: SNANA cat_only; written as ``cat_only/<N>``, it keeps every Nth row of
    #: the concatenation.
    cat_prescale: int = 1
    cat_file_out: str = ""  #: under ``cat_only``, the file to write.
    uave: bool = True  #: report distances against the fitted average M0, rather than against the nominal reference value.
    #: bitmask controlling blinding: 1 adds a cosine in redshift to each
    #: binned distance, 2 offsets the fixed reference cosmology, and 64 blinds
    #: simulations too.
    blindflag: int = BLINDMASK_FIXPAR
    #: SNANA blindpar9, blindpar11; amplitude and phase of the blinding
    #: offset, as ``blindpar9=amplitude,phase`` for dark energy density or
    #: ``blindpar11=amplitude,phase`` for w0. Python and YAML use
    #: ``blindpar={9: (amplitude, phase), 11: (amplitude, phase)}``.
    blindpar: Dict[int, Tuple[float, float]] = field(
        default_factory=lambda: dict(BLINDPAR_DEFAULT))


def lcpar_windows(model, config: BBCConfig) -> Dict[str, Tuple[float, float]]:
    """The model's per-parameter windows: its default cuts overridden by its
    cut shorthands. One value serves the selection cut (:17940) and the map
    axes (SALT2mu.c:6987, :17297), as SNANA's INPUTS do; a generic CUTWIN
    moves neither."""
    windows = dict(model.default_cuts)
    for var, (lo, hi) in config.lcpar_range.items():
        base = windows.get(var, (None, None))
        windows[var] = (base[0] if lo is None else lo,
                        base[1] if hi is None else hi)
    return windows


def map_bins(defaults, overrides):
    """Effective named axes, preserving model order and omitting zero overrides."""
    counts = {**defaults, **overrides}
    return {name: count for name, count in counts.items() if count != 0}


def validate_coordinates(model, config):
    """Validate declarations and direct Python overrides before building maps."""
    windows = lcpar_windows(model, config)

    def window(name, bounds):
        if (len(bounds) != 2 or any(v is None or not math.isfinite(v) for v in bounds)
                or bounds[0] >= bounds[1]):
            raise SNANAError(f"{model.__name__}: invalid range for {name}: {bounds}",
                             function_name="resolve")

    for name, bounds in windows.items():
        window(name, bounds)
    for name, (lo, hi, width) in model.biascor_binning.items():
        window(name, (lo, hi))
        if not math.isfinite(width):
            raise SNANAError(f"Invalid biascor bin width for {name}",
                             function_name="resolve")
    for field_name, defaults, overrides in (
            ("mucovscale_bins", model.mucovscale_bins, config.mucovscale_bins),
            ("ccprior_bins", model.ccprior_bins, config.ccprior_bins)):
        for mapping, minimum in ((defaults, 1), (overrides, 0)):
            for name, count in mapping.items():
                if isinstance(count, bool) or not isinstance(count, int) or count < minimum:
                    raise SNANAError(f"{model.__name__}: {field_name}.{name} needs an integer >= {minimum}",
                                     function_name="resolve")
        for name in map_bins(defaults, overrides):
            if name not in windows:
                raise SNANAError(f"{model.__name__}: {field_name}.{name} needs a parameter range",
                                 function_name="resolve")
    bits = model.bbc_cut_bits
    if (any(pair not in ((1, 14), (2, 15)) for pair in bits.values())
            or len(set(bits.values())) != len(bits)):
        raise SNANAError("BBC parameter cut bits must use distinct reserved pairs (1,14), (2,15)",
                         function_name="resolve")


def resolve_input_settings(cfg: BBCConfig) -> None:
    """Validate input combinations and apply linked settings before fitting."""
    _check_one_zbin_form(cfg)
    # keep_cutmask (SALT2mu.c:25659)
    if cfg.cutmask_write not in (0, -9):
        raise SNANAError(
            f"cutmask_write={cfg.cutmask_write} is not implemented; only 0 (fit "
            "sample) and -9 (no SN rows) are supported",
            function_name="parse_bbc_input")

    if cfg.select_trueia:
        cfg.prescale_simcc = 999999.0
        cfg.simfile_ccprior = []
        cfg.varname_pia = ""
        cfg.use_h11 = False
        cfg.nuisance_float.setdefault('scalePCC', False)

    if [s.upper() for s in cfg.simfile_ccprior] == ["H11"]:   # SALT2mu.c:20234
        cfg.use_h11, cfg.simfile_ccprior = True, []

    # vpecerr (km/s) is the same knob as zpecerr (dz), just unit-converted (:20027).
    if cfg.vpecerr > 0.0:
        cfg.zpecerr = cfg.vpecerr / LIGHT_KM

    if cfg.sigint_mode not in ("snana_secant", "joint"):
        raise SNANAError(f"sigint_mode={cfg.sigint_mode} is not recognised; use "
                         "'snana_secant' or 'joint'",
                         function_name="parse_bbc_input")

    if cfg.muref_zpoly:
        # prep_input (SALT2mu.c:22203): the polynomial replaces the cosmology, so M0
        # has to interpolate between bins and the chi2max cut is turned off.
        cfg.uM0 = M0FITFLAG_ZBINS_INTERP
        cfg.chi2max = 1.0e9
    if cfg.uM0 not in (M0FITFLAG_CONSTANT, M0FITFLAG_ZBINS_FLAT,
                       M0FITFLAG_ZBINS_INTERP):
        raise SNANAError(
            f"uM0={cfg.uM0} is not a valid option; SALT2mu takes "
            f"{M0FITFLAG_CONSTANT} (fixed), {M0FITFLAG_ZBINS_FLAT} (flat per "
            f"z-bin) or {M0FITFLAG_ZBINS_INTERP} (interpolated)",
            function_name="parse_bbc_input")
    if cfg.fixpar_all:
        cfg.uM0 = M0FITFLAG_CONSTANT      # :26363, with u1=u2=0


def _check_one_zbin_form(z) -> None:
    """Exactly one of nzbin / nlogzbin / zbinuser may be given (SALT2mu.c:21951).

    A question about which keys the user typed, so it belongs here and not in
    ``config.resolve``, which sets nzbin from the other two forms, and so would see
    two of them set on a second call and blame the user for its own rewrite.
    """
    given = [n for n, v in (("nzbin", z.nzbin), ("nlogzbin", z.nlogzbin),
                            ("zbinuser", z.zbinuser)) if v]
    if len(given) > 1:
        raise SNANAError(
            f"{len(given)} inputs specify the number of redshift bins ({given}); "
            "only one of nzbin, nlogzbin, zbinuser may be given",
            function_name="parse_bbc_input")


def resolve(config: BBCConfig, model, *, biascor=None, ccprior=None) -> BBCConfig:
    """Return the config with SALT2mu's setup-time rewrites applied, validated.

    SALT2mu overwrites INPUTS during setup and everything downstream reads the
    overwritten values (prep_input :22074, prepare_biasCor :11445,
    setup_BININFO_redshift :2896). This is the single place jax does that, called once
    during cell preparation; nothing after it may re-derive an effective value.

    Supplied correction tables participate in setup without changing file paths.
    Pure, so resolving an already-resolved config with the same inputs is a no-op.
    """
    has_biascor = bool(config.simfile_biascor) or biascor is not None
    has_ccprior = bool(config.simfile_ccprior) or ccprior is not None
    if config.use_h11 and has_ccprior:
        raise SNANAError(
            "H11 and a simulated CC-prior input were both supplied; choose one prior",
            function_name="resolve")
    if config.fixpar_all:
        # "compute MU from initial values": nothing floats, one z bin, no sigint
        # iteration (:21939). The nuisance freeze is in _floated_and_defaults.
        config = replace(config, nzbin=1, nlogzbin=0, zbinuser=[], fitflag_sigmb=0)
    config = replace(config, opt_biascor=_resolve_opt_biascor(config, has_biascor))
    config = replace(config, fitflag_sigmb=_resolve_fitflag_sigmb(config, has_biascor))
    config = _resolve_zbins(config)
    if config.nzbin_ccprior <= 0:
        # nzbin_ccprior=0 means "bins 0.1 wide over the fit range" (:6628), which
        # needs the range _resolve_zbins settles
        nz = int(math.ceil((config.zmax - config.zmin) / CCPRIOR_ZBIN - 1e-12))
        config = replace(config, nzbin_ccprior=nz)
    if (has_ccprior or config.use_h11) and not ccprior_enabled(config):
        log.warn(logger, "BBC", "scalePCC(u13,p13)=0,0 --> turn off CC prior")
        config = replace(config, simfile_ccprior=[], use_h11=False)
        has_ccprior = False
    declared = {spec.attribute: getattr(model, spec.attribute)
                for spec in model_input_keys(model, consumer='bbc').values()}
    config = replace(config, _model_inputs={**declared, **config._model_inputs})
    validate_coordinates(model, config)
    _validate(config, model, has_biascor, has_ccprior)
    return config


def ccprior_enabled(config: BBCConfig) -> bool:
    """A fixed scalePCC of zero turns the CC prior off (prep_input,
    SALT2mu.c:22117).

    That block clears INFO_CCPRIOR.USE, which is what fcn gates the whole BEAMS
    term on (:5209), and it runs before the H11 block re-floats scalePCC
    (:22143), so it silences an H11 prior as well as a sim-file one:
    simfile_ccprior=H11 sets INFO_CCPRIOR.USE itself (:20233).
    """
    floated = config.nuisance_float.get("scalePCC", False)
    return floated or abs(parval(config, "scalePCC")) >= 1.0e-6


def _validate_opt_biascor(opt: int) -> None:
    """Refuse an opt_biascor bit jax does not act on, and the combinations
    prepare_biasCor rejects (SALT2mu.c:11403-11422)."""
    for bit, reason in OPT_BIASCOR_REFUSED.items():
        if opt & bit:
            raise SNANAError(
                f"opt_biascor bit {bit} is not implemented: {reason}",
                function_name="resolve")
    one_d = opt & (MASK_BIASCOR_1DZAVG | MASK_BIASCOR_1DZWGT)
    if one_d and (opt & MASK_BIASCOR_CELLMAP):
        raise SNANAError(f"cannot do 1D and 5D biasCor (opt_biascor={opt})",
                         function_name="resolve")
    if not one_d and not (opt & MASK_BIASCOR_CELLMAP):
        raise SNANAError(f"opt_biascor={opt} asks for neither a 1D nor a 5D "
                         "biasCor", function_name="resolve")


def _resolve_opt_biascor(config: BBCConfig, has_biascor: bool) -> int:
    """Default opt_biascor, force SAMPLE under survey/field grouping (:21194), and
    force MAD under muCOVADD (:10519).

    These rewrites require a bias-correction input (ISFILE_BIASCOR, :21193).
    """
    if not has_biascor:
        return config.opt_biascor
    opt = config.opt_biascor or MASK_BIASCOR_DEFAULT
    if (config.surveygroup_biascor.strip() or config.fieldgroup_biascor.strip()
            or config.fieldlist_biascor.strip()):
        opt |= MASK_BIASCOR_SAMPLE
    if opt & MASK_BIASCOR_MUCOVADD:
        opt |= MASK_BIASCOR_MAD
    return opt


def _resolve_fitflag_sigmb(config: BBCConfig, has_biascor: bool) -> int:
    """prepare_biasCor rewrites a requested fitflag_sigmb to 2 for a 5D cell map and
    to 1 otherwise, so an explicit 2 does not survive a 1D biasCor (:10634)."""
    if config.fitflag_sigmb > 0 and has_biascor:
        return 2 if config.opt_biascor & MASK_BIASCOR_CELLMAP else 1
    return config.fitflag_sigmb


def _resolve_zbins(config: BBCConfig) -> BBCConfig:
    """Leave nzbin as the bin count and zmin/zmax as the range actually used, whichever
    key supplied them (setup_BININFO_userz :2724, conflict_check :21085). The biasCor
    z binsize (:9245) reads these, not the key the user gave."""
    if config.nlogzbin > 0:
        return replace(config, nzbin=config.nlogzbin)
    if config.zbinuser:
        edges = sorted(config.zbinuser)
        return replace(config, nzbin=len(edges) - 1, zmin=edges[0], zmax=edges[-1])
    return config


LOGMASS_TAU_MIN = 0.001    # prep_input aborts below this with the gate on (:21303)


def _validate(config: BBCConfig, model, has_biascor: bool, has_ccprior: bool) -> None:
    """Everything SALT2mu refuses at setup, in one place."""
    if use_gamma0(model, config):
        # a mass step needs a transition width; SNANA's parval leaves logmass_tau
        # at 0, which would divide by zero in the Fermi function (:21300-21307)
        tau = parval(config, "logmass_tau")
        if tau < LOGMASS_TAU_MIN:
            raise SNANAError(
                f"logmass_tau(p8)={tau:.4f} is too small to fit a mass step; "
                f"SALT2mu wants at least {LOGMASS_TAU_MIN}", function_name="resolve")
    if config.sigint_mode == "joint" and config.chi2max <= 1.0e8:
        log.warn(logger, "BBC",
                 "sigint_mode=joint with a chi2max cut: the cut removes the "
                 "residual tail the joint likelihood assumes is present, so the "
                 "fitted sigint is biased low. Prefer sigint_mode=snana_secant "
                 "when cutting on chi2.")
    if not (config.nzbin or config.nlogzbin or config.zbinuser):
        raise SNANAError("no redshift binning: set one of nzbin, nlogzbin, zbinuser",
                         function_name="resolve")
    if config.zbinuser and len(config.zbinuser) < 2:
        raise SNANAError(
            f"zbinuser needs at least two bin edges, got {len(config.zbinuser)}",
            function_name="resolve")
    # zmin/zmax keep SNANA's out-of-range defaults until the user sets them (:22316)
    if not config.zbinuser and not (config.zmin > 0.0 and config.zmax < 9.0):
        raise SNANAError("set zmin and zmax, or give the bin edges with zbinuser",
                         function_name="resolve")

    opt = config.opt_biascor
    if has_biascor:
        _validate_opt_biascor(opt)
    if config.uM0 == M0FITFLAG_ZBINS_INTERP and not has_biascor:
        # the knots are the biasCor's weighted mean z per bin (fcn_M0, :4985)
        raise SNANAError(f"uM0={M0FITFLAG_ZBINS_INTERP} needs simfile_biascor",
                         function_name="resolve")
    if has_biascor and (opt & MASK_BIASCOR_MUCOVADD) \
            and not model.biascor_coeffs:
        # muCOVADD tunes the error per alpha/beta node, which a model with no
        # standardisation coefficients has not got (:10521)
        raise SNANAError("muCOVADD needs a model with standardisation coefficients; "
                         f"{config.model_lcfit} has none", function_name="resolve")
    if has_ccprior and not (has_biascor
                                       and opt & MASK_BIASCOR_CELLMAP):
        # the CC prior needs the 5D cell map (prepare_CCprior)
        if has_biascor:
            raise SNANAError("cannot use a 1D biasCor with the CC likelihood; drop "
                             "the CC prior or use a 5D biasCor", function_name="resolve")
        raise SNANAError(
            "all CCPRIOR events fail cuts: the CC prior reads its bias from the 5D "
            "biasCor cell map, so every event fails the valid-biasCor cut without one",
            function_name="resolve")


def declarations(model_name):
    types, owners = dataclass_settings(BBCConfig)
    model = standardised_model(model_name)
    for spec in model_input_keys(model, 'bbc').values():
        name = spec.attribute
        if name not in types:
            types[name] = get_type_hints(model, include_extras=True)[name]
            owners[name] = model
    return types, owners


def merge_named_settings(values, settings):
    """Merge supplied fields, replacing SNANA modes only when explicitly changed."""
    if any(name.startswith('_') for name in settings):
        raise SNANAError('Private BBC fields are not input settings')
    result = merge_settings(values, settings)
    model = standardised_model(result.get('model_lcfit', BBCConfig.model_lcfit))
    specs = {**BBC_NUISANCE_SPEC, **model.nuisance_spec}
    flags = result.get('_u_flags', {})
    for field in ('nuisance_float', 'nuisance_onebin'):
        if field not in settings:
            continue
        names = settings[field] or values.get(field, {})
        for name in names:
            if name in specs:
                flags.pop(f'u{specs[name][3]}', None)
    return result


def build_config(settings, config=None, *, _snana_flags=None):
    """Build named BBC settings without loading FITRES or bias-correction data."""
    base = configuration_settings(BBCConfig() if config is None else config)
    values = merge_settings(base, settings)
    model = standardised_model(values.get('model_lcfit', BBCConfig.model_lcfit))
    types, owners = declarations(values.get('model_lcfit', BBCConfig.model_lcfit))
    unknown = set(values) - set(types)
    if unknown:
        raise SNANAError(f'Unknown BBC setting(s): {sorted(unknown)}')
    core, model_values = {}, {}
    for name, value in values.items():
        converted = convert(value, types[name], name)
        (core if owners[name] is BBCConfig else model_values)[name] = converted
    cfg = build_dataclass(BBCConfig, core)
    if config is not None:
        cfg._model_inputs = copy.deepcopy(config._model_inputs)
        cfg._u_flags = copy.deepcopy(config._u_flags)
        cfg._unblind = copy.deepcopy(config._unblind)
    cfg._model_inputs.update(model_values)
    original = {k: v for k, v in asdict(cfg).items() if not k.startswith('_')}
    allowed = set(model.nuisance_spec) | set(BBC_NUISANCE_SPEC)
    for mapping in (cfg.nuisance_val, cfg.nuisance_float, cfg.nuisance_onebin):
        unknown = set(mapping) - allowed
        if unknown:
            raise SNANAError(f'{cfg.model_lcfit} has no nuisance parameters {sorted(unknown)}')
    specs = {**BBC_NUISANCE_SPEC, **model.nuisance_spec}
    # Unrelated overrides must preserve SNANA modes such as u13=2.
    changed = set(settings.get('nuisance_float', {})) | set(settings.get('nuisance_onebin', {}))
    if config is not None:
        changed.update(name for name, value in config.nuisance_onebin.items()
                       if base['nuisance_onebin'].get(name) != value)
    for field in ('nuisance_float', 'nuisance_onebin'):
        if field in settings and not settings[field] and config is not None:
            changed.update(getattr(config, field))
    for name in changed:
        floated = cfg.nuisance_float.get(name, False)
        onebin = cfg.nuisance_onebin.get(name, not floated)
        onebin = onebin if floated else True
        cfg.nuisance_onebin[name] = onebin
        slot = specs[name][3]
        cfg._u_flags[f'u{slot}'] = (3 if onebin else 1) if floated else 0
    if _snana_flags is not None:
        cfg._u_flags.update(_snana_flags)
    declared = {s.attribute for s in model_input_keys(model, 'bbc').values()}
    if set(cfg._model_inputs) - declared:
        raise SNANAError(f'Unknown {cfg.model_lcfit} model settings: {sorted(set(cfg._model_inputs) - declared)}')
    resolve_input_settings(cfg)
    validate_coordinates(model, cfg)
    resolved = {k: v for k, v in asdict(cfg).items() if not k.startswith("_")}
    cfg._derived_settings = derived_changes(original, resolved)
    return cfg


def command_settings(tokens, settings):
    parser = argparse.ArgumentParser(allow_abbrev=False, add_help=False)
    types, owners = dataclass_settings(BBCConfig)
    add_settings(parser, types, owners)
    selectors, _ = parser.parse_known_args(tokens)
    model_name = vars(selectors).get('model_lcfit', settings.get('model_lcfit', BBCConfig.model_lcfit))
    types, owners = declarations(model_name)
    parser = argparse.ArgumentParser(description='Fit bias-corrected supernova distances.', allow_abbrev=False)
    add_settings(parser, types, owners)
    supplied, snana = parser.parse_known_args(tokens)
    return parsed_settings(supplied, types, readers={}), snana


def bootstrap_settings(tokens):
    """Read shared and model options before locating the input file."""
    types, owners = dataclass_settings(BBCConfig)
    return parse_bootstrap(tokens, types, owners, *model_input_declarations('bbc'), readers={})


def parse_bbc_input(path: Optional[str], overrides=(), *, settings=None) -> BBCConfig:
    """Read a file, SNANA overrides and typed settings, in that order.

    ``overrides`` uses SALT2mu keys and values, either as command-line tokens
    or a mapping. ``settings`` uses the typed Python/YAML field names.
    """
    return build_bbc_config(path, [(overrides, settings or {})])


def _override_tokens(overrides):
    """Encode a SNANA mapping using the same assignment syntax as its CLI."""
    if not isinstance(overrides, Mapping):
        return list(overrides)

    def scalar(value):
        return str(int(value)) if isinstance(value, bool) else str(value)

    tokens = []
    for key, value in overrides.items():
        cut = key.split()[0] == 'CUTWIN'
        if isinstance(value, Mapping):
            raise SNANAError(f'{key}: SNANA overrides do not take mappings; use settings')
        if isinstance(value, (list, tuple)):
            value = (' ' if cut else ',').join(scalar(item) for item in value)
        else:
            value = scalar(value)
        tokens.append(f'{key} {value}' if cut else f'{key}={value}')
    return tokens


def build_bbc_config(path, layers):
    """Merge ordered SNANA/typed input layers, then prepare one configuration.

    A pipeline supplies a nominal layer followed by a MUOPT layer. Each
    layer retains its input syntax; values never determine which reader runs.
    """
    text = '' if path is None else Path(path).read_text()
    text = re.split(r'(?m)^\s*#END_YAML\s*$', text, maxsplit=1)[-1]
    is_yaml = any(re.match(r'^[a-zA-Z_][\w]*\s*:', line) for line in text.splitlines())
    values = read_settings(path, text=text) if is_yaml else {}
    file_raw, file_cuts, file_off, file_repeated = _read_words(
        [] if is_yaml else _input_words(text))
    model_name = values.get('model_lcfit', file_raw.get('model_lcfit', BBCConfig.model_lcfit))
    parsed_layers = []
    for overrides, settings in layers:
        tokens = _override_tokens(overrides)
        selectors = {'model_lcfit': model_name}
        selectors.update(word.split('=', 1) for word in tokens if word.startswith('model_lcfit='))
        named, tokens = command_settings(tokens, merge_settings(selectors, settings))
        named = merge_settings(named, settings)
        raw, cuts, off, repeated = _read_words([word for token in tokens for word in token.split()])
        model_name = named.get('model_lcfit', raw.get('model_lcfit', model_name))
        parsed_layers.append((raw, cuts, off, repeated, named))
    model = standardised_model(model_name)
    values['model_lcfit'] = model_name
    if is_yaml:
        values = merge_named_settings({}, values)
    else:
        values = _read_snana_settings(list(file_raw.items()), file_cuts, file_off,
                                      file_repeated, set(), model, values)
    for raw, cuts, off, repeated, named in parsed_layers:
        values = _read_snana_settings(list(raw.items()), cuts, off, repeated,
                                      set(raw), model, values)
        values = merge_named_settings(values, named)
    flags = values.pop('_u_flags', {})
    return build_config(values, _snana_flags=flags)


def _double_precision(raw: Dict[str, str]) -> bool:
    """The floating-point frame the input asks for. A jax-only key."""
    val = raw.get("double_precision", "")
    if not val:
        return DEFAULT_DOUBLE_PRECISION
    try:
        return bool(int(val))
    except ValueError:
        raise SNANAError(f"double_precision={val} is not a valid int",
                         function_name="parse_bbc_input") from None


def _input_words(text: str) -> List[str]:
    """The input file's whitespace-separated words, comments stripped."""
    return [w for line in text.splitlines() for w in line.split("#", 1)[0].split()]


def _read_words(words: List[str]):
    """Stage 1: words -> (key=value dict, CUTWIN list, CUTWIN NONE flag, keys
    given more than once). No validation. ``CUTWIN var lo hi`` takes four words
    and a later CUTWIN on the same variable replaces the earlier
    (parseLine_SELECT_VAR, SALT2mu.c:20983, :21128); ``CUTWIN NONE`` disables
    the cuts (:21012). The qualified forms ``CUTWIN(...)``,
    ``CUTWIN_IDSAMPLE(...)`` and ``CUTWIN_SURVEY(...)`` are not implemented."""
    raw: Dict[str, str] = {}
    cutwins: List[Tuple[str, float, float]] = []
    disable = False
    assigned = set()
    repeated = set()
    i = 0
    while i < len(words):
        word = words[i]
        if word == "CUTWIN":
            if words[i + 1:i + 2] == ["NONE"]:
                disable = True
                i += 2
                continue
            if len(words) < i + 4:
                raise SNANAError("CUTWIN needs a variable and two limits: "
                                 f"'{' '.join(words[i:])}'",
                                 function_name="parse_bbc_input")
            var, lo, hi = words[i + 1:i + 4]
            cutwins = [c for c in cutwins if c[0] != var]
            cutwins.append((var, float(lo), float(hi)))
            i += 4
            continue
        if word.startswith("CUTWIN"):
            raise SNANAError(f"{word} is not implemented; only 'CUTWIN var lo hi' "
                             "and 'CUTWIN NONE' are",
                             function_name="parse_bbc_input")
        # a word with no '=' records an empty value, which take() treats as unset
        key, eq, val = word.partition("=")
        if eq:
            if key in assigned:
                repeated.add(key)
            assigned.add(key)
        raw[key] = val
        i += 1
    return raw, cutwins, disable, repeated


def _read_snana_settings(items, cutwins, cutwin_disable, repeated,
                         cli_keys, model, values):
    """Apply supplied SNANA values without adding defaults or resolving rules."""
    cfg = merge_settings(values)
    if cutwins:
        replaced = {cut[0] for cut in cutwins}
        cfg['cutwin'] = [cut for cut in cfg.get('cutwin', []) if cut[0] not in replaced] + list(cutwins)
    if cutwin_disable:
        cfg['cutwin_disable'] = True

    nuis_names = set(model.nuisance_spec)
    slot_names = {**_SLOT_NAMES,
                  **{slot: name for name, (_, _, _, slot)
                     in BBC_NUISANCE_SPEC.items()},
                  **{slot: name for name, (_, _, _, slot)
                     in model.nuisance_spec.items()}}

    # refuse before aliasing, so the refusal names the key as typed
    keyvals: Dict[str, str] = {}
    for key, val in items:
        if key in _REJECTED_KEYS:
            raise SNANAError(f"{key}: {_REJECTED_KEYS[key]}",
                             function_name="parse_bbc_input")
        keyvals[_ALIASES.get(key, key)] = val

    if 'chi2max' in keyvals and cfg.get('chi2max_surveys'):
        raise SNANAError('Cannot define chi2max after chi2max(SURVEYLIST)',
                         function_name='parse_bbc_input')

    used_names = set()
    valueless = set()

    def take(name, cast=str, default=None):
        val = keyvals.get(name, "")
        if not val:
            if name in keyvals:
                used_names.add(name)
                valueless.add(name)
            return default
        used_names.add(name)
        if cast is str:
            # remove_quote (sntools.c:6590) drops every quote, not just a surrounding pair.
            val = val.replace("'", "").replace('"', "")
        try:
            return cast(val)
        except ValueError:
            raise SNANAError(f"{name}={val} is not a valid {cast.__name__}",
                             function_name="parse_bbc_input") from None

    used_names.add("model_lcfit")
    used_names.add("double_precision")
    if 'double_precision' in keyvals:
        cfg['double_precision'] = _double_precision(keyvals)
    if _split_list(take("datafile_override", default="")):
        raise SNANAError("datafile_override is not implemented",
                         function_name="parse_bbc_input")

    for f in fields(BBCConfig):
        if f.name.startswith("_") or f.name in _BESPOKE:
            continue
        cast, convert = _READ_AS[f.type]
        value = take(_FIELD_KEYS.get(f.name, f.name), cast)
        if value is not None:
            cfg[f.name] = convert(value) if convert else value

    pz = _split_list(take("powzbin", str, ""))
    if pz:
        cfg['powzbin'] = float(pz[0])
        # Each SNANA powzbin assignment replaces both components.
        cfg['znhalf'] = float(pz[1]) if len(pz) > 1 else BBCConfig.znhalf
    # blindpar is indexed per fit-parameter slot; SNANA accepts only 9 (OL) and 11
    # (w0), and aborts on a bare blindpar= (parse_blindpar, SALT2mu.c:20556).
    for slot in (9, 11):
        amp_phase = _split_list(take(f"blindpar{slot}", str, ""))
        if amp_phase:
            cfg.setdefault('blindpar', {})[slot] = (float(amp_phase[0]), float(amp_phase[1]))

    if 'surveylist_nobiascor' in keyvals:
        cfg['surveylist_nobiascor'] = [v for v in _split_list(
            take("surveylist_nobiascor", str, "")) if v.upper() != "NONE"]

    # only GAUSS+UPDATE (SALT2mu.c:484-489)
    opt_ccprior = take("opt_ccprior", int, 0)
    if opt_ccprior not in (0, _OPT_CCPRIOR_DEFAULT):
        raise SNANAError(
            f"opt_ccprior={opt_ccprior} is not implemented; only "
            f"{_OPT_CCPRIOR_DEFAULT} (GAUSS+UPDATE) is supported",
            function_name="parse_bbc_input")
    zwin = _floats(take("zwin_vpec_check", str, ""))
    if zwin:
        if len(zwin) != 2:
            raise SNANAError(
                f"zwin_vpec_check takes zmin,zmax; got {len(zwin)} value(s)",
                function_name="parse_bbc_input")
        cfg['zwin_vpec_check'] = (zwin[0], zwin[1])
    if "force_pIa" in keyvals:                      # 'perfect' -> -1 (SALT2mu.c:19722)
        used_names.add("force_pIa")
        fp = keyvals["force_pIa"].strip()
        cfg['force_pia'] = -1.0 if fp.lower() == "perfect" else float(fp)
    # force_probcc0: spec-confirmed SNe -> pIa=1 (SALT2mu.c:19731/19734/19736)
    if 'type_list_probcc0' in keyvals:
        cfg['type_list_probcc0'] = [int(t) for t in _split_list(take("type_list_probcc0", default=""))]
    ids = _split_list(take("idsurvey_list_probcc0", default=""))
    if ids:
        name_to_id, _ = load_survey_ids()
        cfg['idsurvey_list_probcc0'] = [
            int(s) if s.lstrip("-").isdigit() else name_to_id[s] for s in ids]

    _resolve_chi2max_options(keyvals, cfg, used_names)
    nmax = take("nmax", str, "")
    if nmax:
        if "(" in nmax:
            raise SNANAError(
                "per-survey/field nmax is not implemented; only a bare total is",
                function_name="parse_bbc_input")
        parts = _split_list(nmax)
        if len(parts) != 1:
            raise SNANAError(f"cannot parse nmax={nmax}",
                             function_name="parse_bbc_input")
        cfg['nmax'] = int(parts[0])
    ps = take("prescale_biascor", str, "")
    if ps:
        parts = _split_list(ps)
        if len(parts) != 2:
            raise SNANAError("prescale_biascor needs <subset>,<prescale>",
                             function_name="parse_bbc_input")
        subset, prescale = int(parts[0]), int(parts[1])
        if subset >= prescale:                       # SALT2mu.c:20616
            raise SNANAError(
                f"prescale_biascor subset={subset} must be < prescale={prescale}",
                function_name="parse_bbc_input")
        cfg['prescale_biascor'] = (subset, prescale)

    for alias, attr in _COSMO_ALIASES.items():
        if alias in keyvals:
            used_names.add(alias)
            cfg[attr] = float(keyvals[alias])
    _resolve_nuisance(keyvals, cfg, nuis_names, slot_names, used_names)
    _resolve_named_coordinates(items, cfg, model, used_names)
    cfg.update(_model_key_values(keyvals, model, used_names))

    cfg.setdefault('_u_flags', {}).update({k: int(v) for k, v in keyvals.items() if _U_SLOT.match(k)})

    _check_repeated_and_cli_keys(repeated, cli_keys, used_names, valueless)
    _warn_unrecognized(keyvals, used_names)
    if valueless:
        log.warn(logger, "BBC", "SALT2mu key(s) given with no value, ignored: "
                 + ", ".join(sorted(valueless)))
    return cfg


def _resolve_chi2max_options(keyvals, cfg, used_names) -> None:
    """chi2max(FITWGT0)= deweights instead of rejecting; chi2max(SURVEYLIST)=
    sets per-survey thresholds, with unlisted surveys keeping the global value
    (parse_chi2max, SALT2mu.c:20428-20536)."""
    keys = list(keyvals)
    for key in keys:
        if not (key.startswith("chi2max(") and key.endswith(")")):
            continue
        used_names.add(key)
        opt = key[len("chi2max("):-1]
        val = float(keyvals[key])
        if opt == "FITWGT0":
            cfg['chi2max'] = val
            cfg['chi2max_fitwgt0'] = True
            continue
        surveys = [s.strip() for s in opt.split(",")]
        if "FITWGT0" in surveys:
            raise SNANAError(
                "Cannot mix FITWGT0 option with SURVEY",
                function_name="parse_bbc_input")
        if "chi2max" in keys and keys.index("chi2max") > keys.index(key):
            raise SNANAError(
                "Cannot define chi2max after chi2max(SURVEYLIST)",
                function_name="parse_bbc_input")
        for survey in surveys:
            cfg.setdefault('chi2max_surveys', {})[survey] = val


def _model_key_values(keyvals, model, used_names) -> Dict[str, object]:
    """The supplied values of the model's BBC keys, by attribute name, typed
    as the model declares them. SALT2mu keys are lowercase, so the match is
    case-insensitive."""
    given = {key.lower(): key for key in keyvals}
    values = {}
    for key, spec in model_input_keys(model, consumer='bbc').items():
        if key.lower() in given:
            used_names.add(given[key.lower()])
            values[spec.attribute] = parse_model_input_value(
                key, spec, keyvals[given[key.lower()]])
    return values


def _check_slot(key: str, name, settable) -> None:
    """A slot key must name a parameter this fit has: SNANA reads any p<N>
    into parval[N], where an unused slot does nothing, silently."""
    if name is None:
        raise SNANAError(f"{key}: no fit parameter has number {key[1:]} for "
                         f"this model", function_name="parse_bbc_input")
    if name not in settable and name not in _COSMO_SLOTS:
        raise SNANAError(f"{key}: {name} is not a parameter this fit sets",
                         function_name="parse_bbc_input")


# the spellings SALT2mu still reads for the renamed slopes (SALT2mu.c:19960-19977)
_NUISANCE_ALIASES = {"beta1": "dbeta_dz", "alphaHost": "dalpha_dlogm",
                     "betaHost": "dbeta_dlogm"}


def _resolve_nuisance(keyvals, cfg, nuis_names, slot_names, used_names) -> None:
    settable = nuis_names | set(BBC_NUISANCE_SPEC)
    for key, val in keyvals.items():
        name = None
        m = _P_SLOT.match(key)
        if m:
            name = slot_names.get(int(m.group(1)))
            used_names.add(key)
            _check_slot(key, name, settable)
        elif key in settable or _NUISANCE_ALIASES.get(key) in settable:
            # SNANA also accepts the core params by name (scalePCC -> parval[13],
            # SALT2mu.c:19970), not only through the p<N> slots.
            name = _NUISANCE_ALIASES.get(key, key)
            used_names.add(key)
        if name is None:
            continue
        if name in _COSMO_SLOTS:
            cfg[_COSMO_SLOTS[name]] = float(val)
        elif name in settable:
            cfg.setdefault('nuisance_val', {})[name] = float(val)
        # names not owned by this model (e.g. gamma0 for BayeSN) fall through to
        # the unrecognized-key check, which names the model.

    for key, val in keyvals.items():
        m = _U_SLOT.match(key)
        if not m:
            continue
        name = slot_names.get(int(m.group(1)))
        used_names.add(key)
        _check_slot(key, name, settable)
        try:
            flag = int(val)
        except ValueError:
            raise SNANAError(f"{key}={val} is not a valid int",
                             function_name="parse_bbc_input") from None
        if name in _COSMO_SLOTS and flag != 0:
            raise SNANAError(
                f"{key}=1 floats {name}, but jax holds the reference cosmology fixed",
                function_name="parse_bbc_input")
        if name in settable:
            cfg.setdefault('nuisance_float', {})[name] = flag != 0
            if flag == 0:
                cfg.setdefault('nuisance_onebin', {}).pop(name, None)
            else:
                cfg.setdefault('nuisance_onebin', {})[name] = flag == 3
    # s<N> step sizes have no Newton-LM analogue; recognised, ignored.
    for key in keyvals:
        if _S_SLOT.match(key):
            used_names.add(key)


def _resolve_named_coordinates(items, cfg, model, used_names):
    """Resolve file then CLI assignments by named target, including aliases."""
    for key, value in items:
        # These structural bounds were already read as BBCConfig fields.
        if key in ("logmass_min", "logmass_max"):
            continue
        try:
            if key in model.cut_aliases:
                var, side = model.cut_aliases[key]
                prev = cfg.get('lcpar_range', {}).get(var, (None, None))
                val = float(value)
                cfg.setdefault('lcpar_range', {})[var] = (val, prev[1]) if side == "lo" else (prev[0], val)
            elif key.startswith("lcpar_range."):
                var = key.split(".", 1)[1]
                bounds = tuple(float(v) for v in value.split(","))
                if not var or len(bounds) != 2:
                    raise ValueError()
                cfg.setdefault('lcpar_range', {})[var] = bounds
            else:
                target = getattr(model, "bbc_input_aliases", {}).get(key)
                if target is None and key.startswith(("mucovscale_bins.", "ccprior_bins.")):
                    target = key.split(".", 1)
                if target is None:
                    continue
                field, var = target
                if not var:
                    raise ValueError()
                cfg.setdefault(field, {})[var] = int(value)
        except ValueError:
            raise SNANAError(f"Invalid named BBC setting {key}={value}",
                             function_name="parse_bbc_input") from None
        used_names.add(key)


# refused keys, matched exactly
_REJECTED_KEYS = {
    **refused('set intrinsic scatter with sig1= (or sigmB=)',
              "sigint"),
    **refused('different standardisation slopes above and below a stretch break are not supported',
              "alpha1"),
    **refused('intrinsic stretch/colour scatter and its correlations with magnitude scatter are not supported',
              "sigx1", "sigc", "sig2", "sig3", "xi01", "xi0c", "xi1c"),
    **refused('use p7 and u7 for the host-mass split',
              "host_logmass_split"),
    **refused('use zpecerr for the peculiar-velocity uncertainty in redshift units', "pecv"),
    **refused("only the numbered forms blindpar9 and blindpar11 are supported",
              "blindpar"),
    # Event synchronisation prepares pass-1 acceptance without fitting. If
    # opt_chi2max >= 2 is implemented, that pass must fit before selecting.
    **refused('reapplying chi2max selection with recalculated event chi-squares is not supported',
              "opt_chi2max"),
    # ZPOLY_*: SALT2mu uses whichever row was read last (SALT2mu.c:9276)
    **refused('redshift- and survey-dependent intrinsic-scatter covariance is not supported',
              "ZPOLY_sigmB", "ZPOLY_sigx1", "ZPOLY_sigc", "ZPOLY_xi01",
              "ZPOLY_xi0c", "ZPOLY_xi1c"),
    **refused('split-sample execution is configured through the Pippin interface, not BBC inputs',
              "NSPLITRAN", "JOBID_SPLITRAN"),
    **refused('SNANA diagnostic output and tracing controls are not supported',
              "NDUMPLOG", "debug_flag", "debug_malloc", "debug_mucovscale",
              "snid_mucovdump", "cidlist_debug_biascor"),
    **refused('the SNANA population-reweighting subprocess interface is not supported',
              "SUBPROCESS_FILES", "SUBPROCESS_HELP", "SUBPROCESS_ISEED",
              "SUBPROCESS_OPTMASK", "SUBPROCESS_SIMREF_FILE",
              "SUBPROCESS_CID_REWGT_DUMP", "SUBPROCESS_NEVT_SIM_PRESCALE",
              "SUBPROCESS_STDOUT_CLOBBER", "SUBPROCESS_VARNAMES_GENPDF"),
    **refused('switches that restore earlier SNANA calculations are not supported',
              "restore_bug_mucovadd", "restore_bug2_mucovadd",
              "restore_bug_mucovscale", "restore_bug_mumodel_zhel",
              "restore_bug_muzerr", "restore_bug_sigint0",
              "restore_bug_zmax_biascor", "restore_des5yr", "restore_sigz"),
    **refused('MINUIT profile uncertainties are not supported in BBC',
              "minos", "minos2"),
    **refused('manual CPU-thread control through BBC inputs is not supported',
              "nthread"),
    **refused('selecting and rewriting a table without standardisation is not supported',
              "cutwin_only"),
}


def _check_repeated_and_cli_keys(repeated, cli_keys, used_names, valueless) -> None:
    """Abort on a key repeated within the file or within the command line
    (uniqueOverlap), and on a command-line key nothing reads
    (override_parFile). chi2max keys may repeat."""
    def used(key):
        return _ALIASES.get(key, key) in used_names

    dup = sorted(k for k in repeated if used(k) and not k.startswith("chi2max"))
    if dup:
        raise SNANAError(f"key(s) given more than once: {', '.join(dup)}",
                         function_name="parse_bbc_input")
    invalid = sorted(k for k in cli_keys
                     if not used(k) or _ALIASES.get(k, k) in valueless)
    if invalid:
        raise SNANAError(f"Invalid command-line override: {', '.join(invalid)}",
                         function_name="parse_bbc_input")


def _warn_unrecognized(keyvals, used_names) -> None:
    """A key SALT2mu does not know is a typo, and an input file carrying one
    should still run, so it warns; the keys jax refuses were refused as
    the input spelt them, before aliasing."""
    unknown = sorted(k for k in keyvals if k not in used_names)
    if unknown:
        log.warn(logger, "BBC", f"Unrecognized SALT2mu key(s) ignored: {', '.join(unknown)}")


def _split_list(value: str) -> List[str]:
    if not value:
        return []
    return [v for v in value.split(",") if v]


def _split_idlist(value: str) -> List[int]:
    """parse_select_IDLIST (SALT2mu.c:20703): comma- or plus-separated integer IDs;
    mixing the two separators aborts. An empty list selects everything."""
    if not value:
        return []
    if "," in value and "+" in value:
        raise SNANAError(f"cannot mix ',' and '+' separators in '{value}'",
                         function_name="parse_bbc_input")
    sep = "+" if "+" in value else ","
    return [int(v) for v in value.split(sep) if v]


def _floats(value: str) -> List[float]:
    return [float(v) for v in _split_list(value)]


def _float_tuple(value: str) -> Tuple[float, ...]:
    return tuple(_floats(value))


# per field type, the cast take() applies and what then makes it the field's value
_READ_AS = {
    int: (int, None), float: (float, None), str: (str, None), bool: (int, bool),
    List[str]: (str, _split_list), List[float]: (str, _floats),
    List[int]: (str, _split_idlist), Tuple[float, ...]: (str, _float_tuple),
}


def output_path(config, suffix: str) -> str:
    """Path for an output file. SNANA leaves PREFIX unset ("NONE", SALT2mu.c:6492)
    and has no default filename, so writing without prefix= is an error."""
    if config.prefix == "NONE":
        raise SNANAError("prefix= is required to write output files")
    return f"{config.prefix}{suffix}"
