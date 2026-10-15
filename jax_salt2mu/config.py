"""Typed configuration for the BBC/SALT2mu fit.

One flat :class:`BBCConfig` whose fields are SALT2mu's own input keys, grouped by
comment. Model-specific cut fields, nuisance parameters and keys are deliberately
absent: cuts are resolved through the model's ``cut_aliases`` onto a generic
CUTWIN list, nuisance values onto its ``nuisance_spec``, and the model's own BBC
keys into ``model_inputs``, so adding a distance model needs no config change.

:func:`resolve` is the one place SALT2mu's setup-time rewrites are applied.
"""

import logging
import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

from jax_snlc_core.physics.constants import LIGHT_KM
from jax_snlc_core.models import model_input_keys
from jax_snlc_core.utils import snana_log as slog
from jax_snlc_core.utils.error_handling import SNANAError

logger = logging.getLogger(__name__)

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

# blindflag bits, mirroring SALT2mu.c:1619-1621.
BLINDMASK_MUz = 1        # add cos(10*z) to each M0DIF bin
BLINDMASK_FIXPAR = 2     # offset the fixed reference OL/w0
BLINDMASK_SIM = 64       # extend blinding to sims, which are otherwise unblinded

# blind offset = amplitude * cos(phase), per fit-parameter slot (SALT2mu.c:6749-6753).
BLINDPAR_DEFAULT = {9: (0.06, 23434.0), 11: (0.20, 8430.0)}
BLIND_COSINE_M0 = 10.0   # cos(BLIND_COSINE_M0 * z) for MUz (SALT2mu.c:25191)

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


# blindpar slot -> the config attribute it offsets.
_BLIND_SLOTS = {9: "omega_l", 11: "w0"}


def resolve_blinding(config: "BBCConfig", isdata_real: bool) -> "BBCConfig":
    """The config with the reference cosmology offset by ``amp * cos(phase)`` and
    ``blindflag`` reduced to what actually applies (apply_blindpar / ISBLIND_FIXPAR,
    SALT2mu.c:3154/25141).

    ``unblind`` keeps the pre-offset values, which SNANA holds in COSPAR_UNBLIND
    (:22195) and the CC prior reads. Called once the data read has established
    real-vs-sim, before any reference distance is computed (:6905).
    """
    blindflag = config.blindflag
    if not isdata_real and not (blindflag & BLINDMASK_SIM):
        blindflag = 0        # sims are unblinded unless asked for (:8844)
    if not (blindflag & BLINDMASK_FIXPAR):
        return replace(config, blindflag=blindflag)
    unblind, blinded = dict(config._unblind), {}
    for slot, attr in _BLIND_SLOTS.items():
        amp, phase = config.blindpar.get(slot, (0.0, 0.0))
        if amp == 0.0:
            continue
        base = unblind.setdefault(attr, getattr(config, attr))
        blinded[attr] = base + amp * math.cos(phase)
    return replace(config, blindflag=blindflag, _unblind=unblind, **blinded)


def unblinded(config: "BBCConfig") -> "BBCConfig":
    """The config with the reference cosmology back at the values the user gave
    (COSPAR_UNBLIND, SALT2mu.c:22195)."""
    return replace(config, **config._unblind) if config._unblind else config


@dataclass
class BBCConfig:
    """Everything the bias-correction fit reads from its SALT2mu input file.

    Field names are SALT2mu's own input keys, so a key in the input file and an
    attribute here are spelled the same. Keys which SALT2mu accepts but which
    have no effect there, and those not implemented here, are listed in
    ``config_parser._REJECTED_KEYS`` and are refused with a reason rather than
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


def map_bins(model, config, field):
    """Effective named axes, preserving model order and omitting zero overrides."""
    counts = {**getattr(model, field, {}), **getattr(config, field)}
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
    for field_name in ("mucovscale_bins", "ccprior_bins"):
        defaults = getattr(model, field_name, {})
        for mapping, minimum in ((defaults, 1), (getattr(config, field_name), 0)):
            for name, count in mapping.items():
                if isinstance(count, bool) or not isinstance(count, int) or count < minimum:
                    raise SNANAError(f"{model.__name__}: {field_name}.{name} needs an integer >= {minimum}",
                                     function_name="resolve")
        for name in map_bins(model, config, field_name):
            if name not in windows:
                raise SNANAError(f"{model.__name__}: {field_name}.{name} needs a parameter range",
                                 function_name="resolve")
    bits = getattr(model, 'bbc_cut_bits', {})
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
    from prepare_cell; nothing after it may re-derive an effective value.

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
        slog.warn(logger, "BBC", "scalePCC(u13,p13)=0,0 --> turn off CC prior")
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
        slog.warn(logger, "BBC",
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
