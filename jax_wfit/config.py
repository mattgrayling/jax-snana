"""WFIT configuration, YAML inputs, and command-line settings."""

import argparse
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, get_args

import yaml

from jax_snlc_core.input_settings import (build_dataclass, configuration_settings,
                                          dataclass_settings, merge_settings, parsed_settings,
                                          settings_parser)
from jax_snlc_core.logging import SNANAError


#: Refused rather than ignored, because ignoring them would either withhold an
#: output or silently change the fit.
_REJECTED_KEYS = {
    "csv": (
        'use YAML or text output'
    ),
    "fitswrite": (
        'use YAML or text output'
    ),
    "debug_flag": 'SNANA diagnostic output controls are not supported',
    "speed_flag_chi2": 'SNANA grid-acceleration controls are not supported',
    "speed_nsig_multiplier": 'SNANA grid-acceleration controls are not supported',
    "refit": (
        'estimating additional intrinsic scatter and refitting is not supported'
    ),
    "weightmin": "the chi2-grid file is written without a weight threshold",
    "ndump_mucov": 'diagnostic covariance dumps are not supported',
}

#: argparse dests that spell something other than a single config field.
_TRANSLATED = frozenset({
    "hubble_diagrams", "config", "verbose", "lcdm", "wa",
    "mucovsys_file", "cov_total", "mucovtot_inv_file", "mucovtot_factorized_file",
    "no_omega_m_prior", "cmb", "cmb_sim", "bao", "bao_sim",
    "blind", "blind_auto", "unblind", "muerr_ideal", "minchi2",
    "double_precision",
})


ModelName = Literal["lcdm", "wcdm", "w0wa"]
SigType = Literal["std", "68"]
MethodName = Literal["laplace", "nuts", "grid"]
CovarianceKind = Literal["systematics", "total", "total_inverse", "factorized"]
BlindingMode = Literal["none", "blind", "auto"]

BaoSample = Literal["DESI_2024", "SDSS4_2020"]

#: The cosmology SNANA assumes where none is given: the mean of the Gaussian
#: Omega_m prior and the COSPAR_SIM the residual file reports.
OMEGA_MATTER_DEFAULT = 0.315
W0_DEFAULT = -1.0
WA_DEFAULT = 0.0


@dataclass
class WFITConfig:
    """Every option the cosmology fit accepts, from the command line or YAML."""

    #: Double or single precision. Omit to use the active precision in Python;
    #: command-line runs default to single.
    double_precision: bool | None = None

    # Hubble diagram and covariance
    #: Binned M0DIF or unbinned FITRES to fit; a list fits each as an independent realisation.
    hubble_diagram: str | list[str] = ""
    #: Distance covariance matrix; a list supplies one per realisation.
    covariance: str | list[str] | None = None
    #: Covariance form: systematics, total, total_inverse, or factorized (diag + U U.T).
    covariance_kind: CovarianceKind | None = None
    zmin: float = 0.0  #: Discard supernovae below this redshift.
    zmax: float = 9.0  #: Discard supernovae above this redshift.
    #: Intrinsic scatter in magnitudes, added in quadrature to every distance error.
    snrms: float = 0.0
    muerr_force: float | None = None  #: Replace every distance error with this value.
    #: Column to read distance errors from, when it is not the usual MUERR.
    varname_muerr: str | None = None

    # Idealised data generated at the simulation truth (-muerr_ideal)
    #: Replace the observed distances with the simulation truth plus fresh Gaussian noise.
    muerr_ideal: bool = False
    #: Coefficients of a polynomial in redshift giving the idealised distance error.
    muerr_ideal_polynomial: tuple[float, ...] | None = None
    #: Take idealised errors from the input file's own column instead of a polynomial.
    muerr_ideal_use_input: bool = False

    # Cosmology and grid extent
    #: Cosmology fitted: flat LCDM, constant-w wCDM, or the w0-wa CPL parameterisation.
    model: ModelName = "wcdm"
    ommin: float = 0.0  #: Lower bound on Omega_m for grid evaluation, the NUTS prior and Laplace optimisation.
    ommax: float = 1.0  #: Upper bound on Omega_m for grid evaluation, the NUTS prior and Laplace optimisation.
    wmin: float = -2.0  #: Lower bound on w0 for grid evaluation, the NUTS prior and Laplace optimisation.
    wmax: float = 0.0  #: Upper bound on w0 for grid evaluation, the NUTS prior and Laplace optimisation.
    wamin: float = -4.0  #: Lower bound on wa for grid evaluation, the NUTS prior and Laplace optimisation (w0wa only).
    wamax: float = 4.0  #: Upper bound on wa for grid evaluation, the NUTS prior and Laplace optimisation (w0wa only).
    h0: float = 70.0  #: Hubble constant for the distance-modulus model, in km/s/Mpc.
    #: Include Omega_r = 0.9e-4 in distances; CMB R and BAO DH remain
    #: radiation-free, matching SNANA wfit's prior convention.
    radiation: bool = True
    # The magnitude offset
    #: Width, in magnitudes, of the Gaussian prior on the magnitude offset
    #: which is profiled out analytically.
    intercept_sigma: float = 5.0 * math.log10(1.0 + 1000.0 / 70.0)

    # Priors
    ompri_enabled: bool = True  #: Apply the Gaussian prior on Omega_m.
    ompri: float = OMEGA_MATTER_DEFAULT  #: Central value of the Gaussian prior on Omega_m.
    #: Width of the Omega_m prior; SNANA treats values of 1 or more as uninformative.
    dompri: float = 99.0
    cmb: bool = False  #: Apply the CMB shift-parameter prior.
    #: Recompute the CMB prior's centre from the simulation truth instead of the literature value.
    cmb_sim: bool = False
    Rcmb: float = 1.710  #: CMB shift parameter R.
    sigma_Rcmb: float = 0.019  #: Uncertainty on the CMB shift parameter.
    #: Redshift of last scattering used to evaluate the shift parameter.
    cmb_redshift: float = 1089.0
    bao: str | None = None  #: BAO sample supplying the distance-ratio prior.
    #: Recompute the BAO prior's means from the simulation truth, using this sample's redshifts.
    bao_sim: str | None = None
    bao_h0: float = 67.4  #: Hubble constant the BAO distance ratios are built with, in km/s/Mpc.
    bao_h0_sigma: float = 0.5  #: Uncertainty on bao_h0, which inflates the BAO covariance.
    bao_rd: float = 147.05  #: Sound horizon normalising the BAO distance ratios, in Mpc.
    bao_rd_sigma: float = 0.30  #: Uncertainty on bao_rd, which inflates the BAO covariance.
    #: Centre of the Gaussian penalty evaluated at the fixed bao_rd, in Mpc.
    #: This adds a constant to chi-squared and does not fit the sound horizon.
    rd_prior: float = 147.0
    #: Width in Mpc of the penalty on the fixed bao_rd.
    rd_prior_sig: float = 1.0e8
    #: Randomly displace the prior centres, for ensemble studies of prior sensitivity.
    perturb_priors: bool = False
    #: Seed for those displacements; 1 derives one from the Hubble diagram itself.
    prior_perturbation_seed: int = 12345

    # Simulation truth, for validation and simulated priors
    #: INFO.YML giving the cosmology the data were simulated at; a list gives one per realisation.
    truth_info: str | list[str] | None = None
    om_sim: float | None = None  #: Simulated Omega_m, overriding truth_info.
    w0_sim: float | None = None  #: Simulated w0, overriding truth_info.
    wa_sim: float | None = None  #: Simulated wa, overriding truth_info.

    # Blinding and HDIBC
    #: Whether reported values are offset: never, always, or only when the input is real data.
    blind_mode: BlindingMode = "none"
    blind_seed: int = 48901  #: Seed for the blinding offsets.
    #: Interpolate between two Hubble diagrams built with different bias-correction cosmologies.
    hdibc: bool = False
    #: The two INFO.YML files describing those bias-correction cosmologies.
    hdibc_info: list[str] | None = None

    # Inference
    #: Inference method: Laplace approximation, NUTS sampling, or an exhaustive grid.
    #: Omit to use grid for requested grid outputs and Laplace otherwise.
    method: MethodName | None = None
    _derived_settings: dict = field(default_factory=dict, init=False, repr=False)
    seed: int = 12345  #: Seed for the sampler and for posterior draws.
    nuts_chains: int = 4  #: Number of independent NUTS chains.
    nuts_warmup: int = 1_000  #: NUTS warm-up iterations per chain, discarded before sampling.
    nuts_samples: int = 1_000  #: NUTS samples retained per chain.
    target_accept: float = 0.9  #: Target acceptance probability for the NUTS step-size adaptation.
    #: The fit is reported unsuccessful if any parameter's R-hat exceeds this.
    max_rhat: float = 1.05
    min_ess: float = 400.0  #: Advisory threshold on effective sample size; falling below it warns.
    omsteps: int = 81  #: Number of Omega_m grid points.
    wsteps: int = 201  #: Number of w0 grid points.
    wasteps: int = 101  #: Number of wa grid points.
    #: Grid points evaluated per chunk; 0 sizes the chunk from the Hubble-diagram
    #: length, so memory use stays bounded.
    grid_chunk_size: int = 0

    #: Scalar summary uncertainty: half the central 68.3% interval width,
    #: or the standard deviation.
    sig_type: SigType = "68"
    #: Report the marginalised mean instead of the best-fit point. On the
    #: command line ``-minchi2`` selects the best-fit point and ``-marg`` the mean.
    marg: bool = True

    # Outputs
    #: Summary YAML; defaults to the input's name with .wfit.yaml appended.
    cospar_yaml: str | None = None
    outfile_resid: str | None = None  #: Per-supernova residuals about the fitted cosmology.
    outfile_cospar: str | None = None  #: Fitted parameters in SNANA's cospar text format.
    outfile_chi2grid: str | None = None  #: The full chi-square grid; requires the grid method.
    #: Marginalised one-dimensional probabilities; requires the grid method.
    outfile_prob1d: str | None = None
    #: Posterior samples, as NPZ; requires the NUTS method.
    output_samples: str | None = None
    #: The total distance covariance the fit used, as text.
    output_covariance: str | None = None
    #: The inverse of the total distance covariance the fit used, as text.
    outfile_mucovtot_inv: str | None = None
    label: str = "none"  #: Free-text label copied into the outputs.

    def __post_init__(self):
        if self.method is None:
            self.method = 'grid' if self.outfile_chi2grid or self.outfile_prob1d else 'laplace'
            self._derived_settings['method'] = (None, self.method)

    def resolve_paths(self) -> None:
        """Fill in output paths that default to the input's name."""
        primary = (
            self.hubble_diagram
            if isinstance(self.hubble_diagram, str)
            else self.hubble_diagram[0]
        )
        if self.cospar_yaml is None:
            self.cospar_yaml = f"{primary}.wfit.yaml"
        if self.outfile_cospar is None:
            # wfit writes the cospar unconditionally, naming it from the HD
            # input when no -outfile_cospar is given (wfit.c:5151-5153)
            self.outfile_cospar = f"{primary}.cospar"

    def bounds(self, name: str) -> tuple[float, float]:
        """Return the grid/prior support of one free parameter."""
        return {
            "omega_m": (self.ommin, self.ommax),
            "w0": (self.wmin, self.wmax),
            "wa": (self.wamin, self.wamax),
        }[name]

    def steps(self, name: str) -> int:
        """Return the number of grid points along one free parameter."""
        return {
            "omega_m": self.omsteps,
            "w0": self.wsteps,
            "wa": self.wasteps,
        }[name]

    def omega_m_prior_applies(self) -> bool:
        """Whether the Gaussian Omega_m prior contributes to the chi-square.

        A BAO prior replaces it rather than adding to it: ``get_chi2_priors``
        takes one branch or the other (wfit.c:4446-4460).
        """
        return self.ompri_enabled and not self.bao

    def validate(self) -> None:
        def fail(message: str) -> None:
            raise SNANAError(message, function_name="WFITConfig.validate")

        if self.zmin > self.zmax:
            fail("zmin must not exceed zmax")
        if self.snrms < 0:
            fail("snrms must be non-negative")
        if self.muerr_force is not None and self.muerr_force <= 0:
            fail("muerr_force must be positive")
        if bool(self.covariance) != bool(self.covariance_kind):
            fail("covariance and covariance_kind must be specified together")
        if (
            self.covariance_kind in {"total", "total_inverse", "factorized"}
            and self.muerr_force is not None
        ):
            fail("muerr_force requires diagonal/systematics covariance")
        if self.covariance_kind == "total_inverse" and self.snrms != 0:
            fail("snrms cannot modify an already-inverted covariance")

        if self.muerr_ideal:
            rules = (
                int(self.muerr_ideal_polynomial is not None)
                + int(self.muerr_ideal_use_input)
            )
            if rules != 1:
                fail(
                    "idealised data require exactly one uncertainty rule: "
                    "muerr_ideal_polynomial or muerr_ideal_use_input"
                )
            if (
                self.muerr_ideal_polynomial is not None
                and len(self.muerr_ideal_polynomial) == 0
            ):
                fail("muerr_ideal_polynomial must contain coefficients")
            if self.covariance_kind in {"total", "total_inverse", "factorized"}:
                fail(
                    "idealised data cannot replace uncertainties in a total "
                    "or total-inverse covariance"
                )
            if self.hdibc:
                fail("idealised data are not defined for HDIBC")
            explicit_truth = all(
                value is not None
                for value in (self.om_sim, self.w0_sim, self.wa_sim)
            )
            if self.truth_info is None and not explicit_truth:
                fail(
                    "idealised data require simulation truth from truth_info "
                    "or explicit om_sim/w0_sim/wa_sim"
                )

        if self.model not in get_args(ModelName):
            fail(f"unknown model {self.model!r}")
        if self.intercept_sigma <= 0:
            fail("intercept_sigma must be positive")
        for name in ("omega_m", "w0", "wa"):
            lo, hi = self.bounds(name)
            if lo >= hi:
                fail(f"invalid bounds for {name}: {(lo, hi)}")
        if self.sig_type not in get_args(SigType):
            fail(f"unknown sig_type {self.sig_type!r}")
        if not self.marg and self.method == "nuts":
            fail(
                "-minchi2 has no meaning for NUTS, whose reported point is "
                "already the posterior mean; use --method laplace or grid"
            )

        if self.method not in get_args(MethodName):
            fail(f"unknown inference method {self.method!r}")
        if self.blind_mode not in get_args(BlindingMode):
            fail(f"unknown blinding mode {self.blind_mode!r}")
        if self.grid_chunk_size < 0:
            fail("grid_chunk_size must be positive, or 0 for automatic")

        if self.hdibc:
            if self.covariance_kind == "factorized":
                fail("factorized covariance is not supported with HDIBC; supply a dense covariance")
            if (
                not isinstance(self.hubble_diagram, list)
                or len(self.hubble_diagram) != 2
            ):
                fail("HDIBC requires exactly two Hubble-diagram paths")
            if self.hdibc_info is None or len(self.hdibc_info) != 2:
                fail("HDIBC requires two bias-correction INFO.YML paths")
            if isinstance(self.truth_info, list):
                fail(
                    "HDIBC produces one combined realisation and therefore "
                    "accepts at most one truth INFO.YML path"
                )

        if self.cmb and self.ommax <= 0:
            fail("CMB prior requires positive omega_m support")
        if self.ompri_enabled and self.dompri <= 0:
            fail("Omega_m prior sigma must be positive")
        if self.cmb and self.sigma_Rcmb <= 0:
            fail("CMB prior sigma must be positive")
        if self.cmb_sim and not self.cmb:
            fail("simulation-centred CMB prior requires CMB to be enabled")
        if self.bao_sim and not self.bao:
            fail("simulation-centred BAO prior requires BAO to be enabled")
        for name, sample in (("bao", self.bao), ("bao_sim", self.bao_sim)):
            if sample is not None and sample not in get_args(BaoSample):
                fail(
                    f"unsupported {name} sample {sample!r}; expected one of "
                    + ", ".join(get_args(BaoSample))
                )
        if self.bao and self.rd_prior_sig <= 0:
            fail("rd prior sigma must be positive")
        if self.perturb_priors and not (self.omega_m_prior_applies()
                                        or self.cmb):
            fail(
                "prior perturbations require an enabled Gaussian Omega_m "
                "or CMB prior"
            )

        if self.truth_info is not None and any(
            value is not None
            for value in (self.om_sim, self.w0_sim, self.wa_sim)
        ):
            fail("truth_info cannot be combined with explicit truth parameters")
        if self.outfile_chi2grid and self.method != "grid":
            fail("grid output requires method=grid")
        if self.outfile_prob1d and self.method != "grid":
            fail("prob1d output requires method=grid")

    def to_dict(self) -> dict[str, Any]:
        return {name: value for name, value in asdict(self).items()
                if not name.startswith('_')}


def build_parser() -> argparse.ArgumentParser:
    """The wfit command line, one argument per wfit.exe option."""
    parser = argparse.ArgumentParser(
        description="JAX WFIT cosmology fitter for FITRES/M0DIF Hubble diagrams",
        # wfit.exe matches option names exactly (strcasecmp), and prefix
        # matching would make -cospar mean -cospar_yaml
        allow_abbrev=False,
    )
    parser.add_argument(
        "hubble_diagrams",
        nargs="*",
        help="one or more shape-compatible Hubble-diagram realisations",
    )
    parser.add_argument("--config")
    parser.add_argument("--method", choices=get_args(MethodName))
    model = parser.add_mutually_exclusive_group()
    model.add_argument("-lcdm", action="store_true")
    model.add_argument("-wa", action="store_true")

    covariance = parser.add_mutually_exclusive_group()
    covariance.add_argument("-mucovsys_file", "-mucov_file")
    covariance.add_argument("--cov-total")
    covariance.add_argument("-mucovtot_inv_file")
    covariance.add_argument("-mucovtot_factorized_file")
    parser.add_argument("-h0", type=float)
    parser.add_argument("-zmin", type=float)
    parser.add_argument("-zmax", type=float)
    parser.add_argument("-snrms", type=float)
    parser.add_argument("-muerr_force", type=float)
    parser.add_argument("-varname_muerr")
    parser.add_argument("-sig_type", choices=get_args(SigType))
    estimate = parser.add_mutually_exclusive_group()
    estimate.add_argument("-marg", action="store_true")
    estimate.add_argument("-minchi2", action="store_true")

    parser.add_argument("-ompri", type=float)
    parser.add_argument("-dompri", type=float)
    parser.add_argument("--no-omega-m-prior", action="store_true")
    parser.add_argument("-cmb", action="store_true")
    parser.add_argument("-cmb_sim", action="store_true")
    parser.add_argument("-Rcmb", type=float)
    parser.add_argument("-sigma_Rcmb", type=float)
    parser.add_argument("-bao", choices=get_args(BaoSample))
    parser.add_argument("-bao_sim", choices=get_args(BaoSample))
    parser.add_argument("-rd_prior", type=float)
    parser.add_argument("-rd_prior_sig", type=float)
    parser.add_argument("--perturb-priors", action="store_true")
    parser.add_argument("--prior-perturbation-seed", "-ranseed_Rcmb",
                        type=int)
    parser.add_argument("--truth-info", nargs="+")
    parser.add_argument("-om_sim", type=float)
    parser.add_argument("-w0_sim", type=float)
    parser.add_argument("-wa_sim", type=float)
    blinding = parser.add_mutually_exclusive_group()
    blinding.add_argument("-blind", action="store_true")
    blinding.add_argument("-blind_auto", action="store_true")
    blinding.add_argument("-unblind", action="store_true")
    parser.add_argument("-blind_seed", type=int)
    parser.add_argument("--hdibc", action="store_true")
    parser.add_argument("--hdibc-info", nargs=2, metavar=("INFO0", "INFO1"))
    parser.add_argument("-muerr_ideal")

    parser.add_argument("-ommin", type=float)
    parser.add_argument("-ommax", type=float)
    parser.add_argument("-omsteps", type=int)
    parser.add_argument("-wmin", "-w0min", type=float)
    parser.add_argument("-wmax", "-w0max", type=float)
    parser.add_argument("-wsteps", "-w0steps", type=int)
    parser.add_argument("-wamin", type=float)
    parser.add_argument("-wamax", type=float)
    parser.add_argument("-wasteps", type=int)
    parser.add_argument("--grid-chunk-size", type=int)

    parser.add_argument("--seed", type=int)
    parser.add_argument("--nuts-chains", type=int)
    parser.add_argument("--nuts-warmup", type=int)
    parser.add_argument("--nuts-samples", type=int)
    parser.add_argument("--target-accept", type=float)

    parser.add_argument("-cospar_yaml")
    parser.add_argument("-outfile_resid", "-resid")
    parser.add_argument("-outfile_cospar", "-cospar")
    parser.add_argument("-outfile_chi2grid")
    parser.add_argument("-outfile_prob1d")
    parser.add_argument("--output-samples")
    parser.add_argument("--output-covariance")
    parser.add_argument("-outfile_mucovtot_inv")
    parser.add_argument("-label")
    parser.add_argument("-double_precision", type=int,
                        default=None,
                        help="1 for double precision, 0 for single")
    for name in _REJECTED_KEYS:
        parser.add_argument(f"-{name}", nargs="?", const=True)
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def cli_keys(args: argparse.Namespace) -> dict[str, object]:
    """Translate parsed flags into config field names.

    Only keys the user actually supplied are emitted, so a flag left unset
    never overwrites the input file. SNANA's presets are expanded here rather
    than in `build`, because they are a property of the flag: an input file
    composes the same fields explicitly and must not have them overridden.

    A refused flag aborts here rather than at `build`, so a caller that only
    translates flags, as the Pippin reader does, refuses while it reads.
    """
    # unset flags are None, except store_true ones, which are False
    raw: dict[str, object] = {
        key: value
        for key, value in vars(args).items()
        if key not in _TRANSLATED and value is not None and value is not False
    }

    if args.hubble_diagrams:
        raw["hubble_diagram"] = (
            args.hubble_diagrams[0]
            if len(args.hubble_diagrams) == 1 else args.hubble_diagrams
        )
    # one truth file serves every realisation, as one Hubble diagram does
    if args.truth_info:
        raw["truth_info"] = (
            args.truth_info[0]
            if len(args.truth_info) == 1 else args.truth_info
        )
    if args.lcdm:
        raw["model"] = "lcdm"
    if args.wa:
        raw["model"] = "w0wa"

    for path, kind in (
        (args.mucovsys_file, "systematics"),
        (args.cov_total, "total"),
        (args.mucovtot_inv_file, "total_inverse"),
        (args.mucovtot_factorized_file, "factorized"),
    ):
        if path:
            raw["covariance"], raw["covariance_kind"] = path, kind

    if args.no_omega_m_prior:
        raw["ompri_enabled"] = False
    if args.cmb or args.cmb_sim:
        raw["cmb"] = True
        raw["ompri_enabled"] = True
        raw["dompri"] = 0.9
        raw["cmb_sim"] = bool(args.cmb_sim)
    if args.bao or args.bao_sim:
        raw["bao"] = args.bao or args.bao_sim
        raw["bao_sim"] = args.bao_sim
    if args.om_sim is not None and args.ompri is None:
        raw["ompri"] = args.om_sim   # -om_sim recentres the prior (wfit.c:1051)
    if args.prior_perturbation_seed is not None:
        # SNANA's -ranseed_Rcmb is the trigger as well as the seed
        raw["perturb_priors"] = args.prior_perturbation_seed > 0
    if args.minchi2:
        raw["marg"] = False

    if args.blind:
        raw["blind_mode"] = "blind"
    if args.blind_auto:
        raw["blind_mode"] = "auto"
    if args.unblind:
        raw["blind_mode"] = "none"

    if args.muerr_ideal is not None:
        raw["muerr_ideal"] = True
        if args.muerr_ideal.lower() == "muerr":
            raw["muerr_ideal_use_input"] = True
        else:
            try:
                raw["muerr_ideal_polynomial"] = tuple(
                    float(value) for value in args.muerr_ideal.split(",")
                )
            except ValueError as error:
                raise SNANAError(
                    "-muerr_ideal must be 'muerr' or comma-separated "
                    "polynomial coefficients",
                    function_name="cli_keys",
                ) from error
    if args.double_precision is not None:
        raw["double_precision"] = bool(args.double_precision)
    _check_rejected_keys(raw)
    return raw


def read_keys(path: str | Path) -> dict[str, Any]:
    """Read a flat WFIT YAML file into raw key/value pairs.

    Syntax only: the keys and values are neither checked nor interpreted,
    which `build` does.
    """
    with open(path, "rt") as stream:
        values = yaml.safe_load(stream)
    if not isinstance(values, Mapping):
        raise SNANAError(
            f"WFIT configuration {path} must be a non-empty YAML mapping",
            function_name="read_keys",
        )
    return dict(values)


def _check_rejected_keys(raw: Mapping[str, Any]) -> None:
    """Abort on keys jax refuses, whichever input route named them."""
    found = sorted(set(raw) & set(_REJECTED_KEYS))
    if found:
        raise SNANAError(
            "wfit option(s) not supported:\n"
            + "\n".join(f"  {k}: {_REJECTED_KEYS[k]}" for k in found),
            function_name="_check_rejected_keys",
        )


def build(raw: Mapping[str, Any] | WFITConfig, overrides=None) -> WFITConfig:
    """Interpret raw keys into a WFITConfig, in one pass.

    Both the YAML and the command line reach here as field names, so this is
    the only place a value becomes configuration. A key on the refused table
    aborts with its reason; unknown keys are rejected.
    """
    if isinstance(raw, WFITConfig):
        raw = configuration_settings(raw)
    raw = merge_settings(raw, overrides or {})
    _check_rejected_keys(raw)
    config = build_dataclass(WFITConfig, raw)
    if config.hdibc:
        config.hubble_diagram = _split_pair(config.hubble_diagram)
        config.covariance = _split_pair(config.covariance)
    return config


def _split_pair(value: Any) -> Any:
    """HDIBC names its two inputs as one comma-separated string."""
    if isinstance(value, str) and "," in value:
        return value.split(",")
    return value


def load_config(path: str | Path) -> WFITConfig:
    """Load a flat WFIT YAML configuration."""
    return build(read_keys(path))


def parse_command(argv=None):
    """Named options and SNANA flags converge on the same typed settings."""
    named = settings_parser(WFITConfig, description=(
        'Fit a Hubble diagram supplied with --hubble-diagram or as a positional path.'))
    named.add_argument('--config', help='Read settings from a YAML file')
    supplied, snana = named.parse_known_args(argv)
    args = build_parser().parse_args(snana)
    types, _ = dataclass_settings(WFITConfig)
    settings = merge_settings(read_keys(supplied.config) if supplied.config else {},
                              cli_keys(args), parsed_settings(supplied, types, readers={}))
    return build(settings), args.verbose
