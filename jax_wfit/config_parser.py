"""Reads WFIT keys, from the command line or a YAML file, into a WFITConfig.

wfit.exe has no input file, so its command line is the input: every flag here
is a SNANA option name, and a YAML key spells the same field. Both routes end
in :func:`build`, which is the only place a value becomes configuration.
"""

import argparse
from pathlib import Path
from typing import Any, Mapping, get_args

import yaml

from jax_snlc_core.utils import SNANAError

from jax_snlc_core.utils.input_settings import (build_dataclass, dataclass_settings,
    settings_parser, parsed_settings, merge_settings, configuration_settings)

from .config import BaoSample, MethodName, SigType, WFITConfig

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
    types, owners = dataclass_settings(WFITConfig)
    settings = merge_settings(read_keys(supplied.config) if supplied.config else {},
                              cli_keys(args), parsed_settings(supplied, types, owners))
    return build(settings), args.verbose
