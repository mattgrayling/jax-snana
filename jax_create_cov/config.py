"""Covariance settings, COVOPT selection, and YAML input parsing."""

import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Literal

import yaml

from jax_snlc_core import logging as log
from jax_snlc_core.input_settings import (UsedTrackingDict, build_dataclass,
                                          configuration_settings, derived_changes,
                                          merge_settings, refused)
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import parse_mushift

logger = logging.getLogger(__name__)

#: Keys refused with a reason rather than ignored, because ignoring them would
#: silently change the covariance.
_REJECTED_KEYS = {
    **refused("define MUOPT systematic scales in "
              "SYS_SCALE_FILE alongside the FITOPT ones", "MUOPT_SCALES"),
}


# split on whitespace first, as SNANA (:1418)
_BRACKET = re.compile(r"\[(.*)\]")


HDMode = Literal["binned", "unbinned", "rebin"]
CovFormat = Literal["text", "npz"]


def resolve_rebin_bins(bins, nbin_x1=0, nbin_c=0):
    """Validate named coordinates; translate the legacy SALT bin arguments."""
    if bins is not None and (nbin_x1 or nbin_c):
        raise SNANAError("use REBIN_BINS or legacy nbin_x1/nbin_c, not both",
                         function_name="resolve_rebin_bins")
    if bins is None:
        if not (nbin_x1 and nbin_c):
            raise SNANAError("rebin needs REBIN_BINS (use {} for redshift only)",
                             function_name="resolve_rebin_bins")
        bins = {"x1": nbin_x1, "c": nbin_c}
    if not isinstance(bins, Mapping):
        raise SNANAError("REBIN_BINS must map FITRES column names to bin counts",
                         function_name="resolve_rebin_bins")
    for name, count in bins.items():
        if not isinstance(name, str) or not name.isidentifier() or name == "IZBIN":
            raise SNANAError(f"invalid rebin coordinate {name!r}; IZBIN is already included",
                             function_name="resolve_rebin_bins")
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
            raise SNANAError(f"rebin count for {name} must be a positive integer",
                             function_name="resolve_rebin_bins")
    return dict(bins)


@dataclass
class Covopt:
    """Systematic contributions and uncertainty scale for one covariance matrix.

    Parsed from ``[label] [fitopt_filter,muopt_filter]`` with an optional third
    comma-separated term scaling the *error*, so the covariance is scaled by its
    square (create_covariance.py:1434).
    """

    #: Output matrix label, also recorded in INFO.YML. Specified in the first
    #: pair of brackets in a COVOPT entry.
    label: str
    #: FITOPT label filter: ``=X`` for an exact match, ``+X`` to include labels
    #: containing X, ``-X`` to exclude them, or empty to include all. First item
    #: in the second pair of brackets.
    fitopt_filter: str | dict[str, str] = ""
    #: MUOPT label filter, using the same syntax as fitopt_filter. Second item
    #: in the second pair of brackets.
    muopt_filter: str | dict[str, str] = ""
    #: Systematic uncertainty scale. Optional third item in the second pair of
    #: brackets, e.g. ``[CALIBx2] [+zpt,=DEFAULT,2.0]``.
    scale: float = 1.0


@dataclass
class Cosmology:
    """Reference cosmology of the bias-correction simulation.

    Read from the BBC output header or, when absent, the COSMOLOGY input
    block. Reference distances use H0 = 70 km/s/Mpc, following SNANA.
    """

    #: SNANA OMEGA_MATTER; matter density of the bias-correction simulation.
    #: Required whenever the ``COSMOLOGY`` block is given.
    omega_matter: float
    #: SNANA OMEGA_LAMBDA; dark energy density of the bias-correction
    #: simulation. Required whenever the ``COSMOLOGY`` block is given.
    omega_lambda: float
    w0: float = -1.0  #: SNANA w0_LAMBDA; dark energy equation of state.
    #: SNANA wa_LAMBDA; CPL coefficient in w(a) = w0 + wa(1 - a).
    wa: float = 0.0
    #: SNANA MUSHIFT; the simulation's distance-modulus shift, as (lo, hi):
    #: equal for a constant shift, a range when each event drew its own.
    mushift: tuple[float, float] = (0.0, 0.0)


@dataclass
class CreateCovConfig:
    """Covariance construction settings.

    Field names are shared by Python, flat YAML and generated command-line
    options. SNANA YAML keys and command-line flags are also accepted.
    """

    # Inputs
    #: SNANA INPUT_DIR; directory containing the bias-correction runs, one per
    #: FITOPT and MUOPT, and the ``SUBMIT.INFO`` that labels them.
    #: Required, from the file or from ``-i`` on the command line.
    input_dir: Path | None = None
    #: SNANA VERSION; sub-directory of ``INPUT_DIR`` containing this analysis,
    #: with a file per FITOPT and MUOPT named ``FITOPTnnn_MUOPTmmm.M0DIF``.
    #: Required, from the file or from ``-v`` on the command line.
    version: str = ""
    #: SNANA OUTDIR; where the Hubble diagram, the covariance files and
    #: ``INFO.YML`` are written. Required, from the file or from ``-o``.
    outdir: Path | None = None

    # Covariance matrices
    #: SNANA COVOPTS; one entry per output covariance matrix, each selecting
    #: which FITOPTs and MUOPTs contribute, written as
    #: ``[label] [fitopt filter,muopt filter]`` with an optional third
    #: comma-separated scale.
    covopts: list[Covopt] = field(default_factory=list)
    #: Explicit systematic scales, overriding matching labels in sys_scale_file.
    sys_scales: dict[str, float] = field(default_factory=dict)
    # Systematics
    #: SNANA SYS_SCALE_FILE/SYSFILE; YAML file of systematic label to scale
    #: factor, giving the size of each systematic. A label it does not name
    #: has a scale of one.
    sys_scale_file: Path | None = None
    #: SNANA EXTRA_COVS; additional covariance matrices read from disk, each
    #: given as ``label weight file``, the weight scaling the uncertainty.
    #: Needs an unbinned diagram, since the file is matched by supernova.
    extra_covs: list[tuple[str, float, str]] = field(default_factory=list)
    #: SNANA CALIBRATORS; CIDs of Cepheid calibrators. They have no
    #: redshift-dependent systematic, so the VPEC and ZSHIFT terms omit them.
    calibrators: list[str] = field(default_factory=list)

    # Hubble diagram binning
    #: SNANA UNBINNED; ``UNBINNED: true`` in the file, or ``--unbinned`` on the
    #: command line, builds the covariance per supernova from the ``.FITRES``
    #: files instead of the ``.M0DIF`` redshift bins. REBIN_BINS subdivides
    #: the BBC redshift bins by named FITRES columns. Omit mode to infer it
    #: from those settings, using binned mode otherwise.
    mode: HDMode | None = None
    _derived_settings: dict = field(default_factory=dict, init=False, repr=False)
    #: SNANA REBIN_BINS; ordered mapping of FITRES column to number of equal-width
    #: bins. An empty mapping uses only the BBC redshift bins (IZBIN).
    #: Also ``--rebin COLUMN=COUNT ...`` on the command line.
    rebin_bins: dict[str, int] | None = None
    #: number of x1 bins of the rebinned diagram. Also on the command line as
    #: ``--nbin_x1``.
    nbin_x1: int = 0
    #: number of colour bins of the rebinned diagram. Also on the command line as
    #: ``--nbin_c``.
    nbin_c: int = 0
    #: SNANA SUBTRACT_VPEC; subtract the peculiar-velocity contribution from
    #: MUERR. Selects unbinned output unless rebinning is requested.
    #: For unbinned MUERR_RENORM,
    #: retain the BBC rescaling and remove the same fractional variance.
    #: Also ``-s`` on the command line.
    subtract_vpec: bool = False

    # Output
    #: SNANA WRITE_MASK_COV; sum of the bits selecting which matrices to write:
    #: 1 the systematic covariance, 2 the inverse of the total, 4 the total.
    #: Standalone default is 2; pipelines default to compact output (8).
    #: 8 writes compact factors, with a dense inverse fallback for external covariances.
    write_mask_cov: int = 2
    #: SNANA WRITE_FORMAT_COV; ``npz`` for a compressed NumPy archive or
    #: ``text`` for gzipped text.
    write_format_cov: CovFormat = "npz"

    # Systematic selection
    #: Restrict the default covariance to the unmodified MUOPT; FITOPT
    #: contributions remain included, following SNANA.
    #: Also on the command line as ``--nosys``.
    nosys: bool = False
    #: use only this MUOPT index; -1 uses all of them. Also on the command line as
    #: ``--muopt``.
    muopt: int = -1
    #: write a separate Hubble diagram for every systematic variation, not
    #: only for the reference. Also on the command line as ``--systematic_HD``.
    systematic_hd: bool = False
    #: annotate each covariance row with its CID, survey and redshift, which
    #: forces the text format. Also on the command line as ``--label_cov_rows``.
    label_cov_rows: bool = False

    # Numerical
    #: skip the positive-definite check for Hubble diagrams larger than this.
    #: Also on the command line as ``--mxsize_test_posdef``.
    mxsize_test_posdef: int = 6000
    #: SNANA DOUBLE_PRECISION; precision for reference-distance calculations.
    #: Omit to use the active precision in Python; command-line runs default
    #: to single.
    double_precision: bool | None = None

    # Reference cosmology
    #: SNANA COSMOLOGY; the bias-correction simulation's cosmology, as a
    #: block of the keys tabulated under Reference cosmology below. The
    #: reference distance grid is built from it. It is needed only for
    #: outputs that do not record that cosmology, and must agree with one
    #: that does.
    cosmology: Cosmology | None = None
    #: SNANA RADIATION; include radiation in reference distances;
    #: False matches SNANA create_covariance.
    radiation: bool = True

    def __post_init__(self):
        supplied_mode = self.mode
        has_bins = self.rebin_bins is not None or self.nbin_x1 or self.nbin_c
        if has_bins or self.mode == 'rebin':
            if self.mode not in (None, 'rebin'):
                raise SNANAError("Rebin settings conflict with mode=" + repr(self.mode))
            resolve_rebin_bins(self.rebin_bins, self.nbin_x1, self.nbin_c)
            self.mode = 'rebin'
        elif self.subtract_vpec:
            if self.mode not in (None, 'unbinned'):
                raise SNANAError('subtract_vpec requires unbinned or rebin mode')
            self.mode = 'unbinned'
        elif self.mode is None:
            self.mode = 'binned'
        if supplied_mode is None:
            self._derived_settings['mode'] = (None, self.mode)

    @property
    def data_dir(self) -> Path:
        """Directory containing the FITOPT/MUOPT results (create_covariance.py:2808)."""
        return self.input_dir / self.version


def read_yaml(path) -> dict:
    """A YAML file's contents. An empty file aborts, where SNANA's read_yaml
    returns None and fails at its first use (:312)."""
    values = yaml.safe_load(Path(path).read_text())
    if values is None:
        raise SNANAError(f"{path} is empty", function_name="read_yaml")
    return values


def parse_covopt(text: str) -> Covopt:
    if isinstance(text, Mapping):
        return build_dataclass(Covopt, text)
    terms = text.split()
    if len(terms) != 2:
        raise SNANAError(
            f"COVOPT must be two bracketed terms, got {text!r}",
            function_name="parse_covopt",
        )
    try:
        label = _BRACKET.findall(terms[0])[0]
        inner = _BRACKET.findall(terms[1])[0]
    except IndexError:
        raise SNANAError(f"COVOPT has unbracketed term: {text!r}",
                         function_name="parse_covopt") from None

    parts = inner.split(",")
    if len(parts) not in (2, 3):
        raise SNANAError(
            f"COVOPT second term takes 2 or 3 comma-separated items, got {text!r}",
            function_name="parse_covopt",
        )
    scale = 1.0
    if len(parts) == 3:
        try:
            scale = float(parts[2])
        except ValueError:
            raise SNANAError(f"COVOPT scale is not a number: {text!r}",
                             function_name="parse_covopt") from None
    return Covopt(label=label.strip(), fitopt_filter=parts[0].strip(),
                  muopt_filter=parts[1].strip(), scale=scale)


def apply_filter(string: str, pattern: str) -> bool:
    """Match a FITOPT/MUOPT label against one covopt filter term (:1392).

    ``=X`` exact, ``+X`` contains, ``-X`` excludes, empty matches everything.
    """
    if isinstance(pattern, Mapping):
        unknown = set(pattern) - {'exact', 'include', 'exclude'}
        if unknown:
            raise SNANAError(f'Unknown systematic selector(s): {sorted(unknown)}')
        label = string.upper()
        return all((label == value.upper() if key == 'exact' else
                    value.upper() in label if key == 'include' else
                    value.upper() not in label) for key, value in pattern.items())
    string, pattern = string.upper(), pattern.upper()
    if pattern.startswith("="):
        return string == pattern[1:]
    if pattern.startswith("+"):
        return pattern[1:] in string
    if pattern.startswith("-"):
        return pattern[1:] not in string
    if pattern == "":
        return True
    raise SNANAError(f"cannot parse COVOPT matching pattern {pattern!r}",
                     function_name="apply_filter")


def load_sys_scales(path: Path) -> dict[str, float]:
    """label -> systematic scale, from a SYS_SCALE_FILE."""
    if not Path(path).is_file():
        raise SNANAError(f"SYS_SCALE_FILE not found: {path}",
                         function_name="load_sys_scales")
    return {str(k): float(v) for k, v in read_yaml(path).items()}


def _resolve(value) -> Path | None:
    """Expand $VARS and ~; relative paths stay relative to the cwd.

    SNANA expandvars these paths (prep_config, :2792): real configs lean on it,
    Pippin's libraries being written in terms of $PS1_ROOT and $DES_ROOT. It then
    uses them as given, so a relative path means relative to where you ran it.
    """
    return None if value is None else Path(os.path.expandvars(str(value))).expanduser()


def load_config(path: str | Path) -> CreateCovConfig:
    """Read a create_covariance input YAML."""
    return build_config(read_yaml(path))


def build_config(keys: Mapping | CreateCovConfig, overrides: Mapping | None = None) -> CreateCovConfig:
    """Translate file syntax, merge overrides, then validate and load resources."""
    sources = []
    if isinstance(keys, CreateCovConfig):
        keys = configuration_settings(keys)
    for source in (keys, overrides or {}):
        named = {key: value for key, value in source.items()
                 if isinstance(key, str) and key == key.lower()}
        snana = {key: value for key, value in source.items() if key not in named}
        sources.append(merge_settings(snana_settings(snana), named))
    values = merge_settings(*sources)
    bins = values.get('rebin_bins')
    if isinstance(bins, Mapping) and any(isinstance(count, bool) for count in bins.values()):
        raise SNANAError('Rebin counts must be positive integers, not booleans')
    cfg = build_dataclass(CreateCovConfig, values)
    if cfg.mode != 'unbinned' and (cfg.calibrators or cfg.extra_covs):
        raise SNANAError("CALIBRATORS and EXTRA_COVS require mode='unbinned'")
    if not 0 <= cfg.write_mask_cov <= 15:
        raise SNANAError("WRITE_MASK_COV must use only bits 1, 2, 4 and 8")
    if cfg.mxsize_test_posdef < 0:
        raise SNANAError("mxsize_test_posdef must be a non-negative integer")
    cfg.input_dir = _resolve(cfg.input_dir)
    cfg.outdir = _resolve(cfg.outdir)
    cfg.sys_scale_file = _resolve(cfg.sys_scale_file)
    if cfg.sys_scale_file is not None:
        original = cfg.sys_scales.copy()
        cfg.sys_scales = {**load_sys_scales(cfg.sys_scale_file), **cfg.sys_scales}
        cfg._derived_settings.update(derived_changes(
            {'sys_scales': original}, {'sys_scales': cfg.sys_scales}))
    return cfg


def snana_settings(keys: Mapping) -> dict:
    """Translate SNANA YAML syntax without resolving defaults or opening files."""
    raw = UsedTrackingDict(dict(keys))
    found = sorted(set(raw) & set(_REJECTED_KEYS))
    if found:
        raise SNANAError("\n".join(f"{key}: {_REJECTED_KEYS[key]}" for key in found),
                         function_name="build_config")
    values = {}
    for key in ('INPUT_DIR', 'VERSION', 'OUTDIR', 'CALIBRATORS', 'SUBTRACT_VPEC',
                'WRITE_MASK_COV', 'WRITE_FORMAT_COV', 'DOUBLE_PRECISION', 'REBIN_BINS',
                'RADIATION'):
        if key in raw:
            values[key.lower()] = raw[key]
    if values.get('calibrators') is None and 'calibrators' in values:
        values['calibrators'] = []
    if 'COVOPTS' in raw:
        values['covopts'] = [parse_covopt(text) for text in raw['COVOPTS'] or []]
    cosmo = raw.get('COSMOLOGY')
    if cosmo:
        names = {'OMEGA_MATTER': 'omega_matter', 'OMEGA_LAMBDA': 'omega_lambda',
                 'w0_LAMBDA': 'w0', 'wa_LAMBDA': 'wa', 'MUSHIFT': 'mushift'}
        values['cosmology'] = {names.get(key, key): parse_mushift(value) if key == 'MUSHIFT'
                               else value for key, value in cosmo.items()}
    scale_file = raw.get('SYS_SCALE_FILE') or raw.get('SYSFILE')
    if scale_file:
        values['sys_scale_file'] = scale_file
    if 'EXTRA_COVS' in raw:
        extra = []
        for entry in raw['EXTRA_COVS'] or []:
            words = str(entry).split()
            if len(words) != 3:
                raise SNANAError(f"EXTRA_COVS entry needs 'label weight file', got {entry!r}")
            extra.append((words[0], words[1], str(_resolve(words[2]))))
        values['extra_covs'] = extra
    if raw.get('UNBINNED') and 'rebin_bins' not in values:
        values['mode'] = 'unbinned'
    unread = sorted(key for key in raw if key not in raw.used)
    if unread:
        log.warn(logger, "CREATE_COV", f"unrecognised key(s) ignored: {', '.join(unread)}")
    return values
