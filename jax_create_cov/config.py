"""The create_covariance configuration dataclasses.

Field names are SNANA ``create_covariance.py`` config keys, lowercased, so a YAML
key and a config attribute are spelled the same. ``config_parser`` reads the
input YAML into one of these.
"""

from dataclasses import dataclass, field
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Literal

from jax_snlc_core.utils import SNANAError

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
