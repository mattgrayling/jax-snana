"""Driver: a grid of Hubble diagrams in, covariance matrices and one HD out."""

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from jax_snlc_core.io.fits_path import resolve_fits_path
from jax_snlc_core.io.fitres_reader import read_fitres_header
from jax_snlc_core.utils import SNANAError, snana_log as slog
from jax_snlc_core.utils.precision import require_precision

from jax_snlc_core.io.factorized_covariance import write_factorized_covariance

from . import covariance, hubble, inputs, output
from .config import Cosmology, Covopt, CreateCovConfig, resolve_rebin_bins
from .config_parser import parse_covopt

logger = logging.getLogger(__name__)

# always prepended, so index 000 is every systematic summed (:2670). The label must
# stay "ALL": get_muerr_sys matches it exactly to build the MUERR_SYS column.
COVOPT_ALL = "[ALL] [,]"
# --nosys selects the default MUOPT, retaining any FITOPT contributions.
COVOPT_NOSYS = "[=DEFAULT] [,=DEFAULT]"


@dataclass
class CovResult:
    """Everything a cosmology fit needs, without going through disk.

    ``covariances`` has one entry per COVOPT: ``kind="factorized"`` with
    ``diag``/``U`` arrays, or ``kind="systematics"`` with a dense ``matrix``.
    """

    hubble_diagram: pd.DataFrame
    covariances: dict[str, dict] = field(default_factory=dict)
    isdata_real: int = -1
    muerr_sys: np.ndarray | None = None
    rebin: tuple[int, dict[str, np.ndarray]] | None = None
    version_photometry: dict = field(default_factory=dict)
    mu_grid: tuple | None = None
    systematic_diagrams: dict = field(default_factory=dict)

    def covtot_inv(self, label: str = "ALL") -> np.ndarray:
        """Inverted total covariance for one covopt, as the fitters consume it."""
        if label not in self.covariances:
            raise SNANAError(f"no COVOPT labelled {label!r}", function_name="covtot_inv")
        entry = self.covariances[label]
        cov = entry["U"] @ entry["U"].T if entry["kind"] == "factorized" else entry["matrix"]
        covariance.require_finite_errors(self.hubble_diagram)
        return covariance.cov_invert(cov, self.hubble_diagram["MUERR"].to_numpy())


def build_covariance(diagrams: dict[str, pd.DataFrame],
                     fitopt_scales: dict[int, tuple[str, float]],
                     muopt_scales: dict[int, tuple[str, float]],
                     covopts: list[Covopt],
                     *,
                     cosmology: Cosmology | None = None,
                     radiation: bool = True,
                     mode: str = "binned",
                     rebin_bins: dict[str, int] | None = None,
                     nbin_x1: int = 0,
                     nbin_c: int = 0,
                     isdata_real: int = -1,
                     calibrators: list[str] | None = None,
                     extra_covs: list[tuple[str, float, str]] | None = None,
                     nosys: bool = False,
                     factorized: bool = False,
                     ) -> CovResult:
    """Aggregate an in-memory FITOPT x MUOPT grid into covariance matrices.

    ``diagrams`` maps ``FITOPT<i>_MUOPT<j>`` to a Hubble diagram carrying at least
    MU, MUREF and MUERR. With mode="rebin", ``rebin_bins`` maps FITRES columns
    to bin counts; {} groups only by IZBIN. The file-based entry point funnels
    into this. ``factorized=True`` preserves systematic vectors without
    constructing dense matrices; external covariance contributions fall back
    to the dense path for the COVOPTs that select them.
    """
    all_covopts = [parse_covopt(COVOPT_NOSYS if nosys else COVOPT_ALL)]
    all_covopts += list(covopts)
    labels = set()
    for option in all_covopts:
        if option.label in labels:
            raise SNANAError(
                f"duplicate COVOPT label {option.label!r}; labels must be unique, "
                f"including the automatic {all_covopts[0].label!r} option",
                function_name="build_covariance")
        labels.add(option.label)

    # order per get_hubble_diagrams (:751-773). MUERR_RENORM is swapped in for
    # unbinned but not rebin, where it enters through the rebin weights instead.
    rebin_info = None
    if mode != "rebin" and rebin_bins is not None:
        raise SNANAError("REBIN_BINS requires mode='rebin'",
                         function_name="build_covariance")
    if mode in ("unbinned", "rebin"):
        diagrams = hubble.common_set_of_sne(diagrams)
    if mode == "rebin":
        bins = resolve_rebin_bins(rebin_bins, nbin_x1, nbin_c)
        z_grid, mu_grid = hubble.reference_grid(cosmology, radiation=radiation)
        reference = diagrams[inputs.join_name(inputs.REF, inputs.REF)]
        index, coordinates, edges = hubble.rebin_indices(reference, bins)
        nbin_z = len(np.unique(coordinates["IZBIN"]))
        rebin_info = (nbin_z, edges)
        # Preserve SNANA row names and its truncated loop for legacy flags.
        # Named bins retain all occupied redshift bins, including gaps.
        legacy = rebin_bins is None
        limit = nbin_z * nbin_x1 * nbin_c if legacy else None
        diagrams = {label: hubble.rebin_hubble_diagram(
                        df, index, coordinates, z_grid, mu_grid,
                        labels={"x1": "x"} if legacy else None, limit=limit)
                    for label, df in diagrams.items()}
    elif mode == "unbinned":
        hubble.use_muerr_renorm(diagrams)
    diagrams, base = hubble.remove_nans(diagrams)

    contribs = covariance.contributions(diagrams, fitopt_scales, muopt_scales,
                                        extra_covs or (), base)

    covariances = {}
    for c in all_covopts:
        value = covariance.covsys_from_covopt(c, contribs, base, calibrators,
                                              factorized=factorized)
        covariances[c.label] = ({"kind": "factorized", **value} if isinstance(value, dict)
                               else {"kind": "systematics", "matrix": value})
    all_covariance = covariances.get("ALL")
    muerr_sys = None if all_covariance is None else (
        np.sqrt(np.sum(all_covariance["U"] ** 2, axis=1))
        if all_covariance["kind"] == "factorized"
        else covariance.muerr_sys(all_covariance["matrix"]))

    grid = (hubble.reference_grid(cosmology, radiation=radiation)
            if isdata_real == 0 else None)

    return CovResult(hubble_diagram=base, covariances=covariances,
                     isdata_real=isdata_real, muerr_sys=muerr_sys,
                     rebin=rebin_info, systematic_diagrams=dict(diagrams),
                     mu_grid=grid)


def select_covariance_grid(grid, muopt):
    """Select a MUOPT and require the reference cells used by the covariance."""
    if muopt >= 0:
        if muopt not in {inputs.split_name(name)[1] for name in grid}:
            raise SNANAError(f"MUOPT{muopt:03d} is not in the covariance grid")
        grid = {name: value for name, value in grid.items()
                if inputs.split_name(name)[1] == muopt}
    reference_muopt = max(inputs.REF, muopt)
    required = {inputs.join_name(inputs.REF, reference_muopt)} | {
        inputs.join_name(inputs.split_name(name)[0], reference_muopt) for name in grid}
    missing = required - grid.keys()
    if missing:
        raise SNANAError(f"Covariance grid is missing reference cells {sorted(missing)}")
    return grid


def prepare_covariance(config, diagrams, fitopt_scales, muopt_scales, *, isdata_real=-1):
    """Apply configuration to a loaded grid, for file and pipeline callers."""
    require_precision(config.double_precision)
    diagrams = {name: frame.copy() for name, frame in
                select_covariance_grid(diagrams, config.muopt).items()}
    if config.muopt >= 0:
        diagrams = {inputs.join_name(inputs.split_name(name)[0], inputs.REF): frame
                    for name, frame in diagrams.items()}
        muopt_scales = {inputs.REF: ("DEFAULT", 1.0)}
    if config.subtract_vpec:
        hubble.subtract_vpec(diagrams, renormalised=config.mode == "unbinned")
    return build_covariance(
        diagrams, fitopt_scales, muopt_scales, config.covopts,
        cosmology=config.cosmology, radiation=config.radiation, mode=config.mode,
        rebin_bins=config.rebin_bins, nbin_x1=config.nbin_x1, nbin_c=config.nbin_c,
        isdata_real=isdata_real, calibrators=config.calibrators,
        extra_covs=config.extra_covs, nosys=config.nosys,
        factorized=bool(config.write_mask_cov & 8))


def create_covariance(config: CreateCovConfig, *, write: bool = True) -> CovResult:
    """Read a submit_batch BBC output tree and aggregate it."""
    files = inputs.grid_files(config)
    reference_name = inputs.join_name(inputs.REF, max(inputs.REF, config.muopt))
    selected_files = select_covariance_grid(files, config.muopt)
    header = read_fitres_header(str(selected_files[reference_name]))
    config = replace(config, cosmology=inputs.reference_cosmology(config.cosmology, header))

    submit_info = inputs.read_submit_info(config.input_dir)
    fitopt_scales = inputs.sys_scales("FITOPT", submit_info, config.sys_scales)
    muopt_scales = inputs.sys_scales("MUOPT", submit_info, config.sys_scales)

    diagrams = {name: hubble.load_hubble_diagram(path, config.mode)
                for name, path in selected_files.items()}
    result = prepare_covariance(config, diagrams, fitopt_scales, muopt_scales,
                                isdata_real=inputs.isdata_real(header))
    reference = resolve_fits_path(config.data_dir / f"{reference_name}.FITRES")
    if Path(reference).is_file():
        result.version_photometry = inputs.version_photometry(
            read_fitres_header(str(reference)))
    elif write and config.outdir is not None:
        raise SNANAError(f"{config.data_dir} holds no nominal FITRES "
                         f"({reference_name}.FITRES or .gz)",
                         function_name="create_covariance")
    if write and config.outdir is not None:
        write_outputs(config.outdir, result, config)
    return result


def write_outputs(outdir: Path, result: CovResult, config: CreateCovConfig) -> None:
    """Write the covariance products the write mask selects, the HD, and INFO.YML."""
    outdir = Path(outdir)
    mask, fmt = config.write_mask_cov, config.write_format_cov
    unbinned = config.mode == "unbinned"

    writer = output.write_hd_unbinned if unbinned else output.write_hd_binned
    # the reference cell keeps the plain name; only the others are labelled (:1657)
    base_name = inputs.join_name(inputs.REF, inputs.REF)
    diagrams = {output.HD_FILENAME: result.hubble_diagram}
    if config.systematic_hd:
        diagrams = {(output.HD_FILENAME if label == base_name
                     else f"{label}_{output.HD_FILENAME}"): hd
                    for label, hd in result.systematic_diagrams.items()}
    for name, hd in diagrams.items():
        musim = (None if result.mu_grid is None
                 else np.interp(hd["zHD"].to_numpy(), *result.mu_grid))
        rest = (hd, result.isdata_real, result.muerr_sys, musim)
        writer(outdir / name, *(rest if unbinned else rest + (result.rebin,)))

    covariance.require_finite_errors(result.hubble_diagram)
    muerr = result.hubble_diagram["MUERR"].to_numpy()
    rows = None
    if config.label_cov_rows:
        fmt = "text"
        hd = result.hubble_diagram
        if "CID" in hd.columns:
            rows = (hd["CID"].astype(str).to_numpy(),
                    hd["IDSURVEY"].to_numpy(dtype=int),
                    hd["zHD"].to_numpy(dtype=float))

    def write(path, cov):
        if fmt == "npz":
            output.write_covariance_npz(path, cov)
        else:
            output.write_covariance_text(path, cov, rows)
    covsys_names = {}
    covtot_inv_names = {}
    factorized_names = {}
    for i, (label, entry) in enumerate(result.covariances.items()):
        covsys_names[i] = covtot_inv_names[i] = factorized_names[i] = None
        compact = entry["kind"] == "factorized"
        if mask & output.WRITE_MASK_FACTORIZED:
            if compact:
                factorized_names[i] = f"covtot_factorized_{i:03d}.npz"
                write_factorized_covariance(outdir / factorized_names[i], entry)
            else:
                logger.warning("COVOPT %s has no vector factors; writing a dense inverse fallback", label)
        if compact and not mask & 7:
            continue  # Compact-only output needs no dense reconstruction.
        cov = entry["U"] @ entry["U"].T if compact else entry["matrix"]
        if mask & output.WRITE_MASK_COVSYS:
            covsys_names[i] = output.cov_filename(i, output.PREFIX_COVSYS, fmt)
            write(outdir / covsys_names[i], cov)
        if mask & output.WRITE_MASK_COVTOT_INV or (mask & 8 and not compact):
            covtot_inv_names[i] = output.cov_filename(i, output.PREFIX_COVTOT_INV, fmt)
            write(outdir / covtot_inv_names[i],
                  covariance.cov_invert(cov, muerr,
                                        config.mxsize_test_posdef, label))
        if mask & output.WRITE_MASK_COVTOT:
            write(outdir / output.cov_filename(i, output.PREFIX_COVTOT, fmt),
                  cov + np.diag(muerr ** 2))

    nbin_z = len(result.hubble_diagram) if config.mode != "rebin" else (
        result.rebin[0] if result.rebin else 0)
    submit_info = None if config.input_dir is None else config.input_dir / "SUBMIT.INFO"
    output.write_info_yml(
        outdir / output.INFO_YML_FILENAME,
        covsys_labels=list(result.covariances),
        size_hd=len(result.hubble_diagram), isdata_real=result.isdata_real,
        bin_method=output.bin_method(config.mode, nbin_z,
                                     result.rebin[1] if result.rebin else {}),
        cospar=config.cosmology, version_photometry=result.version_photometry,
        bbc_dir=None if config.input_dir is None else config.data_dir,
        bbc_info_file=submit_info if submit_info is not None and submit_info.is_file() else None,
        sys_scale_file=config.sys_scale_file,
        covsys_names=covsys_names, covtot_inv_names=covtot_inv_names,
        factorized_names=factorized_names if mask & 8 else None)
