"""Writing the standard create_covariance outputs.

Formats follow SNANA byte-for-byte: these files are read by cosmology fitters that
key off the VARNAMES line, and by SNANA's own tooling.
"""

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from jax_snlc_core.io.snana_table import SNANA_VERSION_STRING

from jax_snlc_core.io import cov_matrix

HD_FILENAME = "hubble_diagram.txt"
PREFIX_COVSYS = "covsys"
PREFIX_COVTOT_INV = "covtot_inv"
PREFIX_COVTOT = "covtot"
INFO_YML_FILENAME = "INFO.YML"

WRITE_MASK_COVSYS = 1
WRITE_MASK_COVTOT_INV = 2
WRITE_MASK_COVTOT = 4
WRITE_MASK_FACTORIZED = 8


def cov_filename(index: int, prefix: str, fmt: str) -> str:
    suffix = {"npz": "npz", "text": "txt.gz"}[fmt]
    return f"{prefix}_{index:03d}.{suffix}"


def write_covariance_npz(path: Path, cov: np.ndarray) -> None:
    """Upper triangle as float32, plus the row count (:2219)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cov_matrix.write_covariance_npz(path, cov)


def write_covariance_text(path: Path, cov: np.ndarray, rows=None) -> None:
    """Row count then the full flattened matrix, one value per line (:2091).

    ``rows`` adds SNANA's per-element diagnostic label (:2272); it changes no
    number, only readability.
    """
    cov = np.asarray(cov)
    path.parent.mkdir(parents=True, exist_ok=True)
    nrow = len(cov)
    if rows is None:
        values = (f"{c:13.6e}\n" for c in cov.flatten())
    else:
        values = (f"{c:13.6e} {_cov_label(i, nrow, rows)}\n"
                  for i, c in enumerate(cov.flatten()))
    body = f"{nrow}\n" + "".join(values)
    if path.suffix == ".gz":
        path.write_bytes(gzip.compress(body.encode(), compresslevel=6))
    else:
        path.write_text(body)


def _cov_label(index: int, nrow: int, rows) -> str:
    """SNANA's diagnostic label for one flattened covariance element (:2272)."""
    cid, idsurvey, zhd = rows
    col, row = index // nrow, index % nrow
    # both brackets carry the row's zHD, as SNANA writes it (:2290)
    label = (f"# CID,IDSURVEY,zHD,index = "
             f"[ {cid[row]:>8}, {idsurvey[row]:3d}, {zhd[row]:.5f}, {row} ] x "
             f"[ {cid[col]:>8}, {idsurvey[col]:3d}, {zhd[row]:.5f}, {col} ]")
    return label + "  diag" if row == col else label


def _comments(unbinned: bool, isdata_real: int, syserr: bool,
              pbeams: bool, musim: bool, rebin=None) -> str:
    lines = ["# zHD       = redshift in CMB frame with VPEC correction"]
    lines.append("# zHEL      = "
                 + ("helio redshift (beware: no VPEC corr) " if unbinned
                    else "zHD "))
    lines.append("# MU        = distance modulus corrected for bias and contamination")
    lines.append("# MUERR     = stat-uncertainty on MU ")
    if syserr:
        lines.append("# MUERR_SYS = sqrt(COVSYS_DIAG) for 'ALL' sys (diagnostic)")
    if pbeams:
        lines.append("# PROB1A_BEAMS = SNIa BEAMS probability from BBC (diagnostic)")
    if musim:
        lines.append("# MUSIM     = mu computed with biasCor cosmology (diagnostic)")
    lines.append(f"# ISDATA_REAL: {isdata_real}   "
                 "# flag for cosmology fitter to choose blind option")
    if rebin is not None:
        nbin_z, edges = rebin
        lines.append("#")
        names = "/".join(("z", *edges))
        counts = " / ".join(str(n) for n in (nbin_z, *(len(e) - 1 for e in edges.values())))
        lines.append(f"# nrebin({names}) = {counts} ")
        lines.append("# rebin(z):  see IZBIN in BBC-output FITRES file")
        for name, values in edges.items():
            lines.append(f"# rebin({name}): "
                         + " ".join(f"{v:.4f}" for v in values) + " ")
    lines.append("#")
    return "\n".join(lines) + "\n"


def write_hd_binned(path: Path, base: pd.DataFrame, isdata_real: int,
                    muerr_sys: np.ndarray | None = None,
                    musim: np.ndarray | None = None, rebin=None) -> None:
    """Binned Hubble diagram (:1873). zHEL is written as a copy of zHD."""
    nevt = "NEVT" in base.columns
    varlist = "ROW zHD zHEL MU MUERR"
    if nevt:
        varlist += "  NEVT"
    if muerr_sys is not None:
        varlist += " MUERR_SYS"
    if musim is not None:
        varlist += " MUSIM"

    # columns, not iterrows: a row Series upcasts to one dtype, which would turn
    # the integer ROW of an M0DIF into 1.0
    names = base["ROW"].to_numpy()
    zhd = base["zHD"].to_numpy(dtype=float)
    mu = base["MU"].to_numpy(dtype=float)
    muerr = base["MUERR"].to_numpy(dtype=float)
    counts = base["NEVT"].to_numpy(dtype=int) if nevt else None

    rows = []
    for i in range(len(base)):
        values = (f"{names[i]:<6}  {zhd[i]:6.5f} {zhd[i]:6.5f} "
                  f"{mu[i]:8.5f} {muerr[i]:8.5f} ")
        if nevt:
            values += f" {counts[i]:4d} "
        if muerr_sys is not None:
            values += f" {muerr_sys[i]:8.5f}"
        if musim is not None:
            values += f" {musim[i]:7.4f}"
        rows.append(f"ROW: {values}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_comments(False, isdata_real, muerr_sys is not None,
                              False, musim is not None, rebin)
                    + f"VARNAMES: {varlist}\n" + "\n".join(rows) + "\n")


def write_hd_unbinned(path: Path, base: pd.DataFrame, isdata_real: int,
                      muerr_sys: np.ndarray | None = None,
                      musim: np.ndarray | None = None) -> None:
    """Unbinned Hubble diagram, one row per SN (:1937)."""
    vpec = "MUERR_VPEC" in base.columns
    pbeams = "PROBCC_BEAMS" in base.columns
    varlist = "CID IDSURVEY zHD zHEL MU MUERR"
    if vpec:
        varlist += " MUERR_VPEC"
    if muerr_sys is not None:
        varlist += " MUERR_SYS"
    if pbeams:
        # the column is PROBIA_BEAMS; only SNANA's comment line above
        # spells it with a digit (:260 vs :2044)
        varlist += " PROBIA_BEAMS"
    if musim is not None:
        varlist += " MUSIM"

    cid = base["CID"].astype(str).to_numpy()
    idsurvey = base["IDSURVEY"].to_numpy(dtype=int)
    zhd = base["zHD"].to_numpy(dtype=float)
    zhel = base["zHEL"].to_numpy(dtype=float)
    mu = base["MU"].to_numpy(dtype=float)
    muerr = base["MUERR"].to_numpy(dtype=float)
    muerr_vpec = base["MUERR_VPEC"].to_numpy(dtype=float) if vpec else None
    probcc = base["PROBCC_BEAMS"].to_numpy(dtype=float) if pbeams else None

    rows = []
    for i in range(len(base)):
        values = (f"{cid[i]:<10} {idsurvey[i]:3d} "
                  f"{zhd[i]:6.5f} {zhel[i]:6.5f} "
                  f"{mu[i]:8.5f}  {muerr[i]:8.5f}")
        if vpec:
            values += f" {muerr_vpec[i]:8.5f}"
        if muerr_sys is not None:
            values += f" {muerr_sys[i]:8.5f}"
        if pbeams:
            # the column holds the CC probability; the file reports the Ia one
            values += f" {1.0 - probcc[i]:8.5f}"
        if musim is not None:
            values += f" {musim[i]:7.4f}"
        rows.append(f"SN: {values}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_comments(True, isdata_real, muerr_sys is not None,
                              pbeams, musim is not None)
                    + f"VARNAMES: {varlist}\n" + "\n".join(rows) + "\n")


def write_info_yml(path: Path, *, covsys_labels, size_hd: int, isdata_real: int,
                   bin_method: tuple[str, str], cospar, version_photometry: dict,
                   bbc_dir, bbc_info_file, sys_scale_file,
                   covsys_names, covtot_inv_names, factorized_names=None) -> None:
    """The summary a cosmology fitter reads to discover what it has (:2297).

    Two key-sorted YAML blocks then two appended lines, matching SNANA's own
    write order. COSPAR_BIASCOR is written empty when the biasCor cosmology is
    unknown, so a consumer can tell "unknown" from "absent".
    """
    covopts = {i: f"{label:<20} {covsys_names[i]}   {covtot_inv_names[i]}"
               for i, label in enumerate(covsys_labels)}
    if factorized_names is not None:
        covopts = {i: f"{entry}   {factorized_names[i]}" for i, entry in covopts.items()}
    info = {"HD": HD_FILENAME, "ISDATA_REAL": isdata_real, "SIZE_HD": size_hd,
            "COVOPTS": covopts, "SNANA_VERSION": SNANA_VERSION_STRING,
            **version_photometry}
    info2 = {"BBC_DIR": str(bbc_dir) if bbc_dir is not None else None,
             "BBC_INFO_FILE": str(bbc_info_file) if bbc_info_file else None,
             "SYS_SCALE_FILE": str(sys_scale_file) if sys_scale_file else None}

    method, comment = bin_method
    cospar_block = {} if cospar is None else {
        "OMEGA_MATTER": cospar.omega_matter, "OMEGA_LAMBDA": cospar.omega_lambda,
        "w0_LAMBDA": cospar.w0, "wa_LAMBDA": cospar.wa,
        # as the simulation's input gave it: one value, or lo:hi
        "MUSHIFT": (cospar.mushift[0] if cospar.mushift[0] == cospar.mushift[1]
                    else f"{cospar.mushift[0]}:{cospar.mushift[1]}"),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# Dictionary arguments for COVOPTS:\n"
                "# index:  <sysLabel>  <name of covsys file>   "
                "<name of covtot_inv file> [name of factorized file]\n"
                "#  (file name = None -> not written)\n#\n")
        yaml.safe_dump(info, f)
        f.write("\n")
        yaml.safe_dump(info2, f)
        f.write(f"\nHD_BIN_METHOD: {method}   # {comment}\n")
        yaml.safe_dump({"COSPAR_BIASCOR": cospar_block}, f)


def bin_method(mode: str, nbin_z: int, edges) -> tuple[str, str]:
    """BINNED / REBIN / UNBIN plus its comment (:2378)."""
    if mode == "unbinned":
        return "UNBIN", "no binning"
    if mode == "rebin":
        counts = [f"{nbin_z}(z)"] + [f"{len(values) - 1}({name})"
                                     for name, values in reversed(edges.items())]
        return "REBIN", " x ".join(counts) + " "
    return "BINNED", f"{nbin_z} redshift bins"
