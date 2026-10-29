"""Prepare Hubble diagrams, calculate systematic covariances and write their products."""

import gzip
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml

from jax_snlc_core import logging as log
from jax_snlc_core.cosmology import distance_modulus
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import (
    SIM_COSMOLOGY_KEYS, SNANA_VERSION_STRING, fitopt_muopt_indices, fitopt_muopt_name,
    open_text, parse_mushift, read_fitres, read_fitres_header, resolve_fits_path,
    write_covariance_npz, write_factorized_covariance)
from jax_snlc_core.utils import LIGHT_KM, require_precision

from .config import (Covopt, apply_filter, Cosmology, CreateCovConfig, read_yaml, build_config,
                     parse_covopt, resolve_rebin_bins)

logger = logging.getLogger(__name__)

HD_FILENAME = "hubble_diagram.txt"
PREFIX_COVSYS = "covsys"
PREFIX_COVTOT_INV = "covtot_inv"
PREFIX_COVTOT = "covtot"
INFO_YML_FILENAME = "INFO.YML"

WRITE_MASK_COVSYS = 1
WRITE_MASK_COVTOT_INV = 2
WRITE_MASK_COVTOT = 4
WRITE_MASK_FACTORIZED = 8


def mu_difference(df1: pd.DataFrame, df2: pd.DataFrame, scale: float) -> np.ndarray:
    """Scaled distance-modulus difference between two variants (:1168).

    Subtracting MUREF from each side as well as MU is what makes this work for
    systematics that move redshifts (zshift, vpec, every zPHOT variant): the
    distance is shifted onto the reference diagram's redshift first. MUREF is an
    approximation, since the fitted cosmology is not yet known.
    """
    if len(df1) != len(df2):
        raise SNANAError(
            f"variant has {len(df1)} rows against the reference's {len(df2)}; "
            "the grid is not aligned",
            function_name="mu_difference",
        )
    diff = scale * ((df1["MU"] - df1["MUREF"])
                    - (df2["MU"] - df2["MUREF"])).to_numpy()
    return np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class Contribution:
    """One systematic's contribution, held either as the mu difference it came
    from or, for an externally supplied matrix, as the covariance itself."""

    fitopt_label: str
    muopt_label: str
    diff: np.ndarray | None = None
    cov: np.ndarray | None = None

    def matrix(self) -> np.ndarray:
        return self.cov if self.cov is not None else np.outer(self.diff, self.diff)


def covsys_from_covfile(base: pd.DataFrame, path: str) -> np.ndarray:
    """Read an externally supplied covariance, e.g. a vpec systematic (:1223).

    A text file holds ``CID1 IDSURVEY1 CID2 IDSURVEY2 MU_COV`` rows, comma-
    or space-separated, each placed where it names. An ``.npz`` file holds
    arrays ``CID``, ``IDSURVEY`` and ``MU_COV``, the last the upper triangle
    of the covariance over those events, which is read as the whole symmetric
    matrix; SNANA places only one triangle of it
    (get_covsy_from_npzcovfile, create_covariance.py:1243). Neither form
    carries a FIELD, so events are matched on CID_IDSURVEY only, coarser than
    the CID_IDSURVEY_FIELD the diagrams are indexed by. Events outside the
    diagram are skipped.
    """
    if "CID" not in base.columns:
        raise SNANAError(
            "EXTRA_COVS needs an unbinned Hubble diagram: a binned one has no "
            "CID to match the covariance file against",
            function_name="covsys_from_covfile",
        )
    keys = (base["CID"].astype(str) + "_"
            + base["IDSURVEY"].astype(str)).tolist()
    position = {}
    for i, key in enumerate(keys):
        position.setdefault(key, i)   # first match wins, as list.index does

    if path.endswith(".npz"):
        return _covsys_from_npz(path, position, len(keys))

    with open_text(path) as stream:
        text = [line for line in stream
                if line.strip() and not line.lstrip().startswith("#")]
    if not text:
        raise SNANAError(f"empty EXTRA_COVS file: {path}",
                         function_name="covsys_from_covfile")
    separator = "," if "," in text[0] else None

    header = [w.strip() for w in text[0].split(separator)]
    required = ["CID1", "IDSURVEY1", "CID2", "IDSURVEY2", "MU_COV"]
    missing = [c for c in required if c not in header]
    if missing:
        raise SNANAError(f"{path} has no {', '.join(missing)} column",
                         function_name="covsys_from_covfile")
    column = {name: header.index(name) for name in required}

    cov = np.zeros((len(keys), len(keys)))
    for line in text[1:]:
        words = [w.strip() for w in line.split(separator)]
        one = position.get(f"{words[column['CID1']]}_{words[column['IDSURVEY1']]}")
        two = position.get(f"{words[column['CID2']]}_{words[column['IDSURVEY2']]}")
        if one is None or two is None:
            continue
        cov[one, two] = float(words[column["MU_COV"]])
    return cov


def _covsys_from_npz(path: str, position: dict[str, int], n: int) -> np.ndarray:
    """The ``.npz`` form of an EXTRA_COVS file, placed in diagram order."""
    with np.load(path) as npz:
        missing = {"CID", "IDSURVEY", "MU_COV"} - set(npz.files)
        if missing:
            raise SNANAError(f"{path} has no {', '.join(sorted(missing))} array",
                             function_name="covsys_from_covfile")
        names = [f"{c}_{s}" for c, s in zip(npz["CID"].astype(str),
                                            npz["IDSURVEY"].astype(str))]
        full = np.zeros((len(names), len(names)))
        full[np.triu_indices(len(names))] = npz["MU_COV"]
    full = full + full.T - np.diag(np.diag(full))
    rows = np.array([position.get(name, -1) for name in names])
    keep = rows >= 0
    cov = np.zeros((n, n))
    cov[np.ix_(rows[keep], rows[keep])] = full[np.ix_(keep, keep)]
    return cov


def contributions(diagrams: dict[str, pd.DataFrame],
                  fitopt_scales: dict[int, tuple[str, float]],
                  muopt_scales: dict[int, tuple[str, float]],
                  extra_covs: Sequence[tuple[str, float, str]] = (),
                  base: pd.DataFrame | None = None,
                  ) -> list[Contribution]:
    """Keep the last contribution for each FITOPT|MUOPT label, as SNANA does.

    The two axes reference differently: a MUOPT variant is differenced against
    MUOPT000 *of its own FITOPT*, while a FITOPT variant is differenced against
    FITOPT000/MUOPT000. Their scales multiply.
    """
    result = {}
    for name, df in diagrams.items():
        fitopt, muopt = fitopt_muopt_indices(name)
        if fitopt not in fitopt_scales:
            raise SNANAError(f"{name}: FITOPT{fitopt:03d} is not in SUBMIT.INFO",
                             function_name="contributions")
        if muopt not in muopt_scales:
            raise SNANAError(f"{name}: MUOPT{muopt:03d} is not in SUBMIT.INFO",
                             function_name="contributions")
        fitopt_label, fitopt_scale = fitopt_scales[fitopt]
        muopt_label, muopt_scale = muopt_scales[muopt]

        if fitopt == 0 and muopt == 0:
            diff = np.zeros(len(df))
        elif muopt != 0:
            other = diagrams[fitopt_muopt_name((fitopt, 0))]
            diff = mu_difference(df, other, fitopt_scale * muopt_scale)
        else:
            other = diagrams[fitopt_muopt_name((0, 0))]
            diff = mu_difference(df, other, fitopt_scale)

        result[f"{fitopt_label}|{muopt_label}"] = Contribution(
            fitopt_label, muopt_label, diff=diff)

    for label, weight, path in extra_covs:
        # SNANA stores the weight and never applies it (:1453)
        result[f"DEFAULT|{label}"] = Contribution(
            "DEFAULT", label, cov=covsys_from_covfile(base, path) * weight ** 2)
    return list(result.values())


def covsys_from_covopt(covopt: Covopt, contribs: list[Contribution],
                       base: pd.DataFrame,
                       calibrators: list[str] | None = None, *,
                       factorized: bool = False) -> np.ndarray | dict:
    """Sum the contributions this covopt selects into one covariance (:1407)."""
    calib_mask = None
    if calibrators:
        cid = base.reset_index()["CID"] if "CID" in base.columns else None
        if cid is None:
            raise SNANAError("CALIBRATORS need an unbinned Hubble diagram",
                             function_name="covsys_from_covopt")
        calib_mask = cid.isin(calibrators).to_numpy()
        if not calib_mask.any():
            # CIDs compared as strings, as SNANA
            log.warn(logger, "CREATE_COV",
                     f"none of the {len(calibrators)} CALIBRATORS match a CID; "
                     "no redshift-systematic masking will be applied")

    # External covariance contributions keep their existing dense treatment.
    # Only matched contributions determine whether this COVOPT needs fallback.
    selected = [c for c in contribs
                if apply_filter(c.fitopt_label, covopt.fitopt_filter)
                and apply_filter(c.muopt_label, covopt.muopt_filter)]
    compact = factorized and all(c.cov is None for c in selected)
    vectors, labels = [], []
    total = None
    has_systematic = False
    for contribution in selected:
        fitopt_label, muopt_label = contribution.fitopt_label, contribution.muopt_label

        has_systematic |= (fitopt_label, muopt_label) != ("DEFAULT", "DEFAULT")
        if compact:
            vector = contribution.diff.copy() * covopt.scale
            if calib_mask is not None and _is_redshift_systematic(fitopt_label, muopt_label):
                vector[calib_mask] = 0.
            if np.any(vector):
                vectors.append(vector)
                labels.append(f"{fitopt_label}|{muopt_label}")
            continue
        cov = contribution.matrix()
        # Cepheid calibrators carry no redshift systematic, so a VPEC or ZSHIFT
        # term must not touch their rows or columns (:1493)
        if calib_mask is not None and _is_redshift_systematic(fitopt_label,
                                                              muopt_label):
            cov = cov.copy()
            cov[calib_mask, :] = 0.0
            cov[:, calib_mask] = 0.0

        # .scale is the error scale, so the covariance takes its square (:1436)
        scaled = cov * covopt.scale ** 2
        total = scaled if total is None else total + scaled

    if compact and selected:
        U = np.column_stack(vectors) if vectors else np.zeros((len(base), 0))
        return {"diag": base["MUERR"].to_numpy() ** 2, "U": U, "vec_labels": tuple(labels)}
    if total is None:
        raise SNANAError(
            f"COVOPT {covopt.label}: no systematic matches filters "
            f"'{covopt.fitopt_filter}' / '{covopt.muopt_filter}'",
            function_name="covsys_from_covopt",
        )
    if has_systematic and covopt.scale != 0 and not np.any(total):
        log.warn(logger, "CREATE_COV",
                 f"every element of covsys({covopt.label}) is zero")
    return total


def _is_redshift_systematic(fitopt_label: str, muopt_label: str) -> bool:
    return any(apply_filter(label, pattern)
               for label in (fitopt_label, muopt_label)
               for pattern in ("+VPEC", "+ZSHIFT"))


def require_finite_errors(diagram: pd.DataFrame) -> None:
    """A distance with no finite error cannot enter the statistical diagonal.
    BBC writes MUDIFERR 999 for a bin with no events (MUDIFERR_EMPTY,
    SALT2mu.c:501), read here as NaN, and a bin too thin to fit leaves NaN
    itself; either way the diagram, not the inversion, is what to fix."""
    bad = ~np.isfinite(diagram["MUERR"].to_numpy(dtype=float))
    if not bad.any():
        return
    rows = diagram[bad]
    if "NFIT" in rows.columns:
        where = ", ".join(f"zHD={z:.4f} (NFIT={int(n)})"
                          for z, n in zip(rows["zHD"], rows["NFIT"]))
    else:
        where = ", ".join(f"zHD={z:.4f}" for z in rows["zHD"])
    raise SNANAError(
        f"{int(bad.sum())} of {len(diagram)} Hubble-diagram rows have no "
        f"finite MUERR: {where}. Widen the BBC redshift bins or lower the "
        f"redshift range so every bin holds enough events to fit.",
        function_name="require_finite_errors")


def cov_invert(cov_sys: np.ndarray, muerr_stat: np.ndarray,
               mxsize_test_posdef: int = 6000, label: str = "") -> np.ndarray:
    """Invert the total covariance: systematic plus the statistical diagonal (:1522).

    The cosmology fitter consumes this rather than covsys, so the statistical term
    is added here rather than there.
    """
    covtot = cov_sys + np.diag(np.asarray(muerr_stat, dtype=float) ** 2)
    try:
        covtot_inv = np.linalg.inv(covtot)
    except np.linalg.LinAlgError as exc:
        raise SNANAError(f"total covariance is singular: {exc}",
                         function_name="cov_invert") from exc

    # SNANA reports, but does not abort on, an inverse that is not unitary
    # (:1541). The product is rounded first, as it does.
    product = np.round(covtot @ covtot_inv, 2)
    residual = np.linalg.norm(np.eye(len(product)) - product
                              @ product.conjugate().T)
    if not residual < np.finfo(product.dtype).eps * 10.0 * len(product):
        log.warn(logger, "CREATE_COV",
                 f"{label or 'covtot'} inverse is not unitary "
                 f"(residual {residual:.3e})")

    # skipped above mxsize_test_posdef, as SNANA (:1531)
    if len(covtot) <= mxsize_test_posdef:
        if not np.all(np.linalg.eigvals(covtot_inv) > 0):
            log.warn(logger, "CREATE_COV",
                     f"{label or 'covtot'} inverse is not positive definite")
        condition = np.linalg.cond(covtot_inv)
        if condition >= 1.0 / np.finfo(float).eps:
            raise SNANAError(
                f"inverted total covariance is ill-conditioned "
                f"(cond={condition:.3e})",
                function_name="cov_invert",
            )
    return covtot_inv


def muerr_sys(covsys: np.ndarray) -> np.ndarray:
    """Per-entry systematic error, sqrt of the covariance diagonal (:1744).

    Not clipped: a negative diagonal can only come from an EXTRA_COVS file, and
    SNANA lets the NaN through to the MUERR_SYS column rather than hide it.
    """
    with np.errstate(invalid="ignore"):
        return np.sqrt(np.diagonal(covsys))


_SKIP = ("~", "wfit_", "cospar")


_LABEL_COLUMN = {"FITOPT": 2, "MUOPT": 1}


_ERR_SENTINEL = 999.0


def load_hubble_diagram(path: Path, mode: str) -> pd.DataFrame:
    """Read one M0DIF (binned) or FITRES (unbinned/rebin) grid file (:592)."""
    return hubble_diagram_from(read_fitres(str(path)).df.copy(), mode,
                               source=path)


def hubble_diagram_from(df: pd.DataFrame, mode: str,
                        source="<memory>") -> pd.DataFrame:
    """One grid's Hubble diagram, from a frame rather than a file.

    The reading half is :func:`load_hubble_diagram`; a caller holding what BBC
    produced (:func:`jax_salt2mu.output.m0dif_columns`) goes straight here.
    ``source`` names the frame in the error a missing column raises.
    """
    path = source

    if "zHD" not in df.columns and "z" in df.columns:
        # really old M0DIF files carry z instead of zHD (:666)
        df = df.rename(columns={"z": "zHD"})

    for column in ("MUERR", "MUDIFERR"):
        if column in df.columns:
            df.loc[df[column] == _ERR_SENTINEL, column] = np.nan

    # an M0DIF carries the offset from the reference, not the distance itself
    if "MU" not in df.columns:
        _require(df, path, "MUREF", "MUDIF")
        df["MU"] = df["MUREF"] + df["MUDIF"]
    if "MUERR" not in df.columns:
        _require(df, path, "MUDIFERR")
        df["MUERR"] = df["MUDIFERR"]

    if mode == "binned":
        _require(df, path, "zHD")
        return df.sort_values("zHD").reset_index(drop=True)

    _require(df, path, "CID", "IDSURVEY", "FIELD", "zHD", "zHEL", "MUMODEL")
    df["CID"] = df["CID"].astype(str)
    df = df.sort_values(["zHD", "CID"])
    df = df.rename(columns={"MUMODEL": "MUREF"})

    # CID alone is not unique once several surveys are combined (:638)
    df["CIDindex"] = (df["CID"] + "_" + df["IDSURVEY"].astype(str)
                      + "_" + df["FIELD"].astype(str))
    duplicated = df["CIDindex"][df["CIDindex"].duplicated()].tolist()
    if duplicated:
        raise SNANAError(
            f"{len(duplicated)} duplicate CID_IDSURVEY_FIELD in {path}: "
            f"{duplicated[:10]}",
            function_name="hubble_diagram_from",
        )
    return df.set_index("CIDindex")


def _require(df: pd.DataFrame, path: Path, *columns: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SNANAError(f"{path} has no {', '.join(missing)} column",
                         function_name="_require")


def subtract_vpec(diagrams: dict[str, pd.DataFrame], *, renormalised: bool = False) -> None:
    """Remove velocity variance, holding any BBC error rescaling fixed.

    With renormalised=True, apply the same fractional variance removal to
    MUERR_RENORM as to MUERR. The probability/bin normalisation is retained;
    this operation does not rerun the distance fit or estimate new weights.
    """
    for label, df in diagrams.items():
        if "MUERR_VPEC" not in df.columns:
            raise SNANAError(
                f"cannot subtract VPEC: {label} has no MUERR_VPEC column",
                function_name="subtract_vpec",
            )
        original = df["MUERR"].copy()
        variance = original**2 - df["MUERR_VPEC"]**2
        valid = (original.gt(0) & df["MUERR_VPEC"].ge(0)
                 & variance.gt(0) & np.isfinite(variance))
        if not valid.all():
            raise SNANAError(f"{label}: SUBTRACT_VPEC needs finite errors and must leave positive variance",
                             function_name="subtract_vpec")
        adjusted = np.sqrt(variance)
        if renormalised and "MUERR_RENORM" in df:
            if not (np.isfinite(df["MUERR_RENORM"]) & df["MUERR_RENORM"].gt(0)).all():
                raise SNANAError(f"{label}: SUBTRACT_VPEC needs finite positive MUERR_RENORM",
                                 function_name="subtract_vpec")
            df["MUERR_RENORM"] = df["MUERR_RENORM"] * adjusted / original
        df["MUERR"] = adjusted


def use_muerr_renorm(diagrams: dict[str, pd.DataFrame]) -> None:
    """Swap in MUERR_RENORM, which carries the BEAMS 1/sqrt(P_Ia) term (:804)."""
    for df in diagrams.values():
        if "MUERR_RENORM" in df.columns:
            df["MUERR"] = df["MUERR_RENORM"]


def common_set_of_sne(diagrams: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Restrict every diagram to the SNe present in all of them (:778)."""
    combined = None
    for df in diagrams.values():
        combined = df.index if combined is None else combined.intersection(df.index)
    if combined is None or not len(combined):
        raise SNANAError("no SNe common to every FITOPT/MUOPT",
                         function_name="common_set_of_sne")
    return {label: df.loc[combined, :] for label, df in diagrams.items()}


def remove_nans(diagrams: dict[str, pd.DataFrame]):
    """Drop rows the reference diagram cannot use, from every diagram (:2567).

    The mask comes from the reference alone, so the grid stays row-aligned.
    """
    base_name = fitopt_muopt_name((0, 0))
    if base_name not in diagrams:
        raise SNANAError(f"grid has no {base_name}", function_name="remove_nans")
    base = diagrams[base_name]
    # FIELD is a string column and would make every row look non-null
    numeric = base.drop(columns=["FIELD"], errors="ignore")
    mask = ~numeric.isnull().any(axis=1)
    trimmed = {label: df.loc[mask, :] for label, df in diagrams.items()}
    return trimmed, trimmed[base_name]


def rebin_indices(base: pd.DataFrame, bins):
    """Assign each SN to its BBC redshift bin and named coordinate bins.

    Computed once from the reference diagram and reused for every grid cell, so
    the rebinned rows stay aligned across the grid.
    """
    _require(base, "<reference>", "IZBIN", *bins)
    epsilon = 1.0e-6
    zbin = base["IZBIN"].to_numpy(dtype=float)
    if not len(zbin) or not np.all(np.isfinite(zbin) & (zbin >= 0) & (zbin == np.floor(zbin))):
        raise SNANAError("IZBIN must contain non-negative integer bin indices",
                         function_name="rebin_indices")
    coordinates = {"IZBIN": zbin.astype(int)}
    index = coordinates["IZBIN"].copy()
    edges = {}
    for name, count in bins.items():
        values = base[name].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise SNANAError(f"rebin coordinate {name} contains non-finite values",
                             function_name="rebin_indices")
        edges[name] = np.linspace(values.min() - epsilon, values.max() + epsilon,
                                  count + 1)
        coordinates[name] = np.digitize(values, edges[name]) - 1
        index = index * count + coordinates[name]
    return index, coordinates, edges


def rebin_hubble_diagram(df: pd.DataFrame, index, coordinates,
                         z_grid, mu_grid, *, labels=None, limit=None) -> pd.DataFrame:
    """Collapse a per-SN diagram onto the occupied coordinate bins (:885).

    Two different weights are in play and they are not interchangeable: MUREF is
    averaged with 1/MUERR**2, while MU and its error use 1/MUERR_RENORM**2, the
    renormalised error carrying the BEAMS 1/sqrt(P_Ia) term. The bin redshift then
    comes from inverting the reference cosmology at the averaged MUREF rather than
    from averaging z.
    """
    muerr = df["MUERR"].to_numpy(dtype=float)
    renorm = (df["MUERR_RENORM"].to_numpy(dtype=float)
              if "MUERR_RENORM" in df.columns else muerr)
    wgt0 = 1.0 / muerr ** 2
    wgt1 = 1.0 / renorm ** 2
    muref = df["MUREF"].to_numpy(dtype=float)
    mu = df["MU"].to_numpy(dtype=float)

    rows = []
    labels = {"IZBIN": "z", **(labels or {})}
    for i in np.unique(index):
        if limit is not None and i >= limit:
            continue
        mask = index == i
        nevt = int(mask.sum())
        wgt0sum = wgt0[mask].sum()
        wgt1sum = wgt1[mask].sum()
        muref_avg = float((wgt0[mask] * muref[mask]).sum() / wgt0sum)
        label = "-".join(
            f"{labels[name]}{values[mask][0]:02d}" if name == "IZBIN"
            else f"{labels.get(name, name)}{values[mask][0]}"
            for name, values in coordinates.items())
        rows.append({
            "ROW": f"BIN{i:04d}_{label}",
            "zHD": float(np.interp(muref_avg, mu_grid, z_grid)),
            "MU": float((wgt1[mask] * mu[mask]).sum() / wgt1sum),
            "MUERR": float(np.sqrt(1.0 / wgt1sum)),
            "MUREF": muref_avg,
            "NEVT": nevt,
        })
    return pd.DataFrame(rows)


def reference_grid(cosmology: Cosmology | None, n: int = 4000, *,
                   radiation: bool = True):
    """(z, mu) on the biasCor cosmology, for inverting mu -> z (get_HDcalc, :962).

    SNANA hardcodes H0=70 here regardless of the sim's own H0, and covers
    0.001 < z < 4.001 on a geometric grid.
    """
    if cosmology is None:
        # SNANA falls back to flat LCDM with Om0=0.315 here (:982)
        raise SNANAError(
            "The reference distance grid needs the biasCor simulation's "
            "cosmology, which the BBC output does not record; set COSMOLOGY.",
            function_name="reference_grid")
    z_grid = np.geomspace(0.001, 4.001, n)
    z_grid = jnp.asarray(z_grid)
    mu_grid = np.asarray(distance_modulus(
        z_grid, z_grid, cosmology.omega_matter, cosmology.omega_lambda,
        LIGHT_KM / 70.0, cosmology.w0, cosmology.wa, radiation=radiation,
    ))
    return z_grid, mu_grid


def grid_files(config: CreateCovConfig) -> dict[str, Path]:
    """Map ``FITOPT<i>_MUOPT<j>`` to its Hubble-diagram file.

    Binned reads ``.M0DIF``; unbinned and rebin both read ``.FITRES`` (:713-717).
    """
    suffix = "M0DIF" if config.mode == "binned" else "FITRES"
    folder = config.data_dir
    found = {}
    for name in sorted(p.name for p in folder.glob("FITOPT*MUOPT*")):
        if any(s in name for s in _SKIP):
            continue
        stem = name
        for drop in (".gz", f".{suffix}"):
            stem = stem.replace(drop, "")
        if f".{suffix}" not in name:
            continue
        found[stem] = folder / name
    if not found:
        raise SNANAError(
            f"no FITOPT*MUOPT*.{suffix} files under {folder}",
            function_name="grid_files",
        )
    return found


def read_submit_info(input_dir: Path) -> dict:
    path = Path(input_dir) / "SUBMIT.INFO"
    if not path.is_file():
        raise SNANAError(f"no SUBMIT.INFO in {input_dir}",
                         function_name="read_submit_info")
    return read_yaml(path)


def sys_scales(which_opt: str, submit_info: dict,
               scales: dict[str, float]) -> dict[int, tuple[str, float]]:
    """Map each FITOPT/MUOPT number to its ``(label, scale)`` (:1011).

    Index 0 is always ``('DEFAULT', 1.0)``: it is the reference the others are
    differenced against, so scaling it would be meaningless.
    """
    column = _LABEL_COLUMN[which_opt]
    rows = submit_info.get(f"{which_opt}_OUT_LIST")
    if rows is None:
        return {0: ("DEFAULT", 1.0)}

    result = {}
    for row in rows:
        number = int(str(row[0]).replace(which_opt, ""))
        label = str(row[column])
        result[number] = (label, 1.0)
    return apply_sys_scales(result, scales)


def apply_sys_scales(options, scales):
    """Apply label overrides; the reference always has unit scale."""
    result = {number: (label, float(scales.get(label, scale)))
              for number, (label, scale) in options.items() if number != 0}
    result[0] = ("DEFAULT", 1.0)
    return result


def version_photometry(header: dict[str, str]) -> dict[str, str]:
    """The VERSION_PHOTOMETRY keys of a grid FITRES's comment header (:2408).

    SNANA reads the value as ``wdlist[2]`` and raises IndexError when a key has
    none, which SALT2mu writes whenever its input carried no version. jax
    records the empty string instead.
    """
    return {key: value.split()[0] if value else ""
            for key, value in header.items() if "VERSION_PHOTOMETRY" in key}


def isdata_real(header: dict[str, str]) -> int:
    """ISDATA_REAL from a grid file's comment header, -1 when absent (:531)."""
    return int(header["ISDATA_REAL"].split()[0]) if "ISDATA_REAL" in header else -1


def reference_cosmology(given: Cosmology | None,
                        header: dict[str, str]) -> Cosmology | None:
    """The biasCor simulation's cosmology: as BBC records it in its output
    header, or the input's ``COSMOLOGY`` for an output that records none.
    Given both, they must agree."""
    if "SIM_OMEGA_MATTER(BIASCOR)" not in header:
        return given
    value = {key: header[f"SIM_{key}(BIASCOR)"] for key in SIM_COSMOLOGY_KEYS}
    recorded = Cosmology(omega_matter=float(value["OMEGA_MATTER"]),
                         omega_lambda=float(value["OMEGA_LAMBDA"]),
                         w0=float(value["w0_LAMBDA"]), wa=float(value["wa_LAMBDA"]),
                         mushift=parse_mushift(value["MUSHIFT"]))
    if given is not None and given != recorded:
        raise SNANAError(
            f"COSMOLOGY {given} disagrees with the biasCor simulation's, "
            f"{recorded}, recorded in the BBC output",
            function_name="reference_cosmology")
    return recorded


COVOPT_ALL = "[ALL] [,]"


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
        require_finite_errors(self.hubble_diagram)
        return cov_invert(cov, self.hubble_diagram["MUERR"].to_numpy())


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
        diagrams = common_set_of_sne(diagrams)
    if mode == "rebin":
        bins = resolve_rebin_bins(rebin_bins, nbin_x1, nbin_c)
        z_grid, mu_grid = reference_grid(cosmology, radiation=radiation)
        reference = diagrams[fitopt_muopt_name((0, 0))]
        index, coordinates, edges = rebin_indices(reference, bins)
        nbin_z = len(np.unique(coordinates["IZBIN"]))
        rebin_info = (nbin_z, edges)
        # Preserve SNANA row names and its truncated loop for legacy flags.
        # Named bins retain all occupied redshift bins, including gaps.
        legacy = rebin_bins is None
        limit = nbin_z * nbin_x1 * nbin_c if legacy else None
        diagrams = {label: rebin_hubble_diagram(
                        df, index, coordinates, z_grid, mu_grid,
                        labels={"x1": "x"} if legacy else None, limit=limit)
                    for label, df in diagrams.items()}
    elif mode == "unbinned":
        use_muerr_renorm(diagrams)
    diagrams, base = remove_nans(diagrams)

    contribs = contributions(diagrams, fitopt_scales, muopt_scales,
                                        extra_covs or (), base)

    covariances = {}
    for c in all_covopts:
        value = covsys_from_covopt(c, contribs, base, calibrators,
                                              factorized=factorized)
        covariances[c.label] = ({"kind": "factorized", **value} if isinstance(value, dict)
                               else {"kind": "systematics", "matrix": value})
    all_covariance = covariances.get("ALL")
    systematic_errors = None if all_covariance is None else (
        np.sqrt(np.sum(all_covariance["U"] ** 2, axis=1))
        if all_covariance["kind"] == "factorized"
        else muerr_sys(all_covariance["matrix"]))

    grid = (reference_grid(cosmology, radiation=radiation)
            if isdata_real == 0 else None)

    return CovResult(hubble_diagram=base, covariances=covariances,
                     isdata_real=isdata_real, muerr_sys=systematic_errors,
                     rebin=rebin_info, systematic_diagrams=dict(diagrams),
                     mu_grid=grid)


def cov_filename(index: int, prefix: str, fmt: str) -> str:
    suffix = {"npz": "npz", "text": "txt.gz"}[fmt]
    return f"{prefix}_{index:03d}.{suffix}"


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


def write_outputs(outdir: Path, result: CovResult, config: CreateCovConfig) -> None:
    """Write the covariance products the write mask selects, the HD, and INFO.YML."""
    outdir = Path(outdir)
    mask, fmt = config.write_mask_cov, config.write_format_cov
    unbinned = config.mode == "unbinned"

    # the reference cell keeps the plain name; only the others are labelled (:1657)
    base_name = fitopt_muopt_name((0, 0))
    diagrams = {HD_FILENAME: result.hubble_diagram}
    if config.systematic_hd:
        diagrams = {(HD_FILENAME if label == base_name
                     else f"{label}_{HD_FILENAME}"): hd
                    for label, hd in result.systematic_diagrams.items()}
    for name, hd in diagrams.items():
        musim = (None if result.mu_grid is None
                 else np.interp(hd["zHD"].to_numpy(), *result.mu_grid))
        if unbinned:
            write_hd_unbinned(outdir / name, hd, result.isdata_real,
                                    result.muerr_sys, musim)
        else:
            write_hd_binned(outdir / name, hd, result.isdata_real,
                                  result.muerr_sys, musim, result.rebin)

    require_finite_errors(result.hubble_diagram)
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
            path.parent.mkdir(parents=True, exist_ok=True)
            write_covariance_npz(path, cov)
        else:
            write_covariance_text(path, cov, rows)
    covsys_names = {}
    covtot_inv_names = {}
    factorized_names = {}
    for i, (label, entry) in enumerate(result.covariances.items()):
        covsys_names[i] = covtot_inv_names[i] = factorized_names[i] = None
        compact = entry["kind"] == "factorized"
        if mask & WRITE_MASK_FACTORIZED:
            if compact:
                factorized_names[i] = f"covtot_factorized_{i:03d}.npz"
                write_factorized_covariance(outdir / factorized_names[i], entry)
            else:
                logger.warning("COVOPT %s has no vector factors; writing a dense inverse fallback", label)
        if compact and not mask & 7:
            continue  # Compact-only output needs no dense reconstruction.
        cov = entry["U"] @ entry["U"].T if compact else entry["matrix"]
        if mask & WRITE_MASK_COVSYS:
            covsys_names[i] = cov_filename(i, PREFIX_COVSYS, fmt)
            write(outdir / covsys_names[i], cov)
        if mask & WRITE_MASK_COVTOT_INV or (mask & 8 and not compact):
            covtot_inv_names[i] = cov_filename(i, PREFIX_COVTOT_INV, fmt)
            write(outdir / covtot_inv_names[i],
                  cov_invert(cov, muerr, config.mxsize_test_posdef, label))
        if mask & WRITE_MASK_COVTOT:
            write(outdir / cov_filename(i, PREFIX_COVTOT, fmt),
                  cov + np.diag(muerr ** 2))

    nbin_z = len(result.hubble_diagram) if config.mode != "rebin" else (
        result.rebin[0] if result.rebin else 0)
    submit_info = None if config.input_dir is None else config.input_dir / "SUBMIT.INFO"
    write_info_yml(
        outdir / INFO_YML_FILENAME,
        covsys_labels=list(result.covariances),
        size_hd=len(result.hubble_diagram), isdata_real=result.isdata_real,
        bin_method=bin_method(config.mode, nbin_z,
                                     result.rebin[1] if result.rebin else {}),
        cospar=config.cosmology, version_photometry=result.version_photometry,
        bbc_dir=None if config.input_dir is None else config.data_dir,
        bbc_info_file=submit_info if submit_info is not None and submit_info.is_file() else None,
        sys_scale_file=config.sys_scale_file,
        covsys_names=covsys_names, covtot_inv_names=covtot_inv_names,
        factorized_names=factorized_names if mask & 8 else None)


def select_covariance_grid(grid, muopt):
    """Select a MUOPT and require the reference cells used by the covariance."""
    if muopt >= 0:
        if muopt not in {fitopt_muopt_indices(name)[1] for name in grid}:
            raise SNANAError(f"MUOPT{muopt:03d} is not in the covariance grid")
        grid = {name: value for name, value in grid.items()
                if fitopt_muopt_indices(name)[1] == muopt}
    reference_muopt = max(0, muopt)
    required = {fitopt_muopt_name((0, reference_muopt))} | {
        fitopt_muopt_name((fitopt_muopt_indices(name)[0], reference_muopt)) for name in grid}
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
        diagrams = {fitopt_muopt_name((fitopt_muopt_indices(name)[0], 0)): frame
                    for name, frame in diagrams.items()}
        muopt_scales = {0: ("DEFAULT", 1.0)}
    if config.subtract_vpec:
        subtract_vpec(diagrams, renormalised=config.mode == "unbinned")
    return build_covariance(
        diagrams, fitopt_scales, muopt_scales, config.covopts,
        cosmology=config.cosmology, radiation=config.radiation, mode=config.mode,
        rebin_bins=config.rebin_bins, nbin_x1=config.nbin_x1, nbin_c=config.nbin_c,
        isdata_real=isdata_real, calibrators=config.calibrators,
        extra_covs=config.extra_covs, nosys=config.nosys,
        factorized=bool(config.write_mask_cov & 8))


def create_covariance(config: CreateCovConfig, *, write: bool = True) -> CovResult:
    """Read a submit_batch BBC output tree and aggregate it."""
    files = grid_files(config)
    reference_name = fitopt_muopt_name((0, max(0, config.muopt)))
    selected_files = select_covariance_grid(files, config.muopt)
    header = read_fitres_header(str(selected_files[reference_name]))
    config = replace(config, cosmology=reference_cosmology(config.cosmology, header))

    submit_info = read_submit_info(config.input_dir)
    fitopt_scales = sys_scales("FITOPT", submit_info, config.sys_scales)
    muopt_scales = sys_scales("MUOPT", submit_info, config.sys_scales)

    diagrams = {name: load_hubble_diagram(path, config.mode)
                for name, path in selected_files.items()}
    result = prepare_covariance(config, diagrams, fitopt_scales, muopt_scales,
                                isdata_real=isdata_real(header))
    reference = resolve_fits_path(config.data_dir / f"{reference_name}.FITRES")
    if Path(reference).is_file():
        result.version_photometry = version_photometry(
            read_fitres_header(str(reference)))
    elif write and config.outdir is not None:
        raise SNANAError(f"{config.data_dir} holds no nominal FITRES "
                         f"({reference_name}.FITRES or .gz)",
                         function_name="create_covariance")
    if write and config.outdir is not None:
        write_outputs(config.outdir, result, config)
    return result


def compute_covariance(inputs=None, *, diagrams=None, fitopt_scales=None,
                       muopt_scales=None, isdata_real=-1, write=False, **settings):
    """Build systematic covariances using a mapping, YAML file or keyword settings.

    For file inputs, supply ``input_dir`` and ``version``. For in-memory
    inputs, ``diagrams`` maps FITOPT/MUOPT names to pandas DataFrames;
    ``fitopt_scales`` and ``muopt_scales`` map each option number to its
    ``(label, uncertainty_scale)``. Include option 0 as ``('DEFAULT', 1.0)``.
    ``sys_scales`` can override the uncertainty scales by label, as for files.

    ``inputs`` may also be a ``CreateCovConfig``. Keyword settings go through
    the configuration parser. Returns ``CovResult``, with ``hubble_diagram``
    and ``covariances`` keyed by covariance label. ``covtot_inv(label)``
    returns the inverse total covariance for that label. Files are written only with
    ``write=True``, which requires ``outdir``.
    """
    if isinstance(inputs, CreateCovConfig):
        values = inputs
    elif inputs is None or isinstance(inputs, Mapping):
        values = inputs or {}
    else:
        values = read_yaml(inputs)
    config = build_config(values, settings)
    if write and config.outdir is None:
        raise SNANAError('outdir is required to write covariance outputs')
    if diagrams is None:
        if fitopt_scales is not None or muopt_scales is not None:
            raise SNANAError('Option scale mappings require in-memory diagrams')
        if config.input_dir is None or not config.version:
            raise SNANAError('Supply diagrams, or input_dir and version')
        return create_covariance(config, write=write)
    if config.input_dir is not None or config.version:
        raise SNANAError('Supply an in-memory grid or input_dir/version, not both')
    if fitopt_scales is None or muopt_scales is None:
        raise SNANAError('In-memory diagrams require fitopt_scales and muopt_scales')
    if 0 not in fitopt_scales or 0 not in muopt_scales:
        raise SNANAError('Option scale mappings require the reference option 0')
    grid = {name: hubble_diagram_from(frame.copy(), config.mode, source=name)
            for name, frame in select_covariance_grid(diagrams, config.muopt).items()}
    result = prepare_covariance(
        config, grid, apply_sys_scales(fitopt_scales, config.sys_scales),
        apply_sys_scales(muopt_scales, config.sys_scales), isdata_real=isdata_real)
    if write:
        write_outputs(config.outdir, result, config)
    return result
