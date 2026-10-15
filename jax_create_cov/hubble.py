"""Loading and conditioning the Hubble diagrams a covariance is built from."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd

from jax_snlc_core.io.fitres_reader import read_fitres
from jax_snlc_core.physics.constants import LIGHT_KM
from jax_snlc_core.physics.cosmology import distance_modulus
from jax_snlc_core.utils import SNANAError

from .config import Cosmology
from .inputs import REF, join_name

# SNANA's sentinel for an undefined error, turned into NaN so remove_nans drops
# the row (:606)
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
    base_name = join_name(REF, REF)
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
