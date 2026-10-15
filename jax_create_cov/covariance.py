"""Building systematic covariance from a grid of Hubble diagrams.

Each FITOPT/MUOPT variant contributes the outer product of its distance-modulus
difference from a reference; a covopt selects which of those contributions to sum.
"""

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from jax_snlc_core.io.fits_path import open_text
from jax_snlc_core.utils import SNANAError, snana_log as slog

from .config import Covopt
from .config_parser import apply_filter
from .inputs import REF, join_name, split_name

logger = logging.getLogger(__name__)


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
                  ) -> dict[str, Contribution]:
    """Map ``FITOPT_LABEL|MUOPT_LABEL`` to its mu-difference vector (:1313).

    The two axes reference differently: a MUOPT variant is differenced against
    MUOPT000 *of its own FITOPT*, while a FITOPT variant is differenced against
    FITOPT000/MUOPT000. Their scales multiply.
    """
    result = {}
    for name, df in diagrams.items():
        fitopt, muopt = split_name(name)
        if fitopt not in fitopt_scales:
            raise SNANAError(f"{name}: FITOPT{fitopt:03d} is not in SUBMIT.INFO",
                             function_name="contributions")
        if muopt not in muopt_scales:
            raise SNANAError(f"{name}: MUOPT{muopt:03d} is not in SUBMIT.INFO",
                             function_name="contributions")
        fitopt_label, fitopt_scale = fitopt_scales[fitopt]
        muopt_label, muopt_scale = muopt_scales[muopt]

        if fitopt == REF and muopt == REF:
            diff = np.zeros(len(df))
        elif muopt != REF:
            other = diagrams[join_name(fitopt, REF)]
            diff = mu_difference(df, other, fitopt_scale * muopt_scale)
        else:
            other = diagrams[join_name(REF, REF)]
            diff = mu_difference(df, other, fitopt_scale)

        result[f"{fitopt_label}|{muopt_label}"] = Contribution(diff=diff)

    for label, weight, path in extra_covs:
        # SNANA stores the weight and never applies it (:1453)
        result[f"DEFAULT|{label}"] = Contribution(
            cov=covsys_from_covfile(base, path) * weight ** 2)
    return result


def covsys_from_covopt(covopt: Covopt, contribs: dict[str, Contribution],
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
            slog.warn(logger, "CREATE_COV",
                      f"none of the {len(calibrators)} CALIBRATORS match a CID; "
                      "no redshift-systematic masking will be applied")

    # External covariance contributions keep their existing dense treatment.
    # Only matched contributions determine whether this COVOPT needs fallback.
    selected = {key: value for key, value in contribs.items()
                if apply_filter(key.split("|")[0], covopt.fitopt_filter)
                and apply_filter(key.split("|")[1], covopt.muopt_filter)}
    compact = factorized and all(c.cov is None for c in selected.values())
    vectors, labels = [], []
    total = None
    has_systematic = False
    for key, contribution in contribs.items():
        fitopt_label, muopt_label = key.split("|")
        if not (apply_filter(fitopt_label, covopt.fitopt_filter)
                and apply_filter(muopt_label, covopt.muopt_filter)):
            continue

        has_systematic |= (fitopt_label, muopt_label) != ("DEFAULT", "DEFAULT")
        if compact:
            vector = contribution.diff.copy() * covopt.scale
            if calib_mask is not None and _is_redshift_systematic(fitopt_label, muopt_label):
                vector[calib_mask] = 0.
            if np.any(vector):
                vectors.append(vector)
                labels.append(key)
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
        slog.warn(logger, "CREATE_COV",
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
        slog.warn(logger, "CREATE_COV",
                  f"{label or 'covtot'} inverse is not unitary "
                  f"(residual {residual:.3e})")

    # skipped above mxsize_test_posdef, as SNANA (:1531)
    if len(covtot) <= mxsize_test_posdef:
        if not np.all(np.linalg.eigvals(covtot_inv) > 0):
            slog.warn(logger, "CREATE_COV",
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
