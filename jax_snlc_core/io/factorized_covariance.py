"""SNANA's covariance arrays C = diag(D) + U U.T."""

from pathlib import Path

import numpy as np

from jax_snlc_core.utils import SNANAError


def validate_factorized_covariance(covariance):
    """Validate and normalise the factor arrays at input/output boundaries."""
    if not {"diag", "U"} <= covariance.keys():
        raise SNANAError("factorized covariance requires diag and U")
    diag = np.asarray(covariance["diag"], dtype=np.float64)
    vectors = np.asarray(covariance["U"], dtype=np.float64)
    labels = tuple(covariance.get("vec_labels", ()))
    if diag.ndim != 1 or vectors.ndim != 2 or vectors.shape[0] != diag.size:
        raise SNANAError("factorized covariance requires diag (N,) and U (N,K)")
    if not np.isfinite(diag).all() or (diag <= 0).any():
        raise SNANAError("factorized covariance statistical variances must be finite and positive")
    if not np.isfinite(vectors).all():
        raise SNANAError("factorized covariance U contains non-finite values")
    if labels and len(labels) != vectors.shape[1]:
        raise SNANAError("factorized covariance labels must match U columns")
    return {"diag": diag, "U": vectors, "vec_labels": labels}


def dense_factorized_covariance(covariance):
    """Materialize only for explicitly requested legacy output."""
    return np.diag(covariance["diag"]) + covariance["U"] @ covariance["U"].T


def read_factorized_covariance(path, *, size=None):
    with np.load(path, allow_pickle=False) as payload:
        if not {"nsn", "diag", "U"} <= set(payload.files):
            raise SNANAError(f"{path}: factorized covariance needs nsn, diag and U")
        nsn = payload["nsn"]
        if nsn.size != 1 or not np.issubdtype(nsn.dtype, np.integer):
            raise SNANAError(f"{path}: nsn must contain one integer")
        result = validate_factorized_covariance({
            "diag": payload["diag"], "U": payload["U"],
            "vec_labels": tuple(payload["vec_labels"].astype(str)) if "vec_labels" in payload else ()})
        if int(nsn.item()) != result["diag"].size or (size is not None and size != result["diag"].size):
            raise SNANAError(f"{path}: factorized covariance size does not match HD rows/nsn")
    return result


def write_factorized_covariance(path, covariance):
    covariance = validate_factorized_covariance(covariance)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, nsn=np.array([covariance["diag"].size]), diag=covariance["diag"],
             U=covariance["U"], vec_labels=np.asarray(covariance["vec_labels"], dtype=str))
