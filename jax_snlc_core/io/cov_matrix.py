"""SNANA covariance-matrix files.

``create_covariance`` writes its matrices as an NPZ holding the row count in
``nsn`` and only the **upper triangle** in ``cov``, as float32
(create_covariance.py:2314). Read as a flattened square it silently
misreshapes whenever N(N+1)/2 happens to be a perfect square, as for
N = 1, 8, 49, 288.
"""

import numpy as np

from jax_snlc_core.io.fits_path import open_text
from jax_snlc_core.utils import SNANAError


def expand_upper_triangle(values, n: int) -> np.ndarray:
    """Rebuild the full symmetric matrix from its packed upper triangle."""
    values = np.asarray(values, dtype=np.float64)
    expected = n * (n + 1) // 2
    if values.size != expected:
        raise SNANAError(
            f"covariance has {values.size} values, not the {expected} of an "
            f"upper triangle for nsn={n}",
            function_name="read_covariance_npz",
        )
    full = np.zeros((n, n))
    full[np.triu_indices(n)] = values
    return full + np.triu(full, k=1).T


def read_covariance_npz(path) -> np.ndarray:
    """Read a create_covariance NPZ as a full symmetric matrix."""
    with np.load(path) as payload:
        n = int(np.asarray(payload["nsn"]).ravel()[0])
        triangle = payload["cov"]
    return expand_upper_triangle(triangle, n)


def write_covariance_npz(path, cov) -> None:
    """Write a matrix in create_covariance's NPZ form."""
    cov = np.asarray(cov)
    np.savez(path, nsn=[len(cov)],
             cov=cov[np.triu_indices_from(cov)].astype(np.float32))


def read_covariance_text(path) -> np.ndarray:
    """Read SNANA's text covariance: the dimension, then the flattened square
    matrix, one value per line, optionally gzipped.

    Only the first token of a line is read, as ``read_mucov_text`` does
    (wfit.c:2011), so the per-element labels that ``--label_cov_rows`` appends
    are ignored.
    """
    values = []
    with open_text(path) as stream:
        for line in stream:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                values.append(float(stripped.split()[0]))
    if not values:
        raise SNANAError(
            f"empty covariance file: {path}",
            function_name="read_covariance_text",
        )
    size = int(values[0])
    flat = np.asarray(values[1:], dtype=np.float64)
    if flat.size != size * size:
        raise SNANAError(
            f"{path} contains {flat.size} values; expected {size * size}",
            function_name="read_covariance_text",
        )
    return flat.reshape(size, size)
