"""Prepare stable compact covariance weights once for repeated JAX fits."""

import jax.numpy as jnp
import numpy as np
from scipy.linalg import cho_factor, cho_solve


def prepare_precision(diag, U):
    """Precompute weights for stable repeated likelihood evaluations.

    For whitened residual y, a = mapping.T @ y gives the positive quadratic
    ||y - design @ a||² + ||a||² = residual.T C^-1 residual. Preparation uses
    float64; the likelihood needs only matrix-vector products, without solves.
    """
    # Prepare in host float64 even when the JAX fit uses float32.
    diag = np.asarray(diag, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    invstd = 1. / np.sqrt(diag)
    design = invstd[:, None] * U
    small = np.eye(design.shape[1]) + design.T @ design
    mapping = cho_solve(cho_factor(small, lower=True), design.T).T
    # C^-1 1 and 1.T C^-1 1 for the magnitude-offset profile. Evaluate the
    # latter as a positive sum too, including when a systematic is constant.
    coefficients = mapping.T @ invstd
    remainder = invstd - design @ coefficients
    weighted_ones = invstd * remainder
    sum_weights = remainder @ remainder + coefficients @ coefficients
    return {name: jnp.asarray(value) for name, value in (
        ("invstd", invstd), ("design", design), ("mapping", mapping),
        ("weighted_ones", weighted_ones), ("sum_weights", sum_weights))}
