"""Profile errors on a fixed fit objective; no weights or priors are re-centred.

Status per side: 0 crossing found, 1 parameter bound reached, 2 nuisance
minimisation failed, 3 a lower minimum found, 4 search budget exhausted,
5 parameter not profiled.
Errors are positive distances from the supplied minimum. A bound returns its
distance with status 1; failed searches return NaN in the raw profile output.
Reported symmetric errors may use an explicitly flagged Hessian fallback.
Nuisance fits are local: status 0 does not establish a global profile minimum.
"""

import jax.numpy as jnp

from jax_snlc_core.utils.newton_lm import quadratic_decrease


def profile_error_type(distances, status):
    """SNANA error source: 1 for two valid MINOS sides, 2 for Hessian errors."""
    valid = jnp.all((status == 0) & jnp.isfinite(distances) & (distances > 0), axis=-1)
    return jnp.where(valid, 1, 2)


def reported_errors(hessian_errors, distances, status):
    """Average valid profile sides; otherwise retain the full Hessian error."""
    return jnp.where(profile_error_type(distances, status) == 1,
                     jnp.mean(distances, axis=-1), hessian_errors)


def _remaining_decrease(params, lower, upper, hessian, grad):
    """Bound the possible quadratic improvement, excluding blocked directions."""
    blocked = (((params <= lower) & (grad > 0))
               | ((params >= upper) & (grad < 0)) | (lower == upper))
    scale = jnp.sqrt(jnp.maximum(jnp.abs(jnp.diag(hessian)), 1e-6))
    grad = jnp.where(blocked, 0.0, grad) / scale
    curvature = (0.5 * (hessian + hessian.T)) / jnp.outer(scale, scale)
    curvature = (curvature * jnp.outer(~blocked, ~blocked)
                 + jnp.diag(blocked.astype(params.dtype)))
    eigenvalues, vectors = jnp.linalg.eigh(curvature)
    components = vectors.T @ grad
    extent = jnp.where(blocked, 0.0,
                       jnp.maximum(params - lower, upper - params) * scale)
    return quadratic_decrease(eigenvalues, components, vectors, extent)
