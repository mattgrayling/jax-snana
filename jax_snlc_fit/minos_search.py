"""Bounded profile steps shared by redshift and all-parameter uncertainties.

The caller freezes the objective and supplies its minimum. This search does
not repair that minimum, retry failed nuisance fits or search other branches.
Statuses match ``minos``: crossing 0, bound 1, nuisance failure 2, lower
minimum 3, exhausted root budget 4. Failed distances are NaN.
"""

import jax
import jax.numpy as jnp

from jax_snlc_core.utils.newton_lm import _grad_and_hess, _newton_step
from .minos import _remaining_decrease

NUISANCE_STEPS = 20
NUISANCE_TOLERANCE = 1e-5
BFGS_DAMPING = 0.2
MIN_SCALED_CURVATURE = 0.01


def _nuisance_fit(loss, seed, lower, upper, curvature, active):
    """Shared Newton-LM steps with curvature updated from accepted gradients.

    The starting covariance supplies the curvature. BFGS updates retain
    positive curvature; exact Hessians validate the final profile endpoints.
    Under vmap, the loop stops when every active event finishes.
    """
    value, gradient = jax.value_and_grad(loss)(seed)
    scale = jnp.sqrt(jnp.maximum(jnp.abs(jnp.diag(curvature)), 1e-6))
    normal = curvature / jnp.outer(scale, scale)
    eigenvalues, vectors = jnp.linalg.eigh((normal + normal.T) / 2)
    curvature = (((vectors * jnp.maximum(eigenvalues, MIN_SCALED_CURVATURE))
                  @ vectors.T) * jnp.outer(scale, scale))
    carry = (seed, jnp.asarray(1e-3, seed.dtype), value,
             jnp.bool_(False), jnp.asarray(1., seed.dtype))

    def advance(state):
        count, carry, gradient, curvature, done = state
        proposal, diagnostics = _newton_step(
            carry, lower, upper, loss, refine=True, tolerance=NUISANCE_TOLERANCE,
            derivatives=(gradient, curvature, carry[2]))
        remaining, _, _, _, trial_gradient = diagnostics
        shift = proposal[0] - carry[0]
        moved = jnp.any(shift != 0)
        next_gradient = jnp.where(moved, trial_gradient, gradient)
        gradient_change = next_gradient - gradient
        curvature_step = curvature @ shift
        slope_change = gradient_change @ shift
        predicted_change = shift @ curvature_step
        # Powell damping keeps the updated curvature positive.
        denominator = jnp.where(
            predicted_change != slope_change, predicted_change - slope_change, 1.0)
        theta = jnp.where(
            slope_change >= BFGS_DAMPING * predicted_change, 1.0,
            (1.0 - BFGS_DAMPING) * predicted_change / denominator)
        gradient_change = theta * gradient_change + (1.0 - theta) * curvature_step
        slope_change = gradient_change @ shift
        update = (moved & (predicted_change > 0)
                  & (slope_change > 1e-10 * jnp.linalg.norm(gradient_change)
                     * jnp.linalg.norm(shift)))
        changed = (
            curvature
            - jnp.outer(curvature_step, curvature_step)
            / jnp.where(predicted_change > 0, predicted_change, 1.0)
            + jnp.outer(gradient_change, gradient_change)
            / jnp.where(slope_change > 0, slope_change, 1.0))
        new = (proposal, next_gradient, jnp.where(update, changed, curvature))
        old = (carry, gradient, curvature)
        kept = jax.tree.map(lambda a, b: jnp.where(done, a, b), old, new)
        return count + 1, *kept, done | (remaining <= NUISANCE_TOLERANCE)

    initial = (jnp.asarray(0), carry, gradient, curvature, ~jnp.asarray(active))
    final = jax.lax.while_loop(
        lambda state: (state[0] < NUISANCE_STEPS) & ~state[-1], advance, initial)
    _, carry, _, curvature, done = final
    return carry[0], carry[2], done, curvature


def profile_state(params, lower, upper, covariance, minimum, index, sign):
    """Hessian starting distance and empty contour bracket for one side."""
    variance = covariance[index, index]
    sigma = jnp.sqrt(variance)
    limit = jnp.where(sign < 0, params[index] - lower[index],
                      upper[index] - params[index])
    valid = jnp.isfinite(sigma) & (sigma > 0) & jnp.isfinite(minimum)
    status = jnp.where(limit <= 0, 1, jnp.where(valid, -1, 2))
    zero = jnp.asarray(0.0, params.dtype)
    return dict(lower=zero, upper=jnp.minimum(sigma, limit),
                delta_lower=zero, delta_upper=zero,
                curvature=2 * jnp.linalg.inv(covariance),
                bracketed=jnp.asarray(False),
                trial=jnp.minimum(sigma, limit), fitted=params,
                distance=jnp.where(status == 1, zero, jnp.nan), status=status,
                value=minimum, converged=jnp.asarray(True))


def profile_step(loss, params, lower, upper, covariance, minimum, index, sign,
                 state, *, prepare_loss=None):
    """One warm-started nuisance fit and safeguarded contour bracket update."""
    nuisance = jnp.array([[j for j in range(params.size) if j != i]
                         for i in range(params.size)], dtype=int)[index]
    variance = covariance[index, index]
    slope = covariance[:, index] / variance
    slope = jnp.where(jnp.isfinite(slope), slope, 0.0).at[index].set(1.0)
    limit = jnp.where(sign < 0, params[index] - lower[index],
                      upper[index] - params[index])
    target = jnp.clip(params[index] + sign * state['trial'],
                      lower[index], upper[index])
    previous = state['fitted']
    seed = jnp.clip(previous + slope * (target - previous[index]),
                    lower, upper).at[index].set(target)
    curvature = state['curvature']
    if params.size > 1:
        fixed_loss = loss if prepare_loss is None else prepare_loss(seed)
        def objective(q):
            return fixed_loss(seed.at[nuisance].set(q))
        fitted, value, converged, nuisance_curvature = _nuisance_fit(
            objective, seed[nuisance], lower[nuisance], upper[nuisance],
            curvature[nuisance][:, nuisance], state['status'] < 0)
        curvature = curvature.at[nuisance[:, None], nuisance[None, :]].set(nuisance_curvature)
        fitted = seed.at[nuisance].set(fitted)
    else:
        fitted, value, converged = seed, loss(seed), jnp.asarray(True)

    delta = value - minimum
    valid = jnp.isfinite(delta) & jnp.all(jnp.isfinite(fitted)) & converged
    crossed = delta >= 1.0
    lower_distance = jnp.where(valid & ~crossed, state['trial'], state['lower'])
    upper_distance = jnp.where(valid & crossed, state['trial'], state['upper'])
    lower_delta = jnp.where(valid & ~crossed, delta, state['delta_lower'])
    upper_delta = jnp.where(valid & crossed, delta, state['delta_upper'])
    bracketed = state['bracketed'] | (valid & crossed)
    root_lo = jnp.sqrt(jnp.maximum(lower_delta, 0.0))
    root_hi = jnp.sqrt(jnp.maximum(upper_delta, 0.0))
    fraction = (1.0 - root_lo) / jnp.maximum(root_hi - root_lo, 1e-12)
    fraction = jnp.where(jnp.isfinite(fraction),
                         jnp.clip(fraction, 0.1, 0.9), 0.5)
    expansion = jnp.minimum(2.0, jax.lax.rsqrt(jnp.maximum(delta, 1e-12)))
    # Extrapolate in sqrt(delta chi2), which is linear for a quadratic loss.
    previous_root = jnp.sqrt(jnp.maximum(state['delta_lower'], 0.0))
    rise = root_lo - previous_root
    expansion = jnp.where(rise <= 0, 2.0, expansion)
    secant_trial = (state['trial'] + (1.0 - root_lo)
                    * (state['trial'] - state['lower']) / jnp.maximum(rise, 1e-12))
    outward = jnp.where(rise > 0,
        jnp.clip(secant_trial, state['trial'], 2 * state['trial']),
        expansion * state['trial'])
    next_trial = jnp.where(
        bracketed, lower_distance + fraction * (upper_distance - lower_distance),
        jnp.minimum(outward, limit))
    status = jnp.where(valid, -1, 2)
    status = jnp.where(valid & (state['trial'] >= limit) & ~crossed, 1, status)
    status = jnp.where(valid & (jnp.abs(delta - 1.0) <= 0.01), 0, status)
    status = jnp.where(jnp.isfinite(delta) & (delta < -0.01), 3, status)
    distance = jnp.where((status == 0) | (status == 1), state['trial'], jnp.nan)
    updated = dict(lower=lower_distance, upper=upper_distance,
                   delta_lower=lower_delta, delta_upper=upper_delta,
                   curvature=curvature,
                   bracketed=bracketed, trial=next_trial, fitted=fitted,
                   distance=distance, status=status, value=value, converged=converged)
    return jax.tree.map(lambda old, new: jnp.where(state['status'] < 0, new, old),
                         state, updated)


def validate_endpoint(loss, lower, upper, minimum, index, state, status):
    """Check the contour and nuisance stationarity using an exact Hessian."""
    p = state['fitted']
    nuisance = jnp.array([[j for j in range(p.size) if j != i]
                         for i in range(p.size)], dtype=int)[index]
    objective = lambda q: loss(p.at[nuisance].set(q))
    g, H, value = _grad_and_hess(p[nuisance], objective, with_value=True)
    remaining = _remaining_decrease(p[nuisance], lower[nuisance], upper[nuisance], H, g)
    delta = value - minimum
    stationary = jnp.isfinite(remaining) & (remaining <= 1e-3)
    contour = jnp.where(status == 1, delta < 1., jnp.abs(delta - 1.) <= .01)
    pending = (status == 0) | (status == 1)
    status = jnp.where(pending & ~(stationary & contour), 2, status)
    distance = jnp.where((status == 0) | (status == 1), state['distance'], jnp.nan)
    return distance, status
