"""Fitted-parameter errors and covariances.

:func:`errors_from_chi2` takes them from the chi2 a fit minimised, as
MINUIT's are from its FCN, and :func:`profile_errors` profiles selected
parameters on that fixed objective, as MINOS does; :func:`reported_errors`
and :func:`profile_error_type` combine the two. After the fit,
:func:`errmax_bad` replaces an error too small to mean anything by the
parameter's step and flags it, and :func:`fix_covar_lcfit` repairs the
reported covariance.

The profile search re-centres no weights or priors. It selects either the
redshift or every free parameter before compilation, and both use ten
crossing trials, twenty nuisance steps per trial, one validated twenty-step
central repair, and six continuation trials after it.

Profile status per side: 0 crossing found, 1 parameter bound reached, 2
nuisance minimisation failed, 3 a lower minimum found, 4 search budget
exhausted, 5 free parameter not profiled; a held parameter has distance and
status 0. Errors are positive distances from the supplied minimum. A bound
returns its distance with status 1; failed searches return NaN in the raw
profile output. Reported symmetric errors may use an explicitly flagged
Hessian fallback. Nuisance fits are local: status 0 does not establish a
global profile minimum.
"""

import dataclasses
import logging

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core import logging as log
from jax_snlc_core.optimiser import (grad_and_hess, newton_lm, newton_lm_step,
                                     quadratic_decrease)
from jax_snlc_core.utils import fix_covariance

logger = logging.getLogger(__name__)


def errors_from_chi2(chi2, p_scaled, sn_param_scales, free):
    """One SN's errors, covariance, Hessian condition number and EDM of
    ``chi2`` at ``p_scaled``, over the ``free`` parameters, and whether the
    unregularised free Hessian is positive definite.

    The covariance is the inverse of half the Hessian on the free subspace,
    which is the likelihood curvature MINUIT reports; the ridge keeps a
    singular Hessian invertible, and EDM is the same quadratic form on the
    free gradient. A held parameter's rows and columns are zero.
    """
    def half_grad(q):
        g = 0.5 * jax.grad(chi2)(q)
        return g, g

    H, g = jax.jacfwd(half_grad, has_aux=True)(p_scaled)
    return errors_from_derivatives(H[free][:, free], g[free], sn_param_scales, free)


def errors_from_derivatives(H_free, g_free, sn_param_scales, free):
    """Errors from the half-Hessian and half-gradient on the free subspace.

    Profiling already evaluates these derivatives to check its minimum. Reuse
    that evaluation for the covariance, including zero rows for held parameters.
    """
    cov_free = jnp.linalg.inv(H_free + 1e-12 * jnp.eye(H_free.shape[0]))
    size = sn_param_scales.size
    cov_scaled = jnp.zeros((size, size), dtype=H_free.dtype).at[
        free[:, None], free[None, :]].set(cov_free)
    # p_phys = p_scaled * scales, so Cov(phys) = S Cov(scaled) S^T
    cov_phys = cov_scaled * jnp.outer(sn_param_scales, sn_param_scales)
    # LFIXPAR_ALL leaves no free direction
    if H_free.shape[0] == 0:
        cond = jnp.ones((), dtype=H_free.dtype)
        positive = jnp.asarray(True)
    else:
        eigenvalues = jnp.linalg.eigvalsh(H_free)
        positive = jnp.all(jnp.isfinite(eigenvalues) & (eigenvalues > 0))
        ev = jnp.abs(eigenvalues)
        cond = jnp.max(ev) / jnp.maximum(jnp.min(ev), jnp.finfo(ev.dtype).tiny)
    return (jnp.sqrt(jnp.abs(jnp.diag(cov_phys))), cov_phys,
            cond, g_free @ cov_free @ g_free, positive)


def fix_covar_lcfit(results, model, opt):
    """OPT_COVAR_LCFIT: repair the output COV(amplitude, shape, colour).

    FIX_COVAR_LCFIT (snlc_fit.F90:2587) floors the eigenvalues of the 3x3 in
    magnitude space, where the amplitude axis carries a Jacobian the model
    supplies, then divides that back out. Bit 1 rebuilds every matrix. A model
    that declares no such basis is left alone.
    """
    specs = model.get_fit_parameters()
    cov_names = list(specs)
    repaired = [n for n in cov_names
                if n != 'PKMJD' and not specs[n].get('deferred')]
    if model.distance_param not in repaired:
        return results
    idx = [cov_names.index(r) for r in repaired]
    n_par = len(repaired)
    amp_at = repaired.index(model.distance_param)

    amp = np.asarray(results.params[model.distance_param])
    jac = np.ones((len(amp), n_par))
    jac[:, amp_at] = model.amplitude_jacobian(amp)

    cov = np.array(results.param_covariances)
    sub = cov[np.ix_(range(cov.shape[0]), idx, idx)] * jac[:, :, None] * jac[:, None, :]
    # SNANA takes the diagonal from the reported errors, not from FITERRMAT
    err = np.stack([np.asarray(results.param_errors[r]) for r in repaired],
                   axis=1)
    sub[:, range(n_par), range(n_par)] = (err * jac) ** 2

    upd = fix_covariance(sub, force=bool(opt & 2))
    changed = np.any(upd != sub, axis=(1, 2))

    new_errors = dict(results.param_errors)
    for k, name in enumerate(repaired):
        e = np.sqrt(np.abs(upd[:, k, k])) / np.abs(jac[:, k])
        new_errors[name] = np.where(
            changed, e, np.asarray(results.param_errors[name]))
    for a in range(n_par):
        for b in range(n_par):
            if a != b:
                v = upd[:, a, b] / (jac[:, a] * jac[:, b])
                cov[:, idx[a], idx[b]] = np.where(changed, v, cov[:, idx[a], idx[b]])

    n_fixed = int(np.sum(changed))
    if n_fixed:
        log.line(logger, f"OPT_COVAR_LCFIT={opt}: repaired COV for {n_fixed} of "
                 f"{len(amp)} events")
    return dataclasses.replace(results, param_errors=new_errors,
                               param_covariances=cov)


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
        proposal, diagnostics = newton_lm_step(
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
    nuisance = _nuisance_indices(params.size)[index]
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
    nuisance = _nuisance_indices(p.size)[index]
    objective = lambda q: loss(p.at[nuisance].set(q))
    g, H, value = grad_and_hess(p[nuisance], objective, with_value=True)
    remaining = _remaining_decrease(p[nuisance], lower[nuisance], upper[nuisance], H, g)
    delta = value - minimum
    stationary = jnp.isfinite(remaining) & (remaining <= 1e-3)
    contour = jnp.where(status == 1, delta < 1., jnp.abs(delta - 1.) <= .01)
    pending = (status == 0) | (status == 1)
    status = jnp.where(pending & ~(stationary & contour), 2, status)
    distance = jnp.where((status == 0) | (status == 1), state['distance'], jnp.nan)
    return distance, status


CROSSING_STEPS = 10
REPAIR_STEPS = 20
CONTINUATION_STEPS = 6


def profile_errors(settings, strategy, loss_of, **inputs):
    """Profile selected free parameters on each event's final fit objective,
    ``loss_of(data)``, or ``loss_of(data, fixed_z=z)`` with the redshift held.

    Search coordinates and bounds are scaled as in the fitting rounds; returned
    distances and covariances use physical units. The result contains the five
    Hessian diagnostics, (distances, statuses), final parameters and chi-squared.
    A validated repair may replace the centre, but never the frozen objective.
    """
    held, free = _held_and_free(settings, strategy)
    selected = _profile_indices(settings, held)
    original_params = inputs['params']
    events, total_params = original_params.shape
    scales = inputs['scales'][:, free]
    distances = jnp.zeros((events, total_params, 2), original_params.dtype).at[
        :, free].set(jnp.nan)
    statuses = jnp.zeros((events, total_params, 2), int).at[:, free].set(5)
    checked = _minimum_errors(settings, strategy, loss_of, **inputs)
    errors, minimum, original_valid = checked[:5], checked[-2], checked[-1]
    if not len(selected):
        return (*errors, (distances, statuses), original_params, minimum)

    z_index = (list(settings['param_names']).index('zPHOT')
               if 'zPHOT' in settings['param_names'] else None)
    z_only = settings['config'].fit.minos == 'z'
    indices = jnp.repeat(jnp.asarray(selected), 2)
    signs = jnp.tile(jnp.array([-1., 1.], original_params.dtype), len(selected))

    def objective(data):
        p = data['params']
        loss = loss_of(data)
        return lambda q: loss(p.at[free].set(q))

    def covariance(errors, params):
        """Condition the search metric; retain the reported Hessian covariance."""
        cov = errors[1][:, free][:, :, free] / (
            scales[:, :, None] * scales[:, None, :])

        def constrained(cov, center, lo, hi):
            active = (center <= lo) | (center >= hi)

            def adjust(cov):
                hessian = 2 * jnp.linalg.inv(cov)
                scale = jnp.sqrt(jnp.maximum(jnp.abs(jnp.diag(hessian)), 1e-6))
                normal = hessian / jnp.outer(scale, scale)
                normal = (normal * jnp.outer(~active, ~active)
                          + jnp.diag(active.astype(center.dtype)))
                eigenvalues, vectors = jnp.linalg.eigh((normal + normal.T) / 2)
                return (2 * ((vectors / jnp.maximum(eigenvalues, .01)) @ vectors.T)
                        / jnp.outer(scale, scale))

            return jax.lax.cond(jnp.any(active), adjust, lambda c: c, cov)

        return jax.vmap(constrained)(
            cov, params[:, free], inputs['lo'][:, free], inputs['hi'][:, free])

    cov = covariance(errors, original_params)

    def advance_profile(data, cov, value, center, index, sign, state):
        p = data['params']
        loss = objective(data)

        def prepare(q):
            z = p.at[free].set(q)[z_index] * data['scales'][z_index]
            fixed = loss_of(data, fixed_z=z)
            if z_only:
                return lambda r: fixed(p.at[free].set(r))
            return lambda r: jax.lax.cond(
                jnp.asarray(free)[index] == z_index,
                lambda q: fixed(p.at[free].set(q)), loss, r)

        return profile_step(
            loss, center, data['lo'][free], data['hi'][free], cov, value,
            index, sign, state, prepare_loss=prepare if z_index is not None else None)

    def initial_state(index, sign):
        return jax.vmap(profile_state,
                        in_axes=(0, 0, 0, 0, 0, None, None))(
            original_params[:, free], inputs['lo'][:, free], inputs['hi'][:, free],
            cov, minimum, index, sign)

    def search_side(args):
        index, sign = args

        def advance(state, _):
            active = state['status'] < 0
            updated = jax.lax.cond(
                jnp.any(active),
                lambda s: jax.vmap(advance_profile, in_axes=(0, 0, 0, 0, None, None, 0))(
                    inputs, cov, minimum, original_params[:, free], index, sign, s),
                lambda s: s, state)
            # Unfinished nuisance fits may seed repair, but cannot form brackets.
            finite = (active & jnp.isfinite(updated['value'])
                      & jnp.all(jnp.isfinite(updated['fitted']), axis=-1))
            record = (updated['fitted'], updated['value'], finite,
                      finite & updated['converged'])
            return updated, record

        return jax.lax.scan(advance, initial_state(index, sign), None, length=CROSSING_STEPS)

    # scan produces [side, trial, event, ...]; flatten side/trial for each
    # event so the repair can choose a seed from any searched parameter.
    states, trail = jax.lax.map(search_side, (indices, signs))
    saved_points = 2 * CROSSING_STEPS * len(selected)
    points = trail[0].transpose(2, 0, 1, 3).reshape(events, saved_points, len(free))
    values = trail[1].transpose(2, 0, 1).reshape(events, saved_points)
    finite = trail[2].transpose(2, 0, 1).reshape(events, saved_points)
    usable = trail[3].transpose(2, 0, 1).reshape(events, saved_points)
    values = jnp.where(finite, values, jnp.inf)
    best_index = jnp.argmin(values, axis=1)
    seed = jnp.take_along_axis(points, best_index[:, None, None], axis=1)[:, 0]
    best_value = jnp.min(values, axis=1)
    seed = jnp.where((best_value < minimum)[:, None], seed, original_params[:, free])
    needs_repair = (best_value < minimum - .01) | ~original_valid

    def repair(_):
        def refine(data, seed, needed):
            center = data['params'][free]
            lo = jnp.where(needed, data['lo'][free], center)
            hi = jnp.where(needed, data['hi'][free], center)
            return newton_lm(jnp.where(needed, seed, center), lo, hi,
                             objective(data), REPAIR_STEPS, refine=True, tolerance=1e-5)[:2]

        refined, _ = jax.vmap(refine)(inputs, seed, needs_repair)
        candidate = original_params.at[:, free].set(jnp.where(
            needs_repair[:, None], refined, original_params[:, free]))
        checked = _minimum_errors(settings, strategy, loss_of, **dict(inputs, params=candidate))
        valid = checked[-1]
        improved = jnp.where(original_valid, checked[-2] < minimum - .01,
                             checked[-2] <= minimum)
        accepted = needs_repair & valid & jnp.isfinite(checked[-2]) & improved
        params = jnp.where(accepted[:, None], candidate, original_params)
        value = jnp.where(accepted, checked[-2], minimum)
        new_errors = jax.tree.map(lambda new, old: jnp.where(
            accepted.reshape((events,) + (1,) * (new.ndim - 1)), new, old),
            checked[:5], errors)
        return params, value, accepted, new_errors

    params, new_minimum, repaired, new_errors = jax.lax.cond(
        jnp.any(needs_repair), repair,
        lambda _: (original_params, minimum, jnp.zeros(events, bool), errors),
        operand=None)
    new_cov = covariance(new_errors, params)
    centers = params[:, free]
    points = jnp.concatenate((centers[:, None, :], points), axis=1)
    values = jnp.concatenate((new_minimum[:, None], values), axis=1)
    usable = jnp.concatenate((jnp.ones((events, 1), bool), usable), axis=1)
    point_indices = jnp.concatenate((jnp.array([-1]),
                                     jnp.repeat(jnp.asarray(selected), 2 * CROSSING_STEPS)))

    def resumed_state(center, lo, hi, cov, value, points, values, usable, index, sign):
        state = profile_state(center, lo, hi, cov, value, index, sign)
        distance = sign * (points[:, index] - center[index])
        eligible = (usable & ((point_indices < 0) | (point_indices == index))
                    & (distance >= 0) & jnp.isfinite(values))
        delta = values - value
        inner_index = jnp.argmax(jnp.where(eligible & (delta < 1.), distance, -jnp.inf))
        lower, delta_lower = distance[inner_index], delta[inner_index]
        limit = jnp.where(sign < 0, center[index] - lo[index], hi[index] - center[index])
        trial = jnp.minimum(limit, jnp.maximum(state['trial'], jnp.minimum(
            2 * lower, lower / jnp.sqrt(jnp.maximum(delta_lower, 1e-12)))))
        state.update(lower=lower, delta_lower=delta_lower, trial=trial,
                     fitted=points[inner_index])
        return state

    def continue_side(args):
        index, sign, old_state = args
        saved = jax.vmap(resumed_state,
                         in_axes=(0, 0, 0, 0, 0, 0, 0, 0, None, None))(
            centers, inputs['lo'][:, free], inputs['hi'][:, free], new_cov,
            new_minimum, points, values, usable, index, sign)
        state = jax.tree.map(lambda a, b: jnp.where(
            repaired.reshape((events,) + (1,) * (a.ndim - 1)), a, b), saved, old_state)

        def advance(state):
            updated = jax.vmap(advance_profile, in_axes=(0, 0, 0, 0, None, None, 0))(
                inputs, new_cov, new_minimum, centers, index, sign, state)
            return jax.tree.map(lambda a, b: jnp.where(
                repaired.reshape((events,) + (1,) * (a.ndim - 1)), a, b), updated, state)

        def step(_, state):
            return jax.lax.cond(jnp.any(repaired & (state['status'] < 0)),
                                advance, lambda s: s, state)

        state = jax.lax.fori_loop(0, CONTINUATION_STEPS, step, state)
        status = jnp.where(state['status'] < 0, 4, state['status'])

        def checked(data, value, state, status):
            return validate_endpoint(
                objective(data), data['lo'][free], data['hi'][free],
                value, index, state, status)

        return jax.vmap(checked)(inputs, new_minimum, state, status)

    distance, status = jax.lax.map(continue_side, (indices, signs, states))
    # All intervals must refer to a validated common centre. A crossing on
    # another side is not usable if any side still finds a lower minimum.
    status = status.T.reshape(events, len(selected), 2)
    unresolved = jnp.any(status == 3, axis=(1, 2))
    valid_center = original_valid | repaired
    status = jnp.where(unresolved[:, None, None], 3, status)
    status = jnp.where(valid_center[:, None, None], status, 2)
    distance = distance.T.reshape(events, len(selected), 2)
    distance = jnp.where((status == 0) | (status == 1), distance, jnp.nan)
    distances = distances.at[:, free[selected]].set(
        distance * inputs['scales'][:, free[selected], None])
    statuses = statuses.at[:, free[selected]].set(status)
    return (*new_errors, (distances, statuses), params, new_minimum)


def _minimum_errors(settings, strategy, loss_of, **inputs):
    """Evaluate the final centre and its exact constrained stationarity."""
    _, free = _held_and_free(settings, strategy)

    def one(data):
        p = data['params']
        loss = loss_of(data)
        objective = lambda q: loss(p.at[free].set(q))
        lower, upper = data['lo'][free], data['hi'][free]
        fitted = p[free]
        g, H, value = grad_and_hess(fitted, objective, with_value=True)
        errors = errors_from_derivatives(0.5 * H, 0.5 * g, data['scales'], free)
        valid = _remaining_decrease(fitted, lower, upper, H, g) <= 5e-5
        return (*errors, p.at[free].set(fitted), value, valid)
    return jax.vmap(one)(inputs)


def _held_and_free(settings, strategy):
    """The parameters the final round holds, by name, and the indices of the
    rest; the live round 1 holds its own set."""
    held = settings['round1_held_params'] if strategy == 'live' else settings['held_params']
    return held, np.array([i for i, name in enumerate(settings['param_names'])
                           if name not in held], dtype=int)


def _nuisance_indices(n_params):
    """Row ``i``: the indices of every parameter but ``i``, the nuisance
    parameters of profiling parameter ``i``."""
    return jnp.array([[j for j in range(n_params) if j != i]
                      for i in range(n_params)], dtype=int)


def _profile_indices(settings, held):
    """Indices to profile within the free-parameter subspace."""
    free = [n for n in settings['param_names'] if n not in held]
    if settings['config'].fit.minos == 'all':
        return np.arange(len(free), dtype=int)
    return np.array([i for i, name in enumerate(free) if name == 'zPHOT'], dtype=int)


def errmax_bad(config, fit_parameters, held_params, results, init_params):
    """``results`` with each floated error too small to mean anything
    replaced by its step size and flagged as bad.

    SNANA substitutes +/-INISTP and flags ERRTYPE_BAD, keeping the event
    (snana.F90:27879). MINOS_ERRMAX_BAD = F only zeroes the floors
    (snlc_fit.F90:1481); the "never left its start" condition still fires.
    """
    errors = dict(results.param_errors)
    if results.error_flags is not None:
        flags = dict(results.error_flags)
    elif results.minos_errors is not None:
        flags = {name: np.asarray(profile_error_type(
            results.minos_errors[name], results.minos_status[name]))
            for name in results.param_names}
    else:
        flags = {name: np.full(results.n_sne, 2, dtype=int)
                 for name in results.param_names}
    for name, spec in fit_parameters.items():
        err = np.asarray(results.param_errors[name])
        flags[name] = np.where(np.isfinite(err), flags[name], 6)
        if name in held_params:
            continue
        floor = (spec.get('errmax_bad', 0.0)
                 if config.fit.minos_errmax_bad else 0.0)
        bad = err < floor
        if results.minos_errors is not None:
            sides = np.asarray(results.minos_errors[name])
            profiled = np.asarray(profile_error_type(
                sides, results.minos_status[name])) == 1
            bad = np.where(profiled, np.all(sides < floor, axis=-1), bad)
        init = None
        if name in init_params:
            init = np.asarray(init_params[name])
            bad = bad | (np.asarray(results.params[name]) == init)
        if not bad.any():
            continue
        if 'inistp_relative_to_init' in spec:
            step = init * spec['inistp_relative_to_init']
        else:
            step = np.full(len(err), float(
                config.model.inistp.get(name, spec.get('inistp', 0.0))))
        errors[name] = np.where(bad, step, err)
        flags[name] = np.where(bad | ~np.isfinite(errors[name]), 6, flags[name])
        log.line(logger, f"ERRMAX_BAD: {name} error replaced by INISTP "
                 f"for {int(bad.sum())} SNe")
    return dataclasses.replace(results, param_errors=errors, error_flags=flags)
