"""Bounded profile searches with one central repair and saved-point reuse.

The fitter selects either redshift or every free parameter before compilation.
Both modes use ten crossing trials, twenty nuisance steps per trial, one
validated twenty-step central repair, and six continuation trials after repair.
"""

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core.utils.newton_lm import newton_lm
from . import minos_search

CROSSING_STEPS = 10
REPAIR_STEPS = 20
CONTINUATION_STEPS = 6


def profile_errors(settings, model, *, strategy, photometry, z_weights, **inputs):
    """Profile selected free parameters on each event's final fit objective.

    Search coordinates and bounds are scaled as in the fitting rounds; returned
    distances and covariances use physical units. The result contains the five
    Hessian diagnostics, (distances, statuses), final parameters and chi-squared.
    A validated repair may replace the centre, but never the frozen objective.
    """
    from . import fit_kernels

    held = settings['round1_held_params'] if strategy == 'live' else settings['held_params']
    free = np.array([i for i, name in enumerate(settings['param_names'])
                     if name not in held], dtype=int)
    selected = fit_kernels._profile_indices(settings, held)
    original_params = inputs['params']
    events, total_params = original_params.shape
    scales = inputs['scales'][:, free]
    distances = jnp.zeros((events, total_params, 2), original_params.dtype).at[
        :, free].set(jnp.nan)
    statuses = jnp.zeros((events, total_params, 2), int).at[:, free].set(5)
    checked = fit_kernels._minimum_errors(
        settings, model, strategy=strategy, photometry=photometry, z_weights=z_weights, **inputs)
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
        loss = fit_kernels._error_loss(settings, model, strategy, photometry, z_weights, data)
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
            fixed = fit_kernels._error_loss(
                settings, model, strategy, photometry, z_weights, data, fixed_z=z)
            if z_only:
                return lambda r: fixed(p.at[free].set(r))
            return lambda r: jax.lax.cond(
                jnp.asarray(free)[index] == z_index,
                lambda q: fixed(p.at[free].set(q)), loss, r)

        return minos_search.profile_step(
            loss, center, data['lo'][free], data['hi'][free], cov, value,
            index, sign, state, prepare_loss=prepare if z_index is not None else None)

    def initial_state(index, sign):
        return jax.vmap(minos_search.profile_state,
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
        checked = fit_kernels._minimum_errors(
            settings, model, strategy=strategy, photometry=photometry, z_weights=z_weights,
            **dict(inputs, params=candidate))
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
        state = minos_search.profile_state(center, lo, hi, cov, value, index, sign)
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
            return minos_search.validate_endpoint(
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
