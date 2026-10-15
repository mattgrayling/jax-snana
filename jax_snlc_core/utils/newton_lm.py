"""Damped Newton iterations shared by the light-curve and distance fitters."""

import jax
import jax.numpy as jnp


def quadratic_decrease(eigenvalues, components, vectors, extent):
    """Upper bound on quadratic improvement within a coordinate box.

    Projecting the box into each eigenvector gives an enclosing box. Positive
    curvature has its usual Newton optimum, capped by that distance; flat or
    negative curvature attains its largest improvement at the enclosing edge.
    """
    radius = jnp.sum(jnp.where(vectors == 0, 0.0,
                              jnp.abs(vectors) * extent[:, None]), axis=0)
    slope = jnp.abs(components)
    positive = eigenvalues > 0
    distance = jnp.minimum(slope / jnp.where(positive, eigenvalues, 1), radius)
    convex = slope * distance - 0.5 * eigenvalues * distance ** 2
    linear = jnp.where(slope == 0, 0.0, slope * radius)
    concave = linear + jnp.where(eigenvalues == 0, 0.0,
                                -0.5 * eigenvalues * radius ** 2)
    return jnp.sum(jnp.where(positive, convex, concave))


def _grad_and_hess(p, loss_closed, with_value=False):
    """``(gradient, Hessian)`` from one forward-over-reverse pass.

    `jax.hessian` is `jacfwd(jacrev)`, so taking the gradient from the
    same pass costs nothing and leaves one loss primal per iteration
    rather than two.
    """
    def grad_with_itself(q):
        g = jax.grad(loss_closed)(q)
        return g, g
    if with_value:
        def grad_with_value(q):
            value, g = jax.value_and_grad(loss_closed)(q)
            return g, (g, value)
        H, (g, value) = jax.jacfwd(grad_with_value, has_aux=True)(p)
        return g, H, value
    H, g = jax.jacfwd(grad_with_itself, has_aux=True)(p)
    return g, H, None


def _newton_step(carry, bounds_lo, bounds_hi, loss_closed, *, refine=False,
                 tolerance=None, derivatives=None, derivative_fn=None):
    """One bounded Newton-LM step, with optional convergence diagnostics.

    Ordinary fits and profile nuisance minimisations use this same update.
    A supplied tolerance returns the checked quadratic decrease and current
    objective.
    Profile solves can supply a cached ``(gradient, curvature, value)`` and
    receive the trial gradient for updating that curvature. Without it, the
    step evaluates the exact Hessian as usual.
    """
    p, lam, f_val, escaping, radius = carry
    evaluate = _grad_and_hess if derivative_fn is None else derivative_fn
    g, H, value = (evaluate(p, loss_closed, tolerance is not None)
                   if derivatives is None else derivatives)
    if tolerance is not None:
        f_val = value
        carry = p, lam, f_val, escaping, radius

    diag_H = jnp.maximum(jnp.abs(jnp.diag(H)), 1e-6)
    H_reg = H + lam * jnp.diag(diag_H)

    if refine:
        # Hold a bound only while the gradient points outwards.
        blocked = (((p <= bounds_lo) & (g > 0))
                   | ((p >= bounds_hi) & (g < 0))
                   | (bounds_lo == bounds_hi))
        free = ~blocked
        H_reg = (jnp.where(free[:, None] & free[None, :], H_reg, 0.0)
                 + jnp.diag(blocked.astype(p.dtype)))
        rhs = jnp.where(free, g, 0.0)
    else:
        rhs = g

    d = -jnp.linalg.solve(H_reg, rhs)
    slope = jnp.dot(g, d)

    # Bad direction fallback: scaled gradient descent
    is_bad = (slope >= -1e-12) | jnp.any(jnp.isnan(d))
    abs_diag = jnp.abs(jnp.diag(H))
    scale_guess = 1.0 / (jnp.mean(abs_diag) + 1e-8)
    if refine:
        fallback = -rhs * scale_guess
        d = jnp.where(is_bad, fallback, d)

        # A small gradient can still sit on negative curvature.
        inv_scale = jax.lax.rsqrt(diag_H)
        scaled_hessian = (inv_scale[:, None] * H) * inv_scale[None, :]
        scaled_hessian = (jnp.where(free[:, None] & free[None, :], scaled_hessian, 0.0)
                          + jnp.diag(blocked.astype(p.dtype)))
        eigenvalues, eigenvectors = jnp.linalg.eigh(scaled_hessian)
        scaled_gradient = rhs * inv_scale
        stationary = jnp.linalg.norm(scaled_gradient) < 0.01
        curvature_threshold = -0.001 if tolerance is None else 0.0
        use_curvature = (stationary | escaping) & (eigenvalues[0] < curvature_threshold)
        direction = eigenvectors[:, 0] * inv_scale
        direction = jnp.where(jnp.dot(g, direction) > 0, -direction, direction)
        # A clipped curvature step may need the opposite sign.
        curvature_step = radius * direction
        forward = jnp.clip(p + curvature_step, bounds_lo, bounds_hi) - p
        backward = jnp.clip(p - curvature_step, bounds_lo, bounds_hi) - p
        clipped = jnp.any((p + curvature_step < bounds_lo)
                          | (p + curvature_step > bounds_hi))
        forward_model = g @ forward + 0.5 * forward @ H @ forward
        backward_model = g @ backward + 0.5 * backward @ H @ backward
        curvature_step = jnp.where(clipped & (backward_model < forward_model),
                                   -curvature_step, curvature_step)
        d = jnp.where(use_curvature, curvature_step, d)
    else:
        d = jnp.where(is_bad, -g * scale_guess, d)
        use_curvature = jnp.bool_(False)

    trial_p = jnp.clip(p + d, bounds_lo, bounds_hi)
    if derivatives is None:
        f_trial = loss_closed(trial_p)
    else:
        f_trial, trial_gradient = jax.value_and_grad(loss_closed)(trial_p)

    s = trial_p - p

    # Gain ratio using actual (clipped) step
    actual = f_val - f_trial
    predicted = -(jnp.dot(g, s) + 0.5 * s @ H @ s)
    positive_prediction = predicted > 0
    predicted = jnp.maximum(predicted, 1e-30)
    rho = actual / predicted

    # Accept if reasonable gain ratio (0.01 margin for float32)
    accept = rho > 0.01
    p_out = jnp.where(accept, trial_p, p)
    f_out = jnp.where(accept, f_trial, f_val)

    lam_out = jnp.where(rho > 0.75, lam * 0.3,
                        jnp.where(rho > 0.25, lam, lam * 3.0))
    if refine:
        # A downhill clipped step can still disagree with the quadratic model.
        lam_out = jnp.where(accept & ~positive_prediction, lam, lam_out)
        lam_out = jnp.clip(lam_out, 1e-8, 1e8)
        # Skip repeated fallbacks to the first predicted descent Newton step.
        # 35 rungs cover the permitted damping range, 1e-8 to 1e8.
        levels = jnp.minimum(lam_out * 3.0 ** jnp.arange(35), 1e8)
        coefficients = eigenvectors.T @ scaled_gradient
        if tolerance is not None:
            extent = jnp.where(blocked, 0.0,
                jnp.maximum(p - bounds_lo, bounds_hi - p) / inv_scale)
            remaining = quadratic_decrease(
                eigenvalues, coefficients, eigenvectors, extent)
            converged = remaining <= tolerance
        slopes = -jnp.sum(coefficients[None, :] ** 2
                          / (eigenvalues[None, :] + levels[:, None]), axis=1)
        usable = jnp.isfinite(slopes) & (slopes < -1e-12)
        first = jnp.where(jnp.any(usable), jnp.argmax(usable), 34)
        lam_out = jnp.where(is_bad & ~accept & ~use_curvature, levels[first], lam_out)
        # After rejecting a Newton trial, require the next predicted step
        # to be shorter, rather than merely increasing lambda near zero.
        spectral_steps = -coefficients[None, :] / (eigenvalues[None, :] + levels[:, None])
        lengths_squared = jnp.sum(spectral_steps ** 2, axis=1)
        current_length_squared = jnp.sum((d / inv_scale) ** 2)
        contracted = usable & (lengths_squared <= 0.25 * current_length_squared)
        # Preserve the original fallback when increasing damping first
        # makes Newton non-descent; do not jump over that transition.
        transition = jnp.isfinite(slopes) & (slopes >= -1e-12)
        changed = contracted | transition
        first_shorter = jnp.where(jnp.any(changed), jnp.argmax(changed), 34)
        lam_out = jnp.where(~is_bad & ~accept & ~use_curvature,
                            levels[first_shorter], lam_out)

        radius_out = jnp.where(
            use_curvature,
            jnp.clip(radius * jnp.where(rho > 0.75, 2.0,
                                       jnp.where(rho > 0.25, 1.0, 0.5)), 1e-6, 16.0),
            1.0)
    else:
        lam_out = jnp.clip(lam_out, 1e-8, 1e8)
        radius_out = 1.0

    candidate = (p_out, lam_out, f_out, use_curvature, radius_out)
    if tolerance is not None:
        candidate = jax.tree.map(
            lambda old, new: jnp.where(converged, old, new), carry, candidate)
        if derivatives is not None:
            return candidate, (remaining, H, g, f_val, trial_gradient)
        return candidate, (remaining, H, g, f_val)
    return candidate, None


def newton_lm(init_p, bounds_lo, bounds_hi, loss_closed, maxiter, *, refine=False,
              tolerance=None, derivative_fn=None):
    """Run bounded Newton-LM steps and return parameters, loss and max |gradient|.

    ``refine`` enables active-bound solves, saddle escape and damping recovery.
    The light-curve fitter uses it only after freezing the model weights;
    live-weight rounds retain the original steps.
    ``derivative_fn`` optionally supplies the same exact (gradient, Hessian,
    value) contract as _grad_and_hess; step and acceptance rules are unchanged.
    Both modes evaluate one trial per iteration within ``maxiter``.
    With ``refine=True``, ``tolerance`` can stop once the full-Hessian
    bound on remaining quadratic loss decrease is small. The usual fixed-budget
    path is unchanged when it is omitted.
    """

    if init_p.size == 0:
        return init_p, loss_closed(init_p), jnp.zeros((), dtype=init_p.dtype)

    def newton_step(carry, _):
        candidate, diagnostics = _newton_step(
            carry, bounds_lo, bounds_hi, loss_closed, refine=refine,
            tolerance=tolerance, derivative_fn=derivative_fn)
        return candidate, (diagnostics[0] <= tolerance if tolerance is not None else None)

    # the start point obeys the bounds as every trial does
    init_p = jnp.clip(init_p, bounds_lo, bounds_hi)
    f_init = loss_closed(init_p)
    initial = (init_p, 1e-3, f_init, False, 1.0)
    if tolerance is None:
        (p_final, _, f_final, _, _), _ = jax.lax.scan(
            newton_step, initial, None, length=maxiter)
    else:
        if not refine:
            raise ValueError('Convergence stopping requires refine=True')

        def advance(state):
            iteration, done, carry = state
            candidate, converged = newton_step(carry, None)
            kept = jax.tree.map(
                lambda old, new: jnp.where(done, old, new), carry, candidate)
            return iteration + 1, done | converged, kept

        _, _, (p_final, _, f_final, _, _) = jax.lax.while_loop(
            lambda state: (state[0] < maxiter) & ~state[1],
            advance, (0, jnp.bool_(False), initial))
    g_final = jax.grad(loss_closed)(p_final)
    # every parameter pinned leaves nothing to solve for
    grad_norm = jnp.max(jnp.abs(g_final), initial=0.0)
    return p_final, f_final, grad_norm
