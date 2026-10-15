"""Segmented intrinsic-scatter estimation on SNANA's descending trial grid.

MAD crossings are monotone and use a bracketed search. Weighted RMS need not
be monotone, so it retains the reference linear scan. Both searches subtract
0.01 to construct trials, preserving the reference grid's rounding.
"""
from functools import partial

import jax
import jax.numpy as jnp


@jax.jit
def _median(values, cell, count):
    """Exact unweighted medians; cell == len(count) denotes an excluded row.

    Partition finite values until the upper middle value is isolated in every
    cell, then recover the lower middle value for even populations. Bounds
    always move to observed values, so ties need no tolerance or approximation.
    Non-finite inputs retain the existing sort semantics for error diagnostics.
    """
    ncell = len(count)
    if not len(values):
        return jnp.zeros(ncell, dtype=values.dtype)
    safe = jnp.minimum(cell, ncell - 1)

    def sorted_median():
        _, ordered = jax.lax.sort((cell, values), num_keys=2, is_stable=False)
        start = jnp.cumsum(count) - count
        left = start + jnp.maximum(count - 1, 0) // 2
        right = start + count // 2
        result = .5 * (ordered[jnp.minimum(left, len(values) - 1)] +
                       ordered[jnp.minimum(right, len(values) - 1)])
        return jnp.where(count > 0, result, 0.)

    def selected_median():
        lo = jax.ops.segment_min(values, cell, ncell)
        hi = jax.ops.segment_max(values, cell, ncell)
        rank = count // 2

        def partition(bounds):
            lo, hi = bounds
            # Halve separately to avoid overflow; keep the pivot below hi
            # even when the bounds are adjacent floating-point values.
            pivot = jnp.minimum(.5 * lo + .5 * hi,
                                jnp.nextafter(hi, -jnp.inf))
            below = values <= pivot[safe]
            number = jax.ops.segment_sum(below.astype(jnp.int32), cell, ncell)
            lower = jax.ops.segment_max(jnp.where(below, values, -jnp.inf), cell, ncell)
            upper = jax.ops.segment_min(jnp.where(below, jnp.inf, values), cell, ncell)
            active = (count > 0) & (lo < hi)
            return (jnp.where(active & (number <= rank), upper, lo),
                    jnp.where(active & (number > rank), lower, hi))

        upper, _ = jax.lax.while_loop(
            lambda bounds: jnp.any((count > 0) & (bounds[0] < bounds[1])),
            partition, (lo, hi))
        below = values < upper[safe]
        number = jax.ops.segment_sum(below.astype(jnp.int32), cell, ncell)
        lower = jax.ops.segment_max(jnp.where(below, values, -jnp.inf), cell, ncell)
        lower = jnp.where((count % 2 == 0) & (number == rank), lower, upper)
        return jnp.where(count > 0, .5 * (lower + upper), 0.)

    nonfinite = jnp.any((cell < ncell) & ~jnp.isfinite(values))
    return jax.lax.cond(nonfinite, sorted_median, selected_median)


@partial(jax.jit, static_argnames=("ncell", "use_mad", "minimum"))
def estimate(residual, covariance, weights, cell, valid, *, ncell,
             use_mad=False, minimum=1):
    """Return scatter, diagnostics and counts for each cell.

    Diagnostics are (invalid covariance, initial width below noise,
    reached scatter floor, unbracketed first trial). The caller decides which
    are fatal: covariance-map cells permit negative intrinsic scatter.
    """
    cell = jnp.where(valid, cell, ncell)
    safe = jnp.minimum(cell, ncell - 1)

    def sums(values):
        return jax.ops.segment_sum(values, cell, ncell)

    count = sums(jnp.ones_like(cell))
    used = count >= minimum
    # Match the reference normalization and avoid overflow for large weights.
    maximum = jax.ops.segment_max(jnp.where(valid, weights, 0.), cell, ncell)
    normalizer = jnp.where(maximum[safe] > 0, maximum[safe], 1.)
    weights = jnp.where(valid, weights / normalizer, 0.)
    residual = jnp.where(valid, residual, 0.)
    covariance = jnp.where(valid, covariance, 1.)
    total_weight = sums(weights)
    denominator = jnp.where(total_weight > 0, total_weight, 1.)
    mean = sums(weights * residual) / denominator
    mean_cov = sums(weights * covariance) / denominator

    def median(values):
        return _median(values, cell, count)

    def rms(values):
        average = sums(weights * values) / denominator
        variance = sums(weights * values * values) / denominator - average**2
        return jnp.sqrt(jnp.maximum(variance, 0.))

    width0 = 1.48 * median(jnp.abs(residual)) if use_mad else rms(residual)
    initial = jnp.sqrt(jnp.maximum(width0**2 - mean_cov, 0.)) + .5
    centred = residual - mean[safe]

    def pulls(scatter):
        added = scatter * jnp.abs(scatter)
        total = covariance + added[safe]
        total = jnp.where(added[safe] < 0, jnp.maximum(total, .0004), total)
        return centred / jnp.sqrt(total)

    if use_mad:
        def trial(index):
            return jax.lax.fori_loop(
                0, jnp.max(index),
                lambda i, s: jnp.where(i < index, s - .01, s), initial)

        def crosses(index):
            scatter = trial(index)
            absolute = jnp.abs(pulls(scatter))
            below = 1.48 * absolute <= 1.
            number_below = sums(below.astype(jnp.int32))
            lower = jax.ops.segment_max(
                jnp.where(below, absolute, -jnp.inf), cell, ncell)
            upper = jax.ops.segment_min(
                jnp.where(below, jnp.inf, absolute), cell, ncell)
            half = count // 2
            even = jnp.where(number_below < half, True,
                            jnp.where(number_below > half, False,
                                      1.48 * (.5 * (lower + upper)) > 1.))
            crossing = jnp.where(count % 2, number_below <= half, even)
            return (scatter < -.3) | crossing

        # Carry the crossing result so the condition and body do not both
        # reduce the full event arrays at the same upper bound.
        def expand(bounds):
            lo, hi, pending = bounds
            lo = jnp.where(pending, hi, lo)
            hi = jnp.where(pending, 2 * hi, hi)
            return lo, hi, used & ~crosses(hi)

        hi = jnp.full(ncell, 50)
        lo, hi, _ = jax.lax.while_loop(
            lambda bounds: jnp.any(bounds[2]), expand,
            (jnp.full(ncell, -1), hi, used & ~crosses(hi)))

        def bisect(bounds):
            lo, hi = bounds
            mid = (lo + hi) // 2
            crossing = crosses(mid)
            pending = used & (hi - lo > 1)
            return (jnp.where(pending & ~crossing, mid, lo),
                    jnp.where(pending & crossing, mid, hi))

        _, index = jax.lax.while_loop(
            lambda bounds: jnp.any(used & (bounds[1] - bounds[0] > 1)),
            bisect, (lo, hi))
        scatter = trial(index)
        previous = trial(jnp.maximum(index - 1, 0))
        width = 1.48 * median(jnp.abs(pulls(scatter)))
        previous_width = 1.48 * median(jnp.abs(pulls(previous)))
        fraction = (width - 1.) / jnp.where(width != previous_width,
                                           width - previous_width, 1.)
        answer = scatter + (previous - scatter) * fraction
        answer = jnp.where(index == 0, scatter, answer)
        floor = scatter < -.3
        bad = (index == 0) & (width > 1.) & (jnp.abs(width - 1.) >= .001)
    else:
        def step(state):
            scatter, previous_width, answer, active, bad, floor = state
            width = rms(pulls(scatter))
            crossing = width > 1.
            first = jnp.isinf(previous_width)
            fraction = (width - 1.) / jnp.where(width != previous_width,
                                               width - previous_width, 1.)
            root = jnp.where(first, scatter, scatter + .01 * fraction)
            at_floor = scatter < -.3
            finished = active & (crossing | at_floor)
            answer = jnp.where(finished, root, answer)
            bad |= active & ~at_floor & crossing & first & (jnp.abs(width - 1.) >= .001)
            floor |= active & at_floor
            return (scatter - .01, width, answer, active & ~finished, bad, floor)

        _, _, answer, _, bad, floor = jax.lax.while_loop(
            lambda state: jnp.any(state[3]), step,
            (initial, jnp.full(ncell, -jnp.inf), jnp.zeros(ncell), used,
             jnp.zeros(ncell, bool), jnp.zeros(ncell, bool)))
    answer = jnp.where(used, jnp.where(floor, -.3, answer), 0.)
    diagnostics = (sums((covariance <= 0).astype(jnp.int32)) > 0,
                   used & (width0 < jnp.sqrt(mean_cov)), used & floor, used & bad)
    return answer, diagnostics, count
