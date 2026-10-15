"""Exact BBC curvature using the local dependence on redshift-bin offsets."""

import jax
import jax.numpy as jnp


def loss_derivatives(p, loss, *, n_nuisance, nzbin, interpolate, with_value=False):
    """Gradient and full Hessian, without differentiating each M0 separately.

    The M0 block is diagonal for ordinary bins and tridiagonal when events
    interpolate between adjacent knots. Columns separated by three bins then
    have disjoint support. One direction per colour recovers every entry of
    that block. Nuisance parameters (and joint-fit sigint) retain individual
    directions; their columns also supply the cross terms by symmetry.

    This is exact sparse-Hessian recovery, not a curvature approximation.
    The loss must couple each event to only its own or an adjacent M0 bin,
    as m0_for and _m0_interp_weights do. Shared model/CC-prior quantities may
    depend on nuisance parameters but must not depend on M0.
    """
    if with_value:
        (value, gradient), linear = jax.linearize(jax.value_and_grad(loss), p)

        def curvature(direction):
            return linear(direction)[1]
    else:
        gradient, curvature = jax.linearize(jax.grad(loss), p)
        value = None

    n = p.shape[0]
    # A joint-fit sigint follows M0; otherwise this trailing slice is empty.
    global_indices = jnp.concatenate((jnp.arange(n_nuisance),
                                      jnp.arange(n_nuisance + nzbin, n)))
    ng = global_indices.shape[0]
    colours = min(3 if interpolate else 1, nzbin)
    bin_colour = jnp.arange(nzbin) % colours
    local = jnp.zeros((colours, n), dtype=p.dtype)
    local = local.at[:, n_nuisance:n_nuisance + nzbin].set(
        jax.nn.one_hot(bin_colour, colours, dtype=p.dtype).T)
    directions = jnp.concatenate((jnp.eye(n, dtype=p.dtype)[global_indices], local))
    columns = jax.vmap(curvature)(directions).T

    hessian = jnp.zeros((n, n), dtype=p.dtype)
    hessian = hessian.at[:, global_indices].set(columns[:, :ng])
    hessian = hessian.at[global_indices, n_nuisance:n_nuisance + nzbin].set(
        columns[n_nuisance:n_nuisance + nzbin, :ng].T)
    bins = jnp.arange(nzbin)
    band = jnp.abs(bins[:, None] - bins[None, :]) <= int(interpolate)
    block = columns[n_nuisance:n_nuisance + nzbin, ng:][:, bin_colour]
    hessian = hessian.at[n_nuisance:n_nuisance + nzbin,
                          n_nuisance:n_nuisance + nzbin].set(jnp.where(band, block, 0.))
    return gradient, hessian, value
