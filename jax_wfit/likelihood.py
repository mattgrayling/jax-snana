"""WFIT Hubble-diagram likelihood and analytic magnitude-offset profile.

Mirrors ``get_chi2_fit`` in SNANA ``src/wfit.c`` while evaluating the exact
quadratic form through dense or compact factors. No covariance terms are dropped.
"""

from typing import Mapping

import jax.numpy as jnp

from jax_snlc_core.cosmology import distance_modulus, flat_cpl_e2
from jax_snlc_core.utils import LIGHT_KM

from .config import WFITConfig
from .priors import bao_prior_chi2, cmb_prior_chi2, omega_m_prior_chi2


def hdibc_observed_mu(
    mu_model: jnp.ndarray,
    mu: jnp.ndarray,
    mu_second: jnp.ndarray,
    bias0: jnp.ndarray,
    bias1: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Interpolate two HDs as in SNANA ``get_chi2_fit``.

    SNANA explicitly limits the interpolation fraction to ``[-0.5, 1.5]`` to
    prevent excessive extrapolation beyond the two bias-correction cosmologies.
    """
    fraction = jnp.clip((mu_model - bias0) / (bias1 - bias0), -0.5, 1.5)
    return mu + fraction * (mu_second - mu), fraction


def resolved_parameters(
    params: Mapping[str, jnp.ndarray], config: WFITConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    omega_m = jnp.asarray(params["omega_m"])
    if config.model == "lcdm":
        return omega_m, jnp.asarray(-1.0), jnp.asarray(0.0)
    w0 = jnp.asarray(params["w0"])
    if config.model == "wcdm":
        return omega_m, w0, jnp.asarray(0.0)
    return omega_m, w0, jnp.asarray(params["wa"])


def offset_marginalised_chi2(
    residual: jnp.ndarray,
    precision: jnp.ndarray | dict[str, jnp.ndarray],
    valid: jnp.ndarray,
    intercept_sigma: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """The chi-square with the magnitude offset marginalised, and that offset.

    The Goliath et al. (2001) appendix as ``get_chi2_fit`` implements it
    (wfit.c:4386-4416): ``chi_hat - Bsum^2/Csum``, with ``Csum`` carrying the
    Gaussian H0 prior term, and ``Bsum/Csum`` the weighted mean residual the
    fit absorbs, which SNANA calls ``muresid_avg``.
    """
    ones = valid.astype(residual.dtype)
    if isinstance(precision, dict):
        # weighted_ones and sum_weights refer to all retained rows; stack_arrays
        # gives padded rows zero precision, so they make no contribution.
        b_term = jnp.sum(precision["weighted_ones"] * residual, axis=-1)
        c_term = precision["sum_weights"] + 1.0 / intercept_sigma**2
        intercept = b_term / c_term
        # Evaluate at the profiled offset instead of subtracting B^2/C from
        # a potentially large quadratic form (especially important in float32).
        centered = residual - intercept[..., None] * ones
        whitened = centered * precision["invstd"]
        coefficients = jnp.einsum("...nk,...n->...k", precision["mapping"], whitened)
        remainder = whitened - jnp.einsum(
            "...nk,...k->...n", precision["design"], coefficients)
        # A sum of squares avoids cancellation along coherent systematic
        # directions, where the two terms of the Woodbury quadratic nearly match.
        chi2 = (jnp.sum(remainder**2, axis=-1)
                + jnp.sum(coefficients**2, axis=-1) + (intercept / intercept_sigma)**2)
        return chi2, intercept
    if precision.ndim == residual.ndim:
        p_residual = precision * residual
        p_ones = precision * ones
    else:
        p_residual = jnp.einsum("...ij,...j->...i", precision, residual)
        p_ones = jnp.einsum("...ij,...j->...i", precision, ones)
    a_term = jnp.sum(residual * p_residual, axis=-1)
    b_term = jnp.sum(ones * p_residual, axis=-1)
    c_term = jnp.sum(ones * p_ones, axis=-1) + 1.0 / intercept_sigma**2
    return a_term - b_term * b_term / c_term, b_term / c_term


def evaluate_likelihood(
    params: Mapping[str, jnp.ndarray],
    redshift: jnp.ndarray,
    mu: jnp.ndarray,
    precision: jnp.ndarray | dict[str, jnp.ndarray],
    valid: jnp.ndarray,
    omega_m_prior_mean: jnp.ndarray,
    cmb_r: jnp.ndarray,
    config: WFITConfig,
    priors: Mapping[str, jnp.ndarray],
    hdibc: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
) -> dict[str, jnp.ndarray]:
    """Evaluate the WFIT chi-square for ``(N,)`` or ``(R, N)`` arrays."""
    omega_m, w0, wa = resolved_parameters(params, config)
    mu_model = distance_modulus(redshift, redshift, omega_m, 1.0 - omega_m,
                                LIGHT_KM / config.h0, w0, wa,
                                radiation=config.radiation)
    mu_observed = (
        mu if hdibc is None
        else hdibc_observed_mu(mu_model, mu, *hdibc)[0]
    )
    residual = jnp.where(valid, mu_observed - mu_model, 0.0)
    hd_chi2, intercept = offset_marginalised_chi2(
        residual, precision, valid, config.intercept_sigma)

    omega_m_chi2 = omega_m_prior_chi2(omega_m, omega_m_prior_mean, config)
    cmb_chi2 = cmb_prior_chi2(omega_m, w0, wa, cmb_r, config)
    bao_chi2 = bao_prior_chi2(omega_m, w0, wa, config, priors)
    finite_expansion = jnp.all(
        jnp.where(valid, flat_cpl_e2(
            redshift, omega_m, w0, wa, radiation=config.radiation) > 0.0, True),
        axis=-1,
    )
    hd_chi2 = jnp.where(
        finite_expansion & jnp.isfinite(hd_chi2), hd_chi2, jnp.inf,
    )
    return {
        "hd_chi2": hd_chi2,
        "omega_m_chi2": omega_m_chi2,
        "cmb_chi2": cmb_chi2,
        "bao_chi2": bao_chi2,
        "intercept": intercept,
        "total_chi2": hd_chi2 + omega_m_chi2 + cmb_chi2 + bao_chi2,
    }
