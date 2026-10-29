"""
SNANA-equivalent parameter sampling distributions using inverse CDF methods.

All sampling functions are pure JAX, vmap-compatible, and use inverse CDF
methods instead of rejection sampling for O(1) per-sample performance.

Instead of SNANA's rejection-based truncation (re-draw if out of bounds),
the code computes CDF(lo) and CDF(hi) analytically, samples
u ~ U(CDF(lo), CDF(hi)), and inverts. This produces samples exactly within
[lo, hi] by construction with correct truncated density: no clipping needed.
"""

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm as jax_norm

from jax_snlc_core.logging import SNANAError

# SNANA's epsilon for deciding a width is zero (sntools_genExpHalfGauss.c:226)
_EXP_HALFGAUSS_EPS = 1.0e-14

BIGAUSSNORMCON = 1.25331413732


@dataclass
class GenGaussAsym:
    """Mirrors SNANA's GENGAUSS_ASYM_DEF for asymmetric Gaussian distributions.

    Supports: bifurcated (split-normal) Gaussian, flat peak regions, dual peaks,
    skew-modified sigma, bounded truncation, grid snapping, and exponential reweighting.
    """
    peak: float = 0.0
    sigma: Tuple[float, float] = (0.0, 0.0)
    range: Tuple[float, float] = (0.0, 0.0)
    peakrange: Tuple[float, float] = (0.0, 0.0)  # flat plateau boundaries
    skew: Tuple[float, float] = (0.0, 0.0)
    prob2: float = 0.0
    peak2: float = 0.0
    sigma2: Tuple[float, float] = (0.0, 0.0)
    ngrid: int = 0
    prob_expon_rewgt: float = 1.0


def validate_gaussian(distribution):
    """Validate the Gaussian population record shared by all input formats."""
    if not np.isfinite(distribution.range).all() or distribution.range[0] > distribution.range[1]:
        raise SNANAError(f'Invalid population range {distribution.range}: non-finite or inverted')
    widths = (*distribution.sigma, *distribution.sigma2)
    if any(not np.isfinite(v) or v < 0 for v in widths):
        raise SNANAError('Population widths must be finite and non-negative')
    if not 0 <= distribution.prob2 <= 1:
        raise SNANAError('The second population component probability must lie in [0, 1]')
    if not 0 <= distribution.ngrid <= 100:
        raise SNANAError('Gaussian population ngrid must be 0-100')


@dataclass
class GenExpHalfGauss:
    """Mirrors SNANA's GEN_EXP_HALFGAUSS_DEF for exponential + half-Gaussian hybrids.

    Used for AV/EBV host extinction distributions.
    """
    exp_tau: float = 0.0
    peak: float = 0.0
    sigma: float = 0.0
    ratio: float = 0.0  # Gauss(0)/Expon(0)
    range: Tuple[float, float] = (0.0, 0.0)
    prob_expon_rewgt: float = 1.0


def flat_distribution(bounds, *, dust=False):
    """Use the existing SNANA flat-population widths and sampling kernels."""
    lo, hi = bounds
    if dust:
        return GenExpHalfGauss(exp_tau=1.e6, range=(lo, hi))
    return GenGaussAsym(peak=0.5 * (lo + hi), sigma=(1.e6, 1.e6), range=(lo, hi))


def bifurcated_cdf(x, peak, sigma_lo, sigma_hi):
    """CDF of the bifurcated (split-normal) Gaussian.

    F(x) = 2*w_lo * Phi((x - peak) / sigma_lo)          for x < peak
    F(x) = 1 - 2*w_hi * (1 - Phi((x - peak) / sigma_hi)) for x >= peak

    where w_lo = sigma_lo / (sigma_lo + sigma_hi),
          w_hi = sigma_hi / (sigma_lo + sigma_hi).
    """
    sigma_lo = jnp.maximum(sigma_lo, 1e-30)
    sigma_hi = jnp.maximum(sigma_hi, 1e-30)
    w_lo = sigma_lo / (sigma_lo + sigma_hi)
    w_hi = sigma_hi / (sigma_lo + sigma_hi)
    cdf_left = 2.0 * w_lo * jax_norm.cdf((x - peak) / sigma_lo)
    cdf_right = 1.0 - 2.0 * w_hi * (1.0 - jax_norm.cdf((x - peak) / sigma_hi))
    return jnp.where(x < peak, cdf_left, cdf_right)


def bifurcated_ppf(u, peak, sigma_lo, sigma_hi):
    """Inverse CDF (percent-point function) of bifurcated Gaussian.

    if u < p_left:
        x = peak + sigma_lo * Phi_inv(u / (2 * p_left))
    else:
        x = peak + sigma_hi * Phi_inv(1 - (1 - u) / (2 * (1 - p_left)))
    """
    sigma_lo = jnp.maximum(sigma_lo, 1e-30)
    sigma_hi = jnp.maximum(sigma_hi, 1e-30)
    p_left = sigma_lo / (sigma_lo + sigma_hi)
    x_left = peak + sigma_lo * jax_norm.ppf(u / (2.0 * p_left))
    x_right = peak + sigma_hi * jax_norm.ppf(1.0 - (1.0 - u) / (2.0 * (1.0 - p_left)))
    return jnp.where(u < p_left, x_left, x_right)


def sample_truncated_gaussian(key, peak, sigma_lo, sigma_hi, range_lo, range_hi):
    """Sample from truncated bifurcated Gaussian via inverse CDF.

    Computes CDF at the range bounds, samples u ~ U(cdf_lo, cdf_hi),
    then applies the inverse CDF. Result is exactly within [range_lo, range_hi].
    """
    cdf_lo = bifurcated_cdf(range_lo, peak, sigma_lo, sigma_hi)
    cdf_hi = bifurcated_cdf(range_hi, peak, sigma_lo, sigma_hi)
    u = jax.random.uniform(key, minval=cdf_lo, maxval=cdf_hi)
    return bifurcated_ppf(u, peak, sigma_lo, sigma_hi)



def _sample_peakrange(key, peak_lo, peak_hi, sigma_lo, sigma_hi,
                      range_lo, range_hi):
    """Sample from flat-plateau + half-Gaussian-tail distribution.

    Distribution is flat between [peak_lo, peak_hi] with half-Gaussian tails
    below/above, all truncated to [range_lo, range_hi].

    Three-component mixture:
    - Component 0: half-normal below peak_lo, intersected with the range
    - Component 1: uniform in [max(range_lo, peak_lo), min(range_hi, peak_hi)]
    - Component 2: half-normal above peak_hi, intersected with the range
    """
    key1, key2 = jax.random.split(key)

    flat_lo = jnp.maximum(range_lo, peak_lo)
    flat_hi = jnp.minimum(range_hi, peak_hi)
    flat_width = jnp.maximum(flat_hi - flat_lo, 0.0)
    scale_lo = jnp.where(sigma_lo > 0., sigma_lo, 1.)
    scale_hi = jnp.where(sigma_hi > 0., sigma_hi, 1.)

    left_tail_exists = (range_lo < peak_lo) & (sigma_lo > 0.0)
    left_cdf_min = jax_norm.cdf((peak_lo - jnp.minimum(range_hi, peak_lo)) / scale_lo)
    left_cdf_max = jax_norm.cdf((peak_lo - range_lo) / scale_lo)
    left_erf = left_cdf_max - left_cdf_min
    weight_lo = jnp.where(left_tail_exists,
                          sigma_lo * BIGAUSSNORMCON * 2.0 * left_erf,
                          0.0)

    right_tail_exists = (range_hi > peak_hi) & (sigma_hi > 0.0)
    right_cdf_min = jax_norm.cdf((jnp.maximum(range_lo, peak_hi) - peak_hi) / scale_hi)
    right_cdf_max = jax_norm.cdf((range_hi - peak_hi) / scale_hi)
    right_erf = right_cdf_max - right_cdf_min
    weight_hi = jnp.where(right_tail_exists,
                          sigma_hi * BIGAUSSNORMCON * 2.0 * right_erf,
                          0.0)

    weight_flat = flat_width
    total = weight_lo + weight_flat + weight_hi

    u_comp = jax.random.uniform(key1)
    cum_lo = weight_lo / total
    cum_flat = (weight_lo + weight_flat) / total

    u_left = jax.random.uniform(key2, minval=left_cdf_min, maxval=left_cdf_max)
    val_left = peak_lo - sigma_lo * jax_norm.ppf(u_left)

    val_flat = jax.random.uniform(key2, minval=flat_lo, maxval=jnp.maximum(flat_lo + 1e-30, flat_hi))

    u_right = jax.random.uniform(key2, minval=right_cdf_min, maxval=right_cdf_max)
    val_right = peak_hi + sigma_hi * jax_norm.ppf(u_right)

    val = jnp.where(u_comp < cum_lo, val_left,
                    jnp.where(u_comp < cum_flat, val_flat, val_right))
    return val


# getRan_GENGAUSS_ASYM restart loop as mass-reweighted mixture (sntools_genGauss_asym.c:331-454)
_SKEW_EPS = 1.0e-9        # DO_SKEWSIGMA threshold (sntools_genGauss_asym.c:389)
_PEAKRANGE_EPS = 1.0e-12  # DO_PEAKRANGE threshold (:390)
_PROB2_EPS = 1.0e-7       # PROB2 activation gate (:338)


def _component_plan(sig_lo, sig_hi, skew_lo, skew_hi, pr_lo, pr_hi,
                    range_lo, range_hi, ngrid, sigma_shifted=False,
                    skew_shifted=False):
    """Static branch/grid plan for one mixture component.

    Mirrors the per-draw setup of getRan_GENGAUSS_ASYM (:372-404): the grid
    decision and range extension use the selected component's sigmas
    (DO_GRID = NGRID>=2 && hi>lo && siglo>0 && sighi>0), and the flat
    threshold is sigmax = 10 * (extended range).

    Branch order matches SNANA: delta, flat, skew, peakrange, bifurcated.
    With ZVARIATION-shifted sigmas the flat-vs-bifurcated choice is per
    event ('dyn_flat'); build-time validation guarantees shifted sigmas keep
    the base sign, so the grid decision stays static.
    """
    do_grid = (ngrid >= 2 and range_hi > range_lo
               and sig_lo > 0.0 and sig_hi > 0.0)
    if do_grid:
        _, _, lo, hi = _grid_params(range_lo, range_hi, ngrid)
    else:
        lo, hi = range_lo, range_hi
    width = hi - lo

    skew_on = (abs(skew_lo) > _SKEW_EPS or abs(skew_hi) > _SKEW_EPS
               or skew_shifted)
    peakrange_on = (pr_hi - pr_lo) > _PEAKRANGE_EPS

    if width < 1e-30:
        branch = 'delta'
    elif not sigma_shifted and sig_lo > 10.0 * width and sig_hi > 10.0 * width:
        branch = 'flat'
    elif skew_on:
        branch = 'skew'
    elif peakrange_on:
        branch = 'peakrange'
    elif sigma_shifted:
        branch = 'dyn_flat'
    else:
        branch = 'bifurcated'
    return branch, lo, hi, do_grid


def _peakrange_mass(sig_lo, sig_hi, pr_lo, pr_hi, lo, hi):
    """Accepted mass of the plateau + half-Gaussian-tail law on [lo, hi].

    The full (untruncated) law drawn by getRan_GaussAsym(siglo, sighi,
    interval) has mass sig_lo*sqrt(pi/2) + interval + sig_hi*sqrt(pi/2)."""
    interval = pr_hi - pr_lo
    full = sig_lo * BIGAUSSNORMCON + interval + sig_hi * BIGAUSSNORMCON
    scale_lo = jnp.where(sig_lo > 0., sig_lo, 1.)
    scale_hi = jnp.where(sig_hi > 0., sig_hi, 1.)
    flat_width = jnp.maximum(jnp.minimum(hi, pr_hi) - jnp.maximum(lo, pr_lo),
                             0.0)
    w_lo = jnp.where(
        (lo < pr_lo) & (sig_lo > 0.0),
        sig_lo * BIGAUSSNORMCON * 2.0
        * (jax_norm.cdf((pr_lo - lo) / scale_lo)
           - jax_norm.cdf((pr_lo - jnp.minimum(hi, pr_lo)) / scale_lo)),
        0.0)
    w_hi = jnp.where(
        (hi > pr_hi) & (sig_hi > 0.0),
        sig_hi * BIGAUSSNORMCON * 2.0
        * (jax_norm.cdf((hi - pr_hi) / scale_hi)
           - jax_norm.cdf((jnp.maximum(lo, pr_hi) - pr_hi) / scale_hi)),
        0.0)
    return (w_lo + flat_width + w_hi) / full


def _component_mass(branch, peak, sig_lo, sig_hi, pr_lo, pr_hi, lo, hi):
    """Accepted mass (probability of no outer restart) for one component.

    The skew branch has mass 1: getRan_skewGauss rejects proposals outside
    [xmin, xmax] inside its own loop (sntools.c:7591-7593), so the outer
    range check never fires for it. The flat and delta branches draw within
    range by construction. The bifurcated and peakrange branches draw
    untruncated values that the outer loop rejects, so their mass is the
    truncated fraction. Returns a jnp scalar (concrete for unshifted
    parameters)."""
    if branch in ('delta', 'flat', 'skew'):
        return jnp.float32(1.0)
    if branch == 'dyn_flat':
        width = hi - lo
        flat = (sig_lo > 10.0 * width) & (sig_hi > 10.0 * width)
        m = (bifurcated_cdf(hi, peak, sig_lo, sig_hi)
             - bifurcated_cdf(lo, peak, sig_lo, sig_hi))
        return jnp.where(flat, 1.0, m)
    if branch == 'peakrange':
        return _peakrange_mass(sig_lo, sig_hi, pr_lo, pr_hi, lo, hi)
    return (bifurcated_cdf(hi, peak, sig_lo, sig_hi)
            - bifurcated_cdf(lo, peak, sig_lo, sig_hi))


def _component_value(key, branch, peak, sig_lo, sig_hi, skew_lo, skew_hi,
                     pr_lo, pr_hi, lo, hi, range_lo, range_hi, ngrid,
                     do_grid, traced_shifts=False):
    """Draw one value from a component's truncated law, grid-snapped when
    the component's DO_GRID is active."""
    if branch == 'delta':
        val = jnp.float32(lo)
    elif branch == 'flat':
        val = jax.random.uniform(key, minval=lo, maxval=hi)
    elif branch == 'dyn_flat':
        # shifted sigmas: flat-vs-bifurcated decided per event (:590)
        width = hi - lo
        flat = (sig_lo > 10.0 * width) & (sig_hi > 10.0 * width)
        val_flat = jax.random.uniform(key, minval=lo, maxval=hi)
        val_gauss = sample_truncated_gaussian(key, peak, sig_lo, sig_hi,
                                              lo, hi)
        val = jnp.where(flat, val_flat, val_gauss)
    elif branch == 'skew':
        if traced_shifts:
            x_grid, cdf = _skew_cdf_grids_traced(
                peak, sig_lo, sig_hi, skew_lo, skew_hi, lo, hi)
            u = jax.random.uniform(key)
            val = jnp.interp(u, cdf, x_grid)
        else:
            skew_cdf = precompute_skew_cdf(
                peak, sig_lo, sig_hi, skew_lo, skew_hi, lo, hi)
            val = _sample_from_precomputed_cdf(key, skew_cdf)
    elif branch == 'peakrange':
        val = _sample_peakrange(key, pr_lo, pr_hi, sig_lo, sig_hi, lo, hi)
    else:
        val = sample_truncated_gaussian(key, peak, sig_lo, sig_hi, lo, hi)

    if do_grid:
        val = _apply_grid_snapping(val, range_lo, range_hi, ngrid)
    return val



def precompute_skew_cdf(peak, sigma_lo, sigma_hi, skew_lo, skew_hi,
                        range_lo, range_hi, n_grid=1000):
    """Pre-tabulate the skew-modified Gaussian CDF as ``(x_grid, cdf)``.

    SNANA's "hack-skew": ``sigma(x) = sigma_base + skew * |x - peak|``.
    No closed-form CDF, so it is tabulated on a grid by cumulative trapezoid.
    """
    x = np.linspace(range_lo, range_hi, n_grid)
    x_rel = x - peak
    sig_base = np.where(x_rel < 0, sigma_lo, sigma_hi)
    skew_val = np.where(x_rel < 0, skew_lo, skew_hi)
    sig = sig_base + skew_val * np.abs(x_rel)
    sig = np.maximum(sig, 1e-30)
    density = np.exp(-0.5 * (x_rel / sig) ** 2)

    dx = np.diff(x)
    cdf = np.cumsum((density[:-1] + density[1:]) / 2.0 * dx)
    cdf = np.concatenate([[0.0], cdf])
    cdf = cdf / cdf[-1]

    return jnp.array(x), jnp.array(cdf)


def _sample_from_precomputed_cdf(key, precomputed):
    """Sample via inverse CDF interpolation from a ``(x_grid, cdf)`` pair."""
    x_grid, cdf = precomputed
    u = jax.random.uniform(key)
    return jnp.interp(u, cdf, x_grid)



def _sample_truncated_exponential(key, tau, range_lo, range_hi):
    """Sample from truncated exponential via inverse CDF.

    CDF: F(x) = 1 - exp(-x/tau), truncated to [range_lo, range_hi].
    Inverse: x = -tau * ln(exp(-range_lo/tau) - u * (exp(-range_lo/tau) - exp(-range_hi/tau)))
    """
    exp_lo = jnp.exp(-range_lo / tau)
    exp_hi = jnp.exp(-range_hi / tau)
    u = jax.random.uniform(key)
    return -tau * jnp.log(exp_lo - u * (exp_lo - exp_hi))


def _sample_truncated_halfnormal(key, sigma, range_lo, range_hi):
    """Sample from truncated half-normal |N(0, sigma)| in [range_lo, range_hi].

    For half-normal, CDF(x) = erf(x / (sigma*sqrt(2))) for x >= 0.
    Using normal CDF: CDF_half(x) = 2*(Phi(x/sigma) - 0.5) for x >= 0.
    """
    cdf_lo = jax_norm.cdf(range_lo / sigma)
    cdf_hi = jax_norm.cdf(range_hi / sigma)
    cdf_lo_mapped = jnp.maximum(cdf_lo, 0.5)
    u = jax.random.uniform(key, minval=cdf_lo_mapped, maxval=cdf_hi)
    return sigma * jax_norm.ppf(u)


def _sample_truncated_normal(key, mean, sigma, range_lo, range_hi):
    """Sample from truncated normal N(mean, sigma) in [range_lo, range_hi]."""
    cdf_lo = jax_norm.cdf((range_lo - mean) / sigma)
    cdf_hi = jax_norm.cdf((range_hi - mean) / sigma)
    u = jax.random.uniform(key, minval=cdf_lo, maxval=cdf_hi)
    return mean + sigma * jax_norm.ppf(u)


def sample_exp_halfgauss(key, dist, shifts=None):
    """Sample from exponential + half-Gaussian hybrid distribution.

    Matches SNANA's sntools_genExpHalfGauss.c logic.

    Uses Python if/else dispatch since dist attributes are static floats,
    so only the active branch is traced/compiled by JAX. With ZVARIATION
    ``shifts`` (traced scalars, EXPHALFGAUSS_ZSHIFT_FIELDS order) every
    branch condition can depend on a shifted parameter, so all branches
    are evaluated and selected per SN with jnp.where.

    ``dist`` is a GenExpHalfGauss instance; ``shifts`` is an optional traced
    per-field additive shift. Returns a sampled value within dist.range.
    """
    if shifts is not None:
        return _sample_exp_halfgauss_shifted(key, dist, shifts)
    range_lo, range_hi = dist.range
    tau = dist.exp_tau
    sigma = dist.sigma
    peak = dist.peak
    ratio = dist.ratio

    # PROB_EXPON_REWGT: tau /= rewgt, sigma /= sqrt(rewgt)
    if dist.prob_expon_rewgt != 1.0:
        tau = tau / dist.prob_expon_rewgt
        sigma = sigma / jnp.sqrt(dist.prob_expon_rewgt)

    if (range_hi - range_lo) < 1e-30:
        return jnp.float32(range_lo)

    # sntools_genExpHalfGauss.c:235-241
    if (tau < _EXP_HALFGAUSS_EPS) and (sigma < _EXP_HALFGAUSS_EPS):
        raise SNANAError(
            f"GENRANGE = {range_lo} to {range_hi}, but EXP_TAU=0 and SIG=0; "
            f"EXP_TAU and/or SIG must not be zero.",
            function_name="sample_exp_halfgauss")

    if (tau > 50.0) and (ratio == 0.0):
        return jax.random.uniform(key, minval=range_lo, maxval=range_hi)

    # pure branches select on the two widths alone (:259-260)
    if (tau > _EXP_HALFGAUSS_EPS) and (sigma < _EXP_HALFGAUSS_EPS):
        return _sample_truncated_exponential(key, tau, range_lo, range_hi)

    if (sigma > _EXP_HALFGAUSS_EPS) and (tau < _EXP_HALFGAUSS_EPS):
        # peak <= 0.0001 (including the -9 init) draws the half-normal
        if peak > 1e-4:
            return _sample_truncated_normal(key, peak, sigma, range_lo, range_hi)
        else:
            return _sample_truncated_halfnormal(key, sigma, range_lo, range_hi)

    if not (ratio > _EXP_HALFGAUSS_EPS):
        # :337-340
        raise SNANAError(
            f"Could not determine the value from exponential or Gaussian: "
            f"tau={tau}, sig={sigma}, ratio={ratio}.",
            function_name="sample_exp_halfgauss")

    key1, key2 = jax.random.split(key)

    w_exp = tau * (jnp.exp(-range_lo / tau) - jnp.exp(-range_hi / tau))
    w_gauss = 0.5 * ratio * jnp.sqrt(2.0 * jnp.pi) * sigma

    u_comp = jax.random.uniform(key1)
    use_exp = u_comp < (w_exp / (w_exp + w_gauss))

    val_exp = _sample_truncated_exponential(key2, tau, range_lo, range_hi)
    if peak > 1e-4:
        val_gauss = _sample_truncated_normal(key2, peak, sigma, range_lo, range_hi)
    else:
        val_gauss = _sample_truncated_halfnormal(key2, sigma, range_lo, range_hi)

    return jnp.where(use_exp, val_exp, val_gauss)



def _grid_params(range_lo, range_hi, ngrid):
    """Compute grid parameters matching SNANA's NGRID setup.

    Returns (gridsize, grid0, extended_lo, extended_hi).
    SNANA extends the sampling range by gridsize/2 on each side.
    """
    gridsize = (range_hi - range_lo) / max(ngrid - 1, 1)
    grid0 = range_lo
    extended_lo = range_lo - gridsize / 2.0
    extended_hi = range_hi + gridsize / 2.0
    return gridsize, grid0, extended_lo, extended_hi


def _apply_grid_snapping(val, range_lo, range_hi, ngrid):
    """Quantise sampled value to grid, matching SNANA's NGRID behaviour.

    SNANA uses (int) truncation from the extended lower bound.
    """
    gridsize, grid0, extended_lo, _ = _grid_params(range_lo, range_hi, ngrid)
    ibin = jnp.floor((val - extended_lo) / gridsize).astype(jnp.int32)
    ibin = jnp.clip(ibin, 0, ngrid - 1)
    return grid0 + ibin * gridsize



def sample_gengauss_asym(key, g, shifts=None, active=frozenset()):
    """Sample from a GenGaussAsym distribution.

    Composes SNANA's mixture semantics (getRan_GENGAUSS_ASYM,
    sntools_genGauss_asym.c:331-454): the optional secondary peak is
    selected first, the chosen component's shape (flat, skew, peakrange,
    or bifurcated Gaussian) is applied, and the out-of-range restart loop
    becomes accepted-mass reweighting of the mixture plus truncated draws.
    The grid decision, range extension, and the 10x-range flat threshold
    follow the selected component's sigmas (:372-404), and grid snapping
    applies only when that component's DO_GRID is active (:439-445).

    ZVARIATION support: ``shifts`` (traced scalars in
    GENGAUSS_ZSHIFT_FIELDS order) add to the raw parameters before the
    PROB_EXPON_REWGT division, matching SNANA's get_zvariation_GENGAUSS;
    ``active`` statically names the shifted fields so branch dispatch stays
    at the Python level. With shifts=None the added terms are concrete 0.0
    and the traced graph is unchanged.

    ``g`` is a GenGaussAsym instance; ``shifts`` an optional traced
    per-field additive shift; ``active`` a static collection of shifted
    field names. Returns a sampled value within g.range.
    """
    if g.ngrid < 0 or g.ngrid > 100:
        raise SNANAError(f"Crazy NGRID={g.ngrid}",
                         function_name="sample_gengauss_asym")

    d = (dict.fromkeys(GENGAUSS_ZSHIFT_FIELDS, 0.0) if shifts is None
         else dict(zip(GENGAUSS_ZSHIFT_FIELDS, shifts)))
    traced = shifts is not None
    active = frozenset(active)
    range_lo, range_hi = g.range

    sqrt_rewgt = (float(g.prob_expon_rewgt) ** 0.5
                  if g.prob_expon_rewgt != 1.0 else 1.0)

    # the rewgt divides sigmas and skews (:555-558)
    peak = g.peak + d['peak']
    sig_lo = (g.sigma[0] + d['sigma_lo']) / sqrt_rewgt
    sig_hi = (g.sigma[1] + d['sigma_hi']) / sqrt_rewgt
    skew_lo = (g.skew[0] + d['skew_lo']) / sqrt_rewgt
    skew_hi = (g.skew[1] + d['skew_hi']) / sqrt_rewgt
    pr_lo, pr_hi = g.peakrange
    prob2 = g.prob2 + d['prob2']
    peak2 = g.peak2 + d['peak2']
    sig2_lo = (g.sigma2[0] + d['sigma2_lo']) / sqrt_rewgt
    sig2_hi = (g.sigma2[1] + d['sigma2_hi']) / sqrt_rewgt

    # unshifted copies drive the static branch plans
    b1 = (g.peak, g.sigma[0] / sqrt_rewgt, g.sigma[1] / sqrt_rewgt,
          g.skew[0] / sqrt_rewgt, g.skew[1] / sqrt_rewgt)
    b2 = (g.peak2, g.sigma2[0] / sqrt_rewgt, g.sigma2[1] / sqrt_rewgt,
          0.0, 0.0)

    sigma_shifted = bool({'sigma_lo', 'sigma_hi'} & active)
    sigma2_shifted = bool({'sigma2_lo', 'sigma2_hi'} & active)
    skew_shifted = bool({'skew_lo', 'skew_hi'} & active)
    prob2_on = (g.prob2 > _PROB2_EPS
                or bool({'prob2', 'peak2', 'sigma2_lo', 'sigma2_hi'}
                        & active))

    br1, lo1, hi1, grid1 = _component_plan(
        b1[1], b1[2], b1[3], b1[4], pr_lo, pr_hi,
        range_lo, range_hi, g.ngrid, sigma_shifted, skew_shifted)

    if not prob2_on:
        return _component_value(
            key, br1, peak, sig_lo, sig_hi, skew_lo, skew_hi, pr_lo, pr_hi,
            lo1, hi1, range_lo, range_hi, g.ngrid, grid1, traced)

    # Secondary component: degenerate peakrange, no skew (:545-553).
    br2, lo2, hi2, grid2 = _component_plan(
        b2[1], b2[2], 0.0, 0.0, 0.0, 0.0,
        range_lo, range_hi, g.ngrid, sigma2_shifted, False)

    m1 = _component_mass(br1, peak, sig_lo, sig_hi, pr_lo, pr_hi, lo1, hi1)
    m2 = _component_mass(br2, peak2, sig2_lo, sig2_hi, 0.0, 0.0, lo2, hi2)
    total = (1.0 - prob2) * m1 + prob2 * m2
    eff_prob2 = prob2 * m2 / total

    k_sel, k1, k2 = jax.random.split(key, 3)
    v1 = _component_value(
        k1, br1, peak, sig_lo, sig_hi, skew_lo, skew_hi, pr_lo, pr_hi,
        lo1, hi1, range_lo, range_hi, g.ngrid, grid1, traced)
    v2 = _component_value(
        k2, br2, peak2, sig2_lo, sig2_hi, 0.0, 0.0, 0.0, 0.0,
        lo2, hi2, range_lo, range_hi, g.ngrid, grid2, traced)
    use2 = jax.random.uniform(k_sel) < eff_prob2
    return jnp.where(use2, v2, v1)



def mixture_mass(g):
    """The probability that a draw from ``g``, at its declared parameters,
    lands within its range. SNANA redraws an out-of-range value and aborts
    when no draw can land (the MXTRY abort, sntools_genGauss_asym.c:598-606),
    so a population needs this above zero. Only a two-component mixture can
    have none: a single component draws within its range by construction."""
    if g.prob2 <= _PROB2_EPS:
        return 1.0
    sqrt_rewgt = float(g.prob_expon_rewgt) ** 0.5
    sig_lo, sig_hi = g.sigma[0] / sqrt_rewgt, g.sigma[1] / sqrt_rewgt
    sig2_lo, sig2_hi = g.sigma2[0] / sqrt_rewgt, g.sigma2[1] / sqrt_rewgt
    br1, lo1, hi1, _ = _component_plan(
        sig_lo, sig_hi, g.skew[0] / sqrt_rewgt, g.skew[1] / sqrt_rewgt,
        *g.peakrange, *g.range, g.ngrid)
    br2, lo2, hi2, _ = _component_plan(
        sig2_lo, sig2_hi, 0.0, 0.0, 0.0, 0.0, *g.range, g.ngrid)
    m1 = _component_mass(br1, g.peak, sig_lo, sig_hi, *g.peakrange, lo1, hi1)
    m2 = _component_mass(br2, g.peak2, sig2_lo, sig2_hi, 0.0, 0.0, lo2, hi2)
    return float((1.0 - g.prob2) * m1 + g.prob2 * m2)


GENGAUSS_ZSHIFT_FIELDS = ('peak', 'sigma_lo', 'sigma_hi', 'skew_lo',
                          'skew_hi', 'prob2', 'peak2', 'sigma2_lo',
                          'sigma2_hi')
EXPHALFGAUSS_ZSHIFT_FIELDS = ('exp_tau', 'peak', 'sigma', 'ratio')


def _skew_cdf_grids_traced(peak, sigma_lo, sigma_hi, skew_lo, skew_hi,
                           range_lo, range_hi, n_grid=1000):
    """Traced twin of precompute_skew_cdf for per-SN (shifted) parameters.

    The value grid is static (RANGE is never z-shifted); only the density
    depends on traced parameters.
    """
    x = jnp.linspace(range_lo, range_hi, n_grid)
    x_rel = x - peak
    sig_base = jnp.where(x_rel < 0, sigma_lo, sigma_hi)
    skew_val = jnp.where(x_rel < 0, skew_lo, skew_hi)
    sig = sig_base + skew_val * jnp.abs(x_rel)
    density = jnp.exp(-0.5 * (x_rel / sig) ** 2)
    dx = (range_hi - range_lo) / (n_grid - 1)
    cdf = jnp.concatenate([
        jnp.zeros(1),
        jnp.cumsum((density[:-1] + density[1:]) / 2.0 * dx)])
    return x, cdf / cdf[-1]


def _sample_exp_halfgauss_shifted(key, dist, shifts):
    """Traced-shift path of sample_exp_halfgauss (ZVARIATION)."""
    d = dict(zip(EXPHALFGAUSS_ZSHIFT_FIELDS, shifts))
    range_lo, range_hi = dist.range
    tau = dist.exp_tau + d['exp_tau']
    peak = dist.peak + d['peak']
    sigma = dist.sigma + d['sigma']
    ratio = dist.ratio + d['ratio']

    if dist.prob_expon_rewgt != 1.0:
        tau = tau / dist.prob_expon_rewgt
        sigma = sigma / jnp.sqrt(dist.prob_expon_rewgt)

    if (range_hi - range_lo) < 1e-30:
        return jnp.float32(range_lo)

    key1, key2 = jax.random.split(key)

    val_flat = jax.random.uniform(key2, minval=range_lo, maxval=range_hi)
    val_exp = _sample_truncated_exponential(key2, tau, range_lo, range_hi)
    val_norm = _sample_truncated_normal(key2, peak, sigma,
                                        range_lo, range_hi)
    val_half = _sample_truncated_halfnormal(key2, sigma,
                                            range_lo, range_hi)
    val_gauss = jnp.where(peak > 1e-4, val_norm, val_half)

    w_exp = tau * (jnp.exp(-range_lo / tau) - jnp.exp(-range_hi / tau))
    w_gauss = 0.5 * ratio * jnp.sqrt(2.0 * jnp.pi) * sigma
    u_comp = jax.random.uniform(key1)
    val_mixed = jnp.where(u_comp < w_exp / (w_exp + w_gauss),
                          val_exp, val_gauss)

    # ratio -> 0 takes the exponential where the eager path aborts (:337)
    return jnp.where(
        (tau > 50.0) & (ratio == 0.0), val_flat,
        jnp.where((tau > _EXP_HALFGAUSS_EPS) & (sigma < _EXP_HALFGAUSS_EPS),
                  val_exp,
                  jnp.where((sigma > _EXP_HALFGAUSS_EPS)
                            & (tau < _EXP_HALFGAUSS_EPS),
                            val_gauss, val_mixed)))


# SIM_WGT_POPULATION population functions (rewgt_genPDF, snlc_sim.c:31492)
def pdf_gengauss_asym(x, g, rewgt=1.0):
    """funVal_GENGAUSS_ASYM (sntools_genGauss_asym.c:186): the peakrange
    plateau with Gaussian wings, plus the second peak scaled by the width
    ratio, every width divided by sqrt(rewgt); skew and grid are ignored,
    as there. A width of zero is a delta, whose value is one everywhere it
    can be drawn."""
    scale = np.sqrt(rewgt)
    sig_lo, sig_hi = g.sigma[0] / scale, g.sigma[1] / scale
    pr_lo, pr_hi = g.peakrange
    if pr_hi - pr_lo <= _PEAKRANGE_EPS:
        pr_lo = pr_hi = g.peak
    if sig_lo <= 0.0 and sig_hi <= 0.0:
        return jnp.ones_like(x)
    below = jnp.exp(-0.5 * ((x - pr_lo) / jnp.maximum(sig_lo, 1e-300)) ** 2)
    above = jnp.exp(-0.5 * ((x - pr_hi) / jnp.maximum(sig_hi, 1e-300)) ** 2)
    value = jnp.where(x < pr_lo, below, jnp.where(x < pr_hi, 1.0, above))
    if g.prob2 <= 0.0:
        return value
    sig2_lo, sig2_hi = g.sigma2[0] / scale, g.sigma2[1] / scale
    sig2 = jnp.where(x < g.peak2, sig2_lo, sig2_hi)
    second = jnp.exp(-0.5 * ((x - g.peak2) / sig2) ** 2)
    sig_ratio = (sig_hi + sig_lo) / (sig2_hi + sig2_lo)
    return (1.0 - g.prob2) * value + g.prob2 * second * sig_ratio


def pdf_exp_halfgauss(x, d, rewgt=1.0):
    """funVal_GEN_EXP_HALFGAUSS (sntools_genExpHalfGauss.c:150): the
    exponential with tau divided by rewgt, plus ratio times the half
    Gaussian with sigma divided by sqrt(rewgt)."""
    tau = d.exp_tau / rewgt
    value = jnp.exp(-x / tau)
    if d.ratio > 0.0:
        sigma = d.sigma / np.sqrt(rewgt)
        value = value + d.ratio * jnp.exp(-0.5 * ((x - d.peak) / sigma) ** 2)
    return value


jax.tree_util.register_dataclass(
    GenGaussAsym,
    data_fields=[],
    meta_fields=[
        'peak',
        'sigma',
        'range',
        'peakrange',
        'skew',
        'prob2',
        'peak2',
        'sigma2',
        'ngrid',
        'prob_expon_rewgt',
    ],
)


jax.tree_util.register_dataclass(
    GenExpHalfGauss,
    data_fields=[],
    meta_fields=[
        'exp_tau',
        'peak',
        'sigma',
        'ratio',
        'range',
        'prob_expon_rewgt',
    ],
)
