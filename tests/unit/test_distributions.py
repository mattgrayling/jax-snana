"""Unit tests for the parameter-distribution samplers.

Covers both low-level helpers (``bifurcated_cdf``/``bifurcated_ppf``,
``_sample_truncated_<dist>``, ``precompute_skew_cdf``,
``_sample_from_precomputed_cdf``) and the top-level dispatchers
(``sample_gengauss_asym``, ``sample_exp_halfgauss``,
plus the grid helpers ``_grid_params`` / ``_apply_grid_snapping``).
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import norm, truncnorm

from jax_snlc_core.logging import SNANAError
from jax_snlc_sim.distributions import (EXPHALFGAUSS_ZSHIFT_FIELDS, GenExpHalfGauss,
                                        GenGaussAsym, _apply_grid_snapping, bifurcated_cdf,
                                        bifurcated_ppf, _grid_params, _peakrange_mass,
                                        _sample_truncated_exponential,
                                        _sample_truncated_normal, distribution_shifts,
                                        mixture_mass, pdf_exp_halfgauss, pdf_gengauss_asym,
                                        precompute_skew_cdf, sample_exp_halfgauss,
                                        sample_gengauss_asym, sample_truncated_gaussian)


def _draw(sampler, key, n=2000):
    keys = jax.random.split(key, n)
    return np.asarray(jax.vmap(sampler)(keys))


# ----------------------------------------------------------------------------
# Bifurcated CDF / PPF round-trip
# ----------------------------------------------------------------------------


def test_bifurcated_cdf_at_peak():
    """At the peak, CDF = sigma_lo / (sigma_lo + sigma_hi) (left tail mass)."""
    sigma_lo, sigma_hi = 1.0, 2.0
    cdf = float(bifurcated_cdf(0.0, peak=0.0,
                                sigma_lo=sigma_lo, sigma_hi=sigma_hi))
    expected = sigma_lo / (sigma_lo + sigma_hi)
    assert abs(cdf - expected) < 1e-5


def test_bifurcated_ppf_inverts_cdf():
    """ppf(cdf(x)) = x for arbitrary x."""
    peak, slo, shi = 0.5, 0.3, 0.6
    xs = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
    for x in xs:
        u = float(bifurcated_cdf(x, peak, slo, shi))
        x_back = float(bifurcated_ppf(u, peak, slo, shi))
        assert abs(x_back - x) < 1e-4


def test_sample_truncated_gaussian_mean_offset_for_asymmetric():
    """When σ_lo > σ_hi the distribution is left-skewed → sample mean < peak."""
    key = jax.random.PRNGKey(1)
    keys = jax.random.split(key, 5000)
    samples = jax.vmap(
        lambda k: sample_truncated_gaussian(k, 0.0, 1.0, 0.2, -5.0, 5.0)
    )(keys)
    assert float(np.mean(np.asarray(samples))) < 0.0


# ----------------------------------------------------------------------------
# Low-level truncated samplers
# ----------------------------------------------------------------------------


def test_truncated_exponential_within_range():
    """A narrow truncation range is actually enforced (the mean test below
    uses range_hi=20.0 which leaves the exponential effectively unbounded)."""
    keys = jax.random.split(jax.random.PRNGKey(0), 1000)
    samples = jax.vmap(
        lambda k: _sample_truncated_exponential(k, tau=0.4, range_lo=0.0, range_hi=2.0)
    )(keys)
    s = np.asarray(samples)
    assert s.min() >= 0.0
    assert s.max() <= 2.0


def test_truncated_exponential_mean_close_to_tau():
    """Mean of an unbounded exponential ≈ tau."""
    keys = jax.random.split(jax.random.PRNGKey(1), 5000)
    samples = jax.vmap(
        lambda k: _sample_truncated_exponential(k, tau=0.5, range_lo=0.0, range_hi=20.0)
    )(keys)
    assert abs(float(np.mean(np.asarray(samples))) - 0.5) < 0.05


def test_truncated_normal_centred_within_range():
    keys = jax.random.split(jax.random.PRNGKey(3), 5000)
    samples = jax.vmap(
        lambda k: _sample_truncated_normal(k, mean=3.0, sigma=0.5,
                                           range_lo=2.5, range_hi=3.5)
    )(keys)
    s = np.asarray(samples)
    assert s.min() >= 2.5
    assert s.max() <= 3.5
    assert abs(float(np.mean(s)) - 3.0) < 0.05


# ----------------------------------------------------------------------------
# Precomputed skew CDF
# ----------------------------------------------------------------------------


def test_precompute_skew_cdf_normalised_and_monotonic():
    _, cdf = precompute_skew_cdf(
        peak=0.0, sigma_lo=0.5, sigma_hi=0.5,
        skew_lo=0.0, skew_hi=0.0, range_lo=-2.0, range_hi=2.0, n_grid=200,
    )
    cdf = np.asarray(cdf)
    assert cdf[0] == 0.0
    assert abs(cdf[-1] - 1.0) < 1e-5
    assert np.all(np.diff(cdf) >= -1e-9)


# ----------------------------------------------------------------------------
# sample_gengauss_asym branches
# ----------------------------------------------------------------------------


def test_gengauss_delta_function_when_range_collapsed():
    """range_lo == range_hi → returns range_lo (delta function)."""
    g = GenGaussAsym(peak=0.5, range=(2.0, 2.0), sigma=(1.0, 1.0))
    val = float(sample_gengauss_asym(jax.random.PRNGKey(0), g))
    assert val == pytest.approx(2.0)


def test_gengauss_dispatches_to_truncated_bifurcated():
    """Default branch — peak with sigma, no skew/peakrange/peak2 — yields
    samples within the truncation range and concentrated near peak."""
    g = GenGaussAsym(peak=0.0, sigma=(0.3, 0.3), range=(-1.0, 1.0))
    s = _draw(lambda k: sample_gengauss_asym(k, g), jax.random.PRNGKey(1))
    assert s.min() >= -1.0 - 1e-5 and s.max() <= 1.0 + 1e-5
    assert abs(float(np.mean(s))) < 0.05


def test_gengauss_peakrange_concentrates_inside_plateau():
    """peakrange=(a,b) with small wings → most samples land in [a,b]."""
    g = GenGaussAsym(
        peak=0.0,  # ignored when peakrange active
        peakrange=(-0.2, 0.2),
        sigma=(0.05, 0.05),
        range=(-2.0, 2.0),
    )
    s = _draw(lambda k: sample_gengauss_asym(k, g), jax.random.PRNGKey(2))
    in_plateau = np.mean((s >= -0.2 - 0.01) & (s <= 0.2 + 0.01))
    assert in_plateau > 0.5, (
        f"only {in_plateau:.2f} of samples landed in the flat plateau"
    )


@pytest.mark.parametrize("side", [-1, 1])
def test_peakrange_truncated_tail(side):
    lo, hi = (-3., -2.) if side < 0 else (2., 3.)
    g = GenGaussAsym(peakrange=(-1., 1.), sigma=(1., 1.), range=(lo, hi))
    samples = _draw(lambda key: sample_gengauss_asym(key, g),
                    jax.random.PRNGKey(9))
    assert np.all((samples >= lo) & (samples <= hi))
    expected = truncnorm.mean(lo - side, hi - side, loc=side)
    assert samples.mean() == pytest.approx(expected, abs=0.025)
    mass = (norm.cdf(hi - side) - norm.cdf(lo - side)) / (
        1. + 2. / np.sqrt(2. * np.pi))
    assert float(_peakrange_mass(1., 1., -1., 1., lo, hi)) == pytest.approx(mass)


def test_gengauss_dual_peak_produces_bimodal_distribution():
    """prob2>0 with peak2 well separated → both peaks produce samples."""
    g = GenGaussAsym(
        peak=-1.0,
        sigma=(0.1, 0.1),
        prob2=0.5,
        peak2=+1.0,
        sigma2=(0.1, 0.1),
        range=(-2.0, 2.0),
    )
    s = _draw(lambda k: sample_gengauss_asym(k, g), jax.random.PRNGKey(3), n=4000)
    near_lo = np.mean(s < -0.5)
    near_hi = np.mean(s > +0.5)
    assert near_lo > 0.2, f"too few samples near peak1: {near_lo:.2f}"
    assert near_hi > 0.2, f"too few samples near peak2: {near_hi:.2f}"


def test_gengauss_grid_snapping_quantises_samples():
    """ngrid > 0 collapses samples to discrete grid values."""
    g = GenGaussAsym(
        peak=0.0,
        sigma=(0.5, 0.5),
        range=(-1.0, 1.0),
        ngrid=5,
    )
    s = _draw(lambda k: sample_gengauss_asym(k, g), jax.random.PRNGKey(5), n=2000)
    # Grid points: linspace(-1, 1, 5) = [-1.0, -0.5, 0.0, 0.5, 1.0]
    expected = np.linspace(-1.0, 1.0, 5)
    unique = np.unique(np.round(s, 4))
    # Every observed value should be (approximately) one of the grid points.
    for u in unique:
        diffs = np.abs(expected - u)
        assert diffs.min() < 1e-3, f"value {u:.4f} not on grid {expected}"


def test_gengauss_prob_expon_rewgt_narrows_distribution():
    """``prob_expon_rewgt > 1`` divides sigma by sqrt(rewgt) → narrower draws."""
    base = GenGaussAsym(peak=0.0, sigma=(0.5, 0.5), range=(-2.0, 2.0))
    rewgt = GenGaussAsym(
        peak=0.0, sigma=(0.5, 0.5), range=(-2.0, 2.0), prob_expon_rewgt=4.0
    )
    s_base = _draw(lambda k: sample_gengauss_asym(k, base), jax.random.PRNGKey(7), n=4000)
    s_rewgt = _draw(lambda k: sample_gengauss_asym(k, rewgt), jax.random.PRNGKey(7), n=4000)
    assert float(np.std(s_rewgt)) < float(np.std(s_base)), (
        "PROB_EXPON_REWGT=4 should narrow the sampled std"
    )


# ----------------------------------------------------------------------------
# sample_exp_halfgauss branches
# ----------------------------------------------------------------------------


def test_exp_halfgauss_delta_when_range_collapsed():
    d = GenExpHalfGauss(range=(1.5, 1.5), exp_tau=0.1, sigma=0.1)
    val = float(sample_exp_halfgauss(jax.random.PRNGKey(8), d))
    assert val == pytest.approx(1.5)


def test_exp_halfgauss_pure_exponential_branch():
    """sigma=0: SNANA draws -tau*log of a flat random between exp(-range/tau)
    (sntools_genExpHalfGauss.c, DOFUN_EXPON), an exponential truncated to the
    range: mean tau - R e^(-R/tau)/(1 - e^(-R/tau)) and P(x < tau) =
    (1 - 1/e)/(1 - e^(-R/tau)) on [0, R]."""
    tau, upper, n = 0.5, 5.0, 40000
    d = GenExpHalfGauss(exp_tau=tau, sigma=0.0, ratio=0.0, range=(0.0, upper))
    s = _draw(lambda k: sample_exp_halfgauss(k, d), jax.random.PRNGKey(9), n=n)
    cut = np.exp(-upper / tau)
    assert s.min() >= 0.0 and s.max() <= upper
    assert abs(s.mean() - (tau - upper * cut / (1.0 - cut))) < 4 * tau / np.sqrt(n)
    assert abs(np.mean(s < tau) - (1.0 - np.exp(-1.0)) / (1.0 - cut)) < 0.01


def test_exp_halfgauss_pure_halfgaussian_branch():
    """tau=0, peak=0: SNANA draws sigma*|Gauss| until it lands in the range
    (sntools_genExpHalfGauss.c, DOFUN_GAUSS), a half-normal: mean
    sigma*sqrt(2/pi) and P(x < sigma) = erf(1/sqrt(2)) when the range reaches
    six sigma."""
    sigma, n = 0.5, 40000
    d = GenExpHalfGauss(exp_tau=0.0, sigma=sigma, ratio=0.5, peak=0.0,
                        range=(0.0, 3.0))
    s = _draw(lambda k: sample_exp_halfgauss(k, d), jax.random.PRNGKey(10), n=n)
    assert s.min() >= 0.0 and s.max() <= 3.0
    std_err = sigma * np.sqrt(1.0 - 2.0 / np.pi) / np.sqrt(n)
    assert abs(s.mean() - sigma * np.sqrt(2.0 / np.pi)) < 4 * std_err
    assert abs(np.mean(s < sigma) - 0.6826894921) < 0.01


@pytest.mark.parametrize("tau, sigma, message", [
    # neither width set: nothing to draw from (sntools_genExpHalfGauss.c:235-241)
    (0.0, 0.0, "must not be zero"),
    # both set, but ratio=0 says nothing about how to mix them (:337-340)
    (0.4, 0.5, "Could not determine"),
])
def test_exp_halfgauss_aborts_on_degenerate_profile(tau, sigma, message):
    """SNANA refuses these rather than guessing.

    jax used to read ratio=0 as "pure exponential", which sampled one without
    comment in the second case and divided by a zero tau in the first.
    """
    d = GenExpHalfGauss(exp_tau=tau, sigma=sigma, ratio=0.0, range=(0.0, 3.0))
    with pytest.raises(SNANAError, match=message):
        sample_exp_halfgauss(jax.random.PRNGKey(11), d)


def test_exp_halfgauss_pure_truncated_normal_branch():
    """tau=0 + peak≠0 → pure truncated normal centred at peak."""
    d = GenExpHalfGauss(exp_tau=0.0, sigma=0.2, ratio=0.5, peak=1.0, range=(0.0, 2.0))
    s = _draw(lambda k: sample_exp_halfgauss(k, d), jax.random.PRNGKey(11), n=2000)
    assert s.min() >= 0.0 - 1e-5 and s.max() <= 2.0 + 1e-5
    assert abs(float(np.mean(s)) - 1.0) < 0.1


def test_exp_halfgauss_flat_branch_when_tau_huge():
    """tau>50 + ratio=0 → flat sampling within range."""
    d = GenExpHalfGauss(exp_tau=100.0, sigma=0.0, ratio=0.0, range=(0.0, 1.0))
    s = _draw(lambda k: sample_exp_halfgauss(k, d), jax.random.PRNGKey(13), n=2000)
    assert abs(float(np.mean(s)) - 0.5) < 0.05


# ----------------------------------------------------------------------------
# Grid helpers
# ----------------------------------------------------------------------------


def test_grid_params_matches_snana_extension():
    """SNANA extends sampling range by gridsize/2 on each side."""
    gridsize, grid0, ext_lo, ext_hi = _grid_params(0.0, 4.0, ngrid=5)
    assert gridsize == pytest.approx(1.0)
    assert grid0 == 0.0
    assert ext_lo == pytest.approx(-0.5)
    assert ext_hi == pytest.approx(4.5)


def test_apply_grid_snapping_quantises_to_known_grid():
    snapped = float(_apply_grid_snapping(jnp.float32(0.7), 0.0, 4.0, ngrid=5))
    # Grid: 0,1,2,3,4 with extended_lo = -0.5; (0.7 - -0.5)/1.0 = 1.2 → ibin=1 → 1.0
    assert snapped == pytest.approx(1.0)


# ----------------------------------------------------------------------------
# Mixture composition (getRan_GENGAUSS_ASYM, sntools_genGauss_asym.c:331-454):
# the secondary peak is selected before the component's shape is applied, so
# PROB2 composes with a skewed or peakrange primary instead of being dropped.
# ----------------------------------------------------------------------------


def test_gengauss_prob2_composes_with_skew_primary():
    g = GenGaussAsym(peak=0.0, sigma=(0.5, 0.5), skew=(0.0, 0.4),
                     range=(-2.0, 6.0), prob2=0.3, peak2=4.0,
                     sigma2=(0.05, 0.05))
    vals = _draw(lambda k: sample_gengauss_asym(k, g),
                 jax.random.PRNGKey(3), n=20000)
    # the narrow secondary peak stands out from the skew tail
    frac_peak2 = np.mean(np.abs(vals - 4.0) < 0.2)
    assert frac_peak2 > 0.25, frac_peak2


def test_gengauss_prob2_composes_with_peakrange_primary():
    g = GenGaussAsym(peak=0.0, sigma=(0.3, 0.3), peakrange=(-1.0, 1.0),
                     range=(-3.0, 8.0), prob2=0.25, peak2=6.0,
                     sigma2=(0.2, 0.2))
    vals = _draw(lambda k: sample_gengauss_asym(k, g),
                 jax.random.PRNGKey(4), n=20000)
    frac_peak2 = np.mean(np.abs(vals - 6.0) < 1.0)
    assert 0.20 < frac_peak2 < 0.30, frac_peak2


def test_gengauss_prob2_reweighted_by_truncated_mass():
    """A secondary peak mostly outside RANGE loses mixture weight in
    proportion to its accepted mass (the restart re-selects the component)."""
    g = GenGaussAsym(peak=0.0, sigma=(1.0, 1.0), range=(-3.0, 2.0),
                     prob2=0.5, peak2=5.0, sigma2=(1.0, 1.0))
    vals = _draw(lambda k: sample_gengauss_asym(k, g),
                 jax.random.PRNGKey(5), n=40000)
    # mass2 = P(N(5,1) in (-3,2)) ~ 1.35e-3; mass1 ~ 0.976
    # eff_prob2 = 0.5*m2/(0.5*m1 + 0.5*m2) ~ 1.4e-3, so the range top is
    # dominated by the primary tail, not a half-weight secondary peak
    assert np.mean(vals > 1.5) < 0.07


def test_mixture_with_no_probability_in_range_is_refused_at_setup():
    """SNANA aborts when no draw from a two-component mixture can land in
    range (sntools_genGauss_asym.c:598-606). The draw runs compiled, so the
    run refuses such a population when it is set up instead."""
    from jax_snlc_sim.population import Population, validate

    outside = GenGaussAsym(peak=50.0, sigma=(0.1, 0.1), range=(-1.0, 1.0),
                           prob2=0.5, peak2=60.0, sigma2=(0.1, 0.1))
    assert mixture_mass(outside) == 0.0
    with pytest.raises(SNANAError, match="no probability in its range"):
        validate('c', Population(distributions=(outside,), shifts=((),)), {})
    reachable = GenGaussAsym(peak=0.0, sigma=(1.0, 1.0), range=(-3.0, 2.0),
                             prob2=0.5, peak2=5.0, sigma2=(1.0, 1.0))
    assert 0.0 < mixture_mass(reachable) < 1.0


def test_gengauss_flat_threshold_is_ten_times_range():
    """SNANA flattens when both sigmas exceed 10x the (extended) range
    (sntools_genGauss_asym.c:387,404); 15x the width must sample flat."""
    g = GenGaussAsym(peak=0.5, sigma=(15.0, 15.0), range=(0.0, 1.0))
    vals = _draw(lambda k: sample_gengauss_asym(k, g),
                 jax.random.PRNGKey(6), n=40000)
    hist, _ = np.histogram(vals, bins=10, range=(0.0, 1.0))
    assert hist.std() / hist.mean() < 0.05


def test_gengauss_ngrid_one_does_not_snap():
    """DO_GRID needs NGRID >= 2 (:374); NGRID=1 leaves the law unsnapped."""
    g = GenGaussAsym(peak=0.5, sigma=(0.2, 0.2), range=(0.0, 1.0), ngrid=1)
    vals = _draw(lambda k: sample_gengauss_asym(k, g),
                 jax.random.PRNGKey(7), n=2000)
    assert len(np.unique(np.round(vals, 6))) > 1000


def test_gengauss_ngrid_out_of_bounds_aborts():
    """NGRID outside [0, 100] is fatal (:312-325)."""
    g = GenGaussAsym(peak=0.5, sigma=(1.0, 1.0), range=(0.0, 1.0), ngrid=101)
    with pytest.raises(SNANAError):
        sample_gengauss_asym(jax.random.PRNGKey(0), g)


def test_zshift_pure_halfgauss_stays_finite():
    """The traced ZVARIATION path selects the pure branches on the two
    widths alone (sntools_genExpHalfGauss.c:259-260): a pure half-Gaussian
    must never reach the exponential draw's 1/tau."""
    d = GenExpHalfGauss(exp_tau=0.0, peak=0.0, sigma=0.3, ratio=0.0,
                        range=(0.0, 3.0))
    keys = jax.random.split(jax.random.PRNGKey(3), 64)
    zeros = [jnp.zeros(64)] * len(EXPHALFGAUSS_ZSHIFT_FIELDS)
    vals = jax.vmap(lambda key, *shifts: sample_exp_halfgauss(key, d, shifts))(keys, *zeros)
    assert bool(jnp.all(jnp.isfinite(vals)))
    assert bool(jnp.all((vals >= 0.0) & (vals <= 3.0)))


def test_population_weight_undoes_a_reweighted_gaussian():
    """PROB_EXPON_REWGT e narrows a Gaussian's sigma by sqrt(e), so the
    nominal-over-reweighted ratio SNANA writes as SIM_WGT_POPULATION is
    exp(0.5 d^2 (e - 1) / sigma^2)."""
    g = GenGaussAsym(peak=0.0, sigma=(0.05, 0.05), range=(-0.3, 0.3))
    x = jnp.array([-0.1, 0.0, 0.07])

    ratio = pdf_gengauss_asym(x, g) / pdf_gengauss_asym(x, g, 4.0)

    np.testing.assert_allclose(ratio, np.exp(0.5 * (np.asarray(x) / 0.05) ** 2 * 3.0),
                               rtol=1e-6)


def test_population_weight_undoes_a_reweighted_exponential():
    d = GenExpHalfGauss(exp_tau=0.4, range=(0.0, 2.0))
    x = jnp.array([0.1, 0.5])

    ratio = pdf_exp_halfgauss(x, d) / pdf_exp_halfgauss(x, d, 2.0)

    np.testing.assert_allclose(ratio, np.exp(np.asarray(x) / 0.4), rtol=1e-6)


def test_traced_gaussian_hyperparameters_match_prepared_distribution_and_density():
    """Numerical overrides use the same draw and density as configured values."""
    base =GenGaussAsym(peak=0., sigma=(.3, .4), range=(-2., 2.))
    hyperparameters = {'peak': .2, 'sigma': (.4, .5), 'prob2': .2,
                       'peak2': -.4, 'sigma2': (.2, .3)}
    changed = replace(base, **hyperparameters)
    shifts, active = distribution_shifts(base, hyperparameters)
    key = jax.random.PRNGKey(21)
    actual = sample_gengauss_asym(key, base, shifts, active)
    expected = sample_gengauss_asym(key, changed)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    x = jnp.linspace(-1., 1., 21)
    for reweighting in (1., 2.):
        density = jax.jit(lambda peak: pdf_gengauss_asym(
            x, replace(changed, peak=peak), reweighting))
        np.testing.assert_allclose(density(hyperparameters['peak']),
                                   pdf_gengauss_asym(x, changed, reweighting), rtol=1e-6)


def test_traced_exponential_scale_has_finite_gradient():
    """Inactive zero-width Gaussian terms do not contaminate dust gradients."""
    dist =GenExpHalfGauss(exp_tau=.3, range=(0., 2.))
    keys = jax.random.split(jax.random.PRNGKey(28), 20)

    def mean_draw(tau):
        shifts, _ = distribution_shifts(dist, {'exp_tau': tau})
        return jax.vmap(lambda key: sample_exp_halfgauss(key, dist, shifts))(keys).mean()

    value = jnp.array(.4)
    derivative = jax.jit(jax.grad(mean_draw))(value)
    step = 1e-4
    finite = (mean_draw(value + step) - mean_draw(value - step)) / (2 * step)
    assert np.isfinite(derivative)
    np.testing.assert_allclose(derivative, finite, rtol=2e-3)
    x = jnp.array([.1, .5])
    ratio = (pdf_exp_halfgauss(x, replace(dist, exp_tau=value))
             / pdf_exp_halfgauss(x, replace(dist, exp_tau=value), 2.))
    np.testing.assert_allclose(ratio, np.exp(np.asarray(x) / value), rtol=1e-6)


@pytest.mark.parametrize('widths', [(0., .3), (.3, 0.)])
def test_call_time_mixture_density_retains_one_sided_widths(widths):
    """Safe divisors must not change the normalization of a one-sided component."""
    g = GenGaussAsym(peak=0., sigma=(.4, .5), peak2=.2, sigma2=widths, prob2=.3)
    x = jnp.array([-.5, .5])

    def density(probability):
        return pdf_gengauss_asym(x, replace(g, prob2=probability))

    np.testing.assert_allclose(jax.jit(density)(.3), pdf_gengauss_asym(x, g), rtol=1e-6)
    step = 1e-4
    numerical = (density(.3 + step) - density(.3 - step)) / (2 * step)
    np.testing.assert_allclose(jax.jacfwd(density)(.3), numerical, rtol=1e-5)


def test_call_time_dust_density_omits_zero_width_gaussian():
    """A zero-width Gaussian must not turn into a unit-width dust component."""
    d = GenExpHalfGauss(exp_tau=.4, sigma=0., ratio=.5, range=(0., 2.))
    x = jnp.array([.1, .5])

    def density(tau):
        return pdf_exp_halfgauss(x, replace(d, exp_tau=tau))

    expected = np.exp(-np.asarray(x) / .4)
    np.testing.assert_allclose(jax.jit(density)(.4), expected, rtol=1e-6)
    np.testing.assert_allclose(pdf_exp_halfgauss(x, d), expected, rtol=1e-6)
    derivative = jax.jacrev(density)(.4)
    np.testing.assert_allclose(derivative, expected * np.asarray(x) / .4**2, rtol=1e-6)


@pytest.mark.parametrize('ratio', [0., 2.])
def test_call_time_pure_gaussian_dust_has_finite_reweighting(ratio):
    """Removing the exponential leaves Gaussian reweighting, regardless of mixture ratio."""
    d = GenExpHalfGauss(exp_tau=.4, sigma=0., ratio=ratio, range=(0., 2.))
    x = jnp.array([.1, .5])

    def weight(sigma):
        gaussian = replace(d, exp_tau=0., sigma=sigma)
        return pdf_exp_halfgauss(x, gaussian) / pdf_exp_halfgauss(x, gaussian, 2.)

    expected = np.exp(.5 * (np.asarray(x) / .3) ** 2)
    np.testing.assert_allclose(jax.jit(weight)(.3), expected, rtol=1e-6)
    gradient = jax.jacrev(weight)(.3)
    np.testing.assert_allclose(gradient, -expected * np.asarray(x)**2 / .3**3, rtol=1e-6)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_one_sided_gaussian_sampling_gradient(side):
    """A zero-width side must not spoil the active width's gradient."""
    sigma = (.3, 0.) if side == 'left' else (0., .3)
    dist = GenGaussAsym(sigma=sigma, range=(-2., 2.))
    keys = jax.random.split(jax.random.PRNGKey(7), 32)

    def draws(width):
        sigma = (width, 0.) if side == 'left' else (0., width)
        shifts, active = distribution_shifts(dist, {'sigma': sigma})
        return jax.vmap(lambda key: sample_gengauss_asym(key, dist, shifts, active))(keys)

    values = jax.jit(draws)(.3)
    derivative = jax.jit(jax.jacrev(draws))(.3)
    step = 1e-5
    numerical = (draws(.3 + step) - draws(.3 - step)) / (2. * step)
    assert np.isfinite(values).all()
    assert np.isfinite(derivative).all()
    np.testing.assert_allclose(derivative, numerical, rtol=2e-5, atol=1e-8)


@pytest.mark.parametrize('sigma', [0.1, 0.01])
def test_truncated_normal_outside_support(sigma):
    """Far-tail draws obey the bound and agree with SciPy's tail quantiles."""
    keys =jax.random.split(jax.random.PRNGKey(78), 1000)
    values = np.asarray(jax.jit(jax.vmap(
        lambda key: _sample_truncated_normal(key, 1.1, sigma, 1.2, 6.)))(keys))
    u = np.asarray(jax.vmap(jax.random.uniform)(keys))
    expected = truncnorm.ppf(1. - u, (1.2 - 1.1) / sigma,
                            (6. - 1.1) / sigma, loc=1.1, scale=sigma)
    assert np.all(values >= 1.2)
    np.testing.assert_allclose(values, expected, rtol=1.e-6, atol=1.e-7)
    gradient = jax.grad(lambda scale: _sample_truncated_normal(
        keys[0], 1.1, scale, 1.2, 6.))(sigma)
    assert np.isfinite(gradient)


def test_zero_width_distribution_limits():
    """Zero-width conditional normals and exponentials have explicit limits."""
    key = jax.random.PRNGKey(14)
    for mean, expected in [(1.1, 1.2), (2., 2.), (7., 6.)]:
        assert _sample_truncated_normal(key, mean, 0., 1.2, 6.) == expected
        assert sample_truncated_gaussian(key, mean, 0., 0., 1.2, 6.) == expected
    assert _sample_truncated_exponential(key, 0., 0.2, 6.) == 0.2


def test_truncated_normal_float32_tail():
    """Tail draws stay inside the bound in single precision."""
    with jax.enable_x64(False):
        key = jax.random.PRNGKey(9)
        for sigma in (.01, 0.):
            value = _sample_truncated_normal(key, 1.1, sigma, 1.2, 6.)
            assert np.isfinite(value) and value >= jnp.float32(1.2)
            assert value < 1.21


@pytest.mark.parametrize('extra', [
    {}, {'sigma': (100., 100.)}, {'prob2': .3, 'peak2': 4., 'sigma2': (.4, .6)},
    {'skew': (.05, .05)}, {'peakrange': (2.8, 3.2)}, {'ngrid': 5},
])
def test_call_time_gaussian_bounds_match_configured_bounds(extra):
    """Changing bounds preserves normal, mixture, flat, plateau, skew and grid laws."""
    dist =GenGaussAsym(**dict({'peak': 3., 'sigma': (.3, .5), 'range': (3., 3.)}, **extra))
    keys = jax.random.split(jax.random.PRNGKey(39), 128)
    traces = []
    @jax.jit
    def draw(bounds):
        traces.append(True)
        return jax.vmap(lambda key: sample_gengauss_asym(key, dist, bounds=bounds))(keys)
    for limits in ((1.2, 6.), (2.5, 3.5), (2.9, 2.91), (3., 3.)):
        values = draw(jnp.asarray(limits))
        configured = replace(dist, range=limits)
        expected = jax.vmap(lambda key: sample_gengauss_asym(key, configured))(keys)
        np.testing.assert_allclose(values, expected, rtol=1.e-5, atol=1.e-6)
        assert np.isfinite(values).all()
    assert len(traces) == 1


def test_call_time_bounds_gradients_and_exponential_limit():
    """Bounds remain traced, including a point distribution and its endpoint derivative."""
    dist = GenGaussAsym(peak=3., sigma=(.4, .4), range=(2.61, 2.61))
    exp = GenExpHalfGauss(exp_tau=.4, range=(0., .5))
    keys = jax.random.split(jax.random.PRNGKey(91), 10000)
    @jax.jit
    def normal_mean(bounds):
        return jax.vmap(lambda key: sample_gengauss_asym(key, dist, bounds=bounds))(keys).mean()
    bounds = jnp.array([1.2, 4.])
    gradient = jax.grad(normal_mean)(bounds)
    step = 1.e-4
    for i in range(2):
        shift = jnp.eye(2)[i] * step
        finite = (normal_mean(bounds + shift) - normal_mean(bounds - shift)) / (2 * step)
        np.testing.assert_allclose(gradient[i], finite, rtol=1.e-3, atol=1.e-7)
    np.testing.assert_allclose(jax.grad(normal_mean)(jnp.array([3., 3.])), [1., 0.])
    values = jax.jit(jax.vmap(lambda key: sample_exp_halfgauss(
        key, exp, bounds=jnp.array([0., 20.]))))(keys)
    assert abs(float(values.mean()) - .4) < .012
    fixed = jax.jit(lambda bounds: sample_exp_halfgauss(keys[0], exp, bounds=bounds))
    assert fixed(jnp.array([.2, .2])) == .2
    np.testing.assert_allclose(jax.grad(fixed)(jnp.array([.2, .2])), [1., 0.])


@pytest.mark.parametrize('dynamic_bounds', [False, True])
@pytest.mark.parametrize('mixture', [False, True])
@pytest.mark.parametrize('prepared_widths', [(0., 0.), (.3, .3)])
def test_grid_bounds_follow_call_time_widths(dynamic_bounds, mixture, prepared_widths):
    """Grid expansion, mixture acceptance and snapping use the actual widths."""
    dist =GenGaussAsym(peak=1.1, sigma=prepared_widths, range=(1.2, 6.), ngrid=5,
                       prob2=.4 if mixture else 0., peak2=4., sigma2=prepared_widths)
    keys = jax.random.split(jax.random.PRNGKey(0), 256)
    traces = []
    @jax.jit
    def draw(widths):
        traces.append(True)
        hyperparameters = {'sigma': widths}
        if mixture:
            hyperparameters['sigma2'] = widths[::-1]
        shifts, active = distribution_shifts(dist, hyperparameters)
        return jax.vmap(lambda key: sample_gengauss_asym(
            key, dist, shifts, active,
            bounds=jnp.array(dist.range) if dynamic_bounds else None))(keys)

    for widths in ((0., 0.), (0., .3), (.3, 0.), (.3, .3)):
        values = draw(jnp.array(widths))
        configured = replace(dist, sigma=widths,
                             sigma2=widths[::-1] if mixture else dist.sigma2)
        expected = jax.vmap(lambda key: sample_gengauss_asym(key, configured))(keys)
        assert np.all((values >= 1.2) & (values <= 6.))
        np.testing.assert_allclose(values, expected, rtol=1.e-6, atol=1.e-7)
    assert len(traces) == 1
