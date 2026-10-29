"""The photo-z redshift prior: its z-PDF spline and its chi2."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_snlc_core.optimiser import newton_lm
from jax_snlc_fit.photoz import chi2_z_prior_gaussian


def test_chi2_z_prior_gaussian_quadratic():
    """A 1-sigma shift adds 1, once the error dwarfs the 1e-6 floor."""
    chi2 = float(chi2_z_prior_gaussian(jnp.float64(0.80), jnp.float64(0.30),
                                       jnp.float64(0.50), jnp.float64(1.0)))
    assert chi2 == pytest.approx(1.0, abs=1e-5)


def test_chi2_z_prior_gaussian_survives_a_zero_error():
    """PRIOR_ZPULL protects the width before scaling it, so a zero-error
    event still widens with PRIOR_ZERRSCALE (snlc_fit.F90:12515)."""
    tight, loose = (
        float(chi2_z_prior_gaussian(jnp.float64(0.35), jnp.float64(0.30),
                                    jnp.float64(0.0), jnp.float64(scale)))
        for scale in (1.0, 10.0))
    assert np.isfinite(tight)
    assert loose == pytest.approx(tight / 100.0, rel=1e-6)


# ---------------------------------------------------------------------------
# precompute_zpdf_spline
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("per_event", [False, True])
def test_precompute_zpdf_spline_is_the_natural_cubic_spline(per_event):
    """The batched spline is scipy's natural CubicSpline per event, sliced
    evaluation included: the same second derivatives, the same intervals."""
    from scipy.interpolate import CubicSpline
    from jax_snlc_fit import photoz
    from jax_snlc_fit.photoz import precompute_zpdf_spline

    rng = np.random.default_rng(7)
    n_sne, n_q = 3 * photoz.ZPDF_SPLINE_ROWS // 2, 11
    cdf = np.linspace(0.0, 1.0, n_q)
    z_q = np.sort(rng.uniform(0.1, 1.2, (n_sne, 1))
                  + rng.uniform(0.02, 0.3, (n_sne, 1)) * rng.normal(size=(n_sne, n_q)),
                  axis=1)
    if per_event:
        cdf = cdf[None, :] ** rng.uniform(0.8, 1.2, (n_sne, 1))
    out = precompute_zpdf_spline(z_q, cdf, n_fine=257)
    for k in (0, 1, n_sne // 2, photoz.ZPDF_SPLINE_ROWS, n_sne - 1):
        z_fine = np.linspace(z_q[k, 0], z_q[k, -1], 257)
        pdf = np.maximum(CubicSpline(z_q[k], cdf[k] if per_event else cdf, bc_type='natural')(z_fine, 1), 0.0)
        np.testing.assert_allclose(out['z_fine'][k], z_fine, rtol=0, atol=1e-12)
        np.testing.assert_allclose(out['pdf_fine'][k], pdf / pdf.max(), rtol=0, atol=1e-12)
        np.testing.assert_allclose(out['pdf_max'][k], pdf.max(), rtol=1e-12)


def test_steffen_pdf_never_dips_below_zero():
    """Steffen's interpolant of an increasing CDF is monotone, so its
    derivative, the PDF, is nonnegative between every pair of quantiles,
    where the natural spline's overshoots on a skewed set; both are the
    secant's slope through collinear quantiles."""
    from jax_snlc_core.interpolation import (hermite_spline_derivative, natural_spline_slopes,
                                             spline_knots, steffen_slopes)
    from jax_snlc_fit.photoz import precompute_zpdf_spline

    cdf = np.linspace(0.0, 1.0, 11)
    # half the mass in a narrow peak, the rest in a long tail
    peaked = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.501, 0.502, 0.503, 0.504, 0.505, 0.9]])
    knots = spline_knots(peaked, cdf)
    _, cubic = hermite_spline_derivative(knots, natural_spline_slopes(knots), 400)
    _, steffen = hermite_spline_derivative(knots, steffen_slopes(knots), 400)
    assert cubic.min() < 0.0
    assert steffen.min() >= 0.0

    line = np.array([[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]])
    for spline in ('cubic', 'steffen'):
        out = precompute_zpdf_spline(line, cdf, n_fine=50, spline=spline)
        np.testing.assert_allclose(out['pdf_max'], 1.0 / 0.5, rtol=1e-12)
        np.testing.assert_allclose(out['pdf_fine'], 1.0, rtol=1e-12)


def test_quantile_prior_charges_1000_where_the_pdf_is_zero():
    """SNANA's CHI2PRIOR is -2 ln PDF where the PDF is positive and 1000
    where it is not (snlc_fit.F90:4893), and the wall's derivatives are
    finite. The natural spline's overshoot on skewed quantiles leaves a
    stretch of zero PDF inside the support."""
    from jax_snlc_core.interpolation import compute_catmullrom_slopes
    from jax_snlc_fit.photoz import chi2_z_prior_quantile
    from jax_snlc_fit.photoz import precompute_zpdf_spline

    peaked = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.501, 0.502, 0.503, 0.504, 0.505, 0.9]])
    out = precompute_zpdf_spline(peaked, np.linspace(0.0, 1.0, 11), n_fine=400)
    z_fine, pdf = jnp.asarray(out['z_fine'][0]), jnp.asarray(out['pdf_fine'][0])
    slopes = compute_catmullrom_slopes(z_fine, pdf)

    def prior(z):
        return chi2_z_prior_quantile(z, z_fine, pdf, slopes)

    zero = np.flatnonzero(pdf == 0.0)
    z_wall = 0.5 * (z_fine[zero[1]] + z_fine[zero[2]])
    assert float(prior(z_wall)) == 1000.0
    assert np.isfinite(float(jax.grad(prior)(z_wall)))
    assert np.isfinite(float(jax.hessian(prior)(z_wall)))

    k = int(np.argmax(pdf))
    assert float(prior(z_fine[k])) == pytest.approx(-2.0 * np.log(float(pdf[k])), abs=1e-9)


# ---------------------------------------------------------------------------
# newton_lm
# ---------------------------------------------------------------------------


def test_newton_lm_solves_the_free_direction_at_a_bound():
    """The constrained minimum is (0, 0.1); clipping the unconstrained
    Newton step instead repeatedly overshoots the free coordinate."""
    h = jnp.array([[1.0, 0.9], [0.9, 1.0]])
    def loss(p):
        residual = p - jnp.array([-1.0, 1.0])
        return residual @ h @ residual

    p, _, _ = newton_lm(jnp.zeros(2), jnp.array([0.0, -10.0]),
                         jnp.full(2, 10.0), loss, 3, refine=True)
    np.testing.assert_allclose(p, [0.0, 0.1], atol=2e-6)


def test_newton_lm_releases_a_bound_when_the_gradient_changes():
    """The first x gradient points outwards, but moving y makes x free;
    the solution is the interior minimum, not x pinned at zero."""
    h = jnp.array([[1.0, 0.9], [0.9, 1.0]])
    def loss(p):
        residual = p - jnp.array([1.0, -2.0])
        return residual @ h @ residual

    p, _, _ = newton_lm(jnp.zeros(2), jnp.array([0.0, -3.0]),
                         jnp.full(2, 3.0), loss, 15, refine=True)
    np.testing.assert_allclose(p, [1.0, -2.0], atol=1e-4)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("upper_x", [3.0, 0.0])
def test_newton_lm_refinement_escapes_a_stationary_saddle(dtype, upper_x):
    """A zero gradient is not a minimum; an upper bound permits only the
    negative branch of the same saddle's downhill direction."""
    def loss(p):
        return (p[0] ** 2 - 1.0) ** 2 + p[1] ** 2

    p, value, _ = newton_lm(
        jnp.zeros(2, dtype), jnp.full(2, -3.0, dtype),
        jnp.array([upper_x, 3.0], dtype), loss, 15, refine=True)
    assert float(value) < 1e-8
    np.testing.assert_allclose(abs(float(p[0])), 1.0, atol=1e-6)
    np.testing.assert_allclose(p[1], 0.0, atol=1e-6)
    assert float(p[0]) <= upper_x


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_newton_lm_makes_progress_after_rejected_fallbacks(dtype):
    """Off-diagonal curvature survives when the Hessian diagonals cancel.
    The scalar fallback repeatedly overshoots from the origin; the fit must
    make progress down the coupled valley within the production budget."""
    def loss(p):
        u, v = p[0] + p[1], p[0] - p[1]
        return 0.25 * u ** 4 - 0.5 * u ** 2 + 0.5 * v ** 2 - u

    p, _, _ = newton_lm(
        jnp.zeros(2, dtype), jnp.full(2, -10.0, dtype),
        jnp.full(2, 10.0, dtype), loss, 15, refine=True)
    assert float(loss(p)) < float(loss(jnp.array([0.5, 0.5], dtype)))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_newton_lm_makes_progress_after_rejected_newton_steps(dtype):
    """A descent Newton direction can overshoot a nonquadratic objective.
    Tiny damping increases repeat the rejection; the fit should reach below
    the feasible origin's loss within six steps despite negative curvature."""
    def loss(p):
        return 0.25 * p[0] ** 4 - 0.5 * p[0] ** 2 + jnp.exp(p[1]) - p[1]

    p, _, _ = newton_lm(
        jnp.array([0.1, -2.0], dtype), jnp.full(2, -3.0, dtype),
        jnp.full(2, 5.0, dtype), loss, 6, refine=True)
    assert float(loss(p)) < float(loss(jnp.zeros(2, dtype)))


@pytest.mark.parametrize("refine", [False, True])
def test_newton_lm_handles_no_active_parameters(refine):
    """An all-fixed fitter round passes an empty active vector under jit/vmap."""
    def solve(p):
        return newton_lm(p, p, p, lambda q: 7.0 + jnp.sum(q ** 2),
                         15, refine=refine)

    p, value, gradient = jax.jit(jax.vmap(solve))(jnp.empty((2, 0)))
    assert p.shape == (2, 0)
    np.testing.assert_array_equal(value, [7.0, 7.0])
    np.testing.assert_array_equal(gradient, [0.0, 0.0])


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_newton_lm_refinement_keeps_fixed_parameters_on_negative_curvature(dtype):
    """Curvature escape cannot move fixed coordinates even at a saddle."""
    p0 = jnp.zeros(2, dtype)
    def loss(p):
        return (p[0] ** 2 - 1.0) ** 2 - p[1]

    p, value, gradient = newton_lm(p0, p0, p0, loss, 15, refine=True)
    np.testing.assert_array_equal(p, p0)
    assert float(value) == 1.0
    # The returned gradient remains the raw maximum, not the projected norm.
    assert float(gradient) == 1.0


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_newton_lm_default_keeps_original_stationary_saddle_behavior(dtype):
    """Saddle escape is opt-in; shared live/Bazin/BBC calls keep their steps."""
    def loss(p):
        return (p[0] ** 2 - 1.0) ** 2 + p[1] ** 2

    p0 = jnp.zeros(2, dtype)
    p, value, gradient = newton_lm(
        p0, jnp.full(2, -3.0, dtype), jnp.full(2, 3.0, dtype), loss, 15)
    np.testing.assert_array_equal(p, p0)
    assert float(value) == 1.0
    assert float(gradient) == 0.0


def test_newton_lm_stopping_accounts_for_correlation():
    hessian = jnp.array([[1., .999999], [.999999, 1.]])
    loss = lambda p: .5 * p @ hessian @ p
    p, value, _ = newton_lm(jnp.array([800., -800.]),
                            jnp.full(2, -1e4), jnp.full(2, 1e4), loss, 60,
                            refine=True, tolerance=5e-5)
    assert value <= 5e-5
    np.testing.assert_allclose(value, loss(p), atol=1e-12)


def test_newton_lm_stopping_rejects_negative_curvature():
    loss = lambda p: p[0]**2 + (p[1]**2 - 1)**2
    p, value, _ = newton_lm(jnp.zeros(2), jnp.full(2, -3.),
                            jnp.full(2, 3.), loss, 60,
                            refine=True, tolerance=5e-5)
    assert value <= 5e-5
    np.testing.assert_allclose(jnp.abs(p[1]), 1., atol=.01)
