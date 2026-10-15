"""Redshift priors and the z-PDF spline the photo-z fit reads them from.

The chi2 the fitter minimises is not here: it is
`LightCurveFitter._sn_chi2`, which carries the model-error and sigma-floor
terms.
"""

import jax.numpy as jnp
import numpy as np

from jax_snlc_core.utils.error_handling import SNANAError
from jax_snlc_core.utils.interp import hermite_interp_1d


def zpdf_rows_usable(z_quantiles, n_quantiles=None):
    """Which events' quantiles the spline can take: at least two, strictly
    increasing across the row, a null marker included (init_spline aborts on
    fewer than two and returns an error code otherwise,
    sntools_spline.c:188-210). ``n_quantiles`` excludes padded columns."""
    q = np.asarray(z_quantiles)
    counts = (np.full(len(q), q.shape[1]) if n_quantiles is None
              else np.asarray(n_quantiles))
    used = np.arange(max(q.shape[1] - 1, 0)) < counts[:, None] - 1
    return (np.all((np.diff(q, axis=1) > 0) | ~used, axis=1)
            & (counts >= 2) & (counts <= q.shape[1]))


# events whose PDFs are evaluated at once
ZPDF_SPLINE_ROWS = 2048


def spline_knots(x, y):
    """The strictly increasing knots of a batch of splines, their spacing
    and the secant slopes between them, ``(x, h, secant)``: ``x`` [n,
    n_knots], ``y`` [n_knots] or [n, n_knots]. What the slope rules and the
    evaluation read."""
    x = np.asarray(x, dtype=np.float64)
    y = np.broadcast_to(np.asarray(y, dtype=np.float64), x.shape)
    h = np.diff(x, axis=1)
    if not np.all(h > 0):
        raise SNANAError("the redshift quantiles must increase strictly",
                         function_name="precompute_zpdf_spline")
    return x, h, np.diff(y, axis=1) / h


def natural_spline_slopes(knots):
    """The natural cubic spline's first derivative at every knot: its
    second derivatives at the interior knots solve a tridiagonal system with
    zero at both ends (GSL's cspline)."""
    x, h, slope = knots
    n, n_knots = x.shape
    k = n_knots - 2
    system = np.zeros((n, k, k))
    rows = np.arange(k)
    system[:, rows, rows] = 2.0 * (h[:, :-1] + h[:, 1:])
    system[:, rows[:-1], rows[1:]] = h[:, 1:-1]
    system[:, rows[1:], rows[:-1]] = h[:, 1:-1]
    second = np.zeros((n, n_knots))
    second[:, 1:-1] = np.linalg.solve(
        system, 6.0 * np.diff(slope, axis=1)[..., None])[..., 0]
    left = slope - h * (2.0 * second[:, :-1] + second[:, 1:]) / 6.0
    last = slope[:, -1:] + h[:, -1:] * (second[:, -2:-1] + 2.0 * second[:, -1:]) / 6.0
    return np.concatenate([left, last], axis=1)


def steffen_slopes(knots):
    """Steffen's monotone first derivative at every knot (GSL's steffen,
    Steffen 1990): the interior slope is the parabola's through the three
    knots, capped at twice the smaller neighbouring secant and zero where
    the secants change sign, and the end slopes are the end secants."""
    x, h, slope = knots
    parabola = ((slope[:, :-1] * h[:, 1:] + slope[:, 1:] * h[:, :-1])
                / (h[:, :-1] + h[:, 1:]))
    interior = ((np.sign(slope[:, :-1]) + np.sign(slope[:, 1:]))
                * np.minimum(np.abs(slope[:, :-1]),
                             np.minimum(np.abs(slope[:, 1:]),
                                        0.5 * np.abs(parabola))))
    return np.concatenate([slope[:, :1], interior, slope[:, -1:]], axis=1)


def hermite_spline_derivative(knots, knot_slopes, n_fine):
    """The derivative of the piecewise cubic through the ``knots`` with
    first derivative ``knot_slopes`` at them, on ``n_fine`` evenly spaced
    points over each row's knots. Returns ``(points, derivative)``, each
    [n, n_fine]."""
    x, h, slope = knots
    n, n_knots = x.shape
    points = np.linspace(x[:, 0], x[:, -1], n_fine, axis=1)
    step = (x[:, -1] - x[:, 0]) / (n_fine - 1)
    first = np.ceil((x[:, 1:-1] - x[:, :1]) / step[:, None]).astype(int)
    edges = np.concatenate([np.zeros((n, 1), int), np.clip(first, 0, n_fine),
                            np.full((n, 1), n_fine)], axis=1)
    counts = np.diff(edges, axis=1).ravel()               # [n * (n_knots-1)]

    def per_point(per_interval):
        return np.repeat(per_interval.ravel(), counts).reshape(n, n_fine)

    d_lo, d_hi = knot_slopes[:, :-1], knot_slopes[:, 1:]
    # y = d + c t + b t^2 + a t^3 on each interval, t from its left knot
    a = per_point((d_lo + d_hi - 2.0 * slope) / h ** 2)
    b = per_point((3.0 * slope - 2.0 * d_lo - d_hi) / h)
    t = points - per_point(x[:, :-1])
    return points, per_point(d_lo) + t * (2.0 * b + 3.0 * a * t)


def precompute_zpdf_spline(z_quantiles, cdf_percentiles, n_fine=1000,
                           spline='cubic', n_quantiles=None):
    """Precompute the fine-grid PDF for the quantile redshift prior, returning
    a dict of ``z_fine``, ``pdf_fine`` (PDF/PDF_max), ``pdf_max``, ``pdf_mean``
    and ``pdf_std``, each [n_sne] or [n_sne, n_fine].
    ``n_quantiles`` excludes padded columns when rows have different counts.

    The PDF is the derivative of the ``spline`` through the CDF at
    ``z_quantiles`` ([n_sne, n_q] or [n_q], the redshift at each of the
    ``cdf_percentiles``, e.g. ``[0.0, 0.1, ..., 1.0]``), on a fine z-grid
    over each SN's quantile range; the fit Hermite-interpolates it at
    runtime for correct AD gradients and Hessians. The splines are the
    interpolants SNANA reads the PDF from (init_spline, sntools_spline.c):
    GSL's cspline and its steffen.
    """
    knot_slopes = {'cubic': natural_spline_slopes,
                   'steffen': steffen_slopes}[spline]
    z_quantiles = np.atleast_2d(np.asarray(z_quantiles, dtype=np.float64))
    cdf_percentiles = np.broadcast_to(cdf_percentiles, z_quantiles.shape)
    n_sne = z_quantiles.shape[0]
    z_fine = np.empty((n_sne, n_fine))
    pdf = np.empty((n_sne, n_fine))
    counts = (np.full(n_sne, z_quantiles.shape[1]) if n_quantiles is None
              else np.asarray(n_quantiles))
    if not zpdf_rows_usable(z_quantiles, counts).all():
        raise SNANAError("each redshift PDF needs at least two increasing quantiles",
                         function_name="precompute_zpdf_spline")
    for count in np.unique(counts):
        indices = np.flatnonzero(counts == count)
        for start in range(0, len(indices), ZPDF_SPLINE_ROWS):
            rows = indices[start:start + ZPDF_SPLINE_ROWS]
            knots = spline_knots(z_quantiles[rows, :count],
                                 cdf_percentiles[rows, :count])
            z_fine[rows], pdf[rows] = hermite_spline_derivative(
                knots, knot_slopes(knots), n_fine)
    pdf = np.maximum(pdf, 0.0)  # sntools_zPDF_spline.c:145
    pdf_max = np.max(pdf, axis=1)
    pdf_max = np.where(pdf_max > 0, pdf_max, 1.0)
    pdf_fine = pdf / pdf_max[:, None]

    # fine-grid mean/std; SNANA's integrate the same spline over 100 bins
    # (NBIN_SPLINE_GEN, sntools_spline.h:70)
    pdf_mean = np.average(z_fine, weights=pdf_fine, axis=1)
    pdf_std = np.sqrt(np.average((z_fine - pdf_mean[:, None])**2, weights=pdf_fine, axis=1))

    return {
        'z_fine': z_fine,
        'pdf_fine': pdf_fine,
        'pdf_max': pdf_max,
        'pdf_mean': pdf_mean,
        'pdf_std': pdf_std,
    }


def z_prior_sigma(z_prior_err, zerrscale):
    """PRIOR_ZPULL's width, protected against a zero error then scaled.

    SNANA adds the floor in quadrature before applying PRIOR_ZERRSCALE
    (snlc_fit.F90:12598), so the scale reaches a zero-error event too.
    """
    return jnp.sqrt(z_prior_err ** 2 + 1.0e-6) * zerrscale


def chi2_z_prior_gaussian(z_fit, z_prior, z_prior_err, zerrscale):
    """Gaussian redshift prior chi2, SNANA PRIOR_ZPULL (snlc_fit.F90:12610)."""
    return ((z_fit - z_prior) / z_prior_sigma(z_prior_err, zerrscale)) ** 2


def chi2_z_prior_quantile(z_fit, z_fine, pdf_fine, pdf_slopes, pdf_max):
    """Photo-z PDF quantile prior chi2 via Hermite-interpolated PDF.

    Matches SNANA eval_zPDF_spline (sntools_zPDF_spline.c):
        1. Cubic spline (natural or Steffen) fit to CDF(z) from quantile
           data (at setup)
        2. PDF evaluated on fine z-grid from spline derivative (at setup)
        3. Runtime: Catmull-Rom Hermite interpolation of PDF on fine grid
        4. chi2 = -2 * log(PDF / PDF_max), or 1000 where the PDF is not
           positive (snlc_fit.F90:4893)

    The fine-grid Hermite interpolation is C1-continuous, so the AD
    gradients and Hessians are exact inside the PDF's support.

    ``z_fine`` and ``pdf_fine`` (PDF/PDF_max) are the fine grid
    :func:`precompute_zpdf_spline` returns and ``pdf_slopes`` their
    Catmull-Rom slopes, each shape ``(n_fine,)``.
    ``pdf_max`` is unused: ``pdf_fine`` already carries the normalisation.
    Returns the scalar chi2 contribution.
    """
    outside = (z_fit < z_fine[0]) | (z_fit > z_fine[-1])
    z_clipped = jnp.clip(z_fit, z_fine[0], z_fine[-1])
    pdf_norm = hermite_interp_1d(z_clipped, z_fine, pdf_fine, pdf_slopes)
    wall = outside | (pdf_norm <= 0.0)
    # log(1) where discarded, so the wall's derivatives are zero, not NaN
    return jnp.where(wall, 1000.0, -2.0 * jnp.log(jnp.where(wall, 1.0, pdf_norm)))


