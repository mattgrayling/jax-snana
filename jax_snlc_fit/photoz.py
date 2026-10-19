"""Photo-z steps of :class:`LightCurveFitter`'s fit.

When ``config.photoz.dofit_photoz`` is set, redshift becomes a free fit
parameter, the last of the fit vector, alongside the model parameters. The
fit itself is the fitter's one round loop; this module holds what only a
photo-z fit does:

- :func:`redshift_priors` is every SN's redshift prior as the fit reads
  it, and :func:`scan_size` sizes the coarse z grid from the whole
  sample's so all batches share a single compiled scan.
- :func:`coarse_z_grids` and :func:`coarse_grid_axes` are the axes of the
  coarse grid search that seeds the fit.
- :func:`iter1_band_mask`, :func:`round1_band_set` and
  :func:`dropped_epochs` are PHOTOZ_ANA's band set: round 1's
  PHOTOZ_ITER1_LAMRANGE, then the margin test at the fitted z (SNANA
  PHOTOZ_DROPFILTER) over each band's :func:`band_z_ranges`.

Photo-z with non-SALT models (the generic path, e.g. BayeSN) is an
extension beyond SNANA: snlc_fit.exe hard-stops in INIT_COURSEBIN_PHOTOZ
for the BAYESN model, so no SNANA reference exists and only SALT photo-z
carries replay-parity coverage.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core.physics.cosmology import reference_distance_modulus
from jax_snlc_core.utils import SNANA_NULLFLOAT, ZSPEC_SOURCE_MASK
from jax_snlc_core.utils.interp import compute_catmullrom_slopes, hermite_interp_1d
from .loss_functions import precompute_zpdf_spline

# SNANA MXRATIO_INIVAL_ZPHOT (snlc_fit.F90:8568)
LZDONE_FRACTIONAL_THRESHOLD = 0.05

# a best-z error below this replaces PHOTODZ_REJECT (snlc_fit.F90:8448)
PHOTODZ_REJECT_ZERR_MAX = 0.05

# NBIN = int(42*(PARMAX-PARMIN)) + 1 (snlc_fit.F90:9452)
PHOTOZ_GRID_DENSITY = 42.0


def photoz_zot_window(cfg, z_prior, z_prior_err, lzdone, xp):
    """Per-SN (lo, hi) of the coarse z grid, in z/(1+z).

    RANGEz_COURSE_PHOTOZ overrides the prior-derived window when its lower edge
    is positive (snlc_fit.F90:9436); a pinned SN keeps its delta function,
    inside PHOTOZ_BOUND (snlc_fit.F90:8459-8467).
    Shared by the sizing pass and the search itself, so the two cannot drift:
    ``xp`` is numpy for the former and jax.numpy for the latter.
    """
    lo_b, hi_b = cfg.photoz.photoz_bound
    z_pin = xp.clip(z_prior, lo_b, hi_b)
    if cfg.photoz.rangez_course[0] > 0:
        lo_b, hi_b = cfg.photoz.rangez_course
        sn_z_lo = xp.where(lzdone, z_pin, xp.full_like(z_prior, lo_b))
        sn_z_hi = xp.where(lzdone, z_pin, xp.full_like(z_prior, hi_b))
    elif cfg.photoz.photoz_grid_fullrange and not cfg.photoz.has_photoz_host_gaussian():
        sn_z_lo = xp.where(lzdone, z_pin, xp.full_like(z_prior, lo_b))
        sn_z_hi = xp.where(lzdone, z_pin, xp.full_like(z_prior, hi_b))
    else:
        sn_z_lo = xp.where(lzdone, z_pin,
                           xp.maximum(z_prior - 2.0 * z_prior_err, lo_b))
        sn_z_hi = xp.where(lzdone, z_pin,
                           xp.minimum(z_prior + 2.0 * z_prior_err, hi_b))
    return sn_z_lo / (1.0 + sn_z_lo), sn_z_hi / (1.0 + sn_z_hi)


def photoz_nz_per_sn(cfg, zot_lo, zot_hi, xp):
    """Coarse z-bin count per SN: NBINz overrides the width-derived count, then
    SCALE_NBIN multiplies it (snlc_fit.F90:9444-9455)."""
    if cfg.photoz.nbinz_course > 0:
        n_z = xp.full(zot_lo.shape, cfg.photoz.nbinz_course, dtype=int)
    else:
        n_z = (xp.floor(PHOTOZ_GRID_DENSITY
                        * (zot_hi - zot_lo)).astype(int) + 1)
    return xp.maximum(n_z * cfg.photoz.scale_nbin_course, 1)


# points on the distance priors' ln z grid; the interpolated reference
# distance stays within 1e-6 mag of the exact one across PHOTOZ_BOUND
DISTANCE_PRIOR_NZ = 4000
# SNANA's correlation between the W0_REF and WA_REF uncertainties in the
# PRIOR_MUERRSCALE width (PRIOR_MUPULL, snlc_fit.F90:12610)
RHO_W0WA = -0.8


def distance_prior_tables(config):
    """The distance priors' reference distance modulus and the
    PRIOR_MUERRSCALE width on a grid uniform in ln z across PHOTOZ_BOUND,
    with their Hermite slopes, so the fit interpolates rather than integrating
    the cosmology at every step; None without a distance prior.

    The width is how far the reference distance moves when OMAT_REF, W0_REF
    and WA_REF each shift by their uncertainty, added in quadrature with the
    W0_REF-WA_REF correlation; OLAM_REF stays put and H0_REF's uncertainty is
    not used (PRIOR_MUPULL, snlc_fit.F90:12544-12621).
    """
    photoz, fit = config.photoz, config.fit
    if photoz.prior_muerrscale is None and photoz.prior_dmu_range is None:
        return None
    ln_z = jnp.linspace(*np.log(photoz.photoz_bound), DISTANCE_PRIOR_NZ)

    def distance(omat=fit.omat_ref[0], w0=fit.w0_ref[0], wa=fit.wa_ref[0]):
        return reference_distance_modulus(
            jnp.exp(ln_z), fit.h0_ref[0], omat, fit.olam_ref[0], w0, wa)

    mu = distance()
    d_omat = distance(omat=fit.omat_ref[0] + fit.omat_ref[1]) - mu
    d_w0 = distance(w0=fit.w0_ref[0] + fit.w0_ref[1]) - mu
    d_wa = distance(wa=fit.wa_ref[0] + fit.wa_ref[1]) - mu
    sigma = jnp.sqrt(d_omat ** 2 + d_w0 ** 2 + d_wa ** 2 + 2.0 * RHO_W0WA * d_w0 * d_wa)
    return {'ln_z': ln_z,
            'mu': mu, 'mu_slopes': compute_catmullrom_slopes(ln_z, mu),
            'sigma': sigma, 'sigma_slopes': compute_catmullrom_slopes(ln_z, sigma)}


# no-host STD and the DOFIT_PHOTOZ_HOST gate (snlc_fit.F90:8522, :13961)
PHOTOZ_STD_NOHOST = 9999.0
PRIOR_ZERRSCALE_MAX = 30.0


class RedshiftPriors(NamedTuple):
    """Every SN's redshift prior as the fit reads it: the start, the mean,
    the step (SNANA's STD, snlc_fit.F90:8501, which sizes the coarse grid's
    window and LZDONE), PRIOR_ZPULL's Gaussian width, whether a host prior
    applies, whether to use host quantiles, and their splined PDF
    (None when no event uses quantiles)."""
    z_init: jnp.ndarray
    z_prior: jnp.ndarray
    z_prior_err: jnp.ndarray
    z_prior_width: jnp.ndarray
    host_ok: jnp.ndarray
    use_quantiles: jnp.ndarray
    pdf_grid: jnp.ndarray
    pdf_vals: jnp.ndarray
    pdf_slopes: jnp.ndarray
    pdf_max: jnp.ndarray


def is_zspec(config, batch, n_sne):
    """Per-SN IS_ZSPEC (snana.F90:17896)."""
    zerr = np.asarray(batch.redshift_errs)
    small = ((zerr < config.photoz.quantile_zerrmin) & (zerr >= 0.0)
             if config.photoz.quantile_zerrmin > 0.0
             else np.zeros(n_sne, dtype=bool))
    if batch.redshift_source_mask is None:
        return small
    return small | ((np.asarray(batch.redshift_source_mask).astype(int)
                     & ZSPEC_SOURCE_MASK) > 0)


def quantile_prior_mask(config, batch):
    """Events using host quantiles rather than a spectroscopic Gaussian."""
    use = np.full(batch.n_sne, config.photoz.dofit_photoz
                  and config.photoz.has_photoz_quantiles(), dtype=bool)
    if config.photoz.has_photoz_bestz_gaussian():
        use &= ~is_zspec(config, batch, batch.n_sne)
    return use


def photoz_host_mask(config, batch, n_sne):
    """Per-SN DOFIT_PHOTOZ_HOST (snlc_fit.F90:13961)."""
    if config.photoz.prior_zerrscale >= PRIOR_ZERRSCALE_MAX:
        return jnp.zeros(n_sne, dtype=bool)
    has_z = (jnp.asarray(batch.photoz) > 0.0 if batch.photoz is not None
             else jnp.zeros(n_sne, dtype=bool))
    if batch.zpdf_quantiles is not None:
        has_z = has_z | jnp.any(jnp.asarray(batch.zpdf_quantiles) > 0.0,
                                axis=1)
    return has_z


def redshift_priors(config, batch) -> RedshiftPriors:
    """The batch's redshift priors, by OPT_PHOTOZ: bit 2 (value 4) the host
    quantile PDF, bit 1 (value 2) REDSHIFT_FINAL as a Gaussian, bit 0
    (value 1) the host photo-z. An event without the redshift its prior needs
    starts at the middle of PHOTOZ_BOUND with an uninformative width, as a
    host-less event does in SNANA (snlc_fit.F90:8517-8523): it is fitted from
    its light curve alone, and zPRIORERR reports that width. Under
    REDSHIFT_FINAL, SNANA instead starts the fit at the missing value. Only
    the host branch has a per-SN host check."""
    n_sne = batch.n_sne
    use_quantiles = quantile_prior_mask(config, batch)
    quantiles = use_quantiles.any()
    if quantiles:
        # Only rows actually using a host PDF need valid quantiles. Spec-z
        # rows can have missing quantiles or a host PDF far from their zspec.
        percentiles = np.asarray(batch.zpdf_percentiles) / 100.0
        if percentiles.ndim == 2:
            percentiles = percentiles[use_quantiles]
        spline_data = precompute_zpdf_spline(
            np.asarray(batch.zpdf_quantiles)[use_quantiles],
            percentiles,
            spline=config.photoz.quantile_spline(),
            n_quantiles=(None if batch.zpdf_nbin is None else
                         np.asarray(batch.zpdf_nbin)[use_quantiles]))
        # Keep rectangular arrays for batching; unused rows have a finite
        # placeholder PDF and are evaluated with the analytic Gaussian.
        def expand(values):
            out = np.broadcast_to(values[0], (n_sne, *values.shape[1:])).copy()
            out[use_quantiles] = values
            return out

        z_fine_np = expand(spline_data['z_fine'])
        pdf_grid = jnp.asarray(z_fine_np)
        pdf_vals_np = expand(spline_data['pdf_fine'])
        pdf_max_np = expand(spline_data['pdf_max'])
        pdf_vals = jnp.asarray(pdf_vals_np)
        pdf_max = jnp.asarray(pdf_max_np)

        # SNHOST_QZPHOT_MEAN/STD from the splined PDF
        pdf_fine_np = pdf_vals_np * pdf_max_np[:, None]
        dz = np.diff(z_fine_np, axis=1)
        pdf_mid = (pdf_fine_np[:, :-1] + pdf_fine_np[:, 1:]) / 2
        z_mid = (z_fine_np[:, :-1] + z_fine_np[:, 1:]) / 2
        norm = np.maximum(np.sum(pdf_mid * dz, axis=1), 1e-30)
        zpdf_mean = np.sum(z_mid * pdf_mid * dz, axis=1) / norm
        zpdf_var = np.sum((z_mid - zpdf_mean[:, None])**2 * pdf_mid * dz, axis=1) / norm
        zpdf_std = np.sqrt(np.maximum(zpdf_var, 1e-10))
        zpdf_mean = np.where(use_quantiles, zpdf_mean, np.asarray(batch.redshifts))
        zpdf_std = np.where(use_quantiles, zpdf_std, np.asarray(batch.redshift_errs))
        pdf_slopes = jax.vmap(compute_catmullrom_slopes)(pdf_grid, pdf_vals)
    else:
        pdf_grid = pdf_vals = pdf_slopes = pdf_max = None

    host_ok = jnp.ones(n_sne, dtype=bool)
    lo_b, hi_b = config.photoz.photoz_bound
    if quantiles:
        z_init = jnp.array(zpdf_mean)
        z_prior = jnp.array(zpdf_mean)
        z_prior_err = jnp.array(zpdf_std)
        z_prior_width = z_prior_err
    elif config.photoz.has_photoz_bestz_gaussian():
        has_z = jnp.asarray(batch.redshifts) >= 0
        z_init = jnp.where(has_z, batch.redshifts, 0.5 * (lo_b + hi_b))
        z_prior = z_init
        z_prior_err = jnp.where(has_z, batch.redshift_errs, PHOTOZ_STD_NOHOST)
        z_prior_width = z_prior_err
    else:
        # no host photo-z column is no host for any event (SNHOST_ZPHOT = -9)
        host_ok = photoz_host_mask(config, batch, n_sne)
        z_host = (jnp.array(batch.photoz) if batch.photoz is not None
                  else jnp.full(n_sne, SNANA_NULLFLOAT))
        z_host_err = (jnp.array(batch.photoz_err)
                      if batch.photoz_err is not None
                      else jnp.full(n_sne, SNANA_NULLFLOAT))
        z_init = jnp.where(host_ok, z_host, 0.5 * (lo_b + hi_b))
        z_prior = z_init
        z_prior_err = jnp.where(host_ok, z_host_err, PHOTOZ_STD_NOHOST)
        z_prior_width = z_prior_err

    # the -9 sentinel takes REDSHIFT_FINAL
    sentinel = z_init < 0
    z_init = jnp.where(sentinel, batch.redshifts, z_init)
    z_prior = jnp.where(sentinel, batch.redshifts, z_prior)
    z_prior_err = jnp.where(sentinel, batch.redshift_errs, z_prior_err)
    z_prior_width = jnp.where(sentinel, batch.redshift_errs, z_prior_width)
    return RedshiftPriors(z_init, z_prior, z_prior_err, z_prior_width,
                          host_ok, jnp.asarray(use_quantiles),
                          pdf_grid, pdf_vals, pdf_slopes, pdf_max)


def scan_size(config, z_priors):
    """The coarse z-grid length covering the widest window in a sample, from
    its :func:`redshift_priors`.

    Sized from every SN before batching, so every batch shares one compiled
    scan; a subset could size the grid short.
    """
    z_prior = np.asarray(z_priors.z_prior)
    if not len(z_prior):
        return 1
    z_prior_err = np.asarray(z_priors.z_prior_err)
    lzdone = z_prior_err / z_prior < LZDONE_FRACTIONAL_THRESHOLD
    zot_lo, zot_hi = photoz_zot_window(config, z_prior, z_prior_err, lzdone,
                                       np)
    return int(np.max(photoz_nz_per_sn(config, zot_lo, zot_hi, np)))


def coarse_grid_axes(config, fit_parameters):
    """The axes the coarse grid searches, one per parameter declaring a
    coarse grid, the input's range and bin count taking precedence and
    SCALE_NBIN_COURSE_PHOTOZ multiplying every axis (snlc_fit.F90:9455).
    With NBINT_COURSE_PHOTOZ above 1, a PEAKMJD axis of offsets from each
    event's estimate joins them (snlc_fit.F90:9352-9364)."""
    scale = config.photoz.scale_nbin_course
    axes = {}
    if config.photoz.nbint_course > 1:
        lo, hi = config.photoz.ranget_course
        axes['PKMJD'] = jnp.linspace(lo, hi, config.photoz.nbint_course * scale)
    for name, spec in fit_parameters.items():
        if 'coarse_grid' not in spec:
            continue
        grid = {**spec['coarse_grid'], **config.model.coarse_grid.get(name, {})}
        lo, hi = grid['range']
        axes[name] = jnp.linspace(lo, hi, grid['nbin'] * scale)
    return axes


def redshift_bounds(cfg, z_priors):
    """Intersect fitting limits with each quantile PDF's interpolation domain."""
    low, high = cfg.photoz.photoz_bound
    lower = jnp.full_like(z_priors.z_init, low)
    upper = jnp.full_like(z_priors.z_init, high)
    if cfg.model.opt_prior > 0 and z_priors.pdf_grid is not None:
        lower = jnp.where(z_priors.use_quantiles,
                          jnp.maximum(lower, z_priors.pdf_grid[:, 0]), lower)
        upper = jnp.where(z_priors.use_quantiles,
                          jnp.minimum(upper, z_priors.pdf_grid[:, -1]), upper)
    return lower, upper


def pdf_support_mask(cfg, z_priors):
    """Whether each enabled host PDF has positive density inside PHOTOZ_BOUND."""
    if cfg.model.opt_prior <= 0 or z_priors.pdf_grid is None:
        return jnp.ones_like(z_priors.z_init, dtype=bool)
    lower, upper = redshift_bounds(cfg, z_priors)
    grid, values = z_priors.pdf_grid, z_priors.pdf_vals
    inside = (grid >= lower[:, None]) & (grid <= upper[:, None])
    positive = jnp.any(inside & (values > 0), axis=1)
    # Bounds can fall between knots, including a window narrower than one
    # grid interval. Evaluate its edges using the objective's interpolation.
    edges = jax.vmap(hermite_interp_1d)(
        jnp.stack((lower, upper), axis=1), grid, values, z_priors.pdf_slopes)
    positive = positive | jnp.any(edges > 0, axis=1)
    return ~z_priors.use_quantiles | ((lower < upper) & positive)


def coarse_z_grids(cfg, n_z, z_priors):
    """Each event's coarse redshift grid, ``n_z`` long, and which of its
    points are real ([n_sne, n_z] each): the window
    :func:`photoz_zot_window` gives, in equal steps of z/(1+z), with as many
    points as :func:`photoz_nz_per_sn` asks for the event and the rest
    padding (snlc_fit.F90:9436-9461)."""
    # SNANA LZDONE (snlc_fit.F90:8569)
    lzdone = (z_priors.z_prior_err / z_priors.z_prior
              < LZDONE_FRACTIONAL_THRESHOLD)
    zot_lo, zot_hi = photoz_zot_window(cfg, z_priors.z_prior,
                                       z_priors.z_prior_err, lzdone, jnp)
    lower, upper = redshift_bounds(cfg, z_priors)
    domain_lo, domain_hi = lower / (1.0 + lower), upper / (1.0 + upper)
    zot_lo = jnp.clip(zot_lo, domain_lo, domain_hi)
    zot_hi = jnp.clip(zot_hi, domain_lo, domain_hi)
    n_z_sn = photoz_nz_per_sn(cfg, zot_lo, zot_hi, jnp).astype(jnp.int32)
    i_vals = jnp.arange(n_z)[None, :]
    # PARBIN = (PARMAX - PARMIN)/(NBIN - 1) (snlc_fit.F90:9461)
    spans = jnp.maximum(n_z_sn - 1, 1)[:, None]
    t_frac = jnp.where(n_z_sn[:, None] > 1, i_vals / spans, 0.0)
    zot = zot_lo[:, None] + t_frac * (zot_hi - zot_lo)[:, None]
    return zot / (1.0 - jnp.clip(zot, 0, 0.999)), i_vals < n_z_sn[:, None]


def band_z_ranges(restlambda, lam):
    """Each fit band's valid ``(z_min, z_max)``, each [n_filters]: the z range
    over which its SNANA-convention mean wavelength ``lam`` de-redshifts into
    ``restlambda``, the fit's RESTLAMBDA_USEFIT (snlc_fit.F90:3334-3335)."""
    restlam_lo, restlam_hi = restlambda
    return lam / restlam_hi - 1.0, lam / restlam_lo - 1.0


def margin_dropped_filters(
    config,
    restlambda,             # RESTLAMBDA_USEFIT
    lam,                    # [n_filters] mean wavelengths
    fitted_z,               # [n_sne]
    z_prior_err,            # [n_sne]
):
    """PHOTOZ_DROPFILTER as ``[n_sne, n_filters]``, True where dropped.

    A band is dropped when the fitted z sits within PHOTODZ_REJECT of either
    edge of its :func:`band_z_ranges` (snlc_fit.F90:3357-3389). Every fit
    band is tested, present in the light curve or not (:3355). In the best-z
    Gaussian mode FITINI_PHOTOZ replaces PHOTODZ_REJECT with the event's own
    z error when that is below a fixed 0.05 (:8448-8449); LFIXPAR_ALL skips
    FITINI_PHOTOZ (:8385).
    """
    z_min, z_max = band_z_ranges(restlambda, lam)
    z_min, z_max = z_min[None, :], z_max[None, :]
    dz_at_min = fitted_z[:, None] - z_min
    dz_at_max = z_max - fitted_z[:, None]

    if (config.photoz.has_photoz_bestz_gaussian()
            and not config.model.lfixpar_all):
        photodz = jnp.where(z_prior_err < PHOTODZ_REJECT_ZERR_MAX, z_prior_err,
                            config.photoz.photodz_reject)[:, None]
    else:
        photodz = config.photoz.photodz_reject
    dropped = (dz_at_min < photodz) | (dz_at_max < photodz)

    # PHOTODZ1Z_REJECT (snlc_fit.F90:3384)
    dz1z = config.photoz.photodz1z_reject
    if dz1z > -99.0:
        dropped = dropped | (dz_at_min / (1.0 + z_min) < dz1z) \
            | (dz_at_max / (1.0 + z_max) < dz1z)
    return dropped


def iter1_bands(config, lam):
    """Which bands round 1 of a photo-z fit uses: those whose mean wavelength
    ``lam`` is inside PHOTOZ_ITER1_LAMRANGE, which OPT_PHOTOZ=2 lowers to
    2000 A (snlc_fit.F90:1125)."""
    lo, hi = config.photoz.photoz_iter1_lamrange
    if config.photoz.opt_photoz == 2:
        lo = 2000.0
    return (lam >= lo) & (lam <= hi)


def iter1_band_mask(config, lam, filter_indices):
    """The epochs round 1 of a photo-z fit uses, of the bands
    :func:`iter1_bands` keeps."""
    return iter1_bands(config, lam)[filter_indices]


def round1_band_set(config, restlambda, lam, filter_indices, mask, physical,
                    z_priors):
    """PHOTOZ_ANA after round 1 (snlc_fit.F90:3350-3418): the epochs of
    ``mask`` in the bands the margin test keeps at the fitted redshift, and
    which events round 1 is repeated for (ERRFLAG=-1, :3388-3414), because
    the test dropped a band or a band round 1 held out fits at that
    redshift."""
    fitted_z = physical[:, -1]
    dropped = margin_dropped_filters(config, restlambda, lam, fitted_z,
                                     z_priors.z_prior_width)
    kept = mask & ~jnp.take_along_axis(dropped, filter_indices, axis=1)
    has_epochs = jnp.any(mask[:, :, None] & (
        filter_indices[:, :, None]
        == jnp.arange(lam.shape[0])[None, None, :]), axis=1)
    z_min, z_max = band_z_ranges(restlambda, lam)
    in_range = ((fitted_z[:, None] >= z_min[None, :])
                & (fitted_z[:, None] <= z_max[None, :]))
    added = (has_epochs & ~dropped & ~iter1_bands(config, lam)[None, :]
             & in_range)
    return kept, jnp.any(dropped | added, axis=1)


def dropped_epochs(config, restlambda, lam, filter_indices, physical,
                   z_priors, which):
    """The epochs of every band the margin test drops at the fitted redshift,
    for the events ``which`` selects (LTEST_ZPHFILTER,
    snlc_fit.F90:3569); ``lam`` is each band's mean wavelength."""
    dropped = margin_dropped_filters(config, restlambda, lam, physical[:, -1],
                                     z_priors.z_prior_width)
    dropped = dropped & jnp.reshape(which, (-1, 1))
    return jnp.take_along_axis(dropped, filter_indices, axis=1)
