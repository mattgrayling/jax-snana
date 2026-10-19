"""Numerical fit functions, independent of a LightCurveFitter instance.

LightCurveFitter._prepare_kernels creates the compilation snapshot in the
settings dictionary. These functions treat it as read-only. Repreparing a
fitter replaces its snapshot; existing compiled functions retain theirs.
Model and observation values are explicit arguments, allowing a run to reuse
compiled functions when their captured settings match.
"""

from functools import partial, wraps

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap

from jax_snlc_core.models.base import ratio_to_dmu
from jax_snlc_core.models.fitted import flux_error_from_fracerr
from jax_snlc_core.photometry.filter_system import mw_attenuation_rest
from jax_snlc_core.utils.interp import compute_catmullrom_slopes, hermite_interp_1d_batch
from jax_snlc_core.utils.newton_lm import newton_lm, _grad_and_hess
from jax_snlc_core.utils.precision import float_dtype

from . import photoz, priors, minos, minos_continuation
from .fit_spec import error_surface_overrides
from .flux_evaluator import band_weights as flux_band_weights, fit_flux, rest_band_weights
from .loss_functions import chi2_z_prior_gaussian, chi2_z_prior_quantile
from .uncertainties import errors_from_chi2, errors_from_derivatives

T0 = 0


def _static_data(settings, photometry, model=None):
    """What a model evaluates against: this fitter's static data and a
    data version's ``photometry``."""
    data = {**settings['static_data'], **photometry}
    if model is not None:
        data.update(host_rv=model.host_rv,
                    fluxcal_scale_mag_offset=model.fluxcal_scale_mag_offset)
    return data


def band_weights(settings, photometry, z, mw_att):
    """(flux weights, error weights) of one event at redshift ``z``, its
    Milky Way attenuation ``mw_att`` folded into the first
    (:func:`~jax_snlc_fit.flux_evaluator.band_weights`)."""
    return flux_band_weights(_static_data(settings, photometry), z, mw_att)


def events_band_weights(settings, photometry, z, mw_att):
    """:func:`band_weights` of every event, each at its own redshift in
    ``z`` [n_sne] and attenuation in ``mw_att`` [n_sne, n_common]."""
    return vmap(partial(band_weights, settings, photometry))(z, mw_att)


def z_grid_band_weights(settings, photometry):
    """The band weights over PHOTOZ_BOUND a photo-z fit interpolates, as
    (z grid, weights [n_z, n_filters * n_wave], their slopes in z)."""
    n_z = settings['config'].photoz.n_zgrid
    z_grid = jnp.linspace(*settings['config'].photoz.photoz_bound, n_z)
    static_data = _static_data(settings, photometry)
    flat = vmap(lambda z: rest_band_weights(static_data, z, None))(
        z_grid).reshape(n_z, -1)
    return z_grid, flat, compute_catmullrom_slopes(z_grid, flat)


def interpolated_band_weights(settings, photometry, z_weights, z, mw_att):
    """:func:`band_weights` at a fitted redshift, Catmull-Rom in the
    table :func:`z_grid_band_weights` made, ``z_weights``, so they are
    C1 in ``z``."""
    z_grid, flat, slopes = z_weights
    bw = hermite_interp_1d_batch(z[None], z_grid, flat, slopes)[0].reshape(
        settings['n_filters'], -1)
    mw_attenuation = mw_attenuation_rest(
        settings['static_data']['wavelength_grid'],
        photometry['common_filter_wave_grid'] / (1.0 + z), mw_att)
    return bw * mw_attenuation[None, :], bw


def model_flux(settings, model, photometry, params_arr, mjds, filter_indices, mask,
               band_weights, redshift, band_weights_dz=None,
               band_weights_dz2=None, *, static_data=None):
    """One event's model FLUXCAL at the fit vector ``params_arr``
    (:func:`~jax_snlc_fit.flux_evaluator.fit_flux`). Under photo-z the
    band weights' first two redshift derivatives come with them."""
    if static_data is None:
        static_data = _static_data(settings, photometry, model)
    return fit_flux(model.evaluate(
        model.sed_params_from_fit(params_arr, redshift), mjds,
        filter_indices, mask, band_weights, static_data,
        band_weights_dz=band_weights_dz,
        band_weights_dz2=band_weights_dz2), static_data['zp_fluxcal'])


def model_fracerr(settings, model, photometry, params_arr, mjds, filter_indices,
                  mask, band_weights, band_weights_err, redshift,
                  error_overrides=None, error_redshift=None,
                  band_weights_err_dz=None, *, static_data=None):
    """One event's model fractional error in nats, as the model computes
    it; ``error_overrides`` and ``error_redshift`` read the error surface
    at parameters or a redshift other than the trial ones."""
    if static_data is None:
        static_data = _static_data(settings, photometry, model)
    return model.compute_raw_fracerr(
        model.sed_params_from_fit(params_arr, redshift), mjds,
        filter_indices, mask, band_weights, static_data,
        error_overrides=error_overrides, error_redshift=error_redshift,
        band_weights_err=band_weights_err,
        band_weights_err_dz=band_weights_err_dz)


def model_flux_error(settings, flux, fracerr, mask):
    """The model error a measurement's chi2 divides by, from the model's
    ``flux`` and ``fracerr``, scaled by FUDGE_MODELERR_SCALE
    (snlc_fit.F90:11064).

    FUDGE_MAGERR_MODEL replaces it with a constant magnitude error,
    converted as SNANA does at snlc_fit.F90:4311: errfrac = 1 -
    10^(-0.4*MAG_ERR). It overrides USE_MODEL_MAGERR, which SNANA applies
    first (:5917).
    """
    magerr = settings['config'].model.fudge_magerr_model
    if magerr >= 0.0:
        return flux * (1.0 - 10.0 ** (-0.4 * magerr))
    return settings['config'].model.fudge_modelerr_scale * jnp.where(
        mask, flux_error_from_fracerr(flux, fracerr), 0.0)


def restlambda_mask(settings, photometry, filter_indices, redshifts):
    """Epochs whose band centre lands inside RESTLAMBDA_USEFIT once
    de-redshifted (FCNSNLC, snlc_fit.F90:4193).

    ``redshifts`` is one per SN alongside ``filter_indices``, or a grid of
    trial values broadcasting against it.
    """
    lo, hi = settings['restlambda_usefit']
    rest = (photometry['filter_mean_wavelengths'][filter_indices]
            / (1.0 + redshifts[:, None]))
    return (rest >= lo) & (rest <= hi)


def _redshifts(settings, params, scales, z_data):
    """The redshift a fit evaluates at: the fitted zPHOT under photo-z,
    else the data's."""
    if settings['config'].photoz.dofit_photoz:
        return params[:, -1] * scales[:, -1]
    return z_data


def _z_prior_chi2(settings, z, z_prior):
    """One event's redshift-prior chi2 at ``z``: SNANA's
    CHI2PRIOR(IPAR_zPHOT) (snlc_fit.F90:4903-4921), zero where OPT_PRIOR
    leaves it out (:4891)."""
    if settings['config'].model.opt_prior <= 0:
        return jnp.zeros(())
    def gaussian():
        return chi2_z_prior_gaussian(
            z, z_prior.z_prior, z_prior.z_prior_width,
            settings['config'].photoz.prior_zerrscale)

    if z_prior.pdf_grid is not None:
        return jax.lax.cond(
            z_prior.use_quantiles,
            lambda: chi2_z_prior_quantile(
                z, z_prior.pdf_grid, z_prior.pdf_vals,
                z_prior.pdf_slopes, z_prior.pdf_max),
            gaussian)
    return gaussian()


def _bands(settings, photometry, z_weights, z_data, bw, mw_attenuation):
    """One event's band weights as its chi2 reads them, a function of the
    physical parameters returning ``(z, bw, bw_err, (bw_dz, bw_err_dz,
    bw_dz2))``: ``bw`` at the data redshift with no derivatives, or under
    photo-z at the fitted redshift, interpolated in ``z_weights``, with
    their first two z-derivatives."""
    if not settings['config'].photoz.dofit_photoz:
        return lambda p: (z_data, *bw, (None, None, None))

    def at_fitted_z(p):
        z = p[settings['n_model_params']]

        def weights_dz(q):
            return jax.jvp(lambda r: interpolated_band_weights(
                settings, photometry, z_weights, r, mw_attenuation), (q,), (1.0,))
        ((bw, bw_err), (bw_dz, bw_err_dz)), (_, (bw_dz2, _)) = jax.jvp(
            weights_dz, (z,), (1.0,))
        return z, bw, bw_err, (bw_dz, bw_err_dz, bw_dz2)
    return at_fitted_z


def _sn_chi2(settings, model, strategy, photometry, p_scaled, start, scales, mjd, filter_indices,
             mask, fluxcals, fluxcalerrs, bands, prior_data, weights, z_prior):
    """One event's chi2 at ``p_scaled``, the parameters' priors included,
    and the redshift's under photo-z; ``bands`` are its band weights
    (:meth:`_bands`).

    ``strategy`` sets the weights. ``'live'`` (round 1) recomputes the
    model error with the trial parameters; ``'live_reweight'`` does too,
    rounds 2+ under USE_DIAG_ONLY; ``'frozen_diag'`` divides by the
    ``weights`` ``(1/sigma^2,)`` a round froze at its start and
    ``'frozen_woodbury'`` by the full covariance in its Woodbury form
    ``(1/sigma^2, d_inv, V, M_inv)``. ``start`` is where the round began.
    """
    cfg = settings['config']
    photo = cfg.photoz.dofit_photoz
    n_model = settings['n_model_params']
    p = p_scaled * scales
    # zero on the first iteration (FCNCHI2_SIGMA, snlc_fit.F90:5098)
    sigma_term = bool(cfg.fit.opt_chi2_sigma) and strategy != 'live'
    floor2 = cfg.fit.lm_sigma_floor ** 2

    z, bw, bw_err, (bw_dz, bw_err_dz, bw_dz2) = bands(p)
    model_params = p[:n_model]
    flux = model_flux(settings, model, photometry, model_params, mjd, filter_indices, mask, bw, z,
                           band_weights_dz=bw_dz, band_weights_dz2=bw_dz2)

    def model_error(**surface):
        return model_flux_error(settings, flux, model_fracerr(
            settings, model, photometry, model_params, mjd, filter_indices, mask, bw, bw_err, z,
            band_weights_err_dz=bw_err_dz, **surface), mask)

    if strategy == 'live':
        flux_err = model_error(error_overrides=error_surface_overrides(
            settings['fit_parameters'], first_round=True))
        chi2 = jnp.sum(jnp.where(mask, (flux - fluxcals) ** 2 / jnp.maximum(
            fluxcalerrs ** 2 + flux_err ** 2, floor2), 0.0))
    else:
        round_start = start * scales
        if strategy == 'live_reweight' or sigma_term:
            # SALT2xx1 freezes the error surface unless the spec-z sigma
            # term is on; SALT2zz evaluates it at the round's start z
            # (snlc_fit.F90:13792, :13843)
            freeze = not (cfg.fit.opt_chi2_sigma and not photo)
            flux_err = model_error(
                error_overrides=(error_surface_overrides(
                    settings['fit_parameters'], ref_params=round_start) if freeze else {}),
                error_redshift=round_start[n_model] if photo else None)
            s2 = jnp.maximum(fluxcalerrs ** 2 + flux_err ** 2, floor2)
        r = jnp.where(mask, flux - fluxcals, 0.0)
        if strategy == 'live_reweight':
            chi2 = jnp.sum(jnp.where(mask, r ** 2 / s2, 0.0))
        elif strategy == 'frozen_diag':
            chi2 = jnp.sum(weights[0] * r ** 2)
        else:
            _, d_inv, V, M_inv = weights
            dinv_r = d_inv * r
            cinv_r = dinv_r - d_inv * (V @ (M_inv @ (V.T @ dinv_r)))
            chi2 = jnp.sum(r * jnp.where(mask, cinv_r, 0.0))
        if sigma_term:
            # 2 ln(sigma/sigma_last) (FCNCHI2_SIGMA, snlc_fit.F90:5116)
            chi2 = chi2 + jnp.sum(
                jnp.log(jnp.where(mask, s2 * weights[0], 1.0)))

    chi2 = chi2 + priors.prior_chi2(model_params, settings['priors'], prior_data)
    if photo:
        chi2 = chi2 + _z_prior_chi2(settings, z, z_prior)
    return chi2


def _round(settings, model, *, strategy, photometry, z_weights, start, scales, lo, hi,
           mjd, filter_indices, mask, fluxcals, fluxcalerrs, z, bw,
           prior_data, weights, z_priors, mw_attenuation):
    """One round of the fit, every event its own Newton-LM from
    ``start``, each step clipped to its bounds ``lo``, ``hi``. Round 1
    (``strategy='live'``) also holds the parameters the model defers.
    Returns the parameters and their chi2. Photometry arrays have shape
    (events, epochs); parameter arrays have shape (events, parameters).
    """
    held = settings['round1_held_params'] if strategy == 'live' else settings['held_params']
    active = jnp.asarray([i for i, n in enumerate(settings['param_names'])
                          if n not in held], dtype=int)
    maxiter = (settings['config'].fit.maxiter_round1 if strategy == 'live'
               else settings['config'].fit.maxiter_round2)

    def one(p0, sc, lower, upper, sn_mjd, sn_filter_indices, sn_mask,
            sn_fluxcals, sn_fluxcalerrs, sn_z, sn_bw, sn_prior, sn_weights,
            sn_z_prior, sn_mw_attenuation):
        bands = _bands(settings, photometry, z_weights, sn_z, sn_bw,
                            sn_mw_attenuation)

        def chi2(p_active):
            return _sn_chi2(
                settings, model, strategy, photometry, p0.at[active].set(p_active), p0, sc,
                sn_mjd, sn_filter_indices, sn_mask, sn_fluxcals,
                sn_fluxcalerrs, bands, sn_prior, sn_weights, sn_z_prior)
        p_active, chi2_min, _ = newton_lm(
            p0[active], lower[active], upper[active], chi2, maxiter,
            refine=strategy in ('frozen_diag', 'frozen_woodbury'))
        return p0.at[active].set(p_active), chi2_min

    return vmap(one)(start, scales, lo, hi, mjd, filter_indices, mask,
                     fluxcals, fluxcalerrs, z, bw, prior_data, weights,
                     z_priors, mw_attenuation)


def _error_loss(settings, model, strategy, photometry, z_weights, data, *, fixed_z=None):
    """One event's final-round objective, with observations and priors fixed.

    A profile trial can hold its physical redshift and band weights while
    differentiating the other parameters. Endpoint checks use the full
    objective without this shortcut.
    """
    bands = _bands(settings, photometry, z_weights, data['z'], data['bw'],
                        data['mw_attenuation'])
    if fixed_z is not None:
        bw, bw_err = interpolated_band_weights(
            settings, photometry, z_weights, fixed_z, data['mw_attenuation'])
        bands = lambda _: (fixed_z, bw, bw_err, (None, None, None))
    return lambda p: _sn_chi2(
        settings, model, strategy, photometry, p, data['start'], data['scales'], data['mjd'],
        data['filter_indices'], data['mask'], data['fluxcals'],
        data['fluxcalerrs'], bands, data['prior_data'], data['weights'],
        data['z_priors'])


def _errors(settings, model, *, strategy, photometry, z_weights, **inputs):
    """Hessian errors on each event's unchanged final-round objective.

    Held parameters have zero covariance and no contribution to the EDM.
    Array shapes follow :meth:`_round`.
    """
    held = settings['round1_held_params'] if strategy == 'live' else settings['held_params']
    free = jnp.asarray([i for i, n in enumerate(settings['param_names'])
                        if n not in held], dtype=int)

    def one(data):
        loss = _error_loss(settings, model, strategy, photometry, z_weights, data)
        return errors_from_chi2(loss, data['params'], data['scales'], free)
    return vmap(one)(inputs)


def _minimum_errors(settings, model, *, strategy, photometry, z_weights, **inputs):
    """Evaluate the final centre and its exact constrained stationarity."""
    held = settings['round1_held_params'] if strategy == 'live' else settings['held_params']
    free = jnp.asarray([i for i, n in enumerate(settings['param_names'])
                        if n not in held], dtype=int)

    def one(data):
        p = data['params']
        loss = _error_loss(settings, model, strategy, photometry, z_weights, data)
        objective = lambda q: loss(p.at[free].set(q))
        lower, upper = data['lo'][free], data['hi'][free]
        fitted = p[free]
        g, H, value = _grad_and_hess(fitted, objective, with_value=True)
        errors = errors_from_derivatives(0.5 * H, 0.5 * g, data['scales'], free)
        valid = minos._remaining_decrease(fitted, lower, upper, H, g) <= 5e-5
        return (*errors, p.at[free].set(fitted), value, valid)
    return vmap(one)(inputs)


def _minos_errors(settings, model, *, strategy, photometry, z_weights, **inputs):
    """Shared bounded continuation for redshift or all free parameters."""
    return minos_continuation.profile_errors(
        settings, model, strategy=strategy, photometry=photometry, z_weights=z_weights,
        **inputs)


def _profile_indices(settings, held):
    """Indices to profile within the free-parameter subspace."""
    free = [n for n in settings['param_names'] if n not in held]
    if settings['config'].fit.minos == 'all':
        return np.arange(len(free), dtype=int)
    return np.array([i for i, name in enumerate(free) if name == 'zPHOT'], dtype=int)


def _colour_fraction(settings, model, photometry, filter_indices, z, mask):
    """Each epoch's colour-dispersion share of its model error at ``z``,
    the correlated part of the epoch covariance; zero without it."""
    if not settings['colour_dispersion']:
        return jnp.zeros(mask.shape)
    return model.color_dispersion_fraction(
        photometry['filter_mean_wavelengths'], filter_indices, z, mask)


def _frozen_weights(settings, model, photometry, z_weights, params, scales, mjd, filter_indices,
                    mask, fluxcals, fluxcalerrs, z_data, bw, kcor_frac, mw_attenuation,
                    mwxt_magerrs):
    """The weights a round 2+ fits against, from the parameters it starts
    at: ``(1/sigma^2,)``, or where the fit carries the full epoch
    covariance ``(1/sigma^2, d_inv, V, M_inv)`` with its Woodbury factors,
    built at the fitted redshift under photo-z as SNANA builds it at
    REDSHIFT_FIT (snlc_fit.F90:10588, :13911-13913). ``1/sigma^2`` is the
    full diagonal in both, the sigma term's sigma_last (JEP_FLUX_ERRTOT,
    :5100).

    The covariance is diagonal sigma^2 less the correlated part, plus a
    column per band for the colour dispersion (the k-correction share of
    the model error), one per band for FUDGE_MAG_COVERR and one for the
    Milky Way extinction error under OPT_COVAR_MWXTERR=1; =2 adds that
    error to the diagonal instead (snlc_fit.F90:11084). Its model error
    is the model's own under OPT_COVAR_FLUX, which FUDGE_MAGERR_MODEL
    turns off only at 0, and the fudged diagonal one without it
    (snlc_fit.F90:10668-10691).
    """
    cfg = settings['config']
    photo = cfg.photoz.dofit_photoz
    n_model, n_filters = settings['n_model_params'], settings['n_filters']
    floor2 = cfg.fit.lm_sigma_floor ** 2
    z = _redshifts(settings, params, scales, z_data)
    if photo:
        kcor_frac = _colour_fraction(settings, model, photometry, filter_indices, z, mask)
    coverr = (jnp.asarray(settings['fudge_mag_coverr_r'])
              if any(settings['fudge_mag_coverr_r']) else None)
    opt_mwxt = cfg.fit.opt_covar_mwxterr

    def one(p_scaled, sc, sn_mjd, sn_filter_indices, sn_mask, sn_fluxcals, sn_fluxcalerrs, sn_z,
            sn_bw, sn_mw_attenuation, sn_kcor_frac, sn_mwxt):
        p = p_scaled * sc
        bw, bw_err = (interpolated_band_weights(
            settings, photometry, z_weights, sn_z, sn_mw_attenuation) if photo else sn_bw)
        flux = model_flux(settings, model, photometry, p[:n_model], sn_mjd, sn_filter_indices,
                               sn_mask, bw, sn_z)
        fracerr = model_fracerr(settings, model, photometry, p[:n_model], sn_mjd,
                                     sn_filter_indices, sn_mask, bw, bw_err, sn_z)
        model_err = model_flux_error(settings, flux, fracerr, sn_mask)
        inv_sigma2 = jnp.where(sn_mask, 1.0 / jnp.maximum(
            sn_fluxcalerrs ** 2 + model_err ** 2, floor2), 0.0)
        if not settings['full_covariance']:
            return (inv_sigma2,)
        if not cfg.fit.use_offdiag_covariance:
            err_flux = model_err
        elif cfg.model.fudge_magerr_model == 0.0:
            err_flux = jnp.zeros_like(flux)
        else:
            # scaled by FUDGE_MODELERR_SCALE (snlc_fit.F90:10642)
            err_flux = (cfg.model.fudge_modelerr_scale
                        * jnp.maximum(jnp.abs(flux), 1e-30)
                        * (1.0 - jnp.exp(-fracerr)))
        kcor_ratio = jnp.minimum(
            sn_kcor_frac ** 2 / jnp.maximum(fracerr ** 2, 1e-30), 1.0)
        d = sn_fluxcalerrs ** 2 + err_flux ** 2 * (1.0 - kcor_ratio)
        # MW extinction error, flux units (snlc_fit.F90:11084)
        mwxt_err = jnp.where(
            sn_mask, (1.0 - 10.0 ** (-0.4 * sn_mwxt[sn_filter_indices])) * flux, 0.0)
        if opt_mwxt == 2:
            d = d + mwxt_err ** 2
        d_inv = jnp.where(sn_mask, 1.0 / jnp.maximum(d, floor2), 0.0)
        onehot = ((sn_filter_indices[:, None] == jnp.arange(n_filters)[None, :])
                  .astype(float_dtype()) * sn_mask[:, None])
        v = jnp.where(sn_mask,
                      err_flux * jnp.sqrt(jnp.maximum(kcor_ratio, 0.0)),
                      0.0)
        V = v[:, None] * onehot
        if coverr is not None:
            cov = jnp.where(sn_mask, coverr[sn_filter_indices] * sn_fluxcals, 0.0)
            V = jnp.concatenate([V, cov[:, None] * onehot], axis=1)
        if opt_mwxt == 1:
            V = jnp.concatenate([V, mwxt_err[:, None]], axis=1)
        k = V.shape[1]
        M = jnp.eye(k) + (V.T * d_inv[None, :]) @ V
        return inv_sigma2, d_inv, V, jnp.linalg.inv(M + 1e-12 * jnp.eye(k))

    return vmap(one)(params, scales, mjd, filter_indices, mask, fluxcals,
                     fluxcalerrs, z, bw, mw_attenuation, kcor_frac,
                     mwxt_magerrs)


def _reject_epochs(settings, model, photometry, mask, physical, mjd, filter_indices, fluxcals, fluxcalerrs,
                   reference_mjds, z, bw):
    """``mask`` less the epochs SNANA rejects after round 1 (FCNSNLC,
    snlc_fit.F90:4487-4516): a chi2 above DELCHI2_REJECT or S/N below
    SIGNOISE_REJECT at the physical parameters ``physical``, redshift
    ``z`` and band weights ``bw`` there, or an MJD outside FITWIN_MJD."""
    cfg = settings['config']
    if not cfg.fit.epoch_rejection_active():
        return mask
    # FITWIN_MJD is an absolute-MJD window (snlc_fit.F90:499, :4489)
    mjd_lo, mjd_hi = cfg.fit.fitwin_mjd
    flux, flux_err = _flux(settings, model, photometry,
                                physical[:, :settings['n_model_params']], mjd,
                                filter_indices, mask, *bw, z,
                                band_window=settings['static_data']['band_window'])
    # no floor on the errors, as in SNANA (snlc_fit.F90:4461-4469)
    chi2 = (flux - fluxcals) ** 2 / (fluxcalerrs ** 2 + flux_err ** 2)
    signoise = fluxcals / fluxcalerrs
    mjd_abs = mjd + reference_mjds[:, None]
    reject = ((chi2 > cfg.fit.delchi2_reject)
              | (signoise < cfg.fit.signoise_reject)
              | (mjd_abs < mjd_lo) | (mjd_abs > mjd_hi))
    return mask & ~reject


def _amplitude_ratio(settings, model, photometry, physical, mjd, filter_indices,
                     mask, fluxcals, fluxcalerrs, bw, bw_err, z, trest_range):
    """DATA_MODEL_RATIO averaged over the bands in the fit
    (snlc_fit.F90:6650, averaged at :9713)."""
    trest_lo, trest_hi = trest_range
    n_filt = bw.shape[1]
    overrides = error_surface_overrides(settings['fit_parameters'], first_round=True)
    flux = vmap(partial(model_flux, settings, model, photometry))(
        physical, mjd, filter_indices, mask, bw, z)
    fracerr = vmap(partial(model_fracerr, settings, model, photometry,
                           error_overrides=overrides))(
        physical, mjd, filter_indices, mask, bw, bw_err, z)
    model_err = jnp.where(
        mask, jnp.abs(flux) * (1.0 - jnp.exp(-fracerr)), 0.0)

    phases_rest = (mjd - physical[:, T0][:, None]) / (1.0 + z[:, None])
    ratio = fluxcals / flux
    ratio_err_sq = ratio ** 2 * ((fluxcalerrs / fluxcals) ** 2
                                 + (model_err / flux) ** 2)
    valid = (mask & (phases_rest >= trest_lo) & (phases_rest <= trest_hi)
             & (flux > 0.0) & (jnp.abs(flux) > 1e-21)
             & (ratio > 0.0) & (ratio_err_sq > 0.0))

    onehot = jax.nn.one_hot(filter_indices, n_filt)
    # padded epochs have zero flux: mask before dividing
    wsum = jnp.sum(jnp.where(valid[:, :, None],
                             onehot / ratio_err_sq[:, :, None], 0.0), axis=1)
    rsum = jnp.sum(jnp.where(valid[:, :, None],
                             onehot * (ratio / ratio_err_sq)[:, :, None],
                             0.0), axis=1)
    in_fit = jnp.sum(mask[:, :, None] * onehot, axis=1) > 0
    per_filt = jnp.where(wsum > 0, rsum / jnp.maximum(wsum, 1e-30), 1.0)
    per_filt = jnp.where(in_fit, per_filt, 0.0)
    n_use = jnp.sum(in_fit, axis=1)                    # [n_sne]
    avg = jnp.where(n_use > 0,
                    jnp.sum(per_filt, axis=1) / jnp.maximum(n_use, 1), 1.0)
    return jnp.clip(avg, 0.01, 100.0)


def _chi2red_ok(settings, window, chi2, n_obs):
    """One FITWIN_CHI2RED_INI verdict: FCNSNLC's chi2 per stored epoch,
    over NEPOCH_ALL(0) (snlc_fit.F90:9595)."""
    if not window:
        return None
    lo, hi = window
    chi2red = chi2 / jnp.maximum(n_obs, 1)
    return (chi2red >= lo) & (chi2red <= hi)


def _pre_fit_adjust(settings, model, photometry, z_weights, params, scales, mjd, filter_indices,
                    mask, fluxcals, fluxcalerrs, z_data, bw, n_obs, z_priors, mw_attenuation,
                    reference_mjds=0., limits=None):
    """FITINI_ADJUST: move the starting parameters towards the data
    (snlc_fit.F90:9471).

    The model declares the axes, where the t0 axis is offsets in days and
    every other axis absolute values; a photo-z fit searches its own,
    with the redshift where the coarse grid left it.

    Returns ``(adjusted, chi2red_ini_ok, chi2red_ini2_ok)``: SNANA windows
    CHI8INI before the adjustment (:9596) and CHI8END after it (:9803),
    and only then switches back to the starting values (:9818).
    """
    cfg = settings['config']
    # these inputs skip the adjustment and both windows (snlc_fit.F90:9553-9555)
    if (cfg.model.lfixpar_all or cfg.model.usesim_inival
            or cfg.fit._dofit_ideal):
        return params, None, None

    key = ('photoz_grid_search' if cfg.photoz.dofit_photoz
           else 'grid_search')
    search_params = [(name, jnp.asarray(spec[key]))
                     for name, spec in settings['fit_parameters'].items()
                     if key in spec]
    # INIVAL_GRIDSEARCH_<param> replaces that axis
    for name, (lo, hi, step) in cfg.model.inival_gridsearch.items():
        grid = jnp.arange(lo, hi + step * 0.5, step)
        search_params = [(n, grid if n == name else v)
                         for n, v in search_params]
        if name not in [n for n, _ in search_params]:
            search_params.append((name, grid))
    # held parameters stay put (snlc_fit.F90:9403)
    search_params = [(n, v) for n, v in search_params
                     if n not in settings['held_params']]
    adjust_amplitude = (model.distance_param is not None
                        and model.distance_param not in settings['held_params'])
    run_grid_search = bool(not cfg.fit.skip_grid_search and search_params)
    if not (run_grid_search or adjust_amplitude):
        return params, None, None

    z = _redshifts(settings, params, scales, z_data)
    # the amplitude ratio's, at the start's redshift
    bw_start = (events_band_weights(settings, photometry, z, mw_attenuation)
                if cfg.photoz.dofit_photoz else bw)

    # FCNSNLC at a starting point: model errors and priors included,
    # amplitude untouched (snlc_fit.F90:9617)
    def chi2_at(p):
        prior_data = priors.per_sn(p[:, T0] * scales[:, T0],
                                   cfg.fit.distance_modulus(z), reference_mjds)
        return vmap(lambda sn_p, sc, sn_mjd, sn_filter_indices, sn_mask,
                    sn_fluxcals, sn_fluxcalerrs, sn_z, sn_bw, sn_prior,
                    sn_z_prior, sn_mw_attenuation: _sn_chi2(
                        settings, model, 'live', photometry, sn_p, sn_p, sc, sn_mjd, sn_filter_indices,
                        sn_mask, sn_fluxcals, sn_fluxcalerrs,
                        _bands(settings, photometry, z_weights, sn_z, sn_bw,
                                    sn_mw_attenuation),
                        sn_prior, None, sn_z_prior))(
            p, scales, mjd, filter_indices, mask, fluxcals, fluxcalerrs, z_data, bw, prior_data,
            z_priors, mw_attenuation)

    chi2_ini = chi2_at(params)   # CHI8INI (:9594)
    chi2red_ini_ok = _chi2red_ok(
        settings, cfg.fit.fitwin_chi2red_ini, chi2_ini, n_obs)

    adjusted = params
    if run_grid_search:
        names = [name for name, _ in search_params]
        indices = [settings['param_names'].index(name) for name in names]
        flat_values = [g.ravel() for g in jnp.meshgrid(
            *[vals for _, vals in search_params], indexing='ij')]
        # [n_grid, n_sne, n_params]
        trial = jnp.broadcast_to(params[None], (len(flat_values[0]),)
                                 + params.shape).copy()
        for values, name, i in zip(flat_values, names, indices):
            step = values[:, None] / scales[None, :, i]
            trial = (trial.at[:, :, i].add(step) if name == 'PKMJD'
                     else trial.at[:, :, i].set(step))
        if limits is not None:
            trial = jnp.clip(trial, limits[0][None] / scales[None], limits[1][None] / scales[None])
        chi2_grid = vmap(chi2_at)(trial)               # [n_grid, n_sne]
        best = jnp.argmin(chi2_grid, axis=0)           # [n_sne]
        # the winner stands only where it beat the starting point (:9661)
        took = jnp.min(chi2_grid, axis=0) < chi2_ini
        for values, name, i in zip(flat_values, names, indices):
            best_val = trial[best, jnp.arange(params.shape[0]), i]
            moved = adjusted.at[:, i].set(best_val)
            adjusted = jnp.where(took[:, None], moved, adjusted)

    # One rescale, at whichever starting point survived (:9691-9747)
    if adjust_amplitude:
        ratio = _amplitude_ratio(
            settings, model, photometry, (adjusted * scales)[:, :settings['n_model_params']], mjd,
            filter_indices, mask,
            fluxcals, fluxcalerrs, *bw_start, z, model.grid_trest_range)
        adjusted = model.shift_amplitude(
            adjusted, ratio_to_dmu(ratio), scales, names=settings['param_names'])
        if limits is not None:
            adjusted = jnp.clip(adjusted, limits[0] / scales, limits[1] / scales)

    # before the revert (:9803 vs :9818)
    chi2_end = chi2_at(adjusted)
    chi2red_ini2_ok = _chi2red_ok(
        settings, cfg.fit.fitwin_chi2red_ini2, chi2_end, n_obs)

    # CHI8END > CHI8INI reverts all but t0 (:9818-9831)
    restored = params.at[:, T0].set(adjusted[:, T0])
    return (jnp.where((chi2_end > chi2_ini)[:, None], restored, adjusted),
            chi2red_ini_ok, chi2red_ini2_ok)


def _coarse_grid(settings, model, photometry, params, scales, mjd, filter_indices,
                 mask, fluxcals, fluxcalerrs, z_priors, mw_attenuation, limits=None):
    """The photo-z coarse grid over redshift and the model's own axes,
    ``n_z`` redshifts wide (:func:`~jax_snlc_fit.photoz.scan_size`):
    every event's parameters and redshift at its lowest-chi2 point.

    SNANA's INIPAR_PHOTOZ_COURSEGRID (snlc_fit.F90:8892) visits every
    grid point twice: first with the amplitude set from cosmology alone,
    which fills FFSUM_CROSS and FFSUM_MODEL over the SNR>3 epochs, then
    again with the amplitude rescaled by their ratio. The lowest chi2 of
    the second pass, plus the z prior, seeds the fit. That chi2 goes
    through FCNSNLC (:8931), so the grid's redshift prior is
    PRIOR_ZPULL's, LPULL gate included, while the window the grid spans
    and LZDONE read the step.
    """
    cfg = settings['config']
    static_data = _static_data(settings, photometry, model)
    z_grids, valid_z = photoz.coarse_z_grids(cfg, settings['n_z'], z_priors)
    if 'zPHOT' in settings['held_params']:
        z_grids = jnp.broadcast_to((params * scales)[:, -1, None], z_grids.shape)
        valid_z = jnp.ones_like(valid_z)
    elif limits is not None:
        z_grids = jnp.clip(z_grids, limits[0][:, -1, None], limits[1][:, -1, None])
    # [n_sne, n_z, n_filters, n_wave] each
    bw, bw_err = vmap(lambda zs, a: vmap(
        lambda z: band_weights(settings, photometry, z, a))(zs))(z_grids, mw_attenuation)
    # GET_DIST8 at unit scale (snlc_fit.F90:8918)
    mu_grid = cfg.fit.distance_modulus(z_grids.ravel()).reshape(
        z_grids.shape)

    axes = photoz.coarse_grid_axes(cfg, settings['fit_parameters'])
    axis_names = list(axes)
    axis_idx = [settings['param_names'].index(name) for name in axis_names]
    # pinned axes collapse to one bin at the per-SN start (snlc_fit.F90:9402)
    held_axes = [name in settings['held_params'] for name in axis_names]
    n_points = int(np.prod([1 if held else len(axes[name])
                            for name, held in zip(axis_names, held_axes)]))
    error_overrides = error_surface_overrides(settings['fit_parameters'], first_round=True)

    def one(sn_params, sn_scales, sn_mjd, sn_filter_indices, sn_mask,
            sn_fluxcals, sn_fluxcalerrs, sn_z_prior, sn_z_grid, sn_valid_z,
            sn_bw, sn_bw_err, sn_mu, sn_limits):
        sn_phys = sn_params * sn_scales
        sn_axes = {}
        for name, i, held in zip(axis_names, axis_idx, held_axes):
            if held:
                sn_axes[name] = sn_phys[i][None]
            elif i == T0:
                # offsets from the event's own PEAKMJD estimate
                sn_axes[name] = sn_phys[i] + axes[name]
            else:
                sn_axes[name] = axes[name]
        if sn_limits is not None:
            sn_axes = {name: jnp.clip(value, sn_limits[0][settings['param_names'].index(name)],
                                     sn_limits[1][settings['param_names'].index(name)])
                       for name, value in sn_axes.items()}
        axis_flat = [g.ravel() for g in jnp.meshgrid(
            *[sn_axes[name] for name in axis_names], indexing='ij')]
        keep = (sn_mask & restlambda_mask(settings, photometry, sn_filter_indices,
                                               sn_z_grid))[:, None, :]
        data_sqerr = sn_fluxcalerrs ** 2
        candidates = None
        if model.distance_param in settings['held_params'] or sn_limits is not None:
            # The grid's default amplitude comes from cosmology. A held
            # amplitude must instead be evaluated at its supplied value.
            def constrained_flux(z, mu, weights, errors):
                trial = jnp.broadcast_to(sn_phys, (n_points, sn_phys.shape[0]))
                for axis, index in zip(axis_flat, axis_idx):
                    trial = trial.at[:, index].set(axis)
                trial = trial.at[:, -1].set(z)
                if model.distance_param not in settings['held_params']:
                    trial = vmap(lambda p: model.params_at_distance(
                        p, mu, jnp.ones_like(p), names=settings['param_names']))(trial)
                if sn_limits is not None:
                    trial = jnp.clip(trial, *sn_limits)
                def evaluate(p):
                    return (model_flux(settings, model, photometry, p, sn_mjd,
                                sn_filter_indices, sn_mask, weights, z),
                            model_fracerr(settings, model, photometry, p, sn_mjd,
                                sn_filter_indices, sn_mask, weights, errors, z,
                                error_overrides=error_overrides))
                flux, fraction = vmap(evaluate)(trial)
                if model.distance_param is not None and model.distance_param not in settings['held_params']:
                    keep = sn_mask & restlambda_mask(settings, photometry, sn_filter_indices,
                                                     z[None])[0]
                    signal = keep & (sn_fluxcals > 3. * sn_fluxcalerrs)
                    variance = sn_fluxcalerrs**2 + flux_error_from_fracerr(flux, fraction)**2
                    cross = jnp.sum(jnp.where(signal, sn_fluxcals*flux/variance, 0.), axis=-1)
                    norm = jnp.sum(jnp.where(signal, flux**2/variance, 0.), axis=-1)
                    factor = jnp.where((cross > 0.) & (norm > 0.), cross/jnp.where(norm > 0., norm, 1.), 1.)
                    trial = model.shift_amplitude(trial, ratio_to_dmu(factor),
                                                 jnp.ones_like(trial), names=settings['param_names'])
                    if sn_limits is not None:
                        trial = jnp.clip(trial, *sn_limits)
                    flux, fraction = vmap(evaluate)(trial)
                return flux, fraction, trial
            flux, fracerr, candidates = vmap(constrained_flux)(sn_z_grid, sn_mu, sn_bw, sn_bw_err)
            model_err = jnp.where(
                flux > 0, flux_error_from_fracerr(flux, fracerr), 0.)
            ratio = jnp.ones(flux.shape[:-1])
        else:
            flux, model_err = model.evaluate_redshift_grid(
                sn_axes, sn_phys[:settings['n_model_params']], sn_mu, sn_z_grid,
                sn_bw, sn_bw_err, sn_mjd, sn_filter_indices, sn_mask, static_data,
                error_overrides=error_overrides)
            flux = fit_flux(flux.reshape(-1, n_points, flux.shape[-1]),
                            static_data['zp_fluxcal'])
            model_err = jnp.where(flux > 0.0, model_err.reshape(flux.shape), 0.0)

            ffsum = keep & (sn_fluxcals > 3.0 * sn_fluxcalerrs)
            sigma2 = data_sqerr + model_err ** 2
            cross = jnp.sum(jnp.where(ffsum, sn_fluxcals * flux / sigma2, 0.0),
                            axis=-1)
            model_sum = jnp.sum(jnp.where(ffsum, flux ** 2 / sigma2, 0.0),
                                axis=-1)
            # both sums positive or the amplitude stands (snlc_fit.F90:8933)
            rescale = (cross > 0.0) & (model_sum > 0.0)
            ratio = jnp.where(rescale,
                              cross / jnp.where(rescale, model_sum, 1.0), 1.0)
        sigma2 = data_sqerr + (model_err * ratio[..., None]) ** 2
        chi2 = jnp.sum(jnp.where(
            keep, (sn_fluxcals - flux * ratio[..., None]) ** 2 / sigma2, 0.0),
            axis=-1)
        prior = vmap(lambda z: _z_prior_chi2(settings, z, sn_z_prior))(sn_z_grid)
        chi2 = jnp.where(sn_valid_z[:, None], chi2 + prior[:, None], jnp.inf)

        iz, ip = jnp.unravel_index(jnp.argmin(chi2), chi2.shape)
        if candidates is not None:
            found = candidates[iz, ip] / sn_scales
            for name in settings['held_params']:
                index = settings['param_names'].index(name)
                found = found.at[index].set(sn_params[index])
            return found
        found = sn_params
        for k, i in enumerate(axis_idx):
            found = found.at[i].set(axis_flat[k][ip] / sn_scales[i])
        found = found.at[-1].set(sn_z_grid[iz] / sn_scales[-1])
        updated = model.params_at_distance(
            found, sn_mu[iz] + ratio_to_dmu(ratio[iz, ip]), sn_scales,
            names=settings['param_names'])
        for name in settings['held_params']:
            index = settings['param_names'].index(name)
            updated = updated.at[index].set(sn_params[index])
        return updated

    return vmap(one)(params, scales, mjd, filter_indices, mask, fluxcals, fluxcalerrs, z_priors,
                     z_grids, valid_z, bw, bw_err, mu_grid, limits)


def _flux(settings, model, photometry, physical, mjd, filter_indices, mask, bw, bw_err, z,
          *, band_window):
    """Every event's model flux and its error, ``(flux, flux_err)``."""
    static_data = {**_static_data(settings, photometry, model), 'band_window': band_window}

    def one(sn_physical, sn_mjd, sn_filter_indices, sn_mask, sn_bw, sn_bw_err, sn_z):
        flux = model_flux(settings, model, photometry, sn_physical, sn_mjd, sn_filter_indices,
                               sn_mask, sn_bw, sn_z, static_data=static_data)
        return flux, model_flux_error(settings, flux, model_fracerr(
            settings, model, photometry, sn_physical, sn_mjd, sn_filter_indices, sn_mask, sn_bw,
            sn_bw_err, sn_z, static_data=static_data), sn_mask)
    return vmap(one)(physical, mjd, filter_indices, mask, bw, bw_err, z)


def _same_settings(left, right):
    """Exact equality of captured settings, including prepared array values.

    Opaque resources such as a custom dust law compare by their own equality;
    distinct resources therefore do not accidentally share compiled code.
    """
    left_values, left_tree = jax.tree_util.tree_flatten(left)
    right_values, right_tree = jax.tree_util.tree_flatten(right)
    if left_tree != right_tree:
        return False
    for a, b in zip(left_values, right_values):
        if isinstance(a, (np.ndarray, jax.Array)) or isinstance(b, (np.ndarray, jax.Array)):
            a, b = np.asarray(a), np.asarray(b)
            if a.dtype != b.dtype or not np.array_equal(a, b):
                return False
        elif a != b:
            return False
    return True


def _with_captured_model(function, settings, model):
    """Keep ordinary models outside JAX, as with the original bound methods."""
    @wraps(partial(function, settings))
    def evaluate(unused_model, *args, **kwargs):
        return function(settings, model, *args, **kwargs)
    return evaluate


def compiled_functions(settings, cache, model):
    """Reuse functions within one run when all captured inputs are identical.

    The caller supplies a compilation snapshot and must not mutate it after
    this call. Both the cache and compiled functions retain its contents.
    The cache belongs to the caller, not to a process-wide registry. Model
    array values and photometry are arguments and do not participate in this
    lookup. Ordinary models are captured separately for each instance.
    """
    cache = cache.setdefault('functions', {})
    captured_model = model if settings['static_model'] else None
    error_function = _errors if settings['config'].fit.minos is None else _minos_errors
    functions = {
        'round': (_round, ('strategy',)),
        'weights': (_frozen_weights, ()),
        'errors': (error_function, ('strategy',)),
        'adjust': (_pre_fit_adjust, ()),
        'grid': (_coarse_grid, ()),
        'flux': (_flux, ('band_window',)),
        'reject_epochs': (_reject_epochs, ()),
    }
    compiled = {}
    for name, (function, static_names) in functions.items():
        # Only the coarse-grid function uses the scan length. A wider host
        # redshift prior must not recompile the fitting rounds or uncertainties.
        captured = (settings if name == 'grid' else
                    {key: value for key, value in settings.items() if key != 'n_z'})
        entries = cache.setdefault(name, [])
        for previous, previous_model, existing in entries:
            if captured_model is previous_model and _same_settings(captured, previous):
                compiled[name] = existing
                break
        else:
            evaluate = (partial(function, captured) if captured_model is None else
                        _with_captured_model(function, captured, captured_model))
            compiled[name] = jax.jit(evaluate, static_argnames=static_names)
            entries.append((captured, captured_model, compiled[name]))
    return compiled
