"""The light-curve fit: the model flux and chi2 each event is fitted by, the
rounds, ADJUST step and photo-z coarse grid that minimise it, and
`LightCurveFitter`, which compiles these and runs the fit. Its methods are
the Python interface (`fit`, `predict`), the fit lifecycle the snlc_fit
command shares with it (`fit_prepared`) and the SIMFIT_IDEAL refit
(`ideal_fit`); `BatchFitResult` holds what a fit returns. The module also
builds the fitted model (`fit_model`) and sizes the shape a fitter is
compiled at (`shape_for`).

The compiled functions capture a read-only snapshot of the fitter's
settings, made when it is prepared, including the arrays the model is
evaluated against (`fit_static_data`). A fitter changed later never alters a
function already compiled, and fitters whose snapshots match share their
compiled functions (`compiled_functions`). A data version's calibrated
filters (`calibration.fit_photometry`) are an argument instead, so a
calibration systematic runs on the same compiled code. The model is an
argument too: registered model pytrees supply arrays at runtime, while
ordinary model classes remain compilation constants.

The fit runs SNANA's rounds, each a Newton-LM minimisation
(`jax_snlc_core.optimiser`) for every event. Round 1 recomputes the
model error at every step; later rounds fix the weights at each round's
start, in Woodbury form when the fit carries the full epoch covariance, or
recompute the diagonal error under USE_DIAG_ONLY. The redshift priors and
the photo-z band set are built in `photoz`, and their chi2 terms evaluated
here; the Hessian and profile errors are in `uncertainties`.
"""

import copy
import dataclasses
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass
from functools import partial, wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from astropy.table import MaskedColumn, Table
from jax import vmap

from jax_snlc_core import logging as log
from jax_snlc_core.filters import (KcorReader, band_weights_at_z, band_window_width,
                                   filter_values, mw_attenuation_rest)
from jax_snlc_core.interpolation import (compute_catmullrom_slopes,
                                         hermite_interp_1d_batch)
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.models import FittedModel, build_model, infer_model_type_from_path
from jax_snlc_core.models.roles import flux_error_from_fracerr
from jax_snlc_core.models.sed_model import SNModel
from jax_snlc_core.optimiser import newton_lm
from jax_snlc_core.utils import float_dtype, require_precision

from . import parameters, photoz, uncertainties
from .batch import (ERRFLAG_FIT_CUTS, ERRFLAG_FITPREP_NOEPOCHS, ERRFLAG_MNFIT_NAN,
                    FIT_REASON_DESCRIPTIONS, LightCurveBatch, choose_epoch_lengths, pad_epochs,
                    padded_batches, plan_fit_batches)
from .calibration import (attach_mw_attenuation, calibration_bands, fit_photometry,
                          flux_factor, mw_attenuation, normalise_flux, read_kcor, resolve_filtlist,
                          resolved_filters, zero_point_shift_scales)
from .config import FitConfig, Selection
from .config_parser import build_config, fitted_model_type, supplied_configuration
from .cuts import PHOTOZ_TREST_CUTS, cut_windows, post_fit_cuts
from .loader import (batch_from_tables, input_tables, prepare_batch, read_shared_files,
                     validate_tables)
from .parameters import (PHOTOZ_INISTP_DEFAULT, adaptive_param_scales,
                         covariance_output_params, fit_parameters, ndof_n_params,
                         resolve_fitwin_windows, resolve_parameter_key, resolve_parameter_keys,
                         resolve_usesim_columns, snana_parameter_aliases,
                         start_distance_modulus, validate_fit_parameters,
                         validate_inival_bounds)
from .photoz import photoz_priors, scan_size
from .uncertainties import errmax_bad, errors_from_chi2, profile_errors, reported_errors


logger = logging.getLogger(__name__)

# PKMJD's column: FittedModel declares it first in every fit vector
T0 = 0


def _static_data(settings, photometry, model=None):
    """What a model evaluates against: this fitter's static data and a
    data version's ``photometry``."""
    data = {**settings['static_data'], **photometry}
    if model is not None:
        data.update(host_rv=model.host_rv,
                    fluxcal_scale_mag_offset=model.fluxcal_scale_mag_offset)
    return data


def fit_flux(flux, zp_fluxcal):
    """Keep positive model flux up to SNANA's bright limit (magnitude 5).

    Unlike SNANA's magnitude-40 cutoff, arbitrarily faint positive flux is
    retained to avoid a discontinuity in the fitting objective. Nonpositive
    phase extrapolations still contribute zero flux and zero model error.
    """
    hi = 10.0 ** (-0.4 * (5.0 - zp_fluxcal))
    return jnp.where((flux > 0.0) & (flux < hi), flux, 0.0)


def _rest_band_weights(static_data, z, mw_att):
    """One set of :func:`band_weights`, [n_filters, n_wave]: with the Milky
    Way attenuation ``mw_att`` folded in, or without it when that is None."""
    return band_weights_at_z(
        z, static_data['common_filter_wave_grid'],
        static_data['common_filter_trans_grid'],
        static_data['wavelength_grid'], mw_att)


def band_weights(settings, photometry, z, mw_att):
    """(flux weights, error weights), each [n_filters, n_wave], of one event
    at redshift ``z``: with its observer-frame Milky Way attenuation
    ``mw_att`` [n_common] folded in, and without it for the error
    denominators (Finteg_filter vs Finteg_forErr, genmag_SALT2.c:2867-2868).
    """
    static_data = _static_data(settings, photometry)
    return (_rest_band_weights(static_data, z, mw_att),
            _rest_band_weights(static_data, z, None))


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
    flat = vmap(lambda z: _rest_band_weights(static_data, z, None))(
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
    (:func:`fit_flux`). Under photo-z the
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


def _fit_redshifts(settings, params, scales, z_data):
    """The redshift a fit evaluates at: the fitted zPHOT under photo-z,
    else the data's."""
    if settings['config'].photoz.dofit_photoz:
        return params[:, -1] * scales[:, -1]
    return z_data


def event_bands(settings, photometry, z_weights, z_data, bw, mw_attenuation):
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


def sn_chi2(settings, model, strategy, photometry, p_scaled, start, scales, mjd, filter_indices,
             mask, fluxcals, fluxcalerrs, bands, prior_data, weights, z_prior):
    """One event's chi2 at ``p_scaled``, the parameters' priors included,
    and the redshift's under photo-z; ``bands`` are its band weights
    (:func:`event_bands`).

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
        flux_err = model_error(error_overrides=parameters.error_surface_overrides(
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
                error_overrides=(parameters.error_surface_overrides(
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

    chi2 = chi2 + parameters.prior_chi2(model_params, settings['priors'], prior_data)
    if photo:
        chi2 = photoz_priors(settings, chi2, z, z_prior,
                             lambda: model.distance_of(p, settings['param_names']))
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
        bands = event_bands(settings, photometry, z_weights, sn_z, sn_bw,
                            sn_mw_attenuation)

        def chi2(p_active):
            return sn_chi2(
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
    bands = event_bands(settings, photometry, z_weights, data['z'], data['bw'],
                        data['mw_attenuation'])
    if fixed_z is not None:
        bw, bw_err = interpolated_band_weights(
            settings, photometry, z_weights, fixed_z, data['mw_attenuation'])
        bands = lambda _: (fixed_z, bw, bw_err, (None, None, None))
    return lambda p: sn_chi2(
        settings, model, strategy, photometry, p, data['start'], data['scales'], data['mjd'],
        data['filter_indices'], data['mask'], data['fluxcals'],
        data['fluxcalerrs'], bands, data['prior_data'], data['weights'],
        data['z_priors'])


def _errors(settings, model, *, strategy, photometry, z_weights, **inputs):
    """Hessian errors on each event's unchanged final-round objective.

    Held parameters have zero covariance and no contribution to the EDM.
    Array shapes follow :func:`_round`.
    """
    held = settings['round1_held_params'] if strategy == 'live' else settings['held_params']
    free = jnp.asarray([i for i, n in enumerate(settings['param_names'])
                        if n not in held], dtype=int)

    def one(data):
        loss = _error_loss(settings, model, strategy, photometry, z_weights, data)
        return errors_from_chi2(loss, data['params'], data['scales'], free)
    return vmap(one)(inputs)


def _minos_errors(settings, model, *, strategy, photometry, z_weights, **inputs):
    """Shared bounded continuation for redshift or all free parameters."""
    return profile_errors(
        settings, strategy,
        partial(_error_loss, settings, model, strategy, photometry, z_weights), **inputs)


def colour_fraction(settings, model, photometry, filter_indices, z, mask):
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
    z = _fit_redshifts(settings, params, scales, z_data)
    if photo:
        kcor_frac = colour_fraction(settings, model, photometry, filter_indices, z, mask)
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
            err_flux = flux_error_from_fracerr(
                cfg.model.fudge_modelerr_scale * jnp.maximum(jnp.abs(flux), 1e-30), fracerr)
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
    overrides = parameters.error_surface_overrides(settings['fit_parameters'], first_round=True)
    flux = vmap(partial(model_flux, settings, model, photometry))(
        physical, mjd, filter_indices, mask, bw, z)
    fracerr = vmap(partial(model_fracerr, settings, model, photometry,
                           error_overrides=overrides))(
        physical, mjd, filter_indices, mask, bw, bw_err, z)
    model_err = jnp.where(mask, flux_error_from_fracerr(jnp.abs(flux), fracerr), 0.0)

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


def ratio_to_dmu(ratio):
    """A flux ratio as the magnitude shift SNModel.shift_amplitude wants."""
    return -2.5 * jnp.log10(ratio)


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

    z = _fit_redshifts(settings, params, scales, z_data)
    # the amplitude ratio's, at the start's redshift
    bw_start = (events_band_weights(settings, photometry, z, mw_attenuation)
                if cfg.photoz.dofit_photoz else bw)

    # FCNSNLC at a starting point: model errors and priors included,
    # amplitude untouched (snlc_fit.F90:9617)
    def chi2_at(p):
        prior_data = parameters.per_sn(p[:, T0] * scales[:, T0],
                                   cfg.fit.distance_modulus(z), reference_mjds)
        return vmap(lambda sn_p, sc, sn_mjd, sn_filter_indices, sn_mask,
                    sn_fluxcals, sn_fluxcalerrs, sn_z, sn_bw, sn_prior,
                    sn_z_prior, sn_mw_attenuation: sn_chi2(
                        settings, model, 'live', photometry, sn_p, sn_p, sc, sn_mjd, sn_filter_indices,
                        sn_mask, sn_fluxcals, sn_fluxcalerrs,
                        event_bands(settings, photometry, z_weights, sn_z, sn_bw,
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


def _grid_amplitude_ratio(signal, fluxcals, flux, variance):
    """The factor the coarse grid rescales a point's amplitude by: the
    variance-weighted ratio of data to model over the ``signal`` epochs
    (FFSUM_CROSS / FFSUM_MODEL over S/N > 3), or 1, leaving the amplitude
    as it stands, unless both sums are positive (snlc_fit.F90:8933)."""
    cross = jnp.sum(jnp.where(signal, fluxcals * flux / variance, 0.0), axis=-1)
    norm = jnp.sum(jnp.where(signal, flux ** 2 / variance, 0.0), axis=-1)
    rescale = (cross > 0.0) & (norm > 0.0)
    return jnp.where(rescale, cross / jnp.where(rescale, norm, 1.0), 1.0)


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
    error_overrides = parameters.error_surface_overrides(settings['fit_parameters'], first_round=True)

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
                    factor = _grid_amplitude_ratio(
                        keep & (sn_fluxcals > 3. * sn_fluxcalerrs), sn_fluxcals, flux,
                        sn_fluxcalerrs**2 + flux_error_from_fracerr(flux, fraction)**2)
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

            ratio = _grid_amplitude_ratio(keep & (sn_fluxcals > 3.0 * sn_fluxcalerrs),
                                          sn_fluxcals, flux, data_sqerr + model_err ** 2)
        sigma2 = data_sqerr + (model_err * ratio[..., None]) ** 2
        chi2 = jnp.sum(jnp.where(
            keep, (sn_fluxcals - flux * ratio[..., None]) ** 2 / sigma2, 0.0),
            axis=-1)
        # each point's distance after the amplitude rescaling
        chi2 = photoz_priors(
            settings, chi2, sn_z_grid[:, None], sn_z_prior,
            lambda: (sn_mu[:, None] + ratio_to_dmu(ratio) if candidates is None
                     else model.distance_of(candidates, settings['param_names'])))
        chi2 = jnp.where(sn_valid_z[:, None], chi2, jnp.inf)

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
    """``function`` with ``settings`` and an ordinary ``model`` captured, so the
    model stays outside JAX and the model argument passed in is ignored."""
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


def _prob_ndof(ndof, prior_chi2):
    """The NDOF a fit probability is taken at: the fit's ``ndof`` plus one
    where the prior's chi2 exceeds 0.01, as the chi2 it is taken of includes
    the prior (NDOF_STORE(1), snana.F90:28092-28108)."""
    return ndof + (prior_chi2 > 0.01)


@dataclass
class BatchFitResult:
    """Fit results for a batch of supernovae; every array is indexed by
    supernova (first dimension n_sne)."""
    snids: np.ndarray

    params: Dict[str, jnp.ndarray]           # each [n_sne]
    param_errors: Dict[str, jnp.ndarray]     # each [n_sne]
    param_covariances: jnp.ndarray           # [n_sne, n_params, n_params]
    cov_output_params: List[str]             # names covered by param_covariances

    chi_squared: jnp.ndarray  # the minimised chi2, prior included (SNANA's FITCHI2)
    # epochs fitted, less the parameters the data constrain (not the held
    # ones, nor those a prior constrains; zPHOT counts): SNANA's NDOF
    ndof: jnp.ndarray
    prior_chi2: jnp.ndarray   # [n_sne]; included in chi_squared

    observation_mask: jnp.ndarray  # epochs the fit used [n_sne, max_obs]
    hessian_cond: jnp.ndarray      # [n_sne]
    hessian_positive: jnp.ndarray  # [n_sne], before covariance repair
    edm: jnp.ndarray               # estimated distance to minimum [n_sne]

    # zPRIOR and zPRIORERR (snlc_fit.F90:8561); None for spec-z
    z_prior: Optional[Tuple[jnp.ndarray, jnp.ndarray]]

    # {FITWIN_ suffix: [n_sne] ok mask}
    iter1_verdicts: Dict[str, jnp.ndarray]
    outcomes: Optional[pd.DataFrame] = None
    context: Optional[list] = None
    model_columns: Optional[Dict[str, np.ndarray]] = None

    # Positive lower/upper profile distances and per-side status, [n_sne, 2].
    minos_errors: Optional[Dict[str, jnp.ndarray]] = None
    minos_status: Optional[Dict[str, jnp.ndarray]] = None

    # SNANA error sources, including step-size replacements (code 6).
    error_flags: Optional[Dict[str, jnp.ndarray]] = None

    @property
    def n_sne(self):
        return len(self.snids)

    @property
    def fitprob_ndof(self):
        """The NDOF FITPROB is taken at (:func:`_prob_ndof`)."""
        return _prob_ndof(self.ndof, self.prior_chi2)

    def fitprob(self, model):
        """Each fit's FITPROB under ``model``, on :attr:`fitprob_ndof`."""
        return np.asarray(model.compute_fitprob(
            np.asarray(self.chi_squared), np.asarray(self.fitprob_ndof)))

    def reference_mu(self, config, batch):
        """The reference distance modulus the fitted parameters are read
        against: at the fitted redshift under photo-z, as the fit's own
        distance prior is, and at ``batch``'s otherwise."""
        return config.fit.distance_modulus(
            photoz.fitted_z(config, self.params, batch.redshifts))

    @property
    def param_names(self):
        """List of fitted parameter names."""
        return list(self.params.keys())

    def take(self, indices) -> 'BatchFitResult':
        """Return the objects selected by ``indices``, in that order."""
        indices = np.asarray(indices)
        if indices.ndim != 1 or (indices.size and not np.issubdtype(indices.dtype, np.integer)):
            raise SNANAError('indices must be a one-dimensional sequence of integers')
        indices = np.arange(self.n_sne)[indices.astype(np.intp)]
        outcomes = None
        if self.outcomes is not None:
            rows = self.outcomes.set_index('fit_index')
            outcomes = rows.loc[indices].reset_index()
            outcomes['fit_index'] = np.arange(len(indices))
        return BatchFitResult(
            snids=self.snids[indices],
            params={k: v[indices] for k, v in self.params.items()},
            param_errors={k: v[indices] for k, v in self.param_errors.items()},
            minos_errors=(None if self.minos_errors is None else
                          {k: v[indices] for k, v in self.minos_errors.items()}),
            minos_status=(None if self.minos_status is None else
                          {k: v[indices] for k, v in self.minos_status.items()}),
            error_flags=(None if self.error_flags is None else
                         {k: v[indices] for k, v in self.error_flags.items()}),
            param_covariances=self.param_covariances[indices],
            cov_output_params=self.cov_output_params,
            chi_squared=self.chi_squared[indices],
            ndof=self.ndof[indices],
            prior_chi2=self.prior_chi2[indices],
            observation_mask=self.observation_mask[indices],
            hessian_cond=self.hessian_cond[indices],
            edm=self.edm[indices],
            hessian_positive=self.hessian_positive[indices],
            z_prior=(None if self.z_prior is None
                     else tuple(a[indices] for a in self.z_prior)),
            iter1_verdicts={k: v[indices]
                            for k, v in self.iter1_verdicts.items()},
            outcomes=outcomes,
            context=None if self.context is None else [self.context[i] for i in indices],
            model_columns=(None if self.model_columns is None else
                           {name: values[indices] for name, values in self.model_columns.items()}),
        )

    def filter_by_mask(self, mask) -> 'BatchFitResult':
        """Keep only the SNe where ``mask`` is True."""
        mask = np.asarray(mask)
        if mask.shape != (self.n_sne,):
            raise SNANAError(
                f"mask of {mask.shape} does not match n_sne={self.n_sne}; "
                f"a shorter mask would silently drop the tail of the sample",
                function_name="filter_by_mask")
        return self.take(np.flatnonzero(mask))

    @classmethod
    def concatenate(cls, results: list) -> 'BatchFitResult':
        """One result holding every SN in ``results``, in list order."""
        def cat(name):
            return jnp.concatenate([getattr(r, name) for r in results])

        if not results:
            raise SNANAError('Cannot concatenate an empty result list')
        if any(r.param_names != results[0].param_names for r in results):
            raise SNANAError('Cannot concatenate results with different parameter vectors')
        for field in ('outcomes', 'model_columns', 'minos_errors', 'minos_status',
                      'error_flags'):
            present = [getattr(result, field) is not None for result in results]
            if any(present) and not all(present):
                raise SNANAError(f'Cannot mix results with and without {field}')
        fitted = [r for r in results if r.n_sne]
        verdict_names = set(fitted[0].iter1_verdicts) if fitted else set()
        if any(set(r.iter1_verdicts) != verdict_names for r in fitted):
            raise SNANAError('Cannot concatenate fits with different first-round verdicts')
        contexts = [entry for r in results for entry in (r.context or [])]
        if contexts:
            first = contexts[0][0]
            if any(c['calibration'] != first['calibration'] or c['config'].model != first['config'].model
                   for c, _ in contexts):
                raise SNANAError('Cannot concatenate results with different models or calibration')
            if any(r.context is None and r.n_sne for r in results):
                raise SNANAError('Cannot mix results with and without prediction context')
        outcomes, offset, input_offset = [], 0, 0
        for r in results:
            if r.outcomes is not None:
                frame = r.outcomes.copy()
                frame.loc[frame.fit_index >= 0, 'fit_index'] += offset
                frame['input_index'] += input_offset
                outcomes.append(frame)
                input_offset += len(frame)
            offset += r.n_sne
        width = max(r.observation_mask.shape[1] for r in results)
        masks = [jnp.pad(r.observation_mask, ((0, 0), (0, width-r.observation_mask.shape[1]))) for r in results]
        z_prior = results[0].z_prior
        return cls(
            snids=np.concatenate([r.snids for r in results]),
            params={name: jnp.concatenate([r.params[name] for r in results])
                    for name in results[0].param_names},
            param_errors={
                name: jnp.concatenate([r.param_errors[name] for r in results])
                for name in results[0].param_names},
            minos_errors=(None if results[0].minos_errors is None else
                {name: jnp.concatenate([r.minos_errors[name] for r in results])
                 for name in results[0].param_names}),
            minos_status=(None if results[0].minos_status is None else
                {name: jnp.concatenate([r.minos_status[name] for r in results])
                 for name in results[0].param_names}),
            error_flags=(None if results[0].error_flags is None else
                {name: jnp.concatenate([r.error_flags[name] for r in results])
                 for name in results[0].param_names}),
            param_covariances=cat('param_covariances'),
            cov_output_params=results[0].cov_output_params,
            chi_squared=cat('chi_squared'),
            ndof=cat('ndof'),
            prior_chi2=cat('prior_chi2'),
            observation_mask=jnp.concatenate(masks),
            hessian_cond=cat('hessian_cond'),
            edm=cat('edm'),
            hessian_positive=cat('hessian_positive'),
            z_prior=(None if z_prior is None else tuple(
                map(jnp.concatenate, zip(*(r.z_prior for r in results))))),
            iter1_verdicts={
                name: jnp.concatenate([r.iter1_verdicts[name]
                                       for r in fitted])
                for name in sorted(verdict_names)},
            outcomes=pd.concat(outcomes, ignore_index=True) if outcomes else None,
            context=contexts or None,
            model_columns=(None if results[0].model_columns is None else {
                name: np.concatenate([result.model_columns[name] for result in results])
                for name in results[0].model_columns}),
        )

    def to_table(self):
        """An Astropy table in input order, including objects without a fit."""
        outcomes = self.outcomes
        if outcomes is None:
            outcomes = pd.DataFrame({'snid': self.snids, 'fit_index': np.arange(self.n_sne)})
        table = Table.from_pandas(outcomes)
        if 'reason' in table.colnames:
            table['reason_description'] = [FIT_REASON_DESCRIPTIONS[code]
                                           for code in table['reason']]
        rows = np.asarray(outcomes.fit_index)
        valid = rows >= 0
        parameters = (self.model_columns if self.model_columns is not None else
                      {**self.params, **{name+'_err': value for name, value in self.param_errors.items()}})
        columns = {**parameters,
                   'chi_squared': self.chi_squared, 'prior_chi2': self.prior_chi2,
                   'ndof': self.ndof, 'edm': self.edm}
        for name, values in columns.items():
            data = np.zeros(len(rows))
            data[valid] = np.asarray(values)[rows[valid]]
            table[name] = MaskedColumn(data, mask=~valid)
        return table

    def write_table(self, path, **kwargs):
        """Explicitly export a result table using Astropy's supported formats."""
        self.to_table().write(path, **kwargs)

# SIMFIT_IDEAL_PRESCALE debug value (snlc_fit.F90:915)
SIMFIT_IDEAL_DEBUG = 777


def fiterr_windowed(config, passband_mask, filter_mean_wave, t0_phys,
                    redshifts, mjd_offsets, filter_indices, iteration):
    """SNANA GET_FUDGE_FITERR_MAXFRAC (snlc_fit.F90:11188): per-epoch
    FUDGE_FITERR_MAXFRAC where TREST, rest-lambda, passband and iteration all
    fall inside their windows; 0.0 (scalar) when the fudge is off.
    ``passband_mask`` is the fitter's per-filter FUDGE_FITERR_PASSBANDS
    gate."""
    fit = config.fit
    if fit.fudge_fiterr_maxfrac <= 0.0 or iteration > fit.fudge_fiterr_maxiter:
        return 0.0
    trest = (mjd_offsets - t0_phys[:, None]) / (1.0 + redshifts[:, None])
    t = fit.fudge_fiterr_trest
    lt = (((trest >= t[0]) & (trest <= t[1]))
          | ((trest >= t[2]) & (trest <= t[3])))
    lamrest = filter_mean_wave[filter_indices] / (1.0 + redshifts[:, None])
    lam_w = fit.fudge_fiterr_restlam
    lam = (((lamrest >= lam_w[0]) & (lamrest <= lam_w[1]))
           | ((lamrest >= lam_w[2]) & (lamrest <= lam_w[3])))
    passband = jnp.asarray(passband_mask)[filter_indices]
    return jnp.where(lt & lam & passband, fit.fudge_fiterr_maxfrac, 0.0)


def maxfrac_floor_errs(maxfrac, fluxcals, fluxcalerrs, masks):
    """SNANA MAXFRAC error floor (snlc_fit.F90:10988): add (maxfrac * the SN's
    peak flux over every band)^2 to the data variance. maxfrac is a scalar
    (all epochs) or a per-epoch array (windowed FUDGE_FITERR)."""
    if jnp.ndim(maxfrac) == 0 and maxfrac <= 0.0:
        return fluxcalerrs
    peak = jnp.max(jnp.where(masks, fluxcals, -jnp.inf), axis=1, keepdims=True)
    peak = jnp.where(jnp.isfinite(peak), peak, 0.0)
    return jnp.sqrt(fluxcalerrs**2 + jnp.where(masks, maxfrac * peak, 0.0)**2)


def fitwin_trest_bounds(config, filters):
    """Per-filter (lo, hi) for the round-1 phase cut, in fit-band order.

    SNANA fills every filter's window with the global FITWIN_TREST and then
    overwrites the bands FITWIN_TREST_FILTER names, testing only the
    per-filter array (SETCUTS_FITWIN_TREST, snlc_fit.F90:2774-2812).
    Returns None when no band is overridden, so the scalar window stands.
    """
    if not config.fit.fitwin_trest_filter:
        return None
    lo = np.full(len(filters), config.fit.fitwin_trest[0], dtype=np.float64)
    hi = np.full(len(filters), config.fit.fitwin_trest[1], dtype=np.float64)
    for i, window in enumerate(filter_values(config.fit.fitwin_trest_filter, filters, None)):
        if window is not None:
            lo[i], hi[i] = window
    return lo, hi


def fitwin_trest_edges(config, filter_indices, filters):
    """The (lo, hi) FITWIN_TREST edges for every epoch, per its band.

    Scalars unless FITWIN_TREST_FILTER named a band, matching SNANA, which
    tests only the per-filter array (snlc_fit.F90:4486).
    """
    bounds = fitwin_trest_bounds(config, filters)
    if bounds is None:
        return config.fit.fitwin_trest
    lo, hi = bounds
    return jnp.asarray(lo)[filter_indices], jnp.asarray(hi)[filter_indices]


def check_photoz_trest_cuts(config):
    """INISTP_PHOTOZ=1 leaves the rest frame undetermined (snlc_fit.F90:3289),
    so a rest-frame cut is refused unless ABORT_ON_TRESTCUT is off. ``config``
    is resolved against the model (:func:`resolved_config`), and is checked
    before any data are read."""
    if not (config.photoz.dofit_photoz
            and config.model.inistp.get('zPHOT') == 1.0):
        return
    named = [n for n in PHOTOZ_TREST_CUTS if n in config.selection.cutwin
             or n in config.selection.epoch_cutwin]
    if named and config.fit.abort_on_trestcut:
        raise SNANAError(
            f"CUTWIN_{', CUTWIN_'.join(n.upper() for n in named)} cut "
            f"on the rest frame, which INISTP_PHOTOZ=1 leaves "
            f"undetermined until the fit. Override with "
            f"ABORT_ON_TRESTCUT = F.",
            function_name="ABORT_ON_CUTWIN_TREST")


def resolved_config(config, model):
    """``config`` with its ``INIVAL_``, ``INISTP_`` and ``PRIOR_`` suffixes
    resolved against the fit parameters, which only ``model`` can identify,
    so the spellings the input used, such as SHAPE or PHOTOZ, go."""
    if model.distance_param is None and (
            config.photoz.prior_muerrscale is not None
            or config.photoz.prior_dmu_range is not None):
        raise SNANAError(
            "PRIOR_MUERRSCALE and PRIOR_DMU_RANGE constrain a fitted distance, "
            f"which {type(model).__name__} does not have.",
            function_name="resolved_config")
    result = dataclasses.replace(config, model=dataclasses.replace(
        config.model, **resolve_parameter_keys(
            config.model, list(fit_parameters(config, model)),
            snana_parameter_aliases(model))))
    check_photoz_trest_cuts(result)
    return result


def ideal_config(config):
    """The config of the SIMFIT_IDEAL refit: at most two iterations, and
    PKMJD pinned at its truth except under the debug value, which refits the
    measured flux (SIMFIT_IDEAL_PREP, snlc_fit.F90:15808, :15843)."""
    debug = config.fit.simfit_ideal_prescale == SIMFIT_IDEAL_DEBUG
    return dataclasses.replace(
        config,
        fit=dataclasses.replace(
            config.fit, _dofit_ideal=True, simfit_ideal_prescale=0,
            n_fit_iterations=min(config.fit.n_fit_iterations, 2)),
        **({} if debug else dict(model=dataclasses.replace(
            config.model, inistp={**config.model.inistp, 'PKMJD': 0.0}))))


def rounds2_strategy(config, full_covariance):
    """The weights rounds 2+ fit against. USE_FITCOV (snlc_fit.F90:10566)
    freezes them at each round's start, in Woodbury form when the fit carries
    the full epoch covariance: its colour dispersion, FUDGE_MAG_COVERR or the
    Milky Way extinction error (:10569). Without it, USE_DIAG_ONLY keeps
    sigma live (:10572)."""
    if full_covariance:
        return 'frozen_woodbury'
    return ('frozen_diag' if config.fit.use_offdiag_covariance
            else 'live_reweight')


def fit_band_window(model, photometry):
    """How many points of ``model``'s wavelength grid the band sum spans for
    ``photometry``, or None for the whole grid (see SNModel.evaluate)."""
    return band_window_width(model.wavelength_grid,
                             photometry["common_filter_wave_grid"],
                             photometry["common_filter_trans_grid"])


def fit_static_data(model, band_window, compiled_cache=None):
    """What a fitter evaluates ``model`` against beyond the data's
    photometry: its wavelength grid and range, the host dust law the
    simulator also builds, and the band sum's ``band_window``."""
    # The default preparation depends only on the law and wavelength grid.
    # Keep custom preparation independent: it can depend on additional state.
    if (compiled_cache is not None
            and type(model).build_host_dust_law is SNModel.build_host_dust_law):
        dust = compiled_cache.setdefault('host_dust', {})
        key = (model.host_colorlaw, tuple(np.asarray(model.wavelength_grid)),
               jax.config.jax_enable_x64)
        if key not in dust:
            dust[key] = model.build_host_dust_law()
        host_dust_law = dust[key]
    else:
        host_dust_law = model.build_host_dust_law()
    return {
        "wavelength_grid": model.wavelength_grid,
        "host_dust_law": host_dust_law,
        # the fitter intersects it with RESTLAMBDA_FITRANGE (snlc_fit.F90:1421)
        "model_wave_range": tuple(float(v)
                                  for v in (model.filter_center_range
                                            or model.get_wavelength_range())),
        "host_rv": model.host_rv,
        "fluxcal_scale_mag_offset": model.fluxcal_scale_mag_offset,
        # the fitter's SED calls exclude the sim-only late-time extrapolation
        "latetime_extrap": False,
        "band_window": band_window,
    }


def empty_results(fit_parameters, max_obs, minos=None):
    """Zero-SN result, so cuts rejecting everything still write a FITRES header."""
    names = list(fit_parameters)
    zero = jnp.zeros(0)
    return BatchFitResult(
        snids=np.zeros(0, dtype=int), params={n: zero for n in names},
        param_errors={n: zero for n in names},
        error_flags={n: jnp.zeros(0, dtype=jnp.int32) for n in names},
        minos_errors=({n: jnp.zeros((0, 2)) for n in names} if minos is not None else None),
        minos_status=({n: jnp.zeros((0, 2), dtype=int) for n in names}
                      if minos is not None else None),
        param_covariances=jnp.zeros((0, len(names), len(names))),
        cov_output_params=covariance_output_params(fit_parameters),
        chi_squared=zero, ndof=zero, prior_chi2=zero,
        observation_mask=jnp.zeros((0, max_obs), dtype=bool),
        hessian_cond=zero, edm=zero,
        hessian_positive=jnp.zeros(0, dtype=bool),
        z_prior=(zero, zero) if 'zPHOT' in fit_parameters else None,
        iter1_verdicts={},
    )


def _fit_sizes(n_obs, config):
    """``(epoch_lengths, batch_size)`` for a fitter fitting samples whose
    epoch counts are ``n_obs``, one array per sample: epoch bins chosen over
    all of them together, and BATCH_SIZE events per padded batch."""
    return (choose_epoch_lengths(np.concatenate([np.asarray(n) for n in n_obs]),
                                 config.fit.num_nobs_bins,
                                 config.fit.min_bin_gain),
            config.fit.batch_size)


def batch_needs(config, batch):
    """What fitting ``batch`` under ``config`` asks of a fitter's shape: its
    light-curve lengths and, under photo-z, the coarse grid its redshift
    priors span (:func:`~jax_snlc_fit.photoz.scan_size`)."""
    return {'n_obs': np.asarray(batch.n_obs_per_sn),
            'n_z': (scan_size(config, batch.z_priors)
                    if config.photoz.dofit_photoz else None)}


def shape_for(config, needs, windows):
    """The shape a :class:`LightCurveFitter` is built at to fit every batch
    whose :func:`batch_needs` are in ``needs``, with the band ``windows`` of
    their photometry, at one set of compiled shapes: epoch lengths chosen
    over all their light curves and a batch size, by ``config``'s
    BATCH_SIZE, NUM_NOBS_BINS and MIN_BIN_GAIN, and the widest band window
    and coarse grid. A plain dict, which a pipeline records."""
    epoch_lengths, batch_size = _fit_sizes([need['n_obs'] for need in needs],
                                           config)
    n_z = [need['n_z'] for need in needs if need['n_z'] is not None]
    return {'epoch_lengths': [int(n) for n in epoch_lengths],
            'batch_size': int(batch_size), 'band_window': _widest_window(windows),
            'n_z': max(n_z) if n_z else None}


def _widest_window(windows):
    """The band window covering every one of ``windows``: None, the whole
    grid, where any is."""
    return None if None in windows else max(windows)


def substitute_true_flux(batch, zp_shift, key):
    """Replace measured fluxes with the simulated truth.

    SNANA carries TRUEFLUX per epoch; jax's PHOT files carry SIM_MAGOBS, so the
    flux is recovered as 10^(-0.4*(SIM_MAGOBS - ZP_FLUXCAL)). Undefined epochs
    (SIM_MAGOBS >= 99, SNANA's no-flux sentinel) become zero flux. The truth is
    on the unshifted scale, so MAGOBS_SHIFT_ZP is re-applied to it as it was to
    the measured flux (FLUXSCALE_ZP_FILT, snlc_fit.F90:15915).
    """
    trueflux = batch.true_fluxcals()
    if trueflux is None:
        raise SNANAError(
            f"{key} needs the SIM_MAGOBS column, which this data version does "
            f"not carry; it is written only for simulations.",
            function_name=key)
    if zp_shift:
        scale = np.ones(max(batch.filter_names) + 1)
        scale[list(batch.filter_names)] = zero_point_shift_scales(
            zp_shift, list(batch.filter_names.values()))
        trueflux = trueflux * scale[np.asarray(batch.filter_indices)]
    return dataclasses.replace(batch, fluxcals=trueflux)


def _ideal_batch(config, batch):
    """The noise-free batch the SIMFIT_IDEAL fit runs on.

    SET_IDEAL_FLUXCAL (snlc_fit.F90:15880) swaps in the true flux and moves
    MWEBV to its simulated value with zero error; PKMJD is pinned to truth by
    the caller. Errors are deliberately left as measured, so chi2 keeps its
    scale.
    """
    batch = substitute_true_flux(batch, config.data.magobs_shift_zp,
                                  'SIMFIT_IDEAL_PRESCALE')
    return dataclasses.replace(
        batch, mwebvs=np.asarray(batch.sim_truth['SIM_MWEBV'], dtype=float),
        mwebv_errs=np.zeros(batch.n_sne))


def fit_settings(config):
    """What a fitter is built from: configs equal in it share a fitter."""
    return config.model, config.fit, config.photoz


class LightCurveFitter:
    """Fit light curves using a shared model, calibration and fit configuration."""

    def __init__(self, model, *, calibration=None, config=None, **settings):
        """Load a named model and calibration for repeated light-curve fits.

        ``model`` is a bundled model name, a model directory or a Redback
        model name. Omit ``calibration`` to use the bundled filters, or supply
        a calibration file. Fit settings can be supplied as keyword arguments
        or through a :class:`~jax_snlc_fit.config.FitConfig` supplied as
        ``config``; keywords override the configuration. See the fitting
        settings page for accepted keyword names.
        """
        assets = Path(__file__).resolve().parents[1] / 'jax_snlc_core' / 'data'

        def model_path(value):
            supplied = Path(os.path.expandvars(str(value))).expanduser()
            bundled = assets / 'models' / str(value)
            if not supplied.is_dir() and bundled.is_dir() and supplied.name == str(value):
                return bundled.resolve()
            return supplied.resolve() if supplied.exists() else supplied

        name = str(model_path(model))
        configured = config.model.fitmodel_name if config is not None else None
        for value in (configured, settings.get('fitmodel_name')):
            if value is not None and model_path(value).resolve() != Path(name).resolve():
                raise SNANAError('model and fitmodel_name specify different models')
        cfg = build_config({**settings, 'fitmodel_name': name}, config=config)
        cfg = build_config({'fitmodel_name': name,
                            'model_type': cfg.model.model_type or infer_model_type_from_path(name)}, config=cfg)
        model_instance = fit_model(cfg)
        if calibration is not None and cfg.data.kcor_file is not None:
            if not isinstance(calibration, (str, Path)) or str(calibration) != cfg.data.kcor_file:
                raise SNANAError('calibration and kcor_file specify different sources')
        source = calibration if calibration is not None else cfg.data.kcor_file
        if source is None or source == 'builtin':
            source = assets / 'calibration' / 'filters.yaml'
        if isinstance(source, KcorReader):
            kcor = source
        else:
            cfg.data.kcor_file = str(source)
            kcor = read_kcor(cfg)
        cfg.data.kcor_file = str(kcor.kcor_file_path)
        requested = copy.deepcopy(cfg)
        cfg.fit.filtlist_fit = (resolve_filtlist(cfg.fit.filtlist_fit, kcor)
                                if cfg.fit.filtlist_fit is not None else list(kcor.filter_names))
        self._initialize(cfg, model_instance, kcor, dict(epoch_lengths=[],
                     batch_size=cfg.fit.batch_size, band_window=None, n_z=None))
        self._input_config = requested

    @classmethod
    def _from_prepared(cls, config, model, kcor, shape, *, compiled_cache=None):
        """Use resources and batch dimensions already prepared by the pipeline."""
        fitter = cls.__new__(cls)
        fitter._initialize(config, model, kcor, shape, compiled_cache=compiled_cache)
        return fitter

    def _initialize(self, config: FitConfig, model: Any, kcor: KcorReader,
                    shape: Dict[str, Any], *, compiled_cache=None):
        self._compiled_cache = {} if compiled_cache is None else compiled_cache
        config = copy.deepcopy(config)
        self.calibration = kcor
        self._input_config = copy.deepcopy(config)
        self._warned_limits = set()
        model_parameters = model.get_fit_parameters()
        validate_fit_parameters(model, model_parameters)
        self.model = model
        self.epoch_lengths = list(shape['epoch_lengths'])
        self.batch_size = shape['batch_size']
        self.n_z = shape['n_z']
        self.static_data = fit_static_data(model, shape['band_window'], self._compiled_cache)
        # the epoch lengths a batch has been compiled at
        self._compiled_lengths = set()

        # the fit vector: the model's parameters, then zPHOT under photo-z
        self.model_param_names = list(model_parameters)
        self.n_model_params = len(self.model_param_names)
        self.fit_parameters = fit_parameters(config, model)
        self.param_names = list(self.fit_parameters)

        self.aliases = snana_parameter_aliases(model)
        config = resolved_config(config, model)
        validate_inival_bounds(config, model_parameters)
        self.config = config
        # USESIM_<p> pins, USESIM_INIVAL floats (snlc_fit.F90:8305, :8341)
        self.usesim_columns, usesim_pinned = resolve_usesim_columns(
            config, model_parameters, self.param_names, self.aliases)
        self.ideal_fitter = (
            type(self)._from_prepared(ideal_config(config), model, kcor, shape,
                                      compiled_cache=self._compiled_cache)
            if config.fit.simfit_ideal_prescale else None)

        self.priors = parameters.active(model, config, self.model_param_names)

        # INISTP = 0 pins the parameter (SNANA FLOATPAR=.FALSE.)
        inistp = {name: spec['inistp'] for name, spec in
                  self.fit_parameters.items() if 'inistp' in spec}
        inistp.update(config.model.inistp)
        for name, value in config.model.fixed.items():
            if value is None:
                inistp.pop(name, None)
            else:
                inistp[name] = 0.0
        if config.model.lfixpar_all or config.model.lfixpar_adjust:
            # zeroes INISTP for every fit parameter, zPHOT included
            # (snlc_fit.F90:1974-1978)
            inistp.update({name: 0.0 for name in self.param_names})
        self.held_params = tuple(sorted(
            {name for name, step in inistp.items() if step == 0.0}
            | set(usesim_pinned)))
        for name, value in config.model.fixed.items():
            if value is not None and name in config.model.initial and config.model.initial[name] != value:
                raise SNANAError(f"Conflicting initial and fixed values for {name}")
        # round 1 also holds what the model defers (BayeSN's epsilon)
        self.round1_held_params = tuple(sorted(set(self.held_params) | {
            n for n, spec in self.fit_parameters.items() if spec.get('deferred')}))

        # params_scaled = params_physical / param_scales; zPHOT's is per event
        self.param_scales = jnp.array(
            [spec.get('scale', 1.0) for spec in model_parameters.values()])

        # RESTLAMBDA_USEFIT (snlc_fit.F90:1421, :4195)
        model_lo, model_hi = self.static_data['model_wave_range']
        user_lo, user_hi = config.fit.restlambda_fitrange
        self.restlambda_usefit = (max(model_lo, user_lo), min(model_hi, user_hi))
        if self.restlambda_usefit[1] <= self.restlambda_usefit[0]:
            raise SNANAError(
                f"RESTLAMBDA_FITRANGE {config.fit.restlambda_fitrange} and the "
                f"model's range {self.static_data['model_wave_range']} do not "
                f"overlap, so no epoch can be fitted.",
                function_name="LightCurveFitter.__init__")

        self._set_filters(resolved_filters(config, kcor))

        self._prepare_compiled_functions()

    def _prepare_compiled_functions(self):
        """Snapshot numerical settings and select compatible compiled functions.

        The fitter owns the editable settings. Compiled functions retain the
        snapshot made here, so later fitter changes cannot alter a shared
        function, including when JAX traces it for a new input shape.
        Replace the snapshot on preparation; never update it in place.
        """
        self._compiled_lengths.clear()
        config = self.config
        # Copy mutable configuration records into the compilation snapshot.
        # Omit resource paths and model-loading inputs: they are already
        # resolved into model arguments and must not prevent compilation reuse.
        compiled_config = FitConfig(
            model=dataclasses.replace(copy.deepcopy(config.model),
                                      fitmodel_name=None, model_type=None,
                                      _model_key_values={}),
            fit=copy.deepcopy(config.fit),
            photoz=copy.deepcopy(config.photoz),
        )
        self._compiled_settings = {
            'config': compiled_config,
            'static_data': {key: value for key, value in self.static_data.items()
                            if key not in ('host_rv', 'fluxcal_scale_mag_offset')},
            'static_model': jax.tree_util.all_leaves([self.model]),
            'fit_parameters': copy.deepcopy(self.fit_parameters),
            'priors': copy.deepcopy(self.priors),
            'param_names': tuple(self.param_names),
            'held_params': self.held_params,
            'round1_held_params': self.round1_held_params,
            'n_model_params': self.n_model_params,
            'n_filters': self.n_filters,
            'n_z': self.n_z,
            'restlambda_usefit': self.restlambda_usefit,
            'distance_prior': photoz.distance_prior_tables(config),
            'colour_dispersion': self.colour_dispersion,
            'full_covariance': self.full_covariance,
            'fudge_mag_coverr_r': self.fudge_mag_coverr_r,
        }
        # Unregistered ordinary models remain captured, without imposing JAX
        # pytree or hashing requirements on existing model authors.
        self._model_input = (None if self._compiled_settings['static_model'] else self.model)
        compiled = compiled_functions(
            self._compiled_settings, self._compiled_cache, self.model)
        self._run_round = compiled['round']
        self._run_weights = compiled['weights']
        self._run_errors = compiled['errors']
        self._run_adjust = compiled['adjust']
        self._run_grid = compiled['grid']
        self._run_flux = compiled['flux']
        self.reject_epochs = compiled['reject_epochs']

    def _set_filters(self, filters):
        """Prepare the per-band settings captured by the compiled fit functions."""
        config = self.config
        model = self.model
        self.filters = list(filters)
        self.config.fit.filtlist_fit = list(filters)
        pb = config.fit.fudge_fiterr_passbands
        selected = ([] if isinstance(pb, str) and pb.strip().upper() == "NULL"
                    else resolve_filtlist(pb, self.calibration))
        self.fudge_fiterr_passband_mask = tuple(f in selected for f in self.filters)
        cov = config.fit.fudge_mag_coverr
        self.fudge_mag_coverr_r = tuple(
            1.0 - 10.0 ** (-0.4 * value)
            for value in filter_values(cov, self.filters, 0.0)) if cov else ()

        self.n_filters = len(self.filters)

        self.colour_dispersion = bool(config.fit.use_offdiag_covariance
                                      and model.has_offdiag_covariance)
        self.full_covariance = bool(self.colour_dispersion
                                    or any(self.fudge_mag_coverr_r)
                                    or config.fit.opt_covar_mwxterr > 0)
        self.rounds2_strategy = rounds2_strategy(config, self.full_covariance)

    def _prepare_shape(self, epoch_lengths, n_z, filters, band_window):
        """Update reusable batch dimensions and the current fit bands."""
        self.epoch_lengths = list(epoch_lengths)
        bands_changed = list(filters) != self.filters
        window_changed = band_window != self.static_data['band_window']
        self.static_data['band_window'] = band_window
        grid_changed = n_z != self.n_z
        self.n_z = n_z
        if bands_changed:
            self._set_filters(filters)
        if bands_changed or window_changed or grid_changed:
            self._prepare_compiled_functions()
        if self.ideal_fitter is not None:
            self.ideal_fitter._prepare_shape(epoch_lengths, n_z, filters, band_window)

    def fit(self, photometry=None, *, mjd=None, flux=None, flux_err=None, band=None,
            redshift=None, mwebv=0., snid='0', flux_unit='fluxcal', zp=None,
            redshift_cmb=None, metadata=None, objects=None,
            column_map=None, band_map=None, progress=True, apply_selection=False):
        """Fit single-object arrays or photometry/object tables in memory.

        Supply aligned ``mjd``, ``flux`` and ``flux_err`` arrays; ``band``
        may be one calibration filter name or an aligned array. ``band_map``
        maps observation labels to full calibration names. ``redshift`` is
        required unless fitting photometric redshift. ``mwebv`` defaults to
        zero; ``metadata`` carries additional object information such as
        ``peakmjd`` or host photometric-redshift measurements.

        Alternatively, pass a ``photometry`` table with one row per
        observation and a ``snid`` identifying each object. Object properties
        such as ``redshift`` and ``mwebv`` may be repeated in this table; each
        must agree across all observations of the same object. Results follow
        the order of first appearance of each ``snid``.

        You can also supply separate ``photometry`` and ``objects`` tables joined
        by ``snid``. The photometry columns are ``snid``, ``mjd``, ``flux``,
        ``flux_err`` and ``band``. Object columns include ``snid`` and
        ``redshift``. ``column_map`` maps these names to your column names.
        Object columns for ``mwebv``, ``flux_unit`` and ``zp`` take precedence
        over the corresponding keyword defaults.

        ``flux_unit`` is ``'fluxcal'`` (default), ``'Jy'`` or ``'uJy'``.
        ``zp`` applies to fluxcal only and defaults to 31.4. ``redshift_cmb``
        is retained as metadata; the fit uses ``redshift``.

        Sample selection is disabled by default. Set ``apply_selection=True``
        to apply configured event and epoch cuts and the CLI's post-fit cuts.
        Supplying selection settings without enabling selection raises an error.

        ``progress=False`` silences the notice shown before preparing new
        fit functions. Repeated fits with reusable functions show no notice.

        Returns a :class:`BatchFitResult` with fitted parameters and an
        outcome for every input object. No output files are written.
        """
        obs, obj = input_tables(
            mjd=mjd, flux=flux, flux_err=flux_err, band=band, redshift=redshift,
            mwebv=mwebv, snid=snid, flux_unit=flux_unit, zp=zp,
            redshift_cmb=redshift_cmb, metadata=metadata, photometry=photometry,
            objects=objects, column_map=column_map)
        cfg = resolved_config(build_config(config=self.configuration()), self.model)
        if not apply_selection and cfg.selection != Selection():
            raise SNANAError('Selection settings require fit(..., apply_selection=True)')
        validate_tables(obs, obj, cfg)
        obs['band'] = calibration_bands(obs.band, self.calibration, band_map or {})
        units, zp_fluxcal = normalise_flux(obs, obj, self.calibration,
                                           self.model.wavelength_grid)
        active = (resolve_filtlist(cfg.fit.filtlist_fit, self.calibration)
                  if cfg.fit.filtlist_fit is not None else list(dict.fromkeys(obs.band)))
        if not active or not len(obj):
            result = empty_results(self.fit_parameters, 0, self.config.fit.minos)
            result.model_columns = _model_columns(self, result, np.empty(0))
            result.outcomes = pd.DataFrame(dict(snid=obj.snid, input_index=np.arange(len(obj)),
                status='rejected', reason=ERRFLAG_FITPREP_NOEPOCHS, fit_index=-1))
            return result
        cfg.fit.filtlist_fit = active
        _, epoch_cuts, _ = cut_windows(cfg)
        shared = read_shared_files(cfg)
        batch = batch_from_tables(obs, obj, self, cfg, epoch_cuts, shared, zp_fluxcal)
        batch = dataclasses.replace(batch, photometry={
            **fit_photometry(cfg, self.model, self.calibration),
            'zp_fluxcal': batch.zp_fluxcal})
        selected, _, marks = prepare_batch(cfg, self.param_names, batch,
                                        apply_selection=apply_selection, ladder=[])
        input_batch = selected
        selected, result, keep, codes, _, _ = self.fit_prepared(cfg, selected, apply_selection=apply_selection, progress=progress)
        result.outcomes = _outcomes(obj, result, marks, keep, codes)
        result.model_columns = _model_columns(
            self, result, photoz.fitted_z(cfg, result.params, selected.redshifts))
        context = dict(redshifts=input_batch.redshifts,
                       reference_mjds=input_batch.reference_mjds,
                       mwebvs=input_batch.mwebvs, mwebv_errs=input_batch.mwebv_errs,
                       sim_mw_colour_law=input_batch.sim_mw_colour_law,
                       zp_fluxcal=input_batch.zp_fluxcal,
                       units=units, config=copy.deepcopy(self.config),
                       calibration=self.calibration.kcor_file_path)
        result.context = [(context, i) for i in range(result.n_sne)]
        return result

    def configuration(self):
        """Return a detached copy of the supplied configuration."""
        return supplied_configuration(self._input_config)

    def reset_shape(self):
        """Recompute observation bins and photo-z search size on the next fit."""
        self.epoch_lengths = []
        self.n_z = None
        self._prepare_compiled_functions()
        if self.ideal_fitter is not None:
            self.ideal_fitter.reset_shape()

    def predict(self, result, mjd, band, *, snid=None, flux_unit=None, zp=None):
        """Evaluate a stored fit at the requested MJDs and bands.

        Returns a dictionary with ``mjd``, ``band``, ``flux`` and
        ``model_flux_err`` arrays. ``model_flux_err`` is the model's
        intrinsic photometric uncertainty, not uncertainty propagated from
        the fitted parameters.

        For a result containing several objects, ``snid`` selects the one
        to evaluate. Output uses that object's fitted flux units and zero
        point unless ``flux_unit`` or ``zp`` specifies another calibration.
        """
        if result.context is None:
            raise SNANAError('This result has no in-memory prediction context')
        if snid is None:
            if result.n_sne != 1:
                raise SNANAError('Specify snid when predicting a result with several objects')
            snid = str(result.snids[0])
        found = np.flatnonzero(np.asarray(result.snids).astype(str) == str(snid))
        if len(found) != 1:
            raise SNANAError(f'No unique fitted object {snid!r}')
        i = int(found[0])
        context, batch_row = result.context[i]
        if context['calibration'] != self.calibration.kcor_file_path or context['config'].model != self.config.model:
            raise SNANAError('Result and self use different model or calibration settings')
        if not all(np.isfinite(np.asarray(value)[i]) for value in result.params.values()):
            raise SNANAError(f'Object {snid} has no finite fitted parameters')
        times = np.asarray(mjd, dtype=float)
        if times.ndim != 1 or not np.isfinite(times).all():
            raise SNANAError('Prediction MJD must be a finite one-dimensional array')
        bands = calibration_bands(np.broadcast_to(band, times.shape), self.calibration, {})
        if not len(times):
            return dict(mjd=times, band=bands, flux=np.empty(0), model_flux_err=np.empty(0))
        names = list(dict.fromkeys(bands))
        cfg = copy.deepcopy(context['config'])
        cfg.fit.filtlist_fit = names
        zp_fluxcal = context['zp_fluxcal']
        phot = {**fit_photometry(cfg, self.model, self.calibration), 'zp_fluxcal': zp_fluxcal}
        attenuation, _ = mw_attenuation(
            cfg, phot, np.asarray(context['mwebvs'])[[batch_row]],
            np.asarray(context['mwebv_errs'])[[batch_row]], context['sim_mw_colour_law'])
        params = {name: np.asarray(values)[[i]] for name, values in result.params.items()}
        redshift = params.get('zPHOT', np.asarray(context['redshifts'])[[batch_row]])
        flux, error = self.evaluate_flux(params, times[None], redshift,
            np.array([[names.index(b) for b in bands]]), np.ones((1, len(times)), dtype=bool),
            reference_mjds=np.asarray(context['reference_mjds'])[[batch_row]],
            mw_att_obs=attenuation, photometry=phot)
        original_unit, original_zp = context['units'][str(snid)]
        unit = original_unit if flux_unit is None else flux_unit
        zero = original_zp if flux_unit is None and zp is None else zp
        factor = flux_factor(unit, zero, bands, self.calibration,
                              self.model.wavelength_grid, zp_fluxcal=zp_fluxcal)
        return dict(mjd=times, band=bands, flux=np.asarray(flux)[0]/factor,
                    model_flux_err=np.asarray(error)[0]/factor)

    def fudged_fluxcalerrs(self, filter_mean_wave, maxfrac, t0_offsets,
                           redshifts, mjd_offsets, filter_indices, masks,
                           fluxcals, fluxcalerrs, iteration):
        """The error the fit divides by: the measured one with the MAXFRAC
        fudge in quadrature (JEP_FUDGEFLUX_ERR, snlc_fit.F90:4208-4215), for
        the fit's rounds and for the cut that recounts SNR afterwards.
        """
        window = fiterr_windowed(self.config, self.fudge_fiterr_passband_mask,
                                 filter_mean_wave, t0_offsets, redshifts,
                                 mjd_offsets, filter_indices, iteration)
        return maxfrac_floor_errs(jnp.maximum(maxfrac, window), fluxcals,
                                  fluxcalerrs, masks)

    def fitwin_trest_ok(self, filter_indices, mjd_offsets, t0_offsets,
                        redshifts):
        """Epochs inside FITWIN_TREST at the given peak time and redshift
        (REJECT_TREST, snlc_fit.F90:4486)."""
        trest_lo, trest_hi = fitwin_trest_edges(self.config, filter_indices,
                                                self.filters)
        trest = ((mjd_offsets - t0_offsets[:, None])
                 / (1.0 + redshifts[:, None]))
        return (trest >= trest_lo) & (trest <= trest_hi)

    def _iter1_param_index(self, key):
        """Column a round-1 FITWIN key names, in this model's fit vector."""
        return self.param_names.index(resolve_parameter_key(
            key, self.param_names, self.aliases))

    def fit_sample(self, sample: LightCurveBatch,
                   start: Dict[str, jnp.ndarray], *, progress=False):
        """Fit ``sample`` from ``start`` (one [n_sne] array per parameter).
        The sample carries its photometry and, under photo-z, its redshift
        priors, as :func:`~jax_snlc_fit.loader.load` gives them, and its
        Milky Way attenuation, as :meth:`as_fitted` adds it.

        Returns the results in the sample's own order and one record per
        fitted batch: its events, seconds, epoch length, and whether it was
        the first at its length, which compiles the fit. Every batch is
        ``batch_size`` events at one of ``epoch_lengths``, a short one padded
        by repeating its own events; the padding is dropped from the results,
        whose epoch mask is at the sample's length.
        """
        if not sample.n_sne:
            return empty_results(self.fit_parameters, sample.max_obs,
                                 self.config.fit.minos), []
        items = plan_fit_batches(sample.n_obs_per_sn, self.epoch_lengths,
                                 self.batch_size)
        if not items:
            return empty_results(self.fit_parameters, sample.max_obs,
                                 self.config.fit.minos), []
        order = np.concatenate([rows for _, rows, _ in items])
        input_order = np.empty(sample.n_sne, dtype=int)
        input_order[order] = np.arange(sample.n_sne)
        log.line(logger, f"Fitting {sample.n_sne} SNe in {len(items)} "
                 f"batch(s) of {self.batch_size}")
        lengths = sorted({length for length, _, _ in items})
        if len(lengths) > 1:
            log.line(logger, f"Epoch bins padded to {lengths} "
                     f"rather than all to {sample.max_obs}")

        if any(length not in self._compiled_lengths for length in lengths):
            message = "Preparing fit functions (JAX compilation may take a moment)..."
            if progress:
                print(message, file=sys.stderr, flush=True)
            else:
                log.line(logger, message)

        z_weights = None
        if self.config.photoz.dofit_photoz:
            n_nohost = int(np.sum(~np.asarray(sample.z_priors.host_ok)))
            if n_nohost:
                log.warn(logger, "PHOTOZ",
                         f"{n_nohost} of {sample.n_sne} events have no host "
                         f"photo-z: fitted with a flat redshift prior "
                         f"(width {photoz.PHOTOZ_STD_NOHOST:g}) over "
                         f"PHOTOZ_BOUND")
            z_weights = z_grid_band_weights(self._compiled_settings, sample.photometry)

        results, records = [], []
        for ic, (epoch_length, rows, idx) in enumerate(items):
            carries_jit = epoch_length not in self._compiled_lengths
            t_batch = time.time()
            batch_results = self._fit_batch(
                sample.take(idx, epoch_length),
                {k: v[idx] for k, v in start.items()}, z_weights)
            dt_batch = time.time() - t_batch
            self._compiled_lengths.add(epoch_length)
            batch_results = batch_results.take(np.arange(len(rows)))
            results.append(dataclasses.replace(
                batch_results, observation_mask=pad_epochs(
                    batch_results.observation_mask[:, :sample.max_obs],
                    sample.max_obs)))

            done = (f"Batch {ic + 1}/{len(items)}: {len(rows)} SNe at "
                    f"{epoch_length} epochs, {dt_batch:.1f}s")
            log.line(logger, f"{done} (may include JIT)" if carries_jit
                     else f"{done} ({len(rows) / dt_batch:.0f} SNe/sec)")
            records.append({'n_sne': len(rows), 'seconds': dt_batch,
                            'epoch_length': epoch_length,
                            'carries_jit': carries_jit})
        return BatchFitResult.concatenate(results).take(input_order), records

    def stack(self, params, reference_mjds):
        """``params`` (one [n_sne] array per fitted name) as the
        [n_sne, n_params] physical fit vector, PKMJD as its offset from
        ``reference_mjds``."""
        return jnp.stack([params[n] - reference_mjds if n == 'PKMJD'
                          else params[n] for n in self.param_names], axis=-1)

    def _unstack(self, stacked, reference_mjds):
        """The inverse of :meth:`stack`: one [n_sne] array per fitted name,
        PKMJD back in MJD."""
        return {n: (stacked[:, i] + reference_mjds if n == 'PKMJD'
                    else stacked[:, i])
                for i, n in enumerate(self.param_names)}

    def _bounds(self, physical, peak_estimate, reference_mjds=0.):
        """Per-SN bounds [n_sne, n_params] from each parameter's declaration.

        A parameter declares its bounds under exactly one key: ``bounds`` for an
        absolute ``(lo, hi)``, ``bounds_relative_to_init`` for ``(lo_factor,
        hi_factor)`` applied to its starting value: SALT's x0, which spans
        orders of magnitude with redshift, or ``bounds_about_peak_estimate``
        for ``(lo, hi)`` about ``peak_estimate``.
        """
        lows, highs = [], []
        for i, spec in enumerate(self.fit_parameters.values()):
            if 'bounds_relative_to_init' in spec:
                lo_factor, hi_factor = spec['bounds_relative_to_init']
                low, high = physical[:, i] * lo_factor, physical[:, i] * hi_factor
            elif 'bounds_about_peak_estimate' in spec:
                lo, hi = spec['bounds_about_peak_estimate']
                low, high = peak_estimate + lo, peak_estimate + hi
            else:
                lo, hi = spec['bounds']
                low = jnp.full(physical.shape[0], lo)
                high = jnp.full(physical.shape[0], hi)
            lows.append(low)
            highs.append(high)
        low, high = jnp.stack(lows, axis=1), jnp.stack(highs, axis=1)
        for name, requested in self.config.model.bounds.items():
            index = self.param_names.index(name)
            lo, hi = requested
            if not np.isfinite(requested).all() or lo >= hi:
                raise SNANAError(f'Invalid bounds for {name}: {requested}')
            offset = reference_mjds if index == T0 else 0.
            lo, hi = lo-offset, hi-offset
            accepted = (lo >= low[:, index]) & (hi <= high[:, index])
            if not bool(jnp.all(accepted)) and name not in self._warned_limits:
                warnings.warn(f"Rejected bounds {requested} for {name} outside model limits; "
                              "retaining the model limits for those objects", stacklevel=2)
                self._warned_limits.add(name)
            low = low.at[:, index].set(jnp.where(accepted, lo, low[:, index]))
            high = high.at[:, index].set(jnp.where(accepted, hi, high[:, index]))
        for form, index, coefficients in self.priors:
            support = parameters.support(form, coefficients)
            if support is None:
                continue
            offset = reference_mjds if coefficients.get('public_time') else 0.
            lo, hi = support[0] - offset, support[1] - offset
            low = low.at[:, index].set(jnp.maximum(low[:, index], lo))
            high = high.at[:, index].set(jnp.minimum(high[:, index], hi))
        if bool(jnp.any(low >= high)):
            raise SNANAError('Parameter bounds and prior support do not overlap')
        return low, high

    def _scales(self, batch, z_priors):
        """Per-SN parameter scales [n_sne, n_params]: each parameter's own,
        the amplitude's from the starting distance, and zPHOT's the prior's
        width, or its default step when that width is zero."""
        scales = jnp.tile(self.param_scales[None, :], (batch.n_sne, 1))
        start_mu = start_distance_modulus(self.config, batch.redshifts)
        for name, at_distance in adaptive_param_scales(self.model, start_mu).items():
            scales = scales.at[:, self.param_names.index(name)].set(at_distance)
        if self.config.photoz.dofit_photoz:
            z_scale = jnp.where(z_priors.z_prior_err > 0.0,
                                z_priors.z_prior_err, PHOTOZ_INISTP_DEFAULT)
            scales = jnp.column_stack([scales, z_scale])
        return scales

    def _fit_batch(self, batch, start, z_weights) -> BatchFitResult:
        """Fit one batch with vmap over Newton-LM, each SN an independent
        optimisation in scaled space, each step clipped to the SN's bounds.

        ``start`` holds the starting values and ``z_weights`` the sample's
        :func:`z_grid_band_weights`, None for a spec-z fit, as the batch's
        redshift priors are.
        """
        cfg = self.config
        photo = cfg.photoz.dofit_photoz
        ref, photometry = batch.reference_mjds, batch.photometry
        z_priors = batch.z_priors
        if photo and 'zPHOT' not in start:
            start = {**start, 'zPHOT': z_priors.z_init}
        scales = self._scales(batch, z_priors)
        physical = self.stack(start, ref)
        explicit = set(cfg.model.initial) | {
            n for n, v in cfg.model.fixed.items() if v is not None}
        if photo:
            z_lower, z_upper = photoz.redshift_bounds(cfg, z_priors)
            if 'zPHOT' not in explicit:
                physical = physical.at[:, -1].set(jnp.clip(
                    physical[:, -1], z_lower, z_upper))
        params = physical / scales
        # PKMJD's bound centre; without an estimate the start is the reference
        peak_estimate = (jnp.asarray(batch.peakmjd_offsets)
                         if batch.peakmjd_offsets is not None
                         else jnp.zeros(batch.n_sne))

        constrained = bool(cfg.model.bounds or any(
            value is not None and 'distribution' in value for value in cfg.model.priors.values()))
        input_limits = self._bounds(physical, peak_estimate, ref) if constrained or explicit else None
        if photo and input_limits is not None:
            lower, upper = input_limits
            input_limits = (lower.at[:, -1].set(jnp.maximum(lower[:, -1], z_lower)),
                            upper.at[:, -1].set(jnp.minimum(upper[:, -1], z_upper)))
            if bool(jnp.any(input_limits[0] >= input_limits[1])):
                raise SNANAError('Parameter bounds and redshift PDF support do not overlap')
        for name in explicit:
            index = self.param_names.index(name)
            values = physical[:, index]
            invalid = (~jnp.isfinite(values) | (values < input_limits[0][:, index])
                       | (values > input_limits[1][:, index]))
            if bool(jnp.any(invalid)):
                raise SNANAError(f'Explicit value for {name} is outside accepted bounds/prior support')
        limits = input_limits if constrained else None
        if limits is not None:
            physical = jnp.clip(physical, *limits)
            params = physical / scales

        if (photo and not cfg.fit.skip_grid_search
                and batch.sncid_inival is None):
            params = self._grid_start(batch, params, scales, z_priors, limits)
            # bounds about the grid's start (snlc_fit.F90:8631-8635)
            physical = params * scales
        lo, hi = limits if limits is not None else self._bounds(physical, peak_estimate, ref)
        if photo:
            lo = lo.at[:, -1].set(jnp.maximum(lo[:, -1], z_lower))
            hi = hi.at[:, -1].set(jnp.minimum(hi[:, -1], z_upper))
        lo, hi = lo / scales, hi / scales
        if photo:
            params = params.at[:, -1].set(jnp.clip(
                params[:, -1], lo[:, -1], hi[:, -1]))

        # bands in range at the start z (REDSHIFT_FIT, snlc_fit.F90:8679);
        # photo-z tests RESTLAMBDA at its trial z (Z8TEST, :1776)
        mask = batch.observation_mask & restlambda_mask(self._compiled_settings, photometry, batch.filter_indices,
            _fit_redshifts(self._compiled_settings, params, scales, batch.redshifts))

        # a spec-z fit's band weights, once for the batch; a photo-z fit's
        # follow its fitted redshift
        bw = (None if photo else events_band_weights(self._compiled_settings, photometry, batch.redshifts, batch.mw_att_obs))

        # FITINI_ADJUST (snlc_fit.F90:9594, :9801), after FITINI_PHOTOZ
        # under photo-z (:1962-1966)
        params, chi2red_ok, chi2red_ok2 = self._run_adjust(
            self._model_input, photometry, z_weights, params, scales, batch.mjd_offsets,
            batch.filter_indices, mask, batch.fluxcals, batch.fluxcalerrs,
            batch.redshifts, bw, batch.n_obs_per_sn, z_priors,
            batch.mw_att_obs, reference_mjds=batch.reference_mjds,
            limits=limits)
        if photo:
            params = params.at[:, -1].set(jnp.clip(
                params[:, -1], lo[:, -1], hi[:, -1]))

        (params, chi2, prior_chi2, mask, verdicts, uncertainty) = self._fit_rounds(
            batch, params, scales, lo, hi, mask, z_priors, z_weights, bw)
        errors, covariances, hessian_cond, edm, hessian_positive = uncertainty[:5]
        profiles = uncertainty[5] if cfg.fit.minos is not None else None
        if profiles is not None:
            errors = reported_errors(errors, *profiles)
        for name, verdict in (('CHI2RED_INI', chi2red_ok),
                              ('CHI2RED_INI2', chi2red_ok2)):
            if verdict is not None:
                verdicts[name] = verdict

        physical = params * scales
        reported_z_prior = None
        if photo:
            # what zPRIOR/zPRIORERR report (snlc_fit.F90:8561)
            reported_z_prior = (z_priors.z_init, jnp.maximum(
                0.001, z_priors.z_prior_err * cfg.photoz.prior_zerrscale))

        return BatchFitResult(
            snids=batch.snids,
            params=self._unstack(physical, ref),
            param_errors={n: errors[:, i]
                          for i, n in enumerate(self.param_names)},
            minos_errors=(None if profiles is None else
                {n: profiles[0][:, i] for i, n in enumerate(self.param_names)}),
            minos_status=(None if profiles is None else
                {n: profiles[1][:, i] for i, n in enumerate(self.param_names)}),
            param_covariances=covariances,
            # FITCHI2 includes the prior (snana.F90:27794)
            chi_squared=chi2,
            ndof=self._ndof(mask),
            hessian_cond=hessian_cond,
            edm=edm,
            hessian_positive=hessian_positive,
            observation_mask=mask,
            prior_chi2=prior_chi2,
            z_prior=reported_z_prior,
            # zPHOT last (TABLE_VARLIST_COV, snlc_fit.F90:19399)
            cov_output_params=covariance_output_params(self.fit_parameters),
            iter1_verdicts=verdicts,
        )

    def _grid_start(self, batch, params, scales, z_priors, limits=None):
        """The photo-z coarse grid's start (INIPAR_PHOTOZ_COURSEGRID),
        PHOTOZ_GRID_BATCH events at a time, a short last slice padded by
        repeating its own events. SKIP_PHOTOZ_INIT leaves an event without
        a host at its midpoint (snlc_fit.F90:8523)."""
        found = []
        for rows, s in padded_batches(
                np.arange(batch.n_sne),
                min(self.config.photoz.photoz_grid_batch, batch.n_sne)):
            found.append(self._run_grid(
                self._model_input, batch.photometry, params[s], scales[s],
                batch.mjd_offsets[s], batch.filter_indices[s],
                batch.observation_mask[s], batch.fluxcals[s],
                batch.fluxcalerrs[s], jax.tree.map(lambda a: a[s], z_priors),
                batch.mw_att_obs[s], None if limits is None else tuple(a[s] for a in limits))[:len(rows)])
        grid = jnp.concatenate(found)
        if self.config.photoz.skip_photoz_init:
            grid = jnp.where(z_priors.host_ok[:, None], grid, params)
        return grid

    def _fit_rounds(self, batch, params, scales, lo, hi, mask, z_priors,
                    z_weights, bw):
        """SNANA's fit iterations over one batch: round 1 on live weights,
        then rounds 2..NFIT_ITERATION on weights frozen at each round's start.

        Between round 1 and round 2 the epochs FITWIN_TREST leaves out at the
        fitted (or data) redshift and peak go, and so do the epochs the
        rejection tests drop. Under photo-z, round 1 drops the bands outside
        PHOTOZ_ITER1_LAMRANGE, is repeated on the band set the fitted redshift
        supports (PHOTOZ_ANA, snlc_fit.F90:3350-3418), and every round but
        the last drops a band the fitted redshift no longer supports.

        Returns the parameters, their chi2 and its prior part at the fitted
        redshift, the epochs the fit used, the round-1 window verdicts, and
        the errors, covariances, Hessian condition numbers and EDM on what
        the last round minimised, as MINUIT's are on its FCN.
        """
        cfg = self.config
        photo = cfg.photoz.dofit_photoz
        photometry = batch.photometry
        lam = photometry['filter_mean_wavelengths']
        mjd, filter_indices = batch.mjd_offsets, batch.filter_indices
        fluxcals = batch.fluxcals
        z_data, mw_attenuation = batch.redshifts, batch.mw_att_obs

        def fudged(maxfrac, p, m, iteration):
            return self.fudged_fluxcalerrs(
                lam, maxfrac, p[:, T0] * scales[:, T0],
                _fit_redshifts(self._compiled_settings, p, scales, z_data), mjd, filter_indices, m, fluxcals,
                batch.fluxcalerrs, iteration)

        def round_(strategy, p, m, fluxcalerrs, prior_data, weights):
            p, chi2 = self._run_round(
                self._model_input, strategy=strategy, photometry=photometry, z_weights=z_weights,
                start=p, scales=scales, lo=lo, hi=hi, mjd=mjd,
                filter_indices=filter_indices, mask=m, fluxcals=fluxcals,
                fluxcalerrs=fluxcalerrs, z=z_data, bw=bw, prior_data=prior_data,
                weights=weights, z_priors=z_priors, mw_attenuation=mw_attenuation)
            return p, chi2, dict(strategy=strategy, mask=m, fluxcalerrs=fluxcalerrs,
                                 prior_data=prior_data, weights=weights)

        # each pass's t0 prior is centred on its INIVAL(IPAR_PEAKMJD), which
        # after the first iteration is the last one's fit (snlc_fit.F90:4840,
        # :7913-7921); the distance prior is at the redshift prior's mean
        # under photo-z until a redshift is fitted
        def prior_at(p, mu):
            return parameters.per_sn(p[:, T0] * scales[:, T0], mu, batch.reference_mjds)

        mu_round1 = cfg.fit.distance_modulus(
            z_priors.z_prior if photo else z_data)
        prior_data = prior_at(params, mu_round1)
        maxfrac_r1 = (cfg.fit.fudgeall_iter1_maxfrac
                      if cfg.fit.fudgeall_iter1_maxfrac > 0
                      else cfg.fit.fudgeall_maxfrac)

        mask_r1 = ((mask & photoz.iter1_band_mask(cfg, lam, filter_indices)) if photo
                   else mask)
        round_start = params
        fluxcalerrs = fudged(maxfrac_r1, params, mask_r1, 1)
        params, chi2, last_likelihood = round_(
            'live', params, mask_r1, fluxcalerrs, prior_data, None)
        if photo:
            kept, repeat = photoz.round1_band_set(
                cfg, self.restlambda_usefit, lam, filter_indices, batch.observation_mask,
                params * scales, z_priors)
            if bool(jnp.any(repeat)):
                # round 1 again on the band set (snana.F90:9514-9516), the
                # other events keeping the errors they were fitted with
                mask_r1 = jnp.where(repeat[:, None], kept, mask_r1)
                fluxcalerrs = jnp.where(repeat[:, None],
                                fudged(maxfrac_r1, params, mask_r1, 1), fluxcalerrs)
                round_start = jnp.where(repeat[:, None], params, round_start)
                prior_data = jnp.where(repeat[:, None],
                                       prior_at(params, mu_round1), prior_data)
                again, chi2_again, last_likelihood = round_(
                    'live', round_start, mask_r1, fluxcalerrs, prior_data, None)
                params = jnp.where(repeat[:, None], again, params)
                chi2 = jnp.where(repeat, chi2_again, chi2)
            mask = kept & ~photoz.dropped_epochs(
                cfg, self.restlambda_usefit, lam, filter_indices, params * scales,
                z_priors, repeat)
        verdicts = self.iter1_verdicts(params, scales, chi2, mask_r1,
                                       prior_data, z_priors)

        # FITWIN_TREST and the epoch rejects, from round 1's last pass
        # (snlc_fit.F90:4481-4510, :10885, :10967)
        z = _fit_redshifts(self._compiled_settings, params, scales, z_data)
        mask = mask & self.fitwin_trest_ok(
            filter_indices, mjd, params[:, T0] * scales[:, T0], z)
        mask = self.reject_epochs(
            self._model_input, photometry, mask, params * scales, mjd, filter_indices, fluxcals, batch.fluxcalerrs,
            batch.reference_mjds, z,
            events_band_weights(self._compiled_settings, photometry, z, mw_attenuation) if photo else bw)

        # USE_DIAG_ONLY reads the frozen weights only for the sigma term
        frozen = (self.rounds2_strategy != 'live_reweight'
                  or bool(cfg.fit.opt_chi2_sigma))
        # a spec-z fit's colour-dispersion fraction holds over its rounds 2+;
        # a photo-z fit's follows its fitted redshift
        kcor_frac = (colour_fraction(self._compiled_settings, self.model, photometry, filter_indices, z_data, mask)
                     if frozen and not photo else None)
        for round_index in range(cfg.fit.n_fit_iterations - 1):
            prior_data = prior_at(params, cfg.fit.distance_modulus(z)
                                  if photo else mu_round1)
            # MAXFRAC at every iteration (snlc_fit.F90:4327)
            fluxcalerrs = fudged(cfg.fit.fudgeall_maxfrac, params, mask,
                         round_index + 2)
            weights = (self._run_weights(self._model_input, photometry, z_weights, params,
                                         scales, mjd, filter_indices, mask, fluxcals, fluxcalerrs,
                                         z_data, bw, kcor_frac, mw_attenuation,
                                         batch.mwxt_magerrs)
                       if frozen else None)
            round_start = params
            params, chi2, last_likelihood = round_(self.rounds2_strategy, params, mask,
                                        fluxcalerrs, prior_data, weights)
            z = _fit_redshifts(self._compiled_settings, params, scales, z_data)
            if photo and round_index < cfg.fit.n_fit_iterations - 2:
                mask = mask & ~photoz.dropped_epochs(
                    cfg, self.restlambda_usefit, lam, filter_indices, params * scales,
                    z_priors, True)

        errors = self._run_errors(
            self._model_input, **last_likelihood, photometry=photometry, z_weights=z_weights,
            lo=lo, hi=hi,
            params=params, start=round_start, scales=scales, mjd=mjd,
            filter_indices=filter_indices, fluxcals=fluxcals, z=z_data, bw=bw,
            z_priors=z_priors, mw_attenuation=mw_attenuation)
        if cfg.fit.minos is not None:
            params, chi2 = errors[-2:]
            errors = errors[:-2]
        prior_chi2 = self._prior_chi2(
            params * scales, last_likelihood['prior_data'], z_priors)
        return params, chi2, prior_chi2, mask, verdicts, errors

    def iter1_verdicts(self, params, scales, chi2, mask, prior_data, z_priors):
        """Which round-1 window accepted each SN: ``{window: ok_mask}`` over
        the windows this fit configured, empty when it configured none.

        SNANA abandons the event mid-fit (FITPAR_ANA, snlc_fit.F90:2139-2155);
        a vmapped batch records the verdict instead and the caller drops those
        events. The window is named as the input names it, so what rejected
        an event survives into the results; SNANA's own numbering for it
        belongs to the tables that report it.

        PROB is FITPROBCHI2_STORE(1): the probability of the minimised chi2,
        prior included, on NDOF + NDOF_PRIOR (:func:`_prob_ndof`,
        MNFIT_STOREPAR, snana.F90:28085-28125).
        """
        verdicts = {}
        physical = params * scales
        windows = self.config.fit
        if windows.fitwin_prob_iter1:
            lo, hi = windows.fitwin_prob_iter1
            prob = self.model.compute_fitprob(chi2, _prob_ndof(
                self._ndof(mask),
                self._prior_chi2(physical, prior_data, z_priors)))
            verdicts['PROB'] = (prob >= lo) & (prob <= hi)
        for key, (lo, hi) in windows.fitwin_iter1.items():
            val = physical[:, self._iter1_param_index(key)]
            verdicts[key] = (val >= lo) & (val <= hi)
        return verdicts

    def _prior_chi2(self, physical, prior_data, z_priors):
        """Every event's prior chi2 at the physical parameters: the model
        parameters' priors, and under photo-z the redshift's
        (CHI2PRIOR(IPAR_zPHOT), snlc_fit.F90:4903) and any distance prior's
        (CHI2PRIOR(IPAR_DLMAG), :4910)."""
        chi2 = vmap(lambda p, d: parameters.prior_chi2(p, self.priors, d))(
            physical[:, :self.n_model_params], prior_data)
        if self.config.photoz.dofit_photoz:
            chi2 = vmap(lambda c, p, z_prior: photoz_priors(
                self._compiled_settings, c, p[-1], z_prior,
                lambda: self.model.distance_of(p, self.param_names)))(chi2, physical, z_priors)
        return chi2

    def _ndof(self, mask):
        """Every event's NDOF as SNANA reports it: the fitted epochs less
        the floated parameters, zPHOT among them under photo-z
        (XNFITDOF(0), snlc_fit.F90:13132-13140)."""
        return (jnp.sum(mask.astype(jnp.int32), axis=1)
                - ndof_n_params(self.model, self.held_params)
                - (1 if self.config.photoz.dofit_photoz
                   and 'zPHOT' not in self.held_params else 0))

    def fitted_epochs(self, batch, params, z, in_bands):
        """The epochs a fit at ``params`` and redshifts ``z`` uses: those
        observed in a band of ``in_bands`` ([n_sne, n_obs]), inside
        FITWIN_TREST, and kept by the epoch rejects, which run in the fit's
        batches. The fit applies the window and the rejects at its round-1
        values; here they are at ``params``."""
        fidx = np.asarray(batch.filter_indices)
        reference_mjds = np.asarray(batch.reference_mjds)
        fit_mask = (np.asarray(batch.observation_mask) & in_bands
                    & np.asarray(self.fitwin_trest_ok(
                        fidx, np.asarray(batch.mjd_offsets),
                        params['PKMJD'] - reference_mjds, z)))
        physical = self.stack(params, reference_mjds)
        for epoch_length, rows, idx in plan_fit_batches(
                batch.n_obs_per_sn, self.epoch_lengths, self.batch_size):
            chunk = batch.take(idx, epoch_length)
            kept = self.reject_epochs(
                self._model_input, chunk.photometry, fit_mask[idx, :epoch_length],
                physical[idx], chunk.mjd_offsets, chunk.filter_indices, chunk.fluxcals,
                chunk.fluxcalerrs, chunk.reference_mjds, z[idx],
                events_band_weights(self._compiled_settings, chunk.photometry,
                                    jnp.asarray(z[idx]), chunk.mw_att_obs))
            fit_mask[np.ix_(rows, np.arange(epoch_length))] = (
                np.asarray(kept)[:len(rows)])
        return fit_mask

    def evaluate_flux(self, params, mjds, redshifts, filter_indices, masks, *,
                      reference_mjds, mw_att_obs, photometry):
        """Model flux and its error, ``(model_flux, model_flux_err)`` each
        [n_sne, max_obs], for the tables that report them.

        ``params`` holds one [n_sne] array per fitted parameter, PKMJD in
        absolute MJD; ``mjds`` are absolute MJDs, taken as offsets from
        ``reference_mjds`` here.
        """
        physical = self.stack(params, reference_mjds)[:, :self.n_model_params]
        bw, bw_err = events_band_weights(self._compiled_settings, photometry, redshifts,
                                              mw_att_obs)
        return self._run_flux(self._model_input, photometry, physical,
                              mjds - reference_mjds[:, None], filter_indices,
                              masks, bw, bw_err, redshifts,
                              band_window=fit_band_window(self.model, photometry))

    def evaluate_model(self, batch, params):
        """Model flux and its error at ``params`` (one [n_sne] array per
        parameter), [n_sne, max_obs] each, for the tables that report them.

        Batched like the fit, at the fit's shapes: evaluating a whole sample at
        once costs the memory fitting it in one go would. ``batch`` carries its
        photometry and Milky Way attenuation (:func:`attach_mw_attenuation`).
        """
        params = {k: np.asarray(v) for k, v in params.items()}
        flux = np.zeros((batch.n_sne, batch.max_obs))
        flux_err = np.zeros_like(flux)
        for epoch_length, rows, idx in plan_fit_batches(
                batch.n_obs_per_sn, self.epoch_lengths, self.batch_size):
            chunk = batch.take(idx, epoch_length)
            chunk_params = {k: v[idx] for k, v in params.items()}
            mf, mfe = self.evaluate_flux(
                chunk_params,
                chunk.reference_mjds[:, None] + chunk.mjd_offsets,
                photoz.fitted_z(self.config, chunk_params, chunk.redshifts),
                chunk.filter_indices,
                chunk.observation_mask, reference_mjds=chunk.reference_mjds,
                mw_att_obs=chunk.mw_att_obs, photometry=chunk.photometry)
            n = len(rows)
            flux[np.ix_(rows, np.arange(epoch_length))] = np.asarray(mf)[:n]
            flux_err[np.ix_(rows, np.arange(epoch_length))] = np.asarray(mfe)[:n]
        return flux, flux_err

    def start_values(self, config, batch, usesim_columns):
        """Starting values for ``batch``: the model's own guess, then
        ``INIVAL_<p>``, then the simulated truth in ``usesim_columns``, then the
        SNCID list file where OPT_SNCID_LIST bit 1 asks for it. SNANA applies
        them in that order, so truth wins over an explicit ``INIVAL_`` for a
        parameter that has both (FITINI_PARVAL, snlc_fit.F90:8290-8318).

        ``config`` is the data version's; the list file's values are the ones
        :func:`load` put on the batch. Every value is per event. A photo-z fit's
        zPHOT starts at its redshift prior before these overrides.
        """
        t0_init = jnp.array(batch.reference_mjds if batch.peakmjd_offsets is None
                            else batch.peakmjd_offsets + batch.reference_mjds)
        start = parameters.init_params(
            self.model, batch.n_sne, t0_init,
            start_distance_modulus(config, batch.redshifts))
        if config.photoz.dofit_photoz:
            start['zPHOT'] = batch.z_priors.z_init
        for name, value in config.model.inival.items():
            start[name] = jnp.full(batch.n_sne, value)
        for name, value in config.model.initial.items():
            start[name] = jnp.full(batch.n_sne, value)
        for name, column in usesim_columns.items():
            start[name] = jnp.asarray(_batch_truth(batch, column))
        if batch.sncid_inival is not None:
            for name, seed in batch.sncid_inival.items():
                start[name] = jnp.where(jnp.isnan(seed), start[name], seed)
        for name, value in config.model.fixed.items():
            if value is not None:
                start[name] = jnp.full(batch.n_sne, value)
        return start

    def _prepare_fit_shape(self, config, batch):
        """Widen this fitter's compiled shape to fit ``batch`` too, by the
        shape rules file inputs use (:func:`shape_for`)."""
        if not batch.n_sne:
            return
        shape = shape_for(config, [batch_needs(config, batch)],
                          [fit_band_window(self.model, batch.photometry)])
        if self.epoch_lengths:
            shape['epoch_lengths'] = list(self.epoch_lengths)
            shape['epoch_lengths'][-1] = max(
                shape['epoch_lengths'][-1], int(np.max(batch.n_obs_per_sn)))
            if shape['n_z'] is not None:
                shape['n_z'] = max(self.n_z or 1, shape['n_z'])
            shape['band_window'] = _widest_window(
                [self.static_data['band_window'], shape['band_window']])
        self._prepare_shape(shape['epoch_lengths'], shape['n_z'],
                             resolved_filters(config, self.calibration), shape['band_window'])

    def fitwin_windows(self, config):
        """The ``FITWIN_`` windows of ``config`` against the FITRES columns of
        this fitter's model (:func:`resolve_fitwin_windows`)."""
        return resolve_fitwin_windows(config, self.model, self.param_names,
                                      self.aliases)

    def as_fitted(self, config, batch):
        """``batch`` as the fit sees it: padded to this fitter's longest epoch
        length, which the model flux of every table is evaluated at
        (:meth:`evaluate_model`); with the simulated true flux under
        USESIM_TRUEFLUX (snana.F90:17341); and with its Milky Way attenuation."""
        self._prepare_fit_shape(config, batch)
        if not batch.n_sne:
            return batch
        batch = batch.take(np.arange(batch.n_sne), self.epoch_lengths[-1])
        if config.data.usesim_trueflux:
            batch = substitute_true_flux(batch, config.data.magobs_shift_zp,
                                          'USESIM_TRUEFLUX')
        return attach_mw_attenuation(config, batch)

    def finish_fit(self, config, batch, results, start_params, *,
                   apply_selection, fitwins=None):
        """Check fit validity, apply requested cuts and correct covariances.

        Retain every fitted row here; file output can select rows afterwards,
        while Python can report the outcome of every supplied object.
        """
        results = errmax_bad(config, self.fit_parameters, self.held_params,
                             results, start_params)
        if apply_selection and results.n_sne:
            keep, codes, rows = post_fit_cuts(
                self, config, self.fitwin_windows(config) if fitwins is None else fitwins,
                batch, results)
        else:
            keep = np.ones(results.n_sne, dtype=bool)
            codes = np.zeros(results.n_sne, dtype=int)
            rows = []
        # Model wavelength/phase support can leave a prepared object without data.
        # This is fit validity, independent of optional scientific sample cuts.
        has_observations = np.asarray(results.observation_mask).any(axis=1)
        keep &= has_observations
        codes = np.where(has_observations, codes, ERRFLAG_FITPREP_NOEPOCHS)
        finite = np.all(np.stack([np.isfinite(v) for v in results.params.values()]), axis=0)
        keep &= finite
        codes = np.where(~finite, ERRFLAG_MNFIT_NAN,
                         np.where(~keep & (codes == 0), ERRFLAG_FIT_CUTS, codes))
        if config.fit.opt_covar_lcfit and keep.any():
            indices = np.flatnonzero(keep)
            corrected = uncertainties.fix_covar_lcfit(
                results.take(indices), self.model, config.fit.opt_covar_lcfit)
            errors = {name: np.array(value) for name, value in results.param_errors.items()}
            for name, values in corrected.param_errors.items():
                errors[name][indices] = values
            covariance = np.array(results.param_covariances)
            covariance[indices] = corrected.param_covariances
            results = dataclasses.replace(results, param_errors=errors, param_covariances=covariance)
        return results, keep, codes, rows

    def fit_prepared(self, config, batch, *, apply_selection, progress=False):
        """Run the common fit lifecycle on a calibrated, fittable sample.

        Both interfaces use the same shape preparation, starting values, optimiser
        and result corrections. Sample cuts are explicit; writing files is left
        to the caller. Results retain every fitted object alongside its verdict.
        """
        timings = {}
        # a misspelt FITWIN_ suffix aborts before anything is fitted
        fitwins = self.fitwin_windows(config) if apply_selection else None
        batch = self.as_fitted(config, batch)

        log.banner(logger, "Fit lightcurves")
        start = time.perf_counter()
        start_params = self.start_values(config, batch, self.usesim_columns)
        results, timings['batches'] = self.fit_sample(batch, start_params, progress=progress)
        fit_seconds = time.perf_counter() - start
        # First-batch time is an upper bound: compatible fitters may already
        # share compiled functions, and execution is included in this estimate.
        timings['compile'] = sum(b['seconds'] for b in timings['batches']
                                 if b['carries_jit'])
        timings['fit'] = fit_seconds - timings['compile']

        start = time.perf_counter()
        results, keep, codes, cut_rows = self.finish_fit(config, batch, results, start_params,
            apply_selection=apply_selection, fitwins=fitwins)
        timings['cuts'] = time.perf_counter() - start
        return batch, results, keep, codes, cut_rows, timings

    def ideal_fit(self, config, batch):
        """SIMFIT_IDEAL_PRESCALE: refit a prescaled subset on noise-free fluxes.

        SNANA runs this ahead of the nominal fit for each selected event and then
        re-initialises (SIMFIT_IDEAL_PREP, snlc_fit.F90:15808), so the two are
        independent and this is a separate pass. Every parameter starts at its
        simulated truth and PKMJD is pinned there; at most two iterations run.
        Returns {column: [n_sne]}, 999 where an event was not refitted
        (FITPAR_IDEAL init, snlc_fit.F90:15980), or None when the key is inert.
        """
        if batch.sim_truth is None:
            # the LSIM_SNANA gate: real data silently disables the key (:15796)
            log.line(logger, "SIMFIT_IDEAL_PRESCALE ignored: not a simulation")
            return None
        ps = config.fit.simfit_ideal_prescale
        debug = ps == SIMFIT_IDEAL_DEBUG
        idx = np.asarray(batch.snana_index, dtype=float)
        mask = (np.ones(batch.n_sne, dtype=bool) if debug
                else (np.mod(idx, np.pi) / np.pi) <= (1.0 / ps))
        if not mask.any():
            log.line(logger, "SIMFIT_IDEAL_PRESCALE selected no events")
            return None

        ideal = self.ideal_fitter
        refit = ideal.with_fit_settings(config)
        # the refit starts at the truth, not from the SNCID list file
        subset = dataclasses.replace(batch.filter_by_mask(mask), sncid_inival=None)
        # debug value skips SET_IDEAL_FLUXCAL (snlc_fit.F90:15843)
        if not debug:
            subset = attach_mw_attenuation(refit, _ideal_batch(refit, subset))
        truth = {name: spec['sim_truth']
                 for name, spec in self.model.get_fit_parameters().items()
                 if 'sim_truth' in spec}
        log.line(logger, f"SIMFIT_IDEAL: fitting {subset.n_sne} of "
                 f"{batch.n_sne} SNe on {'measured' if debug else 'true'} flux")
        results, _ = ideal.fit_sample(
            subset, ideal.start_values(refit, subset, truth))

        columns = {}
        for name, vals, _ in self.model.get_fitres_param_columns(
                results.params, results.param_errors,
                np.asarray(batch.redshifts)[mask],
                results.reference_mu(refit, subset)):
            column = np.full(batch.n_sne, 999.0)
            column[mask] = np.asarray(vals)
            columns[f"{name}_IDEAL"] = column
        return columns

    def with_fit_settings(self, config):
        """``config`` reading and writing its data as it says, with this
        fitter's fit settings (:func:`fit_settings`), resolved against its
        model."""
        model, fit, photoz = fit_settings(self.config)
        return dataclasses.replace(config, model=model, fit=fit, photoz=photoz)


def prepare(config, model, kcor, batch):
    """The fitter for ``batch`` alone."""
    return LightCurveFitter._from_prepared(config, model, kcor, shape_for(
        config, [batch_needs(config, batch)],
        [fit_band_window(model, batch.photometry)]))


def _batch_truth(batch, column):
    """A simulated-truth column for one batch, for the USESIM_* keys."""
    truth = batch.sim_truth or {}
    if column not in truth:
        raise SNANAError(
            f"USESIM needs the truth column {column}, which this data version "
            f"does not carry. Available: {', '.join(sorted(truth)) or 'none'}.",
            function_name="_batch_truth")
    # FITS columns are big-endian (>f4), which JAX will not accept.
    return np.asarray(truth[column], dtype=np.float64)


def fit_model(config):
    """The model a fit of ``config`` fits, built once before the data is
    read, since the loader carries the truth columns its fit spec names.

    Shared construction path with the simulator. The grid spans the model's
    own support (post-UVLAM) at the fitter's sampling density; SNANA's
    fitter likewise initialises the model over its native range, and it
    deliberately passes no late-time extrapolation file (snlc_fit.F90
    initialises the model with a blank late-time argument).
    """
    require_precision(config.fit.double_precision)
    model_type = fitted_model_type(config)
    model_path = os.path.expandvars(os.path.expanduser(config.model.fitmodel_name))
    log.line(logger, "Building the model...")
    model_instance = build_model(model_path, config.model._model_key_values,
                                 model_type=model_type)
    if not isinstance(model_instance, FittedModel):
        raise SNANAError(
            f"Model type '{model_type}' cannot be fitted: it does not play "
            f"the FittedModel role.", function_name="FITMODEL_NAME")
    return model_instance


def _model_columns(fitter, result, redshifts):
    """Use the model's output conversion, including derived distances."""
    distance = fitter.config.fit.distance_modulus(redshifts)
    columns = {}
    for name, values, errors in fitter.model.get_fitres_param_columns(
            result.params, result.param_errors, redshifts, distance):
        columns[name] = np.asarray(values)
        columns[name + '_err'] = np.asarray(errors)
    for name, values in fitter.model.get_fitres_derived_columns(result.params):
        columns[name] = np.asarray(values)
    return columns


def _outcomes(obj, result, marks, keep, codes):
    """Report preparation and fit outcomes in the caller's object order."""
    outcome = pd.DataFrame(dict(snid=obj.snid, input_index=np.arange(len(obj)),
        status='rejected', reason=ERRFLAG_FIT_CUTS, fit_index=-1)).set_index('snid', drop=False)
    for ids, code in marks:
        outcome.loc[list(map(str, ids)), 'reason'] = code
    if result.n_sne:
        ids = np.asarray(result.snids).astype(str)
        outcome.loc[ids, 'fit_index'] = np.arange(result.n_sne)
        outcome.loc[ids, 'reason'] = codes
        outcome.loc[ids, 'status'] = np.where(keep, 'accepted', 'rejected')
    return outcome.reset_index(drop=True)
