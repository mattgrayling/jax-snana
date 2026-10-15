"""Bazin pre-fit for the peak-MJD estimate the main fit starts from.

SNANA fits every band separately and combines them (SET_PEAKMJD and
MNFIT_PKMJD, snana.F90:26576-26737):

  0. drop the weakest band, if its S/N at maximum is below 2
  1. per band, fit f(t) = A0 exp(-(t-t0)/Tfall) / (1 + exp(-(t-t0)/Trise))
     by MINUIT; here by newton_lm
  2. per band, PKMJD = t0 + Trise ln((Tfall - Trise)/Trise), or the band's own
     flux maximum where that shape is too flat to place a peak
  3. above two bands, reject the one furthest from the average of the others,
     if that distance exceeds PKMJD_OUTLIER
  4. average what is left, each band weighted by 1/err on its own PKMJD
"""

import jax
import jax.numpy as jnp
from jax import vmap

from jax_snlc_core.utils.newton_lm import newton_lm

MAXITER = 100

T0_WINDOW = 50.0        # ±50 days around MJDatFLUXMAX
TRISE_INIT = 5.0
TRISE_MIN = 1.0
TRISE_MAX = 40.0
TFALL_INIT = 10.0
TFALL_MIN = 1.0
TFALL_MAX = 100.0
A0_SCALE = 1.2
BAZIN_MODEL_MAGERR = 0.05  # MODEL_MAGERR (snana.F90:27113)

OUTLIER_THRESHOLD = 30.0     # PKMJD_OUTLIER (snana.F90:26368)
MAX_SNRMIN_REJECT = 2.0      # weakest band goes below this S/N (:26369)
BAZIN_MIN_SHAPE_RATIO = 0.1  # (Tfall-Trise)/Trise placing a peak (:27033)
PKMJD_ERR_MAX = 4.0          # days; the cap on a band's PKMJD error (:27035)


def bazin_flux(params, t):
    """Bazin function with numerical stability clamps."""
    t0, trise, tfall, a0 = params
    dt = t - t0
    exp_fall = jnp.exp(jnp.clip(-dt / tfall, -50.0, 50.0))
    exp_rise = jnp.exp(jnp.clip(-dt / trise, -50.0, 50.0))
    return a0 * exp_fall / (1.0 + exp_rise)


def bazin_loss(params, mjd_data, flux_data, err_data, mask_data):
    model = bazin_flux(params, mjd_data)
    sqerr = err_data ** 2 + (model * BAZIN_MODEL_MAGERR) ** 2
    return jnp.sum(jnp.where(mask_data,
                             (model - flux_data) ** 2 / sqerr, 0.0))


def fit_single_bazin(init_p, lo, hi, mjd_data, flux_data, err_data, mask_data):
    def loss(p):
        return bazin_loss(p, mjd_data, flux_data, err_data, mask_data)

    p_final, _, grad_norm = newton_lm(init_p, lo, hi, loss, MAXITER)
    cov = jnp.linalg.inv(0.5 * jax.hessian(loss)(p_final)
                         + 1e-12 * jnp.eye(p_final.shape[0]))
    return p_final, grad_norm, jnp.sqrt(jnp.abs(cov[0, 0]))


_fit_batch = jax.jit(vmap(fit_single_bazin))


def bazin_prefit(batch, cutwin_tobs, fallback_t0):
    """Refine t0 estimates via per-filter Bazin function fits.

    ``cutwin_tobs`` is CUTWIN_TOBS, the window the fitted epochs must fall in
    once measured from the brightest epoch of the whole light curve
    (FITWIN_MJD8, snana.F90:26609); SNANA initialises it wide open
    (:11732-11733). ``fallback_t0`` is the peak for an event no band could
    place. Returns the refined t0, the [n_sne] mask of the events no band
    reached the fit for (NPKMJD = 0), and the mask of those that took
    ``fallback_t0`` because no band's fit survived.
    """
    n_sne = batch.n_sne
    max_obs = batch.max_obs
    n_filters = len(batch.filter_names)

    mjds = batch.mjd_offsets              # [n_sne, max_obs]
    fluxcals = batch.fluxcals            # [n_sne, max_obs]
    fluxcalerrs = batch.fluxcalerrs      # [n_sne, max_obs]
    filter_indices = batch.filter_indices  # [n_sne, max_obs]
    obs_mask = batch.observation_mask      # [n_sne, max_obs]

    filter_ids = jnp.arange(n_filters)[None, :, None]  # [1, n_filters, 1]
    fi_expanded = filter_indices[:, None, :]             # [n_sne, 1, max_obs]
    mask_expanded = obs_mask[:, None, :]                 # [n_sne, 1, max_obs]
    filter_mask = (fi_expanded == filter_ids) & mask_expanded  # [n_sne, n_filters, max_obs]

    n_obs_per_filter = jnp.sum(filter_mask, axis=-1)  # [n_sne, n_filters]
    # any band with an epoch is fitted (snana.F90:26618)
    filter_has_epoch = n_obs_per_filter > 0

    flux_expanded = fluxcals[:, None, :]  # [n_sne, 1, max_obs]
    masked_flux = jnp.where(filter_mask, flux_expanded, -jnp.inf)
    argmax_idx = jnp.argmax(masked_flux, axis=-1)  # [n_sne, n_filters]

    mjd_expanded = mjds[:, None, :]  # [n_sne, 1, max_obs]
    mjd_at_fluxmax = jnp.take_along_axis(
        mjd_expanded, argmax_idx[:, :, None], axis=-1
    ).squeeze(-1)  # [n_sne, n_filters]

    flux_at_max = jnp.take_along_axis(
        flux_expanded, argmax_idx[:, :, None], axis=-1
    ).squeeze(-1)  # [n_sne, n_filters]

    # drop the weakest band below MAX_SNRMIN_REJECT (snana.F90:26584)
    err_at_max = jnp.take_along_axis(
        fluxcalerrs[:, None, :], argmax_idx[:, :, None], axis=-1).squeeze(-1)
    snr_at_max = jnp.where(err_at_max > 0, flux_at_max / err_at_max, 0.0)
    droppable = (filter_has_epoch & (snr_at_max > 0.0)
                 & (snr_at_max < MAX_SNRMIN_REJECT))
    weakest = jnp.argmin(jnp.where(droppable, snr_at_max, jnp.inf), axis=-1)
    # what NPKMJD counts: the bands that reach MNFIT_PKMJD
    filter_fitted = filter_has_epoch & ~(
        droppable.any(axis=-1)[:, None]
        & (jnp.arange(n_filters)[None, :] == weakest[:, None]))

    # [t0, trise, tfall, a0]
    safe_flux_max = jnp.maximum(jnp.abs(flux_at_max), 1.0)

    t0_init = mjd_at_fluxmax
    trise_init = jnp.full((n_sne, n_filters), TRISE_INIT)
    tfall_init = jnp.full((n_sne, n_filters), TFALL_INIT)
    a0_init = A0_SCALE * safe_flux_max

    init_params = jnp.stack([t0_init, trise_init, tfall_init, a0_init], axis=-1)

    t0_lo = mjd_at_fluxmax - T0_WINDOW
    t0_hi = mjd_at_fluxmax + T0_WINDOW
    trise_lo = jnp.full((n_sne, n_filters), TRISE_MIN)
    trise_hi = jnp.full((n_sne, n_filters), TRISE_MAX)
    tfall_lo = jnp.full((n_sne, n_filters), TFALL_MIN)
    tfall_hi = jnp.full((n_sne, n_filters), TFALL_MAX)
    a0_lo = 0.2 * safe_flux_max
    a0_hi = 5.0 * safe_flux_max

    bounds_lo = jnp.stack([t0_lo, trise_lo, tfall_lo, a0_lo], axis=-1)
    bounds_hi = jnp.stack([t0_hi, trise_hi, tfall_hi, a0_hi], axis=-1)

    N = n_sne * n_filters

    flat_mjds = jnp.broadcast_to(mjd_expanded, (n_sne, n_filters, max_obs)).reshape(N, max_obs)
    flat_flux = jnp.broadcast_to(flux_expanded, (n_sne, n_filters, max_obs)).reshape(N, max_obs)
    flat_errs_raw = jnp.broadcast_to(
        fluxcalerrs[:, None, :], (n_sne, n_filters, max_obs)
    ).reshape(N, max_obs)
    # FITWIN_MJD8 (snana.F90:26609, :27132)
    tobs_lo, tobs_hi = cutwin_tobs
    brightest = jnp.argmax(jnp.where(obs_mask & (fluxcals >= 0.0), fluxcals,
                                     -jnp.inf), axis=-1)
    mjd_brightest = jnp.take_along_axis(mjds, brightest[:, None], axis=-1)
    in_fit_window = ((mjds >= mjd_brightest + tobs_lo)
                     & (mjds <= mjd_brightest + tobs_hi))
    flat_mask = (filter_mask & in_fit_window[:, None, :]).reshape(N, max_obs)

    flat_errs = jnp.where(flat_mask, flat_errs_raw, 1.0)

    flat_init = init_params.reshape(N, 4)
    flat_lo = bounds_lo.reshape(N, 4)
    flat_hi = bounds_hi.reshape(N, 4)

    fitted_params, fit_grad_norm, fit_t0_err = _fit_batch(
        flat_init, flat_lo, flat_hi, flat_mjds, flat_flux, flat_errs, flat_mask)

    fitted_params = fitted_params.reshape(n_sne, n_filters, 4)
    fit_grad_norm = fit_grad_norm.reshape(n_sne, n_filters)
    fit_t0_err = fit_t0_err.reshape(n_sne, n_filters)

    fit_t0 = fitted_params[:, :, 0]
    fit_trise = fitted_params[:, :, 1]
    fit_tfall = fitted_params[:, :, 2]

    # flat shapes fall back to the band's flux max (snana.F90:27032-27039)
    ratio = (fit_tfall - fit_trise) / fit_trise
    peaked = ratio > BAZIN_MIN_SHAPE_RATIO
    pkmjd = jnp.where(peaked, fit_t0 + fit_trise * jnp.log(ratio),
                      mjd_at_fluxmax)
    pkmjd_err = jnp.where(peaked, jnp.minimum(fit_t0_err, PKMJD_ERR_MAX),
                          PKMJD_ERR_MAX)

    valid_filter = (filter_fitted
                    # not diverged, and NaN fails this too
                    & (fit_grad_norm < 1e10)
                    & (pkmjd_err > 0.0))

    # leave-one-out outlier rejection (snana.F90:26674-26706)
    n_valid = jnp.sum(valid_filter, axis=-1)                    # [n_sne]
    total = jnp.sum(jnp.where(valid_filter, pkmjd, 0.0), axis=-1)
    avg_without = (total[:, None] - pkmjd) / (n_valid[:, None] - 1)
    deviation = jnp.where(valid_filter, jnp.abs(avg_without - pkmjd), -1.0)
    worst = jnp.argmax(deviation, axis=-1)                      # [n_sne]
    reject = (n_valid >= 3) & (jnp.max(deviation, axis=-1) > OUTLIER_THRESHOLD)
    final_valid = valid_filter & ~(
        reject[:, None] & (jnp.arange(n_filters)[None, :] == worst[:, None]))

    # each band weighted by 1/err on its own PKMJD (snana.F90:26723)
    weight = jnp.where(final_valid, 1.0 / pkmjd_err, 0.0)
    wsum = jnp.sum(weight, axis=-1)                             # [n_sne]
    avg_pkmjd = (jnp.sum(jnp.where(final_valid, weight * pkmjd, 0.0), axis=-1)
                 / wsum)

    all_failed = wsum <= 0.0
    refined_t0 = jnp.where(all_failed, fallback_t0, avg_pkmjd)

    # unplaceable is NPKMJD == 0: no band reached the fit (snana.F90:26640)
    no_filter_fitted = ~jnp.any(filter_fitted, axis=-1)
    return refined_t0, no_filter_fitted, all_failed
