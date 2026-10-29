"""Simulated photometry: magnitudes, observed fluxes and their uncertainties.

Band integrals and noise variances shared with fitting live in core.filters
and core.noise. This module evaluates the model at each observation, applies
scatter and noise, estimates peak times, and checks the resulting fluxes.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core import logging as log
from jax_snlc_core.cosmology import cosmo_params
from jax_snlc_core.filters import (ZEROPOINT_FLUXCAL_SNANA_ORIG, band_weights_at_z,
                                   filter_band)
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.noise import (MASK_APPLY_DATA, check_redcov_entries,
                                 fluxerr_maps_serving, fluxerr_model_sqscales,
                                 interpolate_fluxerr_maps, noise_variances,
                                 parse_redcov_string, read_fluxerr_model_file,
                                 redcov_entries_serving, stacked_fluxerr_maps)
from jax_snlc_core.peak_time import (OPTMASK_SETPKMJD_FLUXMAX, OPTMASK_SETPKMJD_FLUXMAX3,
                                     fmax_clump_peakmjd, naive_fluxmax_peakmjd)
from jax_snlc_core.snana_files import MAG_UNDEFINED, MAG_ZEROFLUX, is_ignorefile
from jax_snlc_core.random import stream_key
from jax_snlc_sim.scatter import scatter_factors
from jax_snlc_sim.hostlib.sersic import (APERTURE_PSFSIG, compute_galfrac,
                                       compute_sersic_bn)

from .observations import epoch_layout

logger = logging.getLogger(__name__)

# PSF sigma of the 1" aperture SIM_GALFRAC uses (snlc_sim.c:26147)
GALFRAC_PSFSIG = 1.0 / 2.355
LOG_SAFE_FLOOR = 1e-30

# GEN_SNHOST_GALMAG uses R_V=3.1 and law 94 (sntools_host.c:9256, :9278).
HOSTGAL_MW_RV = 3.1


def fluxcal_to_magnitude(fluxcal: jnp.ndarray, filter_indices: jnp.ndarray,
                         static_data: Dict) -> jnp.ndarray:
    """Magnitude from FLUXCAL, with SNANA's null sentinel where there is no
    flux to take a logarithm of.

    The floor is on the band integral rather than on FLUXCAL, so the epochs
    called null do not move with the zero point: FLUXCAL carries a factor
    ``10^(0.4 zp) / ref`` on top of it.
    """
    zp = static_data['zp_fluxcal']
    scale = static_data['fluxcal_scale_mag_offset']
    floor = (LOG_SAFE_FLOOR * jnp.power(10.0, 0.4 * zp) * scale
             / static_data['filter_ref_fluxes'][filter_indices])
    return jnp.where(
        (fluxcal <= floor) | jnp.isnan(fluxcal),
        MAG_ZEROFLUX,
        zp - 2.5 * jnp.log10(jnp.maximum(fluxcal, LOG_SAFE_FLOOR)))


def compute_rest_wavelengths(resources, filter_indices, redshifts):
    """Per-obs rest-frame wavelengths = filter_mean / (1+z); NaN where invalid."""
    valid = filter_indices >= 0
    safe_fi = jnp.maximum(filter_indices, 0)
    mean_wave = jnp.array(resources.photometry.filter_mean_wavelengths, dtype=jnp.float32)
    rest = mean_wave[safe_fi] / (1.0 + redshifts[:, None])
    return jnp.where(valid, rest, jnp.nan)


def host_mwxt_mag_per_band(resources, params):
    """MW extinction in mag added to every HOSTGAL_MAG, at each output
    filter's mean wavelength.

    GEN_SNHOST_GALMAG hard-wires R_V = 3.1 and colour law 94 for the host
    magnitudes, whatever RV_MWCOLORLAW and OPT_MWCOLORLAW say, and adds the
    result to every DDLR-sorted host before the GALMAG gate
    (sntools_host.c:9256, :9278, :9287).
    """
    order = [resources.filter_name_to_index[n] for n in resources.filter_names]
    av_true = HOSTGAL_MW_RV * jnp.asarray(params['sim_mwebv'])
    axav = resources.host_mw_law_at_mean.extinction_curve(
        resources.photometry.filter_mean_wavelengths, HOSTGAL_MW_RV, av=av_true[:, None])
    return av_true[:, None] * axav[..., jnp.asarray(order)]


def mwxt_mag_per_band(simulation, resources, params):
    """MW extinction in mag at each output filter's mean wavelength.

    SNANA's GALextinct(RV, RV*MWEBV_true, LAMAVG) per band
    (GENLC.MWXT_MAG, snlc_sim.c:13713-13724), written to the TEXT MWXT_MAG
    block. The host magnitudes use their own law; see host_mwxt_mag_per_band.
    """
    rv = simulation.mw_dust.rv
    order = [resources.filter_name_to_index[n] for n in resources.filter_names]
    av_true = rv * jnp.asarray(params['sim_mwebv'])
    axav = resources.mw_law_at_mean.extinction_curve(
        resources.photometry.filter_mean_wavelengths, rv, av=av_true[:, None])
    return av_true[:, None] * axav[..., jnp.asarray(order)]


def _interp_galmag(galmag_table, psfsig):
    """GALMAG linearly interpolated across the 9 PSF apertures at ``psfsig``
    arcsec (interp_GALMAG_HOSTLIB, sntools_host.c:4922-4962). Out-of-range
    PSF values are pulled just inside the grid as SNANA does.

    ``galmag_table`` is (..., 10), aperture-mag table with the aperture axis
    last: col 0 is the total mag, cols 1..9 the 9 PSF apertures. ``psfsig``
    is a scalar or (...) PSF sigma in arcsec.
    """
    profiles = galmag_table[..., 1:]
    psf_grid = jnp.asarray(APERTURE_PSFSIG[1:])
    n_ap = psf_grid.shape[0]
    psfsig = jnp.broadcast_to(jnp.asarray(psfsig), profiles.shape[:-1])
    psf_c = jnp.clip(psfsig, psf_grid[0] + 1.0e-4, psf_grid[-1] - 1.0e-4)
    idx = jnp.clip(jnp.searchsorted(psf_grid, psf_c, side='right') - 1,
                   0, n_ap - 2)
    lo = jnp.take_along_axis(profiles, idx[..., None], axis=-1)[..., 0]
    hi = jnp.take_along_axis(profiles, (idx + 1)[..., None], axis=-1)[..., 0]
    frac = (psf_c - psf_grid[idx]) / (psf_grid[idx + 1] - psf_grid[idx])
    return lo + frac * (hi - lo)


@jax.jit
def _compute_fluxgal_pe(galfrac, host_mags_total, axav_at_mean, av_mw,
                        neas, pixsize, filter_idx, zptavgs, gains):
    """JIT-compiled: GALFRAC → GALMAG → interpolate at PSF sigma → fluxgal_pe.

    ``galfrac`` is (n_sne, 10), GALFRAC at 9 PSF apertures plus index 0.
    ``host_mags_total`` is (n_sne, n_filters), the raw HOSTLIB magnitudes.
    ``axav_at_mean`` is (n_filters,) or (n_sne, n_filters), A(λ)/A(V)
    at filter mean wavelengths (per event when the law's curve reads AV).
    ``av_mw`` is (n_sne,), MW A_V per SN. ``neas`` is (n_sne, max_obs),
    noise-equivalent area in pixel^2; ``pixsize`` (n_sne,) pixel size in
    arcsec; ``filter_idx`` (n_sne, max_obs) filter index per observation
    (clipped >=0); ``zptavgs``/``gains`` (n_sne, max_obs) zeropoints/CCD
    gains.

    Returns ``(fluxgal_pe, galmag_all)``: ``fluxgal_pe`` is (n_sne,
    max_obs), the host galaxy photoelectrons; ``galmag_all`` is (n_sne,
    n_filters, 10), GALMAG at all PSF apertures (for SB_FLUXCAL).
    """
    dm = jnp.where(galfrac > 0, -2.5 * jnp.log10(jnp.maximum(galfrac, LOG_SAFE_FLOOR)), 30.0)

    mwxt = av_mw[:, None] * axav_at_mean
    host_mags_dimmed = host_mags_total + mwxt

    galmag_all = host_mags_dimmed[:, :, None] + dm[:, None, :]

    # epoch PSF sigma from NEA (gen_fluxNoise_calc, snlc_sim.c:27794)
    psf_sigma_arcsec = pixsize[:, None] * jnp.sqrt(neas / (4.0 * jnp.pi))
    galmag_per_obs = jax.vmap(lambda g, idx: g[idx])(galmag_all, filter_idx)
    galmag = _interp_galmag(galmag_per_obs, psf_sigma_arcsec)

    arg = 0.4 * (zptavgs - galmag)
    fluxgal_pe = gains * jnp.power(10.0, arg)

    valid = (galmag < 90.0)
    fluxgal_pe = jnp.where(valid, fluxgal_pe, 0.0)

    return fluxgal_pe, galmag_all


class Epoch(NamedTuple):
    """One observation's inputs to the noise model: its true brightness, the
    three random draws, its zero point and gain, the host's photo-electrons,
    the FLUXERRMODEL scales and its calculated variances (noise_variances)."""
    magnitude: jnp.ndarray
    fluxcal: jnp.ndarray
    noise_value: jnp.ndarray
    noise_T: jnp.ndarray
    noise_F: jnp.ndarray
    zptavg: jnp.ndarray
    gain: jnp.ndarray
    fluxgal_pe: jnp.ndarray
    sqscale_true: jnp.ndarray
    sqscale_data: jnp.ndarray
    filter_idx: jnp.ndarray
    flux_pe: jnp.ndarray
    npe_over_fluxcal: jnp.ndarray
    sqsig_S: jnp.ndarray          # source, host, sky and readout
    sqsig_SZ: jnp.ndarray         # sqsig_S plus the drawn zero-point term
    sqsig_T: jnp.ndarray          # template
    sqsig_calc_data: jnp.ndarray  # SQSIG_CALC_DATA
    snr_calc_szt: jnp.ndarray
    snr_calc_st: jnp.ndarray


def _noise_sigma(variance):
    """An inactive noise component has zero width and zero derivative."""
    active = variance != 0.0
    return jnp.where(active, jnp.sqrt(jnp.where(active, variance, 1.0)), 0.0)


def _noise_model_single_observation(
    epoch: Epoch,
    *,
    smearflag_flux: int,
    random_template_noise: bool = False,
    magmonitor_snr: int = 0,
    monitor_enabled: bool | None = None,
    fluxerr_cor_log10snr_grid=None,
    fluxerr_cor_scale_per_filter=None,
    fluxerr_add_per_filter=None,
    exposure_time_per_filter=None,
    fluxerr_cor_active: bool = False,
    fudge_fluxerr_sq_per_filter=None,
    fudge_fluxerr2_sq_per_filter=None,
    fudge_magerr_per_filter=None,
):
    """Mirror SNANA's per-observation flux/noise bookkeeping.

    It starts from the epoch's calculated variances (`noise_variances`,
    gen_fluxNoise_calc); the error-fudge chain follows
    gen_fluxNoise_fudge_diag (:27968-28134) in SNANA's order: FUDGE_MAGERR,
    FUDGESCALE_FLUXERR, FUDGESCALE_FLUXERR2, FLUXERRMODEL, then the older
    SIMLIB FLUXERR_ADD/FLUXERR_COR block, and the noise application follows
    gen_fluxNoise_apply (:28423-28526).

    Returns (fluxcal_noisy, sigma_fluxcal, snr_calc_szt, snr_measured,
    snr_mon).
    """
    (magnitude, fluxcal, noise_value, noise_T, noise_F, zptavg, gain,
     fluxgal_pe, sqscale_true, sqscale_data, filter_idx, flux_pe, npe_over_fluxcal,
     sqsig_S_true, sqsig_SZ_true, sqsig_T_true, sqsig_calc_data, snr_calc_szt,
     snr_calc_st) = epoch
    valid = (
        jnp.isfinite(magnitude)
        & jnp.isfinite(fluxcal)
        & jnp.isfinite(zptavg)
        & (gain > 0)
    )

    # error-fudge chain (gen_fluxNoise_fudge_diag, snlc_sim.c:27968)
    sqsig_data = sqsig_calc_data

    fi_safe = jnp.maximum(filter_idx, 0)

    # FUDGE_MAGERR (snlc_sim.c:28005-28012): quadrature add to SQSIG_TRUE[S]
    # and the reported error only; the SZ draw variance is untouched.
    if fudge_magerr_per_filter is not None:
        magerr = fudge_magerr_per_filter[fi_safe]
        flux_err = flux_pe * (1.0 - jnp.power(10.0, -0.4 * magerr))
        sq_magerr = jnp.where(magerr > 1.0e-9, flux_err * flux_err, 0.0)
        sqsig_S_true = sqsig_S_true + sq_magerr
        sqsig_data = sqsig_data + sq_magerr

    # FUDGESCALE_FLUXERR (:28015-28022): every TRUE component and DATA.
    if fudge_fluxerr_sq_per_filter is not None:
        f1sq = fudge_fluxerr_sq_per_filter[fi_safe]
        sqsig_S_true = sqsig_S_true * f1sq
        sqsig_SZ_true = sqsig_SZ_true * f1sq
        sqsig_T_true = sqsig_T_true * f1sq
        sqsig_data = sqsig_data * f1sq

    # FUDGESCALE_FLUXERR2 (:28026-28031): reported error only.
    if fudge_fluxerr2_sq_per_filter is not None:
        sqsig_data = sqsig_data * fudge_fluxerr2_sq_per_filter[fi_safe]

    # FLUXERRMODEL (:28034-28094): the extra TRUE variance is the F component
    sqsig_F_true = (sqscale_true - 1.0) * sqsig_S_true
    sqsig_T_true = sqsig_T_true * sqscale_true
    sqsig_data = sqsig_data * sqscale_data

    # SIMLIB FLUXERR_ADD + FLUXERR_COR (:28105-28134)
    if fluxerr_cor_active:
        st_orig = sqsig_S_true + sqsig_T_true
        add_val = fluxerr_add_per_filter[fi_safe]
        xt = exposure_time_per_filter[fi_safe]
        sqadderr = jnp.where((add_val > 1.0e-6) & (xt == 1.0),
                             jnp.square(add_val * npe_over_fluxcal), 0.0)
        sqsig_S_true = sqsig_S_true + sqadderr
        sqsig_SZ_true = sqsig_SZ_true + sqadderr
        sqsig_data = sqsig_data + sqadderr
        # :28127-28129
        snr_fec = snr_calc_st * jnp.sqrt(st_orig / (st_orig + sqadderr))
        fec_scale = jnp.interp(
            jnp.log10(snr_fec), fluxerr_cor_log10snr_grid,
            fluxerr_cor_scale_per_filter[fi_safe])
        fec_sq = fec_scale * fec_scale
        sqsig_S_true = sqsig_S_true * fec_sq
        sqsig_SZ_true = sqsig_SZ_true * fec_sq
        sqsig_T_true = sqsig_T_true * fec_sq
        sqsig_F_true = sqsig_F_true * fec_sq
        sqsig_data = sqsig_data * fec_sq

    # MAGMONITOR_SNR (snlc_sim.c:27884-27888, :28186-28190)
    if (magmonitor_snr > 10 if monitor_enabled is None else monitor_enabled):
        fluxmon_pe = gain * jnp.power(10.0, 0.4 * (zptavg - magmonitor_snr))
        sqsig_mon = sqsig_calc_data - flux_pe + fluxmon_pe
        snr_mon = fluxmon_pe / jnp.sqrt(sqsig_mon)
        snr_mon = snr_mon / jnp.sqrt(sqsig_data / sqsig_calc_data)
        snr_mon = jnp.where(valid, snr_mon, 0.0)
    else:
        snr_mon = jnp.zeros_like(flux_pe)

    # SHIFT_SZ, SHIFT_T, SHIFT_F (snlc_sim.c:28474-28482)
    sigma_SZ_calc = _noise_sigma(sqsig_SZ_true) / npe_over_fluxcal
    sigma_T_calc = _noise_sigma(sqsig_T_true) / npe_over_fluxcal
    sigma_F_calc = _noise_sigma(sqsig_F_true) / npe_over_fluxcal

    # snlc_sim.c:28464-28472
    if smearflag_flux <= 0:
        fluxcal_noisy = fluxcal
    elif random_template_noise:
        # SIMLIB_MSKOPT 8: search random over the combined sigma (snlc_sim.c:28493-28500)
        fluxcal_noisy = fluxcal + noise_value * _noise_sigma(
            sigma_SZ_calc * sigma_SZ_calc + sigma_T_calc * sigma_T_calc)
    else:
        fluxcal_noisy = (
            fluxcal
            + noise_value * sigma_SZ_calc
            + noise_T * sigma_T_calc
            + noise_F * sigma_F_calc
        )
    flux_pe_obs = fluxcal_noisy * npe_over_fluxcal

    # Reported error re-centred on the observed flux (:28504-28512).
    sqerr_ran = jnp.where(flux_pe_obs > 0.0, flux_pe_obs - flux_pe, -flux_pe)
    sqsig_final_data = sqsig_data + sqerr_ran

    # SMEARFLAG_FLUX 2: no source or host shot noise in the reported error (:28518-28523)
    if (smearflag_flux & 2) > 0:
        sqsig_final_data = sqsig_final_data - (flux_pe + fluxgal_pe)

    sigma_fluxcal = jnp.sqrt(sqsig_final_data) / npe_over_fluxcal

    snr_measured = jnp.where(
        sigma_fluxcal > 0, fluxcal_noisy / sigma_fluxcal, 0.0
    )

    fluxcal_noisy = jnp.where(valid, fluxcal_noisy, jnp.nan)
    sigma_fluxcal = jnp.where(valid, sigma_fluxcal, jnp.nan)
    snr_calc_szt = jnp.where(valid, snr_calc_szt, jnp.nan)
    snr_measured = jnp.where(valid, snr_measured, jnp.nan)

    return fluxcal_noisy, sigma_fluxcal, snr_calc_szt, snr_measured, snr_mon


def _draw_redcov_fudge_factored(F_per_obs, w_shared, z_indep):
    """Per-SN factor-decomposition draw of the correlated FLUXERRMODEL fudge,
    as a unit-variance standard normal per obs (the noise kernel supplies the
    per-obs sigma).

    For an SN in a SIMLIB field, every obs i has a factor-loading vector
    `F_per_obs[i, :]` of shape (n_factors,) with at most one nonzero entry
    `sqrt(rho_k)` (corresponding to the REDCOV entry that applies to obs i,
    or all-zero if no REDCOV applies, in which case the draw reduces to
    iid z).

    The draw is:
        u_i = F_obs[i, :] · w  +  sqrt(1 - F_obs · F_obs) · z_i

    where w ~ N(0, I_{n_factors}) is sampled ONCE per SN (shared across the
    SN's obs) and z_i ~ N(0, 1) are independent per obs.

    Verification:
      E[u_i u_j] for i != j = F_obs[i,:] · F_obs[j,:]
        = rho_k  (if both in icov group k)
        = 0      (otherwise; F_obs is zero for one)
      E[u_i^2] = F_obs · F_obs + (1 - F_obs · F_obs) = 1

    IMPORTANT: SNR filtering (SNRMIN_REDCOV) must be handled by the CALLER
    by zeroing F_per_obs for below-SNR epochs before calling this function.
    SNANA's rule (snlc_sim.c:28301-28306) excludes below-SNR epochs from the
    Cholesky matrix, but they still get an iid fudge draw in the main noise
    step. Zeroing F_per_obs reproduces that: below-SNR epochs get
    `u_i = z_i` (uncorrelated fudge noise), matching SNANA.

    ``F_per_obs`` is (max_obs, n_factors), the factor loadings (caller must
    zero rows that shouldn't participate in the correlated factor).
    ``w_shared`` is (n_factors,), the shared standard normals, one per
    factor; ``z_indep`` is (max_obs,), the independent standard normals per
    obs. Returns the (max_obs,) per-obs unit-variance fudge draw.
    """
    F_dot_w = F_per_obs @ w_shared
    F_self = jnp.sum(F_per_obs * F_per_obs, axis=-1)
    residual = jnp.sqrt(jnp.maximum(1.0 - F_self, 0.0))
    return F_dot_w + residual * z_indep


def add_photometric_noise(simulator, catalogue, zp_fluxcal,
                          magnitudes_all, fluxcals_all, template_fluxcals,
                          obs_schedule, noise_key, sn_parameters):
    """
    Add photometric noise using JAX for deterministic reproducibility, the
    host's light read from ``catalogue`` (the HostCatalogue, None without a
    HOSTLIB). ``template_fluxcals`` is the SN's own FLUXCAL in each epoch's
    template image (MJD_TEMPLATE), zero without one.

    Uses vmap over a small JIT-compiled function for fast compilation.
    """
    n_sne, max_obs = magnitudes_all.shape

    obs_arrays = {
        'filter_indices': jnp.asarray(obs_schedule['filter_indices']),
        'mjds': jnp.asarray(obs_schedule['mjds']),
        'zptavgs': jnp.asarray(obs_schedule['zptavgs']),
        'zpterrs': jnp.asarray(obs_schedule['zpterrs']),
        'skysigs': jnp.asarray(obs_schedule['skysigs']),
        'psf1s': jnp.asarray(obs_schedule['psf1s']),
        'neas': jnp.asarray(obs_schedule['neas']),
        'gains': jnp.asarray(obs_schedule['gains']),
        'readnoises': jnp.asarray(obs_schedule['readnoises']),
        'template_skysigs': jnp.asarray(obs_schedule['template_skysigs']),
        'template_readnoises': jnp.asarray(obs_schedule['template_readnoises']),
        'template_zpts': jnp.asarray(obs_schedule['template_zpts'])
    }

    # FUDGE_ZPTERR replaces the SIMLIB's per-epoch ZPTERR (snlc_sim.c:20707).
    fudge_zpterr = simulator.zeropoint_error_override
    if fudge_zpterr is not None:
        override = fudge_zpterr[jnp.maximum(obs_arrays['filter_indices'], 0)]
        obs_arrays['zpterrs'] = jnp.where(
            override > 1.0e-7, override, obs_arrays['zpterrs'])

    # search, fudge and template streams (gen_fluxNoise_randoms, snlc_sim.c:27636-27698)
    noise_shape = (n_sne, max_obs)
    key_S, key_F, key_T = jax.random.split(noise_key, 3)

    sn_idx = jnp.asarray(sn_parameters['sn_global_idx'], dtype=jnp.int32)
    keys_S = jax.vmap(lambda i: jax.random.fold_in(key_S, i))(sn_idx)
    keys_T = jax.vmap(lambda i: jax.random.fold_in(key_T, i))(sn_idx)
    keys_F = jax.vmap(lambda i: jax.random.fold_in(key_F, i))(sn_idx)
    noise_values = jax.vmap(lambda k: jax.random.normal(k, (max_obs,)))(keys_S)

    simlib_mskopt = simulator.survey.simlib_mskopt
    random_template_noise = bool(simlib_mskopt & 8)
    if random_template_noise:
        noise_T = jnp.zeros((n_sne, max_obs), dtype=jnp.float32)
    else:
        # one template draw per (event, field, filter) (snlc_sim.c:27683-27698)
        n_filters = len(simulator.photometry.transmissions)
        field_idx_per_sn = jnp.asarray(
            sn_parameters['field_idx'], dtype=jnp.int32)
        fi_safe = jnp.maximum(
            jnp.asarray(obs_arrays['filter_indices'], dtype=jnp.int32), 0)
        keys_TF = jax.vmap(jax.random.fold_in)(keys_T, field_idx_per_sn)
        template_randoms = jax.vmap(
            lambda k: jax.random.normal(k, (n_filters,)))(keys_TF)
        noise_T = jnp.take_along_axis(template_randoms, fi_safe, axis=1)

    hostlib_mskopt = simulator.host_settings.mskopt
    smearflag_hostgal = bool(hostlib_mskopt & 2)
    need_host_for_fluxerr = bool(simulator.fluxerr_needs_host_vars)
    if smearflag_hostgal or need_host_for_fluxerr:
        fluxgal_pe_all, galmag_aperture_table, sb_fluxcal_stacked = (
            _compute_host_galaxy_noise(simulator, catalogue,
                                       zp_fluxcal, sn_parameters, obs_arrays,
                                       n_sne, max_obs))
        if not smearflag_hostgal:
            # FLUXERRMODEL reads SBMAG/GALMAG; the host shot noise stays out
            fluxgal_pe_all = jnp.zeros_like(fluxgal_pe_all)
    else:
        fluxgal_pe_all = jnp.zeros((n_sne, max_obs))
        galmag_aperture_table = None
        sb_fluxcal_stacked = None
    # HOSTMAG and the 1" aperture mag SIM_GALFRAC references (snlc_sim.c:26147)
    if galmag_aperture_table is not None:
        host_galmag_total = galmag_aperture_table[:, :, 0]
        host_galmag_galfrac = _interp_galmag(galmag_aperture_table, GALFRAC_PSFSIG)
    else:
        host_galmag_total = host_galmag_galfrac = None

    (flux_pe, npe_over_fluxcal, sqsig_S, sqsig_SZ, sqsig_T, sqsig_calc_data,
     snr_calc_szt, snr_calc_st) = noise_variances(
        fluxcals_all, obs_arrays['zptavgs'], obs_arrays['zpterrs'],
        obs_arrays['gains'], obs_arrays['neas'], obs_arrays['skysigs'],
        obs_arrays['readnoises'], obs_arrays['template_skysigs'],
        obs_arrays['template_readnoises'], obs_arrays['template_zpts'],
        fluxgal_pe_all, template_fluxcals,
        smearflag_zeropt=simulator.instrument.smearflag_zeropt,
        zp_fluxcal=zp_fluxcal)

    fluxerrmodel_enabled = bool(simulator.fluxerr_enabled)
    if fluxerrmodel_enabled:
        (err_model_val, err_scale_true, err_scale_data,
         err_applies_true, err_applies_data) = _evaluate_fluxerr_model(
            simulator, zp_fluxcal,
            sn_parameters, obs_arrays, snr_calc_szt,
            sb_fluxcal_stacked=sb_fluxcal_stacked,
            galmag_aperture_table=galmag_aperture_table,
        )

        sqscale_true, sqscale_data = fluxerr_model_sqscales(
            err_model_val, err_scale_true, err_scale_data,
            err_applies_true, err_applies_data, npe_over_fluxcal,
            sqsig_calc_data,
            bool(simulator.fluxerr_is_errscale))

        # snlc_sim.c:28065-28079; under a trace the kernel gives NaN flux instead
        if not isinstance(sqscale_true, jax.core.Tracer):
            bad = (jnp.asarray(obs_arrays['filter_indices']) >= 0) \
                & (sqscale_true < 1.0)
            if bool(jnp.any(bad)):
                raise SNANAError(
                    "FLUXERRMODEL gives SQSIG_FUDGE < 0 (fluxErrScale probably < 1); "
                    "SNANA aborts here.",
                    function_name="add_photometric_noise",
                )

        if simulator.fluxerr_has_redcov:
            snrmin = simulator.instrument.fluxerr_snrmin_redcov
            field_idx = jnp.asarray(sn_parameters['field_idx'], dtype=jnp.int32)
            fi_safe = jnp.maximum(obs_arrays['filter_indices'], 0)
            F_per_obs = simulator.fluxerr_redcov_factors[field_idx[:, None], fi_safe]
            # below-SNR epochs draw iid (SNR_CALC_ST, snlc_sim.c:27894, :28307-28308)
            in_redcov_group = (
                (obs_arrays['filter_indices'] >= 0)
                & (snr_calc_st >= snrmin)
            )
            F_masked = jnp.where(in_redcov_group[..., None], F_per_obs, 0.0)
            n_factors = simulator.fluxerr_n_factors
            keys_wz = jax.vmap(
                lambda k: jax.random.split(k, 2))(keys_F)
            w_shared = jax.vmap(
                lambda k: jax.random.normal(k, (n_factors,)))(keys_wz[:, 0])
            z_indep = jax.vmap(
                lambda k: jax.random.normal(k, (max_obs,)))(keys_wz[:, 1])
            noise_F = jax.vmap(_draw_redcov_fudge_factored)(
                F_masked, w_shared, z_indep)
        else:
            noise_F = jax.vmap(
                lambda k: jax.random.normal(k, (max_obs,)))(keys_F)
    else:
        sqscale_true = jnp.ones(noise_shape, dtype=jnp.float32)
        sqscale_data = jnp.ones(noise_shape, dtype=jnp.float32)
        noise_F = jnp.zeros(noise_shape, dtype=jnp.float32)

    (fluxcals_noisy, fluxcalerrs, snrs_model, snrs_measured,
     snrs_mon) = simulate_noise(simulator.noise, simulator.noise_options, Epoch(
        magnitude=magnitudes_all, fluxcal=fluxcals_all,
        noise_value=noise_values, noise_T=noise_T, noise_F=noise_F,
        zptavg=obs_arrays['zptavgs'], gain=obs_arrays['gains'],
        fluxgal_pe=fluxgal_pe_all,
        sqscale_true=sqscale_true, sqscale_data=sqscale_data,
        filter_idx=obs_arrays['filter_indices'], flux_pe=flux_pe,
        npe_over_fluxcal=npe_over_fluxcal, sqsig_S=sqsig_S, sqsig_SZ=sqsig_SZ,
        sqsig_T=sqsig_T, sqsig_calc_data=sqsig_calc_data,
        snr_calc_szt=snr_calc_szt, snr_calc_st=snr_calc_st))

    return (fluxcals_noisy, fluxcalerrs, snrs_model, snrs_measured,
            fluxgal_pe_all, host_galmag_total, host_galmag_galfrac,
            sb_fluxcal_stacked, snrs_mon)


def _compute_host_galaxy_noise(simulator, catalogue, zp_fluxcal,
                               sn_parameters, obs_arrays, n_sne, max_obs):
    """Compute host galaxy photoelectrons per observation matching SNANA GEN_SNHOST_GALMAG.

    Uses position-dependent GALFRAC (fraction of galaxy flux in PSF aperture)
    computed via polar grid integration over the Sersic surface brightness profile.

    Per-epoch: interpolates GALMAG across 9 PSF aperture bins at the epoch's
    actual PSF sigma, then converts to photoelectrons.
    """
    filter_names = simulator.calibration_filter_names
    n_filters = len(filter_names)

    a_sn = jnp.asarray(sn_parameters.get('host_a_SNGALSEP', jnp.zeros(n_sne)))
    b_sn = jnp.asarray(sn_parameters.get('host_b_SNGALSEP', jnp.zeros(n_sne)))

    sersic = catalogue.sersic
    gal_idx = jnp.asarray(sn_parameters['host_galaxy_index'],
                          dtype=jnp.int32)
    a_half = sersic['a'][:, gal_idx].T
    b_half = sersic['b'][:, gal_idx].T
    sersic_n_arr = sersic['n'][:, gal_idx].T
    sersic_w_arr = sersic['w'][:, gal_idx].T

    # SNANA uses INTEG_SUM[NSERSIC_TABLE-1] for every component
    sersic_data = simulator.hostlib.selector.sersic_table
    integ_sum_fixed = sersic_data.integ_total[-1]
    sersic_bn_arr = compute_sersic_bn(
        sersic_n_arr, sersic_data.bn_n_lookup, sersic_data.bn_lookup)

    galfrac = compute_galfrac(
        a_sn, b_sn, a_half, b_half,
        sersic_n_arr, sersic_w_arr, sersic_bn_arr, integ_sum_fixed,
        sersic_data.gauss2d_integ
    )

    mag_columns = []
    for fname in filter_names:
        short = filter_band(fname)
        key = f'host_{short}_obs'
        if key in sn_parameters:
            mag_columns.append(jnp.asarray(sn_parameters[key]))
        else:
            mag_columns.append(jnp.full(n_sne, MAG_UNDEFINED))
    host_mags_lib = jnp.stack(mag_columns, axis=1)

    mwebv = jnp.asarray(sn_parameters['sim_mwebv'])
    av_mw = HOSTGAL_MW_RV * mwebv
    axav_at_mean = simulator.host_mw_law_at_mean.extinction_curve(
        simulator.photometry.filter_mean_wavelengths, HOSTGAL_MW_RV, av=av_mw[:, None])

    pixsize = _get_field_pixsizes(simulator, sn_parameters)
    filter_idx = obs_arrays['filter_indices']
    filter_idx_safe = jnp.clip(filter_idx, 0, n_filters - 1)

    fluxgal_pe, galmag_all = _compute_fluxgal_pe(
        galfrac, host_mags_lib, axav_at_mean, av_mw,
        obs_arrays['neas'], pixsize, filter_idx_safe,
        obs_arrays['zptavgs'], obs_arrays['gains']
    )

    fluxgal_pe = jnp.where(filter_idx >= 0, fluxgal_pe, 0.0)
    sb_fluxcal_stacked = _compute_sb_fluxcal(
        simulator.host_settings, zp_fluxcal, galmag_all)
    return fluxgal_pe, galmag_all, sb_fluxcal_stacked


def _get_field_pixsizes(resources, sn_parameters):
    """Get pixel size (arcsec) per SN from assigned SIMLIB field."""
    field_indices = jnp.asarray(sn_parameters['field_idx'], dtype=jnp.int32)
    pixsizes = resources.flat_field_index['pixsizes']
    return jnp.take(pixsizes, field_indices)


def _compute_sb_fluxcal(host, zp_fluxcal, galmag_all):
    """Surface-brightness FLUXCAL per (sn, filter) at HOSTLIB_SBRADIUS/2.

    ``galmag_all`` col 0 is the total mag; cols 1..9 are the 9 PSF apertures.
    """
    psfsig = host.sbradius / 2.0
    area = 4.0 * np.pi * psfsig**2
    sb_mag = (_interp_galmag(galmag_all, psfsig)
              + 2.5 * jnp.log10(area))
    return jnp.power(10.0, -0.4 * (sb_mag - zp_fluxcal))


def per_filter_array(resources, cfg_map, neutral):
    """A {band: value} map as a per-filter-index array; None when inert."""
    names = resources.filter_names
    short = {filter_band(n): v for n, v in cfg_map.items()}
    vals = np.array([float(short.get(filter_band(n), neutral))
                     for n in names])
    return jnp.asarray(vals) if np.any(vals != neutral) else None


def _fudge_fluxerr_arrays(simulator):
    """FUDGESCALE_FLUXERR(2) as squared per-filter-index arrays (None = off)."""
    def build(cfg_map):
        arr = per_filter_array(simulator, cfg_map, 1.0)
        return None if arr is None else arr ** 2
    return (build(simulator.instrument.fudgescale_fluxerr),
            build(simulator.instrument.fudgescale_fluxerr2))


def prepare_noise(simulator, zp_fluxcal):
    """Prepare noise coefficients and structural options for the shared kernel."""
    smearflag_flux = simulator.instrument.smearflag_flux
    random_template_noise = bool(simulator.survey.simlib_mskopt & 8)

    fc = simulator.simlib_fluxerr_cor
    fluxerr_cor_active = bool(fc.enabled) and not bool(
        simulator.survey.simlib_mskopt & 32)
    fc_log10snr_grid = jnp.asarray(fc.log10snr) if fluxerr_cor_active else None
    fc_scale_per_filter = jnp.asarray(fc.scale) if fluxerr_cor_active else None
    # FLUXERR_ADD is given on the original zero point (snlc_sim.c:18704-18706)
    fc_add_per_filter = (
        jnp.asarray(simulator.simlib_fluxerr_add)
        * 10.0 ** (0.4 * (zp_fluxcal - ZEROPOINT_FLUXCAL_SNANA_ORIG))
        if fluxerr_cor_active else None)
    if fluxerr_cor_active:
        # EXPOSURE_TIME_FILTER gates the older additive term
        # (snlc_sim.c:28114-28116); default 1.0 for every band.
        xt = per_filter_array(
            simulator, simulator.survey.exposure_time, 1.0)
        n_filters = len(simulator.filter_names)
        exposure_per_filter = (xt if xt is not None
                               else jnp.ones(n_filters))
    else:
        exposure_per_filter = None
    fudge1_sq, fudge2_sq = _fudge_fluxerr_arrays(simulator)
    fudge_magerr = per_filter_array(
        simulator, simulator.instrument.fudge_magerr, 0.0)
    magmonitor_snr = int(simulator.output.magmonitor_snr)

    arrays = {
        'magmonitor_snr': float(magmonitor_snr),
        'fluxerr_cor_log10snr_grid': fc_log10snr_grid,
        'fluxerr_cor_scale_per_filter': fc_scale_per_filter,
        'fluxerr_add_per_filter': fc_add_per_filter,
        'exposure_time_per_filter': exposure_per_filter,
        'fudge_fluxerr_sq_per_filter': fudge1_sq,
        'fudge_fluxerr2_sq_per_filter': fudge2_sq,
        'fudge_magerr_per_filter': fudge_magerr,
    }
    options = (
        ('smearflag_flux', smearflag_flux),
        ('random_template_noise', random_template_noise),
        ('fluxerr_cor_active', fluxerr_cor_active),
        ('monitor_enabled', magmonitor_snr > 10),
    )
    return arrays, options


@jax.jit(static_argnames=('options',))
def simulate_noise(arrays, options, epochs):
    """Noise for a batch, with numerical coefficients supplied explicitly."""
    def observation(epoch):
        return _noise_model_single_observation(epoch, **dict(options), **arrays)
    return jax.vmap(jax.vmap(observation))(epochs)


def _gather_sbmag_per_obs(sb_fluxcal_stacked, obs_arrays, zp_fluxcal):
    """Per-(sn, obs) SBMAG from the (n_sne, n_filters) SB_FLUXCAL array."""
    sb_fluxcal = jnp.asarray(sb_fluxcal_stacked, dtype=jnp.float32)
    sb_mag = -2.5 * jnp.log10(sb_fluxcal) + zp_fluxcal

    n_sne = obs_arrays['filter_indices'].shape[0]
    fi = jnp.maximum(obs_arrays['filter_indices'], 0)
    sn_arange = jnp.arange(n_sne)[:, None]
    return sb_mag[sn_arange, fi]


def _gather_galmag_nea_per_obs(galmag_aperture_table, obs_arrays, pixsize):
    """Per-(sn, obs) GALMAG_NEA, interpolated across the 9 PSF apertures at
    the epoch PSF sigma psfsig_arcsec = pixsize * sqrt(NEA/(4π))
    (gen_fluxNoise_calc, snlc_sim.c:27794).

    ``galmag_aperture_table`` has shape (n_sne, n_filters, 10); col 0 is the
    total mag, cols 1..9 the 9 per-aperture mags. ``pixsize`` is (n_sne,).
    """
    if galmag_aperture_table is None:
        n_sne, max_obs = obs_arrays['filter_indices'].shape
        return jnp.full((n_sne, max_obs), MAG_UNDEFINED, dtype=jnp.float32)
    galmag_all = jnp.asarray(galmag_aperture_table, dtype=jnp.float32)
    n_sne = galmag_all.shape[0]

    nea = jnp.asarray(obs_arrays['neas'], dtype=jnp.float32)
    psfsig_arcsec = pixsize[:, None] * jnp.sqrt(nea / (4.0 * jnp.pi))

    fi = jnp.maximum(obs_arrays['filter_indices'], 0)
    sn_arange = jnp.arange(n_sne)[:, None]
    galmag_per_obs = galmag_all[sn_arange, fi]

    return _interp_galmag(galmag_per_obs, psfsig_arcsec)


def _build_fluxerr_query_per_obs(resources, zp_fluxcal, sn_parameters,
                                 obs_arrays, snr_calc_szt,
                                 sb_fluxcal_stacked, galmag_aperture_table):
    """Build the per-(sn, obs, n_dims) query tensor for the active variables.

    The dim axis order matches `resources.fluxerr_active_vars`, which in turn
    matches the file's VARNAMES (and therefore
    `resources.fluxerr_grid_origins/steps`).
    """
    active = resources.fluxerr_active_vars
    n_sne, max_obs = obs_arrays['filter_indices'].shape

    values = {}
    if 'MJD' in active:
        values['MJD'] = jnp.asarray(obs_arrays['mjds'], dtype=jnp.float32)
    if 'PSF' in active:
        pixsize = _get_field_pixsizes(resources, sn_parameters)
        psf1 = jnp.asarray(obs_arrays['psf1s'], dtype=jnp.float32)
        values['PSF'] = psf1 * pixsize[:, None] * 2.3548  # FWHM, arcsec
    if 'SKYSIG' in active:
        values['SKYSIG'] = jnp.asarray(obs_arrays['skysigs'], dtype=jnp.float32)
    if 'ZP' in active:
        values['ZP'] = jnp.asarray(obs_arrays['zptavgs'], dtype=jnp.float32)
    if 'LOGSNR' in active:
        # bounded below at -0.9 (snlc_sim.c:28037-28040)
        logsnr = jnp.log10(snr_calc_szt)
        values['LOGSNR'] = jnp.maximum(logsnr, -0.9)
    if 'SBMAG' in active:
        # capped at 32 (snlc_sim.c:27998)
        sbmag = _gather_sbmag_per_obs(sb_fluxcal_stacked, obs_arrays, zp_fluxcal)
        values['SBMAG'] = jnp.minimum(sbmag, 32.0)
    if 'GALMAG' in active:
        pixsize = _get_field_pixsizes(resources, sn_parameters)
        values['GALMAG'] = _gather_galmag_nea_per_obs(
            galmag_aperture_table, obs_arrays, pixsize)
    if 'SNSEP' in active:
        snsep = jnp.asarray(
            sn_parameters.get('host_SNSEP', jnp.zeros(n_sne)), dtype=jnp.float32)
        values['SNSEP'] = jnp.broadcast_to(snsep[:, None], (n_sne, max_obs))

    return jnp.stack([values[name] for name in active], axis=-1)


def _evaluate_fluxerr_model(resources, zp_fluxcal, sn_parameters, obs_arrays,
                            snr_calc_szt, sb_fluxcal_stacked,
                            galmag_aperture_table):
    """Per-(sn, obs) FLUXERRMODEL value from the map serving the event's
    SIMLIB field and the epoch's band.

    Returns ``(errModelVal, scale_true, scale_data, applies_true,
    applies_data)``, each (n_sne, max_obs).
    """
    fx = resources
    field_idx = jnp.asarray(sn_parameters['field_idx'], dtype=jnp.int32)
    fi = obs_arrays['filter_indices']
    rows = fx.fluxerr_map_index_per_field[field_idx[:, None], jnp.maximum(fi, 0)]
    query = _build_fluxerr_query_per_obs(
        resources, zp_fluxcal, sn_parameters, obs_arrays, snr_calc_szt,
        sb_fluxcal_stacked=sb_fluxcal_stacked,
        galmag_aperture_table=galmag_aperture_table,
    )
    err_model_val = interpolate_fluxerr_maps(
        fx.fluxerr_grids, rows, query, fx.fluxerr_grid_origins, fx.fluxerr_grid_steps)

    # padding lanes get the no-op value
    no_op = 1.0 if fx.fluxerr_is_errscale else 0.0
    err_model_val = jnp.where(fi >= 0, err_model_val, no_op)

    return (err_model_val, fx.fluxerr_scale_true[rows], fx.fluxerr_scale_data[rows],
            fx.fluxerr_applies_true[rows], fx.fluxerr_applies_data[rows])


def precompute_restframe_filters(simulator, redshifts,
                                 mwebv) -> jnp.ndarray:
    """Observer-frame filters blue-shifted to rest frame, per SN.

    Processes all (n_sne × n_filters) combinations in one nested-vmap call
    over the pre-interpolated common-grid filter data. The band weights
    fold in the blueshift λ_rest = λ_obs / (1+z), the (T·λ·dλ)/∫(T·λ·dλ)
    normalisation, the observer-frame 1/(1+z) scaling, and the per-SN MW
    attenuation evaluated on the common grid.
    """
    common_wave = simulator.photometry.common_wave_grid
    rv_mw = simulator.mw_dust.rv
    axav_obs = simulator.mw_law_obs.extinction_curve(
        common_wave, rv_mw, av=rv_mw * mwebv[:, None])
    mw_att_all = jnp.power(10.0, -0.4 * rv_mw * mwebv[:, None] * axav_obs)

    compute_for_all_filters_and_sne = jax.vmap(
        band_weights_at_z,
        in_axes=(0, None, None, None, 0))

    return compute_for_all_filters_and_sne(
        redshifts,
        simulator.photometry.common_wave_grid,
        simulator.photometry.common_trans_grid,
        simulator.wavelength_grid,
        mw_att_all)


def grey_offsets(parameters, model, non1a_templates):
    """SN brightness offsets, grouped by the photometric validity checks."""
    grey_shift = jnp.asarray(parameters['genmag_off'])
    if non1a_templates:
        grey_shift = grey_shift + model.genmag_off_non1a
    host_shift = jnp.asarray(
        parameters.get('snmagshift', jnp.zeros_like(grey_shift)))
    magnitude_shift = (parameters['magoff'] + parameters['magsmear']
                       + parameters['lensdmu'] + parameters['mushift']
                       + model.mag_offset)
    return grey_shift, host_shift, magnitude_shift


# Retain the photometry boundary to limit changes from XLA fusion in float32.
@jax.jit
def simulate_photometry(simulator, sed_params, mjds, filter_indices, mask,
                        band_weights, sed_factor):
    """Evaluate batch photometry with model surfaces supplied as arguments."""
    data = dict(simulator.photometry_data, band_window=simulator.band_window,
                latetime_extrap=True)

    def one_sn(sed_params, mjds, filter_indices, mask, band_weights, factor):
        return simulator.model.evaluate(sed_params, mjds, filter_indices, mask,
                                        band_weights, data, sed_factor=factor)

    factor_axis = 0 if simulator.gensmear is not None else None
    return jax.vmap(
        one_sn, in_axes=(0, 0, 0, 0, 0, factor_axis))(
            sed_params, mjds, filter_indices, mask, band_weights, sed_factor)


def simulate_lightcurves(simulator, catalogue, sn_parameters,
                         obs_schedule, *, key) -> Dict:
    """Simulate light curves for one batch: SEDs at the observed phases,
    magnitudes/FLUXCAL through the rest-frame filters, the magnitude offsets and
    smear terms, PEAKMAG, photometric noise, and the per-obs validity mask.

    This is the core performance-critical GPU function; it is also traced
    whole by the SBI pipeline.
    """
    static_data = simulator.photometry_data
    zp_fluxcal = simulator.instrument.zp_fluxcal
    survey_mjd_span = simulator.survey_mjd_span
    n_sne = len(sn_parameters['sim_redshift_cmb'])

    # filter blueshift uses zHEL
    restframe_filters = precompute_restframe_filters(
        simulator,
        sn_parameters['sim_redshift_helio'], sn_parameters['sim_mwebv'])

    # (epoch, filter) pairs: the observations, SIM_PEAKMAG per filter, MJD_TEMPLATE per band
    max_obs = obs_schedule['phases'].shape[1]
    n_filters = restframe_filters.shape[1]
    obs_mask = (jnp.arange(max_obs)[None, :]
                < obs_schedule['n_obs_per_sn'][:, None])
    every_filter = jnp.broadcast_to(
        jnp.arange(n_filters, dtype=obs_schedule['filter_indices'].dtype),
        (n_sne, n_filters))
    peak_mjd = jnp.asarray(sn_parameters['peakmjd'])

    pair_mjds = [obs_schedule['mjds'],
                 jnp.broadcast_to(peak_mjd[:, None], (n_sne, n_filters))]
    # padded epochs carry -1, which would gather the wrong filter's weights
    pair_filters = [jnp.maximum(obs_schedule['filter_indices'], 0),
                    every_filter]
    pair_mask = [obs_mask, jnp.ones((n_sne, n_filters), dtype=bool)]

    per_band = [simulator.survey.mjd_template.get(filter_band(n), 0.0)
                for n in simulator.filter_names]
    has_template = any(v > 1.0 for v in per_band)
    template_per_band = jnp.asarray(per_band)
    if has_template:
        pair_mjds.append(jnp.broadcast_to(template_per_band[None, :],
                                          (n_sne, n_filters)))
        pair_filters.append(every_filter)
        pair_mask.append(jnp.broadcast_to(template_per_band[None, :] > 1.0,
                                          (n_sne, n_filters)))

    mjds = jnp.concatenate(pair_mjds, axis=1)
    filter_indices = jnp.concatenate(pair_filters, axis=1)
    mask = jnp.concatenate(pair_mask, axis=1)

    sed_params = {k: sn_parameters[k]
                  for k in simulator.model.sed_param_names()
                  if k in sn_parameters}
    sed_params['t0'] = peak_mjd
    sed_params['redshift'] = jnp.asarray(sn_parameters['sim_redshift_helio'])
    sed_params['av'] = jnp.asarray(sn_parameters['av'])
    sed_params['rv'] = jnp.asarray(sn_parameters['rv'])

    # GENMAG_SMEAR_MODELNAME: per-SN DeltaM(lam) on the rest-frame SED (genmag_SALT2.c:2862)
    smear_frac = None
    if simulator.gensmear is not None:
        smear_frac = scatter_factors(
            simulator.gensmear,
            jnp.asarray(sn_parameters['_gensmear_randoms']),
            simulator.wavelength_grid)

    model = simulator.model

    pair_fluxcal = simulate_photometry(
        simulator, sed_params, mjds, filter_indices, mask,
        restframe_filters, smear_frac)

    # grey offsets: GENMAG_OFF_GLOBAL (snlc_sim.c:12941) and GENMAG_OFF_NON1A (:14460)
    grey, host_shift, magnitude_shift = grey_offsets(
        sn_parameters, simulator.model_settings, model.non1a_templates)
    off_frac = jnp.power(10.0, -0.4 * grey)[:, None]
    pair_fluxcal = pair_fluxcal * off_frac

    pair_magnitudes = fluxcal_to_magnitude(
        pair_fluxcal, filter_indices, static_data)
    # A null magnitude carries the flux SNANA re-derives from mag 99
    # (genmag_SALT2.c:2267, snlc_sim.c:27781), not the model's raw value.
    null_fluxcal = jnp.power(
        10.0, 0.4 * (simulator.instrument.zp_fluxcal - MAG_ZEROFLUX))
    pair_fluxcal = jnp.where(
        pair_magnitudes == MAG_ZEROFLUX, null_fluxcal, pair_fluxcal)
    pair_magnitudes = jnp.where(mask, pair_magnitudes, jnp.nan)
    # Keep padding finite through the offsets; outputs are masked below.
    pair_fluxcal = jnp.where(mask, pair_fluxcal, 0.0)

    # HOSTLIB SNMAGSHIFT: genmag += shift (snlc_sim.c:8490)
    if 'snmagshift' in sn_parameters:
        host_shift = host_shift[:, None]
        pair_magnitudes = jnp.where(
            pair_magnitudes < 90,
            pair_magnitudes + host_shift,
            pair_magnitudes
        )
        flux_scale = jnp.power(10.0, -0.4 * host_shift)
        pair_fluxcal = jnp.where(
            pair_fluxcal != 0,
            pair_fluxcal * flux_scale,
            pair_fluxcal
        )

    # MAGOFF + MAGSMEAR + MAG_OFFSET_GENPDF (snlc_sim.c:8491)
    genmag_off_model = simulator.model_settings.genmag_off_model
    genmag_smear_filter = simulator.model_settings.genmag_smear_filter
    # WEAKLENS_DMU and MUSHIFT are terms of SNANA's MAGOFF sum, so they
    # share its -2<mag<50 gate (snlc_sim.c:8485-8497).
    total_shift = magnitude_shift[:, None]
    # GENMAG_OFF_MODEL: per-filter additive offset (snlc_sim.c:8486).
    if genmag_off_model:
        off_pf = jnp.array([genmag_off_model.get(filter_band(b), 0.0)
                            for b in simulator.photometry.filter_names])
        total_shift = total_shift + off_pf[filter_indices]
    # GENMAG_SMEAR_FILTER: per-filter coherent smear σ[band]·ran[SN,band]
    # (snlc_sim.c:29904); ran drawn once per (SN,filter) in the sampler.
    if genmag_smear_filter and '_magsmear_filter_ran' in sn_parameters:
        sig_pf = jnp.array([genmag_smear_filter.get(filter_band(b), 0.0)
                            for b in simulator.photometry.filter_names])
        ran = jnp.asarray(sn_parameters['_magsmear_filter_ran'])
        smear_pf = sig_pf[filter_indices] * jnp.take_along_axis(ran, filter_indices, axis=1)
        total_shift = total_shift + smear_pf
    # Keep the offset dependence at zero so autodiff sees its derivative.
    mag_in_range = (pair_magnitudes > -2) & (pair_magnitudes < 50)
    pair_magnitudes = jnp.where(
        mag_in_range,
        pair_magnitudes + total_shift,
        pair_magnitudes
    )
    flux_scale = jnp.power(10.0, -0.4 * total_shift)
    pair_fluxcal = jnp.where(
        mag_in_range,
        pair_fluxcal * flux_scale,
        pair_fluxcal
    )

    # GENMAG_SMEAR_ADDPHASECOR
    if '_phasecor_grid' in sn_parameters:
        pc_grid = jnp.asarray(sn_parameters['_phasecor_grid'])
        phases = [jnp.asarray(obs_schedule['phases']),
                  jnp.zeros((n_sne, n_filters))]
        if has_template:
            phases.append((template_per_band[None, :] - peak_mjd[:, None])
                          / (1.0 + sed_params['redshift'][:, None]))
        phases = jnp.concatenate(phases, axis=1)
        inst = simulator.instrument
        idx = (phases - inst.phasecor_phase_min) / inst.phasecor_phase_step
        idx = jnp.clip(idx, 0.0, float(inst.phasecor_n_grid - 1))
        idx_lo = jnp.floor(idx).astype(jnp.int32)
        idx_hi = jnp.minimum(idx_lo + 1, inst.phasecor_n_grid - 1)
        frac = idx - idx_lo.astype(idx.dtype)
        sn_ax = jnp.arange(n_sne)[:, None]
        pc_smear = (1.0 - frac) * pc_grid[sn_ax, idx_lo] + frac * pc_grid[sn_ax, idx_hi]
        mag_ok = (pair_magnitudes > -2) & (pair_magnitudes < 50)
        pair_magnitudes = jnp.where(mag_ok, pair_magnitudes + pc_smear, pair_magnitudes)
        pair_fluxcal = jnp.where(
            mag_ok, pair_fluxcal * jnp.power(10.0, -0.4 * pc_smear), pair_fluxcal)

    # Split only after applying the same corrections at every evaluated epoch.
    fluxcals_all = pair_fluxcal[:, :max_obs]
    magnitudes_all = pair_magnitudes[:, :max_obs]
    peakmags = pair_magnitudes[:, max_obs:max_obs + n_filters]
    # the SN's own flux in each epoch's template image, zero without one
    template_fluxcal_per_obs = jnp.zeros_like(fluxcals_all)
    if has_template:
        template_fluxcal = pair_fluxcal[:, max_obs + n_filters:]
        template_mags = pair_magnitudes[:, max_obs + n_filters:]
        template_fluxcal = jnp.where(
            (template_fluxcal > LOG_SAFE_FLOOR) & (template_mags != MAG_ZEROFLUX),
            template_fluxcal, 0.0)
        template_fluxcal_per_obs = template_fluxcal[
            jnp.arange(fluxcals_all.shape[0])[:, None],
            jnp.maximum(jnp.asarray(obs_schedule['filter_indices']), 0)]

    noise_key = stream_key(key, 'FLUX_NOISE')
    (fluxcals_noisy, fluxcalerrs, snrs_model, snrs, fluxgal_pe,
     host_galmag_total, host_galmag_galfrac, host_sb_fluxcal_stacked,
     snrs_mon) = add_photometric_noise(
        simulator, catalogue, zp_fluxcal,
        magnitudes_all, fluxcals_all, template_fluxcal_per_obs, obs_schedule, noise_key,
        sn_parameters=sn_parameters)

    # template source flux subtracted after the noise draw, from the true flux too (snlc_sim.c:28530)
    if has_template:
        true_before = fluxcals_all
        fluxcals_all = fluxcals_all - template_fluxcal_per_obs
        fluxcals_noisy = fluxcals_noisy - template_fluxcal_per_obs
        snrs_model = snrs_model * jnp.where(
            true_before > LOG_SAFE_FLOOR, fluxcals_all / true_before, 1.0)
        # snlc_sim.c:28580
        snrs = jnp.where(fluxcalerrs > 0, fluxcals_noisy / fluxcalerrs, snrs)

    # DOFILT: NaN outside the model's zHEL range per filter (OBSFLAG_GEN)
    filter_indices = obs_schedule['filter_indices']
    filter_indices_safe = jnp.maximum(filter_indices, 0)
    z_min_table = jnp.array(simulator.filter_z_min, dtype=jnp.float32)
    z_max_table = jnp.array(simulator.filter_z_max, dtype=jnp.float32)
    tidx = jnp.asarray(sn_parameters['template_idx'], dtype=jnp.int32)
    obs_z_min = z_min_table[tidx[:, None], filter_indices_safe]
    obs_z_max = z_max_table[tidx[:, None], filter_indices_safe]
    redshifts = sn_parameters['sim_redshift_helio'][:, None]
    filter_z_valid = (redshifts >= obs_z_min) & (redshifts < obs_z_max) & (obs_z_max > 0)
    filter_z_valid = filter_z_valid & (filter_indices >= 0)

    # SIM_PEAKMJD; the reported peak below may be estimated or smeared
    sn_parameters = {**sn_parameters, 'peakmjd_true': sn_parameters['peakmjd']}
    if simulator.rates.opt_setpkmjd & (8 | 16 | 32):  # flux-based estimators
        snrs_for_fmax = jnp.where(filter_z_valid, snrs, 0.0)

        observed_peak_mjds = estimate_peak_mjd(
            simulator.rates, obs_schedule,
            snrs_for_fmax, fluxcals_noisy, survey_mjd_span)

        sn_parameters = {**sn_parameters, 'peakmjd': observed_peak_mjds}

    elif (simulator.rates.gensigma_peakmjd > 1.0E-9
          and 'peakmjd_smear' in sn_parameters):
        # gen_peakmjd_smear (snlc_sim.c:17120)
        sn_parameters = {**sn_parameters, 'peakmjd': (
            sn_parameters['peakmjd'] + sn_parameters['peakmjd_smear'])}

    magnitudes_out = jnp.where(filter_z_valid, magnitudes_all, jnp.nan)
    fluxcals_out = jnp.where(filter_z_valid, fluxcals_all, jnp.nan)
    fluxcals_noisy_out = jnp.where(filter_z_valid, fluxcals_noisy, jnp.nan)
    fluxcalerrs_out = jnp.where(filter_z_valid, fluxcalerrs, jnp.nan)
    snrs_out = jnp.where(filter_z_valid, snrs, jnp.nan)
    snrs_model_out = jnp.where(filter_z_valid, snrs_model, jnp.nan)

    return {
        'parameters': sn_parameters,
        'observation_schedule': obs_schedule,
        'magnitudes': magnitudes_out,
        'fluxcals': fluxcals_out,
        'fluxcals_noisy': fluxcals_noisy_out,
        'fluxcalerrs': fluxcalerrs_out,
        'snrs': snrs_out,
        'snrs_model': snrs_model_out,
        'snrs_mon': snrs_mon,
        'fluxgal_pe': fluxgal_pe,
        'host_galmag_total': host_galmag_total,
        'host_galmag_galfrac': host_galmag_galfrac,
        'host_sb_fluxcal_stacked': host_sb_fluxcal_stacked,
        'filter_z_valid': filter_z_valid,
        'peakmags': peakmags,
        **epoch_layout(n_sne, max_obs, obs_schedule['n_obs_per_sn']),
    }


def check_finite_flux(simulator, results, obs_schedule, global_start_idx,
                      n_events) -> None:
    """Abort on a non-finite model flux at an observed, valid epoch.

    A model must return a finite flux for every event it is given; one
    that does not would otherwise fail every cut and vanish from the
    output in silence. Runs on the finished batch, so it names the
    model, the event and its drawn parameters; only the first ``n_events``
    events are read. An epoch whose filter is invalid at the event's
    redshift is NaN by design and is not read.
    """
    observed = (obs_schedule['filter_indices'] >= 0) & results['filter_z_valid']
    bad = jnp.any(observed & ~jnp.isfinite(results['fluxcals']), axis=1)
    bad = bad[:n_events]
    if not bool(jnp.any(bad)):
        return
    i = int(jnp.argmax(bad))
    sn_parameters = results['parameters']
    names = [name for name, _ in simulator.model.configured_population]
    drawn = ', '.join(f"{n}={float(sn_parameters[n][i]):g}"
                      for n in names if n in sn_parameters)
    raise SNANAError(
        f"{type(simulator.model).__name__} produced a non-finite flux "
        f"for event {global_start_idx + i} ({drawn}); check the "
        f"GENPEAK_/GENSIGMA_/GENRANGE_ configuration.",
        function_name="check_finite_flux")


def crazy_flux_bound(fluxcal_err, redshift, zp_fluxcal, exposure_ratio=1.0,
                     model_factor=1.0, mag_smear=0.0):
    """SNANA's crazy-flux bound (check_crazyFlux, snlc_sim.c:28648-28689), in
    FLUXCAL rather than ADU.

    SNANA bounds the ADU flux by `1e5 * 10^(0.4(ZPADU-31)) * xt / z^2`, plus 10
    sigma of noise. FLUXCAL is that ADU flux over `10^(0.4(ZPADU-ZP_FLUXCAL))`,
    so ZPADU cancels and only the FLUXCAL zeropoint remains. A brightening
    intrinsic smear (`mag_smear` < 0) widens the bound by its flux factor
    (:28669-28672).

    `exposure_ratio` is EXPOSURE_TIME_FILTER for the band, floored at 1 as SNANA
    floors `xt`. SNANA's z<1e-9 branch is for LCLIB galactic sources, which have
    no jax analogue; a clamp keeps the bound finite instead.
    """
    xt = jnp.maximum(exposure_ratio, 1.0)
    z = jnp.maximum(redshift, 1.0e-9)
    base = 1.0e5 * 10.0 ** (0.4 * (zp_fluxcal - 31.0)) * xt / (z * z)
    bound = base + 10.0 * fluxcal_err
    bound = bound * 10.0 ** (-0.4 * jnp.minimum(mag_smear, 0.0))
    return model_factor * bound


def check_crazy_flux(simulator, fluxcals, fluxcalerrs,
                     obs_schedule, sn_parameters, n_events) -> None:
    """Abort on absurd fluxes, as SNANA does (check_crazyFlux,
    snlc_sim.c:28628), over every event, rejected ones included. Runs on
    the finished batch; only the first ``n_events`` events are read.
    """
    cfg = simulator.config
    names = simulator.filter_names
    ratios = cfg.survey.exposure_time
    xt_per_filter = jnp.array(
        [ratios.get(filter_band(n), 1.0) for n in names], dtype=jnp.float32)

    filter_indices = obs_schedule['filter_indices']
    observed = filter_indices >= 0
    xt = xt_per_filter[jnp.maximum(filter_indices, 0)]
    z = jnp.asarray(sn_parameters['sim_redshift_helio'])[:, None]
    # per-epoch intrinsic smear (GENLC.magsmear8): the coherent per-SN
    # term plus GENMAG_SMEAR_FILTER's per-band term
    mag_smear = jnp.asarray(sn_parameters['magsmear'])[:, None]
    smear_filter = simulator.config.model.genmag_smear_filter
    if smear_filter and '_magsmear_filter_ran' in sn_parameters:
        sig_pf = jnp.array(
            [float(smear_filter.get(filter_band(b), 0.0))
             for b in simulator.photometry.filter_names])
        ran = jnp.asarray(sn_parameters['_magsmear_filter_ran'])
        fclip = jnp.clip(filter_indices, 0, ran.shape[1] - 1)
        mag_smear = mag_smear + jnp.where(
            observed,
            sig_pf[fclip] * jnp.take_along_axis(ran, fclip, axis=1), 0.0)
    bound = crazy_flux_bound(
        jnp.abs(jnp.nan_to_num(fluxcalerrs)), z, exposure_ratio=xt,
        zp_fluxcal=cfg.instrument.zp_fluxcal,
        model_factor=simulator.model.crazyflux_factor,
        mag_smear=mag_smear)

    over = (observed & (jnp.abs(fluxcals) > bound))[:n_events]
    if not bool(jnp.any(over)):
        return

    over_np = np.asarray(over)
    isn, iobs = [int(v[0]) for v in np.where(over_np)]
    cids = np.asarray(sn_parameters['cid'])
    band = names[int(filter_indices[isn, iobs])]
    raise SNANAError(
        f"Crazy flux for CID {int(cids[isn])} in {band} at MJD "
        f"{float(obs_schedule['mjds'][isn, iobs]):.3f}: FLUXCAL="
        f"{float(fluxcals[isn, iobs]):.4g} exceeds {float(bound[isn, iobs]):.4g} "
        f"at z={float(z[isn, 0]):.4f}. {int(over_np.sum())} epoch(s) "
        f"are over the bound; the model is returning unphysical magnitudes. "
        f"Set SKIP_CHECK_CRAZYFLUX: 1 to generate anyway.",
        function_name="check_crazy_flux")


def estimate_peak_mjd(rates, obs_schedule, snrs, fluxcals, mjd_span):
    """Estimate the observed peak MJD per SN for the OPT_SETPKMJD bit in
    use: 8 naive max-flux, 16 Fmax-clump, 32 clump weighted by log10(SNR)
    (sntools.c:1614). Thin wrapper over the shared core; ``mjd_span`` is the
    survey's static MJD extent, so the call traces cleanly."""
    cfg = rates
    if cfg.opt_setpkmjd & OPTMASK_SETPKMJD_FLUXMAX:
        return naive_fluxmax_peakmjd(
            obs_schedule['mjds'], fluxcals, snrs,
            obs_schedule['n_obs_per_sn'], snrcut_backup=cfg.pkmjd_snrcut_backup)
    return fmax_clump_peakmjd(
        obs_schedule['mjds'], fluxcals, snrs, obs_schedule['n_obs_per_sn'],
        mjdwin=cfg.pkmjd_mjdwin, snrcut=cfg.pkmjd_snrcut,
        snrcut_backup=cfg.pkmjd_snrcut_backup,
        log_snr_weight=bool(cfg.opt_setpkmjd & OPTMASK_SETPKMJD_FLUXMAX3),
        mjd_span=mjd_span)


_FLUXERR_DISABLED = dict(
    fluxerr_enabled=False, fluxerr_active_vars=(), fluxerr_is_errscale=True,
    fluxerr_grids=None, fluxerr_grid_origins=(), fluxerr_grid_steps=(),
    fluxerr_scale_true=None, fluxerr_scale_data=None, fluxerr_applies_true=None,
    fluxerr_applies_data=None, fluxerr_map_index_per_field=None,
    fluxerr_needs_host_vars=False,
    fluxerr_has_redcov=False, fluxerr_redcov_factors=None, fluxerr_n_factors=0,
    fluxerr_map_counts=None, fluxerr_redcov_counts=None)


def load_fluxerr_model(config, simlib, kcor, hostlib) -> Dict[str, Any]:
    """Load FLUXERRMODEL_FILE (if configured) and compile its JIT-ready state.

    Returns the fluxerr_* LightCurveSimulator fields, disabled when no file is
    configured.
    """
    path = config.instrument.fluxerr_file
    if not path or str(path).strip().upper() in ('NONE', 'NULL', '', 'BLANK'):
        return dict(_FLUXERR_DISABLED)

    logger.info(f"Loading FLUXERRMODEL: {path}")
    data = read_fluxerr_model_file(Path(os.path.expandvars(str(path))))

    # FLUXERRMODEL_REDCOV replaces the file's REDCOV keys and NONE turns
    # REDCOV off; SNANA checks the entries it then reads, with or without
    # maps (sntools_fluxErrModels.c:121-126, :150, :255, :440)
    redcov_override = config.instrument.fluxerr_redcov
    if redcov_override:
        data.redcov = ([] if is_ignorefile(str(redcov_override))
                       else parse_redcov_string(str(redcov_override)))
    check_redcov_entries(data.redcov)

    ignore_str = config.instrument.fluxerr_ignore_dataerr
    if ignore_str:
        ignore_set = {s.strip() for s in str(ignore_str).split(',') if s.strip()}
        seen = set()
        for m in data.maps:
            if m.name in ignore_set:
                m.mask_apply = m.mask_apply & ~MASK_APPLY_DATA
                seen.add(m.name)
        missing = ignore_set - seen
        if missing:
            raise SNANAError(
                f"FLUXERRMAP_IGNORE_DATAERR references unknown MAPNAME(s) "
                f"{sorted(missing)}; valid names are "
                f"{sorted(m.name for m in data.maps)}",
                function_name="load_fluxerr_model",
            )
    # with no maps SNANA applies neither a map nor REDCOV (snlc_sim.c:28035)
    if not data.maps:
        return dict(_FLUXERR_DISABLED)

    fx = _compile_fluxerr_model(data, simlib, kcor)
    fx['fluxerr_enabled'] = True

    host_dependent = {'SBMAG', 'GALMAG'}
    needed_host_vars = sorted(set(fx['fluxerr_active_vars']) & host_dependent)
    if needed_host_vars and hostlib is None:
        raise SNANAError(
            f"FLUXERRMODEL_FILE {path!r} references variable(s) "
            f"{needed_host_vars} which require host galaxy properties "
            f"from a HOSTLIB, but no HOSTLIB_FILE is configured. "
            f"Either configure HOSTLIB_FILE or use a FLUXERRMODEL file "
            f"that doesn't reference SBMAG/GALMAG.",
            function_name="load_fluxerr_model",
        )

    log.line(logger, f"FLUXERRMODEL ready: {len(data.maps)} maps, "
             f"{len(fx['fluxerr_active_vars'])}-D, vars={fx['fluxerr_active_vars']}, "
             f"redcov={fx['fluxerr_has_redcov']}")
    return fx


def _compile_fluxerr_model(data, simlib, kcor) -> Dict[str, Any]:
    """Compile the parsed FluxErrModelData into JIT-ready arrays: the maps
    stacked by :func:`stacked_fluxerr_maps`, and the map and REDCOV factors
    serving each SIMLIB field and band. Returns the compiled fluxerr_*
    fields.
    """
    fx: Dict[str, Any] = {}

    active_vars = tuple(data.maps[0].dim_varnames)
    fx['fluxerr_active_vars'] = active_vars
    fx['fluxerr_is_errscale'] = bool(data.maps[0].is_errscale)
    maps = stacked_fluxerr_maps(data)
    fx['fluxerr_grids'] = jnp.asarray(maps['grids'])
    fx['fluxerr_grid_origins'] = maps['origins']
    fx['fluxerr_grid_steps'] = maps['steps']
    fx['fluxerr_scale_true'] = jnp.asarray(maps['scale_true'], dtype=jnp.float32)
    fx['fluxerr_scale_data'] = jnp.asarray(maps['scale_data'], dtype=jnp.float32)
    fx['fluxerr_applies_true'] = jnp.asarray(maps['applies_true'])
    fx['fluxerr_applies_data'] = jnp.asarray(maps['applies_data'])

    filter_names = kcor.get_filter_names()
    shape = (len(simlib.fields), len(filter_names))
    map_indices = np.full(shape, len(data.maps), dtype=np.int32)
    map_counts = np.zeros(shape, dtype=np.int32)
    redcov_counts = np.zeros(shape, dtype=np.int32)
    redcov_factors = np.zeros((*shape, len(data.redcov)), dtype=np.float32)
    for i, sf in enumerate(simlib.fields):
        for b, fn in enumerate(filter_names):
            band = filter_band(fn)
            map_matches = fluxerr_maps_serving(data.maps, data.field_groups, band, sf.field_name)
            entries = redcov_entries_serving(data.redcov, data.field_groups, band, sf.field_name)
            map_counts[i, b] = len(map_matches)
            redcov_counts[i, b] = len(entries)
            if len(map_matches) == 1:
                map_indices[i, b] = map_matches[0]
            if len(entries) == 1:
                redcov_factors[i, b, entries[0]] = np.sqrt(data.redcov[entries[0]].redcov)
    # Ambiguous pairs use inert values during generation. Validation rejects
    # them only if a generated epoch actually uses the field and band.
    fx['fluxerr_map_index_per_field'] = jnp.asarray(map_indices)
    fx['fluxerr_redcov_factors'] = jnp.asarray(redcov_factors)
    fx['fluxerr_map_counts'] = map_counts
    fx['fluxerr_redcov_counts'] = redcov_counts
    fx['fluxerr_has_redcov'] = bool(np.any(redcov_factors != 0))
    fx['fluxerr_n_factors'] = len(data.redcov)

    fx['fluxerr_needs_host_vars'] = bool(
        'SBMAG' in active_vars or 'GALMAG' in active_vars)
    return fx


def prepare_photometry_data(simulator) -> Dict:
    """The numerical inputs of the photometry kernels, built once per run
    setup: filter/AB arrays from the shared photometry loader, dust laws,
    cosmology scalars, and the model's own contribution."""
    logger.info("Preparing numerical tables...")

    photometry = simulator.photometry

    static_data = {
        'wavelength_grid': simulator.wavelength_grid,
        'filter_transmissions_array': photometry.transmissions_stacked,
        'filter_ref_fluxes': photometry.filter_ref_fluxes,
        'filter_ref_fluxes_snana': photometry.filter_ref_fluxes_snana,
        'filter_ref_mag_terms': photometry.filter_ref_mag_terms,
        'filter_ref_mag_terms_snana': photometry.filter_ref_mag_terms_snana,
        'mw_dust_law': simulator.mw_law,
        'rv_mw': simulator.config.mw_dust.rv,
        'host_dust_law': simulator.host_dust_law,
        'zp_fluxcal': simulator.config.instrument.zp_fluxcal,
        'filter_mean_wavelengths': photometry.filter_mean_wavelengths,
    }

    omega_m, omega_k, omega_l, hubble_distance, w0, wa = cosmo_params(
        simulator.config.cosmology)
    static_data.update({
        'omega_m': omega_m, 'omega_l': omega_l, 'omega_k': omega_k,
        'hubble_distance': hubble_distance, 'w0': w0, 'wa': wa,
    })

    static_data['host_rv'] = simulator.model.host_rv
    static_data['fluxcal_scale_mag_offset'] = (
        simulator.model.fluxcal_scale_mag_offset)

    log.line(logger, "Numerical tables ready")
    return static_data
