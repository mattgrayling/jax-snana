"""Observer-frame magnitude to FLUXCAL and its error (SMEARFLAG_FLUX).

Mirrors SNANA's per-observation noise bookkeeping, including FLUXERRMODEL,
the host-galaxy surface-brightness term and the FUDGE_* knobs.
"""

import logging
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core.photometry.filter_system import filter_band
from jax_snlc_core.photometry.noise import ZEROPOINT_FLUXCAL_SNANA_ORIG
from jax_snlc_core.utils.error_handling import SNANAError
from jax_snlc_sim.photometry.magnitudes import HOSTGAL_MW_RV, SNANA_NULL_MAG
from jax_snlc_sim.hostlib.sersic_sampling import (
    APERTURE_PSFSIG, compute_galfrac, compute_sersic_bn,
)
from jax_snlc_core.photometry.fluxerr_model import (
    interp_1d, interp_2d, interp_3d,
)

logger = logging.getLogger(__name__)

# PSF sigma of the 1" aperture SIM_GALFRAC uses (snlc_sim.c:26147)
GALFRAC_PSFSIG = 1.0 / 2.355
LOG_SAFE_FLOOR = 1e-30


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


def _base_noise_components_pe(zptavgs, gains, fluxcals, neas,
                              skysigs, readnoises, zp_fluxcal):
    """Per-obs source flux, sky variance, and CCD-read variance in
    photoelectrons (gen_fluxNoise_calc, snlc_sim.c:27767-27799). Returns
    ``(flux_pe, sqerr_sky, sqerr_ccd, npe_over_fluxcal)``. The SIMLIB NEA is
    used directly, as SNANA does; it is validated positive at construction."""
    npe_over_fluxcal = jnp.power(10.0, 0.4 * (zptavgs - zp_fluxcal)) * gains
    flux_pe = fluxcals * npe_over_fluxcal
    sky_e = skysigs * gains
    sqerr_sky = neas * sky_e * sky_e
    sqerr_ccd = neas * jnp.square(readnoises)
    return flux_pe, sqerr_sky, sqerr_ccd, npe_over_fluxcal


def _template_noise_var_pe(neas, gains, zptavgs, template_skysigs,
                           template_readnoises, template_zpts):
    """Template sky+read variance in photoelectrons, scaled to the search
    zero-point (snlc_sim.c:27803-27827). SNANA's activation rule is
    ``template_skysig > 0`` with TEMPLATE_ZPT validated at construction
    (>= 10, snlc_sim.c:27808-27813)."""
    has_template = template_skysigs > 0.0
    template_sky_e = template_skysigs * gains
    template_var = (
        neas * template_sky_e * template_sky_e
        + neas * jnp.square(template_readnoises)
    )
    zpt_scale = jnp.power(10.0, 0.8 * (zptavgs - template_zpts))
    return jnp.where(has_template, template_var * zpt_scale, 0.0)


def _fluxerr_model_sqscales(err_model_val, err_scale_true, err_scale_data,
                            err_mask_true, err_mask_data, npe_over_fluxcal,
                            sqsig_calc_data, is_errscale):
    """FLUXERRMODEL variance scale factors (SQSCALE) for the TRUE and DATA
    sides, from the interpolated map value and the pre-fudge calculated error.

    SNANA forms SCALE = FLUXCALERR_TRUE/FLUXCALERR_in where FLUXCALERR_in is
    captured from SQSIG_CALC_DATA before any error fudge (snlc_sim.c:27975,
    :28051-28058). ERRSCALE maps multiply the error and ERRADD maps add
    errModelVal in quadrature (sntools_fluxErrModels.c:972-1002); a side whose
    MASK_APPLY bit is off keeps the input error (scale 1). A TRUE-side scale
    of exactly 1.0 is bumped to 1.001 (snlc_sim.c:28059-28062).
    """
    if is_errscale:
        scale_true = err_model_val * err_scale_true
        scale_data = err_model_val * err_scale_data
    else:
        # FLUXCALERR_TRUE² = FLUXCALERR_in² + errModelVal² (FLUXCAL units),
        # with FLUXCALERR_in² = SQSIG_CALC_DATA/npe².
        ratio = jnp.sqrt(
            1.0 + err_model_val * err_model_val
            * (npe_over_fluxcal * npe_over_fluxcal) / sqsig_calc_data)
        scale_true = ratio * err_scale_true
        scale_data = ratio * err_scale_data
    scale_true = err_mask_true * scale_true + (1.0 - err_mask_true)
    scale_data = err_mask_data * scale_data + (1.0 - err_mask_data)
    scale_true = jnp.where(scale_true == 1.0, 1.001, scale_true)
    return scale_true * scale_true, scale_data * scale_data


class NoiseVariances(NamedTuple):
    """Each epoch's calculated (pre-fudge) photo-electron variances and the
    S/N they give (gen_fluxNoise_calc, snlc_sim.c:27853-27895)."""
    flux_pe: jnp.ndarray
    npe_over_fluxcal: jnp.ndarray
    sqsig_S: jnp.ndarray          # source, host, sky and readout
    sqsig_SZ: jnp.ndarray         # sqsig_S plus the drawn zero-point term
    sqsig_T: jnp.ndarray          # template
    sqsig_calc_data: jnp.ndarray  # SQSIG_CALC_DATA
    snr_calc_szt: jnp.ndarray
    snr_calc_st: jnp.ndarray


def noise_variances(fluxcal, zptavg, zpterr, gain, nea, skysig, readnoise,
                    template_skysig, template_readnoise, template_zpt,
                    fluxgal_pe, *, smearflag_zeropt, zp_fluxcal) -> NoiseVariances:
    """Each epoch's pre-fudge variances in photo-electrons and the S/N they
    give, elementwise over any shape of epochs (gen_fluxNoise_calc).

    The source term is the true FLUXCAL in photo-electrons; the sky and
    readout terms scale with the NEA; the template term applies where a
    template sky noise is given. SMEARFLAG_ZEROPT decides where the
    zero-point jitter goes: bit 1 adds it to the drawn error (``sqsig_SZ``),
    bit 2 to the reported one (``sqsig_calc_data``). ``fluxgal_pe`` is the
    host's photo-electrons, zero unless SMEARFLAG_HOSTGAL adds them.
    """
    flux_pe, sqerr_sky, sqerr_ccd, npe_over_fluxcal = \
        _base_noise_components_pe(zptavg, gain, fluxcal, nea,
                                  skysig, readnoise, zp_fluxcal=zp_fluxcal)
    template_var = _template_noise_var_pe(
        nea, gain, zptavg, template_skysig, template_readnoise, template_zpt)

    # Zero-point jitter variance; SMEARFLAG_ZEROPT bit 1 puts it in the TRUE
    # (drawn) error, bit 2 in the DATA (reported) error (snlc_sim.c:27871-27881).
    relerr = jnp.power(10.0, 0.4 * zpterr) - 1.0
    sqerr_zp = jnp.square(flux_pe * relerr)
    sqerr_zp_true = jnp.where((smearflag_zeropt & 1) > 0, sqerr_zp, 0.0)
    sqerr_zp_data = jnp.where((smearflag_zeropt & 2) > 0, sqerr_zp, 0.0)

    # fluxgal_pe is the host-galaxy shot term (SMEARFLAG_HOSTGAL bit 1)
    sqsig_S = flux_pe + fluxgal_pe + sqerr_sky + sqerr_ccd
    sqsig_calc_data = sqsig_S + sqerr_zp_data + template_var
    return NoiseVariances(
        flux_pe=flux_pe, npe_over_fluxcal=npe_over_fluxcal,
        sqsig_S=sqsig_S, sqsig_SZ=sqsig_S + sqerr_zp_true,
        sqsig_T=template_var, sqsig_calc_data=sqsig_calc_data,
        snr_calc_szt=flux_pe / jnp.sqrt(sqsig_calc_data),
        snr_calc_st=flux_pe / jnp.sqrt(sqsig_S + template_var))


class Epoch(NamedTuple):
    """One observation's inputs to the noise model: its true brightness, the
    three random draws, its zero point and gain, the host's photo-electrons,
    the FLUXERRMODEL scales and its calculated variances."""
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
    variances: NoiseVariances


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
     fluxgal_pe, sqscale_true, sqscale_data, filter_idx, v) = epoch
    valid = (
        jnp.isfinite(magnitude)
        & jnp.isfinite(fluxcal)
        & jnp.isfinite(zptavg)
        & (gain > 0)
    )
    flux_pe, npe_over_fluxcal = v.flux_pe, v.npe_over_fluxcal
    sqsig_calc_data = v.sqsig_calc_data
    snr_calc_szt, snr_calc_st = v.snr_calc_szt, v.snr_calc_st

    # error-fudge chain (gen_fluxNoise_fudge_diag, snlc_sim.c:27968)
    sqsig_S_true = v.sqsig_S
    sqsig_SZ_true = v.sqsig_SZ
    sqsig_T_true = v.sqsig_T
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

    For an SN with field group fg, every obs i has a factor-loading vector
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
                          magnitudes_all, fluxcals_all, obs_schedule,
                          noise_key, sn_parameters):
    """
    Add photometric noise using JAX for deterministic reproducibility, the
    host's light read from ``catalogue`` (the HostCatalogue, None without a
    HOSTLIB).

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

    variances = noise_variances(
        fluxcals_all, obs_arrays['zptavgs'], obs_arrays['zpterrs'],
        obs_arrays['gains'], obs_arrays['neas'], obs_arrays['skysigs'],
        obs_arrays['readnoises'], obs_arrays['template_skysigs'],
        obs_arrays['template_readnoises'], obs_arrays['template_zpts'],
        fluxgal_pe_all,
        smearflag_zeropt=simulator.instrument.smearflag_zeropt,
        zp_fluxcal=zp_fluxcal)

    fluxerrmodel_enabled = bool(simulator.fluxerr_enabled)
    if fluxerrmodel_enabled:
        (err_model_val, err_scale_true, err_scale_data,
         err_mask_true, err_mask_data) = _evaluate_fluxerr_model(
            simulator, zp_fluxcal,
            sn_parameters, obs_arrays, variances.snr_calc_szt,
            sb_fluxcal_stacked=sb_fluxcal_stacked,
            galmag_aperture_table=galmag_aperture_table,
        )

        sqscale_true, sqscale_data = _fluxerr_model_sqscales(
            err_model_val, err_scale_true, err_scale_data,
            err_mask_true, err_mask_data, variances.npe_over_fluxcal,
            variances.sqsig_calc_data,
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
            sn_fg = _fluxerr_sn_field_group_idx(simulator, sn_parameters)
            fi_safe = jnp.maximum(obs_arrays['filter_indices'], 0)
            F_per_obs = simulator.fluxerr_redcov_factors[sn_fg[:, None], fi_safe]
            # below-SNR epochs draw iid (SNR_CALC_ST, snlc_sim.c:27894, :28307-28308)
            in_redcov_group = (
                (obs_arrays['filter_indices'] >= 0)
                & (variances.snr_calc_st >= snrmin)
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
        filter_idx=obs_arrays['filter_indices'], variances=variances))

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
            mag_columns.append(jnp.full(n_sne, SNANA_NULL_MAG))
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


def _fluxerr_sn_field_group_idx(resources, sn_parameters):
    """Per-SN field-group index, gathered from the SIMLIB field assignment."""
    field_idx = jnp.asarray(sn_parameters['field_idx'], dtype=jnp.int32)
    return resources.fluxerr_field_group_idx_per_field[field_idx]


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
        return jnp.full((n_sne, max_obs), SNANA_NULL_MAG, dtype=jnp.float32)
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
    `resources.fluxerr_grid_origins/steps/lengths`).
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
    """Per-(sn, obs) FLUXERRMODEL gather + interpolation.

    Returns ``(errModelVal, scale_true, scale_data, mask_true, mask_data)``,
    each (n_sne, max_obs).
    """
    sn_fg = _fluxerr_sn_field_group_idx(resources, sn_parameters)
    fi = obs_arrays['filter_indices']
    fi_safe = jnp.maximum(fi, 0)

    fx = resources
    grid_per_obs = fx.fluxerr_grid[sn_fg[:, None], fi_safe]
    scale_true_per_obs = fx.fluxerr_scale_true[sn_fg[:, None], fi_safe]
    scale_data_per_obs = fx.fluxerr_scale_data[sn_fg[:, None], fi_safe]
    mask_true_per_obs = fx.fluxerr_mask_true[sn_fg[:, None], fi_safe]
    mask_data_per_obs = fx.fluxerr_mask_data[sn_fg[:, None], fi_safe]

    query = _build_fluxerr_query_per_obs(
        resources, zp_fluxcal, sn_parameters, obs_arrays, snr_calc_szt,
        sb_fluxcal_stacked=sb_fluxcal_stacked,
        galmag_aperture_table=galmag_aperture_table,
    )

    if fx.fluxerr_n_dims == 1:
        err_model_val = interp_1d(
            grid_per_obs, query,
            origin=fx.fluxerr_grid_origins[0],
            step=fx.fluxerr_grid_steps[0],
            n_grid=fx.fluxerr_grid_lengths[0],
        )
    elif fx.fluxerr_n_dims == 2:
        err_model_val = interp_2d(
            grid_per_obs, query,
            origins=fx.fluxerr_grid_origins,
            steps=fx.fluxerr_grid_steps,
            n_grid=fx.fluxerr_grid_lengths,
        )
    else:
        err_model_val = interp_3d(
            grid_per_obs, query,
            origins=fx.fluxerr_grid_origins,
            steps=fx.fluxerr_grid_steps,
            n_grid=fx.fluxerr_grid_lengths,
        )

    # padding lanes get the no-op value
    no_op = 1.0 if fx.fluxerr_is_errscale else 0.0
    valid_obs = fi >= 0
    err_model_val = jnp.where(valid_obs, err_model_val, no_op)

    return (err_model_val, scale_true_per_obs, scale_data_per_obs,
            mask_true_per_obs, mask_data_per_obs)
