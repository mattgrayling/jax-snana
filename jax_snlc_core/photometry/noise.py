"""SNANA photometric-noise bookkeeping shared by the simulator and the fit
loader: the noise-equivalent PSF aperture and the flux-error recomputation
that backs CUTWIN_ERRTEST (snana.F90 FLUXERRCALC, :20264-20323).
"""

import jax.numpy as jnp
import numpy as np

# Default zero point of simulated and Python-input FLUXCAL photometry.
ZEROPOINT_FLUXCAL_SNANA = 31.4

# pre-nJy FLUXCAL zero point (sndata.h:46)
ZEROPOINT_FLUXCAL_SNANA_ORIG = 27.5


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


def noise_equivalent_aperture(psf1, psf2, psf_ratio):
    """SNANA's NoiseEquivAperture (sntools.c:5345-5419): effective sky aperture
    in pixel^2 from the (possibly double-) Gaussian PSF parameters. The only
    branch is SNANA's own single-Gaussian check; lanes taking it use dummy
    denominators so the unselected double-Gaussian expression stays finite."""
    sqsig1 = psf1 * psf1
    sqsig2 = psf2 * psf2
    sqsum = sqsig1 + sqsig2
    single_gauss = (psf_ratio < 1.0e-5) | (psf2 < 0.0001)
    nea_single = 4.0 * np.pi * sqsig1

    safe_sqsig1 = np.where(single_gauss, 1.0, sqsig1)
    safe_psf1 = np.where(single_gauss, 1.0, psf1)
    safe_psf2 = np.where(single_gauss, 1.0, psf2)
    tmp = psf_ratio * sqsig2 / safe_sqsig1
    a1 = 1.0 / (1.0 + tmp)
    a2 = 1.0 - a1
    tmp2 = a1 * safe_psf2 / safe_psf1 + a2 * safe_psf1 / safe_psf2
    nea_double = 4.0 * np.pi * sqsum / (1.0 + tmp2 * tmp2)

    return np.where(single_gauss, nea_single, nea_double)


def fluxcal_errcalc(fluxcal, fluxcal_err, zeropt, zpterr, skysig, skysig_t,
                    psf1, psf2, psf_ratio, gain, zp_fluxcal, sb_fluxcal=0.0,
                    pixsize=0.0):
    """SNANA's ERRCALC: the flux error recomputed from the noise model
    (sky + signal-Poisson + zeropoint jitter + host SB Poisson), in FLUXCAL
    units. Backs ERRTEST = ERRCALC/ERRTRUE (snana.F90:20578). Read noise is
    omitted: SNANA never populates SNLC_RDNOISE at fit time (only
    initialised to 0, snana.F90:15190), so RDNOISE_ADU=0 in its
    FLUXERRCALC. sb_fluxcal is the host surface brightness (FLUXCAL/arcsec^2,
    HOSTGAL_SB_FLUXCAL of the epoch's band); the host term (snana.F90:20548)
    needs it and pixsize."""
    zpscale = np.power(10.0, 0.4 * (zeropt - zp_fluxcal))  # FLUXADU/FLUXCAL
    fluxadu = zpscale * fluxcal
    fluxadu_err = zpscale * fluxcal_err

    tmprat = np.where(fluxadu_err != 0.0, fluxadu / np.where(fluxadu_err != 0.0,
                      fluxadu_err, 1.0), 0.0)
    sqerr_signal = np.where((tmprat > 0.5) & (gain > 0.0),
                            fluxadu / np.where(gain > 0.0, gain, 1.0), 0.0)

    area = noise_equivalent_aperture(psf1, psf2, psf_ratio)
    sqerr_sky = area * (skysig ** 2 + np.maximum(0.0, skysig_t) ** 2)

    # host SB Poisson term (snana.F90:20548)
    sbfluxadu = np.asarray(sb_fluxcal) * np.asarray(pixsize) ** 2 * zpscale
    sqerr_host = np.where((np.asarray(sb_fluxcal) > 0.001) & (gain > 0.0),
                          area * sbfluxadu / np.where(gain > 0.0, gain, 1.0),
                          0.0)

    tmperr = np.power(10.0, 0.4 * zpterr) - 1.0
    sqerr_zp = (fluxadu * tmperr) ** 2

    return np.sqrt(sqerr_sky + sqerr_host + sqerr_signal + sqerr_zp) / zpscale


def apply_fluxcal_fudges(fluxcal, fluxcal_err, offset, err_scale, err_add,
                         err_pix, mag_err, nea):
    """SNANA EXEC_FUDGE_FLUXCAL (snana.F90:18700-18948): user systematic
    fudges on calibrated fluxes. Returns (flux, err1, add_var): the offset flux,
    the scaled-original error `err1 = err*err_scale`, and the additive fudge
    variance: additive (`err_add`, signed so a negative value subtracts in
    quadrature), per-pixel (`err_pix`, weighted by the noise-equivalent aperture
    `nea`, and contributing nothing unless it is positive), and fractional-mag
    (``mag_err * flux``). The caller sums err1^2 + add_var
    (plus any FLUXERRMODEL/HOSTNOISE contribution, also from err1) and sqrts."""
    flux = fluxcal + offset
    err1 = fluxcal_err * err_scale
    # negative ERRPIX adds nothing (snana.F90:18865)
    area = np.where(err_pix > 0.0, nea, 0.0)
    add_var = (np.sign(err_add) * (err_add * err_add)
               + (err_pix * err_pix) * area + (mag_err * flux) ** 2)
    return flux, err1, add_var
