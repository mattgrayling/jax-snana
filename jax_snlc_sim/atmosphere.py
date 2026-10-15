"""Atmosphere / differential-chromatic-refraction (DCR) post-pass.

Port of SNANA ``sntools_sim_atmosphere.c`` (ATMOSPHERE_OPTMASK, FORMAT_MASK bit
128): per-observation airmass and DCR coordinate/magnitude shifts. The whole
subsystem is output-only: it never feeds back into simulated flux or noise, so
it runs as a numpy float64 post-pass on already-materialised results, mirroring
``GEN_ATMOSPHERE_DRIVER``. The chromatic SED that drives DCR is the SED-true
``sim_flam`` (SNANA auto-enables SPECTROGRAPH_OPTMASK_SEDMODEL for atmosphere).

Physics in double precision independent of the global JAX x64 flag; only the
per-epoch coordinate smear uses JAX (batch-invariant per-SN keys).
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core.utils.interp import eval_genpoly
from jax_snlc_sim.random_streams import event_keys

RADIAN = math.pi / 180.0
FAC_DCR = 206265.0                      # arcsec per radian
COORD_SHIFT_NULL_DEG = 99.0 / 3600.0
COORD_SHIFT_NULL_ARCSEC = 99.0
SNRMIN = 3.0                            # ATMOS_INFO.SNRMIN
UNIT_DELTA = 3600.0                     # deg -> arcsec
DCR_NULL_ARCSEC = 999.0
# observations per DCR batch
DCR_BATCH = 4096


def airmass_from_mjd(mjd, ra_deg, dec_deg, geo_lat_deg, geo_lon_deg):
    """Airmass, altitude and geometry per obs (gen_airmass, atmosphere.c:371).

    ``mjd`` is (..., ) ; ``ra_deg``/``dec_deg`` broadcast against it. Returns a
    dict of float64 arrays: airmass, altitude_deg, tan_zenith, sin_h, sin_alt,
    cos_alt, cos_dec, sin_dec.
    """
    mjd = np.asarray(mjd, dtype=np.float64)
    ra = np.asarray(ra_deg, dtype=np.float64)
    dec = np.asarray(dec_deg, dtype=np.float64)

    jd2000 = 2451545.0
    jd = mjd + 2400000.5
    t_u = (jd - jd2000) / 36525.0
    gmst_deg = (np.mod(24110.54841 + 8640184.812866 * t_u + 0.093104 * t_u * t_u,
                       86400.0) / 86400.0 + 1.0027379 * np.mod(mjd, 1.0)) * 360.0
    lst_deg = geo_lon_deg + gmst_deg
    h_deg = lst_deg - ra
    cos_h = np.cos(h_deg * RADIAN)
    sin_h = np.sin(h_deg * RADIAN)

    sin_lat = math.sin(geo_lat_deg * RADIAN)
    cos_lat = math.cos(geo_lat_deg * RADIAN)
    sin_dec = np.sin(dec * RADIAN)
    cos_dec = np.cos(dec * RADIAN)

    sin_alt = sin_lat * sin_dec + cos_lat * cos_dec * cos_h
    alt_rad = np.arcsin(sin_alt)
    ang_zenith_rad = 0.5 * math.pi - alt_rad
    return {
        'airmass': 1.0 / np.cos(ang_zenith_rad),
        'altitude_deg': alt_rad / RADIAN,
        'tan_zenith': np.tan(ang_zenith_rad),
        'sin_h': sin_h,
        'sin_alt': sin_alt,
        'cos_alt': np.cos(alt_rad),
        'cos_dec': cos_dec,
        'sin_dec': sin_dec,
    }


def index_refrac_minus1(lam_ang, temp_c, bp_mmhg, pwv_mmhg):
    """n(lambda) - 1 at the site (compute_index_refrac_atmos, atmosphere.c:844).

    ``lam_ang`` is (n_wave,); ``temp_c``/``bp_mmhg``/``pwv_mmhg`` are scalars or
    broadcastable arrays (per-obs when site sigmas are on). Returns n-1.
    """
    lam = np.asarray(lam_ang, dtype=np.float64)
    invlamsq = 1.0e8 / (lam * lam)
    temp = np.asarray(temp_c, dtype=np.float64)
    bp = np.asarray(bp_mmhg, dtype=np.float64)
    pwv = np.asarray(pwv_mmhg, dtype=np.float64)

    denom_t = 1.0 + 0.003661 * temp
    n0m1 = (64.328 + 29498.1 / (146.0 - invlamsq) + 255.4 / (41.0 - invlamsq)) * 1.0e-6
    tmp1 = bp * (1.0 + (1.049 - 0.0157 * temp) * 1.0e-6 * bp)
    tmp2 = 720.883 * denom_t
    n1m1 = n0m1 * (tmp1 / tmp2)
    pwv_term = (0.0624 - 0.000680 * invlamsq) * pwv / denom_t
    return n1m1 - pwv_term * 1.0e-6


def dcr_weighted_arcsec(sim_flam, cal_flux, trans, wave, n_minus_1, tan_zenith):
    """SED-weighted net DCR angle per obs (compute_DCR_angle, atmosphere.c:683).

    ``sim_flam``/``trans`` are (n_obs, n_wave); ``cal_flux`` (n_obs, n_wave);
    ``wave`` (n_wave,); ``n_minus_1`` (n_wave,) or (n_obs, n_wave); ``tan_zenith``
    (n_obs,). Returns (net_dcr_arcsec, lamavg_sed_wgted); net is DCR_NULL where
    the SED or calstar weight sum is non-positive.
    """
    st_sed = sim_flam * trans
    w_sed = st_sed * wave
    sum0_sed = w_sed.sum(-1)
    dcr_lam = FAC_DCR * n_minus_1 * tan_zenith[:, None]
    sum1_sed = (w_sed * dcr_lam).sum(-1)

    st_cal = cal_flux * trans
    w_cal = st_cal * wave
    sum0_cal = w_cal.sum(-1)
    sum1_cal = (w_cal * dcr_lam).sum(-1)

    sum0_lam = st_sed.sum(-1)
    valid = (sum0_sed > 0.0) & (sum0_cal > 0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        dcr_sed = sum1_sed / sum0_sed
        dcr_cal = sum1_cal / sum0_cal
        lamavg = np.where(sum0_lam > 0.0, sum0_sed / sum0_lam, 0.0)
    net = np.where(valid, dcr_sed - dcr_cal, DCR_NULL_ARCSEC)
    return net, lamavg


def _parallactic_shift(dcr_arcsec, geom, sin_geo_lat, cos_geo_lat):
    """Project net DCR onto RA/DEC shifts in degrees (gen_dcr_coordShift:593).

    Returns (dcr_deg, ra_shift_deg, dec_shift_deg); null (COORD_SHIFT_NULL_DEG)
    where the DCR angle is the 999 sentinel.
    """
    dcr_deg = dcr_arcsec / 3600.0
    sin_h = geom['sin_h']
    sin_alt = geom['sin_alt']
    cos_alt = geom['cos_alt']
    sin_dec = geom['sin_dec']
    cos_dec = geom['cos_dec']

    with np.errstate(invalid='ignore', divide='ignore'):
        sin_az = np.where(cos_alt != 0.0, (-sin_h * cos_dec) / cos_alt, 0.0)
        sin_q = np.where(cos_dec != 0.0, (-cos_geo_lat * sin_az) / cos_dec, 0.0)
        cos_product = cos_dec * cos_alt
        cos_q = np.where(cos_product != 0.0,
                         (sin_geo_lat - sin_dec * sin_alt) / cos_product, 0.0)
        ra_shift = dcr_deg * sin_q / cos_dec
    dec_shift = dcr_deg * cos_q

    null = dcr_arcsec >= DCR_NULL_ARCSEC
    dcr_deg = np.where(null, COORD_SHIFT_NULL_DEG, dcr_deg)
    ra_shift = np.where(null, COORD_SHIFT_NULL_DEG, ra_shift)
    dec_shift = np.where(null, COORD_SHIFT_NULL_DEG, dec_shift)
    return dcr_deg, ra_shift, dec_shift


def site_refrac_draws(sn_keys, obs_idx, n_wave, site, sigma_site):
    """Perturbed site conditions per (obs, wavelength bin) for
    ATMOSPHERE_SIGMA_SITE.

    SNANA redraws temperature, pressure and PWV for every wavelength bin of
    the DCR integral (compute_index_refrac_atmos,
    sntools_sim_atmosphere.c:867-877, called per bin at :752-755), so the
    perturbations partly average over the band. Draws are clipped at ±3 sigma
    like getRan_GaussClip(1,-3,3), and are keyed by (event, obs): each obs
    folds its epoch index into the event key and the position inside the
    (n_wave, 3) block keys the bin, so a value is stable per
    (event, obs, bin) under any batching.

    Returns (temp, bp, pwv) float64 arrays of shape (n_obs, n_wave).
    """
    keys = jax.vmap(jax.random.fold_in)(sn_keys, jnp.asarray(obs_idx))
    r = np.asarray(jax.vmap(lambda k: jax.random.truncated_normal(
        k, -3.0, 3.0, (n_wave, 3)))(keys), dtype=np.float64)
    return (site.temperature + r[..., 0] * sigma_site[0],
            site.pressure + r[..., 1] * sigma_site[1],
            site.pwv + r[..., 2] * sigma_site[2])


def _coord_smear_draws(key, global_indices, max_obs):
    """Per-SN, per-epoch Gaussian draws (ran_RA, ran_DEC) for coordinate smear.

    Chunk-invariant: keyed on the global SN index, so a SN draws
    the same values regardless of batching. Returns (n_sne, max_obs, 2) float64.
    """
    sn_keys = event_keys(key, 'DCR_COORD', np.asarray(global_indices))
    draws = jax.vmap(lambda k: jax.random.normal(k, (max_obs, 2)))(sn_keys)
    return np.asarray(draws, dtype=np.float64)


def compute_atmosphere(*, mjd, filter_idx, valid_mask, ra_deg, dec_deg,
                       global_indices, psf_fwhm, snr_true, snr_obs, spec_flam,
                       spec_row, wave, trans_by_filter, cal_flux, site,
                       coordres_poly, magshift_poly, coordres_floor,
                       sigma_site, key):
    """Full atmosphere/DCR post-pass (GEN_ATMOSPHERE_DRIVER, atmosphere.c:273).

    All per-obs arrays are (n_sne, max_obs); ``ra_deg``/``dec_deg``/
    ``global_indices`` are (n_sne,). The chromatic SEDs arrive deduplicated:
    ``spec_flam`` is (n_spec, n_wave) on the shared ``wave`` grid and
    ``spec_row`` is the (n_sne, max_obs) row index of each observation's
    MJD-matched SED (-1 for no match -> DCR null). The DCR integral runs over
    the valid flattened observations in DCR_BATCH-sized batches, so no
    (n_sne, max_obs, n_wave) tensor is ever materialised; float64 is kept
    throughout because the whole SNANA chain is double precision
    (sntools_sim_atmosphere.c) and the net DCR is a small difference of large
    weighted sums. Returns a dict of (n_sne, max_obs) float64 output grids.
    """
    n_sne, max_obs = mjd.shape
    n_wave = wave.shape[0]
    valid = np.asarray(valid_mask, dtype=bool)

    geom = airmass_from_mjd(mjd, ra_deg[:, None], dec_deg[:, None],
                            site.geo_lat, site.geo_lon)

    # site n-1; per-(obs, bin) under ATMOSPHERE_SIGMA_SITE
    apply_sigma = sigma_site is not None and any(s > 0.0 for s in sigma_site)
    if apply_sigma:
        sn_keys = event_keys(key, 'DCR_SITE', np.asarray(global_indices))
    else:
        n_site_1d = index_refrac_minus1(wave, site.temperature, site.pressure,
                                        site.pwv)

    fi_flat = np.asarray(filter_idx).ravel()
    spec_row_flat = np.asarray(spec_row).ravel()
    idx = np.where(valid.ravel() & (spec_row_flat >= 0))[0]
    tanz_flat = geom['tan_zenith'].ravel()
    trans_all = np.asarray(trans_by_filter)
    cal_flux = np.asarray(cal_flux, dtype=np.float64)
    sn_of_flat = idx // max_obs
    obs_of_flat = idx % max_obs

    net_dcr = np.full(n_sne * max_obs, DCR_NULL_ARCSEC)
    lamavg = np.zeros(n_sne * max_obs)
    for start in range(0, len(idx), DCR_BATCH):
        b = idx[start:start + DCR_BATCH]
        flam_b = np.asarray(spec_flam)[spec_row_flat[b]]
        trans_b = trans_all[fi_flat[b]]
        cal_b = np.broadcast_to(cal_flux, (len(b), n_wave))
        if apply_sigma:
            temp_b, bp_b, pwv_b = site_refrac_draws(
                sn_keys[sn_of_flat[start:start + DCR_BATCH]],
                obs_of_flat[start:start + DCR_BATCH], n_wave, site, sigma_site)
            n_b = index_refrac_minus1(wave[None, :], temp_b, bp_b, pwv_b)
        else:
            n_b = np.broadcast_to(n_site_1d, (len(b), n_wave))
        net_b, lam_b = dcr_weighted_arcsec(
            flam_b, cal_b, trans_b, wave, n_b, tanz_flat[b])
        net_dcr[b] = net_b
        lamavg[b] = lam_b
    net_dcr = net_dcr.reshape(n_sne, max_obs)
    lamavg = lamavg.reshape(n_sne, max_obs)
    net_dcr = np.where(valid, net_dcr, DCR_NULL_ARCSEC)

    dcr_deg, ra_shift, dec_shift = _parallactic_shift(
        net_dcr, geom, math.sin(site.geo_lat * RADIAN),
        math.cos(site.geo_lat * RADIAN))

    # coordinate smear and band-averaged coords (genSmear_coords:505)
    cos_dec = geom['cos_dec']
    snr_t = np.maximum(np.asarray(snr_true, dtype=np.float64), 0.01)
    snr_o = np.asarray(snr_obs, dtype=np.float64)
    psf = np.asarray(psf_fwhm, dtype=np.float64)
    statres_true = eval_genpoly(coordres_poly, np.where(snr_t > 0, psf / snr_t, 0.0))
    statres_obs = eval_genpoly(coordres_poly, np.where(snr_o > 0, psf / snr_o, 0.0))
    angres_true = np.sqrt(statres_true ** 2 + coordres_floor ** 2)
    angres_obs = np.sqrt(statres_obs ** 2 + coordres_floor ** 2)
    valid_shift = ra_shift < COORD_SHIFT_NULL_DEG
    angres_true = np.where(valid_shift, angres_true, 0.0)

    unit_convert = 1.0 / (3600.0 * math.sqrt(2.0))
    angres_true_deg = angres_true * unit_convert
    angres_obs_deg = angres_obs * unit_convert

    draws = _coord_smear_draws(key, global_indices, max_obs)
    ran_ra, ran_dec = draws[..., 0], draws[..., 1]
    ra_true = ra_deg[:, None] + ra_shift
    dec_true = dec_deg[:, None] + dec_shift
    ra_obs = ra_true + (angres_true_deg * ran_ra) / cos_dec
    dec_obs = dec_true + (angres_true_deg * ran_dec)

    use_obs = (snr_o > SNRMIN) & valid
    with np.errstate(invalid='ignore', divide='ignore'):
        wgt = np.where(angres_true_deg > 0.0,
                       1.0e-6 / (angres_obs_deg * angres_obs_deg), 1.0e-20)
    wgt = np.where(use_obs, wgt, 0.0)

    n_filt = np.asarray(trans_by_filter).shape[0]
    sn_ax = np.repeat(np.arange(n_sne)[:, None], max_obs, axis=1)
    fi = fi_flat.reshape(n_sne, max_obs)
    fi_safe = np.where(valid & (fi >= 0), fi, 0)

    def _band_avg(x):
        """Weighted per-(SN, band) average; returns (matrix, per-obs gather)."""
        sumw = np.zeros((n_sne, n_filt))
        sumwx = np.zeros((n_sne, n_filt))
        np.add.at(sumw, (sn_ax, fi_safe), wgt)
        np.add.at(sumwx, (sn_ax, fi_safe), wgt * x)
        with np.errstate(invalid='ignore', divide='ignore'):
            avg = np.where(sumw > 0.0, sumwx / sumw, 0.0)
        return avg, avg[sn_ax, fi_safe]

    ra_avg = _band_avg(ra_obs)[1]
    dec_avg = _band_avg(dec_obs)[1]
    ra_sim_avg_band, ra_sim_avg = _band_avg(ra_true)
    dec_sim_avg_band, dec_sim_avg = _band_avg(dec_true)

    # gen_dcr_magShift:927
    dcr_asec = dcr_deg * 3600.0
    with np.errstate(invalid='ignore', divide='ignore'):
        frac_psf = np.where(psf > 0, np.abs(dcr_asec) / psf, 0.0)
    mag_shift = np.where(valid_shift, eval_genpoly(magshift_poly, frac_psf), 0.0)

    # coords_to_SNDATA, snlc_sim.c:25815
    d_ra = UNIT_DELTA * (ra_obs - ra_avg)
    d_dec = UNIT_DELTA * (dec_obs - dec_avg)
    sim_dcr = UNIT_DELTA * dcr_deg
    sim_d_ra = UNIT_DELTA * (ra_true - ra_sim_avg)
    sim_d_dec = UNIT_DELTA * (dec_true - dec_sim_avg)

    null = d_ra > (COORD_SHIFT_NULL_ARCSEC - 5.0)
    d_ra = np.where(null, COORD_SHIFT_NULL_ARCSEC, d_ra)
    d_dec = np.where(null, COORD_SHIFT_NULL_ARCSEC, d_dec)
    sim_dcr = np.where(null, COORD_SHIFT_NULL_ARCSEC, sim_dcr)
    sim_d_ra = np.where(null, COORD_SHIFT_NULL_ARCSEC, sim_d_ra)
    sim_d_dec = np.where(null, COORD_SHIFT_NULL_ARCSEC, sim_d_dec)

    return {
        'airmass': geom['airmass'],
        'd_ra': d_ra,
        'd_dec': d_dec,
        'sim_dcr_dra': sim_d_ra,
        'sim_dcr_ddec': sim_d_dec,
        'sim_dcr_dmag': mag_shift,
        # extra per-obs diagnostics for the DUMP_DCR table
        'sim_dcr': sim_dcr,
        'coordres': np.where(use_obs, angres_obs, 0.0),
        'lamavg_sed_wgted': lamavg,
        'psf_fwhm': psf,
        'snr_true': np.asarray(snr_true, dtype=np.float64),
        'snr_obs': snr_o,
        # per-(SN, band) SIM band-averaged true coords (SNDATA.RA_AVG_BAND)
        'ra_avg_band': ra_sim_avg_band,
        'dec_avg_band': dec_sim_avg_band,
    }

