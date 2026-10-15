"""1/Vmax volume per SN for the MAGLIM_VMAX / OPT_VMAX keys.

Post-fit and vectorised over SNe with plain numpy: nothing here runs inside the
optimiser, so SNANA's own quadrature is reproduced directly rather than
approximated on a jittable grid.
"""

import logging

import numpy as np

from jax_snlc_core.photometry.filter_system import filter_band
from jax_snlc_core.physics.constants import LIGHT_KM
from jax_snlc_core.physics.cosmology import cpl_e2
from jax_snlc_core.utils.error_handling import SNANAError

logger = logging.getLogger(__name__)

MAGDIF_CONVERGE = 0.001     # snlc_fit.F90:937
MAX_ITER = 20               # :15311
MXFILT_VMAX = 5             # :940

# per-band init when the SN has no usable epoch (:15176)
MAG_NULL, TREST_NULL, Z_NULL, VOL_NULL = 99.0, 99.0, -99.0, -99.0


def _hzfun(z, h0, om, ol, w0, wa):
    """H(z) for wCDM (Hzfun_wCDM, sntools_cosmology.c:479)."""
    return h0 * np.sqrt(np.asarray(cpl_e2(z, om, ol, w0, wa)))


def _curvature(total, kappa):
    """Open/closed correction applied to the comoving integral (:387)."""
    sk = np.sqrt(abs(kappa))
    if kappa < -1e-5:
        return np.sin(sk * total) / sk
    if kappa > 1e-5:
        return np.sinh(sk * total) / sk
    return total


def dvdz_integral(zmax, cospar):
    """Comoving volume out to each ``zmax`` (dVdz_integral, :285).

    Accepts an array. r(z) on the quadrature grid is a running sum of the same
    integrand, so one prefix sum replaces the per-point re-integration.
    """
    zmax = np.atleast_1d(np.asarray(zmax, dtype=float))
    h0, om, ol, w0, wa = cospar
    kappa = 1.0 - om - ol
    nz = np.maximum((zmax * 1000.0).astype(int), 10)
    width = int(nz.max())
    dz = zmax / nz
    steps = np.arange(width)
    z = dz[:, None] * (steps + 0.5)
    live = steps < nz[:, None]
    inv_h = np.where(live, 1.0 / _hzfun(z, h0, om, ol, w0, wa), 0.0)
    # midpoint rule
    prefix = (np.cumsum(inv_h, axis=1) - 0.5 * inv_h) * dz[:, None] * h0
    r = _curvature(prefix, kappa) * LIGHT_KM / h0
    integrand = np.where(live, LIGHT_KM * r * r * inv_h, 0.0)
    out = integrand.sum(axis=1) * dz
    return np.where(zmax > 0.0, out, VOL_NULL)


def compute_vmax(fitter, results, batch, fit_mask, model_flux, maglim, cospar):
    """Per-band zmax and Vmax for each SN (VMAX_ANA, snlc_fit.F90:15170).

    ``maglim`` maps short band name to its limiting magnitude. Returns
    {column: array}, five columns per band. Post-fit and vectorised over SNe:
    the secant iteration runs on the whole live set at once, so the model is
    evaluated once per iteration rather than once per SN.
    """
    if len(maglim) > MXFILT_VMAX:
        raise SNANAError(
            f"MAGLIM_VMAX names {len(maglim)} bands; SNANA's bound is "
            f"MXFILT_VMAX={MXFILT_VMAX}", function_name="MAGLIM_VMAX")
    if fitter.model.distance_param is None:
        raise SNANAError(
            "MAGLIM_VMAX rescales the fitted distance to a trial redshift, "
            "and this model declares none.", function_name="MAGLIM_VMAX")

    names = batch.filter_names
    zp = batch.zp_fluxcal
    n = results.n_sne
    mjd = np.asarray(batch.reference_mjds)[:, None] + np.asarray(batch.mjd_offsets)
    fidx = np.asarray(batch.filter_indices)
    z_sn = np.asarray(batch.redshifts)
    ref_mjd = np.asarray(batch.reference_mjds)
    t0 = np.asarray(results.params['PKMJD'])
    params = {k: np.asarray(v) for k, v in results.params.items()}
    mw_att = np.asarray(batch.mw_att_obs)

    out = {}
    for band, lim in maglim.items():
        idx = [i for i, nm in names.items() if filter_band(nm) == band]
        if not idx:
            raise SNANAError(
                f"MAGLIM_VMAX names band '{band}', which is not among the "
                f"fitted filters", function_name="MAGLIM_VMAX")
        in_band = fit_mask & np.isin(fidx, idx)
        flux = np.where(in_band, model_flux, -9.9e8)
        best = np.argmax(flux, axis=1)
        rows = np.arange(n)
        has = in_band.any(axis=1)
        fbest = flux[rows, best]
        mag = np.where(has & (fbest > 0),
                       zp - 2.5 * np.log10(np.maximum(fbest, 1e-30)), MAG_NULL)
        trest = np.where(has, (mjd[rows, best] - t0) / (1.0 + z_sn), TREST_NULL)
        nep = in_band.sum(axis=1)

        zmax = np.full(n, Z_NULL)
        vol = np.full(n, VOL_NULL)
        live = np.flatnonzero(has & (np.abs(mag) < 98.0))
        if live.size:
            # epoch fixed from the fit (snlc_fit.F90:15250)
            epoch_mjd = mjd[live, best[live]][:, None]
            epoch_filt = fidx[live, best[live]][:, None]
            epoch_ok = np.ones_like(epoch_filt, dtype=bool)
            att = mw_att[live]
            base = {k: v[live] for k, v in params.items()}
            mu0 = np.asarray(fitter.config.fit.distance_modulus(z_sn[live]))
            z_trial = z_sn[live].copy()
            m = mag[live].copy()
            for _ in range(MAX_ITER + 1):
                magdif = lim - m
                settled = np.abs(magdif) <= MAGDIF_CONVERGE
                if settled.all():
                    break
                z_trial = np.where(settled, z_trial,
                                   z_trial * 10.0 ** (0.2 * magdif))
                mu1 = np.asarray(fitter.config.fit.distance_modulus(z_trial))
                trial = fitter.model.shift_amplitude(dict(base), mu1 - mu0)
                f, _ = fitter.evaluate_flux(
                    trial, epoch_mjd, z_trial, epoch_filt, epoch_ok,
                    mw_att_obs=att, reference_mjds=ref_mjd[live],
                    photometry=batch.photometry)
                fnew = np.asarray(f)[:, 0]
                m = np.where(settled, m,
                             zp - 2.5 * np.log10(np.maximum(fnew, 1e-30)))
            else:
                unconverged = np.abs(lim - m) > MAGDIF_CONVERGE
                bad = [str(batch.snids[i]) for i in live[unconverged]][:5]
                raise SNANAError(
                    f"Vmax did not converge for {int(unconverged.sum())} SN(e) "
                    f"in band {band} after {MAX_ITER} iterations: {bad}",
                    function_name="MAGLIM_VMAX")
            zmax[live] = z_trial
            vol[live] = 4.0 * np.pi * dvdz_integral(z_trial, cospar)

        out[f'mag_Vmax_{band}'] = mag
        out[f'Trest_Vmax_{band}'] = trest
        out[f'z_Vmax_{band}'] = zmax
        out[f'vol_Vmax_{band}'] = vol
        out[f'NEP_Vmax_{band}'] = nep.astype(int)
    return out
