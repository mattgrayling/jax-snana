"""Maximum redshift and accessible comoving volume at a magnitude limit."""

import numpy as np

from .cosmology import comoving_volume_per_steradian, reference_distance_modulus
from .filters import filter_band
from .logging import SNANAError


MAGDIF_CONVERGE = 0.001     # snlc_fit.F90:937
MAX_ITER = 20               # :15311
MXFILT_VMAX = 5             # :940


# per-band init when the SN has no usable epoch (:15176)
MAG_NULL, TREST_NULL, Z_NULL, VOL_NULL = 99.0, 99.0, -99.0, -99.0


def compute_vmax(model, params, maglim, cospar, *, evaluate_flux,
                 mjds, reference_mjds, redshifts, filter_indices, filter_names,
                 fit_mask, model_flux, zp_fluxcal, mw_att_obs, photometry, snids):
    """Per-band zmax and Vmax for each SN (VMAX_ANA, snlc_fit.F90:15170).

    ``maglim`` maps short band name to its limiting magnitude. Returns
    {column: array}, five columns per band. Post-fit and vectorised over SNe:
    the secant iteration runs on the whole live set at once, so the model is
    evaluated once per iteration rather than once per SN.

    Observation arrays have shape (events, epochs); parameter arrays have one
    value per event. ``evaluate_flux`` evaluates the supplied model at trial
    parameters and redshifts, returning flux and flux uncertainty. ``cospar``
    is (H0, Omega_m, Omega_Lambda, w0, wa).
    """
    if len(maglim) > MXFILT_VMAX:
        raise SNANAError(
            f"MAGLIM_VMAX names {len(maglim)} bands; SNANA's bound is "
            f"MXFILT_VMAX={MXFILT_VMAX}", function_name="MAGLIM_VMAX")
    if model.distance_param is None:
        raise SNANAError(
            "MAGLIM_VMAX rescales the fitted distance to a trial redshift, "
            "and this model declares none.", function_name="MAGLIM_VMAX")

    names = filter_names
    zp = zp_fluxcal
    n = len(redshifts)
    mjd = np.asarray(mjds)
    fidx = np.asarray(filter_indices)
    z_sn = np.asarray(redshifts)
    ref_mjd = np.asarray(reference_mjds)
    params = {k: np.asarray(v) for k, v in params.items()}
    t0 = params['PKMJD']
    mw_att = np.asarray(mw_att_obs)

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
            mu0 = np.asarray(reference_distance_modulus(z_sn[live], *cospar))
            z_trial = z_sn[live].copy()
            m = mag[live].copy()
            for _ in range(MAX_ITER + 1):
                magdif = lim - m
                settled = np.abs(magdif) <= MAGDIF_CONVERGE
                if settled.all():
                    break
                z_trial = np.where(settled, z_trial,
                                   z_trial * 10.0 ** (0.2 * magdif))
                mu1 = np.asarray(reference_distance_modulus(z_trial, *cospar))
                trial = model.shift_amplitude(dict(base), mu1 - mu0)
                f, _ = evaluate_flux(
                    trial, epoch_mjd, z_trial, epoch_filt, epoch_ok,
                    mw_att_obs=att, reference_mjds=ref_mjd[live],
                    photometry=photometry)
                fnew = np.asarray(f)[:, 0]
                m = np.where(settled, m,
                             zp - 2.5 * np.log10(np.maximum(fnew, 1e-30)))
            else:
                unconverged = np.abs(lim - m) > MAGDIF_CONVERGE
                bad = [str(snids[i]) for i in live[unconverged]][:5]
                raise SNANAError(
                    f"Vmax did not converge for {int(unconverged.sum())} SN(e) "
                    f"in band {band} after {MAX_ITER} iterations: {bad}",
                    function_name="MAGLIM_VMAX")
            zmax[live] = z_trial
            vol[live] = 4.0 * np.pi * comoving_volume_per_steradian(z_trial, *cospar)

        out[f'mag_Vmax_{band}'] = mag
        out[f'Trest_Vmax_{band}'] = trest
        out[f'z_Vmax_{band}'] = zmax
        out[f'vol_Vmax_{band}'] = vol
        out[f'NEP_Vmax_{band}'] = nep.astype(int)
    return out
