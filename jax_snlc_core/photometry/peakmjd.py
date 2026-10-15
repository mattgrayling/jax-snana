"""SNANA Fmax-clump peak-MJD estimator (OPT_SETPKMJD=16, get_obs_atFLUXMAX).

Shared by the simulator (writes the observed PEAKMJD column) and the fitter
(sets SNLC_SEARCH_PEAKMJD for the pre-fit TREST cuts and t0 init).
"""

from functools import partial

import jax
import jax.numpy as jnp

# OPT_SETPKMJD bits (sntools.h:412-414)
OPTMASK_SETPKMJD_FLUXMAX = 8
OPTMASK_SETPKMJD_FLUXMAX2 = 16
OPTMASK_SETPKMJD_FLUXMAX3 = 32

PEAKMJD_MISSING = -9.0
_MJDSTEP = 10.0


def _find_peak_single_sn(mjd, flux, snr, n_obs, snrcut, nwin_combine,
                         n_bins, log_snr_weight=False, real_bins=None):
    if real_bins is None:
        real_bins = n_bins
    max_obs = mjd.shape[0]
    obs_mask = jnp.arange(max_obs) < n_obs

    sort_indices = jnp.argsort(mjd)
    sorted_mjd = mjd[sort_indices]
    sorted_flux = flux[sort_indices]
    sorted_snr = snr[sort_indices]
    sorted_obs_mask = obs_mask[sort_indices]

    mjd_min = jnp.min(jnp.where(sorted_obs_mask, sorted_mjd, jnp.inf))

    snr_mask = (sorted_snr >= snrcut) & sorted_obs_mask

    wgt_snrcut = jnp.zeros(n_bins, dtype=jnp.float32)
    omin_snrcut = jnp.full(n_bins, max_obs, dtype=jnp.int32)
    omax_snrcut = jnp.full(n_bins, -1, dtype=jnp.int32)

    bin_indices = jnp.floor((sorted_mjd - mjd_min) / _MJDSTEP).astype(jnp.int32)
    bin_indices = jnp.clip(bin_indices, 0, real_bins - 1)

    obs_indices = jnp.arange(max_obs)

    # OPT_SETPKMJD bit 16 weights each qualifying epoch by 1; bit 32 adds
    # log10(SNR) on top of that 1 (sntools.c:1805-1806).
    weights = snr_mask.astype(jnp.float32)
    if log_snr_weight:
        weights = weights + jnp.where(
            snr_mask, jnp.log10(jnp.maximum(sorted_snr, 1.0e-9)), 0.0)
    wgt_snrcut = wgt_snrcut.at[bin_indices].add(weights)
    valid_obs_indices = jnp.where(snr_mask, obs_indices, max_obs).astype(jnp.int32)
    omin_snrcut = omin_snrcut.at[bin_indices].min(valid_obs_indices)
    invalid_obs_indices = jnp.where(snr_mask, obs_indices, -1).astype(jnp.int32)
    omax_snrcut = omax_snrcut.at[bin_indices].max(invalid_obs_indices)

    kernel = jnp.ones(nwin_combine)
    wgt_sums = jnp.convolve(wgt_snrcut, kernel, mode='valid')
    wgt_sums = jnp.where(jnp.arange(wgt_sums.size) <= real_bins - nwin_combine,
                         wgt_sums, -jnp.inf)
    best_wgt = jnp.max(wgt_sums)
    best_window_start_idx = jnp.argmax(wgt_sums)

    best_window_bins = jax.lax.dynamic_slice(
        jnp.arange(n_bins), (best_window_start_idx,), (nwin_combine,))
    window_omins = omin_snrcut[best_window_bins]
    window_omaxs = omax_snrcut[best_window_bins]
    omin_best = jnp.min(jnp.where(window_omins < max_obs, window_omins, max_obs))
    omax_best = jnp.max(window_omaxs)

    final_mask = (obs_indices >= omin_best) & (obs_indices <= omax_best) & snr_mask
    flux_in_clump = jnp.where(final_mask, sorted_flux, -jnp.inf)
    idx_of_max_flux = jnp.argmax(flux_in_clump)
    max_flux_value = flux_in_clump[idx_of_max_flux]
    peak_mjd = sorted_mjd[idx_of_max_flux]

    valid_result = (n_obs >= 3) & (best_wgt > 0) & (max_flux_value > -jnp.inf)
    return jnp.where(valid_result, peak_mjd, PEAKMJD_MISSING)


@partial(jax.jit, static_argnames=('nwin_combine', 'n_bins', 'log_snr_weight'))
def _clump_peaks(mjds, fluxcals, snrs, n_obs, snrcut, snrcut_backup,
                 real_bins, *, nwin_combine, n_bins, log_snr_weight):
    """Both S/N thresholds at one padded shape; real bins bound valid windows."""
    run = jax.vmap(_find_peak_single_sn,
                   in_axes=(0, 0, 0, 0, None, None, None, None, None))
    primary = run(mjds, fluxcals, snrs, n_obs, snrcut, nwin_combine, n_bins,
                  log_snr_weight, real_bins)
    backup = run(mjds, fluxcals, snrs, n_obs, snrcut_backup, nwin_combine,
                 n_bins, log_snr_weight, real_bins)
    return jnp.where(primary == PEAKMJD_MISSING, backup, primary)


@jax.jit
def naive_fluxmax_peakmjd(mjds, fluxcals, snrs, n_obs_per_sn, *, snrcut_backup):
    """OPT_SETPKMJD bit 8: the MJD of the brightest epoch passing SNRCUT_BACKUP,
    with no clumping. SNANA runs a single iteration for this option and starts
    from the backup cut (sntools.c:1682)."""
    mjds, fluxcals = jnp.asarray(mjds), jnp.asarray(fluxcals)
    snrs, n_obs = jnp.asarray(snrs), jnp.asarray(n_obs_per_sn)
    valid = (jnp.arange(mjds.shape[1])[None, :] < n_obs[:, None]) & \
            (snrs >= snrcut_backup)
    flux = jnp.where(valid, fluxcals, -jnp.inf)
    peak = jnp.take_along_axis(mjds, jnp.argmax(flux, axis=1)[:, None], 1)[:, 0]
    return jnp.where(jnp.any(valid, axis=1), peak, PEAKMJD_MISSING)


def fmax_clump_peakmjd(mjds, fluxcals, snrs, n_obs_per_sn, *,
                       mjdwin, snrcut, snrcut_backup, log_snr_weight=False,
                       mjd_span=None, histogram_bins=None):
    """Estimate the observed peak MJD per SN via SNANA's Fmax-clump method,
    returning [n_sne] peak MJDs with PEAKMJD_MISSING where no valid clump was
    found.

    ``mjds``, ``fluxcals`` and ``snrs`` are [n_sne, max_obs] observation
    arrays; padded slots are ignored via ``n_obs_per_sn`` ([n_sne] valid-
    observation count per SN), and ``snrs`` must be non-negative and finite
    (0 for padded/invalid). ``mjdwin`` is the MJD width of the clump window
    (MJDWIN_SETPKMJD); ``snrcut`` and ``snrcut_backup`` are the primary and
    fallback S/N thresholds. ``mjd_span`` is the MJD extent covering every
    light curve, sizing the clump histogram as SNANA sizes MXWIN_SNRCUT
    (get_obs_atFLUXMAX, sntools.c): a traced caller must pass it; a concrete
    caller may omit it and the span is read from the data. ``histogram_bins``
    can reserve a common capacity for different spans; only bins belonging
    to this sample participate in the peak search.
    """
    mjds = jnp.asarray(mjds)
    fluxcals = jnp.asarray(fluxcals)
    snrs = jnp.asarray(snrs)
    n_obs = jnp.asarray(n_obs_per_sn)
    nwin_combine = int(mjdwin / _MJDSTEP + 0.01)

    if mjd_span is None:
        valid = jnp.arange(mjds.shape[1])[None, :] < n_obs[:, None]
        mjd_span = float(jnp.max(jnp.where(valid, mjds, -jnp.inf))
                         - jnp.min(jnp.where(valid, mjds, jnp.inf)))
    n_bins = max(int(mjd_span / _MJDSTEP) + 1 if mjd_span > 0.0 else 1,
                 nwin_combine)

    if histogram_bins is not None and histogram_bins < n_bins:
        raise ValueError('Peak histogram capacity is smaller than the data span')
    return _clump_peaks(
        mjds, fluxcals, snrs, n_obs, snrcut, snrcut_backup, n_bins,
        nwin_combine=nwin_combine,
        n_bins=n_bins if histogram_bins is None else histogram_bins,
        log_snr_weight=log_snr_weight)
