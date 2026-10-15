"""1D redshift-interpolated biasCor (SALT2mu.c prepare_biasCor_zinterp:11250).

MASK_BIASCOR_1DZAVG (WGT=1) / MASK_BIASCOR_1DZWGT (WGT=1/muerrsq): the distance
bias in each z bin is the weighted mean of ``mu_fit - SIM_MUz`` over the biasCor
sample at the fixed input alpha/beta, minus a high-SNR reference, linearly
interpolated onto each data event's z. It is frozen at init (independent of the
fitted alpha/beta), so it folds straight into mumodel.
"""

import numpy as np

from .config import MASK_BIASCOR_1DZAVG, parval
from .distances import mu_coef, mu_variance_scaled, muerr_vpec_sq
from .samples import assign_biascor, sample_binsize


def biascor_1d_grids(biascor_tv, samples, config, model):
    """Each sample's frozen 1D distance bias from the biasCor, on the data's
    partition ``samples``: ``(zavg, mubias, zlo, zhi)`` over its populated z
    bins, or None for a sample with no bias (unselected or unpopulated). Also
    the biasCor events' IDSAMPLE, their cutmask and how many were in range.
    Nothing here reads a data event, so cells sharing settings, biasCor and
    partition share it; :func:`biascor_1d_data` applies it to one cell's data."""

    # muBias is frozen at the fit's initial parval (SALT2mu.c:22354)
    nuis = {n: parval(config, n) for n in model.nuisance_spec}
    m0ref = model.m0_reference
    sigmb = config.sigint

    use_avg = (config.opt_biascor & MASK_BIASCOR_1DZAVG) > 0

    bc_idsample = assign_biascor(biascor_tv, samples, config, model)

    names = model.lcpar_names
    lcpar = {n: biascor_tv.lcpar[n] for n in names}
    host = {"z": biascor_tv.zhd, "gamma_var": biascor_tv.logmass}
    mu_fit = np.asarray(model.compute_mu(
        lcpar, nuis, host, config._u_flags, apply_mass_step=False)) - m0ref
    bias = mu_fit - np.asarray(biascor_tv.mumodel)      # mu_fit - SIM_MUz

    if use_avg:
        wgt_all = np.ones(biascor_tv.size)
    else:
        v = np.tile(mu_coef(model, nuis, config._u_flags), (biascor_tv.size, 1))
        muerrsq = (mu_variance_scaled(v, biascor_tv.covmat_fit, biascor_tv.zhd,
                                      config.lensing_zpar,
                                      sigint_sq=sigmb * sigmb)
                   + muerr_vpec_sq(biascor_tv.zhd, biascor_tv.zmuerr))
        wgt_all = 1.0 / muerrsq

    # each sample bins on its own RANGE_REDSHIFT / BINSIZE_REDSHIFT (:11243)
    binsize = [sample_binsize(s, config, model) for s in samples]
    zc, snr = biascor_tv.zhd, biascor_tv.snrmax
    select = set(config.idsample_select)

    in_range = np.zeros(biascor_tv.size, dtype=bool)
    for s, bs in zip(samples, binsize):
        zlo0, zhi0 = bs.zrange
        in_range |= ((bc_idsample == s.idsample) & (zc >= zlo0) & (zc <= zhi0))
    in_range &= biascor_tv.cutmask

    # the high-SNR reference is global, not per-sample (SALT2mu.c:12160, :12221)
    hisnr = in_range & (snr > config.snrmin_sigint_biascor)
    w_all = np.where(in_range, wgt_all, 0.0)
    sw_hi = w_all[hisnr].sum()
    mu_hisnr = (w_all[hisnr] * bias[hisnr]).sum() / sw_hi if sw_hi > 0 else 0.0

    grids = []
    for s, bs in zip(samples, binsize):
        # DOFLAG_SELECT=0: the sample's data are cut, not corrected (:12259)
        if select and s.idsample not in select:
            grids.append(None)
            continue
        zlo0, zhi0 = bs.zrange
        zbin = bs.zbin
        nzbin = int(np.ceil((zhi0 - zlo0) / zbin))
        bmask = in_range & (bc_idsample == s.idsample)
        iz = np.clip(((zc - zlo0) / zbin).astype(int), 0, nzbin - 1)
        w = np.where(bmask, wgt_all, 0.0)
        sumwgt = np.bincount(iz, weights=w, minlength=nzbin)
        summu = np.bincount(iz, weights=w * bias, minlength=nzbin)

        pop = sumwgt > 1e-9
        mubias = np.zeros(nzbin)
        mubias[pop] = summu[pop] / sumwgt[pop] - mu_hisnr
        izs = np.nonzero(pop)[0]
        if izs.size == 0:
            grids.append(None)
            continue
        izmin, izmax = izs[0], izs[-1]
        zavg = zlo0 + zbin * (np.arange(nzbin) + 0.5)
        grids.append((zavg[izmin:izmax + 1], mubias[izmin:izmax + 1],
                      zlo0 + zbin * izmin, zlo0 + zbin * (izmax + 1)))

    return grids, bc_idsample, biascor_tv.cutmask, int(in_range.sum())


def biascor_1d_data(grids, samples, data_tv, data_idsample):
    """Return ``(muBias_zinterp, accept)`` for one cell's data events: each
    event's bias interpolated on its sample's grid (:func:`biascor_1d_grids`),
    and the mask of events with one."""
    muBias_data = np.zeros(data_tv.size)
    accept = np.zeros(data_tv.size, dtype=bool)
    zd = data_tv.zhd
    for s, grid in zip(samples, grids):
        if grid is None:
            continue
        zavg, mubias, zlo, zhi = grid
        inrange = (data_idsample == s.idsample) & (zd >= zlo) & (zd <= zhi)
        muBias_data[inrange] = np.interp(zd[inrange], zavg, mubias)
        accept |= inrange
    return muBias_data, accept
