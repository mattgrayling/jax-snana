"""SNANA-format text-table construction (FITRES, SNANA, OUTLIER, LCPLOT).

SNTABLE_LIST itself is parsed in config_parser; this module builds the columns.
"""

import logging

import numpy as np
import pandas as pd

from jax_snlc_core.io.fitres_reader import FitresTable
from jax_snlc_core.io.snana_table import (jax_version_comment, program_comment,
                                          snana_version_comment)

from jax_snlc_core.models import MODEL_REGISTRY, FittedModel
from jax_snlc_core.photometry.filter_system import filter_band
from jax_snlc_core.utils import (HOSTGAL_PREFIXES, HOSTGAL_PROPERTY_NULL,
                                 SIM_MAGOBS_UNSET, SNANA_NULLFLOAT)

from .batch import band_snr_maxima, snrmax_band_mask
from .data_loader import FITRES_SIM_COLUMNS
from .host_noise import sbmag_from_fluxcal
from .loss_functions import precompute_zpdf_spline, zpdf_rows_usable
from .survey import event_survey_ids
from .minos import profile_error_type
from jax_snlc_core.physics.constants import LIGHT_KM
from jax_snlc_core.physics.cosmology import zhelio_to_zcmb
from jax_snlc_core.photometry.noise import fluxcal_errcalc

logger = logging.getLogger(__name__)


_HOST_PROPERTY_COLS = ('LOGMASS', 'LOGMASS_ERR', 'LOGSFR', 'LOGSFR_ERR',
                       'LOGsSFR', 'LOGsSFR_ERR', 'COLOR', 'COLOR_ERR')


def _head_scalar(batch, col, default=0.0):
    v = batch.head_extras.get(col)
    return (np.asarray(v, dtype=float) if v is not None
            else np.full(batch.n_sne, default))


def _data_zhel(batch):
    """(zHEL, zHELERR) from the HEAD helio columns, falling back to the
    batch redshifts (REDSHIFT_FINAL)."""
    if 'REDSHIFT_HELIO' in batch.head_extras:
        return (_head_scalar(batch, 'REDSHIFT_HELIO'),
                _head_scalar(batch, 'REDSHIFT_HELIO_ERR', 0.001))
    return np.asarray(batch.redshifts), np.asarray(batch.redshift_errs)


def build_event_columns(batch, fit_mask=None, zfit=None, zfit_err=None,
                        nepoch_outlier=None, snrmax_filters='',
                        restore_wrong_vpec=False):
    """Per-SN FITRES/SNANA-table event columns, split around the host block:
    (pre_host, post_host, fmt). For photo-z fits SNANA fills the redshift
    columns from the fitted zPHOT: pass it as zfit/zfit_err. When the
    OUTLIER table is active both tables gain NEPOCH_OUTLIER after DEC:
    pass its per-CID counts."""
    n = batch.n_sne

    def extra(col, default=0.0):
        return _head_scalar(batch, col, default)

    ras = np.asarray(batch.ras)
    decs = np.asarray(batch.decs)
    if zfit is not None:
        # a fitted redshift replaces the data one, so zCMB follows it
        zhel = np.asarray(zfit)
        zhelerr = np.asarray(zfit_err)
        zcmb = np.asarray(zhelio_to_zcmb(zhel, ras, decs))
    else:
        zhel, zhelerr = _data_zhel(batch)
        zcmb = np.asarray(batch.zcmb)

    # zHD = zCMB + VPEC correction (REDSHIFT_HD, snana.F90:17040)
    vpec = extra('VPEC')
    vpec_err = extra('VPEC_ERR')
    zpec = vpec / LIGHT_KM
    docor = (vpec != 0.0) | (vpec_err != 0.0)
    # the key also restores the wrong sign of the VPEC term (snana.F90:17045)
    corrected = ((1.0 + zcmb) * (1.0 + zpec) - 1.0 if restore_wrong_vpec
                 else (1.0 + zcmb) / (1.0 + zpec) - 1.0)
    zhd = np.where(docor, corrected, zcmb)
    zhderr = np.where(
        docor, np.sqrt(zhelerr**2 + (vpec_err / LIGHT_KM * (1.0 + zcmb))**2),
        zhelerr)

    sntype = batch.head_extras.get('SNTYPE')
    field = batch.head_extras.get('FIELD')
    pre = {
        'IDSURVEY': event_survey_ids(
            batch.survey, batch.head_extras.get('SUBSURVEY'), n),
        'TYPE': (np.asarray(sntype, dtype=int) if sntype is not None
                 else np.zeros(n, dtype=int)),
        'FIELD': (np.array([str(f).strip() or 'VOID' for f in field])
                  if field is not None else np.full(n, 'VOID')),
        'CUTFLAG_SNANA': np.full(n, 3),
        'ERRFLAG_FIT': np.zeros(n, dtype=int),
        'RA': ras,
        'DEC': decs,
        **({'NEPOCH_OUTLIER': np.array(
            [nepoch_outlier.get(c, 0)
             for c in np.char.strip(np.asarray(batch.snids).astype(str))])}
           if nepoch_outlier is not None else {}),
        'zHEL': zhel,
        'zHELERR': zhelerr,
        'zCMB': zcmb,
        'zCMBERR': zhelerr,
        'zHD': zhd,
        'zHDERR': zhderr,
        'VPEC': vpec,
        'VPECERR': vpec_err,
        'LENSDMU': extra('LENSDMU'),
        'LENSDMUERR': extra('LENSDMU_ERR'),
        'MWEBV': np.asarray(batch.mwebvs),
    }

    snr, band_max, has_epoch = band_snr_maxima(batch, mask=fit_mask)
    # only SNRMAX honours SNRMAX_FILTERS (snana.F90:17454)
    use_band = snrmax_band_mask(batch.filter_names, snrmax_filters)
    ranked = np.sort(np.where(use_band, band_max, SNANA_NULLFLOAT), axis=1)[:, ::-1]
    obs_ok = np.asarray(fit_mask if fit_mask is not None
                        else batch.observation_mask)
    snrsum_sq = np.sum(np.where(snr > SNANA_NULLFLOAT, snr, 0.0)**2 * obs_ok, axis=1)
    band_chars = [filter_band(batch.filter_names[b])
                  for b in range(len(batch.filter_names))]
    post = {
        # SNLC_SEARCH_PEAKMJD, which SET_PEAKMJD overwrites with the
        # OPT_SETPKMJD estimate (snana.F90:26797)
        'PKMJDINI': extra('PEAKMJD', SNANA_NULLFLOAT),
        'SNRMAX1': ranked[:, 0],
        'SNRMAX2': ranked[:, 1],
        'SNRMAX3': ranked[:, 2],
        'SNRSUM': np.sqrt(snrsum_sq),
        'BANDLIST': np.array([''.join(c for b, c in enumerate(band_chars)
                                      if has_epoch[i, b]) for i in range(n)]),
    }

    fmt = {'IDSURVEY': '%d', 'TYPE': '%d', 'FIELD': '%s',
           'CUTFLAG_SNANA': '%d', 'ERRFLAG_FIT': '%d',
           'NEPOCH_OUTLIER': '%d', 'BANDLIST': '%s'}
    return pre, post, fmt


# (column, HEAD source) in SNANA order (snana.F90:24280-24380)
_SNANA_SIM_COLUMNS = [
    ('SIM_GENTYPE', 'SIM_GENTYPE'),
    ('SIM_TEMPLATE_INDEX', 'SIM_TEMPLATE_INDEX'),
    ('SIM_LIBID', 'SIM_LIBID'),
    ('SIM_NGEN_LIBID', 'SIM_NGEN_LIBID'),
    ('SIM_ZCMB', 'SIM_REDSHIFT_CMB'),
    ('SIM_ZHOST_MATCH', 'SIM_REDSHIFT_HOST_MATCH'),
    ('SIM_ZFLAG', 'SIM_REDSHIFT_FLAG'),
    ('SIM_VPEC', 'SIM_VPEC'),
    ('SIM_DLMAG', 'SIM_DLMU'),
    ('SIM_LENSDMU', 'SIM_LENSDMU'),
    ('SIM_PEAKMJD', 'SIM_PEAKMJD'),
    ('SIM_WGT_POP', 'SIM_WGT_POPULATION'),
]


def snana_sim_block(sim_truth, host_text):
    """Ordered SIM output columns (SNANA names) from HEAD-named sim truth;
    shared by the FITRES and SNANA tables. The model's columns come before
    SIM_AV/SIM_RV (snana.F90:24344-24380): the renamed ones as declared,
    the rest as the HEAD names them. SIM_HOSTLIB block is text:host-gated
    (snana.F90:24424)."""
    sim_truth = sim_truth or {}
    columns = list(_SNANA_SIM_COLUMNS)
    for cls in dict.fromkeys(MODEL_REGISTRY.values()):
        if issubclass(cls, FittedModel):
            model_columns = (cls.SNANA_TABLE_SIM_COLUMNS or
                             tuple((c, c) for c in cls.SIM_TRUTH_COLUMNS))
            columns.extend(pair for pair in model_columns if pair not in columns)
    out = {}
    for col, src in columns:
        if src in sim_truth:
            out[col] = np.asarray(sim_truth[src])
    declared = {src for _, src in columns}
    for k in sim_truth:
        if (k not in declared and k not in FITRES_SIM_COLUMNS
                and not k.startswith('SIM_HOSTLIB_')):
            out[k] = np.asarray(sim_truth[k])
    for col in ('SIM_AV', 'SIM_RV'):
        if col in sim_truth:
            out[col] = np.asarray(sim_truth[col])
    if host_text:
        for k in sim_truth:
            if k.startswith('SIM_HOSTLIB_'):
                out[k] = np.asarray(sim_truth[k])
    return out


def build_snana_table(batch, output, cutflag, errflag,
                      zfit=None, zfit_err=None, nepoch_outlier=None,
                      snrmax_filters='', restore_wrong_vpec=False,
                      *, quantile_spline):
    """SNANA-table columns for every event in ``batch``, which is the batch as
    loaded rather than the selected one so cut events keep their rows.

    ``cutflag`` and ``errflag`` are per-event and come from the run, which
    is where the outcome of each stage is known. Prefiltered events carry
    all-masked epochs, so the SNR columns fall out as SNANA's sentinels (-9,
    SNRSUM 0). Returns (varnames, columns, fmt)."""
    n = batch.n_sne
    snids = np.asarray(batch.snids)
    fitted = np.asarray(errflag) == 0

    if zfit is not None:
        zfull, zefull = (np.array(a) for a in _data_zhel(batch))
        zfull[fitted] = np.asarray(zfit)
        zefull[fitted] = np.asarray(zfit_err)
        pre, post, fmt = build_event_columns(
            batch, zfit=zfull, zfit_err=zefull,
            nepoch_outlier=nepoch_outlier, snrmax_filters=snrmax_filters,
            restore_wrong_vpec=restore_wrong_vpec)
    else:
        pre, post, fmt = build_event_columns(
            batch, nepoch_outlier=nepoch_outlier,
            snrmax_filters=snrmax_filters,
            restore_wrong_vpec=restore_wrong_vpec)
    pre['CUTFLAG_SNANA'] = np.asarray(cutflag)
    pre['ERRFLAG_FIT'] = np.asarray(errflag)

    columns = {'CID': snids}
    if output.iauc:
        iauc = batch.head_extras.get('NAME_IAUC')
        columns['IAUC'] = (np.array(['UNKNOWN' if not str(v).strip() else
                                     str(v).strip() for v in iauc])
                           if iauc is not None else np.full(n, 'UNKNOWN'))
    columns.update(pre)
    if batch.host_data is not None:
        columns.update(build_host_columns(
            batch.host_data, batch.zp_fluxcal, host_text=output.host_text,
            host2_text=output.host2_text,
            zpdf_quantiles=batch.zpdf_quantiles,
            zpdf_percentiles=batch.zpdf_percentiles,
            zpdf_nbin=batch.zpdf_nbin,
            quantile_spline=quantile_spline))
    post.pop('BANDLIST')
    columns.update(post)

    columns.update(snana_sim_block(
        batch.sim_truth, output.host_text or output.host2_text))

    # SNLC_CCID is a string (sntools_output_text.c)
    fmt = {'CID': '%s', 'IAUC': '%s', **fmt}
    for col in columns:
        if col not in fmt:
            fmt[col] = ('%d' if np.issubdtype(np.asarray(columns[col]).dtype,
                                              np.integer) else None)

    # row gating (snana.F90:23149): 63 -> all, 0 -> CUTFLAG==0,
    # else CUTFLAG & mask > 0
    cm = output.cutmask
    if cm != 63:
        cutflag = columns['CUTFLAG_SNANA']
        keep = (cutflag == 0) if cm == 0 else (cutflag & cm) > 0
        columns = {k: np.asarray(v)[keep] for k, v in columns.items()}
    return list(columns), columns, fmt


# IFILTOBS = index in SNANA's filter alphabet (FILTERSTRING_DEFAULT,
# sntools.h:159)
_FILTERSTRING = (' ugrizYJHK UBVRIXy0123456789 abcdef ACDEFGLMNOPQSTWZ '
                 'hjklmnopqstvwx')


def build_outlier_table(batch, output, model_flux, model_flux_err):
    """OUTLIER rows: epochs with NSIG >= the nsig knob, where
    ``NSIG = |F_data-F_fit| / sqrt(err_data^2 + err_fit^2)``
    (TABLE_OUTLIER, snana.F90:22485; NSIG_OUTLIER_FIT, snana.F90:22921).
    The ftrue/sbmag knobs only act in snana.exe sim jobs (fit jobs pass
    constants, snana.F90:22557). Returns (varnames, columns, fmt,
    summary_lines).

    The population is every epoch SNRECON accepted (snana.F90:22548), not the
    narrower set the fit used: an epoch the fit rejected is still an outlier,
    and is the one most worth reporting."""
    ox = batch.obs_extras or {}
    mask = np.asarray(batch.observation_mask, dtype=bool)
    flux = np.asarray(batch.fluxcals)
    err = np.asarray(batch.fluxcalerrs)
    mfit = np.asarray(model_flux)
    mferr = np.asarray(model_flux_err)

    errtot = np.sqrt(err**2 + mferr**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        nsig = np.where(mask & (mferr > 1e-5),
                        np.abs(flux - mfit) / errtot, SNANA_NULLFLOAT)
    used = mask & (nsig > -0.01)
    out2d = used & (nsig >= output.outlier_nsig)
    isn, iep = np.nonzero(out2d)

    # pull against truth (snana.F90:22884); -9 for data (:17350)
    simmag = (batch.obs_extras or {}).get('SIM_MAGOBS')
    if batch.sim_truth is not None and simmag is not None:
        simmag = np.asarray(simmag)
        trueflux = np.where(simmag < SIM_MAGOBS_UNSET,
                            10.0 ** (-0.4 * (simmag - batch.zp_fluxcal)), 0.0)
        sim_pull = (trueflux - flux) / err
    else:
        sim_pull = np.full_like(flux, SNANA_NULLFLOAT)
    nsig_report = np.abs(sim_pull)

    fidx = np.asarray(batch.filter_indices)
    band_chars = np.array([filter_band(batch.filter_names[b])
                           for b in range(len(batch.filter_names))])
    summary = ["==========================================",
               "    FLUX-OUTLIER SUMMARY per BAND for",
               f"    |F_data-F_fit|/Sigma_F > {output.outlier_nsig:5.2f} :"]
    for b, ch in enumerate(band_chars):
        n_used = int((used & (fidx == b)).sum())
        n_out = int((out2d & (fidx == b)).sum())
        frac = n_out / n_used if n_used else 0.0
        summary.append(f"       {ch}-Outlier frac = {frac:8.5f}   "
                       f"({n_out:6d}/{n_used:6d})")
    summary.append("")

    def ep(arr2d):
        return np.asarray(arr2d)[isn, iep]

    def ox_ep(col, default):
        return (ep(ox[col]) if col in ox
                else np.full(len(isn), default))

    mjd = np.asarray(batch.reference_mjds)[:, None] + \
        np.asarray(batch.mjd_offsets)
    zhel, _ = _data_zhel(batch)
    pkini = _head_scalar(batch, 'PEAKMJD', SNANA_NULLFLOAT)
    trest = (mjd - pkini[:, None]) / (1.0 + zhel[:, None])

    host = batch.host_data or {}
    sbflux_band = np.zeros((batch.n_sne, len(band_chars)))
    sbmag_band = np.full((batch.n_sne, len(band_chars)), 99.0)
    for b, ch in enumerate(band_chars):
        k = f'HOSTGAL_SB_FLUXCAL_{ch}'
        if k in host:
            sbflux_band[:, b] = np.asarray(host[k], dtype=float)
            sbmag_band[:, b] = sbmag_from_fluxcal(sbflux_band[:, b], batch.zp_fluxcal)

    # ERRCALC takes the fitted flux (snana.F90:20127)
    errcalc = fluxcal_errcalc(
        mfit, err, ox.get('ZEROPT', np.zeros_like(flux)),
        ox.get('ZEROPT_ERR', np.zeros_like(flux)),
        ox.get('SKY_SIG', np.zeros_like(flux)),
        ox.get('SKY_SIG_T', np.zeros_like(flux)),
        ox.get('PSF_SIG1', np.zeros_like(flux)),
        ox.get('PSF_SIG2', np.zeros_like(flux)),
        ox.get('PSF_RATIO', np.zeros_like(flux)),
        ox.get('GAIN', np.ones_like(flux)), batch.zp_fluxcal,
        sb_fluxcal=sbflux_band[np.arange(batch.n_sne)[:, None], fidx],
        pixsize=_head_scalar(batch, 'PIXSIZE')[:, None])

    chars = band_chars[ep(fidx)]
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = np.where(ep(err) > 0, ep(flux) / ep(err), 0.0)
        logsnr = np.where(snr > 0.1, np.log10(np.where(snr > 0, snr, 1.0)),
                          -1.0)
    columns = {
        'CID': np.asarray(batch.snids)[isn],
        'ROWNUM': np.arange(1, len(isn) + 1),
        'FIELD': ox_ep('FIELD', 'VOID'),
        'MJD': ep(mjd),
        'BAND': chars,
        'IFILTOBS': np.array([_FILTERSTRING.index(c) for c in chars]),
        'DETNUM': ox_ep('DETNUM', -9).astype(int),
        'IXPIX': ox_ep('XPIX', SNANA_NULLFLOAT).astype(int),
        'IYPIX': ox_ep('YPIX', SNANA_NULLFLOAT).astype(int),
        'AREAFRAC': np.full(len(isn), SNANA_NULLFLOAT),
        'PHOTFLAG': ox_ep('PHOTFLAG', 0).astype(int),
        'FLUXCAL_DATA': ep(flux),
        'FLUXCAL_ERR_DATA': ep(err),
        'FLUXCAL_ERR_CALC': ep(errcalc),
        'FLUXCAL_FIT': ep(mfit),
        'FLUXCAL_ERR_FIT': ep(mferr),
        'FLUXCAL_ERRTOT_FIT': ep(errtot),
        'NSIG': ep(nsig_report),
        'LOGSNR': logsnr,
        'SBMAG': sbmag_band[isn, ep(fidx)],
        'ZP': ox_ep('ZEROPT', SNANA_NULLFLOAT),
        'PSF': ox_ep('PSF_SIG1', SNANA_NULLFLOAT),
        'SKYSIG': ox_ep('SKY_SIG', SNANA_NULLFLOAT),
        'z': zhel[isn],
        'TREST': ep(trest),
    }
    fmt = {'CID': '%s', 'ROWNUM': '%d', 'FIELD': '%s', 'MJD': '%.4f',
           'BAND': '%s', 'IFILTOBS': '%d', 'DETNUM': '%d', 'IXPIX': '%d',
           'IYPIX': '%d', 'AREAFRAC': '%.6f', 'PHOTFLAG': '%d',
           'FLUXCAL_DATA': '%.5f', 'FLUXCAL_ERR_DATA': '%.5f',
           'FLUXCAL_ERR_CALC': '%.5f', 'FLUXCAL_FIT': '%.5f',
           'FLUXCAL_ERR_FIT': '%.5f', 'FLUXCAL_ERRTOT_FIT': '%.5f',
           'NSIG': '%.5f', 'LOGSNR': '%.5f', 'SBMAG': '%.5f', 'ZP': '%.5f',
           'PSF': '%.5f', 'SKYSIG': '%.5f', 'z': '%.5f', 'TREST': '%.5f'}
    return list(columns), columns, fmt, summary


# best-fit model grid: TOBS = -60 + 2*(i+0.5), 100 bins
# (SNLCPLOT, snlc_fit.F90:18567; DTOBS_MODEL_PLOT=2, snana.F90:5706)
LCPLOT_TOBS_GRID = -60.0 + 2.0 * (np.arange(100) + 0.5)

# SNANA's fixed LCLIST legend (sntools_output_text.c:912)
LCLIST_HEADER = """\
column defintion in each lightcurve file:
col  1 -> CID            :  Candidate ID
col  2 -> MJD            :  time of observation
col  3 -> Tobs           :  MJD-PKMJD
col  4 -> SIM_FLUXCAL    :  true flux
col  5 -> FLUXCAL        :  calibrated flux
col  6 -> FLUXCAL_ERR    :  error on calibrated flux
col  7 -> DATAFLAG       :  1=>data, 0=>fit, -1=>data rejected in fit
col  8 -> BAND           :  1 char band/filter
col  9 -> CHI2           :  data-fit chi2
col 10 -> IFIT           :  fit number out of 1 fits per SN
col 11 -> BAND_REST      :  optional rest-frame band
col 12 -> FLUX_REST      :  optional rest-frame flux
col 13 -> KCOR           :  optional kcor
col 14 -> AVWARP         :  optional AVwarp
col 15 -> SIM_FLUX_REST  :  optional rest-frame SIM flux"""


def build_lcplot_table(batch, fit_mask, model_flux, model_flux_err,
                       pkmjd, plot_idx, grid_flux, grid_flux_err):
    """LCPLOT rows for the plotted SNe, band-major: each band's data epochs
    (DATAFLAG 1 used / -1 rejected, per-epoch chi2) then the 100-point
    best-fit grid (DATAFLAG 0). grid_flux[_err] are [n_plot, n_bands, 100]
    on LCPLOT_TOBS_GRID. Returns (varnames, columns, fmt)."""
    ox = batch.obs_extras or {}
    mask = np.asarray(fit_mask, dtype=bool)
    obs_ok = np.asarray(batch.observation_mask)
    flux = np.asarray(batch.fluxcals)
    err = np.asarray(batch.fluxcalerrs)
    mfit = np.asarray(model_flux)
    mferr = np.asarray(model_flux_err)
    fidx = np.asarray(batch.filter_indices)
    mjd = np.asarray(batch.reference_mjds)[:, None] + \
        np.asarray(batch.mjd_offsets)
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2 = np.where(obs_ok, (flux - mfit) ** 2 / (err**2 + mferr**2),
                        0.0)
    is_sim = batch.sim_truth is not None
    simmag = ox.get('SIM_MAGOBS')
    band_chars = [filter_band(batch.filter_names[b])
                  for b in range(len(batch.filter_names))]

    rows = {c: [] for c in
            ['CID', 'MJD', 'Tobs', 'SIM_FLUXCAL', 'FLUXCAL', 'FLUXCAL_ERR',
             'DATAFLAG', 'BAND', 'CHI2', 'IFIT']}
    if not is_sim:
        del rows['SIM_FLUXCAL']

    def add(cid, mjd_v, tobs, simf, f, fe, flag, band, c2):
        rows['CID'].append(cid)
        rows['MJD'].append(mjd_v)
        rows['Tobs'].append(tobs)
        if is_sim:
            rows['SIM_FLUXCAL'].append(simf)
        rows['FLUXCAL'].append(f)
        rows['FLUXCAL_ERR'].append(fe)
        rows['DATAFLAG'].append(flag)
        rows['BAND'].append(band)
        rows['CHI2'].append(c2)
        rows['IFIT'].append(1)

    snids = np.asarray(batch.snids)
    for ip, i in enumerate(plot_idx):
        cid = snids[i]
        for b, ch in enumerate(band_chars):
            in_band = obs_ok[i] & (fidx[i] == b)
            # used epochs first, fit-rejected at the end of the band block
            for e in np.concatenate([np.nonzero(in_band & mask[i])[0],
                                     np.nonzero(in_band & ~mask[i])[0]]):
                simf = 0.0
                if simmag is not None and simmag[i, e] < SIM_MAGOBS_UNSET:
                    simf = 10.0 ** (-0.4 * (simmag[i, e] - batch.zp_fluxcal))
                add(cid, mjd[i, e], mjd[i, e] - pkmjd[i], simf,
                    flux[i, e], err[i, e], 1 if mask[i, e] else -1, ch,
                    chi2[i, e])
            # the fit curve only for a band the fit used an epoch of
            # (EXIST_FILT, snana.F90:17454, :22250)
            if not np.any(in_band & mask[i]):
                continue
            for g, tobs in enumerate(LCPLOT_TOBS_GRID):
                add(cid, pkmjd[i] + tobs, tobs, 0.0,
                    grid_flux[ip, b, g], grid_flux_err[ip, b, g], 0, ch, 0.0)

    columns = {c: np.array(v) for c, v in rows.items()}
    fmt = {'CID': '%s', 'MJD': '%.4f', 'Tobs': '%.2f',
           'SIM_FLUXCAL': '%.4e', 'FLUXCAL': '%.4e', 'FLUXCAL_ERR': '%.4e',
           'DATAFLAG': '%d', 'BAND': '%s', 'CHI2': '%.2f', 'IFIT': '%d'}
    return list(columns), columns, fmt


def _host_confusion(nmatch2, ddlrs, objids):
    """Gupta 2016 Eq 3 over the stored host slots (host_confusion,
    sntools.c:4256; SET_SNHOST_CONFUSION, snana.F90:18134): the first
    NMATCH2 slots, capped at the slots the file holds, count only where the
    slot carries an OBJID, and a third slot without one drops the count to
    two. ``ddlrs`` and ``objids`` are [n_sne, n_slots]."""
    eps = 1.0e-5
    d = np.asarray(ddlrs, dtype=float)
    objid = np.asarray(objids)
    n_slots = d.shape[1]
    n = np.minimum(np.asarray(nmatch2), n_slots)
    if n_slots > 2:
        n = np.where((n > 2) & (objid[:, 2] == 0), 2, n)
    defined = (np.arange(n_slots)[None, :] < n[:, None]) & (objid != 0)
    n_ddlr = defined.sum(axis=1)
    # the defined DDLRs, packed to the front as the Fortran list is filled
    order = np.argsort(~defined, axis=1, kind='stable')
    d = np.take_along_axis(d, order, axis=1)
    d1, d2 = d[:, 0], d[:, 1]
    with np.errstate(invalid='ignore', divide='ignore'):
        prefac = (d1 * d1 / d2 + eps) / (d2 - d1 + eps)
        tmp = np.zeros(len(d))
        for i in range(1, n_slots):
            for j in range(i + 1, n_slots + 1):
                di, dj = d[:, i - 1], d[:, j - 1]
                term = ((di / dj) + eps) / (i * i * (dj - di + eps))
                tmp = tmp + np.where(j <= n_ddlr, term, 0.0)
        hc = np.log10(prefac * tmp)
    hc = np.where(d2 <= 1.0e-6, 9.0, hc)
    return np.where(n_ddlr <= 1, SNANA_NULLFLOAT, hc)


def _slot_quantile_stats(host_data, prefix, n, quantile_spline):
    """A host slot's quantile mean and width from its HOSTGALz vector, the
    null where the file carries none."""
    if f'{prefix}_NBIN_QUANTILE_ZPHOT' not in host_data:
        return np.full(n, SNANA_NULLFLOAT), np.full(n, SNANA_NULLFLOAT)
    n_q = int(np.asarray(host_data[f'{prefix}_NBIN_QUANTILE_ZPHOT']).max())
    return _quantile_stats(
        np.asarray(host_data[f'{prefix}_QUANTILE_ZPHOT'])[:, :n_q],
        np.asarray(host_data[f'{prefix}_QUANTILE_PERCENT'])[:, :n_q],
        quantile_spline, np.asarray(host_data[f'{prefix}_NBIN_QUANTILE_ZPHOT']))


def _quantile_stats(quantiles, percentiles, quantile_spline, n_quantiles=None):
    """The mean and width of each event's splined quantile PDF, -9 for an
    event the spline cannot take (SNHOST_QZPHOT_MEAN/STD, SET_SNHOST_QZPHOT)."""
    usable = zpdf_rows_usable(quantiles, n_quantiles)
    mean = np.full(len(quantiles), SNANA_NULLFLOAT)
    std = np.full(len(quantiles), SNANA_NULLFLOAT)
    if usable.any():
        sp = precompute_zpdf_spline(quantiles[usable], percentiles[usable] / 100.0,
                                    spline=quantile_spline,
                                    n_quantiles=(None if n_quantiles is None else
                                                 np.asarray(n_quantiles)[usable]))
        mean[usable] = sp['pdf_mean']
        std[usable] = sp['pdf_std']
    return mean, std


def build_host_columns(host_data, zp_fluxcal, host_text=False, host2_text=False,
                       zpdf_quantiles=None, zpdf_percentiles=None, *, zpdf_nbin=None,
                       quantile_spline):
    """FITRES HOST_* columns from raw HOSTGAL* HEAD columns, in SNANA column
    order (INIT_TABLE_HOSTVAR, snana.F90:23778). Identity columns need
    text:host; properties and mags of the best match are always written;
    the quantile mean and width read the ``quantile_spline`` the fit does."""
    identity_on = host_text or host2_text  # snana.F90:23440
    out = {}

    def copy(raw, name):
        if host_data.get(raw) is not None:
            out[name] = host_data[raw]

    def mags(raw_pre, out_pre):
        for kind in ('MAG_', 'MAGERR_'):
            for k in host_data:
                if k.startswith(raw_pre + kind):
                    copy(k, out_pre + k[len(raw_pre):])

    if identity_on:
        copy('HOSTGAL_NMATCH', 'HOST_NMATCH')
        copy('HOSTGAL_NMATCH2', 'HOST_NMATCH2')
        copy('HOSTGAL_OBJID', 'HOST_OBJID')
        copy('HOSTGAL_PHOTOZ', 'HOST_ZPHOT')
        copy('HOSTGAL_PHOTOZ_ERR', 'HOST_ZPHOTERR')
        if zpdf_quantiles is not None:
            out['HOST_QZPHOT'], out['HOST_QZPHOTSTD'] = _quantile_stats(
                np.asarray(zpdf_quantiles), np.asarray(zpdf_percentiles),
                quantile_spline, zpdf_nbin)
        copy('HOSTGAL_SPECZ', 'HOST_ZSPEC')
        copy('HOSTGAL_SPECZ_ERR', 'HOST_ZSPECERR')
        copy('HOSTGAL_RA', 'HOST_RA')
        copy('HOSTGAL_DEC', 'HOST_DEC')
        copy('HOSTGAL_SNSEP', 'HOST_ANGSEP')
        copy('HOSTGAL_DDLR', 'HOST_DDLR')
        nm2 = host_data.get('HOSTGAL_NMATCH2')
        slots = [p for p in HOSTGAL_PREFIXES if f'{p}_DDLR' in host_data]
        if nm2 is not None and len(slots) > 1:
            # recomputed, not copied (SET_SNHOST_CONFUSION)
            out['HOST_CONFUSION'] = _host_confusion(
                nm2,
                np.stack([host_data[f'{p}_DDLR'] for p in slots], axis=1),
                np.stack([host_data[f'{p}_OBJID'] for p in slots], axis=1))
    def properties(raw_pre, out_pre):
        for sfx in _HOST_PROPERTY_COLS:
            copy(f'{raw_pre}{sfx}', f'{out_pre}{sfx}')
            if sfx == 'LOGMASS_ERR' and f'{out_pre}LOGMASS_ERR' in out:
                # SNANA declares these and never sets them, so they hold
                # the null for every event (snana.F90:15076, :24149)
                n = len(out[f'{out_pre}LOGMASS_ERR'])
                out[f'{out_pre}LOGMASS_ZPHOT'] = np.full(n, HOSTGAL_PROPERTY_NULL)
                out[f'{out_pre}LOGMASS_ZPHOTERR'] = np.full(n, HOSTGAL_PROPERTY_NULL)

    properties('HOSTGAL_', 'HOST_')
    mags('HOSTGAL_', 'HOST_')

    if host2_text:
        copy('HOSTGAL2_OBJID', 'HOST2_OBJID')
        copy('HOSTGAL2_PHOTOZ', 'HOST2_ZPHOT')
        copy('HOSTGAL2_PHOTOZ_ERR', 'HOST2_ZPHOTERR')
        if zpdf_quantiles is not None:
            # null when the file has no second-host quantiles
            out['HOST2_QZPHOT'], out['HOST2_QZPHOTSTD'] = _slot_quantile_stats(
                host_data, 'HOSTGAL2z', len(np.asarray(zpdf_quantiles)),
                quantile_spline)
        copy('HOSTGAL2_SPECZ', 'HOST2_ZSPEC')
        copy('HOSTGAL2_SPECZ_ERR', 'HOST2_ZSPECERR')
        copy('HOSTGAL2_RA', 'HOST2_RA')
        copy('HOSTGAL2_DEC', 'HOST2_DEC')
        copy('HOSTGAL2_SNSEP', 'HOST2_ANGSEP')
        copy('HOSTGAL2_DDLR', 'HOST2_DDLR')
        properties('HOSTGAL2_', 'HOST2_')
        mags('HOSTGAL2_', 'HOST2_')

    if identity_on:
        sb_cols = [k for k in host_data
                   if k.startswith('HOSTGAL_SB_FLUXCAL_')]
        for k in sb_cols:
            out[f'HOST_SBFLUXCAL_{k[len("HOSTGAL_SB_FLUXCAL_"):]}'] = \
                host_data[k]
        for k in sb_cols:
            out[f'HOST_SBMAG_{k[len("HOSTGAL_SB_FLUXCAL_"):]}'] = sbmag_from_fluxcal(
                np.asarray(host_data[k], dtype=float), zp_fluxcal)
    return out


def build_fitres_table(results, redshifts, mu_ref, model=None, sim_truth=None,
                       event_columns=None, event_fmt=None,
                       opt_photoz=0, version_photometry=None,
                       sim_cosmology=None,
                       physical_only=False, ideal_columns=None):
    """The FITRES table, its column formats and its header comments.

    The table is a :class:`FitresTable`, what reading the written file gives
    back, so a caller that goes on to use it needs no round trip. The writer
    takes the columns off ``table.df``, as its sibling builders pass them.

    Column layout: CID, the per-SN event/host block (or bare zHEL when
    event_columns is None), the model's FITRES parameter columns with
    errors, covariance terms, [zPRIOR/zPRIORERR], NDOF, FITCHI2, FITPROB,
    [``SIM_`` truth block].

    ``redshifts`` are heliocentric, one per SN. ``OPT_PHOTOZ`` is always a
    column, and a photo-z fit writes no zPHOT/zPHOTERR: the fitted redshift
    reaches the table as zHD (SKIPTABLE_FITPAR, snlc_fit.F90:19042).
    ``event_fmt`` gives formats
    for ``event_columns``; a column left out takes the writer's per-value
    float rule, except the host NMATCH and OBJID columns, written as ``%d``.
    ``ideal_columns`` are SIMFIT_IDEAL_PRESCALE's ``<param>_IDEAL`` columns.
    ``sim_cosmology`` is the simulated data's generating cosmology
    (:mod:`jax_snlc_core.io.sim_cosmology`), written as ``SIM_<key>`` header
    lines.
    """
    param_names = results.param_names

    if model is not None:
        param_cols = model.get_fitres_param_columns(
            results.params, results.param_errors, redshifts, mu_ref,
            physical_only=physical_only)
    else:
        param_cols = [(n, results.params[n], results.param_errors[n])
                      for n in param_names]
    # the fitted redshift reaches the table as zHD (SKIPTABLE_FITPAR, snlc_fit.F90:19042)
    param_cols = [pc for pc in param_cols if pc[0] != 'zPHOT']

    varnames = ["CID"]
    varnames.extend(event_columns.keys() if event_columns else ["zHEL"])
    varnames.append("OPT_PHOTOZ")
    # SALT2mu spells the distance uncertainty MUERR_LCFIT.
    for colname, _, _ in param_cols:
        varnames.append(colname)
        varnames.append("MUERR_LCFIT" if colname == "MU_LCFIT" else f"{colname}ERR")
    # SIMFIT_IDEAL_PRESCALE mirrors the parameter block, values only
    # (snlc_fit.F90:18991)
    if ideal_columns:
        varnames.extend(ideal_columns.keys())
    cov_columns = []
    cov_indices = [param_names.index(p) for p in results.cov_output_params
                   if p in param_names]
    for ii, i in enumerate(cov_indices):
        for j in cov_indices[ii + 1:]:
            col = f"COV_{param_names[i]}_{param_names[j]}"
            varnames.append(col)
            cov_columns.append((i, j, col))
    if results.z_prior is not None:
        varnames.extend(["zPRIOR", "zPRIORERR"])

    varnames.extend(["NDOF", "FITCHI2", "FITPROB", "HESSIAN_POSDEF"])
    derived_cols = (model.get_fitres_derived_columns(results.params)
                    if model is not None else [])
    varnames.extend(name for name, _ in derived_cols)

    sim_col_names = []
    if sim_truth is not None:
        sim_col_names = list(sim_truth.keys())
        varnames.extend(sim_col_names)

    fitprob_arr = model.compute_fitprob(
        np.asarray(results.chi_squared), np.asarray(results.fitprob_ndof))

    columns = {'CID': np.asarray(results.snids)}
    if event_columns:
        for col, arr in event_columns.items():
            columns[col] = np.asarray(arr)
    else:
        columns['zHEL'] = np.asarray(redshifts)

    for colname, vals, errs in param_cols:
        columns[colname] = np.asarray(vals)
        errname = "MUERR_LCFIT" if colname == "MU_LCFIT" else f"{colname}ERR"
        columns[errname] = np.asarray(errs)
    for colname, vals in (ideal_columns or {}).items():
        columns[colname] = np.asarray(vals)
    columns["OPT_PHOTOZ"] = np.full(results.n_sne, opt_photoz, dtype=int)

    cov_np = np.asarray(results.param_covariances)  # [n_sne, n_params, n_params]
    for i, j, col in cov_columns:
        columns[col] = cov_np[:, i, j]

    if results.z_prior is not None:
        columns['zPRIOR'], columns['zPRIORERR'] = map(np.asarray, results.z_prior)

    columns['NDOF'] = np.asarray(results.ndof).astype(int)
    columns['FITCHI2'] = np.asarray(results.chi_squared)
    columns['FITPROB'] = np.asarray(fitprob_arr)
    columns['HESSIAN_POSDEF'] = np.asarray(results.hessian_positive, dtype=int)
    for name, vals in derived_cols:
        columns[name] = np.asarray(vals)

    for col in sim_col_names:
        columns[col] = np.asarray(sim_truth[col])
    # Profile distances use native fitted parameters, including x0 and zPHOT.
    # Standard errors already use the profile mean or flagged Hessian fallback.
    if results.minos_errors is not None:
        for name in results.param_names:
            flag_col = f"ERRFLAG_{name}"
            varnames.append(flag_col)
            if results.error_flags is not None:
                flags = results.error_flags[name]
            else:
                flags = np.where(
                    np.isfinite(results.param_errors[name]),
                    np.asarray(profile_error_type(
                        results.minos_errors[name], results.minos_status[name])), 6)
            columns[flag_col] = np.asarray(flags)
            for side, label in enumerate(('LO', 'HI')):
                error_col = f"{name}_MINOS_{label}"
                status_col = f"{name}_MINOS_STATUS_{label}"
                varnames.extend((error_col, status_col))
                columns[error_col] = np.asarray(results.minos_errors[name])[:, side]
                columns[status_col] = np.asarray(results.minos_status[name])[:, side]
    df = pd.DataFrame(columns)

    # floats use the writer's per-value rule (sntools_output_text.c:445)
    fmt = {'CID': '%s', 'NDOF': '%d'}
    if event_fmt:
        fmt.update(event_fmt)
    for col in event_columns or ():
        if col.split('_', 1)[-1] in ('NMATCH', 'NMATCH2', 'OBJID'):
            fmt.setdefault(col, '%d')
    fmt['OPT_PHOTOZ'] = '%d'
    if results.minos_errors is not None:
        for name in results.param_names:
            fmt[f'ERRFLAG_{name}'] = '%d'
            for side in ('LO', 'HI'):
                fmt[f'{name}_MINOS_STATUS_{side}'] = '%d'
    for col in sim_col_names:
        if np.issubdtype(np.asarray(sim_truth[col]).dtype, np.integer):
            fmt[col] = '%d'
    for col in varnames:
        fmt.setdefault(col, None)

    comments = []
    if version_photometry:
        comments.append(f"VERSION_PHOTOMETRY: {version_photometry}")
    comments += [f"SIM_{key}: {value}" for key, value in (sim_cosmology or {}).items()]
    comments.append(snana_version_comment())
    comments.append(program_comment("snlc_fit.py"))
    comments.append(jax_version_comment())
    comments.append("TABLE_NAME:    FITRES")
    if results.minos_errors is not None:
        comments.append("ERROR_SOURCE: ERRFLAG_<parameter>: 1=MINOS mean, 2=Hessian, 6=bad error")
        comments.append("MINOS_STATUS: 0=crossing, 1=bound, 2=failed minimisation, "
                        "3=lower minimum, 4=search exhausted, 5=not profiled")
    meta = {k.strip(): v.strip() for k, v in
            (c.split(':', 1) for c in comments if ':' in c)}
    # native byte order
    cols = {c: df[c].values for c in varnames}
    return FitresTable(df=pd.DataFrame(cols), varnames=list(varnames),
                       meta=meta, varnames_by_file=[list(varnames)]), fmt, comments
