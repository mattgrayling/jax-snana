"""
Light curve data loading utilities for fitting.

This module provides data structures and loaders for supernova light curves,
converting from SNANA FITS format to JAX-friendly batched arrays suitable
for GPU-accelerated fitting.

Entry points:
    - load_and_batch_snana_fits: a HEAD/PHOT FITS pair
    - load_and_batch_snana_text: a TEXT data version
    - read_rubin_hats: a Rubin HATS catalogue of DiaObjects
"""

import logging
import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads

from astropy.table import Table

from jax_snlc_core.io.docana import skip_docana
from jax_snlc_core.io.fits_path import read_fits_table, resolve_fits_path
from jax_snlc_core.photometry.noise import (ZEROPOINT_FLUXCAL_SNANA_ORIG, apply_fluxcal_fudges,
                                            fluxcal_errcalc,
                                            noise_equivalent_aperture)
from jax_snlc_core.photometry.filter_system import filter_band, filter_values
from jax_snlc_core.photometry.peakmjd import (PEAKMJD_MISSING,
                                              fmax_clump_peakmjd,
                                              naive_fluxmax_peakmjd)
from jax_snlc_core.utils import (HOSTGAL_PREFIXES, MXBIN_HOSTGALz,
                                 MXBIN_HOSTGALz_QUANTILE, SNANA_NULLFLOAT,
                                 ZSPEC_SOURCE_MASK)
from jax_snlc_core.utils import snana_log as slog
from jax_snlc_core.utils.error_handling import SNANAError

from .batch import (ERRFLAG_NO_FIT, LightCurveBatch, choose_epoch_lengths,
                    group_by_epoch_length, trest_z1)
from .bazin_prefit import bazin_prefit
from .host_noise import hostnoise_errscale
from .loss_functions import (natural_spline_slopes, precompute_zpdf_spline,
                             spline_knots, zpdf_rows_usable)
from .nonlinearity import evaluate_nonlinearity, OPTMASK_COUNT_RATE
from .survey import survey_fields
from jax_snlc_core.physics.cosmology import zcmb_to_zhelio, zhelio_to_zcmb
from jax_snlc_core.physics.dust_laws import (OPT_MWCOLORLAW_DEFAULT,
                                             OPT_MWCOLORLAW_NAMES,
                                             RV_MWCOLORLAW_DEFAULT)
from jax_snlc_core.utils.cutwin import CUTVAL_OPEN, in_window
from jax_snlc_core.photometry.fluxerr_model import evaluate_fluxerrmodel_data

logger = logging.getLogger(__name__)

# the Milky Way colour law and R_V a simulation records in its header
SIM_MW_COLOUR_LAW_KEYS = ('SIMOPT_MWCOLORLAW', 'SIM_MWRV')


def snana_version_float(text: str) -> float:
    """'v10_78c' -> 10.78 (get_SNANA_VERSION_FLOAT, sntools.c:3296)."""
    text = str(text).split('-')[0]

    def atof(part):
        try:
            return float(re.match(r"[-+]?\d*\.?\d*", part).group())
        except ValueError:
            return 0.0

    return atof(text[1:]) + atof(text[4:]) / 100.0


def vpec_sign_is_correct(version: str) -> bool:
    """Whether the file's VPEC uses the current sign convention; an unknown
    version counts as correct (correct_sign_vpec_data, sntools.c:3328)."""
    if not version or str(version).strip().upper() == "UNKNOWN":
        return True
    return snana_version_float(version) >= 11.02


# SNLC8_MJD_TRIGGER before any HEAD column is read (snana.F90:15144)
MJD_TRIGGER_MISSING = -99.0

# PKMJD carried by an event the Bazin fit could not place (snana.F90:26654)
PKMJD_NOFIT = -1.0
# OPT_SETPKMJD bit 2: do not abort on that (snana.F90:2124)
OPTMASK_SETPKMJD_NOABORT = 4
# largest EPCUT_SNRMIN floor that still reads as a cut (snana.F90:13506)
EPCUT_SNRMIN_OFF = 99.9999
# PSF sigma to FWHM (SET_EPVAR_MISC, snana.F90:13963)
FWHM_PER_SIGMA = 2.355

# the cuts SELECT_EPOCH_DRIVER(2) applies after SET_PEAKMJD (snana.F90:17221)
_POST_EPOCH_CUTS = {'tobs', 'trest', 'lamobs', 'lamrest'}

# the CUTWIN epoch cuts applied as the light curve is read (PASS_EPOCH_CUTS,
# snana.F90:13243); the rest wait for LSIG
_READ_EPOCH_CUTS = {'mjd', 'mjd_exclude', 'zpadu', 'zpnpe', 'psf',
                    'snr_nodetect'}

# RDOBS_DRIVER's MJD_SAFETEY around the read window (snana.F90:4858)
_READ_MARGIN_DAYS = 10.0


# INIT_TABLE_SIMVAR list (snana.F90:24214); absent columns skipped
FITRES_SIM_COLUMNS = [
    'SIM_GENTYPE', 'SIM_TEMPLATE_INDEX', 'SIM_LIBID',
    'SIM_NGEN_LIBID', 'SIM_SUBSAMPLE_INDEX',
    'SIM_REDSHIFT_CMB', 'SIM_REDSHIFT_HELIO',
    'SIM_REDSHIFT_HOST_MATCH', 'SIM_REDSHIFT_FLAG', 'SIM_VPEC',
    'SIM_DLMU', 'SIM_LENSDMU',
    'SIM_PEAKMJD', 'SIM_MWEBV', 'SIM_MAGSMEAR_COH',
    'SIM_WGT_POPULATION',
    'SIM_HOSTLIB_GALID',
    'SIM_AV', 'SIM_RV',
]


def model_sim_columns(model) -> tuple:
    """The ``SIM_`` columns the loader carries for the fitted model: every
    ``sim_truth`` its fit spec names, and the ``SIM_TRUTH_COLUMNS`` it
    declares beyond those."""
    named = [spec['sim_truth'] for spec in model.get_fit_parameters().values()
             if 'sim_truth' in spec]
    return tuple(dict.fromkeys(named + list(model.SIM_TRUTH_COLUMNS)))


ZSHIFT_MATCH_TOL = 1.0e-5   # SNANA's round-off tolerance (snana.F90:17862)


def apply_redshift_shifts(shifts, redshifts, head_extras, photoz,
                          zpdf_quantiles, specz, zcmb):
    """SNANA SET_zSHIFT (snana.F90:17848): systematic shifts on the redshifts.

    REDSHIFT_FINAL_SHIFT moves every redshift. The host shifts apply only when
    that host measurement *is* the SN redshift, within a round-off tolerance:
    shifting a host-z that was not used would change nothing about the fit while
    silently corrupting the reported host column.
    """
    final = shifts.get('final', 0.0)
    redshifts = np.asarray(redshifts)
    zshift = np.full(len(redshifts), float(final))
    shifted = np.full(len(redshifts), bool(final))

    if shifts.get('zspec') and specz is not None:
        is_the_redshift = np.abs(specz - redshifts) < ZSHIFT_MATCH_TOL
        zshift = np.where(is_the_redshift, shifts['zspec'] + final, zshift)
        shifted |= is_the_redshift

    if shifts.get('zphot') and photoz is not None:
        photoz_np = np.asarray(photoz)
        is_the_redshift = np.abs(photoz_np - redshifts) < ZSHIFT_MATCH_TOL
        zshift = np.where(is_the_redshift, shifts['zphot'] + final, zshift)
        shifted |= is_the_redshift
        photoz = np.where(is_the_redshift, photoz_np + zshift, photoz_np)
        if zpdf_quantiles is not None:
            zpdf_quantiles = np.where(is_the_redshift[:, None],
                                      zpdf_quantiles + zshift[:, None],
                                      zpdf_quantiles)

    if shifted.any():
        redshifts = np.where(shifted & (redshifts > 0.0),
                             redshifts + zshift, redshifts)
        # SNANA shifts SNLC_REDSHIFT, SNLC_ZHELIO and SNLC_ZCMB alike
        zcmb = np.where(shifted & (zcmb > 0.0), zcmb + zshift, zcmb)
        for column in ('REDSHIFT_HELIO', 'REDSHIFT_FINAL'):
            if column in head_extras:
                z = np.asarray(head_extras[column], dtype=float)
                head_extras[column] = np.where(shifted & (z > 0.0),
                                               z + zshift, z)
    return redshifts, photoz, zpdf_quantiles, zcmb


# A correction beyond this is a mistake, not a correction (snana.F90:16898).
MAGCOR_MAX = 0.2


def read_magcor_file(path, version):
    """SNANA MAGCOR_FILE (INIT_MAGCOR): FITRES-format rows keyed
    ``[CID]-[MJD]-[BAND]`` with a MAGCOR column; other columns ignored, bar
    WAVECOR, the per-epoch wavelength shift, which is refused. A leading '-'
    on the path subtracts MAGCOR instead of adding it. VERSION_ADD
    rows name the data versions already carrying the correction: adding to
    one of them, or subtracting from any other, is refused (ISTAT_MAGCOR_ADD,
    snana.F90:16750-16760). Returns {(cid, round(mjd, 3), band): signed_magcor}."""
    sign = 1.0
    if path.startswith('-'):
        sign, path = -1.0, path[1:]
    varnames, out, added = None, {}, set()
    with open(os.path.expandvars(path)) as fh:
        for line in fh:
            toks = line.split('#')[0].split()
            if not toks:
                continue
            if toks[0] == 'VARNAMES:':
                varnames = toks[1:]
                if 'WAVECOR' in varnames:
                    raise SNANAError(
                        f"MAGCOR_FILE {path} carries a WAVECOR column, the "
                        f"per-epoch wavelength shift, which jax does not "
                        f"implement.", function_name='read_magcor_file')
            elif toks[0] == 'VERSION_ADD:':
                added.add(toks[1])
            elif toks[0] in ('ROW:', 'SN:'):
                cid, mjd, band = toks[1].rsplit('-', 2)
                mag = float(toks[1 + varnames.index('MAGCOR')])
                out[(cid, round(float(mjd), 3), band)] = sign * mag
    if sign > 0.0 and version in added:
        raise SNANAError(
            f"MAGCOR already in the {version} fluxes: will not add {path} twice",
            function_name='read_magcor_file')
    if sign < 0.0 and version not in added:
        raise SNANAError(
            f"MAGCOR not in the {version} fluxes: will not subtract {path}",
            function_name='read_magcor_file')
    return out


def _magcor_values(corrections, cids, mjds, bands):
    """Match correction keys in bulk, preserving Python's timestamp rounding."""
    rounded = np.fromiter((round(float(mjd), 3) for mjd in mjds),
                          dtype=float, count=len(mjds))
    values = pd.Series(list(corrections.values()),
                       index=pd.MultiIndex.from_tuples(corrections))
    epochs = pd.MultiIndex.from_arrays([cids, rounded, bands])
    return values.reindex(epochs, fill_value=0.0).to_numpy()


# same night within 0.4 day (DT_SAMENIGHT, snana.F90:528)
_DT_SAMENIGHT = 0.400


def _earlylc_mask(cfg, sn_idx, n_sne, mjds, bands, fluxcal, fluxcalerr,
                  phot_data, phot_indices):
    """EARLYLC_STRING: keep epochs until MAXOBS obs or MAXNIGHT nights pass.

    SELECT_EARLYLC (snana.F90:16050) walks each SN's epochs in file order,
    carrying counters that no vectorised form reproduces, so this is a per-SN
    scan. It is called for every epoch regardless of earlier rejections, so the
    counters here run over the full epoch list too.
    """
    photmask = int(cfg["photmask"])
    photmask_start = int(cfg["photmask_start"])
    need_flag = photmask or photmask_start
    if need_flag and 'PHOTFLAG' not in phot_data.colnames:
        raise SNANAError(
            "EARLYLC_STRING PHOTMASK configured but PHOT file has no "
            "PHOTFLAG column", function_name="_earlylc_mask")
    flags = (np.array(phot_data['PHOTFLAG'], dtype=np.int64)[phot_indices]
             if need_flag else np.zeros(len(mjds), dtype=np.int64))
    if 'PHOTPROB' in phot_data.colnames:
        probs = np.array(phot_data['PHOTPROB'], dtype=np.float64)[phot_indices]
    elif cfg["photprobmin"] > -900.0:
        raise SNANAError(
            "EARLYLC_STRING PHOTPROBMIN configured but PHOT file has no "
            "PHOTPROB column", function_name="_earlylc_mask")
    else:
        probs = np.full(len(mjds), 9.0)

    short = np.array([filter_band(b) for b in bands])
    in_filt = (np.ones(len(mjds), dtype=bool) if cfg["filters"] is None
               else np.isin(short, list(cfg["filters"])))
    snr = np.divide(fluxcal, fluxcalerr, out=np.zeros_like(fluxcal),
                    where=fluxcalerr > 0)

    keep = np.ones(len(mjds), dtype=bool)
    order = np.argsort(sn_idx, kind='stable')
    starts = np.searchsorted(sn_idx[order], np.arange(n_sne))
    ends = np.searchsorted(sn_idx[order], np.arange(n_sne), side='right')
    for lo, hi in zip(starts, ends):
        n_obs = n_night = n_snr_start = n_mask_start = 0
        mjd_last = mjd_last_select = -999.0
        for ep in order[lo:hi]:
            if fluxcalerr[ep] > 0 and snr[ep] > cfg["snr_start"]:
                n_snr_start += 1
            if not photmask_start or (flags[ep] & photmask_start):
                n_mask_start += 1
            passes = (in_filt[ep]
                      and snr[ep] >= cfg["snrmin"] and n_snr_start > 0
                      and probs[ep] >= cfg["photprobmin"]
                      and (not photmask or (flags[ep] & photmask))
                      and n_mask_start > 0)
            if passes:
                n_obs += 1
                if (mjds[ep] - mjd_last) >= _DT_SAMENIGHT:
                    n_night += 1
                    mjd_last = mjds[ep]
            select = n_obs <= cfg["maxobs"] and n_night <= cfg["maxnight"]
            if select:
                mjd_last_select = mjds[ep]
            elif cfg["ndayadd"] > 0 and mjd_last_select > 10000:
                select = (int(mjds[ep] - mjd_last_select + 0.5)
                          < cfg["ndayadd"])
            keep[ep] = select
    return keep


def check_zphot_quantiles(quantiles, abort: bool) -> None:
    """SET_SNHOST_QZPHOT (snana.F90:17734) requires the quantiles to increase,
    skipping entries left at their null marker. Runs on every event read, so a
    malformed host is caught whether or not the event survives the cuts."""
    q = np.asarray(quantiles)
    bad = ~np.all((np.diff(q, axis=1) > 0) | (q[:, 1:] <= -8.0), axis=1)
    if not bad.any():
        return
    where = int(np.flatnonzero(bad)[0])
    message = (f"ZPHOT_Q must be monotonically increasing; {int(bad.sum())} "
               f"host(s) are not, first at row {where}: {q[where]}")
    if abort:
        raise SNANAError(message, function_name="SET_SNHOST_QZPHOT")
    slog.warn(logger, "SNHOST_QZPHOT", message)


def _head_metadata(head_data):
    """Per-SN HEAD scalars, with SNANA's column fallbacks."""
    def get_column(table, names, default=0.0):
        for name in names:
            if name in table.colnames:
                return np.array(table[name], dtype=np.float64)
        return np.full(len(table), default, dtype=np.float64)

    mwebvs = get_column(head_data, ['MWEBV'], 0.0)
    # MWEBV_ERR defaults to XTMW_FRACERR = 16% of MWEBV; the header column
    # overrides that only above 1e-12 (snana.F90:3908-3916).
    mwebv_errs = get_column(head_data, ['MWEBV_ERR'], 0.0)
    peakmjds = get_column(head_data, ['PEAKMJD', 'SIM_PEAKMJD'], np.nan)
    # the fit uses zHELIO where defined, else zCMB (snana.F90:4256, :4292),
    # and zCMB itself falls back to REDSHIFT_CMB where undefined (:4277)
    zhel = get_column(head_data, ['REDSHIFT_HELIO'], SNANA_NULLFLOAT)
    zcmb = get_column(head_data, ['REDSHIFT_FINAL'], SNANA_NULLFLOAT)
    zcmb_err = get_column(head_data, ['REDSHIFT_FINAL_ERR'], 0.001)
    alt = zcmb < 0.0
    zcmb = np.where(alt, get_column(head_data, ['REDSHIFT_CMB'], SNANA_NULLFLOAT), zcmb)
    zcmb_err = np.where(
        alt, get_column(head_data, ['REDSHIFT_CMB_ERR'], 0.001), zcmb_err)
    ras = get_column(head_data, ['RA'], np.nan)
    decs = get_column(head_data, ['DEC', 'DECL'], np.nan)
    # SET_ZHELIO converts zCMB back to helio (snana.F90:17687-17697)
    use_hel = zhel >= 0.0
    # SNLC_REDSHIFT at read time (snana.F90:4496, :4533)
    read_redshifts = np.where(use_hel, zhel, zcmb)
    # no sky position: keep zCMB
    on_sky = ~(np.isnan(ras) | np.isnan(decs))
    if not use_hel.all():
        converted = np.asarray(zcmb_to_zhelio(zcmb, ras, decs))
        zhel = np.where(use_hel, zhel, np.where(on_sky, converted, zcmb))
    return {
        'redshifts': zhel,
        'redshift_errs': np.where(
            use_hel, get_column(head_data, ['REDSHIFT_HELIO_ERR'], 0.001),
            zcmb_err),
        'mwebvs': mwebvs,
        'mwebv_errs': np.where(mwebv_errs >= 1.0e-12, mwebv_errs,
                               0.16 * mwebvs),
        'ras': ras,
        'decs': decs,
        'read_redshifts': read_redshifts,
        'peakmjds': peakmjds,
        'has_peakmjd': not np.all(np.isnan(peakmjds)),
        'snids': np.char.strip(np.array(head_data['SNID'], dtype='U')),
    }


def _sim_truth_columns(head_data, is_sim, model_columns):
    """SIM_ truth columns carried through to FITRES, or None for real data."""
    if not is_sim:
        return None
    out = {col: np.array(head_data[col])
           for col in (*FITRES_SIM_COLUMNS, *model_columns)
           if col in head_data.colnames}
    absent = [col for col in model_columns if col not in head_data.colnames]
    if absent:
        slog.warn(logger, "SIM_TRUTH",
                  f"the HEAD file has no {', '.join(absent)}; the FITRES "
                  f"carries no truth for those parameters")
    for col in head_data.colnames:
        # HOSTLIB params are stored as SIM_HOSTLIB(<par>)
        m = re.fullmatch(r'SIM_HOSTLIB\((\w+)\)', col)
        if m:
            out[f'SIM_HOSTLIB_{m.group(1)}'] = np.array(head_data[col])
    return out


def _label_codes(column, rows):
    """The distinct labels of a string column at ``rows``, stripped, and each
    row's index into them."""
    values = np.ascontiguousarray(np.asarray(column)[rows])
    # FITS labels are fixed-width bytes; Arrow can hash their buffer directly.
    labels = (pa.Array.from_buffers(pa.binary(values.dtype.itemsize), len(values),
                                    [None, pa.py_buffer(values)])
              if values.dtype.kind == 'S' else pa.array(values))
    encoded = labels.dictionary_encode()
    labels = np.asarray(encoded.dictionary.to_numpy(zero_copy_only=False),
                        dtype=values.dtype)
    order = np.argsort(labels)
    inverse = np.empty(len(order), dtype=np.intp)
    inverse[order] = np.arange(len(order))
    codes = inverse[encoded.indices.to_numpy()]
    return np.char.strip(labels[order].astype('U')), codes


def _event_fields(unique_fields, field_codes, sn_idx_all, n_sne, survey):
    """Each event's FIELDLIST: the SURVEY.DEF fields present among its
    epochs, '+'-joined in SURVEY.DEF order (snana.F90:18885)."""
    present = np.zeros((n_sne, len(unique_fields)), dtype=bool)
    present[sn_idx_all, field_codes] = True
    code = {f: c for c, f in enumerate(unique_fields)}
    label = np.full(n_sne, '', dtype='U1')
    for f in survey_fields(survey):
        if f in code:
            hit = present[:, code[f]]
            sep = np.where(hit & (label != ''), '+', '')
            label = np.char.add(np.char.add(label, sep), np.where(hit, f, ''))
    return np.where(label == '', 'VOID', label)


def _head_extras(head_data, private_varnames, event_fields, peakmjds):
    """Per-SN HEAD scalars the FITRES event columns and the SN cuts read.

    ``event_fields`` stands in for a FIELD column the HEAD lacks, None
    without a PHOT FIELD column; ``peakmjds`` is None when no header peak
    was found."""
    out = {
        col: np.array(head_data[col])
        for col in ('SNTYPE', 'FIELD', 'NAME_TRANSIENT', 'PIXSIZE', 'SUBSURVEY',
                    'VPEC', 'VPEC_ERR', 'LENSDMU', 'LENSDMU_ERR',
                    'REDSHIFT_HELIO', 'REDSHIFT_HELIO_ERR',
                    # cut quantities only, not FITRES output
                    'SIM_SEARCHEFF_MASK', 'SIM_MODEL_INDEX')
        if col in head_data.colnames
    }
    # IAUC before CODE_IVERSION 25 (sntools_dataformat_fits.c:3511)
    for col in ('NAME_IAUC', 'IAUC'):
        if col in head_data.colnames:
            out['NAME_IAUC'] = np.array(head_data[col])
            break
    for col in private_varnames:
        if col not in head_data.colnames:
            raise SNANAError(
                f"PRIVATE_VARNAME_READLIST names {col}, absent from the "
                f"HEAD file", function_name="_head_extras")
        out[col] = np.array(head_data[col])

    if 'FIELD' not in out and event_fields is not None:
        out['FIELD'] = event_fields
    if peakmjds is not None:
        out['PEAKMJD'] = peakmjds
    return out


def _apply_vpec_corrections(batch_head_extras, restore_wrong_vpec,
                            snana_version, opt_vpec_cor, vpec_err_override):
    """The three peculiar-velocity corrections SNANA applies to the HEAD
    columns, in place."""
    # VPEC was written with the opposite sign below SNANA v11.02
    # (DOFIX_WRONG_SIGN_VPEC, snana.F90:8475).
    if ('VPEC' in batch_head_extras and not restore_wrong_vpec
            and not vpec_sign_is_correct(snana_version)):
        slog.line(logger, f"VPEC sign flipped: data written by SNANA "
                          f"{snana_version or 'UNKNOWN'}")
        batch_head_extras['VPEC'] = -batch_head_extras['VPEC']

    # VPEC_ERR_OVERRIDE replaces the per-SN peculiar-velocity error
    # (snana.F90:4320); SNANA treats |value| <= 0.001 as unset.
    if abs(vpec_err_override) > 0.001 and 'VPEC_ERR' in batch_head_extras:
        batch_head_extras['VPEC_ERR'] = np.full_like(
            batch_head_extras['VPEC_ERR'], vpec_err_override)

    # OPT_VPEC_COR = 0 drops the correction by zeroing both VPEC and its
    # error, and runs last so it also clears an override (snana.F90:4325).
    if not opt_vpec_cor:
        for column in ('VPEC', 'VPEC_ERR'):
            if column in batch_head_extras:
                batch_head_extras[column] = np.zeros_like(
                    batch_head_extras[column])


def _combine_flux_errors(err1, add_var, kept_flux, bands_e, fields_e,
                         sn_idx_dense, kept_phot_indices, head_data, phot_data,
                         phot_mjd, is_sim, fudge_flux, dataerr_scale,
                         fluxerr_model, sim_fluxerr_model, hostnoise_maps,
                         zp_fluxcal):
    """The per-epoch flux error: additive fudges, FLUXERRMODEL and HOSTNOISE
    in quadrature off the scaled-original ERR1, then DATAERR_SCALE."""
    fem = sim_fluxerr_model if is_sim else fluxerr_model
    if (fudge_flux or dataerr_scale != 1.0 or fem is not None
            or hostnoise_maps is not None):
        var = err1 ** 2 + add_var
        if fem is not None:  # get_FLUXERRMODEL (DATA path)
            col = lambda c: np.asarray(phot_data[c], dtype=np.float64)[kept_phot_indices]
            pixsize = (np.asarray(head_data['PIXSIZE'], dtype=np.float64)[sn_idx_dense]
                       if 'PIXSIZE' in head_data.colnames else np.ones(len(sn_idx_dense)))
            snr = np.where(err1 > 0, kept_flux / np.where(err1 > 0, err1, 1.0), 0.0)
            cond = {'MJD': phot_mjd[kept_phot_indices],
                    'PSF': col('PSF_SIG1') * pixsize * FWHM_PER_SIGMA,
                    'SKYSIG': col('SKY_SIG'), 'ZP': col('ZEROPT'),
                    'LOGSNR': np.maximum(np.log10(np.maximum(snr, 1e-9)), -0.9)}
            err_data = evaluate_fluxerrmodel_data(fem, bands_e, fields_e, err1, cond)
            var = var + err_data ** 2 - err1 ** 2
        if hostnoise_maps is not None:  # FUDGE_HOSTNOISE_FILE (host SB errscale)
            sbf = _host_surface_brightness(head_data, bands_e, sn_idx_dense)
            scale = hostnoise_errscale(hostnoise_maps, bands_e, fields_e, sbf,
                                       zp_fluxcal)
            var = var + err1 ** 2 * (scale ** 2 - 1.0)
        kept_err = np.sqrt(var) * dataerr_scale
    else:
        kept_err = err1
    return kept_err


def _host_surface_brightness(head_data, bands, sn_indices):
    """Gather each epoch's host brightness with one bulk lookup per band."""
    out = np.empty(len(bands), dtype=np.float64)
    for band, rows in pd.Series(sn_indices).groupby(bands).indices.items():
        sb = np.asarray(head_data[f'HOSTGAL_SB_FLUXCAL_{band}'], dtype=np.float64)
        out[rows] = sb[sn_indices[rows]]
    return out


def _padded_slots(keep, sn_idx_all, n_sne):
    """Per-event counts, each kept epoch's event and slot, and the width of
    the padded arrays."""
    counts = np.bincount(sn_idx_all[keep], minlength=n_sne).astype(np.int32)
    starts = np.zeros(n_sne, dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])
    slot = np.arange(int(keep.sum())) - np.repeat(starts, counts)
    width = int(counts.max()) if n_sne > 0 else 0
    return counts, sn_idx_all[keep], slot, width


def _read_window(mjds, sn_idx_all, peakmjds, read_redshifts, trest, zmax):
    """The epochs SNANA reads: within ``trest`` of the file's PEAKMJD plus the
    margin, where it has one (RDOBS_DRIVER, snana.F90:4878-4889), in its
    single precision. ``zmax`` stands in for a missing redshift."""
    peak = peakmjds.astype(np.float32)
    z = read_redshifts.astype(np.float32)
    z1 = np.float32(1.0) + np.where(z < 0.0, np.float32(zmax), z)
    lo = (peak + np.float32(trest[0]) * z1).astype(np.float64) - _READ_MARGIN_DAYS
    hi = (peak + np.float32(trest[1]) * z1).astype(np.float64) + _READ_MARGIN_DAYS
    windowed = (peak > 40000.0)[sn_idx_all]
    return ~windowed | ((mjds >= lo[sn_idx_all]) & (mjds <= hi[sn_idx_all]))


def _epoch_quality_cuts(obs_keep_mask, epoch_windows, epoch_snrmin, head_data,
                        phot_data, phot_mjd, phot_fluxcal, phot_fluxcalerr,
                        valid_phot_indices, sn_idx_all, band_codes,
                        unique_bands, photflag_detect, fluxerrcalc_zpterr,
                        zp_fluxcal, *, at_read):
    """Epoch CUTWIN cuts and EPCUT_SNRMIN, ANDed into the keep mask: with
    ``at_read`` those PASS_EPOCH_CUTS applies (snana.F90:13243), otherwise
    the LSIG ones (:17421)."""
    if epoch_windows:
        mjds = phot_mjd[valid_phot_indices]

        def _phot_col(col, key):
            if col not in phot_data.colnames:
                raise SNANAError(f"CUTWIN_{key.upper()} configured but PHOT "
                                 f"file has no {col} column",
                                 function_name="_epoch_quality_cuts")
            return np.asarray(phot_data[col], dtype=np.float64)[valid_phot_indices]

        for key, (lo, hi) in epoch_windows.items():
            if (key in _READ_EPOCH_CUTS) != at_read:
                continue
            if key == 'mjd':
                obs_keep_mask &= in_window(mjds, lo, hi)
            elif key == 'mjd_exclude':
                obs_keep_mask &= ~in_window(mjds, lo, hi)
            elif key == 'zperr':
                obs_keep_mask &= in_window(_phot_col('ZEROPT_ERR', key), lo, hi)
            elif key == 'zpadu':  # SNLC_ZEROPT_forCUT in ADU (snana.F90:13934)
                obs_keep_mask &= in_window(_phot_col('ZEROPT', key), lo, hi)
            elif key == 'zpnpe':  # the same zeropoint in photoelectrons (:13944)
                gain = _phot_col('GAIN', key)
                zp_pe = (_phot_col('ZEROPT', key)
                         + 2.5 * np.log10(np.where(gain < 0.01, 0.001, gain)))
                obs_keep_mask &= in_window(zp_pe, lo, hi)
            elif key == 'photprob':
                pp = _phot_col('PHOTPROB', key)
                obs_keep_mask &= (pp > lo) | (pp < -0.001)
            elif key == 'psf':  # LPSF on SNLC_PSF_FWHM_ARCSEC (snana.F90:13963)
                if 'PIXSIZE' not in head_data.colnames:
                    raise SNANAError(
                        "CUTWIN_PSF configured but HEAD file has no PIXSIZE "
                        "column", function_name="_epoch_quality_cuts")
                pixsize = np.asarray(head_data['PIXSIZE'],
                                     dtype=np.float64)[sn_idx_all]
                fwhm = _phot_col('PSF_SIG1', key) * pixsize * FWHM_PER_SIGMA
                obs_keep_mask &= in_window(fwhm, lo, hi)
            elif key == 'errtest':
                ferr = phot_fluxcalerr[valid_phot_indices]
                total_obs = obs_keep_mask.size
                valid_bands = unique_bands[band_codes]
                sb_ep = np.zeros(total_obs)
                if 'PIXSIZE' in head_data.colnames:
                    for b, ch in ((b, filter_band(b))
                                  for b in np.unique(valid_bands)):
                        k = f'HOSTGAL_SB_FLUXCAL_{ch}'
                        if k in head_data.colnames:
                            sb_sn = np.asarray(head_data[k],
                                               dtype=np.float64)
                            sb_ep[valid_bands == b] = \
                                sb_sn[sn_idx_all][valid_bands == b]
                    pixsize_ep = np.asarray(
                        head_data['PIXSIZE'],
                        dtype=np.float64)[sn_idx_all]
                else:
                    pixsize_ep = np.zeros(total_obs)
                errcalc = fluxcal_errcalc(
                    phot_fluxcal[valid_phot_indices], ferr,
                    _phot_col('ZEROPT', key),
                    (np.full(total_obs, fluxerrcalc_zpterr)
                     if fluxerrcalc_zpterr >= 0.0
                     else _phot_col('ZEROPT_ERR', key)),
                    _phot_col('SKY_SIG', key), _phot_col('SKY_SIG_T', key),
                    _phot_col('PSF_SIG1', key), _phot_col('PSF_SIG2', key),
                    _phot_col('PSF_RATIO', key), _phot_col('GAIN', key),
                    zp_fluxcal, sb_fluxcal=sb_ep, pixsize=pixsize_ep)
                errtest = np.where(ferr > 0, errcalc / np.where(ferr > 0, ferr, 1.0),
                                   SNANA_NULLFLOAT)
                obs_keep_mask &= in_window(errtest, lo, hi)
            elif key == 'snr_nodetect':
                # only non-detections are tested (snana.F90:13514)
                if not photflag_detect:
                    raise SNANAError(
                        "CUTWIN_SNR_NODETECT configured but PHOTFLAG_DETECT "
                        "is unset, so no epoch can be called a non-detection",
                        function_name="_epoch_quality_cuts")
                pf = _phot_col('PHOTFLAG', key).astype(np.int64)
                ferr = phot_fluxcalerr[valid_phot_indices]
                snr = np.where(ferr > 0,
                               phot_fluxcal[valid_phot_indices]
                               / np.where(ferr > 0, ferr, 1.0), 0.0)
                detected = (pf & int(photflag_detect)) > 0
                obs_keep_mask &= detected | in_window(snr, lo, hi)
            elif key not in _POST_EPOCH_CUTS:
                raise SNANAError(
                    f"CUTWIN_{key.upper()} is not a supported epoch cut",
                    function_name="_epoch_quality_cuts")

    if epoch_snrmin:
        floors = np.array(filter_values(epoch_snrmin, unique_bands, -np.inf))[band_codes]
        if at_read:
            # a floor above 99.9999 drops the band outright, which is how
            # 'u 999' turns a band off (snana.F90:13355)
            return obs_keep_mask & (floors <= EPCUT_SNRMIN_OFF)
        # the per-filter S/N floor itself (snana.F90:17390, LSIG >= floor)
        snr_ep = np.where(
            phot_fluxcalerr[valid_phot_indices] > 0,
            phot_fluxcal[valid_phot_indices]
            / np.where(phot_fluxcalerr[valid_phot_indices] > 0,
                       phot_fluxcalerr[valid_phot_indices], 1.0),
            SNANA_NULLFLOAT)
        obs_keep_mask &= snr_ep >= floors
    return obs_keep_mask


def _host_photoz_columns(head_data, abort_on_badqzphot, dofit_photoz):
    """The host photo-z HEAD columns: (photoz, photoz_err, zpdf_quantiles,
    zpdf_percentiles, zpdf_nbin, redshift_source_mask), each None where the
    file has no such column. The quantiles and their percentiles are
    [n_sne, n_q], the null marker throughout for an event with none
    (zpdf_nbin 0)."""
    photoz = photoz_err = zpdf_quantiles = zpdf_percentiles = zpdf_nbin = None
    redshift_source_mask = None

    if 'HOSTGAL_PHOTOZ' in head_data.colnames:
        photoz = np.array(head_data['HOSTGAL_PHOTOZ'], dtype=np.float64)
    if 'HOSTGAL_PHOTOZ_ERR' in head_data.colnames:
        photoz_err = np.array(
            head_data['HOSTGAL_PHOTOZ_ERR'], dtype=np.float64)
    if 'MASK_REDSHIFT_SOURCE' in head_data.colnames:
        redshift_source_mask = np.array(
            head_data['MASK_REDSHIFT_SOURCE'], dtype=np.int32)

    # quantile vector (sntools_dataformat_fits.c:3761), else legacy Qnnn columns
    if 'HOSTGALz_QUANTILE_ZPHOT' in head_data.colnames:
        nbin = np.asarray(head_data['HOSTGALz_NBIN_QUANTILE_ZPHOT'], dtype=int)
        n_q = int(nbin.max()) if len(nbin) else 0
        if n_q > 0:
            zpdf_quantiles = np.array(
                head_data['HOSTGALz_QUANTILE_ZPHOT'], dtype=np.float64)[:, :n_q]
            zpdf_percentiles = np.array(
                head_data['HOSTGALz_QUANTILE_PERCENT'], dtype=np.float64)[:, :n_q]
            zpdf_nbin = nbin
    else:
        q_cols = sorted([c for c in head_data.colnames
                         if c.startswith('HOSTGAL_ZPHOT_Q')])
        if q_cols:
            zpdf_quantiles = np.column_stack(
                [np.array(head_data[c], dtype=np.float64) for c in q_cols])
            zpdf_percentiles = np.broadcast_to(
                np.array([float(c[len('HOSTGAL_ZPHOT_Q'):]) for c in q_cols]),
                zpdf_quantiles.shape).copy()
            zpdf_nbin = np.full(len(zpdf_quantiles), len(q_cols))
    if zpdf_quantiles is not None:
        check_zphot_quantiles(zpdf_quantiles, abort_on_badqzphot)

    return (photoz, photoz_err, zpdf_quantiles, zpdf_percentiles, zpdf_nbin,
            redshift_source_mask)


def _check_helio2cmb(head_data, ras, decs, snids, ztol_helio2cmb):
    """zTOL_HELIO2CMB: the file's zCMB must agree with the value recomputed
    from zHEL, or the header is inconsistent (snana.F90:17670)."""
    if 'REDSHIFT_HELIO' not in head_data.colnames or np.all(np.isnan(ras)):
        return
    zhel_col = np.array(head_data['REDSHIFT_HELIO'], dtype=float)
    # tested only above z = 0.003, and skipped when the zHEL error
    # marks it unknown (snana.F90:17655, :17671)
    zhel_err = (np.array(head_data['REDSHIFT_HELIO_ERR'], dtype=float)
                if 'REDSHIFT_HELIO_ERR' in head_data.colnames
                else np.zeros_like(zhel_col))
    # against the file's own zCMB, not the fitting redshift, which is
    # heliocentric (zRATIO8 = zCMB8/SNLC_zCMB, snana.F90:17666)
    zcmb_col = (np.array(head_data['REDSHIFT_FINAL'], dtype=float)
                if 'REDSHIFT_FINAL' in head_data.colnames
                else np.full(len(zhel_col), SNANA_NULLFLOAT))
    good = (zcmb_col > 0.0) & (zhel_col > 0.003) & (zhel_err < 0.9999)
    if not good.any():
        return
    zcmb_calc = np.asarray(zhelio_to_zcmb(zhel_col, ras, decs))
    dz = np.abs(1.0 - zcmb_calc / np.where(zcmb_col > 0, zcmb_col, 1.0))
    bad = good & (dz > ztol_helio2cmb)
    if bad.any():
        i = int(np.argmax(bad))
        raise SNANAError(
            f"CID {snids[i]}: zCMB recomputed from zHEL "
            f"({zcmb_calc[i]:.5f}) differs from the file's "
            f"{zcmb_col[i]:.5f} by {dz[i]:.4f}, over "
            f"ZTOL_HELIO2CMB={ztol_helio2cmb}.",
            function_name="load_and_batch_snana_fits")


def _resolve_redshifts(head_data, redshifts, redshift_errs, ras, decs,
                       snids, batch_head_extras, *, abort_on_badqzphot,
                       dofit_photoz, quantile_spline, use_snhost_zphot,
                       use_snhost_qzphot, zshifts):
    """Host photo-z columns, the zCMB consistency check, the host-z
    substitution and REDSHIFT_*_SHIFT.

    Rewrites the REDSHIFT_HELIO entries of ``batch_head_extras`` where the
    substitution applies; returns the per-SN redshift columns."""
    (photoz, photoz_err, zpdf_quantiles, zpdf_percentiles, zpdf_nbin,
     redshift_source_mask) = _host_photoz_columns(
        head_data, abort_on_badqzphot, dofit_photoz)
    # SET_SNHOST_QZPHOT (snana.F90:17728), SET_SNHOST_ZPHOT (:17984)
    if use_snhost_zphot or use_snhost_qzphot:
        if use_snhost_zphot:
            if photoz is None:
                raise SNANAError(
                    "USE_SNHOST_ZPHOT configured but the HEAD file has no "
                    "HOSTGAL_PHOTOZ column",
                    function_name="_resolve_redshifts")
            zph = np.asarray(photoz, dtype=float)
        else:
            if zpdf_quantiles is None:
                raise SNANAError(
                    "USE_SNHOST_QZPHOT configured but the HEAD file has no "
                    "photo-z quantile columns",
                    function_name="_resolve_redshifts")
            usable = zpdf_rows_usable(zpdf_quantiles, zpdf_nbin)
            zph = np.full(len(zpdf_quantiles), SNANA_NULLFLOAT)
            if usable.any():
                zph[usable] = precompute_zpdf_spline(
                    zpdf_quantiles[usable], zpdf_percentiles[usable] / 100.0,
                    spline=quantile_spline, n_quantiles=zpdf_nbin[usable])['pdf_mean']
        # both branches need EXIST_SNHOST_ZPHOT
        exists = (np.asarray(photoz, dtype=float) > 0 if photoz is not None
                  else np.zeros(len(zph), dtype=bool))
        if not exists.all():
            bad = snids[~exists][:3]
            raise SNANAError(
                f"No host photo-z for CID(s) {', '.join(map(str, bad))}; "
                f"check the data files, or unset USE_SNHOST_ZPHOT.",
                function_name="SET_SNHOST_ZPHOT")
        zerr = (np.asarray(photoz_err, dtype=float)
                if photoz_err is not None else np.zeros_like(zph))
        redshifts = zph
        redshift_errs = zerr
        for col, val in (('REDSHIFT_HELIO', zph),
                         ('REDSHIFT_HELIO_ERR', zerr)):
            if col in batch_head_extras:
                batch_head_extras[col] = val

    # zCMB from the unshifted zHEL; shifted below
    zhel_src = np.asarray(
        batch_head_extras.get('REDSHIFT_HELIO', redshifts), dtype=float)
    zcmb = np.asarray(zhelio_to_zcmb(zhel_src, ras, decs))

    if zshifts:
        specz = (np.array(head_data['HOSTGAL_SPECZ'],
                          dtype=np.float64)
                 if 'HOSTGAL_SPECZ' in head_data.colnames else None)
        (redshifts, photoz, zpdf_quantiles, zcmb) = apply_redshift_shifts(
            zshifts, redshifts, batch_head_extras, photoz,
            zpdf_quantiles, specz, zcmb)

    return {'redshifts': redshifts, 'redshift_errs': redshift_errs,
            'photoz': photoz, 'photoz_err': photoz_err,
            'zpdf_quantiles': zpdf_quantiles,
            'zpdf_percentiles': zpdf_percentiles, 'zpdf_nbin': zpdf_nbin,
            'zcmb': zcmb,
            'redshift_source_mask': redshift_source_mask}


# MWEBV_FORCE applies above this (snana.F90:5968)
MWEBV_FORCE_UNSET = -0.01


def _resolve_mwebv(mwebvs, mwebv_errs, ras, decs, mwebv):
    """The Galactic E(B-V) and error every later step reads, set as the light
    curve is read (snana.F90:17195-17205): OPT_MWEBV takes none (0), the
    header's (1), the SFD98 map at RA/DEC with a sixth of itself as error (2)
    or 0.86 of it with five percent (3) (modify_MWEBV_SFD, MWgaldust.c), then
    MWEBV_SCALE, MWEBV_SHIFT and MWEBV_FORCE apply."""
    opt = mwebv['opt']
    if opt == 0:
        ebv, err = np.zeros_like(mwebvs), np.zeros_like(mwebvs)
    elif opt == 1:
        ebv, err = mwebvs, mwebv_errs
    elif opt in (2, 3):
        if np.any(np.isnan(ras)):
            raise SNANAError(f"OPT_MWEBV={opt} needs RA/DEC in the HEAD file",
                             function_name="modify_MWEBV_SFD")
        ebv = np.asarray(mwebv['sfd_map'].get_ebv_batch(ras, decs),
                         dtype=np.float64)
        if opt == 3:
            ebv = 0.86 * ebv
            err = 0.05 * ebv
        else:
            err = ebv / 6.0
    else:
        raise SNANAError(f"Unsupported OPT_MWEBV={opt}",
                         function_name="modify_MWEBV_SFD")
    ebv = ebv * mwebv['scale'] + mwebv['shift']
    err = err * mwebv['scale']
    if mwebv['force'] > MWEBV_FORCE_UNSET:
        ebv = np.full_like(ebv, mwebv['force'])
        err = np.zeros_like(err)
    return ebv, err


def _peak_view(batch, keep, sn_idx_all, mjds, fluxcals, fluxcalerrs,
               filter_indices):
    """``batch`` holding the epochs SET_PEAKMJD sees: ``keep``, every band,
    flux and error before any fudge (snana.F90:17224, :17318)."""
    counts, rows, slots, width = _padded_slots(keep, sn_idx_all, batch.n_sne)

    def padded(values, dtype=np.float64):
        out = np.zeros((batch.n_sne, width), dtype=dtype)
        out[rows, slots] = values[keep]
        return out

    return replace(
        batch,
        mjd_offsets=padded(mjds) - np.asarray(batch.reference_mjds)[:, None],
        fluxcals=padded(fluxcals), fluxcalerrs=padded(fluxcalerrs),
        filter_indices=padded(filter_indices, np.int32),
        observation_mask=np.arange(width)[None, :] < counts[:, None],
        n_obs_per_sn=counts, obs_extras=None)


def peak_sizes(sources, setpkmjd, shared):
    """Peak-estimator allocation bounds from the available HEAD/PHOT tables.

    Raw observation counts bound every epoch selection. Only rows referenced
    by HEAD pointers contribute to the span; PHOT separator rows are excluded.
    """
    if not setpkmjd or not (setpkmjd['fmax_clump'] or setpkmjd['fluxmax']):
        return None
    counts = []
    max_span = 0.0
    for _, (head, phot, metadata) in sources:
        head = Table(head, copy=False)
        override = shared.get('sim_header_override' if metadata['is_sim']
                              else 'header_override')
        if override is not None:
            head = apply_header_override(head, override)
        starts = np.asarray(head['PTROBS_MIN'], dtype=np.int64) - 1
        stops = np.asarray(head['PTROBS_MAX'], dtype=np.int64)
        nobs = stops - starts
        counts.append(nobs)
        offsets = np.arange(nobs.sum()) - np.repeat(np.cumsum(nobs) - nobs, nobs)
        times = np.asarray(phot['MJD'], dtype=np.float64)[
            np.repeat(starts, nobs) + offsets]
        if len(times):
            max_span = max(max_span, float(times.max() - times.min()))
    return {
        'epoch_lengths': choose_epoch_lengths(np.concatenate(counts),
            setpkmjd['num_nobs_bins'], setpkmjd['min_bin_gain']),
        'histogram_bins': max(int(max_span / 10.0) + 1,
                              int(setpkmjd['mjdwin'] / 10.0 + 0.01))}


def _set_peakmjd(batch, setpkmjd, peak_shape=None):
    """OPT_SETPKMJD: photometric PEAKMJD estimate (SNANA SET_PEAKMJD).

    Bit 4 (=16) Fmax clump, bit 0 (=1) Bazin09. The Fmax clump falls back to
    the header value where it finds no peak; an event the Bazin fit cannot
    place aborts, or carries PKMJD -1 where ABORT_ON_NOPKMJD or bit 2 allows
    it. SNANA estimates with or without a header peak (snana.F90:26376);
    with none, the fallback is the mean MJD the fit otherwise starts from."""
    if not setpkmjd:
        return batch
    reference_mjds = np.asarray(batch.reference_mjds)
    # every estimate runs BATCH_SIZE events at a time, so device memory does
    # not grow with the sample; each event's peak is its own
    size = setpkmjd['batch_size']
    new_off = (np.array(batch.peakmjd_offsets)
               if batch.peakmjd_offsets is not None
               else np.zeros(batch.n_sne))

    if setpkmjd['fmax_clump'] or setpkmjd['fluxmax']:
        obs_ok = np.asarray(batch.observation_mask)
        mjd_offsets = np.asarray(batch.mjd_offsets)
        fluxcals = np.asarray(batch.fluxcals)
        fluxcalerrs = np.asarray(batch.fluxcalerrs)
        n_obs = obs_ok.sum(axis=1)
        groups = []
        lengths = (peak_shape['epoch_lengths'] if peak_shape is not None
                   else choose_epoch_lengths(n_obs, setpkmjd['num_nobs_bins'],
                                              setpkmjd['min_bin_gain']))
        for length, idx in group_by_epoch_length(n_obs, lengths):
            ok = obs_ok[idx, :length]
            err = fluxcalerrs[idx, :length]
            flux = fluxcals[idx, :length]
            mjd = mjd_offsets[idx, :length] + reference_mjds[idx, None]
            snr = np.where(ok & (err > 0),
                           flux / np.where(err > 0, err, 1.0), 0.0)
            groups.append((length, idx, mjd, flux, snr, ok))
        # one MJD span for every group's clump histogram
        span = (max(np.max(np.where(ok, mjd, -np.inf), initial=-np.inf)
                    for _, _, mjd, _, _, ok in groups)
                - min(np.min(np.where(ok, mjd, np.inf), initial=np.inf)
                      for _, _, mjd, _, _, ok in groups))
        peak_abs = np.full(batch.n_sne, PEAKMJD_MISSING)
        for length, idx, mjd, flux, snr, ok in groups:
            if not length:
                continue
            for start in range(0, len(idx), size):
                rows = slice(start, start + size)
                count = len(idx[rows])
                padding = ((0, size - count), (0, length - mjd.shape[1]))
                chunk_mjd, chunk_flux, chunk_snr = (
                    np.pad(values[rows], padding) for values in (mjd, flux, snr))
                chunk_nobs = np.pad(n_obs[idx[rows]], (0, size - count))
                if setpkmjd['fmax_clump']:
                    peak_abs[idx[rows]] = np.asarray(fmax_clump_peakmjd(
                        chunk_mjd, chunk_flux, chunk_snr, chunk_nobs,
                        mjdwin=setpkmjd['mjdwin'], snrcut=setpkmjd['snrcut'],
                        snrcut_backup=setpkmjd['snrcut_backup'], mjd_span=span,
                        histogram_bins=(peak_shape['histogram_bins']
                                        if peak_shape is not None else None)))[:count]
                else:
                    peak_abs[idx[rows]] = np.asarray(naive_fluxmax_peakmjd(
                        chunk_mjd, chunk_flux, chunk_snr, chunk_nobs,
                        snrcut_backup=setpkmjd['snrcut_backup']))[:count]
        found = peak_abs != PEAKMJD_MISSING
        new_off[found] = peak_abs[found] - reference_mjds[found]

    if (setpkmjd.get('trigger') and not setpkmjd['fmax_clump']
            and not setpkmjd['fluxmax']):
        # PKMJD = SNLC8_MJD_TRIGGER, the HEAD column. SNANA reaches it only
        # where no max-flux option is set (snana.F90:26557-26562).
        trig = (np.asarray(batch.head_extras['MJD_TRIGGER'], dtype=float)
                if 'MJD_TRIGGER' in batch.head_extras
                else np.full(batch.n_sne, MJD_TRIGGER_MISSING))
        new_off = trig - reference_mjds

    if setpkmjd['bazin']:
        slog.line(logger, "Running Bazin pre-fit...")
        # bazin_prefit reads batch.mjd_offsets: offset frame in and out.
        # to host per chunk, so asynchronous dispatch cannot queue them all
        fits = [[np.asarray(out) for out in bazin_prefit(
                    batch.take(slice(start, start + size)),
                    setpkmjd['cutwin_tobs'], jnp.array(new_off[start:start + size]))]
                for start in range(0, batch.n_sne, size)]
        peak_off, failed, fallback = (np.concatenate(column) for column in zip(*fits))
        slog.line(logger, f"Bazin pre-fit: {int((~fallback).sum())}/{batch.n_sne} SNe "
                          f"refined, {int(fallback.sum())} using fallback")
        # an event no filter could place aborts, or carries PKMJD=-1
        # when the key or OPT_SETPKMJD bit 2 allows it (snana.F90:26638)
        if failed.any():
            lost = [str(batch.snids[i]) for i in np.flatnonzero(failed)][:5]
            if setpkmjd['abort_on_nopkmjd'] and not (
                    setpkmjd['opt'] & OPTMASK_SETPKMJD_NOABORT):
                raise SNANAError(
                    f"no PKMJD estimate for {int(failed.sum())} "
                    f"event(s): {lost}. Override with "
                    f"ABORT_ON_NOPKMJD = F or OPT_SETPKMJD += 4.",
                    function_name="SET_PEAKMJD")
            logger.warning("Could not estimate PKMJD for %d event(s): %s",
                           int(failed.sum()), lost)
        new_off[~failed] = peak_off[~failed]
        new_off[failed] = PKMJD_NOFIT - reference_mjds[failed]
        # label 800 shifts every Bazin path, the -1 included
        # (snana.F90:26793); the branches above return before it
        new_off += setpkmjd['shift']

    # SET_PEAKMJD overwrites SNLC_SEARCH_PEAKMJD with its estimate
    # (snana.F90:26797), which is what the FITRES reports as PKMJDINI
    head_extras = dict(batch.head_extras)
    head_extras['PEAKMJD'] = new_off + reference_mjds
    return replace(batch, peakmjd_offsets=new_off, head_extras=head_extras)


def _apply_post_peak_epoch_cuts(batch, epoch_windows, filter_lamavg):
    """Peak/wavelength-dependent epoch cuts (SNANA LSIG: LTobs/LTrest/LLAM),
    applied after the final peak is known. ANDed into observation_mask so they
    feed the SNRMAX/TREST cut variables and the fit, exactly as the column-only
    epoch cuts do at load time."""
    post_cuts = {k: v for k, v in (epoch_windows or {}).items()
                 if k in _POST_EPOCH_CUTS}
    if not post_cuts:
        return batch
    obs_ok = np.asarray(batch.observation_mask)
    z = np.asarray(batch.redshifts)
    z1 = trest_z1(z)[:, None]
    keep_ep = np.ones_like(obs_ok)
    if 'tobs' in post_cuts or 'trest' in post_cuts:
        if batch.peakmjd_offsets is None:
            raise SNANAError("CUTWIN_TOBS/TREST need a peak-MJD estimate",
                             function_name="_apply_post_peak_epoch_cuts")
        tobs = (np.asarray(batch.mjd_offsets)
                - np.asarray(batch.peakmjd_offsets)[:, None])
        if 'tobs' in post_cuts:
            keep_ep &= in_window(tobs, *post_cuts['tobs'])
        if 'trest' in post_cuts:
            keep_ep &= in_window(tobs / z1, *post_cuts['trest'])
    if 'lamobs' in post_cuts or 'lamrest' in post_cuts:
        if not filter_lamavg:
            raise SNANAError("CUTWIN_LAMOBS/LAMREST need filter_lamavg",
                             function_name="_apply_post_peak_epoch_cuts")
        lam_by_idx = np.array([filter_lamavg[batch.filter_names[i]]
                               for i in range(len(batch.filter_names))])
        lamobs = lam_by_idx[np.asarray(batch.filter_indices)]
        # both wavelength cuts wait for a redshift (snana.F90:17309)
        has_z = ((z > 0.0) & (z < 998.0))[:, None]
        if 'lamobs' in post_cuts:
            keep_ep &= in_window(lamobs, *post_cuts['lamobs']) | ~has_z
        if 'lamrest' in post_cuts:
            keep_ep &= in_window(lamobs / z1, *post_cuts['lamrest']) | ~has_z
    return _compact_epochs(batch, obs_ok & keep_ep)


def _compact_epochs(batch, keep):
    """Shift surviving epochs down so each row stays a contiguous prefix, as
    SELECT_EPOCH_DRIVER does via MOVE_SNLC_ARRAYS (snana.F90:13392).

    Masking in place would leave holes, while observation_mask is built as
    ``arange(max_obs) < n_obs_per_sn``; code assuming that prefix, as
    fmax_clump_peakmjd does, would then read the wrong epochs."""
    keep = np.asarray(keep)
    n_obs = keep.sum(axis=1)
    order = np.argsort(~keep, axis=1, kind='stable')
    rows = np.arange(keep.shape[0])[:, None]

    def gather(arr):
        return np.asarray(arr)[rows, order]

    extras = batch.obs_extras
    if extras is not None:
        extras = {k: gather(v) for k, v in extras.items()}
    return replace(
        batch,
        mjd_offsets=gather(batch.mjd_offsets),
        fluxcals=gather(batch.fluxcals),
        fluxcalerrs=gather(batch.fluxcalerrs),
        filter_indices=gather(batch.filter_indices),
        obs_extras=extras,
        observation_mask=np.arange(keep.shape[1])[None, :] < n_obs[:, None],
        n_obs_per_sn=n_obs)


def _apply_jobsplit(batch, jobsplit):
    """JOBSPLIT strides the raw file index (DO isn = IJOB, NSN_VERS, NJOBTOT,
    snana.F90:2893), so events outside this job's share are never read."""
    ijob, njobtot = jobsplit
    if njobtot <= 1:
        return batch
    return batch.filter_by_mask(
        ((np.arange(batch.n_sne) + 1 - ijob) % njobtot) == 0)


def read_epoch_ignore_file(path):
    """SNANA EPOCH_IGNORE_FILE (READ_EPIGNORE_FILE, snana.F90:12957).

    ``IGNORE: <CID> <MJD> <FILTER>`` rows name single epochs to drop; a
    ``PHOTFLAG_MSKREJ: <mask>`` row supplies a rejection mask alongside them.
    Returns ``(ignored_epochs, photflag_mskrej_or_None)`` where the first is a
    set of ``(cid, round(mjd, 3), band)``, the same keying MAGCOR_FILE uses.
    """
    ignored, mask = set(), None
    with open(os.path.expandvars(path)) as fh:
        for line in fh:
            toks = line.split('#')[0].split()
            if not toks:
                continue
            if toks[0] == 'IGNORE:':
                ignored.add((toks[1], round(float(toks[2]), 3), toks[3]))
            elif toks[0] == 'PHOTFLAG_MSKREJ:':
                mask = int(float(toks[1]))
    return ignored, mask


# HEAD columns matched by the override's first VARNAMES column
_OVERRIDE_MATCH_COL = {'CID': 'SNID', 'SNID': 'SNID',
                       'GALID': 'HOSTGAL_OBJID',
                       'HOSTGAL_OBJID': 'HOSTGAL_OBJID'}
# quantile overrides: QZPHOTnn columns with implied uniform percentiles
# (PREFIX_QZPHOT, sndata.h:131), or one row per quantile in these two columns
QZPHOT_OVERRIDE_PREFIX = 'QZPHOT'
QUANTILE_ROW_COLUMNS = ('HOSTGALz_QUANTILE_ZPHOT', 'HOSTGALz_QUANTILE_PERCENT')
# a host LOGMASS(z) grid, one row per redshift (rd_override_hostgal2z)
LOGMASS_ROW_COLUMNS = ('HOSTGALz_LOGMASS_ZGRID', 'HOSTGALz_LOGMASS_VALGRID',
                       'HOSTGALz_LOGMASS_ERRGRID')
OBSOLETE_QZPHOT_PREFIX = 'HOSTGAL_ZPHOT_Q'


def read_header_override_file(file_list=None, override_dir=None):
    """SNANA HEADER_OVERRIDE_FILE/DIR (RD_OVERRIDE_INIT): per-CID (or per-GALID)
    replacements for existing HEAD columns. Reads one or more SNANA text tables
    (optional DOCANA block, then ``VARNAMES:`` + ``SN:``/``ROW:``/``GAL:``
    rows, a key of ``*`` repeating the previous row's); the first VARNAMES
    column is the match key (CID/SNID or GALID/HOSTGAL_OBJID), the rest are
    override columns. Returns ``(match_col, values)`` where ``match_col`` is
    the HEAD column to match on and ``values`` maps ``varname -> {key_str:
    value}``, the value a list for the per-quantile row columns. All files
    must share the match mode."""
    paths = []
    for entry in (file_list or []):
        paths.extend(p for p in os.path.expandvars(entry).split(',') if p)
    if override_dir:
        d = Path(os.path.expandvars(override_dir))
        paths.extend(sorted(str(p) for p in d.iterdir()))
    if not paths:
        return None

    match_col = None
    values = {}
    for path in paths:
        varnames = None
        key = None
        with open(path) as fh:
            for line in skip_docana(fh):
                toks = line.split('#')[0].split()
                if not toks:
                    continue
                if toks[0] == 'VARNAMES:':
                    varnames = toks[1:]
                    obsolete = [v for v in varnames
                                if v.startswith(OBSOLETE_QZPHOT_PREFIX)]
                    if obsolete:
                        raise SNANAError(
                            f"Found {len(obsolete)} obsolete OVERRIDE columns with "
                            f"'{OBSOLETE_QZPHOT_PREFIX}' prefix; replace with "
                            f"QZPHOT00 QZPHOT01 ... QZPHOT{len(obsolete) - 1:02d}",
                            function_name="rd_override_qzphot_implicit")
                    key_name = varnames[0].upper()
                    if key_name not in _OVERRIDE_MATCH_COL:
                        raise SNANAError(
                            f"header-override match key '{varnames[0]}' must be "
                            "CID/SNID or GALID/HOSTGAL_OBJID",
                            function_name="read_header_override_file")
                    this_col = _OVERRIDE_MATCH_COL[key_name]
                    if match_col is None:
                        match_col = this_col
                    elif this_col != match_col:
                        raise SNANAError(
                            "header-override files mix CID and GALID keys; all "
                            "must use the same match key",
                            function_name="read_header_override_file")
                elif toks[0] in ('ROW:', 'SN:', 'GAL:'):
                    if toks[1] == '*' and key is None:
                        raise SNANAError(
                            f"the first row of {path} repeats the previous "
                            f"key with '*', and there is none",
                            function_name="read_header_override_file")
                    if toks[1] != '*':
                        key = toks[1]
                    for j, vname in enumerate(varnames[1:], start=1):
                        col = ('REDSHIFT_FINAL' if vname.upper() == 'REDSHIFT_CMB'
                               else vname)
                        raw = toks[1 + j]
                        try:
                            val = float(raw)
                        except ValueError:
                            val = raw
                        if col in QUANTILE_ROW_COLUMNS or col in LOGMASS_ROW_COLUMNS:
                            values.setdefault(col, {}).setdefault(key, []).append(val)
                        else:
                            values.setdefault(col, {})[key] = val
    return match_col, values


def _quantile_overrides(values):
    """The quantile overrides as ``{key: (z, percent)}`` arrays, and whether
    they came as QZPHOTnn columns. Those carry the quantiles as read, with
    the percentiles uniform over the count of values >= 0 and -9 after
    (rd_override_qzphot_implicit, sntools_data.c:2288); the per-quantile rows
    carry both lists as given (rd_override_hostgal2z, :2400)."""
    implicit = sorted((v for v in values
                       if v.startswith(QZPHOT_OVERRIDE_PREFIX)
                       and v[len(QZPHOT_OVERRIDE_PREFIX):].isdigit()),
                      key=lambda v: int(v[len(QZPHOT_OVERRIDE_PREFIX):]))
    if implicit:
        table = {}
        for key in values[implicit[0]]:
            z = np.array([values[v][key] for v in implicit])
            n_valid = int((z >= 0).sum())
            percent = np.full(len(z), SNANA_NULLFLOAT)
            percent[:n_valid] = 100.0 * np.arange(n_valid) / max(n_valid - 1, 1)
            table[key] = (z, percent)
        return table, True
    z_col, pct_col = QUANTILE_ROW_COLUMNS
    if z_col in values:
        return {key: (np.array(z), np.array(values[pct_col][key]))
                for key, z in values[z_col].items()}, False
    return {}, False


def _override_slot_quantiles(head_data, prefix, keys, table, every_event):
    """Write the override's quantiles into one host slot's HOSTGALz vector
    columns (created at the 20-wide null layout when the HEAD lacks them),
    for the events whose ``keys`` the ``table`` holds, or for every event
    under the QZPHOTnn spelling, which leaves an absent event with none
    (rd_override_qzphot_implicit runs for each event). Returns the mask of
    events written."""
    n = len(keys)
    names = [f'{prefix}_NBIN_QUANTILE_ZPHOT', f'{prefix}_QUANTILE_ZPHOT',
             f'{prefix}_QUANTILE_PERCENT']
    if names[0] not in head_data.colnames:
        head_data[names[0]] = np.zeros(n, dtype=np.int16)
        for name in names[1:]:
            head_data[name] = np.full((n, MXBIN_HOSTGALz_QUANTILE), SNANA_NULLFLOAT,
                                      dtype=np.float32)
    nbin, z, percent = (np.array(head_data[name]) for name in names)
    written = np.zeros(n, dtype=bool)
    for i, key in enumerate(keys):
        if key in table:
            zq, pq = table[key]
            nbin[i] = int((zq >= 0).sum())
            z[i], percent[i] = SNANA_NULLFLOAT, SNANA_NULLFLOAT
            z[i, :len(zq)], percent[i, :len(pq)] = zq, pq
            written[i] = True
        elif every_event:
            nbin[i], z[i], percent[i] = 0, SNANA_NULLFLOAT, SNANA_NULLFLOAT
            written[i] = True
    for name, col in zip(names, (nbin, z, percent)):
        head_data[name] = col
    return written


def _override_logmass_grid(head_data, keys, values):
    """Write the override's LOGMASS(z) grid into the reported host's
    HOSTGALz vector columns, created at the 40-wide null layout when the
    HEAD lacks them, for the events whose key the override holds. SNANA
    reads the error grid beside the mass grid for every host, so an override
    without it would leave each error at the -9 fill; it is refused."""
    missing = [c for c in LOGMASS_ROW_COLUMNS if c not in values]
    if missing:
        raise SNANAError(
            f"the header override gives {LOGMASS_ROW_COLUMNS[0]} without "
            f"{', '.join(missing)}", function_name="apply_header_override")
    n = len(keys)
    nbin_name = 'HOSTGALz_NBIN_LOGMASS_ZGRID'
    if nbin_name not in head_data.colnames:
        head_data[nbin_name] = np.zeros(n, dtype=np.int16)
        for name in LOGMASS_ROW_COLUMNS:
            head_data[name] = np.full((n, MXBIN_HOSTGALz), SNANA_NULLFLOAT,
                                      dtype=np.float32)
    nbin = np.array(head_data[nbin_name])
    grids = [np.array(head_data[name]) for name in LOGMASS_ROW_COLUMNS]
    for i, key in enumerate(keys):
        if key not in values[LOGMASS_ROW_COLUMNS[0]]:
            continue
        rows = [np.asarray(values[name][key], dtype=float)
                for name in LOGMASS_ROW_COLUMNS]
        nbin[i] = len(rows[0])
        for grid, row in zip(grids, rows):
            grid[i] = SNANA_NULLFLOAT
            grid[i, :len(row)] = row
    head_data[nbin_name] = nbin
    for name, grid in zip(LOGMASS_ROW_COLUMNS, grids):
        head_data[name] = grid


def _cubic_at(knots, values, slopes, q):
    """The piecewise cubic through ``values`` at the ``knots`` with first
    derivative ``slopes`` there, at one point per row, and 0 outside the
    knots, as eval_spline returns (sntools_spline.c:266)."""
    x = knots[0]
    n_knots = x.shape[1]
    i = np.clip((x <= q[:, None]).sum(axis=1) - 1, 0, n_knots - 2)
    r = np.arange(len(q))
    h = x[r, i + 1] - x[r, i]
    t = (q - x[r, i]) / h
    t2, t3 = t * t, t * t * t
    cubic = ((2 * t3 - 3 * t2 + 1) * values[r, i]
             + (t3 - 2 * t2 + t) * h * slopes[r, i]
             + (3 * t2 - 2 * t3) * values[r, i + 1]
             + (t3 - t2) * h * slopes[r, i + 1])
    return np.where((q >= x[:, 0]) & (q <= x[:, -1]), cubic, 0.0)


def _set_logmass(host, redshifts, redshift_errs, photoz_fit):
    """Replace the reported host's mass and error by its LOGMASS(z) grid
    (SET_LOGMASS, snana.F90:17831), which SNANA does only when a header
    override carries the grid. A natural cubic spline through the grid
    gives the mass at a redshift and a second through the error grid its
    error, to which half the summed change in mass across one redshift
    error either side is added in quadrature; a mass of 1 or less, which
    includes a redshift outside the grid, leaves the event as it was. The
    step is taken at the redshift the fit uses and, in a photo-z fit, again
    at the host's photo-z (FITANA_LOGMASS, snlc_fit.F90:13047)."""
    nz = np.asarray(host['HOSTGALz_NBIN_LOGMASS_ZGRID'], dtype=int)
    if np.any(nz == 1):
        raise SNANAError(
            "a host LOGMASS(z) grid needs at least two redshifts",
            function_name="_set_logmass")
    mass = np.asarray(host['HOSTGAL_LOGMASS'], dtype=float).copy()
    mass_err = np.asarray(host['HOSTGAL_LOGMASS_ERR'], dtype=float).copy()
    stages = [(redshifts, redshift_errs)]
    if photoz_fit:
        stages.append((host['HOSTGAL_PHOTOZ'], host['HOSTGAL_PHOTOZ_ERR']))
    for n in np.unique(nz[nz > 1]):
        rows = np.flatnonzero(nz == n)
        z, val, err = (np.asarray(host[name], dtype=float)[rows, :n]
                       for name in LOGMASS_ROW_COLUMNS)
        if not np.all(np.diff(z, axis=1) > 0):
            raise SNANAError(
                "a host LOGMASS(z) grid's redshifts must increase strictly",
                function_name="_set_logmass")
        knots_val, knots_err = spline_knots(z, val), spline_knots(z, err)
        slopes_val = natural_spline_slopes(knots_val)
        slopes_err = natural_spline_slopes(knots_err)
        for zc, dz in stages:
            zc = np.asarray(zc, dtype=float)[rows]
            dz = np.asarray(dz, dtype=float)[rows]
            m = _cubic_at(knots_val, val, slopes_val, zc)
            shift = 0.5 * (np.abs(m - _cubic_at(knots_val, val, slopes_val, zc - dz))
                           + np.abs(m - _cubic_at(knots_val, val, slopes_val, zc + dz)))
            e = np.hypot(_cubic_at(knots_err, err, slopes_err, zc), shift)
            ok = m > 1.0
            mass[rows[ok]], mass_err[rows[ok]] = m[ok], e[ok]
    host['HOSTGAL_LOGMASS'] = mass.astype(np.asarray(host['HOSTGAL_LOGMASS']).dtype)
    host['HOSTGAL_LOGMASS_ERR'] = mass_err.astype(
        np.asarray(host['HOSTGAL_LOGMASS_ERR']).dtype)


def _grid_photoz(nbin, z, percent):
    """The mean and width of a quantile grid, weighting each quantile by the
    percent it spans (the first by its own percent, the last by the gap to
    its predecessor, the rest by half the gap between their neighbours) over
    the integer sum of those weights (rd_override_zphot,
    sntools_data.c:2251-2280; STD_from_SUMS); -9 for an empty grid."""
    mean = np.full(len(nbin), SNANA_NULLFLOAT)
    std = np.full(len(nbin), SNANA_NULLFLOAT)
    for i, nz in enumerate(nbin):
        if nz == 0:
            continue
        zq, pq = z[i, :nz].astype(float), percent[i, :nz].astype(float)
        weight = np.empty(nz)
        weight[0] = pq[0]
        weight[-1] = pq[-1] - pq[-2] if nz > 1 else pq[0]
        weight[1:-1] = 0.5 * (pq[2:] - pq[:-2])
        n = int(weight.sum() + 0.5)
        if n == 0:
            continue
        mean[i] = (weight * zq).sum() / n
        spread = (weight * zq * zq).sum() / n - mean[i] ** 2
        std[i] = np.sqrt(spread) if spread > 0 else 0.0
    return mean, std


def apply_header_override(head_data, override):
    """Apply a header override (from :func:`read_header_override_file`) to a HEAD
    table, returning an overridden copy rather than editing the caller's table.
    Overrides columns already present (SNANA-FITS ignores absent ones) and
    the photo-z quantile vectors, then recomputes what follows from them in
    SNANA's order (RD_OVERRIDE_POSTPROC, sntools_data.c:1975): zHEL <-> zCMB
    and their errors, the quantiles of each host slot with a sim's zSN/zHOST
    rescale on the reported host, the point photo-z of a slot whose quantiles
    were overridden, and the redshift of an event without a spec-z whose
    point photo-z was."""
    match_col, values = override
    head_data = head_data.copy()
    if match_col not in head_data.colnames:
        logger.warning("header-override match column %s absent from HEAD; "
                       "no overrides applied", match_col)
        return head_data
    keys = np.char.strip(np.array(head_data[match_col], dtype='U'))
    quantiles, implicit = _quantile_overrides(values)
    plain = {v: kv for v, kv in values.items()
             if not v.startswith(QZPHOT_OVERRIDE_PREFIX)
             and v not in QUANTILE_ROW_COLUMNS and v not in LOGMASS_ROW_COLUMNS}
    if LOGMASS_ROW_COLUMNS[0] in values:
        _override_logmass_grid(head_data, keys, values)

    ra = (np.asarray(head_data['RA'], dtype=float)
          if 'RA' in head_data.colnames else None)
    dec = None
    for c in ('DEC', 'DECL'):
        if c in head_data.colnames:
            dec = np.asarray(head_data[c], dtype=float)
            break

    # every row sharing a key is overridden, as SNANA does
    overridden = {}  # varname -> boolean row mask
    for vname, kv in plain.items():
        if vname not in head_data.colnames:
            logger.warning("header-override column %s absent from HEAD; ignored "
                           "(SNANA-FITS no-op)", vname)
            continue
        ids = np.asarray(list(kv), dtype=keys.dtype)
        vals = np.asarray(list(kv.values()))
        order = np.argsort(ids)
        slot = order[np.minimum(np.searchsorted(ids, keys, sorter=order),
                                len(ids) - 1)]
        hit = ids[slot] == keys
        col = np.array(head_data[vname])
        col[hit] = vals[slot[hit]]
        head_data[vname] = col
        overridden[vname] = hit

    def rows(name):
        return overridden.get(name, np.zeros(len(keys), dtype=bool))

    # zHEL <-> zCMB recompute (rd_override_zspec); REDSHIFT_FINAL == zCMB
    if ra is not None and dec is not None:
        z_hel = rows('REDSHIFT_HELIO')
        z_cmb = rows('REDSHIFT_FINAL')
        m = z_hel & ~z_cmb
        # the TEXT loader omits REDSHIFT_FINAL when the DAT files lack it
        if m.any() and 'REDSHIFT_FINAL' in head_data.colnames:
            zh = np.asarray(head_data['REDSHIFT_HELIO'], dtype=float)
            zcmb = np.asarray(zhelio_to_zcmb(zh, ra, dec))
            zf = np.array(head_data['REDSHIFT_FINAL'])
            zf[m] = zcmb[m]
            head_data['REDSHIFT_FINAL'] = zf
            if 'REDSHIFT_HELIO_ERR' in head_data.colnames \
                    and 'REDSHIFT_FINAL_ERR' in head_data.colnames:
                fe = np.array(head_data['REDSHIFT_FINAL_ERR'])
                fe[m] = np.asarray(head_data['REDSHIFT_HELIO_ERR'], dtype=float)[m]
                head_data['REDSHIFT_FINAL_ERR'] = fe
        m = z_cmb & ~z_hel
        if m.any():
            zc = np.asarray(head_data['REDSHIFT_FINAL'], dtype=float)
            zhel = np.asarray(zcmb_to_zhelio(zc, ra, dec))
            zh = np.array(head_data['REDSHIFT_HELIO'])
            zh[m] = zhel[m]
            head_data['REDSHIFT_HELIO'] = zh

    if quantiles:
        _override_host_quantiles(head_data, match_col, keys, quantiles, implicit,
                                 plain)

    # an overridden point photo-z becomes the redshift of an event that has
    # no spec-z (rd_override_zphot, sntools_data.c:2206-2218)
    zp = rows('HOSTGAL_PHOTOZ')
    if zp.any() and ra is not None and dec is not None:
        zphot = np.asarray(head_data['HOSTGAL_PHOTOZ'], dtype=float)
        zphoterr = np.asarray(head_data['HOSTGAL_PHOTOZ_ERR'], dtype=float)
        # the spec-z is either column, a TEXT version carrying whichever its
        # data files name
        spec = np.zeros(len(keys), dtype=bool)
        if 'HOSTGAL_SPECZ' in head_data.colnames:
            spec |= np.asarray(head_data['HOSTGAL_SPECZ'], dtype=float) > 0.0
        if 'MASK_REDSHIFT_SOURCE' in head_data.colnames:
            spec |= (np.asarray(head_data['MASK_REDSHIFT_SOURCE'], dtype=int)
                     & ZSPEC_SOURCE_MASK) > 0
        m = zp & ~spec
        if m.any():
            zcmb = np.asarray(zhelio_to_zcmb(zphot, ra, dec))
            for name, src in (('REDSHIFT_HELIO', zphot),
                              ('REDSHIFT_HELIO_ERR', zphoterr),
                              ('REDSHIFT_FINAL', zcmb),
                              ('REDSHIFT_FINAL_ERR', zphoterr)):
                c = np.array(head_data[name])
                c[m] = src[m]
                head_data[name] = c
    return head_data


def _override_host_quantiles(head_data, match_col, keys, quantiles, implicit,
                             plain):
    """The quantile part of an override: the reported host's vector from the
    event's key, rescaled by zSN over the true host's redshift in a sim
    (sntools_data.c:2232); under a GALID key the neighbour slots' vectors
    from their own GALIDs, unscaled (rd_override_hostgal2z). Unless the
    override also carries the slot's point photo-z column, a slot whose
    quantiles were overridden takes the grid's mean and width as its
    photo-z; under a GALID key every neighbour of an event whose reported
    host was overridden does (rd_override_zphot, FOUND0_QZPHOT), an absent
    neighbour's grid being empty."""
    written = _override_slot_quantiles(head_data, 'HOSTGALz', keys, quantiles,
                                       implicit)
    if 'SIM_REDSHIFT_HOST' in head_data.colnames:
        zhost = np.asarray(head_data['SIM_REDSHIFT_HOST'], dtype=float)
        ratio = np.where(zhost > 0.0,
                         np.asarray(head_data['SIM_REDSHIFT_HELIO'], dtype=float)
                         / np.where(zhost > 0.0, zhost, 1.0), 1.0)
        z = np.array(head_data['HOSTGALz_QUANTILE_ZPHOT'])
        rescale = written[:, None] & (z > 0.0)
        head_data['HOSTGALz_QUANTILE_ZPHOT'] = np.where(rescale, z * ratio[:, None], z)
    slots = ['HOSTGAL']
    if match_col == 'HOSTGAL_OBJID':
        for prefix in HOSTGAL_PREFIXES[1:]:
            if f'{prefix}_OBJID' not in head_data.colnames:
                break
            galids = np.char.strip(np.array(head_data[f'{prefix}_OBJID'], dtype='U'))
            _override_slot_quantiles(head_data, f'{prefix}z', galids, quantiles,
                                     implicit)
            slots.append(prefix)
    if not written.any():
        return
    for prefix in slots:
        if f'{prefix}_PHOTOZ' in plain:
            continue
        mean, std = _grid_photoz(
            np.asarray(head_data[f'{prefix}z_NBIN_QUANTILE_ZPHOT']),
            np.asarray(head_data[f'{prefix}z_QUANTILE_ZPHOT']),
            np.asarray(head_data[f'{prefix}z_QUANTILE_PERCENT']))
        for name, src in ((f'{prefix}_PHOTOZ', mean), (f'{prefix}_PHOTOZ_ERR', std)):
            col = np.array(head_data[name])
            col[written] = src[written]
            head_data[name] = col


def _apply_flux_fudges(flux, err, band_codes, unique_bands, phot_data,
                       phot_indices, fudge_flux, is_sim):
    """Apply SNANA FUDGE_FLUXCAL_* fudges (EXEC_FUDGE_FLUXCAL) to flat per-epoch
    calibrated flux/error arrays. Each fudge is a per-band map (band char ->
    value); err_pix couples to the PSF noise-equivalent aperture; SIM_FUDGE_MAG_
    ERROR adds a further mag-error term only for sim data. Returns (flux, err1,
    add_var): the scaled-original error and the additive fudge variance."""

    def band_vals(name, default):  # per-filter map, with '' meaning all bands
        spec = fudge_flux.get(name, {})
        per_code = np.array(filter_values(spec, unique_bands, default))
        return per_code[band_codes]

    mag_err = band_vals('mag_err', 0.0)
    if is_sim:
        mag_err = np.sqrt(mag_err**2 + band_vals('sim_mag_err', 0.0)**2)

    err_pix = band_vals('err_pix', 0.0)
    nea = 0.0
    if np.any(err_pix != 0.0):
        col = lambda c: np.asarray(phot_data[c], dtype=np.float64)[phot_indices]
        nea = noise_equivalent_aperture(col('PSF_SIG1'), col('PSF_SIG2'),
                                        col('PSF_RATIO'))
    return apply_fluxcal_fudges(
        flux, err, band_vals('offset', 0.0), band_vals('err_scale', 1.0),
        band_vals('err_add', 0.0), err_pix, mag_err, nea)


def _flux_baselines(flux, err, group, n_groups):
    """Each group's flux baseline for FLUXCAL_BASELINE_CORRECT: the median
    flux of its epochs below -3 sigma, where it has at least four of them,
    and zero elsewhere. The error is the standard error of a median, 1.2533
    times that of their weighted mean."""
    sel = flux < -3.0 * err
    g, f = group[sel], flux[sel]
    n = np.bincount(g, minlength=n_groups)
    ivar = np.bincount(g, weights=err[sel] ** -2, minlength=n_groups)
    ordered = f[np.lexsort((f, g))]  # by group, then by flux
    start = np.cumsum(n) - n
    use = n >= 4
    base = np.zeros(n_groups)
    base_err = np.zeros(n_groups)
    base[use] = 0.5 * (ordered[start[use] + (n[use] - 1) // 2]
                       + ordered[start[use] + n[use] // 2])
    base_err[use] = 1.2533 / np.sqrt(ivar[use])
    return base, base_err


def parse_snana_dat(path):
    """Parse one SNANA TEXT light-curve file (<SN>.DAT): ``KEY: value`` header
    keywords (comments after ``#`` stripped) and the ``VARLIST:`` / ``OBS:``
    photometry table. Returns (header dict, varlist, {col: [str values]})."""
    header, varlist, obs = {}, None, {}
    with open(path) as fh:
        for raw in fh:
            line = raw.split('#', 1)[0].strip()
            if not line:
                continue
            if line.startswith('VARLIST:'):
                varlist = line.split()[1:]
                obs = {c: [] for c in varlist}
            elif line.startswith('OBS:'):
                for col, val in zip(varlist, line.split()[1:]):
                    obs[col].append(val)
            elif line.startswith(('END', 'DETECTION', 'SPECTRUM', 'NSPECTRA')):
                continue
            elif ':' in line:
                key, rest = line.split(':', 1)
                header[key.strip()] = rest.strip()
    return header, varlist, obs


def read_snana_fits(head_path: str, phot_path: str):
    """The HEAD and PHOT tables of one SNANA FITS pair, and the values the
    format supplies itself (is_sim, survey, snana_version, zp_fluxcal,
    sim_mw_colour_law) as the keyword arguments :func:`_build_batch` takes.
    The HEAD string columns are decoded, since they pass through to the output
    as labels; the PHOT ones stay the bytes stored, coded once per distinct
    value by the batch.

    The header must carry DATATYPE and SURVEY, and a SNANA simulation's also
    SIMOPT_MWCOLORLAW and SIM_MWRV, the Milky Way law it was simulated with;
    ``sim_mw_colour_law`` is that pair, None for any other data.
    """
    primary, head = read_fits_table(resolve_fits_path(head_path))
    datatype = str(primary.get('DATATYPE', '')).strip()
    required = ('DATATYPE', 'SURVEY', *(
        SIM_MW_COLOUR_LAW_KEYS if datatype.startswith('SIM_SNANA') else ()))
    missing = [k for k in required if k not in primary]
    if missing:
        raise SNANAError(
            f"{head_path} header lacks {', '.join(missing)}",
            function_name="read_snana_fits")
    is_sim = datatype.startswith('SIM')
    sim_mw_colour_law = (
        (int(primary['SIMOPT_MWCOLORLAW']), float(primary['SIM_MWRV']))
        if datatype.startswith('SIM_SNANA') else None)
    survey = str(primary['SURVEY']).strip()
    snana_version = str(primary.get('SNANA_VERSION', '')).strip()
    # Absent in files written before SNANA Jul 2025
    # (sntools_dataformat_fits.c:4203).
    zp_fluxcal = float(primary.get('ZP_FLUXCAL', ZEROPOINT_FLUXCAL_SNANA_ORIG))
    head_data = Table({name: head[name].astype(str) if head.dtype[name].kind == 'S'
                       else head[name] for name in head.dtype.names}, copy=False)

    _, phot = read_fits_table(resolve_fits_path(phot_path))
    phot_data = Table(phot, copy=False)

    return head_data, phot_data, dict(
        is_sim=is_sim, survey=survey, snana_version=snana_version,
        zp_fluxcal=zp_fluxcal, sim_mw_colour_law=sim_mw_colour_law)


# Rubin DPDD column names
RUBIN_LIGHTCURVE = 'diaObjectForcedSource'
RUBIN_HEAD_COLUMNS = {'SNID': 'diaObjectId', 'RA': 'ra', 'DEC': 'dec'}
RUBIN_PHOT_COLUMNS = {'MJD': 'midpointMjdTai', 'BAND': 'band',
                      'FLUXCAL': 'psfDiffFlux', 'FLUXCALERR': 'psfDiffFluxErr'}
# the PHOTFLAG bit each Rubin flag sets, for PHOTFLAG_MSKREJ
RUBIN_PHOTFLAG_BITS = {'psfDiffFlux_flag': 1, 'invalidPsfFlag': 2}
# Rubin fluxes are nJy, AB magnitude 31.4 at unit flux
RUBIN_ZP_NJY = 31.4


def read_rubin_hats(path, column_overrides):
    """The HEAD and PHOT tables of a Rubin HATS catalogue of DiaObjects, and
    the values :func:`read_snana_fits` takes from a FITS header.

    ``column_overrides`` (PHOTOMETRY_COLUMNS) replaces Rubin columns by SNANA
    name. The light curve is a struct of lists sharing one set of offsets,
    which are the PTROBS pointers. Rubin's nJy fluxes are FLUXCAL at a zero
    point of 31.4.
    """
    head_names, phot_names = dict(RUBIN_HEAD_COLUMNS), dict(RUBIN_PHOT_COLUMNS)
    # an override replaces its entry in whichever table holds that name; a
    # name in neither is a KeyError
    holder = {**dict.fromkeys(head_names, head_names),
              **dict.fromkeys(phot_names, phot_names)}
    for name, column in column_overrides.items():
        holder[name][name] = column

    lc_fields = [*phot_names.values(), *RUBIN_PHOTFLAG_BITS]
    catalogue = pads.dataset(Path(path) / 'dataset', format='parquet')
    table = catalogue.to_table(columns={
        **{c: pads.field(c) for c in head_names.values()},
        **{f: pads.field(RUBIN_LIGHTCURVE, f) for f in lc_fields}})

    def epochs(field):
        values = table[field].combine_chunks().flatten()
        if not pa.types.is_string(values.type):
            return values.to_numpy(zero_copy_only=False)
        coded = values.dictionary_encode()
        return coded.dictionary.to_numpy(zero_copy_only=False).astype(str)[
            coded.indices.to_numpy()]

    offsets = table[phot_names['MJD']].combine_chunks().offsets.to_numpy()
    head = {name: table[c].to_numpy() for name, c in head_names.items()}
    head['SNID'] = head['SNID'].astype(str)
    head['PTROBS_MIN'] = offsets[:-1] + 1
    head['PTROBS_MAX'] = offsets[1:]
    phot = {name: epochs(c) for name, c in phot_names.items()}
    phot['PHOTFLAG'] = sum(bit * epochs(flag).astype(np.int32)
                           for flag, bit in RUBIN_PHOTFLAG_BITS.items())
    return Table(head), Table(phot), dict(
        is_sim=False, survey='LSST', snana_version='', zp_fluxcal=RUBIN_ZP_NJY,
        sim_mw_colour_law=None)


def load_and_batch_snana_fits(head_path: str, phot_path: str,
                              **kwargs) -> LightCurveBatch:
    """Load SNANA FITS files directly into a LightCurveBatch.

    Every numeric step is vectorised over numpy arrays, looping at most over
    bands or over sim-truth columns. The one per-SN loop assembles the FIELD
    label, which is string concatenation with no array form.

    ``**kwargs`` are the loader options, declared once on :func:`_build_batch`
    (filter_map, fit_bands, photflag_mskrej, epoch_windows, ...).
    """
    head_data, phot_data, header = read_snana_fits(head_path, phot_path)
    return batch_from_columns(head_data, phot_data, **header, **kwargs)


def batch_from_columns(header_columns, phot_columns, *, is_sim=True,
                       survey: str = '', snana_version: str = '',
                       zp_fluxcal: float, sim_mw_colour_law,
                       **kwargs) -> LightCurveBatch:
    """A batch from HEAD/PHOT columns already in memory: the tables
    :func:`read_snana_fits` returns, or what ``write_snana_outputs`` returns
    with no FITS round trip. Both go through the same ``_build_batch``, so
    the cuts, band mapping, sim-truth columns and epoch grouping are the ones
    parity validates. The five values a FITS header would supply are arguments
    here, since there is no header to read them from.
    """
    return _build_batch(Table(header_columns, copy=False),
                        Table(phot_columns, copy=False), is_sim,
                        survey=survey, snana_version=snana_version,
                        zp_fluxcal=zp_fluxcal,
                        sim_mw_colour_law=sim_mw_colour_law, **kwargs)


def resolve_mw_colour_law(data_config, sim_mw_colour_law):
    """(OPT_MWCOLORLAW, RV_MWCOLORLAW): each the user's where set, else the
    simulation's own from the data header, else the default for real data."""
    opt_file, rv_file = (sim_mw_colour_law if sim_mw_colour_law is not None
                         else (OPT_MWCOLORLAW_DEFAULT, RV_MWCOLORLAW_DEFAULT))
    opt = (opt_file if data_config.opt_mwcolorlaw is None
           else data_config.opt_mwcolorlaw)
    rv = (rv_file if data_config.rv_mwcolorlaw is None
          else data_config.rv_mwcolorlaw)
    if opt not in OPT_MWCOLORLAW_NAMES:
        raise SNANAError(
            f"Invalid SIMOPT_MWCOLORLAW = {opt} in the data header; valid "
            f"codes: {', '.join(str(c) for c in sorted(OPT_MWCOLORLAW_NAMES))}.",
            function_name="INIT_GALextinct")
    return opt, rv


def _build_batch(
    head_data, phot_data, is_sim, *, survey: str = '',
    zp_fluxcal: float, sim_mw_colour_law,
    filter_map: Optional[Dict[str, int]] = None,
    fit_bands: Optional[List[str]] = None,
    photflag_mskrej: tuple = (), photflag_detect: int = 0,
    filter_replace: Optional[dict] = None,
    private_varnames: tuple = (),
    fluxerrcalc_zpterr: float = -9.9,
    epoch_ignore: Optional[set] = None,
    use_snhost_zphot: bool = False, use_snhost_qzphot: bool = False,
    snfield_list: Optional[list] = None,
    setpkmjd: Optional[dict] = None,
    peak_shape=None,
    mwebv: Optional[dict] = None,
    epoch_windows: Optional[dict] = None, epoch_snrmin: Optional[dict] = None,
    read_zmax: float = CUTVAL_OPEN,
    earlylc: Optional[dict] = None,
    filter_lamavg: Optional[dict] = None, fudge_flux: Optional[dict] = None,
    fluxcal_baseline_correct: bool = False,
    dataerr_scale: float = 1.0, magcor_map: Optional[dict] = None,
    sim_magcor_map: Optional[dict] = None, nonlin_model=None,
    fluxerr_model=None, sim_fluxerr_model=None, hostnoise_maps=None,
    keep_obs_extras: bool = False,
    abort_on_badqzphot: bool = True,
    dofit_photoz: bool = False,
    quantile_spline: str = 'cubic',
    jobsplit: tuple = (1, 1),
    zshifts: Optional[Dict[str, float]] = None,
    ztol_helio2cmb: float = 0.02,
    magobs_shift_zp: Optional[Dict[str, float]] = None,
    vpec_err_override: float = 0.0,
    opt_vpec_cor: int = 1,
    restore_wrong_vpec: bool = False,
    snana_version: str = '',
    header_override=None, sim_header_override=None,
    model_sim_columns: tuple = (),
) -> LightCurveBatch:
    """Build a LightCurveBatch from HEAD/PHOT astropy Tables. Shared by the
    FITS and TEXT loaders: everything from the raw tables onward is
    format-agnostic (PTROBS spans, band/PHOTFLAG/field masks, epoch cuts,
    selection, batch assembly)."""
    override = sim_header_override if is_sim else header_override
    if override is not None:
        head_data = apply_header_override(head_data, override)
    n_sne = len(head_data)
    phot_mjd = np.array(phot_data['MJD'], dtype=np.float64)
    phot_fluxcal = np.array(phot_data['FLUXCAL'], dtype=np.float64)
    phot_fluxcalerr = np.array(phot_data['FLUXCALERR'], dtype=np.float64)

    if 'BAND' in phot_data.colnames:
        band_column = phot_data['BAND']
    elif 'FLT' in phot_data.colnames:
        band_column = phot_data['FLT']
    else:
        raise SNANAError("No BAND or FLT column in PHOT file",
                         function_name="_build_batch")

    # PTROBS pointers (1-indexed in SNANA)
    ptrobs_min = np.array(head_data['PTROBS_MIN'], dtype=np.int64) - 1
    ptrobs_max = np.array(head_data['PTROBS_MAX'], dtype=np.int64)

    obs_counts_raw = ptrobs_max - ptrobs_min  # [n_sne]
    total_obs = obs_counts_raw.sum()
    sn_idx_all = np.repeat(np.arange(n_sne), obs_counts_raw)  # [total_obs]
    group_starts = np.zeros(n_sne, dtype=np.int64)
    np.cumsum(obs_counts_raw[:-1], out=group_starts[1:])
    valid_offsets = np.arange(total_obs) - np.repeat(group_starts, obs_counts_raw)
    valid_phot_indices = np.repeat(ptrobs_min, obs_counts_raw) + valid_offsets

    # the epochs SNANA reads; nothing below sees the others
    head_cols = _head_metadata(head_data)
    read = _read_window(
        phot_mjd[valid_phot_indices], sn_idx_all, head_cols['peakmjds'],
        head_cols['read_redshifts'],
        (epoch_windows or {}).get('trest', (-CUTVAL_OPEN, CUTVAL_OPEN)),
        read_zmax)

    # NEPOCH: read epochs with a valid error, before epoch cuts (snana.F90:21285, :14926)
    stored_valid = read & (phot_fluxcalerr[valid_phot_indices] >= 0.0)
    nepoch_store_all = np.bincount(sn_idx_all[stored_valid],
                                   minlength=n_sne).astype(np.int32)

    unique_bands, band_codes = _label_codes(band_column, valid_phot_indices)

    # FILTER_REPLACE 'LIST1 -> LIST2': char-for-char band substitution as
    # the photometry is read (INIT_FILTER_REPLACE, snana.F90:12729).
    if filter_replace:
        short = np.char.rpartition(unique_bands, '-')
        prefix = np.where(short[:, 1] == '-', short[:, 0] + '-', '')
        band_c = np.where(short[:, 1] == '-', short[:, 2], unique_bands)
        for orig, repl in filter_replace.items():
            band_c = np.where(band_c == orig, repl, band_c)
        unique_bands, merged = np.unique(np.char.add(prefix, band_c),
                                         return_inverse=True)
        band_codes = merged[band_codes]

    if fit_bands is not None:
        # a band is the label's last character (INTFILTER, sntools.c:10330),
        # so 'g' and 'LSST-g' are one band, named here as the kcor names it
        by_short = {}
        for name in fit_bands:
            by_short.setdefault(filter_band(name), []).append(name)
        kcor_name = {short: names[0] for short, names in by_short.items()
                     if len(names) == 1}
        unique_bands, merged = np.unique(
            [b if b in fit_bands else kcor_name.get(b, b) for b in unique_bands],
            return_inverse=True)
        band_codes = merged[band_codes]
        fit_bands_set = set(fit_bands)
        keep_band = np.array([b in fit_bands_set for b in unique_bands])
        fit_band = keep_band[band_codes]
    else:
        keep_band = np.ones(len(unique_bands), dtype=bool)
        fit_band = np.ones(total_obs, dtype=bool)
    obs_keep_mask = read.copy()

    if 'FIELD' in phot_data.colnames:
        unique_fields, field_codes = _label_codes(phot_data['FIELD'],
                                                  valid_phot_indices)
    else:
        unique_fields = field_codes = None

    # SNFIELD_LIST is an epoch cut (snana.F90:14798, :13510)
    if snfield_list and 'ALL' not in snfield_list:
        if unique_fields is None:
            raise SNANAError(
                "SNFIELD_LIST configured but PHOT file has no FIELD "
                "column", function_name="_build_batch")
        obs_keep_mask &= np.isin(unique_fields, list(snfield_list))[field_codes]

    # EPOCH_IGNORE_FILE drops named (CID, MJD, band) epochs; the band is
    # a single character and the MJD match allows 0.002 d (snana.F90:13221).
    if epoch_ignore:
        head_cids = np.char.strip(np.array(head_data['SNID'], dtype='U'))
        # Only scan epochs belonging to SNe named in the ignore file.
        ignored_cids = [cid for cid, _, _ in epoch_ignore]
        rows = np.flatnonzero(np.isin(head_cids, ignored_cids)[sn_idx_all])
        ep_cids = head_cids[sn_idx_all[rows]]
        ep_mjds = np.asarray(phot_mjd[valid_phot_indices[rows]], dtype=float)
        ep_bands = np.array([filter_band(b) for b in unique_bands])[band_codes[rows]]
        drop = np.zeros(len(rows), dtype=bool)
        for cid, mjd, band in epoch_ignore:
            drop |= ((ep_cids == cid) & (ep_bands == band)
                     & (np.abs(ep_mjds - mjd) < 0.002))
        obs_keep_mask[rows[drop]] = False

    # mask 1 ORs, masks 2-5 AND (snana.F90:13461)
    mskrej = tuple(int(m) for m in photflag_mskrej)
    if any(mskrej):
        if 'PHOTFLAG' not in phot_data.colnames:
            raise SNANAError(
                "PHOTFLAG_MSKREJ configured but PHOT file has no "
                "PHOTFLAG column", function_name="_build_batch")
        photflags = np.array(phot_data['PHOTFLAG'],
                             dtype=np.int64)[valid_phot_indices]
        reject = np.zeros(total_obs, dtype=bool)
        if mskrej[0]:
            reject |= (photflags & mskrej[0]) > 0
        for m in mskrej[1:5]:
            if m:
                reject |= (photflags & m) == m
        obs_keep_mask &= ~reject

    if earlylc:
        obs_keep_mask &= _earlylc_mask(
            earlylc, sn_idx_all, n_sne,
            phot_mjd[valid_phot_indices], unique_bands[band_codes],
            phot_fluxcal[valid_phot_indices],
            phot_fluxcalerr[valid_phot_indices],
            phot_data, valid_phot_indices)

    cut_args = (epoch_windows, epoch_snrmin, head_data, phot_data, phot_mjd,
                phot_fluxcal, phot_fluxcalerr, valid_phot_indices, sn_idx_all,
                band_codes, unique_bands, photflag_detect, fluxerrcalc_zpterr,
                zp_fluxcal)
    obs_keep_mask = _epoch_quality_cuts(obs_keep_mask, *cut_args, at_read=True)
    # the OPT_SETPKMJD estimate sees every band (snana.F90:17224)
    peak_keep = obs_keep_mask.copy()
    obs_keep_mask &= fit_band
    obs_keep_mask = _epoch_quality_cuts(obs_keep_mask, *cut_args, at_read=False)

    kept_phot_indices = valid_phot_indices[obs_keep_mask]
    obs_counts_filtered, sn_idx_dense, obs_idx_within, max_obs_val = \
        _padded_slots(obs_keep_mask, sn_idx_all, n_sne)

    kept_band_codes = band_codes[obs_keep_mask]

    kept_flux = phot_fluxcal[kept_phot_indices]
    err1 = phot_fluxcalerr[kept_phot_indices]  # scaled-original error (ERR1)
    batch_flux_baselines = None
    if fluxcal_baseline_correct:
        # measured before the S/N cuts and fudges
        n_codes = len(unique_bands)
        sample = peak_keep & fit_band
        base, base_err = _flux_baselines(
            phot_fluxcal[valid_phot_indices[sample]],
            phot_fluxcalerr[valid_phot_indices[sample]],
            (sn_idx_all * n_codes + band_codes)[sample], n_sne * n_codes)
        kept_group = sn_idx_dense * n_codes + kept_band_codes
        kept_flux = kept_flux - base[kept_group]
        err1 = np.hypot(err1, base_err[kept_group])
        per_code = base.reshape(n_sne, n_codes)
        batch_flux_baselines = {filter_band(b): per_code[:, j]
                                for j, b in enumerate(unique_bands)
                                if keep_band[j]}
    add_var = 0.0
    if fudge_flux:
        kept_flux, err1, add_var = _apply_flux_fudges(
            kept_flux, err1, kept_band_codes, unique_bands,
            phot_data, kept_phot_indices, fudge_flux, is_sim)
    bands_e = np.array([filter_band(b) for b in unique_bands])[kept_band_codes]
    fields_e = (unique_fields[field_codes[obs_keep_mask]]
                if unique_fields is not None else None)

    kept_err = _combine_flux_errors(
        err1, add_var, kept_flux, bands_e, fields_e, sn_idx_dense,
        kept_phot_indices, head_data, phot_data, phot_mjd, is_sim,
        fudge_flux, dataerr_scale, fluxerr_model, sim_fluxerr_model,
        hostnoise_maps, zp_fluxcal)

    if nonlin_model is not None:  # NONLINEARITY_FILE (GET_NONLIN)
        col = lambda c: np.asarray(phot_data[c], dtype=np.float64)[kept_phot_indices]
        nea = noise_equivalent_aperture(col('PSF_SIG1'), col('PSF_SIG2'),
                                        col('PSF_RATIO'))
        if nonlin_model.optmask & OPTMASK_COUNT_RATE:
            if 'TEXPOSE' not in phot_data.colnames:
                raise SNANAError(
                    "NONLINEARITY count-rate mode (OPTMASK_NONLIN bit 2) "
                    "requires a TEXPOSE column", function_name="_build_batch")
            texpose = col('TEXPOSE')
        else:
            texpose = np.ones(len(kept_phot_indices))
        kept_flux = kept_flux * evaluate_nonlinearity(
            nonlin_model, kept_flux, col('ZEROPT'), col('GAIN'), nea,
            col('SKY_SIG'), texpose, bands_e, zp_fluxcal)

    # EXEC_MAGCOR: SIM_MAGCOR_FILE for sim data, MAGCOR_FILE otherwise.
    mm = sim_magcor_map if is_sim else magcor_map
    if mm:
        cids = np.char.strip(np.array(head_data['SNID'], dtype='U'))[sn_idx_dense]
        mjds_e = phot_mjd[kept_phot_indices]
        magcor = _magcor_values(mm, cids, mjds_e, bands_e)
        too_large = np.abs(magcor) > MAGCOR_MAX
        if too_large.any():
            k = int(np.argmax(too_large))
            raise SNANAError(
                f"MAGCOR = {magcor[k]:.4g} at {cids[k]}-{mjds_e[k]:.3f}-{bands_e[k]} "
                f"exceeds {MAGCOR_MAX} mag",
                function_name='load_and_batch_snana_fits')
        kept_flux = kept_flux * 10.0 ** (-0.4 * magcor)

    mjds_2d = np.zeros((n_sne, max_obs_val), dtype=np.float64)
    fluxcals_2d = np.zeros((n_sne, max_obs_val), dtype=np.float64)
    fluxcalerrs_2d = np.zeros((n_sne, max_obs_val), dtype=np.float64)
    filter_indices_2d = np.zeros((n_sne, max_obs_val), dtype=np.int32)

    mjds_2d[sn_idx_dense, obs_idx_within] = phot_mjd[kept_phot_indices]
    fluxcals_2d[sn_idx_dense, obs_idx_within] = kept_flux
    fluxcalerrs_2d[sn_idx_dense, obs_idx_within] = kept_err

    # MAGOBS_SHIFT_ZP: a per-band zero-point systematic, applied to the
    # data as a common scale on flux and error (snana.F90:17369).
    if magobs_shift_zp:
        shifts = np.asarray(filter_values(magobs_shift_zp, unique_bands, 0.0))
        scale = 10.0 ** (-0.4 * shifts[kept_band_codes])
        fluxcals_2d[sn_idx_dense, obs_idx_within] *= scale
        fluxcalerrs_2d[sn_idx_dense, obs_idx_within] *= scale

    # a copy: numbering an unlisted band must not touch the caller's mapping
    bands = dict(filter_map or {})
    for band in unique_bands:
        bands.setdefault(band, len(bands))

    filter_names = {v: k for k, v in bands.items()}

    code_to_fidx = np.array([bands[unique_bands[c]]
                             for c in range(len(unique_bands))], dtype=np.int32)
    filter_indices_2d[sn_idx_dense, obs_idx_within] = code_to_fidx[kept_band_codes]

    batch_obs_extras = None
    if keep_obs_extras:
        batch_obs_extras = {}
        for cname in ('FIELD', 'PHOTFLAG', 'DETNUM', 'XPIX', 'YPIX',
                      'ZEROPT', 'ZEROPT_ERR', 'PSF_SIG1', 'PSF_SIG2',
                      'PSF_RATIO', 'SKY_SIG', 'SKY_SIG_T', 'GAIN',
                      'SIM_MAGOBS'):
            if cname not in phot_data.colnames:
                continue
            raw = np.asarray(phot_data[cname])[kept_phot_indices]
            if raw.dtype.kind in 'US':
                raw = np.char.strip(raw.astype('U'))
                arr = np.full((n_sne, max_obs_val), '', dtype='U20')
            else:
                arr = np.zeros((n_sne, max_obs_val), dtype=raw.dtype)
            arr[sn_idx_dense, obs_idx_within] = raw
            batch_obs_extras[cname] = arr

    obs_range = np.arange(max_obs_val)[np.newaxis, :]
    observation_mask = obs_range < obs_counts_filtered[:, np.newaxis]

    redshifts = head_cols['redshifts']
    redshift_errs = head_cols['redshift_errs']
    ras = head_cols['ras']
    decs = head_cols['decs']
    # unset keeps the header's, SNANA's default OPT_MWEBV of 1
    mwebvs, mwebv_errs = (
        _resolve_mwebv(head_cols['mwebvs'], head_cols['mwebv_errs'], ras,
                       decs, mwebv)
        if mwebv else (head_cols['mwebvs'], head_cols['mwebv_errs']))
    peakmjds = head_cols['peakmjds']
    has_peakmjd = head_cols['has_peakmjd']
    snids = head_cols['snids']

    # the mean observed MJD; NaN for an event with no observation, which
    # the epoch cuts remove
    n_obs = observation_mask.sum(axis=1)
    reference_mjds = np.where(
        n_obs > 0, np.where(observation_mask, mjds_2d, 0.0).sum(axis=1)
        / np.maximum(n_obs, 1), np.nan)
    mjd_offsets = mjds_2d - reference_mjds[:, np.newaxis]
    peakmjd_offsets = (peakmjds - reference_mjds) if has_peakmjd else None

    batch_sim_truth = _sim_truth_columns(head_data, is_sim, model_sim_columns)

    event_fields = None
    if unique_fields is not None:
        event_fields = _event_fields(unique_fields, field_codes, sn_idx_all,
                                     n_sne, survey)
    batch_head_extras = _head_extras(head_data, private_varnames, event_fields,
                                     peakmjds if has_peakmjd else None)

    # HOSTGAL* columns (raw; mapped at FITRES write)
    batch_host_data = {
        col: np.array(head_data[col])
        for col in head_data.colnames
        if col.startswith('HOSTGAL') and not col.startswith('HOSTGAL_ZPHOT_Q')
    } or None

    _check_helio2cmb(head_data, ras, decs, snids, ztol_helio2cmb)
    zcols = _resolve_redshifts(
        head_data, redshifts, redshift_errs, ras, decs, snids,
        batch_head_extras, abort_on_badqzphot=abort_on_badqzphot,
        dofit_photoz=dofit_photoz, quantile_spline=quantile_spline,
        use_snhost_zphot=use_snhost_zphot,
        use_snhost_qzphot=use_snhost_qzphot, zshifts=zshifts)
    redshifts = zcols['redshifts']
    redshift_errs = zcols['redshift_errs']
    if override is not None and LOGMASS_ROW_COLUMNS[0] in override[1]:
        _set_logmass(batch_host_data, redshifts, redshift_errs, dofit_photoz)
    photoz = zcols['photoz']
    photoz_err = zcols['photoz_err']
    zpdf_quantiles = zcols['zpdf_quantiles']
    zpdf_percentiles = zcols['zpdf_percentiles']
    zpdf_nbin = zcols['zpdf_nbin']
    redshift_source_mask = zcols['redshift_source_mask']
    zcmb = zcols['zcmb']

    _apply_vpec_corrections(batch_head_extras, restore_wrong_vpec,
                            snana_version, opt_vpec_cor,
                            vpec_err_override)

    batch = LightCurveBatch(
        snids=snids,
        redshifts=np.asarray(redshifts),
        redshift_errs=np.asarray(redshift_errs),
        mwebvs=np.asarray(mwebvs),
        mwebv_errs=np.asarray(mwebv_errs),
        n_obs_per_sn=obs_counts_filtered,
        reference_mjds=reference_mjds,
        mjd_offsets=mjd_offsets,
        filter_indices=filter_indices_2d,
        fluxcals=fluxcals_2d,
        fluxcalerrs=fluxcalerrs_2d,
        observation_mask=observation_mask,
        filter_names=filter_names,
        peakmjd_offsets=peakmjd_offsets,
        sim_truth=batch_sim_truth,
        host_data=batch_host_data,
        head_extras=batch_head_extras,
        obs_extras=batch_obs_extras,
        flux_baselines=batch_flux_baselines,
        survey=survey,
        zp_fluxcal=zp_fluxcal,
        sim_mw_colour_law=sim_mw_colour_law,
        photoz=photoz,
        photoz_err=photoz_err,
        zpdf_quantiles=zpdf_quantiles,
        zpdf_percentiles=zpdf_percentiles,
        zpdf_nbin=zpdf_nbin,
        redshift_source_mask=redshift_source_mask,
        ras=ras,
        decs=decs,
        errflag_fit=np.full(len(snids), ERRFLAG_NO_FIT),
        nepoch_store=nepoch_store_all,
        zcmb=zcmb,
    )

    if setpkmjd:
        estimated = _set_peakmjd(_peak_view(
            batch, peak_keep, sn_idx_all, phot_mjd[valid_phot_indices],
            phot_fluxcal[valid_phot_indices],
            phot_fluxcalerr[valid_phot_indices], code_to_fidx[band_codes]),
            setpkmjd, peak_shape)
        batch = replace(batch, peakmjd_offsets=estimated.peakmjd_offsets,
                        head_extras=estimated.head_extras)
    batch = _apply_post_peak_epoch_cuts(batch, epoch_windows,
                                        filter_lamavg)
    return _apply_jobsplit(batch, jobsplit)

def load_and_batch_snana_text(version_dir: str, **kwargs) -> LightCurveBatch:
    """Load a SNANA TEXT-format data version (a directory holding a
    ``<VERSION>.LIST`` and one ``<SN>.DAT`` per SN) into a LightCurveBatch,
    via the same builder as the FITS loader, so it takes the same loader
    options, declared once on :func:`_build_batch` (fit_bands,
    photflag_mskrej, epoch_windows, ...)."""
    version_dir = Path(version_dir)
    list_files = list(version_dir.glob('*.LIST'))
    if not list_files:
        raise SNANAError(f"No .LIST file in {version_dir}",
                         function_name="load_and_batch_snana_text")
    dat_names = [ln.strip() for ln in list_files[0].read_text().splitlines()
                 if ln.strip() and not ln.lstrip().startswith('#')]

    def hval(hdr, key, idx=0):
        s = hdr.get(key)
        if s is None:
            return None
        toks = s.split()
        return toks[idx] if idx < len(toks) else None

    heads, nobs, phot = [], [], {}
    is_sim = False
    for name in dat_names:
        header, varlist, obs = parse_snana_dat(version_dir / name)
        heads.append(header)
        fake = hval(header, 'FAKE')
        is_sim = is_sim or (fake is not None and int(float(fake)) == 2)
        nobs.append(len(obs.get('MJD', [])))
        for col in varlist:
            phot.setdefault(col, []).extend(obs[col])
    zp = hval(heads[0], 'ZP_FLUXCAL') if heads else None
    kwargs['zp_fluxcal'] = float(zp) if zp is not None else ZEROPOINT_FLUXCAL_SNANA_ORIG
    kwargs['sim_mw_colour_law'] = None
    if is_sim:
        missing = [k for k in SIM_MW_COLOUR_LAW_KEYS
                   if hval(heads[0], k) is None]
        if missing:
            raise SNANAError(f"{version_dir / dat_names[0]} lacks "
                             f"{', '.join(missing)}",
                             function_name="load_and_batch_snana_text")
        kwargs['sim_mw_colour_law'] = (
            int(hval(heads[0], 'SIMOPT_MWCOLORLAW')),
            float(hval(heads[0], 'SIM_MWRV')))

    # HEAD table: PTROBS spans are the cumulative per-SN obs counts
    starts = np.concatenate([[0], np.cumsum(nobs)[:-1]]).astype(np.int64)
    _f = lambda h, k, i=0: float(hval(h, k, i) or 'nan')
    present = lambda k, i=0: any(hval(h, k, i) is not None for h in heads)
    head_cols = {
        'SNID': [str(hval(h, 'SNID')) for h in heads],
        'PTROBS_MIN': starts + 1,
        'PTROBS_MAX': starts + np.asarray(nobs, dtype=np.int64),
    }
    # only add present keywords, so the builder's column fallbacks
    # (REDSHIFT_FINAL->HELIO, PEAKMJD->SIM_PEAKMJD) still apply
    for out, key, idx in [
        ('RA', 'RA', 0), ('DEC', 'DEC', 0),
        ('REDSHIFT_FINAL', 'REDSHIFT_FINAL', 0),
        ('REDSHIFT_FINAL_ERR', 'REDSHIFT_FINAL', 2),
        ('REDSHIFT_HELIO', 'REDSHIFT_HELIO', 0),
        ('MWEBV', 'MWEBV', 0), ('PEAKMJD', 'PEAKMJD', 0),
        ('PIXSIZE', 'PIXSIZE', 0),
    ]:
        if present(key, idx):
            head_cols[out] = [_f(h, key, idx) for h in heads]
    if present('SNTYPE'):
        head_cols['SNTYPE'] = [int(float(hval(h, 'SNTYPE'))) for h in heads]
    if is_sim:
        for col in (*FITRES_SIM_COLUMNS, *kwargs.get('model_sim_columns', ())):
            if present(col):
                head_cols[col] = [_f(h, col) for h in heads]
    # scalar HOSTGAL* header keys (per-filter DAT mag lists not supported)
    for col in sorted({k for h in heads for k in h
                       if k.startswith('HOSTGAL')}):
        if present(col):
            head_cols[col] = [_f(h, col) for h in heads]
    head_data = Table(head_cols)

    # PHOT table: rename TEXT columns to the FITS names the builder uses
    rename = {'ZPT': 'ZEROPT', 'PSF': 'PSF_SIG1'}
    str_cols = {'BAND', 'FLT', 'FIELD'}
    phot_cols = {}
    for col, vals in phot.items():
        name = rename.get(col, col)
        if col in str_cols:
            phot_cols[name] = np.array(vals, dtype='U')
        elif col == 'PHOTFLAG':
            phot_cols[name] = np.array([int(float(v)) for v in vals])
        else:
            phot_cols[name] = np.array([float(v) for v in vals])

    return _build_batch(head_data, Table(phot_cols), is_sim, **kwargs)
