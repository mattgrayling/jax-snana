"""Loading light curves for fitting: finding a data version and its files
(`discover_inputs`, `version_sources`), reading SNANA FITS and TEXT data and
Rubin HATS catalogues, and building the padded `LightCurveBatch` from them
(`batch_from_columns`). Building a batch resolves each event's redshifts and
peculiar velocity, applies the header overrides, the epoch cuts and the
photometric corrections (from `calibration`, in SNANA's order) and makes the
PEAKMJD estimate. `prepare_batch` then makes it fittable: the event selection,
the redshift priors and the SNCID_LIST_FILE starting values. Tables supplied
in memory through the Python interface take the same path
(`batch_from_tables`).

Entry points:
    - load: every data version the config names, as the snlc_fit command
      reads them
    - batch_from_columns: HEAD and PHOT tables already read
    - batch_from_tables: photometry and object tables supplied in memory
    - read_shared_files: the files every data version of a config shares
"""

import dataclasses
import glob
import logging
import os
import re
import time
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
from astropy.table import Table
from astropy.io import fits

from jax_snlc_core import logging as log
from jax_snlc_core.cosmology import zcmb_to_zhelio, zhelio_to_zcmb
from jax_snlc_core.filters import ZEROPOINT_FLUXCAL_SNANA_ORIG, filter_band
from jax_snlc_core.interpolation import hermite_spline_at, natural_spline_slopes, spline_knots
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.models import MODEL_REGISTRY, metadata_files
from jax_snlc_core.noise import read_fluxerr_model_file
from jax_snlc_core.peak_time import (OPTMASK_SETPKMJD_FLUXMAX3, OPTMASK_SETPKMJD_NOABORT,
                                     PEAKMJD_MISSING, bazin_peakmjd, fmax_clump_peakmjd,
                                     naive_fluxmax_peakmjd)
from jax_snlc_core.snana_files import (HOSTGAL_PREFIXES, SNANA_NULLFLOAT, MXBIN_HOSTGALz,
                                       MXBIN_HOSTGALz_QUANTILE, check_file_docana,
                                       is_ignorefile, load_survey_fields, read_fits_table,
                                       read_simulation_cosmology, resolve_fits_path,
                                       skip_docana, sndata_root, open_text,
                                       HOSTGAL_NO_HOST_NULL, hostgal_null_key)
from jax_snlc_core.utils import CUTVAL_OPEN

from .batch import (ERRFLAG_FIT_CUTS, ERRFLAG_FITPREP_NEPOCH, ERRFLAG_FITPREP_NOEPOCHS,
                    ERRFLAG_FITPREP_QUANTILES, ERRFLAG_NO_FIT, LightCurveBatch,
                    choose_epoch_lengths, group_by_epoch_length)
from .calibration import (apply_flux_fudges, combined_flux_errors,
                          fit_photometry, flux_baselines, magcor_scales, mwebv_options,
                          nonlinearity_scales, read_hostnoise_file, read_magcor_file,
                          read_nonlinearity_file, resolve_mwebv, resolved_filters,
                          shift_map_with_poly, zero_point_shift_scales)
from .config_parser import fitted_model_type
from .cuts import (build_selection, cid_keys, cut_windows, drop_unfittable,
                   early_lightcurve_mask, epoch_quality_cuts, mark_errflags,
                   post_peak_epoch_cuts, read_epoch_mask, read_epoch_window,
                   select_events, sncid_list)
from .parameters import fit_parameters
from .photoz import (ZSPEC_SOURCE_MASK, pdf_support_mask, precompute_zpdf_spline,
                     quantile_prior_mask, redshift_priors, zpdf_rows_usable)

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

# Missing HEAD values, shared by FITS loading and unions of TEXT event headers.
# NaN marks unknown sky coordinates and a peak time that must be estimated.
_HEAD_DEFAULTS = {
    'MWEBV': 0., 'MWEBV_ERR': 0.,
    'VPEC': 0., 'VPEC_ERR': 0., 'LENSDMU': 0., 'LENSDMU_ERR': 0.,
    'REDSHIFT_HELIO_ERR': .001, 'REDSHIFT_FINAL_ERR': .001, 'REDSHIFT_CMB_ERR': .001,
    'PEAKMJD': np.nan, 'SIM_PEAKMJD': np.nan,
    'RA': np.nan, 'DEC': np.nan, 'DECL': np.nan,
    'MJD_TRIGGER': MJD_TRIGGER_MISSING,
}

# PKMJD carried by an event the Bazin fit could not place (snana.F90:26654)
PKMJD_NOFIT = -1.0


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
    log.warn(logger, "SNHOST_QZPHOT", message)


def _head_metadata(head_data):
    """Per-SN HEAD scalars, with SNANA's column fallbacks."""
    def get_column(table, names):
        for name in names:
            if name in table.colnames:
                return np.array(table[name], dtype=np.float64)
        return np.full(len(table), _HEAD_DEFAULTS.get(names[0], SNANA_NULLFLOAT),
                       dtype=np.float64)

    mwebvs = get_column(head_data, ['MWEBV'])
    # MWEBV_ERR defaults to XTMW_FRACERR = 16% of MWEBV; the header column
    # overrides that only above 1e-12 (snana.F90:3908-3916).
    mwebv_errs = get_column(head_data, ['MWEBV_ERR'])
    peakmjds = get_column(head_data, ['PEAKMJD', 'SIM_PEAKMJD'])
    # the fit uses zHELIO where defined, else zCMB (snana.F90:4256, :4292),
    # and zCMB itself falls back to REDSHIFT_CMB where undefined (:4277)
    zhel = get_column(head_data, ['REDSHIFT_HELIO'])
    zcmb = get_column(head_data, ['REDSHIFT_FINAL'])
    zcmb_err = get_column(head_data, ['REDSHIFT_FINAL_ERR'])
    alt = zcmb < 0.0
    zcmb = np.where(alt, get_column(head_data, ['REDSHIFT_CMB']), zcmb)
    zcmb_err = np.where(
        alt, get_column(head_data, ['REDSHIFT_CMB_ERR']), zcmb_err)
    ras = get_column(head_data, ['RA'])
    decs = get_column(head_data, ['DEC', 'DECL'])
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
            use_hel, get_column(head_data, ['REDSHIFT_HELIO_ERR']),
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
        log.warn(logger, "SIM_TRUTH",
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
    for f in load_survey_fields(survey):
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
                    'VPEC', 'VPEC_ERR', 'LENSDMU', 'LENSDMU_ERR', 'MJD_TRIGGER',
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
        log.line(logger, f"VPEC sign flipped: data written by SNANA "
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


def _padded_slots(keep, sn_idx_all, n_sne):
    """Per-event counts, each kept epoch's event and slot, and the width of
    the padded arrays."""
    counts = np.bincount(sn_idx_all[keep], minlength=n_sne).astype(np.int32)
    starts = np.zeros(n_sne, dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])
    slot = np.arange(int(keep.sum())) - np.repeat(starts, counts)
    width = int(counts.max()) if n_sne > 0 else 0
    return counts, sn_idx_all[keep], slot, width


def _host_photoz_columns(head_data, abort_on_badqzphot):
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
            function_name="ZTOL_HELIO2CMB")


def _resolve_redshifts(head_data, redshifts, redshift_errs, ras, decs,
                       snids, batch_head_extras, *, abort_on_badqzphot,
                       quantile_spline, use_snhost_zphot,
                       use_snhost_qzphot, zshifts):
    """Host photo-z columns, the zCMB consistency check, the host-z
    substitution and REDSHIFT_*_SHIFT.

    Rewrites the REDSHIFT_HELIO entries of ``batch_head_extras`` where the
    substitution applies; returns the per-SN redshift columns."""
    (photoz, photoz_err, zpdf_quantiles, zpdf_percentiles, zpdf_nbin,
     redshift_source_mask) = _host_photoz_columns(
        head_data, abort_on_badqzphot)
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

    Bit 0 (=1) Bazin09, bit 3 (=8) the flux maximum, bit 4 (=16) or 5 (=32)
    the Fmax clump, and bit 6 (=64) the MJD_TRIGGER header column, used only
    where no flux-maximum option is set. The Fmax clump falls back to the
    header value where it finds no peak; an event the Bazin fit cannot
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
                        snrcut_backup=setpkmjd['snrcut_backup'],
                        log_snr_weight=setpkmjd['log_snr_weight'], mjd_span=span,
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
        log.line(logger, "Running Bazin pre-fit...")
        # each chunk's fits come back to the host before the next starts, so
        # asynchronous dispatch cannot queue them all
        def chunk_peaks(rows):
            return [np.asarray(out) for out in bazin_peakmjd(
                batch.mjd_offsets[rows], batch.fluxcals[rows], batch.fluxcalerrs[rows],
                batch.filter_indices[rows], batch.observation_mask[rows],
                len(batch.filter_names), setpkmjd['cutwin_tobs'],
                jnp.array(new_off[rows]))]
        fits = [chunk_peaks(slice(start, start + size))
                for start in range(0, batch.n_sne, size)]
        peak_off, failed, fallback = (np.concatenate(column) for column in zip(*fits))
        log.line(logger, f"Bazin pre-fit: {int((~fallback).sum())}/{batch.n_sne} SNe "
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
        # SNANA's label 800 shifts every Bazin result, the -1 included
        # (snana.F90:26793), and no other estimate
        new_off += setpkmjd['shift']

    # SET_PEAKMJD overwrites SNLC_SEARCH_PEAKMJD with its estimate
    # (snana.F90:26797), which is what the FITRES reports as PKMJDINI
    head_extras = dict(batch.head_extras)
    head_extras['PEAKMJD'] = new_off + reference_mjds
    return replace(batch, peakmjd_offsets=new_off, head_extras=head_extras)


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
            m = hermite_spline_at(knots_val, val, slopes_val, zc)
            shift = 0.5 * (np.abs(m - hermite_spline_at(knots_val, val, slopes_val, zc - dz))
                           + np.abs(m - hermite_spline_at(knots_val, val, slopes_val, zc + dz)))
            e = np.hypot(hermite_spline_at(knots_err, err, slopes_err, zc), shift)
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


def parse_snana_dat(path, *, header_only=False):
    """Parse one SNANA TEXT light-curve file (<SN>.DAT): ``KEY: value`` header
    keywords (comments after ``#`` stripped) and the ``VARLIST:`` / ``OBS:``
    photometry table. Returns (header dict, varlist, {col: [str values]}).
    ``header_only`` stops at the photometry table for survey discovery."""
    header, varlist, obs = {}, None, {}
    key = None
    with open_text(path) as fh:
        for raw in fh:
            line = raw.split('#', 1)[0].strip()
            if not line:
                continue
            if line.startswith('VARLIST:'):
                if header_only:
                    break
                varlist = line.split()[1:]
                obs = {c: [] for c in varlist}
            elif line.startswith('OBS:'):
                for col, val in zip(varlist, line.split()[1:]):
                    obs[col].append(val)
            elif line.startswith(('END', 'DETECTION', 'SPECTRUM', 'NSPECTRA')):
                continue
            elif ':' in line:
                key, rest = line.split(':', 1)
                key = key.strip()
                header[key] = rest.strip()
            elif varlist is None and key is not None:
                header[key] += ' ' + line
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
    head.convert_bytestring_to_unicode()

    _, phot = read_fits_table(resolve_fits_path(phot_path))

    return head, phot, dict(
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
    """Build a LightCurveBatch from HEAD/PHOT astropy Tables. Shared by every
    input, FITS, TEXT, HATS and in-memory tables: everything from the raw
    tables onward is format-agnostic (PTROBS spans, band/PHOTFLAG/field
    masks, epoch cuts, selection, batch assembly)."""
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
    read = read_epoch_window(
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

    obs_keep_mask = read_epoch_mask(
        obs_keep_mask, snfield_list, unique_fields, field_codes,
        epoch_ignore, head_data, sn_idx_all, phot_mjd, valid_phot_indices,
        unique_bands, band_codes, photflag_mskrej, phot_data)

    if earlylc:
        obs_keep_mask &= early_lightcurve_mask(
            earlylc, sn_idx_all, n_sne,
            phot_mjd[valid_phot_indices], unique_bands[band_codes],
            phot_fluxcal[valid_phot_indices],
            phot_fluxcalerr[valid_phot_indices],
            phot_data, valid_phot_indices)

    cut_args = (epoch_windows, epoch_snrmin, head_data, phot_data, phot_mjd,
                phot_fluxcal, phot_fluxcalerr, valid_phot_indices, sn_idx_all,
                band_codes, unique_bands, photflag_detect, fluxerrcalc_zpterr,
                zp_fluxcal)
    obs_keep_mask = epoch_quality_cuts(obs_keep_mask, *cut_args, at_read=True)
    # the OPT_SETPKMJD estimate sees every band (snana.F90:17224)
    peak_keep = obs_keep_mask.copy()
    obs_keep_mask &= fit_band
    obs_keep_mask = epoch_quality_cuts(obs_keep_mask, *cut_args, at_read=False)

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
        base, base_err = flux_baselines(
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
        kept_flux, err1, add_var = apply_flux_fudges(
            kept_flux, err1, kept_band_codes, unique_bands,
            phot_data, kept_phot_indices, fudge_flux, is_sim)
    bands_e = np.array([filter_band(b) for b in unique_bands])[kept_band_codes]
    fields_e = (unique_fields[field_codes[obs_keep_mask]]
                if unique_fields is not None else None)

    kept_err = combined_flux_errors(
        err1, add_var, kept_flux, bands_e, fields_e, sn_idx_dense,
        kept_phot_indices, head_data, phot_data, phot_mjd, is_sim,
        fudge_flux, dataerr_scale, fluxerr_model, sim_fluxerr_model,
        hostnoise_maps, zp_fluxcal)

    if nonlin_model is not None:
        kept_flux = kept_flux * nonlinearity_scales(
            nonlin_model, kept_flux, phot_data, kept_phot_indices, bands_e, zp_fluxcal)

    # EXEC_MAGCOR: SIM_MAGCOR_FILE for sim data, MAGCOR_FILE otherwise.
    corrections = sim_magcor_map if is_sim else magcor_map
    if corrections:
        kept_flux = kept_flux * magcor_scales(
            corrections, np.char.strip(np.array(head_data['SNID'], dtype='U'))[sn_idx_dense],
            phot_mjd[kept_phot_indices], bands_e)

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
        scale = zero_point_shift_scales(magobs_shift_zp, unique_bands)[kept_band_codes]
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
        resolve_mwebv(head_cols['mwebvs'], head_cols['mwebv_errs'], ras,
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
        quantile_spline=quantile_spline,
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
    batch = post_peak_epoch_cuts(batch, epoch_windows, filter_lamavg)
    return _apply_jobsplit(batch, jobsplit)


def _text_head_columns(header):
    """Expand TEXT error pairs, filter lists and host grids into HEAD columns."""
    bands = header.get('FILTERS', '').strip()
    per_band = {'MWXT_MAG', 'RA_AVG', 'DEC_AVG', 'SIM_GALFRAC',
                'SIM_PEAKMAG', 'SIM_EXPOSURE', 'HOSTGAL_SB_FLUXCAL',
                'HOSTGAL_SB_FLUXCAL_ERR'}
    per_band.update(f'{prefix}_{suffix}' for prefix in HOSTGAL_PREFIXES
                    for suffix in ('MAG', 'MAGERR'))
    grids = {f'{prefix}z_{suffix}': (f'{prefix}z_NBIN_{count}', size)
             for prefix in HOSTGAL_PREFIXES
             for count, suffixes, size in (
                 ('QUANTILE_ZPHOT', ('QUANTILE_ZPHOT', 'QUANTILE_PERCENT'),
                  MXBIN_HOSTGALz_QUANTILE),
                 ('LOGMASS_ZGRID', ('LOGMASS_ZGRID', 'LOGMASS_VALGRID', 'LOGMASS_ERRGRID'),
                  MXBIN_HOSTGALz))
             for suffix in suffixes}
    row = {}
    for name, value in header.items():
        if name in ('SNID', 'FIELD', 'NAME_TRANSIENT', 'NAME_IAUC', 'IAUC', 'SUBSURVEY'):
            row[name] = value
        elif name == 'SURVEY':
            if '(' in value:
                row['SUBSURVEY'] = value.split('(', 1)[1].rstrip(')')
        elif name in ('FILTERS', 'SNANA_VERSION', 'SIM_MODEL_NAME', 'SIM_TYPE_NAME', 'TRIGGER',
                      'VARNAMES_SPEC', 'SPEC', 'SIM_SYNMAG'):
            continue
        elif name in per_band:
            values = value.split()
            if len(values) != len(bands):
                raise SNANAError(f"{name} has {len(values)} values for FILTERS={bands}",
                                 function_name='read_snana_text')
            row.update((f'{name}_{band}', float(v)) for band, v in zip(bands, values))
        elif name in grids:
            values = np.asarray(value.split(), dtype=float)
            count, size = grids[name]
            row[name] = np.full(size, SNANA_NULLFLOAT)
            row[name][:len(values)] = values
        elif '+-' in value:
            number, error = value.split('+-')
            row[name] = float(number)
            row[f'{name}_ERR'] = float(error)
        elif name.startswith('SIM_HOSTLIB('):
            # TEXT lists the true host and neighbours; FITS stores the true host.
            row[name] = float(value.split()[0])
        else:
            # Preserve scalar numeric metadata, including user-defined columns.
            # IDs must not pass through float, which loses large integer bits.
            try:
                row[name] = int(value)
            except ValueError:
                row[name] = float(value)
    for name, (count, size) in grids.items():
        if row.get(count) == 0:
            row[name] = np.full(size, SNANA_NULLFLOAT)
    return row


def read_snana_text(paths):
    """Read a TEXT version's light curves as HEAD/PHOT tables and metadata.

    Like FITS and HATS readers, this leaves fitting options and batch
    construction to the common loading path.
    """
    def hval(hdr, key, idx=0):
        s = hdr.get(key)
        if s is None:
            return None
        toks = s.split()
        return toks[idx] if idx < len(toks) else None

    heads, nobs, phot = [], [], []
    rename = {'FLT': 'BAND', 'ZPT': 'ZEROPT', 'PSF': 'PSF_SIG1'}
    is_sim = False
    for path in paths:
        header, _, obs = parse_snana_dat(path)
        obs = {rename.get(name, name): values for name, values in obs.items()}
        missing = {'MJD', 'BAND', 'FLUXCAL', 'FLUXCALERR'} - obs.keys()
        if missing:
            raise SNANAError(f"{path} lacks photometry columns {sorted(missing)}",
                             function_name="read_snana_text")
        heads.append(header)
        fake = hval(header, 'FAKE')
        is_sim = is_sim or (fake is not None and int(float(fake)) == 2)
        nobs.append(len(obs['MJD']))
        phot.append(obs)
    zp = hval(heads[0], 'ZP_FLUXCAL') if heads else None
    zp_fluxcal = float(zp) if zp is not None else ZEROPOINT_FLUXCAL_SNANA_ORIG
    sim_mw_colour_law = None
    if is_sim:
        missing = [k for k in SIM_MW_COLOUR_LAW_KEYS
                   if hval(heads[0], k) is None]
        if missing:
            raise SNANAError(f"{paths[0]} lacks "
                             f"{', '.join(missing)}",
                             function_name="read_snana_text")
        sim_mw_colour_law = (
            int(hval(heads[0], 'SIMOPT_MWCOLORLAW')),
            float(hval(heads[0], 'SIM_MWRV')))

    # Convert TEXT values into the same event columns as FITS before batching.
    rows = [_text_head_columns(header) for header in heads]
    columns = dict.fromkeys(name for row in rows for name in row)
    head_cols = {}
    for name in columns:
        example = next(row[name] for row in rows if name in row)
        if isinstance(example, str):
            missing = ''
        elif 'NBIN_' in name:
            missing = 0
        elif name.startswith('HOSTGAL'):
            key = re.sub(r'^HOSTGAL[2-9]', 'HOSTGAL', name)
            key = hostgal_null_key(key, heads[0].get('FILTERS', ''))
            missing = np.full_like(example, HOSTGAL_NO_HOST_NULL.get(key, SNANA_NULLFLOAT))
        else:
            missing = _HEAD_DEFAULTS.get(name, SNANA_NULLFLOAT)
        head_cols[name] = [row.get(name, missing) for row in rows]
    starts = np.concatenate([[0], np.cumsum(nobs)[:-1]]).astype(np.int64)
    head_cols['PTROBS_MIN'] = starts + 1
    head_cols['PTROBS_MAX'] = starts + np.asarray(nobs, dtype=np.int64)
    head_data = Table(head_cols)

    # Optional columns may differ between events. Missing flags mean no flags;
    # other absent measurements remain unknown rather than acquiring a value.
    str_cols = {'BAND', 'FIELD'}
    phot_cols = {}
    for name in dict.fromkeys(name for obs in phot for name in obs):
        missing = '' if name in str_cols else 0 if name == 'PHOTFLAG' else np.nan
        values = [value for obs, count in zip(phot, nobs)
                  for value in obs.get(name, [missing] * count)]
        if name in str_cols:
            phot_cols[name] = np.array(values, dtype='U')
        elif name == 'PHOTFLAG':
            phot_cols[name] = np.array([int(float(v)) for v in values])
        else:
            phot_cols[name] = np.array(values, dtype=float)

    return head_data, Table(phot_cols), dict(
        is_sim=is_sim, survey=(hval(heads[0], 'SURVEY') or '').split('(', 1)[0],
        snana_version=hval(heads[0], 'SNANA_VERSION') or '',
        zp_fluxcal=zp_fluxcal, sim_mw_colour_law=sim_mw_colour_law)


# the file marking a directory as a HATS catalogue
HATS_PROPERTIES = 'hats.properties'


def version_files(sim_dir):
    """Locate a version's HATS directory, TEXT light curves, or FITS HEAD files."""
    sim_dir = Path(sim_dir)
    if (sim_dir / HATS_PROPERTIES).is_file():
        return [sim_dir]
    lists = sorted(sim_dir.glob('*.LIST'))
    if len(lists) > 1:
        raise SNANAError(f"{sim_dir} holds more than one .LIST: {lists}",
                         function_name="version_files")
    if lists:
        names = [line.split('#', 1)[0].strip()
                 for line in lists[0].read_text().splitlines()]
        names = [name for name in names if name]
        if not names:
            raise SNANAError(f"{lists[0]} contains no data files",
                             function_name="version_files")
        fits_names = [name.endswith(('_HEAD.FITS', '_HEAD.FITS.gz'))
                      for name in names]
        if any(fits_names) and not all(fits_names):
            raise SNANAError(f"{lists[0]} mixes FITS and TEXT files",
                             function_name="version_files")
        return [Path(resolve_fits_path(sim_dir / name)) for name in names]
    files = sorted(sim_dir.glob('*_HEAD.FITS*'))
    if not files:
        raise SNANAError(f"No .LIST or *_HEAD.FITS files found in {sim_dir}",
                         function_name="version_files")
    return files


def version_survey(sim_dir):
    """Read only the survey metadata, using the normal input-file discovery."""
    path = version_files(sim_dir)[0]
    if path.is_dir():
        return 'LSST'  # Rubin HATS, as in read_rubin_hats.
    if path.name.endswith(('_HEAD.FITS', '_HEAD.FITS.gz')):
        header = fits.getheader(path, 0)
    else:
        header, _, _ = parse_snana_dat(path, header_only=True)
    survey = str(header.get('SURVEY', '')).strip().split('(', 1)[0]
    if not survey:
        raise SNANAError(f"{path} names no SURVEY in its header",
                         function_name='version_survey')
    return survey


def version_sources(sim_dir, column_overrides):
    """A version's ``(path, read)`` sources yielding HEAD/PHOT and metadata."""
    files = version_files(sim_dir)
    if files[0].is_dir():
        return [(str(files[0]), partial(read_rubin_hats, files[0], column_overrides))]
    if not files[0].name.endswith(('_HEAD.FITS', '_HEAD.FITS.gz')):
        return [(str(sim_dir), partial(read_snana_text, files))]
    sources = []
    for head_path in files:
        prefix = head_path.name.removesuffix('.gz').removesuffix('_HEAD.FITS')
        phot_path = resolve_fits_path(head_path.parent / f"{prefix}_PHOT.FITS")
        if not Path(phot_path).exists():
            raise SNANAError(f"{head_path} has no {prefix}_PHOT.FITS beside it",
                             function_name="version_sources")
        sources.append((str(head_path), partial(read_snana_fits, head_path, phot_path)))
    return sources


def resolve_version_dir(version: str,
                        private_data_path: Optional[str] = None) -> str:
    """Locate the VERSION_PHOTOMETRY data dir
    (getInfo_PHOTOMETRY_VERSION, sntools.c:5440): an SNANA version, or a
    Rubin HATS catalogue."""
    root = sndata_root()
    private = (None if is_ignorefile(private_data_path)
               else Path(os.path.expandvars(private_data_path)))

    dirs = []
    if private:
        dirs.append(private)
        dirs.append(private / version)
    else:
        dirs.append(root / "lcmerge" / version)
        if "_" in version:
            prefix = version.split("_")[0]
            dirs.append(root / "lcmerge" / prefix / version)

    sim_paths = []
    list_file = root / "SIM" / "PATH_SNDATA_SIM.LIST"
    if list_file.is_file():
        for tok in list_file.read_text().split():
            p = os.path.expandvars(tok)
            if "$" in p:  # undefined env var: skip (getList_PATH_SNDATA_SIM)
                continue
            sim_paths.append(Path(p))
    sim_paths.append(root / "SIM")
    for p in sim_paths:
        if private and p == private:
            continue
        dirs.append(p / version)

    found = [d for d in dirs if (d / f"{version}.LIST").is_file()
             or (d / HATS_PROPERTIES).is_file()]
    if not found:
        checked = "\n  ".join(str(d) for d in dirs)
        raise SNANAError(
            f"Could not find SNANA version '{version}'; checked:\n  {checked}",
            function_name="resolve_version_dir")
    if len(found) > 1:
        raise SNANAError(
            f"Found {len(found)} copies of VERSION='{version}': "
            f"{[str(d) for d in found]}; only one unique version allowed",
            function_name="resolve_version_dir")
    return str(found[0])


def discover_inputs(config):
    """Resolve the data versions, locate the data files, and run the
    REQUIRE_DOCANA checks. Returns the versions' directories and the first
    one's HEAD path."""
    d = config.data
    if d.kcor_file is None:
        raise SNANAError(
            "KCOR_FILE is required", function_name="KCOR_FILE")
    if d.version_photometry_wildcard:
        # glob data folders; each match's basename is a version (snana.F90:6094)
        pattern = os.path.expandvars(d.version_photometry_wildcard)
        sim_dirs = sorted(p for p in glob.glob(pattern) if Path(p).is_dir())
        if not sim_dirs:
            raise SNANAError(
                f"Found zero data versions matching "
                f"VERSION_PHOTOMETRY_WILDCARD = {pattern}",
                function_name="PREP_VERSION_PHOTOMETRY")
        log.line(logger, f"Wildcard matched {len(sim_dirs)} data version(s): "
                 f"{', '.join(Path(p).name for p in sim_dirs)}")
    elif d.version_photometry:
        sim_dirs = [resolve_version_dir(d.version_photometry,
                                        d.private_data_path)]
    else:
        raise SNANAError(
            "VERSION_PHOTOMETRY or VERSION_PHOTOMETRY_WILDCARD is required",
            function_name="VERSION_PHOTOMETRY")

    data_path = version_sources(sim_dirs[0],
                                config.data.photometry_columns)[0][0]
    sim_dir = sim_dirs[0]
    # the version README and the model's metadata files (snana.F90:3042)
    model_files = metadata_files(MODEL_REGISTRY[fitted_model_type(config)],
                                 config.model.fitmodel_name)
    check_docana(config.fit.require_docana,
                 sorted(glob.glob(str(Path(sim_dir) / '*.README')))
                 + list(model_files))
    # FLUXERRMODEL_OPTMASK 1024: the same check on the map file (sntools_fluxErrModels.c:85)
    if config.data.fluxerrmodel_optmask & 1024:
        check_docana(1, [os.path.expandvars(p) for p in
                         (config.data.fluxerrmodel_file,
                          config.data.sim_fluxerrmodel_file) if p is not None])
    log.banner(logger, "Read light curves")
    log.kv(logger, "DATA", data_path)
    return sim_dirs, data_path


def check_docana(optmask, paths):
    """REQUIRE_DOCANA: each file must open with the DOCUMENTATION key.
    Bit 0 aborts, otherwise it warns; the check itself is core's."""
    if not optmask:
        return
    for path in paths:
        check_file_docana(path, require=bool(optmask & 1))


def simulation_cosmology(sim_dirs):
    """The cosmology the data versions were simulated with, from each
    version's README, for the FITRES header; {} for real data. Merged
    versions must agree, as their zero points must."""
    found = {tuple(read_simulation_cosmology(readme).items())
             for readme in (Path(d) / f"{Path(d).name}.README" for d in sim_dirs)
             if readme.is_file()}
    if len(found) > 1:
        raise SNANAError(
            f"Data versions were simulated with different cosmologies "
            f"({[dict(c) for c in found]}); the FITRES records one.",
            function_name="VERSION_PHOTOMETRY")
    return dict(found.pop()) if found else {}


def read_shared_files(config):
    """The files ``config`` names that every data version it loads shares,
    read once as loader keyword arguments: the flux-error and host-noise
    maps, the header overrides, the non-linearity model and the E(B-V)
    options with their dust map. Only the Data record is read."""
    return dict(
        header_override=read_header_override_file(
            config.data.header_override_file, config.data.header_override_dir),
        sim_header_override=read_header_override_file(
            config.data.sim_header_override_file,
            config.data.sim_header_override_dir),
        nonlin_model=(read_nonlinearity_file(config.data.nonlinearity_file)
                      if config.data.nonlinearity_file is not None else None),
        fluxerr_model=(read_fluxerr_model_file(
            os.path.expandvars(config.data.fluxerrmodel_file))
            if config.data.fluxerrmodel_file is not None else None),
        sim_fluxerr_model=(read_fluxerr_model_file(
            os.path.expandvars(config.data.sim_fluxerrmodel_file))
            if config.data.sim_fluxerrmodel_file is not None else None),
        hostnoise_maps=(read_hostnoise_file(config.data.fudge_hostnoise_file)
                        if config.data.fudge_hostnoise_file is not None
                        else None),
        mwebv=mwebv_options(config),
    )


def load_inputs(config, epoch_cutwin, filter_lamavg, sim_dir, shared):
    """The loader keyword arguments for the data versions in ``sim_dir``:
    ``shared`` (:func:`read_shared_files`) and the files the version itself
    names. ``filter_lamavg`` is FILTOBS_LAMAVG of each fit band, in index
    order. ``sim_dir`` is the first data version's directory, whose name is
    the version SNANA checks file-based fudges against."""
    version = Path(sim_dir).name if sim_dir is not None else (config.data.version_photometry or '')
    zp_shift = shift_map_with_poly(config.data.magobs_shift_zp,
                                    config.data.magobs_shift_zp_params,
                                    filter_lamavg)

    # DEFAULT reads <version>.IGNORE when present (snana.F90:5583, :13040)
    epoch_ignore, mskrej = None, config.selection.photflag_mskrej
    ignore_file = config.selection.epoch_ignore_file
    if ignore_file == 'DEFAULT':
        ignore_file = Path(sim_dir) / f'{version}.IGNORE' if sim_dir is not None else None
        if ignore_file is not None and not ignore_file.exists():
            ignore_file = None
    if ignore_file is not None:
        epoch_ignore, file_mskrej = read_epoch_ignore_file(ignore_file)
        if file_mskrej is not None:
            mskrej = (file_mskrej,)

    return dict(
        **shared,
        fit_bands=list(filter_lamavg),
        jobsplit=config.selection.jobsplit,
        zshifts={'final': config.data.redshift_final_shift,
                 'zphot': config.data.hostgal_zphot_shift,
                 'zspec': config.data.hostgal_zspec_shift} if (
                     config.data.redshift_final_shift or config.data.hostgal_zphot_shift
                     or config.data.hostgal_zspec_shift) else None,
        ztol_helio2cmb=config.data.ztol_helio2cmb,
        vpec_err_override=config.data.vpec_err_override,
        opt_vpec_cor=config.data.opt_vpec_cor,
        restore_wrong_vpec=config.data.restore_wrong_vpec,
        abort_on_badqzphot=config.fit.abort_on_badqzphot,
        dofit_photoz=config.photoz.dofit_photoz,
        quantile_spline=config.photoz.quantile_spline(),
        epoch_ignore=epoch_ignore,
        photflag_mskrej=mskrej,
        photflag_detect=config.data.photflag_detect,
        filter_replace=config.data.filter_replace,
        private_varnames=config.data.private_varname_readlist,
        fluxerrcalc_zpterr=config.data.fluxerrcalc_zpterr,
        use_snhost_zphot=config.data.use_snhost_zphot,
        use_snhost_qzphot=config.data.use_snhost_qzphot,
        snfield_list=config.selection.snfield_list,
        setpkmjd=setpkmjd_options(config),
        epoch_windows=epoch_cutwin,
        epoch_snrmin=config.selection.epcut_snrmin,
        read_zmax=config.selection.cutwin.get(
            'redshift', (-CUTVAL_OPEN, CUTVAL_OPEN))[1],
        earlylc=config.selection.earlylc,
        filter_lamavg=filter_lamavg,
        fudge_flux=config.data.fudge_fluxcal,
        fluxcal_baseline_correct=config.data.fluxcal_baseline_correct,
        dataerr_scale=config.data.fudge_dataerr_scale,
        magobs_shift_zp=zp_shift,
        magcor_map=(read_magcor_file(config.data.magcor_file, version)
                    if config.data.magcor_file is not None else None),
        sim_magcor_map=(read_magcor_file(config.data.sim_magcor_file, version)
                        if config.data.sim_magcor_file is not None else None),
    )


def setpkmjd_options(config):
    """Peak-MJD estimator settings, or None to keep the header PEAKMJD."""
    if not (config.data.has_setpkmjd_bazin() or config.data.has_setpkmjd_fmax_clump()
            or config.data.has_setpkmjd_fluxmax() or config.data.has_setpkmjd_trigger()):
        return None
    return {
        "bazin": config.data.has_setpkmjd_bazin(),
        "fmax_clump": config.data.has_setpkmjd_fmax_clump(),
        "log_snr_weight": bool(config.data.opt_setpkmjd & OPTMASK_SETPKMJD_FLUXMAX3),
        "fluxmax": config.data.has_setpkmjd_fluxmax(),
        "trigger": config.data.has_setpkmjd_trigger(),
        "shift": config.data.shift_setpkmjd,
        "opt": config.data.opt_setpkmjd,
        "abort_on_nopkmjd": config.fit.abort_on_nopkmjd,
        "mjdwin": config.data.mjdwin_setpkmjd,
        "snrcut": config.data.snrcut_setpkmjd,
        "snrcut_backup": 3.0,
        "num_nobs_bins": config.fit.num_nobs_bins,
        "min_bin_gain": config.fit.min_bin_gain,
        "batch_size": config.fit.batch_size,
        # the Bazin fit window about the brightest epoch (snana.F90:26609)
        "cutwin_tobs": config.selection.epoch_cutwin.get(
            'tobs', (-CUTVAL_OPEN, CUTVAL_OPEN)),
    }


def _merge_versions(sources, load_one):
    """One batch from every source of every data version (`version_sources`),
    selected and fitted in a single pass."""
    batches, zps, laws, sims = [], [], [], []
    for path, columns in sources:
        b = load_one(columns)
        zps.append(b.zp_fluxcal)
        laws.append(b.sim_mw_colour_law)
        sims.append(b.sim_truth is not None)
        if b.n_sne:
            batches.append(b)
        log.line(logger, f"{Path(path).name}: {b.n_sne} SNe")
    if len(set(zps)) > 1:
        raise SNANAError(
            f"Data versions disagree on ZP_FLUXCAL ({sorted(set(zps))}); their "
            f"FLUXCAL values are on different scales and jax carries one "
            f"zeropoint for the run.", function_name="VERSION_PHOTOMETRY")
    if len(set(laws)) > 1:
        raise SNANAError(
            f"Data versions were simulated with different Milky Way laws "
            f"({sorted(set(laws), key=str)}); the fit applies one.",
            function_name="VERSION_PHOTOMETRY")
    if len(set(sims)) > 1:
        raise SNANAError(
            "Data versions mix simulated and real data; the merged batch would "
            "carry SIM_ truth for some events only.",
            function_name="VERSION_PHOTOMETRY")
    if len(zps) == 1:
        return b
    if not batches:
        raise SNANAError("No events in any data version",
                         function_name="VERSION_PHOTOMETRY")
    return LightCurveBatch.concatenate(batches)


def start_values_from_sncid_list(table, snids, param_names):
    """OPT_SNCID_LIST bit 1: each fit parameter's starting value from the
    list file's fit parameters (FITINI_LIST_FILE, snlc_fit.F90:9997), NaN
    for an event the file does not list."""
    if table is None:
        raise SNANAError(
            "OPT_SNCID_LIST bit 1 seeds the fit from SNCID_LIST_FILE, which "
            "must be a FITRES table; this file is a plain CID list.",
            function_name="OPT_SNCID_LIST")
    by_cid = {c: i for i, c in enumerate(cid_keys(table.df['CID']))}
    rows = np.array([by_cid.get(c, -1) for c in cid_keys(snids)])
    seeds = {}
    for name in param_names:
        col = 'zHD' if name == 'zPHOT' else name
        if not table.has(col):
            continue
        vals = np.asarray(table.df[col], dtype=float)[np.maximum(rows, 0)]
        if np.any((rows >= 0) & (vals <= -999.0)):
            raise SNANAError(
                f"SNCID_LIST_FILE column {col} has values <= -999 for events "
                f"being fitted; SNANA aborts on these.",
                function_name="OPT_SNCID_LIST")
        seeds[name] = np.where(rows >= 0, vals, np.nan)
    return seeds


def prepare_batch(config, parameter_names, batch, *, apply_selection, ladder):
    """Prepare loaded columns for fitting, with sample selection when requested.

    Usable-epoch checks, redshift priors and CID starting values
    are shared by file and in-memory inputs. Return the fittable batch, the
    batch before selection, and marks identifying excluded objects.
    """
    cutwin, _, _ = cut_windows(config)
    listed = sncid_list(config) if apply_selection else ([], [], None)
    selection = build_selection(config, cutwin, listed) if apply_selection else None
    prepared = batch
    if apply_selection:
        # event caps span data versions (snana.F90:4955), and a CID repeated
        # across them is a duplicate (:11708)
        batch = select_events(
            batch, snana_selection=selection,
            max_process=config.selection.mxevt_process,
            max_fit=min(x for x in (config.selection.max_fit, config.selection.mxevt_cuts)
                        if x is not None),
            ladder=ladder)
    selected = np.asarray(batch.snids)
    batch, empty, thin, badq = drop_unfittable(
        batch, cutwin.get('nepoch') if apply_selection else None,
        abort_on_noepochs=config.fit.abort_on_noepochs,
        quantile_prior=quantile_prior_mask(config, batch),
        abort_on_missing_qzphot=config.fit.abort_on_missing_qzphot)
    batch = dataclasses.replace(
        batch,
        z_priors=(redshift_priors(config, batch)
                  if config.photoz.dofit_photoz else None),
        sncid_inival=(start_values_from_sncid_list(listed[2], np.asarray(batch.snids),
                                                   parameter_names)
                      if apply_selection and config.selection.opt_sncid_list & 2 else None))
    if batch.z_priors is not None:
        supported = np.asarray(pdf_support_mask(config, batch.z_priors))
        if not supported.all():
            badq = np.concatenate((badq, np.asarray(batch.snids)[~supported]))
            log.line(logger, f"Skipped {int((~supported).sum())} SNe whose host z-PDF "
                     f"has no positive probability within PHOTOZ_BOUND")
            batch = batch.filter_by_mask(supported)
    ladder.append(('FITTABLE', batch.n_sne))
    marks = ((selected, ERRFLAG_FIT_CUTS),
             (empty, ERRFLAG_FITPREP_NOEPOCHS),
             (thin, ERRFLAG_FITPREP_NEPOCH),
             (badq, ERRFLAG_FITPREP_QUANTILES))
    return batch, prepared, marks


def select_for_fit(config, parameter_names, precut_batch, *, keep_preselection=False):
    """The pre-fit selection of a loaded batch (:func:`prepare_batch`).
    Returns the batch to fit; the loaded batch with the ERRFLAG_FIT of each
    event the selection or fit prep dropped, when the SNANA table is written
    (None otherwise, or unmarked under ``keep_preselection``); and the
    CUTSTAT rows."""
    ladder = []
    batch, precut_batch, marks = prepare_batch(
        config, parameter_names, precut_batch, apply_selection=True, ladder=ladder)
    if not keep_preselection:
        precut_batch = (mark_errflags(precut_batch, *marks)
                        if 'snana' in config.output.tables else None)
    return batch, precut_batch, ladder


def load(config, model, kcor, sim_dirs, shared, *, keep_preselection=False,
         sources=None, photometry=None, peak_shape=None):
    """Read every data version, merge, and apply the pre-fit selection.

    ``shared`` holds the files every version shares
    (:func:`read_shared_files`). The batch carries its photometry: the fit
    bands of ``kcor`` as ``config`` calibrates them for ``model``
    (:func:`fit_photometry`) and its ZP_FLUXCAL. Returns the batch to fit;
    the batch as loaded, with the ERRFLAG_FIT of each event the selection or
    fit prep dropped, when the SNANA table is written (None otherwise); the
    CUTSTAT rows; and the seconds spent reading files, so a caller timing
    this stage can separate the read from the batching, which is what scales
    with max_obs.

    ``keep_preselection`` retains the unmarked batch before sample selection,
    even without SNANA output, so pipeline CID synchronization can reselect it.
    ``sources`` supplies already-read ``(path, (HEAD, PHOT, metadata))`` pairs;
    omitted, files stream one at a time. ``photometry`` supplies the cell's
    calibration and ``peak_shape`` bounds peak-estimator execution shapes.
    """
    for msg in config.output._unsupported:
        log.warn(logger, "SNTABLE_LIST", f"ignoring {msg}")
    filter_lamavg = kcor.filter_lamavg(resolved_filters(config, kcor))
    _, epoch_cutwin, _ = cut_windows(config)
    common = dict(
        keep_obs_extras=bool({'outlier', 'lcplot'} & set(config.output.tables)
                             or config.data.usesim_trueflux
                             or config.fit.simfit_ideal_prescale),
        model_sim_columns=model_sim_columns(model),
        **load_inputs(config, epoch_cutwin, filter_lamavg, sim_dirs[0],
                      shared),
        filter_map={name: idx for idx, name in enumerate(filter_lamavg)})

    read_seconds = 0.0

    def read_sources():
        nonlocal read_seconds
        for directory in sim_dirs:
            for path, read in version_sources(directory, config.data.photometry_columns):
                start = time.perf_counter()
                columns = read()
                read_seconds += time.perf_counter() - start
                yield path, columns

    def _load_one(columns):
        head_data, phot_data, header = columns
        return batch_from_columns(head_data, phot_data, **header, **common,
                                  peak_shape=peak_shape)

    precut_batch = _merge_versions(
        read_sources() if sources is None else sources, _load_one)
    if photometry is None:
        photometry = fit_photometry(config, model, kcor)
    precut_batch = dataclasses.replace(precut_batch, photometry={
        **photometry, 'zp_fluxcal': precut_batch.zp_fluxcal})
    batch, precut_batch, ladder = select_for_fit(
        config, list(fit_parameters(config, model)), precut_batch,
        keep_preselection=keep_preselection)
    log.line(logger, f"Loaded and batched {batch.n_sne} SNe, "
             f"max_obs={batch.max_obs}")
    return batch, precut_batch, ladder, read_seconds


_OBJECT_HEADER_NAMES = {
    'redshift': 'REDSHIFT_HELIO', 'redshift_err': 'REDSHIFT_HELIO_ERR',
    'redshift_cmb': 'REDSHIFT_CMB', 'mwebv': 'MWEBV', 'peakmjd': 'PEAKMJD',
    'photoz': 'HOSTGAL_PHOTOZ', 'photoz_err': 'HOSTGAL_PHOTOZ_ERR',
    'ra': 'RA', 'dec': 'DEC',
}


def _frame(value):
    if isinstance(value, Table):
        return value.to_pandas()
    return pd.DataFrame(value).copy()


def _split_photometry(obs):
    """Separate repeated object properties from measurements, preserving order."""
    if 'snid' not in obs:
        raise SNANAError('Missing input columns: snid')
    if obs.snid.isna().any():
        raise SNANAError('Object IDs must not be missing')
    obs['snid'] = obs.snid.astype(str)
    columns = [name for name in (*_OBJECT_HEADER_NAMES, 'flux_unit', 'zp') if name in obs]
    groups = obs.groupby('snid', sort=False)
    for name in columns:
        inconsistent = groups[name].nunique(dropna=False) > 1
        if inconsistent.any():
            ids = inconsistent.index[inconsistent].tolist()
            raise SNANAError(f"Inconsistent {name!r} for objects {ids}; "
                             'object properties must agree across their observations')
    objects = obs[['snid', *columns]].drop_duplicates('snid').reset_index(drop=True)
    return obs.drop(columns=columns), objects


def input_tables(*, mjd, flux, flux_err, band, redshift, mwebv, snid,
                  flux_unit, zp, redshift_cmb, metadata, photometry, objects,
                  column_map):
    """Copy supplied tables or assemble them from single-object arrays."""
    if photometry is not None or objects is not None:
        if photometry is None or any(
                v is not None for v in (mjd, flux, flux_err, band, redshift, redshift_cmb, metadata)):
            raise SNANAError('Supply either single-object arrays or photometry, with an optional objects table')
        rename = {actual: canonical for canonical, actual in (column_map or {}).items()}
        obs = _frame(photometry).rename(columns=rename)
        if objects is None:
            obs, obj = _split_photometry(obs)
        else:
            obj = _frame(objects).rename(columns=rename)
        for name, default in {'flux_unit': flux_unit, 'zp': zp, 'mwebv': mwebv}.items():
            if name not in obj:
                obj[name] = default
            elif default is not None:
                obj[name] = obj[name].fillna(default)
    else:
        if column_map:
            raise SNANAError('column_map applies to photometry and objects tables')
        if any(v is None for v in (mjd, flux, flux_err, band)):
            raise SNANAError('mjd, flux, flux_err and band are required')
        times, measured, error = (np.asarray(v) for v in (mjd, flux, flux_err))
        if times.ndim != 1 or measured.shape != times.shape or error.shape != times.shape:
            raise SNANAError('mjd, flux and flux_err must be aligned one-dimensional arrays')
        bands = np.broadcast_to(band, times.shape)
        obs = pd.DataFrame(dict(snid=str(snid), mjd=times, flux=measured, flux_err=error, band=bands))
        values = dict(metadata or {})
        values.update(snid=str(snid), mwebv=mwebv, flux_unit=flux_unit, zp=zp)
        if redshift is not None:
            values['redshift'] = redshift
        if redshift_cmb is not None:
            values['redshift_cmb'] = redshift_cmb
        obj = pd.DataFrame([values])
    return obs, obj


def validate_tables(obs, obj, config):
    """Validate measurements, object associations and required redshifts."""
    for frame, required in ((obs, ('snid', 'mjd', 'flux', 'flux_err', 'band')), (obj, ('snid',))):
        missing = set(required) - set(frame)
        if missing:
            raise SNANAError(f"Missing input columns: {sorted(missing)}")
    if obs.snid.isna().any() or obj.snid.isna().any():
        raise SNANAError('Object IDs must not be missing')
    obs['snid'], obj['snid'] = obs.snid.astype(str), obj.snid.astype(str)
    if obj.snid.duplicated().any() or not set(obs.snid) <= set(obj.snid):
        raise SNANAError('Object IDs must be unique and every photometry ID must identify an object')
    if not np.isfinite(obs[['mjd', 'flux', 'flux_err']].to_numpy(dtype=float)).all() or (obs.flux_err <= 0).any():
        raise SNANAError('Measurements must be finite and flux_err must be positive')
    if 'redshift' not in obj:
        if not config.photoz.dofit_photoz:
            raise SNANAError('redshift is required for fixed-redshift fitting')
        # The established hostless/photo-z preparation handles its sentinel.
        obj['redshift'] = -9.
    if not config.photoz.dofit_photoz and (not np.isfinite(obj.redshift).all() or (obj.redshift <= 0).any()):
        raise SNANAError('redshift must be finite and positive')


def batch_from_tables(obs, obj, fitter, cfg, epoch_cuts, shared, zp_fluxcal):
    """Use the common light-curve reader with in-memory HEAD/PHOT columns."""
    active = cfg.fit.filtlist_fit
    indices = {sid: i for i, sid in enumerate(obj.snid)}
    obs['_object_index'] = obs.snid.map(indices)
    obs = obs.sort_values(['_object_index', 'mjd'], kind='stable')
    counts = obs.groupby('snid').size().reindex(obj.snid, fill_value=0).to_numpy()
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    heads = {_OBJECT_HEADER_NAMES.get(key, key): np.asarray(value) for key, value in obj.items()
             if key not in ('snid', 'flux_unit', 'zp')}
    heads.update(SNID=obj.snid.to_numpy(), PTROBS_MIN=starts+1, PTROBS_MAX=starts+counts)
    columns = {key.upper(): value.to_numpy() for key, value in obs.items()
               if key not in ('snid', '_object_index', 'flux', 'flux_err')}
    columns.update(FLUXCAL=obs.flux.to_numpy(), FLUXCALERR=obs.flux_err.to_numpy())
    options = load_inputs(cfg, epoch_cuts, fitter.calibration.filter_lamavg(active), None, shared)
    batch = batch_from_columns(heads, columns, is_sim=False, zp_fluxcal=zp_fluxcal,
        sim_mw_colour_law=None, filter_map={name: i for i, name in enumerate(active)},
        keep_obs_extras=True, **options)
    return batch
