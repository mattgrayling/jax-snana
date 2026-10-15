"""In-memory fitting using the same preparation and kernels as the CLI."""

import copy
import dataclasses
import os
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import Table

from jax_snlc_core.io.kcor_reader import KcorReader
from jax_snlc_core.models import infer_model_type_from_path
from jax_snlc_core.photometry.noise import ZEROPOINT_FLUXCAL_SNANA
from jax_snlc_core.photometry.resources import load_photometry_resources, resolve_filtlist
from jax_snlc_core.utils.error_handling import SNANAError
from .batch import ERRFLAG_FIT_CUTS, ERRFLAG_FITPREP_NOEPOCHS
from .data_loader import batch_from_columns
from .fitter import attach_mw_attenuation, empty_results, fit_photometry, resolved_config
from .inputs import build_config
from .config import Selection
from .run import (cut_windows, fit_prepared, fit_model, load_inputs, prepare_batch,
                  read_kcor, read_shared_files)


_OBJECT_HEADER_NAMES = {
    'redshift': 'REDSHIFT_HELIO', 'redshift_err': 'REDSHIFT_HELIO_ERR',
    'redshift_cmb': 'REDSHIFT_CMB', 'mwebv': 'MWEBV', 'peakmjd': 'PEAKMJD',
    'photoz': 'HOSTGAL_PHOTOZ', 'photoz_err': 'HOSTGAL_PHOTOZ_ERR',
    'ra': 'RA', 'dec': 'DEC',
}


def initialize(fitter, model, *, calibration=None, config=None, **settings):
    """Load explicit scientific resources, leaving shapes until data arrive."""
    assets = Path(__file__).resolve().parents[1] / 'jax_snlc_core' / 'data'

    def model_path(value):
        supplied = Path(os.path.expandvars(str(value))).expanduser()
        bundled = assets / 'models' / str(value)
        if not supplied.is_dir() and bundled.is_dir() and supplied.name == str(value):
            return bundled.resolve()
        return supplied.resolve() if supplied.exists() else supplied

    name = str(model_path(model))
    configured = config.model.fitmodel_name if config is not None else None
    for value in (configured, settings.get('fitmodel_name')):
        if value is not None and model_path(value).resolve() != Path(name).resolve():
            raise SNANAError('model and fitmodel_name specify different models')
    cfg = build_config({**settings, 'fitmodel_name': name}, config=config)
    cfg = build_config({'fitmodel_name': name,
                        'model_type': cfg.model.model_type or infer_model_type_from_path(name)}, config=cfg)
    model_instance = fit_model(cfg)
    if calibration is not None and cfg.data.kcor_file is not None:
        if not isinstance(calibration, (str, Path)) or str(calibration) != cfg.data.kcor_file:
            raise SNANAError('calibration and kcor_file specify different sources')
    source = calibration if calibration is not None else cfg.data.kcor_file
    if source is None or source == 'builtin':
        source = assets / 'calibration' / 'filters.yaml'
    if isinstance(source, KcorReader):
        kcor = source
    else:
        cfg.data.kcor_file = str(source)
        kcor = read_kcor(cfg)
    cfg.data.kcor_file = str(kcor.kcor_file_path)
    requested = copy.deepcopy(cfg)
    cfg.fit.filtlist_fit = (resolve_filtlist(cfg.fit.filtlist_fit, kcor)
                            if cfg.fit.filtlist_fit is not None else list(kcor.filter_names))
    fitter._initialize(cfg, model_instance, kcor, dict(epoch_lengths=[],
                 batch_size=cfg.fit.batch_size, band_window=None, n_z=None))
    fitter._input_config = requested


def _frame(value):
    if isinstance(value, Table):
        return value.to_pandas()
    return pd.DataFrame(value).copy()


def _bands(values, kcor, band_map):
    available = set(kcor.filter_names)
    result = []
    for value in np.asarray(values).astype(str).tolist():
        value = band_map.get(value, value)
        if value not in available:
            raise SNANAError(f"Unknown band {value!r}; use a full calibration name or band_map")
        result.append(value)
    return np.asarray(result)


def _unit_scales(units, zero_points, zp_fluxcal):
    """Validate object units and return their scale before band calibration."""
    units = np.asarray(units)
    zero_points = np.asarray(zero_points, dtype=float)
    supplied = ~np.isnan(zero_points)
    if np.any(supplied & ~np.isfinite(zero_points)):
        raise SNANAError('zp must be finite')
    if not np.isin(units, ('fluxcal', 'Jy', 'uJy')).all():
        raise SNANAError('flux_unit must be fluxcal, Jy or uJy')
    calibrated = units == 'fluxcal'
    if np.any(~calibrated & supplied):
        raise SNANAError('zp is only meaningful with flux_unit="fluxcal"')
    zp = np.where(supplied, zero_points, ZEROPOINT_FLUXCAL_SNANA)
    return np.where(calibrated, 10.**(0.4 * (zp_fluxcal - zp)),
                    np.where(units == 'uJy', 1.e-6, 1.))


def _jy_factors(bands, kcor, model_wave, zp_fluxcal):
    """Calibrate Jy once for each distinct band in a set of observations."""
    if not len(bands):
        return np.empty(0)
    names = list(dict.fromkeys(bands))
    resources = load_photometry_resources(kcor, model_wave, names)
    factors = {}
    for i, name in enumerate(names):
        wave, trans = map(np.asarray, kcor.get_filter_transmission(name))
        jy_ref = np.trapezoid(1.e-23 * 2.99792458e18 / wave**2 * wave * trans, wave)
        jy_ref /= np.trapezoid(wave * trans, wave)
        factors[name] = jy_ref / float(resources.filter_ref_fluxes[i]) * 10.**(0.4 * zp_fluxcal)
    return np.asarray([factors[name] for name in bands])


def _flux_factor(unit, zp, bands, kcor, model_wave, *, zp_fluxcal):
    """Convert flux and error from the supplied units to the batch's zero point."""
    scale = _unit_scales(unit, np.nan if zp is None else zp, zp_fluxcal)
    if unit == 'fluxcal':
        return np.full(len(bands), scale)
    return scale * _jy_factors(bands, kcor, model_wave, zp_fluxcal)


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


def _input_tables(*, mjd, flux, flux_err, band, redshift, mwebv, snid,
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


def _validate_tables(obs, obj, config):
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


def _normalise_flux(obs, obj, fitter):
    """Keep the supplied zero point, converting other units to the same scale."""
    obs[['flux', 'flux_err']] = obs[['flux', 'flux_err']].astype(float)
    units = obj['flux_unit'].to_numpy() if 'flux_unit' in obj else np.full(len(obj), 'fluxcal')
    zeros = obj['zp'].to_numpy(dtype=float) if 'zp' in obj else np.full(len(obj), np.nan)
    supplied_zeros = zeros[(units == 'fluxcal') & ~np.isnan(zeros)]
    zp_fluxcal = float(supplied_zeros[0]) if len(supplied_zeros) else ZEROPOINT_FLUXCAL_SNANA
    scales = _unit_scales(units, zeros, zp_fluxcal)
    indices = pd.Index(obj.snid).get_indexer(obs.snid)
    factors = scales[indices]
    needs_calibration = units[indices] != 'fluxcal'
    factors[needs_calibration] *= _jy_factors(
        obs.loc[needs_calibration, 'band'].to_numpy(),
        fitter.calibration, fitter.model.wavelength_grid, zp_fluxcal)
    obs['flux'] = obs.flux.to_numpy() * factors
    obs['flux_err'] = obs.flux_err.to_numpy() * factors
    input_units = {snid: (unit, None if np.isnan(zero) else float(zero))
                   for snid, unit, zero in zip(obj.snid, units, zeros)}
    return input_units, zp_fluxcal


def _batch_from_tables(obs, obj, fitter, cfg, epoch_cuts, shared, zp_fluxcal):
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


def _model_columns(fitter, result, redshifts):
    """Use the model's output conversion, including derived distances."""
    distance = fitter.config.fit.distance_modulus(redshifts)
    columns = {}
    for name, values, errors in fitter.model.get_fitres_param_columns(
            result.params, result.param_errors, redshifts, distance):
        columns[name] = np.asarray(values)
        columns[name + '_err'] = np.asarray(errors)
    for name, values in fitter.model.get_fitres_derived_columns(result.params):
        columns[name] = np.asarray(values)
    return columns


def _outcomes(obj, result, marks, keep, codes):
    """Report preparation and fit outcomes in the caller's object order."""
    outcome = pd.DataFrame(dict(snid=obj.snid, input_index=np.arange(len(obj)),
        status='rejected', reason=ERRFLAG_FIT_CUTS, fit_index=-1)).set_index('snid', drop=False)
    for ids, code in marks:
        outcome.loc[list(map(str, ids)), 'reason'] = code
    if result.n_sne:
        ids = np.asarray(result.snids).astype(str)
        outcome.loc[ids, 'fit_index'] = np.arange(result.n_sne)
        outcome.loc[ids, 'reason'] = codes
        outcome.loc[ids, 'status'] = np.where(keep, 'accepted', 'rejected')
    return outcome.reset_index(drop=True)


def fit(fitter, *, mjd=None, flux=None, flux_err=None, band=None,
        redshift=None, mwebv=0., snid='0', flux_unit='fluxcal', zp=None,
        redshift_cmb=None, metadata=None, photometry=None, objects=None,
        column_map=None, band_map=None, progress=True, apply_selection=False):
    """Convert public data to existing batches, then select and fit them."""
    obs, obj = _input_tables(
        mjd=mjd, flux=flux, flux_err=flux_err, band=band, redshift=redshift,
        mwebv=mwebv, snid=snid, flux_unit=flux_unit, zp=zp,
        redshift_cmb=redshift_cmb, metadata=metadata, photometry=photometry,
        objects=objects, column_map=column_map)
    cfg = resolved_config(build_config(config=fitter.configuration()), fitter.model)
    if not apply_selection and cfg.selection != Selection():
        raise SNANAError('Selection settings require fit(..., apply_selection=True)')
    _validate_tables(obs, obj, cfg)
    obs['band'] = _bands(obs.band, fitter.calibration, band_map or {})
    units, zp_fluxcal = _normalise_flux(obs, obj, fitter)
    active = (resolve_filtlist(cfg.fit.filtlist_fit, fitter.calibration)
              if cfg.fit.filtlist_fit is not None else list(dict.fromkeys(obs.band)))
    if not active or not len(obj):
        result = empty_results(fitter.fit_parameters, 0, fitter.config.fit.minos)
        result.model_columns = _model_columns(fitter, result, np.empty(0))
        result.outcomes = pd.DataFrame(dict(snid=obj.snid, input_index=np.arange(len(obj)),
            status='rejected', reason=ERRFLAG_FITPREP_NOEPOCHS, fit_index=-1))
        return result
    cfg.fit.filtlist_fit = active
    _, epoch_cuts, _ = cut_windows(cfg)
    shared = read_shared_files(cfg)
    batch = _batch_from_tables(obs, obj, fitter, cfg, epoch_cuts, shared, zp_fluxcal)
    batch = dataclasses.replace(batch, photometry={
        **fit_photometry(cfg, fitter.model, fitter.calibration),
        'zp_fluxcal': batch.zp_fluxcal})
    selected, _, marks = prepare_batch(cfg, fitter.param_names, batch,
                                    apply_selection=apply_selection, ladder=[])
    input_batch = selected
    selected, result, keep, codes, _, _ = fit_prepared(
        fitter, cfg, selected, apply_selection=apply_selection, progress=progress)
    result.outcomes = _outcomes(obj, result, marks, keep, codes)
    result.model_columns = _model_columns(
        fitter, result, result.params.get('zPHOT', selected.redshifts))
    context = dict(batch=input_batch, units=units, config=copy.deepcopy(fitter.config),
                   calibration=fitter.calibration.kcor_file_path)
    result.context = [(context, i) for i in range(result.n_sne)]
    return result


def predict(fitter, result, mjd, band, *, snid=None, flux_unit=None, zp=None):
    """Evaluate one stored fit without depending on the most recent fit call."""
    if result.context is None:
        raise SNANAError('This result has no in-memory prediction context')
    if snid is None:
        if result.n_sne != 1:
            raise SNANAError('Specify snid when predicting a result with several objects')
        snid = str(result.snids[0])
    found = np.flatnonzero(np.asarray(result.snids).astype(str) == str(snid))
    if len(found) != 1:
        raise SNANAError(f'No unique fitted object {snid!r}')
    i = int(found[0])
    context, batch_row = result.context[i]
    if context['calibration'] != fitter.calibration.kcor_file_path or context['config'].model != fitter.config.model:
        raise SNANAError('Result and fitter use different model or calibration settings')
    if not all(np.isfinite(np.asarray(value)[i]) for value in result.params.values()):
        raise SNANAError(f'Object {snid} has no finite fitted parameters')
    times = np.asarray(mjd, dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all():
        raise SNANAError('Prediction MJD must be a finite one-dimensional array')
    bands = _bands(np.broadcast_to(band, times.shape), fitter.calibration, {})
    if not len(times):
        return dict(mjd=times, band=bands, flux=np.empty(0), model_flux_err=np.empty(0))
    names = list(dict.fromkeys(bands))
    cfg = copy.deepcopy(context['config'])
    cfg.fit.filtlist_fit = names
    zp_fluxcal = context['batch'].zp_fluxcal
    phot = {**fit_photometry(cfg, fitter.model, fitter.calibration), 'zp_fluxcal': zp_fluxcal}
    batch = dataclasses.replace(context['batch'].take(np.array([batch_row])), photometry=phot)
    batch = attach_mw_attenuation(cfg, batch)
    params = {name: np.asarray(values)[[i]] for name, values in result.params.items()}
    redshift = params.get('zPHOT', batch.redshifts)
    flux, error = fitter.evaluate_flux(params, times[None], redshift,
        np.array([[names.index(b) for b in bands]]), np.ones((1, len(times)), dtype=bool),
        reference_mjds=batch.reference_mjds, mw_att_obs=batch.mw_att_obs, photometry=phot)
    original_unit, original_zp = context['units'][str(snid)]
    unit = original_unit if flux_unit is None else flux_unit
    zero = original_zp if flux_unit is None and zp is None else zp
    factor = _flux_factor(unit, zero, bands, fitter.calibration,
                          fitter.model.wavelength_grid, zp_fluxcal=zp_fluxcal)
    return dict(mjd=times, band=bands, flux=np.asarray(flux)[0]/factor,
                model_flux_err=np.asarray(error)[0]/factor)
