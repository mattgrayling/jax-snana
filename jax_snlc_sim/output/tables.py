"""Shared construction of SNANA HEAD and PHOT tables."""

import logging
from typing import Dict

import numpy as np

from jax_snlc_core.dust import OPT_MWCOLORLAW_NAMES
from jax_snlc_core.filters import filter_band
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import (HOSTGAL_COORD_NULL, HOSTGAL_PREFIXES, HOSTGAL_PROPERTY_NULL,
                                       SNANA_NULLFLOAT, MXBIN_HOSTGALz,
                                       MXBIN_HOSTGALz_QUANTILE, HOSTGAL_SHAPE_NULL,
                                       HOSTGAL_MAG_NULL, HOSTGAL_SB_NULL, HOSTGAL_CONFUSION_NULL,
                                       HOSTGAL_NO_HOST_NULL, hostgal_null_key)
from jax_snlc_sim.observations import (LookupColumn, simlib_epochs, span_indices)
from jax_snlc_sim.simulation import SimulationConfig
from jax_snlc_sim.spectroscopic_efficiency import FLAG_ALL_FAIL


logger = logging.getLogger(__name__)


SIM_HOSTLIB_GALID_NULL = -9

# string column widths (sndata.h:115-116)
MXCHAR_CCID = 40

MXCHAR_FIELDNAME = 32

MODEL_INDEX_NON1ASED = 10   # sntools.h:149

MODEL_INDEX_NON1AGRID = 11  # sntools.h:150

# MODEL_ indices SNANA generates as Type Ia (LGEN_SNIA, snlc_sim.c:26628):
# STRETCH, MLCS2k2, SNOOPY, S11DM15, SALT2 (SALT3 too) and BayeSN
SNIA_GENMODEL_INDICES = (1, 3, 4, 5, 6, 13)


def subsurvey_flag(resources) -> bool:
    """Whether the SIMLIB names sub-surveys, globally or per entry
    (wr_snfitsio_SET_SUBSURVEY_FLAG, sntools_dataformat_fits.c:1420-1438)."""
    return bool(resources.simlib_subsurvey_list) or any(
        f.subsurvey not in ('', resources.survey_name) for f in resources.fields)


# The colour law's registry name back to its OPT_MWCOLORLAW code.
_MWCOLORLAW_CODES = {name: code for code, name in OPT_MWCOLORLAW_NAMES.items()}


def mw_option_keys(config) -> Dict[str, float]:
    """The Milky Way dust options every data file records
    (sntools_dataformat_fits.c:1267-1281, sntools_dataformat_text.c:285-446)."""
    mw = config.mw_dust
    return {'SIMOPT_MWCOLORLAW': _MWCOLORLAW_CODES[mw.dust_law],
            'SIM_MWRV': float(mw.rv),
            'SIMOPT_MWEBV': int(mw.opt_mwebv)}


def effective_gentype(config: SimulationConfig) -> int:
    """SIM_GENTYPE = SNTYPE_Ia_SPEC, overridden by GENTYPE when > 0
    (snlc_sim.c:26638-26641). GENTYPE is optional; SNANA's unset value is -9."""
    return (config.rates.gentype if config.rates.gentype > 0
            else config.rates.sntype_spec)


# per-SN arrays every writer needs, one entry per written event
_REQUIRED_EVENT_KEYS = (
    'cid', 'sim_redshift_cmb', 'sim_redshift_helio', 'peakmjd', 'mu',
    'field_idx', 'mwebv', 'mwebv_err', 'sim_mwebv',
    'redshift_helio', 'redshift_final', 'redshift_smear_err', 'specid_pass',
    'mjd_trigger', 'mjd_detect_first', 'mjd_detect_last',
)

# per-obs arrays the PHOT table serialises verbatim; _FINITE_OBS_KEYS must be finite
_REQUIRED_OBS_KEYS = (
    'mjd', 'filter_idx', 'fluxcal', 'fluxcalerr', 'is_detection', 'zpt',
    'zpterr', 'skysig', 'template_skysig', 'psf1', 'psf2', 'psf_ratio',
    'readnoise', 'gain', 'nea', 'mag_noiseless',
)

_FINITE_OBS_KEYS = ('fluxcal', 'fluxcalerr', 'zpt', 'skysig', 'gain')


def check_output_schema(columnar_data, n_sne: int, total_obs: int) -> None:
    """Validate the writer's input arrays before serialising anything.

    Required per-event arrays must exist with exactly ``n_sne`` entries and
    per-observation arrays with exactly ``total_obs``; the physical
    observation quantities must be finite. Optional fields (host, VPEC,
    atmosphere, lensing) take documented SNANA sentinels instead.
    """
    params = columnar_data['parameters']
    obs = columnar_data['observations']
    for key in _REQUIRED_EVENT_KEYS:
        if key not in params:
            raise SNANAError(
                f"output writer input is missing per-event array '{key}'",
                function_name='check_output_schema')
        n = np.asarray(params[key]).shape[0]
        if n != n_sne:
            raise SNANAError(
                f"per-event array '{key}' has {n} entries for {n_sne} events",
                function_name='check_output_schema')
    for key in _REQUIRED_OBS_KEYS:
        if key not in obs:
            raise SNANAError(
                f"output writer input is missing per-obs array '{key}'",
                function_name='check_output_schema')
        n = np.asarray(obs[key]).shape[0]
        if n != total_obs:
            raise SNANAError(
                f"per-obs array '{key}' has {n} entries for {total_obs} "
                f"observations", function_name='check_output_schema')
    for key in _FINITE_OBS_KEYS:
        if not np.all(np.isfinite(np.asarray(obs[key], dtype=np.float64))):
            raise SNANAError(
                f"per-obs array '{key}' contains non-finite values",
                function_name='check_output_schema')


def per_event_types(config, resources, params, n, specid_pass):
    """Per-event SIM_GENTYPE and SNTYPE, shared by the HEAD, TEXT and DUMP
    writers so the three cannot drift.

    SIM_GENTYPE is the per-template NON1ASED SNTAG (snlc_sim.c:14403), or
    SNTYPE_Ia_SPEC for SNIa models (:26638), overridden by GENTYPE when > 0
    (:14404, :26640). gen_spectype (:24826-24883) builds SNTYPE from the same
    base; an event is photometric ONLY when SPECID maps exist and it failed
    them, or when the spec file forces EFF=0 (L_PHOTID, :24834-24839); with
    no spec maps every event keeps the spec type, whatever the trigger did.
    The photometric type is SNTYPE_Ia_PHOT for Ia (from the SNTYPES_Ia pair)
    or base+OFFSET_TYPE_PHOT otherwise, overridden by GENTYPE_PHOT when > 0.
    """
    gentype_phot = config.rates.gentype_phot

    sim_gentype = sim_gentypes(config, params, n)
    if config.rates.gentype > 0 or 'sntype' in params:
        phot = sim_gentype + 100  # OFFSET_TYPE_PHOT
    else:
        phot = np.full(n, config.rates.sntype_phot, dtype=np.int32)
    if gentype_phot is not None and gentype_phot > 0:
        phot = np.full(n, gentype_phot, dtype=np.int32)

    has_spec_maps = bool(resources.searcheff_specid_maps)
    effzero = resources.searcheff_specid_flag == FLAG_ALL_FAIL
    is_phot = ((has_spec_maps & ~np.asarray(specid_pass, dtype=bool))
               | effzero)
    return sim_gentype, np.where(is_phot, phot, sim_gentype).astype(np.int32)


def sim_gentypes(config, params, n):
    """Per-event SIM_GENTYPE: GENTYPE when > 0, else the NON1ASED template's
    SNTAG, else SNTYPE_Ia_SPEC."""
    if config.rates.gentype > 0:
        return np.full(n, config.rates.gentype, dtype=np.int32)
    if 'sntype' in params:
        return np.asarray(params['sntype'], dtype=np.int32)
    return np.full(n, config.rates.sntype_spec, dtype=np.int32)


def generates_snia(model) -> bool:
    """SNANA's LGEN_SNIA: the model is one SNANA generates as a Type Ia
    (snlc_sim.c:26628), whatever GENTYPE the input gives."""
    return model.sim_model_index in SNIA_GENMODEL_INDICES


def gentype_to_name(config, model, generated) -> Dict[int, str]:
    """SNANA's GENTYPE_TO_NAME map, by increasing GENTYPE: the type names of
    the events that reached the magnitudes (snlc_sim.c:27549-27551).

    Several names under one GENTYPE are joined by ``+`` in the order the
    events first meet them, and a name is skipped when the list already
    contains it, even inside a longer name, as SNANA's ``strstr`` test does.
    """
    reached = generated['reached_flux']
    n = int(reached.sum())
    params = {k: generated[k][reached] for k in ('sntype', 'non1a_index')
              if k in generated}
    gentypes = sim_gentypes(config, params, n)
    if model.sim_model_index == MODEL_INDEX_NON1ASED:
        names = model.get_snana_header_params(params)['SIM_TYPE_NAME']
    else:
        names = np.full(n, _sim_type_name(config, model)
                        or sim_model_name(config, model))
    pairs, first = np.unique(np.stack([gentypes.astype(str), names], axis=1),
                             axis=0, return_index=True)
    joined: Dict[int, str] = {}
    for gentype, name in pairs[np.argsort(first)]:
        current = joined.get(int(gentype), '')
        if name not in current:
            joined[int(gentype)] = f'{current}+{name}' if current else name
    return dict(sorted(joined.items()))

def _genmodel_dirname(config) -> str:
    genmodel_path = config.model.genmodel
    return genmodel_path.rstrip('/').split('/')[-1] if genmodel_path else ''

def sim_model_name(config, model) -> str:
    """SIM_MODEL_NAME: the GENMODEL directory, for every model."""
    return _genmodel_dirname(config) or model.model_name.upper()


def _sim_type_name(config, model) -> str:
    """SIM_TYPE_NAME: the GENMODEL directory less whatever prefix the model
    declares (SNANA strips it for SIMSED alone, snlc_sim.c:29450)."""
    dirname = _genmodel_dirname(config)
    prefix = model.sim_type_name_prefix
    if prefix and dirname.upper().startswith(prefix.upper()):
        return dirname[len(prefix):]
    return dirname

def model_header_params(config, model, params) -> Dict:
    """Model columns for the data files, honouring WRFLAG_MODELPAR."""
    if model.writes_model_params and not config.output.wrflag_modelpar:
        return {}
    return model.get_snana_header_params(params)

def model_dump_columns(config, model) -> tuple:
    """Model-contributed DUMP columns, honouring WRFLAG_MODELPAR."""
    if not config.output.wrflag_modelpar:
        return ()
    return tuple(model.dump_columns)


def compute_derived_values(columnar_data, full_results, kcor_filter_names,
                           n_sne, selection_result):
    """The distance moduli and per-filter peak magnitudes the tables report."""
    params = columnar_data['parameters']

    # the distance modulus applied at generation
    distance_moduli = np.asarray(params['mu'])

    filter_names = list(kcor_filter_names)
    if (full_results is not None
            and 'peakmags' in full_results
            and selection_result is not None
            and 'passing_mask' in selection_result):
        # DOFILT-masked: NULLFLOAT for a filter with no kept epoch (snlc_sim.c:9639)
        peakmags_all = np.array(selection_result['peakmags_masked'])
        peakmags_selected = peakmags_all[selection_result['passing_mask']]
        peak_magnitudes = {
            filt: peakmags_selected[:, i].astype(np.float32)
            for i, filt in enumerate(filter_names)
        }
    else:
        peak_magnitudes = {f: np.full(n_sne, -99.0) for f in filter_names}

    return distance_moduli, peak_magnitudes

def _hostgalz_columns(prefix, param_prefix, params, quantiles, n, stored):
    """A host slot's photo-z quantile grid and its (empty) LOGMASS(z) grid, as
    SNANA packs them: NBIN is the HOSTLIB's quantile count for a slot with a
    galaxy stored, the vector carries the galaxy's quantiles as the HOSTLIB
    has them (-9 for one it lacks) and -9 after (load_SNDATA_HOSTGAL,
    snlc_sim.c:26106). A slot with no galaxy stored keeps NBIN 0 and the -9
    initialisation."""
    quantile_keys = [f'{param_prefix}{name}' for name, _ in quantiles]
    percentiles = [p for _, p in quantiles]
    z = np.full((n, MXBIN_HOSTGALz_QUANTILE), SNANA_NULLFLOAT, dtype=np.float32)
    pct = np.full((n, MXBIN_HOSTGALz_QUANTILE), SNANA_NULLFLOAT, dtype=np.float32)
    if quantile_keys and all(k in params for k in quantile_keys):
        nq = len(percentiles)
        zq = np.stack([np.asarray(params[k], dtype=np.float32)
                       for k in quantile_keys], axis=1)
        z[:, :nq] = np.where(stored[:, None], zq, SNANA_NULLFLOAT)
        pct[:, :nq] = np.where(stored[:, None],
                               np.asarray(percentiles, dtype=np.float32)[None, :],
                               SNANA_NULLFLOAT)
        nbin = np.where(stored, nq, 0)
    else:
        nbin = np.zeros(n, dtype=np.int64)
    return {
        f'{prefix}_NBIN_QUANTILE_ZPHOT': nbin.astype(np.int16),
        f'{prefix}_QUANTILE_ZPHOT': z,
        f'{prefix}_QUANTILE_PERCENT': pct,
        f'{prefix}_NBIN_LOGMASS_ZGRID': np.zeros(n, dtype=np.int16),
        f'{prefix}_LOGMASS_ZGRID': np.full((n, MXBIN_HOSTGALz), SNANA_NULLFLOAT, dtype=np.float32),
        f'{prefix}_LOGMASS_VALGRID': np.full((n, MXBIN_HOSTGALz), SNANA_NULLFLOAT, dtype=np.float32),
        f'{prefix}_LOGMASS_ERRGRID': np.full((n, MXBIN_HOSTGALz), SNANA_NULLFLOAT, dtype=np.float32),
    }


def _galid_unique(params, key, unset, empty):
    """HOSTGAL_OBJID_UNIQUE, CID * 57 + the slot's drawn offset
    (set_GALID_UNIQUE, sntools_host.c:9085), ``empty`` on a slot with no
    host; ``unset`` without HOSTLIB_GALID_UNIQUE."""
    offset = params.get(f'{key}galid_unique_offset')
    if offset is None:
        return unset
    offset = np.asarray(offset, dtype=np.int64)
    cid = np.asarray(params['cid'], dtype=np.int64)
    return np.where(offset >= 0, cid * 57 + offset, np.int64(empty))


def _neighbour_columns(prefix, params, n, bands, sentinel, prop_null, dmucor,
                       mwxt, redshifts_obs_helio, redshift_err):
    """The HEAD block of one neighbour slot (HOSTGAL2, HOSTGAL3), from the
    ``host_<prefix>_*`` values the DDLR sort produced; every column carries
    the no-neighbour sentinel where the slot is unmatched."""
    key = f'host_{prefix}_'
    data = {}
    data[f'{prefix}_OBJID'] = np.array(
        params.get(f'{key}GALID', np.zeros(n, dtype=np.int64))).astype(np.int64)
    data[f'{prefix}_FLAG'] = np.zeros(n, dtype=np.int16)
    for hostlib_key, fits_key in (('ZPHOT', 'PHOTOZ'),
                                  ('ZPHOT_ERR', 'PHOTOZ_ERR')):
        data[f'{prefix}_{fits_key}'] = np.asarray(
            params.get(f'{key}{hostlib_key}', sentinel)).astype(np.float32)

    ztrue = np.asarray(params.get(f'{key}ZTRUE', sentinel))
    valid = ztrue > -8.0
    is_true = valid & (np.asarray(params.get(f'{key}ZSPEC_TRUEMATCH', np.zeros(n))) > 0.5)
    data[f'{prefix}_SPECZ'] = np.where(
        valid, np.where(is_true, redshifts_obs_helio, ztrue),
        SNANA_NULLFLOAT).astype(np.float32)
    data[f'{prefix}_SPECZ_ERR'] = np.where(
        valid, np.where(is_true, redshift_err, 0.0005),
        SNANA_NULLFLOAT).astype(np.float32)

    data[f'{prefix}_RA'] = np.asarray(
        params.get(f'{key}RA_GAL',
                   np.full(n, HOSTGAL_COORD_NULL))).astype(np.float64)
    data[f'{prefix}_DEC'] = np.asarray(
        params.get(f'{key}DEC_GAL',
                   np.full(n, HOSTGAL_COORD_NULL))).astype(np.float64)
    data[f'{prefix}_SNSEP'] = np.asarray(
        params.get(f'{key}SNSEP', sentinel)).astype(np.float32)
    data[f'{prefix}_DDLR'] = np.asarray(
        params.get(f'{key}DDLR', sentinel)).astype(np.float32)
    # declared per slot but filled only for the matched host
    # (sntools_dataformat_fits.c:669 vs :1938), so it reads back as 0
    data[f'{prefix}_CONFUSION'] = np.zeros(n, dtype=np.float32)

    # as for the primary host (load_SNHOSTGAL_perGal, sntools_host.c:7502)
    for prop in ('LOGMASS', 'LOGSFR', 'LOGsSFR', 'COLOR'):
        data[f'{prefix}_{prop}'] = np.asarray(params.get(
            f'{key}{prop}_OBS',
            params.get(f'{key}{prop}', prop_null))).astype(np.float32)
        data[f'{prefix}_{prop}_ERR'] = np.asarray(params.get(
            f'{key}{prop}_ERR', prop_null)).astype(np.float32)

    # missing shape/ID columns: reset values if matched (sntools_host.c:8638-8642)
    matched = data[f'{prefix}_OBJID'] > 0
    shape_null = np.where(matched, HOSTGAL_SHAPE_NULL,
                          HOSTGAL_PROPERTY_NULL).astype(np.float32)
    objid_null = np.where(matched, -9, 0).astype(np.int64)
    data[f'{prefix}_ELLIPTICITY'] = np.asarray(
        params.get(f'{key}ELLIPTICITY', shape_null)).astype(np.float32)
    data[f'{prefix}_OBJID2'] = np.asarray(
        params.get(f'{key}OBJID2', objid_null)).astype(np.int64)
    data[f'{prefix}_SQRADIUS'] = np.asarray(
        params.get(f'{key}SQRADIUS', shape_null)).astype(np.float32)
    data[f'{prefix}_OBJID_UNIQUE'] = _galid_unique(params, key, objid_null, 0)

    # 999 without a neighbour (sntools.c:9276)
    mag_null = np.full(n, HOSTGAL_MAG_NULL, dtype=np.float32)
    magerr_null = np.where(matched, SNANA_NULLFLOAT,
                           HOSTGAL_MAG_NULL).astype(np.float32)
    for k, filt in enumerate(bands):
        raw = params.get(f'{key}{filt}_obs')
        mag_val = (mag_null if raw is None
                   else np.asarray(raw) + dmucor + mwxt[:, k])
        data[f'{prefix}_MAG_{filt}'] = np.where(
            matched, mag_val, HOSTGAL_MAG_NULL).astype(np.float32)
    for filt in bands:
        err_val = np.asarray(
            params.get(f'{key}{filt}_obs_err', magerr_null))
        data[f'{prefix}_MAGERR_{filt}'] = np.where(
            matched, err_val, HOSTGAL_MAG_NULL).astype(np.float32)
    return data


def build_header_columns(columnar_data, config, resources, kcor_filter_names,
                         distance_moduli, model, n_sne, peak_magnitudes,
                         to_written_cid) -> Dict:
    """The HEAD columns as {name: array}.

    The column set and their order follow ``wr_snfitsio_init_head``
    (sntools_dataformat_fits.c:283): every column SNANA writes, in SNANA's
    order, carrying SNANA's sentinel where there is no value to report.
    """
    params = columnar_data['parameters']
    obs_counts = columnar_data['obs_counts']
    metadata = columnar_data['metadata']
    n = n_sne
    # SNDATA_FILTER: the band set for every per-band column family
    bands = [filter_band(f) for f in kcor_filter_names]
    sentinel = np.full(n, SNANA_NULLFLOAT, dtype=np.float32)

    ptrobs_max = np.cumsum(obs_counts, dtype=np.int32)
    ptrobs_min = np.concatenate([[1], ptrobs_max[:-1] + 1]).astype(np.int32)

    data = {}

    # the model's columns; SIM_TEMPLATE_INDEX and SIM_TYPE_NAME are SNANA's
    # own columns, written whatever WRFLAG_MODELPAR says
    header_params = dict(model.get_snana_header_params(params))
    template_index = np.asarray(
        header_params.pop('SIM_TEMPLATE_INDEX', np.zeros(n)), dtype=np.int32)
    type_name = header_params.pop('SIM_TYPE_NAME', None)
    model_params = ({} if model.writes_model_params
                    and not config.output.wrflag_modelpar else header_params)

    # SUBSURVEY per event, blank where the entry names none
    # (sntools_dataformat_fits.c:317, :1432-1438, :1589)
    if subsurvey_flag(resources):
        names = np.array([f.subsurvey for f in resources.fields], dtype='S40')
        data['SUBSURVEY'] = names[np.asarray(params['field_idx'], dtype=np.int32)]
    cids = to_written_cid(params['cid']).astype(np.int32)
    if config.output.append_snid_sedindex > 0:
        # SNANA APPEND_SNID_SEDINDEX: SNID string = "{CID}-{TEMPLATE_INDEX}"
        data['SNID'] = np.array(
            [f"{int(c)}-{int(t)}" for c, t in zip(cids, template_index)],
            dtype=f'S{MXCHAR_CCID}')
    else:
        # the CID as SNANA prints it, unpadded (snlc_sim.c:25393)
        data['SNID'] = cids.astype(str).astype(f'S{MXCHAR_CCID}')
    data['FAKE'] = np.full(n, 2, dtype=np.int16)
    data['RA'] = metadata['ra'].astype(np.float64)
    data['DEC'] = metadata['dec'].astype(np.float64)
    # Atmosphere/DCR band-averaged true coords (SNDATA.RA_AVG_BAND,
    # snlc_sim.c:25853). All RA_AVG columns precede all DEC_AVG ones.
    if 'ra_avg_band' in metadata:
        for k, band in enumerate(metadata['avg_band_names']):
            data[f'RA_AVG_{band}'] = metadata['ra_avg_band'][:, k].astype(np.float64)
        for k, band in enumerate(metadata['avg_band_names']):
            data[f'DEC_AVG_{band}'] = metadata['dec_avg_band'][:, k].astype(np.float64)
    # SNDATA.PIXSIZE = SIMLIB_OBS_GEN.PIXSIZE (snlc_sim.c:25479), per SN
    # from its SIMLIB entry.
    pixsizes = np.asarray(resources.flat_field_index['pixsizes'],
                          dtype=np.float32)
    data['PIXSIZE'] = pixsizes[np.asarray(params['field_idx'], dtype=np.int32)]
    data['NXPIX'] = np.full(n, -9, dtype=np.int16)
    data['NYPIX'] = np.full(n, -9, dtype=np.int16)
    specid_pass = np.asarray(params['specid_pass'], dtype=bool)
    sim_gentype, sntype = per_event_types(config, resources, params, n,
                                          specid_pass)
    data['SNTYPE'] = sntype
    data['NOBS'] = obs_counts.astype(np.int32)
    data['PTROBS_MIN'] = ptrobs_min
    data['PTROBS_MAX'] = ptrobs_max
    data['MWEBV'] = metadata['mwebv'].astype(np.float32)
    data['MWEBV_ERR'] = np.asarray(params['mwebv_err']).astype(np.float32)

    # REDSHIFT_FINAL = GENLC.REDSHIFT_CMB_SMEAR (snlc_sim.c:25414)
    redshifts_obs_helio = np.asarray(params['redshift_helio']).astype(np.float32)
    redshifts_obs_cmb = np.asarray(params['redshift_final']).astype(np.float32)
    redshift_err = np.asarray(params['redshift_smear_err']).astype(np.float32)
    data['REDSHIFT_HELIO'] = redshifts_obs_helio
    data['REDSHIFT_HELIO_ERR'] = redshift_err
    data['REDSHIFT_FINAL'] = redshifts_obs_cmb
    data['REDSHIFT_FINAL_ERR'] = redshift_err

    host_photoz = params.get('host_ZPHOT', sentinel).astype(np.float32)
    host_photoz_err = params.get('host_ZPHOT_ERR', sentinel).astype(np.float32)
    has_photoz = (host_photoz_err > 0) & (host_photoz_err < 9.0)

    # zsource_to_SNDATA; bits ZHOST_SPEC=1, ZSN_SPEC=2, ZHOST_PHOT=4 (sndata.h)
    redshift_flag = params['redshift_flag']
    mask_z_source = np.zeros(n, dtype=np.int16)
    mask_z_source[redshift_flag == 2] |= 1
    mask_z_source[redshift_flag == 1] |= 2
    mask_z_source[has_photoz] |= 4
    data['MASK_REDSHIFT_SOURCE'] = mask_z_source

    data['VPEC'] = np.array(params['vpec'], dtype=np.float32)
    data['VPEC_ERR'] = np.array(params['vpec_err'], dtype=np.float32)
    # smeared LENSDMU and its error (snlc_sim.c:25422-25423); truth is SIM_LENSDMU
    data['LENSDMU'] = np.asarray(params['lensdmu_smear'], dtype=np.float32)
    data['LENSDMU_ERR'] = np.asarray(params['lensdmu_err'], dtype=np.float32)

    # drawn galaxy versus DDLR match (snlc_sim.c:26128 vs sntools.c:9163)
    if 'host_GALID' in params:
        drawn_galid = np.asarray(params['drawn_GALID']).astype(np.int64)
        ranked_galid = np.asarray(params['host_GALID']).astype(np.int64)
        nmatch, nmatch2 = params['host_NMATCH'], params['host_NMATCH2']
    else:
        drawn_galid = ranked_galid = nmatch = nmatch2 = np.zeros(
            n, dtype=np.int64)
    data['HOSTGAL_NMATCH'] = np.array(nmatch).astype(np.int16)
    data['HOSTGAL_NMATCH2'] = np.array(nmatch2).astype(np.int16)

    # Neighbour GALIDs null to -9 (reset_SNHOSTGAL_DDLR_SORT).
    matched = np.asarray(data['HOSTGAL_NMATCH2']) > 0
    host_galid = np.where(matched, ranked_galid, 0).astype(np.int64)
    data['HOSTGAL_OBJID'] = host_galid
    data['HOSTGAL_FLAG'] = np.zeros(n, dtype=np.int16)

    data['HOSTGAL_PHOTOZ'] = host_photoz
    data['HOSTGAL_PHOTOZ_ERR'] = host_photoz_err
    host_ztrue = np.asarray(params.get('host_ZTRUE', sentinel))
    valid_host = host_galid > 0
    is_true = valid_host & (np.asarray(params.get('host_ZSPEC_TRUEMATCH', np.ones(n))) > 0.5)
    data['HOSTGAL_SPECZ'] = np.where(
        valid_host, np.where(is_true, redshifts_obs_helio, host_ztrue),
        SNANA_NULLFLOAT).astype(np.float32)
    data['HOSTGAL_SPECZ_ERR'] = np.where(
        valid_host, np.where(is_true, redshift_err, 0.0005),
        SNANA_NULLFLOAT).astype(np.float32)

    host_coord_null = np.full(n, HOSTGAL_COORD_NULL)
    data['HOSTGAL_RA'] = params.get('host_RA_GAL', host_coord_null).astype(np.float64)
    data['HOSTGAL_DEC'] = params.get('host_DEC_GAL', host_coord_null).astype(np.float64)
    data['HOSTGAL_SNSEP'] = params.get('host_SNSEP', sentinel).astype(np.float32)
    data['HOSTGAL_DDLR'] = params.get('host_DDLR', sentinel).astype(np.float32)
    data['HOSTGAL_CONFUSION'] = params.get(
        'host_CONFUSION',
        np.full(n, HOSTGAL_CONFUSION_NULL, dtype=np.float32)).astype(np.float32)

    # observed property, else the true one; exact basenames (sndata.h:85-88)
    prop_null = np.full(n, HOSTGAL_PROPERTY_NULL, dtype=np.float32)
    for prop in ('LOGMASS', 'LOGSFR', 'LOGsSFR', 'COLOR'):
        data[f'HOSTGAL_{prop}'] = params.get(
            f'host_{prop}_OBS',
            params.get(f'host_{prop}', prop_null)).astype(np.float32)
        data[f'HOSTGAL_{prop}_ERR'] = params.get(
            f'host_{prop}_ERR', prop_null).astype(np.float32)

    shape_null = np.where(host_galid > 0, HOSTGAL_SHAPE_NULL,
                          HOSTGAL_PROPERTY_NULL).astype(np.float32)
    data['HOSTGAL_ELLIPTICITY'] = params.get(
        'host_ELLIPTICITY', shape_null).astype(np.float32)
    objid_null = np.where(host_galid > 0, -9, 0).astype(np.int64)
    data['HOSTGAL_OBJID2'] = params.get(
        'host_OBJID2', objid_null).astype(np.int64)
    data['HOSTGAL_SQRADIUS'] = params.get(
        'host_SQRADIUS', shape_null).astype(np.float32)
    data['HOSTGAL_OBJID_UNIQUE'] = _galid_unique(params, 'host_', objid_null,
                                                 -9)

    # HOSTLIB mag + DMUCOR + MW extinction (sntools_host.c:8805, :8955, :9287)
    dmucor = np.asarray(params.get('host_dmucor', np.zeros(n, dtype=np.float32)))
    mwxt = np.asarray(params['host_mwxt_mag'], dtype=np.float32)

    magerr_null = np.where(host_galid > 0, SNANA_NULLFLOAT,
                           HOSTGAL_MAG_NULL).astype(np.float32)
    for k, filt in enumerate(bands):
        raw = params.get(f'host_{filt}_obs')
        data[f'HOSTGAL_MAG_{filt}'] = (
            np.full(n, HOSTGAL_MAG_NULL, dtype=np.float32) if raw is None
            else (np.asarray(raw) + dmucor + mwxt[:, k]).astype(np.float32))
    for filt in bands:
        # Undefined once a host is matched, still 999 when none is.
        data[f'HOSTGAL_MAGERR_{filt}'] = params.get(
            f'host_{filt}_obs_err', magerr_null).astype(np.float32)

    # the photo-z quantile and LOGMASS(z) grids of each host slot follow
    # its magnitudes (wr_snfitsio_init_host, sntools_dataformat_fits.c:626-716)
    quantiles = ([] if resources.hostlib is None
                 else resources.hostlib.zphot_quantiles)
    data.update(_hostgalz_columns('HOSTGALz', 'host_', params, quantiles, n,
                                  matched))

    for prefix in HOSTGAL_PREFIXES[1:]:
        data.update(_neighbour_columns(
            prefix, params, n, bands, sentinel, prop_null, dmucor, mwxt,
            redshifts_obs_helio, redshift_err))
        data.update(_hostgalz_columns(
            f'{prefix}z', f'host_{prefix}_', params, quantiles, n,
            np.asarray(data[f'{prefix}_SPECZ']) > -8.0))

    for filt in bands:
        data[f'HOSTGAL_SB_FLUXCAL_{filt}'] = params.get(
            f'host_SB_FLUXCAL_{filt}',
            np.full(n, HOSTGAL_SB_NULL, dtype=np.float32)).astype(np.float32)

    # no DDLR match: HOSTGAL_ columns take the no-host null, except SB_FLUXCAL
    unmatched = ~matched
    if unmatched.any():
        for col, value in list(data.items()):
            if (not col.startswith('HOSTGAL')
                    or col.startswith('HOSTGAL_SB_FLUXCAL')
                    or col.endswith(('NMATCH', 'NMATCH2'))):
                continue
            null = HOSTGAL_NO_HOST_NULL.get(
                hostgal_null_key(col, bands))
            if null is None:
                continue
            arr = np.asarray(value)
            data[col] = np.where(unmatched, null, arr).astype(arr.dtype)

    peak_mjds = params['peakmjd'].astype(np.float64)
    data['PEAKMJD'] = peak_mjds.astype(np.float32)
    data['MJD_TRIGGER'] = np.asarray(params['mjd_trigger']).astype(np.float32)
    data['MJD_DETECT_FIRST'] = np.asarray(
        params['mjd_detect_first']).astype(np.float32)
    data['MJD_DETECT_LAST'] = np.asarray(
        params['mjd_detect_last']).astype(np.float32)
    # SDSS only (snlc_sim.c:25404)
    if resources.survey_name.upper() == 'SDSS':
        non1a = model.sim_model_index in (
            MODEL_INDEX_NON1ASED, MODEL_INDEX_NON1AGRID)
        search_type = 110 if non1a else 120
    else:
        search_type = int(SNANA_NULLFLOAT)
    data['SEARCH_TYPE'] = np.full(n, search_type, dtype=np.int32)

    model_name = sim_model_name(config, model)
    data['SIM_MODEL_NAME'] = np.array([model_name] * n, dtype='S32')
    data['SIM_MODEL_INDEX'] = np.full(n, model.sim_model_index, dtype=np.int16)
    data['SIM_GENTYPE'] = sim_gentype.astype(np.int16)
    # both serialise SNDATA.SIM_GENTYPE (sntools_dataformat_fits.c:2000-2005)
    model_params.pop('SIM_TYPE_INDEX', None)
    data['SIM_TYPE_INDEX'] = sim_gentype.astype(np.int16)
    # per event where the model names it, else the model's name
    if type_name is None:
        type_name = np.array([_sim_type_name(config, model) or model_name] * n)
    data['SIM_TYPE_NAME'] = np.asarray(type_name, dtype='S12')
    data['SIM_TEMPLATE_INDEX'] = template_index
    data['SIM_LIBID'] = metadata['field_id'].astype(np.int32)
    data['SIM_NGEN_LIBID'] = np.full(n, 1, dtype=np.int32)
    data['SIM_NOBS_UNDEFINED'] = np.full(n, 0, dtype=np.int32)
    data['SIM_SEARCHEFF_MASK'] = params['sim_searcheff_mask'].astype(np.int32)
    true_zcmb = np.asarray(params['sim_redshift_cmb']).astype(np.float32)
    true_zhel = np.asarray(params['sim_redshift_helio']).astype(np.float32)
    data['SIM_REDSHIFT_HELIO'] = true_zhel
    data['SIM_REDSHIFT_CMB'] = true_zcmb
    # the host-draw redshift; the matched host's is separate (snlc_sim.c:25460)
    z_matched = params.get('host_ZTRUE')
    data['SIM_REDSHIFT_HOST'] = np.asarray(
        params['sim_redshift_host']).astype(np.float32)
    # SNHOSTGAL_DDLR_SORT[0].REDSHIFT is zero until a host is matched.
    data['SIM_REDSHIFT_HOST_MATCH'] = (
        np.zeros(n, dtype=np.float32) if z_matched is None
        else np.asarray(z_matched).astype(np.float32))
    # REDSHIFT_FLAG: 0=NONE, 1=SNSPEC, 2=HOSTSPEC, 3=HOSTPHOT, 4=WRONGHOST
    data['SIM_REDSHIFT_FLAG'] = params['redshift_flag'].astype(np.int16)
    data['SIM_VPEC'] = np.array(params['vpec_true'], dtype=np.float32)
    # the drawn class; jax-only, so only a run declaring classes carries it
    if config.model.subpops:
        data['SIM_SUBPOP'] = np.asarray(params['subpop']).astype(np.int16)
    data['SIM_HOSTLIB_GALID'] = np.where(
        drawn_galid > 0, drawn_galid, SIM_HOSTLIB_GALID_NULL).astype(np.int64)
    # HOSTLIB_STOREPAR order; a stored name may carry the _TRUE rename
    storepar = config.host.storepar

    def _storepar_rank(key):
        name = key[len('sim_hostlib_'):]
        base = name[:-len('_TRUE')] if name.endswith('_TRUE') else name
        for cand in (name, base):
            if cand in storepar:
                return storepar.index(cand)
        return len(storepar)

    hostlib_keys = [k for k in params if k.startswith('sim_hostlib_')]
    for key in sorted(hostlib_keys, key=_storepar_rank):
        data[f"SIM_HOSTLIB({key[len('sim_hostlib_'):]})"] = np.asarray(
            params[key], dtype=np.float32)

    # The TRUE host's DDLR/separation, which differs from HOSTGAL_DDLR when
    # the DDLR-closest match is a neighbour (snlc_sim.c:26052); zero without
    # a HOSTLIB, where SNANA leaves SNDATA untouched.
    if resources.hostlib is not None:
        data['SIM_SNHOST_DDLR'] = np.asarray(
            params['host_DDLR_TRUE'], dtype=np.float32)
        data['SIM_SNHOST_SEP'] = np.asarray(
            params['host_SNSEP_TRUE'], dtype=np.float32)
    else:
        data['SIM_SNHOST_DDLR'] = np.zeros(n, dtype=np.float32)
        data['SIM_SNHOST_SEP'] = np.zeros(n, dtype=np.float32)

    data['SIM_DLMU'] = distance_moduli.astype(np.float32)
    data['SIM_LENSDMU'] = np.asarray(params['lensdmu'], dtype=np.float32)
    data['SIM_MUSHIFT'] = np.asarray(params['mushift'], dtype=np.float32)
    data['SIM_RA'] = metadata['ra'].astype(np.float64)
    data['SIM_DEC'] = metadata['dec'].astype(np.float64)
    data['SIM_MWEBV'] = metadata['sim_mwebv'].astype(np.float32)
    sim_peak_mjds = params['peakmjd_true'].astype(np.float64)
    data['SIM_PEAKMJD'] = sim_peak_mjds.astype(np.float32)
    data['SIM_MJD_EXPLODE'] = np.full(n, SNANA_NULLFLOAT, dtype=np.float32)
    # SIM_MAGSMEAR_COH: coherent magnitude smearing (per-SN for NON1ASED, else 0)
    if 'magsmear' in params:
        data['SIM_MAGSMEAR_COH'] = np.asarray(params['magsmear'], dtype=np.float32)
    else:
        data['SIM_MAGSMEAR_COH'] = np.full(n, 0.0, dtype=np.float32)
    data['SIM_WGT_POPULATION'] = np.asarray(params['wgt_population'], dtype=np.float32)

    # drawn for every model, sentinel where none is configured (snlc_sim.c:14325)
    data['SIM_AV'] = np.asarray(params['sim_av'], dtype=np.float32)
    data['SIM_RV'] = np.asarray(params['sim_rv'], dtype=np.float32)

    for col_name, value_array in model_params.items():
        # integer columns keep their type (SNANA writes the indices as J)
        arr = np.asarray(value_array)
        data[col_name] = (arr.astype(np.float32)
                          if np.issubdtype(arr.dtype, np.floating) else arr)

    # SNANA computes a peak mag only for filters in the SN's SIMLIB entry.
    band_mask = columnar_data['parameters'].get('simlib_band_mask')
    for bi, (fn, filt) in enumerate(zip(kcor_filter_names, bands)):
        peak = np.asarray(peak_magnitudes[fn], dtype=np.float32)
        if band_mask is not None:
            peak = np.where(np.asarray(band_mask)[:, bi], peak,
                            SNANA_NULLFLOAT).astype(np.float32)
        data[f'SIM_PEAKMAG_{filt}'] = peak
    for filt in bands:
        # EXPOSURE_TIME_FILTER scale (snlc_sim.c:25537)
        data[f'SIM_EXPOSURE_{filt}'] = np.full(
            n, config.survey.exposure_time.get(filt, 1.0), dtype=np.float32)

    # Host/SN peak flux ratio in a 1" aperture (snlc_sim.c:26158), written
    # whenever a HOSTLIB is in use; NULLFLOAT when host mags are unmodelled.
    if resources.hostlib is not None:
        for fn, filt in zip(kcor_filter_names, bands):
            galmag = params.get(f'host_GALMAG_{filt}')
            if galmag is None:
                data[f'SIM_GALFRAC_{filt}'] = sentinel.copy()
                continue
            peakmag = np.asarray(peak_magnitudes[fn])
            data[f'SIM_GALFRAC_{filt}'] = np.where(
                peakmag < 98.0,
                np.power(10.0, -0.4 * (np.asarray(galmag) - peakmag)),
                SNANA_NULLFLOAT).astype(np.float32)

    # NSUBSAMPLE_MARK: cyclic label over written SNe = NGENLC_WRITE % N, with
    # NGENLC_WRITE incremented before assignment (snlc_sim.c:30728,30772); -9 off.
    _nsub = config.output.nsubsample_mark
    if _nsub > 1:
        data['SIM_SUBSAMPLE_INDEX'] = ((np.arange(n) + 1) % _nsub).astype(np.int16)
    else:
        data['SIM_SUBSAMPLE_INDEX'] = np.full(n, -9, dtype=np.int16)

    return data

def build_phot_columns(columnar_data, resources, snrmon_colname,
                       total_observations) -> Dict:
    """The PHOT columns as {name: array}, in the order the FITS header
    declares them (wr_snfitsio_init_phot, sntools_dataformat_fits.c:773)."""
    obs = columnar_data['observations']
    params = columnar_data['parameters']
    n_obs = total_observations

    data = {}

    data['MJD'] = obs['mjd'].astype(np.float64)
    # BAND is as wide as the kcor's longest filter name, FIELD as wide as
    # MXCHAR_FIELDNAME (sntools_dataformat_fits.c:833, :851); IMGNUM, DETNUM,
    # XPIX and YPIX exist only under USE_SIMLIB_IDEXPT
    data['BAND'] = LookupColumn(
        np.array(columnar_data['filter_names'],
                 dtype=f'S{resources.filter_name_width}'), obs['filter_idx'])
    # FIELD: the SN's SIMLIB entry field name on every epoch; 'VOID' only
    # when the SIMLIB defines no field (FIELD_NONAME, sntools.h:117).
    field_names = np.array([f.field_name for f in resources.fields],
                           dtype=f'S{MXCHAR_FIELDNAME}')
    field_names = np.where(field_names == b'FIELD_NONAME', b'VOID', field_names)
    counts = np.asarray(columnar_data['obs_counts'])
    data['FIELD'] = LookupColumn(
        field_names, np.repeat(np.asarray(params['field_idx'], dtype=np.int32),
                               counts))

    fluxcal = np.asarray(obs['fluxcal'], dtype=np.float32)
    fluxcalerr = np.asarray(obs['fluxcalerr'], dtype=np.float32)
    photflag = np.where(obs['is_detection'], params['photflag_detect'],
                        0).astype(np.int32)
    photflag = np.where(obs['is_trigger'],
                        photflag | params['photflag_trigger'], photflag)
    # the SIMLIB's APPEND flag on its appended rows (snlc_sim.c:25633)
    photflag |= np.asarray(obs['append_photflag'].astype(np.int32))
    # snlc_sim.c:25637-25638
    pf_saturate = params['photflag_saturate']
    if pf_saturate != 0:
        photflag = np.where(obs['is_saturated'], photflag | pf_saturate, photflag)
    # max-SNR and nearest-peak epochs (snlc_sim.c:25640-25644)
    pf_snrmax = params['photflag_snrmax']
    if pf_snrmax > 0:
        photflag = np.where(obs['is_snrmax_epoch'], photflag | pf_snrmax, photflag)
    pf_nearpeak = params['photflag_nearpeak']
    if pf_nearpeak > 0:
        photflag = np.where(obs['is_nearpeak_epoch'], photflag | pf_nearpeak, photflag)
    data['PHOTFLAG'] = photflag
    data['PHOTPROB'] = np.asarray(obs['photprob'], dtype=np.float32)
    data['FLUXCAL'] = fluxcal
    data['FLUXCALERR'] = fluxcalerr

    # NEA units replace the Gaussian triple (sntools_dataformat_fits.c:864)
    if params['nea_psf_unit']:
        data['PSF_NEA'] = obs['nea'].astype(np.float32)
    else:
        data['PSF_SIG1'] = obs['psf1'].astype(np.float32)
        data['PSF_SIG2'] = obs['psf2'].astype(np.float32)
        data['PSF_RATIO'] = obs['psf_ratio'].astype(np.float32)
    data['SKY_SIG'] = obs['skysig'].astype(np.float32)
    data['SKY_SIG_T'] = obs['template_skysig'].astype(np.float32)
    data['RDNOISE'] = obs['readnoise'].astype(np.float32)
    data['ZEROPT'] = obs['zpt'].astype(np.float32)
    data['ZEROPT_ERR'] = obs['zpterr'].astype(np.float32)
    # SNANA leaves TEXPOSE at zero; the SIMLIB S: row carries no exposure.
    data['TEXPOSE'] = np.broadcast_to(np.float32(0.0), n_obs)
    data['GAIN'] = obs['gain'].astype(np.float32)

    # Atmosphere / DCR per-obs columns (WRFLAG_ATMOS, fits.c:889)
    if 'airmass' in obs:
        data['dRA'] = np.asarray(obs['d_ra'], dtype=np.float32)
        data['dDEC'] = np.asarray(obs['d_dec'], dtype=np.float32)
        data['AIRMASS'] = np.asarray(obs['airmass'], dtype=np.float32)
        data['SIM_DCR_dRA'] = np.asarray(obs['sim_dcr_dra'], dtype=np.float32)
        data['SIM_DCR_dDEC'] = np.asarray(obs['sim_dcr_ddec'], dtype=np.float32)
        data['SIM_DCR_dMAG'] = np.asarray(obs['sim_dcr_dmag'], dtype=np.float32)

    data['SIM_MAGOBS'] = np.asarray(obs['mag_noiseless'], dtype=np.float32)

    if snrmon_colname:
        data[snrmon_colname] = np.asarray(obs['snr_mon'], dtype=np.float32)

    return data


def filters_string(filter_names) -> str:
    """Concatenated single-char band labels (SNANA FILTERS header)."""
    return ''.join(filter_band(fn) for fn in filter_names)


def build_columnar_data(simulator, results, selection_result):
    """Build columnar output data from JAX selection results.

    Converts JAX selection output (``results``, and ``selection_result``
    from :func:`apply_selection_effects`) to NumPy for the SNANA writer.

    Returns ``{'parameters': {...}, 'observations': {...},
    'obs_counts': np.ndarray, 'metadata': {...}, 'filter_names': [...]}``
    (``filter_names`` absent when no SN passes), with per-SN/per-obs
    arrays restricted to passing SNe.
    """
    passing = np.asarray(selection_result['passing_mask'])
    detection_mask = selection_result['detection_mask']
    valid_mask = np.asarray(selection_result['valid_mask'])

    n_sne = len(passing)
    passing_indices = np.where(passing)[0]
    n_passing = len(passing_indices)

    if n_passing == 0:
        return {'parameters': {}, 'observations': {}, 'obs_counts': np.array([]),
                'metadata': {}}

    params_out = {}
    for k, v in results['parameters'].items():
        arr = np.asarray(v)
        if arr.ndim >= 1 and arr.shape[0] == n_sne:
            params_out[k] = arr[passing_indices]

    for sel_key in ('redshift_flag', 'cutmask',
                    'redshift_helio', 'redshift_final',
                    'redshift_smear_err',
                    'sim_searcheff_mask', 'sim_searcheff_spec', 'specid_pass',
                    'mwxt_mag', 'host_mwxt_mag', 'simlib_band_mask',
                    'mjd_detect_first', 'mjd_detect_last', 'mjd_trigger'):
        params_out[sel_key] = np.asarray(selection_result[sel_key])[passing_indices]

    params_out['photflag_detect'] = simulator.searcheff_photflag_detect
    params_out['photflag_trigger'] = simulator.searcheff_photflag_trigger
    params_out['photflag_saturate'] = simulator.photflag_saturate
    params_out['photflag_snrmax'] = simulator.instrument.photflag_snrmax
    params_out['photflag_nearpeak'] = simulator.instrument.photflag_nearpeak
    params_out['nea_psf_unit'] = simulator.nea_psf_unit

    sb_stacked = results.get('host_sb_fluxcal_stacked')
    if sb_stacked is not None:
        sb_stacked_np = np.asarray(sb_stacked)[passing_indices]
        filter_names_all = simulator.kcor.get_filter_names()
        for i, fname in enumerate(filter_names_all):
            short = filter_band(fname)
            params_out[f'host_SB_FLUXCAL_{short}'] = sb_stacked_np[:, i]

    # Host mag in the 1" aperture SIM_GALFRAC references (snlc_sim.c:26147)
    galmag = results.get('host_galmag_galfrac')
    if galmag is not None:
        galmag_np = np.asarray(galmag)[passing_indices]
        for i, fname in enumerate(simulator.kcor.get_filter_names()):
            short = filter_band(fname)
            params_out[f'host_GALMAG_{short}'] = galmag_np[:, i]

    # the written epochs: the passing events' valid ones, in event and epoch order
    epoch_count = np.asarray(results['epoch_count'])[passing_indices]
    flat_kept, seg = span_indices(
        np.asarray(results['epoch_start'])[passing_indices], epoch_count)
    valid = np.asarray(valid_mask).reshape(-1)[flat_kept]
    written = np.flatnonzero(valid)
    flat_w, seg_w = flat_kept[written], seg[written]
    rows = passing_indices[seg_w]
    n_obs = len(flat_w)
    obs_schedule = results['observation_schedule']
    filter_names = list(simulator.photometry.filter_names)
    in_place = np.array_equal(flat_w, np.arange(n_obs))

    def column(array):
        if in_place and np.ndim(array) == 1 and len(array) == n_obs:
            return array
        return np.asarray(array).reshape(-1)[flat_w]

    def simlib(key, transform=None):
        return column(simlib_epochs(simulator, results, key, transform))

    fi_obs = column(obs_schedule['filter_indices'])

    # the saturation-masked flux, error and SNR (snlc_sim.c:28544-28548)
    fluxcal = column(results['fluxcals_noisy'])
    fluxcalerr = column(results['fluxcalerrs'])
    snr = column(results['snrs'])

    # APPLYFLAG_MWEBV corrects with the map MWEBV (MWEBVfluxCor_to_SNDATA, snlc_sim.c:25862)
    if simulator.mw_dust.applyflag_mwebv:
        params_mw = results['parameters']
        rv = simulator.mw_dust.rv
        mwebv_map = np.asarray(params_mw['mwebv'], dtype=np.float64)[passing_indices]
        mwebv_err = np.asarray(params_mw['mwebv_err'], dtype=np.float64)[rows]
        mwebv_smear = np.asarray(params_mw['sim_mwebv'], dtype=np.float64)[rows]
        axav = np.asarray(simulator.mw_law_at_mean.extinction_curve(
            simulator.photometry.filter_mean_wavelengths, rv,
            av=rv * mwebv_map[:, None]), dtype=np.float64)
        axav_ev = np.broadcast_to(axav, (n_passing, axav.shape[-1]))
        mcor = rv * mwebv_map[seg_w] * axav_ev[seg_w, fi_obs]
        fcor = np.power(10.0, 0.4 * mcor)
        fluxcal = fluxcal * fcor
        fluxcalerr = fluxcalerr * fcor
        ratio = np.where(mwebv_smear > 0.0, mwebv_err / np.where(
            mwebv_smear > 0.0, mwebv_smear, 1.0), 0.0)
        flux_add = (np.power(10.0, 0.4 * mcor * ratio) - 1.0) * np.abs(fluxcal)
        fluxcalerr = np.sqrt(fluxcalerr ** 2 + flux_add ** 2)

    photprob_2d = selection_result['photprob_2d']
    observations = {
        'mjd': simlib('mjds'),
        'filter_idx': fi_obs,
        'mag_noiseless': column(results['magnitudes']),
        'fluxcal': fluxcal,
        'fluxcalerr': fluxcalerr,
        'snr': snr,
        'is_detection': column(detection_mask),
        'is_trigger': column(selection_result['trigger_mask']),
        'is_saturated': column(selection_result['saturated_mask']),
        'zpt': simlib('zptavgs'),
        'skysig': simlib('skysigs'),
        # SIMLIB per-epoch values the PHOT table reports verbatim
        'psf1': simlib('psf1s'),
        'psf2': simlib('psf2s'),
        'psf_ratio': simlib('psf_ratios'),
        'readnoise': simlib('readnoises'),
        'zpterr': simlib('zpterrs'),
        'gain': simlib('gains'),
        'template_skysig': simlib('template_skysigs', np.nan_to_num),
        'nea': simlib('neas'),
        'append_photflag': simlib('append_photflags'),
        'photprob': (column(photprob_2d) if photprob_2d is not None
                     else np.full(n_obs, SNANA_NULLFLOAT, dtype=np.float32)),
    }

    # written whenever the key is set, zeros included (snlc_sim.c:7439, :27786)
    if simulator.magmonitor_snr:
        observations['snr_mon'] = column(results['snrs_mon'])

    for key in ('snrmax_epoch', 'nearpeak_epoch'):
        if key in selection_result:
            observations[f'is_{key}'] = column(selection_result[key])

    obs_counts = np.bincount(seg_w, minlength=n_passing)

    params = results['parameters']
    metadata = {
        'ra': np.asarray(params['ra'])[passing_indices],
        'dec': np.asarray(params['dec'])[passing_indices],
        'mwebv': np.asarray(params['mwebv'], dtype=np.float32)[passing_indices],
        'mwebv_err': np.asarray(params['mwebv_err'], dtype=np.float32)[passing_indices],
        'sim_mwebv': np.asarray(params['sim_mwebv'], dtype=np.float32)[passing_indices],
        'field_id': np.asarray(obs_schedule['libid'])[passing_indices].astype(int),
    }

    if simulator.config.instrument.atmos_optmask > 0:
        atmosphere = results['atmosphere']
        for name, values in atmosphere.items():
            if name in ('ra_avg_band', 'dec_avg_band'):
                metadata[name] = values[passing_indices]
            else:
                observations[name] = column(values)
        metadata['avg_band_names'] = [filter_band(n) for n in simulator.filter_names]

    snrmax = np.full(n_passing, -np.inf)
    np.maximum.at(snrmax, seg_w, column(results['snrs']))
    metadata['snrmax'] = snrmax

    return {
        'parameters': params_out,
        'observations': observations,
        'obs_counts': obs_counts,
        'metadata': metadata,
        # what observations['filter_idx'] indexes
        'filter_names': filter_names,
    }
