"""SEARCHEFF and CUTWIN: which generated SNe the survey detects.

Combines detection, spectroscopic efficiency and CUTWIN in SNANA's order,
reporting the passing events and rejection reasons.
"""

import logging
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core import logging as log
from jax_snlc_core.cosmology import zhelio_to_zcmb
from jax_snlc_core.filters import filter_band
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import MAG_UNDEFINED, SNANA_NULLFLOAT
from jax_snlc_sim.hostlib.selection import gather_deferred_hostlib_columns
from jax_snlc_sim.photometry import (LOG_SAFE_FLOOR, compute_rest_wavelengths,
                                     host_mwxt_mag_per_band, mwxt_mag_per_band,
                                     per_filter_array)
from jax_snlc_core.random import stream_key
from jax_snlc_sim.sampler import PREFILTER_REASON_CODES
from jax_snlc_sim.cutwin import (
    CUTBIT_MAP, CUTMASK_ALL_PASS, GAP_UNSET, apply_generic_range_cuts, cutwin_all,
    per_sn_windows, prepare_cutwin, range_cut_quantities)
from jax_snlc_sim.spectroscopic_efficiency import (
    FLAG_ALL_FAIL, eff_outside_window, evaluate_searcheff, field_name_mask,
    is_filter_variable, prepare_searcheff, resolve_fieldlist_indices, split_fieldlist)
from jax_snlc_sim.detection import (
    evaluate_photprob, build_searcheff_config, detect_all, load_pipeline_curves,
    trigger_all)
from jax_snlc_sim.simlib import compute_dtseason_peak

logger = logging.getLogger(__name__)

# Indexed by the rejection code each stage assigns.
_REJECTION_CODE_LABELS = np.array(
    ['PASS', 'GENRANGE', 'NEPOCH', 'SEARCHEFF', 'SEARCHEFF', 'CUTWIN'],
    dtype='<U16')


def _build_validity_mask(results):
    """Build the per-SN/per-obs validity mask used by selection effects.

    Returns a `(n_sne, max_obs)` boolean array.
    """
    fi = jnp.array(results['observation_schedule']['filter_indices'])
    magnitudes = jnp.array(results['magnitudes'])
    fluxcals = jnp.array(results['fluxcals'])
    fluxcals_noisy = jnp.array(results['fluxcals_noisy'])
    fluxcalerrs = jnp.array(results['fluxcalerrs'])

    # NaN also marks an epoch outside the model's redshift range (DOFILT)
    valid = ((fi >= 0) & ~jnp.isnan(magnitudes) & ~jnp.isnan(fluxcals)
             & ~jnp.isnan(fluxcals_noisy) & ~jnp.isnan(fluxcalerrs))

    return valid & jnp.array(results['parameters']['prefilter_pass'])[:, None]


def _build_epoch_mask(cuts, valid_mask, n_obs_per_sn,
                      phases, snrs, rest_wavelengths, include_snr_filter=True):
    """Per-obs epoch mask: SNANA's gen_cutwin epoch filter (validity, the
    Trest pre-window, EPCUTWIN_LAMREST and (optionally) EPCUTWIN_SNRMIN
    (snlc_sim.c:24203-24235).
    """
    max_obs = valid_mask.shape[1]
    obs_presence = jnp.arange(max_obs)[None, :] < n_obs_per_sn[:, None]

    epcut = cuts
    lam_min, lam_max = epcut.epcutwin_lamrest
    rest_mask = (jnp.isfinite(rest_wavelengths)
                 & (rest_wavelengths >= lam_min)
                 & (rest_wavelengths <= lam_max))

    base = obs_presence & valid_mask & rest_mask
    base = (base & (phases >= cuts.trestmin_range[0])
            & (phases <= cuts.trestmax_range[1]))

    if include_snr_filter:
        # SNANA never reads the upper bound (snlc_sim.c:24234)
        snr_limits = epcut.epcutwin_snrmin
        snr_mask = jnp.isfinite(snrs)
        if np.isfinite(snr_limits[0]):
            snr_mask = snr_mask & (snrs >= snr_limits[0])
        return base & snr_mask
    return base


def _apply_dofilt_peakmag_mask(resources, params, peakmags, obs_schedule):
    """Apply DOFILT + z-range masking to peak magnitudes.

    Returns a new `(n_sne, n_filters)` JAX array (does NOT mutate inputs).
    Masks to -9 each (SN, filter) combination where the filter has no
    valid observation or the SN's heliocentric redshift falls outside the
    filter's observer-frame z-range.

    `params` requires 'sim_redshift_helio' and 'template_idx'
    (for per-template z-range lookup). `obs_schedule` needs
    'filter_indices' (n_sne, max_obs) JAX int.
    """
    z_arr = jnp.asarray(params['sim_redshift_helio'], dtype=jnp.float32)
    obs_filt = jnp.asarray(obs_schedule['filter_indices'])
    valid_obs = obs_filt >= 0
    n_filters = peakmags.shape[1]
    filt_eq = obs_filt[:, :, None] == jnp.arange(n_filters)[None, None, :]
    has_obs = jnp.any(valid_obs[:, :, None] & filt_eq, axis=1)

    z_min_table = jnp.asarray(resources.filter_z_min, dtype=jnp.float32)
    z_max_table = jnp.asarray(resources.filter_z_max, dtype=jnp.float32)
    tidx = jnp.asarray(params['template_idx'], dtype=jnp.int32)
    z_min = z_min_table[tidx]
    z_max = z_max_table[tidx]
    z_2d = z_arr[:, None]
    filt_invalid = (z_2d < z_min) | (z_2d >= z_max) | (z_max <= 0)

    return jnp.where(has_obs & ~filt_invalid,
                     jnp.asarray(peakmags, dtype=jnp.float32),
                     SNANA_NULLFLOAT)


def _build_searcheff_vars(resources, catalogue, season_boundaries, params,
                          obs_schedule, peakmags_masked,
                          sb_fluxcal_stacked=None):
    """Build the per-SN variable dict passed into SEARCHEFF maps.

    All ops are jnp primitives so the returned dict is safe to pass into
    the traced `prepare_searcheff`. Inputs are JAX arrays (traced or
    concrete). Field matching uses the int `field_idx` via `fieldlist_mask`
    at prepare time (no FIELDNAME string entry).

    `params` is the per-SN parameter dict (JAX arrays), typically
    `results['parameters']`. `peakmags_masked` is the `(n_sne, n_filters)`
    DOFILT-masked peak magnitudes produced by `_apply_dofilt_peakmag_mask`.
    Returns the `sn_vars` dict passed into `prepare_searcheff` /
    `_resolve_variable`.
    """
    z_helio = params['sim_redshift_helio']
    # the generated peak, not the estimate (LOAD_SEARCHEFF_DATA, snlc_sim.c:24667)
    peak_mjd = params['peakmjd_true']
    n_sne = peak_mjd.shape[0]
    sn_vars = {
        'sim_redshift_cmb': jnp.asarray(z_helio, dtype=jnp.float32),
        'REDSHIFT': jnp.asarray(z_helio, dtype=jnp.float32),
        'peakmjd': jnp.asarray(peak_mjd, dtype=jnp.float32),
        'PEAKMJD': jnp.asarray(peak_mjd, dtype=jnp.float32),
        'field_idx': jnp.asarray(params['field_idx'], dtype=jnp.int32),
    }

    if 'host_LOGMASS' in params:
        sn_vars['host_LOGMASS'] = jnp.asarray(params['host_LOGMASS'], dtype=jnp.float32)
        sn_vars['LOGMASS'] = sn_vars['host_LOGMASS']

    filter_names = list(resources.photometry.filter_names)
    for i, filt in enumerate(filter_names):
        filt_char = filter_band(filt)
        sn_vars[f'peakmag_{filt_char}'] = peakmags_masked[:, i]
        sn_vars[f'peakmag_{filt}'] = peakmags_masked[:, i]

    sn_vars.update(resources.model.sn_variables(params))

    mjds = jnp.asarray(obs_schedule['mjds'], dtype=jnp.float32)
    filter_indices = jnp.asarray(obs_schedule['filter_indices'])
    peak_mjds_2d = sn_vars['peakmjd'][:, None]
    dt_obs = jnp.abs(mjds - peak_mjds_2d)
    valid_obs = filter_indices >= 0
    dt_obs = jnp.where(valid_obs, dt_obs, 1e30)
    sn_vars['dtpeak'] = jnp.min(dt_obs, axis=1)
    sn_vars['DTPEAK'] = sn_vars['dtpeak']

    season_starts, season_ends = season_boundaries
    sn_vars['dtseason_peak'] = compute_dtseason_peak(
        sn_vars['peakmjd'], sn_vars['field_idx'],
        season_starts, season_ends)
    sn_vars['DTSEASON_PEAK'] = sn_vars['dtseason_peak']

    for key, val in params.items():
        if (key.startswith('host_')
                and hasattr(val, 'shape')
                and val.ndim == 1 and val.shape[0] == n_sne):
            sn_vars[key] = jnp.asarray(val, dtype=jnp.float32)

    if resources.hostlib is not None:
        # a plain HOSTLIB column reads the drawn host, LOGMASS the DDLR-ranked one
        selector = resources.hostlib.selector
        drawn = catalogue.columns[:, params['host_galaxy_index']]
        for name in {v for m in (resources.searcheff_specid_maps
                                 + resources.searcheff_zhost_maps)
                     for v in m.varnames}:
            if (name in selector.varnames and name not in sn_vars
                    and not is_filter_variable(name)):
                sn_vars[name] = drawn[selector.varnames.index(name)]

    if sb_fluxcal_stacked is not None:
        sb_jax = jnp.asarray(sb_fluxcal_stacked, dtype=jnp.float32)
        for i, filt in enumerate(filter_names):
            filt_char = filter_band(filt)
            sn_vars[f'host_SB_FLUXCAL_{filt_char}'] = sb_jax[:, i]

    # zero unless spectra came before the search (snlc_sim.c:9753)
    sn_vars['SNRSUM_REST_V'] = (
        jnp.asarray(params['snrsum_rest_v'], dtype=jnp.float32)
        if 'snrsum_rest_v' in params else jnp.zeros(n_sne, dtype=jnp.float32))

    return sn_vars


def load_searcheff(config, simlib, filter_name_to_index, spec_zhost) -> dict:
    """Load all SEARCHEFF maps + pipeline state, as Survey field values."""
    pipeline = load_pipeline_curves(
        config, simlib, filter_name_to_index)
    specid_maps, specid_flag, zhost_maps, zhost_flag = spec_zhost
    simlib_field_names = [f.field_name for f in simlib.fields]
    if specid_maps:
        resolve_fieldlist_indices(specid_maps, simlib_field_names)
    if zhost_maps:
        resolve_fieldlist_indices(zhost_maps, simlib_field_names)
    spec_file = config.survey.searcheff_spec_eff_file or ''
    has_spec = bool(spec_file) and spec_file.upper() not in ('', 'NONE', 'ONE')
    return dict(
        searcheff_curves=pipeline['curves'],
        searcheff_filter_masks=pipeline['filter_masks'],
        searcheff_photprob_maps=pipeline['photprob_maps'],
        searcheff_specid_maps=specid_maps,
        searcheff_specid_flag=specid_flag,
        evaluate_redshift_efficiency=(has_spec or config.survey._searcheff_apply_spec),
        searcheff_zhost_maps=zhost_maps,
        searcheff_zhost_flag=zhost_flag,
        searcheff_n_trigger_detections=pipeline['n_trigger_detections'],
        searcheff_trigger_logic=pipeline['trigger_logic'],
        searcheff_photflag_detect=pipeline['photflag_detect'],
        searcheff_photflag_trigger=pipeline['photflag_trigger'],
        searcheff_spec_scale=config.survey.searcheff_spec_scale,
    )


def prepare_search_efficiency(simulator):
    """Prepare numerical detection and trigger tables from the loaded maps,
    or None when there are none: SNANA then treats the search as 100%
    efficient (sntools_trigger.c:2041) and proceeds to SPECID/zHOST."""
    se = simulator
    if not se.searcheff_curves:
        return None
    filter_names = list(simulator.photometry.filter_names)
    config = build_searcheff_config(
        se.searcheff_curves,
        se.searcheff_filter_masks,
        se.searcheff_n_trigger_detections,
        simulator.survey.newmjd_dif,
        filter_names,
        [f.field_name for f in simulator.fields],
        apply_detect_single=simulator.survey.searcheff_apply_detect_single,
        zp_fluxcal=simulator.instrument.zp_fluxcal)
    config.pop("n_filters")
    return config


def build_cutwin(resources, cuts):
    """Prepare per-observation cut thresholds and filter selections."""
    filter_names = list(resources.photometry.filter_names)
    simlib_field_names = [f.field_name for f in resources.fields]
    byfield_masks = [
        field_name_mask(split_fieldlist(fields), simlib_field_names,
                        'CUTWIN_PEAKMAG_BYFIELD')
        for _, _, fields in cuts.peakmag_byfield]
    return prepare_cutwin(
        cuts, filter_names, byfield_masks)


def _nepoch_cadence_pass(cuts, fi, n_obs_per_sn, prefilter_pass):
    """NEPOCH cadence check (obs presence + valid filter only, no
    flux/wavelength): SNANA's early NEPOCH < CUTWIN_NEPOCH[0] rejection,
    before the trigger and CUTWIN (snlc_sim.c:296-299).
    """
    n_sne, max_obs = fi.shape
    obs_presence = jnp.arange(max_obs)[None, :] < n_obs_per_sn[:, None]
    cadence_mask = obs_presence & (fi >= 0) & prefilter_pass[:, None]
    nepoch_min = cuts.nepoch_min
    n_cadence = jnp.sum(cadence_mask, axis=1)
    return n_cadence >= nepoch_min if nepoch_min > 0 else jnp.ones(n_sne, dtype=bool)


def _saturation_masks(simulator, zp_fluxcal, results, params,
                      valid_mask, fi, snrs, fluxcals_true_all, nexposes_all):
    """Saturation: per-obs peak-pixel photoelectron count vs.
    NPE_PIXEL_SATURATE (snlc_sim.c:25091-25159). Always computed
    (matching SNANA's unconditional npe_above_saturation call);
    with the default 1e9 threshold no obs ever saturates.
    Computed BEFORE the CUTWIN stage so saturated obs can be
    masked out of the observed-flux arrays (matching SNANA's
    fluxObs=0, fluxcalerr=1e8 sentinels at snlc_sim.c:28544-28548).

    Returns ``(saturated_obs, fluxcals_noisy_sat, fluxcalerrs_sat, snrs_sat)``.
    """
    n_sne = fi.shape[0]
    npe_sat_threshold = simulator.npe_pixel_saturate
    zpts_arr = jnp.asarray(results['observation_schedule']['zptavgs'],
                           dtype=jnp.float32)
    gains_arr = jnp.asarray(results['observation_schedule']['gains'],
                            dtype=jnp.float32)
    skysigs_arr = jnp.asarray(results['observation_schedule']['skysigs'],
                              dtype=jnp.float32)
    psf1_arr = jnp.asarray(results['observation_schedule']['psf1s'],
                           dtype=jnp.float32)
    pixsizes_table = jnp.asarray(
        simulator.flat_field_index['pixsizes'], dtype=jnp.float32)
    pixsize_per_sn = pixsizes_table[jnp.asarray(
        params['field_idx'], dtype=jnp.int32)]
    pixsize_2d = pixsize_per_sn[:, None] * jnp.ones_like(psf1_arr)
    # npe_above_saturation (snlc_sim.c:25140-25164)
    npe_over_fluxcal = jnp.power(10.0, 0.4 * (zpts_arr - zp_fluxcal)) * gains_arr
    # SNANA passes fluxTrue+fluxgal into the areaFrac product (snlc_sim.c:28548)
    fluxgal_pe = jnp.asarray(results['fluxgal_pe'], dtype=jnp.float32)
    flux_pe_all = fluxcals_true_all * npe_over_fluxcal + fluxgal_pe
    skysig_pe = skysigs_arr * gains_arr
    sky_pe = skysig_pe * skysig_pe
    area_pix = pixsize_2d * pixsize_2d
    sq_psfsig = psf1_arr * psf1_arr * area_pix
    sq_psfsig_safe = jnp.maximum(sq_psfsig, 1e-12)
    area_cor = area_pix / (4.0 * jnp.pi * sq_psfsig_safe)
    area_frac = (area_pix / (2.0 * jnp.pi * sq_psfsig_safe)) * (1.0 - area_cor)
    nexposes_f = jnp.maximum(nexposes_all.astype(jnp.float32), 1.0)
    fluxtot_pe = flux_pe_all * area_frac + sky_pe
    # host surface-brightness term (snlc_sim.c:25156-25161)
    sb_fluxcal_for_sat = results.get('host_sb_fluxcal_stacked')
    if sb_fluxcal_for_sat is not None and simulator.hostlib is not None:
        sb_2d = jnp.asarray(sb_fluxcal_for_sat, dtype=jnp.float32)
        sb_per_obs = sb_2d[jnp.arange(n_sne)[:, None], jnp.maximum(fi, 0)]
        sb_per_obs = jnp.where(fi >= 0, sb_per_obs, 0.0)
        sbr = simulator.host_settings.sbradius
        fluxtot_pe = fluxtot_pe + (
            area_pix / (jnp.pi * sbr * sbr)) * sb_per_obs * gains_arr
    npe_above_sat = fluxtot_pe / nexposes_f - npe_sat_threshold
    saturated_obs = (npe_above_sat >= 1.0) & valid_mask
    # the bare EXPOSURE_TIME, not the per-band one (snlc_sim.c:25137)
    if simulator.survey.exposure_time_global > 10.0:
        saturated_obs = jnp.zeros_like(saturated_obs)

    # saturated epochs are written as zero flux (snlc_sim.c:28544-28548)
    FLUXCALERR_SATURATE = 1.0e8  # sntools.h:122
    fluxcals_noisy_sat = jnp.where(
        saturated_obs, 0.0, jnp.asarray(results['fluxcals_noisy']))
    fluxcalerrs_sat = jnp.where(
        saturated_obs, FLUXCALERR_SATURATE, jnp.asarray(results['fluxcalerrs']))
    snrs_sat = jnp.where(saturated_obs, 0.0, snrs)
    return saturated_obs, fluxcals_noisy_sat, fluxcalerrs_sat, snrs_sat


def _evaluate_search_efficiency(simulator, zp_fluxcal,
                                search_efficiency,
                                photprob_prepared, season_boundaries,
                                results, params, sn_idx, valid_mask,
                                saturated_obs, peakmags_masked, snrs_searcheff,
                                fluxcals_true_all, nexposes_all, fi, mjds,
                                redshifts_helio, field_idx_arr, catalogue,
                                *, key):
    """SEARCHEFF: pipeline detect / PHOTPROB / trigger, SPECID and zHOST,
    the GENLC.NEPOCH count with the SEARCHEFF_MINOBS gate, and the
    SIM_SEARCHEFF_MASK / OVP selection gate.

    Returns a dict of per-SN and per-obs SEARCHEFF quantities plus the
    config-derived `apply_pipeline` and `eval_spec` flags.
    """
    n_sne = valid_mask.shape[0]
    searcheff_cfg = simulator.survey
    apply_pipeline = searcheff_cfg._searcheff_apply_pipeline or searcheff_cfg._searcheff_apply
    photprob_2d = None

    # detect, then PHOTPROB, then trigger (sntools_trigger.c:2232-2252)
    if search_efficiency is not None:
        base_key = stream_key(key, 'SEARCHEFF_PIPELINE')
        rng_keys = jax.vmap(
            lambda i: jax.random.fold_in(base_key, i))(sn_idx)
        # saturated epochs have EFF 0 (sntools_trigger.c:2496)
        searcheff_valid = valid_mask & ~saturated_obs
        detection_mask = detect_all(
            search_efficiency, snrs_searcheff, fluxcals_true_all, nexposes_all, fi,
            searcheff_valid, field_idx_arr, rng_keys)
    else:
        # no maps: no epoch detected, the SN found (sntools_trigger.c:2041)
        detection_mask = jnp.zeros_like(valid_mask, dtype=bool)

    detection_for_trigger = detection_mask
    # PHOTPROB is drawn only inside the detection pass, so with no curves
    # every PHOTPROB stays -9 (sntools_trigger.c:2041, :2211)
    if search_efficiency is not None and photprob_prepared:
        n_filters = len(simulator.photometry.transmissions)
        sb_fluxcal_stacked = results.get('host_sb_fluxcal_stacked')
        if sb_fluxcal_stacked is not None:
            safe = jnp.maximum(
                jnp.asarray(sb_fluxcal_stacked, dtype=jnp.float32),
                LOG_SAFE_FLOOR,
            )
            sbmag_2d = -2.5 * jnp.log10(safe) + zp_fluxcal
        else:
            sbmag_2d = jnp.full((n_sne, n_filters), MAG_UNDEFINED, dtype=jnp.float32)
        galmag_2d = jnp.full((n_sne, n_filters), MAG_UNDEFINED, dtype=jnp.float32)
        filter_names_list = list(simulator.photometry.filter_names)
        for f_i, fn in enumerate(filter_names_list):
            gal_key = f'host_{fn}_obs'
            if gal_key in params:
                galmag_2d = galmag_2d.at[:, f_i].set(
                    jnp.asarray(params[gal_key], dtype=jnp.float32))
        # a map with no fieldlist_mask matches every field
        field_match_per_map = []
        for pm in photprob_prepared:
            fl_mask = pm.fieldlist_mask
            if fl_mask is not None:
                field_match_per_map.append(fl_mask[field_idx_arr])
            else:
                field_match_per_map.append(jnp.ones(n_sne, dtype=bool))
        photprob_key = stream_key(key, 'SEARCHEFF_PHOTPROB')
        photprob_2d, photprob_pass = evaluate_photprob(
            photprob_prepared,
            snrs_searcheff, fi, detection_mask, valid_mask,
            redshifts_helio, sbmag_2d, galmag_2d,
            field_match_per_map, photprob_key, sn_idx=sn_idx)
        detection_for_trigger = detection_mask & photprob_pass

    # the written detection flag stays the raw one (snlc_sim.c:25624-25626)
    if search_efficiency is not None:
        (triggered, _, mjd_detect_first, mjd_detect_last, mjd_trigger,
         trigger_mask) = trigger_all(
            search_efficiency, detection_for_trigger, fi, searcheff_valid, mjds,
            n_filters=len(simulator.photometry.transmissions))
        searcheff_pass = (triggered if apply_pipeline
                          else jnp.ones(n_sne, dtype=bool))
    else:
        triggered = jnp.ones(n_sne, dtype=bool)
        searcheff_pass = jnp.ones(n_sne, dtype=bool)
        mjd_detect_first = jnp.full(n_sne, 1.0e6, dtype=jnp.float32)
        mjd_detect_last = jnp.full(n_sne, 1.0e6, dtype=jnp.float32)
        mjd_trigger = jnp.full(n_sne, 1.0e6, dtype=jnp.float32)
        trigger_mask = jnp.zeros_like(valid_mask, dtype=bool)

    specid_pass = jnp.ones(n_sne, dtype=bool)
    specid_eff = jnp.ones(n_sne, dtype=jnp.float32)
    specid_eff_unscaled = jnp.zeros(n_sne, dtype=jnp.float32)
    zhost_eff = jnp.zeros(n_sne, dtype=jnp.float32)
    zhost_pass = jnp.zeros(n_sne, dtype=bool)
    specid_n_matches = jnp.zeros(n_sne, dtype=jnp.int32)
    zhost_n_matches = jnp.zeros(n_sne, dtype=jnp.int32)
    eval_spec = simulator.evaluate_redshift_efficiency

    if eval_spec:
        sn_vars = _build_searcheff_vars(
            simulator, catalogue, season_boundaries, params,
            results['observation_schedule'], peakmags_masked,
            sb_fluxcal_stacked=results.get('host_sb_fluxcal_stacked'))

        spec_key = stream_key(key, 'SEARCHEFF_SPECID')
        # the file-level flag holds even without maps (SEARCHEFF_SPEC_FILE: ZERO)
        if simulator.searcheff_specid_flag == FLAG_ALL_FAIL:
            specid_pass = jnp.zeros(n_sne, dtype=bool)
            specid_eff = jnp.zeros(n_sne, dtype=jnp.float32)
        # NONE, ONE, or a file with no map: EFF=1 (gen_searcheff_map, sntools_trigger.c:2898-2908)
        elif not simulator.searcheff_specid_maps:
            specid_pass = jnp.ones(n_sne, dtype=bool)
        else:
            prepared_spec, specid_n_matches = prepare_searcheff(
                simulator.searcheff_specid_maps, sn_vars,
                var_shifts=searcheff_cfg.searcheff_spec_shifts,
                magshift=searcheff_cfg.searcheff_spec_magshift,
                zp_fluxcal=zp_fluxcal)
            spec_scale = simulator.searcheff_spec_scale
            specid_eff, specid_pass, specid_eff_unscaled = evaluate_searcheff(
                prepared_spec, simulator.searcheff_specid_flag, n_sne,
                spec_key, spec_scale=spec_scale, sn_idx=sn_idx)

        # zHOST, for SPECID failures (read_searcheff_map, sntools_trigger.c:1297-1316)
        if simulator.searcheff_zhost_flag == FLAG_ALL_FAIL:
            zhost_pass_raw = jnp.zeros(n_sne, dtype=bool)
        elif not simulator.searcheff_zhost_maps:
            zhost_pass_raw = jnp.ones(n_sne, dtype=bool)
        else:
            prepared_zhost, zhost_n_matches = prepare_searcheff(
                simulator.searcheff_zhost_maps, sn_vars,
                var_shifts=searcheff_cfg.searcheff_zhost_shifts,
                magshift=searcheff_cfg.searcheff_zhost_magshift,
                zp_fluxcal=zp_fluxcal)
            zhost_key = stream_key(key, 'SEARCHEFF_ZHOST')
            # REFAC_SEARCHEFF_MAP 1: REQUIRE maps read the SPECID randoms (sntools_trigger.c:2886-2891)
            shared = (jax.random.split(spec_key)[1]
                      if searcheff_cfg.searcheff_map_refac == 1 else None)
            _, zhost_pass_raw, zhost_eff = evaluate_searcheff(
                prepared_zhost, simulator.searcheff_zhost_flag, n_sne,
                zhost_key, sn_idx=sn_idx, require_key=shared)

        # no host candidate, no host redshift, whatever the maps (sntools_trigger.c:3253)
        if 'host_NMATCH' in params and simulator.hostlib is not None:
            has_neighbor = jnp.asarray(params['host_NMATCH']) > 0
            zhost_pass_raw = zhost_pass_raw & has_neighbor

        zhost_pass = zhost_pass_raw & ~specid_pass

    # GENLC.NEPOCH counts the peak and template epochs too (snlc_sim.c:12979-12989)
    sched = results['observation_schedule']
    band_mask = _simlib_band_mask(simulator, simulator.cuts, sched)
    nepoch = jnp.asarray(sched['n_obs_per_sn']) + band_mask.sum(axis=1)
    tmpl = per_filter_array(simulator, simulator.survey.mjd_template, 0.0)
    if tmpl is not None:
        nepoch = nepoch + (band_mask & (tmpl > 1.0)[None, :]).sum(axis=1)

    # too few epochs fails every stage (sntools_trigger.c:2033)
    if searcheff_cfg.searcheff_minobs > 1:
        enough_epochs = nepoch >= searcheff_cfg.searcheff_minobs
        triggered = triggered & enough_epochs
        specid_pass = specid_pass & enough_epochs
        zhost_pass = zhost_pass & enough_epochs

    # SPECID and zHOST need the pipeline trigger (sntools_trigger.c:2050-2061)
    specid_pass = specid_pass & triggered
    zhost_pass = zhost_pass & triggered
    specid_eff = jnp.where(triggered, specid_eff, 0.0)

    # SIM_SEARCHEFF_MASK bits (sntools_trigger.c:2064-2076)
    pipeline_passed = triggered
    searcheff_mask = jnp.zeros(n_sne, dtype=jnp.int32)
    searcheff_mask = jnp.where(pipeline_passed, searcheff_mask | 1, searcheff_mask)
    searcheff_mask = jnp.where(specid_pass & pipeline_passed,
                               searcheff_mask | 2, searcheff_mask)
    searcheff_mask = jnp.where(zhost_pass & pipeline_passed,
                               searcheff_mask | 4, searcheff_mask)
    searcheff_mask = jnp.where((specid_pass | zhost_pass) & pipeline_passed,
                               searcheff_mask | 8, searcheff_mask)
    apply_opt = searcheff_cfg.apply_searcheff_opt
    if apply_opt > 0 and apply_pipeline:
        ovp_pass = (searcheff_mask & apply_opt) == apply_opt
        searcheff_pass = searcheff_pass & ovp_pass

    return {
        'apply_pipeline': apply_pipeline,
        'eval_spec': eval_spec,
        'triggered': triggered,
        'searcheff_pass': searcheff_pass,
        'detection_mask': detection_mask,
        'photprob_2d': photprob_2d,
        'mjd_detect_first': mjd_detect_first,
        'mjd_detect_last': mjd_detect_last,
        'mjd_trigger': mjd_trigger,
        'trigger_mask': trigger_mask,
        'specid_pass': specid_pass,
        'sim_searcheff_spec': specid_eff,
        'zhost_pass': zhost_pass,
        'specid_n_matches': specid_n_matches,
        'zhost_n_matches': zhost_n_matches,
        'specid_eff_unscaled': specid_eff_unscaled,
        'zhost_eff': zhost_eff,
        'sim_searcheff_mask': searcheff_mask,
        'nepoch': nepoch,
        'simlib_band_mask': band_mask,
    }


def _observed_redshift(simulation, params, redshifts_helio,
                       specid_pass, zhost_pass, eval_spec):
    """setz_unconfirmed: set the OBSERVED heliocentric redshift before CUTWIN.
    Truth params['sim_redshift_cmb'] (zCMB) and params['sim_redshift_helio'] (zHEL) are
    never modified; the modified value is the observed zHEL.

    Returns ``(redshift_helio_observed, redshift_cmb_observed, redshift_flag,
    redshift_err)``.
    """
    n_sne = redshifts_helio.shape[0]
    redshift_helio_observed = redshifts_helio.copy()
    gensigma_z = simulation.rates.gensigma_redshift
    genbias_z = simulation.rates.genbias_redshift
    force_host_photoz = gensigma_z < 0.0 or gensigma_z > 0.999
    if not force_host_photoz and (gensigma_z > 0.0 or genbias_z != 0.0):
        # a replay has no smear draw, only the bias
        zsmear = genbias_z
        if '_zsmear_normal' in params:
            zsmear = zsmear + jnp.asarray(params['_zsmear_normal']) * gensigma_z
        redshift_helio_observed = redshift_helio_observed + zsmear
    redshift_flag = jnp.ones(n_sne, dtype=jnp.int32)
    # SNANA reports GENSIGMA_REDSHIFT as the smear error (snlc_sim.c:17409)
    redshift_err = jnp.full(n_sne, gensigma_z, dtype=jnp.float32)
    phot_only = ~specid_pass
    use_photoz = jnp.zeros(n_sne, dtype=bool)
    wrong_zhost = jnp.zeros(n_sne, dtype=bool)
    if force_host_photoz or eval_spec:
        # SNHOSTGAL.ZPHOT, not the reported DDLR-sorted one (snlc_sim.c:24922)
        host_photoz = jnp.asarray(
            params.get('host_ZPHOT_EVENT', jnp.full(n_sne, SNANA_NULLFLOAT)),
            dtype=jnp.float32)
        host_photoz_err = jnp.asarray(
            params.get('host_ZPHOT_EVENT_ERR', jnp.full(n_sne, SNANA_NULLFLOAT)),
            dtype=jnp.float32)
    if force_host_photoz:
        redshift_helio_observed = host_photoz
        redshift_err = host_photoz_err
    if eval_spec:
        correct_match = jnp.asarray(params['CORRECT_HOSTMATCH'], dtype=bool)
        host_ztrue = jnp.asarray(
            params.get('host_ZTRUE', redshifts_helio), dtype=jnp.float32)

        correct_zhost = phot_only & zhost_pass & correct_match
        redshift_flag = jnp.where(correct_zhost, 2, redshift_flag)
        wrong_zhost = phot_only & zhost_pass & ~correct_match
        redshift_flag = jnp.where(wrong_zhost, 4, redshift_flag)
        redshift_helio_observed = jnp.where(
            wrong_zhost, host_ztrue, redshift_helio_observed)
        no_zhost = phot_only & ~zhost_pass
        has_photoz = host_photoz >= 0.0
        use_photoz = no_zhost & has_photoz
        use_no_z = no_zhost & ~has_photoz
        redshift_helio_observed = jnp.where(
            use_photoz, host_photoz, redshift_helio_observed)
        redshift_helio_observed = jnp.where(
            use_no_z, SNANA_NULLFLOAT, redshift_helio_observed)
        redshift_flag = jnp.where(use_photoz & correct_match, 3, redshift_flag)
        redshift_flag = jnp.where(use_photoz & ~correct_match, 4, redshift_flag)
        redshift_flag = jnp.where(use_no_z, 0, redshift_flag)
        redshift_err = jnp.where(use_photoz, host_photoz_err, redshift_err)
        redshift_err = jnp.where(use_no_z, SNANA_NULLFLOAT, redshift_err)

    # snlc_sim.c:24948, :24986
    ras_obs = jnp.asarray(params['ra'], dtype=jnp.float32)
    decs_obs = jnp.asarray(params['dec'], dtype=jnp.float32)
    # a host photo-z is converted even with the dipole off
    vel_cmbapex = simulation.cosmology.vel_cmbapex
    convert = jnp.where(wrong_zhost, vel_cmbapex > 1.0,
                        vel_cmbapex >= 1.0) | use_photoz
    if gensigma_z < 0.0:
        # GEN_SNHOST_ZPHOT assigns the same photo-z in both frames.
        # Later selection can still replace it with an unconfirmed source.
        convert &= use_photoz | wrong_zhost
    redshift_cmb_observed = jnp.where(
        convert,
        zhelio_to_zcmb(redshift_helio_observed, ras_obs, decs_obs),
        redshift_helio_observed,
    )
    redshift_cmb_observed = jnp.where(
        redshift_helio_observed >= 0, redshift_cmb_observed, SNANA_NULLFLOAT)
    return (redshift_helio_observed, redshift_cmb_observed, redshift_flag,
            redshift_err)


def _evaluate_cuts(simulator, cutwin, params,
                   valid_mask, fi, n_obs_per_sn, phases, mjds, snrs_sat,
                   saturated_obs, peakmags_masked, field_idx_arr,
                   redshifts_helio, redshift_cmb_observed):
    """CUTWIN: per-observation cuts on ALL valid obs (not just detected)
    plus the per-SN range cuts. The kernel counts over `epoch_base_mask`,
    SNANA's filtered epoch set (snlc_sim.c:24203-24235).

    Returns ``(pass_cutwin_all, pass_range, cutwin_fail_masks, params_for_range,
    epoch_stats)``.
    """
    n_sne = valid_mask.shape[0]
    rest_wavelengths = compute_rest_wavelengths(simulator, fi, redshifts_helio)
    epoch_base_mask = _build_epoch_mask(
        simulator.cuts, valid_mask, n_obs_per_sn, phases, snrs_sat,
        rest_wavelengths, include_snr_filter=True)

    peakmags_cut = peakmags_masked
    if simulator.mw_dust.applyflag_mwebv:
        # SNANA cuts on the MW-corrected mag (snlc_sim.c:24575, :13727)
        rv = simulator.mw_dust.rv
        mwebv_true = jnp.asarray(params['sim_mwebv'], dtype=jnp.float32)
        axav = jnp.asarray(simulator.mw_law_at_mean.extinction_curve(
            simulator.photometry.filter_mean_wavelengths, rv,
            av=rv * mwebv_true[:, None]), dtype=jnp.float32)
        peakmags_cut = peakmags_cut - rv * mwebv_true[:, None] * axav
    pass_cutwin, cutwin_fail_masks, epoch_stats = cutwin_all(cutwin, simulator.cutwin_options,
        snrs_sat, phases, fi, valid_mask, mjds, epoch_base_mask,
        peakmags_cut, field_idx_arr, saturated_obs)

    # ZTRUE, ZFINAL, ZPHOT and the generated MWEBV (snlc_sim.c:24164-24176, :24454)
    params_for_range = range_cut_quantities(
        {k: jnp.asarray(v, dtype=jnp.float32) for k, v in params.items()
         if hasattr(v, 'shape') and v.ndim == 1 and v.shape[0] == n_sne},
        mwebv=jnp.asarray(params['mwebv'], dtype=jnp.float32),
        redshift_true=jnp.asarray(params['sim_redshift_cmb'],
                                  dtype=jnp.float32),
        redshift_final=redshift_cmb_observed,
        host_zphot=jnp.asarray(
            params.get('host_ZPHOT_EVENT', jnp.full(n_sne, SNANA_NULLFLOAT)),
            dtype=jnp.float32))
    pass_range = apply_generic_range_cuts(
        simulator.cuts, params_for_range)

    pass_cutwin_all = pass_cutwin & pass_range
    return (pass_cutwin_all, pass_range, cutwin_fail_masks, params_for_range,
            epoch_stats)


def _assemble_pass_and_reasons(cuts, nepoch_pass, triggered, searcheff_pass,
                               apply_pipeline, pass_cutwin_all,
                               cutwin_fail_masks, params_for_range,
                               prefilter_pass, prefilter_reason):
    """Final pass mask, per-SN rejection reason codes, and the CUTMASK
    bit assembly.

    Returns ``(passing, rejection_codes, cutmask_arr)``.
    """
    n_sne = triggered.shape[0]
    # APPLY_CUTWIN_OPT=0 records CUTWIN in CUTMASK only (snlc_sim.c:24542)
    if cuts.apply_cutwin_opt == 0:
        cutwin_gate = jnp.ones(n_sne, dtype=bool)
    else:
        cutwin_gate = pass_cutwin_all
    passing = searcheff_pass & cutwin_gate & prefilter_pass

    # 0 PASS, 1 GENRANGE, 2 NEPOCH, 3 PIPELINE, 4 SPECID/zHOST, 5 CUTWIN; earliest stage wins
    rejection_codes = jnp.zeros(n_sne, dtype=jnp.int32)
    rejection_codes = jnp.where(~pass_cutwin_all, 5, rejection_codes)
    rejection_codes = jnp.where(triggered & ~searcheff_pass, 4, rejection_codes)
    if apply_pipeline:
        rejection_codes = jnp.where(~triggered, 3, rejection_codes)
    rejection_codes = jnp.where(~nepoch_pass, 2, rejection_codes)
    # empty LIBID is NEPOCH (snlc_sim.c:296), no host is GENRANGE (:23858)
    rejected_before = ~prefilter_pass
    rejection_codes = jnp.where(
        rejected_before & (prefilter_reason == PREFILTER_REASON_CODES['NEPOCH']),
        2, rejection_codes)
    rejection_codes = jnp.where(
        rejected_before & (prefilter_reason == PREFILTER_REASON_CODES['HOSTLIB_NOHOST']),
        1, rejection_codes)

    # GENLC.CUTBIT_MASK: every bit set, then cleared on failure (gen_cutwin)
    cutmask_arr = jnp.full(n_sne, CUTMASK_ALL_PASS, dtype=jnp.int32)
    for check_name, fail_mask in cutwin_fail_masks.items():
        if check_name in CUTBIT_MAP:
            bit_pos = CUTBIT_MAP[check_name]
            cutmask_arr = jnp.where(fail_mask, cutmask_arr & ~(1 << bit_pos), cutmask_arr)
    for name, value in per_sn_windows(cuts).items():
        param_name = name.upper()
        if param_name not in CUTBIT_MAP:
            continue
        arr = params_for_range.get(name, params_for_range.get(param_name))
        if arr is not None:
            fail = (arr < value[0]) | (arr > value[1])
            cutmask_arr = jnp.where(
                fail, cutmask_arr & ~(1 << CUTBIT_MAP[param_name]), cutmask_arr)
    cutmask_arr = jnp.where(~prefilter_pass, 0, cutmask_arr)
    cutmask_arr = jnp.where(prefilter_pass & ~nepoch_pass, 0, cutmask_arr)
    cutmask_arr = jnp.where(prefilter_pass & nepoch_pass & ~searcheff_pass, 0, cutmask_arr)

    return passing, rejection_codes, cutmask_arr


def _first_extreme_epoch(values):
    """Per event, marks the first epoch holding the row's maximum of
    ``values``; a row whose maximum is not finite marks nothing."""
    best = jnp.argmax(values, axis=1)
    finite = jnp.isfinite(jnp.take_along_axis(values, best[:, None], axis=1))
    return (jnp.arange(values.shape[1])[None, :] == best[:, None]) & finite


def evaluate_selection(simulator, zp_fluxcal, search_efficiency, cutwin,
                        season_boundaries, photprob_prepared, catalogue,
                        results: Dict, *, key) -> Dict:
    """Evaluate selection stages in SNANA's order using JAX arrays.

    All operations are JAX primitives, safe to call from inside
    `@jax.jit` (as the compiled batch does). DOFILT peakmag masking
    and deferred HOSTLIB column gather happen inline using the JAX
    helpers, so callers don't need to pre-populate anything.

    The Python wrapper `apply_selection_effects` calls this inner
    function, then builds the generated record and runs the post-JIT
    finalisation: selection-summary prints and the SEARCHEFF field-coverage
    and EFF-window checks. MJD_DETECT and the trigger mask are computed
    here, inside the trace.
    """
    deferred = gather_deferred_hostlib_columns(
        simulator.host_settings, simulator.hostlib, catalogue, results['parameters'])
    params = {**results['parameters'], **deferred}
    sn_idx = jnp.asarray(results['parameters']['sn_global_idx'], dtype=jnp.int32)

    peakmags_masked = _apply_dofilt_peakmag_mask(
        simulator, params, results['peakmags'], results['observation_schedule'])

    # Per-obs validity, SNANA's OBSFLAG_GEN (snlc_sim.c:24204)
    valid_mask = _build_validity_mask(results)

    snrs_model = jnp.array(results['snrs_model'], dtype=jnp.float32)
    snrs = jnp.array(results['snrs'], dtype=jnp.float32)
    mjds = jnp.array(results['observation_schedule']['mjds'], dtype=jnp.float32)
    phases = jnp.array(results['observation_schedule']['phases'], dtype=jnp.float32)
    fi = jnp.array(results['observation_schedule']['filter_indices'])
    n_obs_per_sn = jnp.array(results['observation_schedule']['n_obs_per_sn'])
    redshifts_helio = jnp.asarray(params['sim_redshift_helio'], dtype=jnp.float32)
    field_idx_arr = jnp.asarray(params['field_idx'], dtype=jnp.int32)
    prefilter_pass = jnp.asarray(params['prefilter_pass'])
    prefilter_reason = jnp.asarray(params['prefilter_reason_code'])
    fluxcals_true_all = jnp.nan_to_num(
        jnp.asarray(results['fluxcals'], dtype=jnp.float32),
        nan=0.0, posinf=0.0, neginf=0.0)
    nexposes_all = jnp.asarray(
        results['observation_schedule']['nexposes'], dtype=jnp.float32)

    # NEPOCH on the cadence (snlc_sim.c:296), then on NOBS_MODELFLUX (:342)
    reached_flux = prefilter_pass & _nepoch_cadence_pass(
        simulator.cuts, fi, n_obs_per_sn, prefilter_pass)
    nepoch_pass = reached_flux & jnp.any(valid_mask, axis=1)

    saturated_obs, fluxcals_noisy_sat, fluxcalerrs_sat, snrs_sat = \
        _saturation_masks(simulator, zp_fluxcal, results, params,
                          valid_mask, fi, snrs, fluxcals_true_all,
                          nexposes_all)

    # SNR_CALC is the measured SNR under SURVEY: SDSS (snlc_sim.c:24713-24716)
    snrs_searcheff = (snrs if simulator.survey_name.upper() == 'SDSS'
                      else snrs_model)
    search = _evaluate_search_efficiency(
        simulator, zp_fluxcal, search_efficiency,
        photprob_prepared,
        season_boundaries, results, params, sn_idx, valid_mask,
        saturated_obs, peakmags_masked, snrs_searcheff, fluxcals_true_all,
        nexposes_all, fi, mjds, redshifts_helio, field_idx_arr, catalogue,
        key=key)

    (redshift_helio_observed, redshift_cmb_observed, redshift_flag,
     redshift_err) = _observed_redshift(
        simulator, params, redshifts_helio,
        search['specid_pass'], search['zhost_pass'], search['eval_spec'])

    (pass_cutwin_all, pass_range, cutwin_fail_masks, params_for_range,
     epoch_stats) = _evaluate_cuts(
        simulator, cutwin, params,
        valid_mask, fi, n_obs_per_sn, phases, mjds, snrs_sat,
        saturated_obs, peakmags_masked, field_idx_arr,
        redshifts_helio, redshift_cmb_observed)

    passing, rejection_codes, cutmask_arr = _assemble_pass_and_reasons(
        simulator.cuts, nepoch_pass, search['triggered'],
        search['searcheff_pass'], search['apply_pipeline'], pass_cutwin_all,
        cutwin_fail_masks, params_for_range, prefilter_pass, prefilter_reason)

    # an event that never reaches gen_cutwin keeps the initial gap and SNR (snlc_sim.c:9561-9563, :9655)
    reached = passing | (rejection_codes == 5)
    epoch_stats = {
        'trestmin': jnp.where(
            reached_flux & jnp.isfinite(epoch_stats['trestmin']),
            epoch_stats['trestmin'], 99999.0),
        'trestmax': jnp.where(
            reached_flux & jnp.isfinite(epoch_stats['trestmax']),
            epoch_stats['trestmax'], -99999.0),
        'tgapmax': jnp.where(reached, epoch_stats['tgapmax'], GAP_UNSET),
        'snrmax_band': jnp.where(reached[:, None], epoch_stats['snrmax_band'],
                                 -9.0),
        'nobs': jnp.where(reached_flux, epoch_stats['nobs'], 0),
    }

    # NEARPEAK looks over every scheduled epoch, written or not (keep_SIMLIB_OBS)
    epoch_flags = {}
    if simulator.instrument.photflag_snrmax > 0:
        epoch_flags['snrmax_epoch'] = _first_extreme_epoch(
            jnp.where(valid_mask, jnp.asarray(results['snrs_model']), -jnp.inf))
    if simulator.instrument.photflag_nearpeak > 0:
        peak = jnp.asarray(params['peakmjd_true'])
        epoch_flags['nearpeak_epoch'] = _first_extreme_epoch(
            jnp.where(fi >= 0, -jnp.abs(mjds - peak[:, None]), -jnp.inf))

    return {
        'passing_mask': passing,
        'triggered': search['triggered'],
        'detection_mask': search['detection_mask'],
        'pass_cutwin': pass_cutwin_all,
        'pass_range': pass_range,
        'valid_mask': valid_mask,
        'peakmags_masked': peakmags_masked,
        'rejection_codes': rejection_codes,
        'reached_flux': reached_flux,
        'cutmask': cutmask_arr,
        'redshift_flag': redshift_flag,
        'redshift_helio': redshift_helio_observed,
        'redshift_final': redshift_cmb_observed,
        'redshift_smear_err': redshift_err,
        'sim_searcheff_mask': search['sim_searcheff_mask'],
        'sim_searcheff_spec': search['sim_searcheff_spec'],
        'mwxt_mag': mwxt_mag_per_band(simulator, simulator, params),
        # what HOSTGAL_MAG_<band> is dimmed by (sntools_host.c:9287)
        'host_mwxt_mag': host_mwxt_mag_per_band(simulator, params),
        'simlib_band_mask': search['simlib_band_mask'],
        'specid_pass': search['specid_pass'],
        'zhost_pass': search['zhost_pass'],
        'cutwin_fail_masks': cutwin_fail_masks,
        'photprob_2d': search['photprob_2d'],
        'mjd_detect_first': search['mjd_detect_first'],
        'mjd_detect_last': search['mjd_detect_last'],
        'mjd_trigger': search['mjd_trigger'],
        'trigger_mask': search['trigger_mask'],
        'nepoch': search['nepoch'],
        'saturated_mask': saturated_obs,
        'fluxcals_noisy_sat': fluxcals_noisy_sat,
        'fluxcalerrs_sat': fluxcalerrs_sat,
        'snrs_sat': snrs_sat,
        'specid_n_matches': search['specid_n_matches'],
        'zhost_n_matches': search['zhost_n_matches'],
        'specid_eff_unscaled': search['specid_eff_unscaled'],
        'zhost_eff': search['zhost_eff'],
        **epoch_stats,
        **epoch_flags,
    }


def apply_selection_effects(simulator, zp_fluxcal, search_efficiency, cutwin,
                        season_boundaries, photprob_prepared, catalogue,
                        results: Dict, *, key) -> Dict:
    """Evaluate selection, validate map coverage, and report the results.

    Returns the same shape dict as `evaluate_selection`
    plus the Python-side finalisation (the generated record, pass and
    fail counts, selection summary, per-field coverage checks), without
    the per-check CUTWIN fail masks. The batched run calls the inner
    within each compiled batch and finalises the merged result via
    :func:`finalize_selection`.
    """
    selection = evaluate_selection(
        simulator, zp_fluxcal, search_efficiency, cutwin, season_boundaries,
        photprob_prepared, catalogue, results, key=key)
    selection['generated'] = generated_record(results, selection)
    selection.pop('cutwin_fail_masks')
    return finalize_selection(simulator, results, selection)


def finalize_selection(simulator, results, selection: Dict) -> Dict:
    """Host-side selection finalisation, once per run.

    Takes an inner-selection dict (whole-run, or merged from per-batch
    pieces) and adds the pieces that need materialised numpy arrays: the
    counts of generated events that passed and failed, summary logging,
    and the SPECEFF/zHOST field-coverage and EFF-window checks, which
    abort on an invalid event.
    """
    generated = selection['generated']
    rc_np = generated['rejection_codes']
    cutwin_fail_masks = generated['cutwin_fail_masks']
    nepoch_min = simulator.cuts.nepoch_min

    n_sne = rc_np.shape[0]
    n_passing = int(generated['passing'].sum())
    selection['n_passed'] = n_passing
    selection['n_failed'] = n_sne - n_passing
    n_genrange = int((rc_np == 1).sum())
    n_nep = int((rc_np == 2).sum())
    n_pipeline = int((rc_np == 3).sum())
    n_zhost = int((rc_np == 4).sum())
    n_cutwin_total = int((rc_np == 5).sum())
    log.line(logger, f"Selection summary: pass={n_passing} / fail={n_sne - n_passing}")
    log.line(logger, f"NREJECT detail: [GENRANGE={n_genrange}, NEP<{nepoch_min}={n_nep}, "
             f"SEARCHEFF={n_pipeline + n_zhost}, CUTWIN={n_cutwin_total}]")
    log.line(logger, f"SEARCHEFF breakdown: [pipeline_fail={n_pipeline}, "
             f"zHOST/SPECID_fail={n_zhost}]")
    # only events whose final rejection was CUTWIN
    actual_cutwin_rejects = (rc_np == 5)
    cutwin_breakdown = {
        k: int((np.asarray(v) & actual_cutwin_rejects).sum())
        for k, v in cutwin_fail_masks.items()
    }
    cutwin_breakdown = {k: v for k, v in cutwin_breakdown.items() if v > 0}
    if cutwin_breakdown:
        bd = ", ".join(f"{k}={v}" for k, v in
                       sorted(cutwin_breakdown.items(), key=lambda x: -x[1]))
        log.line(logger, f"CUTWIN breakdown: [{bd}]")

    _check_unmatched_searcheff_fields(
        'SPECEFF', simulator.searcheff_specid_maps,
        generated['specid_unmatched'], generated, simulator)
    _check_unmatched_searcheff_fields(
        'zHOST', simulator.searcheff_zhost_maps,
        generated['zhost_unmatched'], generated, simulator)
    _check_searcheff_eff_window('SPEC', generated['specid_eff'], generated,
                                simulator)
    _check_searcheff_eff_window('zHOST', generated['zhost_eff'], generated,
                                simulator)
    return selection


def _check_unmatched_searcheff_fields(label, maps, no_match, generated,
                                      resources):
    """An evaluated SN whose field and peak MJD match no map of this type
    aborts, unless the map file set OPT_FIELDMATCH_REQUIRE: 0, in which case
    it keeps EFF=0 with one warning per uncovered field (gen_searcheff_map,
    sntools_trigger.c:2983-2991)."""
    if not maps or not no_match.any():
        return
    field_of = np.array([f.field_name for f in resources.fields])
    fields = field_of[generated['field_idx']]
    if maps[0].opt_fieldmatch_require > 0:
        first = int(np.flatnonzero(no_match)[0])
        raise SNANAError(
            f"Invalid NMATCH=0 {label} maps for FIELD = '{fields[first]}' "
            f"PEAKMJD={float(generated['peakmjd_true'][first]):.3f} (set "
            f"OPT_FIELDMATCH_REQUIRE: 0 in the map file to give such events EFF=0 "
            f"instead)", function_name="gen_searcheff_map")
    for field in sorted(set(fields[no_match])):
        logger.warning(
            f"No SEARCHEFF {label} map matches FIELD={field!r}; "
            f"those SNe get EFF=0 (OPT_FIELDMATCH_REQUIRE: 0).")


def _check_searcheff_eff_window(label, eff, generated, resources):
    """An evaluated SN whose combined EFF is outside the round-off window aborts (gen_searcheff_map, sntools_trigger.c:3011-3019)."""
    outside = np.asarray(eff_outside_window(eff))
    if not outside.any():
        return
    first = int(np.flatnonzero(outside)[0])
    field = resources.fields[int(generated['field_idx'][first])].field_name
    raise SNANAError(
        f"Invalid EFF({label}) = {float(eff[first]):.4f} for FIELD = "
        f"'{field}' PEAKMJD={float(generated['peakmjd_true'][first]):.3f}",
        function_name="gen_searcheff_map")


def _simlib_band_mask(resources, cuts, obs_schedule):
    """Filters present in each SN's SIMLIB entry (SIMLIB_USEFILT_ENTRY).

    SNANA inserts the artificial PEAKMJD epoch only for these filters
    (snlc_sim.c:12971), so SIM_PEAKMAG stays at its sentinel for the rest.
    The skip applies only when CUTWIN_NEPOCH is in force.
    """
    n_bands = len(resources.filter_names)
    fidx = jnp.asarray(obs_schedule['filter_indices'])
    if cuts.nepoch_min <= 0:
        return jnp.ones((fidx.shape[0], n_bands), dtype=bool)

    nobs = jnp.asarray(obs_schedule['n_obs_per_sn'])
    within = jnp.arange(fidx.shape[1])[None, :] < nobs[:, None]
    bands = jnp.arange(n_bands)
    return jnp.any(within[..., None] & (fidx[..., None] == bands), axis=1)


def kept_mask(simulation, selection):
    """The events of a batch an output reads after it: the passing ones,
    and under APPLY_CUTWIN_OPT=3 the CUTWIN-rejected ones the DUMP lists
    beside them (doReject_SIMGEN_DUMP, snlc_sim.c:16927)."""
    passing = np.asarray(selection['passing_mask'])
    if simulation.cuts.apply_cutwin_opt == 3:
        return passing | (np.asarray(selection['rejection_codes']) == 5)
    return passing


def generated_record(results, selection):
    """What the run keeps of every generated event once its batch has
    packed to the kept ones: what the summaries and the NGEN_LC stopping
    rule count, a few bytes per event."""
    params = results['parameters']
    # which events each map type is evaluated for (sntools_trigger.c:2052)
    specid_evaluated = np.asarray(selection['triggered'], dtype=bool)
    zhost_evaluated = specid_evaluated & ~np.asarray(selection['specid_pass'],
                                                     dtype=bool)
    # a hostless event never reaches the zHOST maps (sntools_trigger.c:3253)
    if 'host_NMATCH' in params:
        zhost_evaluated &= np.asarray(params['host_NMATCH']) > 0
    record = {
        'rejection_codes': np.asarray(selection['rejection_codes'], dtype=np.int8),
        'passing': np.asarray(selection['passing_mask'], dtype=bool),
        'field_idx': np.asarray(params['field_idx'], dtype=np.int32),
        'peakmjd_true': np.asarray(params['peakmjd_true'], dtype=np.float32),
        'cutwin_fail_masks': {k: np.asarray(v, dtype=bool)
                              for k, v in selection['cutwin_fail_masks'].items()},
        # an evaluated SN whose field matches no SEARCHEFF map of the type
        'specid_unmatched': specid_evaluated & (np.asarray(selection['specid_n_matches']) == 0),
        'zhost_unmatched': zhost_evaluated & (np.asarray(selection['zhost_n_matches']) == 0),
        'specid_eff': np.where(specid_evaluated,
                               np.asarray(selection['specid_eff_unscaled']), 0.0),
        'zhost_eff': np.where(zhost_evaluated,
                              np.asarray(selection['zhost_eff']), 0.0),
        # the events whose types the README's GENTYPE_TO_NAME collects
        'reached_flux': np.asarray(selection['reached_flux'], dtype=bool),
    }
    # the NON1ASED template of each event, for the quotas and the type names
    for name in ('template_slot', 'sntype', 'non1a_index'):
        if name in params:
            record[name] = np.asarray(params[name])
    return record
