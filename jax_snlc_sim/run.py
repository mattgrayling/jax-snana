"""Prepare and run survey simulations.

``prepare`` resolves a SimulationConfig and loads a LightCurveSimulator.
Numerical kernels take that simulator explicitly, sharing compiled programs
across compatible realisations. ``run_simulation`` owns the batch loop and
its host reuse, quota and output records; they never persist on the simulator.
"""

import logging
import copy
import os
import time
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core.photometry.filter_system import band_weights_at_z, filter_band
from jax_snlc_core.photometry.noise import crazy_flux_bound
from jax_snlc_core.utils import HOSTGAL_PREFIXES, SNANA_NULLFLOAT, snana_log as slog
from jax_snlc_core.utils.error_handling import SNANAError
from jax_snlc_core.utils.precision import require_precision
from jax_snlc_core.utils.interp import eval_genpoly
from jax_snlc_sim import atmosphere as atm_mod
from jax_snlc_sim.builder import build_simulation
from jax_snlc_sim.hostlib.wronghost import DZTRUEMAX
from jax_snlc_sim.output import event_counts, write_snana_outputs
from jax_snlc_sim.output.tables import HOSTGAL_COORD_NULL
from jax_snlc_sim.output.dump import noise_dump_rows, print_simgen_dump_variables
from jax_snlc_sim.photometry.magnitudes import (
    LOG_SAFE_FLOOR, SNANA_NULL_MAG, fluxcal_to_magnitude,
)
from jax_snlc_sim.photometry.noise import add_photometric_noise
from jax_snlc_sim.host_association import (
    add_host_associations, apply_host_snpar_overrides, galaxy_labels,
    host_pec1a_flags, random_coord_shift,
    wgtmap_sn_variables,
)
from jax_snlc_sim.random_streams import event_keys, root_key, run_key, stream_key
from jax_snlc_sim.sampler import (
    PREFILTER_REASON_CODES, apply_lensdmu_smear, draw_dust,
    draw_grey_offset, draw_input_checks, draw_input_failures, draw_model,
    draw_placement,
)
from jax_snlc_sim.selection import (
    _apply_fmax_clump_method,
    _apply_selection_effects_jit_inner,
    _build_columnar_data,
    apply_selection_effects,
    SIMLIB_EPOCH_KEYS,
    epoch_layout,
    finalize_selection,
    generated_record,
    keep_events,
    kept_mask,
    padded_rows,
    row_slice,
    simlib_epochs,
)
from jax_snlc_sim.dashboard import run_dashboard
from jax_snlc_sim.simlib_dump import run_simlib_dump
from jax_snlc_sim.simulation import (
    LightCurveSimulator,
    SimulationConfig,
    parse_format_mask,
)
from jax_snlc_sim.genmag_smear import scatter_factors
from jax_snlc_sim.spectrograph.genspec import (
    compute_sed_true_synmags, generate_all_spectra, generate_simlib_spectra,
    generate_simlib_take_spectrum, keep_spectrum,
)
from jax_snlc_sim.survey.cutwin import check_generic_cut_names
from jax_snlc_sim.survey.simlib_reader import (
    FLAT_OBS_KEYS,
    FLOAT_PROPERTY_NAMES,
    schedule_all,
)

logger = logging.getLogger(__name__)


SED_PHASE_EDGE_EPS = 0.01  # genmag_SALT2.c:3564-3565

# padding when merging batches of different max_obs; anything else pads with zero
OBS_PAD_VALUES = {'mjds': -999.0, 'filter_indices': -1,
                  'field_indices': -1}


# the host photometry a batch keeps for its kept events, (n, n_filters)
HOST_TABLE_KEYS = ('host_galmag_total', 'host_galmag_galfrac',
                   'host_sb_fluxcal_stacked')


def keeps_rejected_events(simulation, config) -> bool:
    """Whether the run holds every generated event to the end: a caller
    that selects it again asks for that, and SIMGEN_DUMPALL writes every
    event."""
    return config.keep_rejected_events or simulation.output.simgen_dumpall







def precompute_restframe_filters(simulator, redshifts,
                                 mwebv) -> jnp.ndarray:
    """Observer-frame filters blue-shifted to rest frame, per SN.

    Processes all (n_sne × n_filters) combinations in one nested-vmap call
    over the pre-interpolated common-grid filter data. The band weights
    fold in the blueshift λ_rest = λ_obs / (1+z), the (T·λ·dλ)/∫(T·λ·dλ)
    normalisation, the observer-frame 1/(1+z) scaling, and the per-SN MW
    attenuation evaluated on the common grid.
    """
    common_wave = simulator.photometry.common_wave_grid
    rv_mw = simulator.mw_dust.rv
    axav_obs = simulator.mw_law_obs.extinction_curve(
        common_wave, rv_mw, av=rv_mw * mwebv[:, None])
    mw_att_all = jnp.power(10.0, -0.4 * rv_mw * mwebv[:, None] * axav_obs)

    compute_for_all_filters_and_sne = jax.vmap(
        band_weights_at_z,
        in_axes=(0, None, None, None, 0))

    return compute_for_all_filters_and_sne(
        redshifts,
        simulator.photometry.common_wave_grid,
        simulator.photometry.common_trans_grid,
        simulator.wavelength_grid,
        mw_att_all)


def schedule_observations(simulator, tables, field_indices,
                          peak_mjds, redshifts_helio) -> Dict:
    """JIT-compiled observation scheduling over the SIMLIB field arrays, by
    the simulator's scheduler from :func:`build_scheduler`, reading the
    SIMLIB epochs from ``tables`` (:func:`batch_tables`).

    The epochs are windowed on the drawn peak and redshift, as SNANA reads
    the SIMLIB before the host can change either (set_SIMLIB_MJDrange,
    snlc_sim.c:21300); ``redshifts_helio`` sets the window because it is an
    observer-frame Doppler effect. Phases come later, from the final values
    (:func:`epoch_phases`).
    """
    trest_min, trest_max = simulator.phase_range
    pad = simulator.survey.genrange_tobs_pad
    mjd_mins = peak_mjds + trest_min * (1.0 + redshifts_helio) - pad
    mjd_maxs = peak_mjds + trest_max * (1.0 + redshifts_helio) + pad

    genrange_mjd = simulator.rates.genrange_mjd
    if genrange_mjd is not None:
        mjd_mins = jnp.maximum(mjd_mins, genrange_mjd[0])
        mjd_maxs = jnp.minimum(mjd_maxs, genrange_mjd[1])

    (mjds, filter_indices, float_stack, simlib_obs_idx, valid, n_valid,
     libid) = schedule_all(
        tables.schedule, field_indices, mjd_mins, mjd_maxs, peak_mjds,
        redshifts_helio, max_obs=simulator.max_obs,
        mskopt=simulator.survey.simlib_mskopt,
        tobs_pad=simulator.survey.genrange_tobs_pad)

    obs = {
        'mjds': mjds, 'filter_indices': filter_indices,
        # what a finished run keeps of each epoch (`simlib_epochs`)
        'simlib_obs_idx': simlib_obs_idx,
        'n_obs_per_sn': n_valid,
        'libid': libid,
    }
    for i, prop in enumerate(FLOAT_PROPERTY_NAMES):
        obs[FLAT_OBS_KEYS[prop]] = float_stack[:, :, i]
    return obs


def epoch_phases(obs_schedule, params):
    """Rest-frame phase of every epoch, from the final peak MJD and zHEL
    once the host has had its say over both (LOAD_TOBS, snlc_sim.c:13005);
    NaN on padded epochs."""
    return ((obs_schedule['mjds'] - params['peakmjd'][:, None])
            / (1.0 + params['sim_redshift_helio'][:, None]))


def apply_random_coord_shift(simulator, params, n_sne,
                             global_start_idx, *, key) -> None:
    """MXRADIUS_RANDOM_SHIFT (gen_random_coord_shift, snlc_sim.c:13113): one
    random offset moves the SN and every host candidate, scaled by cos(DEC)
    at the SN's position. It follows the host's say over the SN, and MWEBV
    and the redshifts keep the unshifted position, an approximation SNANA
    documents as valid for MXRADIUS << 1 deg. Mutates ``params`` in place.
    """
    dec = params['dec']
    shift = random_coord_shift(simulator.host_settings, key, n_sne, global_start_idx,
                               dec)
    if shift is None:
        return

    ra_lo, ra_hi = simulator.survey.genrange_ra

    def shifted_ra(ra):
        # apply_RA_convention (snlc_sim.c:13208)
        ra = ra + shift['ra']
        return jnp.where(ra > ra_hi, ra - 360.0,
                         jnp.where(ra < ra_lo, ra + 360.0, ra))

    params['ra'] = shifted_ra(params['ra'])
    params['dec'] = dec + shift['dec']
    for prefix in HOSTGAL_PREFIXES:
        column = 'host_' if prefix == 'HOSTGAL' else f'host_{prefix}_'
        if f'{column}RA_GAL' not in params:
            continue
        ra_gal = params[f'{column}RA_GAL']
        dec_gal = params[f'{column}DEC_GAL']
        # an unfilled neighbour slot keeps its sentinel
        listed = ra_gal > HOSTGAL_COORD_NULL
        params[f'{column}RA_GAL'] = jnp.where(listed, shifted_ra(ra_gal), ra_gal)
        params[f'{column}DEC_GAL'] = jnp.where(listed, dec_gal + shift['dec'],
                                               dec_gal)
    params['random_shift_RA'] = shift['ra']
    params['random_shift_DEC'] = shift['dec']
    params['random_shift_RADIUS'] = shift['radius']
    params['random_shift_PHI'] = shift['phi']


def _grey_offsets(parameters, model, non1a_templates):
    """SN brightness offsets, grouped by the photometric validity checks."""
    grey_shift = jnp.asarray(parameters['genmag_off'])
    if non1a_templates:
        grey_shift = grey_shift + model.genmag_off_non1a
    host_shift = jnp.asarray(
        parameters.get('snmagshift', jnp.zeros_like(grey_shift)))
    magnitude_shift = (parameters['magoff'] + parameters['magsmear']
                       + parameters['lensdmu'] + parameters['mushift']
                       + model.mag_offset)
    return grey_shift, host_shift, magnitude_shift


def simulate_lightcurves(simulator, catalogue, sn_parameters,
                         obs_schedule, *, key) -> Dict:
    """Simulate light curves for one batch: SEDs at the observed phases,
    magnitudes/FLUXCAL through the rest-frame filters, the magnitude offsets and
    smear terms, PEAKMAG, photometric noise, and the per-obs validity mask.

    This is the core performance-critical GPU function; it is also traced
    whole by the SBI pipeline.
    """
    static_data = simulator.photometry_data
    zp_fluxcal = simulator.instrument.zp_fluxcal
    survey_mjd_span = simulator.survey_mjd_span
    n_sne = len(sn_parameters['sim_redshift_cmb'])

    # filter blueshift uses zHEL
    restframe_filters = precompute_restframe_filters(
        simulator,
        sn_parameters['sim_redshift_helio'], sn_parameters['sim_mwebv'])

    # (epoch, filter) pairs: the observations, SIM_PEAKMAG per filter, MJD_TEMPLATE per band
    max_obs = obs_schedule['phases'].shape[1]
    n_filters = restframe_filters.shape[1]
    obs_mask = (jnp.arange(max_obs)[None, :]
                < obs_schedule['n_obs_per_sn'][:, None])
    every_filter = jnp.broadcast_to(
        jnp.arange(n_filters, dtype=obs_schedule['filter_indices'].dtype),
        (n_sne, n_filters))
    peak_mjd = jnp.asarray(sn_parameters['peakmjd'])

    pair_mjds = [obs_schedule['mjds'],
                 jnp.broadcast_to(peak_mjd[:, None], (n_sne, n_filters))]
    # padded epochs carry -1, which would gather the wrong filter's weights
    pair_filters = [jnp.maximum(obs_schedule['filter_indices'], 0),
                    every_filter]
    pair_mask = [obs_mask, jnp.ones((n_sne, n_filters), dtype=bool)]

    per_band = [simulator.survey.mjd_template.get(filter_band(n), 0.0)
                for n in simulator.filter_names]
    has_template = any(v > 1.0 for v in per_band)
    template_per_band = jnp.asarray(per_band)
    if has_template:
        pair_mjds.append(jnp.broadcast_to(template_per_band[None, :],
                                          (n_sne, n_filters)))
        pair_filters.append(every_filter)
        pair_mask.append(jnp.broadcast_to(template_per_band[None, :] > 1.0,
                                          (n_sne, n_filters)))

    mjds = jnp.concatenate(pair_mjds, axis=1)
    filter_indices = jnp.concatenate(pair_filters, axis=1)
    mask = jnp.concatenate(pair_mask, axis=1)

    sed_params = {k: sn_parameters[k]
                  for k in simulator.model.sed_param_names()
                  if k in sn_parameters}
    sed_params['t0'] = peak_mjd
    sed_params['redshift'] = jnp.asarray(sn_parameters['sim_redshift_helio'])
    sed_params['av'] = jnp.asarray(sn_parameters['av'])
    sed_params['rv'] = jnp.asarray(sn_parameters['rv'])

    # GENMAG_SMEAR_MODELNAME: per-SN DeltaM(lam) on the rest-frame SED (genmag_SALT2.c:2862)
    smear_frac = None
    if simulator.gensmear is not None:
        smear_frac = scatter_factors(
            simulator.gensmear,
            jnp.asarray(sn_parameters['_gensmear_randoms']),
            simulator.wavelength_grid)

    model = simulator.model

    pair_fluxcal = simulate_photometry(
        simulator, sed_params, mjds, filter_indices, mask,
        restframe_filters, smear_frac)

    # grey offsets: GENMAG_OFF_GLOBAL (snlc_sim.c:12941) and GENMAG_OFF_NON1A (:14460)
    grey, host_shift, magnitude_shift = _grey_offsets(
        sn_parameters, simulator.model_settings, model.non1a_templates)
    off_frac = jnp.power(10.0, -0.4 * grey)[:, None]
    pair_fluxcal = pair_fluxcal * off_frac

    pair_magnitudes = fluxcal_to_magnitude(
        pair_fluxcal, filter_indices, static_data)
    # A null magnitude carries the flux SNANA re-derives from mag 99
    # (genmag_SALT2.c:2267, snlc_sim.c:27781), not the model's raw value.
    null_fluxcal = jnp.power(
        10.0, 0.4 * (simulator.instrument.zp_fluxcal - SNANA_NULL_MAG))
    pair_fluxcal = jnp.where(
        pair_magnitudes == SNANA_NULL_MAG, null_fluxcal, pair_fluxcal)
    pair_magnitudes = jnp.where(mask, pair_magnitudes, jnp.nan)
    # Keep padding finite through the offsets; outputs are masked below.
    pair_fluxcal = jnp.where(mask, pair_fluxcal, 0.0)

    # HOSTLIB SNMAGSHIFT: genmag += shift (snlc_sim.c:8490)
    if 'snmagshift' in sn_parameters:
        host_shift = host_shift[:, None]
        pair_magnitudes = jnp.where(
            pair_magnitudes < 90,
            pair_magnitudes + host_shift,
            pair_magnitudes
        )
        flux_scale = jnp.power(10.0, -0.4 * host_shift)
        pair_fluxcal = jnp.where(
            pair_fluxcal != 0,
            pair_fluxcal * flux_scale,
            pair_fluxcal
        )

    # MAGOFF + MAGSMEAR + MAG_OFFSET_GENPDF (snlc_sim.c:8491)
    genmag_off_model = simulator.model_settings.genmag_off_model
    genmag_smear_filter = simulator.model_settings.genmag_smear_filter
    # WEAKLENS_DMU and MUSHIFT are terms of SNANA's MAGOFF sum, so they
    # share its -2<mag<50 gate (snlc_sim.c:8485-8497).
    total_shift = magnitude_shift[:, None]
    # GENMAG_OFF_MODEL: per-filter additive offset (snlc_sim.c:8486).
    if genmag_off_model:
        off_pf = jnp.array([genmag_off_model.get(filter_band(b), 0.0)
                            for b in simulator.photometry.filter_names])
        total_shift = total_shift + off_pf[filter_indices]
    # GENMAG_SMEAR_FILTER: per-filter coherent smear σ[band]·ran[SN,band]
    # (snlc_sim.c:29904); ran drawn once per (SN,filter) in the sampler.
    if genmag_smear_filter and '_magsmear_filter_ran' in sn_parameters:
        sig_pf = jnp.array([genmag_smear_filter.get(filter_band(b), 0.0)
                            for b in simulator.photometry.filter_names])
        ran = jnp.asarray(sn_parameters['_magsmear_filter_ran'])
        smear_pf = sig_pf[filter_indices] * jnp.take_along_axis(ran, filter_indices, axis=1)
        total_shift = total_shift + smear_pf
    # Keep the offset dependence at zero so autodiff sees its derivative.
    mag_in_range = (pair_magnitudes > -2) & (pair_magnitudes < 50)
    pair_magnitudes = jnp.where(
        mag_in_range,
        pair_magnitudes + total_shift,
        pair_magnitudes
    )
    flux_scale = jnp.power(10.0, -0.4 * total_shift)
    pair_fluxcal = jnp.where(
        mag_in_range,
        pair_fluxcal * flux_scale,
        pair_fluxcal
    )

    # GENMAG_SMEAR_ADDPHASECOR
    if '_phasecor_grid' in sn_parameters:
        pc_grid = jnp.asarray(sn_parameters['_phasecor_grid'])
        phases = [jnp.asarray(obs_schedule['phases']),
                  jnp.zeros((n_sne, n_filters))]
        if has_template:
            phases.append((template_per_band[None, :] - peak_mjd[:, None])
                          / (1.0 + sed_params['redshift'][:, None]))
        phases = jnp.concatenate(phases, axis=1)
        inst = simulator.instrument
        idx = (phases - inst.phasecor_phase_min) / inst.phasecor_phase_step
        idx = jnp.clip(idx, 0.0, float(inst.phasecor_n_grid - 1))
        idx_lo = jnp.floor(idx).astype(jnp.int32)
        idx_hi = jnp.minimum(idx_lo + 1, inst.phasecor_n_grid - 1)
        frac = idx - idx_lo.astype(idx.dtype)
        sn_ax = jnp.arange(n_sne)[:, None]
        pc_smear = (1.0 - frac) * pc_grid[sn_ax, idx_lo] + frac * pc_grid[sn_ax, idx_hi]
        mag_ok = (pair_magnitudes > -2) & (pair_magnitudes < 50)
        pair_magnitudes = jnp.where(mag_ok, pair_magnitudes + pc_smear, pair_magnitudes)
        pair_fluxcal = jnp.where(
            mag_ok, pair_fluxcal * jnp.power(10.0, -0.4 * pc_smear), pair_fluxcal)

    # Split only after applying the same corrections at every evaluated epoch.
    fluxcals_all = pair_fluxcal[:, :max_obs]
    magnitudes_all = pair_magnitudes[:, :max_obs]
    peakmags = pair_magnitudes[:, max_obs:max_obs + n_filters]
    template_fluxcal = None
    if has_template:
        template_fluxcal = pair_fluxcal[:, max_obs + n_filters:]
        template_mags = pair_magnitudes[:, max_obs + n_filters:]
        template_fluxcal = jnp.where(
            (template_fluxcal > LOG_SAFE_FLOOR) & (template_mags != SNANA_NULL_MAG),
            template_fluxcal, 0.0)

    noise_key = stream_key(key, 'FLUX_NOISE')
    (fluxcals_noisy, fluxcalerrs, snrs_model, snrs, fluxgal_pe,
     host_galmag_total, host_galmag_galfrac, host_sb_fluxcal_stacked,
     snrs_mon) = add_photometric_noise(
        simulator, catalogue, zp_fluxcal,
        magnitudes_all, fluxcals_all, obs_schedule, noise_key,
        sn_parameters=sn_parameters)

    # template source flux subtracted after the noise draw, from the true flux too (snlc_sim.c:28530)
    if template_fluxcal is not None:
        per_obs = template_fluxcal[
            jnp.arange(fluxcals_all.shape[0])[:, None],
            jnp.maximum(jnp.asarray(obs_schedule['filter_indices']), 0)]
        true_before = fluxcals_all
        fluxcals_all = fluxcals_all - per_obs
        fluxcals_noisy = fluxcals_noisy - per_obs
        snrs_model = snrs_model * jnp.where(
            true_before > LOG_SAFE_FLOOR, fluxcals_all / true_before, 1.0)
        # snlc_sim.c:28580
        snrs = jnp.where(fluxcalerrs > 0, fluxcals_noisy / fluxcalerrs, snrs)

    # DOFILT: NaN outside the model's zHEL range per filter (OBSFLAG_GEN)
    filter_indices = obs_schedule['filter_indices']
    filter_indices_safe = jnp.maximum(filter_indices, 0)
    z_min_table = jnp.array(simulator.filter_z_min, dtype=jnp.float32)
    z_max_table = jnp.array(simulator.filter_z_max, dtype=jnp.float32)
    tidx = jnp.asarray(sn_parameters['template_idx'], dtype=jnp.int32)
    obs_z_min = z_min_table[tidx[:, None], filter_indices_safe]
    obs_z_max = z_max_table[tidx[:, None], filter_indices_safe]
    redshifts = sn_parameters['sim_redshift_helio'][:, None]
    filter_z_valid = (redshifts >= obs_z_min) & (redshifts < obs_z_max) & (obs_z_max > 0)
    filter_z_valid = filter_z_valid & (filter_indices >= 0)

    # SIM_PEAKMJD; the reported peak below may be estimated or smeared
    sn_parameters = {**sn_parameters, 'peakmjd_true': sn_parameters['peakmjd']}
    if simulator.rates.opt_setpkmjd & (8 | 16 | 32):  # flux-based estimators
        snrs_for_fmax = jnp.where(filter_z_valid, snrs, 0.0)

        observed_peak_mjds = _apply_fmax_clump_method(
            simulator.rates, sn_parameters, obs_schedule,
            snrs_for_fmax, fluxcals_noisy, survey_mjd_span)

        sn_parameters = {**sn_parameters, 'peakmjd': observed_peak_mjds}

    elif (simulator.rates.gensigma_peakmjd > 1.0E-9
          and 'peakmjd_smear' in sn_parameters):
        # gen_peakmjd_smear (snlc_sim.c:17120)
        sn_parameters = {**sn_parameters, 'peakmjd': (
            sn_parameters['peakmjd'] + sn_parameters['peakmjd_smear'])}

    magnitudes_out = jnp.where(filter_z_valid, magnitudes_all, jnp.nan)
    fluxcals_out = jnp.where(filter_z_valid, fluxcals_all, jnp.nan)
    fluxcals_noisy_out = jnp.where(filter_z_valid, fluxcals_noisy, jnp.nan)
    fluxcalerrs_out = jnp.where(filter_z_valid, fluxcalerrs, jnp.nan)
    snrs_out = jnp.where(filter_z_valid, snrs, jnp.nan)
    snrs_model_out = jnp.where(filter_z_valid, snrs_model, jnp.nan)

    return {
        'parameters': sn_parameters,
        'observation_schedule': obs_schedule,
        'magnitudes': magnitudes_out,
        'fluxcals': fluxcals_out,
        'fluxcals_noisy': fluxcals_noisy_out,
        'fluxcalerrs': fluxcalerrs_out,
        'snrs': snrs_out,
        'snrs_model': snrs_model_out,
        'snrs_mon': snrs_mon,
        'fluxgal_pe': fluxgal_pe,
        'host_galmag_total': host_galmag_total,
        'host_galmag_galfrac': host_galmag_galfrac,
        'host_sb_fluxcal_stacked': host_sb_fluxcal_stacked,
        'filter_z_valid': filter_z_valid,
        'peakmags': peakmags,
        **epoch_layout(n_sne, max_obs, obs_schedule['n_obs_per_sn']),
    }


def simlib_fieldlist_pass(simulator, field_indices):
    """Per-SN SIMLIB_FIELDLIST test: reject the event when its LIBID's field
    is unlisted, or fails that field's `/N` prescale (SKIP_SIMLIB_FIELD,
    snlc_sim.c:22587-22630). SNANA keeps the full LIBID cycle and drops the
    event, unlike SIMLIB_PRESCALE which drops the entry."""
    return jnp.take(jnp.asarray(simulator.field_data['keep']), field_indices)


def _nrepeat_leader_source_arrays(simulator, params, start_index, template_budget):
    """Per-SN inputs to host selection that the repeat takes from the leader."""
    arrays = {'sim_redshift_helio': params['sim_redshift_helio'],
              'field_idx': params['field_idx']}
    for name, values in wgtmap_sn_variables(simulator, params).items():
        arrays[f'wgtvar_{name}'] = values
    pec1a = host_pec1a_flags(simulator, params['sim_redshift_helio'].shape[0],
                             start_index, template_budget)
    if pec1a is not None:
        arrays['pec1a'] = pec1a
    return arrays


def nrepeat_host_leaders(simulator, params, start_index, n_events,
                         carried, template_budget):
    """HOSTLIB_NREPEAT_GALID_SNPOS: each SN's block leader, whose host it
    reuses, and the leader's host-selection inputs.

    SNANA's flag is `(NCALL_GEN_SNHOST_DRIVER % NREPEAT) > 0`
    (sntools_host.c:5990), and the counter advances only for events that
    reach GEN_SNHOST_DRIVER: those passing the NEPOCH gate at
    snlc_sim.c:12864. So blocks are counted over `prefilter_pass`, not over
    event number; events past ``n_events`` count for nothing. A block may open
    in an earlier batch, so ``carried`` holds the host-call count, the
    in-flight block's leader and its values. Returns the new columns and
    the updated ``carried``.
    """
    nrepeat = simulator.host_settings.nrepeat_galid_snpos
    n = params['prefilter_pass'].shape[0]
    lane = jnp.arange(n)
    passing = params['prefilter_pass'] & (lane < n_events)
    count = jnp.cumsum(passing.astype(jnp.int32))
    # the leader's rank among this batch's passing events; negative when the
    # block opened in an earlier batch
    leader_rank = ((carried['ncall'] + count - 1) // nrepeat) * nrepeat \
        - carried['ncall']
    leader_lane = jnp.searchsorted(count, leader_rank + 1)
    leader = jnp.where(passing,
                       jnp.where(leader_rank < 0, carried['leader'],
                                 start_index + leader_lane),
                       start_index + lane).astype(jnp.int32)
    n_passing = count[-1]
    last = jnp.max(jnp.where(passing, lane, 0))
    local = leader - start_index

    columns = {'host_leader_global_idx': leader}
    values = {}
    for name, source in _nrepeat_leader_source_arrays(
            simulator, params, start_index, template_budget).items():
        gathered = jnp.where(local >= 0, source[jnp.maximum(local, 0)],
                             carried['values'][name])
        columns[f'host_leader_{name}'] = gathered
        values[name] = jnp.where(n_passing > 0, gathered[last],
                                 carried['values'][name])
    return columns, {
        'ncall': carried['ncall'] + n_passing,
        'leader': jnp.where(n_passing > 0, leader[last], carried['leader']),
        'values': values}


def check_finite_flux(simulator, results, obs_schedule, global_start_idx,
                      n_events) -> None:
    """Abort on a non-finite model flux at an observed, valid epoch.

    A model must return a finite flux for every event it is given; one
    that does not would otherwise fail every cut and vanish from the
    output in silence. Runs on the finished batch, so it names the
    model, the event and its drawn parameters; only the first ``n_events``
    events are read. An epoch whose filter is invalid at the event's
    redshift is NaN by design and is not read.
    """
    observed = (obs_schedule['filter_indices'] >= 0) & results['filter_z_valid']
    bad = jnp.any(observed & ~jnp.isfinite(results['fluxcals']), axis=1)
    bad = bad[:n_events]
    if not bool(jnp.any(bad)):
        return
    i = int(jnp.argmax(bad))
    sn_parameters = results['parameters']
    names = [name for name, _ in simulator.model._population]
    drawn = ', '.join(f"{n}={float(sn_parameters[n][i]):g}"
                      for n in names if n in sn_parameters)
    raise SNANAError(
        f"{type(simulator.model).__name__} produced a non-finite flux "
        f"for event {global_start_idx + i} ({drawn}); check the "
        f"GENPEAK_/GENSIGMA_/GENRANGE_ configuration.",
        function_name="check_finite_flux")


def check_crazy_flux(simulator, fluxcals, fluxcalerrs,
                     obs_schedule, sn_parameters, n_events) -> None:
    """Abort on absurd fluxes, as SNANA does (check_crazyFlux,
    snlc_sim.c:28628), over every event, rejected ones included. Runs on
    the finished batch; only the first ``n_events`` events are read.
    """
    cfg = simulator.config
    names = simulator.filter_names
    ratios = cfg.survey.exposure_time
    xt_per_filter = jnp.array(
        [ratios.get(filter_band(n), 1.0) for n in names], dtype=jnp.float32)

    filter_indices = obs_schedule['filter_indices']
    observed = filter_indices >= 0
    xt = xt_per_filter[jnp.maximum(filter_indices, 0)]
    z = jnp.asarray(sn_parameters['sim_redshift_helio'])[:, None]
    # per-epoch intrinsic smear (GENLC.magsmear8): the coherent per-SN
    # term plus GENMAG_SMEAR_FILTER's per-band term
    mag_smear = jnp.asarray(sn_parameters['magsmear'])[:, None]
    smear_filter = simulator.config.model.genmag_smear_filter
    if smear_filter and '_magsmear_filter_ran' in sn_parameters:
        sig_pf = jnp.array(
            [float(smear_filter.get(filter_band(b), 0.0))
             for b in simulator.photometry.filter_names])
        ran = jnp.asarray(sn_parameters['_magsmear_filter_ran'])
        fclip = jnp.clip(filter_indices, 0, ran.shape[1] - 1)
        mag_smear = mag_smear + jnp.where(
            observed,
            sig_pf[fclip] * jnp.take_along_axis(ran, fclip, axis=1), 0.0)
    bound = crazy_flux_bound(
        jnp.abs(jnp.nan_to_num(fluxcalerrs)), z, exposure_ratio=xt,
        zp_fluxcal=cfg.instrument.zp_fluxcal,
        model_factor=simulator.model.crazyflux_factor,
        mag_smear=mag_smear)

    over = (observed & (jnp.abs(fluxcals) > bound))[:n_events]
    if not bool(jnp.any(over)):
        return

    over_np = np.asarray(over)
    isn, iobs = [int(v[0]) for v in np.where(over_np)]
    cids = np.asarray(sn_parameters['cid'])
    band = names[int(filter_indices[isn, iobs])]
    raise SNANAError(
        f"Crazy flux for CID {int(cids[isn])} in {band} at MJD "
        f"{float(obs_schedule['mjds'][isn, iobs]):.3f}: FLUXCAL="
        f"{float(fluxcals[isn, iobs]):.4g} exceeds {float(bound[isn, iobs]):.4g} "
        f"at z={float(z[isn, 0]):.4f}. {int(over_np.sum())} epoch(s) "
        f"are over the bound; the model is returning unphysical magnitudes. "
        f"Set SKIP_CHECK_CRAZYFLUX: 1 to generate anyway.",
        function_name="check_crazy_flux")



def pad_events(tree, n_from, n_to):
    """Edge-repeat the per-event leaves of ``tree`` from ``n_from`` to
    ``n_to`` rows. Padded lanes replicate the last event (its global index
    included, so every keyed draw repeats too) and are trimmed from all
    outputs."""
    def pad(v):
        if hasattr(v, 'shape') and v.ndim >= 1 and v.shape[0] == n_from:
            reps = jnp.broadcast_to(v[-1:], (n_to - n_from,) + v.shape[1:])
            return jnp.concatenate([v, reps], axis=0)
        return v
    return {k: pad(v) for k, v in tree.items()}


def trim_events(value, n):
    """The first ``n`` events of every per-event leaf of ``value``."""
    if isinstance(value, dict):
        return {k: trim_events(v, n) for k, v in value.items()}
    if hasattr(value, 'shape') and value.ndim >= 1 and value.shape[0] != n:
        return value[:n]
    return value


def initial_batch_state(simulator, capacity):
    """What one batch hands the next: the in-flight HOSTLIB_NREPEAT block,
    and the reuse record of USEONCE or MINDAYSEP_SAMEGAL, the latter with
    room for ``capacity`` passing events."""
    hostlib = simulator.hostlib
    names = ['sim_redshift_helio'] + [
        f'wgtvar_{name}' for name in
        (hostlib.selector.snvar_names
         if hostlib is not None and hostlib.selector.has_snvars else ())]
    values = {name: jnp.float32(0.0) for name in names}
    values['field_idx'] = jnp.int32(0)
    if hostlib is not None and hostlib.selector.pec1a_row_shift is not None:
        values['pec1a'] = jnp.bool_(False)
    return {'nrepeat': {'ncall': jnp.int32(0), 'leader': jnp.int32(-1),
                        'values': values},
            'reuse': ({} if hostlib is None
                      else hostlib.selector.initial_reuse(capacity))}


@jax.jit(static_argnames=("adjust",))
def photometry_stage(simulator, key, start_index, n_events,
                     host_state, params, template_budget, *, adjust=None):
    """The forward model up to the photometry for one batch, every draw off
    the root ``key``, in SNANA's order (gen_event_driver,
    snlc_sim.c:12826-13030).

    Draws the placement of the events from ``start_index`` on, schedules them on
    it, applies the prefilter and forms the NREPEAT blocks; then the host,
    the population, the host's overrides and the phases from the final peak;
    then smears the lensing and simulates the photometry. ``params``
    replaces the placement and the population, and must carry every column
    they give: supplied parameters that carry a ``host_galaxy_index`` keep
    their hosts, and otherwise hosts are attached. Of the ``n`` lanes the
    first ``n_events`` are events; the rest are draws past the run's end, or
    repeats of the last supplied event, kept for the batch shape and trimmed
    by the caller.

    ``adjust(key, params) -> params``, when given, sees the parameters once
    the hosts, the population and the host's overrides are in and before
    the photometry; ``params['sn_global_idx']`` keys any per-event draw it
    makes from ``key``.

    Returns ``(results, record, host_state)``: ``results`` at the batch shape,
    ``record`` what the batch reports beside them for :func:`finish_batch`
    and the selection (the population-draw failures and the host draw's
    report, :func:`select_hosts`), and the updated ``host_state``.
    """
    tables = batch_tables(simulator)
    n = simulator.batch_size
    supplied = params is not None
    if supplied:
        batch_parameters = params
        model_first = dust_first = False
    else:
        batch_parameters = draw_placement(simulator, n, start_index,
                                          template_budget, key=key)
        model_first = simulator.model_before_host
        dust_first = simulator.dust_before_host
    record = {}

    def with_model(parameters, host_properties):
        if draw_input_checks(simulator):
            record['draw_input_failure'] = draw_input_failures(
                simulator, parameters['sim_redshift_cmb'], host_properties)
        return {**parameters, **draw_model(
            simulator, parameters, host_properties,
            template_budget, key=key)}

    def with_dust(parameters, host_properties):
        return {**parameters, **draw_dust(
            simulator, parameters, host_properties, key=key)}

    if model_first:
        batch_parameters = with_model(batch_parameters, None)
    if dust_first:
        batch_parameters = with_dust(batch_parameters, None)

    field_indices = jnp.asarray(batch_parameters['field_idx'], dtype=jnp.int32)
    batch_obs_schedule = schedule_observations(
        simulator, tables, field_indices,
        batch_parameters['peakmjd'], batch_parameters['sim_redshift_helio'])
    # NEPOCH < CUTWIN_NEPOCH[0] rejection, whatever APPLY_CUTWIN_OPT (snlc_sim.c:296)
    batch_parameters['prefilter_pass'] = (
        batch_parameters['prefilter_pass']
        & (batch_obs_schedule['n_obs_per_sn'] >= simulator.cuts.nepoch_min)
        & simlib_fieldlist_pass(simulator, field_indices))

    if simulator.host_settings.nrepeat_galid_snpos > 0:
        leaders, carried = nrepeat_host_leaders(
            simulator, batch_parameters, start_index, n_events,
            host_state['nrepeat'], template_budget)
        batch_parameters.update(leaders)
        host_state = {**host_state, 'nrepeat': carried}

    host_properties = None
    attach_hosts = ('host_galaxy_index' not in batch_parameters
                    and simulator.hostlib is not None)
    if attach_hosts:
        mu_before_host = batch_parameters['mu']
        batch_parameters, host_properties, reuse, report =\
            add_host_associations(simulator, tables.catalogue,
                                  batch_parameters, start_index, key=key,
                                  reuse=host_state['reuse'], n_events=n_events,
                                  template_budget=template_budget)
        record.update(report)
        host_state = {**host_state, 'reuse': reuse}
        if supplied or model_first:
            # the amplitude follows the host's distance (gen_modelPar
            # OBS runs after the host, snlc_sim.c:12909)
            batch_parameters.update(simulator.model.shift_amplitude(
                batch_parameters, batch_parameters['mu'] - mu_before_host))
    if not supplied and not model_first:
        batch_parameters = with_model(batch_parameters, host_properties)
    if not supplied and not dust_first:
        batch_parameters = with_dust(batch_parameters, host_properties)
    batch_parameters = apply_host_snpar_overrides(
        simulator, batch_parameters, host_properties)
    if not supplied:
        batch_parameters['genmag_off'] = draw_grey_offset(
            simulator, batch_parameters, host_properties)
    if adjust is not None:
        batch_parameters = adjust(stream_key(key, 'ADJUST'), batch_parameters)
    batch_obs_schedule['phases'] = epoch_phases(batch_obs_schedule,
                                                batch_parameters)
    apply_random_coord_shift(simulator, batch_parameters,
                             n, start_index, key=key)

    # an unmatched CUTWIN_<VAR> name aborts at trace, before the batch runs
    check_generic_cut_names(simulator.cuts, batch_parameters)

    # after the HOSTLIB WEAKLENS_DMU override, as SNANA's own call is
    apply_lensdmu_smear(simulator, batch_parameters, key=key)

    results = simulate_lightcurves(simulator,
                                   tables.catalogue, batch_parameters,
                                   batch_obs_schedule, key=key)
    return results, record, host_state


@jax.jit
def selection_stage(simulator, results, record,
                    host_state, key, start_index, n_events):
    """This batch's selection. Draws are keyed by global event index, so
    per-batch selection concatenates to the whole-population result.

    Under MINDAYSEP_SAMEGAL the passing events then hold their galaxy
    against the later batches: SNANA frees a rejected event's galaxy
    (FREEHOST_GALID, snlc_sim.c:13512), repeat followers never reach
    GEN_SNHOST_GALID, and padding lanes are not events. Returns
    ``(selection, host_state)``.
    """
    tables = batch_tables(simulator)
    selection = _apply_selection_effects_jit_inner(
        simulator, simulator.instrument.zp_fluxcal,
        simulator.search_efficiency, simulator.cutwin, tables.seasons,
        simulator.photprob_prepared, tables.catalogue, results, key=key)
    if 'peak_day' in record:
        params = results['parameters']
        lane = jnp.arange(params['host_galaxy_index'].shape[0])
        leader = (params['host_leader_global_idx']
                  if simulator.host_settings.nrepeat_galid_snpos > 0
                  else start_index + lane)
        register = (selection['passing_mask'] & (lane < n_events)
                    & (leader == start_index + lane))
        host_state = {**host_state, 'reuse': simulator.hostlib.selector.record_peaks(
            host_state['reuse'], params['host_galaxy_index'], record['peak_day'],
            register)}
    return selection, host_state


class BatchTables(NamedTuple):
    """The tables that scale with the run, which the compiled batch takes as
    arguments rather than holding as constants."""
    schedule: Dict
    seasons: Any
    catalogue: Optional[Any]


def batch_tables(simulator) -> BatchTables:
    """The run's :class:`BatchTables`."""
    return BatchTables(
        schedule=simulator.scheduler_tables, seasons=simulator.season_boundaries,
        catalogue=(None if simulator.hostlib is None
                   else simulator.hostlib.catalogue))


@jax.jit(static_argnames=("adjust",))
def simulate_batch(simulator, key, start_index, n_events, host_state,
                   params=None, template_budget=None, *, adjust=None):
    """Generate and select a batch, sharing one program across realisations."""
    results, record, host_state = photometry_stage(
        simulator, key, start_index, n_events, host_state, params,
        template_budget, adjust=adjust)
    selection, host_state = selection_stage(
        simulator, results, record, host_state, key, start_index, n_events)
    return results, selection, record, host_state


def simulate_batch_with_spectra(simulator, key, start_index, n_events,
                               host_state, params, template_budget):
    """Generate spectra between shared photometry and selection kernels.

    A SPECID map reading SNRSUM_REST_V needs these host-side spectra
    before selection, matching SNANA's gen_SEARCHEFF ordering.
    """
    results, record, host_state = photometry_stage(
        simulator, key, start_index, n_events, host_state, params, template_budget)
    spectra, snrsum_rest_v = compute_spectra_for_mask(
        simulator, results,
        np.asarray(results['parameters']['prefilter_pass']), key=key)
    results['parameters']['snrsum_rest_v'] = jnp.asarray(snrsum_rest_v)
    selection, host_state = selection_stage(
        simulator, results, record, host_state, key, start_index, n_events)
    return results, selection, {**record, 'spectra': spectra}, host_state


def run_batch(simulator, *, key, start_index, n_events, host_state,
              params=None, template_budget=None):
    """Run one fixed-capacity batch and validate its real events."""
    if params is not None:
        params = {**params, 'sn_global_idx': start_index + jnp.arange(
            n_events, dtype=jnp.int32)}
        params = pad_events(params, n_events, simulator.batch_size)
    generate = (simulate_batch_with_spectra
                if simulator.specid_uses_snrsum_rest_v else simulate_batch)
    results, selection, record, host_state = generate(
        simulator, key, start_index, n_events, host_state, params, template_budget)
    finish_batch(simulator, results, record, start_index, n_events)
    return results, selection, record.get('spectra'), host_state


def finish_batch(simulator, results, record, start_index, n_events):
    """Report generation failures and attach host-galaxy identifiers.

    Call after compiled simulation, using the returned ``results`` and
    ``record``. Generation failures raise :class:`SNANAError`; when a HOSTLIB
    is used, host identifiers are attached to ``results['parameters']``.

    ``start_index`` is the global event-number offset used in error messages.
    ``n_events`` is the number of real events to check, excluding padding.
    For a complete survey from :func:`build_sbi_pipeline`, use
    ``start_index=0`` and the survey's event count.
    """
    validate_batch(simulator, results, record, start_index, n_events)
    if simulator.hostlib is not None:
        results['parameters'].update(galaxy_labels(simulator,
                                                   results['parameters']))


def validate_batch(simulator, results, record, start_index, n_events):
    """What SNANA aborts on, read from finished results and their record: a
    population-draw input it refuses, an SN variable outside the HOSTLIB
    weight map, a wrong host with no table row nearby, no HOSTLIB galaxy
    to draw within DZTOL, a HOSTLIB_GALID_FORCE host outside
    DZTOL, a USEONCE catalogue run out, a non-finite or an absurd flux.
    Raises SNANAError naming the first offending event, counting from
    ``start_index``; events past ``n_events`` are padding and are not read."""
    params = results['parameters']
    schedule = results['observation_schedule']
    real = np.arange(len(params['cid'])) < n_events
    # the events that reach the host and the population draw, past the
    # NEPOCH gate (snlc_sim.c:12864)
    reason = np.asarray(params['prefilter_reason_code'])
    reached = real & (np.asarray(params['prefilter_pass'])
                      | (reason == PREFILTER_REASON_CODES['HOSTLIB_NOHOST']))

    def first(mask):
        return int(np.argmax(mask))

    if 'draw_input_failure' in record:
        failure = np.where(reached, np.asarray(record['draw_input_failure']), 0)
        if failure.any():
            i = first(failure > 0)
            message = draw_input_checks(simulator)[failure[i] - 1][0]
            raise SNANAError(f"Event {start_index + i}: {message}",
                             function_name="validate_batch")

    if 'wgtmap_snvar_outside' in record:
        selector = simulator.hostlib.selector
        outside = np.where(reached, np.asarray(record['wgtmap_snvar_outside']),
                           0)
        if outside.any():
            i = first(outside > 0)
            k = outside[i] - 1
            centers = selector.snvar_bin_centers[k]
            raise SNANAError(
                f"Event {start_index + i}: {selector.snvar_names[k]} is outside the WGTMAP "
                f"range {float(centers[0]):.4f} to {float(centers[-1]):.4f}.",
                function_name="validate_batch")
    if 'wronghost_uncovered' in record:
        uncovered = reached & np.asarray(record['wronghost_uncovered'])
        if uncovered.any():
            i = first(uncovered)
            raise SNANAError(
                f"Found 0 WRONGHOST entries with |dz| < {DZTRUEMAX} for the "
                f"wrong-host event {start_index + i} at zSN = "
                f"{float(params['sim_redshift_helio'][i]):.5f}; extend the "
                f"ZTRUE/ZMATCH table of WRONGHOST_FILE to cover the generated "
                f"redshift range.", function_name="validate_batch")
    if 'host_window_empty' in record:
        empty = reached & np.asarray(record['host_window_empty'])
        if empty.any():
            i = first(empty)
            z = float(record['host_window_z'][i])
            dztol = float(eval_genpoly(
                np.asarray(simulator.hostlib.selector.dztol_coeffs,
                           dtype=np.float64), z))
            raise SNANAError(
                f"Event {start_index + i}: no HOSTLIB galaxy to draw "
                f"within DZTOL={dztol:.5f} of zHEL={z:.5f}. Loosen "
                f"HOSTLIB_DZTOL or extend the HOSTLIB.",
                function_name="validate_batch")
    if 'host_zdif' in record:
        zdif = np.asarray(record['host_zdif'], dtype=np.float64)
        dztol = eval_genpoly(
            np.asarray(simulator.hostlib.selector.dztol_coeffs,
                       dtype=np.float64),
            np.asarray(record['host_window_z'], dtype=np.float64))
        outside = reached & (np.abs(zdif) > dztol)
        if outside.any():
            i = first(outside)
            raise SNANAError(
                f"Event {start_index + i}: ZDIF={zdif[i]:f} between the SN and "
                f"the HOSTLIB_GALID_FORCE host exceeds dztol={dztol[i]:f}.",
                function_name="validate_batch")
    if 'unplaced' in record:
        exhausted = reached & np.asarray(record['unplaced'])
        if exhausted.any():
            raise SNANAError(
                f"USEONCE: HOSTLIB exhausted: {int(exhausted.sum())} SNe from event "
                f"{start_index + first(exhausted)} have no unused host within the redshift "
                f"tolerance. Use a larger HOSTLIB.",
                function_name="validate_batch")

    check_finite_flux(simulator, results, schedule, start_index, n_events)
    if not simulator.config.run.skip_check_crazyflux:
        check_crazy_flux(simulator, results['fluxcals_noisy'],
                         results['fluxcalerrs'], schedule, params, n_events)


def generate_events(simulator,
                    n_events: int, first_event_index: int,
                    override_arrays: Optional[Dict] = None,
                    template_budget: Optional[np.ndarray] = None,
                    spectra_cache: Optional[Dict] = None,
                    timings=None, *, key, host_state):
    """Simulate ``n_events`` SNe starting at ``first_event_index`` in batches of
    :func:`run_batch`, every draw off the root ``key`` and ``host_state``
    carried from batch to batch. Returns ``(lists, spectra_cache, host_state)``:
    the per-batch accumulation lists for :func:`finalize_results`, the
    pre-selection spectra cache (populated only on the SNRSUM SPECID path)
    and the host_state after the last batch.
    The start_index drives both field round-robin and per-SN RNG (sampler folds
    it into every draw), so batches are distinct and reproducible.

    Each batch's outputs move to host arrays as they accumulate, so device
    memory holds only one batch's tensors at a time. Every random draw is
    keyed by global event index, so the batch split does not change any
    generated value. With ``timings`` a list, each batch appends its event
    and observation counts and the seconds spent simulating it and
    accumulating it.
    """
    batch_size = simulator.batch_size
    n_batches = int(np.ceil(n_events / batch_size))

    lists = {k: [] for k in (
        'parameters', 'observations', 'magnitudes', 'fluxcals',
        'fluxcals_noisy', 'fluxcalerrs', 'snrs', 'snrs_model',
        'peakmags', 'snrs_mon', 'fluxgal_pe', 'host_tables',
        'selection', 'generated', 'epoch_start', 'epoch_count',
        'noise_rows', 'sed_true_mjds', 'sed_true_count')}
    keep_rejected = keeps_rejected_events(simulator.config, simulator.config.numerical)

    def to_host(value):
        return np.asarray(value) if hasattr(value, 'shape') else value

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_events)
        n_batch = batch_end - batch_start
        start_index = first_event_index + batch_start
        batch_budget = (template_budget - batch_start
                        if template_budget is not None else None)
        params = (None if override_arrays is None else
                  {name: val[batch_start:batch_end]
                   for name, val in override_arrays.items()})
        slog.line(logger, f"Batch {batch_idx + 1}/{n_batches}: "
                          f"{n_batch} events from {start_index}")

        start = time.perf_counter()
        batch_results, sel_batch, spectra, host_state = run_batch(
            simulator, key=key, start_index=start_index,
            n_events=n_batch, host_state=host_state, params=params,
            template_budget=batch_budget)
        if spectra is not None:
            spectra_cache = {**(spectra_cache or {}), **spectra}
        if timings is not None:
            jax.block_until_ready(sel_batch['passing_mask'])
            batch_seconds = time.perf_counter() - start
            start = time.perf_counter()

        if n_batch < batch_size:
            batch_results = {k: trim_events(v, n_batch)
                             for k, v in batch_results.items()}
            sel_batch = {k: trim_events(v, n_batch)
                         for k, v in sel_batch.items()}
        n_obs_batch = int(np.sum(np.asarray(
            batch_results['observation_schedule']['n_obs_per_sn'])))

        # the saturation-masked flux, error and SNR (snlc_sim.c:28544-28548)
        for name in ('fluxcals_noisy', 'fluxcalerrs', 'snrs'):
            batch_results[name] = sel_batch.pop(f'{name}_sat')
        # the NOISE dump covers every generated event, before the trigger and cuts
        if simulator.config.output.simgen_dump_noise:
            lists['noise_rows'].append(noise_dump_rows(
                simulator.config, simulator, batch_results, sel_batch, start_index))
        generated = generated_record(batch_results, sel_batch)
        lists['generated'].append(generated)
        sel_batch.pop('cutwin_fail_masks')
        if not keep_rejected:
            kept = kept_mask(simulator.config, sel_batch)
            if simulator.config.effective_spectrograph_optmask & 128:
                epochs = sed_true_epochs(batch_results)
                passing = np.asarray(sel_batch['passing_mask'])
                taken = [epochs[i] if passing[i] else epochs[i][:0]
                         for i in np.flatnonzero(kept)]
                lists['sed_true_mjds'].append(np.concatenate(taken or [np.zeros(0)]))
                lists['sed_true_count'].append(np.array(
                    [len(m) for m in taken], dtype=np.int64))
            batch_results, sel_batch = keep_events(batch_results, sel_batch,
                                                   kept, written_only=True)
        for name in ('epoch_start', 'epoch_count'):
            lists[name].append(np.asarray(batch_results[name]))

        lists['parameters'].append(
            {k: to_host(v) for k, v in batch_results['parameters'].items()})
        # SIMLIB epoch copies and phases are read again by SIMLIB index
        dropped = () if keep_rejected else SIMLIB_EPOCH_KEYS + ('phases',)
        lists['observations'].append(
            {k: to_host(v)
             for k, v in batch_results['observation_schedule'].items()
             if k not in dropped})
        lists['magnitudes'].append(np.asarray(batch_results['magnitudes']))
        lists['fluxcals'].append(np.asarray(batch_results['fluxcals']))
        lists['fluxcals_noisy'].append(np.asarray(batch_results['fluxcals_noisy']))
        lists['fluxcalerrs'].append(np.asarray(batch_results['fluxcalerrs']))
        lists['snrs'].append(np.asarray(batch_results['snrs']))
        lists['snrs_model'].append(np.asarray(batch_results['snrs_model']))
        lists['peakmags'].append(np.asarray(batch_results['peakmags']))
        if simulator.config.output.magmonitor_snr:
            lists['snrs_mon'].append(np.asarray(batch_results['snrs_mon']))
        if keep_rejected:
            lists['fluxgal_pe'].append(np.asarray(batch_results['fluxgal_pe']))
        lists['selection'].append({
            k: ({k2: np.asarray(v2) for k2, v2 in v.items()}
                if isinstance(v, dict)
                else v if v is None else to_host(v))
            for k, v in sel_batch.items()})
        lists['host_tables'].append({
            name: (to_host(batch_results[name])
                   if batch_results.get(name) is not None else None)
            for name in HOST_TABLE_KEYS})
        if timings is not None:
            timings.append({'n_sne': n_batch, 'n_obs': n_obs_batch,
                            'batch': batch_seconds,
                            'accumulate': time.perf_counter() - start})

    return lists, spectra_cache, host_state


def merge_field(name, batches, pad_value=0.0):
    """Concatenate one per-event field across batches, strictly.

    Every batch entry must share a leading event axis; 2-D entries are
    padded on axis 1 to the widest batch with ``pad_value``. A scalar or
    0-d entry must be identical across batches and is kept once. Anything
    else is an internal error: silently keeping the first batch would
    discard later events.
    """
    first = batches[0]
    if isinstance(first, (int, float)) or (
            hasattr(first, 'shape') and first.ndim == 0):
        values = {float(c) for c in batches}
        if len(values) > 1:
            raise SNANAError(
                f"batch merge: scalar field '{name}' differs across "
                f"batches ({sorted(values)}); a per-event field must "
                f"carry an event axis", function_name="merge_field")
        return first
    if not hasattr(first, 'shape'):
        raise SNANAError(
            f"batch merge: field '{name}' is {type(first).__name__}, "
            f"not an array", function_name="merge_field")
    if len(batches) == 1:
        return first
    if first.ndim >= 2:
        max_obs = max(c.shape[1] for c in batches)
        padded = []
        for c in batches:
            if c.shape[1] < max_obs:
                widths = [(0, 0)] * c.ndim
                widths[1] = (0, max_obs - c.shape[1])
                c = np.pad(np.asarray(c), widths, constant_values=pad_value)
            padded.append(c)
        batches = padded
    merged = np.concatenate([np.asarray(c) for c in batches], axis=0)
    batches.clear()
    return merged


def merge_selection(batches):
    """Concatenate per-batch inner-selection dicts along the event axis."""
    merged = {}
    for key in list(batches[0]):
        vals = [c.pop(key) for c in batches]
        if isinstance(vals[0], dict):
            merged[key] = {k: merge_field(k, [v.pop(k) for v in vals])
                           for k in list(vals[0])}
        elif vals[0] is None:
            merged[key] = None
        else:
            merged[key] = merge_field(key, vals)
    return merged


def concatenate_batch_results(all_parameters, all_observations,
                              all_magnitudes, all_fluxcals,
                              all_fluxcals_noisy, all_fluxcalerrs,
                              all_snrs, all_snrs_model) -> Dict:
    """Merge the per-batch result groups into one results dict.

    Every batch must carry the same keys; each field merges through
    ``merge_field`` (strict: no first-batch fallback)."""
    def merge_dicts(dicts, group, pad_values=None):
        keys = set(dicts[0])
        for i, d in enumerate(dicts[1:], start=1):
            if set(d) != keys:
                raise SNANAError(
                    f"batch merge: {group} keys differ between batch 0 "
                    f"and batch {i}: {sorted(keys ^ set(d))}",
                    function_name="concatenate_batch_results")
        pad_values = pad_values or {}
        merged = {}
        for k in keys:
            merged[k] = merge_field(k, [d.pop(k) for d in dicts],
                                    pad_values.get(k, 0.0))
        return merged

    return {
        'parameters': merge_dicts(all_parameters, 'parameter'),
        'observation_schedule': merge_dicts(
            all_observations, 'observation', OBS_PAD_VALUES),
        'magnitudes': merge_field(
            'magnitudes', all_magnitudes, SNANA_NULL_MAG),
        'fluxcals': merge_field('fluxcals', all_fluxcals),
        'fluxcals_noisy': merge_field(
            'fluxcals_noisy', all_fluxcals_noisy),
        'fluxcalerrs': merge_field(
            'fluxcalerrs', all_fluxcalerrs, SNANA_NULLFLOAT),
        'snrs': merge_field('snrs', all_snrs),
        'snrs_model': merge_field('snrs_model', all_snrs_model),
    }


def finalize_results(lists: Dict):
    """Concatenate per-batch accumulation lists into the final results dict.

    Returns ``(results, selection_inner)``: the merged results and the
    authoritative per-batch inner selection, merged. The selection is
    finalised (pass and fail counts, logging, the SEARCHEFF checks) once,
    afterwards, by `finalize_selection`.
    """
    # the flattened per-epoch arrays concatenate, so each batch's spans
    # move by the epochs before it
    flat_starts = np.cumsum([0] + [int(m.size) for m in lists['magnitudes']])[:-1]
    epoch_start = np.concatenate([s + f for s, f
                                  in zip(lists['epoch_start'], flat_starts)])
    epoch_count = np.concatenate(lists['epoch_count'])
    noise_rows = ({k: np.concatenate([r[k] for r in lists['noise_rows']])
                   for k in lists['noise_rows'][0]}
                  if lists['noise_rows'] else None)
    results = concatenate_batch_results(
        lists['parameters'], lists['observations'], lists['magnitudes'],
        lists['fluxcals'], lists['fluxcals_noisy'], lists['fluxcalerrs'],
        lists['snrs'], lists['snrs_model'])
    results['peakmags'] = merge_field('peakmags', lists['peakmags'])
    if lists['snrs_mon']:
        results['snrs_mon'] = merge_field('snrs_mon', lists['snrs_mon'])
    if lists['fluxgal_pe']:
        results['fluxgal_pe'] = merge_field('fluxgal_pe', lists['fluxgal_pe'])
    results['epoch_start'] = epoch_start
    results['epoch_count'] = epoch_count
    results['noise_rows'] = noise_rows
    if lists['sed_true_mjds']:
        results['sed_true_mjds'] = np.concatenate(lists['sed_true_mjds'])
        results['sed_true_count'] = np.concatenate(lists['sed_true_count'])

    for key in HOST_TABLE_KEYS:
        batches = [cr[key] for cr in lists['host_tables']
                   if cr.get(key) is not None]
        results[key] = (merge_field(key, batches) if batches
                        else None)

    selection = merge_selection(lists['selection'])
    selection['generated'] = merge_selection(lists['generated'])
    return results, selection


def _remap_spectra_cids(spectra, cid_map):
    """Re-key a spectra dict (and each SpectrumResult.snid) old CID -> new."""
    out = {}
    for old, specs in spectra.items():
        new = cid_map.get(int(old), int(old))
        for s in specs:
            s.snid = new
        out[new] = specs
    return out


def ngen_lc_batch_alloc(batch_size: int, needed: np.ndarray) -> np.ndarray:
    """Split one batch across classes in proportion to what they still need.

    A finished class gets nothing. The one-event floor keeps classes whose
    share rounds to zero progressing alongside the rest rather than waiting
    for the others to finish; quotas fill either way.
    """
    short = needed > 0
    alloc = np.zeros(len(needed), dtype=np.int64)
    alloc[short] = 1
    spare = batch_size - int(short.sum())
    if spare > 0:
        extra = (needed[short] / needed[short].sum() * spare).astype(np.int64)
        extra[np.argmax(needed[short])] += spare - int(extra.sum())
        alloc[short] += extra
    return alloc


def renumber_ngen_lc_cids(final: Dict, ilc: np.ndarray, spectra_cache):
    """Set CIDs to SNANA's `CIDOFF + ilc`, the accepted slot being filled.

    Retried attempts on the same slot share its CID, as in SNANA.
    `sn_global_idx` is left alone: selection randomness is keyed off it.
    Returns the spectra cache re-keyed to the new CIDs.
    """
    params = final['parameters']
    old = np.asarray(params['cid'])
    new = ilc.astype(old.dtype)
    params['cid'] = new
    cid_map = {int(o): int(n) for o, n in zip(old, new)}

    if spectra_cache:
        spectra_cache = _remap_spectra_cids(
            {c: s for c, s in spectra_cache.items() if c in cid_map},
            cid_map)
    return spectra_cache


def run_ngen_lc(simulator, target: int, *, key,
                host_state, timings=None):
    """Generate fixed-size batches until ``target`` SNe pass trigger+cuts.

    SNANA NGEN_LC semantics (snlc_sim.c: reject → don't count, loop until
    N accepted). jax-specific generation-budget guard replaces SNANA's
    unbounded loop (SNANA has no generic cap for SEARCHEFF/CUTWIN).

    Selection runs once per batch inside `generate_events`; the merged
    per-batch results are both this loop's stopping rule and the final
    authoritative selection (subset along with the kept events).

    Returns ``(results, selection_inner, spectra_cache)``.
    """
    budget = simulator.config.run.ngen_lc_maxgen_factor * target
    if simulator.specid_uses_snrsum_rest_v and simulator.config.run.ngen_lc_maxgen_factor > 20:
        logger.warning(
            "NGEN_LC + SNRSUM_REST_V SPECID: spectra are generated for every "
            "candidate per batch; large NGEN_LC_MAXGEN_FACTOR (%d) may be slow.",
            simulator.config.run.ngen_lc_maxgen_factor)

    sed_model = simulator.model
    quotas = sed_model.generation_quotas(int(target))
    accepted = np.zeros_like(quotas)

    spectra_cache = None
    combined = None
    n_generated = 0
    it = 0
    while (accepted < quotas).any():
        cum_passing = int(accepted.sum())
        if n_generated >= budget:
            raise SNANAError(
                f"NGEN_LC jax generation-budget guard: {cum_passing} of "
                f"{target} accepted after generating {n_generated} "
                f"(budget {budget} = {simulator.config.run.ngen_lc_maxgen_factor}x "
                f"target, pass rate p≈{cum_passing / max(n_generated, 1):.4f}). "
                f"Cuts may be too tight; raise NGEN_LC_MAXGEN_FACTOR for a "
                f"legitimately low-yield sim.",
                function_name="run_ngen_lc",
            )
        alloc = ngen_lc_batch_alloc(simulator.batch_size, quotas - accepted)
        batch, spectra_cache, host_state = generate_events(
            simulator,
            int(alloc.sum()), n_generated,
            template_budget=(np.cumsum(alloc)
                             if sed_model.non1a_templates else None),
            spectra_cache=spectra_cache, timings=timings, key=key, host_state=host_state)
        passing = np.concatenate([g['passing'] for g in batch['generated']])
        slots = (np.concatenate([g['template_slot'] for g in batch['generated']])
                 if sed_model.non1a_templates
                 else np.zeros(len(passing), dtype=np.int64))
        accepted = accepted + np.bincount(slots[passing], minlength=len(quotas))
        n_generated += len(passing)
        it += 1

        if combined is None:
            combined = batch
        else:
            for name, vals in batch.items():
                combined[name].extend(vals)
        slog.line(logger, f"NGEN_LC batch {it}: generated {len(passing)}, "
                  f"passing {int(passing.sum())}, quota filled "
                  f"{int(np.minimum(accepted, quotas).sum())}/{target}")

    start = time.perf_counter()
    final, selection_inner = finalize_results(combined)
    _closing_phase(timings, start)

    # each class's attempts up to the one that fills its quota, labelled with SNANA's ilc
    generated = selection_inner.pop('generated')
    passing = generated['passing']
    slots = (generated['template_slot'] if sed_model.non1a_templates
             else np.zeros(len(passing), dtype=np.int64))
    keep = np.zeros(len(passing), dtype=bool)
    ilc = np.zeros(len(passing), dtype=np.int64)
    start_index = 0
    for slot, quota in enumerate(quotas):
        idx = np.flatnonzero(slots == slot)
        attempt = np.cumsum(passing[idx]) - passing[idx]
        keep[idx[attempt < quota]] = True
        ilc[idx] = start_index + attempt + 1
        start_index += quota

    # the generated record and the NOISE rows are by generated event, the
    # rest by kept event
    noise_rows = final.pop('noise_rows')
    global_idx = np.asarray(final['parameters']['sn_global_idx'])
    final, selection_inner = keep_events(final, selection_inner,
                                         keep[global_idx], written_only=False)
    selection_inner['generated'] = {
        k: ({k2: v2[keep] for k2, v2 in v.items()} if isinstance(v, dict)
            else v[keep])
        for k, v in generated.items()}
    if noise_rows is not None:
        rows = keep[noise_rows['event']]
        final['noise_rows'] = {k: v[rows] for k, v in noise_rows.items()}
        final['noise_rows']['cid'] = ilc[noise_rows['event'][rows]]
    else:
        final['noise_rows'] = None
    spectra_cache = renumber_ngen_lc_cids(
        final, ilc[global_idx[keep[global_idx]]], spectra_cache)
    return final, selection_inner, spectra_cache


def _closing_phase(timings, start):
    """Add the time since ``start`` to the run's closing phase: the merge of
    the batches and the selection's finalisation, after the last batch, on
    the last batch's record."""
    if timings:
        timings[-1]['finalize'] = (timings[-1].get('finalize', 0.0)
                                   + time.perf_counter() - start)


def run_simulation(simulator, timings=None,
                   override_parameters: Optional[Dict] = None):
    """Run the complete vectorised simulation, one batch at a time.

    Each batch is sampled, scheduled, simulated and selected in turn, so
    device memory is bounded by ``batch_size``; the per-batch selections
    are merged and finalised once over the accumulated events.

    Returns ``(results, selection, spectra_cache)``. ``override_parameters``
    replaces the sampled parameter dictionary, for replay and debugging.
    ``timings``, a list, receives one record per batch (see
    :func:`generate_events`), the last carrying the closing merge.
    """
    simulation = simulator.config
    config = simulation.numerical

    if (simulator.hostlib is not None
            and simulator.hostlib.selector.uses_mindaysep
            and simulator.batch_size > 1):
        logger.warning(
            "MINDAYSEP_SAMEGAL with batch_size=%d is exact across batches "
            "but approximate within a batch (rejected events free their "
            "host only at the batch boundary). Set batch_size=1 for "
            "exact SNANA free-on-reject behaviour.",
            simulator.batch_size)

    key = run_key(simulation)
    if override_parameters is not None:
        override_arrays = {k: jnp.array(v) for k, v in override_parameters.items()}
        n_sne_total = len(override_arrays['sim_redshift_cmb'])
        host_state = initial_batch_state(simulator, n_sne_total)
        lists, spectra_cache, _ = generate_events(
            simulator,
            n_sne_total, 0, override_arrays=override_arrays,
            timings=timings, key=key, host_state=host_state)
        start = time.perf_counter()
        results, selection_inner = finalize_results(lists)
        _closing_phase(timings, start)
    elif simulator.ngen_lc <= 0:
        host_state = initial_batch_state(simulator, simulator.n_sne)
        lists, spectra_cache, _ = generate_events(
            simulator, simulator.n_sne, 0,
            timings=timings, key=key, host_state=host_state)
        start = time.perf_counter()
        results, selection_inner = finalize_results(lists)
        _closing_phase(timings, start)
    else:
        # NGEN_LC stops once the target passes, which the last instalment
        # can overshoot by at most one batch
        host_state = initial_batch_state(simulator,
                                    simulator.ngen_lc + simulator.batch_size)
        results, selection_inner, spectra_cache = run_ngen_lc(
            simulator, simulator.ngen_lc, key=key,
            host_state=host_state, timings=timings)

    start = time.perf_counter()
    selection = finalize_selection(simulator, results, dict(selection_inner))
    _closing_phase(timings, start)
    return results, selection, spectra_cache


def spectrum_prescales(simulator, results, *, key):
    """Keep decisions for each event's spectrum entries, before scheduling."""
    cfg = simulator.config.spectra
    params = results['parameters']
    field_indices = np.asarray(params['field_idx'])
    fields = np.array([simulator.fields[int(i)].field_name for i in field_indices])
    default = {'field': '', 'wgt': 1.0, 'epoch_frame': ''}
    if simulator.config.effective_spectrograph_optmask & 128:
        groups = [(np.arange(len(fields)), [default])]
    elif simulator.config.survey.use_simlib_take_spectrum:
        groups = [(np.flatnonzero(field_indices == i), simulator.simlib_take_entries[i])
                  for i in np.unique(field_indices)]
    else:
        entries = ([default] if simulator.config.survey.use_simlib_spectrograph
                   else cfg.take_spectrum or [default])
        groups = [(np.arange(len(fields)), entries)]
    if not (cfg.take_spectrum_prescale or cfg.take_spectrum_dumpcid >= 0
            or any(e['field'] or e['wgt'] < 1 for _, es in groups for e in es)):
        return None
    key_draws = event_keys(key, 'SPEC_PRESCALE_KEY', params['sn_global_idx'])
    field_draws = np.asarray(jax.vmap(jax.random.uniform)(
        event_keys(key, 'SPEC_PRESCALE_FIELD', params['sn_global_idx'])))
    max_entries = max((len(es) for _, es in groups), default=0)
    draws = []
    for index in range(max_entries):
        keys = (key_draws if index == 0 else
                jax.vmap(lambda k: jax.random.fold_in(k, index))(key_draws))
        draws.append(np.asarray(jax.vmap(jax.random.uniform)(keys)))
    keep = np.zeros((len(fields), max_entries), dtype=bool)
    for rows, entries in groups:
        for c, entry in enumerate(entries):
            keep[rows, c] = keep_spectrum(
                fields[rows], entry['field'], entry['wgt'],
                cfg.take_spectrum_prescale,
                draws[entry.get('prescale_key', c)][rows], field_draws[rows],
                is_host=entry['epoch_frame'] == 'HOST')
    for i in np.flatnonzero(np.asarray(params['cid']) == cfg.take_spectrum_dumpcid):
        logger.debug("TAKE_SPECTRUM_DUMPCID %d: field=%s keep=%s",
                     int(params['cid'][i]), fields[i], keep[i])
    return keep



def sed_true_epochs(results):
    """Per event, the MJDs the SED-true spectra are taken at: every
    scheduled epoch, deduplicated at 0.05 d (snlc_sim.c:10953). A packed
    run carries them under ``sed_true_mjds`` (taken per batch, since the
    pack keeps only the written epochs); a batch reads its schedule."""
    n_events = len(results['observation_schedule']['n_obs_per_sn'])
    if 'sed_true_mjds' in results:
        mjds, count = results['sed_true_mjds'], results['sed_true_count']
        start = np.cumsum(count) - count
        return [mjds[a:a + c] for a, c in zip(start, count)]
    mjds = np.asarray(results['observation_schedule']['mjds'],
                      dtype=np.float64).reshape(-1)
    epochs = []
    for row in range(n_events):
        m = mjds[row_slice(results, row)]
        epochs.append(m[np.diff(m, prepend=-99.0) > 0.05])
    return epochs


def compute_spectra_for_mask(simulator, results, mask, *,
                             key):
    """Generate spectra for the SNe in ``mask``, every draw off the root
    ``key``. Returns ``(spectra_by_snid, snrsum_rest_v)``.
    """
    cfg = simulator.config
    if cfg.spectra.spectrograph_optmask & 2048:  # NOSPEC
        n_sne = len(results['parameters']['sim_redshift_cmb'])
        return {}, jnp.zeros(n_sne)

    grey, host_shift, magnitude_shift = _grey_offsets(
        results['parameters'], simulator.config.model, simulator.model.non1a_templates)
    sn_flux_factor = jnp.power(10.0, -0.4 * (grey + host_shift + magnitude_shift))
    static_data_with_model = dict(
        simulator.photometry_data, sn_flux_factor=sn_flux_factor,
        model_instance=simulator.model, latetime_extrap=True,
        gensmear=simulator.gensmear)

    static_data_with_model['spectrum_keep'] = spectrum_prescales(
        simulator, results, key=key)

    rng_key = stream_key(key, 'SPECTRUM_PRESCALE')

    if cfg.effective_spectrograph_optmask & 128:
        # true-SED spectra at every photometry epoch, dedup 0.05 d, TEXPOSE 20 (snlc_sim.c:10953)
        spec_mjds_list = sed_true_epochs(results)
        # epochs outside the model's phase support (genmag_SALT2.c:3592-3595)
        mi = simulator.model
        daymin, daymax = mi.phase_range()
        trest_hi = (daymax - SED_PHASE_EDGE_EPS
                    if mi.extrap_phase_method == 1 else np.inf)
        return generate_simlib_spectra(
            calib=simulator.calibration,
            sn_parameters=results['parameters'],
            spec_mjds_list=spec_mjds_list,
            spec_texposes_list=[np.full(len(m), 20.0) for m in spec_mjds_list],
            static_data=static_data_with_model,
            template_texpose=0.0,
            trest_range=(daymin + 0.01, trest_hi),
            rng_key=rng_key,
            passing_mask=np.asarray(mask),
            # the noise-free SED: SCALE_TEXPOSE and SCALE_SNR have nothing to act on
            spectro_options={
                'optmask': cfg.effective_spectrograph_optmask,
                'zero_neg_flam': bool(cfg.model.genmodel_mskopt & 16),
            },
        )

    if cfg.survey.use_simlib_take_spectrum:
        fields = simulator.fields
        fidx = np.asarray(results['parameters']['field_idx'])
        entries_by_field = simulator.simlib_take_entries
        return generate_simlib_take_spectrum(
            calib=simulator.calibration,
            entries_per_sn=[entries_by_field[i] for i in fidx],
            sn_parameters=results['parameters'],
            static_data=static_data_with_model,
            texpose_template_scale=cfg.spectra.take_spectrum_template_texpose_scale,
            trest_range=cfg.rates.trest_range,
            rng_key=rng_key,
            passing_mask=np.asarray(mask),
            spectro_options={
                'optmask': cfg.spectra.spectrograph_optmask,
                'scale_texpose': cfg.spectra.spectrograph_scale_texpose,
                'scale_snr': cfg.spectra.spectrograph_scale_snr,
                'warp_lampoly': cfg.spectra.warp_spectrum_lampoly,
                'zero_neg_flam': bool(cfg.model.genmodel_mskopt & 16),
            },
        )

    if cfg.survey.use_simlib_spectrograph:
        fields = simulator.fields
        fidx = np.asarray(results['parameters']['field_idx'])
        return generate_simlib_spectra(
            calib=simulator.calibration,
            sn_parameters=results['parameters'],
            spec_mjds_list=[fields[i].spec_mjds for i in fidx],
            spec_texposes_list=[fields[i].spec_texposes for i in fidx],
            static_data=static_data_with_model,
            # GENSPEC.TEXPOSE_TEMPLATE stays 0 for SIMLIB spectra
            template_texpose=0.0,
            trest_range=cfg.rates.trest_range,
            rng_key=rng_key,
            passing_mask=np.asarray(mask),
            spectro_options={
                'optmask': cfg.spectra.spectrograph_optmask,
                'scale_texpose': cfg.spectra.spectrograph_scale_texpose,
                'scale_snr': cfg.spectra.spectrograph_scale_snr,
                'warp_lampoly': cfg.spectra.warp_spectrum_lampoly,
                'zero_neg_flam': bool(cfg.model.genmodel_mskopt & 16),
            },
        )

    return generate_all_spectra(
        calib=simulator.calibration,
        take_spectrum_configs=simulator.config.spectra.take_spectrum,
        sn_parameters=results['parameters'],
        static_data=static_data_with_model,
        texpose_template_scale=cfg.spectra.take_spectrum_template_texpose_scale,
        trest_range=cfg.rates.trest_range,
        rng_key=rng_key,
        passing_mask=np.asarray(mask),
        spectro_options={
            'optmask': cfg.spectra.spectrograph_optmask,
            'scale_texpose': cfg.spectra.spectrograph_scale_texpose,
            'scale_snr': cfg.spectra.spectrograph_scale_snr,
            'warp_lampoly': cfg.spectra.warp_spectrum_lampoly,
            'zero_neg_flam': bool(cfg.model.genmodel_mskopt & 16),
        },
        host_spec=(None if simulator.hostlib is None
                   else simulator.hostlib.spectra),
        hostfrac=cfg.spectra.take_spectrum_hostfrac,
        hostsnfrac=cfg.spectra.take_spectrum_hostsnfrac,
    )



def compute_atmosphere_for_mask(simulator, atmos_static,
                                results, selection_result, columnar_data,
                                spectra_by_sn, *, key) -> None:
    """Airmass + DCR per obs, attached to columnar_data['observations'].

    Mirrors GEN_ATMOSPHERE_DRIVER: output-only, so it runs as a post-pass on
    the passing-obs grid and never perturbs simulated flux. The chromatic SED
    is the SED-true sim_flam interpolated onto the shared filter grid.
    """
    atm = simulator.config.instrument
    site, wave, trans_by_filter, cal_flux = atmos_static
    n_wave = wave.shape[0]

    passing = np.asarray(selection_result['passing_mask'])
    passing_indices = np.where(passing)[0]
    obs = results['observation_schedule']

    def grid(array, fill):
        return padded_rows(results, array, passing_indices, fill)

    mjd = grid(np.asarray(simlib_epochs(simulator, results, 'mjds'),
                          dtype=np.float64), 0.0)
    n_sne, max_obs = mjd.shape
    filter_idx = grid(obs['filter_indices'], -1)
    ra = np.asarray(results['parameters']['ra'], dtype=np.float64)[passing_indices]
    dec = np.asarray(results['parameters']['dec'], dtype=np.float64)[passing_indices]
    psf1s = grid(np.asarray(simlib_epochs(simulator, results, 'psf1s'),
                            dtype=np.float64), 0.0)
    pixsizes = np.asarray(simulator.flat_field_index['pixsizes'])
    pixsize = pixsizes[np.asarray(results['parameters']['field_idx'])[passing_indices]]
    psf_fwhm = psf1s * pixsize[:, None] * 2.355          # SIMLIB_OBS_GEN.PSF_FWHM
    snr_true = grid(np.asarray(results['snrs_model'], dtype=np.float64), 0.0)
    snr_obs = grid(np.asarray(results['snrs'], dtype=np.float64), 0.0)
    obs_valid = grid(selection_result['valid_mask'], False)

    # each epoch's SED is the nearest SED-true spectrum in MJD (IMJD_GENSPEC); none gives DCR null
    cids = np.asarray(results['parameters']['cid'])[passing_indices]
    lam_avg = None
    spec_rows = []
    spec_row = np.full((n_sne, max_obs), -1, dtype=np.int32)
    for i in range(n_sne):
        if not obs_valid[i].any():
            continue
        specs = spectra_by_sn.get(int(cids[i]))
        if not specs:
            continue
        if lam_avg is None:
            lam_avg = 0.5 * (np.asarray(specs[0].lam_min)
                             + np.asarray(specs[0].lam_max))
        spec_mjds = np.array([s.mjd for s in specs])
        row_of_spec = {}
        for j in np.where(obs_valid[i])[0]:
            k = int(np.argmin(np.abs(spec_mjds - mjd[i, j])))
            if abs(spec_mjds[k] - mjd[i, j]) < 0.06:
                if k not in row_of_spec:
                    row_of_spec[k] = len(spec_rows)
                    spec_rows.append(np.interp(
                        wave, lam_avg, np.asarray(specs[k].sim_flam),
                        left=0.0, right=0.0))
                spec_row[i, j] = row_of_spec[k]
    spec_flam = (np.asarray(spec_rows, dtype=np.float64) if spec_rows
                 else np.zeros((0, n_wave), dtype=np.float64))

    sigma_site = (atm.sigma_site_temp, atm.sigma_site_bp, atm.sigma_site_pwv)
    out = atm_mod.compute_atmosphere(
        mjd=mjd, filter_idx=filter_idx, valid_mask=obs_valid,
        ra_deg=ra, dec_deg=dec, global_indices=passing_indices,
        psf_fwhm=psf_fwhm, snr_true=snr_true, snr_obs=snr_obs,
        spec_flam=spec_flam, spec_row=spec_row, wave=wave,
        trans_by_filter=trans_by_filter,
        cal_flux=cal_flux, site=site, coordres_poly=atm.dcr_coordres_poly,
        magshift_poly=atm.dcr_magshift_poly, coordres_floor=atm.coordres_floor,
        sigma_site=sigma_site, key=key)

    flat_valid = obs_valid.ravel()
    obs_cols = columnar_data['observations']
    for name in ('airmass', 'd_ra', 'd_dec', 'sim_dcr_dra', 'sim_dcr_ddec',
                 'sim_dcr_dmag', 'sim_dcr', 'coordres', 'lamavg_sed_wgted',
                 'psf_fwhm', 'snr_true', 'snr_obs'):
        obs_cols[name] = out[name].ravel()[flat_valid]

    # per-(SN, band) SIM band-averaged true coords -> RA_AVG_<band> HEAD cols
    meta = columnar_data['metadata']
    meta['ra_avg_band'] = out['ra_avg_band']
    meta['dec_avg_band'] = out['dec_avg_band']
    meta['avg_band_names'] = [filter_band(n) for n in simulator.filter_names]


def draw_cidran(simulation, cids):
    """FORMAT_MASK CIDRAN(16): unique random CIDs strictly between
    CIDRAN_MIN and CIDRAN_MAX, less CIDRAN_SKIPLIST, one per event
    (init_CIDRAN). The list is keyed by RANSEED alone, so jobs sharing a
    seed take disjoint slices of one list.

    As SNANA does, the first CIDOFF - CIDRAN_MIN draws are another job's,
    and a draw that repeats an earlier one or falls on the skip list is
    redrawn, all such draws at once per round; not byte-matched to SNANA.
    SNANA never draws either end of the range (snlc_sim.c:23430-23431) and
    aborts once CIDOFF - CIDRAN_MIN plus the event count reaches CIDRAN_MAX
    (:23366), and so does this.
    """
    cids = np.asarray(cids)
    n = len(cids)
    cfg = simulation
    cmin, cmax, coff = cfg.output.cidran_min, cfg.output.cidran_max, cfg.output.cidoff
    if coff < cmin:
        raise SNANAError(f"CIDOFF={coff} must be >= CIDRAN_MIN={cmin}")
    skip = jnp.asarray(sorted(c for c in set(cfg.output.cidran_skiplist)
                              if cmin < c < cmax), dtype=jnp.int32)
    off = coff - cmin
    m = off + n
    if m >= cmax:
        # init_CIDRAN's NSTORE_ALL bound (snlc_sim.c:23366)
        raise SNANAError(f"CIDOFF-CIDRAN_MIN + {n} events = {m} reaches "
                         f"CIDRAN_MAX={cmax}")
    # SNANA never draws either end of the range (snlc_sim.c:23430-23431)
    pool = cmax - cmin - 1
    if m + skip.shape[0] > pool:
        raise SNANAError(
            f"CIDRAN pool ({cmin},{cmax}) too small for {n} events (CIDOFF={coff})")
    key = stream_key(root_key(simulation.run.random_seed), 'CIDRAN')

    def draw(round_):
        return cmin + 1 + jax.random.randint(jax.random.fold_in(key, round_),
                                             (m,), 0, pool, dtype=jnp.int32)

    def refused(values):
        # a repeat of an earlier draw, the earlier kept, or a skipped CID
        order = jnp.argsort(values, stable=True)
        ranked = values[order]
        repeat = jnp.zeros(m, dtype=bool).at[order[1:]].set(
            ranked[1:] == ranked[:-1])
        return repeat | jnp.isin(values, skip)

    values, round_ = draw(0), 0
    pending = refused(values)
    while bool(jnp.any(pending)):
        round_ += 1
        values = jnp.where(pending, draw(round_), values)
        pending = refused(values)
    return np.asarray(values[off:]).astype(cids.dtype)


def to_written_cid(simulation, cids, fmt):
    """Return the map from event CID to the CID written to file, SNANA's
    GENLC.CID_FINAL (snlc_sim.c:13007): CIDOFF added (GENLC.CID = CIDOFF +
    ilc, :12863) unless FORMAT_MASK CIDRAN(16) draws them. `cid` itself stays
    the working identity that GALID_UNIQUE and the spectra keys derive from,
    as GENLC.CID does."""
    if not fmt['cidran']:
        return lambda c: np.asarray(c) + simulation.output.cidoff
    drawn = dict(zip(np.asarray(cids).tolist(),
                     draw_cidran(simulation, cids).tolist()))
    return lambda c: np.array([drawn[int(x)] for x in np.asarray(c)])


def write_filter_transmission_files(simulator, base_dir, version_name):
    """FORMAT_MASK FILTERS(256): write per-band transmission curves
    (wr_SIMGEN_FILTERS, snlc_sim.c:14636)."""
    kcor = simulator.kcor
    out = Path(base_dir) / version_name / f'{version_name}.FILTERS'
    out.mkdir(parents=True, exist_ok=True)
    for name in kcor.get_filter_names():
        wave, trans = (np.asarray(a) for a in kcor.get_filter_transmission(name))
        char = filter_band(name)
        with open(out / f'{char}.dat', 'w') as f:
            for w, t in zip(wave, trans):
                f.write(f"{float(w):7.1f}  {float(t):.6f}\n")


def write_verify_sed_true(simulator, base_dir, version_name, spectra_by_sn,
                          results):
    """Write the VERIFY_SED_TRUE table (wr_VERIFY_SED_TRUE, snlc_sim.c:11538):
    one row per spectrum epoch with broadband MAG and synthetic MAG_VERIFY."""
    params = results['parameters']
    obs = results['observation_schedule']
    z_cmb = np.asarray(params['sim_redshift_cmb'])
    peak = np.asarray(params['peakmjd'])
    mjds = np.asarray(simlib_epochs(simulator, results, 'mjds'))
    fidx = np.asarray(obs['filter_indices'])
    mags = np.asarray(results['magnitudes'])
    obs_bands = [filter_band(n)
                 for n in simulator.photometry.filter_names]
    mean_wl = np.asarray(simulator.photometry.filter_mean_wavelengths)
    cid_to_idx = {int(c): i for i, c in enumerate(np.asarray(params['cid']))}
    mjds, fidx, mags = (np.asarray(a).reshape(-1) for a in (mjds, fidx, mags))

    lines = ["VARNAMES: CID z MJD TREST LAMREST BAND MAG MAG_VERIFY MAG_DIF\n"]
    for cid in sorted(spectra_by_sn):
        si = cid_to_idx[cid]
        span = row_slice(results, si)
        z = float(z_cmb[si]); z1 = 1.0 + z
        for spec in spectra_by_sn[cid]:
            dt = np.abs(mjds[span] - spec.mjd)
            oi = int(np.argmin(dt))
            if dt[oi] > 0.06:
                continue
            bi = int(fidx[span][oi])
            band = obs_bands[bi]
            genmag = float(mags[span][oi])
            if genmag < 1.0 or genmag > 60.0:
                continue
            synmag = (spec.synmag or {}).get(band, 99.0)
            trest = (spec.mjd - float(peak[si])) / z1
            lines.append(
                f"SN: {cid:8d}  {z:.3f}  {spec.mjd:.3f}  {trest:6.2f}  "
                f"{mean_wl[bi] / z1:6.0f}  {band}   "
                f"{genmag:.5f} {synmag:.5f} {synmag - genmag:8.5f}\n")
    # SNANA's own VERFIY spelling (snlc_sim.c:11569)
    path = Path(base_dir) / f"{version_name}_VERFIY_SED_TRUE.DAT"
    with open(path, 'w') as f:
        f.writelines(lines)


def validate_format_mask(simulation) -> None:
    """Enforce SNANA FORMAT_MASK rules (snlc_sim.c:7451-7468, :30986-30994)."""
    m = int(simulation.output.format_mask)
    f = parse_format_mask(m)
    if f['vbose']:
        raise SNANAError(
            "FORMAT_MASK bit 1 (VBOSE) is no longer supported")
    if m > 0 and not f['text'] and not f['fits']:
        raise SNANAError(
            f"Invalid FORMAT_MASK={m}: must specify TEXT (2) or FITS (32)")
    if f['text'] and f['fits']:
        raise SNANAError(
            f"Cannot mix TEXT and FITS format; FORMAT_MASK={m}")


def write_yaml_summary(simulator, version_name, counts, n_spec, seconds,
                       output_dir=None):
    """SNANA WRFLAG_YAML_FILE (wr_SIMGEN_YAML_SUMMARY, snlc_sim.c:14736):
    `{GENVERSION}.YAML` job summary consumed by batch/pipeline scripts.
    SNANA writes it with a bare GENVERSION name (snlc_sim.c:30917), i.e. to
    the process working directory, so batch scripts find it locally;
    ``output_dir`` is how a caller owning its own run directory says
    otherwise. ``counts`` is `event_counts`; ``seconds`` is the time since
    setup ended."""
    n_event, n_lc, n_write = counts
    path = Path(output_dir or Path.cwd()) / f'{version_name}.YAML'
    with open(path, 'w') as fp:
        fp.write(f"SURVEY:          {simulator.survey_name}\n")
        fp.write(f"IDSURVEY:        {simulator.survey_id}\n")
        fp.write(f"NGENEV_TOT:      {n_event}\n")
        fp.write(f"NGENLC_TOT:      {n_lc}\n")
        fp.write(f"NGENLC_WRITE:    {n_write}\n")
        fp.write(f"NGENSPEC_WRITE:  {n_spec}\n")
        fp.write(f"CPU_MINUTES:     {seconds / 60.0:.2f}\n")
        fp.write(f"ABORT_IF_ZERO:   {n_write}\n")


def log_output_summary(counts, base_dir, version_name, timings):
    """SNANA-style OUTPUT_SUMMARY + AUXILIARY FILES + DONE block.

    ``counts`` is `event_counts`. ``timings`` holds seconds per phase of the
    run: ``init`` and ``generate`` from the caller that ran those phases,
    ``write`` from `save_snana_outputs`. A phase the caller did not time has
    no row.

    The time after init is SNANA's CPUTIME_PROCESS_ALL (simEnd,
    snlc_sim.c:530), the number its YAML and README carry as CPU_MINUTES;
    WALLTIME_MIN adds the init to it. Every rate names its denominator,
    because SNANA reports one per event count (:532-539): GENLC is per
    generated light curve, ACCEP per written one.
    """
    n_event, n_lc, n_write = counts
    p = n_write / n_lc if n_lc else 0.0
    eff_err = float(np.sqrt(p * (1.0 - p) / n_lc)) if n_lc else 0.0

    def minutes(seconds):
        return f"{seconds / 60.0:.3f}"

    def per_lc(seconds):
        return f"{n_lc / seconds if seconds else 0.0:.1f} SNe/sec (GENLC)"

    def per_accept(seconds):
        return f"{n_write / seconds if seconds else 0.0:.1f} SNe/sec (ACCEP)"

    rows = [
        ("NGENEV_TOT", n_event),
        ("NGENLC_TOT", n_lc),
        ("NGENLC_WRITE", n_write, "pass trigger+cuts"),
        ("NREJECT", n_lc - n_write, "see selection breakdown above"),
        ("EFF(SEARCH+CUTS)", f"{p:.4f} +- {eff_err:.4f}"),
    ]
    if 'generate' in timings:
        rows.append(("GENTIME_MIN", minutes(timings['generate']),
                     f"{per_lc(timings['generate'])}, "
                     f"{per_accept(timings['generate'])}"))
    rows.append(("WRITETIME_MIN", minutes(timings['write']),
                 per_accept(timings['write'])))
    if 'generate' in timings:
        process = timings['generate'] + timings['write']
        rows.append(("CPUTIME_PROCESS_ALL", minutes(process),
                     f"generate + write, the YAML's CPU_MINUTES; "
                     f"{per_lc(process)}, {per_accept(process)}"))
    if {'init', 'generate'} <= timings.keys():
        total = timings['init'] + timings['generate'] + timings['write']
        rows.append(("WALLTIME_MIN", minutes(total),
                     f"init + generate + write; {per_lc(total)}, "
                     f"{per_accept(total)}"))
    slog.summary(logger, "OUTPUT_SUMMARY", rows)
    out = Path(base_dir) / version_name
    aux = sorted(f.name for f in out.iterdir()
                 if f.name.startswith(version_name)) if out.is_dir() else []
    if aux:
        logger.info("\n AUXILIARY FILES:")
        for f in aux:
            slog.line(logger, f"{out / f}")
    slog.done(logger, "snlc_sim")


def save_snana_outputs(simulator, results,
                       version_name, selection_result, spectra_cache=None,
                       output_dir=None, timings=None, combined=False):
    """Write every SNANA output for a finished run: the writer's dataset
    plus the resource-dependent diagnostics (spectra, DCR, VERIFY_SED_TRUE,
    FILTERS, the YAML summary).

    ``spectra_cache`` is the pre-selection spectra cache returned by
    `run_simulation` (populated on the SNRSUM SPECID path); when present
    the writer reuses it instead of regenerating spectra.

    ``timings`` is an optional dict of seconds per phase the caller has
    already run (``init``, ``generate``); this adds ``write`` and the
    OUTPUT_SUMMARY reports each phase it holds.
    """
    simulation = simulator.config

    if timings is None:
        timings = {}
    start_time = time.time()
    fmt = parse_format_mask(simulation.output.format_mask)
    key = run_key(simulation)

    columnar_data = _build_columnar_data(simulator, results, selection_result)
    write_timing = {'columnar_data': time.time() - start_time}

    n_passing = selection_result['n_passed']
    n_sne = n_passing + selection_result['n_failed']

    if n_passing == 0:
        # SNANA aborts only when NEPOCH took every event (simEnd, snlc_sim.c:566).
        codes = selection_result['generated']['rejection_codes']
        if codes.size and np.all(codes == 2):   # 2 = NEPOCH
            raise SNANAError(
                f"Every generated event fails NEPOCH>="
                f"{simulation.cuts.nepoch_min}. "
                f"GENRANGE_PEAKMJD and SIMLIB probably don't overlap.",
                function_name="save_snana_outputs")
        logger.warning("No supernovae passed selection cuts. Skipping SNANA output files.")
        return {
            'n_passed': 0,
            'n_failed': int(n_sne)
        }

    spectra_by_sn = {}
    cfg = simulation
    if simulator.makes_spectra:
        if spectra_cache is not None:
            passing = np.asarray(selection_result['passing_mask'])
            keep_snids = set(np.asarray(results['parameters']['cid'])[passing].tolist())
            spectra_by_sn = {sid: spec for sid, spec in spectra_cache.items()
                             if sid in keep_snids}
        else:
            spectra_by_sn, snrsum_rest_v = compute_spectra_for_mask(
                simulator, results,
                np.asarray(selection_result['passing_mask']), key=key)
            results['parameters']['snrsum_rest_v'] = np.asarray(snrsum_rest_v)

    # SNANA writes SIM_SYNMAG for every simulated spectrum, not just the
    # SED-true reduced format (sntools_dataformat_text.c:1269).
    if spectra_by_sn:
        compute_sed_true_synmags(spectra_by_sn, simulator.photometry,
                                 simulator.calibration)

    if cfg.instrument.atmos_optmask > 0:
        compute_atmosphere_for_mask(
            simulator, simulator.atmosphere_static,
            results, selection_result, columnar_data, spectra_by_sn, key=key)
        # written only when asked for in their own right (snlc_sim.c:30970)
        if not (cfg.spectra.spectrograph_optmask & 128):
            spectra_by_sn = {}

    if fmt['nospec']:
        spectra_by_sn = {}

    write_timing['spectra_and_dcr'] = (
        time.time() - start_time - write_timing['columnar_data'])
    base_dir = output_dir or os.getcwd()
    phases, header_cols, phot_cols = write_snana_outputs(
        base_dir,
        version_name,
        columnar_data,
        simulation.run._input_file,
        simulation,
        simulator,
        simulator.filter_names,
        model=simulator.model,
        full_results=results,
        selection_result=selection_result,
        spectra_by_sn=spectra_by_sn,
        to_written_cid=to_written_cid(simulation, results['parameters']['cid'], fmt),
        makes_spectra=simulator.makes_spectra,
        seconds_since_init=time.time() - simulator.time_end_init,
        combined=combined,
    )
    write_timing.update(phases)

    # independent of FORMAT_MASK (snlc_sim.c:11295-11298)
    if (spectra_by_sn
            and simulation.effective_spectrograph_optmask & 128
            and simulation.spectra.verify_sed_true):
        write_verify_sed_true(simulator, base_dir, version_name,
                              spectra_by_sn, results)
    if fmt['filters']:
        write_filter_transmission_files(simulator, base_dir, version_name)

    timings['write'] = time.time() - start_time
    counts = event_counts(selection_result)
    if simulation.output.wrflag_yaml_file:
        n_spec = sum(len(v) for v in spectra_by_sn.values()) if spectra_by_sn else 0
        write_yaml_summary(simulator, version_name, counts, n_spec,
                           time.time() - simulator.time_end_init)
    log_output_summary(counts, base_dir, version_name, timings)

    return {
        'n_passed': int(n_passing),
        'n_failed': int(n_sne - n_passing),
        'timings': timings,
        'write_timing': write_timing,
        'header_columns': header_cols,
        'phot_columns': phot_cols,
    }


# Retain the photometry boundary to limit changes from XLA fusion in float32.
@jax.jit
def simulate_photometry(simulator, sed_params, mjds, filter_indices, mask,
                        band_weights, sed_factor):
    """Evaluate batch photometry with model surfaces supplied as arguments."""
    data = dict(simulator.photometry_data, band_window=simulator.band_window,
                latetime_extrap=True)

    def one_sn(sed_params, mjds, filter_indices, mask, band_weights, factor):
        return simulator.model.evaluate(sed_params, mjds, filter_indices, mask,
                                        band_weights, data, sed_factor=factor)

    factor_axis = 0 if simulator.gensmear is not None else None
    return jax.vmap(
        one_sn, in_axes=(0, 0, 0, 0, 0, factor_axis))(
            sed_params, mjds, filter_indices, mask, band_weights, sed_factor)



def prepare(snana_input=None, **settings) -> LightCurveSimulator | None:
    """Prepare a simulation from an input path, mapping or SimulationConfig.

    Keyword settings override a file or mapping before random input choices
    are resolved. A parsed configuration already contains those choices and
    accepts no further overrides. Setup-only modes return None after running.
    """
    return _prepare(snana_input, settings, allow_setup_modes=True)


def _prepare(snana_input, settings, *, allow_setup_modes):
    if isinstance(snana_input, SimulationConfig):
        if settings:
            raise SNANAError(
                'Apply simulation settings to the input file or mapping before '
                'building a SimulationConfig; random input choices have already been resolved.')
        simulation_config = snana_input
    else:
        from .inputs import parse_input
        simulation_config = parse_input(snana_input, settings)
    config = simulation_config.numerical
    simulation_config = copy.deepcopy(simulation_config)
    require_precision(simulation_config.run.double_precision)
    validate_format_mask(simulation_config)
    out = simulation_config.output
    dump_variables = (out.simgen_dump or out.simgen_dumpall) and not out.simgen_dumpall_columns
    setup_only = (simulation_config.survey.simlib_dump >= 0
                  or simulation_config.run.dashboard or simulation_config.run.init_only
                  or dump_variables)
    if setup_only and not allow_setup_modes:
        raise SNANAError(
            'The input requested a setup-only mode. Use prepare for setup-only '
            'runs or remove that option to simulate.')

    if simulation_config.survey.simlib_dump >= 0:
        for path in run_simlib_dump(simulation_config):
            slog.line(logger, f"wrote {path}")
        slog.line(logger, "SIMLIB_DUMP: done")
        return None

    simulator = build_simulation(simulation_config, config.wavelength_grid_points)

    if simulation_config.run.dashboard:
        run_dashboard(simulation_config, simulator)
        return None

    if simulation_config.run.init_only:
        slog.line(logger, "INIT_ONLY: stopping after initialization")
        return None

    # SIMGEN_DUMP: 0 lists the dump variables and stops (snlc_sim.c:14860-14862)
    if dump_variables:
        print_simgen_dump_variables(simulation_config, simulator)
        return None

    return simulator


def simulate(inputs=None, **settings):
    """Run a survey simulation and return its arrays without writing data files.

    ``inputs`` is an SNANA or YAML input path, a Python settings dictionary,
    or a :class:`~jax_snlc_sim.simulation.SimulationConfig` built with
    :func:`~jax_snlc_sim.inputs.build_config`. It supplies the survey, model,
    populations and random seed. Keyword settings override a file or mapping;
    without ``inputs``, they supply the complete configuration. A built
    ``SimulationConfig`` already has its random input choices resolved, so apply
    overrides before building it. Numerical controls such as ``batch_size``
    are ordinary keyword settings.

    Returns ``(results, selection, spectra)`` as in :func:`run_simulation`.
    ``results`` holds the generated parameters and photometry; ``selection``
    holds the survey selection and observation masks; ``spectra`` contains
    any requested spectra. Padded observations must be masked when using
    the arrays. Input modes that stop after setup do not generate events
    and raise an error here; use :func:`prepare` for those modes.

    Each call prepares the survey and runs it with the supplied seed. For
    repeated compiled realisations, use :func:`build_sbi_pipeline`.
    """
    simulator = _prepare(inputs, settings, allow_setup_modes=False)
    return run_simulation(simulator)


def apply_selection(simulator, results):
    """Trigger, SEARCHEFF and CUTWIN for a batch, plus the passing events'
    columns, drawing off the run's root key. See
    `jax_snlc_sim.selection`."""
    simulation = simulator.config
    return apply_selection_effects(
        simulator, simulation.instrument.zp_fluxcal,
        simulator.search_efficiency, simulator.cutwin,
        simulator.season_boundaries, simulator.photprob_prepared,
        batch_tables(simulator).catalogue, results, key=run_key(simulation))


@jax.jit(static_argnames=("n_sne", "adjust"))
def simulate_surveys(simulator, key, params, *, n_sne, adjust):
    batch_size = simulator.batch_size
    n_batches = -(-n_sne // batch_size)
    offsets = jnp.arange(n_batches, dtype=jnp.int32) * batch_size
    n_events = jnp.minimum(n_sne - offsets, batch_size)
    chunks = None
    if params is not None:
        params = {**params, "sn_global_idx": jnp.arange(n_sne, dtype=jnp.int32)}
        padded = pad_events(params, n_sne, n_batches * batch_size)
        chunks = jax.tree.map(
            lambda x: x.reshape(n_batches, batch_size, *x.shape[1:]),
            padded)

    def chunk(host_state, xs):
        start_index, n_events, params = xs
        results, selection, record, host_state = simulate_batch(
            simulator, key, start_index, n_events, host_state, params,
            adjust=adjust)
        return host_state, (results, selection, record)

    _, outputs = jax.lax.scan(
        chunk, initial_batch_state(simulator, n_sne),
        (offsets, n_events, chunks))
    results, selection, record = jax.tree.map(
        lambda x: x.reshape(n_batches * batch_size, *x.shape[2:])[:n_sne],
        outputs)
    # the chunks numbered their rows locally
    results.update(epoch_layout(
        n_sne, results['magnitudes'].shape[1],
        results['observation_schedule']['n_obs_per_sn']))
    return results, selection, record


def build_sbi_pipeline(simulator):
    """Prepare a compiled function generating survey realisations.

    Returns ``pipeline_fn(key, params=None, adjust=None)``, which produces
    ``(results, selection, record)`` for ``simulator.n_sne`` events, using
    the prepared simulator's batch size.

    ``key`` controls all random draws. Reusing a key reproduces a survey;
    changing it generates a new realisation without recompilation.
    ``params``, when supplied, replaces the complete per-event inputs,
    including placement, population and scatter draws. Its leading dimension
    sets the event count. A different event count or dictionary structure
    requires recompilation.

    ``adjust(key, params) -> params`` can change each batch's parameters
    after host assignment and population sampling, before photometry is
    generated. Host columns are available under their ``host_`` names.
    Pass survey-wide values through the adjustment function. Changing that
    function requires recompilation.

    ``results`` contains noise-free photometry, observing conditions and
    generated parameters. ``selection`` contains noisy photometry,
    uncertainties and selection masks. ``record`` contains generation checks
    evaluated by :func:`finish_batch`, which must be called after each survey
    outside compiled code. The function can be vectorised over random keys
    with ``jax.vmap``.
    """
    if simulator.specid_uses_snrsum_rest_v:
        raise SNANAError(
            "A SPECID map on SNRSUM_REST_V reads spectra, which are generated "
            "outside compiled code, so this input cannot run as one compiled "
            "function.", function_name="build_sbi_pipeline")
    def pipeline_fn(key, params=None, adjust=None):
        n_sne = (simulator.n_sne if params is None
                 else next(iter(params.values())).shape[0])
        return simulate_surveys(simulator, key, params,
                        n_sne=n_sne, adjust=adjust)

    return pipeline_fn
