"""Prepare and run survey simulations.

``prepare`` resolves a SimulationConfig and loads a LightCurveSimulator.
Numerical kernels take that simulator explicitly, sharing compiled programs
across compatible realisations. ``run_simulation`` owns the batch loop and
its host reuse, quota and output records; they never persist on the simulator.
"""

import copy
from collections.abc import Mapping
import logging
import time
from typing import Any, Dict, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core import logging as log
from jax_snlc_core.filters import filter_band
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import MAG_ZEROFLUX, SNANA_NULLFLOAT
from jax_snlc_core.utils import eval_genpoly, require_precision
from jax_snlc_sim.atmosphere import compute_atmosphere_for_mask
from jax_snlc_sim.builder import build_simulation
from jax_snlc_sim.diagnostics import run_dashboard, run_simlib_dump
from jax_snlc_sim.hostlib.association import (
    add_host_associations, apply_host_snpar_overrides, apply_random_coord_shift,
    galaxy_labels, nrepeat_host_leaders)
from jax_snlc_sim.hostlib.wronghost import DZTRUEMAX
from jax_snlc_sim.output.write import validate_format_mask
from jax_snlc_sim.output.dump import noise_dump_rows, print_simgen_dump_variables
from jax_snlc_sim.photometry import (check_crazy_flux, check_finite_flux, simulate_lightcurves)
from jax_snlc_sim.inputs import parse_input
from jax_snlc_sim.lensing import apply_lensdmu_smear
from jax_snlc_sim.simulation import run_key
from jax_snlc_sim.sampler import (PREFILTER_REASON_CODES, draw_dust,
                                  draw_grey_offset, draw_input_checks, draw_input_failures,
                                  draw_model, draw_placement, prepare_population_functions)
from jax_snlc_sim.selection import (apply_selection_effects, evaluate_selection,
                                    finalize_selection, generated_record, kept_mask)
from jax_snlc_sim.observations import SIMLIB_EPOCH_KEYS, epoch_layout, keep_events
from jax_snlc_sim.simulation import LightCurveSimulator, SimulationConfig
from jax_snlc_sim.population import (HOST_DUST_PARAMETERS, model_spellings,
                                     validate_genpdf_coverage, validate_population_overrides)
from jax_snlc_sim.spectra import (compute_sed_true_synmags, compute_spectra_for_mask,
                                  sed_true_epochs)
from jax_snlc_sim.cutwin import check_generic_cut_names
from jax_snlc_sim.simlib import epoch_phases, schedule_observations

logger = logging.getLogger(__name__)


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
    return jax.tree.map(pad, tree)


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


@jax.jit(static_argnames=("population_functions",))
def photometry_stage(simulator, key, start_index, n_events,
                     host_state, params, template_budget, *, overrides=None,
                     population_functions=()):
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

    Returns ``(results, record, host_state)``: ``results`` at the batch shape,
    ``record`` what the batch reports beside them for :func:`validate_batch`
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
            template_budget, key=key, overrides=overrides,
            population_functions=population_functions)}

    def with_dust(parameters, host_properties):
        return {**parameters, **draw_dust(
            simulator, parameters, host_properties, key=key,
            overrides=overrides, population_functions=population_functions)}

    if model_first:
        batch_parameters = with_model(batch_parameters, None)
    if dust_first:
        batch_parameters = with_dust(batch_parameters, None)

    field_indices = jnp.asarray(batch_parameters['field_idx'], dtype=jnp.int32)
    batch_obs_schedule = schedule_observations(
        simulator, tables.schedule, field_indices,
        batch_parameters['peakmjd'], batch_parameters['sim_redshift_helio'])
    # NEPOCH < CUTWIN_NEPOCH[0] rejection, whatever APPLY_CUTWIN_OPT (snlc_sim.c:296)
    batch_parameters['prefilter_pass'] = (
        batch_parameters['prefilter_pass']
        & (batch_obs_schedule['n_obs_per_sn'] >= simulator.cuts.nepoch_min)
        & jnp.take(jnp.asarray(simulator.field_data['keep']), field_indices))

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
            simulator, batch_parameters, host_properties, overrides=overrides,
            population_functions=population_functions)
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
    selection = evaluate_selection(
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


@jax.jit(static_argnames=("population_functions",))
def simulate_batch(simulator, key, start_index, n_events, host_state,
                   params=None, template_budget=None, *, overrides=None,
                   population_functions=()):
    """Generate and select a batch, sharing one program across realisations."""
    results, record, host_state = photometry_stage(
        simulator, key, start_index, n_events, host_state, params,
        template_budget, overrides=overrides, population_functions=population_functions)
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
    validate_batch(simulator, results, record, start_index, n_events)
    if simulator.hostlib is not None:
        results['parameters'].update(galaxy_labels(simulator, results['parameters']))
    return results, selection, record.get('spectra'), host_state


def validate_batch(simulator, results, record, start_index, n_events):
    """What SNANA aborts on, read from finished results and their record: a
    population-draw input it refuses, an SN variable outside the HOSTLIB
    weight map, a wrong host with no table row nearby, no HOSTLIB galaxy
    to draw within DZTOL, a HOSTLIB_GALID_FORCE host outside
    DZTOL, a USEONCE catalogue run out, an epoch served by multiple
    flux-error maps or REDCOV entries, a non-finite or an absurd flux.
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

    # Map and REDCOV matching apply only to generated epochs (snlc_sim.c:28035-28090).
    if simulator.fluxerr_enabled:
        generated = ((real & np.asarray(params['prefilter_pass']))[:, None]
                     & np.asarray(results['filter_z_valid']))
        field_idx = np.asarray(params['field_idx'])[:, None]
        bands = np.asarray(schedule['filter_indices'])
        for name, counts in (('FLUXERRMODEL maps', simulator.fluxerr_map_counts),
                             ('REDCOV entries', simulator.fluxerr_redcov_counts)):
            matches = counts[field_idx, np.maximum(bands, 0)]
            bad = generated & (bands >= 0) & (matches > 1)
            if bad.any():
                event, epoch = np.argwhere(bad)[0]
                field = simulator.fields[field_idx[event, 0]].field_name
                band = filter_band(simulator.filter_names[bands[event, epoch]])
                raise SNANAError(f"Invalid NMATCH={matches[event, epoch]} {name} for "
                                 f"BAND={band!r} and FIELD={field!r}.",
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
        log.line(logger, f"Batch {batch_idx + 1}/{n_batches}: "
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
                simulator.config, batch_results, sel_batch, start_index))
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
            'magnitudes', all_magnitudes, MAG_ZEROFLUX),
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
        log.line(logger, f"NGEN_LC batch {it}: generated {len(passing)}, "
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
    the batches, final selection, spectra and atmospheric effects, on the
    last batch's record."""
    if timings:
        timings[-1]['finalize'] = (timings[-1].get('finalize', 0.0)
                                   + time.perf_counter() - start)


def run_simulation(simulator, timings=None,
                   override_parameters: Optional[Dict] = None):
    """Run the complete vectorised simulation, one batch at a time.

    Each batch is sampled, scheduled, simulated and selected in turn, so
    device memory is bounded by ``batch_size``; the per-batch selections
    are merged and finalised once over the accumulated events.

    Returns ``(results, selection, spectra_by_sn)`` with completed spectra
    for the selected events and atmospheric values in ``results`` when enabled.
    ``override_parameters``
    replaces the sampled parameter dictionary, for replay and debugging.
    ``timings``, a list, receives one record per batch (see
    :func:`generate_events`), the last carrying the closing merge.
    """
    simulation = simulator.config

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
    if not selection["n_passed"]:
        _closing_phase(timings, start)
        return results, selection, {}

    spectra_by_sn = {}
    if simulator.makes_spectra:
        if spectra_cache is not None:
            passing = np.asarray(selection['passing_mask'])
            keep_snids = set(np.asarray(results['parameters']['cid'])[passing].tolist())
            spectra_by_sn = {sid: spec for sid, spec in spectra_cache.items()
                             if sid in keep_snids}
        else:
            spectra_by_sn, snrsum_rest_v = compute_spectra_for_mask(
                simulator, results,
                np.asarray(selection['passing_mask']), key=key)
            results['parameters']['snrsum_rest_v'] = np.asarray(snrsum_rest_v)

    # SNANA writes SIM_SYNMAG for every simulated spectrum, not just the
    # SED-true reduced format (sntools_dataformat_text.c:1269).
    if spectra_by_sn:
        compute_sed_true_synmags(spectra_by_sn, simulator.photometry,
                                 simulator.calibration)

    if simulation.instrument.atmos_optmask > 0:
        results['atmosphere'] = compute_atmosphere_for_mask(
            simulator, results, selection, spectra_by_sn, key=key)

    _closing_phase(timings, start)
    return results, selection, spectra_by_sn


def prepare(snana_input=None, **settings) -> LightCurveSimulator | None:
    """Prepare a simulation from an input path, mapping or SimulationConfig.

    Keyword settings override a file or mapping before random input choices
    are resolved. A parsed configuration already contains those choices and
    accepts no further overrides. Setup-only modes return None after running.
    """
    return _prepare(snana_input, settings, allow_setup_modes=True)


def _prepare(snana_input, settings, *, allow_setup_modes, host_columns=()):
    if isinstance(snana_input, SimulationConfig):
        if settings:
            raise SNANAError(
                'Apply simulation settings to the input file or mapping before '
                'building a SimulationConfig; random input choices have already been resolved.')
        simulation_config = snana_input
    else:
        simulation_config = parse_input(snana_input, settings)
    config = simulation_config.numerical
    simulation_config = copy.deepcopy(simulation_config)
    simulation_config.host.storepar = list(dict.fromkeys(
        [*simulation_config.host.storepar, *host_columns]))
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
            log.line(logger, f"wrote {path}")
        log.line(logger, "SIMLIB_DUMP: done")
        return None

    simulator = build_simulation(simulation_config, config.wavelength_grid_points)

    if simulation_config.run.dashboard:
        run_dashboard(simulation_config, simulator)
        return None

    if simulation_config.run.init_only:
        log.line(logger, "INIT_ONLY: stopping after initialization")
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


@jax.jit(static_argnames=("n_sne", "population_functions"))
def simulate_surveys(simulator, key, events, overrides, *, n_sne, population_functions):
    batch_size = simulator.batch_size
    n_batches = -(-n_sne // batch_size)
    offsets = jnp.arange(n_batches, dtype=jnp.int32) * batch_size
    n_events = jnp.minimum(n_sne - offsets, batch_size)
    settings = {name: value for name, value in overrides.items() if isinstance(value, Mapping)}
    samples = {name: value for name, value in overrides.items() if not isinstance(value, Mapping)}
    if events is not None:
        events = {**events, "sn_global_idx": jnp.arange(n_sne, dtype=jnp.int32)}
    chunks = jax.tree.map(
        lambda x: x.reshape(n_batches, batch_size, *x.shape[1:]),
        pad_events((events, samples), n_sne, n_batches * batch_size))

    def chunk(host_state, xs):
        start_index, n_events, (events, samples) = xs
        results, selection, _, host_state = simulate_batch(
            simulator, key, start_index, n_events, host_state, events,
            overrides={**settings, **samples}, population_functions=population_functions)
        return host_state, (results, selection)

    _, outputs = jax.lax.scan(
        chunk, initial_batch_state(simulator, n_sne), (offsets, n_events, chunks))
    results, selection = jax.tree.map(
        lambda x: x.reshape(n_batches * batch_size, *x.shape[2:])[:n_sne],
        outputs)
    # the chunks numbered their rows locally
    results.update(epoch_layout(
        n_sne, results['magnitudes'].shape[1],
        results['observation_schedule']['n_obs_per_sn']))
    return results, selection


def build_sbi_pipeline(inputs=None, *, conditional_settings=None, samplers=None,
                       relations=None, dependencies=None, **settings):
    """Build ``pipeline(key, params=None, *, events=None)`` for SBI.

    ``inputs`` accepts the same input path, settings mapping or SimulationConfig
    as :func:`simulate`. Keyword settings apply during setup, once when building
    the pipeline. An already-loaded LightCurveSimulator can also be supplied,
    without setup settings.

    ``params`` maps population names to numerical distribution settings or
    exact arrays of ``simulator.n_sne`` values. Omitted populations use their
    prepared distributions. Model aliases such as THETA and SALT2c are accepted.
    ``conditional_settings`` functions return distribution fields; ``samplers``
    return custom draws; ``relations`` return deterministic population values
    or GENMAG_OFF_GLOBAL. ``dependencies`` lists positional input quantities for
    each function (HOST.<column>, REDSHIFT_CMB, REDSHIFT_HELIO, or earlier draws).
    Coefficients come from that target's ``params`` mapping as keyword arguments.
    Only samplers receive an event key, before their dependencies. The simulator
    batches the functions and computes dependent quantities afterwards.

    ``events`` replaces complete per-event inputs, including placement and
    scatter, and sets the event count. It cannot be combined with ``params``
    or population functions. Shapes, dictionary keys and population functions select
    the compiled program; numerical values can vary without recompilation.

    Returns ``(results, selection)`` as JAX arrays, without host-side
    runtime validation. HOSTLIB references are catalogue row indices.
    """
    dependencies = dependencies or {}
    host_columns = tuple(dict.fromkeys(
        dep.removeprefix("HOST.") for deps in dependencies.values()
        for dep in deps if dep.startswith("HOST.")))
    if isinstance(inputs, LightCurveSimulator):
        if settings:
            raise SNANAError("Apply setup settings before loading the simulator",
                             function_name="build_sbi_pipeline")
        simulator = inputs
    else:
        simulator = _prepare(inputs, settings, allow_setup_modes=False,
                             host_columns=host_columns)
    if simulator.specid_uses_snrsum_rest_v:
        raise SNANAError("SBI cannot compile a SPECID map requiring host-side spectra",
                         function_name="build_sbi_pipeline")
    validate_genpdf_coverage(simulator)
    populations = simulator.model_settings.populations
    names = {**model_spellings(simulator.model),
             **{alias.lower(): p.key for p in HOST_DUST_PARAMETERS
                for alias in (p.key, *p.input_names)}}

    names["genmag_off_global"] = "GENMAG_OFF_GLOBAL"
    available = {*populations, "GENMAG_OFF_GLOBAL"}

    def resolve_names(parameters):
        resolved = {}
        for given, value in (parameters or {}).items():
            name = names.get(given.lower())
            if name not in available:
                raise SNANAError(f"Unknown or unconfigured population {given!r}; "
                                 f"choose from {sorted(available)}",
                                 function_name="build_sbi_pipeline")
            if name in resolved:
                raise SNANAError(f"Parameter {name} was supplied more than once",
                                 function_name="build_sbi_pipeline")
            resolved[name] = value
        return resolved

    population_functions = prepare_population_functions(
        simulator, names, resolve_names(conditional_settings), resolve_names(samplers),
        resolve_names(relations), resolve_names(dependencies))
    function_kinds = {name: kind for name, kind, _, _ in population_functions}

    def pipeline_fn(key, params=None, *, events=None):
        if events is not None and (params or function_kinds):
            raise SNANAError("Supply complete events or partial params/population functions, not both",
                             function_name="build_sbi_pipeline")
        n_sne = (simulator.n_sne if events is None
                 else next(iter(events.values())).shape[0])
        overrides = resolve_names(params)
        for name, value in overrides.items():
            if isinstance(value, Mapping):
                overrides[name] = dict(value)
            else:
                value = jnp.asarray(value)
                if value.shape != (n_sne,):
                    raise SNANAError(f"{name} needs one sample per event: expected {(n_sne,)}, "
                                     f"got {value.shape}",
                                     function_name="build_sbi_pipeline")
                overrides[name] = value
        for name in function_kinds:
            overrides.setdefault(name, {})
        validate_population_overrides(simulator, overrides, function_kinds)
        return simulate_surveys(simulator, key, events, overrides,
                                n_sne=n_sne, population_functions=population_functions)

    return pipeline_fn
