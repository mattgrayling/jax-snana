"""Load a survey and prepare one LightCurveSimulator.

The builder owns file reads and numerical preparation. It returns a complete
simulator containing its configuration snapshot, model and survey tables.
Random keys, batch counters and host reuse records belong to each run.
"""

import logging
import os
import time
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core.io import is_ignorefile
from jax_snlc_core.io.docana import check_file_docana
from jax_snlc_core.io.fits_path import open_text
from jax_snlc_core.io.kcor_reader import KcorReader
from jax_snlc_core.physics.constants import HC_ERG_A, LIGHT_KM
from jax_snlc_core.io.sfd_reader import SFDDustMap
from jax_snlc_core.models import (MODEL_REGISTRY, SimulatedModel, build_model,
                                  metadata_files, resolve_model_type)
from jax_snlc_core.photometry.filter_system import filter_band, band_window_width
from jax_snlc_core.photometry.resources import (
    PhotometryResources, load_photometry_resources, read_calib_shifts,
    resolve_calib_shifts,
)
from jax_snlc_core.physics.cosmology import (
    _cmb_vdotn, _eq_to_gal, cos_sep_cmb_apex_galactic, cosmo_params,
    distance_modulus, distance_modulus_to_redshift,
)
from jax_snlc_core.physics.dust_laws import load_dust_law
from jax_snlc_sim.rate import (
    calculate_expected_sn_count, dndz_density_on_grid,
)
from jax_snlc_core.utils import snana_log as slog
from jax_snlc_core.utils.interp import eval_genpoly
from jax_snlc_core.utils.error_handling import SNANAError
from jax_snlc_core.utils.precision import float_dtype
from jax_snlc_sim.config_parser import (parse_take_spectrum_entry, _parse_string_prescales,
                                        _resolve_input_path, prepare_spectrum_requests)
from jax_snlc_sim.genmag_smear import build_smear_model
from jax_snlc_sim.photometry.noise import per_filter_array, prepare_noise
from jax_snlc_sim.selection import prepare_search_efficiency, build_cutwin
from jax_snlc_sim.hostlib import (
    Hostlib, build_genzphot, build_host_selection, compute_wgtsum,
    essential_columns, hostlib_declared, read_hostlib, read_wgtmap,
    read_wgtmap_varnames, read_zphoteff_file,
)
from jax_snlc_sim.hostlib.hostlib_selection import (MINDAYSEP_SAMEGAL_MAX,
                                                    hostlib_snpar_columns,
                                                    resolve_igal_force,
                                                    wgtmap_snvar_names)
from jax_snlc_sim.hostlib.hostlib_reader import (
    DECADE_QUANTILES, discover_output_only_columns, zphot_columns,
    zphot_quantile_columns)
from jax_snlc_sim.hostlib.hostspec_reader import (
    PREFIX_SPECBASIS, PREFIX_SPECDATA, HostSpectra, hostlib_coefficients,
    hostlib_column_names, read_host_spec_table)
from jax_snlc_sim.hostlib.wgtsum_computer import WgtSumData
from jax_snlc_sim.hostlib.wronghost import read_wronghost_file
from jax_snlc_sim.population import (model_spellings, population_spellings,
                                     validate as validate_population)
from jax_snlc_sim.sampler import anisotropy_pars, simlib_job_start
from jax_snlc_sim.simulation import (HOST_DUST_PARAMETERS, HOSTLIB_MSKOPT_ZPHOT_QGAUSS,
                                     Instrument, LightCurveSimulator, NumericalSettings, field_default,
                                     neutral_rate_mods)
from jax_snlc_sim.zvariation import REDSHIFT
from jax_snlc_core.photometry.spectrograph import (
    build_ideal_spectrograph,
    read_spectrograph, synthetic_band_zptpsfsky,
)
from jax_snlc_core.photometry.fluxerr_model import (
    ALL_GROUP, MASK_APPLY_DATA, MASK_APPLY_TRUE, parse_redcov_string,
    read_fluxerr_model_file,
)
from jax_snlc_sim.survey.lensing_probmap import read_lensing_probmap
from jax_snlc_sim.survey.searcheff_map import (
    FLAG_NORMAL, check_applymask_searcheff, read_searcheff_maps,
    resolve_fieldlist_indices, resolve_searcheff_file, prepare_season_boundaries,
)
from jax_snlc_sim.survey.searcheff_pipeline import read_pipeline_eff_file
from jax_snlc_sim.survey.searcheff_photprob import (
    resolve_photprob_fieldlist_indices, prepare_photprob,
)
from jax_snlc_sim.survey.simlib_reader import (
    SimlibData, _PER_OBS_KEYS, _build_simlib_field, _compute_seasons,
    _rebuild_flat_structures, header_column, read_simlib,
    build_jax_field_cache, max_obs_per_event, scheduler_tables,
)
from jax_snlc_sim.survey.survey_def import parse_survey_def, resolve_survey

logger = logging.getLogger(__name__)


_SIMLIB_CACHE: Dict[str, SimlibData] = {}


def build_simulation(config, wavelength_grid_points: int) -> LightCurveSimulator:
    """Build the run's LightCurveSimulator from a parsed SimulationConfig, in one pass.

    Orchestration order (matches the dependency DAG):
        1. generation counts (rate integrals over the parsed config)
        2. simlib → saturation overrides → survey identity
        3. kcor → filter lookup
        4. cosmology + dust laws + model (shared build_model: UVLAM +
           latetime extrap + wavelength grid + resampling)
        5. SEARCHEFF SPEC/zHOST maps (their variables are HOSTLIB columns),
           HOSTLIB, FLUXERRMODEL, lensing, spectrograph, phase-correlated
           scatter
        6. filter transmissions + z-limits + AB ref + MW ext on the grid
        7. SEARCHEFF pipeline state and field lists (needs simlib fields +
           filter lookup)
    """
    hz_map = config.hz_map()
    n_simulated, ngen_lc, frac_pec1a, season_count = resolve_counts(
        config, hz_map)

    simlib = _load_simlib(config)
    simlib_windows = _simlib_redshift_windows(simlib, config.rates.redshift_range)
    survey_name, survey_id = resolve_survey(
        simlib.survey_name, config.survey.survey_override)
    kcor = _load_kcor(config)
    filter_names, name_to_index = _build_filter_lookup(kcor)

    dust_law = config.mw_dust.build_law()
    rv_mw = config.mw_dust.rv
    spec_calibration = _load_spectrograph(config, kcor)
    (sed_model, model_type, genmodel_path, wavelength_grid,
     model_wave_range, filter_center_range) = _create_model(
        config, kcor, wavelength_grid_points, n_simulated, frac_pec1a,
        _working_wavelength_range(config, kcor, spec_calibration))
    names, aliases, _ = config.model.population_declaration()
    sed_model._population = tuple((name, tuple(aliases.get(name, ())))
                                  for name in names)
    host_dust_law = sed_model.build_host_dust_law()
    if config.effective_spectrograph_optmask & 128:
        spec_calibration = _build_sed_true_spectrograph(
            config, kcor, model_wave_range)
    if sed_model.forced_ngen is not None:
        n_simulated = sed_model.forced_ngen
        ngen_lc = 0
        logger.info(f"{type(sed_model).__name__} forces NGEN to {n_simulated}")

    spec_zhost = _read_searcheff_specid_zhost(config, model_spellings(sed_model))
    hostlib = _load_hostlib(config, simlib, sed_model, spec_zhost, hz_map,
                            frac_pec1a)
    _validate_populations(config, sed_model, hostlib)
    npe_pixel_saturate, photflag_saturate = \
        _resolve_saturation(config.instrument, simlib)
    fluxerr = _load_fluxerr_model(config, simlib, kcor, hostlib)
    lensing_probmap, lensing_inv_cdf = _load_lensing(config)
    gensmear = _init_gensmear(config, genmodel_path, model_wave_range)
    if config.survey.use_simlib_spectrograph:
        _append_spectrograph_photometry(
            simlib, spec_calibration, kcor, config)
    _validate_simlib_observations(simlib)

    phase_range, filter_fields, photometry = _build_grids_and_filters(
        config, kcor, sed_model, wavelength_grid, filter_center_range,
        filter_names, name_to_index,
        dust_law, rv_mw,
    )

    searcheff_fields = _load_searcheff(config, simlib, name_to_index,
                                       spec_zhost)

    field_mwebv = np.array(
        [f.mwebv for f in simlib.fields], dtype=np.float32)
    sfd_map, field_sfd_ebv = _load_field_sfd(config, simlib, field_mwebv)
    cdf_z, cdf_vals, cdf_vals_pec1a = _build_redshift_cdf(config, hz_map)
    murewgt = _build_murewgt(config, hz_map)
    mw_law_obs = config.mw_dust.build_law()
    mw_law_obs.prepare_wavelength_grid(
        np.asarray(photometry.common_wave_grid))
    mw_law_at_mean = config.mw_dust.build_law()
    mw_law_at_mean.prepare_wavelength_grid(
        np.asarray(photometry.filter_mean_wavelengths))
    # The host-galaxy magnitudes dim by a law of their own: GEN_SNHOST_GALMAG
    # hard-wires RVMW = 3.1 and colour law 94 whatever OPT_MWCOLORLAW and
    # RV_MWCOLORLAW say (sntools_host.c:9256, :9278).
    host_mw_law_at_mean = load_dust_law('ODONNELL94')
    host_mw_law_at_mean.prepare_wavelength_grid(
        np.asarray(photometry.filter_mean_wavelengths))

    registered_model = not jax.tree_util.all_leaves((sed_model,))
    simulator = LightCurveSimulator(
        config=config,
        rates=config.rates, cosmology=config.cosmology,
        model_settings=config.model, survey=config.survey,
        host_settings=config.host, mw_dust=config.mw_dust,
        instrument=config.instrument, lensing=config.lensing, cuts=config.cuts,
        magmonitor_snr=config.output.magmonitor_snr,
        field_data=jax.device_put(_prepare_field_data(simlib, sed_model)),
        field_start_index=simlib_job_start(config, len(simlib.fields)),
        n_simulated=n_simulated, ngen_lc=ngen_lc,
        season_count=season_count,
        _model=sed_model if registered_model else None,
        _static_model=None if registered_model else sed_model,
        wavelength_grid=wavelength_grid, phase_range=phase_range,
        model_wave_range=model_wave_range,
        filter_center_range=filter_center_range,
        gensmear_n_randoms=gensmear.n_randoms if gensmear else 0,
        gensmear=gensmear,
        survey_name=survey_name, survey_id=survey_id,
        site=parse_survey_def().get(survey_name),
        fields=simlib.fields,
        flat_field_index=simlib.flat_field_index,
        flat_obs_data=simlib.flat_obs_data,
        field_prescales=simlib.field_prescales,
        nea_psf_unit=simlib.nea_psf_unit,
        simlib_filter_names=simlib.filter_names,
        simlib_fluxerr_cor=simlib.fluxerr_cor,
        simlib_fluxerr_add=simlib.fluxerr_add,
        simlib_template_texpose=simlib.template_texpose_spectrograph,
        simlib_subsurvey_list=simlib.subsurvey_list,
        simlib_windows=simlib_windows,
        field_mwebv=jax.device_put(field_mwebv),
        kcor=kcor, photometry=photometry, calibration=spec_calibration,
        **filter_fields, **searcheff_fields,
        hostlib=hostlib,
        host_dust_law=host_dust_law,
        hz_map=hz_map,
        redshift_cdf_z=cdf_z, redshift_cdf_vals=cdf_vals,
        redshift_cdf_vals_pec1a=cdf_vals_pec1a,
        murewgt=murewgt,
        mw_law=dust_law, mw_law_obs=mw_law_obs, mw_law_at_mean=mw_law_at_mean,
        host_mw_law_at_mean=host_mw_law_at_mean,
        sfd_map=sfd_map, field_sfd_ebv=field_sfd_ebv,
        npe_pixel_saturate=npe_pixel_saturate,
        photflag_saturate=photflag_saturate,
        phasecor_cholesky_L=_init_phasecor(config),
        **fluxerr,
        lensing_probmap=lensing_probmap, lensing_inv_cdf=lensing_inv_cdf)
    return _prepare_kernels(simulator)


def resolve_counts(config, hz_map) -> Tuple[int, int, float, float]:
    """Resolve how many events to generate and the PEC1A share of them.

    SNANA supports three modes (snlc_sim.c:7146-9221): NGENTOT_LC (fixed
    total), NGEN_LC (generate until N pass cuts), or NGEN_SEASON (rate × time ×
    area). NGEN_SCALE multiplies whichever is in play. Exactly one of the three
    must be set. Returns ``(n_simulated, ngen_lc, frac_pec1a, season_count)``,
    with ``ngen_lc`` already NGEN_SCALE-scaled and ``season_count`` the rate
    model's events per season (SEASON_COUNT, snlc_sim.c:9178-9179).
    """
    # NGEN_SCALE_NON1A scales the count for NON1A/SIMSED models only
    # (snlc_sim.c:7124).
    scale = config.run.ngen_scale
    if MODEL_REGISTRY[config.model.model_type].uses_nonia_rates:
        scale *= config.run.ngen_scale_non1a
    ngentot_lc, ngen_lc, ngen_season = (
        config.run.ngentot_lc, config.run.ngen_lc, config.run.ngen_season)

    # prep_SIMLIB_DUMP forces NGENTOT_LC (snlc_sim.c:32558)
    if config.survey.simlib_dump >= 0:
        ngentot_lc, ngen_lc, ngen_season = 150000, 0, 0.0

    n_modes_set = (ngentot_lc > 0) + (ngen_lc > 0) + (ngen_season > 0)
    if n_modes_set > 1:
        raise SNANAError(
            "NGEN_LC, NGENTOT_LC and NGEN_SEASON are mutually exclusive. Set "
            "exactly one.",
            function_name="resolve_counts")
    if n_modes_set == 0 and not config.run.readme_dumpflag:
        raise SNANAError(
            "Required: one of NGEN_LC, NGENTOT_LC or NGEN_SEASON must be set.",
            function_name="resolve_counts")

    # (int)(NGEN_LC*NGEN_SCALE), snlc_sim.c:7146
    ngen_lc = int(ngen_lc * scale)

    def _expected(entries, rate_mods):
        c = config.cosmology
        return calculate_expected_sn_count(
            config.dndz_params(entries), config.rates.solid_angle,
            config.rates.redshift_range, config.rates.peakmjd_range,
            config.rates.peakmjd_exclude,
            c.omega_matter, c.omega_lambda, c.h0, c.w0, c.wa,
            hz_map, rate_mods)

    # PEC1Asum: SNcount_model over RATEPAR_PEC1A (snlc_sim.c:9006)
    sn_sum = _expected(None, config.rates.dndz_rate_mods)
    pec1a_sum = (_expected(config.rates.dndz_entries_pec1a,
                           neutral_rate_mods())
                 if config.rates.dndz_entries_pec1a else 0.0)
    total = sn_sum + pec1a_sum
    frac_pec1a = pec1a_sum / total if total else 0.0

    if ngen_season > 0:
        n_simulated = int(ngen_season * total * scale)
        logger.info(
            f"NGEN_SEASON={ngen_season:.3f} × rate={total:.2f}/season × "
            f"NGEN_SCALE={scale:.3f} → NGENTOT_LC={n_simulated}")
    elif ngen_lc > 0:
        n_simulated = ngen_lc
    else:
        n_simulated = int(ngentot_lc * scale)
    return n_simulated, ngen_lc, frac_pec1a, total


def _simlib_redshift_windows(simlib, z_range):
    """The entries grouped by distinct CUTWIN_REDSHIFT window for the entry
    draw, or None when no entry has one. The windows must between them cover
    the generated range, or SNANA exhausts the library.
    """
    z_lo, z_hi = z_range
    windows = np.array([f.cutwin_redshift for f in simlib.fields])
    if np.isnan(windows).all():
        return None
    windows = np.where(np.isnan(windows), [[-np.inf, np.inf]], windows)
    order = np.lexsort((windows[:, 1], windows[:, 0]))
    distinct, counts = np.unique(windows[order], axis=0, return_counts=True)
    covered = z_lo
    for lo, hi in distinct:
        if lo <= covered:
            covered = max(covered, hi)
    if covered < z_hi:
        raise SNANAError(f"SIMLIB CUTWIN_REDSHIFT windows do not cover "
                         f"GENRANGE_REDSHIFT {z_range}", function_name="build_simulation")
    starts = np.concatenate(([0], np.cumsum(counts)))
    return {'lo': distinct[:, 0], 'hi': distinct[:, 1], 'start': starts,
            'count': counts, 'order': order}


def _load_simlib(config) -> SimlibData:
    """Load the SIMLIB reader (with per-process cache)."""
    simlib_file = _resolve_input_path(config.survey.simlib_file, config.run)
    s = config.survey
    spellings = population_spellings(*config.model.population_declaration()[:2])
    # EXPOSURE_TIME per-band ZPT shift merges additively with FUDGESHIFT_ZPT.
    exp_zpt, exp_sky, exp_read = _exposure_scales(config)
    zpt_map = {k: v for k, v in config.instrument.fudgeshift_zpt.items() if v != 0.0}
    for b, v in exp_zpt.items():
        zpt_map[b] = zpt_map.get(b, 0.0) + v
    zpt_shift = tuple(sorted(zpt_map.items()))
    exp_noise = (tuple(sorted(exp_sky.items())), tuple(sorted(exp_read.items())))
    noise_scales = (
        config.instrument.fudgescale_psf, config.instrument.fudgescale_noise_sky,
        config.instrument.fudgescale_noise_read,
        config.instrument.fudgescale_noise_template,
    )
    forceval_psf = config.survey.forceval_psf if config.survey.forceval_psf > 0.001 else 0.0
    ignore_template_noise = bool(s.simlib_mskopt & 16)
    cache_key = (
        simlib_file, s.simlib_minseason, s.simlib_minobs, s.simlib_maxobs, s.simlib_idlock, s.simlib_idstart, s.simlib_idskip,
        s.simlib_prescale, s.simlib_mjd_shift, s.genrange_mjd_exclude, s.simlib_fieldlist,
        s.genrange_ra, s.genrange_decl, s.genrange_b, zpt_shift, noise_scales,
        exp_noise,
        config.survey.use_simlib_spectrograph, config.spectra.spectrograph_scale_texpose,
        forceval_psf, ignore_template_noise, config.run.require_docana,
        config.rates.redshift_range, config.rates.peakmjd_range,
        s.use_simlib_redshift, s.use_simlib_peakmjd, s.use_simlib_distance,
        s.use_simlib_salt2,
        tuple(sorted(config.model._input_genranges.items())) if s.use_simlib_salt2 else None,
        cosmo_params(config.cosmology) if s.use_simlib_distance else None,
        tuple(sorted(spellings.items())),
    )
    slog.banner(logger, "Read SIMLIB")
    if cache_key in _SIMLIB_CACHE:
        slog.line(logger, f"Using cached reader for {Path(simlib_file).name}")
        return _SIMLIB_CACHE[cache_key]
    slog.line(logger, f"SIMLIB file: {simlib_file}")
    simlib = read_simlib(simlib_file, require_docana=bool(config.run.require_docana))
    _spell_entry_headers(simlib, spellings)
    distance_z = _field_distance_redshift(config, simlib)
    _apply_simlib_filters(simlib, s, InputGenranges(
        redshift=config.rates.redshift_range, peakmjd=config.rates.peakmjd_range,
        population=config.model._input_genranges,
        distance_redshift=(np.full(len(simlib.fields), np.nan) if distance_z is None
                           else np.asarray(distance_z))))
    if forceval_psf:
        _apply_forceval_psf(simlib, forceval_psf)
    if zpt_shift:
        _apply_zpt_shift(simlib, dict(zpt_shift))
    _apply_noise_scales(simlib, *noise_scales)
    if exp_sky or exp_read:
        _apply_noise_scales_per_filter(simlib, exp_sky, exp_read)
    if ignore_template_noise:
        _apply_ignore_template_noise(simlib)
    if (forceval_psf or zpt_shift or exp_sky or exp_read
            or ignore_template_noise
            or any(v != 1.0 for v in noise_scales)):
        _rebuild_flat_structures(simlib)
    _SIMLIB_CACHE[cache_key] = simlib
    slog.line(logger, f"Loaded {len(simlib.fields)} SIMLIB fields")
    return simlib


def _append_spectrograph_photometry(simlib, calib, kcor, config) -> None:
    """Append synthetic-band observations at each SIMLIB SPECTROGRAPH epoch so they
    flow through the normal photometry pipeline (SIMLIB_addCadence_SPECTROGRAPH)."""

    if simlib.spectro_applied or not kcor.syn_filters:
        return
    scale, texpose_t = config.spectra.spectrograph_scale_texpose, simlib.template_texpose_spectrograph
    for name in kcor.syn_filters:  # register bands at the end of the shared order
        simlib.filter_mapping.setdefault(name, len(simlib.filter_names))
        if name not in simlib.filter_names:
            simlib.filter_names.append(name)

    derived = set(_PER_OBS_KEYS) | {'season_starts', 'season_ends', 'nobs'}
    for i, fld in enumerate(simlib.fields):
        if fld.spec_mjds.size == 0:
            continue
        rows = []
        for name, (lammin, lammax) in kcor.syn_filters.items():
            fidx = simlib.filter_mapping[name]
            for mjd, tex in zip(fld.spec_mjds, fld.spec_texposes):
                zpt, psf, sks, skt, nea = synthetic_band_zptpsfsky(
                    calib, lammin, lammax, tex * scale, texpose_t, fld.pixsize)
                rows.append((mjd, fidx, sks, psf, 0., 0., nea, zpt, 0., 1., 0., 1., skt, 0., zpt, 0))
        new = np.array(rows).T
        obs = {k: np.concatenate([getattr(fld, k), new[j]]).astype(getattr(fld, k).dtype)
               for j, k in enumerate(_PER_OBS_KEYS)}
        meta = {f.name: getattr(fld, f.name) for f in dc_fields(fld) if f.name not in derived}
        simlib.fields[i] = _build_simlib_field(**meta, **obs, nobs=len(obs['mjds']))

    simlib.spectro_applied = True
    _rebuild_flat_structures(simlib)


def _prune_short_seasons(field, min_season: float):
    """SIMLIB_MINSEASON: drop seasons shorter than `min_season` from the entry's
    season bookkeeping, leaving its epochs in place (remove_short_SIMLIB_SEASON,
    snlc_sim.c:21584). The entry stays in the library even when no season
    survives: SNANA leaves the LIBID in the cycle with an empty MJD range
    (:21649), and the scheduler rejects its events.
    """
    if min_season <= 0.0:
        return field
    starts, ends = field.season_starts, field.season_ends
    short = np.isfinite(starts) & ((ends - starts) < min_season)
    if short.any():
        field.season_starts = np.where(short, np.inf, starts)
        field.season_ends = np.where(short, -np.inf, ends)
    return field


def _apply_zpt_shift(simlib: SimlibData, shifts: Dict[str, float]) -> None:
    """FUDGESHIFT_ZPT: add per-filter shifts to every epoch's SIMLIB ZPT,
    search and template alike (snlc_sim.c get_SIMLIB_SCALES, :20835)."""
    per_idx = np.array([shifts.get(name, 0.0) for name in simlib.filter_names],
                       dtype=np.float32)
    for f in simlib.fields:
        d = per_idx[f.filter_indices]
        f.zptavgs = f.zptavgs + d
        # SNANA shifts TEMPLATE_ZPT unconditionally for fields carrying a
        # template block (fields without one keep their sentinel).
        if np.any(f.template_skysigs > 0.0) or np.any(f.template_zpts != 0.0):
            f.template_zpts = f.template_zpts + d


def _apply_noise_scales(simlib: SimlibData, psf: float, sky: float,
                        read: float, template: float) -> None:
    """FUDGESCALE_PSF / NOISE_SKY / NOISE_READ / NOISE_TEMPLATE: scale every
    epoch's SIMLIB noise terms (snlc_sim.c:20777-20783, :20827-20832).
    Template noise carries the product of the search scale and the template
    scale; NEA scales as psf^2."""
    factors = {
        'skysigs': sky, 'readnoises': read,
        'psf1s': psf, 'psf2s': psf, 'neas': psf * psf,
        'template_skysigs': sky * template,
        'template_readnoises': read * template,
    }
    for name, fac in factors.items():
        if fac == 1.0:
            continue
        for f in simlib.fields:
            setattr(f, name, getattr(f, name) * fac)


def _apply_ignore_template_noise(simlib: SimlibData) -> None:
    """SIMLIB_MSKOPT bit 16 (IGNORE_TEMPLATENOISE): SNANA's reader returns
    before storing any correlated template-noise key (snlc_sim.c:19288-19289),
    so no template variance ever reaches an error term. Zeroing the parsed
    template arrays reproduces that at the same boundary."""
    for f in simlib.fields:
        f.template_skysigs = np.zeros_like(f.template_skysigs)
        f.template_readnoises = np.zeros_like(f.template_readnoises)
        f.template_zpts = np.zeros_like(f.template_zpts)


def _validate_simlib_observations(simlib) -> None:
    """One construction-time validation pass over every parsed SIMLIB
    observation.

    Mirrors SNANA's reader sanity ranges (snlc_sim.c:20573-20580), the fatal
    per-epoch observing-condition bounds in gen_fluxNoise_calc
    (snlc_sim.c:27753-27764) and its template-ZPT requirement
    (snlc_sim.c:27808-27813). The noise kernels consume these arrays
    unguarded, so every observation is checked here, once.
    """
    flat = simlib.flat_obs_data
    if len(flat.get('mjds', ())) == 0:
        return
    starts = np.asarray(simlib.flat_field_index['obs_start_idx'])
    libids = np.asarray(simlib.flat_field_index['libids'])
    mjds = np.asarray(flat['mjds'])

    def _check(name, values, bad_mask, rule):
        idx = np.flatnonzero(bad_mask)
        if idx.size == 0:
            return
        i = int(idx[0])
        entry = int(np.searchsorted(starts, i, side='right') - 1)
        raise SNANAError(
            f"SIMLIB {name}={values[i]} (LIBID {libids[entry]}, "
            f"MJD {mjds[i]:.3f}) violates {rule}",
            function_name="_validate_simlib_observations",
        )

    zptavg = np.asarray(flat['zptavgs'])
    zpterr = np.asarray(flat['zpterrs'])
    skysig = np.asarray(flat['skysigs'])
    psf1 = np.asarray(flat['psf1s'])
    psf2 = np.asarray(flat['psf2s'])
    psf_ratio = np.asarray(flat['psf_ratios'])
    nea = np.asarray(flat['neas'])
    gain = np.asarray(flat['gains'])
    readnoise = np.asarray(flat['readnoises'])
    nexpose = np.asarray(flat['nexposes'])

    _check('ZPTAVG', zptavg,
           ~np.isfinite(zptavg) | (zptavg < 10.0) | (zptavg > 50.0),
           'ZPTAVG in [10, 50]')
    _check('ZPTERR', zpterr,
           ~np.isfinite(zpterr) | (zpterr < 0.0) | (zpterr > 5.0),
           'ZPTERR in [0, 5]')
    _check('SKYSIG', skysig,
           ~np.isfinite(skysig) | (skysig < 1.0e-4) | (skysig > 1.0e5),
           'SKYSIG in [1e-4, 1e5]')
    _check('PSFSIG1', psf1,
           ~np.isfinite(psf1) | (psf1 < 1.0e-4) | (psf1 > 30.0),
           'PSFSIG1 in [1e-4, 30] pixels')
    _check('PSFSIG2', psf2,
           ~np.isfinite(psf2) | (psf2 < 0.0) | (psf2 > 30.0),
           'PSFSIG2 in [0, 30] pixels')
    _check('PSFRATIO', psf_ratio,
           ~np.isfinite(psf_ratio) | (psf_ratio < 0.0) | (psf_ratio > 1.0),
           'PSFRATIO in [0, 1]')
    _check('NEA', nea, ~np.isfinite(nea) | (nea <= 0.0),
           'NEA > 0 pixel^2')
    _check('GAIN', gain, ~np.isfinite(gain) | (gain <= 0.0),
           'CCD gain > 0 (converts flux to photoelectrons)')
    _check('READNOISE', readnoise,
           ~np.isfinite(readnoise) | (readnoise < 0.0), 'READNOISE >= 0')
    _check('NEXPOSE', nexpose, ~np.isfinite(nexpose) | (nexpose < 1.0),
           'NEXPOSE >= 1 (saturation divides per-exposure flux by it)')

    t_sky = np.asarray(flat['template_skysigs'])
    t_read = np.asarray(flat['template_readnoises'])
    t_zpt = np.asarray(flat['template_zpts'])
    _check('TEMPLATE_SKYSIG', t_sky,
           ~np.isfinite(t_sky) | (t_sky < 0.0),
           'TEMPLATE_SKYSIG >= 0')
    _check('TEMPLATE_READNOISE', t_read,
           ~np.isfinite(t_read) | (t_read < 0.0),
           'TEMPLATE_CCDSIG >= 0')
    _check('TEMPLATE_ZPT', t_zpt,
           (t_sky > 0.0) & (~np.isfinite(t_zpt) | (t_zpt < 10.0)),
           'TEMPLATE_ZPT >= 10 where template noise is present')


def _exposure_scales(config):
    """EXPOSURE_TIME(_FILTER): per-band ZPT shift and sky/read noise scales from
    the exposure-time ratio, gated by EXPOSURE_TIME_MSKOPT bits 0/1/2
    (get_SIMLIB_SCALES, snlc_sim.c:7190, :22636). ratio -> SHIFT_ZPT =
    2.5*log10(ratio), SCALE_NOISE = sqrt(ratio)."""
    mskopt = int(config.survey.exposure_time_mskopt)
    ratios = {b: float(r) for b, r in config.survey.exposure_time.items()
              if float(r) != 1.0}
    zpt = {b: 2.5 * float(np.log10(r)) for b, r in ratios.items()} if mskopt & 1 else {}
    sky = {b: float(np.sqrt(r)) for b, r in ratios.items()} if mskopt & 2 else {}
    read = {b: float(np.sqrt(r)) for b, r in ratios.items()} if mskopt & 4 else {}
    return zpt, sky, read


def _apply_noise_scales_per_filter(simlib: SimlibData, sky_scale: Dict[str, float],
                                   read_scale: Dict[str, float]) -> None:
    """EXPOSURE_TIME per-band sky/read noise scaling (snlc_sim.c:20778, :20827-20832);
    template sky/read carry the same per-band scale."""
    sky = np.array([sky_scale.get(name, 1.0) for name in simlib.filter_names],
                   dtype=np.float32)
    read = np.array([read_scale.get(name, 1.0) for name in simlib.filter_names],
                    dtype=np.float32)
    for f in simlib.fields:
        si = sky[f.filter_indices]
        ri = read[f.filter_indices]
        f.skysigs = f.skysigs * si
        f.template_skysigs = f.template_skysigs * si
        f.readnoises = f.readnoises * ri
        f.template_readnoises = f.template_readnoises * ri


def _field_prescale(field_name: str, prescales) -> int:
    """Prescale for a field, or -1 when it is not listed (SNANA get_string_dict
    with OPT_DICT=1: a **partial** name match, first hit wins)."""
    if prescales is None:
        return 1
    for name, value in prescales.items():
        if name in field_name:
            return int(value)
    return -1


def _spell_entry_headers(simlib, spellings) -> None:
    """Key each entry header's population values by parameter name, as an
    entry may spell a parameter any way the input may."""
    for f in simlib.fields:
        f.header_values = {spellings.get(k.lower(), k): v
                           for k, v in f.header_values.items()}
        f.gengauss = {spellings.get(k.lower(), k): v
                      for k, v in f.gengauss.items()}


class InputGenranges(NamedTuple):
    """The input ranges a SIMLIB entry's own redshift, peak-MJD and
    population-parameter windows must overlap, and each entry's DISTANCE as
    a redshift (NaN where it has none or USE_SIMLIB_DISTANCE is off),
    aligned with the unfiltered entries."""
    redshift: Tuple[float, float]
    peakmjd: Tuple[float, float]
    population: Dict[str, Tuple[float, float]]
    distance_redshift: np.ndarray


def _clip_to_genrange(window, genrange):
    """check_SIMLIB_GENRANGE (snlc_sim.c:22288): ``window`` clipped onto
    ``genrange``, or None when they do not overlap. The comparisons are
    strict, so a single value on either edge fails."""
    lo, hi = window
    if lo >= genrange[1] or hi <= genrange[0]:
        return None
    return max(lo, genrange[0]), min(hi, genrange[1])


def _entry_redshift_window(f, s, distance_z):
    """The window set by the last of the entry's REDSHIFT, DISTANCE and
    GENRANGE_REDSHIFT keys that applies, a value as a zero-width window
    (parse_SIMLIB_GENRANGES, snlc_sim.c:22138-22165)."""
    window = (np.nan, np.nan)
    for key in f.redshift_keys:
        if key == 'REDSHIFT' and s.use_simlib_redshift:
            window = (float(f.header_values['REDSHIFT']),) * 2
        elif key == 'DISTANCE' and np.isfinite(distance_z):
            window = (float(distance_z),) * 2
        elif key == 'GENRANGE_REDSHIFT':
            window = f.genrange_redshift
    return window


def _entry_genranges_pass(f, s, genranges, distance_z) -> bool:
    """Whether the entry's redshift and peak-MJD windows overlap the input
    ranges, clipping the windows it draws from in place. The window checked
    is the one the draw uses: the last of the entry's REDSHIFT, DISTANCE
    (each under its USE_SIMLIB flag) and GENRANGE_REDSHIFT keys, a value
    as a zero-width window; a PEAKMJD value under
    USE_SIMLIB_PEAKMJD, else GENRANGE_PEAKMJD, which is read whatever the
    flag (parse_SIMLIB_GENRANGES, snlc_sim.c:22135-22185). SNANA skips an
    entry that fails and reads the next (:20152-20180)."""
    window = _entry_redshift_window(f, s, distance_z)
    if window[0] > 0.0:
        window = _clip_to_genrange(window, genranges.redshift)
        if window is None:
            return False
    f.genrange_redshift = window

    t_value = float(f.header_values.get('PEAKMJD', 'nan')) if s.use_simlib_peakmjd else np.nan
    if np.isfinite(t_value):
        if _clip_to_genrange((t_value, t_value), genranges.peakmjd) is None:
            return False
    elif 'PEAKMJD' in f.gengauss:
        lo, hi, sigma = f.gengauss['PEAKMJD']
        clipped = _clip_to_genrange((lo, hi), genranges.peakmjd)
        if clipped is None:
            return False
        f.gengauss = {**f.gengauss, 'PEAKMJD': (*clipped, sigma)}

    # USE_SIMLIB_SALT2: a population parameter the entry sets, as a value or
    # a GENRANGE, against the input's range for it (:20184-20205)
    if s.use_simlib_salt2:
        for name, genrange in genranges.population.items():
            value = float(f.header_values.get(name, 'nan'))
            if np.isfinite(value):
                if _clip_to_genrange((value, value), genrange) is None:
                    return False
            elif name in f.gengauss:
                lo, hi, sigma = f.gengauss[name]
                clipped = _clip_to_genrange((lo, hi), genrange)
                if clipped is None:
                    return False
                f.gengauss = {**f.gengauss, name: (*clipped, sigma)}
    return True


def _apply_simlib_filters(simlib: SimlibData, s, genranges: InputGenranges) -> None:
    """Mutate simlib.fields in-place per the SIMLIB_* knobs and the input
    ranges, and rebuild flat structures."""
    prescales = (None if str(s.simlib_fieldlist).upper() == "ALL"
                 else _parse_string_prescales(s.simlib_fieldlist))
    surviving = []
    for f, distance_z in zip(simlib.fields, genranges.distance_redshift):
        pruned = _prune_short_seasons(f, s.simlib_minseason)
        if pruned.nobs < s.simlib_minobs or pruned.nobs > s.simlib_maxobs:
            continue
        if s.simlib_idstart > 0 and pruned.libid < s.simlib_idstart:
            continue
        if s.simlib_idlock > 0 and pruned.libid != s.simlib_idlock:
            continue
        if pruned.libid in s.simlib_idskip:
            continue
        if s.simlib_prescale > 1 and pruned.libid % s.simlib_prescale != 0:
            continue
        if not (s.genrange_ra[0] <= pruned.ra <= s.genrange_ra[1]):
            continue
        if not (s.genrange_decl[0] <= pruned.dec <= s.genrange_decl[1]):
            continue
        if s.genrange_b != (-90.0, 90.0):
            _, b = _eq_to_gal(pruned.ra, pruned.dec)
            if not (s.genrange_b[0] <= float(b) <= s.genrange_b[1]):
                continue
        if not _entry_genranges_pass(pruned, s, genranges, distance_z):
            continue
        surviving.append(pruned)
    if not surviving:
        raise SNANAError(
            f"no SIMLIB entry survives the SIMLIB_* cuts and the input ranges "
            f"GENRANGE_REDSHIFT {genranges.redshift} and GENRANGE_PEAKMJD "
            f"{genranges.peakmjd}", function_name="_apply_simlib_filters")
    if s.simlib_mjd_shift != 0.0:
        for f in surviving:
            f.mjds = f.mjds + s.simlib_mjd_shift
            f.season_starts = f.season_starts + s.simlib_mjd_shift
            f.season_ends = f.season_ends + s.simlib_mjd_shift
    _apply_genrange_mjd_exclude(s.genrange_mjd_exclude, surviving)
    simlib.fields = surviving
    # SIMLIB_FIELDLIST rejects the event rather than the entry (SKIP_SIMLIB_FIELD,
    # snlc_sim.c:22587), so the cycle keeps every LIBID and the test is per-SN.
    simlib.field_prescales = np.array(
        [_field_prescale(f.field_name, prescales) for f in surviving],
        dtype=np.int64)
    _rebuild_flat_structures(simlib)


def _apply_genrange_mjd_exclude(window, fields) -> None:
    """Drop observations inside GENRANGE_MJD_EXCLUDE (snlc_sim.c:19824)."""
    if window is None or window[0] >= window[1]:
        return
    lo, hi = float(window[0]), float(window[1])
    for f in fields:
        keep = ~((f.mjds > lo) & (f.mjds < hi))
        if keep.all():
            continue
        for k in _PER_OBS_KEYS:
            setattr(f, k, getattr(f, k)[keep])
        f.nobs = len(f.mjds)
        f.season_starts, f.season_ends = _compute_seasons(f.mjds)


def _resolve_saturation(instrument, simlib) -> Tuple[int, int]:
    """The effective saturation values after the SIMLIB global-header override.

    SNANA semantics (snlc_sim.c:18479-18481): the SIMLIB header value
    overrides the input-file value only when the latter is still the
    default. NPE_PIXEL_SATURATE default = 1e9; PHOTFLAG_SATURATE default = 0.
    """
    npe, photflag = instrument.npe_pixel_saturate, instrument.photflag_saturate
    if (npe == field_default(Instrument, 'npe_pixel_saturate')
            and simlib.global_npe_pixel_saturate is not None):
        npe = simlib.global_npe_pixel_saturate
    if photflag == 0 and simlib.global_photflag_saturate is not None:
        photflag = simlib.global_photflag_saturate
    return npe, photflag


def _apply_forceval_psf(simlib, value: float) -> None:
    """SNANA FORCEVAL_PSF (snlc_sim.c:19864): overwrite every per-obs PSF sigma
    (PSFSIG1) with a constant."""
    for field in simlib.fields:
        field.psf1s[:] = value


def _load_kcor(config) -> KcorReader:
    """Load the filter calibration and restrict to GENFILTERS."""
    kcor_file = _resolve_input_path(config.survey.kcor_file, config.run)
    kcor = KcorReader(kcor_file, leakage_cut=config.survey.leakage_cut)
    # Restrict to GENFILTERS: SNANA convention matches last char of filter name
    # (e.g. 'LSST-g' → 'g').
    genfilters = config.survey.genfilters
    if genfilters:
        genfilter_chars = set(genfilters)
        # Synthetic spectrograph bands are kept regardless of GENFILTERS: they
        # are generated from SIMLIB SPECTROGRAPH epochs, not requested by name.
        syn = kcor.syn_filters if config.survey.use_simlib_spectrograph else {}
        all_names = kcor.get_filter_names()
        keep = [n for n in all_names if n[-1] in genfilter_chars or n in syn]
        drop = [n for n in all_names if n not in keep]
        for n in drop:
            del kcor.filter_data[n]
        kcor.filter_names = keep
    slog.line(logger, f"Loaded {len(kcor.get_filter_names())} filters")
    return kcor


def _build_filter_lookup(kcor) -> Tuple[List[str], Dict[str, int]]:
    """Build the filter name→index map.

    Returns ``(names, name_to_index)`` where `name_to_index`
    keys include both the canonical name (e.g., 'LSST-g') and its band letter
    ('g'). Mean wavelengths are produced by
    :func:`jax_snlc_core.photometry.resources.load_photometry_resources`
    during grid setup.
    """
    filter_names = kcor.get_filter_names()

    # case kept: SNANA's alphabet holds both cases (INTFILTER, sntools.c:11191)
    name_to_index: Dict[str, int] = {}
    for idx, name in enumerate(filter_names):
        name_to_index[name] = idx
        name_to_index[filter_band(name)] = idx
    return filter_names, name_to_index


def _field_distance_redshift(config, simlib):
    """zCMB of each SIMLIB entry's DISTANCE header (Mpc), NaN where the
    entry has none, or None unless USE_SIMLIB_DISTANCE. Inverted once
    here, when the SIMLIB entries are filtered, so a distance the inversion
    cannot converge on aborts at setup."""
    if not config.survey.use_simlib_distance:
        return None
    distance = header_column(simlib.fields, 'DISTANCE')
    has = np.isfinite(distance)
    z = np.full(len(distance), np.nan)
    if has.any():
        om, _, ol, hd, w0, wa = cosmo_params(config.cosmology)
        mu = 5.0 * np.log10(distance[has]) + 25.0
        z[has] = np.asarray(distance_modulus_to_redshift(
            jnp.asarray(mu), om, ol, hd, w0, wa))
    return jnp.asarray(z)


def _load_field_sfd(config, simlib, field_mwebv
                    ) -> Tuple[Optional[Any], Optional[np.ndarray]]:
    """The SFD98 map and its E(B-V) at each SIMLIB field, when MWEBV needs them.

    OPT_MWEBV=1 takes the SIMLIB field value and only falls back to SFD where a
    field carries MWEBV<=0, and GENRANGE_MWEBV draws uniformly without looking
    anything up, so both leave the maps unread, as does a run with Milky Way
    extinction off.
    """
    mw = config.mw_dust
    if not mw.extinction_applied or mw.genrange[0] >= 0:
        reachable = False
    elif mw.opt_mwebv in (2, 3):
        reachable = True
    else:
        reachable = mw.opt_mwebv == 1 and bool((field_mwebv <= 0).any())
    if not reachable:
        return None, None
    sfd = SFDDustMap()
    field_ras = np.array([f.ra for f in simlib.fields])
    field_decs = np.array([f.dec for f in simlib.fields])
    return sfd, sfd.get_ebv_batch(field_ras, field_decs).astype(np.float32)


def _working_wavelength_range(config, kcor, spectrograph):
    """The rest-frame interval the run evaluates the SED on, which its
    working grid is cut to: nothing redder than the red edge of the reddest
    GENFILTERS band, or of the spectrograph's last bin, is ever observed at
    z >= 0. The blue side is left to the model, since the bluest observed
    rest wavelength depends on the redshift. SED-true spectra sample the
    whole model (get_LAMRANGE_SED_TRUE), so they take the whole support.
    """
    if config.effective_spectrograph_optmask & 128:
        return None
    red = 0.0
    for name in kcor.get_filter_names():
        waves, trans = (np.asarray(a) for a in kcor.get_filter_transmission(name))
        red = max(red, _get_filter_wavelength_edges(waves, trans)[1])
    if spectrograph is not None:
        red = max(red, float(np.max(np.asarray(spectrograph.lam_max))))
    return (0.0, red)


def _create_model(config, kcor, wavelength_grid_points: int,
                  n_simulated: int, frac_pec1a: float,
                  wavelength_range: Optional[Tuple[float, float]]
                  ) -> Tuple[Any, str, str, jnp.ndarray,
                             Tuple[float, float], Tuple[float, float]]:
    """Build the SN model through the shared construction path and apply the
    simulation-only sampling configuration.

    Returns ``(model, model_type, genmodel_path, wavelength_grid,
    model_wave_range, filter_center_range)`` with wave ranges reflecting any
    UVLAM extension and the model already resampled to the working grid,
    cut to ``wavelength_range``.
    """
    genmodel_path = config.model.genmodel
    model_type = resolve_model_type(genmodel_path, hint=config.model.model_type)
    slog.banner(logger, f"Init {model_type.upper()} model")
    slog.kv(logger, "GENMODEL", genmodel_path)

    model = build_model(genmodel_path, config.model_config(),
                        n_wave_points=wavelength_grid_points,
                        wavelength_range=wavelength_range,
                        model_type=model_type)
    if not isinstance(model, SimulatedModel):
        raise SNANAError(
            f"{type(model).__name__} cannot be simulated: it does not play "
            f"the SimulatedModel role.", function_name="build_simulation")
    model.configure_generation(config.model_config(), n_simulated, frac_pec1a)

    wavelength_grid = model.wavelength_grid
    model_wave_range = model.get_wavelength_range()
    filter_center_range = model.filter_center_range or model_wave_range

    slog.kv(logger, "wave range",
            f"{model_wave_range[0]:.0f} - {model_wave_range[1]:.0f} A")
    slog.kv(logger, "filter centre range",
            f"{filter_center_range[0]:.0f} - {filter_center_range[1]:.0f} A")
    slog.line(logger, f"Wavelength grid: {float(wavelength_grid[0]):.0f} - "
              f"{float(wavelength_grid[-1]):.0f} Å "
              f"({len(wavelength_grid)} points)")

    return (model, model_type, genmodel_path, wavelength_grid,
            model_wave_range, filter_center_range)


def _get_filter_wavelength_edges(waves: np.ndarray, trans: np.ndarray):
    """Return ``(blue, red, center)`` matching SNANA ``sntools_calib.c`` filter trimming.

    SNANA: ``trans > 1e-6`` then one extra bin on each side. Returns ``None``
    if the filter has no transmission.
    """
    nonzero = np.where(trans > 1e-6)[0]
    if len(nonzero) == 0:
        return None
    ilam_min, ilam_max = nonzero[0], nonzero[-1]
    if ilam_min > 0:
        ilam_min -= 1
    if ilam_max < len(waves) - 1:
        ilam_max += 1
    mask = trans > 1e-6
    center = np.sum(waves[mask] * trans[mask]) / np.sum(trans[mask])
    return waves[ilam_min], waves[ilam_max], center


def _calculate_filter_redshift_limits(filters: List[str], kcor,
                                      model_wave_range: Tuple[float, float],
                                      filter_center_range: Tuple[float, float]
                                      ) -> Dict[str, Tuple[float, float]]:
    """Compute per-filter ``(z_min, z_max)`` redshift limits from wavelength coverage."""
    logger.info("Calculating filter redshift limits...")
    model_wave_min, model_wave_max = model_wave_range
    filter_center_min, filter_center_max = filter_center_range
    z_limits: Dict[str, Tuple[float, float]] = {}
    for filter_name in filters:
        waves, trans = kcor.get_filter_transmission(filter_name)
        waves = np.array(waves)
        trans = np.array(trans)
        edges = _get_filter_wavelength_edges(waves, trans)
        if edges is None:
            z_limits[filter_name] = (0.0, 0.0)
            continue
        wave_blue, wave_red, wave_center = edges
        z_max_blue_edge = (wave_blue / model_wave_min) - 1.0
        z_max_center = (wave_center / filter_center_min) - 1.0
        z_min_red_edge = (wave_red / model_wave_max) - 1.0
        z_min_center = (wave_center / filter_center_max) - 1.0
        z_max = min(z_max_blue_edge, z_max_center)
        z_min = max(z_min_red_edge, z_min_center)
        z_min = max(0.0, z_min)
        if z_min >= z_max:
            z_limits[filter_name] = (0.0, 0.0)
        else:
            z_limits[filter_name] = (z_min, z_max)
    slog.line(logger, f"Filter redshift limits: {len(filters)} filters")
    for filt in filters:
        zlo, zhi = z_limits[filt]
        slog.line(logger, f"{filt}: {zlo:.3f} < z < {zhi:.3f}", indent=2)
    return z_limits


def _build_grids_and_filters(
    config,
    kcor,
    model,
    wavelength_grid: jnp.ndarray,
    filter_center_range: Tuple[float, float],
    filter_names: List[str],
    name_to_index: Dict[str, int],
    dust_law,
    rv_mw: float,
) -> Tuple[Tuple[float, float], Dict[str, Any], PhotometryResources]:
    """Set up the filter transmissions, z-limits and AB reference on the
    working wavelength grid the model was built with.

    Returns ``(phase_range, filter_fields, photometry)``.

    Side effects: prepares the MW dust law for the grid.
    """
    logger.info("Setting up computation grids...")
    wavelength_grid_points = len(wavelength_grid)
    phase_range = config.rates.trest_range
    logger.info(f"Using TREST range from input file: "
                f"{phase_range[0]:.1f} to {phase_range[1]:.1f} days")

    dust_law.prepare_wavelength_grid(np.array(wavelength_grid))

    all_kcor_filter_names = kcor.get_filter_names()
    shifts = [shift for path in metadata_files(type(model), config.model.genmodel)
              for shift in read_calib_shifts(path)]
    mag_shifts, wave_shifts = resolve_calib_shifts(
        shifts, all_kcor_filter_names, kcor.filter_survey,
        model.enable_magshift, model.enable_waveshift)
    photometry = load_photometry_resources(
        kcor, wavelength_grid, all_kcor_filter_names,
        genmag_off_zp_map=config.instrument.genmag_off_zp,
        lam_shift_map={**config.instrument.fudgeshift_lam, **wave_shifts},
        primary_mag_shift_map=mag_shifts,
    )
    slog.line(logger, f"Pre-interpolated {len(all_kcor_filter_names)} filters to common "
              f"{wavelength_grid_points}-point grid "
              f"({photometry.common_wave_grid[0]:.0f}-"
              f"{photometry.common_wave_grid[-1]:.0f} Å)")
    z_limits = _calculate_filter_redshift_limits(
        all_kcor_filter_names, kcor,
        (float(wavelength_grid[0]), float(wavelength_grid[-1])),
        filter_center_range)
    z_min_per_filter = np.array([z_limits[f][0] for f in all_kcor_filter_names])
    z_max_per_filter = np.array([z_limits[f][1] for f in all_kcor_filter_names])

    # NON1ASED: per-template blue edge
    tpl_bounds = model.per_template_wavelength_bounds
    if tpl_bounds is not None:
        t_wl_min, _t_wl_max = tpl_bounds
        blue_edges = np.zeros(len(all_kcor_filter_names))
        for i, name in enumerate(all_kcor_filter_names):
            waves, trans = kcor.get_filter_transmission(name)
            edges = _get_filter_wavelength_edges(np.array(waves), np.array(trans))
            blue_edges[i] = edges[0]
        z_max_table = blue_edges[None, :] / t_wl_min[:, None] - 1.0
        z_min_table = np.zeros_like(z_max_table)
        slog.line(logger, f"Per-template z-limits: {len(t_wl_min)} templates × "
                  f"{len(all_kcor_filter_names)} filters")
    else:
        z_max_table = z_max_per_filter[None, :]
        z_min_table = z_min_per_filter[None, :]

    filter_fields = dict(
        filter_names=filter_names,
        filter_name_width=kcor.filter_name_width,
        filter_name_to_index=name_to_index,
        filter_z_min=jnp.asarray(z_min_table),
        filter_z_max=jnp.asarray(z_max_table),
    )

    slog.line(logger, f"Grids: {len(wavelength_grid)} wavelengths, realistic phase scheduling")

    return phase_range, filter_fields, photometry


def _cdf_on_grid(config, hz_map, z_grid, dndz_params, rate_mods) -> jnp.ndarray:
    """Normalised cumulative dN/dz over `z_grid`."""
    cosmo = config.cosmology
    omega_m, omega_l = cosmo.omega_matter, cosmo.omega_lambda
    dndz = dndz_density_on_grid(
        dndz_params, z_grid,
        omega_m, 1.0 - omega_m - omega_l, omega_l,
        2997.92458 * (100.0 / cosmo.h0), cosmo.w0, cosmo.wa,
        hz_map, rate_mods)
    cdf = np.cumsum(0.5 * (dndz[:-1] + dndz[1:]) * np.diff(z_grid))
    cdf = np.concatenate([[0.0], cdf])
    return jnp.array(cdf / cdf[-1], dtype=jnp.float32)


def _build_murewgt(config, hz_map):
    """The DNDZ reweight's CDF against distance modulus, from which a MUSHIFT
    range is drawn so that a shifted event keeps the reweighted redshift
    distribution (prep_user_murewgt, snlc_sim.c:7662). The grid runs in
    steps of 0.001 from the lowest generated redshift to 0.1 past the one
    whose distance is the highest generated distance plus the top of the
    MUSHIFT range. None unless there is both a MUSHIFT range and a
    z-dependent DNDZ_ZPOLY_REWGT or DNDZ_Z1POLY_REWGT (snlc_sim.c:7650)."""
    lo, hi = config.model.mushift
    mods = config.rates.dndz_rate_mods
    zpoly, z1poly = mods['zpoly'], mods['z1poly']
    if not (hi - lo > 0 and (len(zpoly) > 1 or len(z1poly) > 1)):
        return None
    omega_m, _omega_k, omega_l, hubble_distance, w0, wa = cosmo_params(config.cosmology)
    anisotropy = anisotropy_pars(config.cosmology)
    # gen_dLmag at galactic (0, 0) on the grid, dLmag for the upper edge
    cos_sep = float(cos_sep_cmb_apex_galactic(0.0, 0.0))

    def mu_of(z, aniso):
        return np.asarray(jax.vmap(lambda zi: distance_modulus(
            zi, zi, omega_m, omega_l, hubble_distance, w0, wa, 0.0, hz_map,
            anisotropy=aniso, cos_sep=cos_sep))(jnp.asarray(z)))

    z_min, z_max = config.rates.redshift_range
    mu_max_grid = float(mu_of([z_max], None)[0]) + hi
    z_max_grid = float(distance_modulus_to_redshift(
        mu_max_grid, omega_m, omega_l, hubble_distance, w0, wa)) + 0.1
    z = np.arange(z_min, z_max_grid, 0.001)
    rewgt = np.ones_like(z)
    if zpoly:
        rewgt = rewgt * eval_genpoly(zpoly, z)
    if z1poly:
        rewgt = rewgt * eval_genpoly(z1poly, 1.0 + z)
    return jnp.asarray(mu_of(z, anisotropy)), jnp.asarray(np.cumsum(rewgt))


def _build_redshift_cdf(config, hz_map):
    """Numerical CDF for dN/dz including the comoving volume element.

    DNDZ_PEC1A gives the peculiar-Ia population its own rate, hence its own
    CDF; set_RATEPAR (snlc_sim.c:14525) picks per event by template class,
    taking none of the DNDZ_* reweights (snlc_sim.c:2722).
    """
    z_min, z_max = config.rates.redshift_range
    z_grid = np.linspace(z_min, z_max, 2000)
    pec1a_params = config.dndz_params(config.rates.dndz_entries_pec1a)
    return (
        jnp.array(z_grid, dtype=jnp.float32),
        _cdf_on_grid(config, hz_map, z_grid, config.dndz_params(),
                     config.rates.dndz_rate_mods),
        _cdf_on_grid(config, hz_map, z_grid, pec1a_params,
                     neutral_rate_mods())
        if pec1a_params else None,
    )


def _load_lensing(config) -> Tuple[Optional[Any], Optional[Any]]:
    """Load the weak-lensing probability map if WEAKLENS_PROBMAP_FILE is
    configured. Returns ``(probmap, inv_cdf)``."""
    path = config.lensing.probmap_file
    if not path or str(path).upper() in ('', 'NONE'):
        return None, None
    probmap = read_lensing_probmap(str(path), dmuscale=config.lensing.dmuscale)
    return probmap, jnp.asarray(probmap.inv_cdf)


def _init_gensmear(config, genmodel_path: str, model_wave_range):
    """Build the GENMAG_SMEAR_MODELNAME scatter model. Only G10 needs a SALT
    directory, for its colour-dispersion file; the rest apply to any GENMODEL."""
    modelname = config.model.genmag_smear_modelname
    if not modelname:
        return None
    smear = build_smear_model(
        modelname, salt_model_dir=Path(os.path.expandvars(genmodel_path)),
        sed_lam_max=(model_wave_range[1] if model_wave_range else None),
        zmin=config.rates.redshift_range[0])
    slog.line(logger, f"GENMAG_SMEAR_MODELNAME {modelname} ready: "
              f"{smear.n_randoms} randoms/SN")
    return smear


def _init_phasecor(config) -> Optional[jnp.ndarray]:
    """The Cholesky factor for GENMAG_SMEAR_ADDPHASECOR, or None when off.

    SNANA's rest-frame phase grid is NBIN_PHASECOR=60 points at
    TMIN=-18.0 with TBIN=1.0 (phases -18..+41;
    init_genSmear_phaseCor, sntools_genSmear.c:3754-3801), with covariance
    C(i,j) = sigma^2 * exp(-|T_i - T_j| / tau) and tau floored at 1e-12
    (snlc_sim.c:10099-10100). Per-SN draws are L @ z with z ~ N(0, I). The
    grid constants live on Instrument (phasecor_phase_min/step/n_grid) so
    construction and lookup share one definition.
    """
    addpc = config.model.genmag_smear_addphasecor
    if addpc is None:
        return None
    sigma, tau = float(addpc[0]), float(addpc[1])
    if sigma <= 0:
        return None
    tau = max(tau, 1.0e-12)
    inst = config.instrument
    phase_grid = (inst.phasecor_phase_min
                  + inst.phasecor_phase_step * np.arange(inst.phasecor_n_grid))
    diff = np.abs(phase_grid[:, None] - phase_grid[None, :])
    cov = sigma * sigma * np.exp(-diff / tau)
    L = np.linalg.cholesky(cov)
    slog.line(logger, f"ADDPHASECOR ready: sigma={sigma}, tau={tau}, "
              f"grid={len(phase_grid)} pts "
              f"[{phase_grid[0]:.0f}, {phase_grid[-1]:.0f}] days")
    # Preserve float32 coefficients, then match the draws' working precision.
    # Mixed-width inputs can make XLA unroll this reduction across the batch.
    return jnp.asarray(L.astype(np.float32), dtype=float_dtype())


def _compute_zp_sedmodel(lam_min, lam_max, kcor) -> np.ndarray:
    """Per-bin spectrograph zero point for SED-model genmag.

    Matches SNANA getZP_SPECTROGRAPH_SEDMODEL (genmag_SEDtools.c:3804):
    ZP = 2.5*log10(sum primFlux(lamCen)*lamCen*lamStep/hc) + magPrimary,
    summed in 10 Å sub-steps, where the primary is the kcor's primary for the
    obs filters and magPrimary is the per-filter "Primary Mag" (ZPoff HDU)
    interpolated at the bin centre (interp_primaryMag_SEDMODEL, clamped at
    the filter wavelength ends).
    """
    filter_names = kcor.get_filter_names()

    primary_name = kcor.filter_primary[filter_names[0]]
    prim_waves, prim_flux = kcor.get_primary_sed(primary_name)
    prim_waves = np.asarray(prim_waves)
    prim_flux = np.asarray(prim_flux)

    mean_lams = np.zeros(len(filter_names))
    prim_mags = np.zeros(len(filter_names))
    for i, name in enumerate(filter_names):
        wave, trans = kcor.get_filter_transmission(name)
        wave = np.asarray(wave)
        trans = np.asarray(trans)
        mean_lams[i] = np.sum(wave * trans) / np.sum(trans)
        prim_mags[i] = kcor.filter_primary_mag[name]
    order = np.argsort(mean_lams)
    mean_lams = mean_lams[order]
    prim_mags = prim_mags[order]

    lam_min = np.asarray(lam_min)
    lam_max = np.asarray(lam_max)
    lamstep = 10.0
    max_steps = int(np.ceil((lam_max - lam_min).max() / lamstep))
    start = lam_min[:, None] + lamstep * np.arange(max_steps)[None, :]
    width = np.clip(lam_max[:, None] - start, 0.0, lamstep)
    lam_cen = start + 0.5 * width
    flux_sum = np.sum(np.interp(lam_cen, prim_waves, prim_flux) * lam_cen * width,
                      axis=1) / HC_ERG_A
    mag_primary = np.interp(0.5 * (lam_min + lam_max), mean_lams, prim_mags)
    zp_sedmodel = np.zeros(len(lam_min))
    pos = flux_sum > 0
    zp_sedmodel[pos] = 2.5 * np.log10(flux_sum[pos]) + mag_primary[pos]
    return zp_sedmodel


def _get_lamrange_sed_true(lamrange_user, kcor, model_wave_range):
    """Ideal-spectrograph wavelength range (get_LAMRANGE_SED_TRUE, genmag_SEDtools.c:2036)."""
    names = kcor.get_filter_names()
    # per-filter nonzero-transmission support (the kcor grid spans all bands)
    supports = []
    for n in names:
        wave, trans = (np.asarray(a) for a in kcor.get_filter_transmission(n))
        nz = np.where(trans > 0)[0]
        supports.append((float(wave[nz[0]]), float(wave[nz[-1]])))
    lmin_flt = min(s[0] for s in supports)
    lmax_flt = max(s[1] for s in supports)
    if lamrange_user is not None and lamrange_user[0] > 0.01 and lamrange_user[1] > 0.01:
        if lamrange_user[0] > lmin_flt or lamrange_user[1] < lmax_flt:
            raise SNANAError(
                f"LAMRANGE_SED_TRUE {lamrange_user} can extend the filter range "
                f"({lmin_flt:.0f}-{lmax_flt:.0f}), not reduce it")
        return lamrange_user[0], lamrange_user[1]
    lmin_sed, lmax_sed = model_wave_range
    if lmin_sed > 0.01:
        return min(lmin_flt, lmin_sed), max(lmax_flt, lmax_sed)
    return lmin_flt, lmax_flt


def _load_spectrograph(config, kcor):
    """Load the spectrograph calibration: a standalone SPECTROGRAPH_FILE,
    else the spectrograph supplied by KCOR_FILE. None without either,
    and under SEDMODEL mode (bit 128), whose ideal spectrograph is
    fabricated from the model's range once the model is built, whatever the
    kcor holds (init_genSpec, snlc_sim.c:10264)."""
    if config.effective_spectrograph_optmask & 128:
        return None
    calibration = None
    spec_file = config.spectra.spectrograph_file
    if spec_file:
        calibration = read_spectrograph(Path(os.path.expandvars(spec_file)))
    if calibration is None:
        calibration = kcor.spectrograph
    if calibration is not None:
        calibration.zp_sedmodel = jnp.array(_compute_zp_sedmodel(
            calibration.lam_min, calibration.lam_max, kcor))
    return calibration


def _build_sed_true_spectrograph(config, kcor, model_wave_range):
    """The ideal spectrograph of SEDMODEL mode (bit 128), over the SED-true
    range."""
    if config.spectra.lambin_sed_true <= 0.001:
        raise SNANAError(
            "SPECTROGRAPH_OPTMASK bit 128 requires LAMBIN_SED_TRUE > 0.001 "
            "(bare bit 128 crashes SNANA)")
    lammin, lammax = _get_lamrange_sed_true(
        config.spectra.lamrange_sed_true, kcor, model_wave_range)
    calibration = build_ideal_spectrograph(
        lammin, lammax, config.spectra.lambin_sed_true)
    calibration.zp_sedmodel = jnp.array(_compute_zp_sedmodel(
        calibration.lam_min, calibration.lam_max, kcor))
    return calibration


def _read_host_spec_table(config):
    """The SPECBASIS/SPECDATA table this config asks for, or None.

    One table slot (HOSTSPEC.ITABLE): SNANA silently lets SPECDATA win when
    both are named, which jax rejects rather than picking for the user.
    """
    basis, data = config.host.specbasis_file, config.host.specdata_file
    if basis and data:
        raise SNANAError(
            "HOSTLIB_SPECBASIS_FILE and HOSTLIB_SPECDATA_FILE are mutually "
            "exclusive; SNANA stores one host-spectrum table.",
            function_name="_read_host_spec_table")
    if not (basis or data):
        return None
    path, mode = ((basis, PREFIX_SPECBASIS) if basis
                  else (data, PREFIX_SPECDATA))
    return read_host_spec_table(Path(os.path.expandvars(path)), mode)


DZ_SAFE_HOSTLIB = 0.002


def _check_hostlib_redshift_coverage(table, redshift_range) -> None:
    """Abort unless the HOSTLIB spans GENRANGE_REDSHIFT (check_redshift_HOSTLIB,
    sntools_host.c:3322). dz_safe converts the zCMB range to heliocentric."""
    ztrue = table.columns['ZTRUE']
    zmin_lib, zmax_lib = float(np.min(ztrue)), float(np.max(ztrue))
    zmin_gen = redshift_range[0] - DZ_SAFE_HOSTLIB
    zmax_gen = redshift_range[1] + DZ_SAFE_HOSTLIB
    if zmin_lib > zmin_gen or zmax_lib < zmax_gen:
        raise SNANAError(
            f"HOSTLIB z-HEL range does not contain user GENRANGE_REDSHIFT.\n"
            f"  HOSTLIB z-HEL range: {zmin_lib:.5f} - {zmax_lib:.5f}\n"
            f"  Gen     z-HEL range: {zmin_gen:.5f} - {zmax_gen:.5f} "
            f"(GENRANGE_REDSHIFT +- {DZ_SAFE_HOSTLIB})\n"
            f"  Gen     z-CMB range: {redshift_range[0]:.5f} - "
            f"{redshift_range[1]:.5f} (GENRANGE_REDSHIFT)",
            function_name="_check_hostlib_redshift_coverage")


def _validate_populations(config, sed_model, hostlib) -> None:
    """Check every parameter's record once the ranges of the quantities the
    populations depend on are known.

    Such a quantity is the redshift, another drawn parameter, or a HOSTLIB
    column, and only the last of those needs the catalogue, which is why
    this waits for ``hostlib`` (None without one) to load. The ranges are
    the run's, so a run is accepted or rejected on its configuration rather
    than on the events a batch draws. A model parameter a shift depends on
    must be drawn before the one it shifts, as the draws go in the model's
    declaration order.
    """
    populations = config.model.populations
    drawn = sed_model.drawn_parameters(populations)
    declared = {name for name, _ in sed_model._population}
    for i, name in enumerate(drawn):
        later = [quantity for quantity in populations[name].depends_on()
                 if quantity in declared and quantity not in drawn[:i]]
        if later:
            raise SNANAError(
                f"A shift on {name} depends on {later[0]}, which is not drawn "
                f"before it. The model draws its population parameters in "
                f"the order {', '.join(drawn)}.",
                function_name="_validate_populations")
    ranges = {REDSHIFT: config.rates.redshift_range}
    for name in set(config.model.quantities_depended_on()):
        if name in ranges:
            continue
        span = config.model.parameter_range(name)
        if span is None:
            columns = [] if hostlib is None else hostlib.selector.varnames
            if name not in columns:
                raise SNANAError(
                    f"A shift depends on '{name}', which is neither a "
                    f"population parameter nor a column this HOSTLIB carries. "
                    f"Columns: {sorted(columns)}.",
                    function_name="_validate_populations")
            column = np.asarray(
                hostlib.catalogue.columns[columns.index(name)])
            span = (float(column.min()), float(column.max()))
        ranges[name] = span
    for param_name, pop in config.model.populations.items():
        validate_population(param_name, pop, ranges)


def _check_snpar_conflict(config, sed_model, hostlib_varnames) -> None:
    """Refuse a population statement the HOSTLIB would then overwrite.

    Under USESNPAR the host's column replaces the drawn value
    (apply_host_snpar_overrides), so a class or a shift on that parameter
    would be drawn and discarded.
    """
    spellings = model_spellings(sed_model)
    columns = [(spellings[column.lower()], column)
               for column in hostlib_varnames if column.lower() in spellings]
    columns += [(p.key, p.hostlib_column) for p in HOST_DUST_PARAMETERS
                if p.hostlib_column in hostlib_varnames]
    clashing = sorted(
        f"{name} (HOSTLIB column {column})"
        for name, column in columns
        if name in config.model.populations
        and config.model.populations[name].is_shaped())
    if clashing:
        raise SNANAError(
            f"This run's population shapes {', '.join(clashing)} through a "
            f"SUBPOP class or a VARIATION, but HOSTLIB_MSKOPT bit 64 "
            f"(USESNPAR) makes the HOSTLIB column replace the drawn value, so "
            f"it would have no effect. Drop it or the USESNPAR bit.",
            function_name="_check_snpar_conflict")


def _load_hostlib(config, simlib, sed_model, spec_zhost,
                  hz_map, frac_pec1a) -> Optional[Hostlib]:
    """The HOSTLIB the run draws its hosts from, built from the input's host
    keys, or None when HOSTLIB_FILE names none. The file is read once, into
    a table the build consumes and drops."""
    host = config.host
    if host.hostlib_file is None:
        logger.info("No HOSTLIB_FILE configured - skipping host galaxy selection")
        return None
    logger.info("Loading HOSTLIB for host galaxy selection...")
    hostlib_file = _existing_file(host.hostlib_file)
    wgtmap_file = (_existing_file(host.wgtmap_file) if host.wgtmap_file
                   else None)
    declared = hostlib_declared(hostlib_file)
    spec_table = _read_host_spec_table(config)
    specid_maps, _, zhost_maps, _ = spec_zhost
    stacked, extra = _hostlib_columns(config, simlib, sed_model, declared,
                                      wgtmap_file, specid_maps + zhost_maps)
    build_only = ([] if spec_table is None
                  else hostlib_column_names(spec_table))
    table = read_hostlib(
        hostlib_file, list(dict.fromkeys(stacked + extra + build_only)),
        genrange_ra=host.hostlib_genrange_ra,
        genrange_dec=host.hostlib_genrange_dec,
        genrange_redshift=config.rates.redshift_range,
        maxread=host.hostlib_maxread,
        snr_store=(host.snr_store, config.survey.genfilters),
        varname_zphot=host.varname_zphot,
        swap_zphot=bool(host.mskopt & 128),
        fixsersic=host.fixsersic,
        scale_sersic_size=host.scale_sersic_size,
    )
    _check_hostlib(config, sed_model, table, declared)
    selector, catalogue = _hostlib_selection(config, simlib, sed_model, table,
                                             stacked, extra, wgtmap_file,
                                             hz_map, frac_pec1a)
    quantiles = zphot_quantile_columns(declared)
    if host.mskopt & HOSTLIB_MSKOPT_ZPHOT_QGAUSS and not quantiles:
        # the forced 11-decade set (sntools_host.c:4583-4585)
        quantiles = DECADE_QUANTILES
    return Hostlib(
        selector=selector,
        catalogue=catalogue,
        galids=table.columns['GALID'],
        output_only_columns=discover_output_only_columns(declared),
        zphot_quantiles=quantiles,
        genzphot=build_genzphot(host, config.rates.redshift_range,
                                (table.z_min, table.z_max)),
        z_max=table.z_max,
        vpec_max=table.vpec_max,
        vpec_err=table.vpec_err,
        wronghost=(None if is_ignorefile(host.wronghost_file)
                   else read_wronghost_file(
                       os.path.expandvars(host.wronghost_file),
                       *config.rates.redshift_range)),
        zphoteff=(None if is_ignorefile(host.zphoteff_file)
                  else read_zphoteff_file(
                      os.path.expandvars(host.zphoteff_file),
                      table.columns['ZTRUE'][[0, -1]])),
        spectra=(None if spec_table is None
                 else HostSpectra(spec_table,
                                  hostlib_coefficients(spec_table,
                                                       table.columns))),
    )


def _existing_file(path: str) -> str:
    """``path`` with its environment variables expanded, or its gzipped
    twin when only that exists."""
    path = os.path.expandvars(path)
    if not Path(path).exists() and Path(path + '.gz').exists():
        return path + '.gz'
    return path


def _hostlib_columns(config, simlib, sed_model, declared, wgtmap_file,
                     searcheff_maps):
    """The HOSTLIB columns the run reads, as ``(stacked, extra)``: those
    the draw gathers into each host's properties, and those read by name
    (the photo-z, its quantiles, STOREPAR and the output-only columns a
    SEARCHEFF map may name). The variables of ``searcheff_maps``, the
    SEARCHEFF SPEC and zHOST maps, are among those gathered. A column the
    file lacks is simply not read."""
    host = config.host
    mskopt = host.mskopt
    stacked = essential_columns()
    if mskopt & 128:
        stacked.extend(['ZPHOT', 'ZPHOT_ERR'])
    if mskopt & 512:
        stacked.extend(['VPEC', 'VPEC_ERR'])
    if mskopt & 2:
        # GALMAG: host photometry for whatever [band]_obs columns exist
        # (sntools_host.c:2757-2766)
        stacked.extend(v for v in declared if v.endswith('_obs'))
    if (config.survey.use_simlib_groupid
            and any(f.groupid_hostlib for f in simlib.fields)):
        stacked.append('GROUPID')
    detect = (host.snr_detect is not None or host.mag_detect is not None
              or host.snr_store is not None)
    if detect:
        stacked.extend(v for v in declared
                       if v.endswith('_obs') and not v.endswith('_obs_err'))
        if host.snr_detect is not None or host.snr_store is not None:
            stacked.extend(v for v in declared if v.endswith('_obs_err'))
    stacked.extend(['WEAKLENS_DMU', 'TRUE'])
    # the variables a SEARCHEFF map or the WGTMAP reads
    stacked.extend(v for m in searcheff_maps for v in m.varnames)
    if wgtmap_file:
        stacked.extend(read_wgtmap_varnames(Path(wgtmap_file)))
    stacked.extend(host.storepar)
    # host properties a population depends on (sntools_host.c:951-969)
    generated = {name for name, _ in sed_model._population} | {REDSHIFT}
    stacked.extend(name for name in config.model.quantities_depended_on()
                   if name not in generated)
    if mskopt & 64:
        # USESNPAR: the model's SN-parameter columns and SNMAGSHIFT
        # (sntools_host.c:671-686)
        stacked.extend([*hostlib_snpar_columns(model_spellings(sed_model),
                                               declared), 'SNMAGSHIFT'])
    extra = (['ZPHOT', 'ZPHOT_ERR']
             + [name for name, _ in zphot_quantile_columns(declared)]
             + host.storepar + discover_output_only_columns(declared))
    return list(dict.fromkeys(stacked)), list(dict.fromkeys(extra))


def _check_hostlib(config, sed_model, table, declared) -> None:
    """The HOSTLIB checks SNANA aborts on once the catalogue is read."""
    host = config.host
    mskopt = host.mskopt
    # a redshift taken from the SIMLIB is not drawn from GENRANGE_REDSHIFT,
    # so the catalogue need not span it (check_redshift_HOSTLIB, :3320)
    if not config.survey.use_simlib_redshift:
        _check_hostlib_redshift_coverage(table, config.rates.redshift_range)
    if mskopt & 64:
        _check_snpar_conflict(config, sed_model, declared)
        # USESNPAR with no SN-param column is fatal (sntools_host.c:2616-2624)
        snpar = (*hostlib_snpar_columns(model_spellings(sed_model),
                                        table.columns), 'SNMAGSHIFT')
        if not any(name in table.columns for name in snpar):
            names = [name for name, _ in sed_model._population]
            raise SNANAError(
                f"Found zero SN params for HOSTLIB_MSKOPT ({mskopt}) & 64 "
                f"option. Expected a column for 1 or more of "
                f"'{' '.join(names + list(snpar))}'.",
                function_name="_check_hostlib")
    # USEVPEC needs the VPEC column and an error, as a column or the
    # header's VPEC_ERR (sntools_host.c:2797-2812)
    if mskopt & 512:
        if 'VPEC' not in table.columns:
            raise SNANAError(
                "Could not find required VPEC column. Add VPEC column, or "
                "remove 512-bit of HOSTLIB_MSKOPT", function_name="_check_hostlib")
        if 'VPEC_ERR' not in table.columns and table.vpec_err is None:
            raise SNANAError(
                "Found VPEC column, but VPEC_ERR not defined. Add VPEC_ERR "
                "column, or define VPEC_ERR in header",
                function_name="_check_hostlib")
    # Morphology-dependent MSKOPT bits are fatal without a Sersic profile
    if table.sersic_profiles is None:
        if mskopt & 2:
            raise SNANAError(
                "HOSTLIB_MSKOPT bit 2 (GALMAG): Galaxy (Sersic) profile NOT defined "
                "==> cannot compute galaxy mag at SN position.",
                function_name="_check_hostlib")
        if mskopt & 8:
            raise SNANAError(
                "HOSTLIB_MSKOPT bit 8 (SN2GAL_RADEC): must define host Sersic profile "
                "to move SN near host.",
                function_name="_check_hostlib")
    zphot_col, zphot_err_col = zphot_columns(host.varname_zphot, declared)
    # The forced Gaussian quantiles need catalogue ZPHOT/ZPHOT_ERR
    # ("Cannot compute forced Gaussian quantiles", sntools_host.c:4587-4607)
    if mskopt & HOSTLIB_MSKOPT_ZPHOT_QGAUSS and (
            zphot_col not in declared or zphot_err_col is None):
        raise SNANAError(
            f"HOSTLIB_MSKOPT bit {HOSTLIB_MSKOPT_ZPHOT_QGAUSS} (ZPHOT_QGAUSS) "
            "needs ZPHOT and ZPHOT_ERR columns in the HOSTLIB.",
            function_name="_check_hostlib")
    # init_HOSTLIB_ZPHOTEFF (sntools_host.c:4687-4692)
    if not is_ignorefile(host.zphoteff_file) and zphot_col not in declared:
        raise SNANAError(
            "Cannot implement HOSTLIB_ZPHOTEFF_FILE because ZPHOT is not "
            "defined in the HOSTLIB.", function_name="_check_hostlib")
    absent = [name for name in host.storepar if name not in table.columns]
    if absent:
        raise SNANAError(
            f"HOSTLIB_STOREPAR names {', '.join(absent)}, which the "
            f"HOSTLIB does not provide.", function_name="_check_hostlib")


def _hostlib_selection(config, simlib, sed_model, table, stacked, extra,
                       wgtmap_file, hz_map, frac_pec1a):
    """The host draw's settings and the catalogue it draws from, as
    ``(HostSelector, HostCatalogue)``: the weights (WGTMAP, SALT2GAMMA_GRID,
    TRUE, GROUPID) and the input's host keys over ``table``."""
    host = config.host
    wgtsum_data = _hostlib_weights(config, sed_model, table, wgtmap_file)
    groupid_wgtsums, field_groupid_set = _groupid_weights(
        config, simlib, table, wgtsum_data.weights)

    dztol_coeffs = tuple(float(x) for x in host.dztol.split(','))
    useonce = bool(host.mskopt & 32)
    z_min, z_max = config.rates.redshift_range
    dndz_density, dndz_density_pec1a, dndz_z_grid = None, None, None
    field_vdotn = np.zeros(len(simlib.fields))
    if host.mskopt & 16:
        cosmo = config.cosmology
        dndz_z_grid = np.linspace(z_min, z_max, 1000, dtype=np.float64)

        def density(dndz_params, rate_mods):
            return dndz_density_on_grid(
                dndz_params, dndz_z_grid, cosmo.omega_matter,
                1.0 - cosmo.omega_matter - cosmo.omega_lambda,
                cosmo.omega_lambda, 2.99792458e5 / cosmo.h0,
                cosmo.w0, cosmo.wa, hz_map, rate_mods)

        dndz_density = density(config.dndz_params(),
                               config.rates.dndz_rate_mods)
        if frac_pec1a > 0.0:
            dndz_density_pec1a = density(
                config.dndz_params(config.rates.dndz_entries_pec1a),
                neutral_rate_mods())
        if cosmo.vel_cmbapex >= 1.0:
            field_ras = np.array([f.ra for f in simlib.fields], dtype=np.float64)
            field_decs = np.array([f.dec for f in simlib.fields], dtype=np.float64)
            field_vdotn = np.asarray(_cmb_vdotn(field_ras, field_decs),
                                     dtype=np.float64)

    mindaysep = host.mindaysep_samegal
    if mindaysep <= MINDAYSEP_SAMEGAL_MAX:
        # strict test, as SNANA's (sntools_host.c:5654-5663)
        if useonce and mindaysep < MINDAYSEP_SAMEGAL_MAX:
            raise SNANAError(
                "Cannot set both HOSTLIB_MSKOPT USEONCE (bit 32) "
                "and HOSTLIB_MINDAYSEP_SAMEGAL",
                function_name="_hostlib_selection",
            )
        mindaysep = (mindaysep, config.rates.peakmjd_range)
    else:
        mindaysep = None

    # GALID_PRIORITY: active for 0 < min <= max (GEN_SNHOST_GALID,
    # sntools_host.c:6300); '0,0' is "not set"
    galid_pri_parts = [int(x) for x in host.galid_priority.split(',')]
    galid_priority = (tuple(galid_pri_parts[:2])
                      if len(galid_pri_parts) >= 2
                      and 0 < galid_pri_parts[0] <= galid_pri_parts[1] else None)

    if host.snr_detect is not None:
        detection_filter = ('snr', host.snr_detect, host.snr_scale)
    elif host.mag_detect is not None:
        detection_filter = ('mag', host.mag_detect, None)
    else:
        detection_filter = None

    return build_host_selection(
        table, stacked, extra, wgtsum_data.wgtsum, wgtsum_data.weights,
        wgtsum_data.galaxy_snmagshift, wgtsum_data.snvar_names,
        wgtsum_data.snvar_bin_centers,
        dztol_coeffs=dztol_coeffs, useonce=useonce,
        hostlib_mskopt=host.mskopt,
        genrange_ra=config.survey.genrange_ra,
        genrange_decl=config.survey.genrange_decl,
        genrange_b=config.survey.genrange_b,
        dndz_density=dndz_density,
        dndz_density_pec1a=dndz_density_pec1a,
        dndz_z_grid=dndz_z_grid,
        z_range=(z_min, z_max),
        vpec=_host_draw_vpec(config),
        field_vdotn=field_vdotn,
        maxddlr=(host.maxddlr, host.maxddlr2),
        mindaysep=mindaysep,
        smear_sersic=host.smear_sersic,
        position_settings=dict(
            fixran_wgt=float(host.fixran_wgt),
            fixran_radius=float(host.fixran_radius),
            fixran_phi=float(host.fixran_phi),
            mnintflux_snpos=float(host.mnintflux_snpos),
            mxintflux_snpos=float(host.mxintflux_snpos)),
        abmag=(host.abmag_offset, host.abmag_force),
        igal_force=resolve_igal_force(table, host.galid_force),
        galid_priority=galid_priority,
        detection_filter=detection_filter,
        groupid_wgtsums=groupid_wgtsums,
        field_groupid_set=field_groupid_set,
        field_z_window=np.array([f.genrange_redshift for f in simlib.fields],
                                dtype=np.float64),
    )


def _host_draw_vpec(config):
    """``(shift, sigma)`` in km/s of the peculiar velocity in the zHEL the host draw sees (gen_redshift_helio, snlc_sim.c:17226-17238)."""
    cosmo = config.cosmology
    if cosmo.vpec_force < 1.0e8:
        return cosmo.vpec_force, 0.0
    if config.host.mskopt & 512:
        return 0.0, 0.0
    return 0.0, cosmo.gensigma_vpec


def _hostlib_weights(config, sed_model, table, wgtmap_file) -> WgtSumData:
    """The galaxies' draw weights: the WGTMAP's, or uniform without one,
    with SALT2GAMMA_GRID's SNMAGSHIFT and the TRUE column's zeroing."""
    host = config.host
    if wgtmap_file:
        wgtmap = read_wgtmap(wgtmap_file,
                             wgtmap_snvar_names(model_spellings(sed_model)),
                             opt_extrap=host.wgtmap_extrap)
        slog.line(logger, f"Loaded WGTMAP ({wgtmap.ndim}D)")
        wgtsum_data = compute_wgtsum(table, wgtmap)
    else:
        slog.line(logger, "No WGTMAP - using uniform galaxy weights")
        uniform_weights = np.ones(table.ngal, dtype=np.float32)
        uniform_wgtsum = np.cumsum(uniform_weights) / table.ngal
        wgtsum_data = WgtSumData(
            galaxy_indices=np.arange(table.ngal),
            wgtsum=uniform_wgtsum[np.newaxis, :],
            weights=uniform_weights[np.newaxis, :],
            total_galaxies=table.ngal,
        )

    gamma_parts = [float(x) for x in host.biascor_salt2gamma_grid.split(',')]
    if len(gamma_parts) >= 2 and gamma_parts[0] != gamma_parts[1]:
        gamma_min, gamma_max = gamma_parts[0], gamma_parts[1]
        is_odd = (table.columns['GALID'] % 2) == 1
        gamma_values = np.where(is_odd, gamma_min, gamma_max)
        if (wgtsum_data.galaxy_snmagshift is not None
                and wgtsum_data.galaxy_snmagshift.ndim == 2):
            gamma_values = np.broadcast_to(
                gamma_values[np.newaxis, :],
                wgtsum_data.galaxy_snmagshift.shape,
            ).copy()
        wgtsum_data.galaxy_snmagshift = gamma_values
        slog.line(logger, f"SALT2GAMMA_GRID: [{gamma_min}, {gamma_max}] "
                  f"(overrides SNMAGSHIFT by GALID parity)")

    if 'TRUE' in table.columns:
        true_mask = table.columns['TRUE'] == 0
        n_zeroed = int(np.sum(true_mask))
        if n_zeroed > 0:
            wgt = np.array(wgtsum_data.weights, copy=True)
            wgt[..., true_mask] = 0.0
            wgtsum_data.wgtsum = np.cumsum(wgt, axis=-1)
            wgtsum_data.weights = wgt
            slog.line(logger, f"TRUE column: zeroed weights for {n_zeroed} non-true galaxies")
    return wgtsum_data


def _groupid_weights(config, simlib, table, base_weights):
    """GROUPID matching (RDFLAG_GROUPID, snlc_sim.c:22115): the weight
    cumsum restricted to each unique GROUPID set the SIMLIB names, and each
    SIMLIB entry's set index (-1 for none), or ``(None, None)``."""
    if not config.survey.use_simlib_groupid:
        return None, None
    unique_sets: Dict[frozenset, int] = {}
    field_to_set: List[int] = []
    for f in simlib.fields:
        gset = frozenset(f.groupid_hostlib)
        if gset and gset not in unique_sets:
            unique_sets[gset] = len(unique_sets)
        field_to_set.append(unique_sets.get(gset, -1))
    if not unique_sets:
        return None, None
    if 'GROUPID' not in table.columns:
        raise SNANAError(
            "SIMLIB requires a HOSTLIB GROUPID but the HOSTLIB has no "
            "GROUPID column.", function_name="_groupid_weights")
    groupid = np.asarray(table.columns['GROUPID'], dtype=np.int64)
    filtered_wgtsums: Dict[int, np.ndarray] = {}
    for gset, idx in unique_sets.items():
        mask = np.isin(groupid, list(gset))
        filtered_wgtsums[idx] = np.cumsum(
            base_weights * mask[np.newaxis, :], axis=1).astype(np.float32)
    n_matched = sum(1 for x in field_to_set if x >= 0)
    slog.line(logger, f"GROUPID matching: {len(unique_sets)} unique sets, "
              f"{n_matched}/{len(field_to_set)} fields with GROUPID")
    return filtered_wgtsums, np.array(field_to_set, dtype=np.int32)


_FLUXERR_DISABLED = dict(
    fluxerr_enabled=False, fluxerr_active_vars=(),
    fluxerr_n_dims=0, fluxerr_is_errscale=True, fluxerr_grid=None,
    fluxerr_grid_origins=(), fluxerr_grid_steps=(), fluxerr_grid_lengths=(),
    fluxerr_scale_true=None, fluxerr_scale_data=None, fluxerr_mask_true=None,
    fluxerr_mask_data=None, fluxerr_field_group_idx_per_field=None,
    fluxerr_needs_host_vars=False,
    fluxerr_has_redcov=False, fluxerr_redcov_factors=None, fluxerr_n_factors=0)


def _load_fluxerr_model(config, simlib, kcor, hostlib) -> Dict[str, Any]:
    """Load FLUXERRMODEL_FILE (if configured) and compile its JIT-ready state.

    Returns the fluxerr_* LightCurveSimulator fields, disabled when no file is
    configured.
    """
    path = config.instrument.fluxerr_file
    if not path or str(path).strip().upper() in ('NONE', 'NULL', '', 'BLANK'):
        return dict(_FLUXERR_DISABLED)

    logger.info(f"Loading FLUXERRMODEL: {path}")
    data = read_fluxerr_model_file(Path(os.path.expandvars(str(path))))

    redcov_override = config.instrument.fluxerr_redcov
    if redcov_override:
        data.redcov = parse_redcov_string(str(redcov_override))

    ignore_str = config.instrument.fluxerr_ignore_dataerr
    if ignore_str:
        ignore_set = {s.strip() for s in str(ignore_str).split(',') if s.strip()}
        seen = set()
        for m in data.maps:
            if m.name in ignore_set:
                m.mask_apply = m.mask_apply & ~MASK_APPLY_DATA
                seen.add(m.name)
        missing = ignore_set - seen
        if missing:
            raise SNANAError(
                f"FLUXERRMAP_IGNORE_DATAERR references unknown MAPNAME(s) "
                f"{sorted(missing)}; valid names are "
                f"{sorted(m.name for m in data.maps)}",
                function_name="_load_fluxerr_model",
            )

    fx = dict(_FLUXERR_DISABLED)
    fx['fluxerr_enabled'] = True
    fx.update(_compile_fluxerr_model(data, simlib, kcor))

    host_dependent = {'SBMAG', 'GALMAG'}
    needed_host_vars = sorted(set(fx['fluxerr_active_vars']) & host_dependent)
    if needed_host_vars and hostlib is None:
        raise SNANAError(
            f"FLUXERRMODEL_FILE {path!r} references variable(s) "
            f"{needed_host_vars} which require host galaxy properties "
            f"from a HOSTLIB, but no HOSTLIB_FILE is configured. "
            f"Either configure HOSTLIB_FILE or use a FLUXERRMODEL file "
            f"that doesn't reference SBMAG/GALMAG.",
            function_name="_load_fluxerr_model",
        )

    slog.line(logger, f"FLUXERRMODEL ready: {len(data.maps)} maps, "
              f"{fx['fluxerr_n_dims']}-D, vars={fx['fluxerr_active_vars']}, "
              f"redcov={fx['fluxerr_has_redcov']}")
    return fx


def _compile_fluxerr_model(data, simlib, kcor) -> Dict[str, Any]:
    """Compile the parsed FluxErrModelData into JIT-ready arrays.

    Assumes the file passed the uniform-VARNAMES + uniform-grid_axes
    validators in the reader, so the maps stack into one tensor per
    (field_group, band). Returns the compiled fluxerr_* fields.
    """
    fx: Dict[str, Any] = {}

    shared_varnames = list(data.maps[0].varnames)
    active_vars = tuple(v for v in shared_varnames[:-1])
    if len(active_vars) > 3:
        raise SNANAError(
            f"FLUXERRMODEL with n_dims > 3 not yet supported (got {len(active_vars)})",
            function_name="_compile_fluxerr_model")
    fx['fluxerr_active_vars'] = active_vars
    fx['fluxerr_n_dims'] = len(active_vars)
    is_errscale = bool(data.maps[0].is_errscale)
    fx['fluxerr_is_errscale'] = is_errscale

    shared_axes = data.maps[0].grid_axes
    grid_shape = tuple(int(a.shape[0]) for a in shared_axes)
    fx['fluxerr_grid_origins'] = tuple(float(a[0]) for a in shared_axes)
    fx['fluxerr_grid_steps'] = tuple(
        float(a[1] - a[0]) if a.shape[0] > 1 else 1.0 for a in shared_axes
    )
    fx['fluxerr_grid_lengths'] = grid_shape

    group_names_in_use = sorted({m.field_group for m in data.maps})
    if ALL_GROUP not in group_names_in_use:
        group_names_in_use.append(ALL_GROUP)
    group_to_idx = {name: i for i, name in enumerate(group_names_in_use)}
    n_groups = len(group_names_in_use)

    n_simlib = len(simlib.fields)
    field_group_idx_per_field = np.full(
        n_simlib, group_to_idx[ALL_GROUP], dtype=np.int32)
    field_to_group_idx: Dict[str, int] = {}
    for group_name, fields_in_group in data.field_groups.items():
        if group_name not in group_to_idx:
            continue
        gi = group_to_idx[group_name]
        for fn in fields_in_group:
            field_to_group_idx.setdefault(fn, gi)
    for m in data.maps:
        if m.field_group != ALL_GROUP and m.field_group not in data.field_groups:
            field_to_group_idx.setdefault(m.field_group, group_to_idx[m.field_group])
    for i, sf in enumerate(simlib.fields):
        if sf.field_name in field_to_group_idx:
            field_group_idx_per_field[i] = field_to_group_idx[sf.field_name]
    fx['fluxerr_field_group_idx_per_field'] = jnp.asarray(field_group_idx_per_field)

    filter_names = kcor.get_filter_names()
    n_bands = len(filter_names)
    band_to_idx: Dict[str, int] = {}
    for i, fn in enumerate(filter_names):
        short = filter_band(fn)
        band_to_idx[short] = i

    no_op_value = 1.0 if is_errscale else 0.0
    stacked = np.full((n_groups, n_bands) + grid_shape, no_op_value, dtype=np.float32)
    scale_true = np.ones((n_groups, n_bands), dtype=np.float32)
    scale_data = np.ones((n_groups, n_bands), dtype=np.float32)
    mask_true = np.zeros((n_groups, n_bands), dtype=np.float32)
    mask_data = np.zeros((n_groups, n_bands), dtype=np.float32)

    # BAND matched by substring (sntools_fluxErrModels.c:900)
    for m in data.maps:
        if m.field_group not in group_to_idx:
            continue
        gi = group_to_idx[m.field_group]
        band_str = m.band
        if band_str.upper() == 'ALL':
            target_bis = np.arange(n_bands)
        else:
            target_bis = np.fromiter(
                (band_to_idx[c] for c in band_str if c in band_to_idx),
                dtype=np.int64,
            )
        if target_bis.size == 0:
            continue
        mt = 1.0 if (m.mask_apply & MASK_APPLY_TRUE) else 0.0
        md = 1.0 if (m.mask_apply & MASK_APPLY_DATA) else 0.0
        stacked[gi, target_bis] = m.grid_values.astype(np.float32)
        scale_true[gi, target_bis] = float(m.scale_true)
        scale_data[gi, target_bis] = float(m.scale_data)
        mask_true[gi, target_bis] = mt
        mask_data[gi, target_bis] = md

    fx['fluxerr_grid'] = jnp.asarray(stacked)
    fx['fluxerr_scale_true'] = jnp.asarray(scale_true)
    fx['fluxerr_scale_data'] = jnp.asarray(scale_data)
    fx['fluxerr_mask_true'] = jnp.asarray(mask_true)
    fx['fluxerr_mask_data'] = jnp.asarray(mask_data)

    # F[fg,b,k] = sqrt(rho_k) where REDCOV entry k applies (sntools_fluxErrModels.c)
    n_factors = len(data.redcov)
    F = np.zeros((n_groups, n_bands, n_factors), dtype=np.float32)
    for k, entry in enumerate(data.redcov):
        sqrt_rho = float(np.sqrt(entry.redcov))
        if entry.field_group == ALL_GROUP:
            fg_indices = list(range(n_groups))
        elif entry.field_group in group_to_idx:
            fg_indices = [group_to_idx[entry.field_group]]
        else:
            continue
        for band_char in entry.bands:
            if band_char not in band_to_idx:
                continue
            bi = band_to_idx[band_char]
            for gi in fg_indices:
                F[gi, bi, k] = sqrt_rho
    fx['fluxerr_redcov_factors'] = jnp.asarray(F)

    groups_in_use = np.unique(np.asarray(field_group_idx_per_field))
    fx['fluxerr_has_redcov'] = bool(
        np.any(F[groups_in_use] != 0)) if len(groups_in_use) > 0 else False
    fx['fluxerr_n_factors'] = n_factors

    fx['fluxerr_needs_host_vars'] = bool(
        'SBMAG' in active_vars or 'GALMAG' in active_vars)
    return fx


def _parse_searcheff_logic(logic_string: str,
                           filter_name_to_index: Dict[str, int]) -> List[int]:
    """Convert logic string (e.g., 'gr+ri+gi') into list of per-filter bitmasks."""
    tokens: List[str] = []
    current = ''
    for char in logic_string:
        if char == '+':
            if current:
                tokens.append(current)
                current = ''
        elif not char.isspace():
            current += char
    if current:
        tokens.append(current)
    if not tokens:
        raise SNANAError(f"Invalid SEARCHEFF logic string: {logic_string}")

    # unknown filters get an unreachable bit (INTFILTER, sntools_trigger.c:1018)
    ungenerated: Dict[str, int] = {}
    n_generated = len(filter_name_to_index)
    masks: List[int] = []
    for token in tokens:
        mask = 0
        for symbol in token:
            if symbol in filter_name_to_index:
                idx = filter_name_to_index[symbol]
            else:
                idx = ungenerated.setdefault(symbol, n_generated + len(ungenerated))
            mask |= (1 << idx)
        masks.append(mask)
    return masks


def _read_searcheff_logic_file(logic_file, survey_name: str,
                               path_user_input: str) -> Tuple[int, str]:
    """(NMJD, logic) from the SEARCHEFF pipeline logic file.

    Mirrors init_SEARCHEFF_LOGIC (sntools_trigger.c:877-923): 'DEFAULT'
    names SEARCHEFF_PIPELINE_LOGIC.DAT, and the file is found as the other
    SEARCHEFF files are (`resolve_searcheff_file`). The key is required: an
    unset key is SNANA's NONE, which names no file (snlc_sim.c:1301). The
    survey match is the exact token "<SURVEY>:" over the file's token
    stream with the last match winning; an absent survey entry is fatal.
    """
    if logic_file is None or is_ignorefile(logic_file):
        raise SNANAError(
            "SEARCHEFF_PIPELINE_EFF_FILE holds detection curves, which need "
            "a SEARCHEFF_PIPELINE_LOGIC_FILE (or DEFAULT)",
            function_name="init_SEARCHEFF_LOGIC")
    name = ('SEARCHEFF_PIPELINE_LOGIC.DAT' if logic_file == 'DEFAULT'
            else logic_file)
    logic_path = resolve_searcheff_file(
        name, 'SEARCHEFF_PIPELINE_LOGIC_FILE', path_user_input)
    # DOCANA required whatever REQUIRE_DOCANA says (snlc_sim.c:1297)
    check_file_docana(logic_path)

    survey_key = f'{survey_name}:'
    with open_text(logic_path) as fp:
        tokens = fp.read().split()
    n_mjd, logic_string = 0, ''
    for i, token in enumerate(tokens[:-2]):
        if token == survey_key:
            try:
                n_mjd = int(tokens[i + 1])
            except ValueError:
                raise SNANAError(
                    f"Malformed NMJD '{tokens[i + 1]}' for survey "
                    f"'{survey_name}' in {logic_path}",
                    function_name="_read_searcheff_logic_file")
            logic_string = tokens[i + 2]
    if n_mjd == 0:
        raise SNANAError(
            f"Could NOT find search pipeline logic for {survey_name}; "
            f"check {logic_path}",
            function_name="_read_searcheff_logic_file")
    slog.line(logger, f"SEARCHEFF logic: {n_mjd} MJDs require filters="
              f"{logic_string}")
    return n_mjd, logic_string


def _load_searcheff_pipeline_curves(config, simlib, filter_name_to_index
                                    ) -> Dict[str, Any]:
    """The SEARCHEFF pipeline state: the detection curves and PHOTPROB maps
    the eff file holds, the PHOTFLAG bits, and the trigger logic.

    APPLY_SEARCHEFF_OPT gates event rejection, not evaluation: SNANA loads the
    maps and detects regardless (init_SEARCHEFF, sntools_trigger.c:114).
    """
    sc = config.survey
    # no file, or DEFAULT: every event pipeline-found (sntools_trigger.c:258-310,
    # :2037-2044)
    if (sc.searcheff_pipeline_eff_file == 'DEFAULT'
            or is_ignorefile(sc.searcheff_pipeline_eff_file)):
        return {
            'curves': [], 'photprob_maps': [],
            'photflag_detect': sc.searcheff_photflag_detect,
            'photflag_trigger': sc.searcheff_photflag_trigger,
            'n_trigger_detections': 0,
            'trigger_logic': '',
            'filter_masks': [],
        }

    pipeline_eff_file = resolve_searcheff_file(
        sc.searcheff_pipeline_eff_file, 'SEARCHEFF_PIPELINE_EFF_FILE',
        config.run.path_user_input)
    eff = read_pipeline_eff_file(pipeline_eff_file,
                                 sc.searcheff_photflag_detect,
                                 sc.searcheff_photflag_trigger)
    # the logic is read only for a file with curves (sntools_trigger.c:115)
    n_trigger_detections, trigger_logic = (_read_searcheff_logic_file(
        sc.searcheff_pipeline_logic_file, simlib.survey_name,
        config.run.path_user_input) if eff.curves else (0, ''))
    slog.line(logger, f"Loaded SEARCHEFF from {pipeline_eff_file.name} "
              f"for filters: {[c.filterlist for c in eff.curves]} "
              f"(PHOTFLAG_DETECT={eff.photflag_detect}, "
              f"PHOTFLAG_TRIGGER={eff.photflag_trigger})")

    if eff.photprob_maps:
        resolve_photprob_fieldlist_indices(
            eff.photprob_maps, [f.field_name for f in simlib.fields])
        slog.line(logger, f"Loaded {len(eff.photprob_maps)} PHOTPROB maps")

    return {
        'curves': eff.curves, 'photprob_maps': eff.photprob_maps,
        'photflag_detect': eff.photflag_detect,
        'photflag_trigger': eff.photflag_trigger,
        'n_trigger_detections': n_trigger_detections,
        'trigger_logic': trigger_logic,
        'filter_masks': (_parse_searcheff_logic(trigger_logic,
                                                filter_name_to_index)
                         if eff.curves else []),
    }


def _read_searcheff_specid_zhost(config, spellings
                                 ) -> Tuple[List[Any], int, List[Any], int]:
    """``(specid_maps, specid_flag, zhost_maps, zhost_flag)`` from the
    SEARCHEFF SPEC and zHOST files, after checking that APPLY_SEARCHEFF_OPT
    asks only for stages they can pass (init_SEARCHEFF,
    sntools_trigger.c:119-125). A map variable spelling a population
    parameter, in any of the ways ``spellings`` holds, is named as the
    simulation names it."""
    sc = config.survey
    loaded = {}
    for maptype, named in (('SPEC', sc.searcheff_spec_eff_file),
                           ('zHOST', sc.searcheff_zhost_eff_file)):
        maps, flag = read_searcheff_maps(named, maptype,
                                         config.run.path_user_input)
        for m in maps:
            m.varnames = [spellings.get(v.lower(), v) for v in m.varnames]
        if flag == FLAG_NORMAL:
            slog.line(logger, f"Loaded SEARCHEFF_{maptype} from "
                              f"{Path(named).name} ({len(maps)} maps)")
        loaded[maptype] = (maps, flag)
    (specid_maps, specid_flag), (zhost_maps, zhost_flag) = (
        loaded['SPEC'], loaded['zHOST'])
    check_applymask_searcheff(sc.apply_searcheff_opt, specid_maps,
                              specid_flag, zhost_maps, zhost_flag)
    return specid_maps, specid_flag, zhost_maps, zhost_flag


def _load_searcheff(config, simlib, filter_name_to_index, spec_zhost) -> dict:
    """Load all SEARCHEFF maps + pipeline state, as Survey field values."""
    pipeline = _load_searcheff_pipeline_curves(
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


def _prepare_field_data(simlib, model):
    """Numerical entry headers used by the batch; prepare them once."""
    fields = simlib.fields
    names = {name for name, _ in model._population} | {'PEAKMJD'}
    names.update(name for field in fields for name in field.header_values)
    gengauss_names = {name for field in fields for name in field.gengauss}
    redshift_windows = np.array([field.genrange_redshift for field in fields])
    if not (redshift_windows[:, 0] > 1.0e-5).any():
        redshift_windows = None
    libids = np.array([field.libid for field in fields], dtype=np.int64)
    prescales = simlib.field_prescales
    return {
        'ra': np.array([field.ra for field in fields], dtype=np.float32),
        'dec': np.array([field.dec for field in fields], dtype=np.float32),
        'keep': (prescales > 0) & (libids % np.maximum(prescales, 1) == 0),
        'redshift_windows': redshift_windows,
        'headers': {name: header_column(fields, name) for name in sorted(names)},
        'gengauss': {
            name: np.array([field.gengauss.get(name, (np.nan, np.nan, 0.0))
                            for field in fields]) for name in sorted(gengauss_names)},
    }


def prepare_photometry_data(simulator) -> Dict:
    """The numerical inputs of the photometry kernels, built once per run
    setup: filter/AB arrays from the shared photometry loader, dust laws,
    cosmology scalars, and the model's own contribution."""
    logger.info("Preparing numerical tables...")

    photometry = simulator.photometry

    static_data = {
        'wavelength_grid': simulator.wavelength_grid,
        'filter_transmissions_array': photometry.transmissions_stacked,
        'filter_ref_fluxes': photometry.filter_ref_fluxes,
        'filter_ref_fluxes_snana': photometry.filter_ref_fluxes_snana,
        'filter_ref_mag_terms': photometry.filter_ref_mag_terms,
        'filter_ref_mag_terms_snana': photometry.filter_ref_mag_terms_snana,
        'mw_dust_law': simulator.mw_law,
        'rv_mw': simulator.config.mw_dust.rv,
        'host_dust_law': simulator.host_dust_law,
        'latetime_extrap': True,
        'zp_fluxcal': simulator.config.instrument.zp_fluxcal,
        'filter_mean_wavelengths': photometry.filter_mean_wavelengths,
    }

    omega_m, omega_k, omega_l, hubble_distance, w0, wa = cosmo_params(
        simulator.config.cosmology)
    static_data.update({
        'omega_m': omega_m, 'omega_l': omega_l, 'omega_k': omega_k,
        'hubble_distance': hubble_distance, 'w0': w0, 'wa': wa,
    })

    static_data['host_rv'] = simulator.model.host_rv
    static_data['fluxcal_scale_mag_offset'] = (
        simulator.model.fluxcal_scale_mag_offset)

    # the band sum runs over a window of the wavelength grid
    static_data['band_window'] = band_window_width(
        simulator.wavelength_grid,
        photometry.common_wave_grid,
        photometry.common_trans_grid)

    slog.line(logger, "Numerical tables ready")
    return static_data


def observation_window_days(simulator) -> float:
    """The widest MJD span the scheduler can ask one SIMLIB entry for.

    An event keeps the epochs between ``peak + Trest_min*(1+zHEL)`` and
    ``peak + Trest_max*(1+zHEL)``, GENRANGE_TOBS_PAD wider at each end, so
    the span follows the top of the generated redshift range. zHEL sits
    above the generated zCMB by the CMB dipole and the peculiar velocity;
    SNANA allows for both with a five-sigma margin on GENSIGMA_VPEC when it
    sizes its own redshift grid (snlc_sim.c:26993-26998), and the same
    margin bounds the window here, and HOSTLIB USEVPEC (MSKOPT bit 512),
    which replaces the drawn velocity with the host's, contributes the
    library's largest. SN2GAL_Z rebinds the redshift to the host's, which
    can lie above that range, so the library's top redshift joins the
    maximum; a SIMLIB REDSHIFT or DISTANCE window is clipped to
    GENRANGE_REDSHIFT and stays inside it.
    """
    survey = simulator.config.survey
    cosmo = simulator.config.cosmology
    trest_min, trest_max = simulator.phase_range
    z_max = simulator.config.rates.redshift_range[1]
    hostlib, mskopt = simulator.hostlib, simulator.config.host.mskopt
    if hostlib is not None and mskopt & 16:
        z_max = max(z_max, hostlib.z_max)

    vpec = 5.0 * cosmo.gensigma_vpec
    if hostlib is not None and mskopt & 512:
        vpec = max(vpec, hostlib.vpec_max)
    dz = (cosmo.vel_cmbapex + vpec) / LIGHT_KM
    return ((trest_max - trest_min) * (1.0 + z_max) * (1.0 + dz)
            + 2.0 * survey.genrange_tobs_pad)


def build_scheduler(simulator):
    """The compiled observation scheduler over padded per-field arrays,
    the arrays it takes and the per-event slot count it was built for."""
    trans = np.array(
        [simulator.filter_name_to_index.get(str(c), -1)
         for c in simulator.simlib_filter_names], dtype=np.int32)
    field_cache = build_jax_field_cache(simulator, filter_translation=trans)
    survey = simulator.config.survey
    max_obs = max_obs_per_event(
        field_cache, observation_window_days(simulator),
        mskopt=survey.simlib_mskopt)
    tables = {
        **scheduler_tables(field_cache),
        'nskip_prescale': jnp.asarray([e[0] for e in survey.simlib_nskipmjd], dtype=jnp.int32),
        'nskip_trestmin': jnp.asarray([e[1] for e in survey.simlib_nskipmjd], dtype=jnp.float32),
    }
    return tables, max_obs


def population_before_host(simulator):
    """``(model, dust)``: which parts of the population are drawn before the
    host rather than after it.

    SNANA draws the host first, except that when the HOSTLIB weight map reads
    SN variables the model's parameters must exist before the host is chosen
    (snlc_sim.c:12889-12891, :12899-12900); host dust the map reads is drawn
    first with them. A population that reads the host in turn, through a
    conditional GENPDF or a VARIATION on a host column, cannot then be
    drawn, so that combination aborts.
    """
    if simulator.hostlib is None or not simulator.hostlib.selector.has_snvars:
        return False, False
    selector = simulator.hostlib.selector
    has_conditional_genpdf = any(
        pop.genpdf is not None and pop.genpdf.cond_varnames
        for pop in simulator.config.model.populations.values())
    # only a shift on a host column needs the host; one depending on the
    # redshift or an already-drawn parameter reads none
    drawn = {name for name, _ in simulator.model._population} | {REDSHIFT}
    has_variation = any(
        name not in drawn
        for name in simulator.config.model.quantities_depended_on())
    if has_conditional_genpdf or has_variation:
        cause = 'a conditional GENPDF' if has_conditional_genpdf else 'a VARIATION'
        raise SNANAError(
            f"Cannot combine {cause} on a host column with WGTMAP SN "
            "variables (e.g., x1): the weight map needs the SN drawn before "
            "its host, and the population needs the host first.")
    dust = {p.key for p in HOST_DUST_PARAMETERS}
    return True, any(name in dust for name in selector.snvar_names)


def build_atmosphere_static(simulator):
    """Site geo, cal-star SED and per-filter transmission on the shared
    photometry grid for the DCR integral (INIT_ATMOSPHERE)."""
    atm = simulator.config.instrument
    site = simulator.site
    if site is None:
        raise SNANAError(
            f"ATMOSPHERE: survey '{simulator.survey_name}' has no "
            f"geo: entry in SURVEY.DEF")

    photom = simulator.photometry
    wave = np.asarray(photom.common_wave_grid, dtype=np.float64)
    name_to_row = {n: i for i, n in enumerate(list(photom.filter_names))}
    rows = [name_to_row[n] for n in simulator.filter_names]
    trans_by_filter = np.asarray(photom.common_trans_grid, dtype=np.float64)[rows]

    star_path = Path(os.path.expandvars(atm.atmos_sedstar_file)).expanduser()
    star = np.loadtxt(star_path)
    cal_flux = np.interp(wave, star[:, 0], star[:, 1], left=0.0, right=0.0)

    return site, wave, trans_by_filter, cal_flux


def simlib_take_spectrum_entries(fields):
    """Parse each LIBID's TAKE_SPECTRUM header lines, with the aborts.

    SNANA runs these through the very same parse_input_TAKE_SPECTRUM as the
    input-file key (snlc_sim.c:22269), so the grammar is shared.
    """
    out = []
    for f in fields:
        entries = []
        for line in f.take_spectrum:
            if line.startswith('WARP_SPECTRUM'):
                raise SNANAError(
                    f"WARP_SPECTRUM in SIMLIB LIBID {f.libid} is not implemented.",
                    function_name="simlib_take_spectrum_entries")
            key_part, _, value_part = line.partition(':')
            # a SIMLIB entry is one spectrum: MJD(min:max:step) is refused
            entry = parse_take_spectrum_entry(key_part.strip(), value_part.strip(),
                                              from_simlib=True)
            if entry['epoch_frame'] == 'HOST':
                raise SNANAError(
                    f"HOST TAKE_SPECTRUM in SIMLIB LIBID {f.libid} needs host SEDs, "
                    f"which jax does not model.",
                    function_name="simlib_take_spectrum_entries")
            entries.append(entry)
        out.append(prepare_spectrum_requests(entries))
    return out


def _prepare_kernels(simulator) -> LightCurveSimulator:
    """Prepare numerical arrays and structural options for the shared kernels."""
    simulation = simulator.config
    config = simulation.numerical

    n_sne = simulator.n_simulated if config.n_sne is None else config.n_sne
    batch_size = (field_default(NumericalSettings, "batch_size")
                  if config.batch_size is None else config.batch_size)

    all_mjds = np.asarray(simulator.flat_obs_data['mjds'])
    survey_mjd_span = (
        float(all_mjds.max() - all_mjds.min()) if all_mjds.size else 0.0)

    # the spans GALextinct is evaluated over (fill_TABLE_MWXT_SEDMODEL)
    kcor = simulator.kcor
    spans = []
    for name in simulator.filter_names:
        waves, trans = kcor.get_filter_transmission(name)
        waves = np.asarray(waves)[np.asarray(trans) > 0.0]
        spans.append((float(waves.min()), float(waves.max())))
    calib = simulator.calibration
    if calib is not None:
        spans.append((float(np.min(np.asarray(calib.lam_min))),
                      float(np.max(np.asarray(calib.lam_max)))))
    simulator.mw_law.check_wave_range(
        min(lo for lo, _ in spans), max(hi for _, hi in spans))

    if simulator.model.forced_ngen is not None:
        n_sne = simulator.n_simulated

    s = simulation.survey
    if s.simlib_mskopt & 4:
        cap = len(simulator.fields) * s.simlib_nrepeat
        if n_sne > cap:
            raise SNANAError(
                f"SIMLIB_MSKOPT bit 4 (QUIT_NOREWIND) caps n_sne at "
                f"{cap} (n_fields={len(simulator.fields)} * "
                f"nrepeat={s.simlib_nrepeat}); got n_sne={n_sne}. "
                f"Lower NGENTOT_LC / NGEN_SCALE or unset bit 4.",
                function_name="_prepare_kernels",
            )

    # only the three flux-max OPT_SETPKMJD estimators exist (snlc_sim.c:17101-17156)
    opt_setpkmjd = simulation.rates.opt_setpkmjd
    if opt_setpkmjd > 0 and not opt_setpkmjd & (8 | 16 | 32):
        raise SNANAError(
            f"OPT_SETPKMJD {opt_setpkmjd} sets none of bits 8, 16 or 32. "
            f"SNANA still overwrites PEAKMJD in that case; jax implements "
            f"only the three estimators those bits select.",
            function_name="_prepare_kernels")
    model_before_host, dust_before_host = population_before_host(simulator)

    zp_fluxcal = simulation.instrument.zp_fluxcal
    static_data = prepare_photometry_data(simulator)
    band_window = static_data.pop("band_window")
    static_data.pop("latetime_extrap")
    noise, noise_options = prepare_noise(simulator, zp_fluxcal)
    tables, max_obs = build_scheduler(simulator)
    slog.line(logger, f"Observation padding: {max_obs} epochs per event")
    search_efficiency = prepare_search_efficiency(simulator)
    cutwin, cutwin_options = build_cutwin(simulator, simulation.cuts)
    season_boundaries = prepare_season_boundaries(simulator)
    photprob_prepared = None
    if search_efficiency is not None and simulator.searcheff_photprob_maps:
        photprob_prepared = prepare_photprob(
            simulator.searcheff_photprob_maps,
            list(simulator.photometry.filter_names))

    spec = simulation.spectra
    makes_spectra = bool(
        spec.take_spectrum
        or simulation.survey.use_simlib_spectrograph
        or simulation.survey.use_simlib_take_spectrum
        or simulation.effective_spectrograph_optmask & 128)   # REQUEST_USER_SPECTRA
    # a SPECEFF map on the spectrum's SNR needs the spectra, whichever
    # source requests them
    specid_uses_snrsum_rest_v = bool(
        simulator.calibration is not None and makes_spectra
        and simulator.searcheff_specid_maps
        and any('SNRSUM_REST_V' in [v.upper() for v in m.varnames]
                for m in simulator.searcheff_specid_maps))
    if makes_spectra and simulator.calibration is None:
        raise SNANAError(
            "Cannot generate spectra because SPECTROGRAPH is not defined in the "
            "kcor file.",
            function_name="_prepare_kernels")
    simlib_take_entries = (
        simlib_take_spectrum_entries(simulator.fields)
        if simulation.survey.use_simlib_take_spectrum else None)
    atmosphere_static = None
    if simulation.instrument.atmos_optmask > 0:
        atmosphere_static = build_atmosphere_static(simulator)

    slog.line(logger, "Simulator ready")
    simulator.sfd_data = (
        simulator.sfd_map.device_data()
        if simulator.sfd_map is not None and simulation.host.mskopt & 8
        else None)
    simulator.photometry_data = static_data
    simulator.calibration_filter_names = tuple(simulator.kcor.get_filter_names())
    simulator.noise = noise
    simulator.noise_options = noise_options
    simulator.zeropoint_error_override = per_filter_array(
        simulator, simulation.instrument.fudge_zpterr, 0.0)
    simulator.band_window = band_window
    simulator.max_obs = max_obs
    simulator.scheduler_tables = tables
    simulator.search_efficiency = search_efficiency
    simulator.cutwin = cutwin
    simulator.cutwin_options = cutwin_options
    simulator.season_boundaries = season_boundaries
    simulator.photprob_prepared = photprob_prepared
    simulator.survey_mjd_span = survey_mjd_span
    simulator.model_before_host = model_before_host
    simulator.dust_before_host = dust_before_host
    simulator.specid_uses_snrsum_rest_v = specid_uses_snrsum_rest_v
    simulator.makes_spectra = makes_spectra
    simulator.simlib_take_entries = simlib_take_entries
    simulator.atmosphere_static = atmosphere_static
    simulator.n_sne = n_sne
    simulator.batch_size = batch_size
    simulator.time_end_init = time.time()
    return simulator
