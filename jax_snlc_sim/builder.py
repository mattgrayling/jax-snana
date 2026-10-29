"""Load a survey and prepare one LightCurveSimulator.

The builder assembles the model, survey and prepared domain resources into a
complete simulator containing its configuration snapshot and numerical tables.
Random keys, batch counters and host reuse records belong to each run.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core import logging as log
from jax_snlc_core.cosmology import cosmo_params, distance_modulus_to_redshift
from jax_snlc_core.dust import SFDDustMap, load_dust_law
from jax_snlc_core.filters import (KcorReader, PhotometryResources, band_window_width,
                                   filter_band, load_photometry_resources, read_calib_shifts,
                                   resolve_calib_shifts)
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.models import (SimulatedModel, build_model, metadata_files,
                                  resolve_model_type)
from jax_snlc_core.utils import LIGHT_KM
from jax_snlc_sim.atmosphere import build_atmosphere_static, read_observatory_sites
from jax_snlc_sim.inputs import resolve_input_path
from jax_snlc_sim.spectra import (build_sed_true_spectrograph, load_spectrograph,
                                  simlib_take_spectrum_entries)
from jax_snlc_sim.scatter import build_smear_model, prepare_phase_scatter
from jax_snlc_sim.hostlib.association import load_hostlib
from jax_snlc_sim.lensing import read_lensing_probmap
from jax_snlc_sim.photometry import (load_fluxerr_model, per_filter_array,
                                      prepare_noise, prepare_photometry_data)
from jax_snlc_sim.population import (HOST_DUST_PARAMETERS, model_spellings,
                                     population_spellings, validate_config_populations)
from jax_snlc_sim.rate import build_murewgt, build_redshift_cdf, resolve_counts
from jax_snlc_sim.selection import build_cutwin, load_searcheff, prepare_search_efficiency
from jax_snlc_sim.simulation import (Instrument, LightCurveSimulator, NumericalSettings,
                                     field_default)
from jax_snlc_sim.spectroscopic_efficiency import read_specid_zhost
from jax_snlc_sim.detection import prepare_photprob
from jax_snlc_sim.simlib import (
    InputGenranges, SimlibData, append_spectrograph_photometry,
    rebuild_flat_observations, apply_forceval_psf, apply_ignore_template_noise,
    apply_noise_scales, apply_noise_scales_per_filter, apply_simlib_filters,
    apply_zpt_shift, build_jax_field_cache, header_column, max_obs_per_event,
    prepare_field_data, prepare_season_boundaries, read_simlib, resolve_survey,
    scheduler_tables, simlib_job_start, simlib_redshift_windows, spell_entry_headers,
    validate_simlib_observations)
from jax_snlc_sim.zvariation import REDSHIFT

logger = logging.getLogger(__name__)


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
    simlib_windows = simlib_redshift_windows(simlib, config.rates.redshift_range)
    survey_name, survey_id = resolve_survey(
        simlib.survey_name, config.survey.survey_override)
    kcor = _load_kcor(config)
    filter_names, name_to_index = _build_filter_lookup(kcor)

    dust_law = config.mw_dust.build_law()
    spec_calibration = load_spectrograph(config, kcor)
    (sed_model, genmodel_path, wavelength_grid,
     model_wave_range, filter_center_range) = _create_model(
        config, wavelength_grid_points, n_simulated, frac_pec1a,
        _working_wavelength_range(config, kcor, spec_calibration))
    names, aliases, _ = config.model.population_declaration()
    sed_model.configured_population = tuple((name, tuple(aliases.get(name, ())))
                                            for name in names)
    host_dust_law = sed_model.build_host_dust_law()
    if config.effective_spectrograph_optmask & 128:
        spec_calibration = build_sed_true_spectrograph(
            config, kcor, model_wave_range)
    if sed_model.forced_ngen is not None:
        n_simulated = sed_model.forced_ngen
        ngen_lc = 0
        logger.info(f"{type(sed_model).__name__} forces NGEN to {n_simulated}")

    spec_zhost = read_specid_zhost(config, model_spellings(sed_model))
    hostlib = load_hostlib(config, simlib, sed_model, spec_zhost, hz_map,
                            frac_pec1a)
    validate_config_populations(config, sed_model, hostlib)
    npe_pixel_saturate, photflag_saturate = \
        _resolve_saturation(config.instrument, simlib)
    fluxerr = load_fluxerr_model(config, simlib, kcor, hostlib)
    lensing_path = config.lensing.probmap_file
    lensing_probmap = (
        read_lensing_probmap(str(lensing_path), dmuscale=config.lensing.dmuscale)
        if lensing_path and str(lensing_path).upper() != 'NONE' else None)
    modelname = config.model.genmag_smear_modelname
    gensmear = None
    if modelname:
        gensmear = build_smear_model(
            modelname, salt_model_dir=Path(os.path.expandvars(genmodel_path)),
            sed_lam_max=(model_wave_range[1] if model_wave_range else None),
            zmin=config.rates.redshift_range[0])
        log.line(logger, f"GENMAG_SMEAR_MODELNAME {modelname} ready: "
                 f"{gensmear.n_randoms} randoms/SN")
    if config.survey.use_simlib_spectrograph:
        append_spectrograph_photometry(
            simlib, spec_calibration, kcor, config.spectra.spectrograph_scale_texpose)
    validate_simlib_observations(simlib)

    phase_range, filter_fields, photometry = _build_grids_and_filters(
        config, kcor, sed_model, wavelength_grid, filter_center_range,
        filter_names, name_to_index,
        dust_law,
    )

    searcheff_fields = load_searcheff(config, simlib, name_to_index,
                                       spec_zhost)

    field_mwebv = np.array(
        [f.mwebv for f in simlib.fields], dtype=np.float32)
    sfd_map, field_sfd_ebv = _load_field_sfd(config, simlib, field_mwebv)
    cdf_z, cdf_vals, cdf_vals_pec1a = build_redshift_cdf(config, hz_map)
    murewgt = build_murewgt(config, hz_map)
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
        field_data=jax.device_put(prepare_field_data(simlib, names)),
        field_start_index=simlib_job_start(config, len(simlib.fields)),
        n_simulated=n_simulated, ngen_lc=ngen_lc,
        season_count=season_count,
        _model=sed_model if registered_model else None,
        _static_model=None if registered_model else sed_model,
        wavelength_grid=wavelength_grid, phase_range=phase_range,
        model_wave_range=model_wave_range,
        filter_center_range=filter_center_range,
        gensmear=gensmear,
        survey_name=survey_name, survey_id=survey_id,
        site=read_observatory_sites().get(survey_name),
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
        phasecor_cholesky_L=prepare_phase_scatter(config),
        **fluxerr,
        lensing_probmap=lensing_probmap)
    return _prepare_kernels(simulator)


def _load_simlib(config) -> SimlibData:
    """Read a fresh SIMLIB and apply this run's observing conditions."""
    simlib_file = resolve_input_path(config.survey.simlib_file, config.run)
    s = config.survey
    spellings = population_spellings(*config.model.population_declaration()[:2])
    # EXPOSURE_TIME per-band ZPT shift merges additively with FUDGESHIFT_ZPT.
    exp_zpt, exp_sky, exp_read = _exposure_scales(config)
    zpt_map = {k: v for k, v in config.instrument.fudgeshift_zpt.items() if v != 0.0}
    for b, v in exp_zpt.items():
        zpt_map[b] = zpt_map.get(b, 0.0) + v
    noise_scales = (
        config.instrument.fudgescale_psf, config.instrument.fudgescale_noise_sky,
        config.instrument.fudgescale_noise_read,
        config.instrument.fudgescale_noise_template,
    )
    forceval_psf = config.survey.forceval_psf if config.survey.forceval_psf > 0.001 else 0.0
    ignore_template_noise = bool(s.simlib_mskopt & 16)
    log.banner(logger, "Read SIMLIB")
    log.line(logger, f"SIMLIB file: {simlib_file}")
    simlib = read_simlib(simlib_file, require_docana=bool(config.run.require_docana))
    spell_entry_headers(simlib, spellings)
    distance_z = _field_distance_redshift(config, simlib)
    apply_simlib_filters(simlib, s, InputGenranges(
        redshift=config.rates.redshift_range, peakmjd=config.rates.peakmjd_range,
        population=config.model._input_genranges,
        distance_redshift=(np.full(len(simlib.fields), np.nan) if distance_z is None
                           else np.asarray(distance_z))))
    if forceval_psf:
        apply_forceval_psf(simlib, forceval_psf)
    if zpt_map:
        apply_zpt_shift(simlib, zpt_map)
    apply_noise_scales(simlib, *noise_scales)
    if exp_sky or exp_read:
        apply_noise_scales_per_filter(simlib, exp_sky, exp_read)
    if ignore_template_noise:
        apply_ignore_template_noise(simlib)
    if (forceval_psf or zpt_map or exp_sky or exp_read
            or ignore_template_noise
            or any(v != 1.0 for v in noise_scales)):
        rebuild_flat_observations(simlib)
    log.line(logger, f"Loaded {len(simlib.fields)} SIMLIB fields")
    return simlib


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


def _load_kcor(config) -> KcorReader:
    """Load the filter calibration and restrict to GENFILTERS."""
    kcor_file = resolve_input_path(config.survey.kcor_file, config.run)
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
    log.line(logger, f"Loaded {len(kcor.get_filter_names())} filters")
    return kcor


def _build_filter_lookup(kcor) -> Tuple[List[str], Dict[str, int]]:
    """Build the filter name→index map.

    Returns ``(names, name_to_index)`` where `name_to_index`
    keys include both the canonical name (e.g., 'LSST-g') and its band letter
    ('g'). Mean wavelengths are produced by
    :func:`jax_snlc_core.filters.load_photometry_resources`
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


def _create_model(config, wavelength_grid_points: int,
                  n_simulated: int, frac_pec1a: float,
                  wavelength_range: Optional[Tuple[float, float]]
                  ) -> Tuple[Any, str, jnp.ndarray,
                             Tuple[float, float], Tuple[float, float]]:
    """Build the SN model through the shared construction path and apply the
    simulation-only sampling configuration.

    Returns ``(model, genmodel_path, wavelength_grid,
    model_wave_range, filter_center_range)`` with wave ranges reflecting any
    UVLAM extension and the model already resampled to the working grid,
    cut to ``wavelength_range``.
    """
    genmodel_path = config.model.genmodel
    model_type = resolve_model_type(genmodel_path, hint=config.model.model_type)
    log.banner(logger, f"Init {model_type.upper()} model")
    log.kv(logger, "GENMODEL", genmodel_path)

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

    log.kv(logger, "wave range",
           f"{model_wave_range[0]:.0f} - {model_wave_range[1]:.0f} A")
    log.kv(logger, "filter centre range",
           f"{filter_center_range[0]:.0f} - {filter_center_range[1]:.0f} A")
    log.line(logger, f"Wavelength grid: {float(wavelength_grid[0]):.0f} - "
             f"{float(wavelength_grid[-1]):.0f} Å "
             f"({len(wavelength_grid)} points)")

    return (model, genmodel_path, wavelength_grid,
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
    log.line(logger, f"Filter redshift limits: {len(filters)} filters")
    for filt in filters:
        zlo, zhi = z_limits[filt]
        log.line(logger, f"{filt}: {zlo:.3f} < z < {zhi:.3f}", indent=2)
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
    log.line(logger, f"Pre-interpolated {len(all_kcor_filter_names)} filters to common "
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
        log.line(logger, f"Per-template z-limits: {len(t_wl_min)} templates × "
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

    log.line(logger, f"Grids: {len(wavelength_grid)} wavelengths, realistic phase scheduling")

    return phase_range, filter_fields, photometry


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
    drawn = {name for name, _ in simulator.model.configured_population} | {REDSHIFT}
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
    band_window = band_window_width(
        simulator.wavelength_grid,
        simulator.photometry.common_wave_grid,
        simulator.photometry.common_trans_grid)
    noise, noise_options = prepare_noise(simulator, zp_fluxcal)
    tables, max_obs = build_scheduler(simulator)
    log.line(logger, f"Observation padding: {max_obs} epochs per event")
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

    log.line(logger, "Simulator ready")
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
