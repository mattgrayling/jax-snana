"""The SIMGEN_DUMP file and the SIMGEN_DUMP_* diagnostic dumps."""

import logging
import os
from pathlib import Path
from typing import Any, Dict

import jax.numpy as jnp
import numpy as np

from jax_snlc_core import logging as log
from jax_snlc_core.cosmology import eq_to_gal
from jax_snlc_core.dust import load_dust_law
from jax_snlc_core.filters import filter_band
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.noise import noise_components_pe
from jax_snlc_core.snana_files import (HOSTGAL_PROPERTY_BASENAMES, SNANA_NULLFLOAT,
                                       canonical_property_name)
from jax_snlc_sim.cutwin import CUTMASK_ALL_PASS
from jax_snlc_sim.output.tables import model_dump_columns, model_header_params, per_event_types
from jax_snlc_sim.rate import compute_rate_on_grid, apply_rate_modifiers

logger = logging.getLogger(__name__)


_MWCL_LAWS = (
    ("FITZ99_EXACT", "FITZ99_EXACT"),
    ("FITZ99_APPROX", "FITZ99_APPROX"),
    ("FITZ04", "FITZ04"),
    ("CCM89", "CCM89"),
    ("ODONNELL94", "ODON94"),
)

_NOISE_COMMENT = {
    1: "Dump only the observation (per band) closest to PEAKMJD",
    2: "Dump all observations with defined transient model mag",
}

_DUMP_FORMAT_OVERRIDES = {
    'PEAKMJD':          '%.3f',
    'MJD_TRIGGER':      '%.3f',
    'MJD_DETECT_FIRST': '%.3f',
    'MJD_DETECT_LAST':  '%.3f',
    'RA':               '%.6f',
    'DEC':              '%.6f',
}

_DUMP_EVENT_COLUMNS = (
    'cid', 'gentype', 'sntype', 'non1a_index', 'genz', 'libid', 'field',
    'ra', 'dec', 'mwebv', 'mwebvmap', 'mwebverr', 'mu', 'peakmjd',
    'cutmask', 'trestmin', 'trestmax', 'tgapmax', 'nobs')

# SNANA name -> parameter key (PREP_SIMGEN_DUMP, snlc_sim.c:16588, :16650-16695)
_DUMP_PARAM_COLUMNS = (
    ('SIM_SEARCHEFF_MASK', 'sim_searcheff_mask'),
    ('SIM_EFFMASK',        'sim_searcheff_mask'),  # older name, :16730
    ('MJD_TRIGGER',        'mjd_trigger'),
    ('MJD_DETECT_FIRST',   'mjd_detect_first'),
    ('MJD_DETECT_LAST',    'mjd_detect_last'),
    ('REDSHIFT_FINAL',     'redshift_final'),
    ('ZCMB_SMEAR',         'redshift_final'),
    ('REDSHIFT_FINAL_ERR', 'redshift_smear_err'),
    ('ZHELIO_ERR',         'redshift_smear_err'),
    ('FLAG_ACCEPT',        'passing_mask'),  # snlc_sim.c:416, 5061-5065
    ('NEPOCH',             'nepoch'),
    ('VPEC',               'vpec_true'),
    ('LENSDMU',            'lensdmu'),
    ('SPEC_SNRSUM_REST_V', 'snrsum_rest_v'),
    ('WGT_POPULATION',     'wgt_population'),  # snlc_sim.c:16417
    ('AV',                 'sim_av'),
    ('RV',                 'sim_rv'),
    ('ZHELIO',             'sim_redshift_helio'),
    ('SHIFT_RA',           'random_shift_RA'),
    ('SHIFT_DEC',          'random_shift_DEC'),
    ('SHIFT_RADIUS',       'random_shift_RADIUS'),
    ('SHIFT_PHI',          'random_shift_PHI'),
    ('GALID',              'drawn_GALID'),
    ('GALZTRUE',           'host_ZTRUE'),
    ('GALSNDM',            'host_SNMAGSHIFT'),
    ('GALWGT',             'host_WGTMAP_WGT'),
)

_DUMP_ALIASES = (
    ('SIM_TEMPLATE_INDEX', 'NON1A_INDEX'),
    ('NONIA_INDEX',        'NON1A_INDEX'),
    ('TMIN',               'TRESTMIN'),
    ('ZCMB',               'GENZ'),
    ('SNRMAX',             'SNRMAX1'),
    ('TYPE',               'SNTYPE'),
    ('MWEBV_TRUE',         'MWEBVMAP'),
)

_DUMP_SNRMAX_RANKS = 5

# refused names and their replacements (VARNAME_ABORT, snlc_sim.c:16658-16663)
_DUMP_REFUSED_NAMES = {'TMAX': 'TRESTMAX'}


def canonical_dump_columns(columns, model):
    """A loaded dump's columns, each alias also under the name the replay
    reads, whichever spelling the dump was written with."""
    targets = dict(_DUMP_ALIASES)
    targets.update((name.upper(), column)
                   for name, column in model.SNANA_DUMP_SIM_COLUMNS)
    out = dict(columns)
    for name, values in columns.items():
        if name.upper() in targets:
            out.setdefault(targets[name.upper()], values)
    return out


def cutwin_reject_rows(config, full_results, selection_result):
    """Rows for APPLY_CUTWIN_OPT=3: the accepted events plus the ones CUTWIN
    rejected (doReject_SIMGEN_DUMP, snlc_sim.c:16927). None when unset."""
    if (config.cuts.apply_cutwin_opt != 3
            or not full_results or not selection_result):
        return None
    codes = np.asarray(selection_result['rejection_codes'])
    return np.asarray(selection_result['passing_mask']) | (codes == 5)


def write_dump_dcr(columnar_data, version_name, output_dir, to_written_cid):
    """Write the per-obs <VERSION>.DCR table (ATMOSPHERE_OPTMASK bit 512,
    wr_SIMGEN_DUMP_DCR, snlc_sim.c:15127). One row per obs with SNR_OBS>3 and
    a valid DCR shift."""
    obs = columnar_data['observations']
    counts = np.asarray(columnar_data['obs_counts'])
    cid = np.repeat(to_written_cid(columnar_data['parameters']['cid']), counts)
    peak = np.repeat(np.asarray(columnar_data['parameters']['peakmjd']), counts)
    mjd = np.asarray(obs['mjd'])
    band = [filter_band(n) for n in
            np.asarray(columnar_data['filter_names'])[np.asarray(obs['filter_idx'])]]
    tobs = mjd - peak
    snr_obs = np.asarray(obs['snr_obs'])
    lamavg = np.asarray(obs['lamavg_sed_wgted'])
    psf_fwhm = np.asarray(obs['psf_fwhm'])
    fluxcal = np.asarray(obs['fluxcal'])
    airmass = np.asarray(obs['airmass'])
    d_ra = np.asarray(obs['d_ra'])
    d_dec = np.asarray(obs['d_dec'])
    coordres = np.asarray(obs['coordres'])
    snr_true = np.asarray(obs['snr_true'])
    sim_dcr = np.asarray(obs['sim_dcr'])
    sim_dcr_dra = np.asarray(obs['sim_dcr_dra'])
    sim_dcr_ddec = np.asarray(obs['sim_dcr_ddec'])
    sim_dcr_dmag = np.asarray(obs['sim_dcr_dmag'])
    cols = ("CID MJD BAND LAMAVG_SED_WGTED SNR PSF_FWHM FLUXCAL TOBS AIRMASS "
            "dRA dDEC COORDRES SIM_SNR_TRUE SIM_DCR SIM_dRA SIM_dDEC SIM_dMAG")
    path = Path(output_dir) / f'{version_name}.DCR'
    with open(path, 'w') as fp:
        fp.write("# Simulation SUMMARY: one row per observation.\n")
        fp.write("# RA,DEC in degrees\n")
        fp.write("# dRA   = RA_OBS  - band-average RA_OBS  (arcsec)\n")
        fp.write("# dDEC  = DEC_OBS - band-average DEC_OBS (arcsec)\n")
        fp.write("# COORDRES = computed RA,DEC resolution (asec)( from PSF/SNR\n")
        fp.write("#\n")
        fp.write(f"VARNAMES: {cols}\n")
        for i in range(len(mjd)):
            if snr_obs[i] < 3.0 or sim_dcr_dra[i] > 90.0:
                continue
            fp.write(
                f"ROW: {int(cid[i]):6d} {mjd[i]:.4f} {band[i]} "
                f"{lamavg[i]:.1f} {snr_obs[i]:5.1f} "
                f"{psf_fwhm[i]:5.3f} {fluxcal[i]:10.4e} "
                f"{tobs[i]:5.1f}  {airmass[i]:.2f}  "
                f"{d_ra[i]:7.4f} {d_dec[i]:7.4f}  "
                f"{coordres[i]:5.3f} {snr_true[i]:5.1f} "
                f"{sim_dcr[i]:7.4f} {sim_dcr_dra[i]:7.4f} "
                f"{sim_dcr_ddec[i]:7.4f} {sim_dcr_dmag[i]:7.4f}  \n")


def write_dump_rate(config, version_name, output_dir):
    """SNANA SIMGEN_DUMP_RATE (wr_SIMGEN_DUMP_RATE, snlc_sim.c:15622): genz_wgt
    on a 0.01 z-grid over GENRANGE_REDSHIFT: the volumetric rate times the
    full DNDZ reweight/scale chain (genz_wgt, snlc_sim.c:17791-17851)."""
    z0, z1 = config.rates.redshift_range
    dndz_params = config.dndz_params()
    zs = []
    z = float(z0)
    while z <= float(z1):
        zs.append(z)
        z += 0.01
    zarr = np.asarray(zs, dtype=np.float64)
    rate = np.zeros_like(zarr)
    for p in dndz_params:
        in_range = (zarr >= p['zmin']) & (zarr <= p['zmax'])
        if np.any(in_range):
            rate[in_range] += np.asarray(
                compute_rate_on_grid(zarr[in_range], p))
    rate = apply_rate_modifiers(rate, zarr, config.rates.dndz_rate_mods)
    path = Path(output_dir) / f'{version_name}.RATE'
    with open(path, 'w') as fp:
        fp.write("# RATE = Volumetric Rate (per year per Mpc^3) w/ H0 = 70.0 \n")
        fp.write("VARNAMES: ROW REDSHIFT RATE \n")
        for nrow, (zv, rv) in enumerate(zip(zs, rate), start=1):
            fp.write(f"ROW: {nrow:3d}  {zv:.3f}  {rv:.4e} \n")
    log.line(logger, f"wrote {path}", indent=2)
    return path


def write_dump_mwcl(version_name, output_dir):
    """SNANA SIMGEN_DUMP_MWCL (wr_SIMGEN_DUMP_MWCL, snlc_sim.c:15681): MW mag
    extinction A(λ)=AV·A(λ)/A(V) (AV=0.1 fixed) over an RV×WAVE grid for 5
    colour laws."""
    RV_MIN, RV_MAX, RV_BIN = 1.0, 6.0, 1.0
    AV = 0.1
    waves = np.arange(1000.0, 25000.0, 100.0)  # SNANA: WAVE < WAVE_MAX (24900 last)
    wj = jnp.asarray(waves)
    laws = []
    for regname, label in _MWCL_LAWS:
        law = load_dust_law(regname)
        law.prepare_wavelength_grid(waves)  # FITZ99-family precompute; no-op otherwise
        laws.append((label, law))
    path = Path(output_dir) / f'{version_name}.MWCL'
    with open(path, 'w') as fp:
        fp.write("# MW mag extinction vs. wavelength for RV grid and several color laws,\n")
        fp.write(f"# RV grid: {RV_MIN:.2f} to {RV_MAX:.2f},  {RV_BIN:.2f} bin size \n")
        fp.write(f"# AV = {AV:.2f} is fixed\n")
        fp.write("\nVARNAMES:  ROW  RV WAVE   "
                 + " ".join(f"MAG_MWXT_{lbl}" for lbl, _ in laws) + "\n")
        row = 0
        rv = RV_MIN
        while rv <= RV_MAX:
            fp.write(f"\n# Next RV = {rv:.2f}\n")
            mags = [np.asarray(AV * law.extinction_curve(wj, float(rv)))
                    for _, law in laws]
            for k, w in enumerate(waves):
                row += 1
                vals = " ".join(f"{mags[j][k]:7.3f}" for j in range(len(laws)))
                fp.write(f"ROW: {row:6d} {rv:.2f} {w:7.1f}    {vals} \n")
            rv += RV_BIN
    log.line(logger, f"wrote {path}", indent=2)
    return path


def write_dump_trainsalt(columnar_data, version_name, output_dir, to_written_cid):
    """SNANA SIMGEN_DUMP_TRAINSALT (wr_SIMGEN_DUMP_TRAINSALT, snlc_sim.c:15559):
    one row per SN with the estimated PEAKMJD (PEAKMJD_SMEAR = the reported
    peak_mjd) and the true PEAKMJD. SNANA hard-wires ADD_PKMJD_TRUE=true."""
    params = columnar_data['parameters']
    cid = to_written_cid(params['cid']).astype(np.int64)
    pkmjd = np.asarray(params['peakmjd'], dtype=np.float64)
    pkmjd_true = np.asarray(
        params['peakmjd_true'], dtype=np.float64)
    path = Path(output_dir) / f'{version_name}.TRAINSALT'
    with open(path, 'w') as fp:
        fp.write("#SNID  PKMJD  PKMJD_TRUE\n")
        for i in range(len(cid)):
            fp.write(f"{int(cid[i]):8d}  {pkmjd[i]:.4f}  {pkmjd_true[i]:.4f}\n")
    log.line(logger, f"wrote {path}", indent=2)
    return path


def noise_dump_rows(config, results, selection, global_start_idx):
    """One batch's rows of the SIMGEN_DUMP_NOISE table, before the batch
    trims its epochs: the Npe noise breakdown of every kept observation
    (mode 1 the epoch nearest peak per band, mode 2 every one), with the
    event's index and the columns the writer takes from it. SNANA fills
    this dump inside the noise routine, before the trigger and CUTWIN
    stages, so rows cover every event that reached flux generation
    (snlc_sim.c:286-300)."""
    params = results['parameters']
    sched = results['observation_schedule']
    n_sne = len(np.asarray(params['cid']))
    reached_flux = np.asarray(selection['reached_flux'], dtype=bool)
    valid = np.asarray(selection['valid_mask'], dtype=bool) & reached_flux[:, None]
    mjds_2d = np.asarray(sched['mjds'], dtype=np.float64)
    fi_2d = np.asarray(sched['filter_indices'])
    mag_2d = np.asarray(results['magnitudes'], dtype=np.float64)
    keep = valid & (mag_2d < 90.0)
    if config.output.simgen_dump_noise == 1:
        keep &= _nearest_peak_per_band(
            keep, fi_2d, mjds_2d,
            np.asarray(params['peakmjd'], dtype=np.float64))
    sn_idx, obs_idx = np.nonzero(keep)
    # the host mag under the SN, GALMAG at aperture 0 in the epoch's filter;
    # 666 is SNANA's no-host sentinel
    galmag = results['host_galmag_total']
    if galmag is not None:
        mag_gal = np.asarray(galmag)[sn_idx, np.clip(fi_2d[sn_idx, obs_idx], 0, None)]
    else:
        mag_gal = np.full(len(sn_idx), 666.0)
    galsnsep = np.asarray(params.get('host_SNSEP', np.zeros(n_sne)),
                          dtype=np.float64)[sn_idx]
    flux_pe, sqerr_sky, sqerr_ccd, _ = noise_components_pe(
        np.asarray(sched['zptavgs'], dtype=np.float64)[sn_idx, obs_idx],
        np.asarray(sched['gains'])[sn_idx, obs_idx],
        np.asarray(results['fluxcals'])[sn_idx, obs_idx],
        np.asarray(sched['neas'])[sn_idx, obs_idx],
        np.asarray(sched['skysigs'])[sn_idx, obs_idx],
        np.asarray(sched['readnoises'])[sn_idx, obs_idx],
        config.instrument.zp_fluxcal)
    return {
        'event': sn_idx + global_start_idx,
        'cid': np.asarray(params['cid'], dtype=np.int64)[sn_idx],
        'galid': np.asarray(params.get(
            'drawn_GALID', np.full(n_sne, config.host.galid_null)),
            dtype=np.int64)[sn_idx],
        'libid': np.asarray(sched['libid'], dtype=np.int64)[sn_idx],
        'zcmb': np.asarray(params['sim_redshift_cmb'], dtype=np.float64)[sn_idx],
        'galsnsep': np.where(galsnsep < 0, 0.0, galsnsep),  # SNANA no-host GALSNSEP = 0
        'mag_gal': mag_gal,
        'mjd': mjds_2d[sn_idx, obs_idx],
        'band': fi_2d[sn_idx, obs_idx],
        'mag_sn': mag_2d[sn_idx, obs_idx],
        'zp_pe': np.asarray(sched['zptavgs'], dtype=np.float64)[sn_idx, obs_idx],
        'nea': np.asarray(sched['neas'], dtype=np.float64)[sn_idx, obs_idx],
        'cov_sn': np.asarray(flux_pe, dtype=np.float64),
        'cov_sky': np.asarray(sqerr_sky, dtype=np.float64),
        'cov_read': np.asarray(sqerr_ccd, dtype=np.float64),
        'cov_gal': np.asarray(results['fluxgal_pe'], dtype=np.float64)[sn_idx, obs_idx],
    }


def write_dump_noise(full_results, config, resources, version_name,
                     output_dir):
    """SNANA SIMGEN_DUMP_NOISE (wr_SIMGEN_DUMP_NOISE, snlc_sim.c:15236),
    from the rows the batches collected (`noise_dump_rows`). The CID is
    GENLC.CID, CIDOFF plus the event's (:15362)."""
    rows = full_results['noise_rows']
    mode = config.output.simgen_dump_noise
    cid = rows['cid'] + config.output.cidoff
    galid, libid, zcmb = rows['galid'], rows['libid'], rows['zcmb']
    galsnsep, mag_gal = rows['galsnsep'], rows['mag_gal']
    mjd = rows['mjd']
    filter_names = list(resources.filter_names)
    band = [filter_band(filter_names[k]) for k in rows['band']]
    mag_sn, zp_pe, nea = rows['mag_sn'], rows['zp_pe'], rows['nea']
    cov_sn, cov_sky, cov_read, cov_gal = (
        rows['cov_sn'], rows['cov_sky'], rows['cov_read'], rows['cov_gal'])
    cov_tot = cov_sn + cov_gal + cov_sky + cov_read
    simlib = config.survey.simlib_file
    hostlib = config.host.hostlib_file or ''
    path = Path(output_dir) / f'{version_name}.NOISE'
    with open(path, 'w') as fp:
        fp.write(f"# Simulation NOISE dump with SIMGEN_DUMP_NOISE={mode} --> \n")
        fp.write(f"#      {_NOISE_COMMENT.get(mode, '')}\n#\n")
        fp.write(f"# GALID identifies galaxy  in {hostlib}\n")
        fp.write(f"# LIBID identifies cadence in {simlib}\n")
        fp.write("# COV_XXX units are Npe. \n")
        fp.write("# COV_SKY and COV_GAL are summed over NEA.\n\n")
        fp.write("VARNAMES: CID GALID LIBID MJD BAND ZCMB   "
                 "MAG_SN MAG_GAL GALSNSEP ZP_pe NEA "
                 "COV_TOT COV_SN COV_GAL COV_SKY COV_READ\n")
        for i in range(len(mjd)):
            fp.write(
                f"SN: {int(cid[i]):8d} {int(galid[i]):8d} {int(libid[i]):4d} {mjd[i]:.4f} "
                f"{band[i]} {zcmb[i]:.3f}   {mag_sn[i]:.3f} {mag_gal[i]:.3f} {galsnsep[i]:.2f} {zp_pe[i]:.3f} "
                f"{nea[i]:.1f}  {cov_tot[i]:9.2f} {cov_sn[i]:9.2f} {cov_gal[i]:9.2f} "
                f"{cov_sky[i]:9.2f} {cov_read[i]:9.2f} \n")
    log.line(logger, f"wrote {path}", indent=2)
    return path


def _nearest_peak_per_band(keep, fi_2d, mjds_2d, peak):
    """(n_sne, max_obs) mask keeping, per (SN, band), the kept obs whose MJD
    is nearest that SN's peak."""
    dt = np.where(keep, np.abs(mjds_2d - peak[:, None]), np.inf)
    out = np.zeros_like(keep)
    n_filters = int(fi_2d.max()) + 1 if fi_2d.size else 0
    for k in range(n_filters):
        dt_k = np.where(fi_2d == k, dt, np.inf)
        best = np.argmin(dt_k, axis=1)
        has = np.isfinite(dt_k[np.arange(len(best)), best])
        out[np.arange(len(best))[has], best[has]] = True
    return out


def _build_peak_magnitude_dict(kcor_filter_names, peak_magnitudes,
                               selection_result,
                               include_all: bool) -> Dict[str, np.ndarray]:
    """Return peak magnitude arrays for each filter."""
    filter_names = (sorted(peak_magnitudes) if peak_magnitudes
                    else list(kcor_filter_names))
    peak_mags = {}
    if include_all:
        peakmags_all = np.asarray(selection_result['peakmags_masked'])
        # -9 for events rejected before flux generation (snlc_sim.c:12992, :9639)
        reached_flux = np.asarray(selection_result['reached_flux'], dtype=bool)
        name_to_idx = {fn: i for i, fn in enumerate(kcor_filter_names)}
        for filt in filter_names:
            column = peakmags_all[:, name_to_idx[filt]].astype(np.float32)
            peak_mags[filt] = np.where(reached_flux, column,
                                       np.float32(SNANA_NULLFLOAT))
        return peak_mags

    for filt in filter_names:
        peak_mags[filt] = peak_magnitudes[filt]
    return peak_mags


def epoch_statistics(selection_result, kcor_filter_names):
    """gen_cutwin's per-event statistics as the selection carried them out
    of its kernel (snlc_sim.c:24197-24409), with the per-band SNR maxima
    also in descending order for SNRMAX, SNRMAX2, SNRMAX3."""
    band_max = np.asarray(selection_result['snrmax_band'])
    return {
        'trestmin': np.asarray(selection_result['trestmin']),
        'trestmax': np.asarray(selection_result['trestmax']),
        'tgapmax': np.asarray(selection_result['tgapmax']),
        'nobs': np.asarray(selection_result['nobs']),
        'snrmax_sorted': -np.sort(-band_max, axis=1),
        'snrmax_by_band': {f: band_max[:, i]
                           for i, f in enumerate(kcor_filter_names)},
    }


def _take_rows(value, keep):
    """``value`` restricted to the rows ``keep`` marks, through nested dicts."""
    if isinstance(value, dict):
        return {k: _take_rows(v, keep) for k, v in value.items()}
    return np.asarray(value)[keep]


def _prepare_dump_data(columnar_data, config, resources, full_results,
                       kcor_filter_names, model, peak_magnitudes,
                       selection_result, include_all: bool, to_written_cid,
                       rows=None) -> Dict[str, Any]:
    """Assemble all arrays needed to serialise the dump file."""
    if include_all:
        if full_results is None or selection_result is None:
            raise SNANAError(
                "SIMGEN_DUMPALL requested but full simulation results are "
                "unavailable.", function_name='_prepare_dump_data')
        params_raw = full_results['parameters']
        params = {k: np.asarray(v) for k, v in params_raw.items()}
        n_events = len(params['sim_redshift_cmb'])
        obs_schedule = full_results['observation_schedule']

        libids = np.asarray(obs_schedule['libid'], dtype=np.int32)
        if len(libids) != n_events:
            raise SNANAError(
                f"observation schedule 'libid' has {len(libids)} entries "
                f"for {n_events} events", function_name='_prepare_dump_data')

        metadata = {
            'ra': params['ra'].astype(np.float64),
            'dec': params['dec'].astype(np.float64),
            'field_id': libids
        }
        keep_events = slice(None)
        specid_pass_dump = np.asarray(selection_result['specid_pass'],
                                      dtype=bool)
    else:
        params = columnar_data['parameters']
        metadata = columnar_data['metadata']
        n_events = len(params['sim_redshift_cmb'])
        keep_events = np.asarray(selection_result['passing_mask'], dtype=bool)
        specid_pass_dump = np.asarray(params['specid_pass'], dtype=bool)

    stats = _take_rows(epoch_statistics(selection_result, kcor_filter_names),
                       keep_events)
    band_tables = {'PEAKMAG_': None, 'SNRMAX_': stats['snrmax_by_band']}
    galmag = full_results['host_galmag_total']
    if galmag is not None:
        # HOSTLIB columns (snlc_sim.c:16255-16284); aperture 0 is the total mag
        total_mag = _take_rows(galmag, keep_events)
        sb_fluxcal = _take_rows(full_results['host_sb_fluxcal_stacked'], keep_events)
        zp = config.instrument.zp_fluxcal
        band_tables['HOSTMAG_'] = {
            f: total_mag[:, i] for i, f in enumerate(kcor_filter_names)}
        band_tables['SBFLUX_'] = {
            f: sb_fluxcal[:, i] for i, f in enumerate(kcor_filter_names)}
        band_tables['SBMAG_'] = {
            f: zp - 2.5 * np.log10(sb_fluxcal[:, i])
            for i, f in enumerate(kcor_filter_names)}

    sn_ids = to_written_cid(params['cid']).astype(np.int64)
    # the HEAD's GENTYPE and SNTYPE (snlc_sim.c:16573-16584)
    sim_gentype, sntypes = per_event_types(
        config, resources, params, n_events, specid_pass_dump)
    gentypes = sim_gentype.astype(np.int32)
    # DUMPALL: gen_spectype runs only for (MASK & OPT) == OPT
    if include_all:
        mask_full = np.asarray(selection_result['sim_searcheff_mask'])
        apply_opt = config.survey.apply_searcheff_opt
        if apply_opt > 0:
            ovp_pass = (mask_full & apply_opt) == apply_opt
        else:
            ovp_pass = np.ones(n_events, dtype=bool)
        sntypes = np.where(ovp_pass, sntypes, 0).astype(np.int32)
    redshifts = np.array(params['sim_redshift_cmb'], dtype=np.float64)
    libids = metadata['field_id'].astype(np.int32)
    ras = metadata['ra'].astype(np.float64)
    decs = metadata['dec'].astype(np.float64)
    # MWEBV is GENLC.MWEBV_SMEAR (snlc_sim.c:16023)
    mwebvs = np.asarray(params['sim_mwebv'], dtype=np.float32)
    mwebv_map = np.asarray(params['mwebv'], dtype=np.float32)
    mwebv_errs = np.asarray(params['mwebv_err'], dtype=np.float32)
    # SIMLIB_HEADER.FIELD (snlc_sim.c:16008)
    field_name_table = np.array(
        [f.field_name for f in resources.fields], dtype='U20')
    field_names_per_sn = field_name_table[
        np.asarray(params['field_idx'], dtype=np.int32)]
    field_names_per_sn = np.where(
        field_names_per_sn == 'FIELD_NONAME', 'VOID', field_names_per_sn)
    distance_moduli = np.asarray(params['mu'], dtype=np.float64)
    model_params = {
        key: np.array(value) for key, value in model_header_params(config, model, params).items()
    }
    # GENLC.TEMPLATE_INDEX, 0 for a model without templates
    non1a_indices = (np.asarray(model_params['SIM_TEMPLATE_INDEX'], dtype=np.int32)
                     if 'SIM_TEMPLATE_INDEX' in model_params
                     else np.zeros(n_events, dtype=np.int32))
    peak_mjd = params['peakmjd_true'].astype(np.float64)
    band_tables['PEAKMAG_'] = _build_peak_magnitude_dict(
        kcor_filter_names, peak_magnitudes, selection_result, include_all)

    if include_all:
        cutmask_array = np.array(selection_result['cutmask'], dtype=np.int32)
    else:
        cutmask_array = np.full(n_events, CUTMASK_ALL_PASS, dtype=np.int32)

    # selection outputs are per generated event; subset them unless DUMPALL
    params = dict(params)
    if selection_result is not None:
        keep_sel = (None if include_all
                    else np.asarray(selection_result['passing_mask'], dtype=bool))
        for key in ('sim_searcheff_mask', 'mjd_trigger', 'mjd_detect_first',
                    'mjd_detect_last', 'redshift_final', 'passing_mask',
                    'nepoch'):
            if key not in selection_result:
                continue
            col = np.asarray(selection_result[key])
            params[key] = col if keep_sel is None else col[keep_sel]
        if 'passing_mask' in params:
            params['passing_mask'] = params['passing_mask'].astype(np.int32)
    # SNANA's initial values where the run makes none: no lensing map
    # (GENLC.LENSDMU) and no spectra (GENSPEC.SNRSUM_REST_V, snlc_sim.c:9753)
    for key in ('lensdmu', 'snrsum_rest_v'):
        params.setdefault(key, np.zeros(n_events, dtype=np.float32))

    data = {
        'n_events': n_events,
        'params': params,
        'cid': sn_ids,
        'gentype': gentypes,
        'sntype': sntypes,
        'non1a_index': non1a_indices,
        'genz': redshifts,
        'libid': libids,
        'ra': ras,
        'dec': decs,
        'mwebv': mwebvs,
        'mwebvmap': mwebv_map,
        'mwebverr': mwebv_errs,
        'field': field_names_per_sn,
        'mu': distance_moduli,
        'model_params': model_params,
        'peakmjd': peak_mjd,
        'snrmax_sorted': stats['snrmax_sorted'],
        'trestmin': stats['trestmin'],
        'trestmax': stats['trestmax'],
        'tgapmax': stats['tgapmax'],
        'nobs': stats['nobs'],
        'band_tables': band_tables,
        'cutmask': cutmask_array,
        'selection_label': (
            "NONE    (write every generated event)" if include_all
            else "PASS_TRIGGER+CUTS  (write accepted events passing these criteria)")
    }

    if rows is None:
        return data
    keep = np.asarray(rows, dtype=bool)
    data['n_events'] = int(keep.sum())
    # APPLY_CUTWIN_OPT=3 keeps the accepted-set label (snlc_sim.c:14897)
    data['selection_label'] = (
        "PASS_TRIGGER+CUTS  (write accepted events passing these criteria)")
    for key, value in data.items():
        if key in ('n_events', 'selection_label'):
            continue
        data[key] = _take_rows(value, keep)
    return data


def _build_dump_column_pool(dump_data):
    """Collate all named per-SN columns into a single UPPER-keyed dict
    for case-insensitive resolution in `resolve_dump_content`.

    Pattern columns (`PEAKMAG_<filter>`, `SIM_<MODEL><PARAM>`) are
    handled by the resolver, not stored here.
    """
    pool = {name.upper(): dump_data[name] for name in _DUMP_EVENT_COLUMNS}
    snrmax_sorted = dump_data['snrmax_sorted']
    for rank in range(_DUMP_SNRMAX_RANKS):
        pool[f'SNRMAX{rank + 1}'] = (
            snrmax_sorted[:, rank] if rank < snrmax_sorted.shape[1]
            else np.full(len(pool['CID']), SNANA_NULLFLOAT))
    glon, glat = eq_to_gal(pool['RA'], pool['DEC'])
    pool['GLON'], pool['GLAT'] = np.asarray(glon), np.asarray(glat)
    params = dump_data['params']
    for snana_name, jax_key in _DUMP_PARAM_COLUMNS:
        if jax_key in params:
            pool[snana_name] = np.asarray(params[jax_key])
    for key, value in params.items():
        if key.startswith('sim_hostlib_'):
            pool[key[len('sim_hostlib_'):].upper()] = np.asarray(value)

    for alias, source in _DUMP_ALIASES:
        pool[alias] = pool[source]
    return pool


def _default_dump_format(arr):
    if np.issubdtype(arr.dtype, np.integer):
        return '%d'
    if np.issubdtype(arr.dtype, np.str_):
        return '%s'
    return '%.5e'


def _resolve_dump_column(name, pool, model_params, band_tables):
    """Resolve a requested DUMP column name to (array, format_string).

    Resolution order:
      1. Case-insensitive exact match in the pool (CID, GENZ, RA, …).
      2. Bare model parameter (AV, RV, THETA, …) aliased to SIM_<NAME>.
      3. <prefix><filter> → the per-band table for that prefix
         (PEAKMAG_, SNRMAX_, and with a HOSTLIB HOSTMAG_, SBFLUX_, SBMAG_).
      4. A model's own column, under the name the model gives it.
      5. Unknown → SNANAError naming every column that exists.
    """
    upper = name.upper()
    if upper in _DUMP_REFUSED_NAMES:
        raise SNANAError(
            f"SIMGEN_DUMP varname '{name}' is a legacy name; use "
            f"'{_DUMP_REFUSED_NAMES[upper]}'.",
            function_name='_resolve_dump_column')
    if upper in pool:
        arr = pool[upper]
        fmt = _DUMP_FORMAT_OVERRIDES.get(upper)
        if fmt is None:
            fmt = _default_dump_format(arr)
        return arr, fmt

    sim_alias = f'SIM_{upper}'
    for k, v in model_params.items():
        if k.upper() == sim_alias:
            arr = np.asarray(v)
            return arr, _default_dump_format(arr)

    for prefix, table in band_tables.items():
        if not upper.startswith(prefix):
            continue
        short_req = upper[len(prefix):]
        for filt, values in table.items():
            if filter_band(filt).upper() == short_req:
                return np.asarray(values), '%.5e'
        raise SNANAError(
            f"Unknown DUMP column '{name}': no filter matching "
            f"{prefix}{short_req}. Known filters: "
            f"{[filter_band(f) for f in table]}",
            function_name='_resolve_dump_column')

    # a SIM_<name> request also tries the pool without the prefix
    by_column = {k.upper(): v for k, v in model_params.items()}
    if upper in by_column:
        arr = np.asarray(by_column[upper])
        return arr, _default_dump_format(arr)
    if upper.startswith('SIM_') and upper[len('SIM_'):] in pool:
        stripped = upper[len('SIM_'):]
        arr = pool[stripped]
        fmt = _DUMP_FORMAT_OVERRIDES.get(stripped)
        if fmt is None:
            fmt = _default_dump_format(arr)
        return arr, fmt

    raise SNANAError(
        f"Unknown DUMP column '{name}'. Known columns: "
        f"{sorted(set(pool) | set(by_column))}; pattern columns: "
        f"{', '.join(p + '<filter>' for p in band_tables)}.",
        function_name='_resolve_dump_column')


def _resolve_requested_columns(config, model):
    """The VARNAMES column list: the user's `SIMGEN_DUMP: N col1 col2 …`
    list, with host-property names written as `<name>_TRUE`, plus the
    model-contributed columns. The dump only exists
    when the user configured columns (wr_SIMGEN_DUMP, snlc_sim.c:14833)."""
    user_cols = config.output.simgen_dumpall_columns
    # SNANA renames host-property basenames before matching, so a
    # requested LOGMASS or LOGMASS_TRUE is written as LOGMASS_TRUE.
    resolved = [f'{canonical_property_name(c)}_TRUE'
                if canonical_property_name(c) in HOSTGAL_PROPERTY_BASENAMES else c
                for c in map(str, user_cols)]
    model_cols = model_dump_columns(config, model)
    clash = [c for c in resolved if c in model_cols]
    if clash:
        raise SNANAError(
            f"Model parameter '{clash[0]}' cannot be in the SIMGEN_DUMP "
            f"list because model parameters are added automatically.",
            function_name='_resolve_requested_columns')
    return resolved + list(model_cols)


def resolve_dump_content(columnar_data, config, resources, full_results,
                         kcor_filter_names, model, peak_magnitudes,
                         selection_result, include_all: bool, to_written_cid,
                         rows=None) -> Dict[str, Any]:
    """The DUMP's header text and its columns as (array, format) pairs,
    resolved before any file is written so a bad column name leaves
    nothing behind; `write_dump_file` formats them.

    Honours the user's `SIMGEN_DUMP: N col1 col2 …` column list, in its
    order, with host-property names written as `<name>_TRUE`.
    Per-column resolution is generic (case-insensitive lookup +
    `PEAKMAG_<filter>` / `SIM_<model><param>` pattern rules), so
    adding new columns only requires populating them in
    `_build_dump_column_pool`.
    """
    dump_data = _prepare_dump_data(
        columnar_data, config, resources, full_results,
        kcor_filter_names, model, peak_magnitudes, selection_result,
        include_all, to_written_cid, rows)
    model_params = dump_data['model_params']
    pool = _build_dump_column_pool(dump_data)
    # the model's truth columns under every name SNANA gives them
    pool.update({name.upper(): model_params[column]
                 for name, column in model.SNANA_DUMP_SIM_COLUMNS})

    varnames = _resolve_requested_columns(config, model)

    resolved = [
        (name, *_resolve_dump_column(name, pool, model_params,
                                     dump_data['band_tables']))
        for name in varnames
    ]

    varnames_line = 'VARNAMES:  ' + ' '.join(varnames) + '\n'
    # Header lines follow wr_SIMGEN_DUMP (snlc_sim.c:14893-14911): the real
    # GENMODEL and PRESCALE_SIMGEN_DUMP, and SNANA's two selection labels.
    header = (
        '#\n'
        '# Simulation SUMMARY: one row per event. \n'
        f'#  MODEL:     {os.path.expandvars(config.model.genmodel)} \n'
        f'#  PRESCALE:  {config.output.prescale_simgen_dump} \n'
        f"#  SELECTION: {dump_data['selection_label']}\n"
        '\n'
        f'{varnames_line}'
        '\n'
    )

    n_events = dump_data['n_events']
    ps = config.output.prescale_simgen_dump
    # PRESCALE_SIMGEN_DUMP: N keeps the events with NEVT % N == 0
    keep = (np.flatnonzero(np.arange(1, n_events + 1) % ps == 0) if ps > 1
            else np.arange(n_events))
    return {'header': header, 'keep': keep,
            'columns': [(np.asarray(arr), fmt) for _, arr, fmt in resolved]}


# rows formatted per chunk
_EVENTS_PER_WRITE = 100_000


def write_dump_file(dump: Dict[str, Any], version_name, output_dir: str) -> str:
    """Write the DUMP: the header, then the rows in chunks."""
    output_file = Path(output_dir) / f'{version_name}.DUMP'
    keep = dump['keep']
    with open(output_file, 'w') as f:
        f.write(dump['header'])
        for start in range(0, len(keep), _EVENTS_PER_WRITE):
            rows = keep[start:start + _EVENTS_PER_WRITE]
            columns = [np.char.mod(fmt, arr[rows]).tolist()
                       for arr, fmt in dump['columns']]
            f.write(''.join('SN:  ' + ' '.join(row) + '\n'
                            for row in zip(*columns)))

    log.line(logger, f"wrote {output_file}", indent=2)
    return output_file


# Every column the pool can hold, whatever the run supplies
# (PREP_SIMGEN_DUMP(0) prints SNANA's equivalent list).
_STATIC_DUMP_VARIABLES = (
    tuple(name.upper() for name in _DUMP_EVENT_COLUMNS)
    + tuple(f'SNRMAX{rank + 1}' for rank in range(_DUMP_SNRMAX_RANKS))
    + ('GLON', 'GLAT')
    + tuple(name for name, _ in _DUMP_PARAM_COLUMNS)
    + tuple(alias for alias, _ in _DUMP_ALIASES))


def print_simgen_dump_variables(config, resources):
    """List the available SIMGEN_DUMP variables, for `SIMGEN_DUMP: 0`
    (PREP_SIMGEN_DUMP(0), snlc_sim.c:14860-14862). The caller exits after."""
    model, kcor_filter_names = resources.model, resources.filter_names
    names = list(_STATIC_DUMP_VARIABLES)
    prefixes = ['PEAKMAG_', 'SNRMAX_']
    if resources.hostlib is not None:
        prefixes += ['HOSTMAG_', 'SBFLUX_', 'SBMAG_']
    names += [f'{p}{filter_band(f)}' for p in prefixes for f in kcor_filter_names]
    names += [name for name, _ in model.SNANA_DUMP_SIM_COLUMNS]
    names += sorted(model_dump_columns(config, model))
    print("SIMGEN_DUMP: 0 --> list variables and quit")
    for name in names:
        print(f"  {name}")
    return names
