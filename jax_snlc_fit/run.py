"""The fit run: what snlc_fit.exe does between reading its namelist and
writing its tables. `run_fitting` orchestrates; `snlc_fit.py` is the CLI shim
that parses inputs and calls it."""

import dataclasses
import glob
import logging
import os
import time
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from jax_snlc_core.io.docana import check_file_docana
from jax_snlc_core.io.fitres_reader import read_fitres
from jax_snlc_core.io.kcor_reader import KcorReader
from jax_snlc_core.io.sfd_reader import SFDDustMap
from jax_snlc_core.io.sim_cosmology import read_simulation_cosmology
from jax_snlc_core.io.snana_table import (jax_version_comment, program_comment,
                                          snana_version_comment,
                                          write_varnames_table)
from jax_snlc_core.models import (MODEL_REGISTRY, FittedModel, build_model,
                                  metadata_files)
from jax_snlc_fit import fit_spec
from jax_snlc_fit.inputs import selected_model
from jax_snlc_core.photometry.filter_system import filter_band, filter_values
from jax_snlc_core.photometry.fluxerr_model import read_fluxerr_model_file
from jax_snlc_core.utils import SIM_MAGOBS_UNSET, group_equal
from jax_snlc_core.utils import snana_log as slog
from jax_snlc_core.utils.cutwin import (CUTVAL_OPEN, apply_range_cuts,
                                        gap_maxima, in_window, narrow)
from jax_snlc_core.utils.error_handling import SNANAError
from jax_snlc_core.utils.precision import require_precision

from . import photoz, priors, uncertainties
from .batch import (ERRFLAG_FITPREP_NEPOCH, ERRFLAG_FITPREP_NOEPOCHS,
                    ERRFLAG_FITPREP_QUANTILES, ERRFLAG_FIT_CUTS,
                    ERRFLAG_MNFIT_NAN, ERRFLAG_NO_FIT, ERRFLAG_ZFIT_RANGE,
                    ITER1_ERRFLAG, LightCurveBatch, band_snr_maxima,
                    choose_epoch_lengths, cid_keys, padded_batches,
                    plan_fit_batches, select_events)
from .config import FitConfig
from .config_parser import (check_photoz_trest_cuts, parse_sntable_list,
                            relax_photoz_trest_cuts, resolve_fitwin_windows,
                            resolve_version_dir, until_sentinel)
from .data_loader import (batch_from_columns, model_sim_columns, peak_sizes,
                          read_epoch_ignore_file, read_header_override_file,
                          read_magcor_file)
from .fitter import (SIMFIT_IDEAL_DEBUG, LightCurveFitter,
                     attach_mw_attenuation, fit_band_window,
                     fit_parameters, fit_photometry, resolved_config,
                     resolved_filters, shift_map_with_poly)
from .host_noise import read_hostnoise_file
from .io import version_sources
from .loss_functions import zpdf_rows_usable
from .nonlinearity import read_nonlinearity_file
from .minos import profile_error_type
from .sntable import (LCLIST_HEADER, LCPLOT_TOBS_GRID,
                      build_event_columns, build_fitres_table,
                      build_host_columns, build_lcplot_table,
                      build_outlier_table, build_snana_table, snana_sim_block)
from .vmax import compute_vmax

logger = logging.getLogger(__name__)

# FIT_TIMING phases, in order
FIT_PHASES = ('startup', 'config', 'read', 'batch', 'model', 'compile', 'fit',
              'cuts', 'write')


def _check_docana(optmask, paths):
    """REQUIRE_DOCANA: each file must open with the DOCUMENTATION key.
    Bit 0 aborts, otherwise it warns; the check itself is core's."""
    if not optmask:
        return
    for path in paths:
        check_file_docana(path, require=bool(optmask & 1))


def _dump_fluxerr_maps(model, source, out_dir):
    """FLUXERRMODEL_OPTMASK bit 8: re-write each map in fitres row format as
    FLUXERRMAP-<MAPNAME>.TEXT (DUMP_MAP_FLUXERRMODEL, sntools_fluxErrModels.c:280).
    Maps sharing a MAPNAME append to one file, with ROW numbering continuing."""
    rows_written = {}
    for imap, m in enumerate(model.maps):
        path = Path(out_dir) / f"FLUXERRMAP-{m.name}.TEXT"
        first = m.name not in rows_written
        with open(path, 'w' if first else 'a') as fh:
            if first:
                rows_written[m.name] = 0
                fh.write(f"# FLUXERRMAP from file:\n#\t {source}\n#\n")
                names = " ".join(["ROW", "IMAP", "FIELD", "BAND"] + m.varnames)
                fh.write(f"NVAR: {4 + len(m.varnames)} \n")
                fh.write(f"VARNAMES: {names} \n")
            for idx in np.ndindex(m.grid_values.shape):
                rows_written[m.name] += 1
                coords = [ax[i] for ax, i in zip(m.grid_axes, idx)]
                vals = "".join(f" {v:7.3f}"
                               for v in coords + [m.grid_values[idx]])
                fh.write(f"ROW: {rows_written[m.name]:3d}  "
                         f"{imap} {m.field_group} {m.band} {vals}\n")
        slog.kv(logger, "FLUXERRMAP dump", path)


def read_kcor(config):
    """KCOR_FILE, with LEAKAGE_CUT applied."""
    return KcorReader(os.path.expandvars(config.data.kcor_file),
                      leakage_cut=config.data.leakage_cut)


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


def write_fluxerr_maps(config, shared):
    """FLUXERRMODEL_OPTMASK bit 256: each flux-error map of ``shared``
    rewritten beside the tables ``config`` writes, the simulation's after
    the data's, which they overwrite, as in SNANA."""
    if not config.data.fluxerrmodel_optmask & 256:
        return
    for model, src in ((shared['fluxerr_model'],
                        config.data.fluxerrmodel_file),
                       (shared['sim_fluxerr_model'],
                        config.data.sim_fluxerrmodel_file)):
        if model is not None:
            _dump_fluxerr_maps(model, os.path.expandvars(src),
                               Path(config.output.output_name or '').parent)


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
        setpkmjd=_setpkmjd_options(config),
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


def mwebv_options(config):
    """OPT_MWEBV and its modifiers from the Data record, with the SFD dust
    map reader when the E(B-V) comes from the map. The maps are 64 MB each,
    so the reader is built once and passed to every data version."""
    d = config.data
    return {"opt": d.opt_mwebv, "scale": d.mwebv_scale,
            "shift": d.mwebv_shift, "force": d.mwebv_force,
            "sfd_map": SFDDustMap() if d.opt_mwebv in (2, 3) else None}


def _setpkmjd_options(config):
    """Peak-MJD estimator settings, or None to keep the header PEAKMJD."""
    if not (config.data.has_setpkmjd_bazin() or config.data.has_setpkmjd_fmax_clump()
            or config.data.has_setpkmjd_fluxmax() or config.data.has_setpkmjd_trigger()):
        return None
    return {
        "bazin": config.data.has_setpkmjd_bazin(),
        "fmax_clump": config.data.has_setpkmjd_fmax_clump(),
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


def _read_cid_file(path):
    """CIDs from a SNCID list file, plus its table when it is in FITRES format.

    match_cidlist_init (sntools.c:259) takes either format; only the table one
    carries the fit parameters OPT_SNCID_LIST bit 1 seeds from.
    """
    path = os.path.expandvars(path)
    with open(path) as fh:
        text = fh.read()
    if 'VARNAMES:' not in text:
        return [tok for line in text.splitlines()
                for tok in line.split("#")[0].split()], None
    table = read_fitres(path)
    return [str(c) for c in table.df['CID']], table


def _cid_list(numeric, strings, path):
    """One CID list from its numeric, string and file spellings, the names
    among the strings and the file, and the list-file table.

    SNANA keeps SNCID_ and SNCCID_ apart because Fortran cannot hold both in one
    array; here every CID is a string, matched by :func:`cid_keys`, so they
    merge. The numeric list ends at its first entry below 1 (DO WHILE
    SNCID_LIST(i) >= 1, snana.F90:11888); the string list ends at its first
    blank instead (:11897). Every string and every file entry is also a
    transient or IAUC name to match, a numeric one included, since a Rubin
    object id is a number (LCIDSELECT, snana.F90:11902).
    """
    cids = [str(c) for c in until_sentinel([int(c) for c in (numeric or [])], 1)]
    blank = next((i for i, c in enumerate(strings or [])
                  if not str(c).strip()), None)
    names = [str(c).strip() for c in (strings or [])[:blank]]
    table = None
    if path:
        from_file, table = _read_cid_file(path)
        names += from_file
    return cids + names, names, table


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
        slog.line(logger, f"{Path(path).name}: {b.n_sne} SNe")
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


def drop_unfittable(batch, nepoch_window, *, abort_on_noepochs,
                    quantile_prior, abort_on_missing_qzphot):
    """Events SNANA skips at fit prep rather than at the cuts: nothing to fit
    (NFITDATA <= 0, snlc_fit.F90:10610), fewer epochs than CUTWIN_NEPOCH's
    lower bound (:10630), or a host z-PDF the spline cannot take
    (:8540-8543). They passed the SNLC cuts, so they keep
    CUTFLAG_SNANA = 1. An empty event aborts under ABORT_ON_NOEPOCHS, and
    one missing its z-PDF quantiles under ABORT_ON_MISSING_QZPHOT.

    Returns (fittable batch, empty SNIDs, too-thin SNIDs, bad-quantile
    SNIDs); each carries its own ERRFLAG_FIT code."""
    nobs = np.asarray(batch.n_obs_per_sn)
    snids = np.asarray(batch.snids)
    empty = nobs <= 0
    if abort_on_noepochs and empty.any():
        raise SNANAError(
            f"Nothing to fit for CID {snids[empty][0]}; to avoid abort, set "
            f"ABORT_ON_NOEPOCHS = F.", function_name="FITINI_COV")
    floor = max(1, int(nepoch_window[0])) if nepoch_window else 1
    epoch_ok = nobs >= floor
    if not epoch_ok.all():
        slog.line(logger, f"Skipped {int((~epoch_ok).sum())} SNe with fewer than "
                          f"{floor} epochs to fit")
    badq = np.zeros(batch.n_sne, dtype=bool)
    needs_quantiles = epoch_ok & np.asarray(quantile_prior, dtype=bool)
    if needs_quantiles.any():
        # FITINI_PHOTOZ, snlc_fit.F90:8468-8478, :8540-8543
        missing = (needs_quantiles if batch.zpdf_quantiles is None else
                   needs_quantiles & (np.asarray(batch.zpdf_nbin) == 0))
        if missing.any() and abort_on_missing_qzphot:
            raise SNANAError(
                f"Missing zPDF quantiles for photo-z fit to CID="
                f"{snids[missing][0]}. To skip such events, set &SNLCINP "
                f"ABORT_ON_MISSING_QZPHOT=F", function_name="FITINI_PHOTOZ")
        badq = (missing if batch.zpdf_quantiles is None else
                needs_quantiles & ~zpdf_rows_usable(
                    batch.zpdf_quantiles, batch.zpdf_nbin))
        if badq.any():
            slog.line(logger, f"Skipped {int(badq.sum())} SNe whose host z-PDF "
                              f"quantiles are missing or do not increase")
    fittable = epoch_ok & ~badq
    return (batch.filter_by_mask(fittable) if not fittable.all() else batch,
            snids[empty], snids[~epoch_ok & ~empty], snids[badq])


def sncid_list(config):
    """SNCID_LIST, SNCCID_LIST and SNCID_LIST_FILE as one CID list, with
    the names and the list file's table (:func:`_cid_list`)."""
    return _cid_list(config.selection.sncid_list,
                     config.selection.snccid_list,
                     config.selection.sncid_list_file)


def build_selection(config, cutwin, listed):
    """The per-SN selection for the loader; ``listed`` is the config's
    :func:`sncid_list`."""
    selection = {"abort_on_duplcid": config.selection.abort_on_duplcid}
    cids, names, _ = listed
    prescale = config.selection.sim_prescale
    if cids or names:
        # any CID list zeroes the prescale (snana.F90:11872)
        prescale = 1.0
        selection["sncid_list"] = cids
        selection["snccid_names"] = names
    ignored, _, _ = _cid_list(config.selection.sncid_ignore,
                              config.selection.snccid_ignore,
                              config.selection.sncid_ignore_file)
    if ignored:
        selection["sncid_ignore"] = ignored
    if config.selection.sntype_ignore:
        # ends at the first negative entry (snana.F90:5915)
        selection["sntype_ignore"] = until_sentinel(
            [int(t) for t in config.selection.sntype_ignore], 0)
    # OPT_SNCID_LIST bit 0: CID cuts only, ignore lists still apply (snana.F90:15444)
    if (cids or names) and config.selection.opt_sncid_list & 1:
        return selection
    if cutwin:
        selection["windows"] = cutwin
    if config.selection.sntype_list:
        selection["sntype_list"] = list(config.selection.sntype_list)
    if config.selection.require_epochs:
        selection["require_epochs"] = dict(config.selection.require_epochs)
    if config.selection.private_cutwin:
        selection["private_windows"] = dict(config.selection.private_cutwin)
    if config.selection.private_veto:
        selection["private_veto"] = dict(config.selection.private_veto)
    if config.selection.sncut_nobs_trest:
        selection["nobs_trest"] = list(config.selection.sncut_nobs_trest)
    if config.selection.simvar_cutwin:
        selection["simvar_windows"] = dict(config.selection.simvar_cutwin)
    if prescale > 1.0:
        selection["sim_prescale"] = prescale
    if config.selection.snrmax_filters:
        selection["snrmax_filters"] = config.selection.snrmax_filters
    # USESIM_SNIA/NONIA select on the simulated true type (snana.F90:3952).
    if not config.selection.usesim_snia:
        selection["usesim_snia"] = False
    if not config.selection.usesim_nonia:
        selection["usesim_nonia"] = False
    return selection


def _fitted_z(config, params, redshifts):
    """Each event's redshift as fitted: its zPHOT under photo-z, and
    ``redshifts``, the data's, otherwise."""
    return np.asarray(params['zPHOT'] if config.photoz.dofit_photoz
                      else redshifts)


def _reference_mu(config, results, batch):
    """The reference distance modulus the fitted parameters are read against:
    at the fitted redshift under photo-z, as the fit's own distance prior is,
    and at the data's otherwise."""
    return config.fit.distance_modulus(
        _fitted_z(config, results.params, batch.redshifts))


def _fitwin_mask(windows, model, results, redshifts, mu_ref, batch, config):
    """Per-SN mask for the FITWIN_ windows on the finished fit.

    Quantities come from the model's own FITRES columns, so derived ones (SALT's
    mB) and their errors are covered without naming any parameter here.
    """
    z1 = 1.0 + np.asarray(redshifts)
    quantities = {}
    for column, values, errs in model.get_fitres_param_columns(
            results.params, results.param_errors, redshifts, mu_ref):
        quantities[column] = np.asarray(values)
        quantities[f"{column}_ERR"] = np.asarray(errs)
        quantities[f"{column}_ERRZ1"] = np.asarray(errs) / z1

    quantities['PROB'] = np.asarray(model.compute_fitprob(
        np.asarray(results.chi_squared), np.asarray(results.fitprob_ndof)))

    if 'ZPULL' in windows:
        if 'zPHOT' not in results.params or batch.photoz is None:
            raise SNANAError(
                "FITWIN_ZPULL cuts on (z_fit - z_prior)/z_prior_err, which "
                "needs a fitted redshift and a host photo-z prior; this fit "
                "has neither.", function_name="FITWIN_ZPULL")
        zfit = np.asarray(results.params['zPHOT'])
        zfiterr = np.asarray(results.param_errors['zPHOT'])
        dz = 0.01
        mu = config.fit.distance_modulus(zfit)
        mu2 = config.fit.distance_modulus(zfit + dz)
        dzdmu = dz / (np.asarray(mu) - np.asarray(mu2))
        zmagerr = config.data.snmagrms_for_photoz * np.abs(dzdmu)
        zerr = np.maximum(
            np.sqrt(zfiterr ** 2 + np.asarray(batch.photoz_err) ** 2
                    + zmagerr ** 2), 0.00001)
        quantities['ZPULL'] = (zfit - np.asarray(batch.photoz)) / zerr

    named = {}
    for column, window in windows.items():
        # SNANA keeps only the upper edge of _ERR windows (snlc_fit.F90:1403-1417)
        lo, hi = window
        if column.endswith(('_ERR', '_ERRZ1')):
            lo = -CUTVAL_OPEN
        named[column] = (lo, hi)

    # two loose error cuts SNANA hard-wires and always applies
    # (snlc_fit.F90:6835, tested at :12958)
    named.setdefault('COV_OFFDIAG', (-8.0, 8.0))
    if 'mB_ERR' in quantities:
        named.setdefault('mB_ERR', (-CUTVAL_OPEN, 1.0))
    windows = named

    cov = np.asarray(results.param_covariances)
    off = ~np.eye(cov.shape[1], dtype=bool)
    terms = cov[:, off]
    # a term left at -9 means the pair was never fitted, and SNANA skips it
    # rather than testing it (snlc_fit.F90:12963)
    unfitted = np.abs(terms + 9.0) < 1.0e-6
    lo_edge, hi_edge = windows['COV_OFFDIAG']
    # the window must hold for every off-diagonal term, so test the extremes
    quantities['COV_OFFDIAG'] = np.where(unfitted, hi_edge, terms).min(axis=1)
    extra = apply_range_cuts(
        {'COV_OFFDIAG': (lo_edge, hi_edge)},
        {'COV_OFFDIAG': np.where(unfitted, lo_edge, terms).max(axis=1)})

    return apply_range_cuts(windows, quantities) & extra


def _fit_error_bounds_mask(fit_parameters, held_params, results):
    """Drop events whose fitted errors are NaN or outside SNANA's fixed ranges
    (CRAZYFITERR, snlc_fit.F90:2436). SNANA sets ERRFLAG_FIT_CRAZYERR and
    returns, so the event never reaches the FITRES.

    Each parameter carries its own bounds, so a model is checked on exactly the
    parameters it declares them for.
    """
    bounds = {name: spec['error_bounds']
              for name, spec in fit_parameters.items()
              if 'error_bounds' in spec}
    keep = np.ones(results.n_sne, dtype=bool)
    for name, (lo, hi) in bounds.items():
        err = np.asarray(results.param_errors[name])
        keep &= ~np.isnan(err)
        if name not in held_params:
            keep &= (err >= lo) & (err <= hi)
    return keep


def errmax_bad(config, fit_parameters, held_params, results, init_params):
    """``results`` with each floated error too small to mean anything
    replaced by its step size and flagged as bad.

    SNANA substitutes +/-INISTP and flags ERRTYPE_BAD, keeping the event
    (snana.F90:27879). MINOS_ERRMAX_BAD = F only zeroes the floors
    (snlc_fit.F90:1481); the "never left its start" condition still fires.
    """
    errors = dict(results.param_errors)
    if results.error_flags is not None:
        flags = dict(results.error_flags)
    elif results.minos_errors is not None:
        flags = {name: np.asarray(profile_error_type(
            results.minos_errors[name], results.minos_status[name]))
            for name in results.param_names}
    else:
        flags = {name: np.full(results.n_sne, 2, dtype=int)
                 for name in results.param_names}
    for name, spec in fit_parameters.items():
        err = np.asarray(results.param_errors[name])
        flags[name] = np.where(np.isfinite(err), flags[name], 6)
        if name in held_params:
            continue
        floor = (spec.get('errmax_bad', 0.0)
                 if config.fit.minos_errmax_bad else 0.0)
        bad = err < floor
        if results.minos_errors is not None:
            sides = np.asarray(results.minos_errors[name])
            profiled = np.asarray(profile_error_type(
                sides, results.minos_status[name])) == 1
            bad = np.where(profiled, np.all(sides < floor, axis=-1), bad)
        init = None
        if name in init_params:
            init = np.asarray(init_params[name])
            bad = bad | (np.asarray(results.params[name]) == init)
        if not bad.any():
            continue
        if 'inistp_relative_to_init' in spec:
            step = init * spec['inistp_relative_to_init']
        else:
            step = np.full(len(err), float(
                config.model.inistp.get(name, spec.get('inistp', 0.0))))
        errors[name] = np.where(bad, step, err)
        flags[name] = np.where(bad | ~np.isfinite(errors[name]), 6, flags[name])
        slog.line(logger, f"ERRMAX_BAD: {name} error replaced by INISTP "
                          f"for {int(bad.sum())} SNe")
    return dataclasses.replace(results, param_errors=errors, error_flags=flags)


def _extreme(reduce, values, live, initial):
    """A row's extreme over its live epochs, NaN where it has none."""
    return np.where(live.any(axis=1),
                    reduce(values, axis=1, initial=initial, where=live), np.nan)


def _sncid_pegged_mask(config, model, results, fitwin_windows):
    """OPT_SNCID_LIST bit 0 keeps every event but those whose fit pegged against
    a flat-prior edge, and those whose FITPROB fell under a tenth of
    FITWIN_PROB's lower bound (snlc_fit.F90:12700-12708).

    The margin is the parameter's own, because 0.05 is a width rather than a
    fraction: it means one thing on a shape running over units and quite
    another on a colour running over tenths. SNANA tests its shape slot alone,
    so declaring the margin only there reproduces it.
    """
    keep = np.ones(results.n_sne, dtype=bool)
    for name, spec in model.get_fit_parameters().items():
        margin = spec.get('peg_margin')
        window = priors.flat_window(config, name, spec)
        if margin is None or window is None or name not in results.params:
            continue
        lo, hi, _ = window
        value = np.asarray(results.params[name])
        keep &= (value >= lo + margin) & (value <= hi - margin)
    if 'PROB' in fitwin_windows:
        fitprob = np.asarray(model.compute_fitprob(
            np.asarray(results.chi_squared), np.asarray(results.fitprob_ndof)))
        keep &= fitprob >= fitwin_windows['PROB'][0] / 10.0
    return keep


def _sncid_inival(table, snids, param_names):
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


def _post_fit_trest_mask(config, saved_nominal, effective_cutwin, results,
                         batch, fit_mask):
    """The Trest block of FITANA_CUTS, on the finished fit.

    FITANA_STORE recomputes the quantities through GET_TRESTVAR with the
    *nominal* windows (FITANA_CUTWIN_*, snlc_fit.F90:13236), which under a
    photo-z fit are not the widened ones PHOTOZ_SAVECUT left in cutwin_var,
    and cutwin_var is what the cuts then test against (:12851-12862).
    """
    fitted_z = _fitted_z(config, results.params, batch.redshifts)
    t0_offset = (np.asarray(results.params['PKMJD'])
                 - np.asarray(batch.reference_mjds))
    trest = ((np.asarray(batch.mjd_offsets) - t0_offset[:, None])
             / (1.0 + fitted_z)[:, None])
    fidx = np.asarray(batch.filter_indices)
    n_filt = len(batch.filter_names)
    effective = effective_cutwin
    nominal = {**effective_cutwin, **saved_nominal}

    # fitted epochs only (snlc_fit.F90:13221, snana.F90:20872)
    obs_ok = np.asarray(fit_mask)
    if 'trest' in nominal:
        obs_ok = obs_ok & in_window(trest, *nominal['trest'])
    keep = np.ones(batch.n_sne, dtype=bool)
    fitted = {'trestmin': _extreme(np.min, trest, obs_ok, np.inf),
              'trestmax': _extreme(np.max, trest, obs_ok, -np.inf)}
    # T0GAPWIN is built from the nominal edges (snana.F90:20851)
    fitted['tgapmax'], fitted['t0gapmax'] = gap_maxima(
        trest, obs_ok,
        nominal.get('trestmin', (0.0, CUTVAL_OPEN))[1],
        nominal.get('trestmax', (-CUTVAL_OPEN, 0.0))[0])

    for name, value in fitted.items():
        if name in effective:
            keep &= in_window(value, *effective[name]) | np.isnan(value)
    # TrestRange_FIT against the lower edge only (snlc_fit.F90:12905-12909)
    if 'trestrange' in effective:
        span = fitted['trestmax'] - fitted['trestmin']
        keep &= (span >= effective['trestrange'][0]) | np.isnan(span)

    # NFILT_TRESTMIN/MAX (snana.F90:20880, snlc_fit.F90:13065)
    for name in ('trestmin', 'trestmax'):
        threshold = effective.get(f'nfilt_{name}', (1.0, 100.0))[0]
        inside = obs_ok & in_window(
            trest, *nominal.get(name, (-CUTVAL_OPEN, CUTVAL_OPEN)))
        per_filter = np.stack([(inside & (fidx == b)).any(axis=1)
                               for b in range(n_filt)], axis=1)
        keep &= per_filter.sum(axis=1) >= threshold

    if not keep.all():
        slog.line(logger, f"Post-fit Trest cut: dropped {int((~keep).sum())} SNe")
    return keep


def _peakmjd_window_mask(config, results):
    """PEAKMJD_SANITY, the post-fit window on the fitted PKMJD
    (snlc_fit.F90:12813-12828).

    A zero lower edge means unset, and SNANA substitutes CUTWIN_PEAKMJD +/- 20.
    Both edges are strict here, unlike the inclusive FITWIN_ windows.
    """
    lo, hi = config.data.peakmjd_sanity
    if lo == 0.0:
        lo, hi = config.selection.cutwin.get('peakmjd', (-CUTVAL_OPEN, CUTVAL_OPEN))
        lo, hi = lo - 20.0, hi + 20.0
    pkmjd = np.asarray(results.params['PKMJD'])
    return (pkmjd > lo) & (pkmjd < hi)


def _substitute_true_flux(batch, zp_shift, key):
    """Replace measured fluxes with the simulated truth.

    SNANA carries TRUEFLUX per epoch; jax's PHOT files carry SIM_MAGOBS, so the
    flux is recovered as 10^(-0.4*(SIM_MAGOBS - ZP_FLUXCAL)). Undefined epochs
    (SIM_MAGOBS >= 99, SNANA's no-flux sentinel) become zero flux. The truth is
    on the unshifted scale, so MAGOBS_SHIFT_ZP is re-applied to it as it was to
    the measured flux (FLUXSCALE_ZP_FILT, snlc_fit.F90:15915).
    """
    extras = batch.obs_extras or {}
    if 'SIM_MAGOBS' not in extras:
        raise SNANAError(
            f"{key} needs the SIM_MAGOBS column, which this data version does "
            f"not carry; it is written only for simulations.",
            function_name=key)
    magobs = np.asarray(extras['SIM_MAGOBS'])
    trueflux = np.where(magobs < SIM_MAGOBS_UNSET,
                        10.0 ** (-0.4 * (magobs - batch.zp_fluxcal)), 0.0)
    if zp_shift:
        fidx = np.asarray(batch.filter_indices)
        scale = np.ones(max(batch.filter_names) + 1)
        shifts = filter_values(zp_shift, list(batch.filter_names.values()), 0.0)
        for i, shift in zip(batch.filter_names, shifts):
            scale[i] = 10.0 ** (-0.4 * shift)
        trueflux = trueflux * scale[fidx]
    return dataclasses.replace(batch, fluxcals=trueflux)


def _ideal_batch(config, batch):
    """The noise-free batch the SIMFIT_IDEAL fit runs on.

    SET_IDEAL_FLUXCAL (snlc_fit.F90:15880) swaps in the true flux and moves
    MWEBV to its simulated value with zero error; PKMJD is pinned to truth by
    the caller. Errors are deliberately left as measured, so chi2 keeps its
    scale.
    """
    batch = _substitute_true_flux(batch, config.data.magobs_shift_zp,
                                  'SIMFIT_IDEAL_PRESCALE')
    return dataclasses.replace(
        batch, mwebvs=np.asarray(batch.sim_truth['SIM_MWEBV'], dtype=float),
        mwebv_errs=np.zeros(batch.n_sne))


def _nfilt_snrmax_mask(windows, batch, fit_mask, filter_names, fiterr):
    """NFILT_SNRMAX recounted over the epochs that survived the fit
    (snlc_fit.F90:13038), or None when neither window is set.

    The SNR is SNR_FIT, over the error carrying the MAXFRAC fudge, so a
    deweighted passband stops counting (snlc_fit.F90:4215, :13277).
    """
    cuts = [(windows[w][0], windows.get(s, (-CUTVAL_OPEN,))[0])
            for w, s in (('nfilt_snrmax', 'snrmax'),
                         ('nfilt_snrmax2', 'snrmax2'))
            if w in windows]
    if not cuts:
        return None
    err = np.asarray(fiterr)
    fidx = np.asarray(batch.filter_indices)
    snr = np.where(fit_mask & (err > 0), np.asarray(batch.fluxcals) / err,
                   -99.0)
    band_max = np.stack(
        [np.where(fit_mask & (fidx == b), snr, -99.0).max(axis=1)
         for b in range(len(filter_names))], axis=1)
    keep = np.ones(batch.n_sne, dtype=bool)
    for nfilt_lo, snrmax_lo in cuts:
        keep &= (band_max > snrmax_lo).sum(axis=1) >= nfilt_lo
    return keep


def evaluate_model(fitter, batch, params):
    """Model flux and its error at ``params`` (one [n_sne] array per
    parameter), [n_sne, max_obs] each, for the tables that report them.

    Batched like the fit, at the fit's shapes: evaluating a whole sample at
    once costs the memory fitting it in one go would. ``batch`` carries its
    photometry and Milky Way attenuation (:func:`attach_mw_attenuation`).
    """
    params = {k: np.asarray(v) for k, v in params.items()}
    flux = np.zeros((batch.n_sne, batch.max_obs))
    flux_err = np.zeros_like(flux)
    for epoch_length, rows, idx in plan_fit_batches(
            batch.n_obs_per_sn, fitter.epoch_lengths, fitter.batch_size):
        chunk = batch.take(idx, epoch_length)
        chunk_params = {k: v[idx] for k, v in params.items()}
        mf, mfe = fitter.evaluate_flux(
            chunk_params,
            chunk.reference_mjds[:, None] + chunk.mjd_offsets,
            _fitted_z(fitter.config, chunk_params, chunk.redshifts),
            chunk.filter_indices,
            chunk.observation_mask, reference_mjds=chunk.reference_mjds,
            mw_att_obs=chunk.mw_att_obs, photometry=chunk.photometry)
        n = len(rows)
        flux[np.ix_(rows, np.arange(epoch_length))] = np.asarray(mf)[:n]
        flux_err[np.ix_(rows, np.arange(epoch_length))] = np.asarray(mfe)[:n]
    return flux, flux_err


def start_values(fitter, config, batch, usesim_columns):
    """Starting values for ``batch``: the model's own guess, then
    ``INIVAL_<p>``, then the simulated truth in ``usesim_columns``, then the
    SNCID list file where OPT_SNCID_LIST bit 1 asks for it. SNANA applies
    them in that order, so truth wins over an explicit ``INIVAL_`` for a
    parameter that has both (FITINI_PARVAL, snlc_fit.F90:8290-8318).

    ``config`` is the data version's; the list file's values are the ones
    :func:`load` put on the batch. Every value is per event. A photo-z fit's
    zPHOT starts at its redshift prior before these overrides.
    """
    t0_init = jnp.array(batch.reference_mjds if batch.peakmjd_offsets is None
                        else batch.peakmjd_offsets + batch.reference_mjds)
    start = fit_spec.init_params(
        fitter.model, batch.n_sne, t0_init,
        config.fit.start_distance_modulus(batch.redshifts))
    if config.photoz.dofit_photoz:
        start['zPHOT'] = batch.z_priors.z_init
    for name, value in config.model.inival.items():
        start[name] = jnp.full(batch.n_sne, value)
    for name, value in config.model.initial.items():
        start[name] = jnp.full(batch.n_sne, value)
    for name, column in usesim_columns.items():
        start[name] = jnp.asarray(_batch_truth(batch, column))
    if batch.sncid_inival is not None:
        for name, seed in batch.sncid_inival.items():
            start[name] = jnp.where(jnp.isnan(seed), start[name], seed)
    for name, value in config.model.fixed.items():
        if value is not None:
            start[name] = jnp.full(batch.n_sne, value)
    return start


def fit(fitter, config, batch, *, progress=False):
    """Fit ``batch``, which carries its Milky Way attenuation: its results in
    its own order, the starting values :func:`start_values` gave, and one
    record per fitted batch."""
    start = start_values(fitter, config, batch, fitter.usesim_columns)
    results, records = fitter.fit_sample(batch, start, progress=progress)
    return results, start, records


def ideal_fit(fitter, config, batch):
    """SIMFIT_IDEAL_PRESCALE: refit a prescaled subset on noise-free fluxes.

    SNANA runs this ahead of the nominal fit for each selected event and then
    re-initialises (SIMFIT_IDEAL_PREP, snlc_fit.F90:15808), so the two are
    independent and this is a separate pass. Every parameter starts at its
    simulated truth and PKMJD is pinned there; at most two iterations run.
    Returns {column: [n_sne]}, 999 where an event was not refitted
    (FITPAR_IDEAL init, snlc_fit.F90:15980), or None when the key is inert.
    """
    if batch.sim_truth is None:
        # the LSIM_SNANA gate: real data silently disables the key (:15796)
        slog.line(logger, "SIMFIT_IDEAL_PRESCALE ignored: not a simulation")
        return None
    ps = config.fit.simfit_ideal_prescale
    debug = ps == SIMFIT_IDEAL_DEBUG
    idx = np.asarray(batch.snana_index, dtype=float)
    mask = (np.ones(batch.n_sne, dtype=bool) if debug
            else (np.mod(idx, np.pi) / np.pi) <= (1.0 / ps))
    if not mask.any():
        slog.line(logger, "SIMFIT_IDEAL_PRESCALE selected no events")
        return None

    ideal = fitter.ideal_fitter
    refit = with_fit_settings(config, ideal)
    # the refit starts at the truth, not from the SNCID list file
    subset = dataclasses.replace(batch.filter_by_mask(mask), sncid_inival=None)
    # debug value skips SET_IDEAL_FLUXCAL (snlc_fit.F90:15843)
    if not debug:
        subset = attach_mw_attenuation(refit, _ideal_batch(refit, subset))
    truth = {name: spec['sim_truth']
             for name, spec in fitter.model.get_fit_parameters().items()
             if 'sim_truth' in spec}
    slog.line(logger, f"SIMFIT_IDEAL: fitting {subset.n_sne} of "
                      f"{batch.n_sne} SNe on {'measured' if debug else 'true'} flux")
    results, _ = ideal.fit_sample(
        subset, start_values(ideal, refit, subset, truth))

    columns = {}
    for name, vals, _ in fitter.model.get_fitres_param_columns(
            results.params, results.param_errors,
            np.asarray(batch.redshifts)[mask],
            _reference_mu(refit, results, subset)):
        column = np.full(batch.n_sne, 999.0)
        column[mask] = np.asarray(vals)
        columns[f"{name}_IDEAL"] = column
    return columns


def cut_windows(config):
    """``(cutwin, epoch_cutwin, nominal)``: the ``CUTWIN_`` windows as the
    run applies them. A photo-z fit widens the Trest ones for the fit and
    re-tests ``nominal``, the windows as given, after it (snlc_fit.F90:3289,
    PHOTOZ_SAVECUT :3703)."""
    selection = config.selection
    if not config.photoz.dofit_photoz:
        return selection.cutwin, selection.epoch_cutwin, {}
    return relax_photoz_trest_cuts(config, selection.cutwin,
                                   selection.epoch_cutwin)


def _record(batch, *marks):
    """``batch`` with the ERRFLAG_FIT of each ``(snids, code)`` mark set, a
    later mark winning over an earlier one."""
    sid = np.char.strip(np.asarray(batch.snids).astype(str))
    errflag = np.array(batch.errflag_fit)
    for snids, code in marks:
        errflag[np.isin(sid, np.char.strip(np.asarray(snids).astype(str)))] = code
    return dataclasses.replace(batch, errflag_fit=errflag)


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
        quantile_prior=photoz.quantile_prior_mask(config, batch),
        abort_on_missing_qzphot=config.fit.abort_on_missing_qzphot)
    batch = dataclasses.replace(
        batch,
        z_priors=(photoz.redshift_priors(config, batch)
                  if config.photoz.dofit_photoz else None),
        sncid_inival=(_sncid_inival(listed[2], np.asarray(batch.snids),
                                    parameter_names)
                      if apply_selection and config.selection.opt_sncid_list & 2 else None))
    if batch.z_priors is not None:
        supported = np.asarray(photoz.pdf_support_mask(config, batch.z_priors))
        if not supported.all():
            badq = np.concatenate((badq, np.asarray(batch.snids)[~supported]))
            slog.line(logger, f"Skipped {int((~supported).sum())} SNe whose host z-PDF "
                              f"has no positive probability within PHOTOZ_BOUND")
            batch = batch.filter_by_mask(supported)
    ladder.append(('FITTABLE', batch.n_sne))
    marks = ((selected, ERRFLAG_FIT_CUTS),
             (empty, ERRFLAG_FITPREP_NOEPOCHS),
             (thin, ERRFLAG_FITPREP_NEPOCH),
             (badq, ERRFLAG_FITPREP_QUANTILES))
    return batch, prepared, marks


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
        slog.warn(logger, "SNTABLE_LIST", f"ignoring {msg}")
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
    ladder = []
    batch, precut_batch, marks = prepare_batch(
        config, list(fit_parameters(config, model)), precut_batch,
        apply_selection=True, ladder=ladder)
    if not keep_preselection:
        precut_batch = (_record(precut_batch, *marks)
                        if 'snana' in config.output.tables else None)
    slog.line(logger, f"Loaded and batched {batch.n_sne} SNe, "
                      f"max_obs={batch.max_obs}")
    return batch, precut_batch, ladder, read_seconds


def fit_sizes(n_obs, config):
    """``(epoch_lengths, batch_size)`` for a fitter fitting samples whose
    epoch counts are ``n_obs``, one array per sample: epoch bins chosen over
    all of them together, and BATCH_SIZE events per padded batch."""
    return (choose_epoch_lengths(np.concatenate([np.asarray(n) for n in n_obs]),
                                 config.fit.num_nobs_bins,
                                 config.fit.min_bin_gain),
            config.fit.batch_size)


def batch_needs(config, batch):
    """What fitting ``batch`` under ``config`` asks of a fitter's shape: its
    light-curve lengths and, under photo-z, the coarse grid its redshift
    priors span (:func:`~jax_snlc_fit.photoz.scan_size`)."""
    return {'n_obs': np.asarray(batch.n_obs_per_sn),
            'n_z': (photoz.scan_size(config, batch.z_priors)
                    if config.photoz.dofit_photoz else None)}


def shape_for(config, needs, windows):
    """The shape a :class:`LightCurveFitter` is built at to fit every batch
    whose :func:`batch_needs` are in ``needs``, with the band ``windows`` of
    their photometry, at one set of compiled shapes: the epoch lengths and
    batch size :func:`fit_sizes` gives over all their light curves, by
    ``config``'s BATCH_SIZE, NUM_NOBS_BINS and MIN_BIN_GAIN, and the widest
    band window and coarse grid. A plain dict, which a pipeline records."""
    epoch_lengths, batch_size = fit_sizes([need['n_obs'] for need in needs],
                                          config)
    n_z = [need['n_z'] for need in needs if need['n_z'] is not None]
    return {'epoch_lengths': [int(n) for n in epoch_lengths],
            'batch_size': int(batch_size), 'band_window': _widest(windows),
            'n_z': max(n_z) if n_z else None}


def _widest(windows):
    """The band window covering every one of ``windows``: None, the whole
    grid, where any is."""
    return None if None in windows else max(windows)


def prepare_fit_shape(fitter, config, batch):
    """Extend a prepared fitter using the same shape rules as file inputs."""
    if not batch.n_sne:
        return
    shape = shape_for(config, [batch_needs(config, batch)],
                      [fit_band_window(fitter.model, batch.photometry)])
    if fitter.epoch_lengths:
        shape['epoch_lengths'] = list(fitter.epoch_lengths)
        shape['epoch_lengths'][-1] = max(
            shape['epoch_lengths'][-1], int(np.max(batch.n_obs_per_sn)))
        if shape['n_z'] is not None:
            shape['n_z'] = max(fitter.n_z or 1, shape['n_z'])
        shape['band_window'] = _widest(
            [fitter.static_data['band_window'], shape['band_window']])
    fitter.prepare_shape(shape['epoch_lengths'], shape['n_z'],
                         resolved_filters(config, fitter.calibration), shape['band_window'])


def prepare(config, model, kcor, batch):
    """The fitter for ``batch`` alone."""
    return LightCurveFitter._from_prepared(config, model, kcor, shape_for(
        config, [batch_needs(config, batch)],
        [fit_band_window(model, batch.photometry)]))


def post_fit_cuts(fitter, config, fitwins, batch, results):
    """FITANA_CUTS on the finished fit of a data version ``config`` names;
    ``fitwins`` are its ``FITWIN_`` windows (:func:`fitwin_windows`).
    Returns the events kept, the ERRFLAG_FIT of each event the fit itself
    abandoned (a round-1 window or ZFIT_RANGE), 0 elsewhere, and one CUTSTAT
    row per cut.
    """
    model = fitter.model
    rows = []
    lam = batch.photometry['filter_mean_wavelengths']
    n_total = results.n_sne
    fit_mask = np.asarray(results.observation_mask)
    keep = np.ones(n_total, dtype=bool)
    codes = np.zeros(n_total, dtype=int)
    # OPT_SNCID_LIST bit 0 skips the other post-fit cuts (snlc_fit.F90:12808)
    if config.selection.opt_sncid_list & 1:
        keep = narrow(keep, 'SNCID_PEGGED',
                      _sncid_pegged_mask(config, model, results, fitwins),
                      rows)
    else:
        if results.iter1_verdicts:
            # the last rejecting window's code wins, as in SNANA
            iter1_keep = np.ones(n_total, dtype=bool)
            iter1_codes = np.zeros(n_total, dtype=int)
            for name, errflag in ITER1_ERRFLAG:
                verdict = results.iter1_verdicts.get(name)
                if verdict is None:
                    continue
                verdict = np.asarray(verdict)
                iter1_codes = np.where(verdict, iter1_codes, errflag)
                iter1_keep &= verdict
            codes = np.where(iter1_keep, codes, iter1_codes)
            keep = narrow(keep, 'FITWIN_ITER1', iter1_keep, rows)

        # LTEST_ZPHFILTER (snlc_fit.F90:3569-3572, :3616-3630)
        if config.photoz.dofit_photoz and config.fit.n_fit_iterations > 1:
            z_min, z_max = map(np.asarray, photoz.band_z_ranges(
                fitter.restlambda_usefit, lam))
            zphot = np.asarray(results.params['zPHOT'])[:, None]
            zerr = np.asarray(results.param_errors['zPHOT'])[:, None]
            # the fit bands are the first indices, in z_min's order; a band
            # the data has beyond them is numbered after
            fitted_band = band_snr_maxima(batch, mask=fit_mask)[2][
                :, :fitter.n_filters]
            outside = fitted_band & ((zphot - zerr < z_min) | (zphot + zerr > z_max))
            zfit_ok = fitted_band.any(axis=1) & ~outside.any(axis=1)
            codes = np.where(keep & ~zfit_ok, ERRFLAG_ZFIT_RANGE, codes)
            keep = narrow(keep, 'ZFIT_RANGE', zfit_ok, rows)

        cutwin, _, photoz_trest_nominal = cut_windows(config)
        fitted_z = _fitted_z(config, results.params, batch.redshifts)
        # the last round's errors, as SNANA accumulates SNRMAX on its final
        # iteration (snlc_fit.F90:4411)
        nfilt_keep = _nfilt_snrmax_mask(
            cutwin, batch, fit_mask, fitter.filters,
            fitter.fudged_fluxcalerrs(
                lam, config.fit.fudgeall_maxfrac,
                np.asarray(results.params['PKMJD']) - batch.reference_mjds,
                np.asarray(fitted_z), batch.mjd_offsets,
                batch.filter_indices, fit_mask, batch.fluxcals,
                batch.fluxcalerrs, config.fit.n_fit_iterations))
        if nfilt_keep is not None:
            keep = narrow(keep, 'NFILT_SNRMAX_FIT', nfilt_keep, rows)
        keep = narrow(keep, 'TREST', _post_fit_trest_mask(
            config, photoz_trest_nominal, cutwin, results, batch, fit_mask),
            rows)
        # always run: two of the windows are hard-wired defaults in SNANA
        keep = narrow(keep, 'FITWIN', _fitwin_mask(
            fitwins, model, results,
            np.asarray(fitted_z),
            _reference_mu(config, results, batch), batch, config), rows)
        keep = narrow(keep, 'PEAKMJD',
                      _peakmjd_window_mask(config, results), rows)
    if config.fit.check_crazyfiterr:
        keep = narrow(keep, 'CRAZYFITERR', _fit_error_bounds_mask(
            fitter.fit_parameters, fitter.held_params, results), rows)

    # MXLC_FIT caps fitted events (N_SNLC_FITCUTS, snana.F90:2895)
    if keep.sum() > config.selection.mxlc_fit:
        cap = np.flatnonzero(keep)[config.selection.mxlc_fit:]
        keep[cap] = False
        rows.append(('MXLC_FIT', int(np.count_nonzero(keep))))
    return keep, codes, rows


def select(batch, results, precut_batch, keep, codes):
    """Select output rows using the shared fit verdict and record its codes."""
    written = keep
    if precut_batch is not None:
        snids = np.asarray(results.snids)
        precut_batch = _record(
            precut_batch,
            *[(snids[codes == code], code)
              for code in np.unique(codes[codes != 0])],
            (snids[written], 0))
    return (batch.filter_by_mask(written), results.filter_by_mask(written),
            precut_batch)


def _snana_flags(precut_batch):
    """CUTFLAG_SNANA and ERRFLAG_FIT for every loaded event.

    CUTFLAG's first bit is set by the SNLC cuts (snana.F90:21459) and its
    second only once the fit finishes clean (snlc_fit.F90:2315), so an event
    dropped by the post-fit cuts keeps 1. ERRFLAG_FIT is the code
    FITPAR_ANA returns through its third argument (snana.F90:9415), which is
    the no-fit sentinel only for an event the selection dropped.
    """
    errflag = np.asarray(precut_batch.errflag_fit)
    cutflag = np.where(errflag == 0, 3,
                       np.where(errflag != ERRFLAG_NO_FIT, 1, 0))
    return cutflag, errflag


def _batch_truth(batch, column):
    """A simulated-truth column for one batch, for the USESIM_* keys."""
    truth = batch.sim_truth or {}
    if column not in truth:
        raise SNANAError(
            f"USESIM needs the truth column {column}, which this data version "
            f"does not carry. Available: {', '.join(sorted(truth)) or 'none'}.",
            function_name="_batch_truth")
    # FITS columns are big-endian (>f4), which JAX will not accept.
    return np.asarray(truth[column], dtype=np.float64)


def _model_type(config):
    """The fitted model's type, from MODEL_TYPE or FITMODEL_NAME."""
    if config.model.fitmodel_name is None:
        raise SNANAError(
            "FITMODEL_NAME is required in the input file",
            function_name="_discover_inputs")
    model_type = selected_model({'model_type': config.model.model_type,
                                 'fitmodel_name': config.model.fitmodel_name})
    if model_type is None:
        raise SNANAError(
            f"Cannot infer model type from FITMODEL_NAME "
            f"'{config.model.fitmodel_name}'. Set MODEL_TYPE "
            f"({', '.join(MODEL_REGISTRY)}).",
            function_name="_discover_inputs")
    return model_type


def _discover_inputs(config):
    """Resolve the data versions, locate the data files, and run the
    REQUIRE_DOCANA checks. Returns the versions' directories and the first
    one's HEAD path."""
    d = config.data
    if d.kcor_file is None:
        raise SNANAError(
            "KCOR_FILE is required", function_name="_discover_inputs")
    if d.version_photometry_wildcard:
        # glob data folders; each match's basename is a version (snana.F90:6094)
        pattern = os.path.expandvars(d.version_photometry_wildcard)
        sim_dirs = sorted(p for p in glob.glob(pattern) if Path(p).is_dir())
        if not sim_dirs:
            raise SNANAError(
                f"Found zero data versions matching "
                f"VERSION_PHOTOMETRY_WILDCARD = {pattern}",
                function_name="PREP_VERSION_PHOTOMETRY")
        slog.line(logger, f"Wildcard matched {len(sim_dirs)} data version(s): "
                          f"{', '.join(Path(p).name for p in sim_dirs)}")
    elif d.version_photometry:
        sim_dirs = [resolve_version_dir(d.version_photometry,
                                        d.private_data_path)]
    else:
        raise SNANAError(
            "VERSION_PHOTOMETRY or VERSION_PHOTOMETRY_WILDCARD is required",
            function_name="_discover_inputs")

    data_path = version_sources(sim_dirs[0],
                                config.data.photometry_columns)[0][0]
    sim_dir = sim_dirs[0]
    # the version README and the model's metadata files (snana.F90:3042)
    model_files = metadata_files(MODEL_REGISTRY[_model_type(config)],
                                 config.model.fitmodel_name)
    _check_docana(config.fit.require_docana,
                  sorted(glob.glob(str(Path(sim_dir) / '*.README')))
                  + list(model_files))
    # FLUXERRMODEL_OPTMASK 1024: the same check on the map file (sntools_fluxErrModels.c:85)
    if config.data.fluxerrmodel_optmask & 1024:
        _check_docana(1, [os.path.expandvars(p) for p in
                          (config.data.fluxerrmodel_file,
                           config.data.sim_fluxerrmodel_file) if p is not None])
    slog.banner(logger, "Read light curves")
    slog.kv(logger, "DATA", data_path)
    return sim_dirs, data_path


def _simulation_cosmology(sim_dirs):
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


def fit_model(config):
    """The model a fit of ``config`` fits, built once before the data is
    read, since the loader carries the truth columns its fit spec names.

    Shared construction path with the simulator. The grid spans the model's
    own support (post-UVLAM) at the fitter's sampling density; SNANA's
    fitter likewise initialises the model over its native range, and it
    deliberately passes no late-time extrapolation file (snlc_fit.F90
    initialises the model with a blank late-time argument).
    """
    require_precision(config.fit.double_precision)
    model_type = _model_type(config)
    model_path = os.path.expandvars(os.path.expanduser(config.model.fitmodel_name))
    slog.line(logger, "Building the model...")
    model_instance = build_model(model_path, config.model._model_key_values,
                                 model_type=model_type)
    if not isinstance(model_instance, FittedModel):
        raise SNANAError(
            f"Model type '{model_type}' cannot be fitted: it does not play "
            f"the FittedModel role.", function_name="run_fitting")
    return model_instance


def _log_cutstat(ladder):
    """The CUTSTAT block: events still surviving after each cut, in order.

    A row appears only where the count falls, as SNANA prints its incremental
    cut table (snana.F90:8925-8950). Every cut narrows the sample through
    ``narrow``, so the last row is the number of rows written.
    """
    slog.banner(logger, "CUTSTAT")
    slog.line(logger, "Number of SN passing each cut, in order")
    previous = None
    for name, surviving in ladder:
        if previous is None or surviving < previous:
            logger.info(f"CUTSTAT:  {name:<24s} {surviving:8d}")
        previous = surviving


def _log_summary(jax_results, dt_total):
    """The FIT_SUMMARY block."""
    n_total = jax_results.n_sne
    rate_total = n_total / dt_total if dt_total > 0 else 0
    chi2_dof = np.array(jax_results.chi_squared / jax_results.ndof)
    rows = [("NFIT", n_total)]
    edm = np.array(jax_results.edm)
    positive = np.asarray(jax_results.hessian_positive)
    rows.append(("NINVALID_HESSIAN", int(np.sum(~positive))))
    valid = positive & (edm >= 0)
    n_converged = int(np.sum(valid & (edm < 0.001)))
    n_good = int(np.sum(valid & (edm < 0.01)))
    rows.append(("NCONVERGE(EDM<0.001)", f"{n_converged}/{n_total}"))
    rows.append(("NGOOD(EDM<0.01)", f"{n_good}/{n_total}"))
    if n_total:
        rows.append(("MEDIAN(chi2/dof)", f"{np.median(chi2_dof):.4f}"))
    rows.append(("WALLTIME_MIN", f"{dt_total / 60.0:.3f}", f"{rate_total:.0f} SNe/sec"))
    slog.summary(logger, "FIT_SUMMARY", rows)
    slog.line(logger, "EDM is the estimated distance to the chi2 minimum; every "
                      "fit is written, converged or not. chi2/dof well above 1 "
                      "means the flux errors do not cover the scatter (a model "
                      "without an error term, or a SIMLIB zero-point smear).")


def _write_lcplot(fitter, batch, fit_mask, model_flux, model_flux_err, params,
                  n_plot, stem, style):
    """``<stem>.LCPLOT.TEXT`` and ``<stem>.LCLIST.TEXT`` for the first
    ``n_plot`` events: the data with their fit flags beside the fitted curve
    on LCPLOT_TOBS_GRID, the curve evaluated in the fit's batches. ``batch``
    carries its photometry and Milky Way attenuation."""
    params = {k: np.asarray(v) for k, v in params.items()}
    pkmjd = params['PKMJD']
    zeval = _fitted_z(fitter.config, params, batch.redshifts)
    n_bands, grid = len(fitter.filters), LCPLOT_TOBS_GRID
    shape = (n_plot, n_bands, len(grid))
    grid_flux, grid_flux_errors = np.zeros(shape), np.zeros(shape)
    for rows, idx in padded_batches(np.arange(n_plot), fitter.batch_size):
        grid_mjds = pkmjd[idx, None] + np.tile(grid, n_bands)[None, :]
        grid_filter_indices = np.repeat(np.arange(n_bands), len(grid))[None, :].repeat(
            len(idx), axis=0)
        chunk = batch.take(idx)
        flux, flux_err = fitter.evaluate_flux(
            {k: v[idx] for k, v in params.items()},
            grid_mjds, zeval[idx], grid_filter_indices, np.ones_like(grid_mjds, dtype=bool),
            mw_att_obs=chunk.mw_att_obs, photometry=chunk.photometry,
            reference_mjds=np.asarray(chunk.reference_mjds))
        n = len(rows)
        grid_flux[rows] = np.asarray(flux)[:n].reshape(n, n_bands, len(grid))
        grid_flux_errors[rows] = np.asarray(flux_err)[:n].reshape(n, n_bands, len(grid))
    plot_idx = np.arange(n_plot)
    varnames, table_cols, table_fmt = build_lcplot_table(
        batch, fit_mask, model_flux, model_flux_err, pkmjd, plot_idx,
        grid_flux, grid_flux_errors)
    lcplot_path = Path(f"{stem}.LCPLOT.TEXT")
    write_varnames_table(lcplot_path, varnames, table_cols, table_fmt,
                         style=style, rowkey='OBS:')
    write_varnames_table(
        Path(f"{stem}.LCLIST.TEXT"), ['CID', 'IFIT'],
        {'CID': np.asarray(batch.snids)[plot_idx],
         'IFIT': np.ones(n_plot, dtype=int)},
        {'CID': '%s', 'IFIT': '%d'}, style=style,
        header_comments=LCLIST_HEADER.splitlines(), blank_separator=False)
    slog.line(logger, f"Wrote LCPLOT: {lcplot_path} ({n_plot} SNe)")


def write_tables(fitter, config, full_batch, jax_results, precut_batch,
                 ideal_columns, sim_cosmology):
    """Every requested SNTABLE for the written events, through the shared
    varnames writer, each named ``<TEXTFILE_PREFIX>.<TABLE>.TEXT`` beside
    OUTPUT_NAME. ``full_batch`` carries its Milky Way attenuation. Returns the
    FITRES table, or None when SNTABLE_LIST asks for none."""
    model_instance = fitter.model
    prefix = config.output.textfile_prefix
    for name in config.output.tables:
        if name not in ('fitres', 'snana', 'outlier', 'lcplot'):
            slog.line(logger, f"SNTABLE '{name}' not written (not yet implemented)")
    fit_mask = np.asarray(jax_results.observation_mask)
    model_flux = model_flux_err = None
    if ({'outlier', 'lcplot'} & set(config.output.tables)
            or config.output.maglim_vmax):
        model_flux, model_flux_err = evaluate_model(fitter, full_batch,
                                                    jax_results.params)

    out_dir = Path(config.output.output_name or '').parent

    outlier_style = config.output.tables.get('outlier')
    nepoch_outlier = None
    if outlier_style is not None:
        varnames, table_cols, table_fmt, summary = build_outlier_table(
            full_batch, config.output, model_flux, model_flux_err)
        cids, counts = np.unique(table_cols['CID'].astype(str),
                                 return_counts=True)
        nepoch_outlier = dict(zip(cids, counts))
        comments = list(summary)
        if config.data.version_photometry:
            comments.append(f"VERSION_PHOTOMETRY: {config.data.version_photometry}")
        comments.append(snana_version_comment())
        comments.append(program_comment("snlc_fit.py"))
        comments.append(jax_version_comment())
        comments.append("TABLE_NAME:    OUTLIER")
        outlier_path = out_dir / f"{prefix}.OUTLIER.TEXT"
        write_varnames_table(outlier_path, varnames, table_cols, table_fmt,
                             style=outlier_style, header_comments=comments)
        slog.line(logger, f"Wrote OUTLIER table: {outlier_path} "
                          f"({len(table_cols['CID'])} epochs)")

    lcplot_style = config.output.tables.get('lcplot')
    if lcplot_style is not None:
        _write_lcplot(fitter, full_batch, fit_mask, model_flux, model_flux_err,
                      jax_results.params,
                      min(config.output.mxlc, jax_results.n_sne),
                      out_dir / prefix, lcplot_style)

    snana_style = config.output.tables.get('snana')
    if snana_style is not None:
        cutflag, errflag = _snana_flags(precut_batch)
        varnames, table_cols, table_fmt = build_snana_table(
            precut_batch, config.output, cutflag=cutflag, errflag=errflag,
            zfit=jax_results.params.get('zPHOT'),
            zfit_err=jax_results.param_errors.get('zPHOT'),
            nepoch_outlier=nepoch_outlier,
            snrmax_filters=config.selection.snrmax_filters,
            restore_wrong_vpec=config.data.restore_wrong_vpec,
            quantile_spline=config.photoz.quantile_spline())
        comments = []
        if config.data.version_photometry:
            comments.append(f"VERSION_PHOTOMETRY: {config.data.version_photometry}")
        comments.append(snana_version_comment())
        comments.append(program_comment("snlc_fit.py"))
        comments.append(jax_version_comment())
        comments.append("TABLE_NAME:    SNANA")
        snana_path = out_dir / f"{prefix}.SNANA.TEXT"
        write_varnames_table(snana_path, varnames, table_cols,
                             table_fmt, style=snana_style,
                             header_comments=comments)
        slog.line(logger, f"Wrote SNANA table: {snana_path} "
                          f"({precut_batch.n_sne} events)")

    fitres_style = config.output.tables.get('fitres')
    if fitres_style is None:
        slog.line(logger, "SNTABLE_LIST has no FITRES entry; skipping FITRES output")
        return None

    output = config.output.output_name or f"{prefix}.FITRES.TEXT"

    # per-SN event/host columns in SNANA order: event, host, event tail
    event_pre, event_post, event_fmt = build_event_columns(
        full_batch, fit_mask=fit_mask,
        zfit=jax_results.params.get('zPHOT'),
        zfit_err=jax_results.param_errors.get('zPHOT'),
        nepoch_outlier=nepoch_outlier,
        snrmax_filters=config.selection.snrmax_filters,
        restore_wrong_vpec=config.data.restore_wrong_vpec)
    event_columns = dict(event_pre)
    if full_batch.host_data is not None:
        event_columns.update(build_host_columns(
            full_batch.host_data, full_batch.zp_fluxcal,
            host_text=config.output.host_text,
            host2_text=config.output.host2_text,
            zpdf_quantiles=full_batch.zpdf_quantiles,
            zpdf_percentiles=full_batch.zpdf_percentiles,
            zpdf_nbin=full_batch.zpdf_nbin,
            quantile_spline=config.photoz.quantile_spline()))
    event_columns.update(event_post)
    if full_batch.flux_baselines is not None:
        event_columns.update({f'FLUXCAL_BASELINE_{band}': base for band, base
                              in full_batch.flux_baselines.items()})
    sim_truth = full_batch.sim_truth
    if sim_truth is not None:
        sim_truth = snana_sim_block(
            sim_truth, config.output.host_text or config.output.host2_text)
    if config.output.maglim_vmax and full_batch.n_sne:
        cospar = (config.fit.h0_ref[0], config.fit.omat_ref[0], config.fit.olam_ref[0],
                  config.fit.w0_ref[0], config.fit.wa_ref[0])
        event_columns.update(compute_vmax(
            fitter, jax_results, full_batch, fit_mask, model_flux,
            config.output.maglim_vmax, cospar))

    redshifts = np.asarray(full_batch.redshifts)
    fitres, table_fmt, comments = build_fitres_table(
        jax_results, redshifts,
        _reference_mu(config, jax_results, full_batch), model=model_instance,
        ideal_columns=ideal_columns,
        sim_truth=None if config.output.nosimvar else sim_truth,
        event_columns=event_columns, event_fmt=event_fmt,
        opt_photoz=config.photoz.opt_photoz,
        version_photometry=config.data.version_photometry,
        sim_cosmology=sim_cosmology,
        physical_only=config.output.nonuisance)
    write_varnames_table(
        output, fitres.varnames,
        {c: fitres.df[c].values for c in fitres.varnames}, table_fmt,
        style=fitres_style, header_comments=comments)
    return fitres


def _inputs(config):
    """The data versions, the first one's HEAD path, the kcor and the model."""
    sim_dirs, data_path = _discover_inputs(config)
    return sim_dirs, data_path, read_kcor(config), fit_model(config)


def run_fitting(config: FitConfig, timings=None, process_start=None):
    """Load light curves, fit, write tables: the phases of snlc_fit.exe.

    Returns the fit results and the FITRES table, the latter None when
    SNTABLE_LIST asked for no FITRES. Every table is still written; a caller
    that goes on to use the results needs no round trip through the files.

    ``timings`` is an optional dict, filled with seconds per phase (the
    :data:`FIT_PHASES` this function times; a caller may have put ``startup``
    and ``config`` there already) for callers that report the breakdown,
    plus ``batches``, one record per fitted
    batch saying whether it carried compilation. ``process_start`` is the
    ``time.perf_counter()`` the process began at, before its imports; the
    FIT_TIMING block measures the wall clock from it, or from this call when
    the caller has none.
    """
    if timings is None:
        timings = {}
    if process_start is None:
        process_start = time.perf_counter()

    start = time.perf_counter()
    sim_dirs, data_path, kcor, model = _inputs(config)
    check_photoz_trest_cuts(resolved_config(config, model))
    timings['model'] = time.perf_counter() - start

    start = time.perf_counter()
    shared = read_shared_files(config)
    write_fluxerr_maps(config, shared)
    batch, precut_batch, ladder, timings['read'] = load(
        config, model, kcor, sim_dirs, shared)
    timings['batch'] = time.perf_counter() - start - timings['read']

    start = time.perf_counter()
    fitter = prepare(config, model, kcor, batch)
    timings['model'] += time.perf_counter() - start

    results, fitres, phases = _fit_and_write(
        fitter, _with_prefix(with_fit_settings(config, fitter), data_path),
        batch, precut_batch, ladder, _simulation_cosmology(sim_dirs))
    timings.update(phases)
    _log_timing(timings, time.perf_counter() - process_start)
    return results, fitres


def prepare_datasets(cells, data_dirs):
    """Prepare distinct measured datasets once and attach each cell's calibration.

    Multiple FITOPTs share raw tables and peak-estimator shapes across all
    datasets. A single cell streams its files, including large biascor samples.
    Each cell owns its dataset mapping and batch containers; measured arrays
    can be shared until its Fit stage consumes them.
    """
    prepared = [{} for _ in cells]
    photometry = [fit_photometry(cell['config'], cell['model'], cell['kcor'])
                  for cell in cells]
    groups = group_equal({
        i: (cell['config'].data, cell['config'].selection,
            cell['config'].fit, cell['config'].photoz, cell['config'].output,
            model_sim_columns(cell['model']),
            list(fit_parameters(cell['config'], cell['model'])))
        for i, cell in enumerate(cells)})

    sources = [{} for _ in cells]
    peak_shapes = [None for _ in cells]
    if len(cells) > 1:
        for same_columns in group_equal({
                i: cell['config'].data.photometry_columns
                for i, cell in enumerate(cells)}):
            config = cells[same_columns[0]]['config']
            tables = {directory: [(path, read()) for path, read in version_sources(
                directory, config.data.photometry_columns)] for directory in data_dirs}
            for i in same_columns:
                sources[i] = tables
        for group in groups:
            i = group[0]
            cell = cells[i]
            peak_shapes[i] = peak_sizes(
                [source for tables in sources[i].values() for source in tables],
                _setpkmjd_options(cell['config']), cell['shared'])

    for data_dir in data_dirs:
        for group in groups:
            i = group[0]
            cell = cells[i]
            config = _at_version(cell['config'], data_dir)
            batch, precut, ladder, read_seconds = load(
                config, cell['model'], cell['kcor'], [data_dir], cell['shared'],
                keep_preselection=bool(config.selection.opt_sncid_list),
                sources=sources[i].get(data_dir), photometry=photometry[i],
                peak_shape=peak_shapes[i])
            for j in group:
                calibrated = {**photometry[j], 'zp_fluxcal': batch.zp_fluxcal}
                prepared[j][Path(data_dir).resolve()] = (
                    dataclasses.replace(batch, photometry=calibrated),
                    dataclasses.replace(precut, photometry=calibrated)
                    if precut is not None else None,
                    list(ladder), read_seconds)
        for tables in sources:
            tables.pop(data_dir, None)

    # Epoch-bin selection weights each distinct FITOPT configuration/model.
    needs, windows = [], []
    for group in group_equal({i: (cell['config'], id(cell['model']))
                             for i, cell in enumerate(cells)}):
        i = group[0]
        cell = cells[i]
        needs.extend(batch_needs(cell['config'], prepared[i][Path(directory).resolve()][0])
                     for directory in data_dirs)
        windows.append(fit_band_window(cell['model'], photometry[i]))
    return shape_for(cells[0]['config'], needs, windows), prepared


def fit_settings(config):
    """What a fitter is built from: configs equal in it share a fitter."""
    return config.model, config.fit, config.photoz


def with_fit_settings(config, fitter):
    """``config`` reading and writing its data as it says, with the fit
    settings ``fitter`` resolved against its model."""
    return dataclasses.replace(config, model=fitter.config.model,
                               fit=fitter.config.fit,
                               photoz=fitter.config.photoz)


def _at_version(config, data_dir):
    """``config`` reading the data version in the directory ``data_dir``."""
    data_dir = Path(data_dir)
    return dataclasses.replace(config, data=dataclasses.replace(
        config.data, version_photometry=data_dir.name,
        private_data_path=str(data_dir.parent),
        version_photometry_wildcard=None))


def dataset_config(config, data_dir, stem):
    """``config`` reading the data version in ``data_dir`` and writing its
    tables as ``<stem>.<TABLE>.TEXT``, for :func:`fit_dataset`."""
    located = _at_version(config, data_dir)
    return dataclasses.replace(located, output=dataclasses.replace(
        located.output, output_name=f"{stem}.FITRES.TEXT",
        textfile_prefix=Path(stem).name))


def fit_dataset(fitter, config, kcor, shared, *, prepared_data=None):
    """Load, fit, cut and write the data version ``config`` names with
    ``fitter``, as :func:`run_fitting` does, so a fitter fitting one dataset
    after another compiles nothing after the first. ``config`` carries the
    fitter's fit settings (:func:`with_fit_settings`) and that version's data
    and outputs (:func:`dataset_config`); ``kcor`` and ``shared`` are its
    kcor and :func:`read_shared_files`. Returns the fit results and the
    FITRES table, as :func:`run_fitting`.

    ``prepared_data`` is the retained result of the upfront load, or None when
    this pipeline execution restored only the fitter's saved shape.
    """
    sim_dirs, data_path = _discover_inputs(config)
    # FITOPTs sharing a fitter can name different kcor files
    filters = resolved_filters(config, kcor)
    if filters != fitter.filters:
        raise SNANAError(
            f"{data_path} is fitted in the bands {filters}; the fitter was "
            f"built for {fitter.filters}", function_name="fit_dataset")
    if prepared_data is None:
        prepared_data = load(config, fitter.model, kcor, sim_dirs, shared)
    elif config.selection.opt_sncid_list:
        # A systematic can use the nominal fit's CID list, which did not
        # exist during upfront sizing. Reapply selection without rereading.
        ladder = []
        batch, precut, marks = prepare_batch(
            config, fitter.param_names, prepared_data[1],
            apply_selection=True, ladder=ladder)
        precut = _record(precut, *marks) if 'snana' in config.output.tables else None
        prepared_data = batch, precut, ladder, prepared_data[3]
    batch, precut_batch, ladder, _ = prepared_data
    config = _with_prefix(config, data_path)
    write_fluxerr_maps(config, shared)
    results, fitres, _ = _fit_and_write(fitter, config, batch, precut_batch,
                                        ladder, _simulation_cosmology(sim_dirs))
    return results, fitres


def _with_prefix(config, data_path):
    """``config`` with TEXTFILE_PREFIX defaulted to the data version's name."""
    return dataclasses.replace(config, output=dataclasses.replace(
        config.output, textfile_prefix=(
            config.output.textfile_prefix
            or Path(data_path).name.replace("_HEAD.FITS", ""))))


def fitwin_windows(fitter, config):
    """The ``FITWIN_`` windows of ``config`` against the FITRES columns of
    ``fitter``'s model (:func:`resolve_fitwin_windows`)."""
    return resolve_fitwin_windows(config, fitter.model, fitter.param_names,
                                  fitter.aliases)


def as_fitted(fitter, config, batch):
    """``batch`` as the fit sees it: padded to the fitter's longest epoch
    length, which the model flux of every table is evaluated at
    (:func:`evaluate_model`); with the simulated true flux under
    USESIM_TRUEFLUX (snana.F90:17341); and with its Milky Way attenuation."""
    prepare_fit_shape(fitter, config, batch)
    if not batch.n_sne:
        return batch
    batch = batch.take(np.arange(batch.n_sne), fitter.epoch_lengths[-1])
    if config.data.usesim_trueflux:
        batch = _substitute_true_flux(batch, config.data.magobs_shift_zp,
                                      'USESIM_TRUEFLUX')
    return attach_mw_attenuation(config, batch)


def finish_fit(fitter, config, batch, results, start_params, *,
               apply_selection, fitwins=None):
    """Check fit validity, apply requested cuts and correct covariances.

    Retain every fitted row here; file output can select rows afterwards,
    while Python can report the outcome of every supplied object.
    """
    results = errmax_bad(config, fitter.fit_parameters, fitter.held_params,
                         results, start_params)
    if apply_selection and results.n_sne:
        keep, codes, rows = post_fit_cuts(
            fitter, config, fitwin_windows(fitter, config) if fitwins is None else fitwins,
            batch, results)
    else:
        keep = np.ones(results.n_sne, dtype=bool)
        codes = np.zeros(results.n_sne, dtype=int)
        rows = []
    # Model wavelength/phase support can leave a prepared object without data.
    # This is fit validity, independent of optional scientific sample cuts.
    has_observations = np.asarray(results.observation_mask).any(axis=1)
    keep &= has_observations
    codes = np.where(has_observations, codes, ERRFLAG_FITPREP_NOEPOCHS)
    finite = np.all(np.stack([np.isfinite(v) for v in results.params.values()]), axis=0)
    keep &= finite
    codes = np.where(~finite, ERRFLAG_MNFIT_NAN,
                     np.where(~keep & (codes == 0), ERRFLAG_FIT_CUTS, codes))
    if config.fit.opt_covar_lcfit and keep.any():
        indices = np.flatnonzero(keep)
        corrected = uncertainties.fix_covar_lcfit(
            results.take(indices), fitter.model, config.fit.opt_covar_lcfit)
        errors = {name: np.array(value) for name, value in results.param_errors.items()}
        for name, values in corrected.param_errors.items():
            errors[name][indices] = values
        covariance = np.array(results.param_covariances)
        covariance[indices] = corrected.param_covariances
        results = dataclasses.replace(results, param_errors=errors, param_covariances=covariance)
    return results, keep, codes, rows


def fit_prepared(fitter, config, batch, *, apply_selection, progress=False):
    """Run the common fit lifecycle on a calibrated, fittable sample.

    Both interfaces use the same shape preparation, starting values, optimiser
    and result corrections. Sample cuts are explicit; writing files is left
    to the caller. Results retain every fitted object alongside its verdict.
    """
    timings = {}
    # a misspelt FITWIN_ suffix aborts before anything is fitted
    fitwins = fitwin_windows(fitter, config) if apply_selection else None
    batch = as_fitted(fitter, config, batch)

    slog.banner(logger, "Fit lightcurves")
    start = time.perf_counter()
    results, start_params, timings['batches'] = fit(fitter, config, batch, progress=progress)
    fit_seconds = time.perf_counter() - start
    # First-batch time is an upper bound: compatible fitters may already
    # share compiled functions, and execution is included in this estimate.
    timings['compile'] = sum(b['seconds'] for b in timings['batches']
                             if b['carries_jit'])
    timings['fit'] = fit_seconds - timings['compile']

    start = time.perf_counter()
    results, keep, codes, cut_rows = finish_fit(
        fitter, config, batch, results, start_params,
        apply_selection=apply_selection, fitwins=fitwins)
    timings['cuts'] = time.perf_counter() - start
    return batch, results, keep, codes, cut_rows, timings


def _fit_and_write(fitter, config, batch, precut_batch, ladder, sim_cosmology):
    """Fit a loaded data version, cut and write its tables. ``config`` is
    that version's, with the fitter's fit settings, ``ladder`` the CUTSTAT
    rows its load recorded and ``sim_cosmology`` its README's
    (:func:`_simulation_cosmology`). Returns the fit results, the FITRES table, and
    the seconds of the compile, fit, cuts and write phases with the record
    of each fitted batch."""
    batch, results, keep, codes, cut_rows, timings = fit_prepared(
        fitter, config, batch, apply_selection=True)
    fit_seconds = timings['fit'] + timings['compile']
    start = time.perf_counter()
    batch, results, precut_batch = select(batch, results, precut_batch,
                                          keep, codes)
    ideal_columns = (ideal_fit(fitter, config, batch)
                     if config.fit.simfit_ideal_prescale else None)
    timings['cuts'] += time.perf_counter() - start

    _log_cutstat(ladder + cut_rows)
    _log_summary(results, fit_seconds)
    start = time.perf_counter()
    fitres = write_tables(fitter, config, batch, results, precut_batch,
                          ideal_columns, sim_cosmology)
    timings['write'] = time.perf_counter() - start
    return results, fitres, timings


def lcplot_from_fitres(config: FitConfig, fitres_path, prefix):
    """``<prefix>.LCPLOT.TEXT`` and ``<prefix>.LCLIST.TEXT`` for the events of
    a FITRES table, from the namelist that fitted them, without refitting.

    The events are those the config names (SNCID_LIST, SNCCID_LIST,
    SNCID_LIST_FILE), or every event of the FITRES when it names none. They
    load as the fit loaded them, the parameters come from
    the FITRES and the fitted bands from its BANDLIST. The fit sets its
    FITWIN_TREST window and its epoch rejects at its round-1 PKMJD and z,
    which the FITRES does not hold, so here they are at the final values, and
    an epoch can be flagged differently wherever the fit moved after round 1.
    """
    table = read_fitres(fitres_path)
    by_cid = table.df.set_index(table.df['CID'].astype(str))
    wanted, _, _ = sncid_list(config)
    if not wanted:
        wanted = list(by_cid.index)
    # the list as strings, which match a CID or a name, so the loader reads
    # no file again
    selection = dataclasses.replace(
        config.selection, sncid_list=None, sncid_list_file=None,
        snccid_list=wanted)
    style = (config.output.tables.get('lcplot')
             or parse_sntable_list('LCPLOT')['tables']['lcplot'])
    plotting = dataclasses.replace(
        config, selection=selection,
        output=dataclasses.replace(config.output, tables={'lcplot': style}))
    sim_dirs, _, kcor, model = _inputs(plotting)
    batch, _, _, _ = load(plotting, model, kcor, sim_dirs,
                          read_shared_files(plotting))
    fitter = prepare(plotting, model, kcor, batch)
    batch = as_fitted(fitter, plotting, batch)

    snids = np.char.strip(np.asarray(batch.snids).astype(str))
    # an entry has loaded when it is an event's SNID or one of its names
    wanted = np.asarray(wanted, dtype=str)
    found = np.isin(cid_keys(wanted), cid_keys(snids))
    for col in ('NAME_TRANSIENT', 'NAME_IAUC'):
        if col in batch.head_extras:
            found |= np.isin(wanted, np.char.strip(
                np.asarray(batch.head_extras[col]).astype(str)))
    unloaded = sorted(set(wanted[~found]))
    if unloaded:
        raise SNANAError(f"{len(unloaded)} events do not load with this "
                         f"namelist: {unloaded[:10]}",
                         function_name="lcplot_from_fitres")
    fitted = by_cid.loc[snids]

    params = ({'zPHOT': fitted['zHEL'].to_numpy()} if plotting.photoz.dofit_photoz
              else {})
    z = _fitted_z(plotting, params, batch.redshifts)
    # each parameter's own column, else the one its model writes it as
    written_as = dict(model.fitres_param_columns(fitter.model_param_names))
    params.update({name: fitted[name if name in fitted else written_as[name]]
                   .to_numpy() for name in fitter.model_param_names})

    band_chars = [filter_band(batch.filter_names[b])
                  for b in range(len(batch.filter_names))]
    in_bandlist = np.array([[c in bands for c in band_chars]
                            for bands in fitted['BANDLIST'].astype(str)])
    fidx = np.asarray(batch.filter_indices)
    reference_mjds = np.asarray(batch.reference_mjds)
    fit_mask = (np.asarray(batch.observation_mask)
                & np.take_along_axis(in_bandlist, fidx, axis=1)
                & np.asarray(fitter.fitwin_trest_ok(
                    fidx, np.asarray(batch.mjd_offsets),
                    params['PKMJD'] - reference_mjds, z)))
    # the epoch rejects, in the fit's batches
    physical = fitter.stack(params, reference_mjds)
    for epoch_length, rows, idx in plan_fit_batches(
            batch.n_obs_per_sn, fitter.epoch_lengths, fitter.batch_size):
        chunk = batch.take(idx, epoch_length)
        kept = fitter.reject_epochs(
            fitter._model_input, chunk.photometry, fit_mask[idx, :epoch_length], physical[idx],
            chunk.mjd_offsets, chunk.filter_indices, chunk.fluxcals,
            chunk.fluxcalerrs, chunk.reference_mjds, z[idx],
            fitter.events_band_weights(chunk.photometry, jnp.asarray(z[idx]),
                                       chunk.mw_att_obs))
        fit_mask[np.ix_(rows, np.arange(epoch_length))] = (
            np.asarray(kept)[:len(rows)])

    model_flux, model_flux_err = evaluate_model(fitter, batch, params)
    _write_lcplot(fitter, batch, fit_mask, model_flux, model_flux_err, params,
                  batch.n_sne, prefix, style)


def _log_timing(timings, wall):
    """The FIT_TIMING block: seconds per phase against the process wall
    clock, and one line of the same for grep. Accounted falls short of the
    wall by whatever was not timed."""
    phases = [(name, timings[name]) for name in FIT_PHASES if name in timings]
    accounted = sum(seconds for _, seconds in phases)
    rows = [(name, f"{seconds:8.2f} s", f"{100.0 * seconds / wall:5.1f}%")
            for name, seconds in phases]
    rows.append(("Accounted", f"{accounted:8.2f} s",
                 f"{100.0 * accounted / wall:5.1f}%"))
    rows.append(("Wall clock", f"{wall:8.2f} s"))
    slog.summary(logger, "FIT_TIMING", rows)
    slog.line(logger, "Fit phases: " + "  ".join(
        f"{name} {seconds:.1f}s" for name, seconds in phases)
        + f"  total {wall:.1f}s")
