"""The `snlc_fit` command: what snlc_fit.exe does between reading its namelist
and writing its tables. `run_fitting` orchestrates; `snlc_fit.py` is the CLI
shim that parses inputs and calls it. The pipeline fits one data version after
another through `prepare_datasets`, `dataset_config` and `fit_dataset`, and
`lcplot_from_fitres` rebuilds the LCPLOT and LCLIST tables from an existing
FITRES.
"""

import dataclasses
import logging
import time
from pathlib import Path

import numpy as np

from jax_snlc_core import logging as log
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import read_fitres
from jax_snlc_core.utils import group_equal
from jax_snlc_core.vmax import compute_vmax

from . import photoz
from .calibration import fit_photometry, read_kcor, resolved_filters
from .config import FitConfig
from .config_parser import parse_sntable_list
from .cuts import cid_keys, select, sncid_list
from .fitter import (batch_needs, fit_band_window, fit_model, prepare, resolved_config,
                     shape_for)
from .loader import (discover_inputs, load, model_sim_columns, peak_sizes, read_shared_files,
                     select_for_fit, setpkmjd_options, simulation_cosmology, version_sources)
from .output import write_fluxerr_maps, write_lcplot, write_tables
from .parameters import fit_parameters

logger = logging.getLogger(__name__)

# FIT_TIMING phases, in order
FIT_PHASES = ('startup', 'config', 'read', 'batch', 'model', 'compile', 'fit',
              'cuts', 'write')


def _log_cutstat(ladder):
    """The CUTSTAT block: events still surviving after each cut, in order.

    A row appears only where the count falls, as SNANA prints its incremental
    cut table (snana.F90:8925-8950). Every cut narrows the sample through
    ``narrow``, so the last row is the number of rows written.
    """
    log.banner(logger, "CUTSTAT")
    log.line(logger, "Number of SN passing each cut, in order")
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
    log.summary(logger, "FIT_SUMMARY", rows)
    log.line(logger, "EDM is the estimated distance to the chi2 minimum; every "
             "fit is written, converged or not. chi2/dof well above 1 "
             "means the flux errors do not cover the scatter (a model "
             "without an error term, or a SIMLIB zero-point smear).")


def _inputs(config):
    """The data versions, the first one's HEAD path, the kcor and the model."""
    sim_dirs, data_path = discover_inputs(config)
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
    # checks the keys against the model before any data are read
    resolved_config(config, model)
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
        fitter, _with_prefix(fitter.with_fit_settings(config), data_path),
        batch, precut_batch, ladder, simulation_cosmology(sim_dirs))
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
                setpkmjd_options(cell['config']), cell['shared'])

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
    sim_dirs, data_path = discover_inputs(config)
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
        batch, precut, ladder = select_for_fit(config, fitter.param_names, prepared_data[1])
        prepared_data = batch, precut, ladder, prepared_data[3]
    batch, precut_batch, ladder, _ = prepared_data
    config = _with_prefix(config, data_path)
    write_fluxerr_maps(config, shared)
    results, fitres, _ = _fit_and_write(fitter, config, batch, precut_batch,
                                        ladder, simulation_cosmology(sim_dirs))
    return results, fitres


def _with_prefix(config, data_path):
    """``config`` with TEXTFILE_PREFIX defaulted to the data version's name."""
    return dataclasses.replace(config, output=dataclasses.replace(
        config.output, textfile_prefix=(
            config.output.textfile_prefix
            or Path(data_path).name.replace("_HEAD.FITS", ""))))


def _fit_and_write(fitter, config, batch, precut_batch, ladder, sim_cosmology):
    """Fit a loaded data version, cut and write its tables. ``config`` is
    that version's, with the fitter's fit settings, ``ladder`` the CUTSTAT
    rows its load recorded and ``sim_cosmology`` its README's
    (:func:`~jax_snlc_fit.loader.simulation_cosmology`). Returns the fit
    results, the FITRES table, and the seconds of the compile, fit, cuts and
    write phases with the record of each fitted batch."""
    batch, results, keep, codes, cut_rows, timings = fitter.fit_prepared(config, batch, apply_selection=True)
    fit_seconds = timings['fit'] + timings['compile']
    start = time.perf_counter()
    batch, results, precut_batch = select(batch, results, precut_batch,
                                          keep, codes)
    ideal_columns = (fitter.ideal_fit(config, batch)
                     if config.fit.simfit_ideal_prescale else None)
    model_flux = model_flux_err = None
    if ({'outlier', 'lcplot'} & set(config.output.tables)
            or config.output.maglim_vmax):
        model_flux, model_flux_err = fitter.evaluate_model(batch, results.params)
    vmax_columns = {}
    if config.output.maglim_vmax and batch.n_sne:
        cospar = (config.fit.h0_ref[0], config.fit.omat_ref[0], config.fit.olam_ref[0],
                  config.fit.w0_ref[0], config.fit.wa_ref[0])
        vmax_columns = compute_vmax(
            fitter.model, results.params, config.output.maglim_vmax, cospar,
            evaluate_flux=fitter.evaluate_flux,
            mjds=np.asarray(batch.reference_mjds)[:, None] + np.asarray(batch.mjd_offsets),
            reference_mjds=batch.reference_mjds,
            redshifts=batch.redshifts,
            filter_indices=batch.filter_indices, filter_names=batch.filter_names,
            fit_mask=np.asarray(results.observation_mask), model_flux=model_flux,
            zp_fluxcal=batch.zp_fluxcal, mw_att_obs=batch.mw_att_obs,
            photometry=batch.photometry, snids=batch.snids)
    timings['cuts'] += time.perf_counter() - start

    _log_cutstat(ladder + cut_rows)
    _log_summary(results, fit_seconds)
    start = time.perf_counter()
    fitres = write_tables(fitter, config, batch, results, precut_batch,
                          ideal_columns, sim_cosmology, model_flux, model_flux_err,
                          vmax_columns)
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
    batch = fitter.as_fitted(plotting, batch)

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
    z = photoz.fitted_z(plotting, params, batch.redshifts)
    # each parameter's own column, else the one its model writes it as
    written_as = dict(model.fitres_param_columns(fitter.model_param_names))
    params.update({name: fitted[name if name in fitted else written_as[name]]
                   .to_numpy() for name in fitter.model_param_names})

    band_chars = batch.band_names()
    in_bandlist = np.array([[c in bands for c in band_chars]
                            for bands in fitted['BANDLIST'].astype(str)])
    fidx = np.asarray(batch.filter_indices)
    fit_mask = fitter.fitted_epochs(
        batch, params, z, np.take_along_axis(in_bandlist, fidx, axis=1))

    model_flux, model_flux_err = fitter.evaluate_model(batch, params)
    write_lcplot(fitter, batch, fit_mask, model_flux, model_flux_err, params,
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
    log.summary(logger, "FIT_TIMING", rows)
    phase_times = "  ".join(f"{name} {seconds:.1f}s" for name, seconds in phases)
    log.line(logger, f"Fit phases: {phase_times}  total {wall:.1f}s")
