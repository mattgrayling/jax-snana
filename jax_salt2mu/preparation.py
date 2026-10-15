"""Explicit conversion, correction construction and grid-local BBC preparation."""

import logging
import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core.io.fitres_reader import FitresTable, concat_tables, read_fitres_multi
from jax_snlc_core.models import standardised_model
from jax_snlc_core.utils import group_equal
from jax_snlc_core.utils import snana_log as slog
from jax_snlc_core.utils.error_handling import SNANAError
from jax_snlc_core.utils.precision import require_precision
from .config import (BBCConfig, MASK_BIASCOR_1DZAVG, MASK_BIASCOR_1DZWGT,
                     MASK_BIASCOR_CELLMAP, ccprior_enabled, parval, resolve, resolve_blinding)
from .tablevar import (EVENT_TYPE_DATA, EVENT_TYPE_BIASCOR, EVENT_TYPE_CCPRIOR,
                       CUTBIT_DUPL, CUTBIT_IDSAMPLE, _convert_table, _conversion_key,
                       _passthrough_column, isdata_real, output_varnames,
                       select_tablevar, merge_duplicates, sample_coordinates, read_data_table,
                       readonly_variables)
from .samples import partition_idsamples, assign_to_samples
from .biascor_build import BiasCorMaps, build_biascor_partitions, sample_counts
from .biascor_apply import precompute_biascor
from .biascor_1d import biascor_1d_grids, biascor_1d_data
from .ccprior import build_ccprior
from .fitter import pad_ccprior
from .distances import bias_parameters
from .output import output_path, write_passcuts_list

logger = logging.getLogger(__name__)


@contextmanager
def preparation_step(name, keys):
    """Report shared work once and identify its requesting cells on failure."""
    started = time.perf_counter()
    try:
        yield
    except (SNANAError, OSError) as error:
        message = error.message if isinstance(error, SNANAError) else str(error)
        raise SNANAError(f"BBC {name} for cells {keys}: {message}",
                         function_name=name) from error
    logger.info("BBC shared preparation %s for cells %s", name, keys)
    slog.kv(logger, "seconds", f"{name} {time.perf_counter() - started:.2f}")


def read_cells(cells):
    """Resolve configurations and read each distinct data table once."""
    named = {}
    data_sources = {}
    for key, (config, tables) in cells.items():
        with preparation_step("resolve", [key]):
            require_precision(config.double_precision)
            if tables is None:
                tables = {}
            data = tables.get(EVENT_TYPE_DATA)
            biascor = tables.get(EVENT_TYPE_BIASCOR)
            ccprior = tables.get(EVENT_TYPE_CCPRIOR)
            if data is not None and config.datafile:
                raise SNANAError("datafile names a file and a data table was supplied; give one or the other",
                                 function_name="run_bbc_fit")
            if biascor is not None and config.simfile_biascor:
                raise SNANAError("simfile_biascor names a file and a biascor table was supplied; give one or the other",
                                 function_name="run_bbc_fit")
            if ccprior is not None and config.simfile_ccprior:
                raise SNANAError("simfile_ccprior names a file and a CC-prior table was supplied; give one or the other",
                                 function_name="run_bbc_fit")
            config = resolve(config, standardised_model(config.model_lcfit),
                             biascor=biascor, ccprior=ccprior)
            if ccprior is not None and not ccprior_enabled(config):
                tables = {kind: table for kind, table in tables.items() if kind != EVENT_TYPE_CCPRIOR}
            named[key] = config, tables
            data_sources[key] = (id(data),) if data is not None else tuple(config.datafile)

    result = {}
    for keys in group_equal(data_sources):
        config, tables = named[keys[0]]
        data = tables.get(EVENT_TYPE_DATA)
        with preparation_step("read_data", keys):
            if data is None:
                data, text = read_data_table(config.datafile)
            else:
                text = None
        for key in keys:
            config, tables = named[key]
            result[key] = (resolve_blinding(config, isdata_real(data, config)),
                           {**tables, EVENT_TYPE_DATA: data}, text)
    return result


def prepare_biascor_maps(cells):
    """Build bias-correction maps, sharing conversions and event quantities."""
    partitions = {}
    biascor_sources = {}
    for key, (config, tables, _) in cells.items():
        biascor = tables.get(EVENT_TYPE_BIASCOR)
        if not config.simfile_biascor and biascor is None:
            continue
        with preparation_step("partition", [key]):
            _, partitions[key] = partition_idsamples(
                *sample_coordinates(tables[EVENT_TYPE_DATA], config), config)
        if biascor is None:
            biascor_sources[key] = tuple(config.simfile_biascor)
        elif isinstance(biascor, FitresTable):
            biascor_sources[key] = (id(biascor),)
        else:
            biascor_sources[key] = tuple(id(table) for table in biascor)

    maps = {key: None for key in cells}
    for source_keys in group_equal(biascor_sources):
        config, tables, _ = cells[source_keys[0]]
        biascor = tables.get(EVENT_TYPE_BIASCOR)
        with preparation_step("read_biascor", source_keys):
            if biascor is None:
                source = read_fitres_multi(config.simfile_biascor)
            elif isinstance(biascor, FitresTable):
                source = biascor
            else:
                source = concat_tables(biascor)
        conversions = {}
        for key in source_keys:
            config, tables, _ = cells[key]
            model = standardised_model(config.model_lcfit)
            conversions[key] = (config.model_lcfit, _conversion_key(
                EVENT_TYPE_BIASCOR, config, model, biascor=source,
                ccprior=tables.get(EVENT_TYPE_CCPRIOR)))
        for conversion_keys in group_equal(conversions):
            config, tables, _ = cells[conversion_keys[0]]
            model = standardised_model(config.model_lcfit)
            with preparation_step("read_biascor", conversion_keys):
                base = readonly_variables(_convert_table(
                    source, EVENT_TYPE_BIASCOR, config, model, biascor=source,
                    ccprior=tables.get(EVENT_TYPE_CCPRIOR)))
            settings = {key: correction_config(cells[key][0]) for key in conversion_keys}
            for settings_keys in group_equal(settings):
                config = cells[settings_keys[0]][0]
                sample_groups = group_equal({key: partitions[key] for key in settings_keys})
                unique_partitions = {keys[0]: partitions[keys[0]] for keys in sample_groups}
                one_dimensional = (config.opt_biascor & (MASK_BIASCOR_1DZAVG | MASK_BIASCOR_1DZWGT)
                                   and not config.opt_biascor & MASK_BIASCOR_CELLMAP)
                with preparation_step("biascor_maps", settings_keys):
                    if one_dimensional:
                        built = {}
                        for key, samples in unique_partitions.items():
                            tv = select_tablevar(base, source, config, model)
                            built[key] = _biascor_1d_setup(tv, samples, config, model)
                    else:
                        tv = select_tablevar(base, source, config, model)
                        dust = _force_dust_coefficients(tv, config, model)
                        if config.check_duplicates_biascor:
                            _report_biascor_duplicates(tv, model)
                        built = build_biascor_partitions(tv, unique_partitions, config, model, dust)
                for keys in sample_groups:
                    for key in keys:
                        maps[key] = built[keys[0]]
    return maps


def prepare_ccpriors(cells, maps):
    """Build simulated CC priors and pad matching numerical structures together."""
    ccprior_sources = {}
    for key, (config, tables, _) in cells.items():
        ccprior = tables.get(EVENT_TYPE_CCPRIOR)
        if not config.simfile_ccprior and ccprior is None:
            continue
        if ccprior is None:
            ccprior_sources[key] = tuple(config.simfile_ccprior)
        elif isinstance(ccprior, FitresTable):
            ccprior_sources[key] = (id(ccprior),)
        else:
            ccprior_sources[key] = tuple(id(table) for table in ccprior)

    priors = {key: None for key in cells}
    for source_keys in group_equal(ccprior_sources):
        config, tables, _ = cells[source_keys[0]]
        ccprior = tables.get(EVENT_TYPE_CCPRIOR)
        with preparation_step("read_ccprior", source_keys):
            if ccprior is None:
                source = read_fitres_multi(config.simfile_ccprior)
            elif isinstance(ccprior, FitresTable):
                source = ccprior
            else:
                source = concat_tables(ccprior)
        conversions = {}
        for key in source_keys:
            config, tables, _ = cells[key]
            model = standardised_model(config.model_lcfit)
            conversions[key] = (config.model_lcfit, _conversion_key(
                EVENT_TYPE_CCPRIOR, config, model, biascor=tables.get(EVENT_TYPE_BIASCOR),
                ccprior=source))
        for conversion_keys in group_equal(conversions):
            config, tables, _ = cells[conversion_keys[0]]
            model = standardised_model(config.model_lcfit)
            with preparation_step("read_ccprior", conversion_keys):
                base = readonly_variables(_convert_table(
                    source, EVENT_TYPE_CCPRIOR, config, model,
                    biascor=tables.get(EVENT_TYPE_BIASCOR), ccprior=source))
            requests = {key: (correction_config(cells[key][0]), id(maps[key]))
                        for key in conversion_keys}
            for keys in group_equal(requests):
                config = cells[keys[0]][0]
                mapping = maps[keys[0]]
                with preparation_step("ccprior", keys):
                    tv = select_tablevar(base, source, config, model)
                    idsample = assign_to_samples(tv.idsurvey, tv.field, tv.is_photoz, mapping.samples, config)
                    tv.cutbits[idsample < 0] |= 1 << CUTBIT_IDSAMPLE
                    if config.write_ccprior:
                        write_passcuts_list(output_path(config, "_CCPRIOR.FITRES"), "CCPRIOR", tv, idsample)
                    prior, ncell = build_ccprior(tv, idsample, mapping, model, config)
                    result = (prior, ncell, sample_counts(idsample, tv.cutmask, mapping.samples),
                              tv.version_photometry)
                for key in keys:
                    priors[key] = result
    structures = {}
    for key, result in priors.items():
        if result is None:
            continue
        prior = result[0]
        structures[key] = (standardised_model(cells[key][0].model_lcfit), tuple(prior.cc_lcpar),
                           prior.cc_fitparbias_nodes.shape[1:],
                           tuple(len(axis) for axis in prior.coeff_avg))
    for keys in group_equal(structures):
        width = max(len(priors[key][0].cc_z) for key in keys)
        ncell = max(priors[key][1] for key in keys)
        template = next((priors[key][0] for key in keys if len(priors[key][0].cc_z)), None)
        padded = {}
        for key in keys:
            prior, _, counts, version = priors[key]
            if id(prior) not in padded:
                padded[id(prior)] = pad_ccprior(prior, width, template)
            priors[key] = padded[id(prior)], ncell, counts, version
    return priors


def prepare_data(cells, *, cosmology_widths=None):
    """Unmerged and merged numerical data, shared within one grid."""
    requests = {}
    for key, (config, tables, _) in cells.items():
        model = standardised_model(config.model_lcfit)
        requests[key] = (id(tables[EVENT_TYPE_DATA]), config.model_lcfit, _conversion_key(
            EVENT_TYPE_DATA, config, model, biascor=tables.get(EVENT_TYPE_BIASCOR),
            ccprior=tables.get(EVENT_TYPE_CCPRIOR)))
    result = {}
    for keys in group_equal(requests):
        config, tables, text = cells[keys[0]]
        source = tables[EVENT_TYPE_DATA]
        with preparation_step("read_data", keys):
            base = _convert_table(
                source, EVENT_TYPE_DATA, config, standardised_model(config.model_lcfit),
                biascor=tables.get(EVENT_TYPE_BIASCOR), ccprior=tables.get(EVENT_TYPE_CCPRIOR),
                cosmology_width=(cosmology_widths[standardised_model(config.model_lcfit)]
                                 if cosmology_widths is not None else None))
            base = readonly_variables(base)
        merges = {key: (cells[key][0].iflag_duplicate, cells[key][0].sigint) for key in keys}
        with preparation_step("duplicates", keys):
            for merge_keys in group_equal(merges):
                config = cells[merge_keys[0]][0]
                merged = replace(base, cutbits=np.zeros(base.size, dtype=np.int64), lcpar=dict(base.lcpar))
                if not merged.sim_truth:
                    duplicates = merge_duplicates(merged, config)
                    merged.cutbits[duplicates] |= 1 << CUTBIT_DUPL
                merged = readonly_variables(merged)
                for key in merge_keys:
                    result[key] = base, merged
    return result


def format_data(cells, data):
    """Format shared output columns only for the grid being fitted."""
    for keys in group_equal({key: id(pair[0]) for key, pair in data.items()}):
        config, tables, text = cells[keys[0]]
        source = tables[EVENT_TYPE_DATA]
        names = dict.fromkeys(name for key in keys for name in
                              output_varnames(source.varnames_by_file, cells[key][0]))
        columns = {name: _passthrough_column(text if text is not None else source, name)
                   for name in names}
        for values in columns.values():
            values.setflags(write=False)
        for key in keys:
            base, merged = data[key]
            base.passthrough = merged.passthrough = columns


def interpolate_corrections(cells, data, maps):
    """Full-table numerical corrections, before local exemptions and cuts."""
    requests = {key: (id(data[key][1]), id(maps[key]), config.min_per_cell_biascor)
                for key, (config, _, _) in cells.items() if maps[key] is not None}
    result = {key: None for key in cells}
    for keys in group_equal(requests):
        key = keys[0]
        config = cells[key][0]
        model = standardised_model(config.model_lcfit)
        with preparation_step("biascor_lookup", keys):
            tv = replace(data[key][1])
            tv.idsample, _ = partition_idsamples(tv.idsurvey, tv.field, tv.zhd, tv.is_photoz, config)
            mapping = maps[key]
            if isinstance(mapping, tuple):
                grids, summary = mapping
                mubias, valid = biascor_1d_data(grids, summary.samples, tv, tv.idsample)
                values = dict(mubias=mubias, valid=valid)
            else:
                fval, nevt, ferr, scale, added, valid = precompute_biascor(
                    tv, mapping, model, config.min_per_cell_biascor)
                m0 = mapping.maps[0]
                values = dict(fitparbias_nodes=fval, fitparbias_nevt=nevt,
                              fitparbias_err=ferr, mucovscale_nodes=scale, mucovadd_nodes=added,
                              valid=valid, bias_names=bias_parameters(model, config),
                              coeff_avg=tuple(jnp.asarray(m0.bininfo[j].avg) for j in range(m0.n_coeff)),
                              coeff_binsize=tuple(float(m0.bininfo[j].binsize) for j in range(m0.n_coeff)))
        for key in keys:
            result[key] = values
    return result


def select_data(config, sources, data, maps):
    """Local cuts on original values, followed by shared merged values."""
    base, merged = data
    tv = select_tablevar(base, sources[EVENT_TYPE_DATA], config,
                         standardised_model(config.model_lcfit))
    tv.lcpar, tv.covmat_fit = dict(merged.lcpar), merged.covmat_fit
    if not config.cutwin_disable:
        tv.cutbits |= merged.cutbits
    if maps is not None:
        tv.idsample, _ = partition_idsamples(tv.idsurvey, tv.field, tv.zhd, tv.is_photoz, config)
    return tv

def _force_dust_coefficients(biascor_tv, config, model) -> bool:
    """set_DUST_FLAG_biasCor (SALT2mu.c:11930): a biasCor sample whose first
    20 rows hold a Ia with SIM_AV > 0 comes from a dust model, whose per-event
    SIM_alpha and SIM_beta are the intrinsic draws and not a fitted
    standardisation, so every event's are overwritten with the input's p1 and
    p2 (INPUTS.parval, 0 where not given) before the maps are built. Returns
    whether the sample is one."""
    truth = biascor_tv.sim_truth
    if "SIM_AV" not in truth:
        return False
    first = slice(0, 20)
    is_ia = truth["SIM_TEMPLATE_INDEX"][first] == 0
    if not is_ia.any():
        raise SNANAError("Found no SNIa in first 20 BiasCor events; unable "
                         "to check for BS21 dust model.",
                         function_name="_force_dust_coefficients")
    if not (truth["SIM_AV"][first][is_ia] > 0.0).any():
        return False
    forced = []
    for name, sim_col, _ in model.biascor_coeffs:
        if name in model.dust_forced_coeffs:
            value = parval(config, name)
            truth[sim_col] = np.full(biascor_tv.size, value, dtype=np.float32)
            forced.append(f"{sim_col} = {value:g} (p{model.nuisance_spec[name][3]})")
    slog.warn(logger, "BBC", "Detected BS21 dust model for biasCor: every "
              "event's " + ", ".join(forced))
    return True


def _report_biascor_duplicates(tv, model):
    """check_duplicates_biascor: biasCor duplicates match on redshift and map coordinates
    rather than CID (SALT2mu.c:4412). Diagnostic only: no event is cut."""
    names = tuple(model.biascor_binning)
    key = np.stack([tv.zhd] + [np.asarray(tv.lcpar[n]) for n in names], axis=1)
    _, counts = np.unique(key, axis=0, return_counts=True)
    ndupl = int((counts > 1).sum())
    slog.warn(logger, "BBC", f"check_duplicates_biascor: {ndupl} duplicate "
                             f"(z, {', '.join(names)}) set(s) in the biasCor sample")


def correction_config(config):
    """Resolved settings that affect correction construction."""
    # These are consumed after map construction, or only by data-table cuts.
    # All unlisted fields remain dependencies, including nuisance values and
    # flags: those also control coefficient grids and simulation residuals.
    defaults = BBCConfig()
    independent = (
        "datafile", "cid_select_file", "cid_reject_file", "cutwin_disable",
        "chi2max", "chi2max_fitwgt0", "chi2max_surveys", "min_per_zbin",
        "prescale_simdata", "prescale_simcc", "prescale_simia", "iflag_duplicate",
        "izbin_from_cid_file", "force_realdata", "nmax", "zphot_shift", "zwin_vpec_check",
        "fitflag_sigmb", "sigint_fix", "redchi2_tol", "sigint_step1",
        "dchi2red_dsigint", "sigint_mode", "sigint_floor", "sigint_ceiling",
        "ndump_nobiascor", "frac_warn_nobiascor", "surveylist_nobiascor",
        "write_yaml", "write_csv", "write_chi2grid", "cutmask_write",
        "append_varname_missing", "cat_only", "cat_prescale", "cat_file_out", "uave",
    )
    ignored = {name: getattr(defaults, name) for name in independent}
    if not (config.write_biascor or config.write_ccprior):
        ignored["prefix"] = defaults.prefix
    config = deepcopy(replace(config, **ignored,
                              double_precision=jax.config.x64_enabled))
    return config


def _biascor_1d_setup(biascor_tv, samples, config, model):
    """The 1D path's per-sample bias grids, and the maps the .YAML reports
    (the 1D path has no cell maps, but reports the same by-sample block)."""
    dust = _force_dust_coefficients(biascor_tv, config, model)
    grids, bc_idsample, bc_cutmask, n_inrange = biascor_1d_grids(
        biascor_tv, samples, config, model)
    return grids, BiasCorMaps(samples=samples,
                              maps=[], mucov_maps=[],
                              biascor_counts=sample_counts(bc_idsample, bc_cutmask, samples),
                              biascor_nevt_total=n_inrange, ndim=1,
                              # SIGINT_AVG on the 1D path is the input sigmB
                              # (prepare_biasCor_zinterp, SALT2mu.c:12041)
                              sigint_biascor=config.sigint,
                              version_photometry=biascor_tv.version_photometry,
                              sim_cosmology=biascor_tv.sim_cosmology,
                              dust_flag=dust)
