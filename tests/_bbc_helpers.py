"""Shared real-input setup for BBC tests."""

import numpy as np

from jax_salt2mu.binning import BinInfo
from jax_salt2mu.config import BBCConfig
from jax_salt2mu.preparation import (
    read_cells, prepare_biascor_maps, prepare_ccpriors, prepare_data, format_data,
    interpolate_corrections,
)
from jax_salt2mu.run import BBCCell, prepare_cells
from jax_salt2mu.tablevar import (
    TableVar, EVENT_TYPE_DATA, read_data_table, convert_table, passthrough_column,
    output_varnames, select_tablevar, merge_duplicates, CUTBIT_DUPL,
)
from jax_snlc_core.snana_files import read_fitres_multi


def prepare_cell(config: BBCConfig, tables=None) -> BBCCell:
    """Prepare a single cell through the same producers used by complete grids."""
    cells = read_cells({None: (config, tables)})
    maps = prepare_biascor_maps(cells)
    priors = prepare_ccpriors(cells, maps)
    data = prepare_data(cells)
    format_data(cells, data)
    biascor = interpolate_corrections(cells, data, maps)
    return prepare_cells(cells, data, maps, biascor, priors)[None]


def read_tablevar(paths, event_type, config: BBCConfig, model) -> TableVar:
    """The table variables of the FITRES files at ``paths``."""
    if event_type == EVENT_TYPE_DATA:
        table, text = read_data_table(paths)
    else:
        table, text = read_fitres_multi(list(paths)), None
    return tablevar_from(table, event_type, config, model, text_table=text)


def uniform_bininfo(lo: float, hi: float, nbin: int) -> BinInfo:
    """Construct evenly spaced bins for numerical test inputs."""
    edges = np.linspace(lo, hi, nbin + 1)
    los, his = edges[:-1], edges[1:]
    return BinInfo(nbin=nbin, lo=los, hi=his, avg=0.5 * (los + his),
                   binsize=(hi - lo) / nbin)


def tablevar_from(table, event_type, config: BBCConfig, model,
                  text_table=None) -> TableVar:
    """The table variables of an already-read FITRES table.

    ``text_table`` contains the original text tokens for the writer.
    Without it, the table's own
    values supply the tokens, as in an in-memory handoff.
    """
    tv = convert_table(table, event_type, config, model)
    if event_type == EVENT_TYPE_DATA:
        text = text_table if text_table is not None else table
        for name in output_varnames(table.varnames_by_file, config):
            tv.passthrough[name] = passthrough_column(text, name)
    tv = select_tablevar(tv, table, config, model)
    # SNANA only checks duplicates for real data (SALT2mu.c:2270, !IS_SIM).
    if event_type == EVENT_TYPE_DATA and not tv.sim_truth:
        duplicates = merge_duplicates(tv, config)
        if not config.cutwin_disable:
            tv.cutbits[duplicates] |= 1 << CUTBIT_DUPL
    return tv
