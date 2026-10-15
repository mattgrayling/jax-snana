"""Distance fits from Python settings and fitted light-curve tables."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from jax_snlc_core.io.fitres_reader import FitresTable
from jax_snlc_core.utils import SNANAError
from jax_snlc_core.utils.input_settings import merge_settings

from .config import BBCConfig
from .config_parser import parse_bbc_input
from .fitter import BBCResult
from .inputs import build_config
from .output import m0dif_columns, output_path
from .results import BBCOutput
from .run import run_bbc_fit, write_outputs
from .tablevar import EVENT_TYPE_DATA, EVENT_TYPE_BIASCOR, EVENT_TYPE_CCPRIOR, TableVar
from .zbins import ZBinGrid


@dataclass
class DistanceFitResult:
    """A BBC fit, its distances, redshift bins and accepted-event mask.

    ``fit`` contains fitted nuisance parameters and their uncertainties.
    ``output`` contains per-event distances and per-bin distance offsets.
    ``accepted_mask`` selects rows of ``data`` used for those per-event arrays.
    """

    fit: BBCResult
    output: BBCOutput
    bins: ZBinGrid
    data: TableVar
    accepted_mask: np.ndarray

    @property
    def hubble_diagram(self):
        """The binned M0DIF columns as a pandas DataFrame, ready for covariance construction."""
        return pd.DataFrame(m0dif_columns(self.bins, self.output))


def fit_distances(inputs=None, *, data=None, biascor=None, ccprior=None,
                  write=False, **settings):
    """Fit BBC distances using keyword settings, a mapping or an input file.

    ``data``, ``biascor`` and ``ccprior`` accept ``FitresTable`` objects or
    pandas DataFrames. Use ``FitresTable`` when header metadata is needed;
    a DataFrame supplies columns only. Files can instead be named with
    ``datafile``, ``simfile_biascor`` and ``simfile_ccprior`` settings.
    A table and a file for the same input are rejected.

    Keyword settings override ``inputs``.
    Returns a ``DistanceFitResult``. Set ``write=True`` and supply ``prefix``
    to write the configured outputs.
    """
    if isinstance(inputs, BBCConfig):
        config = build_config(settings, config=inputs)
    elif inputs is None or isinstance(inputs, Mapping):
        config = build_config(merge_settings(inputs or {}, settings))
    else:
        config = parse_bbc_input(inputs, settings=settings)
    tables = {}
    for event_type, table in ((EVENT_TYPE_DATA, data), (EVENT_TYPE_BIASCOR, biascor),
                              (EVENT_TYPE_CCPRIOR, ccprior)):
        if table is None:
            continue
        if isinstance(table, pd.DataFrame):
            table = FitresTable(df=table.copy(), varnames=list(table.columns), meta={},
                                varnames_by_file=[list(table.columns)])
        if not isinstance(table, FitresTable):
            raise SNANAError('BBC tables must be FitresTable objects or pandas DataFrames')
        tables[event_type] = table
    if not config.datafile and EVENT_TYPE_DATA not in tables:
        raise SNANAError('Supply data or the datafile setting for a distance fit')
    if config.cat_only:
        raise SNANAError('cat_only joins files; use run_bbc for that operation')
    path = output_path(config, '.FITRES') if write else None
    result = DistanceFitResult(*run_bbc_fit(config, tables=tables, write=write))
    if write:
        write_outputs(path, result.fit, result.output, result.bins,
                      result.data, result.accepted_mask)
    return result
