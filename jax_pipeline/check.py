"""Validate a Pippin analysis and describe its stages without executing them."""

import dataclasses
from collections.abc import Mapping
from pathlib import Path

from jax_snlc_core.utils.error_handling import SNANAError
from jax_wfit.config_parser import build as build_cosmology_config
from jax_wfit.config_parser import read_keys as read_cosmology_keys

from .classify import Classify
from .run import validate_graph
from .stages import BBC, Cosmology, Covariance, Data, Fit, Sim
from .systematics import dirname


def _paths(value):
    if isinstance(value, Path):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _paths(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _paths(item)


def check(path, data_dirs=()) -> dict:
    """Read inputs, validate stage configurations and return an execution plan.

    Does not initialise models, generate events, fit data or write outputs.
    Checks requiring generated tables are performed when the analysis runs.
    """
    from .pippin import read

    stages = validate_graph(read(path, data_dirs))
    plan = []
    for stage in stages:
        paths = [p for field in dataclasses.fields(stage)
                 for p in _paths(getattr(stage, field.name))]
        paths.extend(stage.cache_files())
        if isinstance(stage, Fit):
            paths.extend(stage.source_files())
        for source in paths:
            if not source.exists():
                raise SNANAError(f"input {source} does not exist", function_name=stage.name)
        entry = dict(name=stage.name, stage=type(stage).__name__,
                     inputs={role: used.name for role, used in stage.inputs().items()})
        if paths:
            entry["files"] = list(dict.fromkeys(str(p) for p in paths))
        if isinstance(stage, Sim):
            seeds = []
            for index in range(len(stage.input_files)):
                config = stage.configuration(index)
                seeds.append(config.run.random_seed)
            entry["simulation_seeds"] = seeds
        elif isinstance(stage, Fit) and stage.fitres is None:
            for cell in stage.grid_cells():
                stage.configuration(cell)
        elif isinstance(stage, BBC):
            entry["sync_events"] = stage.event_sync()
            probability_columns = {}
            for muopt in sorted({j for _, j in stage.grid_cells()}):
                config = stage.configuration(muopt)
                if config.varname_pia:
                    probability_columns[f"MUOPT{muopt:03d}"] = config.varname_pia
            if probability_columns:
                entry["probability_columns"] = probability_columns
            entry.update(realisation=stage.realisation, split_index=stage.split_index,
                         split_count=stage.split_count)
            if stage.require_probability_column:
                entry["requires_existing_probability_columns"] = True
        elif isinstance(stage, Covariance):
            config, _, _ = stage.configuration()
            entry.update(mode=config.mode,
                         covopts=["ALL", *(c.label for c in config.covopts)])
        elif isinstance(stage, Cosmology):
            keys = {} if stage.input_file is None else read_cosmology_keys(str(stage.input_file))
            if {"cospar_yaml", "outfile_cospar"} & (keys.keys() | stage.overrides.keys()):
                raise SNANAError("cosmology stage names its own output files", function_name=stage.name)
            build_cosmology_config({**keys, **stage.overrides})
            entry["covopts"] = list(stage.covopts)
        if isinstance(stage, Data):
            entry["reuse"] = str(stage.directory)
        elif isinstance(stage, Fit) and stage.fitres:
            entry["reuse"] = str(stage.fitres)
        elif isinstance(stage, Classify) and stage.predictions:
            entry["reuse"] = str(stage.predictions)
        if isinstance(stage, (Fit, Classify, BBC)):
            fitopts, muopts = stage.systematics()
            entry["combinations"] = [dict(directory=dirname((i, j)),
                fitopt="DEFAULT" if i == 0 else fitopts[i - 1],
                muopt="DEFAULT" if j == 0 else muopts[j - 1]) for i, j in stage.grid_cells()]
        plan.append(entry)
    return {"configuration": str(path), "stages": plan}
