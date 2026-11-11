"""Pippin analysis configs as graphs of stages.

A Pippin YAML names tasks by section and joins them by substring masks and
exact task names. ``read`` turns one into the stage objects a script would
build by hand, so nothing runs differently for having come from YAML.

The AGGREGATION, MERGE and ANALYSE sections are skipped, as are COSMOFIT
entries other than WFIT and sections Pippin does not read, such as one
holding YAML anchors. A classification joins its own column, and a BIASCOR
task reading several classifiers' columns of one fit reads their merge, so
no MERGE task is needed. A task that depends on something skipped or
unsupported aborts naming it, as does a LEGACY AGGREGATION RECALIBRATION.
Settings jax does not need, such as those of Pippin's batch layer, are
ignored with a log line each.
"""

import argparse
import dataclasses
import functools
import logging
import math
import os
import shlex
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from jax_create_cov.config import parse_covopt
from jax_snlc_core import logging as log
from jax_snlc_core.logging import SNANAError, aborting
from jax_snlc_core.snana_files import fitopt_muopt_name
from jax_snlc_core.utils import configure_precision, group_equal
from jax_snlc_fit.config_parser import group_override_tokens, known_keys
from jax_snlc_fit.loader import version_survey
from jax_snlc_sim.simlib import survey_from_simlib_file
from jax_snlc_sim.simulation import FORMAT_MASK_CIDRAN
from jax_wfit.config import (build as build_cosmology_config, build_parser, cli_keys,
                             read_keys as read_cosmology_keys)

from .classify import CLASSIFIER_REGISTRY, Classify, Merge, Train
from .run import run, validate_graph
from .stages import BBC, Cosmology, Covariance, Data, Fit, FitVersion, Sim, Stage, sync_setting

logger = logging.getLogger(__name__)

#: Pippin's default distance-fit input, relative to its data directories
_DEFAULT_BBC = "surveys/des/bbc/bbc_5yr.input"
#: the keys of a CLASSIFICATION task read here
_CLASSIFICATION_KEYS = ("CLASSIFIER", "MODE", "OPTS", "MASK", "MASK_SIM", "EXTERNAL_DIRS")
#: Pippin's DATAPREP types when TYPES is not given (dataprep.py:81-104)
_DATAPREP_TYPES = {"IA": [1], "NONIA": [2, 20, 21, 22, 29, 30, 31, 32, 33, 39, 40, 41, 42, 43, 80, 81]}
#: the model names Pippin takes for type Ia simulations (snana_sim.py:28-36),
#: matched regardless of case, as its mixed-case list intends
_SNIA_MODELS = ("SALT", "BAYESN", "SNEMO", "BYOSED", "MLCS", "SNOOPY")
#: the queue prefix of Pippin's own cfg.yml
_DEFAULT_QUEUE_PREFIX = "PIP"
#: the RANSEED_REPEAT and FORMAT_MASK of Pippin's own combine.input, used when a SIM task gives none
_DEFAULT_RANSEED_REPEAT = "1 12345"
_DEFAULT_FORMAT_MASK = 48
#: settings ignored here: those of Pippin's batch layer and installation;
#: submit_batch's APPEND_TABLE_VARLIST; MASK_FIT, which Pippin's classifiers
#: never apply; BIASCOR's MAKE_ALL_HUBBLE and CONSISTENT_SAMPLE, which
#: only control Pippin's plots and an iteration it never runs
#: (biascor.py:84-86, 144, 315); and a SIM task's GENVERSION, which Pippin
#: replaces with its own (snana_sim.py:90, 647-648)
_IGNORED_KEYS = {"BATCH_INFO", "BATCH_FILE", "BATCH_REPLACE", "BATCH_MEM", "BATCH_WALLTIME",
                 "NUM_JOBS", "GPU", "CLEAN", "INI_DIR", "COSMOMC", "CFG_PATH",
                 "APPEND_TABLE_VARLIST", "MASK_FIT", "MAKE_ALL_HUBBLE", "CONSISTENT_SAMPLE",
                 "GENVERSION"}
#: BIASCOR OPTS that apply to the whole task, read here rather than by SALT2mu
_TASK_ONLY_OPTS = {"FLAG_USE_SAME_EVENTS", "OPT_SNCID_LIST", "FITOPTxMUOPT",
                   "REPLACE_SIMFILE_BIASCOR", "REPLACE_SIMFILE_CCPRIOR", "NSPLITRAN"}


def _ignoring(section, name, values) -> dict:
    """A block of settings, which YAML gives as None when it is empty,
    without the settings in ``_IGNORED_KEYS``, each of which is logged."""
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise SNANAError(f"{section} {name}: expected a mapping of settings", function_name="pippin")
    kept = {}
    for key, value in values.items():
        if key in _IGNORED_KEYS:
            logger.info(f"{section} {name}: {key} is ignored")
        else:
            kept[key] = value
    return kept


def _settings(section, name, values, supported) -> dict:
    """A block of settings without its ignored ones, aborting on any setting
    that is not ``supported``."""
    kept = _ignoring(section, name, values)
    unknown = sorted(str(key) for key in set(kept) - set(supported))
    if unknown:
        raise SNANAError(f"{section} {name}: unsupported setting(s): {', '.join(unknown)}",
                         function_name="pippin")
    return kept


def _data_location(text: str, data_dirs: Sequence[Path]) -> Path | None:
    """Pippin's path rule (config.py, get_data_loc): ``$VAR`` expanded, an
    absolute path as given, a relative one searched for under the data
    directories in order; None when no such path exists."""
    if "$" in text or text.startswith("/"):
        path = Path(os.path.expandvars(text))
        return path if path.exists() else None
    for directory in data_dirs:
        path = Path(directory) / text
        if path.exists():
            return path
    return None


def _locate(text: str, data_dirs: Sequence[Path], section: str, name: str) -> Path:
    """A path the config names, found by Pippin's path rule
    (:func:`_data_location`), aborting when it does not exist."""
    path = _data_location(text, data_dirs)
    if path is None:
        raise SNANAError(f"{section} {name}: {text} ({os.path.expandvars(text)}) does not exist; "
                         f"relative paths are searched for under {list(data_dirs)}",
                         function_name="pippin")
    return path


def _included_genmodel(lines: list[str]) -> str | None:
    """The first GENMODEL in the files a SIM input file's lines include, as
    Pippin finds it (snana_sim.py:573-637), or None if they set none."""
    for line in lines:
        words = line.split()
        if len(words) > 1 and words[0] in ("INPUT_INCLUDE_FILE:", "INPUT_FILE_INCLUDE:"):
            for included in Path(os.path.expandvars(words[1])).read_text().splitlines():
                included_words = included.split("#")[0].split()
                if len(included_words) > 1 and included_words[0] == "GENMODEL:":
                    return " ".join(included_words[1:])
    return None


def _component_type(base: Path, keys: Mapping, group: str, component: str) -> tuple[int, str]:
    """A SIM component's GENTYPE, labelled Ia if its GENMODEL is a type Ia
    model and II otherwise, found as Pippin finds them (snana_sim.py:122-180):
    in the BASE file's own lines, then the component's keys, and for GENMODEL
    last, the files the BASE file includes."""
    lines = base.read_text().splitlines()
    gentype = genmodel = None
    for line in lines:
        upper = line.upper()
        if upper.strip().startswith("GENTYPE:"):
            gentype = upper.split(":")[1].strip()
        if upper.strip().startswith("GENMODEL:"):
            genmodel = upper.split(":")[1].strip()
    if not gentype:
        gentype = keys.get("GENTYPE")
    if not genmodel:
        genmodel = keys.get("GENMODEL")
    if gentype is None:
        raise SNANAError(f"SIM {group}: component {component} sets no GENTYPE, in {base} or its "
                         f"keys", function_name="pippin")
    if genmodel is None:
        genmodel = _included_genmodel(lines)
    if genmodel is None:
        raise SNANAError(f"SIM {group}: component {component} sets no GENMODEL, in {base}, its "
                         f"keys or the files it includes", function_name="pippin")
    is_ia = any(model in genmodel.upper() for model in _SNIA_MODELS)
    return int(gentype), "Ia" if is_ia else "II"


def _external_dirs(entries) -> dict[str, str]:
    """A task's EXTERNAL_DIRS by folder name, the first entry winning where
    two share one, as Pippin matches them (task.py:46-74)."""
    if not isinstance(entries, list):
        entries = [entries]
    by_name = {}
    for entry in entries:
        by_name.setdefault(os.path.basename(entry), entry)
    return by_name


def _base_config(path: Path) -> dict:
    """The CONFIG block of an LCFIT BASE file's YAML header, the lines before
    the first one containing #END_YAML, as Pippin reads it (base.py:18-30);
    empty when the file has no header."""
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        if "#END_YAML" in line:
            header = yaml.safe_load("\n".join(lines[:index]))
            return dict(header["CONFIG"])
    return {}


def _fitopts_file(path: Path) -> tuple[int, list[tuple[str, str, str | None]]]:
    """A FITOPTS file's FLAG_USE_SAME_EVENTS, 0 when it sets none, and its
    entries as ``(block, label, command)`` in file order. Each entry is a
    systematic's scale followed by the fit's settings, the command, which is
    None when the entry gives only a scale, as a SYS_SCALE file may."""
    blocks = yaml.safe_load(path.read_text())
    same_events = blocks.pop("FLAG_USE_SAME_EVENTS", 0)
    entries = []
    for block, labelled in blocks.items():
        for label, scale_command in labelled.items():
            words = str(scale_command).split(maxsplit=1)
            command = None
            if len(words) == 2:
                command = words[1]
            entries.append((block, label, command))
    return same_events, entries


def _survey(stage: Stage) -> str:
    """The survey a source stage's data belongs to, which chooses the FITOPTS
    block: for a simulation, its SIMLIB_SURVEY or else the SURVEY in its
    SIMLIB header, the survey the simulation records once run, where Pippin
    reads it (snana_sim.py:540-545); for data, the version's survey
    metadata."""
    if isinstance(stage, Sim):
        survey = stage.configuration().survey
        if not survey.simlib_file:
            raise SNANAError(f"{stage.input_files[0]} names no SIMLIB_FILE", function_name="pippin")
        return survey_from_simlib_file(survey.simlib_file, survey.survey_override)[0]
    return version_survey(stage.directory)


class _Graph:
    """The stages a Pippin config has been translated into so far, kept by
    Pippin task, as each section is read in Pippin's order."""

    def __init__(self, config: Mapping, prefix: str, data_dirs: Sequence[Path]):
        self.config = config
        #: the prefix of Pippin's simulation folders for the config
        self.prefix = prefix
        self.data_dirs = list(data_dirs)
        #: each DATAPREP or SIM task's data or simulation versions
        self.sources: dict[str, list[Stage]] = {}
        #: each DATAPREP or SIM task's BLIND
        self.blind: dict[str, bool] = {}
        #: each DATAPREP or SIM task's event types, labelled Ia or II as Pippin
        #: labels them, or None for a completed simulation that recorded none
        self.types: dict[str, dict[int, str] | None] = {}
        self.fit_stages: list[Fit] = []
        #: the fits of each version, named <LCFIT task>_<version>
        self.fits: dict[str, Fit | FitVersion] = {}
        #: the names of each source task's fits by each LCFIT task, one per version
        self.fitted: dict[str, dict[str, list[str]]] = {}
        #: each CLASSIFICATION training task's trainings, one per source task
        self.trained: dict[str, list[Train]] = {}
        #: every classification stage
        self.classifications: list[Classify] = []
        #: the classification BIASCOR reads for each (CLASSIFICATION task, fit)
        self.classified: dict[tuple[str, str], Classify] = {}
        #: the merges of several classifications of one fit, by name
        self.merges: dict[str, Merge] = {}
        #: the probability column BIASCOR reads for each predicting
        #: CLASSIFICATION task, as Pippin records only those (aggregator.py:449-456)
        self.columns: dict[str, str] = {}
        #: each CLASSIFICATION task with no stage here, and its classifier
        self.unsupported: dict[str, str] = {}
        #: each BIASCOR and CREATE_COV task's stages
        self.bbcs: dict[str, list[BBC]] = {}
        self.covariances: dict[str, list[Covariance]] = {}
        self.cosmologies: list[Cosmology] = []
        #: each LCFIT task's FITOPTS file, None when it has only inline FITOPTS
        self.fitopts_files: dict[str, Path | None] = {}
        #: each BIASCOR task's FITOPTS files, of the LCFIT tasks fitting its DATA
        self.data_fitopts_files: dict[str, list[Path]] = {}
        #: each BIASCOR task's MUOPT SCALEs
        self.muopt_scales: dict[str, dict[str, float]] = {}

    def stages(self) -> list[Stage]:
        stages = []
        for versions in self.sources.values():
            stages.extend(versions)
        stages.extend(self.fit_stages)
        for trainings in self.trained.values():
            stages.extend(trainings)
        stages.extend(self.classifications)
        stages.extend(self.merges.values())
        for bbcs in self.bbcs.values():
            stages.extend(bbcs)
        for covariances in self.covariances.values():
            stages.extend(covariances)
        stages.extend(self.cosmologies)
        return stages

    def _section(self, key: str) -> dict:
        value = self.config.get(key)
        if value is not None and not isinstance(value, Mapping):
            raise SNANAError(f"{key} config: expected a mapping of named tasks", function_name="pippin")
        return value or {}

    def read_dataprep(self):
        for name, task in self._section("DATAPREP").items():
            task = _settings("DATAPREP", name, task, {"OPTS"})
            opts = _settings("DATAPREP OPTS", name, task.get("OPTS"), {"RAW_DIR", "TYPES", "BLIND", "SIM"})
            if "RAW_DIR" not in opts:
                raise SNANAError(f"DATAPREP {name}: OPTS requires RAW_DIR", function_name="pippin")
            if "SIM" in opts:
                logger.info(f"DATAPREP {name}: SIM is metadata; source tables are read unchanged")
            listed = opts.get("TYPES")
            if listed is None:
                listed = _DATAPREP_TYPES
            # the Ia types, then the others (dataprep.py:116-121)
            types = {}
            for sntype in listed["IA"]:
                types[int(sntype)] = "Ia"
            for sntype in listed["NONIA"]:
                types[int(sntype)] = "II"
            directory = _locate(opts["RAW_DIR"], self.data_dirs, "DATAPREP", name)
            self.sources[name] = [Data(name, directory)]
            self.types[name] = types
            self.blind[name] = opts.get("BLIND", True)
            self.fitted[name] = {}

    def read_sim(self):
        for group, task in self._section("SIM").items():
            if not isinstance(task, Mapping):
                raise SNANAError(f"SIM {group}: expected components and GLOBAL settings",
                                 function_name="pippin")
            task = _ignoring("SIM", group, task)
            opts = _settings("SIM OPTS", group, task.get("OPTS"), {"BLIND"})
            if "EXTERNAL" in task:
                self.sources[group], self.types[group], self.blind[group] = (
                    self._external_simulation(group, task))
            else:
                self.sources[group], self.types[group] = self._simulation(group, task)
                self.blind[group] = opts.get("BLIND", False)
            self.fitted[group] = {}

    def _external_simulation(self, group, task) -> tuple[list[Data], dict[int, str] | None, bool]:
        """The versions an earlier Pippin SIM task recorded in its config.yml
        (task.py:111-122), in its order, each read as it is; the types it
        recorded, or None if it recorded none; and its BLIND, as Pippin
        reads it from there (task.py:119-121, snana_sim.py:110, 202)."""
        task = _settings("SIM", group, task, {"EXTERNAL", "OPTS"})
        recorded = _locate(task["EXTERNAL"], self.data_dirs, "SIM", group)
        if recorded.is_dir():
            recorded = recorded / "config.yml"
        if not recorded.is_file():
            raise SNANAError(f"SIM {group}: {recorded} does not exist", function_name="pippin")
        recorded_config = yaml.safe_load(recorded.read_text())
        output = recorded_config.get("OUTPUT", {})
        folders = output.get("sim_folders")
        if not isinstance(folders, list) or not folders:
            raise SNANAError(f"SIM {group}: {recorded} needs OUTPUT.sim_folders", function_name="pippin")
        if "blind" not in output:
            raise SNANAError(f"SIM {group}: {recorded} needs OUTPUT.blind", function_name="pippin")
        if len(folders) == 1:
            versions = [Data(group, Path(folders[0]))]
        else:
            versions = [Data(f"{group}-{index:04d}", Path(folder))
                        for index, folder in enumerate(folders, start=1)]
        return versions, output.get("types"), output["blind"]

    def _simulation(self, group, task) -> tuple[list[Sim], dict[int, str]]:
        """The versions a SIM task simulates: one per RANSEED_CHANGE job, or
        one made of every RANSEED_REPEAT job. Each job simulates every
        component, with the GLOBAL settings on top of the component's own.
        Also each component's types, labelled Ia or II by its model, as
        Pippin labels them (snana_sim.py:110-190)."""
        components = {}
        for name, keys in task.items():
            if name not in ("GLOBAL", "OPTS"):
                components[name] = keys
        if not components:
            raise SNANAError(f"SIM {group}: no components", function_name="pippin")
        for name, keys in components.items():
            if not isinstance(keys, Mapping) or "BASE" not in keys:
                raise SNANAError(f"SIM {group}: component {name} needs a BASE input",
                                 function_name="pippin")
        global_keys = _ignoring("SIM GLOBAL", group, task.get("GLOBAL"))
        # of RANSEED_CHANGE and RANSEED_REPEAT, Pippin passes SNANA only the later (snana_sim.py:348-351)
        seeding = [key for key in global_keys if key in ("RANSEED_CHANGE", "RANSEED_REPEAT")]
        changing = False
        ranseed = _DEFAULT_RANSEED_REPEAT
        if seeding:
            changing = seeding[-1] == "RANSEED_CHANGE"
            ranseed = global_keys[seeding[-1]]
        for key in seeding:
            del global_keys[key]
        words = str(ranseed).split()
        if len(words) != 2 or not words[0].isdigit() or not words[1].isdigit():
            raise SNANAError(f"SIM {group}: RANSEED_CHANGE/REPEAT needs 'count seed'",
                             function_name="pippin")
        jobs = int(words[0])
        seed = int(words[1])
        if jobs < 1 or seed < 1:
            raise SNANAError(f"SIM {group}: RANSEED_CHANGE/REPEAT count and seed must be positive",
                             function_name="pippin")
        format_mask = global_keys.pop("FORMAT_MASK", _DEFAULT_FORMAT_MASK)
        # SNANA's batch layer reads NGEN_UNIT, not the simulator
        ngen_unit = None
        if "NGEN_UNIT" in global_keys:
            ngen_unit = float(global_keys.pop("NGEN_UNIT"))
        # and RESET_CIDOFF, 2 meaning 1 when each job is a version (submit_prog_sim.py:976-977)
        reset_cidoff = int(global_keys.pop("RESET_CIDOFF", 0))
        if changing and reset_cidoff == 2:
            reset_cidoff = 1
        # SNANA's batch layer gives each job its own seed unless CIDs are
        # random and the jobs make one version (submit_prog_sim.py:524-533)
        if not changing and format_mask & FORMAT_MASK_CIDRAN:
            seeds = [seed] * jobs
        else:
            seeds = [seed + 10000 * job + job * job + 13 for job in range(jobs)]
        if changing:
            versions = {f"{group}-{job + 1:04d}": [job] for job in range(jobs)}
        else:
            versions = {group: list(range(jobs))}
        ia_inputs, other_inputs = [], []
        types = {}
        for component, keys in components.items():
            base = _locate(keys["BASE"], self.data_dirs, "SIM", group)
            # the GLOBAL settings over the component's, a list value one SNANA
            # token of space-separated words
            settings = dict(keys)
            settings.update(global_keys)
            del settings["BASE"]
            tokens = {}
            for key, value in settings.items():
                if isinstance(value, list):
                    tokens[key] = " ".join(str(word) for word in value)
                else:
                    tokens[key] = str(value)
            # SNANA's batch layer gives each job its own systematics seed, the
            # same for every component; a GLOBAL one is passed unchanged and
            # wins (submit_prog_sim.py:1698-1702, 1744-1769)
            systematics_seed = None
            if "RANSYSTPAR_RANSEED_SYST" in keys and "RANSYSTPAR_RANSEED_SYST" not in global_keys:
                systematics_seed = int(keys["RANSYSTPAR_RANSEED_SYST"])
            gentype, label = _component_type(base, keys, group, component)
            # and its photometric type, 100 higher (snana_sim.py:169)
            types[gentype] = label
            types[gentype + 100] = label
            if label == "Ia":
                ia_inputs.append((base, tokens, systematics_seed))
            else:
                other_inputs.append((base, tokens, systematics_seed))
        # SNANA simulates the Ia inputs first (submit_prog_sim.py:1139-1140)
        inputs = ia_inputs + other_inputs
        simulations = []
        for version, version_jobs in versions.items():
            files, overrides, positions = [], [], []
            for base, tokens, systematics_seed in inputs:
                for job in version_jobs:
                    files.append(base)
                    positions.append((job, jobs))
                    # a RANSEED or FORMAT_MASK of the task's own wins
                    override = {"RANSEED": seeds[job], "FORMAT_MASK": format_mask}
                    override.update(tokens)
                    if systematics_seed is not None:
                        override["RANSYSTPAR_RANSEED_SYST"] = systematics_seed * 7 + 41 * (job + 1)
                    overrides.append(override)
            simulations.append(Sim(version, files, overrides=overrides, ngen_unit=ngen_unit,
                                   jobs=positions, reset_cidoff=reset_cidoff))
        return simulations, dict(sorted(types.items()))

    def read_lcfit(self):
        for name, task in self._section("LCFIT").items():
            task = _settings("LCFIT", name, task, {"BASE", "MASK", "OPTS", "EXTERNAL_DIRS", "FITOPTS",
                                                   "SNLCINP", "FITINP"})
            # LCFIT OPTS hold settings for Pippin's batch layer, and the
            # same-events keys submit_batch reads (submit_prog_lcfit.py:765-777)
            opts = _settings("LCFIT OPTS", name, task.get("OPTS"),
                             {"FLAG_USE_SAME_EVENTS", "OPT_SNCID_LIST"})
            if "BASE" not in task:
                raise SNANAError(f"LCFIT {name}: BASE is required", function_name="pippin")
            base = _locate(task["BASE"], self.data_dirs, "LCFIT", name)
            # Pippin writes a list as one line per item, and the namelist keeps
            # the last; a null removes the key from the BASE file (base.py:117-120)
            overrides = {}
            for section in ("SNLCINP", "FITINP"):
                section_keys = task.get(section)
                if section_keys is None:
                    continue
                for key, value in section_keys.items():
                    if value is None:
                        raise SNANAError(f"LCFIT {name}: {section} {key} is null, which Pippin uses to "
                                         f"remove the key from the BASE file; remove it there instead",
                                         function_name="pippin")
                    if isinstance(value, list):
                        value = value[-1]
                    overrides[key] = str(value)
            # Pippin's mask (task.py:411-431): a list selects the tasks any of
            # its patterns is in, and each pattern must select one; a single
            # pattern of * or none selects every task
            mask = task.get("MASK")
            if mask == []:
                raise SNANAError(f"LCFIT {name}: MASK [] selects no SIM or DATAPREP task",
                                 function_name="pippin")
            if isinstance(mask, list):
                patterns = mask
            elif mask in (None, "*"):
                patterns = [""]
            else:
                patterns = [mask]
            for pattern in patterns:
                if not any(pattern in source_task for source_task in self.sources):
                    raise SNANAError(f"LCFIT {name}: mask {pattern!r} matches no SIM or DATAPREP task",
                                     function_name="pippin")
            source_tasks = [source_task for source_task in self.sources
                            if any(pattern in source_task for pattern in patterns)]
            external = self._external_fits(name, task.get("EXTERNAL_DIRS", []), source_tasks)
            # submit_batch reads FITOPT and the same-events keys from the BASE
            # file's CONFIG block, which Pippin updates with the task's FITOPTS,
            # if any, and OPTS (snana_fit.py:197-198, 296-298)
            batch_config = _base_config(base)
            batch_config.update(opts)
            asked = task.get("FITOPTS", [])
            if not asked:
                asked = batch_config.get("FITOPT", [])
            entries, same_events, loaded = self._fitopt_entries(name, asked)
            self.fitopts_files[name] = loaded
            # submit_batch takes OPT_SNCID_LIST when both keys are given, and
            # uses it when positive; Pippin writes a FITOPTS file's positive
            # flag over both (snana_fit.py:190; submit_prog_lcfit.py:765-777)
            flag = 0
            for key in ("FLAG_USE_SAME_EVENTS", "OPT_SNCID_LIST"):
                if key in batch_config:
                    flag = batch_config[key]
            if same_events > 0:
                flag = same_events
            if flag > 0:
                overrides["OPT_SNCID_LIST"] = str(flag)
            fitted = {}
            for source_task in source_tasks:
                for source in self.sources[source_task]:
                    if source.name not in external:
                        fitted[source.name] = source
            fitopts = {}
            for version, source in fitted.items():
                fitopts[version] = self._fitopts(name, entries, source)
            # one Fit per FITOPT set, over every version it fits, so they
            # share its compiled shapes
            groups = group_equal(fitopts)
            fit_of = {}
            for i, group in enumerate(groups):
                suffix = f"_{i}" if len(groups) > 1 else ""
                fit = Fit(f"LCFIT_{name}{suffix}", base,
                          datasets=[fitted[version] for version in group],
                          overrides=dict(overrides),
                          fitopts=fitopts[group[0]])
                self.fit_stages.append(fit)
                for version in group:
                    fit_of[version] = fit
            # Pippin's LCFIT task for a source task fits every version of it
            for source_task in source_tasks:
                fit_names = []
                for source in self.sources[source_task]:
                    fit_name = f"{name}_{source.name}"
                    if source.name in external:
                        self.fits[fit_name] = Fit(fit_name, fitres=external[source.name])
                        self.fit_stages.append(self.fits[fit_name])
                    else:
                        self.fits[fit_name] = fit_of[source.name].version(source)
                    fit_names.append(fit_name)
                self.fitted[source_task][name] = fit_names

    def _external_fits(self, name, entries, source_tasks) -> dict[str, Path]:
        """The saved fits of an earlier Pippin run, by version. An
        EXTERNAL_DIRS entry is the folder of Pippin's LCFIT task for a source
        task, matched by its name (task.py:46-74), and holds each version's
        fits under output/, named after its simulation folder
        (snana_fit.py:61-66): a data directory's own name, or
        <PREFIX>_<CONFIG>_<SIM> for a simulation (manager.py:64)."""
        saved = _external_dirs(entries)
        fits = {}
        for source_task in source_tasks:
            task_name = f"{name}_{source_task}"
            if task_name not in saved:
                continue
            directory = _locate(saved[task_name], self.data_dirs, "LCFIT", name)
            for source in self.sources[source_task]:
                folder = (source.directory.name if isinstance(source, Data)
                          else f"{self.prefix}_{source.name}")
                fits[source.name] = directory / "output" / folder
        return fits

    def _fitopt_entries(self, name, asked) -> tuple[list[tuple[str | None, str, str | None]], int, Path | None]:
        """A task's FITOPTS as ``(block, label, command)``, block None for an
        inline ``/label/ KEY value`` entry and command None for a file entry
        giving only a scale; the FLAG_USE_SAME_EVENTS of its file, 0 when it
        sets none; and the file, None when it names none."""
        if not isinstance(asked, list):
            asked = [asked]
        entries = []
        same_events = 0
        loaded = None
        for item in asked:
            path = _data_location(item, self.data_dirs)
            if path is not None:
                if loaded is not None:
                    raise SNANAError(f"LCFIT {name}: FITOPTS names two files, {loaded} and {path}; "
                                     f"Pippin reads one", function_name="pippin")
                loaded = path
                same_events, from_file = _fitopts_file(path)
                entries.extend(from_file)
            elif item.strip().startswith("/"):
                parts = item.strip()[1:].split("/", 1)
                if len(parts) != 2:
                    raise SNANAError(f"LCFIT {name}: FITOPTS entry {item!r} needs /label/ KEY value",
                                     function_name="pippin")
                entries.append((None, parts[0], parts[1]))
            else:
                raise SNANAError(f"LCFIT {name}: FITOPTS entry {item!r} is neither a file nor "
                                 f"/label/ KEY value", function_name="pippin")
        return entries, same_events, loaded

    def _fitopts(self, name, entries, source) -> dict[str, dict]:
        """One source's FITOPTS as the fitter's keys: the inline entries and
        the GLOBAL and survey blocks, in file order, the only ones Pippin
        reads (snana_fit.py:166-178), each needing fit settings."""
        known = known_keys()
        fitopts = {}
        survey = None
        for block, label, command in entries:
            if block not in (None, "GLOBAL"):
                if survey is None:
                    survey = _survey(source)
                if block != survey:
                    continue
            if command is None:
                raise SNANAError(f"LCFIT {name}: FITOPTS entry {label} gives a scale but no fit "
                                 f"settings", function_name="pippin")
            # Pippin passes each entry as its own FITOPT (snana_fit.py:166-182),
            # but a FITOPT here is known by its label
            if label in fitopts:
                raise SNANAError(f"LCFIT {name}: FITOPTS gives the label {label} twice",
                                 function_name="pippin")
            keys = {}
            for key, words in group_override_tokens(shlex.split(command), known):
                keys[key] = " ".join(words)
            fitopts[label] = keys
        return fitopts

    def read_classification(self):
        # in the config's order, so a predicting task finds only the training
        # tasks before it, as in Pippin (classifier.py:459-467)
        for name, task in self._section("CLASSIFICATION").items():
            if not isinstance(task, Mapping) or "CLASSIFIER" not in task or "MODE" not in task:
                raise SNANAError(f"CLASSIFICATION {name}: task requires CLASSIFIER and MODE",
                                 function_name="pippin")
            if task["CLASSIFIER"] not in CLASSIFIER_REGISTRY:
                self.unsupported[name] = task["CLASSIFIER"]
                logger.info(f"CLASSIFICATION {name}: {task['CLASSIFIER']} has no stage here")
                continue
            task = _settings("CLASSIFICATION", name, task, _CLASSIFICATION_KEYS)
            # Pippin lowercases MODE (classifier.py:235)
            mode = task["MODE"].lower()
            if mode not in ("train", "predict"):
                raise SNANAError(f"CLASSIFICATION {name}: MODE must be train or predict, not {mode}",
                                 function_name="pippin")
            entries = task.get("EXTERNAL_DIRS", [])
            external = _external_dirs(entries)
            if external and mode == "train":
                raise SNANAError(f"CLASSIFICATION {name}: EXTERNAL_DIRS on a training task is not "
                                 f"supported; name the model with MODEL", function_name="pippin")
            classifier_class = CLASSIFIER_REGISTRY[task["CLASSIFIER"]]
            supported = ("MODEL", "PROB_COLUMN_NAME") + classifier_class.PIPPIN_OPTS
            opts = _settings("CLASSIFICATION OPTS", name, task.get("OPTS"), supported)
            # Pippin names a training from its MODEL too (classifier.py:419-498),
            # which training here does not use
            if mode == "train" and "MODEL" in opts:
                raise SNANAError(f"CLASSIFICATION {name}: MODEL on a training task is not supported",
                                 function_name="pippin")
            locate = functools.partial(_locate, data_dirs=self.data_dirs, section="CLASSIFICATION",
                                       name=name)
            # each mask, when given, a plain substring of the SIM or DATAPREP
            # task's name, applied to every version of it (classifier.py:369-374)
            mask = task.get("MASK")
            mask_sim = task.get("MASK_SIM")
            for value in (mask, mask_sim):
                if value is not None and not isinstance(value, str):
                    raise SNANAError(f"CLASSIFICATION {name}: MASK and MASK_SIM must be strings",
                                     function_name="pippin")
            source_tasks = []
            for source_task in self.sources:
                if mask and mask not in source_task:
                    continue
                if mask_sim and mask_sim not in source_task:
                    continue
                source_tasks.append(source_task)
            if not source_tasks:
                raise SNANAError(f"CLASSIFICATION {name}: masks match no SIM or DATAPREP task",
                                 function_name="pippin")
            for source_task in source_tasks:
                classifier = classifier_class.from_pippin(name, opts, self.types[source_task], locate)
                if mode == "train":
                    self._train(name, classifier, opts, source_task)
                else:
                    self._predict(name, classifier, opts, source_task, external)

    def _train(self, name, classifier, opts, source_task):
        """A training on one source task, whose column is PROB_<task>_<source
        task> unless PROB_COLUMN_NAME is given (classifier.py:156-181).
        Pippin trains on one version only (classifier.py:393-401)."""
        versions = self.sources[source_task]
        if len(versions) > 1:
            raise SNANAError(f"CLASSIFICATION {name}: cannot train on {source_task}, which has "
                             f"{len(versions)} realisations", function_name="pippin")
        stage_name = f"{name}_{source_task}"
        column = opts.get("PROB_COLUMN_NAME", f"PROB_{stage_name}")
        trainee = dataclasses.replace(classifier, column=column)
        training = Train(stage_name, trainee, photometry=versions[0])
        self.trained.setdefault(name, []).append(training)

    def _predict(self, name, classifier, opts, source_task, external):
        """The predictions for every fit of one source task's versions, one
        for each model. BIASCOR reads the last model's, as Pippin's
        prob_cols keeps the last (biascor.py:615-619). Where EXTERNAL_DIRS
        holds the folder Pippin names for a prediction (classifier.py:187-202),
        its saved predictions are read instead."""
        if "MODEL" in opts:
            models = self._models(name, classifier, opts)
        else:
            column = opts.get("PROB_COLUMN_NAME", classifier.column)
            models = [(None, "", column)]
        versions = self.sources[source_task]
        for model, suffix, column in models:
            predictor = dataclasses.replace(classifier, column=column)
            self.columns[name] = column
            for index, source in enumerate(versions):
                if len(versions) > 1:
                    folder = f"{name}_{index + 1}_{source_task}{suffix}"
                else:
                    folder = f"{name}_{source_task}{suffix}"
                # this version's fit by each LCFIT task
                for fits in self.fitted[source_task].values():
                    fit_name = fits[index]
                    stage_name = f"{name}_{fit_name}{suffix}"
                    if folder not in external:
                        classify = Classify(stage_name, predictor, fit=self.fits[fit_name], model=model,
                                            photometry=source)
                    else:
                        saved = _locate(external[folder], self.data_dirs, "CLASSIFICATION", name)
                        classify = Classify(stage_name, fit=self.fits[fit_name],
                                            predictions=saved / "predictions.csv")
                    self.classifications.append(classify)
                    self.classified[(name, fit_name)] = classify

    def _models(self, name, classifier, opts) -> list[tuple[Train | Path, str, str]]:
        """Each model a prediction uses, with the suffix Pippin
        adds to the task's name for it and the probability column it writes,
        unless PROB_COLUMN_NAME is given (classifier.py:156-181, 419-498):
        every training of the training task MODEL names, which must be of
        the same classifier, with suffix _<training> and its column; or a
        model file, with suffix _<folder> and column PROB_<folder>, from the
        folder holding it."""
        model = opts["MODEL"]
        if "/" in model or "." in model:
            path = Path(os.path.expandvars(model))
            # Pippin reads a relative path from its own output directory, which
            # jax has not got
            if not path.is_absolute():
                raise SNANAError(f"CLASSIFICATION {name}: MODEL file {model} must be an absolute path",
                                 function_name="pippin")
            if not path.exists():
                raise SNANAError(f"CLASSIFICATION {name}: MODEL file {path} does not exist",
                                 function_name="pippin")
            column = opts.get("PROB_COLUMN_NAME", f"PROB_{path.parent.name}")
            return [(path, f"_{path.parent.name}", column)]
        if model not in self.trained:
            raise SNANAError(f"CLASSIFICATION {name}: MODEL {model} names no training task before it",
                             function_name="pippin")
        models = []
        for training in self.trained[model]:
            if type(training.classifier) is not type(classifier):
                raise SNANAError(f"CLASSIFICATION {name}: MODEL {model} trains a different classifier",
                                 function_name="pippin")
            column = opts.get("PROB_COLUMN_NAME", training.classifier.column)
            models.append((training, f"_{training.name}", column))
        return models

    def read_aggregation(self):
        """AGGREGATION has no stage here. A RECALIBRATION is ignored, as in
        Pippin's aggregator, unless the task is LEGACY (aggregator.py:52-61),
        and MERGE_CLASSIFIERS is refused, as Pippin refuses it
        (aggregator.py:125-129)."""
        for name, task in self._section("AGGREGATION").items():
            # an empty task takes Pippin's defaults (aggregator.py:415-416)
            if task is None:
                continue
            if task.get("MERGE_CLASSIFIERS") is not None:
                raise SNANAError(f"AGGREGATION {name}: MERGE_CLASSIFIERS is no longer accepted; "
                                 f"name the probability columns in CLASSIFICATION",
                                 function_name="pippin")
            if not task.get("RECALIBRATION"):
                continue
            if task.get("LEGACY"):
                raise SNANAError(f"AGGREGATION {name}: the LEGACY aggregator's RECALIBRATION is "
                                 f"not supported", function_name="pippin")
            logger.info(f"AGGREGATION {name}: RECALIBRATION is ignored")

    def _tables(self, name, fit_names, groups) -> list[Stage | FitVersion]:
        """The fits a BIASCOR task reads, each with the column of every group
        of classification tasks, as Pippin's MERGE joins every classifier's
        column (merge.py:384-405): the fit itself when none classified it,
        the classification when one did, else their merge. Two tasks writing
        one column of a fit are refused now, before either runs."""
        tables = []
        for fit_name in fit_names:
            picked = {}
            writers = {}
            for classifiers in groups:
                for classifier in classifiers:
                    if (classifier, fit_name) not in self.classified:
                        continue
                    column = self.columns[classifier]
                    if column in writers and writers[column] != classifier:
                        raise SNANAError(f"BIASCOR {name}: {writers[column]} and {classifier} both "
                                         f"write {column} for {fit_name}", function_name="pippin")
                    writers[column] = classifier
                    picked[classifier] = self.classified[(classifier, fit_name)]
            # in name order, so tasks listing the same classifiers share a merge
            ordered = sorted(picked)
            classifications = []
            for classifier in ordered:
                classifications.append(picked[classifier])
            if not classifications:
                tables.append(self.fits[fit_name])
            elif len(classifications) == 1:
                tables.append(classifications[0])
            else:
                classifier_names = "_".join(ordered)
                merge_name = f"MERGE_{fit_name}_{classifier_names}"
                if merge_name not in self.merges:
                    self.merges[merge_name] = Merge(merge_name, classifications)
                tables.append(self.merges[merge_name])
        return tables

    def _classifiers(self, name, named) -> list[str]:
        """The classification tasks writing the probability column a BIASCOR
        CLASSIFIER resolves to, as Pippin resolves it (biascor.py:640-665):
        the column of the classification tasks it names, which must share
        one, ignoring entries that name none; or, when it names no task, the
        one column it names. Pippin's MERGE gives each fit that column from
        whichever task wrote it (merge.py:384-405)."""
        if named is None:
            return []
        if isinstance(named, list):
            entries = named
        else:
            entries = [named]
        columns = set()
        for entry in entries:
            if entry in self.unsupported:
                raise SNANAError(f"BIASCOR {name}: classifier {entry} is {self.unsupported[entry]}, "
                                 f"which has no stage here", function_name="pippin")
            if entry in self.columns:
                columns.add(self.columns[entry])
        if len(columns) > 1:
            raise SNANAError(f"BIASCOR {name}: CLASSIFIER {entries} write different probability "
                             f"columns, {sorted(columns)}", function_name="pippin")
        if columns:
            column = columns.pop()
        elif len(entries) > 1:
            raise SNANAError(f"BIASCOR {name}: CLASSIFIER {entries} names no classification task, "
                             f"and a probability column must be named alone", function_name="pippin")
        else:
            column = entries[0]
        writers = []
        for task, written in self.columns.items():
            if written == column:
                writers.append(task)
        if not writers:
            raise SNANAError(f"BIASCOR {name}: CLASSIFIER {entries[0]} names no classification "
                             f"task or probability column", function_name="pippin")
        return writers

    def _fit_refs(self, name, given, fits_of) -> list[str]:
        """The fits a BIASCOR setting names, one or a list, each
        ``<LCFIT task>_<source task>``."""
        if isinstance(given, list):
            refs = given
        else:
            refs = [given]
        for ref in refs:
            if ref not in fits_of:
                raise SNANAError(f"BIASCOR {name}: {ref} names no LCFIT task; the LCFIT tasks "
                                 f"are {sorted(fits_of)}", function_name="pippin")
        return refs

    def _corrections(self, name, key, given, numbers, fits_of, groups) -> tuple[list, list[str | None]]:
        """The correction simulations the BIASCOR setting ``key`` names, the
        first realisation of each fit, as Pippin uses them
        (biascor.py:414-435), with the columns of ``groups``; and the FITOPT
        each is read at, from ``numbers``, its ``<key>_FITOPTS`` indices, 0
        being the nominal fit."""
        fit_names = []
        for ref in self._fit_refs(name, given, fits_of):
            if len(fits_of[ref]) > 1:
                logger.warning(f"BIASCOR {name}: {ref} has {len(fits_of[ref])} realisations; "
                               f"its first is used, as in Pippin")
            fit_names.append(fits_of[ref][0])
        tables = self._tables(name, fit_names, groups)
        if not isinstance(numbers, list):
            numbers = [numbers]
        if numbers and len(numbers) != len(tables):
            raise SNANAError(f"BIASCOR {name}: {key}_FITOPTS must have one entry per {key} source",
                             function_name="pippin")
        labels = []
        for table, number in zip(tables, numbers):
            fitopts = table.systematics()[0]
            if type(number) is not int or not 0 <= number <= len(fitopts):
                raise SNANAError(f"BIASCOR {name}: {table.name} has no FITOPT index {number!r}",
                                 function_name="pippin")
            if number == 0:
                labels.append(None)
            else:
                labels.append(fitopts[number - 1])
        return tables, labels

    def read_biascor(self):
        # Pippin names an LCFIT task's fits of a source task <lcfit>_<source> (biascor.py:667-672)
        fits_of, source_of, lcfit_of = {}, {}, {}
        for source_task, fits_by_task in self.fitted.items():
            for lcfit_task, fits in fits_by_task.items():
                ref = f"{lcfit_task}_{source_task}"
                fits_of[ref] = fits
                source_of[ref] = source_task
                lcfit_of[ref] = lcfit_task
        for name, task in self._section("BIASCOR").items():
            task = _settings("BIASCOR", name, task, {
                "BASE", "DATA", "OPTS", "MUOPTS", "CLASSIFIER", "PROB_COLUMN_NAME", "BLIND",
                "USE_RECALIBRATED", "SIMFILE_BIASCOR", "SIMFILE_BIASCOR_FITOPTS", "SIMFILE_CCPRIOR",
                "SIMFILE_CCPRIOR_FITOPTS"})
            if "DATA" not in task:
                raise SNANAError(f"BIASCOR {name}: DATA is required", function_name="pippin")
            opts = _ignoring("BIASCOR OPTS", name, task.get("OPTS"))
            for key in ("REPLACE_SIMFILE_BIASCOR", "REPLACE_SIMFILE_CCPRIOR"):
                replacement = opts.get(key)
                if replacement is None:
                    continue
                if not isinstance(replacement, Mapping):
                    raise SNANAError(f"BIASCOR {name}: {key} must map FITOPT labels to files",
                                     function_name="pippin")
                # SNANA reads these files as written, from a job folder of its
                # own (submit_prog_bbc.py:941, 1952), so only absolute paths work
                for path in replacement.values():
                    if not Path(path).is_absolute():
                        raise SNANAError(f"BIASCOR {name}: {key} file {path} must be an absolute "
                                         f"path", function_name="pippin")
            split = opts.get("NSPLITRAN", 1)
            if type(split) is not int or split < 1:
                raise SNANAError(f"BIASCOR {name}: NSPLITRAN must be a positive integer",
                                 function_name="pippin")
            data_refs = self._fit_refs(name, task["DATA"], fits_of)
            self.data_fitopts_files[name] = []
            for ref in data_refs:
                path = self.fitopts_files[lcfit_of[ref]]
                if path is not None:
                    self.data_fitopts_files[name].append(path)
            classifiers = self._classifiers(name, task.get("CLASSIFIER"))
            # PROB_COLUMN_NAME is the column read whatever the CLASSIFIER
            # (biascor.py:118-121), so the fits are read as classified by the
            # tasks writing it, as Pippin's MERGE joins every classifier's
            # column to them (merge.py:384-405)
            if "PROB_COLUMN_NAME" in task:
                column = task["PROB_COLUMN_NAME"]
                classifiers = []
                for classifier, written in self.columns.items():
                    if written == column:
                        classifiers.append(classifier)
            elif classifiers:
                column = self.columns[classifiers[0]]
            else:
                column = None
            recalibrated = task.get("USE_RECALIBRATED", False)
            if recalibrated:
                if column is None:
                    raise SNANAError(f"BIASCOR {name}: USE_RECALIBRATED needs CLASSIFIER or "
                                     f"PROB_COLUMN_NAME", function_name="pippin")
                column = column.replace("PROB_", "CPROB_")
            overrides = self._bbc_overrides(name, opts)
            if column is not None:
                overrides["varname_pIa"] = column
            # Pippin's blinding: the task's BLIND, else its blindflag, else its data's (biascor.py:198-204)
            if "BLIND" in task:
                blind = task["BLIND"]
            elif "blindflag" in opts:
                blind = opts["blindflag"] != 0
            else:
                blind = any(self.blind[source_of[ref]] for ref in data_refs)
            if blind:
                overrides["blindflag"] = 2
            else:
                overrides["blindflag"] = 0
            # the task's fits carry the column of its classifiers and of each
            # MUOPT's own, as Pippin's MERGE gives them every classifier's
            groups = [classifiers]
            muopts = {}
            muopt_biascor_fitopts = {}
            muopt_ccprior_fitopts = {}
            self.muopt_scales[name] = {}
            for label, given in task.get("MUOPTS", {}).items():
                variant = _settings("BIASCOR MUOPT", f"{name}/{label}", given, {
                    "OPTS", "CLASSIFIER", "SIMFILE_BIASCOR", "SIMFILE_BIASCOR_FITOPTS",
                    "SIMFILE_CCPRIOR", "SIMFILE_CCPRIOR_FITOPTS", "SCALE"})
                variant_opts = _ignoring("BIASCOR MUOPT OPTS", f"{name}/{label}", variant.get("OPTS"))
                misplaced = sorted(_TASK_ONLY_OPTS & set(variant_opts))
                if misplaced:
                    raise SNANAError(f"BIASCOR MUOPT {name}/{label}: {', '.join(misplaced)} belongs "
                                     f"in the task OPTS, not a MUOPT", function_name="pippin")
                # Pippin writes a MUOPT's list value as the list's text (biascor.py:550-560)
                for key, value in variant_opts.items():
                    if isinstance(value, list):
                        raise SNANAError(f"BIASCOR MUOPT {name}/{label}: {key} must be one value, "
                                         f"not a list", function_name="pippin")
                keys = self._bbc_overrides(name, variant_opts)
                # a MUOPT's own classifier gives its column, not recalibrated
                # (biascor.py:541-546)
                own = self._classifiers(name, variant.get("CLASSIFIER"))
                muopt_classifiers = classifiers
                if own:
                    muopt_classifiers = own
                    keys["varname_pIa"] = self.columns[own[0]]
                    groups.append(own)
                # Pippin reads a MUOPT's FITOPTS only with its simulations (biascor.py:512-541)
                if "SIMFILE_BIASCOR" in variant:
                    tables, fitopts = self._corrections(
                        name, "SIMFILE_BIASCOR", variant["SIMFILE_BIASCOR"],
                        variant.get("SIMFILE_BIASCOR_FITOPTS", []), fits_of, [muopt_classifiers])
                    keys["simfile_biascor"] = tables
                    if "SIMFILE_BIASCOR_FITOPTS" in variant:
                        muopt_biascor_fitopts[label] = fitopts
                if "SIMFILE_CCPRIOR" in variant:
                    tables, fitopts = self._corrections(
                        name, "SIMFILE_CCPRIOR", variant["SIMFILE_CCPRIOR"],
                        variant.get("SIMFILE_CCPRIOR_FITOPTS", []), fits_of, [muopt_classifiers])
                    keys["simfile_ccprior"] = tables
                    if "SIMFILE_CCPRIOR_FITOPTS" in variant:
                        muopt_ccprior_fitopts[label] = fitopts
                if "SCALE" in variant:
                    self.muopt_scales[name][label] = variant["SCALE"]
                muopts[label] = keys
            biascor = None
            biascor_fitopts = []
            if "SIMFILE_BIASCOR" in task:
                biascor, biascor_fitopts = self._corrections(
                    name, "SIMFILE_BIASCOR", task["SIMFILE_BIASCOR"],
                    task.get("SIMFILE_BIASCOR_FITOPTS", []), fits_of, groups)
            ccprior = None
            ccprior_fitopts = []
            if "SIMFILE_CCPRIOR" in task:
                ccprior, ccprior_fitopts = self._corrections(
                    name, "SIMFILE_CCPRIOR", task["SIMFILE_CCPRIOR"],
                    task.get("SIMFILE_CCPRIOR_FITOPTS", []), fits_of, groups)
            fitopt_muopt = opts.get("FITOPTxMUOPT", [])
            if not isinstance(fitopt_muopt, list):
                fitopt_muopt = [fitopt_muopt]
            input_file = _locate(task.get("BASE", _DEFAULT_BBC), self.data_dirs, "BIASCOR", name)
            sync_events = sync_setting(opts)
            # the realisation counts of the DATA that has several
            counts = set()
            for ref in data_refs:
                if len(fits_of[ref]) > 1:
                    counts.add(len(fits_of[ref]))
            if len(counts) > 1:
                raise SNANAError(f"BIASCOR {name}: its DATA must have the same number of "
                                 f"realisations", function_name="pippin")
            if counts:
                realisations = range(1, counts.pop() + 1)
            else:
                realisations = [None]
            for realisation in realisations:
                data = []
                for ref in data_refs:
                    fits = fits_of[ref]
                    if len(fits) > 1:
                        data.append(fits[realisation - 1])
                    else:
                        data.append(fits[0])
                data_tables = self._tables(name, data, groups)
                if realisation is None:
                    version = name
                else:
                    version = f"{name}-{realisation:04d}"
                for partition in range(1, split + 1):
                    if split > 1:
                        stage_name = f"{version}-SPLIT{partition:03d}"
                    else:
                        stage_name = version
                    bbc = BBC(
                        stage_name, data=data_tables, input_file=input_file,
                        biascor=biascor, ccprior=ccprior,
                        biascor_fitopts=biascor_fitopts, ccprior_fitopts=ccprior_fitopts,
                        muopt_biascor_fitopts=muopt_biascor_fitopts,
                        muopt_ccprior_fitopts=muopt_ccprior_fitopts,
                        replace_biascor=opts.get("REPLACE_SIMFILE_BIASCOR", {}),
                        replace_ccprior=opts.get("REPLACE_SIMFILE_CCPRIOR", {}),
                        overrides=overrides, muopts=muopts,
                        split_count=split, split_index=partition, realisation=realisation,
                        require_probability_column=bool(recalibrated),
                        sync_events=sync_events, fitopt_muopt=fitopt_muopt)
                    # its FITOPT and MUOPT grid checked now, while the config is read
                    bbc.grid_cells()
                    self.bbcs.setdefault(name, []).append(bbc)

    def _bbc_overrides(self, name, opts) -> dict:
        """Pippin's BIASCOR OPTS as SALT2mu keys (biascor.py:462-478, 550-560):
        ``CUTWIN_X``, in any case, is the cut ``CUTWIN X``, and
        ``CUTWIN_PROB_IA`` the cut on SALT2mu's ``varname_pIa``, the fit's
        probability column. Of a list of cuts SALT2mu keeps the last; a list
        for any other key would repeat it, which SALT2mu refuses. Task OPTS
        read here, not by SALT2mu, are left out."""
        overrides = {}
        for key, value in opts.items():
            if key in _TASK_ONLY_OPTS:
                continue
            if key.upper().startswith("CUTWIN_"):
                variable = key.split("_", 1)[1]
                if variable.upper() == "PROB_IA":
                    variable = "varname_pIa"
                if isinstance(value, list):
                    value = value[-1]
                overrides[f"CUTWIN {variable}"] = str(value)
            elif isinstance(value, list):
                raise SNANAError(f"BIASCOR {name}: {key} must be one value, not a list, as SALT2mu "
                                 f"refuses a repeated key", function_name="pippin")
            else:
                overrides[key] = str(value)
        return overrides

    def read_create_cov(self):
        supported = {"COVOPTS", "BINNED", "SYS_SCALE", "FITOPT_SCALES",
                     "MUOPT_SCALES", "EXTRA_COVS", "CALIBRATORS", "REBINNED_X1",
                     "REBINNED_C", "REBIN_BINS", "SYSTEMATIC_HD", "SUBTRACT_VPEC", "EXTRA_OPTS"}
        for name, task in self._section("CREATE_COV").items():
            task = _settings("CREATE_COV", name, task, {"MASK", "MASK_BIASCOR", "OPTS"})
            opts = _settings("CREATE_COV OPTS", name, task.get("OPTS"), supported)
            # an empty EXTRA_OPTS adds nothing (create_cov.py:228-230)
            flags = ""
            if opts.get("EXTRA_OPTS"):
                flags = opts["EXTRA_OPTS"]
            writer_keys, label_cov_rows, mxsize_test_posdef = _cov_extra_options(name, flags)
            # a plain substring of the BIASCOR task's name, MASK before
            # MASK_BIASCOR (create_cov.py:340-345)
            mask = ""
            if "MASK" in task:
                mask = task["MASK"]
            elif "MASK_BIASCOR" in task:
                mask = task["MASK_BIASCOR"]
            if not isinstance(mask, str):
                raise SNANAError(f"CREATE_COV {name}: MASK must be a string", function_name="pippin")
            bbc_tasks = [bbc_task for bbc_task in self.bbcs if mask in bbc_task]
            if not bbc_tasks:
                raise SNANAError(f"CREATE_COV {name}: mask {mask!r} matches no BIASCOR task",
                                 function_name="pippin")
            scales = self._cov_scales(name, opts, bbc_tasks)
            # Pippin takes rebinning, else SUBTRACT_VPEC, else BINNED (create_cov.py:207-221)
            rebinning = False
            if opts.get("REBIN_BINS") is not None or opts.get("REBINNED_X1") or opts.get("REBINNED_C"):
                rebinning = True
            subtract_vpec = None
            mode = None
            if not rebinning:
                if opts.get("SUBTRACT_VPEC"):
                    subtract_vpec = True
                elif not opts.get("BINNED", True):
                    mode = "unbinned"
            extra = []
            # SNANA reads a label, a scale and a file, the first three words
            # (create_covariance.py:2853-2856)
            for entry in opts.get("EXTRA_COVS", []):
                words = str(entry).split()
                if len(words) < 3:
                    raise SNANAError(f"CREATE_COV {name}: EXTRA_COVS needs 'label scale file' entries",
                                     function_name="pippin")
                try:
                    scale = float(words[1])
                    valid = math.isfinite(scale)
                except ValueError:
                    valid = False
                if not valid:
                    raise SNANAError(f"CREATE_COV {name}: invalid EXTRA_COVS scale {words[1]!r}",
                                     function_name="pippin")
                # SNANA expands variables and reads the file from a job folder of its
                # own (create_covariance.py:1471, submit_prog_covmat.py:332), so only
                # absolute paths work
                path = Path(os.path.expandvars(words[2]))
                if not path.is_absolute():
                    raise SNANAError(f"CREATE_COV {name}: EXTRA_COVS file {words[2]} must be an "
                                     f"absolute path", function_name="pippin")
                extra.append((words[0], scale, path))
            calibrators = opts.get("CALIBRATORS", [])
            if not isinstance(calibrators, list):
                calibrators = [calibrators]
            covopts = opts.get("COVOPTS", [])
            if not isinstance(covopts, list):
                covopts = [covopts]
            matched = []
            for bbc_task in bbc_tasks:
                matched.extend(self.bbcs[bbc_task])
            covariances = []
            for bbc in matched:
                stage_name = name
                if len(matched) > 1:
                    stage_name = f"{name}_{bbc.name}"
                covariance = Covariance(
                    stage_name, distances=bbc, mode=mode,
                    rebin_bins=opts.get("REBIN_BINS"),
                    nbin_x1=opts.get("REBINNED_X1"), nbin_c=opts.get("REBINNED_C"),
                    overrides=writer_keys,
                    label_cov_rows=label_cov_rows,
                    mxsize_test_posdef=mxsize_test_posdef,
                    extra_covs=extra,
                    calibrators=calibrators,
                    systematic_hd=opts.get("SYSTEMATIC_HD"),
                    subtract_vpec=subtract_vpec,
                    covopts=covopts,
                    scales=scales[bbc.name])
                # its settings checked now, while the config is read
                covariance.configuration()
                covariances.append(covariance)
            self.covariances[name] = covariances

    def _cov_scales(self, name, opts, bbc_tasks) -> dict[str, dict[str, float]]:
        """Each BBC fit's systematic scales, by fit name, as Pippin and SNANA
        take them (create_cov.py:234-283, create_covariance.py:1098). Later
        sources override earlier ones: the BIASCOR task's MUOPT SCALEs; a
        FITOPTS file's scales, SYS_SCALE's or else that of the first task
        whose DATA was fitted with one; FITOPT_SCALES; MUOPT_SCALES. A label
        the fit lacks is left out."""
        scale_file = None
        if "SYS_SCALE" in opts:
            scale_file = _locate(opts["SYS_SCALE"], self.data_dirs, "CREATE_COV", name)
        else:
            for bbc_task in bbc_tasks:
                files = self.data_fitopts_files[bbc_task]
                if len(set(files)) > 1:
                    raise SNANAError(f"CREATE_COV {name}: the DATA of BIASCOR {bbc_task} was fitted "
                                     f"with more than one FITOPTS file; give SYS_SCALE",
                                     function_name="pippin")
                if files:
                    scale_file = files[0]
                    break
        given = {}
        if scale_file is not None:
            # a later block's scale wins
            for block, labelled in yaml.safe_load(scale_file.read_text()).items():
                if block == "FLAG_USE_SAME_EVENTS":
                    continue
                for label, scale_command in labelled.items():
                    given[label] = float(str(scale_command).split()[0])
        given.update(opts.get("FITOPT_SCALES", {}))
        given.update(opts.get("MUOPT_SCALES", {}))
        scales = {}
        for bbc_task in bbc_tasks:
            declared = dict(self.muopt_scales[bbc_task])
            declared.update(given)
            for bbc in self.bbcs[bbc_task]:
                fitopts, muopts = bbc.systematics()
                scales[bbc.name] = {}
                for label, scale in declared.items():
                    if label in fitopts or label in muopts:
                        scales[bbc.name][label] = float(scale)
        return scales

    def read_cosmofit(self):
        section = self._section("COSMOFIT")
        wfit = section.get("WFIT") or {}
        if not isinstance(wfit, Mapping):
            raise SNANAError("COSMOFIT WFIT: expected a mapping of named tasks", function_name="pippin")
        # Pippin skips a fitter it does not know, finds wfit in any case, but
        # reads its tasks only from WFIT, and refuses a wfit that makes none
        # (cosmofit.py:33-49, wfit.py:128-129)
        for fitter in section:
            if fitter.lower() == "wfit" and not wfit:
                raise SNANAError(f"COSMOFIT {fitter} gives no WFIT tasks; name them under WFIT",
                                 function_name="pippin")
            if fitter != "WFIT":
                logger.info(f"COSMOFIT {fitter} has no stage here, skipped")
        for name, task in wfit.items():
            task = _settings("COSMOFIT WFIT", name, task, {"MASK", "OPTS"})
            opts = _settings("COSMOFIT WFIT OPTS", name, task.get("OPTS"),
                             {"WFITOPTS", "WFITOPT_GLOBAL", "BLIND_DATA", "BLIND_SIM", "COVOPT", "COVOPTS",
                              "USE_COVTOT_FACTORIZED", "USE_COVSYS_INV"})
            if not opts.get("WFITOPTS"):
                raise SNANAError(f"COSMOFIT {name}: WFITOPTS must name at least one fit",
                                 function_name="pippin")
            # SNANA blinds data, and not simulations, unless told otherwise
            # (submit_prog_cosmofit.py:732-784); wfit tells them apart itself
            blind_data = opts.get("BLIND_DATA", True)
            blind_sim = opts.get("BLIND_SIM", False)
            if blind_data and blind_sim:
                blind_mode = "blind"
            elif blind_data:
                blind_mode = "auto"
            elif blind_sim:
                raise SNANAError(f"COSMOFIT {name}: BLIND_SIM without BLIND_DATA is not supported",
                                 function_name="pippin")
            else:
                blind_mode = "none"
            # COVOPTS, else COVOPT, picks the covariances fitted, by label
            # (submit_prog_cosmofit.py:103, 274-277)
            selected = None
            for key in ("COVOPT", "COVOPTS"):
                if key in opts:
                    if not isinstance(opts[key], str):
                        raise SNANAError(f"COSMOFIT {name}: {key} must be one string of labels",
                                         function_name="pippin")
                    selected = opts[key].split()
            # a plain substring of the CREATE_COV task's name (wfit.py:134-137)
            mask = task.get("MASK", "")
            if not isinstance(mask, str):
                raise SNANAError(f"COSMOFIT {name}: MASK must be a string", function_name="pippin")
            matched = []
            for covariance_task, covariances in self.covariances.items():
                if mask in covariance_task:
                    matched.extend(covariances)
            if not matched:
                raise SNANAError(f"COSMOFIT {name}: mask {mask!r} matches no CREATE_COV task",
                                 function_name="pippin")
            # each covariance's labels fitted: ALL and its COVOPTS, or those selected
            covopts_of = {}
            for covariance in matched:
                covopts = ["ALL"]
                for covopt in covariance.covopts:
                    covopts.append(parse_covopt(covopt).label)
                if selected is not None:
                    covopts = [covopt for covopt in covopts if covopt in selected]
                covopts_of[covariance.name] = tuple(covopts)
            global_flags = opts.get("WFITOPT_GLOBAL", "")
            entries = opts["WFITOPTS"]
            if not isinstance(entries, list):
                entries = [entries]
            for index, entry in enumerate(entries):
                # /label/ flags with a one-word label, or flags alone, which SNANA
                # numbers; any / makes it a label (submit_util.py:399-435)
                label = f"WFITOPT{index:03d}"
                flags = entry
                if "/" in entry:
                    if not entry.startswith("/"):
                        raise SNANAError(f"COSMOFIT {name}: WFITOPTS entry {entry!r} needs a one-word "
                                         f"/label/", function_name="pippin")
                    label, slash, flags = entry[1:].partition("/")
                    if not slash or len(label.split()) != 1:
                        raise SNANAError(f"COSMOFIT {name}: WFITOPTS entry {entry!r} needs a one-word "
                                         f"/label/", function_name="pippin")
                    label = label.strip()
                flags = f"{flags} {global_flags}"
                # SNANA refuses -blind in any WFITOPT (submit_prog_cosmofit.py:753-766)
                if "-blind" in flags:
                    raise SNANAError(f"COSMOFIT {name}: -blind cannot be a WFITOPT; set BLIND_DATA "
                                     f"and BLIND_SIM", function_name="pippin")
                overrides = _wfit_keys(name, flags)
                overrides["blind_mode"] = blind_mode
                for covariance in matched:
                    # SNANA makes no fit of a covariance none of whose labels
                    # is selected (submit_prog_cosmofit.py:608-628)
                    if not covopts_of[covariance.name]:
                        continue
                    stage_name = f"WFIT_{name}_{label}"
                    if len(matched) > 1:
                        stage_name = f"{stage_name}_{covariance.name}"
                    # the covariance form wfit reads (submit_prog_cosmofit.py:533-543)
                    cosmology = Cosmology(
                        stage_name, covariance=covariance, overrides=overrides,
                        covopts=covopts_of[covariance.name],
                        use_covtot_factorized=opts.get("USE_COVTOT_FACTORIZED", True),
                        use_covsys_inv=opts.get("USE_COVSYS_INV", True))
                    self.cosmologies.append(cosmology)


def _cov_extra_options(name, value) -> tuple[dict, bool | None, int | None]:
    """The create_cov writer flags given in EXTRA_OPTS: the writer keys,
    --label_cov_rows and --mxsize_test_posdef, each None when not given; the
    covariance stage checks them and supplies defaults."""
    if not isinstance(value, str):
        raise SNANAError(f"CREATE_COV {name}: EXTRA_OPTS must be a string of flags", function_name="pippin")
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False, exit_on_error=False)
    parser.add_argument("--write_mask_cov", type=int)
    parser.add_argument("--write_format_cov", choices=("text", "npz"))
    parser.add_argument("--label_cov_rows", action="store_true", default=None)
    parser.add_argument("--mxsize_test_posdef", type=int)
    try:
        args, unknown = parser.parse_known_args(shlex.split(value))
    except (argparse.ArgumentError, ValueError) as error:
        raise SNANAError(f"CREATE_COV {name}: EXTRA_OPTS: {error}", function_name="pippin") from error
    if unknown:
        raise SNANAError(f"CREATE_COV {name}: EXTRA_OPTS: unrecognised flags {' '.join(unknown)}",
                         function_name="pippin")
    overrides = {}
    if args.write_mask_cov is not None:
        overrides["WRITE_MASK_COV"] = args.write_mask_cov
    if args.write_format_cov is not None:
        overrides["WRITE_FORMAT_COV"] = args.write_format_cov
    return overrides, args.label_cov_rows, args.mxsize_test_posdef


def _wfit_keys(name: str, flags: str) -> dict:
    """wfit.exe flags as the config keys the jax wfit takes, through its own
    command line, so a flag it refuses is refused here, while the config is
    read."""
    parser = build_parser()
    try:
        args = parser.parse_args(shlex.split(flags))
    except (SystemExit, ValueError) as error:
        raise SNANAError(f"COSMOFIT {name}: wfit does not take the flags {flags!r}",
                         function_name="pippin") from error
    return cli_keys(args)


def _include_files(text: str) -> str:
    """Pippin's preprocessing (run.py:103-142): each ``# % include: PATH %``
    line is replaced by the text of PATH, relative to the working directory."""
    lines = []
    for line in text.split("\n"):
        words = line[1:].split() if line.startswith("#") else []
        if not words or not (words[0].startswith("%") and words[-1].endswith("%")):
            lines.append(line)
            continue
        step, *values = line[1:].replace("%", "").split()
        if step != "include:":
            logger.warning(f"Unknown Pippin preprocessing step {step}, skipped")
            lines.append(line)
            continue
        included = Path(os.path.expandvars(values[0]))
        if not included.is_file():
            raise SNANAError(f"included file {included} does not exist", function_name="pippin")
        lines.append(included.read_text())
    return "\n".join(lines)


def read(path, data_dirs: Sequence = ()) -> list[Stage]:
    """The stages a Pippin config describes.

    Relative paths the config names are searched for in its GLOBAL
    DATA_DIRS and then in ``data_dirs``, as Pippin searches its config's
    then its installation's.
    """
    config = yaml.safe_load(_include_files(Path(path).read_text()))
    if not isinstance(config, Mapping):
        raise SNANAError(f"{path} is not a mapping of sections", function_name="pippin")
    # GLOBAL overrides Pippin's installation settings (run.py:158-170), of
    # which only DATA_DIRS and the QUEUE prefix mean anything here
    global_settings = config.get("GLOBAL") or {}
    if not isinstance(global_settings, Mapping):
        raise SNANAError("GLOBAL config: expected a mapping of settings", function_name="pippin")
    for key in global_settings:
        if key not in ("DATA_DIRS", "QUEUE"):
            logger.info(f"GLOBAL config: {key} is ignored")
    global_dirs = global_settings.get("DATA_DIRS", [])
    if not isinstance(global_dirs, list):
        raise SNANAError("GLOBAL config: DATA_DIRS must be a list", function_name="pippin")
    entries = list(global_dirs)
    entries.extend(data_dirs)
    dirs = []
    for entry in entries:
        directory = Path(os.path.expandvars(str(entry)))
        if not directory.is_dir():
            raise SNANAError(f"data directory {entry} ({directory}) does not exist", function_name="pippin")
        dirs.append(directory)
    for section in config:
        if section not in ("GLOBAL", "DATAPREP", "SIM", "LCFIT", "CLASSIFICATION", "BIASCOR",
                           "CREATE_COV", "COSMOFIT"):
            logger.info(f"{section} has no stage here, skipped")
    # Pippin's simulation folders start with its queue prefix and the
    # config's name (run.py:172, manager.py:64)
    queue = global_settings.get("QUEUE") or {}
    if not isinstance(queue, Mapping):
        raise SNANAError("GLOBAL QUEUE: expected a mapping of settings", function_name="pippin")
    for key in queue:
        if key != "prefix":
            logger.info(f"GLOBAL QUEUE: {key} is ignored")
    queue_prefix = queue.get("prefix", _DEFAULT_QUEUE_PREFIX)
    config_name = Path(path).name.split(".")[0].upper()
    graph = _Graph(config, f"{queue_prefix}_{config_name}", dirs)
    graph.read_dataprep()
    graph.read_sim()
    graph.read_lcfit()
    graph.read_classification()
    graph.read_aggregation()
    graph.read_biascor()
    graph.read_create_cov()
    graph.read_cosmofit()
    return graph.stages()


def _paths(value):
    """Every path in a stage's setting, including those inside mappings,
    tuples and lists."""
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
    stages = validate_graph(read(path, data_dirs))
    plan = []
    for stage in stages:
        paths = []
        for setting in dataclasses.fields(stage):
            paths.extend(_paths(getattr(stage, setting.name)))
        paths.extend(stage.cache_files())
        if isinstance(stage, Fit):
            paths.extend(stage.source_files())
        for source in paths:
            if not source.exists():
                raise SNANAError(f"input {source} does not exist", function_name=stage.name)
        inputs = {}
        for role, used in stage.inputs().items():
            inputs[role] = used.name
        entry = {"name": stage.name, "stage": type(stage).__name__, "inputs": inputs}
        if paths:
            files = []
            for source in paths:
                if str(source) not in files:
                    files.append(str(source))
            entry["files"] = files
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
            muopt_indices = set()
            for cell in stage.grid_cells():
                muopt_indices.add(cell[1])
            for muopt in sorted(muopt_indices):
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
            covopts = ["ALL"]
            for covopt in config.covopts:
                covopts.append(covopt.label)
            entry.update(mode=config.mode, covopts=covopts)
        elif isinstance(stage, Cosmology):
            keys = {}
            if stage.input_file is not None:
                keys = read_cosmology_keys(str(stage.input_file))
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
        if isinstance(stage, (Fit, Classify, Merge, BBC)):
            fitopts, muopts = stage.systematics()
            combinations = []
            for i, j in stage.grid_cells():
                fitopt = "DEFAULT"
                if i > 0:
                    fitopt = fitopts[i - 1]
                muopt = "DEFAULT"
                if j > 0:
                    muopt = muopts[j - 1]
                combinations.append({"directory": fitopt_muopt_name((i, j)),
                                     "fitopt": fitopt, "muopt": muopt})
            entry["combinations"] = combinations
        plan.append(entry)
    return {"configuration": str(path), "stages": plan}


def run_pippin(path, outdir, data_dirs: Sequence = (), refresh=()) -> dict[str, Any]:
    """Run a Pippin config as a pipeline; see :func:`read` and :func:`run`."""
    return run(*read(path, data_dirs), outdir=outdir, refresh=refresh)


@aborting
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="jax_pipeline", description="Run an analysis from a Pippin YAML configuration",
        epilog="Completed stages are reused when their inputs are unchanged. "
               "Use --check to inspect the analysis before running it.")
    parser.add_argument("config", help="Pippin YAML configuration file")
    parser.add_argument("--outdir", default="run", metavar="DIR",
                        help="output directory (default: run)")
    parser.add_argument("--data-dir", action="append", default=[], metavar="DIR",
                        help="search directory for relative input paths, after the configuration's "
                             "DATA_DIRS; repeatable, searched in order")
    refresh = parser.add_mutually_exclusive_group()
    refresh.add_argument("--refresh", nargs="+", default=(), metavar="STAGE",
                         help="rerun named stages and all stages depending on them")
    refresh.add_argument("--refresh-all", action="store_true",
                         help="rerun every stage, replacing its previous outputs")
    parser.add_argument("--double-precision", action="store_true",
                        help="use 64-bit floating-point calculations throughout the analysis")
    parser.add_argument("--check", action="store_true",
                        help="validate configuration and print the planned analysis without running")
    args = parser.parse_args(argv)
    if args.check and (args.refresh or args.refresh_all):
        parser.error("--check does not run stages; omit --refresh and --refresh-all")
    logging.basicConfig(format="%(message)s", level=logging.INFO,
                        stream=sys.stdout)
    log.startup(logger, "jax_pipeline",
                sys.argv if argv is None else ["jax_pipeline"] + argv)
    configure_precision(args.double_precision)
    if args.check:
        print(yaml.safe_dump(check(args.config, args.data_dir), sort_keys=False))
    else:
        run_pippin(args.config, args.outdir, args.data_dir,
                   refresh=True if args.refresh_all else args.refresh)
    log.done(logger, "jax_pipeline")


if __name__ == "__main__":
    raise SystemExit(main())
