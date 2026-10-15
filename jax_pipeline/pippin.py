"""Pippin analysis configs as graphs of stages.

A Pippin YAML names tasks by section and joins them by substring masks and
exact task names. ``read`` turns one into the stage objects a script would
build by hand, so nothing runs differently for having come from YAML.
Sections with no stage here, AGGREGATION and
MERGE (the classifier join is the classify stage's), COSMOMC and ANALYSE, are
skipped; a task that depends on something skipped or unsupported aborts
naming it. Scheduling keys are ignored with a log line each.
"""

import argparse
import copy
import glob
import logging
import math
import os
import re
import shlex
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from astropy.io import fits as pyfits

from jax_create_cov.config import resolve_rebin_bins
from jax_create_cov.config_parser import parse_covopt
from jax_snlc_core.io.fits_path import open_text
from jax_snlc_core.utils import group_equal
from jax_snlc_core.utils import snana_log as slog
from jax_snlc_core.utils.error_handling import SNANAError, aborting
from jax_snlc_core.utils.precision import configure_precision
from jax_snlc_fit.config_parser import group_override_tokens, known_keys
from jax_snlc_sim.simulation import FORMAT_MASK_CIDRAN
from jax_wfit.config_parser import build_parser, cli_keys

from .classify import Classify, Train
from .classifiers.perfect import Perfect
from .classifiers.supernnova import SuperNNova
from .run import run
from .stages import (BBC, Cosmology, Covariance, Data, Fit, FitVersion, Sim, Stage,
                     sync_setting)

logger = logging.getLogger(__name__)

#: Pippin's default distance-fit input, relative to its data directories
_DEFAULT_BBC = "surveys/des/bbc/bbc_5yr.input"
#: SIM GLOBAL keys that steer SNANA's batch layer rather than a simulation
_SIM_BATCH_KEYS = ("BATCH_INFO", "BATCH_MEM", "RESET_CIDOFF")
#: the keys of a CLASSIFICATION task read here, and the OPTS of each classifier
_CLASSIFICATION_KEYS = ("CLASSIFIER", "MODE", "OPTS", "MASK", "MASK_SIM", "MASK_FIT",
                        "EXTERNAL_DIRS", "GPU", "CLEAN")
_PERFECT_OPTS = ("MODEL", "PROB_COLUMN_NAME", "PROB_IA", "PROB_CC")
#: the SuperNNova OPTS read here, and Pippin's default LIST_FILTERS
_SNN_OPTS = ("MODEL", "PROB_COLUMN_NAME", "LIST_FILTERS", "VARIANT", "REDSHIFT",
             "NORM", "CYCLIC", "SEED", "BATCH_SIZE", "NUM_LAYERS", "HIDDEN_DIM",
             "SNTYPES")
_SNN_FILTERS = ("DES-g", "DES-i", "DES-r", "DES-z")
#: the RANSEED_REPEAT and FORMAT_MASK of Pippin's own combine.input, used when a SIM task gives none
_DEFAULT_RANSEED_REPEAT = "1 12345"
_DEFAULT_FORMAT_MASK = 48


def _seed(seed: int, job: int, format_mask: int) -> int:
    """The seed SNANA's batch layer gives job ``job`` of a RANSEED_REPEAT
    (submit_prog_sim.py:532).

    SNANA uses this formula only when the CIDs are not random; with random
    CIDs (the CIDRAN bit of ``format_mask``) it gives every job the same
    seed, and each job's CIDOFF shifts its randoms instead
    (submit_prog_sim.py:525-527).
    """
    if format_mask & FORMAT_MASK_CIDRAN:
        return seed
    return seed + 10000 * job + job * job + 13


def _fail(section: str, name: str, why: str):
    raise SNANAError(f"{section} {name}: {why}", function_name="pippin")


_BATCH_KEYS = {"BATCH_INFO", "BATCH_FILE", "BATCH_REPLACE", "BATCH_MEM",
               "BATCH_WALLTIME", "NUM_JOBS"}


def _validate_keys(section, name, values, supported, *, batch=False):
    if not isinstance(values, Mapping):
        _fail(section, name, "expected a mapping of settings")
    ignored = _BATCH_KEYS if batch else set()
    unknown = set(values) - set(supported) - ignored
    if unknown:
        _fail(section, name, f"unsupported setting(s): {', '.join(sorted(map(str, unknown)))}")
    for key in set(values) & ignored:
        logger.info(f"{section} {name}: {key} is a batch setting, ignored")


def _matches(mask, name: str) -> bool:
    """Pippin's mask: absent, empty or ``*`` matches all; a string is a
    substring; a list matches if any of its strings does."""
    if mask is None or mask in ("", "*"):
        return True
    return any(m in name for m in ([mask] if isinstance(mask, str) else mask))


def _value(value) -> str:
    """A YAML value as a SNANA input token."""
    return " ".join(str(v) for v in value) if isinstance(value, list) else str(value)


def _locate(text: str, data_dirs: Sequence[Path], section: str, name: str) -> Path:
    """Pippin's path rule: ``$VAR`` expanded, an absolute path as given, a
    relative one searched for under the data directories in order."""
    if "$" in text:
        expanded = os.path.expandvars(text)
        if "$" in expanded:
            _fail(section, name, f"{text} names an unset variable")
        if not Path(expanded).exists():
            _fail(section, name, f"{text} expands to {expanded}, which does not exist")
        return Path(expanded)
    if text.startswith("/"):
        if not Path(text).exists():
            _fail(section, name, f"{text} does not exist")
        return Path(text)
    for directory in data_dirs:
        if (Path(directory) / text).exists():
            return Path(directory) / text
    _fail(section, name, f"{text} is under none of the data directories {list(data_dirs)}")


def _survey(stage: Stage) -> str:
    """The survey a source stage's data belongs to, which chooses the FITOPTS
    block: the SIMLIB header for a simulation, the HEAD header for data."""
    if isinstance(stage, Sim):
        input_file, overrides = stage.input_files[0], stage.overrides[0]
        simlib = overrides.get("SIMLIB_FILE")
        if simlib is None:
            for line in input_file.read_text().splitlines():
                tokens = line.split()
                if tokens[:1] == ["SIMLIB_FILE:"]:
                    simlib = tokens[1]
        if simlib is None:
            raise SNANAError(f"{input_file} names no SIMLIB_FILE", function_name="pippin")
        with open_text(os.path.expandvars(simlib)) as f:
            for line in f:
                tokens = line.split()
                if "LIBID:" in tokens:
                    break
                if "SURVEY:" in tokens:
                    return tokens[tokens.index("SURVEY:") + 1]
        raise SNANAError(f"{simlib} names no SURVEY in its header", function_name="pippin")
    heads = sorted(glob.glob(str(stage.directory / "*_HEAD.FITS*")))
    if not heads:
        raise SNANAError(f"{stage.directory} holds no HEAD file", function_name="pippin")
    return str(pyfits.getheader(heads[0], 0)["SURVEY"]).strip()


class _Graph:
    """The stages read so far, by section, while a config is translated."""

    def __init__(self, config: Mapping, data_dirs: Sequence[Path]):
        self.config = config
        self.data_dirs = list(data_dirs)
        self.sources: dict[str, Stage] = {}
        self.source_groups: dict[str, list[str]] = {}
        self.fit_groups: dict[str, list[str]] = {}
        self.blind: dict[str, bool] = {}
        self.fit_stages: dict[str, Fit] = {}
        #: each LCFIT task's fits of one source, by Pippin's name for them
        self.fits: dict[str, Fit | FitVersion] = {}
        self.fit_sources: dict[str, str] = {}
        self.classifications: dict[str, dict] = {}
        self.classified: dict[tuple[str, str], Classify] = {}
        self.trained: dict[str, Train] = {}
        self.columns: dict[str, str] = {}
        self.unsupported: dict[str, str] = {}
        self.bbcs: dict[str, BBC] = {}
        self.covariances: dict[str, Covariance] = {}
        self.cosmologies: dict[str, Cosmology] = {}
        #: the scale each systematic declares where it is defined, the FITOPTS
        #: file's scale column or a MUOPT's SCALE, which the covariance takes
        self.label_scales: dict[str, float] = {}

    def stages(self) -> list[Stage]:
        return [*self.sources.values(), *self.fit_stages.values(),
                *self.trained.values(),
                *self.classified.values(), *self.bbcs.values(),
                *self.covariances.values(), *self.cosmologies.values()]

    def _section(self, key: str) -> dict:
        value = self.config.get(key)
        if value is not None and not isinstance(value, Mapping):
            _fail(key, "config", "expected a mapping of named tasks")
        return value or {}

    def _record_scale(self, section: str, name: str, label: str, scale: float):
        if self.label_scales.get(label, scale) != scale:
            _fail(section, name, f"scales {label} differently from where it was first declared")
        self.label_scales[label] = scale

    def read_dataprep(self):
        for name, task in self._section("DATAPREP").items():
            _validate_keys("DATAPREP", name, task or {}, {"OPTS", "EXTERNAL"})
            opts = (task or {}).get("OPTS") or {}
            if "EXTERNAL" in (task or {}):
                _fail("DATAPREP", name, "EXTERNAL is not supported; name the version directory as RAW_DIR")
            _validate_keys("DATAPREP OPTS", name, opts, {"RAW_DIR", "TYPES", "BLIND", "SIM"}, batch=True)
            if "RAW_DIR" not in opts:
                _fail("DATAPREP", name, "OPTS requires RAW_DIR")
            for key in ("TYPES", "SIM"):
                if key in opts:
                    logger.info(f"DATAPREP {name}: {key} is metadata; source tables are read unchanged")
            self.sources[name] = Data(name, _locate(opts["RAW_DIR"], self.data_dirs, "DATAPREP", name))
            self.blind[name] = bool(opts.get("BLIND", True))

    def read_sim(self):
        for group, task in self._section("SIM").items():
            if not isinstance(task, Mapping):
                _fail("SIM", group, "expected components and GLOBAL settings")
            _validate_keys("SIM OPTS", group, task.get("OPTS") or {}, {"BLIND"}, batch=True)
            if "EXTERNAL" in task:
                _validate_keys("SIM", group, task, {"EXTERNAL", "OPTS"})
                directories = _external_versions(_locate(task["EXTERNAL"], self.data_dirs, "SIM", group), group)
                versions = [group] if len(directories) == 1 else [f"{group}-{i:04d}" for i in sorted(directories)]
                if len(versions) > 1:
                    self.source_groups[group] = versions
                for version, index in zip(versions, sorted(directories)):
                    if version in self.sources:
                        _fail("SIM", group, f"source name {version} is already used")
                    self.sources[version] = Data(version, directories[index])
                    self.blind[version] = bool((task.get("OPTS") or {}).get("BLIND", False))
                continue
            everyone = dict(task.get("GLOBAL") or {})
            for key in (*_SIM_BATCH_KEYS, *_BATCH_KEYS):
                if key in everyone:
                    logger.info(f"SIM {group}: {key} is a batch setting, ignored")
                    del everyone[key]
            changing = "RANSEED_CHANGE" in everyone
            if changing and "RANSEED_REPEAT" in everyone:
                _fail("SIM", group, "use RANSEED_CHANGE or RANSEED_REPEAT, not both")
            repeat = everyone.pop("RANSEED_CHANGE" if changing else "RANSEED_REPEAT",
                                  _DEFAULT_RANSEED_REPEAT)
            try:
                jobs, seed = (int(x) for x in str(repeat).split())
            except ValueError:
                _fail("SIM", group, "RANSEED_CHANGE/REPEAT needs 'count seed'")
            if jobs < 1 or seed < 1:
                _fail("SIM", group, "RANSEED_CHANGE/REPEAT count and seed must be positive")
            format_mask = int(everyone.pop("FORMAT_MASK", _DEFAULT_FORMAT_MASK))
            if "NGEN_UNIT" in everyone:
                # SNANA's batch layer takes the count from the rate and drops
                # NGENTOT_LC (submit_prog_sim.py:28); the two are exclusive here
                everyone.update({"NGENTOT_LC": 0, "NGEN_LC": 0})
            variants = [f"{group}-{i+1:04d}" for i in range(jobs)] if changing else [group]
            if changing:
                self.source_groups[group] = variants
            for version_index, version in enumerate(variants):
                if version in self.sources or (changing and version in self._section("SIM")):
                    _fail("SIM", group, f"generated version name {version} is already used")
                self.blind[version] = bool((task.get("OPTS") or {}).get("BLIND", False))
                files, overrides = [], []
                for component, keys in task.items():
                    if component in ("GLOBAL", "OPTS"):
                        continue
                    if not isinstance(keys, Mapping) or "BASE" not in keys:
                        _fail("SIM", group, f"component {component} needs a BASE input")
                    keys = dict(keys)
                    base = _locate(keys.pop("BASE"), self.data_dirs, "SIM", group)
                    if changing and ("RANSEED" in keys or "RANSEED" in everyone):
                        _fail("SIM", group, "RANSEED would override RANSEED_CHANGE; set its seed in RANSEED_CHANGE")
                    for job in ([version_index] if changing else range(jobs)):
                        files.append(base)
                        job_seed = seed + 10000*job + job*job + 13 if changing else _seed(seed, job, format_mask)
                        overrides.append({"RANSEED": job_seed, "FORMAT_MASK": format_mask,
                                          **{k: _value(v) for k, v in keys.items()},
                                          **{k: _value(v) for k, v in everyone.items()}})
                self.sources[version] = Sim(version, files, overrides=overrides)

    def read_lcfit(self):
        for name, task in self._section("LCFIT").items():
            _validate_keys("LCFIT", name, task, {"BASE", "MASK", "OPTS", "EXTERNAL_DIRS", "FITOPTS", "SNLCINP", "FITINP"})
            _validate_keys("LCFIT OPTS", name, task.get("OPTS") or {}, set(), batch=True)
            overrides = {k: _value(v) for section in ("SNLCINP", "FITINP")
                         for k, v in (task.get(section) or {}).items()}
            matched = [s for s in self.sources if _matches(task.get("MASK"), s)]
            if not matched:
                _fail("LCFIT", name, f"mask {task.get('MASK')!r} matches no SIM or DATAPREP")
            external = self._external_fit_map(name, task.get("EXTERNAL_DIRS") or [], matched)
            fitted = [source for source in matched if f"{name}_{source}" not in external]
            if fitted and "BASE" not in task:
                _fail("LCFIT", name, "BASE is required for sources without external fits")
            needs_fitopts = matched if task.get("FITOPTS") else fitted
            if needs_fitopts and "BASE" not in task:
                _fail("LCFIT", name, "BASE is required to interpret FITOPTS")
            base = _locate(task["BASE"], self.data_dirs, "LCFIT", name) if needs_fitopts else None
            fitopts = {source: self._fitopts(name, task.get("FITOPTS") or [],
                                             self.sources[source], overrides)
                       for source in needs_fitopts}
            # one Fit per FITOPT set, over every source it fits, so they
            # share its compiled shapes
            fitted = [s for s in matched if f"{name}_{s}" not in external]
            groups = group_equal({s: fitopts[s] for s in fitted})
            fit_of = {}
            for i, group in enumerate(groups):
                suffix = f"_{i}" if len(groups) > 1 else ""
                fit = Fit(f"LCFIT_{name}{suffix}", base,
                          datasets=[self.sources[s] for s in group],
                          overrides=dict(overrides),
                          fitopts=fitopts[group[0]])
                self.fit_stages[fit.name] = fit
                fit_of.update(dict.fromkeys(group, fit))
            for source in matched:
                stage_name = f"{name}_{source}"
                self.fit_sources[stage_name] = source
                if stage_name in external:
                    external_fit = Fit(stage_name, fitres=external[stage_name])
                    self.fit_stages[stage_name] = self.fits[stage_name] = external_fit
                    continue
                self.fits[stage_name] = fit_of[source].version(self.sources[source])

            for group, versions in self.source_groups.items():
                names = [f"{name}_{version}" for version in versions if version in matched]
                # The group alias denotes all realisations, in their original order.
                # A mask selecting only some versions must name those fits explicitly.
                if len(names) == len(versions):
                    self.fit_groups[f"{name}_{group}"] = names

    def _external_fit_map(self, name, entries, matched):
        result = {}
        expected = {f"{name}_{source}" for source in matched}
        for entry in _as_list(entries):
            directory = _locate(str(entry), self.data_dirs, "LCFIT", name)
            alias = directory.name
            if alias in expected:
                paths = {alias: _external_fits(directory, name)}
            else:
                groups = [versions for group, versions in self.source_groups.items()
                          if alias == f"{name}_{group}"]
                if not groups:
                    _fail("LCFIT", name, f"EXTERNAL_DIRS entry {directory} matches no selected source")
                versions = _numbered_versions(_fit_versions(directory), "LCFIT", name)
                wanted = {int(source.rsplit("-", 1)[1]): f"{name}_{source}"
                          for source in groups[0] if f"{name}_{source}" in expected}
                if not wanted or set(wanted) - set(versions):
                    _fail("LCFIT", name, f"{directory} does not contain every selected realisation")
                paths = {target: versions[index] for index, target in wanted.items()}
            overlap = set(result) & set(paths)
            if overlap:
                _fail("LCFIT", name, f"duplicate external fits for {sorted(overlap)}")
            result.update(paths)
        return result

    def _fitopts(self, name, asked, source, overrides) -> dict[str, dict]:
        """Pippin's FITOPTS: one yml of per-survey blocks, and inline
        ``/label/ KEY value`` entries, each command split into keys as the
        fitter itself splits a command line. The file's scale column is
        recorded for the covariance."""
        asked = [asked] if isinstance(asked, str) else list(asked)
        known = known_keys()
        entries: list[tuple[str, float, str]] = []
        for item in asked:
            if self._is_file(item):
                path = _locate(item, self.data_dirs, "LCFIT", name)
                blocks = yaml.safe_load(path.read_text())
                if "FLAG_USE_SAME_EVENTS" in blocks:
                    overrides["OPT_SNCID_LIST"] = _value(blocks.pop("FLAG_USE_SAME_EVENTS"))
                chosen = {k: v for k, v in blocks.items() if k == "GLOBAL"}
                if len(blocks) > len(chosen):
                    survey = _survey(source)
                    chosen.update({k: v for k, v in blocks.items() if k == survey})
                for block in chosen.values():
                    for label, scale_command in block.items():
                        scale, command = str(scale_command).split(maxsplit=1)
                        entries.append((label, float(scale), command))
            elif item.startswith("/"):
                _, label, command = item.split("/", 2)
                entries.append((label, 1.0, command))
            else:
                _fail("LCFIT", name, f"FITOPTS entry {item!r} is neither a file nor /label/ KEY value")
        fitopts = {}
        for label, scale, command in entries:
            groups = group_override_tokens(shlex.split(command), known)
            fitopts[label] = {k: " ".join(v) for k, v in groups}
            self._record_scale("LCFIT", name, label, scale)
        return fitopts

    def _is_file(self, text: str) -> bool:
        """Whether a FITOPTS entry names a file, by Pippin's path rule, before
        an inline entry is assumed."""
        if "$" in text or text.startswith("/"):
            return Path(os.path.expandvars(text)).is_file()
        return any((Path(d) / text).is_file() for d in self.data_dirs)

    def read_classification(self):
        # a predicting task names its training task, so those are read first
        declared = self._section("CLASSIFICATION")
        for name, task in declared.items():
            if not isinstance(task, Mapping) or not {"CLASSIFIER", "MODE"} <= set(task):
                _fail("CLASSIFICATION", name, "task requires CLASSIFIER and MODE")
        tasks = sorted(declared.items(), key=lambda item: item[1]["MODE"] != "train")
        for name, task in tasks:
            self.classifications[name] = task
            if task["CLASSIFIER"] not in ("PerfectClassifier", "SuperNNovaClassifier"):
                self.unsupported[name] = task["CLASSIFIER"]
                logger.info(f"CLASSIFICATION {name}: {task['CLASSIFIER']} has no stage here")
                continue
            unknown = sorted(set(task) - set(_CLASSIFICATION_KEYS))
            if unknown:
                _fail("CLASSIFICATION", name, f"{', '.join(unknown)} is not supported")
            for key in ("GPU", "CLEAN"):
                if key in task:
                    logger.info(f"CLASSIFICATION {name}: {key} is a batch setting, ignored")
            if task["MODE"] not in ("train", "predict"):
                _fail("CLASSIFICATION", name, f"MODE must be train or predict, not {task['MODE']}")
            external = self._external_dirs(name, task)
            if task["CLASSIFIER"] == "SuperNNovaClassifier":
                self._supernnova(name, task, external)
                continue
            opts = task.get("OPTS") or {}
            unknown = sorted(set(opts) - set(_PERFECT_OPTS))
            if unknown:
                _fail("CLASSIFICATION", name, f"option {', '.join(unknown)} is not supported")
            column = opts.get("PROB_COLUMN_NAME", f"PROB_{name}")
            # Perfect's fields hold the default probabilities
            stated = {}
            if "PROB_IA" in opts:
                stated["prob_ia"] = float(opts["PROB_IA"])
            if "PROB_CC" in opts:
                stated["prob_cc"] = float(opts["PROB_CC"])
            classifier = Perfect(column=column, **stated)
            self.columns[name] = column
            for fit_name, fit in self.fits.items():
                source = self.fit_sources[fit_name]
                if not (_matches(task.get("MASK"), fit_name) or _matches(task.get("MASK"), source)):
                    continue
                if not (_matches(task.get("MASK_FIT"), fit_name) and _matches(task.get("MASK_SIM"), source)):
                    continue
                stage_name = f"{name}_{fit_name}"
                if task["MODE"] == "train":
                    self.trained[stage_name] = Train(stage_name, classifier, fit=fit)
                    continue
                model, suffix = self._named_model(opts.get("MODEL"), fit_name)
                directory = self._external_dir(name, external, f"{name}_{fit_name}{suffix}")
                self.classified[(name, fit_name)] = (
                    Classify(stage_name, fit=fit, predictions=self._predictions(name, directory))
                    if directory else
                    Classify(stage_name, classifier, fit=fit, model=self._model(name, model)))

    def _supernnova(self, name, task, external):
        """A SuperNNova task: trained on, or predicting for, each simulation or
        data version its masks match, as Pippin's photometry-only classifiers
        are. A prediction is joined onto every fit of its version, as
        Pippin's MERGE does."""
        opts = dict(task.get("OPTS") or {})
        for key in ("BATCH_FILE", "BATCH_REPLACE"):
            if opts.pop(key, None) is not None:
                logger.info(f"CLASSIFICATION {name}: {key} is a batch setting, ignored")
        unknown = sorted(set(opts) - set(_SNN_OPTS))
        if unknown:
            _fail("CLASSIFICATION", name, f"SuperNNova option {', '.join(unknown)} is not supported")
        model = opts.get("MODEL")
        if task["MODE"] == "predict" and model is None:
            _fail("CLASSIFICATION", name, "predicting needs a MODEL: a training task or a model file")
        if "PROB_COLUMN_NAME" in opts:
            column = opts["PROB_COLUMN_NAME"]
        elif model in self.columns:
            column = self.columns[model]      # the training task's column
        elif model is not None:
            column = f"PROB_{Path(str(model)).parent.name}"
        else:
            column = f"PROB_{name}"
        self.columns[name] = column
        # Pippin's REDSHIFT is True, False or the label itself
        redshift = opts.get("REDSHIFT", "zspe")
        redshift = {True: "zspe", False: "none"}.get(redshift, redshift)
        sntypes = opts.get("SNTYPES")
        if isinstance(sntypes, str):
            path = _locate(sntypes, self.data_dirs, "CLASSIFICATION", name)
            # Pippin reads the file with numpy's loadtxt: a pair per line, # comments
            sntypes = dict(line.split()[:2] for line in Path(path).read_text().splitlines()
                           if line.strip() and not line.lstrip().startswith("#"))
        classifier = SuperNNova(
            filters=tuple(opts.get("LIST_FILTERS", _SNN_FILTERS)), column=column,
            variant=str(opts.get("VARIANT", "vanilla")).lower(), redshift=redshift,
            norm=opts.get("NORM", "cosmo"), cyclic=bool(opts.get("CYCLIC", True)),
            seed=int(opts.get("SEED", 0)), batch_size=int(opts.get("BATCH_SIZE", 128)),
            num_layers=int(opts.get("NUM_LAYERS", 2)),
            hidden_dim=int(opts.get("HIDDEN_DIM", 32)),
            sntypes=None if sntypes is None else {int(k): v for k, v in sntypes.items()})
        for source_name, source in self.sources.items():
            if not (_matches(task.get("MASK"), source_name)
                    and _matches(task.get("MASK_SIM"), source_name)):
                continue
            if task["MODE"] == "train":
                stage_name = f"{name}_{source_name}"
                self.trained[stage_name] = Train(stage_name, classifier, photometry=source)
                continue
            named, suffix = self._named_model(model, source_name)
            directory = self._external_dir(name, external, f"{name}_{source_name}{suffix}")
            for fit_name, fit in self.fits.items():
                if self.fit_sources[fit_name] != source_name:
                    continue
                stage_name = f"{name}_{fit_name}"
                self.classified[(name, fit_name)] = (
                    Classify(stage_name, fit=fit, predictions=self._predictions(name, directory))
                    if directory else
                    Classify(stage_name, classifier, fit=fit, model=self._model(name, named),
                             photometry=source))

    def _external_dirs(self, name, task) -> dict[str, str]:
        """A task's EXTERNAL_DIRS by directory name: the same task's outputs
        from an earlier run, each used where its name is the one Pippin gives
        the task here (task.py). Predictions only; a model is reused through
        MODEL."""
        dirs = task.get("EXTERNAL_DIRS") or []
        if dirs and task["MODE"] == "train":
            _fail("CLASSIFICATION", name, "EXTERNAL_DIRS on a training task is not "
                                          "supported; name the model with MODEL")
        return {Path(os.path.expandvars(d)).name: d for d in dirs}

    def _external_dir(self, name, external, task_name):
        """The external directory named as Pippin names this task, or None,
        logged as Pippin logs it, when the task is to run here."""
        if external and task_name not in external:
            logger.info(f"CLASSIFICATION {name}: no external match found for {task_name}")
        return external.get(task_name)

    def _predictions(self, name, directory) -> Path:
        return _locate(directory, self.data_dirs, "CLASSIFICATION", name) / "predictions.csv"

    def _named_model(self, model, unit):
        """The model a predicting task uses, a training stage or the path as
        given, and the suffix Pippin adds to the task's name for it
        (classifier.py, get_tasks): the training task's name, or the name of
        the directory holding the model file."""
        if model is None:
            return None, ""
        trained = self.trained.get(f"{model}_{unit}")
        by_task = [t for n, t in self.trained.items() if n.startswith(f"{model}_")]
        if trained is None and len(by_task) == 1:
            trained = by_task[0]
        if trained is not None:
            return trained, f"_{trained.name}"
        return model, f"_{Path(str(model)).parent.name}"

    def _model(self, name, model):
        """A training stage as it is, a model file located."""
        if model is None or isinstance(model, Stage):
            return model
        return _locate(str(model), self.data_dirs, "CLASSIFICATION", name)

    def _tables(self, section, name, names, classifier) -> list[Stage]:
        """The fit stages Pippin's task names resolve to, classified where
        the named classifier classified them, as Pippin's MERGE would."""
        stages = []
        for fit_name in ([names] if isinstance(names, str) else names):
            if fit_name not in self.fits:
                _fail(section, name, f"{fit_name} names no LCFIT task; the fits are {sorted(self.fits)}")
            if classifier in self.unsupported:
                _fail(section, name, f"classifier {classifier} is {self.unsupported[classifier]}, "
                                     f"which has no stage here")
            stages.append(self.classified.get((classifier, fit_name), self.fits[fit_name]))
        return stages

    def _column(self, section, name, classifier):
        if classifier is None:
            return None
        if classifier in self.unsupported:
            _fail(section, name, f"classifier {classifier} is {self.unsupported[classifier]}, "
                                 f"which has no stage here")
        if classifier in self.classifications:
            return self.columns[classifier]
        return classifier   # Pippin also takes the column's own name

    def _correction_fitopts(self, name, config, role, classifier):
        key = f"SIMFILE_{role.upper()}"
        sources = self._tables("BIASCOR", name, config.get(key, []), classifier)
        numbers = _as_list(config.get(f"{key}_FITOPTS"))
        if numbers and len(numbers) != len(sources):
            _fail("BIASCOR", name, f"{key}_FITOPTS must have one entry per {key} source")
        labels = []
        for source, number in zip(sources, numbers):
            try:
                index = int(number)
            except (TypeError, ValueError):
                _fail("BIASCOR", name, f"invalid {key}_FITOPTS index {number!r}")
            if isinstance(number, bool) or str(index) != str(number) or not 0 <= index <= len(source.systematics()[0]):
                _fail("BIASCOR", name, f"{source.name} has no FITOPT index {number!r}")
            labels.append(None if index == 0 else source.systematics()[0][index - 1])
        return labels

    def _bbc_tasks(self):
        """Expand matched realisations and disjoint data partitions separately."""
        source_keys = ("DATA", "SIMFILE_BIASCOR", "SIMFILE_CCPRIOR")
        for name, task in self._section("BIASCOR").items():
            _validate_keys("BIASCOR", name, task, {
                "BASE", "DATA", "OPTS", "MUOPTS", "CLASSIFIER", "PROB_COLUMN_NAME", "BLIND", "USE_RECALIBRATED",
                "SIMFILE_BIASCOR", "SIMFILE_BIASCOR_FITOPTS", "SIMFILE_CCPRIOR", "SIMFILE_CCPRIOR_FITOPTS"})
            for label, variant in (task.get("MUOPTS") or {}).items():
                _validate_keys("BIASCOR MUOPT", f"{name}/{label}", variant or {}, {
                    "OPTS", "CLASSIFIER", "SIMFILE_BIASCOR", "SIMFILE_BIASCOR_FITOPTS",
                    "SIMFILE_CCPRIOR", "SIMFILE_CCPRIOR_FITOPTS", "SCALE", "FITOPT"})
            configs = [task, *((task.get("MUOPTS") or {}).values())]
            aliases = {ref for config in configs if config for key in source_keys
                       for ref in _as_list(config.get(key)) if ref in self.fit_groups}
            counts = {len(self.fit_groups[ref]) for ref in aliases}
            if len(counts) > 1:
                _fail("BIASCOR", name, "RANSEED_CHANGE sources must have the same number of realisations")
            count = next(iter(counts), 1)
            opts = task.get("OPTS") or {}
            split = opts.get("NSPLITRAN", 1)
            if type(split) is not int or split < 1:
                _fail("BIASCOR", name, "NSPLITRAN must be a positive integer")
            for index in range(count):
                resolved = copy.deepcopy(task)
                resolved.setdefault("OPTS", {})
                if resolved["OPTS"] is None:
                    resolved["OPTS"] = {}
                resolved["OPTS"].pop("NSPLITRAN", None)
                for config in [resolved, *((resolved.get("MUOPTS") or {}).values())]:
                    if not config:
                        continue
                    for key in source_keys:
                        if key in config:
                            config[key] = [self.fit_groups[ref][index] if ref in self.fit_groups else ref
                                           for ref in _as_list(config[key])]
                version = f"{name}-{index+1:04d}" if aliases else name
                for partition in range(1, split + 1):
                    stage_name = f"{version}-SPLIT{partition:03d}" if split > 1 else version
                    yield stage_name, resolved, split, partition, index + 1 if aliases else None

    def read_biascor(self):
        for name, task, split_count, split_index, realisation in self._bbc_tasks():
            if name in self.bbcs:
                _fail("BIASCOR", name, "generated task name is already used")
            base = _locate(task.get("BASE", _DEFAULT_BBC), self.data_dirs, "BIASCOR", name)
            classifier = task.get("CLASSIFIER")
            column = task.get("PROB_COLUMN_NAME") or self._column("BIASCOR", name, classifier)
            recalibrated = task.get("USE_RECALIBRATED", False)
            if type(recalibrated) is not bool:
                _fail("BIASCOR", name, "USE_RECALIBRATED must be true or false")
            if recalibrated:
                if column is None:
                    _fail("BIASCOR", name, "USE_RECALIBRATED needs CLASSIFIER or PROB_COLUMN_NAME")
                column = _recalibrated_column(column)
            data = self._tables("BIASCOR", name, task["DATA"], classifier)
            overrides = self._bbc_overrides("BIASCOR", name, task.get("OPTS") or {}, column)
            if column is not None:
                overrides["varname_pIa"] = column
            blind = task.get("BLIND", "blindflag" not in overrides and any(
                self.blind[self.fit_sources[f]] for f in _as_list(task["DATA"])))
            if "blindflag" not in overrides:
                overrides["blindflag"] = 2 if blind else 0
            muopts = {}
            for label, variant in (task.get("MUOPTS") or {}).items():
                variant = variant or {}
                if "FITOPT" in variant:
                    _fail("BIASCOR", name, f"MUOPT {label} names a FITOPT, which is not supported")
                if {"FLAG_USE_SAME_EVENTS", "OPT_SNCID_LIST"} & (variant.get("OPTS") or {}).keys():
                    _fail("BIASCOR", name, "event synchronisation belongs in task OPTS, not a MUOPT")
                swap = variant.get("CLASSIFIER")
                swapped_column = self._column("BIASCOR", name, swap)
                if recalibrated and swapped_column is not None:
                    swapped_column = _recalibrated_column(swapped_column)
                if "FITOPTxMUOPT" in (variant.get("OPTS") or {}):
                    _fail("BIASCOR", name, "FITOPTxMUOPT belongs in task OPTS, not a MUOPT")
                if any(k.upper().startswith("REPLACE_SIMFILE") for k in (variant.get("OPTS") or {})):
                    _fail("BIASCOR", name, "REPLACE_SIMFILE options belong in the task OPTS, not a MUOPT")
                keys = self._bbc_overrides("BIASCOR", name, variant.get("OPTS") or {},
                                           swapped_column or column)
                # a MUOPT naming tasks under SALT2mu's own keys consumes their
                # stages, as Pippin's does
                if swap is not None:
                    keys["varname_pIa"] = swapped_column
                    keys["datafile"] = self._tables("BIASCOR", name, task["DATA"], swap)
                if "SIMFILE_BIASCOR" in variant:
                    keys["simfile_biascor"] = self._tables(
                        "BIASCOR", name, variant["SIMFILE_BIASCOR"], classifier)
                if "SIMFILE_CCPRIOR" in variant:
                    keys["simfile_ccprior"] = self._tables(
                        "BIASCOR", name, variant["SIMFILE_CCPRIOR"], classifier)
                if "SCALE" in variant:
                    self._record_scale("BIASCOR", name, label, float(variant["SCALE"]))
                muopts[label] = keys
            correction_options = {}
            for role, key in (("biascor", "SIMFILE_BIASCOR"), ("ccprior", "SIMFILE_CCPRIOR")):
                correction_options[f"{role}_fitopts"] = self._correction_fitopts(name, task, role, classifier)
                correction_options[f"muopt_{role}_fitopts"] = {
                    label: self._correction_fitopts(name, variant, role, classifier)
                    for label, variant in (task.get("MUOPTS") or {}).items()
                    if variant and f"{key}_FITOPTS" in variant}
                replacements = (task.get("OPTS") or {}).get(f"REPLACE_{key}") or {}
                if not isinstance(replacements, Mapping):
                    _fail("BIASCOR", name, f"REPLACE_{key} must map FITOPT labels to files")
                correction_options[f"replace_{role}"] = {
                    label: _locate(str(path), self.data_dirs, "BIASCOR", name)
                    for label, path in replacements.items()}
            self.bbcs[name] = BBC(
                name, data=data, input_file=base,
                biascor=(None if "SIMFILE_BIASCOR" not in task else
                         self._tables("BIASCOR", name, task["SIMFILE_BIASCOR"], classifier)),
                ccprior=(None if "SIMFILE_CCPRIOR" not in task else
                         self._tables("BIASCOR", name, task["SIMFILE_CCPRIOR"], classifier)),
                overrides=overrides, muopts=muopts, **correction_options,
                split_count=split_count, split_index=split_index, realisation=realisation,
                require_probability_column=recalibrated,
                sync_events=sync_setting(task.get("OPTS") or {}),
                fitopt_muopt=_as_list((task.get("OPTS") or {}).get("FITOPTxMUOPT")))
            self.bbcs[name].grid_cells()

    def _bbc_overrides(self, section, name, opts, column) -> dict:
        """Pippin's BIASCOR OPTS as SALT2mu keys: ``CUTWIN_X`` is the cut
        ``CUTWIN X``, ``CUTWIN_PROB_IA`` the cut on the classifier's column."""
        overrides = {}
        for key, value in opts.items():
            if key in _BATCH_KEYS:
                logger.info(f"{section} {name}: {key} is a batch setting, ignored")
            elif key.upper().startswith("BATCH"):
                _fail(section, name, f"unknown batch setting {key}")
            elif key in ("REPLACE_SIMFILE_BIASCOR", "REPLACE_SIMFILE_CCPRIOR", "FITOPTxMUOPT",
                         "FLAG_USE_SAME_EVENTS", "OPT_SNCID_LIST"):
                continue
            elif key.upper() in ("NSPLITRAN",) or key.upper().startswith("REPLACE_SIMFILE"):
                _fail(section, name, f"{key} is not supported")
            elif key.upper().startswith("CUTWIN"):
                if not key.upper().startswith("CUTWIN_") or not key.split("_", 1)[1]:
                    _fail(section, name, f"{key} must be CUTWIN_<column>")
                variable = key.split("_", 1)[1]
                if variable.upper() == "PROB_IA":
                    if column is None:
                        _fail(section, name, "CUTWIN_PROB_IA needs a CLASSIFIER")
                    variable = column
                overrides[f"CUTWIN {variable}"] = _value(value)
            else:
                overrides[key] = _value(value)
        return overrides

    def read_create_cov(self):
        supported = {"COVOPTS", "BINNED", "SYS_SCALE", "FITOPT_SCALES",
                     "MUOPT_SCALES", "EXTRA_COVS", "CALIBRATORS", "REBINNED_X1",
                     "REBINNED_C", "REBIN_BINS", "SYSTEMATIC_HD", "SUBTRACT_VPEC", "EXTRA_OPTS"}
        for name, task in self._section("CREATE_COV").items():
            task = task or {}
            unknown = set(task) - {"MASK", "MASK_BIASCOR", "OPTS"}
            if unknown:
                _fail("CREATE_COV", name, f"unsupported task key(s): {', '.join(sorted(unknown))}")
            opts = task.get("OPTS") or {}
            _validate_keys("CREATE_COV OPTS", name, opts, supported | {"INI_DIR", "COSMOMC"}, batch=True)
            for key in ("INI_DIR", "COSMOMC"):
                if key in opts:
                    logger.info(f"CREATE_COV {name}: {key} is ignored")
            extra_options = _cov_extra_options(name, opts.get("EXTRA_OPTS", ""))
            mask = task.get("MASK", task.get("MASK_BIASCOR"))
            matched = [b for b in self.bbcs if _matches(mask, b)]
            if not matched:
                _fail("CREATE_COV", name, f"mask {mask!r} matches no BIASCOR")
            bins = opts.get("REBIN_BINS")
            legacy_bins = [opts.get("REBINNED_X1", 0), opts.get("REBINNED_C", 0)]
            if any(legacy_bins):
                if bins is not None:
                    _fail("CREATE_COV", name, "use REBIN_BINS or REBINNED_X1/REBINNED_C, not both")
                if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0
                       for n in legacy_bins):
                    _fail("CREATE_COV", name, "REBINNED_X1 and REBINNED_C must both be positive integers")
            if bins is not None:
                bins = resolve_rebin_bins(bins)
            mode = ("rebin" if bins is not None or any(legacy_bins) else
                    "unbinned" if opts.get("SUBTRACT_VPEC") or not opts.get("BINNED", True) else "binned")
            extra = []
            for entry in opts.get("EXTRA_COVS", []) or []:
                words = str(entry).split(maxsplit=2)
                if len(words) != 3:
                    _fail("CREATE_COV", name, "EXTRA_COVS needs 'label scale file' entries")
                try:
                    scale = float(words[1])
                    if not math.isfinite(scale):
                        raise ValueError
                except ValueError:
                    _fail("CREATE_COV", name, f"invalid EXTRA_COVS scale {words[1]!r}")
                extra.append((words[0], scale,
                              _locate(words[2], self.data_dirs, "CREATE_COV", name)))
            for bbc in matched:
                stage_name = name if len(matched) == 1 else f"{name}_{bbc}"
                self.covariances[stage_name] = Covariance(
                    stage_name, distances=self.bbcs[bbc], mode=mode, rebin_bins=bins,
                    **extra_options,
                    nbin_x1=legacy_bins[0], nbin_c=legacy_bins[1],
                    extra_covs=extra, calibrators=_as_list(opts.get("CALIBRATORS")),
                    systematic_hd=bool(opts.get("SYSTEMATIC_HD", False)),
                    subtract_vpec=bool(opts.get("SUBTRACT_VPEC", False)),
                    covopts=list(opts.get("COVOPTS") or []),
                    scales=self._cov_scales(name, opts, self.bbcs[bbc]))
                self.covariances[stage_name].configuration()

    def _cov_scales(self, name: str, opts: Mapping, bbc: BBC) -> dict[str, float]:
        """The scale of each of one distance fit's systematics: what it
        declared where it was defined, then CREATE_COV's SYS_SCALE file,
        FITOPT_SCALES (matched by substring, as Pippin's docs say) and
        MUOPT_SCALES (matched exactly) on top."""
        fitopts, muopts = bbc.systematics()
        labels = set(fitopts) | set(muopts)
        scales = {label: scale for label, scale in self.label_scales.items() if label in labels}
        if "SYS_SCALE" in opts:
            path = _locate(opts["SYS_SCALE"], self.data_dirs, "CREATE_COV", name)
            scales.update({label: scale for label, scale in
                           _file_scales(yaml.safe_load(path.read_text())).items()
                           if label in labels})
        for pattern, scale in (opts.get("FITOPT_SCALES") or {}).items():
            matched = [label for label in fitopts if str(pattern) in label]
            if not matched:
                _fail("CREATE_COV", name, f"FITOPT_SCALES {pattern} matches no FITOPT of {bbc.name}")
            scales.update({label: float(scale) for label in matched})
        for label, scale in (opts.get("MUOPT_SCALES") or {}).items():
            if label not in muopts:
                _fail("CREATE_COV", name, f"MUOPT_SCALES {label} names no MUOPT of {bbc.name}")
            scales[label] = float(scale)
        return scales

    def read_cosmofit(self):
        section = self._section("COSMOFIT")
        for fitter in section:
            if fitter == "COSMOMC":
                logger.info(f"COSMOFIT {fitter} has no stage here, skipped")
            elif fitter != "WFIT":
                _fail("COSMOFIT", fitter, "unknown cosmology backend")
        if "COSMOMC" in self.config:
            logger.info("COSMOMC has no stage here, skipped")
        wfit = section.get("WFIT") or {}
        if not isinstance(wfit, Mapping):
            _fail("COSMOFIT", "WFIT", "expected a mapping of named tasks")
        for name, task in wfit.items():
            _validate_keys("COSMOFIT WFIT", name, task, {"MASK", "OPTS"})
            opts = task.get("OPTS") or {}
            _validate_keys("COSMOFIT WFIT OPTS", name, opts, {"WFITOPTS", "WFITOPT_GLOBAL"}, batch=True)
            if "WFITOPTS" not in opts or not opts["WFITOPTS"]:
                _fail("COSMOFIT", name, "WFITOPTS must name at least one fit")
            matched = [c for c in self.covariances if _matches(task.get("MASK"), c)]
            if not matched:
                _fail("COSMOFIT", name, f"mask {task.get('MASK')!r} matches no CREATE_COV")
            for entry in _as_list(opts["WFITOPTS"]):
                if not isinstance(entry, str) or not entry.startswith("/") or entry.count("/") < 2:
                    _fail("COSMOFIT", name, "WFITOPTS entries must be /label/ flags")
                _, label, flags = entry.split("/", 2)
                if not label.strip():
                    _fail("COSMOFIT", name, "WFITOPTS labels must not be empty")
                flags = f"{flags} {opts.get('WFITOPT_GLOBAL', '')}"
                for covariance in matched:
                    stage_name = f"WFIT_{name}_{label}" + ("" if len(matched) == 1 else f"_{covariance}")
                    held = self.covariances[covariance]
                    self.cosmologies[stage_name] = Cosmology(
                        stage_name, covariance=held, overrides=_wfit_keys(flags, name),
                        covopts=("ALL", *(parse_covopt(c).label for c in held.covopts)))


def _recalibrated_column(column: str) -> str:
    return "C" + column if column.startswith("PROB_") else column


def _cov_extra_options(name, value) -> dict:
    """Parse the supported writer options without invoking a shell."""
    class Parser(argparse.ArgumentParser):
        def error(self, message):
            _fail("CREATE_COV", name, f"EXTRA_OPTS: {message}")

    if not isinstance(value, str):
        _fail("CREATE_COV", name, "EXTRA_OPTS must be a string of flags")
    parser = Parser(add_help=False, allow_abbrev=False)
    parser.add_argument("--write_mask_cov", type=int, default=8)
    parser.add_argument("--write_format_cov", choices=("text", "npz"), default="npz")
    parser.add_argument("--label_cov_rows", action="store_true")
    parser.add_argument("--mxsize_test_posdef", type=int, default=6000)
    try:
        args = parser.parse_args(shlex.split(value))
    except ValueError as error:
        _fail("CREATE_COV", name, f"EXTRA_OPTS: {error}")
    if not 0 <= args.write_mask_cov <= 15 or not args.write_mask_cov & 10:
        _fail("CREATE_COV", name, "--write_mask_cov must include inverse covariance (2) or factors (8), and use only bits 1, 2, 4 and 8")
    if args.mxsize_test_posdef < 0:
        _fail("CREATE_COV", name, "--mxsize_test_posdef must be non-negative")
    return dict(overrides={"WRITE_MASK_COV": args.write_mask_cov,
                           "WRITE_FORMAT_COV": args.write_format_cov},
                label_cov_rows=args.label_cov_rows,
                mxsize_test_posdef=args.mxsize_test_posdef)


def _as_list(value) -> list:
    if value is None:
        return []
    return [value] if isinstance(value, (str, int)) else list(value)


def _file_scales(blocks: Mapping) -> dict[str, float]:
    """The scale column of a FITOPTS file, every survey block together, as
    create_cov reads it for its scale table."""
    return {str(label): float(str(entry).split(maxsplit=1)[0])
            for block, entries in blocks.items() if block != "FLAG_USE_SAME_EVENTS"
            for label, entry in entries.items()}


def _numbered_versions(paths, section, name):
    result = {}
    for path in paths:
        match = re.search(r"-(\d+)$", path.name)
        if not match:
            _fail(section, name, f"version {path.name} needs a numbered realisation suffix, such as -0001")
        index = int(match.group(1))
        if index in result:
            _fail(section, name, f"duplicate realisation index {index}")
        result[index] = path
    if not result or set(result) != set(range(1, len(result) + 1)):
        _fail(section, name, "external realisations must be numbered consecutively from 1")
    return result


def _external_versions(directory: Path, group: str) -> dict[int, Path]:
    """All versions recorded by an external Pippin simulation task."""
    if list(directory.glob("*.LIST")):
        return {1: directory}
    recorded = directory / "config.yml"
    if not recorded.is_file():
        _fail("SIM", group, f"{directory} is neither a data version nor a Pippin task directory")
    config = yaml.safe_load(recorded.read_text()) or {}
    folders = (config.get("OUTPUT") or {}).get("sim_folders")
    if not isinstance(folders, list) or not folders:
        _fail("SIM", group, f"{recorded} needs OUTPUT.sim_folders")
    paths = []
    for folder in folders:
        path = Path(os.path.expandvars(str(folder)))
        if not path.is_absolute():
            path = directory / path
        if not path.is_dir() or not list(path.glob("*.LIST")):
            _fail("SIM", group, f"recorded version {path} has no simulation LIST file")
        paths.append(path)
    return {1: paths[0]} if len(paths) == 1 else _numbered_versions(paths, "SIM", group)


def _fit_versions(directory: Path) -> list[Path]:
    if list(directory.glob("FITOPT*.FITRES*")):
        return [directory]
    return sorted(path for path in (directory / "output").glob("*")
                  if path.is_dir() and list(path.glob("FITOPT*.FITRES*")))


def _external_fits(directory: Path, name: str) -> Path:
    versions = _fit_versions(directory)
    if len(versions) != 1:
        _fail("LCFIT", name, f"{directory} holds {len(versions)} fit versions; "
                             "use the simulation-group task directory for multiple realisations")
    return versions[0]


def _wfit_keys(flags: str, name: str) -> dict:
    """wfit.exe flags as the config keys the jax wfit takes, through its own
    command line, so a flag it refuses is refused here, while the config is
    read, with its reason."""
    parser = build_parser()
    try:
        args = parser.parse_args(shlex.split(flags))
    except SystemExit as exit_:
        raise SNANAError(f"COSMOFIT {name}: wfit does not take the flags {flags!r}",
                         function_name="pippin") from exit_
    return cli_keys(args)


def read(path, data_dirs: Sequence = ()) -> list[Stage]:
    """The stages a Pippin config describes.

    ``data_dirs`` are searched, in order, for the relative paths the config
    names, before any the config's own GLOBAL DATA_DIRS adds; Pippin's own
    ``data_files`` directory is the usual last entry.
    """
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, Mapping):
        raise SNANAError(f"{path} is not a mapping of sections", function_name="pippin")
    _validate_keys("GLOBAL", "config", config.get("GLOBAL") or {}, {"DATA_DIRS"}, batch=True)
    dirs = [Path(os.path.expandvars(str(d))) for d in data_dirs]
    dirs += [Path(os.path.expandvars(str(d)))
             for d in (config.get("GLOBAL") or {}).get("DATA_DIRS") or []]
    graph = _Graph(config, dirs)
    for section in config:
        if section not in ("GLOBAL", "DATAPREP", "SIM", "LCFIT", "CLASSIFICATION",
                           "AGGREGATION", "MERGE", "BIASCOR", "CREATE_COV", "COSMOFIT",
                           "COSMOMC", "ANALYSE"):
            _fail("CONFIG", str(section), "unknown Pippin section")
    for section in ("AGGREGATION", "MERGE", "ANALYSE"):
        if section in config:
            logger.info(f"{section} has no stage here, skipped")
    graph.read_dataprep()
    graph.read_sim()
    graph.read_lcfit()
    graph.read_classification()
    graph.read_biascor()
    graph.read_create_cov()
    graph.read_cosmofit()
    return graph.stages()


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
                        help="search directory for relative input paths; repeatable, searched in order")
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
    slog.startup(logger, "jax_pipeline",
                 sys.argv if argv is None else ["jax_pipeline"] + argv)
    configure_precision(args.double_precision)
    if args.check:
        from .check import check
        print(yaml.safe_dump(check(args.config, args.data_dir), sort_keys=False))
    else:
        run_pippin(args.config, args.outdir, args.data_dir,
                   refresh=True if args.refresh_all else args.refresh)
    slog.done(logger, "jax_pipeline")


if __name__ == "__main__":
    raise SystemExit(main())
