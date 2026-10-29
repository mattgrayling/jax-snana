"""Ordering a graph of stages, running it, and skipping what is already done."""

import dataclasses
import hashlib
import logging
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jax
import pandas as pd
import yaml

from jax_salt2mu.preparation import preparation_step
from jax_snlc_core.logging import SNANAError, version_string

from .classify import Classify, Train
from .stages import (BBC, Cosmology, Covariance, Data, Fit, FitVersion, Sim, Stage,
                     prepare_bbc_stages, run_bbc_stages)

logger = logging.getLogger(__name__)

#: what a finished stage leaves in its directory, written last
DONE = "DONE.yml"
#: the hash of the run a stage's directory was started for, written first
STARTED = "STARTED.yml"

#: the keys a SNANA input file uses to pull another file in
_INCLUDE_KEYS = ("INPUT_FILE_INCLUDE:", "INPUT_INCLUDE_FILE:")

# Stages within a phase keep their traversal order. Dependencies always take
# precedence.
_PHASES = ((Data, Sim), (Fit,), (Train, Classify),
           (BBC,), (Covariance,), (Cosmology,))


def _producer(used) -> Stage:
    """The stage that runs to supply ``used``: the fit a version's fits
    come from, or the stage itself."""
    return used.fit if isinstance(used, FitVersion) else used


def _result(used, produced: Mapping[Stage, Any]) -> Any:
    """What a stage consuming ``used`` receives: one version's tables from
    its fit's result, or the stage's own result."""
    if isinstance(used, FitVersion):
        return produced[used.fit][used.data.name]
    return produced[used]


def _reachable(terminals: Sequence[Stage]) -> list[Stage]:
    """Every stage the terminals depend on, each once, in dependency order.

    Depth-first from each terminal, so a stage comes after everything it
    consumes and the order is the same on every run.
    """
    ordered: list[Stage] = []
    done: set[int] = set()
    open_: set[int] = set()

    def visit(stage: Stage, trail: tuple[str, ...]) -> None:
        if not isinstance(stage, Stage):
            raise SNANAError(
                "expected a Stage; pass stages as run(a, b), or unpack a list with run(*stages)",
                function_name="run")
        if id(stage) in done:
            return
        if id(stage) in open_:
            raise SNANAError(
                f"the stages form a cycle: {' -> '.join(trail + (stage.name,))}",
                function_name="run")
        open_.add(id(stage))
        for used in stage.inputs().values():
            visit(_producer(used), trail + (stage.name,))
        open_.discard(id(stage))
        done.add(id(stage))
        ordered.append(stage)

    for terminal in terminals:
        visit(terminal, ())
    return ordered


def _clashing_names(stages: Sequence[Stage]) -> list[str]:
    """Names used by more than one stage, whose directories would collide."""
    counts: dict[str, int] = {}
    for stage in stages:
        counts[stage.name] = counts.get(stage.name, 0) + 1
    return sorted(name for name, count in counts.items() if count > 1)


def file_digest(path: Path, *, include_dir: Path | None = None) -> dict:
    """The sha256 of a file's bytes, with those of any file it includes as a
    SNANA input does; for a directory, relative names and file contents. A
    file that is not text, such as a trained model, includes nothing.
    Relative includes use the primary input directory unless the path already
    exists in the working directory, matching the simulation parser."""
    if path.is_dir():
        files = {}
        for child in sorted(path.rglob("*")):
            if child.is_file():
                with child.open("rb") as stream:
                    files[str(child.relative_to(path))] = hashlib.file_digest(
                        stream, "sha256").hexdigest()
        return {"directory": files}
    if not path.is_file():
        raise SNANAError(f"{path} does not exist", function_name="stage_hash")
    content = path.read_bytes()
    digest = {"file": hashlib.sha256(content).hexdigest()}
    try:
        text = content.decode()
    except UnicodeDecodeError:
        return digest
    # a SNANA input line ends at its first '#', as the parser reads it
    # (jax_snlc_sim/inputs.py, _strip_comment)
    words = [line.split("#", 1)[0].split() for line in text.splitlines()]
    included = [os.path.expandvars(token)
                for w in words if w[:1] and w[0] in _INCLUDE_KEYS
                for token in w[1:]]
    if included:
        include_dir = path.parent if include_dir is None else include_dir
        digest["includes"] = {}
        for name in included:
            child = Path(name)
            if not child.is_absolute() and not child.exists():
                child = include_dir / child
            digest["includes"][name] = file_digest(child, include_dir=include_dir)
    return digest


def _canonical(value: Any, hashes: Mapping[int, str]) -> Any:
    """A value as the hash sees it: stages by their hashes, paths by their
    contents, dataclasses by their fields."""
    if isinstance(value, Stage):
        return {"stage": hashes[id(value)]}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _fields(value, hashes)
    if isinstance(value, Path):
        return file_digest(value)
    if isinstance(value, Mapping):
        return {str(k): _canonical(v, hashes) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(v, hashes) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise SNANAError(f"cannot hash a {type(value).__name__}",
                     function_name="stage_hash")


def _fields(value: Any, hashes: Mapping[int, str]) -> dict:
    fields = {f.name: _canonical(getattr(value, f.name), hashes)
              for f in dataclasses.fields(value)}
    if isinstance(value, Stage):
        dependencies = value.cache_files()
        if dependencies:
            fields["file_dependencies"] = _canonical(dependencies, hashes)
    return {type(value).__name__: fields}



def stage_hash(described: Mapping[str, Any]) -> str:
    """The hash that decides whether a finished stage is loaded or rerun.

    The hash covers the stage's fields as ``_fields`` describes them (so
    every file they name and the hashes of the stages it consumes) and the
    precision the process runs in. It leaves out the code revision and the
    resources an input file names (SIMLIB, KCOR, models), as Pippin does."""
    return hashlib.sha256(yaml.safe_dump(
        {**described, "double_precision": _double_precision()},
        # Mapping order sets systematic numbering and override precedence.
        sort_keys=False).encode()).hexdigest()


def _double_precision() -> bool:
    return bool(jax.config.jax_enable_x64)


def validate_graph(stages: Sequence[Stage]) -> list[Stage]:
    """Validate the graph and return its execution order without running.

    Group built-in stages by processing phase, preserving dependency order
    within each phase. Custom stages follow their latest dependency's phase.
    """
    if not stages:
        raise SNANAError("an analysis needs at least one stage", function_name="run")
    ordered = _reachable(stages)
    for stage in ordered:
        if not dataclasses.is_dataclass(stage):
            raise SNANAError(f"stage '{stage.name}' is not a dataclass, which is "
                             f"what its hash reads", function_name="run")
    clashing = _clashing_names(ordered)
    if clashing:
        raise SNANAError(
            f"more than one stage is named {', '.join(clashing)}, so their "
            f"output directories would collide", function_name="run")
    for name in {stage.name for stage in ordered}:
        if not name or name in (".", "..") or Path(name).name != name:
            raise SNANAError(
                f"stage name {name!r} must be a single directory name",
                function_name="run")
    phases = {}
    for stage in ordered:
        phase = next((i for i, types in enumerate(_PHASES)
                      if isinstance(stage, types)), 0)
        phases[stage] = max([phase, *(phases[_producer(used)]
                                    for used in stage.inputs().values())])
    return sorted(ordered, key=phases.__getitem__)


def run(*stages: Stage, outdir: os.PathLike | str = ".", refresh=()) -> dict[str, Any]:
    """Run every stage the given ones depend on, in dependency order.

    Run data and simulations first, then light-curve preparation and fits,
    classification, BBC, covariance and cosmology. Dependencies take precedence
    over this grouping; stages within a phase retain their traversal order.

    Each stage writes its SNANA-format outputs under ``outdir/<name>`` and
    passes its result to the steps that use it. A stage whose directory
    contains a finished run with the same hash is loaded from it instead. It
    is rerun if it is named in ``refresh`` (or ``refresh`` is True), or if a
    step supplying its inputs was rerun. A manifest of the run is written beside the
    outputs.

    Precision is set for the whole process. A script that wants double calls
    ``configure_precision(True)`` before ``run``.

    Parameters
    ----------
    stages
        The stages to run; everything they depend on runs too.
    outdir
        The directory each stage writes its own subdirectory under, and
        where the manifest goes.
    refresh
        The names of stages to rerun whether or not their inputs changed,
        or True to rerun every stage. Steps that depend on a rerun stage
        are rerun as well.

    Returns
    -------
    dict
        Results keyed by stage name. A simulation returns its output
        directory. A ``Fit`` generating light-curve fits returns
        ``{dataset_name: {fitopt: FitresTable}}``; one importing existing
        FITRES files returns ``{fitopt: FitresTable}``. See each stage's
        documentation for its return values.
    """
    ordered = validate_graph(stages)
    names = {stage.name for stage in ordered}
    if refresh is True:
        refresh = names
    elif isinstance(refresh, str):
        refresh = {refresh}
    else:
        refresh = set(refresh)
    if refresh - names:
        raise SNANAError(f"refresh names stages not in the graph: "
                         f"{sorted(refresh - names)}", function_name="run")

    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)
    produced: dict[Stage, Any] = {}
    bbc_results = None
    bbc_pending = set()
    hashes: dict[int, str] = {}
    reran: set[str] = set()
    entries = {stage.name: {
        "stage": type(stage).__name__,
        "inputs": {role: used.name for role, used in stage.inputs().items()},
        "status": "pending",
    } for stage in ordered}
    manifest = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "version": version_string(),
        "double_precision": _double_precision(),
        "status": "running",
        "stages": entries,
    }
    manifest_path = root / "MANIFEST.yml"
    _write_manifest(manifest_path, manifest)
    active = None
    try:
        for stage_index, stage in enumerate(ordered):
            active = stage.name
            entry = entries[active]
            started = time.perf_counter()
            entry["status"] = "preparing"
            manifest["current_stage"] = active
            _write_manifest(manifest_path, manifest)
            stage_dir = root / stage.name
            described = _fields(stage, hashes)
            digest = stage_hash(described)
            hashes[id(stage)] = digest
            entry.update(next(iter(described.values())))
            entry["hash"] = digest
            consumed = {role: _result(used, produced) for role, used in stage.inputs().items()}
            finished = _finished(stage_dir)
            unchanged = stage.name not in refresh and not any(
                _producer(used).name in reran for used in stage.inputs().values())
            loaded = finished.get("hash") == digest and unchanged
            if loaded:
                entry["status"] = "loading"
                _write_manifest(manifest_path, manifest)
                try:
                    produced[stage] = stage.load(consumed, stage_dir)
                except NotImplementedError as why:
                    logger.info(str(why))
                    loaded = False
            if loaded:
                entry["produced_version"] = finished.get("version")
                logger.info("loaded %s (produced by %s)", stage.name,
                            entry["produced_version"] or "unknown version")
            else:
                entry["status"] = "running"
                _write_manifest(manifest_path, manifest)
                # an interrupted Fit of the same inputs keeps the versions it finished
                resumed = (isinstance(stage, Fit) and unchanged
                           and _started(stage_dir) == digest)
                if stage_dir.exists() and not resumed:
                    shutil.rmtree(stage_dir)
                stage_dir.mkdir(parents=True, exist_ok=True)
                (stage_dir / STARTED).write_text(yaml.safe_dump({"hash": digest}))
                logger.info(f"running {stage.name}")
                if type(stage) is BBC and not (bbc_pending and stage not in bbc_pending):
                    if stage not in bbc_pending:
                        requests = {stage: (consumed, stage_dir)}
                        configurations = [stage.configuration(muopt)
                                          for muopt in range(len(stage.systematics()[1]) + 1)]
                        diagnostics = any(config.write_biascor or config.write_ccprior
                                          for config in configurations)
                        if not diagnostics:
                            for following in ordered[stage_index + 1:]:
                                if type(following) is not BBC or any(
                                        _producer(used) not in produced
                                        for used in following.inputs().values()):
                                    break
                                entry["status"] = "preparing"
                                active = following.name
                                manifest["current_stage"] = active
                                configurations = []
                                with preparation_step("configuration", [(following.name, (0, 0))]):
                                    muopts = following.systematics()[1]
                                for muopt in range(len(muopts) + 1):
                                    with preparation_step("configuration", [(following.name, (0, muopt))]):
                                        configurations.append(following.configuration(muopt))
                                if any(config.write_biascor or config.write_ccprior
                                       for config in configurations):
                                    break
                                description = _fields(following, hashes)
                                following_hash = stage_hash(description)
                                hashes[id(following)] = following_hash
                                previous = _finished(root / following.name)
                                reusable = (previous.get("hash") == following_hash
                                            and following.name not in refresh and not any(
                                                _producer(used).name in reran
                                                for used in following.inputs().values()))
                                if not reusable:
                                    following_inputs = {role: _result(used, produced)
                                                        for role, used in following.inputs().items()}
                                    requests[following] = following_inputs, root / following.name
                            active = stage.name
                            manifest["current_stage"] = active
                            entry["status"] = "running"
                        if len(requests) > 1:
                            active = None
                            bbc_results = _prepare_shared_bbc(requests, manifest, manifest_path)
                            bbc_pending = set(requests)
                            active = stage.name
                            entry["status"] = "running"
                            manifest["current_stage"] = active
                            _write_manifest(manifest_path, manifest)
                    if stage in bbc_pending:
                        produced[stage] = next(bbc_results)
                        bbc_pending.remove(stage)
                    else:
                        produced[stage] = stage.run(consumed, stage_dir)
                else:
                    produced[stage] = stage.run(consumed, stage_dir)
                reran.add(stage.name)
                entry["produced_version"] = manifest["version"]
                (stage_dir / DONE).write_text(yaml.safe_dump({
                    "stage": type(stage).__name__, "hash": digest,
                    "version": manifest["version"],
                    "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "seconds": round(time.perf_counter() - started, 3)}))
            entry.update(status="loaded" if loaded else "completed", loaded=loaded,
                         seconds=round(time.perf_counter() - started, 3))
            _write_manifest(manifest_path, manifest)

        active = None
        manifest.pop("current_stage", None)
        manifest["status"] = "reporting"
        _write_manifest(manifest_path, manifest)
        results = {stage.name: produced[stage] for stage in ordered}
        summary = cosmology_summary(ordered, results)
        summary_path = root / "COSMOLOGY_SUMMARY.csv"
        if not summary.empty:
            summary.to_csv(summary_path, index=False)
        else:
            summary_path.unlink(missing_ok=True)
        manifest["status"] = "completed"
        _write_manifest(manifest_path, manifest)
    except (Exception, KeyboardInterrupt) as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest.update(status=status, error=f"{type(error).__name__}: {error}")
        if active is not None:
            entries[active].update(status=status, error=manifest["error"],
                                   seconds=round(time.perf_counter() - started, 3))
        try:
            _write_manifest(manifest_path, manifest)
        except OSError:
            logger.exception("Could not update the run manifest after failure")
        logger.error("Pipeline %s %s; progress: %s", status,
                     f"at stage {active}" if active else
                     f"during {manifest.get('current_stage', 'reporting')}",
                     manifest_path.resolve())
        raise
    finally:
        if bbc_results is not None:
            bbc_results.close()
    logger.info("Analysis complete. Outputs: %s", root.resolve())
    logger.info("Run manifest: %s", manifest_path.resolve())
    if not summary.empty:
        logger.info("Cosmology summary: %s", summary_path.resolve())
    return results


def _prepare_shared_bbc(requests, manifest, manifest_path):
    """Prepare ready BBC grids and report failures for all requesting stages."""
    entries = manifest["stages"]
    for stage in requests:
        entries[stage.name]["status"] = "preparing"
    manifest["current_stage"] = "shared BBC preparation"
    try:
        _write_manifest(manifest_path, manifest)
        prepared = prepare_bbc_stages(requests)
    except (Exception, KeyboardInterrupt) as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        for stage in requests:
            entries[stage.name].update(status=status, error=f"{type(error).__name__}: {error}")
        raise
    return run_bbc_stages(requests, *prepared)


def _write_manifest(path: Path, manifest: dict) -> None:
    """Replace the manifest atomically so readers see a complete snapshot."""
    temporary = path.with_suffix(".yml.tmp")
    temporary.write_text(yaml.safe_dump(manifest, sort_keys=False, default_flow_style=False))
    temporary.replace(path)


def _started(stage_dir: Path) -> str | None:
    """The hash of the run the stage's directory was started for, if any."""
    started = stage_dir / STARTED
    return yaml.safe_load(started.read_text())["hash"] if started.is_file() else None


def _finished(stage_dir: Path) -> dict:
    """The completion record, or an empty mapping for an unfinished stage."""
    done = stage_dir / DONE
    if not done.is_file():
        return {}
    return yaml.safe_load(done.read_text())


def cosmology_summary(stages: Sequence[Stage], results: Mapping[str, Any]) -> pd.DataFrame:
    """One row per cosmology fit and COVOPT, preserving any applied blinding.

    Supply the stages and the results returned by ``run``. Results from separate
    realisations, sample partitions and cosmology configurations remain separate.
    The full posterior summaries and diagnostics remain in each WFIT YAML file.
    """
    rows = []
    for stage in stages:
        if not isinstance(stage, Cosmology):
            continue
        covariance = stage.covariance
        distances = getattr(covariance, "distances", None)
        for covopt, result in results[stage.name].items():
            row = dict(cosmology=stage.name, covariance=covariance.name,
                       distances=getattr(distances, "name", None), covopt=covopt,
                       realisation=getattr(distances, "realisation", None),
                       split_index=getattr(distances, "split_index", None),
                       split_count=getattr(distances, "split_count", None))
            row.update({name: result.get(name) for name in
                        ("success", "blinded", "method", "point_estimate_kind")})
            row.update({name: result.get("fit", {}).get(name) for name in
                        ("chi2_total", "ndof_hd", "n_rows")})
            for parameter in result["parameter_order"]:
                row[parameter] = result["point_estimate"][parameter]
                summary = result["parameters"][parameter]
                for field in ("sigma", "q16", "q84"):
                    row[f"{parameter}_{field}"] = summary.get(field)
            rows.append(row)
    return pd.DataFrame(rows)
