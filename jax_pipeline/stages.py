"""The stage contract and the built-in stages.

A stage names the stages it consumes, and runs. Running writes the stage's own
SNANA-format outputs under its directory and returns exactly what those files
contain, so any intermediate can be inspected, or loaded back on a later run
without being rerun.

The built-in stages contain no science code. Each resolves its input file,
calls its package and returns what that package wrote. They differ only in
the form their package takes its overrides in. Each is given a name, then the
stages it consumes, then its input file and options. A stage of your own is a
dataclass with the same three required members, and is used like the
built-in ones.
"""

import dataclasses
import glob
import os
import shutil
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args

import yaml

from jax_create_cov import write_outputs
from jax_create_cov.config import HDMode, build_config as build_covariance_config, parse_covopt
from jax_create_cov.covariance import (hubble_diagram_from, read_submit_info,
                                    reference_cosmology, version_photometry)
from jax_create_cov.covariance import prepare_covariance, select_covariance_grid
from jax_salt2mu.config import build_bbc_config
from jax_salt2mu.preparation import (preparation_step, prepare_biascor_maps, prepare_ccpriors,
                                     prepare_data, read_cells, select_data)
from jax_salt2mu.run import fit_spec, run_prepared_grid
from jax_salt2mu.output import write_outputs as write_bbc_outputs
from jax_salt2mu.tablevar import EVENT_TYPE_BIASCOR, EVENT_TYPE_CCPRIOR, EVENT_TYPE_DATA
from jax_snlc_core.input_settings import merge_settings
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.models import standardised_model
from jax_snlc_core.snana_files import (concat_tables, fitopt_muopt_name, read_covariance, read_factorized_covariance,
                                       read_fitres, resolve_fits_path)
from jax_snlc_core.utils import group_equal
from jax_snlc_fit.calibration import read_kcor
from jax_snlc_fit.config_parser import parse_fit_input_file
from jax_snlc_fit.fitter import LightCurveFitter, fit_model, fit_settings, resolved_config
from jax_snlc_fit.loader import read_shared_files
from jax_snlc_fit.run import dataset_config, fit_dataset, prepare_datasets
from jax_snlc_sim import run as sim_run
from jax_snlc_sim.output import save_snana_outputs
from jax_snlc_sim.inputs import parse_input as parse_sim_input
from jax_snlc_sim.output.write import write_combined_readme
from jax_wfit.config import (build as build_cosmology_config,
                                    read_keys as read_cosmology_keys)
from jax_wfit.hubble import hubble_diagram_from_covariance, load_simulation_truths
from jax_wfit.run import run_wfit

from .synchronise import fit_prepared_common_sample, synchronised_row_counts
from .systematics import (NOMINAL, cells, check_labels, key, overrides_for, scales,
                          selected_cells)


class Stage(ABC):
    """The interface for running an analysis step and sharing its results.

    ``name`` is also the directory the stage writes under. ``inputs`` maps input names to the
    stages that supply them. ``run`` receives their results under those names. A stage that runs once per
    systematic lists them in ``systematics``. A stage that can read its
    result back from what it wrote implements ``load``, which receives the
    same results ``run`` did. Such a stage is not rerun while its inputs are
    unchanged.
    """

    name: str

    @abstractmethod
    def inputs(self) -> Mapping[str, "Stage"]:
        """A mapping from input names to the stages that supply them."""

    def cache_files(self) -> Sequence[Path]:
        """Additional file dependencies named inside configuration files."""
        return ()

    def systematics(self) -> tuple[Sequence[str], Sequence[str]]:
        """The (FITOPT, MUOPT) variants this stage's results are keyed by."""
        return (), ()

    def event_sync(self) -> bool:
        """Whether the light-curve fits request common-event BBC refitting."""
        return False

    def grid_cells(self) -> list[tuple[int, int]]:
        """FITOPT/MUOPT combinations produced by this stage."""
        return cells(*self.systematics())

    @abstractmethod
    def run(self, inputs: Mapping[str, Any], outdir: Path) -> Any:
        """Run, writing this stage's outputs under ``outdir``, and return what
        subsequent stages receive."""

    def load(self, inputs: Mapping[str, Any], outdir: Path) -> Any:
        """What ``run`` returned, read back from ``outdir`` and ``inputs``."""
        raise NotImplementedError(
            f"stage '{self.name}' has no loader, so it runs again")


def sync_setting(options) -> bool | None:
    """Read SNANA's two names for the event-synchronisation setting."""
    values = []
    for name in ("FLAG_USE_SAME_EVENTS", "OPT_SNCID_LIST"):
        if name in options:
            try:
                value = int(options[name])
                if value < 0 or (isinstance(options[name], float) and options[name] != value):
                    raise ValueError
            except (ValueError, TypeError, OverflowError):
                raise SNANAError(f"{name} must be a non-negative integer", function_name="event_sync") from None
            values.append(value > 0)
    if len(set(values)) > 1:
        raise SNANAError("FLAG_USE_SAME_EVENTS and OPT_SNCID_LIST disagree", function_name="event_sync")
    return values[0] if values else None


def one_version(source):
    """``source`` as one data version's fits: a :class:`Fit` fitting a single
    version stands for that version; one fitting several must be narrowed
    with :meth:`Fit.version`. Anything else is returned unchanged."""
    if not isinstance(source, Fit) or source.fitres is not None:
        return source
    if len(source.datasets) != 1:
        names = ", ".join(data.name for data in source.datasets)
        raise SNANAError(f"stage '{source.name}' fits {len(source.datasets)} "
                         f"versions ({names}); pick one with "
                         f"{source.name}.version(...)", function_name=source.name)
    return source.version(source.datasets[0])


def _stages(one_or_more) -> "tuple[Stage | FitVersion, ...]":
    """A stage, or a list of them, as a tuple, each fit narrowed to one
    version."""
    if one_or_more is None:
        return ()
    if isinstance(one_or_more, (Stage, FitVersion)):
        one_or_more = (one_or_more,)
    return tuple(one_version(source) for source in one_or_more)


def _is_stages(value) -> bool:
    """Whether an override's value names stages rather than files: one
    stage or version's fits, or a list of them."""
    return isinstance(value, (Stage, FitVersion)) or (
        isinstance(value, (list, tuple)) and len(value) > 0
        and all(isinstance(v, (Stage, FitVersion)) for v in value))


def _keys_only(overrides: Mapping[str, Any]) -> dict[str, Any]:
    """The overrides a package is given: those not naming stages."""
    return {k: v for k, v in overrides.items() if not _is_stages(v)}


# eq=False: two stages with equal settings stay distinct nodes
@dataclass(eq=False)
class Data(Stage):
    """A data version on disk: the directory containing its .LIST, HEAD and
    PHOT files.

    Returns the directory.
    """

    name: str
    directory: Path

    def __post_init__(self):
        self.directory = Path(self.directory)

    def inputs(self) -> Mapping[str, Stage]:
        return {}

    def run(self, inputs: Mapping[str, Any], outdir: Path) -> Path:
        return self.load(inputs, outdir)

    def load(self, inputs: Mapping[str, Any], outdir: Path) -> Path:
        return self.directory


@dataclass(eq=False)
class Sim(Stage):
    """A simulated data version, from one input file or from several written
    into one version, as SNANA's batch jobs combine models or seeds.

    ``input_files`` is the path of one sim-input file, or a list of paths
    whose events are written into one version. ``overrides`` is
    ``{KEY: value}`` with the keys spelt as the input file spells them:
    either one mapping applied to every input, or a list of one mapping per
    input in the same order. ``batch_size`` is the number of events simulated
    at once. The number of events and the seed are set by input-file keys.
    Returns the directory.

    Several inputs make one version: one ``<name>.LIST`` naming each part's
    ``<name>_<stem>_HEAD.FITS.gz``. Each input's ``CIDOFF`` is ``CIDRAN_MIN``
    plus the number of events generated before it, keeping sequential CIDs
    distinct unless CIDOFF is overridden.
    With random CIDs (``FORMAT_MASK`` bit 16), inputs sharing a seed take
    disjoint slices of one CID list and the offset changes their other random
    numbers too, as in SNANA; inputs with different seeds draw separate lists,
    which can share CIDs.
    """

    name: str
    input_files: Any
    overrides: Any = field(default_factory=dict)
    batch_size: int | None = None

    def __post_init__(self):
        one = isinstance(self.input_files, (str, os.PathLike))
        self.input_files = tuple(
            Path(f) for f in ([self.input_files] if one else self.input_files))
        per_input = (self.overrides if not isinstance(self.overrides, Mapping)
                     else [self.overrides] * len(self.input_files))
        if len(per_input) != len(self.input_files):
            raise SNANAError(
                f"stage '{self.name}' has {len(self.input_files)} input files "
                f"but {len(per_input)} override mappings", function_name=self.name)
        self.overrides = tuple(dict(o) for o in per_input)

    def inputs(self) -> Mapping[str, Stage]:
        return {}

    def configuration(self, index: int = 0, *, overrides=None):
        """Parse one simulation input without initialising or generating events."""
        settings = self.overrides[index] if overrides is None else overrides
        return parse_sim_input(str(self.input_files[index]), settings)

    def run(self, inputs: Mapping[str, Any], outdir: Path) -> Path:
        # the writer creates the version dir, so pass its parent
        parent = str(Path(outdir).parent)
        stems = [f.stem for f in self.input_files]
        combined = len(self.input_files) > 1
        generated = 0
        parts = []
        for index, (path, user) in enumerate(zip(self.input_files, self.overrides)):
            # seconds per phase, for the OUTPUT_SUMMARY, as snlc_sim.py
            # reports them: init is everything before the first event
            timings = {}
            start = time.perf_counter()
            simulation = self.configuration(index)
            if combined and not ({"GENPREFIX", "genprefix"} & user.keys()):
                stem = path.stem if stems.count(path.stem) == 1 else f"{path.stem}_{index}"
                simulation.output.genprefix = f"{self.name}_{stem}"
            if combined and not ({"CIDOFF", "cidoff"} & user.keys()):
                # the batch layer starts CIDOFF at CIDRAN_MIN (submit_prog_sim.py:980)
                simulation.output.cidoff = simulation.output.cidran_min + generated
            if self.batch_size is not None:
                simulation.numerical = dataclasses.replace(
                    simulation.numerical, batch_size=self.batch_size)
            sim = sim_run.prepare(simulation)
            if sim is None:
                # a set-up-and-stop mode (SIMLIB_DUMP, INIT_ONLY, ...)
                continue
            timings['init'] = time.perf_counter() - start
            start = time.perf_counter()
            results, selection, spectra = sim_run.run_simulation(sim)
            timings['generate'] = time.perf_counter() - start
            written = save_snana_outputs(
                sim, results, self.name,
                selection, spectra_by_sn=spectra, output_dir=parent,
                combined=combined, timings=timings)
            generated += written["n_passed"] + written["n_failed"]
            if combined:
                parts.append((path, simulation.output.genprefix))
        if parts:
            write_combined_readme(outdir, self.name, parts)
        return self.load(inputs, outdir)

    def load(self, inputs: Mapping[str, Any], outdir: Path) -> Path:
        return Path(outdir)


@dataclass(eq=False)
class Fit(Stage):
    """Light-curve fits of data versions, once per FITOPT: fitted here from
    ``input_file`` for every version in ``datasets``, or read from
    ``fitres`` made elsewhere.

    Fitted here, the versions share one preparation and the fitters compiled
    for it. Every version is read first, so the compiled arrays hold the
    longest light curves and redshift ranges among them, and each version
    is then fitted without compiling again. ``overrides`` is ``{KEY:
    value}``, using the input file's key names. ``fitopts`` is ``{label:
    {KEY: value}}``: each entry varies the specified settings relative to
    the nominal fit, which always runs as well.

    A run interrupted part way resumes when run again with the same inputs:
    the versions it finished are kept, and the rest are fitted at the shape
    it prepared, to the same tables.

    ``fitres`` is one FITRES file, the nominal alone, or a version directory
    of ``FITOPTnnn.FITRES`` files as SNANA's batch fit writes them, labelled
    by the ``FITOPT_LIST`` of the ``SUBMIT.INFO`` it writes beside the
    version directories.

    Fitted here, with ``OPT_SNCID_LIST`` set every FITOPT fits the nominal's
    events, as ``submit_batch_jobs`` does by setting ``SNCID_LIST_FILE`` to
    FITOPT000. As there, a label containing ``NOREJECT`` is exempt.

    Fitted here, returns ``{version: {key: FitresTable}}``, keyed by each
    data stage's name; read from ``fitres``, returns ``{key: FitresTable}``.
    The nominal is under ``"nominal"`` and each FITOPT under its label.
    Later stages take one version's fits through :meth:`version`, or this
    stage itself when it fits one version or reads ``fitres``.
    """

    name: str
    input_file: Path | None = None
    datasets: Sequence[Stage] = ()
    overrides: Mapping[str, Any] = field(default_factory=dict)
    fitopts: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    fitres: Path | None = None
    # the FITOPT labels after the nominal: fitopts', or SUBMIT.INFO's
    labels: tuple[str, ...] = field(init=False)

    def __post_init__(self):
        self.datasets = tuple(self.datasets)
        if self.fitres is not None:
            if self.input_file is not None or self.datasets or self.overrides or self.fitopts:
                raise SNANAError(f"stage '{self.name}' reads its fits from "
                                 f"fitres, so takes no input file, datasets, "
                                 f"overrides or fitopts", function_name=self.name)
            self.fitres = Path(resolve_fits_path(self.fitres))
            self.labels = (_submitted_labels(self.fitres, self.name)
                           if self.fitres.is_dir() else ())
            return
        if self.input_file is None:
            raise SNANAError(f"stage '{self.name}' fits data with an input "
                             f"file, or reads fitres", function_name=self.name)
        self.input_file = Path(self.input_file)
        check_labels(self.name, tuple(self.fitopts))
        swapping = [label for label, keys in self.fitopts.items()
                    if any(_is_stages(v) for v in keys.values())]
        if swapping:
            raise SNANAError(f"stage '{self.name}': a fit systematic changes "
                             f"keys only, but {swapping} name stages",
                             function_name=self.name)
        self.labels = tuple(self.fitopts)

    def inputs(self) -> Mapping[str, Stage]:
        return {f"data{i}": stage for i, stage in enumerate(self.datasets)}

    def version(self, data: Stage) -> "FitVersion":
        """The fits of ``data``, one of this stage's ``datasets``."""
        if self.fitres is not None:
            raise SNANAError(f"stage '{self.name}' reads one version from "
                             f"fitres; use the stage itself", function_name=self.name)
        if not any(data is d for d in self.datasets):
            raise SNANAError(f"stage '{self.name}' does not fit "
                             f"'{data.name}'", function_name=self.name)
        return FitVersion(self, data)

    def systematics(self) -> tuple[Sequence[str], Sequence[str]]:
        return self.labels, ()

    def cache_files(self) -> Sequence[Path]:
        if self.fitres is not None and self.fitres.is_dir():
            return (self.fitres.parent / "SUBMIT.INFO",)
        return ()

    def event_sync(self) -> bool:
        if self.fitres is None:
            return self.configuration().selection.opt_sncid_list > 0
        if not self.fitres.is_dir():
            return False
        info = read_submit_info(self.fitres.parent)
        return sync_setting(info) or False

    def configuration(self, cell: tuple[int, int] = (0, 0)):
        """Parse one fit configuration without initialising the model or fitting data."""
        return parse_fit_input_file(self.input_file,
            merge_settings(self.overrides, overrides_for(self.fitopts, cell[0])))

    def _groups(self):
        """Each cell's config, and the cells grouped by fit settings with the
        model each group fits."""
        configs = {cell: self.configuration(cell) for cell in cells(self.fitopts)}
        return configs, [(group, fit_model(configs[group[0]]))
                         for group in group_equal({
                             cell: fit_settings(config)
                             for cell, config in configs.items()})]

    @staticmethod
    def _files(configs):
        """Each cell's kcor and the files its data versions share, read once
        for each Data record among the cells."""
        files = {}
        for same in group_equal({cell: config.data
                                 for cell, config in configs.items()}):
            config = configs[same[0]]
            files.update(dict.fromkeys(
                same, dict(kcor=read_kcor(config), shared=read_shared_files(config))))
        return files

    @staticmethod
    def _fitters(configs, groups, files, shape):
        """One fitter per group of cells, and for each cell its config with
        the fitter's resolved settings, its kcor and its shared files."""
        fitters = {}
        compiled_cache = {}
        for group, model in groups:
            fitter = LightCurveFitter._from_prepared(configs[group[0]], model,
                                      files[group[0]]["kcor"], shape,
                                      compiled_cache=compiled_cache)
            fitters.update({
                cell: dict(fitter=fitter,
                           config=fitter.with_fit_settings(configs[cell]),
                           **files[cell])
                for cell in group})
        return fitters

    def _table(self, outdir: Path, cell: tuple[int, int], version: str) -> Path:
        return Path(outdir) / fitopt_muopt_name(cell) / f"{version}.FITRES.TEXT"

    def source_files(self) -> Sequence[Path]:
        """Existing FITRES inputs in FITOPT order; empty for fits produced here."""
        if self.fitres is None:
            return ()
        if not self.fitres.is_dir():
            return (self.fitres,)
        return tuple(_fitopt_file(self.fitres, i, self.name)
                     for i in range(len(self.labels) + 1))

    def run(self, inputs: Mapping[str, Any], outdir: Path) -> Mapping[str, Any]:
        if self.fitres is not None:
            return self.load(inputs, outdir)
        directories = [inputs[f"data{i}"] for i in range(len(self.datasets))]
        configs, groups = self._groups()
        for group, model in groups:
            for cell in group:
                configs[cell] = resolved_config(configs[cell], model)
        files = self._files(configs)
        shape_file = Path(outdir) / "SHAPE.yml"
        finished_file = Path(outdir) / "FINISHED.yml"
        finished = (yaml.safe_load(finished_file.read_text())
                    if finished_file.is_file() else [])
        if shape_file.is_file():
            # resuming an interrupted run: the versions left are fitted at the
            # shape it prepared, each read when it is fitted
            shape = yaml.safe_load(shape_file.read_text())
            prepared = None
        else:
            ordered_cells = [cell for group, _ in groups for cell in group]
            shape, datasets = prepare_datasets([
                dict(config=configs[cell], model=model, **files[cell])
                for group, model in groups for cell in group], directories)
            shape_file.write_text(yaml.safe_dump(shape))
            prepared = dict(zip(ordered_cells, datasets))
            del datasets
        fitters = self._fitters(configs, groups, files, shape)
        for data, directory in zip(self.datasets, directories):
            if data.name in finished:
                continue
            for cell in cells(self.labels):
                stem = Path(outdir) / fitopt_muopt_name(cell) / data.name
                stem.parent.mkdir(parents=True, exist_ok=True)
                config = dataset_config(fitters[cell]["config"], directory, stem)
                if (cell[0] and config.selection.opt_sncid_list
                        and "NOREJECT" not in self.labels[cell[0] - 1]):
                    config = dataclasses.replace(config, selection=dataclasses.replace(
                        config.selection,
                        sncid_list_file=str(self._table(outdir, (0, 0), data.name))))
                dataset = (None if prepared is None
                           else prepared[cell].pop(Path(directory).resolve()))
                _, written = fit_dataset(fitters[cell]["fitter"], config,
                                         fitters[cell]["kcor"], fitters[cell]["shared"],
                                         prepared_data=dataset)
                del dataset
                if written is None:
                    raise SNANAError(
                        f"stage '{self.name}' wrote no FITRES table; its "
                        f"SNTABLE_LIST must ask for one", function_name=self.name)
            finished.append(data.name)
            finished_file.write_text(yaml.safe_dump(finished))
        return self.load(inputs, outdir)

    def load(self, inputs: Mapping[str, Any], outdir: Path) -> Mapping[str, Any]:
        if self.fitres is None:
            return {data.name: {key(cell, self.labels): read_fitres(
                                    str(self._table(outdir, cell, data.name)))
                                for cell in cells(self.labels)}
                    for data in self.datasets}
        if not self.fitres.is_dir():
            return {NOMINAL: read_fitres(str(self.fitres))}
        return {key((i, 0), self.labels): read_fitres(str(f))
                for i, f in enumerate(self.source_files())}


@dataclass(frozen=True)
class FitVersion:
    """One data version's fits from a :class:`Fit`, made by
    :meth:`Fit.version`. Later stages take it where they take a fit stage,
    and receive that version's tables."""

    fit: Fit
    data: Stage

    @property
    def name(self) -> str:
        return f"{self.fit.name}/{self.data.name}"

    def systematics(self) -> tuple[Sequence[str], Sequence[str]]:
        return self.fit.systematics()

    def event_sync(self) -> bool:
        return self.fit.event_sync()


def _submitted_labels(version_dir: Path, name: str) -> tuple[str, ...]:
    """The FITOPT labels after the nominal that SNANA's batch fit lists in
    the SUBMIT.INFO beside ``version_dir``, each row ``[FITOPTnnn, label,
    arguments]``."""
    rows = read_submit_info(version_dir.parent)["FITOPT_LIST"][1:]
    unlabelled = [num for num, label, _ in rows if label is None]
    if unlabelled:
        raise SNANAError(f"stage '{name}': {version_dir.parent}/SUBMIT.INFO "
                         f"gives {unlabelled} no label", function_name=name)
    return tuple(label for _, label, _ in rows)


def _fitopt_file(version_dir: Path, number: int, name: str) -> Path:
    """FITOPT ``number``'s FITRES file in ``version_dir``, gzipped or not."""
    found = glob.glob(str(version_dir / f"FITOPT{number:03d}.FITRES*"))
    if len(found) != 1:
        raise SNANAError(f"stage '{name}': {version_dir} holds {len(found)} "
                         f"FITOPT{number:03d}.FITRES files, not one",
                         function_name=name)
    return Path(found[0])


@dataclass(eq=False)
class BBC(Stage):
    """Distance fits (SALT2mu) of fitted tables, once per FITOPT x MUOPT.

    ``data``, ``biascor`` and ``ccprior`` are each a stage or a list of them.
    A list is concatenated as SALT2mu's ``datafile=a,b`` is. The FITOPT axis
    is the union of the data stages' FITOPTs, in order of first appearance.
    A label every survey shares is a systematic of all of them, and a label
    on one survey is that survey's alone. In a cell, a survey without the
    label contributes its nominal, as with Pippin's FITOPT_MAP. Every cell
    uses the nominal bias-correction and classifier-prior tables unless
    correction FITOPTs or replacement files are specified.

    ``settings`` supplies typed Python/YAML fields; ``muopt_settings`` maps
    labels to the changed fields, in order after the nominal. Setting
    ``datafile``, ``simfile_biascor`` or
    ``simfile_ccprior`` to a stage instead of a file makes that MUOPT use
    its results, as a Pippin MUOPT names another simulation or classifier.
    ``overrides`` and ``muopts`` accept SALT2mu keys and values instead.
    Each MUOPT overrides nominal inputs before scientific rules are applied;
    typed settings follow SNANA overrides within each layer.

    Returns ``{key: {"M0DIF": FitresTable, "FITRES": FitresTable}}``, read
    from what it wrote: the nominal under ``"nominal"``, a cell with one
    systematic under its label, and one with both under the ``(fitopt,
    muopt)`` pair. Alpha, beta and the rest of the fit are in the M0DIF's
    ``meta``.
    """

    name: str
    data: Any
    input_file: Path | None = None
    biascor: Any = None
    ccprior: Any = None
    overrides: Mapping[str, Any] = field(default_factory=dict)
    muopts: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    #: Typed Python/YAML fields, applied after the nominal SNANA overrides.
    settings: Mapping[str, Any] = field(default_factory=dict)
    #: Typed changes for each MUOPT, applied after that MUOPT's SNANA overrides.
    muopt_settings: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    #: FITOPT label per correction source, in the same order as biascor/ccprior.
    #: None selects the nominal fit.
    biascor_fitopts: Sequence[str | None] = ()
    ccprior_fitopts: Sequence[str | None] = ()
    #: Correction FITOPT selections for samples supplied by individual MUOPTs.
    muopt_biascor_fitopts: Mapping[str, Sequence[str | None]] = field(default_factory=dict)
    muopt_ccprior_fitopts: Mapping[str, Sequence[str | None]] = field(default_factory=dict)
    #: External correction FITRES files keyed by data FITOPT label.
    replace_biascor: Mapping[str, Path] = field(default_factory=dict)
    replace_ccprior: Mapping[str, Path] = field(default_factory=dict)

    #: FITOPTxMUOPT expressions; empty selects the complete grid.
    fitopt_muopt: Sequence[str] = ()
    #: Partition data by integer CID modulo split_count; indices start at one.
    split_count: int = 1
    split_index: int = 1
    #: Simulation realisation index, when expanded from a Pippin ensemble.
    realisation: int | None = None
    #: Require an explicitly supplied probability column in each data source.
    require_probability_column: bool = False
    #: None inherits the light-curve fits; False disables common-event refitting.
    sync_events: bool | None = None

    def event_sync(self) -> bool:
        if self.sync_events is False:
            return False
        settings = {stage.event_sync() for label in (None, *self.muopts)
                    for event_type, _, stage in self._sources(label)
                    if event_type == EVENT_TYPE_DATA}
        if len(settings) > 1:
            raise SNANAError("all data fits must agree on event synchronisation; "
                             "set sync_events=False to disable it", function_name=self.name)
        inherited = next(iter(settings), False)
        if self.sync_events is True and not inherited:
            raise SNANAError("enable FLAG_USE_SAME_EVENTS in the light-curve fits before BBC",
                             function_name=self.name)
        if inherited and (0, 0) not in self.grid_cells():
            raise SNANAError("event synchronisation requires FITOPT000/MUOPT000", function_name=self.name)
        return inherited

    def grid_cells(self) -> list[tuple[int, int]]:
        return selected_cells(*self.systematics(), self.fitopt_muopt)

    def __post_init__(self):
        self.input_file = None if self.input_file is None else Path(self.input_file)
        self.muopts = dict(self.muopts)
        for label in self.muopt_settings:
            self.muopts.setdefault(label, {})
        if self.sync_events is not None and type(self.sync_events) is not bool:
            raise SNANAError("sync_events must be True, False or None", function_name=self.name)
        if (type(self.split_count) is not int or type(self.split_index) is not int
                or not 1 <= self.split_index <= self.split_count):
            raise SNANAError("split_index must be between 1 and split_count (positive integers)", function_name=self.name)
        self.data = _stages(self.data)
        self.biascor = _stages(self.biascor)
        self.ccprior = _stages(self.ccprior)
        self.replace_biascor = {label: Path(path) for label, path in self.replace_biascor.items()}
        self.replace_ccprior = {label: Path(path) for label, path in self.replace_ccprior.items()}
        for role, sources, selected in (
                ("biascor", self.biascor, self.biascor_fitopts),
                ("ccprior", self.ccprior, self.ccprior_fitopts)):
            if selected and len(selected) != len(sources):
                raise SNANAError(f"{role}_fitopts must have one entry per {role} source", function_name=self.name)
            for stage, label in zip(sources, selected):
                if label is not None and label not in stage.systematics()[0]:
                    raise SNANAError(f"{role} source {stage.name} has no FITOPT {label}", function_name=self.name)
        for role, selections in (("biascor", self.muopt_biascor_fitopts),
                                  ("ccprior", self.muopt_ccprior_fitopts)):
            for muopt, selected in selections.items():
                sources = {**self.muopts.get(muopt, {}),
                           **self.muopt_settings.get(muopt, {})}.get(f"simfile_{role}")
                if not _is_stages(sources) or len(selected) != len(_stages(sources)):
                    raise SNANAError(f"MUOPT {muopt}: {role} selection needs one entry per swapped source", function_name=self.name)
                for stage, label in zip(_stages(sources), selected):
                    if label is not None and label not in stage.systematics()[0]:
                        raise SNANAError(f"{stage.name} has no FITOPT {label}", function_name=self.name)
        named = [name for name, values in (("overrides", self.overrides),
                                          ("settings", self.settings))
                 if "prefix" in values]
        named.extend(f"MUOPT {label}" for label in self.muopts
                     if "prefix" in self.muopts[label]
                     or "prefix" in self.muopt_settings.get(label, {}))
        if named:
            raise SNANAError(f"stage '{self.name}' names its own outputs; "
                             f"drop prefix from {', '.join(named)}",
                             function_name=self.name)

    def _sources(self, label: str | None) -> list[tuple[int, str, Stage]]:
        """(event type, driver key, stage) for every table one cell reads:
        the MUOPT's stages where it overrides an input, else this stage's.
        File or analytic-prior overrides supply no stage table. The key
        is ``role:stage``, unique because stage names are."""
        swaps = ({} if label is None else
                 {**self.muopts[label], **self.muopt_settings.get(label, {})})
        roles = (
            ("data", "datafile", EVENT_TYPE_DATA, self.data),
            ("biascor", "simfile_biascor", EVENT_TYPE_BIASCOR, self.biascor),
            ("ccprior", "simfile_ccprior", EVENT_TYPE_CCPRIOR, self.ccprior),
        )
        chosen = []
        for role, salt_key, event_type, own in roles:
            if salt_key in swaps:
                given = swaps[salt_key]
                stages = _stages(given) if _is_stages(given) else ()
            else:
                stages = own
            chosen += [(event_type, f"{role}:{stage.name}", stage) for stage in stages]
        return chosen

    def inputs(self) -> Mapping[str, Stage]:
        return {slot: stage
                for label in (None, *self.muopts)
                for _, slot, stage in self._sources(label)}

    def systematics(self) -> tuple[Sequence[str], Sequence[str]]:
        fitopts = []
        for stage in self.data:
            fitopts += [f for f in stage.systematics()[0] if f not in fitopts]
        # data a MUOPT swaps in is the same fit reclassified, so it brings
        # no FITOPT of its own
        for label in self.muopts:
            for event_type, _, stage in self._sources(label):
                unknown = [f for f in stage.systematics()[0] if f not in fitopts]
                if event_type == EVENT_TYPE_DATA and unknown:
                    raise SNANAError(f"stage '{self.name}': MUOPT {label} swaps in "
                                     f"data with FITOPTs the nominal data lacks: "
                                     f"{unknown}", function_name=self.name)
        unknown = (set(self.replace_biascor) | set(self.replace_ccprior)) - set(fitopts) - {"DEFAULT"}
        if unknown:
            raise SNANAError(f"correction replacements name unknown FITOPTs: {sorted(unknown)}", function_name=self.name)
        check_labels(self.name, tuple(fitopts), tuple(self.muopts))
        return tuple(fitopts), tuple(self.muopts)

    def configuration(self, muopt: int = 0):
        """Parse BBC settings for a MUOPT without reading fitted tables."""
        label = None if muopt == 0 else tuple(self.muopts)[muopt - 1]
        return build_bbc_config(self.input_file, [
            (self.overrides, self.settings),
            (_keys_only(overrides_for(self.muopts, muopt)),
             _keys_only(self.muopt_settings.get(label, {}))),
        ])

    def _cell(self, inputs, cell, shared_tables):
        """The config and tables of one cell of the grid.

        ``shared_tables`` holds replacement files and concatenated data tables
        shared by cells. Bias-correction and CC-prior sources remain separate
        tables until correction preparation.
        """
        fitopts, muopts = self.systematics()
        fitopt, muopt = cell
        label = None if muopt == 0 else muopts[muopt - 1]
        config = self.configuration(muopt)
        sources = {}
        applied = "DEFAULT" if fitopt == 0 else fitopts[fitopt - 1]
        correction_index = {EVENT_TYPE_BIASCOR: 0, EVENT_TYPE_CCPRIOR: 0}
        swaps = ({} if label is None else
                 {**self.muopts[label], **self.muopt_settings.get(label, {})})
        replacements = {
            EVENT_TYPE_BIASCOR: {} if "simfile_biascor" in swaps else self.replace_biascor,
            EVENT_TYPE_CCPRIOR: {} if "simfile_ccprior" in swaps else self.replace_ccprior,
        }
        correction_fitopts = {
            EVENT_TYPE_BIASCOR: (self.muopt_biascor_fitopts.get(label, ())
                                if "simfile_biascor" in swaps else self.biascor_fitopts),
            EVENT_TYPE_CCPRIOR: (self.muopt_ccprior_fitopts.get(label, ())
                                if "simfile_ccprior" in swaps else self.ccprior_fitopts),
        }
        for event_type, slot, stage in self._sources(label):
            which = NOMINAL
            if event_type == EVENT_TYPE_DATA and fitopt:
                which = applied if applied in stage.systematics()[0] else NOMINAL
            elif event_type in correction_fitopts:
                if applied in replacements[event_type]:
                    continue
                selected = correction_fitopts[event_type]
                index = correction_index[event_type]
                correction_index[event_type] += 1
                if selected:
                    which = selected[index] or NOMINAL
            table = inputs[slot][which]
            if (self.require_probability_column and event_type == EVENT_TYPE_DATA
                    and config.varname_pia not in table.df):
                raise SNANAError(
                    f"{stage.name}:{which} needs existing probability column {config.varname_pia!r}; "
                    "supply it in the FITRES table or imported predictions",
                    function_name=self.name)
            sources.setdefault(event_type, []).append((slot, which))
        for event_type, files in replacements.items():
            if applied in files:
                sources[event_type] = [str(files[applied])]
        tables = {}
        for event_type, keys in sources.items():
            source_tables = []
            for source in keys:
                if isinstance(source, tuple):
                    source_tables.append(inputs[source[0]][source[1]])
                else:
                    if source not in shared_tables:
                        shared_tables[source] = read_fitres(source)
                    source_tables.append(shared_tables[source])
            if event_type == EVENT_TYPE_DATA:
                data_sources = tuple(id(table) for table in source_tables)
                if data_sources not in shared_tables:
                    shared_tables[data_sources] = concat_tables(source_tables)
                tables[event_type] = shared_tables[data_sources]
            else:
                tables[event_type] = tuple(source_tables)
        if self.split_count > 1:
            table = tables[EVENT_TYPE_DATA]
            ids = table.df["CID"].astype(str)
            if not ids.str.fullmatch(r"[0-9]+").all():
                raise SNANAError("NSPLITRAN requires non-negative integer CIDs", function_name=self.name)
            keep = ids.map(lambda cid: int(cid) % self.split_count + 1 == self.split_index)
            if not keep.any():
                raise SNANAError(f"NSPLITRAN partition {self.split_index} contains no events", function_name=self.name)
            tables[EVENT_TYPE_DATA] = dataclasses.replace(table, df=table.df.loc[keep].reset_index(drop=True))
        return config, tables

    def run(self, inputs: Mapping[str, Any], outdir: Path) -> Mapping[str, Any]:
        requests = {self: (inputs, outdir)}
        prepared = prepare_bbc_stages(requests)
        return next(run_bbc_stages(requests, *prepared))

    def export_covariance_inputs(self, run_dir: Path, outdir: Path) -> None:
        """Copy this completed BBC stage into create_covariance's CLI layout.

        ``run_dir`` is the directory passed to :func:`jax_pipeline.run`.
        ``outdir`` must not exist: it receives SUBMIT.INFO and a subdirectory
        named after this stage, containing the flat FITOPT/MUOPT grid.
        Use this same BBC configuration that produced the run, so the labels
        in SUBMIT.INFO match the numbered cells.
        """
        fitopts, muopts = self.systematics()
        source = Path(run_dir) / self.name
        copies = [
            (source / fitopt_muopt_name(cell) / f"{self.name}.{suffix}",
             f"{fitopt_muopt_name(cell)}.{suffix}")
            for cell in self.grid_cells()
            for suffix in ("M0DIF", "FITRES")
        ]
        for path, _ in copies:
            if not path.is_file():
                raise SNANAError(f"missing BBC output {path}; run the stage first",
                                 function_name="export_covariance_inputs")
        destination = Path(outdir)
        if destination.exists():
            raise SNANAError(f"export directory already exists: {destination}",
                             function_name="export_covariance_inputs")
        version_dir = destination / self.name
        version_dir.mkdir(parents=True)
        for path, name in copies:
            shutil.copyfile(path, version_dir / name)
        submit_info = {
            f"{kind}_OUT_LIST": [
                [f"{kind}{number:03d}", label, label]
                for number, label in enumerate(("DEFAULT", *labels))]
            for kind, labels in (("FITOPT", fitopts), ("MUOPT", muopts))
        }
        (destination / "SUBMIT.INFO").write_text(
            yaml.safe_dump(submit_info, sort_keys=False))

    def load(self, inputs: Mapping[str, Any], outdir: Path) -> Mapping[str, Any]:
        fitopts, muopts = self.systematics()
        return {
            key(cell, fitopts, muopts): {
                "M0DIF": read_fitres(str(Path(outdir) / fitopt_muopt_name(cell) / f"{self.name}.M0DIF")),
                "FITRES": read_fitres(str(Path(outdir) / fitopt_muopt_name(cell) / f"{self.name}.FITRES")),
            }
            for cell in self.grid_cells()
        }


def prepare_bbc_stages(requests):
    """Assemble ready BBC grids and prepare their shared corrections."""
    t_start = time.time()
    shared_tables = {}
    grids = {}
    for stage, (inputs, _) in requests.items():
        for cell in stage.grid_cells():
            key = stage.name, cell
            with preparation_step("sources", [key]):
                grids[key] = stage._cell(inputs, cell, shared_tables)
    del shared_tables
    cells = read_cells(grids)
    maps = prepare_biascor_maps(cells)
    priors = prepare_ccpriors(cells, maps)
    fit_widths = cosmology_widths = None
    if len(requests) > 1:
        cosmology_widths = {}
        for config, sources, _ in cells.values():
            model = standardised_model(config.model_lcfit)
            cosmology_widths[model] = max(cosmology_widths.get(model, 0), sources[EVENT_TYPE_DATA].size)
        fit_widths = {}
        with preparation_step("fit_sizes", list(cells)):
            for stage in requests:
                grid = {cell: cells[stage.name, cell] for cell in stage.grid_cells()}
                data = prepare_data(grid, cosmology_widths=cosmology_widths)
                if stage.event_sync():
                    counts = synchronised_row_counts(grid, data, stage.systematics()[0])
                else:
                    counts = {cell: int(select_data(config, sources, data[cell], None).cutmask.sum())
                              for cell, (config, sources, _) in grid.items()}
                for cell, (config, _, _) in grid.items():
                    mapping = maps[stage.name, cell]
                    mapping = None if isinstance(mapping, tuple) else mapping
                    if mapping is not None and not mapping.maps:
                        mapping = None
                    prior = priors[stage.name, cell]
                    spec = fit_spec(config, standardised_model(config.model_lcfit), mapping,
                                    prior[0] if prior is not None else None,
                                    prior[1] if prior is not None else 0)
                    fit_widths[spec] = max(fit_widths.get(spec, 0), counts[cell])
                del data
    return cells, maps, priors, fit_widths, cosmology_widths, t_start


def run_bbc_stages(requests, cells, maps, priors, fit_widths, cosmology_widths, t_start):
    """Fit, write and yield one BBC grid at a time, releasing its inputs."""
    for stage, (inputs, outdir) in requests.items():
        grid = {cell: cells.pop((stage.name, cell)) for cell in stage.grid_cells()}
        grid_maps = {cell: maps.pop((stage.name, cell)) for cell in grid}
        grid_priors = {cell: priors.pop((stage.name, cell)) for cell in grid}
        if stage.event_sync():
            fitopts, muopts = stage.systematics()
            fitted = fit_prepared_common_sample(grid, grid_maps, grid_priors,
                                                fitopts, muopts, Path(outdir), t_start=t_start,
                                                fit_widths=fit_widths, cosmology_widths=cosmology_widths)
        else:
            fitted = run_prepared_grid(grid, grid_maps, grid_priors, t_start=t_start,
                                       fit_widths=fit_widths, cosmology_widths=cosmology_widths)
        for cell, (result, output, zbins, tv, accept) in fitted.items():
            cell_dir = Path(outdir) / fitopt_muopt_name(cell)
            cell_dir.mkdir(parents=True, exist_ok=True)
            write_bbc_outputs(cell_dir / f"{stage.name}.FITRES", result, output,
                              zbins, tv, accept)
        del result, output, zbins, tv, accept
        result = stage.load(inputs, outdir)
        del fitted, grid, grid_maps, grid_priors
        yield result
        t_start = time.time()


@dataclass(eq=False)
class Covariance(Stage):
    """The systematic covariance of a distance-fit grid.

    The matrices are built from the systematic variations of its input distances. An
    unfiltered ``ALL`` is always built first. ``covopts`` add to it, each in
    create_covariance's own form, ``"[label] [fitopt filter,muopt filter]"``.
    ``scales`` multiplies the error a systematic contributes, by label, as
    create_covariance's scale file does. An unlisted label scales by one.
    ``overrides`` are input-file keys (``WRITE_MASK_COV`` and the like),
    applied over ``input_file``.

    Returns ``hubble_diagram``, ``covariances`` keyed by label, and
    ``isdata_real``, read from what it wrote. Each covariance entry carries
    its ``kind`` and either factor arrays or a dense inverse ``matrix``.
    Compact output (mask 8) is the default; explicit settings take precedence.
    """

    name: str
    distances: Stage
    input_file: Path | None = None
    overrides: Mapping[str, Any] = field(default_factory=dict)
    covopts: Sequence[str | Mapping[str, Any]] = ()
    scales: Mapping[str, float] = field(default_factory=dict)
    mode: HDMode | None = None
    #: FITRES column to bin count for mode="rebin"; {} uses IZBIN alone.
    #: When omitted, REBIN_BINS may come from input_file or overrides.
    rebin_bins: Mapping[str, int] | None = None
    #: SALT stretch and colour bin counts, matching --nbin_x1/--nbin_c.
    #: Use these together in mode="rebin", or use rebin_bins instead.
    nbin_x1: int | None = None
    nbin_c: int | None = None
    #: Additional systematic covariances as (label, uncertainty scale, file).
    extra_covs: Sequence[tuple[str, float, Path]] = ()
    #: Event identifiers of calibrators, whose redshift systematics are excluded.
    calibrators: Sequence[str] = ()
    #: Write a Hubble diagram for each systematic variation.
    systematic_hd: bool | None = None
    #: Remove velocity variance while preserving the BBC error-rescaling factors.
    subtract_vpec: bool | None = None
    #: Add event identifiers to text covariance rows when available.
    label_cov_rows: bool | None = None
    #: Maximum matrix size for the output positive-definiteness check.
    mxsize_test_posdef: int | None = None

    def __post_init__(self):
        self.input_file = None if self.input_file is None else Path(self.input_file)
        self.extra_covs = tuple((label, float(scale), Path(path))
                                for label, scale, path in self.extra_covs)
        if self.mode is not None and self.mode not in get_args(HDMode):
            raise SNANAError(f"stage '{self.name}' has mode {self.mode!r}; "
                             f"create_covariance builds one of "
                             f"{', '.join(get_args(HDMode))}",
                             function_name=self.name)

    def cache_files(self) -> Sequence[Path]:
        _, _, extra_covs = self.configuration()
        return tuple(Path(path) for _, _, path in extra_covs)

    def inputs(self) -> Mapping[str, Stage]:
        return {"distances": self.distances}

    def systematics(self) -> tuple[Sequence[str], Sequence[str]]:
        return self.distances.systematics()

    def configuration(self):
        """Merge all supplied settings before validating the covariance."""
        fitopts, muopts = self.systematics()
        unknown = sorted(set(self.scales) - set(fitopts) - set(muopts))
        if unknown:
            raise SNANAError(f"stage '{self.name}' scales {unknown}, which name no "
                             f"systematic of {self.distances.name}", function_name=self.name)
        keys = ({} if self.input_file is None
                else yaml.safe_load(self.input_file.read_text()) or {})
        settings = dict(self.overrides)
        for name, value in (
                ('mode', self.mode), ('rebin_bins', self.rebin_bins),
                ('nbin_x1', self.nbin_x1), ('nbin_c', self.nbin_c),
                ('systematic_hd', self.systematic_hd), ('subtract_vpec', self.subtract_vpec),
                ('label_cov_rows', self.label_cov_rows), ('mxsize_test_posdef', self.mxsize_test_posdef)):
            if value is not None:
                settings = merge_settings(settings, {name: value})
        if self.covopts:
            settings['covopts'] = [parse_covopt(text) for text in self.covopts]
        if self.calibrators:
            settings['calibrators'] = list(self.calibrators)
        if self.extra_covs:
            settings['extra_covs'] = [(label, scale, str(path))
                                      for label, scale, path in self.extra_covs]
        # Use compact covariance throughout new pipelines unless explicitly
        # configured otherwise. Standalone create_cov retains its legacy default.
        keys = merge_settings({"WRITE_MASK_COV": 8}, keys)
        config = build_covariance_config(keys, settings)
        unusable = [name for name, value in (("sys_scale_file", config.sys_scale_file),
                                            ("sys_scales", config.sys_scales)) if value]
        if unusable:
            raise SNANAError(
                f"stage '{self.name}' cannot honour {', '.join(unusable)} from "
                f"{self.input_file}; use the stage's scales setting",
                function_name=self.name)
        calibrators = [str(cid) for cid in config.calibrators]
        extra_covs = [(label, scale, str(path)) for label, scale, path in config.extra_covs]
        select_covariance_grid({fitopt_muopt_name(cell): None for cell in self.distances.grid_cells()},
                               config.muopt)
        if not config.write_mask_cov & 10:
            raise SNANAError("WRITE_MASK_COV must include 2 or 8", function_name=self.name)
        return config, calibrators, extra_covs

    def run(self, inputs: Mapping[str, Any], outdir: Path) -> Mapping[str, Any]:
        config, _, _ = self.configuration()
        fitopts, muopts = self.systematics()
        mode = config.mode
        selected = select_covariance_grid(
            {fitopt_muopt_name(cell): cell for cell in self.distances.grid_cells()}, config.muopt)
        # binned reads the M0DIF, unbinned and rebin the FITRES
        # (get_hubble_diagrams, create_covariance.py:713-717)
        table = "M0DIF" if mode == "binned" else "FITRES"
        # the builder wants the grid by SNANA's numbered cell names
        diagrams = {
            fitopt_muopt_name(cell): hubble_diagram_from(
                inputs["distances"][key(cell, fitopts, muopts)][table].df.copy(),
                mode, source=f"{self.distances.name}:{fitopt_muopt_name(cell)}")
            for cell in selected.values()
        }
        config = dataclasses.replace(
            config, outdir=Path(outdir),
            input_dir=Path(outdir).parent, version=self.distances.name)
        reference = inputs["distances"][key((0, max(0, config.muopt)), fitopts, muopts)]
        config = dataclasses.replace(config, cosmology=reference_cosmology(
            config.cosmology, reference["M0DIF"].meta))
        result = prepare_covariance(
            config, diagrams, scales(fitopts, self.scales), scales(muopts, self.scales),
            isdata_real=int(reference["M0DIF"].meta["ISDATA_REAL"]))
        result.version_photometry = version_photometry(reference["FITRES"].meta)
        write_outputs(Path(outdir), result, config)
        return self.load(inputs, outdir)

    def load(self, inputs: Mapping[str, Any], outdir: Path) -> Mapping[str, Any]:
        info = yaml.safe_load((Path(outdir) / "INFO.YML").read_text())
        covariances = {}
        diagram = read_fitres(str(Path(outdir) / info["HD"])).df
        for entry in info["COVOPTS"].values():
            fields = entry.split()
            if len(fields) not in (3, 4):
                raise SNANAError(f"invalid COVOPTS entry: {entry!r}", function_name=self.name)
            label, _, inverse_name = fields[:3]
            factor_name = fields[3] if len(fields) == 4 else "None"
            if factor_name != "None":
                covariances[label] = {"kind": "factorized", **read_factorized_covariance(
                    Path(outdir) / factor_name, size=len(diagram))}
            elif inverse_name != "None":
                covariances[label] = {"kind": "total_inverse",
                    "matrix": read_covariance(Path(outdir) / inverse_name)}
            else:
                raise SNANAError(
                    f"stage '{self.name}' wrote neither factors nor inverse for {label}; "
                    "WRITE_MASK_COV must include 2 or 8", function_name=self.name)
        return {"hubble_diagram": diagram, "covariances": covariances,
                "isdata_real": info["ISDATA_REAL"]}


@dataclass(eq=False)
class Cosmology(Stage):
    """Cosmology fits of one covariance, one per covopt, in one wfit call.

    Returns ``{covopt: result}``, each the fit's own summary as written to
    its YAML, so a blinded fit loads back exactly as it ran. Simulation truth
    uses wfit's ``om_sim``/``w0_sim``/``wa_sim`` or ``truth_info`` settings;
    a list of truth files follows ``covopts`` order.
    """

    name: str
    covariance: Stage
    input_file: Path | None = None
    overrides: Mapping[str, Any] = field(default_factory=dict)
    covopts: Sequence[str] = ("ALL",)

    def __post_init__(self):
        self.input_file = None if self.input_file is None else Path(self.input_file)

    def inputs(self) -> Mapping[str, Stage]:
        return {"covariance": self.covariance}

    def cache_files(self) -> Sequence[Path]:
        keys = {} if self.input_file is None else read_cosmology_keys(str(self.input_file))
        config = build_cosmology_config({**keys, **self.overrides})
        truth = config.truth_info
        if truth is None:
            return ()
        return (Path(truth),) if isinstance(truth, str) else tuple(map(Path, truth))

    def _summary(self, outdir: Path) -> Path:
        return Path(outdir) / f"{self.name}.wfit.yaml"

    def run(self, inputs: Mapping[str, Any], outdir: Path) -> Mapping[str, Any]:
        keys = {} if self.input_file is None else read_cosmology_keys(str(self.input_file))
        # name wfit's two unconditional outputs for this stage
        named = {"cospar_yaml": str(self._summary(outdir)),
                 "outfile_cospar": str(Path(outdir) / f"{self.name}.cospar")}
        taken = sorted(named.keys() & (keys.keys() | self.overrides.keys()))
        if taken:
            raise SNANAError(f"stage '{self.name}' names its own outputs; "
                             f"drop {taken}", function_name=self.name)
        config = build_cosmology_config({**keys, **self.overrides, **named})
        held = inputs["covariance"]
        covariances = held["covariances"]
        absent = [c for c in self.covopts if c not in covariances]
        if absent:
            raise SNANAError(f"stage '{self.name}' asks for covopts {absent}; "
                             f"the covariance has {sorted(covariances)}",
                             function_name=self.name)
        truths = load_simulation_truths(config, len(self.covopts))
        diagrams = [hubble_diagram_from_covariance(
                        held["hubble_diagram"], covariances[covopt],
                        held["isdata_real"], source=f"{self.covariance.name}:{covopt}",
                        truth=truth)
                    for covopt, truth in zip(self.covopts, truths)]
        run_wfit(config, diagrams=diagrams)
        return self.load(inputs, outdir)

    def load(self, inputs: Mapping[str, Any], outdir: Path) -> Mapping[str, Any]:
        payload = yaml.safe_load(self._summary(outdir).read_text())
        realisations = payload["realisations"] if "realisations" in payload else [payload]
        return dict(zip(self.covopts, realisations))
