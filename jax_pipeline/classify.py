"""The classification stages: training a classifier, classifying a fit's
events so its tables carry a Type Ia probability, and merging several
classifications of one fit.

Pippin runs a classifier, aggregates its predictions and merges them onto
every FITRES as separate tasks because each is a program handing files to
the next. Here the fitted tables are objects, so a classification predicts
once, writes the predictions, and hands on the fit's tables with its
probability column joined. A distance fit consumes it exactly as it
consumes a fit. One that needs the columns of several classifications of a
fit consumes their merge.

Classifiers are dataclasses with a probability-column name and a
``predict(fit, photometry, model)`` method returning probabilities indexed
by CID. Trainable classifiers also implement ``train(fit, photometry, outdir)``
and return the saved model path. The dataclass fields participate in the
pipeline cache key.
"""

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from jax_snlc_core.logging import SNANAError

from .stages import FitVersion, Stage, one_version
from .supernnova import SuperNNova
from .systematics import NOMINAL

PREDICTIONS = "predictions.csv"
MODEL = "MODEL"


@dataclass(eq=False)
class Train(Stage):
    """Trains a classifier on a fitted table, a data version, or both.

    Hands on the path of the model it wrote.
    """

    name: str
    classifier: Any
    fit: Stage | FitVersion | None = None
    photometry: Stage | None = None

    def __post_init__(self):
        self.fit = one_version(self.fit)
        if not hasattr(self.classifier, "train"):
            raise SNANAError(f"stage '{self.name}': {type(self.classifier).__name__} "
                             f"cannot be trained", function_name=self.name)

    def inputs(self) -> Mapping[str, Stage]:
        return {role: stage
                for role, stage in (("fit", self.fit), ("photometry", self.photometry))
                if stage is not None}

    def run(self, inputs: Mapping[str, Any], outdir: Path) -> Path:
        model = self.classifier.train(
            None if self.fit is None else inputs["fit"][NOMINAL],
            None if self.photometry is None else inputs["photometry"],
            Path(outdir))
        (Path(outdir) / MODEL).write_text(str(model))
        return self.load(inputs, outdir)

    def load(self, inputs: Mapping[str, Any], outdir: Path) -> Path:
        return Path((Path(outdir) / MODEL).read_text())


@dataclass(eq=False)
class Classify(Stage):
    """The events of a fit classified, its tables handed on with the
    probability columns joined onto every cell, as Pippin's MERGE does.

    The predictions are made here by ``classifier``, or read from
    ``predictions``, a CSV made elsewhere such as by an earlier Pippin run,
    whose first column is the event's ID (``CID``, or ``SNID`` as Pippin
    writes it). ``model`` is a :class:`Train` stage or the path of a trained
    model, or None for a classifier that needs none. A classifier predicts
    once, on the nominal cell, and its predictions are written to
    ``predictions.csv``; an event with no prediction gets 0.0, as Pippin's
    merge gives it.
    """

    name: str
    classifier: Any = None
    fit: Stage | FitVersion = field(kw_only=True)
    model: Any = None
    photometry: Stage | None = None
    predictions: Path | None = None

    def __post_init__(self):
        self.fit = one_version(self.fit)
        if isinstance(self.model, str):
            self.model = Path(self.model)
        if self.predictions is not None:
            self.predictions = Path(self.predictions)
        if (self.classifier is None) == (self.predictions is None):
            raise SNANAError(f"stage '{self.name}' takes a classifier or the "
                             f"predictions, one of the two", function_name=self.name)
        if self.predictions is not None and (self.model, self.photometry) != (None, None):
            raise SNANAError(f"stage '{self.name}' reads its predictions, so takes "
                             f"no model or photometry", function_name=self.name)
        if hasattr(self.classifier, "train") and self.model is None:
            raise SNANAError(f"stage '{self.name}': {type(self.classifier).__name__} is trained, "
                             f"so predicting needs a model", function_name=self.name)

    def inputs(self) -> Mapping[str, Stage]:
        named = {"fit": self.fit}
        if isinstance(self.model, Stage):
            named["model"] = self.model
        if self.photometry is not None:
            named["photometry"] = self.photometry
        return named

    def event_sync(self) -> bool:
        return self.fit.event_sync()

    def systematics(self) -> tuple[Sequence[str], Sequence[str]]:
        return self.fit.systematics()

    def run(self, inputs: Mapping[str, Any], outdir: Path) -> Mapping[str, Any]:
        if self.predictions is not None:
            return self.load(inputs, outdir)
        model = inputs["model"] if isinstance(self.model, Stage) else self.model
        # a Series is one column, named by the Series
        predictions = pd.DataFrame(self.classifier.predict(
            inputs["fit"][NOMINAL],
            None if self.photometry is None else inputs["photometry"],
            model))
        predictions.index = predictions.index.astype(str)
        predictions.to_csv(Path(outdir) / PREDICTIONS, index_label="CID")
        return self.load(inputs, outdir)

    def load(self, inputs: Mapping[str, Any], outdir: Path) -> Mapping[str, Any]:
        return _joined_onto(inputs["fit"], self.predictions or Path(outdir) / PREDICTIONS)


@dataclass(eq=False)
class Merge(Stage):
    """One fit's tables with the probability columns of several of its
    classifications, as Pippin's MERGE gives the distance fit every
    classifier's column to choose from with ``varname_pIa``. As each
    classification adds columns of its own, two writing the same column
    are refused."""

    name: str
    classifications: Sequence[Classify]

    def __post_init__(self):
        self.classifications = tuple(self.classifications)
        fit = self.classifications[0].fit
        for classify in self.classifications[1:]:
            if classify.fit != fit:
                raise SNANAError(f"stage '{self.name}': {classify.name} classifies {classify.fit.name}, "
                                 f"not {fit.name}", function_name=self.name)

    def inputs(self) -> Mapping[str, Stage]:
        named = {"fit": self.classifications[0].fit}
        for classify in self.classifications:
            named[f"classification:{classify.name}"] = classify
        return named

    def event_sync(self) -> bool:
        return self.classifications[0].event_sync()

    def systematics(self) -> tuple[Sequence[str], Sequence[str]]:
        return self.classifications[0].systematics()

    def run(self, inputs: Mapping[str, Any], outdir: Path) -> Mapping[str, Any]:
        return self.load(inputs, outdir)

    def load(self, inputs: Mapping[str, Any], outdir: Path) -> Mapping[str, Any]:
        merged = {}
        for cell, fitted in inputs["fit"].items():
            table = fitted
            for classify in self.classifications:
                classified = inputs[f"classification:{classify.name}"][cell]
                columns = [name for name in classified.varnames if name not in fitted.varnames]
                cids = classified.df["CID"].astype(str)
                predictions = classified.df[columns].set_axis(cids)
                table = _joined(table, predictions)
            merged[cell] = table
        return merged


def _joined_onto(tables: Mapping[str, Any], path: Path) -> Mapping[str, Any]:
    """Every cell's table with the predictions in ``path`` joined on, the
    file's first column the event's ID."""
    predictions = pd.read_csv(path, dtype=str)
    predictions = predictions.set_index(predictions.columns[0]).astype(float)
    return {cell: _joined(table, predictions) for cell, table in tables.items()}


def _joined(table, predictions: pd.DataFrame):
    """The table with the prediction columns added, every row filled. A
    column the table already has is refused."""
    cids = table.df["CID"].astype(str)
    columns = list(predictions.columns)
    existing = [column for column in columns if column in table.varnames]
    if existing:
        raise SNANAError(f"the table already has {existing}; a classification adds "
                         f"columns of its own", function_name="classify")
    return dataclasses.replace(
        table,
        df=table.df.assign(**{c: cids.map(predictions[c]).fillna(0.0).to_numpy()
                              for c in columns}),
        varnames=table.varnames + columns,
        varnames_by_file=[names + columns for names in table.varnames_by_file])


@dataclass(frozen=True)
class Perfect:
    """Probabilities from each event's known type: ``prob_ia`` where its
    ``TYPE`` is one of ``ia_types``, ``prob_cc`` otherwise, as Pippin's
    PerfectClassifier gives them.

    Use this classifier to validate a pipeline with simulated or
    spectroscopically typed events. It needs no model, so it cannot be
    trained.
    """

    ia_types: tuple[int, ...]
    prob_ia: float = 1.0
    prob_cc: float = 0.0
    column: str = "PROB_PERFECT"

    #: the OPTS of Pippin's PerfectClassifier read by from_pippin
    PIPPIN_OPTS: ClassVar[tuple[str, ...]] = ("PROB_IA", "PROB_CC")

    @classmethod
    def from_pippin(cls, name, opts, types, locate) -> "Perfect":
        """The classifier a Pippin PerfectClassifier task's OPTS describe,
        with Pippin's default probabilities, whose Ia types are those
        ``types`` labels Ia and whose column is PROB_<task>, as in Pippin
        (perfect.py:63-69 there)."""
        if types is None:
            raise SNANAError(f"CLASSIFICATION {name}: the config.yml of the completed simulation "
                             f"records no OUTPUT.types", function_name="pippin")
        ia_types = tuple(sntype for sntype, label in types.items() if label == "Ia")
        prob_ia = opts.get("PROB_IA", 1.0)
        prob_cc = opts.get("PROB_CC", 0.0)
        return cls(ia_types=ia_types, prob_ia=prob_ia, prob_cc=prob_cc, column=f"PROB_{name}")

    def predict(self, fit, photometry, model) -> pd.Series:
        if "TYPE" not in fit.df:
            raise SNANAError("Perfect reads TYPE from a fitted table; there is none here",
                             function_name="Perfect")
        is_ia = fit.df["TYPE"].astype(int).isin(self.ia_types)
        return pd.Series(np.where(is_ia, self.prob_ia, self.prob_cc),
                         index=fit.df["CID"].astype(str), name=self.column)


#: each classifier a Pippin CLASSIFICATION task can name, by that name
CLASSIFIER_REGISTRY = {"PerfectClassifier": Perfect, "SuperNNovaClassifier": SuperNNova}


def register_classifier(name: str, classifier_cls) -> None:
    """Let Pippin CLASSIFICATION tasks name ``classifier_cls`` as ``name``."""
    if name in CLASSIFIER_REGISTRY:
        raise SNANAError(f"classifier {name!r} is already registered",
                         function_name="register_classifier")
    CLASSIFIER_REGISTRY[name] = classifier_cls
