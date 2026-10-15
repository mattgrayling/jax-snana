"""The two classification stages: training a classifier, and classifying a
fit's events so its tables carry a Type Ia probability.

Pippin runs a classifier, aggregates its predictions and merges them onto
every FITRES as separate tasks because each is a program handing files to
the next. Here the fitted tables are objects, so classifying and merging is
one stage: it predicts once, writes the predictions, and hands on the fit's
tables with the probability column joined. A distance fit consumes it
exactly as it consumes a fit.
"""

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from jax_snlc_core.utils.error_handling import SNANAError

from .stages import FitVersion, Stage, one_version
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


def _joined_onto(tables: Mapping[str, Any], path: Path) -> Mapping[str, Any]:
    """Every cell's table with the predictions in ``path`` joined on, the
    file's first column the event's ID."""
    predictions = pd.read_csv(path, dtype=str)
    predictions = predictions.set_index(predictions.columns[0]).astype(float)
    return {cell: _joined(table, predictions) for cell, table in tables.items()}


def _joined(table, predictions: pd.DataFrame):
    """The table with the prediction columns added, every row filled."""
    cids = table.df["CID"].astype(str)
    columns = list(predictions.columns)
    return dataclasses.replace(
        table,
        df=table.df.assign(**{c: cids.map(predictions[c]).fillna(0.0).to_numpy()
                              for c in columns}),
        varnames=[*table.varnames, *columns],
        varnames_by_file=[[*names, *columns] for names in table.varnames_by_file])
