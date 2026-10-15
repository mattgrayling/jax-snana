"""What a classifier is to the pipeline.

A classifier is a dataclass with a ``column``, the name of the Type Ia
probability it produces (``PROB_<name>``, as Pippin names them), and

    predict(fit, photometry, model) -> pandas.Series or pandas.DataFrame

giving that probability for every event, indexed by CID as a string: a
Series named ``column``, or a DataFrame whose columns include ``column`` and
whatever else the classifier reports, such as its error. ``fit``
is the nominal fitted table, ``photometry`` the data version's directory,
and ``model`` a path to a trained model; each is None where the stage was not
given it. A classifier that learns from data also has

    train(fit, photometry, outdir) -> Path

returning the model it wrote under ``outdir``. Being a dataclass is what lets
a run tell whether a classifier's settings changed. Third-party classifiers
plug in through an adapter of this shape, as
:class:`~jax_pipeline.classifiers.supernnova.SuperNNova` does; the toy
:class:`~jax_pipeline.classifiers.perfect.Perfect` is for development.
"""

from typing import Protocol

import pandas as pd


class Classifier(Protocol):
    column: str

    def predict(self, fit, photometry, model) -> pd.Series | pd.DataFrame:
        ...
