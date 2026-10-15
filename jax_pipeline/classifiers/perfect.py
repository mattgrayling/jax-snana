"""A classifier that reads the simulation's truth."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from jax_snlc_core.utils.error_handling import SNANAError


@dataclass(frozen=True)
class Perfect:
    """Probabilities from the simulation truth: ``prob_ia`` for every true
    type Ia, ``prob_cc`` for other simulated types.

    Use this classifier to validate a pipeline with known simulated types.
    Its training method saves the two probabilities in a model file for
    the classification stage.
    """

    prob_ia: float = 1.0
    prob_cc: float = 0.0
    column: str = "PROB_PERFECT"

    def train(self, fit, photometry, outdir) -> Path:
        model = Path(outdir) / "perfect.model"
        model.write_text(f"{self.prob_ia} {self.prob_cc}\n")
        return model

    def predict(self, fit, photometry, model) -> pd.Series:
        if fit is None or "SIM_TEMPLATE_INDEX" not in fit.df:
            raise SNANAError("Perfect reads SIM_TEMPLATE_INDEX from a fitted "
                             "simulation; there is none here",
                             function_name="Perfect")
        prob_ia, prob_cc = ((self.prob_ia, self.prob_cc) if model is None
                            else (float(x) for x in Path(model).read_text().split()))
        # SIM_TEMPLATE_INDEX is 0 for a Type Ia model (SNANA's convention,
        # which BBC's own force_pIa=perfect reads the same way)
        is_ia = fit.df["SIM_TEMPLATE_INDEX"].astype(int) == 0
        return pd.Series(np.where(is_ia, prob_ia, prob_cc),
                         index=fit.df["CID"].astype(str), name=self.column)
