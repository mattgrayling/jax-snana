"""Tabular cosmology results, using the values written by WFIT."""

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from .stages import Cosmology, Stage


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
