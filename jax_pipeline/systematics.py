"""The cells of the FITOPT x MUOPT grid, and what each applies.

A systematic is a label and the input keys it overrides, spelt as the stage's
input file spells them: a fit takes its FITOPTs and a distance fit its MUOPTs
as a dict of those, in order. SNANA keeps systematics out of the
executables: ``snlc_fit.exe`` and ``SALT2mu.exe`` know nothing of FITOPTs,
and ``submit_batch_jobs`` expands each into override tokens and launches one
job per cell. The same split holds here, so a package never sees a
systematic, only the config one resolved to.

A cell is the pair ``(fitopt, muopt)`` of indices into those dicts, zero
being the nominal on each axis. On disk it is SNANA's numbered directory; in
a stage's result it is keyed by what was applied.
"""

import re
from collections.abc import Mapping
from typing import Any, Sequence

from jax_snlc_core.logging import SNANAError

#: the key of the analysis with no systematic applied
NOMINAL = "nominal"


def cells(fitopts: Sequence, muopts: Sequence = ()) -> list[tuple[int, int]]:
    """Every cell of the grid, nominal first, in SNANA's numbering order."""
    return [(i, j)
            for i in range(len(fitopts) + 1)
            for j in range(len(muopts) + 1)]



def key(cell: tuple[int, int], fitopts: Sequence[str], muopts: Sequence[str] = ()):
    """What a cell's result is filed under: ``"nominal"``, the label of the
    one systematic applied, or the pair ``(fitopt, muopt)`` where both are."""
    fitopt, muopt = cell
    applied = []
    if fitopt:
        applied.append(list(fitopts)[fitopt - 1])
    if muopt:
        applied.append(list(muopts)[muopt - 1])
    if not applied:
        return NOMINAL
    return applied[0] if len(applied) == 1 else tuple(applied)


def check_labels(name: str, fitopts: Sequence[str], muopts: Sequence[str] = ()):
    """Labels key results, so none may be the nominal's name and none may
    sit on both axes."""
    taken = [label for label in (*fitopts, *muopts) if label == NOMINAL]
    shared = sorted(set(fitopts) & set(muopts))
    if taken or shared:
        raise SNANAError(f"stage '{name}': a systematic may not be called "
                         f"{NOMINAL!r}, and a label may not be on both axes; "
                         f"got {taken + shared}", function_name=name)


def overrides_for(systematics: Mapping[str, Mapping[str, Any]],
                  index: int) -> Mapping[str, Any]:
    """The overrides of the systematic at ``index`` in order, none at the
    nominal index 0."""
    return {} if index == 0 else list(systematics.values())[index - 1]


def scales(labels: Sequence[str],
           scale_of: Mapping[str, float]) -> dict[int, tuple[str, float]]:
    """``{index: (label, scale)}`` including the nominal, which is what
    ``jax_create_cov.build_covariance`` takes in place of SUBMIT.INFO. The
    nominal is labelled DEFAULT as create_covariance and Pippin label it; a
    label ``scale_of`` does not name scales by one, as an unlisted one does
    in create_covariance's scale file."""
    numbered = {0: ("DEFAULT", 1.0)}
    for index, label in enumerate(labels, start=1):
        numbered[index] = (label, float(scale_of[label]) if label in scale_of else 1.0)
    return numbered


def selected_cells(fitopts, muopts, selection=()):
    """Union of SNANA FITOPTxMUOPT expressions: i+j (OR), i&j (AND)."""
    requested = [selection] if isinstance(selection, str) else list(selection)
    requested = [item for item in requested if item != "DUMP"]
    grid = cells(fitopts, muopts)
    if not requested or requested == ["ALL"]:
        return grid
    rules = []
    for expression in requested:
        match = re.fullmatch(r"\s*(\d+)\s*([+&])\s*(\d+)\s*", str(expression))
        if not match:
            raise SNANAError(f"invalid FITOPTxMUOPT expression {expression!r}; use i+j or i&j", function_name="selected_cells")
        i, op, j = match.groups()
        i, j = int(i), int(j)
        if i > len(fitopts) or j > len(muopts):
            raise SNANAError(f"FITOPTxMUOPT {expression!r} names an absent FITOPT or MUOPT", function_name="selected_cells")
        rules.append((i, op, j))
    return [(i, j) for i, j in grid if any(
        (i == a or j == b) if op == "+" else (i == a and j == b)
        for a, op, b in rules)]
