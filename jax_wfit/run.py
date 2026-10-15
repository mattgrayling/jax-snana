"""End-to-end WFIT orchestration."""

import hashlib
import logging
import time
from dataclasses import replace
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core.utils import SNANAError, snana_log as slog
from jax_snlc_core.utils.precision import require_precision

from .blinding import blinding_statuses
from .config import WFITConfig
from .data import (HubbleDiagram, load_hubble_diagrams, stack_arrays,
                   fit_indices, conditioned_precision, selected_covariance,
                   validate_prepared_precision)
from .inference import fit_grid, fit_laplace, fit_nuts
from .output import write_outputs
from .priors import bao_root, prepare_priors

logger = logging.getLogger(__name__)

#: Options that read a file column the caller's diagram does not carry: the
#: uncertainty column and its override, and the idealised-data rule, which
#: needs the simulation truth block (wfit.c:1583-1607).
_FILE_COLUMN_OPTIONS = ("muerr_force", "varname_muerr", "muerr_ideal")


def hd_seed(diagrams) -> int:
    """Per-dataset seed for -ranseed_Rcmb 1: SNANA seeds from the data so the
    shift is reproducible per dataset (ISEED_UNIQUE, wfit.c:2539). This seed
    cannot reproduce SNANA's value, only its per-dataset determinism."""
    digest = hashlib.sha256(
        b"".join(np.asarray(d.mu).tobytes() for d in diagrams)
    ).digest()
    return int.from_bytes(digest[:4], "big")


def _cut_supplied_diagram(config: WFITConfig, diagram: HubbleDiagram) -> HubbleDiagram:
    """The diagram a caller supplied, cut and conditioned as the file route
    cuts and conditions one it reads: the z window (wfit.c:1539), and snrms,
    neither of which an already-inverted covariance can take. The NFIT cut
    needs the file's NFIT column, as do the three options that abort here; a
    diagram in memory has none."""
    named = [name for name in _FILE_COLUMN_OPTIONS
             if getattr(config, name) != getattr(WFITConfig(), name)]
    if named:
        raise SNANAError(
            f"{', '.join(named)} read a column of the Hubble-diagram file, "
            "so they cannot apply to a diagram supplied in memory",
            function_name="run_wfit")
    kind = diagram.covariance_kind
    indices = fit_indices(config, diagram.redshift, diagram.mu, diagram.muerr, kind=kind)
    matrix = diagram.precision if kind == "total_inverse" else diagram.covariance
    if matrix is not None:
        matrix = selected_covariance(matrix, indices, diagram.n_rows)
    elif kind != "diagonal":
        raise SNANAError("Dense Hubble diagram requires its covariance")
    validate_prepared_precision(diagram.precision, kind, diagram.n_rows)
    if kind == "total_inverse" and (config.snrms != 0 or diagram.snrms != 0):
        raise SNANAError("snrms cannot modify an already-inverted covariance")
    if len(indices) == diagram.n_rows and config.snrms == diagram.snrms:
        return diagram
    precision, covariance = conditioned_precision(
        kind, np.asarray(diagram.muerr)[indices], matrix, config.snrms,
        previous_scatter=diagram.snrms,
        precision=np.asarray(diagram.precision)[indices] if kind == "diagonal" else None)
    def subset(values):
        return None if values is None else values[indices]

    return replace(
        diagram,
        cid=diagram.cid[indices],
        redshift=diagram.redshift[indices],
        mu=diagram.mu[indices],
        muerr=diagram.muerr[indices],
        precision=precision,
        covariance=covariance,
        snrms=config.snrms,
        raw_indices=(indices if diagram.raw_indices is None
                     else diagram.raw_indices[indices]),
        zerr=subset(diagram.zerr),
        mu_second=subset(diagram.mu_second),
        z_second=subset(diagram.z_second),
        bias0=subset(diagram.bias0),
        bias1=subset(diagram.bias1))


def run_wfit(
    config: WFITConfig,
    *,
    write: bool = True,
    diagrams: Sequence[HubbleDiagram] | None = None,
) -> tuple[dict[str, object], ...]:
    """Fit one or more independent HD realisations through one code path.

    ``diagrams`` are the realisations to fit; without them they are read from
    the files the config names, which is what a caller holding a covariance
    result (``data.hubble_diagram_from_covariance``) skips. The options that
    select file columns are refused on that route. Redshift cuts and
    ``snrms`` apply to supplied diagrams too; ``snrms`` replaces any extra
    scatter recorded on the diagram. Already-inverted covariances cannot
    take row-removing cuts or extra scatter.
    """
    require_precision(config.double_precision)
    t_start = time.monotonic()
    if diagrams is None:
        diagrams = load_hubble_diagrams(config)      # validates as it loads
    else:
        config.validate()
        diagrams = [_cut_supplied_diagram(config, d) for d in diagrams]
    diagrams = tuple(diagrams)
    if not diagrams:
        raise SNANAError('At least one Hubble diagram is required')
    if write:
        config.resolve_paths()
    blinded = blinding_statuses(config, diagrams)
    if config.perturb_priors and config.prior_perturbation_seed == 1:
        # the caller's config stays untouched (it is dumped as provenance)
        config = replace(config, prior_perturbation_seed=hd_seed(diagrams))
    slog.line(logger, f"Fitting {len(diagrams)} {config.model} realisation"
                      f"{'s' if len(diagrams) > 1 else ''} by {config.method}")
    priors = prepare_priors(config, [d.truth for d in diagrams],
                            bao_root() if config.bao else None)
    # External covariance fallbacks may coexist with compact COVOPTs. Batch
    # within each representation, preserving the caller's order in outputs.
    groups = {}
    for index, diagram in enumerate(diagrams):
        groups.setdefault(diagram.covariance_kind, []).append(index)
    results, grids = [None] * len(diagrams), [None] * len(diagrams)
    fitter = {"laplace": fit_laplace, "nuts": fit_nuts, "grid": fit_grid}[config.method]
    # Assign streams by original realisation index before format grouping.
    nuts_keys = None
    if config.method == "nuts":
        base_key = jax.random.fold_in(jax.random.PRNGKey(config.seed), 2)
        nuts_keys = jax.vmap(lambda i: jax.random.fold_in(base_key, i))(
            jnp.arange(len(diagrams)))
    for indices in groups.values():
        selected = tuple(diagrams[i] for i in indices)
        group_priors = {**priors, **{key: priors[key][np.asarray(indices)]
                                   for key in ("omega_m_mean", "cmb_r")}}
        sampling = {} if nuts_keys is None else {"keys": nuts_keys[jnp.asarray(indices)]}
        fitted, evaluated = fitter(selected, stack_arrays(selected), config,
                                   group_priors, [blinded[i] for i in indices], **sampling)
        for i, result, grid in zip(indices, fitted, evaluated):
            results[i], grids[i] = result, grid
    for index, diagram in enumerate(diagrams):
        if diagram.zerr is not None:
            results[index]["warnings"].append(
                "redshift uncertainties were read but are not propagated, "
                "matching WFIT"
            )
    if write:
        write_outputs(results, grids, config, diagrams,
                      cpu_minutes=(time.monotonic() - t_start) / 60.0)
        slog.line(logger, f"Wrote outputs beside {config.cospar_yaml}")
    return tuple(results)
