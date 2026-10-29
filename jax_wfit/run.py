"""End-to-end WFIT orchestration."""

import hashlib
import logging
import time
from collections.abc import Mapping
from dataclasses import replace
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core import logging as log
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.utils import require_precision

from .blinding import blinding_statuses
from .config import WFITConfig, build, read_keys
from .hubble import HubbleDiagram, cut_supplied_diagram, load_hubble_diagrams, stack_arrays
from .inference import fit_grid, fit_laplace, fit_nuts
from .output import write_outputs
from .priors import bao_root, prepare_priors

logger = logging.getLogger(__name__)


def hd_seed(diagrams) -> int:
    """Per-dataset seed for -ranseed_Rcmb 1: SNANA seeds from the data so the
    shift is reproducible per dataset (ISEED_UNIQUE, wfit.c:2539). This seed
    cannot reproduce SNANA's value, only its per-dataset determinism."""
    digest = hashlib.sha256(
        b"".join(np.asarray(d.mu).tobytes() for d in diagrams)
    ).digest()
    return int.from_bytes(digest[:4], "big")


def run_wfit(
    config: WFITConfig,
    *,
    write: bool = True,
    diagrams: Sequence[HubbleDiagram] | None = None,
) -> tuple[dict[str, object], ...]:
    """Fit one or more independent HD realisations through one code path.

    ``diagrams`` are the realisations to fit; without them they are read from
    the files the config names, which is what a caller holding a covariance
    result (``hubble.hubble_diagram_from_covariance``) skips. The options that
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
        diagrams = [cut_supplied_diagram(config, d) for d in diagrams]
    diagrams = tuple(diagrams)
    if not diagrams:
        raise SNANAError('At least one Hubble diagram is required')
    if write:
        config.resolve_paths()
    blinded = blinding_statuses(config, diagrams)
    if config.perturb_priors and config.prior_perturbation_seed == 1:
        # the caller's config stays untouched (it is dumped as provenance)
        config = replace(config, prior_perturbation_seed=hd_seed(diagrams))
    log.line(logger, f"Fitting {len(diagrams)} {config.model} realisation"
             f"{'s' if len(diagrams) > 1 else ''} by {config.method}")
    priors = prepare_priors(config, [d.truth for d in diagrams],
                            bao_root() if config.bao else None)
    # External covariance fallbacks may coexist with compact COVOPTs. Batch
    # within each representation, preserving the caller's order in outputs.
    groups = {}
    for index, diagram in enumerate(diagrams):
        groups.setdefault(diagram.covariance_kind, []).append(index)
    results, grids = [None] * len(diagrams), [None] * len(diagrams)
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
        arrays = stack_arrays(selected)
        group_blinded = [blinded[i] for i in indices]
        if config.method == "nuts":
            fitted, evaluated = fit_nuts(
                selected, arrays, config, group_priors, group_blinded,
                keys=nuts_keys[jnp.asarray(indices)])
        elif config.method == "laplace":
            fitted, evaluated = fit_laplace(
                selected, arrays, config, group_priors, group_blinded)
        else:
            fitted, evaluated = fit_grid(
                selected, arrays, config, group_priors, group_blinded)
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
        log.line(logger, f"Wrote outputs beside {config.cospar_yaml}")
    return tuple(results)


def fit_cosmology(inputs=None, *, diagrams=None, write=False, **settings):
    """Fit cosmology with keyword settings, a mapping or a YAML configuration.

    Supply ``hubble_diagram`` and optional covariance file settings, or pass
    a sequence of ``HubbleDiagram`` objects as ``diagrams``. ``inputs`` may
    also be a ``WFITConfig``. Keyword settings override the supplied settings without changing the
    caller's configuration.

    Returns one result dictionary per realisation, in a tuple. Each contains
    ``point_estimate``, ``summaries``, ``likelihood`` and ``diagnostics``.
    Files are written only with ``write=True``.
    """
    if isinstance(inputs, WFITConfig):
        values = inputs
    elif inputs is None or isinstance(inputs, Mapping):
        values = inputs or {}
    else:
        values = read_keys(inputs)
    config = build(values, settings)
    if diagrams is not None:
        if config.hubble_diagram or config.covariance:
            raise SNANAError('Supply diagrams or Hubble-diagram/covariance file settings, not both')
        if write and (config.cospar_yaml is None or config.outfile_cospar is None):
            raise SNANAError('In-memory fits require cospar_yaml and outfile_cospar to write outputs')
    return run_wfit(config, diagrams=diagrams, write=write)
