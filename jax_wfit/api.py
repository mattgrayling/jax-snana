"""Cosmology fits from Python settings and prepared Hubble diagrams."""

from collections.abc import Mapping

from jax_snlc_core.utils import SNANAError

from .config import WFITConfig
from .config_parser import build, read_keys
from .run import run_wfit


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
