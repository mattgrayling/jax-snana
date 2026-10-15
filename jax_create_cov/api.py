"""Covariance construction from Python settings and Hubble-diagram grids."""

from collections.abc import Mapping

from jax_snlc_core.utils import SNANAError

from .config import CreateCovConfig
from .config_parser import build_config
from .hubble import hubble_diagram_from
from .inputs import apply_sys_scales, read_yaml
from .run import (create_covariance, prepare_covariance, select_covariance_grid,
                  write_outputs)


def compute_covariance(inputs=None, *, diagrams=None, fitopt_scales=None,
                       muopt_scales=None, isdata_real=-1, write=False, **settings):
    """Build systematic covariances using a mapping, YAML file or keyword settings.

    For file inputs, supply ``input_dir`` and ``version``. For in-memory
    inputs, ``diagrams`` maps FITOPT/MUOPT names to pandas DataFrames;
    ``fitopt_scales`` and ``muopt_scales`` map each option number to its
    ``(label, uncertainty_scale)``. Include option 0 as ``('DEFAULT', 1.0)``.
    ``sys_scales`` can override the uncertainty scales by label, as for files.

    ``inputs`` may also be a ``CreateCovConfig``. Keyword settings go through
    the configuration parser. Returns ``CovResult``, with ``hubble_diagram``
    and ``covariances`` keyed by covariance label. ``covtot_inv(label)``
    returns the inverse total covariance for that label. Files are written only with
    ``write=True``, which requires ``outdir``.
    """
    if isinstance(inputs, CreateCovConfig):
        values = inputs
    elif inputs is None or isinstance(inputs, Mapping):
        values = inputs or {}
    else:
        values = read_yaml(inputs)
    config = build_config(values, settings)
    if write and config.outdir is None:
        raise SNANAError('outdir is required to write covariance outputs')
    if diagrams is None:
        if fitopt_scales is not None or muopt_scales is not None:
            raise SNANAError('Option scale mappings require in-memory diagrams')
        if config.input_dir is None or not config.version:
            raise SNANAError('Supply diagrams, or input_dir and version')
        return create_covariance(config, write=write)
    if config.input_dir is not None or config.version:
        raise SNANAError('Supply an in-memory grid or input_dir/version, not both')
    if fitopt_scales is None or muopt_scales is None:
        raise SNANAError('In-memory diagrams require fitopt_scales and muopt_scales')
    if 0 not in fitopt_scales or 0 not in muopt_scales:
        raise SNANAError('Option scale mappings require the reference option 0')
    grid = {name: hubble_diagram_from(frame.copy(), config.mode, source=name)
            for name, frame in select_covariance_grid(diagrams, config.muopt).items()}
    result = prepare_covariance(
        config, grid, apply_sys_scales(fitopt_scales, config.sys_scales),
        apply_sys_scales(muopt_scales, config.sys_scales), isdata_real=isdata_real)
    if write:
        write_outputs(config.outdir, result, config)
    return result
