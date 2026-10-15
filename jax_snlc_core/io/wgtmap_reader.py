"""SNANA WGTMAP reader (read_WGTMAP, sntools_wgtmap.c): N-dimensional weight
grids for preferential selection, serving both HOSTLIB galaxy selection and
SIMSED surface selection.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import jax

from jax_snlc_core.io.host_properties import canonical_property_name
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

from jax_snlc_core.io.fits_path import open_text, resolve_fits_path
from jax_snlc_core.utils.error_handling import SNANAError
from jax_snlc_core.utils import snana_log as slog

logger = logging.getLogger(__name__)


@dataclass
class WgtMapData:
    """Parsed WGTMAP data: the N-dimensional weight grid, its axes, and
    which dimensions are SN variables (e.g. x1, c) rather than catalogue
    columns."""
    ndim: int
    varnames: List[str]
    nbins: np.ndarray  # Shape: (ndim,)
    ranges: np.ndarray  # Shape: (ndim, 2)
    wgt: jnp.ndarray
    snmagshift: Optional[jnp.ndarray]
    opt_extrap: int  # 0 = abort on out-of-bounds, 1 = clamp to edge
    snvar_names: List[str]
    snvar_indices: List[int]


def _parse(filepath: str) -> Tuple[List[str], List[str], np.ndarray]:
    """One pass over the file: (dimension names, function names, data rows).

    Dimension variables are the VARNAMES columns before the WGT token and
    function variables (WGT, SNMAGSHIFT, ...) those from it on, matched
    positionally as SNANA does (read_WGTMAP, sntools_wgtmap.c:78-92); both
    VARNAMES: and VARNAMES_WGTMAP: spellings are accepted (:35).
    """
    varnames, fun_names, rows = None, None, []
    with open_text(filepath) as f:
        for line in f:
            toks = line.split('#')[0].split()
            if not toks:
                continue
            if toks[0] in ('VARNAMES:', 'VARNAMES_WGTMAP:'):
                names = toks[1:]
                iwgt = names.index('WGT') if 'WGT' in names else len(names)
                varnames = [canonical_property_name(n) for n in names[:iwgt]]
                fun_names = names[iwgt:]
            elif toks[0] == 'WGT:' and varnames is not None:
                rows.append([float(x) for x in toks[1:]])

    if varnames is None:
        raise SNANAError(f"VARNAMES_WGTMAP line not found in {filepath}",
                         function_name="read_wgtmap")
    if len(varnames) == 0 or len(fun_names) == 0:
        raise SNANAError(
            f"Invalid VARNAMES_WGTMAP; NDIM={len(varnames)} NFUN={len(fun_names)}",
            function_name="read_wgtmap")

    data = np.array(rows, dtype=np.float64)
    ncol = len(varnames) + len(fun_names)
    if data.ndim != 2 or data.shape[1] != ncol:
        raise SNANAError(
            f"WGTMAP rows must have {ncol} values ({len(varnames)} dimensions + "
            f"{len(fun_names)} functions); got "
            f"{data.shape[1] if data.ndim == 2 else 'ragged rows'}",
            function_name="read_wgtmap")
    return varnames, fun_names, data


def read_wgtmap_varnames(filepath: str) -> List[str]:
    """The WGTMAP's dimension variable names, without loading the grid
    (read_VARNAMES_WGTMAP's role in SNANA)."""
    varnames, _, _ = _parse(resolve_fits_path(filepath, required=True))
    return varnames


def read_wgtmap(filepath: str, sn_variables, opt_extrap: int = 0) -> WgtMapData:
    """Read and parse a WGTMAP file (``filepath``, .gz or plain text) into a
    :class:`WgtMapData`.

    ``sn_variables`` maps each lower-cased spelling of a per-SN quantity,
    rather than a catalogue column, to the name the caller stores it under;
    a dimension it spells takes that name. The caller derives it from its
    model (checkSNvar_HOSTLIB_WGTMAP, sntools_host.c:2868).
    ``opt_extrap`` is 0 to abort if any galaxy is out of bounds (SNANA
    default) or 1 to clamp out-of-bounds values to the grid edge.
    """
    filepath = resolve_fits_path(filepath, required=True)
    logger.info(f"Reading WGTMAP file: {filepath}")

    varnames, fun_names, data = _parse(filepath)
    ndim = len(varnames)
    slog.line(logger, f"Dimensions: {ndim} ({', '.join(varnames)})")
    slog.line(logger, f"Loaded {len(data)} grid points")

    # uniform bins, tolerance 1e-6, as read_GRIDMAP
    axes = [np.unique(data[:, i]) for i in range(ndim)]
    for name, ax in zip(varnames, axes):
        d = np.diff(ax)
        if len(d) and np.any(np.abs(d - d[0]) / d > 1.0e-6):
            raise SNANAError(
                f"non-uniform bin in WGTMAP dimension '{name}'",
                function_name="read_wgtmap")

    nbins = np.array([len(ax) for ax in axes], dtype=np.int32)
    ranges = np.array([[ax[0], ax[-1]] for ax in axes])

    # every cell exactly once, in any row order
    idx = tuple(np.searchsorted(axes[i], data[:, i]) for i in range(ndim))
    counts = np.zeros(tuple(nbins), dtype=np.int64)
    np.add.at(counts, idx, 1)
    if np.any(counts != 1):
        cell = np.argwhere(counts != 1)[0]
        coords = ', '.join(f"{v}={axes[i][cell[i]]:g}"
                           for i, v in enumerate(varnames))
        raise SNANAError(
            f"WGTMAP cell at {coords} appears "
            f"{int(counts[tuple(cell)])} times; every grid cell must be "
            f"tabulated exactly once.", function_name="read_wgtmap")

    wgt_grid = np.zeros(tuple(nbins))
    wgt_grid[idx] = data[:, ndim]

    # an all-zero SNMAGSHIFT means unused
    snmagshift_grid = None
    if 'SNMAGSHIFT' in fun_names:
        vals = data[:, ndim + fun_names.index('SNMAGSHIFT')]
        if not np.allclose(vals, 0.0):
            snmagshift_grid = np.zeros(tuple(nbins))
            snmagshift_grid[idx] = vals
            snmagshift_grid = jnp.array(snmagshift_grid, dtype=jnp.float32)

    # float64 against underflow
    wgt_grid = jnp.array(wgt_grid, dtype=jnp.float64)
    slog.line(logger, f"Grid shape: {' x '.join(map(str, nbins))}, weight range: "
              f"{float(jnp.min(wgt_grid)):.3e} to {float(jnp.max(wgt_grid)):.3e}")

    varnames = [sn_variables.get(var.lower(), var) for var in varnames]
    snvar_indices = [i for i, var in enumerate(varnames)
                     if var.lower() in sn_variables]
    snvar_names = [varnames[i] for i in snvar_indices]
    if snvar_names:
        slog.line(logger, f"Detected SN variables: {snvar_names}")

    return WgtMapData(
        ndim=ndim,
        varnames=varnames,
        nbins=nbins,
        ranges=ranges,
        wgt=wgt_grid,
        snmagshift=snmagshift_grid,
        opt_extrap=opt_extrap,
        snvar_names=snvar_names,
        snvar_indices=snvar_indices,
    )


@jax.jit
def interpolate_wgtmap(
    wgt_grid: jnp.ndarray,
    grid_ranges: jnp.ndarray,  # (ndim, 2)
    grid_nbins: jnp.ndarray,   # (ndim,)
    galaxy_properties: jnp.ndarray  # (n_galaxies, ndim)
) -> jnp.ndarray:
    """N-dimensional weight interpolation via map_coordinates, vectorised
    over all galaxies at once, returning the interpolated weight per galaxy,
    shape (n_galaxies,).

    ``wgt_grid`` is the N-dimensional weight map; ``grid_ranges`` the
    min/max values for each dimension, shape (ndim, 2); ``grid_nbins`` the
    number of bins for each dimension, shape (ndim,); ``galaxy_properties``
    the physical properties for each galaxy, shape (n_gal, ndim), e.g.
    columns [LOGMASS, LOG_SFR] or [LOGMASS, LOG_SFR, x1].
    """
    # a single-bin axis maps to index 0
    min_ranges = grid_ranges[:, 0]
    span = grid_ranges[:, 1] - min_ranges
    scale = jnp.where(grid_nbins > 1,
                      (grid_nbins - 1) / jnp.where(span > 0.0, span, 1.0),
                      0.0)
    coords = (galaxy_properties - min_ranges) * scale
    return map_coordinates(wgt_grid, coordinates=coords.T, order=1,
                           mode='nearest')
