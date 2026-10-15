"""
WGTSUM (Weight Sum) computation for host galaxy selection.

This module computes interpolated weights for all galaxies from a WGTMAP
and builds cumulative sums (WGTSUM) for efficient inverse transform sampling.

Following SNANA, all galaxies are interpolated in one vectorised pass and the
cumulative sum is global rather than binned, so selection is a binary search.
"""

import logging
import time
import numpy as np
import jax
import jax.numpy as jnp
from typing import List
from dataclasses import dataclass

from jax_snlc_core.io.wgtmap_reader import interpolate_wgtmap
from jax_snlc_core.utils import SNANAError, snana_log as slog

logger = logging.getLogger(__name__)

# Expected flux = 0 (e.g. undetected band), sntools.h:124
MAG_ZEROFLUX = 99.0


@dataclass
class WgtSumData:
    """Container for computed WGTSUM data: cumulative weight sums for
    efficient host selection.

    For a 2D WGTMAP (galaxy properties only), ``wgtsum``/``weights`` have
    shape (1, n_galaxies): one global cumulative sum. For a 3D+ WGTMAP (with
    SN variables like x1, c), they have shape (n_snvar_bins, n_galaxies):
    one cumulative sum per SN variable bin, matching SNANA's pre-computed
    multi-bin approach. ``wgtsum`` is unnormalised: its last element is the
    total weight, not 1.0. ``snvar_bin_centers`` is
    None when the WGTMAP has no SN variables, else one array per SN
    variable dimension (a stretch, or a stretch and a colour) and
    ``snvar_names`` names them.
    """
    galaxy_indices: np.ndarray  # Shape: (n_galaxies,)
    wgtsum: np.ndarray          # Shape: (n_snvar_bins, n_galaxies)
    weights: np.ndarray         # Shape: (n_snvar_bins, n_galaxies)
    total_galaxies: int
    snvar_bin_centers: List[np.ndarray] = None  # one array per SN-variable dimension
    snvar_names: List[str] = None
    galaxy_snmagshift: np.ndarray = None         # Shape: (n_galaxies,) or (n_snvar_bins, n_galaxies)


@jax.jit
def _interpolate_weights_with_snvar(
    wgt_grid: jnp.ndarray,
    grid_ranges: jnp.ndarray,
    grid_nbins: jnp.ndarray,
    galaxy_properties: jnp.ndarray,  # Shape: (n_galaxies, n_galaxy_dims)
    snvar_bin_centers: jnp.ndarray,  # Shape: (n_snvar_bins_total, n_snvar_dims)
    snvar_indices: jnp.ndarray,
    galaxy_indices: jnp.ndarray
) -> jnp.ndarray:
    """Vectorised interpolation over all SN variable bins using jax.vmap.

    For a 3D WGTMAP [LOGMASS, LOGSFR, x1] with 9 x1 bins, computes weights
    for all 9 bins simultaneously using vmap, returning shape
    (9, n_galaxies) instead of looping 9 times.

    ``wgt_grid`` is the N-dimensional weight map; ``grid_ranges``/
    ``grid_nbins`` give the min/max and bin count per dimension, shape
    (ndim, 2)/(ndim,). ``galaxy_properties`` is (n_galaxies, n_galaxy_dims),
    the galaxy properties only; ``snvar_bin_centers`` is (n_bins_total,
    n_snvar_dims), all SN bin centre combinations. ``snvar_indices``/
    ``galaxy_indices`` place the SN and galaxy dimensions within the full
    dimension list, shape (n_snvar_dims,)/(n_galaxy_dims,).

    Returns the weights for all bins and galaxies, shape
    (n_snvar_bins_total, n_galaxies).
    """

    def interpolate_single_bin(snvar_values):
        """Interpolate for one SN bin (vmapped over bins)."""
        n_galaxies = galaxy_properties.shape[0]
        ndim_total = wgt_grid.ndim
        full_properties = jnp.zeros((n_galaxies, ndim_total), dtype=jnp.float32)
        full_properties = full_properties.at[:, galaxy_indices].set(galaxy_properties)
        snvar_broadcast = jnp.broadcast_to(snvar_values, (n_galaxies, len(snvar_values)))
        full_properties = full_properties.at[:, snvar_indices].set(snvar_broadcast)
        return interpolate_wgtmap(wgt_grid, grid_ranges, grid_nbins, full_properties)

    interp = jax.vmap(interpolate_single_bin, in_axes=0)
    return interp(snvar_bin_centers)


def compute_wgtsum(table, wgtmap_data) -> WgtSumData:
    """Compute WGTSUM for all galaxies of ``table`` (a HostlibTable) from
    ``wgtmap_data`` (a WgtMapData).

    For a 2D WGTMAP (galaxy properties only) it is a single global
    cumulative sum for all galaxies; for a 3D+ WGTMAP (with SN variables) it pre-computes a WGTSUM
    per SN variable bin, returning shape (n_snvar_bins, n_galaxies).

    Returns the WgtSumData containing the cumulative sum(s).
    """
    logger.info(f"Computing WGTSUM for {table.ngal:,} galaxies...")
    t_start = time.time()
    snvar_idx = wgtmap_data.snvar_indices
    galaxy_idx = [i for i in range(wgtmap_data.ndim) if i not in snvar_idx]
    galaxy_varnames = [wgtmap_data.varnames[i] for i in galaxy_idx]
    missing_vars = [v for v in galaxy_varnames if v not in table.columns]
    if missing_vars:
        raise SNANAError(
            f"WGTMAP requires galaxy variables {galaxy_varnames} but "
            f"HOSTLIB is missing: {missing_vars}"
        )

    # one bin per combination of SN-variable bin centres; a map with no SN
    # variables is one bin with none
    centers = [np.linspace(*wgtmap_data.ranges[i], wgtmap_data.nbins[i])
               for i in snvar_idx]
    snvar_grid = (np.stack([g.ravel() for g in np.meshgrid(*centers, indexing='ij')],
                           axis=-1) if centers else np.zeros((1, 0)))
    n_snvar_bins = snvar_grid.shape[0]

    galaxy_properties = jnp.stack(
        [table.columns[var] for var in galaxy_varnames], axis=-1)
    # galaxy dims only; SN dims are checked per event (sntools_host.c:1655-1660, :4330-4352)
    galaxy_ranges = np.asarray(wgtmap_data.ranges)[galaxy_idx]
    galaxy_properties = _apply_magobs_zeroflux(
        galaxy_properties, galaxy_varnames, galaxy_ranges)
    _check_bounds(galaxy_properties, galaxy_varnames, galaxy_ranges,
                  wgtmap_data.opt_extrap)

    def interpolate(grid):
        return _interpolate_weights_with_snvar(
            grid,
            jnp.array(wgtmap_data.ranges, dtype=jnp.float32),
            jnp.array(wgtmap_data.nbins, dtype=jnp.float32),
            galaxy_properties,
            jnp.array(snvar_grid, dtype=jnp.float32),
            jnp.array(snvar_idx, dtype=jnp.int32),
            jnp.array(galaxy_idx, dtype=jnp.int32))

    slog.line(logger, f"Interpolating WGTMAP for {n_snvar_bins} SN bin(s) × "
              f"{table.ngal:,} galaxies...")
    weights = interpolate(wgtmap_data.wgt)
    # per galaxy and SN bin, as SNANA's I2SNMAGSHIFT_SNVAR[ibin][igal]
    galaxy_snmagshift = (None if wgtmap_data.snmagshift is None
                         else np.array(interpolate(wgtmap_data.snmagshift)))
    # raw cumsums, as SNANA (per-bin totals matter)
    wgtsum = jnp.cumsum(weights, axis=1)
    logger.info(f"WGTSUM computation complete: {n_snvar_bins} bin(s) × "
                f"{table.ngal:,} galaxies in {time.time() - t_start:.2f}s")

    if not snvar_idx and galaxy_snmagshift is not None:
        galaxy_snmagshift = galaxy_snmagshift[0]
    return WgtSumData(
        galaxy_indices=np.arange(table.ngal, dtype=np.int32),
        wgtsum=np.array(wgtsum),
        weights=np.array(weights),
        total_galaxies=table.ngal,
        snvar_bin_centers=centers or None,
        snvar_names=wgtmap_data.snvar_names or None,
        galaxy_snmagshift=galaxy_snmagshift
    )


def _apply_magobs_zeroflux(
    properties: jnp.ndarray,
    varnames: List[str],
    ranges: np.ndarray,
) -> jnp.ndarray:
    """Substitute [band]_obs = MAG_ZEROFLUX values with VALMAX - 0.001.

    A galaxy mag of 99 means zero flux; SNANA replaces it with the WGTMAP
    dimension's maximum minus 0.001 before interpolation so these galaxies
    do not trip the bounds abort (sntools_host.c:4266-4275, 4316-4322).
    Applies to every WGTMAP dimension whose name contains '_obs'
    (HOSTLIB_SUFFIX_MAGOBS, matched with strstr).
    """
    magobs_dims = np.array(['_obs' in v for v in varnames])
    if not magobs_dims.any():
        return properties
    ranges = np.asarray(ranges)
    substitute = jnp.array(ranges[:, 1] - 0.001, dtype=properties.dtype)
    hit = (properties >= MAG_ZEROFLUX) & jnp.array(magobs_dims)[None, :]
    return jnp.where(hit, substitute[None, :], properties)


def _check_bounds(
    galaxy_properties: jnp.ndarray,
    varnames: List[str],
    ranges: np.ndarray,
    opt_extrap: int,
):
    """Check if galaxies are within WGTMAP bounds.

    Implements SNANA's OPT_EXTRAP behaviour (interp_GRIDMAP failure is fatal
    in the wgtsum loop, sntools_host.c:4330-4352): ``opt_extrap = 0``
    (default) aborts if any galaxy is out of bounds; ``opt_extrap > 0``
    allows extrapolation (clamp to edge) and warns if out of bounds.

    Raises `SNANAError` if opt_extrap==0 and any galaxy is out of bounds.
    """
    props_np = np.array(galaxy_properties)
    ranges = np.asarray(ranges)
    below_min = props_np < ranges[:, 0]
    above_max = props_np > ranges[:, 1]
    oob_mask = np.logical_or(below_min, above_max)
    any_oob_per_galaxy = np.any(oob_mask, axis=1)
    n_oob = np.sum(any_oob_per_galaxy)

    if n_oob == 0:
        return

    frac_oob = n_oob / len(galaxy_properties)

    if opt_extrap == 0:
        error_lines = [
            f"\n{n_oob:,} galaxies ({frac_oob*100:.2f}%) have properties outside WGTMAP bounds:",
        ]

        for idim, varname in enumerate(varnames):
            n_below = np.sum(below_min[:, idim])
            n_above = np.sum(above_max[:, idim])
            min_val, max_val = ranges[idim]

            if n_below > 0 or n_above > 0:
                error_lines.append(
                    f"  {varname}: [{min_val:.3f}, {max_val:.3f}] "
                    f"- {n_below:,} below, {n_above:,} above"
                )

        error_lines.append(
            "\nTo allow extrapolation (clamp to grid edge), "
            "set HOSTLIB_WGTMAP_EXTRAP: 1 in your input file."
        )

        raise SNANAError("\n".join(error_lines), function_name="_check_bounds")

    logger.warning(f"{n_oob:,} galaxies ({frac_oob*100:.2f}%) "
                   f"out of WGTMAP bounds")
    logger.warning(f"  Values will be clamped to grid edge "
                   f"(opt_extrap={opt_extrap})")

    for idim, varname in enumerate(varnames):
        n_below = np.sum(below_min[:, idim])
        n_above = np.sum(above_max[:, idim])
        min_val, max_val = ranges[idim]

        if n_below > 0 or n_above > 0:
            logger.warning(f"    {varname}: {n_below:,} below {min_val:.3f}, "
                           f"{n_above:,} above {max_val:.3f}")
