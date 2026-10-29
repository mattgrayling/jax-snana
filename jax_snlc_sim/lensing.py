"""Weak-lensing magnitude offsets, probability maps and measurement scatter.

The file contains a 2D probability distribution P(deltaMU | z) at discrete
redshift bins. At init the reader builds a pre-inverted CDF table on a fine
uniform quantile grid so that runtime sampling is a single vectorised bilinear
interpolation: O(n_sne), fully JIT-compatible, no per-SN binary search.

File format: 3-column text (z, deltaMU, prob) with DOCUMENTATION header.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core import logging as log
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import open_text, resolve_fits_path, skip_docana
from jax_snlc_core.random import event_keys

logger = logging.getLogger(__name__)


@jax.tree_util.register_dataclass
@dataclass
class LensingProbmap:
    """Pre-inverted CDF table for lensing magnification sampling.

    ``inv_cdf[iz, iq]`` is the deltaMU value at redshift bin ``iz`` and
    quantile fraction ``iq / (n_quantiles - 1)``. Sampling is a single
    ``map_coordinates`` call in (iz, iq) space.
    """
    z_bins: np.ndarray        # (nz,), redshift bin centres
    inv_cdf: jax.Array        # (nz, n_quantiles)


def read_lensing_probmap(path: str, dmuscale: float,
                         n_quantiles: int = 2000) -> LensingProbmap:
    """Read a LENSING_PROBMAP_FILE and build the pre-inverted CDF table.

    ``path`` may contain env vars. ``dmuscale`` is a scale factor applied
    to all deltaMU values at init (matches SNANA's ``WEAKLENS_DMUSCALE``).
    ``n_quantiles`` is the resolution of the uniform quantile grid for the
    pre-inverted CDF; 2000 is ~10x finer than the typical 199-bin deltaMU
    grid in the file.

    Returns the LensingProbmap ready for sampling.
    """
    path = Path(resolve_fits_path(os.path.expandvars(path)))
    if not path.exists():
        raise SNANAError(f"Lensing probmap file not found: {path}",
                         function_name="read_lensing_probmap")

    rows = []
    with open_text(path) as f:
        for line in skip_docana(f):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            parts = stripped.split()
            if len(parts) >= 3:
                try:
                    rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
                except ValueError:
                    continue

    if not rows:
        raise SNANAError(f"No data rows found in {path}")

    data = np.array(rows, dtype=np.float64)
    z_all, dmu_all, prob_all = data[:, 0], data[:, 1], data[:, 2]

    # rows per z bin may differ
    z_unique = np.unique(z_all)
    nz = len(z_unique)

    q_grid = np.linspace(0.0, 1.0, n_quantiles)
    inv_cdf = np.empty((nz, n_quantiles), dtype=np.float64)

    for iz in range(nz):
        mask = z_all == z_unique[iz]
        dmu_iz = dmu_all[mask] * dmuscale
        prob_iz = prob_all[mask]
        order = np.argsort(dmu_iz)
        dmu_iz = dmu_iz[order]
        prob_iz = prob_iz[order]
        cdf = np.cumsum(prob_iz)
        total = cdf[-1]
        if total <= 0:
            inv_cdf[iz] = 0.0
            continue
        cdf = cdf / total
        inv_cdf[iz] = np.interp(q_grid, cdf, dmu_iz)

    log.line(logger, f"Loaded lensing probmap: {nz} z-bins "
             f"[{z_unique[0]:.3f}, {z_unique[-1]:.3f}], "
             f"inv_cdf quantiles={n_quantiles}")

    return LensingProbmap(
        z_bins=z_unique,
        inv_cdf=jnp.asarray(inv_cdf, dtype=jnp.float32),
    )


def lensing_offset(simulator, redshifts, keys, host_properties):
    """The weak-lensing mu offset (gen_distanceMag,
    snlc_sim.c:17510-17519): WEAKLENS_DSIGMADZ wins over
    WEAKLENS_PROBMAP_FILE, which wins over the host's HOSTLIB WEAKLENS_DMU
    column; zero with none of them. ``host_properties`` is None before the
    host is drawn."""
    if simulator.lensing.dsigmadz > 1.0e-8:
        return (redshifts * simulator.lensing.dsigmadz
                * jax.vmap(jax.random.normal)(keys))
    if simulator.lensing_probmap is not None:
        return _sample_lensdmu(simulator.lensing_probmap.z_bins,
                                simulator.lensing_probmap.inv_cdf, redshifts, keys)
    if host_properties is not None and 'WEAKLENS_DMU' in host_properties:
        return host_properties['WEAKLENS_DMU']
    return jnp.zeros_like(redshifts)


def _sample_lensdmu(z_bins, inv_cdf, redshifts, keys):
    """Sample weak-lensing deltaMU per SN via the pre-inverted CDF.

    Fully vectorised: a single ``map_coordinates`` call over all SNe.
    Below/above the map's z-range, deltaMU scales by z / z_edge.
    """
    ran1 = jax.vmap(jax.random.uniform)(keys)
    ran1 = jnp.clip(ran1, 1e-8, 1.0 - 1e-8)
    iz = jnp.interp(redshifts, jnp.asarray(z_bins), jnp.arange(len(z_bins)))
    iq = ran1 * (inv_cdf.shape[1] - 1)
    coords = jnp.stack([iz, iq], axis=0)
    dmu = jax.scipy.ndimage.map_coordinates(
        inv_cdf, coords, order=1, mode='nearest')
    z_min = z_bins[0]
    z_max = z_bins[-1]
    dmu = jnp.where(redshifts < z_min, dmu * (redshifts / z_min), dmu)
    dmu = jnp.where(redshifts > z_max, dmu * (redshifts / z_max), dmu)
    return dmu


def apply_lensdmu_smear(simulator, params: dict[str, jnp.ndarray], *,
                        key) -> None:
    """WEAKLENS_DMUERR_FRAC: the reported lensing offset and its error.

    gen_lensDMU_smear draws ``lensDMU_smear = lensDMU + |lensDMU * FRAC| * G``
    and the data file reports it with ``LENSDMU_ERR = |lensDMU * FRAC|``
    (sntools_weaklens.c:256-276, snlc_sim.c:12938, :25422-25423). SIM_LENSDMU
    keeps the unsmeared truth. SNANA smears after the HOSTLIB WEAKLENS_DMU
    override, so this runs once host association is done, and it burns a
    random only when the fraction is set.
    """
    lensdmu = jnp.asarray(params['lensdmu'])
    frac = simulator.lensing.dmuerr_frac
    err = jnp.abs(lensdmu * frac)
    params['lensdmu_err'] = err
    if frac <= 0.0:
        params['lensdmu_smear'] = lensdmu
        return
    keys = event_keys(key, 'LENSDMU_SMEAR', params['sn_global_idx'])
    params['lensdmu_smear'] = lensdmu + err * jax.vmap(jax.random.normal)(keys)
