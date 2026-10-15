"""Reader for SNANA LENSING_PROBMAP_FILE (weak lensing magnification maps).

The file contains a 2D probability distribution P(deltaMU | z) at discrete
redshift bins. At init the reader builds a pre-inverted CDF table on a fine
uniform quantile grid so that runtime sampling is a single vectorised bilinear
interpolation: O(n_sne), fully JIT-compatible, no per-SN binary search.

File format: 3-column text (z, deltaMU, prob) with DOCUMENTATION header.
"""

from jax_snlc_core.io.docana import skip_docana
from jax_snlc_core.utils import snana_log as slog
from jax_snlc_core.utils.error_handling import SNANAError
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from jax_snlc_core.io.fits_path import open_text, resolve_fits_path

import jax
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LensingProbmap:
    """Pre-inverted CDF table for lensing magnification sampling.

    ``inv_cdf[iz, iq]`` is the deltaMU value at redshift bin ``iz`` and
    quantile fraction ``iq / (n_quantiles - 1)``. Sampling is a single
    ``map_coordinates`` call in (iz, iq) space.
    """
    z_bins: np.ndarray        # (nz,), redshift bin centres
    inv_cdf: np.ndarray       # (nz, n_quantiles)
    n_quantiles: int


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

    slog.line(logger, f"Loaded lensing probmap: {nz} z-bins "
              f"[{z_unique[0]:.3f}, {z_unique[-1]:.3f}], "
              f"inv_cdf quantiles={n_quantiles}")

    return LensingProbmap(
        z_bins=z_unique,
        inv_cdf=inv_cdf.astype(np.float32),
        n_quantiles=n_quantiles,
    )


jax.tree_util.register_dataclass(
    LensingProbmap,
    data_fields=[
        'z_bins',
        'inv_cdf',
    ],
    meta_fields=[
        'n_quantiles',
    ],
)
