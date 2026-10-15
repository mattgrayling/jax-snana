"""Redshift binning for the per-z-bin M0 offsets (SNANA setup_BININFO_redshift)."""

from dataclasses import dataclass

import numpy as np

from .config import BBCConfig


@dataclass(frozen=True)
class ZBinGrid:
    lo: np.ndarray
    hi: np.ndarray

    @property
    def nbin(self) -> int:
        return len(self.lo)

    @property
    def centers(self) -> np.ndarray:
        return 0.5 * (self.lo + self.hi)

    def index(self, z) -> np.ndarray:
        """Bin index, using SNANA's first inclusive match (-1 outside the grid)."""
        z = np.asarray(z, dtype=float)
        idx = np.full(z.shape, -1, dtype=np.int64)
        for i in range(self.nbin):
            hi = self.hi[i] + (1e-10 if i == self.nbin - 1 else 0.0)
            idx = np.where((idx < 0) & (z >= self.lo[i]) & (z <= hi), i, idx)
        return idx


def _powz_edges(cfg: BBCConfig) -> np.ndarray:
    """setup_BININFO_powz (SALT2mu.c:2941): binSize ~ (1+z)^powzbin. With znhalf>0 the
    first nzbin/2 bins follow the power law up to znhalf and the rest are uniform."""
    p1 = 1.0 - cfg.powzbin
    if p1 == 0.0:
        p1 = 1.0e-12
    lhalf = cfg.znhalf > 0.0
    zmax = cfg.znhalf if lhalf else cfg.zmax
    n1 = cfg.nzbin // 2 if lhalf else cfg.nzbin
    znorm = p1 / ((1.0 + zmax) ** p1 - (1.0 + cfg.zmin) ** p1)
    z1pbin = 1.0 / (znorm * n1)

    edges = [cfg.zmin]
    zlast = cfg.zmin
    for _ in range(n1):
        zlast = (p1 * z1pbin + (1.0 + zlast) ** p1) ** (1.0 / p1) - 1.0
        edges.append(zlast)
    if lhalf:
        n2 = cfg.nzbin - n1
        zbin2 = (cfg.zmax - cfg.znhalf) / n2
        edges.extend(cfg.znhalf + (i + 1) * zbin2 for i in range(n2))
    return np.asarray(edges, dtype=float)


def make_zbins(cfg: BBCConfig) -> ZBinGrid:
    """User bins take precedence, then log bins, then power-law bins, then uniform.

    Pure: ``config.resolve`` has already left ``cfg.nzbin``/``zmin``/``zmax`` describing
    the bins to build, and validated that one of the three forms is given.
    """
    if cfg.zbinuser:
        edges = np.asarray(sorted(cfg.zbinuser), dtype=float)
    elif cfg.nlogzbin > 0:
        edges = np.logspace(np.log10(cfg.zmin), np.log10(cfg.zmax), cfg.nlogzbin + 1)
    elif cfg.powzbin != 0.0:
        edges = _powz_edges(cfg)
    else:
        edges = np.linspace(cfg.zmin, cfg.zmax, cfg.nzbin + 1)
    return ZBinGrid(lo=edges[:-1], hi=edges[1:])
