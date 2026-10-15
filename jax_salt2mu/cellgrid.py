"""Shared multi-dimensional binning primitive for the BBC maps.

The biasCor map, muCOVscale map and CC-prior MUZMAP all bin events onto a grid
and accumulate per-cell statistics. ``BinInfo`` is a single dimension's bins
(SNANA ``BININFO_DEF``); ``CellGrid`` flattens several dimensions to a 1-D cell
index (``CELLINFO_DEF`` / ``MAPCELL`` / ``J1D``). The flattening is C-order
``ravel_multi_index``, which reproduces SNANA's nested-loop J1D exactly, so a
``write_biascor`` dump is row-comparable.
"""

from dataclasses import dataclass
from functools import cached_property
from itertools import product
from typing import Optional, Sequence, Tuple

import numpy as np

_OUT_OF_RANGE = -9  # SNANA IBINFUN sentinel


@dataclass(frozen=True)
class BinInfo:
    nbin: int
    lo: np.ndarray
    hi: np.ndarray
    avg: np.ndarray
    binsize: float

    def index(self, x, opt: int = 0) -> np.ndarray:
        """Bin index for each value (SNANA IBINFUN).

        Bins are inclusive on both ends with first-match (lowest index) winning,
        so a value on a shared edge falls in the lower bin; the last bin gets
        SNANA's ``+1e-10`` fuzz. ``opt=0`` returns ``-9`` out of range; ``opt=2``
        clamps to the edge bin using the bin averages.
        """
        x = np.asarray(x, dtype=float)
        hi = np.array(self.hi, dtype=float, copy=True)
        hi[-1] += 1e-10
        # Ordered bins: the first upper edge containing x gives the lower
        # bin at shared edges. Check the lower edge as grids can have gaps.
        candidate = np.searchsorted(hi, x, side="left")
        safe = np.minimum(candidate, self.nbin - 1)
        inside = ((candidate < self.nbin) & (x >= self.lo[safe]) & (x <= hi[safe]))
        idx = np.where(inside, candidate, _OUT_OF_RANGE)
        if opt == 2:
            idx = np.where(x < self.avg[0], 0, idx)
            idx = np.where(x > self.avg[-1], self.nbin - 1, idx)
        return idx


def uniform_bininfo(lo: float, hi: float, nbin: int) -> BinInfo:
    edges = np.linspace(lo, hi, nbin + 1)
    los, his = edges[:-1], edges[1:]
    return BinInfo(nbin=nbin, lo=los, hi=his, avg=0.5 * (los + his),
                   binsize=(hi - lo) / nbin)


@dataclass(frozen=True)
class CellGrid:
    bins: Tuple[BinInfo, ...]

    @cached_property
    def dims(self) -> Tuple[int, ...]:
        return tuple(b.nbin for b in self.bins)

    @cached_property
    def ncell(self) -> int:
        return int(np.prod(self.dims))

    @property
    def ndim(self) -> int:
        return len(self.bins)

    def j1d(self, indices: Sequence[np.ndarray]) -> np.ndarray:
        """C-order flattened cell index for per-dimension bin indices."""
        return np.ravel_multi_index(tuple(indices), self.dims).astype(np.int64)

    def unravel(self, j1d) -> Tuple[np.ndarray, ...]:
        return np.unravel_index(np.asarray(j1d), self.dims)

    def neighbours(self, j1d: int, vary_dims: Optional[Sequence[int]] = None) -> np.ndarray:
        """Neighbour cells of ``j1d`` (SNANA get_J1DNBR_LIST).

        Each dimension in ``vary_dims`` (default: all) is offset by {-1,0,+1};
        out-of-range offsets are skipped (not clipped) and the centre cell is
        included exactly once.
        """
        if vary_dims is None:
            vary_dims = range(self.ndim)
        vary_dims = list(vary_dims)
        centre = list(np.unravel_index(int(j1d), self.dims))
        out = []
        for offsets in product((-1, 0, 1), repeat=len(vary_dims)):
            idx = list(centre)
            ok = True
            for d, off in zip(vary_dims, offsets):
                idx[d] += off
                if idx[d] < 0 or idx[d] >= self.dims[d]:
                    ok = False
                    break
            if ok:
                out.append(int(np.ravel_multi_index(tuple(idx), self.dims)))
        return np.asarray(out, dtype=np.int64)

    def scatter(self, j1d: np.ndarray, values: np.ndarray,
                weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Per-cell ``(sum of weight*value, sum of weight)`` via bincount."""
        sum_wv = np.bincount(j1d, weights=weights * values, minlength=self.ncell)
        sum_w = np.bincount(j1d, weights=weights, minlength=self.ncell)
        return sum_wv, sum_w

    def count(self, j1d: np.ndarray) -> np.ndarray:
        return np.bincount(j1d, minlength=self.ncell)
