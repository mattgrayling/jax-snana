"""BBC event bins, bias-correction grids, and coefficient interpolation."""

import re
from dataclasses import dataclass
from functools import cached_property
from itertools import product
from typing import Dict, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np

from jax_snlc_core.logging import SNANAError

from .config import BBCConfig, use_gamma0


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


def coeff_node(coeff_idx, coeff_bins, n) -> np.ndarray:
    """Flat coefficient-node index per event; all-zero when there are no axes."""
    if not coeff_bins:
        return np.zeros(n, dtype=np.int64)
    return np.ravel_multi_index(tuple(coeff_idx), tuple(b.nbin for b in coeff_bins))


def bininfo_from_range(vmin: float, vmax: float, vbin: float, name: str) -> BinInfo:
    """setup_BININFO_biasCor bin loop (LO_MAX = vmax - 0.99*vbin; vbin<=0 -> 1 bin).

    A width wider than the range leaves no bins at all. SNANA builds the empty axis
    and aborts on the first event it cannot place (IBINFUN, "range: 0.000 to 0.000"),
    so the same configs are rejected either way.
    """
    if vbin <= 0.0:
        return BinInfo(1, np.array([vmin]), np.array([vmax]),
                       np.array([0.5 * (vmin + vmax)]), max(vmax - vmin, 0.0))
    lo_max = vmax - 0.99 * vbin
    los = []
    v = vmin
    while v <= lo_max:
        los.append(v)
        v += vbin
    if not los:
        raise SNANAError(
            f"{name} bin size {vbin:g} leaves no bins over [{vmin:g}, {vmax:g}]",
            function_name="bininfo_from_range")
    los = np.asarray(los)
    his = los + vbin
    return BinInfo(len(los), los, his, 0.5 * (los + his), vbin)


def coeff_bininfo(sim_vals, is_continuous: bool, onebin: bool, name: str,
                  is_ia=None) -> BinInfo:
    """get_BININFO_biasCor_abg: native grid from distinct SIM values, or a
    forced single bin under u=0|3 or for a continuous coefficient (>5
    values). The distinct values are counted over true Ia rows alone
    (NONIA_INDEX != 0 is skipped, SALT2mu.c:16373)."""
    vals = np.asarray(sim_vals, dtype=float)             # float32 SIM grid values
    if is_ia is not None:
        vals = vals[np.asarray(is_ia, dtype=bool)]
    v = np.unique(vals)
    nval = len(v)
    val1st, val_last = v[0], v[-1]
    val2nd = v[1] if nval >= 2 else v[0]

    force_onebin = (is_continuous and nval > 5) or onebin
    if force_onebin:
        nval, val2nd = 1, val_last
    elif nval > 2:
        raise SNANAError(
            f"{nval} biasCor SIM values exceed the 2-value grid bound; a gridded "
            f"coefficient must be defined on a 2-point grid.",
            function_name="coeff_bininfo")

    vbin = val2nd - val1st
    vmin = val1st - 0.5 * vbin
    vmax = val_last + 0.5 * vbin
    if force_onebin:
        vbin = vmax - vmin
    return bininfo_from_range(vmin, vmax, vbin, name)


def parse_binsize_stringopt(surveygroup_biascor: str) -> Dict[str, float]:
    """Extract the ``zbin`` and ``<axis>bin`` overrides from the ``(...)`` STRINGOPT."""
    out: Dict[str, float] = {}
    m = re.search(r"\(([^)]*)\)", surveygroup_biascor or "")
    if m:
        for tok in m.group(1).split(":"):
            if "=" in tok:
                key, val = tok.split("=", 1)
                out[key.strip()] = float(val)
    return out


def build_coeff_bins(config, model, sim_truth) -> Tuple[BinInfo, ...]:
    """One BinInfo per ``biascor_coeffs`` axis, shared across IDSAMPLEs."""
    bins = []
    template = sim_truth.get("SIM_TEMPLATE_INDEX")
    is_ia = None if template is None else np.asarray(template) == 0
    for name, sim_col, continuous in model.biascor_coeffs:
        # a parameter given no u<N> is held (ipar 0, SALT2mu.c:6702), hence one bin
        onebin = config.nuisance_onebin.get(name, True)
        bins.append(coeff_bininfo(sim_truth[sim_col], continuous, onebin, name, is_ia))
    return tuple(bins)


def biascor_gammadm_offset(model, coeff_bins, config) -> float:
    """GAMMADM_OFFSET: the midpoint of the biasCor axis the mass step reads.

    ``(VAL_MIN + VAL_MAX)/2`` of the model's ``gammadm_coeff`` axis
    (setup_CELLINFO_biasCor, SALT2mu.c:16225-16228), which get_gammadm_host
    subtracts from the step (:16782). Zero where USE_GAMMA0 leaves the term out.
    """
    if not use_gamma0(model, config):
        return 0.0
    axis = [name for name, _, _ in model.biascor_coeffs].index(model.gammadm_coeff)
    return 0.5 * (float(coeff_bins[axis].lo[0]) + float(coeff_bins[axis].hi[-1]))


def sample_z_bininfo(binsize) -> BinInfo:
    """The sample's biasCor z grid: one bin past zrange (+1 z-bin, ipar=-1)."""
    zbin = binsize.zbin
    r0, r1 = binsize.zrange
    return bininfo_from_range(r0, r1 + zbin, zbin, "zbin")


def build_biascor_cellgrid(config, model, coeff_bins, binsize
                           ) -> Tuple[CellGrid, Tuple[BinInfo, ...]]:
    """biasCor CellGrid for one IDSAMPLE: the global coefficient axes plus this
    sample's z/logmass/lcpar grid from its ``SampleBinsize``."""
    b_z = sample_z_bininfo(binsize)

    mmin, mmax = config.logmass_min, config.logmass_max
    if config.nbin_logmass > 1 and mmin == -40.0 and mmax == 40.0:
        raise SNANAError(
            f"nbin_logmass={config.nbin_logmass} but did not specify log mass "
            f"range; must provide inputs logmass_min and logmass_max",
            function_name="build_biascor_cellgrid")
    mbin = (mmax - mmin) / config.nbin_logmass
    b_m = bininfo_from_range(mmin, mmax, mbin, "logmass")

    lc_bins = []
    for name, (vmin, vmax, default_bin) in model.biascor_binning.items():
        lc_bins.append(bininfo_from_range(
            vmin, vmax, binsize.lcpar_bins.get(name, default_bin), name))

    bins = (*coeff_bins, b_z, b_m, *lc_bins)
    return CellGrid(bins=bins), bins


def axis_weights(x, avg, binsize):
    """Per-event 1-D interpolation weights ``[n, n_nodes]``
    (get_INTERPWGT_abg), normalised; a single-bin axis gets weight 1. ``x``
    is clamped to the grid to avoid extrapolation."""
    x = jnp.atleast_1d(x)
    if avg.shape[0] == 1:
        return jnp.ones((x.shape[0], 1))
    xc = jnp.clip(x, avg[0], avg[-1])[:, None]
    w = jnp.maximum(0.0, 1.0 - jnp.abs(xc - avg[None, :]) / binsize)
    return w / jnp.sum(w, axis=1, keepdims=True)


def coeff_weights(coeff_vals, coeff_avg, coeff_binsize, n_events):
    """Per-event ``[n, N]`` weights over the coefficient nodes: outer product
    of per-axis weights (get_INTERPWGT_abg), C-order over the nodes. SNANA
    evaluates them at each event's effective alpha/beta/gammaDM
    (SALT2mu.c:5332-5338)."""
    w = None
    for val, avg, binsize in zip(coeff_vals, coeff_avg, coeff_binsize):
        wj = axis_weights(jnp.broadcast_to(val, (n_events,)), avg, binsize)
        w = wj if w is None else \
            (w[:, :, None] * wj[:, None, :]).reshape(n_events, w.shape[1] * wj.shape[1])
    return jnp.ones((n_events, 1)) if w is None else w


def below_centroid(value, centroid, count, width, xp=np):
    """Choose the lower interpolation neighbour only for a resolved difference.

    A weighted centroid carries summation error. Values within that error bound
    take the equality branch, so duplicated events do not change neighbours
    when a reduction backend changes its rounding. gamma_n is the standard
    floating-point error bound for n additions, plus multiplication/division.
    """
    error = (count + 2) * xp.finfo(centroid.dtype).eps
    gamma = error / (1. - error)
    tolerance = gamma * (xp.abs(value) + xp.abs(centroid) + width)
    return value < centroid - tolerance


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
