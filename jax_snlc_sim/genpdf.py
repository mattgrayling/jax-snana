"""
GENPDF file parser and sampler for SNANA GENPDF_FILE distributions.

GENPDF files specify arbitrary probability distributions on a grid. SNANA
draws from them by rejection against the multilinearly interpolated raw PROB
values (getRan_genPDF, sntools_genPDF.c:691-865). Here each grid slice's
piecewise-linear PDF is inverted exactly at load time, and conditional draws
reproduce the interpolated raw density as a mixture over the bracketing
corners; see sample_genpdf_conditional.

Supports:
- 1D PDFs (e.g., SALT2c → PROB)
- 2D+ conditional PDFs (e.g., SALT2c × LOGMASS → PROB)
- LOG parameter support (LOGEBV, LOGAV → exponentiated after sampling)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core import logging as log
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import open_text, skip_docana

logger = logging.getLogger(__name__)


@dataclass
class GenPDF:
    """Validated per-slice PDF tables for a GENPDF distribution.

    A slice is one combination of conditioning-grid values (a single slice
    for an unconditional map). Construction validates the full Cartesian
    grid, so every slice shares ``x_grid`` and has positive total mass.

    Attributes:
        varname: Parameter being sampled (e.g., "SALT2c")
        x_grid: 1D array of parameter values (sorted), shared by all slices
        prob: (n_slices, n_x) per-slice PDF values, normalised to unit
              trapezoid integral over x_grid
        cdf: (n_slices, n_x) per-slice normalised CDF at the x nodes
        raw_integral: (n_slices,) trapezoid integral of the raw PROB values,
              the I_i weights of the conditional corner mixture
        is_log: If True, exponentiate after sampling (10^x)
        cond_varnames: Names of conditioning variables (empty for 1D)
        cond_grids: List of unique values for each conditioning variable
        cond_shape: Tuple of conditioning grid sizes (row-major slice order)
    """
    varname: str
    x_grid: jnp.ndarray
    prob: jnp.ndarray
    cdf: jnp.ndarray
    raw_integral: jnp.ndarray
    is_log: bool = False
    cond_varnames: List[str] = field(default_factory=list)
    cond_grids: List[jnp.ndarray] = field(default_factory=list)
    cond_shape: Tuple[int, ...] = ()
    # GENPDF_EXPON_REWGT: the sampled density is interp(prob)^e
    expon_rewgt: float = 1.0
    prob_raw: jnp.ndarray = None


def parse_genpdf_file(filepath: str) -> Tuple[Dict[str, dict], Dict[str, str]]:
    """Parse a GENPDF file and return raw grid data plus embedded parameter overrides.

    GENPDF files have the format::

        VARNAMES: SALT2c PROB
        PDF: -0.3  0.001
        PDF: -0.2  0.05
        ...

    Or multi-dimensional::

        VARNAMES: SALT2c LOGMASS PROB
        PDF: -0.3  9.0  0.001
        ...

    A single file can contain multiple maps separated by VARNAMES lines.

    GENPDF files also contain embedded parameter overrides like::

        GENPEAK_SALT2ALPHA: 0.115
        GENSIGMA_SALT2ALPHA: 0.003 0.003
        GENRANGE_SALT2ALPHA: 0.09 0.13

    SNANA reads these to set ALPHA/BETA (and potentially other params).

    Returns ``(maps, embedded_params)``: ``maps`` maps the sampled variable
    name to ``{'varnames': list, 'data': np.ndarray of shape (n_rows,
    n_cols)}``; ``embedded_params`` maps SNANA keyword to value string.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise SNANAError(f"GENPDF file not found: {filepath}",
                         function_name="parse_genpdf_file")

    _OVERRIDE_PREFIXES = (
        'GENPEAK_', 'GENMEAN_', 'GENSIGMA_', 'GENRANGE_',
        'GENSIGMA2_', 'GENPEAK2_', 'GENPROB2_',
        'GENPEAKRANGE_', 'GENSKEW_', 'GENNGRID_',
        # the whole-run keys the file may carry (sntools_genPDF.c:213-225)
        'MAG_OFFSET:', 'GENMAG_SMEAR:',
    )

    maps = {}
    embedded_params = {}
    current_varnames = None
    current_rows = []

    with open_text(filepath) as f:
        for line in skip_docana(f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if line.startswith('VARNAMES:'):
                if current_varnames is not None and current_rows:
                    _store_map(maps, current_varnames, current_rows)
                parts = line.split(':', 1)[1].split()
                current_varnames = parts
                current_rows = []
            elif line.startswith('PDF:'):
                values = line.split(':', 1)[1].split()
                current_rows.append([float(v) for v in values])
            elif line.startswith(_OVERRIDE_PREFIXES):
                key, _, val = line.partition(':')
                if val.strip():
                    embedded_params[key.strip()] = val.strip()

    if current_varnames is not None and current_rows:
        _store_map(maps, current_varnames, current_rows)

    return maps, embedded_params


def _store_map(maps, varnames, rows):
    """Store a parsed GENPDF map. The sampled variable is the first one (before PROB)."""
    data = np.array(rows)
    sampled_var = varnames[0]
    maps[sampled_var] = {
        'varnames': varnames,
        'data': data
    }


def _build_genpdf(varname, map_data, prob_expon_rewgt):
    """Build a GenPDF from already-parsed map data.

    Validates the map: probabilities must be non-negative, every Cartesian
    (conditioning, x) cell must appear exactly once (so all slices share one
    x grid), and every slice must carry positive finite total mass. A
    violation is a malformed recognised input, hence fatal.

    ``varname`` is the variable name (e.g., "SALT2c"); ``map_data`` is the
    dict with 'varnames' and 'data' from parse_genpdf_file;
    ``prob_expon_rewgt`` reweights probability as prob -> prob^rewgt.

    Returns the GenPDF with per-slice normalised PDF/CDF tables.
    """
    all_varnames = map_data['varnames']
    data = map_data['data']
    is_log = varname.startswith('LOG')
    cond_varnames = list(all_varnames[1:-1])
    n_cond = len(cond_varnames)

    x_vals = data[:, 0]
    prob = data[:, -1]

    if not np.all(np.isfinite(data)):
        raise SNANAError(
            f"GENPDF map for '{varname}' contains non-finite values.",
            function_name="_build_genpdf")
    if np.any(prob < 0.0):
        bad = int(np.argmax(prob < 0.0))
        raise SNANAError(
            f"GENPDF map for '{varname}' has negative PROB={prob[bad]:g} "
            f"at {varname}={x_vals[bad]:g}.",
            function_name="_build_genpdf")

    x_unique = np.unique(x_vals)
    n_x = len(x_unique)
    if n_x < 2:
        raise SNANAError(
            f"GENPDF map for '{varname}' needs at least two grid values of "
            f"{varname}; found {n_x}.",
            function_name="_build_genpdf")
    cond_uniques = [np.unique(data[:, 1 + ic]) for ic in range(n_cond)]
    cond_shape = tuple(len(cu) for cu in cond_uniques)
    n_slices = int(np.prod(cond_shape)) if n_cond else 1

    x_idx = np.searchsorted(x_unique, x_vals)
    slice_idx = np.zeros(len(data), dtype=np.int64)
    for ic in range(n_cond):
        slice_idx = (slice_idx * cond_shape[ic]
                     + np.searchsorted(cond_uniques[ic], data[:, 1 + ic]))

    counts = np.zeros((n_slices, n_x), dtype=np.int64)
    np.add.at(counts, (slice_idx, x_idx), 1)
    if np.any(counts > 1):
        s, k = np.argwhere(counts > 1)[0]
        raise SNANAError(
            f"GENPDF map for '{varname}' defines the cell at "
            f"{varname}={x_unique[k]:g} (conditioning slice {int(s)}) "
            f"{int(counts[s, k])} times.",
            function_name="_build_genpdf")
    if np.any(counts == 0):
        s, k = np.argwhere(counts == 0)[0]
        raise SNANAError(
            f"GENPDF map for '{varname}' is missing the cell at "
            f"{varname}={x_unique[k]:g} (conditioning slice {int(s)}): "
            f"every conditioning combination must tabulate the same "
            f"{varname} grid.",
            function_name="_build_genpdf")

    pgrid = np.zeros((n_slices, n_x))
    pgrid[slice_idx, x_idx] = prob

    dx = np.diff(x_unique)
    seg = _segment_masses(pgrid, dx, prob_expon_rewgt)
    raw_integral = seg.sum(axis=1)
    invalid = ~np.isfinite(raw_integral) | (raw_integral <= 0.0)
    if np.any(invalid):
        s = int(np.where(invalid)[0][0])
        raise SNANAError(
            f"GENPDF map for '{varname}' has total probability mass "
            f"{raw_integral[s]:g} in conditioning slice {s}; SNANA aborts when no "
            f"value has PROB above threshold.",
            function_name="_build_genpdf")

    pnorm = np.power(pgrid, prob_expon_rewgt) / raw_integral[:, None]
    cdf = np.concatenate(
        [np.zeros((n_slices, 1)), np.cumsum(seg, axis=1)], axis=1)
    cdf = cdf / cdf[:, -1:]

    if n_cond:
        log.line(logger, f"Loaded GENPDF '{varname}' conditioned on "
                 f"{cond_varnames}: grid "
                 f"{'×'.join(str(s) for s in cond_shape)}×{n_x}")

    return GenPDF(
        varname=varname,
        x_grid=jnp.array(x_unique),
        prob=jnp.array(pnorm),
        cdf=jnp.array(cdf),
        raw_integral=jnp.array(raw_integral),
        is_log=is_log,
        cond_varnames=cond_varnames,
        cond_grids=[jnp.array(cu) for cu in cond_uniques],
        cond_shape=cond_shape,
        expon_rewgt=float(prob_expon_rewgt),
        prob_raw=jnp.array(pgrid),
    )


def load_all_genpdf(filepath: str,
                    prob_expon_rewgt: float) -> Tuple[Dict[str, GenPDF], Dict[str, str]]:
    """Load all variable maps from a GENPDF file.

    ``prob_expon_rewgt`` reweights probability as prob -> prob^rewgt.
    Returns ``(genpdfs, embedded_params)``: a dict mapping variable name to
    GenPDF, and a dict of embedded parameter overrides (SNANA keyword to
    value string).
    """
    maps, embedded_params = parse_genpdf_file(filepath)
    result = {}
    for varname, map_data in maps.items():
        result[varname] = _build_genpdf(varname, map_data, prob_expon_rewgt)
    return result, embedded_params


def _segment_masses(pgrid, dx, e):
    """Per-segment integral of ``interp(prob)^e`` over each x interval.

    SNANA raises the exponent on the interpolated probability
    (sntools_genPDF.c:670-672), so for e != 1 the density inside a segment
    is (p0 + (p1-p0)t)^e, whose integral is closed-form.
    """
    p0, p1 = pgrid[..., :-1], pgrid[..., 1:]
    if e == 1.0:
        return 0.5 * (p0 + p1) * dx
    delta = p1 - p0
    safe = jnp.where(delta == 0.0, 1.0, delta)
    general = (jnp.power(p1, e + 1.0) - jnp.power(p0, e + 1.0)) \
        / ((e + 1.0) * safe)
    return jnp.where(delta == 0.0, jnp.power(p0, e), general) * dx


def _invert_power_linear_cdf(u, x_grid, prob_raw, cdf, raw_integral, e):
    """Exact inverse CDF of ``interp(prob)^e`` at quantile ``u``.

    The segment mass into t is ((q0+Δt)^(e+1) - q0^(e+1)) * L / ((e+1)Δ),
    inverted in closed form; a flat segment (Δ=0) reduces to the linear
    solution.
    """
    k = jnp.clip(jnp.searchsorted(cdf, u, side='right') - 1,
                 0, x_grid.shape[0] - 2)
    L = x_grid[k + 1] - x_grid[k]
    q0, q1 = prob_raw[k], prob_raw[k + 1]
    delta = q1 - q0
    w = (u - cdf[k]) * raw_integral          # unnormalised mass into segment
    safe_delta = jnp.where(delta == 0.0, 1.0, delta)
    rhs = jnp.power(q0, e + 1.0) + (e + 1.0) * safe_delta * w / L
    t_gen = (jnp.power(jnp.maximum(rhs, 0.0), 1.0 / (e + 1.0)) - q0) \
        / safe_delta
    t_flat = w / (L * jnp.power(jnp.maximum(q0, 1e-300), e))
    t = jnp.where(delta == 0.0, t_flat, t_gen)
    return x_grid[k] + jnp.clip(t, 0.0, 1.0) * L


def _invert_piecewise_linear_cdf(u, x_grid, prob, cdf):
    """Exact inverse CDF of a piecewise-linear PDF at quantile ``u``.

    Within an x interval the CDF of the linearly interpolated PDF is
    quadratic, 0.5*s*t^2 + p0*t = u - cdf[k] with s the PDF slope, solved in
    the citardauq form t = 2*du / (p0 + sqrt(p0^2 + 2*s*du)), which reduces
    exactly to the linear solution du/p0 when the slope is zero. The
    discriminant equals p1^2 at u = cdf[k+1], so it is never negative. A zero
    denominator can only occur on an interval with no probability mass, which
    ``searchsorted(side='right')`` never selects for a reachable u.

    ``u`` is the quantile in [0, 1] (scalar); ``x_grid`` is (n_x,) sorted
    parameter values; ``prob`` is (n_x,) PDF values normalised to unit
    integral; ``cdf`` is (n_x,) the normalised CDF at the x nodes.
    """
    k = jnp.clip(jnp.searchsorted(cdf, u, side='right') - 1,
                 0, x_grid.shape[0] - 2)
    du = u - cdf[k]
    p0 = prob[k]
    s = (prob[k + 1] - p0) / (x_grid[k + 1] - x_grid[k])
    denom = p0 + jnp.sqrt(p0 * p0 + 2.0 * s * du)
    t = jnp.where(denom > 0.0, 2.0 * du / denom, 0.0)
    return x_grid[k] + t


def sample_genpdf(key, genpdf):
    """Sample from an unconditional GENPDF via the exact inverse CDF.

    ``genpdf`` is a GenPDF instance with no conditioning variables.
    Returns the sampled value (scalar); if is_log, returns
    10^(sampled_value).
    """
    u = jax.random.uniform(key)
    if genpdf.expon_rewgt == 1.0:
        val = _invert_piecewise_linear_cdf(
            u, genpdf.x_grid, genpdf.prob[0], genpdf.cdf[0])
    else:
        val = _invert_power_linear_cdf(
            u, genpdf.x_grid, genpdf.prob_raw[0], genpdf.cdf[0],
            genpdf.raw_integral[0], genpdf.expon_rewgt)
    if genpdf.is_log:
        val = jnp.power(10.0, val)
    return val


def sample_genpdf_conditional(key, genpdf, *cond_values):
    """Sample from an N-D conditional GENPDF at a conditioning coordinate.

    SNANA rejection-samples the multilinearly interpolated raw PDF at the
    requested conditioning coordinate (getRan_genPDF,
    sntools_genPDF.c:691-865). That interpolated density is
    sum_i a_i * p_i(x), with a_i the multilinear weight of bracketing corner
    i; normalised, it is the mixture that selects corner i with probability
    a_i * I_i / sum_j a_j * I_j (I_i the corner's raw integral) and then
    samples corner i's exact normalised inverse. Both pieces are drawn here,
    so the draw follows exactly the normalised interpolated raw density.

    Conditioning values outside a grid are clamped to its edge (the
    GENPDF_OPTMASK EXTRAP behaviour; the caller validates the range when
    extrapolation is off).

    ``genpdf`` is a GenPDF with conditioning variables; ``cond_values`` is
    one scalar per conditioning variable. Returns the sampled value
    (scalar).
    """
    key_corner, key_x = jax.random.split(key)
    corner_slices, corner_weights = _bracketing_corners(genpdf, cond_values)

    u = jax.random.uniform(key_x)
    if genpdf.expon_rewgt == 1.0:
        # corner ~ a_i * I_i, then x within the corner
        weights = jnp.stack(corner_weights) * genpdf.raw_integral[corner_slices]
        csum = jnp.cumsum(weights)
        u_corner = jax.random.uniform(key_corner) * csum[-1]
        sel = corner_slices[jnp.searchsorted(csum, u_corner, side='right')]
        val = _invert_piecewise_linear_cdf(
            u, genpdf.x_grid, genpdf.prob[sel], genpdf.cdf[sel])
    else:
        # pow(interp_GRIDMAP, e) on the mixture (sntools_genPDF.c:773-774)
        mix = jnp.sum(jnp.stack(corner_weights)[:, None]
                      * genpdf.prob_raw[corner_slices], axis=0)
        seg = _segment_masses(mix, genpdf.x_grid[1:] - genpdf.x_grid[:-1],
                              genpdf.expon_rewgt)
        total = jnp.sum(seg)
        cdf = jnp.concatenate([jnp.zeros(1), jnp.cumsum(seg)]) / total
        val = _invert_power_linear_cdf(
            u, genpdf.x_grid, mix, cdf, total, genpdf.expon_rewgt)
    if genpdf.is_log:
        val = jnp.power(10.0, val)
    return val


def pdf_genpdf(genpdf, x, *cond_values):
    """The map's interpolated raw probability at ``x`` (on the map's own
    axis, so log10 of a LOG parameter) and the conditioning values
    (interp_GRIDMAP in funVal_genPDF, sntools_genPDF.c)."""
    corner_slices, corner_weights = _bracketing_corners(genpdf, cond_values)
    mix = jnp.sum(jnp.stack(corner_weights)[:, None]
                  * genpdf.prob_raw[corner_slices], axis=0)
    return jnp.interp(x, genpdf.x_grid, mix)


def _bracketing_corners(genpdf, cond_values):
    """The map slices bracketing a conditioning coordinate and their
    multilinear weights; a coordinate outside a grid is clamped to its
    edge (the GENPDF_OPTMASK EXTRAP behaviour)."""
    n_cond = len(genpdf.cond_varnames)
    cond_shape = genpdf.cond_shape

    # Row-major stride of each conditioning dimension in the slice axis.
    strides = []
    stride = 1
    for n_bins in reversed(cond_shape):
        strides.append(stride)
        stride *= n_bins
    strides = strides[::-1]

    los, his, fracs = [], [], []
    for ic in range(n_cond):
        grid = genpdf.cond_grids[ic]
        n_bins = cond_shape[ic]
        if n_bins == 1:
            los.append(jnp.zeros((), dtype=jnp.int32))
            his.append(jnp.zeros((), dtype=jnp.int32))
            fracs.append(jnp.zeros(()))
            continue
        lo = jnp.clip(jnp.searchsorted(grid, cond_values[ic]) - 1,
                      0, n_bins - 2)
        frac = jnp.clip((cond_values[ic] - grid[lo])
                        / (grid[lo + 1] - grid[lo]), 0.0, 1.0)
        los.append(lo * strides[ic])
        his.append((lo + 1) * strides[ic])
        fracs.append(frac)

    # 2^n_cond bracketing corners: geometric weight a_i times raw integral I_i.
    corner_slices = []
    corner_weights = []
    for corner in range(2 ** n_cond):
        flat = 0
        w = 1.0
        for ic in range(n_cond):
            if (corner >> ic) & 1:
                flat = flat + his[ic]
                w = w * fracs[ic]
            else:
                flat = flat + los[ic]
                w = w * (1.0 - fracs[ic])
        corner_slices.append(flat)
        corner_weights.append(w)
    return jnp.stack(corner_slices), corner_weights


jax.tree_util.register_dataclass(
    GenPDF,
    data_fields=[
        'x_grid',
        'prob',
        'cdf',
        'raw_integral',
        'cond_grids',
        'prob_raw',
    ],
    meta_fields=[
        'varname',
        'is_log',
        'cond_varnames',
        'cond_shape',
        'expon_rewgt',
    ],
)
