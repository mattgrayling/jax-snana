"""
FLUXERRMODEL_FILE reader and JAX evaluator.

Mirrors SNANA's FLUXERRMODEL feature (sntools_fluxErrModels.c) for empirical
per-observation flux uncertainty corrections. Each map is keyed by some subset
of {MJD, PSF, SKYSIG, ZP, LOGSNR, SBMAG, GALMAG, SNSEP} and supplies either an
ERRSCALE (multiplicative) or ERRADD (quadrature-additive) correction to the
flux uncertainty before noise sampling.

REDCOV entries describe between-band reduced correlations of the *fudge*
component (the additional variance introduced by FLUXERRMODEL on top of the
baseline noise model) within a field group.

File format (skipping DOCUMENTATION blocks)::

    OPT_EXTRAP: 1                  # skipped as inert: SNANA always clips
                                   # (FLUXERRMAP_EXTRAP.FLAG=1 unconditionally,
                                   # sntools_fluxErrModels.c:114)
    DEFINE_FIELDGROUP: SHALLOW  S1+S2+C1+C2
    DEFINE_FIELDGROUP: DEEP     X3+C3
    REDCOV(SHALLOW):  g:0.6,r:0.6,i:0.6,z:0.6
    REDCOV(DEEP):     g:0.6,r:0.6,i:0.6,z:0.6

    MAPNAME: FLUXERR_SCALE
    BAND:    g
    FIELD:   SHALLOW
    SCALE_FLUXERR_TRUE: 1.0
    SCALE_FLUXERR_DATA: 1.0
    VARNAMES: SBMAG ERRSCALE
    ROW: 20.5  2.967  # comment
    ROW: 21.5  2.967
    ...
    ENDMAP:

Reference: SNANA src/sntools_fluxErrModels.c
Variables: SNANA src/sntools_fluxErrModels.h
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import jax.numpy as jnp
from scipy.interpolate import interpn

from jax_snlc_core.io.docana import skip_docana
from jax_snlc_core.io.fits_path import open_text, resolve_fits_path
from jax_snlc_core.utils.error_handling import SNANAError


# Variable identifiers, matching SNANA's IPAR_FLUXERRMAP_*
SUPPORTED_VARNAMES: Tuple[str, ...] = (
    'MJD',
    'PSF',     # FWHM, arcsec
    'SKYSIG',  # ADU/pixel
    'ZP',      # ADU
    'LOGSNR',
    'SBMAG',   # mag/arcsec^2
    'GALMAG',  # galmag at NEA aperture, mag
    'SNSEP',   # SN-host separation, arcsec
)
VALUE_VARNAMES: Tuple[str, ...] = ('ERRSCALE', 'ERRADD')

# Default field-group name when a map's FIELD: is "ALL" or unset
ALL_GROUP = 'ALL'

# bit 1 SIM, bit 2 DATA; both by default (sntools_fluxErrModels.c:168)
MASK_APPLY_TRUE = 1
MASK_APPLY_DATA = 2
MASK_APPLY_DEFAULT = MASK_APPLY_TRUE | MASK_APPLY_DATA


@dataclass
class FluxErrMap:
    """One FLUXERRMODEL map (one MAPNAME/BAND/FIELD/VARNAMES block).

    The LAST varname is always 'ERRSCALE' or 'ERRADD' (the value column);
    the dimension varnames are `varnames[:-1]`.
    """
    name: str
    band: str
    field_group: str                       # DEFINE_FIELDGROUP name, or ALL_GROUP
    varnames: List[str]                    # including the value column
    grid_axes: List[np.ndarray]            # 1-D axis per dim varname (NDIM = NVAR-1)
    grid_values: np.ndarray
    is_errscale: bool
    mask_apply: int = MASK_APPLY_DEFAULT
    scale_true: float = 1.0
    scale_data: float = 1.0

    @property
    def dim_varnames(self) -> List[str]:
        return self.varnames[:-1]

    @property
    def n_dims(self) -> int:
        return len(self.varnames) - 1


@dataclass
class RedCovEntry:
    """One REDCOV(...): bands_str entry."""
    field_group: str          # ALL_GROUP for global, else a DEFINE_FIELDGROUP name
    bands: str
    redcov: float             # in [0, 1)


@dataclass
class FluxErrModelData:
    """Parsed contents of a FLUXERRMODEL_FILE."""
    maps: List[FluxErrMap] = field(default_factory=list)
    field_groups: Dict[str, List[str]] = field(default_factory=dict)
    redcov: List[RedCovEntry] = field(default_factory=list)


def read_fluxerr_model_file(filepath: str) -> FluxErrModelData:
    """Parse a FLUXERRMODEL_FILE into its maps, field groups and REDCOV rows.

    Environment variables in ``filepath`` are expanded.
    """
    expanded = Path(os.path.expandvars(filepath))
    with open_text(resolve_fits_path(expanded, required=True)) as f:
        lines = f.readlines()

    data = FluxErrModelData()

    # Per-map state, accumulated until ENDMAP:
    cur_name: Optional[str] = None
    cur_band: str = 'ALL'
    cur_field: str = ALL_GROUP
    cur_scale_true: float = 1.0
    cur_scale_data: float = 1.0
    cur_varnames: Optional[List[str]] = None
    cur_rows: List[List[float]] = []

    def _finalise_map():
        nonlocal cur_name, cur_band, cur_field, cur_scale_true, cur_scale_data
        nonlocal cur_varnames, cur_rows
        if cur_name is None:
            return
        if cur_varnames is None:
            raise SNANAError(
                f"FLUXERRMODEL map {cur_name!r} missing VARNAMES",
                function_name="_finalise_map")
        if not cur_rows:
            raise SNANAError(
                f"FLUXERRMODEL map {cur_name!r} has no ROW entries",
                function_name="_finalise_map")
        m = _build_map(cur_name, cur_band, cur_field, cur_varnames, cur_rows,
                       cur_scale_true, cur_scale_data)
        data.maps.append(m)
        cur_name = None
        cur_band = 'ALL'
        cur_field = ALL_GROUP
        cur_scale_true = 1.0
        cur_scale_data = 1.0
        cur_varnames = None
        cur_rows = []

    for raw in skip_docana(lines):
        stripped = raw.strip()
        if not stripped or stripped.startswith('#'):
            continue

        upper = stripped.upper()

        # inert: SNANA always clips (sntools_fluxErrModels.c:114)
        if upper.startswith('OPT_EXTRAP:'):
            continue

        if upper.startswith('DEFINE_FIELDGROUP:'):
            tail = stripped.split(':', 1)[1].strip().split()
            if len(tail) < 2:
                raise SNANAError(
                    f"DEFINE_FIELDGROUP needs name + field list: {stripped!r}",
                    function_name="read_fluxerr_model_file")
            group_name = tail[0]
            field_list_str = ' '.join(tail[1:])
            field_names = [f.strip() for f in field_list_str.replace(',', '+').split('+')
                           if f.strip()]
            data.field_groups[group_name] = field_names
            continue

        if upper.startswith('REDCOV(') or upper.startswith('REDCOV:'):
            data.redcov.extend(parse_redcov_string(stripped))
            continue

        if upper.startswith('MAPNAME:'):
            _finalise_map()
            cur_name = stripped.split(':', 1)[1].strip().split()[0]
            cur_band = 'ALL'
            cur_field = ALL_GROUP
            cur_scale_true = 1.0
            cur_scale_data = 1.0
            cur_varnames = None
            cur_rows = []
            continue

        # BAND:/FILTER: and FIELD: may share a line
        if (upper.startswith('BAND:') or upper.startswith('FILTER:')
                or upper.startswith('FIELD:')):
            tokens = stripped.split()
            i = 0
            while i < len(tokens):
                tok = tokens[i].upper().rstrip(':')
                if tok in ('BAND', 'FILTER') and i + 1 < len(tokens):
                    cur_band = tokens[i + 1]
                    i += 2
                elif tok == 'FIELD' and i + 1 < len(tokens):
                    cur_field = tokens[i + 1]
                    i += 2
                else:
                    i += 1
            continue

        if upper.startswith('SCALE_FLUXERR_TRUE:'):
            cur_scale_true = float(stripped.split(':', 1)[1].strip().split()[0])
            continue
        if upper.startswith('SCALE_FLUXERR_DATA:'):
            cur_scale_data = float(stripped.split(':', 1)[1].strip().split()[0])
            continue

        if upper.startswith('VARNAMES:'):
            parts = stripped.split(':', 1)[1].strip().split()
            if len(parts) < 2:
                raise SNANAError(
                    f"VARNAMES needs at least one dim + value column: {stripped!r}",
                    function_name="read_fluxerr_model_file")
            value_col = parts[-1].upper()
            if value_col not in VALUE_VARNAMES:
                raise SNANAError(
                    f"VARNAMES last column must be one of {VALUE_VARNAMES}, "
                    f"got {value_col!r} in map {cur_name!r}",
                    function_name="read_fluxerr_model_file")
            for v in parts[:-1]:
                vu = v.upper()
                if vu not in SUPPORTED_VARNAMES:
                    raise SNANAError(
                        f"FLUXERRMODEL: unsupported variable {v!r} in map "
                        f"{cur_name!r}. Supported: {SUPPORTED_VARNAMES}",
                        function_name="read_fluxerr_model_file")
            cur_varnames = [v.upper() for v in parts]
            continue

        # NVAR is informational; skip
        if upper.startswith('NVAR:'):
            continue

        if upper.startswith('ROW:'):
            row_str = stripped.split(':', 1)[1].strip()
            if '#' in row_str:
                row_str = row_str.split('#', 1)[0].strip()
            vals = [float(x) for x in row_str.split()]
            if cur_varnames is None:
                raise SNANAError(
                    f"ROW before VARNAMES in map {cur_name!r}",
                    function_name="read_fluxerr_model_file")
            if len(vals) != len(cur_varnames):
                raise SNANAError(
                    f"ROW has {len(vals)} columns but VARNAMES has "
                    f"{len(cur_varnames)} in map {cur_name!r}",
                    function_name="read_fluxerr_model_file")
            cur_rows.append(vals)
            continue

        if upper.startswith('ENDMAP'):
            _finalise_map()
            continue

    _finalise_map()

    # one VARNAMES and grid across maps (SNANA allows mixed; jax does not)
    if data.maps:
        varname_set = {tuple(m.varnames) for m in data.maps}
        if len(varname_set) != 1:
            raise SNANAError(
                f"FLUXERRMODEL_FILE {expanded!r}: maps use mixed VARNAMES, "
                f"which is not supported. Maps use: {sorted(varname_set)}",
                function_name="read_fluxerr_model_file")

        ax_signatures = {
            tuple(tuple(round(float(v), 9) for v in a) for a in m.grid_axes)
            for m in data.maps
        }
        if len(ax_signatures) != 1:
            raise SNANAError(
                f"FLUXERRMODEL_FILE {expanded!r}: maps use different grid "
                f"axes, which is not supported. All maps must share the same "
                f"grid for each dimension.", function_name="read_fluxerr_model_file")

        _check_fluxerrmodel_map_overlap(data)

    return data


def _map_fieldlist_str(m: FluxErrMap, field_groups: Dict[str, List[str]]) -> str:
    """The FIELDLIST string a map matches against: a FIELD: naming a
    DEFINE_FIELDGROUP expands to the group's field list, any other FIELD:
    stands as a literal SIMLIB field list (set_FIELDLIST_FLUXERRMODEL,
    sntools_fluxErrModels.c:846-874)."""
    if m.field_group == ALL_GROUP:
        return ALL_GROUP
    if m.field_group in field_groups:
        return '+'.join(field_groups[m.field_group])
    return m.field_group


def match_fluxerrmodel_map(maps: List[FluxErrMap],
                           field_groups: Dict[str, List[str]],
                           band_char: str, field_name: str) -> int:
    """Index of the single map serving (band, field), or -1 for none.

    SNANA's INDEX_MAP_FLUXERRMODEL rule (sntools_fluxErrModels.c:877-923):
    BANDLIST 'ALL' or the band char as a substring; FIELDLIST 'ALL' or the
    field name as a substring of the (group-expanded) field list. More
    than one match is fatal.
    """
    n_match, imap = 0, -1
    for i, m in enumerate(maps):
        if not (m.band == ALL_GROUP or band_char in m.band):
            continue
        fieldlist = _map_fieldlist_str(m, field_groups)
        if not (fieldlist == ALL_GROUP or field_name in fieldlist):
            continue
        n_match += 1
        imap = i
    if n_match > 1:
        raise SNANAError(
            f"Invalid NMATCH={n_match} FLUXERRMODEL maps for BAND={band_char!r} "
            f"and FIELD={field_name!r}.",
            function_name='match_fluxerrmodel_map')
    return imap


def _check_fluxerrmodel_map_overlap(data: FluxErrModelData) -> None:
    """Abort at parse time when two maps can claim the same (band, field)
    observation, which SNANA's NMATCH>1 abort would hit on the first such
    obs (INDEX_MAP_FLUXERRMODEL, sntools_fluxErrModels.c:917-922)."""
    def field_tokens(m):
        fl = _map_fieldlist_str(m, data.field_groups)
        return None if fl == ALL_GROUP else {
            t for t in fl.replace(',', '+').split('+') if t}

    def band_tokens(m):
        return None if m.band == ALL_GROUP else set(m.band)

    for i, a in enumerate(data.maps):
        for b in data.maps[i + 1:]:
            ba, bb = band_tokens(a), band_tokens(b)
            if ba is not None and bb is not None and not (ba & bb):
                continue
            fa, fb = field_tokens(a), field_tokens(b)
            if fa is not None and fb is not None and not (fa & fb):
                continue
            raise SNANAError(
                f"FLUXERRMODEL maps {a.name!r} (BAND={a.band!r}, "
                f"FIELD={a.field_group!r}) and {b.name!r} (BAND={b.band!r}, "
                f"FIELD={b.field_group!r}) both match some (band, field); SNANA "
                f"aborts on NMATCH>1.",
                function_name='_check_fluxerrmodel_map_overlap')


def _build_map(name: str, band: str, field_group: str,
               varnames: List[str], rows: List[List[float]],
               scale_true: float, scale_data: float) -> FluxErrMap:
    """Build a FluxErrMap from parsed VARNAMES + ROW data.

    SNANA enforces uniform binning per axis (`sntools_gridmap.c:31` aborts on
    non-uniform), and so does this. Vectorised, with no nested per-row
    loops.
    """
    n_dim = len(varnames) - 1
    is_errscale = varnames[-1].upper() == 'ERRSCALE'

    arr = np.asarray(rows, dtype=np.float64)
    dim_data = arr[:, :n_dim]      # (n_rows, n_dim)
    val_data = arr[:, n_dim]       # (n_rows,)

    grid_axes: List[np.ndarray] = [np.unique(dim_data[:, d]) for d in range(n_dim)]

    for d, ax in enumerate(grid_axes):
        if len(ax) < 2:
            continue
        diffs = np.diff(ax)
        if not np.allclose(diffs, diffs[0], rtol=1e-5, atol=1e-12):
            raise SNANAError(
                f"FLUXERRMODEL map {name!r}, dim {varnames[d]!r}: "
                f"non-uniform grid spacing not supported (diffs={diffs[:5]}...)",
                function_name="_build_map")

    grid_shape = tuple(len(a) for a in grid_axes)
    expected = int(np.prod(grid_shape))
    if expected != len(val_data):
        raise SNANAError(
            f"FLUXERRMODEL map {name!r}: expected {expected} rows "
            f"(from axes {[len(a) for a in grid_axes]}), got {len(val_data)}",
            function_name="_build_map")

    idx_per_dim: List[np.ndarray] = []
    for d, ax in enumerate(grid_axes):
        idx = np.searchsorted(ax, dim_data[:, d])
        if not np.allclose(ax[np.clip(idx, 0, len(ax) - 1)], dim_data[:, d]):
            raise SNANAError(
                f"FLUXERRMODEL map {name!r}, dim {varnames[d]!r}: row values "
                f"not on grid axis",
                function_name="_build_map")
        idx_per_dim.append(idx)

    grid_values = np.empty(grid_shape, dtype=np.float32)
    grid_values[tuple(idx_per_dim)] = val_data.astype(np.float32)

    return FluxErrMap(
        name=name,
        band=band,
        field_group=field_group,
        varnames=varnames,
        grid_axes=grid_axes,
        grid_values=grid_values,
        is_errscale=is_errscale,
        mask_apply=MASK_APPLY_DEFAULT,
        scale_true=scale_true,
        scale_data=scale_data,
    )


def parse_redcov_string(s: str) -> List[RedCovEntry]:
    """Parse a REDCOV string in the SNANA format.

    Forms:
        "REDCOV(GROUP): g:0.6,r:0.6"          (file-level keyword)
        "REDCOV: g:0.6"                        (file-level keyword, global group)
        "FLUXERRMODEL_REDCOV(GROUP) g:0.6,r:0.6"  (config override)
        "FLUXERRMODEL_REDCOV g:0.6"               (config override, global)
        "g:0.6,r:0.6"                          (raw bands string, global group)

    Returns one entry per band token.

    Multiple bands sharing one REDCOV value are returned as a single entry
    with `bands` as the concatenated string (e.g., "griz").
    """
    s = s.strip()

    field_group = ALL_GROUP

    upper = s.upper()
    if upper.startswith('FLUXERRMODEL_REDCOV') or upper.startswith('REDCOV'):
        i_paren = s.find('(')
        i_colon = s.find(':')
        if i_paren >= 0 and (i_colon < 0 or i_paren < i_colon):
            j = s.find(')', i_paren + 1)
            if j < 0:
                raise SNANAError(f"Unclosed REDCOV group in {s!r}",
                                 function_name="parse_redcov_string")
            field_group = s[i_paren + 1:j].strip()
            tail = s[j + 1:].strip()
            if tail.startswith(':'):
                tail = tail[1:].strip()
        else:
            tail = s.split(':', 1)[1].strip() if ':' in s else s.split(None, 1)[1]
        s = tail

    entries: List[RedCovEntry] = []
    for token in s.split(','):
        token = token.strip()
        if not token:
            continue
        if ':' not in token:
            raise SNANAError(f"REDCOV token missing colon: {token!r}",
                             function_name="parse_redcov_string")
        bands_str, val_str = token.split(':', 1)
        bands_str = bands_str.strip()
        val = float(val_str.strip())
        if not (0.0 <= val < 1.0):
            raise SNANAError(
                f"REDCOV value must be in [0, 1), got {val} for bands {bands_str!r}",
                function_name="parse_redcov_string")
        entries.append(RedCovEntry(field_group=field_group, bands=bands_str, redcov=val))

    return entries


# queries clipped to the grid: SNANA always clips (sntools_fluxErrModels.c:114)


def interp_1d(grid_per_obs: jnp.ndarray, query: jnp.ndarray,
              origin: float, step: float, n_grid: int) -> jnp.ndarray:
    """Linear interpolation along one axis, returning (n_sne, max_obs)
    interpolated values, clipped to the axis range.

    ``grid_per_obs`` is the per-(sn, obs) gathered grid slice, shape
    (n_sne, max_obs, n_grid); ``query`` is the per-(sn, obs) query value,
    shape (n_sne, max_obs, 1); ``origin`` and ``step`` are the axis's first
    value and uniform step; ``n_grid`` is the axis length.
    """
    x = (query[..., 0] - origin) / step                   # index space
    x = jnp.clip(x, 0.0, float(n_grid - 1))
    idx_lo = jnp.floor(x).astype(jnp.int32)
    idx_hi = jnp.minimum(idx_lo + 1, n_grid - 1)
    frac = x - idx_lo.astype(x.dtype)
    v_lo = jnp.take_along_axis(grid_per_obs, idx_lo[..., None], axis=-1)[..., 0]
    v_hi = jnp.take_along_axis(grid_per_obs, idx_hi[..., None], axis=-1)[..., 0]
    return (1.0 - frac) * v_lo + frac * v_hi


def interp_2d(grid_per_obs: jnp.ndarray, query: jnp.ndarray,
              origins: Tuple[float, float], steps: Tuple[float, float],
              n_grid: Tuple[int, int]) -> jnp.ndarray:
    """Bilinear interpolation on a 2-D regular grid, returning (n_sne,
    max_obs) interpolated values.

    ``grid_per_obs`` is the per-(sn, obs) gathered slice, shape (n_sne,
    max_obs, n0, n1); ``query`` is the per-(sn, obs) query values, shape
    (n_sne, max_obs, 2); ``origins`` is (origin0, origin1); ``steps`` is
    (step0, step1); ``n_grid`` is (n0, n1).
    """
    o0, o1 = origins
    s0, s1 = steps
    n0, n1 = n_grid

    x0 = jnp.clip((query[..., 0] - o0) / s0, 0.0, float(n0 - 1))
    x1 = jnp.clip((query[..., 1] - o1) / s1, 0.0, float(n1 - 1))

    i0 = jnp.floor(x0).astype(jnp.int32)
    j0 = jnp.floor(x1).astype(jnp.int32)
    i1 = jnp.minimum(i0 + 1, n0 - 1)
    j1 = jnp.minimum(j0 + 1, n1 - 1)
    f0 = x0 - i0.astype(x0.dtype)
    f1 = x1 - j0.astype(x1.dtype)

    def gather(g, ii, jj):
        # g: (n_sne, max_obs, n0, n1) -> select one element per (sn, obs)
        col = jnp.take_along_axis(g, ii[..., None, None].repeat(n1, axis=-1), axis=-2)[..., 0, :]
        return jnp.take_along_axis(col, jj[..., None], axis=-1)[..., 0]

    v00 = gather(grid_per_obs, i0, j0)
    v01 = gather(grid_per_obs, i0, j1)
    v10 = gather(grid_per_obs, i1, j0)
    v11 = gather(grid_per_obs, i1, j1)

    return ((1.0 - f0) * (1.0 - f1) * v00 +
            (1.0 - f0) * f1 * v01 +
            f0 * (1.0 - f1) * v10 +
            f0 * f1 * v11)


def interp_3d(grid_per_obs: jnp.ndarray, query: jnp.ndarray,
              origins: Tuple[float, float, float],
              steps: Tuple[float, float, float],
              n_grid: Tuple[int, int, int]) -> jnp.ndarray:
    """Trilinear interpolation on a 3-D regular grid.

    Same shape conventions as `interp_2d`, with one extra axis.
    """
    o0, o1, o2 = origins
    s0, s1, s2 = steps
    n0, n1, n2 = n_grid

    x0 = jnp.clip((query[..., 0] - o0) / s0, 0.0, float(n0 - 1))
    x1 = jnp.clip((query[..., 1] - o1) / s1, 0.0, float(n1 - 1))
    x2 = jnp.clip((query[..., 2] - o2) / s2, 0.0, float(n2 - 1))

    i0 = jnp.floor(x0).astype(jnp.int32); i1 = jnp.minimum(i0 + 1, n0 - 1)
    j0 = jnp.floor(x1).astype(jnp.int32); j1 = jnp.minimum(j0 + 1, n1 - 1)
    k0 = jnp.floor(x2).astype(jnp.int32); k1 = jnp.minimum(k0 + 1, n2 - 1)
    f0 = x0 - i0.astype(x0.dtype)
    f1 = x1 - j0.astype(x1.dtype)
    f2 = x2 - k0.astype(x2.dtype)

    flat = grid_per_obs.reshape(grid_per_obs.shape[0],
                                grid_per_obs.shape[1],
                                n0 * n1 * n2)

    def lin(ii, jj, kk):
        return ii * (n1 * n2) + jj * n2 + kk

    def take(idx_flat):
        return jnp.take_along_axis(flat, idx_flat[..., None], axis=-1)[..., 0]

    v000 = take(lin(i0, j0, k0))
    v001 = take(lin(i0, j0, k1))
    v010 = take(lin(i0, j1, k0))
    v011 = take(lin(i0, j1, k1))
    v100 = take(lin(i1, j0, k0))
    v101 = take(lin(i1, j0, k1))
    v110 = take(lin(i1, j1, k0))
    v111 = take(lin(i1, j1, k1))

    g0 = 1.0 - f0; g1 = 1.0 - f1; g2 = 1.0 - f2
    return (g0 * g1 * g2 * v000 +
            g0 * g1 * f2 * v001 +
            g0 * f1 * g2 * v010 +
            g0 * f1 * f2 * v011 +
            f0 * g1 * g2 * v100 +
            f0 * g1 * f2 * v101 +
            f0 * f1 * g2 * v110 +
            f0 * f1 * f2 * v111)



def evaluate_fluxerrmodel_data(model, band_chars, fields, err, conditions):
    """Fit-side FLUXERRMODEL (get_FLUXERRMODEL): each epoch gets THE single
    DATA-masked map selected by `match_fluxerrmodel_map` (SNANA
    INDEX_MAP_FLUXERRMODEL, sntools_fluxErrModels.c:877-923): substring
    BAND/FIELD matching with 'ALL', group expansion with the literal
    SIMLIB-field fallback, and a fatal on multiple matches. `conditions`
    maps each dim VARNAME to a per-epoch array; queries are clipped to the
    grid edges (SNANA always clips). An ERRSCALE map multiplies the error,
    ERRADD adds in quadrature, then SCALE_FLUXERR_DATA.
    """
    band_chars = np.asarray(band_chars)
    fields = np.asarray(fields)
    err = np.array(err, dtype=np.float64)
    for band, field_name in sorted(set(zip(band_chars.tolist(),
                                           fields.tolist()))):
        imap = match_fluxerrmodel_map(
            model.maps, model.field_groups, band, field_name)
        if imap < 0:
            continue
        m = model.maps[imap]
        if not (m.mask_apply & MASK_APPLY_DATA):
            continue
        sel = (band_chars == band) & (fields == field_name)
        pts = np.stack([np.clip(np.asarray(conditions[v], dtype=np.float64),
                                m.grid_axes[i][0], m.grid_axes[i][-1])
                        for i, v in enumerate(m.dim_varnames)], axis=-1)
        val = interpn(m.grid_axes, m.grid_values, pts, method='linear',
                      bounds_error=False, fill_value=None)
        out = (err * val if m.is_errscale
               else np.sqrt(err * err + val * val)) * m.scale_data
        err = np.where(sel, out, err)
    return err
