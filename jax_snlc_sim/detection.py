"""Survey detection, PHOTPROB scores and trigger logic.

Detection draws mark individual observations. PHOTPROB scores can exclude
those observations from the event trigger without changing their detection
flags. Scores are sampled from conditional CDFs, with optional correlations
between observations.

The efficiency file carries three kinds of block, and SNANA reads them in one pass
(init_SEARCHEFF, sntools_trigger.c:330-334): the detection curves that give
a per-observation efficiency against SNR, |SNR| or observed magnitude
(readMap_SEARCHEFF_DETECT, :358-423), the PHOTPROB maps
(readMap_SEARCHEFF_PHOTPROB), and the PHOTFLAG_DETECT / PHOTFLAG_TRIGGER
bits, each of which the file supplies only while the input file has left it
at zero (:321-325).
"""

import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm as jax_norm

from jax_snlc_core import logging as log
from jax_snlc_core.filters import filter_band
from jax_snlc_core.interpolation import interp_nd_regular
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import (MAG_ZEROFLUX, check_file_docana, is_ignorefile,
                                       open_text, skip_docana)
from jax_snlc_sim.spectroscopic_efficiency import (
    resolve_searcheff_file, resolve_fieldlist_indices)

logger = logging.getLogger(__name__)


# Variable type indices matching SNANA sntools_trigger.h:96-101
IVARABS_SNR = 0
IVARABS_LOGSNR = 1
IVARABS_SBMAG = 2
IVARABS_REDSHIFT = 3
IVARABS_GALMAG = 4

VARNAME_TO_IVARABS = {
    'SNR': IVARABS_SNR,
    'LOGSNR': IVARABS_LOGSNR,
    'SBMAG': IVARABS_SBMAG,
    'GALMAG': IVARABS_GALMAG,
    'REDSHIFT': IVARABS_REDSHIFT,
}


@dataclass
class PhotoprobMap:
    """One PHOTPROB map from SEARCHEFF_PIPELINE_EFF_FILE."""
    name: str
    varnames: List[str]
    ivarabs: List[int]            # absolute variable type indices
    nvar_map: int
    nfun_cdf: int
    grid_axes: List[np.ndarray]   # unique values per dimension
    grid_cdf: np.ndarray          # (*axis_sizes, nfun_cdf), cumulative CDF weights
    cdf_bins: np.ndarray          # [0, 1/K, ..., 1], length K+1
    valmin: np.ndarray            # per-variable clamp range
    valmax: np.ndarray
    fieldlist: str = 'ALL'
    filterlist: str = 'ALL'
    require_detection: bool = False
    cutval: float = -999.0        # reject obs with PHOTPROB <= cutval
    reduced_corr: float = 0.0     # inter-observation correlation
    # per SIMLIB entry, from `fieldlist`; None for 'ALL'
    fieldlist_mask: Optional[np.ndarray] = None
    filter_mask: Optional[np.ndarray] = None


def build_photprob_map(info: dict, data_rows: list) -> PhotoprobMap:
    """Build a PhotoprobMap from parsed data.

    The grid is filled by each row's coordinates rather than assuming row
    order; a duplicate coordinate, a missing cell, a negative weight or a
    zero-mass weight row is malformed input and fatal.
    """
    name = info['name']
    nvar_map = info['nvar_map']
    nfun_cdf = info['nfun_cdf']
    data = np.array(data_rows)
    var_data = data[:, :nvar_map]
    wgt_data = data[:, nvar_map:]

    if np.any(wgt_data < 0.0):
        raise SNANAError(
            f"PHOTPROB map {name!r} has negative PHOTPROB_WGTLIST entries.",
            function_name='build_photprob_map')
    row_sums = wgt_data.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise SNANAError(
            f"PHOTPROB map {name!r} has ROW(s) whose PHOTPROB_WGTLIST sums to "
            f"zero; the CDF is undefined.",
            function_name='build_photprob_map')
    # Convert weights to cumulative CDF (SNANA LOAD_PHOTPROB_CDF)
    cumulative = np.cumsum(wgt_data, axis=1) / row_sums

    # Track VALMIN/VALMAX per variable (sntools_trigger.c:631-636)
    valmin = var_data.min(axis=0)
    valmax = var_data.max(axis=0)

    grid_axes = [np.unique(var_data[:, d]) for d in range(nvar_map)]
    shape = tuple(len(ax) for ax in grid_axes)
    idx_per_dim = tuple(
        np.searchsorted(grid_axes[d], var_data[:, d]) for d in range(nvar_map))
    filled = np.zeros(shape, dtype=bool)
    filled[idx_per_dim] = True
    flat_cell = np.ravel_multi_index(idx_per_dim, shape)
    if len(np.unique(flat_cell)) < len(data_rows):
        raise SNANAError(
            f"PHOTPROB map {name!r} defines the same grid coordinate more "
            f"than once.",
            function_name='build_photprob_map')
    if not filled.all():
        raise SNANAError(
            f"PHOTPROB map {name!r} is missing "
            f"{int(filled.size - filled.sum())} grid cell(s) of the "
            f"{shape} outer-product grid.",
            function_name='build_photprob_map')

    grid_cdf = np.zeros(shape + (nfun_cdf,), dtype=np.float64)
    grid_cdf[idx_per_dim] = cumulative
    cdf_bins = np.linspace(0.0, 1.0, nfun_cdf + 1)

    return PhotoprobMap(
        name=info['name'],
        varnames=info['varnames'],
        ivarabs=info['ivarabs'],
        nvar_map=nvar_map,
        nfun_cdf=nfun_cdf,
        grid_axes=grid_axes,
        grid_cdf=grid_cdf,
        cdf_bins=cdf_bins,
        fieldlist=info['fieldlist'],
        filterlist=info['filterlist'],
        require_detection=info['require_detection'],
        cutval=info['cutval'],
        reduced_corr=info['reduced_corr'],
        valmin=valmin,
        valmax=valmax,
    )


def prepare_photprob(maps, filter_names):
    """Pre-resolve filter/field matching for JIT-compiled PHOTPROB evaluation.

    Called ONCE at setup time (Python). ``maps`` is the List[PhotoprobMap]
    from read_pipeline_eff_file(), with `fieldlist_mask` resolved by
    resolve_fieldlist_indices; ``filter_names`` is the list of
    filter names (e.g., ['LSST-u', ...]). Returns prepared maps with JAX arrays.

    Raises `SNANAError` when more than one map matches a (field, filter)
    observation (setObs_for_PHOTPROB, sntools_trigger.c:2589-2594).
    """
    filt_chars = [filter_band(fn) for fn in filter_names]

    prepared = []
    for m in maps:
        if m.filterlist == 'ALL':
            filter_mask = np.ones(len(filter_names), dtype=bool)
        else:
            filter_mask = np.zeros(len(filter_names), dtype=bool)
            for fi, fc in enumerate(filt_chars):
                if fc in m.filterlist:
                    filter_mask[fi] = True

        prepared.append(replace(
            m,
            grid_axes=[jnp.asarray(ax, dtype=jnp.float32) for ax in m.grid_axes],
            grid_cdf=jnp.asarray(m.grid_cdf, dtype=jnp.float32),
            cdf_bins=jnp.asarray(m.cdf_bins, dtype=jnp.float32),
            filter_mask=jnp.asarray(filter_mask),
            valmin=jnp.asarray(m.valmin, dtype=jnp.float32),
            valmax=jnp.asarray(m.valmax, dtype=jnp.float32),
            fieldlist_mask=(None if m.fieldlist_mask is None
                            else jnp.asarray(m.fieldlist_mask)),
        ))

    # setObs_for_PHOTPROB NMATCH>1 aborts (sntools_trigger.c:2589-2594)
    n_entries = max((len(m.fieldlist_mask) for m in maps
                     if m.fieldlist_mask is not None), default=1)
    n_match = np.zeros((n_entries, len(filter_names)), dtype=np.int32)
    for m, pm in zip(maps, prepared):
        fmask = np.asarray(pm.filter_mask)
        emask = (np.asarray(m.fieldlist_mask)
                 if m.fieldlist_mask is not None
                 else np.ones(n_entries, dtype=bool))
        n_match += emask[:, None] & fmask[None, :]
    if np.any(n_match > 1):
        e, f = np.argwhere(n_match > 1)[0]
        raise SNANAError(
            f"{int(n_match[e, f])} PHOTPROB maps match SIMLIB field entry "
            f"{int(e)}, filter {filter_names[int(f)]!r}; SNANA aborts on multiple "
            f"matches.",
            function_name='prepare_photprob')

    return prepared


def evaluate_photprob(prepared_maps, snr_2d, filter_indices_2d,
                      detection_mask_2d, valid_mask_2d, redshift, sbmag_2d,
                      galmag_2d, field_match_per_map, rng_key, sn_idx):
    """Evaluate prepared PHOTPROB maps (the output of prepare_photprob)
    against batched per-obs inputs.

    Per-observation CDF interpolation + inverse CDF sampling. Operates on
    2D (n_sne, max_obs) arrays with vmap. ``field_match_per_map`` is a list
    of (n_sne,) bool per map, pre-computed from FIELDLIST strings.
    ``sbmag_2d`` and ``galmag_2d`` are (n_sne, n_filters) host surface
    brightness and galaxy magnitudes, 99 where unavailable; ``sn_idx`` is
    the per-SN index the random draws fold in.

    Returns ``(photprob_2d, photprob_pass_2d)``: ``photprob_2d`` is
    (n_sne, max_obs) float, -9.0 where not evaluated; ``photprob_pass_2d``
    is (n_sne, max_obs) bool.
    """
    n_sne, max_obs = snr_2d.shape
    photprob_2d = jnp.full((n_sne, max_obs), -9.0, dtype=jnp.float32)
    photprob_pass_2d = jnp.ones((n_sne, max_obs), dtype=bool)

    # disjoint: prepare_photprob allows one map per (field, filter)
    applicable_per_map = []
    for pm, field_match in zip(prepared_maps, field_match_per_map):
        applicable = valid_mask_2d
        filt_match = pm.filter_mask[jnp.maximum(filter_indices_2d, 0)]
        filt_match = filt_match & (filter_indices_2d >= 0)
        applicable = applicable & filt_match & field_match[:, None]
        if pm.require_detection:
            applicable = applicable & detection_mask_2d
        applicable_per_map.append(applicable)

    # shared-normal form of setRan_for_PHOTPROB's covariance (sntools_trigger.c:2679)
    corr_2d = jnp.zeros((n_sne, max_obs), dtype=jnp.float32)
    for pm, applicable in zip(prepared_maps, applicable_per_map):
        corr_2d = jnp.where(applicable, jnp.float32(pm.reduced_corr),
                            corr_2d)
    sn_keys = jax.vmap(lambda i: jax.random.fold_in(rng_key, i))(sn_idx)

    def draw_one_sn(key):
        k_common, k_indep = jax.random.split(key)
        eps_common = jax.random.normal(k_common)
        eps_indep = jax.random.normal(k_indep, (max_obs,))
        return eps_common, eps_indep

    eps_common, eps_indep = jax.vmap(draw_one_sn)(sn_keys)
    z_corr = (jnp.sqrt(1.0 - corr_2d) * eps_indep
              + jnp.sqrt(corr_2d) * eps_common[:, None])
    uniforms_2d = jax_norm.cdf(z_corr)

    for imap, pm in enumerate(prepared_maps):
        applicable = applicable_per_map[imap]
        n_vars = pm.nvar_map
        obs_points = jnp.zeros((n_sne, max_obs, n_vars), dtype=jnp.float32)
        for d, ivarabs in enumerate(pm.ivarabs):
            if ivarabs == IVARABS_SNR:
                vals = jnp.maximum(snr_2d, 0.1)
            elif ivarabs == IVARABS_LOGSNR:
                vals = jnp.log10(jnp.maximum(snr_2d, 0.1))
            elif ivarabs == IVARABS_SBMAG:
                safe_fi = jnp.maximum(filter_indices_2d, 0)
                vals = sbmag_2d[jnp.arange(n_sne)[:, None], safe_fi]
            elif ivarabs == IVARABS_GALMAG:
                safe_fi = jnp.maximum(filter_indices_2d, 0)
                vals = galmag_2d[jnp.arange(n_sne)[:, None], safe_fi]
            else:  # IVARABS_REDSHIFT
                vals = jnp.broadcast_to(redshift[:, None], (n_sne, max_obs))
            obs_points = obs_points.at[:, :, d].set(vals)

        for d in range(n_vars):
            safety = 1e-9 * (pm.valmax[d] - pm.valmin[d])
            obs_points = obs_points.at[:, :, d].set(
                jnp.clip(obs_points[:, :, d],
                         pm.valmin[d] + safety, pm.valmax[d] - safety))

        flat_points = obs_points.reshape(-1, n_vars)
        cdf_values = jnp.zeros((n_sne * max_obs, pm.nfun_cdf + 1))
        for k in range(pm.nfun_cdf):
            grid_k = pm.grid_cdf[..., k]
            cdf_k = jnp.clip(interp_nd_regular(flat_points, pm.grid_axes,
                                               grid_k, opt_extrap=0), 0.0, 1.0)
            cdf_values = cdf_values.at[:, k + 1].set(cdf_k)
        cdf_values_2d = cdf_values.reshape(n_sne, max_obs, pm.nfun_cdf + 1)

        cdf_bins = pm.cdf_bins

        def inv_cdf_one_obs(uniform, cdf_row):
            return jnp.interp(uniform, cdf_row, cdf_bins)

        photprob_vals = jax.vmap(jax.vmap(inv_cdf_one_obs))(uniforms_2d, cdf_values_2d)
        photprob_2d = jnp.where(applicable, photprob_vals, photprob_2d)

        # PHOTPROB <= CUTVAL fails (sntools_trigger.c:2238-2241)
        fail = applicable & (pm.cutval > -900) & (photprob_vals <= pm.cutval)
        photprob_pass_2d = jnp.where(fail, False, photprob_pass_2d)

    return photprob_2d, photprob_pass_2d


jax.tree_util.register_dataclass(
    PhotoprobMap,
    data_fields=[
        'grid_axes',
        'grid_cdf',
        'cdf_bins',
        'valmin',
        'valmax',
        'cutval',
        'reduced_corr',
        'fieldlist_mask',
        'filter_mask',
    ],
    meta_fields=[
        'name',
        'varnames',
        'ivarabs',
        'nvar_map',
        'nfun_cdf',
        'fieldlist',
        'filterlist',
        'require_detection',
    ],
)


@dataclass(frozen=True)
class DetectionCurve:
    """One detection map: an efficiency read against SNR, |SNR| (``use_abs``)
    or observed magnitude (``is_mag``), for the bands in ``filterlist`` and
    the fields in ``fieldlist`` ('ALL' matches every entry)."""

    filterlist: str
    fieldlist: str
    vals: np.ndarray
    effs: np.ndarray
    use_abs: bool
    is_mag: bool


@dataclass(frozen=True)
class PipelineEffFile:
    """What one SEARCHEFF_PIPELINE_EFF_FILE holds."""

    curves: List[DetectionCurve]
    photprob_maps: List[PhotoprobMap]
    photflag_detect: int
    photflag_trigger: int


def read_pipeline_eff_file(filepath: str, photflag_detect: int,
                           photflag_trigger: int) -> PipelineEffFile:
    """Read the file once. ``photflag_detect``/``photflag_trigger`` are the
    input file's own values, which the file fills in only while they are 0."""
    expanded = Path(os.path.expandvars(filepath))
    check_file_docana(expanded)

    curves: List[DetectionCurve] = []
    photprob_maps: List[PhotoprobMap] = []
    detect_map = None
    named_detect = False    # inside MAPNAME_DETECT ... ENDMAP
    photprob = None
    photprob_rows: List[List[float]] = []

    def flush_detect():
        nonlocal detect_map
        if detect_map is not None and detect_map['vals']:
            curves.append(DetectionCurve(
                filterlist=detect_map['filterlist'],
                fieldlist=detect_map['fieldlist'],
                vals=np.array(detect_map['vals']),
                effs=np.array(detect_map['effs']),
                use_abs=detect_map['use_abs'],
                is_mag=detect_map['is_mag']))
        detect_map = None

    with open_text(expanded) as f:
        for raw_line in skip_docana(f):
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            key = parts[0]

            if key == 'PHOTFLAG_DETECT:' and not photflag_detect:
                photflag_detect = int(parts[1])
                continue
            if key == 'PHOTFLAG_TRIGGER:' and not photflag_trigger:
                photflag_trigger = int(parts[1])
                continue
            if key in ('PHOTFLAG_DETECT:', 'PHOTFLAG_TRIGGER:'):
                continue

            if key == 'MAPNAME_PHOTPROB:':
                flush_detect()
                photprob = {
                    'name': parts[1] if len(parts) > 1 else 'UNNAMED',
                    'fieldlist': 'ALL', 'filterlist': 'ALL', 'nvar_tot': 0,
                    'require_detection': False, 'cutval': -999.0,
                    'reduced_corr': 0.0, 'varnames': [], 'ivarabs': [],
                    'nvar_map': 0, 'nfun_cdf': 0,
                }
                photprob_rows = []
                continue

            if key == 'ENDMAP:':
                if photprob is not None:
                    photprob_maps.append(build_photprob_map(photprob,
                                                            photprob_rows))
                    photprob, photprob_rows = None, []
                else:
                    flush_detect()
                    named_detect = False
                continue

            if photprob is not None:
                _read_photprob_line(photprob, photprob_rows, key, parts)
                continue

            if key == 'MAPNAME_DETECT:':
                flush_detect()
                detect_map = _new_detect_map()
                named_detect = True
                continue
            if key in ('FILTER:', 'BAND:'):
                if not (named_detect and detect_map is not None):
                    flush_detect()
                    detect_map = _new_detect_map()
                detect_map['filterlist'] = parts[1]
                continue
            if key == 'FIELD:' and detect_map is not None:
                detect_map['fieldlist'] = parts[1]
                continue
            if detect_map is not None and len(parts) >= 3:
                if key == 'ABS(SNR):':
                    detect_map['use_abs'] = True
                elif key == 'MAG:':
                    detect_map['is_mag'] = True
                elif key != 'SNR:':
                    continue
                detect_map['vals'].append(float(parts[1]))
                detect_map['effs'].append(float(parts[2]))
    flush_detect()

    return PipelineEffFile(curves=curves, photprob_maps=photprob_maps,
                           photflag_detect=photflag_detect,
                           photflag_trigger=photflag_trigger)


def _new_detect_map() -> dict:
    return {'filterlist': '', 'fieldlist': 'ALL', 'vals': [], 'effs': [],
            'use_abs': False, 'is_mag': False}


def _read_photprob_line(photprob: dict, rows: List[List[float]], key: str,
                        parts: List[str]) -> None:
    """One line inside a MAPNAME_PHOTPROB block."""
    if key in ('FIELD:', 'FIELDLIST:'):
        photprob['fieldlist'] = parts[1] if len(parts) > 1 else 'ALL'
    elif key in ('FILTER:', 'BAND:'):
        photprob['filterlist'] = parts[1] if len(parts) > 1 else 'ALL'
    elif key == 'NVAR:':
        photprob['nvar_tot'] = int(parts[1])
    elif key == 'REQUIRE_DETECTION:':
        photprob['require_detection'] = int(parts[1]) != 0
    elif key == 'CUTVAL:':
        photprob['cutval'] = float(parts[1])
    elif key == 'REDUCED_CORR:':
        photprob['reduced_corr'] = float(parts[1])
    elif key == 'VARNAMES:':
        varnames = []
        for token in parts[1:]:
            if token == 'PHOTPROB_WGTLIST':
                break
            if token.upper() not in VARNAME_TO_IVARABS:
                raise SNANAError(
                    f"Invalid PHOTPROB variable '{token}'. "
                    f"Valid: {list(VARNAME_TO_IVARABS.keys())}",
                    function_name='read_pipeline_eff_file')
            varnames.append(token.upper())
        photprob['varnames'] = varnames
        photprob['ivarabs'] = [VARNAME_TO_IVARABS[v] for v in varnames]
        photprob['nvar_map'] = len(varnames)
        photprob['nfun_cdf'] = photprob['nvar_tot'] - photprob['nvar_map']
    elif key == 'ROW:':
        rows.append([float(x) for x in parts[1:]])


def read_pipeline_logic(logic_file, survey_name: str,
                               path_user_input: str) -> tuple[int, str]:
    """(NMJD, logic) from the SEARCHEFF pipeline logic file.

    Mirrors init_SEARCHEFF_LOGIC (sntools_trigger.c:877-923): 'DEFAULT'
    names SEARCHEFF_PIPELINE_LOGIC.DAT, and the file is found as the other
    SEARCHEFF files are (`resolve_searcheff_file`). The key is required: an
    unset key is SNANA's NONE, which names no file (snlc_sim.c:1301). The
    survey match is the exact token "<SURVEY>:" over the file's token
    stream with the last match winning; an absent survey entry is fatal.
    """
    if logic_file is None or is_ignorefile(logic_file):
        raise SNANAError(
            "SEARCHEFF_PIPELINE_EFF_FILE holds detection curves, which need "
            "a SEARCHEFF_PIPELINE_LOGIC_FILE (or DEFAULT)",
            function_name="init_SEARCHEFF_LOGIC")
    name = ('SEARCHEFF_PIPELINE_LOGIC.DAT' if logic_file == 'DEFAULT'
            else logic_file)
    logic_path = resolve_searcheff_file(
        name, 'SEARCHEFF_PIPELINE_LOGIC_FILE', path_user_input)
    # DOCANA required whatever REQUIRE_DOCANA says (snlc_sim.c:1297)
    check_file_docana(logic_path)

    survey_key = f'{survey_name}:'
    with open_text(logic_path) as fp:
        tokens = fp.read().split()
    n_mjd, logic_string = 0, ''
    for i, token in enumerate(tokens[:-2]):
        if token == survey_key:
            try:
                n_mjd = int(tokens[i + 1])
            except ValueError:
                raise SNANAError(
                    f"Malformed NMJD '{tokens[i + 1]}' for survey "
                    f"'{survey_name}' in {logic_path}",
                    function_name="read_pipeline_logic")
            logic_string = tokens[i + 2]
    if n_mjd == 0:
        raise SNANAError(
            f"Could NOT find search pipeline logic for {survey_name}; "
            f"check {logic_path}",
            function_name="read_pipeline_logic")
    log.line(logger, f"SEARCHEFF logic: {n_mjd} MJDs require filters="
             f"{logic_string}")
    return n_mjd, logic_string


def parse_trigger_logic(logic_string: str,
                        filter_name_to_index: dict[str, int]) -> list[int]:
    """Convert logic string (e.g., 'gr+ri+gi') into list of per-filter bitmasks."""
    tokens = [token for token in ''.join(logic_string.split()).split('+') if token]
    if not tokens:
        raise SNANAError(f"Invalid SEARCHEFF logic string: {logic_string}")

    # unknown filters get an unreachable bit (INTFILTER, sntools_trigger.c:1018)
    ungenerated: dict[str, int] = {}
    n_generated = len(filter_name_to_index)
    masks: list[int] = []
    for token in tokens:
        mask = 0
        for symbol in token:
            if symbol in filter_name_to_index:
                idx = filter_name_to_index[symbol]
            else:
                idx = ungenerated.setdefault(symbol, n_generated + len(ungenerated))
            mask |= (1 << idx)
        masks.append(mask)
    return masks


def load_pipeline_curves(config, simlib, filter_name_to_index) -> dict:
    """The SEARCHEFF pipeline state: the detection curves and PHOTPROB maps
    the eff file holds, the PHOTFLAG bits, and the trigger logic.

    APPLY_SEARCHEFF_OPT gates event rejection, not evaluation: SNANA loads the
    maps and detects regardless (init_SEARCHEFF, sntools_trigger.c:114).
    """
    sc = config.survey
    # no file, or DEFAULT: every event pipeline-found (sntools_trigger.c:258-310,
    # :2037-2044)
    if (sc.searcheff_pipeline_eff_file == 'DEFAULT'
            or is_ignorefile(sc.searcheff_pipeline_eff_file)):
        return {
            'curves': [], 'photprob_maps': [],
            'photflag_detect': sc.searcheff_photflag_detect,
            'photflag_trigger': sc.searcheff_photflag_trigger,
            'n_trigger_detections': 0,
            'trigger_logic': '',
            'filter_masks': [],
        }

    pipeline_eff_file = resolve_searcheff_file(
        sc.searcheff_pipeline_eff_file, 'SEARCHEFF_PIPELINE_EFF_FILE',
        config.run.path_user_input)
    eff = read_pipeline_eff_file(pipeline_eff_file,
                                 sc.searcheff_photflag_detect,
                                 sc.searcheff_photflag_trigger)
    # the logic is read only for a file with curves (sntools_trigger.c:115)
    n_trigger_detections, trigger_logic = (read_pipeline_logic(
        sc.searcheff_pipeline_logic_file, simlib.survey_name,
        config.run.path_user_input) if eff.curves else (0, ''))
    log.line(logger, f"Loaded SEARCHEFF from {pipeline_eff_file.name} "
             f"for filters: {[c.filterlist for c in eff.curves]} "
             f"(PHOTFLAG_DETECT={eff.photflag_detect}, "
             f"PHOTFLAG_TRIGGER={eff.photflag_trigger})")

    if eff.photprob_maps:
        resolve_fieldlist_indices(
            eff.photprob_maps, [f.field_name for f in simlib.fields])
        log.line(logger, f"Loaded {len(eff.photprob_maps)} PHOTPROB maps")

    return {
        'curves': eff.curves, 'photprob_maps': eff.photprob_maps,
        'photflag_detect': eff.photflag_detect,
        'photflag_trigger': eff.photflag_trigger,
        'n_trigger_detections': n_trigger_detections,
        'trigger_logic': trigger_logic,
        'filter_masks': (parse_trigger_logic(trigger_logic, filter_name_to_index)
                         if eff.curves else []),
    }


def build_searcheff_config(searcheff_curves, filter_logic_masks,
                           n_trigger_detections, newmjd_dif, filter_names,
                           simlib_field_names, zp_fluxcal,
                           apply_detect_single):
    """Convert SEARCHEFF detection maps to JAX-compatible config arrays.

    ``searcheff_curves`` is the file's DetectionCurve records: `fieldlist`
    is 'ALL' or a FIELDLIST string matched by substring like SNANA's
    MATCH_SEARCHEFF_FIELD (sntools_trigger.c:3266-3307), and `vals` is SNR,
    ``|SNR|`` under use_abs, or observed magnitude under is_mag (SNR/ABS(SNR)/
    MAG rows, sntools_trigger.c:412-423). ``filter_logic_masks`` is the list
    of int bitmasks for trigger logic; ``n_trigger_detections`` the number
    of detection periods required; ``newmjd_dif`` the MJD threshold for
    period separation. ``filter_names`` is the list of full filter names
    (e.g., ['LSST-u', ...]); ``simlib_field_names`` the FIELD name per
    SIMLIB entry, which the per-SN field_idx indexes into the (entry,
    filter) map-index table. ``apply_detect_single``, if True, applies
    curves per exposure, combining across NEXPOSE (sntools_trigger.c:2506,
    2527-2531).

    Raises `SNANAError` when more than one map matches an (entry, filter)
    observation (GETEFF_PIPELINE_DETECT, sntools_trigger.c:2449-2456).
    """
    n_filters = len(filter_names)
    n_maps = len(searcheff_curves)
    filter_chars = [filter_band(name) for name in filter_names]

    max_grid = max(len(m.vals) for m in searcheff_curves)
    # padding repeats the last eff, so jnp.interp gives the endpoint EFFs
    # (sntools_trigger.c:2518-2519)
    val_grids = np.empty((n_maps, max_grid), dtype=np.float32)
    eff_grids = np.empty((n_maps, max_grid), dtype=np.float32)
    use_abs = np.zeros(n_maps, dtype=bool)
    is_mag = np.zeros(n_maps, dtype=bool)
    val_lo = np.zeros(n_maps, dtype=np.float32)
    val_hi = np.zeros(n_maps, dtype=np.float32)
    for i, m in enumerate(searcheff_curves):
        vals = np.asarray(m.vals, dtype=np.float32)
        effs = np.asarray(m.effs, dtype=np.float32)
        n = len(vals)
        val_grids[i, :n] = vals
        eff_grids[i, :n] = effs
        n_pad = max_grid - n
        if n_pad:
            val_grids[i, n:] = vals[-1] + np.arange(1, n_pad + 1)
            eff_grids[i, n:] = effs[-1]
        use_abs[i] = m.use_abs
        is_mag[i] = m.is_mag
        val_lo[i] = vals[0]
        val_hi[i] = vals[-1]

    # FIELD matches as a substring of FIELDLIST; -1 = no curve, EFF=0
    # (MATCH_SEARCHEFF_FIELD, sntools_trigger.c:2428-2446, :3266-3307)
    map_index = np.full((len(simlib_field_names), n_filters), -1, dtype=np.int32)
    for e, field_name in enumerate(simlib_field_names):
        for f, fchar in enumerate(filter_chars):
            matches = [i for i, m in enumerate(searcheff_curves)
                       if fchar in m.filterlist
                       and (m.fieldlist == 'ALL'
                            or field_name in m.fieldlist)]
            if len(matches) > 1:
                raise SNANAError(
                    f"Found {len(matches)} PIPELINE/DETECT maps for filter '{fchar}', "
                    f"FIELD '{field_name}'.",
                    function_name='build_searcheff_config')
            if matches:
                map_index[e, f] = matches[0]

    return {
        'zp_fluxcal': zp_fluxcal,
        'val_grids': jnp.array(val_grids),
        'eff_grids': jnp.array(eff_grids),
        'use_abs': jnp.array(use_abs),
        'is_mag': jnp.array(is_mag),
        'val_lo': jnp.array(val_lo),
        'val_hi': jnp.array(val_hi),
        'map_index': jnp.array(map_index),
        'filter_logic_masks': jnp.array(filter_logic_masks, dtype=jnp.int32),
        'n_trigger_detections': n_trigger_detections,
        'newmjd_dif': newmjd_dif,
        'n_filters': n_filters,
        'apply_detect_single': apply_detect_single,
    }


def _detect_one_sn(snrs_model, fluxcals_true, nexposes, filter_indices, valid,
                   field_idx, rng_key, config):
    """Per-obs SEARCHEFF detection for one SN (GETEFF_PIPELINE_DETECT,
    sntools_trigger.c:2387-2540).

    The map serving each obs comes from config['map_index'][field_idx,
    filter]; an obs with no matching map gets EFF=0 (other maps exist,
    else this kernel is never built). Curve type selects the x-axis:
      - is_mag=False, use_abs=False: SNR, with SNR<=0 -> EFF=0 (:2492)
      - is_mag=False, use_abs=True:  |SNR|
      - is_mag=True: TRUE observed magnitude, from the noise-free obs mag
        (genmag_obs, fed into SEARCHEFF at snlc_sim.c:24721); non-positive
        true flux is the model's null magnitude, 99.

    Out-of-range values return the curve's endpoint efficiencies
    (VAL>VAL_atmax -> EFF_atmax, VAL<VAL_atmin -> EFF_atmin;
    sntools_trigger.c:2518-2519).

    When config['apply_detect_single'] is True, SNR is divided by
    sqrt(NEXPOSE) before evaluation (MAG is not shifted; :2506) and the
    interpolated in-range eff combines as 1-(1-eff_1)^NEXPOSE for
    NEXPOSE>1 (:2527-2534); the endpoint returns skip the combine, as
    SNANA returns them before it.
    """
    apply_single = config['apply_detect_single']
    m = config['map_index'][field_idx, filter_indices]
    has_map = m >= 0
    safe_m = jnp.maximum(m, 0)
    val_grid = config['val_grids'][safe_m]
    eff_grid = config['eff_grids'][safe_m]
    use_abs_o = config['use_abs'][safe_m]
    is_mag_o = config['is_mag'][safe_m]
    val_lo = config['val_lo'][safe_m]
    val_hi = config['val_hi'][safe_m]

    pos = fluxcals_true > 0
    obs_mag = jnp.where(
        pos,
        config['zp_fluxcal'] - 2.5 * jnp.log10(jnp.where(pos, fluxcals_true, 1.0)),
        MAG_ZEROFLUX,
    )

    nexpose_f = jnp.maximum(nexposes.astype(jnp.float32), 1.0)
    snr = snrs_model / jnp.where(apply_single, jnp.sqrt(nexpose_f), 1.0)
    val = jnp.where(is_mag_o, obs_mag,
                    jnp.where(use_abs_o, jnp.abs(snr), snr))

    eff_interp = jax.vmap(jnp.interp)(val, val_grid, eff_grid)
    in_range = (val >= val_lo) & (val <= val_hi)
    # per-exposure curves combine over NEXPOSE, not at the endpoints
    # (sntools_trigger.c:2527-2534)
    eff_combined = 1.0 - jnp.power(1.0 - eff_interp, nexpose_f)
    eff = jnp.where(apply_single & in_range & (nexpose_f > 1.0),
                    eff_combined, eff_interp)

    # raw SNR<=0 -> 0 for SNR curves only (sntools_trigger.c:2492)
    zero_snr = ~is_mag_o & ~use_abs_o & (snrs_model <= 0.0)
    eff = jnp.where(zero_snr | ~has_map | ~valid, 0.0, eff)
    random_vals = jax.random.uniform(rng_key, snrs_model.shape)
    return valid & (random_vals < eff)


def _periods_one_sn(mjds, valid, threshold):
    """Detection-period IDs for one SN, matching SNANA
    gen_SEARCHEFF_PIPELINE's fabs(MJD - MJD_LAST) > TDIF_NEXT rule
    (sntools_trigger.c:2150-2179).

    SNANA advances MJD_LAST only when a period starts (:2175), so the
    comparison is against the period's first MJD, not the previous obs.

    Iterative refinement: seed from consecutive gaps, then split each period
    at its first obs exceeding the span, until nothing moves.
    """
    max_obs = mjds.shape[0]
    obs_indices = jnp.arange(max_obs, dtype=jnp.int32)
    consec_gaps = jnp.concatenate(
        [jnp.array([jnp.inf], dtype=mjds.dtype), mjds[1:] - mjds[:-1]])
    period_start = valid & ((consec_gaps > threshold) | (obs_indices == 0))

    def refine(period_start):
        period_ids = jnp.cumsum(period_start.astype(jnp.int32)) - 1
        safe_indices = jnp.where(valid, obs_indices, max_obs)
        period_start_idx = jnp.full(max_obs, max_obs, dtype=jnp.int32)
        period_start_idx = period_start_idx.at[period_ids].min(safe_indices)
        period_start_mjd = mjds[period_start_idx[period_ids]]
        gap = jnp.abs(mjds - period_start_mjd)
        exceeds = valid & (gap > threshold) & ~period_start
        exceed_idx = jnp.where(exceeds, obs_indices, max_obs)
        first_exceed = jnp.full(max_obs, max_obs, dtype=jnp.int32)
        first_exceed = first_exceed.at[period_ids].min(exceed_idx)
        is_first = exceeds & (obs_indices == first_exceed[period_ids])
        return period_start | is_first

    def _body(state):
        starts, _, i = state
        nxt = refine(starts)
        return nxt, jnp.any(nxt != starts), i + 1

    period_start, _, _ = jax.lax.while_loop(
        lambda s: s[1] & (s[2] < max_obs),
        _body,
        (period_start, jnp.array(True), jnp.array(0)))
    period_ids = jnp.cumsum(period_start.astype(jnp.int32)) - 1
    return jnp.where(valid, period_ids, -1)


def _trigger_one_sn(detected, filter_indices, period_ids, valid, mjds,
                    required_masks, n_trigger, n_filters, max_obs):
    """Evaluate trigger logic + MJD_DETECT diagnostics for one SN.

    Returns ``(triggered, n_det_periods, mjd_detect_first, mjd_detect_last,
    mjd_trigger, trigger_mask)``: ``triggered`` is whether the SN passes the
    filter-combination + n_trigger test; ``n_det_periods`` the number of
    detection periods that passed logic; ``mjd_detect_first``/
    ``mjd_detect_last`` the earliest/latest detected-obs MJD (1e6 if none,
    as SNANA initialises it, sntools_trigger.c:2027); ``mjd_trigger`` the
    first detected-obs MJD inside the n_trigger-th passing period (1e6 when
    not triggered); ``trigger_mask`` a (max_obs,) bool array, True at the
    single obs where the trigger fires.
    """
    period_has_filter = jnp.zeros((max_obs, n_filters), dtype=bool)
    for f in range(n_filters):
        filter_det = detected & (filter_indices == f) & valid
        count = jnp.zeros(max_obs, dtype=jnp.float32)
        count = count.at[period_ids].add(
            jnp.where(filter_det & (period_ids >= 0), 1.0, 0.0))
        period_has_filter = period_has_filter.at[:, f].set(count > 0)

    period_bitmask = jnp.zeros(max_obs, dtype=jnp.int32)
    for f in range(n_filters):
        period_bitmask = period_bitmask + jnp.where(
            period_has_filter[:, f], 1 << f, 0)

    period_passes = jnp.any(
        (period_bitmask[:, None] & required_masks[None, :]) == required_masks[None, :],
        axis=1)

    period_exists = jnp.zeros(max_obs, dtype=jnp.float32)
    period_exists = period_exists.at[period_ids].add(
        jnp.where(valid & (period_ids >= 0), 1.0, 0.0))
    period_exists = period_exists > 0

    passing_and_exists = period_passes & period_exists
    n_passing = jnp.sum(passing_and_exists)
    triggered = n_passing >= n_trigger

    # MJD_DETECT_* stand whether or not the SN triggers
    masked_min = jnp.where(detected, mjds, jnp.inf)
    masked_max = jnp.where(detected, mjds, -jnp.inf)
    any_det = detected.any()
    mjd_detect_first = jnp.where(any_det, masked_min.min(), 1.0e6)
    mjd_detect_last = jnp.where(any_det, masked_max.max(), 1.0e6)

    # the trigger fires at the first detection of the n_trigger-th passing period
    passing_cumcount = jnp.cumsum(passing_and_exists.astype(jnp.int32))
    is_trigger_period = passing_and_exists & (passing_cumcount == n_trigger)
    safe_pid = jnp.where(valid & (period_ids >= 0), period_ids, 0)
    obs_in_trigger_period = is_trigger_period[safe_pid] & valid & (period_ids >= 0)
    trig_obs_mjds = jnp.where(obs_in_trigger_period & detected, mjds, jnp.inf)
    has_trigger = triggered & jnp.any(trig_obs_mjds < jnp.inf)
    mjd_trigger = jnp.where(has_trigger, trig_obs_mjds.min(), 1.0e6)
    trigger_obs_idx = jnp.argmin(trig_obs_mjds)
    trigger_mask = jnp.zeros(max_obs, dtype=bool)
    trigger_mask = trigger_mask.at[trigger_obs_idx].set(has_trigger)

    return (triggered, n_passing, mjd_detect_first, mjd_detect_last,
            mjd_trigger, trigger_mask)


@jax.jit
def detect_all(config, snrs_model, fluxcals_true, nexposes, filter_indices,
               valid, field_idx, rng_keys):
    """Detect epochs using the supplied efficiency tables."""
    return jax.vmap(_detect_one_sn, in_axes=(0, 0, 0, 0, 0, 0, 0, None))(
        snrs_model, fluxcals_true, nexposes, filter_indices, valid,
        field_idx, rng_keys, config)


@jax.jit(static_argnames=('n_filters',))
def trigger_all(config, detected, filter_indices, valid, mjds, *, n_filters):
    """Evaluate detection periods and trigger logic for a batch."""
    def trigger_one(detected, filter_indices, valid, mjds):
        period_ids = _periods_one_sn(mjds, valid, config['newmjd_dif'])
        return _trigger_one_sn(
            detected, filter_indices, period_ids, valid, mjds,
            config['filter_logic_masks'], config['n_trigger_detections'],
            n_filters, mjds.shape[0])

    return jax.vmap(trigger_one)(detected, filter_indices, valid, mjds)
