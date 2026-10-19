"""The SALT2/SALT3 SED model, mirroring SNANA's genmag_SALT2.c
(INTEG_zSED_SALT2 for the interpolation it reproduces).

References:
    - Guy et al. 2007 (SALT2 original): https://ui.adsabs.harvard.edu/abs/2007A%26A...466...11G
    - Guy et al. 2010 (JLA-B14): https://ui.adsabs.harvard.edu/abs/2010A%26A...523A...7G
    - Betoule et al. 2014 (JLA cosmology): https://ui.adsabs.harvard.edu/abs/2014A%26A...568A..22B
    - Kenworthy et al. 2021 (SALT3): https://ui.adsabs.harvard.edu/abs/2021ApJ...923..265K
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Dict, Literal, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax, vmap

from .base import (BBC_ONLY, FIT_ONLY, SNModel,
                   resample_surfaces, uvlam_extrapflux_sedmodel)
from .fitted import FittedModel, flux_error_from_fracerr
from .simulated import SimulatedModel
from .standardised import StandardisedModel
from jax_snlc_core.io.docana import skip_docana, check_file_docana
from jax_snlc_core.io.fits_path import open_text, resolve_fits_path
from jax_snlc_core.io.sed_ascii import SED_WAVELENGTH_RANGE, read_sed_file
from jax_snlc_core.physics.constants import LN10
from jax_snlc_core.physics.mass_step import gammadm
from jax_snlc_core.photometry.filter_system import band_window_starts
from jax_snlc_core.utils import SNANA_NULLFLOAT
from jax_snlc_core.utils import snana_log as slog
from jax_snlc_core.utils.extrap import modelflux_extrap, LatetimeExtrapIa
from jax_snlc_core.utils.error_handling import SNANAError
from jax_snlc_core.utils.interp import (
    compute_catmullrom_slopes,
    hermite_grid_slopes,
    hermite_interp_1d,
    hermite_interp_1d_batch,
    hermite_interp_grid,
    upsample_grid_linear,
    hermite_interp_1d_select,
    hermite_weights,
)

logger = logging.getLogger(__name__)

# SALT2 flux normalisation (genmag_SALT2.h:6)
X0SCALE_SALT2 = 1.0e-12


def compute_expected_x0_scale(mu: jnp.ndarray, alpha: float, beta: float,
                              x1: float = 0.0, c: float = 0.0) -> jnp.ndarray:
    """Expected SALT x0 from the distance modulus, matching SNANA SALT2x0calc.

    SNANA formula (genmag_SALT2.c:2968):
        arg = 0.4 * (mu - alpha*x1 + beta*c)
        x0 = 1 / (X0SCALE_SALT2 * 10^arg)

    where X0SCALE_SALT2 = 1e-12 (arbitrary normalisation constant). ``mu``
    arrives already computed so no cosmology is embedded here.
    """
    arg = 0.4 * (mu - alpha * x1 + beta * c)
    return 1.0 / (X0SCALE_SALT2 * jnp.power(10.0, arg))


# OPT_SALT2FIT=1 log10(x0) bounds, x0 at mu=70 and 20 (snlc_fit.F90:8095-8103)
LOGX0_BOUNDS = (float(np.log10(compute_expected_x0_scale(70.0, 0.13, 3.20))),
                float(np.log10(compute_expected_x0_scale(20.0, 0.13, 3.20))))


# SEDFLUX_INTERP_OPT=2 rebin factors (INTERP_SEDREBIN_DAY/LAM)
NREBIN_DAY_DEFAULT = 5
NREBIN_LAM_DEFAULT = 2

# mB = SALT2_MB_OFFSET - 2.5*log10(x0), hard-wired to the SNLS VEGA system
# (load_mBoff_SALT2, genmag_SALT2.c:3013).
SALT2_MB_OFFSET = 10.635


def detect_salt_version(model_path: str) -> int:
    """Auto-detect SALT version from model directory name ('SALT2.JLA-B14' → 2)."""
    path = Path(model_path)
    dirname = path.name
    match = re.match(r'SALT(\d+)', dirname, re.IGNORECASE)
    if match:
        return int(match.group(1))
    raise SNANAError(
        f"Cannot detect SALT version from directory name: {dirname}. "
        f"Expected format like 'SALT2.JLA-B14' or 'SALT3.K21'",
        function_name="detect_salt_version")


# the phase window every SALT surface and error map is read within
# (genmag_SALT2.c:287-288)
SALT_DAY_RANGE = (-20.0, 200.0)


def _parse_template_file(filepath: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A SALT surface (M0, M1) or error map in 3-column (phase, wave, value)
    form, read as rd_sedFlux reads it inside SNANA's SALT window
    (genmag_SALT2.c:287-290): the same comment lines, row and grid checks,
    uniform day bins (no OPTMASK_DAYLIST_SEDMODEL), and a NaN value read as
    zero. Returns ``(phases, wavelengths, grid)``."""
    return read_sed_file(filepath, SALT_DAY_RANGE, SED_WAVELENGTH_RANGE,
                         t0shift_peakmag=False, uniform_days=True)


def _parse_color_dispersion_file(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a 2-column (wavelength, dispersion) file as rd2columnFile does
    (sntools.c:7242-7330): a line of two characters or fewer, a line whose
    first word starts with ``#``, ``!``, ``%`` or ``@``, and a DOCANA block
    are skipped. A data line without two numbers, which rd2columnFile has no
    defined reading of, aborts."""
    wavelengths = []
    dispersions = []
    with open_text(filepath) as f:
        for line in skip_docana(f):
            parts = line.split()
            if len(line) <= 2 or not parts or parts[0][0] in '#!%@':
                continue
            try:
                wavelengths.append(float(parts[0]))
                dispersions.append(float(parts[1]))
            except (IndexError, ValueError):
                raise SNANAError(f"{filepath}: '{line.rstrip()}' is not two "
                                 f"numbers", function_name="rd2columnFile"
                                 ) from None
    return np.array(wavelengths), np.array(dispersions)


def _parse_info_file(filepath: Path) -> Dict:
    """Parse SALT{2,3}.INFO file into a flat parameter dict.

    Defaults follow init_SALT2infoDefaults (genmag_SALT2.c:1180-1216).
    """
    params = {
        'colorlaw_version': 0,
        'mag_offset': 0.0,
        'sedflux_interp_opt': 2,
        'errmap_interp_opt': 2,
        'magerr_floor': 0.005,
        'magerr_lamobs': (0.0, 0.0, 0.0),
        'magerr_lamrest': (0.0, 0.0, 0.0),
        'color_disp_max': 2.0,
        'color_offset': 0.0,
        'restlam_forcezeroflux': (0.0, 0.0),
        'errmap_kcor_opt': 1,
        'errmap_badval_abort': 1,
    }
    with open(filepath, 'r') as f:
        for line in skip_docana(f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' not in line:
                continue
            parts = line.split(':', 1)
            key = parts[0].strip()
            value = parts[1].strip()
            if '#' in value:
                value = value.split('#')[0].strip()

            if key == 'RESTLAMBDA_RANGE':
                lam_min, lam_max = value.split()
                params['restlambda_min'] = float(lam_min)
                params['restlambda_max'] = float(lam_max)
            elif key == 'COLORLAW_VERSION':
                params['colorlaw_version'] = int(value)
            elif key in ('COLORLAW_PARAMS', 'COLORCOR_PARAMS'):
                # either spelling (genmag_SALT2.c:1253); law 0 stores COR0, COR1 in the lam slots (:3779)
                fields = value.split()
                params['colorlaw_lam_min'] = float(fields[0])
                params['colorlaw_lam_max'] = float(fields[1])
                if len(fields) == 2:
                    params['colorlaw_coeffs'] = np.array(
                        [float(fields[0]), float(fields[1])])
                else:
                    params['colorlaw_coeffs'] = np.array(
                        [float(x) for x in fields[3:3 + int(fields[2])]])
            elif key == 'COLOR_OFFSET':
                params['color_offset'] = float(value)
            elif key == 'MAG_OFFSET':
                params['mag_offset'] = float(value)
            elif key == 'SEDFLUX_INTERP_OPT':
                params['sedflux_interp_opt'] = int(value)
            elif key == 'ERRMAP_INTERP_OPT':
                params['errmap_interp_opt'] = int(value)
            elif key == 'MAGERR_FLOOR':
                params['magerr_floor'] = float(value)
            elif key == 'MAGERR_LAMOBS':
                params['magerr_lamobs'] = tuple(float(x) for x in value.split()[:3])
            elif key == 'MAGERR_LAMREST':
                params['magerr_lamrest'] = tuple(float(x) for x in value.split()[:3])
            elif key == 'COLOR_DISP_MAX':
                params['color_disp_max'] = float(value)
            elif key == 'RESTLAM_FORCEZEROFLUX':
                params['restlam_forcezeroflux'] = tuple(
                    float(x) for x in value.split()[:2])
            elif key == 'ERRMAP_KCOR_OPT':
                params['errmap_kcor_opt'] = int(value.split()[0])
            elif key == 'ERRMAP_BADVAL_ABORT':
                params['errmap_badval_abort'] = int(value.split()[0])
    return params


# error-map upsampling per axis
ERRMAP_UPSAMPLE = 4

# The range outside which an error-map value is crazy, per map type
# (init_BADVAL_SALT2errmap, genmag_SALT2.c:1561-1588)
ERRMAP_VALID_VARIANCE = (-10.0, 500.0)
ERRMAP_VALID_COVARIANCE = (-30.0, 100.0)
ERRMAP_VALID_ERRSCALE = (0.0, 200.0)
ERRMAP_VALID_COLORDISP = (0.0, 5.0)
# a phase-dependent map is judged only this far inside the variance map's
# phase range, whose edges really are crazy (genmag_SALT2.c:1617-1619)
ERRMAP_DAYEDGE_TOLERANCE = 9.0


def _check_errmap_values(maps, abort: bool) -> None:
    """SNANA's bad-value check of the error maps (check_BADVAL_SALT2errmap,
    genmag_SALT2.c:1602-1760). ``maps`` holds ``(label, file, phases,
    values, valid_range)`` per map, ``phases`` None for the colour
    dispersion, the first map being the variance map whose phase range
    sets the window. A NaN anywhere is bad; a value outside its map's valid
    range is crazy when its phase lies more than ERRMAP_DAYEDGE_TOLERANCE
    days inside that window, or always for a map with one phase or none
    (the colour dispersion; genmag_SALT2.c:1733-1735). With
    ``abort`` (ERRMAP_BADVAL_ABORT, on by default) any bad value aborts;
    without it the counts are a warning."""
    day_lo = maps[0][2][0] + ERRMAP_DAYEDGE_TOLERANCE - 0.001
    day_hi = maps[0][2][-1] - ERRMAP_DAYEDGE_TOLERANCE + 0.001
    lines = []
    for label, name, phases, values, (lo, hi) in maps:
        values = np.asarray(values, dtype=np.float64)
        nan = np.isnan(values)
        judged = (np.ones(values.shape, dtype=bool)
                  if phases is None or len(phases) == 1 else
                  ((np.asarray(phases) > day_lo)
                   & (np.asarray(phases) < day_hi))[:, None])
        crazy = judged & ~nan & ((values < lo) | (values > hi))
        if nan.any() or crazy.any():
            lines.append(f"{label} ({name}): {int(nan.sum())} NaN, "
                         f"{int(crazy.sum())} outside [{lo:g}, {hi:g}]")
    if not lines:
        return
    message = "bad ERRMAP values (NaN and/or crazy): " + "; ".join(lines)
    if abort:
        raise SNANAError(message, function_name="check_BADVAL_SALT2errmap")
    slog.warn(logger, 'SALT', message)


def _load_salt_error_model(model_dir: Path, prefix: str, version: int,
                           color_disp_max: float, kcor_opt: int,
                           badval_abort: int) -> Dict:
    """Load SALT error model components (variance, covariance, colour dispersion).

    SALT2 stores relative variance (``salt2_lc_relative_variance_*``) plus the
    error-scale map (``salt2_lc_dispersion_scaling``); SALT3 stores absolute
    variance (``salt3_lc_variance_*``) and has no scale map. Every map is
    required, as SNANA requires it, save the colour dispersion under
    ``kcor_opt`` 0 (ERRMAP_KCOR_OPT). The maps then pass SNANA's bad-value
    check, which aborts under ``badval_abort`` (ERRMAP_BADVAL_ABORT).
    """
    error_model = {
        'variance_0': None,
        'variance_1': None,
        'covariance_01': None,
        'color_dispersion': None,
        'color_dispersion_wavelengths': None,
        'color_dispersion_slopes': None,
        'error_phases': None,
        'error_wavelengths': None,
        'errscale': None,
        'errscale_phases': None,
        'errscale_wavelengths': None,
        'variance_0_slopes': None,
        'variance_1_slopes': None,
        'covariance_01_slopes': None,
        'errscale_slopes': None,
    }

    if version == 2:
        var0_name = f"{prefix}_lc_relative_variance_0.dat"
        var1_name = f"{prefix}_lc_relative_variance_1.dat"
        cov01_name = f"{prefix}_lc_relative_covariance_01.dat"
    else:
        var0_name = f"{prefix}_lc_variance_0.dat"
        var1_name = f"{prefix}_lc_variance_1.dat"
        cov01_name = f"{prefix}_lc_covariance_01.dat"
    color_disp_name = f"{prefix}_color_dispersion.dat"

    def existing(name):
        path = Path(resolve_fits_path(model_dir / name))
        if not path.exists():
            raise SNANAError(f"Missing SALT error map {name} in {model_dir}",
                             function_name="_load_salt_error_model")
        return path

    # (label, file, phases or None, values, valid range) of every map, for
    # the bad-value check
    checked = []
    wave_ranges = []
    phases, wavelengths, var0 = _parse_template_file(existing(var0_name))
    wave_ranges.append((var0_name, (wavelengths[0], wavelengths[-1])))
    error_model['variance_0'] = jnp.array(var0)
    error_model['error_phases'] = jnp.array(phases)
    error_model['error_wavelengths'] = jnp.array(wavelengths)
    checked.append(('VAR0', var0_name, phases, var0, ERRMAP_VALID_VARIANCE))

    var1_phases, var1_waves, var1 = _parse_template_file(existing(var1_name))
    wave_ranges.append((var1_name, (var1_waves[0], var1_waves[-1])))
    error_model['variance_1'] = jnp.array(var1)
    checked.append(('VAR1', var1_name, var1_phases, var1,
                    ERRMAP_VALID_VARIANCE))

    cov01_phases, cov01_waves, cov01 = _parse_template_file(existing(cov01_name))
    wave_ranges.append((cov01_name, (cov01_waves[0], cov01_waves[-1])))
    error_model['covariance_01'] = jnp.array(cov01)
    checked.append(('COVAR01', cov01_name, cov01_phases, cov01,
                    ERRMAP_VALID_COVARIANCE))

    if version == 2:
        scale_name = f"{prefix}_lc_dispersion_scaling.dat"
        phases_s, waves_s, scale = _parse_template_file(existing(scale_name))
        wave_ranges.append((scale_name, (waves_s[0], waves_s[-1])))
        error_model['errscale'] = jnp.array(scale)
        error_model['errscale_phases'] = jnp.array(phases_s)
        error_model['errscale_wavelengths'] = jnp.array(waves_s)
        checked.append(('ERRSCALE', scale_name, phases_s, scale,
                        ERRMAP_VALID_ERRSCALE))

    # ERRMAP_KCOR_OPT 0 leaves the colour dispersion unread, and so out of
    # every error (read_SALT2colorDisp, genmag_SALT2.c:1047-1050)
    if kcor_opt:
        wavelengths, dispersion = _parse_color_dispersion_file(
            existing(color_disp_name))
        if not np.any(dispersion > 1.0e-12):
            raise SNANAError(
                f"All colorDisp elements are zero in {color_disp_name}; "
                f"check training options", function_name="read_SALT2colorDisp")
        wave_ranges.append((color_disp_name, (wavelengths[0], wavelengths[-1])))
        # COLOR_DISP_MAX clamp (genmag_SALT2.c:1075)
        dispersion = np.minimum(dispersion, color_disp_max)
        error_model['color_dispersion'] = jnp.array(dispersion)
        error_model['color_dispersion_wavelengths'] = jnp.array(wavelengths)
        error_model['color_dispersion_slopes'] = compute_catmullrom_slopes(
            wavelengths, dispersion)
        checked.append(('COLOR-DISP', color_disp_name, None, dispersion,
                        ERRMAP_VALID_COLORDISP))

    error_model['wave_ranges'] = wave_ranges
    _check_errmap_values(checked, badval_abort)
    error_model['phase_ranges'] = [
        (name, (phases[0], phases[-1]))
        for _, name, phases, _, _ in checked if phases is not None]

    # linear upsample then Hermite: SNANA's linear read (genmag_SALT2.c:3136), C1 at the cell edges
    for keys, phases_key, waves_key in (
            (('variance_0', 'variance_1', 'covariance_01'),
             'error_phases', 'error_wavelengths'),
            (('errscale',), 'errscale_phases', 'errscale_wavelengths')):
        phases, waves = error_model[phases_key], error_model[waves_key]
        for key in keys:
            if error_model[key] is None:
                continue
            error_model[phases_key], error_model[waves_key], error_model[key] = (
                upsample_grid_linear(phases, waves, error_model[key], ERRMAP_UPSAMPLE))
            error_model[f'{key}_slopes'] = hermite_grid_slopes(
                error_model[phases_key], error_model[waves_key], error_model[key])

    loaded = [k for k in ('variance_0', 'variance_1', 'covariance_01', 'color_dispersion')
              if error_model[k] is not None]
    slog.line(logger, f"Loaded error model: {', '.join(loaded)}")
    return error_model


def load_salt_data(model_path: str, mskopt: int = 0) -> Dict:
    """Load all SALT model data from a model directory.

    Works for SALT2 and SALT3 by auto-detecting the version and selecting
    filename conventions + error-model style accordingly.
    """
    model_dir = Path(model_path).expanduser()
    if not model_dir.exists():
        raise SNANAError(f"Model directory not found: {model_dir}",
                         function_name="load_salt_data")

    version = detect_salt_version(model_path)
    model_name = model_dir.name
    logger.info(f"Loading SALT{version} model: {model_name}")

    prefix = f"salt{version}"
    template_0_file = Path(resolve_fits_path(model_dir / f"{prefix}_template_0.dat"))
    template_1_file = Path(resolve_fits_path(model_dir / f"{prefix}_template_1.dat"))
    info_file = model_dir / f"SALT{version}.INFO"

    missing_files = [str(f) for f in (template_0_file, template_1_file, info_file)
                     if not f.exists()]
    if missing_files:
        raise SNANAError(
            f"Missing SALT{version} model files:\n" + "\n".join(missing_files),
            function_name="load_salt_data")
    if mskopt & 2:
        check_file_docana(info_file, header_lines=3)
    # SNANA reads a third surface, x2*M2, when template_2 exists
    # (NSURFACE_SALT2, genmag_SALT2.c:445-470); jax models two
    template_2_file = Path(resolve_fits_path(model_dir / f"{prefix}_template_2.dat"))
    if template_2_file.exists():
        raise SNANAError(
            f"{template_2_file.name} makes {model_dir.name} a three-surface "
            f"SALT model, which is not implemented in jax",
            function_name="load_salt_data")

    logger.debug(f"  Loading M0 template from {template_0_file.name}")
    phases_0, waves_0, m0 = _parse_template_file(template_0_file)
    logger.debug(f"  Loading M1 template from {template_1_file.name}")
    phases_1, waves_1, m1 = _parse_template_file(template_1_file)
    if not np.allclose(phases_0, phases_1) or not np.allclose(waves_0, waves_1):
        raise SNANAError("M0 and M1 templates have mismatched phase/wavelength grids",
                         function_name="load_salt_data")
    phases = phases_0
    wavelengths = waves_0

    logger.debug(f"  Loading parameters from {info_file.name}")
    info_params = _parse_info_file(info_file)

    colorlaw_version = info_params['colorlaw_version']
    if colorlaw_version not in (0, 1):
        raise SNANAError(
            f"Invalid COLORLAW_VERSION={colorlaw_version}",
            function_name="load_salt_data")
    # the colour law is the INFO file's, as SNANA reads it (genmag_SALT2.c:1253)
    if 'colorlaw_coeffs' not in info_params:
        raise SNANAError(f"{info_file.name} has no COLORLAW_PARAMS line",
                         function_name="load_salt_data")
    colorlaw_coeffs = info_params['colorlaw_coeffs']
    n_coeffs = len(colorlaw_coeffs)
    colorlaw_lam_min = info_params['colorlaw_lam_min']
    colorlaw_lam_max = info_params['colorlaw_lam_max']

    slog.line(logger, f"Model loaded: {len(phases)} phases, {len(wavelengths)} wavelengths")
    slog.line(logger, f"Phase range: [{phases.min():.1f}, {phases.max():.1f}] days")
    slog.line(logger, f"Wavelength range: [{wavelengths.min():.1f}, {wavelengths.max():.1f}] Å")
    slog.line(logger, f"Colour law: {n_coeffs} coefficients, version {colorlaw_version}")

    error_model = _load_salt_error_model(
        model_dir, prefix, version, info_params['color_disp_max'],
        info_params['errmap_kcor_opt'], info_params['errmap_badval_abort'])
    for ranges, sed_grid, tolerance, unit in (
            (error_model.pop('wave_ranges'), waves_0, 10.0, 'Å'),
            (error_model.pop('phase_ranges'), phases_0, 1.1, 'days')):
        for name, (lo, hi) in ranges:
            if lo - tolerance > sed_grid[0] or hi + tolerance < sed_grid[-1]:
                message = (f"SALT error map {name} covers {lo:g}–{hi:g} {unit}, "
                           f"short of the SED range {sed_grid[0]:g}–{sed_grid[-1]:g} {unit}.")
                if mskopt & 64:
                    raise SNANAError(message, function_name="load_salt_data")
                logger.warning(message)

    return {
        'version': version,
        'model_name': model_name,
        'phases': jnp.array(phases),
        'wavelengths': jnp.array(wavelengths),
        'm0': jnp.array(m0),
        'm1': jnp.array(m1),
        'colorlaw_coeffs': jnp.array(colorlaw_coeffs),
        'colorlaw_version': colorlaw_version,
        'colorlaw_lam_min': colorlaw_lam_min,
        'colorlaw_lam_max': colorlaw_lam_max,
        'info_params': info_params,
        'error_model': error_model,
    }


def _quad_interp(
    x: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    f0: np.ndarray,
    f1: np.ndarray,
    f2: np.ndarray,
) -> np.ndarray:
    """Vectorized SNANA quadInterp for 3-point quadratic interpolation."""
    if x.shape != f0.shape:
        x = np.broadcast_to(x, f0.shape)
    if v0.shape != f0.shape:
        v0 = np.broadcast_to(v0, f0.shape)
        v1 = np.broadcast_to(v1, f0.shape)
        v2 = np.broadcast_to(v2, f0.shape)

    flat = (f0 == f1) & (f1 == f2)

    dum = np.divide(
        (f2 - f1),
        (f1 - f0),
        out=np.zeros_like(f0),
        where=(f1 != f0),
    )
    linear = (f1 != f0) & (np.abs(dum - 1.0) < 1.0e-12)

    dval = v1 - v0
    slope = np.divide(
        (f1 - f0),
        dval,
        out=np.zeros_like(f0),
        where=(dval != 0.0),
    )
    linear_val = f0 + slope * (x - v0)

    sqv0 = v0 * v0
    sqv1 = v1 * v1
    sqv2 = v2 * v2

    sqdval10 = sqv1 - sqv0
    sqdval21 = sqv2 - sqv1
    sqdval02 = sqv0 - sqv2

    dval10 = v1 - v0
    dval21 = v2 - v1
    dval02 = v0 - v2
    dprod = dval21 * dval10 * dval02

    fprod = f0 * dval21 + f1 * dval02 + f2 * dval10
    b = np.divide(
        -fprod,
        dprod,
        out=np.zeros_like(f0),
        where=(dprod != 0.0),
    )

    top = f0 * sqdval21 + f1 * sqdval02 + f2 * sqdval10
    lbar = np.divide(
        0.5 * top,
        fprod,
        out=np.zeros_like(f0),
        where=(fprod != 0.0),
    )

    a = f0 - b * (v0 - lbar) * (v0 - lbar)
    quad_val = a + b * (x - lbar) * (x - lbar)

    out = np.where(flat, f1, np.where(linear, linear_val, quad_val))
    return out


def _spline_smooth_sed(
    sed: np.ndarray,
    phases: np.ndarray,
    wavelengths: np.ndarray,
    nrebin_day: int,
    nrebin_lam: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply a spline-like smoothing similar to SNANA's SEDFLUX_INTERP_OPT=2.

    For each (phase, wavelength) node, use a 3x3 neighbourhood (clamped at edges)
    and perform quadratic interpolation in wavelength, then in phase.
    """
    phase_steps = np.diff(phases)
    if nrebin_day > 1 and np.allclose(phase_steps, phase_steps[0], atol=1e-6, rtol=0.0):
        base_step = phase_steps[0]
        new_step = base_step / float(nrebin_day)
        n_new = (len(phases) - 1) * nrebin_day + 1
        phases_out = phases[0] + np.arange(n_new) * new_step
    else:
        phases_out = phases

    wave_steps = np.diff(wavelengths)
    if nrebin_lam > 1 and np.allclose(wave_steps, wave_steps[0], atol=1e-6, rtol=0.0):
        base_step = wave_steps[0]
        new_step = base_step / float(nrebin_lam)
        n_new = (len(wavelengths) - 1) * nrebin_lam + 1
        wavelengths_out = wavelengths[0] + np.arange(n_new) * new_step
    else:
        wavelengths_out = wavelengths

    phases_in = phases.astype(np.float64, copy=False)
    waves_in = wavelengths.astype(np.float64, copy=False)
    phases_out_j = phases_out.astype(np.float64, copy=False)
    waves_out_j = wavelengths_out.astype(np.float64, copy=False)
    sed_in = sed.astype(np.float64, copy=False)

    wave_steps_in = np.diff(waves_in)
    uniform_wave = np.allclose(wave_steps_in, wave_steps_in[0], atol=1e-6, rtol=0.0)
    if uniform_wave:
        lamstep_orig = wave_steps_in[0]
        dif = waves_out_j - waves_in[0] + 1.0e-4
        idx_w = np.floor(dif / lamstep_orig).astype(int)
        idx_w = np.clip(idx_w, 0, len(waves_in) - 1)
        frac = (waves_out_j - waves_in[idx_w]) / lamstep_orig
        idx_w = np.where((frac < 0.5) & (idx_w > 0), idx_w - 1, idx_w)
        idx_w = np.minimum(idx_w, len(waves_in) - 3)
        w0 = idx_w
        w1 = idx_w + 1
        w2 = idx_w + 2
    else:
        idx_w = np.searchsorted(waves_in, waves_out_j) - 1
        idx_w = np.clip(idx_w, 0, len(waves_in) - 2)
        w0 = np.clip(idx_w - 1, 0, len(waves_in) - 1)
        w1 = idx_w
        w2 = np.clip(idx_w + 1, 0, len(waves_in) - 1)

    x0 = waves_in[w0]
    x1 = waves_in[w1]
    x2 = waves_in[w2]
    x = waves_out_j

    y0 = sed_in[:, w0]
    y1 = sed_in[:, w1]
    y2 = sed_in[:, w2]

    x = waves_out_j[None, :]
    v0 = x0[None, :]
    v1 = x1[None, :]
    v2 = x2[None, :]
    interp_w = _quad_interp(
        x,
        v0,
        v1,
        v2,
        y0,
        y1,
        y2,
    )

    phase_steps_in = np.diff(phases_in)
    uniform_phase = np.allclose(phase_steps_in, phase_steps_in[0], atol=1e-6, rtol=0.0)
    if uniform_phase:
        daystep_orig = phase_steps_in[0]
        dif = phases_out_j - phases_in[0] + 1.0e-4
        idx_p = np.floor(dif / daystep_orig).astype(int)
        idx_p = np.clip(idx_p, 0, len(phases_in) - 1)
        frac = (phases_out_j - phases_in[idx_p]) / daystep_orig
        idx_p = np.where((frac < 0.5) & (idx_p > 0), idx_p - 1, idx_p)
        idx_p = np.minimum(idx_p, len(phases_in) - 3)
        p0 = idx_p
        p1 = idx_p + 1
        p2 = idx_p + 2
    else:
        idx_p = np.searchsorted(phases_in, phases_out_j) - 1
        idx_p = np.clip(idx_p, 0, len(phases_in) - 2)
        p0 = np.clip(idx_p - 1, 0, len(phases_in) - 1)
        p1 = idx_p
        p2 = np.clip(idx_p + 1, 0, len(phases_in) - 1)

    p0v = phases_in[p0]
    p1v = phases_in[p1]
    p2v = phases_in[p2]

    x = phases_out_j[:, None]
    v0 = p0v[:, None]
    v1 = p1v[:, None]
    v2 = p2v[:, None]
    f0 = interp_w[p0]
    f1 = interp_w[p1]
    f2 = interp_w[p2]
    smoothed = _quad_interp(x, v0, v1, v2, f0, f1, f2)

    return smoothed, phases_out, wavelengths_out


def _smooth_interp_two_seds(phase, grid_phases, sed0, sed1, slopes0, slopes1):
    """Interpolate the M0 and M1 surfaces at one phase on a shared basis.

    Forward pass and AD gradients use the same cubic Hermite basis, so the
    derivatives are exact at all orders.
    """
    idx, h, (h00, h10, h01, h11) = hermite_weights(phase, grid_phases)

    def _cubic_eval(sed, slopes):
        return (h00 * sed[idx] + h10 * h * slopes[idx]
                + h01 * sed[idx + 1] + h11 * h * slopes[idx + 1])

    return _cubic_eval(sed0, slopes0), _cubic_eval(sed1, slopes1)


@jit
def _interp_sed_at_phase(
    phase: float,
    grid_phases: jnp.ndarray,
    sed_grid: jnp.ndarray
) -> jnp.ndarray:
    """Interpolate a 2D SED grid to one phase (phase axis only; the grid must
    already be resampled onto the working wavelength grid). One searchsorted
    covers every wavelength, then the two neighbouring rows are gathered and
    linearly interpolated. ``grid_phases`` is (n_phases,), ``sed_grid`` is
    (n_phases, n_wavelengths); returns (n_wavelengths,).
    """
    n = grid_phases.shape[0]
    phase_clamped = jnp.clip(phase, grid_phases[0], grid_phases[-1])
    idx = jnp.clip(jnp.searchsorted(grid_phases, phase_clamped, side='right') - 1, 0, n - 2)
    y0 = sed_grid[idx]
    y1 = sed_grid[idx + 1]
    x0 = grid_phases[idx]
    x1 = grid_phases[idx + 1]
    t = (phase_clamped - x0) / (x1 - x0 + 1e-12)
    return y0 + t * (y1 - y0)


@jit
def _salt_colorfun_pol(rl: jnp.ndarray, params: jnp.ndarray, alpha: float) -> jnp.ndarray:
    """SALT2 colour law polynomial P(rl) = alpha*rl + sum_i(params[i] * rl^(i+2))
    (SALT2colorfun_pol, genmag_SALT2.c:3918). ``rl`` is the reduced
    wavelength (λ - λ_B)/(λ_V - λ_B); ``alpha = 1 - sum(params)``.
    """
    v = alpha * rl
    rlp = rl * rl

    for i in range(len(params)):
        v = v + params[i] * rlp
        rlp = rlp * rl

    return v


@jit
def _salt_colorfun_dpol(rl: jnp.ndarray, params: jnp.ndarray, alpha: float) -> jnp.ndarray:
    """Derivative dP/drl = alpha + sum_i((i+2)*params[i] * rl^(i+1)) of
    :func:`_salt_colorfun_pol` (SALT2colorfun_dpol, genmag_SALT2.c:3906)."""
    v = alpha
    rlp = rl

    for i in range(len(params)):
        v = v + (i + 2) * params[i] * rlp
        rlp = rlp * rl

    return v


def _make_band_integrator(m0, m1, extrap_seds, basis, offset, phase_grid):
    """Colour-weighted M0/M1 and boundary integrals at each observation,
    with their colour derivatives in closed form.

    Colour enters only as ``exp((c - offset) * basis)``, so d/dc multiplies
    the integrand by ``basis`` and the derivatives are the same products
    against ``CLW * basis**k``. One stacked matrix product gives a value and
    its derivative together, where autodiff pushes a tangent through the
    whole integral for every fit parameter, three of which do not touch it.

    ``bw`` denotes band weights, ``dbw`` their directional derivative, and
    ``bwz`` their redshift derivative. ``dc`` and ``dz`` are the colour and
    redshift components of the tangent direction.
    """
    def _stack(c, bw, powers):
        cc = jnp.exp((c - offset) * basis)
        W = bw * cc[None, :]
        S = jnp.concatenate([W * basis[None, :] ** k for k in powers], axis=0)
        nf = bw.shape[0]
        A0, A1 = m0 @ S.T, m1 @ S.T
        B = jnp.einsum('scw,fw->scf', extrap_seds, S)
        return [(A0[:, i*nf:(i+1)*nf], A1[:, i*nf:(i+1)*nf], B[..., i*nf:(i+1)*nf])
                for i in range(len(powers))]

    def _tangent(c, bw, dc, dbw, higher, power, bwz=None, dz=0.0):
        """Tangent of the integral taken against ``CLW * basis**power``.

        Colour raises the power by one, which the caller supplies as
        ``higher``. The band weights enter linearly and so carry the same
        power as the quantity being differentiated; they are constant under
        spec-z and vary with the fitted redshift under photo-z.
        """
        cc = jnp.exp((c - offset) * basis) * basis ** power
        if bwz is not None:
            # photo-z: products against d(bw)/dz, scaled by dz
            Sz = bwz * cc[None, :]
            A0, A1 = (m0 @ Sz.T) * dz, (m1 @ Sz.T) * dz
            Bw = jnp.einsum('scw,fw->scf', extrap_seds, Sz) * dz
        else:
            Sw = dbw * cc[None, :]
            A0, A1 = m0 @ Sw.T, m1 @ Sw.T
            Bw = jnp.einsum('scw,fw->scf', extrap_seds, Sw)
        return (higher[0] * dc + A0, higher[1] * dc + A1, higher[2] * dc + Bw)

    @jax.custom_jvp
    def _moments(c, bw, z, bwz):
        val, first = _stack(c, bw, (0, 1))
        return (*val, *first)

    @_moments.defjvp
    def _moments_jvp(primals, tangents):
        c, bw, z, bwz = primals
        dc, dbw, dz, _ = tangents
        sg = jax.lax.stop_gradient
        val, first, second = _stack(sg(c), sg(bw), (0, 1, 2))
        return ((*val, *first),
                (*_tangent(c, bw, dc, dbw, first, 0, bwz, dz),
                 *_tangent(c, bw, dc, dbw, second, 1, bwz, dz)))

    def at_observations(integrals, phases, filters):
        M0, M1, boundary = integrals
        values = tuple(hermite_interp_1d_select(
            phases, phase_grid, surface,
            compute_catmullrom_slopes(phase_grid, surface), filters)
            for surface in (M0, M1))
        side = jnp.where(phases < phase_grid[0], 0, 1)
        return (*values, boundary[side, :, filters])

    @jax.custom_jvp
    def integrate(c, bw, z, bwz, phases, filters):
        return at_observations(_moments(c, bw, z, bwz)[:3], phases, filters)

    @integrate.defjvp
    def integrate_jvp(primals, tangents):
        c, bw, z, bwz, phases, filters = primals
        dc, dbw, dz, _, dphases, _ = tangents
        M0, M1, B, P0, P1, PB = _moments(c, bw, z, bwz)
        value, phase_tangent = jax.jvp(
            lambda phase: at_observations((M0, M1, B), phase, filters),
            (phases,), (dphases,))
        colour = at_observations((P0, P1, PB), phases, filters)
        weights = at_observations(
            _stack(c, dbw if bwz is None else bwz, (0,))[0], phases, filters)
        # Apply scalar tangents after lookup: transposition then reduces over
        # observations without scattering into shared phase/filter entries.
        tangent = tuple(phase_part + dc * colour_part
                        + (weight_part if bwz is None else dz * weight_part)
                        for phase_part, colour_part, weight_part
                        in zip(phase_tangent, colour, weights))
        return value, tangent

    return integrate


def _precompute_colorlaw_basis(
    wavelength_grid: jnp.ndarray,
    lam_b: float,
    lam_v: float,
    lam_min: float,
    lam_max: float,
    coeffs: jnp.ndarray,
    version: int,
) -> jnp.ndarray:
    """Precompute ``constant * P(rl)`` per wavelength, so the runtime colour
    correction reduces to ``CL(c, λ) = exp(c * basis)``.

    ``version`` selects SALT2colorlaw0 (Guy 2007) or SALT2colorlaw1
    (Guy 2010); law 0 has no lam_min/lam_max extrapolation. ``lam_b`` and
    ``lam_v`` are the B- and V-band central wavelengths (~4300 Å, ~5400 Å).
    Beyond ``lam_min``/``lam_max``, where the polynomial is not valid, the
    basis is extrapolated linearly from its edge value and slope. Returns
    the basis, shape (n_wavelengths,).
    """
    constant = jnp.log(10.0) / 2.5

    if version == 0:
        cor0, cor1 = coeffs[0], coeffs[1]
        lr = (wavelength_grid - lam_b) / (lam_v - lam_b)
        clam = (lr + cor0 * lr**2 + cor1 * lr**3) / (1.0 + cor0 + cor1)
        return constant * clam

    alpha = 1.0 - jnp.sum(coeffs)

    rl = (wavelength_grid - lam_b) / (lam_v - lam_b)
    rlmin = (lam_min - lam_b) / (lam_v - lam_b)
    rlmax = (lam_max - lam_b) / (lam_v - lam_b)

    Pmin = _salt_colorfun_pol(rlmin, coeffs, alpha)
    dPmin = _salt_colorfun_dpol(rlmin, coeffs, alpha)
    Pmax = _salt_colorfun_pol(rlmax, coeffs, alpha)
    dPmax = _salt_colorfun_dpol(rlmax, coeffs, alpha)

    P_interior = _salt_colorfun_pol(rl, coeffs, alpha)
    P_uv = Pmin + dPmin * (rl - rlmin)
    P_ir = Pmax + dPmax * (rl - rlmax)

    val = jnp.where(
        wavelength_grid < lam_min,
        P_uv,
        jnp.where(
            wavelength_grid > lam_max,
            P_ir,
            P_interior
        )
    )

    return constant * val


@jit
def generate_salt_sed(
    phase: float,
    x0: float,
    x1: float,
    c: float,
    wavelength_grid: jnp.ndarray,
    salt_phases: jnp.ndarray,
    m0_grid: jnp.ndarray,
    m1_grid: jnp.ndarray,
    colorlaw_basis: jnp.ndarray,
    extrap_boundary_seds: jnp.ndarray,
    m0_slopes: jnp.ndarray,
    m1_slopes: jnp.ndarray,
    latetime_daymin: float = 0.0,
    latetime_tau1: jnp.ndarray = None,
    latetime_tau2: jnp.ndarray = None,
    latetime_expratio: jnp.ndarray = None,
    latetime_m0: jnp.ndarray = None,
    latetime_m1: jnp.ndarray = None,
) -> jnp.ndarray:
    """SALT SED at one phase (pure functional, JIT-compiled):
    ``F(phase, λ) = x0 * [M0(phase, λ) + x1*M1(phase, λ)] * CL(c, λ)``.

    ``wavelength_grid`` is the output wavelength array (Angstroms); ``m0_grid``
    and ``m1_grid`` must already be resampled onto it, since this function
    interpolates in phase only. ``colorlaw_basis`` is the precomputed colour
    law basis ``constant * P(rl)``, shape (n_wavelengths,).
    ``extrap_boundary_seds`` holds the precomputed boundary SEDs used for
    phase extrapolation, shape (2, 4, n_wavelengths): index 0 is the low side
    ``[m0[0], m1[0], m0(phase_min+3), m1(phase_min+3)]``, index 1 the high
    side ``[m0[-1], m1[-1], m0(phase_max-3), m1(phase_max-3)]``.
    ``m0_slopes``/``m1_slopes`` are the Catmull-Rom slopes, shape
    (n_phases, n_wavelengths), used for a smooth (AD-differentiable) cubic
    Hermite interpolation in phase. Returns the rest-frame SED before
    dust/cosmology (erg/s/cm²/Å).
    """
    phase_min = salt_phases[0]
    phase_max = salt_phases[-1]

    outside_low = phase < phase_min
    outside_high = phase > phase_max
    outside = outside_low | outside_high

    color_correction = jnp.exp(c * colorlaw_basis)

    main_phase = jnp.clip(phase, phase_min, phase_max)
    main_m0, main_m1 = _smooth_interp_two_seds(
        main_phase, salt_phases, m0_grid, m1_grid, m0_slopes, m1_slopes
    )
    main_flux = x0 * (main_m0 + x1 * main_m1) * color_correction

    if latetime_tau1 is not None:
        # FLAM replaces SEDFLUX: zero before phase_min, decay after DAYMIN
        flam_flux_daymin = x0 * (latetime_m0 + x1 * latetime_m1) * color_correction
        decay = LatetimeExtrapIa.decay_factor(
            phase, latetime_daymin, latetime_tau1, latetime_tau2, latetime_expratio
        )
        flam_flux = flam_flux_daymin * decay

        use_flam = phase > latetime_daymin
        sed = jnp.where(use_flam, flam_flux, main_flux)
        sed = jnp.where(outside_low, 0.0, sed)
    else:
        side = jnp.where(outside_low, extrap_boundary_seds[0], extrap_boundary_seds[1])
        edge_m0 = side[0]
        edge_m1 = side[1]
        inner_m0 = side[2]
        inner_m1 = side[3]

        edge_flux = x0 * (edge_m0 + x1 * edge_m1) * color_correction
        inner_flux = x0 * (inner_m0 + x1 * inner_m1) * color_correction

        delta_phase = jnp.where(outside_low, -3.0, 3.0)
        slope_flux = jnp.where(outside, (edge_flux - inner_flux) / delta_phase, 0.0)
        safe_edge_phase = jnp.where(outside,
                                    jnp.where(outside_low, phase_min, phase_max),
                                    phase)
        safe_edge_flux = jnp.where(outside, edge_flux, main_flux)
        extrap_flux = modelflux_extrap(phase, safe_edge_phase, safe_edge_flux, slope_flux)

        sed = jnp.where(outside, extrap_flux, main_flux)

    return sed


@dataclass(eq=False, repr=False, kw_only=True)
class SALTModel(SNModel, SimulatedModel, FittedModel, StandardisedModel):
    """SALT2/SALT3 SED model:
    ``F(phase, λ) = x0 * [M0(phase, λ) + x1*M1(phase, λ)] * CL(c, λ)``, with
    ``x0`` the normalisation (amplitude), ``x1`` the stretch (light curve
    shape) and ``c`` the colour parameter; ``av``/``rv`` (dust extinction)
    are applied separately. The version (SALT2 or SALT3) is auto-detected
    from the model path.

    References: Guy et al. 2010 (SALT2,
    https://ui.adsabs.harvard.edu/abs/2010A%26A...523A...7G), Kenworthy et
    al. 2021 (SALT3, https://ui.adsabs.harvard.edu/abs/2021ApJ...923..265K).
    """

    wavelength_grid: jax.Array = None
    phases: jax.Array = None
    wavelengths: jax.Array = None
    m0: jax.Array = None
    m1: jax.Array = None
    m0_phase_slopes: jax.Array = None
    m1_phase_slopes: jax.Array = None
    colorlaw_basis: jax.Array = None
    extrap_boundary_seds: jax.Array = None
    latetime_tau1: jax.Array = None
    latetime_tau2: jax.Array = None
    latetime_expratio: jax.Array = None
    latetime_daymin: jax.Array = None
    m0_at_daymin: jax.Array = None
    m1_at_daymin: jax.Array = None
    fluxcal_scale_mag_offset: jax.Array = None
    magerr_floor: jax.Array = None
    color_offset: jax.Array = None
    colorlaw_coeffs: jax.Array = None
    error_model: dict = field(default_factory=dict)
    magerr_lamobs: tuple = ()
    magerr_lamrest: tuple = ()
    version: int = field(default=3, metadata={"static": True, "input": False})
    errmap_interp_opt: int = field(default=1, metadata={"static": True, "input": False})
    restlam_forcezeroflux: tuple = field(
        default=(0., 0.), metadata={"static": True, "input": False})
    has_offdiag_covariance: bool = field(
        default=False, metadata={"static": True, "input": False})
    distance_param: str = field(default='x0', metadata={"static": True, "input": False})
    extrap_phase_method: int = field(default=1, metadata={"static": True, "input": False})
    model_name: object = field(init=False, default=None, repr=False)
    model_dir_name: object = field(init=False, default=None, repr=False)
    native_wavelength_range: object = field(init=False, default=None, repr=False)
    model_info: object = field(init=False, default=None, repr=False)
    filter_center_range: object = field(init=False, default=None, repr=False)
    colorlaw_lam_b: object = field(init=False, default=None, repr=False)
    colorlaw_lam_v: object = field(init=False, default=None, repr=False)
    colorlaw_lam_min: object = field(init=False, default=None, repr=False)
    colorlaw_lam_max: object = field(init=False, default=None, repr=False)
    colorlaw_version: object = field(init=False, default=None, repr=False)
    extrap_latetime: object = field(init=False, default=None, repr=False)
    enable_magshift: object = field(init=False, default=None, repr=False)
    enable_waveshift: object = field(init=False, default=None, repr=False)
    # SALT2BETA_cPOLY's terms, each a coefficient or a (lo, hi) range
    _beta_cpoly: tuple = field(
        default=(), repr=False, metadata={"static": True, "input": False})

    GENMODEL_ALIASES = ('salt2', 'salt3')

    # SNANA's spellings and init values (snlc_sim.c:895-917); alpha and beta
    # are also the HOSTLIB columns USESNPAR reads (sntools_host.c:676-680)
    population_parameters = ('x1', 'c', 'SALT2alpha', 'SALT2beta')
    population_aliases = {'x1': ('SALT2x1',), 'c': ('SALT2c',),
                          'SALT2alpha': ('SALT2ALPHA', 'alpha'),
                          'SALT2beta': ('SALT2BETA', 'beta')}
    population_defaults = {
        'x1': {'peak': -9.0, 'sigma': (5.0, 5.0), 'range': (-5.0, 5.0)},
        'c': {'peak': -9.0, 'sigma': (1.0, 1.0), 'range': (-0.5, 1.0)},
        'SALT2alpha': {'range': (0.001, 0.40)},
        'SALT2beta': {'range': (0.5, 9.9)},
    }
    INPUT_KEY_ALIASES = (('GENPEAK_SALT2ALPHA', 'GENALPHA_SALT2'),
                         ('GENPEAK_SALT2BETA', 'GENBETA_SALT2'))

    def configure_generation(self, snana_config, n_simulated, frac_pec1a):
        self._beta_cpoly = tuple(
            tuple(term) if isinstance(term, (tuple, list)) else term
            for term in snana_config['salt2beta_cpoly'])

    def replay_params_from_dump(self, dump_columns, head_columns, *, mu,
                                distributions):
        required = ('SIM_SALT2x0', 'SIM_SALT2x1', 'SIM_SALT2c')
        missing = [column for column in required if column not in dump_columns]
        if missing:
            names = [[name for name, target in self.SNANA_DUMP_SIM_COLUMNS
                      if target == column] for column in missing]
            raise SNANAError(
                f"Missing required SALT columns in dump, each under any of: "
                f"{names}", function_name="replay_params_from_dump")

        salt_x0 = dump_columns['SIM_SALT2x0']
        salt_x1 = dump_columns['SIM_SALT2x1']
        salt_c = dump_columns['SIM_SALT2c']
        invalid = (salt_x0 <= -8.5) | (salt_x1 <= -8.5) | (salt_c <= -8.5)

        # alpha/beta absent from the dump: population peak
        def _population(name, column):
            if column in dump_columns:
                return dump_columns[column].astype(np.float32)
            return np.full(len(salt_x0), float(distributions[name].peak),
                           dtype=np.float32)

        return {
            'x0': salt_x0,
            'x1': salt_x1,
            'c': salt_c,
            'SALT2alpha': _population('SALT2alpha', 'SIM_SALT2alpha'),
            'SALT2beta': _population('SALT2beta', 'SIM_SALT2beta'),
        }, invalid
    # beyond the fit parameters' truth: BBC's biasCor reads these
    SIM_TRUTH_COLUMNS = ('SIM_SALT2mB', 'SIM_SALT2alpha', 'SIM_SALT2beta',
                         'SIM_SALT2gammaDM')
    # SIMNAME_SHAPEPAR/COLORPAR/SHAPELAW/COLORLAW naming (snana.F90:24614-24650)
    SNANA_TABLE_SIM_COLUMNS = (
        ('SIM_x1', 'SIM_SALT2x1'), ('SIM_c', 'SIM_SALT2c'),
        ('SIM_alpha', 'SIM_SALT2alpha'), ('SIM_beta', 'SIM_SALT2beta'),
        ('SIM_gammaDM', 'SIM_SALT2gammaDM'), ('SIM_x0', 'SIM_SALT2x0'),
        ('SIM_mB', 'SIM_SALT2mB'))

    #: Fit x0 directly (0) or log10(x0) (1). The logarithmic fit uses fixed
    #: parameter bounds; direct x0 bounds are relative to its initial value.
    opt_salt2fit: Annotated[Literal[0, 1], FIT_ONLY] = field(
        default=0, metadata={"static": True})

    #: Shape coefficient used, with SALT2BETA, to compute the x0 expected at a
    #: given distance modulus.
    salt2alpha: Annotated[float, FIT_ONLY] = 0.14
    #: Colour coefficient used, with SALT2ALPHA, to compute the x0 expected
    #: at a given distance modulus.
    salt2beta: Annotated[float, FIT_ONLY] = 3.20

    # DOCANA-checked, read for MAGSHIFT/WAVESHIFT (genmag_SALT2.c:1307)
    METADATA_FILES = ('SALT?.INFO',)

    def _amplitude_to_x0(self, value):
        """The fit vector's amplitude entry as x0, for the SED."""
        return jnp.power(10.0, value) if self.opt_salt2fit == 1 else value

    def _x0_to_amplitude(self, x0):
        """x0 as the fit vector's amplitude entry, for initial guesses."""
        return jnp.log10(x0) if self.opt_salt2fit == 1 else x0

    def _amplitude_at_distance(self, mu, params):
        """SALT2x0calc, then the encoding the fit vector uses (SNANA
        GET_DIST8, snlc_fit.F90:6122). Brightness depends on shape and colour,
        so those must arrive physical."""
        return self._x0_to_amplitude(compute_expected_x0_scale(
            mu, self.salt2alpha, self.salt2beta, params['x1'], params['c']))

    def _distance_at_amplitude(self, value, params):
        """The distance modulus FCNSNLC forms from SALT's parameters,
        30 - 2.5 log10(x0) + alpha x1 - beta c (snlc_fit.F90:4832)."""
        return (-2.5 * jnp.log10(X0SCALE_SALT2 * self._amplitude_to_x0(value))
                + self.salt2alpha * params['x1'] - self.salt2beta * params['c'])

    @classmethod
    def load(cls, model_path: str, config, n_wave_points=500,
             wavelength_range=None):
        """Load the M0/M1 surfaces and the colour law from ``model_path``
        (e.g. '$SNDATA_ROOT/models/SALT3/SALT3.K21'), extend them into the UV,
        take any late-time file, and resample onto the working grid.
        """
        model = cls()
        SNModel.__init__(model, f"salt{detect_salt_version(model_path)}", config,
                         n_wave_points, wavelength_range)
        mskopt = model.genmodel_mskopt
        unsupported = mskopt & ~(2 | 4 | 8 | 16 | 64)
        if unsupported:
            raise SNANAError(
                f"SALT GENMODEL_MSKOPT bits {unsupported} are not supported; "
                "supported bits are 2, 4, 8, 16 and 64.",
                function_name="SALTModel")
        model.enable_magshift = not bool(mskopt & 4)
        model.enable_waveshift = not bool(mskopt & 8)
        salt_data = load_salt_data(model_path, mskopt)

        model.distance_param = 'logx0' if model.opt_salt2fit == 1 else 'x0'

        model.version = salt_data['version']
        model.model_dir_name = salt_data['model_name']

        phases_np = np.array(salt_data['phases'])
        waves_np = np.array(salt_data['wavelengths'])
        m0_np = np.array(salt_data['m0'])
        m1_np = np.array(salt_data['m1'])

        interp_opt = int(salt_data['info_params']['sedflux_interp_opt'])
        if interp_opt == 2:
            slog.line(logger, "Applying SALT SED spline smoothing (SEDFLUX_INTERP_OPT=2)")
            phases_in = phases_np
            waves_in = waves_np
            m0_np, phases_np, waves_np = _spline_smooth_sed(
                m0_np, phases_in, waves_in, NREBIN_DAY_DEFAULT, NREBIN_LAM_DEFAULT
            )
            m1_np, _, _ = _spline_smooth_sed(
                m1_np, phases_in, waves_in, NREBIN_DAY_DEFAULT, NREBIN_LAM_DEFAULT
            )

        model.phases = jnp.array(phases_np)
        model.wavelengths = jnp.array(waves_np)
        model.native_wavelength_range = (float(waves_np.min()), float(waves_np.max()))
        model.m0 = jnp.array(m0_np)
        model.m1 = jnp.array(m1_np)

        model.model_info = salt_data['info_params']
        model.fluxcal_scale_mag_offset = float(np.power(
            10.0, -0.4 * float(model.model_info["mag_offset"])))
        model.filter_center_range = (float(model.model_info['restlambda_min']),
                                    float(model.model_info['restlambda_max']))
        model.magerr_floor = float(model.model_info['magerr_floor'])
        model.magerr_lamobs = tuple(model.model_info['magerr_lamobs'])
        model.magerr_lamrest = tuple(model.model_info['magerr_lamrest'])
        model.errmap_interp_opt = int(model.model_info['errmap_interp_opt'])
        model.color_offset = float(model.model_info['color_offset'])
        model.restlam_forcezeroflux = tuple(
            model.model_info['restlam_forcezeroflux'])

        slog.line(logger, f"SALT{model.version} model ready: {model.model_dir_name}")

        # SNANA uses fixed B and V wavelengths (not from INFO file)
        model.colorlaw_lam_b = 4302.57  # B-band central wavelength (Å)
        model.colorlaw_lam_v = 5428.55  # V-band central wavelength (Å)
        model.colorlaw_lam_min = salt_data['colorlaw_lam_min']
        model.colorlaw_lam_max = salt_data['colorlaw_lam_max']
        model.colorlaw_version = salt_data['colorlaw_version']
        model.colorlaw_coeffs = jnp.array(salt_data['colorlaw_coeffs'])

        model.error_model = salt_data['error_model']
        model.has_offdiag_covariance = (
            model.error_model['color_dispersion'] is not None)

        slog.line(logger, f"Phase range: [{model.phases.min():.1f}, {model.phases.max():.1f}] days", indent=2)
        slog.line(logger, f"Wavelength range: [{model.wavelengths.min():.1f}, {model.wavelengths.max():.1f}] Å", indent=2)
        slog.line(logger, f"Colour law: {len(model.colorlaw_coeffs)} coefficients", indent=2)

        model.extrap_latetime = None
        model.extrap_phase_method = 1      # 1=SEDFLUX (default), 3=FLAM

        model._apply_uvlam_extrap(model.uvlam_extrapflux)
        if model.genmodel_extrap_latetime:
            model.set_extrap_latetime(model.genmodel_extrap_latetime)
        model.resample_to_grid(model.choose_grid())

        return model

    def get_wavelength_range(self) -> tuple:
        """The SALT template extent, UV extension included, as
        (min_wavelength, max_wavelength) in Angstroms: the support the
        working grid is cut from, not the grid."""
        return self.native_wavelength_range

    def resample_to_grid(self, wavelength_grid: jnp.ndarray) -> None:
        """Resample the M0/M1 templates onto ``wavelength_grid``, called once
        during setup so :func:`generate_salt_sed` only ever interpolates in
        phase."""
        native_wavelengths = self.wavelengths

        slog.line(logger, f"Resampling SALT M0/M1 from {len(native_wavelengths)} to {len(wavelength_grid)} wavelengths...")

        self.m0 = jnp.array(resample_surfaces(native_wavelengths, self.m0, wavelength_grid))
        self.m1 = jnp.array(resample_surfaces(native_wavelengths, self.m1, wavelength_grid))
        self.wavelengths = wavelength_grid

        self.m0_phase_slopes = compute_catmullrom_slopes(self.phases, self.m0)
        self.m1_phase_slopes = compute_catmullrom_slopes(self.phases, self.m1)

        self.colorlaw_basis = _precompute_colorlaw_basis(
            wavelength_grid,
            self.colorlaw_lam_b,
            self.colorlaw_lam_v,
            self.colorlaw_lam_min,
            self.colorlaw_lam_max,
            self.colorlaw_coeffs,
            self.colorlaw_version,
        )

        phase_min = self.phases[0]
        phase_max = self.phases[-1]
        inner_low = jnp.clip(phase_min + 3.0, phase_min, phase_max)
        inner_high = jnp.clip(phase_max - 3.0, phase_min, phase_max)
        inner_low_m0 = _interp_sed_at_phase(inner_low, self.phases, self.m0)
        inner_low_m1 = _interp_sed_at_phase(inner_low, self.phases, self.m1)
        inner_high_m0 = _interp_sed_at_phase(inner_high, self.phases, self.m0)
        inner_high_m1 = _interp_sed_at_phase(inner_high, self.phases, self.m1)
        self.extrap_boundary_seds = jnp.array([
            [self.m0[0], self.m1[0], inner_low_m0, inner_low_m1],
            [self.m0[-1], self.m1[-1], inner_high_m0, inner_high_m1],
        ])

        if self.extrap_latetime is not None:
            tau1, tau2, ratio = self.extrap_latetime.precompute_decay_params(wavelength_grid)
            self.latetime_tau1 = tau1
            self.latetime_tau2 = tau2
            self.latetime_expratio = ratio
            self.latetime_daymin = self.extrap_latetime.daymin

            daymin = jnp.asarray([float(self.latetime_daymin)])
            self.m0_at_daymin = hermite_interp_1d_batch(
                daymin, self.phases, self.m0, self.m0_phase_slopes)[0]
            self.m1_at_daymin = hermite_interp_1d_batch(
                daymin, self.phases, self.m1, self.m1_phase_slopes)[0]

            if float(self.phases[0]) > -20.0:
                raise SNANAError(
                    f"SALT phase_min={float(self.phases[0]):.1f} > -20: "
                    f"FLAM late-time extrap assumes phase_min <= T0_explosion=-20d. "
                    f"Pre-max fireball rise would be skipped for this model.",
                    function_name="resample_to_grid")

            slog.line(logger, f"Late-time decay params interpolated to {len(wavelength_grid)} wavelengths")

        slog.line(logger, f"SALT templates resampled to {len(wavelength_grid)} wavelengths")

    def _apply_uvlam_extrap(self, uvlam: float) -> None:
        """UVLAM_EXTRAPFLUX on the M0 and M1 surfaces.

        One call per surface, as SNANA's does (genmag_SALT2.c:320-321): each
        surface gets its own per-day anchor, and original blue bins below the
        anchor are overwritten by the ramp.
        """
        waves = np.asarray(self.wavelengths)
        lam_step = waves[1] - waves[0]
        lam_min_orig = waves[0]

        m0_np = np.asarray(self.m0)
        m1_np = np.asarray(self.m1)

        new_waves, m0_new = uvlam_extrapflux_sedmodel(uvlam, waves, m0_np)
        _, m1_new = uvlam_extrapflux_sedmodel(uvlam, waves, m1_np)
        n_uv = len(new_waves) - len(waves)
        if not n_uv:
            return

        self.wavelengths = jnp.array(new_waves)
        self.native_wavelength_range = (float(new_waves[0]), float(new_waves[-1]))
        self.m0 = jnp.array(m0_new)
        self.m1 = jnp.array(m1_new)

        # Filter centres follow the extrapolated edge
        # (UVLAM_EXTRAP_RESTLAMMIN_FILTERCEN, genmag_SEDtools.c:3171).
        self.filter_center_range = (uvlam + lam_step,
                                    self.filter_center_range[1])

        slog.line(logger, f"SALT UV extrapolation: extended from {lam_min_orig:.0f} to "
                  f"{new_waves[0]:.0f} A ({n_uv} new bins, step={lam_step:.0f} A)")

    def set_extrap_latetime(self, filename: str) -> None:
        """Load the late-time Ia double-exponential decay model from
        ``filename`` (GENMODEL_EXTRAP_LATETIME). Must be called before
        :meth:`resample_to_grid`, which interpolates the decay parameters
        onto the working wavelength grid."""
        self.extrap_latetime = LatetimeExtrapIa(filename)
        # SNANA preferred method for SALT with latetime file is FLAM (method=3)
        self.extrap_phase_method = 3
        slog.line(logger, "SALT extrap phase method: FLAM (per-wavelength decay)")

    def derived_parameters(self, params, keys_for, **kwargs):
        """x0 from the distance and the drawn x1, c, alpha and beta, by
        SNANA's SALT2x0calc. SALT2BETA_cPOLY adds a polynomial in c to beta,
        and is the whole of beta when beta has no population
        (snlc_sim.c:915, :14071-14078); a (lo, hi) term is drawn per SN."""
        c = params['c']
        # without cPOLY, beta must have been drawn
        if 'SALT2beta' in params or not self._beta_cpoly:
            beta = params['SALT2beta']
        else:
            beta = jnp.zeros_like(c)
        cpow = jnp.ones_like(c)
        for i, term in enumerate(self._beta_cpoly):
            if isinstance(term, tuple):
                lo, hi = float(term[0]), float(term[1])
                coef = jax.vmap(lambda k, lo=lo, hi=hi: jax.random.uniform(
                    k, minval=lo, maxval=hi))(keys_for(f'SALT2BETA_CPOLY{i}'))
            else:
                coef = float(term)
            beta = beta + coef * cpow
            cpow = cpow * c

        # SNANA aborts on negative alpha/beta (snlc_sim.c:14083); not checked
        x0 = compute_expected_x0_scale(params['mu'], params['SALT2alpha'],
                                       beta, params['x1'], c)
        return {'x0': x0, 'SALT2beta': beta}

    def apply_host_snpar_overrides(self, params, host_values):
        params = super().apply_host_snpar_overrides(params, host_values)
        # x0 must stay consistent with the overridden parameters
        # (SALT2x0calc re-run, snlc_sim.c:13099-13102)
        params['x0'] = compute_expected_x0_scale(
            params['mu'], params['SALT2alpha'], params['SALT2beta'],
            params['x1'], params['c'])
        return params

    def _compute_model_fractional_error(
        self,
        phase: jnp.ndarray,
        wavelength: jnp.ndarray,
        x1_err: float,
        x1: float,
        m0_int: jnp.ndarray,
        m1_int: jnp.ndarray,
        lam_obs: jnp.ndarray,
    ) -> jnp.ndarray:
        """SALT2magerr (genmag_SALT2.c:2362-2455) as a fractional flux error
        in nats, shape (n_obs,).

        ``phase`` is the rest-frame phases (days) and ``wavelength`` the
        filter mean rest-frame wavelengths (Å), both shape (n_obs,); ``x1_err``
        is the x1 the variance surface is read at (SNANA's x1_forErr) and
        ``x1`` the trial one, which the SALT3 training flux dividing the
        variance carries (Finteg_errPar, genmag_SALT2.c:2960); ``m0_int``/
        ``m1_int`` are the MW-free band integrals of M0/M1 with the colour law
        applied (Finteg_forErr), shape (n_obs,); ``lam_obs`` is the filter
        mean observer-frame wavelengths (Å), shape (n_obs,).
        """
        em = self.error_model
        err_phases = em['error_phases']
        err_wavelengths = em['error_wavelengths']

        # SNANA clamps Trest to the map range (genmag_SALT2.c:2386-2392)
        phase = jnp.clip(phase, err_phases[0], err_phases[-1])

        if self.errmap_interp_opt == 0:
            fracerr_snake = jnp.zeros_like(phase)
        else:
            var0 = hermite_interp_grid(
                phase, wavelength, err_phases, err_wavelengths,
                em['variance_0'], em['variance_0_slopes'])
            if isinstance(x1_err, (int, float)) and x1_err == 0.0:
                # x1_forErr=0 on iteration 1: vartot = var0 (x1 terms vanish exactly)
                vartot = var0
            else:
                var1 = hermite_interp_grid(
                    phase, wavelength, err_phases, err_wavelengths,
                    em['variance_1'], em['variance_1_slopes'])
                cov01 = hermite_interp_grid(
                    phase, wavelength, err_phases, err_wavelengths,
                    em['covariance_01'], em['covariance_01_slopes'])
                vartot = var0 + 2.0 * x1_err * cov01 + x1_err * x1_err * var1

            if self.version == 2:
                # relative variance: 0.01^2 floor, error-scale map, and the
                # |1 + x1*S1/S0| denominator, at the surface's x1
                vartot = jnp.where(vartot < 0.0, 1.0e-4, vartot)
                errscale = hermite_interp_grid(
                    phase, wavelength, em['errscale_phases'],
                    em['errscale_wavelengths'], em['errscale'],
                    em['errscale_slopes'])
                fratio = m1_int / jnp.where(m0_int != 0.0, m0_int, 1.0)
                fracerr_snake = (errscale * jnp.sqrt(vartot)
                                 / jnp.abs(1.0 + x1_err * fratio))
            else:
                # SALT3 flux variance; zeros kept out of sqrt (where() differentiates both branches)
                vartot = jnp.abs(vartot)
                flux_train = jnp.abs(m0_int + x1 * m1_int)
                defined = (flux_train > 0.0) & (vartot > 0.0)
                fracerr_snake = jnp.where(
                    defined,
                    jnp.sqrt(jnp.where(defined, vartot, 1.0))
                    / jnp.where(defined, flux_train, 1.0), 0.0)

        if em['color_dispersion'] is not None:
            fracerr_kcor = hermite_interp_1d(
                wavelength, em['color_dispersion_wavelengths'],
                em['color_dispersion'], em['color_dispersion_slopes'])
        else:
            fracerr_kcor = jnp.zeros_like(wavelength)

        fracerr_sq = fracerr_snake**2 + fracerr_kcor**2
        fracerr_tot = jnp.where(
            fracerr_sq > 0.0,
            jnp.sqrt(jnp.where(fracerr_sq > 0.0, fracerr_sq, 1.0)), 0.0)

        # magerrFudge_SALT2: each fudge replaces from magerr_model, so the
        # floor and the fudges do not stack
        mag_per_nat = 2.5 / np.log(10.0)
        magerr_model = mag_per_nat * fracerr_tot
        magerr = jnp.maximum(magerr_model, self.magerr_floor)
        obs_add, obs_lo, obs_hi = self.magerr_lamobs
        magerr = jnp.where((lam_obs >= obs_lo) & (lam_obs <= obs_hi),
                           jnp.sqrt(magerr_model**2 + obs_add**2), magerr)
        rest_add, rest_lo, rest_hi = self.magerr_lamrest
        magerr = jnp.where((wavelength >= rest_lo) & (wavelength <= rest_hi),
                           jnp.sqrt(magerr_model**2 + rest_add**2), magerr)
        return magerr / mag_per_nat

    def sed_param_names(self):
        return ('x0', 'x1', 'c')

    def phase_range(self):
        """SALT's own DAYMIN/DAYMAX (genmag_SALT2.c:3592-3595)."""
        return float(self.phases[0]), float(self.phases[-1])

    def generate_sed(self, phase: float, params: Dict[str, float],
                     static_data: Dict) -> jnp.ndarray:
        """The rest-frame SALT SED at ``phase`` (days relative to peak), with
        SALT normalisation applied (erg/s/cm²/Å).

        ``params`` holds 'x0', 'x1', 'c' (av and rv are applied separately as
        dust extinction); ``static_data`` holds the wavelength grid.
        """
        x0 = params['x0']
        x1 = params['x1']
        c = params['c']

        latetime_kw = {}
        if self.latetime_tau1 is not None:
            latetime_kw = dict(
                latetime_daymin=self.latetime_daymin,
                latetime_tau1=self.latetime_tau1,
                latetime_tau2=self.latetime_tau2,
                latetime_expratio=self.latetime_expratio,
                latetime_m0=self.m0_at_daymin,
                latetime_m1=self.m1_at_daymin,
            )
        sed = generate_salt_sed(
            phase=phase,
            x0=x0,
            x1=x1,
            # cc = c - COLOR_OFFSET (SALT2colorCor, genmag_SALT2.c:2028)
            c=c - self.color_offset,
            wavelength_grid=static_data['wavelength_grid'],
            salt_phases=self.phases,
            m0_grid=self.m0,
            m1_grid=self.m1,
            colorlaw_basis=self.colorlaw_basis,
            extrap_boundary_seds=self.extrap_boundary_seds,
            m0_slopes=self.m0_phase_slopes,
            m1_slopes=self.m1_phase_slopes,
            **latetime_kw,
        )

        # x0 carries 1/X0SCALE_SALT2 (genmag_SALT2.c:2985)
        sed = sed * X0SCALE_SALT2

        return jnp.maximum(sed, 0.0) if self.genmodel_mskopt & 16 else sed

    def get_snana_header_params(self, parameters: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """SALT2 parameters mapped to SNANA FITS header column names
        ('SIM_SALT2x0', 'SIM_SALT2x1', etc).

        ``parameters`` holds 'x0', 'x1', 'c', 'sim_av', 'sim_rv': the sim_*
        dust pair is the reported one, carrying SNANA's -9 when no dust was
        generated, while 'av'/'rv' hold the values the extinction maths
        uses.
        """
        x0 = np.asarray(parameters['x0'])
        n = len(x0)
        # SALT2mBcalc (genmag_SALT2.c:3031); NULLFLOAT for the x0 sentinel
        mb = np.where(x0 > 0,
                      SALT2_MB_OFFSET - 2.5 * np.log10(np.where(x0 > 0, x0, 1.0)),
                      SNANA_NULLFLOAT)
        return {
            'SIM_SALT2x0': x0,
            'SIM_SALT2x1': parameters['x1'],
            'SIM_SALT2c': parameters['c'],
            'SIM_SALT2mB': mb,
            'SIM_SALT2alpha': parameters['SALT2alpha'],
            'SIM_SALT2beta': parameters['SALT2beta'],
            # HOSTLIB SNMAGSHIFT correction; 0 when unset (snlc_sim.c:9585)
            'SIM_SALT2gammaDM': parameters.get('snmagshift', np.zeros(n)),
        }

    # SNANA's name lists, verbatim (snlc_sim.c:16461-16507, :16524-16531)
    SNANA_DUMP_SIM_COLUMNS = (
        ('S2x0',         'SIM_SALT2x0'),
        ('SALT2x0',      'SIM_SALT2x0'),
        ('SIM_x0',       'SIM_SALT2x0'),
        ('S2x1',         'SIM_SALT2x1'),
        ('SALT2x1',      'SIM_SALT2x1'),
        ('SIM_x1',       'SIM_SALT2x1'),
        ('S2c',          'SIM_SALT2c'),
        ('SALT2c',       'SIM_SALT2c'),
        ('SIM_c',        'SIM_SALT2c'),
        ('S2mb',         'SIM_SALT2mB'),
        ('SALT2mb',      'SIM_SALT2mB'),
        ('SALT2mB',      'SIM_SALT2mB'),
        ('SIM_mb',       'SIM_SALT2mB'),
        ('SIM_mB',       'SIM_SALT2mB'),
        ('SALT2alpha',   'SIM_SALT2alpha'),
        ('S2alpha',      'SIM_SALT2alpha'),
        ('SIM_alpha',    'SIM_SALT2alpha'),
        ('SALT2beta',    'SIM_SALT2beta'),
        ('S2beta',       'SIM_SALT2beta'),
        ('SIM_beta',     'SIM_SALT2beta'),
        ('SALT2gammaDM', 'SIM_SALT2gammaDM'),
        ('SNMAGSHIFT',   'SIM_SALT2gammaDM'),
    )

    sim_model_index = 6                        # MODEL_SALT2, also for SALT3

    def sn_variables(self, params: Dict) -> Dict:
        """The declared population parameters, plus mB, which SNANA derives
        from x0 rather than sampling (SALT2mBcalc, genmag_SALT2.c:3031)."""
        sn_vars = super().sn_variables(params)
        if 'x0' in params:
            x0 = jnp.asarray(params['x0'], dtype=jnp.float32)
            safe_x0 = jnp.where(x0 > 0, x0, 1e-30)
            sn_vars['SALT2mB'] = SALT2_MB_OFFSET - 2.5 * jnp.log10(safe_x0)
        return sn_vars

    def get_fit_parameters(self) -> Dict[str, Dict[str, Any]]:
        """SALT fit parameters: peak time, amplitude, stretch, colour.

        ``x0`` spans orders of magnitude with redshift, so it is scaled to O(1)
        and bounded relative to its initial value rather than absolutely.
        OPT_SALT2FIT=1 fits log10(x0), which is
        already O(1) across the whole distance range and so takes absolute
        bounds from x0 at mu=70 and mu=20 (snlc_fit.F90:8095-8103).
        """
        if self.opt_salt2fit == 1:
            amplitude = {'logx0': {'bounds': LOGX0_BOUNDS, 'scale': 1.0,
                                   'errmax_bad': 0.0,
                                   'error_bounds': (1.0e-15, 3.0),
                                   'snana_keys': ('DLMAG',)}}
        else:
            amplitude = {'x0': {'bounds_relative_to_init': (0.01, 100.0),
                                'scale': 1e-5, 'scale_at_distance': True,
                                'sim_truth': 'SIM_SALT2x0',
                                'inistp_relative_to_init': 0.1,
                                'errmax_bad': 0.0,
                                'error_bounds': (1.0e-15, 3.0),
                                'snana_keys': ('DLMAG',)}}
        # SNANA's FITRES covariance order, COV_x1_c COV_x1_x0 COV_c_x0, with
        # no PKMJD term (snlc_fit.F90 WRITE_FITRES)
        return {
            'PKMJD': {**super().get_fit_parameters()['PKMJD'],
                      'covariance': False},
            # SALT2xx1 (snlc_fit.F90:13725): the error surface takes x1 = 0
            # on the first round
            'x1': {'bounds': (-5.0, 5.0), 'scale': 1.0, 'init': 0.001,
                   'sim_truth': 'SIM_SALT2x1', 'inistp': 0.3,
                   'errmax_bad': 0.001, 'error_bounds': (1.0e-8, 3.0),
                   'priors': (('flat_rolloff',
                               {'lo': -9.0, 'hi': 9.0, 'sigma': 0.1}),),
                   'peg_margin': 0.05, 'snana_keys': ('SHAPE', 'LUMIPAR'),
                   'grid_search': (-2.0, 0.0, 2.0),
                   'coarse_grid': {'range': (-3.0, 1.0), 'nbin': 3},
                   'error_surface_round1': 0.0},
            # CMAX_SALT2 = 0.5, on top of the flat prior (snlc_fit.F90:4770)
            'c': {'bounds': (-1.0, 1.0), 'scale': 1.0, 'init': 0.01,
                  'sim_truth': 'SIM_SALT2c', 'inistp': 0.03,
                  'errmax_bad': 0.001, 'error_bounds': (1.0e-8, 3.0),
                  'priors': (('flat_rolloff',
                              {'lo': -9.0, 'hi': 9.0, 'sigma': 0.01}),
                             ('protect_max', {'max': 0.5})),
                  'snana_keys': ('COLOR',),
                  'grid_search': (0.0,),
                  'coarse_grid': {'range': (-0.2, 0.25), 'nbin': 4}},
            **amplitude,
        }

    def sed_params_from_fit(self, params_arr, redshift):
        """The fit vector named for the SED, with the amplitude converted to
        x0 so everything below sees x0 whichever coordinate was fitted."""
        params = super().sed_params_from_fit(params_arr, redshift)
        params['x0'] = self._amplitude_to_x0(params.pop(self.distance_param))
        return params

    def _band_integrate(self, c, band_weights, phases, filter_indices,
                        redshift, band_weights_dz):
        """M0/M1 integrated through the colour-weighted filters and interpolated
        to each observation phase, with the extrapolation boundary values.

        This is the expensive part of a SALT evaluation and the only part the
        flux and the error surface share, so it is computed once and handed to
        both. It depends on colour but not on shape, which is what lets a grid
        over shape reuse it.
        """
        # cc = c - COLOR_OFFSET (SALT2colorCor, genmag_SALT2.c:2028)
        integrate = _make_band_integrator(
            self.m0, self.m1, self.extrap_boundary_seds,
            self.colorlaw_basis, self.color_offset, self.phases)
        return integrate(c, band_weights, redshift, band_weights_dz,
                         phases, filter_indices)

    def _host_reddened_weights(self, sed_params, band_weights, static_data):
        """Band weights carrying host extinction, or the weights unchanged
        where no host dust was generated.

        SALT has no SED to redden: it contracts wavelength against the weights
        once per SN. SNANA reddens the same way, as HOSTXT_FRAC beside
        MWXT_FRAC in the filter integrand (genmag_SALT2.c:2748-2752). Host
        dust is a simulation input for this model and is never fitted, so the
        factor carries no derivative and the error denominators do not see it.
        """
        if 'av' not in sed_params:
            return band_weights
        ext = static_data['host_dust_law'].apply_extinction(
            jnp.ones(band_weights.shape[-1]), static_data['wavelength_grid'],
            sed_params['av'], (sed_params['rv'] if 'rv' in sed_params
                            else static_data['host_rv']))
        return band_weights * ext

    def evaluate(self, sed_params, mjds, filter_indices, mask, band_weights,
                 static_data, sed_factor=None, band_weights_dz=None,
                 band_weights_dz2=None):
        """SALT FLUXCAL, pre-integrating M0/M1 through the filters.

        Overrides the base implementation, which would generate a full SED per
        observation and integrate it; contracting wavelength once per SN gives
        the same numbers far more cheaply.

        EXTRAP_LATETIME's decay is per wavelength, so it cannot fold into the
        phase-interpolated table; it is its own band integral against the
        DAYMIN surfaces, which needs no SED and no phase interpolation.
        ``band_weights_dz2`` is unused: SALT's redshift terms go through its
        per-filter integrals.
        """
        if self.genmodel_mskopt & 16:
            # Clipping depends on x1, so it must precede the band integral.
            lo, hi = self.restlam_forcezeroflux
            if hi > 0:
                mean = static_data['filter_mean_wavelengths'][filter_indices]
                rest_mean = mean / (1.0 + sed_params['redshift'])
                mask = mask & ~((rest_mean > lo) & (rest_mean < hi))
            def flux_at_phase(phases):
                return super(SALTModel, self).evaluate(
                    sed_params, sed_params['t0'] + phases * (1 + sed_params['redshift']),
                    filter_indices, mask, band_weights, static_data, sed_factor,
                    band_weights_dz, band_weights_dz2)

            phases = jnp.where(mask, (mjds - sed_params['t0']) /
                               (1 + sed_params['redshift']), 0.)
            if self.latetime_tau1 is not None:
                return flux_at_phase(phases)
            phase_min, phase_max = self.phases[0], self.phases[-1]
            flux = flux_at_phase(jnp.clip(phases, phase_min, phase_max))
            outside = mask & ((phases < phase_min) | (phases > phase_max))

            def extrapolate(flux):
                # SEDFLUX extrapolates the band integral, after clipping.
                edge = jnp.where(phases < phase_min, phase_min, phase_max)
                step = jnp.where(phases < phase_min, -3., 3.)
                inner_flux = flux_at_phase(edge - step)
                edge_flux = jnp.where(outside, flux, 1.)
                slope = jnp.where(outside, (flux - inner_flux) / step, 0.)
                return jnp.where(outside, modelflux_extrap(
                    phases, edge, edge_flux, slope), flux)

            return lax.cond(jnp.any(outside), extrapolate, lambda flux: flux, flux)
        t0, redshift = sed_params['t0'], sed_params['redshift']
        x0, x1, c = sed_params['x0'], sed_params['x1'], sed_params['c']
        fi = filter_indices

        reddened = self._host_reddened_weights(sed_params, band_weights,
                                              static_data)
        if sed_factor is not None:
            reddened = reddened * sed_factor
        if band_weights_dz is not None and reddened is not band_weights:
            # the factor is z-independent, so it scales the tangent too
            band_weights_dz = band_weights_dz * (reddened / band_weights)
        band_weights = reddened

        phases = jnp.where(mask, (mjds - t0) / (1.0 + redshift), 0.0)
        m0_val, m1_val, boundary = self._band_integrate(
            c, band_weights, phases, fi, redshift, band_weights_dz)
        latetime = (self._latetime_band_integrate(
                        c, band_weights, phases, fi, static_data)
                    if self.latetime_tau1 is not None else None)
        return self._combine_flux(x0, x1, m0_val, m1_val, boundary, phases,
                                  mask, fi, static_data, redshift,
                                  latetime=latetime)

    def _latetime_band_integrate(self, c, band_weights, phases,
                                 filter_indices, static_data):
        """The DAYMIN surfaces integrated through the colour-weighted filters
        with the FLAM decay applied per wavelength.

        SALT2_EXTRAP_LATETIME decays each wavelength on its own timescale, so
        unlike M0/M1 this cannot be integrated once and interpolated in phase.
        It still needs no SED: the surfaces are fixed at DAYMIN and only the
        decay moves, so each observation is one windowed dot product.
        """
        width = static_data['band_window'] or band_weights.shape[-1]
        obs_lo = band_window_starts(band_weights, filter_indices, width)
        cc = jnp.exp((c - self.color_offset) * self.colorlaw_basis)
        m0_cc, m1_cc = self.m0_at_daymin * cc, self.m1_at_daymin * cc
        obs_band_weights = band_weights[filter_indices]

        def one_obs(phase, lo, obs_bw):
            cut = lambda a: lax.dynamic_slice(a, (lo,), (width,))
            weights = cut(obs_bw) * LatetimeExtrapIa.decay_factor(
                phase, self.latetime_daymin, cut(self.latetime_tau1),
                cut(self.latetime_tau2), cut(self.latetime_expratio))
            return jnp.sum(cut(m0_cc) * weights), jnp.sum(cut(m1_cc) * weights)

        return vmap(one_obs)(phases, obs_lo, obs_band_weights)

    def compute_raw_fracerr(self, sed_params, mjds, filter_indices, mask,
                             band_weights, static_data, error_overrides=None,
                             error_redshift=None, band_weights_err=None,
                             band_weights_err_dz=None, **kwargs):
        """The fractional error snake.

        ``error_redshift`` sets the phases and rest wavelengths the surface is
        read at, where those differ from the trial redshift (SALT2zz,
        snlc_fit.F90:13776). ``band_weights_err`` integrates the error
        denominators (Finteg_forErr, genmag_SALT2.c:2899).
        """
        t0 = sed_params['t0']
        redshift = sed_params['redshift']
        x1, c = sed_params['x1'], sed_params['c']
        fi = filter_indices
        phases = jnp.where(mask, (mjds - t0) / (1.0 + redshift), 0.0)
        z_err = redshift if error_redshift is None else error_redshift
        phases_err = jnp.where(mask, (mjds - t0) / (1.0 + z_err), 0.0)
        rest_wave = jnp.where(
            mask, static_data['filter_mean_wavelengths'][fi] / (1.0 + z_err),
            0.0)
        bw_err = band_weights if band_weights_err is None else band_weights_err
        m0_err, m1_err, _ = self._band_integrate(
            c, bw_err, self._training_phases(phases), fi, redshift,
            band_weights_err_dz)
        frac = self._compute_model_fractional_error(
            phases_err, rest_wave, (error_overrides or {}).get('x1', x1), x1,
            jnp.where(mask, m0_err, 1.0), jnp.where(mask, m1_err, 0.0),
            static_data['filter_mean_wavelengths'][fi])
        return jnp.where(mask, frac, 0.0)

    def _training_phases(self, phases):
        """The phase the error's training flux is integrated at: the flux's
        own inside the SED's range, and three days inside the edge past it,
        where genmag_SALT2 measures its extrapolation slope with a second
        integration that overwrites Finteg_errPar (genmag_SALT2.c:2235-2240)."""
        lo, hi = self.phases[0], self.phases[-1]
        return jnp.where(phases < lo, lo + 3.0,
                         jnp.where(phases > hi, hi - 3.0, phases))

    def _combine_flux(self, x0, x1, m0_val, m1_val, boundary, phases, mask,
                      fi, static_data, redshift, latetime=None):
        """The cheap half: linear in shape, so a shape grid reuses the
        integration above and only repeats this."""
        main_flux = x0 * (m0_val + x1 * m1_val)
        phase_min, phase_max = self.phases[0], self.phases[-1]
        outside_low = phases < phase_min
        outside = outside_low | (phases > phase_max)

        if latetime is not None:
            # FLAM replaces the boundary extrapolation, as in generate_salt_sed
            m0_late, m1_late = latetime
            flux_raw = jnp.where(phases > self.latetime_daymin,
                                 x0 * (m0_late + x1 * m1_late), main_flux)
            flux_raw = jnp.where(outside_low, 0.0, flux_raw)
            return self._to_fluxcal(flux_raw, mask, fi, static_data, redshift)

        edge_flux = x0 * (boundary[:, 0] + x1 * boundary[:, 1])
        inner_flux = x0 * (boundary[:, 2] + x1 * boundary[:, 3])
        delta_phase = jnp.where(outside_low, -3.0, 3.0)
        slope = jnp.where(outside, (edge_flux - inner_flux) / delta_phase, 0.0)
        # constants where the extrapolation is discarded: fed a near-zero
        # model flux, its derivatives overflow float32 and the Hessian goes NaN
        safe_phase = jnp.where(outside_low, phase_min, phase_max)
        safe_flux = jnp.where(outside, edge_flux, 1.0)
        flux_raw = jnp.where(
            outside, modelflux_extrap(phases, safe_phase, safe_flux, slope),
            main_flux)
        return self._to_fluxcal(flux_raw, mask, fi, static_data, redshift)

    def _to_fluxcal(self, flux_raw, mask, fi, static_data, redshift):
        """Model flux as FLUXCAL, with RESTLAM_FORCEZEROFLUX applied
        (genmag_SALT2.c:2252-2255)."""
        zero_lo, zero_hi = self.restlam_forcezeroflux
        if zero_hi > 0.0:
            rest_mean = static_data['filter_mean_wavelengths'][fi] / (1.0 + redshift)
            flux_raw = jnp.where(
                (rest_mean > zero_lo) & (rest_mean < zero_hi), 0.0, flux_raw)

        zp = static_data['zp_fluxcal']
        scale = static_data['fluxcal_scale_mag_offset']
        fluxcal = (flux_raw * X0SCALE_SALT2
                   / static_data['filter_ref_fluxes'][fi]
                   * jnp.power(10.0, 0.4 * zp) * scale)
        return jnp.where(mask, fluxcal, 0.0)

    def evaluate_redshift_grid(self, axes, base_params, mu_grid, z_grid, band_weights,
                      band_weights_err, mjds, filter_indices, mask,
                      static_data, error_overrides=None):
        """The (z, colour, shape, PEAKMJD) grid with each band integral
        computed once.

        The band integral depends on redshift and colour only, and PEAKMJD
        only moves the phases it is read at, so each (z, colour) is integrated
        once and read once per PEAKMJD; shape enters only where the flux is
        linear in it. Every grid point then combines its own entries, in the
        order ``axes`` gives. Any other axis falls back to visiting every
        point.
        """
        if not set(axes) <= {'c', 'x1', 'PKMJD'}:
            return super().evaluate_redshift_grid(
                axes, base_params, mu_grid, z_grid, band_weights, band_weights_err,
                mjds, filter_indices, mask, static_data, error_overrides)
        names = list(self.get_fit_parameters())

        def values(name):
            """The axis's values, or the event's own where the grid has no
            such axis."""
            if name in axes:
                return jnp.asarray(axes[name])
            return base_params[names.index(name)][None]

        c_vals, s_vals, t_vals = values('c'), values('x1'), values('PKMJD')
        n_z, n_c, n_t = z_grid.shape[0], c_vals.shape[0], t_vals.shape[0]
        fi = filter_indices
        # [n_z, n_t, n_obs]
        phases = jnp.where(mask, (mjds - t_vals[None, :, None])
                           / (1.0 + z_grid[:, None, None]), 0.0)

        cc = jnp.exp((c_vals[:, None] - self.color_offset)
                     * self.colorlaw_basis[None, :])
        CLW = band_weights[:, None, :, :] * cc[None, :, None, :]
        M0 = jnp.einsum('pw,zcfw->zcpf', self.m0, CLW)
        M1 = jnp.einsum('pw,zcfw->zcpf', self.m1, CLW)
        # b = phase side (low/high), k = component (edge/inner x M0/M1)
        bnd = jnp.einsum('bkw,zqfw->zqbkf', self.extrap_boundary_seds, CLW)
        CLW_err = band_weights_err[:, None, :, :] * cc[None, :, None, :]
        M0e = jnp.einsum('pw,zcfw->zcpf', self.m0, CLW_err)
        M1e = jnp.einsum('pw,zcfw->zcpf', self.m1, CLW_err)

        def interp(surface, at):
            return hermite_interp_1d_select(
                at, self.phases, surface,
                compute_catmullrom_slopes(self.phases, surface), fi)

        def read(iz, ic, it):
            """The integrated surfaces at the phases of one (z, colour, PEAKMJD)."""
            at = phases[iz, it]
            train = self._training_phases(at)
            side = jnp.where(at < self.phases[0], 0, 1)
            return (interp(M0[iz, ic], at), interp(M1[iz, ic], at),
                    interp(M0e[iz, ic], train), interp(M1e[iz, ic], train),
                    bnd[iz, ic][side, :, fi])

        zct = [g.ravel() for g in jnp.meshgrid(
            jnp.arange(n_z), jnp.arange(n_c), jnp.arange(n_t), indexing='ij')]
        surfaces = [a.reshape(n_z, n_c, n_t, *a.shape[1:]) for a in vmap(read)(*zct)]

        def point(iz, ic, i_s, it):
            m0v, m1v, m0e, m1e, b = (a[iz, ic, it] for a in surfaces)
            at, s = phases[iz, it], s_vals[i_s]
            x0 = compute_expected_x0_scale(mu_grid[iz], self.salt2alpha,
                                           self.salt2beta, s, c_vals[ic])
            rest = jnp.where(
                mask,
                static_data['filter_mean_wavelengths'][fi] / (1.0 + z_grid[iz]),
                0.0)
            flux = self._combine_flux(x0, s, m0v, m1v, b, at, mask, fi,
                                      static_data, z_grid[iz])
            frac = jnp.where(mask, self._compute_model_fractional_error(
                at, rest, (error_overrides or {}).get('x1', s), s,
                jnp.where(mask, m0e, 1.0), jnp.where(mask, m1e, 0.0),
                static_data['filter_mean_wavelengths'][fi]), 0.0)
            return flux, jnp.where(mask, flux_error_from_fracerr(flux, frac), 0.0)

        # every point, in the order of the caller's axes
        lengths = [len(axis) for axis in axes.values()]
        grid = [g.ravel() for g in jnp.meshgrid(
            jnp.arange(n_z), *map(jnp.arange, lengths), indexing='ij')]
        index = dict(zip(axes, grid[1:]))
        # an axis the grid leaves out has only its one value
        only = jnp.zeros_like(grid[0])
        flux, flux_err = vmap(point)(grid[0], index.get('c', only),
                                     index.get('x1', only), index.get('PKMJD', only))
        shape = (n_z, *lengths, -1)
        return flux.reshape(shape), flux_err.reshape(shape)

    def _amplitude_shifted(self, value, dmu):
        """x0 scales by 10^(-0.4dmu) (snlc_fit.F90:15210), whichever of x0 and
        log10(x0) the fit vector holds."""
        return self._x0_to_amplitude(
            self._amplitude_to_x0(value) * jnp.power(10.0, -0.4 * dmu))

    def color_dispersion_fraction(self, filter_mean_wavelengths, filter_indices,
                                  redshifts, masks):
        """The colour dispersion at each observation's rest-frame filter mean,
        the per-observation term the Woodbury covariance carries."""
        rest_wave = (filter_mean_wavelengths[filter_indices]
                     / (1.0 + redshifts[:, None]))
        fraction = hermite_interp_1d(
            rest_wave, self.error_model['color_dispersion_wavelengths'],
            self.error_model['color_dispersion'],
            self.error_model['color_dispersion_slopes'])
        return jnp.where(masks, fraction, 0.0)

    def fitres_param_columns(self, param_names, physical_only=False):
        """SNANA column order PKMJD x1 c mB x0 [zPHOT]; mB is derived from the
        amplitude. SALT has no nuisance params, so ``physical_only`` has no
        effect."""
        amp = self.distance_param
        order = ['PKMJD', 'x1', 'c', amp, 'zPHOT']
        cols = [(n, n) for n in order if n in param_names]
        cols.insert(3, (amp, 'mB'))
        return cols

    def get_fitres_param_columns(self, params, param_errors, redshifts, mu_ref,
                                 physical_only=False):
        """The columns of :meth:`fitres_param_columns`, with
        mB = SALT2mBcalc (genmag_SALT2.c:3031)."""
        amp = self.distance_param
        val = np.asarray(params[amp])
        err = np.asarray(param_errors[amp])
        x0 = np.asarray(self._amplitude_to_x0(val))
        derived = {'mB': (SALT2_MB_OFFSET - 2.5 * np.log10(x0),
                          np.abs(self.amplitude_jacobian(val)) * err)}
        return [(column, *derived.get(column, (params[name], param_errors[name])))
                for name, column in self.fitres_param_columns(list(params))]

    def get_fitres_derived_columns(self, params):
        """RADIUS_POP = sqrt((x1/3)^2 + (c/0.3)^2) (RADIUS_CALC,
        snlc_fit.F90:14444)."""
        x1 = np.asarray(params['x1'])
        c = np.asarray(params['c'])
        return [('RADIUS_POP', np.sqrt((x1 / 3.0)**2 + (c / 0.3)**2))]

    # BBC: Tripp standardisation
    m0_reference = -30.0
    lcpar_names = ("mB", "x1", "c")

    # COV columns carry both the jax writer's and SNANA's spellings as fallbacks
    fitres_aliases = {
        "x0": ("x0",),
        "x0err": ("x0ERR", "x0err"),
        "x1": ("x1",),
        "x1err": ("x1ERR", "x1err"),
        "c": ("c",),
        "cerr": ("cERR", "cerr"),
        "COV_x1_x0": ("COV_x1_x0", "COV_x0x1", "COV_x0_x1"),
        "COV_x1_c": ("COV_x1_c", "COV_x1c"),
        "COV_c_x0": ("COV_c_x0", "COV_x0c", "COV_x0_c"),
    }

    # bounds from prep_fitpar (SALT2mu.c:22354-22369)
    nuisance_spec = {
        "alpha0": (0.14, 0.02, 0.30, 1),
        "beta0": (3.2, 1.0, 6.0, 2),
        "dalpha_dz": (0.0, -0.5, 0.5, 3),
        "dbeta_dz": (0.0, -3.0, 3.0, 4),
        "gamma0": (0.0, -0.5, 0.5, 5),
        "dgamma_dz": (0.0, -2.0, 2.0, 6),
        "logmass_cen": (10.0, 9.0, 11.0, 7),
        "logmass_tau": (0.01, 8.0e-4, 1.0, 8),
        "dalpha_dlogm": (0.0, -0.5, 0.5, 15),
        "dbeta_dlogm": (0.0, -3.0, 3.0, 16),
    }
    mucovscale_bins = {"c": 3}
    ccprior_bins = {"c": 1, "x1": 1}
    bbc_input_aliases = {
        "nbinc_mucovscale": ("mucovscale_bins", "c"),
        "nsbin_ccprior": ("ccprior_bins", "x1"),
        "ncbin_ccprior": ("ccprior_bins", "c"),
    }
    bbc_cut_bits = {"x1": (1, 14), "c": (2, 15)}
    biascor_binning = {"x1": (-4.0, 4.0, 0.5),
                       "c": (-0.30, 0.50, 0.05)}          # SALT2mu.c:547-549
    biascor_coeffs = (("alpha0", "SIM_alpha", False),
                      ("beta0", "SIM_beta", False),
                      ("gamma0", "SIM_gammaDM", True))
    gammadm_coeff = "gamma0"
    dust_forced_coeffs = ("alpha0", "beta0")
    biascor_truth_aliases = {
        "SIM_x0": ("SIM_x0", "SIM_SALT2x0"),
        "SIM_x1": ("SIM_x1", "SIM_SALT2x1"),
        "SIM_c": ("SIM_c", "SIM_SALT2c"),
        "SIM_alpha": ("SIM_alpha", "SIM_SALT2alpha"),
        "SIM_beta": ("SIM_beta", "SIM_SALT2beta"),
        "SIM_gammaDM": ("SIM_gammaDM", "SIM_SALT2gammaDM"),
    }
    cut_aliases = {
        "x1min": ("x1", "lo"), "x1max": ("x1", "hi"),
        "cmin": ("c", "lo"), "cmax": ("c", "hi"),
        "logmass_min": ("logmass", "lo"), "logmass_max": ("logmass", "hi"),
    }
    default_cuts = {"x1": (-3.0, 3.0), "c": (-0.3, 0.3)}

    #: Reject an event whose fitted x0 error is at or below this value; the
    #: default 0.0 rejects only a non-positive error.
    maxerr_abort_x0: Annotated[float, BBC_ONLY] = 0.0
    #: Reject an event whose fitted x1 error is at or below this value; the
    #: default 0.0 rejects only a non-positive error.
    maxerr_abort_x1: Annotated[float, BBC_ONLY] = 0.0
    #: Reject an event whose fitted colour error is at or below this value;
    #: the default 0.0 rejects only a non-positive error.
    maxerr_abort_c: Annotated[float, BBC_ONLY] = 0.0

    # The CC-simulation map uses alpha0 and beta0 for every event, even when
    # the data fit includes evolution (SALT2mu.c: fcn_ccprior_muzmap).
    ccprior_fixed_nuisance = {
        "dalpha_dz": 0.0, "dbeta_dz": 0.0,
        "dalpha_dlogm": 0.0, "dbeta_dlogm": 0.0,
    }

    @staticmethod
    def _mass_step_pars(host):
        """``(GAMMADM_OFFSET, USE_GAMMA0)`` for these events, both from BBC.

        The offset is the midpoint of the biasCor SIM_gammaDM range
        (SALT2mu.c:16228), and zero where there is no such axis. USE_GAMMA0 is
        SNANA's flag for the mass step being floated or started non-zero
        (prep_input_gamma, SALT2mu.c:22820); with it off get_gammadm_host
        returns zero before reading anything (SALT2mu.c:16776).
        """
        missing = [k for k in ("gammadm_offset", "use_gamma0") if k not in host]
        if missing:
            raise SNANAError(
                f"host has no {missing}: the caller must supply the midpoint of the "
                f"biasCor SIM_gammaDM range that get_gammadm_host subtracts, or 0 "
                f"where there is no biasCor, and SNANA's USE_GAMMA0 flag.",
                function_name="compute_mu")
        return host["gammadm_offset"], host["use_gamma0"]

    @staticmethod
    def _alpha_beta(nuis, host, u_flags):
        """Per-event effective alpha/beta (fcnFetch_AlphaBetaGamma). Host-mass
        dependence: slope and split are independent flags in SNANA
        (u15<=1 / u16<=1 slope, ==2 split); both can fire."""
        u15 = u_flags.get("u15", 0)
        u16 = u_flags.get("u16", 0)
        z = host["z"]
        dlogmass = host["gamma_var"] - nuis["logmass_cen"]
        alpha = nuis["alpha0"] + z * nuis["dalpha_dz"]
        beta = nuis["beta0"] + z * nuis["dbeta_dz"]
        if (u15 <= 1) or (u16 <= 1):
            alpha = alpha + nuis["dalpha_dlogm"] * dlogmass
            beta = beta + nuis["dbeta_dlogm"] * dlogmass
        if (u15 == 2) or (u16 == 2):
            half = jnp.where(dlogmass > 0.0, 0.5, -0.5)
            alpha = alpha + nuis["dalpha_dlogm"] * half
            beta = beta + nuis["dbeta_dlogm"] * half
        return alpha, beta

    @classmethod
    def compute_mu(cls, lcpar, nuis, host, u_flags, apply_mass_step=True):
        alpha, beta = cls._alpha_beta(nuis, host, u_flags)
        mu = lcpar["mB"] + alpha * lcpar["x1"] - beta * lcpar["c"]
        if apply_mass_step:
            mu = mu - gammadm(
                nuis["gamma0"], nuis["dgamma_dz"], nuis["logmass_cen"],
                nuis["logmass_tau"], host["z"], host["gamma_var"],
                *cls._mass_step_pars(host),
            )
        return mu

    @classmethod
    def biascor_coeff_values(cls, nuis, host, u_flags):
        alpha, beta = cls._alpha_beta(nuis, host, u_flags)
        gdm = gammadm(nuis["gamma0"], nuis["dgamma_dz"], nuis["logmass_cen"],
                      nuis["logmass_tau"], host["z"], host["gamma_var"],
                      *cls._mass_step_pars(host))
        return (alpha, beta, gdm)

    @classmethod
    def read_lcpar(cls, cols, inputs):
        x0 = cols["x0"]
        lcpar = {"mB": -2.5 * jnp.log10(x0), "x1": cols["x1"], "c": cols["c"]}
        sf = -2.5 / (x0 * LN10)
        n = x0.shape[0]
        cov = jnp.zeros((n, 3, 3))
        dd = (cols["x0err"] * sf) ** 2
        ds = cols["COV_x1_x0"] * sf
        dc = cols["COV_c_x0"] * sf
        ss = cols["x1err"] ** 2
        sc = cols["COV_x1_c"]
        cc = cols["cerr"] ** 2
        cov = cov.at[:, 0, 0].set(dd)
        cov = cov.at[:, 1, 1].set(ss)
        cov = cov.at[:, 2, 2].set(cc)
        cov = cov.at[:, 0, 1].set(ds).at[:, 1, 0].set(ds)
        cov = cov.at[:, 0, 2].set(dc).at[:, 2, 0].set(dc)
        cov = cov.at[:, 1, 2].set(sc).at[:, 2, 1].set(sc)
        bad_error = ((cols["x0err"] <= inputs["maxerr_abort_x0"])
                     | (cols["x1err"] <= inputs["maxerr_abort_x1"])
                     | (cols["cerr"] <= inputs["maxerr_abort_c"]))
        return lcpar, cov, bad_error

    @classmethod
    def bbc_bad_covariance(cls, cols, covariance):
        # SALT2mu's BADCOV cut uses repaired off-diagonal covariances,
        # but the original mB uncertainty derived from x0 and x0err.
        mberr = np.abs(-2.5 * cols["x0err"] / (cols["x0"] * LN10))
        return ((mberr > 1.0)
                | (np.abs(covariance[:, 0, 1]) > 8.0)
                | (np.abs(covariance[:, 0, 2]) > 8.0)
                | (np.abs(covariance[:, 1, 2]) > 8.0))

    @classmethod
    def sim_lcpar(cls, tv):
        # true mB includes SIM_gammaDM
        sim = tv.sim_truth
        return {"mB": -2.5 * np.log10(sim["SIM_x0"]) + sim["SIM_gammaDM"],
                "x1": sim["SIM_x1"], "c": sim["SIM_c"]}
