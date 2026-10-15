"""SN rate models and rate-integral physics.

DNDZ rate functions (POWERLAW, POWERLAW2, CC_S15, MD14), per-z rate
evaluation, and the SOLID_ANGLE × time × ∫R(z) dV(z) integral that
resolves NGEN_SEASON inputs. Pure config-driven; no `LightCurveSimulator` or
sampler state.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core.io.two_column import read_two_column_file
from jax_snlc_core.physics.constants import LIGHT_KM
from jax_snlc_core.physics.cosmology import (
    hubble_parameter, transverse_comoving_distance,
)
from jax_snlc_core.utils.error_handling import SNANAError
from jax_snlc_core.utils.interp import eval_genpoly

logger = logging.getLogger(__name__)

PC_KM = 3.085678e13   # parsec in km (sntools.h:70)


def _SFRfun_MD14(z: np.ndarray, A: float, B: float, C: float, D: float) -> np.ndarray:
    """Madau & Dickinson 2014 star formation rate function.

    From sntools_cosmology.c:263-281.
    """
    z1 = 1.0 + z
    return A * z1**C / (1.0 + (z1 / B)**D)


def _parse_dndz_multiplier(name_str: str) -> Tuple[float, str]:
    """Parse optional multiplier from DNDZ model name.

    Handles: '0.7*CC_S15', 'CC_S15*0.7', 'CC_S15' (scale=1.0).
    Matches sntools.c:2999 parse_multiplier.
    """
    if '*' in name_str:
        parts = name_str.split('*')
        try:
            scale = float(parts[0])
            model_name = parts[1]
        except ValueError:
            model_name = parts[0]
            scale = float(parts[1])
        return scale, model_name
    return 1.0, name_str


def _rate_powerlaw(dndz_param: Dict, z_grid: np.ndarray) -> np.ndarray:
    """R(z) = A * (1+z)^B, used by both POWERLAW and POWERLAW2."""
    return dndz_param['A'] * (1.0 + z_grid) ** dndz_param['B']


def _rate_cc_s15(dndz_param: Dict, z_grid: np.ndarray) -> np.ndarray:
    """Strolger 2015 CC rate: k * h^2 * SFR_MD14(z) * user_scale
    (SNANA snlc_sim.c:18009-18020)."""
    A_md, B_md, C_md, D_md = 0.015, 1.50, 5.00, 6.10
    k = 0.0091
    h = dndz_param['h0'] / 100.0
    scale = dndz_param['scale']
    return scale * k * h**2 * _SFRfun_MD14(z_grid, A_md, B_md, C_md, D_md)


def _rate_from_file(dndz_param: Dict, z_grid: np.ndarray) -> np.ndarray:
    """DNDZ_FILE rate table, interpolated linearly (interp_1DFUN OPT_INTERP=1).

    SNANA's quickBinSearch aborts when z falls outside the table, so a
    generation range the table does not cover is an error, not an
    extrapolation.
    """
    z_table, rate_table = dndz_param['z_table'], dndz_param['rate_table']
    if z_grid.min() < z_table[0] or z_grid.max() > z_table[-1]:
        raise SNANAError(
            f"DNDZ_FILE {dndz_param['file']} covers z={z_table[0]} to "
            f"{z_table[-1]}, but the rate is evaluated over "
            f"z={z_grid.min():.4f} to {z_grid.max():.4f}.",
            function_name="_rate_from_file")
    return np.interp(z_grid, z_table, rate_table)


def _rate_md14(dndz_param: Dict, z_grid: np.ndarray) -> np.ndarray:
    """Madau & Dickinson 2014 SFR(z), normalised by R0 at z=0
    (SNANA snlc_sim.c:18021-18027)."""
    A_md, B_md, C_md, D_md = 0.015, 2.90, 2.70, 5.60
    R0 = dndz_param['R0']
    sfr0 = _SFRfun_MD14(np.array([0.0]), A_md, B_md, C_md, D_md)[0]
    return R0 * _SFRfun_MD14(z_grid, A_md, B_md, C_md, D_md) / sfr0


def _rate_constant(dndz_param: Dict, z_grid: np.ndarray) -> np.ndarray:
    """HUBBLE (3e-5, "approx SNIa rate at z ~ 0") and FLAT (1e-8, which SNANA
    calls meaningless, and exists only to avoid an abort), snlc_sim.c:18003."""
    return np.full_like(z_grid, dndz_param['rate'], dtype=np.float64)


def _rate_pisn(dndz_param: Dict, z_grid: np.ndarray) -> np.ndarray:
    """Pair-instability SN rate polynomial in Gpc^-3 (snlc_sim.c:18030)."""
    z = z_grid
    rate = (1.98 + 6.38 * z + 6.558 * z**2
            - 4.42 * z**3 + 0.8312 * z**4 - 0.0508 * z**5)
    return dndz_param['scale'] * rate / 1.0e9


def _rate_tde(dndz_param: Dict, z_grid: np.ndarray) -> np.ndarray:
    """Tidal-disruption-event rate R0 * 10^(-0.5z/0.6) (snlc_sim.c:18041)."""
    return dndz_param['R0'] * 10.0 ** (-0.5 * z_grid / 0.6)


def _rate_zpoly(dndz_param: Dict, z_grid: np.ndarray) -> np.ndarray:
    """Rate as a polynomial in z (snlc_sim.c:18069, eval_GENPOLY)."""
    return eval_genpoly(dndz_param['coeffs'], z_grid)


def _sfr_bg03(z: np.ndarray, h0: float) -> np.ndarray:
    """Baldry & Glazebrook 2003 star-formation rate (sntools_cosmology.c:SFRfun_BG03)."""
    a, b, c, d = 0.0118, 0.08, 3.3, 5.2
    return (h0 / 100.0) * (a + b * z) / (1.0 + (z / c) ** d)


def _sfr_integral(z_grid: np.ndarray, cosmology) -> np.ndarray:
    """Stellar mass formed by each redshift, integrated over the scale factor
    with SNANA's 100-bin midpoint rule (sntools_cosmology.c:SFR_integral)."""
    n_bins = 100
    z_grid = np.asarray(z_grid, dtype=np.float64)
    a_max = 1.0 / (1.0 + z_grid)
    a_bin = a_max / n_bins
    a = a_bin[:, None] * (np.arange(1, n_bins + 1) - 0.5)[None, :]
    z_a = 1.0 / a - 1.0
    omega_k = 1.0 - cosmology.omega_matter - cosmology.omega_lambda
    ez = np.asarray(hubble_parameter(
        jnp.asarray(z_a), cosmology.omega_matter, omega_k,
        cosmology.omega_lambda, cosmology.w0, cosmology.wa))
    h0 = cosmology.h0
    total = np.sum(_sfr_bg03(z_a, h0) / (a * h0 * ez), axis=1)
    seconds_per_year = 3600.0 * 24.0 * 365.0
    return total * a_bin * (1.0e6 * PC_KM) / seconds_per_year


def _rate_abmodel(dndz_param: Dict, z_grid: np.ndarray) -> np.ndarray:
    """Two-component A+B rate: A * (stellar mass) + B * SFR (snlc_sim.c:18043).

    SNANA's prompt term is inert: it passes (H0, z) to a function declared
    (z, H0) (snlc_sim.c:18051), collapsing B's contribution to ~1e-13. jax
    passes them the right way round.
    """
    A, B, cosmology = dndz_param['A'], dndz_param['B'], dndz_param['cosmology']
    sfr_int = _sfr_integral(z_grid, cosmology)
    return A * sfr_int + B * _sfr_bg03(z_grid, cosmology.h0)


_DNDZ_RATE_MODELS = {
    'POWERLAW':  _rate_powerlaw,
    'POWERLAW2': _rate_powerlaw,
    'CC_S15':    _rate_cc_s15,
    'MD14':      _rate_md14,
    'FILE':      _rate_from_file,
    'HUBBLE':    _rate_constant,
    'FLAT':      _rate_constant,
    'PISN_PLK12': _rate_pisn,
    'TDE':       _rate_tde,
    'ZPOLY':     _rate_zpoly,
    'ABMODEL':   _rate_abmodel,
}

# sntools.h:76. DNDZ_FILE declares the full SNANA redshift range
# (snlc_sim.c:2915); the table itself bounds what can be evaluated.
ZMAX_SNANA = 10.0


def _unknown_dndz_model(model: str) -> str:
    return (f"Unknown DNDZ model: '{model}'. "
            f"Supported: {', '.join(_DNDZ_RATE_MODELS)}.")


def _compute_rate_on_grid(z_grid: np.ndarray, dndz_param: Dict) -> np.ndarray:
    """Compute volumetric rate R(z) on a redshift grid for a single DNDZ model."""
    model = dndz_param['model']
    if model not in _DNDZ_RATE_MODELS:
        raise SNANAError(_unknown_dndz_model(model),
                         function_name="_compute_rate_on_grid")
    return _DNDZ_RATE_MODELS[model](dndz_param, z_grid)


def _parse_poly_coeffs(tokens: List[str]) -> List[float]:
    """ZPOLY coefficients, comma-separated or SNANA's legacy space-separated
    form (read_genpoly, sntools.c:11586)."""
    if len(tokens) == 1 and ',' in tokens[0]:
        tokens = tokens[0].split(',')
    try:
        return [float(t) for t in tokens]
    except ValueError as e:
        raise SNANAError(
            f"Could not parse DNDZ ZPOLY coefficients {tokens}: {e}",
            function_name="_parse_poly_coeffs")


def _check_powerlaw_bounds(A: float, B: float) -> None:
    """Fatal A/B bounds shared by POWERLAW and POWERLAW2 (snlc_sim.c:18060-18066)."""
    if A > 10.0 or B > 10.0:
        raise SNANAError(
            f"Invalid parameters for powerlaw rate model: alpha={A} beta={B} "
            f"(requires A <= 10 and B <= 10).",
            function_name="parse_dndz_entries")


def parse_dndz_entries(dndz_entries: List[Any],
                       redshift_range: Tuple[float, float],
                       h0: float, cosmology=None) -> List[Dict[str, Any]]:
    """Turn the DNDZ_* entries into per-model rate parameters.

    `redshift_range` bounds the models that do not carry their own z range.
    `cosmology` is read only by ABMODEL, whose stellar-mass term integrates
    SFR(z)/H(z) over the scale factor.
    """
    dndz_params = []
    z_min, z_max = redshift_range

    for entry in dndz_entries:
        tokens = [str(t) for t in (entry if isinstance(entry, list) else [entry])]
        scale, name = _parse_dndz_multiplier(tokens[0])
        full_range = {'zmin': z_min, 'zmax': z_max}

        if name == 'POWERLAW2':
            if len(tokens) < 5:
                raise SNANAError(
                    f"DNDZ POWERLAW2 requires A B zmin zmax; got {tokens[1:]}.",
                    function_name="parse_dndz_entries")
            try:
                A, B = float(tokens[1]), float(tokens[2])
                zlo, zhi = float(tokens[3]), float(tokens[4])
            except ValueError as e:
                raise SNANAError(
                    f"Could not parse DNDZ POWERLAW2 row {tokens[1:]}: {e}",
                    function_name="parse_dndz_entries")
            _check_powerlaw_bounds(A, B)
            dndz_params.append({
                'model': 'POWERLAW2', 'A': A, 'B': B, 'zmin': zlo, 'zmax': zhi,
            })

        elif name == 'POWERLAW':
            if len(tokens) < 3:
                raise SNANAError(
                    f"DNDZ POWERLAW requires A B; got {tokens[1:]}.",
                    function_name="parse_dndz_entries")
            try:
                A, B = float(tokens[1]), float(tokens[2])
            except ValueError as e:
                raise SNANAError(
                    f"Could not parse DNDZ POWERLAW row {tokens[1:]}: {e}",
                    function_name="parse_dndz_entries")
            _check_powerlaw_bounds(A, B)
            dndz_params.append({
                'model': 'POWERLAW', 'A': A, 'B': B, **full_range,
            })

        elif name == 'FILE' and len(tokens) >= 2:
            path = Path(os.path.expandvars(tokens[1]))
            z_table, rate_table = read_two_column_file(path)
            dndz_params.append({
                'model': 'FILE', 'file': path,
                'z_table': z_table, 'rate_table': rate_table,
                'zmin': 0.0, 'zmax': ZMAX_SNANA,
            })

        elif name == 'MD14':
            if len(tokens) < 2:
                raise SNANAError(
                    f"DNDZ MD14 requires R0; got {tokens[1:]}.",
                    function_name="parse_dndz_entries")
            try:
                R0 = float(tokens[1])
            except ValueError as e:
                raise SNANAError(
                    f"Could not parse DNDZ MD14 row {tokens[1:]}: {e}",
                    function_name="parse_dndz_entries")
            dndz_params.append({
                'model': 'MD14', 'R0': R0, 'h0': h0, **full_range,
            })

        elif name == 'CC_S15':
            dndz_params.append({'model': 'CC_S15', 'scale': scale,
                                'h0': h0, **full_range})

        elif name == 'PISN_PLK12':
            dndz_params.append({'model': 'PISN_PLK12', 'scale': scale,
                                **full_range})

        elif name in ('HUBBLE', 'FLAT'):
            rate = 3.0e-5 if name == 'HUBBLE' else 1.0e-8
            dndz_params.append({'model': name, 'rate': rate, **full_range})

        elif name == 'TDE':
            if len(tokens) < 2:
                raise SNANAError(
                    f"DNDZ TDE requires R0; got {tokens[1:]}.",
                    function_name="parse_dndz_entries")
            dndz_params.append({'model': 'TDE', 'R0': float(tokens[1]),
                                **full_range})

        elif name == 'ZPOLY':
            dndz_params.append({'model': 'ZPOLY',
                                'coeffs': _parse_poly_coeffs(tokens[1:]),
                                **full_range})

        elif name == 'ABMODEL':
            A, B = float(tokens[1]), float(tokens[2])
            if A > 1.0E-13 or B > 0.01:
                raise SNANAError(
                    f"Invalid parameters for A+B rate model: A={A} B={B}.",
                    function_name="parse_dndz_entries")
            dndz_params.append({'model': 'ABMODEL', 'A': A, 'B': B,
                                'cosmology': cosmology, **full_range})

        else:
            raise SNANAError(_unknown_dndz_model(tokens[0]),
                             function_name="parse_dndz_entries")

    return dndz_params


def apply_rate_modifiers(rate: np.ndarray, z: np.ndarray,
                         mods: Dict) -> np.ndarray:
    """DNDZ_* reweights and scales, in SNANA's genz_wgt order (snlc_sim.c:17826).

    `mods` carries `scale` (DNDZ_ALLSCALE x the DNDZ_SCALE slot for this model
    type, resolved at parse time), `zexp` (DNDZ_ZEXP_REWGT) and the two REWGT
    polynomials. Absent polynomials evaluate to the constant 1, as SNANA's
    init_RATEPAR seeds them.
    """
    if mods['zexp']:
        rate = rate * np.power(z, mods['zexp'])
    for coeffs, var in ((mods['zpoly'], z), (mods['z1poly'], 1.0 + z)):
        if coeffs:
            rate = rate * eval_genpoly(coeffs, var)
    return rate * mods['scale']


def _calculate_sn_integrand(z_array: np.ndarray, dndz_param: Dict,
                            omega_m: float, omega_k: float, omega_l: float,
                            hubble_distance: float,
                            w0: float, wa: float,
                            hz_map, rate_mods) -> np.ndarray:
    """SN rate integrand `rate(z) × dV/dz / (1+z)` for any DNDZ model."""
    rate = apply_rate_modifiers(
        _compute_rate_on_grid(z_array, dndz_param), z_array, rate_mods)

    z_jax = jnp.array(z_array)
    dm = jax.vmap(lambda z: transverse_comoving_distance(
        z, omega_m, omega_k, omega_l, hubble_distance, w0, wa, hz_map))(z_jax)
    ez = jax.vmap(lambda z: hubble_parameter(
        z, omega_m, omega_k, omega_l, w0, wa, hz_map))(z_jax)
    dv_dz = np.array(hubble_distance * dm**2 / ez)

    return rate * dv_dz / (1.0 + z_array)


def dndz_density_on_grid(dndz_params: List[Dict], z_grid: np.ndarray,
                         omega_m: float, omega_k: float, omega_l: float,
                         hubble_distance: float, w0: float, wa: float,
                         hz_map, rate_mods) -> np.ndarray:
    """Build dN/dz density on `z_grid` summed over all dndz_params entries."""
    density = np.zeros_like(z_grid, dtype=np.float64)
    for params in dndz_params:
        # inclusive zmax, where the caller's linspace ends (cf. snlc_sim.c:17977)
        mask = (z_grid >= params['zmin']) & (z_grid <= params['zmax'])
        if np.any(mask):
            density[mask] += _calculate_sn_integrand(
                z_grid[mask], params,
                omega_m, omega_k, omega_l, hubble_distance, w0, wa, hz_map,
                rate_mods)
    return density


def calculate_expected_sn_count(dndz_params: List[Dict],
                                solid_angle: float,
                                redshift_range: Tuple[float, float],
                                peakmjd_range: Tuple[float, float],
                                peakmjd_exclude: Tuple[float, float],
                                omega_m: float, omega_l: float,
                                h0: float, w0: float, wa: float,
                                hz_map, rate_mods) -> float:
    """Expected number of SNe per season over SOLID_ANGLE. The season is
    the peak-MJD range less the excluded window (SNcount_model,
    snlc_sim.c:17926)."""
    if not dndz_params:
        return 0.0

    if solid_angle <= 0:
        return 0.0

    z_min, z_max = redshift_range
    mjd_min, mjd_max = peakmjd_range
    exclude_min, exclude_max = peakmjd_exclude
    t_year = ((mjd_max - mjd_min) - (exclude_max - exclude_min)) / 365.0

    omega_k = 1.0 - omega_m - omega_l
    hubble_distance = LIGHT_KM / h0

    n_bins = max(10, int((z_max - z_min) * 1000.0))
    dz = (z_max - z_min) / n_bins
    z_centers = np.linspace(z_min + dz/2, z_max - dz/2, n_bins)

    total_sn_sum = 0.0

    for params in dndz_params:
        mask = (z_centers >= params['zmin']) & (z_centers < params['zmax'])
        z_valid = z_centers[mask]

        if len(z_valid) > 0:
            integrands = _calculate_sn_integrand(
                z_valid, params, omega_m, omega_k, omega_l, hubble_distance, w0, wa,
                hz_map=hz_map,
                rate_mods=rate_mods,
            )
            total_sn_sum += float(np.sum(integrands))

    sn_expected = total_sn_sum * solid_angle * t_year * dz
    return sn_expected
