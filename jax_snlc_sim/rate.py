"""SN rate models, generated counts, and redshift sampling distributions.

DNDZ rate functions (POWERLAW, POWERLAW2, CC_S15, MD14), per-z rate
evaluation, and the SOLID_ANGLE × time × ∫R(z) dV(z) integral that
resolves NGEN_SEASON inputs. Pure config-driven; no `LightCurveSimulator` or
sampler state.
"""

import logging
from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core.cosmology import (
    ANISOTROPY_V04_PARS, cos_sep_cmb_apex_galactic, cosmo_params,
    distance_modulus, distance_modulus_to_redshift, hubble_parameter,
    transverse_comoving_distance)
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.models import MODEL_REGISTRY
from jax_snlc_core.snana_files import read_two_column_file
from jax_snlc_core.utils import LIGHT_KM, eval_genpoly

logger = logging.getLogger(__name__)

PC_KM = 3.085678e13   # parsec in km (sntools.h:70)


@dataclass
class RateModifiers:
    """Multiplicative DNDZ reweighting, neutral unless supplied."""

    scale: float = 1.0
    zexp: float = 0.0
    zpoly: List[float] = field(default_factory=list)
    z1poly: List[float] = field(default_factory=list)


def neutral_rate_mods() -> Dict[str, Any]:
    """Default DNDZ modifiers, leaving the rate unchanged."""
    return asdict(RateModifiers())


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


# sntools.h:76. DNDZ_FILE declares the full SNANA redshift range
# (snlc_sim.c:2915); the table itself bounds what can be evaluated.
ZMAX_SNANA = 10.0


def _unknown_dndz_model(model: str) -> str:
    return (f"Unknown DNDZ model: '{model}'. Supported: POWERLAW, POWERLAW2, "
            "CC_S15, MD14, FILE, HUBBLE, FLAT, PISN_PLK12, TDE, ZPOLY, ABMODEL.")


def compute_rate_on_grid(z_grid: np.ndarray, dndz_param: Dict) -> np.ndarray:
    """Compute volumetric rate R(z) on a redshift grid for one DNDZ model."""
    model = dndz_param['model']
    if model in ('POWERLAW', 'POWERLAW2'):
        return dndz_param['A'] * (1.0 + z_grid) ** dndz_param['B']
    if model == 'CC_S15':
        # Strolger 2015: k * h^2 * SFR_MD14 (snlc_sim.c:18009-18020).
        h = dndz_param['h0'] / 100.0
        return (dndz_param['scale'] * 0.0091 * h**2
                * _SFRfun_MD14(z_grid, 0.015, 1.50, 5.00, 6.10))
    if model == 'MD14':
        # Madau & Dickinson SFR normalised to R0 at z=0.
        sfr0 = _SFRfun_MD14(np.array([0.0]), 0.015, 2.90, 2.70, 5.60)[0]
        return (dndz_param['R0']
                * _SFRfun_MD14(z_grid, 0.015, 2.90, 2.70, 5.60) / sfr0)
    if model == 'FILE':
        return _rate_from_file(dndz_param, z_grid)
    if model in ('HUBBLE', 'FLAT'):
        return np.full_like(z_grid, dndz_param['rate'], dtype=np.float64)
    if model == 'PISN_PLK12':
        # Pair-instability SN rate in Gpc^-3 (snlc_sim.c:18030).
        z = z_grid
        rate = (1.98 + 6.38 * z + 6.558 * z**2
                - 4.42 * z**3 + 0.8312 * z**4 - 0.0508 * z**5)
        return dndz_param['scale'] * rate / 1.0e9
    if model == 'TDE':
        return dndz_param['R0'] * 10.0 ** (-0.5 * z_grid / 0.6)
    if model == 'ZPOLY':
        return eval_genpoly(dndz_param['coeffs'], z_grid)
    if model == 'ABMODEL':
        cosmology = dndz_param['cosmology']
        # SNANA swaps the prompt SFR's z/H0 arguments (snlc_sim.c:18051),
        # making B nearly inert. Here both terms use the physical redshift.
        return (dndz_param['A'] * _sfr_integral(z_grid, cosmology)
                + dndz_param['B'] * _sfr_bg03(z_grid, cosmology.h0))
    raise SNANAError(_unknown_dndz_model(model),
                     function_name="compute_rate_on_grid")


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
        compute_rate_on_grid(z_array, dndz_param), z_array, rate_mods)

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


def resolve_counts(config, hz_map) -> Tuple[int, int, float, float]:
    """Resolve how many events to generate and the PEC1A share of them.

    SNANA supports three modes (snlc_sim.c:7146-9221): NGENTOT_LC (fixed
    total), NGEN_LC (generate until N pass cuts), or NGEN_SEASON (rate × time ×
    area). NGEN_SCALE multiplies whichever is in play. Exactly one of the three
    must be set. Returns ``(n_simulated, ngen_lc, frac_pec1a, season_count)``,
    with ``ngen_lc`` already NGEN_SCALE-scaled and ``season_count`` the rate
    model's events per season (SEASON_COUNT, snlc_sim.c:9178-9179).
    """
    # NGEN_SCALE_NON1A scales the count for NON1A/SIMSED models only
    # (snlc_sim.c:7124).
    scale = config.run.ngen_scale
    if MODEL_REGISTRY[config.model.model_type].uses_nonia_rates:
        scale *= config.run.ngen_scale_non1a
    ngentot_lc, ngen_lc, ngen_season = (
        config.run.ngentot_lc, config.run.ngen_lc, config.run.ngen_season)

    # prep_SIMLIB_DUMP forces NGENTOT_LC (snlc_sim.c:32558)
    if config.survey.simlib_dump >= 0:
        ngentot_lc, ngen_lc, ngen_season = 150000, 0, 0.0

    n_modes_set = (ngentot_lc > 0) + (ngen_lc > 0) + (ngen_season > 0)
    if n_modes_set > 1:
        raise SNANAError(
            "NGEN_LC, NGENTOT_LC and NGEN_SEASON are mutually exclusive. Set "
            "exactly one.",
            function_name="resolve_counts")
    if n_modes_set == 0 and not config.run.readme_dumpflag:
        raise SNANAError(
            "Required: one of NGEN_LC, NGENTOT_LC or NGEN_SEASON must be set.",
            function_name="resolve_counts")

    # (int)(NGEN_LC*NGEN_SCALE), snlc_sim.c:7146
    ngen_lc = int(ngen_lc * scale)

    def _expected(entries, rate_mods):
        c = config.cosmology
        return calculate_expected_sn_count(
            config.dndz_params(entries), config.rates.solid_angle,
            config.rates.redshift_range, config.rates.peakmjd_range,
            config.rates.peakmjd_exclude,
            c.omega_matter, c.omega_lambda, c.h0, c.w0, c.wa,
            hz_map, rate_mods)

    # PEC1Asum: SNcount_model over RATEPAR_PEC1A (snlc_sim.c:9006)
    sn_sum = _expected(None, config.rates.dndz_rate_mods)
    pec1a_sum = (_expected(config.rates.dndz_entries_pec1a,
                           neutral_rate_mods())
                 if config.rates.dndz_entries_pec1a else 0.0)
    total = sn_sum + pec1a_sum
    frac_pec1a = pec1a_sum / total if total else 0.0

    if ngen_season > 0:
        n_simulated = int(ngen_season * total * scale)
        logger.info(
            f"NGEN_SEASON={ngen_season:.3f} × rate={total:.2f}/season × "
            f"NGEN_SCALE={scale:.3f} → NGENTOT_LC={n_simulated}")
    elif ngen_lc > 0:
        n_simulated = ngen_lc
    else:
        n_simulated = int(ngentot_lc * scale)
    return n_simulated, ngen_lc, frac_pec1a, total


def _cdf_on_grid(config, hz_map, z_grid, dndz_params, rate_mods) -> jnp.ndarray:
    """Normalised cumulative dN/dz over `z_grid`."""
    dndz = dndz_density_on_grid(
        dndz_params, z_grid, *cosmo_params(config.cosmology), hz_map, rate_mods)
    cdf = np.cumsum(0.5 * (dndz[:-1] + dndz[1:]) * np.diff(z_grid))
    cdf = np.concatenate([[0.0], cdf])
    return jnp.array(cdf / cdf[-1], dtype=jnp.float32)


def build_murewgt(config, hz_map):
    """The DNDZ reweight's CDF against distance modulus, from which a MUSHIFT
    range is drawn so that a shifted event keeps the reweighted redshift
    distribution (prep_user_murewgt, snlc_sim.c:7662). The grid runs in
    steps of 0.001 from the lowest generated redshift to 0.1 past the one
    whose distance is the highest generated distance plus the top of the
    MUSHIFT range. None unless there is both a MUSHIFT range and a
    z-dependent DNDZ_ZPOLY_REWGT or DNDZ_Z1POLY_REWGT (snlc_sim.c:7650)."""
    lo, hi = config.model.mushift
    mods = config.rates.dndz_rate_mods
    zpoly, z1poly = mods['zpoly'], mods['z1poly']
    if not (hi - lo > 0 and (len(zpoly) > 1 or len(z1poly) > 1)):
        return None
    omega_m, _omega_k, omega_l, hubble_distance, w0, wa = cosmo_params(config.cosmology)
    anisotropy = (ANISOTROPY_V04_PARS
                  if config.cosmology.anisotropy_modelname else None)
    # gen_dLmag at galactic (0, 0) on the grid, dLmag for the upper edge
    cos_sep = float(cos_sep_cmb_apex_galactic(0.0, 0.0))

    def mu_of(z, aniso):
        return np.asarray(jax.vmap(lambda zi: distance_modulus(
            zi, zi, omega_m, omega_l, hubble_distance, w0, wa, 0.0, hz_map,
            anisotropy=aniso, cos_sep=cos_sep))(jnp.asarray(z)))

    z_min, z_max = config.rates.redshift_range
    mu_max_grid = float(mu_of([z_max], None)[0]) + hi
    z_max_grid = float(distance_modulus_to_redshift(
        mu_max_grid, omega_m, omega_l, hubble_distance, w0, wa)) + 0.1
    z = np.arange(z_min, z_max_grid, 0.001)
    rewgt = np.ones_like(z)
    if zpoly:
        rewgt = rewgt * eval_genpoly(zpoly, z)
    if z1poly:
        rewgt = rewgt * eval_genpoly(z1poly, 1.0 + z)
    return jnp.asarray(mu_of(z, anisotropy)), jnp.asarray(np.cumsum(rewgt))


def build_redshift_cdf(config, hz_map):
    """Numerical CDF for dN/dz including the comoving volume element.

    DNDZ_PEC1A gives the peculiar-Ia population its own rate, hence its own
    CDF; set_RATEPAR (snlc_sim.c:14525) picks per event by template class,
    taking none of the DNDZ_* reweights (snlc_sim.c:2722).
    """
    z_min, z_max = config.rates.redshift_range
    z_grid = np.linspace(z_min, z_max, 2000)
    pec1a_params = config.dndz_params(config.rates.dndz_entries_pec1a)
    return (
        jnp.array(z_grid, dtype=jnp.float32),
        _cdf_on_grid(config, hz_map, z_grid, config.dndz_params(),
                     config.rates.dndz_rate_mods),
        _cdf_on_grid(config, hz_map, z_grid, pec1a_params,
                     neutral_rate_mods())
        if pec1a_params else None,
    )
