"""
Sersic profile sampling for SN position generation within host galaxies.

This module implements JAX-compatible Sersic profile sampling matching SNANA's
GEN_SNHOST_POS algorithm:

1. Compute b_n (concentration parameter) for each Sersic index n
2. Pre-compute cumulative integral tables for inverse CDF sampling
3. Sample radial position using inverse transform sampling
4. Sample angle uniformly over elliptical area
5. Transform to RA/DEC offsets with rotation
6. Calculate SNSEP (SN-galaxy separation) and DLR/DDLR

References:
- SNANA: src/sntools_host.c (GEN_SNHOST_POS, init_Sersic_integrals)
- Sersic (1968): Galaxy profile I(r) = I_e * exp[-b_n * ((r/r_e)^(1/n) - 1)]
- Ciotti & Bertin (1999): b_n approximation for n > 0.36
"""

import logging
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import jax
import jax.numpy as jnp

from jax_snlc_core.utils import SNANAError, sndata_root, snana_log as slog

logger = logging.getLogger(__name__)


NSERSIC_TABLE = 50
SERSIC_INDEX_MIN = 0.15
SERSIC_INDEX_MAX = 11.00
NBIN_RADIUS_SERSIC = 200
MINRADIUS_SERSIC = 1.0e-4  # R/Re
MAXRADIUS_SERSIC = 100.0   # R/Re

SERSIC_BN_FILE = "simlib/Sersic_bn.dat"
GAUSS2D_FILE = "simlib/Gauss2dIntegrals.dat"


def load_sersic_bn_table() -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Exact b_n lookup for low Sersic indices from Sersic_bn.dat.

    SNANA reads the same file at init (read_Sersic_bn,
    sntools_host.c:5194) and aborts when it is missing. Called from the
    hostlib build path, never at import.
    """
    path = sndata_root() / SERSIC_BN_FILE
    if not path.exists():
        raise SNANAError(
            f"Cannot find Sersic b_n table: {path}\nCheck $SNDATA_ROOT.",
            function_name="load_sersic_bn_table")
    data = np.loadtxt(path)
    return (jnp.array(data[:, 0], dtype=jnp.float32),
            jnp.array(data[:, 1], dtype=jnp.float32))


def load_gauss2d_table() -> jnp.ndarray:
    """Gauss2d overlap integrals for GALFRAC PSF convolution.

    51 entries: R from 0.0 to 5.0 in steps of 0.1, SIG always 0.5; each row
    is the fraction of a 2D Gaussian within the unit circle at offset R.
    SNANA reads the same table at init (init_Gauss2d_Overlap,
    sntools_host.c:4815) and aborts when it is missing. Called from the
    hostlib build path, never at import.
    """
    path = sndata_root() / GAUSS2D_FILE
    if not path.exists():
        raise SNANAError(
            f"Cannot find Gauss2d overlap table: {path}\nCheck $SNDATA_ROOT.",
            function_name="load_gauss2d_table")
    integ_vals = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('ROW:'):
                parts = line.split()
                integ_vals.append(float(parts[4]))
    return jnp.array(integ_vals, dtype=jnp.float32)


NMAGPSF_HOSTLIB = 9
NRBIN_GALMAG = 100
NTHBIN_GALMAG = 36
PSFMAX_SNANA = 5.0

# PSF sigma (arcsec) per aperture; index 0 is the total
APERTURE_PSFSIG = np.array([
    0.0,
    0.03 / 2.35,
    0.07 / 2.35,
    0.10 / 2.35,
    0.20 / 2.35,
    0.40 / 2.35,
    0.80 / 2.35,
    1.30 / 2.35,
    2.10 / 2.35,
    PSFMAX_SNANA / 2.35
], dtype=np.float32)

APERTURE_RADIUS = 2.0 * APERTURE_PSFSIG

# numpy: built at import, before the input picks the floating-point frame
_THBIN = 2.0 * np.pi / NTHBIN_GALMAG
_COS_TH = np.cos(np.arange(NTHBIN_GALMAG) * _THBIN)
_SIN_TH = np.sin(np.arange(NTHBIN_GALMAG) * _THBIN)


@dataclass
class SersicTableData:
    """Pre-computed Sersic integral tables for fast inverse CDF sampling.

    All arrays stored as JAX arrays for GPU compatibility.
    """
    n: jnp.ndarray           # Shape: (n_table,)
    inv_n: jnp.ndarray       # Shape: (n_table,)
    bn: jnp.ndarray          # Shape: (n_table,)
    reduced_logR: jnp.ndarray  # Shape: (n_radius,)
    integ_cum: jnp.ndarray   # Shape: (n_table, n_radius)
    integ_total: jnp.ndarray # Shape: (n_table,), unnormalised integral per Sersic index
    n_table: int
    n_radius: int
    bn_n_lookup: jnp.ndarray    # Sersic_bn.dat n column
    bn_lookup: jnp.ndarray      # Sersic_bn.dat b_n column
    gauss2d_integ: jnp.ndarray  # Gauss2dIntegrals.dat INTEG2D column


def compute_sersic_bn(n: jnp.ndarray, bn_n_lookup: jnp.ndarray,
                      bn_lookup: jnp.ndarray) -> jnp.ndarray:
    """Sersic concentration parameter b_n.

    Ciotti & Bertin (1999) polynomial for n > 0.36; exact lookup from
    Sersic_bn.dat (load_sersic_bn_table) for n ≤ 0.36 where the polynomial
    breaks down.
    """
    n2 = n * n
    n3 = n2 * n
    n4 = n2 * n2
    bn_poly = (
        2.0 * n - 1.0 / 3.0 +
        4.0 / (405.0 * n) +
        46.0 / (25515.0 * n2) +
        131.0 / (1148175.0 * n3) +
        2194697.0 / (30690717750.0 * n4)
    )
    bn_interp = jnp.interp(n, bn_n_lookup, bn_lookup)
    return jnp.where(n > 0.36, bn_poly, bn_interp)


def compute_sersic_integral_table() -> SersicTableData:
    """Pre-compute Sersic cumulative integral tables for all Sersic indices.

    For each Sersic index n, computes the cumulative radial integral
    ``CumFlux(R) = integral_0^R r * I(r) dr``, where
    ``I(r) = exp[-b_n * ((r/r_e)^(1/n) - 1)]`` is the Sersic intensity
    profile, using the trapezoidal rule on a grid built in 1/n space (for
    linear interpolation) and log-space radius. NSERSIC_TABLE,
    SERSIC_INDEX_MIN and SERSIC_INDEX_MAX set the Sersic index grid;
    NBIN_RADIUS_SERSIC, MINRADIUS_SERSIC and MAXRADIUS_SERSIC the
    reduced-radius grid.

    Returns the SersicTableData with pre-computed integral tables. Raises
    `SNANAError` if the half-light radius sanity check fails: R_half should
    land at R/Re = 1.0 by definition, and a deviation beyond SNANA's 10%
    tolerance means the b_n calculation or integration is wrong.
    """
    n_table, n_radius = NSERSIC_TABLE, NBIN_RADIUS_SERSIC
    n_min, n_max = SERSIC_INDEX_MIN, SERSIC_INDEX_MAX
    r_min, r_max = MINRADIUS_SERSIC, MAXRADIUS_SERSIC
    logger.info("Computing Sersic integral tables...")
    slog.line(logger, f"Sersic indices: {n_table} values from n={n_min:.2f} to {n_max:.2f}")
    slog.line(logger, f"Radius bins: {n_radius} from R/Re={r_min:.1e} to {r_max:.1f}")

    t0 = time.time()

    inv_n_min = 1.0 / n_max
    inv_n_max = 1.0 / n_min
    inv_n_values = np.linspace(inv_n_min, inv_n_max, n_table)
    n_values = 1.0 / inv_n_values

    bn_n_lookup, bn_lookup = load_sersic_bn_table()
    gauss2d_integ = load_gauss2d_table()
    bn_values = np.asarray(
        compute_sersic_bn(jnp.asarray(n_values), bn_n_lookup, bn_lookup))

    log_r_min = np.log10(r_min)
    log_r_max = np.log10(r_max)
    log_r_grid = np.linspace(log_r_min, log_r_max, n_radius + 1)
    r_grid = 10.0 ** log_r_grid

    inv_n_2d = inv_n_values[:, np.newaxis]
    bn_2d = bn_values[:, np.newaxis]
    r_2d = r_grid[np.newaxis, :]

    arg = r_2d ** inv_n_2d - 1.0
    integrand = r_2d * np.exp(-bn_2d * arg)

    r_diff = np.diff(r_grid)
    f_avg = 0.5 * (integrand[:, :-1] + integrand[:, 1:])
    d_integ = f_avg * r_diff[np.newaxis, :]

    integ_cum = np.zeros((n_table, n_radius + 1))
    integ_cum[:, 1:] = np.cumsum(d_integ, axis=1)

    # x 2*pi, SNANA INTEG_SUM
    integ_total = integ_cum[:, -1].copy() * (2.0 * np.pi)

    integ_max = integ_cum[:, -1:]
    integ_cum = np.where(integ_max > 0, integ_cum / integ_max, 0.0)

    # SNANA's 10% half-light check
    idx_half = np.argmax(integ_cum >= 0.5, axis=1)
    r_half = r_grid[idx_half]
    tolerance = 0.1
    deviations = np.abs(r_half - 1.0)
    failed_mask = deviations > tolerance

    if np.any(failed_mask):
        failed_indices = np.where(failed_mask)[0]
        failed_n = n_values[failed_indices]
        failed_r_half = r_half[failed_indices]
        failed_bn = bn_values[failed_indices]

        error_msg = f"Sersic integral sanity check FAILED for {len(failed_indices)} profile(s):\n"
        for i in range(min(len(failed_indices), 10)):
            error_msg += f"  n={failed_n[i]:.3f}: R_half={failed_r_half[i]:.3f} (b_n={failed_bn[i]:.3f})\n"
        if len(failed_indices) > 10:
            error_msg += f"  ... and {len(failed_indices) - 10} more\n"
        error_msg += f"Expected R_half ~ 1.0 (within {tolerance*100:.0f}%)\n"
        error_msg += "SNANA would abort here - aborting to prevent corrupted simulation"
        raise SNANAError(error_msg)

    t_compute = time.time() - t0

    sersic_data = SersicTableData(
        n=jnp.array(n_values, dtype=jnp.float32),
        inv_n=jnp.array(inv_n_values, dtype=jnp.float32),
        bn=jnp.array(bn_values, dtype=jnp.float32),
        reduced_logR=jnp.array(log_r_grid, dtype=jnp.float32),
        integ_cum=jnp.array(integ_cum, dtype=jnp.float32),
        integ_total=jnp.array(integ_total, dtype=jnp.float32),
        n_table=n_table,
        n_radius=n_radius + 1,
        bn_n_lookup=bn_n_lookup,
        bn_lookup=bn_lookup,
        gauss2d_integ=gauss2d_integ,
    )

    slog.line(logger, f"Computed in {t_compute:.3f}s ({t_compute*1000:.1f}ms)")
    slog.line(logger, f"Sanity check passed: All {n_table} profiles have R_half within 10% of 1.0")

    return sersic_data


def _interp_gauss2d(rcen_frac, gauss2d_integ):
    """Interpolate Gauss2d overlap table at given R/ApertureRadius values.

    Linear interpolation on the 51-entry table (R from 0.0 to 5.0, step 0.1).
    SNANA returns 0.0 for R/SIG > 6.0; since SIG=0.5 always, cutoff at R > 3.0.
    """
    rbin = 0.1
    idx = jnp.floor(rcen_frac / rbin).astype(jnp.int32)
    idx = jnp.clip(idx, 0, 49)
    frac = (rcen_frac - idx * rbin) / rbin
    frac = jnp.clip(frac, 0.0, 1.0)
    ovp = gauss2d_integ[idx] + frac * (gauss2d_integ[idx + 1] - gauss2d_integ[idx])
    ovp = jnp.where(rcen_frac > 3.0, 0.0, ovp)
    return ovp


def _compute_galfrac_single(x_sn, y_sn, a_half, b_half, sersic_n, sersic_w,
                            sersic_bn, integ_sum_fixed, gauss2d_integ):
    """Compute GALFRAC at 9 PSF apertures for a single SN.

    Matches SNANA GEN_SNHOST_GALMAG polar grid integration.

    ``x_sn``/``y_sn`` is the SN position in the galaxy intrinsic frame
    (arcsec); ``a_half``/``b_half`` the half-light semi-major/minor axes
    per component, and ``sersic_n``/``sersic_w``/``sersic_bn`` the other
    per-component parameters, all shape (n_prof,). ``integ_sum_fixed`` is
    the scalar SNANA fixed INTEG_SUM[NSERSIC_TABLE-1]; ``gauss2d_integ`` is
    the (51,) Gauss2d overlap table (load_gauss2d_table).

    Returns the (10,) GALFRAC for each PSF aperture (index 0 unused).
    """
    # polar grid out to twice the largest aperture radius
    rbin = 2.0 * APERTURE_RADIUS[NMAGPSF_HOSTLIB] / NRBIN_GALMAG
    r_centers = (jnp.arange(NRBIN_GALMAG) + 0.5) * rbin

    xgal = (x_sn + r_centers[:, None] * _COS_TH[None, :])[:, :, None]
    ygal = (y_sn + r_centers[:, None] * _SIN_TH[None, :])[:, :, None]

    # FGAL_TOT = a * b * INTEG_SUM[NSERSIC_TABLE-1] for every component, as SNANA
    sqsum = (xgal / a_half)**2 + (ygal / b_half)**2
    arg = jnp.power(sqsum, 0.5 / sersic_n) - 1.0
    amp = sersic_w / (a_half * b_half * integ_sum_fixed)
    fgal_sum_th = jnp.sum(amp * jnp.exp(-sersic_bn * arg), axis=(1, 2))

    jac = r_centers * rbin * _THBIN
    rcen_frac_all = r_centers[:, None] / APERTURE_RADIUS[None, 1:NMAGPSF_HOSTLIB + 1]
    galfrac_1_9 = (jac * fgal_sum_th) @ _interp_gauss2d(rcen_frac_all,
                                                        gauss2d_integ)

    return jnp.concatenate([jnp.array([1.0]), jnp.minimum(galfrac_1_9, 1.0)])


@jax.jit
def compute_galfrac(
    x_sn, y_sn, a_half, b_half, sersic_n, sersic_w, sersic_bn,
    integ_sum_fixed, gauss2d_integ
):
    """Compute GALFRAC at 9 PSF apertures for a batch of SNe.

    Matches SNANA GEN_SNHOST_GALMAG polar grid integration. ``x_sn``/
    ``y_sn`` are (n_sne,), the intrinsic major/minor-axis offsets (arcsec);
    ``a_half``/``b_half``/``sersic_n``/``sersic_w``/``sersic_bn`` are
    (n_sne, n_prof), the per-component half-light axes, Sersic index,
    weight and b_n. ``integ_sum_fixed`` is the scalar SNANA
    INTEG_SUM[NSERSIC_TABLE-1] (fixed for all components); ``gauss2d_integ``
    is the (51,) Gauss2d overlap table (load_gauss2d_table).

    Returns the (n_sne, 10) GALFRAC for each PSF aperture.
    """
    return jax.vmap(
        lambda x, y, a, b, n, w, bn: _compute_galfrac_single(
            x, y, a, b, n, w, bn, integ_sum_fixed, gauss2d_integ
        )
    )(x_sn, y_sn, a_half, b_half, sersic_n, sersic_w, sersic_bn)


@jax.jit
def _generate_sn_position_jit(
    a_half: jnp.ndarray,
    b_half: jnp.ndarray,
    sersic_n: jnp.ndarray,
    rotation_angle_deg: jnp.ndarray,
    random_radius: jnp.ndarray,
    random_u1: jnp.ndarray,
    random_u2: jnp.ndarray,
    mnintflux: jnp.ndarray,
    mxintflux: jnp.ndarray,
    fixran_phi: jnp.ndarray,
    sersic_integ_cum: jnp.ndarray,
    sersic_inv_n: jnp.ndarray,
    sersic_reduced_logR: jnp.ndarray,
    sersic_n_table: int,
    sersic_n_radius: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JIT-compiled core of generate_sn_position. All ops fused into one kernel."""
    def _sample_radius_single(n_scalar, random_cumflux_scalar):
        inv_n = 1.0 / n_scalar
        inv_n_min = sersic_inv_n[0]
        inv_n_bin = sersic_inv_n[1] - sersic_inv_n[0]
        frac_idx = (inv_n - inv_n_min) / inv_n_bin
        k0 = jnp.floor(frac_idx).astype(jnp.int32)
        k1 = k0 + 1
        k0 = jnp.clip(k0, 0, sersic_n_table - 2)
        k1 = jnp.clip(k1, 1, sersic_n_table - 1)
        integ0 = sersic_integ_cum[k0]
        integ1 = sersic_integ_cum[k1]
        idx = jnp.searchsorted(integ0, random_cumflux_scalar)
        idx = jnp.clip(idx, 1, sersic_n_radius - 1)
        f0 = integ0[idx - 1]; f1 = integ0[idx]
        r0 = sersic_reduced_logR[idx - 1]; r1 = sersic_reduced_logR[idx]
        frac = (random_cumflux_scalar - f0) / (f1 - f0 + 1e-10)
        log_r_at_k0 = r0 + frac * (r1 - r0)
        idx1 = jnp.searchsorted(integ1, random_cumflux_scalar)
        idx1 = jnp.clip(idx1, 1, sersic_n_radius - 1)
        f0_1 = integ1[idx1 - 1]; f1_1 = integ1[idx1]
        r0_1 = sersic_reduced_logR[idx1 - 1]; r1_1 = sersic_reduced_logR[idx1]
        frac1 = (random_cumflux_scalar - f0_1) / (f1_1 - f0_1 + 1e-10)
        log_r_at_k1 = r0_1 + frac1 * (r1_1 - r0_1)
        frac_n = frac_idx - k0
        log_r_reduced = log_r_at_k0 + frac_n * (log_r_at_k1 - log_r_at_k0)
        return 10.0 ** log_r_reduced

    # radial uniform -> [MN,MX]INTFLUX_SNPOS window (GEN_SNHOST_POS, sntools_host.c:7713)
    random_radius_capped = mnintflux + random_radius * (mxintflux - mnintflux)

    reduced_r = jax.vmap(_sample_radius_single)(sersic_n, random_radius_capped)

    r_circle = jnp.sqrt(random_u1)
    theta_circle = 2.0 * jnp.pi * random_u2
    x_ellipse = a_half * r_circle * jnp.cos(theta_circle)
    y_ellipse = b_half * r_circle * jnp.sin(theta_circle)
    phi = jnp.arctan2(y_ellipse, x_ellipse)
    phi = jnp.where(phi < 0, phi + 2.0 * jnp.pi, phi)
    # FIXRAN_PHI: phi = 2*pi*ran directly (GEN_SNHOST_ANGLE, sntools_host.c:7922)
    phi = jnp.where(fixran_phi >= 0.0, 2.0 * jnp.pi * fixran_phi, phi)

    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)
    dlr = (a_half * b_half) / jnp.sqrt(
        (b_half * cos_phi)**2 + (a_half * sin_phi)**2 + 1e-20
    )

    a = reduced_r * dlr * cos_phi
    b = reduced_r * dlr * sin_phi
    rot_rad = rotation_angle_deg * (jnp.pi / 180.0)
    cos_rot = jnp.cos(rot_rad)
    sin_rot = jnp.sin(rot_rad)
    ra_offset_asec = a * cos_rot + b * sin_rot
    dec_offset_asec = b * cos_rot - a * sin_rot

    snsep = jnp.sqrt(ra_offset_asec**2 + dec_offset_asec**2)
    ddlr = snsep / (dlr + 1e-20)

    return ra_offset_asec, dec_offset_asec, snsep, dlr, ddlr, reduced_r, a, b


def generate_sn_position(
    a_half: jnp.ndarray,
    b_half: jnp.ndarray,
    sersic_n: jnp.ndarray,
    rotation_angle_deg: jnp.ndarray,
    random_radius: jnp.ndarray,
    random_u1: jnp.ndarray,
    random_u2: jnp.ndarray,
    sersic_data: SersicTableData,
    mnintflux: float,
    mxintflux: float,
    fixran_phi: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate SN position within host galaxy using Sersic profile sampling.

    Thin wrapper that unpacks SersicTableData for the JIT-compiled core.

    Returns ``(ra_offset, dec_offset, snsep, dlr, ddlr, reduced_r,
    a_SNGALSEP, b_SNGALSEP)``, where a_SNGALSEP/b_SNGALSEP are intrinsic
    galaxy-frame coordinates (arcsec).
    """
    return _generate_sn_position_jit(
        a_half, b_half, sersic_n, rotation_angle_deg,
        random_radius, random_u1, random_u2,
        jnp.asarray(mnintflux, dtype=jnp.float32),
        jnp.asarray(mxintflux, dtype=jnp.float32),
        jnp.asarray(fixran_phi, dtype=jnp.float32),
        sersic_data.integ_cum, sersic_data.inv_n,
        sersic_data.reduced_logR,
        sersic_data.n_table, sersic_data.n_radius
    )


jax.tree_util.register_dataclass(
    SersicTableData,
    data_fields=[
        'n',
        'inv_n',
        'bn',
        'reduced_logR',
        'integ_cum',
        'integ_total',
        'bn_n_lookup',
        'bn_lookup',
        'gauss2d_integ',
    ],
    meta_fields=[
        'n_table',
        'n_radius',
    ],
)
