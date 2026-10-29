"""Cosmology: the expansion rate, distances, the distance modulus, comoving
volume and the Visser (2004) anisotropy, and the frames a redshift and a sky
position are given in (heliocentric and CMB redshifts, galactic coordinates,
the CMB dipole).
"""

import logging
import os
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap

from jax_snlc_core.interpolation import linear_between, linear_bin
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import read_two_column_file
from jax_snlc_core.utils import LIGHT_KM

logger = logging.getLogger(__name__)

# static cap on Hzinv_integral's Nzbin (z <= 10)
_HZ_MAP_MAX_BINS = 10_000

# Photon + neutrino density used by SNANA wfit.c:EofZ.
OMEGA_RADIATION = 0.9e-4


def _legendre_rule(order: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return fixed Gauss--Legendre nodes and weights."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return jnp.asarray(nodes), jnp.asarray(weights)


@jit
def _e2(
    z: jnp.ndarray,
    omega_m: float,
    omega_k: float,
    omega_lambda: float,
    w0: float = -1.0,
    wa: float = 0.0,
    *,
    radiation: bool = True,
) -> jnp.ndarray:
    """CPL ``E(z)^2`` with SNANA wfit's photon + neutrino contribution.

    Matter and dark energy are rescaled by ``1 - Omega_r``; curvature is
    supplied before that rescaling. No floor is imposed on invalid E².
    ``radiation=False`` recovers SNANA's simulator/BBC expansion rate.
    """
    omega_r = jnp.where(radiation, OMEGA_RADIATION, 0.0)
    zp1 = 1.0 + z
    dark_energy = (
        jnp.power(zp1, 3.0 * (1.0 + w0 + wa))
        * jnp.exp(-3.0 * wa * z / zp1)
    )
    return (
        (1.0 - omega_r) * omega_m * zp1**3
        + omega_k * zp1**2
        + (1.0 - omega_r) * omega_lambda * dark_energy
        + omega_r * zp1**4
    )


@jit
def cpl_e2(
    z: jnp.ndarray,
    omega_m: float,
    omega_lambda: float,
    w0: float = -1.0,
    wa: float = 0.0,
    *,
    radiation: bool = True,
) -> jnp.ndarray:
    """``E(z)^2`` with the curvature implied by ``omega_m + omega_lambda``."""
    return _e2(
        z, omega_m, 1.0 - omega_m - omega_lambda, omega_lambda,
        w0, wa, radiation=radiation,
    )


@jit
def flat_cpl_e2(
    z: jnp.ndarray,
    omega_m: float,
    w0: float = -1.0,
    wa: float = 0.0,
    *,
    radiation: bool = True,
) -> jnp.ndarray:
    """Dimensionless ``E(z)^2`` for a flat CPL cosmology."""
    return cpl_e2(z, omega_m, 1.0 - omega_m, w0, wa, radiation=radiation)


@partial(jit, static_argnames=("order",))
def _comoving_integral(
    redshift: jnp.ndarray,
    omega_m: float,
    omega_k: float,
    omega_lambda: float,
    w0: float = -1.0,
    wa: float = 0.0,
    hz_map: Optional[Tuple] = None,
    *,
    order: int = 64,
    radiation: bool = True,
) -> jnp.ndarray:
    """``integral_0^z dz'/E(z')``, the one integral every distance in the
    repo uses: Gauss--Legendre in ``log(1+z)`` for the smooth analytic
    ``E(z)``, and SNANA's midpoint rule for an HzFUN_FILE ``(z, E)`` table,
    whose linear interpolation (Hzfun_interp, OPT_INTERP=1) is only
    piecewise smooth."""
    z = jnp.asarray(redshift)
    if hz_map is not None:
        # Hzinv_integral (sntools_cosmology.c:352-397), masked to a static size
        nbin = jnp.clip(jnp.floor(z * 1000.0), 10.0, _HZ_MAP_MAX_BINS)
        dz = jnp.where(nbin > 0.0, z / nbin, 0.0)
        i = jnp.arange(_HZ_MAP_MAX_BINS)
        z_mid = (i + 0.5) * dz[..., None]
        i_bin, frac = linear_bin(z_mid, hz_map[0])
        e = linear_between(hz_map[1][i_bin], hz_map[1][i_bin + 1], frac)
        live = i < nbin[..., None]
        integral = dz * jnp.sum(jnp.where(live, 1.0 / e, 0.0), axis=-1)
        return jnp.where(z > 0.0, integral, 0.0)
    nodes, weights = _legendre_rule(order)
    umax = jnp.log1p(jnp.maximum(z, 0.0))
    u = 0.5 * umax[..., None] * (nodes + 1.0)
    z_nodes = jnp.expm1(u)
    e = jnp.sqrt(_e2(
        z_nodes, omega_m, omega_k, omega_lambda, w0, wa,
        radiation=radiation,
    ))
    integral = 0.5 * umax * jnp.sum(weights * jnp.exp(u) / e, axis=-1)
    return jnp.where(z > 0.0, integral, 0.0)


@partial(jit, static_argnames=("order",))
def cpl_comoving_integral(
    redshift: jnp.ndarray,
    omega_m: float,
    omega_lambda: float,
    w0: float = -1.0,
    wa: float = 0.0,
    *,
    order: int = 64,
    radiation: bool = True,
) -> jnp.ndarray:
    """``integral_0^z dz'/E(z')`` with the implied curvature."""
    return _comoving_integral(
        redshift, omega_m, 1.0 - omega_m - omega_lambda, omega_lambda,
        w0, wa, order=order,
        radiation=radiation,
    )


@partial(jit, static_argnames=("order",))
def flat_cpl_comoving_integral(
    redshift: jnp.ndarray,
    omega_m: float,
    w0: float = -1.0,
    wa: float = 0.0,
    *,
    order: int = 64,
    radiation: bool = True,
) -> jnp.ndarray:
    """Return the flat ``integral_0^z dz/E(z)``."""
    return cpl_comoving_integral(
        redshift,
        omega_m,
        1.0 - omega_m,
        w0,
        wa,
        order=order,
        radiation=radiation,
    )


def load_hz_map(path: str, hubble_constant: float,
                z_max: Optional[float] = None) -> Tuple:
    """Read an HzFUN_FILE into (z, E=H/H0) for interpolation.

    SNANA reads the table with rd2columnFile (sntools_cosmology.c:63).

    ``z_max`` is the largest redshift the run evaluates. SNANA's Hzfun_interp
    delegates to interp_1DFUN, which aborts outside the table
    (sntools_cosmology.c:512-522), so a table ending below ``z_max`` is
    fatal here rather than silently held at its edge value.
    """
    z, hz = read_two_column_file(Path(os.path.expandvars(str(path))))
    if z[0] != 0.0:
        raise SNANAError(
            f"HzFUN_FILE {path} starts at z={z[0]}, but the first bin must be "
            f"z=0 because the distance integrals start there.",
            function_name="load_hz_map")
    if z_max is not None and z[-1] < z_max:
        raise SNANAError(
            f"HzFUN_FILE {path} ends at z={z[-1]}, below the configured redshift "
            f"maximum {z_max}; SNANA's interp_1DFUN aborts outside the table.",
            function_name="load_hz_map")
    return (jnp.asarray(z), jnp.asarray(hz / hubble_constant))


@jit
def hubble_parameter(z: float, omega_m: float, omega_k: float, omega_l: float,
                     w0: float = -1.0, wa: float = 0.0,
                     hz_map: Optional[Tuple] = None, *,
                     radiation: bool = True) -> float:
    """E(z) = H(z)/H0 for a CPL cosmology with optional radiation.

    w(a) = w0 + wa*(1-a) with a = 1/(1+z), so w0=-1, wa=0 is LCDM.
    The analytic rate includes radiation by default. ``hz_map`` is an
    HzFUN_FILE map as (z, E), linearly interpolated as SNANA's
    Hzfun_interp does (OPT_INTERP=1); when given it replaces the analytic
    form entirely.
    """
    if hz_map is not None:
        z_map, e_map = hz_map
        i_bin, frac = linear_bin(z, z_map)
        return linear_between(e_map[i_bin], e_map[i_bin + 1], frac)
    return jnp.sqrt(_e2(z, omega_m, omega_k, omega_l, w0, wa, radiation=radiation))


@jit
def _comoving_distance(z: float, omega_m: float, omega_k: float, omega_l: float,
                       hubble_distance: float, w0: float = -1.0, wa: float = 0.0,
                       hz_map: Optional[Tuple] = None, *,
                       radiation: bool = True) -> float:
    """Radial comoving distance in Mpc."""
    integral = _comoving_integral(
        z, omega_m, omega_k, omega_l, w0, wa, hz_map, radiation=radiation)
    return hubble_distance * integral


@jit
def transverse_comoving_distance(z: float, omega_m: float, omega_k: float,
                                 omega_l: float, hubble_distance: float,
                                 w0: float = -1.0, wa: float = 0.0,
                                 hz_map: Optional[Tuple] = None, *,
                                 radiation: bool = True) -> float:
    """Transverse comoving distance in Mpc, with SNANA's curvature rule."""
    dc = _comoving_distance(
        z, omega_m, omega_k, omega_l, hubble_distance, w0, wa, hz_map,
        radiation=radiation)

    # SNANA treats |KAPPA| <= 1e-5 as flat (Hzinv_integral,
    # sntools_cosmology.c:383-390).
    is_flat = jnp.abs(omega_k) <= 1.0e-5
    is_open = omega_k > 0

    # flat lane: placeholder 1 against 0-division
    sqrt_omega_k = jnp.sqrt(jnp.where(is_flat, 1.0, jnp.abs(omega_k)))
    argument = sqrt_omega_k * dc / hubble_distance
    dm_open = (hubble_distance / sqrt_omega_k) * jnp.sinh(argument)
    dm_closed = (hubble_distance / sqrt_omega_k) * jnp.sin(argument)

    return jnp.where(is_flat, dc, jnp.where(is_open, dm_open, dm_closed))


def comoving_volume_per_steradian(zmax, h0, omega_m, omega_l, w0=-1.0, wa=0.0):
    """The comoving volume per steradian out to each ``zmax``, in Mpc^3, by
    SNANA's midpoint rule (dVdz_integral, sntools_cosmology.c:285) with 1000
    bins per unit redshift and at least 10.

    The distance at each bin is a running sum over the same bins, so one
    prefix sum replaces SNANA's re-integration at every bin. The sum and the
    curvature are taken in float64 whatever the run's precision.
    """
    zmax = np.atleast_1d(np.asarray(zmax, dtype=float))
    nz = np.maximum((zmax * 1000.0).astype(int), 10)
    width = int(nz.max())
    dz = zmax / nz
    steps = np.arange(width)
    z = dz[:, None] * (steps + 0.5)
    live = steps < nz[:, None]
    hz = h0 * np.sqrt(np.asarray(cpl_e2(z, omega_m, omega_l, w0, wa)))
    inv_h = np.where(live, 1.0 / hz, 0.0)
    prefix = (np.cumsum(inv_h, axis=1) - 0.5 * inv_h) * dz[:, None] * h0
    kappa = 1.0 - omega_m - omega_l
    sqrt_kappa = np.sqrt(abs(kappa))
    if kappa < -1.0e-5:
        prefix = np.sin(sqrt_kappa * prefix) / sqrt_kappa
    elif kappa > 1.0e-5:
        prefix = np.sinh(sqrt_kappa * prefix) / sqrt_kappa
    r = prefix * LIGHT_KM / h0
    integrand = np.where(live, LIGHT_KM * r * r * inv_h, 0.0)
    return integrand.sum(axis=1) * dz


@jit
def _luminosity_distance(z: float, zhel: float, omega_m: float, omega_k: float,
                         omega_l: float, hubble_distance: float,
                         w0: float = -1.0, wa: float = 0.0, vpec: float = 0.0,
                         hz_map: Optional[Tuple] = None, *,
                         radiation: bool = True) -> float:
    """Luminosity distance for wCDM cosmology with optional peculiar velocity correction.

    Matches SNANA dLmag (sntools_cosmology.c:527-557) and Davis et al. 2011 Eq. 18:
        dl = (1 + zhel) * D_C(zcmb) * (1 + vpec/c)
    where D_C is the transverse comoving distance integrated over zCMB.
    Radiation is included by default, unlike SNANA's simulator/BBC.

    The (1 + zhel) prefactor implicitly contains one factor of (1 + vpec_SN/c)
    if zhel includes the SN's peculiar redshift, and the explicit (1 + vpec/c)
    factor adds the relativistic beaming correction. Together these give the
    (1 + z_pec_SN)^2 factor required by Davis Eq. 18.

    ``z`` is the cosmological CMB redshift, used as the comoving distance
    integral limit; ``zhel`` is the heliocentric redshift; ``vpec`` is the
    peculiar velocity in km/s for the relativistic beaming correction.
    """
    dm = transverse_comoving_distance(
        z, omega_m, omega_k, omega_l, hubble_distance, w0, wa, hz_map,
        radiation=radiation)
    return (1.0 + zhel) * dm * (1.0 + vpec / LIGHT_KM)


def cosmo_params(cosmology) -> Tuple[float, float, float, float, float, float]:
    """``(omega_m, omega_k, omega_l, hubble_distance, w0, wa)`` from a
    cosmology record, the tuple the static distance functions take."""
    omega_m = cosmology.omega_matter
    omega_l = cosmology.omega_lambda
    omega_k = 1.0 - omega_m - omega_l
    hubble_distance = LIGHT_KM / cosmology.h0
    return omega_m, omega_k, omega_l, hubble_distance, cosmology.w0, cosmology.wa



# Visser 2004 dipole defaults (SNANA sntools_cosmology.h:30-34).
ANISOTROPY_V04_PARS = (-0.157, -8.03, 0.0262, -0.489)   # qm, qd, S, J0


@jit
def _mu_anisotropic_V04(zhel, hubble_distance, cos_sep, pars) -> float:
    """Visser 2004 anisotropic distance modulus (SNANA dLmag_anisotropic,
    sntools_cosmology.c:580-586, with q_dipole_V04 at :601)."""
    qm, qd, s_dip, j0 = pars
    q_dip = qm + qd * jnp.exp(-zhel / s_dip) * cos_sep
    az1 = 0.5 * (1.0 - q_dip)
    az2 = -(1.0 - q_dip - 3.0 * q_dip * q_dip + j0) / 6.0
    dl = hubble_distance * zhel * (1.0 + az1 * zhel + az2 * zhel * zhel)
    return 25.0 + 5.0 * jnp.log10(dl)


@jit
def distance_modulus(redshift: float, zhel: float, omega_m: float,
                     omega_l: float, hubble_distance: float,
                     w0: float = -1.0, wa: float = 0.0, vpec: float = 0.0,
                     hz_map: Optional[Tuple] = None,
                     anisotropy: Optional[Tuple] = None,
                     cos_sep: float = 0.0, *, radiation: bool = True) -> float:
    """CPL distance modulus, the one every program in the repository uses.

    ``redshift`` is the CMB-frame redshift the comoving integral runs over and
    ``zhel`` the heliocentric redshift of the ``(1 + z)`` prefactor; a caller
    with no heliocentric redshift passes the CMB one, as SNANA does where it
    has none (snana.F90:19391). ``hubble_distance`` is c/H0 in Mpc, ``vpec``
    the peculiar velocity in km/s of the relativistic beaming term, and the
    curvature is 1 - omega_m - omega_l, treated as flat within 1e-5 as
    Hzinv_integral does (sntools_cosmology.c:383-390). With ``anisotropy``
    the Visser 2004 dipole replaces the result; ``cos_sep`` is then each SN's
    cosine to the CMB apex. Radiation defaults to SNANA wfit's convention;
    set ``radiation=False`` for SNANA simulator/BBC distances. HzFUN_FILE
    and anisotropy overrides retain their supplied expansion/distance laws.
    """
    if anisotropy is not None:
        return _mu_anisotropic_V04(zhel, hubble_distance, cos_sep, anisotropy)
    omega_k = 1.0 - omega_m - omega_l
    dl_mpc = _luminosity_distance(redshift, zhel, omega_m, omega_k, omega_l,
                                  hubble_distance, w0, wa, vpec, hz_map,
                                  radiation=radiation)
    dl_pc = dl_mpc * 1e6
    return 5.0 * jnp.log10(dl_pc / 10.0)


@partial(jit, static_argnums=(1, 2, 3, 4, 5))
def reference_distance_modulus(z, h0: float, omega_m: float, omega_l: float,
                               w0: float = -1.0, wa: float = 0.0, *,
                               radiation: bool = True) -> jnp.ndarray:
    """Vectorised redshift → distance modulus for an explicit cosmology.

    No defaults: the caller states the cosmology, so a fit cannot silently
    adopt one nobody chose (SNANA takes it from H0_REF/OMAT_REF/OLAM_REF/
    W0_REF/WA_REF).
    """
    hubble_distance = LIGHT_KM / h0
    return vmap(lambda zi: distance_modulus(
        zi, zi, omega_m, omega_l, hubble_distance, w0, wa,
        radiation=radiation))(jnp.asarray(z))


def distance_modulus_to_redshift(mu, omega_m, omega_l, hubble_distance,
                                 w0: float = -1.0, wa: float = 0.0, *,
                                 radiation: bool = True):
    """Invert distance modulus -> zCMB with SNANA's fixed-point iteration
    (zcmb_dLmag_invert, sntools_cosmology.c:653-686): from a Hubble-law
    start, ``z *= exp(-dmu/2)`` until ``|dmu| < 1e-4`` mag, unbounded in
    redshift. ``mu`` may be scalar or array; aborts if not converged after
    500 iterations, so it runs on the host, not under a trace."""
    mu_arr = jnp.atleast_1d(jnp.asarray(mu))
    dl = jnp.power(10.0, mu_arr / 5.0) * 1.0e-5   # Mpc
    z0 = (70.0 * dl) / LIGHT_KM
    z0 = z0 * jnp.exp(-z0 / 6.0)                  # SNANA's ad-hoc start

    def mu_of(z):
        return vmap(lambda zi: distance_modulus(
            zi, zi, omega_m, omega_l, hubble_distance, w0, wa,
            radiation=radiation))(z)

    def cond(state):
        _, dmu_max, niter = state
        return (dmu_max > 1.0e-4) & (niter <= 500)

    def body(state):
        z, _, niter = state
        dmu = mu_of(z) - mu_arr
        return z * jnp.exp(-dmu / 2.0), jnp.max(jnp.abs(dmu)), niter + 1

    z, dmu_max, niter = jax.lax.while_loop(cond, body, (z0, jnp.inf, 0))
    if float(dmu_max) > 1.0e-4:
        raise SNANAError(
            f"Could not solve for zCMB after NITER={int(niter)}; max |dmu| = "
            f"{float(dmu_max):e}",
            function_name="distance_modulus_to_redshift")
    return z.reshape(jnp.shape(mu))


# CMB apex in galactic coordinates (SNANA sntools.h:92-94)
_CMBapex_l = 264.031    # deg
_CMBapex_b = 48.253     # deg
_CMBapex_v = 369.82     # km/s


# slaEqgal's rotation from J2000 equatorial to galactic (sntools.c:4464-4472)
_EQ_TO_GAL_MATRIX = np.array([
    [-0.054875539726, -0.873437108010, -0.483834985808],
    [ 0.494109453312, -0.444829589425,  0.746982251810],
    [-0.867666135858, -0.198076386122,  0.455983795705]
])


def eq_to_gal(ra_deg, dec_deg):
    """Equatorial J2000 (RA, DEC) to galactic (l, b), in degrees, by slaEqgal's
    rotation, which SNANA uses for the CMB dipole (sntools_cosmology.c:749),
    the Milky Way dust map and the simulation alike. The rotation is a sum of
    element-wise products rather than a matrix product, which a GPU may run
    at reduced precision in single-precision runs."""
    ra_r = jnp.radians(jnp.asarray(ra_deg))
    dec_r = jnp.radians(jnp.asarray(dec_deg))
    cos_dec = jnp.cos(dec_r)
    v_eq = jnp.stack([cos_dec * jnp.cos(ra_r),
                      cos_dec * jnp.sin(ra_r),
                      jnp.sin(dec_r)], axis=-1)
    rotation = jnp.asarray(_EQ_TO_GAL_MATRIX, dtype=v_eq.dtype)
    v_gal = jnp.sum(v_eq[..., None, :] * rotation, axis=-1)
    l_deg = jnp.degrees(jnp.arctan2(v_gal[..., 1], v_gal[..., 0])) % 360.0
    b_deg = jnp.degrees(jnp.arcsin(jnp.clip(v_gal[..., 2], -1.0, 1.0)))
    return l_deg, b_deg


def cos_sep_cmb_apex(ra_deg, dec_deg):
    """cos of the angular separation between a sky position and the CMB apex
    (SNANA ``angSep``)."""
    return cos_sep_cmb_apex_galactic(*eq_to_gal(ra_deg, dec_deg))


def cos_sep_cmb_apex_galactic(l, b):
    """The same for a position given in galactic coordinates, in degrees."""
    br, ar = jnp.radians(b), jnp.radians(_CMBapex_b)
    return (jnp.sin(br) * jnp.sin(ar)
            + jnp.cos(br) * jnp.cos(ar) * jnp.cos(jnp.radians(l - _CMBapex_l)))


def cmb_vdotn(ra_deg, dec_deg):
    """V_CMB · n̂ / c at a given sky position."""
    return _CMBapex_v * cos_sep_cmb_apex(ra_deg, dec_deg) / LIGHT_KM


@jit
def zhelio_to_zcmb(z_helio, ra_deg, dec_deg):
    """Heliocentric → CMB-frame redshift (SNANA ``zhelio_zcmb_translator``
    with ``OPT=+1``). Preserves sentinel values (``z < 1e-10`` pass through).
    """
    vdotn = cmb_vdotn(ra_deg, dec_deg)
    z_cmb = (1.0 + z_helio) / (1.0 - vdotn) - 1.0
    return jnp.where(z_helio < 1e-10, z_helio, z_cmb)


@jit
def zcmb_to_zhelio(z_cmb, ra_deg, dec_deg):
    """CMB-frame → heliocentric redshift (inverse of :func:`zhelio_to_zcmb`;
    SNANA ``zhelio_zcmb_translator`` with ``OPT=-1``). Preserves sentinel
    values (``z < 1e-10`` pass through).
    """
    vdotn = cmb_vdotn(ra_deg, dec_deg)
    z_helio = (1.0 + z_cmb) * (1.0 - vdotn) - 1.0
    return jnp.where(z_cmb < 1e-10, z_cmb, z_helio)
