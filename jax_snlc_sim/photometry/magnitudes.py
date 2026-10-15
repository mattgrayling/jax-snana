"""FLUXCAL to magnitude, in SNANA's conventions.

The band integral itself is shared with the fitter and lives in
``jax_snlc_core.photometry.filter_system``, reached through
``SNModel.evaluate``. What stays here is the simulator's own: the null-mag
sentinel and flux floor, and the per-band quantities the writers report.
"""

import logging
from typing import Dict

import jax.numpy as jnp

logger = logging.getLogger(__name__)

# SNANA null-mag sentinel (matches SNANA genmag_SALT2.c:2265-2267).
SNANA_NULL_MAG = 99.0
# band-integral floor for SNANA_NULL_MAG
LOG_SAFE_FLOOR = 1e-30
# GEN_SNHOST_GALMAG's own R_V and law 94 (sntools_host.c:9256, :9278)
HOSTGAL_MW_RV = 3.1


def fluxcal_to_magnitude(fluxcal: jnp.ndarray, filter_indices: jnp.ndarray,
                         static_data: Dict) -> jnp.ndarray:
    """Magnitude from FLUXCAL, with SNANA's null sentinel where there is no
    flux to take a logarithm of.

    The floor is on the band integral rather than on FLUXCAL, so the epochs
    called null do not move with the zero point: FLUXCAL carries a factor
    ``10^(0.4 zp) / ref`` on top of it.
    """
    zp = static_data['zp_fluxcal']
    scale = static_data['fluxcal_scale_mag_offset']
    floor = (LOG_SAFE_FLOOR * jnp.power(10.0, 0.4 * zp) * scale
             / static_data['filter_ref_fluxes'][filter_indices])
    return jnp.where(
        (fluxcal <= floor) | jnp.isnan(fluxcal),
        SNANA_NULL_MAG,
        zp - 2.5 * jnp.log10(jnp.maximum(fluxcal, LOG_SAFE_FLOOR)))


def compute_rest_wavelengths(resources, filter_indices, redshifts):
    """Per-obs rest-frame wavelengths = filter_mean / (1+z); NaN where invalid."""
    valid = filter_indices >= 0
    safe_fi = jnp.maximum(filter_indices, 0)
    mean_wave = jnp.array(resources.photometry.filter_mean_wavelengths, dtype=jnp.float32)
    rest = mean_wave[safe_fi] / (1.0 + redshifts[:, None])
    return jnp.where(valid, rest, jnp.nan)


def host_mwxt_mag_per_band(resources, params):
    """MW extinction in mag added to every HOSTGAL_MAG, at each output
    filter's mean wavelength.

    GEN_SNHOST_GALMAG hard-wires R_V = 3.1 and colour law 94 for the host
    magnitudes, whatever RV_MWCOLORLAW and OPT_MWCOLORLAW say, and adds the
    result to every DDLR-sorted host before the GALMAG gate
    (sntools_host.c:9256, :9278, :9287).
    """
    order = [resources.filter_name_to_index[n] for n in resources.filter_names]
    av_true = HOSTGAL_MW_RV * jnp.asarray(params['sim_mwebv'])
    axav = resources.host_mw_law_at_mean.extinction_curve(
        resources.photometry.filter_mean_wavelengths, HOSTGAL_MW_RV, av=av_true[:, None])
    return av_true[:, None] * axav[..., jnp.asarray(order)]


def mwxt_mag_per_band(simulation, resources, params):
    """MW extinction in mag at each output filter's mean wavelength.

    SNANA's GALextinct(RV, RV*MWEBV_true, LAMAVG) per band
    (GENLC.MWXT_MAG, snlc_sim.c:13713-13724), written to the TEXT MWXT_MAG
    block. The host magnitudes use their own law; see host_mwxt_mag_per_band.
    """
    rv = simulation.mw_dust.rv
    order = [resources.filter_name_to_index[n] for n in resources.filter_names]
    av_true = rv * jnp.asarray(params['sim_mwebv'])
    axav = resources.mw_law_at_mean.extinction_curve(
        resources.photometry.filter_mean_wavelengths, rv, av=av_true[:, None])
    return av_true[:, None] * axav[..., jnp.asarray(order)]
