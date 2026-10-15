"""The pieces of flux evaluation the fitter's kernels share with the
photo-z grid: :func:`fit_flux`, the model flux as the chi2 sees it, and
:func:`band_weights`, an event's rest-frame band weights at one redshift,
each set of which :func:`rest_band_weights` gives alone.
The evaluation itself is :meth:`LightCurveFitter.model_flux` and
:meth:`LightCurveFitter.model_fracerr`.
"""

import jax.numpy as jnp

from jax_snlc_core.photometry.filter_system import band_weights_at_z


def fit_flux(flux, zp_fluxcal):
    """Keep positive model flux up to SNANA's bright limit (magnitude 5).

    Unlike SNANA's magnitude-40 cutoff, arbitrarily faint positive flux is
    retained to avoid a discontinuity in the fitting objective. Nonpositive
    phase extrapolations still contribute zero flux and zero model error.
    """
    hi = 10.0 ** (-0.4 * (5.0 - zp_fluxcal))
    return jnp.where((flux > 0.0) & (flux < hi), flux, 0.0)


def band_weights(static_data, z, mw_att):
    """(flux weights, error weights), each [n_filters, n_wave], of one event
    at redshift ``z``: with its observer-frame Milky Way attenuation
    ``mw_att`` [n_common] folded in, and without it for the error
    denominators (Finteg_filter vs Finteg_forErr, genmag_SALT2.c:2867-2868).
    """
    return (rest_band_weights(static_data, z, mw_att),
            rest_band_weights(static_data, z, None))


def rest_band_weights(static_data, z, mw_att):
    """One set of :func:`band_weights`, [n_filters, n_wave]: with the Milky
    Way attenuation ``mw_att`` folded in, or without it when that is None."""
    return band_weights_at_z(
        z, static_data['common_filter_wave_grid'],
        static_data['common_filter_trans_grid'],
        static_data['wavelength_grid'], mw_att)
