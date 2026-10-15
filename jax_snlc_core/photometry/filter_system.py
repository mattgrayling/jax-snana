"""Band weights and band integration, shared by the simulator and fitter.

:func:`compute_single_restframe_filter` builds the weights,
:func:`fluxcal_from_sed` integrates an SED through them, and
:func:`band_window_width` sizes the wavelength window both run on. The
simulator and the fitter reach all three through ``SNModel.evaluate``.
:func:`filter_band` is the single owner of the filter-name-to-band-label
rule.
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from jax_snlc_core.utils.error_handling import SNANAError
from jax_snlc_core.utils.interp import compute_catmullrom_slopes, hermite_interp_1d


def filter_band(name: str) -> str:
    """Band label of a kcor filter name: its last character, as INTFILTER
    takes it (sntools.c:2604)."""
    return name[-1:]


def filter_values(values, names, default):
    """Values in filter order, accepting full names or unambiguous SNANA labels.

    A full name takes precedence. The empty key supplies a value for all bands.
    Keys for inactive filters are retained for later fits but have no effect here.
    """
    labels = [filter_band(name) for name in names]
    result = []
    for name, label in zip(names, labels):
        if name in values:
            result.append(values[name])
        elif label in values:
            if labels.count(label) > 1:
                raise SNANAError(
                    f"Ambiguous filter label {label!r}; use full filter names")
            result.append(values[label])
        else:
            result.append(values.get('', default))
    return result


def mw_attenuation_rest(target_wave_grid, wave_rest, mw_att_obs):
    """Observer-frame Milky Way attenuation ``mw_att_obs``, tabulated on the
    common grid blue-shifted to ``wave_rest``, on the rest-frame grid
    ``target_wave_grid``: Catmull-Rom, and 1 beyond the tabulated range."""
    return jnp.where(
        (target_wave_grid < wave_rest[0]) | (target_wave_grid > wave_rest[-1]), 1.0,
        hermite_interp_1d(target_wave_grid, wave_rest, mw_att_obs,
                          compute_catmullrom_slopes(wave_rest, mw_att_obs)))


def band_weights_at_z(z, filter_wave, filter_trans, model_wave, mw_att=None):
    """Rest-frame weights for all filters at one redshift."""
    return jax.vmap(compute_single_restframe_filter,
                    in_axes=(None, None, 0, None, None))(
        z, filter_wave, filter_trans, model_wave, mw_att)


@jax.jit
def compute_single_restframe_filter(
    z: jnp.ndarray,
    filter_wave_common: jnp.ndarray,
    filter_trans_common: jnp.ndarray,
    target_wave_grid: jnp.ndarray,
    mw_att_obs: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Rest-frame band weights for one filter at one redshift.

    The filter is stored on a common observer-frame wavelength grid
    (``filter_wave_common``); this function blue-shifts to rest frame,
    interpolates onto the simulator's fixed rest-frame wavelength grid,
    normalises the weights to unit total λ·T integral, optionally folds in
    observer-frame MW extinction, and applies the ``1/(1+z)`` photon-energy
    scaling so the returned weights integrate an SED directly to observer-
    frame flux.

    ``z`` is a single redshift value; ``filter_wave_common`` and
    ``filter_trans_common`` are the observer-frame filter wavelengths and
    transmissions on the common grid, each shape ``(n_common,)``;
    ``target_wave_grid`` is the fixed rest-frame wavelength grid for
    interpolation; ``mw_att_obs``, when given, is the MW dust attenuation on
    the observer-frame common grid, shape ``(n_common,)``, applied after
    normalisation so the total flux reduction is preserved.

    Returns the band weights for this (filter, redshift), shape
    ``(len(target_wave_grid),)``.
    """
    wave_obs = filter_wave_common
    z_plus_1 = 1.0 + z

    # Hermite: weights C1 in z
    wave_rest = wave_obs / z_plus_1
    trans_rest = jnp.where(
        (target_wave_grid < wave_rest[0]) | (target_wave_grid > wave_rest[-1]), 0.0,
        hermite_interp_1d(target_wave_grid, wave_rest, filter_trans_common,
                          compute_catmullrom_slopes(wave_rest, filter_trans_common)))

    # normalised before MW extinction
    trans_wave = trans_rest * target_wave_grid
    trans_norm = jnp.maximum(jnp.trapezoid(trans_wave, x=target_wave_grid), 1e-30)

    dlambda = jnp.concatenate([
        jnp.array([target_wave_grid[1] - target_wave_grid[0]]),
        (target_wave_grid[2:] - target_wave_grid[:-2]) / 2.0,
        jnp.array([target_wave_grid[-1] - target_wave_grid[-2]]),
    ])
    band_weights = (trans_rest * target_wave_grid * dlambda) / trans_norm

    if mw_att_obs is not None:
        band_weights = band_weights * mw_attenuation_rest(
            target_wave_grid, wave_rest, mw_att_obs)

    # 1/(1+z) scaling converts integrated rest-frame flux to observer-frame.
    return band_weights / z_plus_1


def band_window_width(wave_rest, filter_wave_common, filter_trans_common,
                      margin: int = 6):
    """Grid points spanned by the widest filter at z = 0, with margin, or
    None where that is the whole grid.

    A filter spans ``[lam_lo, lam_hi] / (1 + z)`` of the rest grid, so its
    point count falls with redshift and z = 0 bounds every event.
    """
    wave_rest = np.asarray(wave_rest)
    wave_obs = np.asarray(filter_wave_common)
    nonzero = np.asarray(filter_trans_common) > 0.0
    covered = nonzero.any(axis=1)
    if not covered.any():
        return None
    # a cell wider each end: interp leaves weight on the zero-side neighbour
    first = np.maximum(nonzero.argmax(axis=1) - 1, 0)
    last = np.minimum(
        nonzero.shape[1] - 1 - nonzero[:, ::-1].argmax(axis=1) + 1,
        nonzero.shape[1] - 1)
    lo, hi = wave_obs[first], wave_obs[last]
    # A band outside the grid at z=0 can redshift fully into it.
    spans = np.ceil((hi - lo) / np.diff(wave_rest).min())
    width = int(spans[covered].max()) + margin
    return None if width >= len(wave_rest) else width


def band_window_starts(band_weights, filter_indices, width):
    """Window start per observation, two points before the filter's non-zero
    run and clipped to the grid: dynamic_slice clamps silently, so an
    out-of-range start would shift the window rather than fail.
    """
    n_wave = band_weights.shape[-1]
    return jnp.clip(jnp.argmax(band_weights != 0.0, axis=-1) - 2,
                    0, n_wave - width)[filter_indices]


def fluxcal_from_sed(sed_at_phase, phases, mask, filter_indices, band_weights,
                     ref_fluxes, zp, scale, width=None, sed_factor=None):
    """FLUXCAL at each (phase, filter), integrating over the filter's window.

    Only the window is read from ``sed_at_phase(phase)``, and every step of
    an SED model is elementwise in wavelength, so XLA never emits the
    arithmetic for the rest. A model that reduced across wavelength would
    lose that speedup.

    ``filter_indices`` must index ``band_weights`` on padded lanes too;
    ``width`` is None for the whole grid; ``sed_factor`` is an optional
    rest-frame ``[n_wavelengths]`` multiplier on the source.
    """
    n_wave = band_weights.shape[-1]
    if width is None:
        width = n_wave
    obs_lo = band_window_starts(band_weights, filter_indices, width)
    obs_band_weights = band_weights[filter_indices]
    obs_ref_fluxes = ref_fluxes[filter_indices]
    fluxcal_scale = jnp.power(10.0, 0.4 * zp) * scale

    def one_obs(phase, lo, obs_bw, obs_ref):
        cut = lambda a: lax.dynamic_slice(a, (lo,), (width,))
        sed = cut(sed_at_phase(phase))
        if sed_factor is not None:
            sed = sed * cut(sed_factor)
        return jnp.sum(sed * cut(obs_bw)) / obs_ref * fluxcal_scale

    return jnp.where(
        mask, jax.vmap(one_obs)(phases, obs_lo, obs_band_weights,
                                obs_ref_fluxes), 0.0)
