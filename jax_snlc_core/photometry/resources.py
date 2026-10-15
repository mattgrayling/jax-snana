"""Shared photometry-resource loader for the simulator and fitter.

`load_photometry_resources` builds the complete set of photometry-
related arrays from a :class:`KcorReader` + a rest-frame wavelength
grid + the list of active filters:

    - rest-frame filter transmissions on the wavelength grid
      (per-filter dict + stacked array), used for rest-frame SED
      integration on the sim side.
    - observer-frame common wavelength + transmission grid
      (stacked), used for observer-frame filter integration by both
      sides.
    - AB reference spectrum from the kcor ``PrimarySED`` HDU (raw
      + interpolated onto the working wavelength grid).
    - per-filter AB integrals (photon-weighted normalised, SNANA
      LAMSTEP/hc convention), with the SNANA ZPoff scaling applied
      (``delta = primary_mag - zpoff_snpot - user_zp``).
    - per-filter mean wavelength in two conventions:
      photon-weighted (``∫T·λ²/∫T·λ``) and SNANA
      transmission-weighted (``∫T·λ/∫T``).
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, List, Mapping, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core.photometry.filter_system import filter_band
from jax_snlc_core.physics.constants import HC_ERG_A
from jax_snlc_core.utils.error_handling import SNANAError
from jax_snlc_core.utils.precision import float_dtype


def resolve_filtlist(filtlist_chars, kcor):
    """Map FILTLIST_FIT single-char shorthand to full filter names using kcor header.

    The kcor primary header has FILT001, FILT002, ... keys declaring full names
    (e.g. ``FILT002 = LSST-g``).  The last char of the name gives the band
    char. A header without NFILTERS or any of the FILTnnn cards it counts is
    an error.

    ``filtlist_chars`` is a string of single-char band names, e.g.
    ``"grizY"``; ``kcor`` is the loaded :class:`KcorReader`. Returns the
    full filter names, e.g. ``['LSST-g', 'LSST-r', ...]``.
    """
    if not isinstance(filtlist_chars, str):
        unknown = set(filtlist_chars) - set(kcor.filter_names)
        if unknown:
            raise SNANAError(f"Unknown filters: {sorted(unknown)}")
        return list(filtlist_chars)
    if filtlist_chars in kcor.filter_names:
        return [filtlist_chars]
    hdr = kcor.header
    cards = ["NFILTERS"]
    if "NFILTERS" in hdr:
        cards += [f"FILT{i:03d}" for i in range(1, hdr["NFILTERS"] + 1)]
    missing = [card for card in cards if card not in hdr]
    if missing:
        raise SNANAError(f"{kcor.kcor_file_path} has no {', '.join(missing)} header card",
                         function_name="resolve_filtlist")
    char_to_full = {}
    for card in cards[1:]:
        full_name = hdr[card].strip()
        if not full_name.startswith('*'):
            char_to_full.setdefault(filter_band(full_name), []).append(full_name)

    result = []
    for ch in filtlist_chars:
        if ch not in char_to_full:
            available = ", ".join(f"'{k}'→{v}" for k, v in sorted(char_to_full.items()))
            raise SNANAError(
                f"FILTLIST_FIT char '{ch}' not found in kcor header. "
                f"Available: {available}", function_name="resolve_filtlist"
            )
        candidates = char_to_full[ch]
        if len(candidates) != 1:
            raise SNANAError(f"Ambiguous filter {ch!r}: {candidates}; use full filter names")
        result.append(candidates[0])
    return result


def filter_mean_wavelength(wave, trans) -> float:
    """Transmission-weighted mean observer wavelength of one filter.

    SNANA's FILTOBS_LAMAVG, sum(dlam*T*lam)/sum(dlam*T) with dlam the central
    difference and one-sided at the ends (kcor.c:2340-2356), which is what
    np.gradient produces. An empty filter falls back to its grid centre.
    """
    wave = np.asarray(wave)
    trans = np.asarray(trans)
    dlam = np.gradient(wave)
    tsum = float(np.sum(dlam * trans))
    if tsum <= 0.0:
        return float(np.mean(wave))
    return float(np.sum(dlam * trans * wave) / tsum)


def _ref_mag_terms(ref_fluxes: jnp.ndarray) -> jnp.ndarray:
    """2.5*log10 of the reference flux, with SNANA's zero-reference branch:
    a filter whose reference integral is zero gets ZP = 0
    (genmag_SEDtools.c:338)."""
    nonzero = ref_fluxes != 0.0
    return jnp.where(
        nonzero, 2.5 * jnp.log10(jnp.where(nonzero, ref_fluxes, 1.0)), 0.0)


@dataclass
class PhotometryResources:
    """Photometry resources shared by the simulator and fitter."""

    filter_names: List[str]

    transmissions: Dict[str, jnp.ndarray]
    transmissions_stacked: jnp.ndarray                  # (n_filters, n_wave)
    common_wave_grid: jnp.ndarray
    common_trans_grid: jnp.ndarray                      # (n_filters, n_common)

    filter_ref_fluxes: jnp.ndarray                      # photon-weighted, normalised
    filter_ref_fluxes_snana: jnp.ndarray                # SNANA LAMSTEP/hc convention
    filter_ref_mag_terms: jnp.ndarray
    filter_ref_mag_terms_snana: jnp.ndarray

    filter_mean_wavelengths: jnp.ndarray                # ∫T·λ / ∫T, FILTOBS_LAMAVG


def load_photometry_resources(
    kcor,
    wavelength_grid: jnp.ndarray,
    filter_names: List[str],
    genmag_off_zp_map: Mapping[str, float] = MappingProxyType({}),
    lam_shift_map: Mapping[str, float] = MappingProxyType({}),
    primary_mag_shift_map: Mapping[str, float] = MappingProxyType({}),
) -> PhotometryResources:
    """Build :class:`PhotometryResources` from an already-loaded
    :class:`KcorReader` (``kcor``), a rest-frame working ``wavelength_grid``
    (Angstroms) the model was resampled to, and ``filter_names`` (the
    filters to process, in the desired index order).

    ``genmag_off_zp_map``, if given, is a ``{short_band: mag_offset}`` map
    from the sim's ``GENMAG_OFF_ZP`` input key, left empty by the fitter
    (which has no equivalent config). ``lam_shift_map``, if given, is
    a ``{short_band: wavelength_shift}`` map added to the filter's
    wavelength grid before use. ``primary_mag_shift_map``, if given, is
    a ``{short_band: mag_shift}`` map from the fitter's
    ``MAGOBS_SHIFT_PRIMARY``, added to the kcor primary magnitude. These
    last two may also key a full filter name, which wins over its band.
    """
    if not filter_names:
        raise SNANAError(
            "load_photometry_resources: filter_names is empty.",
            function_name="load_photometry_resources",
        )

    def _shift(m: Mapping[str, float], name: str) -> float:
        # full name (SALT2.INFO) first, then the band (namelist)
        if name in m:
            return float(m[name])
        return float(m.get(filter_band(name), 0.0))

    def _lam_shift(name: str) -> float:
        return _shift(lam_shift_map, name)

    wavelength_grid = jnp.asarray(wavelength_grid)
    n_wave = int(wavelength_grid.shape[0])

    raw_filters = []
    for name in filter_names:
        w, t = kcor.get_filter_transmission(name)
        raw_filters.append((jnp.asarray(w, dtype=float_dtype()) + _lam_shift(name),
                            jnp.asarray(t, dtype=float_dtype())))
    raw_filters_map = dict(zip(filter_names, raw_filters))

    transmissions: Dict[str, jnp.ndarray] = {
        name: jnp.interp(wavelength_grid, *raw_filters_map[name],
                         left=0.0, right=0.0)
        for name in filter_names
    }
    transmissions_stacked = jnp.stack(
        [transmissions[name] for name in filter_names], axis=0)

    min_wave = float(min(w.min() for w, _ in raw_filters))
    max_wave = float(max(w.max() for w, _ in raw_filters))
    common_wave_grid = jnp.linspace(min_wave, max_wave, n_wave)
    common_trans_list = []
    for name in filter_names:
        wave_obs, trans_obs = raw_filters_map[name]
        common_trans_list.append(jnp.interp(
            common_wave_grid, wave_obs, trans_obs, left=0.0, right=0.0))
    common_trans_grid = jnp.stack(common_trans_list, axis=0)

    primary_seds_cache: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]] = {}
    for name in filter_names:
        primary_name = kcor.get_filter_primary(name)
        if primary_name not in primary_seds_cache:
            pwaves, pflux = kcor.get_primary_sed(primary_name)
            primary_seds_cache[primary_name] = (
                jnp.asarray(pwaves), jnp.asarray(pflux))

    filter_ref_fluxes_list: List[float] = []
    filter_ref_fluxes_snana_list: List[float] = []
    mean_wavelengths: List[float] = []

    for name, (wave_obs_np, trans_obs_np) in zip(filter_names, raw_filters):
        wave_obs = jnp.asarray(wave_obs_np)
        trans_obs = jnp.asarray(trans_obs_np)

        primary_name = kcor.get_filter_primary(name)
        prim_waves, prim_flux = primary_seds_cache[primary_name]
        ref_obs = jnp.interp(
            wave_obs, prim_waves, prim_flux, left=0.0, right=0.0)

        transmission_wave_obs = trans_obs * wave_obs
        transmission_norm_obs = jnp.trapezoid(transmission_wave_obs, wave_obs)
        ref_integrand_obs = ref_obs * transmission_wave_obs
        ref_flux = (
            jnp.trapezoid(ref_integrand_obs, wave_obs) / transmission_norm_obs)
        filter_ref_fluxes_list.append(ref_flux)

        # SNANA-convention reference flux: rectangle rule × LAMSTEP / hc.
        lamstep_obs = jnp.median(jnp.diff(wave_obs))
        ref_integrand_snana = ref_obs * trans_obs * wave_obs
        ref_flux_snana = jnp.sum(ref_integrand_snana) * lamstep_obs / HC_ERG_A
        filter_ref_fluxes_snana_list.append(ref_flux_snana)

        mean_wavelengths.append(filter_mean_wavelength(
            np.asarray(wave_obs_np), np.asarray(trans_obs_np)))

    filter_ref_fluxes_array = jnp.asarray(filter_ref_fluxes_list)
    filter_ref_fluxes_snana_array = jnp.asarray(filter_ref_fluxes_snana_list)

    # delta = magprimary - ZPoff(SNpot) - user_zp (genmag_SEDtools.c:338, snlc_sim.c:8487, sntools_calib.c:1028)
    scale_per_filter: List[float] = []
    for name in filter_names:
        short_band = filter_band(name)
        missing = [label for label, table in
                   (('Primary Mag', kcor.filter_primary_mag),
                    ('ZPoff(SNpot)', kcor.filter_zpoff_snpot))
                   if name not in table]
        if missing:
            raise SNANAError(
                f"KCOR ZPoff table has no {', '.join(missing)} entry for active "
                f"filter '{name}'; SNANA reads these columns for every filter.",
                function_name="load_photometry_resources")
        primary_mag = (float(kcor.filter_primary_mag[name])
                       + _shift(primary_mag_shift_map, name))
        zpoff_snpot = float(kcor.filter_zpoff_snpot[name])
        user_zp = float(genmag_off_zp_map.get(short_band, 0.0))
        delta = primary_mag - zpoff_snpot - user_zp
        scale_per_filter.append(float(np.power(10.0, 0.4 * delta)))
    zp_scale = jnp.asarray(scale_per_filter)
    filter_ref_fluxes_array = filter_ref_fluxes_array * zp_scale
    filter_ref_fluxes_snana_array = filter_ref_fluxes_snana_array * zp_scale

    filter_ref_mag_terms = _ref_mag_terms(filter_ref_fluxes_array)
    filter_ref_mag_terms_snana = _ref_mag_terms(filter_ref_fluxes_snana_array)

    return PhotometryResources(
        filter_names=list(filter_names),
        transmissions=transmissions,
        transmissions_stacked=transmissions_stacked,
        common_wave_grid=common_wave_grid,
        common_trans_grid=common_trans_grid,
        filter_ref_fluxes=filter_ref_fluxes_array,
        filter_ref_fluxes_snana=filter_ref_fluxes_snana_array,
        filter_ref_mag_terms=filter_ref_mag_terms,
        filter_ref_mag_terms_snana=filter_ref_mag_terms_snana,
        filter_mean_wavelengths=jnp.asarray(mean_wavelengths),
    )


def read_calib_shifts(path):
    """MAGSHIFT/WAVESHIFT/LAMSHIFT entries of a model metadata file, as
    ('mag'|'lam', survey_csv, filter, shift) tuples (genmag_SALT2.c:1307)."""
    shifts = []
    with open(path) as fh:
        for line in fh:
            key, _, value = line.split('#')[0].partition(':')
            if key.strip() in ('MAGSHIFT', 'WAVESHIFT', 'LAMSHIFT'):
                survey, filt, shift = value.split()
                shifts.append(('mag' if key.strip() == 'MAGSHIFT' else 'lam',
                               survey, filt, float(shift)))
    return shifts


def resolve_calib_shifts(shifts, filter_names, filter_survey, enable_mag,
                         enable_wave):
    """Calibration shifts as {full_filter_name: shift} pairs.

    Mirrors match_SALT2train (genmag_SALT2.c:1915): the filter's own survey is
    itself a comma-separated list, names are truncated at '/', and a
    single-character calib filter collapses both sides to the trailing band.
    """
    mag_out, lam_out = {}, {}
    for which, survey_csv, filt_calib, shift in shifts:
        if which == 'mag' and not enable_mag:
            continue
        if which == 'lam' and not enable_wave:
            continue
        calib_base = filt_calib.split('/')[0]
        for name in filter_names:
            surveys = [x.strip() for x in
                       str(filter_survey.get(name, 'UNDEFINED')).split(',')]
            if not any(s in surveys for s in survey_csv.split(',')):
                continue
            genmag_base = name.split('/')[0]
            if len(filt_calib) == 1:
                genmag_base, calib_cmp = name[-1], filt_calib[-1]
            else:
                calib_cmp = calib_base
            if genmag_base != calib_cmp:
                continue
            out = mag_out if which == 'mag' else lam_out
            out[name] = out.get(name, 0.0) + shift
    return mag_out, lam_out


jax.tree_util.register_dataclass(
    PhotometryResources,
    data_fields=[
        'transmissions',
        'transmissions_stacked',
        'common_wave_grid',
        'common_trans_grid',
        'filter_ref_fluxes',
        'filter_ref_fluxes_snana',
        'filter_ref_mag_terms',
        'filter_ref_mag_terms_snana',
        'filter_mean_wavelengths',
    ],
    meta_fields=[
        'filter_names',
    ],
)
