"""Prepare host selection, draw galaxies and calculate positions and neighbours."""


import logging
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import ndtr

from jax_snlc_core import logging as log
from jax_snlc_core.cosmology import cmb_vdotn, cosmo_params, eq_to_gal
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import MXHOSTGAL, SNANA_NULLFLOAT, canonical_property_name
from jax_snlc_core.utils import LIGHT_KM, eval_genpoly
from jax_snlc_sim.hostlib.catalogue import HostCatalogue, neighbour_offsets
from jax_snlc_sim.hostlib.host_selection_weights import build_host_weights
from jax_snlc_sim.hostlib.photoz import catalogue_zphot_shift, shifted_catalogue_zphot
from jax_snlc_sim.hostlib.sersic import compute_sersic_integral_table, generate_sn_position
from jax_snlc_sim.rate import dndz_density_on_grid, neutral_rate_mods
from jax_snlc_sim.simlib import ZMIN_SIMLIB_REGEN
from jax_snlc_sim.simulation import HOSTLIB_MSKOPT_ZPHOT_QGAUSS

logger = logging.getLogger(__name__)


MINDAYSEP_SAMEGAL_MAX = 9999


VPEC_NSIGMA = 5.0


CELLS_PER_DZTOL = 200


def resolve_igal_force(table, galid: int) -> int:
    """HOSTLIB_GALID_FORCE as a galaxy index of ``table``; -1 when unset."""
    if galid < 0:
        return -1
    galid_array = table.columns['GALID']
    matches = np.where(galid_array == galid)[0]
    if len(matches) == 0:
        raise SNANAError(
            f"HOSTLIB_GALID_FORCE={galid} not found in HOSTLIB",
            function_name="resolve_igal_force"
        )
    igal = int(matches[0])
    log.line(logger, f"IGAL_FORCE: GALID={galid} → igal={igal}")
    return igal


def build_galid_priority(table, galid_min: int, galid_max: int,
                         z_helio_bounds):
    """Set GALID priority range for preferential selection.

    ``galid_min``/``galid_max`` are inclusive; ``z_helio_bounds`` is an
    optional (lo, hi) heliocentric-z window. Returns the priority lookup,
    or None when the range is empty.
    """
    if not 0 < galid_min <= galid_max:
        return None

    galid_array = table.columns['GALID']
    ztrue_array = table.columns['ZTRUE']

    mask = (galid_array >= galid_min) & (galid_array <= galid_max)
    if z_helio_bounds is not None:
        z_lo, z_hi = z_helio_bounds
        mask &= (ztrue_array >= z_lo) & (ztrue_array <= z_hi)
    priority_igals = np.where(mask)[0]

    priority = {
        'igals': priority_igals,        # sorted by z (inherits from HOSTLIB sort)
        'redshifts': ztrue_array[mask],
    }
    log.line(logger, f"GALID_PRIORITY: [{galid_min}, {galid_max}], "
             f"{len(priority_igals)} galaxies in range")
    return priority


def build_detection_filter(table, mode, thresholds, snr_scale_config):
    """Configure galaxy detection filtering for NBR_LIST neighbours.

    Bands are discovered dynamically from HOSTLIB VARNAMES: any column
    matching ``*_obs`` (with a corresponding ``*_obs_err`` for SNR mode)
    is treated as a photometric band. This mirrors SNANA's approach of
    using whatever bands the HOSTLIB provides rather than hardcoding a
    specific filter set.

    ``mode`` is 'snr' or 'mag'; ``thresholds`` is a list of floats, already
    sorted (descending for SNR, ascending for MAG); ``snr_scale_config`` is
    a list of (mjdmin, mjdmax, value) tuples, or None. Returns ``(columns,
    thresholds, raw_errs, snr_scale_ranges)``.
    """
    sorted_data = table.columns

    all_varnames = list(sorted_data.keys())
    band_names = [v for v in all_varnames
                  if v.endswith('_obs') and not v.endswith('_obs_err')]
    err_names = [b + '_err' for b in band_names]
    if mode == 'snr':
        paired = [(b, e) for b, e in zip(band_names, err_names) if e in sorted_data]
        if not paired:
            raise SNANAError(
                "HOSTLIB_SNR_DETECT requires *_obs and *_obs_err columns in HOSTLIB")
        band_names, err_names = zip(*paired)
        band_names, err_names = list(band_names), list(err_names)

    n_bands = len(band_names)
    if n_bands == 0:
        raise SNANAError(
            f"HOSTLIB_{mode.upper()}_DETECT: no *_obs columns found in HOSTLIB")

    if len(thresholds) > n_bands:
        raise SNANAError(
            f"HOSTLIB_{mode.upper()}_DETECT has {len(thresholds)} thresholds "
            f"but only {n_bands} bands available")

    if mode == 'snr':
        snr_factor = 2.5 / jnp.log(10.0)
        snr_cols = []
        for ename in err_names:
            errs = jnp.array(sorted_data[ename], dtype=jnp.float32)
            errs = jnp.where((errs <= 0) | (errs > 9.0), 9.0, errs)
            snr_cols.append(snr_factor / errs)
        detect_columns = jnp.stack(snr_cols, axis=0)
        detect_thresholds = jnp.array(thresholds, dtype=jnp.float32)
        snr_scale_ranges = snr_scale_config
        if snr_scale_config is not None:
            raw_err_cols = []
            for ename in err_names:
                raw_err_cols.append(jnp.array(sorted_data[ename], dtype=jnp.float32))
            detect_raw_errs = jnp.stack(raw_err_cols, axis=0)
        else:
            detect_raw_errs = jnp.zeros((0, table.ngal), dtype=jnp.float32)
        log.line(logger, f"Galaxy detection: SNR mode, {n_bands} bands, "
                 f"thresholds={thresholds}")

    elif mode == 'mag':
        cols = []
        for name in band_names:
            cols.append(-jnp.array(sorted_data[name], dtype=jnp.float32))
        detect_columns = jnp.stack(cols, axis=0)
        # negated, so -mag >= -threshold
        detect_thresholds = jnp.array([-t for t in thresholds], dtype=jnp.float32)
        snr_scale_ranges = None
        detect_raw_errs = jnp.zeros((0, table.ngal), dtype=jnp.float32)
        log.line(logger, f"Galaxy detection: MAG mode, {n_bands} bands, "
                 f"thresholds={thresholds}")

    return detect_columns, detect_thresholds, detect_raw_errs, snr_scale_ranges


def _stack_catalogue_columns(table, names):
    """The ``names`` columns of ``table`` packed into one (n_vars,
    n_galaxies) float32 array for fused gathers."""
    return jnp.stack([jnp.asarray(table.columns[v], dtype=jnp.float32)
                      for v in names], axis=0)


def _apply_abmag(stacked_columns, varnames, offset, force):
    """HOSTLIB_ABMAG_FORCE (replace) then HOSTLIB_ABMAG_OFFSET (add) on the
    X_obs galaxy-magnitude columns (sntools_host.c:3531)."""
    if force <= -8.0 and offset == 0.0:
        return stacked_columns
    for i, name in enumerate(varnames):
        if name.endswith('_obs') and len(name) == 5:
            col = stacked_columns[i]
            if force > -8.0:
                col = jnp.full_like(col, force)
            col = col + offset
            stacked_columns = stacked_columns.at[i].set(col)
    return stacked_columns


_NEIGHBOUR_SENTINEL_OVERRIDES = {
    'RA_GAL': -999.0, 'DEC_GAL': -999.0, 'OBJID2': 0.0,
    'LOGMASS': -9999.0, 'LOGMASS_OBS': -9999.0, 'LOGMASS_ERR': -9999.0,
    'LOGSFR': -9999.0, 'LOGSFR_OBS': -9999.0, 'LOGSFR_ERR': -9999.0,
    'LOGsSFR': -9999.0, 'LOGsSFR_OBS': -9999.0, 'LOGsSFR_ERR': -9999.0,
    'ELLIPTICITY': -9999.0, 'SQRADIUS': -9999.0,
}


def _neighbour_sentinels(varnames):
    """Per-column sentinel vector for masking an unmatched neighbour slot."""
    return jnp.array(
        [_NEIGHBOUR_SENTINEL_OVERRIDES.get(vn, -9.0) for vn in varnames],
        dtype=jnp.float32)


def _pack_sersic_components(table):
    """Packed Sersic components (n_comp, ngal), z-sorted. FIXSERSIC and
    SCALE_SERSIC_SIZE are written into the catalogue columns at load
    (apply_fixsersic_columns), so the profiles already carry both and the
    position, DDLR and GALMAG paths share one set of sizes.

    Returns None when the HOSTLIB declares no profile; positions then take
    sentinel values (GEN_SNHOST_POS bails at NPROF=0,
    sntools_host.c:7621-7635).
    """
    if table.sersic_profiles is None:
        return None
    prof = table.sersic_profiles
    return {
        'a': jnp.asarray(prof['a'].T, dtype=jnp.float32),
        'b': jnp.asarray(prof['b'].T, dtype=jnp.float32),
        'n': jnp.asarray(prof['n'].T, dtype=jnp.float32),
        'w': jnp.asarray(prof['w'].T),
        'a_rot': jnp.asarray(prof['a_rot'], dtype=jnp.float32),
    }


def _marginal_in_range_mask(table, bit_8, genrange_ra, genrange_decl,
                            genrange_b):
    """Per-galaxy in-bounds mask for the marginal-CDF selection mode.

    With MSKOPT bit 8 (SN2GAL_RADEC) the host coordinates become the SN
    coordinates, so only galaxies inside GENRANGE_RA/DECL/b are viable.
    """
    in_range = np.ones(table.ngal, dtype=np.float32)
    if not bit_8:
        return in_range
    ra_gal = table.columns['RA_GAL']
    dec_gal = table.columns['DEC_GAL']
    _, b_arr = eq_to_gal(ra_gal, dec_gal)
    host_b = np.asarray(b_arr)
    ra_min, ra_max = genrange_ra
    full_ra = (ra_max - ra_min) >= 360.0
    ra_mod = np.mod(ra_gal, 360.0)
    lo, hi = np.mod(ra_min, 360.0), np.mod(ra_max, 360.0)
    if full_ra:
        ra_ok = np.ones_like(ra_gal, dtype=bool)
    elif lo <= hi:
        ra_ok = (ra_mod >= lo) & (ra_mod <= hi)
    else:
        ra_ok = (ra_mod >= lo) | (ra_mod <= hi)
    dec_ok = (dec_gal >= genrange_decl[0]) & (dec_gal <= genrange_decl[1])
    b_ok = (host_b >= genrange_b[0]) & (host_b <= genrange_b[1])
    return (ra_ok & dec_ok & b_ok).astype(np.float32)


class ZhelGrid(NamedTuple):
    """Cells at ``log(1 + zHEL) = u_lo + step * k`` for ``-n_margin <= k <= n_inner + n_margin``, the margins for the peculiar-velocity smear."""
    u_lo: float
    n_inner: int
    step: float
    n_margin: int

    def u(self):
        """Each cell's ``log(1 + zHEL)``."""
        return self.u_lo + self.step * np.arange(
            -self.n_margin, self.n_inner + self.n_margin + 1)

    def z_reach(self):
        """The lowest and highest cell zHEL."""
        return (np.expm1(self.u_lo - self.step * self.n_margin),
                np.expm1(self.u_lo + self.step * (self.n_inner + self.n_margin)))


def zhel_grid(z_draw, dztol_coeffs, vpec):
    """The :class:`ZhelGrid` over ``z_draw`` moved and smeared by the peculiar velocity ``vpec = (shift, sigma)`` in km/s, CELLS_PER_DZTOL cells across the narrowest DZTOL."""
    v_shift, v_sigma = vpec
    shift = np.log1p(v_shift / LIGHT_KM)
    u_lo = np.log1p(z_draw[0]) + shift
    u_hi = np.log1p(z_draw[1]) + shift
    margin = VPEC_NSIGMA * np.log1p(v_sigma / LIGHT_KM)
    span = np.expm1(np.linspace(u_lo - margin, u_hi + margin, 1001))
    dztol = eval_genpoly(np.asarray(dztol_coeffs, dtype=np.float64), span)
    if dztol.min() <= 0.0:
        raise SNANAError(
            f"HOSTLIB_DZTOL is {dztol.min():.5f} at zHEL="
            f"{span[np.argmin(dztol)]:.5f}, so no galaxy can be drawn there.",
            function_name="zhel_grid")
    step = dztol.min() / (CELLS_PER_DZTOL * (1.0 + span[-1]))
    n_margin = int(np.ceil(margin / step))
    return ZhelGrid(u_lo, int(np.ceil((u_hi - u_lo) / step)), step, n_margin)


def refuse_empty_windows(galaxy_z, dztol_coeffs, z_reach, drawable):
    """Abort when a zHEL in ``z_reach`` has no galaxy marked in some row of ``drawable`` strictly within DZTOL. Between two neighbouring such galaxies the window is emptiest at their midpoint, clipped to ``z_reach``, while DZTOL's slope in z lies within (-1, 1), so a steeper DZTOL in ``z_reach`` aborts too."""
    z_lo, z_hi = z_reach
    slope = np.polynomial.Polynomial(np.asarray(dztol_coeffs, dtype=np.float64)).deriv()
    turning = slope.deriv().roots()
    turning = turning.real[(turning.imag == 0) & (turning.real > z_lo) & (turning.real < z_hi)]
    at = np.concatenate([[z_lo, z_hi], turning])
    steep = np.abs(slope(at)) >= 1.0
    if steep.any():
        k = int(np.argmax(steep))
        raise SNANAError(
            f"HOSTLIB_DZTOL has slope {slope(at[k]):.4f} at zHEL={at[k]:.5f}, "
            f"which the SN2GAL_Z draw can reach; the empty-window check needs "
            f"a slope between -1 and 1.",
            function_name="refuse_empty_windows")
    marked = np.pad(drawable, ((0, 0), (1, 1)), constant_values=True)
    z = np.concatenate([[-np.inf], galaxy_z, [np.inf]])
    row, col = np.divmod(np.flatnonzero(marked), marked.shape[1])
    pair = row[:-1] == row[1:]
    lo, hi = z[col[:-1][pair]], z[col[1:][pair]]
    reached = (hi > z_lo) & (lo < z_hi)
    lo, hi = lo[reached], hi[reached]
    # fmax takes the reach's edge where both neighbours are absent
    at = np.fmin(np.fmax(0.5 * (lo + hi), np.maximum(lo, z_lo)),
                 np.minimum(hi, z_hi))
    dztol = eval_genpoly(np.asarray(dztol_coeffs, dtype=np.float64), at)
    empty = np.minimum(at - lo, hi - at) >= dztol
    if empty.any():
        k = int(np.argmax(empty))
        raise SNANAError(
            f"No HOSTLIB galaxy to draw within DZTOL={dztol[k]:.5f} "
            f"of zHEL={at[k]:.5f}, which the SN2GAL_Z draw can reach. Loosen "
            f"HOSTLIB_DZTOL or extend the HOSTLIB.",
            function_name="refuse_empty_windows")


def zhel_cells(z_draw, dndz_density, dndz_z_grid, vpec, grid):
    """``(zHEL, probability)`` on ``grid``: dN/dz over ``z_draw``, a point when it has zero width, moved and smeared by the peculiar velocity ``vpec = (shift, sigma)`` in km/s."""
    v_shift, v_sigma = vpec
    shift = np.log1p(v_shift / LIGHT_KM)
    step, n_margin = grid.step, grid.n_margin
    u = grid.u()

    z_edge = np.expm1(np.concatenate([u - 0.5 * step, u[-1:] + 0.5 * step])
                      - shift)
    if z_draw[0] == z_draw[1]:
        cdf = (z_edge >= z_draw[0]).astype(np.float64)
    else:
        cum = np.concatenate([[0.0], np.cumsum(
            0.5 * (dndz_density[1:] + dndz_density[:-1]) * np.diff(dndz_z_grid))])
        cdf = np.interp(np.clip(z_edge, *z_draw), dndz_z_grid, cum)
    prob = np.diff(cdf)

    if n_margin > 0:
        offset = step * np.arange(-n_margin - 0.5, n_margin + 1.0)
        kernel = np.diff(ndtr(LIGHT_KM * np.expm1(offset) / v_sigma))
        prob = np.convolve(prob, kernel, mode='same')
    return np.expm1(u), prob / prob.sum()


def _build_marginal_cdf(galaxy_redshifts, dztol_coeffs, in_range, z_cells,
                        cell_prob, z_range, wgtsum):
    """Per-(SN-variable bin, galaxy) marginal CDF for MSKOPT bit 16.

    Weights each galaxy by its raw WGTMAP weight times its chance of the
    DZTOL-window draw summed over the zHEL cells ``z_cells`` (probabilities
    ``cell_prob``): a cell whose window strictly contains the galaxy
    contributes its probability divided by the total weight in that window,
    and a window whose galaxies all have zero weight gives its whole
    probability to its first galaxy, as :func:`draw_host_indices` does. Only
    galaxies inside ``z_range`` and ``in_range`` count. Each row is
    normalised to end at 1, and an SN-variable bin with no such galaxy
    aborts.
    """
    z_gal = np.asarray(galaxy_redshifts, dtype=np.float64)
    wgtsum_cum = np.asarray(wgtsum, dtype=np.float64)
    n_bins, n_gal = wgtsum_cum.shape
    padded = np.concatenate([np.zeros((n_bins, 1)), wgtsum_cum], axis=1)

    keep = cell_prob > 0
    z, prob = z_cells[keep], cell_prob[keep]
    dztol = eval_genpoly(np.asarray(dztol_coeffs, dtype=np.float64), z)
    first = np.searchsorted(z_gal, z - dztol, side='right')
    end = np.searchsorted(z_gal, z + dztol, side='left')
    inside = padded[:, end] - padded[:, first]
    share = np.divide(prob, inside, out=np.zeros_like(inside),
                      where=inside > 0).ravel()
    unweighted = np.where(inside > 0, 0.0, prob).ravel()
    row = (np.arange(n_bins)[:, None] * (n_gal + 1))
    size = n_bins * (n_gal + 1)
    steps = (np.bincount((row + first).ravel(), share, size)
             - np.bincount((row + end).ravel(), share, size))
    reach = np.cumsum(steps.reshape(n_bins, n_gal + 1), axis=1)[:, :n_gal]
    first_of_unweighted = np.bincount((row + first).ravel(), unweighted,
                                      size).reshape(n_bins, n_gal + 1)[:, :n_gal]

    zmin, zmax = z_range
    in_z = (z_gal >= zmin) & (z_gal <= zmax) & (in_range > 0)
    weights = (np.diff(padded, axis=1) * reach + first_of_unweighted) * in_z
    totals = weights.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise SNANAError(
            f"HOSTLIB marginal CDF: {int((totals <= 0).sum())} WGTMAP "
            f"SN-variable bin(s) draw no galaxy inside GENRANGE_REDSHIFT. "
            f"Widen GENRANGE_REDSHIFT, loosen HOSTLIB_DZTOL, or extend the "
            f"HOSTLIB.",
            function_name="_build_marginal_cdf",
        )
    cdf = np.minimum(np.cumsum(weights, axis=1) / totals, 1.0)
    return jnp.array(cdf, dtype=jnp.float32)


def marginal_z_helio_bounds(z_range, vdotn_range):
    """A zCMB window ``z_range`` moved to heliocentric by the extreme CMB
    dipole terms in ``vdotn_range`` (lowest, highest): the heliocentric
    redshift range SN2GAL_Z's marginal CDF draws from, for GENRANGE_REDSHIFT
    over all fields or for one SIMLIB entry's redshift window."""
    vdotn_lo, vdotn_hi = vdotn_range
    zmin, zmax = z_range
    return ((1.0 + zmin) * (1.0 - vdotn_lo) - 1.0,
            (1.0 + zmax) * (1.0 - vdotn_hi) - 1.0)


def build_host_selection(table, stacked, extra, wgtsum, weights, snmagshift,
                         snvar_names, snvar_bin_centers, *, dztol_coeffs,
                         useonce, hostlib_mskopt, genrange_ra, genrange_decl,
                         genrange_b, dndz_density, dndz_density_pec1a,
                         dndz_z_grid, z_range, vpec,
                         field_vdotn, maxddlr, mindaysep,
                         smear_sersic, position_settings, abmag, igal_force,
                         galid_priority, detection_filter, groupid_wgtsums,
                         field_groupid_set, field_z_window):
    """The host draw's settings and the catalogue it draws from, as
    ``(HostSelector, HostCatalogue)``. Every knob comes from the input file,
    so none of them has a default here.

    ``table`` is the HOSTLIB as read (:class:`HostlibTable`); ``stacked``
    names its columns the draw gathers into a host's properties, and
    ``extra`` those read by name (photo-z, quantiles, STOREPAR, the
    output-only columns), kept out of the properties so the photo-z guards
    see only what the draw sets. ``wgtsum``, ``weights``, ``snmagshift`` and
    ``snvar_*`` are the weight-sum arrays. ``dztol_coeffs`` is HOSTLIB_DZTOL
    as ascending polynomial coefficients, so (0.002, 0.04) is dztol =
    0.002 + 0.04*z. ``useonce`` is HOSTLIB_MSKOPT bit 32, a galaxy hosts at most one SN.
    ``mindaysep`` is (days, PEAKMJD range) for HOSTLIB_MINDAYSEP_SAMEGAL,
    or None when a galaxy may be reused freely. ``groupid_wgtsums`` is
    ``{set_index: (n_bins, n_galaxies) cumsum}`` of GROUPID-filtered
    weights and ``field_groupid_set`` each SIMLIB entry's set index (-1
    for none); an event draws from its entry's set (SNANA enforces GROUPID
    in the weight search, sntools_host.c:6379-6399).

    Under SN2GAL_Z (HOSTLIB_MSKOPT bit 16) the marginal CDF is built from
    the dN/dz on ``dndz_z_grid`` (``dndz_density``, and
    ``dndz_density_pec1a`` for the DNDZ_PEC1A rate, or None) over
    ``z_range``, moved to heliocentric by ``field_vdotn`` (each SIMLIB
    entry's CMB dipole term) and smeared by ``vpec``, ``(shift, sigma)`` in
    km/s. ``field_z_window`` is each SIMLIB entry's redshift window; an
    entry with one draws from its own blocks of the CDF.
    """
    ngal = table.ngal
    dztol = jnp.array(dztol_coeffs, dtype=jnp.float32)
    if smear_sersic > 0:
        log.line(logger, f"SMEAR_SERSIC = {smear_sersic:.3f} arcsec (FWHM)")

    # init_SAMEHOST returns first under USEONCE (sntools_host.c:5663)
    uses_mindaysep = mindaysep is not None and not useonce
    mindaysep_days, peakmjd_offset = None, 0
    if uses_mindaysep:
        mindaysep_days, mjd_range = mindaysep
        peakmjd_offset = int(mjd_range[0])
        logger.info(
            f"  MINDAYSEP_SAMEGAL={mindaysep_days}: galaxy reuse allowed if "
            f"peak MJD separation >= {mindaysep_days}d")

    has_snvars = snvar_names is not None and len(snvar_names) > 0
    # (n_bins, n_galaxies) even without SN variables
    wgtsum = jnp.array(wgtsum)
    if wgtsum.ndim == 1:
        wgtsum = wgtsum[jnp.newaxis, :]
    redshifts = jnp.array(table.columns['ZTRUE'])

    varnames = [v for v in dict.fromkeys(map(canonical_property_name, stacked))
                if v in table.columns and v != 'GALID']
    columns = _apply_abmag(_stack_catalogue_columns(table, varnames),
                           varnames, *abmag)
    has_neighbours = bool(int(table.neighbours.max()) >= 0)
    use_dlr_columns = 'a_DLR' in varnames and 'b_DLR' in varnames
    if use_dlr_columns:
        log.line(logger, "Using a_DLR/b_DLR columns for DLR (overrides weighted Sersic)")

    # SNANA gives all-zero DDLR without a profile (sntools_host.c:8495); jax refuses
    if has_neighbours and table.sersic_profiles is None:
        raise SNANAError(
            "HOSTLIB has NBR_LIST but no Sersic profile columns; DDLR neighbour "
            "matching requires a{j}_Sersic/b{j}_Sersic morphology.",
            function_name="build_host_selection")

    if detection_filter is not None:
        (detect_columns, detect_thresholds, detect_errs,
         snr_scale_ranges) = build_detection_filter(table, *detection_filter)
    else:
        detect_columns = jnp.zeros((0, ngal), dtype=jnp.float32)
        detect_thresholds = jnp.zeros((0,), dtype=jnp.float32)
        detect_errs = jnp.zeros((0, ngal), dtype=jnp.float32)
        snr_scale_ranges = None

    log.line(logger, f"HostSelector ready: {ngal:,} galaxies")
    log.line(logger, f"Redshift range: {float(redshifts[0]):.4f} - "
             f"{float(redshifts[-1]):.4f}")
    if has_snvars:
        log.line(logger, f"WGTMAP has SN variables: {snvar_names}")
        log.line(logger, f"Multi-bin WGTSUM: {wgtsum.shape[0]} bins × {ngal:,} galaxies")
    else:
        log.line(logger, "WGTMAP has no SN variables (galaxy properties only)")
    if useonce:
        log.line(logger, "USEONCE enabled: Each galaxy assigned to at most one SN")
    _log_dztol(dztol_coeffs)

    # one block of SN-variable bins per GROUPID set after the default block 0
    groupid_order = tuple(sorted(int(k) for k in groupid_wgtsums or ()))
    set_wgtsums = [jnp.asarray(groupid_wgtsums[k]) for k in groupid_order]
    field_block = np.zeros(len(field_z_window), dtype=np.int32)
    if groupid_order:
        slot = {k: i + 1 for i, k in enumerate(groupid_order)}
        field_block = np.array(
            [slot.get(int(k), 0) for k in field_groupid_set], dtype=np.int32)

    bit_8 = bool(hostlib_mskopt & 8)
    bit_16 = bool(hostlib_mskopt & 16)
    narrow_radec = (genrange_ra != (-360.0, 360.0)
                    and genrange_ra != (-180.0, 180.0)) \
        or genrange_decl != (-90.0, 90.0) \
        or genrange_b != (-90.0, 90.0)
    if bit_8 and not bit_16 and narrow_radec:
        raise SNANAError(
            "HOSTLIB_MSKOPT bit 8 (SN2GAL_RADEC) without bit 16 "
            "(SN2GAL_Z) combined with a narrow GENRANGE_RA/DECL/b is "
            "not supported in jax_SNANA: host RA/DEC can drift "
            "outside the configured bounds. Set bit 16 to also "
            "transfer the host's z (enabling pre-emptive bounds "
            "filtering), or widen GENRANGE_RA/DECL/b to defaults.",
            function_name="build_host_selection",
        )
    marginal_cdf, pec1a_row_shift, z_helio_bounds = None, None, None
    window_blocks = {}
    if bit_16 and z_range is not None and dndz_density is not None:
        in_range = _marginal_in_range_mask(
            table, bit_8, genrange_ra, genrange_decl, genrange_b)
        vdotn_range = (float(field_vdotn.min()), float(field_vdotn.max()))
        z_helio_bounds = marginal_z_helio_bounds(z_range, vdotn_range)
        tables = [wgtsum] + set_wgtsums
        n_base = len(tables) * (1 if dndz_density_pec1a is None else 2)
        # one more block per distinct (GROUPID set, zHEL draw window); a
        # fixed redshift is a point at its own entry's zHEL
        windowed = field_z_window[:, 0] > ZMIN_SIMLIB_REGEN
        for i in np.flatnonzero(windowed):
            lo, hi = field_z_window[i]
            z_draw = marginal_z_helio_bounds(
                (lo, hi), (field_vdotn[i],) * 2 if lo == hi else vdotn_range)
            field_block[i] = window_blocks.setdefault(
                (int(field_block[i]), z_draw), n_base + len(window_blocks))
        galaxy_z = np.asarray(table.columns['ZTRUE'], dtype=np.float64)

        def marginal(density, blocks):
            cdfs = []
            for s, z_draw in blocks:
                grid = zhel_grid(z_draw, dztol_coeffs, vpec)
                # a GROUPID set draws only its galaxies of positive weight
                drawable = (np.ones((1, galaxy_z.size), dtype=bool) if s == 0
                            else np.diff(tables[s], axis=1, prepend=0.0) > 0)
                refuse_empty_windows(galaxy_z, dztol_coeffs, grid.z_reach(),
                                     drawable)
                cdfs.append(_build_marginal_cdf(
                    galaxy_z, dztol_coeffs, in_range,
                    *zhel_cells(z_draw, density, dndz_z_grid, vpec, grid),
                    z_helio_bounds, tables[s]))
            return cdfs

        base = [(s, z_helio_bounds) for s in range(len(tables))]
        rows = marginal(dndz_density, base)
        if dndz_density_pec1a is not None:
            rows += marginal(dndz_density_pec1a, base)
            pec1a_row_shift = len(tables) * wgtsum.shape[0]
        # a windowed entry redraws z at the DNDZ rate, peculiar or not
        marginal_cdf = jnp.concatenate(
            rows + marginal(dndz_density, list(window_blocks)), axis=0)

    selector = HostSelector(
        n_galaxies=ngal,
        n_bins=wgtsum.shape[0],
        varnames=varnames,
        dztol_coeffs=dztol,
        useonce=useonce,
        uses_mindaysep=uses_mindaysep,
        mindaysep=mindaysep_days,
        peakmjd_offset=peakmjd_offset,
        igal_force=igal_force,
        galid_priority=(build_galid_priority(table, *galid_priority,
                                             z_helio_bounds)
                        if galid_priority is not None else None),
        field_row_block=(jnp.asarray(field_block)
                         if groupid_order or window_blocks else None),
        pec1a_row_shift=pec1a_row_shift,
        has_snvars=has_snvars,
        snvar_names=snvar_names,
        snvar_bin_centers=([jnp.array(c, dtype=jnp.float32)
                            for c in snvar_bin_centers]
                           if has_snvars else None),
        hostlib_mskopt=hostlib_mskopt,
        maxddlr=maxddlr[0],
        maxddlr2=maxddlr[1],
        smear_sersic=smear_sersic,
        use_dlr_columns=use_dlr_columns,
        radec_rows=jnp.array([varnames.index(n) for n in ('RA_GAL', 'DEC_GAL')]),
        dlr_rows=(jnp.array([varnames.index(n) for n in ('a_DLR', 'b_DLR')])
                  if use_dlr_columns else None),
        neighbour_sentinels=_neighbour_sentinels(varnames),
        has_neighbours=has_neighbours,
        detect_thresholds=detect_thresholds,
        snr_scale_ranges=snr_scale_ranges,
        sersic_table=compute_sersic_integral_table(),
        **position_settings,
    )
    catalogue = HostCatalogue(
        redshifts=redshifts,
        wgtsum=jnp.concatenate([wgtsum] + set_wgtsums, axis=0),
        marginal_cdf=marginal_cdf,
        weights=jnp.array(weights) if weights is not None else None,
        snmagshift=(jnp.array(snmagshift) if snmagshift is not None
                    else None),
        columns=columns,
        sersic=_pack_sersic_components(table),
        neighbours=jnp.array(table.neighbours),
        neighbour_offsets=jnp.asarray(neighbour_offsets(
            table.columns['RA_GAL'], table.columns['DEC_GAL'],
            table.neighbours)),
        detect_columns=detect_columns,
        detect_errs=detect_errs,
        extra_columns={name: jnp.asarray(table.columns[name])
                       for name in extra if name in table.columns},
    )
    return selector, catalogue


def _log_dztol(dztol_coeffs):
    """Report the host redshift tolerance polynomial."""
    if len(dztol_coeffs) == 2 and dztol_coeffs[0] == 0.002 and dztol_coeffs[1] == 0.04:
        log.line(logger, "Using SNANA default tolerance: dztol = 0.002 + 0.04*z")
    elif len(dztol_coeffs) == 1:
        log.line(logger, f"Using fixed tolerance: dztol = {dztol_coeffs[0]:.4f}")
    else:
        terms = []
        for i, coeff in enumerate(dztol_coeffs):
            if coeff != 0:
                if i == 0:
                    terms.append(f"{coeff:.4f}")
                elif i == 1:
                    terms.append(f"{coeff:.4f}*z")
                else:
                    terms.append(f"{coeff:.4f}*z^{i}")
        log.line(logger, f"Using custom tolerance: dztol = {' + '.join(terms)}")


def prepare_host_selection(config, simlib, sed_model, table, stacked, extra,
                       wgtmap_file, hz_map, frac_pec1a):
    """The host draw's settings and the catalogue it draws from, as
    ``(HostSelector, HostCatalogue)``: the weights (WGTMAP, SALT2GAMMA_GRID,
    TRUE, GROUPID) and the input's host keys over ``table``."""
    host = config.host
    wgtsum_data = build_host_weights(config.host, sed_model, table, wgtmap_file)
    groupid_wgtsums, field_groupid_set = _groupid_weights(
        config, simlib, table, wgtsum_data.weights)

    dztol_coeffs = tuple(float(x) for x in host.dztol.split(','))
    useonce = bool(host.mskopt & 32)
    z_min, z_max = config.rates.redshift_range
    dndz_density, dndz_density_pec1a, dndz_z_grid = None, None, None
    field_vdotn = np.zeros(len(simlib.fields))
    if host.mskopt & 16:
        cosmo = config.cosmology
        dndz_z_grid = np.linspace(z_min, z_max, 1000, dtype=np.float64)

        def density(dndz_params, rate_mods):
            return dndz_density_on_grid(
                dndz_params, dndz_z_grid, *cosmo_params(cosmo), hz_map, rate_mods)

        dndz_density = density(config.dndz_params(),
                               config.rates.dndz_rate_mods)
        if frac_pec1a > 0.0:
            dndz_density_pec1a = density(
                config.dndz_params(config.rates.dndz_entries_pec1a),
                neutral_rate_mods())
        if cosmo.vel_cmbapex >= 1.0:
            field_ras = np.array([f.ra for f in simlib.fields], dtype=np.float64)
            field_decs = np.array([f.dec for f in simlib.fields], dtype=np.float64)
            field_vdotn = np.asarray(cmb_vdotn(field_ras, field_decs),
                                     dtype=np.float64)

    mindaysep = host.mindaysep_samegal
    if mindaysep <= MINDAYSEP_SAMEGAL_MAX:
        # strict test, as SNANA's (sntools_host.c:5654-5663)
        if useonce and mindaysep < MINDAYSEP_SAMEGAL_MAX:
            raise SNANAError(
                "Cannot set both HOSTLIB_MSKOPT USEONCE (bit 32) "
                "and HOSTLIB_MINDAYSEP_SAMEGAL",
                function_name="prepare_host_selection",
            )
        mindaysep = (mindaysep, config.rates.peakmjd_range)
    else:
        mindaysep = None

    # GALID_PRIORITY: active for 0 < min <= max (GEN_SNHOST_GALID,
    # sntools_host.c:6300); '0,0' is "not set"
    galid_pri_parts = [int(x) for x in host.galid_priority.split(',')]
    galid_priority = (tuple(galid_pri_parts[:2])
                      if len(galid_pri_parts) >= 2
                      and 0 < galid_pri_parts[0] <= galid_pri_parts[1] else None)

    if host.snr_detect is not None:
        detection_filter = ('snr', host.snr_detect, host.snr_scale)
    elif host.mag_detect is not None:
        detection_filter = ('mag', host.mag_detect, None)
    else:
        detection_filter = None

    if config.cosmology.vpec_force < 1.0e8:
        vpec = (config.cosmology.vpec_force, 0.0)
    elif host.mskopt & 512:
        vpec = (0.0, 0.0)
    else:
        vpec = (0.0, config.cosmology.gensigma_vpec)

    return build_host_selection(
        table, stacked, extra, wgtsum_data.wgtsum, wgtsum_data.weights,
        wgtsum_data.galaxy_snmagshift, wgtsum_data.snvar_names,
        wgtsum_data.snvar_bin_centers,
        dztol_coeffs=dztol_coeffs, useonce=useonce,
        hostlib_mskopt=host.mskopt,
        genrange_ra=config.survey.genrange_ra,
        genrange_decl=config.survey.genrange_decl,
        genrange_b=config.survey.genrange_b,
        dndz_density=dndz_density,
        dndz_density_pec1a=dndz_density_pec1a,
        dndz_z_grid=dndz_z_grid,
        z_range=(z_min, z_max),
        vpec=vpec,
        field_vdotn=field_vdotn,
        maxddlr=(host.maxddlr, host.maxddlr2),
        mindaysep=mindaysep,
        smear_sersic=host.smear_sersic,
        position_settings=dict(
            fixran_wgt=float(host.fixran_wgt),
            fixran_radius=float(host.fixran_radius),
            fixran_phi=float(host.fixran_phi),
            mnintflux_snpos=float(host.mnintflux_snpos),
            mxintflux_snpos=float(host.mxintflux_snpos)),
        abmag=(host.abmag_offset, host.abmag_force),
        igal_force=resolve_igal_force(table, host.galid_force),
        galid_priority=galid_priority,
        detection_filter=detection_filter,
        groupid_wgtsums=groupid_wgtsums,
        field_groupid_set=field_groupid_set,
        field_z_window=np.array([f.genrange_redshift for f in simlib.fields],
                                dtype=np.float64),
    )


def _groupid_weights(config, simlib, table, base_weights):
    """GROUPID matching (RDFLAG_GROUPID, snlc_sim.c:22115): the weight
    cumsum restricted to each unique GROUPID set the SIMLIB names, and each
    SIMLIB entry's set index (-1 for none), or ``(None, None)``."""
    if not config.survey.use_simlib_groupid:
        return None, None
    unique_sets: Dict[frozenset, int] = {}
    field_to_set: List[int] = []
    for f in simlib.fields:
        gset = frozenset(f.groupid_hostlib)
        if gset and gset not in unique_sets:
            unique_sets[gset] = len(unique_sets)
        field_to_set.append(unique_sets.get(gset, -1))
    if not unique_sets:
        return None, None
    if 'GROUPID' not in table.columns:
        raise SNANAError(
            "SIMLIB requires a HOSTLIB GROUPID but the HOSTLIB has no "
            "GROUPID column.", function_name="_groupid_weights")
    groupid = np.asarray(table.columns['GROUPID'], dtype=np.int64)
    filtered_wgtsums: Dict[int, np.ndarray] = {}
    for gset, idx in unique_sets.items():
        mask = np.isin(groupid, list(gset))
        filtered_wgtsums[idx] = np.cumsum(
            base_weights * mask[np.newaxis, :], axis=1).astype(np.float32)
    n_matched = sum(1 for x in field_to_set if x >= 0)
    log.line(logger, f"GROUPID matching: {len(unique_sets)} unique sets, "
             f"{n_matched}/{len(field_to_set)} fields with GROUPID")
    return filtered_wgtsums, np.array(field_to_set, dtype=np.int32)


def sn_sky_position(host_properties):
    """SN sky position: host centre plus the drawn Sersic offset, in degrees.

    SNANA's SNHOSTGAL.RA_SN_DEG / DEC_SN_DEG (GEN_SNHOST_POS). The RA offset is
    an angle on the sky, so it divides by cos(dec).
    """
    ra_gal = host_properties['RA_GAL']
    dec_gal = host_properties['DEC_GAL']
    cos_dec = jnp.cos(jnp.radians(dec_gal))
    return (ra_gal + host_properties['RA_OFFSET'] / (3600.0 * cos_dec),
            dec_gal + host_properties['DEC_OFFSET'] / 3600.0)


@jax.jit
def compute_dztol(z_sn: jnp.ndarray, dztol_coeffs: jnp.ndarray) -> jnp.ndarray:
    """Redshift tolerance matching SNANA's HOSTLIB_GENPOLY_DZTOL.

    Evaluates ``dztol = sum(coeff[i] * z_sn**i for i in range(len(coeffs)))``;
    the SNANA default "0.002,0.04" gives ``dztol = 0.002 + 0.04 * z_sn``.
    ``z_sn`` may be scalar or an array; the result has the same shape.
    """
    return eval_genpoly(dztol_coeffs, z_sn)


def compute_snvar_bin_index_1d(
    snvar_value: float,
    snvar_bin_centers: jnp.ndarray
) -> int:
    """Bin index (int32) for a scalar 1D SN variable (e.g., x1 only).

    Matches SNANA's bin snapping formula, ``IB1D[ivar_SN] =
    (int)(0.5+(VAL-VALMIN)/VALBIN)`` with ``VALBIN = (VALMAX - VALMIN) /
    (NBINS - 1)``. An out-of-range value is fatal in SNANA
    (sntools_host.c:1655-1660) and the run aborts on it
    (:meth:`HostSelector.snvar_outside`), so no index clipping is applied
    here.
    """
    n_bins = len(snvar_bin_centers)
    valmin = snvar_bin_centers[0]
    valmax = snvar_bin_centers[-1]
    valbin = jnp.where(n_bins > 1, (valmax - valmin) / (n_bins - 1), 1.0)
    return jnp.int32(0.5 + (snvar_value - valmin) / valbin)


compute_snvar_bin_index_1d_vectorized = jax.vmap(
    compute_snvar_bin_index_1d,
    in_axes=(0, None)
)


def compute_snvar_bin_index_nd(snvar_values_list, snvar_bin_centers_list):
    """Flat bin index, shape (n_sne,) int32, from N-D SN variable values.

    ``snvar_values_list`` is one (n_sne,) array per SN variable and
    ``snvar_bin_centers_list`` one (n_bins_i,) array of bin centres per
    variable. Uses C-order (last dimension varies fastest), matching numpy
    meshgrid with indexing='ij' + ravel used in compute_wgtsum.
    """
    n_dims = len(snvar_bin_centers_list)
    flat_idx = jnp.zeros(len(snvar_values_list[0]), dtype=jnp.int32)
    stride = 1
    for i in reversed(range(n_dims)):
        ibin = compute_snvar_bin_index_1d_vectorized(
            snvar_values_list[i], snvar_bin_centers_list[i]
        )
        flat_idx = flat_idx + ibin * stride
        stride *= len(snvar_bin_centers_list[i])
    return flat_idx


def _searchsorted_rows(table, row_indices, values, side):
    """Binary search of values[i] within table[row_indices[i], :].

    Matches jnp.searchsorted(table[row], value, side=side) per event, but
    gathers one element per event per bisection step instead of materialising
    an (n_events, n_columns) row copy. Rows must be sorted ascending.
    """
    n = table.shape[1]
    lo = jnp.zeros(values.shape, dtype=jnp.int32)
    hi = jnp.full(values.shape, n, dtype=jnp.int32)

    def bisect(_, bounds):
        lo, hi = bounds
        active = lo < hi
        mid = (lo + hi) // 2
        mid_vals = table[row_indices, jnp.clip(mid, 0, n - 1)]
        if side == 'left':
            go_right = active & (mid_vals < values)
        else:
            go_right = active & (mid_vals <= values)
        lo = jnp.where(go_right, mid + 1, lo)
        hi = jnp.where(active & ~go_right, mid, hi)
        return lo, hi

    lo, _ = jax.lax.fori_loop(0, max(n, 1).bit_length(), bisect, (lo, hi))
    return lo


MAX_RESELECT_ROUNDS = 50


def _useonce_claims(galaxy, candidate, taken):
    """One USEONCE round: each candidate whose galaxy is still free claims it,
    and where several want one galaxy the lowest event index wins, as the
    first of them would in SNANA's event loop. Returns the winners and the
    updated ``taken`` mask."""
    n = galaxy.shape[0]
    lane = jnp.arange(n, dtype=jnp.int32)
    free = candidate & ~taken[galaxy]
    first = jnp.full(taken.shape, n, dtype=jnp.int32).at[galaxy].min(
        jnp.where(free, lane, n))
    won = free & (first[galaxy] == lane)
    taken = taken.at[jnp.where(won, galaxy, taken.shape[0])].set(
        True, mode='drop')
    return won, taken


def _lex_search(hist_gal, hist_day, galaxy, day, side):
    """Per event, the insertion point of (galaxy, day) in the history sorted
    by (galaxy, day)."""
    n = hist_gal.shape[0]
    lo = jnp.zeros(galaxy.shape, dtype=jnp.int32)
    hi = jnp.full(galaxy.shape, n, dtype=jnp.int32)

    def bisect(_, bounds):
        lo, hi = bounds
        active = lo < hi
        mid = jnp.clip((lo + hi) // 2, 0, max(n - 1, 0))
        g, d = hist_gal[mid], hist_day[mid]
        before = (g < galaxy) | ((g == galaxy)
                                 & ((d < day) if side == 'left' else (d <= day)))
        lo = jnp.where(active & before, mid + 1, lo)
        hi = jnp.where(active & ~before, mid, hi)
        return lo, hi

    lo, _ = jax.lax.fori_loop(0, max(n, 1).bit_length(), bisect, (lo, hi))
    return lo


def _mindaysep_blocked(galaxy, day, lanes, hist_gal, hist_day, window):
    """Which events a MINDAYSEP_SAMEGAL window blocks.

    An event is blocked by a peak on its galaxy within ``window`` days: one
    in the history of earlier batches' passing events (sorted by galaxy,
    then day), or one of the batch's earlier events that was not blocked
    itself, taken greedily in event order per galaxy. Events outside
    ``lanes`` neither block nor are blocked.
    """
    n = galaxy.shape[0]
    lane = jnp.arange(n, dtype=jnp.int32)
    lo = _lex_search(hist_gal, hist_day, galaxy, day - window + 1, 'left')
    hi = _lex_search(hist_gal, hist_day, galaxy, day + window - 1, 'right')
    by_history = lanes & (hi > lo)

    # events outside the lanes each form a group of their own
    group = jnp.where(lanes, galaxy, -1 - lane)
    order = jnp.lexsort((lane, group))
    sgroup, sday, shistory = group[order], day[order], by_history[order]
    opens = jnp.concatenate([jnp.array([True]), sgroup[1:] != sgroup[:-1]])
    rank = lane - jax.lax.cummax(jnp.where(opens, lane, 0))

    def settle(r, accepted):
        """Decide the events of rank ``r`` in their galaxy against the
        accepted events before them."""
        def predecessor(k, blocked):
            prev = jnp.clip(lane - k, 0, n - 1)
            return blocked | ((k <= rank) & accepted[prev]
                              & (jnp.abs(sday - sday[prev]) < window))
        blocked = jax.lax.fori_loop(1, r + 1, predecessor,
                                    jnp.zeros(n, dtype=bool))
        return jnp.where(rank == r, ~shistory & ~blocked, accepted)

    accepted = jax.lax.fori_loop(0, jnp.max(rank) + 1, settle,
                                 jnp.zeros(n, dtype=bool))
    blocked = jnp.zeros(n, dtype=bool).at[order].set(~accepted)
    return lanes & blocked


def _reselect(draw, galaxy, reselect_keys, blocked_fn, state):
    """Redraw the blocked events until none is, or MAX_RESELECT_ROUNDS pass:
    one loop over rounds, every blocked event redrawn at once from its own
    key. ``blocked_fn(galaxy, state)`` returns the blocked events and the
    updated state. Returns the galaxies, the events still blocked and the
    state."""
    blocked, state = blocked_fn(galaxy, state)

    def pending(carry):
        round_, _, blocked, _ = carry
        return jnp.any(blocked) & (round_ < MAX_RESELECT_ROUNDS)

    def redraw(carry):
        round_, galaxy, blocked, state = carry
        uniforms = jax.vmap(
            lambda k: jax.random.uniform(jax.random.fold_in(k, round_)))(
            reselect_keys)
        galaxy = jnp.where(blocked, draw(uniforms), galaxy)
        blocked, state = blocked_fn(galaxy, state)
        return round_ + 1, galaxy, blocked, state

    _, galaxy, blocked, state = jax.lax.while_loop(
        pending, redraw, (jnp.int32(0), galaxy, blocked, state))
    return galaxy, blocked, state


def _dztol_window(sn_redshifts, snvar_bin_indices, galaxy_redshifts, wgtsum,
                  dztol_coeffs):
    """Each event's DZTOL window, strict on both sides: ``(first, last, weight before first, weight in window, galaxies in window)``."""
    dztol = compute_dztol(sn_redshifts, dztol_coeffs)
    first = jnp.searchsorted(galaxy_redshifts, sn_redshifts - dztol,
                             side='right')
    end = jnp.searchsorted(galaxy_redshifts, sn_redshifts + dztol, side='left')
    n_galaxies = wgtsum.shape[1]
    first_c = jnp.clip(first, 0, n_galaxies - 1)
    last = jnp.clip(end - 1, first_c, n_galaxies - 1)
    before = jnp.where(first > 0,
                       wgtsum[snvar_bin_indices, jnp.maximum(first - 1, 0)], 0.0)
    inside = jnp.where(end > first,
                       wgtsum[snvar_bin_indices, last] - before, 0.0)
    return first_c, last, before, inside, end - first


@jax.jit
def draw_host_indices(
    sn_redshifts: jnp.ndarray,
    random_uniforms: jnp.ndarray,
    snvar_bin_indices: jnp.ndarray,
    galaxy_redshifts: jnp.ndarray,
    wgtsum: jnp.ndarray,
    dztol_coeffs: jnp.ndarray,
) -> jnp.ndarray:
    """Weighted host selection (GEN_SNHOST_GALID, sntools_host.c:6326):
    each galaxy strictly inside the event's DZTOL window is drawn in
    proportion to its weight.

    One kernel for every event regardless of WGTMAP SN-variable bin: each
    event reads its own bin's cumulative-weight row pointwise, so the
    compiled work depends only on the batch and catalogue shapes and the
    batch never leaves the device.

    ``sn_redshifts``, ``random_uniforms`` (in [0, 1)) and
    ``snvar_bin_indices`` (each event's row of ``wgtsum``) are shape
    (n_sne,); ``galaxy_redshifts`` is (n_galaxies,), sorted ascending, and
    ``wgtsum`` the (n_bins, n_galaxies) cumulative weights.

    Returns the selected galaxy indices, shape (n_sne,) int32.
    """
    first, last, before, inside, _ = _dztol_window(
        sn_redshifts, snvar_bin_indices, galaxy_redshifts, wgtsum,
        dztol_coeffs)
    selected = _searchsorted_rows(wgtsum, snvar_bin_indices,
                                  before + random_uniforms * inside,
                                  side='left')
    # A window whose galaxies all have zero weight gives its first galaxy, a
    # host the WGTMAP excludes, as SNANA does (sntools_host.c:6276-6279,
    # :6396-6398). jax follows SNANA so that a weight map with zero cells,
    # such as DES-SN5YR's 91bg map at low redshift, still runs.
    return jnp.clip(selected, first, last).astype(jnp.int32)


@jax.jit
def empty_host_windows(sn_redshifts, rows, galaxy_redshifts, wgtsum,
                       dztol_coeffs, n_bins):
    """Events with no host to draw: their DZTOL window holds no galaxy, or,
    on a GROUPID row (``rows`` at or past ``n_bins``), no matching galaxy
    of positive weight."""
    _, _, _, inside, n_inside = _dztol_window(
        sn_redshifts, rows, galaxy_redshifts, wgtsum, dztol_coeffs)
    return (n_inside <= 0) | ((rows >= n_bins) & (inside <= 0))


@jax.jit
def _select_from_cdf(cdf, snvar_bin_indices, random_uniforms):
    """Inverse-transform sample each event's per-bin marginal CDF row."""
    idx = _searchsorted_rows(cdf, snvar_bin_indices, random_uniforms,
                             side='right').astype(jnp.int32)
    return jnp.minimum(idx, cdf.shape[1] - 1)


def _dlr_ddlr(a_half, b_half, a_rot_deg, vec_sn_x, vec_sn_y, snsep):
    """DLR along the SN direction and DDLR = SNSEP/DLR.

    Single implementation of SNANA's ellipse projection (GEN_SNHOST_DDLR,
    sntools_host.c:8581-8613), shared by the neighbour and true-host paths.
    The semi-major-axis vector and the galaxy-to-SN vector must be in the
    same frame: raw RA/DEC treated as cartesian, with no cos(DEC) factor.
    cosTH is clamped to +-(1 - 1e-9) as in SNANA (sntools_host.c:8600-8601);
    DDLR = 0 when DLR <= 0 (zero-size galaxy, sntools_host.c:8608-8611).
    """
    a_rot_rad = jnp.radians(a_rot_deg)
    vec_a_x = a_half * jnp.cos(a_rot_rad)
    vec_a_y = -a_half * jnp.sin(a_rot_rad)
    len_sn = jnp.sqrt(vec_sn_x**2 + vec_sn_y**2) + 1e-30
    cos_th = (vec_a_x * vec_sn_x + vec_a_y * vec_sn_y) / (len_sn * a_half + 1e-30)
    cos_th = jnp.clip(cos_th, -1.0 + 1e-9, 1.0 - 1e-9)
    sq_cos = cos_th**2
    sq_sin = 1.0 - sq_cos
    dlr = (a_half * b_half) / jnp.sqrt(
        a_half**2 * sq_sin + b_half**2 * sq_cos + 1e-30)
    ddlr = jnp.where(dlr > 0, snsep / dlr, 0.0)
    return dlr, ddlr


@jax.jit
def _compute_ddlr_and_sort(
    radec_columns,      # (2, n_galaxies): RA_GAL, DEC_GAL in degrees
    sersic_a,
    sersic_b,
    sersic_w,           # normalised to sum 1 per galaxy
    sersic_arot,        # degrees
    dlr_ab_columns,     # a_DLR, b_DLR (zeros when absent)
    nbr_array,
    nbr_offsets,        # degrees
    galaxy_indices,
    ra_sn, dec_sn,      # degrees
    sn_dra, sn_ddec,    # SN offset from its own host, degrees
    maxddlr, maxddlr2,
    smear_sersic,
    use_dlr_columns,
    detect_columns,     # 0 rows when detection is off
    detect_thresholds,
    snr_err_scale,
    detect_raw_errs,    # non-empty only under SNR_SCALE
):
    """Compute DDLR for all neighbours and sort by DDLR.

    For each SN, gathers its true host's NBR_LIST, prepends the true host
    as neighbour 0, computes DDLR for each neighbour, and sorts ascending.

    SNANA ordering (GEN_SNHOST_DDLR, sntools_host.c:8546-8573):
    1. Compute weighted Sersic average for a_half, b_half
    2. Override with a_DLR/b_DLR columns if present
    3. Apply SMEAR_SERSIC on top

    Returns sorted DDLR, SNSEP, galaxy indices, validity mask, the NMATCH
    and NMATCH2 counts, and the true host's own DDLR and SNSEP.
    """
    n_galaxies = radec_columns.shape[1]

    nbr_from_list = nbr_array[galaxy_indices]
    all_nbr = jnp.concatenate([galaxy_indices[:, None], nbr_from_list], axis=1)
    valid_mask = all_nbr >= 0
    safe_idx = jnp.clip(all_nbr, 0, n_galaxies - 1)

    # detect_columns hold SNR (SNR mode) or negated mags (MAG mode)
    n_thresh = detect_thresholds.shape[0]
    n_raw_err_cols = detect_raw_errs.shape[0]

    if n_thresh > 0:
        if n_raw_err_cols > 0:
            raw_errs = detect_raw_errs[:, safe_idx]
            raw_errs = raw_errs * snr_err_scale[None, :, None]
            raw_errs = jnp.where((raw_errs <= 0) | (raw_errs > 9.0), 9.0, raw_errs)
            band_values = (2.5 / jnp.log(10.0)) / raw_errs
        else:
            band_values = detect_columns[:, safe_idx]

        sorted_vals = -jnp.sort(-band_values, axis=0)
        passes = sorted_vals[:n_thresh] >= detect_thresholds[:, None, None]
        detected = jnp.all(passes, axis=0)
        valid_mask = valid_mask & detected

    nbr_dec = radec_columns[1][safe_idx]
    nbr_arot = sersic_arot[safe_idx]

    # weighted Sersic size, then a_DLR/b_DLR, then SMEAR_SERSIC, as SNANA orders them
    nbr_w = sersic_w[:, safe_idx]
    a_half = jnp.sum(nbr_w * sersic_a[:, safe_idx], axis=0)
    b_half = jnp.sum(nbr_w * sersic_b[:, safe_idx], axis=0)
    a_half = jnp.where(use_dlr_columns, dlr_ab_columns[0][safe_idx], a_half)
    b_half = jnp.where(use_dlr_columns, dlr_ab_columns[1][safe_idx], b_half)
    ab_smear = smear_sersic / 2.0
    a_half = jnp.sqrt(a_half**2 + ab_smear**2)
    b_half = jnp.sqrt(b_half**2 + ab_smear**2)

    # haversine on host-relative offsets: float32-stable at small separations
    zero_col = jnp.zeros((galaxy_indices.shape[0], 1), dtype=jnp.float32)
    nbr_dra = jnp.concatenate([zero_col, nbr_offsets[0][galaxy_indices]], axis=1)
    nbr_ddec = jnp.concatenate([zero_col, nbr_offsets[1][galaxy_indices]], axis=1)
    dlon_deg = sn_dra[:, None] - nbr_dra
    dlat_deg = sn_ddec[:, None] - nbr_ddec

    dec1_rad = jnp.radians(nbr_dec)
    dec2_rad = jnp.radians(dec_sn[:, None])
    dlat = jnp.radians(dlat_deg)
    dlon = jnp.radians(dlon_deg)
    a_hav = jnp.sin(dlat / 2.0)**2 + jnp.cos(dec1_rad) * jnp.cos(dec2_rad) * jnp.sin(dlon / 2.0)**2
    a_hav = jnp.clip(a_hav, 0.0, 1.0)
    snsep = 2.0 * jnp.degrees(jnp.arcsin(jnp.sqrt(a_hav))) * 3600.0  # arcsec

    # DLR direction in degrees with no cos(DEC), as SNANA
    _dlr, ddlr = _dlr_ddlr(a_half, b_half, nbr_arot, dlon_deg, dlat_deg, snsep)

    # neighbour 0 is the true host, read before masking
    true_ddlr, true_snsep = ddlr[:, 0], snsep[:, 0]
    ddlr = jnp.where(valid_mask, ddlr, jnp.inf)
    snsep = jnp.where(valid_mask, snsep, jnp.inf)
    sort_order = jnp.argsort(ddlr, axis=1)

    sorted_ddlr = jnp.take_along_axis(ddlr, sort_order, axis=1)
    sorted_snsep = jnp.take_along_axis(snsep, sort_order, axis=1)
    sorted_gal_idx = jnp.take_along_axis(safe_idx, sort_order, axis=1)
    sorted_valid = jnp.take_along_axis(valid_mask, sort_order, axis=1)

    # uncapped, as SNANA writes NNBR_DDLRCUT (snlc_sim.c:26045-26046)
    nmatch = jnp.sum((ddlr < maxddlr) & valid_mask, axis=1)
    nmatch2 = jnp.sum((ddlr < maxddlr2) & valid_mask, axis=1)

    return (sorted_ddlr, sorted_snsep, sorted_gal_idx, sorted_valid,
            nmatch, nmatch2, true_ddlr, true_snsep)


@dataclass(frozen=True, eq=False)
class HostSelector:
    """The host draw's settings, from the input's host keys, and SNANA's
    GEN_SNHOST_GALID over a :class:`HostCatalogue`, which every method that
    reads the galaxies takes:

    - redshift tolerance dztol = 0.002 + 0.04 * z under SNANA's default
    - binary search for the redshift window
    - binary search of the WGTSUM for the inverse-transform draw

    Built with its catalogue by :func:`~jax_snlc_sim.hostlib.catalogue.build_host_selection`. The hosts a
    run has handed out are the run's, a record passed in and returned
    (:meth:`initial_reuse`).
    """
    n_galaxies: int
    n_bins: int                     # WGTMAP SN-variable bins per GROUPID set
    varnames: List[str]             # the rows of HostCatalogue.columns
    dztol_coeffs: jax.Array         # HOSTLIB_DZTOL, ascending powers of z
    useonce: bool
    uses_mindaysep: bool
    mindaysep: Optional[int]
    peakmjd_offset: int             # MJD0 of the MINDAYSEP day count
    igal_force: int
    galid_priority: Optional[Dict[str, np.ndarray]]
    field_row_block: Optional[jax.Array]   # per SIMLIB entry: GROUPID set, and redshift window under SN2GAL_Z
    pec1a_row_shift: Optional[int]  # from a wgtsum row to its DNDZ_PEC1A row of marginal_cdf
    has_snvars: bool
    snvar_names: Optional[List[str]]
    snvar_bin_centers: Optional[List[jax.Array]]
    hostlib_mskopt: int
    maxddlr: float
    maxddlr2: float
    smear_sersic: float
    use_dlr_columns: bool
    radec_rows: jax.Array           # RA_GAL, DEC_GAL rows of the columns
    dlr_rows: Optional[jax.Array]   # a_DLR, b_DLR rows, when present
    neighbour_sentinels: jax.Array  # per column, for an unmatched slot
    has_neighbours: bool
    detect_thresholds: jax.Array
    snr_scale_ranges: Optional[list]
    sersic_table: Any               # SersicTableData
    fixran_wgt: float
    fixran_radius: float
    fixran_phi: float
    mnintflux_snpos: float
    mxintflux_snpos: float

    def _compute_snvar_bin_indices(self, n_sne, sn_properties):
        if not self.has_snvars:
            return jnp.full(n_sne, 0, dtype=jnp.int32)
        if len(self.snvar_names) == 1:
            snvar_values = jnp.array(
                sn_properties[self.snvar_names[0]], dtype=jnp.float32)
            return compute_snvar_bin_index_1d_vectorized(
                snvar_values, self.snvar_bin_centers[0])
        snvar_values_list = [
            jnp.array(sn_properties[name], dtype=jnp.float32)
            for name in self.snvar_names
        ]
        return compute_snvar_bin_index_nd(
            snvar_values_list, self.snvar_bin_centers)

    def _rows(self, snvar_bin_indices, field_indices):
        """Each event's row of the weight tables."""
        if self.field_row_block is None:
            return snvar_bin_indices
        return (jnp.take(self.field_row_block, field_indices) * self.n_bins
                + snvar_bin_indices)

    def _select_hosts_one_pass(self, catalogue, sn_redshifts, random_uniforms,
                               snvar_bin_indices, field_indices, pec1a):
        rows = self._rows(snvar_bin_indices, field_indices)
        if self.pec1a_row_shift is not None:
            rows = jnp.where(pec1a & (rows < self.pec1a_row_shift),
                             rows + self.pec1a_row_shift, rows)
        if catalogue.marginal_cdf is not None:
            return _select_from_cdf(catalogue.marginal_cdf, rows,
                                    random_uniforms)
        return draw_host_indices(sn_redshifts, random_uniforms, rows,
                                 catalogue.redshifts, catalogue.wgtsum,
                                 self.dztol_coeffs)

    def gather_host_properties(self, catalogue, galaxy_indices,
                               snvar_bin_indices):
        """Gather host properties from pre-stacked columns via single fused gather."""
        gathered = catalogue.columns[:, galaxy_indices]
        host_properties = {}
        for i, varname in enumerate(self.varnames):
            host_properties[varname] = gathered[i]
        if catalogue.snmagshift is not None:
            shifts = catalogue.snmagshift
            host_properties['SNMAGSHIFT'] = (shifts[galaxy_indices] if shifts.ndim == 1
                                             else shifts[snvar_bin_indices, galaxy_indices])
        if catalogue.weights is not None:
            host_properties['WGTMAP_WGT'] = catalogue.weights[snvar_bin_indices, galaxy_indices]
        return host_properties

    def _add_position_properties(self, catalogue, host_properties,
                                 galaxy_indices, position_uniforms):
        """Generate SN positions or add sentinel values.

        Without a Sersic profile the position quantities stay at their
        sentinels, as in SNANA (GEN_SNHOST_POS returns before drawing a
        position when NPROF=0, sntools_host.c:7621-7635).
        """
        if position_uniforms is not None and catalogue.sersic is not None:
            position_properties = self._generate_sn_positions(
                catalogue, host_properties, galaxy_indices,
                position_uniforms)
            host_properties.update(position_properties)
        else:
            n_sne = len(galaxy_indices)
            host_properties.update({
                'RA_OFFSET': jnp.full(n_sne, -9.0),
                'DEC_OFFSET': jnp.full(n_sne, -9.0),
                'SNSEP': jnp.full(n_sne, -9.0),
                'DLR': jnp.full(n_sne, -9.0),
                'DDLR': jnp.full(n_sne, -9.0),
                'REDUCED_R': jnp.full(n_sne, -9.0),
                'a_SNGALSEP': jnp.full(n_sne, -9.0),
                'b_SNGALSEP': jnp.full(n_sne, -9.0),
            })
        return host_properties

    def _generate_sn_positions(
        self,
        catalogue,
        host_properties: dict,
        galaxy_indices: jnp.ndarray,
        position_uniforms: jnp.ndarray
    ) -> dict:
        """Generate SN positions within selected host galaxies using Sersic
        profiles.

        ``host_properties`` is the already-gathered host properties dict
        (from stacked gather); ``position_uniforms`` is shape (n_sne, 4) holding
        [component_choice, random_radius, random_u1, random_u2] per SN.
        Returns a dict with 'RA_OFFSET'/'DEC_OFFSET'/'SNSEP' (arcsec),
        'DLR' (Directional Light Radius, arcsec), 'DDLR' (SNSEP/DLR),
        'REDUCED_R' (R/R_half) and 'a_SNGALSEP'/'b_SNGALSEP'.
        """
        a = catalogue.sersic['a'][:, galaxy_indices]
        b = catalogue.sersic['b'][:, galaxy_indices]
        n = catalogue.sersic['n'][:, galaxy_indices]
        w = catalogue.sersic['w'][:, galaxy_indices]
        a_rot = catalogue.sersic['a_rot'][galaxy_indices]

        # component choice (GEN_SNHOST_POS, sntools_host.c:7662-7671)
        cum_w = jnp.cumsum(w, axis=0)
        wgt_ran = position_uniforms[:, 0]
        if self.fixran_wgt >= 0.0:
            wgt_ran = jnp.full_like(wgt_ran, self.fixran_wgt)
        jprof = jnp.argmax(cum_w >= wgt_ran[None, :], axis=0)
        # a float32 cum_w[-1] a ulp below 1 means the last component
        jprof = jnp.where(cum_w[-1] >= wgt_ran, jprof, w.shape[0] - 1)

        a_half = jnp.take_along_axis(a, jprof[None, :], axis=0)[0]
        b_half = jnp.take_along_axis(b, jprof[None, :], axis=0)[0]
        sersic_n = jnp.take_along_axis(n, jprof[None, :], axis=0)[0]

        radius_ran = position_uniforms[:, 1]
        if self.fixran_radius >= 0.0:
            radius_ran = jnp.full_like(radius_ran, self.fixran_radius)

        ra_offset, dec_offset, snsep, _dlr, _ddlr, reduced_r, a_SNGALSEP, b_SNGALSEP = generate_sn_position(
            a_half=a_half,
            b_half=b_half,
            sersic_n=sersic_n,
            rotation_angle_deg=a_rot,
            random_radius=radius_ran,
            random_u1=position_uniforms[:, 2],
            random_u2=position_uniforms[:, 3],
            sersic_data=self.sersic_table,
            mnintflux=self.mnintflux_snpos,
            mxintflux=self.mxintflux_snpos,
            fixran_phi=self.fixran_phi,
        )

        # GEN_SNHOST_DDLR order, as the neighbour path (sntools_host.c:8546-8573)
        a_dlr = jnp.sum(w * a, axis=0)
        b_dlr = jnp.sum(w * b, axis=0)
        if self.use_dlr_columns and 'a_DLR' in host_properties:
            a_dlr = host_properties['a_DLR']
            b_dlr = host_properties['b_DLR']
        if self.smear_sersic > 0.0:
            ab_smear = self.smear_sersic / 2.0
            a_dlr = jnp.sqrt(a_dlr**2 + ab_smear**2)
            b_dlr = jnp.sqrt(b_dlr**2 + ab_smear**2)
        cosdec = jnp.cos(jnp.radians(host_properties['DEC_GAL']))
        vec_sn_x = ra_offset / cosdec          # raw-RA frame
        vec_sn_y = dec_offset
        dlr, ddlr = _dlr_ddlr(a_dlr, b_dlr, a_rot, vec_sn_x, vec_sn_y, snsep)

        return {
            'RA_OFFSET': ra_offset,
            'DEC_OFFSET': dec_offset,
            'SNSEP': snsep,
            'DLR': dlr,
            'DDLR': ddlr,
            'REDUCED_R': reduced_r,
            'a_SNGALSEP': a_SNGALSEP,
            'b_SNGALSEP': b_SNGALSEP,
        }

    def _apply_galid_priority(self, galaxy_indices, sn_redshifts):
        """Override selected galaxies with priority GALID if one falls in z-tolerance.

        For each SN, checks if any galaxy in the priority GALID range is within
        the SN's redshift tolerance window. If so, selects that galaxy instead.

        ``galaxy_indices`` and ``sn_redshifts`` are shape (N,), the standard
        selection's output and the SN redshifts. Returns the updated
        ``galaxy_indices`` with priority overrides applied.
        """
        pri = self.galid_priority
        if len(pri['igals']) == 0:
            return galaxy_indices
        pri_z = jnp.asarray(pri['redshifts'])
        dztol = compute_dztol(sn_redshifts, self.dztol_coeffs)
        idx_lo = jnp.searchsorted(pri_z, sn_redshifts - dztol, side='left')
        idx_hi = jnp.searchsorted(pri_z, sn_redshifts + dztol, side='right')
        # SNANA loops without a break, so IGAL_SELECT keeps the last match
        last = jnp.clip(idx_hi - 1, 0, len(pri['igals']) - 1)
        return jnp.where(idx_lo < idx_hi,
                         jnp.asarray(pri['igals'], dtype=galaxy_indices.dtype)[last],
                         galaxy_indices)

    def compute_neighbor_info(self, catalogue, galaxy_indices, host_properties,
                              field_ra, field_dec, peakmjd, zgen):
        """Compute DDLR-sorted neighbour info for the HOSTGAL slots.

        Two JIT phases: compute DDLR for all neighbours using 9 columns and
        sort by DDLR, then gather full properties for the written ranks only.

        ``galaxy_indices`` is (N,) true host galaxy indices in sorted space;
        ``host_properties`` is the dict from ``select()`` with RA_GAL,
        DEC_GAL, RA_OFFSET, DEC_OFFSET. ``field_ra``/``field_dec`` are (N,)
        SN field coordinates in degrees for coordinate shifting, or None to
        skip it; ``peakmjd`` is (N,) peak MJD per SN for SNR_SCALE; ``zgen``
        is the redshift the host was looked up at, which a neighbour's
        catalogue photo-z is shifted by.

        Returns ``(hostgal_sorted, neighbours, nmatch, nmatch2)``:
        ``hostgal_sorted`` is the dict of DDLR-sorted[0] properties (may
        differ from the true host); ``neighbours`` holds one dict per
        further slot (sorted[1], sorted[2], ...; sentinel where invalid);
        ``nmatch``/``nmatch2`` are (N,) int32 counts with DDLR below
        MAXDDLR/MAXDDLR2.
        """
        n_sne = len(galaxy_indices)

        ra_sn, dec_sn = sn_sky_position(host_properties)
        cos_dec_host = jnp.cos(jnp.radians(host_properties['DEC_GAL']))
        sn_dra = host_properties['RA_OFFSET'] / (3600.0 * cos_dec_host)
        sn_ddec = host_properties['DEC_OFFSET'] / 3600.0

        snr_err_scale = jnp.ones(n_sne, dtype=jnp.float32)
        if self.snr_scale_ranges is not None and peakmjd is not None:
            peakmjd_arr = jnp.asarray(peakmjd, dtype=jnp.float32)
            for mjdmin, mjdmax, scale_val in self.snr_scale_ranges:
                in_range = (peakmjd_arr >= mjdmin) & (peakmjd_arr <= mjdmax)
                snr_err_scale = jnp.where(in_range, 1.0 / scale_val, snr_err_scale)

        (sorted_ddlr, sorted_snsep, sorted_gal_idx, sorted_valid, nmatch, nmatch2,
         true_ddlr, true_snsep) = \
            _compute_ddlr_and_sort(
                catalogue.columns[self.radec_rows], catalogue.sersic['a'],
                catalogue.sersic['b'], catalogue.sersic['w'],
                catalogue.sersic['a_rot'],
                (catalogue.columns[self.dlr_rows] if self.use_dlr_columns
                 else jnp.zeros((2, self.n_galaxies), dtype=jnp.float32)),
                catalogue.neighbours, catalogue.neighbour_offsets,
                galaxy_indices,
                ra_sn, dec_sn, sn_dra, sn_ddec, self.maxddlr, self.maxddlr2,
                self.smear_sersic, self.use_dlr_columns,
                catalogue.detect_columns, self.detect_thresholds,
                snr_err_scale, catalogue.detect_errs)
        n_ranks = sorted_gal_idx.shape[1]

        zdif = (catalogue_zphot_shift(zgen, host_properties['ZTRUE'],
                                      self.hostlib_mskopt)
                if 'ZPHOT' in catalogue.extra_columns else None)
        if field_ra is not None and field_dec is not None:
            field_ra = jnp.asarray(field_ra)
            field_dec = jnp.asarray(field_dec)
            cos_dec_field = jnp.cos(jnp.radians(field_dec))

        def slot(rank):
            """The properties of DDLR rank ``rank``; a rank the neighbour list
            never reaches is an unmatched slot."""
            if rank < n_ranks:
                gal_idx = sorted_gal_idx[:, rank]
                ddlr, snsep = sorted_ddlr[:, rank], sorted_snsep[:, rank]
                valid = sorted_valid[:, rank]
            else:
                gal_idx = jnp.zeros(n_sne, dtype=sorted_gal_idx.dtype)
                ddlr = snsep = jnp.full(n_sne, jnp.inf, dtype=sorted_ddlr.dtype)
                valid = jnp.zeros(n_sne, dtype=bool)
            if rank > 0:
                # SNANA fills a neighbour slot only below MAXDDLR2
                valid = valid & (ddlr < self.maxddlr2)
            props = catalogue.columns[:, gal_idx]
            if rank > 0:
                props = jnp.where(valid[None, :], props,
                                  self.neighbour_sentinels[:, None])
            out = {vn: props[i] for i, vn in enumerate(self.varnames)}
            if rank == 0:
                out['DDLR'], out['SNSEP'] = ddlr, snsep
                out['DDLR_TRUE'], out['SNSEP_TRUE'] = true_ddlr, true_snsep
            else:
                out['DDLR'] = jnp.where(valid, ddlr, jnp.float32(-9.0))
                out['SNSEP'] = jnp.where(valid, snsep, jnp.float32(-9.0))
            out['ZSPEC_TRUEMATCH'] = (
                (gal_idx == galaxy_indices) & valid).astype(jnp.float32)
            if zdif is not None:
                zp, ze = shifted_catalogue_zphot(
                    jnp.take(catalogue.extra_columns['ZPHOT'], gal_idx),
                    jnp.take(catalogue.extra_columns['ZPHOT_ERR'], gal_idx),
                    zdif)
                out['ZPHOT'] = jnp.where(valid, zp, SNANA_NULLFLOAT)
                out['ZPHOT_ERR'] = jnp.where(valid, ze, SNANA_NULLFLOAT)
            # HOSTLIB space to the SN's field frame; unmatched slots keep sentinels
            if field_ra is not None and field_dec is not None:
                ra_lib, dec_lib = out['RA_GAL'], out['DEC_GAL']
                cos_dec_nbr = jnp.cos(jnp.radians(dec_lib))
                delta_ra = (ra_lib - ra_sn) * (cos_dec_nbr / (cos_dec_field + 1e-30))
                out['RA_GAL'] = jnp.where(valid, field_ra + delta_ra, ra_lib)
                out['DEC_GAL'] = jnp.where(valid, field_dec + dec_lib - dec_sn, dec_lib)
            out['reported_galaxy_index'] = jnp.where(valid, gal_idx,
                                                     jnp.int32(-1))
            return out

        hostgal_sorted = slot(0)
        neighbours = [slot(rank) for rank in range(1, MXHOSTGAL)]
        return hostgal_sorted, neighbours, nmatch, nmatch2

    def select(self, catalogue, sn_redshifts, random_uniforms,
               position_uniforms, sn_properties, peak_mjds, field_indices,
               pec1a, *, reuse, reselect_keys, reuse_lanes):
        """Select host galaxies from ``catalogue`` for supernovae and
        generate SN positions.

        ``sn_redshifts`` and ``random_uniforms`` (uniforms in [0, 1) driving
        the host draw) are shape (n_sne,). ``position_uniforms`` is shape
        (n_sne, 4), [component_choice, radius, u1, u2] per SN.
        ``sn_properties`` holds the SN variables a multi-bin WGTMAP reads,
        None when it reads none. ``field_indices`` gives each event's SIMLIB
        entry, whose GROUPID set it draws from. ``pec1a`` flags the events
        drawn at the DNDZ_PEC1A rate, which under SN2GAL_Z read their own
        rows of the marginal CDF.

        Under HOSTLIB_GALID_FORCE every event takes the forced galaxy, or a
        GALID_PRIORITY galaxy inside its DZTOL window when there is one.

        Under USEONCE or MINDAYSEP_SAMEGAL a blocked event redraws from its
        own ``reselect_keys``; ``peak_mjds`` gives the events' peaks for the
        day window, ``reuse`` is the run's record of taken galaxies or
        passing peaks (:meth:`initial_reuse`) and ``reuse_lanes`` the events
        that register against it: repeat followers and padding lanes do not.

        Returns ``(galaxy_indices, host_properties, reuse, report)``: the
        selected galaxy indices, shape (n_sne,), a dict of host properties
        including positions, the updated ``reuse``, and what the draw
        reports beside them: ``unplaced``, the events USEONCE could not
        place, on which SNANA aborts; ``rejected``, the events MINDAYSEP
        left without a host, which SNANA rejects (sntools_host.c:6430-6468);
        ``wgtmap_snvar_outside``, from :meth:`snvar_outside`;
        ``host_window_empty``, from :func:`empty_host_windows`;
        ``host_zdif``, the SN minus forced-host redshift under
        HOSTLIB_GALID_FORCE; and ``host_window_z``, the redshift those two
        are judged at. The run makes the aborts these call for.
        """
        n_sne = len(sn_redshifts)
        report = {}

        if self.igal_force >= 0:
            galaxy_indices = jnp.full(n_sne, self.igal_force, dtype=jnp.int32)
            # a GALID_PRIORITY galaxy in the window wins over the forced one
            if self.galid_priority is not None:
                galaxy_indices = self._apply_galid_priority(
                    galaxy_indices, sn_redshifts)
            snvar_bin_indices = jnp.full(n_sne, 0, dtype=jnp.int32)
            host_properties = self.gather_host_properties(
                catalogue, galaxy_indices, snvar_bin_indices)
            host_properties = self._add_position_properties(
                catalogue, host_properties, galaxy_indices,
                position_uniforms)
            if not self.hostlib_mskopt & 16:
                report['host_zdif'] = (sn_redshifts
                                       - catalogue.redshifts[galaxy_indices])
                report['host_window_z'] = sn_redshifts
            return galaxy_indices, host_properties, reuse, report

        snvar_bin_indices = self._compute_snvar_bin_indices(n_sne, sn_properties)
        if self.has_snvars:
            report['wgtmap_snvar_outside'] = self.snvar_outside(sn_properties,
                                                                n_sne)
        if catalogue.marginal_cdf is None:
            report['host_window_empty'] = empty_host_windows(
                sn_redshifts, self._rows(snvar_bin_indices, field_indices),
                catalogue.redshifts, catalogue.wgtsum, self.dztol_coeffs,
                self.n_bins)
            report['host_window_z'] = sn_redshifts

        def draw(uniforms):
            return self._select_hosts_one_pass(
                catalogue, sn_redshifts, uniforms, snvar_bin_indices,
                field_indices, pec1a)

        galaxy_indices = draw(random_uniforms)
        # a redraw under a reuse constraint takes the plain weights
        if self.galid_priority is not None:
            galaxy_indices = self._apply_galid_priority(
                galaxy_indices, sn_redshifts)

        if self.useonce:
            def blocked_fn(galaxy, taken_and_resolved):
                taken, resolved = taken_and_resolved
                won, taken = _useonce_claims(galaxy, ~resolved, taken)
                resolved = resolved | won
                return ~resolved, (taken, resolved)

            galaxy_indices, report['unplaced'], (taken, _) = _reselect(
                draw, galaxy_indices, reselect_keys, blocked_fn,
                (reuse['taken'], ~reuse_lanes))
            reuse = {**reuse, 'taken': taken}
        elif self.uses_mindaysep:
            day = self.peak_day(peak_mjds)
            order = jnp.lexsort((reuse['day'], reuse['galaxy']))
            hist_gal, hist_day = reuse['galaxy'][order], reuse['day'][order]

            def blocked_fn(galaxy, state):
                return _mindaysep_blocked(galaxy, day, reuse_lanes, hist_gal,
                                          hist_day, self.mindaysep), state

            galaxy_indices, report['rejected'], _ = _reselect(
                draw, galaxy_indices, reselect_keys, blocked_fn, ())

        host_properties = self.gather_host_properties(
            catalogue, galaxy_indices, snvar_bin_indices)
        host_properties = self._add_position_properties(
            catalogue, host_properties, galaxy_indices, position_uniforms)

        return galaxy_indices, host_properties, reuse, report

    def snvar_outside(self, sn_properties, n_sne):
        """Per event, which WGTMAP SN variable lies outside the map's range,
        counted from one in ``snvar_names`` order; zero where none does.
        SNANA aborts on such an event (getBin_SNVAR_HOSTLIB_WGTMAP,
        sntools_host.c:1655-1660); the run makes that call."""
        outside = jnp.zeros(n_sne, dtype=jnp.int32)
        for i in reversed(range(len(self.snvar_names))):
            values = sn_properties[self.snvar_names[i]]
            centers = self.snvar_bin_centers[i]
            outside = jnp.where((values < centers[0]) | (values > centers[-1]),
                                i + 1, outside)
        return outside

    def initial_reuse(self, capacity: int) -> dict:
        """A run's empty reuse record: the taken mask over the catalogue
        under USEONCE; under MINDAYSEP_SAMEGAL room for ``capacity`` passing
        events' (galaxy, peak day), unfilled slots on galaxy -1."""
        if self.useonce:
            return {'taken': jnp.zeros(self.n_galaxies, dtype=bool)}
        if self.uses_mindaysep:
            return {'galaxy': jnp.full(capacity, -1, dtype=jnp.int32),
                    'day': jnp.zeros(capacity, dtype=jnp.int32),
                    'n': jnp.int32(0)}
        return {}

    def peak_day(self, peak_mjds):
        """The whole day of each peak, counted from the start of
        GENRANGE_PEAKMJD, that the MINDAYSEP window compares."""
        return jnp.floor(peak_mjds).astype(jnp.int32) - self.peakmjd_offset

    def record_peaks(self, reuse, galaxy_indices, peak_days, register):
        """Add the ``register`` events' galaxies and peak days
        (:meth:`peak_day`) to the MINDAYSEP history, which blocks the
        batches that follow. SNANA frees a rejected event's galaxy
        (FREEHOST_GALID, snlc_sim.c:13512), so the caller registers passing
        events only."""
        position = reuse['n'] + jnp.cumsum(register.astype(jnp.int32)) - 1
        capacity = reuse['galaxy'].shape[0]
        slot = jnp.where(register, position, capacity)
        return {'galaxy': reuse['galaxy'].at[slot].set(galaxy_indices,
                                                       mode='drop'),
                'day': reuse['day'].at[slot].set(peak_days, mode='drop'),
                'n': reuse['n'] + jnp.sum(register).astype(jnp.int32)}


def gather_deferred_hostlib_columns(host, hostlib, catalogue, params):
    """JIT-safe gather of the output-only HOSTLIB columns from ``catalogue``
    (the HostCatalogue passed into compiled code), indexed by galaxy_idx.
    ``host`` is the input's host keys and ``hostlib`` the built HOSTLIB,
    None without one.

    Returns a new dict of additional per-SN arrays (`host_<col>`, gathered
    at the drawn host only; ZPHOT columns the simulation computes are
    skipped). Caller merges into the sn_vars / params dict it uses for
    SEARCHEFF.
    """
    if hostlib is None or 'host_galaxy_index' not in params:
        return {}
    columns = {name: catalogue.extra_columns[name]
               for name in hostlib.output_only_columns
               if name in catalogue.extra_columns}
    gal_idx = jnp.asarray(params['host_galaxy_index'], dtype=jnp.int32)
    calc_zphot_keys = set()
    if hostlib.genzphot is not None:
        calc_zphot_keys = {'ZPHOT', 'ZPHOT_ERR'}
        if host.mskopt & HOSTLIB_MSKOPT_ZPHOT_QGAUSS:
            calc_zphot_keys.update(name for name, _ in hostlib.zphot_quantiles)
            calc_zphot_keys.update(['P50', 'IQR'])

    gathered = {}
    for col_name, col_full in columns.items():
        if col_name in calc_zphot_keys:
            continue
        gathered[f'host_{col_name}'] = col_full[gal_idx]
    return gathered


jax.tree_util.register_dataclass(
    HostSelector,
    data_fields=[
        'dztol_coeffs',
        'galid_priority',
        'field_row_block',
        'snvar_bin_centers',
        'radec_rows',
        'dlr_rows',
        'neighbour_sentinels',
        'detect_thresholds',
        'sersic_table',
    ],
    meta_fields=[
        'n_galaxies',
        'n_bins',
        'varnames',
        'useonce',
        'uses_mindaysep',
        'mindaysep',
        'peakmjd_offset',
        'igal_force',
        'pec1a_row_shift',
        'has_snvars',
        'snvar_names',
        'hostlib_mskopt',
        'maxddlr',
        'maxddlr2',
        'smear_sersic',
        'use_dlr_columns',
        'has_neighbours',
        'snr_scale_ranges',
        'fixran_wgt',
        'fixran_radius',
        'fixran_phi',
        'mnintflux_snpos',
        'mxintflux_snpos',
    ],
)
