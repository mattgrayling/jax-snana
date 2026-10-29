"""Photometric calibration: the filters and zero points a fit reads (the
kcor file and its calibration shifts), each event's Milky Way E(B-V)
(OPT_MWEBV) with the colour law and the attenuation they give each epoch,
the conversion of the Python interface's flux units to FLUXCAL, and the
corrections applied to the data as they load: FUDGE_FLUXCAL,
FLUXCAL_BASELINE_CORRECT, FLUXERRMODEL, host noise, DATAERR_SCALE,
nonlinearity, MAGCOR and MAGOBS_SHIFT_ZP. The loader applies them in SNANA's
order.

FUDGE_HOSTNOISE_FILE (sntools_fluxErrModels_legacy.c) is the legacy host
surface-brightness noise: per band and field, an ERRSCALE curve against the
host SB magnitude inflates the flux error (``err *= ERRSCALE``), with
SBMAG = ZP - 2.5 log10(SB flux).

NONLINEARITY_FILE (sntools_nonlinearity.c) is the per-filter detector
non-linearity applied to FLUXCAL as it loads (EXEC_FUDGE_FLUXCAL, GET_NONLIN).
The map gives a flux scale F_meas/F_true as a function of total counts (e-),
interpolated in log10(Ftot). Two application modes (OPTMASK_NONLIN bits):
PER_NEA (bit 4) applies the scale to Ftot over the noise-equivalent aperture;
PER_PIX (bit 8) integrates the non-linearity over an effective Gaussian PSF.
COUNT_RATE (bit 2) divides the counts by the exposure time first.

MAGCOR_FILE and SIM_MAGCOR_FILE shift individual epochs' magnitudes.
"""

import dataclasses
import logging
import os

import jax.numpy as jnp
import numpy as np
import pandas as pd

from jax_snlc_core import logging as log
from jax_snlc_core.dust import OPT_MWCOLORLAW_NAMES, SFDDustMap, load_dust_law
from jax_snlc_core.filters import (ZEROPOINT_FLUXCAL_SNANA, KcorReader, filter_band,
                                   filter_values, load_photometry_resources, read_calib_shifts,
                                   resolve_calib_shifts)
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.models import metadata_files
from jax_snlc_core.noise import (apply_fluxcal_fudges, evaluate_fluxerrmodel_data,
                                 noise_equivalent_aperture)
from jax_snlc_core.snana_files import open_text
from jax_snlc_core.utils import LIGHT_A

logger = logging.getLogger(__name__)


def read_hostnoise_file(path):
    """Parse a FUDGE_HOSTNOISE_FILE (INIT_NOISEMODEL_HOST_LEGACY) into per-LIBID
    maps: {band, field, HOSTMAG grid, ERRSCALE grid}."""
    maps, cur = [], None
    with open_text(os.path.expandvars(path)) as fh:
        for line in fh:
            toks = line.split('#')[0].split()
            if not toks:
                continue
            key = toks[0].rstrip(':')
            if key == 'LIBID':
                cur = {'band': '', 'field': 'ALL', 'mag': [], 'noise': []}
                maps.append(cur)
            elif key == 'BAND':
                cur['band'] = toks[1]
            elif key == 'FIELD':
                cur['field'] = toks[1]
            elif key == 'HOSTNOISE':
                cur['mag'].append(float(toks[1]))
                cur['noise'].append(float(toks[2]))
    for m in maps:
        m['mag'], m['noise'] = np.array(m['mag']), np.array(m['noise'])
    return maps


def _errscale(m, sbmag):
    """ERRSCALE of one map at each host SB mag: bright-end linear
    extrapolation, dim-end clamp to the last bin, floored at 1.0
    (GEN_NOISEMODEL_HOST_LEGACY)."""
    g, n = m['mag'], m['noise']
    bright = n[0] + (sbmag - g[0]) * (n[1] - n[0]) / (g[1] - g[0])
    val = np.where(sbmag < g[0], bright, np.interp(sbmag, g, n))
    return np.maximum(val, 1.0)


def sbmag_from_fluxcal(sb_fluxcal, zp_fluxcal):
    """Host surface-brightness magnitude from its FLUXCAL (SET_SBMAG,
    snana.F90:18032). An undefined surface brightness reads as 99."""
    sb_fluxcal = np.asarray(sb_fluxcal, dtype=float)
    return np.where(sb_fluxcal <= -999.0, 99.0,
                    zp_fluxcal - 2.5 * np.log10(np.maximum(sb_fluxcal, 1.0)))


def hostnoise_errscale(maps, bands, fields, sb_fluxcal, zp_fluxcal):
    """Per-epoch ERRSCALE from the host SB flux (HOSTGAL_SB_FLUXCAL) in each
    epoch's band. Band and field match SNANA's strstr semantics (a map entry
    contains the epoch value); epochs are grouped by (band, field) so each map
    interpolates once."""
    sbmag = sbmag_from_fluxcal(sb_fluxcal, zp_fluxcal)
    bands = np.asarray(bands)
    fields = (np.asarray(fields) if fields is not None
              else np.full(len(bands), ''))
    out = np.ones(len(bands))
    groups = pd.DataFrame({'band': bands.astype(str), 'field': fields.astype(str)})
    for (b, f), rows in groups.groupby(['band', 'field']).indices.items():
        matches = [m for m in maps if b in m['band'] and f in m['field']]
        if len(matches) != 1:
            raise SNANAError(
                f"Expected one host-noise map for band '{b}', field '{f}'; "
                f"found {len(matches)}. Check FUDGE_HOSTNOISE_FILE.",
                function_name="hostnoise_errscale")
        out[rows] = _errscale(matches[0], sbmag[rows])
    return out


OPTMASK_COUNT_TOT = 1
OPTMASK_COUNT_RATE = 2
OPTMASK_PER_NEA = 4
OPTMASK_PER_PIX = 8


class NonlinModel:
    """Parsed NONLINEARITY_FILE: per-band log10(Ftot)->flux-scale curves."""

    def __init__(self, optmask, maps):
        self.optmask = optmask
        self.maps = maps  # band char -> (log10_ftot grid, fscale grid)

    def scale_for_band(self, band, ftot):
        """Flux-scale at each epoch's total counts, linear in log10(Ftot).

        A non-positive flux takes 1.0, as SNANA does
        (get_flux_scale_NONLIN, sntools_nonlinearity.c:427). A band no map
        covers also takes 1.0, where SNANA returns 0.0 and so zeroes that
        band's flux (:455).
        """
        grid = self.maps.get(band)
        if grid is None:
            return np.ones_like(ftot)
        xp, yp = grid
        x = np.log10(np.maximum(ftot, 1e-30))
        # SNANA aborts outside the map (sntools.c:7628)
        outside = (np.asarray(ftot) > 0.0) & ((x < xp[0]) | (x > xp[-1]))
        if np.any(outside):
            counts = np.asarray(ftot)[outside].ravel()[0]
            raise SNANAError(
                f"NONLINEARITY_FILE map for band {band} covers "
                f"{10.0 ** xp[0]:.4g} to {10.0 ** xp[-1]:.4g} e-, and an "
                f"epoch has {float(counts):.4g} e-.",
                function_name="scale_for_band")
        idx = np.clip(np.searchsorted(xp, x) - 1, 0, len(xp) - 2)
        slope = (yp[idx + 1] - yp[idx]) / (xp[idx + 1] - xp[idx])
        return np.where(ftot > 0.0, yp[idx] + slope * (x - xp[idx]), 1.0)


def read_nonlinearity_file(path):
    """Parse a NONLINEARITY_FILE into a NonlinModel (one map per FILTERS
    block), with SNANA's validation (sntools_nonlinearity.c:95-99,
    :178-215): MODELNAME is required, and exactly one bit of each of the
    COUNT and PER pairs must be set."""
    optmask, maps, modelname = 0, {}, None
    bands, pairs = None, []

    def flush():
        if bands and pairs:
            arr = np.array(sorted(pairs), dtype=np.float64)
            grid = (np.log10(arr[:, 0]), arr[:, 1])
            for b in bands:
                maps[b] = grid

    with open_text(os.path.expandvars(path)) as fh:
        for line in fh:
            toks = line.split('#')[0].split()
            if not toks:
                continue
            key = toks[0].rstrip(':')
            if key == 'OPTMASK_NONLIN':
                optmask = int(toks[1])
            elif key == 'MODELNAME':
                modelname = toks[1]
            elif key == 'FILTERS':
                flush()
                bands, pairs = toks[1], []
            elif key == 'NONLIN':
                pairs.append((float(toks[1]), float(toks[2])))
    flush()

    if modelname is None:
        raise SNANAError(
            f"NONLINEARITY_FILE {path} must specify MODELNAME: <model>.",
            function_name="read_nonlinearity_file")
    n_count = bool(optmask & OPTMASK_COUNT_TOT) + bool(optmask & OPTMASK_COUNT_RATE)
    n_per = bool(optmask & OPTMASK_PER_NEA) + bool(optmask & OPTMASK_PER_PIX)
    if n_count != 1 or n_per != 1:
        raise SNANAError(
            f"Invalid OPTMASK_NONLIN={optmask}: require exactly one of "
            f"+=1(COUNT-TOTAL)/+=2(COUNT-RATE) and one of "
            f"+=4(PER-NEA)/+=8(PER-PIX).",
            function_name="read_nonlinearity_file")
    return NonlinModel(optmask, maps)


def _per_pix_scales(model, band, fpe_source, fpe_sky, nea):
    """PER_PIX flux-scales for the epochs sharing one (band, nea): integrate
    the non-linearity over an effective Gaussian PSF pixel grid
    (GET_NONLIN, sntools_nonlinearity.c:322-403), vectorised over epochs and
    pixels. Uses full np.pi; SNANA's truncated 3.14159 differs by <=0.06%
    here."""
    nea_over_pi = nea / np.pi
    fpix_sky = fpe_sky / nea                       # [m]
    sigsq = 0.25 * nea_over_pi
    rsq_border = nea_over_pi
    x_half = int(0.5 * np.sqrt(4.0 * nea_over_pi)) + 1.0
    xs = np.arange(-x_half, x_half, 1.0)
    xx, yy = np.meshgrid(xs, xs, indexing='ij')
    rsq = xx * xx + yy * yy
    psf = np.exp(-0.5 * rsq[rsq < rsq_border] / sigsq)
    psf = psf / psf.sum()                          # [npix]
    fpix_sky_nl = fpix_sky * model.scale_for_band(band, fpix_sky)
    fpix_tot = fpe_source[:, None] * psf[None, :] + fpix_sky[:, None]
    fpix_tot_nl = fpix_tot * model.scale_for_band(band, fpix_tot)
    num = np.sum((fpix_tot_nl - fpix_sky_nl[:, None]) * psf[None, :], axis=1)
    den = fpe_source * np.sum(psf * psf)
    return np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), 1.0)


def evaluate_nonlinearity(model, fluxcal, zeropt, gain, nea, skysig, texpose,
                          bands, zp_fluxcal):
    """Per-epoch flux-scale from the non-linearity model (F_meas/F_true)."""
    fluxadu = fluxcal * np.power(10.0, 0.4 * (zeropt - zp_fluxcal))
    fpe_source = np.where(fluxadu * gain < 0.0, 0.01, fluxadu * gain)
    fpe_sky = nea * (skysig * gain) ** 2
    if model.optmask & OPTMASK_COUNT_RATE:
        if np.any(texpose < 2.0):
            bad = int(np.argmax(texpose < 2.0))
            raise SNANAError(
                f"Invalid Texpose = {float(np.asarray(texpose).ravel()[bad]):.3f} for "
                f"COUNT-RATE non-linearity.",
                function_name="evaluate_nonlinearity")
        fpe_source = fpe_source / texpose
        fpe_sky = fpe_sky / texpose

    bands = np.asarray(bands)
    scale = np.ones_like(fluxcal)
    if model.optmask & OPTMASK_PER_NEA:
        ftot = fpe_source + fpe_sky
        for b in np.unique(bands):
            m = bands == b
            scale[m] = model.scale_for_band(b, ftot[m])
    elif model.optmask & OPTMASK_PER_PIX:
        # the pixel grid is fixed by nea, so epochs group by (band, nea)
        keys = np.char.add(bands.astype(str), np.asarray(nea).astype(str))
        for key in np.unique(keys):
            m = keys == key
            scale[m] = _per_pix_scales(model, bands[m][0], fpe_source[m],
                                       fpe_sky[m], float(np.asarray(nea)[m][0]))
    return np.where(fpe_sky < 0.0, 1.0, scale)


def nonlinearity_scales(model, flux, phot_data, phot_indices, bands, zp_fluxcal):
    """Each epoch's NONLINEARITY_FILE flux scale (GET_NONLIN), from the PHOT
    columns of the epochs at ``phot_indices``. The count-rate mode needs
    their TEXPOSE."""
    def col(name):
        return np.asarray(phot_data[name], dtype=np.float64)[phot_indices]
    nea = noise_equivalent_aperture(col('PSF_SIG1'), col('PSF_SIG2'), col('PSF_RATIO'))
    if model.optmask & OPTMASK_COUNT_RATE:
        if 'TEXPOSE' not in phot_data.colnames:
            raise SNANAError(
                "NONLINEARITY count-rate mode (OPTMASK_NONLIN bit 2) "
                "requires a TEXPOSE column", function_name="GET_NONLIN")
        texpose = col('TEXPOSE')
    else:
        texpose = np.ones(len(phot_indices))
    return evaluate_nonlinearity(model, flux, col('ZEROPT'), col('GAIN'), nea,
                                 col('SKY_SIG'), texpose, bands, zp_fluxcal)


# A correction beyond this is a mistake, not a correction (snana.F90:16898).
MAGCOR_MAX = 0.2


def magcor_scales(corrections, cids, mjds, bands):
    """Each epoch's MAGCOR flux scale, 10^(-0.4 MAGCOR) (EXEC_MAGCOR), aborting
    on a correction above MAGCOR_MAX."""
    magcor = magcor_values(corrections, cids, mjds, bands)
    too_large = np.abs(magcor) > MAGCOR_MAX
    if too_large.any():
        k = int(np.argmax(too_large))
        raise SNANAError(
            f"MAGCOR = {magcor[k]:.4g} at {cids[k]}-{mjds[k]:.3f}-{bands[k]} "
            f"exceeds {MAGCOR_MAX} mag", function_name="EXEC_MAGCOR")
    return 10.0 ** (-0.4 * magcor)


def read_magcor_file(path, version):
    """SNANA MAGCOR_FILE (INIT_MAGCOR): FITRES-format rows keyed
    ``[CID]-[MJD]-[BAND]`` with a MAGCOR column; other columns ignored, bar
    WAVECOR, the per-epoch wavelength shift, which is refused. A leading '-'
    on the path subtracts MAGCOR instead of adding it. VERSION_ADD
    rows name the data versions already carrying the correction: adding to
    one of them, or subtracting from any other, is refused (ISTAT_MAGCOR_ADD,
    snana.F90:16750-16760). Returns {(cid, round(mjd, 3), band): signed_magcor}."""
    sign = 1.0
    if path.startswith('-'):
        sign, path = -1.0, path[1:]
    varnames, out, added = None, {}, set()
    with open(os.path.expandvars(path)) as fh:
        for line in fh:
            toks = line.split('#')[0].split()
            if not toks:
                continue
            if toks[0] == 'VARNAMES:':
                varnames = toks[1:]
                if 'WAVECOR' in varnames:
                    raise SNANAError(
                        f"MAGCOR_FILE {path} carries a WAVECOR column, the "
                        f"per-epoch wavelength shift, which jax does not "
                        f"implement.", function_name='read_magcor_file')
            elif toks[0] == 'VERSION_ADD:':
                added.add(toks[1])
            elif toks[0] in ('ROW:', 'SN:'):
                cid, mjd, band = toks[1].rsplit('-', 2)
                mag = float(toks[1 + varnames.index('MAGCOR')])
                out[(cid, round(float(mjd), 3), band)] = sign * mag
    if sign > 0.0 and version in added:
        raise SNANAError(
            f"MAGCOR already in the {version} fluxes: will not add {path} twice",
            function_name='read_magcor_file')
    if sign < 0.0 and version not in added:
        raise SNANAError(
            f"MAGCOR not in the {version} fluxes: will not subtract {path}",
            function_name='read_magcor_file')
    return out


def magcor_values(corrections, cids, mjds, bands):
    """Match correction keys in bulk, preserving Python's timestamp rounding."""
    rounded = np.fromiter((round(float(mjd), 3) for mjd in mjds),
                          dtype=float, count=len(mjds))
    values = pd.Series(list(corrections.values()),
                       index=pd.MultiIndex.from_tuples(corrections))
    epochs = pd.MultiIndex.from_arrays([cids, rounded, bands])
    return values.reindex(epochs, fill_value=0.0).to_numpy()


def resolve_mwebv(mwebvs, mwebv_errs, ras, decs, mwebv):
    """The Galactic E(B-V) and error every later step reads, set as the light
    curve is read (snana.F90:17195-17205): OPT_MWEBV takes none (0), the
    header's (1), the SFD98 map at RA/DEC with a sixth of itself as error (2)
    or 0.86 of it with five percent (3) (modify_MWEBV_SFD, MWgaldust.c), then
    MWEBV_SCALE, MWEBV_SHIFT and MWEBV_FORCE apply."""
    opt = mwebv['opt']
    if opt == 0:
        ebv, err = np.zeros_like(mwebvs), np.zeros_like(mwebvs)
    elif opt == 1:
        ebv, err = mwebvs, mwebv_errs
    elif opt in (2, 3):
        if np.any(np.isnan(ras)):
            raise SNANAError(f"OPT_MWEBV={opt} needs RA/DEC in the HEAD file",
                             function_name="modify_MWEBV_SFD")
        ebv = np.asarray(mwebv['sfd_map'].get_ebv_batch(ras, decs),
                         dtype=np.float64)
        if opt == 3:
            ebv = 0.86 * ebv
            err = 0.05 * ebv
        else:
            err = ebv / 6.0
    else:
        raise SNANAError(f"Unsupported OPT_MWEBV={opt}",
                         function_name="modify_MWEBV_SFD")
    ebv = ebv * mwebv['scale'] + mwebv['shift']
    err = err * mwebv['scale']
    if mwebv['force'] is not None:
        ebv = np.full_like(ebv, mwebv['force'])
        err = np.zeros_like(err)
    return ebv, err


def mwebv_options(config):
    """OPT_MWEBV and its modifiers from the Data record, with the SFD dust
    map reader when the E(B-V) comes from the map. The maps are 64 MB each,
    so the reader is built once and passed to every data version."""
    d = config.data
    return {"opt": d.opt_mwebv, "scale": d.mwebv_scale,
            "shift": d.mwebv_shift, "force": d.mwebv_force,
            "sfd_map": SFDDustMap() if d.opt_mwebv in (2, 3) else None}


# the colour law (FITZ99_EXACT) and R_V the fitter takes for real data
OPT_MWCOLORLAW_DEFAULT = 99
RV_MWCOLORLAW_DEFAULT = 3.1


def resolve_mw_colour_law(data_config, sim_mw_colour_law):
    """(OPT_MWCOLORLAW, RV_MWCOLORLAW): each the user's where set, else the
    simulation's own from the data header, else the default for real data."""
    opt_file, rv_file = (sim_mw_colour_law if sim_mw_colour_law is not None
                         else (OPT_MWCOLORLAW_DEFAULT, RV_MWCOLORLAW_DEFAULT))
    opt = (opt_file if data_config.opt_mwcolorlaw is None
           else data_config.opt_mwcolorlaw)
    rv = (rv_file if data_config.rv_mwcolorlaw is None
          else data_config.rv_mwcolorlaw)
    if opt not in OPT_MWCOLORLAW_NAMES:
        raise SNANAError(
            f"Invalid SIMOPT_MWCOLORLAW = {opt} in the data header; valid "
            f"codes: {', '.join(str(c) for c in sorted(OPT_MWCOLORLAW_NAMES))}.",
            function_name="INIT_GALextinct")
    return opt, rv


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


def resolved_filters(config, kcor):
    """The fit-band names, FILTLIST_FIT's single characters resolved against
    the kcor's own filter names."""
    if config.fit.filtlist_fit is None:
        raise SNANAError("FILTLIST_FIT is required",
                         function_name="resolved_filters")
    return resolve_filtlist(config.fit.filtlist_fit, kcor)


def shift_map_with_poly(base, params, filter_lamavg):
    """A per-band mag-shift map with its A0 + A1*lam + A2*lam^2 companion key
    folded in, lam being the filter's mean wavelength in microns
    (MAGOBS_SHIFT_USRFUN, snana.F90:11337)."""
    shift = dict(base)
    if not any(params):
        return shift
    a0, a1, a2 = params
    values = filter_values(base, list(filter_lamavg), 0.0)
    for (name, lamavg), value in zip(filter_lamavg.items(), values):
        lam_um = lamavg / 1.0e4
        shift[name] = value + a0 + a1 * lam_um + a2 * lam_um ** 2
    return shift


def zero_point_shift_scales(shifts, bands):
    """Each band's flux scale, 10^(-0.4 shift), under a per-band zero-point
    shift such as MAGOBS_SHIFT_ZP (snana.F90:17369)."""
    return 10.0 ** (-0.4 * np.asarray(filter_values(shifts, bands, 0.0)))


def calibration_bands(values, kcor, band_map):
    """``values``, band labels, as the kcor filter names they map to through
    ``band_map``; a label is its own name when the map leaves it out."""
    available = set(kcor.filter_names)
    result = []
    for value in np.asarray(values).astype(str).tolist():
        value = band_map.get(value, value)
        if value not in available:
            raise SNANAError(f"Unknown band {value!r}; use a full calibration name or band_map")
        result.append(value)
    return np.asarray(result)


def _unit_scales(units, zero_points, zp_fluxcal):
    """Validate object units and return their scale before band calibration."""
    units = np.asarray(units)
    zero_points = np.asarray(zero_points, dtype=float)
    supplied = ~np.isnan(zero_points)
    if np.any(supplied & ~np.isfinite(zero_points)):
        raise SNANAError('zp must be finite')
    if not np.isin(units, ('fluxcal', 'Jy', 'uJy')).all():
        raise SNANAError('flux_unit must be fluxcal, Jy or uJy')
    calibrated = units == 'fluxcal'
    if np.any(~calibrated & supplied):
        raise SNANAError('zp is only meaningful with flux_unit="fluxcal"')
    zp = np.where(supplied, zero_points, ZEROPOINT_FLUXCAL_SNANA)
    return np.where(calibrated, 10.**(0.4 * (zp_fluxcal - zp)),
                    np.where(units == 'uJy', 1.e-6, 1.))


def _jy_factors(bands, kcor, model_wave, zp_fluxcal):
    """Calibrate Jy once for each distinct band in a set of observations."""
    if not len(bands):
        return np.empty(0)
    names = list(dict.fromkeys(bands))
    resources = load_photometry_resources(kcor, model_wave, names)
    factors = {}
    for i, name in enumerate(names):
        wave, trans = map(np.asarray, kcor.get_filter_transmission(name))
        jy_ref = np.trapezoid(1.e-23 * LIGHT_A / wave**2 * wave * trans, wave)
        jy_ref /= np.trapezoid(wave * trans, wave)
        factors[name] = jy_ref / float(resources.filter_ref_fluxes[i]) * 10.**(0.4 * zp_fluxcal)
    return np.asarray([factors[name] for name in bands])


def flux_factor(unit, zp, bands, kcor, model_wave, *, zp_fluxcal):
    """Convert flux and error from the supplied units to the batch's zero point."""
    scale = _unit_scales(unit, np.nan if zp is None else zp, zp_fluxcal)
    if unit == 'fluxcal':
        return np.full(len(bands), scale)
    return scale * _jy_factors(bands, kcor, model_wave, zp_fluxcal)


def normalise_flux(obs, obj, kcor, model_wave):
    """Keep the supplied zero point, converting other units to the same scale."""
    obs[['flux', 'flux_err']] = obs[['flux', 'flux_err']].astype(float)
    units = obj['flux_unit'].to_numpy() if 'flux_unit' in obj else np.full(len(obj), 'fluxcal')
    zeros = obj['zp'].to_numpy(dtype=float) if 'zp' in obj else np.full(len(obj), np.nan)
    supplied_zeros = zeros[(units == 'fluxcal') & ~np.isnan(zeros)]
    zp_fluxcal = float(supplied_zeros[0]) if len(supplied_zeros) else ZEROPOINT_FLUXCAL_SNANA
    scales = _unit_scales(units, zeros, zp_fluxcal)
    indices = pd.Index(obj.snid).get_indexer(obs.snid)
    factors = scales[indices]
    needs_calibration = units[indices] != 'fluxcal'
    factors[needs_calibration] *= _jy_factors(
        obs.loc[needs_calibration, 'band'].to_numpy(), kcor, model_wave, zp_fluxcal)
    obs['flux'] = obs.flux.to_numpy() * factors
    obs['flux_err'] = obs.flux_err.to_numpy() * factors
    input_units = {snid: (unit, None if np.isnan(zero) else float(zero))
                   for snid, unit, zero in zip(obj.snid, units, zeros)}
    return input_units, zp_fluxcal


def fit_photometry(config, model, kcor):
    """The fit bands of ``kcor`` as a data version's namelist calibrates
    them, on the common grid ``model`` is evaluated through: FILTER_LAMSHIFT
    and MAGOBS_SHIFT_PRIMARY applied, with the calibration shifts the
    model's training wrote into its metadata files.

    Returns the arrays, in fit-band order, of the photometry the compiled
    fit takes as an argument: ``common_filter_wave_grid``,
    ``common_filter_trans_grid``, ``filter_ref_fluxes`` and
    ``filter_mean_wavelengths``. The data's ``zp_fluxcal`` joins them when
    a batch is loaded.
    """
    filters = resolved_filters(config, kcor)
    lam_shift_map = config.data.filter_lamshift
    primary_mag_shift_map = shift_map_with_poly(
        config.data.magobs_shift_primary,
        config.data.magobs_shift_primary_params, kcor.filter_lamavg(filters))
    model_path = os.path.expandvars(os.path.expanduser(
        config.model.fitmodel_name))
    info_shifts = [s for path in metadata_files(type(model), model_path)
                   for s in read_calib_shifts(path)]
    if info_shifts:
        mag_s, lam_s = resolve_calib_shifts(
            info_shifts, filters, kcor.filter_survey,
            enable_mag=config.data.enable_magshift,
            enable_wave=config.data.enable_waveshift)
        primary_mag_shift_map = {**primary_mag_shift_map, **mag_s}
        lam_shift_map = {**lam_shift_map, **lam_s}
        log.line(logger, f"Model calib shifts applied: "
                 f"{len(mag_s)} mag, {len(lam_s)} wave")
    resources = load_photometry_resources(
        kcor, model.wavelength_grid, filters, lam_shift_map=lam_shift_map,
        primary_mag_shift_map=primary_mag_shift_map)
    return {"common_filter_wave_grid": jnp.asarray(resources.common_wave_grid),
            "common_filter_trans_grid": jnp.asarray(resources.common_trans_grid),
            "filter_ref_fluxes": jnp.asarray(resources.filter_ref_fluxes),
            "filter_mean_wavelengths": jnp.asarray(
                resources.filter_mean_wavelengths)}


def attach_mw_attenuation(config, batch):
    """Attach Milky Way attenuation and extinction errors to a fit batch."""
    attenuation, errors = mw_attenuation(config, batch.photometry, batch.mwebvs,
                                        batch.mwebv_errs, batch.sim_mw_colour_law)
    return dataclasses.replace(batch, mw_att_obs=attenuation, mwxt_magerrs=errors)


def mw_attenuation(config, photometry, mwebvs, mwebv_errs, sim_mw_colour_law=None):
    """Return observer-frame Milky Way attenuation and magnitude errors for each SN's
    E(B-V), as the loader resolved it from OPT_MWEBV, on the common grid of
    its photometry, and the extinction-magnitude error at each band's mean
    wavelength, where SNANA evaluates it (snana.F90:15720-15742).

    ``config`` is the data version's: its OPT_MWCOLORLAW, RV_MWCOLORLAW and
    PARLIST_MWCOLORLAW, the simulation's own law where it sets none. The
    error is the extinction at E(B-V) plus its error less that at E(B-V)
    (snana.F90:15624-15629). A batch with neither carries ones and zero
    errors, and the law is not evaluated, as GALextinct returns at A_V = 0
    (MWgaldust.c:474).
    """
    opt, rv = resolve_mw_colour_law(config.data, sim_mw_colour_law)
    law = load_dust_law(OPT_MWCOLORLAW_NAMES[opt])
    if config.data.parlist_mwcolorlaw:
        law.set_params(config.data.parlist_mwcolorlaw)
    wave = np.asarray(photometry['common_filter_wave_grid'])
    trans = np.asarray(photometry['common_filter_trans_grid'])
    ebv = np.asarray(mwebvs, dtype=np.float64)
    ebv_err = np.asarray(mwebv_errs, dtype=np.float64)
    if not (np.any(ebv != 0.0) or np.any(ebv_err != 0.0)):
        return jnp.ones((len(ebv), len(wave))), jnp.zeros((len(ebv), trans.shape[0]))

    # the law must cover the fit bands, where SNANA tabulates it
    # (fill_TABLE_MWXT_SEDMODEL, genmag_SEDtools.c; MWgaldust.c:688-694)
    law.prepare_wavelength_grid(wave)
    covered = wave[(trans > 0).any(axis=0)]
    law.check_wave_range(float(covered.min()), float(covered.max()))
    lamavg = np.asarray(photometry['filter_mean_wavelengths'])
    hi = np.clip(np.searchsorted(wave, lamavg), 1, len(wave) - 1)
    lo = hi - 1
    frac = (lamavg - wave[lo]) / (wave[hi] - wave[lo])

    def axav(e):
        """A(lam)/A(V) on the grid for each event's E(B-V) ``e``, and A(lam)
        at each band's mean wavelength; Somm25's shape reads AV = RV*E(B-V)
        (genmag_SEDtools.c:2580-2645)."""
        curve = np.atleast_2d(np.asarray(law.extinction_curve(
            wave, rv, av=rv * e[:, None])))
        at_lamavg = curve[..., lo] + (curve[..., hi] - curve[..., lo]) * frac
        return curve, rv * e[:, None] * at_lamavg

    curve, xt = axav(ebv)
    _, xt_plus_err = axav(ebv + ebv_err)
    attenuation = jnp.power(10.0, -0.4 * rv * jnp.asarray(ebv)[:, None]
                            * jnp.asarray(curve))
    return attenuation, jnp.asarray(xt_plus_err - xt)


def read_kcor(config):
    """KCOR_FILE, with LEAKAGE_CUT applied."""
    return KcorReader(os.path.expandvars(config.data.kcor_file),
                      leakage_cut=config.data.leakage_cut)


# PSF sigma to FWHM (SET_EPVAR_MISC, snana.F90:13963)
FWHM_PER_SIGMA = 2.355


def combined_flux_errors(err1, add_var, kept_flux, bands_e, fields_e,
                         sn_idx_dense, kept_phot_indices, head_data, phot_data,
                         phot_mjd, is_sim, fudge_flux, dataerr_scale,
                         fluxerr_model, sim_fluxerr_model, hostnoise_maps,
                         zp_fluxcal):
    """The per-epoch flux error: additive fudges, FLUXERRMODEL and HOSTNOISE
    in quadrature off the scaled-original ERR1, then DATAERR_SCALE."""
    fem = sim_fluxerr_model if is_sim else fluxerr_model
    if (fudge_flux or dataerr_scale != 1.0 or fem is not None
            or hostnoise_maps is not None):
        var = err1 ** 2 + add_var
        if fem is not None:  # get_FLUXERRMODEL (DATA path)
            col = lambda c: np.asarray(phot_data[c], dtype=np.float64)[kept_phot_indices]
            pixsize = (np.asarray(head_data['PIXSIZE'], dtype=np.float64)[sn_idx_dense]
                       if 'PIXSIZE' in head_data.colnames else np.ones(len(sn_idx_dense)))
            snr = np.where(err1 > 0, kept_flux / np.where(err1 > 0, err1, 1.0), 0.0)
            cond = {'MJD': phot_mjd[kept_phot_indices],
                    'PSF': col('PSF_SIG1') * pixsize * FWHM_PER_SIGMA,
                    'SKYSIG': col('SKY_SIG'), 'ZP': col('ZEROPT'),
                    'LOGSNR': np.maximum(np.log10(np.maximum(snr, 1e-9)), -0.9)}
            err_data = evaluate_fluxerrmodel_data(fem, bands_e, fields_e, err1, cond)
            var = var + err_data ** 2 - err1 ** 2
        if hostnoise_maps is not None:  # FUDGE_HOSTNOISE_FILE (host SB errscale)
            sbf = host_surface_brightness(head_data, bands_e, sn_idx_dense)
            scale = hostnoise_errscale(hostnoise_maps, bands_e, fields_e, sbf,
                                       zp_fluxcal)
            var = var + err1 ** 2 * (scale ** 2 - 1.0)
        kept_err = np.sqrt(var) * dataerr_scale
    else:
        kept_err = err1
    return kept_err


def host_surface_brightness(head_data, bands, sn_indices):
    """Gather each epoch's host brightness with one bulk lookup per band."""
    out = np.empty(len(bands), dtype=np.float64)
    for band, rows in pd.Series(sn_indices).groupby(bands).indices.items():
        sb = np.asarray(head_data[f'HOSTGAL_SB_FLUXCAL_{band}'], dtype=np.float64)
        out[rows] = sb[sn_indices[rows]]
    return out


def apply_flux_fudges(flux, err, band_codes, unique_bands, phot_data,
                       phot_indices, fudge_flux, is_sim):
    """Apply SNANA FUDGE_FLUXCAL_* fudges (EXEC_FUDGE_FLUXCAL) to flat per-epoch
    calibrated flux/error arrays. Each fudge is a per-band map (band char ->
    value); err_pix couples to the PSF noise-equivalent aperture; SIM_FUDGE_MAG_
    ERROR adds a further mag-error term only for sim data. Returns (flux, err1,
    add_var): the scaled-original error and the additive fudge variance."""

    def band_vals(name, default):  # per-filter map, with '' meaning all bands
        spec = fudge_flux.get(name, {})
        per_code = np.array(filter_values(spec, unique_bands, default))
        return per_code[band_codes]

    mag_err = band_vals('mag_err', 0.0)
    if is_sim:
        mag_err = np.sqrt(mag_err**2 + band_vals('sim_mag_err', 0.0)**2)

    err_pix = band_vals('err_pix', 0.0)
    nea = 0.0
    if np.any(err_pix != 0.0):
        col = lambda c: np.asarray(phot_data[c], dtype=np.float64)[phot_indices]
        nea = noise_equivalent_aperture(col('PSF_SIG1'), col('PSF_SIG2'),
                                        col('PSF_RATIO'))
    return apply_fluxcal_fudges(
        flux, err, band_vals('offset', 0.0), band_vals('err_scale', 1.0),
        band_vals('err_add', 0.0), err_pix, mag_err, nea)


def flux_baselines(flux, err, group, n_groups):
    """Each group's flux baseline for FLUXCAL_BASELINE_CORRECT: the median
    flux of its epochs below -3 sigma, where it has at least four of them,
    and zero elsewhere. The error is the standard error of a median, 1.2533
    times that of their weighted mean."""
    sel = flux < -3.0 * err
    g, f = group[sel], flux[sel]
    n = np.bincount(g, minlength=n_groups)
    ivar = np.bincount(g, weights=err[sel] ** -2, minlength=n_groups)
    ordered = f[np.lexsort((f, g))]  # by group, then by flux
    start = np.cumsum(n) - n
    use = n >= 4
    base = np.zeros(n_groups)
    base_err = np.zeros(n_groups)
    base[use] = 0.5 * (ordered[start[use] + (n[use] - 1) // 2]
                       + ordered[start[use] + n[use] // 2])
    base_err[use] = 1.2533 / np.sqrt(ivar[use])
    return base, base_err
