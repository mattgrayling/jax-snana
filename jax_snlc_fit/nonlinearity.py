"""SNANA NONLINEARITY_FILE (sntools_nonlinearity.c): per-filter detector
non-linearity applied to FLUXCAL at fit time (EXEC_FUDGE_FLUXCAL, GET_NONLIN).

The map gives a flux-scale F_meas/F_true as a function of total counts (e-),
interpolated in log10(Ftot). Two application modes (OPTMASK_NONLIN bits):
PER_NEA (bit 4) applies the scale to Ftot over the noise-equivalent aperture;
PER_PIX (bit 8) integrates the non-linearity over an effective Gaussian PSF.
COUNT_RATE (bit 2) divides the counts by the exposure time first.
"""

import os

import numpy as np

from jax_snlc_core.io.fits_path import open_text
from jax_snlc_core.utils.error_handling import SNANAError

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
