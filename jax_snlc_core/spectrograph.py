"""The spectrograph: its calibration, read from a table or the kcor FITS file,
solved, and used for the S/N of a spectrum.

Reads a spectrograph calibration table (INSTRUMENT, MAGREF_LIST, TEXPOSE_LIST,
SPECBIN rows) and solves for the zero point (ZP) and sky plus readout noise
(SQSIGSKY) per wavelength bin and exposure time; or reads the solved
calibration from a kcor FITS file, or builds an ideal one. Gives the S/N for
any magnitudes and exposure times, and a synthetic band's zero point, PSF
and sky.

File I/O and calibration solving use numpy (one-time setup).
Runtime SNR computation uses JAX for JIT compilation.

SNANA reference: sntools_spectrograph.c (solve_spectrograph, getSNR_spectrograph)
"""

import logging
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core import logging as log
from jax_snlc_core.interpolation import linear_between, linear_bin
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import open_text, resolve_fits_path, skip_docana
from jax_snlc_core.utils import HC_ERG_A

logger = logging.getLogger(__name__)

# Half-width, in resolution sigmas, of the edge extension (extend_spectrograph_lambins)
NSIG_EXTEND = 2.5

# Solve-consistency default (init_INPUTS_SPECTRO, sntools_spectrograph.c:308);
# overridable by the same-named key in the spectrograph file.
MAGSNR_TOLERANCE_ABORT = 0.001


@dataclass
class SpectrographCalibration:
    """Solved spectrograph calibration.

    All array fields are jnp.ndarray for use in JIT-compiled functions.
    """
    instrument_name: str
    magref: jnp.ndarray             # (2,) reference magnitudes [bright, faint]
    n_lam: int
    n_texpose: int
    lam_min: jnp.ndarray            # (n_lam,) bin lower edges [Angstroms]
    lam_max: jnp.ndarray            # (n_lam,) bin upper edges
    lam_avg: jnp.ndarray            # (n_lam,) bin centres
    lam_bin: jnp.ndarray            # (n_lam,) bin widths
    lam_sigma: jnp.ndarray          # (n_lam,) wavelength resolution (Gaussian sigma)
    texpose_list: jnp.ndarray       # (n_texpose,) exposure times [seconds]
    log_texpose_list: jnp.ndarray   # (n_texpose,) log10(exposure times)
    zp: jnp.ndarray                 # (n_lam, n_texpose) zero-point [magnitudes]
    sqsigsky: jnp.ndarray           # (n_lam, n_texpose) sky+readout noise variance
    zp_sedmodel: Optional[jnp.ndarray] = None  # (n_lam,) per-bin ZP for genmag; filled from the kcor primary at load
    islam_extend: Optional[jnp.ndarray] = None  # (n_lam,) bool; edge bins that smear inward but are never observed


def parse_spectrograph_filename(value):
    """Split SNANA's optional filename(rebin=N) specification."""
    if '(' not in value:
        return value, 1
    match = re.fullmatch(r'(.+)\(rebin=(\d+)\)', value)
    if match is None or int(match[2]) < 1:
        raise SNANAError(f'Invalid spectrograph filename or rebin option: {value!r}')
    return match[1], int(match[2])


def read_spectrograph(filepath: str, *, rebin: int = 1) -> SpectrographCalibration:
    """Read and calibrate a spectrograph from a text file.

    File I/O and calibration solving use numpy. Results are converted to
    jnp.ndarray for downstream JIT-compiled operations.

    Returns the SpectrographCalibration with solved ZP and SQSIGSKY as JAX
    arrays.
    """
    filename, filename_rebin = parse_spectrograph_filename(str(filepath))
    if filename_rebin != 1:
        if rebin != 1 and rebin != filename_rebin:
            raise SNANAError('Conflicting spectrograph rebin options')
        rebin = filename_rebin
    if not isinstance(rebin, int) or rebin < 1:
        raise SNANAError('Spectrograph rebin must be a positive integer')
    expanded = Path(resolve_fits_path(Path(os.path.expandvars(filename)).expanduser()))
    if not expanded.exists():
        raise SNANAError(f"Spectrograph file not found: {expanded}",
                         function_name="read_spectrograph")

    instrument = 'UNKNOWN'
    magref = None
    texpose_list = None
    specbin_rows = []
    magsnr_tolerance = MAGSNR_TOLERANCE_ABORT
    # Poisson-consistency abort tolerances (init_INPUTS_SPECTRO,
    # sntools_spectrograph.c:305-306)
    snr_ratio_abort_vsmagref = 5.0
    snr_ratio_abort_vstexpose = 1.2

    with open_text(expanded) as f:
        for line in skip_docana(f):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue

            parts = stripped.split()
            key = parts[0].rstrip(':')

            if key == 'INSTRUMENT':
                instrument = parts[1] if len(parts) > 1 else 'UNKNOWN'
            elif key == 'MAGREF_LIST':
                magref = np.array([float(parts[1]), float(parts[2])])
            elif key == 'TEXPOSE_LIST':
                texpose_list = np.array([float(x) for x in parts[1:]])
            elif key == 'MAGSNR_TOLERANCE_ABORT':
                magsnr_tolerance = float(parts[1])
            elif key == 'SNR_POISSON_RATIO_ABORT_vsMAGREF':
                snr_ratio_abort_vsmagref = float(parts[1])
            elif key == 'SNR_POISSON_RATIO_ABORT_vsTEXPOSE':
                snr_ratio_abort_vstexpose = float(parts[1])
            elif key == 'SPECBIN':
                vals = [float(x) for x in parts[1:]]
                specbin_rows.append(vals)

    if magref is None:
        raise SNANAError(f"MAGREF_LIST not found in {filepath}",
                         function_name="read_spectrograph")
    if texpose_list is None:
        raise SNANAError(f"TEXPOSE_LIST not found in {filepath}",
                         function_name="read_spectrograph")
    # equal neighbours pass, as in SNANA's read_TEXPOSE_LIST
    if np.any(np.diff(texpose_list) < 0.0):
        raise SNANAError(f"TEXPOSE_LIST must be in increasing order in {filepath}: "
                         f"{texpose_list.tolist()}", function_name="read_spectrograph")
    if not specbin_rows:
        raise SNANAError(f"No SPECBIN rows found in {filepath}",
                         function_name="read_spectrograph")

    n_texpose = len(texpose_list)
    specbin = np.array(specbin_rows)
    if rebin > 1:
        # SNANA combines complete groups: bin edges span the group and SNRs
        # add in quadrature. Incomplete trailing groups are not retained.
        n_groups = len(specbin) // rebin
        if n_groups < 2:
            raise SNANAError('Spectrograph rebin leaves fewer than two wavelength bins')
        groups = specbin[:n_groups * rebin].reshape(n_groups, rebin, -1)
        rebinned = np.zeros((n_groups, specbin.shape[1]))
        rebinned[:, 0] = groups[:, 0, 0]
        rebinned[:, 1] = groups[:, -1, 1]
        rebinned[:, 3:] = np.sqrt(np.sum(groups[:, :, 3:]**2, axis=1))
        specbin = rebinned
    n_lam = len(specbin)

    lam_min = specbin[:, 0]
    lam_max = specbin[:, 1]
    lam_sigma = specbin[:, 2]
    lam_avg = 0.5 * (lam_min + lam_max)
    lam_bin = lam_max - lam_min

    snr_data = specbin[:, 3:]
    expected_snr_cols = 2 * n_texpose
    if snr_data.shape[1] != expected_snr_cols:
        raise SNANAError(
            f"SPECBIN has {snr_data.shape[1]} SNR columns, "
            f"expected {expected_snr_cols} (2 × {n_texpose} exposures)",
            function_name="read_spectrograph")

    snr0 = snr_data[:, 0::2]
    snr1 = snr_data[:, 1::2]

    _check_snr_poisson(snr0, snr1, magref, texpose_list, lam_avg,
                       snr_ratio_abort_vsmagref, snr_ratio_abort_vstexpose)
    _check_snr_monotonic(snr0, snr1, texpose_list, lam_avg)

    zp, sqsigsky = _solve_calibration(snr0, snr1, magref[0], magref[1],
                                      magsnr_tolerance_abort=magsnr_tolerance)

    log_texpose = np.log10(texpose_list)

    log.line(logger, f"Loaded spectrograph '{instrument}': "
             f"{n_lam} wavelength bins ({lam_min[0]:.0f}-{lam_max[-1]:.0f} Å), "
             f"{n_texpose} exposure times ({texpose_list[0]:.0f}-{texpose_list[-1]:.0f} s)")

    return extend_lambins(SpectrographCalibration(
        instrument_name=instrument,
        magref=jnp.array(magref),
        n_lam=n_lam,
        n_texpose=n_texpose,
        lam_min=jnp.array(lam_min),
        lam_max=jnp.array(lam_max),
        lam_avg=jnp.array(lam_avg),
        lam_bin=jnp.array(lam_bin),
        lam_sigma=jnp.array(lam_sigma),
        texpose_list=jnp.array(texpose_list),
        log_texpose_list=jnp.array(log_texpose),
        zp=jnp.array(zp),
        sqsigsky=jnp.array(sqsigsky),
    ))


def extend_lambins(calib: SpectrographCalibration) -> SpectrographCalibration:
    """Add edge bins that feed the lambda smear (extend_spectrograph_lambins,
    sntools_spectrograph.c:1285).

    The extra bins carry model flux and smear it inward, but are never written
    out and never receive smeared flux. Sums over the bin grid therefore run
    wider than the observed spectrum, which is what host contamination does.
    """
    n_orig = calib.n_lam
    lam_min = np.asarray(calib.lam_min, dtype=float)
    lam_max = np.asarray(calib.lam_max, dtype=float)
    lam_bin = np.asarray(calib.lam_bin, dtype=float)
    lam_sigma = np.asarray(calib.lam_sigma, dtype=float)
    zp = np.asarray(calib.zp)
    sqsigsky = np.asarray(calib.sqsigsky)

    # SNANA sizes each side from its own edge bin, skipping the very last one
    # in case it is short, and always adds at least one bin.
    n_blue = int(lam_sigma[0] * NSIG_EXTEND / lam_bin[0]) + 1
    n_red = int(lam_sigma[n_orig - 2] * NSIG_EXTEND / lam_bin[n_orig - 2]) + 1

    blue_src, red_src = 0, n_orig - 2
    blue_bin, red_bin = lam_bin[blue_src], lam_bin[red_src]
    blue_lo = lam_min[blue_src] - blue_bin * np.arange(n_blue, 0, -1)
    red_lo = lam_min[n_orig - 1] + red_bin * np.arange(1, n_red + 1)

    new_min = np.concatenate([blue_lo, lam_min, red_lo])
    new_max = np.concatenate([blue_lo + blue_bin, lam_max, red_lo + red_bin])
    src = np.concatenate([np.full(n_blue, blue_src), np.arange(n_orig),
                          np.full(n_red, red_src)])
    extend = np.concatenate([np.ones(n_blue, bool), np.zeros(n_orig, bool),
                             np.ones(n_red, bool)])

    log.line(logger, f"Extend spectrograph NBLAM = {n_orig} -> {len(new_min)} "
             f"(+{n_blue} blue, +{n_red} red) for wave-resolution")

    return replace(
        calib,
        n_lam=len(new_min),
        lam_min=jnp.asarray(new_min), lam_max=jnp.asarray(new_max),
        lam_avg=jnp.asarray(0.5 * (new_min + new_max)),
        lam_bin=jnp.asarray(lam_bin[src]),
        lam_sigma=jnp.asarray(lam_sigma[src]),
        zp=jnp.asarray(zp[src]), sqsigsky=jnp.asarray(sqsigsky[src]),
        islam_extend=jnp.asarray(extend),
    )


def _check_snr_poisson(snr0, snr1, magref, texpose_list, lam_avg,
                       tol_vsmagref, tol_vstexpose):
    """Abort when a table SNR is far off its Poisson expectation
    (check_SNR_SPECTROGRAPH, sntools_spectrograph.c:391-503): SNR2/SNR1 must
    lie within a factor tol_vsmagref of sqrt(fluxref2/fluxref1), and each
    SNR(Texpose)/SNR(Texpose_prev) within tol_vstexpose of
    sqrt(Texpose/Texpose_prev). Non-positive SNR values are invalid. Error
    text uses the manual's SNR1/SNR2 naming."""
    if (snr0 < 1.0e-12).any() or (snr1 < 1.0e-12).any():
        l, t = np.argwhere((snr0 < 1.0e-12) | (snr1 < 1.0e-12))[0]
        raise SNANAError(
            f"Invalid/negative SNR[1,2] = {snr0[l, t]:.3e}, {snr1[l, t]:.3e} "
            f"at LAM={lam_avg[l]:.1f}, Texpose={texpose_list[t]:.2f}",
            function_name="_check_snr_poisson")

    ratio_poisson = np.sqrt(10.0 ** (0.4 * (magref[0] - magref[1])))
    log_ratio = np.abs(np.log10((snr1 / snr0) / ratio_poisson))
    if (log_ratio > np.log10(tol_vsmagref)).any():
        l, t = np.argwhere(log_ratio > np.log10(tol_vsmagref))[0]
        raise SNANAError(
            f"Crazy SNR2/SNR1 ratio (way off from Poisson) at "
            f"LAM={lam_avg[l]:.1f}, Texpose={texpose_list[t]:.2f}: "
            f"SNR2/SNR1={snr1[l, t] / snr0[l, t]:.4f} vs Poisson "
            f"{ratio_poisson:.4f} (|log10 ratio| "
            f"{log_ratio[l, t]:.3f} > {np.log10(tol_vsmagref):.3f})",
            function_name="_check_snr_poisson")

    ratio_poisson_t = np.sqrt(texpose_list[1:] / texpose_list[:-1])
    for name, snr in (("SNR1", snr0), ("SNR2", snr1)):
        log_ratio = np.abs(np.log10(
            (snr[:, 1:] / snr[:, :-1]) / ratio_poisson_t[None, :]))
        if (log_ratio > np.log10(tol_vstexpose)).any():
            l, t = np.argwhere(log_ratio > np.log10(tol_vstexpose))[0]
            raise SNANAError(
                f"Crazy {name} vs. Texpose at LAM={lam_avg[l]:.1f}: "
                f"{name}({texpose_list[t]:.0f}s)={snr[l, t]:.4f}, "
                f"{name}({texpose_list[t + 1]:.0f}s)={snr[l, t + 1]:.4f}; "
                f"ratio off Poisson sqrt(Texpose) by |log10| "
                f"{log_ratio[l, t]:.3f} > {np.log10(tol_vstexpose):.3f}",
                function_name="_check_snr_poisson")


def _check_snr_monotonic(snr0, snr1, texpose_list, lam_avg):
    """Abort when a table SNR does not increase with Texpose
    (check_SNR_SPECTROGRAPH, sntools_spectrograph.c:489-497). Error text uses
    the manual's SNR1/SNR2 naming."""
    for name, snr in (("SNR1", snr0), ("SNR2", snr1)):
        bad = snr[:, 1:] <= snr[:, :-1]
        if bad.any():
            l, t = np.argwhere(bad)[0]
            raise SNANAError(
                f"{name} is not increasing with Texpose at "
                f"LAM={lam_avg[l]:.1f}, Texpose={texpose_list[t + 1]:.2f}; "
                f"check the SPECTROGRAPH table",
                function_name="_check_snr_monotonic")


def _solve_calibration(
    snr0: np.ndarray,
    snr1: np.ndarray,
    mag0: float,
    mag1: float,
    magsnr_tolerance_abort: float = MAGSNR_TOLERANCE_ABORT,
) -> tuple:
    """
    Solve for ZP and SQSIGSKY from SNR reference table.

    Uses numpy; called once at load time.

    Matches SNANA solve_spectrograph (sntools_spectrograph.c:648-783),
    including its aborts: SNR at or below 1e-9; the MAGREF-vs-SNR consistency
    check against MAGSNR_TOLERANCE_ABORT (default +0.001, :308); TOP/BOT <= 0;
    non-finite ZP/SQSIGSKY; and a per-point round-trip error above 0.001 for
    either reference magnitude. The SNR1 hack (:719-730) inflates SNR1 when
    magCheck < 0, which is reachable only when the spectrograph file sets a
    negative MAGSNR_TOLERANCE_ABORT.
    """
    def _abort(msg):
        raise SNANAError(f"{msg}; check the SPECTROGRAPH table",
                         function_name="_solve_calibration")

    if (snr0 <= 1e-9).any() or (snr1 < 1e-9).any():
        bad = np.argwhere((snr0 <= 1e-9) | (snr1 < 1e-9))[0]
        _abort(f"Invalid SNR[0,1] = {snr0[tuple(bad)]:.3e}, "
               f"{snr1[tuple(bad)]:.3e} at (l={bad[0]}, t={bad[1]})")

    powmag0 = 10.0 ** (-0.4 * mag0)
    powmag1 = 10.0 ** (-0.4 * mag1)
    TOP = powmag0 - powmag1

    magref_dif = mag1 - mag0
    magsnr_dif = 2.5 * np.log10(snr0 / snr1)
    mag_check = magref_dif - magsnr_dif
    if (mag_check < magsnr_tolerance_abort).any():
        bad = np.argwhere(mag_check < magsnr_tolerance_abort)[0]
        _abort(f"failed solution check at (l={bad[0]}, t={bad[1]}): "
               f"mref1-mref0 = {magref_dif:f} < 2.5log10(SNR0/SNR1) = "
               f"{magsnr_dif[tuple(bad)]:f}")

    needs_fix = mag_check < 0.0
    snr1_fixed = snr1.copy()
    snr1_fixed[needs_fix] = 1.005 * (10.0 ** (-0.4 * magref_dif)) * snr0[needs_fix]

    dum0 = powmag0 / snr0
    dum1 = powmag1 / snr1_fixed
    BOT = dum0**2 - dum1**2

    if (TOP <= 0.0) or (BOT <= 0.0).any():
        _abort(f"Cannot solve ZP: TOP = {TOP:.3e}, min(BOT) = {BOT.min():.3e}")

    zp = 2.5 * np.log10(TOP / BOT)
    f0 = 10.0 ** (-0.4 * (mag0 - zp))
    f1 = 10.0 ** (-0.4 * (mag1 - zp))
    # may be negative (SNR above Poisson limit); SNANA allows this
    sqsigsky = (f0 / snr0)**2 - f0

    if np.isinf(zp).any() or np.isinf(sqsigsky).any():
        _abort("inf ZP or SQSIGSKY")

    # per-point round-trip: solved ZP/SQSIGSKY must reproduce both input SNRs
    for snr_in, f in ((snr0, f0), (snr1_fixed, f1)):
        snr_check = f / np.sqrt(sqsigsky + f)
        err = np.abs(snr_in / snr_check - 1.0)
        if (err > 0.001).any():
            bad = np.argwhere(err > 0.001)[0]
            _abort(f"Problem computing ZP and SQSIGSKY at (l={bad[0]}, "
                   f"t={bad[1]}): round-trip error {err[tuple(bad)]:.4f}")

    return zp, sqsigsky


def build_ideal_spectrograph(
    lammin: float,
    lammax: float,
    lambin: float,
) -> SpectrographCalibration:
    """Fabricate the IDEAL spectrograph for SEDMODEL true-SED spectra
    (create_ideal_spectrograph, sntools_spectrograph.c:1658)."""
    ilam_lo = int(lammin / lambin)
    lam_min_tot = lambin * (ilam_lo - 2)
    ilam_hi = int(lammax / lambin)
    lam_max_tot = lambin * (ilam_hi + 3)
    n_lam = int((lam_max_tot - lam_min_tot + 1e-5) / lambin)

    lam_min = lam_min_tot + lambin * np.arange(n_lam)
    lam_max = lam_min + lambin

    texpose_list = np.array([10.0, 100.0])
    magref = np.array([18.0, 25.0])
    n_texpose = len(texpose_list)
    snr0 = np.full((n_lam, n_texpose), 1.0e5)
    snr1 = np.full((n_lam, n_texpose), 3.0e5)
    zp, sqsigsky = _solve_calibration(snr0, snr1, magref[0], magref[1])

    log.line(logger, f"Built IDEAL spectrograph: {n_lam} wavelength bins "
             f"({lam_min_tot:.1f}-{lam_max_tot:.1f} Å), BinSize={lambin:.0f} Å")

    return SpectrographCalibration(
        instrument_name='IDEAL',
        magref=jnp.array(magref),
        n_lam=n_lam,
        n_texpose=n_texpose,
        lam_min=jnp.array(lam_min),
        lam_max=jnp.array(lam_max),
        lam_avg=jnp.array(0.5 * (lam_min + lam_max)),
        lam_bin=jnp.full(n_lam, lambin),
        lam_sigma=jnp.zeros(n_lam),
        texpose_list=jnp.array(texpose_list),
        log_texpose_list=jnp.array(np.log10(texpose_list)),
        zp=jnp.array(zp),
        sqsigsky=jnp.array(sqsigsky),
    )


@jax.jit
def _interp_rows(x: float, xp: jnp.ndarray, fp: jnp.ndarray) -> jnp.ndarray:
    """Interpolate all rows of fp at a single scalar x. JIT-compiled.

    ``x`` is the scalar query point; ``xp`` is (n_xp,) sorted breakpoints;
    ``fp`` is (n_rows, n_xp) values per row. Returns the (n_rows,)
    interpolated values.
    """
    idx, frac = linear_bin(x, xp)
    return linear_between(fp[:, idx], fp[:, idx + 1], frac)


def synthetic_band_zptpsfsky(calib, lammin, lammax, texpose_s, texpose_t, pixsize,
                             unit_arcsec=False):
    """Effective ZP/PSF/SKYSIG for a synthetic top-hat band [lammin, lammax] from the
    spectrograph calib (get_SPECTROGRAPH_ZPTPSFSKY, snlc_sim.c): ZP/SQSIGSKY summed
    over the bins whose centre falls in the band, interpolated linearly in TEXPOSE.
    Returns (zpt, psfsig, skysig_s, skysig_t, nea)."""
    sel = (np.asarray(calib.lam_avg) >= lammin) & (np.asarray(calib.lam_avg) <= lammax)
    if not sel.any():
        raise SNANAError(
            f"No spectrograph bin centre in synthetic band [{lammin}, {lammax}]")

    def band(texpose, table):  # interpolate every bin at texpose, keep the band's bins
        return np.asarray(_interp_rows(texpose, calib.texpose_list, table))[sel]

    zpt = 2.5 * np.log10(np.sum(10.0 ** (0.4 * band(texpose_s, calib.zp))))
    psfsig = (1.0 / 2.355) if unit_arcsec else (1.0 / 2.355) / pixsize  # artificial 1" FWHM
    nea = 4.0 * np.pi * psfsig * psfsig
    skysig_s = np.sqrt(np.sum(band(texpose_s, calib.sqsigsky)) / nea)
    skysig_t = 0.0
    if texpose_t > 0.01:  # IFLAG_TEMPLATE: a spectrograph template is defined
        skysig_t = (np.sqrt(np.sum(band(texpose_t, calib.sqsigsky)) / nea)
                    * texpose_s / texpose_t)
    return float(zpt), float(psfsig), float(skysig_s), float(skysig_t), float(nea)


@jax.jit
def get_snr(
    log_texpose_list: jnp.ndarray,
    texpose_list: jnp.ndarray,
    zp: jnp.ndarray,
    sqsigsky: jnp.ndarray,
    texpose_search: float,
    texpose_template: float,
    genmag: jnp.ndarray,
) -> tuple:
    """
    Compute SNR per wavelength bin. JIT-compiled.

    Matches SNANA getSNR_spectrograph():

    - ZP interpolated in log10(TEXPOSE)
    - SQSIGSKY interpolated in linear TEXPOSE
    - Template noise scaled: ``SQSIGSKY_T *= 10^(0.8*(ZP_S - ZP_T))``

    ``log_texpose_list``/``texpose_list`` are (n_texpose,) log10 and linear
    exposure times; ``zp``/``sqsigsky`` are (n_lam, n_texpose) zero-points
    and sky noise variance; ``texpose_search``/``texpose_template`` are the
    search/template exposure times in seconds; ``genmag`` is (n_lam,)
    magnitude per bin.

    Returns ``(snr, errfrac_t, zp_s, flux)``, all shape (n_lam,).
    """
    t_min = texpose_list[0]
    t_max = texpose_list[-1]

    # out-of-range TEXPOSE aborts before the trace unless TEXTRAP, which rescales after
    texpose_s_local = jnp.clip(texpose_search, t_min, t_max)
    log_texpose_s = jnp.log10(texpose_s_local)

    zp_s = _interp_rows(log_texpose_s, log_texpose_list, zp)
    sq_s = _interp_rows(texpose_s_local, texpose_list, sqsigsky)

    has_template = texpose_template > 0.01
    log_texpose_t = jnp.log10(jnp.maximum(texpose_template, 0.1))
    zp_t = _interp_rows(log_texpose_t, log_texpose_list, zp)
    sq_t_raw = _interp_rows(texpose_template, texpose_list, sqsigsky)
    sq_t = jnp.where(has_template, sq_t_raw * 10.0 ** (0.8 * (zp_s - zp_t)), 0.0)

    valid = (zp_s > -5.0) & (genmag < 90.0)
    flux = jnp.where(valid, 10.0 ** (-0.4 * (genmag - zp_s)), 0.0)

    sq_sum = sq_s + sq_t + flux
    good = valid & (sq_sum > 0)

    flux_err = jnp.where(good, jnp.sqrt(sq_sum), 0.0)
    snr = jnp.where(good & (flux_err > 0), flux / flux_err, 0.0)

    errfrac_t = jnp.where(good & (sq_t > 0) & (flux_err > 0),
                          jnp.sqrt(sq_t) / flux_err, 0.0)

    return snr, errfrac_t, zp_s, flux


def read_spectrograph_fits(hdul, source) -> Optional[SpectrographCalibration]:
    """Load the pre-solved spectrograph calibration from the kcor FITS file.

    None when the primary header names no SPECTROGRAPH_INSTRUMENT
    (read_spectrograph_fits); when it does, a missing SPECTROGRAPH table,
    header key or column is fatal.
    """
    instrument = hdul[0].header.get('SPECTROGRAPH_INSTRUMENT')
    if instrument is None:
        return None

    def item(container, name):
        try:
            return container[name]
        except KeyError:
            raise SNANAError(
                f"{source} names SPECTROGRAPH_INSTRUMENT "
                f"{instrument} but its SPECTROGRAPH table has no {name}.",
                function_name="read_spectrograph_fits") from None

    table = item(hdul, 'SPECTROGRAPH')
    spec, spec_hdr = table.data, table.header
    n_lam = len(spec)
    n_texpose = item(spec_hdr, 'NBT')
    texpose_list = np.array([
        float(item(spec_hdr, f'TEXPOSE{t:02d}')) for t in range(n_texpose)
    ])
    lam_min = item(spec, 'LAMBIN_MIN').astype(np.float64)
    lam_max = item(spec, 'LAMBIN_MAX').astype(np.float64)
    lam_sigma = item(spec, 'LAMRES').astype(np.float64)
    zp = np.zeros((n_lam, n_texpose))
    sqsigsky = np.zeros((n_lam, n_texpose))
    for it in range(n_texpose):
        zp[:, it] = item(spec, f'ZP{it:02d}')
        sqsigsky[:, it] = item(spec, f'SQSKYSIG{it:02d}')
    calib = extend_lambins(SpectrographCalibration(
        instrument_name=instrument,
        magref=jnp.array([0.0, 0.0]),
        n_lam=n_lam, n_texpose=n_texpose,
        lam_min=jnp.array(lam_min), lam_max=jnp.array(lam_max),
        lam_avg=jnp.array(0.5 * (lam_min + lam_max)),
        lam_bin=jnp.array(lam_max - lam_min),
        lam_sigma=jnp.array(lam_sigma),
        texpose_list=jnp.array(texpose_list),
        log_texpose_list=jnp.array(np.log10(texpose_list)),
        zp=jnp.array(zp), sqsigsky=jnp.array(sqsigsky),
    ))
    log.line(logger, f"Loaded spectrograph '{instrument}' from kcor FITS: "
             f"{n_lam} bins, {n_texpose} exposures")
    return calib


def compute_zp_sedmodel(lam_min, lam_max, kcor) -> np.ndarray:
    """Per-bin spectrograph zero point for SED-model genmag.

    Matches SNANA getZP_SPECTROGRAPH_SEDMODEL (genmag_SEDtools.c:3804):
    ZP = 2.5*log10(sum primFlux(lamCen)*lamCen*lamStep/hc) + magPrimary,
    summed in 10 Å sub-steps, where the primary is the kcor's primary for the
    obs filters and magPrimary is the per-filter "Primary Mag" (ZPoff HDU)
    interpolated at the bin centre (interp_primaryMag_SEDMODEL, clamped at
    the filter wavelength ends).
    """
    filter_names = kcor.get_filter_names()

    primary_name = kcor.filter_primary[filter_names[0]]
    prim_waves, prim_flux = kcor.get_primary_sed(primary_name)
    prim_waves = np.asarray(prim_waves)
    prim_flux = np.asarray(prim_flux)

    mean_lams = np.zeros(len(filter_names))
    prim_mags = np.zeros(len(filter_names))
    for i, name in enumerate(filter_names):
        wave, trans = kcor.get_filter_transmission(name)
        wave = np.asarray(wave)
        trans = np.asarray(trans)
        mean_lams[i] = np.sum(wave * trans) / np.sum(trans)
        prim_mags[i] = kcor.filter_primary_mag[name]
    order = np.argsort(mean_lams)
    mean_lams = mean_lams[order]
    prim_mags = prim_mags[order]

    lam_min = np.asarray(lam_min)
    lam_max = np.asarray(lam_max)
    lamstep = 10.0
    max_steps = int(np.ceil((lam_max - lam_min).max() / lamstep))
    start = lam_min[:, None] + lamstep * np.arange(max_steps)[None, :]
    width = np.clip(lam_max[:, None] - start, 0.0, lamstep)
    lam_cen = start + 0.5 * width
    flux_sum = np.sum(np.interp(lam_cen, prim_waves, prim_flux) * lam_cen * width,
                      axis=1) / HC_ERG_A
    mag_primary = np.interp(0.5 * (lam_min + lam_max), mean_lams, prim_mags)
    zp_sedmodel = np.zeros(len(lam_min))
    pos = flux_sum > 0
    zp_sedmodel[pos] = 2.5 * np.log10(flux_sum[pos]) + mag_primary[pos]
    return zp_sedmodel
