"""
Spectrum generation: TEXPOSE solver, wavelength smearing, Poisson noise.

Architecture: each schedule source (input-file TAKE_SPECTRUM, SIMLIB
SPECTROGRAPH epochs, per-LIBID TAKE_SPECTRUM headers) has a thin adapter that
flattens its schedule into plain per-epoch arrays carrying a stable epoch
identity (global event index, schedule source, entry index). One batch engine,
`_generate_spectra_batch`, then owns SED evaluation, exposure solving, noise,
SNR aggregation and result assembly for every source.

SNANA reference: snlc_sim.c (GENSPEC_DRIVER, GENSPEC_SMEAR, GENSPEC_FLAM,
                              GENSPEC_TEXPOSE_TAKE_SPECTRUM)
"""

import dataclasses
import logging
import os
import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from jax.scipy.special import erf

from jax_snlc_core import logging as log
from jax_snlc_core.filters import filter_band
from jax_snlc_core.input_settings import build_dataclass
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import MAG_ZEROFLUX
from jax_snlc_core.spectrograph import (SpectrographCalibration, build_ideal_spectrograph,
                                        compute_zp_sedmodel, get_snr, read_spectrograph)
from jax_snlc_core.utils import HC_ERG_A, eval_genpoly, eval_genpoly_rows
from jax_snlc_sim.photometry import grey_offsets
from jax_snlc_core.random import event_keys, stream_key
from jax_snlc_sim.scatter import scatter_factors
from jax_snlc_sim.observations import row_slice

logger = logging.getLogger(__name__)


SED_PHASE_EDGE_EPS = 0.01  # genmag_SALT2.c:3564-3565

TOL_CONVERGE = 0.03   # 3% convergence
NLAMSIGMA = 3.0       # kernel half-width in resolution sigmas (snlc_sim.c:1481)
GENRANGE_TOBS_PAD = 0.1  # Tobs padding for the TAKE_SPECTRUM Trest gate (snlc_sim.h:73)


# Sentinels for an unrequested SNR computation (snlc_sim.c:11043).
SNR_COMPUTE_NULL = -99.0
OVERLAP_MIN_SYNFILT = 0.99     # sntools_spectrograph.c:1240
LAMOBS_SNR_NULL = -9.0

# schedule-source codes folded into every epoch's PRNG identity
_SCHEDULE_TAKE_SPECTRUM = 0
_SCHEDULE_SIMLIB_SPECTROGRAPH = 1
_SCHEDULE_SIMLIB_TAKE_SPECTRUM = 2


@dataclass
class SpectrumRequest:
    """One TAKE_SPECTRUM observation and its exposure prescription."""

    field: str = ''
    wgt: float = 1.0
    epoch_frame: Literal['', 'REST', 'OBS', 'MJD', 'HOST'] = ''
    epoch_range: List[float] = dataclasses.field(default_factory=lambda: [0.0, 0.0])
    epoch_step: Optional[float] = None
    opt_texpose: Literal[0, 1, 2] = 0
    zpoly_texpose: List[float] = dataclasses.field(default_factory=list)
    zpoly_snr: List[float] = dataclasses.field(default_factory=list)
    snr_lamrange: List[float] = dataclasses.field(default_factory=lambda: [0.0, 0.0])
    snr_lam_frame: Literal['REST', 'OBS'] = 'REST'


def parse_take_spectrum_entry(key_part: str, value_part: str,
                              from_simlib: bool = False) -> Dict[str, Any]:
    """Read one TAKE_SPECTRUM line without expanding its observation schedule.

    Shared by the input-file parser and the SIMLIB header reader: SNANA runs
    both through parse_input_TAKE_SPECTRUM (snlc_sim.c:22269), so the format
    is identical.

    Format:
        TAKE_SPECTRUM:             TREST(-5:20) SNR_ZPOLY(50,0) SNR_LAMREST(5000:6000)
        TAKE_SPECTRUM(DEEP/2):     TOBS(10:40) TEXPOSE_ZPOLY(3600,0)
        TAKE_SPECTRUM:             HOST TEXPOSE_ZPOLY(1800,0)
    """
    entry = asdict(SpectrumRequest())

    field_match = re.match(r'TAKE_SPECTRUM\(([^)]+)\)', key_part)
    if field_match:
        field_str = field_match.group(1)
        if '/' in field_str:
            field_name, prescale_str = field_str.rsplit('/', 1)
            entry['field'] = field_name
            entry['wgt'] = 1.0 / float(prescale_str)
        else:
            entry['field'] = field_str

    for tok in value_part.split():
        tok_upper = tok.upper()
        inner = re.search(r'\(([^)]+)\)', tok)

        for prefix, frame in (('TREST', 'REST'), ('TOBS', 'OBS'), ('MJD', 'MJD')):
            if tok_upper.startswith(prefix + '('):
                entry['epoch_frame'] = frame
                if inner:
                    parts = inner.group(1).split(':')
                    entry['epoch_range'] = [float(p) for p in parts[:2]]
                    if len(parts) == 1:
                        entry['epoch_range'] = [float(parts[0]), float(parts[0])]
                    if len(parts) >= 3:
                        entry['epoch_step'] = float(parts[2])
                break
        else:
            if tok_upper == 'HOST':
                entry['epoch_frame'] = 'HOST'

        # bare SNR( is an alias for SNR_ZPOLY( (snlc_sim.c:5831)
        if tok_upper.startswith('SNR_ZPOLY(') or tok_upper.startswith('SNR('):
            entry['opt_texpose'] = 2
            if inner:
                entry['zpoly_snr'] = [float(c) for c in inner.group(1).split(',') if c.strip()]

        elif tok_upper.startswith('TEXPOSE_ZPOLY('):
            entry['opt_texpose'] = 1
            if inner:
                entry['zpoly_texpose'] = [float(c) for c in inner.group(1).split(',') if c.strip()]

        if tok_upper.startswith('SNR_LAMREST(') or tok_upper.startswith('LAMREST_SNR('):
            entry['snr_lam_frame'] = 'REST'
            if inner:
                parts = inner.group(1).split(':')
                entry['snr_lamrange'] = [float(parts[0]), float(parts[1])]

        elif tok_upper.startswith('SNR_LAMOBS(') or tok_upper.startswith('LAMOBS_SNR('):
            entry['snr_lam_frame'] = 'OBS'
            if inner:
                parts = inner.group(1).split(':')
                entry['snr_lamrange'] = [float(parts[0]), float(parts[1])]

    # SNANA accepts only TREST/TOBS/HOST for a spectrum epoch; an MJD() range in
    # the SIMLIB header aborts (snlc_sim.c:5807-5812)
    if entry['epoch_frame'] == 'MJD' and from_simlib:
        raise SNANAError(
            f"TAKE_SPECTRUM: MJD() epoch frame is not valid in a SIMLIB header; "
            f"use TREST() or TOBS(). Got {value_part!r}.",
            function_name="parse_take_spectrum_entry")

    return entry


def prepare_spectrum_request(values):
    """Complete and validate the spectrum settings shared by all input forms."""
    entry = asdict(build_dataclass(SpectrumRequest, values))
    for name in ('epoch_range', 'snr_lamrange'):
        limits = entry[name]
        if len(limits) != 2 or not np.isfinite(limits).all() or limits[0] > limits[1]:
            raise SNANAError(f'TAKE_SPECTRUM: {name} needs two finite, ordered bounds')
    if entry['opt_texpose'] == 0:
        raise SNANAError('TAKE_SPECTRUM: expected SNR_ZPOLY or TEXPOSE_ZPOLY')
    if entry['epoch_frame'] == 'HOST' and entry['opt_texpose'] == 2:
        raise SNANAError('Cannot use SNR_ZPOLY with HOST spectrum; only TEXPOSE_ZPOLY allowed for HOST.')
    coefficients = entry['zpoly_texpose'] if entry['opt_texpose'] == 1 else entry['zpoly_snr']
    if not coefficients or not np.isfinite(coefficients).all():
        raise SNANAError('TAKE_SPECTRUM: exposure polynomial needs finite coefficients')
    return entry


def prepare_spectrum_requests(requests):
    """Validate merged requests and expand schedules, sharing each request's draw."""
    entries = []
    for request in requests:
        entry = prepare_spectrum_request(request)
        step = entry.pop('epoch_step')
        # Keep the draw index of the first expanded observation for this request.
        entry['prescale_key'] = len(entries)
        if step is None:
            entries.append(entry)
            continue
        if entry['epoch_frame'] != 'MJD' or entry['opt_texpose'] != 1:
            raise SNANAError('TAKE_SPECTRUM: epoch_step needs MJD and TEXPOSE_ZPOLY')
        if not np.isfinite(step) or step <= 0:
            raise SNANAError('TAKE_SPECTRUM: epoch step must be finite and positive')
        mjd, end = entry['epoch_range']
        while mjd <= end:
            entries.append({**entry, 'epoch_range': [mjd, mjd]})
            mjd += step
    return entries


def simlib_take_spectrum_entries(fields):
    """Parse each LIBID's TAKE_SPECTRUM header lines, with the aborts.

    SNANA runs these through the very same parse_input_TAKE_SPECTRUM as the
    input-file key (snlc_sim.c:22269), so the grammar is shared.
    """
    out = []
    for f in fields:
        entries = []
        for line in f.take_spectrum:
            if line.startswith('WARP_SPECTRUM'):
                raise SNANAError(
                    f"WARP_SPECTRUM in SIMLIB LIBID {f.libid} is not implemented.",
                    function_name="simlib_take_spectrum_entries")
            key_part, _, value_part = line.partition(':')
            # a SIMLIB entry is one spectrum: MJD(min:max:step) is refused
            entry = parse_take_spectrum_entry(key_part.strip(), value_part.strip(),
                                              from_simlib=True)
            if entry['epoch_frame'] == 'HOST':
                raise SNANAError(
                    f"HOST TAKE_SPECTRUM in SIMLIB LIBID {f.libid} needs host SEDs, "
                    f"which jax does not model.",
                    function_name="simlib_take_spectrum_entries")
            entries.append(entry)
        out.append(prepare_spectrum_requests(entries))
    return out


@dataclass
class SpectrumResult:
    """Result for a single generated spectrum. Arrays are numpy for I/O."""
    snid: int
    mjd: float
    texpose: float
    instrument: str
    lam_min: np.ndarray       # (n_lam,)
    lam_max: np.ndarray       # (n_lam,)
    flam: np.ndarray          # (n_lam,) observed dF/dlam with noise
    flamerr: np.ndarray       # (n_lam,) error on dF/dlam
    sim_flam: np.ndarray      # (n_lam,) true dF/dlam (no noise)
    snr_compute: float        # aggregate SNR over wavelength range
    lammin_snr: float
    lammax_snr: float
    trest: float
    tobs: float
    sim_genmag: np.ndarray    # (n_lam,) true synthetic mag per bin
    synmag: Optional[Dict[str, float]] = None  # SED-true synthetic mag per band
    n_lam_total: int = 0                       # bins before the validity cut
    is_host: bool = False                      # host spectrum, written at MJD=-9
    scale_host_contam: float = 0.0             # host flux scale added to a SN spectrum
    warp: Optional[np.ndarray] = None          # (n_lam,) WARP_SPECTRUM factor per bin
    # true FLAM on every observed bin, for SIM_SYNMAG (snlc_sim.c:11281)
    sim_flam_full: Optional[np.ndarray] = None


def keep_spectrum(field, field_require, wgt, prescale_dict, r_key, r_field,
                  is_host=False):
    """Whether a TAKE_SPECTRUM entry survives the field/prescale gates.

    Mirrors DO_GENSPEC (snlc_sim.c:10644) for the single-field case jax models:

    - no field at all -> every gate is bypassed (SNANA's NFIELD_OVP==0, :10690)
    - per-key prescale: reject when r_key > wgt (:10698)
    - FIELD_REQUIRE: exact match required when set (:10725)
    - dict prescale: accept when r_field < 1/preScale, absent field -> 1.0.
      Host spectra are exempt (:10731).

    FIELD_REQUIRE is an exact strcmp, but the dict lookup is a partial string
    match (OPT_DICT=1), so a dict key 'C' matches field 'C3' while a
    FIELD_REQUIRE of 'C' does not.

    `field`, `r_key` and `r_field` may be arrays (one entry vs many SNe);
    returns a bool array in that case, else a scalar bool.
    """
    field_arr = np.asarray(field, dtype=object)
    scalar = field_arr.ndim == 0
    field_arr = np.atleast_1d(field_arr)
    r_key = np.atleast_1d(np.asarray(r_key, dtype=np.float64))
    r_field = np.atleast_1d(np.asarray(r_field, dtype=np.float64))

    bypass = np.array([(not f) or f == 'FIELD_NONAME' for f in field_arr])
    keep = np.ones(len(field_arr), dtype=bool)
    if wgt < 1.0:
        keep &= r_key <= wgt
    if field_require:
        keep &= np.array([f == field_require for f in field_arr])
    if prescale_dict and not is_host:
        prescale = np.ones(len(field_arr))
        matched = np.zeros(len(field_arr), dtype=bool)
        for name, value in prescale_dict.items():   # first match wins
            hit = np.array([name in f for f in field_arr]) & ~matched
            prescale = np.where(hit, value, prescale)
            matched |= hit
        keep &= r_field < 1.0 / prescale
    keep |= bypass
    return bool(keep[0]) if scalar else keep


def pad_zpoly_rows(coeff_rows: list) -> np.ndarray:
    """Ragged per-row coefficient lists -> one (n_row, n_coeff) matrix."""
    n = max((len(c) for c in coeff_rows), default=1) or 1
    coeffs = np.zeros((len(coeff_rows), n))
    for i, c in enumerate(coeff_rows):
        coeffs[i, :len(c)] = c
    return coeffs


@jit
def build_smearing_matrix(
    lam_min: jnp.ndarray,
    lam_max: jnp.ndarray,
    lam_avg: jnp.ndarray,
    lam_bin: jnp.ndarray,
    lam_sigma: jnp.ndarray,
    islam_extend: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Precompute the (n_lam, n_lam) Gaussian smearing matrix.

    Element [i, j] = fraction of flux from source bin j landing in target bin i,
    following GENSPEC_LAMSMEAR (snlc_sim.c:12240). The columns are deliberately
    not normalised: SNANA keeps the raw Gaussian integral, so flux beyond the
    +/-NBIN2 kernel, or past either end of the grid, is lost. Extended edge bins
    donate flux but never receive it, so their rows are empty.
    """
    n = lam_avg.shape[0]
    sig_safe = jnp.maximum(lam_sigma, 1e-30)
    inv_sig_sqrt2 = 1.0 / (sig_safe * jnp.sqrt(2.0))

    arg_lo = (lam_min[:, None] - lam_avg[None, :]) * inv_sig_sqrt2[None, :]
    arg_hi = (lam_max[:, None] - lam_avg[None, :]) * inv_sig_sqrt2[None, :]

    matrix = 0.5 * (erf(arg_hi) - erf(arg_lo))
    matrix = jnp.maximum(matrix, 0.0)

    # NBIN2 = (int)(NSIGLAM*LAMSIGMA/LAMBIN + 0.5), with the final bin's width
    # taken from its neighbour in case it is short.
    bin_ref = lam_bin.at[-1].set(lam_bin[-2]) if n > 1 else lam_bin
    nbin2 = jnp.floor(NLAMSIGMA * sig_safe / bin_ref + 0.5)
    dist = jnp.abs(jnp.arange(n)[:, None] - jnp.arange(n)[None, :])
    matrix = jnp.where(dist <= nbin2[None, :], matrix, 0.0)

    # LAMSIGMA = 0 drops all the flux in the source bin (GINT = 1 on the diagonal)
    identity = (dist == 0).astype(matrix.dtype)
    matrix = jnp.where((lam_sigma > 0.0)[None, :], matrix, identity)

    if islam_extend is not None:
        matrix = jnp.where(islam_extend[:, None], 0.0, matrix)
    return matrix


@jit
def apply_wavelength_smearing(flux: jnp.ndarray, smearing_matrix: jnp.ndarray) -> jnp.ndarray:
    """Apply wavelength smearing via matrix multiply. JIT-compiled."""
    return smearing_matrix @ flux


def _add_spectrum_noise(smeared_sf, flamerr, ef_t, rangauss_t, is_nearpeak, key,
                        only_tnoise=False, no_tnoise=False):
    """Search + template noise on a smeared spectrum (optmask bit 32/16 = only/no template).

    Mirrors GENSPEC_OBSFLUX_RANSMEAR (snlc_sim.c:12342): the correlated
    template realisation is drawn at the near-peak epoch and reused elsewhere.
    """
    # flamerr is F_lambda (~1e-19); its square underflows float32, so the
    # search error is formed from the O(1) template fraction instead
    fluxerr_t = ef_t * flamerr
    fluxerr_s = flamerr * jnp.sqrt(jnp.maximum(1.0 - ef_t ** 2, 0.0))
    has_template = ef_t > 1e-7
    key_t, key_s = jax.random.split(key)
    new_randoms = jax.random.normal(key_t, shape=flamerr.shape, dtype=flamerr.dtype)
    updated_rt = jnp.where(is_nearpeak & has_template, new_randoms, rangauss_t)
    gauran_t = jnp.where(has_template, updated_rt, 0.0)
    gauran_s = jax.random.normal(key_s, shape=flamerr.shape, dtype=flamerr.dtype)
    if only_tnoise:
        gauran_s = jnp.zeros_like(gauran_s)
    if no_tnoise:
        gauran_t = jnp.zeros_like(gauran_t)
    noise = fluxerr_s * gauran_s + fluxerr_t * gauran_t
    return smeared_sf + jnp.where(flamerr > 0, noise, 0.0), updated_rt


@jit
def compute_aggregate_snr(
    snr_per_bin: jnp.ndarray,
    lam_avg: jnp.ndarray,
    flux: jnp.ndarray,
    lam_lo: float,
    lam_hi: float,
) -> float:
    """
    Compute flux-weighted aggregate SNR over a wavelength range. JIT-compiled.

    Matches SNANA: SNR = sum(flux) / sqrt(sum((flux/snr)^2)), formed on the
    flux fractions w = flux/sum(flux) as 1 / sqrt(sum((w/snr)^2)): an
    F_lambda flux (~1e-19) squared underflows float32.
    """
    mask = (lam_avg >= lam_lo) & (lam_avg <= lam_hi) & (snr_per_bin > 0) & (flux > 0)
    masked_flux = jnp.where(mask, flux, 0.0)
    sum_flux = jnp.sum(masked_flux)
    w = masked_flux / jnp.where(sum_flux > 0, sum_flux, 1.0)
    werr = jnp.where(mask, w / snr_per_bin, 0.0)

    sum_werrsq = jnp.sum(werr ** 2)
    return jnp.where(sum_werrsq > 0, 1.0 / jnp.sqrt(sum_werrsq), 0.0)


@jit
def _sed_to_bin_flux(
    sed: jnp.ndarray,
    model_wavelengths: jnp.ndarray,
    spec_lam_min_rest: jnp.ndarray,
    spec_lam_max_rest: jnp.ndarray,
) -> jnp.ndarray:
    """Integrate a rest-frame SED into spectrograph bins via cumulative
    trapezoid. JIT-compiled, fully vectorised.

    Matches SNANA's sub-bin integration: computes the integral of the SED
    over each bin's wavelength range, divided by bin width to get mean flux.
    Uses cumulative trapezoid on the model grid, then interpolates at bin
    edges: no loops, works for any bin width vs grid spacing.

    ``sed``/``model_wavelengths`` are (n_model_wave,), the rest-frame SED
    flux and model wavelength grid. ``spec_lam_min_rest``/
    ``spec_lam_max_rest`` are (n_lam,), the spectrograph bin lower/upper
    edges in the rest frame. Returns the (n_lam,) average flux per bin
    (rest-frame units).
    """
    dlam = jnp.diff(model_wavelengths)
    trapz_values = 0.5 * (sed[:-1] + sed[1:]) * dlam
    cum = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(trapz_values)])

    cum_at_lo = jnp.interp(spec_lam_min_rest, model_wavelengths, cum, left=0.0, right=cum[-1])
    cum_at_hi = jnp.interp(spec_lam_max_rest, model_wavelengths, cum, left=0.0, right=cum[-1])
    bin_integral = cum_at_hi - cum_at_lo

    bin_width = spec_lam_max_rest - spec_lam_min_rest
    avg_flux = jnp.where(bin_width > 0, bin_integral / bin_width, 0.0)
    return avg_flux


# SNANA iterates the TEXPOSE solve at most MAXITER=40 times and aborts on
# non-convergence (GENSPEC_TEXPOSE_TAKE_SPECTRUM, snlc_sim.c:11989-12001).
_SCAN_NITER = 40


def solve_texpose_batch(
    calib: SpectrographCalibration,
    genmag_all: jnp.ndarray,
    snr_requests: jnp.ndarray,
    lam_lo_all: jnp.ndarray,
    lam_hi_all: jnp.ndarray,
    texpose_template,
    is_nearpeak: bool,
    sim_flam_all: Optional[jnp.ndarray] = None,
    template_scale: float = 1.0,
    textrap: bool = False,
    scale_snr_lam: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Solve TEXPOSE for all SNe simultaneously using jax.lax.scan + vmap.

    Replicates the iteration of GENSPEC_TEXPOSE_TAKE_SPECTRUM
    (snlc_sim.c:11767-12001) fully vectorised: no Python loops over SNe or
    iterations. The solve runs under SNANA's noNOISE flag, which zeroes
    LAMSIGMA (snlc_sim.c:11835, 12277), so the aggregate-SNR weights are the
    UNSMEARED per-bin fluxes the caller passes in `sim_flam_all`.

    ``genmag_all``/``sim_flam_all`` are (N, n_lam), the per-SN genmag arrays
    and optional physical flux for aggregate weighting. ``snr_requests`` is
    (N,), the target SNR per SN; ``lam_lo_all``/``lam_hi_all`` are (N,), the
    observer-frame wavelength range per SN. ``texpose_template`` is the
    base template exposure time used for the initial guess; ``is_nearpeak``
    selects a near-peak spectrum config. ``scale_snr_lam`` is (n_lam,), the
    SPECTROGRAPH_SCALE_SNR / OPTMASK-bit-8 SNR scale: SNANA applies this
    inside GENSPEC_SMEAR (snlc_sim.c:12184), which is what its own solver
    calls, so the search must see the same scaled SNR the spectrum will
    report.

    Returns ``(texpose, snr_compute)``: (N,) solved TEXPOSE values and the
    aggregate SNR SNANA stores as SNR_COMPUTE for each solve
    (snlc_sim.c:11870-11969).

    Raises `SNANAError` when a SN that needs the iterative solve has not
    converged after 40 iterations, matching SNANA's abort
    (snlc_sim.c:11989-12001).
    """
    use_simflam = sim_flam_all is not None
    sim_flam_arg = sim_flam_all if use_simflam else jnp.zeros_like(genmag_all)

    texpose, snr_compute, converged, iterated = _solve_texpose_batch(
        calib.log_texpose_list, calib.texpose_list,
        calib.zp, calib.sqsigsky, calib.lam_avg, calib.lam_bin,
        genmag_all, snr_requests, lam_lo_all, lam_hi_all,
        sim_flam_arg, jnp.asarray(texpose_template) * jnp.ones_like(snr_requests),
        jnp.asarray(float(template_scale)), is_nearpeak, use_simflam,
        textrap,
        jnp.ones_like(calib.lam_avg) if scale_snr_lam is None
        else jnp.asarray(scale_snr_lam),
    )
    # unconverged at a table bound: SNANA's goto branch; inside the table: abort (snlc_sim.c:11989)
    t_min = float(np.asarray(calib.texpose_list)[0])
    t_max = float(np.asarray(calib.texpose_list)[-1])
    texpose_np = np.asarray(texpose)
    bad = (np.asarray(iterated & ~converged)
           & (texpose_np > t_min) & (texpose_np < t_max))
    if bad.any():
        i = int(np.argmax(bad))
        raise SNANAError(
            f"TEXPOSE solve did not converge after {_SCAN_NITER} iterations for "
            f"{int(bad.sum())} spectra (first failing "
            f"SNR_REQUEST={float(np.asarray(snr_requests)[i]):.2f}); SNANA aborts "
            f"here too",
            function_name="solve_texpose_batch")
    return texpose, snr_compute


@partial(jax.jit, static_argnums=(13, 14, 15))
def _solve_texpose_batch(
    log_texpose_list, texpose_list, zp, sqsigsky, lam_avg, lam_bin,
    genmag_all, snr_requests, lam_lo_all, lam_hi_all,
    sim_flam_arg, texpose_template_val, template_scale, is_nearpeak,
    use_simflam, textrap, scale_snr_lam,
):
    """JIT-compiled TEXPOSE solver kernel. Cached per (is_nearpeak, use_simflam, textrap)."""
    t_min = texpose_list[0]
    t_max = texpose_list[-1]

    def _agg_snr(genmag, texpose_s, texpose_t, lam_lo, lam_hi, sim_flam):
        snr_bins, _, _, flux_pe = get_snr(
            log_texpose_list, texpose_list, zp, sqsigsky,
            texpose_s, texpose_t, genmag,
        )
        snr_bins = snr_bins * scale_snr_lam
        flux_weight = jnp.where(use_simflam, sim_flam * lam_bin, flux_pe)
        return compute_aggregate_snr(snr_bins, lam_avg, flux_weight, lam_lo, lam_hi)

    def _bounds_snr(genmag, texpose, lam_lo, lam_hi, sim_flam):
        return _agg_snr(genmag, texpose, 0.0, lam_lo, lam_hi, sim_flam)

    snr_at_min = vmap(_bounds_snr, in_axes=(0, None, 0, 0, 0))(
        genmag_all, t_min, lam_lo_all, lam_hi_all, sim_flam_arg)
    snr_at_max = vmap(_bounds_snr, in_axes=(0, None, 0, 0, 0))(
        genmag_all, t_max, lam_lo_all, lam_hi_all, sim_flam_arg)

    def _step(state, _, genmag, snr_request, lam_lo, lam_hi, sim_flam,
              snr_at_min_val, snr_at_max_val, template0):
        texpose, snr, flag, sgn, template, converged = state

        # initial guess, the whole near-peak guess doubled (snlc_sim.c:11925-11938)
        arg = (snr_request - snr_at_min_val) / jnp.maximum(snr_at_max_val - snr_at_min_val, 1e-30)
        non_nearpeak_guess = jnp.where(
            template0 > 0,
            template0 * 1.2,
            t_min * 2.0
        )
        texpose_p0 = jnp.where(
            is_nearpeak & (snr_at_min_val > 0) & (snr_at_max_val > snr_at_min_val),
            (t_min + (t_max - t_min) * (arg * arg)) * 2.0,
            non_nearpeak_guess
        )

        # scale by the SNR ratio (snlc_sim.c:11944)
        texpose_p1 = texpose * (snr_request / snr)

        # 2% steps, sign frozen at the first flip (snlc_sim.c:11949, 11980-11985)
        texpose_p2 = texpose * (1.0 - 0.02 * sgn)

        new_texpose = jnp.where(flag == 0, texpose_p0,
                                jnp.where(flag == 1, texpose_p1, texpose_p2))
        new_texpose = jnp.clip(new_texpose, t_min, t_max)

        # near-peak template exposure follows, clipped (snlc_sim.c:11954-11961)
        new_template = jnp.where(is_nearpeak & (template_scale > 0.0),
                                 jnp.clip(new_texpose * template_scale, t_min, t_max),
                                 template)

        snr_new = _agg_snr(genmag, new_texpose, new_template, lam_lo, lam_hi, sim_flam)
        tol = snr_new / snr_request - 1.0
        tol_last = snr / snr_request - 1.0
        newly_converged = jnp.abs(tol) < TOL_CONVERGE

        # a sign flip while scaling switches to 2% steps (:11980-11985)
        flip = (flag == 1) & (tol * tol_last < 0.0)
        new_sgn = jnp.where(flip, jnp.where(tol > 0, 1.0, -1.0), sgn)
        new_flag = jnp.where(flag == 0, 1, jnp.where(flip, 2, flag))

        f_texpose = jnp.where(converged, texpose, new_texpose)
        f_snr = jnp.where(converged, snr, snr_new)
        f_flag = jnp.where(converged, flag, new_flag)
        f_sgn = jnp.where(converged, sgn, new_sgn)
        f_template = jnp.where(converged, template, new_template)
        f_converged = converged | newly_converged

        return (f_texpose, f_snr, f_flag, f_sgn, f_template, f_converged), None

    def _solve_one(genmag, snr_request, lam_lo, lam_hi, sim_flam,
                   snr_min_val, snr_max_val, template0):
        step_fn = partial(_step, genmag=genmag, snr_request=snr_request,
                          lam_lo=lam_lo, lam_hi=lam_hi, sim_flam=sim_flam,
                          snr_at_min_val=snr_min_val, snr_at_max_val=snr_max_val,
                          template0=template0)
        init = (0.0, 0.0, 0, 0.0, template0, False)
        final, _ = jax.lax.scan(step_fn, init, None, length=_SCAN_NITER)
        return final[0], final[1], final[5]

    texpose_solved, snr_scan, converged = vmap(_solve_one)(
        genmag_all, snr_requests, lam_lo_all, lam_hi_all, sim_flam_arg,
        snr_at_min, snr_at_max, texpose_template_val
    )

    # Only lanes strictly between the bound SNRs enter SNANA's loop
    # (snlc_sim.c:11867-11895); the rest take the goto branches below.
    iterated = (snr_requests > snr_at_min) & (snr_requests < snr_at_max)

    # bounds in reverse priority (later wins); TEXTRAP extrapolates (snlc_sim.c:11862-11890)
    if textrap:
        at_max = t_max * (snr_requests / jnp.maximum(snr_at_max, 1e-30)) ** 2
        at_min = t_min * (snr_requests / jnp.maximum(snr_at_min, 1e-30)) ** 2
        snr_at_max_val = snr_at_min_val = snr_requests
    else:
        at_max = jnp.full_like(snr_requests, t_max)
        at_min = jnp.full_like(snr_requests, t_min)
        snr_at_max_val = snr_at_max
        snr_at_min_val = snr_at_min
    snr_compute = snr_scan
    texpose_solved = jnp.where(snr_at_max <= snr_requests, at_max, texpose_solved)
    snr_compute = jnp.where(snr_at_max <= snr_requests, snr_at_max_val, snr_compute)
    texpose_solved = jnp.where(snr_at_min >= snr_requests, at_min, texpose_solved)
    snr_compute = jnp.where(snr_at_min >= snr_requests, snr_at_min_val, snr_compute)
    texpose_solved = jnp.where(snr_at_max <= 0, t_max, texpose_solved)
    snr_compute = jnp.where(snr_at_max <= 0, snr_at_max, snr_compute)
    texpose_solved = jnp.where(snr_requests <= 0, t_min, texpose_solved)
    snr_compute = jnp.where(snr_requests <= 0, snr_at_min, snr_compute)

    return texpose_solved, snr_compute, converged, iterated


def snrsum_rest_v_sq(gen_flam, flamerr, z1_cmb, calib):
    """Per-SN sum of SNR^2 over the rest-frame V window, for one batch of
    spectra. Accumulate across spectra, then take the square root.

    SNANA computes this from the TRUE noise-free GENFLAM over the reported
    FLAMERR (GENSPEC_FLAM, snlc_sim.c:12537-12558), keeping bins whose CENTRE
    lies in [5000, 6000]*(1+z_cmb) with FLAMERR > 0 (GENSPEC_SNR,
    snlc_sim.c:11453-11496).
    """
    lam_avg = jnp.asarray(calib.lam_avg)[None, :]
    window = ((lam_avg >= 5000.0 * z1_cmb[:, None])
              & (lam_avg <= 6000.0 * z1_cmb[:, None])
              & (flamerr > 0))
    snr = jnp.where(window, gen_flam / jnp.where(flamerr > 0, flamerr, 1.0), 0.0)
    return jnp.sum(snr ** 2, axis=1)


def mw_transmission(lam_obs, mwebv, static_data):
    """Galactic flux fraction at the given observer-frame wavelengths, or None
    when no MW law is configured. ``mwebv`` must broadcast against the trailing
    wavelength axis (shape (n_sne, 1)); ``lam_obs`` is (n_sne, n_wave) or
    (1, n_wave) for shared bins. The law sees each event's AV, including
    curves whose shape depends on it (GALextinct_Somm25)."""
    law = static_data.get('mw_dust_law')
    if law is None:
        return None
    # Spectra are assembled outside JIT, so prepare the law at the actual
    # observer wavelengths. Interpolating its rest-grid preparation clamps
    # valid redshifted spectra at the model's wavelength boundary.
    # Copy nested laws too (e.g. Gordon16), preserving their configuration
    # without changing the preparation shared with the photometry kernels.
    law = deepcopy(law)
    wave = np.asarray(lam_obs)
    law.prepare_wavelength_grid(wave.ravel())
    rv_mw = static_data['rv_mw']
    av = rv_mw * jnp.asarray(mwebv)
    if wave.shape[0] > 1:
        av = jnp.broadcast_to(av, wave.shape).ravel()
    transmission = law.apply_extinction(
        1.0, jnp.asarray(wave.ravel()), av, rv_mw)
    return transmission.reshape(np.broadcast_shapes(wave.shape, mwebv.shape))


def _apply_mw_extinction(seds, z1, mwebvs, model_wavelengths, static_data):
    """Observer-frame Milky Way extinction applied to each rest-frame SED."""
    trans = mw_transmission(model_wavelengths[None, :] * z1[:, None],
                            mwebvs[:, None], static_data)
    if trans is None:
        return seds
    return seds * trans


def _seds_to_simflam_genmag(seds, z1, model_wavelengths, calib, lamspike_idx, mag_offset_scale):
    """SED → observer-frame SIM_FLAM per bin + spectrograph genmag (bit 4 = LAMSPIKE)."""
    seds = seds * mag_offset_scale
    lam_min_rest = calib.lam_min[None, :] / z1[:, None]
    lam_max_rest = calib.lam_max[None, :] / z1[:, None]
    bin_flux = vmap(_sed_to_bin_flux, in_axes=(0, None, 0, 0))(
        seds, model_wavelengths, lam_min_rest, lam_max_rest)
    sim_flam = bin_flux / z1[:, None]
    return _simflam_to_genmag(sim_flam, calib, lamspike_idx)


def _simflam_to_genmag(sim_flam, calib, lamspike_idx, mag_flux=None):
    """Per-bin SIM_FLAM -> spectrograph genmag (bit 4 = LAMSPIKE).

    mag_flux overrides the flux the magnitude is built from, for host spectra
    whose magnitude is not the flux that is stored.
    """
    flux = sim_flam if mag_flux is None else mag_flux
    ftmp = (calib.lam_avg[None, :] / HC_ERG_A) * flux * calib.lam_bin[None, :]
    genmag = jnp.where(ftmp > 0, -2.5 * jnp.log10(ftmp) + calib.zp_sedmodel[None, :], 99.0)
    if lamspike_idx >= 0:
        spike = jnp.arange(calib.n_lam) == lamspike_idx
        genmag = jnp.where(spike[None, :], genmag, 40.0)
        sim_flam = jnp.where(spike[None, :], sim_flam, 0.0)
    return sim_flam, genmag


def observed_bins(calib):
    """Bins that reach the output; the extended edge bins never do."""
    if calib.islam_extend is None:
        return np.ones(calib.n_lam, dtype=bool)
    return ~np.asarray(calib.islam_extend)


def _smear_simflam(sim_flam, calib, smearing_matrix):
    """Wavelength-smear SIM_FLAM, preserving zero-flux bins."""
    smeared = vmap(lambda sf: apply_wavelength_smearing(
        sf * calib.lam_bin, smearing_matrix) / calib.lam_bin)(sim_flam)
    return jnp.where(sim_flam > 0, smeared, 0.0)


def nearpeak_mask(sn_index, trest, is_host):
    """Mark each SN's near-peak epoch among a flat list of (SN, epoch) rows.

    Mirrors GENSPEC_MJD_ORDER (snlc_sim.c:10921-10932): SNANA scans the event's
    own TREST_LIST for the smallest ``|TREST|``, so the choice is per event and uses
    where the epoch actually landed, not the config's nominal epoch centre,
    which would also compare OBS-frame values against REST-frame ones. HOST
    epochs are never near-peak.

    The flat (SN, epoch) form serves both schedules: a global one contributes
    n_config rows per SN, a per-LIBID one a ragged number. Ties keep the first
    row, matching SNANA's strictly-decreasing scan.
    """
    sn_index = np.asarray(sn_index)
    key = np.where(np.asarray(is_host, dtype=bool), np.inf,
                   np.abs(np.asarray(trest, dtype=float)))
    order = np.lexsort((key, sn_index))          # stable: ties keep input order
    sn_sorted = sn_index[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = sn_sorted[1:] != sn_sorted[:-1]
    out = np.zeros(len(key), dtype=bool)
    out[order[first]] = True
    return out


def _sed_with_host_dust(model_params, av, rv, phase, static_data):
    """One rest-frame SED, reddened by host dust, on the whole wavelength
    grid. The spectra path wants the SED itself rather than a band integral,
    so it does not go through SNModel.evaluate.
    """
    sed = static_data['model_instance'].generate_sed(
        phase=phase, params=model_params, static_data=static_data)
    return static_data['host_dust_law'].apply_extinction(
        sed, static_data['wavelength_grid'], av, rv)


def _seds_to_smeared(model_params, av, rv, mwebv,
                     phases, z1, model_wavelengths, calib, opt, static_data,
                     sed_factor=None):
    """SED -> MW extinction -> SIM_FLAM/genmag -> lambda smear.

    Shared by every spectra driver. GENMODEL_MSKOPT_SEDMODEL_ZERO_NEGFLAM belongs
    here because SNANA clamps inside the SED integrator (genmag_SALT2.c:2908 and
    genmag_SEDtools.c:1337), so it applies to all spectra regardless of driver.
    """
    seds = vmap(_sed_with_host_dust, in_axes=(0, 0, 0, 0, None))(
        model_params, av, rv, phases, static_data)
    if sed_factor is not None:
        seds = seds * sed_factor
    seds = _apply_mw_extinction(seds, z1, mwebv, model_wavelengths, static_data)
    if opt['zero_neg_flam']:
        seds = jnp.maximum(seds, 0.0)
    sim_flam, genmag = _seds_to_simflam_genmag(
        seds, z1, model_wavelengths, calib, opt['lamspike_idx'],
        static_data['fluxcal_scale_mag_offset'])
    smeared = _smear_simflam(sim_flam, calib, opt['smearing_matrix'])
    return sim_flam, genmag, smeared


def _host_bins_to_smeared(host_spec, gal_indices, redshifts, mwebv, z1,
                          calib, opt, static_data):
    """Host SED -> spectrograph bins -> MW extinction -> lambda smear.

    Returns the same (sim_flam, genmag, smeared) triple as _seds_to_smeared.

    The magnitude is built from the *un-extincted* flux, as SNANA does
    (genSpec_HOSTLIB): the host mag is the galaxy's true brightness, while the
    stored spectrum is what the instrument sees. jax omits the extra 1/(1+z)
    SNANA applies here, which disagrees with its own SIM_SYNMAG.
    """
    table, coeffs = host_spec
    lam_min, lam_max = np.asarray(calib.lam_min), np.asarray(calib.lam_max)
    seds = jnp.asarray(table.event_seds(np.asarray(coeffs)[np.asarray(gal_indices)],
                                        redshifts))
    # SNANA's (1+z) cancels against the observer bin width
    binned = vmap(_sed_to_bin_flux, in_axes=(0, None, 0, 0))(
        seds, jnp.asarray(table.wave),
        jnp.asarray(lam_min)[None, :] / z1[:, None],
        jnp.asarray(lam_max)[None, :] / z1[:, None])
    on_grid = table.on_grid(np.asarray(redshifts)[:, None],
                            lam_min[None, :], lam_max[None, :])
    sim_flam_true = jnp.where(jnp.asarray(on_grid), binned, 0.0)

    mwxt = mw_transmission(jnp.asarray(calib.lam_avg)[None, :], mwebv[:, None],
                           static_data)
    sim_flam = sim_flam_true if mwxt is None else sim_flam_true * mwxt
    sim_flam, genmag = _simflam_to_genmag(sim_flam, calib, opt['lamspike_idx'],
                                          mag_flux=sim_flam_true)
    smeared = _smear_simflam(sim_flam, calib, opt['smearing_matrix'])
    return sim_flam, genmag, smeared


# unset below this (snlc_sim.c:11669)
HOST_CONTAM_MIN = 1.0e-8


def host_contam_scale(hostfrac, hostsnfrac, host_sim_flam, peak_sim_flam, calib):
    """Per-SN host-contamination scale, or None when neither key is set.

    HOSTFRAC scales the host directly; HOSTSNFRAC instead solves for the scale
    that makes the integrated host flux that fraction of the SN's peak flux,
    over the bins where both are non-empty (GENSPEC_HOST_CONTAMINATION).
    """
    if max(hostfrac, hostsnfrac) <= HOST_CONTAM_MIN:
        return None
    if hostfrac > HOST_CONTAM_MIN:
        return jnp.full(host_sim_flam.shape[0], hostfrac)
    lam_bin = jnp.asarray(calib.lam_bin)[None, :]
    both = (host_sim_flam != 0.0) & (peak_sim_flam != 0.0)
    fsum = lambda f: jnp.sum(jnp.where(both, f * lam_bin * lam_bin, 0.0), axis=1)
    fsum_host = fsum(host_sim_flam)
    return jnp.where(fsum_host > 1e-40,
                     hostsnfrac * fsum(peak_sim_flam)
                     / jnp.maximum(fsum_host, 1e-40), 1.0)


def add_host_contamination(sim_flam, genmag, host_sim_flam, scale, calib):
    """Add `scale` x the host spectrum to each SN spectrum (snlc_sim.c:11748).

    SNANA works in per-bin flux and floors the SN at 1e-30 so the magnitude
    shift stays finite in empty bins, keeping the floored value in the sum.
    """
    lam_bin = jnp.asarray(calib.lam_bin)[None, :]
    flux_sn = jnp.maximum(sim_flam * lam_bin, 1e-30)
    flux_tot = flux_sn + host_sim_flam * lam_bin * scale[:, None]
    return flux_tot / lam_bin, genmag - 2.5 * jnp.log10(flux_tot / flux_sn)


def _snr_and_noise(calib, opt, smeared, genmag, texposes, texpose_template_arr,
                   is_nearpeak, rangauss, keys, window=None):
    """Per-bin SNR + search/template noise for a batch of spectra.

    `window` (bool, same shape as `smeared`) restricts the noised bins to
    those whose centre lies inside the per-spectrum SNR window: GENSPEC_SMEAR
    skips out-of-window source bins entirely (snlc_sim.c:12113-12117), so they
    keep FLAMERR = 0 and are never written.
    """
    t_min, t_max = float(calib.texpose_list[0]), float(calib.texpose_list[-1])

    def snr_one(texpose_s, texpose_t, genmag_j):
        snr, ef_t, _, flux = get_snr(calib.log_texpose_list, calib.texpose_list,
                                     calib.zp, calib.sqsigsky, texpose_s, texpose_t, genmag_j)
        if opt['textrap']:
            snr = snr * jnp.sqrt(texpose_s / jnp.clip(texpose_s, t_min, t_max))
        if opt['scale_snr_lam'] is not None:
            snr = snr * opt['scale_snr_lam']
        return snr, ef_t
    snr, errfrac = vmap(snr_one)(texposes, texpose_template_arr, genmag)
    valid_t = texposes[:, None] > 0
    snr = jnp.where(valid_t, snr, 0.0)
    errfrac = jnp.where(valid_t, errfrac, 0.0)
    if window is not None:
        snr = jnp.where(window, snr, 0.0)
    flamerr = jnp.where(snr > 0, jnp.abs(smeared) / snr, 0.0)
    flam, updated_rangauss = vmap(lambda sf, fe, ef, rt, key: _add_spectrum_noise(
        sf, fe, ef, rt, is_nearpeak, key, opt['only_tnoise'], opt['no_tnoise']))(
        smeared, flamerr, errfrac, rangauss, keys)
    return flam, flamerr, snr, updated_rangauss


def _parse_spectro_options(spectro_options, calib):
    """Decode shared SPECTROGRAPH_OPTMASK/SCALE keys (init_genSpec, snlc_sim.c:10246)."""
    opt = spectro_options
    optmask = opt.get('optmask', 0)
    lam_sigma = calib.lam_sigma * 0.0 if (optmask & 1) else calib.lam_sigma
    if optmask & 2:  # bit 2 overrides bit 1
        lam_sigma = calib.lam_sigma * 2.0
    scale_snr_text = '100.0' if (optmask & 8) else str(opt.get('scale_snr', ''))
    scale_snr_lam = None
    if scale_snr_text:
        coefs = [float(c) for c in scale_snr_text.split(',')]
        scale_snr_lam = jnp.asarray(
            eval_genpoly(coefs, np.asarray(calib.lam_avg)))
    # WARP_SPECTRUM scales the measured FLAM only (GENSPEC_FLAM, snlc_sim.c:12503)
    warp_coeffs = opt.get('warp_lampoly') or []
    flam_warp = None
    if warp_coeffs:
        flam_warp = np.asarray(
            eval_genpoly(list(warp_coeffs), np.asarray(calib.lam_avg)))
    return {
        'flam_warp': flam_warp,
        'smearing_matrix': build_smearing_matrix(
            calib.lam_min, calib.lam_max, calib.lam_avg, calib.lam_bin, lam_sigma,
            calib.islam_extend),
        'lamspike_idx': calib.n_lam // 2 if (optmask & 4) else -1,
        'textrap': bool(optmask & 64),
        'only_tnoise': bool(optmask & 32),
        'no_tnoise': bool(optmask & 16),
        'sed_true': bool(optmask & 128),
        'zero_neg_flam': bool(opt.get('zero_neg_flam', False)),
        'scale_snr_lam': scale_snr_lam,
    }


def _epoch_row_keys(rng_key, schedule_source, event_index, entry_index):
    """One PRNG key per spectral epoch from its stable identity.

    The identity is (global event index, schedule source, entry index within
    the event's raw schedule), so no draw depends on filtered row position,
    batch composition, or which other events pass. fold_in(key, 0) is the
    epoch-time stream and fold_in(key, 1) the noise stream for that epoch.
    """
    base = jax.random.fold_in(rng_key, schedule_source)

    def one(evt, ent):
        return jax.random.fold_in(jax.random.fold_in(base, evt), ent)
    return vmap(one)(jnp.asarray(event_index, dtype=jnp.uint32),
                     jnp.asarray(entry_index, dtype=jnp.uint32))


def _spectrum_kept(static_data, event, entry):
    """Standalone spectrum calls have no simulation prescale selection."""
    keep = static_data.get('spectrum_keep')
    return True if keep is None else bool(keep[event][entry])


def _event_context(sn_parameters, passing_mask, static_data):
    """Per-event arrays for the passing subset, shared by every adapter."""
    z_helio_all = np.asarray(
        sn_parameters['sim_redshift_helio'] if 'sim_redshift_helio' in sn_parameters
        else sn_parameters['sim_redshift_cmb'])
    n_sne = len(z_helio_all)
    pass_idx = np.where(np.asarray(passing_mask))[0]
    take = lambda key: np.asarray(sn_parameters[key])[pass_idx]

    z = z_helio_all[pass_idx]
    z_cmb = np.asarray(sn_parameters['sim_redshift_cmb'])[pass_idx]
    snids = np.asarray(sn_parameters['cid'])[pass_idx]
    global_idx = np.asarray(sn_parameters['sn_global_idx'])[pass_idx]
    declared = static_data['model_instance'].sed_param_names()
    model_params = {k: jnp.asarray(np.asarray(sn_parameters[k])[pass_idx])
                    for k in declared if k in sn_parameters}
    gal_idx = (np.asarray(sn_parameters['host_galaxy_index'])[pass_idx]
               if 'host_galaxy_index' in sn_parameters else None)
    sed_factor = None
    smear = static_data.get('gensmear')
    if smear is not None:
        sed_factor = scatter_factors(
            smear, jnp.asarray(take('_gensmear_randoms')),
            static_data['wavelength_grid'])
    if 'sn_flux_factor' in static_data:
        grey_factor = jnp.asarray(static_data['sn_flux_factor'])[pass_idx, None]
        sed_factor = grey_factor if sed_factor is None else sed_factor * grey_factor
    return {
        'n_sne': n_sne,
        'pass_idx': pass_idx,
        'n_pass': len(pass_idx),
        'z': z,
        'z1': jnp.asarray(1.0 + z),
        'z_cmb': z_cmb,
        # the generated peak: 'peakmjd' may be the OPT_SETPKMJD estimate
        'peakmjd': take('peakmjd_true'),
        'snid': snids,
        'global_idx': global_idx,
        'model_params': model_params,
        'gal_idx': gal_idx,
        'av': jnp.asarray(take('av')),
        'rv': jnp.asarray(take('rv')),
        'mwebv': jnp.asarray(take('sim_mwebv')),
        'sed_factor': sed_factor,
        'model_wavelengths': jnp.array(static_data['wavelength_grid']),
        'static_data': static_data,
    }


# column: (cast type, default); None marks the ragged columns padded into a matrix
_SPEC_ROW_FIELDS = {
    'sn_local':      (int,   0),
    'entry_index':   (int,   0),
    'phase':         (float, 0.0),
    'mjd':           (float, 0.0),
    'mode':          (int,   0),
    'texpose_fixed': (float, 0.0),
    'zpoly_texpose': (None,  ()),
    'zpoly_snr':     (None,  ()),
    'snr_lo':        (float, 0.0),
    'snr_hi':        (float, 0.0),
    'snr_rest':      (bool,  False),
    'is_host':       (bool,  False),
    'clip':          (bool,  False),
    'skip':          (bool,  False),
}


def _empty_rows():
    return {name: [] for name in _SPEC_ROW_FIELDS}


def _append_row(rows, **values):
    """Append one scheduled spectrum, defaulting the columns not named."""
    unknown = sorted(set(values) - set(_SPEC_ROW_FIELDS))
    if unknown:
        raise SNANAError(f"{unknown} are not spectrum row columns",
                         function_name="_append_row")
    for name, (_, default) in _SPEC_ROW_FIELDS.items():
        rows[name].append(values.get(name, default))


def _finalize_rows(rows):
    """Column lists -> plain arrays, with the ragged zpoly lists padded."""
    return {name: (pad_zpoly_rows(rows[name]) if dtype is None
                   else np.asarray(rows[name], dtype=dtype))
            for name, (dtype, _) in _SPEC_ROW_FIELDS.items()}


def _generate_spectra_batch(calib, opt, ctx, rows, rng_key, schedule_source,
                            template_scale=0.0, template_texpose=0.0,
                            use_warp=True, host_spec=None,
                            hostfrac=0.0, hostsnfrac=0.0):
    """Generate spectra for one flat batch of prepared epochs.

    Owns SED evaluation, exposure solving, noise, SNR aggregation and result
    assembly for every schedule source; the adapters only interpret their
    schedule and hand over plain arrays with a stable epoch identity.

    Rows marked `skip` still take part in the near-peak scan (SNANA's
    GENSPEC_MJD_ORDER reads TREST_LIST without checking SKIP,
    snlc_sim.c:10921-10932) but are never solved, noised or written.

    Returns (spectra_by_snid, snrsum_rest_v) with snrsum over the passing
    subset (length ctx['n_pass']).
    """
    n_pass = ctx['n_pass']
    n_row = len(rows['sn_local'])
    n_lam = calib.n_lam
    if n_row == 0 or n_pass == 0:
        return {}, np.zeros(n_pass)

    loc = rows['sn_local']
    phase = rows['phase']
    modes = rows['mode']
    is_host = rows['is_host']
    active = ~rows['skip']

    # skipped rows still join the near-peak scan (snlc_sim.c:10921-10932)
    is_np_row = nearpeak_mask(loc, phase, is_host)

    row_keys = _epoch_row_keys(rng_key, schedule_source,
                               ctx['global_idx'][loc], rows['entry_index'])
    noise_keys = vmap(lambda k: jax.random.fold_in(k, 1))(row_keys)

    sim_flam_rows = jnp.zeros((n_row, n_lam))
    genmag_rows = jnp.full((n_row, n_lam), 99.0)
    smeared_rows = jnp.zeros((n_row, n_lam))
    contam_row = np.zeros(n_row)

    host_flam_evt = None
    idx_host = np.where(active & is_host)[0]
    if len(idx_host):
        hloc = loc[idx_host]
        h_flam, h_genmag, h_smeared = _host_bins_to_smeared(
            host_spec, ctx['gal_idx'][hloc], ctx['z'][hloc],
            ctx['mwebv'][hloc], ctx['z1'][hloc],
            calib, opt, ctx['static_data'])
        sim_flam_rows = sim_flam_rows.at[idx_host].set(h_flam)
        genmag_rows = genmag_rows.at[idx_host].set(h_genmag)
        smeared_rows = smeared_rows.at[idx_host].set(h_smeared)
        # host first, one per event, for contamination (snlc_sim.c:10931)
        host_flam_evt = jnp.zeros((n_pass, n_lam)).at[jnp.asarray(hloc)].set(h_flam)

    idx_sn = np.where(active & ~is_host)[0]
    if len(idx_sn):
        sloc = loc[idx_sn]
        mp = {k: v[sloc] for k, v in ctx['model_params'].items()}
        sf, gm, sm = _seds_to_smeared(
            mp,
            ctx['av'][sloc], ctx['rv'][sloc], ctx['mwebv'][sloc],
            jnp.asarray(phase[idx_sn]), ctx['z1'][sloc],
            ctx['model_wavelengths'], calib, opt, ctx['static_data'],
            None if ctx['sed_factor'] is None else ctx['sed_factor'][sloc])
        if host_flam_evt is not None and max(hostfrac, hostsnfrac) > HOST_CONTAM_MIN:
            peak_sim_flam = None
            if hostsnfrac > HOST_CONTAM_MIN:
                # GENSPEC_TRUE(ISPEC_PEAK), snlc_sim.c:11123: the SN at TOBS=0.
                peak_sim_flam = _seds_to_smeared(
                    ctx['model_params'],
                    ctx['av'], ctx['rv'], ctx['mwebv'], jnp.zeros(n_pass),
                    ctx['z1'], ctx['model_wavelengths'], calib, opt,
                    ctx['static_data'], ctx['sed_factor'])[0]
            contam_scale = host_contam_scale(hostfrac, hostsnfrac,
                                             host_flam_evt, peak_sim_flam, calib)
            # contaminated before the smear (GENSPEC_DRIVER, snlc_sim.c:10614)
            sf, gm = add_host_contamination(
                sf, gm, host_flam_evt[sloc], contam_scale[jnp.asarray(sloc)], calib)
            sm = _smear_simflam(sf, calib, opt['smearing_matrix'])
            contam_row[idx_sn] = np.asarray(contam_scale)[sloc]
        sim_flam_rows = sim_flam_rows.at[idx_sn].set(sf)
        genmag_rows = genmag_rows.at[idx_sn].set(gm)
        smeared_rows = smeared_rows.at[idx_sn].set(sm)

    # SNR-window clip: only in-window flux is smeared (snlc_sim.c:10600-10606)
    lam_avg_np = np.asarray(calib.lam_avg)
    win_np = np.ones((n_row, n_lam), dtype=bool)
    idx_clip = np.where(active & rows['clip'])[0]
    if len(idx_clip):
        win_np[idx_clip] = ((lam_avg_np[None, :] >= rows['snr_lo'][idx_clip, None])
                            & (lam_avg_np[None, :] <= rows['snr_hi'][idx_clip, None]))
        smeared_rows = smeared_rows.at[idx_clip].set(_smear_simflam(
            sim_flam_rows[idx_clip] * jnp.asarray(win_np[idx_clip]),
            calib, opt['smearing_matrix']))

    texpose_row = np.zeros(n_row)
    flam_out = np.zeros((n_row, n_lam))
    flamerr_out = np.zeros((n_row, n_lam))
    snrc_row = np.full(n_row, SNR_COMPUTE_NULL)
    # LAMOBS_SNR: 0 without an entry, -9 in TEXPOSE mode, the window in SNR mode (:10537-10550)
    lamlo_row = np.zeros(n_row)
    lamhi_row = np.zeros(n_row)
    lamlo_row[modes == 1] = LAMOBS_SNR_NULL
    lamhi_row[modes == 1] = LAMOBS_SNR_NULL
    m2 = modes == 2
    if m2.any():
        z1c = 1.0 + ctx['z_cmb'][loc]   # GENSPEC_LAMOBS_RANGE uses z_cmb (:10533)
        lamlo_row[m2] = np.where(rows['snr_rest'], rows['snr_lo'] * z1c,
                                 rows['snr_lo'])[m2]
        lamhi_row[m2] = np.where(rows['snr_rest'], rows['snr_hi'] * z1c,
                                 rows['snr_hi'])[m2]

    snrsum_sq = np.zeros(n_pass)
    t_min = float(calib.texpose_list[0])
    t_max = float(calib.texpose_list[-1])

    if opt['sed_true']:
        # noise off: GENSPEC_OBSFLUX_RANSMEAR returns 0 under SEDMODEL
        # (snlc_sim.c:12377); FLAMERR stays 0 in the reduced SPEC format.
        texpose_row[active] = rows['texpose_fixed'][active]
        flam_out[active] = np.asarray(smeared_rows)[active]
    else:
        rangauss = jnp.zeros((n_pass, n_lam))
        texpose_template_arr = jnp.full(n_pass, float(template_texpose))
        # near-peak first: it fixes the template exposure (snlc_sim.c:11954-11958)
        for want_np in (True, False):
            for mode in (2, 1, 0):
                sel = active & (is_np_row == want_np) & (modes == mode)
                idx = np.where(sel)[0]
                if not len(idx):
                    continue
                gloc = loc[idx]
                z = ctx['z'][gloc]
                if mode == 0:
                    texpose = rows['texpose_fixed'][idx]
                elif mode == 1:
                    texpose = eval_genpoly_rows(rows['zpoly_texpose'][idx], z)
                else:
                    z1c = 1.0 + ctx['z_cmb'][gloc]
                    lam_lo = np.where(rows['snr_rest'][idx],
                                      rows['snr_lo'][idx] * z1c,
                                      rows['snr_lo'][idx])
                    lam_hi = np.where(rows['snr_rest'][idx],
                                      rows['snr_hi'][idx] * z1c,
                                      rows['snr_hi'][idx])
                    snr_req = jnp.asarray(eval_genpoly_rows(rows['zpoly_snr'][idx], z))
                    # unsmeared windowed flux: noNOISE zeroes LAMSIGMA (snlc_sim.c:11835, :12113)
                    solve_win = ((lam_avg_np[None, :] >= lam_lo[:, None])
                                 & (lam_avg_np[None, :] <= lam_hi[:, None]))
                    solve_flam = sim_flam_rows[idx] * jnp.asarray(solve_win)
                    texpose_j, snr_solved = solve_texpose_batch(
                        calib, genmag_rows[idx], snr_req,
                        jnp.asarray(lam_lo), jnp.asarray(lam_hi),
                        texpose_template_arr[gloc], want_np,
                        sim_flam_all=solve_flam,
                        template_scale=template_scale,
                        textrap=opt['textrap'],
                        scale_snr_lam=opt['scale_snr_lam'])
                    texpose = np.asarray(texpose_j)
                    snrc_row[idx] = np.asarray(snr_solved)
                    if want_np and template_scale > 0.0:
                        solved = jnp.clip(jnp.asarray(texpose) * template_scale,
                                          t_min, t_max)
                        texpose_template_arr = texpose_template_arr.at[
                            jnp.asarray(gloc)].set(solved)

                texpose = np.asarray(texpose, dtype=float)
                # out-of-range TEXPOSE aborts unless TEXTRAP (getSNR_spectrograph)
                used = texpose > 0
                if mode != 2 and not opt['textrap'] and np.any(
                        used & ((texpose < t_min) | (texpose > t_max))):
                    bad = texpose[used & ((texpose < t_min)
                                          | (texpose > t_max))][0]
                    raise SNANAError(
                        f"Invalid TEXPOSE_S = {bad:.2f}; valid range "
                        f"{t_min:.2f} to {t_max:.2f} (set SPECTROGRAPH_OPTMASK "
                        f"bit 64 to extrapolate)",
                        function_name="_generate_spectra_batch")

                flam, flamerr, _, rg = _snr_and_noise(
                    calib, opt, smeared_rows[idx], genmag_rows[idx],
                    jnp.asarray(texpose), texpose_template_arr[gloc],
                    want_np, rangauss[gloc], noise_keys[idx],
                    window=jnp.asarray(win_np[idx]))
                # the template-noise realisation belongs to this pass's SNe only
                rangauss = rangauss.at[jnp.asarray(gloc)].set(rg)

                texpose_row[idx] = texpose
                flam_out[idx] = np.asarray(flam)
                flamerr_out[idx] = np.asarray(flamerr)

                # SNRSUM_REST_V from the true GENFLAM (snlc_sim.c:12537-12558)
                contrib = np.asarray(snrsum_rest_v_sq(
                    sim_flam_rows[idx], flamerr,
                    jnp.asarray(1.0 + ctx['z_cmb'][gloc]), calib))
                np.add.at(snrsum_sq, gloc, np.where(texpose > 0, contrib, 0.0))

    warp_np = opt['flam_warp'] if use_warp else None
    if warp_np is not None:
        flam_out = flam_out * warp_np[None, :]
    obs = observed_bins(calib)
    sim_flam_np = np.asarray(sim_flam_rows)
    genmag_np = np.asarray(genmag_rows)
    lam_min_np, lam_max_np = np.asarray(calib.lam_min), np.asarray(calib.lam_max)
    z1_np = np.asarray(ctx['z1'])

    spectra = {}
    for i in range(n_row):
        if not active[i] or texpose_row[i] <= 0:
            continue
        if opt['sed_true']:
            valid = obs
        else:
            # bins with FLAMERR > 0; none skips the spectrum (snlc_sim.c:10633)
            valid = (flamerr_out[i] > 0) & obs
            if not valid.any():
                continue
        cid = int(ctx['snid'][loc[i]])
        spectra.setdefault(cid, []).append(SpectrumResult(
            snid=cid,
            mjd=-9.0 if is_host[i] else float(rows['mjd'][i]),
            texpose=float(texpose_row[i]),
            instrument=calib.instrument_name,
            n_lam_total=int(n_lam),
            lam_min=lam_min_np[valid],
            lam_max=lam_max_np[valid],
            flam=flam_out[i][valid],
            flamerr=flamerr_out[i][valid],
            sim_flam=sim_flam_np[i][valid],
            snr_compute=float(snrc_row[i]),
            lammin_snr=float(lamlo_row[i]),
            lammax_snr=float(lamhi_row[i]),
            trest=float(phase[i]),
            tobs=float(phase[i] * z1_np[loc[i]]),
            sim_genmag=genmag_np[i][valid],
            is_host=bool(is_host[i]),
            scale_host_contam=float(contam_row[i]),
            warp=warp_np[valid] if warp_np is not None else None,
            sim_flam_full=sim_flam_np[i][obs],
        ))
    return spectra, np.sqrt(snrsum_sq)


def generate_simlib_spectra(calib, sn_parameters, spec_mjds_list, spec_texposes_list,
                            static_data, template_texpose,
                            trest_range, rng_key, passing_mask, spectro_options):
    """Spectra at SIMLIB SPECTROGRAPH epochs (SIMLIB_addCadence_SPECTROGRAPH).

    Each passing SN's spectrograph epochs are kept when they fall inside its
    GENRANGE_TREST window (SNANA drops out-of-cadence spectrograph epochs).
    No warp: SIMLIB SPECTROGRAPH rows have no TAKE_SPECTRUM entry to carry one
    (GENSPEC_FLAM reads INPUTS.TAKE_SPECTRUM[imjd], ORDER=-9).
    """
    opt = _parse_spectro_options(spectro_options, calib)
    ctx = _event_context(sn_parameters, passing_mask, static_data)
    n_sne = ctx['n_sne']

    rows = _empty_rows()
    for local, gi in enumerate(ctx['pass_idx']):
        peak = ctx['peakmjd'][local]
        z1 = 1.0 + ctx['z'][local]
        for k, (m, t) in enumerate(zip(np.asarray(spec_mjds_list[gi]),
                                       np.asarray(spec_texposes_list[gi]))):
            ph = (float(m) - peak) / z1
            if not trest_range[0] <= ph <= trest_range[1]:
                continue
            # SCALE_TEXPOSE scales synthetic photometry, not the spectrum
            _append_row(rows, sn_local=local, entry_index=k, phase=ph,
                        mjd=float(m), texpose_fixed=float(t),
                        skip=not _spectrum_kept(static_data, gi, 0))
    if not rows['sn_local']:
        return {}, jnp.zeros(n_sne)

    spectra, snrsum_pass = _generate_spectra_batch(
        calib, opt, ctx, _finalize_rows(rows), rng_key,
        _SCHEDULE_SIMLIB_SPECTROGRAPH,
        template_texpose=template_texpose, use_warp=False)

    snrsum = np.zeros(n_sne)
    snrsum[ctx['pass_idx']] = snrsum_pass
    n_total = sum(len(v) for v in spectra.values())
    log.line(logger, f"Generated {n_total} SIMLIB spectra for {len(spectra)} SNe")
    return spectra, jnp.asarray(snrsum)


def compute_sed_true_synmags(spectra_by_snid, photometry_resources, calib) -> None:
    """Attach per-band synthetic mags (SIM_SYNMAG) to each spectrum.

    Integrates the observer-frame TRUE FLAM through each band and applies the
    same broadband ZP as SIM_MAGOBS (GENSPEC_SYNMAG, snlc_sim.c:11364); bands
    the spectrograph does not cover get MAG_ZEROFLUX=99. SNANA integrates the
    full-grid GENFLAM_LIST (snlc_sim.c:11281), not just the written bins, so
    the grid comes from the calibration rather than the ragged spectra.
    """
    if not spectra_by_snid:
        return
    names = list(photometry_resources.filter_names)
    short = [filter_band(n) for n in names]
    wave = np.asarray(photometry_resources.common_wave_grid)      # observer frame
    trans = np.asarray(photometry_resources.common_trans_grid)    # (n_filt, n_wave)
    ref_mag_terms = np.asarray(photometry_resources.filter_ref_mag_terms)
    tl = trans * wave[None, :]
    norm = np.trapezoid(tl, wave, axis=1)                         # ∫ T·λ dλ per band
    obs = observed_bins(calib)
    lam_min = np.asarray(calib.lam_min)[obs]
    lam_max = np.asarray(calib.lam_max)[obs]
    lam_avg = 0.5 * (lam_min + lam_max)

    # transmission overlap test (OVERLAP_MIN, sntools_spectrograph.c:1263)
    lam_lo = float(lam_min[0])
    lam_hi = float(lam_max[-1])
    inside = (wave > lam_lo) & (wave < lam_hi)
    overlap = trans[:, inside].sum(axis=1) / trans.sum(axis=1)
    do_synfilt = overlap > OVERLAP_MIN_SYNFILT

    for specs in spectra_by_snid.values():
        for spec in specs:
            fl = np.interp(wave, lam_avg, spec.sim_flam_full, left=0.0, right=0.0)
            fint = np.trapezoid(fl[None, :] * tl, wave, axis=1) / norm
            mags = np.where(do_synfilt & (fint > 0),
                            -2.5 * np.log10(np.maximum(fint, 1e-30)) + ref_mag_terms,
                            MAG_ZEROFLUX)
            spec.synmag = {short[i]: float(mags[i]) for i in range(len(names))}


def generate_all_spectra(
    calib: SpectrographCalibration,
    take_spectrum_configs: list,
    sn_parameters: Dict[str, jnp.ndarray],
    static_data: Dict,
    texpose_template_scale: float,
    trest_range,
    rng_key: jax.Array,
    passing_mask: np.ndarray,
    spectro_options: Dict,
    host_spec: Optional[Dict],
    hostfrac: float,
    hostsnfrac: float,
) -> Tuple[Dict[int, List[SpectrumResult]], jnp.ndarray]:
    """Spectra from input-file TAKE_SPECTRUM configs, batched across SNe.

    Epochs are drawn from each config's range with per-(event, config) keys.
    An entry whose centre-of-range epoch falls outside the event's
    GENRANGE_TREST window, padded by GENRANGE_TOBS_PAD/(1+z), is skipped for
    that event (SIMLIB_addCadence_TAKE_SPECTRUM, snlc_sim.c:20374-20419); such
    entries still take part in the near-peak scan, as in SNANA.

    Returns ``(spectra_by_snid, snrsum_rest_v)``: a dict mapping SNID to a
    list of SpectrumResult, and the (N_sne,) rest-frame V-band SNR sum.
    """
    opt = _parse_spectro_options(spectro_options, calib)
    ctx = _event_context(sn_parameters, passing_mask, static_data)
    n_sne, n_pass = ctx['n_sne'], ctx['n_pass']
    if n_pass == 0 or not take_spectrum_configs:
        return {}, jnp.zeros(n_sne)

    template_scale = float(texpose_template_scale)
    if template_scale <= 0.01:
        template_scale = 0.0

    peak = ctx['peakmjd']
    z1 = np.asarray(ctx['z1'])
    z1c = 1.0 + ctx['z_cmb']
    pad = GENRANGE_TOBS_PAD / z1c
    trest_lo_pad = trest_range[0] - pad
    trest_hi_pad = trest_range[1] + pad

    rows = _empty_rows()
    for c, config in enumerate(take_spectrum_configs):
        frame = config['epoch_frame']
        lo, hi = config['epoch_range']
        mode = int(config['opt_texpose'])
        if frame not in ('HOST', 'REST', 'OBS', 'MJD') or mode not in (1, 2):
            continue
        if frame == 'HOST':
            ph = np.zeros(n_pass)
            mjd_c = np.full(n_pass, -9.0)
            skip_c = np.zeros(n_pass, dtype=bool)
        else:
            if lo != hi:
                keys_c = _epoch_row_keys(rng_key, _SCHEDULE_TAKE_SPECTRUM,
                                         ctx['global_idx'], np.full(n_pass, c))
                u = np.asarray(vmap(
                    lambda k: jax.random.uniform(jax.random.fold_in(k, 0)))(keys_c))
                ep = lo + u * (hi - lo)
            else:
                ep = np.full(n_pass, float(lo))
            if frame == 'REST':
                ph = ep
                mjd_c = peak + ep * z1
                trest_c = np.full(n_pass, 0.5 * (lo + hi))
            elif frame == 'OBS':
                ph = ep / z1
                mjd_c = peak + ep
                trest_c = 0.5 * (lo + hi) / z1c
            else:  # MJD
                mjd_c = ep
                ph = (ep - peak) / z1
                trest_c = (0.5 * (lo + hi) - peak) / z1c
            skip_c = ~((trest_lo_pad < trest_c) & (trest_c < trest_hi_pad))
        for e in range(n_pass):
            _append_row(
                rows, sn_local=e, entry_index=c, phase=float(ph[e]),
                mjd=float(mjd_c[e]), mode=mode,
                zpoly_texpose=config['zpoly_texpose'],
                zpoly_snr=config['zpoly_snr'],
                snr_lo=float(config['snr_lamrange'][0]),
                snr_hi=float(config['snr_lamrange'][1]),
                snr_rest=config['snr_lam_frame'] == 'REST',
                is_host=frame == 'HOST',
                skip=bool(skip_c[e]) or not _spectrum_kept(
                    static_data, ctx['pass_idx'][e], c))
    if not rows['sn_local']:
        return {}, jnp.zeros(n_sne)

    log.line(logger, f"Generating spectra for {n_pass} SNe, "
             f"{len(take_spectrum_configs)} configs...")
    spectra, snrsum_pass = _generate_spectra_batch(
        calib, opt, ctx, _finalize_rows(rows), rng_key,
        _SCHEDULE_TAKE_SPECTRUM, template_scale=template_scale,
        host_spec=host_spec, hostfrac=hostfrac, hostsnfrac=hostsnfrac)

    snrsum = np.zeros(n_sne)
    snrsum[ctx['pass_idx']] = snrsum_pass
    n_total = sum(len(v) for v in spectra.values())
    log.line(logger, f"Generated {n_total} spectra for {len(spectra)} SNe")
    return spectra, jnp.asarray(snrsum)


def generate_simlib_take_spectrum(
    calib: SpectrographCalibration,
    entries_per_sn: list,
    sn_parameters: Dict[str, jnp.ndarray],
    static_data: Dict,
    texpose_template_scale: float,
    trest_range,
    rng_key,
    passing_mask,
    spectro_options,
):
    """Spectra from per-LIBID TAKE_SPECTRUM headers (USE_SIMLIB_TAKE_SPECTRUM).

    Same semantics as the input-file TAKE_SPECTRUM path, but each SN carries
    its own schedule. SNANA cuts entries at parse time on the range's LOWER
    edge against GENRANGE_TREST (snlc_sim.c:5794-5815), so gated entries never
    reach the GENSPEC lists and take no part in the near-peak scan. An entry
    with SNR_LAMMIN > 0.1 clips the smeared/noised/written wavelength range to
    the RAW SNR window (GENSPEC_DRIVER, snlc_sim.c:10600-10606).
    """
    opt = _parse_spectro_options(spectro_options, calib)
    ctx = _event_context(sn_parameters, passing_mask, static_data)
    n_sne = ctx['n_sne']
    if ctx['n_pass'] == 0:
        return {}, jnp.zeros(n_sne)

    template_scale = float(texpose_template_scale)
    if template_scale <= 0.01:
        template_scale = 0.0

    rows = _empty_rows()
    erange = []
    for local, gi in enumerate(ctx['pass_idx']):
        z1c = 1.0 + ctx['z_cmb'][local]   # parse gate z is z_cmb (snlc_sim.c:5800)
        for k, entry in enumerate(entries_per_sn[gi]):
            lo, hi = entry['epoch_range']
            trest_lo = lo if entry['epoch_frame'] == 'REST' else lo / z1c
            if not (trest_range[0] <= trest_lo <= trest_range[1]):
                continue
            # phase and mjd are filled by the batched draw below
            _append_row(
                rows, sn_local=local, entry_index=k,
                mode=int(entry['opt_texpose']),
                zpoly_texpose=entry['zpoly_texpose'],
                zpoly_snr=entry['zpoly_snr'],
                snr_lo=float(entry['snr_lamrange'][0]),
                snr_hi=float(entry['snr_lamrange'][1]),
                snr_rest=entry['snr_lam_frame'] == 'REST',
                clip=float(entry['snr_lamrange'][0]) > 0.1,
                skip=not _spectrum_kept(static_data, gi, k))
            erange.append((float(lo), float(hi), entry['epoch_frame'] == 'REST'))
    if not rows['sn_local']:
        return {}, jnp.zeros(n_sne)
    prepared = _finalize_rows(rows)

    loc = prepared['sn_local']
    lo_arr = np.array([r[0] for r in erange])
    hi_arr = np.array([r[1] for r in erange])
    rest_arr = np.array([r[2] for r in erange])
    keys = _epoch_row_keys(rng_key, _SCHEDULE_SIMLIB_TAKE_SPECTRUM,
                           ctx['global_idx'][loc], prepared['entry_index'])
    u = np.asarray(vmap(lambda k: jax.random.uniform(jax.random.fold_in(k, 0)))(keys))
    ep = np.where(lo_arr != hi_arr, lo_arr + u * (hi_arr - lo_arr), lo_arr)
    z1 = np.asarray(ctx['z1'])[loc]
    prepared['phase'] = np.where(rest_arr, ep, ep / z1)
    prepared['mjd'] = ctx['peakmjd'][loc] + prepared['phase'] * z1

    spectra, snrsum_pass = _generate_spectra_batch(
        calib, opt, ctx, prepared, rng_key, _SCHEDULE_SIMLIB_TAKE_SPECTRUM,
        template_scale=template_scale)

    snrsum = np.zeros(n_sne)
    snrsum[ctx['pass_idx']] = snrsum_pass
    n_total = sum(len(v) for v in spectra.values())
    log.line(logger, f"Generated {n_total} SIMLIB TAKE_SPECTRUM "
             f"spectra for {len(spectra)} SNe")
    return spectra, jnp.asarray(snrsum)


def _get_lamrange_sed_true(lamrange_user, kcor, model_wave_range):
    """Ideal-spectrograph wavelength range (get_LAMRANGE_SED_TRUE, genmag_SEDtools.c:2036)."""
    names = kcor.get_filter_names()
    # per-filter nonzero-transmission support (the kcor grid spans all bands)
    supports = []
    for n in names:
        wave, trans = (np.asarray(a) for a in kcor.get_filter_transmission(n))
        nz = np.where(trans > 0)[0]
        supports.append((float(wave[nz[0]]), float(wave[nz[-1]])))
    lmin_flt = min(s[0] for s in supports)
    lmax_flt = max(s[1] for s in supports)
    if lamrange_user is not None and lamrange_user[0] > 0.01 and lamrange_user[1] > 0.01:
        if lamrange_user[0] > lmin_flt or lamrange_user[1] < lmax_flt:
            raise SNANAError(
                f"LAMRANGE_SED_TRUE {lamrange_user} can extend the filter range "
                f"({lmin_flt:.0f}-{lmax_flt:.0f}), not reduce it")
        return lamrange_user[0], lamrange_user[1]
    lmin_sed, lmax_sed = model_wave_range
    if lmin_sed > 0.01:
        return min(lmin_flt, lmin_sed), max(lmax_flt, lmax_sed)
    return lmin_flt, lmax_flt


def load_spectrograph(config, kcor):
    """Load the spectrograph calibration: a standalone SPECTROGRAPH_FILE,
    else the spectrograph supplied by KCOR_FILE. None without either,
    and under SEDMODEL mode (bit 128), whose ideal spectrograph is
    fabricated from the model's range once the model is built, whatever the
    kcor holds (init_genSpec, snlc_sim.c:10264)."""
    if config.effective_spectrograph_optmask & 128:
        return None
    calibration = None
    spec_file = config.spectra.spectrograph_file
    if spec_file:
        calibration = read_spectrograph(Path(os.path.expandvars(spec_file)))
    if calibration is None:
        calibration = kcor.spectrograph
    if calibration is not None:
        calibration.zp_sedmodel = jnp.array(compute_zp_sedmodel(
            calibration.lam_min, calibration.lam_max, kcor))
    return calibration


def build_sed_true_spectrograph(config, kcor, model_wave_range):
    """The ideal spectrograph of SEDMODEL mode (bit 128), over the SED-true
    range."""
    if config.spectra.lambin_sed_true <= 0.001:
        raise SNANAError(
            "SPECTROGRAPH_OPTMASK bit 128 requires LAMBIN_SED_TRUE > 0.001 "
            "(bare bit 128 crashes SNANA)")
    lammin, lammax = _get_lamrange_sed_true(
        config.spectra.lamrange_sed_true, kcor, model_wave_range)
    calibration = build_ideal_spectrograph(
        lammin, lammax, config.spectra.lambin_sed_true)
    calibration.zp_sedmodel = jnp.array(compute_zp_sedmodel(
        calibration.lam_min, calibration.lam_max, kcor))
    return calibration


def spectrum_prescales(simulator, results, *, key):
    """Keep decisions for each event's spectrum entries, before scheduling."""
    cfg = simulator.config.spectra
    params = results['parameters']
    field_indices = np.asarray(params['field_idx'])
    fields = np.array([simulator.fields[int(i)].field_name for i in field_indices])
    default = {'field': '', 'wgt': 1.0, 'epoch_frame': ''}
    if simulator.config.effective_spectrograph_optmask & 128:
        groups = [(np.arange(len(fields)), [default])]
    elif simulator.config.survey.use_simlib_take_spectrum:
        groups = [(np.flatnonzero(field_indices == i), simulator.simlib_take_entries[i])
                  for i in np.unique(field_indices)]
    else:
        entries = ([default] if simulator.config.survey.use_simlib_spectrograph
                   else cfg.take_spectrum or [default])
        groups = [(np.arange(len(fields)), entries)]
    if not (cfg.take_spectrum_prescale or cfg.take_spectrum_dumpcid >= 0
            or any(e['field'] or e['wgt'] < 1 for _, es in groups for e in es)):
        return None
    key_draws = event_keys(key, 'SPEC_PRESCALE_KEY', params['sn_global_idx'])
    field_draws = np.asarray(jax.vmap(jax.random.uniform)(
        event_keys(key, 'SPEC_PRESCALE_FIELD', params['sn_global_idx'])))
    max_entries = max((len(es) for _, es in groups), default=0)
    draws = []
    for index in range(max_entries):
        keys = (key_draws if index == 0 else
                jax.vmap(lambda k: jax.random.fold_in(k, index))(key_draws))
        draws.append(np.asarray(jax.vmap(jax.random.uniform)(keys)))
    keep = np.zeros((len(fields), max_entries), dtype=bool)
    for rows, entries in groups:
        for c, entry in enumerate(entries):
            keep[rows, c] = keep_spectrum(
                fields[rows], entry['field'], entry['wgt'],
                cfg.take_spectrum_prescale,
                draws[entry.get('prescale_key', c)][rows], field_draws[rows],
                is_host=entry['epoch_frame'] == 'HOST')
    for i in np.flatnonzero(np.asarray(params['cid']) == cfg.take_spectrum_dumpcid):
        logger.debug("TAKE_SPECTRUM_DUMPCID %d: field=%s keep=%s",
                     int(params['cid'][i]), fields[i], keep[i])
    return keep


def sed_true_epochs(results):
    """Per event, the MJDs the SED-true spectra are taken at: every
    scheduled epoch, deduplicated at 0.05 d (snlc_sim.c:10953). A packed
    run carries them under ``sed_true_mjds`` (taken per batch, since the
    pack keeps only the written epochs); a batch reads its schedule."""
    n_events = len(results['observation_schedule']['n_obs_per_sn'])
    if 'sed_true_mjds' in results:
        mjds, count = results['sed_true_mjds'], results['sed_true_count']
        start = np.cumsum(count) - count
        return [mjds[a:a + c] for a, c in zip(start, count)]
    mjds = np.asarray(results['observation_schedule']['mjds'],
                      dtype=np.float64).reshape(-1)
    epochs = []
    for row in range(n_events):
        m = mjds[row_slice(results, row)]
        epochs.append(m[np.diff(m, prepend=-99.0) > 0.05])
    return epochs


def compute_spectra_for_mask(simulator, results, mask, *,
                             key):
    """Generate spectra for the SNe in ``mask``, every draw off the root
    ``key``. Returns ``(spectra_by_snid, snrsum_rest_v)``.
    """
    cfg = simulator.config
    if cfg.spectra.spectrograph_optmask & 2048:  # NOSPEC
        n_sne = len(results['parameters']['sim_redshift_cmb'])
        return {}, jnp.zeros(n_sne)

    grey, host_shift, magnitude_shift = grey_offsets(
        results['parameters'], simulator.config.model, simulator.model.non1a_templates)
    sn_flux_factor = jnp.power(10.0, -0.4 * (grey + host_shift + magnitude_shift))
    static_data_with_model = dict(
        simulator.photometry_data, sn_flux_factor=sn_flux_factor,
        model_instance=simulator.model, latetime_extrap=True,
        gensmear=simulator.gensmear)

    static_data_with_model['spectrum_keep'] = spectrum_prescales(
        simulator, results, key=key)

    rng_key = stream_key(key, 'SPECTRUM_PRESCALE')

    if cfg.effective_spectrograph_optmask & 128:
        # true-SED spectra at every photometry epoch, dedup 0.05 d, TEXPOSE 20 (snlc_sim.c:10953)
        spec_mjds_list = sed_true_epochs(results)
        # epochs outside the model's phase support (genmag_SALT2.c:3592-3595)
        mi = simulator.model
        daymin, daymax = mi.phase_range()
        trest_hi = (daymax - SED_PHASE_EDGE_EPS
                    if mi.extrap_phase_method == 1 else np.inf)
        return generate_simlib_spectra(
            calib=simulator.calibration,
            sn_parameters=results['parameters'],
            spec_mjds_list=spec_mjds_list,
            spec_texposes_list=[np.full(len(m), 20.0) for m in spec_mjds_list],
            static_data=static_data_with_model,
            template_texpose=0.0,
            trest_range=(daymin + 0.01, trest_hi),
            rng_key=rng_key,
            passing_mask=np.asarray(mask),
            # the noise-free SED: SCALE_TEXPOSE and SCALE_SNR have nothing to act on
            spectro_options={
                'optmask': cfg.effective_spectrograph_optmask,
                'zero_neg_flam': bool(cfg.model.genmodel_mskopt & 16),
            },
        )

    spectro_options = {
        'optmask': cfg.spectra.spectrograph_optmask,
        'scale_snr': cfg.spectra.spectrograph_scale_snr,
        'warp_lampoly': cfg.spectra.warp_spectrum_lampoly,
        'zero_neg_flam': bool(cfg.model.genmodel_mskopt & 16),
    }
    if cfg.survey.use_simlib_take_spectrum:
        fidx = np.asarray(results['parameters']['field_idx'])
        entries_by_field = simulator.simlib_take_entries
        return generate_simlib_take_spectrum(
            calib=simulator.calibration,
            entries_per_sn=[entries_by_field[i] for i in fidx],
            sn_parameters=results['parameters'],
            static_data=static_data_with_model,
            texpose_template_scale=cfg.spectra.take_spectrum_template_texpose_scale,
            trest_range=cfg.rates.trest_range,
            rng_key=rng_key,
            passing_mask=np.asarray(mask),
            spectro_options=spectro_options,
        )

    if cfg.survey.use_simlib_spectrograph:
        fields = simulator.fields
        fidx = np.asarray(results['parameters']['field_idx'])
        return generate_simlib_spectra(
            calib=simulator.calibration,
            sn_parameters=results['parameters'],
            spec_mjds_list=[fields[i].spec_mjds for i in fidx],
            spec_texposes_list=[fields[i].spec_texposes for i in fidx],
            static_data=static_data_with_model,
            # GENSPEC.TEXPOSE_TEMPLATE stays 0 for SIMLIB spectra
            template_texpose=0.0,
            trest_range=cfg.rates.trest_range,
            rng_key=rng_key,
            passing_mask=np.asarray(mask),
            spectro_options=spectro_options,
        )

    return generate_all_spectra(
        calib=simulator.calibration,
        take_spectrum_configs=simulator.config.spectra.take_spectrum,
        sn_parameters=results['parameters'],
        static_data=static_data_with_model,
        texpose_template_scale=cfg.spectra.take_spectrum_template_texpose_scale,
        trest_range=cfg.rates.trest_range,
        rng_key=rng_key,
        passing_mask=np.asarray(mask),
        spectro_options=spectro_options,
        host_spec=(None if simulator.hostlib is None
                   else simulator.hostlib.spectra),
        hostfrac=cfg.spectra.take_spectrum_hostfrac,
        hostsnfrac=cfg.spectra.take_spectrum_hostsnfrac,
    )
