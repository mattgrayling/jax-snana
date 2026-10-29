"""The tabulated-SED tools the models share, and what an SED model is,
:class:`SNModel`.

The tools read 3-column SED files (rd_sedFlux); extrapolate a model's flux
before and after its phase range (modelflux_extrap, genmag_extrap.c) and a
Ia's late-time flux by a double exponential (init_extrap_latetime_Ia and
genflux_extrap_latetime_Ia); find a template's explosion day
(T0shiftExplode); extend a surface to short wavelengths
(UVLAM_EXTRAPFLUX_SEDMODEL); and fit, tabulate and apply the late-time
magnitude decay of a template, and resample surfaces onto a model's grid.
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core import logging as log
from jax_snlc_core.dust import load_dust_law
from jax_snlc_core.filters import fluxcal_from_sed
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import COMMENT_CHARS, open_text
from jax_snlc_core.utils import NEG_0_4_LN10

from .input_keys import model_input_keys

logger = logging.getLogger(__name__)

# LAMMIN/LAMMAX_SEDMODEL
SED_WAVELENGTH_RANGE = (100.0, 30000.0)
# rd_sedFlux's tolerances on uniform day and wavelength steps
# (sntools.c:8426-8427)
SED_DAYSTEP_TOL = 0.5e-3
SED_LAMSTEP_TOL = 0.01


def read_sed_file(filepath: str, day_range: Tuple[float, float],
                  wavelength_range: Tuple[float, float],
                  t0shift_peakmag: bool = True, *, uniform_days: bool
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a 3-column SED file (phase, wavelength, flux) at ``filepath``
    (``.SED`` or ``.SED.gz``), as SNANA's rd_sedFlux does for NON1ASED and
    SIMSED templates and SALT surfaces and error maps, skipping blank lines
    and comment lines (first character ``#``, ``!``, ``%`` or ``@``). A data
    line with fewer than three words, or whose first three words are not
    numbers, is an error.

    Rows outside ``day_range`` or ``wavelength_range`` (both inclusive) are
    excluded, and NaN fluxes are read as zero. Phases must be in increasing
    order, with the same wavelengths in the same order at each phase.
    With ``uniform_days``, phase spacing must be constant within
    ``SED_DAYSTEP_TOL``. Non-uniform wavelength spacing produces a warning.

    When ``t0shift_peakmag`` (SNANA's default-on ``OPTMASK_T0SHIFT_PEAKMAG``,
    disabled by ``OPTFLAG_T0SHIFT_PEAKMAG: 0`` in a SIMSED SED.INFO), the
    phase grid is shifted so that peak bolometric flux occurs at phase=0.

    Returns ``(phases, wavelengths, flux_grid)``: phases is the 1-D array of
    unique phase values (days), wavelengths the 1-D array of unique
    wavelength values (Angstroms), and flux_grid the 2-D array of shape
    (n_phases, n_wavelengths).
    """
    filepath = str(filepath)

    data_rows = []
    with open_text(filepath) as source:
        for number, line in enumerate(source, start=1):
            if not line.strip() or line[0] in COMMENT_CHARS:
                continue
            parts = line.split()
            if len(parts) < 3:
                raise SNANAError(f"{filepath}: {len(parts)} words, expected at "
                                 f"least 3, in line '{line.rstrip()}'",
                                 function_name="read_sed_file")
            try:
                data_rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                raise SNANAError(f"{filepath}: line {number} '{line.rstrip()}' is not "
                                 f"three numbers", function_name="read_sed_file") from None

    if not data_rows:
        raise SNANAError(f"No valid SED data found in {filepath}",
                         function_name="read_sed_file")

    data = np.array(data_rows)
    inside = ((data[:, 0] >= day_range[0]) & (data[:, 0] <= day_range[1])
              & (data[:, 1] >= wavelength_range[0])
              & (data[:, 1] <= wavelength_range[1]))
    data = data[inside]
    if not len(data):
        raise SNANAError(
            f"No SED rows of {filepath} inside days {day_range} and "
            f"wavelengths {wavelength_range}", function_name="read_sed_file")
    raw_phases = data[:, 0]
    raw_wavelengths = data[:, 1]
    raw_flux = np.nan_to_num(data[:, 2], nan=0.0, posinf=np.inf,
                             neginf=-np.inf)

    # days never decrease (sntools.c:8577-8581)
    decreasing = np.flatnonzero(np.diff(raw_phases) < 0)
    if decreasing.size:
        i = decreasing[0]
        raise SNANAError(
            f"{filepath}: day {raw_phases[i + 1]:f} follows day "
            f"{raw_phases[i]:f}; a new day must increment",
            function_name="read_sed_file")
    # every day has as many wavelengths as the first (:8675-8690)
    phases, per_day = np.unique(raw_phases, return_counts=True)
    if np.any(per_day != per_day[0]):
        k = int(np.argmax(per_day != per_day[0]))
        raise SNANAError(
            f"{filepath}: {per_day[k]} wavelengths at day {phases[k]:.2f} but "
            f"{per_day[0]} at day {phases[0]:.2f}; the wavelength grid must "
            f"be the same for each day", function_name="read_sed_file")
    # ... and lists the first day's wavelengths, in order (:8612-8624)
    n_phases, n_wavelengths = len(phases), int(per_day[0])
    lam_grid = raw_wavelengths.reshape(n_phases, n_wavelengths)
    wavelengths = lam_grid[0]
    # a falling wavelength restarts the list at its first entry (:8583)
    falling = np.flatnonzero(np.diff(wavelengths) < 0)
    if falling.size:
        j = falling[0] + 1
        raise SNANAError(
            f"{filepath}: found LAM={wavelengths[j]:.1f} at day "
            f"{phases[0]:f} but expected LAM={wavelengths[0]:.2f}",
            function_name="read_sed_file")
    mismatched = np.argwhere(lam_grid != wavelengths)
    if mismatched.size:
        k, j = mismatched[0]
        raise SNANAError(
            f"{filepath}: found LAM={lam_grid[k, j]:.1f} at day "
            f"{phases[k]:f} but expected LAM={wavelengths[j]:.2f}",
            function_name="read_sed_file")
    # uniform day steps unless the model allows otherwise (:8549-8560)
    day_step = np.diff(phases)
    step_change = np.abs(np.diff(day_step)) > SED_DAYSTEP_TOL
    if uniform_days and step_change.any():
        k = int(np.argmax(step_change)) + 1
        raise SNANAError(
            f"{filepath}: day step {day_step[k]:e} after {day_step[k - 1]:e} "
            f"at day {phases[k + 1]:.4f}; the day bins must be uniform",
            function_name="read_sed_file")
    # a non-uniform wavelength step only warns (:8595-8606)
    if np.any(np.abs(np.diff(np.diff(wavelengths))) > SED_LAMSTEP_TOL):
        log.warn(logger, 'SED', f"{filepath}: non-uniform wavelength step "
                 f"(rd_sedFlux warns and reads on)")

    # phase varies slowest, wavelength fastest
    flux_grid = raw_flux.reshape(n_phases, n_wavelengths)

    # T0shiftPeak_SEDMODEL: peak bolometric flux at day 0
    if t0shift_peakmag:
        bolometric = flux_grid.sum(axis=1)
        peak_idx = np.argmax(bolometric)
        phases = phases - phases[peak_idx]

    return phases, wavelengths, flux_grid


def modelflux_extrap(phase: float, phase_edge: float,
                     flux_edge: jnp.ndarray, flux_slope: jnp.ndarray) -> jnp.ndarray:
    """Extrapolate model flux beyond template phase boundaries.

    Matches SNANA's modelflux_extrap() in genmag_extrap.c.

    Pre-max (phase < phase_edge):
        Fireball rise: F0 * (T-T0)^2 * (T-T1), matching function and derivative
        at phase_edge, with explosion epoch T0 = -20d.

    Post-max (phase > phase_edge):
        Power-law decay: flux_edge * 10^(-(phase-phase_edge)*|slope|/(ln10*flux_edge))
        which is linear mag extrapolation.

    All branches produce finite values regardless of inputs for autodiff safety
    (JAX's jnp.where evaluates gradients through both branches).

    ``phase`` is the target rest-frame phase (days); ``phase_edge`` the
    nearest phase within template support; ``flux_edge`` the flux values at
    ``phase_edge`` (shape = wavelength grid); ``flux_slope`` dflux/dT
    evaluated near ``phase_edge``. Returns the extrapolated flux array.
    """
    T0 = -20.0  # explosion epoch
    slope = jnp.abs(flux_slope)

    # exponent clamped <= 0: the tail only dims
    safe_flux_edge = jnp.where(flux_edge > 0.0, flux_edge, 1.0)
    exp_arg_raw = -(phase - phase_edge) * slope / safe_flux_edge
    exp_arg = jnp.minimum(exp_arg_raw, 0.0)
    tail = flux_edge * jnp.exp(exp_arg)
    tail = jnp.where(flux_edge > 0.0, tail, 1e-30)

    # no rise at or below T0; dummy flux 1, slope 0 keep the untaken branch's derivatives finite
    dt_ref0 = phase_edge - T0
    rises = (dt_ref0 > 0.0) & (phase >= T0)
    safe_dt_ref0 = jnp.where(dt_ref0 > 0.0, dt_ref0, 1.0)
    rise_flux = jnp.where(rises, flux_edge, 1.0)
    rise_slope = jnp.where(rises, flux_slope, 0.0)
    sq_dt = safe_dt_ref0 * safe_dt_ref0
    cube_dt = sq_dt * safe_dt_ref0
    F0 = (rise_slope * safe_dt_ref0 - 2.0 * rise_flux) / cube_dt
    safe_denom = jnp.where(jnp.abs(F0 * sq_dt) > 1e-30, F0 * sq_dt, 1e-30)
    T1 = phase_edge - rise_flux / safe_denom
    dt0 = phase - T0
    dt1 = phase - T1
    rise = jnp.where(rises, F0 * dt0 * dt0 * dt1, 0.0)

    return jnp.where(phase > phase_edge, tail, rise)


class LatetimeExtrapIa:
    """Late-time Ia flux extrapolation: F(t) = exp(-t/tau1) + ratio*exp(-t/tau2).

    Parses SNANA-format latetime model files containing EXTRAP_DAYMIN and
    EXTRAP_PARLIST rows, providing per-wavelength decay factors for phases
    beyond daymin.

    File format::

        EXTRAP_DAYMIN: 45.0
        EXTRAP_PARLIST: 3500.0  30.0  60.0  0.05
        EXTRAP_PARLIST: 5000.0  35.0  80.0  0.10
        ...

    Each EXTRAP_PARLIST row: LAM  TAU1  TAU2  EXPRATIO
    """

    def __init__(self, filename: str):
        """Parse the late-time extrapolation model file."""
        filename = Path(os.path.expandvars(filename)).expanduser()
        if not filename.exists():
            raise SNANAError(f"Late-time model file not found: {filename}",
                             function_name="LatetimeExtrapIa")

        daymin = None
        parlist_rows = []

        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('EXTRAP_DAYMIN:'):
                    daymin = float(line.split(':')[1].strip())
                elif line.startswith('EXTRAP_PARLIST:'):
                    val_str = line.split(':')[1].split('#')[0]
                    vals = [float(x) for x in val_str.split()]
                    if len(vals) != 4:
                        raise SNANAError(
                            f"EXTRAP_PARLIST expects 4 values (LAM TAU1 TAU2 RATIO), "
                            f"got {len(vals)}: {line}",
                            function_name="LatetimeExtrapIa")
                    parlist_rows.append(vals)

        if daymin is None:
            raise SNANAError(f"EXTRAP_DAYMIN not found in {filename}",
                             function_name="LatetimeExtrapIa")
        if len(parlist_rows) == 0:
            raise SNANAError(f"No EXTRAP_PARLIST rows found in {filename}",
                             function_name="LatetimeExtrapIa")
        if daymin < 10.0:
            raise SNANAError(f"Invalid DAYMIN={daymin:.2f} (too small) in {filename}",
                             function_name="LatetimeExtrapIa")

        data = np.array(parlist_rows, dtype=np.float64)
        self.daymin = daymin
        self.lam_bins = data[:, 0]  # wavelength centres
        self.tau1 = data[:, 1]
        self.tau2 = data[:, 2]
        self.expratio = data[:, 3]
        self.nlambin = len(parlist_rows)

        for i in range(self.nlambin):
            if self.tau2[i] < self.tau1[i]:
                raise SNANAError(
                    f"Invalid TAU2({self.tau2[i]:.2f}) < TAU1({self.tau1[i]:.2f}) "
                    f"at lam={self.lam_bins[i]:.1f}", function_name="LatetimeExtrapIa")

        self.jax_lam_bins = jnp.array(self.lam_bins)
        self.jax_tau1 = jnp.array(self.tau1)
        self.jax_tau2 = jnp.array(self.tau2)
        self.jax_expratio = jnp.array(self.expratio)

        log.line(logger, f"Late-time Ia extrap: DAYMIN={daymin:.0f}, "
                 f"{self.nlambin} wavelength bins "
                 f"({self.lam_bins[0]:.0f}-{self.lam_bins[-1]:.0f} Å)")

    def precompute_decay_params(self, wavelength_grid: jnp.ndarray):
        """Interpolate tau1, tau2 and expratio onto the working wavelength grid.

        Called once at init.
        """
        # held at the end values outside the table, as genflux_extrap_latetime_Ia
        tau1_grid = jnp.interp(wavelength_grid, self.jax_lam_bins, self.jax_tau1)
        tau2_grid = jnp.interp(wavelength_grid, self.jax_lam_bins, self.jax_tau2)
        ratio_grid = jnp.interp(wavelength_grid, self.jax_lam_bins, self.jax_expratio)
        return tau1_grid, tau2_grid, ratio_grid

    @staticmethod
    def decay_factor(day: float, daymin: float,
                     tau1: jnp.ndarray, tau2: jnp.ndarray,
                     expratio: jnp.ndarray) -> jnp.ndarray:
        """Compute multiplicative decay factor F(day)/F(daymin) per wavelength.

        The decay function is::

            F(t) = exp(-t/tau1) + ratio * exp(-t/tau2)

        where t = day - daymin.

        Returns F(t) / F(0) so that flux(day) = flux(daymin) * decay_factor.

        Autodiff safe: clamps t >= 0 so unused branch (day < daymin) produces
        decay_factor = 1.0 (finite, no overflow).

        ``day`` is the rest-frame phase (scalar, under vmap); ``daymin`` the
        reference phase for decay start; ``tau1``/``tau2`` the fast/slow
        decay timescales per wavelength, shape (n_wl,); ``expratio`` the
        ratio of slow/fast component per wavelength, shape (n_wl,). Returns
        the per-wavelength decay factor, shape (n_wl,), values in (0, 1].
        """
        safe_t = jnp.maximum(day - daymin, 0.0)
        F_t = jnp.exp(-safe_t / tau1) + expratio * jnp.exp(-safe_t / tau2)
        F_0 = 1.0 + expratio
        return F_t / F_0


def compute_explosion_tshift(flux, day, optmask):
    """Return the SNANA T0shiftExplode day offset for one SED template. Explosion is the
    day bin preceding the first wavelength bin to exceed 1% (OPTMASK&1) or
    0.1% (OPTMASK&2) of the template's global flux max, searched up to the
    day of peak flux; OPTMASK&32 is the bolometric peak plus a fixed 32 d;
    any other mask aborts. ``DAY -= tshift`` then places T=0 at explosion.

    ``flux`` is the (n_phase, n_wl) native flux surface and ``day`` the
    (n_phase,) native day grid, monotonic increasing.
    """
    if optmask & 1:
        frac = 0.01
    elif optmask & 2:
        frac = 0.001
    elif optmask & 32:
        bolometric = np.asarray(flux).sum(axis=1)
        return float(day[int(np.argmax(bolometric))]) + 32.0
    else:
        raise SNANAError(
            f"Invalid OPTMASK={optmask} for T0shiftExplode",
            function_name="compute_explosion_tshift")
    iday_max = int(np.argmax(flux) // flux.shape[1])
    flux_explode = frac * float(flux.max())
    crosses = np.any(flux[1:iday_max] > flux_explode, axis=1)
    return float(day[:iday_max - 1][crosses].min()) if crosses.any() else 9999.0


def uvlam_extrapflux_sedmodel(uvlam, waves, flux):
    """Extrapolate one SED surface to shorter wavelengths.

    Apply this separately to each surface or template.

    Extend the wavelength grid to or below ``uvlam`` using its native
    spacing. At each phase, extend the blue end of the spectrum with a
    linear ramp from zero at the new grid minimum to the first significant
    positive flux sample, including original bins within that interval.

    ``uvlam`` is the target UV minimum wavelength (Angstroms), ``waves`` the
    (n_lam,) native uniform wavelength grid and ``flux`` the (n_day, n_lam)
    flux surface. Returns ``(new_waves, new_flux)``, the inputs unchanged
    when uvlam > waves[0].
    """
    waves = np.asarray(waves, dtype=np.float64)
    flux = np.asarray(flux)
    lam_min_orig = float(waves[0])
    lam_step = float(waves[1] - waves[0])

    if uvlam > lam_min_orig:
        return waves, flux

    # New LAMMIN on the original LAMSTEP grid (genmag_SEDtools.c:3249)
    n_add = 0
    lam_min_new = lam_min_orig
    while lam_min_new > uvlam:
        lam_min_new -= lam_step
        n_add += 1

    n_day, n_lam = flux.shape
    n_lam_new = n_lam + n_add
    new_waves = lam_min_new + lam_step * np.arange(n_lam_new)

    new_flux = np.zeros((n_day, n_lam_new), dtype=flux.dtype)
    new_flux[:, n_add:] = flux

    # Per-DAY anchor: smallest wavelength with signed flux > 1.0e-12
    # (genmag_SEDtools.c:3295-3296); no such bin -> ilam=0.
    above = flux > 1.0e-12
    has_anchor = above.any(axis=1)
    ilam_min = np.where(has_anchor, above.argmax(axis=1), 0)
    flux_edge = flux[np.arange(n_day), ilam_min]
    lammin_day = np.where(has_anchor, waves[ilam_min], lam_min_orig)

    # NLAM_EXTRAP = NLAM_ADD + ILAMMIN (genmag_SEDtools.c:3346-3357)
    n_extrap = n_add + ilam_min
    lam_extrap = lammin_day - lam_min_new
    fill = np.arange(n_lam_new)[None, :] < n_extrap[:, None]
    # lam_extrap is zero only where fill selects no bins for that day.
    denom = np.where(lam_extrap > 0.0, lam_extrap, 1.0)
    ramp = flux_edge[:, None] * (new_waves[None, :] - lam_min_new) / denom[:, None]
    new_flux = np.where(fill, ramp, new_flux).astype(flux.dtype)

    return new_waves, new_flux


def late_time_slope_fit(last_phases, last_seds, minslope):
    """Per-wavelength late-time magnitude slope from a template's last grid
    samples, shared by the
    SIMSED and NON1ASED loaders. SNANA fits six 1-day band-flux samples; jax
    fits the SED's last grid samples per wavelength (a band-integrated vs
    per-wavelength extrapolation difference).

    ``last_phases`` is the (NDAYFIT,) sample phases (days) and ``last_seds``
    the (NDAYFIT, n_lam) fluxes at those phases; ``minslope`` is the
    MINSLOPE_EXTRAPMAG_LATE floor (mag/day). Returns ``(slopes, reject)``
    with slopes (n_lam,) floored at ``minslope`` and reject (n_lam,) bool
    marking wavelengths where any sample is non-positive: SNANA rejects
    the whole extrapolation there and returns zero flux (:1551-1562). None
    when the phase spread is degenerate.
    """
    last_phases = np.asarray(last_phases, dtype=np.float64)
    last_seds = np.asarray(last_seds)
    reject = np.any(last_seds < 1.0e-30, axis=0)
    mags = 30.0 - 2.5 * np.log10(np.maximum(np.abs(last_seds), 1.0e-30))
    dt = last_phases - last_phases.mean()
    ss_t = np.sum(dt * dt)
    if ss_t < 1.0e-30:
        return None
    dm = mags - mags.mean(axis=0)[None, :]
    slopes = np.abs(np.sum(dt[:, None] * dm, axis=0) / ss_t)
    slopes = np.where(np.isfinite(slopes), slopes, 0.0)
    return np.maximum(slopes, minslope), reject


def resample_surfaces(native_waves, flux, grid):
    """Flux surfaces ``(..., n_phase, n_wave)`` on ``native_waves``, linearly
    resampled onto ``grid``; zero outside the native range, where SNANA's SED
    models return no flux."""
    native = np.asarray(native_waves, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    flux = np.asarray(flux)
    idx = np.clip(np.searchsorted(native, grid, side='right') - 1,
                  0, len(native) - 2)
    frac = np.clip((grid - native[idx])
                   / np.maximum(native[idx + 1] - native[idx], 1e-30), 0.0, 1.0)
    out = (1.0 - frac) * flux[..., idx] + frac * flux[..., idx + 1]
    out[..., (grid < native[0]) | (grid > native[-1])] = 0.0
    return out


def late_time_tables(phases, seds, n_valid, minslope):
    """Prepare late-time magnitude-decay slopes and boundary spectra.

    Each wavelength's decay is fitted to the template's last six valid
    phase samples. The boundary flux is set to zero where the extrapolation
    is rejected.

    ``phases`` is (n_templates, n_phase) sample phases, padded past
    ``n_valid``; ``seds`` is (n_templates, n_phase, n_wave) fluxes on the
    working grid; ``n_valid`` gives the number of valid samples in each
    template, shape (n_templates,); ``minslope`` is the minimum decay rate
    (mag/day). Returns ``(slopes, boundary_seds)``, each
    (n_templates, n_wave) float32.
    """
    NDAYFIT = 6
    n_templates, _, n_wave = seds.shape
    slopes = np.zeros((n_templates, n_wave), dtype=np.float32)
    boundary = np.zeros((n_templates, n_wave), dtype=np.float32)
    for i in range(n_templates):
        n = int(n_valid[i])
        if n > 0:
            boundary[i] = seds[i, n - 1]
        if n < NDAYFIT:
            continue
        fit = late_time_slope_fit(phases[i, n - NDAYFIT:n],
                                  seds[i, n - NDAYFIT:n], minslope)
        if fit is not None:
            slopes[i], reject = fit
            boundary[i, reject] = 0.0
    return slopes, boundary


def late_time_decay(sed, phase, phase_max, slopes, boundary_seds):
    """``sed`` with its late-time extrapolation from one day before
    ``phase_max`` on: the boundary spectrum dimmed by ``slopes`` mag/day
    since ``phase_max - 0.01`` (genmag_SEDtools.c:1530-1555).

    ``phase_max`` is a scalar, or one value per row of ``sed``; wavelength is
    the last axis of ``sed``, ``slopes`` and ``boundary_seds``, as
    :func:`late_time_tables` returns them.
    """
    # the -1 only guards the discarded branch
    dt = jnp.maximum(phase - (phase_max - 0.01), -1.0)
    decay = jnp.exp(NEG_0_4_LN10 * (slopes * jnp.asarray(dt)[..., None]))
    is_late = jnp.asarray(phase > phase_max - 1.0)[..., None]
    return jnp.where(is_late, boundary_seds * decay, sed)


@dataclass(init=False, eq=False, repr=False, kw_only=True)
class SNModel(ABC):
    """An SED model: built from its input, it gives a rest-frame SED at a
    phase and the FLUXCAL that SED makes through a set of bands, and moves
    its own copy of the distance where it keeps one.

    A model holding tabulated surfaces defines ``resample_to_grid(grid)``
    as well, which the constructor calls last with the grid
    :meth:`choose_grid` returns: it moves every surface the model loaded onto
    that grid, with :func:`resample_surfaces`, and stores the result as the
    JAX arrays :meth:`generate_sed` reads, so that the surfaces and the
    array the SED is returned on run over the same wavelengths.
    :meth:`SIMSEDModel.resample_to_grid
    <jax_snlc_core.models.simsed.SIMSEDModel.resample_to_grid>` is the
    form the bundled tabulated models take. A model whose spectrum is an
    expression evaluated at each point of the grid needs no such method.
    """

    # the directory-name prefixes that select this model (snlc_sim.c GENMODEL_NAME)
    GENMODEL_ALIASES: ClassVar[Tuple[str, ...]] = ()

    # Preparation state, kept outside compiled numerical calls; dataclass
    # models inherit these defaults.
    config: dict = field(init=False, default_factory=dict, repr=False)
    n_wave_points: int = field(init=False, default=500)
    wavelength_range: object = field(init=False, default=None)

    @classmethod
    def load(cls, model_path, config, n_wave_points=500,
             wavelength_range=None):
        """Load a model by calling its constructor by default.

        A model with a data-only constructor can override this method to read
        and prepare its inputs before constructing the model.
        """
        return cls(model_path, config, n_wave_points, wavelength_range)

    def __init__(self, model_name: str, config: Dict[str, Any],
                 n_wave_points: int = 500, wavelength_range=None):
        """Take the input before loading anything, so a model is fully built
        when its constructor returns.

        A subclass calls this first. A tabulated model then loads its
        surfaces, extends them into the UV (:attr:`uvlam_extrapflux`), and
        ends with ``self.resample_to_grid(self.choose_grid())``, so the
        working grid is set when the constructor returns; a model that
        synthesises its SED on demand calls ``self.choose_grid()`` instead.
        ``wavelength_range``
        is the rest-frame interval the caller will evaluate, which the grid
        is cut to; None takes the whole support.
        """
        self.model_name = model_name
        self.config = config
        self.n_wave_points = n_wave_points
        self.wavelength_range = wavelength_range
        keys = model_input_keys(type(self), consumer=None)
        for key, value in config.items():
            if key in keys:
                setattr(self, keys[key].attribute, value)
        # SNANA extends the SED only for a positive value (genmag_SALT2.c:321)
        if self.uvlam_extrapflux is not None and self.uvlam_extrapflux <= 0.0:
            self.uvlam_extrapflux = None

    #: Host-galaxy extinction law, independent of OPT_MWCOLORLAW. Uses the
    #: registered dust-law name; Fitzpatrick (1999) is the default.
    host_colorlaw: str = field(default='FITZ99_EXACT', metadata={'static': True})

    def build_host_dust_law(self):
        """The configured host law, prepared on this model's wavelength grid."""
        law = load_dust_law(self.host_colorlaw)
        law.prepare_wavelength_grid(np.asarray(self.wavelength_grid))
        return law

    def choose_grid(self):
        """The working wavelength grid: the model's support, cut to the
        caller's range where one was given, at the caller's resolution; read
        after any UV extension so the extension is never resampled away. Cuts
        that interval into ``n_wave_points`` evenly spaced points, stores them
        as ``self.wavelength_grid`` and returns them."""
        lo, hi = self.get_wavelength_range()
        if self.wavelength_range is not None:
            lo, hi = max(lo, self.wavelength_range[0]), min(hi, self.wavelength_range[1])
        self.wavelength_grid = jnp.linspace(lo, hi, self.n_wave_points)
        return self.wavelength_grid

    @abstractmethod
    def get_wavelength_range(self) -> Tuple[float, float]:
        """The model's own rest-frame support, as (min_wavelength,
        max_wavelength) in Angstroms: the extent the working grid is cut
        from, not the grid. A tabulated model reports its templates' range,
        UV extension included."""

    @abstractmethod
    def sed_param_names(self):
        """Per-event keys :meth:`generate_sed` may read from its params dict.

        A superset is allowed (a model with optional inputs declares them
        all); callers pass exactly the declared keys present in the event
        arrays, so simulator bookkeeping fields can never leak into traced
        model inputs.
        """

    @abstractmethod
    def generate_sed(self, phase: float, params: Dict[str, float],
                     static_data: Dict) -> jnp.ndarray:
        """The rest-frame SED at ``phase`` (days relative to the model’s
        phase origin), before
        dust extinction and cosmological effects, which the caller applies
        afterwards.

        ``phase`` is a scalar, and the return is the flux density in
        erg/s/cm²/Å at each point of the working grid, shape
        ``(n_wavelengths,)``; the caller maps this over the epochs of an
        event. ``params`` is the model's own parameters under the names
        :meth:`sed_param_names` declares (for example,
        ``{'width': 12.0, 'mu': 35.0}``), each a scalar for one event.
        ``static_data`` is what the
        simulator and the fitter assemble once per run for every model: the
        ``wavelength_grid`` returned by :meth:`choose_grid` and the
        calibration and extinction entries :meth:`evaluate` reads, among them
        the model's own :attr:`host_rv` and :attr:`fluxcal_scale_mag_offset`
        declarations. It holds none of the model's templates or knots, which
        are attributes set in the constructor and read here through
        ``self``.

        Use JAX-compatible numerical operations. Python control flow must
        not depend on the value of a traced parameter.
        """

    @abstractmethod
    def phase_range(self) -> Tuple[float, float]:
        """The rest-frame phase support, as (daymin, daymax) in days.

        Bounds the epochs at which true-SED spectra are generated. Supply
        this range even if ``generate_sed`` already returns zero or holds the
        spectrum fixed outside it, so callers can select covered epochs.
        """

    #: Extend the model's tabulated SED down to this rest-frame
    #: wavelength in Angstroms, ramping the flux linearly to zero at the new
    #: blue edge. Unset leaves the SED unchanged, as does a value of zero or
    #: below, which SNANA's inputs write for off. A model with no tabulated
    #: SED ignores the key.
    uvlam_extrapflux: Optional[float] = None

    def evaluate(self, sed_params, mjds, filter_indices, mask, band_weights,
                 static_data, sed_factor=None, band_weights_dz=None,
                 band_weights_dz2=None):
        """FLUXCAL for one SN's observations: this model's SED through the
        band weights.

        ``sed_params`` is the model's own parameters, plus ``t0`` and
        ``redshift``. ``band_weights`` is [n_filters, n_wavelengths] at this
        SN's redshift. ``sed_factor`` is a rest-frame per-wavelength
        multiplier on the source that is not the model's own physics, as the
        simulator's GENMAG_SMEAR_MODELNAME is. ``band_weights_dz`` is
        d(band_weights)/dz, supplied when the redshift is fitted: a model
        differentiated by AD ignores it, one with analytic derivatives uses
        it, because the weights' own tangent varies per tangent direction
        while this does not. ``band_weights_dz2``, the second derivative,
        serves an analytic rule whose redshift terms are differentiated again.
        """
        t0, redshift = sed_params['t0'], sed_params['redshift']
        phases = jnp.where(mask, (mjds - t0) / (1.0 + redshift), 0.0)

        if 'av' in sed_params:
            ext = static_data['host_dust_law'].apply_extinction(
                jnp.ones(band_weights.shape[-1]),
                static_data['wavelength_grid'], sed_params['av'],
                (sed_params['rv'] if 'rv' in sed_params
                            else static_data['host_rv']))
            sed_factor = ext if sed_factor is None else sed_factor * ext

        model_params = {k: sed_params[k] for k in self.sed_param_names()
                        if k in sed_params}
        return fluxcal_from_sed(
            lambda phase: self.generate_sed(phase, model_params, static_data),
            phases, mask, filter_indices, band_weights,
            static_data['filter_ref_fluxes'], static_data['zp_fluxcal'],
            static_data['fluxcal_scale_mag_offset'],
            width=static_data['band_window'], sed_factor=sed_factor)

    #: The name of the model's own distance parameter, where it keeps one:
    #: the parameter :meth:`shift_amplitude` and :meth:`params_at_distance`
    #: move, which the simulator's host overrides and the fitter's starting
    #: values both go through. None means the brightness is the simulator's
    #: mu, which the caller has already updated, so there is nothing of the
    #: model's to move.
    distance_param = None

    def _amplitude_shifted(self, value, dmu):
        """The distance parameter, in physical units, moved ``dmu`` magnitudes
        fainter. A magnitude adds; SALT overrides, since x0 is not one."""
        return value + dmu

    def amplitude_jacobian(self, value):
        """d(magnitude)/d(amplitude), for a covariance carried into magnitude
        space. The reciprocal of how far the amplitude moves per magnitude,
        which :meth:`_amplitude_shifted` already defines; a model whose
        amplitude is itself a magnitude gets 1.
        """
        _, per_mag = jax.jvp(lambda d: self._amplitude_shifted(value, d),
                             (0.0,), (1.0,))
        return 1.0 / np.asarray(per_mag)

    def _amplitude_at_distance(self, mu, params):
        """The amplitude, in physical units, of a source at distance modulus
        ``mu``. ``params`` carries the model's other parameters, physical, for
        a model whose brightness depends on them. Default: the amplitude is the
        distance modulus."""
        return mu

    def _distance_at_amplitude(self, value, params):
        """The distance modulus of a source with amplitude ``value``, in
        physical units: the inverse of :meth:`_amplitude_at_distance`, with
        ``params`` as there. Default: the amplitude is the distance modulus."""
        return value

    def distance_of(self, physical, names):
        """The distance modulus the fit vector ``physical``, in physical units
        and laid out as ``names``, places the source at: the inverse of
        :meth:`params_at_distance`."""
        params = {n: physical[..., k] for k, n in enumerate(names)}
        return self._distance_at_amplitude(params[self.distance_param], params)

    def shift_amplitude(self, params, dmu, scales=1.0, names=None):
        """``params`` with the amplitude moved ``dmu`` magnitudes fainter.

        Callers give the shift in magnitudes and need not know how the model
        encodes the amplitude. Accepts either a name-keyed dict of physical
        values or a fit-vector array, whose layout the caller gives as
        ``names`` and ``scales`` (the latter converts to physical and back).
        The simulator's HOSTLIB overrides and the fitter's flux-ratio steps
        are the same operation and both use this.
        """
        return self._map_amplitude(
            params, lambda value, _: self._amplitude_shifted(value, dmu),
            scales, names)

    def params_at_distance(self, params, mu, scales=1.0, names=None):
        """``params`` with the brightness set to that of a source at ``mu``.

        The absolute counterpart of :meth:`shift_amplitude`, for seeding a fit
        from cosmology.

        For a relative change in brightness, use :meth:`shift_amplitude`.
        Converting a simulated SALT amplitude back to distance requires the
        event's alpha and beta, which may differ from the model-level values.
        A relative shift cancels those terms and applies to either case.
        """
        return self._map_amplitude(
            params, lambda _, other: self._amplitude_at_distance(mu, other),
            scales, names)

    def _map_amplitude(self, params, fn, scales=1.0, names=None):
        """Apply ``fn(amplitude, other_params)`` in physical units, to a
        name-keyed dict or to the fitter's vector. ``other_params`` is
        unscaled too: a model whose brightness depends on its other parameters
        must see them physical, and SALT's x1 and c both carry scale 1.0, so
        relying on that would work by accident.
        """
        name = self.distance_param
        if name is None:
            return params
        if isinstance(params, dict):
            return {**params, name: fn(params[name], params)}
        i = names.index(name)
        scales = jnp.broadcast_to(
            jnp.asarray(scales, params.dtype), params.shape)
        physical = params * scales
        other = {n: physical[..., k] for k, n in enumerate(names)}
        return params.at[..., i].set(
            fn(physical[..., i], other) / scales[..., i])

    #: The host total-to-selective extinction ratio :meth:`evaluate` applies
    #: when the event carries ``av`` but no ``rv``, which is the case for a
    #: model fitting the extinction without its law; it reaches
    #: ``generate_sed`` as ``static_data['host_rv']``. None for a model with
    #: no host extinction of its own.
    host_rv = None

    #: A flux factor applied to every band integral, standing for a magnitude
    #: offset the model's training wrote into its files; ``fluxcal_from_sed``
    #: reads it from ``static_data`` and the default 1.0 is no offset.
    fluxcal_scale_mag_offset = 1.0

    #: The rest-frame wavelengths, as ``(minimum, maximum)`` in Angstroms,
    #: between which a band's central wavelength must fall for the simulator
    #: to generate that band at an event's redshift (SNANA's
    #: RESTLAMMIN_FILTERCEN and RESTLAMMAX_FILTERCEN). None uses the range
    #: :meth:`get_wavelength_range` declares.
    filter_center_range = None

    #: How the model continues past the last phase it covers, in SNANA's
    #: numbering: 1 (SEDFLUX) means the simulator writes no SED-true spectrum
    #: past that phase, 3 (FLAM) means it does. SALT sets this per instance
    #: from its own extrapolation configuration; a model that extrapolates no
    #: further than its support leaves the default.
    extrap_phase_method = 1

    def __repr__(self) -> str:
        """String representation of model."""
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"
