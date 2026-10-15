"""SED extrapolation utilities shared across models: fireball-rise /
power-law-tail flux extrapolation (:func:`modelflux_extrap`, matching SNANA
genmag_extrap.c:modelflux_extrap) and the late-time Ia double-exponential
decay model (:class:`LatetimeExtrapIa`, matching SNANA genmag_extrap.c
init_extrap_latetime_Ia + genflux_extrap_latetime_Ia).
"""

import logging
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from jax_snlc_core.utils.error_handling import SNANAError
from jax_snlc_core.utils import snana_log as slog

logger = logging.getLogger(__name__)


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

        slog.line(logger, f"Late-time Ia extrap: DAYMIN={daymin:.0f}, "
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
