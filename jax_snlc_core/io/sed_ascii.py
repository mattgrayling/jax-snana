"""Format-generic 3-column ASCII SED reader, SNANA's rd_sedFlux.

Shared by the NON1ASED, SIMSED and SALT models: each loader calls
`read_sed_file` to parse ``.SED``/``.SED.gz`` templates or SALT surfaces
and error maps, which are plain 3-column (phase, wavelength, value) ASCII
files regardless of which model references them.
"""

import logging
from typing import Tuple

import numpy as np

from jax_snlc_core.io.fits_path import open_text
from jax_snlc_core.utils import snana_log as slog
from jax_snlc_core.utils.error_handling import SNANAError

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
    (``.SED`` or ``.SED.gz``), handling gzipped files transparently and
    skipping blank lines and comment lines (first character ``#``, ``!``,
    ``%`` or ``@``). A data line with fewer than three words, or whose first
    three words are not numbers, is an error.

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

    with open_text(filepath) as f:
        lines = f.readlines()

    data_rows = []
    for number, line in enumerate(lines, start=1):
        if not line.strip() or line[0] in '#!%@':
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
            function_name="rd_sedFlux")
    # every day has as many wavelengths as the first (:8675-8690)
    phases, per_day = np.unique(raw_phases, return_counts=True)
    if np.any(per_day != per_day[0]):
        k = int(np.argmax(per_day != per_day[0]))
        raise SNANAError(
            f"{filepath}: {per_day[k]} wavelengths at day {phases[k]:.2f} but "
            f"{per_day[0]} at day {phases[0]:.2f}; the wavelength grid must "
            f"be the same for each day", function_name="rd_sedFlux")
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
            function_name="rd_sedFlux")
    mismatched = np.argwhere(lam_grid != wavelengths)
    if mismatched.size:
        k, j = mismatched[0]
        raise SNANAError(
            f"{filepath}: found LAM={lam_grid[k, j]:.1f} at day "
            f"{phases[k]:f} but expected LAM={wavelengths[j]:.2f}",
            function_name="rd_sedFlux")
    # uniform day steps unless the model allows otherwise (:8549-8560)
    day_step = np.diff(phases)
    step_change = np.abs(np.diff(day_step)) > SED_DAYSTEP_TOL
    if uniform_days and step_change.any():
        k = int(np.argmax(step_change)) + 1
        raise SNANAError(
            f"{filepath}: day step {day_step[k]:e} after {day_step[k - 1]:e} "
            f"at day {phases[k + 1]:.4f}; the day bins must be uniform",
            function_name="rd_sedFlux")
    # a non-uniform wavelength step only warns (:8595-8606)
    if np.any(np.abs(np.diff(np.diff(wavelengths))) > SED_LAMSTEP_TOL):
        slog.warn(logger, 'SED', f"{filepath}: non-uniform wavelength step "
                  f"(rd_sedFlux warns and reads on)")

    # phase varies slowest, wavelength fastest
    flux_grid = raw_flux.reshape(n_phases, n_wavelengths)

    # T0shiftPeak_SEDMODEL: peak bolometric flux at day 0
    if t0shift_peakmag:
        bolometric = flux_grid.sum(axis=1)
        peak_idx = np.argmax(bolometric)
        phases = phases - phases[peak_idx]

    return phases, wavelengths, flux_grid
