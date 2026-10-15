"""Shared CUTWIN range evaluation: SNANA's inclusive lo <= value <= hi
(snana.F90 LCUTVAR; snlc_sim.c cut loops), and the per-event quantities the
cuts test over a padded light curve. Works on numpy arrays and on traced jax
arrays inside a jit.
"""

from typing import Dict, Tuple

import numpy as np

from jax_snlc_core.utils.error_handling import SNANAError


def in_window(values, lo: float, hi: float):
    return (values >= lo) & (values <= hi)


# A cut edge this wide accepts everything (CUTVAL_OPEN, snana.F90:203)
CUTVAL_OPEN = 1.0e12


def range_cut_masks(windows: Dict[str, Tuple[float, float]],
                    quantities: Dict):
    """``(name, mask)`` for each of the named inclusive [lo, hi] ``windows``.

    ``quantities`` maps each name to an (n_sne,) array, numpy or jax, and the
    masks come back in the same type. A window with no quantity is a window
    the run would silently not apply, so it is an error.
    """
    for name, (lo, hi) in windows.items():
        if name not in quantities:
            raise SNANAError(
                f"CUTWIN '{name}' does not match any selection quantity; "
                f"known: {sorted(quantities)}",
                function_name='apply_range_cuts')
        yield name, in_window(quantities[name], lo, hi)


def apply_range_cuts(windows: Dict[str, Tuple[float, float]],
                     quantities: Dict):
    """The ``windows`` combined into one per-SN mask."""
    ref = next(iter(quantities.values()))
    mask = ref == ref  # all-True, in the caller's array type
    for _, window_mask in range_cut_masks(windows, quantities):
        mask = mask & window_mask
    return mask


def narrow(keep, name: str, mask, ladder):
    """``keep & mask``, recording the survivors under ``name``.

    Every cut narrows the sample through here, so a cut cannot go unaccounted
    and the last entry is the number written (NPASSCUT_INCREMENT,
    snana.F90:8937).
    """
    keep = keep & mask
    ladder.append((name, int(np.count_nonzero(keep))))
    return keep


def band_maxima(values, band_index, accepted, n_bands):
    """Per-band maxima of ``values`` over the ``accepted`` epochs, -9 for a
    band without one, and whether each band has an epoch."""
    band_max = np.full((values.shape[0], n_bands), -9.0)
    has_epoch = np.zeros((values.shape[0], n_bands), dtype=bool)
    for b in range(n_bands):
        in_band = accepted & (band_index == b)
        band_max[:, b] = np.where(in_band, values, -9.0).max(axis=1)
        has_epoch[:, b] = in_band.any(axis=1)
    return band_max, has_epoch


def gap_maxima(times, accepted, t0_lo, t0_hi):
    """Largest gap between consecutive accepted epochs, and the largest one
    touching the peak window (snana.F90:20851, :20912; snlc_sim.c:24040-24070).

    The window is bounded by where the TRESTMIN and TRESTMAX cuts meet, so
    left open it inverts and every gap qualifies. Both start at -999 and only
    a real gap moves them (snana.F90:20835).
    """
    ordered = np.sort(np.where(accepted, times, np.inf), axis=1)
    with np.errstate(invalid='ignore'):     # inf-inf on padding, masked below
        gaps = np.diff(ordered, axis=1)
    real = np.isfinite(gaps)
    before, after = ordered[:, :-1], ordered[:, 1:]
    touches = (((after > t0_lo) & (after < t0_hi))
               | ((before > t0_lo) & (before < t0_hi))
               | ((after > t0_hi) & (before < t0_lo)))
    # initial: an epoch axis of length one leaves no gaps to reduce over
    return (np.where(real, gaps, -999.0).max(axis=1, initial=-999.0),
            np.where(real & touches, gaps, -999.0).max(axis=1, initial=-999.0))
