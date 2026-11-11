"""
Wrong-host contamination for JAX-SNANA.

Implements SNANA's gen_zHOST() logic: with some probability, assigns a SN
to a host at a different redshift (wrong host association).

The wrong-host file contains:
- PROB_WRONGHOST_POLY: polynomial coefficients for P(wrong host) vs z
- Two-column ZTRUE/ZMATCH table for redshift remapping

Key: this runs BEFORE host selection. It modifies the redshift used
for host lookup, so a "wrong host" SN gets assigned a host at the
wrong redshift.

Reference: SNANA sntools_wronghost.c:213 (gen_zHOST)
"""

import logging
from typing import NamedTuple, Tuple

import jax.numpy as jnp
import numpy as np

from jax_snlc_core import logging as log
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import leading_values, line_words, open_text, skip_docana
from jax_snlc_core.utils import eval_genpoly

logger = logging.getLogger(__name__)


DZTRUEMAX = 0.01  # DZTRUEMAX_WRONGHOST, sntools_wronghost.h
ORDER_PROB_WRONGHOST_POLY = 4  # the older space-separated cubic, sntools_wronghost.h:3


class WrongHostTable(NamedTuple):
    """A WRONGHOST_FILE: the wrong-host probability polynomial in z and the
    ZTRUE/ZMATCH table, ZTRUE-sorted and cut to the generated range."""
    poly_coeffs: np.ndarray
    ztrue: np.ndarray
    zmatch: np.ndarray
    z_range: Tuple[float, float]    # GENRANGE_REDSHIFT


def read_wronghost_file(path: str, zmin: float, zmax: float) -> WrongHostTable:
    """Parse a WRONGHOST_FILE (init_WRONGHOST, sntools_wronghost.c:85-145).

    Format:
        # PROB_WRONGHOST_POLY: c0,c1,c2,...    P(z) = c0 + c1*z + c2*z^2 + ...
        # ZTRUE  ZMATCH
        0.05   0.08
        0.10   0.12
        ...

    The key is read wherever it stands, even on a comment line, where
    SNANA's own files keep it; its value is one comma-separated list, or
    SNANA's older four space-separated cubic coefficients, and words after
    it are passed over (sntools_wronghost.c:87-101). Every other line
    outside comments is a ZTRUE ZMATCH pair. ``path`` is the wrong-host
    file; ``zmin``/``zmax`` are the SN redshift range (from
    GENRANGE_REDSHIFT). Returns the :class:`WrongHostTable`.
    """
    poly_coeffs = None
    rows = []
    key = 'PROB_WRONGHOST_POLY:'
    with open_text(path) as f:
        for line in skip_docana(f):
            words = line.split()
            if key in words:
                if poly_coeffs is not None:
                    raise SNANAError(f"{path} has more than one {key}",
                                     function_name="read_wronghost_file")
                values = line_words(' '.join(words[words.index(key) + 1:]))
                first = leading_values(path, key, values, 1)[0]
                coeffs = (first.split(',') if ',' in first
                          else leading_values(path, key, values, ORDER_PROB_WRONGHOST_POLY))
                try:
                    poly_coeffs = np.array([float(c) for c in coeffs])
                except ValueError:
                    raise SNANAError(f"{path}: {key} {' '.join(values)} is not a list of "
                                     f"numbers", function_name="read_wronghost_file") from None
            elif words := line_words(line):
                try:
                    rows.append([float(z) for z in leading_values(path, 'ZTRUE ZMATCH', words, 2)])
                except ValueError:
                    raise SNANAError(f"{path}: '{line.strip()}' is not a ZTRUE ZMATCH pair",
                                     function_name="read_wronghost_file") from None

    if poly_coeffs is None:
        raise SNANAError(f"PROB_WRONGHOST_POLY not found in {path}")

    ztrue, zmatch = np.array(rows, dtype=np.float64).reshape(-1, 2).T

    sort_idx = np.argsort(ztrue)
    ztrue = ztrue[sort_idx]
    zmatch = zmatch[sort_idx]

    mask = (ztrue >= zmin) & (ztrue <= zmax) & (zmatch >= zmin) & (zmatch <= zmax)
    ztrue = ztrue[mask]
    zmatch = zmatch[mask]
    if not len(ztrue):
        raise SNANAError(
            f"WRONGHOST_FILE {path} has no ZTRUE/ZMATCH row with both redshifts "
            f"inside [{zmin}, {zmax}], so no wrong host can be drawn.",
            function_name="read_wronghost_file")

    log.line(logger, f"WRONGHOST_FILE: {path}")
    log.line(logger, f"Polynomial coefficients: {poly_coeffs}", indent=2)
    log.line(logger, f"Table entries: {len(ztrue)} (in z=[{zmin:.3f}, {zmax:.3f}])", indent=2)

    return WrongHostTable(poly_coeffs=poly_coeffs, ztrue=ztrue, zmatch=zmatch,
                          z_range=(zmin, zmax))


def gen_zhost(z_sn, random_prob, random_zmatch, table) -> tuple:
    """Generate host redshifts with possible wrong-host contamination.

    Always consumes 2 random numbers per SN (matching SNANA sync).

    ``z_sn`` is (n_sne,), the SN redshifts; ``random_prob``/
    ``random_zmatch`` are (n_sne,) uniform randoms [0,1) for the
    wrong-host decision and the ZMATCH table lookup; ``table`` is the
    :class:`WrongHostTable`.

    Returns ``(z_host, host_match, uncovered)``, each (n_sne,): the host
    redshift for lookup, whether the match is correct, and whether a wrong
    host found no table row within DZTRUEMAX, which SNANA aborts on
    (sntools_wronghost.c:324-329); such a lookup stays in bounds.
    """
    ztrue = jnp.asarray(table.ztrue, dtype=z_sn.dtype)
    zmatch = jnp.asarray(table.zmatch, dtype=z_sn.dtype)
    p_wrong = jnp.clip(eval_genpoly(table.poly_coeffs, z_sn), 0.0, 1.0)
    is_wrong = random_prob < p_wrong

    # strict |ZTRUE - zSN| < DZTRUEMAX (sntools_wronghost.c:317)
    idx_lo = jnp.searchsorted(ztrue, z_sn - DZTRUEMAX, side='right')
    idx_hi = jnp.searchsorted(ztrue, z_sn + DZTRUEMAX, side='left')
    n_near = idx_hi - idx_lo
    random_idx = idx_lo + (random_zmatch * n_near).astype(idx_lo.dtype)
    random_idx = jnp.clip(random_idx, idx_lo, jnp.maximum(idx_hi - 1, 0))
    safe_idx = jnp.clip(random_idx, 0, ztrue.shape[0] - 1)
    z_remapped = z_sn + (zmatch[safe_idx] - ztrue[safe_idx])

    # clamp only the wrong-host branch (sntools_wronghost.c:344-346)
    zmin, zmax = table.z_range
    z_remapped = jnp.where(z_remapped >= zmax, zmax - 0.001, z_remapped)
    z_remapped = jnp.where(z_remapped <= zmin, zmin + 0.001, z_remapped)

    return (jnp.where(is_wrong, z_remapped, z_sn), ~is_wrong,
            is_wrong & (n_near <= 0))
