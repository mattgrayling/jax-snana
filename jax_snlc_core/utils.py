"""General helpers: the precision a run works in, physical constants,
grouping equal settings, polynomial evaluation, CUTWIN range cuts and covariance repair.
"""

import logging
import math
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core.logging import SNANAError

logger = logging.getLogger(__name__)

#: The precision a command-line run uses when its input omits the key.
DEFAULT_DOUBLE_PRECISION = False


def configure_precision(double: bool | None) -> None:
    """Run in double precision, or in single.

    Every input carries one boolean key, ``DOUBLE_PRECISION``, spelt the way
    that input spells its other logicals: ``DOUBLE_PRECISION: 1`` in a sim
    input, ``DOUBLE_PRECISION = T`` in a fitter namelist,
    ``double_precision=1`` for SALT2mu, ``DOUBLE_PRECISION: true`` in a
    create_covariance YAML, and ``-double_precision 1`` on wfit's command
    line. An omitted key is recorded as None; ``double`` None takes
    :data:`DEFAULT_DOUBLE_PRECISION`.

    This is the only place ``jax_enable_x64`` is set. Each stage's ``main``
    calls it once its input is parsed and before the first array is built;
    the parsers record the key and touch nothing global, so parsing an input
    inside another process leaves that process's precision alone, and
    importing the packages sets nothing.

    The precision is the width of JAX's default float: unannotated array
    constructors, reductions, and the uniform and normal draws that carry the
    per-event random paths. It does not move the explicit ``float32`` dtypes
    in the simulator, which match the single-precision storage SNANA reads
    and writes (SIMLIB and HOSTLIB columns, the TABLEVAR truncation in
    :mod:`jax_salt2mu.tablevar`, the HEAD and PHOT output tables). Code that
    wants the chosen width asks :func:`float_dtype` for it.
    """
    if double is None:
        double = DEFAULT_DOUBLE_PRECISION
    jax.config.update("jax_enable_x64", double)
    if double:
        logger.info("Precision: double (float64)")
    else:
        logger.info("Precision: single (float32); DOUBLE_PRECISION selects double")


def float_dtype():
    """The float dtype of the chosen frame."""
    return jnp.result_type(float)


def require_precision(double: bool | None) -> None:
    """Check an explicit request without changing a Python process's precision."""
    if double is not None and double != jax.config.jax_enable_x64:
        raise SNANAError(
            'double_precision conflicts with the active JAX precision. '
            'Call configure_precision before preparing arrays or models, '
            'or omit double_precision to use the active precision.')


# Physical constants, in SNANA's conventions
LIGHT_A = 2.99792458e18   # Angstroms per second
LIGHT_KM = 2.99792458e5   # km per second (SNANA LIGHT_km)
PLANCK = 6.6260755e-27    # erg s
HC_ERG_A = LIGHT_A * PLANCK
BOLTZMANN_ERG_K = 1.380649e-16   # erg per kelvin (CODATA 2018)

LN10 = math.log(10.0)
# 10^(-0.4*x) = exp(NEG_0_4_LN10 * x)
NEG_0_4_LN10 = -0.4 * LN10


def group_equal(mapping):
    """The keys of ``mapping`` grouped by equal value, each group in the
    order of its first key. Values are compared rather than hashed, so they
    may be dicts or dataclasses holding them."""
    groups = []
    for key, value in mapping.items():
        group = next((keys for v, keys in groups if v == value), None)
        if group is None:
            groups.append((value, [key]))
        else:
            group.append(key)
    return [keys for _, keys in groups]


def in_window(values, lo: float, hi: float):
    """``lo <= values <= hi``, SNANA's inclusive CUTWIN test (snana.F90
    LCUTVAR; snlc_sim.c cut loops), for numpy or traced jax arrays."""
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
                function_name='range_cut_masks')
        yield name, in_window(quantities[name], lo, hi)


def apply_range_cuts(windows: Dict[str, Tuple[float, float]],
                     quantities: Dict):
    """The ``windows`` combined into one per-SN mask."""
    ref = next(iter(quantities.values()))
    mask = np.ones(ref.shape, dtype=bool)
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


# SALT2mu.c:577 and snlc_fit.F90's FIX_COVAR_LCFIT both use this floor.
EIGMIN_COV = 1.0e-4

# a FITRES covariance column carries -9 when the fit left it undefined
UNDEFINED_COV = -9.0


def fix_covariance(cov: np.ndarray, force: bool = False) -> np.ndarray:
    """Eigen-clip non-positive-definite matrices, as SNANA's
    ``update_covMatrix`` (sntools.c:2485) does for both snlc_fit and SALT2mu.

    ``cov`` is an ``[n, k, k]`` stack of symmetric matrices. Each element equal
    to the undefined sentinel -9 is set to zero and that matrix is returned
    unrepaired, as ``update_covMatrix`` returns istat_cov = -9 before reaching
    the eigen decomposition (sntools.c:2537-2548). Of the rest, matrices with a
    non-positive eigenvalue are rebuilt from a spectrum clipped at
    :data:`EIGMIN_COV`; ``force`` rebuilds every matrix, which is what SNANA's
    ``OPT_COVAR_LCFIT`` FIXCOV_RED bit asks for.
    """
    cov = np.asarray(cov, dtype=float)
    undefined = np.abs(cov - UNDEFINED_COV) < 1.0e-6
    cov = np.where(undefined, 0.0, cov)
    leave_alone = undefined.any(axis=(1, 2))

    w, V = np.linalg.eigh(cov)
    repair = np.ones(cov.shape[0], dtype=bool) if force else w.min(axis=1) <= 0.0
    repair &= ~leave_alone
    # rebuild only the matrices that need it; most FITRES covariances are valid
    if np.any(repair):
        vectors = V[repair]
        clipped = np.clip(w[repair], EIGMIN_COV, None)
        cov[repair] = vectors @ (clipped[:, :, None] * np.transpose(vectors, (0, 2, 1)))
    return cov


def eval_genpoly(coeffs, x):
    """SNANA's eval_GENPOLY: c0 + c1*x + c2*x^2 + ... (ascending coeffs).

    Horner form over plain arithmetic, so it evaluates NumPy float64 and
    traced JAX values alike, preserving the caller's dtype. ``coeffs`` needs
    a static length (a sequence, or an array iterated at trace time).
    """
    out = None
    for c in reversed(list(coeffs)):
        out = c if out is None else out * x + c
    return x * 0 if out is None else out + x * 0


def eval_genpoly_rows(coeffs, x):
    """eval_GENPOLY with one coefficient row per element of ``x``.

    ``coeffs`` is (n, n_coeffs) ascending per row (zero-padded rows are
    exact, since zero coefficients contribute nothing); ``x`` is (n,).
    """
    out = None
    n_coeffs = coeffs.shape[1]
    for i in range(n_coeffs - 1, -1, -1):
        c = coeffs[:, i]
        out = c if out is None else out * x + c
    return x * 0 if out is None else out
