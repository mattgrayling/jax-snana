"""Covariance repair shared by the light-curve fitter and BBC.

SNANA's ``update_covMatrix`` (sntools.c:2485) is called from both
``snlc_fit`` (via ``FIX_COVAR_LCFIT``) and ``SALT2mu``, so it lives here
rather than in either package.
"""

import numpy as np

# SALT2mu.c:577 and snlc_fit.F90's FIX_COVAR_LCFIT both use this floor.
EIGMIN_COV = 1.0e-4

# a FITRES covariance column carries -9 when the fit left it undefined
UNDEFINED_COV = -9.0


def fix_covariance(cov: np.ndarray, force: bool = False) -> np.ndarray:
    """Eigen-clip non-positive-definite matrices (``update_covMatrix``).

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
    # Preserve the eigensolver and repair decision, but only reconstruct the
    # matrices that need repair. Most FITRES covariances are already valid.
    if np.any(repair):
        vectors = V[repair]
        clipped = np.clip(w[repair], EIGMIN_COV, None)
        cov[repair] = vectors @ (clipped[:, :, None] * np.transpose(vectors, (0, 2, 1)))
    return cov
