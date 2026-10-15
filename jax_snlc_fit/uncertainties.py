"""Fit errors and covariances: :func:`errors_from_chi2` takes them from
the chi2 a fit minimised, as MINUIT's are from its FCN, and
:func:`fix_covar_lcfit` repairs the reported covariance afterwards.
"""

import dataclasses
import logging

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core.utils import snana_log as slog
from jax_snlc_core.utils.covariance import fix_covariance

logger = logging.getLogger(__name__)


def errors_from_chi2(chi2, p_scaled, sn_param_scales, free):
    """One SN's errors, covariance, Hessian condition number and EDM of
    ``chi2`` at ``p_scaled``, over the ``free`` parameters, and whether the
    unregularised free Hessian is positive definite.

    The covariance is the inverse of half the Hessian on the free subspace,
    which is the likelihood curvature MINUIT reports; the ridge keeps a
    singular Hessian invertible, and EDM is the same quadratic form on the
    free gradient. A held parameter's rows and columns are zero.
    """
    def half_grad(q):
        g = 0.5 * jax.grad(chi2)(q)
        return g, g

    H, g = jax.jacfwd(half_grad, has_aux=True)(p_scaled)
    return errors_from_derivatives(H[free][:, free], g[free], sn_param_scales, free)


def errors_from_derivatives(H_free, g_free, sn_param_scales, free):
    """Errors from the half-Hessian and half-gradient on the free subspace.

    Profiling already evaluates these derivatives to check its minimum. Reuse
    that evaluation for the covariance, including zero rows for held parameters.
    """
    cov_free = jnp.linalg.inv(H_free + 1e-12 * jnp.eye(H_free.shape[0]))
    size = sn_param_scales.size
    cov_scaled = jnp.zeros((size, size), dtype=H_free.dtype).at[
        free[:, None], free[None, :]].set(cov_free)
    # p_phys = p_scaled * scales, so Cov(phys) = S Cov(scaled) S^T
    cov_phys = cov_scaled * jnp.outer(sn_param_scales, sn_param_scales)
    # LFIXPAR_ALL leaves no free direction
    if H_free.shape[0] == 0:
        cond = jnp.ones((), dtype=H_free.dtype)
        positive = jnp.asarray(True)
    else:
        eigenvalues = jnp.linalg.eigvalsh(H_free)
        positive = jnp.all(jnp.isfinite(eigenvalues) & (eigenvalues > 0))
        ev = jnp.abs(eigenvalues)
        cond = jnp.max(ev) / jnp.maximum(jnp.min(ev), jnp.finfo(ev.dtype).tiny)
    return (jnp.sqrt(jnp.abs(jnp.diag(cov_phys))), cov_phys,
            cond, g_free @ cov_free @ g_free, positive)


def fix_covar_lcfit(results, model, opt):
    """OPT_COVAR_LCFIT: repair the output COV(amplitude, shape, colour).

    FIX_COVAR_LCFIT (snlc_fit.F90:2587) floors the eigenvalues of the 3x3 in
    magnitude space, where the amplitude axis carries a Jacobian the model
    supplies, then divides that back out. Bit 1 rebuilds every matrix. A model
    that declares no such basis is left alone.
    """
    specs = model.get_fit_parameters()
    cov_names = list(specs)
    repaired = [n for n in cov_names
                if n != 'PKMJD' and not specs[n].get('deferred')]
    if model.distance_param not in repaired:
        return results
    idx = [cov_names.index(r) for r in repaired]
    n_par = len(repaired)
    amp_at = repaired.index(model.distance_param)

    amp = np.asarray(results.params[model.distance_param])
    jac = np.ones((len(amp), n_par))
    jac[:, amp_at] = model.amplitude_jacobian(amp)

    cov = np.array(results.param_covariances)
    sub = cov[np.ix_(range(cov.shape[0]), idx, idx)] * jac[:, :, None] * jac[:, None, :]
    # SNANA takes the diagonal from the reported errors, not from FITERRMAT
    err = np.stack([np.asarray(results.param_errors[r]) for r in repaired],
                   axis=1)
    sub[:, range(n_par), range(n_par)] = (err * jac) ** 2

    upd = fix_covariance(sub, force=bool(opt & 2))
    changed = np.any(upd != sub, axis=(1, 2))

    new_errors = dict(results.param_errors)
    for k, name in enumerate(repaired):
        e = np.sqrt(np.abs(upd[:, k, k])) / np.abs(jac[:, k])
        new_errors[name] = np.where(
            changed, e, np.asarray(results.param_errors[name]))
    for a in range(n_par):
        for b in range(n_par):
            if a != b:
                v = upd[:, a, b] / (jac[:, a] * jac[:, b])
                cov[:, idx[a], idx[b]] = np.where(changed, v, cov[:, idx[a], idx[b]])

    n_fixed = int(np.sum(changed))
    if n_fixed:
        slog.line(logger, f"OPT_COVAR_LCFIT={opt}: repaired COV for {n_fixed} of "
                          f"{len(amp)} events")
    return dataclasses.replace(results, param_errors=new_errors,
                               param_covariances=cov)
