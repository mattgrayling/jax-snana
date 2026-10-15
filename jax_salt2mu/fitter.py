"""BBC fit driver: inner Newton-LM fit + outer sigma_int loop.

``mode='joint'`` floats sigma_int with everything else under a proper Gaussian
likelihood (a JAX-native, non-SNANA mode). ``mode='snana_secant'`` reproduces
SNANA's outer secant loop on chi2_Ia/dof-1 with sigma_int held fixed inside each
inner fit. Both reuse the single likelihood and the shared Newton-LM kernel.
"""

from dataclasses import dataclass
from functools import partial
from typing import Dict, Optional, Tuple

import logging
import time

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core.utils import snana_log as slog
from jax_snlc_core.utils.error_handling import SNANAError
from jax_snlc_core.utils.newton_lm import newton_lm

from .derivatives import loss_derivatives
from .config import BBC_NUISANCE_SPEC
from .ccprior import CHI2_BEAMS_NOPROB
from .likelihood import (LikelihoodInputs, event_chi2, make_loss, mures_muerrsq_fn,
                         nuisance_at)

logger = logging.getLogger(__name__)


@dataclass
class BBCResult:
    nuisance: Dict[str, float]
    m0: np.ndarray
    m0err: np.ndarray
    sigint: float
    chi2: float          # -2 ln L at the minimum
    niter: int
    nuis_err: Dict[str, float]          # 1-sigma error per floated nuisance
    nsnfit: int                         # events entering the fit
    nfloat: int                         # floated nuisance params, including joint sigint
    cov_m0: np.ndarray                  # [nzbin, nzbin] M0 statistical covariance
    # seconds per step: fit_pad, fit_compile, fit_minimise, fit_stats, fit_hessian
    timings: Dict[str, float]
    # muerrsq_last after the final pass (:4864), for write_chi2grid
    muerrsq_final: np.ndarray
    # reference read by the final pass, for the per-event output
    muerrsq_last: Optional[np.ndarray]
    dchi2red_dsigint: float = 0.0       # the slope the sigint secant last used
    chi2sum_ia: float = 0.0             # pIa-weighted Ia chi2 (CHI2SUM_IA, :4292)
    ndof_ia: float = 0.0                # its dof, fractional under a CC prior
    fitflag_sigmb: int = 0              # the value the fit ran with (log term, :4702)
    max_gradient: float = float('nan')
    max_projected_gradient: float = float('nan')  # excludes outward slopes at bounds


def _fit_covariance(H, nf, nzbin, floated_bins, mode):
    """1-sigma errors and the M0 covariance block from the Hessian of -2lnL
    (cov = 2*inv(H)), restricted to the floated parameters so empty z-bins don't
    make H singular. Returns (nuis_err[nf], m0err[nzbin], cov_m0[nzbin, nzbin]).
    Unfloated M0 bins get a NaN error (never displayed) and SNANA's 1e5 COV
    diagonal (write_M0_cov). A floated parameter the likelihood does not depend
    on has an all-zero row; it gets a NaN error and the rest are inverted
    without it, as MINUIT carries on without a flat direction. Non-finite or
    non-positive-definite curvature on the remaining subspace is rejected."""
    H = np.asarray(H)
    fb = np.where(floated_bins)[0]
    idx = list(range(nf)) + [nf + i for i in fb]
    if mode == "joint":
        idx.append(nf + nzbin)          # sigint, marginalised then dropped
    sub = H[np.ix_(idx, idx)]
    curved = np.any(sub != 0.0, axis=1)
    cov = np.full(sub.shape, np.nan)
    active = sub[np.ix_(curved, curved)]
    if not np.isfinite(active).all():
        raise SNANAError("the fit's Hessian contains non-finite values",
                         function_name="_fit_covariance")
    try:
        np.linalg.cholesky(active)
        cov[np.ix_(curved, curved)] = 2.0 * np.linalg.inv(active)
    except np.linalg.LinAlgError as exc:
        raise SNANAError(f"the fit's Hessian is not positive definite: {exc}",
                         function_name="_fit_covariance") from None
    d = np.sqrt(np.diag(cov))
    nm = nf + len(fb)
    m0err = np.full(nzbin, np.nan)
    m0err[fb] = d[nf:nm]
    cov_m0 = np.zeros((nzbin, nzbin))
    cov_m0[np.ix_(fb, fb)] = cov[nf:nm, nf:nm]
    cov_m0[~floated_bins, ~floated_bins] = 1.0e5
    return d[:nf], m0err, cov_m0


def _nuisance_errors(nerr, floated_nuis) -> Dict[str, float]:
    """The floated nuisance errors by name, saying which have none."""
    nuis_err = {n: float(nerr[i]) for i, n in enumerate(floated_nuis)}
    flat = [n for n, e in nuis_err.items() if np.isnan(e)]
    if flat:
        slog.warn(logger, "BBC", f"the likelihood does not depend on "
                  f"{', '.join(flat)}; the error is undefined")
    return nuis_err


def _pack(floated_nuis, nuis_defaults, m0_init, sigint_init, mode):
    vals = [nuis_defaults[n] for n in floated_nuis] + list(m0_init)
    if mode == "joint":
        vals.append(sigint_init)
    return jnp.array(vals, dtype=float)


def pad_rows(a, n_pad: int):
    """``a`` padded to ``n_pad`` rows by repeating its last, as a device array."""
    a = jnp.asarray(a)
    if n_pad < a.shape[0]:
        raise SNANAError(f"padding capacity {n_pad} is smaller than {a.shape[0]} rows",
                         function_name="pad_rows")
    if n_pad == a.shape[0]:
        return a
    return jnp.concatenate([a, jnp.repeat(a[-1:], n_pad - a.shape[0], axis=0)])


def pad_ccprior(prior, n_pad, template=None):
    """Pad a simulation prior with finite, zero-weight rows."""
    n = len(prior.cc_z)
    if n_pad < n:
        raise SNANAError(f"CC-prior capacity {n_pad} is smaller than {n} rows",
                         function_name="pad_ccprior")
    if n == n_pad:
        return prior
    source = prior if n else template
    if source is None or not len(source.cc_z):
        raise SNANAError("an empty prior needs a compatible finite row for padding",
                         function_name="pad_ccprior")
    arrays = {}
    for name in ("cc_z", "cc_logmass", "cc_mumodel", "cc_fitparbias_nodes", "cc_cell"):
        values = getattr(prior, name)
        row = getattr(source, name)[-1:]
        arrays[name] = jnp.concatenate((jnp.asarray(values), jnp.repeat(row, n_pad - n, axis=0)))
    return prior._replace(
        **arrays,
        cc_lcpar={name: jnp.concatenate((jnp.asarray(values),
                                        jnp.repeat(source.cc_lcpar[name][-1:], n_pad - n, axis=0)))
                  for name, values in prior.cc_lcpar.items()},
        cc_valid=jnp.concatenate((jnp.asarray(prior.cc_valid), jnp.zeros(n_pad - n))))


def padded(inputs: LikelihoodInputs, n_pad: int) -> LikelihoodInputs:
    """The inputs with every per-event array padded to ``n_pad`` rows and
    ``valid`` marking the real rows, so the cells of a grid share one
    compiled fit: every array leaves here as a device array of one dtype
    whatever it arrived as. A padded row's values are finite, so its terms
    are well defined before the mask zeroes them. The CC prior's own sample
    keeps its length; only its per-data-event arrays are padded."""
    n = int(inputs.z.shape[0])

    def pad_fields(obj, names):
        return obj._replace(**{
            name: pad_rows(getattr(obj, name), n_pad)
            for name in names if getattr(obj, name) is not None
        })

    per_event = (
        "z", "logmass", "covmat_fit", "mumodel", "izbin", "muerr_vpec_sq",
        "izbin_nbr", "m0_zfrac", "idsample", "sqsigint", "muerrsq_last",
        "fitwgt0", "fitparbias_nodes", "mucovscale_nodes", "mucovadd_nodes",
        "mubias_const", "fitparbias_nevt", "fitparbias_err",
    )
    out = pad_fields(inputs, per_event)
    out = out._replace(lcpar={k: pad_rows(v, n_pad)
                              for k, v in inputs.lcpar.items()})
    if inputs.ccprior is not None:
        prior = pad_fields(inputs.ccprior, ["data_cell", "pIa"])
        # Padded and unpadded priors must have the same device-array signature.
        out = out._replace(ccprior=jax.tree.map(jnp.asarray, prior))
    if inputs.h11prior is not None:
        out = out._replace(h11prior=pad_fields(
            inputs.h11prior, ["z", "muerrsq_raw", "pIa"]))
    return out._replace(valid=jnp.concatenate([jnp.ones(n), jnp.zeros(n_pad - n)]))


@dataclass(frozen=True)
class FitSpec:
    """Static settings shared by BBC's compiled evaluations.

    Event data and numerical grids live in the input records. Model flags,
    parameter names and the CC cell count determine the compiled calculation.
    ``with_log`` is SNANA's fitflag_sigmb==2.
    """
    model: type
    floated: Tuple[str, ...]
    defaults: Tuple[Tuple[str, float], ...]
    nzbin: int
    mode: str
    with_log: bool
    maxiter: int = 60
    u_flags: Tuple[Tuple[str, int], ...] = ()
    scalepcc_bayes: bool = False
    bias_names: Tuple[str, ...] = ()
    use_gamma0: bool = False
    cc_bias_names: Tuple[str, ...] = ()
    cc_ncell: int = 0

    def loss(self, inp):
        return make_loss(self, inp)

    def closed(self, inp, sigint, last):
        """The loss as a function of the parameter vector alone."""
        loss = self.loss(inp)
        if self.mode == "joint":
            return loss
        return lambda q: loss(q, sigint, last)

    def nuisance(self, p, sigint):
        """The full nuisance dict at ``p``, the floated entries read from it."""
        return nuisance_at(p, self.floated, dict(self.defaults, sigint=sigint))


@partial(jax.jit, static_argnames="spec")
def _fit(spec, p0, lo, hi, inp, sigint, last):
    def derivatives(p, loss, with_value):
        return loss_derivatives(p, loss, n_nuisance=len(spec.floated),
                                nzbin=spec.nzbin, interpolate=inp.m0_zfrac is not None,
                                with_value=with_value)

    return newton_lm(p0, lo, hi, spec.closed(inp, sigint, last), spec.maxiter,
                     derivative_fn=derivatives)


def _timed(timings, name, call, *args):
    """One compiled call, waited for, its wall time added to
    ``timings[name]``; without the wait JAX's asynchronous dispatch would
    charge the work to whichever later call first needs the result."""
    start = time.perf_counter()
    out = jax.block_until_ready(call(*args))
    timings[name] = timings.get(name, 0.0) + time.perf_counter() - start
    return out


def _timed_fit(timings, *args):
    """The fit call, charged to fit_compile when it compiles."""
    before = _fit._cache_size()
    start = time.perf_counter()
    out = jax.block_until_ready(_fit(*args))
    name = "fit_compile" if _fit._cache_size() > before else "fit_minimise"
    timings[name] = timings.get(name, 0.0) + time.perf_counter() - start
    return out


@partial(jax.jit, static_argnames="spec")
def _fit_derivatives(spec, p, inp, sigint, last):
    gradient, hessian, _ = loss_derivatives(
        p, spec.closed(inp, sigint, last), n_nuisance=len(spec.floated),
        nzbin=spec.nzbin, interpolate=inp.m0_zfrac is not None)
    return gradient, hessian


def _gradient_diagnostics(p, gradient, bounds):
    """Raw and bound-projected slopes in parameter units, not a convergence cut."""
    p, gradient = np.asarray(p), np.asarray(gradient)
    lo, hi = (np.asarray(bound) for bound in bounds)
    blocked = ((lo == hi) | ((p <= lo) & (gradient > 0))
               | ((p >= hi) & (gradient < 0)))
    projected = np.where(blocked, 0.0, gradient)
    return dict(max_gradient=float(np.max(np.abs(gradient), initial=0.0)),
                max_projected_gradient=float(np.max(np.abs(projected), initial=0.0)))


@partial(jax.jit, static_argnames="spec")
def _muerrsq(spec, p, inp, sigint):
    """The mu-variance at ``p`` and ``sigint``: the next iteration's
    reference (SALT2mu.c:5661) and what write_chi2grid reads."""
    fn = mures_muerrsq_fn(spec, inp, spec.nuisance(p, sigint), "snana_secant")
    return fn(p, sigint)[1]


@partial(jax.jit, static_argnames="spec")
def _chi2_ia(spec, p, inp, sigint, nfloat):
    """(CHI2SUM_IA, dof): the pIa-weighted Ia chi2 whose reduced form the
    sigint secant drives to 1 (SALT2mu.c:4734, :5086). Under a CC prior each
    event enters weighted by its Ia probability and the dof is the summed
    weight less the floated count (:5085, :5603, :5605)."""
    nuis = spec.nuisance(p, sigint)
    fn = mures_muerrsq_fn(spec, inp, nuis, "snana_secant")
    mures, muerrsq = fn(p, sigint)
    chi2 = inp.valid * mures * mures / muerrsq
    probcc, event_loss = event_chi2(spec, inp, nuis, mures, muerrsq,
                           inp.muerrsq_last)
    wgt = inp.valid * (1.0 - probcc)
    if inp.beams:
        wgt = jnp.where(event_loss == CHI2_BEAMS_NOPROB, 0.0, wgt)
    return jnp.sum(wgt * chi2), jnp.sum(wgt) - nfloat


def run_fit(spec: FitSpec, inputs: LikelihoodInputs, m0_reference: float,
            n_pad: int, sigint_init: float = 0.1,
            redchi2_tol: float = 0.02,
            fitflag_sigmb: int = 1, sigint_step1: float = 0.01,
            sigint_floor: float = 1.0e-3,
            sigint_ceiling: float = 2.0,
            stop_covfix: bool = False,
            dchi2red_dsigint: float = 0.0,
            m0_halfwidth: float = 5.0) -> BBCResult:
    """Fit the nuisance parameters, per-z-bin M0 offsets, and sigma_int.
    ``n_pad`` is the numerical event capacity shared by compatible fits."""
    model, floated_nuis = spec.model, spec.floated
    nuis_defaults = dict(spec.defaults)
    nzbin, mode, with_log = spec.nzbin, spec.mode, spec.with_log
    nf = len(floated_nuis)
    bounds = fit_bounds(model, floated_nuis, nzbin, m0_reference, mode,
                        sigint_floor, sigint_ceiling, m0_halfwidth)
    m0_init = np.full(nzbin, m0_reference)

    def _unpack(p):
        nuis = nuisance_at([float(x) for x in p[:nf]], floated_nuis, nuis_defaults)
        return nuis, np.asarray(p[nf:nf + nzbin])

    nsnfit = int(inputs.z.shape[0])
    floated_bins = np.bincount(np.asarray(inputs.izbin), minlength=nzbin) > 0
    nfloat = nf + int(floated_bins.sum()) + int(mode == "joint")

    timings: Dict[str, float] = {}
    inp = _timed(timings, "fit_pad", padded, inputs, n_pad)
    if mode == "joint":
        p0 = _pack(floated_nuis, nuis_defaults, m0_init, sigint_init, "joint")
        pf, ff, _ = _timed_fit(timings, spec, p0, bounds[0], bounds[1], inp,
                               0.0, inp.muerrsq_last)
        nuis, m0 = _unpack(pf)
        sigint = float(pf[nf + nzbin])
        gradient, H = _timed(timings, "fit_hessian", _fit_derivatives,
                             spec, pf, inp, 0.0, inp.muerrsq_last)
        nerr, m0err, cov_m0 = _fit_covariance(H, nf, nzbin, floated_bins, mode)
        nuis_err = _nuisance_errors(nerr, floated_nuis)
        final = _timed(timings, "fit_stats", _muerrsq, spec, pf, inp, sigint)
        chi2sum_ia, ndof_ia = (float(x) for x in _timed(
            timings, "fit_stats", _chi2_ia, spec, pf, inp, sigint, nfloat))
        return BBCResult(nuis, m0, m0err, sigint, float(ff), 1, nuis_err,
                         nsnfit, nf + 1, cov_m0, timings, final[:nsnfit],
                         inputs.muerrsq_last, chi2sum_ia=chi2sum_ia,
                         ndof_ia=ndof_ia, fitflag_sigmb=2,
                         **_gradient_diagnostics(pf, gradient, bounds))

    sigint = sigint_init
    p = _pack(floated_nuis, nuis_defaults, m0_init, 0.0, "secant")
    # only the floated parameters count toward the dof (NFITPAR_FLOAT, :2525)
    niter = MAXFITITER if fitflag_sigmb > 0 else 1
    chi2red_hist, sigint_hist = [], []
    slope_used = dchi2red_dsigint
    for it in range(niter):
        p, ff, _ = _timed_fit(timings, spec, p, bounds[0], bounds[1], inp,
                              sigint, inp.muerrsq_last)
        # the iflag==3 store (SALT2mu.c:5661): the next iteration reads it, and the
        # last one is what write_chi2grid sees
        muerrsq_final = _timed(timings, "fit_stats", _muerrsq, spec, p, inp, sigint)
        chi2sum_ia, ndof_ia = (float(x) for x in _timed(
            timings, "fit_stats", _chi2_ia, spec, p, inp, sigint, nfloat))
        redchi2 = chi2sum_ia / ndof_ia
        chi2red_hist.append(redchi2)
        sigint_hist.append(sigint)
        # prepNextFit STOP conditions (SALT2mu.c:4737)
        stop_tol = abs(redchi2 - 1.0) < redchi2_tol
        # a CC prior gets at least two iterations (:4745)
        if inputs.beams and it == 0:
            stop_tol = False
        stop_cov0 = it > 0 and sigint <= 0.0
        # STOP_MXFIT: the budget is spent, so sigint is not advanced past the
        # value this fit ran with (:4738)
        stop_mxfit = it == niter - 1
        if (fitflag_sigmb == 0 or stop_covfix or stop_tol or stop_cov0
                or stop_mxfit):
            break
        sigint, slope_used = next_sigint(chi2red_hist, sigint_hist,
                                         sigint_step1, dchi2red_dsigint)
        # retain the reference the final fit read, which
        # the output columns are formed from (:5622, before the :5661 update)
        inp = inp._replace(muerrsq_last=muerrsq_final)
    nuis, m0 = _unpack(p)
    gradient, H = _timed(timings, "fit_hessian", _fit_derivatives,
                         spec, p, inp, sigint_hist[-1], inp.muerrsq_last)
    nerr, m0err, cov_m0 = _fit_covariance(H, nf, nzbin, floated_bins, mode)
    nuis_err = _nuisance_errors(nerr, floated_nuis)
    return BBCResult(nuis, m0, m0err, sigint, float(ff), it + 1,
                     nuis_err, nsnfit, nf, cov_m0, timings,
                     muerrsq_final[:nsnfit],
                     None if inp.muerrsq_last is None else inp.muerrsq_last[:nsnfit],
                     slope_used,
                     fitflag_sigmb=2 if with_log else fitflag_sigmb,
                     chi2sum_ia=chi2sum_ia, ndof_ia=ndof_ia,
                     **_gradient_diagnostics(p, gradient, bounds))


_SIGINT_NEXT_MAX = 100.0    # next_covFitPar aborts above this (:22243)
MAXFITITER = 20             # SALT2mu.c:650


def fit_bounds(model, floated_nuis, nzbin, m0_reference, mode,
               sigint_floor, sigint_ceiling, m0_halfwidth=5.0):
    spec = {**BBC_NUISANCE_SPEC, **model.nuisance_spec}
    lo = [spec[n][1] for n in floated_nuis] \
        + [m0_reference - m0_halfwidth] * nzbin
    hi = [spec[n][2] for n in floated_nuis] \
        + [m0_reference + m0_halfwidth] * nzbin
    if mode == "joint":
        lo.append(sigint_floor)
        hi.append(sigint_ceiling)
    return jnp.array(lo), jnp.array(hi)


def next_sigint(chi2red_hist, sigint_hist, step1, slope_user=0.0):
    """(next sigint toward chi2red = 1, the slope used), clamped at 0
    (SNANA next_covFitPar:22200, which also stores DCHI2RED_DSIGINT)."""
    if slope_user != 0.0:      # user-fixed slope replaces the secant (:22219)
        nxt = sigint_hist[-1] - (chi2red_hist[-1] - 1.0) / slope_user
        slope = slope_user
    elif len(chi2red_hist) == 1:
        step = step1 if chi2red_hist[-1] > 1.0 else -step1
        nxt, slope = sigint_hist[-1] + step, 0.0
    else:
        slope = ((chi2red_hist[-1] - chi2red_hist[-2])
                 / (sigint_hist[-1] - sigint_hist[-2]))
        nxt = sigint_hist[-1] - (chi2red_hist[-1] - 1.0) / slope
    if nxt > _SIGINT_NEXT_MAX:     # "Crazy parval_next" abort (:22243)
        raise SNANAError(
            f"Crazy parval_next(sigint) = {nxt:.4f} at NFIT_ITER="
            f"{len(sigint_hist) - 1}: slope = {slope:.4f}, "
            f"sigint = {sigint_hist[-1]:.4f}, chi2red = {chi2red_hist[-1]:.4f}",
            function_name="next_covFitPar")
    return max(nxt, 0.0), slope
