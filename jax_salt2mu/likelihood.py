"""The BBC distance-fit likelihood (SNANA MNCHI2FUN, SALT2mu.c:5105).

A single ``-2 ln L`` shared by the SNANA-secant and joint-sigma_int modes. The
model-specific physics (Tripp coefficients, mass step) lives entirely in the
model's ``compute_mu``; this module is model-agnostic. The mu-error
projection ``v = d(mu)/d(lcpar) = (1, alpha, -beta)`` is obtained by autodiff of
``compute_mu``, so it is never restated.

biasCor enters as two terms interpolated over the coefficient grid: the distance
bias ``muBias`` (subtracted from the residual) and ``muCOVscale`` (scales the
LC-fit part of the error). The per-event, per-node bias and muCOVscale are
precomputed (numpy); only the coefficient blend runs inside the loss.
"""

from typing import Dict, NamedTuple, Optional, Tuple

import jax.numpy as jnp

from .binning import coeff_weights
from .distances import bias_coef, mu_lcpar_coef, mu_variance_scaled
from .ccprior import beams_dPdmu_CC, beams_probs

MUERR_FITWGT0 = 8888.8   # MUERR of a chi2max(FITWGT0) deweighted event (SALT2mu.c:498)


class LikelihoodInputs(NamedTuple):
    """Numerical inputs to the fit: event arrays, coefficient grids and priors."""
    lcpar: Dict[str, jnp.ndarray]
    z: jnp.ndarray
    logmass: jnp.ndarray
    covmat_fit: jnp.ndarray           # [n, k, k]
    mumodel: jnp.ndarray
    izbin: jnp.ndarray                # int
    muerr_vpec_sq: jnp.ndarray        # [n]
    # 1 for an event, 0 for padding
    valid: jnp.ndarray
    # uM0=2 only: neighbour bin and interpolation fraction between the two M0 knots.
    izbin_nbr: Optional[jnp.ndarray] = None
    m0_zfrac: Optional[jnp.ndarray] = None
    idsample: Optional[jnp.ndarray] = None   # int, per event (output column only)
    sqsigint: Optional[jnp.ndarray] = None   # [n] per-event sigma_int^2 (sigint_fix)
    # INFO_DATA.muerrsq_last (SALT2mu.c:9471, :5661)
    muerrsq_last: Optional[jnp.ndarray] = None
    lensing_zpar: float = 0.0
    # chi2max(FITWGT0): per-event deweight flag (SALT2mu.c:3491)
    fitwgt0: Optional[jnp.ndarray] = None
    # biasCor (None -> no correction): per-event bias / muCOVscale at each coefficient
    # node, plus the per-axis grid used for the interpolation weights.
    fitparbias_nodes: Optional[jnp.ndarray] = None   # [n, N_node, k]
    mucovscale_nodes: Optional[jnp.ndarray] = None   # [n, N_node]
    mucovadd_nodes: Optional[jnp.ndarray] = None      # [n, N_node] additive sigint
    coeff_avg: Optional[Tuple] = None              # per-axis bin averages
    coeff_binsize: Optional[Tuple] = None          # per-axis bin sizes
    # biasCor SIM_gammaDM midpoint (:16228); 0 without biasCor
    gammadm_offset: float = 0.0
    mubias_const: Optional[jnp.ndarray] = None     # [n] frozen 1D z-interp bias
    ccprior: object = None                         # CCPriorInputs -> BEAMS likelihood
    h11prior: object = None                        # H11PriorInputs -> BEAMS likelihood
    # output-only biasCor breakdown (per-node event count / LC-param bias error)
    fitparbias_nevt: Optional[jnp.ndarray] = None  # [n, N_node]
    fitparbias_err: Optional[jnp.ndarray] = None   # [n, N_node, k]

    @property
    def beams(self) -> bool:
        """Whether a CC prior makes the likelihood BEAMS."""
        return self.ccprior is not None or self.h11prior is not None


def nuisance_at(flat, floated_nuis, fixed_nuis) -> Dict[str, float]:
    """``fixed_nuis`` with each floated nuisance read from the head of ``flat``."""
    nuis = dict(fixed_nuis)
    for i, name in enumerate(floated_nuis):
        nuis[name] = flat[i]
    return nuis


def m0_for(inputs, m0):
    """Each event's M0: its bin's value, or the uM0=2 interpolation between the
    two nearest knots (fcn_M0, SALT2mu.c:5821)."""
    m0_bin = m0[inputs.izbin]
    if inputs.m0_zfrac is None:
        return m0_bin
    return m0_bin + (m0[inputs.izbin_nbr] - m0_bin) * inputs.m0_zfrac


def mu_and_parts(spec, inputs: LikelihoodInputs, nuis: Dict[str, float],
                 m0: jnp.ndarray, sigint: float):
    """Per-event residual and the three error pieces, at explicit nuisance/m0/sigint.

    Returns a dictionary where ``mures`` is the distance residual,
    ``uncorrected_variance`` is the LC-fit+intrinsic+lensing variance
    before muCOVscale, ``fit_sq`` the same without the intrinsic scatter (the
    MUERR_RAW column, SALT2mu.c:5642), ``mucovscale`` the node-blended scale (1 if no
    biasCor), ``vpec_sq`` the peculiar-velocity variance, ``parbias`` the per-LC-param
    bias ``[n, k]``, ``mubias = v.parbias`` the distance bias (both zero if no
    biasCor), ``wgt`` the ``[n, N_node]`` per-event interpolation weights
    (None if no biasCor),
    and ``mucovadd`` the node-blended additive term applied where >0 (0 if off).
    The single source for both the loss's ``muerrsq`` and the output columns.
    """
    model = spec.model
    u_flags = dict(spec.u_flags)
    names = model.lcpar_names
    host = {"z": inputs.z, "gamma_var": inputs.logmass,
            "gammadm_offset": inputs.gammadm_offset,
            "use_gamma0": spec.use_gamma0}
    lcdict = {n: inputs.lcpar[n] for n in names}
    mu = model.compute_mu(lcdict, nuis, host, u_flags, apply_mass_step=True)

    v = mu_lcpar_coef(model, names, nuis, inputs.z, inputs.logmass, u_flags)
    fit_sq = mu_variance_scaled(v, inputs.covmat_fit, inputs.z, inputs.lensing_zpar,
                                xp=jnp)
    # sigint_fix: per-IDSAMPLE scatter (get_COVINT_model, SALT2mu.c:23181)
    sq = inputs.sqsigint if inputs.sqsigint is not None else sigint * sigint
    uncorrected_variance = fit_sq + sq * v[:, 0] ** 2
    vpec = inputs.muerr_vpec_sq
    mures = mu - m0_for(inputs, m0) - inputs.mumodel
    if inputs.mubias_const is not None:
        mures = mures - inputs.mubias_const      # frozen 1D biasCor

    mucovscale = jnp.ones_like(uncorrected_variance)
    mucovadd = jnp.zeros_like(uncorrected_variance)
    parbias = jnp.zeros((uncorrected_variance.shape[0], len(names)))
    mubias = jnp.zeros_like(uncorrected_variance)
    wgt = None
    if inputs.fitparbias_nodes is not None:
        coeff_vals = model.biascor_coeff_values(nuis, host, u_flags)
        wgt = coeff_weights(coeff_vals, inputs.coeff_avg, inputs.coeff_binsize,
                            uncorrected_variance.shape[0])
        parbias = jnp.einsum("nNk,nN->nk", inputs.fitparbias_nodes, wgt)
        mubias = jnp.einsum("ni,ni->n", bias_coef(v, spec.bias_names, jnp), parbias)
        mures = mures - mubias
        mucovscale = jnp.einsum("nN,nN->n", inputs.mucovscale_nodes, wgt)
        if inputs.mucovadd_nodes is not None:
            # additive OR multiplicative, not both (SALT2mu.c:5444)
            add = jnp.einsum("nN,nN->n", inputs.mucovadd_nodes, wgt)
            mucovadd = jnp.where(add > 0.0, add, 0.0)
            mucovscale = jnp.where(add > 0.0, 1.0, mucovscale)

    if inputs.fitwgt0 is not None:
        # deweighted events get MUERR_FITWGT0 in place of every error piece
        # (fcn_muerrsq, SALT2mu.c:6185; scales forced to 1, :5301)
        f = inputs.fitwgt0
        uncorrected_variance = jnp.where(f, MUERR_FITWGT0 ** 2, uncorrected_variance)
        mucovscale = jnp.where(f, 1.0, mucovscale)
        mucovadd = jnp.where(f, 0.0, mucovadd)
        vpec = jnp.where(f, 0.0, vpec)

    return dict(mures=mures, uncorrected_variance=uncorrected_variance,
                fit_sq=fit_sq, mucovscale=mucovscale, vpec_sq=vpec,
                parbias=parbias, mubias=mubias, wgt=wgt, mucovadd=mucovadd)


def mures_muerrsq_fn(spec, inputs: LikelihoodInputs, fixed_nuis, mode):
    """The single ``(mures, muerrsq)`` map, shared by the loss and the secant
    helpers: ``flat -> ...`` in joint mode (sigma_int packed in ``flat``),
    ``(flat, sigint) -> ...`` in secant mode."""
    floated_nuis, nzbin = spec.floated, spec.nzbin
    nf = len(floated_nuis)

    def _core(flat, sigint):
        m0 = flat[nf:nf + nzbin]
        parts = mu_and_parts(
            spec, inputs, nuisance_at(flat, floated_nuis, fixed_nuis), m0, sigint)
        variance = (parts["uncorrected_variance"] * parts["mucovscale"]
                    + parts["vpec_sq"] + parts["mucovadd"])
        return parts["mures"], variance

    if mode == "joint":
        return lambda flat: _core(flat, flat[nf + nzbin])
    return _core


def event_chi2(spec, inputs, nuis, mures, muerrsq, muerrsq_last):
    """Per-event ``(PROBCC_BEAMS, chi2)`` as MNCHI2FUN stores them: BEAMS under a
    CC prior, else ``mures^2/muerrsq`` with the ``log(muerrsq/muerrsq_last)``
    term under ``with_log`` and PROBCC zero. Joint fits always use the current
    Gaussian normalization, including ``log(muerrsq)`` for a pure-Ia sample."""
    if inputs.beams:
        dPdmu_CC, pIa = beams_dPdmu_CC(spec, inputs, nuis, mures)
        return beams_probs(nuis, mures, muerrsq, dPdmu_CC, pIa,
                           spec.mode == "joint" or spec.with_log,
                           muerrsq_last, spec.scalepcc_bayes)
    chi2 = mures * mures / muerrsq
    if spec.mode == "joint":
        chi2 = chi2 + jnp.log(muerrsq)
    elif spec.with_log:
        chi2 = chi2 + jnp.log(muerrsq / muerrsq_last)
    return jnp.zeros_like(mures), chi2


def make_loss(spec, inputs: LikelihoodInputs):
    """Return the loss for newton_lm: ``loss(flat)`` in joint mode (sigma_int
    packed after the M0 offsets), ``loss(flat, sigint, muerrsq_last)`` in
    secant mode, so the secant iterations trace once.

    ``with_log`` is SNANA's fitflag_sigmb==2 (i.e. with biasCor): it adds the
    ``log(muerrsq/last)`` normalisation term (:4702) and, under a CC prior,
    switches the Ia prefactor to the current mu-error (:4734).
    """
    floated_nuis, fixed_nuis = spec.floated, dict(spec.defaults)
    _mures_muerrsq = mures_muerrsq_fn(spec, inputs, fixed_nuis, spec.mode)
    valid = inputs.valid

    if spec.mode == "joint":
        def loss(flat):
            mures, muerrsq = _mures_muerrsq(flat)
            _, chi2 = event_chi2(spec, inputs,
                                 nuisance_at(flat, floated_nuis, fixed_nuis),
                                 mures, muerrsq, inputs.muerrsq_last)
            return jnp.sum(valid * chi2)
        return loss

    def loss(flat, sigint, muerrsq_last):
        mures, muerrsq = _mures_muerrsq(flat, sigint)
        _, chi2 = event_chi2(spec, inputs, nuisance_at(flat, floated_nuis, fixed_nuis),
                             mures, muerrsq, muerrsq_last)
        return jnp.sum(valid * chi2)
    return loss
