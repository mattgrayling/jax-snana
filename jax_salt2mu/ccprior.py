"""CC-prior / BEAMS contamination model (SALT2mu.c prepare_CCprior:15900 +
the MNCHI2FUN BEAMS branch:4715).

The default ``opt_ccprior = GAUSS+UPDATE`` path bins bias-corrected CC-simulation
residuals in redshift and named model coordinates. Its per-cell Gaussian
(DMUAVG, DMURMS) is rebuilt every fit iteration from the current nuisance parameters
(so it is autodiff-clean). The data likelihood becomes BEAMS: ``-2 ln[P_Ia + P_CC]``.

This module builds the fixed inputs and evaluates the contaminant prior and
BEAMS probabilities used by the likelihood.
"""

from typing import Dict, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core.logging import SNANAError

from .biascor import precompute_fitparbias_nodes
from .binning import BinInfo, coeff_weights
from .config import lcpar_windows, map_bins, parval
from .distances import bias_coef, mu_coef, mu_lcpar_coef, mu_variance_scaled
from .tablevar import CUTBIT_BIASCOR, force_probcc0

DMUMIN, DMUMAX = -3.0, 4.0        # SALT2mu.c:17288
MINEVT_PER_CELL = 5               # SALT2mu.c:17377 (strict >, so needs >=6)
PIFAC = 0.3989422804014327        # 1/sqrt(2*pi), SALT2mu.c:2222
BEAMS_PROB_NORM = 0.15            # Prob_SUM scale before -2ln (SALT2mu.c:5611)
CHI2_BEAMS_NOPROB = 1.0e8         # Prob_SUM <= 0 (SALT2mu.c:5616)
_BEAMS_ARG_FLOOR = 1.0e-140       # below this, -1/arg^2 overflows the Hessian


def dmu_window_max(dmubin: float) -> float:
    """Top of the dmu window actually covered by whole bins (SALT2mu.c:17290-17295).

    SNANA sets nbin = int((DMUMAX-DMUMIN)/dmubin) and then keeps only events whose bin
    index is in range (:17449), so a width that does not divide the 7-mag span leaves
    the remainder uncovered and those events out of DMUAVG/DMURMS. At the default 0.5
    this is exactly DMUMAX.
    """
    nbin = int((DMUMAX - DMUMIN + 1.0e-8) / dmubin)
    return DMUMIN + nbin * dmubin


class CCPriorInputs(NamedTuple):
    """Numerical inputs for the differentiable BEAMS prior."""
    cc_lcpar: Dict            # CC-simulation distance parameters
    cc_z: np.ndarray          # [ncc]
    cc_logmass: np.ndarray    # [ncc]
    cc_mumodel: np.ndarray    # [ncc] reference-cosmology mu
    cc_fitparbias_nodes: np.ndarray  # [ncc, N_node, k] Ia biasCor at each node
    cc_cell: np.ndarray        # [ncc] cell index (fixed)
    cc_valid: np.ndarray      # [ncc] in-cell & valid-biasCor weight (float)
    data_cell: np.ndarray      # [ndata_accept] cell index
    pIa: np.ndarray           # [ndata_accept] per-event P(Ia)
    coeff_avg: Tuple
    coeff_binsize: Tuple
    m0_default: float
    dmu_max: float = DMUMAX    # top of the dmu window, truncated by dmubin_ccprior


class H11PriorInputs(NamedTuple):
    """Fixed inputs for the analytic Hlozek-2011 polynomial CC prior
    (prob_CCprior_H11, SALT2mu.c:18311). ``muerrsq_raw`` is the CC mu-variance at the
    fixed config-start alpha/beta (INPUTS.parval, deliberately not the fit values,
    SALT2mu.c:18357); the poly params float."""
    z: np.ndarray              # [n] data zHD
    muerrsq_raw: np.ndarray    # [n] fcn_muerrsq at start alpha/beta, no sigint
    pIa: np.ndarray            # [n] per-event P(Ia)


def build_h11prior(data_tv, data_accept, inputs, config, model) -> H11PriorInputs:
    """Per-event H11 constants: the mu-variance at the config-start nuisance values,
    which SNANA takes from INPUTS.parval: the p<N> value, else its default
    (:18357), with no sigInt."""
    nuis = {n: parval(config, n) for n in model.nuisance_spec}
    z = np.asarray(inputs.z)
    v = np.tile(mu_coef(model, nuis, config._u_flags), (z.shape[0], 1))
    muerrsq = (mu_variance_scaled(v, np.asarray(inputs.covmat_fit), z,
                                  config.lensing_zpar)
               + np.asarray(inputs.muerr_vpec_sq))
    return H11PriorInputs(z=z, muerrsq_raw=muerrsq,
                          pIa=_pIa(data_tv, config)[data_accept])


def _ccprior_bininfo(vmin: float, vmax: float, nbin: int) -> BinInfo:
    """setup_BININFO_CCPRIOR edge construction (SALT2mu.c:17329)."""
    xbin = (vmax - vmin) / nbin
    x1 = vmax - xbin + 1.0e-5 * (vmax - vmin)
    los = []
    x = vmin
    while x < x1:
        los.append(x)
        x += xbin
    los = np.asarray(los)
    his = los + xbin
    return BinInfo(len(los), los, his, 0.5 * (los + his), xbin)


def build_ccprior_bins(config, model) -> Tuple[BinInfo, ...]:
    """Redshift followed by the declared contaminant-prior coordinates."""
    cuts = lcpar_windows(model, config)
    return (_ccprior_bininfo(config.zmin, config.zmax, config.nzbin_ccprior),
            *(_ccprior_bininfo(*cuts[name], count)
              for name, count in map_bins(model.ccprior_bins, config.ccprior_bins).items()))


def cell_index(bins, values) -> np.ndarray:
    """Redshift-fastest flattening; out-of-range values clamp to edge bins."""
    indices = tuple(b.index(np.asarray(v), opt=2) for b, v in zip(bins, values))
    return np.ravel_multi_index(indices, tuple(b.nbin for b in bins), order="F")


def _cells_per_sample(bins) -> int:
    """SNANA keeps one DMU Gaussian per (IDSAMPLE, cell): DMUAVG[IDSAMPLE][i3d]
    (SALT2mu.c:17502), filled per sample and skipping foreign events (:17438)."""
    return int(np.prod([b.nbin for b in bins]))


def build_ccprior(ccprior_tv, cc_idsample, maps, model, config
                  ) -> Tuple[CCPriorInputs, int]:
    """Return the BEAMS numerical inputs from the CC events, and the number of
    CC cells. The data's fields (``data_cell``, ``pIa``) are left None for
    :func:`ccprior_data`, so cells sharing the biasCor maps and the CC sample
    share these."""

    bins = build_ccprior_bins(config, model)
    names = tuple(map_bins(model.ccprior_bins, config.ccprior_bins))
    cells_per_sample = _cells_per_sample(bins)
    ncell = cells_per_sample * max(len(maps.samples), 1)

    cc_fval, _, _, cc_ok = precompute_fitparbias_nodes(
        ccprior_tv, maps, config.min_per_cell_biascor,
        idsample=cc_idsample)
    # SNANA records this on the CC sample's CUTMASK too, so it reaches NEVT_CCPRIOR
    ccprior_tv.cutbits[~cc_ok] |= 1 << CUTBIT_BIASCOR

    # Rejected fits can contain non-finite parameters (e.g. log10 of a failed
    # SALT x0). Remove them before binning or evaluating the CC likelihood;
    # multiplying their residuals by zero would still contaminate the moments.
    keep = ccprior_tv.cutmask & (cc_idsample >= 0)
    cc_cell = (cell_index(bins, (ccprior_tv.zhd[keep],
                               *(ccprior_tv.lcpar[n][keep] for n in names)))
               + cc_idsample[keep] * cells_per_sample)

    m0 = maps.maps[0]
    coeff_avg = tuple(m0.bininfo[j].avg for j in range(m0.n_coeff))
    coeff_bin = tuple(float(m0.bininfo[j].binsize) for j in range(m0.n_coeff))

    return CCPriorInputs(
        cc_lcpar={n: ccprior_tv.lcpar[n][keep] for n in model.lcpar_names},
        cc_z=np.asarray(ccprior_tv.zhd[keep]),
        cc_logmass=np.asarray(ccprior_tv.logmass[keep]),
        cc_mumodel=np.asarray(ccprior_tv.mumodel[keep]),
        cc_fitparbias_nodes=cc_fval[keep],
        cc_cell=cc_cell, cc_valid=np.ones(int(keep.sum())),
        data_cell=None, pIa=None,
        coeff_avg=coeff_avg, coeff_binsize=coeff_bin,
        m0_default=model.m0_reference,
        dmu_max=dmu_window_max(config.dmubin_ccprior)), ncell


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
    def pad(values, source_values):
        row = jnp.asarray(source_values[-1:])
        return jnp.concatenate((jnp.asarray(values), jnp.repeat(row, n_pad - n, axis=0)))

    return prior._replace(
        cc_z=pad(prior.cc_z, source.cc_z),
        cc_logmass=pad(prior.cc_logmass, source.cc_logmass),
        cc_mumodel=pad(prior.cc_mumodel, source.cc_mumodel),
        cc_fitparbias_nodes=pad(prior.cc_fitparbias_nodes, source.cc_fitparbias_nodes),
        cc_cell=pad(prior.cc_cell, source.cc_cell),
        cc_lcpar={name: pad(values, source.cc_lcpar[name])
                  for name, values in prior.cc_lcpar.items()},
        cc_valid=jnp.concatenate((jnp.asarray(prior.cc_valid), jnp.zeros(n_pad - n))))


def ccprior_data(prior, data_tv, data_accept, model, config) -> CCPriorInputs:
    """``prior`` (from :func:`build_ccprior`) with the accepted data events'
    CC cells and P(Ia)."""
    bins = build_ccprior_bins(config, model)
    names = tuple(map_bins(model.ccprior_bins, config.ccprior_bins))
    data_cell = (cell_index(bins, (data_tv.zhd[data_accept],
                                 *(data_tv.lcpar[n][data_accept] for n in names)))
                 + np.clip(data_tv.idsample[data_accept], 0, None)
                 * _cells_per_sample(bins))
    return prior._replace(data_cell=data_cell,
                          pIa=_pIa(data_tv, config)[data_accept])


def _pIa(data_tv, config) -> np.ndarray:
    """Per-event P(Ia): force_pIa='perfect' -> 1 for true SNIa else 0, else the
    varname_pIa column (SALT2mu.c:9259). Spec-confirmed SNe (force_probcc0) are
    forced to pIa=1 when a varname_pIa classifier is in use (REQUIRE_pIa)."""
    fp = config.force_pia
    if fp is not None and fp < 0:   # 'perfect' encoded as force_pia = -1
        if not data_tv.sim_truth:                    # !IS_SIM (SALT2mu.c:9263)
            raise SNANAError("force_pIa=perfect works only on simulated data",
                             function_name="build_ccprior")
        pia = (np.asarray(data_tv.sim_truth["SIM_TEMPLATE_INDEX"]) == 0).astype(float)
    elif fp is not None:
        pia = np.full(data_tv.size, float(fp))
    elif data_tv.pIa is None:               # REQUIRE_pIa (SALT2mu.c:9301)
        raise SNANAError(
            "CC prior needs force_pIa or a varname_pIa classifier column",
            function_name="build_ccprior")
    else:
        pia = np.asarray(data_tv.pIa, dtype=float)
    if config.varname_pia:                       # EXIST_pIa -> REQUIRE_pIa (SALT2mu.c:22175)
        pia = np.where(force_probcc0(data_tv, config), 1.0, pia)
    return pia


def _ccprior_dPdmu(spec, cc, nuis, mures):
    """Per data event dPdmu_CC: rebuild the CC-simulation cell Gaussian map at
    the current nuisance parameters (fcn_ccprior_muzmap + prob_CCprior_sim),
    then evaluate at each data event's residual. Autodiff-clean: cell ids are fixed."""
    model = spec.model
    nuis = {**nuis, **model.ccprior_fixed_nuisance}
    u_flags = dict(spec.u_flags)
    names = model.lcpar_names
    lcdict = {n: cc.cc_lcpar[n] for n in names}
    # SNANA evaluates the CC-simulation bias correction at gammaDM=0,
    # independently of the data's fitted mass step and the map's midpoint.
    host = {"z": cc.cc_z, "gamma_var": cc.cc_logmass,
            "gammadm_offset": 0.0, "use_gamma0": False}
    mu_cc = model.compute_mu(lcdict, nuis, host, u_flags, apply_mass_step=False)
    v = mu_lcpar_coef(model, names, nuis, cc.cc_z, cc.cc_logmass, u_flags)
    coeff_vals = model.biascor_coeff_values(nuis, host, u_flags)
    wgt = coeff_weights(coeff_vals, cc.coeff_avg, cc.coeff_binsize,
                        mu_cc.shape[0])
    proj = jnp.einsum("ni,nNi->nN", bias_coef(v, spec.cc_bias_names, jnp),
                      cc.cc_fitparbias_nodes)
    muBias = jnp.einsum("nN,nN->n", proj, wgt)
    dmu = mu_cc - cc.m0_default - cc.cc_mumodel - muBias

    w = cc.cc_valid * ((dmu >= DMUMIN) & (dmu < cc.dmu_max))
    ncc = jax.ops.segment_sum(w, cc.cc_cell, spec.cc_ncell)
    savg = jax.ops.segment_sum(w * dmu, cc.cc_cell, spec.cc_ncell)
    ssq = jax.ops.segment_sum(w * dmu * dmu, cc.cc_cell, spec.cc_ncell)
    denom = jnp.where(ncc > 0, ncc, 1.0)
    avg = savg / denom
    var = ssq / denom - avg * avg
    real = ncc > MINEVT_PER_CELL                       # >=6 events, else flat
    # keep the sqrt argument strictly positive so empty/flat cells (discarded by
    # the where below) don't feed a sqrt(0) NaN into the gradient.
    std = jnp.sqrt(jnp.where(real, jnp.maximum(var, 1e-12), 1.0))
    dmuavg = jnp.where(real, avg, 0.0)
    dmurms = jnp.where(real, std, 9999.0)

    drms = dmurms[cc.data_cell]
    resid = (mures - dmuavg[cc.data_cell]) / drms
    return jnp.where(drms > 1e-7, (PIFAC / drms) * jnp.exp(-0.5 * resid * resid), 0.0)


def h11_dPdmu(h11, nuis, mures):
    """Per data event dPdmu_CC from the H11 polynomial prior (prob_CCprior_H11,
    SALT2mu.c:18353): mu-offset and sigintCC are 2nd-order poly(z); the CC width is
    muerrsq_raw + sigintCC^2 (always positive, so no sqrt guard)."""
    z = h11.z
    ccpoly = nuis["H11mucc0"] + nuis["H11mucc1"] * z + nuis["H11mucc2"] * z * z
    sigcc = nuis["H11sigcc0"] + nuis["H11sigcc1"] * z + nuis["H11sigcc2"] * z * z
    sqsig = h11.muerrsq_raw + sigcc * sigcc
    dmu = mures - ccpoly
    return (PIFAC / jnp.sqrt(sqsig)) * jnp.exp(-0.5 * dmu * dmu / sqsig)


def beams_probs(nuis, mures, muerrsq, dPdmu_CC, pIa, fitflag2, muerrsq_last,
                bayes=False):
    """BEAMS per-event ``(PROBCC_BEAMS, chi2)`` (MNCHI2FUN, SALT2mu.c:5518-5617).

    The Ia term is a Gaussian in the fitted residual scaled by ``PIFAC/muerr_update``,
    where ``muerr_update`` is the current mu-error under fitflag_sigmb=2 and the frozen
    ``muerrsq_last`` otherwise (:5537). ``bayes`` applies scalePCC as the H11
    Eq. 4 Bayes factor instead of the class-weight rescale (u13=2,
    :5530-5535). ``chi2`` is ``-2ln`` of the class-weighted probability sum;
    SNANA substitutes a large penalty where that sum vanishes (:5616).
    The single source for the fit loss, the output columns and the chi2 grid.
    """

    muerrsq_update = muerrsq if fitflag2 else muerrsq_last
    dPdmu_Ia = (jnp.exp(-0.5 * mures * mures / muerrsq)
                * PIFAC / jnp.sqrt(muerrsq_update))
    scale = nuis["scalePCC"]
    if bayes:
        ptot_ia = pIa * scale / (1.0 + (scale - 1.0) * pIa)
        prob_cc = (1.0 - ptot_ia) * dPdmu_CC
        prob_sum = ptot_ia * dPdmu_Ia + prob_cc
    else:
        psum = pIa + scale * (1.0 - pIa)
        prob_cc = (scale * (1.0 - pIa) / psum) * dPdmu_CC
        prob_sum = (pIa / psum) * dPdmu_Ia + prob_cc
    arg = prob_sum * (BEAMS_PROB_NORM / PIFAC)
    chi2 = jnp.where(arg > 0.0,
                     -2.0 * jnp.log(jnp.maximum(arg, _BEAMS_ARG_FLOOR)),
                     CHI2_BEAMS_NOPROB)
    # A zero-density event retains the likelihood penalty, but contributes
    # neither Ia nor CC posterior weight. Callers identify it by that penalty.
    denominator = jnp.where(prob_sum > 0.0, prob_sum, 1.0)
    return jnp.where(prob_sum > 0.0, prob_cc / denominator, 0.0), chi2


def beams_dPdmu_CC(spec, inputs, nuis, mures):
    """dPdmu_CC from whichever CC prior is configured: the binned sim map or H11."""
    if inputs.h11prior is not None:
        return h11_dPdmu(inputs.h11prior, nuis, mures), inputs.h11prior.pIa
    return (_ccprior_dPdmu(spec, inputs.ccprior, nuis, mures),
            inputs.ccprior.pIa)
