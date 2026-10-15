"""Derived per-event and per-z-bin quantities from a BBC fit result."""

from dataclasses import dataclass, field as _dc_field, replace
from functools import partial
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np

from .config import (BLIND_COSINE_M0, BLINDMASK_MUz, MASK_BIASCOR_MU,
                     MASK_BIASCOR_MUCOVADD)
from .fitter import BBCResult, padded
from .ccprior import CHI2_BEAMS_NOPROB
from .likelihood import LikelihoodInputs, event_chi2, m0_for, mu_and_parts
from .zbins import ZBinGrid
from .distances import bias_coef, mu_coef

MUDIFERR_EMPTY = 999.0   # z-bin not floated (SALT2mu.c:501)
MUDIFERR_ZERO = 666.0    # floated z-bin with zero error (SALT2mu.c:502)
# crazy_M0_errors thresholds (SALT2mu.c:4823-4832)
CRAZY_SIGINT_REF = 0.100      # an M0 error cannot sensibly beat sigint/sqrt(N)
CRAZY_ERRMIN_FRAC = 0.67
CRAZY_ERRMAX = 1.5
CRAZY_MIN_NEVT = 3
CRAZY_MUCOVSCALE_MIN = 0.01   # events below this stay out of the bin average
PIA_MIN = 1.0e-6         # pIa floor in muerr_renorm, avoids WGT=0 (SALT2mu.c:25517)


@dataclass
class YamlSummary:
    """Counts for the .YAML aggregator summary (write_yaml_info, SALT2mu.c:23571).

    Gathered in finish_cell, where the IDSAMPLE partition is still in scope.
    SNANA prepends a synthetic "ALL" row at IDSAMPLE=-1 to every by-sample list.
    """
    nevt: Dict[str, int] = _dc_field(default_factory=dict)   # event type -> passing
    nreject_cutwin: int = 0        # any cut that is not the biasCor (:18665)
    nreject_biascor: int = 0       # events rejected only for want of a biasCor
    sample_names: list = _dc_field(default_factory=list)
    sample_ids: list = _dc_field(default_factory=list)
    nevt_by_sample: Dict[str, list] = _dc_field(default_factory=dict)
    nreject_cutwin_by_sample: list = _dc_field(default_factory=list)
    nreject_biascor_by_sample: list = _dc_field(default_factory=list)
    isdata_real: bool = False
    cpu_minutes: float = 0.0
    ncrazy: int = 0
    # contamination under a CC prior (write_yaml_info, SALT2mu.c:23775-23801)
    contam_data: float = None
    contam_true: float = None          # sim data only
    contam_true_err: float = 0.0


@dataclass
class BBCOutput:
    mu: np.ndarray          # per event
    mures: np.ndarray
    muerr: np.ndarray       # sig_mu (muCOVscale-scaled, + vpec)
    mupull: np.ndarray
    m0dif: np.ndarray       # per z bin: M0(iz) - <M0>
    m0diferr: np.ndarray
    muref: np.ndarray       # per z bin: 1/muerrsq-wgt avg reference mumodel
    nfit: np.ndarray
    zbin_z: np.ndarray = None   # per z bin: z inverting the cosmology to muref (zM0)
    # per-event display columns
    mumodel: np.ndarray = None
    muerr_raw: np.ndarray = None       # LC-fit + lensing, excluding intrinsic/muCOVscale/vpec
    muerr_vpec: np.ndarray = None
    chi2: np.ndarray = None            # per event, as MNCHI2FUN stores it (:4826)
    izbin: np.ndarray = None
    idsample: np.ndarray = None
    biascor: Dict[str, np.ndarray] = _dc_field(default_factory=dict)  # biasCor_*
    probcc: np.ndarray = None          # PROBCC_BEAMS per event (CC prior only)
    muerr_renorm: np.ndarray = None    # MUERR_RENORM per event (CC prior only)
    # fitted spec-confirmed Ia and true-CC counts (NSNSPEC_IA / NSNFIT_TRUECC)
    nsnspec_ia: int = 0
    nsnfit_truecc: int = None          # sim data only
    # the contamination block, over the photo-classified subset: the two
    # probability sums and their ratio (SUM_PROBIa / SUM_PROBCC / CONTAM_DATA)
    sum_probia: float = 0.0
    sum_probcc: float = 0.0
    contam_data: float = 0.0
    # z bins whose fitted M0 error is implausible (NWARN_CRAZYERR)
    crazy_m0: np.ndarray = None
    # header lines SNANA writes into both .M0DIF and .FITRES
    warnings: list = _dc_field(default_factory=list)
    isdata_real: bool = False        # no SIM columns present (:1524)
    version_photometry: Dict[str, str] = _dc_field(default_factory=dict)
    # the biasCor simulation's generating cosmology, {} when its header has none
    biascor_cosmology: Dict[str, str] = _dc_field(default_factory=dict)
    yaml: YamlSummary = None
    # seconds per section of the cell, prepare through output (jax only)
    timings: Dict[str, float] = None
    # (NSIM, N_alpha, N_beta, sigint) for the FITRES fit-results block
    biascor_summary: tuple = None
    # derived scalars for the summary block
    m0avg: float = 0.0
    ndof: int = 0
    # the configuration the fit ran with: resolved, and blinded where asked
    config: object = None


@partial(jax.jit, static_argnames="spec")
def _output_parts(spec, inp, nuis, m0, sigint):
    """The fit's residual and error decomposition at the fitted values, as
    MNCHI2FUN leaves them (SALT2mu.c:5463): mu -= muBias, muerr *= scale;
    and each event's PROBCC and chi2 from the reference mu-variance the
    final fcn pass read (:5622, before the :5661 update)."""
    parts = mu_and_parts(spec, inp, nuis, m0, sigint)
    muerrsq = (parts["uncorrected_variance"] * parts["mucovscale"]
               + parts["vpec_sq"] + parts["mucovadd"])
    pcc, chi2 = event_chi2(spec, inp, nuis, parts["mures"], muerrsq,
                           inp.muerrsq_last)
    return parts, pcc, chi2


def compute_output(spec, inputs: LikelihoodInputs, zbins: ZBinGrid,
                   result: BBCResult, config, n_pad: int, dust: bool) -> BBCOutput:
    """The per-event and per-bin output of a fit; ``n_pad`` is the grid's
    width, so cells reuse the compiled output evaluation, and ``dust`` whether
    the biasCor is a dust model, which halves the crazy-error reference."""
    model = spec.model
    spec = replace(spec, with_log=result.fitflag_sigmb == 2)
    inputs = inputs._replace(muerrsq_last=result.muerrsq_last)
    n = int(inputs.z.shape[0])
    nuis = dict(result.nuisance, sigint=result.sigint)
    parts, pcc_j, c2_j = _output_parts(
        spec, padded(inputs, n_pad), nuis,
        jnp.asarray(result.m0), result.sigint)
    # the node weights are None without a cell map
    parts = {name: None if value is None else np.asarray(value[:n])
             for name, value in parts.items()}
    mures = parts["mures"]
    scale = parts["mucovscale"]
    vpec = parts["vpec_sq"]
    muerrsq = (parts["uncorrected_variance"] * scale + vpec
               + parts["mucovadd"])
    muerr = np.sqrt(muerrsq)
    # the two diagnostic errors carry neither sigint nor muCOVscale (SALT2mu.c:5642):
    # MUERR_RAW includes LC-fit and lensing; MUERR_VPEC is peculiar velocity alone
    muerr_raw = np.sqrt(parts["fit_sq"])
    muerr_vpec = np.sqrt(vpec)

    izbin = np.asarray(inputs.izbin)
    m0 = result.m0
    nbin = zbins.nbin
    nfit = np.bincount(izbin, minlength=nbin)
    floated = nfit > 0
    avg_m0 = np.average(m0[floated], weights=nfit[floated]) if floated.any() else 0.0

    # MU = MURES + MUMODEL + M0DIF, i.e. mu - AVEMAG0 (SALT2mu.c:25437/24347).
    m0dif = m0 - avg_m0
    mumodel = np.asarray(inputs.mumodel)
    # SNANA writes MU from a per-event INFO_DATA.M0[n] (SALT2mu.c:25274), which
    # under uM0=2 is the interpolated value rather than the bin's.
    m0_event = np.asarray(m0_for(inputs, jnp.asarray(m0)))
    mu_out = mures + mumodel + (m0_event - avg_m0)
    if not config.uave:
        # uave=0 reports MU against the nominal M0 instead of the fitted average
        # (SNANA SNMAG0, SALT2mu.c:24382).
        mu_out = mu_out + (avg_m0 - model.m0_reference)
    # MUz blinding offsets MUDIF only, after MU is formed (SALT2mu.c:25438).
    if config.blindflag & BLINDMASK_MUz:
        m0dif = m0dif + np.cos(BLIND_COSINE_M0 * zbins.lo)
    # an unfloated bin is written as an exact zero, after the blind offset (:25443)
    m0dif = np.where(floated, m0dif, 0.0)
    mupull = mures / muerr
    chi2 = np.asarray(c2_j[:n])

    m0diferr = np.where(floated, result.m0err, MUDIFERR_EMPTY)
    # a floated bin with zero error writes 666 (SALT2mu.c:25444)
    m0diferr = np.where(floated & (m0diferr < 1.0e-12), MUDIFERR_ZERO, m0diferr)
    # MUREF = 1/muerrsq-weighted mean of the reference mumodel per bin (calc_zM0_data,
    # SALT2mu.c:14904); the reported z inverts the cosmology to it (finish_cell).
    w = 1.0 / muerrsq
    wsum = np.bincount(izbin, weights=w, minlength=nbin)
    muref = np.where(floated,
                     np.bincount(izbin, weights=w * mumodel, minlength=nbin)
                     / np.where(wsum > 0, wsum, 1.0), 0.0)

    # per-event biasCor breakdown (blend of the per-node bias at the fit coefficients)
    biascor = {}
    if inputs.fitparbias_nodes is not None:
        parbias = parts["parbias"]   # [n, k], order = lcpar names
        wgt = parts["wgt"]
        # biasCor_nevt: events summed over the active nodes (get_muBias); Err_mu:
        # sqrt(sum_k (MUCOEF_k * blended_err_k)^2), MUCOEF = d(mu)/d(lcpar) (:15776).
        nevt = np.einsum("nN,nN->n", np.asarray(inputs.fitparbias_nevt),
                         (wgt > 0).astype(float))
        err = np.einsum("nNk,nN->nk", np.asarray(inputs.fitparbias_err), wgt)
        mucoef = bias_coef(mu_coef(model, nuis, dict(spec.u_flags)), spec.bias_names)
        biascor["biasCor_nevt"] = nevt.astype(np.int64)
        biascor["biasCor_mu"] = parts["mubias"]
        biascor["biasCorErr_mu"] = np.sqrt(np.sum((mucoef * err) ** 2, axis=1))
        if not config.opt_biascor & MASK_BIASCOR_MU:
            # with the bias on MU there are no per-parameter biases (:24181)
            for k, nm in enumerate(model.lcpar_names):
                biascor[f"biasCor_{nm}"] = parbias[:, k]
        # as applied / as blended (SALT2mu.c:5426-5430)
        biascor["biasCor_muCOVSCALE"] = scale
        if inputs.mucovadd_nodes is not None:
            biascor["biasCor_muCOVADD"] = np.einsum(
                "nN,nN->n", np.asarray(inputs.mucovadd_nodes), wgt)

    elif inputs.mubias_const is not None:
        # 1D path: SNANA still writes the block (:25324)
        n = len(izbin)
        biascor["biasCor_nevt"] = np.zeros(n, dtype=np.int64)
        biascor["biasCor_mu"] = np.asarray(inputs.mubias_const)
        biascor["biasCorErr_mu"] = np.zeros(n)
        if not config.opt_biascor & MASK_BIASCOR_MU:
            for nm in model.lcpar_names:
                biascor[f"biasCor_{nm}"] = np.zeros(n)
        biascor["biasCor_muCOVSCALE"] = np.ones(n)

    # the muCOVADD column is written on the opt_biascor bit alone (:24733), so it is
    # present-and-zero wherever the bit is set without a cell map behind it
    if biascor and config.opt_biascor & MASK_BIASCOR_MUCOVADD:
        biascor.setdefault("biasCor_muCOVADD", np.zeros(len(izbin)))

    probcc = np.asarray(pcc_j[:n]) if inputs.beams else None

    muerr_renorm = None
    if probcc is not None and not config.fixpar_all:
        muerr_renorm = _muerr_renorm(probcc, chi2, muerr, izbin, m0diferr, nbin)

    crazy = crazy_m0_errors(np.asarray(result.m0err), nfit, izbin, scale, config, dust)
    idsample = np.asarray(inputs.idsample) if inputs.idsample is not None else None
    ndof = result.nsnfit - result.nfloat - int(floated.sum())

    return BBCOutput(mu=mu_out, mures=mures, muerr=muerr, mupull=mupull,
                     m0dif=m0dif, m0diferr=m0diferr, muref=muref, nfit=nfit,
                     zbin_z=zbins.centers.copy(),
                     mumodel=mumodel, muerr_raw=muerr_raw, muerr_vpec=muerr_vpec,
                     chi2=chi2, izbin=izbin, idsample=idsample, biascor=biascor,
                     probcc=probcc, muerr_renorm=muerr_renorm,
                     crazy_m0=crazy, m0avg=float(avg_m0), ndof=ndof,
                     config=config)


def _muerr_renorm(probcc, chi2, muerr, izbin, m0diferr, nbin) -> np.ndarray:
    """Per-event error rescaled so that ``sum 1/MUERR_RENORM^2 == 1/M0DIFERR^2`` in
    each z bin, carrying the BEAMS 1/sqrt(pIa) (muerr_renorm, SALT2mu.c:25456)."""
    pia = np.maximum(1.0 - probcc, PIA_MIN)
    pia = np.where(chi2 == CHI2_BEAMS_NOPROB, 0.0, pia)
    wgt = pia / (muerr * muerr)
    sum_wgt = np.bincount(izbin, weights=wgt, minlength=nbin)
    ratio = np.where(sum_wgt > 0.0, 1.0 / np.sqrt(np.where(sum_wgt > 0.0, sum_wgt, 1.0))
                     / m0diferr, 0.0)
    denominator = ratio[izbin] * np.sqrt(pia)
    return np.divide(muerr, denominator, out=np.full_like(muerr, np.inf),
                     where=denominator > 0.0)


def crazy_m0_errors(m0err, nfit, izbin, mucovscale, config, dust) -> np.ndarray:
    """Per z bin, is the fitted M0 error physically implausible?

    crazy_M0_errors (SALT2mu.c:4811). An M0 error cannot sensibly fall below
    sigint/sqrt(N); SNANA treats anything under 0.67 of that, or above 1.5, as a
    sign the fit is untrustworthy, with the reference sigint halved for a dust
    model's biasCor (:4824). Bins with fewer than 3 events are not judged, and
    fixpar_all skips the check entirely (:4845).

    jax reports these and never re-fits. SNANA's retry nudges alpha/beta to escape
    a kink in MINUIT's numerical derivatives; Newton-LM with autodiff gradients does
    not share that failure mode, so the nudge would be a workaround for a solver
    this code does not use.
    """
    nbin = len(nfit)
    if config.fixpar_all:
        return np.zeros(nbin, dtype=bool)

    avg_scale = np.ones(nbin)
    if config.opt_biascor != 0:
        keep = mucovscale > CRAZY_MUCOVSCALE_MIN
        total = np.bincount(izbin[keep], mucovscale[keep], minlength=nbin)
        count = np.bincount(izbin[keep], minlength=nbin)
        avg_scale = np.where(count > 0, total / np.where(count > 0, count, 1.0), 1.0)

    judged = nfit >= CRAZY_MIN_NEVT
    sigint_ref = CRAZY_SIGINT_REF / 2.0 if dust else CRAZY_SIGINT_REF
    errmin = (np.sqrt(avg_scale) * sigint_ref
              / np.sqrt(np.where(judged, nfit, 1.0))) * CRAZY_ERRMIN_FRAC
    return judged & ((m0err < errmin) | (m0err > CRAZY_ERRMAX))
