"""Derived BBC fit quantities and their SNANA output tables and summaries."""

from dataclasses import dataclass, field, replace
from functools import partial
from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core.snana_files import (open_text, program_comment, snana_version_comment,
                                       write_varnames_table)

from .binning import ZBinGrid
from .ccprior import CHI2_BEAMS_NOPROB
from .config import (MASK_BIASCOR_MU, MASK_BIASCOR_MUCOVADD)
from .blinding import (BLIND_COSINE_M0, BLINDMASK_FIXPAR, BLINDMASK_MUz,
                       blind_distance_offsets)
from .distances import bias_coef, mu_at, mu_coef
from .fitter import BBCResult, padded
from .likelihood import LikelihoodInputs, event_chi2, m0_for, make_loss, mu_and_parts
from .tablevar import CUTBIT_BIASCOR, MISSING_TOKEN, force_probcc0, output_varnames

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
    nevt: Dict[str, int] = field(default_factory=dict)   # event type -> passing
    nreject_cutwin: int = 0        # any cut that is not the biasCor (:18665)
    nreject_biascor: int = 0       # events rejected only for want of a biasCor
    sample_names: list = field(default_factory=list)
    sample_ids: list = field(default_factory=list)
    nevt_by_sample: Dict[str, list] = field(default_factory=dict)
    nreject_cutwin_by_sample: list = field(default_factory=list)
    nreject_biascor_by_sample: list = field(default_factory=list)
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
    biascor: Dict[str, np.ndarray] = field(default_factory=dict)  # biasCor_*
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
    warnings: list = field(default_factory=list)
    isdata_real: bool = False        # no SIM columns present (:1524)
    version_photometry: Dict[str, str] = field(default_factory=dict)
    # the biasCor simulation's generating cosmology, {} when its header has none
    biascor_cosmology: Dict[str, str] = field(default_factory=dict)
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
    m0dif = blind_distance_offsets(config, m0dif, zbins.lo)
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


# wide enough to reproduce any input column (SNANA copies them verbatim, :23348)
_PASSTHROUGH_FLOAT_FMT = "%.9g"

# the written MUPULL is capped here (SALT2mu.c:25296)
MUPULL_MAX = 99.999

# rows cat_only copies; any other leading key is skipped (SALT2mu.c:24436)
_ROW_KEYS = frozenset(("SN:", "ROW:", "GAL:"))


def _fmt_for(arr):
    arr = np.asarray(arr)
    if arr.dtype.kind in "iub":
        return "%d"
    if arr.dtype.kind in "fc":
        return _PASSTHROUGH_FLOAT_FMT
    return "%s"


def _param_label(name, config):
    """u13=2 renames the CC-prior scale in every output (SALT2mu.c:22112)."""
    if name == "scalePCC" and config._u_flags.get("u13", 0) == 2:
        return "scalePIa"
    return name


def _fit_summary(result, out: BBCOutput, zbins: ZBinGrid, config):
    """The ``# SALT2mu fit results:`` block (write_fitres_misc,
    SALT2mu.c:24935-24998): NSNFIT / contamination under a CC prior /
    -2lnL / chi2(Ia)/dof / sigint / each floated nuisance +- err /
    per-bin m0 +- err / M0avg."""
    lines = ["SALT2mu fit results: "]
    if out.biascor_summary:
        ndim, nsim, note, sigint_bcor = out.biascor_summary
        lines += [f" NSIM({ndim}D-BIASCOR)   = {nsim}   {note}",
                  f" sigint({ndim}D-BIASCOR) = {sigint_bcor:.3f} "]
    lines += [f" NSNFIT        = {result.nsnfit} "]
    if out.probcc is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            note = "photo-classified subset"
            lines += [f" SUM_PROBIa    = {out.sum_probia:9.2f}    # {note} ",
                      f" SUM_PROBCC    = {out.sum_probcc:9.2f}    # {note} ",
                      f" CONTAM_DATA   = {out.contam_data:.4f}    "
                      f"# SUM_PROB ratio for {note}"]
            if out.nsnfit_truecc is not None:
                contam_true = float(np.divide(
                    out.nsnfit_truecc, result.nsnfit - out.nsnspec_ia))
                lines += [f" CONTAM_TRUE   = {contam_true:.4f}    "
                          f"# true NCC/(NIa+NCC) for {note} (sim-data only)"]
    chi2_ia, ndof_ia = result.chi2sum_ia, result.ndof_ia
    lines += [f" -2log(L)     = {result.chi2:.2f} ",
              f" dchi2red/dsigint = {result.dchi2red_dsigint:.3f}",
              f" chi2(Ia)/dof = {chi2_ia:.2f}/{ndof_ia:.0f} = "
              f"{chi2_ia / ndof_ia:.3f}  "
              if ndof_ia > 0 else f" chi2(Ia)     = {chi2_ia:.2f} ",
              f" sigint   = {result.sigint:.4f}  ({result.niter} iterations)"]
    for name, err in result.nuis_err.items():
        lines.append(f" {_param_label(name, config):<14s} = "
                     f"{result.nuisance[name]:10.5f} +- {err:8.5f}   ")
    for iz in range(zbins.nbin):
        # an unfloated bin prints "too few events->IGNORED" in SNANA (:23208);
        # jax leaves the line out, so the test is the bin's error, not its count
        if out.m0diferr[iz] >= MUDIFERR_EMPTY:
            continue
        lines.append(
            f" m0_{iz:02d}-<M0avg>  = {out.m0dif[iz]:11.5f} +- {out.m0diferr[iz]:9.5f} "
            f" ({zbins.lo[iz]:.3f} < z < {zbins.hi[iz]:.3f}, N={out.nfit[iz]}) ")
    lines.append(f" M0avg          = {out.m0avg:14.5f} ")
    lines.append(f" sum_N over m0_[nn] bins: {int(out.nfit.sum())} "
                 f" (compare to NSNFIT={result.nsnfit})")
    return lines


BBC_VERSION = 4   # SALT2mu.c output-format version stamp


def _provenance_header(out: BBCOutput) -> list:
    """The block SNANA opens both .M0DIF and .FITRES with (write_version_info).

    SNANA_VERSION carries jax's own build string: the field name is part of the
    format that downstream readers key on, so the name is kept and the value says
    which program actually wrote the file.
    """

    lines = [f"{k + ':':<30s}{v} " for k, v in (out.version_photometry or {}).items()]
    # after SNANA's own lines: its create_covariance reads only the first ten
    cosmology = [f"{f'SIM_{k}(BIASCOR):':<30s}{v} "
                 for k, v in out.biascor_cosmology.items()]
    return lines + [f"ISDATA_REAL:   {int(out.isdata_real)} ",
                    f"{snana_version_comment()} ",
                    f"{program_comment('salt2mu.py')} ",
                    f"BBC_VERSION:   {BBC_VERSION} "] + cosmology


def _cosmo_header(config):
    """Under BLINDMASK_FIXPAR SNANA prints the unblinded value and the offset
    formula rather than the blinded number (prep_blindVal_strings, :23716)."""

    blinded = config.blindflag & BLINDMASK_FIXPAR
    out = []
    for name, attr, value, slot in (("Omega_L", "omega_l", config.omega_l, 9),
                                    ("w0", "w0", config.w0, 11)):
        amp, phase = config.blindpar.get(slot, (0.0, 0.0))
        if blinded and amp != 0.0:
            val = f"{config._unblind.get(attr, value):8.4f} + {amp:6.4f}*cos({phase:f})"
        else:
            val = f"{value:.3f}"
        out.append(f"  REF {name:<12s}:  {val} ")
    return out


def _blind_message(config) -> list:
    """write_blindFlag_message (SALT2mu.c:24880)."""

    flag = config.blindflag
    if not flag:
        return []
    lines = ["", " *** WARNING: RESULTS ARE BLINDED *** "]
    if flag & BLINDMASK_MUz:
        lines.append(f" m0_## values have blind offset = cos( z * {BLIND_COSINE_M0:.3f} ) ")
    else:
        for name, attr, value, slot in (("Omega_L", "omega_l", config.omega_l, 9),
                                        ("w0", "w0", config.w0, 11)):
            amp, phase = config.blindpar.get(slot, (0.0, 0.0))
            if amp != 0.0:
                base = config._unblind.get(attr, value)
                lines.append(f"    {name:<13s}= {base:8.4f} + {amp:6.4f}*cos({phase:f}) ")
    return lines + [""]


def m0dif_columns(zbins: ZBinGrid, out: BBCOutput) -> dict:
    """The M0DIF columns: ROW zHDMIN zHDMAX zHD MUDIF MUDIFERR MUREF NFIT.

    What :func:`write_m0dif` writes, and what a cosmology fit downstream reads
    back, so a caller holding the output needs no file.
    """
    return {
        "ROW": np.arange(1, zbins.nbin + 1),
        "zHDMIN": zbins.lo, "zHDMAX": zbins.hi, "zHD": out.zbin_z,
        "MUDIF": out.m0dif, "MUDIFERR": out.m0diferr, "MUREF": out.muref,
        "NFIT": out.nfit,
    }


def write_m0dif(path, zbins: ZBinGrid, out: BBCOutput, result, config, warnings=()):
    """One row per z bin: ROW zHDMIN zHDMAX zHD MUDIF MUDIFERR MUREF NFIT."""
    header = _provenance_header(out) + \
        [None, "Reference cosmology params for MUDIF: "] + _cosmo_header(config)
    for name in ("alpha0", "beta0"):
        header.append(f"  FIT {name:<11s} :  "
                      f"{result.nuisance.get(name, 0.0):.4f} "
                      f"+- {result.nuis_err.get(name, 0.0):.4f} ")
    header += [""] + _blind_message(config) + list(warnings) + [
        None, "For cosmology fit: zHD, MU=MUREF+MUDIF, MUERR=MUDIFERR", None]
    cols = m0dif_columns(zbins, out)
    fmt = {"ROW": "%d", "zHDMIN": "%.5f", "zHDMAX": "%.5f", "zHD": "%.5f",
           "MUDIF": "%.5f", "MUDIFERR": "%.5f", "MUREF": "%.5f", "NFIT": "%d"}
    write_varnames_table(path, list(cols), cols, fmt, rowkey="ROW:",
                         header_comments=header, blank_separator=False)


# write_fitres_line_append column order and formats (SALT2mu.c:25281)
_BBC_FMT = {"CUTMASK": "%d",
            "MU": "%8.5f", "MUMODEL": "%8.5f", "MUERR": "%8.5f",
            "MUERR_RENORM": "%7.4f", "MUERR_RAW": "%7.4f", "MUERR_VPEC": "%7.4f",
            "MURES": "%7.4f", "MUPULL": "%6.3f",
            "M0DIF": "%7.4f", "M0DIFERR": "%7.4f",
            "CHI2": "%.2f", "CHI2_BEAMS": "%.2f", "PROBCC_BEAMS": "%.8e",
            "biasCor_nevt": "%4d", "biasCor_mu": "%6.4f", "biasCorErr_mu": "%6.3f",
            "biasCor_mB": "%6.4f", "biasCor_x1": "%6.3f", "biasCor_c": "%6.4f",
            "biasCor_muCOVSCALE": "%6.3f", "biasCor_muCOVADD": "%6.4f",
            "IDSAMPLE": "%d", "IZBIN": "%2d"}


def _cutmask_write_warning(config) -> List[str]:
    """write_NWARN(FLAG=1) (SALT2mu.c:24752): the FITRES, and only the FITRES,
    says when the written row set is not the fitted sample."""
    if config.cutmask_write == 0:
        return []
    return [f"WARNING(MINOR): cutmask_write={config.cutmask_write}"]


def write_fitres(path, out: BBCOutput, zbins: ZBinGrid, result, config,
                 passthrough, warnings=()):
    """One row per (accepted) SN: passthrough columns then the BBC set.

    ``passthrough`` is an ordered ``{name: array}`` (already masked to the fitted
    events, aligned with the output arrays) and must start with ``CID``.
    """
    cols, fmt = {}, {}
    for name, arr in passthrough.items():
        cols[name] = np.asarray(arr)
        fmt[name] = _fmt_for(arr)

    izbin = out.izbin
    beams = out.probcc is not None
    # 0 for every written row: only cutmask_write=0 (the fit sample) is supported
    bbc = {"CUTMASK": np.zeros(len(izbin), dtype=np.int64),
           "MU": out.mu, "MUMODEL": out.mumodel, "MUERR": out.muerr}
    if out.muerr_renorm is not None:
        bbc["MUERR_RENORM"] = out.muerr_renorm
    bbc.update({"MUERR_RAW": out.muerr_raw, "MUERR_VPEC": out.muerr_vpec,
                "MURES": out.mures,
                # the written pull is capped, one-sided, so the column keeps its
                # width (SALT2mu.c:25296); the CHI2 column is not capped
                "MUPULL": np.minimum(out.mupull, MUPULL_MAX),
                "M0DIF": out.m0dif[izbin], "M0DIFERR": out.m0diferr[izbin]})
    bbc["CHI2_BEAMS" if beams else "CHI2"] = out.chi2
    if beams:
        bbc["PROBCC_BEAMS"] = out.probcc
    if out.biascor:
        bbc.update(out.biascor)
        bbc["IDSAMPLE"] = out.idsample
    # Rebinning also needs the redshift bin when bias corrections are off.
    bbc["IZBIN"] = izbin
    for name, arr in bbc.items():
        cols[name] = np.asarray(arr)
        fmt[name] = _BBC_FMT.get(name, "%.5g")

    header = _provenance_header(out) + [
        None,
        "HESSIAN",
        f"CPU: {out.yaml.cpu_minutes:.2f} minutes",
    ] + _blind_message(config) + [
        "MU-RESIDUAL NOTE: MURES = MU-MUMODEL ",
        "For cosmology fit: use zHD, MU, MUERR_RENORM",
        "  [MUERR_RENORM is Eq 10 of arxiv:2306.05819, MUERR is sig_{mu,i}]",
        None] + _cutmask_write_warning(config) + list(warnings)
    post = [None] * 3 + _fit_summary(result, out, zbins, config) + [None]
    if config.cutmask_write == -9:
        # no SN rows, so no VARNAMES line either: the file is the header and the
        # fit-results block (write_fitres_driver, SALT2mu.c:24254, :24399)
        with open(path, "w") as f:
            for line in header + post:
                f.write("\n" if line is None else f"# {line}\n")
        return
    write_varnames_table(path, list(cols), cols, fmt, rowkey="SN:",
                         header_comments=header + [None],
                         blank_separator=False, post_varnames_comments=post)


def write_cov(path, cov_m0):
    """M0-block statistical covariance (write_M0_cov): nzbin, then row-major
    ``%le`` values; SNANA's 1e5 diagonal marks unfloated bins."""
    cov = np.asarray(cov_m0)
    nbin = cov.shape[0]
    with open(path, "w") as f:
        f.write(f"{nbin}\n")
        for iz in range(nbin):
            f.write(f"# ---------- begin row {iz} ------------ \n")
            for val in cov[iz]:
                f.write(f"{val:.6e}\n")


def write_yaml(path, summary, result, config, timings=None):
    """SNANA's .YAML summary for the pipeline aggregator (write_yaml_info,
    SALT2mu.c:23571): per-event-type counts, reject tallies, the by-sample
    breakdown, and the fitted parameters; then, jax's own, the seconds each
    section of the cell took, so a slow run says where.

    NFIT_EXEC is always 1: jax does not re-fit (see crazy_m0_errors).
    """
    rows = [(f"NEVT_{k}:", f"{summary.nevt.get(k, 0)} ")
            for k in ("DATA", "BIASCOR", "CCPRIOR")]
    rows += [("NEVT_REJECT_CUTWIN:", summary.nreject_cutwin),
             ("NEVT_REJECT_BIASCOR:", summary.nreject_biascor),
             ("ABORT_IF_ZERO:", summary.nevt.get("DATA", 0)),
             ("NWARN_CRAZYERR:", summary.ncrazy),
             ("NFIT_EXEC:", 1)]

    with open(path, "w") as f:
        for key, val in rows:
            f.write(f"{key:<22.22s} {val}\n")
        f.write(f"{'CPU_MINUTES:':<22.22s} {summary.cpu_minutes:.2f}\n")
        f.write(f"{'ISDATA_REAL:':<22.22s} {int(summary.isdata_real)}\n")

        if summary.sample_names:
            f.write("\n")
            _yaml_list(f, "SAMPLE_LIST:", summary.sample_names, lead_first=False)
            _yaml_list(f, "IDSAMPLE_LIST:", summary.sample_ids)
            for evtype in ("DATA", "BIASCOR", "CCPRIOR"):
                counts = summary.nevt_by_sample.get(evtype)
                if counts:
                    _yaml_list(f, f"NEVT_{evtype}_bySAMPLE:", counts)
            f.write("\n")
            _yaml_list(f, "NREJECT_CUTWIN_bySAMPLE:",
                       summary.nreject_cutwin_by_sample)
            _yaml_list(f, "NREJECT_BIASCOR_bySAMPLE:",
                       summary.nreject_biascor_by_sample)

        f.write("\nBBCFIT_RESULTS:\n")
        nsnfit = result.nsnfit
        f.write(f"  - NSNFIT:            {nsnfit:5d}     "
                f"{int(np.sqrt(nsnfit) + 0.5)}\n")
        f.write(f"  - SIGINT:            {result.sigint:.5f}    0\n")
        f.write(f"  - CHI2RED_IA:        "
                f"{result.chi2sum_ia / (result.ndof_ia + 1.0e-8):.3f}    0\n")
        for name, err in result.nuis_err.items():
            f.write(f"  - {_param_label(name, config) + ':':<18s}"
                    f"{result.nuisance[name]:11.5f} {err:8.5f} \n")
        if summary.contam_data is not None:
            f.write(f"  - {'CONTAM_DATA:':<20s}  "
                    f"{summary.contam_data:.5f}  {0.0:.5f}   "
                    f"# photo-classified subset \n")
        if summary.contam_true is not None:
            f.write(f"  - {'CONTAM_TRUE:':<20s}  "
                    f"{summary.contam_true:.5f}  "
                    f"{summary.contam_true_err:.5f}    "
                    f"# photo-classified subset (sim data only) \n")
        f.write("\nFIT_DIAGNOSTICS:\n")
        f.write(f"  max_gradient: {result.max_gradient:.8e}\n")
        f.write(f"  max_projected_gradient: {result.max_projected_gradient:.8e}\n")
        if timings:
            f.write("\nTIMING_SEC:\n")
            for name, value in timings.items():
                f.write(f"  {name + ':':<18s} {value:.3f}\n")


def _yaml_list(f, key, values, lead_first=True):
    """catVarList_with_comma joins with "," after prefixing each entry with a
    space, so the body reads " a, b". SAMPLE_LIST is the exception: its first
    entry ("ALL") is written without one."""
    parts = [f" {v}" for v in values]
    if not lead_first and parts:
        parts[0] = parts[0].lstrip()
    f.write(f"{key:<25.25s} {','.join(parts)}\n")


def write_m0_csv(path, zbins, out, config):
    """Binned Hubble diagram for CosmoMC (write_M0_csv, SALT2mu.c:23962). MU is the
    reference distance at the reported z plus M0DIF, not the weighted MUREF column."""
    zhd = out.zbin_z
    muref = mu_at(zhd, config)
    with open(path, "w") as f:
        f.write("# name   zcmb zhel zerr   mu muerr \n")
        for iz in range(zbins.nbin):
            f.write(f"BIN{iz:02d}   {zhd[iz]:.5f} {zhd[iz]:.5f} {0.0:.1f}   "
                    f"{muref[iz] + out.m0dif[iz]:.4f} {out.m0diferr[iz]:.4f} \n")


def write_chi2grid(path, spec, inputs, result, nbin):
    """chi2 scan over the model's ``chi2grid_axes`` (write_chi2grid,
    SALT2mu.c:24085): each row varies one parameter over its fixed range with
    the others held at their fitted values.

    SNANA scans ``fcn`` itself, so under a CC prior this is the BEAMS sum and not the
    Ia chi2. It runs from outFile_driver (:23511), after the fit has restored
    muerrsq_last to its own errors, hence the swapped-in muerrsq_final.
    Joint mode scans its normalized likelihood at the fitted intrinsic scatter.
    """
    post_fit = inputs._replace(muerrsq_last=result.muerrsq_final)
    spec = replace(spec, floated=(),
                               with_log=result.fitflag_sigmb == 2)
    axes = spec.model.chi2grid_axes
    # a scanned parameter the model has not got stays at SNANA's parval 0
    fitted = {name: result.nuisance.get(name, 0.0) for _, name, _ in axes}
    steps = {name: (hi - lo) / nbin for _, name, (lo, hi) in axes}
    m0 = jnp.asarray(result.m0)

    def chi2(values):
        fixed = dict(result.nuisance, sigint=result.sigint, **values)
        loss = make_loss(replace(spec, defaults=tuple(sorted(fixed.items()))),
                         post_fit)
        if spec.mode == "joint":
            return float(loss(jnp.append(m0, result.sigint)))
        return float(loss(m0, result.sigint, post_fit.muerrsq_last))

    with open(path, "w") as f:
        labels = [label for label, _, _ in axes]
        f.write("VARNAMES:  ROW  " + "  ".join(labels)
                + " " + " ".join(f"chi2_{label}" for label in labels) + " \n")
        for i in range(nbin):
            point = {name: lo + steps[name] * i for _, name, (lo, _) in axes}
            grid = " ".join(f"{point[name]:.4f}" for _, name, _ in axes)
            scans = "  ".join(f"{chi2({**fitted, name: point[name]}):.6e}"
                              for _, name, _ in axes)
            f.write(f"ROW: {i+1:5d}  {grid}   {scans} \n")


_DOCANA = "DOCUMENTATION:"
_DOCANA_END = "DOCUMENTATION_END:"
_PAD = "    "


def _cat_docana_notes(paths, config, droplist) -> List[str]:
    """The CAT_FILE_NOTES block embedded before DOCUMENTATION_END (:23890)."""

    body = [snana_version_comment()]
    if config.cat_prescale <= 1:
        body.append(f"COLUMN_APPEND:  {config.append_varname_missing} ")
        body.append(f"COLUMN_DROP:    {droplist} ")
    else:
        body.append(f"PRESCALE: {config.cat_prescale}")
    body.append("INPUT_CAT_FILES: ")
    body += [f" - {p} " for p in paths]
    return [f"{_PAD}CAT_FILE_NOTES:"] + [f"{_PAD}{_PAD}{n}" for n in body]


def _cat_comment_notes(tables, paths, config, droplist) -> List[str]:
    """The same provenance as a comment block when there is no DOCANA (:23923)."""

    lines = []
    # SNTABLE_VERSION_PHOTOMETRY accumulates one entry per file read, comma-joined.
    photometry = ",".join(t.meta["VERSION_PHOTOMETRY"] for t in tables
                          if t.meta.get("VERSION_PHOTOMETRY"))
    if photometry:
        lines.append(f"# VERSION_PHOTOMETRY: {photometry} ")
    lines.append(f"# {snana_version_comment()} ")
    lines.append("# Catenated data files: ")
    lines += [f"#   + {p} " for p in paths]
    if config.cat_prescale <= 1:
        lines.append(f"#\n# Appended columns: {config.append_varname_missing} ")
        lines.append(f"#\n# Dropped columns: {droplist} ")
    else:
        lines.append(f"# PRESCALE = {config.cat_prescale}")
    lines.append("#")
    return lines


def _cat_header(tables, paths, config, droplist) -> str:
    """write_cat_info (SALT2mu.c:25005).

    With a DOCUMENTATION block in the first input, copy it and embed the cat notes
    just before its end key; otherwise write the same provenance as comments.
    """
    docana = _read_docana(paths[0])
    if docana is not None:
        notes = _cat_docana_notes(paths, config, droplist)
        return "\n".join(docana + [""] + notes + [_DOCANA_END, "", ""]) + "\n"
    return "\n".join(_cat_comment_notes(tables, paths, config, droplist)) + "\n"


def _read_docana(path):
    """Lines of the leading DOCUMENTATION block, excluding its end key; None if absent."""

    out = []
    with open_text(path) as f:
        for line in f:
            stripped = line.rstrip("\n")
            words = stripped.split()
            key = words[0] if words else ""
            if not out and key != _DOCANA:
                return None
            if key == _DOCANA_END:
                return out
            out.append(stripped)
    return None


def _cat_droplist(varnames_by_file, kept) -> str:
    """Input columns present in some file but not carried into the output.

    catVarList_with_comma (SALT2mu.c:9784) joins them with commas.
    """
    seen = dict.fromkeys(n for names in varnames_by_file for n in names)
    kept = set(kept)
    return ",".join(n for n in seen if n not in kept)


def write_cat_fitres(path, tables, paths, config) -> int:
    """Concatenate the input rows and return how many were written.

    write_fitres_driver under cat_only (SALT2mu.c:24172): no cuts, no fit, and no
    appended BBC block (define_varnames_append returns early, :24688). Values are
    re-emitted as their original tokens with "-9.0" where a file lacks the column,
    matching write_fitres_line (:24501), so the read must be ``as_text``.
    """

    varnames_by_file = [t.varnames_by_file[0] for t in tables]
    varnames = output_varnames(varnames_by_file, config)
    droplist = _cat_droplist(varnames_by_file, varnames)
    prescale = config.cat_prescale
    nline = nwrite = 0

    with open(path, "w") as f:
        f.write(_cat_header(tables, paths, config, droplist))
        f.write("VARNAMES: " + "".join(f"{n} " for n in varnames) + "\n\n")
        for table in tables:
            columns = [table.get(n) for n in varnames]
            for irow in range(table.size):
                if table.rowkey[irow] not in _ROW_KEYS:
                    continue
                # the prescale counter runs across all files, and only over rows
                # that got this far (:23283 rejects other keys before :23290)
                nline += 1
                if nline % prescale != 0:
                    continue
                words = "".join(
                    f"{MISSING_TOKEN if col is None else col[irow]} "
                    for col in columns)
                f.write(f"{table.rowkey[irow]} {words}\n")
                nwrite += 1
    return nwrite


def yaml_summary(cell, output, cpu_minutes):
    """Gather write_yaml_info's counts (SALT2mu.c:23571).

    The two reject counts are not the bit tallies they look like. SNANA's
    NREJECT_CUTWIN counts events failing *any* cut that is not the biasCor
    (cutmask_noBCOR > 0, :18665), and NREJECT_BIASCOR counts only events that
    would otherwise have passed, so an event cut for two reasons appears once.
    """

    tv, maps, config = cell.tv, cell.summary_maps, cell.config
    biascor_bit = 1 << CUTBIT_BIASCOR
    other_bits = tv.cutbits & ~biascor_bit
    only_biascor = (tv.cutbits & biascor_bit) != 0

    summary = YamlSummary(
        nevt={"DATA": int(tv.cutmask.sum())},
        nreject_biascor=int((only_biascor & (other_bits == 0)).sum()),
        isdata_real=not tv.sim_truth,
        cpu_minutes=cpu_minutes,
        ncrazy=int(output.crazy_m0.sum()),
    )
    if output.probcc is not None:
        # 0/0 writes nan, as SNANA (write_yaml_info, SALT2mu.c:23775)
        summary.contam_data = output.contam_data
        with np.errstate(invalid="ignore", divide="ignore"):
            if tv.sim_truth:
                passing = tv.cutmask
                tmpl = np.asarray(tv.sim_truth["SIM_TEMPLATE_INDEX"])
                xncc_true = float((passing & (tmpl != 0)).sum())
                xntot = float(passing.sum()
                              - (passing & force_probcc0(tv, config)).sum())
                summary.contam_true = float(np.divide(xncc_true, xntot))
                summary.contam_true_err = float(
                    np.divide(np.sqrt(xncc_true), xntot))
    if maps is None:
        # NDATA_REJECT_CUTWIN is summed over biasCor samples (:22486); with none
        # to sum over SNANA reports 0 however many events were cut
        return summary

    ids = [s.idsample for s in maps.samples]
    summary.sample_ids = [-1] + ids
    summary.sample_names = ["ALL"] + [s.name for s in maps.samples]

    def by_sample(idsample, keep, total):
        return [total] + [int((keep & (idsample == i)).sum()) for i in ids]

    summary.nreject_cutwin = int(sum(
        ((other_bits != 0) & (tv.idsample == i)).sum() for i in ids))
    summary.nevt_by_sample["DATA"] = by_sample(
        tv.idsample, tv.cutmask, summary.nevt["DATA"])
    summary.nreject_cutwin_by_sample = by_sample(
        tv.idsample, other_bits != 0, summary.nreject_cutwin)
    summary.nreject_biascor_by_sample = by_sample(
        tv.idsample, only_biascor & (other_bits == 0), summary.nreject_biascor)

    for kind, counts in (("BIASCOR", maps.biascor_counts),
                         ("CCPRIOR", cell.ccprior_counts)):
        n = counts[0] if counts else 0
        summary.nevt[kind] = n
        if kind == "BIASCOR" and maps.biascor_nevt_total is not None:
            summary.nevt[kind] = maps.biascor_nevt_total
        summary.nevt_by_sample[kind] = list(counts) if counts else [0] * (len(ids) + 1)
    return summary


def write_outputs(fitres_path, result, output, zbins, tv, accept):
    """Every file the BBC program writes for one fit, beside ``fitres_path``:
    FITRES, then M0DIF, COV and CSV unless cutmask_write=-9, then the YAML.
    ``output.config`` is the resolved config the fit ran with, blinded where
    asked, which is what the headers report."""
    config = output.config
    passthrough = {n: np.asarray(a)[accept] for n, a in tv.passthrough.items()}
    write_fitres(fitres_path, output, zbins, result, config,
                 passthrough=passthrough, warnings=output.warnings)
    # cutmask_write=-9 keeps the FITRES fit-results block and writes none of the
    # binned outputs (write_M0_fitres :22701, write_M0_csv :22821,
    # write_M0_cov :22882)
    stem = str(fitres_path)[:-len(".FITRES")]
    if config.cutmask_write != -9:
        write_m0dif(f"{stem}.M0DIF", zbins, output, result, config,
                    warnings=output.warnings)
        write_cov(f"{stem}.COV", result.cov_m0)
        if config.write_csv:
            write_m0_csv(f"{stem}.CSV", zbins, output, config)
    if config.write_yaml:          # SNANA default is off (SALT2mu.c:6485)
        write_yaml(f"{stem}.YAML", output.yaml, result, config,
                   timings=output.timings)
