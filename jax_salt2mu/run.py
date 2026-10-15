"""End-to-end BBC driver: read FITRES -> fit -> write M0DIF/FITRES/COV."""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import jax.numpy as jnp
import numpy as np

from jax_snlc_core.io.fitres_reader import read_fitres
from jax_snlc_core.io.survey_def import load_survey_ids
from jax_snlc_core.models import standardised_model
from jax_snlc_core.physics.constants import LIGHT_KM
from jax_snlc_core.utils import snana_log as slog
from jax_snlc_core.utils.error_handling import SNANAError

from .config import (BBCConfig, BBC_NUISANCE_SPEC, H11_PARAMS,
                     M0FITFLAG_CONSTANT,
                     M0_HALFWIDTH, M0_HALFWIDTH_FIXED,
                     M0_HALFWIDTH_ZPOLY, MASK_BIASCOR_CELLMAP, MASK_BIASCOR_MUCOVADD,
                     parval, use_gamma0)
from .distances import (bias_parameters, mu_at, mu_coef, muerr_vpec_sq,
                        mu_variance_scaled)
from .fitter import BBCResult, FitSpec, run_fit
from .likelihood import LikelihoodInputs, event_chi2, mures_muerrsq_fn
from .results import (CRAZY_ERRMAX, CRAZY_ERRMIN_FRAC, CRAZY_SIGINT_REF,
                      YamlSummary, compute_output)
from .tablevar import (CUTBIT_BIASCOR, CUTBIT_CHI2, CUTBIT_MINBIN, CUTBIT_z,
                       force_probcc0, match_cid_selection)
from .zbins import ZBinGrid, make_zbins
from .biascor_build import BiasCorMaps
from .preparation import (read_cells, prepare_biascor_maps, prepare_ccpriors, prepare_data, format_data,
                          interpolate_corrections, select_data)
from .output import (output_path, write_cat_fitres, write_chi2grid, write_cov,
                     write_fitres, write_m0_csv, write_m0dif,
                     write_yaml)
from .ccprior import CHI2_BEAMS_NOPROB, build_h11prior, ccprior_data

logger = logging.getLogger(__name__)


def _m0_halfwidth(config):
    """M0 bound half-width about M0_DEFAULT (SALT2mu.c:2627-2634)."""
    if config.uM0 == M0FITFLAG_CONSTANT:
        return M0_HALFWIDTH_FIXED
    if config.muref_zpoly:
        return M0_HALFWIDTH_ZPOLY
    return M0_HALFWIDTH


def _zmu_solve(muref, config):
    """Invert the reference cosmology mu(z)=muref per z bin (SNANA zmu_solve): the
    z reported in the M0DIF. Interpolated on a fine grid over the fit z range."""
    zgrid = np.linspace(max(config.zmin * 0.5, 1e-4),
                        config.zmax * 1.2, 200_000)
    return np.interp(np.asarray(muref), mu_at(zgrid, config), zgrid)


def _yaml_summary(cell, output, cpu_minutes):
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


def build_inputs(tv, izbin, config, model, accept, biascor=None,
                 zM0=None, idsample=None, gammadm_offset=0.0) -> LikelihoodInputs:
    m = accept
    vpec_sq = muerr_vpec_sq(tv.zhd[m], tv.zmuerr[m])
    # compute_more_INFO_DATA (SALT2mu.c:9463): start values; sigint_fix[0] overrides (:20852)
    sigmB = (config.sigint_fix[0] if config.sigint_fix
             else config.sigint)
    start = {n: parval(config, n) for n in model.nuisance_spec}
    v_start = np.tile(mu_coef(model, start, config._u_flags), (int(m.sum()), 1))
    inp = LikelihoodInputs(
        lcpar={k: jnp.array(v[m]) for k, v in tv.lcpar.items()},
        z=jnp.array(tv.zhd[m]), logmass=jnp.array(tv.logmass[m]),
        covmat_fit=jnp.array(tv.covmat_fit[m]), mumodel=jnp.array(tv.mumodel[m]),
        izbin=jnp.array(izbin[m]),
        muerr_vpec_sq=jnp.array(vpec_sq), valid=jnp.ones(int(m.sum())),
        idsample=jnp.asarray((tv.idsample if idsample is None else idsample)[m]),
        lensing_zpar=config.lensing_zpar,
        muerrsq_last=jnp.array(
            mu_variance_scaled(v_start, tv.covmat_fit[m], tv.zhd[m],
                               config.lensing_zpar, sigmB * sigmB) + vpec_sq),
        gammadm_offset=gammadm_offset,
    )
    if zM0 is not None:
        izbin = np.asarray(inp.izbin)
        # same "has events" test results.py uses for the M0 average
        isfloat = np.bincount(izbin[izbin >= 0], minlength=len(zM0)) > 0
        iz1, zfrac = _m0_interp_weights(tv.zhd[m], izbin, zM0, isfloat)
        inp = inp._replace(izbin_nbr=jnp.asarray(iz1), m0_zfrac=jnp.asarray(zfrac))
    if biascor is not None:
        inp = inp._replace(coeff_avg=biascor["coeff_avg"],
                           coeff_binsize=biascor["coeff_binsize"])
        for name in ("fitparbias_nodes", "mucovscale_nodes", "mucovadd_nodes"):
            if biascor[name] is not None:
                inp = inp._replace(**{name: jnp.asarray(biascor[name][m])})
        # Output-only biasCor_nevt and biasCorErr_mu.
        for name in ("fitparbias_nevt", "fitparbias_err"):
            if biascor[name] is not None:
                inp = inp._replace(**{name: np.asarray(biascor[name][m])})
    return inp


def check_vpec_sign(tv, config, model):
    """Abort if the peculiar velocities look sign-inverted (SALT2mu.c:4131).

    Measures RMS(MURES) over a low-z window twice: at the reported zHD, and at zHD
    recomputed with the vpec sign flipped. A smaller RMS after flipping means the
    input's vpec convention is inverted, which biases every low-z distance; SNANA
    treats that as fatal rather than fitting it.

    Runs on every job: the window defaults to (0.01, 0.05) and only zwin_vpec_check=0,0
    switches it off.

    Returns ``(nsn, rms_nominal, rms_flip)``, or None where SNANA measures nothing
    (the check disabled, or fewer than 70 qualifying events). The caller
    ignores it; it is what lets a test compare against SALT2mu's printed
    values.
    """
    zwin = config.zwin_vpec_check
    if zwin[1] < 0.0001:
        return None

    zhd, zcmb, vpec = tv.zhd, tv.zcmb, tv.vpec
    sel = (tv.cutmask & (zhd >= zwin[0]) & (zhd <= zwin[1]) & (vpec != 0.0))
    if tv.is_photoz is not None:
        sel = sel & ~tv.is_photoz          # zSPEC only (:4143)
    nsn = int(sel.sum())
    if nsn < 70:                           # too few to measure an RMS (:4236)
        return None

    zpec = np.abs(vpec[sel]) / LIGHT_KM
    # zHD < zCMB means vpec was subtracted, so undoing it moves z the other way
    sgn_flip = np.where(zhd[sel] < zcmb[sel], 1.0, -1.0)
    # SNANA runs this before the fit, on INPUTS.parval, the initial values, which
    # p<N> overrides (:4147)
    nuis = {n: v[0] for n, v in model.nuisance_spec.items()}
    nuis.update(config.nuisance_val)
    lcdict = {n: tv.lcpar[n][sel] for n in model.lcpar_names}
    host = {"z": zhd[sel], "gamma_var": tv.logmass[sel]}
    mu = np.asarray(model.compute_mu(lcdict, nuis, host, config._u_flags,
                                     apply_mass_step=False))

    rms = []
    for ztest in (zhd[sel], zhd[sel] + sgn_flip * 2.0 * zpec):
        mures = mu - model.m0_reference - mu_at(ztest, config)
        rms.append(float(np.sqrt(np.mean((mures - mures.mean()) ** 2))))

    slog.line(logger,
              f"check_vpec_sign over {zwin[0]:.3f}<z<{zwin[1]:.3f} ({nsn} events): "
              f"RMS(MURES) nominal={rms[0]:.4f}, flip-vpec={rms[1]:.4f}")
    if rms[1] < rms[0]:
        raise SNANAError(
            f"RMS(MURES) is smaller with sign-flip of vpec "
            f"({rms[1]:.4f} < {rms[0]:.4f}); check the VPEC sign convention",
            function_name="check_vpec_sign")
    return nsn, rms[0], rms[1]


def _exempt_nobiascor(ok, tv, maps, config):
    """surveylist_nobiascor: those samples are fit without requiring a biasCor
    (DOFLAG_BIASCOR=0, SALT2mu.c:10119)."""
    names = {n.upper() for n in config.surveylist_nobiascor}
    if not names:
        return ok
    exempt_ids = {s.idsample for s in maps.samples
                  if s.surveygroup.upper() in names}
    if not exempt_ids:
        return ok
    return ok | np.isin(tv.idsample, list(exempt_ids))


def _warn_biascor_loss(ok, tv, maps, config, model):
    """BCOR_LOSS warning above frac_warn_nobiascor (SALT2mu.c:24797), plus the
    ndump_nobiascor listing of individual events with no valid biasCor (:11587)."""
    ndump = config.ndump_nobiascor
    if ndump > 0:
        for cid in tv.name[~ok & tv.cutmask][:ndump]:
            slog.warn(logger, "BBC", f"no valid biasCor for CID={cid}")
    lines = []
    frac_warn = config.frac_warn_nobiascor
    warned = False
    for smp in maps.samples:
        # counted among events passing the other cuts, as SNANA does
        in_sample = (tv.idsample == smp.idsample) & tv.cutmask
        ntot = int(in_sample.sum())
        if ntot == 0:
            continue
        nrej = int((in_sample & ~ok).sum())
        frac = nrej / ntot
        warn = "***** WARNING *****" if frac > frac_warn else ""
        line = (f"BCOR_LOSS: {nrej} of {ntot} events ({100*frac:.1f} %) "
                f"have no biasCor for {smp.name}   {warn}")
        lines.append(line)
        if warn:
            slog.warn(logger, "BBC", line)
            warned = True
    if warned:
        widths = ":".join(f"{axis}bin=<width>" for axis in model.biascor_binning)
        slog.warn(logger, "BBC",
                  f"an event needs min_per_cell_biascor={config.min_per_cell_biascor} "
                  f"biasCor events in its cell; the cells are "
                  f"{model.__name__}.biascor_binning {model.biascor_binning}, "
                  f"widened per sample by surveygroup_biascor's STRINGOPT, "
                  f"e.g. SAMPLE(zbin=<width>:{widths})")
    return lines


def floated_and_defaults(model, config, *, ccprior=None):
    defaults = {n: v[0] for n, v in model.nuisance_spec.items()}
    defaults["scalePCC"] = BBC_NUISANCE_SPEC["scalePCC"][0]
    if config.use_h11:
        for name in H11_PARAMS:
            defaults[name] = BBC_NUISANCE_SPEC[name][0]
    defaults.update(config.nuisance_val)
    floated = [name for name, on in config.nuisance_float.items()
               if on and name in defaults]
    # scalePCC never floats without CC contamination (SALT2mu.c:20230/19725); H11
    # floats it (INPUTS.ipar[IPAR_scalePCC]=1, SALT2mu.c:22143).
    perfect = config.force_pia is not None and config.force_pia < 0
    if (ccprior is None and not config.use_h11) or perfect:
        floated = [n for n in floated if n != "scalePCC"]
    if config.use_h11:
        # SALT2mu enables all six coefficients (NPAR_H11_USER remains zero).
        requested = {k: config.nuisance_float[k] for k in H11_PARAMS
                     if k in config.nuisance_float}
        if (requested and not config.fixpar_all
                and not all(requested.get(k, False) for k in H11_PARAMS)):
            slog.warn(logger, "BBC", "H11 floats all six polynomial coefficients, "
                      "as in SNANA; individual H11 float settings do not restrict the fit")
        floated = [k for k in floated if k not in H11_PARAMS] + list(H11_PARAMS)
        if "scalePCC" not in floated:
            floated.append("scalePCC")
    if config.fixpar_all:
        frozen = {n: parval(config, n) for n in defaults}
        _warn_held_at_parval(frozen, config)
        return [], frozen            # every nuisance frozen; M0 still floats
    # a fixed model nuisance holds parval, not the model's default (SALT2mu.c:22354)
    held = {}
    for name in model.nuisance_spec:
        if name not in floated and name not in config.nuisance_val:
            held[name] = defaults[name] = parval(config, name)
    _warn_held_at_parval(held, config)
    return floated, defaults


def _warn_held_at_parval(held, config):
    """A held nuisance the input gave no value takes SNANA's parval, zero
    for alpha and beta, which is a fit with no correction; SALT2mu does
    this silently."""
    unset = {n: v for n, v in held.items() if n not in config.nuisance_val}
    if unset:
        slog.warn(logger, "BBC", "held with no p<N> value, at SNANA's parval: "
                  + ", ".join(f"{n}={v:g}" for n, v in unset.items()))


def _chi2max_keep(spec, inputs, config, m0_reference, chi2max):
    """applyCut_chi2max (SALT2mu.c:3347-3487): evaluate the fit chi2 at the
    initial parameters, shift M0 by the pIa-weighted H0-marg offset
    (:3411-3432), re-evaluate at the bound-clamped values MINUIT holds, and
    keep chi2 <= the per-event threshold. Under a CC prior the per-event chi2
    is the BEAMS value (:3353-3354), not a Hubble residual.

    Both passes run with the FCN's final-iteration flag (DO_H0marg forces
    iflag=3, :5024), so each stores its own mu-variance as the reference the
    next chi2's log term divides by (:5660). The cut therefore reads the
    second pass, whose log term is the ratio to the first pass's variance.
    Returns ``(keep, muerrsq_last)``, the reference the fit then starts from.
    """
    spec = replace(spec, floated=(), mode="snana_secant")
    model, nzbin = spec.model, spec.nzbin
    # every parameter the FCN reads starts at SNANA's parval (:5192, :6701)
    bounds = {n: (lo, hi) for n, (_, lo, hi, _) in
              {**BBC_NUISANCE_SPEC, **model.nuisance_spec}.items()}

    def _nuis(clamp):
        # the chi2 pass sees the values MINUIT clamps into the fit bounds
        nuis = {n: float(np.clip(parval(config, n), lo, hi)) if clamp
                else float(parval(config, n))
                for n, (lo, hi) in bounds.items()}
        nuis["sigint"] = config.sigint
        return nuis

    def _eval(nuis, m0, muerrsq_last):
        fn = mures_muerrsq_fn(spec, inputs, nuis, "snana_secant")
        mures, muerrsq = fn(jnp.full(nzbin, m0), nuis["sigint"])
        probcc, chi2 = event_chi2(spec, inputs, nuis, mures, muerrsq,
                                  muerrsq_last)
        return (np.asarray(mures), np.asarray(muerrsq),
                np.asarray(probcc), np.asarray(chi2))

    last = jnp.asarray(inputs.muerrsq_last)
    mures, muerrsq, probcc, chi2 = _eval(_nuis(clamp=False), m0_reference, last)
    pia = np.maximum(1.0 - probcc, 1.0e-6)
    if inputs.beams:
        pia = np.where(chi2 == CHI2_BEAMS_NOPROB, 0.0, pia)
    wgt = pia / muerrsq
    muoff = np.sum(mures * wgt) / np.sum(wgt) if np.sum(wgt) > 0 else 0.0
    last = jnp.asarray(muerrsq)
    _, muerrsq, _, chi2 = _eval(_nuis(clamp=True), m0_reference + muoff, last)
    return chi2 <= chi2max, jnp.asarray(muerrsq)


def _chi2max_per_event(tv, accept, config):
    """The accepted events' chi2max thresholds: per-survey values where
    chi2max(SURVEYLIST) named the survey, the global value elsewhere."""
    thresholds = np.full(int(np.sum(accept)), config.chi2max)
    if config.chi2max_surveys:
        name_to_id, _ = load_survey_ids()
        idsurvey = tv.idsurvey[accept]
        for survey, value in config.chi2max_surveys.items():
            if survey not in name_to_id:
                raise SNANAError(f"Invalid SURVEY={survey} in chi2max(...); "
                                 f"check SURVEY.DEF",
                                 function_name="run_bbc_fit")
            thresholds[idsurvey == name_to_id[survey]] = value
    return thresholds


def _m0_interp_weights(zhd, izbin, zM0, isfloat):
    """Per-event (iz1, zfrac) for uM0=2, so M0 is linear between bin knots.

    fcn_M0 (SALT2mu.c:5826): the neighbour is the next bin up at the low edge, down
    at the high edge, else whichever side the event's z falls on relative to its own
    bin's knot. A neighbour whose M0 is fixed leaves the event on a flat M0 (:5859),
    which is zfrac=0 here.
    """
    nbin = len(zM0)
    iz0 = np.clip(izbin, 0, nbin - 1)
    step = np.where(zhd > zM0[iz0], 1, -1)
    iz1 = np.clip(np.where(iz0 == 0, iz0 + 1,
                           np.where(iz0 == nbin - 1, iz0 - 1, iz0 + step)),
                  0, nbin - 1)
    dz = zM0[iz1] - zM0[iz0]
    safe = dz != 0.0
    zfrac = np.where(safe, (zhd - zM0[iz0]) / np.where(safe, dz, 1.0), 0.0)
    return iz1, np.where(isfloat[iz1] & (izbin >= 0), zfrac, 0.0)


def _izbin_for(tv, zbins, config):
    """Per-event z bin. izbin_from_cid_file overrides the z-derived bin with the
    IZBIN column of cid_select_file, matching CID and any survey/field columns,
    where the selected bin is >= 0
    (SALT2mu.c:3075)."""
    izbin = zbins.index(tv.zhd)
    if config.izbin_from_cid_file and config.cid_select_file:
        _, selected_bins = match_cid_selection(tv, config.cid_select_file)
        izbin = np.where(selected_bins >= 0, selected_bins, izbin)
    return izbin


def _sqsigint_fix(config: BBCConfig, samples, idsample, acc) -> np.ndarray:
    """Per-event sigma_int^2 from ``sigint_fix``, indexed by IDSAMPLE
    (parse_sigint_fix, SALT2mu.c:20820). One value applies to every IDSAMPLE;
    otherwise the count must match."""
    vals = config.sigint_fix
    if len(vals) == 1:
        vals = vals * len(samples)
    elif len(vals) != len(samples):
        raise SNANAError(
            f"sigint_fix has {len(vals)} values but there are {len(samples)} IDSAMPLEs",
            function_name="run_bbc_fit")
    return np.asarray(vals, dtype=float)[idsample[acc]] ** 2


@contextmanager
def _section(timings: Dict[str, float], name: str):
    """Time one section of a cell into ``timings[name]``, in seconds. The
    caller's device work is waited for at the end, as the fit blocks on its
    results, so a section's compute is charged to it and not to whichever
    later section first waits."""
    start = time.perf_counter()
    yield
    timings[name] = timings.get(name, 0.0) + time.perf_counter() - start


@dataclass
class BBCCell:
    """One fit of a grid, prepared: what the fit reads and what the writers
    need afterwards. ``accept`` is the mask of fitted events (a subset of
    ``tv.cutmask``); the per-event output arrays are aligned to it, so CID
    and passthrough columns must use ``tv.<col>[accept]``, not
    ``tv.cutmask``."""
    config: BBCConfig
    model: type
    zbins: ZBinGrid
    tv: object
    accept: np.ndarray
    inputs: LikelihoodInputs
    spec: FitSpec
    maps: Optional[BiasCorMaps]
    summary_maps: Optional[BiasCorMaps]
    nevt_zfit: np.ndarray          # NEVT_zFIT, the M0DIF's NFIT (SALT2mu.c:3091)
    timings: Dict[str, float]      # seconds per section, prepare through output
    warnings: list = field(default_factory=list)
    ccprior_counts: tuple = ()
    ccprior_version: str | None = None


def _reject(tv, accept, drop, bit):
    """Set ``bit`` on the accepted events ``drop`` selects, and drop them."""
    if not drop.any():
        return accept
    tv.cutbits[np.where(accept)[0][drop]] |= 1 << bit
    accept = accept.copy()
    accept[accept] = ~drop
    return accept


def _apply_zbin_cuts(tv, accept, izbin, zbins, config):
    """setup_zbins_fit (SALT2mu.c:3042), on the bin the fit uses: a bin index
    outside the grid fails CUTBIT_z (:3087), and a bin holding fewer than
    min_per_zbin events fails CUTBIT_MINBIN (:3127). CUTWIN NONE leaves the
    second but not the first (setbit_CUTMASK, :19016). Also returns NEVT_zFIT,
    the per-bin count taken before the min_per_zbin pass, which the M0DIF's
    NFIT column reports (SALT2mu.c:3091, :24349)."""
    out_of_grid = (izbin < 0) | (izbin >= zbins.nbin)
    accept = _reject(tv, accept, out_of_grid[accept], CUTBIT_z)
    izb = izbin[accept]
    nevt_zfit = np.bincount(izb, minlength=zbins.nbin)
    sparse = nevt_zfit < config.min_per_zbin
    if sparse.all():
        raise SNANAError(
            f"no z bin with at least {config.min_per_zbin} SN",
            function_name="prepare_cell")
    if config.min_per_zbin <= 1 or config.cutwin_disable:
        return accept, nevt_zfit
    return _reject(tv, accept, sparse[izb], CUTBIT_MINBIN), nevt_zfit


def fit_spec(config, model, maps=None, ccprior=None, cc_ncell=0):
    """The static fit program, determined without selecting or assembling events."""
    floated, defaults = floated_and_defaults(model, config, ccprior=ccprior)
    return FitSpec(
        model=model, floated=tuple(floated), defaults=tuple(sorted(defaults.items())),
        nzbin=make_zbins(config).nbin, mode=config.sigint_mode,
        with_log=config.fitflag_sigmb == 2,
        u_flags=tuple(sorted(config._u_flags.items())),
        scalepcc_bayes=config._u_flags.get("u13", 0) == 2,
        bias_names=tuple(bias_parameters(model, config)) if maps is not None else (),
        use_gamma0=use_gamma0(model, config),
        cc_bias_names=tuple(bias_parameters(model, config)) if ccprior is not None else (),
        cc_ncell=cc_ncell,
    )


def prepare_cell(config: BBCConfig, tables=None) -> BBCCell:
    """Prepare a single cell through the same producers used by complete grids."""
    cells = read_cells({None: (config, tables)})
    maps = prepare_biascor_maps(cells)
    priors = prepare_ccpriors(cells, maps)
    data = prepare_data(cells)
    format_data(cells, data)
    biascor = interpolate_corrections(cells, data, maps)
    return prepare_cells(cells, data, maps, biascor, priors)[None]


def prepare_cells(cells, data, maps, biascor, priors):
    """Apply each request's local selection to prepared scientific results."""
    prepared = {}
    for key, (config, sources, _) in cells.items():
        tv = select_data(config, sources, data[key], maps[key])
        mapping = maps[key][1] if isinstance(maps[key], tuple) else maps[key]
        prepared[key] = _prepare_cell(config, tv, mapping, biascor[key], priors[key])
    return prepared


def _prepare_cell(config, tv, summary_maps, biascor, ccprior):
    """Select and assemble one fit from its prepared scientific inputs."""
    model = standardised_model(config.model_lcfit)
    zbins = make_zbins(config)
    timings = {}
    check_vpec_sign(tv, config, model)
    maps = summary_maps if summary_maps is not None and summary_maps.maps else None
    mubias_1d = None
    biascor_data = biascor
    warnings = []
    if summary_maps is not None:
        ok = biascor["valid"]
        if maps is not None:
            ok = _exempt_nobiascor(ok, tv, maps, config)
            warnings = _warn_biascor_loss(ok, tv, maps, config, model)
        else:
            mubias_1d = biascor["mubias"]
            biascor_data = None
        tv.cutbits[~ok] |= 1 << CUTBIT_BIASCOR
        if maps is None:
            warnings = _warn_biascor_loss(ok, tv, summary_maps, config, model)
    accept = tv.cutmask
    cc_prior, cc_ncell, cc_counts, cc_version = (ccprior if ccprior is not None
                                               else (None, 0, (), None))
    izbin = _izbin_for(tv, zbins, config)

    def assemble(acc):
        inp = build_inputs(
            tv, izbin, config, model, acc, biascor_data,
            zM0=maps.zM0 if maps is not None else None,
            idsample=(tv.idsample
                      if summary_maps is not None else None),
            gammadm_offset=maps.gammadm_offset if maps is not None else 0.0)
        # sigint_fix applies only on the cell-map path without muCOVADD (SALT2mu.c:10151, :13101)
        if (config.sigint_fix and config.fitflag_sigmb > 0
                and config.opt_biascor & MASK_BIASCOR_CELLMAP
                and not config.opt_biascor & MASK_BIASCOR_MUCOVADD):
            inp = inp._replace(sqsigint=jnp.asarray(_sqsigint_fix(
                config, summary_maps.samples, tv.idsample, acc)))
        if mubias_1d is not None:
            inp = inp._replace(mubias_const=jnp.asarray(mubias_1d[acc]))
        ncell = 0
        with _section(timings, "prior_inputs"):
            if cc_prior is not None:
                inp = inp._replace(ccprior=ccprior_data(cc_prior, tv, acc, maps,
                                                        model, config))
                ncell = cc_ncell
            elif config.use_h11:
                inp = inp._replace(h11prior=build_h11prior(tv, acc, inp, config, model))
        return inp, ncell

    accept, nevt_zfit = _apply_zbin_cuts(tv, accept, izbin, zbins, config)
    with _section(timings, "inputs"):
        inputs, ncell = assemble(accept)
    spec = fit_spec(config, model, maps, cc_prior, ncell)
    # chi2max: reject high-chi2 events pre-fit and rebuild on the reduced
    # sample, or deweight them in place under FITWGT0 (SALT2mu.c:3491);
    # skipped under CUTWIN NONE (:3396).
    if (config.chi2max <= 1.0e8 or config.chi2max_surveys
            or config.chi2max_fitwgt0) and not config.cutwin_disable:
        with _section(timings, "chi2max"):
            keep, muerrsq_last = _chi2max_keep(
                spec, inputs, config, model.m0_reference,
                _chi2max_per_event(tv, accept, config))
        # the pass leaves its own mu-variance as the reference the fit starts
        # from (:5660)
        inputs = inputs._replace(muerrsq_last=muerrsq_last)
        if config.chi2max_fitwgt0:
            inputs = inputs._replace(fitwgt0=jnp.asarray(~keep))
        elif not keep.all():
            # SNANA holds that reference per event, so the survivors keep
            # theirs through the rebuild on the reduced sample
            reference = np.zeros(len(tv.zhd))
            reference[accept] = np.asarray(muerrsq_last)
            accept = _reject(tv, accept, ~keep, CUTBIT_CHI2)
            # SNANA re-runs setup_zbins_fit after this cut (SALT2mu.c:3542)
            accept, nevt_zfit = _apply_zbin_cuts(tv, accept, izbin, zbins,
                                                 config)
            with _section(timings, "inputs"):
                inputs, _ = assemble(accept)
            inputs = inputs._replace(muerrsq_last=jnp.asarray(reference[accept]))

    return BBCCell(config, model, zbins, tv, accept, inputs, spec, maps, summary_maps,
                   nevt_zfit, timings, warnings, cc_counts, cc_version)


def grid_width(cells: Mapping[Any, BBCCell]) -> int:
    """The event count the grid's compiled functions are built for: its
    largest cell. Compatible cells can reuse compiled fits at this width.
    """
    return max(int(cell.inputs.z.shape[0]) for cell in cells.values())


def fit_cells(cells: Mapping[Any, BBCCell], fit_widths) -> Dict[Any, BBCResult]:
    """Fit every cell at its fit program's fixed event capacity."""
    results = {}
    for key, cell in cells.items():
        config, model = cell.config, cell.model
        fitflag = config.fitflag_sigmb      # resolved, so 1 or 2 with a biasCor
        width = fit_widths[cell.spec]
        if len(cell.inputs.z) > width:
            raise SNANAError(f"BBC cell {key}: fit capacity {width} is smaller than "
                             f"{len(cell.inputs.z)} accepted rows", function_name="fit_cells")
        with _section(cell.timings, "fit"):
            results[key] = run_fit(
                cell.spec, cell.inputs, model.m0_reference,
                width,
                sigint_init=config.sigint, redchi2_tol=config.redchi2_tol,
                fitflag_sigmb=fitflag, sigint_step1=config.sigint_step1,
                sigint_floor=config.sigint_floor,
                sigint_ceiling=config.sigint_ceiling,
                stop_covfix=bool(config.sigint_fix),
                dchi2red_dsigint=config.dchi2red_dsigint,
                m0_halfwidth=_m0_halfwidth(config))
        cell.timings.update(results[key].timings)
        cell.timings["fit_iterations"] = float(results[key].niter)
    return results


def finish_cell(cell: BBCCell, result: BBCResult, n_pad: int, t_start: float, *, write=True):
    """The fit's outputs: (result, output, zbins, tv, accept); ``t_start``
    is when the cell's preparation began, for the YAML's CPU minutes."""
    config, model, zbins, tv, accept, inputs = (
        cell.config, cell.model, cell.zbins, cell.tv, cell.accept, cell.inputs)
    maps, summary_maps = cell.maps, cell.summary_maps
    if write and config.write_chi2grid > 0:
        write_chi2grid(output_path(config, ".CHI2GRID"), cell.spec, inputs,
                       result, config.write_chi2grid)
    with _section(cell.timings, "output"):
        output = compute_output(
            cell.spec, inputs, zbins, result, config, n_pad,
            dust=summary_maps is not None and summary_maps.dust_flag)
    output.timings = cell.timings
    # M0DIF zHD/MUREF: 1/muerrsq-weighted (via zmu_solve) with biasCor, else the
    # bin center and its reference mu (SALT2mu.c:23927).
    if config.opt_biascor > 0:
        output.zbin_z = np.where(output.nfit > 0,
                                 _zmu_solve(output.muref, config), zbins.centers)
    else:
        output.muref = mu_at(zbins.centers, config)
        output.zbin_z = zbins.centers.copy()
    # NFIT is the pre-min_per_zbin count (SALT2mu.c:3091)
    output.nfit = cell.nevt_zfit

    if output.probcc is not None:
        # fitted-subset counters for the contamination lines (NSNSPEC_IA and
        # NSNFIT_TRUECC, SALT2mu.c:5607/5657)
        valid = output.chi2 != CHI2_BEAMS_NOPROB
        output.nsnspec_ia = (int((force_probcc0(tv, config)[accept] & valid).sum())
                             if config.varname_pia else 0)   # REQUIRE_pIa (:9301)
        # spec-confirmed Ia are excluded from the photo-classified subset
        # (write_fitres_misc, SALT2mu.c:24968-24986); on an all-spec sample the
        # 0/0 ratio prints as nan, as SNANA's C division does
        photo = (~force_probcc0(tv, config)[accept] if config.varname_pia
                 else np.ones_like(valid))
        output.sum_probcc = float(np.sum(output.probcc[valid]))
        output.sum_probia = float(np.sum((1.0 - output.probcc)[valid & photo]))
        with np.errstate(invalid="ignore", divide="ignore"):
            output.contam_data = float(np.divide(
                output.sum_probcc, output.sum_probia + output.sum_probcc))
        if tv.sim_truth:
            tmpl = np.asarray(tv.sim_truth["SIM_TEMPLATE_INDEX"])[accept]
            output.nsnfit_truecc = int((tmpl != 0).sum())

    # provenance SNANA opens both output files with (write_version_info)
    output.isdata_real = not tv.sim_truth
    versions = {"VERSION_PHOTOMETRY(DATA)": tv.version_photometry}
    if summary_maps is not None and summary_maps.biascor_counts:
        versions["VERSION_PHOTOMETRY(BIASCOR)"] = \
            summary_maps.version_photometry or ""
        output.biascor_cosmology = summary_maps.sim_cosmology
    if cell.ccprior_version is not None:
        versions["VERSION_PHOTOMETRY(CCPRIOR)"] = cell.ccprior_version or ""
    output.version_photometry = versions

    if summary_maps is not None and summary_maps.biascor_counts:
        m = summary_maps
        nalpha, nbeta = (list(m.coeff_nbin) + [0, 0])[:2]
        nsim = m.biascor_nevt_total
        if nsim is None:
            nsim = m.biascor_counts[0]
        note = (f"(N_alpha x N_beta = {nalpha} x {nbeta})"
                if model.biascor_coeffs else "")
        output.biascor_summary = (m.ndim, nsim, note, m.sigint_biascor)

    output.yaml = _yaml_summary(cell, output, (time.time() - t_start) / 60.0)

    ncrazy = int(output.crazy_m0.sum())
    if ncrazy:
        # SNANA writes this into both .M0DIF and .FITRES (SALT2mu.c write_NWARN)
        output.warnings.append(
            f"WARNING(SEVERE): {ncrazy} fitted M0ERR have crazy values "
            "after final fit. ")
        slog.warn(logger, "BBC",
                  f"{ncrazy} z bin(s) with M0ERR outside [{CRAZY_ERRMIN_FRAC} * "
                  f"{CRAZY_SIGINT_REF}/sqrt(N), {CRAZY_ERRMAX}] (crazy_M0_errors): "
                  f"a sample scattering less than "
                  f"sigint={CRAZY_SIGINT_REF} trips the floor while sigint is not "
                  f"fitted (fitflag_sigmb={config.fitflag_sigmb})")
    output.warnings += cell.warnings

    slog.kv(logger, "N(data)", int(tv.cutmask.sum()))
    for name in result.nuis_err:
        slog.kv(logger, name, f"{result.nuisance[name]:.4f}")
    slog.kv(logger, "sigint", f"{result.sigint:.4f}")
    slog.kv(logger, "seconds", "  ".join(
        f"{name} {value:.2f}" for name, value in cell.timings.items()
        if name != "fit_iterations"))
    return result, output, zbins, tv, accept


def run_bbc_grid(cells: Mapping[Any, Tuple[BBCConfig, Any]], *, write=True) -> Dict[Any, tuple]:
    """Prepare and fit a grid, sharing each distinct input and correction."""
    t_start = time.time()
    resolved = read_cells(cells)
    maps = prepare_biascor_maps(resolved)
    priors = prepare_ccpriors(resolved, maps)
    return run_prepared_grid(resolved, maps, priors, write=write, t_start=t_start)


def run_prepared_grid(cells, maps, priors, *, write=True, t_start=None,
                      fit_widths=None, cosmology_widths=None):
    """Fit one grid using maps and simulated priors already constructed."""
    if t_start is None:
        t_start = time.time()
    data = prepare_data(cells, cosmology_widths=cosmology_widths)
    format_data(cells, data)
    biascor = interpolate_corrections(cells, data, maps)
    return finish_grid(prepare_cells(cells, data, maps, biascor, priors),
                       write=write, t_start=t_start, fit_widths=fit_widths)


def finish_grid(prepared, *, write=True, t_start=None, fit_widths=None):
    """Fit and finish cells at supplied capacities, or the grid's largest cell."""
    if t_start is None:
        t_start = time.time()
    if fit_widths is None:
        n_pad = grid_width(prepared)
        fit_widths = dict.fromkeys((cell.spec for cell in prepared.values()), n_pad)
    results = fit_cells(prepared, fit_widths)
    return {key: finish_cell(cell, results[key], fit_widths[cell.spec], t_start, write=write)
            for key, cell in prepared.items()}


def run_bbc_fit(config: BBCConfig, tables=None, *, write=True):
    """One fit, a grid of one cell: (result, output, zbins, tv, accept).

    ``write=False`` suppresses diagnostic CHI2GRID files. Main output tables
    are written separately by :func:`write_outputs`.
    """
    return run_bbc_grid({None: (config, tables)}, write=write)[None]


def run_cat(config: BBCConfig) -> BBCConfig:
    """Concatenate the datafiles and stop (SALT2mu.c:2266)."""
    if not config.cat_file_out:
        raise SNANAError("cat_only needs cat_file_out=<file>",
                         function_name="write_fitres_driver")
    tables = [read_fitres(p, as_text=True) for p in config.datafile]
    n = write_cat_fitres(config.cat_file_out, tables, config.datafile, config)
    slog.kv(logger, "wrote", f"{n} events to {config.cat_file_out}")
    slog.done(logger, Path(config.cat_file_out).name)
    return config


def run_bbc(config: BBCConfig, cat_prescale=None, tables=None) -> BBCConfig:
    """The whole BBC program: fit, then write M0DIF, FITRES, COV, CSV and YAML.

    Returns the resolved config the fit ran with, blinded where asked.
    ``cat_prescale`` comes from a ``cat_only/<N>`` command-line argument.
    """
    if cat_prescale is not None:
        config.cat_only = True
        config.cat_prescale = cat_prescale
    slog.banner(logger, "SALT2mu / BBC")
    slog.kv(logger, "model_lcfit", config.model_lcfit)
    slog.kv(logger, "datafile", ", ".join(config.datafile) or "(none)")

    if not config.datafile and tables is None:
        slog.warn(logger, "BBC", "no datafile given; nothing to fit")
        return config

    if config.cat_only:
        return run_cat(config)

    fitres_path = output_path(config, ".FITRES")   # fail before fitting, not after
    result, output, zbins, tv, accept = run_bbc_fit(config, tables=tables)
    write_outputs(fitres_path, result, output, zbins, tv, accept)
    slog.done(logger, Path(config.prefix).name)
    return output.config


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
