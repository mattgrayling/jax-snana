"""CC-prior / BEAMS contamination model (SALT2mu.c prepare_CCprior:15900 +
the MNCHI2FUN BEAMS branch:4715).

The default ``opt_ccprior = GAUSS+UPDATE`` path bins bias-corrected CC-simulation
residuals in redshift and named model coordinates. Its per-cell Gaussian
(DMUAVG, DMURMS) is rebuilt every fit iteration from the current nuisance parameters
(so it is autodiff-clean). The data likelihood becomes BEAMS: ``-2 ln[P_Ia + P_CC]``.

The per-cell reduction and the BEAMS chi2 live in likelihood.py (they run inside
the jitted loss); this module builds the fixed init inputs: cell assignments,
per-node CC biasCor, mumodel, and pIa.
"""

from typing import Dict, NamedTuple, Tuple

import numpy as np

from jax_snlc_core.utils.error_handling import SNANAError

from .cellgrid import BinInfo
from .config import lcpar_windows, map_bins, parval
from .distances import mu_coef, mu_variance_scaled
from .tablevar import force_probcc0
from .biascor_apply import precompute_fitparbias_nodes
from .tablevar import CUTBIT_BIASCOR

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
              for name, count in map_bins(model, config, "ccprior_bins").items()))


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
    names = tuple(map_bins(model, config, "ccprior_bins"))
    cells_per_sample = _cells_per_sample(bins)
    ncell = cells_per_sample * max(len(maps.samples), 1)

    cc_fval, _, _, cc_ok = precompute_fitparbias_nodes(
        ccprior_tv, maps, model, config.min_per_cell_biascor,
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


def ccprior_data(prior, data_tv, data_accept, maps, model, config) -> CCPriorInputs:
    """``prior`` (from :func:`build_ccprior`) with the accepted data events'
    CC cells and P(Ia)."""
    bins = build_ccprior_bins(config, model)
    names = tuple(map_bins(model, config, "ccprior_bins"))
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
