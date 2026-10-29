"""Intrinsic brightness scatter: coherent, per-filter, phase and wavelength.

SNANA's G10 (SALT colour dispersion) and C11 (Chotard+2011 UBVRI correlations),
from sntools_genSmear.c. Per SN: a few gaussian draws define a phase-independent
DeltaM(lam_rest), applied to the SED as flux * 10^(-0.4*DeltaM): the per-lambda
application SNANA uses inside the SALT2 band integral (IFLAG_GENSMEAR_LAM).
Preparation uses NumPy; per-event draws and SED factors are JAX-traceable.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core import logging as log
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.utils import eval_genpoly, float_dtype
from jax_snlc_core.snana_files import sndata_root, read_two_column_file

logger = logging.getLogger(__name__)

# SNANA constants (sntools_genSmear.c / genmag_SALT2.h)
G10_LAMSEP_NODE = 800.0
G10_FUDGE_LAMSWITCH = 2157.3
G10_FUDGE_DSMEAR_DLAM = 1.08e-4
COLOR_DISP_MAX_DEFAULT = 2.0

C11_LAMCEN = np.array([2500.0, 3560.0, 4390.0, 5490.0, 6545.0, 8045.0])
C11_SIGMA = np.array([0.5900, 0.06001, 0.040034, 0.050014, 0.040017, 0.080007])
# Chotard thesis Table 14.2 (EWSi), fudged by Marriner to be invertible;
# v row/column zero by default (replaced for the farUV variants).
C11_RHO = np.array([
    [+1.000000, 0.000000,  0.000000,  0.000000,  0.000000,  0.000000],
    [0.000000, +1.000000, -0.118516, -0.768635, -0.908202, -0.219447],
    [0.000000, -0.118516, +1.000000, +0.570333, -0.238470, -0.888611],
    [0.000000, -0.768635, +0.570333, +1.000000, +0.530320, -0.399538],
    [0.000000, -0.908202, -0.238470, +0.530320, +1.000000, +0.490134],
    [0.000000, -0.219447, -0.888611, -0.399538, +0.490134, +1.000000],
])
C11_COV_SCALE = 1.3
C11_COV_DIAG_FUDGE = 1.0e-9
C11_IFILT_V = 0
C11_IFILT_U = 1


def scatter_factors(smear, randoms, wavelengths):
    """The same per-event intrinsic scatter on photometric and spectral SEDs."""
    offsets = jax.vmap(smear.delta_mag, in_axes=(0, None))(randoms, wavelengths)
    return jnp.power(10.0, -0.4 * offsets)


def interp_sinfun(val, val_ref0, val_ref1, fun_ref0, fun_ref1):
    """SNANA's interp_SINFUN (sntools.c:7922): sinusoid with zero slope at
    both edges. Traceable; assumes val_ref0 <= val <= val_ref1."""
    val_mean = 0.5 * (val_ref0 + val_ref1)
    val_dif = val_ref1 - val_ref0
    fun_mean = 0.5 * (fun_ref0 + fun_ref1)
    fun_dif = fun_ref1 - fun_ref0
    s = jnp.sin(jnp.pi * (val - val_mean) / val_dif)
    return fun_mean + 0.5 * fun_dif * s


def _read_info_scalars(info_path):
    """SIGMA_INT (or SIGMA_COH) and COLOR_DISP_MAX from SALT{2,3}.INFO.

    The raw-dispersion cap defaults to 5.0, a local constant of
    init_genSmear_SALT2 (sntools_genSmear.c:1082) distinct from the 2.0
    node-level cap; the INFO key overrides it. A missing SIGMA_INT/SIGMA_COH
    key is fatal (:1367-1374). The parenthesised SIGMA_INT(lamlist) form is
    read as its first sigma: SNANA's multi-bin interpolation branch is
    disabled upstream (:1590-1593), so only the first value is live.
    """
    sigcoh = None
    colour_disp_max = 5.0
    with open(info_path) as f:
        for line in f:
            stripped = line.split('#')[0].strip()
            if stripped.startswith(('SIGMA_INT:', 'SIGMA_COH:',
                                    'SIGMA_INT(', 'SIGMA_COH(')):
                sigcoh = float(stripped.split(':', 1)[1].split()[0])
            elif stripped.startswith('COLOR_DISP_MAX:'):
                colour_disp_max = float(stripped.split(':', 1)[1].split()[0])
    if sigcoh is None:
        raise SNANAError(
            f"Could not find SIGMA_INT or SIGMA_COH in {info_path}",
            function_name="_read_info_scalars")
    return sigcoh, colour_disp_max


COH_SIGMA_DEFAULT = 0.13   # bare 'COH' (init_genSmear_COH)
OIR_COV_SCALE = 1.3   # filter-COV from the lambda model (init_genSmear_OIR)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SmearModel:
    """Prepared intrinsic-scatter coefficients and their wavelength rule."""

    kind: str = field(metadata={"static": True})
    n_randoms: int = field(metadata={"static": True})
    wavelengths: object = None
    coefficients: object = None
    coherent_sigma: float | None = None
    wavelength_range: tuple | None = None

    def delta_mag(self, randoms, wavelengths):
        """Magnitude scatter for one event, using its Gaussian draws."""
        if self.kind == "coherent":
            return jnp.full_like(wavelengths, jnp.sum(self.coefficients * randoms))

        nodes = self.wavelengths
        n_nodes = nodes.shape[0]
        coherent = (0.0 if self.coherent_sigma is None
                    else self.coherent_sigma * randoms[n_nodes])
        idx = jnp.clip(jnp.searchsorted(nodes, wavelengths) - 1, 0, n_nodes - 2)
        if self.kind == "nodes":
            offsets = randoms[:n_nodes] * self.coefficients
            node_term = interp_sinfun(
                wavelengths, nodes[idx], nodes[idx + 1],
                offsets[idx], offsets[idx + 1])
            lo, hi = self.wavelength_range
            inside = (wavelengths > lo + 0.001) & (wavelengths < hi - 0.001)
            return coherent + jnp.where(inside, node_term, 0.0)

        offsets = self.coefficients @ randoms[:n_nodes]
        interp = interp_sinfun(
            jnp.clip(wavelengths, nodes[0], nodes[-1]),
            nodes[idx], nodes[idx + 1], offsets[idx], offsets[idx + 1])
        return coherent + jnp.where(
            wavelengths < nodes[0], offsets[0],
            jnp.where(wavelengths > nodes[-1], offsets[-1], interp))


def _c11_cholesky(far_uv):
    """C11's band covariance, Cholesky-decomposed (init_genSmear_Chotard11).

    far_uv replaces the v row/column with a scaled copy of u's, so the three
    variants differ only in that correlation.
    """
    covscale_v = {0: 0.0, 1: +0.999, 2: -0.999}[far_uv]

    rho = C11_RHO.copy()
    for j in range(6):
        if j != C11_IFILT_V:
            rho[C11_IFILT_V, j] = covscale_v * C11_RHO[C11_IFILT_U, j]
            rho[j, C11_IFILT_V] = covscale_v * C11_RHO[j, C11_IFILT_U]

    cov = rho * np.outer(C11_SIGMA, C11_SIGMA)
    cov[np.diag_indices(6)] += C11_COV_DIAG_FUDGE
    cov *= C11_COV_SCALE
    return np.linalg.cholesky(cov)


def oir_bands(model_dir):
    """An OIR model's band centres, Cholesky-decomposed covariance and coherent
    sigma (init_genSmear_OIR).

    Bands are sorted by wavelength before interpolation (sort_OIR_BANDS,
    sntools_genSmear.c:3102-3160), and the diagonal carries SNANA's 1e-9
    fudge before the covariance scale (:2830-2903). A COLOR_SIGMA or
    COLOR_CORMAT whose size disagrees with LAMCEN_BAND is fatal
    (:3089-3096).
    """
    lamcen, sigma, cormat, sigmacoh = _read_oir_info(model_dir)
    n = len(lamcen)
    if len(sigma) != n or cormat.shape != (n, n):
        raise SNANAError(
            f"OIR.INFO in {model_dir}: {n} LAMCEN_BAND entries but "
            f"{len(sigma)} COLOR_SIGMA values and a "
            f"{cormat.shape} COLOR_CORMAT", function_name="oir_bands")
    order = np.argsort(lamcen)
    lamcen = lamcen[order]
    sigma = sigma[order]
    cormat = cormat[np.ix_(order, order)]
    cov = cormat * np.outer(sigma, sigma)
    cov[np.diag_indices(n)] += C11_COV_DIAG_FUDGE
    cov *= OIR_COV_SCALE
    return lamcen, np.linalg.cholesky(cov), sigmacoh


def build_node_grid_smear(salt_model_dir, fig8, sed_lam_max, zmin):
    """G10: node grid plus a coherent term, from a SALT model directory.

    The node sigmas come from the model's colour-dispersion file; DeltaM is
    SINFUN-interpolated between nodes and zero outside the file's wavelength
    range (get_genSmear_G10). The dispersion file must cover the SED's red
    edge (sntools_genSmear.c:1234-1245), and the last node must be reachable
    at zmin (:1298-1314); both are fatal in SNANA.
    """
    salt_model_dir = Path(salt_model_dir)
    if (salt_model_dir / 'SALT3.INFO').exists():
        info_name = 'SALT3.INFO'
        disp_name = 'salt3_color_dispersion.dat'
    elif (salt_model_dir / 'SALT2.INFO').exists():
        info_name = 'SALT2.INFO'
        disp_name = 'salt2_color_dispersion.dat'
    else:
        raise SNANAError(
            f"GENMAG_SMEAR_MODELNAME G10 reads the colour dispersion a SALT "
            f"model directory ships; {salt_model_dir} carries no SALT2.INFO "
            f"or SALT3.INFO.",
            function_name="build_node_grid_smear")

    sigcoh, colour_disp_max = _read_info_scalars(salt_model_dir / info_name)
    sigcoh = 0.0 if fig8 else sigcoh

    lam, sigma = read_two_column_file(salt_model_dir / disp_name)
    sigma = np.minimum(sigma, colour_disp_max)

    if not fig8:
        fudge = np.where(lam < G10_FUDGE_LAMSWITCH, 1.0,
                         1.0 + (lam - G10_FUDGE_LAMSWITCH) * G10_FUDGE_DSMEAR_DLAM)
        sigma = sigma * fudge

    minlam, maxlam = lam[0], lam[-1]

    # final node at MAXLAM-0.001 (sntools_genSmear.c)
    nodes = list(np.arange(minlam, maxlam + 1e-9, G10_LAMSEP_NODE))
    if maxlam - nodes[-1] > 1e-6:
        nodes.append(maxlam - 0.001)
    if sed_lam_max is not None and maxlam < sed_lam_max:
        raise SNANAError(
            f"wave-range for {disp_name} is too small: MAXLAM={maxlam:.2f} "
            f"< SED_LAMMAX={sed_lam_max:.2f}",
            function_name="build_node_grid_smear")

    node_lam = np.array(nodes)
    node_sig = np.interp(node_lam, lam, sigma)
    node_sig = np.minimum(node_sig, COLOR_DISP_MAX_DEFAULT - 1.0e-8)

    if zmin is not None and sed_lam_max is not None:
        lamcheck = node_lam[-1] * (1.0 + zmin)
        if lamcheck < maxlam and lamcheck < sed_lam_max:
            raise SNANAError(
                f"genSmear model can't handle zmin={zmin:.3e}: last node "
                f"{node_lam[-1]:.1f} A redshifts to {lamcheck:.1f} A, below "
                f"the dispersion range; increase zmin above "
                f"{maxlam / node_lam[-1] - 1.0:.3e}",
                function_name="build_node_grid_smear")

    node_lam = jnp.asarray(node_lam)
    node_sig = jnp.asarray(node_sig)
    n_nodes = len(nodes)

    # Coherent draw is last, including when its sigma is zero.
    return SmearModel("nodes", n_nodes + 1, node_lam, node_sig,
                      sigcoh, (minlam, maxlam))


def build_correlated_band_smear(lamcen, cholesky, sigmacoh=None):
    """C11 and OIR: correlated per-band scatter, SINFUN-interpolated between
    band centres and held flat outside (get_genSmear_Chotard11, _OIR).

    sigmacoh is None for a model with no coherent term (C11, which draws
    NBAND_C11 randoms) and a float for one that has it (OIR, which draws
    NBAND_OIR + 1). It is structural, not inferred from the value: an OIR model
    declaring SIGMACOH 0.0 still draws its extra random.
    """
    lamcen = jnp.asarray(lamcen)
    cholesky = jnp.asarray(cholesky)
    n_bands = len(lamcen)
    n_randoms = n_bands if sigmacoh is None else n_bands + 1
    return SmearModel("bands", n_randoms, lamcen, cholesky, sigmacoh)


def _read_oir_info(model_dir):
    """Band centres, colour sigmas, correlation matrix and the coherent sigma
    (read_OIR_INFO)."""
    lamcen, sigma, cormat, scale, sigmacoh = [], None, [], 1.0, 0.0
    with open(Path(model_dir) / 'OIR.INFO') as f:
        for line in f:
            words = line.split('#')[0].split()
            if not words:
                continue
            key, values = words[0], words[1:]
            if key == 'LAMCEN_BAND:':
                lamcen.append(float(values[1]))
            elif key == 'COLOR_SIGMA:':
                sigma = [float(v) for v in values]
            elif key == 'COLOR_SIGMA_SCALE:':
                scale = float(values[0])
            elif key in ('SIGMACOH_MB:', 'SIGMACOH:'):
                sigmacoh = float(values[0])
            elif key == 'COLOR_CORMAT:':
                cormat.append([float(v) for v in values])
    return (np.array(lamcen), np.array(sigma) * scale, np.array(cormat),
            sigmacoh)


def _parse_coh_sigmas(name):
    """Sigmas from COH, COH(sig) or COH(sig0+sig1); '+' is the only separator
    SNANA accepts between them (init_genSmear_COH)."""
    head, _, rest = name.partition('(')
    inner = rest.rstrip(')')
    if head != 'COH' or (rest and not rest.endswith(')')):
        raise SNANAError(f"Invalid GENMAG_SMEAR_MODELNAME: {name}. "
                         f"Must be COH or COH(sig) or COH(sig0+sig1)")
    if not inner:
        return [COH_SIGMA_DEFAULT]
    if any(c in inner for c in ':!,%'):
        raise SNANAError(f"Invalid GENMAG_SMEAR_MODELNAME: {name}. "
                         f"Only '+' may separate the two sigmas")
    sigmas = inner.split('+')
    if len(sigmas) > 2:   # MAGSIGMA holds two; SNANA's splitString aborts
        raise SNANAError(f"Invalid GENMAG_SMEAR_MODELNAME: {name}. "
                         f"At most two sigmas, as COH(sig0+sig1)")
    return [float(s) for s in sigmas]


def build_smear_model(modelname, salt_model_dir=None, sed_lam_max=None,
                      zmin=None):
    """Instantiate a smear model from a GENMAG_SMEAR_MODELNAME value."""
    name = modelname.strip()
    if name in ('G10', 'G10Fig8'):
        if salt_model_dir is None:
            raise SNANAError(
                f"GENMAG_SMEAR_MODELNAME {name} requires a SALT model "
                "(needs the colour-dispersion file)")
        return build_node_grid_smear(salt_model_dir, fig8=(name == 'G10Fig8'),
                                     sed_lam_max=sed_lam_max, zmin=zmin)
    # SNANA treats Chotard11 as a synonym of C11 (snlc_sim.c:10041)
    if name in ('C11', 'C11_0', 'C11_1', 'C11_2', 'Chotard11'):
        far_uv = 0 if name in ('C11', 'C11_0', 'Chotard11') else int(name[-1])
        # no coherent term (sntools_genSmear.c:1799)
        return build_correlated_band_smear(C11_LAMCEN, _c11_cholesky(far_uv))
    # matched by substring in SNANA, so COH(...) reaches the same branch
    if 'COH' in name:
        # COH sums one wavelength-independent offset per sigma linearly.
        sigmas = jnp.asarray(_parse_coh_sigmas(name))
        return SmearModel('coherent', len(sigmas), coefficients=sigmas)
    # Matched by substring, so the name may carry a path (snlc_sim.c:10071)
    if 'OIR.' in name:
        path = Path(os.path.expandvars(name))
        if path.parent == Path('.'):
            path = sndata_root() / 'models' / 'OIR' / name
        if not path.is_dir():
            raise SNANAError(f"No OIR model directory at {path}")
        lamcen, cholesky, sigmacoh = oir_bands(path)
        # OIR draws NBAND_OIR + 1: one extra for the coherent term
        # (sntools_genSmear.c:2937).
        return build_correlated_band_smear(lamcen, cholesky, sigmacoh=sigmacoh)
    raise SNANAError(f"Unknown GENMAG_SMEAR_MODELNAME: {modelname}")


def prepare_phase_scatter(config):
    """The Cholesky factor for GENMAG_SMEAR_ADDPHASECOR, or None when off.

    SNANA's rest-frame phase grid is NBIN_PHASECOR=60 points at
    TMIN=-18.0 with TBIN=1.0 (phases -18..+41;
    init_genSmear_phaseCor, sntools_genSmear.c:3754-3801), with covariance
    C(i,j) = sigma^2 * exp(-|T_i - T_j| / tau) and tau floored at 1e-12
    (snlc_sim.c:10099-10100). Per-SN draws are L @ z with z ~ N(0, I). The
    grid constants live on Instrument (phasecor_phase_min/step/n_grid) so
    construction and lookup share one definition.
    """
    addpc = config.model.genmag_smear_addphasecor
    if addpc is None:
        return None
    sigma, tau = float(addpc[0]), float(addpc[1])
    if sigma <= 0:
        return None
    tau = max(tau, 1.0e-12)
    inst = config.instrument
    phase_grid = (inst.phasecor_phase_min
                  + inst.phasecor_phase_step * np.arange(inst.phasecor_n_grid))
    diff = np.abs(phase_grid[:, None] - phase_grid[None, :])
    cov = sigma * sigma * np.exp(-diff / tau)
    L = np.linalg.cholesky(cov)
    log.line(logger, f"ADDPHASECOR ready: sigma={sigma}, tau={tau}, "
             f"grid={len(phase_grid)} pts "
             f"[{phase_grid[0]:.0f}, {phase_grid[-1]:.0f}] days")
    # Preserve float32 coefficients, then match the draws' working precision.
    # Mixed-width inputs can make XLA unroll this reduction across the batch.
    return jnp.asarray(L.astype(np.float32), dtype=float_dtype())


def draw_phase_scatter(phasecor_L, keys, clip):
    """Phase-correlated coherent scatter (GENMAG_SMEAR_ADDPHASECOR).

    Pre-draws correlated per-phase-grid scatter; epoch interpolation happens
    later in simulate_lightcurves. None when the model is not configured.
    """
    if phasecor_L is None:
        return None
    n_grid = phasecor_L.shape[0]
    normals = jax.vmap(
        lambda k: jax.random.truncated_normal(k, *clip, shape=(n_grid,)))(keys)
    # row-wise reduce, not a matmul: bits independent of the batch size
    L = phasecor_L
    return jax.vmap(lambda z: jnp.sum(L * z[None, :], axis=1))(normals)


def draw_coherent_scatter(model, keys):
    """GENMAG_SMEAR: one gaussian per SN, applied later as a grey shift
    (snlc_sim.c genmodelSmear). Truncated at SIGMACLIP_MAGSMEAR, as SNANA's
    getRan_GaussClip rejection sampling does."""
    genmag_smear = model.genmag_smear
    if genmag_smear <= 1e-4:
        return None
    clip_lo, clip_hi = model.sigmaclip_magsmear
    return genmag_smear * jax.vmap(
        lambda k: jax.random.truncated_normal(k, clip_lo, clip_hi))(keys)


def draw_filter_scatter(model, n_filters, keys):
    """GENMAG_SMEAR_FILTER: one clipped gaussian per (SN, filter), added to
    that filter's epochs (snlc_sim.c:29904)."""
    if not model.genmag_smear_filter:
        return None
    clip_lo, clip_hi = model.sigmaclip_magsmear
    return jax.vmap(
        lambda k: jax.random.truncated_normal(
            k, clip_lo, clip_hi, shape=(n_filters,)))(keys)


def draw_model_scatter(model, smear, keys):
    """GENMAG_SMEAR_MODELNAME randoms; correlation and interpolation are
    applied at SED time. GENSMEAR_RANGauss_FIX pins every draw to a constant
    for reproducibility (sntools_genSmear.c:307)."""
    if smear is None:
        return None
    n_ran = smear.n_randoms
    ran_fix = model.gensmear_rangauss_fix
    if ran_fix > -99.0:
        return jnp.full((len(keys), n_ran), float(ran_fix))
    clip_lo, clip_hi = model.sigmaclip_magsmear
    return jax.vmap(
        lambda k: jax.random.truncated_normal(
            k, clip_lo, clip_hi, shape=(n_ran,)))(keys)


def scale_model_scatter(model, gensmear_arr, model_params):
    """GENMAG_SMEAR_SCALE scales the GENMAG_SMEAR_MODELNAME smear only
    (sntools_genSmear.c:415-418): a global scalar, or poly(c|x1) evaluated per
    SN, which is why it runs once the model parameters exist."""
    if gensmear_arr is None:
        return None
    poly = model.genmag_smear_scale_poly
    if poly is not None and poly[0] in model_params:
        scale = eval_genpoly([float(c) for c in poly[1]], model_params[poly[0]])
        return gensmear_arr * scale[:, None]
    return gensmear_arr * model.genmag_smear_scale
