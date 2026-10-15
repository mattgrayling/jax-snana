"""
Vectorized photo-z generation from HOSTLIB_GENZPHOT_FUDGEPAR.

Matches SNANA's GEN_SNHOST_ZPHOT_from_CALC (sntools_host.c).
All operations are fully vectorised over (N,) arrays with no per-SN loops.
"""

import logging
from pathlib import Path
from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats import norm

from jax_snlc_core.utils import SNANA_NULLFLOAT, SNANAError, snana_log as slog
from jax_snlc_core.utils.interp import eval_genpoly
from jax_snlc_sim.distributions import _bifurcated_cdf, _bifurcated_ppf

logger = logging.getLogger(__name__)


# SNANA constants from sntools_host.h / sntools_host.c
ZMIN = 0.01
MINSIGMA0 = 3.0
ZRATIO_MIN_SOLVE = 0.75
SQRT8_OVER_PI = np.sqrt(8.0 / np.pi)
ZPHOT_MIN = 0.001  # SNANA redraws component + value while ZPHOT < this
SIGMA_OUTLIER_FLAT = 9.999  # flat outlier mode when sig_outlier exceeds this
NRAND_ZPHOT = 3    # uniform randoms per SN: [ran_prob, u_z, flat_draw]


@dataclass(frozen=True)
class GenZphot:
    """The photo-z model the HOSTLIB_GENZPHOT_* keys define
    (GEN_SNHOST_ZPHOT_from_CALC, sntools_host.c:6850-6952)."""
    fudgepar: tuple                      # a0 a1 a2 prob_outlier sig_outlier
    fudgemap_z: Optional[np.ndarray]      # core width in z; replaces a0-a2
    fudgemap_rms: Optional[np.ndarray]
    bias: Optional[np.ndarray]            # ZBIAS polynomial in the host's ZTRUE
    outlier_range: Tuple[float, float]

    def __post_init__(self):
        # These coefficients select the core/outlier algorithms; keep them
        # immutable while the interpolation and bias tables remain dynamic.
        object.__setattr__(self, "fudgepar", tuple(self.fudgepar))


def build_genzphot(host, redshift_range, hostlib_z_range) -> Optional[GenZphot]:
    """The photo-z model of the HOSTLIB_GENZPHOT_* keys in ``host``, or None
    when they leave the catalogue photo-z in use: the model is on when
    FUDGEPAR has a core or outlier term, FUDGEMAP is given, or BIAS has a
    non-zero coefficient. FUDGEPAR's fifth value may be FLAT(z1:z2), flat
    outliers over that range; otherwise they span ``hostlib_z_range``
    (sntools_host.c:3257-3261). FUDGEMAP must cover ``redshift_range``.
    Malformed values abort. SNANA misreads them instead: a FUDGEPAR short
    of five values takes the next key as its fifth and crashes
    (snlc_sim.c:4013-4020), and a FUDGEMAP entry without a colon takes the
    previous entry's rms (snlc_sim.c:3377-3381)."""
    fudgepar, flat_range = _parse_fudgepar(host.genzphot_fudgepar)
    fudgemap = _parse_fudgemap(host.genzphot_fudgemap)
    bias = (None if host.genzphot_bias is None
            else np.asarray(host.genzphot_bias, dtype=float))
    active = (fudgemap is not None
              or (fudgepar is not None
                  and (np.any(fudgepar[:3] != 0.0) or fudgepar[3] > 0.0))
              or (bias is not None and np.any(bias != 0.0)))
    if not active:
        return None
    if fudgemap is not None:
        validate_genzphot_fudgemap(*fudgemap, redshift_range)
    slog.line(logger, "GENZPHOT_FUDGEPAR active: photo-z from formula")
    return GenZphot(
        fudgepar=np.zeros(5) if fudgepar is None else fudgepar,
        fudgemap_z=None if fudgemap is None else fudgemap[0],
        fudgemap_rms=None if fudgemap is None else fudgemap[1],
        bias=bias,
        outlier_range=flat_range or hostlib_z_range)


def _parse_fudgepar(value):
    """HOSTLIB_GENZPHOT_FUDGEPAR as ``(fudgepar, flat_range)``: five values
    'a0 a1 a2 prob_outlier sig_outlier', the last possibly FLAT(z1:z2),
    which is sig_outlier 99 with flat outliers over (z1, z2); ``(None,
    None)`` when the key is unset."""
    if value is None:
        return None, None
    items = (list(value) if isinstance(value, (list, tuple))
             else str(value).split())
    if len(items) != 5:
        raise SNANAError(
            f"HOSTLIB_GENZPHOT_FUDGEPAR takes five values 'a0 a1 a2 "
            f"prob_outlier sig_outlier', got {len(items)}: {value!r}",
            function_name="build_genzphot")
    flat_range = None
    last = str(items[4])
    if last.upper().startswith('FLAT'):
        z1, z2 = (float(x) for x in
                  last[last.find('(') + 1:last.find(')')].split(':'))
        flat_range, items[4] = (z1, z2), 99.0
    return np.array([float(x) for x in items]), flat_range


def _parse_fudgemap(value):
    """HOSTLIB_GENZPHOT_FUDGEMAP 'z0:rms0,z1:rms1,...' as ``(z, rms)``
    arrays, or None when the key is unset."""
    if value is None:
        return None
    pairs = [pair.strip().split(':') for pair in str(value).split(',')]
    if any(len(pair) != 2 for pair in pairs):
        raise SNANAError(
            f"HOSTLIB_GENZPHOT_FUDGEMAP takes 'z:rms' pairs separated by "
            f"commas, got {value!r}", function_name="build_genzphot")
    return (np.array([float(z) for z, _ in pairs]),
            np.array([float(rms) for _, rms in pairs]))


class ZphotEfficiency(NamedTuple):
    """HOSTLIB_ZPHOTEFF_FILE: the probability of obtaining a host photo-z,
    against the host's true redshift."""
    ztrue: np.ndarray
    eff: np.ndarray


def read_zphoteff_file(path, host_z_range) -> ZphotEfficiency:
    """The two columns ZTRUE and EFF of a HOSTLIB_ZPHOTEFF_FILE
    (init_HOSTLIB_ZPHOTEFF, sntools_host.c:4671-4695). A missing file
    aborts, as rd2columnFile does; so does one without two columns, on
    which SNANA crashes. It must cover ``host_z_range``, the catalogue's
    redshifts, as SNANA aborts on a host outside it (sntools.c:8234)."""
    path = Path(path)
    if not path.exists():
        raise SNANAError(f"HOSTLIB_ZPHOTEFF_FILE {path} not found",
                         function_name="read_zphoteff_file")
    data = np.loadtxt(path, comments='#', ndmin=2)
    if data.shape[1] < 2:
        raise SNANAError(f"HOSTLIB_ZPHOTEFF_FILE {path} needs two columns, "
                         f"ZTRUE and EFF", function_name="read_zphoteff_file")
    if host_z_range[0] < data[0, 0] or host_z_range[1] > data[-1, 0]:
        raise SNANAError(f"HOSTLIB_ZPHOTEFF_FILE {path} does not cover the "
                         f"HOSTLIB's redshifts, {host_z_range[0]:.4f} to "
                         f"{host_z_range[1]:.4f}",
                         function_name="read_zphoteff_file")
    slog.line(logger, f"ZPHOTEFF loaded ({len(data)} entries)")
    return ZphotEfficiency(ztrue=data[:, 0], eff=data[:, 1])


def catalogue_zphot_shift(zgen, ztrue, hostlib_mskopt):
    """The shift a catalogue photo-z takes to the SN's redshift, ZGEN - ZTRUE
    of the drawn host, or zero under SN2GAL_Z, where the SN already sits at
    its host's redshift (GEN_SNHOST_ZPHOT_from_HOSTLIB,
    sntools_host.c:7182-7185). One shift serves every neighbour."""
    if hostlib_mskopt & 16:
        return jnp.zeros_like(zgen)
    return zgen - ztrue


def shifted_catalogue_zphot(zphot, zphot_err, zdif):
    """A catalogue photo-z and its error moved by ``zdif``; both -9 where
    either is negative, a failed photo-z fit (sntools_host.c:7191-7192)."""
    zphot = zphot + zdif
    bad = (zphot < 0.0) | (zphot_err < 0.0)
    return (jnp.where(bad, SNANA_NULLFLOAT, zphot),
            jnp.where(bad, SNANA_NULLFLOAT, zphot_err))


def zphoterr_asym(
    ztrue: np.ndarray,
    zphoterr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised asymmetric photo-z error solver.

    Matches SNANA's zphoterr_asym() two-iteration grid search.
    Prevents negative photo-z by computing asymmetric Gaussian parameters.

    ``ztrue``/``zphoterr`` are (N,), the true redshift and photo-z error
    (sigma). Returns ``(zpeak, sigma_lo, sigma_hi, rms)``, each (N,): the
    peak of the asymmetric Gaussian, its lower/upper sigma, and the RMS of
    the asymmetric distribution.
    """
    N = len(ztrue)
    zphoterr_local = zphoterr.copy()
    zratio = (ztrue - ZMIN) / zphoterr_local
    is_symmetric = zratio > MINSIGMA0

    too_small = zratio < ZRATIO_MIN_SOLVE
    zphoterr_local = jnp.where(too_small, (ztrue - ZMIN) / ZRATIO_MIN_SOLVE, zphoterr_local)
    zratio = jnp.where(too_small, ZRATIO_MIN_SOLVE, zratio)
    sqzphoterr = zphoterr_local ** 2

    def _eval_grid(zratio_grid, valid_mask, ztrue_col, sqzphoterr_col):
        """Evaluate zphoterr_asym objective on a grid.

        ``zratio_grid``/``valid_mask`` are (N, G), the grid of zratio
        values and its validity mask; ``ztrue_col``/``sqzphoterr_col`` are
        (N, 1), the true redshift and squared photo-z error.

        Returns ``(vdif, sigma_minus, sigma_plus, variance)``, each (N, G):
        the objective values (1e10 where invalid), lower/upper sigmas, and
        variances.
        """
        sigma_minus = (ztrue_col - ZMIN) * zratio_grid
        sigma_dif = (ztrue_col - ZMIN - MINSIGMA0 * sigma_minus) / SQRT8_OVER_PI
        sigma_plus = 2.0 * sigma_dif + sigma_minus
        valid_mask = valid_mask & (sigma_plus > 0.001)
        sigma_sum = 0.5 * (sigma_plus + sigma_minus)
        variance = sigma_sum ** 2 + sigma_dif ** 2 * (1.0 - 2.0 / jnp.pi)
        vdif = jnp.abs(variance / sqzphoterr_col - 1.0)
        vdif = jnp.where(valid_mask, vdif, 1e10)
        return vdif, sigma_minus, sigma_plus, variance

    # SNANA steps from 0.02, multiplying by 1.10 each step
    n_grid0 = 54
    grid0 = 0.02 * (1.10 ** jnp.arange(n_grid0))
    valid0 = grid0[None, :] < zratio[:, None]

    vdif0, _, _, _ = _eval_grid(
        grid0[None, :] * jnp.ones((N, 1)),
        valid0,
        ztrue[:, None],
        sqzphoterr[:, None],
    )
    best_idx0 = jnp.argmin(vdif0, axis=1)
    best_zratio0 = grid0[best_idx0]

    # fine pass around the coarse best
    n_grid1 = 28
    zr_start = jnp.maximum(best_zratio0 * 0.96, 0.02)
    grid1 = zr_start[:, None] * (1.003 ** jnp.arange(n_grid1)[None, :])
    zr_end = jnp.minimum(best_zratio0 * 1.04, zratio)
    valid1 = grid1 < zr_end[:, None]

    vdif1, sigma_minus_g1, sigma_plus_g1, variance_g1 = _eval_grid(
        grid1,
        valid1,
        ztrue[:, None],
        sqzphoterr[:, None],
    )
    best_idx1 = jnp.argmin(vdif1, axis=1)

    zpeak_asym = ZMIN + MINSIGMA0 * jnp.take_along_axis(
        sigma_minus_g1, best_idx1[:, None], 1).squeeze(1)
    sigma_lo_asym = jnp.take_along_axis(
        sigma_minus_g1, best_idx1[:, None], 1).squeeze(1)
    sigma_hi_asym = jnp.take_along_axis(
        sigma_plus_g1, best_idx1[:, None], 1).squeeze(1)
    rms_asym = jnp.sqrt(jnp.take_along_axis(
        variance_g1, best_idx1[:, None], 1).squeeze(1))

    zpeak = jnp.where(is_symmetric, ztrue, zpeak_asym)
    sigma_lo = jnp.where(is_symmetric, zphoterr, sigma_lo_asym)
    sigma_hi = jnp.where(is_symmetric, zphoterr, sigma_hi_asym)
    rms = jnp.where(is_symmetric, zphoterr, rms_asym)

    return zpeak, sigma_lo, sigma_hi, rms


def validate_genzphot_fudgemap(
    fudgemap_z: np.ndarray,
    fudgemap_rms: np.ndarray,
    redshift_range: Tuple[float, float],
) -> None:
    """Fatal validation of HOSTLIB_GENZPHOT_FUDGEMAP at build time.

    SNANA aborts when the map's z range does not cover GENRANGE_REDSHIFT
    (snlc_sim.c:7568-7588); an out-of-range interp_1DFUN request is also
    fatal (sntools.c:7996-8004), so the map must be strictly ordered and
    finite.
    """
    z = jnp.asarray(fudgemap_z, dtype=np.float64)
    rms = jnp.asarray(fudgemap_rms, dtype=np.float64)
    if not (np.all(np.isfinite(z)) and np.all(np.isfinite(rms))):
        raise SNANAError(
            "HOSTLIB_GENZPHOT_FUDGEMAP contains non-finite values",
            function_name="validate_genzphot_fudgemap")
    if len(z) > 1 and not np.all(np.diff(z) > 0):
        raise SNANAError(
            "HOSTLIB_GENZPHOT_FUDGEMAP z values must be strictly increasing",
            function_name="validate_genzphot_fudgemap")
    zmin_gen, zmax_gen = redshift_range
    if z[0] > zmin_gen or z[-1] < zmax_gen:
        raise SNANAError(
            f"HOSTLIB_GENZPHOT_FUDGEMAP covers {z[0]:.4f} < z < {z[-1]:.4f} but "
            f"GENRANGE_REDSHIFT covers {zmin_gen:.4f} {zmax_gen:.4f}",
            function_name="validate_genzphot_fudgemap")


def generate_zphot_from_calc(
    ztrue: np.ndarray,
    model: GenZphot,
    uniforms: np.ndarray,
    ztrue_for_bias: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic photo-z values from GENZPHOT_FUDGEPAR formula.

    Matches SNANA's GEN_SNHOST_ZPHOT_from_CALC (sntools_host.c:6850-6952).
    SNANA redraws the outlier classification and the value whenever
    ZPHOT < 0.001 (the PICKRAN loop), so the accepted law is a mixture whose
    component probabilities are reweighted by each component's mass above
    0.001, with each chosen component sampled from its lower-truncated
    distribution. That law is implemented here directly via the shared
    bifurcated CDF/PPF, with no retries.

    ``ztrue`` is (N,), the redshifts for photo-z generation (SNANA's ZGEN =
    SN redshift); ``model`` is the :class:`GenZphot`; ``uniforms`` is (N,
    NRAND_ZPHOT), pre-generated uniform [0,1] random numbers per SN
    (columns [ran_prob, u_z, flat_draw]), produced via JAX fold_in for
    batch-invariant reproducibility. ``ztrue_for_bias`` is (N,), the host
    ZTRUE the ZBIAS polynomial reads (SNANA uses the host's z, not the SN's).

    Returns ``(zphot, zphot_err)``, each (N,): the generated photo-z
    values and their errors (RMS of the sigma_core asymmetric Gaussian).
    """
    a0, a1, a2, prob_outlier, sig_outlier = model.fudgepar

    if model.fudgemap_z is not None:
        sigma_core = jnp.interp(ztrue, model.fudgemap_z, model.fudgemap_rms)
    else:
        onepz = 1.0 + ztrue
        sigma_core = a0 + a1 * onepz + a2 * onepz ** 2

    # flat outliers above SIGMA_OUTLIER_FLAT (sntools_host.c:6880,6927)
    sigma_outlier_arr = sig_outlier * (1.0 + ztrue)
    is_outlier_flat = sig_outlier > SIGMA_OUTLIER_FLAT

    ran_prob = uniforms[:, 0]
    u_z = uniforms[:, 1]
    u_flat = uniforms[:, 2]

    # ZPHOT_ERR is the core's asymmetric RMS (sntools_host.c:6949-6951)
    zpeak_c, sig_lo_c, sig_hi_c, zphot_err = zphoterr_asym(ztrue, sigma_core)
    cdfmin_core = _bifurcated_cdf(ZPHOT_MIN, zpeak_c, sig_lo_c, sig_hi_c)
    mass_core = 1.0 - cdfmin_core

    if prob_outlier > 0.0 and is_outlier_flat:
        outlier_zmin, outlier_zmax = model.outlier_range
        flat_lo = max(outlier_zmin, ZPHOT_MIN)
        mass_out = (max(outlier_zmax - flat_lo, 0.0)
                    / (outlier_zmax - outlier_zmin))
    elif prob_outlier > 0.0:
        zpeak_o, sig_lo_o, sig_hi_o, _ = zphoterr_asym(
            ztrue, sigma_outlier_arr)
        cdfmin_out = _bifurcated_cdf(ZPHOT_MIN, zpeak_o, sig_lo_o, sig_hi_o)
        mass_out = 1.0 - cdfmin_out
    else:
        mass_out = 0.0

    # PICKRAN re-selects the component, so each weight carries its accepted mass
    w_out = prob_outlier * mass_out
    w_core = (1.0 - prob_outlier) * mass_core
    p_out_acc = w_out / (w_out + w_core)
    is_outlier = ran_prob <= p_out_acc

    u_core = cdfmin_core + u_z * (1.0 - cdfmin_core)
    zphot = _bifurcated_ppf(u_core, zpeak_c, sig_lo_c, sig_hi_c)
    if prob_outlier > 0.0 and is_outlier_flat:
        flat_draw = flat_lo + u_flat * (outlier_zmax - flat_lo)
        zphot = jnp.where(is_outlier, flat_draw, zphot)
    elif prob_outlier > 0.0:
        u_out = cdfmin_out + u_z * (1.0 - cdfmin_out)
        zphot_out = _bifurcated_ppf(u_out, zpeak_o, sig_lo_o, sig_hi_o)
        zphot = jnp.where(is_outlier, zphot_out, zphot)

    # SNANA evaluates ZBIAS at the host ZTRUE, not the SN ZGEN
    if model.bias is not None:
        zphot = zphot + eval_genpoly(model.bias, ztrue_for_bias)

    return zphot, zphot_err


def generate_zphot_quantiles_gaussian(zphot, zphot_err,
                                      quantiles) -> dict:
    """Generate photo-z quantiles from Gaussian (HOSTLIB_MSKOPT ZPHOT_QGAUSS).

    SNANA: quantile = ZPHOT + ZPHOT_ERR * InverseNormal(percent/100), with
    the 0 and 100 percentiles evaluated at 0.001/0.999
    (sntools_host.c:4622-4624).

    ``zphot``/``zphot_err`` are (N,), the photo-z values and errors;
    ``quantiles`` the (column, percentile) pairs to generate. Returns a dict
    mapping each column to its (N,) array.
    """
    result = {}
    for name, p in quantiles:
        frac = np.clip(p / 100.0, 0.001, 0.999)
        qval = zphot + zphot_err * norm.ppf(frac)
        # a failed catalogue photo-z has no quantiles
        qval = jnp.where(zphot < 0, -9.0, qval)
        result[name] = qval

    return result


jax.tree_util.register_dataclass(
    GenZphot,
    data_fields=[
        'fudgemap_z',
        'fudgemap_rms',
        'bias',
    ],
    meta_fields=[
        'fudgepar',
        'outlier_range',
    ],
)
