"""External cosmology constraints used by SNANA WFIT."""

from pathlib import Path
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core.cosmology import flat_cpl_comoving_integral, flat_cpl_e2
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import sndata_root
from jax_snlc_core.utils import LIGHT_KM, float_dtype

from .config import WFITConfig


_BAO_TYPES = {
    "DM_over_rd": 1,
    "DM_over_rs": 1,
    "DH_over_rd": 2,
    "DH_over_rs": 2,
    "DV_over_rd": 3,
    "DV_over_rs": 3,
}


def bao_root() -> Path:
    """Where SNANA keeps the BAO mean and covariance files."""
    return sndata_root() / "models" / "BAO"


def load_bao_data(config: WFITConfig,
                  root: Path) -> dict[str, jnp.ndarray | str]:
    """Read SNANA's BAO mean and covariance resources from ``root``."""
    sample = config.bao
    mean_path = root / f"{sample}_mean.dat"
    cov_path = root / f"{sample}_cov.dat"

    redshift: list[float] = []
    mean: list[float] = []
    observable: list[int] = []
    with open(mean_path, "rt") as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            words = stripped.split()
            if len(words) != 3 or words[2] not in _BAO_TYPES:
                raise SNANAError(
                    f"invalid BAO mean row in {mean_path}: {line}",
                    function_name="load_bao_data",
                )
            redshift.append(float(words[0]))
            mean.append(float(words[1]))
            observable.append(_BAO_TYPES[words[2]])

    covariance = np.loadtxt(cov_path, dtype=np.float64, ndmin=2)
    ndata = len(mean)
    if covariance.shape != (ndata, ndata):
        raise SNANAError(
            f"BAO covariance {covariance.shape} does not match {ndata} means",
            function_name="load_bao_data",
        )

    fractional_variance = (
        (config.bao_h0_sigma / config.bao_h0) ** 2
        + (config.bao_rd_sigma / config.bao_rd) ** 2
    )
    mean_array = np.asarray(mean, dtype=np.float64)
    covariance += fractional_variance * np.outer(mean_array, mean_array)
    if not np.allclose(covariance, covariance.T, rtol=1.0e-10, atol=1.0e-12):
        raise SNANAError(
            f"BAO covariance is not symmetric: {cov_path}",
            function_name="load_bao_data",
        )
    np.linalg.cholesky(covariance)

    return {
        "bao_redshift": jnp.asarray(redshift),
        "bao_mean": jnp.asarray(mean_array),
        "bao_observable": jnp.asarray(observable),
        "bao_precision": jnp.asarray(np.linalg.inv(covariance)),
        "bao_source_mean": str(mean_path),
        "bao_source_covariance": str(cov_path),
    }


def _common_simulation_cosmology(
    truths: Sequence[Mapping[str, float] | None],
    feature: str,
) -> tuple[float, float, float]:
    values = []
    for index, truth in enumerate(truths):
        if truth is None or any(
            truth.get(name) is None for name in ("omega_m", "w0", "wa")
        ):
            raise SNANAError(
                f"{feature} requires omega_m, w0, and wa truth for "
                f"realisation {index}",
                function_name="prepare_priors",
            )
        values.append((truth["omega_m"], truth["w0"], truth["wa"]))
    if any(value != values[0] for value in values[1:]):
        raise SNANAError(
            f"{feature} requires a common simulation cosmology across "
            "realisations",
            function_name="prepare_priors",
        )
    return values[0]


def _bao_model(
    data: Mapping[str, jnp.ndarray],
    omega_m: float,
    w0: float,
    wa: float,
    config: WFITConfig,
) -> jnp.ndarray:
    integral = flat_cpl_comoving_integral(
        data["bao_redshift"], omega_m, w0, wa, order=64,
        radiation=config.radiation,
    )
    dm = (LIGHT_KM / config.bao_h0) * integral
    # SNANA DH_bao_prior uses the radiation-free Hzfun, unlike DM_bao_prior.
    dh = (LIGHT_KM / config.bao_h0) / jnp.sqrt(
        flat_cpl_e2(data["bao_redshift"], omega_m, w0, wa, radiation=False)
    )
    dv = jnp.cbrt(data["bao_redshift"] * dm * dm * dh)
    return jnp.where(
        data["bao_observable"] == 1,
        dm / config.bao_rd,
        jnp.where(
            data["bao_observable"] == 2,
            dh / config.bao_rd,
            dv / config.bao_rd,
        ),
    )


def prepare_priors(
    config: WFITConfig,
    truths: Sequence[Mapping[str, float] | None],
    bao_directory: Path | None,
) -> dict[str, jnp.ndarray | str]:
    """Prepare per-realisation prior targets and the BAO resources.

    ``bao_directory`` holds the BAO mean and covariance files, and is read only
    when a BAO prior is configured; :func:`bao_root` is SNANA's own.
    """
    realisation_count = len(truths)
    bao = load_bao_data(config, bao_directory) if config.bao else {}
    cmb_target = jnp.full(
        realisation_count,
        config.Rcmb,
        dtype=float_dtype(),
    )
    if config.cmb_sim:
        omega_m, w0, wa = _common_simulation_cosmology(
            truths, "simulation-centred CMB prior",
        )
        integral = flat_cpl_comoving_integral(
            jnp.asarray(config.cmb_redshift),
            omega_m,
            w0,
            wa,
            order=128,
            radiation=False,  # init_cmb_prior uses radiation-free Hainv_integral.
        )
        cmb_target = jnp.full(
            realisation_count,
            jnp.sqrt(omega_m) * integral,
        )
    if config.bao_sim:
        omega_m, w0, wa = _common_simulation_cosmology(
            truths, "simulation-centred BAO prior",
        )
        bao = {
            **bao,
            "bao_mean": _bao_model(bao, omega_m, w0, wa, config),
        }
    omega_m_target = jnp.full(
        realisation_count,
        config.ompri,
        dtype=float_dtype(),
    )
    if config.cmb_sim:
        # -cmb_sim recentres the Omega_m prior on the simulation (wfit.c:1045)
        omega_m_target = jnp.full(realisation_count, omega_m,
                                  dtype=float_dtype())
    if config.perturb_priors:
        # SNANA re-seeds per call (get_prior_ranshift, wfit.c:2548), so the
        # omm and Rcmb shifts use the same gaussian draw
        base_key = jax.random.PRNGKey(config.prior_perturbation_seed)
        draws = jax.vmap(
            lambda index: jax.random.normal(jax.random.fold_in(base_key, index))
        )(jnp.arange(realisation_count))
        # only an informative Omega_m prior is perturbed (wfit.c:2408);
        # Rcmb only when centred on the simulation (wfit.c:2489)
        if config.omega_m_prior_applies() and config.dompri < 1.0:
            omega_m_target = omega_m_target + config.dompri * draws
        if config.cmb_sim:
            cmb_target = cmb_target + config.sigma_Rcmb * draws
    return {
        **bao,
        "omega_m_mean": omega_m_target,
        "cmb_r": cmb_target,
    }


def omega_m_prior_chi2(
    omega_m: float,
    target: jnp.ndarray,
    config: WFITConfig,
) -> jnp.ndarray:
    if not config.omega_m_prior_applies():
        return jnp.asarray(0.0)
    pull = (omega_m - target) / config.dompri
    return pull * pull


def cmb_prior_chi2(
    omega_m: float,
    w0: float,
    wa: float,
    target: jnp.ndarray,
    config: WFITConfig,
) -> jnp.ndarray:
    if not config.cmb:
        return jnp.asarray(0.0)
    # Match chi2_cmb_prior: the same radiation-free convention as the target.
    integral = flat_cpl_comoving_integral(
        jnp.asarray(config.cmb_redshift), omega_m, w0, wa, order=128,
        radiation=False,
    )
    r_model = jnp.sqrt(omega_m) * integral
    pull = (r_model - target) / config.sigma_Rcmb
    chi2 = pull * pull
    return jnp.where(
        (omega_m > 0.0) & jnp.isfinite(chi2),
        chi2,
        jnp.inf,
    )


def bao_prior_chi2(
    omega_m: float,
    w0: float,
    wa: float,
    config: WFITConfig,
    priors: Mapping[str, jnp.ndarray],
) -> jnp.ndarray:
    if not config.bao:
        return jnp.asarray(0.0)
    model = _bao_model(priors, omega_m, w0, wa, config)
    delta = priors["bao_mean"] - model
    bao_chi2 = delta @ priors["bao_precision"] @ delta
    rd_pull = (config.bao_rd - config.rd_prior) / config.rd_prior_sig
    chi2 = bao_chi2 + rd_pull * rd_pull
    return jnp.where(jnp.isfinite(chi2), chi2, jnp.inf)
