"""Laplace, NUTS, and chunked-grid inference for JAX WFIT."""

from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import summary as numpyro_summary
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value

from jax_snlc_core.utils import SNANAError
from jax_snlc_core.utils.newton_lm import newton_lm

from .config import WFITConfig
from .data import HubbleDiagram
from .likelihood import evaluate_likelihood

#: Convergence: the Newton step still to take, inv(H) g, as a fraction of the
#: parameter error from the same Hessian.
_STEP_TOLERANCE = 0.1
#: A parameter this close to its bound (as a fraction of the range) is pinned.
_BOUNDARY_FRACTION = 1.0e-4
#: Grid posterior mass at an axis edge above this fraction of the peak warns.
_EDGE_PROBABILITY_WARN = 0.05
#: Newton-LM iterations for the chi2 minimum; convergence is enforced by the
#: step tolerance above.
_NEWTON_MAXITER = 100


#: One fitted realisation. Keys: method, point_estimate_kind, point_estimate,
#: summaries, covariance, correlation, likelihood, ndof_hd, n_external,
#: prior_targets, diagnostics, warnings, blinded, samples.
FitResult = dict[str, Any]


def diagnostics(
    success: bool = True,
    message: str = "",
    divergences: int = 0,
    rhat: dict[str, float] | None = None,
    ess: dict[str, float] | None = None,
    newton_step_over_error: float | None = None,
    precision_eigenvalues: list[float] | None = None,
    edge_probability_ratio: dict[str, tuple[float, float]] | None = None,
) -> dict[str, object]:
    """Build the diagnostics block written verbatim into the summary YAML."""
    return {
        "success": success,
        "message": message,
        "divergences": divergences,
        "rhat": rhat or {},
        "ess": ess or {},
        "newton_step_over_error": newton_step_over_error,
        "precision_eigenvalues": precision_eigenvalues,
        "edge_probability_ratio": edge_probability_ratio or {},
    }


def summary(
    center: float, mean: float, median: float, std: float,
    q16: float, q84: float, sig_type: str = "68", marg: bool = True,
) -> dict[str, float]:
    """One parameter's posterior summary.

    ``value`` and ``sigma`` are the estimate and uncertainty every consumer
    reports. -marg selects the first: the marginalised mean, or the best-fit
    point under -minchi2 (wfit.c:3885). -sig_type selects the second: "68"
    averages the two half-widths of the 68.3% interval about the mean, which
    reduces to half the q16-q84 width, or the standard deviation (wfit.c:3892);
    -minchi2 always reports the standard deviation (wfit.c:3907-3910).
    """
    return {
        "center": center, "mean": mean, "median": median,
        "std": std, "q16": q16, "q84": q84,
        "value": mean if marg else center,
        "sigma": std if (sig_type == "std" or not marg)
        else 0.5 * (q84 - q16),
    }


def free_parameter_names(config: WFITConfig) -> tuple[str, ...]:
    if config.model == "lcdm":
        return ("omega_m",)
    if config.model == "wcdm":
        return ("omega_m", "w0")
    return ("omega_m", "w0", "wa")


def initial_values(config: WFITConfig) -> dict[str, float]:
    preferred = {"omega_m": config.ompri, "w0": -1.0, "wa": 0.0}
    values: dict[str, float] = {}
    for name in free_parameter_names(config):
        lo, hi = config.bounds(name)
        width = hi - lo
        values[name] = float(np.clip(
            preferred[name], lo + 1.0e-6 * width, hi - 1.0e-6 * width,
        ))
    return values


def _model_for_data(
    diagrams: Sequence[HubbleDiagram],
    config: WFITConfig,
    priors: Mapping[str, jnp.ndarray],
):
    """Build the NumPyro model over one realisation's arrays."""
    hdibc = _hdibc_arrays(diagrams[0])

    def model(redshift, mu, precision, valid, omega_m_prior_mean, cmb_r):
        params: dict[str, jnp.ndarray] = {}
        for name in free_parameter_names(config):
            lo, hi = config.bounds(name)
            params[name] = numpyro.sample(name, dist.Uniform(lo, hi))
        components = evaluate_likelihood(
            params, redshift, mu, precision, valid,
            omega_m_prior_mean, cmb_r, config, priors, hdibc=hdibc,
        )
        numpyro.factor(
            "wfit_log_likelihood", -0.5 * components["total_chi2"],
        )

    return model


def _model_arrays(
    arrays: tuple, priors: Mapping[str, jnp.ndarray],
) -> tuple:
    """The six per-realisation arrays the model takes, in argument order."""
    redshift, mu, precision, valid = arrays
    return (
        redshift, mu, precision, valid,
        priors["omega_m_mean"], priors["cmb_r"],
    )


def _summaries_from_samples(
    samples: Mapping[str, np.ndarray],
    center: Mapping[str, float],
    sig_type: str,
    marg: bool,
    names: tuple[str, ...],
) -> tuple[dict[str, dict[str, float]], np.ndarray, np.ndarray]:
    matrix = np.column_stack([np.asarray(samples[name]).reshape(-1) for name in names])
    covariance = np.atleast_2d(np.cov(matrix, rowvar=False, ddof=1))
    std = np.sqrt(np.diag(covariance))
    denom = np.outer(std, std)
    correlation = np.divide(
        covariance, denom, out=np.eye(len(names)), where=denom > 0,
    )
    summaries: dict[str, dict[str, float]] = {}
    for index, name in enumerate(names):
        values = matrix[:, index]
        summaries[name] = summary(
            center=float(center[name]),
            mean=float(np.mean(values)),
            median=float(np.median(values)),
            std=float(np.std(values, ddof=1)),
            q16=float(np.quantile(values, 0.5 - 0.683 / 2.0)),
            q84=float(np.quantile(values, 0.5 + 0.683 / 2.0)),
            sig_type=sig_type,
            marg=marg,
        )
    return summaries, covariance, correlation


def _hdibc_arrays(diagram: HubbleDiagram):
    if diagram.mu_second is None:
        return None
    return (diagram.mu_second, diagram.bias0, diagram.bias1)


def _external_count(
    config: WFITConfig, priors: Mapping[str, jnp.ndarray],
) -> int:
    count = int(config.omega_m_prior_applies()) + int(config.cmb)
    if config.bao:
        count += int(priors["bao_redshift"].size)
    return count


def _prior_ndof(config: WFITConfig, priors: Mapping[str, jnp.ndarray]) -> int:
    """The degrees of freedom the external priors add: one for the CMB, one
    per BAO measurement, none for the Omega_m prior (set_Ndof,
    wfit.c:3011-3024)."""
    count = int(config.cmb)
    if config.bao:
        count += int(priors["bao_redshift"].size)
    return count


def _finish_result(
    *,
    method: str,
    summaries: dict[str, dict[str, float]],
    covariance: np.ndarray,
    correlation: np.ndarray,
    diagnostics: dict[str, object],
    diagram: HubbleDiagram,
    arrays: tuple,
    realisation_index: int,
    config: WFITConfig,
    priors: Mapping[str, jnp.ndarray],
    blinded: bool,
    samples: dict[str, np.ndarray] | None = None,
    warnings: list[str] | None = None,
) -> FitResult:
    point = {
        name: summaries[name]["value"]
        for name in free_parameter_names(config)
    }
    components = {
        key: float(value)
        for key, value in evaluate_likelihood(
            point,
            *(jax.tree.map(lambda a: a[realisation_index], array) for array in arrays),
            priors["omega_m_mean"][realisation_index],
            priors["cmb_r"][realisation_index],
            config,
            priors,
            hdibc=_hdibc_arrays(diagram),
        ).items()
    }
    return {
        "method": method,
        "point_estimate_kind": (
            "marginal_posterior_mean" if config.marg else "best_fit"
        ),
        "point_estimate": {
            key: float(value) for key, value in point.items()
        },
        "summaries": summaries,
        "covariance": np.asarray(covariance),
        "correlation": np.asarray(correlation),
        "likelihood": components,
        "ndof_hd": (diagram.n_rows - 1 - len(free_parameter_names(config))
                    + _prior_ndof(config, priors)),
        "n_external": _external_count(config, priors),
        "prior_targets": {
            **(
                {
                    "omega_m": float(
                        priors["omega_m_mean"][realisation_index]
                    )
                }
                if config.omega_m_prior_applies() else {}
            ),
            **(
                {"cmb_r": float(priors["cmb_r"][realisation_index])}
                if config.cmb else {}
            ),
        },
        "diagnostics": diagnostics,
        "warnings": list(warnings or []),
        "blinded": blinded,
        "samples": samples,
    }


def _minimise_chi2(
    diagrams: Sequence[HubbleDiagram],
    arrays: tuple,
    config: WFITConfig,
    priors: Mapping[str, jnp.ndarray],
):
    """Newton-LM chi2 minimum per realisation: (centre, chi2, Hessian,
    gradient), each with a leading realisation axis. The same chi2 the grid
    method evaluates; bounds are handled by clipping, so the minimum is in
    the parameter space itself."""
    names = free_parameter_names(config)
    hdibc = _hdibc_arrays(diagrams[0])
    lo = jnp.array([config.bounds(name)[0] for name in names])
    hi = jnp.array([config.bounds(name)[1] for name in names])
    init = initial_values(config)
    p0 = jnp.array([init[name] for name in names])
    model_arrays = _model_arrays(arrays, priors)

    def fit_one(*one_arrays):
        def chi2(flat):
            params = {name: flat[index] for index, name in enumerate(names)}
            return evaluate_likelihood(
                params, *one_arrays, config, priors, hdibc=hdibc,
            )["total_chi2"]

        centre, value, _ = newton_lm(p0, lo, hi, chi2, _NEWTON_MAXITER)
        return centre, value, jax.hessian(chi2)(centre), jax.grad(chi2)(centre)

    return jax.vmap(fit_one)(*model_arrays)


def _summaries_from_gaussian(
    center: Mapping[str, float],
    covariance: np.ndarray,
    sig_type: str,
    marg: bool,
    names: tuple[str, ...],
) -> tuple[dict[str, dict[str, float]], np.ndarray, np.ndarray]:
    std = np.sqrt(np.diag(covariance))
    denom = np.outer(std, std)
    correlation = np.divide(
        covariance, denom, out=np.eye(len(names)), where=denom > 0,
    )
    summaries = {
        name: summary(
            center=float(center[name]), mean=float(center[name]),
            median=float(center[name]), std=float(std[index]),
            q16=float(center[name] - std[index]),
            q84=float(center[name] + std[index]),
            sig_type=sig_type, marg=marg,
        )
        for index, name in enumerate(names)
    }
    return summaries, covariance, correlation


def fit_laplace(
    diagrams: Sequence[HubbleDiagram],
    arrays: tuple,
    config: WFITConfig,
    priors: Mapping[str, jnp.ndarray],
    blinded: Sequence[bool],
) -> tuple[tuple[FitResult, ...], tuple[dict, ...]]:
    """One Newton-LM chi2 minimum per realisation, with Gaussian errors from
    the Hessian: ``cov = 2 inv(H)`` since the loss is a chi-square."""
    names = free_parameter_names(config)
    centres, losses, hessians, gradients = _minimise_chi2(
        diagrams, arrays, config, priors,
    )
    centres = np.asarray(centres)
    losses = np.asarray(losses)
    hessians = np.asarray(hessians)
    gradients = np.asarray(gradients)

    lo_bounds = np.array([config.bounds(name)[0] for name in names])
    hi_bounds = np.array([config.bounds(name)[1] for name in names])

    results: list[FitResult] = []
    grids: list[dict] = []
    for realisation_index in range(len(diagrams)):
        hessian = hessians[realisation_index]
        centre_values = centres[realisation_index]
        eigenvalues = np.linalg.eigvalsh(hessian)
        # projected gradient: a coordinate clipped at its bound with the
        # gradient pointing outward is converged
        gradient = gradients[realisation_index].copy()
        gradient[(centre_values <= lo_bounds) & (gradient > 0.0)] = 0.0
        gradient[(centre_values >= hi_bounds) & (gradient < 0.0)] = 0.0
        laplace = (
            np.isfinite(losses[realisation_index])
            and np.all(np.isfinite(hessian))
            and np.all(eigenvalues > 0.0)
        )
        if not laplace:
            raise SNANAError(
                f"realisation {realisation_index}: the chi2 Hessian is not "
                f"positive definite (smallest eigenvalue "
                f"{eigenvalues.min():.3e})",
                function_name="fit_laplace",
            )
        covariance = 2.0 * np.linalg.inv(hessian)
        step = np.linalg.solve(hessian, gradient)
        step_over_error = float(np.max(np.abs(step)
                                       / np.sqrt(np.diag(covariance))))
        if not step_over_error <= _STEP_TOLERANCE:
            raise SNANAError(
                f"realisation {realisation_index}: chi2 minimisation did not "
                f"converge (remaining Newton step {step_over_error:.3e} of "
                f"the parameter error)",
                function_name="fit_laplace",
            )
        result_diagnostics = diagnostics(
            newton_step_over_error=step_over_error,
            precision_eigenvalues=eigenvalues.tolist(),
        )

        centre = {
            name: float(centres[realisation_index, index])
            for index, name in enumerate(names)
        }
        summaries, covariance, correlation = _summaries_from_gaussian(
            centre, covariance, config.sig_type, config.marg, names,
        )

        boundary_parameters: list[str] = []
        for name in names:
            lo, hi = config.bounds(name)
            span = hi - lo
            if min(
                centre[name] - lo,
                hi - centre[name],
            ) < _BOUNDARY_FRACTION * span:
                boundary_parameters.append(name)
        if boundary_parameters:
            raise SNANAError(
                f"realisation {realisation_index} chi2 minimum is pinned at "
                "the bound for " + ", ".join(boundary_parameters),
                function_name="fit_laplace",
            )

        results.append(_finish_result(
            method="laplace",
            summaries=summaries,
            covariance=covariance,
            correlation=correlation,
            diagnostics=result_diagnostics,
            diagram=diagrams[realisation_index],
            arrays=arrays,
            realisation_index=realisation_index,
            config=config,
            priors=priors,
            blinded=blinded[realisation_index],
        ))
        grids.append({})
    return tuple(results), tuple(grids)


def fit_nuts(
    diagrams: Sequence[HubbleDiagram],
    arrays: tuple,
    config: WFITConfig,
    priors: Mapping[str, jnp.ndarray],
    blinded: Sequence[bool],
    *,
    keys: jnp.ndarray | None = None,
) -> tuple[tuple[FitResult, ...], tuple[dict, ...]]:
    """Run one independently adapted NUTS sampler per realisation."""
    names = free_parameter_names(config)
    centres, _, _, _ = _minimise_chi2(diagrams, arrays, config, priors)
    centers = {name: centres[:, index] for index, name in enumerate(names)}
    model = _model_for_data(diagrams, config, priors)
    model_arrays = _model_arrays(arrays, priors)
    realisation_count = len(diagrams)

    def fit_one(key, init, *one_arrays):
        mcmc = MCMC(
            NUTS(
                model,
                init_strategy=init_to_value(values=init),
                target_accept_prob=config.target_accept,
                dense_mass=True,
            ),
            num_warmup=config.nuts_warmup,
            num_samples=config.nuts_samples,
            num_chains=config.nuts_chains,
            chain_method="vectorized",
            progress_bar=False,
        )
        mcmc.run(key, *one_arrays, extra_fields=("diverging",))
        return (
            mcmc.get_samples(group_by_chain=True),
            mcmc.get_extra_fields(group_by_chain=True)["diverging"],
        )

    if keys is None:
        base_key = jax.random.fold_in(jax.random.PRNGKey(config.seed), 2)
        keys = jax.vmap(lambda index: jax.random.fold_in(base_key, index))(
            jnp.arange(realisation_count)
        )
    grouped_all, diverging = jax.vmap(fit_one)(keys, centers, *model_arrays)
    grouped_all = {
        name: np.asarray(values) for name, values in grouped_all.items()
    }
    diverging = np.asarray(diverging)

    results: list[FitResult] = []
    grids: list[dict] = []
    for realisation_index in range(realisation_count):
        grouped = {
            name: np.asarray(values[realisation_index])
            for name, values in grouped_all.items()
        }
        flat = {
            name: values.reshape((-1,) + values.shape[2:])
            for name, values in grouped.items()
        }
        means = {
            name: float(np.mean(value)) for name, value in flat.items()
        }
        summaries, covariance, correlation = _summaries_from_samples(
            flat, means, config.sig_type, config.marg,
            free_parameter_names(config),
        )
        diagnostic_table = numpyro_summary(
            grouped, prob=0.68, group_by_chain=True,
        )
        rhat = {
            name: float(np.max(np.asarray(values["r_hat"])))
            for name, values in diagnostic_table.items()
        }
        ess = {
            name: float(np.min(np.asarray(values["n_eff"])))
            for name, values in diagnostic_table.items()
        }
        divergence_count = int(np.sum(diverging[realisation_index]))
        success = (
            divergence_count == 0
            and all(
                value <= config.max_rhat for value in rhat.values()
            )
        )
        low_ess = {
            name: value
            for name, value in ess.items()
            if value < config.min_ess
        }
        result_diagnostics = diagnostics(
            success=success,
            message=(
                "" if success else "NUTS convergence diagnostics failed"
            ),
            divergences=divergence_count,
            rhat=rhat,
            ess=ess,
        )
        warnings = []
        if low_ess:
            values = ", ".join(
                f"{name}={value:.1f}" for name, value in low_ess.items()
            )
            warnings.append(
                "NUTS ESS is below the advisory threshold "
                f"{config.min_ess:.1f}: {values}"
            )
        results.append(_finish_result(
            method="nuts",
            summaries=summaries,
            covariance=covariance,
            correlation=correlation,
            diagnostics=result_diagnostics,
            diagram=diagrams[realisation_index],
            arrays=arrays,
            realisation_index=realisation_index,
            config=config,
            priors=priors,
            blinded=blinded[realisation_index],
            samples=grouped,
            warnings=warnings,
        ))
        grids.append({})
    return tuple(results), tuple(grids)


def _grid_axes(config: WFITConfig) -> dict[str, np.ndarray]:
    axes: dict[str, np.ndarray] = {}
    for name in free_parameter_names(config):
        lo, hi = config.bounds(name)
        count = config.steps(name)
        if count < 2:
            raise SNANAError(
                f"grid axis {name} requires at least two steps",
                function_name="fit_grid",
            )
        axes[name] = np.linspace(lo, hi, count)
    return axes


def _make_grid_evaluator(
    names: tuple[str, ...],
    diagrams: Sequence[HubbleDiagram],
    arrays: tuple,
    config: WFITConfig,
    priors: Mapping[str, jnp.ndarray],
) -> Any:
    def evaluate(*values):
        params = dict(zip(names, values))
        components = evaluate_likelihood(
            params, *arrays,
            priors["omega_m_mean"], priors["cmb_r"],
            config, priors, hdibc=_hdibc_arrays(diagrams[0]),
        )
        return components["hd_chi2"], components["total_chi2"]

    return jax.jit(jax.vmap(evaluate))


def _grid_chunk_chi2(
    evaluator: Any,
    coordinates: dict[str, np.ndarray],
    names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    sn, total = evaluator(
        *(jnp.asarray(coordinates[name]) for name in names))
    return np.asarray(sn), np.asarray(total)


def _iter_grid_chunks(axes: dict[str, np.ndarray], chunk_size: int):
    names = tuple(axes)
    shape = tuple(axes[name].size for name in names)
    total = int(np.prod(shape))
    for start in range(0, total, chunk_size):
        flat_index = np.arange(start, min(start + chunk_size, total))
        multi = np.unravel_index(flat_index, shape)
        yield start, {
            name: axes[name][multi[index]]
            for index, name in enumerate(names)
        }, {
            name: multi[index] for index, name in enumerate(names)
        }


def fit_grid(
    diagrams: Sequence[HubbleDiagram],
    arrays: tuple,
    config: WFITConfig,
    priors: Mapping[str, jnp.ndarray],
    blinded: Sequence[bool],
) -> tuple[tuple[FitResult, ...], tuple[dict, ...]]:
    """Fit every HD realisation with one vmapped grid evaluator."""
    axes = _grid_axes(config)
    names = tuple(axes)
    shape = tuple(axis.size for axis in axes.values())
    realisation_count = len(diagrams)
    if config.grid_chunk_size > 0:
        chunk_size = max(1, config.grid_chunk_size // realisation_count)
    else:
        # the evaluator's quadrature temporary is (chunk, R, N, 64) float64;
        # bound it to ~512 MB whatever the Hubble-diagram length
        n_rows = max(int(d.mu.size) for d in diagrams)
        budget = (512 * 1024 ** 2) // 8
        chunk_size = max(1, budget // (realisation_count * n_rows * 64))
    keep_full = bool(config.outfile_chi2grid)
    total_points = int(np.prod(shape))
    full_chi2 = np.empty((total_points, realisation_count))
    full_chi2_sn = (
        np.empty((total_points, realisation_count)) if keep_full else None
    )
    evaluator = _make_grid_evaluator(
        names, diagrams, arrays, config, priors,
    )

    minima = np.full(realisation_count, np.inf)
    minimum_points = {
        name: np.zeros(realisation_count, dtype=np.float64) for name in names
    }
    realisation_indices = np.arange(realisation_count)
    for start, coordinates, _ in _iter_grid_chunks(axes, chunk_size):
        chi2_sn, chi2 = _grid_chunk_chi2(evaluator, coordinates, names)
        full_chi2[start:start + chi2.shape[0], :] = chi2
        if full_chi2_sn is not None:
            full_chi2_sn[start:start + chi2.shape[0], :] = chi2_sn
        local = np.nanargmin(chi2, axis=0)
        local_minima = chi2[local, realisation_indices]
        better = local_minima < minima
        minima = np.where(better, local_minima, minima)
        for name in names:
            candidates = coordinates[name][local]
            minimum_points[name] = np.where(
                better, candidates, minimum_points[name],
            )

    marginals = {
        name: np.zeros((realisation_count, axis.size))
        for name, axis in axes.items()
    }
    first = np.zeros((len(names), realisation_count))
    second = np.zeros((len(names), len(names), realisation_count))
    total_weight = np.zeros(realisation_count)
    full_weight = (
        np.empty((total_points, realisation_count)) if keep_full else None
    )
    for start, coordinates, axis_indices in _iter_grid_chunks(axes, chunk_size):
        count = len(next(iter(coordinates.values())))
        chi2 = full_chi2[start:start + count, :]
        weight = np.exp(-0.5 * (chi2 - minima[None, :]))
        weight = np.where(np.isfinite(weight), weight, 0.0)
        if full_weight is not None:
            full_weight[start:start + weight.shape[0], :] = weight
        total_weight += np.sum(weight, axis=0)
        coords = np.stack([coordinates[name] for name in names])
        first += coords @ weight
        second += np.einsum("ac,cr,bc->abr", coords, weight, coords)
        for name in names:
            indices = axis_indices[name]
            size = axes[name].size
            flat = (np.arange(realisation_count)[:, None] * size
                    + indices[None, :]).ravel()
            marginals[name] += np.bincount(
                flat, weights=weight.T.ravel(),
                minlength=realisation_count * size,
            ).reshape(realisation_count, size)
    if np.any(~np.isfinite(total_weight)) or np.any(total_weight <= 0):
        raise SNANAError(
            "one or more grid posteriors have zero or non-finite weight",
            function_name="fit_grid",
        )

    results: list[FitResult] = []
    grids: list[dict] = []
    for realisation_index in range(realisation_count):
        mean_vector = first[:, realisation_index] / total_weight[
            realisation_index
        ]
        means = dict(zip(names, mean_vector))
        covariance = (
            second[:, :, realisation_index] / total_weight[realisation_index]
            - np.outer(mean_vector, mean_vector)
        )
        std = np.sqrt(np.diag(covariance))
        denominator = np.outer(std, std)
        correlation = np.divide(
            covariance,
            denominator,
            out=np.eye(len(names)),
            where=denominator > 0,
        )

        summaries: dict[str, dict[str, float]] = {}
        edge_ratios: dict[str, tuple[float, float]] = {}
        marginal_probability: dict[str, np.ndarray] = {}
        lower_probability = 0.5 - 0.683 / 2.0
        upper_probability = 0.5 + 0.683 / 2.0
        for parameter_index, name in enumerate(names):
            probability = (
                marginals[name][realisation_index]
                / total_weight[realisation_index]
            )
            marginal_probability[name] = probability
            cdf = np.cumsum(probability)
            summaries[name] = summary(
                center=float(minimum_points[name][realisation_index]),
                mean=means[name],
                median=float(np.interp(0.5, cdf, axes[name])),
                std=float(std[parameter_index]),
                q16=float(np.interp(
                    lower_probability, cdf, axes[name],
                )),
                q84=float(np.interp(
                    upper_probability, cdf, axes[name],
                )),
                sig_type=config.sig_type,
                marg=config.marg,
            )
            peak = float(np.max(probability))
            edge_ratios[name] = (
                float(probability[0] / peak),
                float(probability[-1] / peak),
            )
        warnings = [
            f"{name} grid posterior reaches an axis boundary"
            for name, ratios in edge_ratios.items()
            if max(ratios) > _EDGE_PROBABILITY_WARN
        ]
        result_diagnostics = diagnostics(
            success=True,
            edge_probability_ratio=edge_ratios,
        )
        grid = {
            "axes": axes,
            "chi2": (
                full_chi2[:, realisation_index].reshape(shape)
                if keep_full else None
            ),
            "chi2_sn": (
                full_chi2_sn[:, realisation_index].reshape(shape)
                if full_chi2_sn is not None else None
            ),
            "weight": (
                (
                    full_weight[:, realisation_index]
                    / total_weight[realisation_index]
                ).reshape(shape)
                if full_weight is not None else None
            ),
            "marginal_probability": marginal_probability,
        }
        results.append(_finish_result(
            method="grid",
            summaries=summaries,
            covariance=covariance,
            correlation=correlation,
            diagnostics=result_diagnostics,
            diagram=diagrams[realisation_index],
            arrays=arrays,
            realisation_index=realisation_index,
            config=config,
            priors=priors,
            blinded=blinded[realisation_index],
            warnings=warnings,
        ))
        grids.append(grid)
    return tuple(results), tuple(grids)
