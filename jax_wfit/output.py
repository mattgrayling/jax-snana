"""WFIT YAML and optional SNANA-compatible output writers."""

import math
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import numpy as np
import numpyro
import yaml
from scipy.linalg import cho_factor, cho_solve

from jax_snlc_core.cosmology import distance_modulus
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import dense_factorized_covariance
from jax_snlc_core.utils import LIGHT_KM

from .blinding import apply_blinding, parameter_offsets, parameter_random_integers
from .config import OMEGA_MATTER_DEFAULT, W0_DEFAULT, WA_DEFAULT, WFITConfig
from .hubble import HubbleDiagram
from .inference import free_parameter_names
from .likelihood import hdibc_observed_mu, offset_marginalised_chi2


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, jax.Array) and value.shape == ():
        return float(value)
    return value


def result_to_dict(
    result: Mapping[str, object],
    config: WFITConfig,
    diagram: HubbleDiagram,
    realisation_index: int,
) -> dict[str, Any]:
    components = result["likelihood"]
    blinded = result["blinded"]
    point_estimate = apply_blinding(result["point_estimate"], config, blinded)
    offsets = parameter_offsets(config) if blinded else None
    parameter_summaries = {}
    for name, summary in result["summaries"].items():
        values = dict(summary)
        if offsets is not None:
            for field_name in (
                "center", "mean", "median", "q16", "q84", "value",
            ):
                values[field_name] += offsets[name]
        parameter_summaries[name] = values
    return {
        "success": result["diagnostics"]["success"],
        "method": result["method"],
        "point_estimate_kind": result["point_estimate_kind"],
        "point_estimate": point_estimate,
        "parameters": parameter_summaries,
        "parameter_order": list(free_parameter_names(config)),
        "covariance": result["covariance"],
        "correlation": result["correlation"],
        "fit": {
            "chi2_hd": components["hd_chi2"],
            "chi2_omega_m": components["omega_m_chi2"],
            "chi2_cmb": components["cmb_chi2"],
            "chi2_bao": components["bao_chi2"],
            "chi2_total": components["total_chi2"],
            "intercept": components["intercept"],
            "ndof_hd": result["ndof_hd"],
            "n_external_constraints": result["n_external"],
            "external_prior_targets": result["prior_targets"],
            "n_rows": diagram.n_rows,
        },
        "diagnostics": dict(result["diagnostics"]),
        "warnings": result["warnings"],
        "blinded": blinded,
        "input": {
            "realisation_index": realisation_index,
            "hubble_diagram": diagram.source,
            "covariance_kind": diagram.covariance_kind,
            "retained_raw_indices": diagram.raw_indices,
            "metadata": diagram.meta,
            "simulation_truth": diagram.truth,
        },
    }


def _shared_metadata(
    config: WFITConfig,
    summary_path: str,
    redact_seed: bool,
) -> dict[str, Any]:
    configuration = config.to_dict()
    configuration["cospar_yaml"] = summary_path
    if redact_seed:
        configuration["blind_seed"] = "REDACTED"
    return {
        "assumptions": {
            "radiation": config.radiation,
            "cmb_radiation": False,
            "bao_dh_radiation": False,
            "zerr_propagated": False,
        },
        "configuration": configuration,
        "provenance": {
            "jax": jax.__version__,
            "numpyro": numpyro.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }


def ensemble_summary(
    results: Sequence[Mapping[str, object]],
    diagrams: Sequence[HubbleDiagram],
    config: WFITConfig,
) -> dict[str, dict[str, float | None]]:
    """Summarize recovery bias when truth is available for every realisation."""
    summaries: dict[str, dict[str, float | None]] = {}
    for name in free_parameter_names(config):
        truth_values = [
            diagram.truth.get(name) if diagram.truth is not None else None
            for diagram in diagrams
        ]
        if any(value is None for value in truth_values):
            continue
        truths = np.asarray(truth_values)
        estimates = np.asarray([r["summaries"][name]["value"] for r in results])
        errors = np.asarray([r["summaries"][name]["sigma"] for r in results])
        biases = estimates - truths
        pulls = biases / errors
        covered = np.asarray([
            r["summaries"][name]["q16"] <= truth <= r["summaries"][name]["q84"]
            for r, truth in zip(results, truths)
        ])
        count = len(results)
        bias_std = float(np.std(biases, ddof=1)) if count > 1 else None
        summaries[name] = {
            "count": count,
            "mean_bias": float(np.mean(biases)),
            "bias_std": bias_std,
            "standard_error": (
                bias_std / np.sqrt(count) if bias_std is not None else None
            ),
            "pull_mean": float(np.mean(pulls)),
            "pull_std": (
                float(np.std(pulls, ddof=1)) if count > 1 else None
            ),
            "coverage_68": float(np.mean(covered)),
        }
    return summaries


def write_yaml(
    path: str,
    results: Sequence[Mapping[str, object]],
    config: WFITConfig,
    diagrams: Sequence[HubbleDiagram],
    cpu_minutes: float = 0.0,
) -> None:
    blinded = [result["blinded"] for result in results]
    metadata = _shared_metadata(
        config, summary_path=path, redact_seed=any(blinded),
    )
    if len(results) == 1:
        payload = result_to_dict(
            results[0], config, diagrams[0], 0,
        )
        payload.update(metadata)
    else:
        payload = {
            "n_realisations": len(results),
            **metadata,
            "realisations": [
                result_to_dict(result, config, diagrams[index], index)
                for index, result in enumerate(results)
            ],
        }
    if not any(blinded):
        ensemble = ensemble_summary(results, diagrams, config)
        if ensemble:
            payload["ensemble"] = ensemble
    # the batch aggregator sums this key (wfit.c:5314 writes it to the YAML)
    payload["CPU_MINUTES"] = float(cpu_minutes)
    with open(path, "wt") as stream:
        yaml.safe_dump(
            _plain(payload),
            stream,
            sort_keys=False,
        )


def _resolved_cosmology(
    result: Mapping[str, object], config: WFITConfig,
) -> tuple[float, float, float]:
    """The cosmology the residual file is evaluated at.

    Blinded, when the fit is: ``write_output_resid`` builds its ``cpar`` from
    ``WORKSPACE.omm_final``/``w0_final``/``wa_final`` (wfit.c:5345-5348), which
    ``wfit_final`` has already offset by ``sin(ran)`` (wfit.c:3922-3935).
    """
    point = apply_blinding(
        result["point_estimate"], config, result["blinded"])
    omega_m = point["omega_m"]
    w0 = -1.0 if config.model == "lcdm" else point["w0"]
    wa = point["wa"] if config.model == "w0wa" else 0.0
    return omega_m, w0, wa


def write_residuals(
    path: str,
    result: Mapping[str, object],
    config: WFITConfig,
    diagram: HubbleDiagram,
) -> None:
    omega_m, w0, wa = _resolved_cosmology(result, config)
    mu_model = np.asarray(distance_modulus(
        diagram.redshift, diagram.redshift, omega_m, 1.0 - omega_m,
        LIGHT_KM / config.h0, w0, wa, radiation=config.radiation))
    hdibc = diagram.mu_second is not None
    if hdibc:
        mu_observed_raw, interpolation_fraction = hdibc_observed_mu(
            mu_model, diagram.mu, diagram.mu_second,
            diagram.bias0, diagram.bias1,
        )
    else:
        mu_observed_raw = diagram.mu
        interpolation_fraction = np.zeros_like(np.asarray(diagram.mu))
    # muresid_avg at the written cosmology, blinding included (wfit.c:3952-3956, :5375)
    intercept = float(offset_marginalised_chi2(
        np.asarray(mu_observed_raw) - mu_model,
        diagram.precision,
        np.ones(mu_model.shape, dtype=bool),
        config.intercept_sigma,
    )[1])
    mu_observed = np.asarray(mu_observed_raw) - intercept
    residual = mu_observed - mu_model
    diagonal_chi2 = (
        residual**2
        / np.asarray(diagram.muerr)**2
    )
    # mu_sim: the distance at the simulated cosmology (write_output_resid,
    # wfit.c:5358-5360), which COSPAR_SIM carries (wfit.c:1272-1275)
    om_sim = (config.om_sim if config.om_sim is not None
              else OMEGA_MATTER_DEFAULT)
    w0_sim = config.w0_sim if config.w0_sim is not None else W0_DEFAULT
    wa_sim = config.wa_sim if config.wa_sim is not None else WA_DEFAULT
    mu_sim = np.asarray(distance_modulus(
        diagram.redshift, diagram.redshift, om_sim, 1.0 - om_sim,
        LIGHT_KM / config.h0, w0_sim, wa_sim, radiation=config.radiation))
    with open(path, "wt") as stream:
        stream.write("# mu_res   = mu_obs - mu_model \n")
        stream.write("# mu_model = distance computed from best-fit "
                     "cosmology model.\n")
        stream.write(f"# mu_sim   = distance computed from "
                     f"{{ OM, ODE, w0, wa }} = "
                     f"{{ {om_sim:.3f} {1.0 - om_sim:.3f} {w0_sim:.3f} "
                     f"{wa_sim:.3f} }} \n")
        stream.write("# chi2_diag = mu_res^2 / mu_sig^2 \n")
        stream.write("\n")
        columns = ("VARNAMES: "
                   "CID   zHD  mu_obs  mu_model  mu_sim  mu_res  mu_sig  "
                   "chi2_diag")
        if hdibc:
            columns += ("  mu_obs0_hdibc mu_obs1_hdibc mu_bcor0_hdibc "
                        "mu_bcor1_hdibc z_dif_hdibc f_interp_hdibc")
        stream.write(f"{columns}\n")
        z_dif = (np.asarray(diagram.z_second) - np.asarray(diagram.redshift)
                 if hdibc else np.zeros_like(mu_model))
        for values in zip(
            diagram.cid,
            np.asarray(diagram.redshift),
            mu_observed,
            mu_model,
            mu_sim,
            residual,
            np.asarray(diagram.muerr),
            diagonal_chi2,
            np.asarray(diagram.mu),
            np.asarray(diagram.mu_second if hdibc else diagram.mu),
            np.asarray(diagram.bias0 if hdibc else diagram.mu),
            np.asarray(diagram.bias1 if hdibc else diagram.mu),
            z_dif,
            np.asarray(interpolation_fraction),
        ):
            (cid, z, obs, model, sim, resid, sigma, chi2,
             obs0, obs1, bcor0, bcor1, zd, fraction) = values
            line = (
                f"SN: {cid!s:<12} {z:7.4f} {obs:7.4f} {model:7.4f} "
                f"{sim:7.4f}  {resid:7.4f} {sigma:7.4f} {chi2:7.1f}  "
            )
            if hdibc:
                line += (f"  {obs0:7.4f} {obs1:7.4f} {bcor0:7.4f} "
                         f"{bcor1:7.4f} {zd:7.4f} {fraction:.3f}")
            stream.write(line + "\n")


def _cospar_entries(
    result: Mapping[str, object],
    config: WFITConfig,
) -> list[tuple[str, str]]:
    """The (name, value) list of write_output_cospar (wfit.c:5122-5277):
    w first, sigma columns split by -marg, then the reduced covariances,
    FoM, chi2, Ndof, sigint and the blinding integers."""
    var_w = "w0" if config.model == "w0wa" else "w"
    blinded = result["blinded"]
    free = free_parameter_names(config)
    summaries = result["summaries"]
    reported = apply_blinding(
        {n: summaries[n]["value"] for n in free}, config, blinded)
    rans = parameter_random_integers(config) if blinded else {}
    idx = {name: i for i, name in enumerate(free)}
    correlation = np.asarray(result["correlation"])

    def rho(a, b):
        if a in idx and b in idx:
            return float(correlation[idx[a], idx[b]])
        return 0.0

    entries: list[tuple[str, str]] = []

    fixed = {"w0": -1.0, "wa": 0.0}

    def par_block(name, label, sig_fmt="%8.5f"):
        if name not in summaries:
            # fixed parameters still get their columns (wfit.c:5164)
            entries.append((label, "%8.5f" % fixed[name]))
            if config.marg or name == "omega_m":
                entries.append((f"{label}sig_marg", sig_fmt % 0.0))
            else:
                entries.append((f"{label}sig_lo", sig_fmt % 0.0))
                entries.append((f"{label}sig_up", sig_fmt % 0.0))
            return
        s = summaries[name]
        entries.append((label, "%8.5f" % reported[name]))
        if config.marg:
            entries.append((f"{label}sig_marg", sig_fmt % s["sigma"]))
        elif name == "omega_m":
            # wfit.c has no omm_sig_lower/upper and writes sig_marg
            entries.append((f"{label}sig_marg", sig_fmt % s["sigma"]))
        else:
            entries.append((f"{label}sig_lo", sig_fmt % (s["mean"] - s["q16"])))
            entries.append((f"{label}sig_up", sig_fmt % (s["q84"] - s["mean"])))

    par_block("w0", var_w)
    if config.model == "w0wa":
        par_block("wa", "wa", sig_fmt="%7.4f" if not config.marg else "%8.5f")
    par_block("omega_m", "om")

    entries.append((f"rho_{var_w}om", "%6.3f" % rho("w0", "omega_m")))
    if config.model == "w0wa":
        entries.append((f"rho_{var_w}wa", "%6.3f" % rho("w0", "wa")))
        entries.append(("rho_waom", "%6.3f" % rho("wa", "omega_m")))   # wfit.c:5233

    # FoM = 1/(sig_w * sig_partner * sqrt(1-rho^2)) (wfit_FoM,
    # wfit.c:3988-4023); the partner is wa for w0wa, omega_m for wCDM
    if config.model == "w0wa":
        partner, rho_fom = "wa", rho("w0", "wa")
    else:
        partner, rho_fom = "omega_m", rho("w0", "omega_m")
    fom = -9.0
    if "w0" in summaries and partner in summaries:
        if abs(rho_fom) > 1.0:
            raise SNANAError(
                f"invalid rho({var_w}{partner}) = {rho_fom}; check the "
                "covariance calculation",
                function_name="_cospar_entries",
            )
        sig_product = (summaries["w0"]["sigma"] * summaries[partner]["sigma"]
                       * math.sqrt(1.0 - rho_fom * rho_fom))
        # a zero or negative product reports -9, as wfit_FoM does (:4017-4020)
        if sig_product > 0.0:
            fom = 1.0 / sig_product
    entries.append(("FoM", "%6.1f" % fom))

    entries.append(("chi2", "%.1f" % float(result["likelihood"]["total_chi2"])))
    entries.append(("Ndof", "%5d" % int(result["ndof_hd"])))
    # single-fit runs report sigint 0 (wfit_final, wfit.c:3978-3980)
    entries.append(("sigint", "%6.3f" % 0.0))
    entries.append((f"{var_w}ran", "%d" % rans.get("w0", 0)))
    if config.model == "w0wa":
        entries.append(("waran", "%d" % rans.get("wa", 0)))
    entries.append(("omran", "%d" % rans.get("omega_m", 0)))
    entries.append(("label", str(config.label)))
    return entries


def write_cospar(
    path: str,
    result: Mapping[str, object],
    config: WFITConfig,
) -> None:
    entries = _cospar_entries(result, config)
    with open(path, "wt") as stream:
        stream.write("# " + " ".join(n for n, _ in entries) + " \n")
        stream.write(" ".join(v for _, v in entries) + " \n")


_GRID_LABELS = {"omega_m": "omm", "w0": "w0", "wa": "wa"}


def write_grid(
    path: str,
    grid: Mapping[str, object],
    config: WFITConfig,
    blinded: bool,
) -> None:
    """write_output_chi2grid (wfit.c:5436-5541): both chi2 columns as offsets
    from their grid minima, every row (the -weightmin cut is not carried)."""
    if grid.get("chi2") is None or grid.get("weight") is None:
        raise SNANAError(
            "grid output was requested but the full grid was not retained",
            function_name="write_grid",
        )
    names = tuple(grid["axes"])
    axes = grid["axes"]
    # raw integers, not the sine offsets write_cospar uses (wfit.c:3923)
    offsets = parameter_random_integers(config) if blinded else None
    ran = {"omega_m": 0.0, "w0": 0.0, "wa": 0.0} if offsets is None else offsets
    coordinates = [
        column.ravel()
        for column in np.meshgrid(
            *(axes[name] + ran[name] for name in names), indexing="ij",
        )
    ]
    chi2_sn = grid["chi2_sn"].ravel()
    chi2 = grid["chi2"].ravel()
    weight = grid["weight"].ravel() + 1.0e-50
    snchi = chi2_sn - np.min(chi2_sn)
    extchi = chi2 - np.min(chi2)
    varlist = " ".join(_GRID_LABELS[name] for name in names)
    coord_fmt = " ".join(["%.4f"] * len(names))
    template = "ROW: %4d   " + coord_fmt + "   %12.5e %12.5e  %12.5e\n"
    with open(path, "wt") as stream:
        stream.write(f"# Write {chi2.size} of {chi2.size} rows passing "
                     f"weight > 0.00e+00 \n")
        stream.write("# chi2_sn:   chi2 contribution from SN only\n")
        stream.write("# chi2_tot:  total chi2 from SN plus priors\n")
        stream.write("# weight:    weight used by chainconsumer "
                     "(added April 2025)\n")
        for name in names:
            step = float(axes[name][1] - axes[name][0]) \
                if len(axes[name]) > 1 else 0.0
            stream.write(f"# {_GRID_LABELS[name]} step size: {step:.4f} \n")
        stream.write("\n")
        stream.write(f"VARNAMES: ROW {varlist}   chi2_sn chi2_tot weight\n")
        stream.writelines(
            template % row
            for row in zip(range(1, chi2.size + 1), *coordinates,
                           snchi, extchi, weight)
        )


def write_prob1d(
    path: str,
    grid: Mapping[str, object],
    config: WFITConfig,
) -> None:
    """The marginal probability table (get_prob_1Dmarg): per-variable row
    numbering from 0, SNANA's short variable names and field widths."""
    if not grid.get("axes"):
        raise SNANAError(
            "probability output requires grid inference",
            function_name="write_prob1d",
        )
    var_w = "w0" if config.model == "w0wa" else "w"
    labels = {"omega_m": "om", "w0": var_w, "wa": "wa"}
    with open(path, "wt") as stream:
        stream.write("VARNAMES:  ROW  NAME_COSPAR  VAL  PROB  CDF \n")
        for name, axis in grid["axes"].items():
            probability = grid["marginal_probability"][name]
            cdf = np.cumsum(probability)
            for i, (value, prob, cumulative) in enumerate(
                    zip(axis, probability, cdf)):
                stream.write(
                    f"ROW:  {i:3d}  {labels[name]:>4s}  {value:.5f}  "
                    f"{prob:.5f}  {cumulative:.5f} \n"
                )


def _realisation_path(path: str, index: int, count: int) -> str:
    if count == 1:
        return path
    source = Path(path)
    suffix = "".join(source.suffixes)
    stem = source.name[:-len(suffix)] if suffix else source.name
    return str(source.with_name(f"{stem}.r{index:03d}{suffix}"))


def _final_precision(diagram: HubbleDiagram) -> np.ndarray:
    if diagram.covariance_kind == "factorized":
        # Explicit dense output is opt-in. Reconstruct from the host float64
        # covariance, rather than subtracting rounded float32 factor products.
        covariance = diagram.covariance
        weighted = covariance["U"] / covariance["diag"][:, None]
        small = np.eye(covariance["U"].shape[1]) + covariance["U"].T @ weighted
        return (np.diag(1. / covariance["diag"])
                - weighted @ cho_solve(cho_factor(small), weighted.T))
    precision = np.asarray(diagram.precision)
    if diagram.covariance_kind == "diagonal":
        return np.diag(precision)
    return precision


def _final_covariance(diagram: HubbleDiagram) -> np.ndarray:
    if diagram.covariance_kind == "factorized":
        return dense_factorized_covariance(diagram.covariance)
    if diagram.covariance is not None:
        return np.asarray(diagram.covariance)
    return np.linalg.inv(_final_precision(diagram))


def write_covariance_matrix(path: str, matrix: np.ndarray) -> None:
    """Write a dense matrix in SNANA's dimension-plus-elements format, with
    wfit.exe's element format (wfit.c:4102)."""
    with open(path, "wt") as stream:
        stream.write(f"{matrix.shape[0]}\n")
        for value in matrix.ravel():
            stream.write(f"{value:14.8e}\n")


def write_outputs(
    results: Sequence[Mapping[str, object]],
    grids: Sequence[Mapping[str, object]],
    config: WFITConfig,
    diagrams: Sequence[HubbleDiagram],
    cpu_minutes: float = 0.0,
) -> None:
    write_yaml(config.cospar_yaml, results, config, diagrams, cpu_minutes)
    for index, (result, grid) in enumerate(zip(results, grids)):
        if config.outfile_resid:
            write_residuals(
                _realisation_path(
                    config.outfile_resid, index, len(results),
                ),
                result,
                config,
                diagrams[index],
            )
        if config.outfile_cospar:
            write_cospar(
                _realisation_path(
                    config.outfile_cospar, index, len(results),
                ),
                result,
                config,
            )
        if config.outfile_chi2grid:
            write_grid(
                _realisation_path(
                    config.outfile_chi2grid, index, len(results),
                ),
                grid,
                config,
                result["blinded"],
            )
        if config.outfile_prob1d:
            write_prob1d(
                _realisation_path(
                    config.outfile_prob1d, index, len(results),
                ),
                grid,
                config,
            )
        if config.output_samples:
            if result["samples"] is None:
                raise SNANAError(
                    "sample output requested for a method without samples",
                    function_name="write_outputs",
                )
            np.savez(
                _realisation_path(
                    config.output_samples, index, len(results),
                ),
                **apply_blinding(
                    result["samples"], config, result["blinded"],
                ),
            )
        if config.output_covariance:
            write_covariance_matrix(
                _realisation_path(
                    config.output_covariance, index, len(results),
                ),
                _final_covariance(diagrams[index]),
            )
        if config.outfile_mucovtot_inv:
            write_covariance_matrix(
                _realisation_path(
                    config.outfile_mucovtot_inv, index, len(results),
                ),
                _final_precision(diagrams[index]),
            )
