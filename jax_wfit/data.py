"""Hubble-diagram and covariance ingestion mirroring SNANA WFIT."""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from scipy.linalg import cho_solve

from jax_snlc_core.io.cov_matrix import read_covariance_npz, read_covariance_text
from jax_snlc_core.io.fitres_reader import read_fitres
from jax_snlc_core.io.sim_cosmology import parse_mushift
from jax_snlc_core.physics.constants import LIGHT_KM
from jax_snlc_core.physics.cosmology import distance_modulus
from jax_snlc_core.utils import SNANAError

from jax_snlc_core.io.factorized_covariance import (
    validate_factorized_covariance, read_factorized_covariance)
from .factorized import prepare_precision
from .config import WFITConfig


#: HDIBC syncs redshifts only for diagrams this size or smaller (wfit.c:2282)
_HDIBC_SYNC_MAX_ROWS = 100

_CID_ALIASES = ("CID", "ROW")
_Z_ALIASES = ("zHD", "zCMB", "z", "Z")
_ZERR_ALIASES = ("zHDERR", "zCMBERR", "zERR", "ZERR")
_MU_ALIASES = ("MU", "DLMAG", "MUDIF")
_MUERR_ALIASES = ("MUERR", "DLMAGERR", "MU_ERR", "MUDIFERR")


def _find_column(columns, aliases, *, required=True) -> str | None:
    for name in aliases:
        if name in columns:
            return name
    if required:
        raise SNANAError(
            f"none of the required columns {aliases} are present",
            function_name="_find_column",
        )
    return None


def _load_matrix(path: str) -> np.ndarray:
    source = Path(path)
    if source.suffix.lower() == ".npz":
        with np.load(source) as payload:
            if "cov" not in payload.files:
                raise SNANAError(
                    f"{path} has no 'cov' array",
                    function_name="_load_matrix",
                )
            matrix = np.asarray(payload["cov"], dtype=np.float64)
            has_nsn = "nsn" in payload.files
        if matrix.ndim == 2:
            return matrix
        # the order is 'nsn'; the upper triangle alone is ambiguous
        if not has_nsn:
            raise SNANAError(
                f"{path} holds a 1-D covariance but no 'nsn'; its order cannot "
                "be recovered",
                function_name="_load_matrix",
            )
        return read_covariance_npz(source)

    return read_covariance_text(source)


@dataclass(frozen=True)
class HubbleDiagram:
    """One loaded Hubble diagram: its arrays, its covariance and its metadata.

    ``mu_second``/``bias0``/``bias1`` are set only for the combined diagram
    that HDIBC builds from two inputs; ``bias0``/``bias1`` already carry the
    ``MUSHIFT`` of their bias-correction cosmology.
    """

    cid: np.ndarray
    redshift: jnp.ndarray
    mu: jnp.ndarray
    muerr: jnp.ndarray
    precision: jnp.ndarray | dict[str, jnp.ndarray]
    covariance_kind: str
    covariance: jnp.ndarray | dict | None = None
    raw_indices: np.ndarray | None = None
    zerr: np.ndarray | None = None
    meta: dict[str, str] = field(default_factory=dict)
    source: str = ""
    truth: dict[str, float | str] | None = None
    mu_second: jnp.ndarray | None = None
    z_second: jnp.ndarray | None = None
    bias0: jnp.ndarray | None = None
    bias1: jnp.ndarray | None = None
    snrms: float = 0.0  # extra scatter already included in precision/covariance

    @property
    def n_rows(self) -> int:
        return int(self.mu.shape[0])

    @property
    def is_data_real(self) -> bool | None:
        return _parse_is_data_real(self.meta, self.source)


def stack_arrays(diagrams: Sequence[HubbleDiagram]):
    """Pad every diagram to the longest and stack along a leading axis.

    Returns exactly the arrays the likelihood evaluates, in its argument
    order: ``redshift, mu, precision, valid``. Per-realisation quantities the
    writers need, such as ``muerr``, are read from the diagrams themselves.
    """
    kind = diagrams[0].covariance_kind
    for index, diagram in enumerate(diagrams[1:], start=1):
        if diagram.covariance_kind != kind:
            raise SNANAError(
                "vmapped WFIT requires one covariance representation; "
                f"realisation 0 is {kind!r}, realisation {index} is "
                f"{diagram.covariance_kind!r}",
                function_name="stack_arrays",
            )
    row_counts = np.asarray(
        [diagram.n_rows for diagram in diagrams], dtype=np.int64,
    )
    padded = int(np.max(row_counts))

    def pad_vector(value, fill):
        width = padded - int(value.shape[0])
        return jnp.pad(jnp.asarray(value), (0, width), constant_values=fill)

    def pad_matrix(value):
        width = padded - int(value.shape[0])
        return jnp.pad(jnp.asarray(value), ((0, width), (0, width)))

    if kind == "factorized":
        rank = max(d.precision["design"].shape[1] for d in diagrams)

        def pad_factors(name):
            return jnp.stack([jnp.pad(d.precision[name],
                ((0, padded - d.n_rows), (0, rank - d.precision["design"].shape[1])))
                for d in diagrams])

        weights = {
            "invstd": jnp.stack([pad_vector(d.precision["invstd"], 0.) for d in diagrams]),
            "design": pad_factors("design"), "mapping": pad_factors("mapping"),
            "weighted_ones": jnp.stack([pad_vector(d.precision["weighted_ones"], 0.) for d in diagrams]),
            "sum_weights": jnp.stack([d.precision["sum_weights"] for d in diagrams]),
        }
    else:
        weights = jnp.stack([pad_vector(d.precision, 0.) if kind == "diagonal"
                             else pad_matrix(d.precision) for d in diagrams])
    return (
        jnp.stack([pad_vector(d.redshift, 1.0) for d in diagrams]),
        jnp.stack([pad_vector(d.mu, 0.0) for d in diagrams]),
        weights,
        jnp.arange(padded)[None, :] < jnp.asarray(row_counts)[:, None],
    )


def _parse_is_data_real(
    meta: dict[str, str],
    source: str,
) -> bool | None:
    if "ISDATA_REAL" not in meta:
        return None
    text = meta["ISDATA_REAL"].strip().split()[0]
    try:
        value = int(text)
    except ValueError:
        raise SNANAError(
            f"ISDATA_REAL is {meta['ISDATA_REAL']!r} in {source}, which is "
            "not an integer",
            function_name="_parse_is_data_real",
        ) from None
    # any value at or below zero is simulated data, so -1, which
    # create_covariance writes where the BBC header carried no key, unblinds
    # (read_ISDATA_REAL, wfit.c:1791)
    return value > 0


def _flatten_yaml_mapping(value) -> dict[str, object]:
    flattened: dict[str, object] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            flattened[str(key)] = item
            flattened.update(_flatten_yaml_mapping(item))
    elif isinstance(value, list):
        for item in value:
            flattened.update(_flatten_yaml_mapping(item))
    return flattened


def _load_simulation_truth(
    info_yml: str | None,
    omega_m: float | None = None,
    w0: float | None = None,
    wa: float | None = None,
) -> dict[str, float | str] | None:
    """Read SNANA ``INFO.YML`` cosmology, or use explicit truth values."""
    if info_yml is None:
        if all(value is None for value in (omega_m, w0, wa)):
            return None
        return {
            "omega_m": omega_m,
            "omega_lambda": 1.0 - omega_m if omega_m is not None else None,
            "w0": w0,
            "wa": wa,
            "mushift": None,
            "source": "explicit",
        }

    source = Path(info_yml)
    with open(source, "rt") as stream:
        values = _flatten_yaml_mapping(yaml.safe_load(stream) or {})
    required = {
        "MUSHIFT": "mushift",
        "OMEGA_MATTER": "omega_m",
        "OMEGA_LAMBDA": "omega_lambda",
        "w0_LAMBDA": "w0",
        "wa_LAMBDA": "wa",
    }
    missing = [key for key in required if key not in values]
    if missing:
        raise SNANAError(
            f"{source} is missing simulation cosmology keys {missing}",
            function_name="_load_simulation_truth",
        )
    lo, hi = parse_mushift(values["MUSHIFT"])
    if lo != hi:
        # SNANA's wfit reads the first number of lo:hi
        raise SNANAError(
            f"{source}: the simulation drew MUSHIFT from {lo}:{hi}; HDIBC "
            f"needs the one constant shift it adds to every distance",
            function_name="_load_simulation_truth",
        )
    parsed = {
        field_name: float(values[key])
        for key, field_name in required.items() if key != "MUSHIFT"
    }
    return {"source": str(source), **parsed, "mushift": lo}


def _precision_from_covariance(covariance: np.ndarray) -> np.ndarray:
    """Invert a covariance through its Cholesky factor.

    Also the positive-definiteness check: cholesky raises on a covariance that
    is not usable.
    """
    factor = np.linalg.cholesky(covariance)
    return cho_solve((factor, True), np.eye(covariance.shape[0]))


def _validate_covariance(matrix: np.ndarray, path: str) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise SNANAError(
            f"covariance must be square: {path}",
            function_name="_validate_covariance",
        )
    if not np.all(np.isfinite(matrix)):
        raise SNANAError(
            f"covariance contains non-finite values: {path}",
            function_name="_validate_covariance",
        )
    if not np.allclose(matrix, matrix.T, rtol=1.0e-10, atol=1.0e-12):
        raise SNANAError(
            f"covariance is not symmetric: {path}",
            function_name="_validate_covariance",
        )


def fit_indices(config, z, mu, muerr, *, nfit=None, kind="diagonal"):
    """Validate required measurements and select rows before covariance changes."""
    arrays = [np.asarray(value) for value in (z, mu, muerr)]
    if any(value.ndim != 1 or value.shape != arrays[0].shape for value in arrays):
        raise SNANAError("Hubble-diagram columns must be aligned one-dimensional arrays")
    z, mu, muerr = arrays
    finite = np.isfinite(z) & np.isfinite(mu) & np.isfinite(muerr)
    if not finite.all():
        raise SNANAError("Hubble diagram contains non-finite required values at rows "
                         f"{np.flatnonzero(~finite)[:10].tolist()}")
    keep = (z >= config.zmin) & (z <= config.zmax)
    if nfit is not None:
        keep &= np.asarray(nfit) > 1
    indices = np.flatnonzero(keep)
    if not indices.size:
        raise SNANAError("no Hubble-diagram rows remain after WFIT cuts")
    if kind == "total_inverse" and indices.size != z.size:
        raise SNANAError("cannot apply row-removing cuts to total inverse covariance")
    return indices


def selected_covariance(matrix, indices, size):
    """Validate the complete supplied matrix before selecting its fitted rows."""
    if isinstance(matrix, dict):
        matrix = validate_factorized_covariance(matrix)
        if matrix["diag"].size != size:
            raise SNANAError("factorized covariance size does not match HD rows")
        if np.array_equal(indices, np.arange(size)):
            return matrix
        return {**matrix, "diag": matrix["diag"][indices], "U": matrix["U"][indices]}
    matrix = np.asarray(matrix)
    _validate_covariance(matrix, "Hubble diagram")
    if matrix.shape != (size, size):
        raise SNANAError(f"covariance shape {matrix.shape} does not match {size} HD rows")
    if np.array_equal(indices, np.arange(size)):
        return matrix
    return matrix[np.ix_(indices, indices)]


def validate_prepared_precision(precision, kind, size):
    """Check the prepared weights being reused, without reconstructing them."""
    if kind == "factorized":
        if not isinstance(precision, dict) or not {
                "invstd", "design", "mapping", "weighted_ones", "sum_weights"} <= precision.keys():
            raise SNANAError("factorized covariance requires prepared precision weights")
        if (precision["invstd"].shape != (size,) or precision["design"].ndim != 2
                or precision["design"].shape[0] != size
                or precision["mapping"].shape != precision["design"].shape
                or precision["weighted_ones"].shape != (size,)
                or precision["sum_weights"].shape != ()):
            raise SNANAError("factorized precision shape does not match HD rows")
        if (not all(np.isfinite(v).all() for v in precision.values())
                or (np.asarray(precision["invstd"]) <= 0).any()
                or float(precision["sum_weights"]) <= 0):
            raise SNANAError("factorized precision must be finite with positive weights")
        return
    precision = np.asarray(precision)
    if kind == "diagonal":
        if precision.shape != (size,):
            raise SNANAError("precision shape does not match Hubble-diagram rows")
        diagonal = precision
    elif kind in ("total", "systematics", "total_inverse"):
        _validate_covariance(precision, "Hubble-diagram precision")
        if precision.shape != (size, size):
            raise SNANAError("precision shape does not match Hubble-diagram rows")
        diagonal = np.diag(precision)
    else:
        raise SNANAError(f"unknown covariance kind {kind!r}")
    if not np.isfinite(diagonal).all() or (diagonal <= 0).any():
        raise SNANAError("Hubble-diagram precision must be finite with positive diagonal")
    if kind != "diagonal":
        try:
            np.linalg.cholesky(precision)
        except np.linalg.LinAlgError as exc:
            raise SNANAError("Hubble-diagram precision must be positive definite") from exc


def conditioned_precision(kind, muerr, matrix, scatter, *, previous_scatter=0., precision=None):
    """Replace extra scatter and return precision and total covariance.

    Dense matrices already include statistical errors. Inverse matrices cannot
    be modified; diagonal inputs can carry an already-prepared precision.
    """
    if kind == "total_inverse":
        if scatter != 0 or previous_scatter != 0:
            raise SNANAError("snrms cannot modify an already-inverted covariance")
        _validate_covariance(matrix, "Hubble diagram")
        np.linalg.cholesky(matrix)
        return matrix, None
    delta = scatter**2 - previous_scatter**2
    if kind == "factorized":
        if not isinstance(matrix, dict):
            raise SNANAError("factorized Hubble diagram requires diag and U")
        covariance = validate_factorized_covariance({
            **matrix, "diag": np.asarray(matrix["diag"], dtype=np.float64) + delta})
        return prepare_precision(covariance["diag"], covariance["U"]), covariance
    if kind == "diagonal":
        variance = np.asarray(muerr)**2 if precision is None else 1. / np.asarray(precision)
        variance = variance + delta
        if not np.isfinite(variance).all() or (variance <= 0).any():
            raise SNANAError("Hubble-diagram variance must be finite and positive")
        return 1. / variance, None
    if kind not in ("systematics", "total"):
        raise SNANAError(f"unknown covariance kind {kind!r}")
    _validate_covariance(matrix, "Hubble diagram")
    covariance = np.asarray(matrix) + np.eye(len(muerr)) * delta
    return _precision_from_covariance(covariance), covariance


def _idealize_hubble_diagram(
    config: WFITConfig,
    truth: dict[str, float | str] | None,
    redshift: np.ndarray,
    input_mu: np.ndarray,
    input_muerr: np.ndarray,
    realisation_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate idealised data at the simulation truth."""
    if truth is None or any(
        truth.get(name) is None for name in ("omega_m", "w0", "wa")
    ):
        raise SNANAError(
            "idealised data require complete simulation truth",
            function_name="_idealize_hubble_diagram",
        )
    if config.muerr_ideal_use_input:
        muerr = np.asarray(input_muerr)
    else:
        muerr = np.polynomial.polynomial.polyval(
            redshift,
            np.asarray(config.muerr_ideal_polynomial),
        )
    if not np.all(np.isfinite(muerr)) or np.any(muerr <= 0.0):
        raise SNANAError(
            "idealised uncertainty is non-positive or non-finite at one or "
            "more retained redshifts",
            function_name="_idealize_hubble_diagram",
        )

    mu_model = distance_modulus(
        redshift,
        redshift,
        truth["omega_m"],
        1.0 - truth["omega_m"],
        LIGHT_KM / config.h0,
        truth["w0"],
        truth["wa"],
        radiation=config.radiation,
    )
    # per-dataset seed from the input mu values, as SNANA's ISEED_UNIQUE
    # (wfit.c:2575); the exact value cannot match, only the determinism
    seed = int.from_bytes(
        hashlib.sha256(np.asarray(input_mu).tobytes()).digest()[:4], "big")
    key = jax.random.fold_in(jax.random.PRNGKey(seed), realisation_index)
    noise = jax.random.normal(
        key,
        shape=mu_model.shape,
        dtype=mu_model.dtype,
    )
    mu = np.asarray(mu_model + jnp.asarray(muerr) * noise)
    return mu, np.asarray(muerr)


def _load_hubble_diagram_realisation(
    config: WFITConfig,
    hubble_diagram: str,
    covariance: str | None,
    truth: dict[str, float | str] | None,
    realisation_index: int = 0,
) -> HubbleDiagram:
    """Load and validate one conventional WFIT Hubble diagram."""
    table = read_fitres(hubble_diagram)
    columns = table.df.columns
    cid_col = _find_column(columns, _CID_ALIASES, required=False)
    z_col = _find_column(columns, _Z_ALIASES)
    mu_col = _find_column(columns, _MU_ALIASES)
    muerr_aliases = (
        (config.varname_muerr,) if config.varname_muerr
        else _MUERR_ALIASES
    )
    muerr_col = _find_column(columns, muerr_aliases)
    zerr_col = _find_column(columns, _ZERR_ALIASES, required=False)

    z = table.df[z_col].to_numpy(dtype=np.float64)
    mu = table.df[mu_col].to_numpy(dtype=np.float64)
    if "MUREF" in columns:
        mu = mu + table.df["MUREF"].to_numpy(dtype=np.float64)
    muerr = table.df[muerr_col].to_numpy(dtype=np.float64)
    if config.muerr_force is not None:
        muerr = np.full_like(muerr, config.muerr_force)
    nfit = (
        table.df["NFIT"].to_numpy(dtype=np.int64)
        if "NFIT" in columns else np.full(z.shape, 2, dtype=np.int64)
    )
    cid = (
        table.df[cid_col].astype(str).to_numpy()
        if cid_col else np.arange(1, z.size + 1).astype(str)
    )
    zerr = (
        table.df[zerr_col].to_numpy(dtype=np.float64)
        if zerr_col else None
    )

    raw_size = z.size
    kind = config.covariance_kind if covariance else "diagonal"
    indices = fit_indices(config, z, mu, muerr, nfit=nfit, kind=kind)
    covariance_matrix: np.ndarray | None = None
    if covariance:
        matrix = (read_factorized_covariance(covariance, size=raw_size)
                  if kind == "factorized" else _load_matrix(covariance))
        matrix = selected_covariance(matrix, indices, raw_size)
    else:
        matrix = None

    kept_mu = mu[indices]
    kept_err = muerr[indices]
    if config.muerr_ideal:
        kept_mu, kept_err = _idealize_hubble_diagram(
            config,
            truth,
            z[indices],
            kept_mu,
            kept_err,
            realisation_index,
        )
    if kind == "systematics":
        matrix = matrix + np.diag(kept_err**2)
    precision, covariance_matrix = conditioned_precision(
        kind, kept_err, matrix, config.snrms)

    return HubbleDiagram(
        cid=cid[indices],
        redshift=jnp.asarray(z[indices]),
        mu=jnp.asarray(kept_mu),
        muerr=jnp.asarray(kept_err),
        precision=precision if kind == "factorized" else jnp.asarray(precision),
        covariance_kind=kind,
        snrms=config.snrms,
        covariance=(
            covariance_matrix if kind == "factorized" else jnp.asarray(covariance_matrix)
            if covariance_matrix is not None else None
        ),
        raw_indices=indices,
        zerr=zerr[indices] if zerr is not None else None,
        meta=table.meta,
        source=hubble_diagram,
        truth=truth,
    )


def _load_hdibc(config: WFITConfig) -> HubbleDiagram:
    """Prepare SNANA's two-HD bias-cosmology interpolation method."""
    paths = tuple(config.hubble_diagram)
    covariance_config = config.covariance
    if covariance_config is None:
        covariance_paths: tuple[str | None, ...] = (None, None)
    elif isinstance(covariance_config, str):
        covariance_paths = (covariance_config, None)
    else:
        supplied = tuple(covariance_config)
        if len(supplied) not in {1, 2}:
            raise SNANAError(
                "HDIBC accepts one or two covariance files",
                function_name="_load_hdibc",
            )
        covariance_paths = (
            (supplied[0], None) if len(supplied) == 1 else supplied
        )

    diagrams = []
    bias_truths = []
    for path, covariance_path, info_path in zip(
        paths, covariance_paths, config.hdibc_info,
    ):
        diagrams.append(_load_hubble_diagram_realisation(
            config,
            path,
            covariance_path,
            None,
        ))
        bias_truths.append(_load_simulation_truth(info_path))

    first, second = diagrams
    first_cids = [str(cid) for cid in first.cid]
    if len(set(first_cids)) != len(first_cids):
        raise SNANAError(
            "HDIBC first Hubble diagram contains duplicate CIDs",
            function_name="_load_hdibc",
        )
    second_lookup = {str(cid): index for index, cid in enumerate(second.cid)}
    if len(second_lookup) != len(second.cid):
        raise SNANAError(
            "HDIBC second Hubble diagram contains duplicate CIDs",
            function_name="_load_hdibc",
        )
    index0 = np.asarray([
        index for index, cid in enumerate(first_cids)
        if cid in second_lookup
    ])
    index1 = np.asarray([
        second_lookup[first_cids[index]] for index in index0
    ])
    if index0.size == 0:
        raise SNANAError(
            "HDIBC Hubble diagrams have no common CIDs",
            function_name="_load_hdibc",
        )

    z0 = np.asarray(first.redshift)[index0]
    z1 = np.asarray(second.redshift)[index1]
    mu0 = np.asarray(first.mu)[index0]
    mu1 = np.asarray(second.mu)[index1]
    bias_mu0 = np.asarray(distance_modulus(
        z0,
        z0,
        bias_truths[0]["omega_m"],
        bias_truths[0]["omega_lambda"],
        LIGHT_KM / config.h0,
        bias_truths[0]["w0"],
        bias_truths[0]["wa"],
        radiation=config.radiation,
    ))
    bias_mu1 = np.asarray(distance_modulus(
        z1,
        z1,
        bias_truths[1]["omega_m"],
        bias_truths[1]["omega_lambda"],
        LIGHT_KM / config.h0,
        bias_truths[1]["w0"],
        bias_truths[1]["wa"],
        radiation=config.radiation,
    ))
    if index0.size <= _HDIBC_SYNC_MAX_ROWS:
        synchronized_bias_mu1 = np.asarray(distance_modulus(
            z0,
            z0,
            bias_truths[1]["omega_m"],
            bias_truths[1]["omega_lambda"],
            LIGHT_KM / config.h0,
            bias_truths[1]["w0"],
            bias_truths[1]["wa"],
            radiation=config.radiation,
        ))
        mu1 = mu1 + synchronized_bias_mu1 - bias_mu1

    covariance_values = []
    precision_values = []
    for diagram, indices, covariance_path in zip(
        diagrams, (index0, index1), covariance_paths,
    ):
        if covariance_path is None:
            continue
        if config.covariance_kind == "total_inverse":
            if indices.size != diagram.cid.size:
                raise SNANAError(
                    "cannot remove rows from total inverse covariance when "
                    "matching HDIBC event IDs",
                    function_name="_load_hdibc",
                )
            precision_values.append(
                np.asarray(diagram.precision)[
                    np.ix_(indices, indices)
                ]
            )
        else:
            covariance_values.append(
                np.asarray(diagram.covariance)[
                    np.ix_(indices, indices)
                ]
            )
    if covariance_values:
        covariance = np.mean(covariance_values, axis=0)
        precision = np.linalg.inv(covariance)
        covariance_kind = config.covariance_kind
    elif precision_values:
        covariance = None
        precision = np.mean(precision_values, axis=0)
        covariance_kind = "total_inverse"
    else:
        covariance = None
        precision = np.asarray(first.precision)[index0]
        covariance_kind = "diagonal"

    simulation_truth = _load_simulation_truth(
        config.truth_info, config.om_sim, config.w0_sim, config.wa_sim,
    )
    return HubbleDiagram(
        cid=first.cid[index0],
        redshift=jnp.asarray(z0),
        mu=jnp.asarray(mu0),
        muerr=jnp.asarray(np.asarray(first.muerr)[index0]),
        precision=jnp.asarray(precision),
        covariance_kind=covariance_kind,
        snrms=config.snrms,
        covariance=(
            jnp.asarray(covariance) if covariance is not None else None
        ),
        raw_indices=first.raw_indices[index0],
        zerr=(
            np.asarray(first.zerr)[index0]
            if first.zerr is not None else None
        ),
        meta=first.meta,
        source=f"{paths[0]},{paths[1]}",
        truth=simulation_truth,
        mu_second=jnp.asarray(mu1),
        z_second=jnp.asarray(z1),
        bias0=jnp.asarray(bias_mu0 + bias_truths[0]["mushift"]),
        bias1=jnp.asarray(bias_mu1 + bias_truths[1]["mushift"]),
    )


def hubble_diagram_from_covariance(diagram, covtot_inv, isdata_real: int,
                                   source: str, *,
                                   truth: dict[str, float | str] | None = None) -> HubbleDiagram:
    """Combine an HD with one COVOPT's dense inverse or factor arrays.

    Factor arrays stay compact through preparation and likelihood evaluation.
    """
    cid_col = _find_column(diagram.columns, _CID_ALIASES)
    kind = covtot_inv.get("kind", "factorized") if isinstance(covtot_inv, dict) else "total_inverse"
    if kind not in ("factorized", "total_inverse"):
        raise SNANAError(f"expected covariance factors or a total inverse, got {kind!r}")
    compact = kind == "factorized"
    if compact:
        covtot_inv = validate_factorized_covariance(covtot_inv)
    elif isinstance(covtot_inv, dict):
        covtot_inv = covtot_inv["matrix"]
    return HubbleDiagram(
        cid=np.asarray(diagram[cid_col], dtype=str),
        redshift=jnp.asarray(np.asarray(diagram["zHD"], dtype=float)),
        mu=jnp.asarray(np.asarray(diagram["MU"], dtype=float)),
        muerr=jnp.asarray(np.asarray(diagram["MUERR"], dtype=float)),
        precision=(prepare_precision(covtot_inv["diag"], covtot_inv["U"]) if compact
                   else jnp.asarray(covtot_inv)),
        covariance_kind=kind,
        covariance=covtot_inv if compact else None,
        meta={"ISDATA_REAL": str(isdata_real)},
        source=source,
        truth=truth,
    )


def load_simulation_truths(
    config: WFITConfig, count: int,
) -> tuple[dict[str, float | str] | None, ...]:
    """Resolve explicit or file-based simulation truth for each diagram."""
    if config.truth_info is None:
        truth_paths = (None,) * count
    elif isinstance(config.truth_info, str):
        truth_paths = (config.truth_info,) * count
    else:
        truth_paths = tuple(config.truth_info)
        if len(truth_paths) != count:
            raise SNANAError(
                "the number of truth INFO.YML files must match the number of "
                "Hubble-diagram realisations",
                function_name="load_simulation_truths",
            )
    return tuple(
        _load_simulation_truth(path, config.om_sim, config.w0_sim, config.wa_sim)
        for path in truth_paths
    )


def load_hubble_diagrams(config: WFITConfig) -> tuple[HubbleDiagram, ...]:
    """Load every requested realisation, one HubbleDiagram each."""
    config.validate()
    if not config.hubble_diagram:
        raise SNANAError("hubble_diagram is required",
                         function_name="load_hubble_diagrams")
    if config.hdibc:
        return (_load_hdibc(config),)
    paths = (
        (config.hubble_diagram,)
        if isinstance(config.hubble_diagram, str)
        else tuple(config.hubble_diagram)
    )
    if config.covariance is None:
        covariance_paths = (None,) * len(paths)
    elif isinstance(config.covariance, str):
        covariance_paths = (config.covariance,) * len(paths)
    else:
        covariance_paths = tuple(config.covariance)
        if len(covariance_paths) != len(paths):
            raise SNANAError(
                "the number of covariance files must match the number of "
                "Hubble-diagram realisations",
                function_name="load_hubble_diagrams",
            )
    truths = load_simulation_truths(config, len(paths))
    return tuple(
        _load_hubble_diagram_realisation(
            config,
            path,
            covariance_path,
            truth,
            index,
        )
        for index, (path, covariance_path, truth) in enumerate(
            zip(paths, covariance_paths, truths)
        )
    )
