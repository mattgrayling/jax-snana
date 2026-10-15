"""Output blinding corresponding to SNANA WFIT's blind modes."""

import math
from typing import Sequence

from jax_snlc_core.utils import SNANAError

from .config import WFITConfig
from .data import HubbleDiagram


_RAND_MAX = 2_147_483_647


def blinding_statuses(
    config: WFITConfig,
    diagrams: Sequence[HubbleDiagram],
) -> tuple[bool, ...]:
    """Resolve explicit or ``ISDATA_REAL``-driven blinding per realisation."""
    mode = config.blind_mode
    if mode == "none":
        return (False,) * len(diagrams)
    if mode == "blind":
        return (True,) * len(diagrams)

    statuses: list[bool] = []
    for index, is_data_real in enumerate(d.is_data_real for d in diagrams):
        if is_data_real is None:
            raise SNANAError(
                "automatic blinding requires ISDATA_REAL in every HD header; "
                f"it is missing for realisation {index}",
                function_name="blinding_statuses",
            )
        statuses.append(is_data_real)
    return tuple(statuses)


def _parameter_names(config: WFITConfig) -> list[str]:
    names: list[str] = []
    if config.model in {"wcdm", "w0wa"}:
        names.append("w0")
    if config.model == "w0wa":
        names.append("wa")
    names.append("omega_m")
    return names


def parameter_random_integers(config: WFITConfig) -> dict[str, int]:
    """Reproduce the integers drawn by SNANA WFIT on macOS.

    SNANA ``wfit.c::get_final_values`` calls ``srand(blind_seed)`` followed by
    ``floor(1e6 * rand() / RAND_MAX)`` in free-parameter order. Darwin libc's
    ``rand`` is the Park-Miller sequence implemented here without mutating
    libc's process-global random state.
    """
    state = int(config.blind_seed) & 0xFFFFFFFF
    values: dict[str, int] = {}
    for name in _parameter_names(config):
        if state == 0:
            state = 123_459_876
        high = state // 127_773
        low = state % 127_773
        state = 16_807 * low - 2_836 * high
        if state < 0:
            state += _RAND_MAX
        values[name] = (1_000_000 * state) // _RAND_MAX
    return values


def parameter_offsets(config: WFITConfig) -> dict[str, float]:
    """Return SNANA's sine offsets for reported cosmology estimates."""
    integers = parameter_random_integers(config)
    return {
        name: math.sin(value)
        for name, value in integers.items()
    }


def apply_blinding(values, config: WFITConfig, blinded: bool):
    """Offset every value when blinded. Values may be scalars or arrays.

    Indexes directly: a missing offset would otherwise write an unblinded number
    into a file labelled blinded.
    """
    if not blinded:
        return dict(values)
    offsets = parameter_offsets(config)
    return {name: value + offsets[name] for name, value in values.items()}
