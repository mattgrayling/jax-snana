"""Mixed SDSS SPEC + zHOST: SNANA-vs-JAX aggregate-rate parity per mask bit.

Bundled fixture has both SDSS SPEC and SDSS zHOST attached, with
APPLY_SEARCHEFF_OPT=1. Pipeline-passing SNe split into SPECID by the SDSS
SPEC curve; the zHOST file has no VARNAMES line, so it holds no map and
every SPECID failure gets a host redshift.

Asserts JAX reproduces SNANA's per-bit rate (pipeline, SPEC | pipe,
zHOST | pipe, (SPEC|zHOST) | pipe) within statistical tolerance.
"""

from pathlib import Path

import numpy as np
import pytest


from jax_snlc_core.io.fits_path import open_text
from jax_snlc_sim import run as sim_run  # noqa: E402


def _parse_dump_mask(path: Path) -> np.ndarray:
    varnames: list[str] | None = None
    rows: list[list[str]] = []
    with open_text(path) as f:
        for line in f:
            if line.startswith("VARNAMES:"):
                varnames = line.split(":", 1)[1].split()
            elif line.startswith("SN:"):
                rows.append(line.split()[1:])
    if varnames is None or "SIM_SEARCHEFF_MASK" not in varnames:
        raise ValueError(f"SIM_SEARCHEFF_MASK column missing from {path}")
    idx = varnames.index("SIM_SEARCHEFF_MASK")
    return np.array([int(float(r[idx])) for r in rows])


def _bit_rates(mask: np.ndarray) -> dict[str, float]:
    pipe = (mask & 1) > 0
    n_pipe = int(pipe.sum())
    if n_pipe == 0:
        return {"pipe": 0.0, "spec": 0.0, "zhost": 0.0, "spec_or_zhost": 0.0}
    spec = (mask & 2) > 0
    zhost = (mask & 4) > 0
    return {
        "pipe": n_pipe / len(mask),
        "spec": int((spec & pipe).sum()) / n_pipe,
        "zhost": int((zhost & pipe).sum()) / n_pipe,
        "spec_or_zhost": int(((spec | zhost) & pipe).sum()) / n_pipe,
    }


@pytest.fixture(scope="module")
def mixed_masks(snana_fixture):
    """Run the mixed SPEC + zHOST sim once and return both SNANA and JAX
    mask arrays for downstream tests."""
    fx = snana_fixture("searcheff_mixed_spec_zhost")
    snana_mask = _parse_dump_mask(fx.dump)
    config = dict(
        n_sne=len(snana_mask), random_seed=122459,
    )
    sim = sim_run.prepare(str(fx.input_file), **config)
    results, sel, _ = sim_run.run_simulation(sim)
    jax_mask = np.asarray(sel["sim_searcheff_mask"]).astype(int)
    return snana_mask, jax_mask


def test_searcheff_mixed_spec_zhost_rates(mixed_masks):
    """Per-bit pass rates agree between SNANA and JAX within 3% absolute."""
    snana_mask, jax_mask = mixed_masks
    s_rates = _bit_rates(snana_mask)
    j_rates = _bit_rates(jax_mask)

    for name in ("pipe", "spec", "zhost", "spec_or_zhost"):
        diff = abs(s_rates[name] - j_rates[name])
        assert diff < 0.03, (
            f"{name} rate diverges by {diff:.3f}: "
            f"SNANA={s_rates[name]:.3f}, JAX={j_rates[name]:.3f}"
        )

    # Sanity: rates aren't degenerate (not all-pass / all-fail) — guards
    # against a regression where SPEC or zHOST silently bypasses.
    assert 0.05 < s_rates["spec"] < 0.95
    assert 0.05 < s_rates["zhost"] < 0.95


def test_searcheff_mixed_zhost_only_aggregate_pass_rate(mixed_masks):
    """Bonus aggregate parity: zHOST-only conditional pass rate (no SPEC).

    Among pipeline-passing SNe NOT spec-confirmed, the JAX zHOST pass rate
    matches SNANA's within tolerance — exercises the zHOST branch in
    isolation from the SPEC branch.
    """
    snana_mask, j_mask = mixed_masks
    s_pipe_no_spec = ((snana_mask & 1) > 0) & ((snana_mask & 2) == 0)
    assert int(s_pipe_no_spec.sum()) >= 30, (
        "SNANA's bundled dump has too few SPEC-rejected pipeline-passing SNe"
    )
    s_zhost_only = ((snana_mask & 4) > 0) & s_pipe_no_spec
    s_rate = float(s_zhost_only.sum()) / float(s_pipe_no_spec.sum())

    j_pipe_no_spec = ((j_mask & 1) > 0) & ((j_mask & 2) == 0)
    assert int(j_pipe_no_spec.sum()) >= 30, (
        "jax produced too few SPEC-rejected pipeline-passing SNe"
    )
    j_zhost_only = ((j_mask & 4) > 0) & j_pipe_no_spec
    j_rate = float(j_zhost_only.sum()) / float(j_pipe_no_spec.sum())

    assert abs(s_rate - j_rate) < 0.04, (
        f"zHOST conditional pass rate diverges: SNANA={s_rate:.3f}, "
        f"JAX={j_rate:.3f}"
    )
