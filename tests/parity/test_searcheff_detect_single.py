"""SNANA-vs-JAX cross-validation for ``APPLY_DETECT_SINGLE``.

When set, SEARCHEFF curves are interpreted per-exposure (per-visit eff =
1 - (1 - EFF_single)^NEXPOSE). Test exercises a SIMLIB with mixed NEXPOSE
and asserts:

  (a) with the feature off, SNANA and JAX agree
  (b) with it on, they still agree
  (c) the on-rate is detectably different from off — sanity-checks that
      the feature is being exercised (i.e. NEXPOSE>1 obs exist).
"""

from pathlib import Path

import numpy as np

from jax_snlc_core.snana_files import open_text
from jax_snlc_sim import run as sim_run


def _parse_dump_column(path: Path, column: str) -> np.ndarray:
    varnames = None
    rows = []
    with open_text(path) as f:
        for line in f:
            if line.startswith("VARNAMES:"):
                varnames = line.split(":", 1)[1].split()
            elif line.startswith("SN:"):
                rows.append(line.split()[1:])
    if varnames is None or column not in varnames:
        raise ValueError(f"Column {column!r} not in {path}")
    idx = varnames.index(column)
    return np.array([float(r[idx]) for r in rows])


def _run_jax_pipeline_pass_rate(input_file: Path, n_sne: int) -> float:
    config = dict(
        n_sne=n_sne, random_seed=122459,
    )
    sim = sim_run.prepare(str(input_file), **config)
    results, sel, _ = sim_run.run_simulation(sim)
    mask = np.asarray(sel["sim_searcheff_mask"]).astype(int)
    return int(((mask & 1) > 0).sum()) / len(mask)


def test_detect_single_agreement(snana_fixture):
    """Pipeline-pass rate matches SNANA in both on/off; feature has effect."""
    fx_off = snana_fixture("detect_single_off")
    fx_on = snana_fixture("detect_single_on")

    # OFF baseline
    off_mask = _parse_dump_column(fx_off.dump, "SIM_SEARCHEFF_MASK").astype(int)
    snana_off_rate = int(((off_mask & 1) > 0).sum()) / len(off_mask)
    jax_off_rate = _run_jax_pipeline_pass_rate(fx_off.input_file, len(off_mask))

    # ON: APPLY_DETECT_SINGLE active
    on_mask = _parse_dump_column(fx_on.dump, "SIM_SEARCHEFF_MASK").astype(int)
    snana_on_rate = int(((on_mask & 1) > 0).sum()) / len(on_mask)
    jax_on_rate = _run_jax_pipeline_pass_rate(fx_on.input_file, len(on_mask))

    assert abs(snana_off_rate - jax_off_rate) < 0.03, (
        f"OFF rate diverges: SNANA={snana_off_rate:.3f}, JAX={jax_off_rate:.3f}"
    )
    assert abs(snana_on_rate - jax_on_rate) < 0.03, (
        f"ON rate diverges: SNANA={snana_on_rate:.3f}, JAX={jax_on_rate:.3f}"
    )

    # Sanity: feature actually changed something (otherwise the SIMLIB has
    # NEXPOSE=1 everywhere or the curve is flat).
    feature_delta = abs(snana_on_rate - snana_off_rate)
    assert feature_delta > 0.01, (
        f"ON vs OFF differ by only {feature_delta:.3f} in SNANA — "
        "feature not being exercised (check NEXPOSE>1 in SIMLIB)"
    )
