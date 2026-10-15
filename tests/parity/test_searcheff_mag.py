"""SNANA-vs-JAX cross-validation for MAG-based SEARCHEFF curves.

Drives JAX with a synthetic SEARCHEFF_PIPELINE file that uses ``MAG:`` rows
(matching SNANA's `sntools_trigger.c:412-423` parser) instead of ``SNR:``
rows. Asserts the population-level pipeline-pass fraction agrees with the
bundled SNANA reference within 5%.
"""

from pathlib import Path

import numpy as np


from jax_snlc_sim import run as sim_run  # noqa: E402


def _parse_dump_column(path: Path, column: str) -> np.ndarray:
    """Return the named SIMGEN_DUMP column as a numpy array (floats)."""
    varnames = None
    rows = []
    with path.open() as f:
        for line in f:
            if line.startswith("VARNAMES:"):
                varnames = line.split(":", 1)[1].split()
            elif line.startswith("SN:"):
                rows.append(line.split()[1:])
    if varnames is None or column not in varnames:
        raise ValueError(f"Column {column!r} not in {path}")
    idx = varnames.index(column)
    return np.array([float(r[idx]) for r in rows])


def test_mag_pipeline_pass_rate(snana_fixture, tmp_path):
    """Pipeline-pass fraction agrees between bundled SNANA and JAX within 5%."""
    fx = snana_fixture("searcheff_mag")
    snana_mask = _parse_dump_column(fx.dump, "SIM_SEARCHEFF_MASK").astype(int)
    n_total = len(snana_mask)
    snana_pipe = int(((snana_mask & 1) > 0).sum())
    snana_rate = snana_pipe / n_total

    config = dict(
        n_sne=n_total, random_seed=122459,
    )
    sim = sim_run.prepare(str(fx.input_file), **config)
    results, sel, _ = sim_run.run_simulation(sim)

    jax_mask = np.asarray(sel["sim_searcheff_mask"]).astype(int)
    jax_pipe = int(((jax_mask & 1) > 0).sum())
    jax_rate = jax_pipe / len(jax_mask)

    diff = abs(snana_rate - jax_rate)
    assert diff < 0.03, (
        f"Pipeline-pass rate diverges by {diff:.3f} > 0.03: "
        f"SNANA={snana_pipe}/{n_total} ({snana_rate:.1%}), "
        f"JAX={jax_pipe}/{len(jax_mask)} ({jax_rate:.1%})"
    )
    # Reject ALL_PASS / ALL_FAIL degeneracies of the MAG curve.
    assert 0.05 < snana_rate < 0.95
    assert 0.05 < jax_rate < 0.95
