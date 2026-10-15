"""Phase-2 SEARCHEFF SPEC_SCALE replay parity test.

JAX runs in replay mode against the bundled SNANA reference DUMP/HEAD,
then byte-compares its SIMGEN_DUMPALL output column-by-column. The
fixture exercises ``SEARCHEFF_SPEC_FILE`` + ``SEARCHEFF_SPEC_SCALE`` —
the deferred phase-2-extended scenario from the original Phase-7 plan.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from jax_snlc_core.io.snana_dump_reader import load_snana_dump
from replay_snana_dump import build_override_parameters
from jax_snlc_sim import run as sim_run

from parity._dumpall_helpers import compare_dumps, parse_dump


def _run_jax_replay(input_file: Path, gv: str, snana_dump: Path,
                     snana_head: Path, work_root: Path):
    """Run JAX in replay mode and return ``(varnames, cols)`` of its DUMP."""
    dump_data = load_snana_dump(str(snana_dump), dedup_cid=True)
    if dump_data.size == 0:
        return None, {}

    with fits.open(snana_head) as hdul:
        head_table = hdul[1].data
        head_cols = {name: head_table[name] for name in head_table.names}
        peak_map = {
            int(s): float(p)
            for s, p in zip(head_table["SNID"], head_table["SIM_PEAKMJD"])
        }

    config = dict(
        n_sne=dump_data.size,
    )
    sim = sim_run.prepare(str(input_file), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    override = build_override_parameters(
        dump_data.columns, simulation, resources, head_columns=head_cols
    )
    peaks = [
        peak_map.get(int(cid), float(override["peakmjd"][i]))
        for i, cid in enumerate(override["cid"])
    ]
    override["peakmjd"] = np.asarray(peaks, dtype=np.float32)

    cwd = os.getcwd()
    try:
        os.chdir(work_root)
        results, selection, spectra_cache = sim_run.run_simulation(sim, override_parameters=override)
        sim_run.save_snana_outputs(sim, results, gv, selection, spectra_cache=spectra_cache)
    finally:
        os.chdir(cwd)

    jax_dump = work_root / gv / f"{gv}.DUMP"
    if not jax_dump.exists():
        return None, {}
    return parse_dump(jax_dump)


def test_searcheff_spec_scale_replay_dump_parity(snana_fixture, tmp_path):
    """JAX in replay mode reproduces SNANA's SIMGEN_DUMPALL when
    ``SEARCHEFF_SPEC_FILE`` + ``SEARCHEFF_SPEC_SCALE`` are active."""
    fx = snana_fixture("searcheff_spec_scale_replay")
    snana_dump = fx.dump
    snana_head = fx.head_fits
    assert snana_dump.exists() and snana_head.exists(), (
        f"missing SNANA reference outputs in {fx.root}"
    )

    snana_varnames, snana_cols = parse_dump(snana_dump)
    assert snana_varnames is not None

    jax_varnames, jax_cols = _run_jax_replay(
        fx.input_file, "JAX_TEST_SPEC_SCALE_REPLAY",
        snana_dump, snana_head, tmp_path,
    )
    assert jax_varnames is not None, "JAX replay did not produce a DUMP file"

    compare_dumps(snana_varnames, snana_cols, jax_varnames, jax_cols)
