"""NGEN_LC: generate fixed-size batches until N events pass trigger+cuts."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from jax_snlc_core.utils.error_handling import SNANAError
from jax_snlc_sim import run as sim_run
from jax_snlc_sim.builder import resolve_counts
from jax_snlc_sim.config_parser import parse_snana_input_file


_KCOR = "$JAX_TEST_FIXTURE_ROOT/kcor/JAX_TEST_LSST.fits"
_SIMLIB = "$JAX_TEST_FIXTURE_ROOT/simlib/JAX_TEST_LSST.SIMLIB"
_MODEL = "$JAX_TEST_FIXTURE_ROOT/models/SALT3.K21"


def _write_input(out: Path, *, extra: str = "") -> None:
    out.write_text(f"""\
GENVERSION:  JAX_TEST_NGEN_LC
FORMAT_MASK:  32
GENTYPE:  1
SNTYPES_Ia:  1  101
GENMODEL:  {_MODEL}
GENPEAK_SALT2x1:    0.0
GENRANGE_SALT2x1:  -2.0  2.0
GENSIGMA_SALT2x1:   1.0  1.0
GENPEAK_SALT2c:     0.0
GENRANGE_SALT2c:   -0.2  0.3
GENSIGMA_SALT2c:    0.05  0.05
GENPEAK_SALT2ALPHA: 0.14
GENPEAK_SALT2BETA:  3.1
OMEGA_MATTER:  0.315
OMEGA_LAMBDA:  0.685
w0_LAMBDA:  -1.0
wa_LAMBDA:  0
GENFILTERS:  ugrizY
KCOR_FILE:  {_KCOR}
SIMLIB_FILE:  {_SIMLIB}
GENRANGE_REDSHIFT:  0.05  0.40
GENRANGE_PEAKMJD:  60540  61460
GENSIGMA_REDSHIFT:  0.001
GENRANGE_TREST: -20  50
SOLID_ANGLE: 0.015
OPT_MWEBV: 0
DNDZ: POWERLAW 2.5E-5 1.5
RANSEED:  12345
{extra}
""")


def _run(inp: Path, *, batch_size: int):
    cfg = dict(
        n_sne=999, random_seed=12345,
        batch_size=batch_size)
    sim = sim_run.prepare(str(inp), **cfg)
    results, selection, _ = sim_run.run_simulation(sim)
    return results, selection


def test_ngen_lc_hits_target_over_multiple_batches(tmp_path):
    batch_size = 20
    target = 30
    inp = tmp_path / "input.input"
    # CUTWIN_REDSHIFT narrower than GENRANGE rejects the high-z half, so
    # pass rate < 1 and the loop must run several batches to reach target.
    _write_input(inp, extra=(
        f"NGEN_LC: {target}\nAPPLY_CUTWIN_OPT: 1\n"
        "CUTWIN_REDSHIFT: 0.05 0.20\n"))
    results, selection = _run(inp, batch_size=batch_size)

    n_pass = int(np.asarray(selection['passing_mask']).sum())
    n_generated = selection['n_passed'] + selection['n_failed']
    # attempts past the one that fills the quota are dropped, so the count
    # is not a whole number of batches
    assert n_pass == target
    assert n_generated // batch_size >= 2, "expected the loop to run >= 2 batches"
    assert n_pass < n_generated, "CUTWIN should reject some events"


def test_ngen_lc_budget_brake_fires(tmp_path):
    inp = tmp_path / "input.input"
    # Impossible cut: no SN has negative redshift, so nothing passes.
    _write_input(inp, extra=(
        "NGEN_LC: 10\nNGEN_LC_MAXGEN_FACTOR: 1\nAPPLY_CUTWIN_OPT: 1\n"
        "CUTWIN_REDSHIFT: -1.0 -0.5\n"))
    with pytest.raises(SNANAError, match="generation-budget guard"):
        _run(inp, batch_size=20)


def test_ngen_lc_scaled_by_ngen_scale(tmp_path):
    """SNANA scales the NGEN_LC accept-target by NGEN_SCALE (snlc_sim.c:7146)."""
    inp = tmp_path / "input.input"
    _write_input(inp, extra="NGEN_LC: 30\nNGEN_SCALE: 2.0\n")
    cfg = parse_snana_input_file(str(inp))
    n_simulated, ngen_lc, _, _ = resolve_counts(cfg, cfg.hz_map())
    assert ngen_lc == 60
    assert n_simulated == 60


def test_rerun_clears_the_previous_genversion(tmp_path):
    """A rerun replaces the previous output rather than writing over it."""
    inp = tmp_path / "input.input"
    _write_input(inp, extra="NGENTOT_LC: 10\nGENVERSION: RERUN\n")
    cfg = parse_snana_input_file(str(inp))
    out = tmp_path / "out"

    def run():
        sim_config = dict(
            n_sne=10, random_seed=12345,
            batch_size=10)
        cfg.numerical = replace(cfg.numerical, n_sne=sim_config["n_sne"],
                                batch_size=sim_config["batch_size"])
        sim = sim_run.prepare(cfg)
        results, sel, spectra_cache = sim_run.run_simulation(sim)
        sim_run.save_snana_outputs(sim, results, "RERUN", sel, spectra_cache=spectra_cache, output_dir=str(out))

    run()
    stale = out / "RERUN" / "STALE_FROM_LAST_RUN.TEXT"
    stale.write_text("from the previous run\n")
    run()
    assert not stale.exists(), "previous run's file survived"
    assert (out / "RERUN" / "RERUN.LIST").exists()


def test_command_line_override_reaches_the_writer(tmp_path):
    """`snlc_sim.py my.input GENVERSION <name>` writes to the overridden name.

    Driving the real CLI, since nothing else covers argv reaching the writer.
    """
    import subprocess
    import sys

    inp = tmp_path / "input.input"
    _write_input(inp, extra="NGENTOT_LC: 20\nGENVERSION: FROM_FILE\n")
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(repo_root / "snlc_sim.py"), str(inp),
         "GENVERSION", "FROM_OVERRIDE", "--batch-size", "10"],
        capture_output=True, text=True, cwd=tmp_path, timeout=900)

    assert result.returncode == 0, (
        f"snlc_sim.py exited {result.returncode}\n"
        f"stdout (tail):\n{result.stdout[-1500:]}\n"
        f"stderr (tail):\n{result.stderr[-800:]}")
    assert (tmp_path / "FROM_OVERRIDE").is_dir(), (
        f"expected the overridden GENVERSION directory; got "
        f"{sorted(p.name for p in tmp_path.iterdir())}")
    assert not (tmp_path / "FROM_FILE").exists()


def test_command_line_override_reaches_generation(tmp_path):
    """A command-line NGENTOT_LC changes how many SNe are generated, not just
    the parsed config — the config landing correctly is not evidence that the
    simulator ever read it."""
    inp = tmp_path / "input.input"
    _write_input(inp, extra="NGENTOT_LC: 40\n")

    def n_generated(overrides):
        cfg = parse_snana_input_file(str(inp), overrides)
        sim_config = dict(
            n_sne=None,
            random_seed=12345, batch_size=20)
        cfg.numerical = replace(cfg.numerical, n_sne=sim_config["n_sne"],
                                batch_size=sim_config["batch_size"])
        sim = sim_run.prepare(cfg)
        results, _, _ = sim_run.run_simulation(sim)
        return len(np.asarray(results['parameters']['sim_redshift_cmb']))

    assert n_generated([]) == 40
    assert n_generated(['NGENTOT_LC', '15']) == 15
