"""SNANA-vs-jax parity for NGEN_SEASON → NGENTOT_LC resolution.

Both codes apply the same `(int)(NGEN_SEASON × rate × NGEN_SCALE)` formula
(SNANA snlc_sim.c:9201-9221). This test runs SNANA and jax on the same
NGEN_SEASON input and asserts the resolved counts match exactly.

Live-only: requires `--live-snana` (or `RUN_LIVE_SNANA=1`) and
`snlc_sim.exe` on PATH.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from jax_snlc_sim.inputs import parse_snana_input_file


def _write_input(path: Path, *, kcor_path: str, simlib_path: str,
                 model_path: str, ngen_season: float = 1.0) -> None:
    """Small SALT3 input with NGEN_SEASON. Tiny by design so SNANA finishes fast."""
    path.write_text(f"""\
GENVERSION:  JAX_TEST_NGEN_SEASON_PARITY
NGEN_SEASON: {ngen_season}
FORMAT_MASK: 32
GENTYPE: 1
SNTYPES_Ia: 1 101
CIDRAN_MIN: 0
CIDRAN_MAX: 3000000
GENMODEL: {model_path}
GENPEAK_SALT2x1: 0.0
GENRANGE_SALT2x1: -2.0 2.0
GENSIGMA_SALT2x1: 1.0 1.0
GENPEAK_SALT2c: 0.0
GENRANGE_SALT2c: -0.2 0.3
GENSIGMA_SALT2c: 0.05 0.05
GENPEAK_SALT2ALPHA: 0.14
GENPEAK_SALT2BETA: 3.1
OMEGA_MATTER: 0.3150
OMEGA_LAMBDA: 0.6850
w0_LAMBDA: -1.0000
wa_LAMBDA: 0
GENFILTERS: ugrizY
KCOR_FILE: {kcor_path}
SIMLIB_FILE: {simlib_path}
SIMLIB_MINSEASON: 30
GENRANGE_REDSHIFT: 0.05 0.30
GENRANGE_MJD: 60500.0 61500.0
GENRANGE_PEAKMJD: 60540.0 61460.0
GENSIGMA_REDSHIFT: 0.001
GENRANGE_TREST: -20.0 50.0
GENRANGE_RA: -180.0 180.0
GENRANGE_DECL: -90.0 90.0
SOLID_ANGLE: 0.015
OPT_MWEBV: 0
DNDZ: POWERLAW 2.5E-5 1.5
RANSEED: 12345
""")


@pytest.mark.live_snana
def test_ngen_season_parity_against_live(tmp_path, lsst_kcor_path,
                                          lsst_simlib_path, data_root):
    """SNANA and jax produce identical NGENTOT_LC from the same NGEN_SEASON input."""
    if not shutil.which("snlc_sim.exe"):
        pytest.skip("snlc_sim.exe not on PATH")

    inp = tmp_path / "test.input"
    _write_input(
        inp,
        kcor_path=str(lsst_kcor_path),
        simlib_path=str(lsst_simlib_path),
        model_path=str(data_root / "models" / "SALT3.K21"),
    )

    snana_result = subprocess.run(
        ["snlc_sim.exe", str(inp)],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=120,
    )
    assert snana_result.returncode == 0, (
        f"SNANA failed (exit {snana_result.returncode}):\n"
        f"stderr tail:\n{snana_result.stderr[-800:]}\n"
        f"stdout tail:\n{snana_result.stdout[-800:]}"
    )

    # SNANA logs the resolution as: "NGEN_SEASON=X --> NGENTOT_LC=N"
    match = re.search(
        r"NGEN_SEASON=\S+\s+-->\s+NGENTOT_LC=(\d+)",
        snana_result.stdout,
    )
    assert match is not None, (
        "Could not find 'NGEN_SEASON=... --> NGENTOT_LC=N' in SNANA stdout. "
        f"stdout tail:\n{snana_result.stdout[-800:]}"
    )
    snana_n = int(match.group(1))

    from jax_snlc_sim.rate import resolve_counts
    jax_cfg = parse_snana_input_file(str(inp))
    jax_n = resolve_counts(jax_cfg, jax_cfg.hz_map())[0]
    assert jax_n == snana_n, (
        f"NGEN_SEASON resolution differs: SNANA={snana_n}, jax={jax_n}"
    )
