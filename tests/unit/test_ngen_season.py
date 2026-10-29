"""Build-level resolution of NGEN_SEASON / NGEN_SCALE → n_simulated.

Matches SNANA's `init_DNDZ_Rate` semantics (snlc_sim.c:9201-9221):
`NGENTOT_LC = (int)(NGEN_SEASON × rate × NGEN_SCALE)`.
"""

import pytest

from jax_snlc_core.logging import SNANAError
from jax_snlc_sim.rate import resolve_counts
from jax_snlc_sim.inputs import parse_snana_input_file


def _n_simulated(cfg):
    return resolve_counts(cfg, cfg.hz_map())[0]


def _write_minimal_input(path, *, ngentot_lc=None, ngen_season=None,
                          ngen_scale=None, kcor_path, simlib_path,
                          model_path):
    """Minimal SALT3 input with only the keys the parser strictly needs."""
    lines = [
        "GENVERSION: TEST",
        "FORMAT_MASK: 32",
        "GENTYPE: 1",
        "SNTYPES_Ia: 1 101",
        "CIDRAN_MIN: 0",
        "CIDRAN_MAX: 3000000",
        f"GENMODEL: {model_path}",
        "GENPEAK_SALT2x1: 0.0",
        "GENRANGE_SALT2x1: -2.0 2.0",
        "GENSIGMA_SALT2x1: 1.0 1.0",
        "GENPEAK_SALT2c: 0.0",
        "GENRANGE_SALT2c: -0.2 0.3",
        "GENSIGMA_SALT2c: 0.05 0.05",
        "GENPEAK_SALT2ALPHA: 0.14",
        "GENPEAK_SALT2BETA: 3.1",
        "OMEGA_MATTER: 0.3150",
        "OMEGA_LAMBDA: 0.6850",
        "w0_LAMBDA: -1.0000",
        "wa_LAMBDA: 0",
        "GENFILTERS: ugrizY",
        f"KCOR_FILE: {kcor_path}",
        f"SIMLIB_FILE: {simlib_path}",
        "SIMLIB_MINSEASON: 30",
        "GENRANGE_REDSHIFT: 0.05 0.30",
        "GENRANGE_MJD: 60500.0 61500.0",
        "GENRANGE_PEAKMJD: 60540.0 61460.0",
        "GENSIGMA_REDSHIFT: 0.001",
        "GENRANGE_TREST: -20.0 50.0",
        "GENRANGE_RA: -180.0 180.0",
        "GENRANGE_DECL: -90.0 90.0",
        "SOLID_ANGLE: 0.015",
        "OPT_MWEBV: 0",
        "DNDZ: POWERLAW 2.5E-5 1.5",
        "RANSEED: 12345",
    ]
    if ngentot_lc is not None:
        lines.insert(1, f"NGENTOT_LC: {ngentot_lc}")
    if ngen_season is not None:
        lines.insert(1, f"NGEN_SEASON: {ngen_season}")
    if ngen_scale is not None:
        lines.insert(1, f"NGEN_SCALE: {ngen_scale}")
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def minimal(tmp_path, lsst_kcor_path, lsst_simlib_path, data_root):
    """Write a minimal LSST-style SALT3 input and return the path."""
    def _factory(**kwargs):
        out = tmp_path / "test.input"
        _write_minimal_input(
            out,
            kcor_path=str(lsst_kcor_path),
            simlib_path=str(lsst_simlib_path),
            model_path=str(data_root / "models" / "SALT3.K21"),
            **kwargs,
        )
        return out
    return _factory


def test_ngen_season_scales_linearly(minimal):
    """NGEN_SEASON: 2.0 → twice the NGEN_SEASON: 1.0 count (within truncation)."""
    cfg1 = parse_snana_input_file(minimal(ngen_season=1.0))
    cfg2 = parse_snana_input_file(minimal(ngen_season=2.0))
    # int truncation can push the 2× result off by 1 vs 2×int.
    assert abs(_n_simulated(cfg2) - 2 * _n_simulated(cfg1)) <= 1


def test_ngen_scale_multiplies_ngentot_lc(minimal):
    """NGENTOT_LC: 1000 + NGEN_SCALE: 1.5 → n_simulated == 1500."""
    cfg = parse_snana_input_file(minimal(ngentot_lc=1000, ngen_scale=1.5))
    assert _n_simulated(cfg) == 1500


def test_ngen_scale_multiplies_ngen_season(minimal):
    """NGEN_SCALE multiplies the resolved NGEN_SEASON count."""
    cfg_unit = parse_snana_input_file(minimal(ngen_season=1.0))
    cfg_2x = parse_snana_input_file(minimal(ngen_season=1.0, ngen_scale=2.0))
    assert abs(_n_simulated(cfg_2x) - 2 * _n_simulated(cfg_unit)) <= 1


def test_mutual_exclusion_ngentot_and_ngen_season(minimal):
    """Both NGENTOT_LC and NGEN_SEASON set → SNANAError (matches SNANA)."""
    cfg = parse_snana_input_file(minimal(ngentot_lc=1000, ngen_season=1.0))
    with pytest.raises(SNANAError, match="mutually exclusive"):
        _n_simulated(cfg)


def test_truncation_not_rounding(minimal):
    """SNANA uses `(int)X` cast which truncates toward zero, not rounds.

    Verify by picking NGEN_SCALE that produces a non-integer × NGENTOT_LC
    where rounding would differ from truncation: 1000 × 1.0007 = 1000.7
    → truncation gives 1000, rounding would give 1001.
    """
    cfg = parse_snana_input_file(minimal(ngentot_lc=1000, ngen_scale=1.0007))
    assert _n_simulated(cfg) == 1000
