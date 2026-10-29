"""Unit tests for SEARCHEFF PHOTPROB: per-observation classification probability.

Pure-JAX tests of the PHOTPROB reader, CDF inversion, SNR/redshift
dependence, CUTVAL, FILTER restriction, REQUIRE_DETECTION, REDUCED_CORR
correlated draws, 2-D maps, and the no-maps fallback. The end-to-end
integration test lives in `tests/parity/test_searcheff_photprob.py`.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_snlc_core.snana_files import MAG_UNDEFINED
from jax_snlc_sim.detection import (
    IVARABS_SNR, PhotoprobMap, evaluate_photprob, prepare_photprob,
    read_pipeline_eff_file)
from jax_snlc_sim.spectroscopic_efficiency import resolve_fieldlist_indices


@pytest.fixture(scope="module")
def photprob_file(searcheff_data_dir):
    return searcheff_data_dir / "SEARCHEFF_PIPELINE_PHOTPROB_TEST.DAT"


def _eval_photprob(maps, snr, filt, filter_names, det, valid, z,
                   sbmag_dict=None, galmag_dict=None, *, seed=42):
    """Test wrapper around the JAX evaluator with simple-shape numpy inputs."""
    snr_2d = jnp.asarray(np.atleast_2d(snr), dtype=jnp.float32)
    n_sne, _ = snr_2d.shape
    filt_2d = jnp.asarray(np.atleast_2d(filt), dtype=jnp.int32)
    det_2d = jnp.asarray(np.atleast_2d(det), dtype=bool)
    valid_2d = jnp.asarray(np.atleast_2d(valid), dtype=bool)
    z_arr = jnp.asarray(np.atleast_1d(z), dtype=jnp.float32)

    n_filt = len(filter_names)
    sbmag_2d = jnp.full((n_sne, n_filt), MAG_UNDEFINED, dtype=jnp.float32)
    galmag_2d = jnp.full((n_sne, n_filt), MAG_UNDEFINED, dtype=jnp.float32)
    for src, dst in [(sbmag_dict, sbmag_2d), (galmag_dict, galmag_2d)]:
        if not src:
            continue
        for fname, vals in src.items():
            if fname in filter_names:
                fi = filter_names.index(fname)
                src_arr = jnp.asarray(vals, dtype=jnp.float32)
                # Reassign via .at (jnp arrays are immutable).
                if dst is sbmag_2d:
                    sbmag_2d = sbmag_2d.at[:, fi].set(src_arr)
                else:
                    galmag_2d = galmag_2d.at[:, fi].set(src_arr)

    prepared = prepare_photprob(maps, filter_names)
    field_match = [jnp.ones(n_sne, dtype=bool) for _ in prepared]
    pp, pp_pass = evaluate_photprob(
        prepared, snr_2d, filt_2d, det_2d, valid_2d, z_arr,
        sbmag_2d, galmag_2d, field_match, jax.random.PRNGKey(seed),
        jnp.arange(n_sne, dtype=jnp.int32),
    )
    return np.asarray(pp), np.asarray(pp_pass)


# =============================================================================
# Reader / CDF normalisation
# =============================================================================


def test_reader(photprob_file):
    maps = read_pipeline_eff_file(str(photprob_file), 0, 0).photprob_maps
    assert len(maps) == 1
    m = maps[0]
    assert m.name == "SNR_TEST"
    assert m.varnames == ["SNR"]
    assert m.nvar_map == 1
    assert m.nfun_cdf == 4
    assert len(m.grid_axes[0]) == 6
    assert m.grid_cdf.shape == (6, 4)
    assert len(m.cdf_bins) == 5
    assert np.allclose(m.cdf_bins, [0, 0.25, 0.5, 0.75, 1.0])
    assert np.allclose(m.grid_cdf[:, -1], 1.0)
    assert m.valmin[0] == 0.0
    assert m.valmax[0] == 50.0


def test_cdf_normalization(photprob_file):
    maps = read_pipeline_eff_file(str(photprob_file), 0, 0).photprob_maps
    m = maps[0]
    assert np.allclose(m.grid_cdf[0], [0.7, 0.9, 0.98, 1.0], atol=0.001)
    assert np.allclose(m.grid_cdf[4], [0.01, 0.05, 0.2, 1.0], atol=0.001)


# =============================================================================
# CDF inversion
# =============================================================================


def _flat_map():
    return PhotoprobMap(
        name="TEST", varnames=["SNR"], ivarabs=[IVARABS_SNR],
        nvar_map=1, nfun_cdf=4,
        grid_axes=[np.array([5.0, 10.0])],
        grid_cdf=np.array([[0.25, 0.5, 0.75, 1.0],
                           [0.25, 0.5, 0.75, 1.0]]),
        cdf_bins=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        valmin=np.array([5.0]), valmax=np.array([10.0]),
    )


def _skewed_map():
    return PhotoprobMap(
        name="TEST", varnames=["SNR"], ivarabs=[IVARABS_SNR],
        nvar_map=1, nfun_cdf=4,
        grid_axes=[np.array([5.0, 10.0])],
        grid_cdf=np.array([[0.8, 0.9, 0.95, 1.0],
                           [0.8, 0.9, 0.95, 1.0]]),
        cdf_bins=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        valmin=np.array([5.0]), valmax=np.array([10.0]),
    )


def test_prepared_maps_can_be_runtime_inputs():
    """A compiled evaluator must accept changing numerical map thresholds."""
    maps = prepare_photprob([_flat_map()], ["r"])
    compiled = jax.jit(evaluate_photprob)
    arguments = (
        jnp.full((2, 8), 7.0), jnp.zeros((2, 8), dtype=jnp.int32),
        jnp.ones((2, 8), dtype=bool), jnp.ones((2, 8), dtype=bool),
        jnp.full(2, .5), jnp.full((2, 1), 99.), jnp.full((2, 1), 99.),
        [jnp.ones(2, dtype=bool)], jax.random.PRNGKey(42),
        jnp.arange(2, dtype=jnp.int32),
    )
    probabilities, passing = compiled(maps, *arguments)
    assert np.all(passing)
    changed = [replace(maps[0], cutval=1.1)]
    changed_probabilities, changed_passing = compiled(changed, *arguments)
    np.testing.assert_array_equal(changed_probabilities, probabilities)
    assert not np.any(changed_passing)


def test_inverse_cdf_flat():
    """Flat CDF → uniform PHOTPROB distribution."""
    n = 10000
    snr = np.full((1, n), 7.0)
    pp, _ = _eval_photprob(
        [_flat_map()], snr, np.zeros((1, n), dtype=int), ["r"],
        np.ones((1, n), dtype=bool), np.ones((1, n), dtype=bool),
        np.array([0.5]),
    )
    vals = pp[0]
    assert np.all(vals >= 0) and np.all(vals <= 1)
    for lo, hi in [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]:
        frac = float(np.mean((vals >= lo) & (vals < hi)))
        assert abs(frac - 0.25) < 0.03


def test_expected_value():
    """Mean PHOTPROB matches analytic expectation for known CDF."""
    expected_mean = 0.8 * 0.125 + 0.1 * 0.375 + 0.05 * 0.625 + 0.05 * 0.875
    n = 50000
    snr = np.full((1, n), 7.0)
    pp, _ = _eval_photprob(
        [_skewed_map()], snr, np.zeros((1, n), dtype=int), ["r"],
        np.ones((1, n), dtype=bool), np.ones((1, n), dtype=bool),
        np.array([0.5]),
    )
    assert abs(float(np.mean(pp[0])) - expected_mean) < 0.01


def test_snr_dependent(photprob_file):
    """Higher SNR → higher mean PHOTPROB on the synthetic file."""
    maps = read_pipeline_eff_file(str(photprob_file), 0, 0).photprob_maps
    n = 5000
    z = np.array([0.5])
    pp_lo, _ = _eval_photprob(
        maps, np.full((1, n), 2.0), np.zeros((1, n), dtype=int), ["r"],
        np.ones((1, n), dtype=bool), np.ones((1, n), dtype=bool), z,
    )
    pp_hi, _ = _eval_photprob(
        maps, np.full((1, n), 30.0), np.zeros((1, n), dtype=int), ["r"],
        np.ones((1, n), dtype=bool), np.ones((1, n), dtype=bool), z,
    )
    assert float(np.mean(pp_hi[pp_hi > -1])) > float(np.mean(pp_lo[pp_lo > -1]))


# =============================================================================
# CUTVAL / FILTER / REQUIRE_DETECTION
# =============================================================================


def test_cutval():
    """CUTVAL=0.5 with uniform CDF → ~50% fail."""
    m = PhotoprobMap(
        name="TEST", varnames=["SNR"], ivarabs=[0],
        nvar_map=1, nfun_cdf=4,
        grid_axes=[np.array([1.0, 50.0])],
        grid_cdf=np.array([[0.25, 0.5, 0.75, 1.0],
                           [0.25, 0.5, 0.75, 1.0]]),
        cdf_bins=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        cutval=0.5,
        valmin=np.array([1.0]), valmax=np.array([50.0]),
    )
    n = 10000
    pp, pp_pass = _eval_photprob(
        [m], np.full((1, n), 10.0), np.zeros((1, n), dtype=int), ["r"],
        np.ones((1, n), dtype=bool), np.ones((1, n), dtype=bool),
        np.array([0.5]),
    )
    # Fail is PHOTPROB <= CUTVAL: SNANA keeps the detect bit only for
    # PHOTPROB > CUTVAL (sntools_trigger.c:2238-2241).
    n_fail = int(np.sum(~pp_pass[0]))
    n_at_or_below = int(np.sum(pp[0] <= 0.5))
    assert n_fail == n_at_or_below
    assert abs(n_fail / n - 0.5) < 0.05


def test_filter_restriction():
    """FILTER restriction: only matching bands get PHOTPROB."""
    m = PhotoprobMap(
        name="TEST", varnames=["SNR"], ivarabs=[0],
        nvar_map=1, nfun_cdf=2,
        grid_axes=[np.array([1.0, 50.0])],
        grid_cdf=np.array([[0.5, 1.0], [0.5, 1.0]]),
        cdf_bins=np.array([0.0, 0.5, 1.0]),
        filterlist="ri",
        valmin=np.array([1.0]), valmax=np.array([50.0]),
    )
    snr = np.full((1, 6), 10.0)
    filt = np.array([[0, 1, 2, 3, 4, 5]])
    pp, _ = _eval_photprob(
        [m], snr, filt,
        ["LSST-u", "LSST-g", "LSST-r", "LSST-i", "LSST-z", "LSST-Y"],
        np.ones((1, 6), dtype=bool), np.ones((1, 6), dtype=bool),
        np.array([0.5]),
    )
    assert pp[0, 0] == -9.0
    assert pp[0, 1] == -9.0
    assert pp[0, 2] > -1
    assert pp[0, 3] > -1
    assert pp[0, 4] == -9.0
    assert pp[0, 5] == -9.0


def test_require_detection():
    """REQUIRE_DETECTION: non-detected obs get sentinel PHOTPROB."""
    m = PhotoprobMap(
        name="TEST", varnames=["SNR"], ivarabs=[0],
        nvar_map=1, nfun_cdf=2,
        grid_axes=[np.array([1.0, 50.0])],
        grid_cdf=np.array([[0.5, 1.0], [0.5, 1.0]]),
        cdf_bins=np.array([0.0, 0.5, 1.0]),
        require_detection=True,
        valmin=np.array([1.0]), valmax=np.array([50.0]),
    )
    snr = np.full((1, 4), 10.0)
    det = np.array([[True, False, True, False]])
    pp, _ = _eval_photprob(
        [m], snr, np.zeros((1, 4), dtype=int), ["r"],
        det, np.ones((1, 4), dtype=bool),
        np.array([0.5]),
    )
    assert pp[0, 0] > -1
    assert pp[0, 1] == -9.0
    assert pp[0, 2] > -1
    assert pp[0, 3] == -9.0


# =============================================================================
# REDUCED_CORR + 2-D map + no-maps
# =============================================================================


def test_reduced_corr():
    """REDUCED_CORR=0.9 produces correlated draws; =0 produces uncorrelated."""
    n_sne, n_obs = 200, 20
    snr = np.full((n_sne, n_obs), 10.0)
    filt = np.zeros((n_sne, n_obs), dtype=int)
    det = np.ones((n_sne, n_obs), dtype=bool)
    valid = np.ones((n_sne, n_obs), dtype=bool)
    z = np.full(n_sne, 0.5)

    def _build(rho):
        return PhotoprobMap(
            name="TEST", varnames=["SNR"], ivarabs=[IVARABS_SNR],
            nvar_map=1, nfun_cdf=100,
            grid_axes=[np.array([1.0, 50.0])],
            grid_cdf=np.tile(np.linspace(0.01, 1.0, 100), (2, 1)),
            cdf_bins=np.linspace(0.0, 1.0, 101),
            reduced_corr=rho,
            valmin=np.array([1.0]), valmax=np.array([50.0]),
        )

    pp, _ = _eval_photprob([_build(0.9)], snr, filt, ["r"], det, valid, z)
    corr = np.corrcoef(np.array([pp[i] for i in range(n_sne) if np.all(pp[i] > -1)]).T)
    mean_off = float(np.mean(corr[np.triu_indices(n_obs, k=1)]))
    assert abs(mean_off - 0.9) < 0.15

    pp_nc, _ = _eval_photprob([_build(0.0)], snr, filt, ["r"], det, valid, z)
    corr_nc = np.corrcoef(np.array([pp_nc[i] for i in range(n_sne) if np.all(pp_nc[i] > -1)]).T)
    mean_nc = float(np.mean(corr_nc[np.triu_indices(n_obs, k=1)]))
    assert abs(mean_nc) < 0.15


def test_2d_map(tmp_path):
    """2-D (SNR × REDSHIFT) PHOTPROB map: high SNR + low z → high PHOTPROB."""
    content = """\
DOCUMENTATION:
    PURPOSE: test 2D photprob
DOCUMENTATION_END:
MAPNAME_PHOTPROB: TEST_2D
NVAR: 5
VARNAMES: SNR REDSHIFT PHOTPROB_WGTLIST
ROW:  0.0  0.0   0.8  0.15  0.05
ROW:  0.0  0.5   0.7  0.2   0.1
ROW:  0.0  1.0   0.6  0.25  0.15
ROW: 25.0  0.0   0.2  0.3   0.5
ROW: 25.0  0.5   0.3  0.3   0.4
ROW: 25.0  1.0   0.4  0.3   0.3
ROW: 50.0  0.0   0.05 0.15  0.8
ROW: 50.0  0.5   0.1  0.2   0.7
ROW: 50.0  1.0   0.2  0.3   0.5
ENDMAP:
"""
    path = tmp_path / "photprob_2d.DAT"
    path.write_text(content)
    maps = read_pipeline_eff_file(str(path), 0, 0).photprob_maps
    assert len(maps) == 1
    m = maps[0]
    assert m.nvar_map == 2
    assert m.nfun_cdf == 3
    assert m.grid_cdf.shape == (3, 3, 3)
    assert m.varnames == ["SNR", "REDSHIFT"]

    n = 5000
    pp_hi, _ = _eval_photprob(
        maps, np.full((1, n), 40.0), np.zeros((1, n), dtype=int), ["r"],
        np.ones((1, n), dtype=bool), np.ones((1, n), dtype=bool),
        np.array([0.1]),
    )
    pp_lo, _ = _eval_photprob(
        maps, np.full((1, n), 5.0), np.zeros((1, n), dtype=int), ["r"],
        np.ones((1, n), dtype=bool), np.ones((1, n), dtype=bool),
        np.array([0.9]),
    )
    mean_hi = float(np.mean(pp_hi[pp_hi > -1]))
    mean_lo = float(np.mean(pp_lo[pp_lo > -1]))
    assert mean_hi > mean_lo
    assert mean_hi > 0.5
    assert mean_lo < 0.4


def test_no_maps():
    """No PHOTPROB maps: all entries get sentinel; pass=True."""
    snr = np.full((2, 5), 10.0)
    pp, pp_pass = _eval_photprob(
        [], snr, np.zeros((2, 5), dtype=int), ["r"],
        np.ones((2, 5), dtype=bool), np.ones((2, 5), dtype=bool),
        np.array([0.5, 0.5]),
    )
    assert np.all(pp == -9.0)
    assert np.all(pp_pass)


# =============================================================================
# Reader validation + map-overlap fatal (SNANA sntools_trigger.c:2589-2594)
# =============================================================================


def test_overlapping_maps_fatal():
    """Two maps matching the same (field, filter) abort, as SNANA does on
    NMATCH>1 in setObs_for_PHOTPROB (sntools_trigger.c:2589-2594)."""
    from jax_snlc_core.logging import SNANAError

    def _map(filterlist):
        return PhotoprobMap(
            name=f"TEST_{filterlist}", varnames=["SNR"], ivarabs=[IVARABS_SNR],
            nvar_map=1, nfun_cdf=2,
            grid_axes=[np.array([1.0, 50.0])],
            grid_cdf=np.array([[0.5, 1.0], [0.5, 1.0]]),
            cdf_bins=np.array([0.0, 0.5, 1.0]),
            filterlist=filterlist,
            valmin=np.array([1.0]), valmax=np.array([50.0]),
        )

    with pytest.raises(SNANAError, match="PHOTPROB maps match"):
        prepare_photprob([_map("ri"), _map("iz")], ["r", "i", "z"])

    # Disjoint filter lists are fine.
    prepare_photprob([_map("r"), _map("iz")], ["r", "i", "z"])


def test_reader_rejects_zero_mass_and_duplicate_rows(tmp_path):
    """A ROW whose weights sum to zero, or a repeated grid coordinate, is
    malformed input and fatal."""
    from jax_snlc_core.logging import SNANAError

    header = (
        "DOCUMENTATION:\n    PURPOSE: test\nDOCUMENTATION_END:\n"
        "MAPNAME_PHOTPROB: TEST_BAD\nNVAR: 3\n"
        "VARNAMES: SNR PHOTPROB_WGTLIST\n"
    )
    zero_mass = header + "ROW: 1.0 0.0 0.0\nROW: 50.0 0.5 0.5\nENDMAP:\n"
    path = tmp_path / "zero_mass.DAT"
    path.write_text(zero_mass)
    with pytest.raises(SNANAError, match="sums"):
        read_pipeline_eff_file(str(path), 0, 0).photprob_maps

    dup = header + ("ROW: 1.0 0.5 0.5\nROW: 1.0 0.4 0.6\n"
                    "ROW: 50.0 0.5 0.5\nENDMAP:\n")
    path2 = tmp_path / "dup_rows.DAT"
    path2.write_text(dup)
    with pytest.raises(SNANAError, match="more than once"):
        read_pipeline_eff_file(str(path2), 0, 0).photprob_maps


def test_reduced_corr_spans_maps():
    """One covariance over ALL PHOTPROB observations: obs served by two
    different maps correlate at sqrt(CORR_i*CORR_j)
    (setRan_for_PHOTPROB, sntools_trigger.c:2679-2694)."""
    n_sne, n_obs = 400, 10
    snr = np.full((n_sne, n_obs), 10.0)
    filt = np.tile(np.array([0, 1] * (n_obs // 2)), (n_sne, 1))
    det = np.ones((n_sne, n_obs), dtype=bool)
    valid = np.ones((n_sne, n_obs), dtype=bool)
    z = np.full(n_sne, 0.5)

    def _map(filterlist, rho):
        return PhotoprobMap(
            name=f"TEST_{filterlist}", varnames=["SNR"], ivarabs=[IVARABS_SNR],
            nvar_map=1, nfun_cdf=100,
            grid_axes=[np.array([1.0, 50.0])],
            grid_cdf=np.tile(np.linspace(0.01, 1.0, 100), (2, 1)),
            cdf_bins=np.linspace(0.0, 1.0, 101),
            filterlist=filterlist,
            reduced_corr=rho,
            valmin=np.array([1.0]), valmax=np.array([50.0]),
        )

    pp, _ = _eval_photprob(
        [_map("r", 0.9), _map("i", 0.4)], snr, filt,
        ["r", "i"], det, valid, z)
    assert np.all(pp > -1)
    corr = np.corrcoef(pp.T)
    # Same-map pairs: r obs at 0.9, i obs at 0.4; cross-map pairs at
    # sqrt(0.9*0.4) = 0.6.
    r_idx = np.arange(0, n_obs, 2)
    i_idx = np.arange(1, n_obs, 2)
    rr = np.mean([corr[a, b] for a in r_idx for b in r_idx if a < b])
    ii = np.mean([corr[a, b] for a in i_idx for b in i_idx if a < b])
    ri = np.mean([corr[a, b] for a in r_idx for b in i_idx])
    assert abs(rr - 0.9) < 0.15
    assert abs(ii - 0.4) < 0.15
    assert abs(ri - np.sqrt(0.9 * 0.4)) < 0.15


def test_fieldlist_matches_substrings_across_simlib_entries():
    """FIELDLIST matches substrings and tolerates fields absent from the SIMLIB,
    as SNANA's per-event MATCH_SEARCHEFF_FIELD does, not only the last entry
    of each field name."""
    m = PhotoprobMap(
        name="TEST", varnames=["SNR"], ivarabs=[IVARABS_SNR], nvar_map=1,
        nfun_cdf=100, grid_axes=[np.array([1.0, 50.0])],
        grid_cdf=np.tile(np.linspace(0.01, 1.0, 100), (2, 1)),
        cdf_bins=np.linspace(0.0, 1.0, 101), fieldlist="C3+X1+UNUSED",
        valmin=np.array([1.0]), valmax=np.array([50.0]))
    resolve_fieldlist_indices([m], ["C3", "X1", "E2", "C3", "X"])
    np.testing.assert_array_equal(m.fieldlist_mask,
                                  [True, True, False, True, True])
