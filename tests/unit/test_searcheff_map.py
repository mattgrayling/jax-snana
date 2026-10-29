"""Unit tests for the ``spectroscopic_efficiency`` module — SNANA's SPECID +
zHOST + Type-Assignment selection-efficiency layer.

Pure-JAX tests covering: NONE/ZERO/ONE sentinels, OPT_EXTRAP,
2-D interpolation, multi-map OR, REQUIRE flag, FIELDLIST,
and bundled map formats.

Scenario tests live in
``tests/integration/test_searcheff_spec_zhost_scenarios.py``.
"""

import gzip
import shutil

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_snlc_core.interpolation import interp_nd_regular
from jax_snlc_core.logging import SNANAError
from jax_snlc_sim.detection import read_pipeline_logic
from jax_snlc_sim.simulation import Instrument
from jax_snlc_sim.spectroscopic_efficiency import (
    FLAG_ALL_FAIL, FLAG_ALL_PASS, FLAG_NORMAL, MXMAP_SEARCHEFF_MAP, SearchEffMap,
    check_applymask_searcheff, eff_outside_window, evaluate_searcheff,
    prepare_searcheff, read_searcheff_maps, resolve_fieldlist_indices,
    resolve_searcheff_file)

ZP_FLUXCAL = Instrument.zp_fluxcal


def _eval_maps(maps, sn_vars_np, n_sne, *, spec_scale=1.0,
               simlib_field_names=None, field_idx=None, seed=42):
    """Test helper: prepare + evaluate SEARCHEFF maps via the JAX API."""
    if simlib_field_names is not None:
        resolve_fieldlist_indices(maps, simlib_field_names)
    sn_vars = _sn_vars(sn_vars_np, n_sne, field_idx)
    prepared, _ = prepare_searcheff(maps, sn_vars, ZP_FLUXCAL, var_shifts={},
                                    magshift=0.0)
    eff, passed, _ = evaluate_searcheff(
        prepared, FLAG_NORMAL, n_sne, jax.random.PRNGKey(seed),
        jnp.arange(n_sne, dtype=jnp.int32), spec_scale=spec_scale,
    )
    return np.asarray(eff), np.asarray(passed)


def _sn_vars(sn_vars_np, n_sne, field_idx=None):
    """The per-SN variables with the field index and PEAKMJD every event
    carries; the PEAKMJD lies inside a map's default 10000-90000 range."""
    sn_vars = {k: jnp.asarray(v) for k, v in sn_vars_np.items()}
    sn_vars["field_idx"] = jnp.asarray(
        np.zeros(n_sne) if field_idx is None else field_idx, dtype=jnp.int32)
    sn_vars.setdefault("PEAKMJD", jnp.full(n_sne, 60000.0, dtype=jnp.float32))
    return sn_vars


# =============================================================================
# spectroscopic_efficiency sentinels + interp
# =============================================================================


def test_none_zero_one_sentinels():
    _, f1 = read_searcheff_maps("NONE", "SPEC", "")
    _, f2 = read_searcheff_maps("ONE", "SPEC", "")
    _, f3 = read_searcheff_maps("ZERO", "SPEC", "")
    _, f4 = read_searcheff_maps("", "zHOST", "")
    assert f1 == FLAG_ALL_PASS
    assert f2 == FLAG_ALL_PASS
    assert f3 == FLAG_ALL_FAIL
    assert f4 == FLAG_ALL_PASS


def test_opt_extrap():
    """OPT_EXTRAP=0 returns 0 outside grid; OPT_EXTRAP=1 clamps to edge."""
    axes = [jnp.array([0.0, 0.05])]
    vals = jnp.array([1.0, 1.0])
    pts = jnp.array([[0.0], [0.03], [0.05], [0.10]])
    r0 = np.asarray(interp_nd_regular(pts, axes, vals, opt_extrap=0))
    r1 = np.asarray(interp_nd_regular(pts, axes, vals, opt_extrap=1))
    assert np.allclose(r0, [1, 1, 1, 0])
    assert np.allclose(r1, [1, 1, 1, 1])


def test_2d_interp():
    axes = [jnp.array([0.0, 1.0]), jnp.array([0.0, 1.0])]
    vals = jnp.array([[0.0, 0.5], [0.5, 1.0]])
    pts = jnp.array([[0.5, 0.5], [0.0, 0.0], [1.0, 1.0]])
    result = np.asarray(interp_nd_regular(pts, axes, vals, opt_extrap=0))
    assert np.allclose(result, [0.5, 0.0, 1.0])


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
def test_interp_knot_and_edge_gradients(dtype):
    """An affine map retains its slopes and the clipping derivative at each edge."""
    axes = (jnp.array([0., 1., 3.], dtype=dtype),
            jnp.array([0., 1., 2.], dtype=dtype))
    values = 2 * axes[0][:, None] + axes[1][None, :]
    points = jnp.array([[-1., .4], [0., .4], [.4, .4],
                        [1., .4], [3., .4], [4., .4]], dtype=dtype)
    gradients = jax.jit(jax.vmap(jax.grad(
        lambda p: interp_nd_regular(p[None], axes, values, opt_extrap=1)[0])))(points)
    np.testing.assert_allclose(gradients, [[0., 1.], [1., 1.], [2., 1.],
                                          [2., 1.], [1., 1.], [0., 1.]], atol=1e-6)


def test_eff_roundoff_window():
    """gen_searcheff_map lets EFF overshoot 1 by round-off (then 0.99999)
    and dip below 0 by round-off (then 0)."""
    m = SearchEffMap(varnames=["REDSHIFT"], grid_axes=[np.array([0.0, 1.0])],
                     grid_values=np.array([1.0002, -0.0005]))
    eff, _ = _eval_maps([m], {"sim_redshift_cmb": np.array([0.0, 1.0])}, 2)
    assert np.allclose(eff, [0.99999, 0.0])
    assert not eff_outside_window(eff).any()


@pytest.mark.parametrize("bad_eff, flagged", [
    ("1.2", True), ("-0.002", True), ("-0.001", False)])
def test_eff_outside_roundoff_window_flagged_per_event(tmp_path, bad_eff,
                                                       flagged):
    """A node outside the window loads; only an event whose EFF lands
    outside it is flagged for the abort (sntools_trigger.c:3011-3019). At
    the -0.001 edge SNANA's OR combination 1 - (1 - EFF) lands just above
    -0.001 and is rounded to 0, so that event is not flagged."""
    path = tmp_path / "SEARCHEFF_SPEC_BAD.DAT"
    path.write_text(
        "DOCUMENTATION:\n  PURPOSE: test\nDOCUMENTATION_END:\n\n"
        "VARNAMES: REDSHIFT SPECEFF\n"
        f"SPECEFF: 0.0 0.5\nSPECEFF: 1.0 {bad_eff}\n")
    maps, _ = read_searcheff_maps(str(path), "SPEC", "")
    eff, _ = _eval_maps(maps, {"sim_redshift_cmb": np.array([0.0, 1.0])}, 2)
    assert list(np.asarray(eff_outside_window(eff))) == [False, flagged]


# =============================================================================
# Map combination semantics
# =============================================================================


def test_multimap_or():
    """eff_combined = 1 - (1-eff1)(1-eff2) for two normal maps."""
    m1 = SearchEffMap(varnames=["REDSHIFT"], grid_axes=[np.array([0.0, 1.0])],
                      grid_values=np.array([0.3, 0.3]))
    m2 = SearchEffMap(varnames=["REDSHIFT"], grid_axes=[np.array([0.0, 1.0])],
                      grid_values=np.array([0.4, 0.4]))
    eff, _ = _eval_maps([m1, m2], {"sim_redshift_cmb": np.array([0.5])}, 1)
    assert np.allclose(eff, 0.58)


def test_lone_require_map_folds_into_or():
    """A REQUIRE map folds into the OR efficiency AND gets its own draw
    (sntools_trigger.c:2971-2978): a lone REQUIRE map with eff=b reports
    EFF=b and passes with probability b^2."""
    m = SearchEffMap(varnames=["REDSHIFT"], grid_axes=[np.array([0.0, 1.0])],
                     grid_values=np.array([0.6, 0.6]), require=True)
    n = 4000
    eff, passed = _eval_maps([m], {"sim_redshift_cmb": np.full(n, 0.5)}, n)
    assert np.allclose(eff, 0.6)
    assert abs(passed.mean() - 0.36) < 0.05


def test_require_flag():
    """REQUIRE map with eff=0 blocks all draws."""
    m_normal = SearchEffMap(varnames=["REDSHIFT"], grid_axes=[np.array([0.0, 1.0])],
                            grid_values=np.array([1.0, 1.0]))
    m_req = SearchEffMap(varnames=["REDSHIFT"], grid_axes=[np.array([0.0, 1.0])],
                         grid_values=np.array([0.0, 0.0]), require=True)
    _, passed = _eval_maps([m_normal, m_req], {"sim_redshift_cmb": np.full(100, 0.5)}, 100)
    assert int(np.sum(passed)) == 0


def test_require_draws_shared_by_key():
    """Under REFAC_SEARCHEFF_MAP 1 a zHOST REQUIRE map reads the SPECID
    random of the same map index (snlc_sim.c:24771, sntools_trigger.c:2976),
    so the two map sets pass the same events where their efficiencies agree;
    with their own randoms they do not."""
    maps = [SearchEffMap(varnames=["REDSHIFT"], grid_axes=[np.array([0.0, 1.0])],
                         grid_values=np.array([1.0, 1.0])),
            SearchEffMap(varnames=["REDSHIFT"], grid_axes=[np.array([0.0, 1.0])],
                         grid_values=np.array([0.5, 0.5]), require=True)]
    n = 400
    sn_vars = _sn_vars({"sim_redshift_cmb": np.full(n, 0.5)}, n)
    prepared, _ = prepare_searcheff(maps, sn_vars, ZP_FLUXCAL, var_shifts={},
                                    magshift=0.0)
    spec_key, zhost_key = jax.random.PRNGKey(1), jax.random.PRNGKey(2)
    sn_idx = jnp.arange(n, dtype=jnp.int32)
    _, spec_pass, _ = evaluate_searcheff(prepared, FLAG_NORMAL, n, spec_key,
                                         sn_idx)
    _, shared, _ = evaluate_searcheff(prepared, FLAG_NORMAL, n, zhost_key,
                                      sn_idx,
                                      require_key=jax.random.split(spec_key)[1])
    _, own, _ = evaluate_searcheff(prepared, FLAG_NORMAL, n, zhost_key, sn_idx)
    assert np.array_equal(np.asarray(shared), np.asarray(spec_pass))
    assert not np.array_equal(np.asarray(own), np.asarray(spec_pass))


def test_field_filter():
    """FIELDLIST restricts the map to the SIMLIB fields whose names it
    contains (MATCH_SEARCHEFF_FIELD, sntools_trigger.c:3266-3306)."""
    m = SearchEffMap(varnames=["REDSHIFT"], grid_axes=[np.array([0.0, 1.0])],
                     grid_values=np.array([1.0, 1.0]), fieldlist="C1+C2")
    eff, _ = _eval_maps(
        [m], {"sim_redshift_cmb": np.array([0.5, 0.5, 0.5])}, 3,
        simlib_field_names=["C1", "X1", "C2"], field_idx=[0, 1, 2],
    )
    assert eff[0] == 1.0
    assert eff[1] == 0.0
    assert eff[2] == 1.0


def test_dtseason_peak_eval(searcheff_data_dir):
    """DTSEASON_PEAK map: SNe at season edge fail, mid-season pass."""
    maps, _ = read_searcheff_maps(
        str(searcheff_data_dir / "SEARCHEFF_SPEC_TEST_DTSEASON.DAT"), "SPEC", ""
    )
    assert len(maps) == 1
    assert maps[0].varnames == ["DTSEASON_PEAK"]

    eff, _ = _eval_maps(maps, {"dtseason_peak": np.array([-50.0, -5.0, 0.0, 5.0, 50.0])}, 5)
    assert eff[0] > 0.9
    assert eff[1] < 0.1
    assert eff[2] < 0.1
    assert eff[3] < 0.1
    assert eff[4] > 0.9


def test_compute_dtseason_peak():
    """DTSEASON_PEAK follows SNANA's convention (snlc_sim.c:21542-21559):
    per season the candidates are PEAKMJD - MJD_MIN and MJD_MAX - PEAKMJD,
    the smallest |DT| wins, so DT is positive in season and negative out.
    """
    from jax_snlc_sim.simlib import compute_dtseason_peak

    # Field 0: seasons (60000-60100), (60200-60300). Field 1: one season
    # (60500-60800) plus an inf/-inf padded slot that must be ignored.
    season_starts = jnp.array(
        [[60000.0, 60200.0],
         [60500.0, np.inf]], dtype=jnp.float32)
    season_ends = jnp.array(
        [[60100.0, 60300.0],
         [60800.0, -np.inf]], dtype=jnp.float32)

    # Each row: (peak_mjd, field_idx, expected DTSEASON_PEAK).
    cases = [
        (60050.0, 0, +50.0),   # mid-season: to start and to end both +50
        (60080.0, 0, +20.0),   # in season, nearest edge is the season end
        (60150.0, 0, -50.0),   # between seasons: negative, out of season
        (59900.0, 0, -100.0),  # before all seasons
        (60600.0, 1, +100.0),  # in season, nearest edge is the start
        (60900.0, 1, -100.0),  # after the last season
    ]
    peak_mjds = jnp.array([c[0] for c in cases], dtype=jnp.float32)
    field_idx = jnp.array([c[1] for c in cases], dtype=jnp.int32)
    expected = np.array([c[2] for c in cases], dtype=np.float32)

    dt = np.asarray(compute_dtseason_peak(
        peak_mjds, field_idx, season_starts, season_ends,
    ))
    assert np.allclose(dt, expected, atol=1e-4)


def test_spec_scale(searcheff_data_dir):
    """SEARCHEFF_SPEC_SCALE: multiplicative scale on efficiency, clipped at 1."""
    maps, _ = read_searcheff_maps(
        str(searcheff_data_dir / "SEARCHEFF_SPEC_TEST_SCALE.DAT"), "SPEC", ""
    )

    eff_unity, _ = _eval_maps(maps, {"sim_redshift_cmb": np.array([0.5])}, 1)
    assert np.allclose(eff_unity, 0.5)

    eff_half, _ = _eval_maps(
        maps, {"sim_redshift_cmb": np.array([0.5])}, 1, spec_scale=0.5,
    )
    assert np.allclose(eff_half, 0.25)

    eff_clipped, _ = _eval_maps(
        maps, {"sim_redshift_cmb": np.array([0.5])}, 1, spec_scale=2.0,
    )
    assert np.allclose(eff_clipped, 1.0)


# =============================================================================
# FIELD-dependent pipeline detection maps + endpoint efficiencies
# =============================================================================


def test_detect_map_field_matching_and_overlap_fatal():
    """FIELD-dependent detection maps match on FIELDLIST like SNANA's
    MATCH_SEARCHEFF_FIELD (substring, 'ALL' matches everything;
    sntools_trigger.c:3266-3307); more than one map matching an
    observation aborts (GETEFF_PIPELINE_DETECT, :2449-2456)."""
    from jax_snlc_sim.detection import build_searcheff_config
    from jax_snlc_sim.detection import DetectionCurve

    def _map(filterlist, fieldlist):
        return DetectionCurve(filterlist=filterlist, fieldlist=fieldlist,
                              vals=np.array([0.0, 5.0]),
                              effs=np.array([0.1, 0.9]),
                              use_abs=False, is_mag=False)

    cfg = build_searcheff_config(
        [_map('g', 'X1+X2'), _map('g', 'C3')], [1], 1, 0.4,
        ['LSST-g', 'LSST-r'], ['X1', 'C3', 'E9'], ZP_FLUXCAL,
        apply_detect_single=False)
    mi = np.asarray(cfg['map_index'])
    assert mi[0, 0] == 0   # X1, g -> first map
    assert mi[1, 0] == 1   # C3, g -> second map
    assert mi[2, 0] == -1  # E9, g -> no map (EFF=0)
    assert np.all(mi[:, 1] == -1)  # r has no map at all

    with pytest.raises(SNANAError, match="PIPELINE/DETECT maps"):
        build_searcheff_config(
            [_map('g', 'X1+X2'), _map('g', 'ALL')], [1], 1, 0.4,
            ['LSST-g'], ['X1'], ZP_FLUXCAL, apply_detect_single=False)


def test_detect_curve_endpoint_efficiencies():
    """Out-of-range detection-curve values return the endpoint
    efficiencies, not 0/1 sentinels (GETEFF_PIPELINE_DETECT,
    sntools_trigger.c:2518-2519); SNR<=0 gives 0 for non-ABS curves
    (:2494)."""
    from jax_snlc_sim.detection import build_searcheff_config, detect_all
    from jax_snlc_sim.detection import DetectionCurve

    cfg = build_searcheff_config(
        [DetectionCurve(filterlist='g', fieldlist='ALL',
                        vals=np.array([2.0, 10.0]),
                        effs=np.array([0.25, 0.98]),
                        use_abs=False, is_mag=False)],
        [1], 1, 0.4, ['LSST-g'], ['X1'], ZP_FLUXCAL,
        apply_detect_single=False)

    snrs = jnp.array([[1.0, 50.0, -1.0, 6.0]], dtype=jnp.float32)
    n = 200
    counts = np.zeros(4)
    for seed in range(n):
        keys = jax.vmap(lambda i: jax.random.fold_in(
            jax.random.PRNGKey(seed), i))(jnp.arange(1))
        det = np.asarray(detect_all(
            cfg, snrs, jnp.zeros_like(snrs), jnp.ones_like(snrs, dtype=jnp.int32),
            jnp.zeros_like(snrs, dtype=jnp.int32),
            jnp.ones_like(snrs, dtype=bool),
            jnp.zeros(1, dtype=jnp.int32), keys))
        counts += det[0]
    rates = counts / n
    assert abs(rates[0] - 0.25) < 0.1   # below grid -> EFF_atmin
    assert abs(rates[1] - 0.98) < 0.05  # above grid -> EFF_atmax
    assert rates[2] == 0.0              # SNR<=0 -> 0
    assert 0.3 < rates[3] < 0.9         # in-range interpolation


_DOCANA = "DOCUMENTATION:\n  PURPOSE: test map\nDOCUMENTATION_END:\n\n"


def _map_file(tmp_path, body, name="SEARCHEFF_zHOST_TEST.DAT"):
    f = tmp_path / name
    f.write_text(_DOCANA + body)
    return str(f)


def test_rows_without_varnames_are_not_a_map(tmp_path):
    """Only a VARNAMES line opens a map; keyed rows with none above them
    are passed over, so the file holds no map (read_searcheff_map,
    sntools_trigger.c:1370-1417) and gives every event EFF=1
    (gen_searcheff_map, :2898-2908). A FIELD key before them restricts
    nothing."""
    path = _map_file(tmp_path, "FIELD: C3\n"
                               "HOSTEFF: 0.05 0.90\n"
                               "HOSTEFF: 0.50 0.40\n")
    maps, flag = read_searcheff_maps(path, "zHOST", "")
    assert maps == []
    assert flag == FLAG_NORMAL


def test_rows_of_the_other_map_type_are_not_read(tmp_path):
    """A SPEC map reads SPECEFF rows only (read_searcheff_map,
    sntools_trigger.c:1269-1276): HOSTEFF rows under its VARNAMES line are
    lines without its row key, and ten of those in a row abort
    (read_GRIDMAP, sntools_gridmap.c:193-203)."""
    rows = "".join(f"HOSTEFF: {0.1 * i:.1f} 1.0\n" for i in range(12))
    path = _map_file(tmp_path, "VARNAMES: REDSHIFT SPECEFF\n" + rows)
    with pytest.raises(SNANAError, match="10 lines in a row that are not 'SPECEFF:'"):
        read_searcheff_maps(path, "SPEC", "")


def test_efficiency_column_is_found_by_name(tmp_path):
    """The efficiency column is the one named SPECEFF or HOSTEFF, in
    either file (assign_MAP_VARNAME, sntools_trigger.c:1507-1517), and a
    HOSTLIB column resolves through the HOSTLIB values the run carries."""
    for eff_name in ("HOSTEFF", "SPECEFF"):
        path = _map_file(tmp_path, f"VARNAMES: ZTRUE {eff_name}\n"
                                   "HOSTEFF: 0.0 1.0\n"
                                   "HOSTEFF: 2.0 1.0\n")
        maps, _ = read_searcheff_maps(path, "zHOST", "")
        assert maps[0].varnames == ["ZTRUE"]
        eff, passed = _eval_maps(maps, {"host_ZTRUE": np.array([0.1, 0.5, 1.2])}, 3)
        assert np.allclose(eff, 1.0)
        assert passed.all()


@pytest.mark.parametrize("varnames, match", [
    ("ZTRUE EFF", "no efficiency column"),
    ("HOSTEFF ZTRUE", "efficiency column HOSTEFF must be the last"),
])
def test_efficiency_column_missing_or_not_last_aborts(tmp_path, varnames,
                                                      match):
    """A map with no SPECEFF/HOSTEFF column, or with it before a map
    variable, aborts: SNANA cannot evaluate a column that is neither
    (assign_MAP_VARNAME, sntools_trigger.c:1600-1604; LOAD_SEARCHEFF_VAR,
    :3125-3128)."""
    path = _map_file(tmp_path, f"VARNAMES: {varnames}\n"
                               "HOSTEFF: 0.0 1.0\n"
                               "HOSTEFF: 2.0 1.0\n")
    with pytest.raises(SNANAError, match=match):
        read_searcheff_maps(path, "zHOST", "")


def test_unknown_map_variable_aborts(tmp_path):
    """A map variable that is neither a per-event quantity nor a HOSTLIB
    column is fatal (assign_MAP_VARNAME, sntools_trigger.c:1600-1604), not
    a silent rejection."""
    path = _map_file(tmp_path, "VARNAMES: NOSUCHVAR HOSTEFF\n"
                               "HOSTEFF: 0.0 1.0\n"
                               "HOSTEFF: 2.0 1.0\n")
    maps, _ = read_searcheff_maps(path, "zHOST", "")
    with pytest.raises(SNANAError, match="NOSUCHVAR"):
        _eval_maps(maps, {"sim_redshift_cmb": np.array([0.1])}, 1)


def test_non_uniform_1d_map_aborts(tmp_path):
    """A one-variable map needs uniform steps like any other
    (read_GRIDMAP, sntools_gridmap.c:163-177, :211-215)."""
    path = _map_file(tmp_path, "VARNAMES: REDSHIFT HOSTEFF\n"
                               "HOSTEFF: 0.0 1.0\n"
                               "HOSTEFF: 0.1 0.8\n"
                               "HOSTEFF: 0.3 0.2\n")
    with pytest.raises(SNANAError, match="non-uniform"):
        read_searcheff_maps(path, "zHOST", "")


def test_repeated_knot_keeps_the_later_row(tmp_path):
    """A row repeating a knot passes the uniform-step check, and the later
    row replaces the earlier one (init_interp_GRIDMAP INVMAP,
    sntools_gridmap.c:477)."""
    path = _map_file(tmp_path, "VARNAMES: REDSHIFT HOSTEFF\n"
                               "HOSTEFF: 0.0 1.0\n"
                               "HOSTEFF: 0.5 0.9\n"
                               "HOSTEFF: 0.5 0.4\n"
                               "HOSTEFF: 1.0 0.2\n")
    maps, _ = read_searcheff_maps(path, "zHOST", "")
    np.testing.assert_allclose(maps[0].grid_axes[0], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(maps[0].grid_values, [1.0, 0.4, 0.2])


def test_map_attributes_reset_after_each_map(tmp_path):
    """FIELD, PEAKMJD_RANGE and REQUIRE apply to the next map only, which
    then falls back to all fields and 10000 < PEAKMJD < 90000; OPT_EXTRAP
    applies to the maps after it; OPT_FIELDMATCH_REQUIRE holds for the whole
    file (read_searcheff_map, sntools_trigger.c:1341-1415, :2864). A blank
    line ends a map."""
    path = _map_file(tmp_path,
                     "FIELD: C3+X3\nPEAKMJD_RANGE: 60000 61000\nREQUIRE: 1\n"
                     "VARNAMES: REDSHIFT HOSTEFF\n"
                     "HOSTEFF: 0.0 1.0\nHOSTEFF: 1.0 1.0\n\n"
                     "OPT_EXTRAP: 1\n"
                     "VARNAMES: REDSHIFT HOSTEFF\n"
                     "HOSTEFF: 0.0 0.5\nHOSTEFF: 1.0 0.5\n\n"
                     "OPT_FIELDMATCH_REQUIRE: 0\n")
    first, second = read_searcheff_maps(path, "zHOST", "")[0]
    assert (first.fieldlist, first.peakmjd_range, first.require,
            first.opt_extrap) == ("C3+X3", (60000.0, 61000.0), True, 0)
    assert (second.fieldlist, second.peakmjd_range, second.require,
            second.opt_extrap) == ("ALL", (10000.0, 90000.0), False, 1)
    assert first.opt_fieldmatch_require == second.opt_fieldmatch_require == 0


def test_row_with_an_extra_word_that_is_not_a_comment_aborts(tmp_path):
    """A row holds exactly one value per column, then nothing or a '#'
    comment (read_GRIDMAP, sntools_gridmap.c:139-156)."""
    path = _map_file(tmp_path, "VARNAMES: REDSHIFT HOSTEFF\n"
                               "HOSTEFF: 0.0 1.0  # fine\n"
                               "HOSTEFF: 1.0 1.0 0.5\n")
    with pytest.raises(SNANAError, match="expected 2 values"):
        read_searcheff_maps(path, "zHOST", "")


@pytest.mark.parametrize("apply_opt, spec, zhost, ok", [
    (3, "NONE", "NONE", True),     # a missing key passes every event
    (15, "maps", "maps", True),
    (2, "ZERO", "NONE", False),    # ZERO confirms nothing
    (2, "empty", "NONE", False),   # nor does a file with no map
    (4, "ZERO", "maps", True),     # zHOST needs only some SPEC setting
    (4, "empty", "maps", False),
    (4, "maps", "ZERO", False),
    (8, "ZERO", "ZERO", False),
    (16, "NONE", "NONE", False),   # no such stage
])
def test_apply_searcheff_opt_beyond_the_maps_aborts(apply_opt, spec, zhost,
                                                    ok):
    """APPLY_SEARCHEFF_OPT may ask only for stages the SPEC and zHOST maps
    can pass (check_APPLYMASK_SEARCHEFF, sntools_trigger.c:133-194)."""
    one_map = [SearchEffMap(varnames=["REDSHIFT"],
                            grid_axes=[np.array([0.0, 1.0])],
                            grid_values=np.array([1.0, 1.0]))]
    loaded = {"NONE": ([], FLAG_ALL_PASS), "ZERO": ([], FLAG_ALL_FAIL),
              "empty": ([], FLAG_NORMAL), "maps": (one_map, FLAG_NORMAL)}
    args = (apply_opt, *loaded[spec], *loaded[zhost])
    if ok:
        check_applymask_searcheff(*args)
    else:
        with pytest.raises(SNANAError, match="Invalid APPLY_SEARCHEFF_OPT"):
            check_applymask_searcheff(*args)


def test_pipeline_eff_file_honours_the_input_files_photflag_bits(
        searcheff_data_dir):
    """A PHOTFLAG the input file already set wins; the file fills in only the
    one still at zero (sntools_trigger.c:321-325)."""
    from jax_snlc_sim.detection import read_pipeline_eff_file

    path = str(searcheff_data_dir / "SEARCHEFF_PIPELINE_LSST.DAT")
    both_from_file = read_pipeline_eff_file(path, 0, 0)
    assert (both_from_file.photflag_detect,
            both_from_file.photflag_trigger) == (4096, 2048)
    overridden = read_pipeline_eff_file(path, 64, 0)
    assert (overridden.photflag_detect, overridden.photflag_trigger) == (64, 2048)


def test_pipeline_eff_file_reads_a_gzipped_file(tmp_path, searcheff_data_dir):
    """The detection curves come through the shared io helpers, so a gzipped
    pipeline file reads like a plain one."""
    from jax_snlc_sim.detection import read_pipeline_eff_file

    src = searcheff_data_dir / "SEARCHEFF_PIPELINE_LSST.DAT"
    gz = tmp_path / "SEARCHEFF_PIPELINE_LSST.DAT.gz"
    with open(src, 'rb') as fin, gzip.open(gz, 'wb') as fout:
        shutil.copyfileobj(fin, fout)

    plain = read_pipeline_eff_file(str(src), 0, 0)
    zipped = read_pipeline_eff_file(str(gz), 0, 0)
    assert len(zipped.curves) == len(plain.curves)
    for a, b in zip(zipped.curves, plain.curves):
        assert a.filterlist == b.filterlist
        np.testing.assert_array_equal(a.vals, b.vals)
        np.testing.assert_array_equal(a.effs, b.effs)


def test_pipeline_eff_file_requires_documentation(tmp_path):
    """SEARCHEFF maps are DOCANA-required whatever REQUIRE_DOCANA says
    (INPUTS_SEARCHEFF.OPTMASK_OPENFILE, snlc_sim.c:1297)."""
    from jax_snlc_sim.detection import read_pipeline_eff_file

    f = tmp_path / "SEARCHEFF_PIPELINE_NODOC.DAT"
    f.write_text("FILTER: g\nSNR: 0.0 0.0\nSNR: 5.0 1.0\n")
    with pytest.raises(SNANAError):
        read_pipeline_eff_file(str(f), 0, 0)


def test_skip_docana_drops_the_block_and_only_the_block():
    """The one rule every SNANA reader uses: a line whose first token is
    exactly DOCUMENTATION: opens the block, DOCUMENTATION_END: closes it, and
    both keys go with the body (rd2columnFile, sntools.c:7297-7303)."""
    from jax_snlc_core.snana_files import skip_docana

    lines = ["DOCUMENTATION:\n", "  PURPOSE: SNR: 5.0 1.0\n",
             "DOCUMENTATION_END:\n", "FILTER: g\n",
             "# DOCUMENTATION of the map\n", "SNR: 5.0 1.0\n"]

    assert list(skip_docana(lines)) == ["FILTER: g\n",
                                        "# DOCUMENTATION of the map\n",
                                        "SNR: 5.0 1.0\n"]


def test_pipeline_eff_file_ignores_data_keys_inside_the_docana_block(tmp_path):
    """A DOCANA body may quote the keys it documents; they are not data."""
    from jax_snlc_sim.detection import read_pipeline_eff_file

    f = tmp_path / "SEARCHEFF_PIPELINE_DOC.DAT"
    f.write_text("DOCUMENTATION:\n"
                 "  PURPOSE: detection efficiency\n"
                 "  FILTER: r\n"
                 "  SNR: 99.0 0.5\n"
                 "DOCUMENTATION_END:\n"
                 "FILTER: g\nSNR: 0.0 0.0\nSNR: 5.0 1.0\n")

    eff = read_pipeline_eff_file(str(f), 0, 0)

    assert [c.filterlist for c in eff.curves] == ["g"]
    np.testing.assert_array_equal(eff.curves[0].vals, [0.0, 5.0])
    np.testing.assert_array_equal(eff.curves[0].effs, [0.0, 1.0])


def test_searcheff_file_found_under_path_user_input(tmp_path,
                                                    searcheff_data_dir):
    """A bare file name not found as given is looked for in each
    PATH_USER_INPUT directory, plain or gzipped (snana_openTextFile); a file
    present both ways aborts (open_TEXTgz)."""
    name = "SEARCHEFF_PIPELINE_LSST.DAT"
    plain_dir, gz_dir = tmp_path / "plain", tmp_path / "gz"
    plain_dir.mkdir()
    gz_dir.mkdir()
    shutil.copy(searcheff_data_dir / name, plain_dir / name)
    with open(searcheff_data_dir / name, 'rb') as fin, \
            gzip.open(gz_dir / f"{name}.gz", 'wb') as fout:
        shutil.copyfileobj(fin, fout)

    assert resolve_searcheff_file(
        name, 'SEARCHEFF_PIPELINE_EFF_FILE',
        f"{tmp_path / 'nowhere'} {plain_dir}") == plain_dir / name
    assert resolve_searcheff_file(
        name, 'SEARCHEFF_PIPELINE_EFF_FILE', str(gz_dir)) == gz_dir / f"{name}.gz"
    with pytest.raises(SNANAError, match="SEARCHEFF_PIPELINE_EFF_FILE"):
        resolve_searcheff_file("NO_SUCH_FILE.DAT",
                               'SEARCHEFF_PIPELINE_EFF_FILE', str(plain_dir))

    shutil.copy(gz_dir / f"{name}.gz", plain_dir / f"{name}.gz")
    with pytest.raises(SNANAError, match="found both"):
        resolve_searcheff_file(name, 'SEARCHEFF_PIPELINE_EFF_FILE',
                               str(plain_dir))


def test_pipeline_logic_default_unset_and_docana(tmp_path):
    """DEFAULT names SEARCHEFF_PIPELINE_LOGIC.DAT, found through the search
    path; an unset logic key is SNANA's NONE and aborts once the pipeline
    file holds curves; the file needs a DOCUMENTATION block
    (init_SEARCHEFF_LOGIC)."""
    logic = tmp_path / "SEARCHEFF_PIPELINE_LOGIC.DAT"
    logic.write_text(_DOCANA + "LSST:  2  ugrizY\n")
    assert read_pipeline_logic(
        'DEFAULT', 'LSST', str(tmp_path)) == (2, 'ugrizY')
    for unset in (None, 'NONE'):
        with pytest.raises(SNANAError,
                           match="need a SEARCHEFF_PIPELINE_LOGIC_FILE"):
            read_pipeline_logic(unset, 'LSST', str(tmp_path))

    logic.write_text("LSST:  2  ugrizY\n")
    with pytest.raises(SNANAError, match="DOCUMENTATION"):
        read_pipeline_logic('DEFAULT', 'LSST', str(tmp_path))


def test_maps_past_the_limit_are_passed_over(tmp_path):
    """A file with more maps than MXMAP_SEARCHEFF_MAP keeps the first ones
    and passes over the rest (read_searcheff_map, sntools_trigger.c:1373)."""
    one = ("VARNAMES: REDSHIFT HOSTEFF\n"
           "HOSTEFF: 0.0 1.0\n"
           "HOSTEFF: 1.0 1.0\n\n")
    path = _map_file(tmp_path, one * (MXMAP_SEARCHEFF_MAP + 2))
    maps, flag = read_searcheff_maps(path, "zHOST", "")
    assert len(maps) == MXMAP_SEARCHEFF_MAP
    assert flag == FLAG_NORMAL
