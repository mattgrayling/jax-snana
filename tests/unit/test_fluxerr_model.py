"""Unit tests for jax_snlc_core/noise.py.

Covers:
- File parser on a synthetic 1-D / 2-D / ERRADD file (and the real DES file
  if present, otherwise skipped).
- Uniform VARNAMES + uniform grid_axes validators.
- parse_redcov_string in all its forms.
- The map interpolation in one, two and three dimensions against numpy
  and hand-computed references.
- Edge clipping of out-of-range queries.
- REDCOV factor-decomposition statistical tests.
"""

import os
import tempfile
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_snlc_core.logging import SNANAError
from jax_snlc_core.noise import (ALL_GROUP, check_redcov_entries, evaluate_fluxerrmodel_data,
                                 interpolate_fluxerr_maps, parse_redcov_string,
                                 read_fluxerr_model_file)
from jax_snlc_sim.photometry import _compile_fluxerr_model
from jax_snlc_sim.hostlib.sersic import APERTURE_PSFSIG
from jax_snlc_sim.photometry import (_build_fluxerr_query_per_obs,
                                           _draw_redcov_fudge_factored,
                                           _gather_galmag_nea_per_obs, _interp_galmag)


def _interpolate(grid, query, origins, steps):
    """One map's values at ``query``, whose last axis is the coordinates."""
    return interpolate_fluxerr_maps(jnp.asarray(grid)[None],
                                    jnp.zeros(query.shape[:-1], dtype=jnp.int32),
                                    query, origins, steps)


def _write(text):
    fd, path = tempfile.mkstemp(suffix=".DAT", prefix="fluxerr_test_")
    os.close(fd)
    with open(path, "w") as f:
        f.write(text)
    return path


# ============================================================
# Reader tests
# ============================================================


@pytest.mark.parametrize("names", [("SKYSIG", "ZP"), ("ZP", "SKYSIG")])
def test_map_axis_order(tmp_path, names):
    axes = {"SKYSIG": (4., 6.), "ZP": (28., 32.)}
    rows = []
    for a in axes[names[0]]:
        for b in axes[names[1]]:
            values = dict(zip(names, (a, b)))
            scale = values["SKYSIG"] + values["ZP"] / 10.
            rows.append(f"ROW: {a} {b} {scale}")
    path = tmp_path / "errors.dat"
    path.write_text("MAPNAME: FLUXERR_SCALE\nBAND: g\nVARNAMES: "
                    + " ".join(names) + " ERRSCALE\n"
                    + "\n".join(rows) + "\nENDMAP:\n")
    model = read_fluxerr_model_file(str(path)).maps[0]
    obs = {"filter_indices": jnp.array([[0]]),
           "skysigs": jnp.array([[5.]]), "zptavgs": jnp.array([[30.]])}
    query = _build_fluxerr_query_per_obs(
        SimpleNamespace(fluxerr_active_vars=model.varnames[:-1]),
        27.5, {}, obs, None, None, None)
    result = _interpolate(
        model.grid_values, query,
        origins=tuple(axis[0] for axis in model.grid_axes),
        steps=tuple(axis[1] - axis[0] for axis in model.grid_axes))
    np.testing.assert_allclose(result, [[8.]])


def test_read_synthetic_2d():
    """Parse a synthetic 2-D PSF×SBMAG file."""
    text = """
DOCUMENTATION:
  PURPOSE: 2-D test
DOCUMENTATION_END:

DEFINE_FIELDGROUP: ALLF  F1+F2

MAPNAME: FLUXERR_SCALE
BAND: g
FIELD: ALLF
VARNAMES: PSF SBMAG ERRSCALE
ROW: 0.8  20.0  1.5
ROW: 0.8  21.0  1.4
ROW: 0.8  22.0  1.3
ROW: 1.2  20.0  1.6
ROW: 1.2  21.0  1.5
ROW: 1.2  22.0  1.4
ENDMAP:
"""
    path = _write(text)
    try:
        data = read_fluxerr_model_file(path)
        assert len(data.maps) == 1
        m = data.maps[0]
        assert m.varnames == ["PSF", "SBMAG", "ERRSCALE"]
        assert m.grid_values.shape == (2, 3)
        assert np.allclose(m.grid_axes[0], [0.8, 1.2])
        assert np.allclose(m.grid_axes[1], [20.0, 21.0, 22.0])
        assert np.isclose(m.grid_values[0, 0], 1.5)
        assert np.isclose(m.grid_values[1, 2], 1.4)
    finally:
        os.unlink(path)


def test_read_erradd():
    """Parse a synthetic 1-D ERRADD file."""
    text = """
MAPNAME: FLUXERR_OFFSET
BAND: r
VARNAMES: SBMAG ERRADD
ROW: 20.0  0.5
ROW: 21.0  0.7
ROW: 22.0  1.0
ENDMAP:
"""
    path = _write(text)
    try:
        data = read_fluxerr_model_file(path)
        assert len(data.maps) == 1
        m = data.maps[0]
        assert not m.is_errscale
        assert m.varnames == ["SBMAG", "ERRADD"]
        assert np.allclose(m.grid_values, [0.5, 0.7, 1.0])
    finally:
        os.unlink(path)


def test_validator_nonuniform_grid_spacing():
    """A map with non-uniform axis spacing must crash with ValueError."""
    text = """
MAPNAME: FLUXERR_SCALE
BAND: g
VARNAMES: SBMAG ERRSCALE
ROW: 20.0  1.5
ROW: 21.0  1.3
ROW: 25.0  1.2
ENDMAP:
"""
    path = _write(text)
    try:
        with pytest.raises(SNANAError, match="non-uniform grid spacing"):
            read_fluxerr_model_file(path)
    finally:
        os.unlink(path)


# ============================================================
# REDCOV string parsing
# ============================================================


def test_parse_redcov_string():
    # Form 1: REDCOV(GROUP): bands
    e = parse_redcov_string("REDCOV(SHALLOW): g:0.6,r:0.6,i:0.6,z:0.6")
    assert len(e) == 4
    assert all(x.field_group == "SHALLOW" for x in e)
    assert [x.bands for x in e] == ["g", "r", "i", "z"]
    assert all(x.redcov == 0.6 for x in e)

    # Form 2: REDCOV: bands (no group → ALL)
    e = parse_redcov_string("REDCOV: g:0.4")
    assert len(e) == 1 and e[0].field_group == ALL_GROUP
    assert e[0].redcov == 0.4

    # Form 3: bare bands string (no keyword)
    e = parse_redcov_string("griz:0.5")
    assert len(e) == 1
    assert e[0].bands == "griz" and e[0].redcov == 0.5

    # Form 4: invalid value (>= 1.0) aborts
    with pytest.raises(SNANAError):
        parse_redcov_string("g:1.5")

    # Form 5: missing colon
    with pytest.raises(SNANAError):
        parse_redcov_string("g 0.5")


# ============================================================
# Interpolator tests
# ============================================================


def test_fluxerr_map_interpolation_1d_matches_numpy():
    """One dimension matches np.interp in range and holds to the ends."""
    grid = np.array(
        [2.967, 2.967, 1.738, 1.324, 1.257, 1.219, 1.174, 1.174], dtype=np.float32
    )
    axis = np.arange(20.5, 28.5, 1.0, dtype=np.float32)
    queries = jnp.array([[20.5, 21.0, 22.5, 23.7, 27.5, 19.0, 30.0]])

    got = np.asarray(jax.jit(lambda q: _interpolate(grid, q, (20.5,), (1.0,)))(
        queries[..., None]))[0]
    expected = np.array(
        [np.interp(np.clip(float(q), 20.5, 27.5), axis, grid) for q in queries[0]],
        dtype=np.float32,
    )
    assert np.allclose(got, expected, atol=1e-6)


def test_fluxerr_map_interpolation_2d_is_bilinear():
    """Bilinear interpolation against hand-computed values."""
    grid_2d = np.array([[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]], dtype=np.float32)
    queries = jnp.array([[[0.0, 0.0], [10.0, 2.0], [5.0, 1.0], [5.0, 0.5]]])

    got = np.asarray(jax.jit(lambda q: _interpolate(grid_2d, q, (0.0, 0.0), (10.0, 1.0)))(
        queries))[0]
    assert np.allclose(got, [1.0, 7.0, 4.0, 3.5], atol=1e-5)


def test_fluxerr_map_interpolation_3d_is_trilinear():
    """Trilinear interpolation at the cell midpoint is the mean of the corners."""
    grid_3d = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    queries = jnp.array([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]]])

    got = np.asarray(jax.jit(lambda q: _interpolate(grid_3d, q, (0.0,) * 3, (1.0,) * 3))(
        queries))[0]
    assert np.allclose(got, [0.0, 7.0, 3.5], atol=1e-5)


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
def test_fluxerr_map_selection_with_singleton_axis(dtype):
    """Each epoch uses its own map, with clipping and a constant single-knot axis."""
    grids = jnp.array([[[1., 3., 5.]], [[101., 103., 105.]]], dtype=jnp.float32)
    rows = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
    query = jnp.array([[[9., 19.], [10., 21.]],
                       [[10., 24.], [11., 25.]]], dtype=dtype)
    actual = jax.jit(lambda g, r, q: interpolate_fluxerr_maps(
        g, r, q, (10., 20.), (1., 2.)))(grids, rows, query)
    np.testing.assert_allclose(actual, [[1., 102.], [105., 5.]])
    assert actual.dtype == dtype


def test_interp_oor_clips_to_edge():
    """A query outside the grid takes the grid's edge value in every dimension."""
    got1 = _interpolate(np.array([1.0, 2.0, 3.0]), jnp.array([[[-100.0], [100.0]]]),
                        (0.0,), (1.0,))
    assert np.allclose(got1[0], [1.0, 3.0])
    got2 = _interpolate(np.array([[1.0, 2.0], [3.0, 4.0]]),
                        jnp.array([[[-100.0, -100.0], [100.0, 100.0]]]), (0.0, 0.0), (1.0, 1.0))
    assert np.allclose(got2[0], [1.0, 4.0])
    got3 = _interpolate(np.arange(8.0).reshape(2, 2, 2),
                        jnp.array([[[-100.0] * 3, [100.0] * 3]]), (0.0,) * 3, (1.0,) * 3)
    assert np.allclose(got3[0], [0.0, 7.0])


# ============================================================
# REDCOV factor-decomposition statistical tests
# ============================================================


def test_redcov_factor_single_group():
    """Single icov group: all obs in one band-set should be correlated at REDCOV."""
    rho = 0.6
    n_obs = 4
    n_factors = 1

    F_per_obs = jnp.full((n_obs, n_factors), float(np.sqrt(rho)))

    n_samples = 30000
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, n_samples)

    def one_sample(k):
        kw, kz = jax.random.split(k)
        w = jax.random.normal(kw, shape=(n_factors,))
        z = jax.random.normal(kz, shape=(n_obs,))
        return _draw_redcov_fudge_factored(F_per_obs, w, z)

    samples = jax.vmap(one_sample)(keys)
    cov = np.cov(np.asarray(samples), rowvar=False)
    off = cov[~np.eye(n_obs, dtype=bool)]
    assert abs(np.diag(cov).mean() - 1.0) < 0.05
    assert abs(off.mean() - rho) < 0.03


def test_redcov_factor_two_groups():
    """Two disjoint icov groups: within-group correlated, cross-group uncorrelated."""
    rho_a, rho_b = 0.6, 0.3
    n_obs = 4
    n_factors = 2

    F_per_obs = jnp.array(
        [
            [float(np.sqrt(rho_a)), 0.0],
            [float(np.sqrt(rho_a)), 0.0],
            [0.0, float(np.sqrt(rho_b))],
            [0.0, float(np.sqrt(rho_b))],
        ]
    )

    n_samples = 50000
    key = jax.random.PRNGKey(1)
    keys = jax.random.split(key, n_samples)

    def one_sample(k):
        kw, kz = jax.random.split(k)
        w = jax.random.normal(kw, shape=(n_factors,))
        z = jax.random.normal(kz, shape=(n_obs,))
        return _draw_redcov_fudge_factored(F_per_obs, w, z)

    samples = np.asarray(jax.vmap(one_sample)(keys))
    cov = np.cov(samples, rowvar=False)
    cross = np.array([cov[0, 2], cov[0, 3], cov[1, 2], cov[1, 3]])
    assert np.all(np.abs(np.diag(cov) - 1.0) < 0.05)
    assert abs(cov[0, 1] - rho_a) < 0.03
    assert abs(cov[2, 3] - rho_b) < 0.03
    assert np.all(np.abs(cross) < 0.03)


def test_redcov_factor_below_snr_iid_fallback():
    """Below-SNR obs (F_per_obs row = 0) still get an iid unit draw z (matches SNANA)."""
    n_obs = 5
    n_factors = 1

    F_per_obs = jnp.full((n_obs, n_factors), float(np.sqrt(0.5)))
    F_per_obs = F_per_obs.at[1].set(0.0).at[3].set(0.0)

    key = jax.random.PRNGKey(2)
    kw, kz = jax.random.split(key)
    w = jax.random.normal(kw, shape=(n_factors,))
    z = jax.random.normal(kz, shape=(n_obs,))
    delta = np.asarray(_draw_redcov_fudge_factored(F_per_obs, w, z))
    z_np = np.asarray(z)
    assert np.isclose(delta[1], z_np[1])
    assert np.isclose(delta[3], z_np[3])
    assert not np.isclose(delta[0], z_np[0])


# ============================================================
# GALMAG aperture interpolation (FLUXERRMODEL GALMAG variable)
# ============================================================


def test_interp_galmag_nonuniform_grid_matches_numpy():
    """_interp_galmag interpolates on the true non-uniform APERTURE_PSFSIG
    grid (interp_GALMAG_HOSTLIB, sntools_host.c:4922-4962); a uniform-spacing
    assumption fails on this grid."""
    rng = np.random.default_rng(7)
    table = rng.uniform(18.0, 26.0, size=(3, 4, 10))
    grid = np.asarray(APERTURE_PSFSIG[1:], dtype=float)
    psf = rng.uniform(grid[0], grid[-1], size=(3, 4))
    got = np.asarray(_interp_galmag(jnp.asarray(table), jnp.asarray(psf)))
    psf_c = np.clip(psf, grid[0] + 1e-4, grid[-1] - 1e-4)
    for i in range(3):
        for j in range(4):
            expected = np.interp(psf_c[i, j], grid, table[i, j, 1:])
            assert got[i, j] == pytest.approx(expected, rel=1e-5)


def test_gather_galmag_nea_uses_pixsize():
    """The GALMAG query PSF is pixsize * sqrt(NEA/(4π))
    (gen_fluxNoise_calc, snlc_sim.c:27794), so a different pixel scale must
    move the interpolated GALMAG."""
    rng = np.random.default_rng(3)
    # strong aperture dependence so a PSF change is visible
    table = jnp.asarray(
        np.cumsum(rng.uniform(0.5, 1.0, size=(2, 3, 10)), axis=-1) + 18.0)
    obs = {
        'filter_indices': jnp.zeros((2, 5), dtype=jnp.int32),
        'neas': jnp.full((2, 5), 12.0),
    }
    small = np.asarray(_gather_galmag_nea_per_obs(
        table, obs, jnp.full(2, 0.1)))
    large = np.asarray(_gather_galmag_nea_per_obs(
        table, obs, jnp.full(2, 0.4)))
    assert np.all(np.abs(small - large) > 1e-6)


def test_galmag_without_a_host_is_snanas_undefined_magnitude():
    """With no host, every epoch's GALMAG is the value SNANA initialises it
    to, MAG_UNDEFINED = 666 (sntools_host.c:6559, sntools.h:126), not the
    99 of zero flux."""
    obs = {'filter_indices': jnp.zeros((2, 5), dtype=jnp.int32),
           'neas': jnp.full((2, 5), 12.0)}
    galmag = np.asarray(_gather_galmag_nea_per_obs(None, obs, jnp.full(2, 0.1)))
    np.testing.assert_array_equal(galmag, 666.0)


def test_simulation_chooses_maps_by_snana_substring_rule(tmp_path):
    """The simulation serves each SIMLIB field and band with the map and the
    REDCOV entry SNANA chooses (INDEX_MAP_FLUXERRMODEL and
    INDEX_REDCOV_FLUXERRMODEL, sntools_fluxErrModels.c:877-968): BAND and
    FIELD match as substrings of the band list and of the group's field
    list, so field S takes the SHALLOW (S1+S2) map and REDCOV, and a pair
    nothing serves takes the inert map row and no shared factor."""
    path = tmp_path / "fluxerr.dat"
    path.write_text("DEFINE_FIELDGROUP: SHALLOW  S1+S2\n"
                    "REDCOV(SHALLOW): g:0.64\n"
                    "MAPNAME: A\nBAND: g\nFIELD: SHALLOW\nVARNAMES: PSF ERRSCALE\n"
                    "ROW: 1.0 1.5\nROW: 2.0 1.5\nENDMAP:\n"
                    "MAPNAME: B\nBAND: r\nFIELD: ALL\nVARNAMES: PSF ERRSCALE\n"
                    "ROW: 1.0 2.0\nROW: 2.0 2.0\nENDMAP:\n")
    data = read_fluxerr_model_file(str(path))
    simlib = SimpleNamespace(fields=[SimpleNamespace(field_name=name)
                                     for name in ("S1", "X3", "S")])
    kcor = SimpleNamespace(get_filter_names=lambda: ["LSST-g", "LSST-r"])
    compiled = _compile_fluxerr_model(data, simlib, kcor)
    np.testing.assert_array_equal(compiled['fluxerr_map_index_per_field'],
                                  [[0, 1], [2, 1], [0, 1]])
    np.testing.assert_allclose(compiled['fluxerr_redcov_factors'][:, :, 0],
                               [[0.8, 0.0], [0.0, 0.0], [0.8, 0.0]])


def test_redcov_groups_sharing_a_field_are_recorded_not_fatal_at_setup(tmp_path):
    """Two field groups may share a SIMLIB field: SNANA loads their REDCOV
    entries, and its NMATCH>1 abort fires only for an epoch in the shared
    field and band (snlc_sim.c:28090). Building the simulation records that
    pair and aborts nothing; a pair one entry matches takes its factor."""
    path = tmp_path / "fluxerr.dat"
    path.write_text("DEFINE_FIELDGROUP: DEEP  C3+X3\nDEFINE_FIELDGROUP: WIDE  X3+S1\n"
                    "REDCOV(DEEP): g:0.64\nREDCOV(WIDE): g:0.25\n"
                    "MAPNAME: A\nBAND: g\nFIELD: ALL\nVARNAMES: PSF ERRSCALE\n"
                    "ROW: 1.0 1.5\nROW: 2.0 1.5\nENDMAP:\n")
    data = read_fluxerr_model_file(str(path))
    simlib = SimpleNamespace(fields=[SimpleNamespace(field_name=name)
                                     for name in ("C3", "X3", "S1")])
    kcor = SimpleNamespace(get_filter_names=lambda: ["LSST-g", "LSST-r"])
    compiled = _compile_fluxerr_model(data, simlib, kcor)
    np.testing.assert_array_equal(compiled['fluxerr_redcov_counts'], [[1, 0], [2, 0], [1, 0]])
    np.testing.assert_allclose(compiled['fluxerr_redcov_factors'][:, 0],
                               [[0.8, 0.0], [0.0, 0.0], [0.0, 0.5]])
    np.testing.assert_allclose(compiled['fluxerr_redcov_factors'][:, 1], 0.0)


@pytest.mark.parametrize("lines", [
    "REDCOV: g:0.3\nREDCOV: gr:0.2\n",
    "REDCOV: g:0.3\nREDCOV(DEEP): g:0.2\n",
    "REDCOV(SHALLOW): g:0.3\nREDCOV: g:0.2\n",
])
def test_redcov_band_defined_twice_for_a_group_aborts(tmp_path, lines):
    """SNANA checks each REDCOV entry's bands against the entries before it
    as it loads them (load_REDCOV_FLUXERRMODEL, sntools_fluxErrModels.c:
    556-566, OPT_FIELD=1): an earlier entry for ALL fields matches any group,
    and an earlier group's name matches a group name it contains, so "ALL"
    after "SHALLOW" is a repeat too. Reading the file does not check them:
    the fit never loads REDCOV (snana.F90:18592 passes NONE)."""
    path = tmp_path / "fluxerr.dat"
    path.write_text("DEFINE_FIELDGROUP: DEEP  C3\nDEFINE_FIELDGROUP: SHALLOW  S1\n" + lines)
    data = read_fluxerr_model_file(str(path))
    with pytest.raises(SNANAError, match="Cannot define g-"):
        check_redcov_entries(data.redcov)


def test_fit_fluxerrmodel_file_without_maps_keeps_every_error(tmp_path):
    """A FLUXERRMODEL_FILE holding only REDCOV and field groups gives
    NMAP_FLUXERRMODEL = 0, and get_FLUXERRMODEL then returns the error it
    was given (sntools_fluxErrModels.c:753-755)."""
    path = tmp_path / "redcov_only.dat"
    path.write_text("DEFINE_FIELDGROUP: SHALLOW  S1+S2\nREDCOV(SHALLOW): g:0.64\n")
    data = read_fluxerr_model_file(str(path))
    err = np.array([3.0, 4.5])
    np.testing.assert_array_equal(
        evaluate_fluxerrmodel_data(data, ["g", "r"], ["S1", "X3"], err,
                                   {"PSF": np.array([1.0, 2.0])}), err)


def test_fit_fluxerrmodel_applies_the_chosen_map(tmp_path):
    """Each epoch takes the map INDEX_MAP_FLUXERRMODEL chooses for its band
    and field, interpolated linearly in its VARNAMES, then the ERRADD rule
    of apply_FLUXERRMODEL (sntools_fluxErrModels.c:972-1002), sqrt(err^2 +
    value^2), times SCALE_FLUXERR_DATA; an epoch no map serves keeps its
    error."""
    path = tmp_path / "erradd.dat"
    path.write_text("MAPNAME: A\nBAND: g\nFIELD: ALL\nSCALE_FLUXERR_DATA: 2.0\n"
                    "VARNAMES: PSF ERRADD\nROW: 1.0 0.0\nROW: 3.0 8.0\nENDMAP:\n")
    data = read_fluxerr_model_file(str(path))
    err = np.array([3.0, 3.0, 3.0])
    out = evaluate_fluxerrmodel_data(data, ["g", "g", "r"], ["X", "X", "X"], err,
                                     {"PSF": np.array([2.0, 5.0, 2.0])})
    np.testing.assert_allclose(out, [2.0 * 5.0, 2.0 * np.hypot(3.0, 8.0), 3.0])


@pytest.mark.parametrize('duplicate', [False, True])
def test_grid_positions_are_unique_and_complete(tmp_path, duplicate):
    """Unordered rows fill the grid; duplicate coordinates cannot hide a missing cell."""
    path = tmp_path / 'grid.dat'
    last = '0 1 4' if duplicate else '1 1 4'
    path.write_text('MAPNAME: FLUXERR_SCALE\nBAND: g\n'
                    'VARNAMES: PSF SKYSIG ERRSCALE\n'
                    'ROW: 1 0 3\nROW: 0 1 2\nROW: 0 0 1\n'
                    f'ROW: {last}\nENDMAP:\n')
    if duplicate:
        with pytest.raises(SNANAError, match='each grid position'):
            read_fluxerr_model_file(str(path))
    else:
        data = read_fluxerr_model_file(str(path))
        np.testing.assert_array_equal(data.maps[0].grid_values, [[1., 2.], [3., 4.]])


def test_map_overlap_is_checked_for_observed_band_and_field(tmp_path):
    """Substring field matching must reject ambiguity only for the queried pair."""
    path = tmp_path / 'overlap.dat'
    path.write_text(''.join(
        f'MAPNAME: FLUXERR_SCALE\nBAND: g\nFIELD: {field}\n'
        'VARNAMES: PSF ERRSCALE\nROW: 1 2\nROW: 2 2\nENDMAP:\n'
        for field in ['S1', 'S10']))
    model = read_fluxerr_model_file(str(path))
    np.testing.assert_allclose(evaluate_fluxerrmodel_data(
        model, ['g'], ['S10'], [3.], {'PSF': [1.5]}), [6.])
    with pytest.raises(SNANAError, match='NMATCH=2'):
        evaluate_fluxerrmodel_data(model, ['g'], ['S1'], [3.], {'PSF': [1.5]})


def test_field_group_expanding_to_all_serves_every_field(tmp_path):
    """ALL remains a wildcard after expansion of a named field group."""
    path = tmp_path / 'all_fields.dat'
    path.write_text('DEFINE_FIELDGROUP: EVERYWHERE ALL\n'
                    'MAPNAME: SCALE\nBAND: g\nFIELD: EVERYWHERE\n'
                    'VARNAMES: PSF ERRSCALE\nROW: 1 2\nROW: 2 2\nENDMAP:\n')
    model = read_fluxerr_model_file(str(path))
    np.testing.assert_allclose(evaluate_fluxerrmodel_data(
        model, ['g'], ['TOY_A'], [3.], {'PSF': [1.5]}), [6.])
