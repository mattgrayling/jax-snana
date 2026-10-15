"""Unit tests for jax_snlc_core/photometry/fluxerr_model.py.

Covers:
- File parser on a synthetic 1-D / 2-D / ERRADD file (and the real DES file
  if present, otherwise skipped).
- Uniform VARNAMES + uniform grid_axes validators.
- parse_redcov_string in all its forms.
- interp_1d / interp_2d / interp_3d against numpy/scipy reference.
- Edge clipping of out-of-range queries.
- REDCOV factor-decomposition statistical tests.
"""

from jax_snlc_core.utils.error_handling import SNANAError
import os
import tempfile
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_snlc_sim.hostlib.sersic_sampling import APERTURE_PSFSIG
from jax_snlc_sim.photometry.noise import (
    _build_fluxerr_query_per_obs, _draw_redcov_fudge_factored,
    _gather_galmag_nea_per_obs, _interp_galmag,
)
from jax_snlc_core.photometry.fluxerr_model import (
    ALL_GROUP,
    interp_1d,
    interp_2d,
    interp_3d,
    parse_redcov_string,
    read_fluxerr_model_file,
)


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
    result = interp_2d(
        jnp.asarray(model.grid_values)[None, None], query,
        origins=tuple(axis[0] for axis in model.grid_axes),
        steps=tuple(axis[1] - axis[0] for axis in model.grid_axes),
        n_grid=(2, 2))
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
        assert m.n_dims == 2
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


def test_interp_1d_vs_numpy():
    """interp_1d must match np.interp at all in-range points and clip OOR."""
    grid = np.array(
        [2.967, 2.967, 1.738, 1.324, 1.257, 1.219, 1.174, 1.174], dtype=np.float32
    )
    axis = np.arange(20.5, 28.5, 1.0, dtype=np.float32)

    grid_per_obs = jnp.broadcast_to(jnp.asarray(grid), (1, 7, 8))
    queries = jnp.array([[20.5, 21.0, 22.5, 23.7, 27.5, 19.0, 30.0]])
    query = queries[..., None]

    @jax.jit
    def go(g, q):
        return interp_1d(g, q, origin=20.5, step=1.0, n_grid=8)

    got = np.asarray(go(grid_per_obs, query))[0]
    expected = np.array(
        [np.interp(np.clip(float(q), 20.5, 27.5), axis, grid) for q in queries[0]],
        dtype=np.float32,
    )
    assert np.allclose(got, expected, atol=1e-6)


def test_interp_2d_vs_manual():
    """interp_2d bilinear vs hand-computed reference."""
    grid_2d = np.array([[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]], dtype=np.float32)
    grid_per_obs = jnp.broadcast_to(jnp.asarray(grid_2d), (1, 4, 2, 3))
    queries = jnp.array(
        [
            [
                [0.0, 0.0],
                [10.0, 2.0],
                [5.0, 1.0],
                [5.0, 0.5],
            ]
        ]
    )

    @jax.jit
    def go(g, q):
        return interp_2d(g, q, origins=(0.0, 0.0), steps=(10.0, 1.0), n_grid=(2, 3))

    got = np.asarray(go(grid_per_obs, queries))[0]
    expected = np.array([1.0, 7.0, 4.0, 3.5], dtype=np.float32)
    assert np.allclose(got, expected, atol=1e-5)


def test_interp_3d_vs_manual():
    """interp_3d trilinear at the cell midpoint = mean of the 8 corners."""
    n0, n1, n2 = 2, 2, 2
    grid_3d = np.arange(8, dtype=np.float32).reshape(n0, n1, n2)
    grid_per_obs = jnp.broadcast_to(jnp.asarray(grid_3d), (1, 3, n0, n1, n2))
    queries = jnp.array(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.5, 0.5, 0.5],
            ]
        ]
    )

    @jax.jit
    def go(g, q):
        return interp_3d(
            g, q, origins=(0.0, 0.0, 0.0), steps=(1.0, 1.0, 1.0), n_grid=(n0, n1, n2)
        )

    got = np.asarray(go(grid_per_obs, queries))[0]
    expected = np.array([0.0, 7.0, 3.5], dtype=np.float32)
    assert np.allclose(got, expected, atol=1e-5)


def test_interp_oor_clips_to_edge():
    """All three interpolators must clip out-of-range queries to grid corners."""
    grid_1d = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    g1 = jnp.broadcast_to(jnp.asarray(grid_1d), (1, 2, 3))
    q1 = jnp.array([[[-100.0], [+100.0]]])

    @jax.jit
    def go1(g, q):
        return interp_1d(g, q, origin=0.0, step=1.0, n_grid=3)

    got1 = np.asarray(go1(g1, q1))[0]
    assert np.allclose(got1, [1.0, 3.0])

    grid_2d = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    g2 = jnp.broadcast_to(jnp.asarray(grid_2d), (1, 2, 2, 2))
    q2 = jnp.array([[[-100.0, -100.0], [100.0, 100.0]]])

    @jax.jit
    def go2(g, q):
        return interp_2d(g, q, origins=(0.0, 0.0), steps=(1.0, 1.0), n_grid=(2, 2))

    got2 = np.asarray(go2(g2, q2))[0]
    assert np.allclose(got2, [1.0, 4.0])

    grid_3d = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    g3 = jnp.broadcast_to(jnp.asarray(grid_3d), (1, 2, 2, 2, 2))
    q3 = jnp.array([[[-100.0, -100.0, -100.0], [100.0, 100.0, 100.0]]])

    @jax.jit
    def go3(g, q):
        return interp_3d(
            g, q, origins=(0.0, 0.0, 0.0), steps=(1.0, 1.0, 1.0), n_grid=(2, 2, 2)
        )

    got3 = np.asarray(go3(g3, q3))[0]
    assert np.allclose(got3, [0.0, 7.0])


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
