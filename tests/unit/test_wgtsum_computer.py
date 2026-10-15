"""Unit tests for ``compute_wgtsum`` end-to-end.

The interpolation primitives (``interpolate_wgtmap``,
``_interpolate_weights_with_snvar``) are covered by
``test_wgtsum_interp.py``; this file covers the public
``compute_wgtsum`` orchestration which dispatches to the no-snvars or
with-snvars path based on the WGTMAP header.
"""

import numpy as np
import pytest


from jax_snlc_core.io.wgtmap_reader import read_wgtmap
from jax_snlc_core.models import MODEL_REGISTRY
from jax_snlc_sim.hostlib.hostlib_selection import wgtmap_snvar_names
from jax_snlc_sim.population import declared_population, population_spellings
from jax_snlc_sim.hostlib.hostlib_reader import hostlib_declared, read_hostlib
from jax_snlc_core.utils import SNANAError
from jax_snlc_sim.hostlib.wgtsum_computer import (
    compute_wgtsum,
    WgtSumData,
)

# what a SALT run declares: its population under each spelling, plus the
# host dust every model draws
SNVARS = wgtmap_snvar_names(population_spellings(
    *declared_population(MODEL_REGISTRY["salt"], {})[:2]))


@pytest.fixture
def hostlib_data(jax_test_hostlib_path):
    """Parsed JAX_TEST.HOSTLIB (the bundled toy galaxy catalogue)."""
    path = str(jax_test_hostlib_path)
    return read_hostlib(path, hostlib_declared(path), use_cache=False)


@pytest.fixture
def wgtmap_dir(data_root):
    return data_root / "wgtmap"


def _read_wgtmap(wgtmap_dir, name: str):
    return read_wgtmap(str(wgtmap_dir / name), SNVARS)


def test_compute_wgtsum_with_snvar_dispatches_to_multi_bin_path(
    hostlib_data, wgtmap_dir
):
    """An (x1, LOGMASS) WGTMAP — x1 is an SN var — uses the with-snvars
    code path and emits a (n_snvar_bins, n_galaxies) cumulative sum."""
    wgtmap = _read_wgtmap(wgtmap_dir, "JAX_TEST_X1_LOGMASS.WGTMAP")
    assert wgtmap.snvar_indices, "fixture should have x1 as SN var"

    result = compute_wgtsum(hostlib_data, wgtmap)
    assert isinstance(result, WgtSumData)

    cumsum = np.asarray(result.wgtsum)
    assert cumsum.ndim == 2, f"with-snvars wgtsum should be 2D, got {cumsum.shape}"
    n_snvar_bins, n_gal = cumsum.shape
    assert n_gal == hostlib_data.ngal
    # Cumulative sums are monotonically nondecreasing within each bin.
    assert np.all(np.diff(cumsum, axis=1) >= -1e-6), (
        "wgtsum cumulative sum should be monotonically nondecreasing"
    )
    # Final value per bin is the total weight for that bin (positive).
    assert np.all(cumsum[:, -1] > 0), "every snvar bin must have positive total"


def test_compute_wgtsum_no_snvars_dispatches_to_single_global_path(
    hostlib_data, tmp_path
):
    """A WGTMAP with only HOSTLIB variables (no x1/c) takes the
    no-snvars path and emits a single 1D global cumulative sum."""
    wgtmap_path = tmp_path / "LOGMASS_ONLY.WGTMAP"
    wgtmap_path.write_text(
        "VARNAMES_WGTMAP: LOGMASS WGT SNMAGSHIFT\n"
        "WGT:  8.0  0.10  0.0\n"
        "WGT:  9.0  0.30  0.0\n"
        "WGT: 10.0  1.00  0.0\n"
        "WGT: 11.0  3.00  0.0\n"
        "WGT: 12.0  9.00  0.0\n"
    )
    wgtmap = read_wgtmap(str(wgtmap_path), SNVARS)
    assert not wgtmap.snvar_indices, "fixture WGTMAP shouldn't have SN vars"

    result = compute_wgtsum(hostlib_data, wgtmap)
    cumsum = np.asarray(result.wgtsum)
    # The no-snvars path emits shape (1, n_galaxies) — a single global
    # cumulative sum reshaped to 2D for downstream consistency with the
    # multi-bin path (see the output writer's equivalent in selector).
    assert cumsum.shape == (1, hostlib_data.ngal), f"got {cumsum.shape}"
    assert cumsum[0, -1] > 0.0
    assert np.all(np.diff(cumsum[0]) >= -1e-6)
    # Galaxy with the highest LOGMASS gets the largest weight increment.
    weights = np.asarray(result.weights)
    assert weights.shape == (1, hostlib_data.ngal)
    assert weights[0].max() > weights[0].min()


def _write_snvar_wgtmap(tmp_path, name="X1_LOGMASS.WGTMAP"):
    """x1 (SN var) x LOGMASS WGTMAP whose LOGMASS grid is narrower than the
    catalogue, so some galaxies are out of range."""
    path = tmp_path / name
    rows = []
    for x1 in (-2.0, 0.0, 2.0):
        for lm in (9.5, 10.0, 10.5):
            rows.append(f"WGT: {x1} {lm} 1.0 0.0")
    path.write_text(
        "VARNAMES_WGTMAP: x1 LOGMASS WGT SNMAGSHIFT\n" + "\n".join(rows) + "\n")
    return path


def test_snvar_wgtsum_enforces_opt_extrap_bounds(hostlib_data, tmp_path):
    """The SNVAR wgtsum path aborts on out-of-range galaxy properties with
    OPT_EXTRAP=0, exactly as the no-SNVAR path: interp_GRIDMAP failure is
    fatal for every SNVAR bin (sntools_host.c:4330-4352)."""
    lm = np.asarray(hostlib_data.columns["LOGMASS"])
    assert (lm.min() < 9.5) or (lm.max() > 10.5), "fixture must have OOB rows"
    path = _write_snvar_wgtmap(tmp_path)

    wgtmap0 = read_wgtmap(str(path), SNVARS, opt_extrap=0)
    assert wgtmap0.snvar_indices, "x1 must be detected as an SN variable"
    with pytest.raises(SNANAError, match="LOGMASS"):
        compute_wgtsum(hostlib_data, wgtmap0)

    wgtmap1 = read_wgtmap(str(path), SNVARS, opt_extrap=1)
    result = compute_wgtsum(hostlib_data, wgtmap1)
    assert np.all(np.asarray(result.wgtsum)[:, -1] > 0)


def test_magobs_99_substituted_with_valmax(hostlib_data, tmp_path):
    """A [band]_obs value of MAG_ZEROFLUX=99 is replaced by VALMAX-0.001
    before interpolation and the bounds check (sntools_host.c:4316-4322),
    so it neither aborts nor keeps the raw 99."""
    r_obs = np.array(hostlib_data.columns["r_obs"], copy=True)
    lo, hi = float(r_obs.min()), 30.0
    r_obs[0] = 99.0
    hostlib_data.columns["r_obs"] = r_obs
    path = tmp_path / "ROBS.WGTMAP"
    path.write_text(
        "VARNAMES_WGTMAP: r_obs WGT SNMAGSHIFT\n"
        f"WGT: {lo - 0.1} 1.0 0.0\n"
        f"WGT: {hi} 3.0 0.0\n")
    wgtmap = read_wgtmap(str(path), SNVARS, opt_extrap=0)
    result = compute_wgtsum(hostlib_data, wgtmap)
    weights = np.asarray(result.weights)[0]
    # The engine evaluated galaxy 0 at hi - 0.001, not at 99
    expected = np.interp(hi - 0.001, [lo - 0.1, hi], [1.0, 3.0])
    np.testing.assert_allclose(weights[0], expected, rtol=1e-5)
