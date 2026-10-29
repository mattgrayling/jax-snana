"""``apply_fluxcal_fudges``: SNANA EXEC_FUDGE_FLUXCAL (snana.F90:18700-18948),
and ``flux_baselines``, the jax key FLUXCAL_BASELINE_CORRECT."""

import numpy as np
import pytest

from jax_snlc_core.noise import apply_fluxcal_fudges
from jax_snlc_fit.calibration import flux_baselines

_NEA = 9.0
_ARGS = dict(fluxcal=np.array([100.0]), fluxcal_err=np.array([4.0]),
             offset=np.array([0.0]), err_scale=np.array([1.0]),
             mag_err=np.array([0.0]), nea=np.array([_NEA]))


def _add_var(err_add, err_pix):
    _, _, add_var = apply_fluxcal_fudges(err_add=np.array([err_add]),
                                         err_pix=np.array([err_pix]), **_ARGS)
    return float(add_var[0])


def test_the_offset_and_the_error_scale_are_applied():
    flux, err1, _ = apply_fluxcal_fudges(
        fluxcal=np.array([100.0]), fluxcal_err=np.array([4.0]),
        offset=np.array([2.0]), err_scale=np.array([1.5]),
        err_add=np.array([0.0]), err_pix=np.array([0.0]),
        mag_err=np.array([0.0]), nea=np.array([_NEA]))
    assert (float(flux[0]), float(err1[0])) == (102.0, 6.0)


def test_a_negative_additive_fudge_subtracts_in_quadrature():
    # SQERR2a = -SQERR2a when ERR2a < 0 (snana.F90:18805)
    assert _add_var(0.5, 0.0) == 0.25
    assert _add_var(-0.5, 0.0) == -0.25


def test_pixel_error_uses_noise_equivalent_aperture():
    # SQERR2b * AREA with AREA = NEA (snana.F90:18874, :18878)
    assert _add_var(0.0, 0.5) == 0.25 * _NEA


def test_a_negative_per_pixel_fudge_contributes_nothing():
    # AREA stays 0 unless ERR2b > 0 (snana.F90:18865-18866), so an unphysical
    # negative FUDGE_FLUXCAL_ERRPIX adds no variance rather than a positive one
    assert _add_var(0.0, -0.5) == 0.0
    assert _add_var(0.0, 0.0) == 0.0


def test_the_magnitude_fudge_scales_with_the_offset_flux():
    _, _, add_var = apply_fluxcal_fudges(
        fluxcal=np.array([100.0]), fluxcal_err=np.array([4.0]),
        offset=np.array([2.0]), err_scale=np.array([1.0]),
        err_add=np.array([0.0]), err_pix=np.array([0.0]),
        mag_err=np.array([0.01]), nea=np.array([_NEA]))
    assert float(add_var[0]) == (0.01 * 102.0) ** 2


def test_grouped_flux_baselines():
    """Floors of 0, -300 and -600 under errors of 20 to 80. A floor of 0 has
    too few epochs below -3 sigma and is left at zero; the others take the
    median of those epochs, checked here one group at a time."""
    rng = np.random.default_rng(3)
    n_groups = 60
    group = rng.integers(0, n_groups, 3000)
    err = rng.uniform(20.0, 80.0, group.size)
    flux = rng.normal(-300.0 * (group % 3), err)
    base, base_err = flux_baselines(flux, err, group, n_groups)
    measured = 0
    for k in range(n_groups):
        s = (group == k) & (flux < -3.0 * err)
        if s.sum() >= 4:
            measured += 1
            assert base[k] == pytest.approx(np.median(flux[s]), rel=1e-15)
            assert base_err[k] == pytest.approx(
                1.2533 / np.sqrt(np.sum(err[s] ** -2)), rel=1e-12)
        else:
            assert base[k] == base_err[k] == 0.0
    assert 0 < measured < n_groups


def test_a_lone_extreme_epoch_barely_moves_the_baseline():
    flux = np.array([-2100.0, -1950.0, -2050.0, -145000.0])
    base, _ = flux_baselines(flux, np.full(4, 100.0), np.zeros(4, int), 1)
    assert base[0] == -2075.0
