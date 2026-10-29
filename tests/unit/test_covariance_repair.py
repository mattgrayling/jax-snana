"""``fix_covariance``: SNANA's ``update_covMatrix`` (sntools.c:2485-2607)."""

import numpy as np
import pytest

from jax_snlc_core.utils import EIGMIN_COV, fix_covariance


def _stack(*mats):
    return np.array(mats, dtype=float)


def test_a_positive_definite_matrix_is_returned_untouched():
    cov = _stack([[4.0, 1.0], [1.0, 9.0]])
    np.testing.assert_array_equal(fix_covariance(cov), cov)


def test_a_non_positive_matrix_is_rebuilt_from_a_clipped_spectrum():
    cov = _stack([[1.0, 2.0], [2.0, 1.0]])          # eigenvalues 3 and -1
    out = fix_covariance(cov)
    w = np.linalg.eigvalsh(out[0])
    np.testing.assert_allclose(sorted(w), [EIGMIN_COV, 3.0], rtol=1e-9)


def test_force_rebuilds_a_matrix_that_is_already_positive_definite():
    cov = _stack([[1.0e-8, 0.0], [0.0, 4.0]])
    out = fix_covariance(cov, force=True)
    np.testing.assert_allclose(np.diag(out[0]), [EIGMIN_COV, 4.0], rtol=1e-12)


def test_an_undefined_element_is_zeroed_and_the_matrix_is_not_repaired():
    # COV_x1_c = -9 is the fitter's "not measured" sentinel; SNANA zeroes that
    # element and returns istat_cov = -9 before the eigen work, so the matrix
    # keeps its negative eigenvalue.
    cov = _stack([[1.0, -9.0], [-9.0, -1.0]])
    out = fix_covariance(cov)
    np.testing.assert_array_equal(out[0], [[1.0, 0.0], [0.0, -1.0]])


def test_the_sentinel_only_spares_the_matrix_that_carries_it():
    undefined = [[1.0, -9.0], [-9.0, -1.0]]
    indefinite = [[1.0, 2.0], [2.0, 1.0]]
    out = fix_covariance(_stack(undefined, indefinite))
    np.testing.assert_array_equal(out[0], [[1.0, 0.0], [0.0, -1.0]])
    assert np.linalg.eigvalsh(out[1]).min() > 0.0


def test_force_does_not_repair_a_matrix_with_an_undefined_element():
    cov = _stack([[1.0e-8, -9.0], [-9.0, 4.0]])
    out = fix_covariance(cov, force=True)
    np.testing.assert_array_equal(out[0], [[1.0e-8, 0.0], [0.0, 4.0]])


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("dimension", [1, 2, 3, 5])
def test_selective_reconstruction_matches_full_spectral_repair(dimension, force):
    rng = np.random.default_rng(dimension)
    a = rng.normal(size=(40, dimension, dimension))
    cov = a @ a.transpose(0, 2, 1)
    cov[::3] -= 2 * np.eye(dimension)
    cov[1] = 0.
    cov[2] = 1.e-12 * np.eye(dimension)
    cov[3, 0, 0] = -9.
    original = cov.copy()

    undefined = np.abs(cov + 9.) < 1.e-6
    clean = np.where(undefined, 0., cov)
    w, v = np.linalg.eigh(clean)
    reconstructed = v @ (np.clip(w, EIGMIN_COV, None)[:, :, None] * v.transpose(0, 2, 1))
    repair = np.full(len(cov), True) if force else w.min(axis=1) <= 0.
    expected = np.where((repair & ~undefined.any(axis=(1, 2)))[:, None, None],
                        reconstructed, clean)
    np.testing.assert_array_equal(fix_covariance(cov, force=force), expected)
    np.testing.assert_array_equal(cov, original)
