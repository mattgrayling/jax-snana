"""The accelerated MAD search must preserve the reference trial-grid estimator."""
import numpy as np
import pytest

from jax_salt2mu.covint import sigint_muresid_list
from jax_snlc_core.utils.error_handling import SNANAError


def _reference_mad(x, covariance, weights):
    """Independent linear scan of the original SNANA trial grid."""
    weights = weights / weights.max()
    average = np.sum(weights * x) / weights.sum()
    average_cov = np.sum(weights * covariance) / weights.sum()
    scatter = 1.48 * np.median(np.abs(x))
    trial = np.sqrt(max(scatter**2 - average_cov, 0.)) + .5
    trials, widths = [], []
    while trial >= -.3:
        added = trial * abs(trial)
        total = covariance + added
        if added < 0:
            total = np.maximum(total, .0004)
        width = 1.48 * np.median(np.abs((x - average) / np.sqrt(total)))
        trials.append(trial)
        widths.append(width)
        if width > 1:
            if len(widths) == 1:
                if abs(width - 1) < .001:
                    return trial
                raise ValueError('unbracketed')
            return np.interp(1., widths, trials)
        trial -= .01
    return -.3


@pytest.mark.parametrize('n', [1, 2, 3, 4, 5, 6, 17, 128, 999, 1000])
@pytest.mark.parametrize('case', ['random', 'constant', 'zero', 'equal_variance'])
def test_mad_search_matches_linear_scan(n, case):
    rng = np.random.default_rng(n)
    x = rng.normal(.03, .2, n)
    covariance = rng.uniform(.001, .1, n)
    weights = rng.uniform(.1, 2., n)
    if case == 'constant':
        x[:] = .125
    elif case == 'zero':
        x[:] = 0.
    elif case == 'equal_variance':
        covariance[:] = .02
    expected = _reference_mad(x, covariance, weights)
    actual = sigint_muresid_list(x, covariance, weights, abort=False, use_mad=True)
    np.testing.assert_allclose(actual, expected, rtol=0., atol=2e-14)


@pytest.mark.parametrize('mad', [False, True])
def test_invalid_covariance_and_abort_policy(mad):
    x = np.zeros(10)
    with pytest.raises(SNANAError, match='Invalid MUCOV'):
        sigint_muresid_list(x, np.zeros(10), np.ones(10), use_mad=mad)
    with pytest.raises(SNANAError, match='STD < AVG_MUERR'):
        sigint_muresid_list(x, np.ones(10), np.ones(10), use_mad=mad)
    assert sigint_muresid_list(x, np.ones(10), np.ones(10),
                              abort=False, use_mad=mad) == -.3


def test_mad_floor_with_abort_enabled():
    # Uncentred scatter passes the initial check, but centred residuals vanish.
    with pytest.raises(SNANAError, match='sig trial <'):
        sigint_muresid_list(np.ones(10), np.full(10, .01), np.ones(10), use_mad=True)


@pytest.mark.parametrize('first_width', [1.0005, 1.1])
def test_mad_first_trial_crossing(first_width):
    x = np.array([0., 0., 0., 0., 5 * np.sqrt(.26) * first_width / 1.48])
    covariance = np.full(5, .01)
    if first_width < 1.001:
        assert sigint_muresid_list(x, covariance, np.ones(5),
                                  abort=False, use_mad=True) == .5
    else:
        with pytest.raises(SNANAError, match='ONE NOT CONTAINED'):
            sigint_muresid_list(x, covariance, np.ones(5), abort=False, use_mad=True)


def test_large_scatter_does_not_require_a_grid_down_to_the_floor():
    x = np.tile([-1.e6, 1.e6], 10)
    covariance = np.ones(len(x))
    weights = np.ones(len(x))
    assert sigint_muresid_list(x, covariance, weights, abort=False, use_mad=True) == (
        _reference_mad(x, covariance, weights))
