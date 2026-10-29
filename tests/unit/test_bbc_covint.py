"""The accelerated MAD search must preserve the reference trial-grid estimator."""
import numpy as np
import pytest

from jax_salt2mu.map_kernels import estimate_intrinsic_scatter
from jax_salt2mu.biascor import BiasCorEvents, sigint_coeff_grid
from jax_salt2mu.config import BBCConfig
from jax_salt2mu.tablevar import TableVar, EVENT_TYPE_BIASCOR
from jax_snlc_core.logging import SNANAError


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
    actual, _, _ = estimate_intrinsic_scatter(
        x, covariance, weights, np.zeros(n, dtype=int), np.ones(n, dtype=bool),
        ncell=1, use_mad=True)
    np.testing.assert_allclose(actual, expected, rtol=0., atol=2e-14)


@pytest.mark.parametrize('mad', [False, True])
def test_invalid_covariance_and_scatter_floor_diagnostics(mad):
    """The map estimator reports invalid covariance and permits negative scatter."""
    x = np.zeros(10)
    for variance in (0., 1.):
        result, diagnostics, counts = estimate_intrinsic_scatter(
            x, np.full(10, variance), np.ones(10), np.zeros(10, dtype=int),
            np.ones(10, dtype=bool), ncell=1, use_mad=mad)
        invalid, narrow, floor, unbracketed = diagnostics
        assert bool(invalid[0]) == (variance == 0.)
        assert bool(narrow[0]) == (variance > 0.)
        assert floor[0] and not unbracketed[0]
        assert result[0] == -.3 and counts[0] == 10


@pytest.mark.parametrize('residual, variance, message', [
    (0., 0., 'Invalid MUCOV'),
    (0., 1., 'STD < AVG_MUERR'),
    (1., .01, 'STD < AVG_MUERR'),
    # Equal, binary-exact scatter and noise reach the trial floor without a crossing.
    (np.tile([-1 / 512, 1 / 512], 25), (1 / 512)**2, 'sig trial < -0.300000'),
    # Unequal noise puts the first trial's pull width above one.
    (np.tile([-1., 1.], 25), np.tile([.01, 1.99], 25), 'ONE NOT CONTAINED'),
])
def test_biascor_scatter_rejects_invalid_samples(residual, variance, message):
    """Bias-correction preparation turns estimator diagnostics into named errors."""
    n = 50
    zeros = np.zeros(n)
    indices = np.zeros(n, dtype=int)
    table = TableVar(
        event_type=EVENT_TYPE_BIASCOR, name=np.arange(n).astype(str),
        idsurvey=indices, idsample=indices, zhd=zeros, zmuerr=zeros,
        logmass=zeros, lcpar={}, covmat_fit=np.zeros((n, 1, 1)),
        mumodel=zeros, cutbits=indices, snrmax=np.full(n, 100.),
        sim_truth={'SIM_WGT_POP': np.ones(n)})
    events = BiasCorEvents([], indices, np.zeros((n, 1)),
                           np.broadcast_to(variance, (n,)), np.broadcast_to(residual, (n,)),
                           np.zeros((n, 1)))
    with pytest.raises(SNANAError, match=message):
        sigint_coeff_grid(table, events, [], BBCConfig(sigint_biascor=-1.))


def test_mad_floor_diagnostic():
    """Uncentred width passes the initial check, but centred residuals vanish."""
    _, diagnostics, _ = estimate_intrinsic_scatter(
        np.ones(10), np.full(10, .01), np.ones(10), np.zeros(10, dtype=int),
        np.ones(10, dtype=bool), ncell=1, use_mad=True)
    invalid, narrow, floor, unbracketed = diagnostics
    assert floor[0] and not (invalid[0] or narrow[0] or unbracketed[0])


@pytest.mark.parametrize('first_width', [1.0005, 1.1])
def test_mad_first_trial_crossing(first_width):
    """The first trial accepts only crossings within the reference tolerance."""
    x = np.array([0., 0., 0., 0., 5 * np.sqrt(.26) * first_width / 1.48])
    result, diagnostics, _ = estimate_intrinsic_scatter(
        x, np.full(5, .01), np.ones(5), np.zeros(5, dtype=int),
        np.ones(5, dtype=bool), ncell=1, use_mad=True)
    assert result[0] == .5
    assert bool(diagnostics[3][0]) == (first_width >= 1.001)


def test_large_scatter_does_not_require_a_grid_down_to_the_floor():
    """Large residuals agree with the independent descending trial scan."""
    x = np.tile([-1.e6, 1.e6], 10)
    covariance = np.ones(len(x))
    weights = np.ones(len(x))
    result, _, _ = estimate_intrinsic_scatter(
        x, covariance, weights, np.zeros(len(x), dtype=int),
        np.ones(len(x), dtype=bool), ncell=1, use_mad=True)
    assert result[0] == _reference_mad(x, covariance, weights)
