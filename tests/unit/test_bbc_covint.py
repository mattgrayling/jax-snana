"""The intrinsic-scatter scan of sigint_muresid_list (sntools.c:5985-6279)."""
import numpy as np
import pytest

from jax_salt2mu.map_kernels import estimate_intrinsic_scatter
from jax_salt2mu.biascor import BiasCorEvents, sigint_coeff_grid
from jax_salt2mu.config import BBCConfig
from jax_salt2mu.tablevar import TableVar, EVENT_TYPE_BIASCOR
from jax_snlc_core.logging import SNANAError


def _scan(residual, covariance, use_mad=False):
    """One cell's scatter and diagnostics."""
    n = len(residual)
    result, diagnostics, _ = estimate_intrinsic_scatter(
        np.asarray(residual, float), np.asarray(covariance, float), np.ones(n),
        np.zeros(n, dtype=int), np.ones(n, dtype=bool), ncell=1, use_mad=use_mad)
    return float(result[0]), [bool(d[0]) for d in diagnostics]


@pytest.mark.parametrize('use_mad', [False, True])
@pytest.mark.parametrize('size, variance', [(.2, .01), (.5, .04), (1., .3)])
def test_scatter_gives_unit_pull_width(size, variance, use_mad):
    """The scatter is where the pull width crosses one, found by linear
    interpolation between trials 0.01 apart: for residuals of +-size with
    one variance, sqrt(size^2 - variance), or with the MAD's 1.48,
    sqrt((1.48 size)^2 - variance)."""
    width = 1.48 * size if use_mad else size
    scatter, diagnostics = _scan(np.tile([-size, size], 25), np.full(50, variance), use_mad)
    assert not any(diagnostics)
    assert scatter == pytest.approx(np.sqrt(width**2 - variance), abs=1e-3)


def test_first_trial_wider_than_one_raises_the_scan():
    """Unequal noise puts the pull width above one at the first trial, 0.6
    above the approximate scatter; the scan restarts higher, by the excess
    and 0.1, and finds the crossing below, where the mean of
    1 / (variance + scatter^2) over residuals of +-1 is one."""
    residual = np.tile([-1., 1., -1., 1.], 10)
    covariance = np.tile([.01, .01, 1.99, 1.99], 10)
    scatter, diagnostics = _scan(residual, covariance)
    assert not any(diagnostics)
    # .5 / (.01 + x) + .5 / (1.99 + x) = 1, so x^2 + x - .9801 = 0
    assert scatter == pytest.approx(np.sqrt((np.sqrt(1. + 4. * .9801) - 1.) / 2.), abs=1e-3)


def test_a_sample_narrower_than_its_noise_stops_at_the_floor():
    """With the sample narrower than its noise, the approximate scatter is
    -sqrt(<COV> - STD^2) and the scan returns its floor, 0.5 + 1e-7 below."""
    residual = np.tile([-.01, .01], 5)
    scatter, (invalid, narrow, floor, unbracketed, cap, zero) = _scan(residual, np.ones(10))
    assert narrow and floor and not (invalid or unbracketed or cap or zero)
    assert scatter == pytest.approx(-np.sqrt(1. - .01**2) - .5 - 1e-7, rel=0., abs=1e-12)


def test_a_first_trial_still_wider_than_one_is_unbracketed():
    """When the raised first trial's pull width still exceeds one, no pair of
    trials brackets one ('ONE is NOT CONTAINED', sntools.c:6260)."""
    _, diagnostics = _scan([0., 0., 0., -.1], [4., 4., 4., .01])
    assert diagnostics[3] and not diagnostics[4]


@pytest.mark.parametrize('use_mad', [False, True])
def test_a_scan_of_mxstore_pull_trials_aborts(use_mad):
    """A pull width that has not crossed one by the 120th trial ends the scan
    (MXSTORE_PULL, sntools.c:6223-6235)."""
    _, diagnostics = _scan(np.tile([-.5, .5], 25), np.tile([.01, 2.], 25), use_mad)
    assert diagnostics[4] and not diagnostics[2]


@pytest.mark.parametrize('use_mad', [False, True])
def test_zero_pull_width_and_invalid_covariance(use_mad):
    """Residuals all equal to their mean give a zero pull width, and a
    covariance of zero is invalid."""
    _, diagnostics = _scan(np.ones(10), np.full(10, .01), use_mad)
    assert diagnostics[5]
    _, diagnostics = _scan(np.tile([-.1, .1], 5), np.zeros(10), use_mad)
    assert diagnostics[0]


@pytest.mark.parametrize('residual, variance, message', [
    (0., 0., 'Invalid MUCOV'),
    (0., 1., 'STD < AVG_MUERR'),
    (1., .01, 'STD < AVG_MUERR'),
    # Equal, binary-exact scatter and noise reach the trial floor without a crossing.
    (np.tile([-1 / 512, 1 / 512], 25), (1 / 512)**2, 'sig trial < the scan floor'),
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
