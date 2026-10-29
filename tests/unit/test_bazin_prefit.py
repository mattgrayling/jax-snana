"""Unit tests for the Bazin pre-fit t0 refinement (SNANA helper)."""

import jax.numpy as jnp
import numpy as np

from jax_snlc_core.peak_time import bazin_peakmjd
from jax_snlc_core.utils import CUTVAL_OPEN
from jax_snlc_fit.batch import ERRFLAG_NO_FIT, LightCurveBatch
from jax_snlc_sim.simulation import Instrument

# CUTWIN_TOBS as SNANA initialises it: every epoch is inside FITWIN_MJD8
OPEN_TOBS = (-CUTVAL_OPEN, CUTVAL_OPEN)


def _bazin_peaks(batch, cutwin_tobs, fallback_t0):
    """The Bazin peak estimate of ``batch``'s light curves."""
    return bazin_peakmjd(batch.mjd_offsets, batch.fluxcals, batch.fluxcalerrs,
                         batch.filter_indices, batch.observation_mask,
                         len(batch.filter_names), cutwin_tobs, fallback_t0)


def _bazin_curve(t, t0, a0=100.0, trise=5.0, tfall=15.0):
    """Synthetic Bazin profile: peak at ``t0``, given rise/fall scales."""
    dt = t - t0
    return a0 * np.exp(-dt / tfall) / (1.0 + np.exp(-dt / trise))


def _make_bazin_batch(true_t0s, n_per_filter=15, snr=20.0, seed=0):
    """Build a synthetic LightCurveBatch with one Bazin profile per SN per filter."""
    rng = np.random.default_rng(seed)
    n_sne = len(true_t0s)
    n_filters = 3  # use a small filter set: g, r, i
    max_obs = n_per_filter * n_filters

    mjd_offsets = np.zeros((n_sne, max_obs), dtype=np.float64)
    fluxcals = np.zeros((n_sne, max_obs), dtype=np.float32)
    fluxcalerrs = np.full((n_sne, max_obs), 1.0, dtype=np.float32)
    filter_indices = np.zeros((n_sne, max_obs), dtype=np.int32)
    obs_mask = np.ones((n_sne, max_obs), dtype=bool)

    for i, t0 in enumerate(true_t0s):
        offset = 0
        for fi in range(n_filters):
            t = np.linspace(t0 - 25, t0 + 35, n_per_filter)
            flux = _bazin_curve(t, t0)
            err = np.maximum(flux / snr, 1.0)
            noise = rng.normal(0, err)
            mjd_offsets[i, offset:offset + n_per_filter] = t
            fluxcals[i, offset:offset + n_per_filter] = flux + noise
            fluxcalerrs[i, offset:offset + n_per_filter] = err
            filter_indices[i, offset:offset + n_per_filter] = fi
            offset += n_per_filter

    return LightCurveBatch(
        snids=np.arange(n_sne).astype(str),
        redshifts=np.full(n_sne, 0.1, dtype=np.float32),
        redshift_errs=np.full(n_sne, 0.001, dtype=np.float32),
        mwebvs=np.zeros(n_sne, dtype=np.float32),
        n_obs_per_sn=np.full(n_sne, max_obs, dtype=np.int32),
        reference_mjds=np.zeros(n_sne, dtype=np.float64),
        mjd_offsets=mjd_offsets,
        filter_indices=filter_indices,
        fluxcals=fluxcals,
        fluxcalerrs=fluxcalerrs,
        observation_mask=obs_mask,
        zp_fluxcal=Instrument.zp_fluxcal,
        sim_mw_colour_law=None,
        ras=np.full(n_sne, np.nan),
        decs=np.full(n_sne, np.nan),
        errflag_fit=np.full(n_sne, ERRFLAG_NO_FIT),
        filter_names={0: "LSST-g", 1: "LSST-r", 2: "LSST-i"},
    )


def test_cutwin_tobs_narrows_the_epochs_the_bazin_fit_sees():
    """FCNANYLC sums only the epochs inside FITWIN_MJD8, the brightest epoch
    of the light curve widened by CUTWIN_TOBS (snana.F90:26609, :27132), so a
    window that drops the decline changes the peak the fit returns while an
    open one leaves it where the whole curve puts it."""
    true_t0s = np.array([60100.0, 60200.0, 60300.0])
    batch = _make_bazin_batch(true_t0s, n_per_filter=15, snr=30.0, seed=0)

    fallback = jnp.zeros(3)
    whole = np.asarray(_bazin_peaks(batch, OPEN_TOBS, fallback)[0])
    # a window wide enough to hold every epoch is the same fit
    np.testing.assert_allclose(
        np.asarray(_bazin_peaks(batch, (-400.0, 400.0), fallback)[0]), whole,
        rtol=1e-12)
    # the rise alone is a different one
    rise_only = np.asarray(_bazin_peaks(batch, (-30.0, 0.0), fallback)[0])
    assert not np.allclose(rise_only, whole, atol=0.05)


def test_bazin_prefit_recovers_synthetic_t0():
    """Bazin pre-fit recovers the true t0 of a synthetic Bazin profile."""
    true_t0s = np.array([60100.0, 60200.0, 60300.0])
    batch = _make_bazin_batch(true_t0s, n_per_filter=15, snr=30.0, seed=0)
    refined = np.asarray(_bazin_peaks(batch, OPEN_TOBS, jnp.zeros(3))[0])
    assert refined.shape == (3,)
    diffs = np.abs(refined - true_t0s)
    # The Bazin function has a known offset between its analytic peak (where
    # df/dt=0 from the rise+decay product) and the input ``t0`` parameter; the
    # offset scales with ``trise``. One ``trise`` (=5 d here) is the natural
    # bound — larger than that means the fit isn't converging at all.
    trise = 5.0
    assert float(np.median(diffs)) < trise, (
        f"median |Δt0| = {np.median(diffs):.2f} d > trise = {trise} d"
    )


def test_bazin_reports_only_events_with_no_epoch_at_all():
    """NPKMJD counts every filter that reaches MNFIT_PKMJD, so only an event
    with none is unplaceable (snana.F90:26640). A filter too sparse to yield a
    usable peak is dropped by the tests on the fit, not by an epoch count."""
    from test_lightcurve_batch import _make_batch

    b = _make_batch(n=3, max_obs=2)
    b.fluxcals = jnp.ones((3, 2))
    # clear of MAX_SNRMIN_REJECT, so no band is dropped for being the weakest
    b.fluxcalerrs = jnp.full((3, 2), 0.1)

    fallback = jnp.array([1.0, 2.0, 3.0])
    t0, failed, used_fallback = _bazin_peaks(b, OPEN_TOBS, fallback)
    # sparse, but every filter holds an epoch, so NPKMJD > 0 and the event is
    # placed from the fit rather than reported unplaceable
    assert not failed.any() and not used_fallback.any()
    assert not np.allclose(np.asarray(t0), np.asarray(fallback))

    # the one band each event has is also its weakest, and below S/N 2 SNANA
    # drops it before fitting, leaving NPKMJD = 0 (snana.F90:26584, :26619)
    b.fluxcalerrs = jnp.full((3, 2), 1.0)
    t0, failed, used_fallback = _bazin_peaks(b, OPEN_TOBS, fallback)
    assert failed.all() and used_fallback.all()
    np.testing.assert_allclose(np.asarray(t0), np.asarray(fallback))

    b.observation_mask = jnp.zeros((3, 2), dtype=bool)
    t0, failed, used_fallback = _bazin_peaks(b, OPEN_TOBS, fallback)
    assert failed.all(), "an event with no epoch has no filter to fit"
    assert used_fallback.all()
    # and with nothing to fit it keeps the peak it came in with
    np.testing.assert_allclose(np.asarray(t0), np.asarray(fallback))
