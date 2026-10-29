"""Unit tests for ``LightCurveBatch.take`` and the epoch-length chooser."""

import dataclasses
import functools

import numpy as np
import pytest

from jax_snlc_fit.batch import (ERRFLAG_NO_FIT, LightCurveBatch, choose_epoch_lengths,
                                group_by_epoch_length, plan_fit_batches)
from jax_snlc_fit.cuts import gap_maxima
from jax_snlc_fit.photoz import z_prior_chi2
from jax_snlc_fit.photoz import ZSPEC_SOURCE_MASK
from jax_snlc_sim.simulation import Instrument


def _make_batch(n=4, max_obs=5):
    return LightCurveBatch(
        snids=np.arange(n).astype(str),
        redshifts=np.linspace(0.1, 0.4, n).astype(np.float32),
        redshift_errs=np.full(n, 0.001, dtype=np.float32),
        mwebvs=np.zeros(n, dtype=np.float32),
        n_obs_per_sn=np.full(n, max_obs, dtype=np.int32),
        reference_mjds=np.linspace(60000, 61000, n).astype(np.float64),
        mjd_offsets=np.tile(np.arange(max_obs, dtype=np.float64)[None], (n, 1)),
        filter_indices=np.zeros((n, max_obs), dtype=np.int32),
        fluxcals=np.ones((n, max_obs), dtype=np.float32),
        fluxcalerrs=np.full((n, max_obs), 0.1, dtype=np.float32),
        observation_mask=np.ones((n, max_obs), dtype=bool),
        zp_fluxcal=Instrument.zp_fluxcal,
        sim_mw_colour_law=None,
        ras=np.full(n, np.nan),
        decs=np.full(n, np.nan),
        errflag_fit=np.full(n, ERRFLAG_NO_FIT),
        filter_names={0: "LSST-r"},
    )


def test_an_event_without_a_helio_redshift_writes_the_converted_one():
    """SNANA writes zHEL from SNLC_ZHELIO, which SET_ZHELIO converts from zCMB
    for an event whose REDSHIFT_HELIO is undefined (snana.F90:17710), so the
    -9 of a file holding the column for other events is never written."""
    from jax_snlc_fit.output import _data_zhel

    batch = dataclasses.replace(_make_batch(n=2), head_extras={
        'REDSHIFT_HELIO': np.array([0.25, -9.0]),
        'REDSHIFT_HELIO_ERR': np.array([0.002, -9.0])})
    zhel, zhelerr = _data_zhel(batch)
    np.testing.assert_array_equal(zhel, [0.25, batch.redshifts[1]])
    np.testing.assert_array_equal(zhelerr, [0.002, batch.redshift_errs[1]])


def test_take_pads_rows_and_epochs():
    """A batch padded to a fixed shape: rows repeat the batch's own events,
    epochs past the light curve are masked zeros."""
    b = _make_batch(n=3, max_obs=5)
    padded = b.take(np.resize(np.arange(3), 7), 8)
    assert (padded.n_sne, padded.max_obs) == (7, 8)
    np.testing.assert_array_equal(
        padded.snids, np.array([b.snids[i] for i in [0, 1, 2, 0, 1, 2, 0]]))
    np.testing.assert_array_equal(padded.fluxcals[:, :5], b.fluxcals[[0, 1, 2, 0, 1, 2, 0]])
    assert not padded.observation_mask[:, 5:].any()
    assert not padded.fluxcals[:, 5:].any()


def test_fitres_sim_columns():
    """The loader carries SNANA's generic SIM_ columns (snana.F90:24158) and,
    per fitted model, the truth columns its fit spec names plus what it
    declares beyond them."""
    from jax_snlc_fit.loader import FITRES_SIM_COLUMNS, model_sim_columns
    assert FITRES_SIM_COLUMNS == [
        'SIM_GENTYPE', 'SIM_TEMPLATE_INDEX', 'SIM_LIBID',
        'SIM_NGEN_LIBID', 'SIM_SUBSAMPLE_INDEX',
        'SIM_REDSHIFT_CMB', 'SIM_REDSHIFT_HELIO',
        'SIM_REDSHIFT_HOST_MATCH', 'SIM_REDSHIFT_FLAG', 'SIM_VPEC',
        'SIM_DLMU', 'SIM_LENSDMU',
        'SIM_PEAKMJD', 'SIM_MWEBV', 'SIM_MAGSMEAR_COH',
        'SIM_WGT_POPULATION',
        'SIM_HOSTLIB_GALID',
        'SIM_AV', 'SIM_RV',
    ]

    class _Declares:
        SIM_TRUTH_COLUMNS = ('SIM_EXTRA', 'SIM_B')

        def get_fit_parameters(self):
            return {'PKMJD': {'sim_truth': 'SIM_PEAKMJD'},
                    'a': {'sim_truth': 'SIM_A'}, 'b': {'sim_truth': 'SIM_B'},
                    'c': {}}

    assert model_sim_columns(_Declares()) == (
        'SIM_PEAKMJD', 'SIM_A', 'SIM_B', 'SIM_EXTRA')


def test_absent_model_truth_columns_are_named(caplog):
    """A HEAD file from a sim of another model lacks this model's truth
    columns; the loader carries what exists and names what it did not find."""
    import logging
    from astropy.table import Table
    from jax_snlc_fit.loader import _sim_truth_columns
    head = Table({'SIM_PEAKMJD': [1.0], 'SIM_A': [2.0]})
    with caplog.at_level(logging.WARNING):
        out = _sim_truth_columns(head, True, ('SIM_PEAKMJD', 'SIM_A', 'SIM_B'))
    assert set(out) == {'SIM_PEAKMJD', 'SIM_A'}
    assert "no SIM_B" in caplog.text
    assert _sim_truth_columns(head, False, ('SIM_B',)) is None



def test_sim_prescale_keeps_every_nth_event_read():
    """SIM_PRESCALE keeps the events at read positions 0, N, 2N, ... whatever
    the other cuts do (RDHEAD_DRIVER, snana.F90:4083; REJECT_PRESCALE, :9990),
    and N_SNLC_PROC then counts the survivors."""
    from jax_snlc_fit.cuts import apply_snana_selection
    b = _make_batch(n=300)

    keep_all, index_all = apply_snana_selection(b, ladder=[])
    assert keep_all.all()
    # the index is 1-based and dense over the events reaching the fit driver
    assert index_all[0] == 1 and index_all[-1] == b.n_sne

    keep, index = apply_snana_selection(b, sim_prescale=3.0, ladder=[])
    assert (np.flatnonzero(keep) == np.arange(0, 300, 3)).all()
    assert index[keep].tolist() == list(range(1, 101))


def test_duplicate_cids_are_compared_as_cidassign_integers():
    """PARSE_CID tests CIDASSIGN's integer (snana.F90:14144): a name that does
    not parse as one takes a per-event index and can never collide, while '7'
    and '007' are one CID."""
    import pytest as _pytest
    from jax_snlc_core.logging import SNANAError
    from jax_snlc_fit.cuts import apply_snana_selection

    b = _make_batch(n=4)
    b.snids = np.array(['SN26', 'SN32', 'SN41', 'SN55'])
    keep, _ = apply_snana_selection(b, ladder=[])
    assert keep.all()

    b.snids = np.array(['SN26', 'SN32', 'SN26', 'SN55'])
    keep, _ = apply_snana_selection(b, abort_on_duplcid=True, ladder=[])
    assert keep.all()

    b.snids = np.array(['7', '32', '007', '55'])
    with _pytest.raises(SNANAError):
        apply_snana_selection(b, abort_on_duplcid=True, ladder=[])
    keep, _ = apply_snana_selection(b, abort_on_duplcid=False, ladder=[])
    # the first occurrence survives, the repeat does not
    assert list(keep) == [True, True, False, True]


def test_nobs_trest_window_includes_its_endpoints():
    """sntools.c:1972 tests >= TMIN and <= TMAX."""
    from jax_snlc_fit.cuts import apply_snana_selection

    b = _make_batch(n=1, max_obs=3)
    b.redshifts = np.zeros(1, dtype=np.float32)
    b.peakmjd_offsets = np.zeros(1)
    b.mjd_offsets = np.array([[-5.0, 0.0, 5.0]])
    # all three epochs sit on or inside [-5, 5]
    keep, _ = apply_snana_selection(b, nobs_trest=[(3, -5.0, 5.0)], ladder=[])
    assert keep[0]
    keep, _ = apply_snana_selection(b, nobs_trest=[(3, -4.9, 5.0)], ladder=[])
    assert not keep[0]


def test_duplicate_mjd_is_reported_not_rejected():
    """SNANA's ABORT_ON_DUPLMJD gate is commented out, so it warns on a repeated
    MJD+band unconditionally and never cuts (snana.F90:22414)."""
    from jax_snlc_fit.cuts import warn_duplicate_mjd
    b = _make_batch(n=2, max_obs=4)
    assert warn_duplicate_mjd(b) == 0

    mjd = np.array(b.mjd_offsets)
    mjd[0, 2] = mjd[0, 1]          # same MJD as the previous epoch, same band
    b.mjd_offsets = mjd
    assert warn_duplicate_mjd(b) == 1

    # a repeat in a different band is not a duplicate
    fidx = np.array(b.filter_indices)
    fidx[0, 2] = 1
    b.filter_indices = fidx
    assert warn_duplicate_mjd(b) == 0


def test_snccid_list_matches_the_transient_and_iauc_names(tmp_path):
    """An SNCCID_LIST or SNCID_LIST_FILE entry selects an event by its SNID or
    by either of its names (LCIDSELECT, snana.F90:11902), so a Rubin object
    id or an IAUC name picks the event out of a file whose SNIDs are numbers;
    SNIDs which are not numbers are matched as strings, and a numeric SNID
    as its number."""
    from jax_snlc_fit.cuts import apply_snana_selection
    from jax_snlc_fit.cuts import _cid_list

    cids, names, table = _cid_list(None, ['3', '2026abc', 'ZTF26aaa', ''], None)
    assert (cids, names, table) == (['3', '2026abc', 'ZTF26aaa'],
                                    ['3', '2026abc', 'ZTF26aaa'], None)

    b = _make_batch(n=4)
    b.head_extras = {'NAME_TRANSIENT': np.array(['ZTF26aaa', '', '', 'ZTF26zzz']),
                     'NAME_IAUC': np.array(['', '2026abc', '', ''])}
    keep, _ = apply_snana_selection(b, sncid_list=cids, snccid_names=names,
                                    ladder=[])
    assert list(keep) == [True, True, False, True]
    keep, _ = apply_snana_selection(b, sncid_list=[], snccid_names=['2026abc'],
                                    ladder=[])
    assert list(keep) == [False, True, False, False]

    b = _make_batch(n=4)
    b.snids = np.array(['ecf56b03-f72e', '878b0afa-a1b2', '0042', '7'])
    b.head_extras = {}
    cid_file = tmp_path / "cids.list"
    cid_file.write_text("ecf56b03-f72e  # first\n42\n")
    cids, names, _ = _cid_list([7], ['878b0afa-a1b2'], str(cid_file))
    keep, _ = apply_snana_selection(b, sncid_list=cids, snccid_names=names,
                                    ladder=[])
    assert keep.all()
    keep, _ = apply_snana_selection(b, sncid_ignore=['878b0afa-a1b2', '42'],
                                    ladder=[])
    assert list(keep) == [True, False, False, True]


def test_private_cutwin_veto_rejects_on_equality():
    """The 'VARNAME != VALUE' spelling rejects events matching the value rather
    than windowing them (USE_PRIVATE_CUTWIN = -1, snana.F90:5095)."""
    from jax_snlc_core.logging import SNANAError
    from jax_snlc_fit.cuts import apply_snana_selection
    import pytest as _pytest

    b = _make_batch(n=4)
    b.head_extras = {'PRIVATE(FOO)': np.array([1.0, 3.0, 3.0, 7.0])}

    keep, _ = apply_snana_selection(b, private_veto={'PRIVATE(FOO)': 3.0},
                                    ladder=[])
    assert list(keep) == [True, False, False, True]

    # the window spelling still selects rather than rejects
    keep, _ = apply_snana_selection(
        b, private_windows={'PRIVATE(FOO)': (2.0, 5.0)}, ladder=[])
    assert list(keep) == [False, True, True, False]

    # both spellings compose, and a name the HEAD lacks is an error
    keep, _ = apply_snana_selection(
        b, private_windows={'PRIVATE(FOO)': (2.0, 8.0)},
        private_veto={'PRIVATE(FOO)': 3.0}, ladder=[])
    assert list(keep) == [False, False, False, True]
    with _pytest.raises(SNANAError):
        apply_snana_selection(b, private_veto={'PRIVATE(BAR)': 1.0}, ladder=[])


def test_concatenate_remaps_band_indices_to_the_merged_mapping():
    """A filter index means nothing without the mapping it was assigned under.
    Two data versions can number their bands differently, so a merge that keeps
    both mappings has to translate the epochs of at least one of them."""
    from dataclasses import replace

    zeros = np.zeros((2, 5), dtype=np.int32)
    a = replace(_make_batch(n=2), filter_names={0: "DES-g", 1: "DES-r"},
                filter_indices=zeros.copy())
    b = replace(_make_batch(n=2), filter_names={0: "DES-r", 1: "DES-g"},
                filter_indices=zeros.copy())

    def band(batch, sn):
        return batch.filter_names[int(np.asarray(batch.filter_indices)[sn, 0])]

    assert band(a, 0) == "DES-g" and band(b, 0) == "DES-r"

    merged = LightCurveBatch.concatenate([a, b])
    assert band(merged, 0) == "DES-g", "first version's epochs changed band"
    assert band(merged, a.n_sne) == "DES-r", "second version's epochs changed band"


# ---------------------------------------------------------------------------
# Epoch-length binning
# ---------------------------------------------------------------------------

def _skewed_epoch_counts():
    """Many sparse SNe and a well-observed tail, which is what makes binning
    worth doing: one length for all pays the tail's cost on every SN."""
    rng = np.random.default_rng(0)
    return np.concatenate([rng.integers(2, 12, 200), rng.integers(70, 90, 20)])


def test_every_sn_lands_in_one_bin_that_fits_it():
    n_obs = _skewed_epoch_counts()
    bins = group_by_epoch_length(n_obs, choose_epoch_lengths(n_obs, 4, 0.05))

    seen = np.concatenate([idx for _, idx in bins])
    assert np.array_equal(np.sort(seen), np.arange(len(n_obs))), \
        "every SN belongs to exactly one bin"
    for length, idx in bins:
        assert n_obs[idx].max() <= length, \
            f"length {length} would truncate an SN with {n_obs[idx].max()} epochs"


def test_more_bins_never_cost_more_padding():
    n_obs = _skewed_epoch_counts()
    work = [sum(length * len(idx) for length, idx in
                group_by_epoch_length(n_obs, choose_epoch_lengths(n_obs, k, 0.0)))
            for k in (1, 2, 3, 4)]
    assert work == sorted(work, reverse=True), \
        f"padded work should not grow as bins are added: {work}"
    assert work[1] < work[0], "a skewed sample must gain from a second bin"


def test_one_bin_is_the_whole_sample_at_full_length():
    n_obs = _skewed_epoch_counts()
    assert choose_epoch_lengths(n_obs, 1, 0.05) == [int(n_obs.max())]
    bins = group_by_epoch_length(n_obs, [int(n_obs.max())])
    assert len(bins) == 1 and len(bins[0][1]) == len(n_obs)


def test_a_bin_is_added_only_when_it_earns_its_compilation():
    """min_gain is the whole guard against paying a compile for nothing."""
    n_obs = _skewed_epoch_counts()
    assert len(choose_epoch_lengths(n_obs, 4, 0.99)) == 1, \
        "an unreachable gain threshold should leave one bin"
    assert len(choose_epoch_lengths(n_obs, 4, 0.0)) > 1


def test_take_shortens_the_epoch_axis_and_nothing_else():
    """Four of the batch's arrays are two-dimensional without being per-epoch.

    zpdf_quantiles is per quantile, mw_att_obs per common wavelength and
    mwxt_magerrs per filter, so shortening any of them would silently corrupt
    the quantity rather than drop padding. obs_extras is per epoch and must be
    shortened with the rest.
    """
    from dataclasses import replace
    b = _make_batch(n=4, max_obs=5)
    b = replace(
        b,
        # three real epochs each, so shortening to three drops only padding
        n_obs_per_sn=np.full(4, 3, dtype=np.int32),
        obs_extras={'PHOTFLAG': np.arange(20, dtype=np.float64).reshape(4, 5)},
        zpdf_quantiles=np.arange(4 * 9, dtype=np.float64).reshape(4, 9),
        zpdf_nbin=np.array([9, 5, 0, 9]),
        mw_att_obs=np.arange(4 * 7, dtype=np.float64).reshape(4, 7),
        mwxt_magerrs=np.arange(4 * 3, dtype=np.float64).reshape(4, 3),
    )
    got = b.take(np.array([0, 2]), max_obs=3)

    assert got.max_obs == 3
    assert np.asarray(got.mjd_offsets).shape == (2, 3)
    assert np.asarray(got.obs_extras['PHOTFLAG']).shape == (2, 3)
    assert np.asarray(got.zpdf_quantiles).shape == (2, 9), "quantiles are not epochs"
    assert np.asarray(got.zpdf_nbin).tolist() == [9, 0]
    assert np.asarray(got.mw_att_obs).shape == (2, 7), "wavelengths are not epochs"
    assert np.asarray(got.mwxt_magerrs).shape == (2, 3), "filters are not epochs"
    # and the right rows came through, not merely the right shapes
    np.testing.assert_array_equal(np.asarray(got.mw_att_obs)[1],
                                  np.asarray(b.mw_att_obs)[2])


def test_the_fit_plan_covers_every_sn_once_at_the_given_shapes():
    """The plan bins the sample by epoch count into the lengths it is given
    and splits each bin into batches of at most ``batch_size`` events, so the
    compiled shapes are (batch_size, length) for one length per bin."""
    rng = np.random.default_rng(0)
    n_obs = np.concatenate([rng.integers(2, 12, 2000), rng.integers(70, 90, 200)])
    lengths = choose_epoch_lengths(n_obs, 4, 0.05)

    items = plan_fit_batches(n_obs, lengths, 500)
    seen = np.concatenate([rows for _, rows, _ in items])
    assert np.array_equal(np.sort(seen), np.arange(len(n_obs)))
    assert {length for length, _, _ in items} <= set(lengths)
    for length, rows, padded in items:
        assert n_obs[rows].max() <= length
        assert 0 < len(rows) <= 500
        # a short batch is padded with its own events, which lead it
        assert len(padded) == 500 and set(padded) == set(rows)
        assert np.array_equal(padded[:len(rows)], rows)


# ---------------------------------------------------------------------------
# SNANA sentinel and per-event semantics (audit 2026-08-28)
# ---------------------------------------------------------------------------

def test_redshift_shifts_apply_only_to_the_sne_whose_host_z_was_used():
    """SET_zSHIFT is per event (snana.F90:17861).

    The host shift moves an SN's redshift only when that host measurement *is*
    the SN redshift. Deciding it for the whole file from any one match shifts
    every other SN by a value that has nothing to do with it.
    """
    from jax_snlc_fit.loader import apply_redshift_shifts

    z = np.array([0.20, 0.50, 0.80])
    photoz = np.array([0.20, 0.55, 0.80])      # SN 1's host-z is not its redshift
    out_z, out_photoz, _, out_zcmb = apply_redshift_shifts(
        {'zphot': 0.01}, z, {}, photoz, None, None, z.copy())

    np.testing.assert_allclose(out_z, [0.21, 0.50, 0.81])
    np.testing.assert_allclose(out_zcmb, [0.21, 0.50, 0.81])
    np.testing.assert_allclose(np.asarray(out_photoz), [0.21, 0.55, 0.81])


def test_redshift_final_shift_moves_every_sn():
    """REDSHIFT_FINAL_SHIFT is unconditional, unlike the host shifts."""
    from jax_snlc_fit.loader import apply_redshift_shifts

    z = np.array([0.20, 0.50])
    out_z, _, _, _ = apply_redshift_shifts(
        {'final': 0.02}, z, {}, None, None, None, z.copy())
    np.testing.assert_allclose(out_z, [0.22, 0.52])


def test_gap_maxima_start_at_minus_999_and_only_a_real_gap_moves_them():
    """TGAPMAX and T0GAPMAX are -999 for a light curve with fewer than two
    accepted epochs (snana.F90:21092-21093), on a one-epoch axis too, and are
    otherwise the largest gap: overall, and touching the peak window."""
    times = np.array([[-10.0, 30.0, 0.0, 5.0], [1.0, 2.0, 3.0, 4.0]])
    accepted = np.array([[True, True, True, True], [True, False, False, False]])
    tgap, t0gap = gap_maxima(times, accepted, -5.0, 10.0)
    np.testing.assert_array_equal(tgap, [25.0, -999.0])
    np.testing.assert_array_equal(t0gap, [25.0, -999.0])

    tgap, t0gap = gap_maxima(times[:, :1], accepted[:, :1], -5.0, 10.0)
    np.testing.assert_array_equal(tgap, [-999.0, -999.0])
    np.testing.assert_array_equal(t0gap, [-999.0, -999.0])


@pytest.mark.parametrize('scale', [0.5, 1., 2.])
def test_host_gaussian_constrains_small_redshift_errors(scale):
    from dataclasses import replace
    import jax
    import jax.numpy as jnp
    from jax_snlc_fit import LightCurveFitter
    from jax_snlc_fit.photoz import redshift_priors

    fitter = LightCurveFitter('SALT3.K21', batch_size=1,
        opt_photoz=1, prior_zerrscale=scale)
    errors = np.array([0., .001, .005, .03])
    means = np.full(4, .1)
    batch = replace(_make_batch(), photoz=means, photoz_err=errors)
    priors = redshift_priors(fitter.config, batch)
    sigma = np.sqrt(errors**2 + .001**2) * scale
    evaluate = jax.jit(jax.vmap(functools.partial(z_prior_chi2, fitter._compiled_settings)))
    np.testing.assert_allclose(evaluate(jnp.asarray(means), priors), 0., atol=1e-12)
    np.testing.assert_allclose(evaluate(jnp.asarray(means + sigma), priors), 1., rtol=1e-10)
    curvature = jax.jit(jax.vmap(jax.grad(jax.grad(functools.partial(z_prior_chi2, fitter._compiled_settings)))))
    np.testing.assert_allclose(curvature(jnp.asarray(means), priors), 2. / sigma**2,
                               rtol=1e-10)


@pytest.mark.parametrize("mode", [6, 14])
def test_mixed_redshift_priors(mode):
    """Spec-z rows use an analytic prior, even with wrong or absent host PDFs."""
    from dataclasses import replace
    import jax
    import jax.numpy as jnp
    from jax_snlc_fit.config import FitConfig
    from jax_snlc_fit.photoz import quantile_prior_mask, redshift_priors
    from jax_snlc_fit.cuts import drop_unfittable

    cfg = FitConfig()
    cfg.photoz.dofit_photoz = True
    cfg.photoz.opt_photoz = mode
    cfg.photoz.prior_zerrscale = 1.0
    cfg.model.opt_prior = 1
    batch = replace(_make_batch(),
        redshift_errs=np.array([.001, .001, .1, .1]),
        redshift_source_mask=np.array([ZSPEC_SOURCE_MASK, ZSPEC_SOURCE_MASK, 0, 0]),
        zpdf_quantiles=np.array([[.7, .8, .9], [-9., -9., -9.],
                                [.2, .3, .4], [-9., -9., -9.]]),
        zpdf_percentiles=np.array([[0., 50., 100.], [-9., -9., -9.],
                                  [0., 50., 100.], [-9., -9., -9.]]),
        zpdf_nbin=np.array([3, 0, 3, 0]))
    kept, _, _, bad = drop_unfittable(
        batch, None, abort_on_noepochs=True,
        quantile_prior=quantile_prior_mask(cfg, batch),
        abort_on_missing_qzphot=False)
    np.testing.assert_array_equal(kept.snids, batch.snids[:3])
    np.testing.assert_array_equal(bad, batch.snids[3:])
    priors = redshift_priors(cfg, kept)
    np.testing.assert_array_equal(priors.use_quantiles, [False, False, True])
    # Exercise the same vmapped prior dispatch as the fitter, including AD.
    def evaluate(z, prior):
        return z_prior_chi2({'config': cfg}, z, prior)
    spec = jax.tree.map(lambda a: a[0], priors)
    zspec = float(kept.redshifts[0])
    assert float(evaluate(zspec, spec)) == 0.0
    sigma = np.sqrt(float(kept.redshift_errs[0])**2 + 1e-6)
    assert float(evaluate(zspec + sigma, spec)) == pytest.approx(1., rel=1e-5)
    assert float(jax.grad(jax.grad(evaluate))(zspec, spec)) == pytest.approx(
        2. / sigma**2, rel=1e-5)
    values = jax.jit(jax.vmap(evaluate))(jnp.asarray(kept.redshifts), priors)
    assert np.isfinite(values).all()
    assert float(values[2]) == pytest.approx(0., abs=1e-5)

    # An entirely spectroscopic batch needs no quantile columns at all.
    spec_batch = replace(kept.take(np.array([0, 1]), kept.max_obs),
                         zpdf_quantiles=None, zpdf_percentiles=None, zpdf_nbin=None)
    spec_batch, _, _, bad = drop_unfittable(
        spec_batch, None, abort_on_noepochs=True,
        quantile_prior=quantile_prior_mask(cfg, spec_batch),
        abort_on_missing_qzphot=True)
    assert not len(bad)
    spec_priors = redshift_priors(cfg, spec_batch)
    np.testing.assert_allclose(jax.vmap(evaluate)(
        jnp.asarray(spec_batch.redshifts), spec_priors), 0.)


@pytest.mark.parametrize('mode', [4, 6])
def test_missing_quantiles(mode):
    from jax_snlc_core.logging import SNANAError
    from jax_snlc_fit.config import FitConfig
    from jax_snlc_fit.photoz import quantile_prior_mask, redshift_priors, scan_size
    from jax_snlc_fit.cuts import drop_unfittable

    config = FitConfig()
    config.photoz.dofit_photoz = True
    config.photoz.opt_photoz = mode
    batch = _make_batch(n=2)
    batch.redshift_source_mask = np.array([ZSPEC_SOURCE_MASK, 0])
    needed = quantile_prior_mask(config, batch)
    with pytest.raises(SNANAError, match='Missing zPDF quantiles'):
        drop_unfittable(batch, None, abort_on_noepochs=True,
                        quantile_prior=needed, abort_on_missing_qzphot=True)
    kept, _, _, rejected = drop_unfittable(
        batch, None, abort_on_noepochs=True, quantile_prior=needed,
        abort_on_missing_qzphot=False)
    np.testing.assert_array_equal(rejected, batch.snids[needed])
    np.testing.assert_array_equal(kept.snids, batch.snids[~needed])
    prior = redshift_priors(config, kept)
    assert scan_size(config, prior) >= 1


def test_concatenate_host_priors():
    from dataclasses import replace
    from jax_snlc_core.snana_files import SNANA_NULLFLOAT
    from jax_snlc_fit.config import FitConfig
    from jax_snlc_fit.photoz import redshift_priors, is_zspec

    cfg = FitConfig()
    cfg.photoz.dofit_photoz = True
    cfg.photoz.opt_photoz = 1
    cfg.photoz.prior_zerrscale = 1.
    measured = replace(_make_batch(n=1), photoz=np.array([.3]),
                       photoz_err=np.array([.04]),
                       redshift_source_mask=np.array([ZSPEC_SOURCE_MASK]))
    absent = _make_batch(n=1)
    no_error = replace(absent, photoz=np.array([.5]))
    parts = [measured, absent, no_error]
    merged = LightCurveBatch.concatenate(parts)
    np.testing.assert_array_equal(merged.photoz_err, [.04, SNANA_NULLFLOAT, SNANA_NULLFLOAT])
    expected = [redshift_priors(cfg, b) for b in parts]
    actual = redshift_priors(cfg, merged)
    for field in ('z_init', 'z_prior', 'z_prior_err', 'z_prior_width', 'host_ok'):
        np.testing.assert_allclose(getattr(actual, field),
                                   np.concatenate([getattr(p, field) for p in expected]))
    np.testing.assert_array_equal(is_zspec(cfg, merged, 3),
                                   np.concatenate([is_zspec(cfg, b, 1) for b in parts]))
    empty = LightCurveBatch.concatenate([absent, absent])
    assert empty.photoz is None and empty.redshift_source_mask is None


@pytest.mark.parametrize('mode', [6, 14])
def test_concatenate_quantile_priors(mode):
    from dataclasses import replace
    from jax_snlc_core.logging import SNANAError
    from jax_snlc_fit.config import FitConfig
    from jax_snlc_fit.photoz import quantile_prior_mask, redshift_priors
    from jax_snlc_fit.cuts import drop_unfittable
    from jax_snlc_fit.output import build_host_columns

    cfg = FitConfig()
    cfg.photoz.dofit_photoz = True
    cfg.photoz.opt_photoz = mode
    cfg.photoz.prior_zerrscale = 1.
    parts = []
    for name, q, percent in [('a', [.2, .3, .4], [0., 50., 100.]),
                             ('b', [.1, .15, .3, .4, .6], [0., 10., 50., 90., 100.])]:
        parts.append(replace(_make_batch(n=1), snids=np.array([name]),
            redshift_errs=np.array([.1]), zpdf_quantiles=np.array([q]),
            zpdf_percentiles=np.array([percent]), zpdf_nbin=np.array([len(q)]),
            host_data={'HOSTGAL_PHOTOZ': np.array([.3]),
                       'HOSTGALz_QUANTILE_ZPHOT': np.array([q]),
                       'HOSTGALz_QUANTILE_PERCENT': np.array([percent]),
                       'HOSTGALz_NBIN_QUANTILE_ZPHOT': np.array([len(q)])}))
    spec = replace(_make_batch(n=1), snids=np.array(['spec']),
                   redshift_source_mask=np.array([ZSPEC_SOURCE_MASK]))
    absent = replace(_make_batch(n=1), snids=np.array(['missing']),
                     redshift_errs=np.array([.1]))
    merged = LightCurveBatch.concatenate([*parts, spec, absent])
    np.testing.assert_array_equal(merged.zpdf_nbin, [3, 5, 0, 0])
    np.testing.assert_array_equal(merged.host_data['HOSTGALz_QUANTILE_ZPHOT'],
                                  merged.zpdf_quantiles)
    with pytest.raises(SNANAError, match='CID=missing'):
        drop_unfittable(merged, None, abort_on_noepochs=True,
                       quantile_prior=quantile_prior_mask(cfg, merged),
                       abort_on_missing_qzphot=True)
    kept, _, _, bad = drop_unfittable(merged, None, abort_on_noepochs=True,
        quantile_prior=quantile_prior_mask(cfg, merged), abort_on_missing_qzphot=False)
    np.testing.assert_array_equal(bad, ['missing'])
    np.testing.assert_array_equal(kept.snids, ['a', 'b', 'spec'])
    actual = redshift_priors(cfg, kept)
    for row, part in enumerate([*parts, spec]):
        expected = redshift_priors(cfg, part)
        for field in ('z_prior', 'z_prior_err', 'z_prior_width', 'use_quantiles'):
            np.testing.assert_allclose(getattr(actual, field)[row], getattr(expected, field)[0])
        if row < 2:
            for field in ('pdf_grid', 'pdf_vals', 'pdf_slopes'):
                np.testing.assert_allclose(getattr(actual, field)[row], getattr(expected, field)[0])
    columns = build_host_columns({}, merged.zp_fluxcal, host_text=True,
        zpdf_quantiles=merged.zpdf_quantiles, zpdf_percentiles=merged.zpdf_percentiles,
        zpdf_nbin=merged.zpdf_nbin, quantile_spline=cfg.photoz.quantile_spline())
    for row, part in enumerate(parts):
        expected = build_host_columns({}, part.zp_fluxcal, host_text=True,
            zpdf_quantiles=part.zpdf_quantiles, zpdf_percentiles=part.zpdf_percentiles,
            zpdf_nbin=part.zpdf_nbin, quantile_spline=cfg.photoz.quantile_spline())
        for field in ('HOST_QZPHOT', 'HOST_QZPHOTSTD'):
            np.testing.assert_allclose(columns[field][row], expected[field][0])
    np.testing.assert_array_equal(columns['HOST_QZPHOT'][2:], [-9., -9.])


@pytest.mark.parametrize('quantity', ['REDSHIFT', 'redshift', 'Redshift'])
def test_cut_quantity_case_applies_same_selection(quantity):
    from jax_snlc_fit.cuts import apply_snana_selection
    from jax_snlc_fit.config_parser import build_config

    config = build_config({'cutwin': {quantity: (0.15, 0.35), 'NEPOCH': (4, 6)}})
    keep, _ = apply_snana_selection(_make_batch(),
                                    windows=config.selection.cutwin,
                                    nepoch_store=np.full(4, 5), ladder=[])
    np.testing.assert_array_equal(keep, [False, True, True, False])


@pytest.mark.parametrize('dtype', ['S16', 'U16', object])
def test_epoch_labels_keep_sorted_codes_and_whitespace_semantics(dtype):
    from jax_snlc_fit.loader import _label_codes
    column = np.array(['VOID', ' C3', 'g', 'C3 ', 'LSST-g', '', ' C3',
                       'r', 'g', 'unused'], dtype=dtype)
    for rows in [np.array([8, 6, 4, 2, 1, 3, 7, 5, 0, 2]),
                 np.array([], dtype=int)]:
        expected, inverse = np.unique(column[rows], return_inverse=True)
        labels, codes = _label_codes(column, rows)
        np.testing.assert_array_equal(labels, np.char.strip(expected.astype('U')))
        np.testing.assert_array_equal(codes, inverse)
        assert codes.dtype == inverse.dtype


def test_host_surface_brightness_gathers_the_matching_event_and_band():
    from jax_snlc_fit.calibration import host_surface_brightness
    head = {'HOSTGAL_SB_FLUXCAL_g': np.array([10., 20.], dtype='>f8'),
            'HOSTGAL_SB_FLUXCAL_r': np.array([30., 40.], dtype='>f8')}
    result = host_surface_brightness(head, np.array(['r', 'g', 'r', 'g']),
                                      np.array([1, 0, 0, 1]))
    np.testing.assert_array_equal(result, [40., 10., 30., 20.])


@pytest.mark.parametrize('windows', [{}, {'selected': (0, 3)}])
@pytest.mark.parametrize('reverse', [False, True])
def test_range_cuts_ignore_unselected_nan_column(windows, reverse):
    """Unrelated NaNs cannot change selection, eagerly or under JIT."""
    import jax
    from jax_snlc_core.utils import apply_range_cuts

    quantities = {'unused': np.array([np.nan, 3.]), 'selected': np.array([1., 2.])}
    if reverse:
        quantities = dict(reversed(list(quantities.items())))
    np.testing.assert_array_equal(apply_range_cuts(windows, quantities), [True, True])
    compiled = jax.jit(lambda values: apply_range_cuts(windows, values))
    np.testing.assert_array_equal(compiled(quantities), [True, True])


@pytest.mark.parametrize("text_dtype", [str, object])
def test_concatenate_missing_metadata_preserves_values(text_dtype):
    """Missing metadata has complete null markers and exact integer values."""
    first = dataclasses.replace(_make_batch(n=2), head_extras={
        'NAME': np.array(['A', 'B'], dtype=text_dtype),
        'ID': np.array([2**62 + 1, 2**63 - 1], dtype=np.int64),
        'VECTOR': np.array([['A'], ['B']], dtype=text_dtype),
    }, obs_extras={'PHOTFLAG': np.full((2, 5), 4096, dtype=np.int32)})
    second = dataclasses.replace(_make_batch(n=1, max_obs=3),
                                 head_extras={'VECTOR': np.array([['C', 'D']])})
    merged = LightCurveBatch.concatenate([first, second])
    assert merged.head_extras['NAME'].tolist() == ['A', 'B', 'NULL']
    assert merged.head_extras['ID'].tolist() == [2**62 + 1, 2**63 - 1, -9]
    assert merged.head_extras['VECTOR'].tolist() == [['A', 'NULL'], ['B', 'NULL'], ['C', 'D']]
    np.testing.assert_array_equal(merged.obs_extras['PHOTFLAG'][:2], first.obs_extras['PHOTFLAG'])
    np.testing.assert_array_equal(merged.obs_extras['PHOTFLAG'][2], np.zeros(5))
    assert merged.obs_extras['PHOTFLAG'].dtype == np.dtype('int32')
    merged_again = LightCurveBatch.concatenate([merged, _make_batch(n=1)])
    assert merged_again.head_extras['NAME'].tolist() == ['A', 'B', 'NULL', 'NULL']
    assert merged_again.head_extras['ID'].tolist() == [2**62 + 1, 2**63 - 1, -9, -9]


def test_concatenate_after_band_cuts_remove_all_epochs():
    """An input with no retained epochs can merge with usable light curves."""
    from jax_snlc_fit.loader import batch_from_columns

    batches = []
    for snid, band in [('excluded', 'g'), ('retained', 'r')]:
        batches.append(batch_from_columns(
            {'SNID': np.array([snid]), 'PTROBS_MIN': np.array([1]),
             'PTROBS_MAX': np.array([1]), 'REDSHIFT_HELIO': np.array([0.1])},
            {'MJD': np.array([60000.]), 'FLUXCAL': np.array([100.]),
             'FLUXCALERR': np.array([5.]), 'BAND': np.array([band]),
             'FIELD': np.array(['FIELD1'])},
            is_sim=False, zp_fluxcal=27.5, sim_mw_colour_law=None,
            filter_map={'g': 0, 'r': 1}, fit_bands=['r'], keep_obs_extras=True))

    merged = LightCurveBatch.concatenate(batches)
    assert merged.snids.tolist() == ['excluded', 'retained']
    assert merged.n_obs_per_sn.tolist() == [0, 1]
    assert merged.observation_mask.tolist() == [[False], [True]]
    assert merged.obs_extras['FIELD'].tolist() == [[''], ['FIELD1']]
