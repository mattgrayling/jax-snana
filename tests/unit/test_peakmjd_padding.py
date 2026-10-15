"""Peak-estimator padding preserves logical windows, ties and fallback cuts."""

import logging

import jax
import numpy as np
import pytest

from jax_snlc_core.photometry.peakmjd import fmax_clump_peakmjd


@pytest.mark.parametrize('log_weight', [False, True])
@pytest.mark.parametrize('snrcut', [0., 3., 5., 200.])
def test_padded_peak_histograms_match_unpadded(log_weight, snrcut):
    rng = np.random.default_rng(51)
    for width, bins in [(9, 5), (23, 11), (23, 53)]:
        times = rng.integers(-20, bins * 10 - 20, (32, width)).astype(float)
        flux = rng.integers(-2, 5, times.shape).astype(float)
        snr = rng.choice([0.1, 0.5, 3., 5., 10.], times.shape)
        counts = rng.integers(0, width + 1, len(times))
        options = dict(mjdwin=30., snrcut=snrcut, snrcut_backup=3.,
                       mjd_span=bins * 10 - 1, log_snr_weight=log_weight)
        expected = fmax_clump_peakmjd(times, flux, snr, counts, **options)
        padded = [np.pad(values, ((0, 8), (0, 64 - width)))
                  for values in (times, flux, snr)]
        actual = fmax_clump_peakmjd(*padded, np.pad(counts, (0, 8)),
                                   histogram_bins=80, **options)
        np.testing.assert_array_equal(actual[:len(times)], expected)
        np.testing.assert_array_equal(actual[len(times):], -9.)


def test_different_logical_histograms_reuse_compilation(caplog):
    times = np.tile(np.arange(16.) * 10, (7, 1))
    flux = np.ones_like(times)
    snr = flux * 6.
    counts = np.full(7, 16)
    options = dict(mjdwin=30., snrcut=5., snrcut_backup=3., histogram_bins=37)
    fmax_clump_peakmjd(times, flux, snr, counts, mjd_span=160., **options).block_until_ready()
    with caplog.at_level(logging.WARNING), jax.log_compiles(True):
        fmax_clump_peakmjd(times + 400., flux, snr, counts,
                           mjd_span=200., **options).block_until_ready()
    assert not any('Compiling jit(_clump_peaks)' in record.message
                   for record in caplog.records)


def test_peak_sizes_use_head_pointers_and_overrides(tmp_path):
    from jax_snlc_fit.data_loader import peak_sizes, read_header_override_file

    head = dict(SNID=np.array(['1', '2']), PTROBS_MIN=np.array([1, 6]),
                PTROBS_MAX=np.array([3, 8]))
    phot = dict(MJD=np.array([0., 1., 2., 35., -999999., 20., 30., 40.]))
    sources = [('sample', (head, phot, {'is_sim': True}))]
    options = dict(fmax_clump=True, fluxmax=False, num_nobs_bins=1,
                   min_bin_gain=.1, mjdwin=30.)
    assert peak_sizes(sources, options, {}) == {
        'epoch_lengths': [3], 'histogram_bins': 5}
    override = tmp_path / 'override.txt'
    override.write_text('VARNAMES: CID PTROBS_MAX\nSN: 1 4\n')
    shared = {'sim_header_override': read_header_override_file([str(override)])}
    assert peak_sizes(sources, options, shared) == {
        'epoch_lengths': [4], 'histogram_bins': 5}
    np.testing.assert_array_equal(head['PTROBS_MAX'], [3, 8])
