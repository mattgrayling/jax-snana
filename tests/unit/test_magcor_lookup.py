"""Bulk MAGCOR matching retains the original tuple-key semantics."""

import numpy as np

from jax_snlc_fit.calibration import magcor_values


def test_magcor_rounding_repeated_epochs_and_missing_keys():
    # Decimal halfway cases distinguish Python round from np.round.
    mjds = np.array([55000.0005, 55000.0015, 55000.0005, 55000.0015, 55000.])
    assert round(float(mjds[0]), 3) != np.round(mjds[0], 3)
    cids = np.array(['001', '1', '001', 'other', '001'])
    bands = np.array(['g', 'r', 'g', 'r', 'i'])
    corrections = {('001', round(float(mjds[0]), 3), 'g'): .02,
                   ('1', round(float(mjds[1]), 3), 'r'): -.03,
                   ('missing', 55000., 'g'): .1}
    expected = np.array([corrections.get((cid, round(float(mjd), 3), band), 0.)
                         for cid, mjd, band in zip(cids, mjds, bands)])
    np.testing.assert_array_equal(magcor_values(corrections, cids, mjds, bands),
                                  expected)
    np.testing.assert_array_equal(magcor_values(corrections, cids[:0], mjds[:0], bands[:0]), [])
