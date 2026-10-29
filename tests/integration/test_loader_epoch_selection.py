"""Small loader regressions using recorded photometry, without running fits."""

import numpy as np
import pytest
from astropy.table import Table

from jax_snlc_fit.loader import _build_batch


@pytest.fixture
def tables(fixture_root):
    root = fixture_root / 'snana/salt3_fit_dense'
    return Table.read(root / 'HEAD.FITS.gz', hdu=1), Table.read(root / 'PHOT.FITS.gz', hdu=1)


def build(head, phot, **kwargs):
    return _build_batch(head, phot, True, zp_fluxcal=27.5,
                        sim_mw_colour_law=None, **kwargs)


def test_baseline_without_band_selection_matches_explicit_bands(tables):
    head, phot = tables
    bands = sorted(set(str(b).strip() for b in phot['BAND'] if str(b).strip()))
    implicit = build(head, phot, fluxcal_baseline_correct=True)
    explicit = build(head, phot, fluxcal_baseline_correct=True, fit_bands=bands)
    for name in ('fluxcals', 'fluxcalerrs', 'observation_mask'):
        np.testing.assert_array_equal(getattr(implicit, name), getattr(explicit, name))
    assert implicit.flux_baselines.keys() == explicit.flux_baselines.keys()
    for band in implicit.flux_baselines:
        np.testing.assert_array_equal(implicit.flux_baselines[band],
                                      explicit.flux_baselines[band])


@pytest.mark.parametrize('offset', [0., .0019, .0021])
def test_epoch_ignore_matches_full_photometry_scan(tables, offset):
    head, phot = tables
    ignored = {('absent', 55000., 'g')}
    reject = np.zeros(len(phot), dtype=bool)
    for row in head[:2]:
        start, stop = int(row['PTROBS_MIN']) - 1, int(row['PTROBS_MAX'])
        cid = str(row['SNID']).strip()
        mjd = float(phot['MJD'][start]) + offset
        band = str(phot['BAND'][start]).strip()[-1]
        ignored.add((cid, mjd, band))
        for i in range(start, stop):
            reject[i] = (str(phot['BAND'][i]).strip()[-1] == band
                         and abs(float(phot['MJD'][i]) - mjd) < .002)
    actual = build(head, phot, epoch_ignore=ignored)
    phot['PHOTFLAG'] = reject.astype(int)
    expected = build(head, phot, photflag_mskrej=(1,))
    for name in ('snids', 'mjd_offsets', 'fluxcals', 'fluxcalerrs',
                 'filter_indices', 'observation_mask', 'reference_mjds'):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


def test_absent_epoch_ignore_cid_leaves_observations_unchanged(tables):
    head, phot = tables
    actual = build(head, phot, epoch_ignore={('absent', 55000., 'g')})
    expected = build(head, phot)
    for name in ('snids', 'mjd_offsets', 'fluxcals', 'fluxcalerrs',
                 'filter_indices', 'observation_mask', 'reference_mjds'):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


def test_magcor_corrects_only_the_listed_epochs(tables):
    """A MAGCOR entry dims its own epoch's flux by 10^(-0.4 MAGCOR) and leaves
    every other epoch, and every error, as loaded."""
    head, phot = tables
    corrections = {}
    for row in head[:2]:
        i = int(row['PTROBS_MIN']) - 1
        corrections[(str(row['SNID']).strip(), round(float(phot['MJD'][i]), 3),
                     str(phot['BAND'][i]).strip()[-1])] = .02
    corrected = build(head, phot, sim_magcor_map=corrections)
    plain = build(head, phot)
    for name in ('fluxcalerrs', 'mjd_offsets', 'observation_mask'):
        np.testing.assert_array_equal(getattr(corrected, name), getattr(plain, name))
    changed = np.asarray(corrected.fluxcals) != np.asarray(plain.fluxcals)
    assert changed.sum() == len(corrections)
    np.testing.assert_allclose(np.asarray(corrected.fluxcals)[changed],
                               np.asarray(plain.fluxcals)[changed] * 10 ** (-0.4 * .02))
