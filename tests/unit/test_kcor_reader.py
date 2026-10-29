"""Unit tests for the KcorReader (uses bundled JAX_TEST_LSST.fits)."""

import numpy as np
import pytest

from jax_snlc_core.filters import KcorReader
from jax_snlc_core.logging import SNANAError


def test_leakage_cut_drops_transmission_below_the_cut(lsst_kcor_path):
    """LEAKAGE_CUT drops each bin below the cut times the filter's peak, as
    kcor.exe's cutOOBTrans_filter does (kcor.c:2513), and keeps the rest as
    stored; the default of 0 keeps every bin."""
    stored = KcorReader(str(lsst_kcor_path))
    cut = KcorReader(str(lsst_kcor_path), leakage_cut=1e-3)
    dropped = 0
    for name in stored.get_filter_names():
        before = np.asarray(stored.filter_data[name]['transmission'])
        after = np.asarray(cut.filter_data[name]['transmission'])
        below = before < 1e-3 * before.max()
        assert np.all(after[below] == 0.0), name
        assert np.array_equal(after[~below], before[~below]), name
        dropped += int(np.sum(below & (before > 0.0)))
    assert dropped, "the fixture filters carry no leakage for the cut to drop"


def test_kcor_reader_zpoff_missing_columns_fatal(tmp_path):
    """SNANA reads the ZPT table positionally for every filter and treats a
    read failure as fatal (read_calib_zpoff, sntools_calib.c:563-611); a
    ZPoff table without the calibration columns aborts instead of leaving
    silent zero defaults."""
    from astropy.io import fits

    cols = [
        fits.Column(name='Filter Name', format='20A', array=np.array(['X-g'])),
        fits.Column(name='Primary Name', format='20A', array=np.array(['AB'])),
    ]
    zpoff_hdu = fits.BinTableHDU.from_columns(cols)
    # kcor.exe writes the extension name in mixed case
    zpoff_hdu.header['EXTNAME'] = 'ZPoff'
    hdul = fits.HDUList([fits.PrimaryHDU(), zpoff_hdu])
    path = tmp_path / "bad_zpoff.fits"
    hdul.writeto(str(path))
    with pytest.raises(SNANAError, match="Primary Mag"):
        KcorReader(str(path))


def test_kcor_reader_filtertrans_without_wavelength_fatal(tmp_path):
    """SNANA reads the FilterTrans table and aborts on a read failure; a table
    with no wavelength column aborts rather than leaving no filters."""
    from astropy.io import fits

    trans = fits.BinTableHDU.from_columns(
        [fits.Column(name='X-g', format='E', array=np.ones(3))])
    trans.header['EXTNAME'] = 'FilterTrans'
    path = tmp_path / "no_wave.fits"
    fits.HDUList([fits.PrimaryHDU(), trans]).writeto(str(path))
    with pytest.raises(SNANAError, match="wavelength"):
        KcorReader(str(path))


def test_filter_letters_keep_their_case():
    """SNANA's filter alphabet holds b and B as distinct filters
    (FILTERSTRING_DEFAULT, sntools.h:162; INTFILTER compares the letter
    exactly, sntools.c:11205), and a combined low-z kcor carries both. The
    band table, the SIMLIB letter lookup and the trigger logic must keep
    the case, or one band takes the other's observations."""
    from types import SimpleNamespace

    from jax_snlc_sim.builder import _build_filter_lookup
    from jax_snlc_sim.detection import parse_trigger_logic

    kcor = SimpleNamespace(get_filter_names=lambda: ["CSP-b", "CFA41-B", "CSP-g"])
    names, table = _build_filter_lookup(kcor)
    assert names == ["CSP-b", "CFA41-B", "CSP-g"]
    assert [table[c] for c in ("b", "B", "g")] == [0, 1, 2]
    assert table["CFA41-B"] == 1
    assert "V" not in table

    assert parse_trigger_logic("bB+Bg", table) == [0b011, 0b110]
