"""Unit tests for the KcorReader (uses bundled JAX_TEST_LSST.fits)."""

import numpy as np
import pytest
from astropy.io import fits

from jax_snlc_core.filters import KcorReader
from jax_snlc_core.logging import SNANAError


def test_leakage_cut_drops_transmission_below_the_cut(lsst_kcor_path):
    """LEAKAGE_CUT drops each bin below the cut times the filter's peak, as
    kcor.exe's cutOOBTrans_filter does (kcor.c:2525), and keeps the rest as
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


def _rewritten_kcor(source, path, edit):
    """A copy of the kcor file ``source`` at ``path``, changed by ``edit``."""
    with fits.open(f"{source}.gz") as hdul:
        edit(hdul)
        hdul.writeto(path)
    return str(path)


def _replace_table(hdul, name, columns):
    index = hdul.index_of(name)
    hdul[index] = fits.BinTableHDU.from_columns(columns)
    hdul[index].header['EXTNAME'] = name


def _renamed_columns(hdu):
    return [fits.Column(name=f'C{i}', format=column.format, array=column.array)
            for i, column in enumerate(hdu.columns)]


def test_kcor_tables_are_read_by_position(tmp_path, lsst_kcor_path):
    """SNANA reads every kcor table by column position and takes the filter
    and primary names from the FILTnnn and PRIMnnn cards
    (sntools_calib.c:280-312, 552-611, 1404-1429), so column names, extra
    ZPoff columns (v12_02f adds 'Primary synthetic mag'), the case of a
    table name and names longer than the 20-character ZPoff column, even
    names alike in their first 20 characters, do not change what is read."""
    long_names = ["LSST-wide-survey-band-u", "LSST-wide-survey-band-g"]

    def edit(hdul):
        _replace_table(hdul, 'ZPoff', _renamed_columns(hdul['ZPoff']) + [
            fits.Column(name='Primary synthetic mag', format='E', array=np.ones(6))])
        for i, name in enumerate(long_names):
            hdul[0].header[f'FILT{i + 1:03d}'] = name
            hdul['ZPoff'].data['C0'][i] = name[:20]
        for table in ('FilterTrans', 'PrimarySED'):
            _replace_table(hdul, table, _renamed_columns(hdul[table]))
        for hdu in hdul[1:]:
            hdu.name = hdu.name.lower()

    stored = KcorReader(str(lsst_kcor_path))
    renamed = KcorReader(_rewritten_kcor(lsst_kcor_path, tmp_path / "kcor.fits", edit))
    assert renamed.filter_names == long_names + stored.filter_names[2:]
    for old, new in zip(stored.filter_names, renamed.filter_names):
        for field in ('wavelength', 'transmission'):
            np.testing.assert_array_equal(stored.filter_data[old][field],
                                          renamed.filter_data[new][field])
        assert stored.filter_primary[old] == renamed.filter_primary[new]
        assert stored.filter_primary_mag[old] == renamed.filter_primary_mag[new]
        assert stored.filter_zpoff_snpot[old] == renamed.filter_zpoff_snpot[new]
        assert stored.filter_survey[old] == renamed.filter_survey[new]
    np.testing.assert_array_equal(stored.get_primary_sed('AB'), renamed.get_primary_sed('AB'))


def _drop_last_zpoff_column(hdul):
    _replace_table(hdul, 'ZPoff', hdul['ZPoff'].columns[:4])


def _swap_zpoff_rows(hdul):
    names = hdul['ZPoff'].data['Filter Name']
    names[0], names[1] = names[1], names[0]


@pytest.mark.parametrize("edit, message", [
    (lambda hdul: hdul[0].header.remove('NFILTERS'), "has no NFILTERS header card"),
    (lambda hdul: hdul[0].header.remove('PRIM001'), "has no PRIM001 header card"),
    (lambda hdul: hdul.pop(hdul.index_of('PrimarySED')), "has no PrimarySED table"),
    (_drop_last_zpoff_column, "ZPoff table has no column for SNpot zero-point offset"),
    (_swap_zpoff_rows, "ZPoff row 1 is for LSST-g, but FILT001 is LSST-u"),
])
def test_malformed_kcor_aborts(tmp_path, lsst_kcor_path, edit, message):
    """A kcor file SNANA cannot read is an error naming what is missing, and
    ZPoff rows must follow the FILTnnn order that SNANA assumes."""
    path = _rewritten_kcor(lsst_kcor_path, tmp_path / "kcor.fits", edit)
    with pytest.raises(SNANAError, match=message):
        KcorReader(path)


def test_filter_letters_keep_their_case():
    """SNANA's filter alphabet holds b and B as distinct filters
    (FILTERSTRING_DEFAULT, sntools.h:163; INTFILTER compares the letter
    exactly, sntools.c:10898), and a combined low-z kcor carries both. The
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
