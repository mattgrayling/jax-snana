"""resolve_fits_path: CFITSIO-style .FITS -> .FITS.gz name resolution, and
read_fits_table: decoded SNANA binary tables.

SNANA gzips its FITS output by default and reads via CFITSIO, which given
``X.FITS`` transparently opens ``X.FITS.gz``. astropy does not, so the loader
routes every read through ``resolve_fits_path``.
"""

import gzip

import numpy as np
import pytest
from astropy.io import fits

from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import read_fits_table, resolve_fits_path


def test_resolve_prefers_plain_then_gz_then_passthrough(tmp_path):
    plain = tmp_path / "a_HEAD.FITS"
    gz = tmp_path / "b_HEAD.FITS"

    # plain present -> returned as-is
    plain.write_bytes(b"x")
    assert resolve_fits_path(str(plain)) == str(plain)

    # only .gz present -> the .gz path is returned
    (tmp_path / "b_HEAD.FITS.gz").write_bytes(b"x")
    assert resolve_fits_path(str(gz)) == str(gz) + ".gz"

    # both present -> plain wins (deterministic, no ambiguity abort)
    plain.with_suffix(".FITS.gz").write_bytes(b"x")
    assert resolve_fits_path(str(plain)) == str(plain)

    # neither present -> passed through unchanged (caller raises naturally)
    missing = tmp_path / "none_HEAD.FITS"
    assert resolve_fits_path(str(missing)) == str(missing)


def _snana_style_file(path, n_rows=5000, scaled=False):
    """A HEAD-like file as SNANA writes it: an empty primary HDU carrying
    keywords and one binary table with numeric and string columns."""
    columns = [
        fits.Column(name="SNID", format="16A",
                    array=np.array([f"{i:d}".ljust(16) for i in range(n_rows)])),
        fits.Column(name="MJD", format="D", array=np.linspace(60000.0, 60100.0, n_rows)),
        fits.Column(name="NOBS", format="J", array=np.arange(n_rows, dtype=np.int32)),
        fits.Column(name="FLUX", format="E", array=np.random.default_rng(1).normal(size=n_rows)),
    ]
    if scaled:
        columns.append(fits.Column(name="CCDNUM", format="I", bzero=32768,
                                   array=np.arange(n_rows, dtype=np.uint16)))
    primary = fits.PrimaryHDU()
    primary.header["DATATYPE"] = "SIM_SNANA"
    fits.HDUList([primary, fits.BinTableHDU.from_columns(columns)]).writeto(path)


@pytest.mark.parametrize("gzipped", [False, True])
def test_read_fits_table_matches_decoded_astropy_columns(tmp_path, gzipped):
    """Decoded values, including logical flags and shapes, survive file closure."""
    path = tmp_path / "X_HEAD.FITS"
    _snana_style_file(path)
    with fits.open(path, mode='update') as hdus:
        n = len(hdus[1].data)
        columns = hdus[1].columns + fits.ColDefs([
            fits.Column(name='FLAG', format='L', array=np.arange(n) % 2 == 0),
            fits.Column(name='BITS', format='9X',
                        array=np.arange(n * 9).reshape(n, 9) % 3 == 0),
            fits.Column(name='GRID', format='6E', dim='(3,2)',
                        array=np.arange(n * 6, dtype=np.float32).reshape(n, 2, 3)),
        ])
        hdus[1] = fits.BinTableHDU.from_columns(columns)
    if gzipped:
        gz = path.with_name(path.name + ".gz")
        gz.write_bytes(gzip.compress(path.read_bytes()))
        path.unlink()
        path = gz
    primary, table = read_fits_table(path)
    assert primary["DATATYPE"] == "SIM_SNANA"
    reference = fits.getdata(path, 1, character_as_bytes=True)
    assert table.dtype.names == reference.dtype.names
    for name in table.dtype.names:
        assert table[name].dtype.isnative
        np.testing.assert_array_equal(table[name], reference[name])
    np.testing.assert_array_equal(table['FLAG'][:2], [True, False])
    assert table['GRID'].shape == (5000, 2, 3)


def test_read_fits_table_decodes_scaled_columns(tmp_path):
    """Unsigned FITS integers use TZERO and must be returned as physical values."""
    path = tmp_path / "X_HEAD.FITS"
    _snana_style_file(path, scaled=True)
    _, table = read_fits_table(path)
    np.testing.assert_array_equal(table['CCDNUM'], np.arange(5000))


def test_read_fits_table_aborts_on_a_truncated_file(tmp_path):
    """A truncated binary table produces a named reader error."""
    path = tmp_path / "X_HEAD.FITS"
    _snana_style_file(path)
    whole = path.read_bytes()
    path.write_bytes(whole[:-4000])   # past the block padding, into the rows
    with pytest.raises(SNANAError, match="Cannot read"):
        read_fits_table(path)


def test_open_text_resolves_compressed_name(tmp_path):
    from jax_snlc_core.snana_files import open_text

    path = tmp_path / 'table.TEXT'
    compressed = tmp_path / 'table.TEXT.gz'
    compressed.write_bytes(gzip.compress(b'compressed\n'))
    for name in (path, compressed):
        with open_text(name) as stream:
            assert stream.read() == 'compressed\n'
    path.write_text('plain\n')
    with open_text(path) as stream:
        assert stream.read() == 'plain\n'


@pytest.mark.parametrize('gzipped', [False, True])
def test_read_fits_table_preserves_scaled_and_unscaled_values(tmp_path, gzipped):
    """Byte-order conversion must not alter Astropy's separately scaled values."""
    path = tmp_path / ('scaled.FITS.gz' if gzipped else 'scaled.FITS')
    columns = [fits.Column(name='RAW', format='J', array=[-12, 0, 42]),
               fits.Column(name='SCALED', format='J', array=[-12, 0, 42])]
    hdu = fits.BinTableHDU.from_columns(columns)
    hdu.header['TSCAL2'] = 0.5
    hdu.header['TZERO2'] = 100.
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)
    _, table = read_fits_table(path)
    np.testing.assert_array_equal(table['RAW'], [-12, 0, 42])
    np.testing.assert_array_equal(table['SCALED'], [94., 100., 121.])
    assert all(column.dtype.isnative for column in table.itercols())


def test_read_fits_table_handles_empty_tables(tmp_path):
    """An empty result still has its declared columns and shapes."""
    path = tmp_path / 'empty.FITS'
    _snana_style_file(path, n_rows=0)
    _, table = read_fits_table(path)
    assert len(table) == 0
    assert table.colnames == ['SNID', 'MJD', 'NOBS', 'FLUX']


@pytest.mark.parametrize('gzipped', [False, True])
def test_unsigned_columns_preserve_exact_values(tmp_path, gzipped):
    """Unsigned FITS integers retain their dtype, including IDs above 2**63."""
    values = {
        'SNID': np.array([2**63 + 1, 2**63 + 2, 2**64 - 1], dtype=np.uint64),
        'PHOTFLAG': np.array([0, 2**31, 2**32 - 1], dtype=np.uint32),
        'CCDNUM': np.array([0, 2**15, 2**16 - 1], dtype=np.uint16),
    }
    columns = [fits.Column(name=name, format=fmt, bzero=zero, array=values[name])
               for name, fmt, zero in [('SNID', 'K', 2**63),
                                       ('PHOTFLAG', 'J', 2**31),
                                       ('CCDNUM', 'I', 2**15)]]
    path = tmp_path / ('unsigned.FITS.gz' if gzipped else 'unsigned.FITS')
    fits.BinTableHDU.from_columns(columns).writeto(path)
    _, table = read_fits_table(path)
    for name, expected in values.items():
        np.testing.assert_array_equal(table[name], expected)
        assert table[name].dtype == expected.dtype
