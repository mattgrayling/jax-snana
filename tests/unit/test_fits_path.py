"""resolve_fits_path: CFITSIO-style .FITS -> .FITS.gz name resolution, and
read_fits_table: a SNANA binary table held once.

SNANA gzips its FITS output by default and reads via CFITSIO, which given
``X.FITS`` transparently opens ``X.FITS.gz``. astropy does not, so the loader
routes every read through ``resolve_fits_path``.
"""

import gzip

import numpy as np
import pytest
from astropy.io import fits

from jax_snlc_core.io.fits_path import read_fits_table, resolve_fits_path
from jax_snlc_core.utils.error_handling import SNANAError


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
def test_read_fits_table_matches_astropy_and_holds_one_native_copy(tmp_path, gzipped):
    path = tmp_path / "X_HEAD.FITS"
    _snana_style_file(path)
    if gzipped:
        gz = path.with_name(path.name + ".gz")
        gz.write_bytes(gzip.compress(path.read_bytes()))
        path.unlink()
        path = gz
    primary, table = read_fits_table(path)
    assert primary["DATATYPE"] == "SIM_SNANA"
    reference = fits.getdata(path, 1).view(np.ndarray)   # the bytes as stored
    assert table.dtype.names == reference.dtype.names
    for name in table.dtype.names:
        assert table[name].dtype.isnative
        np.testing.assert_array_equal(table[name], reference[name])
    # every column is a view of the one buffer the rows were read into
    assert all(np.shares_memory(table[name], table) for name in table.dtype.names)


def test_read_fits_table_refuses_a_scaled_column(tmp_path):
    path = tmp_path / "X_HEAD.FITS"
    _snana_style_file(path, scaled=True)
    with pytest.raises(SNANAError, match="scaled columns \\['CCDNUM'\\]"):
        read_fits_table(path)


def test_read_fits_table_aborts_on_a_truncated_file(tmp_path):
    path = tmp_path / "X_HEAD.FITS"
    _snana_style_file(path)
    whole = path.read_bytes()
    path.write_bytes(whole[:-4000])   # past the block padding, into the rows
    with pytest.raises(SNANAError, match="bytes short of its table"):
        read_fits_table(path)


def test_open_text_resolves_compressed_name(tmp_path):
    from jax_snlc_core.io.fits_path import open_text

    path = tmp_path / 'table.TEXT'
    compressed = tmp_path / 'table.TEXT.gz'
    compressed.write_bytes(gzip.compress(b'compressed\n'))
    for name in (path, compressed):
        with open_text(name) as stream:
            assert stream.read() == 'compressed\n'
    path.write_text('plain\n')
    with open_text(path) as stream:
        assert stream.read() == 'plain\n'
