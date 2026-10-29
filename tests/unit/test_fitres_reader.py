"""Unit tests for the generic FITRES/M0DIF reader."""

import gzip

import numpy as np
import pandas as pd
import pytest

from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import load_snana_dump, read_fitres, read_fitres_multi


_FITRES = """\
# VERSION_PHOTOMETRY:  JAX_TEST
# SNANA_VERSION: v12_01d
# TABLE_NAME:    FITRES
#
VARNAMES:  CID IDSURVEY FIELD zHD x1 c BANDLIST
#
SN: SN2011fe 12 C3 0.123 0.45 -0.02 grizY
SN: 2 12 VOID 0.234 -1.2 0.11 griz
"""


def _write(tmp_path, text, name="test.FITRES", gz=False):
    p = tmp_path / (name + (".gz" if gz else ""))
    if gz:
        with gzip.open(p, "wt") as f:
            f.write(text)
    else:
        p.write_text(text)
    return str(p)


def test_string_columns_stay_strings(tmp_path):
    fx = read_fitres(_write(tmp_path, _FITRES))
    # FIELD and BANDLIST are strings; a non-numeric CID stays a string too.
    assert pd.api.types.is_string_dtype(fx.df["FIELD"])
    assert list(fx.df["FIELD"]) == ["C3", "VOID"]
    assert pd.api.types.is_string_dtype(fx.df["CID"])
    assert list(fx.df["CID"]) == ["SN2011fe", "2"]
    assert list(fx.df["BANDLIST"]) == ["grizY", "griz"]


def test_numeric_columns_are_numeric(tmp_path):
    fx = read_fitres(_write(tmp_path, _FITRES))
    assert np.issubdtype(fx.df["zHD"].dtype, np.floating)
    np.testing.assert_allclose(fx.get("zHD"), [0.123, 0.234])
    assert fx.size == 2


def test_meta_header_parsed(tmp_path):
    fx = read_fitres(_write(tmp_path, _FITRES))
    assert fx.meta["VERSION_PHOTOMETRY"] == "JAX_TEST"
    assert fx.meta["SNANA_VERSION"] == "v12_01d"


def test_gzip_transparent(tmp_path):
    fx = read_fitres(_write(tmp_path, _FITRES, gz=True))
    assert fx.size == 2
    assert list(fx.df["FIELD"]) == ["C3", "VOID"]


def test_has_and_get_default(tmp_path):
    fx = read_fitres(_write(tmp_path, _FITRES))
    assert fx.has("zHD")
    assert not fx.has("MISSING")
    assert fx.get("MISSING", default=None) is None


def test_read_multi_concatenates(tmp_path):
    a = _write(tmp_path, _FITRES, name="a.FITRES")
    b = _write(tmp_path, _FITRES, name="b.FITRES")
    fx = read_fitres_multi([a, b])
    assert fx.size == 4


def test_as_text_preserves_tokens_and_rowkey(tmp_path):
    """as_text keeps every value as written and records the row key, so rows can be
    re-emitted without changing value tokens (cat_only). The default read parses to numbers and
    drops the key."""
    path = _write(tmp_path, _FITRES)
    fx = read_fitres(path, as_text=True)
    assert list(fx.get("zHD")) == ["0.123", "0.234"]
    assert list(fx.rowkey) == ["SN:", "SN:"]
    assert list(fx.varnames) == ["CID", "IDSURVEY", "FIELD", "zHD", "x1", "c",
                                 "BANDLIST"]

    parsed = read_fitres(path)
    assert parsed.rowkey is None
    np.testing.assert_allclose(parsed.get("zHD"), [0.123, 0.234])


def test_as_text_keeps_trailing_zeros(tmp_path):
    """The point of as_text: 0.4500 must not come back as 0.45."""
    text = _FITRES.replace("0.45 -0.02", "0.4500 -0.02")
    fx = read_fitres(_write(tmp_path, text), as_text=True)
    assert fx.get("x1")[0] == "0.4500"



def test_identifiers_survive_numeric_tables_and_concatenation(tmp_path):
    text = ("VARNAMES: CID SNID MU\n"
            "SN: 00123 00456 40.25\nSN: 123 456 41.5\n")
    plain = _write(tmp_path, text, name="ids.FITRES")
    compressed = _write(tmp_path, text, name="ids.FITRES", gz=True)
    for path in (plain, compressed):
        table = read_fitres(path)
        verbatim = read_fitres(path, as_text=True)
        for read in (table, verbatim):
            assert read.get("CID").tolist() == ["00123", "123"]
            assert read.get("SNID").tolist() == ["00456", "456"]
        np.testing.assert_allclose(table.get("MU"), [40.25, 41.5])
    merged = read_fitres_multi([plain, compressed])
    assert merged.get("CID").tolist() == ["00123", "123", "00123", "123"]
    np.testing.assert_allclose(merged.get("MU"), [40.25, 41.5, 40.25, 41.5])


def test_values_past_the_last_varname_are_ignored(tmp_path):
    """SNTABLE_READ_EXEC_TEXT reads a row's first NVAR values and ignores the
    rest (sntools_output_text.c:1265-1276), so each column keeps its name."""
    path = _write(tmp_path, "VARNAMES: CID A B\nSN: 1 2.0 3.0 9\nSN: 2 4.0 5.0\n")
    for table in (read_fitres(path), read_fitres(path, as_text=True)):
        assert table.varnames == ["CID", "A", "B"]
        assert table.get("CID").tolist() == ["1", "2"]
        np.testing.assert_array_equal(np.asarray(table.get("B"), dtype=float), [3.0, 5.0])


def test_row_with_fewer_values_than_varnames_aborts(tmp_path):
    """SNANA aborts on a row short of its VARNAMES (sntools_output_text.c:
    1278-1282), whether the table is read as numbers or as tokens, and a
    nan token in the last column is not taken for a short row."""
    for name, rows in (("last.DUMP", "SN: 1 2.0 3.0\nSN: 2 4.0\n"),
                       ("first.DUMP", "SN: 1 2.0\nSN: 2 4.0 5.0\n"),
                       ("every.DUMP", "SN: 1 2.0\nSN: 2 4.0\n")):
        short = _write(tmp_path, "VARNAMES: CID A B\n" + rows, name=name)
        for read in (read_fitres, lambda p: read_fitres(p, as_text=True), load_snana_dump):
            with pytest.raises(SNANAError, match="fewer values than VARNAMES"):
                read(short)
    with_nan = _write(tmp_path, "VARNAMES: CID A B\nSN: 1 2.0 nan\nSN: 2 4.0 5.0\n")
    assert np.isnan(read_fitres(with_nan).get("B")[0])


def test_only_snana_row_keys_are_rows(tmp_path):
    """A line under VARNAMES whose first word is not a row key, such as a
    stray VPEC_INFO: line, is not a row (validRowKey_TEXT,
    sntools_output_text.c:1386): its length is not checked and its words do
    not change the type of a column."""
    path = _write(tmp_path, "VARNAMES: CID A B\nSN: 1 2.0 3.0\nVPEC_INFO: 250.0\n"
                            "NOTE: hello world\nGAL: 2 4.0 5.0\nOBS: 3 6.0 7.0\n")
    table = read_fitres(path)
    assert table.get("CID").tolist() == ["1", "2", "3"]
    np.testing.assert_array_equal(table.get("A"), [2.0, 4.0, 6.0])
    assert read_fitres(path, as_text=True).rowkey.tolist() == ["SN:", "GAL:", "OBS:"]


@pytest.mark.parametrize('gz', [False, True])
@pytest.mark.parametrize('as_text', [False, True])
@pytest.mark.parametrize('body', ['', 'SN: 001 0.123\n'])
def test_empty_and_single_row_tables(tmp_path, gz, as_text, body):
    """A table with no rows reads as empty, and a single row keeps its tokens."""
    table = read_fitres(_write(tmp_path, 'VARNAMES: CID zHD\n' + body, gz=gz),
                       as_text=as_text)
    assert table.size == int('SN:' in body)
    if table.size:
        assert table.df['CID'].iloc[0] == '001'
        assert float(table.df['zHD'].iloc[0]) == .123


def test_missing_varnames_aborts(tmp_path):
    """A file without a VARNAMES line is not a table."""
    with pytest.raises(SNANAError, match='No VARNAMES line'):
        read_fitres(_write(tmp_path, '# no header\n'))


@pytest.mark.parametrize('as_text', [False, True])
def test_original_lines_keep_comments_and_surplus_fields_out_of_values(tmp_path, as_text):
    """Pandas reads original whitespace and ignores comments and surplus tokens."""
    path = _write(tmp_path, 'VARNAMES: CID A B\n'
                            '  SN:\t001\t2.0  3.0  extra # comment\n'
                            'SN: 002 4.0 5.0#comment\n')
    table = read_fitres(path, as_text=as_text)
    assert table.get('CID').tolist() == ['001', '002']
    np.testing.assert_array_equal(np.asarray(table.get('B'), dtype=float), [3., 5.])
