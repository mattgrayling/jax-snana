"""Unit tests for the generic FITRES/M0DIF reader."""

import gzip

import numpy as np
import pandas as pd

from jax_snlc_core.io.fitres_reader import read_fitres, read_fitres_multi


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


def test_usecols_subsets(tmp_path):
    fx = read_fitres(_write(tmp_path, _FITRES), usecols=["CID", "zHD", "c"])
    assert set(fx.df.columns) == {"CID", "zHD", "c"}
    assert "x1" not in fx.df.columns


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
    re-emitted byte-for-byte (cat_only). The default read parses to numbers and
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
            "SN: 00123 00456 40.25\nSN: 123 NA 41.5\n")
    plain = _write(tmp_path, text, name="ids.FITRES")
    compressed = _write(tmp_path, text, name="ids.FITRES", gz=True)
    for path in (plain, compressed):
        table = read_fitres(path)
        subset = read_fitres(path, usecols=["CID", "SNID"])
        verbatim = read_fitres(path, as_text=True)
        for read in (table, subset, verbatim):
            assert read.get("CID").tolist() == ["00123", "123"]
            assert read.get("SNID").tolist() == ["00456", "NA"]
        np.testing.assert_allclose(table.get("MU"), [40.25, 41.5])
    merged = read_fitres_multi([plain, compressed], usecols=["CID", "MU"])
    assert merged.get("CID").tolist() == ["00123", "123", "00123", "123"]
    np.testing.assert_allclose(merged.get("MU"), [40.25, 41.5, 40.25, 41.5])
