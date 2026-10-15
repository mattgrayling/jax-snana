"""Unit tests for the SNANA SIMGEN_DUMP reader."""

import numpy as np
import pytest

from jax_snlc_core.io.snana_dump_reader import SnanaDumpData, load_snana_dump
from jax_snlc_core.utils.error_handling import SNANAError


def _write_dump(tmp_path, lines):
    p = tmp_path / "DUMP"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def test_load_basic_dump(tmp_path):
    path = _write_dump(tmp_path, [
        "# header",
        "VARNAMES: CID GENZ MU CUTMASK",
        "SN: 1 0.10 38.5 4095",
        "SN: 2 0.20 39.7 4095",
        "SN: 3 0.30 40.5 4093",
    ])
    dump = load_snana_dump(path)
    assert isinstance(dump, SnanaDumpData)
    assert dump.size == 3
    np.testing.assert_array_equal(dump.columns["CID"], [1, 2, 3])
    np.testing.assert_allclose(dump.columns["GENZ"], [0.10, 0.20, 0.30])
    np.testing.assert_array_equal(dump.columns["CUTMASK"], [4095, 4095, 4093])


def test_load_skips_comments_and_blanks(tmp_path):
    path = _write_dump(tmp_path, [
        "# top comment",
        "",
        "VARNAMES: CID GENZ",
        "# inline comment",
        "SN: 7 0.1",
        "",
        "SN: 8 0.2",
    ])
    dump = load_snana_dump(path)
    assert dump.size == 2


def test_load_handles_non_numeric_values_as_nan(tmp_path):
    path = _write_dump(tmp_path, [
        "VARNAMES: CID FIELD GENZ",
        "SN: 1 C1 0.10",
        "SN: 2 X3 0.20",
    ])
    dump = load_snana_dump(path)
    assert np.all(np.isnan(dump.columns["FIELD"]))
    np.testing.assert_allclose(dump.columns["GENZ"], [0.10, 0.20])


def test_dedup_cid_refuses_a_tie(tmp_path):
    """Two rows of one CID at the same CUTMASK cannot be told apart, so the
    reader aborts and names the keys that make CIDs unique."""
    path = _write_dump(tmp_path, [
        "VARNAMES: CID GENZ CUTMASK",
        "SN: 5 0.10 4095",
        "SN: 5 0.30 4095",
    ])
    with pytest.raises(SNANAError, match="CIDRAN_MIN"):
        load_snana_dump(path, dedup_cid=True)


def test_dedup_cid_keeps_highest_cutmask(tmp_path):
    path = _write_dump(tmp_path, [
        "VARNAMES: CID GENZ CUTMASK",
        "SN: 5 0.10 0",        # prefiltered
        "SN: 5 0.10 4095",     # passing — should win
        "SN: 6 0.20 4095",
    ])
    dump = load_snana_dump(path, dedup_cid=True)
    assert dump.size == 2
    cids = dump.columns["CID"].astype(int).tolist()
    cms = dump.columns["CUTMASK"].astype(int).tolist()
    assert sorted(cids) == [5, 6]
    # CID 5's surviving row must be the 4095 one.
    cm_for_5 = cms[cids.index(5)]
    assert cm_for_5 == 4095
