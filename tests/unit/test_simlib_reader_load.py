"""Unit tests that load the bundled toy SIMLIB through ``read_simlib``.

This exercises the parse path that the unit-level helper tests don't —
SURVEY/FILTERS header detection, LIBID/RA/DEC bookkeeping, NEXPOSE token
parsing, NEA computation per observation row, end-to-end array shapes.
"""

import re

import pytest

from jax_snlc_core.logging import SNANAError
from jax_snlc_sim.simlib import read_simlib


def test_simlib_reader_loads_two_libids(lsst_simlib_path):
    reader = read_simlib(str(lsst_simlib_path))
    assert len(reader.fields) == 2
    assert reader.survey_name == "LSST"
    assert reader.filter_names == ["u", "g", "r", "i", "z", "Y"]


def test_simlib_nexpose_cycles_through_expected_values(lsst_simlib_path):
    """The toy SIMLIB cycles NEXPOSE through {1, 2, 4}."""
    reader = read_simlib(str(lsst_simlib_path))
    nexposes = reader.fields[0].nexposes
    unique = sorted(set(int(v) for v in nexposes))
    assert set(unique) == {1, 2, 4}


def test_simlib_header_keys_parse_in_any_order(tmp_path, lsst_simlib_path):
    """SNANA scans header keys from the token stream, so NOBS may lead the
    line ('NOBS: 2  RA: 58.1  DEC: -32.0', the LSST DC2 layout) or the keys
    may sit on separate lines; RA/DEC/MWEBV must be read either way."""
    base = open(str(lsst_simlib_path)).read()
    header = base[:re.search(r'^LIBID:', base, re.M).start()]
    obs = ("S: 60132.42 100*1 g 1.00 4.00 6.00 1.00 0.00 0.00 30.470 0.005 99.0\n"
           "S: 60145.37 101*1 r 1.00 4.00 8.00 1.15 0.00 0.00 30.500 0.005 99.0\n")
    sml = tmp_path / "order.SIMLIB"
    sml.write_text(
        header
        + "LIBID: 1\nNOBS: 2  RA: 58.148601  DEC: -31.990482  MWEBV: 0.012  PIXSIZE: 0.2\n"
        + obs + "END_LIBID: 1\n"
        + "LIBID: 2\nRA: 55.06\nDECL: -32.39\nNOBS: 2\nPIXSIZE: 0.2\n"
        + obs + "END_LIBID: 2\nEND_OF_SIMLIB:\n")
    reader = read_simlib(str(sml), require_docana=False)
    f1, f2 = reader.fields
    assert abs(f1.ra - 58.148601) < 1e-6 and abs(f1.dec + 31.990482) < 1e-6
    assert abs(f1.mwebv - 0.012) < 1e-9
    assert abs(f2.ra - 55.06) < 1e-6 and abs(f2.dec + 32.39) < 1e-6


def test_simlib_entry_without_coordinates(tmp_path, lsst_simlib_path):
    """An entry with no RA/DEC keeps SNANA's 9999 (init_SIMLIB_HEADER), which
    fails GENRANGE_RA/DEC; MWEBV is not reset per entry, so one without it
    carries the previous entry's value."""
    base = open(str(lsst_simlib_path)).read()
    header = base[:re.search(r'^LIBID:', base, re.M).start()]
    obs = "S: 60132.42 100*1 g 1.00 4.00 6.00 1.00 0.00 0.00 30.470 0.005 99.0\n"
    sml = tmp_path / "nocoords.SIMLIB"
    sml.write_text(
        header
        + "LIBID: 1\nNOBS: 1  RA: 58.1  DEC: -32.0  MWEBV: 0.012  PIXSIZE: 0.2\n"
        + obs + "END_LIBID: 1\n"
        + "LIBID: 2\nNOBS: 1  PIXSIZE: 0.2\n"
        + obs + "END_LIBID: 2\nEND_OF_SIMLIB:\n")
    reader = read_simlib(str(sml), require_docana=False, cache_dir=tmp_path)
    f2 = reader.fields[1]
    assert f2.ra == 9999.0 and f2.dec == 9999.0
    assert abs(f2.mwebv - 0.012) < 1e-9


def test_global_header_field_is_each_entrys_default(tmp_path,
                                                    lsst_simlib_path):
    """A FIELD in the global header, which ends at BEGIN, starts every
    entry's header (SIMLIB_readGlobalHeader_TEXT, snlc_sim.c:18436, :18529;
    init_SIMLIB_HEADER, :21898); an entry's own FIELD replaces it, and a
    FIELD after BEGIN is not global."""
    base = open(str(lsst_simlib_path)).read()
    obs = "S: 60132.42 100*1 g 1.00 4.00 6.00 1.00 0.00 0.00 30.470 0.005 99.0\n"
    entries = ("LIBID: 1\nNOBS: 1  RA: 58.1  DEC: -32.0  PIXSIZE: 0.2\n"
               + obs + "END_LIBID: 1\n"
               + "LIBID: 2\nNOBS: 1  RA: 55.1  DEC: -32.4  PIXSIZE: 0.2  "
               + "FIELD: DDF\n" + obs + "END_LIBID: 2\nEND_OF_SIMLIB:\n")
    header = base[:re.search(r'^BEGIN', base, re.M).start()]

    def fields(text, name):
        sml = tmp_path / name
        sml.write_text(text)
        reader = read_simlib(str(sml), require_docana=False,
                             cache_dir=tmp_path)
        return [f.field_name for f in reader.fields]

    assert fields(header + "FIELD: WFD\nBEGIN LIBGEN\n" + entries,
                  "global.SIMLIB") == ["WFD", "DDF"]
    assert fields(header + "BEGIN LIBGEN\nFIELD: WFD\n" + entries,
                  "late.SIMLIB") == ["FIELD_NONAME", "DDF"]

def test_docana_key_is_the_first_token_of_the_file(lsst_simlib_path, tmp_path):
    """SNANA opens a SIMLIB through snana_openTextFile, whose DOCANA check is
    one fscanf("%s") (check_openFile_docana, sntools.c): blank lines before
    the key are skipped, but any other word ahead of it fails."""
    base = open(str(lsst_simlib_path)).read()
    passing = tmp_path / "blank_first.SIMLIB"
    passing.write_text("\n\n  \n\n\n" + base)
    assert len(read_simlib(str(passing), cache_dir=tmp_path).fields) == 2

    preceded = tmp_path / "preceded.SIMLIB"
    preceded.write_text("SURVEY: LSST\n" + base)
    with pytest.raises(SNANAError, match="DOCUMENTATION"):
        read_simlib(str(preceded), cache_dir=tmp_path)


def test_texpose_field_header_key_is_refused(tmp_path, lsst_simlib_path):
    """TEXPOSE(<field>) gives every epoch in a matching field a per-band
    exposure time that SNANA writes to the PHOT TEXPOSE column
    (get_TEXPOSE, snlc_sim.c:21945); jax has no such column value, so the
    key cannot pass silently."""
    base = open(str(lsst_simlib_path)).read()
    start = re.search(r'^LIBID:', base, re.M).start()
    sml = tmp_path / "texpose.SIMLIB"
    sml.write_text(base[:start] + "TEXPOSE(WFD): 30 30 30 30 30 30\n"
                   + base[start:])
    with pytest.raises(SNANAError, match=r"TEXPOSE\(WFD\)"):
        read_simlib(str(sml), require_docana=False, cache_dir=tmp_path)
