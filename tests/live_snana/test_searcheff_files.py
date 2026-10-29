"""Optional parsing check against the installed SNANA efficiency maps."""

import os
from pathlib import Path

import pytest

from jax_snlc_sim.spectroscopic_efficiency import read_searcheff_maps


def test_all_real_files_parse():
    """All SEARCHEFF_SPEC* / SEARCHEFF_zHOST* files in $SNDATA_ROOT parse.
    A file without a VARNAMES line holds no map (read_searcheff_map,
    sntools_trigger.c:1370-1417)."""
    searcheff_dir = Path(os.environ["SNDATA_ROOT"]) / "models" / "searcheff"
    if not searcheff_dir.is_dir():
        pytest.skip(f"{searcheff_dir} not present")
    n = 0
    for f in sorted(searcheff_dir.iterdir()):
        if f.name.startswith("SEARCHEFF_SPEC") or f.name.startswith("SEARCHEFF_zHOST"):
            maptype = "SPEC" if f.name.startswith("SEARCHEFF_SPEC") else "zHOST"
            maps, _ = read_searcheff_maps(str(f), maptype, "")
            if "VARNAMES" not in f.read_text():
                assert maps == []
            n += 1
    assert n >= 10
