"""Parity vs SALT2mu.exe for cat_only (concatenate the datafiles and exit).

Every line of the output is compared except SNANA_VERSION, which carries each
program's own build string. See the fixture README for the reference commands.
"""

from jax_snlc_core.io.fits_path import open_text

import contextlib
import os

import pytest

from jax_salt2mu.config_parser import parse_bbc_input
from jax_salt2mu.output import write_cat_fitres
from jax_snlc_core.io.fitres_reader import read_fitres

_DIR = os.path.abspath("tests/fixtures/snana/bbc_des_beams")
_A = "catA.FITRES"
_B = "catB.FITRES"
_HOSTLIB = "docana.HOSTLIB"



def _run_cat(tmp_path, paths, prescale=1):
    """Run from the fixture directory, as the reference commands in the README do:
    the header echoes each input path verbatim, so bare names keep the references
    independent of where the fixture lives."""
    config = parse_bbc_input(None, overrides=[f"datafile={','.join(paths)}"])
    config.cat_only = True
    config.cat_prescale = prescale
    out = str(tmp_path / "cat.FITRES")
    with contextlib.chdir(_DIR):
        tables = [read_fitres(p, as_text=True) for p in paths]
        n = write_cat_fitres(out, tables, paths, config)
    return out, n


def _lines(path):
    """Every line but the version stamp, which is the program's own build string."""
    return [ln for ln in open_text(path).read().splitlines() if "SNANA_VERSION" not in ln]


@pytest.mark.parametrize("paths, prescale, reference, nrow", [
    ([_A], 1, "SALT2mu_cat_single.FITRES", 20),
    ([_A], 3, "SALT2mu_cat_prescale3.FITRES", 6),
    ([_A, _B], 1, "SALT2mu_cat_two.FITRES", 35),
    ([_HOSTLIB], 4, "SALT2mu_cat_docana.FITRES", 3),
    # a non-row key among the data is skipped, and does not advance the prescale
    (["stray.FITRES"], 2, "SALT2mu_cat_stray.FITRES", 10),
])
def test_cat_matches_salt2mu(tmp_path, paths, prescale, reference, nrow):
    out, n = _run_cat(tmp_path, paths, prescale)
    assert n == nrow
    assert _lines(out) == _lines(f"{_DIR}/{reference}")


def test_prescale_counts_across_files(tmp_path):
    """The counter is not reset per file, so a two-file prescale keeps rows that a
    per-file counter would not."""
    out, n = _run_cat(tmp_path, [_A, _B], prescale=10)
    assert n == 3          # rows 10, 20, 30 of the concatenated 35
