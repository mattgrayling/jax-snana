"""Unit tests for SIMLIB-reader pure helpers (NEA + IDEXPT parsing)."""

import pytest

from jax_snlc_core.logging import SNANAError
from jax_snlc_sim.simlib import _parse_idexpt_token


def test_parse_idexpt_basic():
    """``'1234*2'`` → (1234, 2); plain ``'1234'`` → (1234, 1)."""
    assert _parse_idexpt_token("1234*2") == (1234, 2)
    assert _parse_idexpt_token("1234") == (1234, 1)
    assert _parse_idexpt_token("999*16") == (999, 16)


def test_parse_idexpt_detnum_suffix():
    """The ``(DETNUM[,XPIX,YPIX])`` suffix is stripped and NEXPOSE kept, as
    parse_SIMLIB_IDEXPT."""
    assert _parse_idexpt_token("243289(12)") == (243289, 1)
    assert _parse_idexpt_token("243289*4(12)") == (243289, 4)
    assert _parse_idexpt_token("243289*4(12,234,1056)") == (243289, 4)


@pytest.mark.parametrize("token, expected", [
    ("abc42", (-9, 1)), ("", (-9, 1)), ("12*x", (12, 1)), ("1234.0", (1234, 1)),
    ("7*2.5", (7, 2)), ("12*", (12, 1))])
def test_parse_idexpt_reads_the_leading_integers(token, expected):
    """Each number is read as sscanf's %d reads it, its leading integer, with
    IDEXPT -9 and NEXPOSE 1 where none can be read (parse_SIMLIB_IDEXPT)."""
    assert _parse_idexpt_token(token) == expected


def test_parse_idexpt_rejects_two_detnum_args():
    """Two comma-separated arguments in parentheses abort, as SNANA's 1-or-3
    check does."""
    with pytest.raises(SNANAError, match="1 or 3"):
        _parse_idexpt_token("12*3(4,5)")
