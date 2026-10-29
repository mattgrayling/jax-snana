"""BBC on a table gives what BBC on the file gives.

A pipeline hands BBC the table the fitter returned rather than a path. That
is only safe if the two routes agree, so this runs the same fit both ways on
one fixture and compares the nuisance parameters and the binned distances.
"""


import numpy as np
import pytest

from jax_salt2mu.config import parse_bbc_input
from jax_salt2mu.run import run_bbc_fit
from jax_salt2mu.tablevar import EVENT_TYPE_BIASCOR, EVENT_TYPE_DATA
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import read_fitres_multi

_DIR = "tests/fixtures/snana/bbc_des_beams"
_DATA = f"{_DIR}/DATA.FITRES"
_BIASCOR = f"{_DIR}/BIASCOR.FITRES"
_GROUP = "DES(zbin=0.15:x1bin=2.0:cbin=0.2)"



_KEYS = ["nzbin=10", "zmin=0.1", "zmax=1.0", "u1=1", "u2=1"]
_BIASCOR_KEYS = ["min_per_cell_biascor=3", f"surveygroup_biascor={_GROUP}"]


def _config(*extra):
    """The fit with its data named as a file."""
    return parse_bbc_input(None, overrides=[f"datafile={_DATA}", *_KEYS, *extra])


def _table_config(*extra):
    """The same fit with its data to be handed in as a table."""
    return parse_bbc_input(None, overrides=[*_KEYS, *extra])


def test_table_route_matches_the_file_route():
    from_file = run_bbc_fit(_config())
    from_table = run_bbc_fit(
        _table_config(), tables={EVENT_TYPE_DATA: read_fitres_multi([_DATA])})

    res_file, out_file = from_file[0], from_file[1]
    res_table, out_table = from_table[0], from_table[1]

    assert set(res_table.nuisance) == set(res_file.nuisance)
    for name, value in res_file.nuisance.items():
        assert res_table.nuisance[name] == pytest.approx(value, rel=0, abs=0), name
    np.testing.assert_array_equal(out_table.nfit, out_file.nfit)
    for name in ("m0dif", "m0diferr", "muref", "mu", "muerr"):
        np.testing.assert_allclose(
            np.asarray(getattr(out_table, name)),
            np.asarray(getattr(out_file, name)), rtol=0, atol=0, err_msg=name)


def test_the_table_route_selects_the_same_events():
    """The cut mask is where a table that arrived by a different route would
    diverge first, since every cut reads the columns rather than the fit."""
    _, _, _, tv_file, accept_file = run_bbc_fit(_config())
    _, _, _, tv_table, accept_table = run_bbc_fit(
        _table_config(), tables={EVENT_TYPE_DATA: read_fitres_multi([_DATA])})

    np.testing.assert_array_equal(np.asarray(accept_table),
                                  np.asarray(accept_file))
    np.testing.assert_array_equal(np.asarray(tv_table.cutbits),
                                  np.asarray(tv_file.cutbits))


def test_a_biascor_table_corrects_as_a_biascor_file_does():
    """The biasCor is gated on its input being there, so a table handed in
    memory must count as that input; ignored, it would silently give the
    uncorrected answer instead."""
    from_file = run_bbc_fit(
        _config(*_BIASCOR_KEYS, f"simfile_biascor={_BIASCOR}"))
    from_table = run_bbc_fit(
        _config(*_BIASCOR_KEYS),
        tables={EVENT_TYPE_BIASCOR: read_fitres_multi([_BIASCOR])})
    uncorrected = run_bbc_fit(_config())

    res_file, out_file = from_file[0], from_file[1]
    res_table, out_table = from_table[0], from_table[1]

    for name, value in res_file.nuisance.items():
        assert res_table.nuisance[name] == pytest.approx(value, rel=0, abs=0), name
    np.testing.assert_array_equal(np.asarray(out_table.m0dif),
                                  np.asarray(out_file.m0dif))
    np.testing.assert_array_equal(np.asarray(out_table.nfit),
                                  np.asarray(out_file.nfit))
    assert not np.allclose(np.asarray(out_file.m0dif),
                           np.asarray(uncorrected[1].m0dif)), \
        "the biasCor changed nothing, so this proves nothing about the table"


def test_a_table_and_a_file_for_one_input_is_refused():
    """Neither may win silently: a MUOPT naming a simfile while the pipeline
    hands the same input in memory would otherwise be ignored."""
    with pytest.raises(SNANAError, match="simfile_biascor names a file"):
        run_bbc_fit(_config(*_BIASCOR_KEYS, f"simfile_biascor={_BIASCOR}"),
                    tables={EVENT_TYPE_BIASCOR: read_fitres_multi([_BIASCOR])})
    with pytest.raises(SNANAError, match="datafile names a file"):
        run_bbc_fit(_config(),
                    tables={EVENT_TYPE_DATA: read_fitres_multi([_DATA])})
