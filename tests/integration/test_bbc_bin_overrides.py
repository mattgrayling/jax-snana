"""CID-file bin assignments survive preparation, fitting and output."""

import numpy as np
import pytest

from jax_salt2mu.config import parse_bbc_input
from jax_salt2mu.run import run_bbc_fit
from _bbc_helpers import prepare_cell
from jax_salt2mu.output import write_outputs
from jax_salt2mu.tablevar import CUTBIT_MINBIN, CUTBIT_z
from jax_snlc_core.snana_files import read_fitres


@pytest.mark.parametrize("invalid_bin", [None, 3, 99], ids=["assigned", "at-limit", "past-limit"])
def test_bin_overrides(tmp_path, invalid_bin):
    # These accepted DES events span several redshift bins; the file groups them
    # independently of redshift. Four events per bin meet min_per_zbin exactly.
    assignments = {"1": 0, "2": 0, "5": 0, "8": 0,
                   "12": 1, "14": 1, "15": 1, "16": 1,
                   "18": 2, "21": 2, "22": 2, "23": 2}
    if invalid_bin is not None:
        assignments["1"] = invalid_bin  # leaves bin 0 below min_per_zbin
    selection = tmp_path / "selection.list"
    selection.write_text("VARNAMES: CID IDSURVEY FIELD IZBIN\n" + "".join(
        f"SN: {cid} 10 VOID {ibin}\n" for cid, ibin in assignments.items()))
    root = "tests/fixtures/snana/bbc_des_beams"
    config = parse_bbc_input(None, overrides=[
        f"datafile={root}/DATA.FITRES", f"simfile_biascor={root}/BIASCOR.FITRES",
        "opt_biascor=112", "min_per_cell_biascor=3", "snrmin_sigint_biascor=60",
        "surveygroup_biascor=DES(zbin=0.15:x1bin=2.0:cbin=0.2)",
        f"cid_select_file={selection}", "izbin_from_cid_file=1",
        "nzbin=3", "zmin=0.1", "zmax=1.0", "min_per_zbin=4",
        "u1=0", "u2=0", "p1=0.14", "p2=3.1", "p9=0.7", "p11=-1",
        "radiation=0", "blindflag=0", f"prefix={tmp_path / 'bbc'}"])
    expected = dict(assignments)
    if invalid_bin is not None:
        for cid in ("1", "2", "5", "8"):
            del expected[cid]
    counts = [3, 4, 4] if invalid_bin is not None else [4, 4, 4]

    cell = prepare_cell(config)
    assert dict(zip(cell.tv.name[cell.accept], np.asarray(cell.inputs.izbin))) == expected
    np.testing.assert_array_equal(cell.nevt_zfit, counts)
    cutbits = dict(zip(cell.tv.name, cell.tv.cutbits))
    for cid in assignments:
        expected_cut = 0
        if invalid_bin is not None and cid == "1":
            expected_cut = 1 << CUTBIT_z
        elif invalid_bin is not None and cid in ("2", "5", "8"):
            expected_cut = 1 << CUTBIT_MINBIN
        assert cutbits[cid] == expected_cut

    result, output, zbins, table, accepted = run_bbc_fit(config)
    assert dict(zip(table.name[accepted], output.izbin)) == expected
    np.testing.assert_array_equal(output.nfit, counts)
    write_outputs(tmp_path / "bbc.FITRES", result, output, zbins, table, accepted)
    fitres = read_fitres(tmp_path / "bbc.FITRES")
    assert dict(zip(fitres.get("CID"), fitres.get("IZBIN"))) == expected
    m0dif = read_fitres(tmp_path / "bbc.M0DIF")
    np.testing.assert_array_equal(m0dif.get("NFIT"), counts)
