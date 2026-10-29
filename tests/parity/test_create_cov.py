"""create_covariance parity against SNANA's create_covariance.py.

The fixture is a 3x3 FITOPT x MUOPT grid built by
``tests/fixtures/snana/create_cov/build.py``, which also captures SNANA's own
outputs for each case below. Both sides read the same grid, so the comparison is
exact rather than statistical.

The equality is at the precision SNANA's formats preserve, not to float64:
covariances are stored as float32 (~2.4e-8 relative here) and the Hubble diagram
is printed to %8.5f / %7.4f. Measured against those ceilings the agreement is
total -- our float64 rounds to exactly SNANA's stored bits -- but a tighter claim
than the formats allow cannot be made.

Two options have no case here because SNANA produces nothing to compare against:
``--muopt`` aborts and ``--label_cov_rows`` writes only a row count. They are covered in tests/unit instead.
"""

import gzip
import os

import numpy as np
import pytest
import yaml

from jax_create_cov import create_covariance
from jax_create_cov.config import build_config
from jax_create_cov.config import read_yaml
from jax_snlc_core.snana_files import read_covariance_npz

_FIXTURE = os.path.join("tests", "fixtures", "snana", "create_cov")
_N_COVOPT = 8  # the seven configured, plus the one create_covariance prepends

# INFO.YML names the code that wrote it, so this one cannot match by
# construction; its absence should still fail. Every other key, paths included,
# compares exactly -- both sides record them as the config named them.
_MACHINE_KEYS = {"SNANA_VERSION"}


class Case:
    def __init__(self, ref, config="input.yml", **overrides):
        self.ref = ref
        self.config = config
        self.overrides = overrides


_CASES = {
    "binned": Case("ref"),
    "unbinned": Case("ref_unbinned", mode="unbinned"),
    "rebin": Case("ref_rebin", mode="rebin", nbin_x1=2, nbin_c=2),
    "rebin_asymmetric": Case("ref_rebin31", mode="rebin", nbin_x1=3, nbin_c=1),
    # a hole in IZBIN, which drops every bin above it
    "rebin_izbin_gap": Case("ref_izgap", config="input_izgap.yml", mode="rebin",
                            nbin_x1=2, nbin_c=2),
    "covsys_text": Case("ref_covsys", write_mask_cov=1, write_format_cov="text"),
    "covtot": Case("ref_covtot", write_mask_cov=7),
    "extra_covs": Case("ref_extracov", config="input_extracov.yml", mode="unbinned"),
    "calibrators": Case("ref_calibrators", config="input_calibrators.yml",
                        mode="unbinned"),
    "subtract_vpec": Case("ref_vpec", mode="rebin", nbin_x1=2, nbin_c=2,
                          subtract_vpec=True),
    "nosys": Case("ref_nosys", nosys=True),
    "systematic_hd": Case("ref_syshd", systematic_hd=True),
}


def _read_hd(path):
    varnames, rows = None, []
    with open(path) as stream:
        for line in stream:
            if line.startswith("VARNAMES:"):
                varnames = line.split()[1:]
            elif line.startswith(("ROW:", "SN:")):
                rows.append(line.split()[1:])
    assert varnames, f"{path} has no VARNAMES line"
    return (varnames, [r[0] for r in rows],
            np.array([[float(v) for v in r[1:]] for r in rows]))


def _read_cov(path):
    if str(path).endswith(".npz"):
        return read_covariance_npz(path)
    body = gzip.decompress(open(path, "rb").read()).decode().split()
    size = int(body[0])
    return np.array([float(v) for v in body[1:]]).reshape(size, size)


@pytest.fixture(scope="module", params=sorted(_CASES))
def parity_case(request, tmp_path_factory):
    name = request.param
    case = _CASES[name]
    outdir = tmp_path_factory.mktemp(f"createcov_{name}")

    # SNANA create_covariance omits radiation in its reference distance grid.
    config = build_config(read_yaml(os.path.join(_FIXTURE, case.config)),
                          {**case.overrides, "radiation": False})
    config.outdir = outdir
    create_covariance(config)
    return name, case, outdir, os.path.join(_FIXTURE, case.ref)


def test_output_file_set_matches_snana(parity_case):
    name, _, outdir, ref_dir = parity_case
    assert (sorted(p.name for p in outdir.iterdir())
            == sorted(os.listdir(ref_dir))), f"{name}: different files written"


def test_hubble_diagrams_match_snana(parity_case):
    """Every HD in the directory -- one, or one per systematic under -H."""
    name, _, outdir, ref_dir = parity_case
    diagrams = sorted(p for p in os.listdir(ref_dir) if p.endswith(".txt"))
    assert diagrams

    for filename in diagrams:
        ref_names, ref_rows, ref = _read_hd(os.path.join(ref_dir, filename))
        got_names, got_rows, got = _read_hd(os.path.join(outdir, filename))
        assert got_names == ref_names, f"{name}/{filename}: VARNAMES differ"
        assert got_rows == ref_rows, f"{name}/{filename}: row identities differ"
        for i, column in enumerate(ref_names[1:]):
            np.testing.assert_array_equal(
                got[:, i], ref[:, i], err_msg=f"{name}/{filename}: {column}")


def test_covariances_match_snana(parity_case):
    name, _, outdir, ref_dir = parity_case
    matrices = sorted(p for p in os.listdir(ref_dir)
                      if p.startswith(("covsys", "covtot")))
    assert matrices

    nonzero = 0
    for filename in matrices:
        ref = _read_cov(os.path.join(ref_dir, filename))
        got = _read_cov(os.path.join(outdir, filename))
        assert got.shape == ref.shape, f"{name}/{filename}: shape differs"
        np.testing.assert_array_equal(got, ref, err_msg=f"{name}/{filename}")
        nonzero += np.abs(ref).max() > 0
    # a covopt selecting only the reference cell is legitimately zero, but they
    # cannot all be, or this test would pass on empty output
    assert nonzero, f"{name}: every covariance is zero"


def test_info_yml_matches_snana(parity_case):
    name, _, outdir, ref_dir = parity_case
    ref_text = open(os.path.join(ref_dir, "INFO.YML")).read()
    got_text = open(os.path.join(outdir, "INFO.YML")).read()

    assert (got_text.split("COVOPTS")[0] == ref_text.split("COVOPTS")[0]), \
        f"{name}: INFO.YML comment header differs"

    ref, got = yaml.safe_load(ref_text), yaml.safe_load(got_text)
    # SNANA fell back to this cosmology and records none; jax was given it
    assert ref.pop("COSPAR_BIASCOR") == {}
    assert got.pop("COSPAR_BIASCOR") == {"OMEGA_MATTER": 0.315, "OMEGA_LAMBDA": 0.685,
                                         "w0_LAMBDA": -1.0, "wa_LAMBDA": 0.0,
                                         "MUSHIFT": 0.0}
    assert set(got) == set(ref), f"{name}: INFO.YML keys differ"
    for key in sorted(set(ref) - _MACHINE_KEYS):
        assert got[key] == ref[key], f"{name}: INFO.YML {key} differs"
    for key in _MACHINE_KEYS:
        assert got.get(key) is not None or ref.get(key) is None, \
            f"{name}: INFO.YML {key} missing"


def test_extra_covs_and_calibrators_change_the_result():
    """Both would pass their parity check while doing nothing at all.

    EXTRA_COVS is only picked up by the unfiltered covopt, and the calibrator
    masking only fires for a systematic labelled VPEC or ZSHIFT -- the externally
    supplied one here is VPEC_EXTRA.
    """
    def cov(ref):
        return read_covariance_npz(os.path.join(_FIXTURE, ref,
                                                "covtot_inv_000.npz"))

    plain, extra, calib = cov("ref_unbinned"), cov("ref_extracov"), cov("ref_calibrators")
    assert np.abs(plain - extra).max() > 1.0, "EXTRA_COVS contributed nothing"
    assert np.abs(extra - calib).max() > 1.0, \
        "CALIBRATORS masked nothing -- are the CIDs quoted in the config?"

    for index in (1, 2, 3, 4):
        name = f"covtot_inv_{index:03d}.npz"
        np.testing.assert_array_equal(
            read_covariance_npz(os.path.join(_FIXTURE, "ref_unbinned", name)),
            read_covariance_npz(os.path.join(_FIXTURE, "ref_extracov", name)),
            err_msg=f"{name}: EXTRA_COVS leaked past its covopt filters")


def test_subtract_vpec_moves_the_bin_redshifts():
    """Guards the vpec case, which has been vacuous twice: once when MUERR_VPEC
    was all zeros, and again in unbinned mode, where MUERR is overwritten by
    MUERR_RENORM after the subtraction.

    In rebin the subtracted MUERR survives into wgt0, which averages MUREF, whose
    inversion gives the bin redshift. MU and MUERR come from wgt1 and are
    untouched -- watching those would make this test vacuous a third time. Not
    every bin moves visibly: zHD prints to five decimals and the shift is smaller
    than that in some bins.
    """
    def column(ref, name):
        names, rows, values = _read_hd(
            os.path.join(_FIXTURE, ref, "hubble_diagram.txt"))
        return dict(zip(rows, values[:, names.index(name) - 1]))

    for name in ("zHD", "MUSIM"):
        plain, subtracted = column("ref_rebin", name), column("ref_vpec", name)
        shared = sorted(set(plain) & set(subtracted))
        assert shared
        shifts = [abs(plain[c] - subtracted[c]) for c in shared]
        assert sum(s > 0 for s in shifts) > len(shared) // 2, \
            f"subtract_vpec barely moved {name}"
        assert max(shifts) > 1.0e-3, f"{name} shifted only at the print precision"

    plain, subtracted = column("ref_rebin", "MUERR"), column("ref_vpec", "MUERR")
    assert all(plain[c] == subtracted[c] for c in plain), \
        "MUERR comes from wgt1 and must not move"
