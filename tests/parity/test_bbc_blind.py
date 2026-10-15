"""Parity vs SALT2mu.exe for blinding (blindflag / blindpar).

Blinding offsets the fixed reference cosmology by ``amp*cos(phase)`` on bit 2, and
adds ``cos(10*z_lo)`` to each M0DIF bin on bit 1. SNANA unblinds sims unless bit 64
is set, so the sim-input cases below use blindflag=64+n; the real-data case uses
CONTAM_DATA, which carries no SIM columns and so reads as real.

References are SALT2mu.exe on the 5D biasCor config; see the fixture README.
"""


import numpy as np
import pytest

from jax_salt2mu.config import (BBCConfig, BLINDMASK_MUz, resolve_blinding,
                                unblinded)
from jax_salt2mu.config_parser import parse_bbc_input
from jax_salt2mu.run import run_bbc_fit
from jax_snlc_core.io.fitres_reader import read_fitres


_DIR = "tests/fixtures/snana/bbc_des_beams"
# SNANA BBC reference files omit radiation.
_BASE = ["radiation=0",
         "nzbin=10", "zmin=0.1", "zmax=1.0", "u1=1", "u2=1", "p9=0.7", "p11=-1",
         f"simfile_biascor={_DIR}/BIASCOR.FITRES", "opt_biascor=112",
         "surveygroup_biascor=DES(zbin=0.15:x1bin=2.0:cbin=0.2)",
         "min_per_cell_biascor=3", "snrmin_sigint_biascor=60"]


@pytest.mark.parametrize("datafile,keys,ref", [
    # bit 2: Omega_L += 0.06*cos(23434), w0 += 0.20*cos(8430) (SNANA defaults)
    ("DATA.FITRES", ["blindflag=66"], "SALT2mu_blind_fixpar.M0DIF"),
    # bit 1: MUDIF += cos(10*z_lo), applied after MU is formed
    ("DATA.FITRES", ["blindflag=65"], "SALT2mu_blind_muz.M0DIF"),
    # user amplitude/phase for Omega_L; w0 keeps its default
    ("DATA.FITRES", ["blindflag=66", "blindpar9=0.1,4522"],
     "SALT2mu_blind_par9.M0DIF"),
    # both slots overridden. SALT2mu.exe needs these split across the input file and
    # the command line -- uniqueOverlap prefix-matches "blindpar" and permits one per
    # source -- whereas jax has no such limit and takes both together.
    ("DATA.FITRES", ["blindflag=66", "blindpar9=0.1,4522", "blindpar11=0.15,207"],
     "SALT2mu_blind_par_both.M0DIF"),
    # real data (no SIM columns) blinds without needing bit 64
    ("DATA_REAL.FITRES", ["blindflag=2"], "SALT2mu_blind_real.M0DIF"),
])
def test_blinding_matches_salt2mu(datafile, keys, ref):
    refpath = f"{_DIR}/{ref}"
    cfg = parse_bbc_input(None, overrides=[f"datafile={_DIR}/{datafile}"] + keys + _BASE)
    _, out, _, _, _ = run_bbc_fit(cfg)

    r = read_fitres(refpath)
    np.testing.assert_array_equal(out.nfit, np.asarray(r.get("NFIT")))
    assert np.max(np.abs(out.m0dif - np.asarray(r.get("MUDIF")))) < 2e-5
    assert np.max(np.abs(out.muref - np.asarray(r.get("MUREF")))) < 2e-5


def test_blinding_is_idempotent_and_leaves_the_input_config_alone():
    """resolve_blinding returns a new config and keeps the unblinded values
    (COSPAR_UNBLIND), so blinding twice does not compound the offset."""
    cfg = BBCConfig()
    once = resolve_blinding(cfg, isdata_real=True)
    assert cfg.omega_l == 0.7            # the parse result is untouched
    assert once.omega_l != 0.7
    assert resolve_blinding(once, isdata_real=True).omega_l == once.omega_l
    assert unblinded(once).omega_l == 0.7


def test_sim_is_unblinded_without_bit64():
    """A sim input disables blinding outright (SALT2mu.c:8844), which also clears
    the MUz bit, not just the fixed-parameter offset."""
    cfg = BBCConfig()
    cfg.blindflag = BLINDMASK_MUz | 2
    out = resolve_blinding(cfg, isdata_real=False)
    assert out.blindflag == 0
    assert out.omega_l == 0.7


def test_ccprior_reference_distance_is_unblinded():
    """The CC-prior sample's reference distance comes from COSPAR_UNBLIND while the
    data's comes from the blinded cosmology (malloc_PROB_CCprior, SALT2mu.c:17843).
    Both are the same key here, so blinding the CC map too shifts every CC event's
    DMU by a constant and with it DMUAVG and the BEAMS weights.
    """
    cfg = parse_bbc_input(None, overrides=[
        f"datafile={_DIR}/CONTAM_DATA.FITRES",
        f"simfile_ccprior={_DIR}/CCPRIOR.FITRES", "opt_ccprior=5",
        "varname_pIa=PROBIA", "p13=1", "typeIa_ccprior=999",
        "blindflag=2"] + _BASE)
    _, out, _, _, _ = run_bbc_fit(cfg)

    r = read_fitres(f"{_DIR}/SALT2mu_blind_ccprior.M0DIF")
    np.testing.assert_array_equal(out.nfit, np.asarray(r.get("NFIT")))
    assert np.max(np.abs(out.m0dif - np.asarray(r.get("MUDIF")))) < 2e-5


def test_muz_blinding_leaves_an_empty_bin_at_zero():
    """M0dif_calc adds the blind offset and only then overwrites an unfloated bin
    with 0 (SALT2mu.c:25438-25443), so an empty bin reads MUDIF=0 beside its
    MUDIFERR=999 rather than cos(10 z_lo)."""
    cfg = parse_bbc_input(None, overrides=[
        f"datafile={_DIR}/DATA.FITRES", "nzbin=14", "zmin=0.1", "zmax=1.4",
        "u1=1", "u2=1", "p9=0.7", "p11=-1", "blindflag=65", "radiation=0"])
    _, out, _, _, _ = run_bbc_fit(cfg)

    r = read_fitres(f"{_DIR}/SALT2mu_blind_muz_emptybin.M0DIF")
    empty = np.asarray(r.get("NFIT")) == 0
    assert empty.any()
    np.testing.assert_array_equal(out.m0dif[empty], np.zeros(int(empty.sum())))
    assert np.max(np.abs(out.m0dif - np.asarray(r.get("MUDIF")))) < 2e-5
