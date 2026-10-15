"""Parity vs SALT2mu.exe for zspec_errmax_idsample and zphot_shift.

zspec_errmax_idsample splits each sample into -zSPEC/-zPHOT (IS_SPECZ_TABLEVAR,
SALT2mu.c:9384); zphot_shift adds a systematic redshift offset to the photo-z data
subset (:9316). The classification consults zPRIORERR when the key is set and both
zPRIOR columns are present (:8638, :8645), otherwise zHELERR -- whose absent-column
default of -9.0 (:8550) classifies a whole table as spec-z.

References and the doctoring recipe are in the fixture README.
"""


import numpy as np
import pytest

from jax_salt2mu.config import resolve
from jax_salt2mu.config_parser import parse_bbc_input
from jax_salt2mu.run import run_bbc_fit
from jax_salt2mu.samples import partition_idsamples
from jax_salt2mu.tablevar import EVENT_TYPE_BIASCOR, EVENT_TYPE_DATA, read_tablevar
from jax_snlc_core.models import SALTModel
from jax_snlc_core.io.fitres_reader import read_fitres


_DIR = "tests/fixtures/snana/bbc_des_beams"
_DATA = f"{_DIR}/DATA_PHOTOZ.FITRES"
# SNANA BBC reference files omit radiation.
_BASE = ["radiation=0",
         "nzbin=10", "zmin=0.1", "zmax=1.0", "u1=1", "u2=1", "p9=0.7", "p11=-1",
         "blindflag=0", f"datafile={_DATA}",
         "opt_biascor=112",
         "surveygroup_biascor=DES(zbin=0.15:x1bin=2.0:cbin=0.2)",
         "min_per_cell_biascor=3", "snrmin_sigint_biascor=60"]



def _config(bbc_biascor_variants, overrides):
    return parse_bbc_input(None, overrides=_BASE + [
        f"simfile_biascor={bbc_biascor_variants}/BIASCOR_PHOTOZ.FITRES"] + overrides)


def _tablevar(bbc_biascor_variants, path, event_type, overrides):
    cfg = resolve(_config(bbc_biascor_variants, overrides), SALTModel)
    directory = bbc_biascor_variants if event_type == EVENT_TYPE_BIASCOR else _DIR
    return read_tablevar([f"{directory}/{path}"], event_type, cfg, SALTModel)


def _assert_m0dif(out, reference, tol=1e-3):
    r = read_fitres(f"{_DIR}/{reference}")
    assert np.max(np.abs(out.m0dif - np.asarray(r.get("MUDIF")))) < tol
    assert np.max(np.abs(out.muref - np.asarray(r.get("MUREF")))) < tol


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------

@pytest.mark.parametrize("path, event_type, overrides, expected", [
    # SNANA's SAMPLE_INFO dump: DES-zSPEC 1222, DES-zPHOT 401 after cuts.
    ("DATA_PHOTOZ.FITRES", EVENT_TYPE_DATA, ["zspec_errmax_idsample=0.05"], (1332, 443)),
    # Key absent: zHELERR is 0.001, so "zerr < 0" is false and OPT_PHOTOZ alone splits.
    ("DATA_PHOTOZ.FITRES", EVENT_TYPE_DATA, [], (888, 887)),
    # BIASCOR has no zHELERR, so the -9.0 default makes "zerr < 0" true for every row.
    ("BIASCOR_PHOTOZ.FITRES", EVENT_TYPE_BIASCOR, [], (18041, 0)),
    # With the key set, zPRIORERR (0.02 / 0.10) is consulted instead.
    ("BIASCOR_PHOTOZ.FITRES", EVENT_TYPE_BIASCOR, ["zspec_errmax_idsample=0.05"], (13531, 4510)),
])
def test_is_photoz_classification(bbc_biascor_variants, path, event_type, overrides, expected):
    tv = _tablevar(bbc_biascor_variants, path, event_type, overrides)
    assert (int((~tv.is_photoz).sum()), int(tv.is_photoz.sum())) == expected


# --------------------------------------------------------------------------
# fit parity
# --------------------------------------------------------------------------

def test_zspec_errmax_idsample_matches_salt2mu(bbc_biascor_variants):
    """SNANA alpha0=0.18106, beta0=3.80801, NSNFIT=1472 on the split sample."""
    _, out, _, _, _ = run_bbc_fit(_config(bbc_biascor_variants, ["zspec_errmax_idsample=0.05"]))
    _assert_m0dif(out, "SALT2mu_photoz.M0DIF")


def test_zphot_shift_matches_salt2mu(bbc_biascor_variants):
    """zphot_shift=0.01 on the photo-z subset: SNANA alpha0=0.18136."""
    _, out, _, _, _ = run_bbc_fit(
        _config(bbc_biascor_variants, ["zspec_errmax_idsample=0.05", "zphot_shift=0.01"]))
    _assert_m0dif(out, "SALT2mu_zphot_shift.M0DIF")


def test_zphot_shift_is_not_a_noop():
    """The shifted and unshifted references differ, so the parity test above would
    fail if zphot_shift were dropped entirely."""
    a = read_fitres(f"{_DIR}/SALT2mu.M0DIF")
    b = read_fitres(f"{_DIR}/SALT2mu_zphot_shift.M0DIF")
    moved = np.max(np.abs(np.asarray(a.get("MUREF")) - np.asarray(b.get("MUREF"))))
    assert moved > 1e-2


# --------------------------------------------------------------------------
# ordering
# --------------------------------------------------------------------------

def test_zphot_shift_moves_zhd_but_not_mumodel(bbc_biascor_variants):
    """SNANA stores mumodel at :8331 and shifts zhd at :8392, so a fixed-cosmology
    mumodel never sees the shift while z-binning and the biasCor lookup do. Applying
    the shift before mumodel would still fit, just wrongly."""
    base = _tablevar(bbc_biascor_variants, "DATA_PHOTOZ.FITRES", EVENT_TYPE_DATA,
                     ["zspec_errmax_idsample=0.05"])
    shifted = _tablevar(bbc_biascor_variants, "DATA_PHOTOZ.FITRES", EVENT_TYPE_DATA,
                        ["zspec_errmax_idsample=0.05", "zphot_shift=0.01"])
    photoz = base.is_photoz
    np.testing.assert_allclose(shifted.zhd[photoz] - base.zhd[photoz], 0.01, atol=1e-6)
    np.testing.assert_array_equal(shifted.zhd[~photoz], base.zhd[~photoz])
    np.testing.assert_array_equal(shifted.mumodel, base.mumodel)


def test_zphot_shift_skips_biascor_events(bbc_biascor_variants):
    """The shift is data-only (IS_DATA at :8392)."""
    base = _tablevar(bbc_biascor_variants, "BIASCOR_PHOTOZ.FITRES", EVENT_TYPE_BIASCOR,
                     ["zspec_errmax_idsample=0.05"])
    shifted = _tablevar(bbc_biascor_variants, "BIASCOR_PHOTOZ.FITRES", EVENT_TYPE_BIASCOR,
                        ["zspec_errmax_idsample=0.05", "zphot_shift=0.01"])
    np.testing.assert_array_equal(shifted.zhd, base.zhd)


# ---------------------------------------------------------------------------
# the 1D biasCor path, which partitions the same way but corrects differently
# ---------------------------------------------------------------------------

_1D = ["radiation=0",
       "nzbin=10", "zmin=0.1", "zmax=1.0", "u1=1", "u2=1", "p9=0.7", "p11=-1",
       "blindflag=0", f"datafile={_DATA}",
       "opt_biascor=65", "zspec_errmax_idsample=0.05"]


def test_1d_biascor_with_split_sample_matches_salt2mu(bbc_biascor_variants):
    """opt_biascor=65 is 1DZAVG|SAMPLE, so the 1D bias is built per IDSAMPLE while
    the high-SNR reference subtracted from every bin is accumulated over the whole
    biasCor sample (SALT2mu.c:12160, :12221). Taking that reference per sample
    instead agrees only while there is one sample, so this needs two.
    """
    _, out, _, _, _ = run_bbc_fit(parse_bbc_input(None, overrides=_1D + [
        f"simfile_biascor={bbc_biascor_variants}/BIASCOR_PHOTOZ.FITRES"]))
    r = read_fitres(f"{_DIR}/SALT2mu_1d_photoz.M0DIF")
    np.testing.assert_allclose(out.m0dif, np.asarray(r.get("MUDIF")), atol=1e-4)
    np.testing.assert_allclose(out.muref, np.asarray(r.get("MUREF")), atol=1e-4)


def test_1d_biascor_splits_into_two_samples(bbc_biascor_variants):
    """Guards the test above: with one sample the per-sample and global high-SNR
    references coincide and it would pass either way."""
    config = resolve(parse_bbc_input(None, overrides=_1D + [
        f"simfile_biascor={bbc_biascor_variants}/BIASCOR_PHOTOZ.FITRES"]), SALTModel)
    tv = read_tablevar([_DATA], EVENT_TYPE_DATA, config, SALTModel)
    _, samples = partition_idsamples(tv.idsurvey, getattr(tv, "field", None),
                                     tv.zhd, tv.is_photoz, config)
    assert [s.name for s in samples] == ["DES-zSPEC", "DES-zPHOT"]
