"""IDSAMPLE partitioning (SNANA prepare_IDSAMPLE_biasCor) — no SNANA executables.

Reads $SNDATA_ROOT/SURVEY.DEF (bundled fixture on CI) for survey name->id.
"""

import numpy as np
import pytest

from jax_salt2mu.config import MASK_BIASCOR_SAMPLE, BBCConfig
from jax_salt2mu.samples import (FieldGroup, assign_to_samples, field_groups,
                                 match_field_group, partition_idsamples, sample_binsize)
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.models import SALTModel


# SURVEY.DEF ids
CFA1, CFA2, CSP, DES, PS1MD, FOUND, SDSS = 61, 62, 5, 10, 15, 150, 1


def _config(opt_biascor=64, surveygroup="", fieldgroup="", fieldlist="", nzbin=20,
            zmin=0.01, zmax=1.1):
    return BBCConfig(
        zmin=zmin, zmax=zmax, nzbin=nzbin,
        opt_biascor=opt_biascor, nbin_logmass=1,
        surveygroup_biascor=surveygroup, fieldgroup_biascor=fieldgroup,
        fieldlist_biascor=fieldlist)


def _groups(*names):
    return [FieldGroup(n.replace("|", ""), "", "|" in n) for n in names]


# ---------------------------------------------------------------------------
# 1. single-survey / bit-64 off -> one ALL sample

def test_single_sample_bit_off():
    idsurvey = np.array([DES, DES, CFA1])
    zhd = np.array([0.1, 0.5, 0.02])
    isphot = np.zeros(3, dtype=bool)
    cfg = _config(opt_biascor=0)  # bit 64 off
    ids, table = partition_idsamples(idsurvey, None, zhd, isphot, cfg)

    assert np.all(ids == 0)
    assert len(table) == 1
    s = table[0]
    assert (s.name, s.surveygroup, s.fieldgroup) == ("ALL", "ALL", "NONE")
    assert s.zrange == (cfg.zmin, cfg.zmax)


# ---------------------------------------------------------------------------
# 2. DES5YR config -> 7-sample Example-B ordering

def test_des5yr_example_b_ordering():
    surveygroup = "CFA1+CFA2+CSP(zbin=0.02),FOUNDATION(zbin=0.02),DES(zbin=0.075)"
    fieldgroup = "X3,C3,X1+X2,S1+S2,E1+E2+C1+C2"
    cfg = _config(surveygroup=surveygroup, fieldgroup=fieldgroup)

    # (idsurvey, field, z)
    events = [
        (CFA1, "NONE", 0.02), (CFA2, "NONE", 0.03), (CSP, "NONE", 0.015),
        (FOUND, "NONE", 0.05), (FOUND, "NONE", 0.06),
        (DES, "X3", 0.3), (DES, "C3", 0.4),
        (DES, "X1", 0.5), (DES, "X2", 0.55),
        (DES, "S1", 0.6), (DES, "S2", 0.62),
        (DES, "E1", 0.7), (DES, "C1", 0.72), (DES, "C2", 0.8),
    ]
    idsurvey = np.array([e[0] for e in events])
    field = [e[1] for e in events]
    zhd = np.array([e[2] for e in events])
    isphot = np.zeros(len(events), dtype=bool)

    ids, table = partition_idsamples(idsurvey, field, zhd, isphot, cfg)

    names = [s.name for s in table]
    assert names == [
        "CFA1+CFA2+CSP", "FOUNDATION",
        "DES(X3)", "DES(C3)", "DES(X1+X2)", "DES(S1+S2)", "DES(E1+E2+C1+C2)",
    ]
    # per-sample zbin from STRINGOPT (DES field-splits inherit survey-group opt)
    zbins = {s.name: sample_binsize(s, cfg, SALTModel).zbin for s in table}
    assert zbins["CFA1+CFA2+CSP"] == 0.02
    assert zbins["FOUNDATION"] == 0.02
    for nm in ("DES(X3)", "DES(C3)", "DES(X1+X2)", "DES(S1+S2)", "DES(E1+E2+C1+C2)"):
        assert zbins[nm] == 0.075

    # every event mapped; CFA1/CFA2/CSP all land in sample 0
    assert set(ids[:3]) == {0}
    assert ids[3] == ids[4] == 1                      # FOUNDATION
    assert ids[7] == ids[8]                           # X1,X2 -> DES(X1+X2)
    assert table[ids[7]].name == "DES(X1+X2)"
    assert table[ids[11]].name == "DES(E1+E2+C1+C2)"  # E1

    # assign_to_samples reproduces the same mapping for "biasCor" events
    bc = assign_to_samples(idsurvey, field, isphot, table, cfg)
    assert np.array_equal(bc, ids)
    # an unmatched survey -> -9 (dropped)
    assert assign_to_samples(np.array([SDSS]), ["NONE"],
                             np.array([False]), table, cfg)[0] == -9


# ---------------------------------------------------------------------------
# 3. bidirectional field matching edge cases

def test_field_match_bidirectional():
    groups = _groups("X3", "C3", "X1+X2", "MEDIUM")

    # field subset of group name
    assert match_field_group("X1", groups)[0] == "X1+X2"
    # group name subset of field
    assert match_field_group("MEDIUM-01", groups)[0] == "MEDIUM"
    # exact
    assert match_field_group("C3", groups)[0] == "C3"
    # no match
    assert match_field_group("ZZ", groups)[0] == "NONE"


def test_field_overlap_first_match_wins():
    groups = _groups("X3", "C3")
    # overlap event: first group in user order that matches wins (no abort, NFIELD>1)
    assert match_field_group("C3+X3", groups)[0] == "X3"
    assert match_field_group("X3+C3", groups)[0] == "X3"


def test_field_ambiguous_single_field_aborts():
    groups = _groups("DEEP", "EP")  # single field "DEEP" matches both
    with pytest.raises(SNANAError):
        match_field_group("DEEP", groups)


def test_exact_field_matching():
    """fieldlist_biascor compares every FIELD whole, an overlap C3+X3 being its
    own field; a bar-wrapped group in fieldgroup_biascor is exact for the
    FIELD equal to it and skipped for every other (match_fieldGroup,
    SALT2mu.c:10981); the two keys are exclusive."""
    groups = field_groups(_config(fieldlist="C3,X3,C3+X3"))
    assert [g.name for g in groups] == ["C3", "X3", "C3+X3"]
    assert all(g.exact for g in groups)
    assert match_field_group("C3+X3", groups)[0] == "C3+X3"
    assert match_field_group("X3+C3", groups)[0] == "NONE"
    assert match_field_group("C3-01", groups)[0] == "NONE"

    groups = field_groups(_config(fieldgroup="|C3|,X3(zbin=0.1)"))
    assert groups == [FieldGroup("C3", "", True), FieldGroup("X3", "zbin=0.1", False)]
    assert match_field_group("C3", groups)[0] == "C3"        # exact, by the bars
    assert match_field_group("C3+X3", groups)[0] == "X3"     # |C3| skipped, X3 by substring
    assert match_field_group("C3-01", groups)[0] == "NONE"

    with pytest.raises(SNANAError, match="cannot specify both"):
        field_groups(_config(fieldgroup="C3", fieldlist="X3"))


def test_an_unmatched_field_joins_the_survey_s_first_field_sample():
    """An event whose FIELD matches no group carries FIELDGROUP NONE and
    get_IDSAMPLE matches it on SURVEYGROUP, which a field sample stores as
    its survey's name (SALT2mu.c:10036-10044), so it lands in the first
    sample of that survey rather than opening a survey-only one
    (SALT2mu.exe: fieldlist_biascor=C3,X3 on C3, X3, C3+X3, E2 gives two
    samples, DES(C3) holding C3, C3+X3 and E2)."""
    cfg = _config(fieldlist="C3,X3")
    idsurvey = np.full(4, DES)
    field = ["C3", "X3", "C3+X3", "E2"]
    ids, table = partition_idsamples(idsurvey, field, np.full(4, 0.3),
                                     np.zeros(4, dtype=bool), cfg)
    assert [s.name for s in table] == ["DES(C3)", "DES(X3)"]
    assert ids.tolist() == [0, 1, 0, 0]
    assert assign_to_samples(idsurvey, field, np.zeros(4, dtype=bool),
                             table, cfg).tolist() == [0, 1, 0, 0]


# ---------------------------------------------------------------------------
# 4. auto singleton groups for unlisted surveys (SURVEY.DEF order)

def test_auto_singleton_groups():
    surveygroup = "CFA1+CFA2+CSP(zbin=0.02)"  # DES and PS1MD unlisted
    cfg = _config(surveygroup=surveygroup)
    idsurvey = np.array([CFA1, PS1MD, DES, CSP])   # data order: PS1MD before DES
    zhd = np.array([0.02, 0.4, 0.3, 0.02])
    isphot = np.zeros(4, dtype=bool)

    ids, table = partition_idsamples(idsurvey, None, zhd, isphot, cfg)

    # user group first, then auto groups in ascending SURVEY.DEF id (DES=10, PS1MD=15)
    assert [s.name for s in table] == ["CFA1+CFA2+CSP", "DES", "PS1MD"]
    assert table[ids[0]].name == "CFA1+CFA2+CSP"
    assert table[ids[1]].name == "PS1MD"
    assert table[ids[2]].name == "DES"


# ---------------------------------------------------------------------------
# photo-z split: zSPEC block before zPHOT block

def test_photoz_split_ordering():
    cfg = _config(surveygroup="DES")
    idsurvey = np.array([DES, DES, DES])
    zhd = np.array([0.3, 0.4, 0.5])
    isphot = np.array([True, False, True])
    ids, table = partition_idsamples(idsurvey, None, zhd, isphot, cfg)

    assert [s.name for s in table] == ["DES-zSPEC", "DES-zPHOT"]
    assert table[0].is_photoz is False and table[1].is_photoz is True
    assert table[ids[1]].name == "DES-zSPEC"   # the spec-z event
    assert table[ids[0]].name == "DES-zPHOT"


def test_auto_partition_orders_by_idsurvey_not_surveydef():
    """With no surveygroup_biascor, IDSAMPLE follows ascending IDSURVEY. Verified
    against SALT2mu.exe with CNIa(20)/YSE_ZTF(19), a pair SURVEY.DEF lists in the
    opposite order, so the two candidate rules disagree: SNANA puts YSE_ZTF first."""
    cfg = BBCConfig()
    cfg.opt_biascor = MASK_BIASCOR_SAMPLE     # else all surveys collapse to one
    ids = np.array([20, 19] * 10)
    z = np.full(ids.size, 0.3)
    idsample, samples = partition_idsamples(
        ids, None, z, np.zeros(ids.size, bool), cfg)
    by_index = {s.idsample: s.surveygroup for s in samples}
    assert by_index[0] == "YSE_ZTF" and by_index[1] == "CNIa"
    assert list(idsample[:2]) == [1, 0]     # the 20 event -> IDSAMPLE 1


def test_survey_selection_can_reject_every_event():
    """An empty partition retains rejected IDs instead of indexing with floats."""
    config = _config()
    config.idsurvey_select = (SDSS,)
    ids, samples = partition_idsamples(
        np.array([DES]), None, np.array([0.3]), np.array([False]), config)
    np.testing.assert_array_equal(ids, [-9])
    assert samples == []
