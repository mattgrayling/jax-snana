"""IDSAMPLE partitioning (SNANA prepare_IDSAMPLE_biasCor, SALT2mu.c:9839).

An IDSAMPLE is a ``surveygroup x fieldgroup x zSPEC/zPHOT`` partition of the data.
``partition_idsamples`` builds the sample table from data events exactly as
``prepare_IDSAMPLE_biasCor``; ``assign_to_samples`` maps biasCor/CCprior events
onto that existing table (``get_IDSAMPLE``, SALT2mu.c:11154).
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import BBCConfig, MASK_BIASCOR_SAMPLE
from .tablevar import CUTBIT_IDSAMPLE, CUTBIT_z

from jax_snlc_core.io.survey_def import load_survey_ids
from jax_snlc_core.utils.error_handling import SNANAError
from .biascor_grid import parse_binsize_stringopt


_IGNORE = {"null", "none", "blank", ""}  # IGNOREFILE (sntools.c:4832)


def _sample_codes(idsurvey, fields, is_photoz):
    """Integer keys for survey/field/redshift-type combinations, without strings."""
    field_codes, unique_fields = pd.factorize(fields, sort=False)
    survey_codes, _ = pd.factorize(idsurvey, sort=False)
    return (survey_codes * len(unique_fields) + field_codes) * 2 + is_photoz


def _is_none(s: Optional[str]) -> bool:
    return s is None or str(s).strip().lower() in _IGNORE


def _extract_stringopt(s: str) -> Tuple[str, str]:
    """extractStringOpt (sntools.c): 'base(opt)more' -> ('basemore', 'opt')."""
    m = re.search(r"\(([^)]*)\)", s)
    if not m:
        return s, ""
    return (s[:m.start()] + s[m.end():]), m.group(1)


@dataclass
class BiasCorSample:
    idsample: int
    name: str
    surveygroup: str
    fieldgroup: str
    is_photoz: bool
    stringopt: str
    #: the z range the sample bins on: under a zbin= STRINGOPT, the data's
    #: extent rounded out to that grid (set_BINSIZE_SAMPLE_biasCor, SALT2mu.c:9348),
    #: else the config's
    zrange: Tuple[float, float]


@dataclass(frozen=True)
class SampleBinsize:
    zbin: float
    lcpar_bins: Dict[str, float]    # STRINGOPT overrides, by biasCor axis
    zrange: Tuple[float, float]


@dataclass(frozen=True)
class FieldGroup:
    """One entry of fieldgroup_biascor or fieldlist_biascor: the field
    names joined by ``+``, the ``(zbin=...)`` options, and whether an event's
    FIELD must equal the name rather than contain or be contained by it
    (vertical bars, ``|DEEP+WIDE|``, or every entry of fieldlist_biascor)."""
    name: str
    opt: str
    exact: bool


def field_groups(config: BBCConfig) -> List[FieldGroup]:
    """The user's field groups (set_FIELDGROUP_biasCor, SALT2mu.c:10320):
    fieldlist_biascor lists exact field names, fieldgroup_biascor
    substrings, a bar-wrapped one exact; the two keys are exclusive."""
    group_key = config.fieldgroup_biascor.strip()
    list_key = config.fieldlist_biascor.strip()
    if group_key and list_key:
        raise SNANAError(
            "BBC input error: cannot specify both fieldgroup_biascor & "
            "fieldlist_biascor; remove one of these keys from BBC input file",
            function_name="set_FIELDGROUP_biasCor")
    groups = []
    for token in (group_key or list_key).split(","):
        if not token.strip():
            continue
        base, opt = _extract_stringopt(token.strip())
        exact = bool(list_key) or "|" in base
        groups.append(FieldGroup(base.replace("|", ""), opt, exact))
    return groups


def match_field_group(field: Optional[str], groups: List[FieldGroup],
                      snid: str = "") -> Tuple[str, str]:
    """match_fieldGroup (SALT2mu.c:10981): the group an event's FIELD falls in
    and its options. A FIELD equal to a group that demands an exact match is
    compared whole, overlaps included, against every group; every group
    demands one under fieldlist_biascor. Otherwise the FIELD's
    ``+``-separated overlaps are compared by substring in either direction,
    skipping the groups that demand an exact match. The first match wins; a
    single-field event matching two groups aborts."""
    if _is_none(field):
        return "NONE", ""
    exact = any(g.exact and g.name == field for g in groups)
    overlaps = [field] if exact or "+" not in field else field.split("+")
    nmatch = 0
    out_grp, out_opt = "NONE", ""
    for g in groups:
        if g.exact and not exact:
            continue
        for f2 in overlaps:
            hit = g.name == f2 if exact else (g.name in f2 or f2 in g.name)
            if hit:
                if nmatch == 0:
                    out_grp, out_opt = g.name, g.opt
                nmatch += 1
    if nmatch > 1 and len(overlaps) == 1:
        raise SNANAError(
            f"Invalid NMATCH_TOT={nmatch} for CID={snid}; FIELD='{field}' "
            f"matches multiple FIELDGROUPs {[g.name for g in groups]}",
            function_name="match_field_group")
    return out_grp, out_opt


def _sample_matches(fieldgroup, surveygroup, photoz, s_fieldgroup, s_surveygroup,
                    s_photoz) -> bool:
    """get_IDSAMPLE (SALT2mu.c:11154): an event with a field group joins the
    sample of that group, one without joins the first sample of its survey
    group, a field sample of that survey included; the zSPEC/zPHOT side
    must agree."""
    if not _is_none(fieldgroup):
        same = s_fieldgroup == fieldgroup
    else:
        same = s_surveygroup == surveygroup
    return same and s_photoz == photoz


class _SurveyGroups:
    """set_SURVEYGROUP_biasCor (SALT2mu.c:10416): user groups + auto singletons."""

    def __init__(self, surveygroup_str: str, present_ids, name_to_id, id_to_name):
        self.id_to_name = id_to_name
        self.user_names: List[str] = []              # user order
        self.user_opt: Dict[str, str] = {}
        self.ids_of: Dict[str, List[int]] = {}
        self.flag: Dict[int, str] = {}               # id -> 'USER'/'AUTO'

        if surveygroup_str.strip():
            for token in surveygroup_str.split(","):
                base, opt = _extract_stringopt(token.strip())
                ids = []
                for sname in base.split("+"):
                    if sname not in name_to_id:
                        raise SNANAError(
                            f"Undefined SURVEY = '{sname}' in surveygroup_biascor",
                            function_name="_SurveyGroups")
                    sid = name_to_id[sname]
                    ids.append(sid)
                    self.flag[sid] = "USER"
                self.user_names.append(base)
                self.user_opt[base] = opt
                self.ids_of[base] = ids

        # auto singleton groups: surveys in data not already grouped, id ascending
        self.auto_ids: List[int] = []
        for sid in sorted(present_ids):
            if self.flag.get(sid) in ("USER", "AUTO"):
                continue
            name = id_to_name[sid]
            self.flag[sid] = "AUTO"
            self.auto_ids.append(sid)
            self.ids_of[name] = [sid]

    def match(self, idsurvey: int, snid: str = "") -> Tuple[str, str]:
        """match_surveyGroup (SALT2mu.c:11099): unique group for idsurvey."""
        hits = []
        for name, ids in self.ids_of.items():
            if idsurvey in ids:
                opt = self.user_opt.get(name, "")
                hits.append((name, opt))
        if len(hits) != 1:
            raise SNANAError(
                f"Invalid NMATCH_TOT={len(hits)} for CID={snid}; "
                f"IDSURVEY={idsurvey}({self.id_to_name.get(idsurvey)})",
                function_name="_SurveyGroups.match")
        return hits[0]


def _one_sample(config: BBCConfig, n: int) -> Tuple[np.ndarray, List[BiasCorSample]]:
    """opt_biascor bit-64 off -> single 'ALL' sample (SALT2mu.c:9950)."""
    s = BiasCorSample(idsample=0, name="ALL", surveygroup="ALL", fieldgroup="NONE",
                      is_photoz=False, stringopt="", zrange=(config.zmin, config.zmax))
    return np.zeros(n, dtype=np.int64), [s]


def partition_idsamples(idsurvey, field, zhd, is_photoz, config: BBCConfig
                        ) -> Tuple[np.ndarray, List[BiasCorSample]]:
    """Reproduce prepare_IDSAMPLE_biasCor (SALT2mu.c:9839).

    Returns per-event idsample array and the sorted BiasCorSample table.
    """
    idsurvey = np.asarray(idsurvey, dtype=np.int64)
    zhd = np.asarray(zhd, dtype=float)
    is_photoz = np.asarray(is_photoz, dtype=bool)
    n = len(idsurvey)

    if (config.opt_biascor & MASK_BIASCOR_SAMPLE) == 0:
        return _one_sample(config, n)

    name_to_id, id_to_name = load_survey_ids()

    groups = field_groups(config)
    use_fieldgroup = bool(groups) and field is not None

    # an event the user excluded never becomes, or joins, an IDSAMPLE
    # (select_IDLIST/select_FIELD then continue, SALT2mu.c:10011-10014)
    selected = np.ones(n, dtype=bool)
    if config.idsurvey_select:
        selected &= np.isin(idsurvey, config.idsurvey_select)
    if use_fieldgroup and config.fieldlist:
        selected &= np.array([any(f in str(v) for f in config.fieldlist)
                              for v in field])

    present_ids = set(int(x) for x in idsurvey[selected])
    for sid in present_ids:
        if sid not in id_to_name:
            raise SNANAError(f"Invalid IDSURVEY={sid}; check SURVEY.DEF",
                             function_name="partition_idsamples")
    survey_groups = _SurveyGroups(config.surveygroup_biascor,
                                  present_ids, name_to_id, id_to_name)

    has_photoz = bool(is_photoz.any())

    # unsorted samples in creation order (get_IDSAMPLE)
    samples: List[dict] = []
    flds = (np.asarray(field, dtype=str) if use_fieldgroup
            else np.full(n, "NONE", dtype='U4'))
    combo = _sample_codes(idsurvey, flds, is_photoz)
    isn_selected = np.where(selected)[0]
    _, first, inverse = np.unique(combo[selected], return_index=True,
                                  return_inverse=True)
    first = isn_selected[first]
    combo_sample = np.empty(len(first), dtype=np.int64)

    for u in np.argsort(first):    # first event occurrence = creation order
        isn = int(first[u])
        idsrv = int(idsurvey[isn])
        fieldgroup, stringopt = match_field_group(flds[isn], groups)
        if stringopt == "":  # STRINGOPT-empty gate (SALT2mu.c:10037)
            surveygroup, stringopt = survey_groups.match(idsrv)
        else:
            surveygroup = id_to_name[idsrv]

        photoz = bool(is_photoz[isn])
        idx = next((i for i, s in enumerate(samples)
                    if _sample_matches(fieldgroup, surveygroup, photoz,
                                       s["fieldgroup"], s["surveygroup"],
                                       s["is_photoz"])), None)
        if idx is None:
            zgroup = ("-zPHOT" if photoz else "-zSPEC") if has_photoz else ""
            if _is_none(fieldgroup):
                name = f"{surveygroup}{zgroup}"
            else:
                name = f"{surveygroup}{zgroup}({fieldgroup})"
            idx = len(samples)
            samples.append(dict(fieldgroup=fieldgroup, surveygroup=surveygroup,
                                stringopt=stringopt, is_photoz=photoz, name=name))
        combo_sample[u] = idx

    event_idx = np.full(n, -9, dtype=np.int64)
    event_idx[selected] = combo_sample[inverse]
    zmins = np.full(len(samples), +999.0)
    zmaxs = np.full(len(samples), -999.0)
    np.minimum.at(zmins, event_idx[selected], zhd[selected])
    np.maximum.at(zmaxs, event_idx[selected], zhd[selected])
    for idx, s in enumerate(samples):
        s["zmin"], s["zmax"] = float(zmins[idx]), float(zmaxs[idx])

    _check_field_split_consistency(samples)

    # sort_IDSAMPLE_biasCor_driver (SALT2mu.c:10554)
    order = _sorted_order(samples, survey_groups, groups, has_photoz)
    old_to_new = np.empty(len(order), dtype=np.int64)
    old_to_new[np.asarray(order)] = np.arange(len(order))
    idsample_array = np.full(n, -9, dtype=np.int64)
    idsample_array[selected] = old_to_new[event_idx[selected]]

    table = []
    for new, old in enumerate(order):
        s = samples[old]
        table.append(BiasCorSample(
            idsample=new, name=s["name"], surveygroup=s["surveygroup"],
            fieldgroup=s["fieldgroup"], is_photoz=s["is_photoz"],
            stringopt=s["stringopt"],
            zrange=_sample_zrange(s["stringopt"], s["zmin"], s["zmax"], config)))

    if config.surveygroup_biascor_abortflag:
        # abort on a requested surveygroup with no data (SALT2mu.c:10479)
        present = {t.surveygroup for t in table}
        missing = [g for g in survey_groups.user_names if g not in present]
        if missing:
            raise SNANAError(
                f"no data for requested SURVEY(s) {', '.join(missing)}; "
                "check surveygroup_biascor",
                function_name="partition_idsamples")
    return idsample_array, table


def _check_field_split_consistency(samples: List[dict]) -> None:
    """Abort if a survey has both survey-only and field-split events (SALT2mu.c:10627)."""
    for s0 in samples:
        if not _is_none(s0["fieldgroup"]):
            continue
        for s2 in samples:
            if s2 is s0:
                continue
            if (not _is_none(s2["fieldgroup"])
                    and s2["surveygroup"] == s0["surveygroup"]):
                raise SNANAError(
                    f"Found {s0['surveygroup']} events NOT in FIELDGROUP; "
                    f"define all fields in fieldgroup_biascor",
                    function_name="_check_field_split_consistency")


def _sorted_order(samples, survey_groups: _SurveyGroups, groups: List[FieldGroup],
                  has_photoz
                  ) -> List[int]:
    """Order: user survey groups in the order given in surveygroup_biascor, user field
    groups, then auto groups by ascending IDSURVEY; zSPEC block before zPHOT
    (SALT2mu.c:10706). Auto order is numeric, not SURVEY.DEF order: verified against
    SALT2mu.exe with CNIa(20)/YSE_ZTF(19), which SURVEY.DEF lists in the other order."""
    order: List[int] = []
    z_blocks = [False, True] if has_photoz else [None]

    def block(photoz_block):
        for gname in survey_groups.user_names:
            for idx, s in enumerate(samples):
                if photoz_block is not None and s["is_photoz"] != photoz_block:
                    continue
                if s["surveygroup"] == gname and _is_none(s["fieldgroup"]):
                    if idx not in order:
                        order.append(idx)
        for g in groups:
            for idx, s in enumerate(samples):
                if photoz_block is not None and s["is_photoz"] != photoz_block:
                    continue
                if s["fieldgroup"] == g.name and idx not in order:
                    order.append(idx)
        for sid in survey_groups.auto_ids:
            gname = survey_groups.id_to_name[sid]
            for idx, s in enumerate(samples):
                if photoz_block is not None and s["is_photoz"] != photoz_block:
                    continue
                if not _is_none(s["fieldgroup"]):
                    continue
                if s["surveygroup"] == gname and idx not in order:
                    order.append(idx)

    for b in z_blocks:
        block(b)
    return order


def assign_to_samples(idsurvey, field, is_photoz, samples: List[BiasCorSample],
                      config: BBCConfig) -> np.ndarray:
    """get_IDSAMPLE (SALT2mu.c:11154) for biasCor/CCprior events on the data's
    sample table, with the user's field groups; -9 for an event matching
    no sample."""
    idsurvey = np.asarray(idsurvey, dtype=np.int64)
    is_photoz = np.asarray(is_photoz, dtype=bool)
    n = len(idsurvey)

    if len(samples) == 1 and samples[0].surveygroup == "ALL":
        return np.zeros(n, dtype=np.int64)   # bit-64 off: all lumped together

    name_to_id, _ = load_survey_ids()
    groups = field_groups(config)
    # survey group membership from the table: idsurvey -> surveygroup name
    id_to_sg: Dict[int, str] = {}
    for s in samples:
        for sname in s.surveygroup.split("+"):
            if sname in name_to_id:
                id_to_sg.setdefault(name_to_id[sname], s.surveygroup)

    # resolve each distinct (idsurvey, field, photoz) combination once
    flds = (np.asarray(field, dtype=str) if field is not None
            else np.full(n, "NONE", dtype='U4'))
    combo = _sample_codes(idsurvey, flds, is_photoz)
    _, first, inverse = np.unique(combo, return_index=True, return_inverse=True)
    combo_out = np.full(len(first), -9, dtype=np.int64)
    for u, isn in enumerate(first):
        fieldgroup, _ = match_field_group(flds[isn], groups)
        surveygroup = id_to_sg.get(int(idsurvey[isn]), "NONE")
        photoz = bool(is_photoz[isn])
        for s in samples:
            if _sample_matches(fieldgroup, surveygroup, photoz,
                               s.fieldgroup, s.surveygroup, s.is_photoz):
                combo_out[u] = s.idsample
                break
    return combo_out[inverse]


def _sample_zrange(stringopt: str, zmin_data: float, zmax_data: float,
                   config: BBCConfig) -> Tuple[float, float]:
    """A sample's z range from its data's extent (:9348): rounded down to 0.01
    and up to whole zbin steps under a zbin= STRINGOPT, else the config's."""
    ov = parse_binsize_stringopt(f"({stringopt})") if stringopt else {}
    if "zbin" not in ov:
        return (config.zmin, config.zmax)
    r0 = 0.01 * int(100.0 * zmin_data)
    nbz = int((zmax_data - r0) / ov["zbin"]) + 1
    return (r0, r0 + ov["zbin"] * nbz)


def sample_binsize(sample: BiasCorSample, config: BBCConfig, model) -> SampleBinsize:
    """Per-sample z and LC-parameter binsizes from STRINGOPT, the latter as
    ``<axis>bin=`` per biasCor axis (set_BINSIZE_SAMPLE_biasCor:9229)."""
    ov = parse_binsize_stringopt(f"({sample.stringopt})") if sample.stringopt else {}
    unknown = set(ov) - {"zbin", *(f"{name}bin" for name in model.biascor_binning)}
    if unknown:
        raise SNANAError(f"Unknown bias-correction bin override: {', '.join(sorted(unknown))}",
                         function_name="sample_binsize")
    zbin = ov["zbin"] if "zbin" in ov else (config.zmax - config.zmin) / config.nzbin
    lcpar_bins = {name: ov[f"{name}bin"] for name in model.biascor_binning
                  if f"{name}bin" in ov}
    return SampleBinsize(zbin=zbin, lcpar_bins=lcpar_bins, zrange=sample.zrange)


def assign_biascor(biascor_tv, samples, config, model):
    """The biasCor events' IDSAMPLE on the data's partition, cutting those that
    match no sample or fall outside its z range."""
    bc_idsample = assign_to_samples(
        biascor_tv.idsurvey, biascor_tv.field, biascor_tv.is_photoz, samples, config)
    # a biasCor event matching no data IDSAMPLE fails the IDSAMPLE cut
    # (SALT2mu.c:18957-18958), keeping it out of sigInt and zM0
    biascor_tv.cutbits[bc_idsample < 0] |= 1 << CUTBIT_IDSAMPLE
    apply_biascor_zcut(biascor_tv, bc_idsample, samples, config, model)
    return bc_idsample


def apply_biascor_zcut(biascor_tv, bc_idsample, samples, config, model) -> None:
    """The biasCor z cut, whose zmax each sample widens by its own z binsize
    (set_CUTMASK, SALT2mu.c:18866). Needs the IDSAMPLE partition, so it runs here
    rather than in ``compute_cutbits``."""
    # an event matching no sample keeps SNANA's pre-STRINGOPT default (:10178)
    zbin = np.array([sample_binsize(s, config, model).zbin for s in samples]
                    + [(config.zmax - config.zmin) / config.nzbin])
    zmax = config.zmax + zbin[np.where(bc_idsample >= 0, bc_idsample, -1)]
    z = biascor_tv.zhd
    biascor_tv.cutbits[(z < config.zmin) | (z > zmax)] |= 1 << CUTBIT_z
