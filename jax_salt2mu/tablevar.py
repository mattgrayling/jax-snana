"""Read/derive/cut layer for the three BBC event types.

One :class:`TableVar` structure serves data, biasCor and CC-prior inputs
(SNANA's single ``TABLEVAR_DEF`` switched by ``EVENT_TYPE``). Reading resolves
the model's column aliases, derives the LC-parameter vector and the LC-fit
covariance through the model, computes the reference-cosmology distance
modulus, assigns IDSAMPLE, and applies the CUTWIN selection.
"""

import logging
from copy import deepcopy
from dataclasses import dataclass, field as _dc_field, replace
from typing import ClassVar, Dict, List, Optional

import jax
import numpy as np

from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import (SIM_COSMOLOGY_KEYS, SNANA_NULLFLOAT, open_text,
                                       read_fitres_multi)
from jax_snlc_core.utils import LIGHT_KM, fix_covariance

from .config import (MASK_BIASCOR_MUCOVSCALE, BBCConfig, lcpar_windows, map_bins)
from .blinding import unblinded
from .distances import mu_at


logger = logging.getLogger(__name__)

EVENT_TYPE_DATA = 1
EVENT_TYPE_BIASCOR = 2
EVENT_TYPE_CCPRIOR = 3

_ZHELERR_ABSENT = SNANA_NULLFLOAT  # TABLEVAR init for an absent column (:7626)


def _f32(a) -> np.ndarray:
    """Truncate to SNANA's TABLEVAR float32 storage, widened back for float64 math."""
    return np.asarray(a, dtype=np.float32).astype(np.float64)


# SIM truth columns BBC itself needs; the model's biascor_truth_aliases add
# the ones its biasCor needs.
_SIM_TRUTH = {
    "SIM_ZCMB": ("SIM_ZCMB",),
    "SIM_DLMAG": ("SIM_DLMAG",),
    "SIM_WGT_POP": ("SIM_WGT_POP",),
    "SIM_TEMPLATE_INDEX": ("SIM_NONIA_INDEX", "SIM_TEMPLATE_INDEX"),   # 0 for Ia (SALT2mu.c:8822)
    "SIM_AV": ("SIM_AV",),                           # a dust model's biasCor (:11930)
}


# SNANA CUTMASK bit numbering (SALT2mu.c:614-638).
CUTBIT_z = 0
CUTBIT_logmass = 3
CUTBIT_sntype = 4
CUTBIT_BADERR = 7       # an LC-fit error the model's thresholds reject
CUTBIT_BADCOV = 8       # an LC-fit covariance rejected even with CUTWIN NONE
CUTBIT_CUTWIN = 6
CUTBIT_MINBIN = 9       # z-bin with too few events to fit
CUTBIT_SIMPS = 11       # rejected by a sim prescale
CUTBIT_zBIASCOR = 13    # biasCor event outside its sample's z grid
CUTBIT_TRUESNIa = 16    # biasCor event that is not a true SNIa
CUTBIT_TRUESNCC = 17    # CC-prior event that is a true SNIa
CUTBIT_BIASCOR = 12     # failed bias correction or generic simulation grid cut
CUTBIT_DUPL = 19        # all but the first of a duplicate-CID set
CUTBIT_NMAXCUT = 18
CUTBIT_IDSAMPLE = 20
CUTBIT_IDSURVEY = 21
CUTBIT_CHI2 = 22        # chi2max outlier (data only)
CUTBIT_CID = 23         # cid_select_file / cid_reject_file
CUTBIT_FIELD = 24       # fieldlist=


@dataclass
class TableVar:
    PER_EVENT_ARRAYS: ClassVar[tuple[str, ...]] = (
        'name', 'idsurvey', 'idsample', 'zhd', 'zmuerr', 'logmass',
        'covmat_fit', 'mumodel', 'cutbits', 'snrmax', 'field', 'is_photoz',
        'vpec', 'zcmb', 'pIa', 'sntype', 'baderr', 'badcov',
    )

    event_type: int
    name: np.ndarray
    idsurvey: np.ndarray
    idsample: np.ndarray
    zhd: np.ndarray
    zmuerr: np.ndarray                # z-error feeding muerr_z (vpecerr/c, not zhderr)
    logmass: np.ndarray
    lcpar: Dict[str, np.ndarray]
    covmat_fit: np.ndarray            # [n, k, k]
    mumodel: np.ndarray               # reference-cosmology distance modulus
    cutbits: np.ndarray               # per-event failed-cut bitmask (SNANA CUTMASK)
    snrmax: Optional[np.ndarray] = None
    field: Optional[np.ndarray] = None   # FIELD strings, for IDSAMPLE field groups
    is_photoz: Optional[np.ndarray] = None   # zSPEC/zPHOT IDSAMPLE split
    vpec: Optional[np.ndarray] = None    # peculiar velocity, for check_vpec_sign
    zcmb: Optional[np.ndarray] = None    # CMB-frame z, for check_vpec_sign's sign test
    version_photometry: str = ""   # input header key, for output provenance
    # the simulation's generating cosmology, as the light-curve fit's header carries it
    sim_cosmology: Dict[str, str] = _dc_field(default_factory=dict)
    pIa: Optional[np.ndarray] = None
    sntype: Optional[np.ndarray] = None  # SNTYPE; -9 when absent (force_probcc0)
    baderr: Optional[np.ndarray] = None  # LC-fit error rejected (CUTBIT_BADERR)
    badcov: Optional[np.ndarray] = None  # LC-fit covariance rejected (CUTBIT_BADCOV)
    sim_truth: Dict[str, np.ndarray] = _dc_field(default_factory=dict)
    # Input columns carried into the output FITRES, in SNANA's order (data only)
    passthrough: Dict[str, np.ndarray] = _dc_field(default_factory=dict)

    def take(self, rows) -> "TableVar":
        """Select event rows, preserving table metadata."""
        def take(value):
            return None if value is None else value[rows]

        return replace(self,
            **{name: take(getattr(self, name)) for name in self.PER_EVENT_ARRAYS},
            lcpar={key: value[rows] for key, value in self.lcpar.items()},
            sim_truth={key: value[rows] for key, value in self.sim_truth.items()},
            passthrough={key: value[rows] for key, value in self.passthrough.items()})

    @property
    def size(self) -> int:
        return len(self.name)

    @property
    def cutmask(self) -> np.ndarray:
        return self.cutbits == 0

    @property
    def n_pass(self) -> int:
        return int(self.cutmask.sum())

    def n_fail(self, bit: int) -> int:
        """Events failing a given CUTBIT (SNANA NSTORE_CUTBIT)."""
        return int((self.cutbits & (1 << bit) != 0).sum())


def _resolve(table, aliases) -> Dict[str, np.ndarray]:
    """First-present-alias resolution for each raw key; abort if none present."""
    out = {}
    for raw, candidates in aliases.items():
        col = next((c for c in candidates if table.has(c)), None)
        if col is None:
            raise SNANAError(
                f"required column for '{raw}' not found (tried {candidates})",
                function_name="convert_table",
            )
        out[raw] = _f32(table.get(col))
    return out


def isdata_real(table, config: BBCConfig) -> bool:
    """Whether the data table is real data: no template-index column
    (SALT2mu.c:8822-8832), or force_realdata (:8825). What blinding is resolved
    against."""
    if config.force_realdata:
        return True
    return not any(table.has(c) for c in _SIM_TRUTH["SIM_TEMPLATE_INDEX"])


def read_data_table(paths):
    """The data FITRES at ``paths``, parsed and again as text.

    Both readings are needed because the writer re-emits the input's own tokens
    (:func:`passthrough_column`). Split out because blinding is resolved between
    the read and the derive during cell preparation.
    """
    paths = list(paths)
    return read_fitres_multi(paths), read_fitres_multi(paths, as_text=True)


def select_tablevar(tv, table, config, model):
    """Apply a cell's cuts to shared, unmerged numerical variables."""
    if config.varname_pia and tv.pIa is None and tv.event_type == EVENT_TYPE_DATA:
        _require_pia_column(tv, config)
    return replace(tv, cutbits=compute_cutbits(tv, config, model, table),
                   idsample=np.zeros(tv.size, dtype=np.int64),
                   lcpar=dict(tv.lcpar), sim_truth=dict(tv.sim_truth),
                   sim_cosmology=dict(tv.sim_cosmology),
                   passthrough={name: tv.passthrough[name] for name in
                                output_varnames(table.varnames_by_file, config)}
                   if tv.event_type == EVENT_TYPE_DATA and tv.passthrough else {})


def sample_coordinates(table, config):
    """Survey, field, redshift and redshift type, before any event selection."""
    zhd = _read_redshift(table, config)
    if (config.uzsim and not isdata_real(table, config)
            and table.has("SIM_ZCMB")):
        zhd = _f32(table.get("SIM_ZCMB"))
    is_photoz = _is_photoz(table, config)
    if config.zphot_shift != 0.0:
        zhd = np.where(is_photoz, zhd + config.zphot_shift, zhd)
    idsurvey = np.asarray(table.get("IDSURVEY", np.full(table.size, -9)), dtype=np.int64)
    field = np.asarray(table.get("FIELD")).astype(str) if table.has("FIELD") else None
    return idsurvey, field, zhd, is_photoz


def _read_redshift(table, config):
    if not table.has(config.varname_z):
        raise SNANAError(f"varname_z column '{config.varname_z}' is not in the input table",
                         function_name="convert_table")
    return _f32(table.get(config.varname_z))


def _required_lcpar(model, config, *, biascor=None, ccprior=None):
    required = set(lcpar_windows(model, config))
    if config.simfile_biascor or biascor is not None:
        required.update(model.biascor_binning)
        if config.opt_biascor & MASK_BIASCOR_MUCOVSCALE:
            required.update(map_bins(model.mucovscale_bins, config.mucovscale_bins))
    if config.simfile_ccprior or ccprior is not None:
        required.update(map_bins(model.ccprior_bins, config.ccprior_bins))
    return required


def conversion_key(event_type, config, model, *, biascor=None, ccprior=None):
    """Only inputs to numerical conversion; cuts and map grids are applied later.

    Keep this dependency list beside convert_table. Cosmology uses the same
    unblinding rule as _mumodel, including for CC-prior inputs.
    """
    cosmology = unblinded(config) if event_type == EVENT_TYPE_CCPRIOR else config
    return deepcopy((
        jax.config.x64_enabled, config._model_inputs,
        tuple(sorted(_required_lcpar(model, config, biascor=biascor, ccprior=ccprior))),
        config.varname_z, config.varname_gamma, config.varname_pia,
        config.zpecerr, config.uzsim, config.zspec_errmax_idsample,
        config.force_realdata if event_type == EVENT_TYPE_DATA else False,
        config.zphot_shift if event_type == EVENT_TYPE_DATA else 0.,
        (cosmology.omega_l, cosmology.omega_k, cosmology.h0, cosmology.w0,
         cosmology.wa, cosmology.radiation, cosmology.muref_zpoly),
    ))


def readonly_variables(tv):
    """Protect shared arrays through views without changing their source table."""
    def readonly(value):
        if value is None:
            return None
        value = value.view()
        value.setflags(write=False)
        return value

    return replace(tv,
        **{name: readonly(getattr(tv, name)) for name in tv.PER_EVENT_ARRAYS},
        lcpar={key: readonly(value) for key, value in tv.lcpar.items()},
        sim_truth={key: readonly(value) for key, value in tv.sim_truth.items()},
        passthrough={key: readonly(value) for key, value in tv.passthrough.items()})


def convert_table(table, event_type, config, model, *, biascor=None, ccprior=None,
                  cosmology_width=None):
    """Derive numerical columns before output formatting, cuts or duplicates."""
    is_biascor = event_type == EVENT_TYPE_BIASCOR

    cols = _resolve(table, model.fitres_aliases)
    lcpar, covmat_fit, baderr = model.read_lcpar(cols, config._model_inputs)
    lcpar = {k: np.asarray(v) for k, v in lcpar.items()}
    required = _required_lcpar(model, config, biascor=biascor, ccprior=ccprior)
    for name in sorted(required - lcpar.keys()):
        aliases = model.fitres_aliases.get(name, (name,))
        lcpar[name] = cols[name] if name in cols else _resolve(table, {name: aliases})[name]
    covmat_fit = fix_covariance(np.asarray(covmat_fit))
    badcov = model.bbc_bad_covariance(cols, covmat_fit)
    if baderr is not None:
        baderr = np.asarray(baderr)

    zhd = _read_redshift(table, config)
    zhel = _f32(table.get("zHEL", zhd))
    zhel = np.where(zhel < 0.0, zhd, zhel)   # zhel<0 -> zhd (SALT2mu.c:8989)
    # muerr_z uses only the vpec component (restore_sigz=0 default, SALT2mu.c:9202),
    # NOT zhderr.
    # VPEC_ERR before VPECERR, SNANA's read order (SALT2mu.c:8651)
    vpecerr = _f32(table.get("VPEC_ERR", table.get("VPECERR", np.zeros(table.size))))
    if config.zpecerr > 1.0e-9:   # user value REPLACES the file's (:9012)
        vpecerr = np.full(table.size, config.zpecerr * LIGHT_KM)
    zmuerr = vpecerr / LIGHT_KM
    # check_vpec_sign needs the velocity itself and the CMB-frame z (SALT2mu.c:4192)
    vpec = _f32(table.get("VPEC", np.zeros(table.size)))
    # absent zCMB stays at SNANA's -9 sentinel (SALT2mu.c:8547), which makes
    # check_vpec_sign's "zHD < zCMB" test false, as it is there
    zcmb = _f32(table.get("zCMB", np.full(table.size, SNANA_NULLFLOAT)))
    logmass = _f32(table.get(config.varname_gamma,
                             np.full(table.size, SNANA_NULLFLOAT)))
    # anything below -990 reads as "not measured" (SALT2mu.c:9307)
    logmass = np.where(logmass < -990.0, SNANA_NULLFLOAT, logmass)

    sim_truth = {}
    for raw, cands in {**_SIM_TRUTH, **model.biascor_truth_aliases}.items():
        col = next((c for c in cands if table.has(c)), None)
        if col is not None:
            sim_truth[raw] = _f32(table.get(col))

    # without a template index the input is real data and no SIM column is read
    # (SALT2mu.c:8822-8832); force_realdata does the same for data (:8825)
    if "SIM_TEMPLATE_INDEX" not in sim_truth and event_type == EVENT_TYPE_BIASCOR:
        raise SNANAError("biasCor input has no SIM_TEMPLATE_INDEX or SIM_NONIA_INDEX "
                         "column, so it is not a simulation", function_name="convert_table")
    if ("SIM_TEMPLATE_INDEX" not in sim_truth
            or (event_type == EVENT_TYPE_DATA and config.force_realdata)):
        sim_truth = {}
    if sim_truth:
        missing = [c for c in ("SIM_ZCMB", "SIM_DLMAG") if c not in sim_truth]
        if missing:
            raise SNANAError(f"simulated input lacks the required column(s) {missing}",
                             function_name="convert_table")
    if sim_truth and "SIM_WGT_POP" not in sim_truth:
        # every row starts at unit population weight (SALT2mu.c:8557)
        sim_truth["SIM_WGT_POP"] = np.ones(table.size)

    if config.uzsim and "SIM_ZCMB" in sim_truth:
        zhd = sim_truth["SIM_ZCMB"]          # z-cheat, before mumodel (:9048)

    mumodel = _mumodel(zhd, zhel, config, event_type, sim_truth, n_pad=cosmology_width)

    # SNANA shifts zhd at :9316, after mumodel is stored at :9254, so a fixed-cosmology
    # mumodel does not see the shift while z-binning and the biasCor lookup do.
    is_photoz = _is_photoz(table, config)
    shift = config.zphot_shift
    if event_type == EVENT_TYPE_DATA and shift != 0.0:
        zhd = np.where(is_photoz, zhd + shift, zhd)

    # absent IDSURVEY stays at SNANA's -9 (SALT2mu.c:8533)
    idsurvey = np.asarray(table.get("IDSURVEY", np.full(table.size, -9)), dtype=np.int64)
    # one sample without a biasCor; with one, the map builder assigns IDSAMPLE
    idsample = np.zeros(table.size, dtype=np.int64)
    # SNTYPE is read from the TYPE column (SALT2mu.c:8607 "TYPE:S"), -9 if absent
    # (SALT2mu.c:8540), which equals typeIa_ccprior's default -9.
    sntype = np.asarray(table.get("TYPE", np.full(table.size, -9)), dtype=np.int64)

    pIa = None
    if config.varname_pia and table.has(config.varname_pia):
        pIa = np.asarray(table.get(config.varname_pia), dtype=float)

    snrmax = _f32(table.get("SNRMAX1")) if is_biascor and table.has("SNRMAX1") else None
    fields = np.asarray(table.get("FIELD")).astype(str) if table.has("FIELD") else None

    tv = TableVar(
        event_type=event_type, name=np.asarray(table.get("CID")).astype(str),
        idsurvey=idsurvey, idsample=idsample, zhd=zhd, zmuerr=zmuerr,
        logmass=logmass, lcpar=lcpar, covmat_fit=covmat_fit, mumodel=mumodel,
        cutbits=np.zeros(table.size, dtype=np.int64), snrmax=snrmax, field=fields,
        baderr=baderr, badcov=badcov, is_photoz=is_photoz, vpec=vpec, zcmb=zcmb,
        version_photometry=table.meta.get("VERSION_PHOTOMETRY", ""),
        sim_cosmology=({key: table.meta[f"SIM_{key}"] for key in SIM_COSMOLOGY_KEYS}
                       if "SIM_OMEGA_MATTER" in table.meta else {}),
        pIa=pIa, sntype=sntype, sim_truth=sim_truth,
    )
    return tv


# SALT2mu-appended output columns; the input list is chopped at the first of these
# so a SALT2mu output FITRES can be fed straight back in (IS_APPEND_VARNAME, :24484).
_APPEND_VARNAMES = ("CUTMASK", "MU", "MUMODEL")

MISSING_TOKEN = str(SNANA_NULLFLOAT)   # write_fitres_line fill (:24533)


def _chop_appended(varnames) -> List[str]:
    out = []
    for n in varnames:
        if n in _APPEND_VARNAMES:
            break
        out.append(n)
    return out


def output_varnames(varnames_by_file, config) -> List[str]:
    """Input columns to carry into the output FITRES (store_output_varnames, :8686).

    A single datafile keeps every column. Several keep those present in all of them,
    then any name matching ``append_varname_missing``, filled with -9 where absent.
    A trailing ``*`` matches any name containing the stem, otherwise it is exact.
    """
    per_file = [_chop_appended(v) for v in varnames_by_file]
    if len(per_file) == 1:
        return per_file[0]

    others = [set(v) for v in per_file[1:]]
    keep = [n for n in per_file[0] if all(n in o for o in others)]

    patterns = [p for p in config.append_varname_missing.split(",") if p]
    seen = set(keep)
    for pattern in patterns:
        wildcard = pattern.endswith("*")
        stem = pattern[:-1] if wildcard else pattern
        for names in per_file:
            for n in names:
                if n in seen:
                    continue
                if stem in n if wildcard else n == stem:
                    keep.append(n)
                    seen.add(n)
    return keep


def force_probcc0(tv, config) -> np.ndarray:
    """Mask of spec-confirmed SNe forced to pIa=1: SNTYPE in {typeIa_ccprior} U
    type_list_probcc0, or IDSURVEY in idsurvey_list_probcc0 (SALT2mu.c:22772)."""
    sntype = np.asarray(tv.sntype)
    m = sntype == config.typeIa_ccprior
    for t in config.type_list_probcc0:
        m = m | (sntype == t)
    for i in config.idsurvey_list_probcc0:
        m = m | (np.asarray(tv.idsurvey) == i)
    return m


def _require_pia_column(tv, config) -> None:
    """A data table without the varname_pIa column is usable only when every
    event is forced to pIa=1, which store_input_varnames counts only under
    an idsurvey_list_probcc0 (SALT2mu.c:9512, :9542-9579); the biasCor and
    CC-prior tables need the column only for a CUTWIN on it."""
    forced = (force_probcc0(tv, config) if config.idsurvey_list_probcc0
              else np.zeros(tv.size, dtype=bool))
    if forced.all():
        return
    raise SNANAError(
        f"Missing column varname_pIa='{config.varname_pia}': "
        f"{int((~forced).sum())} of {tv.size} events have no pIa and are not "
        f"forced to 1 by idsurvey_list_probcc0",
        function_name="convert_table")


def passthrough_column(table, name) -> np.ndarray:
    """The column's original tokens, with SNANA's fill where a file lacked it.

    write_fitres_line (SALT2mu.c:24501) re-emits the input words verbatim rather
    than reformatting them, so "0.00100" stays "0.00100" and a missing column is
    the literal "-9.0". Reading them as text is what preserves the input's stated
    precision; the fit uses the separately parsed numeric columns.
    """
    col = table.get(name)
    return np.array([MISSING_TOKEN if v is None or (isinstance(v, float) and np.isnan(v))
                     else str(v) for v in col], dtype=object)


def _is_photoz(table, config) -> np.ndarray:
    """IS_SPECZ_TABLEVAR (SALT2mu.c:9384), negated.

    zPRIOR/zPRIORERR are read only when the key is set and both are present
    (:8638, :8645); otherwise the comparison falls back to zHELERR, whose
    absent-column default of -9.0 (:8550) classifies the whole table as spec-z.
    """
    errmax = config.zspec_errmax_idsample
    opt_photoz = np.asarray(table.get("OPT_PHOTOZ", np.zeros(table.size)), dtype=np.int64)
    if errmax > 0.0 and table.has("zPRIOR") and table.has("zPRIORERR"):
        zerr = _f32(table.get("zPRIORERR"))
    else:
        zerr = _f32(table.get("zHELERR", np.full(table.size, _ZHELERR_ABSENT)))
    return ~((opt_photoz == 0) | (zerr < errmax))


def _mumodel(zhd, zhel, config, event_type, sim_truth, *, n_pad=None) -> np.ndarray:
    if event_type == EVENT_TYPE_CCPRIOR:
        # the CC map is built at the cosmology the user gave, not the blinded one
        # (COSPAR_UNBLIND, SALT2mu.c:17843); the zhd approximation is kept (:17842,
        # it averages out in the DMU-PDF).
        return mu_at(zhd, unblinded(config))
    dl_hd = mu_at(zhd, config, n_pad=n_pad)
    if event_type != EVENT_TYPE_BIASCOR:
        if not config.muref_zpoly:
            # zhel flux-dimming prefactor (SALT2mu.c:9252)
            dl_hd = dl_hd + 5.0 * np.log10((1.0 + zhel) / (1.0 + zhd))
        return dl_hd
    # biasCor reference is anchored to sim truth by a dl-ratio (compute_more_TABLEVAR):
    # mumodel = SIM_DLMAG + [mu(zhd) - mu(SIM_ZCMB)].
    return sim_truth["SIM_DLMAG"] + (
        dl_hd - mu_at(sim_truth["SIM_ZCMB"], config))


def accept_zrate_scale(names, zhd, path) -> np.ndarray:
    """Deterministic z-dependent downsampling (accept_zrate_scale, SALT2mu.c:22503):
    the last three digits of the CID act as the draw, kept when <= the linearly
    interpolated scale(z). Rows are "z scale"; blanks and comments are skipped."""
    zs, scales = [], []
    with open_text(path) as stream:
        for line in stream:
            w = line.split()
            if not w or line.startswith((" ", "#")):
                continue
            zs.append(float(w[0]))
            scales.append(float(w[1]))
    if not zs:
        return np.ones(len(names), dtype=bool)
    zhd = np.asarray(zhd)
    outside = (zhd < zs[0]) | (zhd > zs[-1])
    if len(zs) > 1 and outside.any():
        raise SNANAError(f"z={zhd[outside][0]:.5f} is outside the zrate_scale_file "
                         f"range {zs[0]} to {zs[-1]}",
                         function_name="accept_zrate_scale")
    last3 = np.array([int(str(n)[-3:]) if len(str(n)) > 3 else int(str(n))
                      for n in names], dtype=float)
    return (last3 / 1000.0) <= np.interp(zhd, zs, scales)


def read_cid_selection(paths) -> dict:
    """Event selectors from FITRES-format or bare CID lists.

    IDSURVEY and FIELD, when present, distinguish repeated CIDs. Missing
    identity columns are wildcards. IZBIN is optional.
    """
    selected = {}
    for path in paths:
        columns = None
        with open_text(path) as fh:
            for line in fh:
                words = line.split()
                if not words or words[0].startswith("#"):
                    continue
                if words[0] == "VARNAMES:":
                    columns = words[1:]
                    if "CID" not in columns:
                        raise SNANAError(f"{path} has a VARNAMES row without CID",
                                         function_name="read_cid_selection")
                elif columns is not None and words[0].endswith(":"):
                    row = dict(zip(columns, words[1:]))
                    identity = (row["CID"], int(float(row["IDSURVEY"])) if "IDSURVEY" in row else None,
                                row.get("FIELD"))
                    selected[identity] = int(float(row.get("IZBIN", -9)))
                elif columns is None and not words[0].endswith(":"):
                    selected[(words[0], None, None)] = -9
    return selected


def match_cid_selection(tv, paths):
    """Return a membership mask and any selected redshift bins in table order."""
    selected = read_cid_selection(paths)
    fields = tv.field if tv.field is not None else ["NULL"] * len(tv.name)
    bins = []
    keep = []
    for cid, survey, field in zip(tv.name, tv.idsurvey, fields):
        candidates = ((str(cid), int(survey), str(field)), (str(cid), int(survey), None),
                      (str(cid), None, str(field)), (str(cid), None, None))
        match = next((key for key in candidates if key in selected), None)
        keep.append(match is not None)
        bins.append(selected[match] if match is not None else -9)
    return np.asarray(keep, dtype=bool), np.asarray(bins, dtype=int)


def merge_duplicates(tv: TableVar, config: BBCConfig) -> np.ndarray:
    """Merge numerical values and return duplicate rows, independently of cuts."""
    duplicates = np.zeros(tv.size, dtype=bool)
    flag = config.iflag_duplicate
    if flag == 0:
        return duplicates
    order = np.argsort(tv.name, kind="stable")
    groups = [g for g in np.split(order, np.where(
        tv.name[order][1:] != tv.name[order][:-1])[0] + 1) if len(g) > 1]
    if not groups:
        return duplicates
    if flag == 1:
        raise SNANAError(f"{len(groups)} duplicate CID(s) in the data, e.g. "
                         f"{tv.name[groups[0][0]]}; set iflag_duplicate=0 or 2",
                         function_name="merge_duplicates")

    names = list(tv.lcpar)
    # merging writes back into the arrays, which the reader may hand out read-only
    tv.lcpar = {k: np.array(v, copy=True) for k, v in tv.lcpar.items()}
    tv.covmat_fit = np.array(tv.covmat_fit, copy=True)
    sqsigint = config.sigint ** 2
    for g in groups:
        g = np.sort(g)
        keep = g[0]
        # weight uses covmat_tot (fit + intrinsic on the brightness axis)
        wgt = np.stack([1.0 / (tv.covmat_fit[g, i, i] + (sqsigint if i == 0 else 0.0))
                        for i in range(len(names))], axis=1)      # [ndup, k]
        wsum = wgt.sum(axis=0)
        for i, nm in enumerate(names):
            tv.lcpar[nm][keep] = float((wgt[:, i] * tv.lcpar[nm][g]).sum() / wsum[i])
        cov = tv.covmat_fit[g]
        # 1/sum(1/cov), falling back to the plain mean where a term changes sign
        with np.errstate(divide="ignore", invalid="ignore"):
            harmonic = 1.0 / (1.0 / cov).sum(axis=0)
        sign_change = (cov * cov[0]).min(axis=0) < 0.0
        tv.covmat_fit[keep] = np.where(sign_change, cov.mean(axis=0), harmonic)
        duplicates[g[1:]] = True
    return duplicates


def compute_cutbits(tv: TableVar, config: BBCConfig, model, table) -> np.ndarray:
    """Per-event bitmask of failed cuts: inclusive CUTWIN windows, the model's default
    cuts, and the z range. Bit numbering is SNANA's CUTMASK (SALT2mu.c:614).

    The sample-defining cuts come first: the z range, the CID lists and the
    sim prescales, and covariance-quality cuts. CUTWIN NONE retains these
    and the subsequent bias-correction validity cuts for data
    (setbit_CUTMASK, SALT2mu.c:19016)."""

    quantities = {config.varname_z: tv.zhd, "zHD": tv.zhd, "logmass": tv.logmass}
    for name, arr in tv.lcpar.items():
        quantities[name] = np.asarray(arr)

    bits = np.zeros(tv.size, dtype=np.int64)

    def fail(bad, bit):
        bits[bad] |= 1 << bit

    if tv.event_type != EVENT_TYPE_BIASCOR:
        fail(~((tv.zhd >= config.zmin) & (tv.zhd <= config.zmax)), CUTBIT_z)
    # a biasCor event's zmax is widened by its own sample's z binsize (:17981),
    # so its z cut waits for the IDSAMPLE partition (samples.apply_biascor_zcut)
    if tv.event_type == EVENT_TYPE_DATA:
        if config.cid_select_file:
            fail(~match_cid_selection(tv, config.cid_select_file)[0], CUTBIT_CID)
        if config.cid_reject_file:
            fail(match_cid_selection(tv, config.cid_reject_file)[0], CUTBIT_CID)

    # Prescales are deterministic counters over the rows in file order, not random
    # sampling (prescale_reject_simData / _biasCor).
    is_sim = bool(tv.sim_truth)
    if tv.event_type == EVENT_TYPE_DATA and is_sim:
        if config.prescale_simdata > 1.0:
            n_sim = np.arange(1, tv.size + 1, dtype=float)   # NSIMDATA, 1-based
            fail(np.fmod(n_sim, config.prescale_simdata) != 0, CUTBIT_SIMPS)
        tmpl = tv.sim_truth.get("SIM_TEMPLATE_INDEX")
        if tmpl is not None and (config.prescale_simcc > 1.0):
            is_cc = np.asarray(tmpl) != 0
            n_cc = np.cumsum(is_cc).astype(float)            # NSIMCC, CC rows only
            fail(is_cc & ((np.fmod(n_cc, config.prescale_simcc) != 0)
                          | (config.prescale_simcc > 9999.0)), CUTBIT_SIMPS)
        if tmpl is not None and (config.prescale_simia > 1.0):
            is_ia = np.asarray(tmpl) == 0                     # NSIMIa, Ia rows only
            n_ia = np.cumsum(is_ia).astype(float)
            fail(is_ia & (np.fmod(n_ia, config.prescale_simia) != 0), CUTBIT_SIMPS)
        if tmpl is not None and config.zrate_scale_file:
            # only simulated SNIa are scaled, at their true redshift
            is_ia = np.asarray(tmpl) == 0
            accept = np.ones(tv.size, dtype=bool)
            accept[is_ia] = accept_zrate_scale(tv.name[is_ia], tv.sim_truth["SIM_ZCMB"][is_ia],
                                               config.zrate_scale_file)
            fail(~accept, CUTBIT_SIMPS)
    if tv.event_type == EVENT_TYPE_BIASCOR:
        subset, prescale = config.prescale_biascor
        if prescale > 1:                                     # row index, not a counter
            fail(np.fmod(np.arange(tv.size), prescale) != subset, CUTBIT_SIMPS)
        # only true SNIa enter the biasCor sample (SALT2mu.c:18954-18955)
        tmpl = tv.sim_truth.get("SIM_TEMPLATE_INDEX")
        if tmpl is not None:
            fail(np.asarray(tmpl) != 0, CUTBIT_TRUESNIa)
        if config.zrate_scale_file:
            # every event is looked up, but only SNIa are rejected
            accept = accept_zrate_scale(tv.name, tv.sim_truth["SIM_ZCMB"],
                                        config.zrate_scale_file)
            if tmpl is not None:
                fail((np.asarray(tmpl) == 0) & ~accept, CUTBIT_SIMPS)
    if tv.event_type == EVENT_TYPE_CCPRIOR:
        # only true non-Ia enter the CC prior (SALT2mu.c:18983-18985)
        tmpl = tv.sim_truth.get("SIM_TEMPLATE_INDEX")
        if tmpl is not None:
            fail(np.asarray(tmpl) == 0, CUTBIT_TRUESNCC)

    if tv.badcov is not None:
        fail(tv.badcov, CUTBIT_BADCOV)
    if tv.event_type == EVENT_TYPE_DATA and config.cutwin_disable:
        return bits

    if tv.baderr is not None:
        fail(tv.baderr, CUTBIT_BADERR)
    if config.fieldlist and tv.field is not None:
        # select_FIELD (SALT2mu.c:20797): substring match against any listed field
        keep = np.array([any(f in str(v) for f in config.fieldlist) for v in tv.field])
        fail(~keep, CUTBIT_FIELD)
    fail(~((tv.logmass >= config.logmass_min)
           & (tv.logmass <= config.logmass_max)), CUTBIT_logmass)
    bit_of = {name: pair[0] for name, pair in model.bbc_cut_bits.items()}
    for var, (lo, hi) in lcpar_windows(model, config).items():
        if var in quantities:
            bad = ~((quantities[var] >= lo) & (quantities[var] <= hi))
            logger.debug("BBC window %s [%g, %g]: %d rejected", var, lo, hi, bad.sum())
            fail(bad, bit_of.get(var, CUTBIT_CUTWIN))
    # A generic CUTWIN cuts on any column of the table; an absent column aborts
    # (malloc_TABLEVAR_CUTVAL, SALT2mu.c:8359; set_DOFLAG_SELECT_VAR, :21380)
    for var, lo, hi in config.cutwin:
        if var in quantities:
            values = quantities[var]
        elif table.has(var):
            values = _f32(table.get(var))
        else:
            raise SNANAError(f"Invalid CUTWIN on '{var}': no such column in the "
                             f"input table", function_name="compute_cutbits")
        fail(~((values >= lo) & (values <= hi)), CUTBIT_CUTWIN)

    if config.sntype_select and tv.sntype is not None:
        fail(~np.isin(tv.sntype, config.sntype_select), CUTBIT_sntype)
    if config.idsurvey_select:
        fail(~np.isin(tv.idsurvey, config.idsurvey_select), CUTBIT_IDSURVEY)

    # applyCut_nmax runs last, over events that survived every other cut: keep the
    # first nmax in file order (SALT2mu.c:3206).
    if tv.event_type == EVENT_TYPE_DATA and config.nmax is not None:
        passing = np.where(bits == 0)[0]
        if passing.size > config.nmax:
            fail(passing[config.nmax:], CUTBIT_NMAXCUT)
    return bits


def write_passcuts_list(path, which, tv, idsample):
    """SNANA update_fitres_passcuts: the CIDs of a biasCor/CC-prior sample that pass
    the BBC cuts, as a diagnostic list. No effect on the fit.

    ``idsample`` is the per-event sample assignment, which is only known once the
    samples are partitioned, hence after the maps are built.
    """
    has_field = tv.field is not None
    with open(path, "w") as f:
        f.write(f"# list of {which} events passing BBC cuts \n\n")
        f.write("VARNAMES:  CID  IDSAMPLE  IDSURVEY" + ("  FIELD\n" if has_field else "\n"))
        for i in np.where(tv.cutmask)[0]:
            row = (f"SN: {str(tv.name[i]):>12s}  {int(idsample[i]):2d}  "
                   f"{int(tv.idsurvey[i]):3d}  ")
            if has_field:
                row += f"{tv.field[i]} "
            f.write(row + "\n")
