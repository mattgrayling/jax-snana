"""ZVARIATION: additive z-dependent shifts of population parameters.

Mirrors SNANA init_zvariation / get_zvariation (snlc_sim.c:22736-23190):
each entry names a sim parameter (GEN<component>_<par>, GENMAG_OFF_GLOBAL,
EBV_IGM, or the AV/EBV exponential-profile family) and gives its shift vs
zCMB as either a polynomial (ZPOLY, arbitrary order comma-separated or
legacy space-separated order 3) or linear interpolation in z bins (ZBIN).

Init replicates SNANA's checks: ZVARIATION_FILE and ZVARIATION_POLY keys
are mutually exclusive; every parameter must be on the PARDEF registry;
ZBIN ranges must cover GENRANGE_REDSHIFT; every shift must vanish at z=0.
The z=0 check also reproduces SNANA's izmin abort: a ZBIN entry whose
second bin lies above z=0 is fatal, since SNANA needs a leading bin below
the generation range.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core import logging as log
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.utils import eval_genpoly

logger = logging.getLogger(__name__)

# init_zvariation NPREFIX_GENGAUSS list; GENMEAN and GENPEAK sum into the peak
GENGAUSS_COMPONENT_NAMES = {
    'peak': ('GENPEAK_{p}', 'GENMEAN_{p}'),
    'sigma_lo': ('GENSIGMA[0]_{p}',), 'sigma_hi': ('GENSIGMA[1]_{p}',),
    'skew_lo': ('GENSKEW[0]_{p}',), 'skew_hi': ('GENSKEW[1]_{p}',),
    'prob2': ('GENPROB2_{p}',), 'peak2': ('GENPEAK2_{p}',),
    'sigma2_lo': ('GENSIGMA2[0]_{p}',), 'sigma2_hi': ('GENSIGMA2[1]_{p}',),
}
# gen_AV shifts only EXP_TAU, SIGMA and RATIO (snlc_sim.c:18157-18186)
EXPHALFGAUSS_COMPONENT_NAMES = {
    'AV': {'exp_tau': ('GENEXPTAU_AV', 'GENTAU_AV'),
           'sigma': ('GENGAUSIG_AV', 'GENSIG_AV'),
           'ratio': ('GENRATIO_AV0',)},
    'EBV_HOST': {'exp_tau': ('GENTAU_EBV_HOST',),
                 'sigma': ('GENSIG_EBV_HOST',),
                 'ratio': ('GENRATIO_EBV0_HOST',)},
}
PREFIX_GENGAUSS = (
    'PEAK', 'MEAN', 'SKEW[0]', 'SKEW[1]', 'SIGMA[0]', 'SIGMA[1]',
    'PROB2', 'PEAK2', 'SIGMA2[0]', 'SIGMA2[1]', 'EXPSIG', 'EXPTAU',
)

# miscellaneous non-GENGAUSS parameters (init_zvariation :22625-22640)
MISC_PARDEF = (
    'GENEXPTAU_AV', 'GENTAU_AV', 'GENGAUSIG_AV', 'GENSIG_AV',
    'GENGAUPEAK_AV', 'GENPEAK_AV', 'GENRATIO_AV0',
    'GENTAU_EBV_HOST', 'GENSIG_EBV_HOST', 'GENPEAK_EBV_HOST',
    'GENRATIO_EBV0_HOST',
    'VSI', 'GENMAG_OFF_GLOBAL', 'EBV_IGM',
)

#: the quantity a ZVARIATION shift depends on
REDSHIFT = 'REDSHIFT'


@dataclass
class ZVariationEntry:
    """One ZVARIATION entry: a parameter's additive shift as a function of
    one per-SN quantity, as a polynomial or z-bin interpolation table."""
    parname: str
    poly: Optional[np.ndarray] = None          # coefficients, c0 + c1 x + ...
    zbins: Optional[np.ndarray] = None         # (n, 2): x, shift
    source: str = ''
    #: the per-SN quantity the shift varies with; zCMB is SNANA's only one
    depends_on: str = REDSHIFT
    #: the value of that quantity at which the shift vanishes, so the
    #: population key the entry modifies means its value there
    pivot: float = 0.0

    def check_scalar_shift(self, z: float) -> float:
        """get_zvariation for one z, with SNANA's ZBIN izmin abort."""
        if self.poly is not None:
            return float(shift_array(self, z))
        zb, fb = self.zbins[:, 0], self.zbins[:, 1]
        izmin = 0
        for iz in range(len(zb)):
            if zb[iz] <= z:
                izmin = iz
        if izmin <= 0:
            raise SNANAError(
                f"izmin=0 for {self.parname} and z={z:.3f} (SNANA "
                f"get_zvariation abort: need a ZBIN below z)")
        izmax = izmin + 1
        return float(fb[izmin] + (fb[izmax] - fb[izmin])
                     * (z - zb[izmin]) / (zb[izmax] - zb[izmin]))


def _parse_poly(text: str, parname: str, require_comma: bool) -> np.ndarray:
    if require_comma and ',' not in text:
        raise SNANAError(
            f"Must use comma-separated poly coefficients for ZPOLY with "
            f"param = '{parname}'")
    parts = text.split(',') if ',' in text else text.split()
    return np.array([float(p) for p in parts])


def parse_zvariation_file(path: str) -> List[ZVariationEntry]:
    """ZVARIATION_FILE format: PARAMETER: / ZPOLY: / ZBIN: rows, END: stops."""
    entries: List[ZVariationEntry] = []
    tokens = []
    with open(path) as f:
        for line in f:
            line = line.split('#')[0]
            tokens.extend(line.split())
    i = 0
    current: Optional[ZVariationEntry] = None
    zbins: List[Tuple[float, float]] = []

    def close_current():
        if current is not None and zbins:
            current.zbins = np.array(zbins)

    while i < len(tokens):
        tok = tokens[i]
        if tok == 'END:':
            break
        if tok == 'PARAMETER:':
            close_current()
            zbins = []
            current = ZVariationEntry(parname=tokens[i + 1], source=path)
            entries.append(current)
            i += 2
            continue
        if tok == 'ZPOLY:':
            if current.poly is not None or zbins:
                raise SNANAError(
                    f"Found ZPOLY but variation already set for "
                    f"{current.parname}. Check file: {path}")
            current.poly = _parse_poly(tokens[i + 1], current.parname,
                                       require_comma=True)
            i += 2
            continue
        if tok == 'ZBIN:':
            zbins.append((float(tokens[i + 1]), float(tokens[i + 2])))
            i += 3
            continue
        i += 1
    close_current()
    return entries


def parse_zvariation_poly_keys(poly_keys: List[Tuple[str, str]]) -> List[ZVariationEntry]:
    """ZVARIATION_POLY: <par> <poly> sim-input keys (comma or legacy space)."""
    return [ZVariationEntry(parname=name,
                            poly=_parse_poly(text, name, require_comma=False),
                            source='ZVARIATION_POLY')
            for name, text in poly_keys]


def parse_variation_keys(values: List[str]) -> List[ZVariationEntry]:
    """`VARIATION: <parameter> <quantity> POLY <coeffs> PIVOT <value>`.

    Coefficients start at the linear term: PIVOT fixes the constant, so the
    shift vanishes there and slope and anchor stay independent. `ZPOLY`'s
    start at the constant, SNANA anchoring that shift at z = 0.
    """
    entries = []
    for text in values:
        tokens = str(text).split()
        if len(tokens) < 6 or tokens[2] != 'POLY' or 'PIVOT' not in tokens:
            raise SNANAError(
                f"VARIATION takes '<parameter> <quantity> POLY "
                f"<coefficients> PIVOT <value>', got '{text}'",
                function_name="parse_variation_keys")
        ipivot = tokens.index('PIVOT')
        if ipivot == 3 or len(tokens) != ipivot + 2:
            raise SNANAError(
                f"VARIATION needs at least one POLY coefficient and "
                f"exactly one PIVOT value, got '{text}'",
                function_name="parse_variation_keys")
        entries.append(ZVariationEntry(
            parname=tokens[0], depends_on=tokens[1],
            pivot=float(tokens[ipivot + 1]),
            poly=np.array([0.0] + [float(t) for t in tokens[3:ipivot]]),
            source='VARIATION'))
    return entries


def build_pardef(model_parnames: List[str], dynamic: bool) -> set:
    """Valid parameter registry (init_zvariation :22430-22500).

    The misc names register only when the model does not declare its
    parameters in the input (snlc_sim.c:22868-22882)."""
    pars = ['RV', 'AV'] + list(model_parnames)
    names = set()
    for prefix in PREFIX_GENGAUSS:
        for p in pars:
            names.add(f'GEN{prefix}_{p}')
    if not dynamic:
        names.update(MISC_PARDEF)
    return names


def _pardef_for(spellings, names_from_input) -> set:
    """The parameter names a variation may target, for the running model."""
    model_pars = sorted(set(spellings.values()))
    # SNANA's MISC-pardef gate is run-level NPAR_SIMSED > 0, which
    # SIMSED_GRIDONLY: SEQUENTIAL fails (snlc_sim.c:6064, :22865)
    return build_pardef(model_pars, names_from_input and bool(model_pars))


def spelled_by_name(key, stems, spellings) -> str:
    """``key`` with the population parameter after its stem spelled by
    name, so GENPEAK_SALT2x1 and GENPEAK_X1 both read as GENPEAK_x1.
    ``spellings`` maps each lower-cased spelling to its name; a key with
    none of ``stems``, or naming no parameter, is returned unchanged."""
    for stem in stems:
        if key.startswith(stem):
            name = spellings.get(key[len(stem):].lower())
            return key if name is None else stem + name
    return key


# the stems of the targets a shift may name on a GENGAUSS parameter
_TARGET_STEMS = tuple(f'GEN{prefix}_' for prefix in PREFIX_GENGAUSS)


def load_variation(values, spellings, names_from_input, hostlib_file,
                   scope='', extra_targets=()) -> Dict[str, List[ZVariationEntry]]:
    """Parse, validate and index VARIATION lines by the parameter they shift.

    A parameter may carry several, whose shifts sum with each other and with
    ZVARIATION's. A quantity naming a population parameter is that
    parameter's drawn value; anything else is a HOSTLIB column. ``scope``
    names the class the lines came from, for the messages;
    ``extra_targets`` are the non-parameter targets that scope allows, such
    as a class's WGT. ``hostlib_file`` is the run's HOSTLIB, None without
    one. ``spellings`` maps each lower-cased spelling of a population
    parameter to its name, by which the entries name it; whether the model
    declares its parameters in its input is ``names_from_input``.
    """
    if not values:
        return {}
    entries = parse_variation_keys(values)
    for e in entries:
        e.parname = spelled_by_name(e.parname, _TARGET_STEMS, spellings)
        e.depends_on = spellings.get(e.depends_on.lower(), e.depends_on)
    where = f"SUBPOP {scope} " if scope else ""
    parameters = set(spellings.values())
    from_host = sorted({e.depends_on for e in entries
                        if e.depends_on not in parameters})
    if from_host and hostlib_file is None:
        raise SNANAError(
            f"{where}VARIATION depends on {', '.join(from_host)}, which is not "
            f"a population parameter, so it must be a HOSTLIB column; this "
            f"run configures no HOSTLIB_FILE to read one from.",
            function_name="load_variation")

    targets = _pardef_for(spellings, names_from_input)
    targets.update(extra_targets)

    by_parameter: Dict[str, List[ZVariationEntry]] = {}
    for e in entries:
        if e.parname not in targets:
            raise SNANAError(
                f"{where}VARIATION targets '{e.parname}', which is not a "
                f"recognised SIM-parameter"
                + (f" or one of {', '.join(sorted(extra_targets))}"
                   if extra_targets else "") + ".",
                function_name="load_variation")
        by_parameter.setdefault(e.parname, []).append(e)
        log.line(logger, f"{where}VARIATION for {e.parname} vs {e.depends_on}")
    return by_parameter


def load_zvariation(poly_keys, zvar_file, spellings, names_from_input,
                    redshift_range) -> Dict[str, ZVariationEntry]:
    """Parse, validate (SNANA init checks) and index the ZVARIATION config,
    each entry naming its parameter as ``spellings`` maps it (see
    :func:`load_variation`)."""
    use_file = zvar_file and zvar_file.upper() not in ('', 'NONE')
    if use_file and poly_keys:
        raise SNANAError(
            "Cannot specify ZVARIATION_FILE and ZVARIATION_POLY keys in "
            "sim-input file. Pick one or the other.")
    if not use_file and not poly_keys:
        return {}

    entries = (parse_zvariation_file(zvar_file) if use_file
               else parse_zvariation_poly_keys(poly_keys))
    for e in entries:
        e.parname = spelled_by_name(e.parname, _TARGET_STEMS, spellings)

    pardef = _pardef_for(spellings, names_from_input)

    z_min, z_max = redshift_range
    for e in entries:
        if e.parname not in pardef:
            raise SNANAError(
                f"Unrecognized SIM-parameter '{e.parname}'. "
                f"Cannot apply Z-dependence.")
        if e.zbins is not None:
            zb = e.zbins[:, 0]
            if zb[0] > z_min or zb[-1] < z_max:
                raise SNANAError(
                    f"ZBIN range {zb[0]:.3f} - {zb[-1]:.3f} for {e.parname} "
                    f"does not cover ZGEN-range {z_min:.3f} - {z_max:.3f}")
        shift0 = e.check_scalar_shift(0.0)
        if shift0 != 0.0:
            raise SNANAError(
                f"{e.parname}-shift = {shift0} at z=0, "
                f"but shift(z=0) must be zero")
        log.line(logger, f"ZVARIATION for {e.parname}: "
                 f"{'ZPOLY' if e.poly is not None else 'ZBINS'}")

    return {e.parname: e for e in entries}


def shift_array(entry: ZVariationEntry, values):
    """Traced per-SN shift for a jnp array of the quantity the entry
    depends on.

    ``pivot`` is zero for a ZVARIATION entry, whose shift SNANA anchors at
    z = 0 instead.
    """
    x = jnp.asarray(values) - entry.pivot
    if entry.poly is not None:
        return eval_genpoly(entry.poly, x)
    zb, fb = entry.zbins[:, 0], entry.zbins[:, 1]
    return jnp.interp(x, jnp.asarray(zb), jnp.asarray(fb))


jax.tree_util.register_dataclass(
    ZVariationEntry,
    data_fields=[
        'poly',
        'zbins',
        'pivot',
    ],
    meta_fields=[
        'parname',
        'source',
        'depends_on',
    ],
)
