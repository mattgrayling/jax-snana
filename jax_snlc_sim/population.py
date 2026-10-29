"""How each population parameter is drawn.

The config parser builds one record per parameter, and every consumer reads
it rather than assembling an answer from the input keys itself.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Optional, Tuple

import jax
import numpy as np

from jax_snlc_core.input_settings import build_dataclass, convert
from jax_snlc_core.logging import SNANAError
from jax_snlc_sim.distributions import (EXPHALFGAUSS_ZSHIFT_FIELDS, GENGAUSS_ZSHIFT_FIELDS,
                                        GenExpHalfGauss, GenGaussAsym, flat_distribution,
                                        mixture_mass, validate_gaussian)
from jax_snlc_sim.zvariation import (EXPHALFGAUSS_COMPONENT_NAMES, GENGAUSS_COMPONENT_NAMES, REDSHIFT,
                                     ZVariationEntry, shift_array)

#: points per quantity when bounding a shift over that quantity's range
_GRID_POINTS = 101


@dataclass(frozen=True)
class HostDustParameter:
    """One host-dust quantity the sampler draws for every event, whatever the
    SED model, alongside that model's own population parameters."""

    #: the distributions and GENPDF maps key
    key: str
    #: spellings of the same quantity in GENGAUSS and GENPDF input
    input_names: Tuple[str, ...]
    #: the HOSTLIB column USESNPAR reads it from
    hostlib_column: str
    #: an exp + half-Gaussian profile rather than a GENGAUSS population
    exp_halfgauss: bool
    #: the per-SN parameter it lands in, where it has one of its own. EBV is a
    #: parameterisation of AV, so the drawn value is AV either way.
    param: Optional[str] = None


#: SNANA grants the dust HOSTLIB columns to SALT alone and accepts AV and RV as
#: WGTMAP dimensions for any model; jax grants both to every model, since every
#: model draws host dust.
HOST_DUST_PARAMETERS = (
    HostDustParameter(key='RV', input_names=('RV',), hostlib_column='RV',
                      exp_halfgauss=False, param='rv'),
    HostDustParameter(key='AV', input_names=('AV', 'AV_HOST'),
                      hostlib_column='AV', exp_halfgauss=True, param='av'),
    HostDustParameter(key='EBV_HOST', input_names=('EBV', 'EBV_HOST'),
                      hostlib_column='EBV', exp_halfgauss=True),
)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Population:
    """One parameter's draw: a GENPDF map, or a distribution per SUBPOP class
    with the shifts that move it.

    There is a single distribution where the classes do not differ on the
    parameter, so it is drawn once rather than once per class. ``shifts`` runs
    alongside, one tuple per distribution; several shifts on one component sum,
    which is how redshift evolution and an environmental relation compose.
    """

    distributions: Tuple[Any, ...] = ()
    shifts: Tuple[Tuple[ZVariationEntry, ...], ...] = ()
    genpdf: Any = None

    def __post_init__(self):
        if self.genpdf is not None and (self.distributions or self.shifts):
            raise SNANAError(
                "A GENPDF map is the tabulated form of the statement a "
                "distribution and its shifts make, so a parameter cannot "
                "carry both.", function_name="Population")

    def depends_on(self) -> Tuple[str, ...]:
        """Every quantity the draw depends on."""
        if self.genpdf is not None:
            return tuple(self.genpdf.cond_varnames)
        return tuple(s.depends_on for shifts in self.shifts for s in shifts)

    def is_shifted(self) -> bool:
        """Whether any shift moves this parameter, in any class."""
        return any(self.shifts)

    def is_shaped(self) -> bool:
        """Whether the run says more about this parameter than one shared
        distribution.

        Anything that replaces the drawn value, a USESNPAR HOSTLIB column or a
        SIMLIB entry header, discards whatever more it says, so the guards
        refuse that combination rather than letting it pass silently.
        """
        if self.genpdf is not None:
            return bool(self.genpdf.cond_varnames)
        return len(self.distributions) > 1 or self.is_shifted()


def targets(param_name: str) -> frozenset:
    """The component spellings a shift on this parameter can name."""
    field_map, _ = component_names(param_name)
    return frozenset(t for names in field_map.values() for t in names)


def component_names(param_name: str):
    """The input names of each shiftable component of a parameter, and the
    order their shift arrays are passed in.

    The dust parameters are the exp + half-Gaussian ones, and they are the
    ones EXPHALFGAUSS_COMPONENT_NAMES names, so the parameter alone settles
    which family it belongs to.
    """
    if param_name in EXPHALFGAUSS_COMPONENT_NAMES:
        return (EXPHALFGAUSS_COMPONENT_NAMES[param_name],
                EXPHALFGAUSS_ZSHIFT_FIELDS)
    return ({f: tuple(t.format(p=param_name) for t in names)
             for f, names in GENGAUSS_COMPONENT_NAMES.items()},
            GENGAUSS_ZSHIFT_FIELDS)


def validate(param_name: str, pop: Population, ranges) -> None:
    """Refuse a record whose shifts cannot reach the draw or would leave the
    distribution invalid.

    ``ranges`` gives the declared range of each quantity a shift depends on,
    so the answer is a property of the run rather than of whichever events a
    batch drew. A two-component mixture must also have some probability in
    its range at its declared parameters.
    """
    for dist in pop.distributions:
        if isinstance(dist, GenGaussAsym) and mixture_mass(dist) <= 0.0:
            raise SNANAError(
                f"The {param_name} population's two components have no "
                f"probability in its range ({dist.range[0]}, "
                f"{dist.range[1]}), so no draw can land there; check "
                f"GENPEAK, GENPEAK2, GENSIGMA, GENSIGMA2 and GENPROB2.",
                function_name="validate")
    for dist, shifts in zip(pop.distributions, pop.shifts):
        if not shifts:
            continue
        lo, hi = dist.range
        if hi - lo < 1.0e-30:
            raise SNANAError(
                f"A shift on {param_name} cannot take effect: GENRANGE_"
                f"{param_name} is [{lo}, {hi}], a single value every draw is "
                f"pinned to. Widen the range or drop the shift.",
                function_name="validate")
        _assert_domain(param_name, dist, shifts, ranges)


def _shift_values(field_map, fields, shifts, ranges):
    """The total shift on each component, sampled over the ranges of the
    quantities the shifts depend on.

    Where every shift depends on the same quantity, all components are
    evaluated on that one grid, so the checks below that couple two
    components compare them point by point. With several quantities a
    component is summarised by the extremes it reaches; the extremes pair
    lowest with lowest, which is the corner each of those checks tests.
    """
    named = {s.depends_on for s in shifts}
    grids = {q: np.linspace(*ranges[q], _GRID_POINTS) for q in named}
    values = {}
    for fieldname in fields:
        terms = [np.asarray(shift_array(s, grids[s.depends_on]))
                 for s in shifts if s.parname in field_map.get(fieldname, ())]
        values[fieldname] = (
            sum(terms, np.zeros(_GRID_POINTS)) if len(named) == 1
            else np.array([sum(float(t.min()) for t in terms),
                           sum(float(t.max()) for t in terms)]))
    return values


def _assert_domain(param_name, dist, shifts, ranges) -> None:
    """Reject shifts that leave the valid width domain anywhere over the
    declared ranges of the quantities they depend on.

    SNANA fatals on invalid active widths (checkVal_GENGAUSS,
    sntools_genGauss_asym.c:607-639; getRan_skewGauss aborts on a non-positive
    effective skew sigma, sntools.c:7513). The traced samplers carry no
    numerical floors, so this domain guarantee is what keeps every selected
    lane valid. A width whose base is positive must stay positive, the grid
    and branch decisions being made from the base sign class.
    """
    field_map, fields = component_names(param_name)
    values = _shift_values(field_map, fields, shifts, ranges)

    def check(fieldname, base, strict):
        vals = base + values[fieldname]
        if np.any(vals < 0.0) or (strict and np.any(vals <= 0.0)):
            raise SNANAError(
                f"A shift takes {fieldname} of {param_name} out of its valid "
                f"domain over the range of the quantity it depends on",
                function_name="_assert_domain")

    if isinstance(dist, GenExpHalfGauss):
        # abort as getRan_GEN_EXP_HALFGAUSS (sntools_genExpHalfGauss.c:235-241)
        eps = 1.0e-14
        if np.any((dist.exp_tau + values['exp_tau'] < eps)
                  & (dist.sigma + values['sigma'] < eps)):
            raise SNANAError(
                f"A shift takes both EXP_TAU and SIGMA of {param_name} to "
                f"zero over the range of the quantity it depends on",
                function_name="_assert_domain")
        if np.any(dist.ratio + values['ratio'] < 0.0):
            raise SNANAError(
                f"A shift takes the RATIO of {param_name} negative over the "
                f"range of the quantity it depends on",
                function_name="_assert_domain")
        return

    check('sigma_lo', dist.sigma[0], strict=dist.sigma[0] > 0.0)
    check('sigma_hi', dist.sigma[1], strict=dist.sigma[1] > 0.0)
    check('sigma2_lo', dist.sigma2[0], strict=dist.sigma2[0] > 0.0)
    check('sigma2_hi', dist.sigma2[1], strict=dist.sigma2[1] > 0.0)

    if dist.skew == (0.0, 0.0) and not (values['skew_lo'].any()
                                        or values['skew_hi'].any()):
        return
    lo, hi = dist.range
    span = max(abs(lo - dist.peak), abs(hi - dist.peak))
    for side, base_sig, base_skew in (('lo', dist.sigma[0], dist.skew[0]),
                                      ('hi', dist.sigma[1], dist.skew[1])):
        sig = base_sig + values[f'sigma_{side}']
        skew = base_skew + values[f'skew_{side}']
        if np.any(sig + np.minimum(skew, 0.0) * span <= 0.0):
            raise SNANAError(
                f"A shift takes the effective skew sigma_{side} of "
                f"{param_name} non-positive within the sampling range",
                function_name="_assert_domain")


def distribution_from_settings(settings, *, dust=False):
    """Build a distribution using the same records and sampling kernels as SNANA inputs."""
    if isinstance(settings, (GenGaussAsym, GenExpHalfGauss)):
        expected = GenExpHalfGauss if dust else GenGaussAsym
        if not isinstance(settings, expected):
            raise SNANAError(f'This population needs {expected.__name__}')
        settings = asdict(settings)
    if not isinstance(settings, Mapping):
        raise SNANAError('A population distribution must be a mapping')
    values = dict(settings)
    kind = values.pop('distribution', 'exponential_gaussian' if dust else 'asymmetric_normal')
    if 'range' not in values:
        raise SNANAError('A population distribution needs an explicit range')
    if kind == 'uniform':
        if set(values) != {'range'}:
            raise SNANAError('A uniform population accepts only range')
        bounds = convert(values['range'], Tuple[float, float], 'range')
        values = asdict(flat_distribution(bounds, dust=dust))
    elif kind == 'normal':
        if dust:
            raise SNANAError('Use exponential_gaussian for a host-dust population')
        sigma = values.get('sigma')
        if not isinstance(sigma, (int, float)):
            raise SNANAError('A normal population needs one sigma')
        values['sigma'] = (sigma, sigma)
    elif kind == 'exponential':
        if not dust:
            raise SNANAError('Exponential populations are supported for host dust')
        if 'exp_tau' not in values:
            raise SNANAError('An exponential population needs exp_tau')
    elif kind not in ('asymmetric_normal', 'exponential_gaussian'):
        raise SNANAError(f'Unknown population distribution {kind!r}')
    if kind == 'exponential_gaussian' and not dust:
        raise SNANAError('exponential_gaussian is a host-dust distribution')
    result = build_dataclass(GenExpHalfGauss if dust else GenGaussAsym, values)
    if not dust:
        validate_gaussian(result)
        return result
    if not np.isfinite(result.range).all() or result.range[0] > result.range[1]:
        raise SNANAError(f'Invalid population range {result.range}')
    widths = [result.exp_tau, result.sigma]
    if any(not np.isfinite(v) or v < 0 for v in widths):
        raise SNANAError('Population widths must be finite and non-negative')
    return result


def declared_population(model_cls, structured):
    """A model's population parameter names, the other input spellings of
    those names, and SNANA's values for the keys an input leaves out. A model
    that reads its parameters from its own input file (SIMSED, Redback)
    returns them from ``structured_input``; any other declares them on its
    class."""
    return (tuple(structured.get('population_parameters',
                                 model_cls.population_parameters)),
            structured.get('population_aliases', model_cls.population_aliases),
            structured.get('population_defaults', model_cls.population_defaults))


def population_spellings(names, aliases):
    """Every spelling of each population parameter, lower-cased, mapped to
    its name, so the input may write a name in any case. A spelling claimed by
    two parameters, or by a parameter and a host-dust population, is
    refused."""
    dust = {spelling.lower(): p.key for p in HOST_DUST_PARAMETERS
            for spelling in (p.key, *p.input_names)}
    spellings = {}
    for name in names:
        for spelling in (name, *aliases.get(name, ())):
            owner = spellings.get(spelling.lower(), dust.get(spelling.lower()))
            if owner is not None and owner != name:
                raise SNANAError(
                    f"'{spelling}' names both the population parameter "
                    f"{name} and {owner}. Names match whatever their case, "
                    f"so each spelling must belong to one parameter.",
                    function_name="population_spellings")
            spellings[spelling.lower()] = name
    return spellings


def model_spellings(model):
    """:func:`population_spellings` for a built model's own parameters."""
    return population_spellings([name for name, _ in model.configured_population],
                                dict(model.configured_population))


def stream_name(name, aliases):
    """The name a parameter's random stream is keyed by: its first alias,
    which is SNANA's spelling, else its name. Keying by SNANA's spelling keeps
    the draws the same whatever the model calls the parameter."""
    spellings = aliases.get(name)
    return spellings[0] if spellings else name


def configured_distributions(settings, model, structured):
    """The distributions given as settings, by population parameter name,
    for the selected model and the host dust."""
    names, aliases, _ = declared_population(model, structured)
    spellings = population_spellings(names, aliases)
    dust = {p.key: p.exp_halfgauss for p in HOST_DUST_PARAMETERS}
    result = {}
    for given, value in settings.items():
        name = given if given in dust else spellings.get(given.lower())
        if name is None:
            raise SNANAError(f'Unknown population parameter {given!r}; '
                             f'choose from {sorted(names) + sorted(dust)}')
        result[name] = distribution_from_settings(value, dust=dust.get(name, False))
    return result


def validate_config_populations(config, sed_model, hostlib) -> None:
    """Check every parameter's record once the ranges of the quantities the
    populations depend on are known.

    Such a quantity is the redshift, another drawn parameter, or a HOSTLIB
    column, and only the last of those needs the catalogue, which is why
    this waits for ``hostlib`` (None without one) to load. The ranges are
    the run's, so a run is accepted or rejected on its configuration rather
    than on the events a batch draws. A model parameter a shift depends on
    must be drawn before the one it shifts, as the draws go in the model's
    declaration order.
    """
    populations = config.model.populations
    drawn = sed_model.drawn_parameters(populations)
    declared = {name for name, _ in sed_model.configured_population}
    for i, name in enumerate(drawn):
        later = [quantity for quantity in populations[name].depends_on()
                 if quantity in declared and quantity not in drawn[:i]]
        if later:
            raise SNANAError(
                f"A shift on {name} depends on {later[0]}, which is not drawn "
                f"before it. The model draws its population parameters in "
                f"the order {', '.join(drawn)}.",
                function_name="validate_config_populations")
    ranges = {REDSHIFT: config.rates.redshift_range}
    for name in set(config.model.quantities_depended_on()):
        if name in ranges:
            continue
        span = config.model.parameter_range(name)
        if span is None:
            columns = [] if hostlib is None else hostlib.selector.varnames
            if name not in columns:
                raise SNANAError(
                    f"A shift depends on '{name}', which is neither a "
                    f"population parameter nor a column this HOSTLIB carries. "
                    f"Columns: {sorted(columns)}.",
                    function_name="validate_config_populations")
            column = np.asarray(
                hostlib.catalogue.columns[columns.index(name)])
            span = (float(column.min()), float(column.max()))
        ranges[name] = span
    for param_name, pop in config.model.populations.items():
        validate(param_name, pop, ranges)
