"""The fit parameters: what the fitter derives from a model's parameter
spec, the priors on them, and the input keys that name them.

The model declares its parameters once (:meth:`FittedModel.get_fit_parameters`);
every view of that declaration the fitter needs is a function of it here,
so no model answers a question whose answer is already in its spec.

SNANA gives each fitted parameter at most one prior (``USE_PRIOR(ipar)``),
builds it once into a tabulated (value, chi2) grid (``FITINI_PKMJDPRIOR``,
``FITINI_AVPRIOR``, ``FITINI_FLATPRIOR``) and evaluates any of them through
one lookup, ``CHI2_PRIOR(ipar, val)`` (snlc_fit.F90:11280-12500). Here a
model declares the form and coefficients on the parameter's fit spec,
:func:`active` resolves the declarations against the input, and
:func:`prior_chi2` sums them.

Two coefficients are not known until the batch exists and are named rather
than given: :data:`T0_CENTRE`, the SN's initial peak estimate, and
:data:`MU_REF`, its reference distance modulus. Both arrive as arguments to
:func:`prior_chi2`, with each SN's reference MJD for a prior given in
absolute MJD (:func:`per_sn`), so nothing carries a per-event array that a
second place has to index in step. A user's prior is checked and put in
this numerical form by :func:`parse_prior`.

Input keys whose suffix names a parameter (INIVAL_, INISTP_, PRIOR_, FITWIN_,
USESIM_) are resolved against the model's names here as well
(:func:`resolve_parameter_keys`, :func:`resolve_fitwin_windows`,
:func:`resolve_usesim_columns`).
"""

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import log_ndtr

from jax_snlc_core.logging import SNANAError
from jax_snlc_core.models.roles import MU_REF, T0_CENTRE

# DOFIT_PHOTOZ redshift, model-independent (snlc_fit.F90:2449)
PHOTOZ_PARAMETER = {'zPHOT': {'error_bounds': (1.0e-4, 2.0)}}

# zPHOT ERRMAX_BAD, plain and with a best-z prior (snlc_fit.F90:1125, :1128, :7128)
PHOTOZ_ERRMAX_BAD = 0.0005
PHOTOZ_ERRMAX_BAD_BESTZ = 1.0e-8
PHOTOZ_INISTP_DEFAULT = 0.05


def validate_fit_parameters(model, parameters):
    """Check the model's declaration before compiling any fit function."""
    model_name = type(model).__name__
    if next(iter(parameters), None) != 'PKMJD':
        raise SNANAError(f"{model_name}: fit parameters must start with PKMJD",
                         function_name="validate_fit_parameters")
    if model.distance_param is not None and model.distance_param not in parameters:
        raise SNANAError(f"{model_name}: distance parameter {model.distance_param!r} is not declared",
                         function_name="validate_fit_parameters")
    bound_keys = ('bounds', 'bounds_relative_to_init', 'bounds_about_peak_estimate')
    for name, spec in parameters.items():
        bounds = [spec[key] for key in bound_keys if key in spec]
        scale = spec.get('scale', 1.)
        problem = None
        if name.upper() == 'ZPHOT':
            problem = "zPHOT is supplied by the fitter"
        elif len(bounds) != 1:
            problem = f"declare exactly one of {', '.join(bound_keys)}"
        elif np.shape(bounds[0]) != (2,) or not bounds[0][0] < bounds[0][1]:
            problem = "bounds must be a (lower, upper) pair with lower < upper"
        elif not np.isfinite(scale) or scale <= 0:
            problem = "scale must be finite and positive"
        elif name not in ('PKMJD', model.distance_param) and 'init' not in spec:
            problem = "init is required"
        if problem:
            raise SNANAError(f"{model_name} parameter {name!r}: {problem}",
                             function_name="validate_fit_parameters")


def init_params(model, n_sne, t0_init, mu_ref):
    """The fit's starting values: t0 from the light curve, the distance
    parameter at the reference distance with every other parameter at zero
    (SNANA's SALT2x0calc(S2a, S2b, 0, 0, MU), snlc_fit.F90:8166), the rest
    from each parameter's ``init``."""
    spec = model.get_fit_parameters()
    init = {name: jnp.full(n_sne, s['init'])
            for name, s in spec.items() if 'init' in s}
    init['PKMJD'] = t0_init
    if model.distance_param is not None:
        at_distance = model.params_at_distance(
            {name: 0.0 for name in spec}, mu_ref)
        init[model.distance_param] = at_distance[model.distance_param]
    return init


# the fit's starting distance where the redshift is undefined
# (snlc_fit.F90:8156), and the redshift below which it is (snana.F90:17241)
MU_START_NO_REDSHIFT = 35.5555
Z_MIN_DLMAG = 1.0e-5


def start_distance_modulus(config, redshifts):
    """The distance modulus a fit starts from: the reference one where
    the redshift is defined, SNANA's fixed value where it is not."""
    defined = np.asarray(redshifts) > Z_MIN_DLMAG
    return np.where(defined, config.fit.distance_modulus(
        np.where(defined, redshifts, 1.0)), MU_START_NO_REDSHIFT)


def adaptive_param_scales(model, mu_ref):
    """Per-SN scales for the parameters declaring ``scale_at_distance``: the
    amplitude of a zero-shape, zero-colour source at the reference distance,
    for a parameter that spans orders of magnitude with distance."""
    spec = model.get_fit_parameters()
    names = [n for n, s in spec.items() if s.get('scale_at_distance')]
    if not names:
        return {}
    at_distance = model.params_at_distance({name: 0.0 for name in spec}, mu_ref)
    return {n: at_distance[n] for n in names}


def error_surface_overrides(fit_parameters, ref_params=None, first_round=False):
    """Parameters the model error is evaluated at where those differ from the
    trial values (SNANA SALT2xx1, snlc_fit.F90:13792): a parameter declaring
    ``error_surface_round1`` is held there on the first round and at the
    round-start value (``ref_params``, the model's vector) where the caller
    freezes the surface."""
    held = {n: s['error_surface_round1'] for n, s in fit_parameters.items()
            if 'error_surface_round1' in s}
    if first_round:
        return held
    if ref_params is not None:
        names = list(fit_parameters)
        return {n: ref_params[names.index(n)] for n in held}
    return {}


def covariance_output_params(fit_parameters):
    """The parameters whose covariances the FITRES carries: every fitted one
    not declared ``deferred`` (a nuisance parameter nobody reports) or
    ``covariance`` False."""
    return [n for n, s in fit_parameters.items()
            if s.get('covariance', True) and not s.get('deferred')]


def snana_parameter_aliases(model):
    """SNANA's generic spellings (INIVAL_SHAPE, PRIOR_COLOR_RANGE ...) as
    this model's parameter names, from each parameter's ``snana_keys``.

    Backwards compatibility with SNANA's slot-shaped namelist only: the
    parameter's own name is the interface, and a model declaring none is
    addressed by name alone.
    """
    return {alias: name
            for name, spec in model.get_fit_parameters().items()
            for alias in spec.get('snana_keys', ())}


def ndof_n_params(model, held=()):
    """Free parameters for NDOF = n_obs - n_params.

    ``held`` names parameters pinned for this fit, which consume no degrees
    of freedom (SNANA subtracts NFIXPAR, snlc_fit.F90:13102). A parameter
    declaring ``'ndof': False`` consumes none either, being constrained by a
    prior rather than by the data.
    """
    return sum(1 for name, spec in model.get_fit_parameters().items()
               if spec.get('ndof', True) and name not in held)


_SOURCES = (T0_CENTRE, MU_REF)


def _uniform(value, c):
    return jnp.where((value >= c['lower']) & (value <= c['upper']), 0., jnp.inf)


def support(form, coefficients):
    """Hard support of a resolved prior, or None when it has no hard edge."""
    if form in ('uniform', 'loguniform'):
        return coefficients['lower'], coefficients['upper']
    if form == 'exponential':
        return 0., np.inf
    return None


def _loguniform(value, c):
    return _uniform(value, c) + 2. * jnp.log(jnp.maximum(value, c['lower']) / c['lower'])


def _exponential(value, c):
    return jnp.where(value >= 0, 2. * value / c['scale'], jnp.inf)


def _gauss(value, c):
    """chi2 = ((v - centre)/sigma)^2."""
    return ((value - c['centre']) / c['sigma']) ** 2


def _gauss_flare(value, c):
    """The PEAKMJD prior (FITINI_PKMJDPRIOR, snlc_fit.F90:11824).

    Gaussian in the bulk, multiplied by exp(excess^2) beyond ``flare`` so the
    slope outside is much steeper. SNANA tabulates this over ``grid`` and
    CHI2_PRIOR clamps into the grid's ends (:12489), so the penalty stops
    growing once the offset leaves it; clipping the offset is the same thing.
    """
    dt = jnp.clip(value - c['centre'], c['grid'][0], c['grid'][1])
    excess = jnp.maximum(jnp.abs(dt) - c['flare'], 0.0)
    return (dt / c['sigma']) ** 2 * jnp.exp(excess ** 2)


def _flat_rolloff(value, c):
    """Flat inside [lo, hi], rolling off outside (FITINI_FLATPRIOR).

    The roll-off is the fourth power of the offset in sigma: SNANA forms
    CHI2 = (dif/sigma)**2 and stores CHI2*CHI2 (snlc_fit.F90:12145) to
    suppress the boundaries harder than a Gaussian would.
    """
    excess = (jnp.maximum(c['lo'] - value, 0.0)
              + jnp.maximum(value - c['hi'], 0.0))
    return (excess / c['sigma']) ** 4


def _protect_max(value, c):
    """chi2 = exp(0.5*(|v| - max)^2) - 1 beyond ``max``, else zero.

    SNANA's guard against crazy excursions, applied to the colour-like
    parameter with a per-model bound (snlc_fit.F90:4868-4872).
    """
    excess = jnp.maximum(jnp.abs(value) - c['max'], 0.0)
    return jnp.exp(0.5 * excess ** 2) - 1.0


def _exp_convolved(value, c):
    """P(v) = sum_k w_k exp(-v/tau_k) convolved with G(0, sigma) over v >= 0
    (FITINI_AVPRIOR), offset so its minimum chi2 is zero.

    log_ndtr gives log(Phi) stably and differentiably, which SNANA's tabulated
    grid would not. sigma = 0 is the unsmeared exponential, whose closed form
    the convolution cannot be evaluated at (it divides by sigma) and reduces
    to; a model with no smearing declares it that way rather than as a form of
    its own.
    """
    sigma = c['sigma']
    live = [(tau, wgt) for tau, wgt in zip(c['tau'], c['wgt']) if wgt > 0]
    if sigma == 0.0:
        terms = [-value / tau + float(np.log(wgt)) for tau, wgt in live]
    else:
        terms = [(-value / tau + sigma ** 2 / (2 * tau ** 2)
                  + log_ndtr((value - sigma ** 2 / tau) / sigma)
                  + float(np.log(wgt))) for tau, wgt in live]
    if len(terms) == 1:
        log_p = terms[0]
    else:
        stacked = jnp.stack(terms)
        mx = jnp.max(stacked, axis=0)
        log_p = mx + jnp.log(jnp.sum(jnp.exp(stacked - mx), axis=0))
    return -2.0 * log_p - c['offset']


FORMS = {
    'gauss': _gauss,
    'gauss_flare': _gauss_flare,
    'flat_rolloff': _flat_rolloff,
    'protect_max': _protect_max,
    'exp_convolved': _exp_convolved,
    'uniform': _uniform,
    'loguniform': _loguniform,
    'exponential': _exponential,
}


def parse_prior(value):
    """Validate one complete user prior and return its numerical form."""
    if not isinstance(value, dict):
        raise SNANAError('A prior must be a mapping with a distribution and its coefficients')
    form = value.get('distribution')
    required = {
        'normal': ('mean', 'sigma'), 'uniform': ('lower', 'upper'),
        'loguniform': ('lower', 'upper'), 'exponential': ('scale',),
        'flat_rolloff': ('lower', 'upper', 'sigma'),
    }
    if form not in required:
        raise SNANAError(f"Unknown prior {form!r}; choose from {tuple(required)}")
    supplied = set(value) - {'distribution'}
    if supplied != set(required[form]):
        raise SNANAError(f"{form} prior requires {', '.join(required[form])}")
    try:
        c = {key: float(value[key]) for key in required[form]}
    except (TypeError, ValueError) as error:
        raise SNANAError('Prior coefficients must be numbers') from error
    if not all(np.isfinite(v) for v in c.values()):
        raise SNANAError('Prior coefficients must be finite')
    if any(c.get(key, 1.) <= 0 for key in ('sigma', 'scale')):
        raise SNANAError('Prior widths and scales must be positive')
    if 'lower' in c and c['lower'] >= c['upper']:
        raise SNANAError('Prior lower bound must be below its upper bound')
    if form == 'loguniform' and c['lower'] <= 0:
        raise SNANAError('A loguniform prior needs a positive lower bound')
    if form == 'normal':
        return 'gauss', {'centre': c['mean'], 'sigma': c['sigma']}
    if form == 'flat_rolloff':
        return form, {'lo': c['lower'], 'hi': c['upper'], 'sigma': c['sigma']}
    return form, c


def exp_convolved_offset(tau, wgt, sigma):
    """The chi2 offset putting :func:`_exp_convolved`'s minimum at zero.

    Computed once when the prior is assembled, over the range the parameter
    can reach, so the evaluated chi2 is comparable with the other forms. The
    unsmeared exponential is already zero at v = 0 and needs none.
    """
    if sigma == 0.0:
        return 0.0
    chi2 = _exp_convolved(jnp.linspace(0.0, 0.5, 10000),
                          {'tau': tau, 'wgt': wgt, 'sigma': sigma, 'offset': 0.0})
    return float(jnp.min(chi2))


def active(model, config, param_names):
    """``((form, index, coefficients), ...)`` for the priors this fit applies.

    A parameter's fit spec declares its priors as ``(form, coefficients)``
    pairs, and the input's ``PRIOR_`` keys supply coefficients for the parameter
    they name: overriding a declaration, or adding a prior to a parameter
    that declares none. Indices rather than names so the fit loop indexes the
    parameter vector directly.

    OPT_PRIOR = 0 disables every prior (snlc_fit.F90:4891), which is an empty
    tuple rather than a flag the evaluator has to check.
    """
    if config.model.opt_prior <= 0:
        if any(isinstance(value, dict) and 'distribution' in value
               for value in config.model.priors.values()):
            raise SNANAError('Explicit priors conflict with opt_prior=0')
        return ()
    if 'zPHOT' in config.model.priors:
        raise SNANAError('Use the photo-z settings and metadata to configure the redshift prior')
    assembled = []
    for name, spec in model.get_fit_parameters().items():
        for form, coefficients in _for_parameter(config, name, spec):
            if form not in FORMS:
                raise SNANAError(
                    f"{type(model).__name__} declares prior form '{form}' for "
                    f"{name}, which is not one of {', '.join(sorted(FORMS))}.",
                    function_name="active")
            assembled.append(
                (form, param_names.index(name), coefficients))
    return tuple(assembled)


def _for_parameter(config, name, spec):
    """One parameter's priors, the model's declarations carrying the defaults
    and the input the last word."""
    declared = {form: dict(c) for form, c in spec.get('priors', ())}
    if name in config.model.priors:
        supplied = config.model.priors[name]
        if supplied is None or 'distribution' in supplied:
            declared = {form: c for form, c in declared.items() if form == 'protect_max'}
            if supplied is not None:
                form, coefficients = parse_prior(supplied)
                if name == 'PKMJD':
                    coefficients['public_time'] = True
                declared[form] = coefficients
            return declared.items()
    for form, supplied in config.model.priors.get(name, {}).items():
        declared.setdefault(form, {}).update(supplied)
    for form, c in declared.items():
        if form == 'flat_rolloff' and c.get('sigma') is None:
            raise SNANAError(
                f"PRIOR_{name}_RANGE is set but {name} has no roll-off width: "
                f"this model declares no flat prior for it, so give "
                f"PRIOR_{name}_SIGMA too.", function_name="PRIOR_RANGE")
        if form == 'exp_convolved':
            c['offset'] = exp_convolved_offset(c['tau'], c['wgt'], c['sigma'])
    return declared.items()


def flat_window(config, name, spec):
    """``(lo, hi, sigma)`` of one parameter's flat prior, or None where neither
    the model nor the input gives it one. PEG_MARGIN is measured against this
    window, so it resolves the prior exactly as the fit does.
    """
    if config.model.opt_prior <= 0:
        return None
    c = dict(_for_parameter(config, name, spec)).get('flat_rolloff')
    return None if c is None else (c['lo'], c['hi'], c['sigma'])


def per_sn(t0_centres, mu_ref, reference_mjds=0.):
    """The per-SN coefficients as an ``(n_sne, 3)`` array, for the fit to carry
    into its jitted chi2 alongside the parameter vector.

    Packed and read only here and in :func:`prior_chi2`, so the two cannot
    fall out of step, and the contents are the same for every model.
    """
    return jnp.stack([jnp.asarray(t0_centres),
                      jnp.broadcast_to(jnp.asarray(mu_ref),
                                       jnp.shape(t0_centres)),
                      jnp.broadcast_to(jnp.asarray(reference_mjds),
                                       jnp.shape(t0_centres))], axis=1)


def prior_chi2(params_arr, priors, per_sn_row):
    """Summed prior chi2 for one SN, under the fit's vmap over events.

    ``priors`` is static, so the loop and the coefficient lookup resolve at
    trace time; ``per_sn_row`` is that SN's row of :func:`per_sn`.
    """
    supplied = {T0_CENTRE: per_sn_row[0], MU_REF: per_sn_row[1]}
    total = 0.0
    for form, index, coefficients in priors:
        resolved = {k: supplied[v] if isinstance(v, str) and v in _SOURCES
                    else v for k, v in coefficients.items()}
        value = params_arr[index]
        if coefficients.get('public_time', False):
            value = value + per_sn_row[2]
        total = total + FORMS[form](value, resolved)
    return total


# The spellings every model shares; model-specific ones are declared by the
# parameter that carries them (parameters.snana_parameter_aliases).
_UNIVERSAL_PARAM_SPELLINGS = {
    'PEAKMJD': 'PKMJD',
    'DPEAKMJD': 'PKMJD',           # grid axis is t0 offsets
    'PHOTOZ': 'zPHOT',
}


_UNSUPPORTED_PARAM_KEYS = {
    'AVRV': 'selects fitting the AV/RV ratio in place of RV, which is not supported',
    'SHAPE2': 'selects a second light-curve shape parameter, which is not supported',
}


def resolve_parameter_key(suffix: str, param_names, aliases) -> str:
    """Map an ``INIVAL_``/``INISTP_``/``PRIOR_`` suffix onto one of the
    model's parameters.

    The parameter's own name wins, so a model is always addressable by name.
    ``aliases`` is :func:`~jax_snlc_fit.parameters.snana_parameter_aliases` of
    the model, which lets an input written in SNANA's slot spellings still
    resolve.
    """
    if suffix.upper() in _UNSUPPORTED_PARAM_KEYS:
        raise SNANAError(
            f"{suffix} {_UNSUPPORTED_PARAM_KEYS[suffix.upper()]}.",
            function_name="resolve_parameter_key")

    by_lower = {n.lower(): n for n in param_names}
    if suffix.lower() in by_lower:
        return by_lower[suffix.lower()]

    for snana in (aliases.get(suffix.upper()),
                   _UNIVERSAL_PARAM_SPELLINGS.get(suffix.upper())):
        if snana is not None and snana.lower() in by_lower:
            return by_lower[snana.lower()]

    raise SNANAError(
        f"{suffix} is not a parameter of this model. It fits: "
        f"{', '.join(param_names)}.",
        function_name="resolve_parameter_key")


def resolve_fitwin_windows(config, model, param_names, aliases) -> dict:
    """``FITWIN_`` suffixes -> the FITRES column each cuts on, checked here so a
    misspelt suffix aborts before any fit runs.

    ``SHAPE_ERR`` becomes ``x1_ERR`` under SALT: the base name resolves the
    same way ``INIVAL_``/``INISTP_`` do, then to the column the model writes that
    parameter as (BayeSN's Ds is MU_LCFIT), and the _ERR / _ERRz1 suffix rides
    along. A base naming a column directly (SALT's mB, which no parameter
    owns) is taken as it is. ``PROB``, ``ZPULL`` and ``COV_OFFDIAG`` name
    derived quantities rather than columns and pass through unchanged.
    """
    derived = {'PROB', 'ZPULL', 'COV_OFFDIAG'}
    pairs = model.fitres_param_columns(param_names)
    by_lower = {column.lower(): column for _, column in pairs}

    def column_of(name):
        """The parameter's own column where it has one, else the column it
        is written as (SALT's amplitude owns x0 and the derived mB)."""
        columns = [column for owner, column in pairs if owner == name]
        return name if name in columns else columns[0]

    resolved = {}
    for suffix, window in config.selection.fitwin.items():
        upper = suffix.upper()
        if upper in derived:
            resolved[upper] = window
            continue
        base, _, form = suffix.partition('_')
        if form and form.upper() not in ('ERR', 'ERRZ1'):
            base, form = suffix, ''
        try:
            column = column_of(resolve_parameter_key(base, param_names, aliases))
        except SNANAError:
            if base.lower() not in by_lower:
                raise SNANAError(
                    f"FITWIN_{suffix} names no fitted quantity. Available: "
                    f"{', '.join([*by_lower.values(), *sorted(derived)])}, "
                    f"each with _ERR and _ERRz1.",
                    function_name="FITWIN") from None
            column = by_lower[base.lower()]
        resolved[f"{column}_{form.upper()}" if form else column] = window
    return resolved


def resolve_usesim_columns(config, fit_parameters, param_names, aliases):
    """(parameter name -> ``SIM_`` truth column, pinned names) for the
    ``USESIM_*`` keys in effect. USESIM_INIVAL seeds without pinning, so it
    contributes columns but no pinned names."""
    columns = {}
    pinned = []
    for generic in config.model.usesim_params:
        name = resolve_parameter_key(generic, param_names, aliases)
        spec = fit_parameters.get(name)
        if spec is None or 'sim_truth' not in spec:
            raise SNANAError(
                f"USESIM_{generic} resolves to '{name}', which declares no "
                f"simulated-truth column, so there is nothing to fit against.",
                function_name="resolve_usesim_columns")
        columns[name] = spec['sim_truth']
        pinned.append(name)

    if config.model.usesim_inival:
        for name, spec in fit_parameters.items():
            if 'sim_truth' in spec:
                columns.setdefault(name, spec['sim_truth'])

    return columns, pinned


def resolve_parameter_keys(model_config, param_names, aliases) -> dict:
    """The Model record's per-parameter dicts (INIVAL_, INISTP_, PRIOR_, and
    the initial, fixed, bounds, grid-search and coarse-grid ones) with their
    keys resolved to parameter names, so no generic spelling survives past the
    layer that parses them. Returns {attr: resolved dict}; the input is
    untouched."""
    resolved = {}
    for attr in ('inival', 'initial', 'fixed', 'bounds', 'inistp', 'inival_gridsearch', 'priors',
                 'coarse_grid'):
        raw = getattr(model_config, attr)
        resolved[attr] = {resolve_parameter_key(s, param_names, aliases): v
                          for s, v in raw.items()} if raw else raw
    return resolved


def validate_inival_bounds(config, fit_parameters) -> None:
    """Abort on a starting value outside its parameter's declared bounds."""
    for name, value in config.model.inival.items():
        spec = fit_parameters.get(name)
        # only absolute bounds can be checked here
        if not spec or 'bounds' not in spec:
            continue
        lo, hi = spec['bounds']
        if not lo <= value <= hi:
            raise SNANAError(
                f"INIVAL for {name} is {value}, outside its bounds "
                f"({lo}, {hi}).",
                function_name="validate_inival_bounds")


def fit_parameters(config, model):
    """The fit vector's parameters and what each declares: ``model``'s, then
    under photo-z zPHOT, bounded by PHOTOZ_BOUND."""
    parameters = dict(model.get_fit_parameters())
    if config.photoz.dofit_photoz:
        parameters['zPHOT'] = {
            **PHOTOZ_PARAMETER['zPHOT'],
            'bounds': config.photoz.photoz_bound,
            'inistp': PHOTOZ_INISTP_DEFAULT,
            # a floor like any other floated parameter's (snlc_fit.F90:1127)
            'errmax_bad': (PHOTOZ_ERRMAX_BAD_BESTZ
                           if config.photoz.opt_photoz == 2
                           else PHOTOZ_ERRMAX_BAD)}
    return parameters
