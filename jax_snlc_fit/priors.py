"""Fit priors: declared on the parameter, evaluated in one place.

SNANA gives each fitted parameter at most one prior (``USE_PRIOR(ipar)``),
builds it once into a tabulated (value, chi2) grid (``FITINI_PKMJDPRIOR``,
``FITINI_AVPRIOR``, ``FITINI_FLATPRIOR``) and evaluates any of them through
one lookup, ``CHI2_PRIOR(ipar, val)`` (snlc_fit.F90:11280-12500). This module
is that structure: a model declares the form and coefficients on the
parameter's fit spec, :func:`active` resolves the declarations against the
input, and :func:`prior_chi2` sums them.

Two coefficients are not known until the batch exists and are named rather
than given: :data:`T0_CENTRE`, the SN's initial peak estimate, and
:data:`MU_REF`, its reference distance modulus. Both arrive as arguments to
:func:`prior_chi2`, so nothing carries a per-event array that a second place
has to index in step.
"""

import jax.numpy as jnp
import yaml
from jax.scipy.special import log_ndtr
import numpy as np

from jax_snlc_core.models.fitted import MU_REF, T0_CENTRE
from jax_snlc_core.utils.error_handling import SNANAError

_SOURCES = (T0_CENTRE, MU_REF)


def read_prior_tokens(tokens):
    """Read --prior PARAM DISTRIBUTION coefficient=value ... ."""
    if len(tokens) == 1:
        values = yaml.safe_load(tokens[0])
        if isinstance(values, dict):
            for prior in values.values():
                if prior is not None:
                    parse_prior(prior)
            return values
    if len(tokens) < 2:
        raise SNANAError('--prior needs a parameter and a distribution')
    name, form, *coefficients = tokens
    if form.lower() in ('none', 'null'):
        if coefficients:
            raise SNANAError('Removing a prior takes no coefficients')
        return {name: None}
    values = {'distribution': form}
    for coefficient in coefficients:
        key, sep, value = coefficient.partition('=')
        if not sep or key in values:
            raise SNANAError(f'Invalid prior coefficient {coefficient!r}')
        values[key] = yaml.safe_load(value)
    parse_prior(values)
    return {name: values}


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
                    function_name="active_priors")
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
    for form, supplied in _from_input(config, name).items():
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


def _from_input(config, name):
    """``{form: coefficients}`` the input supplies for one parameter."""
    return config.model.priors.get(name, {})


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
    """The per-SN coefficients as an ``(n_sne, 2)`` array, for the fit to carry
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
