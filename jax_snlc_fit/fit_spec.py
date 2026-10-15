"""What the fitter derives from a model's fit-parameter spec.

The model declares its parameters once (:meth:`FittedModel.get_fit_parameters`);
every view of that declaration the fitter needs is a function of it here,
so no model answers a question whose answer is already in its spec.
"""

import jax.numpy as jnp
import numpy as np

from jax_snlc_core.utils import SNANAError


def validate_fit_parameters(model, parameters):
    """Check the model's declaration before building any fitting kernels."""
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
