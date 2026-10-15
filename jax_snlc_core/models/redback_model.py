"""redback-jax bolometric models as a GENMODEL family.

``GENMODEL: REDBACK.<registry_name>`` (e.g. REDBACK.arnett_bolometric) runs a
redback-jax bolometric model through the full simulation machinery: the
bolometric light curve is evaluated per SN on a fixed time grid, and
``generate_sed`` turns log10(L_bol) into a rest-frame blackbody SED via
redback's photosphere. Physics parameters are sampled from GENPEAK_<name>
keys matching the bolometric function's signature.

redback-jax is an optional dependency, imported only when this model is used.
"""

import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .base import SNModel
from .fitted import FittedModel
from .simulated import SimulatedModel
from jax_snlc_core.physics.constants import BOLTZMANN_ERG_K, LIGHT_A, LIGHT_KM, PLANCK
from jax_snlc_core.utils.error_handling import SNANAError
from jax_snlc_core.utils.interp import (compute_catmullrom_slopes,
                                       hermite_interp_1d)

# blue edge: dust-law validity (~912 A)
WAVE_RANGE = (1000.0, 60000.0)
DL10PC_CM = 3.0856775814913673e19
C_CM = LIGHT_A * 1.0e-8  # cm/s

# {name: (low, high, init, scale)}
PARAM_FIT_TABLE = {
    'f_nickel':          (0.001,  1.0,     0.1,    0.1),
    'mej':               (0.1,    30.0,    2.0,    1.0),
    'vej':               (1000.0, 30000.0, 9000.0, 1000.0),
    'kappa':             (0.01,   0.4,     0.1,    0.1),
    'kappa_gamma':       (0.001,  1.0,     0.03,   0.03),
    'temperature_floor': (1000.0, 15000.0, 5000.0, 1000.0),
    'p0':                (0.7,    50.0,    5.0,    1.0),
    'bp':                (0.01,   100.0,   1.0,    1.0),
    'mass_ns':           (1.0,    2.5,     1.4,    0.5),
    'theta_pb':          (0.0,    1.5708,  0.785,  0.5),
    'log10_mass':        (-3.0,   1.0,     -1.0,   1.0),
    'log10_radius':      (12.0,   15.0,    13.0,   1.0),
    'log10_energy':      (49.0,   52.0,    50.5,   1.0),
    'nn':                (6.0,    12.0,    10.0,   1.0),
    'delta':             (0.0,    2.0,     1.0,    1.0),
    'eta':               (0.1,    10.0,    1.0,    1.0),
    'tshock':            (0.1,    20.0,    3.0,    1.0),
    'shocked_fraction':  (0.01,   1.0,     0.5,    0.5),
    'cos_theta_cocoon':  (0.0,    1.0,     0.5,    0.5),
    # TDE fallback normalisation at one second (erg/s), not peak luminosity.
    'log10_l0':          (51.0,   60.0,    55.0,   1.0),
    't_0_turn':          (1.0,    600.0,   100.0,  100.0),
}

# Bolometric parameters retain their engine's units in populations, truth and
# fit vectors. Only the velocity passed to the shared photosphere is in km/s.
# The other supported engines (and external plugins by default) use km/s.
VELOCITY_UNITS = {
    'shocked_cocoon_bolometric': 'c',
    'metzger_kilonova_bolometric': 'c',
    'magnetar_boosted_kilonova_bolometric': 'c',
}

UNFITTABLE = {
    'magnetar_boosted_kilonova_bolometric':
        "its fit parameter bounds and fitting have not yet been validated",
    'metzger_kilonova_bolometric':
        "its fit parameter bounds and fitting have not yet been validated",
}


def _log10_blackbody_flux_density(temperature, log10_r_photosphere, dl, frequency):
    """log10 blackbody F_nu (erg/s/Hz/cm^2), matching the formula inside
    redback's spectra pipeline. Constants: SNANA's PLANCK/LIGHT_A
    (sntools.h:67-68) and the CODATA Boltzmann constant redback also uses.

    Evaluating in log space is mathematically neutral stabilisation: the
    linear Planck formula's frequency**3 overflows float32 at optical
    frequencies. Writing log(expm1(x)) as x + log(-expm1(-x)) also keeps
    the Wien tail and its derivatives finite.
    """
    x = (PLANCK / BOLTZMANN_ERG_K) * frequency / temperature
    return (jnp.log10(2.0 * jnp.pi * PLANCK)
            + 3.0 * jnp.log10(frequency)
            + 2.0 * log10_r_photosphere
            - 2.0 * jnp.log10(dl)
            - jnp.log10(C_CM ** 2)
            - (x + jnp.log(-jnp.expm1(-x))) / jnp.log(10.0))

# Observer-frame wrappers and models needing concrete (untraceable) setup.
UNSUPPORTED = {
    'csm_interaction_bolometric':
        "its CSM coefficient lookup needs concrete nn/eta values "
        "(scipy interpolation), incompatible with per-SN sampling",
    'arnett_with_features_cosmology':
        "it applies redshift and distance internally; use "
        "REDBACK.arnett_bolometric (the engine owns cosmology)",
}


def _redback():
    """redback-jax's model registry and photosphere, imported when a model
    is built rather than with this module: the package builds arrays at
    import, which must wait until the run's precision is set."""
    try:
        from redback_jax.models import MODEL_REGISTRY
        from redback_jax.photosphere import compute_temperature_floor_log10
    except ImportError as err:
        raise SNANAError(
            "GENMODEL REDBACK.* requires the optional dependency "
            "redback-jax. Install with: pip install 'jax-snana[redback]'",
            function_name="_redback") from err
    return MODEL_REGISTRY, compute_temperature_floor_log10


def _bolometric_function(model_path):
    """The redback-jax bolometric function ``REDBACK.<name>`` names, with
    its registry name."""
    registry, _ = _redback()
    name = model_path.split('.', 1)[1] if '.' in model_path else ''
    if not model_path.lower().startswith('redback') or not name:
        raise SNANAError(
            f"GENMODEL '{model_path}' is not of the form REDBACK.<model>",
            function_name="RedbackModel")

    if name in UNSUPPORTED:
        raise SNANAError(
            f"REDBACK.{name} is not supported: {UNSUPPORTED[name]}",
            function_name="RedbackModel")
    if name.endswith('_spectra'):
        raise SNANAError(
            f"REDBACK.{name} is an observer-frame spectra wrapper; use the "
            f"bolometric model REDBACK.{name[:-len('_spectra')]}_bolometric",
            function_name="RedbackModel")
    if name not in registry:
        available = sorted(k for k in registry
                           if k.endswith('_bolometric') and k not in UNSUPPORTED)
        raise SNANAError(
            f"Unknown redback model '{name}'. Available: {available}",
            function_name="RedbackModel")
    return name, registry[name]


def _signature(bolo_fn):
    """A bolometric function's parameters after its time argument: all of
    them in order, the required ones, the optional ones with their defaults,
    and the boolean switches, which are not sampleable physics."""
    signature = inspect.signature(bolo_fn)
    names = list(signature.parameters)[1:]
    required, optional, flags = [], {}, {}
    for pname in names:
        par = signature.parameters[pname]
        if par.default is inspect.Parameter.empty or par.default is None:
            required.append(pname)
        elif isinstance(par.default, bool):
            flags[pname] = par.default
        else:
            optional[pname] = par.default
    # The photosphere needs these even when the bolometric fn doesn't.
    required = list(dict.fromkeys(required + ['vej', 'temperature_floor']))
    return names, required, optional, flags


def _smooth_temperature_floor(time, temperature, log10_radius, vej,
                              temperature_floor, halfwidth):
    """Compact C2 blend of the temperature-floor photosphere.

    ``halfwidth`` is the fractional upper temperature offset: the blend
    covers T_floor/(1+halfwidth)..T_floor*(1+halfwidth) in unfloored
    temperature. Outside this interval the original outputs are exact.
    Radius follows the temperature change at fixed bolometric luminosity.
    """
    if halfwidth == 0:
        return temperature, log10_radius
    log_temperature = jnp.log(temperature)
    log_floor = jnp.log(temperature_floor)
    log_radius_ejecta = jnp.log(vej * 1.e5 * 86400. * time)
    # Recover the unfloored blackbody temperature from the original
    # photosphere's L = 4*pi*sigma*R**2*T**4, without duplicating constants.
    x = (log_temperature - log_floor
         + .5 * (log10_radius * jnp.log(10.) - log_radius_ejecta))
    width = jnp.log1p(halfwidth)
    u = jnp.clip((x + width) / (2 * width), 0., 1.)
    # Integral of smoothstep: y' = 3*u**2 - 2*u**3. Its first two
    # derivatives match max(0,x) at both ends, with y >= 0 throughout.
    y = width * u**3 * (2. - u)
    log_smooth_temperature = log_floor + y
    blended_radius = (log10_radius
                      + 2 * (log_temperature - log_smooth_temperature) / jnp.log(10.))
    inside = jnp.abs(x) < width
    return (jnp.where(inside, jnp.exp(log_smooth_temperature), temperature),
            jnp.where(inside, blended_radius, log10_radius))


@dataclass(eq=False, repr=False, kw_only=True)
class RedbackModel(SNModel, SimulatedModel, FittedModel):
    """SNModel wrapping a redback-jax bolometric function.

    Time anchoring: REDBACK models work on the bolometric model's native
    EXPLOSION-relative time axis. For redback sims, PEAKMJD is the explosion
    MJD, GENRANGE_TREST / Trest windows are explosion-relative, and
    SIM_PEAKMAG is not meaningful (phase 0 = explosion). The fitted t0 is
    likewise the explosion epoch.
    """

    GENMODEL_ALIASES = ('redback',)
    mjd_explode_anchor = 'native'

    #: First time, in days after explosion, of the grid the bolometric light
    #: curve is evaluated on, and so the earliest time the model covers.
    redback_tgrid_tmin: float = 0.02
    #: Last time, in days after explosion, of that grid. If unset, the
    #: simulator uses the end of GENRANGE_TREST plus 150 days and the fitter
    #: 200 days.
    redback_tgrid_tmax: Optional[float] = None
    #: Number of points on the grid, spaced geometrically between the two.
    redback_tgrid_nbin: int = 200

    #: Fractional width w of the smooth transition to the photospheric
    #: temperature floor. Zero retains a sharp floor. Smoothing applies between
    #: T_floor/(1+w) and T_floor*(1+w), preserving luminosity. This is a model
    #: setting, not a fitted parameter.
    redback_photosphere_smooth_width: float = field(
        default=0.003, metadata={"static": True})

    # bright transients get SIMSED's headroom (snlc_sim.c:28686)
    crazyflux_factor = 10.0

    # Numerical state is passed through JAX. Functions and parameter names
    # choose the program; grids and numerical defaults supply its values.
    wavelength_grid: jax.Array = None
    wavelengths: jax.Array = None
    lbol_time_grid: jax.Array = None
    _optional: dict = field(default_factory=dict)
    _registry_name: str = field(default="", metadata={"static": True})
    _bolo_fn: object = field(default=None, metadata={"static": True})
    _photosphere_fn: object = field(default=None, metadata={"static": True})
    _sig_params: tuple = field(default=(), metadata={"static": True})
    _param_names: tuple = field(default=(), metadata={"static": True})
    _flag_defaults: tuple = field(default=(), metadata={"static": True})

    model_name = "redback"

    @property
    def registry_name(self):
        return self._registry_name

    @classmethod
    def load(cls, model_path, config, n_wave_points=500,
             wavelength_range=None):
        # The generated constructor only assigns fields. All preparation and
        # validation belongs here, so tree reconstruction never loads a model.
        model = cls()
        SNModel.__init__(model, "redback", config, n_wave_points, wavelength_range)

        if (not np.isfinite(model.redback_photosphere_smooth_width)
                or model.redback_photosphere_smooth_width < 0):
            raise SNANAError("REDBACK_PHOTOSPHERE_SMOOTH_WIDTH must be finite and nonnegative",
                             function_name="RedbackModel")
        _, model._photosphere_fn = _redback()
        model._registry_name, model._bolo_fn = _bolometric_function(model_path)
        names, required, model._optional, flags = _signature(model._bolo_fn)
        model._sig_params, model._param_names = tuple(names), tuple(required)
        model._flag_defaults = tuple(flags.items())

        phase_range = config.get('genrange_trest')
        tmax = model.redback_tgrid_tmax
        if tmax is None:
            tmax = 150.0 + phase_range[1] if phase_range is not None else 200.0
        model.lbol_time_grid = jnp.asarray(np.geomspace(
            model.redback_tgrid_tmin, tmax, model.redback_tgrid_nbin))
        model.wavelengths = jnp.asarray(np.geomspace(*WAVE_RANGE, 500))
        model.choose_grid()
        return model

    def get_wavelength_range(self) -> Tuple[float, float]:
        return WAVE_RANGE

    @classmethod
    def structured_input(cls, raw):
        """The bolometric function's parameters, which are the population
        parameters: the required ones and the photosphere's must be
        configured, the optional ones may be."""
        _, bolo_fn = _bolometric_function(str(raw['GENMODEL']))
        _, required, optional, _ = _signature(bolo_fn)
        return {'population_parameters': tuple(required) + tuple(optional)}

    def derived_parameters(self, params, keys_for, **kwargs):
        """An optional parameter with no population takes the bolometric
        function's default; a required one must have a population."""
        n_sne = params['mu'].shape[0]
        return {name: jnp.full((n_sne,), value)
                for name, value in self._optional.items() if name not in params}

    def sed_param_names(self):
        return tuple(self._sig_params) + ('vej', 'temperature_floor', 'mu')

    def phase_range(self):
        """The bolometric time grid. Redback phases are explosion-relative
        throughout: PEAKMJD is the explosion MJD and GENRANGE_TREST is
        explosion-relative, so these bounds compare directly with Trest.
        """
        return (float(self.lbol_time_grid[0]),
                float(self.lbol_time_grid[-1]))

    def generate_sed(self, phase: float, params: Dict[str, float],
                     static_data: Dict) -> jnp.ndarray:
        """SED at explosion-relative time ``phase`` (days).

        params keys: the bolometric model's physics parameters (every
        signature parameter is passed explicitly, unconfigured optional ones
        with their introspected defaults), 'vej' and 'temperature_floor' for
        the photosphere, and 'mu' for the distance. The bolometric grid is
        used as computed.
        """
        tgrid = jnp.asarray(self.lbol_time_grid)

        kwargs = dict(self._flag_defaults)
        for p in self._sig_params:
            if p in params:
                kwargs[p] = params[p]
            elif p in self._optional:
                kwargs[p] = self._optional[p]
        lbol_grid = self._bolo_fn(tgrid, **kwargs)

        vej = params['vej']
        if VELOCITY_UNITS.get(self.registry_name, 'km/s') == 'c':
            vej = vej * LIGHT_KM
        tfloor = params['temperature_floor']

        tc = jnp.clip(phase, tgrid[0], tgrid[-1])
        # Recompute slopes from this parameter-dependent luminosity curve so
        # phase and physics derivatives both pass through the interpolation.
        log_tgrid = jnp.log(tgrid)
        slopes = compute_catmullrom_slopes(log_tgrid, lbol_grid)
        log10_lbol = hermite_interp_1d(jnp.log(tc), log_tgrid, lbol_grid, slopes)

        T_ph, log10_r_ph = self._photosphere_fn(
            time=tc, log10_luminosity=log10_lbol,
            vej=vej, temperature_floor=tfloor)

        T_ph, log10_r_ph = _smooth_temperature_floor(
            tc, T_ph, log10_r_ph, vej, tfloor,
            self.redback_photosphere_smooth_width)

        lam = static_data['wavelength_grid']
        nu = LIGHT_A / lam
        log10_f_nu = _log10_blackbody_flux_density(T_ph, log10_r_ph, DL10PC_CM, nu)
        f_lam = 10.0 ** log10_f_nu * LIGHT_A / lam ** 2

        distance_scale = 10.0 ** (-0.4 * params['mu'])
        return jnp.where(phase > tgrid[0], f_lam, 0.0) * distance_scale

    def get_snana_header_params(self, parameters: Dict[str, np.ndarray]
                                ) -> Dict[str, np.ndarray]:
        return {f'SIM_REDBACK_{name}': np.asarray(parameters[name], np.float32)
                for name in dict.fromkeys(list(self._param_names) + list(self._optional))}

    sim_model_index = 0                        # redback has no SNANA model

    def _assert_fittable(self) -> None:
        if self.registry_name in UNFITTABLE:
            raise SNANAError(
                f"REDBACK.{self.registry_name} cannot be fitted: "
                f"{UNFITTABLE[self.registry_name]}",
                function_name="RedbackModel")

    def get_fit_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Redback fit parameters: PKMJD, mu, then the transient function's own
        parameters.

        Bounds and scales come from ``PARAM_FIT_TABLE`` (lo, hi, init, scale),
        with velocities converted from km/s to the engine's native units.
        ``mu`` is pinned by default because it is degenerate with the
        luminosity scale; ``INISTP_mu`` non-zero frees it.
        """
        self._assert_fittable()
        params = super().get_fit_parameters()
        # t0 is the explosion epoch
        params['PKMJD'] = {
            **params['PKMJD'], 'bounds_about_peak_estimate': (-200.0, 200.0),
            'grid_search': (-16.0, -12.0, -8.0, -4.0, -2.0, 0.0, 2.0, 4.0,
                            8.0, 12.0, 16.0)}
        params['mu'] = {'bounds': (20.0, 50.0), 'scale': 1.0, 'inistp': 0.0,
                        'sim_truth': 'SIM_DLMU'}
        for p in self._param_names:
            fit_values = PARAM_FIT_TABLE[p]
            if p == 'vej' and VELOCITY_UNITS.get(self.registry_name, 'km/s') == 'c':
                fit_values = tuple(value / LIGHT_KM for value in fit_values)
            lo, hi, init, scale = fit_values
            params[p] = {'bounds': (lo, hi), 'scale': scale, 'init': init,
                         'sim_truth': f'SIM_REDBACK_{p}'}
        return params

    distance_param = 'mu'

    # t0 is the explosion epoch, so the light curve lies wholly after it
    grid_trest_range = (0.0, 100.0)
