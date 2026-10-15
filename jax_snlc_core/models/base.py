"""What an SED model is: :class:`SNModel`, the input keys a model
declares, and the surface toolkit the models load with.

A model plays the simulator's and the fitter's roles by also inheriting
:class:`simulated.SimulatedModel` and :class:`fitted.FittedModel`; this
module holds only what both consumers need, the SED itself.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, is_dataclass
from types import UnionType
from typing import (Annotated, Any, ClassVar, Dict, Literal, NamedTuple, Optional,
                    Tuple, Union, get_args, get_origin)

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core.photometry.filter_system import fluxcal_from_sed
from jax_snlc_core.physics.dust_laws import load_dust_law
from jax_snlc_core.utils.error_handling import SNANAError

# markers for a key only one input accepts: an unmarked key is the SED's,
# read by the simulation and the fit alike; a BBC key must be marked
FIT_ONLY = 'FIT_ONLY'
SIM_ONLY = 'SIM_ONLY'
BBC_ONLY = 'BBC_ONLY'

INPUT_TYPES = (int, float, bool, str)


class ModelInputKey(NamedTuple):
    """What a model's annotated attribute declares about its input key."""
    attribute: str
    kind: type
    allowed: Optional[Tuple] = None
    fit_only: bool = False
    sim_only: bool = False
    bbc_only: bool = False


def _annotation_spec(attr, ann):
    """Read an attribute's annotation as the key it declares.

    `Annotated[..., FIT_ONLY]`, `Annotated[..., SIM_ONLY]` and
    `Annotated[..., BBC_ONLY]` mark a key only one input accepts,
    `Literal[...]` names the values allowed, and either spelling of an
    optional is unwrapped.
    """
    fit_only = sim_only = bbc_only = False
    if get_origin(ann) is Annotated:
        ann, *markers = get_args(ann)
        fit_only = FIT_ONLY in markers
        sim_only = SIM_ONLY in markers
        bbc_only = BBC_ONLY in markers
    if get_origin(ann) in (Union, UnionType):
        args = [a for a in get_args(ann) if a is not type(None)]
        if len(args) == 1:
            ann = args[0]
    allowed = None
    if get_origin(ann) is Literal:
        allowed = get_args(ann)
        ann = type(allowed[0])
    if ann not in INPUT_TYPES:
        raise SNANAError(
            f"Model attribute '{attr}' is annotated {ann}, but an input key "
            f"carries one value, so it must be annotated "
            f"{', '.join(t.__name__ for t in INPUT_TYPES)}. An attribute with "
            f"no annotation is a model constant, settable by no input.",
            function_name="model_input_keys")
    return ModelInputKey(attr, ann, allowed, fit_only, sim_only, bbc_only)


def _offered(spec, consumer):
    """Whether ``consumer``'s input accepts a declared key."""
    if consumer is None:
        return True
    if consumer == 'bbc':
        return spec.bbc_only
    return not spec.bbc_only and not (spec.fit_only if consumer == 'sim'
                                      else spec.sim_only)


def model_input_keys(model_cls, consumer='fit'):
    """The input-file keys a model class declares, as
    {KEY: :class:`ModelInputKey`}.

    A lowercase scalar annotation declares an input-file setting under its
    upper-cased name. A ``jax.Array`` annotation declares prepared numerical
    data, not an input-file setting. Dataclass dictionary and tuple
    fields likewise describe prepared state rather than scalar input keys.
    Private attributes and dataclass fields excluded from construction or
    marked ``metadata={"input": False}`` are not input settings either.
    Unannotated attributes are model constants.
    ``consumer`` picks the input asking: 'sim' the simulation's, which the
    keys marked FIT_ONLY do not configure, 'fit' the fitter's, which the keys
    marked SIM_ONLY do not, 'bbc' BBC's, which is the keys marked BBC_ONLY
    alone, and None every key the model declares.
    """
    non_input_fields = (
        {f.name for f in fields(model_cls)
         if not f.init or f.metadata.get("input") is False}
        if is_dataclass(model_cls) else set())
    keys = {}
    for klass in reversed(model_cls.__mro__):
        for attr, ann in vars(klass).get('__annotations__', {}).items():
            if (ann is jax.Array or attr.startswith("_") or attr in non_input_fields
                    or ("__dataclass_fields__" in vars(model_cls) and ann in (dict, tuple))):
                continue
            if attr.islower():
                spec = _annotation_spec(attr, ann)
                if _offered(spec, consumer):
                    keys[attr.upper()] = spec
    return keys


def parse_model_input_value(key, spec, value):
    """One input-file value, typed and checked as the model declares it.

    Both parsers read a model key here, so it means the same thing wherever
    it is set: SNANA writes a logical as a Fortran T/F and lets an integer
    key be written as a float.
    """
    if spec.kind is bool:
        typed = str(value).strip().upper() in ("T", "TRUE", ".TRUE.", "1")
    elif spec.kind is int:
        typed = int(float(value))
    elif spec.kind is float:
        typed = float(value)
    else:
        typed = str(value)
    if spec.allowed is not None and typed not in spec.allowed:
        raise SNANAError(
            f"{key} = {value} is not one of "
            f"{', '.join(str(a) for a in spec.allowed)}.",
            function_name=key)
    return typed


def compute_explosion_tshift(flux, day, optmask):
    """Return the SNANA T0shiftExplode day offset for one SED template. Explosion is the
    day bin preceding the first wavelength bin to exceed 1% (OPTMASK&1) or
    0.1% (OPTMASK&2) of the template's global flux max, searched up to the
    day of peak flux; OPTMASK&32 is the bolometric peak plus a fixed 32 d;
    any other mask aborts. ``DAY -= tshift`` then places T=0 at explosion.

    ``flux`` is the (n_phase, n_wl) native flux surface and ``day`` the
    (n_phase,) native day grid, monotonic increasing.
    """
    if optmask & 1:
        frac = 0.01
    elif optmask & 2:
        frac = 0.001
    elif optmask & 32:
        bolometric = np.asarray(flux).sum(axis=1)
        return float(day[int(np.argmax(bolometric))]) + 32.0
    else:
        raise SNANAError(
            f"Invalid OPTMASK={optmask} for T0shiftExplode",
            function_name="compute_explosion_tshift")
    iday_max = int(np.argmax(flux) // flux.shape[1])
    flux_explode = frac * float(flux.max())
    crosses = np.any(flux[1:iday_max] > flux_explode, axis=1)
    return float(day[:iday_max - 1][crosses].min()) if crosses.any() else 9999.0


def ratio_to_dmu(ratio):
    """A flux ratio as the magnitude shift SNModel.shift_amplitude wants."""
    return -2.5 * jnp.log10(ratio)


def uvlam_extrapflux_sedmodel(uvlam, waves, flux):
    """Extrapolate one SED surface to shorter wavelengths.

    Apply this separately to each surface or template.

    Extend the wavelength grid to or below ``uvlam`` using its native
    spacing. At each phase, extend the blue end of the spectrum with a
    linear ramp from zero at the new grid minimum to the first significant
    positive flux sample, including original bins within that interval.

    ``uvlam`` is the target UV minimum wavelength (Angstroms), ``waves`` the
    (n_lam,) native uniform wavelength grid and ``flux`` the (n_day, n_lam)
    flux surface. Returns ``(new_waves, new_flux)``, the inputs unchanged
    when uvlam < 0 or uvlam > waves[0].
    """
    waves = np.asarray(waves, dtype=np.float64)
    flux = np.asarray(flux)
    lam_min_orig = float(waves[0])
    lam_step = float(waves[1] - waves[0])

    if uvlam < 0.0 or uvlam > lam_min_orig:
        return waves, flux

    # New LAMMIN on the original LAMSTEP grid (genmag_SEDtools.c:3249)
    n_add = 0
    lam_min_new = lam_min_orig
    while lam_min_new > uvlam:
        lam_min_new -= lam_step
        n_add += 1

    n_day, n_lam = flux.shape
    n_lam_new = n_lam + n_add
    new_waves = lam_min_new + lam_step * np.arange(n_lam_new)

    new_flux = np.zeros((n_day, n_lam_new), dtype=flux.dtype)
    new_flux[:, n_add:] = flux

    # Per-DAY anchor: smallest wavelength with signed flux > 1.0e-12
    # (genmag_SEDtools.c:3295-3296); no such bin -> ilam=0.
    above = flux > 1.0e-12
    has_anchor = above.any(axis=1)
    ilam_min = np.where(has_anchor, above.argmax(axis=1), 0)
    flux_edge = flux[np.arange(n_day), ilam_min]
    lammin_day = np.where(has_anchor, waves[ilam_min], lam_min_orig)

    # NLAM_EXTRAP = NLAM_ADD + ILAMMIN (genmag_SEDtools.c:3346-3357)
    n_extrap = n_add + ilam_min
    lam_extrap = lammin_day - lam_min_new
    fill = np.arange(n_lam_new)[None, :] < n_extrap[:, None]
    # lam_extrap is zero only where fill selects no bins for that day.
    denom = np.where(lam_extrap > 0.0, lam_extrap, 1.0)
    ramp = flux_edge[:, None] * (new_waves[None, :] - lam_min_new) / denom[:, None]
    new_flux = np.where(fill, ramp, new_flux).astype(flux.dtype)

    return new_waves, new_flux


def late_time_slope_fit(last_phases, last_seds, minslope):
    """Per-wavelength late-time magnitude slope from a template's last grid
    samples, shared by the
    SIMSED and NON1ASED loaders. SNANA fits six 1-day band-flux samples; jax
    fits the SED's last grid samples per wavelength (a band-integrated vs
    per-wavelength extrapolation difference).

    ``last_phases`` is the (NDAYFIT,) sample phases (days) and ``last_seds``
    the (NDAYFIT, n_lam) fluxes at those phases; ``minslope`` is the
    MINSLOPE_EXTRAPMAG_LATE floor (mag/day). Returns ``(slopes, reject)``
    with slopes (n_lam,) floored at ``minslope`` and reject (n_lam,) bool
    marking wavelengths where any sample is non-positive: SNANA rejects
    the whole extrapolation there and returns zero flux (:1551-1562). None
    when the phase spread is degenerate.
    """
    last_phases = np.asarray(last_phases, dtype=np.float64)
    last_seds = np.asarray(last_seds)
    reject = np.any(last_seds < 1.0e-30, axis=0)
    mags = 30.0 - 2.5 * np.log10(np.maximum(np.abs(last_seds), 1.0e-30))
    dt = last_phases - last_phases.mean()
    ss_t = np.sum(dt * dt)
    if ss_t < 1.0e-30:
        return None
    dm = mags - mags.mean(axis=0)[None, :]
    slopes = np.abs(np.sum(dt[:, None] * dm, axis=0) / ss_t)
    slopes = np.where(np.isfinite(slopes), slopes, 0.0)
    return np.maximum(slopes, minslope), reject


def resample_surfaces(native_waves, flux, grid):
    """Flux surfaces ``(..., n_phase, n_wave)`` on ``native_waves``, linearly
    resampled onto ``grid``; zero outside the native range, where SNANA's SED
    models return no flux."""
    native = np.asarray(native_waves, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    flux = np.asarray(flux)
    idx = np.clip(np.searchsorted(native, grid, side='right') - 1,
                  0, len(native) - 2)
    frac = np.clip((grid - native[idx])
                   / np.maximum(native[idx + 1] - native[idx], 1e-30), 0.0, 1.0)
    out = (1.0 - frac) * flux[..., idx] + frac * flux[..., idx + 1]
    out[..., (grid < native[0]) | (grid > native[-1])] = 0.0
    return out


def late_time_tables(phases, seds, n_valid, minslope):
    """Prepare late-time magnitude-decay slopes and boundary spectra.

    Each wavelength's decay is fitted to the template's last six valid
    phase samples. The boundary flux is set to zero where the extrapolation
    is rejected.

    ``phases`` is (n_templates, n_phase) sample phases, padded past
    ``n_valid``; ``seds`` is (n_templates, n_phase, n_wave) fluxes on the
    working grid; ``n_valid`` gives the number of valid samples in each
    template, shape (n_templates,); ``minslope`` is the minimum decay rate
    (mag/day). Returns ``(slopes, boundary_seds)``, each
    (n_templates, n_wave) float32.
    """
    NDAYFIT = 6
    n_templates, _, n_wave = seds.shape
    slopes = np.zeros((n_templates, n_wave), dtype=np.float32)
    boundary = np.zeros((n_templates, n_wave), dtype=np.float32)
    for i in range(n_templates):
        n = int(n_valid[i])
        if n > 0:
            boundary[i] = seds[i, n - 1]
        if n < NDAYFIT:
            continue
        fit = late_time_slope_fit(phases[i, n - NDAYFIT:n],
                                  seds[i, n - NDAYFIT:n], minslope)
        if fit is not None:
            slopes[i], reject = fit
            boundary[i, reject] = 0.0
    return slopes, boundary


@dataclass(init=False, eq=False, repr=False, kw_only=True)
class SNModel(ABC):
    """An SED model: built from its input, it gives a rest-frame SED at a
    phase and the FLUXCAL that SED makes through a set of bands, and moves
    its own copy of the distance where it keeps one.

    A model holding tabulated surfaces defines ``resample_to_grid(grid)``
    as well, which the constructor calls last with the grid
    :meth:`choose_grid` returns: it moves every surface the model loaded onto
    that grid, with :func:`resample_surfaces`, and stores the result as the
    JAX arrays :meth:`generate_sed` reads, so that the surfaces and the
    array the SED is returned on run over the same wavelengths.
    :meth:`SIMSEDModel.resample_to_grid
    <jax_snlc_core.models.simsed_model.SIMSEDModel.resample_to_grid>` is the
    form the bundled tabulated models take. A model whose spectrum is an
    expression evaluated at each point of the grid needs no such method.
    """

    # the directory-name prefixes that select this model (snlc_sim.c GENMODEL_NAME)
    GENMODEL_ALIASES: ClassVar[Tuple[str, ...]] = ()

    # Preparation state stays outside compiled numerical calls. Existing
    # constructors still populate it; dataclass models inherit these defaults.
    config: dict = field(init=False, default_factory=dict, repr=False)
    n_wave_points: int = field(init=False, default=500)
    wavelength_range: object = field(init=False, default=None)

    # Model-specific switches may disable training calibration corrections.
    enable_magshift = True
    enable_waveshift = True

    @classmethod
    def load(cls, model_path, config, n_wave_points=500,
             wavelength_range=None):
        """Load a model by calling its constructor by default.

        A model with a data-only constructor can override this method to read
        and prepare its inputs before constructing the model.
        """
        return cls(model_path, config, n_wave_points, wavelength_range)

    def __init__(self, model_name: str, config: Dict[str, Any],
                 n_wave_points: int = 500, wavelength_range=None):
        """Take the input before loading anything, so a model is fully built
        when its constructor returns.

        A subclass calls this first. A tabulated model then loads its
        surfaces, extends them into the UV (:attr:`uvlam_extrapflux`), and
        ends with ``self.resample_to_grid(self.choose_grid())``, so the
        working grid is set when the constructor returns; a model that
        synthesises its SED on demand calls ``self.choose_grid()`` instead.
        ``wavelength_range``
        is the rest-frame interval the caller will evaluate, which the grid
        is cut to; None takes the whole support.
        """
        self.model_name = model_name
        self.config = config
        self.n_wave_points = n_wave_points
        self.wavelength_range = wavelength_range
        keys = model_input_keys(type(self), consumer=None)
        for key, value in config.items():
            if key in keys:
                setattr(self, keys[key].attribute, value)

    #: Host-galaxy extinction law, independent of OPT_MWCOLORLAW. Uses the
    #: registered dust-law name; Fitzpatrick (1999) is the default.
    host_colorlaw: str = field(default='FITZ99_EXACT', metadata={'static': True})

    def build_host_dust_law(self):
        """The configured host law, prepared on this model's wavelength grid."""
        law = load_dust_law(self.host_colorlaw)
        law.prepare_wavelength_grid(np.asarray(self.wavelength_grid))
        return law

    def choose_grid(self):
        """The working wavelength grid: the model's support, cut to the
        caller's range where one was given, at the caller's resolution; read
        after any UV extension so the extension is never resampled away. Cuts
        that interval into ``n_wave_points`` evenly spaced points, stores them
        as ``self.wavelength_grid`` and returns them."""
        lo, hi = self.get_wavelength_range()
        if self.wavelength_range is not None:
            lo, hi = max(lo, self.wavelength_range[0]), min(hi, self.wavelength_range[1])
        self.wavelength_grid = jnp.linspace(lo, hi, self.n_wave_points)
        return self.wavelength_grid

    @abstractmethod
    def get_wavelength_range(self) -> Tuple[float, float]:
        """The model's own rest-frame support, as (min_wavelength,
        max_wavelength) in Angstroms: the extent the working grid is cut
        from, not the grid. A tabulated model reports its templates' range,
        UV extension included."""

    @abstractmethod
    def sed_param_names(self):
        """Per-event keys :meth:`generate_sed` may read from its params dict.

        A superset is allowed (a model with optional inputs declares them
        all); callers pass exactly the declared keys present in the event
        arrays, so simulator bookkeeping fields can never leak into traced
        model inputs.
        """

    @abstractmethod
    def generate_sed(self, phase: float, params: Dict[str, float],
                     static_data: Dict) -> jnp.ndarray:
        """The rest-frame SED at ``phase`` (days relative to the model’s
        phase origin), before
        dust extinction and cosmological effects, which the caller applies
        afterwards.

        ``phase`` is a scalar, and the return is the flux density in
        erg/s/cm²/Å at each point of the working grid, shape
        ``(n_wavelengths,)``; the caller maps this over the epochs of an
        event. ``params`` is the model's own parameters under the names
        :meth:`sed_param_names` declares (for example,
        ``{'width': 12.0, 'mu': 35.0}``), each a scalar for one event.
        ``static_data`` is what the
        simulator and the fitter assemble once per run for every model: the
        ``wavelength_grid`` returned by :meth:`choose_grid` and the
        calibration and extinction entries :meth:`evaluate` reads, among them
        the model's own :attr:`host_rv` and :attr:`fluxcal_scale_mag_offset`
        declarations. It holds none of the model's templates or knots, which
        are attributes set in the constructor and read here through
        ``self``.

        Use JAX-compatible numerical operations. Python control flow must
        not depend on the value of a traced parameter.
        """

    @abstractmethod
    def phase_range(self) -> Tuple[float, float]:
        """The rest-frame phase support, as (daymin, daymax) in days.

        Bounds the epochs at which true-SED spectra are generated. Supply
        this range even if ``generate_sed`` already returns zero or holds the
        spectrum fixed outside it, so callers can select covered epochs.
        """

    #: Extend the model's tabulated SED down to this rest-frame
    #: wavelength in Angstroms, ramping the flux linearly to zero at the new
    #: blue edge. The default -9 leaves the SED unchanged. A model with no
    #: tabulated SED ignores the key.
    uvlam_extrapflux: float = -9.0

    def evaluate(self, sed_params, mjds, filter_indices, mask, band_weights,
                 static_data, sed_factor=None, band_weights_dz=None,
                 band_weights_dz2=None):
        """FLUXCAL for one SN's observations: this model's SED through the
        band weights.

        ``sed_params`` is the model's own parameters, plus ``t0`` and
        ``redshift``. ``band_weights`` is [n_filters, n_wavelengths] at this
        SN's redshift. ``sed_factor`` is a rest-frame per-wavelength
        multiplier on the source that is not the model's own physics, as the
        simulator's GENMAG_SMEAR_MODELNAME is. ``band_weights_dz`` is
        d(band_weights)/dz, supplied when the redshift is fitted: a model
        differentiated by AD ignores it, one with analytic derivatives uses
        it, because the weights' own tangent varies per tangent direction
        while this does not. ``band_weights_dz2``, the second derivative,
        serves an analytic rule whose redshift terms are differentiated again.
        """
        t0, redshift = sed_params['t0'], sed_params['redshift']
        phases = jnp.where(mask, (mjds - t0) / (1.0 + redshift), 0.0)

        if 'av' in sed_params:
            ext = static_data['host_dust_law'].apply_extinction(
                jnp.ones(band_weights.shape[-1]),
                static_data['wavelength_grid'], sed_params['av'],
                (sed_params['rv'] if 'rv' in sed_params
                            else static_data['host_rv']))
            sed_factor = ext if sed_factor is None else sed_factor * ext

        model_params = {k: sed_params[k] for k in self.sed_param_names()
                        if k in sed_params}
        return fluxcal_from_sed(
            lambda phase: self.generate_sed(phase, model_params, static_data),
            phases, mask, filter_indices, band_weights,
            static_data['filter_ref_fluxes'], static_data['zp_fluxcal'],
            static_data['fluxcal_scale_mag_offset'],
            width=static_data['band_window'], sed_factor=sed_factor)

    #: The name of the model's own distance parameter, where it keeps one:
    #: the parameter :meth:`shift_amplitude` and :meth:`params_at_distance`
    #: move, which the simulator's host overrides and the fitter's starting
    #: values both go through. None means the brightness is the simulator's
    #: mu, which the caller has already updated, so there is nothing of the
    #: model's to move.
    distance_param = None

    def _amplitude_shifted(self, value, dmu):
        """The distance parameter, in physical units, moved ``dmu`` magnitudes
        fainter. A magnitude adds; SALT overrides, since x0 is not one."""
        return value + dmu

    def amplitude_jacobian(self, value):
        """d(magnitude)/d(amplitude), for a covariance carried into magnitude
        space. The reciprocal of how far the amplitude moves per magnitude,
        which :meth:`_amplitude_shifted` already defines; a model whose
        amplitude is itself a magnitude gets 1.
        """
        _, per_mag = jax.jvp(lambda d: self._amplitude_shifted(value, d),
                             (0.0,), (1.0,))
        return 1.0 / np.asarray(per_mag)

    def _amplitude_at_distance(self, mu, params):
        """The amplitude, in physical units, of a source at distance modulus
        ``mu``. ``params`` carries the model's other parameters, physical, for
        a model whose brightness depends on them. Default: the amplitude is the
        distance modulus."""
        return mu

    def shift_amplitude(self, params, dmu, scales=1.0, names=None):
        """``params`` with the amplitude moved ``dmu`` magnitudes fainter.

        Callers give the shift in magnitudes and need not know how the model
        encodes the amplitude. Accepts either a name-keyed dict of physical
        values or a fit-vector array, whose layout the caller gives as
        ``names`` and ``scales`` (the latter converts to physical and back).
        The simulator's HOSTLIB overrides and the fitter's flux-ratio steps
        are the same operation and both use this.
        """
        return self._map_amplitude(
            params, lambda value, _: self._amplitude_shifted(value, dmu),
            scales, names)

    def params_at_distance(self, params, mu, scales=1.0, names=None):
        """``params`` with the brightness set to that of a source at ``mu``.

        The absolute counterpart of :meth:`shift_amplitude`, for seeding a fit
        from cosmology.

        For a relative change in brightness, use :meth:`shift_amplitude`.
        Converting a simulated SALT amplitude back to distance requires the
        event's alpha and beta, which may differ from the model-level values.
        A relative shift cancels those terms and applies to either case.
        """
        return self._map_amplitude(
            params, lambda _, other: self._amplitude_at_distance(mu, other),
            scales, names)

    def _map_amplitude(self, params, fn, scales=1.0, names=None):
        """Apply ``fn(amplitude, other_params)`` in physical units, to a
        name-keyed dict or to the fitter's vector. ``other_params`` is
        unscaled too: a model whose brightness depends on its other parameters
        must see them physical, and SALT's x1 and c both carry scale 1.0, so
        relying on that would work by accident.
        """
        name = self.distance_param
        if name is None:
            return params
        if isinstance(params, dict):
            return {**params, name: fn(params[name], params)}
        i = names.index(name)
        scales = jnp.broadcast_to(
            jnp.asarray(scales, params.dtype), params.shape)
        physical = params * scales
        other = {n: physical[..., k] for k, n in enumerate(names)}
        return params.at[..., i].set(
            fn(physical[..., i], other) / scales[..., i])

    #: The host total-to-selective extinction ratio :meth:`evaluate` applies
    #: when the event carries ``av`` but no ``rv``, which is the case for a
    #: model fitting the extinction without its law; it reaches
    #: ``generate_sed`` as ``static_data['host_rv']``. None for a model with
    #: no host extinction of its own.
    host_rv = None

    #: A flux factor applied to every band integral, standing for a magnitude
    #: offset the model's training wrote into its files; ``fluxcal_from_sed``
    #: reads it from ``static_data`` and the default 1.0 is no offset.
    fluxcal_scale_mag_offset = 1.0

    #: The rest-frame wavelengths, as ``(minimum, maximum)`` in Angstroms,
    #: between which a band's central wavelength must fall for the simulator
    #: to generate that band at an event's redshift (SNANA's
    #: RESTLAMMIN_FILTERCEN and RESTLAMMAX_FILTERCEN). None uses the range
    #: :meth:`get_wavelength_range` declares.
    filter_center_range = None

    #: How the model continues past the last phase it covers, in SNANA's
    #: numbering: 1 (SEDFLUX) means the simulator writes no SED-true spectrum
    #: past that phase, 3 (FLAM) means it does. SALT sets this per instance
    #: from its own extrapolation configuration; a model that extrapolates no
    #: further than its support leaves the default.
    extrap_phase_method = 1

    def __repr__(self) -> str:
        """String representation of model."""
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"
