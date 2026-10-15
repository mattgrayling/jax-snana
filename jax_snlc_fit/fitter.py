"""The light-curve fitter. `LightCurveFitter` holds the compiled kernels and
fits one batch of events. `fit_static_data` assembles the arrays it evaluates
the model against, and `fit_photometry` the calibrated filters of a data
version, which the kernels take as an argument so that a calibration
systematic runs on the same compiled code. The model is also an explicit
kernel argument: registered model pytrees supply arrays at runtime, while
ordinary model classes remain compilation constants.

The fit runs SNANA's rounds, each a Newton-LM minimisation
(`jax_snlc_core.utils.newton_lm`) for every event. Round 1 recomputes the
model error at every step; later rounds fix the weights at each round's
start, in Woodbury form when the fit carries the full epoch covariance, or
recompute the diagonal error under USE_DIAG_ONLY. This module runs the rounds
and applies the data-error fudges; `fit_kernels` contains their numerical
functions. The redshift priors and photo-z band set are in `photoz`, and the
Hessian-to-errors step is in `uncertainties`.
"""

import dataclasses
import copy
import warnings
import logging
import os
import sys
import time
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap

from jax_snlc_core.io.kcor_reader import KcorReader
from jax_snlc_core.models import metadata_files
from jax_snlc_core.models.base import SNModel
from jax_snlc_core.models.fitted import (PHOTOZ_ERRMAX_BAD,
                                         PHOTOZ_ERRMAX_BAD_BESTZ,
                                         PHOTOZ_INISTP_DEFAULT,
                                         PHOTOZ_PARAMETER)
from jax_snlc_core.photometry.filter_system import (
    band_window_width, filter_values)
from jax_snlc_core.photometry.resources import (load_photometry_resources,
                                                read_calib_shifts,
                                                resolve_calib_shifts,
                                                resolve_filtlist)
from jax_snlc_core.physics.dust_laws import OPT_MWCOLORLAW_NAMES, load_dust_law
from jax_snlc_core.utils import snana_log as slog
from jax_snlc_core.utils.error_handling import SNANAError

from . import photoz, priors, fit_kernels
from .batch import (LightCurveBatch, pad_epochs, padded_batches,
                    plan_fit_batches)
from .config import FitConfig
from .config_parser import (check_photoz_trest_cuts, resolve_parameter_key,
                            resolve_parameter_keys,
                            resolve_usesim_columns, validate_inival_bounds)
from .data_loader import resolve_mw_colour_law
from .fit_results import BatchFitResult, prob_ndof
from .fit_spec import (adaptive_param_scales, covariance_output_params,
                       snana_parameter_aliases,
                       ndof_n_params, validate_fit_parameters)
from .minos import reported_errors

logger = logging.getLogger(__name__)

# SIMFIT_IDEAL_PRESCALE debug value (snlc_fit.F90:915)
SIMFIT_IDEAL_DEBUG = 777
# PKMJD's column: FittedModel declares it first in every fit vector
T0 = 0


def resolved_filters(config, kcor):
    """The fit-band names, FILTLIST_FIT's single characters resolved against
    the kcor's own filter names."""
    if config.fit.filtlist_fit is None:
        raise SNANAError("FILTLIST_FIT is required",
                         function_name="resolved_filters")
    return resolve_filtlist(config.fit.filtlist_fit, kcor)


def shift_map_with_poly(base, params, filter_lamavg):
    """A per-band mag-shift map with its A0 + A1*lam + A2*lam^2 companion key
    folded in, lam being the filter's mean wavelength in microns
    (MAGOBS_SHIFT_USRFUN, snana.F90:11337)."""
    shift = dict(base)
    if not any(params):
        return shift
    a0, a1, a2 = params
    values = filter_values(base, list(filter_lamavg), 0.0)
    for (name, lamavg), value in zip(filter_lamavg.items(), values):
        lam_um = lamavg / 1.0e4
        shift[name] = value + a0 + a1 * lam_um + a2 * lam_um ** 2
    return shift


def fiterr_windowed(config, passband_mask, filter_mean_wave, t0_phys,
                    redshifts, mjd_offsets, filter_indices, iteration):
    """SNANA GET_FUDGE_FITERR_MAXFRAC (snlc_fit.F90:11188): per-epoch
    FUDGE_FITERR_MAXFRAC where TREST, rest-lambda, passband and iteration all
    fall inside their windows; 0.0 (scalar) when the fudge is off.
    ``passband_mask`` is the fitter's per-filter FUDGE_FITERR_PASSBANDS
    gate."""
    fit = config.fit
    if fit.fudge_fiterr_maxfrac <= 0.0 or iteration > fit.fudge_fiterr_maxiter:
        return 0.0
    trest = (mjd_offsets - t0_phys[:, None]) / (1.0 + redshifts[:, None])
    t = fit.fudge_fiterr_trest
    lt = (((trest >= t[0]) & (trest <= t[1]))
          | ((trest >= t[2]) & (trest <= t[3])))
    lamrest = filter_mean_wave[filter_indices] / (1.0 + redshifts[:, None])
    lam_w = fit.fudge_fiterr_restlam
    lam = (((lamrest >= lam_w[0]) & (lamrest <= lam_w[1]))
           | ((lamrest >= lam_w[2]) & (lamrest <= lam_w[3])))
    passband = jnp.asarray(passband_mask)[filter_indices]
    return jnp.where(lt & lam & passband, fit.fudge_fiterr_maxfrac, 0.0)


def maxfrac_floor_errs(maxfrac, fluxcals, fluxcalerrs, masks):
    """SNANA MAXFRAC error floor (snlc_fit.F90:10988): add (maxfrac * the SN's
    peak flux over every band)^2 to the data variance. maxfrac is a scalar
    (all epochs) or a per-epoch array (windowed FUDGE_FITERR)."""
    if jnp.ndim(maxfrac) == 0 and maxfrac <= 0.0:
        return fluxcalerrs
    peak = jnp.max(jnp.where(masks, fluxcals, -jnp.inf), axis=1, keepdims=True)
    peak = jnp.where(jnp.isfinite(peak), peak, 0.0)
    return jnp.sqrt(fluxcalerrs**2 + jnp.where(masks, maxfrac * peak, 0.0)**2)


def fitwin_trest_edges(config, filter_indices, filters):
    """The (lo, hi) FITWIN_TREST edges for every epoch, per its band.

    Scalars unless FITWIN_TREST_FILTER named a band, matching SNANA, which
    tests only the per-filter array (snlc_fit.F90:4486).
    """
    bounds = config.fit.fitwin_trest_bounds(filters)
    if bounds is None:
        return config.fit.fitwin_trest
    lo, hi = bounds
    return jnp.asarray(lo)[filter_indices], jnp.asarray(hi)[filter_indices]


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


def resolved_config(config, model):
    """``config`` with its ``INIVAL_``, ``INISTP_`` and ``PRIOR_`` suffixes
    resolved against the fit parameters, which only ``model`` can identify,
    so the spellings the input used, such as SHAPE or PHOTOZ, go."""
    if config.photoz.dofit_photoz and any(
            name.upper() == 'DLMAG' and step == 0
            for name, step in config.model.inistp.items()):
        raise SNANAError(
            "DOFIT_PHOTOZ with INISTP_DLMAG=0 requests a cosmology-constrained "
            "amplitude, which is not implemented. Use a free amplitude.",
            function_name="resolved_config")
    result = dataclasses.replace(config, model=dataclasses.replace(
        config.model, **resolve_parameter_keys(
            config.model, list(fit_parameters(config, model)),
            snana_parameter_aliases(model))))
    check_photoz_trest_cuts(result)
    return result


def ideal_config(config):
    """The config of the SIMFIT_IDEAL refit: at most two iterations, and
    PKMJD pinned at its truth except under the debug value, which refits the
    measured flux (SIMFIT_IDEAL_PREP, snlc_fit.F90:15808, :15843)."""
    debug = config.fit.simfit_ideal_prescale == SIMFIT_IDEAL_DEBUG
    return dataclasses.replace(
        config,
        fit=dataclasses.replace(
            config.fit, _dofit_ideal=True, simfit_ideal_prescale=0,
            n_fit_iterations=min(config.fit.n_fit_iterations, 2)),
        **({} if debug else dict(model=dataclasses.replace(
            config.model, inistp={**config.model.inistp, 'PKMJD': 0.0}))))


def rounds2_strategy(config, full_covariance):
    """The weights rounds 2+ fit against. USE_FITCOV (snlc_fit.F90:10566)
    freezes them at each round's start, in Woodbury form when the fit carries
    the full epoch covariance: its colour dispersion, FUDGE_MAG_COVERR or the
    Milky Way extinction error (:10569). Without it, USE_DIAG_ONLY keeps
    sigma live (:10572)."""
    if full_covariance:
        return 'frozen_woodbury'
    return ('frozen_diag' if config.fit.use_offdiag_covariance
            else 'live_reweight')


def fit_photometry(config, model, kcor):
    """The fit bands of ``kcor`` as a data version's namelist calibrates
    them, on the common grid ``model`` is evaluated through: FILTER_LAMSHIFT
    and MAGOBS_SHIFT_PRIMARY applied, with the calibration shifts the
    model's training wrote into its metadata files.

    Returns the arrays, in fit-band order, of the photometry the compiled
    fit takes as an argument: ``common_filter_wave_grid``,
    ``common_filter_trans_grid``, ``filter_ref_fluxes`` and
    ``filter_mean_wavelengths``. The data's ``zp_fluxcal`` joins them when
    a batch is loaded.
    """
    filters = resolved_filters(config, kcor)
    lam_shift_map = config.data.filter_lamshift
    primary_mag_shift_map = shift_map_with_poly(
        config.data.magobs_shift_primary,
        config.data.magobs_shift_primary_params, kcor.filter_lamavg(filters))
    model_path = os.path.expandvars(os.path.expanduser(
        config.model.fitmodel_name))
    info_shifts = [s for path in metadata_files(type(model), model_path)
                   for s in read_calib_shifts(path)]
    if info_shifts:
        mag_s, lam_s = resolve_calib_shifts(
            info_shifts, filters, kcor.filter_survey,
            enable_mag=config.data.enable_magshift,
            enable_wave=config.data.enable_waveshift)
        primary_mag_shift_map = {**primary_mag_shift_map, **mag_s}
        lam_shift_map = {**lam_shift_map, **lam_s}
        slog.line(logger, f"Model calib shifts applied: "
                          f"{len(mag_s)} mag, {len(lam_s)} wave")
    resources = load_photometry_resources(
        kcor, model.wavelength_grid, filters, lam_shift_map=lam_shift_map,
        primary_mag_shift_map=primary_mag_shift_map)
    return {"common_filter_wave_grid": jnp.asarray(resources.common_wave_grid),
            "common_filter_trans_grid": jnp.asarray(resources.common_trans_grid),
            "filter_ref_fluxes": jnp.asarray(resources.filter_ref_fluxes),
            "filter_mean_wavelengths": jnp.asarray(
                resources.filter_mean_wavelengths)}


def fit_band_window(model, photometry):
    """How many points of ``model``'s wavelength grid the band sum spans for
    ``photometry``, or None for the whole grid (see SNModel.evaluate)."""
    return band_window_width(model.wavelength_grid,
                             photometry["common_filter_wave_grid"],
                             photometry["common_filter_trans_grid"])


def fit_static_data(model, band_window, kernel_cache=None):
    """What a fitter evaluates ``model`` against beyond the data's
    photometry: its wavelength grid and range, the host dust law the
    simulator also builds, and the band sum's ``band_window``."""
    # The default preparation depends only on the law and wavelength grid.
    # Keep custom preparation independent: it can depend on additional state.
    if (kernel_cache is not None
            and type(model).build_host_dust_law is SNModel.build_host_dust_law):
        dust = kernel_cache.setdefault('host_dust', {})
        key = (model.host_colorlaw, tuple(np.asarray(model.wavelength_grid)),
               jax.config.jax_enable_x64)
        if key not in dust:
            dust[key] = model.build_host_dust_law()
        host_dust_law = dust[key]
    else:
        host_dust_law = model.build_host_dust_law()
    return {
        "wavelength_grid": model.wavelength_grid,
        "host_dust_law": host_dust_law,
        # the fitter intersects it with RESTLAMBDA_FITRANGE (snlc_fit.F90:1421)
        "model_wave_range": tuple(float(v)
                                  for v in (model.filter_center_range
                                            or model.get_wavelength_range())),
        "host_rv": model.host_rv,
        "fluxcal_scale_mag_offset": model.fluxcal_scale_mag_offset,
        # the fitter's SED calls exclude the sim-only late-time extrapolation
        "latetime_extrap": False,
        "band_window": band_window,
    }


def attach_mw_attenuation(config, batch):
    """``batch`` with the observer-frame Milky Way attenuation of each SN's
    E(B-V), as the loader resolved it from OPT_MWEBV, on the common grid of
    its photometry, and the extinction-magnitude error at each band's mean
    wavelength, where SNANA evaluates it (snana.F90:15720-15742).

    ``config`` is the data version's: its OPT_MWCOLORLAW, RV_MWCOLORLAW and
    PARLIST_MWCOLORLAW, the simulation's own law where it sets none. The
    error is the extinction at E(B-V) plus its error less that at E(B-V)
    (snana.F90:15624-15629). A batch with neither carries ones and zero
    errors, and the law is not evaluated, as GALextinct returns at A_V = 0
    (MWgaldust.c:474).
    """
    opt, rv = resolve_mw_colour_law(config.data, batch.sim_mw_colour_law)
    law = load_dust_law(OPT_MWCOLORLAW_NAMES[opt])
    if config.data.parlist_mwcolorlaw:
        law.set_params(config.data.parlist_mwcolorlaw)
    wave = np.asarray(batch.photometry['common_filter_wave_grid'])
    trans = np.asarray(batch.photometry['common_filter_trans_grid'])
    ebv = np.asarray(batch.mwebvs, dtype=np.float64)
    ebv_err = np.asarray(batch.mwebv_errs, dtype=np.float64)
    if not (np.any(ebv != 0.0) or np.any(ebv_err != 0.0)):
        return dataclasses.replace(
            batch, mw_att_obs=jnp.ones((batch.n_sne, len(wave))),
            mwxt_magerrs=jnp.zeros((batch.n_sne, trans.shape[0])))

    # the law must cover the fit bands, where SNANA tabulates it
    # (fill_TABLE_MWXT_SEDMODEL, genmag_SEDtools.c; MWgaldust.c:688-694)
    law.prepare_wavelength_grid(wave)
    covered = wave[(trans > 0).any(axis=0)]
    law.check_wave_range(float(covered.min()), float(covered.max()))
    lamavg = np.asarray(batch.photometry['filter_mean_wavelengths'])
    hi = np.clip(np.searchsorted(wave, lamavg), 1, len(wave) - 1)
    lo = hi - 1
    frac = (lamavg - wave[lo]) / (wave[hi] - wave[lo])

    def axav(e):
        """A(lam)/A(V) on the grid for each event's E(B-V) ``e``, and A(lam)
        at each band's mean wavelength; Somm25's shape reads AV = RV*E(B-V)
        (genmag_SEDtools.c:2580-2645)."""
        curve = np.atleast_2d(np.asarray(law.extinction_curve(
            wave, rv, av=rv * e[:, None])))
        at_lamavg = curve[..., lo] + (curve[..., hi] - curve[..., lo]) * frac
        return curve, rv * e[:, None] * at_lamavg

    curve, xt = axav(ebv)
    _, xt_plus_err = axav(ebv + ebv_err)
    mw_attenuation = jnp.power(10.0, -0.4 * rv * jnp.asarray(ebv)[:, None]
                    * jnp.asarray(curve))
    return dataclasses.replace(batch, mw_att_obs=mw_attenuation,
                               mwxt_magerrs=jnp.asarray(xt_plus_err - xt))


def empty_results(fit_parameters, max_obs, minos=None):
    """Zero-SN result, so cuts rejecting everything still write a FITRES header."""
    names = list(fit_parameters)
    zero = jnp.zeros(0)
    return BatchFitResult(
        snids=np.zeros(0, dtype=int), params={n: zero for n in names},
        param_errors={n: zero for n in names},
        error_flags={n: jnp.zeros(0, dtype=jnp.int32) for n in names},
        minos_errors=({n: jnp.zeros((0, 2)) for n in names} if minos is not None else None),
        minos_status=({n: jnp.zeros((0, 2), dtype=int) for n in names}
                      if minos is not None else None),
        param_covariances=jnp.zeros((0, len(names), len(names))),
        cov_output_params=covariance_output_params(fit_parameters),
        chi_squared=zero, ndof=zero, prior_chi2=zero,
        observation_mask=jnp.zeros((0, max_obs), dtype=bool),
        hessian_cond=zero, edm=zero,
        hessian_positive=jnp.zeros(0, dtype=bool),
        z_prior=(zero, zero) if 'zPHOT' in fit_parameters else None,
        iter1_verdicts={},
    )


class LightCurveFitter:
    """Fit light curves using a shared model, calibration and fit configuration."""

    def __init__(self, model, *, calibration=None, config=None, **settings):
        """Load a named model and calibration for repeated light-curve fits.

        ``model`` is a bundled model name, a model directory or a Redback
        model name. Omit ``calibration`` to use the bundled filters, or supply
        a calibration file. Fit settings can be supplied as keyword arguments
        or through a :class:`~jax_snlc_fit.config.FitConfig` supplied as
        ``config``; keywords override the configuration. See the fitting
        settings page for accepted keyword names.
        """
        from .python import initialize
        initialize(self, model, calibration=calibration, config=config, **settings)

    @classmethod
    def _from_prepared(cls, config, model, kcor, shape, *, kernel_cache=None):
        """Use resources and batch dimensions already prepared by the pipeline."""
        fitter = cls.__new__(cls)
        fitter._initialize(config, model, kcor, shape, kernel_cache=kernel_cache)
        return fitter

    def _initialize(self, config: FitConfig, model: Any, kcor: KcorReader,
                    shape: Dict[str, Any], *, kernel_cache=None):
        self._kernel_cache = {} if kernel_cache is None else kernel_cache
        config = copy.deepcopy(config)
        self.calibration = kcor
        self._input_config = copy.deepcopy(config)
        self._warned_limits = set()
        model_parameters = model.get_fit_parameters()
        validate_fit_parameters(model, model_parameters)
        self.model = model
        self.epoch_lengths = list(shape['epoch_lengths'])
        self.batch_size = shape['batch_size']
        self.n_z = shape['n_z']
        self.static_data = fit_static_data(model, shape['band_window'], self._kernel_cache)
        # the epoch lengths a batch has been compiled at
        self._compiled_lengths = set()

        # the fit vector: the model's parameters, then zPHOT under photo-z
        self.model_param_names = list(model_parameters)
        self.n_model_params = len(self.model_param_names)
        self.fit_parameters = fit_parameters(config, model)
        self.param_names = list(self.fit_parameters)

        self.aliases = snana_parameter_aliases(model)
        config = resolved_config(config, model)
        validate_inival_bounds(config, model_parameters)
        self.config = config
        # USESIM_<p> pins, USESIM_INIVAL floats (snlc_fit.F90:8305, :8341)
        self.usesim_columns, usesim_pinned = resolve_usesim_columns(
            config, model_parameters, self.param_names, self.aliases)
        self.ideal_fitter = (
            type(self)._from_prepared(ideal_config(config), model, kcor, shape,
                                      kernel_cache=self._kernel_cache)
            if config.fit.simfit_ideal_prescale else None)

        self.priors = priors.active(model, config, self.model_param_names)

        # INISTP = 0 pins the parameter (SNANA FLOATPAR=.FALSE.)
        inistp = {name: spec['inistp'] for name, spec in
                  self.fit_parameters.items() if 'inistp' in spec}
        inistp.update(config.model.inistp)
        for name, value in config.model.fixed.items():
            if value is None:
                inistp.pop(name, None)
            else:
                inistp[name] = 0.0
        if config.model.lfixpar_all or config.model.lfixpar_adjust:
            # zeroes INISTP for every fit parameter, zPHOT included
            # (snlc_fit.F90:1974-1978)
            inistp.update({name: 0.0 for name in self.param_names})
        self.held_params = tuple(sorted(
            {name for name, step in inistp.items() if step == 0.0}
            | set(usesim_pinned)))
        for name, value in config.model.fixed.items():
            if value is not None and name in config.model.initial and config.model.initial[name] != value:
                raise SNANAError(f"Conflicting initial and fixed values for {name}")
        # round 1 also holds what the model defers (BayeSN's epsilon)
        self.round1_held_params = tuple(sorted(set(self.held_params) | {
            n for n, spec in self.fit_parameters.items() if spec.get('deferred')}))

        # params_scaled = params_physical / param_scales; zPHOT's is per event
        self.param_scales = jnp.array(
            [spec.get('scale', 1.0) for spec in model_parameters.values()])

        # RESTLAMBDA_USEFIT (snlc_fit.F90:1421, :4195)
        model_lo, model_hi = self.static_data['model_wave_range']
        user_lo, user_hi = config.fit.restlambda_fitrange
        self.restlambda_usefit = (max(model_lo, user_lo), min(model_hi, user_hi))
        if self.restlambda_usefit[1] <= self.restlambda_usefit[0]:
            raise SNANAError(
                f"RESTLAMBDA_FITRANGE {config.fit.restlambda_fitrange} and the "
                f"model's range {self.static_data['model_wave_range']} do not "
                f"overlap, so no epoch can be fitted.",
                function_name="LightCurveFitter.__init__")

        self._set_filters(resolved_filters(config, kcor))

        self._prepare_kernels()

    def _prepare_kernels(self):
        """Snapshot numerical settings and select compatible compiled functions.

        The fitter owns the editable settings. Compiled functions retain the
        snapshot made here, so later fitter changes cannot alter a shared
        function, including when JAX traces it for a new input shape.
        Replace the snapshot on preparation; never update it in place.
        """
        self._compiled_lengths.clear()
        config = self.config
        # Copy mutable configuration records into the compilation snapshot.
        # Omit resource paths and model-loading inputs: they are already
        # resolved into model arguments and must not prevent compilation reuse.
        kernel_config = FitConfig(
            model=dataclasses.replace(copy.deepcopy(config.model),
                                      fitmodel_name=None, model_type=None,
                                      _model_key_values={}),
            fit=copy.deepcopy(config.fit),
            photoz=copy.deepcopy(config.photoz),
        )
        self._kernel_settings = {
            'config': kernel_config,
            'static_data': {key: value for key, value in self.static_data.items()
                            if key not in ('host_rv', 'fluxcal_scale_mag_offset')},
            'static_model': jax.tree_util.all_leaves([self.model]),
            'fit_parameters': copy.deepcopy(self.fit_parameters),
            'priors': copy.deepcopy(self.priors),
            'param_names': tuple(self.param_names),
            'held_params': self.held_params,
            'round1_held_params': self.round1_held_params,
            'n_model_params': self.n_model_params,
            'n_filters': self.n_filters,
            'n_z': self.n_z,
            'restlambda_usefit': self.restlambda_usefit,
            'colour_dispersion': self.colour_dispersion,
            'full_covariance': self.full_covariance,
            'fudge_mag_coverr_r': self.fudge_mag_coverr_r,
        }
        # Unregistered ordinary models remain captured, without imposing JAX
        # pytree or hashing requirements on existing model authors.
        self._model_input = (None if self._kernel_settings['static_model'] else self.model)
        compiled = fit_kernels.compiled_functions(
            self._kernel_settings, self._kernel_cache, self.model)
        self._run_round = compiled['round']
        self._run_weights = compiled['weights']
        self._run_errors = compiled['errors']
        self._run_adjust = compiled['adjust']
        self._run_grid = compiled['grid']
        self._run_flux = compiled['flux']
        self.reject_epochs = compiled['reject_epochs']

    def _set_filters(self, filters):
        """Prepare the per-band settings captured by the fitting kernels."""
        config = self.config
        model = self.model
        self.filters = list(filters)
        self.config.fit.filtlist_fit = list(filters)
        pb = config.fit.fudge_fiterr_passbands
        selected = ([] if isinstance(pb, str) and pb.strip().upper() == "NULL"
                    else resolve_filtlist(pb, self.calibration))
        self.fudge_fiterr_passband_mask = tuple(f in selected for f in self.filters)
        cov = config.fit.fudge_mag_coverr
        self.fudge_mag_coverr_r = tuple(
            1.0 - 10.0 ** (-0.4 * value)
            for value in filter_values(cov, self.filters, 0.0)) if cov else ()

        self.n_filters = len(self.filters)

        self.colour_dispersion = bool(config.fit.use_offdiag_covariance
                                      and model.has_offdiag_covariance)
        self.full_covariance = bool(self.colour_dispersion
                                    or any(self.fudge_mag_coverr_r)
                                    or config.fit.opt_covar_mwxterr > 0)
        self.rounds2_strategy = rounds2_strategy(config, self.full_covariance)

    def prepare_shape(self, epoch_lengths, n_z, filters, band_window):
        """Update reusable batch dimensions and the current fit bands."""
        self.epoch_lengths = list(epoch_lengths)
        bands_changed = list(filters) != self.filters
        window_changed = band_window != self.static_data['band_window']
        self.static_data['band_window'] = band_window
        grid_changed = n_z != self.n_z
        self.n_z = n_z
        if bands_changed:
            self._set_filters(filters)
        if bands_changed or window_changed or grid_changed:
            self._prepare_kernels()
        if self.ideal_fitter is not None:
            self.ideal_fitter.prepare_shape(epoch_lengths, n_z, filters, band_window)

    @classmethod
    def from_model(cls, model, *, calibration=None, config=None, **settings):
        """Prepare a model for repeated in-memory fits; no output files are written."""
        return cls(model, calibration=calibration, config=config, **settings)

    def fit(self, photometry=None, *, mjd=None, flux=None, flux_err=None, band=None,
            redshift=None, mwebv=0., snid='0', flux_unit='fluxcal', zp=None,
            redshift_cmb=None, metadata=None, objects=None,
            column_map=None, band_map=None, progress=True, apply_selection=False):
        """Fit single-object arrays or photometry/object tables in memory.

        Supply aligned ``mjd``, ``flux`` and ``flux_err`` arrays; ``band``
        may be one calibration filter name or an aligned array. ``band_map``
        maps observation labels to full calibration names. ``redshift`` is
        required unless fitting photometric redshift. ``mwebv`` defaults to
        zero; ``metadata`` carries additional object information such as
        ``peakmjd`` or host photometric-redshift measurements.

        Alternatively, pass a ``photometry`` table with one row per
        observation and a ``snid`` identifying each object. Object properties
        such as ``redshift`` and ``mwebv`` may be repeated in this table; each
        must agree across all observations of the same object. Results follow
        the order of first appearance of each ``snid``.

        You can also supply separate ``photometry`` and ``objects`` tables joined
        by ``snid``. The photometry columns are ``snid``, ``mjd``, ``flux``,
        ``flux_err`` and ``band``. Object columns include ``snid`` and
        ``redshift``. ``column_map`` maps these names to your column names.
        Object columns for ``mwebv``, ``flux_unit`` and ``zp`` take precedence
        over the corresponding keyword defaults.

        ``flux_unit`` is ``'fluxcal'`` (default), ``'Jy'`` or ``'uJy'``.
        ``zp`` applies to fluxcal only and defaults to 31.4. ``redshift_cmb``
        is retained as metadata; the fit uses ``redshift``.

        Sample selection is disabled by default. Set ``apply_selection=True``
        to apply configured event and epoch cuts and the CLI's post-fit cuts.
        Supplying selection settings without enabling selection raises an error.

        ``progress=False`` silences the notice shown before preparing new
        fit functions. Repeated fits with reusable functions show no notice.

        Returns a :class:`BatchFitResult` with fitted parameters and an
        outcome for every input object. No output files are written.
        """
        from .python import fit
        return fit(self, mjd=mjd, flux=flux, flux_err=flux_err, band=band,
                   redshift=redshift, mwebv=mwebv, snid=snid, flux_unit=flux_unit,
                   zp=zp, redshift_cmb=redshift_cmb, metadata=metadata,
                   photometry=photometry, objects=objects,
                   column_map=column_map, band_map=band_map, progress=progress,
                   apply_selection=apply_selection)

    def configuration(self):
        """Return a detached copy of the supplied configuration."""
        from .inputs import supplied_configuration
        return supplied_configuration(self._input_config)

    def reset_shape(self):
        """Recompute observation bins and photo-z search size on the next fit."""
        self.epoch_lengths = []
        self.n_z = None
        self._prepare_kernels()
        if self.ideal_fitter is not None:
            self.ideal_fitter.reset_shape()

    def predict(self, result, mjd, band, *, snid=None, flux_unit=None, zp=None):
        """Evaluate a stored fit at the requested MJDs and bands.

        Returns a dictionary with ``mjd``, ``band``, ``flux`` and
        ``model_flux_err`` arrays. ``model_flux_err`` is the model's
        intrinsic photometric uncertainty, not uncertainty propagated from
        the fitted parameters.

        For a result containing several objects, ``snid`` selects the one
        to evaluate. Output uses that object's fitted flux units and zero
        point unless ``flux_unit`` or ``zp`` specifies another calibration.
        """
        from .python import predict
        return predict(self, result, mjd, band, snid=snid, flux_unit=flux_unit, zp=zp)

    def band_weights(self, photometry, z, mw_att):
        return fit_kernels.band_weights(self._kernel_settings, photometry, z, mw_att)

    def events_band_weights(self, photometry, z, mw_att):
        return fit_kernels.events_band_weights(self._kernel_settings, photometry, z, mw_att)

    def z_grid_band_weights(self, photometry):
        return fit_kernels.z_grid_band_weights(self._kernel_settings, photometry)

    def interpolated_band_weights(self, photometry, z_weights, z, mw_att):
        return fit_kernels.interpolated_band_weights(
            self._kernel_settings, photometry, z_weights, z, mw_att)

    def model_flux(self, model, photometry, params_arr, mjds, filter_indices, mask,
                   band_weights, redshift, band_weights_dz=None,
                   band_weights_dz2=None, *, static_data=None):
        return fit_kernels.model_flux(
            self._kernel_settings, model, photometry, params_arr, mjds, filter_indices, mask,
            band_weights, redshift, band_weights_dz, band_weights_dz2, static_data=static_data)

    def model_fracerr(self, model, photometry, params_arr, mjds, filter_indices,
                      mask, band_weights, band_weights_err, redshift,
                      error_overrides=None, error_redshift=None,
                      band_weights_err_dz=None, *, static_data=None):
        return fit_kernels.model_fracerr(
            self._kernel_settings, model, photometry, params_arr, mjds, filter_indices, mask,
            band_weights, band_weights_err, redshift, error_overrides, error_redshift,
            band_weights_err_dz, static_data=static_data)

    def model_flux_error(self, flux, fracerr, mask):
        return fit_kernels.model_flux_error(self._kernel_settings, flux, fracerr, mask)

    def fudged_fluxcalerrs(self, filter_mean_wave, maxfrac, t0_offsets,
                           redshifts, mjd_offsets, filter_indices, masks,
                           fluxcals, fluxcalerrs, iteration):
        """The error the fit divides by: the measured one with the MAXFRAC
        fudge in quadrature (JEP_FUDGEFLUX_ERR, snlc_fit.F90:4208-4215), for
        the fit's rounds and for the cut that recounts SNR afterwards.
        """
        window = fiterr_windowed(self.config, self.fudge_fiterr_passband_mask,
                                 filter_mean_wave, t0_offsets, redshifts,
                                 mjd_offsets, filter_indices, iteration)
        return maxfrac_floor_errs(jnp.maximum(maxfrac, window), fluxcals,
                                  fluxcalerrs, masks)

    def fitwin_trest_ok(self, filter_indices, mjd_offsets, t0_offsets,
                        redshifts):
        """Epochs inside FITWIN_TREST at the given peak time and redshift
        (REJECT_TREST, snlc_fit.F90:4486)."""
        trest_lo, trest_hi = fitwin_trest_edges(self.config, filter_indices,
                                                self.filters)
        trest = ((mjd_offsets - t0_offsets[:, None])
                 / (1.0 + redshifts[:, None]))
        return (trest >= trest_lo) & (trest <= trest_hi)

    def iter1_param_index(self, key):
        """Column a round-1 FITWIN key names, in this model's fit vector."""
        return self.param_names.index(resolve_parameter_key(
            key, self.param_names, self.aliases))

    def restlambda_mask(self, photometry, filter_indices, redshifts):
        return fit_kernels.restlambda_mask(
            self._kernel_settings, photometry, filter_indices, redshifts)

    def fit_sample(self, sample: LightCurveBatch,
                   start: Dict[str, jnp.ndarray], *, progress=False):
        """Fit ``sample`` from ``start`` (one [n_sne] array per parameter).
        The sample carries its photometry, its Milky Way attenuation
        (:func:`attach_mw_attenuation`) and, under photo-z, its redshift
        priors, as :func:`~jax_snlc_fit.run.load` gives them.

        Returns the results in the sample's own order and one record per
        fitted batch: its events, seconds, epoch length, and whether it was
        the first at its length, which compiles the fit. Every batch is
        ``batch_size`` events at one of ``epoch_lengths``, a short one padded
        by repeating its own events; the padding is dropped from the results,
        whose epoch mask is at the sample's length.
        """
        if not sample.n_sne:
            return empty_results(self.fit_parameters, sample.max_obs,
                                 self.config.fit.minos), []
        items = plan_fit_batches(sample.n_obs_per_sn, self.epoch_lengths,
                                 self.batch_size)
        if not items:
            return empty_results(self.fit_parameters, sample.max_obs,
                                 self.config.fit.minos), []
        order = np.concatenate([rows for _, rows, _ in items])
        input_order = np.empty(sample.n_sne, dtype=int)
        input_order[order] = np.arange(sample.n_sne)
        slog.line(logger, f"Fitting {sample.n_sne} SNe in {len(items)} "
                          f"batch(s) of {self.batch_size}")
        lengths = sorted({length for length, _, _ in items})
        if len(lengths) > 1:
            slog.line(logger, f"Epoch bins padded to {lengths} "
                              f"rather than all to {sample.max_obs}")

        if any(length not in self._compiled_lengths for length in lengths):
            message = "Preparing fit functions (JAX compilation may take a moment)..."
            if progress:
                print(message, file=sys.stderr, flush=True)
            else:
                slog.line(logger, message)

        z_weights = None
        if self.config.photoz.dofit_photoz:
            n_nohost = int(np.sum(~np.asarray(sample.z_priors.host_ok)))
            if n_nohost:
                slog.warn(logger, "PHOTOZ",
                          f"{n_nohost} of {sample.n_sne} events have no host "
                          f"photo-z: fitted with a flat redshift prior "
                          f"(width {photoz.PHOTOZ_STD_NOHOST:g}) over "
                          f"PHOTOZ_BOUND")
            z_weights = self.z_grid_band_weights(sample.photometry)

        results, records = [], []
        for ic, (epoch_length, rows, idx) in enumerate(items):
            carries_jit = epoch_length not in self._compiled_lengths
            t_batch = time.time()
            batch_results = self._fit_batch(
                sample.take(idx, epoch_length),
                {k: v[idx] for k, v in start.items()}, z_weights)
            dt_batch = time.time() - t_batch
            self._compiled_lengths.add(epoch_length)
            batch_results = batch_results.take(np.arange(len(rows)))
            results.append(dataclasses.replace(
                batch_results, observation_mask=pad_epochs(
                    batch_results.observation_mask[:, :sample.max_obs],
                    sample.max_obs)))

            done = (f"Batch {ic + 1}/{len(items)}: {len(rows)} SNe at "
                    f"{epoch_length} epochs, {dt_batch:.1f}s")
            slog.line(logger, f"{done} (may include JIT)" if carries_jit
                      else f"{done} ({len(rows) / dt_batch:.0f} SNe/sec)")
            records.append({'n_sne': len(rows), 'seconds': dt_batch,
                            'epoch_length': epoch_length,
                            'carries_jit': carries_jit})
        return BatchFitResult.concatenate(results).take(input_order), records

    def stack(self, params, reference_mjds):
        """``params`` (one [n_sne] array per fitted name) as the
        [n_sne, n_params] physical fit vector, PKMJD as its offset from
        ``reference_mjds``."""
        return jnp.stack([params[n] - reference_mjds if n == 'PKMJD'
                          else params[n] for n in self.param_names], axis=-1)

    def unstack(self, stacked, reference_mjds):
        """The inverse of :meth:`stack`: one [n_sne] array per fitted name,
        PKMJD back in MJD."""
        return {n: (stacked[:, i] + reference_mjds if n == 'PKMJD'
                    else stacked[:, i])
                for i, n in enumerate(self.param_names)}

    def _bounds(self, physical, peak_estimate, reference_mjds=0.):
        """Per-SN bounds [n_sne, n_params] from each parameter's declaration.

        A parameter declares its bounds under exactly one key: ``bounds`` for an
        absolute ``(lo, hi)``, ``bounds_relative_to_init`` for ``(lo_factor,
        hi_factor)`` applied to its starting value: SALT's x0, which spans
        orders of magnitude with redshift, or ``bounds_about_peak_estimate``
        for ``(lo, hi)`` about ``peak_estimate``.
        """
        lows, highs = [], []
        for i, spec in enumerate(self.fit_parameters.values()):
            if 'bounds_relative_to_init' in spec:
                lo_factor, hi_factor = spec['bounds_relative_to_init']
                low, high = physical[:, i] * lo_factor, physical[:, i] * hi_factor
            elif 'bounds_about_peak_estimate' in spec:
                lo, hi = spec['bounds_about_peak_estimate']
                low, high = peak_estimate + lo, peak_estimate + hi
            else:
                lo, hi = spec['bounds']
                low = jnp.full(physical.shape[0], lo)
                high = jnp.full(physical.shape[0], hi)
            lows.append(low)
            highs.append(high)
        low, high = jnp.stack(lows, axis=1), jnp.stack(highs, axis=1)
        for name, requested in self.config.model.bounds.items():
            index = self.param_names.index(name)
            lo, hi = requested
            if not np.isfinite(requested).all() or lo >= hi:
                raise SNANAError(f'Invalid bounds for {name}: {requested}')
            offset = reference_mjds if index == T0 else 0.
            lo, hi = lo-offset, hi-offset
            accepted = (lo >= low[:, index]) & (hi <= high[:, index])
            if not bool(jnp.all(accepted)) and name not in self._warned_limits:
                warnings.warn(f"Rejected bounds {requested} for {name} outside model limits; "
                              "retaining the model limits for those objects", stacklevel=2)
                self._warned_limits.add(name)
            low = low.at[:, index].set(jnp.where(accepted, lo, low[:, index]))
            high = high.at[:, index].set(jnp.where(accepted, hi, high[:, index]))
        for form, index, coefficients in self.priors:
            support = priors.support(form, coefficients)
            if support is None:
                continue
            offset = reference_mjds if coefficients.get('public_time') else 0.
            lo, hi = support[0] - offset, support[1] - offset
            low = low.at[:, index].set(jnp.maximum(low[:, index], lo))
            high = high.at[:, index].set(jnp.minimum(high[:, index], hi))
        if bool(jnp.any(low >= high)):
            raise SNANAError('Parameter bounds and prior support do not overlap')
        return low, high

    def _scales(self, batch, z_priors):
        """Per-SN parameter scales [n_sne, n_params]: each parameter's own,
        the amplitude's from the starting distance, and zPHOT's the prior's
        width, or its default step when that width is zero."""
        scales = jnp.tile(self.param_scales[None, :], (batch.n_sne, 1))
        for name, at_distance in adaptive_param_scales(
                self.model, self.config.fit.start_distance_modulus(
                    batch.redshifts)).items():
            scales = scales.at[:, self.param_names.index(name)].set(at_distance)
        if self.config.photoz.dofit_photoz:
            z_scale = jnp.where(z_priors.z_prior_err > 0.0,
                                z_priors.z_prior_err, PHOTOZ_INISTP_DEFAULT)
            scales = jnp.column_stack([scales, z_scale])
        return scales

    def _redshifts(self, params, scales, z_data):
        return fit_kernels._redshifts(self._kernel_settings, params, scales, z_data)

    def _fit_batch(self, batch, start, z_weights) -> BatchFitResult:
        """Fit one batch with vmap over Newton-LM, each SN an independent
        optimisation in scaled space, each step clipped to the SN's bounds.

        ``start`` holds the starting values and ``z_weights`` the sample's
        :meth:`z_grid_band_weights`, None for a spec-z fit, as the batch's
        redshift priors are.
        """
        cfg = self.config
        photo = cfg.photoz.dofit_photoz
        ref, photometry = batch.reference_mjds, batch.photometry
        z_priors = batch.z_priors
        if photo and 'zPHOT' not in start:
            start = {**start, 'zPHOT': z_priors.z_init}
        scales = self._scales(batch, z_priors)
        physical = self.stack(start, ref)
        explicit = set(cfg.model.initial) | {
            n for n, v in cfg.model.fixed.items() if v is not None}
        if photo:
            z_lower, z_upper = photoz.redshift_bounds(cfg, z_priors)
            if 'zPHOT' not in explicit:
                physical = physical.at[:, -1].set(jnp.clip(
                    physical[:, -1], z_lower, z_upper))
        params = physical / scales
        # PKMJD's bound centre; without an estimate the start is the reference
        peak_estimate = (jnp.asarray(batch.peakmjd_offsets)
                         if batch.peakmjd_offsets is not None
                         else jnp.zeros(batch.n_sne))

        constrained = bool(cfg.model.bounds or any(
            value is not None and 'distribution' in value for value in cfg.model.priors.values()))
        input_limits = self._bounds(physical, peak_estimate, ref) if constrained or explicit else None
        if photo and input_limits is not None:
            lower, upper = input_limits
            input_limits = (lower.at[:, -1].set(jnp.maximum(lower[:, -1], z_lower)),
                            upper.at[:, -1].set(jnp.minimum(upper[:, -1], z_upper)))
            if bool(jnp.any(input_limits[0] >= input_limits[1])):
                raise SNANAError('Parameter bounds and redshift PDF support do not overlap')
        for name in explicit:
            index = self.param_names.index(name)
            values = physical[:, index]
            invalid = (~jnp.isfinite(values) | (values < input_limits[0][:, index])
                       | (values > input_limits[1][:, index]))
            if bool(jnp.any(invalid)):
                raise SNANAError(f'Explicit value for {name} is outside accepted bounds/prior support')
        limits = input_limits if constrained else None
        if limits is not None:
            physical = jnp.clip(physical, *limits)
            params = physical / scales

        if (photo and not cfg.fit.skip_grid_search
                and batch.sncid_inival is None):
            params = self._grid_start(batch, params, scales, z_priors, limits)
            # bounds about the grid's start (snlc_fit.F90:8631-8635)
            physical = params * scales
        lo, hi = limits if limits is not None else self._bounds(physical, peak_estimate, ref)
        if photo:
            lo = lo.at[:, -1].set(jnp.maximum(lo[:, -1], z_lower))
            hi = hi.at[:, -1].set(jnp.minimum(hi[:, -1], z_upper))
        lo, hi = lo / scales, hi / scales
        if photo:
            params = params.at[:, -1].set(jnp.clip(
                params[:, -1], lo[:, -1], hi[:, -1]))

        # bands in range at the start z (REDSHIFT_FIT, snlc_fit.F90:8679);
        # photo-z tests RESTLAMBDA at its trial z (Z8TEST, :1776)
        mask = batch.observation_mask & self.restlambda_mask(
            photometry, batch.filter_indices,
            self._redshifts(params, scales, batch.redshifts))

        # a spec-z fit's band weights, once for the batch; a photo-z fit's
        # follow its fitted redshift
        bw = (None if photo else self.events_band_weights(
            photometry, batch.redshifts, batch.mw_att_obs))

        # FITINI_ADJUST (snlc_fit.F90:9594, :9801), after FITINI_PHOTOZ
        # under photo-z (:1962-1966)
        params, chi2red_ok, chi2red_ok2 = self._run_adjust(
            self._model_input, photometry, z_weights, params, scales, batch.mjd_offsets,
            batch.filter_indices, mask, batch.fluxcals, batch.fluxcalerrs,
            batch.redshifts, bw, batch.n_obs_per_sn, z_priors,
            batch.mw_att_obs, reference_mjds=batch.reference_mjds,
            limits=limits)
        if photo:
            params = params.at[:, -1].set(jnp.clip(
                params[:, -1], lo[:, -1], hi[:, -1]))

        (params, chi2, prior_chi2, mask, verdicts, uncertainty) = self._fit_rounds(
            batch, params, scales, lo, hi, mask, z_priors, z_weights, bw)
        errors, covariances, hessian_cond, edm, hessian_positive = uncertainty[:5]
        profiles = uncertainty[5] if cfg.fit.minos is not None else None
        if profiles is not None:
            errors = reported_errors(errors, *profiles)
        for name, verdict in (('CHI2RED_INI', chi2red_ok),
                              ('CHI2RED_INI2', chi2red_ok2)):
            if verdict is not None:
                verdicts[name] = verdict

        physical = params * scales
        reported_z_prior = None
        if photo:
            # what zPRIOR/zPRIORERR report (snlc_fit.F90:8561)
            reported_z_prior = (z_priors.z_init, jnp.maximum(
                0.001, z_priors.z_prior_err * cfg.photoz.prior_zerrscale))

        return BatchFitResult(
            snids=batch.snids,
            params=self.unstack(physical, ref),
            param_errors={n: errors[:, i]
                          for i, n in enumerate(self.param_names)},
            minos_errors=(None if profiles is None else
                {n: profiles[0][:, i] for i, n in enumerate(self.param_names)}),
            minos_status=(None if profiles is None else
                {n: profiles[1][:, i] for i, n in enumerate(self.param_names)}),
            param_covariances=covariances,
            # FITCHI2 includes the prior (snana.F90:27794)
            chi_squared=chi2,
            ndof=self._ndof(mask),
            hessian_cond=hessian_cond,
            edm=edm,
            hessian_positive=hessian_positive,
            observation_mask=mask,
            prior_chi2=prior_chi2,
            z_prior=reported_z_prior,
            # zPHOT last (TABLE_VARLIST_COV, snlc_fit.F90:19399)
            cov_output_params=covariance_output_params(self.fit_parameters),
            iter1_verdicts=verdicts,
        )

    def _grid_start(self, batch, params, scales, z_priors, limits=None):
        """The photo-z coarse grid's start (INIPAR_PHOTOZ_COURSEGRID),
        PHOTOZ_GRID_BATCH events at a time, a short last slice padded by
        repeating its own events. SKIP_PHOTOZ_INIT leaves an event without
        a host at its midpoint (snlc_fit.F90:8523)."""
        found = []
        for rows, s in padded_batches(
                np.arange(batch.n_sne),
                min(self.config.photoz.photoz_grid_batch, batch.n_sne)):
            found.append(self._run_grid(
                self._model_input, batch.photometry, params[s], scales[s],
                batch.mjd_offsets[s], batch.filter_indices[s],
                batch.observation_mask[s], batch.fluxcals[s],
                batch.fluxcalerrs[s], jax.tree.map(lambda a: a[s], z_priors),
                batch.mw_att_obs[s], None if limits is None else tuple(a[s] for a in limits))[:len(rows)])
        grid = jnp.concatenate(found)
        if self.config.photoz.skip_photoz_init:
            grid = jnp.where(z_priors.host_ok[:, None], grid, params)
        return grid

    def _fit_rounds(self, batch, params, scales, lo, hi, mask, z_priors,
                    z_weights, bw):
        """SNANA's fit iterations over one batch: round 1 on live weights,
        then rounds 2..NFIT_ITERATION on weights frozen at each round's start.

        Between round 1 and round 2 the epochs FITWIN_TREST leaves out at the
        fitted (or data) redshift and peak go, and so do the epochs the
        rejection tests drop. Under photo-z, round 1 drops the bands outside
        PHOTOZ_ITER1_LAMRANGE, is repeated on the band set the fitted redshift
        supports (PHOTOZ_ANA, snlc_fit.F90:3350-3418), and every round but
        the last drops a band the fitted redshift no longer supports.

        Returns the parameters, their chi2 and its prior part at the fitted
        redshift, the epochs the fit used, the round-1 window verdicts, and
        the errors, covariances, Hessian condition numbers and EDM on what
        the last round minimised, as MINUIT's are on its FCN.
        """
        cfg = self.config
        photo = cfg.photoz.dofit_photoz
        photometry = batch.photometry
        lam = photometry['filter_mean_wavelengths']
        mjd, filter_indices = batch.mjd_offsets, batch.filter_indices
        fluxcals = batch.fluxcals
        z_data, mw_attenuation = batch.redshifts, batch.mw_att_obs

        def fudged(maxfrac, p, m, iteration):
            return self.fudged_fluxcalerrs(
                lam, maxfrac, p[:, T0] * scales[:, T0],
                self._redshifts(p, scales, z_data), mjd, filter_indices, m, fluxcals,
                batch.fluxcalerrs, iteration)

        def round_(strategy, p, m, fluxcalerrs, prior_data, weights):
            p, chi2 = self._run_round(
                self._model_input, strategy=strategy, photometry=photometry, z_weights=z_weights,
                start=p, scales=scales, lo=lo, hi=hi, mjd=mjd,
                filter_indices=filter_indices, mask=m, fluxcals=fluxcals,
                fluxcalerrs=fluxcalerrs, z=z_data, bw=bw, prior_data=prior_data,
                weights=weights, z_priors=z_priors, mw_attenuation=mw_attenuation)
            return p, chi2, dict(strategy=strategy, mask=m, fluxcalerrs=fluxcalerrs,
                                 prior_data=prior_data, weights=weights)

        # each pass's t0 prior is centred on its INIVAL(IPAR_PEAKMJD), which
        # after the first iteration is the last one's fit (snlc_fit.F90:4840,
        # :7913-7921); the distance prior is at the redshift prior's mean
        # under photo-z until a redshift is fitted
        def prior_at(p, mu):
            return priors.per_sn(p[:, T0] * scales[:, T0], mu, batch.reference_mjds)

        mu_round1 = cfg.fit.distance_modulus(
            z_priors.z_prior if photo else z_data)
        prior_data = prior_at(params, mu_round1)
        maxfrac_r1 = (cfg.fit.fudgeall_iter1_maxfrac
                      if cfg.fit.fudgeall_iter1_maxfrac > 0
                      else cfg.fit.fudgeall_maxfrac)

        mask_r1 = ((mask & photoz.iter1_band_mask(cfg, lam, filter_indices)) if photo
                   else mask)
        round_start = params
        fluxcalerrs = fudged(maxfrac_r1, params, mask_r1, 1)
        params, chi2, last_likelihood = round_(
            'live', params, mask_r1, fluxcalerrs, prior_data, None)
        if photo:
            kept, repeat = photoz.round1_band_set(
                cfg, self.restlambda_usefit, lam, filter_indices, batch.observation_mask,
                params * scales, z_priors)
            if bool(jnp.any(repeat)):
                # round 1 again on the band set (snana.F90:9514-9516), the
                # other events keeping the errors they were fitted with
                mask_r1 = jnp.where(repeat[:, None], kept, mask_r1)
                fluxcalerrs = jnp.where(repeat[:, None],
                                fudged(maxfrac_r1, params, mask_r1, 1), fluxcalerrs)
                round_start = jnp.where(repeat[:, None], params, round_start)
                prior_data = jnp.where(repeat[:, None],
                                       prior_at(params, mu_round1), prior_data)
                again, chi2_again, last_likelihood = round_(
                    'live', round_start, mask_r1, fluxcalerrs, prior_data, None)
                params = jnp.where(repeat[:, None], again, params)
                chi2 = jnp.where(repeat, chi2_again, chi2)
            mask = kept & ~photoz.dropped_epochs(
                cfg, self.restlambda_usefit, lam, filter_indices, params * scales,
                z_priors, repeat)
        verdicts = self.iter1_verdicts(params, scales, chi2, mask_r1,
                                       prior_data, z_priors)

        # FITWIN_TREST and the epoch rejects, from round 1's last pass
        # (snlc_fit.F90:4481-4510, :10885, :10967)
        z = self._redshifts(params, scales, z_data)
        mask = mask & self.fitwin_trest_ok(
            filter_indices, mjd, params[:, T0] * scales[:, T0], z)
        mask = self.reject_epochs(
            self._model_input, photometry, mask, params * scales, mjd, filter_indices, fluxcals, batch.fluxcalerrs,
            batch.reference_mjds, z,
            self.events_band_weights(photometry, z, mw_attenuation) if photo else bw)

        # USE_DIAG_ONLY reads the frozen weights only for the sigma term
        frozen = (self.rounds2_strategy != 'live_reweight'
                  or bool(cfg.fit.opt_chi2_sigma))
        # a spec-z fit's colour-dispersion fraction holds over its rounds 2+;
        # a photo-z fit's follows its fitted redshift
        kcor_frac = (self._colour_fraction(self.model, photometry, filter_indices, z_data, mask)
                     if frozen and not photo else None)
        for round_index in range(cfg.fit.n_fit_iterations - 1):
            prior_data = prior_at(params, cfg.fit.distance_modulus(z)
                                  if photo else mu_round1)
            # MAXFRAC at every iteration (snlc_fit.F90:4327)
            fluxcalerrs = fudged(cfg.fit.fudgeall_maxfrac, params, mask,
                         round_index + 2)
            weights = (self._run_weights(self._model_input, photometry, z_weights, params,
                                         scales, mjd, filter_indices, mask, fluxcals, fluxcalerrs,
                                         z_data, bw, kcor_frac, mw_attenuation,
                                         batch.mwxt_magerrs)
                       if frozen else None)
            round_start = params
            params, chi2, last_likelihood = round_(self.rounds2_strategy, params, mask,
                                        fluxcalerrs, prior_data, weights)
            z = self._redshifts(params, scales, z_data)
            if photo and round_index < cfg.fit.n_fit_iterations - 2:
                mask = mask & ~photoz.dropped_epochs(
                    cfg, self.restlambda_usefit, lam, filter_indices, params * scales,
                    z_priors, True)

        errors = self._run_errors(
            self._model_input, **last_likelihood, photometry=photometry, z_weights=z_weights,
            lo=lo, hi=hi,
            params=params, start=round_start, scales=scales, mjd=mjd,
            filter_indices=filter_indices, fluxcals=fluxcals, z=z_data, bw=bw,
            z_priors=z_priors, mw_attenuation=mw_attenuation)
        if cfg.fit.minos is not None:
            params, chi2 = errors[-2:]
            errors = errors[:-2]
        prior_chi2 = self._prior_chi2(
            params * scales, last_likelihood['prior_data'], z_priors)
        return params, chi2, prior_chi2, mask, verdicts, errors

    def _z_prior_chi2(self, z, z_prior):
        return fit_kernels._z_prior_chi2(self._kernel_settings, z, z_prior)

    def _bands(self, photometry, z_weights, z_data, bw, mw_attenuation):
        return fit_kernels._bands(
            self._kernel_settings, photometry, z_weights, z_data, bw, mw_attenuation)

    def _sn_chi2(self, model, strategy, photometry, p_scaled, start, scales, mjd, filter_indices,
                 mask, fluxcals, fluxcalerrs, bands, prior_data, weights, z_prior):
        return fit_kernels._sn_chi2(
            self._kernel_settings, model, strategy, photometry, p_scaled, start, scales, mjd,
            filter_indices, mask, fluxcals, fluxcalerrs, bands, prior_data, weights, z_prior)

    def _colour_fraction(self, model, photometry, filter_indices, z, mask):
        return fit_kernels._colour_fraction(
            self._kernel_settings, model, photometry, filter_indices, z, mask)

    def iter1_verdicts(self, params, scales, chi2, mask, prior_data, z_priors):
        """Which round-1 window accepted each SN: ``{window: ok_mask}`` over
        the windows this fit configured, empty when it configured none.

        SNANA abandons the event mid-fit (FITPAR_ANA, snlc_fit.F90:2139-2155);
        a vmapped batch records the verdict instead and the caller drops those
        events. The window is named as the input names it, so what rejected
        an event survives into the results; SNANA's own numbering for it
        belongs to the tables that report it.

        PROB is FITPROBCHI2_STORE(1): the probability of the minimised chi2,
        prior included, on NDOF + NDOF_PRIOR (:func:`prob_ndof`,
        MNFIT_STOREPAR, snana.F90:28085-28125).
        """
        verdicts = {}
        physical = params * scales
        windows = self.config.fit
        if windows.fitwin_prob_iter1:
            lo, hi = windows.fitwin_prob_iter1
            prob = self.model.compute_fitprob(chi2, prob_ndof(
                self._ndof(mask),
                self._prior_chi2(physical, prior_data, z_priors)))
            verdicts['PROB'] = (prob >= lo) & (prob <= hi)
        for key, (lo, hi) in windows.fitwin_iter1.items():
            val = physical[:, self.iter1_param_index(key)]
            verdicts[key] = (val >= lo) & (val <= hi)
        return verdicts

    def _prior_chi2(self, physical, prior_data, z_priors):
        """Every event's prior chi2 at the physical parameters: the model
        parameters' priors, and under photo-z the redshift's
        (CHI2PRIOR(IPAR_zPHOT), snlc_fit.F90:4903)."""
        chi2 = vmap(lambda p, d: priors.prior_chi2(p, self.priors, d))(
            physical[:, :self.n_model_params], prior_data)
        if self.config.photoz.dofit_photoz:
            chi2 = chi2 + vmap(self._z_prior_chi2)(physical[:, -1], z_priors)
        return chi2

    def _ndof(self, mask):
        """Every event's NDOF as SNANA reports it: the fitted epochs less
        the floated parameters, zPHOT among them under photo-z
        (XNFITDOF(0), snlc_fit.F90:13132-13140)."""
        return (jnp.sum(mask.astype(jnp.int32), axis=1)
                - ndof_n_params(self.model, self.held_params)
                - (1 if self.config.photoz.dofit_photoz
                   and 'zPHOT' not in self.held_params else 0))

    def evaluate_flux(self, params, mjds, redshifts, filter_indices, masks, *,
                      reference_mjds, mw_att_obs, photometry):
        """Model flux and its error, ``(model_flux, model_flux_err)`` each
        [n_sne, max_obs], for the tables that report them.

        ``params`` holds one [n_sne] array per fitted parameter, PKMJD in
        absolute MJD; ``mjds`` are absolute MJDs, taken as offsets from
        ``reference_mjds`` here.
        """
        physical = self.stack(params, reference_mjds)[:, :self.n_model_params]
        bw, bw_err = self.events_band_weights(photometry, redshifts,
                                              mw_att_obs)
        return self._run_flux(self._model_input, photometry, physical,
                              mjds - reference_mjds[:, None], filter_indices,
                              masks, bw, bw_err, redshifts,
                              band_window=fit_band_window(self.model, photometry))
