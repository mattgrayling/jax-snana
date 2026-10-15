"""The interface the light-curve fitter uses.

A model takes this role by listing :class:`FittedModel` among its bases.
The fitter checks for the role once, when it takes the model, and derives
everything else from the fit-parameter spec the model declares.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Dict, Optional, Tuple

import jax.numpy as jnp
import numpy as np
from jax import vmap
from scipy.stats import chi2 as chi2_dist

from jax_snlc_core.utils.error_handling import SNANAError

from .base import FIT_ONLY

# DOFIT_PHOTOZ redshift, model-independent (snlc_fit.F90:2449)
PHOTOZ_PARAMETER = {'zPHOT': {'error_bounds': (1.0e-4, 2.0)}}

# zPHOT ERRMAX_BAD, plain and with a best-z prior (snlc_fit.F90:1125, :1128, :7128)
PHOTOZ_ERRMAX_BAD = 0.0005
PHOTOZ_ERRMAX_BAD_BESTZ = 1.0e-8
PHOTOZ_INISTP_DEFAULT = 0.05

# FITINI_PKMJDPRIOR grid and flare (snlc_fit.F90:11783, :11809)
_PKMJD_PRIOR_NBIN = 5000
PKMJD_PRIOR_FLARE = 28.0
PKMJD_PRIOR_GRID = (-30.0 + 60.0 / _PKMJD_PRIOR_NBIN, 30.0)

# Per-SN prior coefficients, named in a declaration and supplied by the fitter
T0_CENTRE = 'T0_CENTRE'
MU_REF = 'MU_REF'


def flux_error_from_fracerr(flux, frac):
    """Flux error implied by a fractional error held in nats."""
    return flux * (1.0 - jnp.exp(-frac))


@dataclass(init=False, eq=False, repr=False, kw_only=True)
class FittedModel(ABC):
    """Defines the fitted parameters, model uncertainties and FITRES columns.
    Uses the SED evaluation methods provided by :class:`SNModel`."""

    # SIM_ columns carried beyond the fit spec's sim_truth
    SIM_TRUTH_COLUMNS: ClassVar[Tuple[str, ...]] = ()

    # SNTABLE renames ((column, head_col), ...); None keeps the HEAD names
    SNANA_TABLE_SIM_COLUMNS: ClassVar[Optional[Tuple[Tuple[str, str], ...]]] = None

    #: Width in days of the Gaussian prior on the time of peak.
    prior_mjdsig: Annotated[float, FIT_ONLY] = 20.0

    # phases the pre-fit flux ratio is taken over (snlc_fit.F90:9401)
    grid_trest_range = (-10.0, 40.0)

    # whether the covariance carries the Woodbury off-diagonal term
    has_offdiag_covariance = False

    # glob patterns, relative to the model directory, of the metadata files
    # carrying DOCANA and calibration shifts
    METADATA_FILES: ClassVar[Tuple[str, ...]] = ()

    @abstractmethod
    def get_fit_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Declare this model's fit parameters once, in fit-vector order.

        Insertion order is the fit-vector order, which fixes the column order
        of every parameter array and the covariance layout.

        Return ``{name: settings}``, with one dictionary per parameter.
        The core settings are:

        - ``bounds`` (lo, hi), or ``bounds_relative_to_init`` (lo_factor,
          hi_factor) when the range depends on the starting value, or
          ``bounds_about_peak_estimate`` (lo, hi) about the estimated peak:
          exactly one of these three forms is required.
        - ``scale``: positive scale used to convert the physical parameter
          to the optimiser's scaled coordinates.
        - ``inistp`` or ``inistp_relative_to_init``: the initial step; 0 holds
          the parameter fixed.
        - ``init``: starting value in physical units; the distance parameter
          instead starts at the reference distance.
        - ``priors``: tuple of ``(form, coefficients)`` pairs, for example
          ``(('gauss', {'centre': 0.0, 'sigma': 1.0}),)``.
        - ``sim_truth``: simulated-truth column used for initialisation,
          fixed-truth fits and output comparisons.
        - ``snana_keys``: the COLOR/SHAPE names the input file maps to this
          parameter, such as ``('SHAPE', 'LUMIPAR')``.

        Optional initialisation, uncertainty-selection and output settings
        are listed under :ref:`advanced-fit-spec`.

        The declaration can depend on the loaded model. Numerical settings
        must be ordinary Python numbers, rather than traced arrays.
        Include ``super().get_fit_parameters()`` first to retain ``PKMJD``
        and its shared peak-time prior. ``zPHOT`` is reserved for redshift
        fitting.
        """
        return {'PKMJD': {
            # INIBND: the peak estimate +-100 d (snlc_fit.F90:7981)
            'bounds_about_peak_estimate': (-100.0, 100.0), 'scale': 1.0,
            'sim_truth': 'SIM_PEAKMJD', 'inistp': 2.0,
            'errmax_bad': 1.0e-4, 'error_bounds': (1.0e-8, 10.0),
            'priors': (('gauss_flare',
                        {'centre': T0_CENTRE, 'sigma': self.prior_mjdsig,
                         'grid': PKMJD_PRIOR_GRID,
                         'flare': PKMJD_PRIOR_FLARE}),),
            # offsets, SNANA's default (snlc_fit.F90:6812); other axes absolute
            'grid_search': (-4.0, -2.0, 0.0, 2.0, 4.0),
            'photoz_grid_search': (-4.0, -2.0, 0.0, 2.0, 4.0)}}

    def sed_params_from_fit(self, params_arr, redshift):
        """The fit vector as the dict :meth:`SNModel.evaluate` takes: t0
        leads the vector, then each following element contains a fitted parameter, under
        the name :meth:`SNModel.generate_sed` reads it by. A model whose
        vector holds array-valued blocks or an encoded amplitude overrides
        this and calls it first."""
        names = list(self.get_fit_parameters())[1:]
        unread = [name for name in names if name not in self.sed_param_names()]
        if unread:
            raise SNANAError(
                f"{type(self).__name__} fits {unread}, which its generate_sed "
                f"does not read (sed_param_names: {self.sed_param_names()}); "
                f"name the fit parameters as the SED reads them, or override "
                f"sed_params_from_fit.", function_name="sed_params_from_fit")
        sed_params = {name: params_arr[1 + i] for i, name in enumerate(names)}
        sed_params['t0'] = params_arr[0]
        sed_params['redshift'] = redshift
        return sed_params

    def compute_raw_fracerr(self, sed_params, mjds, filter_indices, mask,
                             band_weights, static_data, **kwargs):
        """The model's fractional flux error per observation, in nats;
        :func:`flux_error_from_fracerr` makes it absolute.

        Default: zeros. SALT supplies its phase- and wavelength-dependent
        model uncertainty.

        Optional kwargs: ``error_redshift`` reads the surface at a redshift
        other than the trial one; ``band_weights_err`` supplies the
        denominators whose band integral carries no MW extinction;
        ``error_overrides`` holds parameters at values other than the trial
        ones.
        """
        return jnp.zeros_like(mjds)

    def color_dispersion_fraction(self, filter_mean_wavelengths, filter_indices,
                                  redshifts, masks):
        """The per-observation colour-dispersion fraction of the Woodbury
        off-diagonal term, for a model declaring ``has_offdiag_covariance``."""
        raise SNANAError(
            f"{type(self).__name__} declares no off-diagonal covariance.",
            function_name="color_dispersion_fraction")

    def evaluate_redshift_grid(self, axes, base_params, mu_grid, z_grid, band_weights,
                      band_weights_err, mjds, filter_indices, mask,
                      static_data, error_overrides=None):
        """Flux and model error over a grid of redshifts crossed with ``axes``.

        Returns two arrays shaped ``[n_z, *axis_lengths, n_obs]``. ``axes`` is
        ``{name: values}`` as :func:`jax_snlc_fit.photoz.coarse_grid_axes`
        built it,
        ``mu_grid`` is the distance modulus at each redshift, and
        ``base_params`` supplies whatever neither varies. Brightness comes from
        ``mu_grid`` through the model, so the caller never handles an amplitude.

        This default visits every grid point. A model whose evaluation
        factorises over the axes can override it with an array form; SALT
        does, because its band integral does not involve its shape parameter
        and so is computed once for the whole grid.
        """
        names = list(self.get_fit_parameters())
        idx = [names.index(n) for n in axes]
        lengths = [len(v) for v in axes.values()]
        mesh = jnp.meshgrid(*[jnp.asarray(v) for v in axes.values()],
                            indexing='ij')
        flat = [g.ravel() for g in mesh]
        n_points = int(np.prod(lengths))

        def one(iz, ip):
            p = base_params
            for k, i in enumerate(idx):
                p = p.at[i].set(flat[k][ip])
            # after the axes, so a brightness that depends on them sees them set
            p = self.params_at_distance(p, mu_grid[iz], names=names)
            sed_params = self.sed_params_from_fit(p, z_grid[iz])
            flux = self.evaluate(sed_params, mjds, filter_indices, mask,
                                 band_weights[iz], static_data)
            frac = self.compute_raw_fracerr(
                sed_params, mjds, filter_indices, mask, band_weights[iz],
                static_data, band_weights_err=band_weights_err[iz],
                error_overrides=error_overrides)
            return flux, jnp.where(mask, flux_error_from_fracerr(flux, frac), 0.0)

        iz, ip = [g.ravel() for g in jnp.meshgrid(
            jnp.arange(z_grid.shape[0]), jnp.arange(n_points), indexing='ij')]
        stacked = vmap(one)(iz, ip)
        shape = (z_grid.shape[0], *lengths, -1)
        return tuple(s.reshape(shape) for s in stacked)

    def compute_fitprob(self, chi_squared, ndof):
        """Chi-square survival probability for the supplied degrees of freedom."""
        ndof_arr = np.asarray(ndof)
        chi2_val = np.asarray(chi_squared)
        return np.where((ndof_arr > 0) & (chi2_val > 0),
                        chi2_dist.sf(chi2_val, ndof_arr), 1.0)

    def fitres_param_columns(self, param_names, physical_only=False):
        """Ordered (parameter, FITRES column) pairs: the parameter columns the
        fit writes, known before any fit runs. A derived column (SALT's mB)
        names the parameter it is computed from; a renamed output column
        names its corresponding fitted parameter. ``physical_only`` drops nuisance
        parameters (BayeSN's epsilon) for a SALT2mu-readable table.
        """
        return [(name, name) for name in param_names]

    def get_fitres_param_columns(self, params, param_errors, redshifts, mu_ref,
                                 physical_only=False):
        """Ordered FITRES parameter columns: [(column, values, errors)], one per
        pair of :meth:`fitres_param_columns`."""
        return [(column, params[name], param_errors[name])
                for name, column in self.fitres_param_columns(list(params),
                                                              physical_only)]

    def get_fitres_derived_columns(self, params):
        """Error-less derived FITRES columns after FITPROB: [(column, values)].

        Default: none. SALT overrides with RADIUS_POP.
        """
        return []
