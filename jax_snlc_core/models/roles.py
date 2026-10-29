"""The three roles a model plays: :class:`FittedModel` for the light-curve
fit, :class:`SimulatedModel` for the simulator and :class:`StandardisedModel`
for BBC (SALT2mu).

A model takes a role by listing its class among its bases. Each program checks
for its role once and uses only what the role declares. BBC works on a FITRES
table and never builds a model, so its role is class-level: declarations and
classmethods, resolved from MODEL_LCFIT by
:func:`jax_snlc_core.models.standardised_model`, with the model's BBC input
keys as ``BBC_ONLY`` attributes that reach the classmethods as the ``inputs``
dict the BBC parser collects.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Annotated, Any, ClassVar, Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
from jax import vmap
from scipy.stats import chi2 as chi2_dist

from jax_snlc_core.logging import SNANAError

from .input_keys import FIT_ONLY, SIM_ONLY


# FITINI_PKMJDPRIOR grid and flare (snlc_fit.F90:388, :11784-11787)
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


@dataclass(init=False, eq=False, repr=False, kw_only=True)
class SimulatedModel(ABC):
    """Per-event sampling, the truth columns, and the declarations the
    simulator's parser, builder and writers read. Reads ``self.config``,
    which :class:`SNModel` sets."""

    # an event count the model dictates (SIMSED_GRIDONLY: SEQUENTIAL)
    forced_ngen = None

    # Model-specific switches may disable training calibration corrections.
    enable_magshift = True
    enable_waveshift = True

    #: Scale factor for the maximum permitted generated flux. Models of
    #: intrinsically brighter transients can override this value.
    crazyflux_factor = 1.0

    #: Counts events in the non-Ia entry of DNDZ_SCALE and NGEN_SCALE_NON1A, so
    #: the non-Ia rate scalings apply to them rather than the Ia ones.
    uses_nonia_rates = False

    # events drawn from a NON1A template catalogue (NON1A_MODELFLAG,
    # snlc_sim.c:4806)
    non1a_templates = False

    #: Whether MJD_EXPLODE can be used with this model. Under 'none' it
    #: aborts, 'optmask' needs OPTMASK_T0SHIFT_EXPLODE >= 1 to re-anchor the
    #: template day grid, and 'native' is already anchored on the explosion.
    #: Any other value aborts every simulation with the model.
    mjd_explode_anchor = 'none'

    #: The parameters the simulation input may give a population, by name.
    #: ``GENPEAK_<name>``, ``GENSIGMA_<name>`` and ``GENRANGE_<name>`` set one,
    #: whatever the case of the name, and the drawn value is stored under the
    #: name. A model that reads its parameters from its own input file returns
    #: them from :meth:`structured_input` under this key instead.
    population_parameters = ()

    #: Other spellings the input may use for a population parameter, as
    #: ``{name: (spelling, ...)}``. SALT reads SNANA's ``SALT2x1`` as ``x1``.
    population_aliases = {}

    #: SNANA's values for the population keys an input leaves out, as
    #: ``{name: {'peak': ..., 'sigma': (lo, hi), 'range': (lo, hi)}}``.
    population_defaults = {}

    # (canonical, *alternates) key spellings, merged into the alias table
    INPUT_KEY_ALIASES: ClassVar[Tuple[Tuple[str, ...], ...]] = ()

    # keys SNANA takes from a sim-input file only (parse_input_NON1ASED, snlc_sim.c:3480-3485)
    FILE_ONLY_INPUT_KEYS: ClassVar[Tuple[str, ...]] = ()

    #: The model's own columns appended to every SIMGEN_DUMP and, where
    #: ``fits_param_block`` names one, listed in the FITS header, spelled
    #: as :meth:`get_snana_header_params` names them; empty for a model whose
    #: truth columns are asked for by name alone.
    dump_columns = ()

    #: SIMGEN_DUMP names for the model's truth columns, as (name, column) pairs
    #: whose column is a :meth:`get_snana_header_params` key. Each name
    #: makes that column available in the simulation dump.
    SNANA_DUMP_SIM_COLUMNS: ClassVar[Tuple[Tuple[str, str], ...]] = ()

    # per-template (wl_min, wl_max), or None
    per_template_wavelength_bounds = None

    #: SNANA's numerical code for this model, written as SIM_MODEL_INDEX in
    #: every output, where the fitter's USESIM_SNIA and USESIM_NONIA selection
    #: reads it back; 0 for a model SNANA does not have.
    sim_model_index = 0

    # prefix stripped from GENMODEL for SIM_TYPE_NAME (snlc_sim.c:29450)
    sim_type_name_prefix = ''

    #: Whether WRFLAG_MODELPAR controls the model's parameter columns in the
    #: data files and the dump: True lets an input set to 0 drop them, as
    #: SNANA does for SIMSED and LCLIB, whose column sets vary with the model
    #: file, so that a version merging several models keeps one schema.
    #: False writes them whatever the key says.
    writes_model_params = False

    # The primary FITS header names those columns under this prefix, as
    # SNANA does for SIMSED and LCLIB alone (sntools_dataformat_fits.c:1316-1346).
    fits_param_block = ''

    #: File of double-exponential decay parameters continuing the SED beyond
    #: the last phase the model covers, in place of the model's own late-time
    #: behaviour. Read by the simulator only.
    genmodel_extrap_latetime: Annotated[str, SIM_ONLY] = field(
        default="", metadata={"static": True})

    #: Option bits whose meaning each model defines. Read by the simulator
    #: only.
    genmodel_mskopt: Annotated[int, SIM_ONLY] = field(
        default=0, metadata={"static": True})

    #: Directory searched, by file name, for a template SED file that is not
    #: in the model directory. The empty default searches the model directory
    #: only.
    path_non1ased: Annotated[str, SIM_ONLY] = field(default="", metadata={"static": True})

    #: Minimum, in magnitudes per day, of the decay rate fitted to a
    #: template's last phase samples and used to continue its flux past its
    #: end. The default 0.0 sets no minimum.
    minslope_extrapmag_late: Annotated[float, SIM_ONLY] = field(
        default=0.0, metadata={"static": True})

    #: Select the template's phase origin. 1 takes the day before the flux
    #: first exceeds one per cent of the template's peak flux, 2 the same
    #: at 0.1 per cent, and 32 the bolometric peak plus 32 days.
    #: The default 0 keeps the file's own
    #: day zero.
    optmask_t0shift_explode: Annotated[int, SIM_ONLY] = field(
        default=0, metadata={"static": True})

    # (name, aliases) for each population parameter this instance's input
    # declares, which the builder sets; a field so that compiled code
    # rebuilding the model keeps it
    configured_population: tuple = field(
        default=(), metadata={"static": True, "input": False})

    @classmethod
    def structured_input(cls, raw) -> Dict[str, Any]:
        """Read model-specific tables or groups of input settings.

        ``raw`` is a mapping of parsed input keys to values. Use
        ``raw.all('KEY')`` to read every occurrence of a repeated key in
        input order, or pass several spellings to combine their rows.
        Multi-value rows are lists. Ordinary mapping access reads the last
        value supplied for a key.

        Return a dictionary of model-specific settings. Its entries are
        included in the ``config`` passed to model loading and in the
        ``snana_config`` passed to ``configure_generation``. The default
        returns an empty dictionary.
        """
        return {}

    def configure_generation(self, snana_config: Dict, n_simulated: int,
                             frac_pec1a: float) -> None:
        """Configure parameter sampling from distributions or GENPDF files,
        template selection and WGTMAP weights. For models that assign an
        event count to each template, use the total event count and the
        fraction of peculiar type Ia supernovae to set those counts.

        Called by the simulator only, on a model that is already built.
        Overrides must not mutate template wavelength or phase state.

        Default: no-op.
        """

    def derived_parameters(self, params, keys_for, **kwargs):
        """The model's per-event parameters beyond its population draws:
        values it computes from them, and draws of its own.

        Called once per batch by the simulator's model draw (``draw_model``),
        after it has drawn every population parameter the input gives a
        distribution, in declaration order.

        - ``params``: the batch so far, by name: the redshift, the distance
          modulus ``mu``, each event's ``sn_global_idx`` and the population
          draws. A population parameter the input gives no distribution is
          absent: the model may supply it, and reading it stops the run with
          the keys to give.
        - ``keys_for``: ``(name) -> (n_sne, 2)``, per-SN RNG keys for a named
          random quantity, for the model's own draws (a NON1ASED template
          pick, BayeSN's epsilon).
        - ``template_budget``, by keyword: the per-template event quota of
          this batch, for a model that limits each template's events.

        Returns ``{name: array}``, merged into the event's parameters; with
        the draws it holds every population parameter, which
        :meth:`sn_variables` and the header read, plus whatever
        :meth:`SNModel.generate_sed` reads. Default: nothing.
        """
        return {}

    def get_snana_header_params(self, parameters: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """The model's per-event truth as SNANA's ``SIM_`` columns: the sampled
        params in (by name, numpy arrays of shape (n_sne,)), the FITS header
        columns out. The default writes each population parameter as
        ``SIM_<NAME>``, its name in upper case; a model with other truth to
        record, or whose columns SNANA spells otherwise, overrides it.

        A model that also fits names the column the fitter compares each
        parameter to as ``sim_truth`` on its fit spec, and the fitter copies
        those columns to the FITRES itself; the optional declarations are
        :attr:`FittedModel.SIM_TRUTH_COLUMNS` (columns beyond the spec's),
        :attr:`FittedModel.SNANA_TABLE_SIM_COLUMNS` (SNTABLE renames) and
        :meth:`replay_params_from_dump` (the DUMP columns a replay rebuilds
        params from).
        """
        return {f'SIM_{name.upper()}': parameters[name]
                for name, _ in self.configured_population}

    def pec1a_flags(self, n_sne, global_start_idx, template_budget=None):
        """Per-event PEC1A flag, which a model declaring non1a_templates
        supplies from its catalogue."""
        raise SNANAError(
            f"{type(self).__name__} has no PEC1A templates to flag.",
            function_name="pec1a_flags")

    def apply_host_snpar_overrides(self, params, host_values):
        """USESNPAR: the host's values, by parameter name, replace the drawn
        ones; a model with derived parameters overrides to keep them
        consistent."""
        params.update(host_values)
        return params

    def drawn_parameters(self, populations) -> Tuple[str, ...]:
        """The population parameters the simulator draws, in declaration
        order: those the input gives a distribution, ``populations`` being
        the run's records by name. A model that draws some of them itself
        overrides this."""
        return tuple(name for name, _ in self.configured_population
                     if name in populations)

    def generation_quotas(self, target: int) -> np.ndarray:
        """Per-template event quotas for NGEN_LC accounting; one open quota
        for a model without template budgets."""
        return np.array([target], dtype=np.int64)

    def text_sim_header_lines(self, hcol, params, i) -> List[str]:
        """Extra ``SIM_`` header lines the TEXT writer emits for this model."""
        return []

    def replay_params_from_dump(self, dump_columns, head_columns, *, mu,
                                distributions):
        """Map recorded SIMGEN_DUMP/HEAD columns onto this model's per-event
        sampling parameters, as ``(params, invalid)`` where ``invalid`` (or
        None) marks events with missing recorded parameter values. A dump
        column written under any name :attr:`SNANA_DUMP_SIM_COLUMNS` declares
        arrives under its column as well."""
        raise SNANAError(
            f"Replay override not implemented for {type(self).__name__}.",
            function_name="replay_params_from_dump")

    def sn_variables(self, params: Dict) -> Dict:
        """Per-SN variables a SEARCHEFF map may cut on or a WGTMAP bin on.
        Every population parameter is one, under its name; a model with
        quantities derived from its parameters rather than sampled adds
        them."""
        return {name: jnp.asarray(params[name], dtype=jnp.float32)
                for name, _ in self.configured_population}


def fitres_aliases_for(names) -> Dict[str, Tuple[str, ...]]:
    """The fitter's own FITRES spellings for LC parameters that are its fit
    parameters: ``<name>``, ``<name>ERR`` and ``COV_<i>_<j>``, the latter
    written in fit-vector order and accepted in either."""
    aliases = {}
    for i, a in enumerate(names):
        aliases[a] = (a,)
        aliases[f"{a}ERR"] = (f"{a}ERR",)
        for b in names[i + 1:]:
            aliases[f"COV_{a}_{b}"] = (f"COV_{a}_{b}", f"COV_{b}_{a}")
    return aliases


class StandardisedModel(ABC):
    """The distance a model's fitted parameters standardise to, the nuisance
    parameters of that standardisation, and how BBC reads the model's FITRES
    columns and bins its biasCor.

    Declarations are plain class attributes (an annotated lowercase attribute
    would be an input key). A model playing the role declares:

    ``m0_reference``
        M0 zero-point, subtracted from ``compute_mu`` to form muFit.
    ``lcpar_names``
        The LC-parameter vector, in order; the first carries sigma_int.
    ``fitres_aliases``
        Each raw input BBC reads -> its FITRES column candidates, first
        present wins, mirroring SNANA's "DLMAG:F MU_LCFIT:F" fallback. A
        model that declares none gets :func:`fitres_aliases_for` its
        ``lcpar_names``, which the default :meth:`read_lcpar` reads.
    ``nuisance_spec``
        Fitted nuisance parameters as name -> (default, lower bound, upper
        bound, parameter number). The number identifies the corresponding
        p<N>/u<N> input keys.
    ``biascor_binning``
        {parameter: (min, max, default_binsize)} for any number of model
        coordinates. Redshift and host mass are supplied by BBC separately.
    ``mucovscale_bins``, ``ccprior_bins``
        Independent {parameter: bin_count} maps. Their ranges are the model's
        default cuts, overridden by named BBC parameter windows. Coordinates
        may be auxiliary quantities outside ``lcpar_names``.
    ``gammadm_coeff``
        The ``biascor_coeffs`` name whose axis midpoint is GAMMADM_OFFSET,
        the value :meth:`compute_mu` subtracts from the mass step. None for a model with no mass step.

    A model whose fitted distance is already a distance modulus declares
    ``m0_reference = 0.0`` and its distance column alone in ``lcpar_names``
    (BayeSN reads ``MU`` from the FITRES column ``DLMAG`` or ``MU_LCFIT``
    through ``fitres_aliases``). It declares an empty ``nuisance_spec``,
    since nothing is fitted globally, and a ``compute_mu`` returning that
    column. It keeps the default ``chi2grid_axes``, so the scan SNANA writes
    for every model comes out flat, and gives ``biascor_binning`` the
    parameters its map is binned in, THETA and AV for BayeSN.
    """

    biascor_binning = {}
    mucovscale_bins = {}
    ccprior_bins = {}

    # Input spelling -> (named configuration field, parameter).
    bbc_input_aliases = {}
    # Parameter -> (selection, bias-grid) CUTMASK bits.
    bbc_cut_bits = {}

    #: The ``biascor_coeffs`` name whose axis midpoint is the GAMMADM_OFFSET
    #: :meth:`compute_mu` subtracts from the mass step; None for a model
    #: with no mass step.
    gammadm_coeff = None

    #: The nuisance parameters the bias-correction map is also gridded in,
    #: one ``(nuisance name, SIM truth column, continuous)`` each: the map
    #: is interpolated at every event's own value of them, the truth column
    #: giving the value the biasCor simulation generated with, on a
    #: two-point grid unless ``continuous`` says the simulation drew it per
    #: event, in which case the axis is a single bin. Empty for a model
    #: standardised without global coefficients.
    biascor_coeffs = ()

    #: Nuisance values held fixed only when constructing the simulated CC prior.
    #: The data likelihood continues to use the fitted values. SALT holds its
    #: redshift and host-mass slopes at zero, as in SNANA's load_DMU_CCprior.
    ccprior_fixed_nuisance = {}

    #: The ``biascor_coeffs`` names a dust-model biasCor is read at the input's
    #: ``p<N>`` values of, rather than at its own per-event truth, since its
    #: own are intrinsic draws.
    dust_forced_coeffs = ()

    #: The SIM truth columns the bias-correction map reads beyond BBC's own,
    #: each mapped to the FITRES spellings accepted for it, first present
    #: winning; :meth:`sim_lcpar` reads them from ``tv.sim_truth``.
    biascor_truth_aliases = {}

    #: The CHI2GRID scan written after the fit, one ``(column label, nuisance
    #: name, (lo, hi))`` per axis. SNANA scans alpha and beta for every model,
    #: so a model without them keeps this default and writes a flat scan, as
    #: SALT2mu does.
    chi2grid_axes = (("alpha", "alpha0", (0.10, 0.20)),
                     ("beta", "beta0", (2.00, 4.00)))

    #: The model's shorthand cut keys, ``{key: (parameter, 'lo' | 'hi')}``,
    #: such as SALT's ``x1min`` and ``cmax``, which the BBC parser resolves
    #: to one bound of that parameter's window.
    cut_aliases = {}

    #: The selection windows ``{parameter: (lo, hi)}`` applied to the fitted
    #: parameters with no CUTWIN given, which the shorthand keys move; the
    #: same windows bound matching error-map and contaminant-prior axes.
    default_cuts = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if 'lcpar_names' in cls.__dict__ and 'fitres_aliases' not in cls.__dict__:
            cls.fitres_aliases = fitres_aliases_for(cls.lcpar_names)

    @classmethod
    @abstractmethod
    def compute_mu(cls, lcpar: Dict, nuis: Dict, host: Dict, u_flags: Dict,
                   apply_mass_step: bool = True):
        """Distance modulus from LC-fit params ``lcpar`` and nuisance params
        ``nuis``. ``host`` carries per-event ``z`` and the mass-step
        variable; ``u_flags`` the input's raw ``u<N>`` flags, whose overloaded
        values the model interprets. ``apply_mass_step=False`` on the
        CC-prior path (no host mass step)."""

    @classmethod
    def biascor_coeff_values(cls, nuis: Dict, host: Dict, u_flags: Dict) -> Tuple:
        """Per-event effective values of the ``biascor_coeffs`` axes; SNANA
        interpolates the biasCor at each event's alpha/beta/gammaDM."""
        return ()

    @classmethod
    def read_lcpar(cls, cols: Dict, inputs: Dict):
        """The LC parameters from the read columns, as ``(lcpar, covariance,
        bad_error)``: ``{name: array}`` per ``lcpar_names``, the per-event
        ``[n, n_lcpar, n_lcpar]`` LC-fit covariance (intrinsic scatter is
        added later), and the events whose LC-fit errors fail the model's
        thresholds, or None. Extra dictionary entries can supply derived map
        coordinates without adding covariance dimensions. BBC loads other
        required coordinates through ``fitres_aliases`` or exact column names.
        ``inputs`` holds the model's BBC keys.

        Default: the LC parameters are the fitter's own, read through
        :func:`fitres_aliases_for`, with no error threshold. SALT overrides
        to derive mB from x0 and apply its thresholds."""
        names = cls.lcpar_names
        n = cols[names[0]].shape[0]
        cov = jnp.zeros((n, len(names), len(names)))
        for i, a in enumerate(names):
            cov = cov.at[:, i, i].set(cols[f"{a}ERR"] ** 2)
            for j, b in enumerate(names[i + 1:], i + 1):
                cov = cov.at[:, i, j].set(cols[f"COV_{a}_{b}"])
                cov = cov.at[:, j, i].set(cols[f"COV_{a}_{b}"])
        return {a: cols[a] for a in names}, cov, None

    @classmethod
    def bbc_bad_covariance(cls, cols, covariance):
        """Events failing covariance-quality cuts, or None for no such cut.

        ``covariance`` is the repaired LC-fit covariance. These cuts remain
        active when BBC refits a selected event list with CUTWIN NONE.
        """
        return None

    @classmethod
    @abstractmethod
    def sim_lcpar(cls, tv) -> Dict:
        """Return true parameter arrays for calculating fitted-minus-true biases.

        The dictionary keys correspond to ``lcpar_names`` and its values
        are NumPy arrays for the BIASCOR sample.

        ``tv`` is the BIASCOR sample read into a
        :class:`~jax_salt2mu.tablevar.TableVar`: its ``sim_truth`` dict holds
        the ``SIM_`` columns named by :attr:`biascor_truth_aliases` and BBC's
        own, its ``lcpar`` the fitted parameters, and its ``mumodel`` the
        reference distance modulus at each event's redshift, which is the
        truth for a model whose fitted parameter is the distance itself.
        """
