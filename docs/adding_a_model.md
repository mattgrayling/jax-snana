# Adding a model

You can add your own models by creating a new class, which inherits from
the general `SNModel` class and can provide support for simulation,
light-curve fitting and/or distance standardisation. You can add this
within `jax_snlc_core.models` to add to the package, or create the class
externally and register it into jax-SNANA using the `register_model`
function. jax-SNANA has been designed so that all model-specific logic
belongs to the model class, you will not have to modify the input file
parser. All declared input settings of the model class will automatically be
parsed from input files or kwargs in Python scripts.

## An illustrative toy model

The following illustrative toy model has a blackbody spectrum at fixed
temperature. Its flux density at every wavelength follows a Gaussian in
rest-frame time about maximum, with standard deviation `width` in days. The
class declares the names used to select the model and configure its
parameter distributions:

- `GENMODEL_ALIASES = ("pulse",)` associates model names beginning with
  `pulse` (case-insensitive) with this class. For example, `GENMODEL: PULSE`
  will select it, as will paths to model directories containing `pulse`.
- `population_parameters = ("width",)` declares `width`
  as a parameter drawn for each simulated event. This enables input settings
  such as `GENPEAK_width`, `GENSIGMA_width` and `GENRANGE_width` to specify its
  distribution. The declaration does not specify the distribution this parameter
  will follow, and simply registers it to enable it to be parsed in input files.

The simulator draws `width` from the distribution specified in the input
and supplies it to `generate_sed` as `params["width"]`. It also calculates
the distance modulus `mu` from each event's redshift. This model needs no
sampling method of its own.

```python
import jax.numpy as jnp
from astropy.constants import c, h, k_B, pc

from jax_snlc_core.models import SNModel, SimulatedModel, FittedModel, register_model

class PulseModel(SNModel, SimulatedModel, FittedModel):
    GENMODEL_ALIASES = ("pulse",)
    population_parameters = ("width",)
    distance_param = "mu"

    #: Blackbody temperature in kelvin.
    pulse_temperature: float = 10000.0
    #: Emitting radius at peak in centimetres.
    pulse_radius: float = 1.e15

    def __init__(self, model_name, config, n_wave_points=500, wavelength_range=None):
        super().__init__(model_name, config, n_wave_points, wavelength_range)
        self.choose_grid()

    def get_wavelength_range(self):
        return 2000.0, 25000.0

    def phase_range(self):
        return -60.0, 120.0

    def sed_param_names(self):
        return ("mu", "width")

    def generate_sed(self, phase, params, static_data):
        wavelength = static_data["wavelength_grid"] * 1.e-8  # centimetres
        exponent = h.cgs.value * c.cgs.value / (
            wavelength * k_B.cgs.value * self.pulse_temperature)
        blackbody = 2.0 * h.cgs.value * c.cgs.value**2 / (
            wavelength**5 * jnp.expm1(exponent))
        distance = 10.0 * pc.cgs.value * 10.0**(0.2 * params["mu"])
        profile = jnp.exp(-0.5 * (phase / params["width"]) ** 2)
        return (jnp.pi * blackbody * (self.pulse_radius / distance)**2
                * profile * 1.e-8)  # flux density per Angstrom

    def get_fit_parameters(self):
        parameters = super().get_fit_parameters()
        parameters.update({
            "mu": {"bounds": (10.0, 60.0), "scale": 1.0,
                   "sim_truth": "SIM_DLMU"},
            "width": {"bounds": (2.0, 40.0), "scale": 10.0,
                      "init": 12.0, "sim_truth": "SIM_WIDTH"},
        })
        return parameters

register_model("pulse", PulseModel)
```

`generate_sed` evaluates Planck's law and converts the surface intensity
into flux density per Angstrom at the event's distance, before host or Milky
Way extinction. `PULSE_TEMPERATURE` and `PULSE_RADIUS` set the peak
luminosity, so distance and luminosity are not both free in the fit.
The shared photometry code handles redshift and integration through filters.

In this example:

- `sed_param_names` lists the per-event quantities read by `generate_sed`.
- `distance_param = "mu"` identifies the brightness parameter for fit
  initialisation and amplitude adjustments. Distance dimming is applied
  explicitly in `generate_sed`.
- `get_fit_parameters` specifies the fitted parameters, bounds and starting
  values. Calling the base method includes the time of maximum, `PKMJD`.
  The fitter initialises `mu` from the reference cosmology at the event's
  redshift, while `width` starts at 12 rest-frame days.

The default truth output records the drawn width as `SIM_WIDTH`.
`sim_truth` identifies this column for comparisons with fitted values.

The width distribution and fixed model settings would be specified as:

```text
GENMODEL: PULSE
GENPEAK_WIDTH: 12
GENSIGMA_WIDTH: 2 2
GENRANGE_WIDTH: 5 25
PULSE_TEMPERATURE: 10000
PULSE_RADIUS: 1.e15
```

## Registration

Import `register_model` from `jax_snlc_core.models` and call
`register_model(name, model_class)` before creating a simulation or fit.
The name is a lowercase identifier, such as `pulse`. A script using an
external model imports its module before constructing a simulation or fitter.

Declare `GENMODEL_ALIASES` on the class so `GENMODEL` in a simulation input
and `FITMODEL_NAME` in a fitter input file can select it. An alias is matched
against the start of the last component of the model path, ignoring case.
For example, SALT declares `('salt2', 'salt3')`, so a path ending in
`SALT3.K21` selects `SALTModel`.

## The SED

Implement the following four methods inherited from
{py:class}`SNModel <jax_snlc_core.models.base.SNModel>`:

{py:meth}`get_wavelength_range() <jax_snlc_core.models.base.SNModel.get_wavelength_range>`
: The model's own rest-frame wavelength support, as `(minimum, maximum)` in
  Angstroms, including any wavelength extension implemented by the model.

{py:meth}`phase_range() <jax_snlc_core.models.base.SNModel.phase_range>`
: The rest-frame phase support, as `(first, last)` in days, used when
  selecting phases for simulated spectra. It does not restrict
  `generate_sed` evaluation, so define any extrapolation within that method.

{py:meth}`sed_param_names() <jax_snlc_core.models.base.SNModel.sed_param_names>`
: The per-event parameter names read by `generate_sed`. Optional parameters
  may be included; they are supplied when present in the event.

{py:meth}`generate_sed(phase, params, static_data) <jax_snlc_core.models.base.SNModel.generate_sed>`
: The rest-frame spectrum at one phase, in erg per second per square
  centimetre per Angstrom, before host and Milky Way extinction. `phase` is
  a scalar and the return has shape `(n_wavelengths,)`, the length of the
  working grid. Each entry of `params` is a scalar for one event. Use
  `jax.numpy` operations so derivatives propagate through the spectrum.
  Conditional calculations depending on parameter values need JAX operations
  such as `jnp.where` or `jax.lax.cond`, rather than Python `if` statements.

### Loading and wavelength grid

The model's `__init__` method should first call `super().__init__` to apply
the configured input settings to its attributes. It can then load any
model files and prepare the wavelength grid. Its arguments are:

- `model_name` is the `GENMODEL` path of a simulation or the `FITMODEL_NAME`
  of a fit, with environment variables and a leading `~` expanded.
- `config` is a dictionary of model input settings, applied by the base
  constructor to the corresponding model attributes.
- `n_wave_points` is the number of points in the working wavelength grid.
- `wavelength_range` restricts the working grid to a rest-frame interval,
  or is `None` for the full model support.

Call `self.choose_grid()` to create `self.wavelength_grid`. This contains
`n_wave_points` evenly spaced points within the model's wavelength support,
restricted to `wavelength_range` when supplied. For a tabulated model,
resample the loaded spectra onto this grid. SALT, BayeSN, NON1ASED and
SIMSED provide examples in their `resample_to_grid` methods.

### Models based on spectral templates

Store tabulated spectra as model attributes, with shape
`(n_phases, n_wavelengths)` for each spectral time series. In `generate_sed`,
interpolate at the requested phase and at the values of any parameters
spanned by the tables.

Interpolation must depend on the fitted parameter values within
`generate_sed` for JAX to calculate their derivatives. SALT and BayeSN use
cubic Hermite interpolation to obtain continuous first derivatives.

The following helpers support tabulated spectra:

- A `SED.INFO` index, read by
  {py:func}`read_sed_info <jax_snlc_core.models.simsed_model.read_sed_info>`,
  names the parameters the grid spans and lists one SED file per line with
  that SED's parameter values. This is the SIMSED grid format.
- Each SED file has three columns: phase in days, wavelength in Angstroms
  and flux density. {py:func}`read_sed_file <jax_snlc_core.io.sed_ascii.read_sed_file>`
  reads one, and
  {py:func}`read_sed_binary <jax_snlc_core.models.simsed_model.read_sed_binary>`
  reads the whole set at once from SNANA's `SED.BINARY` cache.
- {py:func}`resample_surfaces <jax_snlc_core.models.base.resample_surfaces>`
  moves a stack of SED tables from their native wavelengths onto the working
  grid. Call it from `resample_to_grid`.
- {py:func}`uvlam_extrapflux_sedmodel <jax_snlc_core.models.base.uvlam_extrapflux_sedmodel>`
  extends a tabulated SED below its blue edge, as set by the shared
  `UVLAM_EXTRAPFLUX` key.
- {py:func}`late_time_tables <jax_snlc_core.models.base.late_time_tables>`
  fits the per-wavelength decay the simulator uses to continue a tabulated SED
  past its last phase.

### Flux scale

`generate_sed` returns spectral flux density on the rest-frame wavelength
grid, with the model's amplitude or distance normalisation already applied.
The shared photometry calculation accounts for redshift and integrates
through the filters; it applies no further luminosity-distance dimming.
For example, the toy model's `mu` sets its distance, while SALT uses its
amplitude `x0`.

`distance_param` identifies the parameter that the shared amplitude-adjustment
and fit-initialisation methods change. The defaults treat it as a distance
modulus. A model using another amplitude, such as SALT's `x0`, overrides:

- `_amplitude_shifted(value, dmu)` to return the amplitude after a shift
  of `dmu` magnitudes towards fainter fluxes.
- `_amplitude_at_distance(mu, params)` to return the amplitude at distance
  modulus `mu`, given the other model parameters.

### `static_data`

The `static_data` dictionary supplies the wavelength grid, extinction law
and filter calibration used to evaluate the model. The following entries
are available in both simulation and fitting:

`wavelength_grid`
: The working rest-frame grid in Angstroms, used by the toy model.

`host_dust_law` and `host_rv`
: The host extinction law selected by `HOST_COLORLAW`, and the
  total-to-selective ratio declared by a model that fits `av` but not `rv`.
  The extinction is applied in `SNModel.evaluate`, after `generate_sed`.

`filter_ref_fluxes`, `filter_mean_wavelengths`, `zp_fluxcal`, `fluxcal_scale_mag_offset` and `band_window`
: Calibration and band-integration quantities used by `SNModel.evaluate`;
  `generate_sed` does not need them.

`latetime_extrap`
: True in the simulator and False in the fitter, which does not continue a
  tabulated SED past its last phase.

## Input keys

Lowercase scalar annotations declare model input settings. For example,
`pulse_temperature: float = 10000.0` exposes `PULSE_TEMPERATURE` in input
files, `pulse_temperature` in Python and YAML, and `--pulse-temperature`
on the command line. No separate parser declaration is needed.

Use `int`, `float`, `bool` or `str`, or `Literal` to restrict the accepted
values. Unannotated attributes are model constants. Give model-specific
keys a distinctive prefix to avoid collisions with shared settings.

By default, both simulation and fitting accept a setting. Markers restrict
it to a particular stage. For example, SALT declares:

```python
from typing import Annotated, Literal

from jax_snlc_core.models.base import FIT_ONLY

#: Fit x0 itself (0) or its base-10 logarithm (1).
opt_salt2fit: Annotated[Literal[0, 1], FIT_ONLY] = 0
```

`FIT_ONLY` restricts a setting to fitting, `SIM_ONLY` to simulation, and
`BBC_ONLY` to standardisation. BBC settings are passed to the
`StandardisedModel` classmethods through their input dictionary.

In a fitter input file, model settings may appear in either `&SNLCINP` or
`&FITINP`. A `#:` comment above each declaration supplies its description
for the generated tables in {doc}`model_settings`.

## Simulation

Inherit `SimulatedModel` and declare `population_parameters` for simulation.
Override `derived_parameters` for additional quantities calculated or
drawn by the model, and `get_snana_header_params` to customise truth output.

{py:attr}`population_parameters <jax_snlc_core.models.simulated.SimulatedModel.population_parameters>`
: A tuple of parameter names whose distributions can be specified in the
  simulation input. The simulator draws these parameters and stores them
  under the declared names for `generate_sed`.

{py:meth}`derived_parameters(params, keys_for, **kwargs) <jax_snlc_core.models.simulated.SimulatedModel.derived_parameters>`
: Additional per-event parameters, returned as `{name: array}`. Override
  this when the model needs quantities beyond its population draws: SALT
  calculates `x0`, BayeSN draws residual scatter, and NON1ASED selects a
  template. The default returns an empty dictionary.

  `params` contains the population draws, redshift, distance modulus `mu`
  and event indices `sn_global_idx`. `keys_for(name)` supplies per-event
  random keys with shape `(n_sne, 2)` for the model's own draws, independent
  of batch size. Models with per-template quotas also receive
  `template_budget` as a keyword argument.

{py:meth}`get_snana_header_params(parameters) <jax_snlc_core.models.simulated.SimulatedModel.get_snana_header_params>`
: Simulation truth columns as `{column: array}`. By default, each
  population parameter is written as `SIM_<NAME>` in upper case. Override
  this for additional truth columns or different column names.

(model-population-parameters)=
### Population parameters

`GENPEAK_<name>`, `GENSIGMA_<name>` and `GENRANGE_<name>` specify a
parameter's asymmetric-Gaussian distribution; a `GENPDF` map can instead
provide its distribution. Names are case-insensitive. The toy model's
`width` therefore accepts `GENPEAK_WIDTH`, while its SED reads
`params["width"]`. Population distributions, correlations and evolution
are described in {doc}`supernova_populations`.

Parameters are drawn in declaration order. A `VARIATION` can depend on
parameters declared before the one being varied. If the input supplies no
distribution, the model must provide the parameter through
`derived_parameters`, as SALT does for beta with `SALT2BETA_cPOLY`.

Two optional dictionaries provide aliases and default distribution settings:

{py:attr}`population_aliases <jax_snlc_core.models.simulated.SimulatedModel.population_aliases>`
: Alternative input names, as `{name: (alias, ...)}`. For example, SALT
  declares `{"x1": ("SALT2x1",)}`, allowing `GENPEAK_SALT2x1` to configure
  `params["x1"]`.

{py:attr}`population_defaults <jax_snlc_core.models.simulated.SimulatedModel.population_defaults>`
: Default distribution settings, as
  `{name: {"peak": ..., "sigma": (lo, hi), "range": (lo, hi)}}`.
  Omitted settings are zero unless the model supplies a default.

Under `HOSTLIB_MSKOPT` bit 64, a host catalogue column with the parameter's
name can replace its drawn value.

Models whose parameter names come from model files, such as SIMSED and
Redback, return `population_parameters`, `population_aliases` and
`population_defaults` from `structured_input` instead of declaring them
on the class. SIMSED also overrides `drawn_parameters(populations)` to
handle correlated and grid-only parameters in its own sampling method.

### Other simulation declarations

Optional class attributes control rate scaling, explosion epochs and the
maximum permitted flux:

{py:attr}`crazyflux_factor <jax_snlc_core.models.simulated.SimulatedModel.crazyflux_factor>` (`float`, default `1.0`)
: Multiplies the maximum permitted generated flux. SIMSED and Redback
  use `10.0` to accommodate transients brighter than type Ia supernovae.

{py:attr}`uses_nonia_rates <jax_snlc_core.models.simulated.SimulatedModel.uses_nonia_rates>` (`bool`, default `False`)
: Set to `True` to apply the non-Ia rate scalings: the second value of
  `DNDZ_SCALE` and `NGEN_SCALE_NON1A`. NON1ASED and SIMSED use this setting.

{py:attr}`mjd_explode_anchor <jax_snlc_core.models.simulated.SimulatedModel.mjd_explode_anchor>` (`str`, default `'none'`)
: Defines support for `MJD_EXPLODE`, which specifies an explosion epoch
  rather than a peak epoch:
  - `'none'`, the default: the model does not support `MJD_EXPLODE`.
  - `'optmask'`: template phases must be shifted to the explosion with
    `OPTMASK_T0SHIFT_EXPLODE`. NON1ASED and SIMSED use this setting.
  - `'native'`: the model's phase zero is already the explosion. The Redback
    family declares this.

## Light curve fitting

Inherit `FittedModel` and implement `get_fit_parameters`.

### Fit parameters

{py:meth}`get_fit_parameters <jax_snlc_core.models.fitted.FittedModel.get_fit_parameters>`
returns `{name: spec}` in fit-vector order. Call
`super().get_fit_parameters()` first to include `PKMJD`, its bounds and
peak-time prior, then add the model's parameters as in the toy example.

For each parameter, supply exactly one bounds declaration: `bounds`,
`bounds_about_peak_estimate` or `bounds_relative_to_init`. Apart from
`PKMJD` and `distance_param`, also supply an `init` value. Use a positive
`scale` to rescale parameters for the numerical fit; its default is 1.
The fitter operates on `parameter / scale`, so the toy model's width of
12 days becomes 1.2 with `scale=10`. Bounds and starting values are
specified in physical units.

| Key | Description |
|---|---|
| `bounds` | absolute `(lo, hi)` limits |
| `bounds_about_peak_estimate` | lower and upper offsets in days from the initial peak estimate, used for `PKMJD` |
| `bounds_relative_to_init` | limits as factors on the starting value, where the range depends on it |
| `scale` | divisor used to bring the fitted parameter to order unity |
| `init` | the starting value; the distance parameter starts from the reference distance instead |

The default `sed_params_from_fit` converts the fit vector into a dictionary
using the names declared by `get_fit_parameters`, with `PKMJD` renamed to
`t0`. The remaining names must appear in `sed_param_names`. Override this
method if fitted quantities need to be transformed before spectral
evaluation, such as a logarithmic amplitude or an array of residual
coefficients.

Optional entries control priors, initialisation and output:

| Key | Description |
|---|---|
| `inistp` | the initial step in physical units; zero fixes the parameter |
| `priors` | a tuple of `(form, coefficients)` pairs, as illustrated below |
| `sim_truth` | the `SIM_` column the fit is compared against |
| `snana_keys` | the `SHAPE` and `COLOR` parameter names which the input file resolves to this parameter |
| `covariance` | `False` where no FITRES covariance column is wanted |

Options for initialisation grids, error checks and nuisance parameters are
described under {ref}`advanced-fit-spec`.

For example, a Gaussian prior on a model parameter can be declared in its
spec as:

```python
{"priors": (("gauss", {"centre": 0.0, "sigma": 1.0}),)}
```

These are the model's default priors; users can override them through
{doc}`fitting_keys`.

`sim_truth` names a column of the simulation's HEAD table, from which the
fitter reads the truth for simulated data. This can differ from the FITRES
column name. For the distance, the simulation writes `SIM_DLMU`, which is the
name a spec must give, and the FITRES writes the same quantity as
`SIM_DLMAG`.

Use Python numbers for numerical entries in the spec. Convert bounds and
starting values derived from loaded arrays to Python values during model
preparation, before fitting.

### Model uncertainty

Override `compute_raw_fracerr` to supply a model uncertainty for each
observation. Its arguments include the SED parameters, observation times
and bands; the {py:meth}`API description
<jax_snlc_core.models.fitted.FittedModel.compute_raw_fracerr>` gives the
full signature. The default returns zero.

Return uncertainties in natural-log flux units. For a returned value `s`,
the fitter calculates the absolute flux uncertainty as
`flux * (1 - exp(-s))`. For example, a fractional flux uncertainty `f`
corresponds to `s = -log(1 - f)`.

### FITRES columns

A parameter's default FITRES columns are its declared name and that name
with `ERR` appended. Covariances use `COV_<first>_<second>` in fit-vector
order. For the toy model, these are:

```text
PKMJD PKMJDERR mu muERR width widthERR
COV_PKMJD_mu COV_PKMJD_width COV_mu_width
```

A parameter with `covariance=False` or `deferred=True` has no covariance
columns. To omit or rename parameter columns, override
`fitres_param_columns`, which declares `(parameter, column)` pairs.
The default `get_fitres_param_columns` supplies their values and errors.
For derived quantities with uncertainties, also override this method to
return `(column, values, errors)` for each declared pair. SALT uses these
methods to report `mB` alongside `x0`. Covariance columns retain the fitted
parameter names.

`physical_only` allows these methods to omit nuisance parameters when
`SNTABLE_LIST` selects `nonuisance`. Derived columns without uncertainties
can be added through `get_fitres_derived_columns`.

### Parameters fixed at the reference distance

To fix the brightness at the reference cosmology's distance, declare
`"inistp": 0.0` on `distance_param`. The starting value is supplied by
`_amplitude_at_distance`; it is a distance modulus for the toy model and
an amplitude for SALT. Users can free it by supplying a positive
`INISTP_<name>`. Redback models use this fixed-distance setup by default,
as described in {doc}`model_settings`.

## Distance standardisation

Inherit `StandardisedModel` to make the model's fit results available to
BBC for standardisation and bias corrections. SALT and BayeSN
implement this interface in `salt_model.py` and `bayesn_model.py`.

Declare the following unannotated class attributes:

| Attribute | Description |
|---|---|
| `lcpar_names` | Tuple of fitted parameter names, in the order of their covariance matrix. Intrinsic scatter is applied to the first parameter. |
| `m0_reference` | Magnitude zero point subtracted from `compute_mu` to form the distance modulus. |
| `nuisance_spec` | Globally fitted coefficients as `{name: (default, lower, upper, parameter_number)}`. The number corresponds to SNANA's `pN` and `uN` settings. Use `{}` if none are needed. |
| `biascor_binning` | Bias-correction coordinates as `{name: (minimum, maximum, bin_width)}`. BBC also supplies redshift and host mass. |

Implement these two methods with `@classmethod`:

{py:meth}`compute_mu(lcpar, nuis, host, u_flags, apply_mass_step=True) <jax_snlc_core.models.standardised.StandardisedModel.compute_mu>`
: Calculates distances from fitted parameters and global coefficients.
  `lcpar` and `nuis` are dictionaries of these quantities. `host` contains
  per-event redshift and host-mass information; `u_flags` contains the
  SNANA parameter flags. Honour `apply_mass_step=False` if the model
  applies a host-mass step.

{py:meth}`sim_lcpar(tv) <jax_snlc_core.models.standardised.StandardisedModel.sim_lcpar>`
: Returns the simulated true values as `{name: array}` for each name in
  `lcpar_names`. `tv` is the BIASCOR sample, represented by
  {py:class}`TableVar <jax_salt2mu.tablevar.TableVar>`. Its `sim_truth`
  dictionary contains simulation truth, `lcpar` contains fitted values,
  and `mumodel` contains reference distance moduli at the event redshifts.

For a model that already fits a distance modulus, set `m0_reference=0.0`
and include only the distance in `lcpar_names`. `compute_mu` returns that
distance, and `sim_lcpar` returns `tv.mumodel` under the same name.
BayeSN uses this approach, with `THETA` and `AV` as bias-correction
coordinates.

### Reading fitted parameters and simulation truth

The default `read_lcpar` reads the parameters named in `lcpar_names`, their
`ERR` columns and their `COV_<first>_<second>` covariances. Declare
`fitres_aliases` to accept other spellings, as
`{quantity: (column_name, ...)}`. Override `read_lcpar` when parameters or
covariances need to be derived, as SALT does to convert amplitude to `mB`.

Coordinates used only for bias corrections need not be in `lcpar_names`.
BBC reads them by their FITRES column names or `fitres_aliases`; they can
also be supplied by `read_lcpar`. Additional simulation truth needed by
`sim_lcpar` is declared through `biascor_truth_aliases`, using the same
dictionary format.

### Coefficient-dependent corrections and error maps

If the bias correction depends on fitted global coefficients, declare
`biascor_coeffs` as tuples of `(coefficient_name, simulation_truth_column,
continuous)`. The final flag indicates whether the simulation draws the
coefficient continuously. Override `biascor_coeff_values` to return the
coefficient values at which BBC should interpolate, in declaration order.
SALT uses this for alpha, beta and the host-mass step.

`mucovscale_bins` and `ccprior_bins` optionally specify coordinates for
the distance-error map and simulated contaminant prior, respectively.
Both use `{name: bin_count}`; coordinate ranges come from
`default_cuts = {name: (lower, upper)}` and can be changed by the BBC
parameter windows. For example, BayeSN
uses `{"AV": 3}` for its error map. The
{py:class}`StandardisedModel <jax_snlc_core.models.standardised.StandardisedModel>`
API describes the remaining optional declarations.

## Sharing compilation between model variants

A model can reuse compiled fit functions across different templates when the
numerical arrays have the same dimensions and data types, and the fit settings
are compatible. This can reduce compilation time when fitting systematic
variations.

To support this, declare the model as a dataclass and prepare it in `load`
rather than `__init__`. The {ref}`model API <model-compilation>` describes
the fields and loading method. SALT, BayeSN and Redback provide examples.

## Contributing a model to the package

Import the base and role classes from `.base`, `.simulated`, `.fitted` and
`.standardised` within the package, rather than from `jax_snlc_core.models`,
to avoid circular imports. Add the model to the registry in
`jax_snlc_core/models/__init__.py` and add a `modeltable` directive for its
class to {doc}`model_settings`.
