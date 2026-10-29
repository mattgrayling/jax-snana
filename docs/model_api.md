# Model API

{doc}`adding_a_model` introduces the model interfaces through a
worked example. `SNModel` defines the spectrum; the other classes add simulation,
fitting and distance standardisation.

## Registration

```{eval-rst}
.. autofunction:: jax_snlc_core.models.register_model
```

## Spectrum

```{eval-rst}
.. py:class:: jax_snlc_core.models.sed_model.SNModel

   Base class for a model's spectral evaluation and wavelength support.
```

Every model implements `get_wavelength_range`, `phase_range`,
`sed_param_names` and `generate_sed`:

```{eval-rst}
.. automethod:: jax_snlc_core.models.sed_model.SNModel.get_wavelength_range

.. automethod:: jax_snlc_core.models.sed_model.SNModel.phase_range

.. automethod:: jax_snlc_core.models.sed_model.SNModel.sed_param_names

.. automethod:: jax_snlc_core.models.sed_model.SNModel.generate_sed
```

### Preparation and photometry

`load` constructs and prepares a model. `choose_grid` provides a working
wavelength grid, and `evaluate` supplies the shared photometric integration.
{ref}`model-compilation` describes preparation for dataclass models.

```{eval-rst}
.. automethod:: jax_snlc_core.models.sed_model.SNModel.load

.. automethod:: jax_snlc_core.models.sed_model.SNModel.choose_grid

.. automethod:: jax_snlc_core.models.sed_model.SNModel.evaluate
```

### Amplitude adjustment

These shared methods change a model's amplitude for a distance-modulus shift
or a specified distance. A model with a non-magnitude amplitude implements
the scaling described in {doc}`adding_a_model`.

```{eval-rst}
.. automethod:: jax_snlc_core.models.sed_model.SNModel.shift_amplitude

.. automethod:: jax_snlc_core.models.sed_model.SNModel.params_at_distance

.. automethod:: jax_snlc_core.models.sed_model.SNModel.amplitude_jacobian
```

## Simulation

```{eval-rst}
.. py:class:: jax_snlc_core.models.roles.SimulatedModel

   Adds population sampling and simulation truth output to a spectral model.
```

Population draws and truth columns have shared defaults. Models can extend
these through:

```{eval-rst}
.. automethod:: jax_snlc_core.models.roles.SimulatedModel.derived_parameters

.. automethod:: jax_snlc_core.models.roles.SimulatedModel.get_snana_header_params
```

### Input and sampling

These optional methods customise structured inputs and sampling setup.
Models using template libraries can override them to select templates and
set per-template event counts.

```{eval-rst}
.. automethod:: jax_snlc_core.models.roles.SimulatedModel.structured_input

.. automethod:: jax_snlc_core.models.roles.SimulatedModel.configure_generation

.. automethod:: jax_snlc_core.models.roles.SimulatedModel.drawn_parameters

.. automethod:: jax_snlc_core.models.roles.SimulatedModel.generation_quotas

.. automethod:: jax_snlc_core.models.roles.SimulatedModel.pec1a_flags

.. automethod:: jax_snlc_core.models.roles.SimulatedModel.apply_host_snpar_overrides

```

### Output and selection

These optional methods supply model-specific text output, replay parameters
from a simulation dump, or provide quantities for selection:

```{eval-rst}

.. automethod:: jax_snlc_core.models.roles.SimulatedModel.text_sim_header_lines

.. automethod:: jax_snlc_core.models.roles.SimulatedModel.replay_params_from_dump

.. automethod:: jax_snlc_core.models.roles.SimulatedModel.sn_variables
```

The model-development guide describes the population and simulation
{ref}`declarations <model-population-parameters>`. These attributes have
separate descriptions there:

```{eval-rst}
.. autoattribute:: jax_snlc_core.models.roles.SimulatedModel.population_parameters

.. autoattribute:: jax_snlc_core.models.roles.SimulatedModel.population_aliases

.. autoattribute:: jax_snlc_core.models.roles.SimulatedModel.population_defaults

.. autoattribute:: jax_snlc_core.models.roles.SimulatedModel.crazyflux_factor

.. autoattribute:: jax_snlc_core.models.roles.SimulatedModel.uses_nonia_rates

.. autoattribute:: jax_snlc_core.models.roles.SimulatedModel.mjd_explode_anchor
```

## Light-curve fitting

```{eval-rst}
.. py:class:: jax_snlc_core.models.roles.FittedModel

   Adds fitted-parameter declarations, model uncertainties and fit output.
```

Every fitted model implements `get_fit_parameters`. The remaining methods
can be overridden to supply a model uncertainty, convert the fit vector into
SED parameters, customise redshift initialisation or add output columns.

```{eval-rst}
.. automethod:: jax_snlc_core.models.roles.FittedModel.get_fit_parameters

.. automethod:: jax_snlc_core.models.roles.FittedModel.sed_params_from_fit

.. automethod:: jax_snlc_core.models.roles.FittedModel.compute_raw_fracerr

.. automethod:: jax_snlc_core.models.roles.FittedModel.color_dispersion_fraction

.. automethod:: jax_snlc_core.models.roles.FittedModel.evaluate_redshift_grid

.. automethod:: jax_snlc_core.models.roles.FittedModel.compute_fitprob

.. automethod:: jax_snlc_core.models.roles.FittedModel.fitres_param_columns

.. automethod:: jax_snlc_core.models.roles.FittedModel.get_fitres_param_columns

.. automethod:: jax_snlc_core.models.roles.FittedModel.get_fitres_derived_columns
```

(advanced-fit-spec)=
### Advanced fit-parameter settings

These optional entries extend the parameter declarations described in
{doc}`adding_a_model`.

| Entry | Value and behaviour |
|---|---|
| `grid_search`, `photoz_grid_search` | Tuples of candidate initial values for fixed-redshift and photometric-redshift fits, respectively. Values for `PKMJD` are offsets in days from the initial peak estimate; other parameters use physical values. |
| `coarse_grid` | `{"range": (lower, upper), "nbin": count}` includes this parameter in the coarse photometric-redshift initialisation grid. |
| `inistp_relative_to_init` | Sets the fallback error step as a fraction of the parameter's initial value, in place of an absolute `inistp`. |
| `errmax_bad` | Lower uncertainty threshold in the parameter's physical units. With `MINOS_ERRMAX_BAD` enabled, uncertainties below this threshold are replaced by the step size and flagged as invalid. |
| `error_bounds` | `(lower, upper)` limits on the reported uncertainty, in physical units. `CHECK_CRAZYFITERR` uses these to reject events during post-fit selection; held parameters are exempt from the limits. |
| `peg_margin` | Distance in parameter units from the edges of its flat prior window. Common-event selection with `OPT_SNCID_LIST` rejects fits closer to an edge than this margin. |
| `deferred` | `True` holds the parameter fixed in the first fitting round and frees it in later rounds. Its covariance columns are omitted. |
| `covariance` | `False` omits this parameter's covariance columns from FITRES. |
| `ndof` | `False` excludes the parameter from the fitted-parameter count used to calculate degrees of freedom. |
| `scale_at_distance` | `True` scales the parameter per event using the amplitude at the reference cosmology's distance, rather than a single scale for the sample. |
| `error_surface_round1` | Evaluates the model uncertainty at this parameter value in the first fitting round. In subsequent rounds, a frozen error surface uses the parameter's value at the start of the round. |

## Distance standardisation

```{eval-rst}
.. py:class:: jax_snlc_core.models.roles.StandardisedModel

   Defines how fitted light-curve parameters give distances and how BBC reads
   their fitted and simulated values.
```

This role requires `compute_mu` and `sim_lcpar`. The other methods customise
bias-correction coefficients, FITRES reading and covariance checks.
{doc}`adding_a_model` describes the nuisance-parameter and map declarations.

```{eval-rst}
.. automethod:: jax_snlc_core.models.roles.StandardisedModel.compute_mu

.. automethod:: jax_snlc_core.models.roles.StandardisedModel.sim_lcpar

.. automethod:: jax_snlc_core.models.roles.StandardisedModel.biascor_coeff_values

.. automethod:: jax_snlc_core.models.roles.StandardisedModel.read_lcpar

.. automethod:: jax_snlc_core.models.roles.StandardisedModel.bbc_bad_covariance

.. py:class:: jax_salt2mu.tablevar.TableVar

   A sample read by BBC. ``lcpar`` holds fitted parameter arrays,
   ``covmat_fit`` their covariance matrices, ``sim_truth`` the simulation-truth
   columns and ``mumodel`` the reference distance modulus at each event's
   redshift. BBC supplies this object to ``sim_lcpar``.
```

## Tabulated spectra

Use these helpers to read the shared SED format and resample a model onto its
working grid. The model-development guide explains interpolation and
extrapolation choices.

```{eval-rst}
.. autofunction:: jax_snlc_core.models.sed_model.read_sed_file

.. autofunction:: jax_snlc_core.models.simsed.read_sed_info

.. autofunction:: jax_snlc_core.models.simsed.read_sed_binary

.. autofunction:: jax_snlc_core.models.sed_model.resample_surfaces

.. autofunction:: jax_snlc_core.models.sed_model.uvlam_extrapflux_sedmodel

.. autofunction:: jax_snlc_core.models.sed_model.late_time_tables
```

(model-compilation)=
## Sharing compilation between model variants

To share compiled fit functions between templates with matching array
dimensions and data types, declare the model class as a
`@dataclass(eq=False)`. `register_model` registers its fields with JAX; no
additional class or JAX registration is needed.

Declare prepared numerical arrays as `jax.Array` fields. Constructor fields
are passed into compiled fits, so a different template with the same shape
and dtype reuses the compiled program. Use `field(metadata={"static": True})`
for switches that select Python branches, such as an interpolation method.
Changing a static field selects a different compilation. File handles, parsed
input dictionaries and other preparation-only state should use
`field(init=False, ...)`; numerical methods must not read those fields.

Keep file reading and preparation in a `load` classmethod with the same
arguments as the constructor in {doc}`adding_a_model`. It creates the instance,
calls `SNModel.__init__` to apply input settings, loads its arrays, prepares
interpolation coefficients, and returns the finished model. The generated
dataclass constructor only assigns fields: JAX uses it when reconstructing
model arguments during tracing. Do not put loading or preparation in
`__post_init__`. SALT, BayeSN and Redback show this pattern.

Annotated scalar attributes declare input settings. Array, dictionary and
tuple fields describe model data, rather than new input keys. Mark a
prepared scalar field with `metadata={"input": False}` if it is not an
input-file setting; this can be combined with `"static": True`. To load the
bundled models directly, use `SALTModel.load(path, config)` (and likewise
for BayeSN and Redback).

The pipeline shares compiled functions within a `Fit` stage when their
fit settings, filter integration and dust calculations are compatible.
Shapes, dtypes and static model fields still determine JAX's compilation variants. Different wavelength
grids, dust preparations or fit settings can require separate compilation.
