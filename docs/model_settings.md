# Model settings

{doc}`models` describes the available models and their papers. This page
covers their input settings, parameter names and model-specific simulation
and fitting options. For complete examples, see {doc}`getting_started`.

```{contents} On this page
:local:
:depth: 1
```

## SALT

### Simulating with SALT

For each simulated supernova, `x1` and `c` are drawn from the populations
specified in the input. Use the distribution settings with suffixes
`SALT2x1` and `SALT2c`, such as `GENPEAK_SALT2x1`, `GENSIGMA_SALT2x1` and
`GENRANGE_SALT2x1`, or supply a `GENPDF` map. The standardisation
coefficients can also vary between supernovae, using populations named
`SALT2ALPHA` and `SALT2BETA`.

Specify all four populations, except when beta is supplied by the colour
polynomial described below. {doc}`supernova_populations` explains the
available distributions, and {doc}`getting_started` provides a SALT simulation
input. `GENPEAK_SALT2ALPHA` and `GENPEAK_SALT2BETA` may also be spelt
`GENALPHA_SALT2` and `GENBETA_SALT2`.

`SALT2BETA_cPOLY` defines a polynomial in the event's drawn colour. If no
`SALT2BETA` population is specified, the polynomial supplies beta. If a
population is specified, the polynomial is added to the drawn beta.

The simulator adds intrinsic scatter beyond what `x1` and `c` describe
through `GENMAG_SMEAR_MODELNAME`. `G10` reads the
colour dispersion in the SALT directory's `salt2_color_dispersion.dat` or
`salt3_color_dispersion.dat` file. `C11` uses the correlated band-to-band
scatter measured by Chotard et al. (2011) and does not require SALT model
files. Both are described under {ref}`population-scatter`.

(salt-generation-options)=
### SALT generation options: `GENMODEL_MSKOPT`

The simulator applies the training calibration shifts in `SALT.INFO`.
Combine the following bit values to change the model setup:

| Value | Behaviour |
| --- | --- |
| 2 | Require a `DOCUMENTATION:` header in the INFO file. |
| 4 | Disable the training magnitude shifts. |
| 8 | Disable the training wavelength shifts. |
| 16 | Set negative SED flux to zero before integration. |
| 64 | Require the error map to cover the SED's phase and wavelength range. |

Default: **0**. Other bits, including WAVECOR (32), are not supported.

### Fitting with SALT

The fit adjusts the time of peak, amplitude, `x1` and `c`. The fitted
amplitude is also reported as `mB`, the apparent *B*-band magnitude used for
standardisation. The priors on `x1` and `c` are flat within their specified
ranges and penalise values outside them, with an additional penalty for
`c > 0.5`.

The `SALT2ALPHA` and `SALT2BETA` keys below are the fitter's own, separate
from the simulator's populations of the same names. The fitter uses them only
to convert a distance modulus into the amplitude of a supernova with a given
shape and colour, to start a fit and to evaluate a photometric-redshift grid.
They have defaults, 0.14 and 3.2, and do not affect the standardisation.

```{eval-rst}
.. modeltable:: jax_snlc_core.models.salt.SALTModel
```

### SALT distances

The distance fit standardises SALT magnitudes using the fitted shape,
colour and host-mass coefficients, then applies selection-bias corrections.
See {doc}`bias_corrections` for the standardisation and distance-fitting
options.

SALT has three keys of its own in the bias-correction stage, which reject
fits with unusable errors:

```{eval-rst}
.. modeltable:: jax_snlc_core.models.salt.SALTModel
   :context: bbc
```

## BayeSN

BayeSN is selected using a model directory containing `BAYESN.YAML`, or the
path to that YAML file. {doc}`getting_started` provides complete simulation
and fitting inputs for G26.

### Simulating with BayeSN

Set the BayeSN shape population with `GENPEAK_THETA`, `GENSIGMA_THETA` and
`GENRANGE_THETA`, or with a GENPDF map naming `THETA`. Parameter suffixes
are case-insensitive.

Host dust is drawn as for every model, with `GENRANGE_AV` and `GENTAU_AV` for
the extinction, and the `GENPEAK_RV`, `GENSIGMA_RV` and `GENRANGE_RV` family
for the law (with `GENMEAN_RV` accepted for the peak). With no R_V key every
event has `R_V` = 3.1. The trained file's `R_V` is used only as the fitter's
prior.

The host extinction law defaults to Fitzpatrick (1999). `HOST_COLORLAW`
selects another registered law, such as `CCM89`, for simulation or fitting.
`OPT_MWCOLORLAW` separately controls Milky Way extinction.

(bayesn-generation-options)=
### BayeSN generation options: `GENMODEL_MSKOPT`

Combine bit values to select intrinsic scatter and extrapolation:

| Value | Behaviour |
| --- | --- |
| 1 | Draw residuals `epsilon` from the trained covariance and a grey offset `DELTAM` with Gaussian width `SIGMA0`. |
| 2 | Draw residuals only. |
| 4 | Draw the grey offset only. |
| 16 | Set negative SED flux to zero in simulated spectra. |
| 32 | Bring the warping smoothly to zero beyond the phase knots. |
| 64 | Continue the warping smoothly to a constant beyond the phase knots. |
| 512 | Bring the warping smoothly to zero beyond the wavelength knots. |
| 1024 | Continue the warping smoothly to a constant beyond the wavelength knots. |

Default: **0**, with neither scatter component drawn. Phase extrapolation
levels off over one knot spacing and is then constant; wavelength
extrapolation is linear. Use 1 to enable both scatter components, without
combining it with 2 or 4. Select at most one of 32/64 and one of 512/1024.

### Fitting with BayeSN

The fit adjusts the time of peak, `theta`, the distance modulus, `A_V`,
optionally `R_V`, and the intrinsic residuals. The model supplies priors
on shape, host extinction and residuals; these are separate from the
simulation distributions above. {doc}`fitting_keys` describes how to
configure fitting priors.

BayeSN reports distances directly, without additional standardisation.
The {ref}`FITRES table <format-fitres>` contains the distance as
`MU_LCFIT`, with uncertainty `MUERR_LCFIT`. BBC can apply selection-bias
corrections to these distances; see {doc}`bias_corrections`.

For G26, fitted bands must have rest-frame mean wavelengths between
2800 and 10800 Å, also subject to `RESTLAMBDA_FITRANGE`. The training phase
knots span -10 to 40 days. `FITWIN_TREST` selects the fitted phase range;
beyond those knots the spectral warping levels off and is held constant.

## NON1ASED

The model directory contains a `NON1A.LIST` catalogue and one `.SED` or
`.SED.gz` file per template. Each catalogue row gives the template an index,
a type such as `IIP` or `Ib`, and the SED file to read. Which templates a
run uses, and how often each is drawn, is set in the simulation input as a
small table:

```text
NON1A_KEYS: 5
    INDEX  WGT  MAGOFF  MAGSMEAR  SNTYPE
NON1A:  201  0.40  0.0  0.0  20
NON1A:  204  0.20  0.0  0.0  20
NON1A:  101  0.15  0.0  0.0  20
NON1A:  103  0.15  0.0  0.0  20
NON1A:  002  0.10  0.0  0.0  20
```

`NON1A_KEYS` takes the number of columns followed by that many names, on the
same line or the lines below. Each row
starts with `NON1A` for an ordinary core-collapse template or `PEC1A` for a
peculiar type Ia one. The five column names are:

- `INDEX`: the template's index in `NON1A.LIST`.
- `WGT`: its relative weight.
- `MAGOFF`: a magnitude offset applied to every event drawn from it.
- `MAGSMEAR`: the width in magnitudes of a further brightness scatter, drawn
  once per event and truncated at `SIGMACLIP_MAGSMEAR`.
- `SNTYPE` (or `SNTAG`): the type code written for those events, which
  `GENTYPE` overrides wherever it is given.

Use these column names in `NON1A_KEYS`. Without that key, the columns are
`INDEX` and `WGT`.

Each weight sets the share of the run given to its template, subject to
rounding to whole events. The weights are normalised within each
class, and the two classes split the generated count in the ratio of their
rate models (`DNDZ` against `DNDZ_PEC1A`).

`NON1A_INDEX_FORCE` restricts a run to the template with that index. It is
set on the command line, as
`snlc_sim my.input NON1A_INDEX_FORCE 103`, rather than in the input file.

Templates have their own phase grids. By default, phase zero is shifted to
bolometric maximum. `OPTMASK_T0SHIFT_EXPLODE` selects a different phase
origin, including estimates of the explosion time; its choices are listed
under Common keys below. `GENRANGE_TREST` is
not limited to any one template's phase grid. An epoch before a template's
first phase has no flux, and one after its last is continued by the
late-time decay fitted to the template's final phases.

NON1ASED events count towards the non-Ia rate, so they are scaled by the
non-Ia `DNDZ` entries and scaling keys. You choose their type codes:

- `GENTYPE` sets the true type, written as `SIM_GENTYPE`, and where given it
  also sets the `SNTYPE` of every event, overriding the table's `SNTYPE`
  column.
- `SNTYPE_Ia` and `SNTYPES_Ia` name the code a type Ia run writes.

Give a core-collapse run its own `GENTYPE`, conventionally 10 or above.
The photometric-only code is 100 above
the spectroscopic one unless `SNTYPES_Ia` or `GENTYPES` gives the pair
explicitly.

Fit these light curves with SALT to study how
core-collapse supernovae appear in a type Ia analysis.

## SIMSED

The directory contains a `SED.INFO` file and one SED file per grid point.
`SED.INFO` lists how many parameters there are and their names, a global
flux scale and the rest-frame wavelength range the model may be used over,
then lists every grid point with its parameter values and its file.

Which parameters vary, and how, is set in the simulation input with these
keys:

- `SIMSED_PARAM` names a parameter drawn from a continuous distribution, with
  the model interpolated multilinearly between the surrounding grid points.
- `SIMSED_GRIDONLY` names one that takes only the grid values, with no
  interpolation. `SIMSED_GRIDONLY: SEQUENTIAL` generates one event per
  grid point in turn, which fixes the number of events the run produces.
- `SIMSED_REDCOR(p1,p2)` sets a reduced correlation coefficient between two
  continuous parameters, and `SIMSED_COV(p1,p2)` an absolute covariance.
- `SIMSED_WGTMAP_FILE` supplies a weight map over the grid.
- `SIMSED_SHAPEPAR`, `SIMSED_COLORPAR` and `SIMSED_COLORLAW` are alternative
  spellings of `SIMSED_PARAM` that also mark the parameter's role in the
  output tables.

Continuous and `GRIDONLY` parameters cannot be combined in one model.
For example, if `SED.INFO` declares a parameter named `stretch`, the
following settings interpolate in that parameter with a truncated Gaussian
population:

```text
SIMSED_PARAM: stretch
GENPEAK_stretch: 1.0
GENSIGMA_stretch: 0.1 0.1
GENRANGE_stretch: 0.8 1.2
```

The parameter name and range must correspond to the grid in `SED.INFO`.
A parameter set by none of these is written to the output as a truth column
but never sampled.

SIMSED is a simulation model. As with NON1ASED, its events count towards the
non-Ia rate and its templates are extrapolated under the shared late-time keys.

`SIMSED_USE_BINARY` reads a binary cache in place of the text SED files,
which can reduce loading time for large grids.

## Redback

Select a model by its `redback-jax` name, for example
`GENMODEL: REDBACK.arnett_bolometric`.

For Redback models, `PEAKMJD` and the fitted time parameter refer to the
explosion epoch. `GENRANGE_TREST` therefore specifies a phase window about
the explosion rather than maximum light.

The required bolometric parameters depend on the selected `redback-jax`
model. For fitting parameters and bounds, use
`fitter.model.get_fit_parameters()` on a {ref}`Python fitter
<fit-from-python>`. {doc}`models` links to the Redback model documentation.

Bolometric parameters without defaults, or with a default of `None`, require
population distributions in the simulation input; optional physical
parameters can also be given populations. An Arnett model takes a nickel
fraction, an ejecta mass, an ejecta velocity and opacities, while a magnetar
model takes a spin period and a magnetic field. Converting the bolometric
luminosity to a spectrum also requires the ejecta velocity `vej` and a
minimum temperature `temperature_floor` in kelvin, even for a bolometric
model that does not take those arguments.

Supply population distributions for `vej`, `temperature_floor` and every
bolometric-model parameter without a default, using the `GENPEAK_<name>`,
`GENSIGMA_<name>` and `GENRANGE_<name>` settings or a `GENPDF` map. The fit
adjusts the same parameters with model-specific default bounds, together
with the explosion epoch and optionally the distance modulus. The
distance is fixed by default because it is degenerate with the luminosity
scale.

Use the velocity units of the selected `redback-jax` model. For
`shocked_cocoon_bolometric`, `metzger_kilonova_bolometric` and
`magnetar_boosted_kilonova_bolometric`, `vej` is a fraction of the speed of
light; the other supported models use km/s. Simulation inputs, truth columns
and fitted parameters retain these units.

All three models above support simulation. The shocked-cocoon model also
supports fitting, with its velocity bounds and starting value expressed in
fractions of the speed of light. The two kilonova models are available for
simulation only.

```{eval-rst}
.. modeltable:: jax_snlc_core.models.redback.RedbackModel
   :context: simulation
```

These settings also apply when fitting with Redback.
`redback-jax` is an optional dependency, installed with the `redback` extra
as {doc}`installation` describes.

(model-common-keys)=
## Common keys

The simulator and fitter both accept these shared settings. Their
applicability is specified in each description:

```{eval-rst}
.. modeltable:: jax_snlc_core.models.sed_model.SNModel
```

The following settings apply only to simulation:

```{eval-rst}
.. modeltable:: jax_snlc_core.models.roles.SimulatedModel
   :context: simulation
```

The following settings apply only to fitting:

```{eval-rst}
.. modeltable:: jax_snlc_core.models.roles.FittedModel
```
