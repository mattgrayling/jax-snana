# Models

jax-SNANA supports simulation and light curve fitting with SALT, BayeSN
and a selection of transient models from redback-jax. Simulations can also
use user-supplied spectral time series, either as individual templates or
as a grid spanning physical parameters. {doc}`model_settings` gives the
configuration options for each model.

(model-salt)=
## SALT

jax-SNANA supports simulations, light curve fitting and standardisation using
the type Ia supernova SED model SALT. It supports both SALT2 ([Guy et al. 2007](https://arxiv.org/abs/astro-ph/0701828))
and SALT3 ([Kenworthy et al. 2021](https://arxiv.org/abs/2104.07795)).

The examples in {doc}`getting_started` and {doc}`cosmology_analysis` use
the bundled `SALT3.DOVEKIE` model, trained with the Dovekie
cross-calibration described by
[Popovic et al. (2025)](https://arxiv.org/abs/2506.05471).

(model-bayesn)=
## BayeSN

BayeSN ([Mandel et al. 2022](https://arxiv.org/abs/2008.07538)) is a
hierarchical Bayesian model of type Ia supernova spectral energy
distributions across optical and near-infrared wavelengths. It separates
intrinsic supernova variation from host-galaxy dust and infers a distance
modulus directly from each light curve, without a subsequent standardisation
step.

The BayeSN example in {doc}`getting_started` uses the bundled G26 model,
presented in
[Grayling et al. (2026)](https://arxiv.org/abs/2606.19429).

## NON1ASED

NON1ASED is SNANA's interface for simulating events from a library of
spectral time series, commonly used for core-collapse and peculiar type Ia
supernovae. Each event
uses a selected template, with optional brightness offsets and scatter.
The template interface follows SNANA
([Kessler et al. 2009](https://arxiv.org/abs/0908.4280)); the physical models
and relevant citations depend on the chosen library.

## SIMSED

SIMSED is SNANA's interface for simulating events from a grid of spectral
time series indexed by model parameters. It can interpolate within the grid or select individual templates,
allowing tabulated calculations such as radiative-transfer and kilonova models
to be used in survey simulations. This interface also follows SNANA
([Kessler et al. 2009](https://arxiv.org/abs/0908.4280)).

## Redback

jax-snana uses the bolometric models in `redback-jax` for transients including
supernovae, kilonovae, magnetar-powered transients and tidal disruption events.
Their luminosity and photospheric evolution provide the spectra used to
simulate and fit multiband light curves.

See [Sarin et al. (2024)](https://arxiv.org/abs/2308.12806) for Redback and
the [Redback model documentation](https://redback.readthedocs.io/en/latest/)
for the papers describing individual models. The supported fitting models
and parameter conventions are described in {doc}`model_settings`.

## Selecting a model

Select the model with `GENMODEL` when simulating, or `FITMODEL_NAME` when
fitting. For SALT, BayeSN, NON1ASED and SIMSED, give the model directory, for
example:

```text
GENMODEL:  $SNDATA_ROOT/models/SALT3/SALT3.DOVEKIE
```

BayeSN also accepts the path to its `BAYESN.YAML` file. For Redback, give the
model identifier, for example `REDBACK.arnett_bolometric`.

The model type is recognised from the last component of the path or the
identifier, using these prefixes (case-insensitive):

| Model | Recognised prefixes |
|---|---|
| SALT | `salt2`, `salt3` |
| BayeSN | `bayesn` |
| NON1ASED | `nonia`, `non1a`, `noniased`, `non1ased` |
| SIMSED | `simsed` |
| Redback | `redback` |

Environment variables in the path are expanded, so use `$SNDATA_ROOT` for a
model in the SNANA data tree.

In the bias-correction stage the model is set with `model_lcfit`, which can be
SALT or BayeSN.

For a first simulation and fit with BayeSN or SALT, follow
{doc}`getting_started`. To incorporate another model, see {doc}`adding_a_model`.

### Selecting a model in Python

For light curve fitting in Python, the G26 and W22 BayeSN models and the
Dovekie and K21 SALT3 models are included with jax-SNANA. Select one by
passing its name to `LightCurveFitter`, for example
`LightCurveFitter("BAYESN.G26")`. The other names are `BAYESN.W22`,
`SALT3.DOVEKIE` and `SALT3.K21`. To use another SALT or BayeSN model,
pass the path to its model directory instead. For Redback, pass the model
name, for example `LightCurveFitter("REDBACK.arnett_bolometric")`.

{doc}`model_settings` lists the fitted parameters and configuration
options for each model.

{doc}`bundled_data` lists the included model distributions and their sources.
