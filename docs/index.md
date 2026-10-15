# jax-snana

Welcome to the documentation for jax-SNANA, a JAX implementation of the supernova
framework SNANA with support for survey simulation, light curve fitting
and cosmological analysis. jax-SNANA can be run on either CPU or GPU, and has
been designed as a modular, extensible codebase.

## Where to start

- Follow {doc}`installation` and {doc}`getting_started` for a first simulation
  and light-curve fit.
- {doc}`fitting_light_curves` explains how to fit your observations in Python
  or from the command line.
- {doc}`simulations` explains how to simulate your own survey.
- {doc}`cosmology_analysis` takes you through a complete cosmology analysis.

## Support

jax-snana has gone through extensive testing with a wide variety of inputs
to verify that different features are working as intended. However, there
will undoubtedly be edge cases which have not yet been encountered and
do not work quite right at the moment. If you come across something that
does not behave as intended, please raise an issue on the
[GitHub page](https://github.com/mattgrayling/jax-snana/issues).

There are some SNANA features that are not yet implemented in jax-snana,
for example simulations of strongly-lensed supernovae. The features that
are not implemented are detailed in {doc}`snana_compatibility`, though
many of these features may be implemented in time. If you come across
something missing which you would like to be included, please raise an
issue on the [GitHub page](https://github.com/mattgrayling/jax-snana/issues).

Additionally, these documentation pages are newly-written and will be
continually improved. If something is missing or unclear, again please
raise an issue on the [GitHub
page](https://github.com/mattgrayling/jax-snana/issues).

```{toctree}
:caption: Getting started
:maxdepth: 1

installation
getting_started
```

```{toctree}
:caption: Analysis guides
:maxdepth: 1

simulations
fitting_light_curves
classification
bias_corrections
systematics
cosmology
cosmology_analysis
building_pipelines
snana_and_pippin
sbi
```

```{toctree}
:caption: Models and extensions
:maxdepth: 1

models
adding_a_model
model_api
```

```{toctree}
:caption: Configuration
:maxdepth: 1

simulation_keys
fitting_keys
model_settings
bias_correction_keys
covariance_keys
cosmology_keys
pippin_settings
```

```{toctree}
:caption: Data and files
:maxdepth: 1

bundled_data
simulation_inputs
light_curve_data
analysis_outputs
```

```{toctree}
:maxdepth: 1

snana_compatibility
api
citing
```
