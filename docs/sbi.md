# Simulation-based inference

This page describes how to use the simulator inside your own inference code,
for example to train a neural posterior estimator or to fit population
parameters with gradients. You build a pipeline once from a simulation input
file, then call it with a random key and any population parameters you want to
change. Each call reuses the same compiled program. If you want to write
simulated data to disk instead, see {doc}`building_pipelines`.

Run the examples from the repository root, with `SNDATA_ROOT` set as in
{ref}`Preparing the examples <installation-examples>`:

```python
import jax
import jax.numpy as jnp
import jax_snlc_sim.run as sim_run

input_file = "examples/quickstart/sim_bayesn.input"
n_sne = 1000
pipeline = sim_run.build_sbi_pipeline(input_file, n_sne=n_sne)

results, selection = pipeline(
    jax.random.PRNGKey(0),
    params={"THETA": {"peak": 0.5, "sigma": (0.4, 0.4)}},
)
```

Each call generates `n_sne` events before selection. `results` contains the
generated parameters, the noise-free photometry and the observing schedule, and
`selection` contains the observed photometry and the survey's selection masks.
Without `params`, the populations in the input file are used.

Building the pipeline checks the input file in the same way as an ordinary
simulation, and also requires any conditional GENPDF map to cover every HOSTLIB
value it depends on. The compiled pipeline does not make the checks an ordinary
simulation makes on individual events.

## Population parameters

`params` maps each population parameter you want to change to new distribution
settings or to one value per event. Use the input-file names, such as `THETA`
for BayeSN or `SALT2c` for SALT; the model's aliases are also accepted.

For an asymmetric Gaussian the settings are `peak`, `sigma`, `skew`, `prob2`,
`peak2` and `sigma2`, where `sigma`, `skew` and `sigma2` give the lower and
upper values. For the exponential plus half-Gaussian dust distribution they are
`exp_tau`, `peak`, `sigma` and `ratio`. Both also accept `range`, which replaces
the bounds in the input file:

```python
params = {"RV": {"peak": 3.0, "sigma": (0.4, 0.4), "range": (1.2, 6.0)}}
```

To change the settings of a parameter fixed to one value in the input file,
give `range` as well. Settings you leave out keep their input-file values. The new settings apply to
every subpopulation, and any `VARIATION` in the input file is applied on top of
them.

If you already have samples, pass one value per event:

```python
theta = jnp.linspace(-1.0, 1.0, n_sne)
results, selection = pipeline(jax.random.PRNGKey(1), params={"THETA": theta})
```

The values are used as given, without resampling. They cannot be used for a
parameter with population reweighting, since they provide no probability density.

## Custom distributions

To draw a parameter from a distribution the input format does not support, give
a sampler when you build the pipeline. It receives one event's random key, any
dependencies (see {ref}`sbi-conditional`) and the settings you pass on each
call, and returns one value:

```python
def draw_theta(key, lower, width):
    return lower + width * jax.random.uniform(key)

pipeline = sim_run.build_sbi_pipeline(
    input_file, n_sne=n_sne, samplers={"THETA": draw_theta},
)
results, selection = pipeline(
    jax.random.PRNGKey(2), params={"THETA": {"lower": -1.0, "width": 2.0}},
)
```

A sampler replaces the whole draw, including the range and any `VARIATION` in
the input file. Like samples, it cannot be used for a parameter with population
reweighting.

(sbi-conditional)=
## Conditional populations

A parameter can also depend on other quantities, such as a host property or the
redshift. When you build the pipeline, register a function for it together with
the list of quantities it depends on, then pass the function's coefficients on
each call. For example, a colour distribution whose mean changes with host mass,
and a mass step in the magnitude offset:

```python
def colour_settings(mass, mean, slope, scatter):
    return {"peak": mean + slope * (mass - 10), "sigma": (scatter, scatter)}

def mass_step(mass, step):
    return jnp.where(mass > 10, step / 2, -step / 2)

pipeline = sim_run.build_sbi_pipeline(
    "examples/quickstart/sim_host_variation.input", n_sne=n_sne,
    conditional_settings={"SALT2c": (colour_settings, ["HOST.LOGMASS"])},
    relations={"GENMAG_OFF_GLOBAL": (mass_step, ["HOST.LOGMASS"])},
)
results, selection = pipeline(jax.random.PRNGKey(3), params={
    "SALT2c": {"mean": 0.0, "slope": 0.02, "scatter": 0.05},
    "GENMAG_OFF_GLOBAL": {"step": 0.1},
})
```

There are three kinds of function. `conditional_settings` functions return new
settings for the input-file distribution, `samplers` return a random draw and
take a key before their dependencies, and `relations` return a value with no
randomness. `GENMAG_OFF_GLOBAL` takes only a relation. A function with no
dependencies, such as the sampler above, is given on its own. Each function
receives its dependencies in the order listed, and its coefficients by name. A
dependency can be:

- `HOST.<column>`, a HOSTLIB column, which is loaded automatically when you
  build the pipeline from an input file, and must already be loaded in a
  prepared simulator;
- `REDSHIFT_CMB`, the true CMB-frame redshift before peculiar velocities;
- `REDSHIFT_HELIO`, the true heliocentric redshift;
- a population parameter drawn earlier.

Building the pipeline reports a dependency that is not available when the
parameter is drawn, for example a host property when a HOSTLIB weight map needs
the parameter before the host is chosen. A function cannot change a setting that
a `VARIATION` in the input file also changes.

## Complete events

To resimulate a set of events, including their positions, population parameters
and scatter, pass them as `events=`. The random key still sets the photometric
noise and the selection. The number of events is taken from these
arrays, and `params` and registered functions cannot be combined with them:

```python
pipeline = sim_run.build_sbi_pipeline(input_file, n_sne=n_sne)
baseline, _ = pipeline(jax.random.PRNGKey(4))
results, selection = pipeline(
    jax.random.PRNGKey(4), events=baseline["parameters"],
)
```

(sbi-photometry)=
## Using the photometry

The arrays keep rejected events and padded observations, so that their shapes
stay fixed inside compiled code. Mask them when you calculate summary
statistics. For example, the summed observed flux of each event, with rejected
events contributing zero:

```python
usable = (selection["passing_mask"][:, None]
          & selection["valid_mask"] & ~selection["saturated_mask"])
flux = selection["fluxcals_noisy_sat"]
total_flux = jnp.sum(jnp.where(usable, flux, 0.0), axis=1)
```

The observed uncertainties are `selection["fluxcalerrs_sat"]`, and the
noise-free fluxes are `results["fluxcals"]`. The times and filter indices are
`results["observation_schedule"]["mjds"]` and `["filter_indices"]`. These arrays
all have shape `(n_events, n_epochs)`. Fluxes use the configured `ZP_FLUXCAL`,
which defaults to 31.4, so use that zero point if you pass them to the
{doc}`Python light-curve fitter <fitting_light_curves>`.

Host galaxies are given by their row in the HOSTLIB, for example in
`results["parameters"]["host_galaxy_index"]`. To look up filter names or
HOSTLIB galaxy IDs, load the simulator first and build the pipeline from it:

```python
simulator = sim_run.prepare(input_file, n_sne=n_sne)
pipeline = sim_run.build_sbi_pipeline(simulator)
```

The filter indices then refer to `simulator.filter_names`, and
`simulator.hostlib.galaxy_ids(indices)` gives the galaxy IDs.

## Compilation and gradients

Calls reuse the compiled program as long as the dictionary keys and the array
shapes and types stay the same. Changing these, for example by switching a
parameter between settings and an array of values, compiles a new program.
Pipelines can be used inside `jax.jit`, `jax.vmap` and `jax.grad`.

Gradients pass through continuous distribution parameters. Discrete choices,
such as which subpopulation or host galaxy is drawn, grid snapping and the
selection cuts, do not contribute gradients.

## Limitations

Some simulation options are only available in an ordinary simulation
({doc}`building_pipelines`):

- spectroscopic efficiency maps based on `SNRSUM_REST_V`, which need simulated
  spectra;
- `NGEN_LC`, which generates until a number of events pass selection, whereas
  each SBI call generates `n_sne` events;
- changing the populations together with `HOSTLIB_MSKOPT` USESNPAR or
  `USE_SIMLIB_SALT2`, or the dust together with `GENAV_WV07`;
- distribution settings for a parameter drawn from a GENPDF map, although it can
  be replaced by samples or a sampler.
