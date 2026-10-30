# Simulation-based inference

This page describes how to use the simulator inside your own inference code.
You build a pipeline once from a simulation input file, then call it with a
random key. On each call you can change the hyperparameters of the population
distributions in the input file, or give the simulated values directly. When
you build the pipeline, you can also replace a distribution with your own, or
make it depend on other quantities such as the host mass. If you want to write
simulated data to disk instead, see {doc}`simulating`.

## Building a simulation pipeline

Run the examples from the repository root, with `SNDATA_ROOT` set as in
{ref}`Preparing the examples <installation-examples>`:

```python
import jax
import jax.numpy as jnp
import jax_snlc_sim.run as sim_run

input_file = "examples/quickstart/sim_bayesn.input"
n_sne = 1000
simulate = sim_run.build_sbi_pipeline(input_file, n_sne=n_sne)

results, selection = simulate(
    jax.random.PRNGKey(0),
    params={"theta": {"peak": 0.5, "sigma": (0.4, 0.4)}},
)
```

This simulates 1000 events from the BayeSN quick-start input, with the theta
distribution changed to peak at 0.5 with a width of 0.4 on each side. Without
`n_sne`, the number of events set by the input file is used.

`params` maps a parameter name, such as `theta` for BayeSN or `c` for SALT, to
new hyperparameters for its distribution, such as its peak and width. They are
the fields listed with their input-file keys in
{ref}`Population distributions <sim-population-settings>`, except `peakrange`
and `ngrid`, which cannot be changed. Hyperparameters you leave out keep their
input-file values, and any `VARIATION` in the input file still applies on top
of the new ones.

If you already have samples, pass one value per event instead of
hyperparameters, for a population parameter or for the magnitude offset
`GENMAG_OFF_GLOBAL`. The values are used as given:

```python
theta = jnp.linspace(-1.0, 1.0, n_sne)
results, selection = simulate(jax.random.PRNGKey(1), params={"theta": theta})
```

The first call includes JAX compilation, so it takes longer than later calls.
Later calls reuse the compiled program as long as the dictionary keys and the
array shapes and types stay the same. `simulate` can be used inside `jax.jit`
and `jax.vmap`.

## Output

`results` holds the simulated parameters and the noise-free photometry, and
`selection` holds the observed photometry and the survey's selection:

- `results["parameters"]`: the simulated parameters, one array per parameter;
- `selection["fluxcals_noisy_sat"]` and `selection["fluxcalerrs_sat"]`: the
  observed fluxes and their uncertainties;
- `results["fluxcals"]`: the noise-free fluxes;
- `results["observation_schedule"]["mjds"]` and
  `results["observation_schedule"]["filter_indices"]`: the observation times
  and filters, with indices into `simulate.filter_names`.

The photometry arrays have shape `(n_sne, n_epochs)`.

The arrays keep rejected events and padded observations, so that their shapes
stay fixed inside compiled code. Three masks select what you need:

- `selection["passing_mask"]`, of shape `(n_sne,)`: the events that pass
  selection;
- `selection["valid_mask"]`: the real observations among the padding;
- `selection["saturated_mask"]`: the saturated observations, which have no
  usable flux.

For example, the summed observed flux of each event, with rejected events
contributing zero:

```python
usable = (selection["passing_mask"][:, None]
          & selection["valid_mask"] & ~selection["saturated_mask"])
flux = selection["fluxcals_noisy_sat"]
total_flux = jnp.sum(jnp.where(usable, flux, 0.0), axis=1)
```

(sbi-custom-distributions)=
## Custom distributions

The input file can describe only the distribution shapes listed in
{ref}`Population distributions <sim-population-settings>`. To draw a parameter
from any other distribution, write a sampler, a function that draws the value
for one event, and give it under the parameter's name when you build the
pipeline:

```python
def draw_theta(key, lower, width):
    return lower + width * jax.random.uniform(key)

simulate = sim_run.build_sbi_pipeline(
    input_file, n_sne=n_sne, samplers={"theta": draw_theta},
)
results, selection = simulate(
    jax.random.PRNGKey(2), params={"theta": {"lower": -1.0, "width": 2.0}},
)
```

The simulator calls the sampler once for each event, with a random key for that
event as the first argument. The values you give for the parameter in `params`
follow as keyword arguments, so the sampler's argument names must match their
keys. This sampler draws theta uniformly between -1 and 1.

The sampler replaces the input-file distribution for that parameter entirely,
including its range and any `VARIATION`.

## Conditional populations

A parameter can also depend on other quantities, such as the mass of the host
galaxy, through a function given under one of two arguments when you build the
pipeline:

- in `conditional_hyperparameters`, the function returns hyperparameters, such
  as the peak, for the event's distribution in the input file, and the value is
  drawn from that distribution;
- in `conditional_samplers`, the function is a sampler, as in
  {ref}`sbi-custom-distributions`, that also receives the event after its key.

The function receives `event`, a dictionary of the event's values simulated so
far, under their names in `results["parameters"]`, such as `host_LOGMASS`,
`sim_redshift_cmb` or `x1`.

The following example uses `examples/quickstart/sim_host.input`, the SALT
quick-start simulation with the bundled host catalogue. It adds the stretch-mass
relation and the mass step, with the slope and the size of the step given on
each call:

```python
def stretch_by_mass(event, slope):
    return {"peak": slope * (event["host_LOGMASS"] - 10)}

def mass_step(key, event, step):
    return jnp.where(event["host_LOGMASS"] > 10, -step, 0.0)

simulate = sim_run.build_sbi_pipeline(
    "examples/quickstart/sim_host.input", n_sne=n_sne,
    conditional_hyperparameters={"x1": stretch_by_mass},
    conditional_samplers={"GENMAG_OFF_GLOBAL": mass_step},
)
results, selection = simulate(jax.random.PRNGKey(3), params={
    "x1": {"slope": -0.3},
    "GENMAG_OFF_GLOBAL": {"step": 0.06},
})
```

The magnitude offset `GENMAG_OFF_GLOBAL` has no input-file distribution, so it
takes either one value per event or a sampler.

This conditional sampler draws the SALT2 colour with a scatter of 0.1 about a mean that
depends on x1 and redshift:

```python
def draw_colour(key, event, slope):
    mean = slope * event["x1"] + event["sim_redshift_cmb"]
    return mean + 0.1 * jax.random.normal(key)

simulate = sim_run.build_sbi_pipeline(
    "examples/quickstart/sim_host.input", n_sne=n_sne,
    conditional_samplers={"c": draw_colour},
)
results, selection = simulate(
    jax.random.PRNGKey(4), params={"c": {"slope": 0.02}},
)
```

A function can only use values simulated before its parameter. The host is
chosen first, then the model's parameters are drawn in the model's order (x1, c,
alpha and beta for SALT), then the host dust (RV, then AV), and the magnitude
offset last. A parameter that a HOSTLIB weight map uses is drawn before the host
is chosen. To use a HOSTLIB column that the simulation does not otherwise read,
list it in `HOSTLIB_STOREPAR`.
