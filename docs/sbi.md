# Simulation-based inference

jax-SNANA has been designed to support simulation-based inference (SBI) by enabling
a survey simulation to be prepared and compiled, which can then be repeatedly called
within the training loop of an SBI method to create fast on-the-fly simulations with
different distributions of parameters. This page gives an overview of how jax-SNANA
can interface with downstream SBI applications.

## Requirements

Use an input file set up as for {doc}`ordinary simulation <simulating>`.
Specify the observing times and conditions through a {ref}`SIMLIB
<format-simlib>`, the filters and calibration through a {ref}`kcor file
<format-kcor>`, and the source model and its population distributions. The
example below uses `examples/quickstart/sim_bayesn.input`, the survey simulation input from
{doc}`getting_started`. Run from the repository root with
`SNDATA_ROOT` set as described under
{ref}`Preparing the examples <installation-examples>`. The same interface
accepts a SALT simulation input.

## Survey realisations

Prepare the survey once from its simulation input, then generate a
realisation for each random key. Here `n_sne=1000` sets the number of
supernovae generated per call; the input file specifies their population,
observations and selection:

```python
import jax
import jax_snlc_sim.run as sim_run

simulator = sim_run.prepare("examples/quickstart/sim_bayesn.input", n_sne=1000)
pipeline_fn = sim_run.build_sbi_pipeline(simulator)

results, selection, record = pipeline_fn(jax.random.PRNGKey(0))
sim_run.finish_batch(simulator, results, record,
                     start_index=0, n_events=simulator.n_sne)
```

The first simulation call compiles the calculation; subsequent calls reuse
the compiled function. `results` contains the noise-free photometry,
generated parameters and observing conditions. `selection` contains the noisy fluxes, uncertainties
and selection masks used to construct your summary statistics.

The third return value, `record`, contains generation checks. Call
`finish_batch` with the results and record after each realisation to report
any generation failures and attach host-galaxy identifiers when a HOSTLIB is
used. This call runs outside the compiled simulation.

### Varying the population

Use `adjust` to change the sampled parameters before the light curves are
generated. It takes a random key and the per-event parameter dictionary,
and returns the updated dictionary. For example, the following replaces
the BayeSN shape parameter's distribution with a Gaussian truncated to
$-3 < \theta < 3$, with a mean and width supplied on each call:

```python
import jax.numpy as jnp

@jax.jit
def simulate_population(key, theta_mean, theta_sigma):
    def adjust(draw_key, params):
        event_keys = jax.vmap(lambda i: jax.random.fold_in(draw_key, i))(
            params["sn_global_idx"])
        draws = jax.vmap(lambda k: jax.random.truncated_normal(
            k, (-3.0 - theta_mean) / theta_sigma,
            (3.0 - theta_mean) / theta_sigma))(event_keys)
        theta = theta_mean + theta_sigma * draws
        return {**params, "theta": theta}

    return pipeline_fn(key, adjust=adjust)

results, selection, record = simulate_population(
    jax.random.PRNGKey(1), jnp.asarray(0.0), jnp.asarray(0.8))
sim_run.finish_batch(simulator, results, record,
                     start_index=0, n_events=simulator.n_sne)
```

The same function can then generate a survey with a different shape
distribution without recompiling:

```python
results, selection, record = simulate_population(
    jax.random.PRNGKey(2), jnp.asarray(0.5), jnp.asarray(0.4))
sim_run.finish_batch(simulator, results, record,
                     start_index=0, n_events=simulator.n_sne)
```

`theta_mean` and `theta_sigma` are the mean and standard deviation of the
Gaussian before truncation. Their shapes and dtypes must remain unchanged
to reuse the compilation. Other population parameters retain the
distributions specified in the input file. Deriving each event's random
key from `sn_global_idx` keeps its draw independent of the simulation
batch size.

(sbi-photometry)=
### Using the photometry

The returned arrays keep all generated events, including those the survey
would reject. Select events with `passing_mask` and epochs with `valid_mask`.
For example, after the call above you can extract the observed light curve
of the first accepted event:

```python
import numpy as np

accepted = np.flatnonzero(np.asarray(selection["passing_mask"]))
if accepted.size:
    i = accepted[0]
    valid = (np.asarray(selection["valid_mask"][i])
             & ~np.asarray(selection["saturated_mask"][i]))
    flux = np.asarray(selection["fluxcals_noisy_sat"][i])
    error = np.asarray(selection["fluxcalerrs_sat"][i])
    valid = valid & np.isfinite(flux) & np.isfinite(error)
    mjd = np.asarray(results["observation_schedule"]["mjds"][i])[valid]
    band = np.asarray(results["observation_schedule"]["filter_indices"][i])[valid]
    flux, error = flux[valid], error[valid]
```

`mjd`, `band`, `flux` and `error` now have one entry per usable observation.
The band values index `simulator.filter_names`. Fluxes use the simulation's
`ZP_FLUXCAL`, which defaults to 31.4. When passing them to the Python fitter,
use `zp=simulator.config.instrument.zp_fluxcal`.

This NumPy extraction is for inspecting results outside a compiled function.
Inside JAX code, keep the padded arrays and apply the masks when computing
your summary statistic, so the array shapes remain fixed between surveys.

### Output arrays

With `n` generated events and at most `e` observations per event, the main
arrays are:

| array | shape | contents |
| --- | --- | --- |
| `results["fluxcals"]` | `(n, e)` | Noise-free calibrated flux |
| `selection["fluxcals_noisy_sat"]` | `(n, e)` | Observed flux |
| `selection["fluxcalerrs_sat"]` | `(n, e)` | Reported flux uncertainty |
| `results["observation_schedule"]["mjds"]` | `(n, e)` | Observation times |
| `results["observation_schedule"]["filter_indices"]` | `(n, e)` | Filter indices, with -1 for padding |
| `selection["passing_mask"]` | `(n,)` | Events passing the survey selection |
| `selection["valid_mask"]` | `(n, e)` | Valid observations, including saturated epochs |
| `selection["saturated_mask"]` | `(n, e)` | Saturated observations to exclude from the extracted light curve |
| `selection["detection_mask"]` | `(n, e)` | Observations detected by the pipeline |

`results["parameters"]` contains the generated parameters, including true
redshifts, peak times and model parameters. The schedule also contains the
observing conditions. Padding and observations outside the model's wavelength
coverage can have NaN photometry, so retain the masks when passing arrays to
your inference code.

### Applying a host-mass step

Host properties are available in `adjust` with the prefix `host_`. For a
simulation configured with a {ref}`HOSTLIB <simulating-hosts>` containing
`LOGMASS`, this example adds a 0.06 mag step at a host stellar mass of
$10^{10}$ solar masses. Events above that mass become 0.03 mag brighter,
and those below it 0.03 mag fainter. `magoff` is an additive magnitude
offset applied to each event.

```python
def mass_step(key, params):
    step = jnp.where(params["host_LOGMASS"] > 10.0, -0.03, 0.03)
    return {**params, "magoff": params["magoff"] + step}

results, selection, record = pipeline_fn(jax.random.PRNGKey(1),
                                         adjust=mass_step)
sim_run.finish_batch(simulator, results, record,
                     start_index=0, n_events=simulator.n_sne)
```

### Compilation and random numbers

The prepared simulator fixes the batch size and survey tables. Changing
array shapes or dtypes, or supplying a different
`adjust` function, can require recompilation. The compiled wrapper in the
population example allows its numerical arguments to vary between calls.
For unchanged settings, two calls with the same random key give the same
survey. Pass a new key for an independent realisation.

## Relation to the analysis pipeline

{doc}`building_pipelines` describes running the simulation through a `Sim`
stage, which writes a data version for subsequent light-curve fitting,
including spectra and selection summaries when requested. The SBI interface
instead returns arrays in memory.

Spectroscopic efficiency maps based on `SNRSUM_REST_V` require generated
spectra and are supported through the analysis pipeline. The SBI interface
accepts maps based on photometric quantities.

The SBI function generates the specified number of events before selection.
An ordinary simulation can instead use `NGEN_LC` to keep generating until
the requested number of events pass selection.

You can fit the simulated light curves with the
{doc}`Python light-curve fitter <fitting_light_curves>`, using the masks,
band names and flux zero point described above. For simulations and fits
using files on disk, see {doc}`building_pipelines`.

## Differentiability

For a fixed observing schedule and template choice, JAX can differentiate
the photometry with respect to continuous model parameters.

Host and template selection, observation scheduling, detection and follow-up
decisions, and selection cuts involve discrete choices.
