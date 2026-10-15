# Getting started

This example simulates type Ia supernova light curves using DES observing
conditions, then fits the simulated light curves. The commands below use the
example inputs included in the repository.

Before running them, follow {ref}`installation-examples` to get the example
files and set `SNDATA_ROOT`. Run the commands from the repository root.
To fit your own observations directly in a Python script, see
{ref}`fit-from-python`.

## Simulating light curves

Choose either the BayeSN G26 or SALT3.Dovekie example. Both generate 300
supernovae over redshifts 0.05–1.2, with DES *griz* observing times and
conditions supplied by a SIMLIB file.

For BayeSN G26:

```bash
snlc_sim examples/quickstart/sim_bayesn.input
```

For SALT:

```bash
snlc_sim examples/quickstart/sim.input
```

The simulation writes the light curves passing its selection cuts to
`QUICKSTART_BAYESN/`, or `QUICKSTART/` for SALT.

## Fitting light curves

Use the matching fit input to fit the simulated light curves.

For BayeSN:

```bash
snlc_fit examples/quickstart/fit_bayesn.nml
```

For SALT:

```bash
snlc_fit examples/quickstart/fit.nml
```

The first batch includes JAX compilation, so it takes longer than subsequent
batches. The fitter prints a summary of the fits and the number of events
passing the selection cuts.

The fitted parameters and uncertainties are written to
`QUICKSTART_BAYESN.FITRES.TEXT`, or `QUICKSTART.FITRES.TEXT` for SALT.
Read this table with pandas:

```python
import pandas as pd

prefix = "QUICKSTART_BAYESN"  # "QUICKSTART" for SALT
fitres = pd.read_csv(f"{prefix}.FITRES.TEXT", sep=r"\s+", comment="#")
print(fitres.head())
```

(quickstart-bayesn-results)=
### BayeSN results

The fitted shape and host extinction are `THETA` and `AV`, with errors
`THETAERR` and `AVERR`. `MU_LCFIT` is the inferred distance modulus and
`MUERR_LCFIT` its uncertainty. The G26 fit also estimates the host-dust
parameter `RV`, with uncertainty `RVERR`.

### SALT results

The columns `x1`, `c` and `mB`, with their errors and covariances, give the
fitted shape, colour and peak brightness. These enter the standardisation
relation described in {doc}`bias_corrections`.

{doc}`fitting_light_curves` describes the fit settings and output summaries.

(inspect-first-fit)=
## Inspecting a fitted light curve

Generate the plotting tables from the saved BayeSN fit:

```bash
lcplot_from_fitres examples/quickstart/fit_bayesn.nml QUICKSTART_BAYESN.FITRES.TEXT \
    --prefix QUICKSTART_BAYESN
```

For SALT:

```bash
lcplot_from_fitres examples/quickstart/fit.nml QUICKSTART.FITRES.TEXT \
    --prefix QUICKSTART
```

This evaluates the model at the saved fitted parameters without refitting.

With Matplotlib installed (`pip install matplotlib`), plot the first event:

```python
import matplotlib.pyplot as plt
import pandas as pd

prefix = "QUICKSTART_BAYESN"  # "QUICKSTART" for SALT
curves = pd.read_csv(f"{prefix}.LCPLOT.TEXT", sep=r"\s+", comment="#",
                     dtype={"CID": str})
cid = curves["CID"].iloc[0]
event = curves[curves["CID"] == cid]
fig, ax = plt.subplots()
for band, rows in event.groupby("BAND"):
    observed = rows[rows["DATAFLAG"] == 1]
    model = rows[rows["DATAFLAG"] == 0].sort_values("MJD")
    line, = ax.plot(model["MJD"], model["FLUXCAL"], label=band)
    ax.errorbar(observed["MJD"], observed["FLUXCAL"],
                yerr=observed["FLUXCAL_ERR"], fmt="o", color=line.get_color())
ax.set(xlabel="MJD (days)", ylabel="Flux (FLUXCAL)", title=f"Event {cid}")
ax.legend(title="Filter")
plt.show()
```

The points are observations marked as used, with their measurement errors;
the lines are the fitted model in each filter. `FLUXCAL` is calibrated flux
on the common scale described in {doc}`light_curve_data`.

## Running in a Python script

The same simulation and fit can be run together in a Python script using
`jax_pipeline`:

```python
from jax_pipeline import Fit, Sim, run

simulation_input = "examples/quickstart/sim_bayesn.input"
fit_input = "examples/quickstart/fit_bayesn.nml"
survey = Sim("sim", simulation_input)
fitted = Fit("fit", fit_input, datasets=[survey])
results = run(fitted, outdir="run")
fitres = results["fit"]["sim"]["nominal"].df
```

For SALT, use `examples/quickstart/sim.input` and
`examples/quickstart/fit.nml`. Subsequent
runs reuse completed stages whose inputs are unchanged. This script is also
provided as `examples/quickstart/analysis_bayesn.py`.
{doc}`building_pipelines` describes how to combine further analysis stages.

## Next steps

To fit your own observations, see {ref}`fit-from-python` for arrays or
tables, or {ref}`fit-existing-observations` for SNANA datasets.
For your own survey, see {doc}`simulations`; {doc}`supernova_populations`
explains how to change the simulated population, including the settings used
in this BayeSN example.

Both models require a correction for selection and fitting biases when
their distances are used for cosmology. {doc}`bias_corrections` explains
this step, including how BayeSN distances and SALT parameters enter it.
{doc}`cosmology_analysis` connects the stages in a complete Python script.
For an existing Pippin configuration, see {doc}`snana_and_pippin`.
