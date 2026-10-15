# Building pipelines

`jax_pipeline` allows survey simulations, light curve fitting, standardisation
and cosmological inference to be run together in a Python script. A pipeline
can fit observed light curves or use saved results from earlier light curve fits.

## Connecting stages

This example simulates BayeSN light curves and fits them using the inputs
from {doc}`getting_started`. Run it from the repository root after preparing
the example inputs:

```python
from jax_pipeline import Fit, Sim, run

sim = Sim("sim", "examples/quickstart/sim_bayesn.input",
          overrides={"NGENTOT_LC": 5000})
fit = Fit("fit", "examples/quickstart/fit_bayesn.nml", datasets=[sim])
results = run(fit, outdir="run")
table = results["fit"]["sim"]["nominal"]
```

`Fit` takes the fitting input file and the datasets to fit. Several datasets
can share one `Fit` stage, reusing model initialisation and compilation.
To pass one dataset's fits to a later stage, use `fit.version(dataset)`;
when fitting a single dataset, the `Fit` stage itself can be supplied.

Calling `run(fit)` runs the simulation, prepares the fitter and fits the
light curves. If several stages use the same `Sim` stage, its simulation
runs once and they share its results. The returned `results` contains each
stage's result under its name; `"nominal"` selects the fit without
systematic variations. {doc}`cosmology_analysis` provides a complete
example including standardisation and cosmological inference.

### Input overrides

`overrides` sets input values for a stage without editing its input file.
In the example above, it requests 5000 generated events. For these SNANA
inputs, values with several components can be supplied as space-separated
strings, for example:

- `"GENRANGE_REDSHIFT": "0.05 0.5"` in a simulation override;
- `"FITWIN_TREST": "-10.0 40.0"` in a fitting override;
- `"MAGOBS_SHIFT_ZP": "ALL 0.003"` for a zero-point shift in every fitted band.

Systematic variations are configured separately through `fitopts` on a
`Fit` and `muopts` on a `BBC` stage. {doc}`systematics` gives examples.

(combining-pipeline-simulations)=
### Combining simulation inputs

To combine type Ia and core-collapse supernovae in one simulated sample,
supply their simulation inputs together:

```python
survey = Sim("survey", input_files=["ia.input", "cc.input"],
             overrides=[{"NGENTOT_LC": 1200}, {"NGENTOT_LC": 400}])
```

`input_files` is one path or a list of paths. To change settings separately
for each input file, supply a list of dictionaries in `overrides`, in the
same order as the files. A single dictionary applies the same changes to
every input file.

The stage writes the combined sample as one SNANA dataset.
{ref}`combined-simulation-files` describes its files and event
identifiers.

(analysis-existing-files)=
## Starting from existing files

To fit existing observations, use a `Data` stage in place of the simulation:

```python
from jax_pipeline import Data

observed = Data("des_data", "/path/to/DES5YR_DATA")
observed_fit = Fit("data_fit", "fit.nml", datasets=[observed])
```

To start from light curve fits produced by jax-SNANA or SNANA, supply
their FITRES file to `Fit`:

```python
tables = Fit("des_fits", fitres="/path/to/FITOPT000.FITRES")
```

To import all FITOPTs from a SNANA batch fit, pass the directory containing
its `FITOPT*.FITRES` files to `fitres`. Their labels are read from
`FITOPT_LIST` in `SUBMIT.INFO`, which must be beside the version directories.

To use completed BBC results with the command-line covariance tool, see
{ref}`exporting BBC results <covariance-export-bbc>`. Within a pipeline,
the covariance stage reads the BBC results directly.

## Results

For the simulation-and-fit example above, the fitted parameters and
uncertainties can be accessed as a pandas DataFrame:

```python
fitted = results["fit"]["sim"]["nominal"].df
metadata = results["fit"]["sim"]["nominal"].meta
```

Fit and BBC tables provide `.df` for their data and `.meta` for header
metadata. Simulation and analysis outputs are saved under
`<outdir>/<stage_name>`; {doc}`analysis_outputs` describes the files.
The {doc}`Python API <api>` lists the results returned by each stage.

For generated fits, results are indexed by dataset name and then FITOPT.
A `Fit` reading an existing FITRES file instead returns the FITOPT tables
directly, without a dataset-name level.

(pipeline-reruns)=
## Reusing results and rerunning stages

Repeating a run in the same output directory reuses completed results.
If a stage's configuration or input data change, that stage and all later
stages that use its results are rerun. This includes edits to input files
supplied directly to a stage and files in a supplied dataset directory. In the
{doc}`cosmology walkthrough <cosmology_analysis>`, raising
`NGENTOT_LC` reruns the whole chain, while changing `method` reruns only the
cosmology fit.

When several datasets share a `Fit`, changing any one of them reruns the
fits of every dataset in that stage. An interrupted fit resumes from its
completed datasets when run again with unchanged inputs.

Updating jax-SNANA does not automatically rerun a completed analysis.
Neither does editing a {ref}`SIMLIB <format-simlib>`,
{ref}`calibration file <format-kcor>` or model file named inside a stage's
input file.

Use `refresh` to force stages to run again, even when their recorded settings
and inputs have not changed:

```python
run(fit, outdir="run", refresh="sim")
```

This reruns `sim` and every stage that depends on it. `refresh` accepts a
stage name or a sequence of names; `refresh=True` reruns the whole analysis.

## Precision

Python pipelines use the active JAX precision setting. To use double
precision for the whole pipeline, call `configure_precision` before
preparing any stages or arrays:

```python
from jax_pipeline import configure_precision

configure_precision(True)
```

Changing precision reruns the analysis and changes the simulation's random
realisation, even with the same seed. Batch sizes and simulation seeds are
set on individual stages or in their input files, as described in
{doc}`simulating` and {doc}`fitting_light_curves`.

## Logging and run records

To show stage progress and the simulation and fitting logs in a Python
script, configure logging before calling `run`:

```python
import logging

logging.basicConfig(format="%(message)s", level=logging.INFO)
```

The command-line tools configure logging themselves. The pipeline reports
which stages it runs and which it loads from saved results, followed by the
output locations at completion.

`MANIFEST.yml` records the stage settings and execution status, including
whether results were reused. It also records the code version used for
the run and, where available, the version that produced each stage's results.

To define additional analysis stages, see {ref}`custom-pipeline-stage` in the
Python API guide. {doc}`classification` describes classifier training and
prediction stages; {doc}`snana_and_pippin` covers pipelines configured
through Pippin YAML.
