# Python API

This page documents the Python interfaces' arguments and return values.
For worked examples, see {doc}`fitting_light_curves`, {doc}`simulating` and
{doc}`building_pipelines`.

## Survey simulations

`simulate` runs a survey simulation from an input file or settings supplied
as Python keywords, returning the simulated parameters and photometry,
selection arrays and any requested spectra. See {doc}`simulating` for
survey inputs and examples of running simulations in a script. For repeated
compiled survey realisations inside an inference method, see {doc}`sbi`.

```{eval-rst}
.. autofunction:: jax_snlc_sim.simulate
```

### Survey input caches

Parsed survey inputs are cached outside the output directory for reuse by
subsequent runs:

- A SIMLIB is cached as an HDF5 file under `simlib_cache/`, beside the
  installed `jax_snlc_sim` package. For a source checkout this is the
  repository root; for a pip installation it is usually in `site-packages`.
- A HOSTLIB is cached as a parquet file under the platform's cache directory
  for `jax_snana`, `~/Library/Caches/jax_snana/hostlib_cache` on macOS.

Both are keyed on the path of the source file and rebuilt when that file is
newer than the cache.

## Light-curve fits

`LightCurveFitter` fits observations held in arrays or tables. See
{doc}`fitting_light_curves` for examples, filter calibration, priors and
repeated fits.

```{eval-rst}
.. autoclass:: jax_snlc_fit.LightCurveFitter
   :class-doc-from: class
   :no-members:
   :exclude-members: __new__

.. automethod:: jax_snlc_fit.LightCurveFitter.fit

.. automethod:: jax_snlc_fit.LightCurveFitter.predict

.. automethod:: jax_snlc_fit.LightCurveFitter.reset_shape

.. automethod:: jax_snlc_fit.LightCurveFitter.configuration

```

### Fit results

```{eval-rst}
.. py:class:: jax_snlc_fit.fitter.BatchFitResult

   Fitted parameters, uncertainties, covariances and fit diagnostics.
```

The main attributes are:

| Attribute | Contents |
|---|---|
| `snids` | Object identifiers, in result order. |
| `params` | Mapping of fitted parameter names to arrays of shape `(n_sne,)`. |
| `param_errors` | Reported uncertainties, with the same names and shapes as `params`. |
| `param_covariances` | Covariance arrays of shape `(n_sne, n_cov, n_cov)`, ordered by `cov_output_params`. |
| `cov_output_params` | Names of the parameters included in the covariance arrays. |
| `chi_squared`, `ndof` | Final fit objective, including priors, and degrees of freedom. |
| `observation_mask` | Observations used in each fit, shape `(n_sne, max_obs)`. |
| `outcomes` | DataFrame recording acceptance or rejection for the input objects, in input order. `fit_index` maps to the result arrays; -1 means the object did not reach fitting. |

`n_sne` is the number of fitted objects in the result. `outcomes` also
includes objects rejected during preparation. Use the following methods to
select objects, combine results or write fitted tables:

```{eval-rst}

.. automethod:: jax_snlc_fit.fitter.BatchFitResult.to_table

.. automethod:: jax_snlc_fit.fitter.BatchFitResult.write_table

.. automethod:: jax_snlc_fit.fitter.BatchFitResult.take

.. automethod:: jax_snlc_fit.fitter.BatchFitResult.filter_by_mask

.. automethod:: jax_snlc_fit.fitter.BatchFitResult.concatenate
```

(fit-table-inputs)=
### Photometry and object tables

The photometry table has one row per observation, with `snid`, `mjd`, `flux`,
`flux_err` and `band`. It can also hold object properties repeated across
observations, or these can be supplied in a separate `objects` table with
one row per `snid`:

```python
result = fitter.fit(photometry, objects=objects, zp=31.4)
```

Object columns include `redshift`, `mwebv`, `redshift_err`, `redshift_cmb`,
`peakmjd`, `photoz`, `photoz_err`, `ra`, `dec` and `zp`. Redshift can be omitted
when fitting photometric redshift. Values must agree across an object's
observations. Results follow the `objects` table when supplied, or the order
of first appearance in `photometry` otherwise.

Use `column_map={"flux": "measured_flux"}` to match different column names.
Object-table `zp` and `mwebv` values take precedence over the corresponding
arguments to `fit`; those arguments supply values for objects without their
own. A `zp` column allows the zero point to vary between objects.

## Distance fits

`fit_distances` runs BBC from fitted light-curve tables or FITRES files. Keyword
settings use the same names as the YAML inputs, including settings specific to
the chosen model.
See {doc}`bias_corrections` for bias-correction inputs and nuisance parameters.

```python
from jax_salt2mu import fit_distances

result = fit_distances("examples/quickstart/bbc_biascor.input",
                       datafile=["data.FITRES"],
                       simfile_biascor=["biascor.FITRES"])
```

This uses the SALT settings from {doc}`cosmology_analysis`.

For tables already in memory, use `data`, `biascor` and `ccprior`. They accept
`FitresTable` or pandas DataFrames; `FitresTable` also carries header metadata.
`result.fit` holds the fitted parameters and uncertainties, `result.output`
holds the distances, and `result.hubble_diagram` is a DataFrame with the binned
M0DIF columns. `result.accepted_mask` selects the fitted rows in `result.data`.
For example, extract the accepted events and their distances with:

```python
import pandas as pd

distances = pd.DataFrame({
    "CID": result.data.name[result.accepted_mask],
    "MU": result.output.mu,
    "MUERR": result.output.muerr,
})
parameters = result.fit.nuisance
parameter_errors = result.fit.nuis_err
```

Files are written only with `write=True`, which requires `prefix`.

```{eval-rst}
.. autofunction:: jax_salt2mu.fit_distances

.. autoclass:: jax_salt2mu.DistanceFitResult
   :members:
```

(covariance-python-api)=
## Covariance construction

`compute_covariance` accepts a covariance configuration file, a settings mapping,
or keyword settings. To work with Hubble diagrams in memory, supply a mapping
of the nominal and varied diagrams, together with the FITOPT and MUOPT
labels and uncertainty scales:

```python
from jax_create_cov import compute_covariance

covariance = compute_covariance(
    diagrams={"FITOPT000_MUOPT000": nominal,
              "FITOPT001_MUOPT000": zpt_g},
    fitopt_scales={0: ("DEFAULT", 1.0), 1: ("zpt_g", 1.0)},
    muopt_scales={0: ("DEFAULT", 1.0)},
    write_mask_cov=8, isdata_real=1,
)
```

`nominal` and `zpt_g` are the reference and varied DataFrames. Each diagram is a pandas DataFrame with the M0DIF or FITRES columns appropriate
for the selected mode.
The returned `CovResult` holds `hubble_diagram` and one `covariances[label]`
entry per covariance option. Dense entries have `kind="systematics"` and a
`matrix` array. With `write_mask_cov=8`, factorized entries have
`kind="factorized"`, statistical variances `diag`, systematic vectors `U`,
and optional `vec_labels`. Pass a factorized entry directly to
`hubble_diagram_from_covariance` without constructing a dense covariance.
`covtot_inv(label)` constructs the inverse total covariance.
Use `write=True, outdir="covariance"` to save outputs. For configuration
and systematic definitions, see {doc}`systematics`.

```{eval-rst}
.. autofunction:: jax_create_cov.compute_covariance
```

## Cosmology fits

`fit_cosmology` accepts keyword settings, a mapping, a `WFITConfig` or a YAML
configuration file. It returns a tuple with one result dictionary per independent
Hubble diagram:

```python
from jax_wfit import fit_cosmology

fits = fit_cosmology(hubble_diagram="distances.M0DIF", model="wcdm")
parameters = fits[0]["point_estimate"]
summaries = fits[0]["summaries"]
```

Using the factorized `covariance` result from the example above, fit its
Hubble diagram and total covariance together:

```python
from jax_wfit import fit_cosmology, hubble_diagram_from_covariance

diagram = hubble_diagram_from_covariance(
    covariance.hubble_diagram, covariance.covariances["ALL"],
    covariance.isdata_real, source="my_analysis",
)
fits = fit_cosmology(diagrams=[diagram], model="wcdm")
```

File writing requires `write=True`. Fits of in-memory diagrams also require
explicit `cospar_yaml` and `outfile_cospar` paths when writing. See
{doc}`cosmology` for the inference methods, priors and result summaries.

```{eval-rst}
.. autofunction:: jax_wfit.fit_cosmology
```

JAX precision is chosen for the whole Python process. Call
`jax_pipeline.configure_precision(True)` before constructing arrays or preparing
stages when the analysis requires double precision.
If omitted, `double_precision` uses the active precision in Python. An explicit
request must match it; a conflicting request raises an error. Command-line runs
default to single precision and configure JAX from their inputs.

## Pipeline stages

Create the stages you need, then pass the final stage or stages to `run`.
For real data, start with `Data`; for a simulation, start with `Sim`.
`Fit` applies shared fitting settings to the datasets in its `datasets`
argument. Use `fit.version(dataset)` to pass one dataset's fits to a later
stage when fitting several datasets together.

```{eval-rst}
.. autoclass:: jax_pipeline.stages.Data
   :no-members:
   :exclude-members: __new__

.. autoclass:: jax_pipeline.stages.Sim
   :no-members:
   :exclude-members: __new__

.. automethod:: jax_pipeline.stages.Sim.configuration

.. autoclass:: jax_pipeline.stages.Fit
   :no-members:
   :exclude-members: __new__

.. automethod:: jax_pipeline.stages.Fit.configuration

.. automethod:: jax_pipeline.stages.Fit.version

.. autoclass:: jax_pipeline.stages.FitVersion
   :no-members:
   :exclude-members: __new__

.. autoclass:: jax_pipeline.stages.BBC
   :no-members:
   :exclude-members: __new__

.. automethod:: jax_pipeline.stages.BBC.export_covariance_inputs

.. autoclass:: jax_pipeline.stages.Covariance
   :no-members:
   :exclude-members: __new__

.. automethod:: jax_pipeline.stages.Covariance.configuration

.. autoclass:: jax_pipeline.stages.Cosmology
   :no-members:
   :exclude-members: __new__

.. autofunction:: jax_pipeline.run.run

.. autofunction:: jax_pipeline.run.cosmology_summary

.. autofunction:: jax_snlc_core.utils.configure_precision
```

## Classification

{doc}`classification` explains how to train a classifier, attach predictions
and use them in the distance fit.

```{eval-rst}
.. autoclass:: jax_pipeline.classify.Classify
   :no-members:
   :exclude-members: __new__

.. autoclass:: jax_pipeline.classify.Train
   :no-members:
   :exclude-members: __new__

.. autoclass:: jax_pipeline.supernnova.SuperNNova
   :no-members:
   :exclude-members: __new__

.. automethod:: jax_pipeline.supernnova.SuperNNova.train

.. automethod:: jax_pipeline.supernnova.SuperNNova.predict

.. autoclass:: jax_pipeline.classify.Perfect
   :no-members:
   :exclude-members: __new__

.. automethod:: jax_pipeline.classify.Perfect.train

.. automethod:: jax_pipeline.classify.Perfect.predict
```

## Reading Pippin configurations

{doc}`snana_and_pippin` explains how to read an existing Pippin configuration
and run the stages it defines. `read` returns the stages, which can be
inspected before passing them to `run`. `check` takes the configuration file
and returns its execution plan without running the analysis.

```{eval-rst}
.. autofunction:: jax_pipeline.pippin.read

.. autofunction:: jax_pipeline.pippin.check
```

## Reading fitted tables

A pipeline stage returns fitted tables as `FitresTable` objects. The `df`
attribute holds the rows as a pandas DataFrame and `meta` holds the header
metadata. Use `read_fitres` for an existing file; event identifiers are
preserved as strings, including leading zeros.

```{eval-rst}
.. autoclass:: jax_snlc_core.snana_files.FitresTable
   :members:

.. autofunction:: jax_snlc_core.snana_files.read_fitres
```

## Simulation-based inference

Pass an input file to `build_sbi_pipeline` to load a survey once and run
repeated compiled realisations. {doc}`sbi` shows how to select events,
inspect photometry and change the generated parameters.

For a complete simulation, `run_simulation` returns the event results, selection,
and spectra for the selected events. It completes spectra and atmospheric effects
before returning. Pass those spectra as `spectra_by_sn` to `save_snana_outputs`;
saving formats and writes the completed results. A narrower output selection
also restricts the spectra written to those events.

```{eval-rst}
.. autoclass:: jax_snlc_sim.SimulationConfig
   :no-members:
   :exclude-members: __init__, __new__

.. autofunction:: jax_snlc_sim.run.prepare

.. autofunction:: jax_snlc_sim.run.run_simulation

.. autofunction:: jax_snlc_sim.run.build_sbi_pipeline

.. autofunction:: jax_snlc_sim.output.save_snana_outputs
```

(custom-pipeline-stage)=
## Adding a pipeline stage

```{eval-rst}
.. autoclass:: jax_pipeline.stages.Stage
   :members:
   :exclude-members: __new__
```

To connect another analysis step to the pipeline, define a stage as a
dataclass with:

- a `name`;
- an `inputs` method mapping input names to the stages that supply them;
- a `run` method that writes into the directory it is given and returns what
  subsequent stages use.

```python
from dataclasses import dataclass
from pathlib import Path

from jax_pipeline import Stage, run

@dataclass(eq=False)
class Summary(Stage):
    """Summarise the fitted event and redshift-bin counts."""

    name: str
    distances: Stage

    def inputs(self):
        return {"distances": self.distances}

    def run(self, inputs, outdir):
        table = inputs["distances"]["nominal"]["M0DIF"].df
        (Path(outdir) / "summary.txt").write_text(
            f"{len(table)} bins, {int(table['NFIT'].sum())} events\n")
        return self.load(inputs, outdir)

    def load(self, inputs, outdir):
        return (Path(outdir) / "summary.txt").read_text()
```

`run` receives dependency results under the keys returned by `inputs()`.
Here `inputs["distances"]` is the distance fit's result, and `.df` is the
DataFrame held by its `FitresTable`.

Implement `load` to read the saved output and return the same result as
`run`. This lets the pipeline reuse the stage's output when its settings and
inputs are unchanged. Without a `load` method, a custom stage executes on
every pipeline run.

The stage must be a dataclass, because the hash is computed from its fields.
`eq=False` keeps it distinct from another stage with identical settings, so
both run as separate stages.

Using the `distances` and `cosmology` stages from {doc}`cosmology_analysis`,
pass the summary stage to `run` alongside the stage you want results from.
`run` takes any number of stages:

```python
summary = Summary("summary", distances=distances)
results = run(cosmology, summary, outdir="run")
```

This runs both, and every stage they depend on, running shared stages only
once.

(custom-classifier)=
## Adding a classifier

A classifier is a dataclass with a `column`, the name of the probability it
produces, and a `predict` method. `predict` returns the probability for
every event as a `pandas` series indexed by `CID` as a string. The following
toy classifier illustrates the interface using a threshold on the SALT
colour parameter `c`:

```python
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

@dataclass(frozen=True)
class ColourCut:
    """Illustrative classifier returning zero or one from a colour threshold."""

    limit: float = 0.3
    column: str = "PROB_COLOURCUT"

    def predict(self, fit, photometry, model) -> pd.Series:
        probability = (fit.df["c"].abs() < self.limit).astype(float)
        return pd.Series(probability.to_numpy(),
                         index=fit.df["CID"].astype(str), name=self.column)
```

The arguments of `predict`:

- `fit`: the nominal fitted table.
- `photometry`: the data version's directory.
- `model`: the path of a trained model.

Each is `None` where the stage was not given it. A classifier that is trained
also defines

```python
    def train(self, fit, photometry, outdir) -> Path:
```

returning the path of the model it wrote under `outdir`.

`predict` may instead return a `pandas` DataFrame whose columns include
`column` and anything else the classifier reports, such as an error on the
probability. Every column is added to the tables.

The classifier must be a dataclass, because its fields are used to tell
whether its settings have changed since the predictions on disk were made.
