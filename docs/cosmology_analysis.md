# Running a cosmology analysis

jax-SNANA can run simulations, light-curve fitting, standardisation, covariance
construction and cosmology fitting together in a Python script. The example
below connects these stages using SALT3.DOVEKIE and the survey inputs from
{doc}`getting_started`.

(adding-biascor-sample)=
## Example analysis

This example generates two independent type Ia samples: 600 supernovae to
analyse and 6000 for the BIASCOR simulation. Both samples are fitted with the
same light-curve settings. BBC applies the bias corrections, and the corrected
distances are passed to the covariance and cosmology fits.

After {ref}`preparing the example inputs <installation-examples>`, run from
the repository root:

```bash
python examples/quickstart/analysis_biascor.py
```

The analysis is defined by these stages:

```python
from jax_pipeline import BBC, Cosmology, Covariance, Fit, Sim, run

inputs = "examples/quickstart"

survey = Sim("sim", f"{inputs}/sim.input",
             overrides={"NGENTOT_LC": 600, "RANSEED": 1,
                        "GENMAG_SMEAR": 0.10})
biascor = Sim("biascor", f"{inputs}/sim.input",
              overrides={"NGENTOT_LC": 6000, "RANSEED": 2,
                         "GENMAG_SMEAR": 0.10})

fit = Fit("fit", f"{inputs}/fit.nml", datasets=[survey, biascor])

distances = BBC("distances", data=fit.version(survey), biascor=fit.version(biascor),
                input_file=f"{inputs}/bbc_biascor.input")
covariance = Covariance("covariance", distances=distances)
cosmology = Cosmology("cosmology", covariance=covariance,
                      overrides={"model": "lcdm", "method": "grid"})

results = run(cosmology, outdir="run_biascor")
print(results["cosmology"]["ALL"]["point_estimate"])
```

The data and BIASCOR samples share one `Fit` stage, using the same fitting
settings and model preparation. `fit.version` selects each sample's fitted
results for BBC. Passing `cosmology` to `run` executes all
the stages it depends on in order.

The BBC input, `examples/quickstart/bbc_biascor.input`, uses three redshift
bins and a redshift-only bias correction (`opt_biascor=1`). It holds α and β
at their simulated values of 0.14 and 3.1 and fixes the intrinsic scatter at
0.10 mag. The {doc}`standardisation page <bias_corrections>` describes these
settings. The covariance contains statistical uncertainties only, and the
cosmology stage fits flat ΛCDM without an informative matter-density prior.

For a multidimensional SALT correction with interpolation over α and β,
`examples/quickstart/analysis_biascor_grid.py` provides a further example.
{doc}`systematics` explains how to add FITOPT and MUOPT variations and
propagate them into the covariance.

### Results

The results are written under `run_biascor/`, with one directory per stage.
The script's `--outdir` option selects another output directory. The
`distances` directory contains the binned `.M0DIF` and unbinned `.FITRES`
Hubble diagrams. The covariance and cosmology outputs are under
`covariance/` and `cosmology/`.

The same results are available in Python:

```python
hubble_diagram = results["distances"]["nominal"]["M0DIF"].df
cosmology_fit = results["cosmology"]["ALL"]
parameters = cosmology_fit["point_estimate"]
```

{doc}`analysis_outputs` describes the output tables and cosmology summaries.
Running the script again reuses completed stages whose inputs are unchanged;
{doc}`building_pipelines` explains how to rerun selected stages.

## Using your own observations

Use a `Data` stage for existing light curves and a `Sim` stage for the
BIASCOR simulation. The simulation input specifies your survey's population,
observing conditions and selection; the light-curve fitting input specifies
the model, calibration and fitting settings. The BBC input supplies the
standardisation and bias-correction settings.

```python
from jax_pipeline import BBC, Cosmology, Covariance, Data, Fit, Sim, run

observed = Data("observed", "/path/to/MY_DATA")
biascor = Sim("biascor", "sim_biascor.input")
fit = Fit("fit", "fit.nml", datasets=[observed, biascor])

distances = BBC("distances", data=fit.version(observed), biascor=fit.version(biascor),
                input_file="bbc.input")
covariance = Covariance("covariance", distances=distances)
cosmology = Cosmology("cosmology", covariance=covariance,
                      input_file="cosmology.yaml")
results = run(cosmology, outdir="analysis")
```

Leave `datafile`, `simfile_biascor` and `prefix` out of `bbc.input`, since the
pipeline supplies them. Use the same fitting and analysis selection for the
data and BIASCOR samples. If classification probabilities enter that
selection, classify the BIASCOR sample too; see {doc}`classification`.

`examples/cosmology_pipeline.py` shows an analysis with core-collapse
contaminants, classification and several systematic variations. To run an
existing Pippin analysis, see {doc}`snana_and_pippin`.
