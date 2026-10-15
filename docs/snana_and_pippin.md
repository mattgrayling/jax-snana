# Running a Pippin analysis

jax-SNANA supports analysis pipelines configured using Pippin, including
simulations, light curve fitting, standardisation and cosmological inference.
Existing Pippin configuration files can be run directly with `jax_pipeline`,
using the simulation and fitting inputs and systematic variations specified
in the configuration.

## Running a Pippin configuration

`jax_pipeline` runs the tasks in dependency order within one process:

```bash
jax_pipeline analysis.yml --outdir run_pippin
```

## Example configuration

This example uses the SALT inputs from {doc}`getting_started` and
{doc}`cosmology_analysis`. It generates independent data and BIASCOR samples,
fits both, and propagates a zero-point variation and a change to the
Hubble-residual χ² cut into the cosmology fit. The BIASCOR sample uses the
nominal light curve calibration. This small example uses redshift-only
bias corrections and holds α and β fixed through `bbc_biascor.input`.

After {ref}`preparing the example inputs <installation-examples>`, save
this as `analysis.yml` in the repository root:

```yaml
GLOBAL:
  DATA_DIRS:
    - examples/quickstart

SIM:
  DATA:
    IA:
      BASE: sim.input
    GLOBAL:
      NGENTOT_LC: 600
      GENMAG_SMEAR: 0.10
      RANSEED_REPEAT: 1 1
  BIAS:
    IA:
      BASE: sim.input
    GLOBAL:
      NGENTOT_LC: 6000
      GENMAG_SMEAR: 0.10
      RANSEED_REPEAT: 1 2

LCFIT:
  DATAFIT:
    BASE: fit.nml
    MASK: DATA
    FITOPTS:
      - /zpt_g/ MAGOBS_SHIFT_ZP g 0.003
  BIASFIT:
    BASE: fit.nml
    MASK: BIAS

BIASCOR:
  BBC:
    BASE: bbc_biascor.input
    DATA: [DATAFIT_DATA]
    SIMFILE_BIASCOR: [BIASFIT_BIAS]
    MUOPTS:
      CHI2MAX:
        OPTS:
          chi2max: 16

CREATE_COV:
  COV:
    MASK: BBC
    OPTS:
      COVOPTS:
        - '[NOSYS] [=DEFAULT,=DEFAULT]'

COSMOFIT:
  WFIT:
    WF:
      MASK: COV
      OPTS:
        WFITOPTS:
          - /lcdm/ -lcdm --method grid
```

Run it with the command above. `GLOBAL: DATA_DIRS` supplies the directory
containing the input files named by `BASE`. `RANSEED_REPEAT: 1 1` requests
one simulation with seed 1; the BIASCOR sample uses seed 2.

`MASK` selects upstream tasks by matching their names. Here,
`MASK: DATA` selects the `DATA` simulation for `DATAFIT`, while
`MASK: BIAS` selects the `BIAS` simulation for `BIASFIT`.
`DATAFIT_DATA` is the fit produced by the `DATAFIT` task for the `DATA`
simulation; `BIASFIT_BIAS` is the corresponding BIASCOR fit.
`DATA` and `SIMFILE_BIASCOR` connect these to BBC. `FITOPTS` and `MUOPTS`
define the variations described in {doc}`systematics`.

`CREATE_COV` constructs the covariance matrices from the BBC results.
The `ALL` covariance includes the configured systematics; the additional
`NOSYS` choice includes statistical uncertainties only. `COSMOFIT` fits
ΛCDM with each covariance.

For other inputs, `--data-dir PATH` adds a directory to search before those
in `GLOBAL: DATA_DIRS`; repeat it to search several directories. Paths may
contain environment variables. Add `--double-precision` to run the whole
analysis in double precision.

## Checking and rerunning an analysis

To validate the input settings and inspect the planned stages without
running the analysis, use:

```bash
jax_pipeline analysis.yml --check
```

This checks the configuration and stage input files without creating an
output directory. It does not initialise the models or check generated
light curves and fitted results.

Repeating a run in the same output directory reuses completed stages whose
recorded inputs have not changed. `--refresh` forces selected stages and
all stages depending on them to rerun. Use the stage names printed by
`--check`, for example:

```bash
jax_pipeline analysis.yml --outdir run_pippin --refresh BBC
```

`--refresh-all` reruns the whole analysis. See {ref}`pipeline-reruns` for
more on how changes to inputs affect reruns.

## Running in a Python script

The same configuration can be run from Python:

```python
from jax_pipeline.pippin import run_pippin

results = run_pippin("analysis.yml", outdir="run_pippin")
```

To modify the stages in Python before running them, use `read`:

```python
from jax_pipeline import run
from jax_pipeline.pippin import read

stages = read("analysis.yml")
results = run(*stages, outdir="run_pippin")
```

{doc}`building_pipelines` describes how to configure stages in Python.

## Results

Each stage writes its results under the output directory. Analyses that
include cosmology fitting also produce `COSMOLOGY_SUMMARY.csv`, collecting
the cosmological parameter estimates and uncertainties for the covariance
choices and analysis realisations. These retain any blinding applied by
the cosmology fit.

See {doc}`analysis_outputs` for the output files and
{doc}`snana_compatibility` for scientific differences from SNANA.
The supported Pippin tasks and their settings are listed in
{doc}`pippin_settings`.
