# Systematics

A key part of the SNANA cosmology pipeline is the treatment of systematic
uncertainty. A wide range of systematics are considered, for example
cross-calibration systematics, different treatments of Milky Way extinction
and different forward models of the supernova population. In practice, these
represent different choices that can be made within the pipeline, and are
taken into account by repeating various stages of the pipeline with different
configurations.

SNANA distinguishes variations applied in the light curve fitting stage,
known as *FITOPTs*, from those applied in the standardisation to estimate
distances, known as *MUOPTs*. The resulting shifts in distance are used to
construct a systematic covariance matrix, describing the correlated uncertainties
between fitted SN distances arising from analysis systematics, for cosmological
inference.

The examples below show the input settings and their equivalents in Python.
For running the analysis as a pipeline, see
{doc}`snana_and_pippin` or {doc}`building_pipelines`.

## Light curve fitting variations

FITOPTs specify changes to the light curve fitting process, with the entire data set
refit for each FITOPT.

One example of a systematic incorporated through FITOPTs is cross-calibration uncertainty.
Supernova cosmology analyses typically involve combinations of data sets observed with
different instruments, and even if not the SED models such as SALT and BayeSN used to fit
light curves are trained on combined data sets spanning multiple instruments. Calibration
uncertainties can be propagated by training a number of different SED models,
each with different cross-calibration solutions based on a framework such as Dovekie
([Popovic et al. 2025](https://arxiv.org/abs/2506.05471)). Each individual
FITOPT will then fit with a different trained model and set of filter
transmission wavelength- and zero point-shifts. Examples are shown below,
with *g*- and *r*-band zero points shifted by 0.003 mag, and the
*g*-band transmission shifted by 10 Å:

| FITOPT label | SNANA input setting |
|---|---|
| `zpt_g` | `MAGOBS_SHIFT_ZP = 'g 0.003'` |
| `zpt_r` | `MAGOBS_SHIFT_ZP = 'r 0.003'` |
| `filter_g` | `FILTER_LAMSHIFT = 'g 10.0'` |

The same three FITOPTs can be specified in Python as follows:

```python
from jax_pipeline import Fit

calibration = {
    "zpt_g": {"MAGOBS_SHIFT_ZP": "g 0.003"},
    "zpt_r": {"MAGOBS_SHIFT_ZP": "r 0.003"},
    "filter_g": {"FILTER_LAMSHIFT": "g 10.0"},
}

data_fit = Fit(
    "data_fit", "fit.nml", datasets=[data],
    fitopts=calibration,
)
```

Here `data` is the observed or simulated sample to be fitted, and `fit.nml`
contains the light curve fitting settings. Each FITOPT has a label and the
input settings to change. The nominal fit uses the settings in the input
file and runs in addition to the listed variations.

FITOPTs can also change other fitting settings, including the light curve
model.

In a typical analysis the data are fitted under each calibration variation,
while the bias-correction simulation is fitted once at the nominal
calibration, using a separate `Fit` stage.

## Standardisation variations

MUOPTs vary the standardisation settings or the simulation used to calculate
bias corrections. The following example uses a bias-correction simulation
with a different assumed supernova population. The BBC input setting is:

```text
simfile_biascor=alternative_biascor.FITRES
```

The same MUOPT can be specified in Python as follows:

```python
from jax_pipeline import BBC

distances = BBC(
    "distances", data=data_fit, biascor=biascor_fit,
    input_file="bbc.input",
    muopts={
        "alternative_population": {"simfile_biascor": alternative_biascor_fit},
    },
)
```

`biascor_fit` and `alternative_biascor_fit` are the fitting stages for the
nominal and alternative bias-correction simulations. `bbc.input`
contains the nominal standardisation settings. For the
`alternative_population` MUOPT, BBC uses the alternative simulation's
fitted results to calculate the bias corrections.

A MUOPT can also supply a different data sample through `datafile`, or a
contamination simulation through `simfile_ccprior`. To select particular
FITOPTs of a BIASCOR simulation, see {ref}`systematics-correction-fits`.

## Running the analysis

Individual light curve and standardisation fits are run with:

```bash
snlc_fit fit.nml
salt2mu bbc.input
```

To run the FITOPTs and MUOPTs together from a Pippin configuration, use
`jax_pipeline analysis.yml`; see {doc}`snana_and_pippin`.

By default, BBC runs the nominal analysis and every combination of the
FITOPTs and MUOPTs. To restrict which combinations are run, see
{ref}`bbc-systematic-combinations`.

Calibration variations can change which supernovae pass selection cuts.
To compare distances for the same sample across the variations, add the
following to the `&SNLCINP` block of the light curve fitting input:

```text
OPT_SNCID_LIST = 1
```

With this setting, BBC starts from the events accepted in the nominal
analysis and retains those accepted across all the requested variations.
It then fits each combination using that same set of supernovae. For a
comparison intended to vary the sample selection itself, set
`sync_events=False` on the `BBC` stage.
{ref}`bbc-event-synchronisation` describes the selection rules.

In Python, set the same option through the `Fit` stage's overrides, then run
through standardisation:

```python
from jax_pipeline import run

data_fit.overrides["OPT_SNCID_LIST"] = 1
results = run(distances, outdir="run")
nominal_distances = results["distances"]["nominal"]["M0DIF"]
calibration_distances = results["distances"]["zpt_g"]["M0DIF"]
```

Each combination produces a FITRES table of per-event results and an M0DIF
table of binned distances. Results are indexed by `"nominal"` or the
variation label; combined variations use a `(FITOPT label, MUOPT label)`
pair. {doc}`analysis_outputs` describes the output tables and directories.

(covariance-construction)=
## Covariance construction

Systematic analysis choices introduce covariance between inferred distances,
for example when a calibration uncertainty affects the supernovae observed
by one survey. The distance shifts under each variation determine its
contribution to the systematic covariance. These contributions are summed,
and the statistical distance variances are added along the diagonal.

For BBC results on disk, `create_covariance` reads the input and output
locations and covariance selections from a YAML file. For example,
`covariance.yaml` could contain:

```yaml
INPUT_DIR: BBC_OUTPUT
VERSION: DATA
OUTDIR: COVARIANCE
SYS_SCALE_FILE: systematic_scales.yml
COVOPTS:
  - '[NOSYS] [=DEFAULT,=DEFAULT]'
  - '[CALIB] [+zpt,=DEFAULT]'
```

Run this with:

```bash
create_covariance covariance.yaml
```

The input directory contains the BBC results for each FITOPT/MUOPT
combination, with labels and uncertainty scales recorded in `SUBMIT.INFO`.
See {ref}`covariance-input-layout` for the directory layout, or
{ref}`covariance-export-bbc` to export results from a Python pipeline.

`SYS_SCALE_FILE` names a YAML file of uncertainty scales. For example,
`systematic_scales.yml` could contain:

```yaml
zpt_g: 0.5
```

The scale multiplies the distance shift assigned to a systematic. Here,
`zpt_g` was fitted with a zero-point shift of 0.003 mag; an adopted
calibration uncertainty of 0.0015 mag therefore gives a scale of 0.5.
Labels not listed use a scale of one.

The covariance including all variations is labelled `ALL`. `COVOPTS`
requests additional matrices, each with a label and a selection of FITOPTs
and MUOPTs:

- `[NOSYS] [=DEFAULT,=DEFAULT]` selects the nominal analysis, giving the
  statistical covariance alone.
- `[CALIB] [+zpt,=DEFAULT]` selects the FITOPTs whose labels contain `zpt`,
  with the nominal MUOPT. Here this includes the *g*- and *r*-band
  zero-point variations.

{ref}`covariance-selecting-systematics` gives the selection syntax and its
structured YAML equivalent. {doc}`covariance_keys` lists the available
settings.

### Running in Python

The `Covariance` stage uses the FITOPTs and MUOPTs defined above. The
equivalent of the input file is:

```python
from jax_pipeline import Covariance

covariance = Covariance(
    "covariance", distances=distances,
    scales={"zpt_g": 0.5},
    covopts=["[NOSYS] [=DEFAULT,=DEFAULT]", "[CALIB] [+zpt,=DEFAULT]"],
)
```

For Hubble diagrams held directly in DataFrames, see the
{ref}`Python API <covariance-python-api>`.

### Binning and output

By default, the covariance is calculated between the binned distance
estimates in BBC's M0DIF output, with one row and column per redshift bin.
Set `UNBINNED: true` in the input file, or `mode="unbinned"` on the Python
`Covariance` stage, to calculate the covariance between individual
supernova distances, retaining events present in every selected variation.
The covariance can also be rebinned in light curve parameters; see
{ref}`covariance-rebin-settings`.

The main outputs are `hubble_diagram.txt` and the covariance files for each
requested selection. `INFO.YML` records these for use by the cosmology fit.
{ref}`covariance-output-files` describes the file contents and
{ref}`covariance-output-settings` lists the available matrix representations.

The Hubble diagram and covariance are then passed to the cosmology fit:

```bash
wfit COVARIANCE/hubble_diagram.txt -mucovtot_inv_file COVARIANCE/covtot_inv_000.npz
```

In Python, the following compares cosmological constraints with all
systematics, statistical uncertainties alone, or zero-point uncertainties
alone:

```python
from jax_pipeline import Cosmology

cosmology = Cosmology("cosmology", covariance=covariance,
                      covopts=("ALL", "NOSYS", "CALIB"))
results = run(cosmology, outdir="run")
results["cosmology"]["ALL"]["point_estimate"]
```
