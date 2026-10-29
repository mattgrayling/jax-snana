# Light curve fitting

jax-SNANA supports light curve fitting using SALT, BayeSN and redback-jax
models. Fits can be run from the command line using SNANA input files to
specify configuration, or alternatively directly within Python scripts.

## Command-line fits

The command-line fitter operates on a SNANA dataset or a Rubin HATS catalogue.

(fit-existing-observations)=
### SNANA input files

`VERSION_PHOTOMETRY` selects the dataset and `PRIVATE_DATA_PATH` gives its
parent directory. For observations in `/path/to/MY_DATA`, the BayeSN G26
fitting input `fit.nml` could contain:

```text
 &SNLCINP
  VERSION_PHOTOMETRY = 'MY_DATA'
  PRIVATE_DATA_PATH  = '/path/to'
  KCOR_FILE          = '/path/to/calibration.fits'
  SNTABLE_LIST       = 'FITRES'
  TEXTFILE_PREFIX    = 'MY_FIT'
  BATCH_SIZE         = 1000
 &END

 &FITINP
  FITMODEL_NAME       = '/path/to/BAYESN.G26'
  FILTLIST_FIT        = 'griz'
  FITWIN_TREST        = -10.0, 35.0
 &END
```

`KCOR_FILE` supplies the survey's {ref}`filter calibration <format-kcor>`,
including transmission curves and reference standards. The model and
calibration paths can contain environment variables.
{ref}`data-version-location` describes the dataset directory search, and
{ref}`preparing-text-data` explains how to prepare TEXT observations for fitting.

Run the fit with `snlc_fit`. Trailing `KEY value` pairs override the input
settings:

```bash
snlc_fit fit.nml
snlc_fit fit.nml NFIT_ITERATION 4 FITWIN_TREST -15 45
```

### YAML

The equivalent YAML input, `fit.yaml`, is:

```yaml
version_photometry: MY_DATA
private_data_path: /path/to
kcor_file: /path/to/calibration.fits
fitmodel_name: /path/to/BAYESN.G26
filtlist_fit: griz
fitwin_trest: [-10, 35]
textfile_prefix: MY_FIT
batch_size: 1000
tables: {fitres: key}
```

For YAML inputs, named command-line options override settings:

```bash
snlc_fit fit.yaml
snlc_fit fit.yaml --batch-size 128 --n-fit-iterations 4
```

Filter names can be given in full as a YAML list, or as the unambiguous
single-character labels in the calibration file. Command-line options use
hyphens where Python and YAML use underscores. `snlc_fit fit.yaml --help`
lists settings for the selected model; {doc}`fitting_keys` gives the full
input syntax.

(fit-from-python)=
## Fitting from Python

For example, to fit a light curve with the G26 BayeSN model using the built-in
Pan-STARRS 1 *griz* filters:

```python
from jax_snlc_fit import LightCurveFitter

fitter = LightCurveFitter("BAYESN.G26", batch_size=1)
result = fitter.fit(
    mjd=mjd,
    flux=flux,
    flux_err=flux_err,
    zp=31.4,
    band=band,
    redshift=0.05,
    mwebv=0.02,
    band_map={"g": "g_PS1", "r": "r_PS1", "i": "i_PS1", "z": "z_PS1"},
)
table = result.to_table()
print(table["snid", "status", "PKMJD", "THETA", "AV", "MU_LCFIT", "MU_LCFIT_err"])
```

`zp` is the zero point of the supplied fluxes and defaults to 31.4, as in
the simulator. `redshift` is the heliocentric redshift and `mwebv` is the
Milky Way E(B−V). The `band_map` matches the
observation labels to filter transmissions, as described under
{ref}`fit-filters`.

### Fitting a sample

A table of observations needs `mjd`, `flux`, `flux_err`, `band`, `redshift`
and a `snid` identifying each object. Include `mwebv` for each object's
Milky Way E(B−V); it is taken as zero if omitted. Use full filter names,
such as `g_PS1`, or supply a `band_map` as above:

```python
import pandas as pd

photometry = pd.read_csv("photometry.csv")
fitter = LightCurveFitter("BAYESN.G26", batch_size=16)
result = fitter.fit(photometry, zp=31.4)
table = result.to_table()
```

This processes up to 16 objects at once; {ref}`fit-batching` describes how
to choose the batch size. Pandas DataFrames, Astropy tables and dictionaries
of columns are accepted. Object properties can also be supplied in a separate
`objects` table; the {ref}`table input options <fit-table-inputs>` describe
this form, column names and additional metadata.

(fit-filters)=
## Filters and calibration

Built-in filters allow light curves to be fitted directly in Python without
supplying separate filter files. These are intended for exploratory fits.
For cosmological analyses, the adopted filter transmissions and calibration
need to be explicitly recorded; command-line fits therefore require a
user-supplied calibration file. {doc}`bundled_data` describes the included
filters and their calibration.

`fitter.calibration.get_filter_names()` lists the available filters. These
names can be supplied directly in `band` or through `band_map`, as above.

Supply your own calibration using
`calibration="/path/to/kcor.input"` or an existing kcor FITS file.
{ref}`format-kcor` describes how to specify your filters and reference
standards. These specify the transmission curves and magnitude-system
definitions used to interpret the supplied photometry.

## Fitting settings

### Event and epoch selection

Command-line and pipeline fits apply the configured event and observation
cuts. Events can be selected by redshift, extinction, signal-to-noise and
phase coverage, and after fitting by their fitted parameters and
uncertainties. Observations can be excluded by filter, quality flag, phase
or wavelength. {doc}`fitting_keys` lists these settings.

Sample selection is disabled for direct Python fits. To apply the configured
cuts, use `fitter.fit(..., apply_selection=True)`. The result retains every
input object, with its status recording the outcome; predictions remain
available for finite fits that fail a selection cut.

The command-line CUTSTAT summary reports the number of events remaining
after each cut, in the order applied. Cuts that remove no events are
omitted. The final count gives the number of events written to FITRES;
{ref}`cutstat-reference` describes the cut names and diagnostics.

### Event metadata

`HEADER_OVERRIDE_FILE` replaces event metadata for real data, and
`SIM_HEADER_OVERRIDE_FILE` does the same for simulations. For example, an
override file can supply revised redshifts and peculiar velocities, host
photo-z quantiles or host mass as a function of redshift. See
{ref}`header-override-format` for the file format.

### Parameters and priors

Initial values, bounds, held parameters and priors can be configured for
each model parameter. For a Redback Arnett fit, set
`FITMODEL_NAME = 'REDBACK.arnett_bolometric'` in `redback.nml`.
A normal prior on ejecta mass can then be applied with:

```bash
snlc_fit redback.nml --initial mej 3 --bounds mej 1 8 --prior mej normal mean=3 sigma=0.5
```

The same settings can be supplied when creating a Python fitter:

```python
fitter = LightCurveFitter(
    "REDBACK.arnett_bolometric",
    batch_size=1,
    initial={"mej": 3.0},
    bounds={"mej": (1.0, 8.0)},
    priors={"mej": {"distribution": "normal", "mean": 3.0, "sigma": 0.5}},
)
```

Use `--fixed mej 3` on the command line or `fixed={"mej": 3.0}` in Python
to hold a parameter at a specified value. In Python, a fixed value of
`None` releases a parameter held by the model's defaults. Parameter
names and units are defined by the model; time parameters are supplied in
absolute MJD. Bounds must lie within the model's supported range. A requested
range outside those limits is rejected with a warning, leaving the model's
limits in use.

A supplied prior replaces that parameter's optional model prior. Use
`priors={"mej": None}` to remove it. {doc}`fitting_keys` lists the parameter
settings and prior options. `fitter.model.get_fit_parameters()` returns
the selected model's parameter names, bounds and initial values.

An initial peak estimate centres the peak-time prior and the initial
rest-frame phase cuts. A supplied `PEAKMJD` is used by default; without
one, the starting estimate is the mean observation time. `OPT_SETPKMJD`
selects an estimate from the photometry instead; see {doc}`fitting_keys`.
For Python arrays, a peak estimate can be supplied through
`metadata={"peakmjd": ...}`.

The time of peak has a Gaussian prior of width `PRIOR_MJDSIG` days, centred
on the starting peak time of each round. The first round uses the initial
estimate, and later rounds use the preceding round's fitted value.
The fit includes the model's priors in its reported `FITCHI2`;
{doc}`analysis_outputs` describes the fit statistics and degrees of freedom.

### Photometric redshift fits

`DOFIT_PHOTOZ` fits redshift together with the light curve parameters.
To include a Gaussian prior from a host photometric-redshift measurement,
add the following to `&FITINP`:

```text
DOFIT_PHOTOZ = T
OPT_PHOTOZ = 1
PRIOR_ZERRSCALE = 1
```

In Python, configure the redshift fit when creating the fitter, and supply
the host photometric redshift with the light curve:

```python
from jax_snlc_fit import LightCurveFitter

fitter = LightCurveFitter(
    "BAYESN.G26", batch_size=1,
    dofit_photoz=True, opt_photoz=1, prior_zerrscale=1,
)
result = fitter.fit(
    mjd=mjd, flux=flux, flux_err=flux_err, zp=31.4, band=band,
    metadata={"photoz": 0.3, "photoz_err": 0.03}, mwebv=0.02,
    band_map={"g": "g_PS1", "r": "r_PS1", "i": "i_PS1", "z": "z_PS1"},
)
```

Here `photoz` and `photoz_err` are the host galaxy's redshift estimate and
uncertainty. `prior_zerrscale=1` enables the host Gaussian prior, which is
disabled by default.

Events without a measured host photometric redshift are fitted without the
host Gaussian constraint. A non-Gaussian host-redshift distribution can
instead be supplied as quantiles. {ref}`photoz-settings` describes the
Gaussian prior widths, and {ref}`photoz-quantile-settings` gives the quantile
controls. {doc}`light_curve_data` describes the required host-redshift columns.

The fitted redshift replaces the data redshift in the output, so subsequent
analysis stages use the fitted value. The results also contain its
uncertainty, prior information and covariances with the model parameters;
{doc}`analysis_outputs` describes these columns. The numerical search settings
are listed under {ref}`photoz-settings`.

### Minimisation and uncertainties

The fitter uses a Newton method with Levenberg-Marquardt damping.
`NFIT_ITERATION` sets the number of fitting rounds. Rest-frame phase cuts
and epoch rejection are updated after the first round; {doc}`fitting_keys`
lists the controls for these and for updating the model covariance.

For SALT, the flux covariance includes the model's phase- and
wavelength-dependent uncertainty as well as measurement errors. Its colour
dispersion correlates observations, and the fit uses the full covariance.
BayeSN instead fits intrinsic residual coefficients with the trained model's
prior; these describe spectral variation within the fitted SED.

Parameter uncertainties are calculated from the Hessian at the final fit,
including the priors. For profile uncertainties, set `MINOS='z'` in
`&FITINP` to profile redshift, or `MINOS='all'` to profile every free
parameter. In Python, use `minos="z"` or `minos="all"` when creating the
fitter. See {ref}`profile-uncertainties` for the method and
{ref}`profile-output` for the individual errors and flags.

Command-line fits use single precision by default. Set
`DOUBLE_PRECISION = T` in `&FITINP`, or `double_precision: true` in YAML,
to use double precision. For Python pipelines, set precision for the whole
run as described in {doc}`building_pipelines`.

(fit-output)=
## Results and predictions

### Command-line tables

The tables are written to the working directory, with names starting with
`TEXTFILE_PREFIX`.

`SNTABLE_LIST` selects the output tables. For example,
`SNTABLE_LIST = 'FITRES SNANA LCPLOT(mxlc:10)'` writes the fitted results,
event diagnostics and plotting data for up to ten light curves.
The available tables include:

`FITRES`
: Fitted parameters, uncertainties and covariances, fit statistics, redshifts
  and host properties. Simulated samples also include true parameter values.
  These fitted results are used for {doc}`standardisation <bias_corrections>`.

`SNANA`
: Event information and status flags, including events rejected by cuts or
  unsuccessful fits.

`OUTLIER`
: Observations far from the fitted model.

`LCPLOT`
: Observations and fitted curves evaluated on a dense phase grid. A companion
  `.LCLIST.TEXT` file identifies the events and columns. See
  {ref}`plotting table selection <plotting-table-selection>` for the settings and
  {ref}`inspect-first-fit` for a plotting example.

{doc}`analysis_outputs` describes the table contents.

(fit-predictions)=
### Python results and predictions

`result.to_table()` returns an Astropy table of fitted parameters,
uncertainties and the status of each input object. Parameters are masked
where no fit was made; `reason_description` explains the outcome. Use
`result.write_table("fits.ecsv")` to save the table.

`predict` evaluates the fitted model at chosen times and bands, returning
arrays for use in your own plotting code. For example, using the
observations from the {ref}`single-object fit <fit-from-python>`:

```python
import numpy as np
import matplotlib.pyplot as plt

model_mjd = np.linspace(mjd.min(), mjd.max(), 200)
prediction = fitter.predict(result, model_mjd, "g_PS1")

fig, ax = plt.subplots()
g = band == "g"
ax.errorbar(mjd[g], flux[g], yerr=flux_err[g], fmt="o")
ax.plot(prediction["mjd"], prediction["flux"])
ax.set(xlabel="MJD", ylabel="Flux")
```

Predictions use the input flux units by default. `model_flux_err` gives the
model-provided flux uncertainty, excluding uncertainty propagated from the
fitted parameters. It is zero for BayeSN and Redback; BayeSN predictions
evaluate the fitted realisation of the intrinsic scatter.

A result can still be used for prediction after subsequent fits. For a
result containing several objects, pass `snid` to choose the fit to evaluate.

## Repeated and pipeline fits

(fit-batching)=
### Batching and padding

Events are fitted in batches of `BATCH_SIZE`, with light curves padded to a
common number of epochs. By default the whole sample is padded to its
longest light curve, or, for a pipeline `Fit`, to the longest in any of
its data versions. This can require substantial padding when light-curve
lengths vary. Set `NUM_NOBS_BINS` above 1 to group the events into that many
bins of similar length, each padded to its own longest member.
`MIN_BIN_GAIN` is the fraction of padded work a bin must save to be kept.
Each bin is compiled once.

In Python, set `batch_size` and `num_nobs_bins` when creating the fitter.
The defaults are a batch size of 1000 and one bin; a smaller sample is still
padded to the configured batch size. A fitter retains its prepared observation
lengths between calls, extending them when longer light curves arrive.
`fitter.reset_shape()` recalculates these lengths and the photo-z search size
on the next fit, retaining the model, calibration and batch size.

The first fit compiles the fitting functions, which are reused by subsequent
fits with compatible settings and array shapes. Use `progress=False` in
`fit` to suppress progress output.

### Pipeline fits

To connect a simulation to a light curve fit in a Python pipeline:

```python
from jax_pipeline import Fit, Sim, run

sim = Sim("des_ia", "sim.input", batch_size=20_000)
fit = Fit("des_fit", "fit.nml", datasets=[sim])
table = run(fit, outdir="run")["des_fit"]["des_ia"]["nominal"]
```

For existing observations, a `Data` stage supplies the dataset location:

```python
from jax_pipeline import Data

observed = Data("observed", "/path/to/MY_DATA")
fit = Fit("fit", "fit.nml", datasets=[observed])
table = run(fit, outdir="observed_fit")["fit"]["observed"]["nominal"]
```

The returned tables are keyed by dataset name and then systematic variation,
with the nominal fit under `"nominal"`. {doc}`building_pipelines` describes
fitting multiple datasets together, and {doc}`systematics` explains FITOPTs.

## Next steps

{doc}`bias_corrections` explains how to obtain selection-bias-corrected
distances from the light-curve fits. For a photometrically selected sample,
{doc}`classification` shows how to apply SuperNNova to the photometry and
combine its probabilities with the fitted results for use in the distance fit.
