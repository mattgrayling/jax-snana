# Light-curve fitting settings

This page lists the light curve fitter's settings. The **SNANA key** column
gives the corresponding SNANA input spellings. SNANA input files use
`KEY = value` inside `&SNLCINP` (data and selection) or `&FITINP`
(fitting), or these two mappings in YAML. Overrides follow the input filename:

```bash
snlc_fit fit.nml NFIT_ITERATION 4 FITWIN_TREST -15 45
```

For YAML and Python, use the names in the **Python / YAML** column.
Command-line options use the same names with hyphens:

```bash
snlc_fit fit.yaml --n-fit-iterations 4 --fitwin-trest -15 45
```

See {doc}`fitting_light_curves` for worked examples.

Several keys are bit masks. The listed numbers are bit values; combine
options by adding their values.

{doc}`model_settings` lists model-specific keys and shared model settings
such as `PRIOR_MJDSIG`.

## Photometry and calibration

The photometry and calibration inputs are:

- `VERSION_PHOTOMETRY` names the directory of light curves, in the FITS form
  or as a Rubin HATS catalogue, both described in {doc}`light_curve_data`.
- `KCOR_FILE` names a {ref}`calibration input or FITS file <format-kcor>`
  giving the transmissions, primary reference spectrum and zero points.

`PHOTOMETRY_COLUMNS` chooses the Rubin columns a HATS catalogue is read from;
see {doc}`light_curve_data`. `LEAKAGE_CUT` removes out-of-band leakage from
the filters of the calibration file, as described in {ref}`format-kcor`.

The redshift is read from each event's header or fitted as described under
{ref}`photoz-settings`. The version is named without a path and searched for
in several places, `PRIVATE_DATA_PATH` first where it is set. See
{ref}`data-version-location` for the search order. SNANA versions must be in
FITS form; {ref}`preparing-text-data` describes conversion with SNANA's
`snana.exe`.

(host-mass-interpolation)=
`HEADER_OVERRIDE_FILE` and `SIM_HEADER_OVERRIDE_FILE` can supply host
stellar masses on a redshift grid. The fitter interpolates the mass and its
error at the event's redshift, including the contribution from redshift
uncertainty. Outside the grid, the catalogue mass is retained. See
{ref}`header-override-format` for the input format.

Most of the remaining keys change the photometry before it is fitted. They
are used to measure a systematic:

- `MAGOBS_SHIFT_ZP` shifts a band's photometric zero point.
- `FILTER_LAMSHIFT` shifts a band's transmission in wavelength.
- The `FUDGE_FLUXCAL_*` family adds an offset or an error to every
  measurement, and `FUDGE_MAG_ERROR` adds an error as a fraction of the flux
  in magnitudes.
- `FLUXERRMODEL_FILE` scales the reported errors by observing conditions.

`FLUXCAL_BASELINE_CORRECT` removes a negative flux baseline from
difference-imaging light curves, estimating the offset separately for
each event and band.

`OPT_MWEBV` selects the source of each event's Milky Way E(B-V): the data
header or a published map. `OPT_MWCOLORLAW` chooses the law that turns it
into extinction at each wavelength, and 0 turns extinction off.
`RV_MWCOLORLAW` sets the law's R_V. If you leave them unset, a simulation is
fitted with the law and R_V it was made with, and real data with the
Fitzpatrick 1999 law at R_V = 3.1. To generate a Galactic-extinction
systematic, scale and shift it with `MWEBV_SCALE` and `MWEBV_SHIFT`.

The peak-time settings estimate the time of peak before the fit begins.
By default the peak time in the data header is used. `OPT_SETPKMJD` estimates it from
the photometry instead, using high signal-to-noise epochs within a time
window around the brightest one, the trigger time, or a Bazin fit. The
estimate is the starting value and the initial centre of the peak-time prior.
`MJDWIN_SETPKMJD` sets the time window and `SNRCUT_SETPKMJD` the signal-to-noise
threshold.

```{eval-rst}
.. configtable:: jax_snlc_fit.config.Data
   :field-keys:
   :keys-from: jax_snlc_fit.config_parser
```

## Selection

The selection keys remove events whose fits would be poorly constrained, for
example an event observed only after peak or in a single band.

The cuts are applied at three stages:

- Before the fit, the `CUTWIN_` windows select on quantities measured from the
  data, such as the redshift, the maximum signal-to-noise and the phase
  coverage. The epoch members of that family, `PHOTFLAG_MSKREJ` and
  `EPCUT_SNRMIN`, mask single observations.
- During the fit, `FITWIN_TREST` and `FITWIN_MJD` drop the epochs outside the
  phase and date ranges, using the time of peak found in the first round.
- After the fit, the `FITWIN_` windows are applied to the result, for example
  to drop an event with an implausible fitted colour or shape.

For `CUTWIN_` keys, use the quantities listed in the table below and a plain
numeric window. Per-band arguments are not supported for these keys.

The `FITWIN_` keys are restricted in the same way. The suffix is one of the
model's fitted parameters, a derived output column, or one of the shared
quantities listed with the key below. Error windows append `_ERR` to the
quantity's name: for example, `FITWIN_x1_ERR` selects on the SALT shape
error written as `x1ERR` in FITRES. The CUTSTAT
table in the run log shows which cut removed which events; see
{doc}`fitting_light_curves`.

The keep-lists and ignore-lists name events by CID, by type or by field. Use
them to investigate a single event or exclude a known bad one. `SNCID_LIST`
takes numeric CIDs; `SNCCID_LIST` and `SNCID_LIST_FILE` take any CID, and
also match an event's `NAME_TRANSIENT` or `NAME_IAUC`.
`OPT_SNCID_LIST` can also turn off every other per-event cut, so the named
events are fitted regardless of their photometry. A positive
`FLAG_USE_SAME_EVENTS` sets `OPT_SNCID_LIST` to its own value. `SIM_PRESCALE`
and `JOBSPLIT` thin the sample. `SIM_PRESCALE` keeps one event in N, chosen
by index so the same subset is kept every run, and `JOBSPLIT` divides the
sample between batch jobs, so a large simulation can be fitted in parallel
without splitting the data.

Three selection conventions are useful when comparing fitted samples:

- `CHECK_CRAZYFITERR` rejects events with non-finite fitted errors or errors
  outside the model-defined limits.
- `CUTWIN_SNRMAX2` cuts on the second-highest per-band signal-to-noise.
- In a photometric-redshift fit, a filter is dropped when its mean wavelength
  at the trial redshift falls outside the allowed rest-frame range. After
  the last round, an event is rejected if its fitted redshift plus or minus
  its uncertainty crosses this limit for any fitted filter. The wavelength
  criterion is described under {ref}`wavelength selection <fit-wavelength-selection>`.

For phase coverage, `TGAPMAX` is the largest rest-frame gap between
observations. `T0GAPMAX` is the largest gap overlapping the peak interval,
bounded by the upper limit of `CUTWIN_TRESTMIN` and lower limit of
`CUTWIN_TRESTMAX`. `NFILT_SNRMAX` counts bands whose maximum signal-to-noise
exceeds the lower limit of
`CUTWIN_SNRMAX`; set that window alongside the required number of bands.

```{eval-rst}
.. configtable:: jax_snlc_fit.config.Selection
   :field-keys:
   :keys-from: jax_snlc_fit.config_parser
```

## Model and parameter settings

`FITMODEL_NAME` selects the model to fit: a model directory, a BayeSN YAML
file, or a Redback identifier such as `REDBACK.arnett_bolometric`. The model
type is normally identified from the last path component or identifier, as
described in {doc}`models`. `MODEL_TYPE` is needed only when the name does
not identify it. Each model's parameters and input keys are
documented on {doc}`model_settings`. Either section accepts a model key, but write it
in `&FITINP`, beside `FITMODEL_NAME`, where `snlc_fit.exe` reads the shared
ones such as `PRIOR_MJDSIG`.

These keys configure initial values, fixed parameters and parameter bounds:

- `INIVAL_` gives a starting value and `INISTP_` an initial step. If you want
  to fix a parameter while fitting the rest, give it a step of zero.
- The suffix on either is one of the model's own parameter names, or a shared
  name such as `SHAPE` or `COLOR` for the parameter with that role, so an
  input written for one model often runs unchanged on another.
- `USESIM_<PARAM>` fixes that parameter to its simulated true value.
  `USESIM_INIVAL` instead uses simulated values to initialise the fit while
  leaving the parameters free.

The time of peak has a Gaussian prior centred on each round's starting value.
Later rounds recentre it on the preceding fit. Other priors depend on the
model: SALT penalises shape and colour outside specified ranges, while
BayeSN uses population priors on shape and host extinction. The `PRIOR_`
keys override model defaults for the corresponding parameters; `OPT_PRIOR`
turns priors off. See {doc}`model_settings` for each model's assumptions.
{doc}`fitting_light_curves` describes how the priors enter the fit and
`FITCHI2`.

On the command line, use `--prior PARAMETER DISTRIBUTION coefficient=value ...`
for a prior, `--initial PARAMETER VALUE` for a starting value,
`--fixed PARAMETER VALUE` to hold a parameter fixed, and
`--bounds PARAMETER LOWER UPPER` for parameter limits.

For negative values in scientific notation, use `=` for a scalar option
(e.g. `--mwebv-shift=-1e-2`) or a quoted mapping for parameter settings
(e.g. `--fixed '{x1: -1e0}'`). Multi-value options such as
`--fitwin-trest -100 45` accept negative values in decimal notation.

The `priors` mapping in YAML and Python accepts the following distributions.
The command-line `--prior` option uses the same names and coefficients:

| Distribution | Coefficients |
|---|---|
| `normal` | `mean`, `sigma` |
| `uniform` | `lower`, `upper` |
| `loguniform` | `lower`, `upper` |
| `exponential` | `scale` |
| `flat_rolloff` | `lower`, `upper`, `sigma` |

The coarse-grid keys set the colour and shape axes of the photo-z grid search,
a range and a bin count each, replacing the model's defaults.

`USE_MODEL_MAGERR`, `FUDGE_MODELERR_SCALE` and `FUDGE_MAGERR_MODEL`
control the model-provided photometric uncertainty, including SALT's
phase- and wavelength-dependent error model. They can disable, scale or
replace that uncertainty, for example when assessing model-error systematics.

```{eval-rst}
.. configtable:: jax_snlc_fit.config.Model
   :field-keys:
   :keys-from: jax_snlc_fit.config_parser
   :omit-sections: Set by the keys the model declares
```

## Fit controls

(fit-wavelength-selection)=
`FILTLIST_FIT` chooses which bands are fitted. `RESTLAMBDA_FITRANGE` specifies
the allowed rest-frame range for their mean wavelengths, intersected with the
model's own filter-centre range. A filter is used only when its observer-frame
mean wavelength divided by $1+z$ lies within that interval.

The fit runs in rounds, because the weight of each measurement depends on the
model being fitted. Round 1 recomputes the model error at every step, with no
correlation between epochs. Later rounds normally refit with the covariance
fixed at the previous round's best-fitting model, including correlations
between epochs. If `OPT_COVAR_FLUX = 0` and `OPT_COVAR_MWXTERR = 0`, with
no non-zero `FUDGE_MAG_COVERR`, the diagonal weights continue to follow the
parameters. `NFIT_ITERATION` sets the number of rounds.
`DELCHI2_REJECT` and `SIGNOISE_REJECT` remove discrepant epochs between
rounds.

`OPT_COVAR_FLUX` and `OPT_COVAR_MWXTERR` set what that fixed covariance
includes. For SALT, colour dispersion correlates observations within each
band. Milky Way extinction uncertainty can also contribute covariance.
`OPT_CHI2_SIGMA` adds a logarithmic term for the total error changing with the
parameters. It is on by default for a redshift fit, where that error changes
fastest.

The error fudge keys inflate the measurement errors, everywhere with
`FUDGEALL_MAXFRAC` or over a chosen range of phase, wavelength and band with
the `FUDGE_FITERR_` family.

The reference cosmology sets the reference distance modulus written with each
fit. The fit does not constrain it, other than through the photometric
redshift distance priors below, and the cosmology is measured later, in the
{doc}`cosmology stage <cosmology>`. Use the same values in a simulation and in
the fit of that simulation.

`BATCH_SIZE`, `NUM_NOBS_BINS` and `MIN_BIN_GAIN` set how the sample is split
into the compiled calls that fit many events at once, and do not change the
fitted values. `MAXITER_ROUND1` and `MAXITER_ROUND2` limit the iterations of
the optimiser in each round. `DOUBLE_PRECISION` sets the floating-point
precision.

```{eval-rst}
.. configtable:: jax_snlc_fit.config.Fit
   :field-keys:
   :keys-from: jax_snlc_fit.config_parser
   :omit-sections: Not set by an input key
```

(profile-uncertainties)=
### Profile uncertainties (experimental)

The `minos` setting selects profile uncertainties after the final light-curve
fitting round. Its default is `None`, which retains Hessian-only errors.
`minos="z"` profiles redshift and retains Hessian errors for the other
parameters; `minos="all"` profiles every free parameter, including the model
parameters in a fit with fixed redshift. Held parameters are never profiled.

SNANA input files use `MINOS='z'` or `MINOS='all'`; the command-line options
are `--minos z` and `--minos all`. YAML uses `minos: z` or `minos: all`.
For Python, pass `minos="z"` or `minos="all"` when constructing the fitter.

For each parameter, the profiler searches on either side of the best fit
for an increase of one in the final fit objective, minimising over the other
free parameters at each trial value. The observations, weights and prior
settings remain those of the final fitting round.

The reported symmetric error is the mean of the lower and upper profile
errors when both searches reach the required change in the objective.
Otherwise the Hessian error is retained and flagged in the output. A bound
reached before that change is recorded as a bound, not as a completed interval.
If profiling finds a lower minimum that cannot be resolved, the event's
profile intervals are not used. The {ref}`output column descriptions <profile-output>`
describe the individual errors and status flags.

Profiling requires additional minimisations, so it increases the fitting
time, particularly when profiling every free parameter.

(photoz-settings)=
(photoz-quantile-settings)=
## Fitting for redshift

`DOFIT_PHOTOZ` makes the redshift a free parameter alongside the light curve
parameters. `OPT_PHOTOZ` chooses the redshift prior:

- `1`: a Gaussian centred on the host galaxy's photometric redshift, disabled
  by default. Set `OPT_PHOTOZ = 1` together with
  `PRIOR_ZERRSCALE = 1` to enable the host Gaussian prior.
- `2`: a Gaussian on the redshift the data file marks as its best. An event
  without one is fitted with no redshift prior.
- `4`: the host redshift's quantiles, read per event as its full probability
  distribution, which is more accurate when that distribution is not
  Gaussian.
- `12`: the quantiles again, read through Steffen's monotone cubic, so the
  distribution cannot dip below zero between quantiles.

With a quantile prior, the fitted redshift is restricted to the overlap of
`PHOTOZ_BOUND` and the host distribution's range. Events with no probability
in that interval are excluded. Missing or non-increasing quantiles stop the
run by default; `ABORT_ON_MISSING_QZPHOT = F` and
`ABORT_ON_BADQZPHOT = F`, respectively, exclude those events instead.
{doc}`light_curve_data` describes the quantile columns.

A distance prior ties the fitted distance to the fitted redshift through the
reference cosmology, which helps most when an event has no other redshift
information. `PRIOR_MUERRSCALE` adds a Gaussian prior on the fitted distance
less the reference cosmology's, whose width is how far that distance moves
under the `OMAT_REF`, `W0_REF` and `WA_REF` uncertainties, multiplied by the
scale. `PRIOR_DMU_RANGE` instead keeps that difference within a range, rolling
off outside it with width `PRIOR_DMU_SIGMA`. Only one of the two can be set,
and as in SNANA a `PRIOR_MUERRSCALE` of 9 or more applies no prior.

The chi-squared can have multiple minima in redshift, so the fit first
evaluates a coarse grid over redshift and the model's declared axes (colour
and shape for SALT), and starts from the best grid point. By default the
redshift axis spans each event's prior, two widths either side of its centre,
at 42 points per unit of z/(1+z). The keys in the second group change it:
`NBINZ_COURSE_PHOTOZ` and `RANGEZ_COURSE_PHOTOZ` set the redshift axis, and
`SCALE_NBIN_COURSE_PHOTOZ` multiplies the bin count of every axis. The colour
and shape axes are set by the coarse-grid keys of the model section.
`NBINT_COURSE_PHOTOZ` adds a PEAKMJD axis, offset from each event's estimate
across `RANGET_COURSE_PHOTOZ`.

The last group handles bands whose rest-frame wavelength could fall outside
the model's coverage over the allowed redshift range. `PHOTOZ_ITER1_LAMRANGE`
holds bands outside an observer-frame wavelength range out of the first round,
and adds them back once the redshift is known. `PHOTODZ_REJECT` drops a band
after a round when the fitted redshift lies within that margin of the edge of
the band's usable redshift range.

```{eval-rst}
.. configtable:: jax_snlc_fit.config.PhotoZ
   :field-keys:
   :keys-from: jax_snlc_fit.config_parser
   :omit-sections: Not set by an input key
```

## Output

`SNTABLE_LIST` chooses the tables to write:

- `FITRES`: fitted parameters, errors, covariances and chi-squared. The
  {doc}`bias-correction stage <bias_corrections>` reads this.
- `SNANA`: event information and selection flags, including events that
  failed the cuts or could not be fitted.
- `OUTLIER`: the single measurements far from the fitted model, for finding
  bad photometry.
- `LCPLOT`: the data and the fitted curve for a few events, for plotting.

The value is one string, with the table names separated by spaces. A name may
be followed by its own options in brackets, separated by commas:

```text
 SNTABLE_LIST = 'FITRES(text:key) SNANA(cutmask:0) LCPLOT(mxlc:10) nonuisance'
```

`text:key`, `text:csv` and `text:col` select the layout. `cutmask:0` keeps
only accepted events in the SNANA table, and `mxlc:10` requests up to ten
light curves. Options applying to all tables, such as `nonuisance` and
`nosimvar`, are separate words outside the brackets. Their Python/YAML
equivalents appear in the table below. For example:

```yaml
tables: {fitres: key, snana: key, lcplot: key}
cutmask: 0
mxlc: 10
nonuisance: true
```

Asking for `LCPLOT` also writes `<prefix>.LCLIST.TEXT`, which lists the
events in the curve table and its column definitions.

`TEXTFILE_PREFIX` sets the start of every output file name, and `OUTPUT_NAME`
gives the full name of the FITRES file. If neither is set, the prefix is the
name of the data version's HEAD file without its `_HEAD.FITS` ending, and the
tables are written to the working directory. The remaining keys compute each
event's 1/Vmax volume, given the limiting magnitudes you supply, for a rate
measurement.

```{eval-rst}
.. configtable:: jax_snlc_fit.config.Output
   :field-keys:
   :keys-from: jax_snlc_fit.config_parser
   :omit-sections: Not set by an input key
```

(cutstat-reference)=
### CUTSTAT and fit diagnostics

The CUTSTAT summary reports losses from sample selection, phase coverage
and fitted-parameter cuts. `FITWIN_ITER1` records rejection after the first
fitting round; `TREST` and `NFILT_SNRMAX_FIT` apply coverage and band-count
requirements using the fitted peak time and redshift.

`OPT_SNCID_LIST` fits a named list of events regardless of their photometry
and skips the post-fit cuts. A single row, `SNCID_PEGGED`, replaces them. It
removes the events whose fit ended at the edge of a flat prior and, where
`FITWIN_PROB` is set, those whose `FITPROB` is below a tenth of its lower
edge.

The `FITWIN` row can remove events even when you set no `FITWIN_` key,
because two windows are always applied. The off-diagonal terms of the fitted
covariance must lie between -8 and 8, and, for a model with an `mB` column
such as SALT, the error on `mB` must be below 1.0.

`ERRMAX_BAD: <parameter> error replaced by INISTP for N SNe` identifies
uncertainties replaced by the parameter's initial step, either because the
reported error was below the model's minimum or because the parameter did
not move from its starting value. These events remain in the fitted sample.

For explicitly rejected SNANA keys, see {ref}`unsupported-fitting-keys`.
