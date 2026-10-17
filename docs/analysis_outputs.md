# Analysis outputs

For simulation output, see {doc}`light_curve_data`.

```{contents} On this page
:local:
:depth: 1
```

(format-fitres)=
## Light curve fitting

The fitter writes fitted parameters in a FITRES table, with optional tables
for event selection, outliers and fitted light curves. A FITRES table begins:

```
# VERSION_PHOTOMETRY: FF_FITS
# PROGRAM:       snlc_fit.py
# TABLE_NAME:    FITRES
#
VARNAMES: CID IDSURVEY TYPE FIELD CUTFLAG_SNANA ERRFLAG_FIT RA DEC zHEL zHELERR ...
SN: 000027 10 1 C2 3 0 52.500000 -21.500000 0.07347 0.00100 ...
```

A header of commented `# KEY: value` lines gives metadata, `VARNAMES:` names
the columns, and each row starts with `SN:` in a per-supernova table or `ROW:`
in a binned one.

Header metadata includes:

- `VERSION_PHOTOMETRY`: the data version the fit read.
- `SNANA_VERSION`: the SNANA compatibility identifier.
- `JAX_VERSION`: this package, its version and the revision of the checkout
  that ran it.

For simulated data, the header also records the simulation cosmology from
its README: `SIM_OMEGA_MATTER`,
`SIM_OMEGA_LAMBDA`, `SIM_w0_LAMBDA`, `SIM_wa_LAMBDA` and `SIM_MUSHIFT`.
`SIM_MUSHIFT` is a single shift or a range `lo:hi`, and 0 when the simulation
set none.

### FITRES columns

`CID`
: The supernova's identifier.

`IDSURVEY`
: The numeric code of the survey named in the data header.

`TYPE`
: The `SNTYPE` the data header declared, or zero where it declared none.

`FIELD`
: The field the supernova was found in, or `VOID`.

`CUTFLAG_SNANA`
: Selection and fit-completion status, as two bits: 1 for the selection cuts
  and 2 for a fit that finished cleanly. The FITRES contains only fitted
  events, so here the value is always 3. The `SNANA` table keeps a row for
  every event read, and there 0 and 1 appear too.

`ERRFLAG_FIT`
: Zero for a fitted event. The {ref}`SNANA table <fitter-other-tables>` also
  includes unsuccessful events and records their failure codes.

`RA`, `DEC`
: Sky position in degrees.

`zHEL`, `zCMB`, `zHD`
: The heliocentric redshift as the data file gives it, the same corrected to
  the CMB frame, and the redshift the Hubble diagram uses. `zHD` is `zCMB`
  corrected for the peculiar velocity where one is recorded, and a copy of it
  otherwise. A photometric-redshift fit writes its fitted value into `zHEL`,
  with its error in `zHELERR`, and derives the other two from it.

`VPEC`, `LENSDMU`
: Peculiar velocity in km/s and the weak-lensing offset in magnitudes, both as
  the data file recorded them.

`MWEBV`
: Milky Way E(B-V) along the line of sight used in the fit, from the header
  or the dust map as `OPT_MWEBV` chooses, after `MWEBV_SCALE`, `MWEBV_SHIFT`
  and `MWEBV_FORCE` are applied.

The host galaxy columns come from the first block of the data file's
`HOSTGAL_*` keys:

- `HOST_LOGMASS`, `HOST_LOGSFR`, `HOST_LOGsSFR` and `HOST_COLOR`, with
  uncertainties in the corresponding `HOST_*_ERR` columns.
- `HOST_LOGMASS_ZPHOT` and its error, both set to -9999.
- `HOST_MAG_<band>` and `HOST_MAGERR_<band>` per band.

The `text:host` option of `SNTABLE_LIST` adds the host's identity columns:

- `HOST_NMATCH` and `HOST_NMATCH2`, the numbers of matched galaxies.
- `HOST_OBJID`, `HOST_ZPHOT`, `HOST_ZSPEC`, `HOST_RA`, `HOST_DEC`,
  `HOST_ANGSEP` and `HOST_DDLR`, copied from the data.
- `HOST_QZPHOT` and `HOST_QZPHOTSTD`, the mean and standard deviation of the
  photo-z distribution splined through the quantiles.
- `HOST_CONFUSION`, the host confusion of Gupta et al. (2016), calculated from
  the `DDLR` of up to three stored galaxies. It's -9 where fewer than two are
  stored.
- the surface-brightness columns `HOST_SBFLUXCAL_<band>` and
  `HOST_SBMAG_<band>`.

`text:host2` adds the second galaxy's identity columns, properties and
magnitudes, as `HOST2_*`.

`PKMJDINI`
: The time of peak the fit started from: the value the data file declared, the
  estimate `OPT_SETPKMJD` replaced it with, or -9 where there was neither.

`SNRMAX1`, `SNRMAX2`, `SNRMAX3`
: The three largest per-band peak signal-to-noise ratios, in descending order,
  over the bands selected by `SNRMAX_FILTERS`. Missing values are -9.

`SNRSUM`
: The root sum of squares of the signal-to-noise of every epoch the fit used.

`BANDLIST`
: The single characters of the bands the event has at least one epoch in.

`OPT_PHOTOZ`
: The `OPT_PHOTOZ` the fit ran under, 0 for a fit at fixed redshift.

Fitted parameters are reported with their uncertainties:

- SALT: `PKMJD`, `x1`, `c`, `mB` and `x0`. `mB` is derived from the fitted
  amplitude. `RADIUS_POP` is `sqrt((x1/3)^2 + (c/0.3)^2)`.
- BayeSN: `PKMJD`, `THETA`, `MU_LCFIT`, `AV`, `RV` where the fit frees it, and
  residual coefficients `EPS_TFORM_0`, `EPS_TFORM_1`, and so on. Their number
  depends on the trained model.

`MU_LCFIT` and `MUERR_LCFIT` are the fitted distance modulus and its error.
SNANA's fitter uses `DLMAG` and `DLMAGERR`; BBC accepts either spelling.

A parameter held fixed with `INISTP_=0` has zero reported uncertainty and
covariances and is excluded from the free-parameter count in `NDOF`.

Off-diagonal covariances are named `COV_<a>_<b>`. At fixed redshift, SALT
reports `COV_x1_c`, `COV_x1_x0` and `COV_c_x0`; photometric-redshift fits
also report `COV_x1_zPHOT`, `COV_c_zPHOT` and `COV_x0_zPHOT`.

Photometric-redshift fits also report the redshift prior each event was
fitted with, as `zPRIOR` and `zPRIORERR`. The prior is chosen by
`OPT_PHOTOZ`, whose options {doc}`fitting_keys` describes. Under option 1
these are the host's photometric redshift and its quoted error; under option
2, REDSHIFT_FINAL and its uncertainty; and under option 4 or 12, the mean and
standard deviation of the host's quantile distribution. Events without a
REDSHIFT_FINAL under `OPT_PHOTOZ` 2 are fitted with no redshift prior and
report a `zPRIORERR` of 9999.

`NDOF`, `FITCHI2` and `FITPROB` give the degrees of freedom
(the number of epochs fitted less the number of free parameters), the
chi-square at the minimum including priors, and the fit probability.
BayeSN excludes its residual coefficients from the parameter count used
for `NDOF`.

For simulated data, the FITRES also records the simulation truth:

- `SIM_GENTYPE`: the simulated supernova type.
- `SIM_LIBID` and `SIM_NGEN_LIBID`: the SIMLIB entry and the number of events
  generated from it.
- `SIM_TEMPLATE_INDEX`: the template index assigned by the
  `NON1A.LIST` or `SED.INFO` catalogue. It is zero for models without a
  template catalogue, including SALT and BayeSN.
- `SIM_ZCMB`, `SIM_VPEC`, `SIM_DLMAG`, `SIM_LENSDMU` and `SIM_PEAKMJD`: the
  true redshift, peculiar velocity, distance modulus, lensing offset and time
  of peak.
- `SIM_ZHOST_MATCH`: the true redshift of the host matched to the supernova, or
  zero where none was.
- `SIM_ZFLAG`: redshift source. 0 for none, 1 for the
  supernova's own spectrum, 2 for a host spectrum, 3 for a host photometric
  redshift and 4 for a mismatched host.
- `SIM_WGT_POP`: the relative weight from the original and modified
  population profiles when using `GENPDF_EXPON_REWGT`, as described under
  {ref}`population-reweighting`.
  It is 1 when that setting is not used. The corresponding data-file field
  is `SIM_WGT_POPULATION`.
- Model truth columns, including `SIM_x1`, `SIM_c`, `SIM_alpha`, `SIM_beta`,
  `SIM_gammaDM`, `SIM_x0` and `SIM_mB` for SALT, and `SIM_AV` and `SIM_RV`.

Three values mark missing or unavailable quantities:

- -9: a quantity not supplied or unavailable for the event.
- -9999: an unavailable host property, including events with no matched host.
- 999: an unavailable host magnitude or shape measurement.

(profile-output)=
### Profile uncertainty columns

Fits with `minos` enabled also report the lower and upper profile errors for
each fitted parameter. These use the model's parameter names, including `x0`
for SALT amplitude and `zPHOT` for fitted redshift.

| Column | Meaning |
| --- | --- |
| `<parameter>_MINOS_LO`, `<parameter>_MINOS_HI` | Positive distances from the best fit on each side. A search that reaches a parameter bound reports its distance to that bound; failed searches give NaN. |
| `<parameter>_MINOS_STATUS_LO`, `<parameter>_MINOS_STATUS_HI` | Outcome of each search, using the codes below. |
| `ERRFLAG_<parameter>` | Source of the reported symmetric error: 1 for the mean of two completed profile errors, 2 for a Hessian error, or 6 for an invalid error. |

The side-status codes are 0 for a completed interval, 1 for a parameter bound,
2 when minimisation over the other parameters did not converge, 3 for an
unresolved lower minimum, 4 when the search reached its iteration limit, and
5 when the parameter was not profiled. Bound-limited intervals do not
represent completed profile uncertainties.

(fitter-other-tables)=
### Other fitter tables

`SNTABLE_LIST` chooses which tables the fitter writes. Each goes to
`<TEXTFILE_PREFIX>.<TABLE>.TEXT` by default.

`SNANA`
: One row per event read, with the same event, host and truth columns as the
  FITRES but no fitted parameters and no `BANDLIST`. It includes the events
  that never reached a fit, for which `CUTFLAG_SNANA` and `ERRFLAG_FIT` give
  the reason. The `cutmask` option restricts which rows it keeps. Rows start
  `SN:`. `ERRFLAG_FIT` is zero for fitted events; other values are:

  - -9: the fit was never attempted.
  - 2: the event had no usable epochs.
  - 3: it had fewer usable observations than required by `CUTWIN_NEPOCH`.
  - 4: missing or unusable host photometric-redshift quantiles.
  - 5: it failed an initial reduced-χ² cut (`FITWIN_CHI2RED_INI` or `FITWIN_CHI2RED_INI2`).
  - 12: the cuts applied after the fit dropped it.
  - 15, 16 and 17: the first round's probability, shape and colour windows
    rejected it.
  - 93: non-finite fitted parameters.

`OUTLIER`
: One row per epoch whose residual exceeds the `nsig` option. The residual is
  `|F_data - F_fit|` divided by the measurement and model uncertainties added
  in quadrature. The columns identify the epoch (`CID`, `ROWNUM`, `FIELD`,
  `MJD`, `BAND`, `IFILTOBS`) and give both fluxes, their errors and the
  observing conditions. `NSIG` is the
  absolute residual of the measured flux about the simulated true flux,
  divided by the measurement uncertainty; it is 9 for real data. A
  summary of the outlier fraction per band is included in the header.

`LCPLOT`
: One row per epoch for the first `MXLC_PLOT` supernovae, with the data and the
  fitted curve together. For example:

```
VARNAMES: CID MJD Tobs SIM_FLUXCAL FLUXCAL FLUXCAL_ERR DATAFLAG BAND CHI2 IFIT
OBS: 000027 56179.2812 22.45 4.2196e+04 4.0584e+04 1.7207e+02 1 g 0.14 1
OBS: 000027 56188.1758 31.34 2.2327e+04 2.3392e+04 4.1150e+02 1 g 0.75 1
OBS: 000027 56194.3320 37.50 1.5922e+04 1.5936e+04 1.1285e+02 1 g 0.00 1
OBS: 000027 56209.1914 52.36 1.0544e+04 1.0466e+04 1.1159e+02 -1 g 0.00 1
OBS: 000027 56097.8359 -59.00 0.0000e+00 0.0000e+00 0.0000e+00 0 g 0.00 1
```

`DATAFLAG` separates the three kinds of row: 1 for an epoch the fit used, -1
for an observed epoch the fit rejected, and 0 for a point on the fitted curve.
The curve is evaluated on a fixed grid of a hundred epochs, two days apart,
from 59 days before the fitted peak to 139 days after it. On curve rows
`FLUXCAL_ERR` is the model's own flux uncertainty rather than a measurement
error, and `CHI2` is zero. `Tobs` is days from the fitted peak. `SIM_FLUXCAL`
is the true flux, present only for simulated data. `CHI2` is the epoch's
contribution to the fit.

Model-curve rows are written only for bands included in the fit.

(plotting-table-selection)=
`lcplot_from_fitres` evaluates the model at the saved fitted parameters.
It applies `FITWIN_TREST` using the final peak time and redshift, whereas
light-curve fitting applies this window using the first-round values,
which are not stored in FITRES. The resulting `DATAFLAG` values can therefore
differ from those written during fitting. `DELCHI2_REJECT` is not reapplied.

`LCLIST`
: Lists the supernovae in `LCPLOT`, with their `CID` and `IFIT`. Its header
  describes the `LCPLOT` columns.

```text
VARNAMES: CID IFIT
SN: 000027 1
SN: 000038 1
```

## Standardisation

BBC writes an unbinned Hubble diagram as a `.FITRES` table and a binned
Hubble diagram as a `.M0DIF` table. Their headers contain the fitted
standardisation parameters, the reference cosmology and `ISDATA_REAL`.
When available, the BIASCOR simulation cosmology is recorded as
`SIM_OMEGA_MATTER(BIASCOR)`, `SIM_OMEGA_LAMBDA(BIASCOR)` and the corresponding
dark-energy and magnitude-shift keys. These describe the simulation used
for bias corrections, which may differ from the reference cosmology.

(format-bbc-fitres)=
### FITRES: unbinned Hubble diagram

BBC retains the input light curve parameters and adds:

- `CUTMASK`: zero for the fitted sample.
- `MU`: the corrected distance modulus, including standardisation, bias
  corrections and the fitted magnitude zero point. For SALT, the distance
  is estimated from `mB`, shape and colour; for BayeSN, it is based on the
  fitted light curve distance.
- `MUMODEL`: the reference cosmology's distance modulus at the event's `zHD`.
- `MURES`: the residual about the fitted distance relation, including its
  redshift-bin offset. For unblinded output with `uave=1` and binwise offsets
  (`uM0=1`), `MURES = MU - MUMODEL - M0DIF`. With interpolated offsets
  (`uM0=2`), a different magnitude zero point (`uave=0`) or distance blinding,
  this expression does not generally hold for the written columns.
- `MUERR`: the distance uncertainty, including the light curve contribution,
  configured intrinsic scatter, lensing and peculiar-velocity terms, and
  bias-correction variance adjustments.
- `MUERR_RAW`: the light curve and lensing uncertainty, excluding intrinsic
  scatter, peculiar velocities and bias-correction variance adjustments.
- `MUERR_VPEC`: the peculiar-velocity contribution to the distance uncertainty.
- `MUPULL`: `MURES / MUERR`, with positive values capped at 99.999.
- `M0DIF` and `M0DIFERR`: the fitted offset and uncertainty of the event's
  redshift bin, corresponding to `MUDIF` and `MUDIFERR` in the `.M0DIF`.
- `CHI2`: the event's contribution to the minimised objective. For a pure
  type Ia sample this is the squared residual divided by its variance,
  with a log-variance term when enabled. Its sum gives the header's
  `-2log(L)`, rather than the `chi2(Ia)/dof` diagnostic.
- `IZBIN`: the event's redshift bin, starting at 0.

With a contamination model, the table instead uses `CHI2_BEAMS` for the
event's BEAMS likelihood contribution and adds `PROBCC_BEAMS`, its posterior
contaminant probability. Unless all fit parameters are fixed, `MUERR_RENORM`
incorporates the posterior type Ia probability and a normalisation within
each redshift bin, giving `sum(1 / MUERR_RENORM²) = 1 / M0DIFERR²`.

With a BIASCOR simulation, the table also includes:

- `biasCor_mu` and `biasCorErr_mu`: the applied distance correction and its
  uncertainty.
- `biasCor_nevt`: the number of simulated events in the bias-correction
  cells contributing to the interpolation.
- `biasCor_<parameter>`: the correction to each light curve parameter when
  corrections are applied to parameters rather than directly to distance.
  SALT writes `biasCor_mB`, `biasCor_x1` and `biasCor_c`; BayeSN writes
  `biasCor_MU`, equal to `biasCor_mu`.
- `biasCor_muCOVSCALE`: the multiplicative adjustment to the distance
  variance, excluding the peculiar-velocity contribution. It is 1 where
  a positive additive adjustment is applied instead.
- `biasCor_muCOVADD`: the additive variance adjustment in mag², when
  requested through `opt_biascor`.
- `IDSAMPLE`: the sample assigned to the event for bias corrections.

### M0DIF: binned Hubble diagram

For example:

```text
VARNAMES: ROW zHDMIN zHDMAX zHD MUDIF MUDIFERR MUREF NFIT
ROW: 1 0.05000 0.12000 0.08500 -0.00686 0.06179 37.94022 1
ROW: 2 0.12000 0.19000 0.15500 -0.00388 0.02976 39.34394 4
```

- `zHDMIN` and `zHDMAX`: the bin boundaries.
- `zHD`: the redshift at which the bin's distance is reported.
- `MUREF`: the distance modulus of the reference cosmology at `zHD`.
- `MUDIF` and `MUDIFERR`: the fitted offset from `MUREF` and its uncertainty.
  For cosmology fitting, `MU = MUREF + MUDIF` and `MUERR = MUDIFERR`.
- `NFIT`: the number of supernovae in the bin before the minimum-population
  selection.

Bins without a fitted offset, including empty bins, are retained with
`MUDIFERR = 999` and excluded from subsequent cosmology fitting.

## Covariance construction

Covariance construction writes `hubble_diagram.txt`, covariance matrices and
an `INFO.YML` listing the output files. The Hubble diagram can be binned,
unbinned or rebinned, as described in {ref}`covariance-construction`.

(covariance-input-layout)=
### Covariance input

The `create_covariance` command reads the directory layout produced by
SNANA's batch submission. Set `INPUT_DIR` to the submission directory and
`VERSION` to the subdirectory containing the BBC outputs:

```
<INPUT_DIR>/
    SUBMIT.INFO
    <VERSION>/
        FITOPT000_MUOPT000.M0DIF
        FITOPT000_MUOPT000.FITRES.gz
        FITOPT000_MUOPT001.M0DIF
        FITOPT000_MUOPT001.FITRES.gz
        ...
```

A binned run reads the `.M0DIF` files; an unbinned or rebinned run reads
the `.FITRES` files. The nominal `.FITRES` is required in every mode.

`SUBMIT.INFO` is in `INPUT_DIR`, beside the version directory, and gives each
FITOPT and MUOPT its label. It's YAML, with separate lists for the
light-curve-fit and distance-fit variations. Each entry is
the option's directory name followed by two labels. The second label is read
for a FITOPT and the first for a MUOPT. A scale file refers to them by these
labels.

```
FITOPT_OUT_LIST:
- [FITOPT000, DEFAULT, DEFAULT]
- [FITOPT001, ZPSHIFT_CAL, ZPSHIFT_CAL]
- [FITOPT002, LAMSHIFT_CAL, LAMSHIFT_CAL]

MUOPT_OUT_LIST:
- [MUOPT000, DEFAULT, DEFAULT]
- [MUOPT001, CHI2MAX_SCAT, CHI2MAX_SCAT]
- [MUOPT002, SIGINT_SCAT, SIGINT_SCAT]
```

The nominal settings are labelled `DEFAULT`. If either list is absent, only
the reference settings are used for that part of the analysis.

The pipeline writes a different layout, a directory per FITOPT/MUOPT
combination containing files named after the stage, so a pipeline output
directory can't be given to the command directly. Export the results as
described below to use them with `create_covariance`.

(covariance-export-bbc)=
### Exporting BBC results for the command line

To use a completed Python BBC stage from the command line, export its
FITOPT/MUOPT results:

```python
distances.export_covariance_inputs("run", "covariance-inputs")
```

Here `distances` is the BBC stage that produced the run. This copies the
M0DIF and FITRES files into the flat FITOPT/MUOPT layout and writes
`SUBMIT.INFO` with the systematic labels. The destination must be new.
For a BBC stage named `distances`, the covariance YAML then uses:

```yaml
INPUT_DIR: covariance-inputs
VERSION: distances
OUTDIR: covariance-output
```

Add the covariance options to this YAML and run
`create_covariance covariance.yml`.

(external-covariance-files)=
### External covariance files

A text file has columns `CID1 IDSURVEY1 CID2 IDSURVEY2 MU_COV`; an `.npz`
file has arrays `CID`, `IDSURVEY` and `MU_COV`, with the last holding the
upper triangle, including the diagonal. Events are matched by identifier
and survey, so the file need not follow the Hubble diagram's row order.
In a text file, supply both entries of each off-diagonal pair; omitted
entries are zero.

(covariance-output-files)=
### Covariance output

The Hubble diagram's columns depend on the binning and input.
`ISDATA_REAL` is copied from
the reference M0DIF header, or set to -1 when absent.

In binned mode, a reference M0DIF row with uncertainty 999 is excluded from
all variations. Remaining rows retain their original `ROW` identifiers.

Binned output contains `ROW zHD zHEL MU MUERR`, with:

- `MUERR_SYS`, the square root of the diagonal of the `ALL` systematic
  covariance, unless the stage was run with `NOSYS`
- `MUSIM`, the distance modulus of the bias-correction cosmology at `zHD`. It's
  written when the reference `.M0DIF` declares `ISDATA_REAL` as 0, meaning the
  distances came from a simulation. It uses the bias-correction cosmology
  recorded in `COSPAR_BIASCOR`.

`zHEL` is set to `zHD` in binned output. A
rebinned diagram also includes `NEVT`, the number of supernovae in each
group defined by the redshift bin and any additional rebinning columns.
A binned diagram from simulated data reads:

```
# zHD       = redshift in CMB frame with VPEC correction
# zHEL      = zHD
# MU        = distance modulus corrected for bias and contamination
# MUERR     = stat-uncertainty on MU
# MUERR_SYS = sqrt(COVSYS_DIAG) for 'ALL' sys (diagnostic)
# MUSIM     = mu computed with biasCor cosmology (diagnostic)
# ISDATA_REAL: 0   # flag for cosmology fitter to choose blind option
#
VARNAMES: ROW zHD zHEL MU MUERR MUERR_SYS MUSIM
ROW: 1       0.08500 0.08500 37.93336  0.06179   0.00154 37.9382
ROW: 2       0.15500 0.15500 39.34006  0.02976   0.00054 39.3402
```

Unbinned output uses `CID IDSURVEY` to identify events, and `zHEL` is the
heliocentric redshift. `MUERR_VPEC`, the peculiar-velocity part of the error,
is included where available. `PROBIA_BEAMS`, the posterior probability
that the event is a type Ia, is included where
the distances were fitted with a contamination model. `MUSIM` is included
for simulated data as described above.

Covariance matrices are written either as NumPy archives, `covsys_000.npz` and
`covtot_inv_000.npz`, or as gzipped text, `covsys_000.txt.gz`. The archive
stores the row count in `nsn` and only the upper triangle in `cov`, as 32-bit
floats. The text form has the dimension on its first line and then the whole
flattened square matrix, one value per line.

With `WRITE_MASK_COV: 8`, `covtot_factorized_NNN.npz` stores `nsn` (one integer),
`diag` (N float64 statistical variances), `U` (N-by-K float64 systematic
vectors), and `vec_labels` (the K systematic labels). These uncompressed NumPy
archives follow SNANA's factorized format. Zero-column `U` is valid for NOSYS.
The total covariance is `diag(diag) + U @ U.T`.

`INFO.YML` records:

- The Hubble diagram's name and number of rows.
- Whether the data are real, and the binning method.
- The `SNANA_VERSION` string and the data versions the distances came from.
- The directory they were read from, the `SUBMIT.INFO` that labelled them and
  the scale file.
- One line per covopt, giving its label and the files containing its
  systematic and inverted total matrices, followed by an optional fourth field
  naming the factorized file. Unwritten files are `None`. The factorized-file field may be omitted.
- `COSPAR_BIASCOR`, the cosmology the bias-correction simulation was generated
  with. When that's unknown it's written as an empty block.

(format-cosmology-output)=
## Cosmology fitting

The cosmology fit writes a YAML summary and a `.cospar` text file. By default,
the command writes these beside the input Hubble diagram, appending
`.wfit.yaml` and `.cospar` to its name. A pipeline writes them in the
cosmology stage's output directory.

The YAML summary gives the fitted values in `point_estimate` and identifies
the type of estimate in `point_estimate_kind`. The `parameters` block holds
per-parameter means, medians, standard deviations and quantiles. The parameter
covariance and correlation are separate fields, `covariance` and
`correlation`. Three further blocks record fit statistics, convergence
diagnostics and input metadata:

- `fit`: the χ² by term (`chi2_hd` for the supernovae, `chi2_omega_m`,
  `chi2_cmb` and `chi2_bao` for the priors, and `chi2_total` for their sum).
  It also gives the fitted magnitude offset as `intercept`, the degrees of
  freedom `ndof_hd`, and `n_external_constraints`, counting the Ω<sub>m</sub>
  prior where it applies, the CMB and each BAO measurement. The centres of those
  priors are in `external_prior_targets`, and the number of rows fitted is
  `n_rows`.
- `diagnostics`: `success` and a `message`, then the convergence diagnostics of
  the method. For the optimiser, `newton_step_over_error` is the last Newton
  step as a fraction of the parameter error, and `precision_eigenvalues` are the
  eigenvalues of the χ² curvature at the minimum, all positive for an accepted
  fit. For the sampler, the chains' `divergences`, `rhat` and `ess`. For the
  grid, `edge_probability_ratio` is the posterior at each end of each axis
  relative to its peak, indicating whether the posterior extends to the
  grid boundaries.
- `input`: the file fitted, the kind of covariance given, the row numbers that
  survived the redshift cuts as `retained_raw_indices`, the header of the
  diagram, and the simulated truth where one was given.

`assumptions` records `radiation` for distances and the radiation-free prior
conventions as `cmb_radiation: false` and `bao_dh_radiation: false`.
`configuration` and `provenance` record the resolved settings and package
versions.

The `.cospar` file is two lines: a comment naming the fields, and a single row
of values. For example, a `wcdm` fit reports:

```
# w wsig_marg om omsig_marg rho_wom FoM chi2 Ndof sigint wran omran label
-0.95619  0.16826  0.24973  0.04759  0.839  229.4 0.3     2  0.000 0 0 none
```

Under `lcdm` and `wcdm` the header is the one above. Under `w0wa` the
equation of state has two parameters, and the fields are:

```
# w0 w0sig_marg wa wasig_marg om omsig_marg rho_w0om rho_w0wa rho_waom FoM chi2 Ndof sigint w0ran waran omran label
```

The columns are:

- each parameter with its uncertainty;
- the correlations between them. Under `w0wa` there are three: `rho_w0om`,
  `rho_w0wa` and `rho_waom`;
- the figure of merit `FoM = 1 / (sigma_w * sigma_partner * sqrt(1 - rho**2))`,
  where `rho` is the correlation between *w* and its partner. The
  partner is Ω<sub>m</sub> under `wcdm` and *w*<sub>a</sub> under `w0wa`. It is
  written as -9 under `lcdm`, where *w* is not fitted;
- the total χ², and its degrees of freedom `Ndof`: the number of rows less the
  number of parameters and one for the offset, plus one for the CMB and one per
  BAO measurement. The Ω<sub>m</sub> prior does not count, as in `wfit.exe`.
  This is `ndof_hd` in the YAML;
- `sigint`, written as zero by jax-SNANA;
- the blinding integers and the `label`.

Under `-minchi2`, which reports the best-fitting point rather than the
posterior mean (see {doc}`cosmology`), `wsig_marg` is replaced by `wsig_lo` and
`wsig_up`. These are `mean - q16` and `q84 - mean`, measured from the
posterior mean even when the reported point is the best fit. They should not
be interpreted as interval endpoints relative to that best fit for an
asymmetric posterior. The same goes for `wa`, but the
matter density keeps `omsig_marg` either way. A parameter the chosen model
keeps fixed still gets its fields, with a zero uncertainty. The blinding
integers are zero for an unblinded fit.

Optional outputs are selected with:

- `-outfile_resid`: the per-supernova residuals about the fitted cosmology.
- `-outfile_chi2grid`: the full chi-square grid. Grid method only.
- `-outfile_prob1d`: the marginalised one-dimensional probabilities. Grid
  method only.
- `--output-samples`: the posterior draws as a NumPy archive. Sampler only.
- `--output-covariance` and `-outfile_mucovtot_inv`: the total distance
  covariance the fit used, and its inverse. Each is written as the dimension on
  the first line, then the flattened matrix, one value per line.
