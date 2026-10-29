# Standardisation settings

These settings configure BBC standardisation, bias corrections and
contamination modelling. For worked input files, see
{doc}`bias_corrections`.

```{contents} On this page
:local:
:depth: 2
```

In a SALT2mu input file, write settings as `key=value`. Command-line
overrides use the same notation. Multiple values are comma-separated,
without spaces, for example `zbinuser=0.1,0.25,0.4`.

YAML and Python use the **Python / YAML** names in the full settings tables
below. Command-line options replace underscores with hyphens, for example
`--nzbin 10` or `--nuisance-val alpha0 0.14`.

Selection windows use `CUTWIN <column> <low> <high>`. The numbered
parameters, `p<N>` and `u<N>`, are listed under
{ref}`bbc-numbered-parameters`, since the parameter each number names
depends on the distance model. The keys a distance model adds to this fit,
such as SALT's `MAXERR_ABORT_X0`, are listed with the model on
{doc}`model_settings`.

## Redshift binning

Specify the bins with `nzbin` for equal widths in redshift, `nlogzbin` for
equal widths in log redshift, or `zbinuser` for explicit edges. With
`nzbin` or `nlogzbin`, also supply `zmin` and `zmax`. Explicit edges set
both the range and number of bins.

`powzbin` varies the widths in proportion to `(1 + z)^power`. The input
syntax `powzbin=power,znhalf` optionally specifies the redshift above which
bins return to equal widths. In YAML and Python, use separate `powzbin`
and `znhalf` settings.

(bbc-biascor-options)=
## Bias-correction options

Set `opt_biascor` to the sum of the required values:

| Value | Behaviour |
|---|---|
| 1 | Redshift-only mean distance-bias correction, with equal event weights. |
| 2 | Redshift-only correction weighted by inverse distance variance. |
| 16 | Multidimensional correction in the model's bias-map coordinates. |
| 32 | Rescale distance variances using the BIASCOR simulation. |
| 64 | Build separate maps for the survey samples described below. |
| 128 | Correct distance modulus directly rather than the light-curve parameters. |
| 2048 | Estimate the variance scale as the square of `1.48 × median(abs(pull))`, rather than the mean squared pull. |
| 4096 | Derive an additive intrinsic variance in each cell; also enables 2048. Requires a model with standardisation coefficients, so it is not available for BayeSN. |

With a BIASCOR sample, the default is 112 (`16 + 32 + 64`). Without one,
no correction is applied. Select either a redshift-only correction or the
multidimensional correction; they cannot be combined.

## Survey samples

With separate correction maps enabled, BBC divides the data by survey.
`IDSAMPLE` identifies these samples in the output FITRES. Surveys can be
combined using `surveygroup_biascor`; commas separate groups and `+` joins
surveys within a group. For example:

```text
surveygroup_biascor=CFA3S+CFA3K,CSP,DES
```

The equivalent YAML setting is:

```yaml
surveygroup_biascor: CFA3S+CFA3K,CSP,DES
```

`fieldgroup_biascor` further divides samples by field, using the same
comma-separated grouping syntax. Names match substrings of `FIELD`;
vertical bars, as in `|C3|`, require an exact match. Alternatively,
`fieldlist_biascor` lists exact field names for separate samples. Setting
`zspec_errmax_idsample` also separates spectroscopic and photometric
redshift samples at the specified redshift uncertainty.

Groups can have their own map bin widths, for example
`DES(zbin=0.05:x1bin=0.5:cbin=0.05)`. Here colons separate the bin-width
settings within the group.

(bbc-fixed-scatter)=
## Fixed scatter by sample

`sigint_fix` supplies intrinsic scatters in magnitudes in increasing
`IDSAMPLE` order. A single value applies to every sample. The first
value sets the starting scatter. Per-sample values apply when
`fitflag_sigmb > 0`, the correction uses the multidimensional map (bit 16),
and additive cell scatter (bit 4096) is off. The scatter iteration stops
after its first fit when `sigint_fix` is supplied. To set one fixed scatter
without iteration, use `sig1`.

## Named map coordinates

Bias-correction maps, distance-error maps and simulated contaminant priors
can use different coordinates. Use parameter names to change the model's
error-map and contaminant-prior bin counts or its selection windows:

```text
mucovscale_bins.c=3
ccprior_bins.x1=2
lcpar_range.x1=-3,3
```

Names are case-sensitive (`AV` and `THETA` for BayeSN). A bin count of zero
omits a coordinate; one retains a single bin and the coordinate's
interpolation weighting in the error map. The named windows
also set the ranges of matching error-map and contaminant-prior coordinates;
`CUTWIN` changes selection only. Omitting a coordinate does not remove its cut.

The corresponding SALT2mu keys are:

| Key | SALT equivalent | BayeSN equivalent |
|---|---|---|
| `nbinc_mucovscale` | `mucovscale_bins.c` | `mucovscale_bins.AV` |
| `nsbin_ccprior` | `ccprior_bins.x1` | `ccprior_bins.THETA` |
| `ncbin_ccprior` | `ccprior_bins.c` | `ccprior_bins.AV` |

Command-line settings override file settings. If aliases and named settings
address the same parameter, the last assignment wins.

In Python, use `BBCConfig(mucovscale_bins={"c": 3}, ccprior_bins={"x1": 2})`.

(bbc-numbered-parameters)=
## Numbered parameters

`p<N>` sets a parameter's value and `u<N>` controls whether it is fitted.
For example, `p1=0.14 u1=1` starts α at
0.14 and fits it, while `p1=0.14 u1=0` holds α at 0.14 throughout. A third
value, `u<N>=3`, fits the parameter as 1 does, but gives the bias-correction
map a single bin in it, as 0 does, whatever values the simulation was generated
at.

The available standardisation parameters depend on `model_lcfit`.

Parameters 1 to 8, 15 and 16 for SALT:

| Key | Parameter | Description |
|---|---|---|
| `p1` | `alpha0` | Coefficient of stretch in the Tripp relation, α. Starts at 0.14 when fitted. |
| `p2` | `beta0` | Coefficient of colour, β. Starts at 3.2 when fitted. |
| `p3` | `dalpha_dz` | Linear dependence of α on redshift. |
| `p4` | `dbeta_dz` | Linear dependence of β on redshift. Also accepted as `beta1`. |
| `p5` | `gamma0` | Size of the host-mass step, in magnitudes. |
| `p6` | `dgamma_dz` | Linear dependence of that step on redshift. |
| `p7` | `logmass_cen` | Host mass at which the step is centred, in log solar masses. Default 10. |
| `p8` | `logmass_tau` | Host-mass transition width, in log solar masses. Starts at 0.01 when fitted. |
| `p15` | `dalpha_dlogm` | Linear dependence of α on host mass. Also accepted as `alphaHost`. |
| `p16` | `dbeta_dlogm` | Linear dependence of β on host mass. Also accepted as `betaHost`. |

Each parameter may also be given by name, e.g. `dalpha_dz=0.01`.

BayeSN's light-curve fit already produces a distance, so these SALT parameters
do not apply under `model_lcfit=BAYESN`.

Reference-cosmology and contamination parameters are shared by all models:

| Key | Parameter | Description |
|---|---|---|
| `p9` | `Omega_L` | Dark energy density of the fixed reference cosmology. |
| `p10` | `Omega_k` | Curvature density of the reference cosmology. |
| `p11` | `w0` | Dark-energy equation-of-state parameter of the reference cosmology. |
| `p12` | `wa` | The CPL coefficient in `w(a) = w0 + wa(1 - a)`. |
| `p13` | `scalePCC` | Weight given to the core-collapse component of a BEAMS fit. |
| `p17` to `p19` | `H11mucc0` to `H11mucc2` | Coefficients of the Hlozek et al. (2012) polynomial for the mean core-collapse residual. |
| `p20` to `p22` | `H11sigcc0` to `H11sigcc2` | Polynomial coefficients for the width of the core-collapse residual distribution. |

In the {ref}`H11 contamination model <bbc-contamination>`, all six polynomial
coefficients float regardless of the individual `u17`–`u22` settings, unless
`fixpar_all` holds all parameters fixed.

The reference cosmology is fixed during the distance fit, so leave `u9` to
`u12` at zero. Parameters may also be given by name, e.g. `scalePCC=1.2`.

For intrinsic scatter, use `sig1`, `sigint_fix` or `sigint_mode=joint`;
`p14` and `u14` are not supported.

Three of the `u<N>` keys have a second meaning:

- `u13=2` applies `scalePCC` as a Bayes factor on the classifier's probability,
  in the form of Hlozek et al. (2012), instead of as a rescaling of the class
  weights.
- `u15=2` or `u16=2` gives α and β a step in host mass at `logmass_cen`, of
  size `dalpha_dlogm` and `dbeta_dlogm`. While either flag is at most one, the
  linear slope in host mass applies as well, so both can act at once.

For SALT's standardisation coefficients, a held parameter given no `p<N>`
is held at zero, except `logmass_cen`, which is held at 10. Thus
`u1=0 u2=0` alone uses α = 0 and β = 0. Fitted parameters without a supplied
starting value use the model defaults, so `u1=1 u2=1` fits α and β from
their default starting values.

## Full settings

The tables give Python/YAML names alongside the accepted SALT2mu
spellings. Unrecognised YAML and Python settings are rejected;
unrecognised SALT2mu keys produce a warning and are ignored. Explicitly
unsupported keys are listed under {ref}`unsupported-distance-keys`.

```{eval-rst}
.. configtable:: jax_salt2mu.config.BBCConfig
   :field-keys:
   :field-key-case: preserve
   :keys-from: jax_salt2mu.config
   :aliases-from: jax_salt2mu.config
   :omit-sections: Set by the keys the model declares, Not set by an input key
```

## Python pipeline settings

The following settings configure the `BBC` pipeline stage. They are
separate from the BBC input-file settings above. For Pippin inputs, see
{doc}`pippin_settings`.

(bbc-event-synchronisation)=
### Event synchronisation

`OPT_SNCID_LIST=1` on the input light-curve fits enables a common sample
across systematic variations, as described in {doc}`systematics`.

Light curve FITOPTs then use the nominal fit's accepted events instead of
reapplying their pre-fit selection windows. Ignore lists still apply.
Post-fit selection rejects an event if its shape parameter reaches the
edge of its flat prior or its fit probability falls below a tenth of the
`FITWIN_PROB` lower bound, where specified.

BBC starts from the nominal accepted sample and retains events accepted
across the selected FITOPT/MUOPT combinations. It then fits each combination
using this common sample. BBC selection is evaluated before fitting,
including `chi2max` at the starting parameters. With an accepted-event list,
`CUTWIN NONE` disables parameter/host windows, chi-squared cuts and
minimum-bin-population cuts, as listed in the settings table.

A FITOPT whose label contains `NOREJECT` applies its own selection and does
not restrict the common sample. A MUOPT whose label contains `NOREJECT`
also does not restrict that sample, but uses it in the final fit unless
its FITOPT is exempt too.

The input fits must agree on the synchronisation setting. Imported fit
versions supply it through `FLAG_USE_SAME_EVENTS` or `OPT_SNCID_LIST` in
`SUBMIT.INFO`. To run independent BBC selections, set `sync_events=False`
on the `BBC` stage. This can be useful for MUOPTs intended to compare
different event samples.

Each realisation and sample split has its own common sample, matched by
CID, survey and field. `BBC_ACCEPT_SUMMARY.LIST` records the common events.

(bbc-systematic-combinations)=
### FITOPT–MUOPT selection

`fitopt_muopt` selects FITOPT and MUOPT numbers; 0 denotes the nominal fit.
`i+j` selects FITOPT `i` or MUOPT `j`; `i&j` selects only their intersection.
A list selects the union. For example, `["0+0"]` pairs every FITOPT with
MUOPT 0 and every MUOPT with FITOPT 0.
Omitting the argument runs the full grid. Selected combinations retain the
numbering of the full grid in exported files.

For covariance construction, include the nominal combination and each
selected FITOPT with MUOPT 0. These supply the reference distances.

#### Multiple surveys

A distance fit combining several surveys uses the union of their FITOPT
labels, in the order the labels first appear. A shared label varies those
surveys together. A label belonging to one survey varies that survey while
the others contribute their nominal fits, following Pippin's `FITOPT_MAP`.

A MUOPT supplying an alternative `datafile` does not add FITOPT labels.
That data source must use labels present in the nominal data.

(systematics-correction-fits)=
### Selecting BIASCOR and contamination fits

The BBC stage uses nominal fits of the BIASCOR and contamination
simulations by default. To use the BIASCOR simulation's `mwebv` FITOPT
instead, select its label:

```python
distances = BBC(
    "distances", data=data_fit, biascor=biascor_fit, input_file="bbc.input",
    biascor_fitopts=["mwebv"],
)
```

Supply one label per BIASCOR input; `None` selects that input's nominal
fit. Labels refer to FITOPTs defined on the BIASCOR fitting stage.
`ccprior_fitopts` selects fits of the contamination simulations in the same
way, with one label per contamination input.

A particular data FITOPT can instead use an external BIASCOR FITRES file.
For example, to use a separate table for the `zpt_g` variation:

```python
distances = BBC(
    "distances", data=data_fit, biascor=biascor_fit, input_file="bbc.input",
    replace_biascor={"zpt_g": "corrections/zpt_g.FITRES"},
)
```

Other data FITOPTs use the BIASCOR simulation's nominal fit, or the fit
selected by `biascor_fitopts` if supplied. `replace_ccprior` provides the
equivalent replacement for contamination simulations.

A MUOPT supplying a BIASCOR or contamination simulation takes precedence
over a FITOPT-specific replacement and uses that simulation's nominal fit
by default. Set
`muopt_biascor_fitopts={"alternative_population": ["mwebv"]}` to select a
fit of that MUOPT's BIASCOR simulation, or use `muopt_ccprior_fitopts` for
contamination simulations.
