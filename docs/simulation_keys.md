# Simulation settings

This page lists the simulator's settings. YAML and Python use the
**Python / YAML** names; CLI options replace underscores with hyphens, such
as `--ngentot-lc`. The **SNANA key** column gives the keyed-text spelling.
{doc}`simulating` introduces both input forms and running from Python.

Ordinary YAML settings are top-level entries. Population distributions and
subpopulations use the structured `populations` and `subpops` entries described
in {doc}`supernova_populations`.
Unrecognised YAML and Python settings are rejected; unrecognised keys in
SNANA input files produce a warning and are ignored.

```{contents} On this page
:local:
:depth: 1
```

## Setting values and flags

Settings are grouped by what they control. The tables give the Python/YAML
name, SNANA spelling, type, default and description. Value tables alongside
the relevant settings explain flags in more detail:

- **Choose one value** for options that select a mode.
- **Combine bit values** for masks that enable several behaviours. For
  example, host options 2 and 8 together give `HOSTLIB_MSKOPT: 10`.

`GENSOURCE: RANDOM` selects survey simulation. `GENMEAN_<PAR>` is accepted
in place of `GENPEAK_<PAR>` when the latter is not given.

## Run size and random seeds

These settings control the size of the simulation, its random seed and
initialisation-only runs. You can specify the size of a simulation in three
ways:

- `NGEN_LC` generates until that many events have survived the trigger and
  the cuts.
- `NGENTOT_LC` generates a fixed number before selection.
- `NGEN_SEASON` specifies a number of survey seasons. The rate model,
  survey area and season duration determine the expected event count.

`NGEN_SCALE` multiplies whichever is given.

A non-zero `INIT_ONLY` stops after full initialisation, without generating
events or writing simulation output.

`SNDATA_ROOT` locates the SNANA data tree; see {doc}`installation` for setup.
`PATH_SNDATA_SIM` sets the parent directory for simulation output; the
default is the working directory. `--output` changes the name of the data
version written there.

```{eval-rst}
.. configtable:: jax_snlc_sim.simulation.Run
   :field-keys:
   :keys-from: jax_snlc_sim.inputs
   :omit-sections: Set by the program rather than by an input key
```

### Python execution settings

These settings control batching and numerical preparation in YAML and Python.
They also have CLI forms such as `--batch-size`.

| Setting | Default | Meaning |
|---|---|---|
| `batch_size` | 1000 | Number of events simulated in each batch. |
| `wavelength_grid_points` | 500 | Requested wavelength-grid resolution for models that construct a working grid; models using their own tabulated grid determine its length. |
| `n_sne` | unset | Number of events prepared for an in-memory simulation; if unset, derived from the configured generation count. |
| `keep_rejected_events` | false | Retain rejected events in the in-memory result for diagnostics or inference. |

## Rates and generation ranges

These keys set the volumetric rate, the ranges the true redshift and time of
peak are drawn from, and the phase range each light curve is generated over.
`GENRANGE_REDSHIFT` is required and has no default.

A `DNDZ` value is a model name followed by its arguments. The following
are alternative rate settings; `DNDZ_FILE` supplies a tabulated rate instead:

```text
DNDZ:  POWERLAW  2.6E-5  1.5
DNDZ:  POWERLAW2  2.6E-5  1.5  0.0  1.0
DNDZ:  MD14  2.6E-5
DNDZ:  0.7*CC_S15
DNDZ:  ZPOLY  2.6E-5,1.0E-5
DNDZ_FILE:  $SNDATA_ROOT/models/rates/my_rate.dat
```

Every rate is in events per comoving Mpc cubed per rest-frame year. The
models are:

- `POWERLAW A B`: $A(1+z)^B$ over the whole of
  `GENRANGE_REDSHIFT`.
- `POWERLAW2 A B zmin zmax`: the same over the interval `zmin`–`zmax`.
  Repeat the setting to specify a piecewise power law.
- `MD14 R0`: the Madau and Dickinson (2014) star-formation history, scaled to
  R0 at redshift zero.
- `CC_S15`: the Strolger et al. (2015) core-collapse rate. It takes no
  argument, but accepts a multiplier written before or after the name.
- `PISN_PLK12`: a pair-instability rate, taking a multiplier the same way.
- `TDE R0`: a tidal-disruption rate falling with redshift, normalised to
  `R0` at redshift zero.
- `ZPOLY coefficients`: a polynomial in redshift, with comma-separated
  coefficients starting at the constant term.
- `ABMODEL A B`: a two-component rate with coefficients for stellar mass
  and star formation, respectively.
- `HUBBLE` and `FLAT`: constant volumetric rates of 3e-5 and 1e-8 events
  per comoving Mpc³ per rest-frame year, respectively.
- `DNDZ_FILE`: a two-column table of redshift and rate, instead of any of
  the above.

Specify one rate model. `DNDZ_SCALE`,
`DNDZ_ALLSCALE`, `DNDZ_ZEXP_REWGT` and the two `REWGT` polynomials reweight
that model.

The observed time of peak is estimated from the photometry, by default from
high signal-to-noise observations within a time window around the brightest
epoch. `OPT_SETPKMJD` selects the method. The cuts and the output use this
estimate, and the true time of peak is written as a truth column.

`GENRANGE_PEAKMJD_EXCLUDE` cuts a window, such as a gap between seasons, out
of the peak range, and its span is subtracted from the season length used for
the expected count.

`GENRANGE_REDSHIFT` bounds the cosmological redshift, so the reported
heliocentric redshifts extend slightly beyond both ends because of peculiar
velocities and the Sun's motion towards the CMB apex.

```{eval-rst}
.. configtable:: jax_snlc_sim.simulation.Rates
   :omit-sections: Fixed, with no input key of its own
   :field-keys:
   :keys-from: jax_snlc_sim.inputs
```

## Survey observations and detection

The SIMLIB is the survey's observing log, one entry per sightline, giving the
epochs, depth and conditions of every observation the survey made there. The
keys in the first group choose which entries are used and how they are
reused. The kcor file supplies the filter transmissions and the calibration,
and `GENFILTERS` names the bands to generate from it.

The trigger keys set which generated events the survey would have found.
`APPLY_SEARCHEFF_OPT` is a mask over three stages: the detection pipeline, the
spectroscopic follow-up and the host redshift. An event must meet every requirement
the mask selects. The efficiency files supply each stage's probability, and the
shift keys move those probabilities without editing the files, for simulating
a selection systematic. An event whose field no spectroscopic or
host-redshift map covers stops the run, unless the map file sets
`OPT_FIELDMATCH_REQUIRE: 0`.

A SIMLIB can also specify event parameters. The `USE_SIMLIB` keys take
the redshift, the time of peak, the shape and colour or the distance from the
entry instead of drawing them, to replay a known set of events on the cadence
they were observed with.

### Settings

```{eval-rst}
.. configtable:: jax_snlc_sim.simulation.Survey
   :field-keys:
   :keys-from: jax_snlc_sim.inputs
   :omit-sections: Fixed, with no input key of its own
```

(sim-detection-options)=
### Detection requirements: `APPLY_SEARCHEFF_OPT`

Combine bit values to select which stages an event must pass.

| Value | Behaviour |
| --- | --- |
| 1 | Detection pipeline and its trigger requirements. |
| 2 | Spectroscopic follow-up. |
| 4 | Host-galaxy redshift measurement. |
| 8 | Either spectroscopic follow-up or a host-galaxy redshift measurement. |

Default: **0**. Efficiencies are evaluated and reported, but do not reject
events. With a non-zero mask, every selected requirement must be met.

(sim-simlib-options)=
### Observing-log options: `SIMLIB_MSKOPT`

Combine bit values to change scheduling and noise treatment.

| Value | Behaviour |
| --- | --- |
| 4 | Stop after using the available SIMLIB entries, without cycling through them again. |
| 8 | Draw template noise independently for each observation rather than sharing its draw. |
| 16 | Omit template noise. |
| 32 | Ignore the `FLUXERR_COR` map in the SIMLIB header. |
| 128 | Use the observing season containing the event's peak. |
| 256 | Use the entire observing history of the selected SIMLIB entry. |

Default: **0**. Supplying `FLUXERRMODEL_FILE` sets bit 32 automatically.
Bit 256 takes precedence over 128.

(sim-exposure-options)=
### Exposure scaling: `EXPOSURE_TIME_MSKOPT`

Combine bit values to choose what follows the `EXPOSURE_TIME` ratio.

| Value | Behaviour |
| --- | --- |
| 1 | Photometric zero point. |
| 2 | Sky noise. |
| 4 | Read noise. |

Default: **7**, which selects all three.

## Models and populations

`GENMODEL` names the spectral energy distribution (SED) model. You can use
SALT or BayeSN for type Ia supernovae, a library of templates for the other
supernova types, a grid of tabulated spectra indexed by model parameters
through SIMSED, or one of the `redback-jax` models for other transients.

{doc}`model_settings` lists model-specific parameters and options. The
settings below specify population distributions, magnitude offsets and
intrinsic scatter. For worked examples, including evolution, correlations
and `SUBPOP`, see {doc}`supernova_populations`.

Parameter suffixes are case-insensitive: for example, `GENPEAK_SALT2x1`
sets the SALT shape population and `GENPEAK_THETA` the BayeSN one.
`GENPDF_FILE` supplies tabulated distributions in the format described in
{doc}`simulation_inputs`. It can also set `GENMAG_SMEAR` and `MAG_OFFSET`;
if it sets `GENMAG_SMEAR`, omit that setting from the simulation input.

(sim-population-settings)=
### Population distributions

The `populations` mapping accepts the fields below. Distribution choices
and examples are described in {doc}`supernova_populations`.

| YAML field | SNANA key | Meaning |
|---|---|---|
| `peak` | `GENPEAK_<PAR>` or `GENMEAN_<PAR>` | Peak of the distribution. |
| `sigma` | `GENSIGMA_<PAR>` | Widths below and above the peak, as a pair. |
| `range` | `GENRANGE_<PAR>` | Lower and upper limits; required in YAML. |
| `peakrange` | `GENPEAKRANGE_<PAR>` | Lower and upper edges of a flat plateau, with Gaussian tails. |
| `skew` | `GENSKEW_<PAR>` | Change in the width with distance from the peak, on each side. |
| `prob2` | `GENPROB2_<PAR>` | Probability of drawing from a second Gaussian component. |
| `peak2` | `GENPEAK2_<PAR>` | Peak of the second component. |
| `sigma2` | `GENSIGMA2_<PAR>` | Widths below and above the second peak. |
| `ngrid` | `GENGRID_<PAR>` | Snap draws to this many equally spaced points across the range; 0 or 1 disables snapping. |

For YAML, the optional scalar fields default to zero and pairs to `[0, 0]`.
SNANA inputs use the model's declared population defaults for omitted
components. The normal profile shares these fields, with its single width
applied on both sides.

Host extinction uses `exponential`, or `exponential_gaussian` for a combined
exponential and Gaussian density:

| YAML field | SNANA key | Meaning |
|---|---|---|
| `exp_tau` | `GENTAU_<PAR>` | Exponential scale; required for `exponential`. |
| `peak` | `GENGAUPEAK_<PAR>` | Gaussian peak; a non-positive peak selects a half-Gaussian starting at zero. |
| `sigma` | `GENSIGMA_<PAR>` or `GENSIG_<PAR>` | Gaussian width. |
| `ratio` | `GENRATIO_<PAR>` | Ratio of the Gaussian and exponential density amplitudes when both are used. |
| `range` | `GENRANGE_<PAR>` | Lower and upper limits; required. |

These optional YAML fields default to zero. `<PAR>` is `AV` or `EBV_HOST`
for these dust profiles; `RV` uses the Gaussian family.

`prob_expon_rewgt` (default 1) changes the sampling density for an individual
YAML population. `GENPDF_EXPON_REWGT` applies the exponent to all SNANA
population distributions and GENPDF maps. See
{ref}`population reweighting <population-reweighting>` for the output weights.
For host-dependent shifts, `VARIATION` uses
`<target> <quantity> POLY <coefficients> PIVOT <value>`; its
{ref}`population-guide examples <simulating-variation>` explain the targets
and polynomial convention. `ZVARIATION` and `SUBPOP` are described there too.

### Settings

```{eval-rst}
.. configtable:: jax_snlc_sim.simulation.Model
   :field-keys:
   :keys-from: jax_snlc_sim.inputs
   :omit-sections: Resolved from GENMODEL, Set by the model's own keys
```

(sim-model-options)=
### Model generation options: `GENMODEL_MSKOPT`

This bit mask has model-specific meanings. See the value tables for
{ref}`SALT <salt-generation-options>` and {ref}`BayeSN <bayesn-generation-options>`.
The default is **0**. Bit 16 sets negative SED flux to zero in simulated
spectra for every model, and is enabled by `ATMOSPHERE_OPTMASK`.

(sim-population-map-options)=
### Population-map boundaries: `GENPDF_OPTMASK`

This bit controls conditioning values outside a population map.

| Value | Behaviour |
| --- | --- |
| 1 | Use the nearest boundary value when a conditioning value lies outside the map. |

Default: **0**. Out-of-range conditioning values are rejected.

## Host galaxies

`HOSTLIB_FILE` supplies the galaxy catalogue used for host association,
host photometry and redshifts. The settings below control host selection,
placement of events within galaxies, association with nearby galaxies and
host photometric-redshift distributions. {ref}`simulating-hosts` describes
the catalogue requirements and a worked configuration.

This group also includes the host-extinction distribution. Its exponential
and Gaussian components are listed under {ref}`sim-population-settings`;
{doc}`supernova_populations` gives examples of host dust and R_V populations.

### Settings

```{eval-rst}
.. configtable:: jax_snlc_sim.simulation.Host
   :field-keys:
   :keys-from: jax_snlc_sim.inputs
   :omit-sections: Turned on by naming any host-dust key
```

`HOSTLIB_MINDAYSEP_SAMEGAL` limits how closely spaced events can reuse a
host galaxy. A rejected event's host becomes available again at the batch
boundary, so batch size affects reuse. Use `batch_size=1` to release hosts
after each rejected event.

(sim-host-options)=
### Host-library options: `HOSTLIB_MSKOPT`

Combine bit values to select how the host catalogue is used.

| Value | Behaviour |
| --- | --- |
| 2 | Include host surface-brightness noise; requires the galaxy's Sersic profile. |
| 4 | Apply the catalogue's `SNMAGSHIFT` magnitude shift. |
| 8 | Use the host's sky position for the event. |
| 16 | Use the host's redshift for the event. |
| 32 | Use each galaxy at most once. |
| 64 | Read event parameters from catalogue columns. |
| 128 | Replace the true redshift with the photometric redshift. |
| 512 | Read the peculiar velocity from the catalogue. |
| 65536 | Build Gaussian photo-z quantiles from `ZPHOT` and `ZPHOT_ERR`. |

Default: **0**.

(sim-noise-settings)=
## Calibration and noise

The noise settings modify the photometric uncertainties and noise draws,
including contributions from host light and the template image.

- `SMEARFLAG_FLUX=0` produces noiseless fluxes while retaining their errors.
- `SMEARFLAG_ZEROPT` sets whether each observation's zero-point uncertainty
  enters the noise drawn, the error reported, or both.

Additional scatter can arise from subtraction residuals and imperfect PSF
models, particularly over bright host galaxies. `FLUXERRMODEL_FILE` supplies a map of
the extra error against observing conditions and host surface brightness,
measured from the survey's own difference imaging. Its correlated form makes
the extra errors of the bands in one epoch correlated.

The fudge keys are for simulating systematics. Some change the measurement
itself: `GENMAG_OFF_ZP` shifts a band's calibration and `FUDGESHIFT_LAM`
shifts its transmission in wavelength. The `RANSYSTPAR` keys draw them for a
whole run. Two settings control the flux uncertainties:

- `FUDGESCALE_FLUXERR` scales the noise drawn and the error reported
  together.
- `FUDGESCALE_FLUXERR2` scales the reported error alone, so the errors no
  longer match the scatter. Use it to test how an analysis responds to
  mis-stated errors.

The remaining keys cover saturation, where an observation of a bright event
is discarded because its central pixel exceeded the well depth, and the
atmosphere. `ATMOSPHERE_OPTMASK` computes the airmass of every observation
and the shift in apparent position from differential chromatic refraction,
which depends on the event's spectrum. The shift is written to the output and
does not change the flux.

### Settings

```{eval-rst}
.. configtable:: jax_snlc_sim.simulation.Instrument
   :field-keys:
   :keys-from: jax_snlc_sim.inputs
   :omit-sections: Fixed, with no input key of its own
```

(sim-atmosphere-options)=
### Atmosphere options: `ATMOSPHERE_OPTMASK`

Combine bit values to select atmospheric calculations and output.

| Value | Behaviour |
| --- | --- |
| 1 | Compute airmass and the coordinate shift from differential chromatic refraction. |
| 512 | Write the per-observation `.DCR` table. |

Default: **0**, with no atmospheric calculation. Any non-zero value enables
the atmospheric output columns and true-SED spectra.

## Selection cuts

The cuts are the analysis's requirements on a light curve, such as a
pre-maximum observation, a minimum signal-to-noise, no long gaps or a peak
magnitude in range. They are applied to the events that pass the trigger
keys of the survey section, and the run's summary reports the trigger and
the cuts separately.

Most cuts are `CUTWIN` keys, each naming a quantity and the window it must
fall in. Any per-event quantity the simulation computes can be cut on this
way. A few `CUTWIN` keys take more than a
single window. `CUTWIN_SNRMAX`, the most used, names a set of bands, a
requirement over them and a rest-frame window, and may be given up to five
times, every occurrence having to pass.

The `EPCUTWIN` keys set which epochs count towards the cuts, dropping
those whose rest-frame wavelength falls outside the model and those whose
signal-to-noise is too low.

### Settings

```{eval-rst}
.. configtable:: jax_snlc_sim.simulation.Cuts
   :field-keys:
   :keys-from: jax_snlc_sim.inputs
```

(sim-cut-options)=
### Applying cuts: `APPLY_CUTWIN_OPT`

Choose one value; this option is not a bit mask.

| Value | Behaviour |
| --- | --- |
| 0 | Evaluate cuts and record `CUTMASK`, without rejecting events on those cuts. |
| 1 | Reject events that fail the cuts. |
| 3 | Reject events that fail, but include them in the simulation dump. |

Default: **0**. The epoch-count requirement of `CUTWIN_NEPOCH` also applies
during generation, as described in {doc}`simulating`.

## Output

`GENVERSION` names the simulation output directory. {doc}`analysis_outputs`
describes the light curves, spectra and summary files written there.

The dump is a table with one row per event and the columns `SIMGEN_DUMP` or
`SIMGEN_DUMPALL` names, written as a count followed by that many names:

```text
SIMGEN_DUMPALL: 6 CID GENZ PEAKMJD MU SALT2x1 SALT2c
```

`FLAG_ACCEPT` is added to the requested columns. Model parameters must be
requested explicitly, except for SIMSED, which adds its parameters.
{ref}`simulation-dump-columns` lists the available names. `SIMGEN_DUMP`
contains accepted events and, with `APPLY_CUTWIN_OPT=3`, those rejected by
the cuts. `SIMGEN_DUMPALL` contains every generated event. The table below
also lists the diagnostic dumps and their settings.

The remaining keys set the event identifiers (CIDs). CIDs are sequential by
default, so two runs that are combined can share CIDs. `CIDOFF` offsets one
run's CIDs past another's, and `FORMAT_MASK` bit 16 draws the CIDs at random
from the range the `CIDRAN` keys define.

### Settings

```{eval-rst}
.. configtable:: jax_snlc_sim.simulation.Output
   :field-keys:
   :keys-from: jax_snlc_sim.inputs
```

(sim-output-options)=
### Output options: `FORMAT_MASK`

Select one light-curve format, then add any required output bits.

| Value | Behaviour |
| --- | --- |
| 2 | Write one TEXT file per event. |
| 32 | Write FITS header and photometry tables. |
| 8 | Omit simulated-truth columns for a blind test. |
| 16 | Assign random event identifiers. |
| 64 | Keep only five photometry columns in FITS output. |
| 128 | Include atmospheric columns. |
| 256 | Write filter transmissions alongside the data. |
| 2048 | Suppress spectrum output. |

Default: **32** (FITS). Bits 2 and 32 are mutually exclusive.
`ATMOSPHERE_OPTMASK` enables bit 128 automatically.

## Cosmology

These keys set the cosmology distances are computed in, and the peculiar
velocities that separate the heliocentric redshift from the cosmological
one.

```{eval-rst}
.. configtable:: jax_snlc_sim.simulation.Cosmology
   :field-keys:
   :keys-from: jax_snlc_sim.inputs
```

## Milky Way extinction

`OPT_MWEBV` selects the source of E(B−V), and `OPT_MWCOLORLAW` the
extinction law. Their values are listed below.

`APPLYFLAG_MWEBV` corrects the reported photometry for Milky Way extinction,
so the output and the cuts use corrected magnitudes.

### Settings

```{eval-rst}
.. configtable:: jax_snlc_sim.simulation.MWDust
   :field-keys:
   :keys-from: jax_snlc_sim.inputs
```

(sim-mwebv-options)=
### Milky Way reddening: `OPT_MWEBV`

Choose the source of E(B−V).

| Value | Behaviour |
| --- | --- |
| 0 | No Milky Way extinction. |
| 1 | Use the SIMLIB value, falling back to the SFD98 map. |
| 2 | Use the SFD98 map. |
| 3 | Use the Schlafly11 scaling, 0.86 times SFD98. |

Default: **1**, which requires `GENSIGMA_MWEBV_RATIO`.

(sim-mwlaw-options)=
### Milky Way extinction law: `OPT_MWCOLORLAW`

Choose one extinction law.

| SNANA value | Python / YAML `dust_law` | Law |
| --- | --- | --- |
| 0 | `OFF` | No extinction. |
| 89 | `CCM89` | CCM89. |
| 94 | `ODONNELL94` | O’Donnell94. |
| 99 | `FITZ99_EXACT` | Fitzpatrick99, exact. |
| -99 | `FITZ99_APPROX` | Fitzpatrick99, approximate. |
| 203 | `GORD03` | Gordon03. |
| 204 | `FITZ04` | Fitzpatrick04. |
| 208 | `GOOBAR08` | Goobar08. |
| 214 | `MAIZ14` | Maíz14. |
| 216 | `GORD16` | Gordon16. |
| 219 | `FITZ19` | Fitzpatrick19, cubic interpolation. |
| -219 | `FITZ19_LIN` | Fitzpatrick19, linear interpolation. |
| 223 | `GORD23` | Gordon23. |
| 225 | `SOMM25` | Somm25. |
| 226 | `SOMM26` | Somm26. |

Default: **99**. With value 0 and `OPT_MWEBV: 1`, the SIMLIB E(B−V) is
still reported, although no extinction is applied.

## Weak lensing

Weak lensing is applied as an additive magnitude offset.
When enabled, `WEAKLENS_DSIGMADZ` gives a Gaussian offset with standard
deviation `z * WEAKLENS_DSIGMADZ` magnitudes. It takes precedence over
`WEAKLENS_PROBMAP_FILE`, which supplies a distribution of offsets at each
redshift.

```{eval-rst}
.. configtable:: jax_snlc_sim.simulation.Lensing
   :field-keys:
   :keys-from: jax_snlc_sim.inputs
```

## Spectra

Spectra are simulated alongside the photometry when a spectrograph is
defined. Each `TAKE_SPECTRUM` line specifies an epoch and either an
exposure time or a target signal-to-noise ratio.
Host-contamination settings add a host-galaxy spectrum; warping settings
apply a wavelength-dependent flux modification. See
{ref}`simulating-spectra` for a runnable example and how to read its output.

`VERIFY_SED_TRUE` compares each written spectrum's re-integrated magnitude
with its broadband magnitude.

### Settings

```{eval-rst}
.. configtable:: jax_snlc_sim.simulation.Spectra
   :field-keys:
   :keys-from: jax_snlc_sim.inputs
```

(sim-spectrograph-options)=
### Spectrograph options: `SPECTROGRAPH_OPTMASK`

Combine bit values to change the spectrum calculation or output.

| Value | Behaviour |
| --- | --- |
| 1 | Disable wavelength smearing. |
| 2 | Double wavelength smearing. |
| 8 | Multiply signal-to-noise by 100. |
| 16 | Omit template noise. |
| 32 | Include template noise only. |
| 64 | Allow extrapolation beyond the exposure-time table. |
| 128 | Write the true SED rather than the observed spectrum. |

Default: **0**, using the instrument calibration without these changes.

For explicitly rejected SNANA keys, see {ref}`unsupported-simulation-keys`.
