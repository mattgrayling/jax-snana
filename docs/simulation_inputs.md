# Simulation input files

Survey simulations use a SIMLIB for the observing times and conditions, a
calibration file for the filter transmissions and zero points, and efficiency
files for detection and follow-up. A HOSTLIB supplies host galaxies; a GENPDF
file can supply tabulated population distributions.

```{contents} On this page
:local:
:depth: 1
```

## Keyed-text format

The text files use SNANA's format: each line starts with a key ending in a
colon, and anything after `#` is a comment. A table starts with a `VARNAMES:`
line naming the columns, and each row starts with a key such as `SN:`.

Some SNANA input files must open with a `DOCUMENTATION:` block describing where
they came from, for example:

```text
DOCUMENTATION:
  PURPOSE: Observing history for the survey simulation.
DOCUMENTATION_END:
```

Whether the block is required depends on the file:

- The SIMLIB needs it under the simulation's `REQUIRE_DOCANA`. The default of 1
  stops a run whose SIMLIB lacks the block, and 0 turns that into a warning.
- The search-efficiency files need it whatever `REQUIRE_DOCANA` says.
- A HOSTLIB, weight map or population file can be written with or without it.

The following file excerpts omit this block.

(format-simlib)=
## SIMLIB

The SIMLIB describes your survey's observing history, and the simulation uses
it to reproduce the survey's cadence and observing conditions. It has a global
header followed by a list of LIBIDs, each a sequence of observations at one
sky position:

```
SURVEY: DES   FILTERS: griz   TELESCOPE: CTIO
PIXSIZE: 0.270

BEGIN LIBGEN

LIBID: 1
RA: 34.4757  DEC: -4.9295  NOBS: 76  MWEBV: 0.0249  FIELD: C1
#     MJD      IDEXPT  FLT GAIN NOISE SKYSIG  PSF1 PSF2 RATIO  ZPTAVG ZPTSIG  MAG
S: 56176.180  212418   g  4.05  5.20  10.24   1.85 0.00 0.000  31.42  0.011  99
S: 56176.203  212419   r  4.05  5.20  16.71   1.72 0.00 0.000  31.58  0.009  99
```

`SURVEY:` and `FILTERS:` are required, and the filter characters must match
those in the calibration file. Several keys can share one line, as in the
example.

### Global header keys

`PIXSIZE`
: Pixel size in arcsec. It is required, since sky noise and PSF widths are
  converted with it. If you want one file to mix instruments, you can give it
  per LIBID instead.

`PSF_UNIT`
: How the PSF columns are expressed. The default, `PIXEL_SIGMA`, reads three
  columns as a double Gaussian: two widths in pixels and their amplitude ratio.
  `ARCSEC_FWHM` reads the same three columns as FWHM in arcsec. `NEA_PIXEL` and
  `NEA_ARCSECSQ` replace all three with one noise-equivalent-area column.

`SKYSIG_UNIT`
: `ADU_PER_SQPIXEL` by default, or `ADU_PER_SQARCSEC`.

`NPE_PIXEL_SATURATE` and `PHOTFLAG_SATURATE`
: The photo-electron count that saturates the central pixel, and the bit set in
  `PHOTFLAG` when that happens.

`TEMPLATE_ZPT`, `TEMPLATE_SKYSIG`, `TEMPLATE_CCDSIG`
: Noise from the template image used for subtraction, shared by every epoch in
  a filter. Each key takes one value per filter in the global `FILTERS:` list,
  including filters a given LIBID doesn't use. They can appear globally or
  inside a LIBID.

`FIELD`
: The field name of every LIBID that does not give its own.

`FLUXERR_ADD` and `FLUXERR_COR`
: Additive per-band flux errors and signal-to-noise-dependent error scales,
  respectively.

### LIBID keys

`LIBID:` opens a block and the next `LIBID:` closes it. One line inside gives
the position and length of the sequence:

`RA:` and `DEC:`
: Sky position in degrees. `DECL:` is accepted too.

`NOBS:`
: How many observations follow.

`MWEBV:`
: Milky Way E(B-V) at this position. Where it's absent or zero, the simulation
  takes the value from a dust map instead.

`FIELD:`
: Field name, written to the output and usable in selections. It can also
  appear on its own line between observations, so that one LIBID can cover
  overlapping fields.

`GROUPID_HOSTLIB:`
: Restricts host selection for this LIBID to the listed HOSTLIB groups.

A LIBID can also fix properties of the supernovae generated on it, with
`REDSHIFT:`, `PEAKMJD:`, `SALT2x1:`, `SALT2c:` and `DISTANCE:`. These are used
only when the matching `USE_SIMLIB_*` option is set in the simulation input.
The `GENRANGE_` forms of the same keys are also accepted, and give a range to
draw from. Under
`HOSTLIB_MSKOPT` bit 16 a fixed redshift picks the host within
`HOSTLIB_DZTOL` of it, and the supernova then takes the host's redshift.

### Observation rows

An `S:` row has twelve values in a fixed order: MJD, exposure ID, filter
character, gain in photo-electrons per ADU, read noise per pixel, sky noise,
the three PSF values, zero point, its scatter, and an unused magnitude
column, which should be set to `99`. With a noise-equivalent-area `PSF_UNIT`,
the three PSF columns become one and the row has ten values.

MJDs needn't be in order. The exposure ID can include a repeat count and a
detector number, as `ID*NEXPOSE`, `ID(DETNUM)` or `ID*NEXPOSE(DETNUM)`, with no
spaces inside the token.

The zero point converts magnitude to ADU as `flux = 10^(-0.4(m - ZPTAVG))`. It
includes transparency, aperture, throughput and exposure time.

With `USE_SIMLIB_SPECTROGRAPH` enabled, `SPECTROGRAPH: <MJD> <texpose>`
requests a spectrum at that observer-frame MJD with the exposure time in
seconds. These rows schedule spectra separately for each LIBID. A SIMLIB can
also contain `TAKE_SPECTRUM` entries, using the syntax of the simulation-input
key of the same name, enabled with `USE_SIMLIB_TAKE_SPECTRUM`.
{doc}`simulating_spectra` describes spectrum requests and calibration.

`T:` rows are not part of the SIMLIB format and are not read. Template noise
comes from the `TEMPLATE_*` header keys.

(format-kcor)=
## Calibration file

The calibration file specifies the filter transmissions, the reference
standards and their magnitudes in each band, and the photometric zero-point
offsets. `KCOR_FILE` accepts a SNANA calibration input such as `kcor.input`
or an existing kcor FITS file. Both are read directly by jax-snana; no
SNANA executable is required.

### Using your own transmission curves

Put two photon-counting transmission curves in `filters/g.dat` and
`filters/r.dat`, with wavelength in Angstroms and dimensionless throughput
in two columns. Save the following as `mykcor.input`:

```text
MAGSYSTEM: AB
FILTSYSTEM: COUNT
FILTPATH: ./filters
SURVEY: DES
FILTER: Custom-g g.dat 0.0
FILTER: Custom-r r.dat 0.0
LAMBDA_RANGE: 2000 12000
```

Set `KCOR_FILE = '/path/to/mykcor.input'` in your fitter input file and use
`FILTLIST_FIT = 'gr'` for these filters. A simulation uses the same file
through `KCOR_FILE: /path/to/mykcor.input`. In a Python light-curve fit,
pass it as `calibration="/path/to/mykcor.input"`.

`MAGSYSTEM`, `FILTSYSTEM`, `FILTPATH` and `SURVEY` apply to the filters
that follow them, so a single file can describe several surveys and
magnitude systems. `COUNT` describes a photon-counting response; `ENERGY`
describes an energy response, which is converted to the photon-counting
convention used for SED integration. The last number on each `FILTER` line
is the reference standard's magnitude in that band. It can also be written
as a sum, such as `9.824+.011-.002`, to record separate calibration
corrections. The example uses AB magnitudes with zero offsets.

Relative paths are resolved from the directory containing the calibration
input. Filter paths are also searched under `$SNDATA_ROOT/filters` and
reference spectra under `$SNDATA_ROOT/standards`. AB uses SNANA's
`flatnu.dat` reference spectrum unless `AB_SED` names another file.
`VEGA_SED` and `BD17_SED` name the corresponding reference spectra;
`PRIMARY_SED: name filename` defines another standard. Each spectrum has
wavelength in Angstroms and flux density in erg s⁻¹ cm⁻² Å⁻¹ in two columns.
Magnitude-system conversions such as `MAGSYSTEM: BD17->AB` are also supported.

If a filter directory contains `ZPOFF.DAT`, its band and zero-point offset
pairs are included in the calibration. `ZPOFF_FILE` before `FILTPATH` selects
another offset file, and `READ_ZPOFF: 0` disables these offsets. Set `SURVEY`
to the name registered for your dataset in `$SNDATA_ROOT/SURVEY.DEF`.
`LAMBDA_RANGE` covers the supplied transmission curves, and the last
character of each filter name identifies its band in SNANA photometry.

The transmission curves and reference spectra are sampled on a 10 Å grid,
as in SNANA when no `SN_SED` is supplied. If `SN_SED` names a spectral time
series, its wavelength grid is used instead; it does not select the model
you will simulate or fit. Settings for generating K-correction tables or
writing output, such as `KCOR` and `OUTFILE`, do not affect a direct read.

### Spectrograph and additional calibration options

A calibration input can also include a spectrograph response and synthetic
bands, which integrate its response over specified wavelength ranges:

```text
SPECTROGRAPH: spectrograph.dat
SYN_FILTER: IFU-u 4000 4400 0.0
SYN_FILTER: IFU-v 5900 6300 0.0
```

The spectrograph file supplies wavelength bins, exposure times and S/N at
two reference magnitudes. With `USE_SIMLIB_SPECTROGRAPH` enabled, SIMLIB
`SPECTROGRAPH:` epochs generate both spectra and photometry through the
synthetic bands, using this calibration.
`SPECTROGRAPH: spectrograph.dat(rebin=2)` combines pairs of wavelength bins
before calculating the calibration.
`SYN_FILTER` takes a filter name, lower and upper wavelength bounds in
Angstroms, and the reference standard's magnitude in that band; the current
`MAGSYSTEM` and `FILTSYSTEM` apply as for `FILTER`.

`FILTER_OOB: gr 3000 10000 0.001` adds a constant transmission to the listed
bands, at 0.1% of each original curve's peak, extending their wavelength
coverage to the specified range. The wavelength grid is extended using the
sample spacing at each end of the original curve. The extra
transmission is added before `LEAKAGE_CUT` and grid interpolation.

`DUPLICATE_LAMSHIFT_GLOBAL: 10` includes a second set of filters shifted by
10 Å, with names such as `*DES-g`. Ordinary band selection uses the nominal
filters; select a shifted curve by its full name, including the leading `*`.

### Existing FITS calibrations

SNANA's `kcor.exe` produces kcor FITS files from these calibration inputs.
They contain the filter curves, reference spectra and zero-point information,
and may also contain K-correction tables and spectrograph calibration.
jax-SNANA reads these files through `KCOR_FILE`. For the full
calibration input format, see the **K-Corrections** section of the
[SNANA manual](https://github.com/RickKessler/SNANA/blob/master/doc/snana_manual.pdf).

### Filter transmission and naming conventions

Out-of-band leakage in a filter curve widens the wavelength range each band
has to be integrated over. `LEAKAGE_CUT` in the calibration input removes
wavelength bins below that fraction of the filter's peak before sampling
the calibration grid. The simulator and fitter also accept this key,
applying it to the stored calibration curves as they are read.
The default is 0, which keeps every filter as stored. A value of
`1e-3`, for example, removes transmission below 0.1% of each filter's peak;
check its effect on the integrated photometry for your filters and SEDs.
A simulation and the fit of it should use the same value.

Filters can be identified by their full names or by unambiguous SNANA
single-character labels. For example, `SDSS-g` has the short label `g`;
use the full name if more than one filter shares that label.
Transmission curves may have non-uniform wavelength sampling. The reader
interpolates them onto the calibration grid.

(format-spectrograph)=
## Spectrograph response

A spectrograph response specifies wavelength bins and resolution, together
with the expected signal-to-noise at two reference magnitudes for each
exposure time. It is supplied through `SPECTROGRAPH_FILE` or the calibration
file's `SPECTROGRAPH` entry. The {download}`example response
<../tests/fixtures/data/spectrograph/SPECTROGRAPH_TEST.DAT>` begins:

```text
INSTRUMENT: TEST_SPECTRO
MAGREF_LIST: 20.0 24.0
TEXPOSE_LIST: 600 1800 5400
SPECBIN: 3500 3600 2.5  8.0000 0.5000  13.8564 0.8660  24.0000 1.5000
```

`MAGREF_LIST` gives the two reference magnitudes, and `TEXPOSE_LIST` gives
the exposure times in seconds. Each `SPECBIN` row starts with the lower
and upper wavelength bounds and the Gaussian resolution σ, all in
angstroms. The remaining values give the signal-to-noise at the two
reference magnitudes for each exposure time, in `TEXPOSE_LIST` order.
{doc}`simulating_spectra` shows how to request observations using this response.

(format-hostlib)=
## HOSTLIB

A HOSTLIB is a catalogue of galaxies. It declares its columns once and then
lists one galaxy per row:

```
VARNAMES: GALID RA_GAL DEC_GAL ZTRUE ZERR ZPHOT ZPHOTERR g_obs r_obs i_obs z_obs a0_Sersic b0_Sersic n0_Sersic a_rot
GAL: 2940448741 52.674320 -27.177160 0.3964 0.0001 0.4012 0.0413 22.0658 20.9999 20.6560 20.3309 0.9929 0.7381 1.0 154.406
```

Only two columns are required: `GALID`, which must come first, and `ZTRUE`, the
true heliocentric redshift, which can go anywhere. Optional columns supply
the host properties used by the configured simulation. The file needn't be
sorted by redshift.

Coordinates can be named `RA`/`DEC`, `RA_GAL`/`DEC_GAL` or
`RA_HOST`/`DEC_HOST`. The `_GAL` and `_HOST` forms are safer, because they
can't collide with the supernova's own coordinates in output tables.

The light profile is made of up to nine Sersic components, indexed from zero:

- `a0_Sersic` and `b0_Sersic`: the half-light radii along the major and minor
  axes, in arcsec.
- `n0_Sersic`: the Sersic index.
- `w0_Sersic`: the component's weight, where there is more than one.
- `a_rot`: the major-axis orientation in degrees.

For multiple components, their weights must sum to one. One weight may be
omitted and is inferred from the others.

The profile sets where in the galaxy each supernova is placed, and so how much
host light falls in its aperture.

Host properties are named `LOGMASS`, `LOGSFR`, `LOGsSFR` (with a lowercase `s`)
and `COLOR`, each of which can also be given with a `_TRUE`, `_OBS` or `_ERR`
suffix. A name without a suffix is taken as the true value. Supplying `_ERR`
makes the observed value a Gaussian draw about the true one. Supplying `_OBS`
directly lets you use non-Gaussian errors. `ELLIPTICITY` and `SQRADIUS` columns
are copied to the output keys of the same names.

Use the property names above exactly; in particular, `LOGSFR` supplies
`HOSTGAL_LOGSFR` in the output.

### Photo-z quantiles

A HOSTLIB can tabulate each galaxy's photo-z distribution as quantiles, which
the simulation writes to the `HOSTGALz_` keys described in
{doc}`light_curve_data`. There are two
ways to name the quantile columns:

- `ZPHOT_Q<nnn>`, where the name gives the percentile: `ZPHOT_Q000`,
  `ZPHOT_Q010`, ..., `ZPHOT_Q100`.
- `QZPHOT<nn>`, where the name gives only the index: `QZPHOT00` to `QZPHOT10`
  for eleven quantiles. The percentiles are then spaced uniformly from 0 to 100
  over however many columns there are.

Use one naming convention per file.

The quantiles are scaled by the ratio of the supernova's redshift to the
galaxy's `ZTRUE`, unless `HOSTLIB_MSKOPT` bit 16 puts the supernova at the
galaxy's redshift.

`HOSTLIB_MSKOPT` bit 65536 (`ZPHOT_QGAUSS`) makes the simulation compute the
quantiles as a Gaussian from each galaxy's `ZPHOT` and `ZPHOT_ERR` columns. It
uses the percentiles the file's quantile columns declare, or the eleven
percentiles 0, 10, ..., 100 when there are none.

### Weight map

Without a weight map, every galaxy in the redshift-matched subset is equally
likely. If you want to simulate a correlation between supernovae and their
hosts, you can use a weight map to make the choice depend on host properties:

```
VARNAMES_WGTMAP: LOGMASS LOGSFR WGT SNMAGSHIFT
WGT: 4.50 -12.5 9.119e-04 0.0
WGT: 4.50 -12.3 1.114e-03 0.0
```

The header key is `VARNAMES_WGTMAP:` instead of `VARNAMES:`, and the rows are
keyed `WGT:`. The last two columns must be the weight and the coherent
magnitude shift, in that order, though you can leave the shift out. Every
earlier column names a HOSTLIB column, except for the parameters the run's
model draws per supernova. Those needn't exist in the HOSTLIB, and either
spelling matches:

- SALT: `SALT2x1` or `x1`, and `SALT2c` or `c`.
- BayeSN: `THETA` or `theta`.
- Host dust, for any model: `AV` and `RV`.

Use the parameter names of the model being simulated. For example, a map
conditioned on light-curve shape uses `x1` for SALT or `THETA` for BayeSN.

The grid must be uniformly spaced in each dimension. A HOSTLIB value outside it
is an error unless `HOSTLIB_WGTMAP_EXTRAP` is set. Every variable used in the
map is written to the output files automatically.

## GENPDF population file

`GENPDF_FILE` names a file of tabulated probability densities, one per
population parameter it covers. The simulation draws those parameters from it
instead of from the GENGAUSS keys. Use it to give a parameter a distribution
that isn't an asymmetric Gaussian, or to make a distribution depend on a
property of the host galaxy:

```
VARNAMES: SALT2c PROB
PDF: -0.30  0.02
PDF: -0.20  0.35
PDF: -0.10  1.00
PDF:  0.00  0.72
PDF:  0.20  0.18
PDF:  0.40  0.01

VARNAMES: SALT2x1 LOGMASS PROB
PDF: -3.0   9.0  0.05
PDF:  0.0   9.0  1.00
PDF:  3.0   9.0  0.30
PDF: -3.0  11.0  0.30
PDF:  0.0  11.0  1.00
PDF:  3.0  11.0  0.05
```

A `VARNAMES:` line opens a map and names its columns in the order they appear
in the `PDF:` rows: the parameter being drawn, then any conditioning variables,
then `PROB`. Use the parameter name listed for the model, which is the GENGAUSS
key without its prefix. The named parameter must be drawn by a model in the
run. A name spelled `LOG` plus a parameter's name, `LOGAV`
for instance, tabulates its base-10 logarithm; draws are converted back to
the parameter's linear value. One file can contain maps for several parameters.

The columns between the parameter and `PROB` condition the density. Each is
a HOSTLIB column, spelled as the catalogue spells it. A conditional map
requires a host catalogue. The parameter is drawn conditional on the
selected host. Cover the range of conditioning values in your catalogue, or
set bit 1 of `GENPDF_OPTMASK` to use the nearest edge value outside the
tabulated range.

The rows must form a complete grid:

- The parameter's tabulated values needn't be evenly spaced, but every
  combination of conditioning values must tabulate the same parameter values.
- Each cell must appear exactly once.
- Between two tabulated points the density is the straight line joining them.
  No draw falls outside the first and last parameter values, so those truncate
  the population exactly as a `GENRANGE_` key would.
- `PROB` needn't be normalised, since it's integrated over the grid and
  normalised when the file is read. It must be non-negative everywhere, and
  every conditioning slice must have non-zero probability.

The file can be gzipped.

A GENPDF file can also contain keys of its own, which are read as though they
were in the simulation input:

- the GENGAUSS family (`GENPEAK_`, `GENSIGMA_`, `GENRANGE_` and the rest), so a
  published population file can supply the alpha and beta a survey fitted
  alongside the densities it tabulates
- `MAG_OFFSET`
- `GENMAG_SMEAR`, the coherent magnitude scatter. Supply this setting in only
  one of the two files.

(format-efficiency)=
## Search-efficiency files

For a complete input connecting these files, see
{ref}`simulating-selection-example`.

The trigger described under {doc}`simulating` is configured by up to four
files, each named by a key of the simulation input. All four must open with a
`DOCUMENTATION:` block whatever `REQUIRE_DOCANA` says. The first two describe
the detection pipeline and the other two the follow-up. SNANA's own files of
each kind are under `$SNDATA_ROOT/models/searcheff`. A file name not found as
given is looked for in `PATH_USER_INPUT`, then in that directory. Each file
may be gzipped.

### Detection efficiency

`SEARCHEFF_PIPELINE_EFF_FILE` gives the probability that a single observation
is detected, as a curve per filter:

```
PHOTFLAG_DETECT:   4096  # set this PHOTFLAG bit for each detection
PHOTFLAG_TRIGGER:  2048  # flags epoch when trigger is formed

FILTER: g
ABS(SNR):   3.00  0
ABS(SNR):   3.50  0.0597
ABS(SNR):   4.00  0.1123
...
ABS(SNR):  15.00  1.0000

FILTER: r
...
```

A `FILTER:` line, or `BAND:`, opens a curve for the filters it names. The rows
under it tabulate the efficiency against signal-to-noise (`SNR:`), its absolute
value (`ABS(SNR):`) or the observed magnitude (`MAG:`). Between rows the
efficiency is interpolated linearly, and beyond the first and last it stays at
their values. A `FIELD:` line inside a curve restricts it to the SIMLIB fields
it names, and no two curves can claim the same filter and field.

The two `PHOTFLAG_` keys name the bits set in the photometry's `PHOTFLAG`
column for a detected epoch and for the epoch that formed the trigger. They are
used only if the simulation input leaves its keys of the same names at zero.

A `MAPNAME_PHOTPROB:` block, closed by `ENDMAP:`, instead tabulates the
distribution of a per-epoch classification score that some surveys' pipelines
attach to each detection. Its `CUTVAL:` line is the score below which a
detection is discounted. The scores are drawn only when the file also holds a
curve; without one, every `PHOTPROB` is -9.

A file named `NONE` or `DEFAULT`, or no file at all, counts every event as
found by the pipeline, with no observation flagged as detected.

### Detection logic

`SEARCHEFF_PIPELINE_LOGIC_FILE` sets how many detections make a discovery, with
one line per survey keyed by the survey's name:

```
DES:  2 g+r+i+z    # 2 detections in any filter
LSST: 1 u+g+r+i+z+Y
```

Include a line whose key matches the SIMLIB's `SURVEY:`; that line supplies
the detection logic for the run.

The number is how many separate detection periods are required. An observation
opens a new period when it comes more than `NEWMJD_DIF` days after the first
observation of the current one. The expression after the number is the filters
a period must be detected in. Terms separated by `+` are alternatives, and a
term of several letters needs all of them. For example, `gr+ri` requires a
period detected in both g and r, or in both r and i.

The value `DEFAULT` reads SNANA's `SEARCHEFF_PIPELINE_LOGIC.DAT`. The key is
required whenever the detection-efficiency file holds a curve.

### Spectroscopic and host-redshift efficiency

`SEARCHEFF_SPEC_FILE` and `SEARCHEFF_zHOST_FILE` share one format. They give
the probability of a spectroscopic confirmation of the event, and of a redshift
from its host. Each is decided once per event, unlike the detection
efficiency, which applies to each observation. The host redshift is tried only
if the confirmation failed.

Each file contains one or more maps. A map is a `VARNAMES:` line naming the
variables it's tabulated against, then the efficiency column, named `SPECEFF`
or `HOSTEFF`, which must come last. A row per grid point follows. The rows are
keyed `SPECEFF:` in the first file and `HOSTEFF:` in the second, and a blank
line ends the map:

```
VARNAMES: g-r  r  SPECEFF
SPECEFF:  -2.00  15.00  1
SPECEFF:  -2.00  15.05  1
...
```

A map can be tabulated against:

- `REDSHIFT`
- `PEAKMJD`, in observer-frame MJD
- `DTSEASON_PEAK`, the signed separation in days between peak and the
  nearest season edge: positive within a season and negative outside
- `DTPEAK`, the distance of the nearest observation from the peak
- a peak magnitude, named by its filter letter, e.g. `r`
- a colour, named as two letters joined by a minus sign, e.g. `g-r`
- a mean of several peak magnitudes, named by letters joined by `+`, e.g. `r+i`
- `SNRSUM_REST_V`, the quadrature sum of the expected signal-to-noise in
  spectral bins centred within rest-frame 5000–6000 Å, across all requested
  spectra. This requires spectra and a spectrograph calibration, as described
  in {doc}`simulating_spectra`.
- the SALT parameters `SALT2mB`, `SALT2x1` and `SALT2c`
- the host's stellar mass as `LOGMASS`, taken from the galaxy matched as the
  observed host
- the host's catalogue magnitude as `HOSTMAG_<filter>` and its surface
  brightness under the event as `SBMAG_<filter>`
- any other HOSTLIB column, by its own name, taken from the galaxy the event
  was generated in

The rows must fill a complete grid of the variables' values, with uniform steps
along every variable, one-variable maps included. A file that holds no map
gives every event an efficiency of one.

Three keys can come before a map and apply to it alone:

- `FIELD:` restricts it to the SIMLIB fields whose names appear in its value,
  e.g. `FIELD: C3+X3`.
- `PEAKMJD_RANGE:` restricts it to events peaking between two dates. Without
  it, a map covers peaks between MJD 10000 and 90000.
- `REQUIRE: 1` makes the map a separate condition, with its own random draw
  that the event must also pass.

Where several maps apply to one event, their efficiencies combine as one
minus the product of one minus each. An event that reaches a file's maps but
matches none of them stops the run, naming its field and peak date. With
`OPT_FIELDMATCH_REQUIRE: 0` anywhere in the file, such an event has an
efficiency of zero instead.

`OPT_EXTRAP: 1` extends the maps that follow it to their edge values. By
default the efficiency is zero outside the grid. A map with a single row has
that efficiency everywhere.

In place of a file name, you can give `NONE` or `ONE` to give every event an
efficiency of one, or `ZERO` to give none of them a spectrum. You can use
`ZERO` to simulate a photometric sample whose redshifts all come from hosts.

`APPLY_SEARCHEFF_OPT` selects the required detection and redshift outcomes.
The corresponding efficiency inputs must allow that outcome:

| Requirement | Efficiency input |
| --- | --- |
| Supernova spectrum (bit 2) | `SEARCHEFF_SPEC_FILE` containing a map, or set to `NONE` or `ONE`. |
| Host redshift (bit 4) | `SEARCHEFF_zHOST_FILE` containing a map, or set to `NONE` or `ONE`. `SEARCHEFF_SPEC_FILE` may contain a map or be `ZERO`, `NONE` or `ONE`. |
| Either redshift (bit 8) | At least one of the two redshift inputs must allow a redshift. |

Which of the three stages an event passed is written to its
`SIM_SEARCHEFF_MASK`, described in {doc}`light_curve_data`.
