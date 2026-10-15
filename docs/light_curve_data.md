# Light-curve data

The simulator writes light curves as SNANA FITS or text files. The fitter
also reads Rubin HATS catalogues. FITS stores event metadata and photometry
in separate tables, SNANA text stores one event per file, and HATS uses a
partitioned Parquet catalogue.

```{contents} On this page
:local:
:depth: 1
```

The simulation output format is selected with `FORMAT_MASK`: FITS (32)
or TEXT (2). Both contain the same event metadata and photometry.

(data-version-location)=
## Locating a data version

`VERSION_PHOTOMETRY` names the data version without a path. The fitter
searches for the directory containing a file called `<VERSION>.LIST`, in this
order:

1. Where `PRIVATE_DATA_PATH` is set, that directory, then that directory with
   the version name appended.
2. Otherwise, the lcmerge tree of `$SNDATA_ROOT`: first under the version
   name, then, for a name containing an underscore, under the part before the
   first underscore.
3. Either way, each path listed in `$SNDATA_ROOT/SIM/PATH_SNDATA_SIM.LIST`,
   then `$SNDATA_ROOT/SIM` itself.

The command-line fitter reads SNANA FITS data versions and Rubin HATS
catalogues. For Python fits, photometry and metadata can be supplied as
arrays or tables to {ref}`LightCurveFitter <fit-from-python>`.

(preparing-text-data)=
## Preparing a TEXT dataset

If your observations are not already a SNANA data version, write one TEXT
file per event following the {ref}`TEXT light-curve example
<format-text-light-curves>`. Include the event metadata, redshifts,
calibrated fluxes and uncertainties, using filter names consistent with
your calibration file. Put these files in a directory named `MY_DATA_TEXT`
and list their filenames, one per line, in `MY_DATA_TEXT.LIST` inside that
directory. Also create `MY_DATA_TEXT.README` with a documentation block
describing the sample, for example:

```text
DOCUMENTATION:
PURPOSE: Light curves for MY_DATA
DOCUMENTATION_END:
```

For custom transmission curves, {ref}`format-kcor` gives a calibration-file
example.

Convert this dataset with SNANA's `snana.exe`, using its full data tree.
Use a separate output directory that does not contain the TEXT version, so
SNANA cannot discover the same dataset through both the working directory
and `PRIVATE_DATA_PATH`. From that output directory, run:

```bash
snana.exe NOFILE PRIVATE_DATA_PATH /path/to/MY_DATA_TEXT \
  VERSION_PHOTOMETRY MY_DATA_TEXT VERSION_REFORMAT_FITS MY_DATA_FITS \
  OPT_SETPKMJD 20 OPT_MWEBV 0
```

This creates `MY_DATA_FITS/`, including its `.LIST`, `HEAD` and `PHOT` files.
`OPT_SETPKMJD 20` estimates the peak time from the observations; `OPT_MWEBV 0`
keeps the Milky Way reddening supplied in your event files. If you need SNANA
to obtain that reddening from its dust maps instead, use `OPT_MWEBV 2`.
Pass the resulting directory to `Data`, or set
`VERSION_PHOTOMETRY = 'MY_DATA_FITS'` and `PRIVATE_DATA_PATH` in your fitter
input file. The fitting stage also needs the appropriate `KCOR_FILE` and
Milky Way extinction settings.

The **Translating TEXT data-files into FITS Format** section of the [SNANA
manual](https://github.com/RickKessler/SNANA/blob/master/doc/snana_manual.pdf)
describes the conversion options. To merge existing versions, SNANA also
accepts `VERSION_PHOTOMETRY_WILDCARD`.

## FITS

FITS stores event metadata and photometry in a pair of files.
`*_HEAD.FITS` has one row per supernova with its metadata. `*_PHOT.FITS` has every light
curve written end to end. Each header row points into it with `PTROBS_MIN` and
`PTROBS_MAX`. Both pointers are one-based and inclusive, so one supernova's
rows are `phot[PTROBS_MIN - 1:PTROBS_MAX]`, and its `NOBS` is
`PTROBS_MAX - PTROBS_MIN + 1`.

The `.LIST` file, described below, names the version's HEAD files. Each HEAD
file names its own PHOT file in the `PHOTFILE` key of its primary header.
Spectra go in a separate `*_SPEC.FITS`.

The PHOT table's columns are `MJD`, `BAND`, `FIELD`, `PHOTFLAG`, `PHOTPROB`,
`FLUXCAL`, `FLUXCALERR`, the three PSF values `PSF_SIG1`, `PSF_SIG2` and
`PSF_RATIO`, `SKY_SIG`, `SKY_SIG_T`, `RDNOISE`, `ZEROPT`, `ZEROPT_ERR`,
`TEXPOSE`, `GAIN` and `SIM_MAGOBS`. `BAND` is as wide as the longest filter
name in the calibration file, five characters for `DES-g`, and `FIELD` is 32
characters.

For a noise-equivalent-area PSF, `PSF_NEA` replaces `PSF_SIG1`, `PSF_SIG2`
and `PSF_RATIO`.

### FITS storage conventions

The primary header also has `SNANA_VERSION`, the configured `ZP_FLUXCAL`
(default 31.4) and
`CODE_IVERSION`, SNANA's revision number for the format, which is 29.
`DETINFO_FLAG` is 0, since the PHOT table has no detector columns (`IMGNUM`,
`DETNUM`, `XPIX`, `YPIX`).

Unavailable host values are stored as -9. Quantities written as
`value +- error` on one TEXT line occupy separate FITS columns, such as
`HOSTGAL_PHOTOZ` and `HOSTGAL_PHOTOZ_ERR`. The quantile and mass grids are
vector columns of fixed width:
`HOSTGALz_QUANTILE_ZPHOT` and `HOSTGALz_QUANTILE_PERCENT` have 20 values each,
and the three `HOSTGALz_LOGMASS_*GRID` columns have 40. The `NBIN` column gives
how many are in use, and the rest are -9. `SNID` is a 40-character string.

As in SNANA, a marker row closes each light curve in the PHOT file, outside
the pointer range: its `MJD`, `FLUXCAL` and `FLUXCALERR` are -777, its `BAND`
is `-`, its `FIELD` is `XXXX`, and every other column is 0. The next light
curve therefore starts two rows after the previous `PTROBS_MAX`. Readers that
follow the pointers never see these rows. Classifiers such as SuperNNova
instead split the PHOT table at them. Each spectrum's rows in `*_SPEC.FITS`
are closed the same way, with -777 in every flux column.

## Rubin HATS catalogues

The fitter also reads Rubin's DiaObject catalogues in HATS form, the
partitioned parquet that LSDB writes: a directory holding `hats.properties`
and the parquet under `dataset/`. `VERSION_PHOTOMETRY` names it in the same
way as an SNANA version, and the fitter recognises it by its
`hats.properties`.

Each row is one DiaObject. Its light curve is `diaObjectForcedSource`, the
forced photometry on the difference image, stored as a struct of lists that
share one set of offsets. The fitter reads these Rubin columns:

| SNANA | Rubin |
|---|---|
| `SNID` | `diaObjectId` |
| `RA`, `DEC` | `ra`, `dec` |
| `MJD` | `diaObjectForcedSource.midpointMjdTai` |
| `BAND` | `diaObjectForcedSource.band` |
| `FLUXCAL` | `diaObjectForcedSource.psfDiffFlux` |
| `FLUXCALERR` | `diaObjectForcedSource.psfDiffFluxErr` |

`PHOTOMETRY_COLUMNS` reads any of these from another column, given as `NAME
column` pairs of the SNANA name and the Rubin column. For example,
`PHOTOMETRY_COLUMNS = 'FLUXCALERR psfDiffFluxErr_corrected'` fits with
Rubin's rescaled errors where a catalogue has them.

Rubin fluxes are in nJy, AB magnitude 31.4 at unit flux, so they are read as
`FLUXCAL` unchanged, with `ZP_FLUXCAL` 31.4, the zero point of a current SNANA
version. `PHOTFLAG` adds 1 for `psfDiffFlux_flag` and 2 for `invalidPsfFlag`,
so `PHOTFLAG_MSKREJ = 3` drops every flagged epoch. The bands are the bare
letters `u` to `y`. Only the last character of a band name is used, so `g`
matches the kcor file's `LSST-g`. Where the kcor file calls the last band `Y`,
`FILTER_REPLACE = 'y -> Y'` matches it.

A catalogue has no redshift, peak date or Galactic extinction. You can fit it
with a photometric redshift and the flat prior, with `OPT_SETPKMJD` to find
the peak and `OPT_MWEBV = 2` to take the extinction from the dust map at `RA`
and `DEC`.

(format-text-light-curves)=
## SNANA text

TEXT writes one readable file per supernova: a header of `KEY: value` lines,
then the photometry table. To generate TEXT light curves with the SALT
example input:

```bash
snlc_sim examples/quickstart/sim.input FORMAT_MASK 2 NGENTOT_LC 300 GENVERSION FF_TEXT
```

```
SURVEY: DES
SNID: 000027
ZP_FLUXCAL: 31.40
SNTYPE: 1
RA: 52.500000
DEC: -21.500000
FILTERS: griz
PIXSIZE: 0.2700
FAKE: 2
MWEBV: 0.0000 +- 0.0000
MWXT_MAG:  -0.000 -0.000 -0.000 -0.000  # MW mag-XT per band
PEAKMJD: 56155.6797

REDSHIFT_HELIO: 0.073468 +- 0.001000
REDSHIFT_FINAL: 0.073001 +- 0.001000
MASK_REDSHIFT_SOURCE: 2
VPEC: 0.00 +- 0.00
LENSDMU: 0.000 +- 0.000

HOSTGAL_NMATCH: 0
HOSTGAL_NMATCH2: 0

SIM_DEC: -21.5
SIM_DLMU: 37.5905
...

NOBS: 16
NVAR: 11
VARLIST: MJD BAND FIELD FLUXCAL FLUXCALERR PHOTFLAG GAIN ZPT PSF SKY_SIG SIM_MAGOBS
OBS: 56179.2812 DES-g C2 4.0584e+04 1.7207e+02 0 1.000 32.360 1.560 4.860e+01 19.8368
OBS: 56179.2891 DES-r C2 7.9774e+04 6.4899e+02 0 1.000 30.880 2.980 3.009e+01 19.1066
OBS: 56179.2969 DES-i C2 5.2454e+04 4.6478e+02 0 1.000 32.100 2.750 8.427e+01 19.5873
OBS: 56179.3047 DES-z C2 0.0000e+00 1.0000e+08 1024 1.000 33.100 3.020 2.194e+02 19.5318
...
END_PHOTOMETRY:
```

A quantity with an uncertainty is written as `value +- error` on one line.
`FAKE: 2` marks a simulation and `FAKE: 3` a blinded one. The host keys
(`HOSTGAL_*`) and the simulated truth (`SIM_*`) follow the main header, each in
its own block. Under blinding the truth keys are left out.

`FLUXCAL` is calibrated flux, related to magnitude by
`m = ZP_FLUXCAL - 2.5 log10(FLUXCAL)`. Use the `ZP_FLUXCAL` value in the
header, which defaults to 31.4.

The `BAND` column contains the filter's full name as the calibration file
spells it, such as `DES-g`. If your reader expects one character, take the last
character of the name, as the calibration file does. The `FILTERS:` header key
lists the single characters alone.

Which columns appear depends on what the simulation produced. `SKY_SIG_T`
appears only when there is template noise, and the atmospheric columns (`dRA`,
`dDEC`, `AIRMASS` and the differential-refraction truth) only when the
atmosphere model was run.

Four columns are the observing conditions of the epoch, copied from the
SIMLIB:

- `GAIN`: photo-electrons per ADU, exactly as the SIMLIB row gave it.
- `ZPT`: the zero point converting magnitude to ADU, also as given.
- `PSF`: the first Gaussian width, in pixel sigma whatever `PSF_UNIT` the
  SIMLIB declared. An `ARCSEC_FWHM` PSF is divided by the pixel size and the
  FWHM-to-sigma ratio when the SIMLIB is read. Under a noise-equivalent-area
  `PSF_UNIT` the column is `PSF_NEA` instead, in square pixels.
- `SKY_SIG`: the sky noise, in ADU per square pixel whatever `SKYSIG_UNIT` the
  SIMLIB declared. An `ADU_PER_SQARCSEC` value is multiplied by the pixel size
  when it's read. `SKY_SIG_T` is in the same unit.

The remaining columns are measurements or simulated truth. `MJD` is in days,
and `FLUXCAL` and `FLUXCALERR` are on the calibrated scale above. `SIM_MAGOBS`
is the true magnitude before noise, `99` where the model produced nothing.

`PHOTFLAG` is a bit mask whose meanings you define. The simulation sets the
bits you name with `PHOTFLAG_DETECT`, `PHOTFLAG_TRIGGER`, `PHOTFLAG_SATURATE`,
`PHOTFLAG_SNRMAX` and `PHOTFLAG_NEARPEAK`. There is no standard assignment.

`SIM_SEARCHEFF_MASK` in the truth block records which stages of the trigger the
event passed: bit 1 for the detection pipeline, 2 for spectroscopic
confirmation, 4 for a host redshift and 8 for either of the last two.

Spectra, where present, follow the photometry in the same file as `SPECTRUM_*`
blocks.

### Host galaxy keys

The host block's length depends on how many galaxies were matched to the
event. An event with no matched host, which includes every event in a run
without a host catalogue, has `HOSTGAL_NMATCH: 0` and `HOSTGAL_NMATCH2: 0` and
no other `HOSTGAL_` key, as in the example above. A matched event gets up
to three blocks: `HOSTGAL_` for the nearest galaxy, then `HOSTGAL2_` and
`HOSTGAL3_` for the next two. The following example header has two matched
galaxies and host photometric-redshift quantiles:

```
HOSTGAL_NMATCH: 2
HOSTGAL_NMATCH2: 2
HOSTGAL_OBJID: 200101
HOSTGAL_FLAG: 0
HOSTGAL_PHOTOZ: 0.06003 +- 0.03000
HOSTGAL_SPECZ: 0.08734 +- 0.00100
HOSTGAL_RA: 52.500168  # deg
HOSTGAL_DEC: -21.500011  # deg
HOSTGAL_SNSEP: 0.570  # arcsec
HOSTGAL_DDLR: 0.276  # SNSEP/DLR
HOSTGAL_CONFUSION: -99.000
HOSTGAL_LOGMASS: 9.391 +- 0.100  # log10(Mgal/Msolar)
HOSTGAL_LOGsSFR: -1.050e+01 +- -9.999e+03  # specific SFR
HOSTGAL_ELLIPTICITY: 999.000
HOSTGAL_SQRADIUS: 999.000  # Ixx + Iyy
HOSTGAL_MAG:   22.047  20.915  19.236  18.356  # griz
HOSTGAL_MAGERR:   -9.000  -9.000  -9.000  -9.000  # griz

HOSTGALz_NBIN_QUANTILE_ZPHOT:  11
HOSTGALz_QUANTILE_ZPHOT: 0.0010 0.0187 0.0325 0.0424 0.0509 0.0588 0.0668 0.0753 0.0852 0.0990 0.1528
HOSTGALz_QUANTILE_PERCENT: 0.00 10.00 20.00 30.00 40.00 50.00 60.00 70.00 80.00 90.00 100.00

HOSTGALz_NBIN_LOGMASS_ZGRID:  0

HOSTGAL2_OBJID: 200100
...
HOSTGAL2z_NBIN_LOGMASS_ZGRID:  0

HOSTGAL_SB_FLUXCAL:  1238.29 3512.55 16490.08 37087.11  # griz/asec^2
```

Each galaxy's block contains:

- `OBJID`, the catalogue identifier, with `OBJID2` and `OBJID_UNIQUE` where the
  catalogue has them.
- `FLAG`, which the simulation writes as 0.
- `PHOTOZ` and `SPECZ`, the photometric and spectroscopic redshifts, each with
  its error.
- `RA` and `DEC`, then `SNSEP`, the separation in arcsec, and `DDLR`, the
  separation over the directional light radius.
- `CONFUSION`, in the first block only. The simulation writes -99 here. The
  fitter works out its own `HOST_CONFUSION` from the blocks' `DDLR` values.
- `LOGMASS`, `LOGSFR`, `LOGsSFR` and `COLOR`, each with its error. `LOGSFR`,
  `LOGsSFR` and `COLOR` are left out where the catalogue has no value.
- `ELLIPTICITY` and `SQRADIUS`, from the HOSTLIB columns of those names, or 999
  where the catalogue has none.
- `MAG` and `MAGERR`, the catalogue magnitudes, one value per band.

After each galaxy's keys come its two grids, named with a `z` after the prefix:

- `HOSTGALz_NBIN_QUANTILE_ZPHOT` is the number of photo-z quantiles, 0 for
  none. When it's above zero, `HOSTGALz_QUANTILE_ZPHOT` gives the quantile
  redshifts and `HOSTGALz_QUANTILE_PERCENT` the percentile each one sits at.
- `HOSTGALz_NBIN_LOGMASS_ZGRID` is the length of a grid of host mass against
  redshift. The simulation writes it as 0, with no grid.

The light-curve fit's redshift prior reads `HOSTGAL_PHOTOZ`, its error and
the `HOSTGALz_` quantiles of the first galaxy for each event, as
{doc}`fitting_light_curves` describes under `OPT_PHOTOZ`. The fit also accepts
quantiles written as `HOSTGAL_ZPHOT_Q<nnn>` columns, one per percentile.

The last key in the block is `HOSTGAL_SB_FLUXCAL`, the surface brightness of
the host under the event, per band, in `FLUXCAL` units per square arcsec. This
key, and `SIM_GALFRAC` in the truth block (the ratio of host to supernova peak
flux in a one-arcsec aperture), are only computed when `HOSTLIB_MSKOPT` has bit
2 set. Bit 2 also adds host light to the photometric noise.

## The `.LIST` file

Beside the photometry the simulation writes `<GENVERSION>.LIST`, which lists
the files in the version. It has one file name per line, relative to its own
directory, with no keys or header.

In TEXT mode it names every `.DAT` file, in the order they were written:

```
FF_TEXT_SN000012.DAT
FF_TEXT_SN000027.DAT
FF_TEXT_SN000028.DAT
```

In FITS mode it names only the HEAD file, one line per run that contributed to
the version. Each HEAD file names its own PHOT file, so that needs no entry:

```
FF_FITS_HEAD.FITS
```

The name is written without the `.gz` suffix even when the file on disk is
gzipped, so a reader must try both spellings. A run added to an existing
version appends to the list, so one version can collect several runs.

(combined-simulation-files)=
## Combined simulations

A pipeline `Sim` stage can combine several input files in one data version.

- The directory is the stage's name, and a single `<name>.LIST` in it names
  each part's header file.
- Each part is written as `<name>_<stem>_HEAD.FITS.gz` with its photometry
  companion, where `<stem>` is the input file's own name, suffixed by
  position where two inputs share one (two runs of `sim.input` give
  `<name>_sim_0` and `<name>_sim_1`).
- Each part keeps its own dump under the same prefix, and its own `.README`
  under `misc/`. The version's `<name>.README` merges them as SNANA's batch
  jobs do: the summed statistics, the input files, every part's types
  labelled by its model (`SNIaMODEL00`, `NONIaMODEL01` and so on, the type Ia
  models first) and every part's settings.
- Sequential CIDs are offset between parts. Random CIDs use disjoint portions
  of a shared identifier list when the inputs share a seed. With different
  seeds, random identifiers can overlap; use sequential CIDs when combining
  those simulations if identifiers must be unique.

(simulation-dump-columns)=
## Simulation dump

The `.DUMP` table records selected simulation quantities for each event.
`SIMGEN_DUMP` selects its columns; rejected events can be included with
`SIMGEN_DUMPALL`, as described in {doc}`simulation_keys`. Available columns
include:

- Identity and type: `CID`, `GENTYPE`, `SNTYPE`, `LIBID`, `FIELD`, `RA`,
  `DEC`.
- Redshift and distance: `GENZ` (or `ZCMB`), `ZHELIO`, `REDSHIFT_FINAL` for
  the redshift the survey would report, `VPEC`, `MU`, `LENSDMU`.
- Time and cadence: `PEAKMJD`, `NOBS`, `NEPOCH`, `TRESTMIN`, `TRESTMAX`,
  `TGAPMAX`.
- Extinction: `MWEBV`, with `MWEBVMAP` and `MWEBVERR` for the map's value and
  its error, and `AV`, `RV`.
- Selection: `SNRMAX` and `SNRMAX1` to `SNRMAX5`, `CUTMASK`, `FLAG_ACCEPT`,
  the trigger's `SIM_SEARCHEFF_MASK`, `MJD_TRIGGER`, `MJD_DETECT_FIRST`,
  `MJD_DETECT_LAST`.
- Template models: `SIM_TEMPLATE_INDEX` (or `NON1A_INDEX`) and
  `SIM_TYPE_NAME`.
- Host: `GALID`, `GALZTRUE`, `GALWGT` and `GALSNDM` (the magnitude shift the
  weight map applied under `HOSTLIB_MSKOPT` bit 4), plus columns named by
  `HOSTLIB_STOREPAR` and host properties under their `_TRUE` names.
- The model's parameters under their own names, e.g. `SALT2x1` and `SALT2c`,
  or `THETA` for BayeSN.
- Per-band columns using the band's single character: `PEAKMAG_<filter>`
  and `SNRMAX_<filter>`, and `HOSTMAG_<filter>`, `SBFLUX_<filter>` and
  `SBMAG_<filter>` where `HOSTLIB_MSKOPT` bit 2 has computed the host light.

`TRESTMIN` and `TRESTMAX` describe the prepared observing cadence, using
heliocentric redshift for the rest-frame conversion, rather than only the
epochs retained in the light curve.

To list the available columns without generating events, set
`SIMGEN_DUMP: 0` without a `SIMGEN_DUMPALL` column list. After initialising
the model and survey, the run prints the available columns and exits.

(simulation-readme)=
## The `.README` file

`<GENVERSION>.README` records the simulation configuration, filter
wavelengths, generated types and sample statistics. An abbreviated example is:

```
DOCUMENTATION:
  OVERVIEW:
    SURVEY:       DES
    GENMODEL:     jax_snlc_core/data/models/SALT3.DOVEKIE
    SNIA:         True
  INPUT_KEYS:
    GENVERSION:              FF_TEXT
    NGENTOT_LC:              300
  FILTERS:   # mean wavelength (A)
    g:  4827.7
    r:  6434.8
    i:  7828.1
    z:  9181.2
  GENTYPE_TO_NAME:  # GENTYPE:  [non]Ia   type
    1:  Ia       SALT3.DOVEKIE
  OUTPUT_SUMMARY:
    NGENEV_TOT:        300
    NGENLC_TOT:        300
    NGENLC_WRITE:      40
DOCUMENTATION_END:
```

`INPUT_KEYS` records the resolved settings and additional input lines.
`GENTYPE_TO_NAME` associates each generated type code with its supernova
type and model. SuperNNova uses it to identify type Ia events in training
simulations. Templates sharing one `GENTYPE` have their types joined by `+`.

The main event counts in `OUTPUT_SUMMARY` are:

- `NGENEV_TOT`: every event drawn.
- `NGENLC_TOT`: those that survived the generation ranges.
- `NGENLC_WRITE`: those written after the trigger and cuts, which is the number
  of light curves in the version.

The summary also reports selection efficiency, rejected-event counts and,
when applicable, subsurvey counts and host-association statistics.

(header-override-format)=
## Header override files

You can replace header values without rewriting the data by naming override
files: `HEADER_OVERRIDE_FILE` for real data, `SIM_HEADER_OVERRIDE_FILE` for a
simulation. Each file is a keyed-text table whose first column is the key,
`CID` to match events or `GALID` to match host galaxies, and whose other
columns name the header values to replace:

```text
VARNAMES: CID REDSHIFT_HELIO VPEC
SN: 1003  0.2491  250.0
```

A redshift override recomputes the other frame's redshift, and a host's
photo-z quantiles can be given as `QZPHOT00` to `QZPHOTnn` columns or one row
per quantile. A host's stellar mass as a function of redshift is given as one
row per redshift:

```text
VARNAMES: CID HOSTGALz_LOGMASS_ZGRID HOSTGALz_LOGMASS_VALGRID HOSTGALz_LOGMASS_ERRGRID
SN: 1003  0.10  9.84  0.06
SN: 1003  0.30  10.02  0.08
SN: 1003  0.50  10.11  0.10
```

The fitter interpolates this grid to estimate the host mass and its
uncertainty at the redshift used for the fit. For photo-z fitting, it also
evaluates the mass at the host's photo-z. Outside the grid, the catalogue
mass is retained. Supply both masses and errors on an increasing redshift
grid; the {ref}`interpolation details <host-mass-interpolation>` are given
with the fitting settings.
