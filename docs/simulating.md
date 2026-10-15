# Simulating light curves

jax-SNANA simulates transient surveys using SALT, BayeSN and other transient
models. Simulated light curves follow the survey's observing cadence and
conditions, with configurable noise, detection efficiencies and selection
cuts. The supernova population can be specified to include distributions of
model parameters, correlations with host-galaxy properties and evolution
with redshift.

For a first simulation and fit with the supplied BayeSN or SALT inputs, follow
{doc}`getting_started`.

## Running a simulation

### SNANA input files

The survey's observing history is supplied through a {ref}`SIMLIB
<format-simlib>`, with each entry giving a sequence of observations at one
sky position. Each simulated event uses one of these sequences, including
the observing times, filters, zero points, sky noise and PSFs. The
{ref}`calibration file <format-kcor>`, specified by `KCOR_FILE`, supplies the
filter transmissions and reference standards used to calculate the photometry.

To use these survey files for a BayeSN G26 simulation, the simulation input
file `sim.input` would contain:

```text
GENMODEL:           /path/to/BAYESN.G26
GENMODEL_MSKOPT:    1
SIMLIB_FILE:        survey.SIMLIB
KCOR_FILE:          survey.fits
GENFILTERS:         griz
GENVERSION:        BAYESN_SIM
NGENTOT_LC:         1000
RANSEED:            42
BATCH_SIZE:         1000
GENRANGE_REDSHIFT:  0.05 1.2
GENRANGE_PEAKMJD:   60540 61460
GENRANGE_TREST:     -20 50
OPT_MWEBV:          0
DNDZ:               POWERLAW 2.5e-5 1.5
GENPEAK_THETA:      0
GENSIGMA_THETA:     1 1
GENRANGE_THETA:     -3 3
GENTAU_AV:          0.1716
GENRANGE_AV:        0 1
GENPEAK_RV:         1.8347
GENSIGMA_RV:        0.2300 0.2300
GENRANGE_RV:        1.2 6.0
```

`GENPEAK_<PAR>`, `GENSIGMA_<PAR>` and `GENRANGE_<PAR>` specify the peak,
widths and truncation range of a parameter's distribution. `GENTAU_AV`
sets the exponential scale for extinction. {doc}`supernova_populations`
describes the distributions available for BayeSN, SALT and other models.

`NGENTOT_LC` requests 1000 generated events before selection; other run-size
options are described under {ref}`simulation-event-counts`.
`OPT_MWEBV: 0` disables Milky Way extinction; see
{ref}`sim-mwebv-options` for the extinction settings.

`DNDZ` specifies the volumetric supernova rate used to draw redshifts.
`POWERLAW A B` gives a rate $A(1+z)^B$, so this example uses a rate at
redshift zero of $2.5\times10^{-5}$ events per comoving Mpc³ per rest-frame year,
increasing as $(1+z)^{1.5}$. Other rate models are listed in
{doc}`simulation_keys`.

Run the simulation with `snlc_sim`. Trailing `KEY value` pairs override the
input settings:

```bash
snlc_sim sim.input
snlc_sim sim.input NGENTOT_LC 5000 RANSEED 43
```

### YAML

The same model, survey and population settings can be written as
`simulation.yaml`:

```yaml
genmodel: /path/to/BAYESN.G26
genmodel_mskopt: 1
simlib_file: survey.SIMLIB
kcor_file: survey.fits
genfilters: griz
genversion: BAYESN_SIM
ngentot_lc: 1000
random_seed: 42
batch_size: 1000
redshift_range: [0.05, 1.2]
peakmjd_range: [60540, 61460]
trest_range: [-20, 50]
opt_mwebv: 0
dndz_entries: [[POWERLAW, 2.5e-5, 1.5]]
populations:
  THETA: {distribution: normal, peak: 0, sigma: 1, range: [-3, 3]}
  AV: {distribution: exponential, exp_tau: 0.1716, range: [0, 1]}
  RV: {distribution: normal, peak: 1.8347, sigma: 0.2300, range: [1.2, 6.0]}
```

For YAML inputs, named command-line options override settings:

```bash
snlc_sim simulation.yaml
snlc_sim simulation.yaml --ngentot-lc 5000 --random-seed 43
```

YAML settings use the same names as Python keyword arguments, with hyphens
replacing underscores in command-line options. {doc}`simulation_keys` lists
these names alongside the SNANA keys.

### Execution settings

The output goes to `<GENVERSION>/` under `PATH_SNDATA_SIM`, or under the
working directory if that key is unset. The first batch includes JAX
compilation, so it usually takes longer than later batches.

`batch_size` controls how many events are processed at once. The default
is 1000. Larger batches use more memory and can improve throughput,
particularly on a GPU.

For a fixed `RANSEED`, the random draws for each event do not depend on batch
size. Runs use single precision by default; `DOUBLE_PRECISION: 1` selects
double precision. Changing precision can change the simulated sample even
with the same seed. In a pipeline, precision is set once for the whole run,
as described in {doc}`building_pipelines`.

## Running in a Python script

Use `simulate` to run a survey simulation and return its arrays:

```python
from jax_snlc_sim import simulate

results, selection, spectra = simulate("simulation.yaml")
```

To use `sim.input` and change settings for a run:

```python
results, selection, spectra = simulate("sim.input", random_seed=43, batch_size=32)
```

You can also supply the settings as Python keyword arguments or a dictionary,
using the names shown in the YAML example. Keyword arguments override a
supplied file or dictionary.

`results` contains arrays, including `fluxcals`, `fluxcalerrs` and the sampled
`parameters`. `selection` contains the survey's selection masks and noisy
photometry; `spectra` contains any requested spectra. The simulation uses
the same survey, population and selection settings as a command-line run,
but does not write data files.

To extract the photometry of the first event that passes selection, exclude
padded and saturated observations using the masks:

```python
import numpy as np

i = np.flatnonzero(selection["passing_mask"])[0]
valid = (np.asarray(selection["valid_mask"][i])
         & ~np.asarray(selection["saturated_mask"][i]))
flux = np.asarray(selection["fluxcals_noisy_sat"][i])
error = np.asarray(selection["fluxcalerrs_sat"][i])
valid &= np.isfinite(flux) & np.isfinite(error)
mjd = np.asarray(results["observation_schedule"]["mjds"][i])[valid]
filter_index = np.asarray(results["observation_schedule"]["filter_indices"][i])[valid]
flux, error = flux[valid], error[valid]
```

`filter_index` identifies bands in calibration-file order, restricted to
the filters selected by `GENFILTERS`.

Each call prepares the survey and generates a realisation with its specified
seed. For repeated compiled realisations, see {doc}`sbi`.

### Running in a Python pipeline

A `Sim` stage runs a simulation and writes its light curves for use by
later analysis stages:

```python
from jax_pipeline import Sim, run

sim = Sim("des_ia", "sim.input",
          overrides={"NGENTOT_LC": 50_000, "RANSEED": 1},
          batch_size=20_000)
version = run(sim, outdir="run")["des_ia"]
```

`overrides` replaces keys of the input file without editing it, so one input
file can serve a nominal run and each of its systematics. The stage writes
its data version under `outdir`. {doc}`building_pipelines` shows how to
connect the simulation to a light curve fit and describes
{ref}`reuse of completed results <pipeline-reruns>`.

(simulation-combined-inputs)=
A stage can combine several input files into one data version, for example
to include several source models or simulation seeds. See
{ref}`combining-pipeline-simulations` for the setup and
{ref}`combined-simulation-files` for the resulting files.

(simulation-event-counts)=
## Event counts and redshifts

`DNDZ` selects the volumetric rate model used to draw redshifts. Together
with `SOLID_ANGLE` and `GENRANGE_PEAKMJD`, it also sets the expected survey
event count. {doc}`simulation_keys` lists the available rate models.

The number of events is set by one of three keys:

- `NGEN_SEASON` scales the rate-predicted count by the specified number of
  seasons.
- `NGENTOT_LC` generates a fixed number of events before selection.
- `NGEN_LC` generates until that many events have survived the trigger and
  the cuts.

`NGEN_SCALE` multiplies whichever is given.

The time of peak is drawn uniformly over `GENRANGE_PEAKMJD`, skipping any
window `GENRANGE_PEAKMJD_EXCLUDE` names. Leave enough margin either side of
the observing seasons to include events caught only on the rise or the
decline.

Each event is also assigned one SIMLIB entry, which sets its position on the
sky and so its Milky Way extinction, taken from the SIMLIB header or from a
dust map according to `OPT_MWEBV`.

{doc}`supernova_populations` describes the parameter distributions, dust,
redshift evolution and correlations, including {ref}`SUBPOP <simulating-subpop>`
for several populations in one simulation.

(simulating-hosts)=
## Host galaxies

A {ref}`HOSTLIB <format-hostlib>` is a catalogue of potential host galaxies
and their properties. For example:

```text
HOSTLIB_FILE: my_hosts.HOSTLIB
HOSTLIB_WGTMAP_FILE: host_weights.dat
```

`HOSTLIB_FILE` supplies the catalogue; the optional `HOSTLIB_WGTMAP_FILE`
sets the relative hosting probabilities. The {ref}`HOSTLIB input format
<format-hostlib>` describes the required columns and weight-map syntax.
A host is drawn for every event. Candidates are galaxies within
the redshift tolerance specified by `HOSTLIB_DZTOL`. Hosts are selected
using the weight map, or uniformly when no map is supplied.
The map may depend on the event's own parameters as well as the
galaxy's, so you can use it to simulate a correlation between host mass and
supernova brightness or stretch.
With `HOSTLIB_MSKOPT` bit 16, the supernova takes its selected host's
redshift, so the generated redshift distribution also depends on the
catalogue and the matching tolerance.

The event is placed within the chosen galaxy, drawn from the galaxy's light
profile. The simulator assigns an observed host using the directional light
radius (DDLR): the separation between event and galaxy in units of the
galaxy's size along the line joining them. `HOSTLIB_MAXDDLR` sets how large
it may be before the event is recorded as hostless.

Host detection and incorrect associations can be configured with:

- The true host may be too faint to detect, which `HOSTLIB_SNR_DETECT` and
  `HOSTLIB_MAG_DETECT` simulate.
- `WRONGHOST_FILE` supplies a probability of a wrong association and a table
  of true and matched redshifts. For a wrong association, the simulator draws
  a matched redshift from this table and uses it to select a host. This
  models redshift contamination statistically, separately from the
  DDLR-based host association.

To include noise from host-galaxy light within the photometric aperture,
supply a catalogue with a Sersic profile and set bit 2 of `HOSTLIB_MSKOPT`.
The simulator then calculates the host surface brightness at the supernova's
position and includes its contribution to the flux uncertainty.

The host supplies properties such as stellar mass and photometric redshift,
including photo-z quantiles when present. {doc}`simulation_inputs` describes
the catalogue columns, and {doc}`light_curve_data` describes the matched
host, neighbours and true-host identifiers in the output. Host-reuse and
quantile-generation controls are listed in {doc}`simulation_keys`.

## Photometry and noise

Each simulated supernova uses the observing times and conditions from its
assigned SIMLIB entry. `GENRANGE_TREST` specifies the rest-frame phase window
about the time of peak, converted to observer-frame times using the
heliocentric redshift. The resulting light curve follows the survey cadence
on that sightline, including its gaps, seasons and weather.

For each observation, the model spectrum at the rest-frame phase includes
host extinction in the rest frame and Milky Way extinction in the observer
frame. The redshifted spectrum is integrated through the observer-frame
filter using the reference spectrum specified by the calibration.
The calibrated flux is reported on the
`ZP_FLUXCAL` 31.4 scale described in {doc}`light_curve_data`. Noise is then added
from the observing conditions the SIMLIB describes:

- Poisson noise from the source itself.
- Sky noise over the effective aperture.
- Detector read noise.
- The host-galaxy surface brightness beneath the event.
- The noise of the template image that difference imaging subtracts.

Calibration scatter can perturb the photometry, while the flux-error model
and noise settings adjust the simulated noise and reported uncertainties;
see {ref}`sim-noise-settings`. The simulation also accounts for saturation
and source flux already present in the template image.

For intrinsic scatter, including BayeSN residuals and SALT scatter models,
see {ref}`population-scatter`.

## Trigger and selection cuts

A generated event is written out only if it passes the trigger and then the
selection cuts.

`APPLY_SEARCHEFF_OPT` selects the required detection and follow-up stages:

- For each observation, the efficiency curve in
  `SEARCHEFF_PIPELINE_EFF_FILE` gives the probability that the measurement
  was detected. The pipeline logic in `SEARCHEFF_PIPELINE_LOGIC_FILE` then
  determines whether those detections would have flagged the object, for
  example two detections in any of *griz*, with detections close together
  in time counting once.
- For follow-up, a redshift is obtained either from a spectrum of the event,
  using `SEARCHEFF_SPEC_FILE`, or from its host, using
  `SEARCHEFF_zHOST_FILE`. Each is drawn once per event. The map formats and
  field matching are described in {ref}`format-efficiency`.

A stage whose efficiency file is not supplied passes every event. The run
summary counts failures at each stage; `SIM_SEARCHEFF_MASK` records the
individual outcomes in the {doc}`simulation output <light_curve_data>`.

The `CUTWIN_*` keys act on the number of epochs, the signal-to-noise, the
phase coverage, the peak magnitude and any generated variable by name. They
are always evaluated and reported, but reject events only when
`APPLY_CUTWIN_OPT` is non-zero. That key is not a mask, and any non-zero
value enables every cut.

Before applying detection efficiencies, the simulator requires at least
`CUTWIN_NEPOCH` observations in the phase window, or at least one if that
setting is omitted. It also requires model flux in at least one observation;
an event whose filters all lie outside the model's wavelength coverage is
rejected at this point. These requirements apply even when
`APPLY_CUTWIN_OPT` is zero. The rejection counts are described under
{ref}`simulation-output`.

(simulating-selection-example)=
### Selection example

`examples/quickstart/sim_selection.input` adds selection settings to the
SALT quick-start simulation. It requires detections in two separate observing
periods and a redshift from either the supernova or its host. Its settings
are:

```text
SEARCHEFF_PIPELINE_EFF_FILE: examples/quickstart/selection_detection.dat
SEARCHEFF_PIPELINE_LOGIC_FILE: examples/quickstart/selection_logic.dat
SEARCHEFF_SPEC_FILE: examples/quickstart/selection_spec.dat
SEARCHEFF_zHOST_FILE: examples/quickstart/selection_host.dat
NEWMJD_DIF: 0.5
APPLY_SEARCHEFF_OPT: 9
SIMGEN_DUMPALL: 5 CID GENZ SIM_SEARCHEFF_MASK CUTMASK MJD_TRIGGER
```

The four files contain the detection efficiency, discovery logic,
supernova spectroscopic efficiency and host-redshift efficiency.
`APPLY_SEARCHEFF_OPT=9` requires the discovery trigger and a redshift from
either the supernova or its host. `NEWMJD_DIF=0.5` separates the observing
periods used by the trigger.

Run the example from the repository root:

```bash
snlc_sim examples/quickstart/sim_selection.input
```

The `QUICKSTART_SELECTION.DUMP` file includes rejected events and their
selection flags. {doc}`simulation_inputs` describes the efficiency-file
formats and {ref}`simulation output <simulation-output>` explains the dump.

For your survey, use measured efficiencies and the actual trigger
requirements, applying the same selection model to the BIASCOR simulation.

(simulation-output)=
## Output

The closing summary reports how many events were generated, how many passed
the survey's detection and follow-up requirements, and how many passed the
selection cuts.

The run writes a data version, a directory named for `GENVERSION` containing
the light curves and the files that describe them. `FORMAT_MASK` chooses
either FITS (32, the default) or one TEXT file per event (2), and both are
described in {doc}`light_curve_data`. The directory also contains:

- A `.LIST` naming the data files, one per line: a FITS run's single header
  file or a TEXT run's one file per event. The fitter reads it to find them.
- A `.README` recording the simulation settings and the numbers of events
  generated, written and rejected by each selection; see the
  {ref}`README description <simulation-readme>` for its contents.
- A `SIMGEN_DUMP` table of per-event truth, where the input asks for one.
- {doc}`Simulated spectra <simulating_spectra>`, when requested.
- A per-observation `.DCR` table of atmospheric position shifts, when
  enabled through {ref}`ATMOSPHERE_OPTMASK <sim-atmosphere-options>`.

### True parameter values

Every event is written with its generated redshift, time of peak, population
parameters, extinction and true magnitudes as `SIM_` columns, for comparison
with a fit. Events rejected by the trigger or the cuts are counted in the
summary. `SIMGEN_DUMPALL` puts every generated event into the dump table,
rejected ones included, for studying the selection function. The key takes a
count followed by that many column names, or one comma-separated list:

```text
SIMGEN_DUMPALL: 9 CID GENZ PEAKMJD MU THETA AV RV SIM_SEARCHEFF_MASK CUTMASK
```

The dump contains exactly the columns named, with `FLAG_ACCEPT` appended, so
name the model's parameters if you want them. The
{ref}`simulation dump section <simulation-dump-columns>` lists the available
names, including model parameters, host properties and per-band quantities.
