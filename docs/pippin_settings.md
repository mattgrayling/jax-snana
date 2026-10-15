# Pippin settings

This page lists the Pippin tasks and settings supported by jax-SNANA.
For commands and a worked example, see {doc}`snana_and_pippin`.
The scientific input settings are listed under {doc}`simulation_keys`,
{doc}`fitting_keys` and {doc}`bias_correction_keys`.

```{contents} Tasks and settings
:local:
:depth: 1
```

(pippin-simulation-settings)=
## `SIM`

A `SIM` task generates simulated light curves for a survey, using the model,
population and observing conditions specified in its input files. It can
combine several simulations, for example type Ia and core-collapse samples.
Each of these components names its simulation input through `BASE`; their
light curves are combined into one data version.

| Setting | Location | Purpose |
|---|---|---|
| `BASE` | Each component | Simulation input file. |
| Simulation keys | Each component | Changes to that component's input settings. |
| `GLOBAL` | Task | Simulation settings applied to every component. These take precedence over component settings. |
| `RANSEED_REPEAT` | `GLOBAL` | Number of repeated jobs and seed, such as `3 12345`. All jobs contribute to one data version. Default: `1 12345`. |
| `RANSEED_CHANGE` | `GLOBAL` | Number of independent realisations and seed. Each realisation produces a separate data version. |
| `FORMAT_MASK` | `GLOBAL` | Output format. Default: 48, for FITS output with random event identifiers. |
| `NGEN_UNIT` | `GLOBAL` | Rate-based event count, using the simulator's `NGEN_SEASON` setting. Overrides fixed counts from `NGENTOT_LC` or `NGEN_LC`. |
| `BLIND` | Task `OPTS` | Marks the simulated data for blinded BBC fits. Default: false. |
| `EXTERNAL` | Task | Existing data version or completed Pippin simulation task to use instead of generating events. |

Each repeated job requests the event count given in its component's input.
`RANSEED_REPEAT` therefore increases the combined sample size, whereas
`RANSEED_CHANGE` produces separate samples of that size. Use one of these
settings, not both. {ref}`pippin-realisations` describes how later tasks use
the separate realisations; {ref}`pippin-seeds` gives the seed rules.

## `DATAPREP`

`DATAPREP` imports existing light curve data, for example observations from
a survey, for fitting and classification. `RAW_DIR` identifies the SNANA
data version to analyse.

| Setting | Location | Purpose |
|---|---|---|
| `RAW_DIR` | `OPTS` | Required: directory containing the data version. |
| `BLIND` | `OPTS` | Marks the data for blinded BBC fits. Default: true. |
| `TYPES`, `SIM` | `OPTS` | Accepted as metadata; neither filters the input tables. |

This task uses `RAW_DIR`, rather than the `EXTERNAL` setting accepted by
`SIM`.

## `LCFIT`

An `LCFIT` task fits observed or simulated light curves with the chosen
model, writing the fitted parameters and uncertainties to FITRES tables.

`MASK` selects input tasks by name. For example, `MASK: DES` fits the
datasets from `SIM` or `DATAPREP` tasks whose names contain `DES`. If
omitted, all datasets are fitted.

| Setting | Location | Purpose |
|---|---|---|
| `BASE` | Task | Light curve fitting input file. |
| `MASK` | Task | Selects `SIM` or `DATAPREP` task names. |
| `SNLCINP`, `FITINP` | Task | Changes to the corresponding blocks of the fitting input. |
| `FITOPTS` | Task | Light curve fitting variations, supplied inline or through a FITOPTS file. |
| `EXTERNAL_DIRS` | Task | Completed fitting-task directories to import. See {ref}`pippin-existing-fits`. |

The selected dataset supplies `VERSION_PHOTOMETRY` and `PRIVATE_DATA_PATH`.
A fit is named `<task>_<dataset>`; for example, task `D` fitted to dataset
`DATA` produces `D_DATA`, which can be named in `BIASCOR`.
Fitting settings belong in `SNLCINP` or `FITINP`, rather than task `OPTS`.

### FITOPTs

Inline entries have the form `/label/ KEY value`:

```yaml
FITOPTS:
  - /zpt_g/ MAGOBS_SHIFT_ZP g 0.003
```

Alternatively, `FITOPTS` can name a YAML file. Its entries contain an
uncertainty scale followed by the fitting settings:

```yaml
GLOBAL:
  zpt_g: "1.0 MAGOBS_SHIFT_ZP g 0.003"
DES:
  filter_g: "1.0 FILTER_LAMSHIFT g 10.0"
```

`GLOBAL` entries apply to every survey; the other blocks apply to the survey
named by the simulation or data. Here DES fits use both variations.
Inline entries have a scale of one. A task can list a FITOPTS file and
inline entries together.

A top-level `FLAG_USE_SAME_EVENTS: 1` in the FITOPTS file enables a common
event sample across variations, equivalent to `OPT_SNCID_LIST = 1` in the
fitting input. {doc}`systematics` describes FITOPTs and sample selection.

## `CLASSIFICATION`

`CLASSIFICATION` applies a photometric classifier to observed or simulated
light curves, or trains one on simulations. `SuperNNovaClassifier` provides
these operations using SuperNNova. The predicted type Ia probabilities are
added to every FITOPT's fitted-results table for use in BBC.
{doc}`classification` provides training and prediction examples.

| Setting | Location | Purpose |
|---|---|---|
| `CLASSIFIER` | Task | `SuperNNovaClassifier`, or `PerfectClassifier` for simulation-truth tests. |
| `MODE` | Task | Required: `train` or `predict`. |
| `MASK`, `MASK_SIM` | Task | Select simulation or data tasks for SuperNNova. If both are given, both must match. |
| `MODEL` | `OPTS` | Required for prediction: a trained model file or the name of a training task. |
| `PROB_COLUMN_NAME` | `OPTS` | Name of the output type Ia probability column. |
| `EXTERNAL_DIRS` | Task | Directories containing saved `predictions.csv` files to import instead of classifying again; prediction tasks only. |

For an imported prediction directory, retain Pippin's task/version naming so
that the predictions can be matched to the corresponding fitted dataset.

`EXTERNAL_MAP`, `COMBINE_MASK` and other classifiers are not supported by
the Pippin reader. Additional classification tools can use the Python
interface described in {doc}`classification`.

### SuperNNova options

Under `OPTS`, SuperNNova accepts `LIST_FILTERS`, `VARIANT`, `REDSHIFT`,
`NORM`, `CYCLIC`, `SEED`, `BATCH_SIZE`, `NUM_LAYERS`, `HIDDEN_DIM` and
`SNTYPES`. See {doc}`classification` for examples and links to the
SuperNNova settings documentation and API defaults.

## `BIASCOR`

A `BIASCOR` task runs BBC to estimate supernova distances from the fitted
light curve parameters, including standardisation and bias corrections.
It can also account for contamination using classification probabilities
and a contamination simulation. {doc}`bias_corrections` describes the
method and its inputs.

| Setting | Location | Purpose |
|---|---|---|
| `BASE` | Task | BBC input file. |
| `DATA` | Task | Fitted dataset names, such as `[D_DATA]`. |
| `SIMFILE_BIASCOR` | Task | Fitted BIASCOR simulation names. |
| `SIMFILE_CCPRIOR` | Task | Fitted contamination simulation names. |
| `CLASSIFIER` | Task | Classification task whose probabilities are used, or the name of an existing probability column. |
| `PROB_COLUMN_NAME` | Task | Explicit probability-column name; takes precedence over `CLASSIFIER` when selecting the column. |
| `OPTS` | Task | BBC settings and the pipeline controls described below. |
| `MUOPTS` | Task | Named standardisation variations. Each accepts its changed BBC settings under `OPTS`. |

Leave `datafile`, `simfile_biascor` and `simfile_ccprior` out of `BASE` when
the task supplies the corresponding fitted datasets. If `BASE` is omitted,
the default is Pippin's `surveys/des/bbc/bbc_5yr.input`, searched for under
the configured data directories.

In `OPTS`, selection windows are written as `CUTWIN_<column>: low high`.
`CUTWIN_PROB_IA` selects on the chosen type Ia probability column.
The remaining scientific settings are listed in {doc}`bias_correction_keys`.

### MUOPTs and BIASCOR fits

A MUOPT can select a different `SIMFILE_BIASCOR`, `SIMFILE_CCPRIOR` or
`CLASSIFIER` at the same level as its `OPTS`. Its optional `SCALE` sets the
uncertainty scale used in covariance construction. {doc}`systematics`
describes these variations.

`SIMFILE_BIASCOR_FITOPTS` and `SIMFILE_CCPRIOR_FITOPTS`, at task or MUOPT
level, select the FITOPT of each BIASCOR or contamination input. Give one
number per input, in the same order; zero selects its nominal fit.

Within task `OPTS`, `REPLACE_SIMFILE_BIASCOR` and `REPLACE_SIMFILE_CCPRIOR`
map data FITOPT labels to external FITRES files. For example:

```yaml
OPTS:
  REPLACE_SIMFILE_BIASCOR:
    MWEBV: corrections/mwebv.FITRES
```

This uses the supplied BIASCOR file for the `MWEBV` FITOPT. A MUOPT supplying
its own BIASCOR or contamination simulation takes precedence over the
corresponding replacement.

### Event selection

With `OPT_SNCID_LIST=1` in the light curve fitting input, BBC starts from
the events accepted in the nominal analysis and retains those accepted
across all requested variations. It then fits every combination
using that common sample. Set `FLAG_USE_SAME_EVENTS: 0` in task `OPTS`
for independent sample selections. {ref}`bbc-event-synchronisation`
describes the full rules, including `NOREJECT` exceptions.

`FITOPTxMUOPT`, in task `OPTS`, selects combinations by their FITOPT and
MUOPT numbers. Zero denotes the nominal fit. For example:

| Selection | Combinations run |
|---|---|
| `0+0` | Each FITOPT with MUOPT000, and each MUOPT with FITOPT000. |
| `1&2` | FITOPT001 with MUOPT002 only. |

A list runs the combinations selected by any of its entries. Covariance
construction requires FITOPT000/MUOPT000 and MUOPT000 for each selected
FITOPT with a nonzero MUOPT. A separate `FITOPT` key within a MUOPT is not
supported. `NSPLITRAN` divides the data into disjoint subsets, as described
under {ref}`pippin-realisations`.

### Blinding

By default, BBC is blinded if any input data task is marked as blinded.
Task-level `BLIND` overrides this choice; `blindflag` in `OPTS` takes
precedence over both.

### Classification probabilities

Task-level `USE_RECALIBRATED: true` selects existing recalibrated
probabilities. A selected column `PROB_NAME` becomes `CPROB_NAME`, including
when a MUOPT changes classifiers. Alternatively, give the recalibrated
column directly through `PROB_COLUMN_NAME`. The column must be present in
the data FITRES tables or imported predictions; this option does not
recalibrate probabilities. `CUTWIN_PROB_IA` uses the selected column too.

## `CREATE_COV`

A `CREATE_COV` task combines statistical distance uncertainties with the
correlated shifts from FITOPT and MUOPT variations to construct covariance
matrices for cosmology fitting. It uses the distances from the BBC tasks
selected by its mask. Its settings go under `OPTS`.

| Setting | Purpose |
|---|---|
| `MASK` or `MASK_BIASCOR` | Selects BBC task names. This setting goes at task level, outside `OPTS`. |
| `COVOPTS` | Additional covariance selections, such as `[NOSYS] [=DEFAULT,=DEFAULT]`. The full covariance is always labelled `ALL`. |
| `BINNED` | Use BBC's binned distances when true, the default; false selects individual supernova distances unless rebinning is requested. |
| `SYSTEMATIC_HD` | Write a Hubble diagram for each FITOPT/MUOPT combination. Default: false. |

{ref}`covariance-construction` describes the calculation and selections.

### Uncertainty scales

The covariance uses the scale in each FITOPT or MUOPT definition. A matching
label in a `SYS_SCALE` file overrides that value. `FITOPT_SCALES` and
`MUOPT_SCALES` override those values in turn; all three settings go in `OPTS`.
`FITOPT_SCALES` matches labels by substring, while `MUOPT_SCALES` requires
an exact label.

`SYS_SCALE` names a file in the FITOPTS format shown above, with survey or
`GLOBAL` blocks and a scale followed by fitting settings. Only the scales
are used here, and entries from all blocks are read. The standalone
covariance command's `SYS_SCALE_FILE` instead uses a flat mapping of labels
to scales, as shown in {doc}`systematics`.

### Rebinning

Within each BBC redshift bin, rebinning groups supernovae by their fitted
light curve parameters and calculates a distance estimate for each group.
The covariance is then calculated between these distance estimates.

For SALT, `REBINNED_X1` and `REBINNED_C` set the numbers of stretch and colour
bins. Alternatively, `REBIN_BINS` specifies the FITRES columns and bin counts, for
example `REBIN_BINS: {THETA: 2, AV: 3}` for BayeSN. An empty mapping groups
by redshift bin alone.

### External covariances and calibrators

`EXTRA_COVS` adds external covariance matrices to an unbinned analysis:

```yaml
CREATE_COV:
  COV:
    MASK_BIASCOR: BBC
    OPTS:
      BINNED: false
      EXTRA_COVS:
        - calibration 1.0 calibration_covariance.npz
```

Each entry is `label scale file`. The scale multiplies the uncertainty and
is squared for the covariance. {ref}`external-covariance-files` describes
the file formats.

`CALIBRATORS` lists supernova identifiers in Cepheid host galaxies. In an
unbinned analysis, their contributions to redshift systematics labelled
`VPEC` or `ZSHIFT` are excluded; other systematics are retained.

`SUBTRACT_VPEC: true` removes the peculiar-velocity contribution from the
statistical distance uncertainties, for use with an external velocity
covariance. It selects unbinned distances unless rebinning is requested.
{ref}`covariance-peculiar-velocities` gives the definition, including BBC
error rescaling and subtraction before rebinning.

### Covariance output options

`EXTRA_OPTS` accepts the following flags:

| Flag | Purpose |
|---|---|
| `--write_format_cov npz` or `text` | Dense matrices as NumPy archives or gzipped text; default: `npz`. Factorized output always uses NumPy archives. |
| `--write_mask_cov N` | Select output forms using the values in {ref}`covariance-output-settings`. Default: 8, the factorized total covariance. Include inverse covariance (2) or factorized total covariance (8) for cosmology fitting. |
| `--label_cov_rows` | Add identifiers, survey and redshift to text covariance rows. |
| `--mxsize_test_posdef N` | Maximum matrix dimension for the positive-definiteness check; default: 6000. Zero skips the check. |

For example, `EXTRA_OPTS: "--write_format_cov text --write_mask_cov 7"`
writes systematic, inverse total and total covariance matrices as text.

## `COSMOFIT`

`COSMOFIT` estimates cosmological parameters from the Hubble diagram and
covariance matrices produced by `CREATE_COV`. jax-SNANA supports `wfit`.
Pippin allows a choice of cosmology fitter, so its input structure places
the named fits and their settings under `COSMOFIT: WFIT`, as shown below.

| Setting | Location | Purpose |
|---|---|---|
| `MASK` | Task | Selects `CREATE_COV` task names. |
| `WFITOPTS` | `OPTS` | Required list of `/label/ flags` entries. Each specifies a cosmology fit. |
| `WFITOPT_GLOBAL` | `OPTS` | Flags applied to every entry, appended after its individual flags. |

For example:

```yaml
COSMOFIT:
  WFIT:
    cosmology:
      MASK: COV
      OPTS:
        WFITOPT_GLOBAL: --method grid
        WFITOPTS:
          - /lcdm/ -lcdm
```

Each entry is fitted for every matched covariance task, using `ALL` and
each selection requested through `COVOPTS`. {doc}`cosmology_keys` lists the
available flags.

(pippin-existing-fits)=
## Reusing completed simulations and fits

`SIM: EXTERNAL` accepts a data-version directory or a completed Pippin
simulation task. For a task directory, `config.yml` lists the versions under
`OUTPUT: sim_folders`; relative paths are resolved from that directory.

`LCFIT: EXTERNAL_DIRS` imports completed fitting tasks:

```yaml
SIM:
  DATA:
    EXTERNAL: /previous/run/simulation_task
LCFIT:
  D:
    EXTERNAL_DIRS: [/previous/run/D_DATA]
```

The fitting directory's name matches the fit task and dataset (`D_DATA`
here). It contains `output/SUBMIT.INFO` and an `output/<version>` directory
for each realisation, with the FITOPT tables inside. Versions are numbered
consecutively from `-0001`; simulation and fit versions are matched by that
number, independently of their order in the saved configuration.

The imported `SUBMIT.INFO` supplies the FITOPT labels. A task can mix
imported fits and datasets that need fitting. `BASE` is required only for
the latter, or when a supplied `FITOPTS` configuration needs interpreting.

(pippin-realisations)=
## Repeated simulations and sample splits

`RANSEED_CHANGE: 3 12345` under a simulation's `GLOBAL` generates three
independent realisations. A task named `DATA` produces `DATA-0001`,
`DATA-0002` and `DATA-0003`, with its components combined within each.
Do not override `RANSEED` when using `RANSEED_CHANGE`.

References such as `DATA: [D_DATA]` select the corresponding fit of each
realisation. Repeated inputs to the same BBC task must have the same number
of realisations and are paired by index. An input without `RANSEED_CHANGE`,
such as a shared BIASCOR simulation, is reused in each analysis. Every
realisation receives separate BBC, covariance and cosmology outputs. If a
fitting mask selects only some realisations, name their fits explicitly in
`BIASCOR`.

`NSPLITRAN: 3`, under BBC task `OPTS`, divides the data into three disjoint
subsets using `CID % 3 + 1` as the subset index. It requires non-negative
integer CIDs and keeps an event in the same subset across systematic
variations. The BIASCOR and contamination simulations remain whole.
A task named `BBC` produces `BBC-SPLIT001`, `BBC-SPLIT002` and
`BBC-SPLIT003`; with repeated simulations the names include both suffixes,
for example `BBC-0001-SPLIT001`.

(pippin-seeds)=
### Random seeds

The seed supplied in `RANSEED_REPEAT` or `RANSEED_CHANGE` replaces the base
input's `RANSEED`. For `RANSEED_REPEAT` with random CIDs, jobs share that
seed and use distinct event offsets to generate separate events and
non-repeating identifiers.

Without random CIDs, repeated job `j` uses the supplied seed plus
`10000 j + j² + 13`, with `j` starting at zero. `RANSEED_CHANGE` uses this
formula whether or not CIDs are random.

## Unused tasks and scheduling settings

`AGGREGATION`, `MERGE`, `ANALYSE` and `COSMOMC` are skipped. Classification
already joins probabilities to the fitted tables, so no separate `MERGE`
task is needed.

Batch-submission settings are ignored because jax-SNANA runs the analysis
as a single process. These include `BATCH_INFO`, `BATCH_FILE`,
`BATCH_REPLACE`, `BATCH_MEM`, `BATCH_WALLTIME` and `NUM_JOBS`, including
simulation `GLOBAL` settings. Classification task settings `GPU` and
`CLEAN`, and covariance `OPTS` settings `INI_DIR` and `COSMOMC`, are also
ignored. `RESET_CIDOFF` is ignored; event offsets are assigned automatically.
Ignored settings and skipped tasks are reported in the log.
Unrecognised Pippin task sections and task settings are rejected.
