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
light curves are combined into one data version, the type Ia components
first.

| Setting | Location | Purpose |
|---|---|---|
| `BASE` | Each component | Simulation input file. |
| Simulation keys | Each component | Changes to that component's input settings. |
| `GLOBAL` | Task | Simulation settings applied to every component. These take precedence over component settings. |
| `RANSEED_REPEAT` | `GLOBAL` | Number of repeated jobs and seed, such as `3 12345`. All jobs contribute to one data version. Default: `1 12345`. |
| `RANSEED_CHANGE` | `GLOBAL` | Number of independent realisations and seed. Each realisation produces a separate data version. |
| `FORMAT_MASK` | `GLOBAL` | Output format. Default: 48, for FITS output with random event identifiers. |
| `NGEN_UNIT` | `GLOBAL` | Rate-based event count: each component simulates `NGEN_UNIT` times its expected events per season. Replaces the fixed counts `NGENTOT_LC` and `NGEN_LC` of the simulation input files; giving `NGENTOT_LC` in the configuration as well is an error. |
| `RESET_CIDOFF` | `GLOBAL` | How event identifiers are offset between jobs: with 0 or 1 each component's first offset is rounded up on its first digit, with 2 it is not. Default: 1 with random identifiers, 0 otherwise. |
| `BLIND` | Task `OPTS` | Marks the simulated data for blinded BBC fits. Default: false. |
| `EXTERNAL` | Task | Completed Pippin simulation task to use instead of generating events. See {ref}`pippin-existing-fits`. |

Each repeated job generates the count its component's input sets with
`NGENTOT_LC` or `NGEN_SEASON`, or the `NGEN_UNIT` count; an input that sets
only `NGEN_LC` is refused, as SNANA's batch layer runs every job with
`NGEN_LC 0`.
`RANSEED_REPEAT` therefore increases the combined sample size, whereas
`RANSEED_CHANGE` produces separate samples of that size. If both are given,
the later one in `GLOBAL` applies.
{ref}`pippin-realisations` describes how later tasks use the separate
realisations; {ref}`pippin-seeds` gives the seed rules.

## `DATAPREP`

`DATAPREP` imports existing light curve data, for example observations from
a survey, for fitting and classification. `RAW_DIR` identifies the SNANA
data version to analyse.

| Setting | Location | Purpose |
|---|---|---|
| `RAW_DIR` | `OPTS` | Required: directory containing the data version. |
| `BLIND` | `OPTS` | Marks the data for blinded BBC fits. Default: true. |
| `TYPES` | `OPTS` | `IA` and `NONIA` lists of event types, which classifiers read. Default: `IA: [1]`, `NONIA: [2, 20, 21, 22, 29, 30, 31, 32, 33, 39, 40, 41, 42, 43, 80, 81]`. |
| `SIM` | `OPTS` | Accepted as metadata. |

This task uses `RAW_DIR`, rather than the `EXTERNAL` setting accepted by
`SIM`.

## `LCFIT`

An `LCFIT` task fits observed or simulated light curves with the chosen
model, writing the fitted parameters and uncertainties to FITRES tables.

`MASK` selects input tasks by name. For example, `MASK: DES` fits the
datasets from `SIM` or `DATAPREP` tasks whose names contain `DES`, including
every realisation of a simulation. A list selects tasks matching any entry,
and each entry must match at least one task. If omitted, or `*`, all
datasets are fitted.

| Setting | Location | Purpose |
|---|---|---|
| `BASE` | Task | Required: light curve fitting input file. |
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
of the data, or of the simulation: its `SIMLIB_SURVEY`, or else its SIMLIB
file's `SURVEY`. Here DES fits use both variations.
A task can list one FITOPTS file and inline entries together.

A positive top-level `FLAG_USE_SAME_EVENTS` in the FITOPTS file selects a
common event sample across variations, as setting `OPT_SNCID_LIST` to that
value in the fitting input does. The task's `OPTS` can set either key
instead; if both are given, `OPT_SNCID_LIST` applies. A positive flag in
the FITOPTS file takes precedence over both. Each FITOPT label may be given
once. {doc}`systematics` describes FITOPTs and sample selection.

A `BASE` file may begin with a `CONFIG` block ending at `#END_YAML`. Its
`FITOPT` list is used when the task gives no
`FITOPTS`, and its `FLAG_USE_SAME_EVENTS` or `OPT_SNCID_LIST` when the
task's `OPTS` give neither. A null value under `SNLCINP` or `FITINP`, which
Pippin uses to remove a key from the `BASE` file, is refused.

## `CLASSIFICATION`

`CLASSIFICATION` applies a photometric classifier to observed or simulated
light curves, or trains one on simulations. `SuperNNovaClassifier` provides
these operations using SuperNNova. The predicted type Ia probabilities are
added to every FITOPT's fitted-results table for use in BBC.
{doc}`classification` provides training and prediction examples.

| Setting | Location | Purpose |
|---|---|---|
| `CLASSIFIER` | Task | `SuperNNovaClassifier`, or `PerfectClassifier`, which gives each event the probability its known type implies. |
| `MODE` | Task | Required: `train` or `predict`. `PerfectClassifier` only predicts. |
| `MASK`, `MASK_SIM` | Task | Select simulation or data tasks whose names contain this text; the predictions join every fit of a selected task. If both are given, both must match. |
| `MODEL` | `OPTS` | Required for prediction by a classifier that is trained: the name of a training task of the same classifier listed earlier, or the absolute path of a model file. Not accepted on a training task. |
| `PROB_COLUMN_NAME` | `OPTS` | Name of the output type Ia probability column. |
| `EXTERNAL_DIRS` | Task | Directories containing saved `predictions.csv` files to import instead of classifying again; prediction tasks only. |

The type Ia probability column is `PROB_<task>` for `PerfectClassifier`. A
SuperNNova training task writes `PROB_<task>_<simulation>`, which its
predictions inherit; a prediction from a model file writes `PROB_<folder>`,
after the folder holding it.
`PROB_COLUMN_NAME` replaces any of these. A prediction from a training
task that trained on several simulations is made once for each; BBC reads
the last.

Event types come from `DATAPREP` `TYPES`, from the `GENTYPE` and
`GENMODEL` of each simulation component (Ia for SALT, BayeSN and other type
Ia models, with photometric types 100 higher), or from a completed
simulation's `config.yml`. `PerfectClassifier` gives each event the
type Ia probability when its fitted `TYPE` is one of these Ia types.
SuperNNova labels events with the types in `SNTYPES`, or else with these
event types, adding type 1 as Ia when none is Ia and type 29 as core
collapse when none is core collapse. Types 1, 10 and 110 are always Ia.

An imported prediction directory must keep the name Pippin gives the task:
`<task>_<simulation><suffix>`, with the realisation number after `<task>`
when the simulation has several, and the suffix `_<training task>_<simulation>`
for a model trained in the configuration, or `_<folder>` for a model file.

`EXTERNAL_MAP` and `COMBINE_MASK` are not supported by the Pippin reader. A
task with another classifier is skipped, and a task that depends on it is
refused. Additional classification tools can use the Python interface
described in {doc}`classification`.

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
| `CLASSIFIER` | Task | Classification task whose probabilities are used, or the probability column it writes. |
| `PROB_COLUMN_NAME` | Task | Probability column to use, taking precedence over `CLASSIFIER`. |
| `OPTS` | Task | BBC settings and the pipeline controls described below. |
| `MUOPTS` | Task | Named standardisation variations. Each accepts its changed BBC settings under `OPTS`. |

Leave `datafile`, `simfile_biascor` and `simfile_ccprior` out of `BASE` when
the task supplies the corresponding fitted datasets. If `BASE` is omitted,
the default is Pippin's `surveys/des/bbc/bbc_5yr.input`, searched for under
the configured data directories.

In `OPTS`, selection windows are written as `CUTWIN_<column>: low high`,
in either case. `CUTWIN_PROB_IA` selects on BBC's `varname_pIa`, the chosen
type Ia probability column. Of a list of windows for one column, BBC
applies the last. Any other task `OPTS` setting, and every MUOPT `OPTS` setting, takes one value.
The remaining scientific settings are listed in {doc}`bias_correction_keys`.

### MUOPTs and BIASCOR fits

A MUOPT can select a different `SIMFILE_BIASCOR`, `SIMFILE_CCPRIOR` or
`CLASSIFIER` at the same level as its `OPTS`. Its optional `SCALE` sets the
uncertainty scale used in covariance construction. {doc}`systematics`
describes these variations.

`SIMFILE_BIASCOR_FITOPTS` and `SIMFILE_CCPRIOR_FITOPTS`, at task or MUOPT
level, select the FITOPT of each BIASCOR or contamination input. Give one
number per input, in the same order; zero selects its nominal fit. A
MUOPT's numbers apply only to the simulations it names itself.

Within task `OPTS`, `REPLACE_SIMFILE_BIASCOR` and `REPLACE_SIMFILE_CCPRIOR`
map data FITOPT labels to external FITRES files. For example:

```yaml
OPTS:
  REPLACE_SIMFILE_BIASCOR:
    MWEBV: /data/corrections/mwebv.FITRES
```

This uses the supplied BIASCOR file for the `MWEBV` FITOPT. The path must be
absolute. A MUOPT supplying its own BIASCOR or contamination simulation takes
precedence over the corresponding replacement.

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

Task-level `BLIND` decides whether BBC is blinded. Without it, a
`blindflag` in `OPTS` other than 0 blinds BBC; without either, BBC is
blinded if any input data task is. BBC then runs with `blindflag` 2 when
blinded and 0 otherwise.

### Classification probabilities

Task-level `USE_RECALIBRATED: true` selects existing recalibrated
probabilities: `PROB_` in the task's column becomes `CPROB_`, so
`PROB_NAME` becomes `CPROB_NAME`. A MUOPT's own classifier keeps its
column, and a MUOPT without simulations of its own reads the task's with
that column. Alternatively, give the recalibrated column directly through
`PROB_COLUMN_NAME`. The column must be present in the data FITRES tables
or imported predictions; this option does not recalibrate probabilities.
`CUTWIN_PROB_IA` uses the selected column too.

## `CREATE_COV`

A `CREATE_COV` task combines statistical distance uncertainties with the
correlated shifts from FITOPT and MUOPT variations to construct covariance
matrices for cosmology fitting. It uses the distances from the BBC tasks
selected by its mask. Its settings go under `OPTS`.

| Setting | Purpose |
|---|---|
| `MASK` or `MASK_BIASCOR` | Selects BBC tasks whose names contain this text. This setting goes at task level, outside `OPTS`. |
| `COVOPTS` | Additional covariance selections, such as `[NOSYS] [=DEFAULT,=DEFAULT]`. The full covariance is always labelled `ALL`. |
| `BINNED` | Use BBC's binned distances when true, the default; false selects individual supernova distances unless rebinning is requested. |
| `SYSTEMATIC_HD` | Write a Hubble diagram for each FITOPT/MUOPT combination. Default: false. |

{ref}`covariance-construction` describes the calculation and selections.

### Uncertainty scales

The covariance takes its scales from a FITOPTS file: the one `SYS_SCALE`
names, or else the one used to fit the data of the first matched BBC task
that used one. That task's data must not have been fitted with different
FITOPTS files. `FITOPT_SCALES` and `MUOPT_SCALES` override these scales in
turn, each by exact label; all three settings go in `OPTS`. A MUOPT's own
`SCALE` comes first, so any of the three overrides it. A label the BBC fit
does not have is ignored, and a systematic with no scale has a scale of one.

A FITOPTS file holds survey or `GLOBAL` blocks, each entry a scale followed
by fitting settings, as shown above. Only the scales are used here, and
entries from all blocks are read, a later block's winning. The standalone
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
        - calibration 1.0 $DATA/calibration_covariance.npz
      EXTRA_OPTS: --write_mask_cov 2
```

An external matrix has no factorized form, so the write mask must include
the systematic (1) or inverse (2) covariance; the default, 8, writes neither.
Each entry is `label scale file`, the first three words. The scale
multiplies the uncertainty and is squared for the covariance. The file path
may contain environment variables and must be absolute once they are
expanded.
{ref}`external-covariance-files` describes the file formats.

`CALIBRATORS` lists supernova identifiers in Cepheid host galaxies. In an
unbinned analysis, their contributions to redshift systematics labelled
`VPEC` or `ZSHIFT` are excluded; other systematics are retained.

`SUBTRACT_VPEC: true` removes the peculiar-velocity contribution from the
statistical distance uncertainties, for use with an external velocity
covariance. It selects unbinned distances, whatever `BINNED` says.
Rebinning turns it off.
{ref}`covariance-peculiar-velocities` gives the definition, including BBC
error rescaling, and how the standalone covariance command combines
subtraction with rebinning.

### Covariance output options

`EXTRA_OPTS` accepts the following flags:

| Flag | Purpose |
|---|---|
| `--write_format_cov npz` or `text` | Dense matrices as NumPy archives or gzipped text; default: `npz`. Factorized output always uses NumPy archives. |
| `--write_mask_cov N` | Select output forms using the values in {ref}`covariance-output-settings`. Default: 8, the factorized total covariance. Include the systematic (1), inverse (2) or factorized total (8) covariance for cosmology fitting. |
| `--label_cov_rows` | Add identifiers, survey and redshift to covariance rows, which it writes as text. |
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
| `MASK` | Task | Selects `CREATE_COV` tasks whose names contain this text. |
| `WFITOPTS` | `OPTS` | Required list of `/label/ flags` entries, each a cosmology fit. The label is one word; an entry without one, and with no `/`, is named by its position, `WFITOPT000` onwards. |
| `WFITOPT_GLOBAL` | `OPTS` | Flags applied to every entry, appended after its individual flags. |
| `COVOPTS`, or `COVOPT` | `OPTS` | Labels of the covariances to fit, as one string such as `"ALL NOSYS"`. Default: every covariance. |
| `BLIND_DATA`, `BLIND_SIM` | `OPTS` | Blind the fits to data and to simulations. Default: data true, simulations false. Blinding simulations alone is not supported. |
| `USE_COVTOT_FACTORIZED`, `USE_COVSYS_INV` | `OPTS` | Which covariance file each fit reads: the factorized one if written and `USE_COVTOT_FACTORIZED` is true, else the inverse if written and `USE_COVSYS_INV` is true, else the systematic covariance. Default: both true. |

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

Each entry is fitted for every covariance the matched tasks make, one per
BBC fit, using `ALL` and each selection the `CREATE_COV` task's `COVOPTS`
requests, or only those the WFIT `COVOPTS` lists; a covariance with none of
them listed is not fitted. {doc}`cosmology_keys` lists the
available flags. Blinding is set by `BLIND_DATA` and `BLIND_SIM`, not by
`-blind` in the flags. The fits are read only from `COSMOFIT: WFIT`, and a
`wfit` section that gives none is refused.

(pippin-existing-fits)=
## Reusing completed simulations and fits

`SIM: EXTERNAL` names a completed Pippin simulation task: its folder, or the
`config.yml` in it. The versions listed under `OUTPUT: sim_folders` are used
in that order, named `<task>-0001`, `<task>-0002` and so on when there are
several. The simulation is blinded as its `OUTPUT: blind` records,
whatever the task's own `BLIND` says.

`LCFIT: EXTERNAL_DIRS` imports completed fitting tasks:

```yaml
SIM:
  DATA:
    EXTERNAL: /previous/run/simulation_task
LCFIT:
  D:
    BASE: fit.nml
    EXTERNAL_DIRS: [/previous/run/D_DATA]
```

An entry is used for the dataset whose task name it carries
(`D_DATA` here: fit task `D`, dataset `DATA`); a dataset with no matching
entry is fitted. The directory contains `output/SUBMIT.INFO`
and, for each version, the FITOPT tables under `output/<folder>`. The
folder is named after the simulation folder: the data directory's own name
for `DATAPREP` or `SIM: EXTERNAL`, or Pippin's `<PREFIX>_<CONFIG>_<SIM>`
(with `-0001` etc. for realisations) for a simulation the configuration
makes, where `<CONFIG>` is the configuration file's name in capitals and
`<PREFIX>` is `GLOBAL: QUEUE: prefix`, `PIP` by default.

The imported `SUBMIT.INFO` supplies the FITOPT labels.

(pippin-realisations)=
## Repeated simulations and sample splits

`RANSEED_CHANGE: 3 12345` under a simulation's `GLOBAL` generates three
independent realisations. A task named `DATA` produces `DATA-0001`,
`DATA-0002` and `DATA-0003`, with its components combined within each.
A `RANSEED` setting in the task overrides every realisation's seed, so the
realisations become the same sample.

References such as `DATA: [D_DATA]` select the corresponding fit of each
realisation. Repeated `DATA` inputs to the same BBC task must have the same
number of realisations and are paired by index; a `DATA` input without
`RANSEED_CHANGE` is reused in each analysis. A BIASCOR or contamination
simulation is read at its first realisation, with a warning when it has
several. Every realisation of the data receives separate BBC,
covariance and cosmology outputs.

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

Without random CIDs, each repeated job derives its own seed from the
supplied one. `RANSEED_CHANGE` does this whether or not CIDs are random.

Each job's `CIDOFF`, and with random CIDs its `CIDRAN_MIN` and
`CIDRAN_MAX`, are set as SNANA's batch layer sets them, from the number of
events each job generates, so a version has room for all its identifiers
however many events it holds. A `CIDOFF`, `CIDRAN_MIN` or `CIDRAN_MAX` given in the
task's settings replaces them.

## Included anchor files

A comment line `# % include: anchors.yml %` is replaced by the
named file before the configuration is read, so that tasks can use YAML
anchors defined there. The path may contain environment variables; a
relative path is taken from the directory in which `jax_pipeline` runs.

## Unused tasks and scheduling settings

Sections with no stage here are skipped: `AGGREGATION`, `MERGE` and
`ANALYSE`, any `COSMOFIT` entry other than `WFIT`, and sections Pippin
itself does not read, such as one holding YAML anchors. Classification
already joins probabilities to the fitted tables, so no separate `MERGE`
task is needed. An `AGGREGATION` task's `RECALIBRATION` is ignored, and
refused with `LEGACY: True`; `MERGE_CLASSIFIERS` is refused.

Settings for Pippin's batch layer are ignored because jax-SNANA runs the
analysis as a single process: `BATCH_INFO`, `BATCH_FILE`, `BATCH_REPLACE`,
`BATCH_MEM`, `BATCH_WALLTIME`, `NUM_JOBS`, `GPU`, `CLEAN` and `INI_DIR`.

Other settings ignored here:

- Of `GLOBAL`, which overrides Pippin's installation settings, all but
  `DATA_DIRS` and `QUEUE: prefix`.
- `COSMOMC` settings, as no CosmoMC analysis is run.
- A `SIM` task's `GENVERSION`, which Pippin replaces with its own name.
- The `LCFIT` option `APPEND_TABLE_VARLIST`, whose extra fitted-table
  columns are not written.
- The `CLASSIFICATION` setting `MASK_FIT`, which has no effect in Pippin.
- The `BIASCOR` settings `MAKE_ALL_HUBBLE` and `CONSISTENT_SAMPLE`, which
  only control Pippin's own plots and iteration.

Skipped sections and ignored settings are reported in the log; other
unrecognised task settings are rejected.
