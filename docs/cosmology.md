# Cosmology fitting

`wfit` fits cosmological parameters to a Hubble diagram, using its distance
uncertainties and an optional covariance matrix. The input can contain binned
distances or individual supernovae. {ref}`covariance-construction` describes
how to prepare the diagram and covariance, including systematic uncertainties.

For the complete workflow, start with {doc}`cosmology_analysis`.

{doc}`cosmology_keys` lists the available input settings.

## Running the fit

The `wfit` command accepts SNANA's cosmology-fit flags. For example:

```bash
wfit SALT2mu.M0DIF
```

This fits flat wCDM and writes a summary
(`SALT2mu.M0DIF.wfit.yaml`) and a single-row parameter file
(`SALT2mu.M0DIF.cospar`) beside the input.

## Hubble-diagram inputs

The Hubble diagram comes from the
{doc}`bias-correction stage <bias_corrections>`, in either of its two forms. A
`.M0DIF` file gives distances in redshift bins, and a `.FITRES` file gives one
row per supernova. The fitter recognises which you supplied from its columns.

To include systematic uncertainties, also give a matrix from
{ref}`covariance construction <covariance-construction>`.

## Configuring the fit

Cosmology settings use a flat YAML mapping:

```yaml
hubble_diagram: COVARIANCE/hubble_diagram.txt
covariance: COVARIANCE/covtot_inv_000.npz
covariance_kind: total_inverse
model: wcdm
method: grid
ommin: 0.1
ommax: 0.5
wmin: -1.5
wmax: -0.5
```

`covariance_kind` states whether the file contains systematic covariance,
total covariance or inverse total covariance. The inference methods,
external constraints and blinding controls are described later on this page.

Run it with:

```bash
wfit --config cosmology.yaml --method grid --omsteps 101
```

Command-line settings override the configuration file. The Python
`Cosmology` pipeline stage accepts the same file through `input_file` and
changes through `overrides`. {doc}`cosmology_keys` lists the available settings.

`-double_precision` runs the fit in double precision instead of the default
single precision.

## Running in a Python script

Use `fit_cosmology` to fit a Hubble diagram from a script:

```python
from jax_wfit import fit_cosmology

fits = fit_cosmology(hubble_diagram="distances.M0DIF", model="wcdm")
parameters = fits[0]["point_estimate"]
```

It returns one result dictionary per diagram and writes files only with
`write=True`. The {doc}`Python API <api>` also shows how to supply a
Hubble diagram and covariance already in memory.

## Using a pipeline

The `Cosmology` stage takes the covariance stage directly and fits the requested
covariance choices in one call:

```python
from jax_pipeline import Cosmology, run

cosmology = Cosmology("cosmology", covariance=covariance,
                      input_file="wfit.yml", covopts=("ALL", "NOSYS"))
results = run(cosmology, outdir="run")
omega_m = results["cosmology"]["ALL"]["point_estimate"]["omega_m"]
```

In a pipeline, the stage name determines the output filenames, so do not
specify those filenames in the input configuration.

## Cosmological models

`model` selects the cosmology to fit:

`lcdm`
: Flat ΛCDM. The only free parameter is the matter density Ω<sub>m</sub>.

`wcdm`
: Dark energy has a constant equation of state *w*, fitted alongside
  Ω<sub>m</sub>. This is the default.

`w0wa`
: Flat *w*<sub>0</sub>*w*<sub>a</sub>CDM with the CPL parametrisation,
  fitting Ω<sub>m</sub>, *w*<sub>0</sub> and *w*<sub>a</sub>.

All three are flat, so the dark energy density is not a separate parameter.

Reference distances include radiation by default; set `radiation: false`
to omit it. {doc}`cosmology_keys` and {doc}`snana_compatibility` describe the
cosmological settings and the conventions used for external constraints.

The fit marginalises over a common magnitude offset and does not constrain
H<sub>0</sub>. The offset prior is set by `intercept_sigma`.

## External constraints

External constraints can be included through command-line flags or the
corresponding YAML settings:

- `-ompri` and `-dompri` set the centre and width of a Gaussian matter-density prior.
- `-cmb` includes a constraint on the CMB shift parameter, with measurement
  and last-scattering redshift set by `Rcmb`, `sigma_Rcmb` and `cmb_redshift`.
- `-bao DESI_2024` includes the BAO measurements and covariance supplied in
  `$SNDATA_ROOT/models/BAO`. `SDSS4_2020` is also supported.

For example:

```bash
wfit SALT2mu.M0DIF -cmb -ompri 0.315 -dompri 0.02
```

`-cmb_sim` and `-bao_sim` centre these constraints on the simulated cosmology
for bias tests. {ref}`cosmology-prior-settings` gives the adopted measurements,
BAO calibration assumptions and interactions between prior settings.

## Covariance input

If you pass a covariance matrix, the flag you use declares what it contains:

`systematics`
: Systematic terms only, given with `-mucovsys_file`. Your measurement errors
  are added to its diagonal.

`total`
: Total covariance, including measurement uncertainties, given with
  `--cov-total`. The `.COV` file from the bias-correction stage is of this
  kind.

`total_inverse`
: An already-inverted total covariance, given with `-mucovtot_inv_file`. The
  standalone covariance command writes this by default. Redshift cuts and error
  adjustments cannot be applied to an inverse, and are refused.

`factorized`
: Statistical variances and systematic shift vectors, stored together in
  `covtot_factorized_NNN.npz`. This represents the total covariance without
  storing a square matrix over all supernovae, reducing memory use for large
  unbinned samples. Supply it with `-mucovtot_factorized_file` or YAML
  `covariance_kind: factorized`. {ref}`cosmology-covariance-settings` lists
  supported cuts and adjustments; {doc}`analysis_outputs` gives the stored arrays.

The covariance must match the full diagram, before any redshift cut, and a cut
removes the same rows from both.

For example, to use a factorized file with any of the inference methods below:

```bash
wfit COVARIANCE/hubble_diagram.txt --covariance COVARIANCE/covtot_factorized_000.npz --covariance-kind factorized
```

## Inference methods

`--method` selects the inference method:

`laplace`
: Minimises the χ² directly with a damped Newton iteration, and takes the
  uncertainty from the curvature at the minimum. This is the default.
  It reports a mode and gives no result if the minimum is pinned
  against a parameter bound, or if the curvature there is not positive
  definite.

`nuts`
: Samples the posterior using NUTS, with sampling diagnostics in the output.

`grid`
: Evaluates the likelihood over a fixed parameter grid, with configurable
  bounds and spacing.

Parameter bounds are set with `wmin`/`wmax` for *w*<sub>0</sub>,
`wamin`/`wamax` for *w*<sub>a</sub>, and `ommin`/`ommax` for Ω<sub>m</sub>.
The grid and NUTS methods report marginal constraints even when the posterior
extends to a bound; the Laplace method requires an interior minimum.

NUTS runs four chains by default, with 1,000 warm-up steps and 1,000 samples
each. These are set by `nuts_chains`, `nuts_warmup` and `nuts_samples`.

## Output

The `success` field records whether the fit converged.

The default Laplace method reports the best-fitting point. Grid fits report
the posterior mean by default; set `marg: false` in YAML, or pass `--no-marg` (SNANA spelling
`-minchi2`), to report the best-fitting grid point instead. NUTS reports
the posterior mean.

The scalar uncertainty in the YAML and Python summaries is half the width
of the central 68.3% posterior interval. `-sig_type std` reports the standard
deviation instead, as does `-minchi2`. The summaries also give the quantile
endpoints. The asymmetric errors in the text output use those quantiles,
as described in {doc}`analysis_outputs`.

The parameters reported depend on `model`, and have the same names in the YAML
summary and in a Python result:

| `model` | Keys in `point_estimate` |
|---|---|
| `lcdm` | `omega_m` |
| `wcdm` | `omega_m`, `w0` |
| `w0wa` | `omega_m`, `w0`, `wa` |

### The summary YAML

The summary is always written, and is named after the Hubble diagram. It
contains the parameter estimates, uncertainties and correlations, the χ²
contributions from the supernovae and external priors, and a record of the
inputs and settings.

The `diagnostics` block records convergence for the selected inference
method; {doc}`analysis_outputs` describes its fields.

### The `.cospar` file

This file is always written, and gives the main results as a single row in
SNANA's format: parameter estimates and uncertainties, correlations, the
figure of merit and the total χ². Under ΛCDM, the `w` column contains the
fixed value of -1 and its uncertainty is zero.

The {ref}`cosmology output descriptions <format-cosmology-output>` describe the fields
in both files, including the degrees-of-freedom convention and how uncertainty
columns depend on the inference method.

### Optional outputs

The other outputs are written only if you ask for them, each with the key given
here and described in {doc}`cosmology_keys`:

- `-outfile_resid`: residuals per supernova about the fitted cosmology.
- `-outfile_chi2grid`: the χ² grid. Needs `--method grid`.
- `-outfile_prob1d`: the grid's one-dimensional marginals. Needs `--method grid`.
- `--output-samples`: the posterior draws. Needs `--method nuts`.
- `--output-covariance` and `-outfile_mucovtot_inv`: the total distance
  covariance the fit used, and its inverse, in the text form the covariance
  stage writes.

### Several realisations

For several Hubble diagrams or covariance choices, the YAML summary lists
the results under `realisations`. Other output files have a realisation
index, for example `cosmology.r000.cospar`. The {doc}`analysis_outputs` page
lists the output filenames.

## Multiple datasets

If you supply several Hubble diagrams, each is fitted independently. The
diagrams can contain different numbers of supernovae or redshift bins.

```bash
wfit real_0.M0DIF real_1.M0DIF real_2.M0DIF --truth-info INFO_0.YML INFO_1.YML INFO_2.YML
```

`--truth-info` reads the simulated cosmology from each simulation's own
`INFO.YML`, one file per dataset, or one file for all of them when they share a
simulation. Where every dataset has a truth value, the summary includes a
section giving, for each parameter, the mean recovered bias, its scatter and
standard error, the mean and width of the pull, and the fraction of datasets
whose 68% interval contains the truth.

The `Cosmology` pipeline stage accepts the same `truth_info` setting through
its input file or `overrides`. One file supplies truth for all selected
covopts; a list follows their order. Alternatively, supply `om_sim`, `w0_sim`
and `wa_sim` directly. The truth values enter the fit summaries and any
simulation-centred external priors.

(cosmology-blinding)=
## Blinding

Blinding conceals the reported cosmological parameter values while analysis
choices are being made. The command-line controls are:

- `-blind` shifts the reported values by a fixed unknown amount.
- `-unblind`, the default, leaves them as fitted.
- `-blind_auto` shifts them only for real data.

`-blind_auto` uses the `ISDATA_REAL` header written by the covariance stage
to distinguish data from simulations. The output records whether blinding
was applied. Blinded summaries omit the seed and recovered-bias diagnostics;
{ref}`cosmology-blinding-settings` gives the configuration details.
