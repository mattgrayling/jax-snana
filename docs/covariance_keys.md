# Covariance settings

{ref}`covariance-construction` explains how to construct covariance
matrices from systematic variations. This page lists the settings.

YAML and Python use the **Python / YAML** names in the tables below.
Command-line options replace underscores with hyphens, such as
`--outdir covariance`. SNANA's uppercase YAML keys and CLI flags are
also supported, including
`-i`, `-v` and `-o` for input directory, version and output directory.

```{eval-rst}
.. configtable:: jax_create_cov.config.CreateCovConfig
   :field-keys:
   :explicit-keys:
   :omit-sections: Not set by an input key
```

(covariance-output-settings)=
## Covariance file selection

`write_mask_cov` combines these output choices:

| Value | Output |
|---|---|
| 1 | Systematic covariance, `covsys_000.npz`. |
| 2 | Inverse total covariance, `covtot_inv_000.npz`. |
| 4 | Total covariance, `covtot_000.npz`. |
| 8 | Statistical variances and systematic shift vectors, `covtot_factorized_000.npz`. |

The number in the filename identifies the covopt. Python and Pippin pipelines
default to 8; the standalone command defaults to 2. Use 10 to write both.
When a selected covopt includes an external dense covariance, requesting
factorized output writes a dense inverse for that covopt instead. Dense
systematic and total matrices can also be requested.
`write_format_cov` selects `npz` or `text` for dense matrices; factorized
outputs use NumPy archives. {ref}`covariance-output-files` gives the schemas.

(covariance-rebin-settings)=
## Rebinning

`rebin_bins` subdivides each BBC redshift bin using equal-width bins in the
specified FITRES columns. Bins empty in the reference sample are omitted.
Each output redshift corresponds to the bin's average reference distance,
rather than its average redshift.

Each column's bins span the reference sample. The mapping can contain any
number of columns; `{}` uses the BBC redshift bins alone. Input FITRES files
must include `IZBIN`, which BBC writes for every fit. The Python
`build_covariance` function accepts `rebin_bins` with `mode="rebin"`.

The SALT-specific `nbin_x1` and `nbin_c` settings use SNANA's bin-index
convention, which assumes consecutive `IZBIN` values. If the input has gaps
in those indices, use `rebin_bins` to retain all occupied bins.

(covariance-selecting-systematics)=
## Selecting systematics

A FITOPT variation is compared with the nominal analysis. A MUOPT variation
is compared with the nominal distance fit of its own FITOPT. Distances are
compared as residuals relative to each run's reference cosmology. The outer
product of a variation's distance-shift vector gives its systematic
covariance contribution. Where both FITOPT and MUOPT variations apply,
their uncertainty scales multiply.

Each covariance option, or *covopt*, labels a matrix and selects the FITOPT
and MUOPT variations contributing to it:

```yaml
COVOPTS:
  - '[NOSYS] [=DEFAULT,=DEFAULT]'
  - '[CALIB] [+zpt,=DEFAULT]'
```

`NOSYS` contains statistical uncertainties alone. `CALIB` includes FITOPTs
whose labels contain `zpt`, with the nominal MUOPT.

The corresponding structured YAML or Python settings are:

```yaml
covopts:
  - label: NOSYS
    fitopt_filter: {exact: DEFAULT}
    muopt_filter: {exact: DEFAULT}
  - label: CALIB
    fitopt_filter: {include: zpt}
    muopt_filter: {exact: DEFAULT}
```

In the SNANA input, the first bracket gives the output label. The second
contains the FITOPT filter followed by the MUOPT filter:

| Structured filter | SNANA filter | Match |
| --- | --- | --- |
| `exact: NAME` | `=NAME` | The whole label is `NAME`. |
| `include: NAME` | `+NAME` | The label contains `NAME`. |
| `exclude: NAME` | `-NAME` | The label does not contain `NAME`. |

An empty filter accepts everything. A third value in the second bracket
sets the uncertainty scale; for example, `[CALIBx2] [+zpt,=DEFAULT,2.0]`
doubles the assigned uncertainty.

Matching is case-insensitive; all conditions in a structured filter must
hold. An omitted filter accepts every label. A covopt's `scale` multiplies
the uncertainty assigned to its selected variations, so its square
multiplies their covariance contribution.

A matrix containing all systematic variations in the input is built
automatically and labelled `ALL`. Its diagonal supplies the
systematic-uncertainty column of the Hubble diagram. `--nosys` restricts
this matrix to the unmodified MUOPT and omits that diagnostic column.

### Covopt entries

Each `COVOPTS` entry selects the systematic variations contributing to one
covariance matrix. The settings below correspond to the
`[label] [fitopt_filter,muopt_filter]` syntax above.

```{eval-rst}
.. configtable:: jax_create_cov.config.Covopt
```

(covariance-reference-cosmology)=
## Reference cosmology

The reference cosmology is that of the BIASCOR simulation. BBC outputs
record it when the simulation's light curves were fitted by jax-SNANA.
For outputs without this information, supply a `COSMOLOGY` block in the
SNANA input, or a `cosmology` mapping in Python/YAML. It must include the
matter and dark-energy densities and agree with any recorded cosmology.

```{eval-rst}
.. configtable:: jax_create_cov.config.Cosmology
   :field-keys:
```

Reference distances use H<sub>0</sub> = 70 km s⁻¹ Mpc⁻¹.

This cosmology is needed for rebinning and for the `MUSIM` column of a
simulated sample. A cosmology fit that interpolates between two Hubble
diagrams (`--hdibc`) also needs it for each diagram, with a single `MUSHIFT`
value rather than a range.

Reference distances include radiation by default. Set `radiation: false`
to use the convention of SNANA's `create_covariance`. This affects `MUSIM`
and the redshifts assigned to rebinned distances; use the convention of your
input reference distances.

`DOUBLE_PRECISION: true` selects double precision for reference-distance
calculations, used for `MUSIM` and rebinned redshifts. Python uses the active
JAX precision unless this setting is supplied.

## External covariance matrices

`EXTRA_COVS` adds a covariance computed elsewhere, given as a file of
supernova pairs and their covariance. It requires an unbinned diagram,
since it is matched by supernova. Its weight scales the uncertainty and
is therefore applied squared.

For example, the input file can include:

```yaml
UNBINNED: true
EXTRA_COVS:
  - 'calibration 1.0 calibration_covariance.npz'
```

For a Python pipeline, supply each external covariance as a label,
uncertainty scale and file:

```python
covariance = Covariance(
    "covariance", distances=distances, mode="unbinned",
    extra_covs=[("calibration", 1.0, "calibration_covariance.npz")],
)
```

{ref}`external-covariance-files` describes the file formats.

## Calibrators

`CALIBRATORS` names the supernovae in Cepheid host galaxies. Their rows and
columns are zeroed in redshift systematics identified by `VPEC` or `ZSHIFT`
in the FITOPT or MUOPT label, because a calibrator's distance does not
depend on its redshift. Other systematic contributions are retained.

Calibrators require an unbinned diagram:

```yaml
UNBINNED: true
CALIBRATORS: [SN2011fe, SN2012cg]
```

In a Python stage, use `mode="unbinned"` and
`calibrators=["SN2011fe", "SN2012cg"]`.

(covariance-peculiar-velocities)=
## Peculiar-velocity covariance

For an unbinned diagram, `subtract_vpec=True` removes `MUERR_VPEC` in
quadrature before constructing the covariance. This allows an external
velocity covariance to replace the diagonal velocity term. With rebinning,
subtraction is applied to each event's `MUERR` before binning; `MUERR_RENORM` weights are
retained when supplied.

For unbinned output, when `MUERR_RENORM` is present, the probability and
bin-normalisation factors from BBC are held fixed. Writing
`r = MUERR_RENORM / MUERR`, the variance used is
`r² × (MUERR² − MUERR_VPEC²)`. This removes the velocity contribution with
the same rescaling as the other error terms; it does not refit BBC or
recompute its weights. If an external velocity covariance is expressed
before this rescaling, multiply its `(i, j)` element by `r_i r_j` before
supplying it.

For explicitly rejected SNANA keys, see {ref}`unsupported-covariance-keys`.
