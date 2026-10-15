# Standardisation

jax-SNANA implements the `SALT2MU` standardisation code from SNANA. This is
applied to a sample of fitted supernovae to standardise them based on
properties of their light curves to estimate distances. `SALT2MU` uses the
standardisation method of [Marriner et al.
(2011)](https://arxiv.org/abs/1107.4631), an approach designed to be
insensitive to an assumed cosmological model by assigning supernovae into
redshift bins, applied within the BEAMS with Bias Corrections (BBC) method
presented in [Kessler & Scolnic (2017)](https://arxiv.org/abs/1610.04677).
In brief, this method is designed to correct inferred supernova distances
for selection bias and for contamination in photometrically-classified
samples.

{doc}`bias_correction_keys` lists the available input settings for this in
jax-SNANA.

## Running standardisation

Standardisation can be simply run using:

```bash
salt2mu bbc.input
```

with the input file containing all configuration. This input can be in the
same text format read by SNANA, or in YAML format. Either way, this file
must point to the data and bias-correction input FITRES files, containing
fitted light curve parameters and specify the grid on which to calculate
the bias correction. An example of a SALT standardisation input file is
given below, in the text format read by SNANA:

```text
model_lcfit=SALT2
datafile=data.FITRES
simfile_biascor=biascor.FITRES
nzbin=10
zmin=0.02
zmax=0.8
prefix=SALT_BBC
p1=0.14
u1=1
p2=3.1
u2=1
```

`p1` and `p2` set the starting values of α and β, respectively. `u1=1` and
`u2=1` fit these parameters; setting either flag to zero holds that
parameter at its supplied value. `prefix` simply sets the prefix for the
output files. The corresponding YAML input is:

```yaml
model_lcfit: SALT2
datafile: [data.FITRES]
simfile_biascor: [biascor.FITRES]
nzbin: 10
zmin: 0.02
zmax: 0.8
prefix: SALT_BBC
nuisance_val: {alpha0: 0.14, beta0: 3.1}
nuisance_float: {alpha0: true, beta0: true}
```

This fits α and β with starting values of 0.14 and 3.1. When a BIASCOR is
supplied, the default correction bins events in redshift, x1 and c. Each
individual survey in a sample receives its own bias correction grid.
`opt_biascor` can be used to configure more advanced bias-correction
options such as the parameters included in the grid; see below for more
detail on this. Without `simfile_biascor`, no bias correction is applied.

Settings in a SNANA input file can be overridden on the command line
using `key=value`:

```bash
salt2mu bbc.input prefix=BBC_RUN nzbin=10
```

(bbc-correction-options)=
{doc}`bias_correction_keys` lists the correction options, sample-selection
settings and survey-grouping controls. For an analysis pipeline, see
{doc}`cosmology_analysis`.

<!-- APPROVED TO HERE -->

### Running in a Python script

The same configuration can be supplied directly to `fit_distances` in a
Python script:

```python
from jax_salt2mu import fit_distances

result = fit_distances(
    datafile=["data.FITRES"],
    simfile_biascor=["biascor.FITRES"],
    model_lcfit="SALT2",
    nuisance_val={"alpha0": 0.14, "beta0": 3.1},
    nuisance_float={"alpha0": True, "beta0": True},
    nzbin=10, zmin=0.02, zmax=0.8,
)
hubble_diagram = result.hubble_diagram
```

To supply tables in memory, use `data` and `biascor` in place of the filename
settings. `result.fit` contains the fitted parameters and uncertainties,
and `result.output` contains the per-event results. Set `write=True` to
write output files. Further details are in the {doc}`Python API <api>`.

(bbc-selection-bias)=
## BIASCOR simulations

The bias corrections within the BBC method are calculated based on very
large simulations, much larger than the data sample being analysed,
designed to quantify the selection bias present in a survey across the full
space of observed supernova parameters. These are referred to as BIASCOR
simulations. These should reproduce the observed sample's observing
conditions, supernova population and selection, and be processed with the
same analysis pipeline. Even with a large BIASCOR simulation, some events
are excluded because there are not enough simulated events with similar
parameters to estimate a reliable bias correction. This loss is reported by
the `BCOR_LOSS` output.

## Standardisation parameters

A host-mass step and redshift or host-mass dependence of the standardisation
coefficients can also be fitted. {doc}`bias_correction_keys` lists these
parameters, their names in YAML and their numbering in SNANA inputs.

## Intrinsic scatter

`sig1` specifies a fixed intrinsic scatter in magnitudes, added in
quadrature to the distance uncertainties. It defaults to zero.
Alternatively, set `fitflag_sigmb=1` to estimate the scatter by iterating
until the reduced χ² is approximately one. jax-SNANA also supports
`sigint_mode=joint`, which fits the intrinsic scatter jointly with the
other parameters and returns an uncertainty on it. This option is not
available in SNANA.

(bbc-contamination)=
## Contamination

For photometrically classified samples, BBC also incorporates the BEAMS
method ([Kunz et al. 2007](https://arxiv.org/abs/astro-ph/0611004)) to
account for contamination from other transients. jax-SNANA supports two
approaches to modelling contamination: a simulation of core-collapse
supernovae as used in [Vincenzi et al.
(2023)](https://arxiv.org/abs/2111.10382) and [Vincenzi et al.
(2024)](https://arxiv.org/abs/2401.02945) or the analytic model of [Hlozek
et al. (2012)](https://arxiv.org/abs/1111.5328), referred to as H11. For
the simulation-based approach, the distribution of contaminants is
estimated from a simulation of core-collapse supernovae, which are all
fitted with the same model as used for the rest of the analysis. The prior
on the contaminant population is constructed from the fitted distance
residuals of the core-collapse sample.

The likelihood in BEAMS includes a term related to the probability of each
event being a type Ia supernova, typically from a photometric
classification algorithm (see {doc}`classification` for details on
classification in jax-SNANA). Set `simfile_ccprior` to the core-collapse
simulation's FITRES file and `varname_pIa` to the data column containing
type Ia classification probabilities. For example, if SuperNNova
probabilities are stored in `PROB_SNN`:

```text
varname_pIa=PROB_SNN
simfile_ccprior=ccprior.FITRES
u13=1
```

`u13=1` fits `scalePCC`, which adjusts the contaminant contribution to allow
for imperfectly calibrated classification probabilities. It can instead be
fixed at a non-zero value using `p13` and `u13=0`.

To use H11, replace `simfile_ccprior=ccprior.FITRES` with
`simfile_ccprior=H11`. This fits a Gaussian residual distribution whose mean
and width are quadratic functions of redshift, without requiring a
core-collapse simulation.

For samples that include spectroscopically confirmed type Ia supernovae,
set `typeIa_ccprior` to their `TYPE` code. These events are assigned a
type Ia probability of one, irrespective of their classification
probabilities.

## Output

(bbc-output-m0dif)=
The main outputs are:

- `.M0DIF`: the binned Hubble diagram. `MUDIF` gives the distance-modulus
  offset from the fixed reference cosmology, `MUREF`, with uncertainty
  `MUDIFERR`. The header contains the fitted standardisation parameters.
- `.FITRES`: the unbinned Hubble diagram, with one row per supernova,
  including corrected `MU` and `MUERR`, and the residual `MURES`. With a
  contamination model, `PROBCC_BEAMS` gives the posterior contaminant
  probability.

The {doc}`analysis_outputs` page describes these tables. Either the binned
or unbinned Hubble diagram can be used for {ref}`covariance-construction` and
{doc}`cosmology`.

## Blinding

By default (`blindflag=2`), the reference cosmology is blinded, concealing
the cosmological result while analysis choices are being made. This affects
the reported distances but does not blind the fitted standardisation
coefficients. Set `blindflag=0` to turn blinding off. Simulated samples are
not blinded by default. Both `.M0DIF` and `.FITRES` headers include a warning when blinding is
applied.
