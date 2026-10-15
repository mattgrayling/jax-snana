# Specifying the simulated population

jax-SNANA allows population distributions for parameters such as light curve
shape, colour and host extinction to be specified in the simulation input
file. These distributions can vary with redshift or host-galaxy properties,
and correlations between parameters can also be included. Multiple
populations with different parameter distributions can be simulated
together using `SUBPOP`.

(quickstart-bayesn-population)=
## BayeSN example population

The BayeSN example in {doc}`getting_started` uses the following model and
population settings:

```text
GENMODEL: jax_snlc_core/data/models/BAYESN.G26
GENMODEL_MSKOPT: 1
GENPEAK_THETA: 0.0
GENSIGMA_THETA: 1.0 1.0
GENRANGE_THETA: -3.0 3.0
GENRANGE_AV: 0.0 1.0
GENTAU_AV: 0.1716
GENPEAK_RV: 1.8347
GENSIGMA_RV: 0.2300 0.2300
GENRANGE_RV: 1.2 6.0
```

`THETA` controls intrinsic light-curve shape. Here it is drawn from a
unit-width Gaussian truncated to [-3, 3]. Host extinction `AV` is drawn
from an exponential with scale 0.1716 mag, truncated at 1 mag. `RV` follows
a Gaussian with mean 1.8347 and width 0.2300, truncated to [1.2, 6.0].
These dust settings use the values in the G26 training; the
simulator requires the host-dust population in the input rather than taking
it from the trained model file.

`GENMODEL_MSKOPT=1` enables the correlated intrinsic residuals and the grey
offset `DELTAM`, using the covariance and scatter amplitude in the trained
model. With zero, neither is drawn.

## Population parameters

Provide distributions for the selected model's population parameters,
listed in {doc}`model_settings`. The settings below specify their profiles;
`GENPDF_FILE` can instead supply tabulated probability densities.

For a parameter `<PAR>`, `GENPEAK_<PAR>` sets the peak and
`GENSIGMA_<PAR>` gives the Gaussian widths below and above it, in that
order. Equal widths give a symmetric Gaussian. `GENRANGE_<PAR>` sets the
lower and upper limits of the distribution. For an exponential host
extinction distribution, `GENTAU_<PAR>` sets the exponential scale, as
shown for `AV` above.

### Specifying distributions in YAML

In YAML, `populations` maps parameter names to distributions. The BayeSN
shape and dust distributions above can be written as:

```yaml
populations:
  THETA: {distribution: normal, peak: 0, sigma: 1, range: [-3, 3]}
  AV: {distribution: exponential, exp_tau: 0.1716, range: [0, 1]}
  RV: {distribution: normal, peak: 1.8347, sigma: 0.2300, range: [1.2, 6]}
```

Every distribution needs a range; a zero-width range fixes a parameter.

### Distribution profiles

In YAML, `normal` takes a peak and one `sigma`; `asymmetric_normal` takes
`[sigma_left, sigma_right]` for different widths on either side of the peak.
The asymmetric profile can also have a flat peak, a skewed shape or a second
Gaussian component. The corresponding settings are described under
{ref}`sim-population-settings`.

For a uniform distribution, use only the distribution name and bounds,
for example `THETA: {distribution: uniform, range: [-3, 3]}`.

For `AV` or `EBV_HOST`, use `exponential`, with scale `exp_tau`, or
`exponential_gaussian`, which adds a Gaussian or half-Gaussian component
with relative strength set by `ratio`. A peak at zero selects a positive
half-Gaussian; a positive peak selects a Gaussian truncated to the specified
range. `sigma` sets its width. Omitting the distribution
name selects the asymmetric normal for model parameters and the
exponential/Gaussian profile for host extinction. The full distribution
settings are listed under {ref}`sim-population-settings`.

(population-scatter)=
## Intrinsic scatter

BayeSN supplies correlated intrinsic residuals and grey scatter through
its trained model; `GENMODEL_MSKOPT=1` enables both, as shown above.
Additional scatter can be configured with:

```text
GENMAG_SMEAR: 0.1
GENMAG_SMEAR_FILTER: gr 0.03
```

`GENMAG_SMEAR` draws a Gaussian magnitude offset for each event, with the
specified width in magnitudes, applied to every band and epoch.
`GENMAG_SMEAR_FILTER` draws an independent offset for each selected filter,
constant across that event's observations in the filter. This example
uses width 0.03 mag in each of *g* and *r*.

`GENMAG_SMEAR_MODELNAME` selects a wavelength-dependent scatter model.
`G10` uses the SALT colour-dispersion model; the `C11` options use the
correlated scatter of Chotard et al. (2011). Only G10 requires SALT model
files.

### Colour and phase dependence

To vary the amplitude of a named scatter model with a drawn parameter,
use a polynomial scale:

```text
GENMAG_SMEAR_MODELNAME: G10
GENMAG_SMEAR_SCALE(SALT2c): 1.0,2.0
```

This multiplies the G10 magnitude perturbations by $1+2c$. Coefficients
start at the constant term. It replaces the constant
`GENMAG_SMEAR_SCALE` and applies only to the named scatter model.

For a phase-correlated grey perturbation, use:

```text
GENMAG_SMEAR_ADDPHASECOR: 0.03 10.0
```

The values specify the scatter width in magnitudes and the correlation
time in rest-frame days. The perturbation is common to all bands at a
given phase and is added to other configured scatter.

`examples/quickstart/sim_scatter.input` combines the colour-dependent G10
scale and phase-correlated scatter.

(simulating-evolution)=
## Population evolution with redshift

`ZVARIATION_POLY` changes a population distribution with cosmological
redshift. For example, add these lines to the SALT quick-start input,
`examples/quickstart/sim.input`, to increase the peak and both widths of
the stretch distribution:

```text
ZVARIATION_POLY: GENPEAK_SALT2x1 0.0,1.0
ZVARIATION_POLY: GENSIGMA[0]_SALT2x1 0.0,0.2
ZVARIATION_POLY: GENSIGMA[1]_SALT2x1 0.0,0.2
```

The comma-separated coefficients give an additive shift, starting with the
constant term. Here the peak increases by $z$ and each width by $0.2z$.
`[0]` selects the width below
the peak and `[1]` the width above it. The parameter is drawn from this
modified distribution, still within its configured `GENRANGE`.

Polynomials can also be supplied as `ZPOLY:` rows in a file named by
`ZVARIATION_FILE`.

The shift must be zero at redshift zero, so the first coefficient must be
zero. A brightness evolution can instead target
`GENMAG_OFF_GLOBAL`, with a positive shift making events fainter.

The full input is `examples/quickstart/sim_evolution.input`.

(simulating-variation)=
## Correlations with host properties and model parameters

`VARIATION` makes a distribution depend on a host property or on another
model parameter. For example, these lines give stretch a lower peak in
more massive hosts:

```text
HOSTLIB_FILE: tests/fixtures/data/hostlib/JAX_TEST.HOSTLIB
GENRANGE_REDSHIFT: 0.05 0.4
VARIATION: GENPEAK_SALT2x1 LOGMASS POLY -0.30 PIVOT 10.0
```

The host catalogue supplies `LOGMASS`, the logarithm of stellar mass in solar
units. The shift is $-0.30(\mathrm{LOGMASS}-10)$, added to the configured
stretch peak before drawing stretch. The redshift range matches the bundled
catalogue; see {ref}`simulating-hosts` for host-catalogue setup.

The syntax is `VARIATION: <target> <quantity> POLY <coefficients> PIVOT
<value>`. Coefficients are space-separated and start at the **linear** term:
`POLY a b PIVOT p` gives $a(q-p)+b(q-p)^2$. The target is a distribution
component such as `GENPEAK_SALT2x1`, or an applied offset such as
`GENMAG_OFF_GLOBAL`. The driving quantity is a model population parameter
or a column of the configured HOSTLIB. `VARIATION` is a jax-snana extension.

A drawn model parameter can drive another distribution without a host
catalogue. For example:

```text
VARIATION: GENPEAK_SALT2ALPHA SALT2x1 POLY -0.01 PIVOT 0.0
```

With the quick start's baseline `SALT2ALPHA=0.14`, this gives
$\alpha=0.14-0.01x_1$. A parameter-dependent distribution can use
parameters drawn earlier by the model, as stretch is before alpha in SALT.
If several variations affect the same target, their shifts add, including any
`ZVARIATION` shift. A parameter drawn from a `GENPDF_FILE` cannot also
be shifted by `VARIATION` or `ZVARIATION`.

The host-mass example is provided in
`examples/quickstart/sim_host_variation.input`.

### Using host catalogue properties

A HOSTLIB can also be used to specify conditional parameter distributions,
host-selection weights or parameter values directly:

- `GENPDF_FILE` supplies a tabulated probability density, optionally
  conditioned on host properties such as stellar mass. The file may also
  specify `GENMAG_SMEAR` and `MAG_OFFSET`. If it specifies `GENMAG_SMEAR`,
  omit that setting from the simulation input. The file format is described
  in {doc}`simulation_inputs`.
- A HOSTLIB weight map specifies the relative probability of selecting
  each galaxy as a host.
- With `HOSTLIB_MSKOPT` bit 64, catalogue columns can supply population
  parameter values, including host dust. See
  {ref}`HOSTLIB options <sim-host-options>`.

(simulating-subpop)=
## Simulating two populations with SUBPOP

`SUBPOP` allows two or more populations with different parameter
distributions to be simulated together. Each event is assigned to a
population according to its relative weight, then its parameters are drawn
from that population's distributions. This can associate a particular
stretch distribution with a particular colour distribution, whereas adding
a second Gaussian component to an individual parameter does not correlate
it with other parameters. `SUBPOP` is a jax-SNANA extension.

The following input defines two SALT populations. It uses the model and
survey inputs from the {doc}`quick start <getting_started>` and is provided
as `examples/quickstart/sim_subpop.input`:

```text
INPUT_FILE_INCLUDE: examples/quickstart/sim.input
GENVERSION: QUICKSTART_SUBPOP
NGENTOT_LC: 1000
RANSEED: 38177

# Shared distribution widths and ranges; each class sets its own peaks.
GENSIGMA_SALT2x1: 0.7 0.7
GENRANGE_SALT2x1: -3.0 3.0
GENSIGMA_SALT2c: 0.04 0.04
GENRANGE_SALT2c: -0.2 0.3

SUBPOP: A WGT 3
SUBPOP: A GENPEAK_SALT2x1 0.5
SUBPOP: A GENPEAK_SALT2c -0.03

SUBPOP: B WGT 2
SUBPOP: B GENPEAK_SALT2x1 -1.0
SUBPOP: B GENPEAK_SALT2c 0.06
SUBPOP: B GENMAG_OFF_GLOBAL 0.06
```

Run it from the repository root, after completing the quick-start setup:

```bash
snlc_sim examples/quickstart/sim_subpop.input
```

Each line has the form `SUBPOP: <class> <key> <values>`. Class names are your
own labels. Every class needs a `WGT`, giving its relative weight in the
generated population.
Detection and selection can change these fractions in the output sample.

Settings without a `SUBPOP` prefix are shared unless a class overrides them.
Here, both classes use the same widths and ranges for stretch and colour,
but have different peaks. They also inherit the quick start's fixed
`SALT2ALPHA` and `SALT2BETA`. Class B has an additional grey magnitude offset
of +0.06 mag, making it fainter at fixed model parameters.

You can override the widths and ranges in the same way as the peaks, for
example `SUBPOP: B GENSIGMA_SALT2x1 0.5 0.8`. Each parameter configured for
one class must also have a distribution in every other class, either from
shared settings or from explicit class settings. A parameter drawn from a
`GENPDF_FILE` cannot also be overridden by `SUBPOP`. The source model,
survey inputs, redshift range and event count remain shared run settings.

`SIM_SUBPOP` in the output HEAD FITS table records A as 0 and B as 1,
following the order in which their names appear in the input. The
`.README` records the class names, numbers and input weights.

### YAML

The same two populations can be specified through `subpops`, keeping the
model, survey and shared alpha and beta settings above:

```yaml
subpops:
  - name: A
    weight: 3
    populations:
      SALT2x1: {distribution: normal, peak: 0.5, sigma: 0.7, range: [-3, 3]}
      SALT2c: {distribution: normal, peak: -0.03, sigma: 0.04, range: [-0.2, 0.3]}
  - name: B
    weight: 2
    genmag_off_global: 0.06
    populations:
      SALT2x1: {distribution: normal, peak: -1, sigma: 0.7, range: [-3, 3]}
      SALT2c: {distribution: normal, peak: 0.06, sigma: 0.04, range: [-0.2, 0.3]}
```

Each YAML distribution supplies its full profile, including widths and
range. Parameters omitted from a class use their shared distributions;
an omitted `genmag_off_global` uses the shared magnitude offset.

### Host-dependent population fractions

A class's relative weight can also depend on a host property. Add the
following to the two-population input above:

```text
HOSTLIB_FILE: tests/fixtures/data/hostlib/JAX_TEST.HOSTLIB
GENRANGE_REDSHIFT: 0.05 0.4
SUBPOP: A VARIATION WGT LOGMASS POLY -0.5 PIVOT 10.0
```

Class A now has weight $3-0.5(\mathrm{LOGMASS}-10)$, while B keeps weight 2.
The simulator normalises the relative weights for each event before
drawing its class.

You can also put a `VARIATION` on a distribution component inside a class,
for example `SUBPOP: B VARIATION GENPEAK_SALT2x1 LOGMASS POLY -0.3 PIVOT
10.0`. This changes B's stretch distribution with host mass while retaining
A's settings. The polynomial and pivot follow the
{ref}`VARIATION syntax <simulating-variation>`.

(population-reweighting)=
## Population reweighting

`GENPDF_EXPON_REWGT` allows rare parts of the configured population to be
sampled more frequently, without requiring a much larger simulation.
For example, `GENPDF_EXPON_REWGT: 0.25` doubles Gaussian widths, generating
more events away from their peaks while retaining the configured ranges.

The setting gives an exponent `e`: Gaussian widths are divided by
$\sqrt{e}$, exponential scales by `e`, and tabulated probability densities
are raised to the power `e`. Values below one broaden the distributions;
one leaves them unchanged.

The output column `SIM_WGT_POPULATION` gives each event a relative weight
for the original population, reducing the contribution of events that
were sampled more frequently. It is calculated as the ratio of the
original to the modified distribution profiles, multiplied over the
sampled parameters. These profiles are not normalised, so the weights
specify relative contributions rather than absolute event counts.
The weights do not account for distribution shifts from `VARIATION` or
`ZVARIATION`; use them only with distributions that do not have these shifts.
