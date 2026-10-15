# JAX WFIT

`wfit.py` is a JAX/NumPyro rewrite of SNANA `wfit.exe`. It is an indicative
flat-cosmology fitter for binned or unbinned Hubble diagrams, with one common
likelihood and three inference backends:

- `laplace` (default): NumPyro `AutoLaplaceApproximation`;
- `nuts`: NumPyro NUTS with convergence diagnostics;
- `grid`: chunked JAX evaluation using SNANA's default parameter axes.

WFIT does not perform SALT2mu/BBC binning. It consumes the resulting M0DIF, or
an unbinned FITRES containing redshift, distance modulus, and uncertainty.

The other stages read the frame from an input key; wfit, having no input file,
takes `-double_precision`, defaulting to 0. `-double_precision 1` runs the
cosmology chain in double precision.

`run_wfit` returns one result dict per realisation, keyed by `method`,
`point_estimate`, `summaries`, `covariance`, `correlation`, `likelihood`,
`ndof_hd`, `n_external`, `prior_targets`, `diagnostics`, `warnings`,
`blinded` and `samples`. `load_hubble_diagrams` returns one `HubbleDiagram`
per input file.

## Running

Run from the SNANA inputs/output workspace:

```bash
cd ~/Documents/SNANA_inputs
wfit \
    SALT2mu.M0DIF --method laplace -cospar_yaml SALT2mu.WFIT.yaml
```

Grid parity and NUTS modes are explicit:

```bash
wfit \
    SALT2mu.M0DIF --method grid -wa -cmb \
    -outfile_chi2grid SALT2mu.WFIT_GRID.FITRES

wfit \
    SALT2mu.M0DIF --method nuts \
    --output-samples SALT2mu.WFIT_SAMPLES.npz
```

NUTS defaults to four independently adapted chains, 1,000 warmup steps, and
1,000 retained samples per chain. Runs with divergences or excessive R-hat
are still written with `success: false`; the default maximum R-hat is 1.05.
The configurable effective-sample-size
threshold is advisory: values below it are reported as warnings but do not
fail an otherwise healthy indicative fit.
The broad SN-only `w0wa` validation posterior required
`--target-accept 0.99 --nuts-warmup 2000 --nuts-samples 4000` to remove all
divergences; with those settings it reached maximum R-hat 1.0017, minimum ESS
1528, and posterior means within 0.005 grid standard deviations. The faster
defaults correctly marked that difficult case as unconverged.

The default summary is `<HD>.wfit.yaml`. Residual, cospar, grid, marginalized
probability, sample, covariance, and precision files are opt-in.

## Multiple realisations

One invocation can fit multiple independent simulated Hubble diagrams with a
shared cosmology, prior, and inference configuration:

```bash
wfit \
    SIM_00.M0DIF SIM_01.M0DIF SIM_02.M0DIF \
    --method nuts -cospar_yaml bias_test.WFIT.yaml
```

The equivalent YAML input is a list:

```yaml
hubble_diagram:
  - SIM_00.M0DIF
  - SIM_01.M0DIF
  - SIM_02.M0DIF
```

All internal arrays retain a leading realisation axis, including the
one-realisation case. The grid evaluates all realisations natively, while
AutoLaplace and NUTS are vectorized over the axis with `jax.vmap`: each
realisation runs its own NumPyro `MCMC` with `chain_method="vectorized"`, so
every realisation and chain has its own random key, step-size adaptation and
dense mass matrix. Likelihoods are not combined into a joint posterior.

Realisations may retain different numbers of rows: shorter ones are padded to
the longest and masked out of every likelihood, so degrees of freedom stay
per-realisation. Every realisation must share one covariance representation,
and covariance matrices must be supplied for all realisations or none. An
external covariance path may be shared, or YAML may provide one covariance
path per Hubble diagram. The summary YAML contains a `realisations` list.
Per-realisation residual, COSPAR, grid, probability, and sample outputs
receive `.r000`, `.r001`, and so on before their file extension.

When the simulation truth is known for every realisation, the summary also
carries an `ensemble` block reporting mean bias, bias scatter and standard
error, pull mean and scatter, and 68-percent coverage for each fitted
parameter. Truth comes from `--truth-info` (one `INFO.YML` per realisation)
or from `-om_sim`, `-w0_sim`, and `-wa_sim`. The block is suppressed when any
realisation is blinded.

## Structured configuration

CLI values override YAML. The resolved configuration is written into the result.

```yaml
hubble_diagram: SALT2mu.M0DIF
covariance: SALT2mu.COV
covariance_kind: total
zmin: 0.02
zmax: 1.2

model: wcdm
ommin: 0.0
ommax: 1.0
wmin: -2.0
wmax: 0.0

ompri_enabled: true
ompri: 0.315
dompri: 0.03
cmb: true
Rcmb: 1.710
sigma_Rcmb: 0.019

method: laplace
seed: 12345

cospar_yaml: SALT2mu.WFIT.yaml
outfile_resid: SALT2mu.WFIT_RESID.FITRES
```

Keys are flat and named after the SNANA flags, so a YAML key, a command-line
option and a config attribute are spelled the same. An unknown key aborts.

The CLI uses SNANA's spellings, and only invents a name where SNANA has none
(`--method`, `--config`, `--hdibc`, `--nuts-*`, `--seed`, `--truth-info`).
`-bao` replaces the Gaussian Omega-m prior and `-cmb` applies WFIT's weak
`sigma(Omega_m)=0.9` preset, both exactly as WFIT does. Composing priors
independently of those presets is done in YAML.

## Covariance semantics

Covariance meaning must be declared; it is never inferred from a filename:

- `systematics`: WFIT adds `diag(MUERR² + snrms²)`.
- `total`: WFIT uses the supplied covariance instead of adding `MUERR²`.
  SALT2mu's directly written `.COV` is this kind.
- `total_inverse`: WFIT uses an already-inverted total covariance.

Row-removing cuts and error changes cannot be applied safely to a total inverse
covariance and are rejected. Text covariance format is the SNANA dimension
followed by row-major values. NPZ accepts a `cov` array or a single array.

## Blinding

`-blind` offsets reported cosmology estimates by `sin(integer)` per free
parameter, and `-blind_auto` blinds only when the Hubble-diagram header says
the data are real. The integers reproduce `wfit.c:get_final_values`, which
seeds libc `rand` and draws in free-parameter order; the Park--Miller sequence
is reimplemented directly so the process-global random state is untouched. The
blinding seed is redacted from blinded summaries, and the ensemble block is
withheld.

## Bias-corrected two-HD interpolation (HDIBC)

`--hdibc` takes two Hubble diagrams — either as two paths or SNANA's
comma-separated form — plus `--hdibc-info INFO0 INFO1` giving the bias-
correction cosmology of each. Events are matched by CID, and the observed
distance modulus is interpolated between the two bias cosmologies. One or two
covariance files may be supplied. Redshift synchronization follows SNANA
`sync_HD_redshifts`, which skips diagrams of more than 100 rows.

## Idealized data and simulated priors

`-muerr_ideal` replaces the Hubble diagram with data generated at the
simulation truth, using either a redshift polynomial or the input
uncertainties. `-cmb_sim` and `-bao_sim` re-center the CMB and BAO priors on
that truth instead of their published means. `--perturb-priors` offsets the
Omega-m and CMB prior centers by an independent Gaussian draw per
realisation, for ensemble studies of how prior scatter propagates into
recovered cosmology.

## Physics and compatibility

The fit is flat ΛCDM, wCDM, or CPL `w0wa`. A global magnitude offset is
profiled analytically using `wfit.c:get_chi2_fit`, including WFIT's very weak
offset prior. Gaussian Omega-m, WMAP shift-parameter, and DESI 2024/SDSS-IV
2020 BAO constraints are supported.

The shared cosmology includes photon + neutrino radiation by default, using
SNANA wfit's `Omega_r = 0.9e-4`. Matter and dark energy are multiplied by
`1 - Omega_r`, with curvature computed before rescaling. The simulator and
BBC use this same default, unlike their SNANA counterparts.

WFIT follows SNANA's prior convention: CMB R (both the simulation-centred
target and trial values) and BAO `D_H` explicitly omit radiation. SN and BAO
`D_M` distances include it. Set `radiation: false` in WFIT YAML (or
`radiation=False` in Python) to omit it from those distances too. The distance
and prior conventions are recorded in output. Invalid expansion rates are
rejected rather than clamped to SNANA wfit's `E² = 0.01`.

Non-flat HDIBC bias-correction cosmologies apply the physical transverse-
curvature transform, which `wfit.c:get_mu_cos` omits. Flat cosmologies are
unaffected.

## Deferred

Intrinsic-scatter refitting (`-refit`) and WFIT's likelihood-changing speed
approximations are not implemented. The refit calculation has effectively been
removed from current SNANA — `sigmu_int` is set to zero and the function
returns — so implementing it would mean designing new behaviour rather than
reproducing working WFIT behaviour.

## Validation

Run the test suites from the repository root. The focused regression set is:

```bash
python -m pytest tests/unit/test_wfit_*.py
python -m pytest \
    tests/integration/test_wfit_backends.py
RUN_LIVE_SNANA=1 python -m pytest \
    tests/live_snana/test_wfit_parity.py
```

The live suite compares the exact grid to the current SNANA executable for
flat Lambda-CDM, wCDM, and w0wa; no external prior, CMB, DESI-2024 BAO, and
SDSS-IV-2020 BAO; diagonal, systematic, and total-inverse covariance;
redshift cuts, forced errors, and diagonal `snrms`; a 315-row unbinned Hubble
diagram; residual and inverted-covariance output; idealized data from a
redshift polynomial; simulation-centered priors; explicit, `w0wa`-sequence,
and automatic blinding; HDIBC across models, covariance counts, CID
synchronization, and unbinned redshifts; and padded multi-realisation
ensembles. A separate read-only run on a 315-SN SALT2mu FITRES from the
parallel SALT2mu worktree agreed in retained rows and HD dof, with
posterior-mean shifts below `1.1e-4` and 68.3-percent uncertainty shifts
below `1e-4`.

The backend integration test compares Laplace and NUTS with the exact grid.
For a CMB-constrained posterior, NUTS means agree within 0.08 grid standard
deviations and Laplace means within 0.15. On the broader SN-only posterior,
the default NUTS run must pass zero-divergence and R-hat `<1.05` checks, with
means within three ESS-derived Monte Carlo standard errors of the exact grid
means. Its effective sample sizes are reported and checked to trigger the
advisory warning below 400.

The repository-wide unit suite itself must be launched from the repository
root: the pre-existing `tests/unit/test_config_parser_helpers.py` resolves
`tests/fixtures/snana/*/input.input` relative to the current working directory
during test collection. Running that suite from `SNANA_inputs` therefore fails
before any WFIT test runs; this is a test-location assumption, not a WFIT
failure.
