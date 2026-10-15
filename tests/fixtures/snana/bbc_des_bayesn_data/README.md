# Fixture: bbc_des_bayesn_data

BayeSN W22 on the DES cadence, used as input to BBC parity tests. SNANA
simulates the light curves, the JAX fitter produces `FITRES.TEXT.gz`, and
`refs.py` runs `SALT2mu.exe` to produce independent BBC reference answers.
These are BBC tests; the fitted input tables are not fitter reference answers.

## Generation

Refreshed 2026-10-14 using SNANA `v12_02d-12-g74dfb2c5` and the JAX fitter
at `e010076a` with this change's `MUERR_LCFIT` output-column correction.
The namelists specify three fit iterations, double precision, batch size 32,
and physical-parameter-only FITRES output. Simulation uses
`GENMODEL_MSKOPT: 64` (smooth-to-constant time extrapolation), matching the
fitter's default. The fitter applies the model's filter-centre wavelength
limits as well as `RESTLAMBDA_FITRANGE`.

With the pinned SNANA executables on `PATH` and `SNDATA_ROOT` configured,
run from the repository root:

```sh
python tests/fixtures/regenerate.py --only bbc_des_bayesn_data
python tests/fixtures/regenerate.py --only bbc_des_bayesn_biascor
python tests/fixtures/snana/bbc_des_bayesn_data/refs.py
```

The generator runs simulation and fitting, setting `JAX_TEST_FIXTURE_ROOT`
and `JAX_TEST_FIT_SIM_DIR` itself. HEAD/PHOT are used for fitting but are not
bundled. Tables above 100 KB use deterministic gzip.

To verify regeneration without replacing the recordings:

```sh
python tests/fixtures/regenerate.py --diff --only bbc_des_bayesn_data --keep bbc-bayesn
python tests/fixtures/regenerate.py --diff --only bbc_des_bayesn_biascor --keep bbc-bayesn
python tests/fixtures/snana/bbc_des_bayesn_data/refs.py --check
```

The first two commands compare generated table values and retain fit logs.
Numerical-library or hardware changes can affect fitted values; inspect such
differences before refreshing references. `refs.py --check` independently
checks the recorded BBC answers using the bundled fitted inputs.

The BayeSN outputs in `../bbc_output_contract` also depend on these tables;
regenerate them following that directory's README when refreshing the inputs.
