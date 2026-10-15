# bbc_output_contract

`SALT2mu.exe` output files for six configurations, so `test_bbc_output_contract.py`
can compare jax's **files** rather than the values inside them.

Inputs are not duplicated here: each config reads `bbc_biascor_fit`,
`bbc_photoz` or `bbc_des_bayesn_data`/`bbc_des_bayesn_biascor`. Only SNANA's outputs live in this directory.

## Why six

The audit that produced these found eight defects. Only two were visible on the 5D
SALT config; the rest needed a different mode:

| reference | exercises | defect it would have caught |
| --- | --- | --- |
| `salt_5d` | the common path | wrong `.YAML` keys and counts, passthrough precision, column formats |
| `salt_nobiascor` | no biasCor at all | `NEVT_REJECT_CUTWIN` (SNANA sums it over biasCor samples, so it reports 0 with none) |
| `salt_ccprior` | a live BEAMS CC prior | `NEVT_CCPRIOR` reported 0 -- SNANA cuts the CC sample on missing biasCor; and `.CHI2GRID` scanning the Ia chi2 where SNANA scans `fcn` |
| `salt_blinded` | blinding active | the "RESULTS ARE BLINDED" block missing from `.FITRES` |
| `salt_1d_split` | 1D bias, two IDSAMPLEs | the by-sample block absent, and `NSIM`/`sigint` absent and labelled `5D` |
| `bayesn` | MU-biasCor | an extra `biasCor_MU` column, and the alpha/beta note SNANA omits for BayeSN |

Two of those defects were introduced *while* fixing the others, and matched the 5D
reference by coincidence. A single-config contract test cannot catch that.

`salt_ccprior` carries `typeIa_ccprior=999` deliberately. `CONTAM_DATA.FITRES` has no
`TYPE` column, so it reads -9 == the key's own default, `force_probcc0` (`SALT2mu.c:21852`)
fires on every event and pins `pIa` to 1 -- both programs agree, but the CC term is
multiplied by zero and the config tests the BEAMS code path without testing BEAMS.

## Regenerating

The `CONFIGS` table in `test_bbc_output_contract.py` is the single source of the
configurations. Do **not** transcribe them here -- a hand-written copy is how the
first version of these references ended up generated at a different `blindflag`
from the one the test used, which looked like a jax defect.

Run from the repository root with the pinned `SALT2mu.exe` on PATH:

```sh
python tests/fixtures/snana/bbc_output_contract/refs.py
python tests/fixtures/snana/bbc_output_contract/refs.py --check
```

The script imports the test's `CONFIGS` and `CHI2GRID_NBIN`, generates the photo-z
bias-correction variant, and stages compressed tables as plain text for SNANA.
`--check` compares every output, excluding wall-clock timings, and writes nothing.
The BayeSN inputs are produced by their separate simulation-and-fitting recipes.

## What is not compared

`SNANA_VERSION` and `PROGRAM` (each program's build string and name), the error-method line (SNANA runs
MINUIT and prints `MIGRAD`; jax takes the Hessian from Newton-LM and prints
`HESSIAN`), `NCALL_FCN` (a MINUIT call count with no jax equivalent), and `CPU` /
`CPU_MINUTES` (measurements). Values inside the fit-results block are covered by
the per-feature parity tests; this file checks the files' shape.

The BayeSN case separately checks JAX's five `SIM_*(BIASCOR)` cosmology
provenance fields, which SALT2mu does not write. All remaining header lines
are compared with the SNANA reference.

Store generated `.FITRES` references above 100 KB with deterministic gzip
(`gzip.compress(content, compresslevel=6, mtime=0)`). Keep smaller outputs plain.
