# Bundled minimal test data

`python tests/fixtures/build_toy_data.py` regenerates the LSST SIMLIB variants,
synthetic HOSTLIBs, weight maps, flux-error model, dust maps, cropped V19 template,
and copied LSST calibration file (needs `$SNANA_LSST_ROOT` and `$SNDATA_ROOT`).
It does not rebuild the entire directory. Small authored configuration files,
filter curves and test tables are source inputs, rather than generated references.
Model distributions and trained classifier weights are copied upstream assets;
retraining a classifier is not part of fixture regeneration.

Provenance:

- `simlib/JAX_TEST_LSST.SIMLIB` — hand-crafted toy: 2 LIBIDs, 6 LSST
  filters, ~50 obs each over a 1-year MJD window. Sky/ZP/PSF values are
  synthetic (observation conditions are inherently abstract, no real-data
  semantics needed).
- `models/NON1ASED.V19_TEST/` — single template cropped from
  `$SNDATA_ROOT/models/NON1ASED/NON1ASED.V19_CC_noHostXT` (INDEX 834,
  SN2008fq IIn) by `crop.py` in that directory: every 4th phase and every 5th
  wavelength, so 64 × 1880 → 16 × 376 and 4.6 MB → 25 KB. Real flux values
  from V19 at the kept nodes — no interpolation, no synthetic-SED guesswork.
  `python crop.py --check` reports drift; a rerun is byte-identical. The grid
  is coarser than V19's so the tests exercise SNANA's NON1ASED interpolation,
  which also makes the template unfit for anything photometric.
- `models/NON1ASED.K10_TEST/` — 14 CC templates copied **verbatim** (no cropping)
  from `$SNDATA_ROOT/models/NON1ASED/NON1ASED.K10_CC` (Kessler & Jha 2010
  classification-challenge set), for the BBC BEAMS CC prior. Selection rule, applied
  to `NON1A.LIST` in file order: the first N of each subtype for
  `{IIN: 2, IIL: 1, Ibc: 1, Ib: 3, Ic: 3, IIP: 4}` — breadth over depth, because the
  prior's width comes from luminosity diversity across subtypes. That yields INDEX
  002, 021, 022, 101, 102, 103, 104, 105, 201, 204, 206, 208, 209, 210 (4.7 MB). The
  matching `NON1A:` weight lines in `snana/bbc_ccprior_sim/input.input` are that
  directory's `SIMGEN_INCLUDE_NON1A.INPUT` rows for those INDEXes, unchanged.
  Contrast `NON1ASED.V19_TEST` above, which is one template on a sub-sampled grid:
  this set is uncropped because the CC prior needs real light-curve shapes to fit.
- `kcor/JAX_TEST_DES.fits` — verbatim copy of
  `$SNDATA_ROOT/kcor/DES/DES-SN5YR/calib_DES-SN5YR_DES.fits.gz`, gunzipped
  (~470 KB). DES griz, for the `bbc_des_*` and `bbc_ccprior_sim` fixtures.
- `models/BAYESN.W22/` — verbatim copy of `$SNDATA_ROOT/models/bayesn/BAYESN.W22`.
- `models/SALT3.K21/` — verbatim copy of `$SNDATA_ROOT/models/SALT3/SALT3.K21`.
- `kcor/JAX_TEST_LSST.fits` — verbatim copy of
  `$SNANA_LSST_ROOT/kcor/baseline/kcor_LSST.fits` (~475 KB; binary
  multi-HDU FITS not feasible to hand-craft).
- `searcheff/SEARCHEFF_PIPELINE_*.DAT` — verbatim copies of LSST
  defaults from `$SNANA_LSST_ROOT/models/searcheff/`.

Tests reference these via `$JAX_TEST_FIXTURE_ROOT/...`; the top-level
`tests/conftest.py` sets that env var to this directory.

Numerical assets of at least 100 KB are stored as deterministic gzip, preserving
all original bytes. Model and calibration basenames above may therefore have a
`.gz` suffix on disk. After copying model files or running an individual asset
generator, run `python tests/fixtures/build_toy_data.py --compress`; the full
asset builder applies the same compression automatically.

## Checking copied and authored assets

`SHA256SUMS` pins the exact bundled asset bytes, including the compressed forms.
Check them with the standard checksum tool, from this directory:

```sh
shasum -a 256 -c SHA256SUMS
```

Copy the upstream assets from the locations listed above, then run
`python tests/fixtures/build_toy_data.py --compress` from the repository root.
Checksums detect a changed upstream distribution even when its directory name
has not changed. Updating a checksum is an explicit fixture update, not part of
ordinary regeneration. This is an integrity pin, not a claim that an upstream
service will always retain a particular version.

Additional upstream source locations:

- `classifiers/SNNTRAINV19_z_TRAINDES_V19`: `$SNDATA_ROOT/models/classifiers/DES-SN5YR/SNNTRAINV19_z_TRAINDES_V19`.
- `models/SIMSED.SNIa-91bg`: the SNANA/LSST PLASTICC model distribution,
  `$SNANA_LSST_ROOT/PLASTICC/model_libs_updates/SIMSED.SNIa-91bg`.
- `sndata_root/models/EAZY`: `$SNDATA_ROOT/models/EAZY` (the bundled FSPS templates).
- `sndata_root/models/BAO`: `$SNDATA_ROOT/models/BAO`.
- `sndata_root/models/OIR`: `$SNDATA_ROOT/models/OIR`.

The classifier's README records its training convention. Its frozen weights are
an input to inference tests; reproducing the training run is not required.

Legacy `SED.BINARY` formats are exercised with explicit two-SED binary records
constructed in `test_sed_binary_reads_either_snana_layout`. Expected phases,
wavelengths, fluxes, padding and rejected format versions are specified directly
in the test; no historical executable or memory contents are required.
