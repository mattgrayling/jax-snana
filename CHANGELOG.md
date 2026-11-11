# Changelog

## Unreleased

- Validated against SNANA v12_02f.
- Host aperture magnitudes use SNANA's corrected Sersic normalisation and its wider PSF grid.
- Simulations write the `SEASON_INDEX` column and the TEXT `FIELD:` line, size host photo-z quantile vectors by their number of quantiles, and set `HOSTGAL_NMATCH` to -9 without a HOSTLIB.
- Fitter tables use SNANA's new header layout and carry `HOST_FLAG` and `HOST2_FLAG`.
- SALT2mu's intrinsic-scatter scan follows SNANA's new rules.
- `PRIOR_ZERRSCALE` defaults to 1 for quantile photo-z fits.
- create_covariance writes the factorized covariance by default and always lists the factorized file of each covariance option in `INFO.YML`.
- With only the factorized covariance requested, create_covariance writes nothing for a covariance option that includes an external matrix, and an input whose `VERSION_PHOTOMETRY` line has no value is an error.
- wfit writes no `-outfile_mucovtot_inv` file for a factorized covariance.
- Pippin's `USE_COVTOT_FACTORIZED` and `USE_COVSYS_INV` choose the covariance file each cosmology fit reads, which may be the systematic covariance.
- A simulation stage gives each job the CID offset and range SNANA's batch layer does, so a large version no longer runs out of random CIDs; Pippin's `RESET_CIDOFF` is read, and an input that sets only `NGEN_LC` is refused.
- The readers of SNANA's files accept every layout SNANA's own readers accept, including keys sharing a line, comments after values, the alternative column spellings, `PRIVATE(name)` variables, `PSF_NEA` photometry and kcor filter names of any length.
- `HEADER_OVERRIDE_FILE` adds columns the data lack and supports `NULL_ON_MISSING_EVENT`; `EPOCH_IGNORE_FILE` is read for each data version.
- A SIMLIB's `GENSKIP_PEAKMJD` ranges and `PHOTFLAG_SNRMAX` and `PHOTFLAG_NEARPEAK` masks are applied.
- A HOSTLIB may give its redshifts as `ZTRUE_CMB`, which is converted to heliocentric as it is read.
- FITS data written before SNANA recorded host matches give each event one matched host, and fits reading `VERSION_PHOTOMETRY_WILDCARD` name the first matched version in their table headers.
- `HOSTLIB_FIELD_MAP` is refused, and a HOSTLIB with no galaxy inside the cuts is an error.

## 0.1.1

- Restructured the package to make things easier to find. Internal import paths have changed.
- Simplified the SBI interface and added custom and conditional population distributions.
- Added the host mass step `gammaDM` to SBI.
- Renamed the simulation result `snmagshift` to `gammaDM`.
- Photo-z fits of events without a `REDSHIFT_FINAL` now use no redshift prior ([#3](https://github.com/mattgrayling/jax-snana/issues/3)).
- Added the `PRIOR_MUERRSCALE` and `PRIOR_DMU_RANGE` distance priors to photo-z fits.
- Added a `PEAKMJD` axis to the photo-z coarse grid.
- Added a Docker image of jax-snana, published to the GitHub Container Registry (@mcoughlin).
- Fixed matching of non-numeric CIDs in CID lists ([#2](https://github.com/mattgrayling/jax-snana/issues/2)).
- Fixed issues with newer versions of pandas.
- Renamed the `JAX_VERSION` output header line to `JAX-SNANA_VERSION`.

## 0.1.0

Initial release, accompanying Grayling 2026 (arXiv:2610.19000).
