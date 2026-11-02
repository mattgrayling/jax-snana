# Changelog

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
