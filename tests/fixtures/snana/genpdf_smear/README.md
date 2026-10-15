# Fixture: genpdf_smear

Generated: 2026-09-21
SNANA version: v12_02d-12-g74dfb2c5
Command: `snlc_sim.exe genpdf_smear/input.input` (run from a scratch dir with `JAX_TEST_FIXTURE_ROOT=fixtures`, a link to `tests/fixtures/data`)

GENVERSION: `JAX_TEST_GENPDF_SMEAR`

Files:
- HEAD.FITS.gz (45.8 KB)
- PHOT.FITS.gz (26.7 KB)
- DUMP (139.8 KB)
- SIM.README (5.3 KB)

Refresh with `python tests/fixtures/regenerate.py --only genpdf_smear`.

HEAD/PHOT retain only the first 300 distinct DUMP CIDs, using
`regenerate.trim_replay_photometry`. The complete DUMP is retained.
