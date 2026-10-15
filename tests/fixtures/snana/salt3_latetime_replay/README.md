# Fixture: salt3_latetime_replay

Generated: 2026-09-17
SNANA version: v12_02d-12-g74dfb2c5
Command: `snlc_sim.exe salt3_latetime_replay/input.input` (run from a scratch dir with `JAX_TEST_FIXTURE_ROOT=fixtures`, a link to `tests/fixtures/data`)

GENVERSION: `JAX_TEST_SALT3_LATETIME`

Files:
- PHOT.FITS.gz (32.6 KB)
- HEAD.FITS.gz (45.1 KB)
- DUMP (149.2 KB)
- SIM.README (5.2 KB)

Refresh with `python tests/fixtures/regenerate.py --only salt3_latetime_replay`.

HEAD/PHOT retain only the first 300 distinct DUMP CIDs, using
`regenerate.trim_replay_photometry`. The complete DUMP is retained.
