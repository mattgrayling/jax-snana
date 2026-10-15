# Fixture: host_spectra_replay

Generated: 2026-09-17
SNANA version: v12_02d-12-g74dfb2c5
Command: `snlc_sim.exe host_spectra_replay/input.input` (run from a scratch dir with `JAX_TEST_FIXTURE_ROOT=fixtures`, a link to `tests/fixtures/data`)
TEXT reference: `snlc_sim.exe input.input FORMAT_MASK 2`, same env; .DAT copied to `text_ref/`

GENVERSION: `JAX_TEST_HOST_SPECTRA`

Files:
- text_ref/ (20 .DAT from FORMAT_MASK=2)
- PHOT.FITS.gz (5.0 KB)
- HEAD.FITS.gz (11.7 KB)
- SPEC.FITS.gz (18.8 KB)
- DUMP (2.1 KB)
- SIM.README (5.9 KB)

Refresh with `python tests/fixtures/regenerate.py --only host_spectra_replay`.
