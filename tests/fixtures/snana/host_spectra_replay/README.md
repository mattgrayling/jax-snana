# Fixture: host_spectra_replay

Generated: 2026-11-06
SNANA version: v12_02f
Command: `snlc_sim.exe host_spectra_replay/input.input` (run from a scratch dir with `JAX_TEST_FIXTURE_ROOT=fixtures`, a decompressed copy of `tests/fixtures/data`)
TEXT reference: `snlc_sim.exe input.input FORMAT_MASK 2`, same env; .DAT copied to `text_ref/`

GENVERSION: `JAX_TEST_HOST_SPECTRA`

Files:
- PHOT.FITS.gz (5.0 KB)
- HEAD.FITS.gz (10.8 KB)
- SPEC.FITS.gz (18.7 KB)
- DUMP (2.1 KB)
- SIM.README (5.7 KB)
- text_ref/ (20 .DAT from FORMAT_MASK=2)

Refresh with `python tests/fixtures/regenerate.py --only host_spectra_replay`.
