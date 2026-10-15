# HDIBC reference

Two Hubble diagrams of the same ten events, each in its own directory with the
`INFO.YML` that names the cosmology its biasCor sim was generated at. `wfit.exe`
reads that file from the directory holding the diagram (`compute_mu_biascor`,
`wfit.c:1642-1652`), evaluates each event's distance under both cosmologies, and
interpolates between the two diagrams at the trial cosmology
(`get_chi2_fit`, `wfit.c:2298-2299`). With `-outfile_resid` it writes the two
bias legs and the interpolation fraction per event
(`wfit.c:5370-5412`), which is what
`tests/unit/test_wfit_data_likelihood.py` compares against.

The three references are the same two diagrams under different omega_m priors,
chosen so the fraction lands inside the `[-0.5, 1.5]` window
(`ref_plain`, 0.654 to 0.693) and hard on each bound (`ref_clip_hi` at 1.5,
`ref_clip_lo` at -0.5). Both biasCor cosmologies are flat, so jax's transverse
curvature correction, which `get_mu_cos` omits, plays no part.

SNANA version: `v12_01d-24-g563a1bc7`, the version pinned in
`tests/fixtures/snana_version.txt`.

Commands, run from a scratch directory holding `HD0/` and `HD1/`:

```
wfit.exe HD0/HD.FITRES,HD1/HD.FITRES -lcdm -speed_flag_chi2 0 \
    -ompri 0.30 -dompri 0.05 -outfile_resid ref_plain.resid   -outfile_cospar ref_plain.cospar
wfit.exe HD0/HD.FITRES,HD1/HD.FITRES -lcdm -speed_flag_chi2 0 \
    -ompri 0.80 -dompri 0.01 -outfile_resid ref_clip_hi.resid -outfile_cospar ref_clip_hi.cospar
wfit.exe HD0/HD.FITRES,HD1/HD.FITRES -lcdm -speed_flag_chi2 0 \
    -ompri 0.02 -dompri 0.01 -outfile_resid ref_clip_lo.resid -outfile_cospar ref_clip_lo.cospar
```

`python3 tests/fixtures/wfit/build.py` writes the diagrams and runs all three;
`--check` fails if anything differs.
