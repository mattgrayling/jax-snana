# Fixture: salt3_fit_text

SNANA TEXT-format light curves (<VERSION>.LIST + per-SN .DAT) for 6 SNe from
the salt3_fit_select realisation (same RANSEED), written with FORMAT_MASK: 2
instead of 32. Exercises the TEXT loader (load_and_batch_snana_text); the
batch must match the FITS load of the same SNe (flux to TEXT's ~6 sig figs).

Run `python tests/fixtures/snana/salt3_fit_text/build.py` to regenerate, or add
`--check` to compare without writing. The script uses the pinned SNANA version,
the parent's simulation input, FORMAT_MASK=2 and the six explicitly listed CIDs.
It stages plain model assets and keeps simulation outputs in a temporary directory.
