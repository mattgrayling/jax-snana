# sedtrue_replay

SEDMODEL true-SED spectra mode (`SPECTROGRAPH_OPTMASK` bit 128, triggered by
`LAMBIN_SED_TRUE: 20.0` + `LAMRANGE_SED_TRUE: 3000 9000`). SNANA builds an
IDEAL spectrograph and writes one noiseless true-SED spectrum at every
photometry epoch (dedup 0.05 d, TEXPOSE 20). Output uses SNANA's reduced SPEC
format: a uniform wavelength grid stored as per-spectrum header scalars
(`LAMMIN`/`LAMMAX`/`LAMBIN`/`NBIN_LAM`) and `SIM_FLAM` as the sole flux column,
with `-777` end-of-spectrum markers.

The SPEC header also carries `SIM_SYNMAG_<band>` (synthetic photometry of the
true SED per band), and `VERIFY_SED_TRUE.DAT` is the diagnostic table
(`CID z MJD TREST LAMREST BAND MAG MAG_VERIFY MAG_DIF`) comparing broadband MAG
to the synthetic MAG re-integrated from the spectrum.

Generate 50 events directly with the pinned SNANA version; no trimming is needed.
The recipe records HEAD, SPEC, DUMP and the simulator's VERIFY_SED_TRUE diagnostic.
PHOT is omitted because replay uses DUMP and HEAD.

```sh
python tests/fixtures/regenerate.py --only sedtrue_replay
python tests/fixtures/regenerate.py --only sedtrue_replay --diff
```
