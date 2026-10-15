# create_cov parity fixture

A stand-in for a `submit_batch` BBC output tree: a 3x3 FITOPT x MUOPT grid of
Hubble diagrams, plus SNANA's own `create_covariance.py` outputs for it.

Everything here is written by `build.py`, which needs `SALT2mu.exe` on `$PATH`
and SNANA's `create_covariance.py` (which imports `scikit-learn` and `seaborn`
at module level, for code its own `DO_FIT = False` disables). Run `build.py
--check` to report drift without writing.

| path | contents |
| --- | --- |
| `bbc/SUBMIT.INFO` | `FITOPT_OUT_LIST` and `MUOPT_OUT_LIST` — labels and numbering |
| `bbc/OUTPUT_BBC/FITOPT<i>_MUOPT<j>.M0DIF.gz` | binned HD per cell |
| `bbc/OUTPUT_BBC/FITOPT<i>_MUOPT<j>.FITRES.gz` | per-SN HD per cell |
| `sys_scale.yml` | label -> systematic scale |
| `extra_cov.dat` | an externally supplied covariance, over every other event |
| `input*.yml` | create_covariance configs, paths relative to this directory |
| `ref*/` | SNANA outputs, one directory per case below |

| reference | what it covers |
| --- | --- |
| `ref` | binned, the default mode |
| `ref_unbinned` | `--unbinned` |
| `ref_rebin` | `--nbin_x1 2 --nbin_c 2` |
| `ref_covsys` | covsys as gzipped text, which is not a default output |
| `ref_covtot` | `--write_mask_cov 7`: covsys, covtot_inv and covtot together |
| `ref_extracov` | `EXTRA_COVS` |
| `ref_calibrators` | `EXTRA_COVS` + `CALIBRATORS` |
| `ref_vpec` | `--subtract_vpec` |
| `ref_nosys` | `--nosys` |
| `ref_syshd` | `--systematic_HD` |

`--muopt` has no reference: it aborts in SNANA for any N > 0.

`ref_extracov` and `ref_calibrators` would both pass a parity check while
doing nothing, so
`test_extra_covs_and_calibrators_change_the_result` pins that they don't:
`EXTRA_COVS` is only picked up by the unfiltered `[ALL]` covopt, and the
calibrator masking only fires for a systematic labelled VPEC or ZSHIFT — hence
the `VPEC_EXTRA` label. The calibrator CIDs are **quoted** in the config: YAML
would otherwise read them as ints, and CIDs are strings, so the list would match
nothing in SNANA and here alike.

## How the grid is faked, and why

`create_covariance` aggregates a grid; it does not produce one. In SNANA that
grid comes from `submit_batch_jobs` running `snlc_fit` once per FITOPT and
`SALT2mu` once per cell. jax has no `submit_batch`, so `build.py` stands in for
one. The two axes are emulated differently because they mean different things:

- **FITOPT** is a light-curve-fit systematic, so each gets its own input FITRES
  with the fluxes perturbed.
- **MUOPT** is a BBC option, so each is a different `SALT2mu` key.

Provenance is deliberately synthetic. That is sound for parity: both jax and
SNANA read the *same* grid, so the comparison tests the aggregation, not where
the variants came from.

## Three things that had to be got right

**Perturb `x0`, not `mB`.** SALT2mu recomputes the peak magnitude from `x0` and
ignores the `mB` column outright — a 1.0 mag shift written to `mB` leaves the
M0DIF byte-identical.

**The shift must vary with z.** BBC reports `M0DIF = M0(iz) - <M0>`, so a
constant magnitude shift cancels exactly and contributes nothing. Real
ZP/wavelength systematics are per-filter and hence z-dependent, which the
`linear` and `quadratic` shapes imitate.

**MUOPTs cannot be selection cuts.** Keys such as `maxerr_abort_x1` also thin
the biasCor sample and abort in `init_sigInt_biasCor_SNRCUT`. The two used here,
`chi2max` and `sigint_fix`, are ones `bbc_des_beams/refs.py` already exercises
with biasCor on.

## Properties the fixture depends on

**`STRIDE = 6`** thins the 1775-event DES sample to ~296. An unbinned covariance
is one row per SN and there are five of them, so the full sample would commit
7.6 MB of reference matrices. Rows are thinned by stride, not truncated, so every
z bin stays populated.

**Both inputs carry a `VERSION_PHOTOMETRY` header line.** `create_covariance`
reads `wdlist[2]` of every such comment when writing `INFO.YML` and raises
`IndexError` on an empty value, so without the line it cannot write that file at
all. `build.py` stamps the line onto the FITRES copies it feeds SALT2mu.

**`BASE` sets `vpecerr`.** Without it `MUERR_VPEC` is all zeros and
`--subtract_vpec` is an exact no-op, so its reference would pass while testing
nothing.

**The configs name paths relative to the repo root, and both codes run from
there.** SNANA resolves config paths against the current directory, not the
config file, so these only work from the repo root — which is where the suite
runs. That is also why `INFO.YML` records identical paths on both sides: each
writes them exactly as configured, with no rewriting.

**No cosmology is stated, deliberately.** SALT2mu leaves
`VERSION_PHOTOMETRY(BIASCOR)` blank when biasCor comes from a FITRES rather than
a versioned sim folder, so neither code can find a sim README and both fall back
to `FlatLambdaCDM(H0=70, Om0=0.315)` — jax warning as it does so. That keeps the
fixture shaped like a tree SNANA would actually produce, and makes the fallback
itself the thing under test. `COSPAR_BIASCOR` in `INFO.YML` is empty as a
result, which is correct: the biasCor cosmology genuinely is unknown here.
