# bbc_des_beams parity fixture

BBC/BEAMS inputs for the per-event-pIa contamination tests, all DES *griz* on one
SIMLIB, kcor and reference cosmology. Assembled by `build.py` from three registered
SNANA sims; run that after regenerating any of them, or these files go stale.

| file | rows | from |
| --- | --- | --- |
| `DATA.FITRES` | 1775 | `../bbc_des_data` |
| `BIASCOR.FITRES` | 18049 | `../bbc_des_biascor` |
| `CCPRIOR.FITRES` | 3936 | `../bbc_ccprior_sim` |
| `CONTAM_DATA.FITRES` | 1895 | `DATA` + 120 CC events at CIDs 100000+ |

`build.py` writes those four from the parent sims; `refs.py` runs `SALT2mu.exe` over
them for every `SALT2mu*.M0DIF`/`.FITRES`/`.COV` reference here, so the configs live in
that script rather than in this file. Both take `--check` to report drift without
writing.

Bin overrides are checked through normal BBC preparation and fitting in
`tests/integration/test_bbc_bin_overrides.py`, with explicit selection lists and
expected bins. Stock SNANA ignores `izbin_from_cid_file` because of its parser
bug, so these checks do not use a SNANA
reference. Bare CID lists reuse the ordinary CID-selection reference.

One thing a reference file cannot record is a file SNANA did **not** write.
`SALT2mu_cutmask_write_m9.FITRES` is the whole output of its config: the same run with
`write_csv=1` added produced no `.M0DIF`, no `.CSV` and no `.COV`.

All three parents are `IDSURVEY=10`. SALT2mu partitions by survey, so a sample from
a different survey lands in an IDSAMPLE with no biasCor and aborts in
`print_eventStats` -- and relabelling it would leave events whose fitted `x1`/`c`
came from one cadence being bias-corrected by another's.

## Why a separate DES set

The CC prior is the per-z-bin spread of `mu - mumodel` over the CC events, so it
needs `x1` and `c` constrained. The LSST fixtures' CC sample had 77% of its events
sitting on SALT's initial guess -- `x1=0.001, c=0.010, x1ERR=0.30, cERR=0.03`, the
values a non-converged fit is written back with -- because the bundled toy SIMLIB
gives ~50 epochs spread over 920 days. Those identical rows collapsed the top z
bin's spread to `DMURMS = 0.004`, and a prior that narrow flips an ordinary event's
classification under a 1e-3 parameter change, producing a second minimum in the
BEAMS likelihood 3.4 lower in `-2log(L)` than the first.

DES's single-season cadence (~92 epochs per field) fixes it at the source:

| | LSST CC sample | DES CC sample |
| --- | --- | --- |
| events | 2486 | 3936 |
| non-converged fits | 1926 (77%) | 0 |
| median `x1ERR` | 1.06 | 0.69 |
| inside `\|x1\|<3, \|c\|<0.3` | 126 | 654 |
| in-box events per z bin | 2 bins empty | `[41, 84, 115, 89, 81, 67, 58, 72, 47]` |

Depth and sample size were not the problem -- an LSST-toy sim of 400k light curves
gave median `x1ERR = 1.60` and 15 events in the box. Sparse sampling was.

## Measured

`nzbin=10 zmin=0.1 zmax=1.0 u1=1 u2=1 p9=0.7 p11=-1 opt_biascor=112`
`surveygroup_biascor=DES(zbin=0.15:x1bin=2.0:cbin=0.2)`, BEAMS adding
`simfile_ccprior=CCPRIOR.FITRES p13=1 varname_pIa=PROBIA typeIa_ccprior=999`.
`typeIa_ccprior=999` is required: `CONTAM_DATA` has no `TYPE` column, so it reads
-9, which equals the key's own default and would force every `pIa` to 1 through
`force_probcc0` (`SALT2mu.c:21852`) -- silently disabling BEAMS.

| config | alpha0 | beta0 | sigint |
| --- | --- | --- | --- |
| clean `DATA`, no CC prior | 0.18069 | 3.78280 | 0 (1473 fit) |
| BEAMS on `CONTAM_DATA` | 0.18347 | 3.83495 | 0 (1562 fit) |
| + `fitflag_sigmb=1` | 0.15457 | 3.28842 | 0, 5 iterations |
| + `fitflag_sigmb=1 sigint_step1=0.05` | 0.14048 | 2.92793 | 0.0912, 4 iterations |

jax matches SALT2mu.exe to all five printed decimals on every row, including the
iteration counts.

The last row recovers the generated values: `SIM_alpha = 0.14`, and `sigint = 0.0912`
against `GENMAG_SMEAR = 0.10`. The rows above it fit at `sigint = 0`, which leaves
the intrinsic scatter unmodelled and biases alpha high -- `chi2(Ia)/dof = 1.483` on
the clean fit says as much. `sigint_step1` matters because the secant seeds at
`sigint = 0`, where `d(redchi2)/d(sigint)` is identically zero (sigint enters the
variance squared), so its first slope estimate is proportional to the step size.

## Not covered here

`force_probcc0`'s per-event masking stays on `bbc_biascor_fit`. Forcing the
contaminants to full weight runs this sample into the fit bounds -- `alpha0 = 0.3000`,
`beta0 = 6.0000`, both `+- 0.0000` -- so a reference would pin the bound rather than
the mask. Host-mass (`gamma0`, 4D logmass) needs a `HOST_LOGMASS` spread, and every
event here is `-9999`; the LSST fixture injects a uniform one, which a DES HOSTLIB
would do better. H11 needs instrumented SNANA output.

## PROBIA

The one fabricated column: a truth-based stand-in for a classifier, assigned
deterministically by row index in `build.py` (Ia 0.88-0.999, CC 0.001-0.119). It
separates the two classes perfectly, which is fine for a parity test -- both
programs read the same numbers -- but means these fixtures cannot catch an error
confined to genuinely ambiguous events.

### Derived bias-correction variants

`BIASCOR_MASS.FITRES`, `BIASCOR_SDSS.FITRES`, `BIASCOR_FIELD.FITRES`, and
`BIASCOR_PHOTOZ.FITRES` are generated in temporary directories by the
`bbc_biascor_variants` pytest fixture and by `refs.py`. Both use `build()`
from `build.py`. These inputs are not stored in the repository; the separate
SALT2mu reference outputs remain recorded. `build.py --check` checks the
other derived inputs that are stored here.
