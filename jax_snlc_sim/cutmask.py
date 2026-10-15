"""CUTMASK bit positions, matching SNANA's selection-cut tracking.

Each bit corresponds to one selection cut (snlc_sim.h:1786-1797). A bit set
means the cut passed; a SN that passes every cut has CUTMASK_ALL_PASS.
"""

# Bit positions, SNANA snlc_sim.h:1786-1797.
CUTBIT_TRESTMAX    = 0   # (1)    max Trest coverage
CUTBIT_TRESTMIN    = 1   # (2)    min Trest coverage (pre-max)
CUTBIT_SNRMAX      = 2   # (4)    max SNR per filter requirement
CUTBIT_TGAPMAX     = 3   # (8)    max time gap
CUTBIT_T0GAPMAX    = 4   # (16)   max time gap near peak
CUTBIT_NOBS_SNR    = 5   # (32)   NOBS with SNR threshold (NEPOCH)
CUTBIT_NOBS_MJDDIF = 6   # (64)   NOBS with MJD difference
CUTBIT_MWEBV       = 7   # (128)  Milky Way extinction
CUTBIT_REDSHIFT    = 8   # (256)  redshift range
CUTBIT_PEAKMAG     = 9   # (512)  peak magnitude
CUTBIT_TIME_ABOVE  = 10  # (1024) time above SNR threshold
CUTBIT_SATURATE    = 11  # (2048) saturation cuts

NUM_CUTBITS = 12

CUTMASK_ALL_PASS = (1 << NUM_CUTBITS) - 1

CUTBIT_MAP = {
    'TRESTMAX': CUTBIT_TRESTMAX,
    'TRESTMIN': CUTBIT_TRESTMIN,
    'SNRMAX': CUTBIT_SNRMAX,
    'TGAPMAX': CUTBIT_TGAPMAX,
    'T0GAPMAX': CUTBIT_T0GAPMAX,
    'NOBS_SNR': CUTBIT_NOBS_SNR,
    'NOBS_MJDDIF': CUTBIT_NOBS_MJDDIF,
    'MWEBV': CUTBIT_MWEBV,
    # SNANA shares CUTBIT_REDSHIFT across the redshift triad (snlc_sim.c:24182)
    'REDSHIFT': CUTBIT_REDSHIFT,
    'REDSHIFT_TRUE': CUTBIT_REDSHIFT,
    'REDSHIFT_FINAL': CUTBIT_REDSHIFT,
    'HOST_ZPHOT': CUTBIT_REDSHIFT,
    'PEAKMAG': CUTBIT_PEAKMAG,
    'TIME_ABOVE': CUTBIT_TIME_ABOVE,
    'SATURATE': CUTBIT_SATURATE,
}
