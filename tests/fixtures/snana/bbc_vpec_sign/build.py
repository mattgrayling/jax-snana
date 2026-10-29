#!/usr/bin/env python
"""Build the two low-z tables for the check_vpec_sign parity test.

    python build.py [--check]

The guard (SALT2mu.c:4131) aborts when peculiar velocities look sign-inverted, so it
needs input that is unambiguously one or the other. Neither the DES fixtures nor any
other sample here can exercise it: they are generated with GENSIGMA_VPEC=0 and start
at z=0.1, and the check needs >=70 events with non-zero VPEC in 0.01<z<0.05.

Both tables share the same supernovae. zCMB carries the peculiar motion and the
distances follow the true cosmological redshift:

    vpec_ok.FITRES    zHD has the peculiar velocity removed  -- correct
    vpec_bad.FITRES   zHD has it *added* instead             -- the inverted convention

mB is encoded through x0, because that is what BBC reads (mB = -2.5*log10(x0));
writing it only in the mB column leaves every event at the same brightness.
"""


import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))

from jax_snlc_core.cosmology import distance_modulus  # noqa: E402
from jax_snlc_core.utils import LIGHT_KM  # noqa: E402

SEED = 7
NSN = 120
COLUMNS = ("CID IDSURVEY TYPE FIELD zHD zHDERR zHEL zCMB VPEC VPECERR MWEBV "
           "HOST_LOGMASS x1 x1ERR c cERR x0 x0ERR mB mBERR "
           "COV_x1_c COV_x1_x0 COV_c_x0").split()


def build():
    import jax.numpy as jnp
    rng = np.random.default_rng(SEED)
    zcos = rng.uniform(0.02, 0.045, NSN)          # true cosmological redshift
    vpec = rng.normal(0.0, 400.0, NSN)            # km/s
    zpec = vpec / LIGHT_KM
    zcmb = zcos + zpec
    z = jnp.asarray(zcos)                          # zHEL is written equal to zHD
    mu = np.asarray(distance_modulus(z, z, 0.3, 0.7, LIGHT_KM / 70.0, -1.0, 0.0,
                                     radiation=False))
    mB = mu - 30.0 + rng.normal(0.0, 0.05, NSN)   # M0_DEFAULT = -30
    x0 = 10.0 ** (-mB / 2.5)

    def table(zhd, header):
        rows = [f"# {h}" for h in header] + ["#", "VARNAMES: " + " ".join(COLUMNS)]
        rows += ["SN: " + " ".join(str(v) for v in [
            i + 1, 10, 1, "VOID", f"{zhd[i]:.6f}", 0.001, f"{zhd[i]:.6f}",
            f"{zcmb[i]:.6f}", f"{vpec[i]:.3f}", 0.0, 0.0, -9999,
            0.0, 0.5, 0.0, 0.03, f"{x0[i]:.8e}", 1e-9, f"{mB[i]:.5f}", 0.05,
            0.0, 0.0, 0.0]) for i in range(NSN)]
        return "\n".join(rows) + "\n"

    return {
        "vpec_ok.FITRES": table(zcos, [
            f"{NSN} low-z SNe with the peculiar velocity removed from zHD.",
            "check_vpec_sign accepts this. Built by build.py."]),
        "vpec_bad.FITRES": table(zcmb + zpec, [
            f"The same {NSN} SNe with the vpec correction applied backwards, so zHD",
            "is wrong by 2*zpec. check_vpec_sign aborts on this. Built by build.py."]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report what would change without writing")
    args = ap.parse_args()
    for name, text in build().items():
        path = HERE / name
        same = path.is_file() and path.read_text() == text
        print(f"  {name}: {'unchanged' if same else 'WOULD CHANGE'}")
        if not args.check:
            path.write_text(text)
    if not args.check:
        print("written")


if __name__ == "__main__":
    main()
