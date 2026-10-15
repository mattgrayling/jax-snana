#!/usr/bin/env python
"""Record what SALT2mu.exe's check_vpec_sign does with the two tables.

    python refs.py [--check]

The reference is not a fitted output: the guard prints two RMS values and either
continues or aborts, so that is what gets recorded, one row per table:

    <name> <exit_code> <rms_nominal> <rms_flip>

p9/p11 are set explicitly. SALT2mu defaults Omega_L and w0 to **0**, not to the
reference cosmology, and the resulting distances change the RMS by ~2% -- enough to
look like a porting error when it is only an unset key.
"""


import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFERENCE = "SALT2mu_vpec_sign.txt"

KEYS = ["nzbin=2", "zmin=0.015", "zmax=0.05", "u1=1", "u2=1", "p9=0.7", "p11=-1"]
_RMS = re.compile(r"RMS\(MURES,(nominal|flip-vpec)\)\s*=\s*([0-9.]+)")


def run(name, workdir: Path):
    inp = workdir / f"{name}.inp"
    inp.write_text("\n".join([f"datafile={HERE / name}.FITRES", *KEYS,
                              f"prefix={workdir / name}"]) + "\n")
    proc = subprocess.run(["SALT2mu.exe", inp.name], cwd=workdir,
                          capture_output=True, text=True)
    vals = dict(_RMS.findall(proc.stdout))
    if len(vals) != 2:
        sys.exit(f"{name}: check_vpec_sign did not report both RMS values\n"
                 + proc.stdout[-2000:])
    return f"{name} {proc.returncode} {vals['nominal']} {vals['flip-vpec']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report what would change without writing")
    args = ap.parse_args()
    if not shutil.which("SALT2mu.exe"):
        sys.exit("SALT2mu.exe is not on $PATH")

    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        rows = [run(n, Path(tmp)) for n in ("vpec_ok", "vpec_bad")]
    text = ("# name exit_code rms_nominal rms_flip -- SALT2mu.exe check_vpec_sign\n"
            + "\n".join(rows) + "\n")
    path = HERE / REFERENCE
    same = path.is_file() and path.read_text() == text
    print(f"  {REFERENCE}: {'unchanged' if same else 'WOULD CHANGE'}")
    if not args.check:
        path.write_text(text)
        print("written")


if __name__ == "__main__":
    main()
