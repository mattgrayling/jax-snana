#!/usr/bin/env python
"""Crop one V19 CC template onto a coarse grid for the NON1ASED parity tests.

    python crop.py [--check]

Source (needs $SNDATA_ROOT): NON1ASED.V19_CC_noHostXT/pycoco_SN2008fq_noHostCorr.SED,
INDEX 834, SN2008fq IIn. Every 4th phase and every 5th wavelength, no interpolation:
64 x 1880 -> 16 x 376, 4.6 MB -> 25 KB. The grid is coarse so the tests exercise
SNANA's NON1ASED interpolation; the template is not photometrically faithful.

Output is byte-identical to the committed file (gzipped with mtime=0).
"""


import argparse
import gzip
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = "pycoco_SN2008fq_noHostCorr.SED.gz"
SOURCE = "models/NON1ASED/NON1ASED.V19_CC_noHostXT/" + NAME

PHASE_STRIDE, WAVE_STRIDE = 4, 5


def crop() -> str:
    root = os.environ.get("SNDATA_ROOT")
    if not root:
        sys.exit("$SNDATA_ROOT is not set")
    src = Path(root) / SOURCE
    if not src.is_file():
        sys.exit(f"missing source {src}")

    header, rows = [], []
    for line in gzip.open(src, "rt").read().splitlines():
        (header if line.startswith("#") else rows).append(line)

    phases = sorted({line.split()[0] for line in rows}, key=float)
    nwave = len(rows) // len(phases)
    kept = [rows[p * nwave + w]
            for p in range(0, len(phases), PHASE_STRIDE)
            for w in range(0, nwave, WAVE_STRIDE)]

    out = header + [
        f"### Cropped from V19_CC_noHostXT/{NAME[:-3]} for jax-snana tests ###",
        f"### phase stride={PHASE_STRIDE}, wavelength stride={WAVE_STRIDE} ###",
    ]
    for line in kept:
        day, wave, flux = line.split()
        out.append("%7.2f %8.2f %.4E" % (float(day), float(wave), float(flux)))
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report what would change without writing")
    args = ap.parse_args()

    text = crop()
    path = HERE / NAME
    same = path.is_file() and gzip.open(path, "rt").read() == text
    print(f"{NAME}: {len(text.splitlines())} lines, "
          f"{'unchanged' if same else 'WOULD CHANGE'}")
    if not args.check:
        with gzip.GzipFile(path, "wb", mtime=0) as fh:
            fh.write(text.encode())
        print("written")


if __name__ == "__main__":
    main()
