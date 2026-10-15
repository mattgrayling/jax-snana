"""Rebuild the salt3_fit_host2 fixture: the salt3_fit_photoz HEAD with its
first eight events given a second matched host and a peculiar velocity, then
refitted by snlc_fit.exe with fit.nml so every anchored column is data-derived,
and once more with a GALID-keyed quantile override (fit_headover_qgalid.nml)
into FITRES_headover_qgalid.TEXT.

    python tests/fixtures/snana/salt3_fit_host2/build.py

Needs snlc_fit.exe on PATH and the parent fixture's SIM.README (from the
regeneration directory) as README.SIM beside its HEAD, since the fitter
refuses a version without one.
"""
import argparse
import gzip
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "salt3_fit_photoz"
sys.path.insert(0, str(HERE.parents[1]))
from regenerate import prepare_snana_assets, _diff

GENVERSION = "JAX_TEST_SALT3_FIT_PHOTOZ"

DDLR_OFFSETS = [0.05, 0.2, 0.5, 1.0, 2.0, 3.0, 0.01, 5.0]
BANDS = "ugrizY"


def doctor(head):
    """The README's edits, rows 0-7."""
    n = len(DDLR_OFFSETS)
    rows = np.arange(n)
    head["HOSTGAL_NMATCH"][rows] = 2
    head["HOSTGAL_NMATCH2"][rows] = 2
    head["HOSTGAL2_DDLR"][rows] = head["HOSTGAL_DDLR"][rows] + DDLR_OFFSETS
    head["HOSTGAL2_OBJID"][rows] = 200001 + rows
    head["HOSTGAL2_PHOTOZ"][rows] = head["HOSTGAL_PHOTOZ"][rows] + 0.05
    head["HOSTGAL2_PHOTOZ_ERR"][rows] = 0.04
    head["HOSTGAL2_SPECZ"][rows] = head["HOSTGAL_SPECZ"][rows] + 0.04
    head["HOSTGAL2_SPECZ_ERR"][rows] = 0.002
    head["HOSTGAL2_RA"][rows] = head["HOSTGAL_RA"][rows] + 1e-3
    head["HOSTGAL2_DEC"][rows] = head["HOSTGAL_DEC"][rows] - 1e-3
    head["HOSTGAL2_SNSEP"][rows] = head["HOSTGAL_SNSEP"][rows] + 1.5
    head["HOSTGAL2_LOGMASS"][rows] = 9.5 + 0.2 * rows
    head["HOSTGAL2_LOGMASS_ERR"][rows] = 0.15
    for band in BANDS:
        head[f"HOSTGAL2_MAG_{band}"][rows] = head[f"HOSTGAL_MAG_{band}"][rows] + 0.7
        head[f"HOSTGAL2_MAGERR_{band}"][rows] = 0.05
    head["VPEC"][:5] = 300.0
    head["VPEC_ERR"][:5] = 90.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not shutil.which("snlc_fit.exe"):
        sys.exit("snlc_fit.exe is not on $PATH")
    readme = HERE / "README.SIM"
    if not readme.is_file():
        sys.exit(f"{readme} is missing: copy the parent fixture's SIM.README there")
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        assets = prepare_snana_assets(Path(tmp) / "assets")
        generated = Path(tmp) / "generated"
        generated.mkdir()
        version = Path(tmp) / GENVERSION
        version.mkdir()
        with fits.open(PARENT / "HEAD.FITS.gz") as hdul:
            doctor(hdul[1].data)
            hdul.writeto(version / f"{GENVERSION}_HEAD.FITS")
        shutil.copy(PARENT / "PHOT.FITS.gz", version / f"{GENVERSION}_PHOT.FITS.gz")
        (version / f"{GENVERSION}.LIST").write_text(f"{GENVERSION}_HEAD.FITS\n")
        shutil.copy(readme, version / f"{GENVERSION}.README")
        env = {**os.environ, "JAX_TEST_FIXTURE_ROOT": str(assets),
               "JAX_TEST_FIT_SIM_DIR": tmp}
        out = Path(tmp) / "out"
        out.mkdir()
        for nml, table in (("fit.nml", "FITRES.TEXT"),
                           ("fit_headover_qgalid.nml", "FITRES_headover_qgalid.TEXT")):
            log = subprocess.run(["snlc_fit.exe", str(HERE / nml)], cwd=out,
                                 env=env, capture_output=True, text=True)
            if log.returncode != 0:
                sys.exit(log.stdout[-3000:])
            shutil.copy(out / f"{GENVERSION}.FITRES.TEXT", generated / table)
        (generated / "HEAD.FITS.gz").write_bytes(gzip.compress(
            (version / f"{GENVERSION}_HEAD.FITS").read_bytes(), compresslevel=6, mtime=0))
        changed = False
        for path in sorted(generated.iterdir()):
            recorded = HERE / path.name
            difference = _diff(path, recorded) if recorded.is_file() else "missing"
            print(f"{path.name}: {difference or 'unchanged'}")
            changed |= difference is not None
            if not args.check:
                shutil.copy(path, HERE / path.name)
        if args.check and changed:
            sys.exit(1)



if __name__ == "__main__":
    main()
