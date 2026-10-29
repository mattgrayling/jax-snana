#!/usr/bin/env python3
"""Regenerate the file-format references with unmodified SALT2mu.exe.

Run from the repository root; --check reports drift without writing fixtures.
"""
import argparse
import gzip
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from tests.parity.test_bbc_output_contract import CONFIGS, CHI2GRID_NBIN
from tests.fixtures.snana.bbc_des_beams.build import build
from jax_snlc_core.snana_files import open_text, resolve_fits_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    changed = False
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        work = Path(tmp)
        tables, _ = build()
        photoz = work / "BIASCOR_PHOTOZ.FITRES"
        photoz.write_text(tables[photoz.name])
        for name, overrides in CONFIGS.items():
            keys = []
            for item in overrides:
                key, value = item.split("=", 1)
                if key in ("datafile", "simfile_biascor", "simfile_ccprior"):
                    source = ROOT / value
                    target = work / f"{name}_{key}.FITRES"
                    with open_text(source) as stream:
                        target.write_text(stream.read())
                    value = str(target)
                keys.append(f"{key}={value}")
            if name == "salt_1d_split":
                keys.append(f"simfile_biascor={photoz}")
            keys += ["write_yaml=1", "write_csv=1",
                     f"write_chi2grid={CHI2GRID_NBIN}", f"prefix={name}"]
            inp = work / f"{name}.input"
            inp.write_text("\n".join(keys) + "\n")
            process = subprocess.run(["SALT2mu.exe", inp.name], cwd=work,
                                     capture_output=True, text=True)
            if process.returncode:
                sys.exit(process.stdout[-3000:] + process.stderr)
            for suffix in ("FITRES", "M0DIF", "YAML", "COV", "CSV", "CHI2GRID"):
                path = HERE / f"{name}.{suffix}"
                generated = work / path.name
                text = generated.read_text()
                recorded = ""
                if Path(resolve_fits_path(path)).is_file():
                    with open_text(path) as stream:
                        recorded = stream.read()
                generated_lines = [line for line in text.splitlines()
                                   if not line.startswith(("# CPU:", "CPU_MINUTES:"))]
                recorded_lines = [line for line in recorded.splitlines()
                                  if not line.startswith(("# CPU:", "CPU_MINUTES:"))]
                same = generated_lines == recorded_lines
                print(f"{path.name}: {'unchanged' if same else 'WOULD CHANGE'}")
                changed |= not same
                if not args.check:
                    if len(text.encode()) >= 100_000:
                        path.with_name(path.name + ".gz").write_bytes(
                            gzip.compress(text.encode(), compresslevel=6, mtime=0))
                        path.unlink(missing_ok=True)
                    else:
                        path.write_text(text)
                        path.with_name(path.name + ".gz").unlink(missing_ok=True)
    if args.check and changed:
        sys.exit(1)


if __name__ == "__main__":
    main()
