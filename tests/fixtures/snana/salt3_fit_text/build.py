#!/usr/bin/env python3
"""Record six TEXT light curves from the salt3_fit_select simulation."""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from regenerate import check_pinned_version, prepare_snana_assets

CIDS = (26, 61, 77, 84, 116, 148)
VERSION = "JAX_TEST_SALT3_FIT_TEXT"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    check_pinned_version(False)
    changed = False
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        work = Path(tmp)
        assets = prepare_snana_assets(work / "assets")
        sndata = work / "sndata"
        sndata.mkdir()
        for source in Path(os.environ["SNDATA_ROOT"]).iterdir():
            if source.name != "SIM":
                (sndata / source.name).symlink_to(source.resolve(), target_is_directory=source.is_dir())
        (sndata / "SIM").mkdir()
        (sndata / "SIM/PATH_SNDATA_SIM.LIST").touch()
        env = {**os.environ, "JAX_TEST_FIXTURE_ROOT": str(assets), "SNDATA_ROOT": str(sndata)}
        process = subprocess.run([
            "snlc_sim.exe", str(HERE.parent / "salt3_fit_select/input.input"),
            "FORMAT_MASK", "2", "GENVERSION", VERSION,
            "PATH_SNDATA_SIM", str(work)], cwd=work, env=env,
            capture_output=True, text=True)
        if process.returncode:
            sys.exit(process.stdout[-3000:] + process.stderr)
        names = [f"{VERSION}_SN{cid:06d}.DAT" for cid in CIDS]
        outputs = {name: (work / VERSION / name).read_bytes() for name in names}
        outputs[f"{VERSION}.LIST"] = ("\n".join(names) + "\n").encode()
        for name, payload in outputs.items():
            path = HERE / name
            same = path.is_file() and path.read_bytes() == payload
            print(f"{name}: {'unchanged' if same else 'WOULD CHANGE'}")
            changed |= not same
            if not args.check:
                path.write_bytes(payload)
    if args.check and changed:
        sys.exit(1)


if __name__ == "__main__":
    main()
