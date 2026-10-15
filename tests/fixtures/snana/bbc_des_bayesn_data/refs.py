#!/usr/bin/env python
"""Generate the SALT2mu.exe references for this fixture, one run per CONFIGS entry.

    python refs.py [--check] [NAME ...]

The biasCor table lives in the sibling bbc_des_bayesn_biascor fixture.
"""


import argparse
import gzip
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BC = HERE.parent / "bbc_des_bayesn_biascor"

# p9/p11 are given because SNANA defaults them to 0 while jax defaults to the
# reference cosmology; the rest mirrors _cfg() in tests/parity/test_bbc_bayesn.py.
BASE = {
    "model_lcfit": "BAYESN",
    "datafile": str(HERE / "FITRES.TEXT.gz"),
    "simfile_biascor": str(BC / "FITRES.TEXT.gz"),
    "opt_biascor": "144",
    "sigint_biascor": "0.10",
    "nbin_logmass": "1",
    "snrmin_sigint_biascor": "6.0",
    "nzbin": "4", "zmin": "0.05", "zmax": "0.21",
    "p9": "0.7", "p11": "-1",
}

# avmin/avmax/thetamin/thetamax are BayeSN's half of the range inputs that set both
# the cut and the biasCor cell ranges (SALT2mu.c:18816-18823, :18876-18880), the
# same role cmin/cmax/x1min/x1max play for SALT. The cuts below reject events
# that pass the default AV (0..1) and THETA (-3..3) ranges.
CONFIGS = {
    "SALT2mu_bayesn":          (BASE, ["M0DIF"]),
    "SALT2mu_bayesn_avmax":    ({**BASE, "avmax": "0.4"}, ["M0DIF"]),
    "SALT2mu_bayesn_thetamin": ({**BASE, "thetamin": "-1.0"}, ["M0DIF"]),
    # opt_biascor 144 is MU|CELLMAP, which never builds the muCOVscale map; +32
    # turns it on so the AV axis (biascor_binning's second key) is actually binned.
    "SALT2mu_bayesn_mucovscale":       ({**BASE, "opt_biascor": "176"}, ["M0DIF"]),
    "SALT2mu_bayesn_mucovscale_avmax": ({**BASE, "opt_biascor": "176",
                                         "avmax": "0.4"}, ["M0DIF"]),
    # applyCut_chi2max runs off the nuisance parameters MINUIT clamped into their
    # bounds (:2486, :3175); BayeSN declares none, so nothing is clamped. The large
    # MU errors here put every event under chi2=2, so the cut has to be this tight
    # to reject events; at 16 it selects the whole sample.
    "SALT2mu_bayesn_chi2max":  ({**BASE, "chi2max": "0.5"}, ["M0DIF"]),
    # 1DZWGT weights the z-interpolated bias by 1/muerrsq, the only 1D path that
    # touches the mu-variance code, and nothing else exercises it on a model with a
    # single LC parameter. 1DZAVG (opt_biascor=1) takes the WGT=1 branch instead.
    "SALT2mu_bayesn_1dzwgt":   ({**BASE, "opt_biascor": "2"}, ["M0DIF"]),
    # sigint_biascor is always given: dropping it aborts SALT2mu, because BayeSN has
    # no alpha/beta/gamma nodes to spread the events over, so NUSE[0][0][0] reaches
    # NROW_malloc and trips the >= check at :13574.
}


def run(name, keys, keep, workdir: Path):
    inp = workdir / f"{name}.inp"
    lines = [f"prefix={name}"] + [f"{k}={v}" for k, v in keys.items() if v != ""]
    inp.write_text("\n".join(lines) + "\n")
    log = subprocess.run(["SALT2mu.exe", inp.name], cwd=workdir,
                         capture_output=True, text=True)
    out = {}
    for ext in keep:
        path = workdir / f"{name}.{ext}"
        if not path.is_file():
            sys.exit(f"{name}: SALT2mu.exe produced no .{ext}\n"
                     + log.stdout[-3000:] + log.stderr[-2000:])
        # drop SALT2mu's wall-clock stamp so reruns are byte-stable
        out[f"{name}.{ext}"] = "".join(
            ln for ln in path.read_text().splitlines(keepends=True)
            if not ln.startswith("# CPU:"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report what would change without writing")
    ap.add_argument("names", nargs="*", help="subset of CONFIGS to run")
    args = ap.parse_args()

    if not shutil.which("SALT2mu.exe"):
        sys.exit("SALT2mu.exe is not on $PATH")
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        # SALT2mu needs plain text when reading the bias-correction header.
        data = Path(tmp) / "DATA.FITRES"
        biascor = Path(tmp) / "BIASCOR.FITRES"
        for compressed, plain in ((HERE / "FITRES.TEXT.gz", data),
                                  (BC / "FITRES.TEXT.gz", biascor)):
            with gzip.open(compressed, "rb") as source, plain.open("wb") as target:
                shutil.copyfileobj(source, target)
        for name in args.names or list(CONFIGS):
            keys, keep = CONFIGS[name]
            keys = {**keys, "datafile": str(data), "simfile_biascor": str(biascor)}
            for fname, text in run(name, keys, keep, Path(tmp)).items():
                path = HERE / fname
                same = path.is_file() and path.read_text() == text
                print(f"  {fname}: {'unchanged' if same else 'WOULD CHANGE'}")
                if not args.check:
                    path.write_text(text)
    if not args.check:
        print("written")


if __name__ == "__main__":
    main()
