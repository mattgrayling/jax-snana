#!/usr/bin/env python3
"""Build the frozen wfit.exe references this fixture is compared against.

The committed `SALT2mu.M0DIF` is 10 bins and is the only Hubble diagram jax was
pinned against, through one grid fit. This adds a wider diagram and captures
real `wfit.exe` cospar output for several option sets, so the same comparisons
the live_snana suite makes can run without SNANA installed.

It also builds the HDIBC fixture under `hdibc/`: one pair of Hubble diagrams
whose `INFO.YML` files name two biasCor cosmologies, and the per-event
`f_interp_hdibc` / `mu_bcor*_hdibc` columns wfit.exe writes for them
(`wfit.c:5366-5408`). Three omega_m priors put the interpolation fraction
inside the clip window and hard against each of its bounds.

    python3 tests/fixtures/wfit/build.py           # regenerate
    python3 tests/fixtures/wfit/build.py --check   # fail if anything differs

Run from the repository root with wfit.exe on $PATH.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HDIBC_DIR = HERE / "hdibc"

#: The cosmology the wide diagram is generated at, and its scatter. Omega_m sits
#: well away from the default prior mean of 0.315 so that a fit which ignored the
#: data would land visibly off, which a truth near the prior would hide.
TRUTH = {"omega_m": 0.24, "w0": -0.92}
SCATTER = 0.035
SEED = 20240813

#: Each reference: the diagram it uses and the wfit.exe flags that produce it.
REFERENCES = {
    "ref_binned": ("SALT2mu.M0DIF", []),
    "ref_binned_lcdm": ("SALT2mu.M0DIF", ["-lcdm"]),
    "ref_wide": ("wide.M0DIF", []),
    "ref_wide_lcdm": ("wide.M0DIF", ["-lcdm"]),
    "ref_wide_ompri": ("wide.M0DIF", ["-ompri", "0.30", "-dompri", "0.02"]),
    "ref_wide_zcut": ("wide.M0DIF", ["-zmin", "0.10", "-zmax", "0.90"]),
    "ref_wide_snrms": ("wide.M0DIF", ["-snrms", "0.05"]),
    "ref_binned_cmb": ("SALT2mu.M0DIF", ["-cmb"]),
    "ref_binned_bao_desi": ("SALT2mu.M0DIF", ["-bao", "DESI_2024"]),
    # three rows, two cosmology parameters and the intercept: no freedom left
    "ref_three_row": ("three_row.M0DIF", []),
}


#: The HDIBC fixture. Both biasCor cosmologies are flat, so jax's transverse
#: curvature correction (which wfit's get_mu_cos omits) plays no part.
HDIBC_Z = [0.02, 0.05, 0.10, 0.18, 0.28, 0.40, 0.55, 0.72, 0.90, 1.10]
HDIBC_TRUTH = 0.30
HDIBC_MUERR = 0.05
#: HD1 sits above HD0 by an amount that grows with redshift, so the blend is
#: visible per event rather than as one offset every event shares.
HDIBC_OFFSET = (0.05, 0.20)   # constant, slope in z
HDIBC_SEED = 20260912
HDIBC_BIASCOR = {"HD0": 0.20, "HD1": 0.40}

#: Each reference is the same two diagrams under a different omega_m prior, so
#: that the fraction lands inside the window and on each clip bound in turn.
HDIBC_REFERENCES = {
    "ref_plain": ["-ompri", "0.30", "-dompri", "0.05"],
    "ref_clip_hi": ["-ompri", "0.80", "-dompri", "0.01"],
    "ref_clip_lo": ["-ompri", "0.02", "-dompri", "0.01"],
}


def hdibc_files() -> dict:
    """The two Hubble diagrams and their INFO.YML files, by relative path."""
    z = np.asarray(HDIBC_Z)
    cids = [f"SN{i + 1:04d}" for i in range(z.size)]
    truth = distance_modulus(z, HDIBC_TRUTH, -1.0)
    generator = np.random.default_rng(HDIBC_SEED)
    errors = np.full(z.size, HDIBC_MUERR)
    observed = truth + generator.normal(0.0, 1.0, z.size) * errors

    files = {}
    for index, (name, omega_m) in enumerate(HDIBC_BIASCOR.items()):
        mu = observed + index * (HDIBC_OFFSET[0] + HDIBC_OFFSET[1] * z)
        lines = ["VARNAMES: CID zHD MU MUERR"]
        lines += [f"SN: {c} {zz:.5f} {m:.5f} {e:.5f}"
                  for c, zz, m, e in zip(cids, z, mu, errors)]
        files[f"{name}/HD.FITRES"] = "\n".join(lines) + "\n"
        files[f"{name}/INFO.YML"] = (
            "MUSHIFT: 0.0\n"
            f"OMEGA_MATTER: {omega_m:.2f}\n"
            f"OMEGA_LAMBDA: {1.0 - omega_m:.2f}\n"
            "w0_LAMBDA: -1.0\n"
            "wa_LAMBDA: 0.0\n"
        )
    return files


def run_wfit_hdibc(workdir: Path, flags: list) -> tuple:
    """One two-diagram wfit.exe run, returning its (resid, cospar) output."""
    resid, cospar = workdir / "out.resid", workdir / "out.cospar"
    result = subprocess.run(
        [
            "wfit.exe", "HD0/HD.FITRES,HD1/HD.FITRES", "-lcdm",
            "-speed_flag_chi2", "0",
            "-outfile_resid", str(resid), "-outfile_cospar", str(cospar),
            *flags,
        ],
        cwd=workdir, capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0 or not resid.is_file():
        sys.exit(
            f"wfit.exe HDIBC failed for {' '.join(flags)}\n"
            + result.stdout[-3000:] + result.stderr[-2000:]
        )
    return resid.read_text(), cospar.read_text()


def distance_modulus(z, omega_m, w0):
    """Flat wCDM distance modulus at H0 = 70, by the same quadrature as jax."""
    from jax_snlc_core.physics.constants import LIGHT_KM
    from jax_snlc_core.physics.cosmology import distance_modulus as mu_of_z

    z = np.asarray(z)                       # a synthetic diagram: zHEL = zHD
    return np.asarray(mu_of_z(
        z, z, omega_m, 1.0 - omega_m, LIGHT_KM / 70.0, w0, 0.0,
    ))


def wide_diagram() -> str:
    """A 30-bin diagram spanning z = 0.02 to 1.20, with deterministic scatter."""
    edges = np.linspace(0.02, 1.20, 31)
    centres = 0.5 * (edges[:-1] + edges[1:])
    reference = distance_modulus(centres, 0.3, -1.0)
    truth = distance_modulus(centres, TRUTH["omega_m"], TRUTH["w0"])

    generator = np.random.default_rng(SEED)
    # errors shrink with the bin population, as a real binned diagram's do
    counts = np.linspace(60, 8, centres.size).astype(int)
    errors = SCATTER / np.sqrt(counts)
    offsets = generator.normal(0.0, 1.0, centres.size) * errors

    lines = [
        "# Synthetic wide Hubble diagram for frozen wfit.exe parity.",
        f"# Generated at omega_m={TRUTH['omega_m']}, w0={TRUTH['w0']}, "
        f"scatter={SCATTER}, seed={SEED}.",
        "VARNAMES: ROW zHDMIN zHDMAX zHD MUDIF MUDIFERR MUREF NFIT",
    ]
    for index, centre in enumerate(centres):
        mudif = truth[index] - reference[index] + offsets[index]
        lines.append(
            f"ROW: {index + 1} {edges[index]:.5f} {edges[index + 1]:.5f} "
            f"{centre:.5f} {mudif:.5f} {errors[index]:.5f} "
            f"{reference[index]:.5f} {counts[index]}"
        )
    return "\n".join(lines) + "\n"


def run_wfit(workdir: Path, diagram: Path, flags: list[str]) -> str:
    """One wfit.exe run, returning its cospar output."""
    cospar = workdir / "out.cospar"
    result = subprocess.run(
        [
            "wfit.exe", str(diagram),
            # exact evaluation: the interpolation tricks are a SNANA-only
            # approximation jax deliberately does not reproduce
            "-speed_flag_chi2", "0",
            "-outfile_cospar", str(cospar),
            *flags,
        ],
        cwd=workdir, capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0 or not cospar.is_file():
        sys.exit(
            f"wfit.exe failed for {diagram.name} {' '.join(flags)}\n"
            + result.stdout[-3000:] + result.stderr[-2000:]
        )
    return cospar.read_text()


def build(check: bool) -> int:
    if shutil.which("wfit.exe") is None:
        sys.exit("wfit.exe is not on $PATH")

    files = {HERE / "wide.M0DIF": wide_diagram()}
    changed = sum(_sync(path, text, check) for path, text in files.items())

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        for name, text in files.items():
            (workdir / name.name).write_text(text)
        for committed in ("SALT2mu.M0DIF", "three_row.M0DIF"):
            shutil.copy(HERE / committed, workdir / committed)
        for name, (diagram, flags) in REFERENCES.items():
            print(f"  {name}: {diagram} {' '.join(flags)}")
            cospar = run_wfit(workdir, workdir / diagram, flags)
            changed += _sync(HERE / f"{name}.cospar", cospar, check)

    hdibc = hdibc_files()
    for relative, text in hdibc.items():
        path = HDIBC_DIR / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        changed += _sync(path, text, check)

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        for relative, text in hdibc.items():
            path = workdir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        for name, flags in HDIBC_REFERENCES.items():
            print(f"  hdibc/{name}: {' '.join(flags)}")
            resid, cospar = run_wfit_hdibc(workdir, flags)
            changed += _sync(HDIBC_DIR / f"{name}.resid", resid, check)
            changed += _sync(HDIBC_DIR / f"{name}.cospar", cospar, check)
    return changed


def _sync(path: Path, text: str, check: bool) -> int:
    """Write text to path unless already there. Returns 1 if it differed."""
    if path.is_file() and path.read_text() == text:
        return 0
    if check:
        print(f"    differs: {path.relative_to(HERE)}")
        return 1
    path.write_text(text)
    print(f"    wrote {path.relative_to(HERE)}")
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    changed = build(args.check)
    print(f"{changed} file(s) {'differ' if args.check else 'written'}")
    return 1 if (args.check and changed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
