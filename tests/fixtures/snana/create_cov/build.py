#!/usr/bin/env python
"""Build the FITOPT x MUOPT grid this fixture needs, plus its SNANA reference.

    python build.py [--check]

A submit_batch BBC output tree is a directory of FITOPT<i>_MUOPT<j> Hubble diagrams
plus a SUBMIT.INFO naming and scaling each systematic. jax has no submit_batch, so
this script stands in for one: it runs SALT2mu.exe once per grid cell over the
bbc_des_beams inputs and writes the tree create_covariance expects.

The two axes are emulated differently, matching what each means in SNANA:

- FITOPT is a light-curve-fit systematic, so each one gets its own input FITRES with
  mB perturbed. The shift must vary with z: BBC reports M0DIF = M0(iz) - <M0>, so a
  constant mB shift cancels exactly and contributes nothing to the covariance. Real
  ZP/wavelength systematics are per-filter and therefore z-dependent, which is what
  the shapes below imitate.
- MUOPT is a BBC-level option, so each one is a different SALT2mu key.

Each cell contributes a gzipped M0DIF (read by binned mode) and FITRES (read by
unbinned and rebin), plus SUBMIT.INFO and one SNANA reference per mode. The
perturbed input FITRES are intermediates in a temp dir.

SALT2mu.exe runs from a short temp dir: a prefix of 116+ characters overruns
`char PREFIX[100]` (SALT2mu.c:1390).
"""


import argparse
import gzip
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
# paths as they appear in the committed configs: relative to the repo root,
# because SNANA resolves them against the cwd and the suite runs from there
REL = HERE.relative_to(REPO)
BEAMS = HERE.parent / "bbc_des_beams"

# keep every 6th event: 1775 -> ~296, enough to fit 10 z bins while keeping the
# unbinned covariance (one row per SN, five covopts) small enough to commit
STRIDE = 6

INPUT_DIR = HERE / "bbc"
VERSION = "OUTPUT_BBC"

# SALT2mu propagates this into its output, where create_covariance reads it
# for INFO.YML. An empty value makes its parser raise IndexError.
VERSION_PHOTOMETRY = "JAX_TEST_CREATE_COV"

# SALT2mu keys shared by every cell -- the converged DES BEAMS configuration that
# bbc_des_beams/refs.py uses for its own references.
# The CC prior is what makes MUERR_RENORM and PROBCC_BEAMS exist: without it the
# rebin weights collapse to a single scheme and the BEAMS column is never
# written, so neither would be exercised. typeIa_ccprior=999 avoids the -9 trap
# that silently sets pIa=1 for a sample with no TYPE column.
BASE = {
    "simfile_biascor": "BIASCOR.FITRES",
    "simfile_ccprior": "CCPRIOR.FITRES",
    "opt_ccprior": "5",
    "varname_pIa": "PROBIA",
    "p13": "1",
    "typeIa_ccprior": "999",
    "opt_biascor": "112",
    "min_per_cell_biascor": "3",
    "snrmin_sigint_biascor": "60",
    "surveygroup_biascor": "DES(zbin=0.15:x1bin=2.0:cbin=0.2)",
    "nzbin": "10", "zmin": "0.1", "zmax": "1.0",
    "u1": "1", "u2": "1", "p9": "0.7", "p11": "-1",
    "blindflag": "0",
    # without this MUERR_VPEC is all zeros and subtract_vpec is an exact no-op;
    # large enough that the subtraction shows at the %8.5f the HD is printed to
    "vpecerr": "1000",
}

# (label, amplitude, shape) -- mB += amp * shape(zHD). FITOPT000 must be the
# unperturbed reference; create_covariance differences everything against it.
FITOPTS = [
    ("DEFAULT", 0.00, None),
    ("ZPSHIFT_CAL", 0.020, "linear"),
    ("LAMSHIFT_CAL", 0.040, "quadratic"),
]

# (label, extra SALT2mu keys). MUOPT000 must be the unperturbed reference.
# Selection keys such as maxerr_abort_x1 thin the biasCor sample too and abort in
# init_sigInt_biasCor_SNRCUT, so these are ones refs.py runs with biasCor on.
MUOPTS = [
    ("DEFAULT", {}),
    ("CHI2MAX_SCAT", {"chi2max": "16"}),
    ("SIGINT_SCAT", {"fitflag_sigmb": "1", "sigint_fix": "0.11"}),
]

# label -> systematic scale, the SYS_SCALE_FILE create_covariance reads
SYS_SCALES = {
    "ZPSHIFT_CAL": 0.3,
    "LAMSHIFT_CAL": 0.6,
    "SIGINT_SCAT": 1.0,
    "CHI2MAX_SCAT": 1.2,
}

COVOPTS = [
    "[NOSYS] [=DEFAULT,=DEFAULT]",
    "[CAL] [+CAL,=DEFAULT]",
    "[NOCAL] [-CAL,=DEFAULT]",
    "[SCAT] [=DEFAULT,+SCAT]",
    # the optional third term is an ERROR scale, so the covariance takes its
    # square -- the one covopt feature with no other reference
    "[SCALED] [=DEFAULT,+SCAT,1.3]",
    # empty filters on either side, the remaining apply_filter forms
    "[NOCALONLY] [-CAL,]",
    "[ANYSCAT] [,+SCAT]",
]


def _shift(shape: str, z: float) -> float:
    """Unit-amplitude z dependence. Both integrate to roughly zero across the fit
    range so the shift is not mostly absorbed by BBC's <M0> subtraction."""
    if shape == "linear":
        return z - 0.5
    if shape == "quadratic":
        return (z - 0.5) ** 2 - 0.06
    raise ValueError(f"unknown shape {shape}")


def write_shifted_fitres(src: Path, dst: Path, amp: float, shape: str | None) -> None:
    """Thin DATA.FITRES to N_SNE events and brighten by amp*shape(zHD) magnitudes.

    The shift is applied to x0, not mB: SALT2mu recomputes the peak magnitude from
    x0 and ignores the mB column outright -- a 1.0 mag shift written to mB leaves
    the M0DIF byte-identical. x0ERR scales with x0 so the fractional flux error,
    and hence mBERR, is unchanged.

    Rows are thinned by a fixed stride rather than truncated, so every z bin stays
    populated. The unbinned covariance is one row per SN, so the full 1775-event
    sample would put ~7.6 MB of reference matrices in the repo.

    A synthetic PROBIA column is added because SNANA requires varname_pIa to run
    the CC prior, and the CC prior is what produces MUERR_RENORM -- without which
    the two rebin weight schemes are identical and untestable. It falls with
    redshift so the renormalised error actually differs from MUERR. The clean
    sample is used rather than CONTAM_DATA because that one carries no SIM
    columns, which would make SALT2mu treat it as real data and stop writing
    MUSIM.
    """
    varnames = None
    kept = 0
    seen = 0
    out = [f"# VERSION_PHOTOMETRY: {VERSION_PHOTOMETRY}\n"]
    for line in src.read_text().splitlines(keepends=True):
        if line.startswith("VARNAMES:"):
            varnames = line.split()[1:]
            out.append(line.rstrip("\n") + " PROBIA\n")
        elif line.startswith("SN:"):
            seen += 1
            if seen % STRIDE:
                continue
            kept += 1
            words = line.split()
            i_z_always = varnames.index("zHD") + 1
            if shape is not None:
                # words[0] is the SN: key, so column i of VARNAMES is words[i+1]
                i_z = varnames.index("zHD") + 1
                scale = 10.0 ** (-0.4 * amp * _shift(shape, float(words[i_z])))
                for name in ("x0", "x0ERR"):
                    i = varnames.index(name) + 1
                    words[i] = f"{float(words[i]) * scale:.6e}"
            probia = max(0.40, min(1.0, 1.0 - 0.6 * float(words[i_z_always])))
            out.append(" ".join(words) + f" {probia:.4f}\n")
        else:
            out.append(line)
    if varnames is None:
        sys.exit(f"{src} has no VARNAMES line")
    if kept < 100:
        sys.exit(f"STRIDE={STRIDE} leaves only {kept} events; too few to fit")
    dst.write_text("".join(out))


def drop_izbin(text: str, izbin: int) -> str:
    """Remove every SN in one redshift bin, leaving a hole in the IZBIN sequence.

    Rebinning sizes its bin loop by the number of distinct IZBIN but indexes with
    the raw value, so a hole pushes the bins above it past the end of the loop and
    they are dropped. Nothing SALT2mu produces here has such a hole, so this is the
    only way to reach that path.
    """
    varnames = None
    out = []
    for line in text.splitlines(keepends=True):
        if line.startswith("VARNAMES:"):
            varnames = line.split()[1:]
        elif line.startswith("SN:"):
            if int(line.split()[varnames.index("IZBIN") + 1]) == izbin:
                continue
        out.append(line)
    if varnames is None:
        sys.exit("no VARNAMES line to find IZBIN in")
    return "".join(out)


def stamp_version_photometry(src: Path, dst: Path) -> None:
    """Copy a FITRES with a VERSION_PHOTOMETRY header line added."""
    dst.write_text(f"# VERSION_PHOTOMETRY: {VERSION_PHOTOMETRY}\n" + src.read_text())


def run_cell(workdir: Path, name: str, datafile: Path, extra: dict) -> dict[str, str]:
    """One SALT2mu.exe run; returns its M0DIF and FITRES with the CPU stamp dropped.

    The binned mode reads the M0DIF, unbinned and rebin read the FITRES, and all
    three glob the same directory -- so both live in the grid side by side.
    """
    keys = {**BASE, **extra}
    lines = [f"prefix={name}", f"datafile={datafile}"]
    for k, v in keys.items():
        local = v.endswith(".FITRES") if isinstance(v, str) else False
        lines.append(f"{k}={BEAMS / v if local else v}")
    inp = workdir / f"{name}.inp"
    inp.write_text("\n".join(lines) + "\n")

    log = subprocess.run(["SALT2mu.exe", inp.name], cwd=workdir,
                         capture_output=True, text=True)
    out = {}
    for ext in ("M0DIF", "FITRES"):
        path = workdir / f"{name}.{ext}"
        if not path.is_file():
            sys.exit(f"{name}: SALT2mu.exe produced no .{ext}\n"
                     + log.stdout[-3000:] + log.stderr[-2000:])
        out[ext] = "".join(ln for ln in path.read_text().splitlines(keepends=True)
                           if not ln.startswith("# CPU:"))
    return out


def submit_info() -> str:
    """The two lists create_covariance reads for labels and scales.

    Both live in the BBC SUBMIT.INFO, and the label sits in a different column for
    each: get_sys_scales takes FITOPT labels from index 2 and MUOPT labels from
    index 1 (create_covariance.py:1024-1030).
    """
    fit = "\n".join(f"- [FITOPT{i:03d}, {lab}, {lab}]"
                    for i, (lab, _, _) in enumerate(FITOPTS))
    mu = "\n".join(f"- [MUOPT{j:03d}, {lab}, {lab}]"
                   for j, (lab, _) in enumerate(MUOPTS))
    return f"FITOPT_OUT_LIST:\n{fit}\n\nMUOPT_OUT_LIST:\n{mu}\n"


# biasCor comes from a FITRES here, so VERSION_PHOTOMETRY(BIASCOR) is blank and
# SNANA falls back to FlatLambdaCDM(H0=70, Om0=0.315) (:982). The BBC outputs
# record no cosmology either, so the committed inputs state that one, which jax
# requires; SNANA's own runs read none.
LEGACY_COSMOLOGY = {"OMEGA_MATTER": 0.315, "OMEGA_LAMBDA": 0.685,
                    "w0_LAMBDA": -1.0, "wa_LAMBDA": 0.0}


def config_text(input_dir, sys_scale_file, outdir, extra_cov_path=None,
                calibrators=False, cosmology=None) -> str:
    """create_covariance input file. The committed copy uses paths relative to this
    directory; run_reference rewrites them absolute for SNANA."""
    covopts = "\n".join(f"- '{c}'" for c in COVOPTS)
    text = (f"INPUT_DIR: {input_dir}\n"
            f"VERSION: {VERSION}\n"
            f"SYS_SCALE_FILE: {sys_scale_file}\n"
            f"OUTDIR: {outdir}\n"
            f"COVOPTS:\n{covopts}\n")
    if extra_cov_path:
        text += f"EXTRA_COVS:\n- {EXTRA_COV_LABEL} 1.0 {extra_cov_path}\n"
    if calibrators:
        # quoted: YAML would read bare 8 as an int, and CIDs are strings,
        # so an unquoted list silently matches nothing in SNANA and here
        text += "CALIBRATORS:\n" + "".join(f'- "{c}"\n' for c in CALIBRATORS)
    if cosmology:
        text += "COSMOLOGY:\n" + "".join(f"  {k}: {v}\n" for k, v in cosmology.items())
    return text


def sys_scale_text() -> str:
    return "".join(f"{k}: {v}\n" for k, v in sorted(SYS_SCALES.items()))


# each reference directory, the create_covariance.py flags that produce it, and
# whether its config declares EXTRA_COVS
REF_MODES = {
    "ref": ([], False),                                     # binned, the default
    "ref_unbinned": (["--unbinned"], False),
    "ref_rebin": (["--nbin_x1", "2", "--nbin_c", "2"], False),
    # covsys text is not written by default (WRITE_MASK_COV_DEFAULT is covtot_inv
    # only), so it needs its own run to have a reference at all
    "ref_covsys": (["--write_mask_cov", "1", "--write_format_cov", "text"], False),
    "ref_extracov": (["--unbinned"], True),
    "ref_calibrators": (["--unbinned"], True),
    # all three cov products at once: covsys | covtot_inv | covtot
    "ref_covtot": (["--write_mask_cov", "7"], False),
    # rebin, not unbinned: with a CC prior present unbinned overwrites MUERR
    # with MUERR_RENORM after the subtraction, so the flag has no visible
    # effect there. In rebin the subtracted MUERR survives into wgt0.
    "ref_vpec": (["--subtract_vpec", "--nbin_x1", "2", "--nbin_c", "2"], False),
    # asymmetric bin counts: at 2x2 the two factors in the bin index are
    # equal, so transposing them would go unnoticed
    "ref_rebin31": (["--nbin_x1", "3", "--nbin_c", "1"], False),
    # a hole in IZBIN: the bin loop is sized by the distinct count, so the bins
    # above the hole fall outside it and their events are dropped
    "ref_izgap": (["--nbin_x1", "2", "--nbin_c", "2"], False),
    "ref_nosys": (["--nosys"], False),
    "ref_syshd": (["--systematic_HD"], False),
}

# reference dirs whose config also names CALIBRATORS
CALIBRATOR_MODES = {"ref_calibrators"}

# reference dirs reading the gapped grid rather than the full one
IZGAP_MODES = {"ref_izgap"}
IZGAP_DIR = "bbc_izgap"
IZGAP_BIN = 4

EXTRA_COV_FILE = "extra_cov.dat"
EXTRA_COV_LABEL = "VPEC_EXTRA"

# Cepheid hosts carry no redshift systematic, so VPEC/ZSHIFT terms must not touch
# their rows. The label above contains VPEC, which is what triggers that masking.
CALIBRATORS = ["8", "29", "38", "52", "64"]


def extra_cov_text(fitres_text: str) -> str:
    """A diagonal externally-supplied covariance over the grid's own events.

    Real use is a vpec systematic computed once rather than by re-fitting, so a
    z-dependent diagonal is representative. Only a subset of events is listed, so
    the reader's skip-unmatched path is exercised too.
    """
    varnames, rows = None, []
    for line in fitres_text.splitlines():
        if line.startswith("VARNAMES:"):
            varnames = line.split()[1:]
        elif line.startswith("SN:"):
            rows.append(line.split()[1:])
    i_cid, i_id = varnames.index("CID"), varnames.index("IDSURVEY")
    i_z = varnames.index("zHD")

    out = ["CID1 IDSURVEY1 CID2 IDSURVEY2 MU_COV"]
    for row in rows[::2]:                       # every other event
        z = float(row[i_z])
        out.append(f"{row[i_cid]} {row[i_id]} {row[i_cid]} {row[i_id]} "
                   f"{0.0004 * (1.0 + z):.6e}")
    return "\n".join(out) + "\n"


def run_reference(workdir: Path) -> dict:
    """Run SNANA's create_covariance.py over the grid, once per input mode."""
    script = Path(shutil.which("create_covariance.py") or
                  os.path.expandvars("$SNANA_DIR/util/create_covariance.py"))
    if not script.is_file():
        sys.exit(f"cannot find create_covariance.py at {script}")


    out = {}
    for name, (flags, extra) in REF_MODES.items():
        outdir = workdir / name
        cfg = workdir / f"input_{name}.yml"
        cfg.write_text(config_text(
            REL / (IZGAP_DIR if name in IZGAP_MODES else "bbc"),
            REL / "sys_scale.yml", outdir,
            REL / EXTRA_COV_FILE if extra else None,
            calibrators=name in CALIBRATOR_MODES))
        print(f"  reference: {name} {' '.join(flags)}"
              + ("  +EXTRA_COVS" if extra else ""))

        # from HERE, so the relative paths above resolve and INFO.YML records
        # them relative -- the same values our own writer produces
        log = subprocess.run([sys.executable, str(script), str(cfg)] + flags,
                             cwd=REPO, capture_output=True, text=True)
        if log.returncode != 0 or not outdir.is_dir():
            sys.exit(f"create_covariance.py failed for {name} "
                     f"(exit {log.returncode})\n"
                     + log.stdout[-4000:] + log.stderr[-3000:])

        for path in sorted(outdir.rglob("*")):
            if not path.is_file():
                continue
            # _sync gzips .gz targets itself; SNANA's covsys is already
            # gzipped and would otherwise be stored compressed twice
            raw = path.read_bytes()
            out[f"{name}/{path.relative_to(outdir)}"] = (
                gzip.decompress(raw) if path.suffix == ".gz" else raw)
    if not out:
        sys.exit("create_covariance.py wrote no comparable outputs")
    return out


def build(check: bool) -> int:
    if not shutil.which("SALT2mu.exe"):
        sys.exit("SALT2mu.exe is not on $PATH")

    produced = {}
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        # both inputs need the header line: create_covariance reads wdlist[2] of
        # every VERSION_PHOTOMETRY comment and raises IndexError on an empty value
        biascor = workdir / "BIASCOR.FITRES"
        stamp_version_photometry(BEAMS / "BIASCOR.FITRES", biascor)
        ccprior = workdir / "CCPRIOR.FITRES"
        stamp_version_photometry(BEAMS / "CCPRIOR.FITRES", ccprior)

        for i, (flabel, amp, shape) in enumerate(FITOPTS):
            data = workdir / f"D{i}.FITRES"
            write_shifted_fitres(BEAMS / "DATA.FITRES", data, amp, shape)
            for j, (mlabel, extra) in enumerate(MUOPTS):
                name = f"FITOPT{i:03d}_MUOPT{j:03d}"
                print(f"  {name}  [{flabel} | {mlabel}]")
                cell = run_cell(workdir, f"f{i}m{j}", data,
                                {**extra, "simfile_biascor": str(biascor),
                                 "simfile_ccprior": str(ccprior)})
                for ext, text in cell.items():
                    produced[f"{name}.{ext}"] = text

    produced["SUBMIT.INFO"] = submit_info()

    outdir = INPUT_DIR / VERSION
    files = {(INPUT_DIR / n if n == "SUBMIT.INFO" else outdir / (n + ".gz")): t
             for n, t in produced.items()}

    gap_dir = HERE / IZGAP_DIR
    for n, t in produced.items():
        target = (gap_dir / n if n == "SUBMIT.INFO"
                  else gap_dir / VERSION / (n + ".gz"))
        files[target] = drop_izbin(t, IZGAP_BIN) if n.endswith(".FITRES") else t
    files[HERE / "sys_scale.yml"] = sys_scale_text()
    files[HERE / EXTRA_COV_FILE] = extra_cov_text(
        produced["FITOPT000_MUOPT000.FITRES"])
    files[HERE / "input.yml"] = config_text(
        REL / "bbc", REL / "sys_scale.yml", REL / "ref", cosmology=LEGACY_COSMOLOGY)
    files[HERE / "input_extracov.yml"] = config_text(
        REL / "bbc", REL / "sys_scale.yml", REL / "ref_extracov",
        REL / EXTRA_COV_FILE, cosmology=LEGACY_COSMOLOGY)
    files[HERE / "input_calibrators.yml"] = config_text(
        REL / "bbc", REL / "sys_scale.yml", REL / "ref_calibrators",
        REL / EXTRA_COV_FILE, calibrators=True, cosmology=LEGACY_COSMOLOGY)
    files[HERE / "input_izgap.yml"] = config_text(
        REL / IZGAP_DIR, REL / "sys_scale.yml", REL / "ref_izgap",
        cosmology=LEGACY_COSMOLOGY)

    changed = sum(_sync(p, d, check) for p, d in files.items())

    # the reference runs against the grid on disk, so it has to come after
    with tempfile.TemporaryDirectory() as tmp:
        for name, data in run_reference(Path(tmp)).items():
            changed += _sync(HERE / name, data, check)
    return changed


def _sync(path: Path, data, check: bool) -> int:
    """Write data to path unless it is already there. Returns 1 if it differed.

    ``.gz`` targets are compared decompressed, so a rerun that produces identical
    content does not churn the file through gzip's embedded timestamp.
    """
    binary = isinstance(data, bytes)
    if path.is_file():
        raw = path.read_bytes()
        current = gzip.decompress(raw) if path.suffix == ".gz" else raw
        if not binary:
            current = current.decode()
        if current == data:
            return 0
    if check:
        print(f"    DRIFT {path.relative_to(HERE)}")
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = data if binary else data.encode()
    path.write_bytes(gzip.compress(payload) if path.suffix == ".gz" else payload)
    print(f"    wrote {path.relative_to(HERE)}")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report drift without writing")
    args = ap.parse_args()
    changed = build(args.check)
    if args.check and changed:
        sys.exit(f"{changed} file(s) differ; rerun without --check")
    print(f"{changed} file(s) {'differ' if args.check else 'written'}")


if __name__ == "__main__":
    main()
