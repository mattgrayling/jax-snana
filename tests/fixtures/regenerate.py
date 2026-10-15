"""Regenerate SNANA outputs for bundled parity fixtures.

For each registered fixture, runs ``snlc_sim.exe`` against
``tests/fixtures/snana/<name>/input.input`` (with ``$JAX_TEST_FIXTURE_ROOT``
set to ``tests/fixtures/data``) and copies the resulting DUMP / HEAD.FITS
/ PHOT.FITS into the fixture directory, gzipping FITS after any trimming.
Fixtures with ``fit_nml`` then run ``snlc_fit.exe``, or the checkout's
``snlc_fit.py`` when ``fit_with_jax`` is set. The per-fixture README records
the simulation version, fit command and JAX revision when applicable.

Pinned SNANA version lives in ``tests/fixtures/snana_version.txt``.
A version mismatch is a hard error unless ``--force-version-bump`` is
passed.

Usage:
    python tests/fixtures/regenerate.py --list
    python tests/fixtures/regenerate.py --all
    python tests/fixtures/regenerate.py --only searcheff_snii_basic
    python tests/fixtures/regenerate.py --diff               # regenerate to tmp and diff
"""

import argparse
import datetime as dt
import gzip
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits


HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE / "data"
SNANA_DIR = HERE / "snana"
VERSION_PIN = HERE / "snana_version.txt"
# .DAT files bundled per text_ref fixture; matches the existing
# salt3_k21_replay reference. A full run writes one per SN.
TEXT_REF_MAX_FILES = 25


@dataclass(frozen=True)
class Fixture:
    """A bundled parity-test SNANA output."""

    name: str
    genversion: str
    fit_nml: str | None = None
    fit_with_jax: bool = False
    text_ref: bool = False       # also bundle FORMAT_MASK=2 .DAT into text_ref/
    # Refit another fixture's simulation rather than running one: only the fit
    # stage's tables are bundled, and the light curves stay committed once.
    sim_from: str | None = None
    head: bool = True
    phot: bool = True
    replay_events: int | None = None

    @property
    def root(self) -> Path:
        return SNANA_DIR / self.name

    @property
    def input_file(self) -> Path:
        return SNANA_DIR / (self.sim_from or self.name) / "input.input"

    @property
    def fit_nml_file(self) -> Path | None:
        return self.root / self.fit_nml if self.fit_nml else None


# Registered fixtures. Add new entries here as parity tests get bundled.
FIXTURES: list[Fixture] = [
    Fixture(name="sedtrue_replay", genversion="JAX_TEST_SEDTRUE", phot=False),
    Fixture(name="searcheff_snii_basic", genversion="JAX_TEST_SEARCHEFF_SNII_BASIC"),
    Fixture(name="dumpall_minimal_cols", genversion="JAX_TEST_DUMPALL_MIN"),
    Fixture(name="dumpall_extended_cols", genversion="JAX_TEST_DUMPALL_EXT"),
    Fixture(name="dumpall_wide_cols", genversion="JAX_TEST_DUMPALL_WIDE"),
    Fixture(name="dumpall_listed_cols",
            genversion="JAX_TEST_DUMPALL_LISTED"),
    Fixture(name="dumpall_wide_hostlib", genversion="JAX_TEST_DUMPALL_HOSTLIB"),
    Fixture(name="searcheff_mag", genversion="JAX_TEST_SEARCHEFF_MAG"),
    Fixture(name="detect_single_off", genversion="JAX_TEST_DETSINGLE_OFF"),
    Fixture(name="detect_single_on", genversion="JAX_TEST_DETSINGLE_ON"),
    Fixture(name="saturation_off", genversion="JAX_TEST_SAT_OFF", head=False, phot=False),
    Fixture(name="saturation_on", genversion="JAX_TEST_SAT_ON"),
    Fixture(name="bayesn_w22", genversion="JAX_TEST_BAYESN_W22",
            text_ref=True),
    Fixture(name="salt3_k21_replay", genversion="JAX_TEST_SALT3_K21_REPLAY",
            text_ref=True),
    Fixture(name="peakmjd_exclude", genversion="JAX_TEST_PEAKMJD_EXCLUDE"),
    Fixture(name="population_rewgt", genversion="JAX_TEST_POPULATION_REWGT"),
    Fixture(name="simlib_windows", genversion="JAX_TEST_SIMLIB_WINDOWS",
            text_ref=True),
    Fixture(name="salt3_fit_dense", genversion="JAX_TEST_SALT3_FIT_DENSE",
            fit_nml="fit.nml"),
    Fixture(name="salt3_fit_diagcov", genversion="JAX_TEST_SALT3_FIT_DIAGCOV",
            fit_nml="fit.nml"),
    Fixture(name="salt3_fit_modelerr", genversion="JAX_TEST_SALT3_FIT_DENSE",
            fit_nml="fit.nml", sim_from="salt3_fit_dense"),
    Fixture(name="salt3_fit_ignore", genversion="JAX_TEST_SALT3_FIT_DENSE",
            fit_nml="fit.nml", sim_from="salt3_fit_dense"),
    Fixture(name="salt3_fit_photoz_bound",
            genversion="JAX_TEST_SALT3_FIT_PHOTOZ", fit_nml="fit.nml", sim_from="salt3_fit_photoz"),
    Fixture(name="salt3_fit_magcor", genversion="JAX_TEST_SALT3_FIT_DENSE",
            fit_nml="fit.nml", sim_from="salt3_fit_dense"),
    Fixture(name="salt3_fit_fudge", genversion="JAX_TEST_SALT3_FIT_DENSE",
            fit_nml="fit.nml", sim_from="salt3_fit_dense"),
    Fixture(name="salt3_fit_nonlin_nea",
            genversion="JAX_TEST_SALT3_FIT_DENSE", fit_nml="fit.nml", sim_from="salt3_fit_dense"),
    Fixture(name="salt3_fit_nonlin_pix",
            genversion="JAX_TEST_SALT3_FIT_DENSE", fit_nml="fit.nml", sim_from="salt3_fit_dense"),
    Fixture(name="salt3_fit_fluxerr", genversion="JAX_TEST_SALT3_FIT_DENSE",
            fit_nml="fit.nml", sim_from="salt3_fit_dense"),
    Fixture(name="salt3_fit_errcombine",
            genversion="JAX_TEST_SALT3_FIT_DENSE", fit_nml="fit.nml", sim_from="salt3_fit_dense"),
    Fixture(name="salt3_fit_hostnoise",
            genversion="JAX_TEST_SALT3_FIT_HOSTNOISE", fit_nml="fit.nml"),
    Fixture(name="salt3_fit_mwebv", genversion="JAX_TEST_SALT3_FIT_MWEBV",
            fit_nml="fit.nml"),
    Fixture(name="salt3_fit_mjdsig", genversion="JAX_TEST_SALT3_FIT_DENSE",
            fit_nml="fit.nml", sim_from="salt3_fit_dense"),
    Fixture(name="salt3_fit_select", genversion="JAX_TEST_SALT3_FIT_SELECT",
            fit_nml="fit.nml"),
    Fixture(name="salt3_fit_zshift", genversion="JAX_TEST_SALT3_FIT_ZSHIFT",
            fit_nml="fit.nml"),
    Fixture(name="salt3_fit_quality", genversion="JAX_TEST_SALT3_FIT_QUALITY"),
    Fixture(name="salt3_fit_setpkmjd", genversion="JAX_TEST_SALT3_FIT_DENSE",
            fit_nml="fit.nml", sim_from="salt3_fit_dense"),
    Fixture(name="salt3_fit_fmaxclump", genversion="JAX_TEST_SALT3_FIT_DENSE",
            fit_nml="fit.nml", sim_from="salt3_fit_dense"),
    Fixture(name="salt3_fit_pinned", genversion="JAX_TEST_SALT3_FIT_DENSE",
            fit_nml="fit.nml", sim_from="salt3_fit_dense"),
    Fixture(name="salt3_fit_pinned_narrow", genversion="JAX_TEST_SALT3_FIT_DENSE",
            fit_nml="fit.nml", sim_from="salt3_fit_dense"),
    Fixture(name="hostlib_e2e", genversion="JAX_TEST_HOSTLIB_E2E", text_ref=True),
    Fixture(name="hostlib_nbr", genversion="JAX_TEST_HOSTLIB_NBR"),
    Fixture(name="hostlib_nbr_zphot_q",
            genversion="JAX_TEST_HOSTLIB_NBR_ZPHOTQ", text_ref=True),
    Fixture(name="hostlib_nbr3",
            genversion="JAX_TEST_HOSTLIB_NBR3", text_ref=True),
    Fixture(name="hostlib_mwebv", genversion="JAX_TEST_HOSTLIB_MWEBV"),
    Fixture(name="hostlib_zphot_q",
            genversion="JAX_TEST_HOSTLIB_ZPHOTQ", text_ref=True),
    Fixture(name="host_spectra_replay", genversion="JAX_TEST_HOST_SPECTRA",
            text_ref=True),
    Fixture(name="host_spectra_contam", genversion="JAX_TEST_HOST_CONTAM"),
    Fixture(name="host_spectra_specdata", genversion="JAX_TEST_HOST_SPECDATA"),
    Fixture(name="hostlib_wgtmap_x1mass",
            genversion="JAX_TEST_HOSTLIB_WGTMAP_X1MASS", head=False, phot=False),
    Fixture(name="fluxerr_nominal", genversion="JAX_TEST_FLUXERR_NOMINAL", head=False, phot=False),
    Fixture(name="fluxerr_flat", genversion="JAX_TEST_FLUXERR_FLAT", head=False, phot=False),
    Fixture(name="salt3_fit_photoz", genversion="JAX_TEST_SALT3_FIT_PHOTOZ",
            fit_nml="fit.nml"),
    Fixture(name="salt3_fit_photoz_diagcov",
            genversion="JAX_TEST_SALT3_FIT_PHOTOZ_DIAGCOV", fit_nml="fit.nml"),
    Fixture(name="searcheff_mixed_spec_zhost",
            genversion="JAX_TEST_SEARCHEFF_MIXED"),
    Fixture(name="non1ased_v19_replay",
            genversion="JAX_TEST_NON1ASED_V19_REPLAY", text_ref=True),
    Fixture(name="non1ased_hostdust", genversion="JAX_TEST_NON1ASED_HOSTDUST"),
    Fixture(name="mwdust_parity", genversion="JAX_TEST_MWDUST_PARITY"),
    Fixture(name="mwdust_somm25", genversion="JAX_TEST_MWDUST_SOMM25"),
    Fixture(name="mwdust_somm26", genversion="JAX_TEST_MWDUST_SOMM26", replay_events=300),
    Fixture(name="bayesn_fit_dense",
            genversion="JAX_TEST_BAYESN_FIT_DENSE",
            fit_nml="fit.nml"),
    Fixture(name="simsed_91bg_sequential",
            genversion="JAX_TEST_SIMSED_91BG_SEQ", text_ref=True),
    Fixture(name="simsed_91bg_replay",
            genversion="JAX_TEST_SIMSED_91BG", text_ref=True),
    Fixture(name="cutwin_parity", genversion="JAX_TEST_CUTWIN_PARITY"),
    Fixture(name="mwdust_opt1", genversion="JAX_TEST_MWDUST_OPT1",
            text_ref=True),
    Fixture(name="dndz_powerlaw2", genversion="JAX_TEST_DNDZ_POWERLAW2", head=False, phot=False),
    Fixture(name="wgtmap_color", genversion="JAX_TEST_WGTMAP_COLOR", head=False, phot=False),
    Fixture(name="vpec_smear", genversion="JAX_TEST_VPEC_SMEAR", head=False, phot=False),
    Fixture(name="hostlib_mskopt_alt",
            genversion="JAX_TEST_HOSTLIB_MSKOPT_ALT"),
    Fixture(name="bayesn_rv_2p5", genversion="JAX_TEST_BAYESN_RV_2P5", replay_events=300),
    Fixture(name="peakmjd_fmax_clump",
            genversion="JAX_TEST_PEAKMJD_FMAX", head=False, phot=False),
    Fixture(name="salt3_fit_photoz_quantile",
            genversion="JAX_TEST_SALT3_FIT_PHOTOZ",
            fit_nml="fit.nml", sim_from="salt3_fit_photoz"),
    Fixture(name="salt3_fit_photoz_steffen",
            genversion="JAX_TEST_SALT3_FIT_PHOTOZ",
            fit_nml="fit.nml", sim_from="salt3_fit_photoz"),
    Fixture(name="salt3_fit_headover", genversion="JAX_TEST_SALT3_FIT_PHOTOZ",
            fit_nml="fit.nml", sim_from="salt3_fit_photoz"),
    Fixture(name="salt3_fit_headover_qz", genversion="JAX_TEST_SALT3_FIT_PHOTOZ",
            fit_nml="fit.nml", sim_from="salt3_fit_photoz"),
    Fixture(name="salt3_fit_headover_logmass", genversion="JAX_TEST_SALT3_FIT_PHOTOZ",
            fit_nml="fit.nml", sim_from="salt3_fit_photoz"),
    Fixture(name="salt3_fit_headover_logmass_pz", genversion="JAX_TEST_SALT3_FIT_PHOTOZ",
            fit_nml="fit.nml", sim_from="salt3_fit_photoz"),
    Fixture(name="des_salt3_replay",
            genversion="JAX_TEST_DES_SALT3_REPLAY"),
    Fixture(name="salt3_hostdust_replay",
            genversion="JAX_TEST_SALT3_HOSTDUST"),
    Fixture(name="wgtmap_3d", genversion="JAX_TEST_WGTMAP_3D", head=False, phot=False),
    Fixture(name="dndz_md14", genversion="JAX_TEST_DNDZ_MD14", head=False, phot=False),
    Fixture(name="cutwin_mwebv", genversion="JAX_TEST_CUTWIN_MWEBV", head=False, phot=False),
    Fixture(name="searcheff_spec_scale_replay",
            genversion="JAX_TEST_SPEC_SCALE_REPLAY"),
    Fixture(name="salt3_latetime_replay",
            genversion="JAX_TEST_SALT3_LATETIME", replay_events=300),
    Fixture(name="smear_coh_replay",
            genversion="JAX_TEST_SMEAR_COH", replay_events=300),
    Fixture(name="genpdf_smear", genversion="JAX_TEST_GENPDF_SMEAR", replay_events=300),
    Fixture(name="g10_smear_ranfix",
            genversion="JAX_TEST_G10_RANFIX"),
    Fixture(name="c11_smear_ranfix",
            genversion="JAX_TEST_C11_RANFIX"),
    Fixture(name="fudge_knobs_replay",
            genversion="JAX_TEST_FUDGE_KNOBS"),
    Fixture(name="cutwin_opt3_replay",
            genversion="JAX_TEST_CUTWIN_OPT3"),
    Fixture(name="minobs_search_replay",
            genversion="JAX_TEST_MINOBS_SEARCH"),
    Fixture(name="mjd_template_replay",
            genversion="JAX_TEST_MJD_TEMPLATE"),
    Fixture(name="fudge_noise_replay",
            genversion="JAX_TEST_FUDGE_NOISE"),
    Fixture(name="fudge_noise_nea_replay",
            genversion="JAX_TEST_FUDGE_NOISE_NEA"),
    Fixture(name="zvariation_ebvigm",
            genversion="JAX_TEST_ZVAR_EBVIGM"),
    Fixture(name="zvariation_popshift",
            genversion="JAX_TEST_ZVAR_POP"),
    Fixture(name="spectro_optmask_replay",
            genversion="JAX_TEST_SPECTRO_OPTMASK"),
    Fixture(name="spectro_snrmode_replay",
            genversion="JAX_TEST_SPECTRO_SNRMODE"),
    Fixture(name="spectro_bits_replay",
            genversion="JAX_TEST_SPECTRO_BITS"),
    Fixture(name="spectro_mjdstep_replay",
            genversion="JAX_TEST_SPECTRO_MJDSTEP"),
    Fixture(name="simlib_spectro_replay",
            genversion="JAX_TEST_SIMLIB_SPECTRO"),
    Fixture(name="hostlib_sn2galz_bayesn",
            genversion="JAX_TEST_HOSTLIB_SN2GALZ_BAYESN", replay_events=300),
    Fixture(name="hostlib_sn2galz_vpec", genversion="JAX_TEST_HOSTLIB_SN2GALZ_VPEC", phot=False),
    Fixture(name="hostlib_sn2galz_wgtmap", genversion="JAX_TEST_HOSTLIB_SN2GALZ_WGTMAP", phot=False),
    Fixture(name="hostlib_sn2galradec", genversion="JAX_TEST_HOSTLIB_SN2GALRADEC", phot=False),
    Fixture(name="hostlib_usevpec_sn2galz", genversion="JAX_TEST_HOSTLIB_USEVPEC_SN2GALZ", phot=False),
    Fixture(name="hostlib_wronghost", genversion="JAX_TEST_HOSTLIB_WRONGHOST", phot=False),
    Fixture(name="atmosphere_replay",
            genversion="JAX_TEST_ATMOSPHERE_REPLAY"),
    # CC sim behind the BBC BEAMS prior; bbc_biascor_fit/build_contam_data.py
    # reduces its FITRES to CCPRIOR.FITRES.
    Fixture(name="bbc_ccprior_sim", genversion="JAX_TEST_BBC_CCPRIOR",
            fit_nml="fit.nml"),
    # several GENTYPEs with their type names in the README
    Fixture(name="gentype_map_k10", genversion="JAX_TEST_GENTYPE_MAP"),
    Fixture(name="bbc_des_data", genversion="JAX_TEST_BBC_DES_DATA",
            fit_nml="fit.nml"),
    Fixture(name="bbc_des_biascor", genversion="JAX_TEST_BBC_DES_BIASCOR",
            fit_nml="fit.nml"),
    # BayeSN on the DES cadence; both are fitted by jax's snlc_fit.py, not snlc_fit.exe
    Fixture(name="bbc_des_bayesn_data",
            genversion="JAX_TEST_BBC_DES_BAYESN_DATA", fit_nml="fit.nml",
            fit_with_jax=True, head=False, phot=False),
    Fixture(name="bbc_des_bayesn_biascor",
            genversion="JAX_TEST_BBC_DES_BAYESN_BIASCOR", fit_nml="fit.nml",
            fit_with_jax=True, head=False, phot=False),
    Fixture(name="nskipmjd_replay",
            genversion="JAX_TEST_NSKIPMJD"),
    Fixture(name="mxradius_replay",
            genversion="JAX_TEST_MXRADIUS"),
    Fixture(name="spectrum_prescale_replay",
            genversion="JAX_TEST_SPECPS"),
    Fixture(name="spectro_snrscale_replay",
            genversion="JAX_TEST_SNRSCALE"),
    Fixture(name="magmonitor_snr",
            genversion="JAX_TEST_MAGMON"),
    Fixture(name="magmonitor_compact",
            genversion="JAX_TEST_MAGMON_COMPACT"),
    Fixture(name="weaklens_probmap",
            genversion="JAX_TEST_WEAKLENS_PROBMAP"),
    Fixture(name="setpkmjd_fmaxclump",
            genversion="JAX_TEST_SETPKMJD_FMAXCLUMP"),
    Fixture(name="epcutwin_snrmin", genversion="JAX_TEST_EPCUTWIN_SNRMIN"),
    Fixture(name="hostlib_sbradius", genversion="JAX_TEST_HOSTLIB_SBRADIUS"),
]


# --------------------------------------------------------------------------- #
# SNANA version handling
# --------------------------------------------------------------------------- #

def get_snana_version() -> str:
    # SNANA prints "SNANA_VERSION:" then exits 11 because it tries to
    # read "--version" as an input file. Capture stdout regardless of code.
    result = subprocess.run(
        ["snlc_sim.exe", "--version"], capture_output=True, text=True
    )
    m = re.search(r"SNANA_VERSION:\s*(\S+)", result.stdout)
    if not m:
        raise RuntimeError(
            f"Could not parse SNANA version from:\n{result.stdout}\n{result.stderr}"
        )
    return m.group(1)


def check_pinned_version(force_bump: bool) -> str:
    pinned = VERSION_PIN.read_text().strip()
    actual = get_snana_version()
    if actual == pinned:
        return actual
    if force_bump:
        VERSION_PIN.write_text(actual + "\n")
        print(f"[version] bumped pin: {pinned} -> {actual}")
        return actual
    sys.exit(
        f"\nERROR: SNANA version mismatch.\n"
        f"  pinned (tests/fixtures/snana_version.txt): {pinned}\n"
        f"  actual (snlc_sim.exe --version):           {actual}\n"
        f"\n"
        f"Refresh the pin only after auditing the diff:\n"
        f"  python tests/fixtures/regenerate.py --all --force-version-bump\n"
    )


# --------------------------------------------------------------------------- #
# Per-fixture regen
# --------------------------------------------------------------------------- #

def gzip_fixture(path: Path) -> Path:
    """Store fixture recordings as deterministic gzip without changing their bytes."""
    compressed = path.with_suffix(path.suffix + ".gz")
    compressed.write_bytes(gzip.compress(path.read_bytes(), compresslevel=6, mtime=0))
    path.unlink()
    return compressed


def trim_replay_photometry(directory: Path, n_events: int) -> None:
    """Keep photometry for the first distinct DUMP CIDs, retaining the full DUMP."""
    with (directory / "DUMP").open() as stream:
        cids = []
        for line in stream:
            if line.startswith("VARNAMES:"):
                cid_column = line.split()[1:].index("CID")
            elif line.startswith("SN:"):
                cids.append(int(line.split()[1:][cid_column]))
    selected = set(list(dict.fromkeys(cids))[:n_events])
    head_path, phot_path = directory / "HEAD.FITS", directory / "PHOT.FITS"
    with fits.open(head_path) as head, fits.open(phot_path) as phot:
        keep = np.isin(np.asarray(head[1].data["SNID"], dtype=int), list(selected))
        rows = head[1].data.view(np.ndarray)[keep].copy()
        segments = []
        offset = 0
        for row in rows:
            start, stop = int(row["PTROBS_MIN"]) - 1, int(row["PTROBS_MAX"])
            # SNANA stores one end-of-event marker after each event's epochs.
            if phot[1].data["MJD"][stop] != -777:
                raise ValueError(f"{phot_path}: missing end marker for {row['SNID']}")
            segment = phot[1].data.view(np.ndarray)[start:stop + 1].copy()
            row["PTROBS_MIN"], row["PTROBS_MAX"] = offset + 1, offset + stop - start
            segments.append(segment)
            offset += len(segment)
        head[1].data = rows
        phot[1].data = np.concatenate(segments)
        head.writeto(head_path, overwrite=True)
        phot.writeto(phot_path, overwrite=True)


def prepare_snana_assets(destination: Path) -> Path:
    """Stage plain assets for SNANA readers that cannot reopen gzip streams."""
    shutil.copytree(DATA_ROOT, destination)
    for compressed in destination.rglob("*.gz"):
        with gzip.open(compressed, "rb") as source, compressed.with_suffix("").open("wb") as target:
            shutil.copyfileobj(source, target)
        compressed.unlink()
    return destination


def regenerate_fixture(fx: Fixture, snana_version: str, *, out_dir: Path | None = None) -> Path:
    """Run SNANA on ``fx`` and capture outputs into ``out_dir`` (default: in-tree).

    Returns the directory written to (so callers can compare against the
    in-tree copy).
    """
    if out_dir is None:
        out_dir = fx.root
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    fit_provenance = ""

    # Run SNANA from a temp working dir so its log files / leftovers don't
    # pollute the fixture tree.
    with tempfile.TemporaryDirectory(prefix=f"snana_{fx.name}_") as work_str:
        work = Path(work_str)

        # SNANA records the expanded root in its output (snlc_sim.c:25237);
        # a staged tree in the run cwd keeps the generating checkout out of it.
        prepare_snana_assets(work / "fixtures")
        env["JAX_TEST_FIXTURE_ROOT"] = "fixtures"

        def run_sim(log_name: str, args: list[str]) -> None:
            log = work / log_name
            with log.open("w") as log_fp:
                res = subprocess.run(
                    ["snlc_sim.exe", str(fx.input_file), *args],
                    cwd=work, env=env, stdout=log_fp, stderr=subprocess.STDOUT,
                )
            if res.returncode != 0:
                sys.stderr.write(f"\nSNANA failed for {fx.name}; tail of log:\n")
                sys.stderr.write("\n".join(log.read_text().splitlines()[-40:]))
                sys.stderr.write("\n")
                raise SystemExit(res.returncode)

        def locate_sim_out() -> Path:
            # SNANA writes outputs under $SNANA_LSST_SIM/<GENVERSION>/ or
            # $SNDATA_ROOT/SIM/<GENVERSION>/ depending on env. Search both.
            candidates = [
                Path(env[k]) / fx.genversion
                for k in ("SNANA_LSST_SIM", "SNDATA_ROOT")
                if k in env
            ]
            candidates.append(Path(env.get("SNDATA_ROOT", "")) / "SIM" / fx.genversion)
            found = next((c for c in candidates if c.is_dir()), None)
            if found is None:
                raise FileNotFoundError(
                    f"No SNANA output dir found for {fx.genversion!r}; "
                    f"checked: {candidates}"
                )
            return found

        # TEXT pass first, harvested before the FITS run reuses the same
        # GENVERSION dir. The fit stage below then still sees FITS.
        n_dat = 0
        if fx.text_ref:
            run_sim(f"{fx.name}.text.log", ["FORMAT_MASK", "2"])
            dats = sorted(locate_sim_out().glob("*.DAT"))
            if not dats:
                raise FileNotFoundError(
                    f"FORMAT_MASK=2 produced no .DAT for {fx.name}")
            text_dir = out_dir / "text_ref"
            if text_dir.is_dir():
                shutil.rmtree(text_dir)
            text_dir.mkdir(parents=True)
            # A full run writes one .DAT per SN — thousands. The contract this
            # references is per-file, so bundle the same first N the existing
            # salt3_k21_replay reference carries, named by bare SN id.
            for src in dats[:TEXT_REF_MAX_FILES]:
                stem = re.search(r"(SN\d+\.DAT)$", src.name)
                if stem is None:
                    raise ValueError(f"unexpected .DAT name: {src.name}")
                shutil.copy(src, text_dir / stem.group(1))
            n_dat = min(len(dats), TEXT_REF_MAX_FILES)

        run_sim(f"{fx.name}.log", [])
        sim_out = locate_sim_out()

        # Copy outputs into out_dir
        copied: list[str] = []
        bundled = {".DUMP": "DUMP", "_HEAD.FITS": "HEAD.FITS",
                   "_PHOT.FITS": "PHOT.FITS", "_SPEC.FITS": "SPEC.FITS",
                   ".README": "SIM.README"}
        if not fx.head:
            del bundled["_HEAD.FITS"]
        if not fx.phot:
            del bundled["_PHOT.FITS"]
        for src in sim_out.iterdir():
            stem = next((s for tail, s in bundled.items()
                         if src.name.endswith(tail)), None)
            if stem is not None and fx.sim_from is None:
                dst = out_dir / stem
                shutil.copy(src, dst)
                copied.append(stem)

        verification = work / f"{fx.genversion}_VERFIY_SED_TRUE.DAT"
        if verification.is_file():
            shutil.copy(verification, out_dir / "VERIFY_SED_TRUE.DAT")
            copied.append("VERIFY_SED_TRUE.DAT")

        if fx.replay_events is not None:
            trim_replay_photometry(out_dir, fx.replay_events)

        for i, name in enumerate(copied):
            path = out_dir / name
            if name.endswith(".FITS") or (name == "DUMP" and path.stat().st_size >= 100_000):
                copied[i] = gzip_fixture(path).name

        # Optional second stage: fit the simulated photometry and bundle the tables.
        # Move sim output to a private dir that's NOT on PATH_SNDATA_SIM.LIST,
        # then point PRIVATE_DATA_PATH at it so SNANA finds exactly one copy.
        if fx.fit_nml_file is not None:
            if not fx.fit_nml_file.is_file():
                raise FileNotFoundError(f"fit_nml not found: {fx.fit_nml_file}")
            private_root = work / "private_sim"
            private_root.mkdir()
            shutil.move(str(sim_out), str(private_root / fx.genversion))
            # a bundled <version>.IGNORE sits beside the data, where the
            # fitter's default EPOCH_IGNORE_FILE looks (snana.F90:12881)
            for ignore in fx.root.glob("*.IGNORE"):
                shutil.copy(ignore, private_root / fx.genversion / ignore.name)

            fit_work = work / "fit"
            fit_work.mkdir()
            (fit_work / "fixtures").symlink_to(work / "fixtures")
            fit_env = env.copy()
            fit_env["JAX_TEST_FIT_SIM_DIR"] = str(private_root)
            fit_log = fit_work / f"{fx.name}.fit.log"
            if fx.fit_with_jax:
                repo = HERE.parents[1]
                fit_cmd = [sys.executable, str(repo / "snlc_fit.py"), str(fx.fit_nml_file)]
                revision = subprocess.check_output(
                    ["git", "describe", "--always", "--dirty"], cwd=repo, text=True).strip()
                fit_provenance = f"JAX fitter revision: `{revision}`\n"
            else:
                fit_cmd = ["snlc_fit.exe", str(fx.fit_nml_file)]
            fit_provenance += f"Fit command: `{shlex.join(fit_cmd)}`\n"
            fit_provenance += ("Fit environment: `JAX_TEST_FIXTURE_ROOT=fixtures`, "
                               "`JAX_TEST_FIT_SIM_DIR` points to the private simulation directory.\n")
            with fit_log.open("w") as log_fp:
                fit_result = subprocess.run(
                    fit_cmd, cwd=fit_work, env=fit_env,
                    stdout=log_fp, stderr=subprocess.STDOUT,
                )
            if fit_result.returncode != 0:
                sys.stderr.write(f"\nFit failed for {fx.name}; tail of log:\n")
                sys.stderr.write("\n".join(fit_log.read_text().splitlines()[-40:]))
                sys.stderr.write("\n")
                raise SystemExit(fit_result.returncode)
            if out_dir != fx.root:      # a scratch regeneration keeps the fit log
                shutil.copy(fit_log, out_dir / "fit.log")
            # bundle every table the fit wrote (FITRES.TEXT, SNANA.TEXT, ...)
            text_outputs = sorted(fit_work.glob("*.TEXT"))
            if not any(p.name.endswith(".FITRES.TEXT") for p in text_outputs):
                raise FileNotFoundError(f"Fit produced no FITRES.TEXT in {fit_work}")
            for src in text_outputs:
                table = src.name.split(".", 1)[1]  # <PREFIX>.<TABLE>.TEXT
                dst = out_dir / table
                shutil.copy(src, dst)
                if dst.stat().st_size >= 100_000:
                    dst = gzip_fixture(dst)
                else:
                    dst.with_suffix(dst.suffix + ".gz").unlink(missing_ok=True)
                copied.append(dst.name)

    # Provenance README (in-tree only)
    if out_dir == fx.root:
        readme = (
            f"# Fixture: {fx.name}\n\n"
            f"Generated: {dt.date.today().isoformat()}\n"
            f"SNANA version: {snana_version}\n"
            f"Command: `snlc_sim.exe {fx.input_file.parent.name}/input.input` "
            f"(run from a scratch dir with `JAX_TEST_FIXTURE_ROOT=fixtures`, "
            f"a decompressed copy of `tests/fixtures/data`)\n"
            + (f"TEXT reference: `snlc_sim.exe input.input FORMAT_MASK 2`, "
               f"same env; .DAT copied to `text_ref/`\n" if fx.text_ref else "")
            + fit_provenance
            + "\n"
            f"GENVERSION: `{fx.genversion}`\n\n"
            f"Files:\n"
            + "".join(f"- {name} ({(out_dir / name).stat().st_size / 1024:.1f} KB)\n"
                      for name in copied)
            + (f"- text_ref/ ({n_dat} .DAT from FORMAT_MASK=2)\n" if n_dat else "")
            + (f"\nPhotometry: first {fx.replay_events} distinct DUMP CIDs; "
               "full DUMP retained.\n" if fx.replay_events else "")
            + (f"\nShared simulation: `../{fx.sim_from}`.\n" if fx.sim_from else "")
            + ("\nHEAD is not bundled.\n" if not fx.head else "")
            + ("\nPHOT is not bundled.\n" if not fx.phot else "")
            + "\nRefresh with `python tests/fixtures/regenerate.py --only "
            f"{fx.name}`.\n"
        )
        (fx.root / "README.md").write_text(readme)

    print(f"[{fx.name}] wrote {len(copied)} files to {out_dir}")
    return out_dir


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def cmd_list() -> None:
    for fx in FIXTURES:
        print(fx.name)


def cmd_all(snana_version: str) -> None:
    for fx in FIXTURES:
        regenerate_fixture(fx, snana_version)


def cmd_only(name: str, snana_version: str) -> None:
    for fx in FIXTURES:
        if fx.name == name:
            regenerate_fixture(fx, snana_version)
            return
    sys.exit(f"unknown fixture {name!r}; see --list")


def _same_column(u, v) -> bool:
    if u.dtype.kind in "fiu":
        return np.array_equal(np.nan_to_num(u), np.nan_to_num(v))
    return bool((u == v).all())


def _dump_rows(path: Path) -> list[str]:
    """The DUMP data rows; its header lines carry paths that always differ."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return [ln.rstrip("\n") for ln in stream if ln.startswith("SN:")]


def _diff_dump(live: Path, bundled: Path) -> str | None:
    new_rows = _dump_rows(live)
    old_rows = _dump_rows(bundled)
    if new_rows == old_rows:
        return None
    if len(new_rows) != len(old_rows):
        return f"{len(new_rows)} rows vs {len(old_rows)}"
    ndiff = sum(a != b for a, b in zip(new_rows, old_rows))
    return f"{ndiff} of {len(new_rows)} rows differ"


def _tforms(hdu) -> dict[str, str]:
    return {c.name: c.format for c in hdu.columns}


def _diff_fits(live: Path, bundled: Path) -> str | None:
    """Layout first (primary header keys, columns, TFORMs), then values."""
    notes = []
    with fits.open(live) as new_hdus, fits.open(bundled) as old_hdus:
        new_keys = set(new_hdus[0].header) - {"DATE"}
        old_keys = set(old_hdus[0].header) - {"DATE"}
        if new_keys != old_keys:
            notes.append(f"header keys added {sorted(new_keys - old_keys)}, "
                         f"removed {sorted(old_keys - new_keys)}")
        for new, old in zip(new_hdus[1:], old_hdus[1:]):
            if new.data is None or old.data is None:
                continue
            new_tf, old_tf = _tforms(new), _tforms(old)
            added = [f"{n}:{new_tf[n]}" for n in new.data.names if n not in old_tf]
            removed = [f"{n}:{old_tf[n]}" for n in old.data.names if n not in new_tf]
            retyped = [f"{n}:{old_tf[n]}->{new_tf[n]}" for n in new.data.names
                       if n in old_tf and new_tf[n] != old_tf[n]]
            if added or removed or retyped:
                notes.append(f"columns added {added}, removed {removed}, "
                             f"retyped {retyped}")
            if new.data.shape != old.data.shape:
                notes.append(f"shape {new.data.shape} vs {old.data.shape}")
                continue
            # values of the columns both layouts share
            changed = [name for name in new.data.names
                       if name in old_tf
                       and not _same_column(new.data[name], old.data[name])]
            if changed:
                notes.append(f"{len(changed)} shared columns differ: {changed[:8]}")
    return "; ".join(notes) or None


def _text_rows(path: Path) -> tuple[list[str], list[str]]:
    """A FITRES-style table's VARNAMES columns and its data rows."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        lines = stream.read().splitlines()
    varnames = next((ln.split()[1:] for ln in lines if ln.startswith("VARNAMES:")), [])
    rows = [ln for ln in lines if ln.startswith(("SN:", "ROW:", "OBS:"))]
    return varnames, rows


def _diff_text_table(live: Path, bundled: Path) -> str | None:
    new_vars, new_rows = _text_rows(live)
    old_vars, old_rows = _text_rows(bundled)
    notes = []
    if new_vars != old_vars:
        notes.append(f"columns added {[v for v in new_vars if v not in old_vars]}, "
                     f"removed {[v for v in old_vars if v not in new_vars]}")
    if len(new_rows) != len(old_rows):
        notes.append(f"{len(new_rows)} rows vs {len(old_rows)}")
        return "; ".join(notes)
    # values of the columns both tables share, as written
    new_cols = {v: [r.split()[1 + i] for r in new_rows] for i, v in enumerate(new_vars)}
    old_cols = {v: [r.split()[1 + i] for r in old_rows] for i, v in enumerate(old_vars)}
    changed = [v for v in new_vars if v in old_cols and new_cols[v] != old_cols[v]]
    if changed:
        notes.append(f"{len(changed)} shared columns differ: {changed[:8]}")
    return "; ".join(notes) or None


# README provenance and timing fields differ on every run
_README_VOLATILE = re.compile(r"DATE|VERSION|USER|HOST|PATH|FILE|DIR|_ROOT|TIME_START|CWD|CPU_MINUTES")


def _diff_readme(live: Path, bundled: Path) -> str | None:
    def keep(path):
        # Inline comments include generation rates; retain the actual counts.
        return [ln.partition("#")[0].rstrip() for ln in path.read_text().splitlines()
                if ln.partition("#")[0].strip() and not _README_VOLATILE.search(ln)]
    new, old = keep(live), keep(bundled)
    if new == old:
        return None
    gone = [ln for ln in old if ln not in new]
    extra = [ln for ln in new if ln not in old]
    return (f"{len(extra)} lines added, {len(gone)} gone; "
            f"first added {extra[:2]}, first gone {gone[:2]}")


def _find(directory: Path, name: str) -> Path | None:
    """Locate an output, allowing for the gzipped form some fixtures bundle."""
    for candidate in (directory / name, directory / f"{name}.gz"):
        if candidate.exists():
            return candidate
    return None


def _diff(live: Path, bundled: Path) -> str | None:
    """Content difference between a regenerated file and its bundled copy.

    Compares data, not bytes: FITS headers embed a DATE keyword and the DUMP
    header embeds paths, so a byte comparison flags every file as changed and
    can never show whether a fixture still reproduces.
    """
    if live.name in ("DUMP", "DUMP.gz"):
        return _diff_dump(live, bundled)
    if live.name.endswith((".TEXT", ".TEXT.gz")) or live.name == "VERIFY_SED_TRUE.DAT":
        return _diff_text_table(live, bundled)
    if live.name == "SIM.README":
        return _diff_readme(live, bundled)
    return _diff_fits(live, bundled)


_DIFF_FILES = ("DUMP", "HEAD.FITS", "PHOT.FITS", "SPEC.FITS", "SIM.README",
               "FITRES.TEXT", "SNANA.TEXT", "VERIFY_SED_TRUE.DAT")


def cmd_diff(snana_version: str, only: str | None, keep: Path | None) -> None:
    """Regenerate every fixture (or one) into a scratch dir and report content
    differences; ``keep`` names a directory that retains the regenerated files."""
    any_diff = False
    fixtures = [fx for fx in FIXTURES if only is None or fx.name == only]
    if not fixtures:
        sys.exit(f"unknown fixture {only!r}; see --list")
    for fx in fixtures:
        with tempfile.TemporaryDirectory(prefix=f"diff_{fx.name}_") as tmp:
            out = keep / fx.name if keep else Path(tmp)
            out.mkdir(parents=True, exist_ok=True)
            try:
                regenerate_fixture(fx, snana_version, out_dir=out)
            except BaseException as exc:   # a SNANA abort exits rather than raising
                print(f"[{fx.name}] REGENERATION FAILED ({type(exc).__name__})")
                any_diff = True
                continue
            for name in _DIFF_FILES:
                live, bundled = _find(out, name), _find(fx.root, name)
                if bundled is None:
                    continue
                if live is None:
                    why = "missing from the regenerated output"
                else:
                    why = _diff(live, bundled)
                if why:
                    print(f"[{fx.name}/{name}] {why}")
                    any_diff = True
    if any_diff:
        sys.exit(1)
    print(f"all {len(fixtures)} fixtures reproduce their bundled output")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--only", help="fixture name")
    parser.add_argument("--list", dest="list_", action="store_true")
    parser.add_argument("--diff", action="store_true")
    parser.add_argument("--keep", type=Path,
                        help="with --diff: directory that retains the regenerated files")
    parser.add_argument("--force-version-bump", action="store_true")
    args = parser.parse_args()

    if args.list_:
        cmd_list()
        return

    snana_version = check_pinned_version(args.force_version_bump)

    if args.diff:
        cmd_diff(snana_version, args.only, args.keep)
    elif args.only:
        cmd_only(args.only, snana_version)
    elif args.all:
        cmd_all(snana_version)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
