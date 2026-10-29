"""Build the bundled minimal test data tree.

Mixed sources, picked per asset:

- SIMLIB — hand-crafted toy: 2 LIBIDs, 6 LSST filters, ~50 obs each over a
  1-year window. Observation conditions (PSF/ZP/SKY) are inherently
  abstract, so a hand-picked toy is fine.
- NON1ASED — cropped from `$SNDATA_ROOT/models/NON1ASED/NON1ASED.V19_CC_noHostXT`:
  1 real template (INDEX 834, SN2008fq IIn) with phase grid sub-sampled by
  stride 4 and wavelength grid by stride 5. Real model preserves SNANA's
  flux normalisation conventions — no synthetic-SED guesswork.
- kcor — verbatim copy of the LSST baseline kcor FITS (binary multi-HDU,
  not feasible to hand-craft).

Run from the repo root with `$SNANA_LSST_ROOT` and `$SNDATA_ROOT` set:

    python tests/fixtures/build_toy_data.py

Numerical assets of at least 100 KB are stored as deterministic gzip.
"""

import argparse
import gzip
import os
import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from astropy.io import fits


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


# =============================================================================
# Toy LSST SIMLIB
# =============================================================================

def _format_simlib_obs(
    mjd: float, idexpt: int, flt: str,
    *,
    nexpose: int,
    gain: float, noise: float, skysig: float,
    psf1: float, psf2: float, psf_ratio: float,
    zptavg: float, zpterr: float, mag: float,
) -> str:
    return (
        f"S: {mjd:9.3f}  {idexpt:>5d}*{nexpose} {flt:1s}  "
        f"{gain:.2f}  {noise:.2f}  {skysig:5.2f} "
        f"{psf1:.2f} {psf2:.2f} {psf_ratio:.3f} "
        f" {zptavg:6.3f}  {zpterr:.3f} {mag:6.3f}"
    )


# Per-filter typical sky/ZP values (rough LSST orders of magnitude;
# values are intentionally simple and not from a real ETC).
_FILTER_PARAMS = {
    "u": {"skysig": 5.0,  "zpt": 28.5, "psf1": 1.10},
    "g": {"skysig": 6.0,  "zpt": 30.5, "psf1": 1.05},
    "r": {"skysig": 8.0,  "zpt": 30.5, "psf1": 1.00},
    "i": {"skysig": 10.0, "zpt": 30.0, "psf1": 1.00},
    "z": {"skysig": 12.0, "zpt": 29.5, "psf1": 1.05},
    "Y": {"skysig": 14.0, "zpt": 28.5, "psf1": 1.10},
}


_NEXPOSE_CYCLE = (1, 2, 4)  # gives APPLY_DETECT_SINGLE something to act on


def _build_libid_obs(libid_seed: int, *, n_obs: int = 50,
                     mjd_min: float = 60540.0, mjd_max: float = 61460.0) -> list[str]:
    """``n_obs`` cadenced visits cycling through ugrizY.

    NEXPOSE cycles through (1, 2, 4) so APPLY_DETECT_SINGLE has a non-trivial
    effect (per-visit eff = 1 - (1-EFF_single)^NEXPOSE for NEXPOSE>1).
    """
    rng = np.random.default_rng(libid_seed)
    mjds = np.linspace(mjd_min, mjd_max, n_obs)
    rows = []
    for i, mjd in enumerate(mjds):
        flt = "ugrizY"[i % 6]
        params = _FILTER_PARAMS[flt]
        nexpose = _NEXPOSE_CYCLE[i % len(_NEXPOSE_CYCLE)]
        rows.append(_format_simlib_obs(
            mjd=float(mjd), idexpt=1000 + i, flt=flt,
            nexpose=nexpose,
            gain=1.0, noise=0.30,
            skysig=params["skysig"] * float(1.0 + 0.05 * rng.standard_normal()),
            psf1=params["psf1"], psf2=0.0, psf_ratio=0.0,
            zptavg=params["zpt"] + float(0.02 * rng.standard_normal()),
            zpterr=0.005,
            mag=99.0,
        ))
    return rows


def _write_simlib_file(dst: Path, *, n_obs: int, note: str, mwebv: float = 0.02,
                       entry_mwebv: dict[int, float] | None = None,
                       global_header: tuple[str, ...] = (),
                       entry_header: dict[int, list[str]] | None = None,
                       append: dict[int, tuple[int, int, int]] | None = None) -> None:
    """``entry_header`` adds header lines to a LIBID; ``append`` moves a LIBID's
    rows ``first:first+n`` behind an ``APPEND: <flag>`` line at its end, as
    ``{libid: (flag, first, n)}``; ``entry_mwebv`` gives a LIBID its own
    sightline in place of ``mwebv``."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    entry_header = entry_header or {}
    entry_mwebv = entry_mwebv or {}
    append = append or {}
    libids = [
        {"libid": 0, "ra": 10.0,  "dec": -30.0, "field": "TOY_A"},
        {"libid": 1, "ra": 180.0, "dec": -45.0, "field": "TOY_B"},
    ]
    out: list[str] = [
        "DOCUMENTATION:",
        "    PURPOSE: Toy LSST SIMLIB for jax-snana parity tests",
        "    INTENT:  Test only — values are synthetic and not from any real OpSim",
        "    USAGE_KEY: SIMLIB_FILE",
        "    USAGE_CODE: snlc_sim.exe",
        "    NOTES:",
        *[f"        {line}" for line in note.splitlines()],
        "DOCUMENTATION_END:",
        "",
        "",
        "SURVEY: LSST    FILTERS: ugrizY    TELESCOPE: LSST",
        f"NLIBID: {len(libids)}",
        "PSF_UNIT: PIXEL_SIGMA",
        "SKYSIG_UNIT: ADU_PER_SQPIXEL",
        "NPE_PIXEL_SATURATE:   100000",
        "PHOTFLAG_SATURATE:    1024",
        *global_header,
        "",
        "BEGIN LIBGEN",
        "",
    ]
    for entry in libids:
        libid = entry["libid"]
        obs = _build_libid_obs(libid_seed=libid + 1, n_obs=n_obs)
        if libid in append:
            flag, first, n = append[libid]
            obs = (obs[:first] + obs[first + n:] + [f"APPEND: {flag}"]
                   + obs[first:first + n])
        out += [
            "# " + "-" * 50,
            f"LIBID: {libid}",
            f"RA: {entry['ra']:.6f}  DEC: {entry['dec']:.6f}  "
            f"NOBS: {n_obs}  MWEBV: {entry_mwebv.get(libid, mwebv):.4f}  "
            f"PIXSIZE: 0.200  FIELD: {entry['field']}",
            *entry_header.get(libid, []),
            "",
            "#                           CCD  CCD         PSF1 PSF2 PSF2/1                    ",
            "#     MJD      IDEXPT  FLT GAIN NOISE SKYSIG (pixels)  RATIO  ZPTAVG ZPTERR  MAG ",
            *obs,
            f"END_LIBID: {libid}",
            "",
        ]
    out.append("END_OF_SIMLIB:")
    dst.write_text("\n".join(out) + "\n")
    print(f"[simlib] wrote {dst.name} ({dst.stat().st_size / 1024:.1f} KB, "
          f"{len(libids)} LIBIDs × {n_obs} obs)")


def write_simlib() -> None:
    _write_simlib_file(
        DATA / "simlib" / "JAX_TEST_LSST.SIMLIB",
        n_obs=50,
        note="Two LIBIDs, 50 obs each across MJD [60540, 61460], cycling ugrizY.",
    )
    _write_simlib_file(
        DATA / "simlib" / "JAX_TEST_LSST_DENSE.SIMLIB",
        n_obs=300,
        note="Dense cadence — 300 obs per LIBID, ~3-day spacing per filter.",
    )
    _write_simlib_file(
        DATA / "simlib" / "JAX_TEST_LSST_DUSTY.SIMLIB",
        n_obs=50,
        note="Two pointings with differential MWEBV (LIBID 0 = 0.02, LIBID 1 = 0.13)\n"
             "— exercises both the MW dust extinction path (mwdust_parity, mwdust_opt1\n"
             "replay fixtures) and CUTWIN_MWEBV gating (cutwin_mwebv test that asks\n"
             "OPT_MWEBV=1 with one pointing inside [0.0, 0.10] and one outside).",
        entry_mwebv={0: 0.02, 1: 0.13},
    )
    _write_simlib_file(
        DATA / "simlib" / "JAX_TEST_LSST_WINDOWS.SIMLIB",
        n_obs=50,
        note="Per-LIBID header keys: tiling CUTWIN_REDSHIFT windows, a "
             "GENRANGE_REDSHIFT, a GENRANGE/GENSIGMA_PEAKMJD, SUBSURVEY, APPEND.",
        global_header=("SUBSURVEY_LIST: SDSS",),
        entry_header={
            0: ["CUTWIN_REDSHIFT: 0.02 0.20", "GENRANGE_PEAKMJD: 61300 61400",
                "GENSIGMA_PEAKMJD: 40", "SUBSURVEY: SDSS"],
            1: ["CUTWIN_REDSHIFT: 0.20 0.40", "GENRANGE_REDSHIFT: 0.25 0.35"],
        },
        append={0: (16384, 45, 5)},
    )


# =============================================================================
# NON1ASED model — crop a real V19 template
# =============================================================================

def crop_non1ased(
    src_dir: Path | None = None,
    dst_dir: Path = DATA / "models" / "NON1ASED.V19_TEST",
    keep_index: int = 834,
    phase_stride: int = 4,
    wave_stride: int = 5,
) -> None:
    """Subset NON1ASED.V19_CC_noHostXT to a single template; crop its grid.

    Default: INDEX 834 (SN2008fq, IIn), kept at every 4th phase × every 5th
    wavelength → ~30 phases × 376 waves → ~50 KB gzipped.
    """
    if src_dir is None:
        sndata_root = os.environ.get("SNDATA_ROOT")
        if not sndata_root:
            raise SystemExit("SNDATA_ROOT not set; cannot crop NON1ASED")
        src_dir = Path(sndata_root) / "models" / "NON1ASED" / "NON1ASED.V19_CC_noHostXT"

    dst_dir.mkdir(parents=True, exist_ok=True)

    # Find the template filename for keep_index in the source NON1A.LIST.
    src_list = (src_dir / "NON1A.LIST").read_text().splitlines()
    template_filename = None
    template_type = None
    for line in src_list:
        s = line.strip()
        if not s.startswith("NON1A:"):
            continue
        parts = s.split()
        if len(parts) >= 4 and int(parts[1]) == keep_index:
            template_type = parts[2]
            template_filename = parts[3]
            break
    if template_filename is None:
        raise ValueError(f"INDEX {keep_index} not found in {src_dir}/NON1A.LIST")

    # Read + crop the SED.
    src_sed_gz = src_dir / f"{template_filename}.gz"
    with gzip.open(src_sed_gz, "rt") as fp:
        text = fp.read()
    header_lines: list[str] = []
    rows: list[tuple[float, float, float]] = []
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            header_lines.append(line)
            continue
        phase_s, wave_s, flux_s = line.split()
        rows.append((float(phase_s), float(wave_s), float(flux_s)))
    arr = np.array(rows)
    phases = np.unique(arr[:, 0])
    waves = np.unique(arr[:, 1])
    kept_phases = set(phases[::phase_stride])
    kept_waves = set(waves[::wave_stride])

    out_lines = list(header_lines) + [
        f"### Cropped from V19_CC_noHostXT/{template_filename} for jax-snana tests ###",
        f"### phase stride={phase_stride}, wavelength stride={wave_stride} ###",
    ]
    for phase, wave, flux in rows:
        if phase in kept_phases and wave in kept_waves:
            out_lines.append(f"{phase:7.2f} {wave:8.2f} {flux:.4E}")

    gz_dst = dst_dir / f"{template_filename}.gz"
    with gzip.open(gz_dst, "wt") as fp:
        fp.write("\n".join(out_lines) + "\n")

    list_text = (
        "# NON1ASED.V19_CC_noHostXT subset for jax-snana parity tests\n"
        f"# Single template (INDEX {keep_index}), cropped phase × wavelength grid\n"
        "\n"
        f"NON1A:  {keep_index}  {template_type}    {template_filename}\n"
    )
    (dst_dir / "NON1A.LIST").write_text(list_text)

    print(f"[non1ased] cropped INDEX {keep_index} "
          f"({len(kept_phases)} phases × {len(kept_waves)} waves) "
          f"-> {gz_dst.name} ({gz_dst.stat().st_size / 1024:.1f} KB)")


# =============================================================================
# kcor — copy real LSST kcor (FITS, not feasible to hand-craft)
# =============================================================================

def write_fluxerrmodel(
    dst: Path = DATA / "fluxerrmodel" / "JAX_TEST_FLAT.DAT",
) -> None:
    """Flat per-band ERRSCALE table — same scale at all SBMAG bins so the ratio
    FLUXCALERR_fm / FLUXCALERR_nom collapses to the injected constant per band."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    band_scales = {"u": 1.5, "g": 2.0, "r": 2.5, "i": 3.0, "z": 3.5, "Y": 4.0}
    out = [
        "DOCUMENTATION:",
        "    PURPOSE: Toy flat FLUXERRMODEL — distinct ERRSCALE per band, flat in SBMAG",
        "    USAGE_KEY: FLUXERRMODEL_FILE",
        "    USAGE_CODE: snlc_sim.exe",
        "DOCUMENTATION_END:",
        "",
    ]
    for band, scale in band_scales.items():
        out += [
            "MAPNAME: FLUXERR_SCALE",
            f"BAND:    {band}",
            "VARNAMES: PSF ERRSCALE",
            f"ROW:   0.5  {scale:.3f}",
            f"ROW:   3.0  {scale:.3f}",
            "ENDMAP:",
            "",
        ]
    dst.write_text("\n".join(out) + "\n")
    print(f"[fluxerrmodel] wrote {dst.name} ({dst.stat().st_size / 1024:.1f} KB)")


def write_hostlib(
    dst: Path = DATA / "hostlib" / "JAX_TEST.HOSTLIB",
    *,
    n_galaxies: int = 5000,
    seed: int = 17,
    quantile_levels: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                                        0.7, 0.8, 0.9, 1.0),
) -> None:
    """Write a synthetic HOSTLIB with ugrizY photometry + scalar properties.

    ``quantile_levels`` are the photo-z quantiles the catalogue declares, as
    fractions; SNANA names one ``ZPHOT_Q<nnn>`` column per level.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    z = rng.uniform(0.02, 0.5, n_galaxies)
    logmass = rng.uniform(8.5, 11.5, n_galaxies)
    log_sfr = -1.0 + 0.3 * (logmass - 10.0) + rng.normal(0, 0.4, n_galaxies)
    log_ssfr = log_sfr - logmass

    # Synthetic apparent magnitudes — bluer at higher SFR.
    base_r = 17.0 + 5.0 * np.log10(z / 0.05)
    colour = rng.normal(0.5, 0.2, n_galaxies) - 0.3 * (log_ssfr + 10.0)
    mags = {
        "u": base_r + 1.5 + colour,
        "g": base_r + 0.7 + 0.5 * colour,
        "r": base_r,
        "i": base_r - 0.3,
        "z": base_r - 0.5,
        "Y": base_r - 0.7,
    }

    # Sersic profile + position angle (used by SN-host separation).
    a_sersic = rng.uniform(0.8, 4.0, n_galaxies)
    b_over_a = rng.uniform(0.4, 1.0, n_galaxies)
    n_sersic = rng.choice([1.0, 4.0], n_galaxies, p=[0.7, 0.3])
    a_rot = rng.uniform(0.0, 180.0, n_galaxies)

    ras = [10.0, 180.0]
    decs = [-30.0, -45.0]
    field_idx = rng.integers(0, 2, n_galaxies)

    quantile_cols = [f"ZPHOT_Q{int(round(p * 100)):03d}" for p in quantile_levels]
    cols = (
        ["GALID", "ZTRUE", "ZPHOT", "ZPHOT_ERR", "RA_GAL", "DEC_GAL",
         "u_obs", "g_obs", "r_obs", "i_obs", "z_obs", "Y_obs",
         "a0_Sersic", "b0_Sersic", "n0_Sersic", "a_rot",
         "LOGMASS", "LOGMASS_OBS", "LOGMASS_ERR",
         "LOG_SFR", "LOGsSFR"]
        + quantile_cols
    )
    zphot = z + rng.normal(0, 0.03, n_galaxies)
    zphot_err = np.full(n_galaxies, 0.03)
    # Build per-galaxy quantile values from the Gaussian (zphot, zphot_err).
    from scipy.stats import norm
    zphot_q = np.stack([
        norm.ppf(p, loc=zphot, scale=zphot_err) if 0 < p < 1
        else zphot + (-3 if p == 0.0 else 3) * zphot_err
        for p in quantile_levels
    ], axis=1)
    # Enforce strictly-increasing per-galaxy via cumulative max with a small
    # spread floor (some fitter readers reject equal consecutive quantiles).
    zphot_q = np.clip(zphot_q, 0.001, 2.0)
    eps = 1e-4
    for k in range(1, zphot_q.shape[1]):
        zphot_q[:, k] = np.maximum(zphot_q[:, k], zphot_q[:, k - 1] + eps)
    rows: list[str] = []
    for i in range(n_galaxies):
        gid = 100_000 + i
        row = [
            f"{gid}",
            f"{z[i]:.4f}", f"{zphot[i]:.4f}", f"{zphot_err[i]:.4f}",
            f"{ras[field_idx[i]]:.4f}", f"{decs[field_idx[i]]:.4f}",
            f"{mags['u'][i]:.3f}", f"{mags['g'][i]:.3f}",
            f"{mags['r'][i]:.3f}", f"{mags['i'][i]:.3f}",
            f"{mags['z'][i]:.3f}", f"{mags['Y'][i]:.3f}",
            f"{a_sersic[i]:.3f}", f"{a_sersic[i] * b_over_a[i]:.3f}",
            f"{n_sersic[i]:.1f}", f"{a_rot[i]:.2f}",
            f"{logmass[i]:.3f}", f"{logmass[i]:.3f}", "0.100",
            f"{log_sfr[i]:.3f}", f"{log_ssfr[i]:.3f}",
            *(f"{zphot_q[i, k]:.4f}" for k in range(len(quantile_levels))),
        ]
        rows.append("GAL: " + " ".join(row))

    out = [
        "DOCUMENTATION:",
        "    PURPOSE: Synthetic toy HOSTLIB for jax-snana e2e tests",
        "    USAGE_KEY: HOSTLIB_FILE",
        "    USAGE_CODE: snlc_sim.exe",
        "    NOTES:",
        f"        {n_galaxies} synthetic galaxies on the toy SIMLIB pointings.",
        "DOCUMENTATION_END:",
        "",
        "VARNAMES: " + " ".join(cols),
        "",
    ] + rows
    dst.write_text("\n".join(out) + "\n")
    print(f"[hostlib] wrote {dst.name} ({dst.stat().st_size / 1024:.1f} KB, "
          f"{n_galaxies} galaxies)")


def write_nbr3_hostlib(
    src: Path = DATA / "hostlib" / "JAX_TEST_NBR.HOSTLIB",
    dst: Path = DATA / "hostlib" / "JAX_TEST_NBR3.HOSTLIB",
    *,
    seed: int = 43,
) -> None:
    """The NBR_LIST catalogue with a third galaxy in every pair.

    The source pairs rows 2k and 2k+1; the third member sits about three
    arcsec from the first at the same redshift, with its own photometry
    drawn about the pair's, so every event can report three matched hosts.
    NBR_LIST holds file row numbers, so the rows are rewritten in triplets
    and each row names the other two.
    """
    rng = np.random.default_rng(seed)
    lines = src.read_text().splitlines()
    varnames = next(l.split()[1:] for l in lines if l.startswith("VARNAMES:"))
    gals = [l.split()[1:] for l in lines if l.startswith("GAL:")]
    i_ra, i_dec = varnames.index("RA_GAL"), varnames.index("DEC_GAL")
    i_nbr, i_gid = varnames.index("NBR_LIST"), varnames.index("GALID")
    mags = [i for i, v in enumerate(varnames) if v.endswith("_obs")]
    out = [l for l in lines if not l.startswith("GAL:")]
    while out and not out[-1].strip():
        out.pop()
    out.append("")
    for k in range(0, len(gals), 2):
        first, second = gals[k], gals[k + 1]
        third = list(first)
        third[i_gid] = str(200600 + k // 2)
        third[i_ra] = f"{float(first[i_ra]) + 0.0004:.6f}"
        third[i_dec] = f"{float(first[i_dec]) - 0.0006:.6f}"
        for i in mags:
            third[i] = f"{float(first[i]) + rng.normal(0.0, 0.3):.3f}"
        base = 3 * (k // 2)
        for j, row in enumerate((first, second, third)):
            row[i_nbr] = ",".join(str(base + m) for m in range(3) if m != j)
            out.append("GAL: " + " ".join(row))
    dst.write_text("\n".join(out) + "\n")
    print(f"[hostlib] wrote {dst.name} ({dst.stat().st_size / 1024:.1f} KB)")


def write_vpec_hostlib(
    src: Path = DATA / "hostlib" / "JAX_TEST.HOSTLIB",
    dst: Path = DATA / "hostlib" / "JAX_TEST_VPEC.HOSTLIB",
    *,
    seed: int = 7,
) -> None:
    """The toy catalogue with a peculiar velocity and its error per galaxy,
    the columns HOSTLIB_MSKOPT 512 (USEVPEC) reads."""
    rng = np.random.default_rng(seed)
    lines = src.read_text().splitlines()
    n = sum(1 for l in lines if l.startswith("GAL:"))
    vpec = rng.uniform(-900.0, 900.0, n)
    vpec_err = rng.uniform(100.0, 300.0, n)
    out, i = [], 0
    for line in lines:
        if line.startswith("VARNAMES:"):
            line += " VPEC VPEC_ERR"
        elif line.startswith("GAL:"):
            line += f" {vpec[i]:.1f} {vpec_err[i]:.1f}"
            i += 1
        out.append(line)
    dst.write_text("\n".join(out) + "\n")
    print(f"[hostlib] wrote {dst.name} ({dst.stat().st_size / 1024:.1f} KB)")


def write_nbr_zphotq_hostlib(
    src: Path = DATA / "hostlib" / "JAX_TEST_NBR.HOSTLIB",
    dst: Path = DATA / "hostlib" / "JAX_TEST_NBR_ZPHOTQ.HOSTLIB",
    *,
    quantile_levels: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                                        0.7, 0.8, 0.9, 1.0),
    seed: int = 41,
) -> None:
    """The NBR_LIST catalogue with photo-z quantile columns added.

    The source pairs each galaxy with a neighbour at the same redshift, so
    its ZPHOT is redrawn here as ZTRUE + N(0, ZPHOT_ERR) per galaxy: a
    neighbour then reports a photo-z of its own, which is what makes the
    per-galaxy quantile rows distinguishable. The quantiles are
    norm.ppf(p, ZPHOT, ZPHOT_ERR), the rule write_hostlib uses.
    """
    from scipy.stats import norm

    rng = np.random.default_rng(seed)
    lines = src.read_text().splitlines()
    cols = [f"ZPHOT_Q{int(round(p * 100)):03d}" for p in quantile_levels]
    out = []
    varnames = None
    for line in lines:
        if line.startswith("VARNAMES:"):
            varnames = line.split()[1:]
            out.append(line + " " + " ".join(cols))
        elif line.startswith("GAL:"):
            values = line.split()[1:]
            row = dict(zip(varnames, values))
            zphot_err = float(row["ZPHOT_ERR"])
            zphot = max(float(row["ZTRUE"]) + rng.normal(0.0, zphot_err),
                        0.001)
            values[varnames.index("ZPHOT")] = f"{zphot:.4f}"
            q = np.clip([norm.ppf(p, loc=zphot, scale=zphot_err) if 0 < p < 1
                         else zphot + (-3 if p == 0.0 else 3) * zphot_err
                         for p in quantile_levels], 0.001, 2.0)
            # strictly increasing, as write_hostlib leaves its own
            for k in range(1, len(q)):
                q[k] = max(q[k], q[k - 1] + 1e-4)
            out.append("GAL: " + " ".join(
                values + [f"{v:.4f}" for v in q]))
        else:
            out.append(line)
    dst.write_text("\n".join(out) + "\n")
    print(f"[hostlib] wrote {dst.name} ({dst.stat().st_size / 1024:.1f} KB)")


def _write_wgtmap_2d(
    dst: Path,
    *,
    par_name: str,
    par_grid: np.ndarray,
    logmass_grid: np.ndarray,
    pref_at_lm: callable,
    sigma: float,
    note: str,
) -> None:
    """Generic 2-D WGTMAP writer: ``par`` × LOGMASS with Gaussian peak."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for v in par_grid:
        for lm in logmass_grid:
            pref = pref_at_lm(lm)
            wgt = float(np.exp(-((v - pref) ** 2) / (2.0 * sigma ** 2)))
            rows.append(f"WGT:  {v:+.3f}  {lm:.3f}   {wgt:.4f}   0.0000")
    out = [
        "DOCUMENTATION:",
        f"    PURPOSE: Toy WGTMAP coupling {par_name} to LOGMASS for jax-snana tests",
        "    USAGE_KEY: HOSTLIB_WGTMAP_FILE",
        "    USAGE_CODE: snlc_sim.exe",
        "    NOTES:",
        f"        {note}",
        "DOCUMENTATION_END:",
        "",
        f"VARNAMES_WGTMAP: {par_name} LOGMASS WGT SNMAGSHIFT",
    ] + rows
    dst.write_text("\n".join(out) + "\n")
    print(f"[wgtmap] wrote {dst.name} ({dst.stat().st_size / 1024:.1f} KB, "
          f"{len(rows)} rows)")


def _write_wgtmap_3d(dst: Path) -> None:
    """3-D WGTMAP: x1 × c × LOGMASS — exercises multi-bin SN-var machinery."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    x1_grid = np.linspace(-3.0, 3.0, 7)
    c_grid = np.linspace(-0.3, 0.5, 5)
    lm_grid = np.linspace(8.0, 12.0, 5)
    rows = []
    for x1 in x1_grid:
        for c in c_grid:
            for lm in lm_grid:
                # Joint pref: x1 anti-correlated with mass, c anti-correlated with mass.
                x1_pref = 1.0 - 0.5 * (lm - 8.0)
                c_pref = 0.10 - 0.05 * (lm - 8.0)
                wgt = 0.05 + float(np.exp(
                    -((x1 - x1_pref) ** 2) / (2.0 * 1.5 ** 2)
                    - ((c - c_pref) ** 2) / (2.0 * 0.20 ** 2)
                ))
                rows.append(f"WGT:  {x1:+.3f}  {c:+.3f}  {lm:.3f}   {wgt:.4f}   0.0000")
    out = [
        "DOCUMENTATION:",
        "    PURPOSE: 3-D WGTMAP coupling x1+c to host LOGMASS",
        "    USAGE_KEY: HOSTLIB_WGTMAP_FILE",
        "    USAGE_CODE: snlc_sim.exe",
        "    NOTES:",
        "        Multi-bin SN-var WGTMAP — x1 and c are SN vars, LOGMASS is host.",
        "DOCUMENTATION_END:",
        "",
        "VARNAMES_WGTMAP: x1 c LOGMASS WGT SNMAGSHIFT",
    ] + rows
    dst.write_text("\n".join(out) + "\n")
    print(f"[wgtmap] wrote {dst.name} ({dst.stat().st_size / 1024:.1f} KB, "
          f"{len(rows)} rows)")


def write_wgtmap() -> None:
    """Three WGTMAPs covering different SN-var axis counts."""
    logmass = np.linspace(8.0, 12.0, 9)
    _write_wgtmap_2d(
        DATA / "wgtmap" / "JAX_TEST_X1_LOGMASS.WGTMAP",
        par_name="x1", par_grid=np.linspace(-3.0, 3.0, 13),
        logmass_grid=logmass,
        pref_at_lm=lambda lm: 1.0 - 0.5 * (lm - 8.0),
        sigma=0.8,
        note="x1_pref decreases with LOGMASS — emulates the mass-step.",
    )
    _write_wgtmap_2d(
        DATA / "wgtmap" / "JAX_TEST_COLOR_LOGMASS.WGTMAP",
        par_name="c", par_grid=np.linspace(-0.4, 0.6, 11),
        logmass_grid=logmass,
        pref_at_lm=lambda lm: 0.10 - 0.05 * (lm - 8.0),
        sigma=0.10,
        note="c_pref decreases with LOGMASS — exercises 2nd WGTMAP variable axis.",
    )
    _write_wgtmap_3d(DATA / "wgtmap" / "JAX_TEST_X1_C_LOGMASS.WGTMAP")


# =============================================================================
# Synthetic SFD98 dust maps
# =============================================================================

def write_dust_maps() -> None:
    """Stand-in SFD98 E(B-V) maps, one per galactic hemisphere.

    E(B-V) values are invented, not SFD98 extinction. The real 4096x4096
    maps are 64 MB each and ship with SNANA; what the reader needs is the
    Lambert projection and some spatial structure, so the stub carries a
    256x256 map on the same projection. It keeps the `_4096_` filename
    because `SFDDustMap` hardcodes it.

    Structure rises steeply towards the galactic plane (rho -> 1) and is
    modulated in longitude, so two positions on the sky give different
    E(B-V) -- which is what the SN2GAL_RADEC test needs.
    """
    n = 256
    scale = n / 2
    crpix = n / 2 + 0.5
    py, px = np.indices((n, n), dtype=np.float64)
    # invert lambert_lb2fpix (crval=0): pixel -> rho*cos(l), -nsgp*rho*sin(l)
    u = (px - crpix + 1.0) / scale
    v = (py - crpix + 1.0) / scale
    rho = np.clip(np.hypot(u, v), 0.0, 1.0)

    for nsgp, tag in ((1, "ngp"), (-1, "sgp")):
        gal_l = np.arctan2(-nsgp * v, u)
        ebv = 0.015 + 1.5 * rho ** 8 * (1.0 + 0.6 * np.sin(3.0 * gal_l))
        hdu = fits.PrimaryHDU(ebv.astype(np.float32))
        hdu.header["CTYPE1"] = "LAMBERT--X"
        hdu.header["CTYPE2"] = "LAMBERT--Y"
        hdu.header["LAM_NSGP"] = nsgp
        hdu.header["LAM_SCAL"] = scale
        hdu.header["CRPIX1"] = crpix
        hdu.header["CRPIX2"] = crpix
        hdu.header["CRVAL1"] = 0.0
        hdu.header["CRVAL2"] = 0.0
        dst = DATA / "sndata_root" / "MWDUST" / f"SFD_dust_4096_{tag}.fits"
        dst.parent.mkdir(parents=True, exist_ok=True)
        hdu.writeto(dst, overwrite=True)
        print(f"[mwdust] wrote {dst.name} ({dst.stat().st_size / 1024:.1f} KB, "
              f"E(B-V) {ebv.min():.4f}..{ebv.max():.4f})")


def copy_kcor() -> None:
    src_root = os.environ.get("SNANA_LSST_ROOT")
    if not src_root:
        raise SystemExit("SNANA_LSST_ROOT not set; cannot refresh kcor")
    src = Path(src_root) / "kcor" / "baseline" / "kcor_LSST.fits"
    dst = DATA / "kcor" / "JAX_TEST_LSST.fits"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    print(f"[kcor] copied {dst} ({dst.stat().st_size / 1024:.1f} KB)")


# =============================================================================
# Manifest README
# =============================================================================

def compress_assets() -> None:
    """Losslessly compress numerical assets above 100 KB."""
    for path in DATA.rglob("*"):
        if (path.is_file() and path.stat().st_size >= 100_000
                and path.suffix.lower() in (".fits", ".dat", ".hostlib", ".simlib")):
            path.with_name(path.name + ".gz").write_bytes(
                gzip.compress(path.read_bytes(), compresslevel=6, mtime=0))
            path.unlink()


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    write_simlib()
    crop_non1ased()
    write_hostlib()
    write_vpec_hostlib()
    # a second catalogue with the same columns, for the quantile fixtures
    write_hostlib(dst=DATA / "hostlib" / "JAX_TEST_ZPHOTQ.HOSTLIB",
                  n_galaxies=2000, seed=23)
    write_nbr_zphotq_hostlib()
    # the triplet catalogue is bundled only with its quantile columns
    triplets = DATA / "hostlib" / "JAX_TEST_NBR3.HOSTLIB"
    write_nbr3_hostlib(dst=triplets)
    write_nbr_zphotq_hostlib(src=triplets,
                             dst=DATA / "hostlib" / "JAX_TEST_NBR3_ZPHOTQ.HOSTLIB")
    triplets.unlink()
    write_wgtmap()
    write_fluxerrmodel()
    write_dust_maps()
    copy_kcor()
    compress_assets()
    total = sum(f.stat().st_size for f in DATA.rglob("*") if f.is_file())
    print(f"\nTotal bundled data size: {total / 1024:.1f} KB ({total / (1024**2):.2f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compress", action="store_true",
                        help="compress existing assets without regenerating them")
    args = parser.parse_args()
    if args.compress:
        compress_assets()
    else:
        main()
