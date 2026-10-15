"""Live-SNANA parity for the SIMGEN_DUMP_* diagnostic dumps.

Runs snlc_sim.exe with the dump keys enabled and compares against jax:
  - .RATE / .MWCL are pure functions of the config, so they compare directly.
  - .NOISE is realisation-dependent, so jax replays SNANA's SNe (DUMP override)
    and the per-obs noise components are matched by (CID, MJD, BAND).

Requires --live-snana (SNANA installed + $SNDATA_ROOT). Runtime ~1 min.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for extra in ("tests/parity", "."):
    p = str(PROJECT_ROOT / extra)
    if p not in sys.path:
        sys.path.insert(0, p)

from jax_snlc_core.io.snana_dump_reader import load_snana_dump  # noqa: E402
from jax_snlc_core.io.fits_path import resolve_fits_path  # noqa: E402
from replay_snana_dump import build_override_parameters  # noqa: E402
from jax_snlc_sim import run as sim_run  # noqa: E402

_FIXTURE = "salt3_k21_replay"


def _run_snana(extra_lines, genversion, workdir, replace=()):
    """Write the salt3 fixture input + extra lines, run SNANA, return
    (input, out_dir). ``replace`` names keys the extra lines set again,
    dropped from the fixture first since SNANA aborts on a repeated key."""
    src = (PROJECT_ROOT / "tests" / "fixtures" / "snana" / _FIXTURE
           / "input.input").read_text()
    s = re.sub(r"GENVERSION:\s+\S+", f"GENVERSION:  {genversion}", src)
    s = re.sub(r"NGENTOT_LC:\s+\d+", "NGENTOT_LC:  100", s)
    for key in replace:
        s = re.sub(rf"^\s*{key}:.*$", "", s, flags=re.M)
    s += "\n" + extra_lines + "\n"
    inp = workdir / "input.input"
    inp.write_text(s)
    env = os.environ.copy()
    r = subprocess.run(["snlc_sim.exe", str(inp)], cwd=workdir, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]
    for base in (Path(env.get("SNANA_LSST_SIM", "")) / genversion,
                 Path(env.get("SNDATA_ROOT", "")) / "SIM" / genversion):
        if base.is_dir():
            return inp, base
    raise FileNotFoundError(f"no sim output for {genversion}")


def _run_jax(inp, out_dir, genversion, override=None):
    if override is None:
        config = dict(
            n_sne=100, random_seed=12345)
        sim = sim_run.prepare(str(inp), **config)
        simulation = None if sim is None else sim.config
        resources = sim
        res, sel, spectra_cache = sim_run.run_simulation(sim)
    else:
        dump, head_cols, n = override
        config = dict(n_sne=n)
        sim = sim_run.prepare(str(inp), **config)
        simulation = None if sim is None else sim.config
        resources = sim
        ov = build_override_parameters(dump, simulation, resources, head_columns=head_cols)
        res, sel, spectra_cache = sim_run.run_simulation(sim, override_parameters=ov)
    sim_run.save_snana_outputs(sim, res, genversion, sel, spectra_cache=spectra_cache, output_dir=str(out_dir))
    return out_dir / genversion


def _rows(path, prefix):
    return [l.split() for l in Path(path).read_text().splitlines()
            if l.startswith(prefix)]


def test_rate_mwcl_live(tmp_path):
    """.RATE is byte-exact; .MWCL is row-exact with float32-level values."""
    gv = "JAX_LIVE_RM"
    inp, sdir = _run_snana("SIMGEN_DUMP_RATE: 1\nSIMGEN_DUMP_MWCL: 1", gv, tmp_path)
    jdir = _run_jax(inp, tmp_path / "jax", gv)

    srate, jrate = _rows(sdir / f"{gv}.RATE", "ROW:"), _rows(jdir / f"{gv}.RATE", "ROW:")
    assert len(srate) == len(jrate) and srate == jrate            # byte-exact

    smwcl, jmwcl = _rows(sdir / f"{gv}.MWCL", "ROW:"), _rows(jdir / f"{gv}.MWCL", "ROW:")
    assert len(smwcl) == len(jmwcl) > 0
    maxd = max(abs(float(a[k]) - float(b[k]))
               for a, b in zip(smwcl, jmwcl) for k in range(2, len(a)))
    assert maxd < 0.005                                           # float32 dust-law grid


def test_noise_replay_live(tmp_path):
    """.NOISE per-obs components match (ZP/NEA/COV_SKY/COV_READ byte-exact;
    COV_TOT to 1%; no-host MAG_GAL=666, GALSNSEP=0)."""
    gv = "JAX_LIVE_NOISE"
    inp, sdir = _run_snana("SIMGEN_DUMP_NOISE: 2", gv, tmp_path)
    dump = load_snana_dump(str(next(sdir.glob("*.DUMP"))), dedup_cid=True)
    with fits.open(next(sdir.glob("*HEAD.FITS*"))) as h:
        head_cols = {n: h[1].data[n] for n in h[1].data.names}
    jdir = _run_jax(inp, tmp_path / "jax", gv, override=(dump.columns, head_cols, dump.size))

    def load(path):
        d = {}
        for t in _rows(path, "SN:"):
            # CID GALID LIBID MJD BAND ZCMB MAG_SN MAG_GAL GALSNSEP ZP_pe NEA
            # COV_TOT COV_SN COV_GAL COV_SKY COV_READ
            d[(int(t[1]), round(float(t[4]), 2), t[5])] = [float(x) for x in t[7:]]
        return d

    S, J = load(sdir / f"{gv}.NOISE"), load(jdir / f"{gv}.NOISE")
    common = set(S) & set(J)
    assert len(common) > 200
    idx = {"mag_gal": 1, "galsnsep": 2, "zp": 3, "nea": 4,
           "cov_tot": 5, "cov_sn": 6, "cov_gal": 7, "cov_sky": 8, "cov_read": 9}
    for col in ("zp", "nea", "cov_sky", "cov_read", "mag_gal", "galsnsep", "cov_gal"):
        assert max(abs(S[k][idx[col]] - J[k][idx[col]]) for k in common) == 0.0, col
    # COV_TOT within 1%; COV_SN within 2% for the flux-dominated bright epochs
    assert max(abs(S[k][idx["cov_tot"]] - J[k][idx["cov_tot"]])
               / max(1.0, S[k][idx["cov_tot"]]) for k in common) < 0.01
    bright = [k for k in common if S[k][0] < 24]  # mag_sn < 24
    assert max(abs(S[k][idx["cov_sn"]] - J[k][idx["cov_sn"]])
               / max(1.0, S[k][idx["cov_sn"]]) for k in bright) < 0.02


def _photflag_by_sn(head, phot, bit):
    """{CID: sorted MJDs whose PHOTFLAG has `bit`} (SNRMAX/NEARPEAK flag one epoch)."""
    with fits.open(resolve_fits_path(head)) as h:
        hd = h[1].data
        pmin, pmax, cid = hd['PTROBS_MIN'], hd['PTROBS_MAX'], hd['SNID']
    with fits.open(resolve_fits_path(phot)) as h:
        pf, mjd = h[1].data['PHOTFLAG'], h[1].data['MJD']
    out = {}
    for i in range(len(pmin)):
        seg = slice(int(pmin[i]) - 1, int(pmax[i]))
        out[int(str(cid[i]).strip())] = sorted(
            float(m) for m, f in zip(mjd[seg], pf[seg]) if int(f) & bit)
    return out


def test_trest_statistics_live(tmp_path):
    """TRESTMIN/TRESTMAX/TGAPMAX against snlc_sim.exe with Trest cut windows
    narrower than GENRANGE_TREST, so epochs exist outside the windows.
    SNANA fills TRESTMIN/TRESTMAX over every generated epoch, in the CMB
    frame (snlc_sim.c:20499, :20770), then widens them over the filtered
    set in the heliocentric frame (:24239); TGAPMAX is over the filtered
    set alone. jax keeps the heliocentric frame throughout, and with
    these windows the generated-epoch extreme sets the value, so the two
    part by |zHEL - zCMB|/(1+z) of the phase: up to 0.0013 of 60 days. TGAPMAX
    is a difference of two phases in one frame and agrees at the float32-MJD
    level."""
    gv = "JAX_LIVE_TREST"
    inp, sdir = _run_snana(
        "GENRANGE_TREST: -20 60\nCUTWIN_TRESTMIN: -8 50\nCUTWIN_TRESTMAX: -50 25\n"
        "APPLY_CUTWIN_OPT: 0\nSIMGEN_DUMPALL: 10 CID GENZ PEAKMJD LIBID "
        "SALT2x0 SALT2x1 SALT2c TRESTMIN TRESTMAX TGAPMAX",
        gv, tmp_path,
        replace=("GENRANGE_TREST", "CUTWIN_TRESTMIN", "CUTWIN_TRESTMAX",
                 "APPLY_CUTWIN_OPT", "SIMGEN_DUMPALL"))
    dump = load_snana_dump(str(next(sdir.glob("*.DUMP"))), dedup_cid=True)
    with fits.open(next(sdir.glob("*HEAD.FITS*"))) as h:
        head_cols = {n: h[1].data[n] for n in h[1].data.names}
    jdir = _run_jax(inp, tmp_path / "jax", gv, override=(dump.columns, head_cols, dump.size))
    jdump = load_snana_dump(str(next(jdir.glob("*.DUMP"))), dedup_cid=True)
    order = {int(c): i for i, c in enumerate(jdump.columns["CID"])}
    rows = [order[int(c)] for c in dump.columns["CID"]]
    assert len(rows) == dump.size
    for col in ("TRESTMIN", "TRESTMAX", "TGAPMAX"):
        snana = np.asarray(dump.columns[col], dtype=float)
        jax_ = np.asarray(jdump.columns[col], dtype=float)[rows]
        # the narrow windows must leave epochs outside them for the check
        # to mean anything
        if col == "TRESTMIN":
            assert snana.min() < -8.0
        atol = 5e-3 if col == "TGAPMAX" else 5e-3 + 1.4e-3 * np.abs(snana)
        worst = np.abs(jax_ - snana) - atol
        assert worst.max() <= 0.0, f"{col}: {worst.max():.4f} past the bound"


def test_photflag_trainsalt_yaml_live(tmp_path):
    """PHOTFLAG_SNRMAX/NEARPEAK land on the same epoch as SNANA (per SN);
    TRAINSALT PKMJD_TRUE matches; .YAML content format matches."""
    gv = "JAX_LIVE_MISC"
    inp, sdir = _run_snana(
        "PHOTFLAG_SNRMAX: 8192\nPHOTFLAG_NEARPEAK: 16384\n"
        "SIMGEN_DUMP_TRAINSALT: 1\nWRFLAG_YAML_FILE: 1", gv, tmp_path)
    dump = load_snana_dump(str(next(sdir.glob("*.DUMP"))), dedup_cid=True)
    with fits.open(next(sdir.glob("*HEAD.FITS*"))) as h:
        head_cols = {n: h[1].data[n] for n in h[1].data.names}
    # jax writes {gv}.YAML to its cwd (like SNANA); run it in a private dir so
    # its YAML does not collide with SNANA's (written to the SNANA run cwd).
    jax_cwd = tmp_path / "jax"
    jax_cwd.mkdir(exist_ok=True)
    cwd = os.getcwd()
    os.chdir(jax_cwd)
    try:
        jdir = _run_jax(inp, jax_cwd, gv, override=(dump.columns, head_cols, dump.size))
    finally:
        os.chdir(cwd)

    s_head = next(sdir.glob("*HEAD.FITS*"))
    s_phot = next(sdir.glob("*PHOT.FITS*"))
    j_head, j_phot = jdir / f"{gv}_HEAD.FITS", jdir / f"{gv}_PHOT.FITS"
    for bit in (8192, 16384):  # SNRMAX, NEARPEAK
        S = _photflag_by_sn(s_head, s_phot, bit)
        J = _photflag_by_sn(j_head, j_phot, bit)
        for cid in set(S) & set(J):
            assert len(S[cid]) == len(J[cid]), f"bit {bit} count mismatch CID {cid}"
            # same flagged epoch(s), within FITS float32 MJD precision
            for ms, mj in zip(S[cid], J[cid]):
                assert abs(ms - mj) < 0.01, f"PHOTFLAG bit {bit} epoch mismatch CID {cid}"

    # TRAINSALT: true peak matches (the smeared PKMJD is an independent RNG draw)
    def ts_true(p):
        return {int(l.split()[0]): float(l.split()[2])
                for l in open(p) if not l.startswith("#") and l.strip()}
    St, Jt = ts_true(next(sdir.glob("*.TRAINSALT"))), ts_true(jdir / f"{gv}.TRAINSALT")
    assert max(abs(St[k] - Jt[k]) for k in set(St) & set(Jt)) < 0.01

    # YAML: both write it to their run cwd (bare GENVERSION name); content parity
    s_yaml = dict(l.split(":", 1) for l in (tmp_path / f"{gv}.YAML").read_text().splitlines())
    j_yaml = dict(l.split(":", 1) for l in (jax_cwd / f"{gv}.YAML").read_text().splitlines())
    assert set(s_yaml) == set(j_yaml)
    assert s_yaml["SURVEY"].strip() == j_yaml["SURVEY"].strip()
    assert s_yaml["IDSURVEY"].strip() == j_yaml["IDSURVEY"].strip()


@pytest.mark.parametrize("fixture, simlib, filters, mjd, tag", [
    # combined RA/DEC/FIELD line, DEC spelling, ugrizY
    ("salt3_k21_replay", "JAX_TEST_LSST.SIMLIB", "ugrizY",
     ("60500", "61500"), "JAX_TEST_LSST_60500-61500.TEXT"),
    # standalone FIELD: line, DECL spelling, griz
    ("des_salt3_replay", "JAX_TEST_DES.SIMLIB", "griz",
     ("56000", "58000"), "JAX_TEST_DES_56000-58000.TEXT"),
])
def test_simlib_dump_rowvalues_live(tmp_path, fixture, simlib, filters, mjd, tag):
    """SIMLIB_DUMP AVG/OBS row values match SNANA. SNANA's dump is model-free
    (its canonical `NOFILE SIMLIB_FILE ... SIMLIB_DUMP` invocation), and jax's
    run_simlib_dump reads only the SIMLIB, so the two compare directly: AVG is
    byte-exact; OBS matches except float32-MJD-level noise in MJD-derived cols.
    Two SIMLIB layouts exercise both DEC/DECL and inline/standalone FIELD."""
    from jax_snlc_sim.config_parser import parse_snana_input_file
    from jax_snlc_sim.simlib_dump import run_simlib_dump
    fdir = PROJECT_ROOT / "tests" / "fixtures" / "data"
    env = os.environ.copy()
    env["JAX_TEST_FIXTURE_ROOT"] = str(fdir)
    r = subprocess.run(
        ["snlc_sim.exe", "NOFILE", "SIMLIB_FILE", str(fdir / "simlib" / simlib),
         "GENFILTERS", filters, "GENRANGE_MJD", mjd[0], mjd[1], "SIMLIB_DUMP", "3"],
        cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]

    # jax side: a parseable input pointing at the same SIMLIB + MJD window
    inp = tmp_path / "jax.input"
    inp.write_text(
        (PROJECT_ROOT / "tests" / "fixtures" / "snana" / fixture / "input.input")
        .read_text() + "\nSIMLIB_DUMP: 3\n")
    jdir = tmp_path / "jaxout"
    jdir.mkdir()
    prev = os.getcwd()
    os.chdir(tmp_path)
    try:
        cfg = parse_snana_input_file(str(inp))
        run_simlib_dump(cfg, output_dir=str(jdir))
    finally:
        os.chdir(prev)

    def rows(path):
        return [l.split() for l in open(path) if l.startswith("ROW:")]
    sa, ja = rows(tmp_path / f"SIMLIB_DUMP_AVG_{tag}"), rows(jdir / f"SIMLIB_DUMP_AVG_{tag}")
    so, jo = rows(tmp_path / f"SIMLIB_DUMP_OBS_{tag}"), rows(jdir / f"SIMLIB_DUMP_OBS_{tag}")

    # AVG: every column byte-exact (RA, DEC, FIELD, per-band NOBS/ZPT/PSF/M5SIG)
    assert len(sa) == len(ja) > 0
    assert sa == ja
    # OBS: string cols exact; MJD-derived floats within float32-MJD precision
    assert len(so) == len(jo) > 1
    obs_float = {4, 5, 6, 8, 9, 10, 11, 12, 13}
    for s, j in zip(so, jo):
        assert len(s) == len(j)
        for k, (x, y) in enumerate(zip(s, j)):
            if k in obs_float:
                assert abs(float(x) - float(y)) < 0.01, f"OBS col {k}"
            else:
                assert x == y, f"OBS col {k}: {x} != {y}"


def test_photflag_detect_live(tmp_path):
    """PHOTFLAG detect/trigger bits against snlc_sim.exe at APPLY_SEARCHEFF_OPT: 0.

    SNANA evaluates the pipeline whether or not the flag lets it reject
    (snlc_sim.c:1276), so detected epochs carry PHOTFLAG_DETECT and
    MJD_DETECT_* are real. jax used to skip evaluation entirely and stamp the
    detect bit on every epoch.
    """
    gv = "JAX_LIVE_PF"
    inp, snana_dir = _run_snana("", gv, tmp_path)
    jax_dir = _run_jax(inp, tmp_path / "jax", gv)

    def flags(d, tag):
        with fits.open(resolve_fits_path(str(next(Path(d).glob(f"*{tag}*"))))) as f:
            rec = f[1].data
            mjd = np.asarray(rec["MJD"], float)
            return np.asarray(rec["PHOTFLAG"])[mjd > 0]

    s, j = flags(snana_dir, "PHOT.FITS"), flags(jax_dir, "PHOT")
    # Detection is near-deterministic above the SNR threshold, so the detected
    # fraction agrees closely; the epoch counts differ (independent draws).
    for bit, name in ((4096, "DETECT"), (2048, "TRIGGER")):
        fs, fj = np.mean((s & bit) > 0), np.mean((j & bit) > 0)
        assert abs(fs - fj) < 0.05, f"{name} fraction {fs:.3f} vs {fj:.3f}"
    assert ((j & 4096) > 0).any(), "jax set no detect bits"
    assert not ((j & 4096) > 0).all(), "jax flagged every epoch as detected"
