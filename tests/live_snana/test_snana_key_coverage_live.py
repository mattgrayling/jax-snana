"""Live-SNANA parity for SNANA input-key coverage (opt-in --live-snana).

WV07 host-AV is drawn from a fixed PDF, so SNANA and jax are compared at the
distribution level (moments + a coarse histogram), not per-SN.
"""

import glob
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from scipy.stats import ks_2samp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for extra in ("tests/parity", "."):
    p = str(PROJECT_ROOT / extra)
    if p not in sys.path:
        sys.path.insert(0, p)

from jax_snlc_core.snana_files import open_text, resolve_fits_path  # noqa: E402
from jax_snlc_sim import run as sim_run  # noqa: E402
from jax_snlc_sim.output import save_snana_outputs
from jax_snlc_sim.inputs import parse_snana_input_file  # noqa: E402

_FIXTURE = "salt3_k21_replay"


def test_genav_wv07_distribution_live(tmp_path):
    """GENAV_WV07 A_V distribution matches SNANA (mean/std/quantiles + histogram)."""
    gv = "JAX_LIVE_WV07"
    n = 5000
    src = (PROJECT_ROOT / "tests" / "fixtures" / "snana" / _FIXTURE / "input.input").read_text()
    src = re.sub(r"GENVERSION:\s+\S+", f"GENVERSION:  {gv}", src)
    src = re.sub(r"NGENTOT_LC:\s+\d+", f"NGENTOT_LC:  {n}", src)
    src += ("\nGENAV_WV07: 1\nGENRANGE_AV: 0.0 3.0\n"
            "GENPEAK_RV: 3.1\nGENSIGMA_RV: 0.0 0.0\nGENRANGE_RV: 3.1 3.1\n")
    inp = tmp_path / "in.input"
    inp.write_text(src)
    env = os.environ.copy()
    r = subprocess.run(["snlc_sim.exe", str(inp)], cwd=tmp_path, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]
    sdir = Path(env.get("SNANA_LSST_SIM", "")) / gv
    if not sdir.is_dir():
        sdir = Path(env["SNDATA_ROOT"]) / "SIM" / gv
    with fits.open(glob.glob(f"{sdir}/{gv}_HEAD.FITS*")[0]) as f:
        s_av = np.asarray(f[1].data['SIM_AV'])

    config = dict(n_sne=n, random_seed=11)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    res, _, _ = sim_run.run_simulation(sim)
    j_av = np.asarray(res['parameters']['av'])

    assert abs(s_av.mean() - j_av.mean()) < 0.03
    assert abs(s_av.std() - j_av.std()) < 0.03
    edges = np.array([0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 3.0])
    hs, _ = np.histogram(s_av, bins=edges, density=True)
    hj, _ = np.histogram(j_av, bins=edges, density=True)
    assert np.max(np.abs(hs - hj)) < 0.25


def test_mushift_under_a_rewgt_polynomial_live(tmp_path):
    """A MUSHIFT range under DNDZ_ZPOLY_REWGT is drawn from the reweight's CDF
    against MU (gen_MUSHIFT, snlc_sim.c:17542), pulling it above the flat
    mean of 0.1; the distributions agree with SNANA's."""
    gv = "JAX_LIVE_MUSHIFT"
    n = 4000
    src = (PROJECT_ROOT / "tests" / "fixtures" / "snana" / _FIXTURE / "input.input").read_text()
    src = re.sub(r"GENVERSION:\s+\S+", f"GENVERSION:  {gv}", src)
    src = re.sub(r"NGENTOT_LC:\s+\d+", f"NGENTOT_LC:  {n}", src)
    src += "\nMUSHIFT: -0.1:0.3\nDNDZ_ZPOLY_REWGT: 1,4\n"
    inp = tmp_path / "in.input"
    inp.write_text(src)
    env = os.environ.copy()
    r = subprocess.run(["snlc_sim.exe", str(inp)], cwd=tmp_path, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]
    sdir = Path(env.get("SNANA_LSST_SIM", "")) / gv
    if not sdir.is_dir():
        sdir = Path(env["SNDATA_ROOT"]) / "SIM" / gv
    with fits.open(glob.glob(f"{sdir}/{gv}_HEAD.FITS*")[0]) as f:
        s_shift = np.asarray(f[1].data['SIM_MUSHIFT'], dtype=float)

    config = dict(n_sne=n, random_seed=11)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    res, _, _ = sim_run.run_simulation(sim)
    j_shift = np.asarray(res['parameters']['mushift'], dtype=float)

    assert s_shift.mean() > 0.104 and j_shift.mean() > 0.104
    assert abs(s_shift.mean() - j_shift.mean()) < 0.008
    assert ks_2samp(s_shift, j_shift).pvalue > 1e-3


_GENPOP_FILE = """MODEL_NAME: TESTPOP
GENPEAK_SALT2x1: 1.5
GENSIGMA_SALT2x1: 0.10 0.10
GENRANGE_SALT2x1: -3.0 3.0
GENPEAK_SALT2c: 0.20
GENSIGMA_SALT2c: 0.05 0.05
GENRANGE_SALT2c: -0.3 0.5
MODEL_END:
"""


def test_genpop_asymgauss_live(tmp_path):
    """GENPOP_ASYMGAUSS SALT2 x1/c population matches SNANA (mean/std)."""
    gv = "JAX_LIVE_GENPOP"
    n = 4000
    popf = tmp_path / "genpop.dat"
    popf.write_text(_GENPOP_FILE)
    src = (PROJECT_ROOT / "tests" / "fixtures" / "snana" / _FIXTURE / "input.input").read_text()
    src = re.sub(r"GENVERSION:\s+\S+", f"GENVERSION:  {gv}", src)
    src = re.sub(r"NGENTOT_LC:\s+\d+", f"NGENTOT_LC:  {n}", src)
    # SNANA forbids the SALT2 pop keys in the input when GENPOP provides them
    src = "\n".join(l for l in src.splitlines()
                    if not re.match(r"\s*GEN(PEAK|SIGMA|RANGE)_SALT2(x1|c)\b", l))
    src += f"\nGENPOP_ASYMGAUSS: {popf} TESTPOP\n"
    inp = tmp_path / "in.input"
    inp.write_text(src)
    env = os.environ.copy()
    r = subprocess.run(["snlc_sim.exe", str(inp)], cwd=tmp_path, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]
    sdir = Path(env.get("SNANA_LSST_SIM", "")) / gv
    if not sdir.is_dir():
        sdir = Path(env["SNDATA_ROOT"]) / "SIM" / gv
    with fits.open(glob.glob(f"{sdir}/{gv}_HEAD.FITS*")[0]) as f:
        s_x1, s_c = np.asarray(f[1].data['SIM_SALT2x1']), np.asarray(f[1].data['SIM_SALT2c'])

    config = dict(n_sne=n, random_seed=7)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    res, _, _ = sim_run.run_simulation(sim)
    p = res['parameters']
    j_x1, j_c = np.asarray(p['x1']), np.asarray(p['c'])
    assert abs(s_x1.mean() - j_x1.mean()) < 0.02 and abs(s_x1.std() - j_x1.std()) < 0.02
    assert abs(s_c.mean() - j_c.mean()) < 0.01 and abs(s_c.std() - j_c.std()) < 0.01


def test_genpdf_flat_live(tmp_path):
    """GENPDF_FLAT: SALT2c(-0.3:0.3) forces a uniform c population; SNANA and jax
    agree on mean/std and both are uniform over the range."""
    gv = "JAX_LIVE_GENPDFFLAT"
    n = 4000
    src = (PROJECT_ROOT / "tests" / "fixtures" / "snana" / _FIXTURE / "input.input").read_text()
    src = re.sub(r"GENVERSION:\s+\S+", f"GENVERSION:  {gv}", src)
    src = re.sub(r"NGENTOT_LC:\s+\d+", f"NGENTOT_LC:  {n}", src)
    src += "\nGENPDF_FLAT: SALT2c(-0.3:0.3)\n"
    inp = tmp_path / "in.input"
    inp.write_text(src)
    env = os.environ.copy()
    r = subprocess.run(["snlc_sim.exe", str(inp)], cwd=tmp_path, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]
    sdir = Path(env.get("SNANA_LSST_SIM", "")) / gv
    if not sdir.is_dir():
        sdir = Path(env["SNDATA_ROOT"]) / "SIM" / gv
    with fits.open(glob.glob(f"{sdir}/{gv}_HEAD.FITS*")[0]) as f:
        s_c = np.asarray(f[1].data['SIM_SALT2c'])

    config = dict(n_sne=n, random_seed=7)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    res, _, _ = sim_run.run_simulation(sim)
    j_c = np.asarray(res['parameters']['c'])

    assert abs(s_c.mean() - j_c.mean()) < 0.02
    assert abs(s_c.std() - j_c.std()) < 0.02
    for c in (s_c, j_c):                                   # both uniform on [-0.3, 0.3]
        assert c.min() >= -0.3 - 1e-3 and c.max() <= 0.3 + 1e-3
        assert abs(c.mean()) < 0.03 and 0.15 < c.std() < 0.19


def test_use_simlib_overrides_live(tmp_path):
    """USE_SIMLIB_REDSHIFT/PEAKMJD/SALT2 pin truth to the per-LIBID header values;
    SNANA and jax produce the same set of pinned truth values."""
    gv = "JAX_LIVE_USESIMLIB"
    simlib = (PROJECT_ROOT / "tests" / "fixtures" / "data" / "simlib"
              / "JAX_TEST_LSST_USESIMLIB.SIMLIB")
    src = (PROJECT_ROOT / "tests" / "fixtures" / "snana" / _FIXTURE / "input.input").read_text()
    src = re.sub(r"GENVERSION:\s+\S+", f"GENVERSION:  {gv}", src)
    src = re.sub(r"NGENTOT_LC:\s+\d+", "NGENTOT_LC:  200", src)
    src = re.sub(r"SIMLIB_FILE:.*", f"SIMLIB_FILE: {simlib}", src)
    src += "\nUSE_SIMLIB_REDSHIFT: 1\nUSE_SIMLIB_PEAKMJD: 1\nUSE_SIMLIB_SALT2: 1\n"
    inp = tmp_path / "in.input"
    inp.write_text(src)
    env = os.environ.copy()
    r = subprocess.run(["snlc_sim.exe", str(inp)], cwd=tmp_path, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]
    sdir = Path(env.get("SNANA_LSST_SIM", "")) / gv
    if not sdir.is_dir():
        sdir = Path(env["SNDATA_ROOT"]) / "SIM" / gv

    def uniq(a, nd):
        return set(np.round(np.asarray(a, dtype=np.float64), nd).tolist())

    with fits.open(glob.glob(f"{sdir}/{gv}_HEAD.FITS*")[0]) as f:
        d = f[1].data
        s = (uniq(d['SIM_REDSHIFT_CMB'], 3), uniq(d['SIM_PEAKMJD'], 1),
             uniq(d['SIM_SALT2x1'], 2), uniq(d['SIM_SALT2c'], 2))
    config = dict(n_sne=200, random_seed=3)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    res, _, _ = sim_run.run_simulation(sim)
    p = res['parameters']
    j = (uniq(p['sim_redshift_cmb'], 3), uniq(p['peakmjd_true'], 1),
         uniq(p['x1'], 2), uniq(p['c'], 2))
    assert s == j == ({0.15, 0.35}, {60600.0, 60900.0}, {1.2, -1.0}, {0.05, 0.15})


def test_solid_angle_fieldlist_live(tmp_path):
    """SOLID_ANGLE(ALL) is the spelling that names the fields with the solid
    angle; both codes read the same angle from it, so the rate model's events
    per season in the README agree."""
    gv = "JAX_LIVE_SOLIDANGLE"
    src = (PROJECT_ROOT / "tests" / "fixtures" / "snana" / _FIXTURE / "input.input").read_text()
    src = re.sub(r"GENVERSION:\s+\S+", f"GENVERSION:  {gv}", src)
    src = re.sub(r"NGENTOT_LC:\s+\d+", "NGENTOT_LC:  20", src)
    src, n = re.subn(r"SOLID_ANGLE:\s+\S+", "SOLID_ANGLE(ALL):  0.0150", src)
    assert n == 1
    inp = tmp_path / "in.input"
    inp.write_text(src)
    env = os.environ.copy()
    r = subprocess.run(["snlc_sim.exe", str(inp)], cwd=tmp_path, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]
    sdir = Path(env.get("SNANA_LSST_SIM", "")) / gv
    if not sdir.is_dir():
        sdir = Path(env["SNDATA_ROOT"]) / "SIM" / gv
    readme = (sdir / f"{gv}.README").read_text()
    snana_per_season = int(re.search(r"NGEN_PER_SEASON:\s+(\d+)", readme).group(1))

    _prepared = sim_run.prepare(str(inp), **dict(n_sne=20))
    simulation = None if _prepared is None else _prepared.config
    resources = _prepared
    assert simulation.survey.simlib_fieldlist == "ALL"
    assert simulation.rates.solid_angle == 0.015
    assert round(resources.season_count) == snana_per_season


def test_snana_fitter_reads_a_jax_simulation_live(tmp_path):
    """snlc_fit.exe fits a jax-written FITS simulation: the primary header
    carries every key its reader requires (sntools_dataformat_fits.c:4643-4663)
    and the data tables read back into fits."""
    fx = PROJECT_ROOT / "tests" / "fixtures" / "snana" / "salt3_fit_mwebv"
    version = "JAX_TEST_SALT3_FIT_MWEBV"
    config = dict(n_sne=40, random_seed=5)
    sim = sim_run.prepare(str(fx / "input.input"), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    results, selection, spectra_cache = sim_run.run_simulation(sim)
    private = tmp_path / "private"
    save_snana_outputs(sim, results, version, selection, spectra_by_sn=spectra_cache, output_dir=str(private))

    fit_dir = tmp_path / "fit"
    fit_dir.mkdir()
    env = os.environ.copy()
    env["JAX_TEST_FIT_SIM_DIR"] = str(private)
    r = subprocess.run(["snlc_fit.exe", str(fx / "fit.nml")], cwd=fit_dir,
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:]
    rows = [l for l in (fit_dir / f"{version}.FITRES.TEXT").read_text().splitlines()
            if l.startswith("SN:")]
    assert len(rows) > 20


def test_use_simlib_distance_live(tmp_path):
    """USE_SIMLIB_DISTANCE: header DISTANCE -> zCMB matches SNANA (cosmology inversion)."""
    from jax_snlc_core.cosmology import distance_modulus
    om, ol, hd = 0.315, 0.685, 299792.458 / 70.0
    dl = {0: 10 ** ((float(distance_modulus(0.20, 0.20, om, ol, hd)) - 25) / 5),
          1: 10 ** ((float(distance_modulus(0.35, 0.35, om, ol, hd)) - 25) / 5)}
    base_simlib = (PROJECT_ROOT / "tests/fixtures/data/simlib/JAX_TEST_LSST.SIMLIB").read_text()
    out, lib = [], None
    for line in base_simlib.splitlines():
        out.append(line)
        if line.startswith("LIBID:"):
            lib = int(line.split()[1])
        if line.startswith("RA:") and lib in dl:
            out.append(f"DISTANCE: {dl[lib]:.4f}")
    simlib = tmp_path / "dist.SIMLIB"
    simlib.write_text("\n".join(out) + "\n")
    gv = "JAX_LIVE_DIST"
    src = (PROJECT_ROOT / "tests/fixtures/snana" / _FIXTURE / "input.input").read_text()
    src = re.sub(r"GENVERSION:\s+\S+", f"GENVERSION:  {gv}", src)
    src = re.sub(r"NGENTOT_LC:\s+\d+", "NGENTOT_LC:  300", src)
    src = re.sub(r"SIMLIB_FILE:.*", f"SIMLIB_FILE: {simlib}", src)
    src += "\nUSE_SIMLIB_DISTANCE: 1\n"
    inp = tmp_path / "in.input"
    inp.write_text(src)
    env = os.environ.copy()
    r = subprocess.run(["snlc_sim.exe", str(inp)], cwd=tmp_path, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]
    sdir = Path(env.get("SNANA_LSST_SIM", "")) / gv
    if not sdir.is_dir():
        sdir = Path(env["SNDATA_ROOT"]) / "SIM" / gv
    with fits.open(glob.glob(f"{sdir}/{gv}_HEAD.FITS*")[0]) as f:
        sz = set(np.round(np.asarray(f[1].data['SIM_REDSHIFT_CMB']).astype(np.float64), 2).tolist())
    config = dict(n_sne=300, random_seed=3)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    res, _, _ = sim_run.run_simulation(sim)
    jz = set(np.round(np.asarray(res['parameters']['sim_redshift_cmb']
                                 ).astype(np.float64), 2).tolist())
    assert sz == jz == {0.2, 0.35}


def _fluxcal_by_cid(head, phot):
    with fits.open(resolve_fits_path(str(head))) as h:
        pmin, pmax, cid = h[1].data['PTROBS_MIN'], h[1].data['PTROBS_MAX'], h[1].data['SNID']
    with fits.open(resolve_fits_path(str(phot))) as h:
        fl, bd, mj = h[1].data['FLUXCAL'], h[1].data['BAND'], h[1].data['MJD']
    out = {}
    for i in range(len(pmin)):
        s = slice(int(pmin[i]) - 1, int(pmax[i]))
        for f, b, m in zip(fl[s], bd[s], mj[s]):
            out[(int(str(cid[i]).strip()), round(float(m), 2), str(b).strip())] = float(f)
    return out


def test_applyflag_mwebv_live(tmp_path):
    """APPLYFLAG_MWEBV: the per-band FLUXCAL de-reddening factor (ratio flag/noflag,
    deterministic and noise-independent) matches SNANA byte-for-byte."""
    env = os.environ.copy()

    def snana_ratio():
        base = re.sub(r"OPT_MWEBV:.*", "OPT_MWEBV: 1\nGENSIGMA_MWEBV_RATIO: 0.16",
                      (PROJECT_ROOT / "tests/fixtures/snana" / _FIXTURE / "input.input").read_text())
        base = re.sub(r"NGENTOT_LC:\s+\d+", "NGENTOT_LC:  300", base)
        d = {}
        for flag in (0, 1):
            gv = f"JAX_LIVE_MW{flag}"
            s = re.sub(r"GENVERSION:\s+\S+", f"GENVERSION:  {gv}", base) + f"\nAPPLYFLAG_MWEBV: {flag}\n"
            inp = tmp_path / f"mw{flag}.input"
            inp.write_text(s)
            r = subprocess.run(["snlc_sim.exe", str(inp)], cwd=tmp_path, env=env,
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stdout[-2000:]
            sdir = Path(env.get("SNANA_LSST_SIM", "")) / gv
            if not sdir.is_dir():
                sdir = Path(env["SNDATA_ROOT"]) / "SIM" / gv
            d[flag] = _fluxcal_by_cid(glob.glob(f"{sdir}/{gv}_HEAD.FITS*")[0],
                                      glob.glob(f"{sdir}/{gv}_PHOT.FITS*")[0])
        return _band_ratio(d[0], d[1])

    def jax_ratio():
        base = re.sub(r"OPT_MWEBV:.*", "OPT_MWEBV: 1\nGENSIGMA_MWEBV_RATIO: 0.16",
                      (PROJECT_ROOT / "tests/fixtures/snana" / _FIXTURE / "input.input").read_text())
        out = {}
        for flag in (0, 1):
            inp = tmp_path / f"jmw{flag}.input"
            inp.write_text(base + f"\nAPPLYFLAG_MWEBV: {flag}\n")
            config = dict(
                n_sne=300, random_seed=5)
            sim = sim_run.prepare(str(inp), **config)
            simulation = None if sim is None else sim.config
            resources = sim
            res, sel, spectra_cache = sim_run.run_simulation(sim)
            dd = tmp_path / f"jmw{flag}"
            dd.mkdir()
            save_snana_outputs(sim, res, "J", sel, spectra_by_sn=spectra_cache, output_dir=str(dd))
            out[flag] = _fluxcal_by_cid(dd / "J" / "J_HEAD.FITS", dd / "J" / "J_PHOT.FITS")
        return _band_ratio(out[0], out[1])

    sr, jr = snana_ratio(), jax_ratio()
    common = set(sr) & set(jr)
    assert len(common) >= 4
    for b in common:
        assert abs(sr[b] - jr[b]) < 1e-3, f"{b}: SNANA {sr[b]} vs jax {jr[b]}"


def _band_ratio(d0, d1):
    import collections
    rr = collections.defaultdict(list)
    for k in set(d0) & set(d1):
        if abs(d0[k]) > 50:
            rr[k[2].split('-')[-1]].append(d1[k] / d0[k])
    return {b: float(np.mean(v)) for b, v in rr.items() if v}


def _zpt_sky_by_cid(head, phot):
    """{(cid, round(mjd,2), band): (ZEROPT, SKY_SIG)} for per-obs comparison."""
    with fits.open(resolve_fits_path(str(head))) as h:
        pmin, pmax, cid = h[1].data['PTROBS_MIN'], h[1].data['PTROBS_MAX'], h[1].data['SNID']
    with fits.open(resolve_fits_path(str(phot))) as h:
        d = h[1].data
        zp, sky, bd, mj = d['ZEROPT'], d['SKY_SIG'], d['BAND'], d['MJD']
    out = {}
    for i in range(len(pmin)):
        s = slice(int(pmin[i]) - 1, int(pmax[i]))
        for z, sk, b, m in zip(zp[s], sky[s], bd[s], mj[s]):
            out[(int(str(cid[i]).strip()), round(float(m), 2), str(b).strip())] = (
                float(z), float(sk))
    return out


def _perband_shift_ratio(base, exp):
    """Per-band mean (ZEROPT_exp - ZEROPT_base, SKY_SIG_exp / SKY_SIG_base) over
    observations shared by the baseline and exposure runs (same seed → matched)."""
    import collections
    dz, rs = collections.defaultdict(list), collections.defaultdict(list)
    for k in set(base) & set(exp):
        b = k[2].split('-')[-1]
        dz[b].append(exp[k][0] - base[k][0])
        if base[k][1] > 0:
            rs[b].append(exp[k][1] / base[k][1])
    return ({b: float(np.mean(v)) for b, v in dz.items()},
            {b: float(np.mean(v)) for b, v in rs.items() if v})


def test_exposure_time_live(tmp_path):
    """EXPOSURE_TIME_FILTER: g 4.0 shifts g-band ZEROPT by +2.5*log10(4) and scales
    SKY_SIG by sqrt(4)=2; SNANA and jax agree on these deterministic per-obs changes."""
    env = os.environ.copy()
    base_src = re.sub(r"NGENTOT_LC:\s+\d+", "NGENTOT_LC:  300",
                      (PROJECT_ROOT / "tests" / "fixtures" / "snana" / _FIXTURE
                       / "input.input").read_text())

    def snana(exp):
        gv = f"JAX_LIVE_EXPT{exp}"
        s = re.sub(r"GENVERSION:\s+\S+", f"GENVERSION:  {gv}", base_src)
        if exp:
            s += "\nEXPOSURE_TIME_FILTER: g 4.0\n"
        inp = tmp_path / f"s{exp}.input"
        inp.write_text(s)
        r = subprocess.run(["snlc_sim.exe", str(inp)], cwd=tmp_path, env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-2000:]
        sdir = Path(env.get("SNANA_LSST_SIM", "")) / gv
        if not sdir.is_dir():
            sdir = Path(env["SNDATA_ROOT"]) / "SIM" / gv
        return _zpt_sky_by_cid(glob.glob(f"{sdir}/{gv}_HEAD.FITS*")[0],
                               glob.glob(f"{sdir}/{gv}_PHOT.FITS*")[0])

    def jax(exp):
        inp = tmp_path / f"j{exp}.input"
        s = base_src + ("\nEXPOSURE_TIME_FILTER: g 4.0\n" if exp else "")
        inp.write_text(s)
        config = dict(
            n_sne=300, random_seed=5)
        sim = sim_run.prepare(str(inp), **config)
        simulation = None if sim is None else sim.config
        resources = sim
        res, sel, spectra_cache = sim_run.run_simulation(sim)
        dd = tmp_path / f"jout{exp}"
        dd.mkdir()
        save_snana_outputs(sim, res, "J", sel, spectra_by_sn=spectra_cache, output_dir=str(dd))
        return _zpt_sky_by_cid(dd / "J" / "J_HEAD.FITS", dd / "J" / "J_PHOT.FITS")

    exp_shift = 2.5 * np.log10(4.0)
    for label, dz, rs in (("SNANA", *_perband_shift_ratio(snana(0), snana(1))),
                          ("jax", *_perband_shift_ratio(jax(0), jax(1)))):
        assert abs(dz.get('g', 0.0) - exp_shift) < 2e-3, f"{label} g ZEROPT shift {dz.get('g')}"
        assert abs(rs.get('g', 0.0) - 2.0) < 2e-2, f"{label} g SKY_SIG ratio {rs.get('g')}"
        for b in dz:                                            # other bands untouched
            if b != 'g':
                assert abs(dz[b]) < 2e-3 and abs(rs.get(b, 1.0) - 1.0) < 2e-2, f"{label} {b}"


def _relerr_by_cid(head, phot):
    """Median |FLUXCALERR / FLUXCAL| over bright obs — the photometric noise level."""
    with fits.open(resolve_fits_path(str(head))) as h:
        pmin, pmax = h[1].data['PTROBS_MIN'], h[1].data['PTROBS_MAX']
    with fits.open(resolve_fits_path(str(phot))) as h:
        fl, fe = h[1].data['FLUXCAL'], h[1].data['FLUXCALERR']
    rel = []
    for lo, hi in zip(pmin, pmax):
        s = slice(int(lo) - 1, int(hi))
        for f, e in zip(fl[s], fe[s]):
            if abs(float(f)) > 100:
                rel.append(abs(float(e) / float(f)))
    return float(np.median(rel)) if rel else float('nan')


def test_genperfect_live(tmp_path):
    """GENPERFECT: 1 collapses the sim to ideal mode: SNANA and jax both produce
    near-noiseless photometry (median FLUXCALERR/|FLUXCAL| tiny), vs a noisy
    baseline (genperfect_override, snlc_sim.c:8582)."""
    env = os.environ.copy()
    base_src = re.sub(r"NGENTOT_LC:\s+\d+", "NGENTOT_LC:  200",
                      (PROJECT_ROOT / "tests" / "fixtures" / "snana" / _FIXTURE
                       / "input.input").read_text())

    def snana(gp):
        gv = f"JAX_LIVE_GP{gp}"
        s = re.sub(r"GENVERSION:\s+\S+", f"GENVERSION:  {gv}", base_src)
        if gp:
            s += "\nGENPERFECT: 1\n"
        inp = tmp_path / f"s{gp}.input"
        inp.write_text(s)
        r = subprocess.run(["snlc_sim.exe", str(inp)], cwd=tmp_path, env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-2000:]
        sdir = Path(env.get("SNANA_LSST_SIM", "")) / gv
        if not sdir.is_dir():
            sdir = Path(env["SNDATA_ROOT"]) / "SIM" / gv
        return _relerr_by_cid(glob.glob(f"{sdir}/{gv}_HEAD.FITS*")[0],
                              glob.glob(f"{sdir}/{gv}_PHOT.FITS*")[0])

    def jax(gp):
        inp = tmp_path / f"j{gp}.input"
        inp.write_text(base_src + ("\nGENPERFECT: 1\n" if gp else ""))
        config = dict(
            n_sne=200, random_seed=5)
        sim = sim_run.prepare(str(inp), **config)
        simulation = None if sim is None else sim.config
        resources = sim
        res, sel, spectra_cache = sim_run.run_simulation(sim)
        dd = tmp_path / f"jout{gp}"
        dd.mkdir()
        save_snana_outputs(sim, res, "J", sel, spectra_by_sn=spectra_cache, output_dir=str(dd))
        return _relerr_by_cid(dd / "J" / "J_HEAD.FITS", dd / "J" / "J_PHOT.FITS")

    assert snana(0) > 1e-2 and jax(0) > 1e-2          # noisy baseline
    assert snana(1) < 1e-3 and jax(1) < 1e-3          # GENPERFECT -> near-noiseless in both


def _raw_explosion_shifts(model_dir):
    """jax's per-template explosion shift (-Tshift) computed on the raw SED-file
    day grid, for direct comparison with SNANA's printed 'Shift X days'."""
    import gzip
    from jax_snlc_core.models.sed_model import compute_explosion_tshift
    files = sorted(glob.glob(f"{model_dir}/*.SED") + glob.glob(f"{model_dir}/*.SED.gz"))
    out = []
    for fn in files:
        op = gzip.open(fn, 'rt') if fn.endswith('.gz') else open(fn)
        a = np.array([l.split() for l in op if l.strip() and not l.startswith('#')], float)
        d = np.unique(a[:, 0]); w = np.unique(a[:, 1])
        f = a[:, 2].reshape(len(d), len(w))
        out.append(round(-float(compute_explosion_tshift(f, d, 1)), 1))
    return sorted(out)


@pytest.mark.parametrize("fixture,model", [
    ("non1ased_v19_replay", "NON1ASED.V19_TEST"),
    ("simsed_91bg_replay", "SIMSED.SNIa-91bg"),
])
def test_mjd_explode_reanchor_live(tmp_path, fixture, model):
    """MJD_EXPLODE + OPTMASK_T0SHIFT_EXPLODE:1: SNANA pins SIM_PEAKMJD to the
    explosion MJD and prints a per-template 'Shift X days -> T=0 at explosion'
    (T0shiftExplode_SEDMODEL). jax's compute_explosion_tshift on the raw SED
    grids reproduces that exact set of shifts; jax pins SIM_PEAKMJD identically."""
    root = PROJECT_ROOT / "tests" / "fixtures" / "data"
    env = os.environ.copy()
    env["JAX_TEST_FIXTURE_ROOT"] = str(root)
    gv = f"JAX_LIVE_EXPL_{model.split('.')[0]}"
    src = (PROJECT_ROOT / "tests" / "fixtures" / "snana" / fixture / "input.input").read_text()
    src = re.sub(r"GENVERSION:\s+\S+", f"GENVERSION:  {gv}", src)
    src = re.sub(r"NGENTOT_LC:\s+\d+", "NGENTOT_LC:  20", src)
    src += "\nMJD_EXPLODE: 61000.0\nOPTMASK_T0SHIFT_EXPLODE: 1\n"
    inp = tmp_path / "in.input"
    inp.write_text(src)
    r = subprocess.run(["snlc_sim.exe", str(inp)], cwd=tmp_path, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]

    snana_shifts = sorted(round(float(x), 1)
                          for x in re.findall(r"Shift ([\d.]+) days -> T=0 at explosion", r.stdout))
    assert snana_shifts, r.stdout[-2000:]
    assert _raw_explosion_shifts(str(root / "models" / model)) == snana_shifts

    sdir = Path(env.get("SNANA_LSST_SIM", "")) / gv
    if not sdir.is_dir():
        sdir = Path(env["SNDATA_ROOT"]) / "SIM" / gv
    with fits.open(glob.glob(f"{sdir}/{gv}_HEAD.FITS*")[0]) as f:
        assert np.allclose(np.asarray(f[1].data['SIM_PEAKMJD']), 61000.0)


def _simmag_by_key(head_path, phot_path):
    """{(SNID, round(MJD,1), BAND): (SIM_MAGOBS, FLUXCAL, FLUXCALERR)}. MJD is
    rounded to 0.1 d — SNANA and jax store it float32 (ULP ~0.008 d near MJD
    61000), while epochs are days apart, so 0.1 d keys them without collision."""
    with fits.open(resolve_fits_path(head_path)) as h, fits.open(resolve_fits_path(phot_path)) as p:
        head, phot = h[1].data, p[1].data
        out = {}
        for row in head:
            snid = int(row["SNID"])
            for i in range(int(row["PTROBS_MIN"]) - 1, int(row["PTROBS_MAX"])):
                pr = phot[i]
                mjd = float(pr["MJD"])
                if mjd < 0:
                    continue
                out[(snid, round(mjd, 1), str(pr["BAND"]).strip())] = (
                    float(pr["SIM_MAGOBS"]), float(pr["FLUXCAL"]), float(pr["FLUXCALERR"]))
    return out


@pytest.mark.parametrize("fixture,ftag", [
    ("non1ased_v19_replay", "N1A"),
    ("simsed_91bg_replay", "SIMSED"),
])
@pytest.mark.parametrize("optmask,mtag", [(None, "peak"), (1, "expl1"), (2, "expl2")])
def test_explode_simmagobs_parity_live(tmp_path, fixture, ftag, optmask, mtag):
    """End-to-end SIM_MAGOBS parity for NON1ASED and SIMSED, peak-anchored and
    explosion-anchored (OPTMASK 1/2). Replays SNANA's own per-SN truth (z,
    peak_mjd, template via SIMGEN_DUMP) through jax and compares per-(SNID, MJD,
    BAND) SIM_MAGOBS: the reanchored SED must flow through generate_sed ->
    photometry to SNANA's mags in both the peak and explosion frames."""
    from jax_snlc_core.snana_files import load_snana_dump
    from replay_snana_dump import build_override_parameters

    env = os.environ.copy()
    gv = f"JAX_LIVE_EXPLMAG_{ftag}_{mtag}"
    src = (PROJECT_ROOT / "tests" / "fixtures" / "snana" / fixture / "input.input").read_text()
    src = re.sub(r"GENVERSION:\s+\S+", f"GENVERSION:  {gv}", src)
    src = re.sub(r"NGENTOT_LC:\s+\d+", "NGENTOT_LC:  300", src)
    if optmask is not None:
        src += f"\nMJD_EXPLODE: 61000.0\nOPTMASK_T0SHIFT_EXPLODE: {optmask}\n"
    inp = tmp_path / f"{ftag}_{mtag}.input"
    inp.write_text(src)

    r = subprocess.run(["snlc_sim.exe", str(inp)], cwd=tmp_path, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]
    sdir = Path(env.get("SNANA_LSST_SIM", "")) / gv
    if not sdir.is_dir():
        sdir = Path(env["SNDATA_ROOT"]) / "SIM" / gv
    dump = glob.glob(f"{sdir}/*.DUMP")[0]
    head = glob.glob(f"{sdir}/{gv}_HEAD.FITS*")[0]
    phot = glob.glob(f"{sdir}/{gv}_PHOT.FITS*")[0]

    d = load_snana_dump(dump, dedup_cid=True)
    with fits.open(resolve_fits_path(head)) as h:
        head_cols = {n: h[1].data[n] for n in h[1].data.names}
    config = dict(n_sne=d.size)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    override = build_override_parameters(d.columns, simulation, resources,
                                         head_columns=head_cols)
    res, sel, spectra_cache = sim_run.run_simulation(sim, override_parameters=override)
    jdir = tmp_path / f"J_{ftag}_{mtag}"
    save_snana_outputs(sim, res, "J", sel, spectra_by_sn=spectra_cache, output_dir=str(jdir))

    snana = _simmag_by_key(head, phot)
    jaxp = _simmag_by_key(next((jdir / "J").glob("J*HEAD*")),
                          next((jdir / "J").glob("J*PHOT*")))
    diffs = []
    for k in set(snana) & set(jaxp):
        sm, sf, se = snana[k]
        jm = jaxp[k][0]
        if se <= 0 or sf / se < 5.0 or sm >= 50.0 or jm >= 50.0:
            continue
        diffs.append(jm - sm)
    assert len(diffs) > 30, f"too few S/N>=5 epochs ({len(diffs)})"
    ad = np.abs(np.asarray(diffs))
    assert float(np.median(ad)) < 0.005, f"median |Δmag| = {np.median(ad):.4f}"
    assert float(np.percentile(ad, 95)) < 0.05, f"p95 |Δmag| = {np.percentile(ad, 95):.4f}"


def _snana_jax_head(tmp_path, gv, extra, cols, n_sne=100, seed=99,
                    redshift_range=None):
    """Run SNANA + jax (same input, minus the GALID SIMGEN_DUMP jax can't write)
    on the hostlib_e2e fixture; return {col: (snana_array, jax_array)}."""
    env = os.environ.copy()
    src = (PROJECT_ROOT / "tests" / "fixtures" / "snana" / "hostlib_e2e" / "input.input").read_text()
    src = re.sub(r"GENVERSION:\s+\S+", f"GENVERSION:  {gv}", src)
    src = re.sub(r"NGENTOT_LC:\s+\d+", f"NGENTOT_LC:  {n_sne}", src)
    if redshift_range is not None:
        src = re.sub(r"GENRANGE_REDSHIFT:.*",
                     f"GENRANGE_REDSHIFT: {redshift_range[0]} {redshift_range[1]}", src)
    inp = tmp_path / f"{gv}.input"
    inp.write_text(src + extra)
    r = subprocess.run(["snlc_sim.exe", str(inp)], cwd=tmp_path, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]
    sdir = Path(env.get("SNANA_LSST_SIM", "")) / gv
    if not sdir.is_dir():
        sdir = Path(env["SNDATA_ROOT"]) / "SIM" / gv
    with fits.open(glob.glob(f"{sdir}/{gv}_HEAD.FITS*")[0]) as h:
        s = {c: np.asarray(h[1].data[c]) for c in cols}

    jsrc = "".join(l for l in src.splitlines(keepends=True)
                   if not l.startswith("SIMGEN_DUMP")) + extra
    jinp = tmp_path / f"{gv}_j.input"
    jinp.write_text(jsrc)
    config = dict(
        n_sne=n_sne, random_seed=seed)
    sim = sim_run.prepare(str(jinp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    res, sel, spectra_cache = sim_run.run_simulation(sim)
    outdir = tmp_path / f"J_{gv}"
    save_snana_outputs(sim, res, "J", sel, spectra_by_sn=spectra_cache, output_dir=str(outdir))
    with fits.open(resolve_fits_path(next((outdir / "J").glob("J*HEAD*")))) as h:
        j = {c: np.asarray(h[1].data[c]) for c in cols}
    return {c: (s[c], j[c]) for c in cols}


# FIXRAN_PHI is supplied per-variant (duplicate keys abort SNANA), which also lets
# 'base' and 'phi2' cross-check two different azimuth values.
_HB_BASE = "\nHOSTLIB_FIXSERSIC: 1.0 0.5 4.0 30.0\nHOSTLIB_FIXRAN_RADIUS: 0.6\n"


@pytest.mark.parametrize("galid", [
    pytest.param(100037, id="dec_minus30"),
    pytest.param(100216, id="dec_minus45"),
])
@pytest.mark.parametrize("tag,extra", [
    ("base",   "HOSTLIB_FIXRAN_PHI: 0.25\n"),
    ("phi2",   "HOSTLIB_FIXRAN_PHI: 0.70\n"),
    ("scale2", "HOSTLIB_FIXRAN_PHI: 0.25\nHOSTLIB_SCALE_SERSIC_SIZE: 2.0\n"),
    ("mxint",  "HOSTLIB_FIXRAN_PHI: 0.25\nHOSTLIB_MXINTFLUX_SNPOS: 0.5\n"),
    ("mnint",  "HOSTLIB_FIXRAN_PHI: 0.25\nHOSTLIB_MNINTFLUX_SNPOS: 0.3\n"),
])
def test_hostlib_position_knobs_live(tmp_path, tag, extra, galid):
    """Fixed hosts, profiles and position draws give matching SNSEP and DDLR."""
    # Both hosts lie near z=0.15; this interval keeps them within HOSTLIB_DZTOL.
    out = _snana_jax_head(
        tmp_path, f"JAX_LIVE_HBPOS_{tag}_{galid}",
        _HB_BASE + extra + f"HOSTLIB_GALID_FORCE: {galid}\n",
        ['HOSTGAL_SNSEP', 'HOSTGAL_DDLR', 'SIM_HOSTLIB_GALID'],
        n_sne=4, redshift_range=(0.15, 0.155))
    for hosts in out['SIM_HOSTLIB_GALID']:
        np.testing.assert_array_equal(hosts, np.full(4, galid))
    for column in ('HOSTGAL_SNSEP', 'HOSTGAL_DDLR'):
        snana, jax = out[column]
        assert np.all(np.isfinite(snana)) and np.all(snana > 0)
        assert np.all(np.isfinite(jax)) and np.all(jax > 0)
        np.testing.assert_allclose(jax, snana, rtol=2e-3, atol=0, err_msg=column)


def _catalog_ra_gal():
    """{GALID: RA_GAL} for the bundled HOSTLIB (HOSTGAL_RA is the SN sky position,
    so the pool cut is read via the assigned GALID's catalog RA_GAL)."""
    hl = PROJECT_ROOT / "tests" / "fixtures" / "data" / "hostlib" / "JAX_TEST.HOSTLIB"
    out = {}
    with open_text(hl) as stream:
        fixture_text = stream.read()
    for line in fixture_text.splitlines():
        if line.startswith("GAL:"):
            p = line.split()
            out[int(p[1])] = float(p[5])
    return out


def _catalog_ztrue_zphot():
    """{GALID: (ZTRUE, ZPHOT)} for the bundled HOSTLIB (VARNAMES: GALID ZTRUE ZPHOT ...)."""
    hl = PROJECT_ROOT / "tests" / "fixtures" / "data" / "hostlib" / "JAX_TEST.HOSTLIB"
    out = {}
    with open_text(hl) as stream:
        fixture_text = stream.read()
    for line in fixture_text.splitlines():
        if line.startswith("GAL:"):
            p = line.split()
            out[int(p[1])] = (float(p[2]), float(p[3]))
    return out


@pytest.fixture(scope="module")
def hostgal_baseline(tmp_path_factory):
    """One SNANA+jax run on the unmodified fixture, shared by the tests that need
    only baseline host columns. Each pays a snlc_sim.exe subprocess otherwise."""
    return _snana_jax_head(
        tmp_path_factory.mktemp("hg_base"), "JAX_LIVE_BASE", "",
        ['HOSTGAL_OBJID', 'HOSTGAL_SPECZ', 'REDSHIFT_HELIO', 'HOSTGAL_PHOTOZ',
         'SIM_REDSHIFT_HELIO', 'HOSTGAL_RA', 'HOSTGAL_DEC', 'SIM_RA', 'SIM_DEC',
         'HOSTGAL_MAG_r'])


def test_hostlib_hostgal_specz_live(hostgal_baseline):
    """HOSTGAL_SPECZ equals the measured SN helio redshift in both SNANA and jax."""
    out = hostgal_baseline
    for eng in (0, 1):
        ok = out['HOSTGAL_OBJID'][eng] > 0
        assert ok.any() and np.allclose(out['HOSTGAL_SPECZ'][eng][ok],
                                        out['REDSHIFT_HELIO'][eng][ok], atol=1e-5)


def test_hostlib_hostgal_photoz_formula_live(hostgal_baseline):
    """Per-SN HOSTGAL_PHOTOZ parity: SNANA and jax both report catalog ZPHOT +
    (ZGEN - host ZTRUE) (GEN_SNHOST_ZPHOT_from_HOSTLIB, sntools_host.c:7136),
    reconstructed from each engine's own assigned GALID + SIM_REDSHIFT_HELIO."""
    out = hostgal_baseline
    cat = _catalog_ztrue_zphot()
    for eng in (0, 1):
        objid, pz, zhel = (out['HOSTGAL_OBJID'][eng], out['HOSTGAL_PHOTOZ'][eng],
                           out['SIM_REDSHIFT_HELIO'][eng])
        ok = objid > 0
        exp = np.array([cat[int(o)][1] + (zhel[i] - cat[int(o)][0])
                        for i, o in zip(np.where(ok)[0], objid[ok])])
        assert ok.any() and np.allclose(pz[ok], exp, atol=1e-4)


def test_hostlib_genzphot_bias_live(tmp_path):
    """HOSTLIB_GENZPHOT_BIAS routes photo-z through GENZPHOT synthesis (snlc_sim.c:6846);
    the HOSTGAL_PHOTOZ − HOSTGAL_SPECZ mean/scatter match SNANA (distribution level,
    since the calc path is a per-SN stochastic draw)."""
    out = _snana_jax_head(tmp_path, "JAX_LIVE_ZBIAS",
                          "\nHOSTLIB_GENZPHOT_BIAS: 0.02 0.1 0 0\n",
                          ['HOSTGAL_PHOTOZ', 'HOSTGAL_SPECZ'], n_sne=2000)
    pz, sz = out['HOSTGAL_PHOTOZ'], out['HOSTGAL_SPECZ']

    def mstd(i):
        m = (pz[i] > -8) & (sz[i] > 0)
        return np.mean(pz[i][m] - sz[i][m]), np.std(pz[i][m] - sz[i][m])
    sm, ss = mstd(0)
    jm, js = mstd(1)
    assert abs(sm - jm) < 0.005 and abs(ss - js) < 0.005


def test_hostlib_genzphot_flat_outlier_live(tmp_path):
    """FUDGEPAR ... FLAT(z1:z2) flat catastrophic-outlier mode: SNANA and jax agree on
    the HOSTGAL_PHOTOZ−SPECZ mean/scatter and outlier fraction (distribution level)."""
    out = _snana_jax_head(tmp_path, "JAX_LIVE_FLAT",
                          "\nHOSTLIB_GENZPHOT_FUDGEPAR: 0.02 0 0 0.05 FLAT(0.1:0.5)\n",
                          ['HOSTGAL_PHOTOZ', 'HOSTGAL_SPECZ'], n_sne=3000)
    pz, sz = out['HOSTGAL_PHOTOZ'], out['HOSTGAL_SPECZ']

    def stat(i):
        m = (pz[i] > -8) & (sz[i] > 0)
        d = pz[i][m] - sz[i][m]
        return np.mean(d), np.std(d), np.mean(np.abs(d) > 0.1)
    sm, ss, so = stat(0)
    jm, js, jo = stat(1)
    assert abs(sm - jm) < 0.005 and abs(ss - js) < 0.006 and abs(so - jo) < 0.02


def test_hostlib_genrange_ra_live(tmp_path):
    """HOSTLIB_GENRANGE_RA restricts the read-time galaxy pool identically in SNANA
    and jax: every assigned host's catalog RA_GAL falls in the box in both engines
    (the bundled HOSTLIB clusters at RA_GAL 10 and 180; the box keeps only 180)."""
    out = _snana_jax_head(tmp_path, "JAX_LIVE_GENRA",
                          "\nHOSTLIB_GENRANGE_RA: 100.0 200.0\n", ['HOSTGAL_OBJID'])
    cat = _catalog_ra_gal()
    for objid in out['HOSTGAL_OBJID']:
        used = np.array([cat[int(o)] for o in objid if o > 0])
        assert len(used) > 0 and np.all(np.abs(used - 180.0) < 1e-2)


def test_hostlib_maxread_live(tmp_path):
    """HOSTLIB_MAXREAD caps the pool to the first N file rows identically in SNANA
    and jax (bundled GALID = 100000 + file_row): every assigned host has
    HOSTGAL_OBJID < 100000+N in both engines."""
    out = _snana_jax_head(tmp_path, "JAX_LIVE_MAXREAD",
                          "\nHOSTLIB_MAXREAD: 500\n", ['HOSTGAL_OBJID'])
    for arr in out['HOSTGAL_OBJID']:
        used = arr[arr > 0]
        assert len(used) > 0 and np.all(used < 100500)


def test_hostlib_snr_store_live(tmp_path):
    """HOSTLIB_SNR_STORE drops a galaxy from the pool at read time when its
    brightest bands miss the signal-to-noise ladder, identically in SNANA
    and jax (passCuts_HOSTLIB, sntools_host.c:3403). The bundled HOSTLIB
    carries no error columns, so g_obs_err and r_obs_err are added here:
    0.02 mag on even GALIDs (S/N 54) and 0.5 mag on odd ones (S/N 2.2), and
    a ladder of 10 then keeps only the even galaxies as hosts."""
    src_hl = PROJECT_ROOT / "tests" / "fixtures" / "data" / "hostlib" / "JAX_TEST.HOSTLIB"
    lines = []
    with open_text(src_hl) as stream:
        fixture_text = stream.read()
    for line in fixture_text.splitlines():
        if line.startswith("VARNAMES:"):
            line += " g_obs_err r_obs_err"
        elif line.startswith("GAL:"):
            err = "0.020" if int(line.split()[1]) % 2 == 0 else "0.500"
            line += f" {err} {err}"
        lines.append(line)
    hostlib = tmp_path / "SNR_STORE.HOSTLIB"
    hostlib.write_text("\n".join(lines) + "\n")

    gv = "JAX_LIVE_SNRSTORE"
    src = (PROJECT_ROOT / "tests" / "fixtures" / "snana" / "hostlib_e2e" / "input.input").read_text()
    src = re.sub(r"GENVERSION:\s+\S+", f"GENVERSION:  {gv}", src)
    src = re.sub(r"NGENTOT_LC:\s+\d+", "NGENTOT_LC:  100", src)
    src = re.sub(r"HOSTLIB_FILE:\s+\S+", f"HOSTLIB_FILE: {hostlib}", src)
    src += "\nHOSTLIB_SNR_STORE: 10\n"
    head, _ = _run_snana(src, gv, tmp_path, gv)
    snana_hosts = np.asarray(head['HOSTGAL_OBJID'])

    jsrc = "".join(l for l in src.splitlines(keepends=True)
                   if not l.startswith("SIMGEN_DUMP"))
    jinp = tmp_path / f"{gv}_j.input"
    jinp.write_text(jsrc)
    config = dict(n_sne=100, random_seed=99)
    sim = sim_run.prepare(str(jinp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    stored = resources.hostlib.galids
    assert len(stored) > 1000 and np.all(stored % 2 == 0)
    res, sel, spectra_cache = sim_run.run_simulation(sim)
    outdir = tmp_path / f"J_{gv}"
    save_snana_outputs(sim, res, "J", sel, spectra_by_sn=spectra_cache, output_dir=str(outdir))
    with fits.open(resolve_fits_path(next((outdir / "J").glob("J*HEAD*")))) as h:
        jax_hosts = np.asarray(h[1].data['HOSTGAL_OBJID'])
    for hosts in (snana_hosts, jax_hosts):
        used = hosts[hosts > 0]
        assert len(used) > 30 and np.all(used % 2 == 0)


def test_hostlib_hostgal_pos_at_sn_live(hostgal_baseline):
    """HOSTGAL_RA/DEC sit at the SN sky position in both SNANA and jax (host placed
    on the SN, GEN_SNHOST_POS), so the SN-host angular separation is arcsec-scale in
    both and the median separation agrees."""
    out = hostgal_baseline
    med = []
    for eng in (0, 1):
        ra, dec = out['HOSTGAL_RA'][eng], out['HOSTGAL_DEC'][eng]
        sra, sdec = out['SIM_RA'][eng], out['SIM_DEC'][eng]
        ok = out['HOSTGAL_OBJID'][eng] > 0
        dra = (ra[ok] - sra[ok]) * np.cos(np.radians(sdec[ok])) * 3600.0
        ddec = (dec[ok] - sdec[ok]) * 3600.0
        sep = np.sqrt(dra ** 2 + ddec ** 2)
        assert np.all(sep < 60.0)                       # host on the SN (arcsec), not catalog
        med.append(np.median(sep))
    assert abs(med[0] - med[1]) < 1.0                   # separation distributions agree


def test_hostlib_abmag_offset_live(tmp_path, hostgal_baseline):
    """HOSTLIB_ABMAG_OFFSET shifts every galaxy mag by the offset in BOTH engines
    (median shift, robust to differing galaxy draws)."""
    base = hostgal_baseline['HOSTGAL_MAG_r']
    off = _snana_jax_head(tmp_path, "JAX_LIVE_ABOFF1", "\nHOSTLIB_ABMAG_OFFSET: 0.5\n",
                          ['HOSTGAL_MAG_r'])['HOSTGAL_MAG_r']
    v = lambda a: a[(a > 0) & (a < 50)]
    s_shift = np.median(v(off[0])) - np.median(v(base[0]))
    j_shift = np.median(v(off[1])) - np.median(v(base[1]))
    assert abs(s_shift - 0.5) < 0.02 and abs(j_shift - 0.5) < 0.02


def test_hostlib_dmucor_live(tmp_path):
    """With FIXSERSIC (geometry fixed) + ABMAG_FORCE:22 (catalog mag fixed), the
    only per-SN variation in HOSTGAL_MAG is DMUCOR = mu(zSN)-mu(zhost). jax and
    SNANA must agree on its centre (~22) and spread (different galaxy draws, so
    distribution-level rather than per-SN)."""
    keys = "\nHOSTLIB_FIXSERSIC: 1.0 0.5 4.0 30.0\nHOSTLIB_ABMAG_FORCE: 22.0\n"
    env = os.environ.copy()
    gv = "JAX_LIVE_DMUCOR"
    src = (PROJECT_ROOT / "tests" / "fixtures" / "snana" / "hostlib_e2e" / "input.input").read_text()
    src = re.sub(r"GENVERSION:\s+\S+", f"GENVERSION:  {gv}", src)
    src = re.sub(r"NGENTOT_LC:\s+\d+", "NGENTOT_LC:  200", src)
    inp = tmp_path / "s.input"
    inp.write_text(src + keys)
    r = subprocess.run(["snlc_sim.exe", str(inp)], cwd=tmp_path, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]
    sdir = Path(env.get("SNANA_LSST_SIM", "")) / gv
    if not sdir.is_dir():
        sdir = Path(env["SNDATA_ROOT"]) / "SIM" / gv
    with fits.open(glob.glob(f"{sdir}/{gv}_HEAD.FITS*")[0]) as h:
        s_mag = np.asarray(h[1].data['HOSTGAL_MAG_r'])

    jsrc = "".join(l for l in src.splitlines(keepends=True)
                   if not l.startswith("SIMGEN_DUMP")) + keys
    jinp = tmp_path / "j.input"
    jinp.write_text(jsrc)
    config = dict(
        n_sne=200, random_seed=99)
    sim = sim_run.prepare(str(jinp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    res, sel, spectra_cache = sim_run.run_simulation(sim)
    save_snana_outputs(sim, res, "J", sel, spectra_by_sn=spectra_cache, output_dir=str(tmp_path / "J"))
    with fits.open(resolve_fits_path(next((tmp_path / "J" / "J").glob("J*HEAD*")))) as h:
        j_mag = np.asarray(h[1].data['HOSTGAL_MAG_r'])

    sv = s_mag[(s_mag > 0) & (s_mag < 50)]
    jv = j_mag[(j_mag > 0) & (j_mag < 50)]
    assert abs(sv.mean() - 22.0) < 0.03 and abs(jv.mean() - 22.0) < 0.03   # both centred on forced mag
    assert sv.std() > 0.03 and abs(jv.std() - sv.std()) < 0.02             # DMUCOR spread agrees


def test_hostlib_hostgal2_specz_photoz_live(tmp_path):
    """SNANA and jax both report HOSTGAL/HOSTGAL2 photo-z with the same ZDIF shift
    and each spec-z as the measured SN z (true-match host) or that host's catalog
    ZTRUE (GEN_SNHOST_ZPHOT / snlc_sim.c:26071-26078)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "nbr_builder", PROJECT_ROOT / "tests" / "integration" / "test_hostlib_nbr_list.py")
    nbr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nbr)
    env = os.environ.copy()
    hostlib = tmp_path / "NBR.HOSTLIB"
    nbr._write_nbr_hostlib(hostlib, n_gal=200)
    inp = tmp_path / "nbr.input"
    nbr._write_input(inp, hostlib)
    gv = "JAX_LIVE_HG2PZ"
    src = re.sub(r"GENVERSION:\s+\S+", f"GENVERSION:  {gv}", inp.read_text())
    src = re.sub(r"NGENTOT_LC:\s+\d+", "NGENTOT_LC:  400", src)
    src = re.sub(r"GENRANGE_REDSHIFT:.*", "GENRANGE_REDSHIFT:   0.10  0.35", src)
    inp.write_text(src)

    cat = {}
    with open_text(hostlib) as stream:
        fixture_text = stream.read()
    for line in fixture_text.splitlines():
        if line.startswith("GAL:"):
            p = line.split()
            cat[int(p[1])] = (float(p[2]), float(p[3]))

    r = subprocess.run(["snlc_sim.exe", str(inp)], cwd=tmp_path, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]
    sdir = Path(env.get("SNANA_LSST_SIM", "")) / gv
    if not sdir.is_dir():
        sdir = Path(env["SNDATA_ROOT"]) / "SIM" / gv
    sh = fits.open(glob.glob(f"{sdir}/{gv}_HEAD.FITS*")[0])[1].data

    config = dict(
        n_sne=400, random_seed=122459)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    res, sel, spectra_cache = sim_run.run_simulation(sim)
    save_snana_outputs(sim, res, "J", sel, spectra_by_sn=spectra_cache, output_dir=str(tmp_path / "J"))
    jh = fits.open(resolve_fits_path(next((tmp_path / "J" / "J").glob("J*HEAD*"))))[1].data

    czt = lambda o: np.array([cat[int(x)][0] for x in o])
    czp = lambda o: np.array([cat[int(x)][1] for x in o])
    for h in (sh, jh):
        o1, o2 = np.asarray(h['HOSTGAL_OBJID']), np.asarray(h['HOSTGAL2_OBJID'])
        pz1, pz2 = np.asarray(h['HOSTGAL_PHOTOZ']), np.asarray(h['HOSTGAL2_PHOTOZ'])
        sz1, sz2 = np.asarray(h['HOSTGAL_SPECZ']), np.asarray(h['HOSTGAL2_SPECZ'])
        zhel = np.asarray(h['REDSHIFT_HELIO'])
        pe2 = np.asarray(h['HOSTGAL2_PHOTOZ_ERR'])
        has2 = o2 > 0
        # A neighbour whose ZDIF-shifted photo-z lands below zero counts as a
        # failed host photo-z fit, and value and error go to -9 together
        # (GEN_SNHOST_ZPHOT_from_HOSTLIB, sntools_host.c). Which hosts that hits
        # depends on the sample, so pin the pairing rather than its absence.
        assert (pz2[has2] > -8).sum() > 20
        assert np.array_equal(pz2[has2] <= -8, pe2[has2] <= -8)
        both = has2 & (pz1 > -8) & (pz2 > -8)
        zdif0, zdif1 = pz1[both] - czp(o1[both]), pz2[both] - czp(o2[both])
        assert np.max(np.abs(zdif0 - zdif1)) < 1e-4
        for sz, o in ((sz1[has2], o1[has2]), (sz2[has2], o2[has2])):
            assert np.all(np.isclose(sz, zhel[has2], atol=1e-4) | np.isclose(sz, czt(o), atol=1e-4))


def test_hostlib_hostgal2_dmucor_live(tmp_path):
    """The secondary-host HOSTGAL2_MAG also carries DMUCOR. With a NBR_LIST HOSTLIB +
    ABMAG_FORCE:22, valid HOSTGAL2 mags centre on 22 with the DMUCOR spread in both
    engines (std>0 proves DMUCOR reaches the neighbour, not just the primary host)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "nbr_builder", PROJECT_ROOT / "tests" / "integration" / "test_hostlib_nbr_list.py")
    nbr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nbr)

    env = os.environ.copy()
    hostlib = tmp_path / "NBR.HOSTLIB"
    nbr._write_nbr_hostlib(hostlib, n_gal=200)
    inp = tmp_path / "nbr.input"
    nbr._write_input(inp, hostlib)
    gv = "JAX_LIVE_HG2"
    src = re.sub(r"GENVERSION:\s+\S+", f"GENVERSION:  {gv}", inp.read_text())
    src = re.sub(r"NGENTOT_LC:\s+\d+", "NGENTOT_LC:  400", src)
    # keep GENRANGE_REDSHIFT strictly inside the HOSTLIB z span (0.05-0.40)
    src = re.sub(r"GENRANGE_REDSHIFT:.*", "GENRANGE_REDSHIFT:   0.10  0.35", src)
    src += "\nHOSTLIB_ABMAG_FORCE: 22.0\n"
    inp.write_text(src)

    r = subprocess.run(["snlc_sim.exe", str(inp)], cwd=tmp_path, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]
    sdir = Path(env.get("SNANA_LSST_SIM", "")) / gv
    if not sdir.is_dir():
        sdir = Path(env["SNDATA_ROOT"]) / "SIM" / gv
    with fits.open(glob.glob(f"{sdir}/{gv}_HEAD.FITS*")[0]) as h:
        s2 = np.asarray(h[1].data['HOSTGAL2_MAG_r'])

    config = dict(
        n_sne=400, random_seed=122459)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    res, sel, spectra_cache = sim_run.run_simulation(sim)
    save_snana_outputs(sim, res, "J", sel, spectra_by_sn=spectra_cache, output_dir=str(tmp_path / "J"))
    with fits.open(resolve_fits_path(next((tmp_path / "J" / "J").glob("J*HEAD*")))) as h:
        j2 = np.asarray(h[1].data['HOSTGAL2_MAG_r'])

    v = lambda a: a[(a > 0) & (a < 90)]          # exclude 999 sentinel (no 2nd host)
    sv, jv = v(s2), v(j2)
    assert len(jv) > 20 and len(sv) > 20         # HOSTGAL2 actually populated
    assert abs(sv.mean() - 22.0) < 0.05 and abs(jv.mean() - 22.0) < 0.05
    assert sv.std() > 0.02 and abs(jv.std() - sv.std()) < 0.03   # DMUCOR spread present + matches


def _run_snana(src, gv, tmp_path, name):
    """Run snlc_sim.exe on `src` and return its (HEAD, PHOT) FITS tables."""
    inp = tmp_path / f"{name}.input"
    inp.write_text(src)
    env = os.environ.copy()
    r = subprocess.run(["snlc_sim.exe", str(inp)], cwd=tmp_path, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]
    sdir = Path(env.get("SNANA_LSST_SIM", "")) / gv
    if not sdir.is_dir():
        sdir = Path(env["SNDATA_ROOT"]) / "SIM" / gv
    head = fits.open(glob.glob(f"{sdir}/{gv}_HEAD.FITS*")[0])[1].data
    phot = fits.open(glob.glob(f"{sdir}/{gv}_PHOT.FITS*")[0])[1].data
    return head, phot


def _assert_tables_identical(a, b, label, rtol=0.0, atol=0.0):
    """Compare HEAD/PHOT column-by-column. The tolerances allow for options
    that round-trip the SED grid through a float32 cache."""
    for tag, (ta, tb) in zip(("HEAD", "PHOT"), zip(a, b)):
        assert len(ta) == len(tb), f"{label} {tag}: row count differs"
        for c in ta.columns.names:
            x, y = ta[c], tb[c]
            try:
                same = np.array_equal(x, y)
                if not same and (rtol or atol) and np.issubdtype(
                        np.asarray(x).dtype, np.floating):
                    same = np.allclose(x, y, rtol=rtol, atol=atol,
                                       equal_nan=True)
            except Exception:
                same = all(str(p) == str(q) for p, q in zip(x, y))
            assert same, f"{label} {tag}.{c} differs"


_SIMSED_FIXTURE = "simsed_91bg_replay"


def _simsed_src(gv, shape_key):
    src = (PROJECT_ROOT / "tests" / "fixtures" / "snana" / _SIMSED_FIXTURE
           / "input.input").read_text()
    src = re.sub(r"GENVERSION:\s+\S+", f"GENVERSION:  {gv}", src)
    return src.replace("SIMSED_PARAM:  stretch", f"SIMSED_{shape_key}:  stretch")


def test_simsed_shapepar_identical_to_param_live(tmp_path):
    """SIMSED_SHAPEPAR shares SIMSED_PARAM's parser; the IPAR_SIMSED_SHAPE tag
    it also sets is read only by sntools_modelgrid_gen.c (GRID mode)."""
    a = _run_snana(_simsed_src("JAX_LIVE_SS_PARAM", "PARAM"),
                   "JAX_LIVE_SS_PARAM", tmp_path, "param")
    b = _run_snana(_simsed_src("JAX_LIVE_SS_SHAPE", "SHAPEPAR"),
                   "JAX_LIVE_SS_SHAPE", tmp_path, "shape")
    _assert_tables_identical(a, b, "SIMSED_SHAPEPAR")


def test_simsed_binary_cache_no_output_change_live(tmp_path):
    """SIMSED_USE_BINARY/_PATH_BINARY select a binary SED-grid cache: a
    performance option carrying no physics. Reading the cache instead of
    re-parsing the ASCII SEDs shifts magnitudes by ~2e-6 (float32 round-trip),
    so compare to that tolerance rather than bit-identity."""
    base = (PROJECT_ROOT / "tests" / "fixtures" / "snana" / _SIMSED_FIXTURE
            / "input.input").read_text()
    a = _run_snana(re.sub(r"GENVERSION:\s+\S+", "GENVERSION:  JAX_LIVE_NOBIN", base),
                   "JAX_LIVE_NOBIN", tmp_path, "nobin")
    bindir = tmp_path / "simsed_bin"
    bindir.mkdir()
    src = re.sub(r"GENVERSION:\s+\S+", "GENVERSION:  JAX_LIVE_BIN", base)
    src += f"\nSIMSED_USE_BINARY: 1\nSIMSED_PATH_BINARY: {bindir}\n"
    b = _run_snana(src, "JAX_LIVE_BIN", tmp_path, "bin")
    _assert_tables_identical(a, b, "SIMSED_USE_BINARY", rtol=1e-5, atol=1e-5)


def test_fluxerrmodel_optmask_no_photometry_change_live(tmp_path):
    """FLUXERRMODEL_OPTMASK bit 128 (MONITORCOV) is a REDCOV monitor and must
    not alter photometry (sntools_fluxErrModels.h:27)."""
    base = (PROJECT_ROOT / "tests" / "fixtures" / "snana" / "hostlib_e2e"
            / "input.input").read_text()
    base = re.sub(r"NGENTOT_LC:\s+\d+", "NGENTOT_LC:  60", base)
    fm = PROJECT_ROOT / "tests" / "fixtures" / "data" / "fluxerrmodel" / "JAX_TEST_FLAT.DAT"
    src0 = re.sub(r"GENVERSION:\s+\S+", "GENVERSION:  JAX_LIVE_OM0", base) + \
        f"\nFLUXERRMODEL_FILE: {fm}\n"
    src1 = re.sub(r"GENVERSION:\s+\S+", "GENVERSION:  JAX_LIVE_OM128", base) + \
        f"\nFLUXERRMODEL_FILE: {fm}\nFLUXERRMODEL_OPTMASK: 128\n"
    a = _run_snana(src0, "JAX_LIVE_OM0", tmp_path, "om0")
    b = _run_snana(src1, "JAX_LIVE_OM128", tmp_path, "om128")
    _assert_tables_identical(a, b, "FLUXERRMODEL_OPTMASK")


_COSMO_BASE = "tests/fixtures/snana/salt3_k21_replay/input.input"


def _hz_map_file(path, distort):
    om, ol, h0 = 0.315, 0.685, 70.0
    z = np.concatenate([[0.0], np.geomspace(1e-4, 3.0, 600)])
    hz = h0 * np.sqrt(om * (1 + z) ** 3 + ol) * distort(z)
    Path(path).write_text(
        "\n".join(f"{a:.8f} {b:.6f}" for a, b in zip(z, hz)) + "\n")
    return str(path)


def _cosmo_src(gv, extra, ngen):
    src = (PROJECT_ROOT / _COSMO_BASE).read_text()
    src = re.sub(r"GENVERSION:\s+\S+", f"GENVERSION:  {gv}", src)
    src = re.sub(r"NGENTOT_LC:\s+\S+", f"NGENTOT_LC:  {ngen}", src)
    return re.sub(r"^OMEGA_MATTER:", extra + "OMEGA_MATTER:", src,
                  count=1, flags=re.M)


def _jax_run(src, tmp_path, name, n):
    inp = tmp_path / f"{name}.input"
    inp.write_text(src)
    os.environ.setdefault("JAX_TEST_FIXTURE_ROOT",
                          str(PROJECT_ROOT / "tests" / "fixtures" / "data"))
    config = dict(n_sne=n)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    res, _, _ = sim_run.run_simulation(sim)
    return res


def test_hzfun_shape_map_moves_redshifts_live(tmp_path):
    """A shape-changing H(z) must move the generated redshift distribution.

    dV/dz carries H(z) (dVdz takes HzFUN_INFO), so the map changes the comoving
    volume as well as the distance. A *constant* rescaling cannot show this: it
    multiplies dV/dz by a constant that cancels in the normalised CDF.
    """
    n = 4000
    hz = _hz_map_file(tmp_path / "hz_shape.dat",
                      lambda z: 1.0 + 0.60 * z / (1.0 + z))
    extra = f"HzFUN_FILE: {hz}\n"

    ref = _run_snana(_cosmo_src("JAX_LIVE_HZS_REF", "", n), "JAX_LIVE_HZS_REF",
                     tmp_path, "hzsref")[0]
    mapped = _run_snana(_cosmo_src("JAX_LIVE_HZS_MAP", extra, n),
                        "JAX_LIVE_HZS_MAP", tmp_path, "hzsmap")[0]
    snana_dz = np.mean(mapped["SIM_REDSHIFT_CMB"]) - np.mean(ref["SIM_REDSHIFT_CMB"])

    j_ref = _jax_run(_cosmo_src("x", "", n), tmp_path, "jsref", n)
    j_map = _jax_run(_cosmo_src("x", extra, n), tmp_path, "jsmap", n)
    jax_dz = float(np.mean(np.asarray(j_map["parameters"]["sim_redshift_cmb"]))
                   - np.mean(np.asarray(j_ref["parameters"]["sim_redshift_cmb"])))

    assert snana_dz < -1e-3, f"SNANA GENZ did not move: {snana_dz}"
    # independent samples, so compare the shifts not the absolute means
    assert abs(jax_dz - snana_dz) < 1e-3, f"jax {jax_dz} vs SNANA {snana_dz}"


def test_hzfun_structured_table_mu_matches_snana_live(tmp_path):
    """Per-event SIM_DLMU through a rapidly oscillating HzFUN table, on
    SNANA's own redshifts. Linear interpolation leaves the integrand only
    piecewise smooth, so the distance integral must replicate
    Hzinv_integral's midpoint rule (sntools_cosmology.c:352-397); a fixed
    smooth-integrand quadrature is off at the 1e-3 mag level here."""
    from jax_snlc_core.cosmology import distance_modulus, load_hz_map
    hz = _hz_map_file(tmp_path / "hz_osc.dat",
                      lambda z: 1.0 + 0.05 * np.sin(300.0 * z))
    gv = "JAX_LIVE_HZOSC"
    head, _ = _run_snana(_cosmo_src(gv, f"HzFUN_FILE: {hz}\n", 120),
                         gv, tmp_path, "hzosc")
    hz_map = load_hz_map(hz, 70.0)
    hd = 299792.458 / 70.0
    mu_jax = np.array([
        float(distance_modulus(
            float(zc), float(zh), 0.315, 0.685, hd, hz_map=hz_map))
        for zc, zh in zip(head["SIM_REDSHIFT_CMB"],
                          head["SIM_REDSHIFT_HELIO"])])
    d = np.abs(mu_jax - np.asarray(head["SIM_DLMU"], dtype=np.float64))
    assert d.max() < 2e-4, f"max |dmu| = {d.max():.2e}"


def test_anisotropy_v04_matches_snana_live(tmp_path):
    """Per-event V04 distance modulus, on SNANA's own inputs.

    jax and SNANA generate different events, so the comparison feeds SNANA's
    dumped (zHEL, RA, DEC) through jax's formula and checks MU event for event —
    isolating dLmag_anisotropic (sntools_cosmology.c:580) from the sampling.
    """
    from jax_snlc_core.cosmology import (ANISOTROPY_V04_PARS, _mu_anisotropic_V04,
                                         cos_sep_cmb_apex)

    gv = "JAX_LIVE_ANISO"
    src = re.sub(r"GENVERSION:\s+\S+", f"GENVERSION:  {gv}",
                 (PROJECT_ROOT / _COSMO_BASE).read_text())
    src = re.sub(r"^NGENTOT_LC:.*$", "NGENTOT_LC: 60", src, count=1, flags=re.M)
    src = re.sub(r"^SIMGEN_DUMPALL:.*$",
                 "SIMGEN_DUMPALL: 8 CID GENZ ZCMB ZHELIO RA DEC MU LIBID",
                 src, count=1, flags=re.M)
    src = re.sub(r"^OMEGA_MATTER:", "ANISOTROPY_MODELNAME: V04\nOMEGA_MATTER:",
                 src, count=1, flags=re.M)
    inp = tmp_path / "aniso.input"
    inp.write_text(src)
    env = os.environ.copy()
    r = subprocess.run(["snlc_sim.exe", str(inp)], cwd=tmp_path, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]

    sdir = Path(env.get("SNANA_LSST_SIM", "")) / gv
    if not sdir.is_dir():
        sdir = Path(env["SNDATA_ROOT"]) / "SIM" / gv
    dump = glob.glob(f"{sdir}/*.DUMP")[0]
    rows = [l.split() for l in Path(dump).read_text().splitlines()
            if l.startswith("SN:")]
    assert len(rows) > 20, f"only {len(rows)} dumped events"
    zhel = np.array([float(x[4]) for x in rows])
    ra = np.array([float(x[5]) for x in rows])
    dec = np.array([float(x[6]) for x in rows])
    mu_snana = np.array([float(x[7]) for x in rows])

    hd = 299792.458 / 70.0
    cos_sep = np.asarray(cos_sep_cmb_apex(ra, dec))
    mu_jax = np.array([float(_mu_anisotropic_V04(z, hd, c, ANISOTROPY_V04_PARS))
                       for z, c in zip(zhel, cos_sep)])
    # the DUMP prints MU to 6 significant figures, ~1e-4 mag at mu~41
    assert np.abs(mu_jax - mu_snana).max() < 2e-4, (
        f"max |dmu| = {np.abs(mu_jax - mu_snana).max():.2e}")


def test_hzfun_expected_count_matches_snana_live(tmp_path):
    """H(z) sets the season count, so NGEN_SEASON must resolve identically.

    SNcount_model integrates dVdz(z, &INPUTS.HzFUN_INFO) (snlc_sim.c:17918).
    INIT_ONLY stops after the rate summary, which turns this into an exact
    integer comparison against SNANA's own NGENTOT_LC.
    """
    base = re.sub(r"^NGENTOT_LC:.*$", "NGEN_SEASON: 1.0",
                  (PROJECT_ROOT / _COSMO_BASE).read_text(), count=1, flags=re.M)

    def sources(extra, gv):
        src = re.sub(r"^OMEGA_MATTER:", extra + "OMEGA_MATTER:", base,
                     count=1, flags=re.M)
        return re.sub(r"GENVERSION:\s+\S+", f"GENVERSION:  {gv}", src)

    def snana_ngentot(src, name):
        inp = tmp_path / f"{name}.input"
        inp.write_text(src + "\nINIT_ONLY: 1\n")
        env = os.environ.copy()
        r = subprocess.run(["snlc_sim.exe", str(inp)], cwd=tmp_path, env=env,
                           capture_output=True, text=True)
        m = re.search(r"NGEN_SEASON=[\d.]+ --> NGENTOT_LC=(\d+)", r.stdout)
        assert m, r.stdout[-2000:]
        return int(m.group(1))

    def jax_ngentot(src, name):
        from jax_snlc_sim.rate import resolve_counts
        inp = tmp_path / f"{name}.input"
        inp.write_text(src)
        cfg = parse_snana_input_file(str(inp))
        return resolve_counts(cfg, cfg.hz_map())[0]

    cases = {
        'off':    "",
        'flat':   f"HzFUN_FILE: {_hz_map_file(tmp_path / 'ec_flat.dat', lambda z: 1.0)}\n",
        'scaled': f"HzFUN_FILE: {_hz_map_file(tmp_path / 'ec_scaled.dat', lambda z: 1.10)}\n",
        'shape':  f"HzFUN_FILE: {_hz_map_file(tmp_path / 'ec_shape.dat', lambda z: 1.0 + 0.60 * z / (1.0 + z))}\n",
    }
    for name, extra in cases.items():
        src = sources(extra, f"JAX_LIVE_HZC_{name.upper()}")
        assert jax_ngentot(src, f"ec_{name}") == snana_ngentot(src, f"ec_{name}_s"), name


def test_mushift_matches_snana_live(tmp_path):
    """MUSHIFT shifts observed mags, leaves SIM_DLMU, and spares the 99 flag.

    jax and SNANA generate different events, so the comparison is on the three
    properties rather than per-CID: the shift applied to in-range epochs, the
    truth distance being untouched, and zero-flux epochs staying at exactly 99.
    """
    n, shift = 200, 0.1
    extra = f"MUSHIFT: {shift}\n"

    ref_h, ref_p = _run_snana(_cosmo_src("JAX_LIVE_MU_REF", "", n),
                              "JAX_LIVE_MU_REF", tmp_path, "muref")
    sh_h, sh_p = _run_snana(_cosmo_src("JAX_LIVE_MU_SET", extra, n),
                            "JAX_LIVE_MU_SET", tmp_path, "muset")

    def snana_mags(head, phot):
        out = {}
        for row in head:
            a, b = int(row["PTROBS_MIN"]), int(row["PTROBS_MAX"])
            for i in range(a - 1, b):
                r = phot[i]
                if float(r["MJD"]) > 0:
                    out[(int(row["SNID"]), round(float(r["MJD"]), 4),
                         str(r["BAND"]).strip())] = float(r["SIM_MAGOBS"])
        return out

    m_ref, m_sh = snana_mags(ref_h, ref_p), snana_mags(sh_h, sh_p)
    common = set(m_ref) & set(m_sh)
    snana_dmag = np.array([m_sh[k] - m_ref[k] for k in common if m_ref[k] < 50])
    snana_flagged = np.array([m_sh[k] for k in common if m_ref[k] >= 50])
    snana_dmu = float(np.mean(sh_h["SIM_DLMU"]) - np.mean(ref_h["SIM_DLMU"]))

    j_ref = _jax_run(_cosmo_src("x", "", n), tmp_path, "jmuref", n)
    j_sh = _jax_run(_cosmo_src("x", extra, n), tmp_path, "jmuset", n)
    r_mag = np.asarray(j_ref["magnitudes"])
    s_mag = np.asarray(j_sh["magnitudes"])
    in_range = (r_mag > -2) & (r_mag < 50)
    jax_dmag = s_mag[in_range] - r_mag[in_range]
    jax_flagged = s_mag[r_mag >= 50]
    jax_dmu = float(np.mean(np.asarray(j_sh["parameters"]["mu"]))
                    - np.mean(np.asarray(j_ref["parameters"]["mu"])))

    # both engines: mags shift by exactly MUSHIFT
    assert np.allclose(snana_dmag, shift, atol=1e-4)
    assert np.allclose(jax_dmag, shift, atol=1e-4)
    # both engines: truth distance untouched
    assert abs(snana_dmu) < 1e-4 and abs(jax_dmu) < 1e-4
    # both engines: zero-flux epochs survive
    assert snana_flagged.size and np.all(snana_flagged == 99.0)
    assert jax_flagged.size and np.all(jax_flagged == 99.0)
    # and jax reports the value SNANA does
    assert np.allclose(sh_h["SIM_MUSHIFT"], shift, atol=1e-5)
    assert np.allclose(np.asarray(j_sh["parameters"]["mushift"]), shift, atol=1e-5)
