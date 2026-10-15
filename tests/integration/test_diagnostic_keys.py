"""SNANA diagnostic-output keys: APPEND_SNID_SEDINDEX,
SIMGEN_DUMP_RATE/MWCL/NOISE/TRAINSALT, WRFLAG_YAML_FILE, SIMLIB_DUMP.

Each runs a small sim (or SIMLIB-only pass) and checks the diagnostic file is
written with the SNANA format. Formats are transcribed from snlc_sim.c.
"""

import glob
import os
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from jax_snlc_core.io.fits_path import resolve_fits_path
from jax_snlc_sim import run as sim_run

_NON1A_INPUT = "tests/fixtures/snana/non1ased_v19_replay/input.input"
_SALT3_INPUT = "tests/fixtures/snana/salt3_k21_replay/input.input"


def _run(input_file, extra, outdir, version="DIAG", n_sne=30, seed=321):
    """Run a small sim from a fixture input plus appended lines; return version dir."""
    os.makedirs(outdir, exist_ok=True)
    inp = Path(outdir) / "in.input"
    inp.write_text(Path(input_file).read_text() + "\n" + extra)
    config = dict(n_sne=n_sne, random_seed=seed)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    results, sel, spectra_cache = sim_run.run_simulation(sim)
    sim_run.save_snana_outputs(sim, results, version, sel, spectra_cache=spectra_cache, output_dir=outdir)
    return os.path.join(outdir, version)


def test_append_snid_sedindex(tmp_path):
    """APPEND_SNID_SEDINDEX: SNID string becomes '{CID}-{TEMPLATE_INDEX}', while
    the .DAT filename keeps the plain CID."""
    d = _run(_NON1A_INPUT, "APPEND_SNID_SEDINDEX: 1\n", str(tmp_path / "app"))
    with fits.open(resolve_fits_path(os.path.join(d, "DIAG_HEAD.FITS"))) as h:
        snids = [str(s).strip() for s in h[1].data['SNID']]
        tmpl = list(h[1].data['SIM_TEMPLATE_INDEX'])
    assert all('-' in s for s in snids)
    for s, t in zip(snids, tmpl):
        cid, _, sedindex = s.partition('-')
        assert cid.isdigit()
        assert sedindex == str(int(t))
    # TEXT writes the same SNID string (SNANA writes SNDATA.CCID as-is,
    # sntools_dataformat_text.c:124; an int() cast would crash here).
    d2 = _run(_NON1A_INPUT, "APPEND_SNID_SEDINDEX: 1\nFORMAT_MASK: 2\n",
              str(tmp_path / "app_text"))
    dats = sorted(glob.glob(os.path.join(d2, "*.DAT")))
    assert dats
    snid_line = next(l for l in open(dats[0]) if l.startswith("SNID:"))
    assert '-' in snid_line.split(':', 1)[1]


def test_simgen_dump_rate(tmp_path):
    """SIMGEN_DUMP_RATE: .RATE lists the volumetric rate R(z) on a 0.01 z-grid."""
    d = _run(_SALT3_INPUT, "SIMGEN_DUMP_RATE: 1\n", str(tmp_path / "rate"))
    lines = Path(d, "DIAG.RATE").read_text().splitlines()
    assert lines[0].startswith("# RATE = Volumetric Rate")
    assert lines[1] == "VARNAMES: ROW REDSHIFT RATE "
    rows = [l.split() for l in lines if l.startswith("ROW:")]
    assert len(rows) > 5
    # z-grid ascends by 0.01; rate is positive and (POWERLAW2 input) rises with z
    assert all(float(rows[i + 1][2]) > float(rows[i][2]) for i in range(len(rows) - 1))
    assert float(rows[0][3]) > 0
    assert float(rows[-1][3]) > float(rows[0][3])


def test_simgen_dump_mwcl(tmp_path):
    """SIMGEN_DUMP_MWCL: .MWCL is the MW extinction grid for 5 colour laws over
    RV=1..6 (step 1) × WAVE=1000..25000 (step 100), AV=0.1."""
    d = _run(_SALT3_INPUT, "SIMGEN_DUMP_MWCL: 1\n", str(tmp_path / "mwcl"))
    txt = Path(d, "DIAG.MWCL").read_text()
    lines = txt.splitlines()
    assert "# AV = 0.10 is fixed" in txt
    varnames = next(l for l in lines if l.startswith("VARNAMES:"))
    for lbl in ("FITZ99_EXACT", "FITZ99_APPROX", "FITZ04", "CCM89", "ODON94"):
        assert f"MAG_MWXT_{lbl}" in varnames
    rows = [l for l in lines if l.startswith("ROW:")]
    # 6 RV bins × 240 wave bins (SNANA loop is WAVE < WAVE_MAX, so 1000..24900)
    assert len(rows) == 6 * 240
    # each row: ROW: <n> <rv> <wave> + 5 extinction values
    assert len(rows[0].split()) == 1 + 3 + 5


def test_simgen_dump_trainsalt(tmp_path):
    """SIMGEN_DUMP_TRAINSALT: one row per written SN, '#SNID PKMJD PKMJD_TRUE'."""
    d = _run(_SALT3_INPUT, "SIMGEN_DUMP_TRAINSALT: 1\n", str(tmp_path / "ts"))
    lines = Path(d, "DIAG.TRAINSALT").read_text().splitlines()
    assert lines[0] == "#SNID  PKMJD  PKMJD_TRUE"
    rows = lines[1:]
    with fits.open(resolve_fits_path(os.path.join(d, "DIAG_HEAD.FITS"))) as h:
        head_cids = {int(str(s).strip()) for s in h[1].data['SNID']}
    assert len(rows) == len(head_cids)
    row_cids = {int(r.split()[0]) for r in rows}
    assert row_cids == head_cids
    # each row has 3 numeric fields
    assert all(len(r.split()) == 3 for r in rows)


def test_photflag_snrmax_nearpeak(tmp_path):
    """PHOTFLAG_SNRMAX sets its bit on exactly one written epoch per SN. NEARPEAK
    ranges over all real obs (SNANA keep_SIMLIB_OBS, incl. undefined-mag epochs),
    so it flags at most one epoch — none when the nearest real obs is unwritten;
    when flagged, that epoch is the nearest written one."""
    d = _run(_SALT3_INPUT, "PHOTFLAG_SNRMAX: 8192\nPHOTFLAG_NEARPEAK: 16384\n",
             str(tmp_path / "pf"), n_sne=10)
    with fits.open(resolve_fits_path(os.path.join(d, "DIAG_HEAD.FITS"))) as h:
        hd = h[1].data
        pmin, pmax, peak = hd['PTROBS_MIN'], hd['PTROBS_MAX'], hd['SIM_PEAKMJD']
    with fits.open(resolve_fits_path(os.path.join(d, "DIAG_PHOT.FITS"))) as h:
        pf = h[1].data['PHOTFLAG']
        mjd = h[1].data['MJD']
    for i in range(len(pmin)):
        lo, hi = int(pmin[i]) - 1, int(pmax[i])
        seg_pf, seg_mjd = pf[lo:hi], mjd[lo:hi]
        assert int(np.sum((seg_pf & 8192) > 0)) == 1      # SNRMAX on one epoch
        n_near = int(np.sum((seg_pf & 16384) > 0))
        assert n_near <= 1                                # NEARPEAK on 0 or 1 epoch
        if n_near == 1:
            # flagged epoch is the nearest written one
            near_idx = int(np.argmax((seg_pf & 16384) > 0))
            assert near_idx == int(np.argmin(np.abs(seg_mjd - peak[i])))


def test_simlib_dump(tmp_path):
    """SIMLIB_DUMP: cadence-stats mode writes AVG (per-LIBID) + OBS (per-obs)
    TEXT files off the SIMLIB alone (no generation)."""
    from jax_snlc_sim.config_parser import parse_snana_input_file
    from jax_snlc_sim.simlib_dump import run_simlib_dump
    inp = tmp_path / "sd.input"
    inp.write_text(Path(_SALT3_INPUT).read_text() + "\nSIMLIB_DUMP: 3\n")
    cfg = parse_snana_input_file(str(inp))
    written = run_simlib_dump(cfg, output_dir=str(tmp_path))
    assert len(written) == 2
    # SNANA filename carries the GENRANGE_MJD cut string
    assert any(p.endswith("_60500-61500.TEXT") for p in written)
    avg_path = next(p for p in written if "AVG" in p)
    obs_path = next(p for p in written if "OBS" in p)
    avg = Path(avg_path).read_text().splitlines()
    obs = Path(obs_path).read_text().splitlines()
    # SNANA DOCUMENTATION wrapper
    assert avg[0] == "DOCUMENTATION:"
    assert any("DOCUMENTATION_END:" in l for l in avg)
    assert any(l.startswith("  MJD_RANGE:   60500 61500") for l in avg)
    assert any("COLUMN_NOTES:" in l for l in obs)  # OBS table only
    # VARNAMES + data rows
    avg_var = next(l for l in avg if l.startswith("VARNAMES:"))
    obs_var = next(l for l in obs if l.startswith("VARNAMES:"))
    assert avg_var.startswith(
        "VARNAMES: ROW LIBID RA DEC FIELD MWEBV GAPMAX GAPAVG NOBS MJDMIN MJDMAX NSEASON")
    assert "NOBS_g ZPT_g PSF_g M5SIG_g" in avg_var
    assert obs_var.startswith(
        "VARNAMES: ROW LIBID FIELD  RA DEC MJD BAND ZP_pe SKYMAG PSF M5SIG")
    avg_rows = [l.split() for l in avg if l.startswith("ROW:")]
    obs_rows = [l.split() for l in obs if l.startswith("ROW:")]
    assert len(avg_rows) == 2 and len(obs_rows) > 1
    # FIELD parsed from the bundled SIMLIB's RA/DEC line (not defaulted)
    assert [r[5] for r in avg_rows] == ["TOY_A", "TOY_B"]   # AVG col 5 = FIELD
    assert {r[3] for r in obs_rows} == {"TOY_A", "TOY_B"}   # OBS col 3 = FIELD


def test_dashboard(tmp_path, caplog):
    """DASHBOARD: a valueless key that summarises the resolved input files and
    the maps they loaded, with no generation."""
    import logging
    inp = tmp_path / "db.input"
    inp.write_text(Path(_SALT3_INPUT).read_text() + "\nDASHBOARD:\n")
    with caplog.at_level(logging.INFO, logger="jax_snlc_sim.dashboard"):
        sim = sim_run.prepare(str(inp), n_sne=2)
    assert sim is None
    out = caplog.text
    for key in ("GENMODEL", "SIMLIB_FILE", "CALIB_FILE", "HOSTLIB_FILE",
                "WRONGHOST_FILE", "FLUXERRMODEL_FILE", "NONLINEARITY_FILE",
                "ZVARIATION_FILE", "WEAKLENS_PROBMAP_FILE", "STRONGLENS_FILE",
                "SEARCHEFF_PIPELINE_LOGIC_FILE",
                "SEARCHEFF_SPEC_FILE", "SEARCHEFF_zHOST_FILE"):
        assert key in out
    assert "observation sequences" in out
    assert "NMAP_SPECEFF" in out
    # SNANA prints TIME_SINGLE_DETECT (= NEWMJD_DIF) on the logic line
    # (snlc_sim.c:31691-31692, :7523)
    assert "epoch-sep >" in out
    # unset file keys are named, not silently dropped
    assert "(none)" in out


@pytest.mark.parametrize("extra", [
    "DASHBOARD:", "INIT_ONLY: 1", "SIMLIB_DUMP: 3",
    "SIMGEN_DUMP: 0",
])
def test_setup_and_stop_modes_return_no_simulator(tmp_path, extra):
    """SNANA's four set-up-and-stop modes are carried out by `prepare`, which
    then returns no compiled simulator, so every caller stops where the
    command line does rather than generating a full simulation."""
    key = extra.split(':')[0]
    body = [l for l in Path(_SALT3_INPUT).read_text().splitlines()
            if not l.startswith(key) and not l.startswith('SIMGEN_DUMPALL')]
    inp = tmp_path / f"mode_{abs(hash(extra))}.input"
    inp.write_text("\n".join(body) + f"\n{extra}\n")
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        sim = sim_run.prepare(str(inp), **dict(n_sne=2))
    finally:
        os.chdir(cwd)
    assert sim is None


def test_a_plain_input_still_compiles_a_simulator(tmp_path):
    """The guard above is not vacuous: without a stop mode prepare compiles."""
    sim = sim_run.prepare(_SALT3_INPUT, **dict(n_sne=2))
    assert sim is not None


def test_simgen_dump_noise(tmp_path):
    """SIMGEN_DUMP_NOISE: per-obs Npe breakdown; COV_TOT == COV_SN+GAL+SKY+READ.
    Mode 1 (near peak) keeps fewer rows than mode 2 (all obs)."""
    def rows(mode):
        d = _run(_SALT3_INPUT, f"SIMGEN_DUMP_NOISE: {mode}\n", str(tmp_path / f"n{mode}"))
        lines = Path(d, "DIAG.NOISE").read_text().splitlines()
        assert any(l.startswith("VARNAMES: CID GALID LIBID") for l in lines)
        return [l.split() for l in lines if l.startswith("SN:")]

    allobs = rows(2)
    assert len(allobs) > 0
    for r in allobs:
        tot = float(r[12])
        assert abs(tot - (float(r[13]) + float(r[14]) + float(r[15]) + float(r[16]))) < 0.02
        # SNANA no-host sentinels (no HOSTLIB): MAG_GAL=666, GALSNSEP=0, COV_GAL=0
        assert float(r[8]) == 666.0 and float(r[9]) == 0.0 and float(r[14]) == 0.0
    nearpeak = rows(1)
    # NEARPEAK keeps at most one obs per (CID, band)
    keys = [(r[1], r[5]) for r in nearpeak]
    assert 0 < len(nearpeak) <= len(allobs)
    assert len(keys) == len(set(keys))


def test_wrflag_yaml_file(tmp_path):
    """WRFLAG_YAML_FILE: .YAML carries the job counts; NGENLC_WRITE == n_passed.
    SNANA writes it to the run cwd (bare GENVERSION name, snlc_sim.c:30917), so
    run from tmp_path and read it there."""
    salt3_abs = os.path.abspath(_SALT3_INPUT)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        d = _run(salt3_abs, "WRFLAG_YAML_FILE: 1\n", str(tmp_path / "yaml"))
        txt = (tmp_path / "DIAG.YAML").read_text()
    finally:
        os.chdir(cwd)
    y = dict(l.split(":", 1) for l in txt.splitlines() if ":" in l)
    for key in ("SURVEY", "IDSURVEY", "NGENLC_TOT", "NGENLC_WRITE",
                "NGENSPEC_WRITE", "CPU_MINUTES", "ABORT_IF_ZERO"):
        assert key in y
    with fits.open(resolve_fits_path(os.path.join(d, "DIAG_HEAD.FITS"))) as h:
        n_written = len(h[1].data)
    assert int(y["NGENLC_WRITE"]) == n_written
    assert int(y["ABORT_IF_ZERO"]) == n_written


def test_smearflag_flux_zero_leaves_the_flux_unsmeared(tmp_path):
    """SMEARFLAG_FLUX bit 1 gates whether the drawn noise is applied: at 0
    SNANA zeroes GAURAN_SZ/T/F so fluxObs == fluxTrue (snlc_sim.c:28466),
    confirmed against snlc_sim.exe, whose FLUXCAL then reproduces SIM_MAGOBS
    to float32 precision. jax used to add noise regardless."""
    import numpy as np
    base = Path(_SALT3_INPUT).read_text().splitlines()
    body = [l for l in base if not l.startswith('SMEARFLAG_FLUX')]

    def flux_gap(flag):
        inp = tmp_path / f"sf{flag}.input"
        inp.write_text("\n".join(body) + f"\nSMEARFLAG_FLUX: {flag}\n")
        config = dict(
            n_sne=20, random_seed=4242)
        sim = sim_run.prepare(str(inp), **config)
        simulation = None if sim is None else sim.config
        resources = sim
        res, _, _ = sim_run.run_simulation(sim)
        noisy = np.asarray(res['fluxcals_noisy'])
        true = np.asarray(res['fluxcals'])
        ok = np.isfinite(noisy) & np.isfinite(true)
        assert ok.sum() > 50, "too few epochs to be meaningful"
        return np.abs(noisy[ok] - true[ok])

    assert flux_gap(0).max() == 0.0, "SMEARFLAG_FLUX: 0 must not smear the flux"
    assert (flux_gap(1) > 0).any(), "SMEARFLAG_FLUX: 1 must smear the flux"
