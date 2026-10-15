"""Command-line integration tests using bundled simulation and fit data."""

import gzip
import os
import subprocess
import sys
from pathlib import Path

import pytest

from jax_snlc_core.io.fitres_reader import read_fitres_header


def _stage_fit_version(tmp_path, head, phot, data_root):
    """Stage a fixture's light curves where its own nml looks for them, and
    return the environment naming the staging directory.

    The nml resolves VERSION_PHOTOMETRY against PRIVATE_DATA_PATH, which the
    fixtures write as $JAX_TEST_FIT_SIM_DIR.
    """
    version = "JAX_TEST_SALT3_FIT_DENSE"
    sim_dir = tmp_path / version
    sim_dir.mkdir()
    (sim_dir / f"{version}_HEAD.FITS").write_bytes(
        gzip.decompress(head.with_suffix(".FITS.gz").read_bytes()))
    (sim_dir / f"{version}_PHOT.FITS").write_bytes(
        gzip.decompress(phot.with_suffix(".FITS.gz").read_bytes()))
    (sim_dir / f"{version}.LIST").write_text(f"{version}_HEAD.FITS\n")
    env = os.environ.copy()
    env["JAX_TEST_FIXTURE_ROOT"] = str(data_root)
    env["JAX_TEST_FIT_SIM_DIR"] = str(tmp_path)
    return env


def _ndof(fitres):
    """NDOF column, read by VARNAMES position."""
    names, out = None, []
    for line in fitres.read_text().splitlines():
        if line.startswith("VARNAMES:"):
            names = line.split()[1:]
        elif line.startswith("SN:") and names:
            out.append(int(line.split()[1:][names.index("NDOF")]))
    return out


def test_snlc_fit_positional_nml_and_key_overrides(
        tmp_path, repo_root, fixture_root, data_root) -> None:
    """SNANA's `snlc_fit.exe <nml> KEY value` form, and that an override bites.

    DELCHI2_REJECT was one of the keys the fitter implemented but no input file
    could reach; a threshold below the typical per-epoch chi2 must drop epochs.
    An unknown key must abort rather than be ignored.
    """
    fx = fixture_root / "snana" / "salt3_fit_dense"
    nml, head, phot = fx / "fit.nml", fx / "HEAD.FITS", fx / "PHOT.FITS"

    env = _stage_fit_version(tmp_path, head, phot, data_root)

    def run(out_name, *overrides):
        out = tmp_path / out_name
        result = subprocess.run(
            [sys.executable, str(repo_root / "snlc_fit.py"), str(nml), *overrides,
             "MAX_FIT", "3", "BATCH_SIZE", "3", "OUTPUT_NAME", str(out)],
            capture_output=True, text=True, env=env, timeout=300,
        )
        return result, out

    plain, plain_out = run("PLAIN.FITRES.TEXT")
    assert plain.returncode == 0, plain.stdout[-800:] + plain.stderr[-800:]

    cut, cut_out = run("CUT.FITRES.TEXT", "DELCHI2_REJECT", "0.1")
    assert cut.returncode == 0, cut.stdout[-800:] + cut.stderr[-800:]

    base, rejected = _ndof(plain_out), _ndof(cut_out)
    assert base and len(base) == len(rejected)
    assert all(r < b for b, r in zip(base, rejected)), (
        f"DELCHI2_REJECT did not drop epochs: {base} -> {rejected}")

    bad, _ = run("NOPE.FITRES.TEXT", "NOT_A_SNANA_KEY", "1")
    assert bad.returncode != 0, "unknown override key should abort"
    assert "NOT_A_SNANA_KEY" in bad.stdout + bad.stderr


def test_snlc_fit_records_the_simulated_cosmology(
        tmp_path, repo_root, fixture_root, data_root) -> None:
    """A simulated version's README states the cosmology it was generated
    with, under a single run's INPUT_KEYS. The fit copies it into the FITRES
    header, where BBC finds the bias-correction simulation's."""
    fx = fixture_root / "snana" / "salt3_fit_dense"
    env = _stage_fit_version(tmp_path, fx / "HEAD.FITS", fx / "PHOT.FITS", data_root)
    # SNANA's README of another simulation: the fit reads its keys, not its events
    version = tmp_path / "JAX_TEST_SALT3_FIT_DENSE"
    (version / f"{version.name}.README").write_bytes(
        (fixture_root / "snana" / "gentype_map_k10" / "SIM.README").read_bytes())
    out = tmp_path / "FITRES.TEXT"

    result = subprocess.run(
        [sys.executable, str(repo_root / "snlc_fit.py"), str(fx / "fit.nml"),
         "MAX_FIT", "1", "OUTPUT_NAME", str(out)],
        capture_output=True, text=True, env=env, timeout=300)

    assert result.returncode == 0, result.stdout[-800:] + result.stderr[-800:]
    header = read_fitres_header(str(out))
    assert {key: header[f"SIM_{key}"] for key in (
        "OMEGA_MATTER", "OMEGA_LAMBDA", "w0_LAMBDA", "wa_LAMBDA", "MUSHIFT")} == {
        "OMEGA_MATTER": "0.3000", "OMEGA_LAMBDA": "0.7000",
        "w0_LAMBDA": "-1.0000", "wa_LAMBDA": "0", "MUSHIFT": "0"}


def test_snlc_fit_inival_inistp_pins_a_parameter(
        tmp_path, repo_root, fixture_root, data_root) -> None:
    """INISTP_<p> 0 holds a parameter at INIVAL_<p>, for the whole fit.

    The pin has to survive the pre-fit grid search, which moves PKMJD/x1/c
    before round 1 and would otherwise overwrite it silently — the fit would
    report a "fixed" parameter at a value the user never asked for.
    """
    fx = fixture_root / "snana" / "salt3_fit_dense"
    nml, head, phot = fx / "fit.nml", fx / "HEAD.FITS", fx / "PHOT.FITS"

    env = _stage_fit_version(tmp_path, head, phot, data_root)

    def run(out_name, *overrides, max_fit="3"):
        out = tmp_path / out_name
        result = subprocess.run(
            [sys.executable, str(repo_root / "snlc_fit.py"), str(nml), "BATCH_SIZE", "4", *overrides,
             "MAX_FIT", max_fit, "OUTPUT_NAME", str(out)],
            capture_output=True, text=True, env=env, timeout=300)
        return result, out

    def column(path, name):
        names, vals = None, []
        for line in path.read_text().splitlines():
            if line.startswith("VARNAMES:"):
                names = line.split()[1:]
            elif line.startswith("SN:") and names:
                vals.append(float(line.split()[1:][names.index(name)]))
        return vals

    # x1 pinned; c must still be fitted, so the pin is per-parameter
    pinned, pinned_out = run("PIN.FITRES.TEXT", "INIVAL_x1", "0.5", "INISTP_x1", "0")
    assert pinned.returncode == 0, pinned.stdout[-800:] + pinned.stderr[-800:]
    x1 = column(pinned_out, "x1")
    assert x1 and all(v == 0.5 for v in x1), f"x1 not pinned: {x1}"
    assert len(set(column(pinned_out, "c"))) > 1, "c should still vary"

    # PKMJD is grid-searched before round 1, so pinning it is the sharper case.
    # One SN, pinned just off its own peak: near enough that epochs survive the
    # Trest window, far enough that a still-running grid search would move it.
    free, free_out = run("FREE.FITRES.TEXT", max_fit="1")
    assert free.returncode == 0, free.stdout[-800:]
    peak = round(column(free_out, "PKMJD")[0]) - 1.0

    t0, t0_out = run("T0.FITRES.TEXT", "INIVAL_PEAKMJD", str(peak),
                     "INISTP_PEAKMJD", "0", max_fit="1")
    assert t0.returncode == 0, t0.stdout[-800:] + t0.stderr[-800:]
    assert column(t0_out, "PKMJD") == [peak], "PKMJD moved despite INISTP = 0"
    assert column(t0_out, "NDOF")[0] > 0, "pin left no epochs; test says nothing"

    # a suffix naming no parameter aborts, saying what the model does fit
    bad, _ = run("BAD.FITRES.TEXT", "INIVAL_NOT_A_PARAM", "1.0")
    assert bad.returncode != 0
    assert "x1" in bad.stdout + bad.stderr


def test_snlc_fit_cutwin_rejecting_every_sn(
        tmp_path, repo_root, fixture_root, data_root) -> None:
    """A CUTWIN keeping nothing names the cut and still writes the FITRES
    header, matching SNANA (snana.F90:9071; the table is created in FITPAR_INI2
    before any SN). It previously died in BatchFitResult.concatenate."""
    fx = fixture_root / "snana" / "salt3_fit_dense"
    nml, head, phot = fx / "fit.nml", fx / "HEAD.FITS", fx / "PHOT.FITS"

    env = _stage_fit_version(tmp_path, head, phot, data_root)

    def run(out_name, *overrides):
        out = tmp_path / out_name
        return subprocess.run(
            [sys.executable, str(repo_root / "snlc_fit.py"), str(nml), "BATCH_SIZE", "4", *overrides,
             "MAX_FIT", "3", "OUTPUT_NAME", str(out)],
            capture_output=True, text=True, env=env, timeout=300), out

    # MWEBV is 0.0 for every SN in this fixture, so this window keeps none.
    empty, empty_out = run("EMPTY.FITRES.TEXT", "CUTWIN_MWEBV", "0.3", "0.4")
    assert empty.returncode == 0, empty.stdout[-800:] + empty.stderr[-800:]
    assert "every SN fails CUTWIN_MWEBV" in empty.stdout + empty.stderr
    assert empty_out.exists(), "no FITRES written"
    text = empty_out.read_text()
    assert "VARNAMES:" in text
    assert not [ln for ln in text.splitlines() if ln.startswith("SN:")]

    # The header must match a run that fitted something, not a reduced one.
    kept, kept_out = run("KEPT.FITRES.TEXT", "CUTWIN_REDSHIFT", "0.3", "0.4")
    assert kept.returncode == 0, kept.stdout[-800:]
    assert [ln for ln in kept_out.read_text().splitlines() if ln.startswith("SN:")]

    def varnames(p):
        return [ln for ln in p.read_text().splitlines()
                if ln.startswith("VARNAMES:")][0]
    assert varnames(empty_out) == varnames(kept_out)


def test_snlc_fit_fitwin_postfit_cuts(
        tmp_path, repo_root, fixture_root, data_root) -> None:
    """FITWIN_<quantity> cuts the finished fit, on the model's own FITRES
    columns — so a derived column's error (SALT mB) needs no special case, and
    the surviving rows must actually satisfy the window."""
    fx = fixture_root / "snana" / "salt3_fit_dense"
    nml, head, phot = fx / "fit.nml", fx / "HEAD.FITS", fx / "PHOT.FITS"

    env = _stage_fit_version(tmp_path, head, phot, data_root)

    def run(out_name, *overrides):
        out = tmp_path / out_name
        return subprocess.run(
            [sys.executable, str(repo_root / "snlc_fit.py"), str(nml), "BATCH_SIZE", "4", *overrides,
             "MAX_FIT", "10", "OUTPUT_NAME", str(out)],
            capture_output=True, text=True, env=env, timeout=600), out

    def column(path, name):
        names, vals = None, []
        for line in path.read_text().splitlines():
            if line.startswith("VARNAMES:"):
                names = line.split()[1:]
            elif line.startswith("SN:") and names:
                vals.append(float(line.split()[1:][names.index(name)]))
        return vals

    base, base_out = run("FWBASE.FITRES.TEXT")
    assert base.returncode == 0, base.stdout[-800:]
    all_x1 = column(base_out, "x1")
    assert len(all_x1) == 10

    cut, cut_out = run("FWCUT.FITRES.TEXT", "FITWIN_SHAPE", "-0.5", "0.5")
    assert cut.returncode == 0, cut.stdout[-800:]
    kept = column(cut_out, "x1")
    assert kept and len(kept) < len(all_x1), "window kept everything"
    assert all(-0.5 <= v <= 0.5 for v in kept)
    # exactly the ones inside the window, not merely a subset
    assert sorted(kept) == sorted(v for v in all_x1 if -0.5 <= v <= 0.5)

    # an error window on a derived column resolves through the same path, and
    # must actually cut. SNANA copies only the upper edge (snlc_fit.F90:1411).
    base_mberr = sorted(column(base_out, "mBERR"))
    cutoff = base_mberr[len(base_mberr) // 2]
    err, err_out = run("FWERR.FITRES.TEXT", "FITWIN_MB_ERR", "0.0", str(cutoff))
    assert err.returncode == 0, err.stdout[-800:]
    kept_err = column(err_out, "mBERR")
    assert 0 < len(kept_err) < 10, f"window kept {len(kept_err)} of 10"
    assert all(v <= cutoff for v in kept_err)


def test_jobsplit_and_event_cap_with_a_header_cut(
        tmp_path, repo_root, fixture_root, data_root):
    """JOBSPLIT / MXEVT_PROCESS filter the batch, and the header columns the
    selection reads must be filtered with it."""
    fixture = fixture_root / "snana" / "salt3_fit_select"
    version = "JAX_TEST_SALT3_FIT_SELECT"

    sim_root = tmp_path / "sim"
    vdir = sim_root / version
    vdir.mkdir(parents=True)
    (vdir / f"{version}_HEAD.FITS").write_bytes(
        gzip.decompress((fixture / "HEAD.FITS.gz").read_bytes()))
    (vdir / f"{version}_PHOT.FITS").write_bytes(
        gzip.decompress((fixture / "PHOT.FITS.gz").read_bytes()))
    (vdir / f"{version}.LIST").write_text(f"{version}_HEAD.FITS\n")

    env = os.environ.copy()
    env["JAX_TEST_FIXTURE_ROOT"] = str(data_root)
    env["JAX_TEST_FIT_SIM_DIR"] = str(sim_root)

    def run(name, *overrides):
        out = tmp_path / name
        r = subprocess.run(
            [sys.executable, str(repo_root / "snlc_fit.py"),
             str(fixture / "fit.nml"), "BATCH_SIZE", "4", *overrides, "OUTPUT_NAME", str(out)],
            capture_output=True, text=True, env=env, timeout=900)
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
        return sum(1 for l in out.read_text().splitlines() if l.startswith("SN:"))

    # the fixture's nml already carries SNTYPE_LIST, which is the header cut
    # whose columns must stay aligned with the filtered batch
    n_all = run("SPLITALL.FITRES.TEXT")
    n_third = run("SPLIT13.FITRES.TEXT", "JOBSPLIT", "1,3")
    n_capped = run("CAP20.FITRES.TEXT", "MXEVT_PROCESS", "20")

    assert n_all > 0
    # one job in three takes roughly a third, and never more than the whole
    assert 0 < n_third < n_all
    assert abs(n_third - n_all / 3) <= max(3, n_all * 0.15)
    # the cap counts events that passed the header cuts, so it bounds the output
    assert 0 < n_capped <= 20


def test_event_caps_and_selection_scope(tmp_path, repo_root, fixture_root,
                                        data_root):
    """MXLC_FIT caps events surviving the fit rather than events read
    (snana.F90:3204). OPT_SNCID_LIST bit 0 is covered by
    test_opt_sncid_list_bit0_skips_the_post_fit_cuts."""
    fixture = fixture_root / "snana" / "salt3_fit_dense"
    version = "JAX_TEST_SALT3_FIT_DENSE"
    vdir = tmp_path / version
    vdir.mkdir()
    (vdir / f"{version}_HEAD.FITS").write_bytes(
        gzip.decompress((fixture / "HEAD.FITS.gz").read_bytes()))
    (vdir / f"{version}_PHOT.FITS").write_bytes(
        gzip.decompress((fixture / "PHOT.FITS.gz").read_bytes()))
    (vdir / f"{version}.LIST").write_text(f"{version}_HEAD.FITS\n")

    env = os.environ.copy()
    env["JAX_TEST_FIXTURE_ROOT"] = str(data_root)
    env["JAX_TEST_FIT_SIM_DIR"] = str(tmp_path)

    def rows(name, *overrides):
        out = tmp_path / name
        r = subprocess.run(
            [sys.executable, str(repo_root / "snlc_fit.py"),
             str(fixture / "fit.nml"), "BATCH_SIZE", "4", *overrides, "OUTPUT_NAME", str(out)],
            capture_output=True, text=True, env=env, timeout=900)
        assert r.returncode == 0, r.stdout[-1200:] + r.stderr[-1200:]
        return sum(1 for l in out.read_text().splitlines() if l.startswith("SN:"))

    base = rows("CAPBASE.FITRES.TEXT")
    assert base > 12
    # MXLC_FIT bounds the events that survive the fit, not the events read
    assert rows("CAP12.FITRES.TEXT", "MXLC_FIT", "12") == 12
    # and it is a cap, not a target: a bound above the sample changes nothing
    assert rows("CAPBIG.FITRES.TEXT", "MXLC_FIT", str(base + 50)) == base


def test_uvlam_extrapflux_reaches_the_model(tmp_path, repo_root, fixture_root,
                                            data_root):
    """UVLAM_EXTRAPFLUX extends the SED below its native UV edge, which is only
    visible if the working wavelength grid reaches down there too."""
    fixture = fixture_root / "snana" / "salt3_fit_dense"
    version = "JAX_TEST_SALT3_FIT_DENSE"
    vdir = tmp_path / version
    vdir.mkdir()
    (vdir / f"{version}_HEAD.FITS").write_bytes(
        gzip.decompress((fixture / "HEAD.FITS.gz").read_bytes()))
    (vdir / f"{version}_PHOT.FITS").write_bytes(
        gzip.decompress((fixture / "PHOT.FITS.gz").read_bytes()))
    (vdir / f"{version}.LIST").write_text(f"{version}_HEAD.FITS\n")

    env = os.environ.copy()
    env["JAX_TEST_FIXTURE_ROOT"] = str(data_root)
    env["JAX_TEST_FIT_SIM_DIR"] = str(tmp_path)

    def x1(name, *overrides):
        out = tmp_path / name
        r = subprocess.run(
            [sys.executable, str(repo_root / "snlc_fit.py"),
             str(fixture / "fit.nml"), "BATCH_SIZE", "4", *overrides, "MAX_FIT", "8",
             "OUTPUT_NAME", str(out)],
            capture_output=True, text=True, env=env, timeout=900)
        assert r.returncode == 0, r.stdout[-1200:] + r.stderr[-1200:]
        names, vals = None, []
        for line in out.read_text().splitlines():
            if line.startswith("VARNAMES:"):
                names = line.split()[1:]
            elif line.startswith("SN:") and names:
                vals.append(float(line.split()[1:][names.index("x1")]))
        return vals

    off = x1("UVOFF.FITRES.TEXT")
    on = x1("UVON.FITRES.TEXT", "UVLAM_EXTRAPFLUX", "1000")
    assert len(off) == len(on) == 8
    # the key must actually move the fit; it was inert while the working grid
    # started at the model's own 2000 A floor
    assert max(abs(a - b) for a, b in zip(off, on)) > 1e-3


def test_replay_cli_runs_on_bundled_fixture(repo_root, tmp_path) -> None:
    """The documented replay command line runs end-to-end on a bundled
    fixture (the CLI once broke while the parity tests imported its
    functions directly, so the entry point itself is exercised here)."""
    fixture = repo_root / "tests" / "fixtures" / "snana" / "salt3_k21_replay"
    result = subprocess.run(
        [sys.executable, str(repo_root / "replay_snana_dump.py"),
         "--input", str(fixture / "input.input"),
         "--dump", str(fixture / "DUMP"),
         "--head", str(fixture / "HEAD.FITS.gz"),
         "--output", "REPLAY_SMOKE"],
        capture_output=True, text=True, timeout=600, cwd=str(tmp_path),
        env={**os.environ, "PYTHONPATH": str(repo_root)},
    )
    assert result.returncode == 0, (
        f"replay CLI exited {result.returncode}\n"
        f"stdout:\n{result.stdout[-3000:]}\nstderr:\n{result.stderr[-3000:]}"
    )


def test_epoch_binning_leaves_every_output_table_unchanged(
        tmp_path, repo_root, fixture_root, data_root):
    """Binning by epoch count is padding bookkeeping, so the tables must agree.

    SNTABLE_LIST asks for OUTLIER and LCPLOT as well as FITRES, because those
    are the only tables carrying the model fluxes and the fit mask, and they
    are built from accumulators that the binning reorders. A FITRES-only
    comparison passes without ever touching them.
    """
    fixture = fixture_root / "snana" / "salt3_fit_select"
    version = "JAX_TEST_SALT3_FIT_SELECT"
    vdir = tmp_path / "sim" / version
    vdir.mkdir(parents=True)
    (vdir / f"{version}_HEAD.FITS").write_bytes(
        gzip.decompress((fixture / "HEAD.FITS.gz").read_bytes()))
    (vdir / f"{version}_PHOT.FITS").write_bytes(
        gzip.decompress((fixture / "PHOT.FITS.gz").read_bytes()))
    (vdir / f"{version}.LIST").write_text(f"{version}_HEAD.FITS\n")

    # CUTWIN_NFILT_SNRMAX drives the post-fit recount, whose per-SN keep mask
    # is the fourth accumulator the binning reorders; without it that mask is
    # never built and its reordering goes untested.
    nml = tmp_path / "fit.nml"
    nml.write_text("\n".join(
        "  SNTABLE_LIST = 'SNANA FITRES OUTLIER(nsig:1) LCPLOT'\n"
        "  CUTWIN_NFILT_SNRMAX = 3, 99"
        if line.strip().startswith("SNTABLE_LIST") else line
        for line in (fixture / "fit.nml").read_text().splitlines()) + "\n")

    env = os.environ.copy()
    env["JAX_TEST_FIXTURE_ROOT"] = str(data_root)
    env["JAX_TEST_FIT_SIM_DIR"] = str(tmp_path / "sim")

    def run(n_bins):
        out = tmp_path / f"bins{n_bins}"
        out.mkdir()
        r = subprocess.run(
            [sys.executable, str(repo_root / "snlc_fit.py"), str(nml), "BATCH_SIZE", "4",
             "NUM_NOBS_BINS", str(n_bins), "DOUBLE_PRECISION", "T",
             "OUTPUT_NAME", str(out / "FITRES.TEXT")],
            capture_output=True, text=True, env=env, timeout=1800)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        return out, r.stdout

    one, log_one = run(1)
    four, log_four = run(4)
    assert "Epoch bins padded to" in log_four, \
        "the fixture no longer splits, so this would pass vacuously"
    assert "Epoch bins padded to" not in log_one

    def rows(path, tag):
        return [l.split() for l in path.read_text().splitlines()
                if l.startswith(tag)]

    a, b = rows(one / "FITRES.TEXT", "SN:"), rows(four / "FITRES.TEXT", "SN:")
    assert len(a) == len(b) > 0
    assert [r[1] for r in a] == [r[1] for r in b], "binning reordered the CIDs"
    for ra, rb in zip(a, b):
        for x, y in zip(ra, rb):
            try:
                fx, fy = float(x), float(y)
            except ValueError:
                assert x == y
                continue
            assert fx == pytest.approx(fy, rel=1e-4, abs=1e-6, nan_ok=True)

    # OUTLIER lists one row per outlying epoch, LCPLOT one per fitted epoch;
    # both are built from the reordered model fluxes and fit mask. Compare the
    # values too: row counts and CIDs survive a mis-permuted model flux, and
    # those columns are the only place it is written out.
    for tag, name in (("SN:", "OUTLIER.TEXT"), ("OBS:", "LCPLOT.TEXT")):
        ra = rows(one / f"{version}.{name}", tag)
        rb = rows(four / f"{version}.{name}", tag)
        assert len(ra) == len(rb) > 0, f"{name}: row count changed"
        assert [r[1] for r in ra] == [r[1] for r in rb], \
            f"{name}: rows are for different SNe"
        for r_one, r_four in zip(ra, rb):
            for x, y in zip(r_one, r_four):
                try:
                    fx, fy = float(x), float(y)
                except ValueError:
                    assert x == y, f"{name}: {x} != {y}"
                    continue
                assert fx == pytest.approx(fy, rel=1e-3, abs=1e-6, nan_ok=True), \
                    f"{name}: {x} != {y}"


def test_salt2mu_cli_end_to_end(tmp_path, repo_root, fixture_root) -> None:
    """``salt2mu.py`` fits the bundled SALT3 FITRES and writes the three files.

    The values are compared against SALT2mu.exe in tests/parity/; what only a
    subprocess reaches is the key=value argument form and the prefix rule.
    """
    prefix = tmp_path / "OUT"
    fitres = fixture_root / "snana" / "salt3_fit_dense" / "FITRES.TEXT"
    result = subprocess.run(
        [sys.executable, str(repo_root / "salt2mu.py"),
         f"datafile={fitres}", "nzbin=5", "zmin=0.02", "zmax=0.6",
         f"prefix={prefix}"],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    for extension in (".M0DIF", ".FITRES", ".COV"):
        assert prefix.with_suffix(extension).is_file(), f"missing {extension}"


def test_wfit_cli_end_to_end(tmp_path, repo_root, fixture_root) -> None:
    """``wfit.py`` fits the bundled Hubble diagram and writes a cospar file.

    The Hubble diagram is copied into the temporary directory because wfit
    writes its YAML summary beside its input.
    """
    hubble_diagram = tmp_path / "SALT2mu.M0DIF"
    hubble_diagram.write_bytes(
        (fixture_root / "wfit" / "SALT2mu.M0DIF").read_bytes())
    cospar = tmp_path / "out.cospar"
    result = subprocess.run(
        [sys.executable, str(repo_root / "wfit.py"), str(hubble_diagram),
         "-outfile_cospar", str(cospar)],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert cospar.is_file()


def test_create_covariance_cli_end_to_end(tmp_path, repo_root) -> None:
    """``create_covariance.py`` builds the bundled FITOPT x MUOPT grid.

    The config names its inputs relative to the repository root, so the run
    starts there and writes to a temporary directory through ``-o``.
    """
    config = "tests/fixtures/snana/create_cov/input.yml"
    result = subprocess.run(
        [sys.executable, str(repo_root / "create_covariance.py"),
         config, "-o", str(tmp_path)],
        capture_output=True, text=True, timeout=300, cwd=str(repo_root),
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert (tmp_path / "INFO.YML").is_file()
    assert (tmp_path / "hubble_diagram.txt").is_file()
    assert (tmp_path / "covtot_inv_000.npz").is_file()


def test_jax_pipeline_main_runs_a_config(tmp_path, repo_root,
                                         fixture_root, data_root) -> None:
    """``python -m jax_pipeline.pippin`` runs a one-stage Pippin config end
    to end.

    Called in a subprocess with this checkout on PYTHONPATH: the installed
    console script resolves to whichever checkout was pip-installed.
    """
    config = tmp_path / "analysis.yml"
    config.write_text(
        "SIM:\n"
        "  DATA:\n"
        "    IA:\n"
        f"      BASE: {fixture_root / 'snana' / 'bbc_des_data' / 'input.input'}\n"
        "      NGENTOT_LC: 40\n"
        "    GLOBAL:\n"
        "      RANSEED_REPEAT: 1 11\n"
    )
    env = os.environ.copy()
    env["JAX_TEST_FIXTURE_ROOT"] = str(data_root)
    env["PYTHONPATH"] = str(repo_root)
    outdir = tmp_path / "run"
    result = subprocess.run(
        [sys.executable, "-m", "jax_pipeline.pippin",
         str(config), "--outdir", str(outdir)],
        capture_output=True, text=True, timeout=300, env=env,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert (outdir / "MANIFEST.yml").is_file()
    assert (outdir / "DATA" / "DATA.LIST").is_file()
    assert (outdir / "DATA" / "DONE.yml").is_file()
    # the run narrates: without basicConfig every logger.info went nowhere
    assert "running DATA" in result.stdout




def test_named_and_snana_cli_fits_write_identical_rows(tmp_path, repo_root, fixture_root, data_root):
    import yaml
    fixture = fixture_root / 'snana' / 'salt3_fit_dense'
    env = _stage_fit_version(tmp_path, fixture / 'HEAD.FITS', fixture / 'PHOT.FITS', data_root)
    shared = dict(FITMODEL_NAME=str(data_root / 'models/SALT3.K21'),
        KCOR_FILE=str(data_root / 'kcor/JAX_TEST_LSST.fits'),
        VERSION_PHOTOMETRY='JAX_TEST_SALT3_FIT_DENSE', PRIVATE_DATA_PATH=str(tmp_path),
        BATCH_SIZE=2, MAX_FIT=2, NFIT_ITERATION=1, MAXITER_ROUND1=2,
        OPT_MWEBV=0, USE_MODEL_MAGERR=False)
    nml = tmp_path / 'legacy.nml'
    nml.write_text('&FITINP\n' + ''.join(
        f" {key} = {str(value) if isinstance(value, (int, bool)) else repr(value)}\n"
        for key, value in shared.items()) +
        " FILTLIST_FIT = 'griz'\n INIVAL_x1 = 0.5\n INISTP_x1 = 0\n&END\n")
    named = {key.lower(): value for key, value in shared.items() if key != 'NFIT_ITERATION'}
    named.update(n_fit_iterations=1, filtlist_fit=['LSST-' + band for band in 'griz'], fixed={'x1': .5})
    yml = tmp_path / 'named.yaml'
    yml.write_text(yaml.safe_dump(named))
    rows = []
    for path, options in ((nml, ['OUTPUT_NAME']), (yml, ['--output-name'])):
        output = tmp_path / (path.stem + '.FITRES.TEXT')
        result = subprocess.run([sys.executable, str(repo_root / 'snlc_fit.py'),
            str(path), *options, str(output)], env=env, capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, result.stdout + result.stderr
        rows.append([line for line in output.read_text().splitlines() if line.startswith('SN:')])
    assert len(rows[0]) == 2
    assert rows[0] == rows[1]
