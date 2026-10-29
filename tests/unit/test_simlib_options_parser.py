"""Parser-level tests for SIMLIB filter options + GENRANGE_RA/DECL."""

import pytest

from jax_snlc_core.logging import SNANAError
from jax_snlc_sim.inputs import parse_snana_input_file


def _write_input(path, *, kcor_path, simlib_path, model_path, extra_lines=None):
    lines = [
        "GENVERSION: TEST",
        "NGENTOT_LC: 100",
        "FORMAT_MASK: 32",
        "GENTYPE: 1",
        "SNTYPES_Ia: 1 101",
        "CIDRAN_MIN: 0",
        "CIDRAN_MAX: 3000000",
        f"GENMODEL: {model_path}",
        "GENPEAK_SALT2x1: 0.0",
        "GENRANGE_SALT2x1: -2.0 2.0",
        "GENSIGMA_SALT2x1: 1.0 1.0",
        "GENPEAK_SALT2c: 0.0",
        "GENRANGE_SALT2c: -0.2 0.3",
        "GENSIGMA_SALT2c: 0.05 0.05",
        "GENPEAK_SALT2ALPHA: 0.14",
        "GENPEAK_SALT2BETA: 3.1",
        "OMEGA_MATTER: 0.3150",
        "OMEGA_LAMBDA: 0.6850",
        "w0_LAMBDA: -1.0000",
        "wa_LAMBDA: 0",
        "GENFILTERS: ugrizY",
        f"KCOR_FILE: {kcor_path}",
        f"SIMLIB_FILE: {simlib_path}",
        "GENRANGE_REDSHIFT: 0.05 0.30",
        "GENRANGE_MJD: 60500.0 61500.0",
        "GENRANGE_PEAKMJD: 60540.0 61460.0",
        "GENSIGMA_REDSHIFT: 0.001",
        "GENRANGE_TREST: -20.0 50.0",
        "SOLID_ANGLE: 0.015",
        "OPT_MWEBV: 0",
        "DNDZ: POWERLAW 2.5E-5 1.5",
        "RANSEED: 12345",
    ]
    if extra_lines:
        lines.extend(extra_lines)
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def minimal(tmp_path, lsst_kcor_path, lsst_simlib_path, data_root):
    def _factory(extra_lines=None):
        out = tmp_path / "test.input"
        _write_input(out, kcor_path=str(lsst_kcor_path),
                     simlib_path=str(lsst_simlib_path),
                     model_path=str(data_root / "models" / "SALT3.K21"),
                     extra_lines=extra_lines)
        return out
    return _factory


def test_defaults_when_no_keys_set(minimal):
    cfg = parse_snana_input_file(minimal())
    s = cfg.survey
    assert s.simlib_minseason == 0.0
    assert s.simlib_minobs == 1
    assert s.simlib_idlock == -9
    assert s.simlib_idskip == ()
    assert s.simlib_nrepeat == 1
    assert s.simlib_maxranstart == 0
    assert s.simlib_mskopt == 0
    assert s.genrange_ra == (-360.0, 360.0)
    assert s.genrange_decl == (-90.0, 90.0)
    assert s.genrange_b == (-90.0, 90.0)


def test_all_scalar_keys_parse(minimal):
    cfg = parse_snana_input_file(minimal([
        "SIMLIB_MINSEASON: 30",
        "SIMLIB_MINOBS: 50",
        "SIMLIB_IDLOCK: 42",
        "SIMLIB_NREPEAT: 3",
        "SIMLIB_MAXRANSTART: 100",
        "SIMLIB_MSKOPT: 24",
        "GENRANGE_RA: 30.0 60.0",
        "GENRANGE_DECL: -10.0 10.0",
        "GENRANGE_b: -20.0 20.0",
    ]))
    s = cfg.survey
    assert (s.simlib_minseason, s.simlib_minobs, s.simlib_idlock, s.simlib_nrepeat, s.simlib_maxranstart, s.simlib_mskopt,
            s.genrange_ra, s.genrange_decl, s.genrange_b) == (
        30.0, 50, 42, 3, 100, 24, (30.0, 60.0), (-10.0, 10.0), (-20.0, 20.0))


def test_idskip_repeated_lines_accumulate(minimal):
    cfg = parse_snana_input_file(minimal([
        "SIMLIB_IDSKIP: 10", "SIMLIB_IDSKIP: 20", "SIMLIB_IDSKIP: 30",
    ]))
    assert cfg.survey.simlib_idskip == (10, 20, 30)


@pytest.mark.parametrize("bit", [2, 512])
def test_refuse_unsupported_mskopt_bit(minimal, bit):
    with pytest.raises(SNANAError, match=f"bit {bit}"):
        parse_snana_input_file(minimal([f"SIMLIB_MSKOPT: {bit}"]))


def test_cadence_control_keys_parse(minimal):
    """SIMLIB cadence-control keys parse into the Survey knobs."""
    cfg = parse_snana_input_file(minimal([
        "SIMLIB_MAXOBS: 200",
        "SIMLIB_IDSTART: 5",
        "SIMLIB_PRESCALE: 3",
        "SIMLIB_MJD_SHIFT: 12.5",
        "SIMLIB_FIELDLIST: C1,C2",
    ]))
    s = cfg.survey
    assert (s.simlib_maxobs, s.simlib_idstart, s.simlib_prescale, s.simlib_mjd_shift, s.simlib_fieldlist) == (
        200, 5, 3, 12.5, "C1,C2")


def test_cadence_controls_filter_fields(minimal):
    """IDSTART/PRESCALE/MJD_SHIFT/MAXOBS/FIELDLIST select or transform LIBIDs."""
    from jax_snlc_sim import builder

    def load(extra=None):
        return builder._load_simlib(parse_snana_input_file(minimal(extra)))

    base = load()
    libids = sorted(f.libid for f in base.fields)
    assert len(libids) >= 2  # fixture has multiple LIBIDs

    # IDSTART drops LIBIDs below the threshold
    kept = load(["SIMLIB_IDSTART: 1"])
    assert all(f.libid >= 1 for f in kept.fields)
    assert len(kept.fields) < len(base.fields)

    # PRESCALE keeps LIBIDs where id % N == 0
    kept = load(["SIMLIB_PRESCALE: 2"])
    assert all(f.libid % 2 == 0 for f in kept.fields)

    # MJD_SHIFT adds a constant to every MJD
    mjd0 = float(base.fields[0].mjds[0])
    shifted = load(["SIMLIB_MJD_SHIFT: 100.0"])
    assert abs(float(shifted.fields[0].mjds[0]) - (mjd0 + 100.0)) < 1e-3

    # MAXOBS drops LIBIDs with more than N observations; with none left the
    # load aborts rather than handing the draw an empty library
    with pytest.raises(SNANAError, match="no SIMLIB entry survives"):
        load(["SIMLIB_MAXOBS: 1"])

    # FIELDLIST does NOT prune the pool, unlike the keys above: SNANA keeps the
    # full LIBID cycle and rejects the events that land on excluded entries
    # (SKIP_SIMLIB_FIELD, snlc_sim.c:22587). The per-entry verdict is recorded
    # for that per-event test, -1 meaning "field not listed".
    unmatched = load(["SIMLIB_FIELDLIST: NOSUCHFIELD"])
    assert len(unmatched.fields) == len(base.fields)
    assert all(ps == -1 for ps in unmatched.field_prescales)
    matched = load(["SIMLIB_FIELDLIST: " + base.fields[0].field_name])
    assert len(matched.fields) == len(base.fields)
    assert matched.field_prescales[0] == 1


def test_forceval_psf_overrides_all_obs(minimal):
    """FORCEVAL_PSF: X forces every per-obs PSF sigma to X across all SIMLIB fields."""
    from jax_snlc_sim import builder
    cfg = parse_snana_input_file(minimal(["FORCEVAL_PSF: 1.5"]))
    simlib = builder._load_simlib(cfg)
    assert simlib.fields
    for field in simlib.fields:
        assert (field.psf1s == 1.5).all()
    # control: without the key, the real PSFs are not uniformly 1.5
    simlib0 = builder._load_simlib(parse_snana_input_file(minimal()))
    assert not all((f.psf1s == 1.5).all() for f in simlib0.fields)


def test_resolve_input_path_uses_path_user_input(tmp_path):
    """PATH_USER_INPUT provides a fallback search dir for bare-name input files."""
    from jax_snlc_sim.inputs import resolve_input_path
    from jax_snlc_sim.simulation import Run
    d = tmp_path / "inputs"
    d.mkdir()
    f = d / "my.KCOR"
    f.write_text("x")
    run = Run(path_user_input=str(d))
    # a bare name not found in cwd resolves from PATH_USER_INPUT
    assert resolve_input_path("my.KCOR", run) == str(f)
    # an already-resolvable path is returned unchanged
    assert resolve_input_path(str(f), run) == str(f)
    # empty stays empty (no false-positive match to the directory)
    assert resolve_input_path("", run) == ""
    # no PATH_USER_INPUT → plain expandvars passthrough
    assert resolve_input_path("nope.KCOR", Run()) == "nope.KCOR"


def test_reload_reads_updated_simlib(minimal, lsst_simlib_path, tmp_path):
    """Rebuilding from the same path must see changed observing conditions."""
    from jax_snlc_sim.builder import _load_simlib

    source = tmp_path / "updated.SIMLIB"
    original = lsst_simlib_path.read_text()
    source.write_text(original)
    config = parse_snana_input_file(minimal())
    config.survey.simlib_file = str(source)
    first = _load_simlib(config)
    source.write_text(original.replace("MWEBV: 0.0200", "MWEBV: 0.0400"))
    second = _load_simlib(config)
    assert first.fields[0].mwebv == pytest.approx(0.02)
    assert second.fields[0].mwebv == pytest.approx(0.04)


def test_synthetic_epochs_do_not_leak_between_builds(minimal, data_root):
    """Appending synthetic photometry must not alter a later SIMLIB load."""
    from jax_snlc_sim.builder import _load_kcor, _load_simlib
    from jax_snlc_sim.simlib import append_spectrograph_photometry
    from jax_snlc_sim.spectra import load_spectrograph

    config = parse_snana_input_file(minimal())
    config.survey.simlib_file = str(data_root / "simlib/JAX_TEST_LSST_SIMSPEC.SIMLIB")
    config.survey.kcor_file = str(data_root / "kcor/JAX_TEST_LSST_SPECTRO.fits")
    config.survey.use_simlib_spectrograph = True
    first = _load_simlib(config)
    original_counts = [field.nobs for field in first.fields]
    kcor = _load_kcor(config)
    calibration = load_spectrograph(config, kcor)
    append_spectrograph_photometry(
        first, calibration, kcor, config.spectra.spectrograph_scale_texpose)
    assert any(field.nobs > count for field, count in zip(first.fields, original_counts))
    second = _load_simlib(config)
    assert [field.nobs for field in second.fields] == original_counts
