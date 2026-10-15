"""Tests for spectrograph calibration reader, solver, and SNR computation."""

import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.io import fits

from jax_snlc_core.physics.dust_laws import load_dust_law
from jax_snlc_core.utils.error_handling import SNANAError
from jax_snlc_sim.simulation import Output, SimulationConfig
from jax_snlc_sim.config_parser import parse_take_spectrum_entry, prepare_spectrum_requests
from jax_snlc_core.photometry.spectrograph import (
    SpectrographCalibration,
    _solve_calibration,
    get_snr_from_calib,
    read_spectrograph,
)


@pytest.fixture(scope="module")
def testdata(spectrograph_data_dir):
    return spectrograph_data_dir / "SPECTROGRAPH_TEST.DAT"


def _get_snr_simple(calib, texpose_s, texpose_t, genmag):
    snr, ef, _, _ = get_snr_from_calib(calib, texpose_s, texpose_t, genmag)
    return np.asarray(snr), np.asarray(ef)


# =============================================================================
# Reader & solver
# =============================================================================


def test_reader(testdata):
    calib = read_spectrograph(str(testdata))
    assert calib.instrument_name == "TEST_SPECTRO"
    assert np.allclose(calib.magref, [20.0, 24.0])
    assert np.allclose(calib.texpose_list, [600.0, 1800.0, 5400.0])
    # 30 SPECBIN rows plus one extended bin at each edge
    obs = ~np.asarray(calib.islam_extend)
    assert calib.n_lam == 32 and obs.sum() == 30
    assert calib.n_texpose == 3
    assert float(np.asarray(calib.lam_min)[obs][0]) == 3500.0
    assert float(np.asarray(calib.lam_max)[obs][-1]) == 9000.0
    assert np.allclose(calib.lam_avg, 0.5 * (calib.lam_min + calib.lam_max))
    assert np.allclose(calib.lam_bin, calib.lam_max - calib.lam_min)
    assert np.allclose(calib.log_texpose_list, jnp.log10(calib.texpose_list))
    assert calib.zp.shape == (32, 3)
    assert calib.sqsigsky.shape == (32, 3)
    assert bool(jnp.all(calib.zp > 0))
    assert bool(jnp.all(calib.sqsigsky >= 0))
    assert isinstance(calib.zp, jnp.ndarray)


def test_reader_rejects_nonmonotonic_snr():
    """A decreasing SNR(Texpose) aborts. It trips the Poisson vsTEXPOSE
    ratio check, which SNANA runs before its monotonic check and which any
    decrease violates at the 1.2 default tolerance (check_SNR_SPECTROGRAPH,
    sntools_spectrograph.c:466-500)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
        f.write("INSTRUMENT: BAD\nMAGREF_LIST: 20.0 24.0\n"
                "TEXPOSE_LIST: 600 1800\n"
                "SPECBIN: 5000 5200 3.0  20.0 1.3  18.0 1.2\n")
        tmppath = f.name
    try:
        with pytest.raises(SNANAError, match="vs. Texpose"):
            read_spectrograph(tmppath)
    finally:
        os.unlink(tmppath)


def test_reader_rejects_snr_off_poisson_magref():
    """SNR2/SNR1 far from the Poisson sqrt(fluxref ratio) aborts
    (check_SNR_SPECTROGRAPH, sntools_spectrograph.c:440-458). A 4-mag
    reference gap predicts SNR2/SNR1~0.158; 0.9 is over the factor-5
    tolerance."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
        f.write("INSTRUMENT: BAD\nMAGREF_LIST: 20.0 24.0\n"
                "TEXPOSE_LIST: 600 1800\n"
                "SPECBIN: 5000 5200 3.0  20.0 18.0  34.6 31.2\n")
        tmppath = f.name
    try:
        with pytest.raises(SNANAError, match="SNR2/SNR1"):
            read_spectrograph(tmppath)
    finally:
        os.unlink(tmppath)


def test_reader_poisson_tolerance_keys_are_read():
    """SNR_POISSON_RATIO_ABORT_* keys relax the tolerances
    (read_spectrograph_text, sntools_spectrograph.c:305-306 defaults)."""
    body = ("INSTRUMENT: OK\nMAGREF_LIST: 20.0 24.0\n"
            "TEXPOSE_LIST: 600 1800\n"
            "SNR_POISSON_RATIO_ABORT_vsMAGREF: 40.0\n"
            "SNR_POISSON_RATIO_ABORT_vsTEXPOSE: 3.0\n"
            "SPECBIN: 5000 5200 3.0  20.0 18.0  60.0 54.0\n")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
        f.write(body)
        tmppath = f.name
    try:
        calib = read_spectrograph(tmppath)
        assert calib is not None
    finally:
        os.unlink(tmppath)


def test_solve_rejects_inconsistent_magsnr():
    """magCheck below MAGSNR_TOLERANCE_ABORT=+0.001 aborts
    (solve_spectrograph, sntools_spectrograph.c:705-713, init :308)."""
    # SNR0/SNR1 = 50 implies a 4.25 mag gap, above the 4-mag reference gap
    with pytest.raises(SNANAError, match="failed solution check"):
        _solve_calibration(np.array([[100.0]]), np.array([[2.0]]), 20.0, 24.0)


def test_comments_and_blanks():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
        f.write("# Comment\n\nINSTRUMENT: COMMENT_TEST\n# Another\n")
        f.write("MAGREF_LIST: 20.0 24.0\nTEXPOSE_LIST: 1000\n\n")
        f.write("SPECBIN: 5000 5200 3.0  20.0 1.3\n# End\n")
        tmppath = f.name
    try:
        calib = read_spectrograph(tmppath)
        assert calib.instrument_name == "COMMENT_TEST"
        assert int((~np.asarray(calib.islam_extend)).sum()) == 1
    finally:
        os.unlink(tmppath)


def test_zp_increases_with_texpose(testdata):
    calib = read_spectrograph(str(testdata))
    zp = np.asarray(calib.zp)
    assert np.all(zp[:, -1] >= zp[:, 0])


def test_negative_sqsigsky_allowed():
    """SNR above the Poisson limit gives negative SQSIGSKY; SNANA allows this
    (solve_spectrograph aborts only on inf/round-trip, sntools_spectrograph.c:745)
    — the IDEAL spectrograph needs it."""
    _, sq = _solve_calibration(np.array([[10.0]]), np.array([[10.0]]), 20.0, 24.0)
    assert sq[0, 0] < 0.0


def test_snr_monotonicity(testdata):
    calib = read_spectrograph(str(testdata))
    snr_bright, _ = _get_snr_simple(calib, 1800.0, 0.0, jnp.full(calib.n_lam, 20.0))
    snr_faint, _ = _get_snr_simple(calib, 1800.0, 0.0, jnp.full(calib.n_lam, 24.0))
    assert np.all(snr_bright > snr_faint)

    genmag = jnp.full(calib.n_lam, 22.0)
    snr_short, _ = _get_snr_simple(calib, 600.0, 0.0, genmag)
    snr_long, _ = _get_snr_simple(calib, 5400.0, 0.0, genmag)
    assert np.all(snr_long > snr_short)


def test_snr_interpolation(testdata):
    calib = read_spectrograph(str(testdata))
    genmag = jnp.full(calib.n_lam, 22.0)
    snr_low, _ = _get_snr_simple(calib, 600.0, 0.0, genmag)
    snr_mid, _ = _get_snr_simple(calib, 1200.0, 0.0, genmag)
    snr_high, _ = _get_snr_simple(calib, 1800.0, 0.0, genmag)
    assert np.all(snr_mid >= snr_low - 1e-6)
    assert np.all(snr_mid <= snr_high + 1e-6)


def test_template_noise(testdata):
    calib = read_spectrograph(str(testdata))
    genmag = jnp.full(calib.n_lam, 22.0)
    snr_no_t, _ = _get_snr_simple(calib, 1800.0, 0.0, genmag)
    snr_with_t, errfrac = _get_snr_simple(calib, 1800.0, 600.0, genmag)
    assert np.all(snr_with_t <= snr_no_t + 1e-6)
    assert np.all(errfrac >= 0)
    assert np.any(errfrac > 0)


def test_sentinel_magnitude(testdata):
    calib = read_spectrograph(str(testdata))
    snr, _ = _get_snr_simple(calib, 1800.0, 0.0, jnp.full(calib.n_lam, 99.0))
    assert np.all(snr == 0.0)


# =============================================================================
# Config parsing
# =============================================================================


def test_config_parsing():
    """TAKE_SPECTRUM directives parse into per-config dicts."""
    from jax_snlc_sim.config_parser import read_keys

    content = (
        "GENVERSION: TEST\nGENMODEL: SALT3\nGENFILTERS: griz\n"
        "GENTYPE: 1\nSNTYPES_Ia: 1 101\n"
        "GENRANGE_REDSHIFT: 0.01 0.4\nGENRANGE_PEAKMJD: 60000 61000\n"
        "GENRANGE_TREST: -20 50\nKCOR_FILE: d.fits\nSIMLIB_FILE: d.simlib\nNGENTOT_LC: 10\n"
        "OMEGA_MATTER: 0.315\nOMEGA_LAMBDA: 0.685\nw0_LAMBDA: -1.0\n"
        "RANSEED: 42\nSOLID_ANGLE: 0.015\n"
        "TAKE_SPECTRUM: TREST(-5:20) SNR_ZPOLY(50,0) SNR_LAMREST(5000:6000)\n"
        "TAKE_SPECTRUM(DEEP/2): TOBS(10:40) TEXPOSE_ZPOLY(3600,0)\n"
        "TAKE_SPECTRUM: HOST TEXPOSE_ZPOLY(1800,0)\n"
        "SPECTROGRAPH_FILE: $SNDATA_ROOT/test.dat\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".input", delete=False) as f:
        f.write(content)
        tmppath = f.name
    try:
        specs = read_keys(tmppath).get("TAKE_SPECTRUM", [])
        assert len(specs) == 3
        assert specs[0]["epoch_frame"] == "REST"
        assert specs[0]["epoch_range"] == [-5.0, 20.0]
        assert specs[0]["opt_texpose"] == 2
        assert specs[0]["zpoly_snr"] == [50.0, 0.0]
        assert specs[0]["snr_lamrange"] == [5000.0, 6000.0]
        assert specs[1]["field"] == "DEEP"
        assert abs(specs[1]["wgt"] - 0.5) < 1e-10
        assert specs[2]["epoch_frame"] == "HOST"
    finally:
        os.unlink(tmppath)


def test_config_parsing_bare_snr_alias():
    """Bare SNR(...) is an alias for SNR_ZPOLY(...) (snlc_sim.c:5831).

    Line taken verbatim from the SNANA manual's "Creating a SIMLIB from Data"
    section, where SIMLIB headers carry the measured SNR of each real spectrum.
    """
    from jax_snlc_sim.config_parser import read_keys

    content = (
        "GENVERSION: TEST\nGENMODEL: SALT3\nGENFILTERS: griz\n"
        "GENTYPE: 1\nSNTYPES_Ia: 1 101\n"
        "GENRANGE_REDSHIFT: 0.01 0.4\nGENRANGE_PEAKMJD: 60000 61000\n"
        "GENRANGE_TREST: -20 50\nKCOR_FILE: d.fits\nSIMLIB_FILE: d.simlib\nNGENTOT_LC: 10\n"
        "OMEGA_MATTER: 0.315\nOMEGA_LAMBDA: 0.685\nw0_LAMBDA: -1.0\n"
        "RANSEED: 42\nSOLID_ANGLE: 0.015\n"
        "TAKE_SPECTRUM: TOBS(4.6) SNR(1179) SNR_LAMOBS(3590:7535)\n"
        "SPECTROGRAPH_FILE: $SNDATA_ROOT/test.dat\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".input", delete=False) as f:
        f.write(content)
        tmppath = f.name
    try:
        specs = read_keys(tmppath).get("TAKE_SPECTRUM", [])
        assert len(specs) == 1
        assert specs[0]["opt_texpose"] == 2
        assert specs[0]["zpoly_snr"] == [1179.0]
        assert specs[0]["epoch_frame"] == "OBS"
        assert specs[0]["epoch_range"] == [4.6, 4.6]
        # SNR_LAMOBS must not be swallowed by the bare-SNR prefix
        assert specs[0]["snr_lamrange"] == [3590.0, 7535.0]
        assert specs[0]["snr_lam_frame"] == "OBS"
        assert specs[0]["zpoly_texpose"] == []
    finally:
        os.unlink(tmppath)

    # The alias must be indistinguishable from the long spelling
    assert (parse_take_spectrum_entry("TAKE_SPECTRUM", "TOBS(4.6) SNR(1179)")
            == parse_take_spectrum_entry("TAKE_SPECTRUM",
                                         "TOBS(4.6) SNR_ZPOLY(1179)"))


def test_config_parsing_host_rejects_snr():
    """HOST spectra allow only TEXPOSE_ZPOLY, not SNR (snlc_sim.c:5834)."""
    import pytest

    from jax_snlc_core.utils.error_handling import SNANAError

    def parse(value):
        (entry,) = prepare_spectrum_requests([parse_take_spectrum_entry("TAKE_SPECTRUM", value)])
        return entry

    for value in ("HOST SNR_ZPOLY(50,0)", "HOST SNR(1179)"):
        with pytest.raises(SNANAError, match="HOST"):
            parse(value)

    # The legal HOST form is unaffected
    entry = parse("HOST TEXPOSE_ZPOLY(1800,0)")
    assert entry["epoch_frame"] == "HOST"
    assert entry["opt_texpose"] == 1
    assert entry["zpoly_texpose"] == [1800.0, 0.0]


# =============================================================================
# Spectrum generation (genspec)
# =============================================================================


def test_solve_texpose(testdata):
    from jax_snlc_sim.spectrograph.genspec import (
        compute_aggregate_snr, solve_texpose_batch,
    )

    calib = read_spectrograph(str(testdata))
    genmag = jnp.full(calib.n_lam, 22.0)

    # nearpeak=True, target SNR=15 → solver chooses texpose ~ 1800-3600 s
    texpose_arr, snrc_arr = solve_texpose_batch(
        calib, genmag[None, :], jnp.array([15.0]),
        jnp.array([5000.0]), jnp.array([6000.0]), 600.0, True,
    )
    texpose = float(texpose_arr[0])
    snr_bins, _, _, flux = get_snr_from_calib(calib, min(texpose, 5400.0), 600.0, genmag)
    snr_agg = float(compute_aggregate_snr(snr_bins, calib.lam_avg, flux, 5000.0, 6000.0))
    assert abs(snr_agg / 15.0 - 1.0) < 0.05

    # An iterated solve reports its own converged SNR as SNR_COMPUTE
    # (snlc_sim.c:11968-11969), within the 3% stopping tolerance.
    _, snrc_iter = solve_texpose_batch(
        calib, genmag[None, :], jnp.array([25.0]),
        jnp.array([5000.0]), jnp.array([6000.0]), 600.0, True,
    )
    assert abs(float(snrc_iter[0]) / 25.0 - 1.0) < 0.05

    # nearpeak=False with low target also gives a positive texpose.
    texpose_low_arr, _ = solve_texpose_batch(
        calib, genmag[None, :], jnp.array([5.0]),
        jnp.array([5000.0]), jnp.array([6000.0]), 600.0, False,
    )
    assert float(texpose_low_arr[0]) > 0

    # Higher SNR target → longer texpose.
    n = 5
    snr_targets = jnp.array([5.0, 10.0, 15.0, 20.0, 25.0])
    lam_lo = jnp.full(n, 5000.0)
    lam_hi = jnp.full(n, 6000.0)
    texpose_multi, _ = solve_texpose_batch(
        calib, jnp.tile(genmag, (n, 1)), snr_targets, lam_lo, lam_hi, 600.0, True,
    )
    texpose_np = np.asarray(texpose_multi)
    assert np.all(np.diff(texpose_np) >= 0)

    # The SNR-mode solve weights by F_lambda (~1e-19), whose square is below
    # float32's normal range; the aggregate must not depend on the scale.
    rng = np.random.default_rng(1)
    snr_bins = jnp.asarray(rng.uniform(0.5, 30.0, 100).astype(np.float32))
    lam = jnp.linspace(8000.0, 9000.0, 100, dtype=jnp.float32)
    shape = rng.uniform(0.2, 1.0, 100)
    reference = shape.sum() / np.sqrt(np.sum((shape / np.asarray(snr_bins)) ** 2))
    for scale in (1e-15, 1e-19):
        flux = jnp.asarray((shape * scale).astype(np.float32))
        agg = compute_aggregate_snr(snr_bins, lam, flux, 8000.0, 9000.0)
        assert float(agg) == pytest.approx(reference, rel=1e-4)


def test_wavelength_smearing(testdata):
    from jax_snlc_sim.spectrograph.genspec import (
        apply_wavelength_smearing, build_smearing_matrix,
    )

    n_lam = 20
    lam_min = jnp.arange(5000, 5000 + n_lam * 10, 10, dtype=jnp.float32)
    lam_max = lam_min + 10.0
    lam_sigma = jnp.full(n_lam, 20.0)
    lam_avg = 0.5 * (lam_min + lam_max)
    lam_bin = lam_max - lam_min

    matrix = build_smearing_matrix(lam_min, lam_max, lam_avg, lam_bin, lam_sigma)
    assert matrix.shape == (n_lam, n_lam)

    flux = jnp.zeros(n_lam).at[n_lam // 2].set(1000.0)
    smeared = np.asarray(apply_wavelength_smearing(flux, matrix))
    assert np.sum(smeared) > 0
    assert smeared[n_lam // 2] < 1000.0
    assert smeared[n_lam // 2 - 1] > 0
    # SNANA keeps the raw Gaussian integral, so the tails past +/-NBIN2 are lost
    # rather than renormalised away (GENSPEC_LAMSMEAR, snlc_sim.c:12240).
    lost = 1000.0 - np.sum(smeared)
    assert 0.0 < lost < 5.0

    calib = read_spectrograph(str(testdata))
    matrix_real = build_smearing_matrix(
        calib.lam_min, calib.lam_max, calib.lam_avg, calib.lam_bin, calib.lam_sigma,
    )
    flux_real = jnp.ones(calib.n_lam) * 100.0
    smeared_real = apply_wavelength_smearing(flux_real, matrix_real)
    assert abs(float(jnp.sum(smeared_real)) - float(jnp.sum(flux_real))) < 1.0


def test_noise_application():
    """The engine's noise step draws the correlated template realisation at
    the near-peak epoch and reuses it elsewhere (GENSPEC_OBSFLUX_RANSMEAR,
    snlc_sim.c:12342)."""
    from jax_snlc_sim.spectrograph.genspec import _add_spectrum_noise

    n_lam = 10
    flux = jnp.full(n_lam, 100.0)
    flamerr = jnp.full(n_lam, 5.0)
    errfrac_t = jnp.full(n_lam, 0.3)
    rangauss = jnp.zeros(n_lam)

    noisy, updated_template = _add_spectrum_noise(
        flux, flamerr, errfrac_t, rangauss, True, jax.random.PRNGKey(42),
    )
    assert bool(jnp.any(noisy != flux))
    assert bool(jnp.any(updated_template != 0))

    saved = np.asarray(updated_template).copy()
    _, template2 = _add_spectrum_noise(
        flux, flamerr, errfrac_t, updated_template, False, jax.random.PRNGKey(99),
    )
    assert np.allclose(template2, saved)


def test_noise_draw_matches_flamerr_in_single_precision():
    """A sky-dominated red bin has FLAMERR ~1e-20 erg/s/cm^2/A, whose square
    is below float32's normal range; the drawn noise must still carry the
    quoted error, with and without a template fraction."""
    from jax_snlc_sim.spectrograph.genspec import _add_spectrum_noise

    n = 100_000
    for flamerr_value in (1.8e-20, 5e-23):
        flamerr = jnp.full(n, flamerr_value, dtype=jnp.float32)
        for errfrac_t in (0.0, 0.6):
            noisy, _ = _add_spectrum_noise(
                jnp.zeros(n, jnp.float32), flamerr, jnp.float32(errfrac_t),
                jnp.zeros(n, jnp.float32), True, jax.random.PRNGKey(7))
            assert noisy.dtype == jnp.float32
            pull = np.asarray(noisy, np.float64) / flamerr_value
            assert np.std(pull) == pytest.approx(1.0, abs=0.02)


def _make_mock_params(n_sne, z=0.2):
    return {
        "sim_redshift_cmb": jnp.full(n_sne, z),
        "sim_redshift_helio": jnp.full(n_sne, z),
        "sn_global_idx": jnp.arange(n_sne, dtype=jnp.int32),
        "av": jnp.zeros(n_sne),
        "rv": jnp.full(n_sne, 3.1),
        "mwebv": jnp.zeros(n_sne),
        "sim_mwebv": jnp.zeros(n_sne),
        "peakmjd_true": jnp.full(n_sne, 60000.0),
        "cid": jnp.arange(n_sne),
    }


class _MockModel:
    """Stands in for an SNModel: declares no per-event SED parameters,
    reading no parameters."""

    @staticmethod
    def sed_param_names():
        return ()

    def generate_sed(self, phase, params, static_data):
        return jnp.full_like(static_data["wavelength_grid"], 1e-17)


def _make_mock_static_data():
    grid = jnp.linspace(2000.0, 12000.0, 500)
    host_dust_law = load_dust_law("FITZ99_EXACT")
    host_dust_law.prepare_wavelength_grid(np.asarray(grid))
    return {"wavelength_grid": grid,
            "model_instance": _MockModel(),
            "host_dust_law": host_dust_law,
            "fluxcal_scale_mag_offset": 1.0}


# no spectrograph options and no host spectra
_PLAIN_SPECTRA = dict(spectro_options={}, host_spec=None, hostfrac=0.0,
                      hostsnfrac=0.0)


def _with_ab_zp_sedmodel(calib):
    """Fill zp_sedmodel from the analytic AB primary (flam = 3631e-23*c/lam^2),
    as builder._compute_zp_sedmodel would for an AB kcor."""
    hc = 6.62607015e-27 * 2.99792458e18
    c_angs = 2.99792458e18
    lam_avg = np.asarray(calib.lam_avg)
    lam_bin = np.asarray(calib.lam_bin)
    flam_ab = 3631.0e-23 * c_angs / lam_avg**2
    calib.zp_sedmodel = jnp.array(2.5 * np.log10(flam_ab * lam_avg * lam_bin / hc))
    return calib


def test_generate_all_spectra_batch(testdata):
    from jax_snlc_sim.spectrograph.genspec import generate_all_spectra

    calib = _with_ab_zp_sedmodel(read_spectrograph(str(testdata)))
    n_sne = 5
    configs = [
        {
            "field": "", "wgt": 1.0, "epoch_frame": "REST",
            "epoch_range": [-5.0, 5.0], "opt_texpose": 1,
            "zpoly_texpose": [1800.0], "zpoly_snr": [],
            "snr_lamrange": [0.0, 0.0], "snr_lam_frame": "REST",
        },
        {
            "field": "", "wgt": 1.0, "epoch_frame": "REST",
            "epoch_range": [10.0, 10.0], "opt_texpose": 2,
            "zpoly_texpose": [], "zpoly_snr": [15.0, 0.0],
            "snr_lamrange": [5000.0, 6000.0], "snr_lam_frame": "REST",
        },
    ]
    spectra_by_snid, _ = generate_all_spectra(
        calib, configs, _make_mock_params(n_sne), _make_mock_static_data(),
        600.0, (-20.0, 50.0), jax.random.PRNGKey(42), np.ones(n_sne, dtype=bool),
        **_PLAIN_SPECTRA)
    assert len(spectra_by_snid) == n_sne
    total = sum(len(v) for v in spectra_by_snid.values())
    assert total == n_sne * 2
    for specs in spectra_by_snid.values():
        for s in specs:
            assert s.texpose > 0
            assert len(s.flam) == int((~np.asarray(calib.islam_extend)).sum())


def test_batch_partial_passing(testdata):
    from jax_snlc_sim.spectrograph.genspec import generate_all_spectra

    calib = _with_ab_zp_sedmodel(read_spectrograph(str(testdata)))
    n_sne = 10
    configs = [
        {
            "field": "", "wgt": 1.0, "epoch_frame": "REST",
            "epoch_range": [0.0, 0.0], "opt_texpose": 1,
            "zpoly_texpose": [1800.0], "zpoly_snr": [],
            "snr_lamrange": [0.0, 0.0], "snr_lam_frame": "REST",
        },
    ]
    passing = np.array([True, False, True, False, True, False, True, False, True, False])
    spectra_by_snid, snrsum = generate_all_spectra(
        calib, configs, _make_mock_params(n_sne), _make_mock_static_data(),
        600.0, (-20.0, 50.0), jax.random.PRNGKey(42), passing,
        **_PLAIN_SPECTRA)
    assert len(spectra_by_snid) == 5
    assert np.all(np.asarray(snrsum)[~passing] == 0)


def test_snrsum_rest_v(testdata):
    from jax_snlc_sim.spectrograph.genspec import generate_all_spectra

    calib = _with_ab_zp_sedmodel(read_spectrograph(str(testdata)))
    n_sne = 3
    configs = [
        {
            "field": "", "wgt": 1.0, "epoch_frame": "REST",
            "epoch_range": [0.0, 0.0], "opt_texpose": 1,
            "zpoly_texpose": [1800.0], "zpoly_snr": [],
            "snr_lamrange": [0.0, 0.0], "snr_lam_frame": "REST",
        },
    ]
    spectra_by_snid, snrsum = generate_all_spectra(
        calib, configs, _make_mock_params(n_sne, z=0.1), _make_mock_static_data(),
        600.0, (-20.0, 50.0), jax.random.PRNGKey(42), np.ones(n_sne, dtype=bool),
        **_PLAIN_SPECTRA)
    snrsum_np = np.asarray(snrsum)
    assert np.all(snrsum_np[:n_sne] > 0)

    # The batched sum runs on the full bin grid; recomputing it from the ragged
    # spectra that were actually written is an independent route to the same
    # number, and pins the SNANA definition: TRUE GENFLAM over FLAMERR, bin
    # CENTRES inside [5000, 6000]*(1+z_cmb), FLAMERR > 0 only
    # (GENSPEC_FLAM snlc_sim.c:12537-12558, GENSPEC_SNR :11453-11496).
    z1 = 1.1
    for snid, specs in spectra_by_snid.items():
        total = 0.0
        for s in specs:
            centres = 0.5 * (np.asarray(s.lam_min) + np.asarray(s.lam_max))
            m = ((centres >= 5000.0 * z1) & (centres <= 6000.0 * z1)
                 & (s.flamerr > 0))
            total += float(np.sum((s.sim_flam[m] / s.flamerr[m]) ** 2))
        assert total > 0
        assert abs(np.sqrt(total) / snrsum_np[snid] - 1.0) < 1e-6


def test_spectrum_noise_mask_invariant(testdata):
    """Spectrum noise is keyed by global SN index (fold_in), so a SN's
    spectrum is identical whether generated alongside all SNe or a subset —
    the pre-selection spectra cache stays valid for the passing subset."""
    from jax_snlc_sim.spectrograph.genspec import generate_all_spectra

    calib = _with_ab_zp_sedmodel(read_spectrograph(str(testdata)))
    n_sne = 6
    configs = [
        {
            "field": "", "wgt": 1.0, "epoch_frame": "REST",
            "epoch_range": [0.0, 0.0], "opt_texpose": 1,
            "zpoly_texpose": [1800.0], "zpoly_snr": [],
            "snr_lamrange": [0.0, 0.0], "snr_lam_frame": "REST",
        },
    ]
    params = _make_mock_params(n_sne)
    params["sn_global_idx"] = jnp.arange(n_sne)
    key = jax.random.PRNGKey(7)
    full, _ = generate_all_spectra(
        calib, configs, params, _make_mock_static_data(),
        600.0, (-20.0, 50.0), key, np.ones(n_sne, dtype=bool),
        **_PLAIN_SPECTRA)
    mask = np.array([True, False, True, False, True, False])
    part, _ = generate_all_spectra(
        calib, configs, params, _make_mock_static_data(),
        600.0, (-20.0, 50.0), key, mask,
        **_PLAIN_SPECTRA)
    assert set(part) == {0, 2, 4}
    for sid, specs in part.items():
        for s_p, s_f in zip(specs, full[sid]):
            np.testing.assert_array_equal(np.asarray(s_p.flam),
                                          np.asarray(s_f.flam))
            np.testing.assert_array_equal(np.asarray(s_p.flamerr),
                                          np.asarray(s_f.flamerr))


def test_zp_sedmodel_matches_analytic_ab(testdata, lsst_kcor_path):
    """_compute_zp_sedmodel on an AB-primary kcor (zero primary mags)
    reproduces the analytic AB zero point to sub-bin-integration accuracy."""
    from jax_snlc_core.io.kcor_reader import KcorReader
    from jax_snlc_sim.builder import _compute_zp_sedmodel

    calib = read_spectrograph(str(testdata))
    kcor = KcorReader(str(lsst_kcor_path))
    zp = _compute_zp_sedmodel(calib.lam_min, calib.lam_max, kcor)
    analytic = np.asarray(_with_ab_zp_sedmodel(calib).zp_sedmodel)
    np.testing.assert_allclose(zp, analytic, atol=2e-3)


# =============================================================================
# SPEC FITS write/read
# =============================================================================


def test_spec_fits_write_read(tmp_path):
    from astropy.io import fits as afits

    from jax_snlc_sim.output.fits import write_spec_fits
    from jax_snlc_sim.spectrograph.genspec import SpectrumResult

    n_lam = 5
    spec = SpectrumResult(
        snid=42, mjd=60000.0, texpose=1800.0, instrument="TEST",
        lam_min=np.arange(5000, 5000 + n_lam * 100, 100, dtype=np.float32),
        lam_max=np.arange(5100, 5100 + n_lam * 100, 100, dtype=np.float32),
        flam=np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32),
        flamerr=np.array([0.1, 0.2, 0.3, 0.2, 0.1], dtype=np.float32),
        sim_flam=np.array([1.1, 2.1, 3.1, 2.1, 1.1], dtype=np.float32),
        snr_compute=15.0, lammin_snr=5000.0, lammax_snr=5500.0,
        trest=0.0, tobs=0.0, sim_genmag=np.full(n_lam, 20.0),
    )

    write_spec_fits(SimulationConfig(), {'VERSION': 'TEST_SPEC'}, [], "TEST_SPEC",
                    str(tmp_path), {42: [spec]}, np.asarray)
    spec_file = tmp_path / "TEST_SPEC_SPEC.FITS"
    assert spec_file.exists()
    with afits.open(spec_file) as hdul:
        assert len(hdul) == 3
        header = hdul["SPECTRO_HEADER"].data
        flux = hdul["SPECTRO_FLUX"].data
        assert len(header) == 1
        assert header["SNID"][0] == "42"   # the CID unpadded (snlc_sim.c:25393)
        assert header["Texpose"][0] == 1800.0
        # the spectrum's bins, then its end-of-event marker row
        assert len(flux) == n_lam + 1
        assert flux["FLAM"][n_lam] == -777.0
        assert abs(flux["FLAM"][0] - 1.0) < 0.01


def test_wrspec_prescale_skips_odd_cids(tmp_path):
    """WRSPEC_PRESCALE: N writes spectra only for SNe with CID % N == 0."""
    from astropy.io import fits as afits

    from jax_snlc_sim.output.fits import write_spec_fits
    from jax_snlc_sim.spectrograph.genspec import SpectrumResult

    def _spec(cid):
        return SpectrumResult(
            snid=cid, mjd=60000.0, texpose=1800.0, instrument="TEST",
            lam_min=np.array([5000.0], dtype=np.float32),
            lam_max=np.array([5100.0], dtype=np.float32),
            flam=np.array([1.0], dtype=np.float32),
            flamerr=np.array([0.1], dtype=np.float32),
            sim_flam=np.array([1.1], dtype=np.float32),
            snr_compute=15.0, lammin_snr=5000.0, lammax_snr=5100.0,
            trest=0.0, tobs=0.0, sim_genmag=np.array([20.0]),
        )

    spectra = {cid: [_spec(cid)] for cid in (1, 2, 3, 4, 5, 6)}
    write_spec_fits(SimulationConfig(output=Output(wrspec_prescale=2)),
                    {'VERSION': 'TEST_PS'}, [], "TEST_PS", str(tmp_path),
                    spectra, np.asarray)
    with afits.open(tmp_path / "TEST_PS_SPEC.FITS") as hdul:
        written = {int(s) for s in hdul["SPECTRO_HEADER"].data["SNID"]}
    assert written == {2, 4, 6}


def test_warp_spectrum_lampoly_scales_measured_flam_only():
    """WARP_SPECTRUM: LAMPOLY(...) multiplies the measured FLAM by a polynomial
    in the bin-centre wavelength; SIM_FLAM and FLAMERR are untouched
    (GENSPEC_FLAM, snlc_sim.c:12503)."""
    import numpy as np
    from jax_snlc_sim.spectrograph.genspec import _parse_spectro_options

    class _Calib:
        lam_avg = np.array([4000.0, 5000.0])
        lam_min = np.array([3900.0, 4900.0])
        lam_max = np.array([4100.0, 5100.0])
        lam_bin = np.array([200.0, 200.0])
        lam_sigma = np.array([1.0, 1.0])
        n_lam = 2
        islam_extend = None

    # a0 + a1*lam, evaluated at the bin centres
    opt = _parse_spectro_options({'warp_lampoly': [1.0, 1.0e-4]}, _Calib())
    assert np.allclose(opt['flam_warp'], [1.4, 1.5], atol=0)

    # absent key leaves the warp off entirely, so no multiply happens
    assert _parse_spectro_options({}, _Calib())['flam_warp'] is None


def test_warp_spectrum_rejects_non_lampoly():
    """LAMPOLY is the only argument SNANA accepts (snlc_sim.c:5653)."""
    from jax_snlc_sim.config_parser import _parse_warp_spectrum
    from jax_snlc_core.utils.error_handling import SNANAError
    assert _parse_warp_spectrum('LAMPOLY(1.0,2e-5)') == [1.0, 2e-5]
    for bad in ('LAMPOLY', 'WAVEPOLY(1,2)', 'LAMPOLY(1,2'):
        with pytest.raises(SNANAError, match="WARP_SPECTRUM"):
            _parse_warp_spectrum(bad)


def _write_kcor_spectrograph(path, instrument, columns=('ZP00', 'SQSKYSIG00')):
    primary = fits.PrimaryHDU()
    if instrument is not None:
        primary.header['SPECTROGRAPH_INSTRUMENT'] = instrument
    cols = [fits.Column(name='LAMBIN_MIN', format='D', array=[4000.0, 4010.0]),
            fits.Column(name='LAMBIN_MAX', format='D', array=[4010.0, 4020.0]),
            fits.Column(name='LAMRES', format='D', array=[1.0, 1.0])]
    cols += [fits.Column(name=c, format='D', array=[25.0, 25.0]) for c in columns]
    table = fits.BinTableHDU.from_columns(cols, name='SPECTROGRAPH')
    table.header['NBT'] = 1
    table.header['TEXPOSE00'] = 100.0
    fits.HDUList([primary, table]).writeto(path)
    from jax_snlc_core.io.kcor_reader import KcorReader
    return KcorReader(path).spectrograph


def test_kcor_spectrograph_needs_the_instrument_key_and_a_whole_table(tmp_path):
    """Only a missing SPECTROGRAPH_INSTRUMENT means no spectrograph; once it is
    there a broken table aborts (read_spectrograph_fits,
    sntools_spectrograph.c:982-985 and snfitsio_errorCheck after it)."""
    assert _write_kcor_spectrograph(tmp_path / "none.fits", None) is None
    calib = _write_kcor_spectrograph(tmp_path / "ok.fits", "TOY")
    assert calib.instrument_name == "TOY" and calib.n_texpose == 1
    with pytest.raises(SNANAError, match="SPECTROGRAPH table has no ZP00"):
        _write_kcor_spectrograph(
            tmp_path / "broken.fits", "TOY", columns=('SQSKYSIG00',))
