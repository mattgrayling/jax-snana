"""Tests for legacy SIMLIB FLUXERR_COR support."""

import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jax_snlc_core.logging import SNANAError
from jax_snlc_core.noise import fluxerr_model_sqscales, noise_variances
from jax_snlc_sim.photometry import Epoch, _noise_model_single_observation
from jax_snlc_sim.simulation import Instrument
from jax_snlc_sim.simlib import _build_fluxerr_cor_map, read_simlib


def _write_simlib(extra_header_lines=""):
    content = f"""DOCUMENTATION:
    PURPOSE: test SIMLIB
DOCUMENTATION_END:

SURVEY: TEST   FILTERS: ugriz   TELESCOPE: TEST
PIXSIZE: 0.5
{extra_header_lines}
BEGIN LIBID
LIBID: 1
RA: 0.0 DEC: 0.0 MWEBV: 0.0 NOBS: 1 PIXSIZE: 0.5
S: 60000.0 0 r 27.0 0.01 100.0 1.0 1.0 0.5 1.0 1.0 0.0
END_LIBID: 1
"""
    fh = tempfile.NamedTemporaryFile(suffix='.SIMLIB', mode='w', delete=False)
    fh.write(content)
    fh.close()
    return Path(fh.name)


def test_parser_populates_table():
    p = _write_simlib(
        "FLUXERR_COR: ugriz -0.90 1.02 0.99 1.01 0.99 0.98\n"
        "FLUXERR_COR: ugriz -0.70 1.03 0.99 1.02 0.98 0.99\n"
        "FLUXERR_COR: ugriz -0.50 1.04 0.99 1.03 0.97 1.00\n"
    )
    try:
        r = read_simlib(str(p))
        assert r.fluxerr_cor.enabled
        np.testing.assert_allclose(
            r.fluxerr_cor.log10snr, [-0.9, -0.7, -0.5], atol=1e-5)
        # Filter 0 (u) at log10snr=-0.7 → 1.03
        assert r.fluxerr_cor.scale[0, 1] == pytest.approx(1.03)
    finally:
        p.unlink()


def test_parser_no_keys_defaults():
    p = _write_simlib()
    try:
        r = read_simlib(str(p))
        assert not r.fluxerr_cor.enabled
        assert np.all(r.fluxerr_add == 0.0)
    finally:
        p.unlink()


def test_build_rejects_out_of_range_scale():
    rows = [('ugriz', -0.5, np.array([1.0, 0.9, 1.0, 15.0, 1.0]))]
    with pytest.raises(SNANAError, match=r"outside \[0\.1, 10\.0\]"):
        _build_fluxerr_cor_map(rows, {'u': 0, 'g': 1, 'r': 2, 'i': 3, 'z': 4})


def test_build_merges_mixed_filter_sets():
    rows = [
        ('ugri', -0.9, np.array([1.02, 0.99, 1.01, 0.99])),
        ('ugri', -0.5, np.array([1.04, 0.99, 1.03, 0.97])),
        ('zY',   -0.3, np.array([0.98, 0.97])),
        ('zY',   -0.1, np.array([0.99, 0.98])),
    ]
    mapping = {'u': 0, 'g': 1, 'r': 2, 'i': 3, 'z': 4, 'Y': 5}
    m = _build_fluxerr_cor_map(rows, mapping)
    np.testing.assert_allclose(m.log10snr, [-0.9, -0.5, -0.3, -0.1], atol=1e-5)
    # u: native [-0.9, -0.5]; bins at -0.3 and -0.1 edge-clip to u's max value 1.04.
    np.testing.assert_allclose(m.scale[0], [1.02, 1.04, 1.04, 1.04], atol=1e-5)
    # z: native [-0.3, -0.1]; bins at -0.9 and -0.5 edge-clip to z's min value 0.98.
    np.testing.assert_allclose(m.scale[4], [0.98, 0.98, 0.98, 0.99], atol=1e-5)


def test_parser_default_filter_outside_table_is_one():
    p = _write_simlib(
        "FLUXERR_COR: ugri -0.5 1.10 1.10 1.10 1.10\n"
    )
    try:
        r = read_simlib(str(p))
        z_idx = r.filter_mapping['z']
        assert r.fluxerr_cor.scale[z_idx, 0] == pytest.approx(1.0)
    finally:
        p.unlink()


def _call_noise(snr_grid=None, scale_per_filter=None,
                active=False, filter_idx=0, fluxerr_add=None,
                noise_value=0.0, fluxcal=100.0, smearflag_flux=0, **overrides):
    if snr_grid is None:
        snr_grid = jnp.array([-2.0, 0.0, 2.0], dtype=jnp.float32)
    if scale_per_filter is None:
        scale_per_filter = jnp.ones((5, 3), dtype=jnp.float32)
    if fluxerr_add is None:
        fluxerr_add = jnp.zeros(5, dtype=jnp.float32)
    raw = dict(
        magnitude=20.0, fluxcal=fluxcal,
        noise_value=noise_value, noise_T=0.0, noise_F=0.0,
        zptavg=27.0, zpterr=0.0,
        skysig=0.5, nea=4.0,
        gain=1.0, readnoise=0.0,
        template_skysig=0.0, template_readnoise=0.0, template_zpt=0.0,
        fluxgal_pe=0.0, sqscale_true=1.0, sqscale_data=1.0,
        filter_idx=filter_idx,
    )
    settings = dict(
        smearflag_flux=smearflag_flux,
        fluxerr_cor_log10snr_grid=snr_grid,
        fluxerr_cor_scale_per_filter=scale_per_filter,
        fluxerr_add_per_filter=fluxerr_add,
        exposure_time_per_filter=jnp.ones(5, dtype=jnp.float32),
        fluxerr_cor_active=active,
    )
    for key, value in overrides.items():
        (raw if key in raw else settings)[key] = value
    variances = dict(zip(
        ('flux_pe', 'npe_over_fluxcal', 'sqsig_S', 'sqsig_SZ', 'sqsig_T',
         'sqsig_calc_data', 'snr_calc_szt', 'snr_calc_st'),
        noise_variances(
            raw['fluxcal'], raw['zptavg'], raw['zpterr'], raw['gain'], raw['nea'],
            raw['skysig'], raw['readnoise'], raw['template_skysig'],
            raw['template_readnoise'], raw['template_zpt'], raw['fluxgal_pe'], 0.0,
            smearflag_zeropt=0, zp_fluxcal=Instrument.zp_fluxcal)))
    epoch = Epoch(**variances, **{k: raw[k] for k in Epoch._fields if k not in variances})
    # [:4] drops the MAGMONITOR_SNR column, which these tests do not exercise
    return _noise_model_single_observation(epoch, **settings)[:4]


def test_noise_scales_by_fec():
    snr_grid = jnp.array([-2.0, 0.0, 2.0], dtype=jnp.float32)
    scale_per_filter = jnp.full((5, 3), 1.5, dtype=jnp.float32)
    _, sigma_off, _, _ = _call_noise(active=False)
    _, sigma_on, _, _ = _call_noise(
        snr_grid=snr_grid, scale_per_filter=scale_per_filter, active=True)
    assert float(sigma_on) == pytest.approx(float(sigma_off) * 1.5, rel=1e-4)


def test_fluxerrmodel_file_auto_sets_bit_32(tmp_path, lsst_kcor_path,
                                              lsst_simlib_path, data_root):
    inp = tmp_path / "test.input"
    inp.write_text(
        "GENVERSION: TEST\n"
        "NGENTOT_LC: 100\n"
        "FORMAT_MASK: 32\n"
        "GENTYPE: 1\n"
        "SNTYPES_Ia: 1 101\n"
        "CIDRAN_MIN: 0\n"
        "CIDRAN_MAX: 3000000\n"
        f"GENMODEL: {data_root / 'models' / 'SALT3.K21'}\n"
        "GENPEAK_SALT2x1: 0.0\n"
        "GENRANGE_SALT2x1: -2.0 2.0\n"
        "GENSIGMA_SALT2x1: 1.0 1.0\n"
        "GENPEAK_SALT2c: 0.0\n"
        "GENRANGE_SALT2c: -0.2 0.3\n"
        "GENSIGMA_SALT2c: 0.05 0.05\n"
        "GENPEAK_SALT2ALPHA: 0.14\n"
        "GENPEAK_SALT2BETA: 3.1\n"
        "OMEGA_MATTER: 0.315\n"
        "OMEGA_LAMBDA: 0.685\n"
        "w0_LAMBDA: -1.0\n"
        "wa_LAMBDA: 0\n"
        "GENFILTERS: ugrizY\n"
        f"KCOR_FILE: {lsst_kcor_path}\n"
        f"SIMLIB_FILE: {lsst_simlib_path}\n"
        f"FLUXERRMODEL_FILE: {tmp_path / 'FLUXERRMODEL.DAT'}\n"
        "GENRANGE_REDSHIFT: 0.05 0.30\n"
        "GENRANGE_MJD: 60500.0 61500.0\n"
        "GENRANGE_PEAKMJD: 60540.0 61460.0\n"
        "GENSIGMA_REDSHIFT: 0.001\n"
        "GENRANGE_TREST: -20.0 50.0\n"
        "SOLID_ANGLE: 0.015\n"
        "OPT_MWEBV: 0\n"
        "DNDZ: POWERLAW 2.5E-5 1.5\n"
        "RANSEED: 12345\n"
    )
    from jax_snlc_sim.inputs import parse_snana_input_file
    cfg = parse_snana_input_file(str(inp))
    assert cfg.survey.simlib_mskopt & 32


def test_cache_round_trip_preserves_map():
    p = _write_simlib(
        "FLUXERR_COR: ugriz -0.5 1.10 1.00 1.05 0.95 0.90\n"
        "FLUXERR_COR: ugriz -0.1 1.20 1.00 1.10 0.90 0.85\n"
    )
    try:
        fresh = read_simlib(str(p))
        cached = read_simlib(str(p))
        assert cached.fluxerr_cor.enabled
        np.testing.assert_array_equal(
            cached.fluxerr_cor.log10snr, fresh.fluxerr_cor.log10snr)
        np.testing.assert_array_equal(
            cached.fluxerr_cor.scale, fresh.fluxerr_cor.scale)
    finally:
        p.unlink()


def test_parser_populates_fluxerr_add():
    p = _write_simlib(
        "FLUXERR_ADD: ugriz 28.0 10.0 15.0 23.0 60.0\n"
        "FLUXERR_COR: ugriz -0.5 1.02 0.99 1.01 0.99 0.98\n"
    )
    try:
        r = read_simlib(str(p))
        np.testing.assert_array_equal(r.fluxerr_add, [28.0, 10.0, 15.0, 23.0, 60.0])
    finally:
        p.unlink()


def test_noise_fec_scales_true_side_when_noise_on():
    """With non-zero noise draws, FEC must amplify FLUXCAL too (true-side scale)."""
    snr_grid = jnp.array([-2.0, 0.0, 2.0], dtype=jnp.float32)
    scale_per_filter = jnp.full((5, 3), 2.0, dtype=jnp.float32)
    fluxcal_off, _, _, _ = _call_noise(active=False, noise_value=1.0,
                                       smearflag_flux=1)
    fluxcal_on, _, _, _ = _call_noise(
        snr_grid=snr_grid, scale_per_filter=scale_per_filter,
        active=True, noise_value=1.0, smearflag_flux=1)
    assert float(fluxcal_on) != pytest.approx(float(fluxcal_off), rel=1e-3)


def test_noise_fluxerr_add_increases_variance_before_scale():
    """FLUXERR_ADD>0 must increase σ even at scale=1.0 (adds to S, SZ, DATA)."""
    snr_grid = jnp.array([-2.0, 0.0, 2.0], dtype=jnp.float32)
    scale_per_filter = jnp.ones((5, 3), dtype=jnp.float32)
    add_off = jnp.zeros(5, dtype=jnp.float32)
    add_on = jnp.array([0.0, 0.0, 100.0, 0.0, 0.0], dtype=jnp.float32)
    _, sigma_no_add, _, _ = _call_noise(
        snr_grid=snr_grid, scale_per_filter=scale_per_filter,
        active=True, filter_idx=2, fluxerr_add=add_off)
    _, sigma_with_add, _, _ = _call_noise(
        snr_grid=snr_grid, scale_per_filter=scale_per_filter,
        active=True, filter_idx=2, fluxerr_add=add_on)
    assert float(sigma_with_add) > float(sigma_no_add)


def test_noise_explicit_bit_32_suppresses_scale():
    snr_grid = jnp.array([-2.0, 0.0, 2.0], dtype=jnp.float32)
    scale_per_filter = jnp.full((5, 3), 1.5, dtype=jnp.float32)
    _, sigma_off, _, _ = _call_noise(active=False)
    _, sigma_suppressed, _, _ = _call_noise(
        snr_grid=snr_grid, scale_per_filter=scale_per_filter, active=False)
    assert float(sigma_suppressed) == pytest.approx(float(sigma_off), rel=1e-6)


def test_noise_clamps_below_grid_to_edge():
    snr_grid = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    scale_per_filter = jnp.array(
        [[2.0, 1.5, 1.0]] * 5, dtype=jnp.float32)
    _, sigma_off, _, _ = _call_noise(active=False)
    _, sigma_on, _, _ = _call_noise(
        snr_grid=snr_grid, scale_per_filter=scale_per_filter, active=True)
    assert float(sigma_on) == pytest.approx(float(sigma_off) * 2.0, rel=1e-4)


def test_fluxerr_add_applies_after_fudgescale_unscaled():
    """The legacy additive term runs after FUDGESCALE_FLUXERR and is not
    scaled by it (snlc_sim.c:28105-28134): the variance it adds must be the
    same with and without the fudge scale. The pre-ordering-fix chain added
    it before the scale, quadrupling it under FUDGESCALE_FLUXERR=2."""
    f1sq = jnp.full(5, 4.0, dtype=jnp.float32)   # FUDGESCALE_FLUXERR = 2
    add = jnp.array([0.0, 0.0, 100.0, 0.0, 0.0], dtype=jnp.float32)
    s_base = float(_call_noise(active=True, filter_idx=2)[1])
    s_scale = float(_call_noise(active=True, filter_idx=2,
                                fudge_fluxerr_sq_per_filter=f1sq)[1])
    s_add = float(_call_noise(active=True, filter_idx=2, fluxerr_add=add)[1])
    s_both = float(_call_noise(active=True, filter_idx=2, fluxerr_add=add,
                               fudge_fluxerr_sq_per_filter=f1sq)[1])
    assert s_both**2 - s_scale**2 == pytest.approx(
        s_add**2 - s_base**2, rel=1e-4)


def test_fluxerr_add_gated_by_exposure_time():
    """The legacy additive term applies only when EXPOSURE_TIME_FILTER == 1
    for the band (snlc_sim.c:28114-28116)."""
    add = jnp.array([0.0, 0.0, 100.0, 0.0, 0.0], dtype=jnp.float32)
    xt = jnp.array([1.0, 1.0, 2.0, 1.0, 1.0], dtype=jnp.float32)
    s_base = float(_call_noise(active=True, filter_idx=2)[1])
    s_add = float(_call_noise(active=True, filter_idx=2, fluxerr_add=add)[1])
    s_gated = float(_call_noise(active=True, filter_idx=2, fluxerr_add=add,
                                exposure_time_per_filter=xt)[1])
    assert s_add > s_base
    assert s_gated == pytest.approx(s_base, rel=1e-6)


def test_fudge_magerr_reported_error_only():
    """FUDGE_MAGERR adds to SQSIG_TRUE[S] and the reported error but not to
    the SZ draw variance (snlc_sim.c:28005-28012), so the drawn flux is
    unchanged while the reported error grows."""
    magerr = jnp.array([0.0, 0.0, 0.1, 0.0, 0.0], dtype=jnp.float32)
    f_off, s_off, _, _ = _call_noise(noise_value=1.0, smearflag_flux=1,
                                     filter_idx=2)
    f_on, s_on, _, _ = _call_noise(noise_value=1.0, smearflag_flux=1,
                                   filter_idx=2,
                                   fudge_magerr_per_filter=magerr)
    assert float(f_on) == pytest.approx(float(f_off), rel=1e-6)
    assert float(s_on) > float(s_off)


def test_smearflag_flux_any_positive_activates_draws():
    """Any SMEARFLAG_FLUX > 0 applies the noise shifts (snlc_sim.c:28464-28472);
    zero leaves the true flux."""
    f0 = float(_call_noise(noise_value=1.0, smearflag_flux=0)[0])
    f1 = float(_call_noise(noise_value=1.0, smearflag_flux=1)[0])
    f2 = float(_call_noise(noise_value=1.0, smearflag_flux=2)[0])
    assert f0 == pytest.approx(100.0)
    assert f1 != pytest.approx(f0)
    assert f2 == pytest.approx(f1)


def test_smearflag_bit2_subtracts_source_plus_host():
    """SMEARFLAG_FLUX bit 2 removes source AND host shot noise from the
    reported error (snlc_sim.c:28518-28523)."""
    host_pe = 500.0
    s = lambda flag, g: float(_call_noise(smearflag_flux=flag,
                                          fluxgal_pe=g)[1])
    d_host = s(1, host_pe)**2 - s(3, host_pe)**2
    d_zero = s(1, 0.0)**2 - s(3, 0.0)**2
    host_var = s(1, host_pe)**2 - s(1, 0.0)**2
    assert d_host - d_zero == pytest.approx(host_var, rel=1e-4)


def test_template_noise_activates_on_skysig_alone():
    """Template variance enters whenever template_skysig > 0
    (snlc_sim.c:27807); TEMPLATE_ZPT validity is enforced at construction,
    not by the kernel's presence rule."""
    s_off = float(_call_noise()[1])
    s_on = float(_call_noise(template_skysig=2.0, template_zpt=26.0)[1])
    assert s_on > s_off


def test_sub_unity_true_scale_surfaces_as_nan():
    """A FLUXERRMODEL TRUE-side scale below one gives negative fudge
    variance; SNANA aborts (snlc_sim.c:28065-28079) and the kernel must
    surface NaN rather than flooring it to zero."""
    f, sigma, _, _ = _call_noise(smearflag_flux=1, sqscale_true=0.81)
    assert np.isnan(float(f))


def test_fluxerr_sqscale_exact_one_bumped_to_1p001():
    """A TRUE-side error scale of exactly 1.0 is bumped to 1.001
    (snlc_sim.c:28059-28062); the DATA side is not bumped. A masked-off map
    leaves scale 1, so the bump applies there too."""
    one = jnp.array(1.0)
    sqt, sqd = fluxerr_model_sqscales(
        one, one, one, one, one, jnp.array(10.0), jnp.array(100.0), True)
    assert float(sqt) == pytest.approx(1.001**2, rel=1e-6)
    assert float(sqd) == pytest.approx(1.0)
    sqt_masked, sqd_masked = fluxerr_model_sqscales(
        jnp.array(1.7), one, one, jnp.array(0.0), jnp.array(0.0),
        jnp.array(10.0), jnp.array(100.0), True)
    assert float(sqt_masked) == pytest.approx(1.001**2, rel=1e-6)
    assert float(sqd_masked) == pytest.approx(1.0)


def test_zero_point_jitter_scales_the_difference_flux():
    """SNANA's zero-point jitter scales the SN's search flux less its own
    flux in the template (snlc_sim.c:27842-27849): an SN as bright in the
    template as in the search image gets none, while without a template it
    gets the jitter of its whole flux, and the jitter variance goes as the
    square of the difference flux."""
    def jitter_variance(template_fluxcal):
        _, _, sqsig_S, sqsig_SZ, *_ = noise_variances(
            fluxcal=1000.0, zptavg=30.0, zpterr=0.02, gain=1.5, nea=20.0, skysig=10.0,
            readnoise=5.0, template_skysig=0.0, template_readnoise=0.0,
            template_zpt=30.0, fluxgal_pe=0.0, template_fluxcal=template_fluxcal,
            smearflag_zeropt=1, zp_fluxcal=27.5)
        return sqsig_SZ - sqsig_S
    assert jitter_variance(1000.0) == 0.0
    assert jitter_variance(0.0) > 0.0
    assert jitter_variance(500.0) == pytest.approx(0.25 * jitter_variance(0.0), rel=1e-12)
