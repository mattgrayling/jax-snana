"""Unit tests for GENMAG_SMEAR_MODELNAME scatter models (sntools_genSmear.c)."""

import os
import shutil

import jax.numpy as jnp
import numpy as np
import pytest

from jax_snlc_core.utils.error_handling import SNANAError
from jax_snlc_sim.genmag_smear import (
    _parse_coh_sigmas, build_correlated_band_smear, build_smear_model, oir_bands,
)

_WL = jnp.array([3000.0, 5000.0, 8000.0])


def test_bare_coh_uses_snana_hardwired_sigma():
    """'COH' alone is 0.13 mag (init_genSmear_COH)."""
    assert build_smear_model('COH').n_randoms == 1
    assert np.allclose(_parse_coh_sigmas('COH'), [0.13], atol=0)


@pytest.mark.parametrize("name,sigmas", [
    ('COH(0.11)', [0.11]),
    ('COH(0.55+0.11)', [0.55, 0.11]),
])
def test_coh_sigmas_from_parentheses(name, sigmas):
    assert build_smear_model(name).n_randoms == len(sigmas)
    assert np.allclose(_parse_coh_sigmas(name), sigmas, atol=0)


def test_coh_is_flat_in_wavelength_and_sums_linearly():
    """SNANA sums the two terms linearly, not in quadrature
    (get_genSmear_COH: 'note linear sum, not in quadrature')."""
    model = build_smear_model('COH(0.55+0.11)')
    dmag = np.asarray(model.delta_mag(jnp.array([1.0, 1.0]), _WL))
    assert np.allclose(dmag, dmag[0], atol=0)          # flat across wavelength
    assert np.isclose(dmag[0], 0.66, atol=0, rtol=1e-6)
    assert not np.isclose(dmag[0], np.hypot(0.55, 0.11))


@pytest.mark.parametrize("bad", ['COH(0.1,0.2)', 'COH(0.1:0.2)', 'COHX'])
def test_coh_rejects_invalid_forms(bad):
    """Only '+' separates the sigmas; anything else aborts in SNANA."""
    with pytest.raises(SNANAError, match="GENMAG_SMEAR_MODELNAME"):
        build_smear_model(bad)


def test_chotard11_is_the_long_spelling_of_c11():
    """SNANA maps both to init_genSmear_Chotard11(0) (snlc_sim.c:10041)."""
    randoms = jnp.linspace(-1.0, 1.0, 6)
    long_spelling = build_smear_model('Chotard11').delta_mag(randoms, _WL)
    short_spelling = build_smear_model('C11').delta_mag(randoms, _WL)
    assert np.allclose(np.asarray(long_spelling), np.asarray(short_spelling),
                       atol=0)
    # both resolve to far_uv 0, which the farUV variants do not
    assert not np.allclose(np.asarray(long_spelling),
                           np.asarray(build_smear_model('C11_1')
                                      .delta_mag(randoms, _WL)), atol=0)


# ----------------------------------------------------------------------------
# OIR. Expected values are SNANA's own init printout, captured by running
# snlc_sim.exe with the model.
# ----------------------------------------------------------------------------

_SNDATA = os.environ['SNDATA_ROOT']
_OIR = os.path.join(_SNDATA, 'models', 'OIR', 'OIR.J19')


def test_oir_cholesky_matches_snana():
    """SNANA prints its Cholesky at init to 4 dp (init_genSmear_OIR)."""
    snana = np.array([
        [0.1425, 0, 0, 0, 0, 0, 0],
        [0.1321, 0.0254, 0, 0, 0, 0, 0],
        [0.1396, 0.0243, 0.0434, 0, 0, 0, 0],
        [0.1302, -0.0093, 0.0136, 0.0526, 0, 0, 0],
        [0.1026, -0.0014, -0.0398, 0.0084, 0.0728, 0, 0],
        [0.1018, 0.0075, -0.0015, 0.0360, 0.0411, 0.0750, 0],
        [0.0952, -0.0320, 0.0177, -0.0184, 0.0488, 0.0232, 0.1019]])
    assert os.path.isdir(_OIR), f"OIR model missing from $SNDATA_ROOT: {_OIR}"
    lamcen, cholesky, _ = oir_bands(_OIR)
    assert len(lamcen) == 7
    assert np.max(np.abs(cholesky - snana)) < 5e-5


def test_coh_rejects_more_than_two_sigmas():
    """MAGSIGMA holds two; SNANA's splitString aborts above MXsplit=2."""
    with pytest.raises(SNANAError, match="two sigmas"):
        build_smear_model('COH(0.1+0.2+0.3)')


def test_oir_model_name_may_carry_a_path(tmp_path):
    """SNANA matches OIR. by substring and honours a path in the name."""
    src = _OIR
    dst = tmp_path / "OIR.J19"
    shutil.copytree(src, dst)
    model = build_smear_model(str(dst))
    assert model.n_randoms == build_smear_model('OIR.J19').n_randoms


def test_oir_adds_the_coherent_term():
    """SIGMACOH scales one extra random and shifts every wavelength equally
    (get_genSmear_OIR)."""
    lamcen, cholesky, _ = oir_bands(_OIR)
    n_bands = len(lamcen)
    assert build_smear_model('OIR.J19').n_randoms == n_bands + 1

    # The coherent random is the last one and shifts every wavelength equally.
    model = build_correlated_band_smear(lamcen, cholesky, sigmacoh=0.1)
    randoms = jnp.zeros(model.n_randoms).at[n_bands].set(2.0)
    out = np.asarray(model.delta_mag(randoms, jnp.array([5000.0, 12000.0])))
    assert np.allclose(out, 0.2, atol=1e-6)


def test_oir_is_flat_outside_its_node_range():
    """The edge value is held beyond the first/last band centre."""
    model = build_smear_model('OIR.J19')
    nodes, _, _ = oir_bands(_OIR)
    randoms = jnp.linspace(-1.0, 1.0, model.n_randoms)
    lo, hi = float(nodes[0]), float(nodes[-1])
    edge = model.delta_mag(randoms, jnp.array([lo, hi]))
    far = model.delta_mag(randoms, jnp.array([lo - 500.0, hi + 500.0]))
    assert np.allclose(np.asarray(edge), np.asarray(far), atol=1e-6)


@pytest.mark.parametrize("name,n_randoms", [
    ('C11', 6), ('Chotard11', 6), ('C11_1', 6), ('C11_2', 6),
    ('COH', 1), ('COH(0.55+0.11)', 2),
])
def test_n_randoms_matches_snana_per_model(name, n_randoms):
    """The sampler sizes its Gaussian draw from n_randoms (sampler.py:1361), so
    a wrong count shifts every later random and breaks replay far from here.

    C11 draws NBAND_C11 with no coherent term
    (init_genSmear_randoms(NBAND_C11,0), sntools_genSmear.c:1799), and COH one
    per sigma. OIR's NBAND_OIR + 1 (:2937) is covered by the OIR cases; G10's
    node count is model-dependent and covered by the parity fixtures.
    """
    assert build_smear_model(name).n_randoms == n_randoms


def test_coherent_term_is_structural_not_inferred_from_zero():
    """An OIR model declaring SIGMACOH 0.0 must still draw its extra random:
    SNANA sizes the draw from NBAND_OIR + 1 regardless of the value."""
    lamcen = np.array([4000.0, 5000.0, 6000.0])
    cholesky = np.eye(3) * 0.1
    assert build_correlated_band_smear(lamcen, cholesky).n_randoms == 3
    assert build_correlated_band_smear(
        lamcen, cholesky, sigmacoh=0.0).n_randoms == 4


def test_g10_node_grid_and_coherent_random(data_root):
    """G10 draws one random per node plus the coherent term, which is drawn
    LAST and shifts every wavelength equally (get_genSmear_G10)."""
    model = build_smear_model(
        'G10', salt_model_dir=str(data_root / "models" / "SALT3.K21"))
    assert model.n_randoms > 1

    # only the coherent random set: a flat offset, no node structure
    coherent_only = jnp.zeros(model.n_randoms).at[model.n_randoms - 1].set(1.0)
    flat = np.asarray(model.delta_mag(coherent_only, _WL))
    assert np.allclose(flat, flat[0], atol=0), "coherent term is not flat"
    assert flat[0] != 0.0, "SALT3.K21 declares a non-zero SIGMA_INT"

    # G10Fig8 zeroes the coherent term (fig8 branch), leaving only nodes
    fig8 = build_smear_model(
        'G10Fig8', salt_model_dir=str(data_root / "models" / "SALT3.K21"))
    assert fig8.n_randoms == model.n_randoms
    assert np.allclose(np.asarray(fig8.delta_mag(coherent_only, _WL)), 0.0,
                       atol=0)


def _write_salt_dir(tmp_path, info_lines, disp_rows):
    d = tmp_path / "SALT3.TEST"
    d.mkdir()
    (d / "SALT3.INFO").write_text("\n".join(info_lines) + "\n")
    (d / "salt3_color_dispersion.dat").write_text(
        "\n".join(f"{lam} {sig}" for lam, sig in disp_rows) + "\n")
    return str(d)


def test_g10_missing_sigma_int_aborts(tmp_path):
    """A SALT INFO file without SIGMA_INT/SIGMA_COH is fatal
    (sntools_genSmear.c:1367-1374)."""
    d = _write_salt_dir(tmp_path, ["COLOR_DISP_MAX: 1.0"],
                        [(2000.0, 0.1), (9000.0, 0.1)])
    with pytest.raises(SNANAError, match="SIGMA_INT"):
        build_smear_model('G10', salt_model_dir=d)


def test_g10_raw_dispersion_cap_defaults_to_five(tmp_path):
    """Without a COLOR_DISP_MAX key the raw-file cap is SNANA's local 5.0
    (sntools_genSmear.c:1082), so a 3.0 dispersion survives the file read
    and is then capped at the 2.0 node level (:1265-1267)."""
    d = _write_salt_dir(
        tmp_path, ["SIGMA_INT: 0.0"],
        [(2000.0, 3.0), (2400.0, 0.5), (9000.0, 0.5)])
    model = build_smear_model('G10', salt_model_dir=d)
    first_node_only = jnp.zeros(model.n_randoms).at[0].set(1.0)
    val = float(model.delta_mag(first_node_only, jnp.array([2000.1]))[0])
    # interp of (3.0 capped at file level 5 -> stays 3) then node cap 2-1e-8
    assert val == pytest.approx(2.0, abs=1e-6)


def test_g10_dispersion_must_cover_sed_red_edge(tmp_path):
    """MAXLAM < SED_LAMMAX is fatal (sntools_genSmear.c:1234-1245)."""
    d = _write_salt_dir(tmp_path, ["SIGMA_INT: 0.1"],
                        [(2000.0, 0.1), (8000.0, 0.1)])
    with pytest.raises(SNANAError, match="too small"):
        build_smear_model('G10', salt_model_dir=d, sed_lam_max=11000.0)


def test_g10_last_node_reachable_at_zmin(tmp_path):
    """The last node redshifted by (1+zmin) must reach the dispersion range
    (sntools_genSmear.c:1298-1314). The condition needs LAMCHECK below both
    MAXLAM and SED_LAMMAX, so it fires when the appended last node sits just
    under a MAXLAM that equals the SED red edge."""
    d = _write_salt_dir(tmp_path, ["SIGMA_INT: 0.1"],
                        [(2000.0, 0.1), (9000.0, 0.1)])
    with pytest.raises(SNANAError, match="zmin"):
        build_smear_model('G10', salt_model_dir=d, sed_lam_max=9000.0,
                          zmin=1.0e-9)
    # a zmin that carries the last node past MAXLAM is fine
    build_smear_model('G10', salt_model_dir=d, sed_lam_max=9000.0, zmin=0.2)


def test_phasecor_clips_before_correlation():
    import jax
    from jax_snlc_sim.sampler import _draw_phasecor_scatter
    from jax_snlc_sim.simulation import Instrument

    instrument = Instrument(phasecor_n_grid=3)
    keys = jax.random.split(jax.random.PRNGKey(42), 256)
    clip = (-0.7, 1.2)
    normals = _draw_phasecor_scatter(instrument, jnp.eye(3), keys, clip)
    assert np.all((normals >= clip[0]) & (normals <= clip[1]))
    covariance = jnp.array([[1., .8, .4], [.8, 1., .5], [.4, .5, 1.]])
    chol = jnp.linalg.cholesky(covariance)
    correlated = _draw_phasecor_scatter(instrument, chol, keys, clip)
    np.testing.assert_allclose(correlated, normals @ chol.T, atol=1e-12)


@pytest.mark.parametrize('double', [False, True])
def test_phasecor_coefficients_match_working_precision(double):
    import jax
    from jax_snlc_sim.builder import _init_phasecor
    from jax_snlc_sim.simulation import Model, SimulationConfig

    config = SimulationConfig(model=Model(genmag_smear_addphasecor=(.03, 2.)))
    phases = np.arange(config.instrument.phasecor_n_grid)
    covariance = .03**2 * np.exp(-np.abs(phases[:, None] - phases[None, :]) / 2.)
    original_coefficients = np.linalg.cholesky(covariance).astype(np.float32)

    with jax.enable_x64(double):
        coefficients = _init_phasecor(config)

    assert coefficients.dtype == (np.float64 if double else np.float32)
    np.testing.assert_array_equal(coefficients, original_coefficients)
