"""Unit tests for the MW / host extinction-law implementations."""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_snlc_core.dust import CCM89DustLaw, ODonnell94DustLaw, _linear_coeffs_irr
from jax_snlc_core.interpolation import invkd_irr, spline_coefficients
from jax_snlc_core.logging import SNANAError


_OPT_WAVELENGTHS = jnp.array([3500.0, 4500.0, 5500.0, 6500.0, 8000.0])  # ugriz-like

# A(lambda) from SNANA's GALextinct (MWgaldust.c:321) for E(B-V) = 1 (AV = RV),
# at 3500, 4400, 5500, 6500 and 8000 A
_SNANA_GALEXTINCT = {
    (CCM89DustLaw, 3.1): [4.9340520180, 4.1060158865, 3.0964350097,
                          2.5651599487, 1.8522159363],
    (CCM89DustLaw, 5.0): [6.7061424584, 6.0055382106, 4.9958204214,
                          4.3032560366, 3.3371974428],
    (ODonnell94DustLaw, 3.1): [4.9726425116, 4.0988780747, 3.0958681104,
                               2.5451363410, 1.8845983473],
    (ODonnell94DustLaw, 5.0): [6.7004630783, 6.0004124169, 4.9955050045,
                               4.2861530007, 3.3480553690],
}


@pytest.mark.parametrize("law, rv", sorted(_SNANA_GALEXTINCT,
                                           key=lambda k: (k[0].__name__, k[1])))
def test_ccm89_and_odonnell94_match_snana(law, rv):
    """CCM89 (OPT_MWCOLORLAW 89) and O'Donnell94 (94) reproduce SNANA's
    GALextinct, compiled and run, to float32 precision."""
    wave = jnp.array([3500.0, 4400.0, 5500.0, 6500.0, 8000.0])
    np.testing.assert_allclose(
        np.asarray(law().extinction_curve(wave, rv=rv)),
        np.array(_SNANA_GALEXTINCT[(law, rv)]) / rv, rtol=1e-6)


def test_invkd_irr_shape_and_zero_boundary_rows():
    knots = np.array([0.0, 1.0, 2.5, 4.0, 5.0])
    M = invkd_irr(knots)
    assert M.shape == (5, 5)
    np.testing.assert_array_equal(M[0, :], 0.0)
    np.testing.assert_array_equal(M[-1, :], 0.0)


def test_spline_coefficients_partition_of_unity_inside_grid():
    """Within the knot range each row of J sums to 1 (Lagrange property)."""
    knots = np.array([0.0, 1.0, 2.5, 4.0, 5.0])
    invkd = invkd_irr(knots)
    x_int = np.array([0.5, 2.0, 3.5])
    J = spline_coefficients(jnp.asarray(x_int), jnp.asarray(knots), jnp.asarray(invkd), 0)
    np.testing.assert_allclose(J.sum(axis=1), 1.0, atol=1e-9)


def test_linear_coeffs_irr_partition_of_unity():
    knots = np.array([0.0, 1.0, 2.0, 3.0])
    x_int = np.array([0.25, 1.5, 2.75])
    J = _linear_coeffs_irr(x_int, knots)
    np.testing.assert_allclose(J.sum(axis=1), 1.0, atol=1e-9)


def test_linear_coeffs_irr_recovers_knot_values():
    """At a knot, J·y = y at that knot."""
    knots = np.array([0.0, 1.0, 2.0])
    x_int = np.array([1.0])  # exactly on a knot
    J = _linear_coeffs_irr(x_int, knots)
    np.testing.assert_allclose(J[0], [0.0, 1.0, 0.0], atol=1e-9)


def test_fitz99_approx_aborts_above_wavemax():
    """SNANA aborts F99-approx above 25000 A (GALextinct_Fitz99_approx,
    MWgaldust.c:655-662); no silent fallback to raw O'Donnell94."""
    from jax_snlc_core.dust import WAVEMAX_FITZ99, Fitzpatrick99ApproxDustLaw
    law = Fitzpatrick99ApproxDustLaw()
    ok = np.asarray(law.extinction_curve(
        jnp.array([4000.0, WAVEMAX_FITZ99]), rv=3.1))
    assert np.all(np.isfinite(ok))
    with pytest.raises(SNANAError, match="Fitzpatrick"):
        law.extinction_curve(jnp.array([4000.0, 26000.0]), rv=3.1)


def test_somm26_is_the_symbolic_regression_curve():
    """Sommovigo et al. (2026): one at the V band by construction, the
    three terms at two other wavelengths worked by hand from the formula
    (GALextinct_Somm26, MWgaldust.c:1243-1252), four PARLIST_MWCOLORLAW
    values required and B0, B3 not negative."""
    from jax_snlc_core.dust import load_dust_law
    law = load_dust_law("SOMM26")
    with pytest.raises(SNANAError, match="requires PARLIST_MWCOLORLAW"):
        law.extinction_curve(jnp.array([5542.0]), rv=3.1)
    with pytest.raises(SNANAError, match="four values"):
        law.set_params([1.0, 1.0, 0.5])
    with pytest.raises(SNANAError, match="positive B0 and B3"):
        law.set_params([1.0, 1.0, 0.5, -0.1])
    b0, b1, b2, b3 = 1.0, 1.0, 0.5, 1.5
    law.set_params([b0, b1, b2, b3])
    c0, c1, c2, c3, c4 = 0.4002, 285.6, 0.2092, 9.223, 1.016

    def k(wave):
        x = wave / 5542.0
        return (b0 * (np.exp(-c1 * (x - c0) ** 2) - np.exp(-c1 * (1 - c0) ** 2))
                + (b1 + b2 * (x - c2)) * (x - c2) * (np.exp(-c3 * x) - np.exp(-c3))
                + np.exp(b3 * (np.tanh(c4) - np.tanh(c4 * x))))

    curve = np.asarray(law.extinction_curve(jnp.array([2175.0, 5542.0, 8000.0]), rv=3.1))
    np.testing.assert_allclose(curve, [k(2175.0), 1.0, k(8000.0)], rtol=1e-6)
    # the RV and AV arguments do not enter the shape
    same = np.asarray(law.extinction_curve(jnp.array([2175.0, 8000.0]), rv=2.0, av=0.3))
    np.testing.assert_allclose(same, curve[[0, 2]], rtol=1e-12)


def test_per_event_av_reaches_the_law():
    """Every site passes the event's AV to extinction_curve. A law whose
    curve shape reads AV (Somm25) must return per-event rows for a column of
    AVs; a law that ignores AV must return the one shared curve, so the
    factorised arithmetic downstream is unchanged."""
    from jax_snlc_core.dust import load_dust_law
    av = jnp.array([0.05, 0.5, 2.0])[:, None]

    somm = load_dust_law("SOMM25")
    curves = np.asarray(somm.extinction_curve(_OPT_WAVELENGTHS, 3.1, av=av))
    assert curves.shape == (3, len(_OPT_WAVELENGTHS))
    # the rows differ in shape, not just scale: A(lam)/A(V) is AV-dependent
    assert not np.allclose(curves[0], curves[1])

    ccm = load_dust_law("CCM89")
    shared = np.asarray(ccm.extinction_curve(_OPT_WAVELENGTHS, 3.1, av=av))
    assert shared.shape == (len(_OPT_WAVELENGTHS),)


def test_simulator_validates_gordon03_rv_before_compilation():
    import jax
    from jax_snlc_sim.simulation import MWDust

    with pytest.raises(SNANAError, match='only valid for RV=2.74'):
        MWDust(dust_law='GORD03', rv=3.1).build_law()

    law = MWDust(dust_law='GORD03', rv=2.74).build_law()
    law.prepare_wavelength_grid(np.asarray(_OPT_WAVELENGTHS))
    evaluate = jax.jit(lambda law, rv: law.extinction_curve(_OPT_WAVELENGTHS, rv))
    result = evaluate(law, 2.74)
    np.testing.assert_allclose(result, law.extinction_curve(_OPT_WAVELENGTHS, 2.74),
                               rtol=1e-6)
