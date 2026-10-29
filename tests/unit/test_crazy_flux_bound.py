"""Unit tests for the SNANA crazy-flux bound (check_crazyFlux, snlc_sim.c:28628).

The jax bound is SNANA's ADU expression divided by 10^(0.4(ZPADU-ZP_FLUXCAL)),
which cancels ZPADU; these tests pin that algebra and the guards around it.
"""

import numpy as np

from jax_snlc_core.models import MODEL_REGISTRY, SimulatedModel
from jax_snlc_sim.photometry import crazy_flux_bound
from jax_snlc_sim.simulation import Instrument

ZP = Instrument.zp_fluxcal


def _snana_adu_bound(zpadu, z, xt=1.0, fluxerr_adu=0.0):
    """SNANA's own expression, in ADU (snlc_sim.c:28663-28669)."""
    return 1.0e5 * 10.0 ** (0.4 * (zpadu - 31.0)) * xt / (z * z) + 10.0 * fluxerr_adu


def test_matches_snana_after_the_adu_to_fluxcal_conversion():
    z, zpadu, fluxerr_cal = 0.3, 32.5, 12.0
    scale = 10.0 ** (0.4 * (zpadu - ZP))  # FLUXADU per FLUXCAL
    snana_cal = _snana_adu_bound(zpadu, z, fluxerr_adu=fluxerr_cal * scale) / scale
    np.testing.assert_allclose(float(crazy_flux_bound(fluxerr_cal, z, ZP)),
                               snana_cal, rtol=1e-6)


def test_bound_follows_the_fluxcal_zeropoint():
    """ZPADU cancels; the bound is in FLUXCAL, so it scales with ZP_FLUXCAL."""
    a = float(crazy_flux_bound(0.0, 0.3, ZP))
    b = float(crazy_flux_bound(0.0, 0.3, ZP + 2.5))
    np.testing.assert_allclose(b, 10.0 * a, rtol=1e-6)


def test_scales_as_inverse_redshift_squared():
    lo = float(crazy_flux_bound(0.0, 0.1, ZP))
    hi = float(crazy_flux_bound(0.0, 0.2, ZP))
    assert abs(lo / hi - 4.0) < 1e-4


def test_exposure_ratio_below_one_is_floored():
    """SNANA sets xt=1 when EXPOSURE_TIME_FILTER < 1 (snlc_sim.c:28658)."""
    assert float(crazy_flux_bound(0.0, 0.2, ZP, exposure_ratio=0.25)) == \
        float(crazy_flux_bound(0.0, 0.2, ZP, exposure_ratio=1.0))
    assert float(crazy_flux_bound(0.0, 0.2, ZP, exposure_ratio=4.0)) == \
        4.0 * float(crazy_flux_bound(0.0, 0.2, ZP))


def test_noise_term_is_ten_sigma():
    base = float(crazy_flux_bound(0.0, 0.2, ZP))
    assert abs(float(crazy_flux_bound(7.0, 0.2, ZP)) - (base + 70.0)) < 1e-3


def test_model_factor_multiplies_the_total_including_the_noise_term():
    """SNANA applies the model factor after adding 10*fluxerr (:28323-:28340)."""
    plain = float(crazy_flux_bound(7.0, 0.2, ZP))
    assert abs(float(crazy_flux_bound(7.0, 0.2, ZP, model_factor=10.0))
               - 10.0 * plain) < 1e-2


def test_simsed_and_redback_get_the_bright_model_factor():
    assert MODEL_REGISTRY['simsed'].crazyflux_factor == 10.0
    assert MODEL_REGISTRY['redback'].crazyflux_factor == 10.0
    assert (MODEL_REGISTRY['salt'].crazyflux_factor
            == SimulatedModel.crazyflux_factor == 1.0)


def test_vectorises_over_epochs():
    z = np.array([[0.1], [0.4]])
    err = np.zeros((2, 3))
    out = np.asarray(crazy_flux_bound(err, z, ZP))
    assert out.shape == (2, 3)
    assert abs(out[0, 0] / out[1, 0] - 16.0) < 1e-3
