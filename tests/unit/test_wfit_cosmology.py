"""Accuracy tests for the shared CPL quadrature with optional radiation."""

import jax
import numpy as np
import pytest
from scipy.integrate import quad

from jax_snlc_core.cosmology import (cpl_comoving_integral, cpl_e2, distance_modulus,
                                     flat_cpl_comoving_integral, flat_cpl_e2)
from jax_snlc_core.utils import LIGHT_KM

_HUBBLE_DISTANCE_MPC = LIGHT_KM / 70.0


def _reference(z, omega_m, w0, wa, radiation=True):
    omega_r = 0.9e-4 if radiation else 0.0

    def integrand(value):
        zp1 = 1.0 + value
        dark_energy = (
            zp1 ** (3.0 * (1.0 + w0 + wa))
            * np.exp(-3.0 * wa * value / zp1)
        )
        return 1.0 / np.sqrt(
            (1.0 - omega_r) * (omega_m * zp1**3 + (1.0 - omega_m) * dark_energy)
            + omega_r * zp1**4
        )

    return quad(integrand, 0.0, z, epsabs=1.0e-12, epsrel=1.0e-12)[0]


def _curved_reference(z, omega_m, omega_lambda, w0, wa, radiation=True):
    omega_r = 0.9e-4 if radiation else 0.0
    omega_k = 1.0 - omega_m - omega_lambda

    def integrand(value):
        zp1 = 1.0 + value
        dark_energy = (
            zp1 ** (3.0 * (1.0 + w0 + wa))
            * np.exp(-3.0 * wa * value / zp1)
        )
        return 1.0 / np.sqrt(
            (1.0 - omega_r) * omega_m * zp1**3
            + omega_k * zp1**2
            + (1.0 - omega_r) * omega_lambda * dark_energy
            + omega_r * zp1**4
        )

    radial = quad(
        integrand, 0.0, z, epsabs=1.0e-12, epsrel=1.0e-12,
    )[0]
    if omega_k > 0.0:
        root = np.sqrt(omega_k)
        return np.sinh(root * radial) / root
    if omega_k < 0.0:
        root = np.sqrt(-omega_k)
        return np.sin(root * radial) / root
    return radial


@pytest.mark.parametrize(
    "omega_m,w0,wa,z",
    [
        (0.0, -2.0, -4.0, 2.33),
        (0.05, -1.3, 1.0, 2.33),
        (0.3, -1.0, 0.0, 1.0),
        (1.0, 0.0, 4.0, 2.33),
    ],
)
@pytest.mark.parametrize("radiation", [False, True])
def test_low_redshift_quadrature(omega_m, w0, wa, z, radiation):
    actual = float(flat_cpl_comoving_integral(
        z, omega_m, w0, wa, order=64, radiation=radiation))
    expected = _reference(z, omega_m, w0, wa, radiation)
    assert actual == pytest.approx(expected, rel=1.0e-10)


@pytest.mark.parametrize(
    "omega_m,w0,wa",
    [
        (1.0e-8, -2.0, -4.0),
        (0.05, -1.0, 4.0),
        (0.3, -1.0, 0.0),
        (1.0, 0.0, 0.0),
    ],
)
@pytest.mark.parametrize("radiation", [False, True])
def test_cmb_quadrature(omega_m, w0, wa, radiation):
    actual = float(flat_cpl_comoving_integral(
        1089.0, omega_m, w0, wa, order=128, radiation=radiation,
    ))
    expected = _reference(1089.0, omega_m, w0, wa, radiation)
    assert actual == pytest.approx(expected, rel=1.0e-7)


def test_general_cpl_reduces_to_flat_functions():
    redshift = np.asarray([0.05, 0.4, 1.1])
    omega_m = 0.27
    w0 = -0.9
    wa = 0.3
    omega_lambda = 1.0 - omega_m

    np.testing.assert_allclose(
        np.asarray(cpl_e2(redshift, omega_m, omega_lambda, w0, wa)),
        np.asarray(flat_cpl_e2(redshift, omega_m, w0, wa)),
        rtol=1.0e-13,
    )
    np.testing.assert_allclose(
        np.asarray(cpl_comoving_integral(
            redshift, omega_m, omega_lambda, w0, wa,
        )),
        np.asarray(flat_cpl_comoving_integral(
            redshift, omega_m, w0, wa,
        )),
        rtol=1.0e-13,
    )
    np.testing.assert_allclose(
        np.asarray(distance_modulus(redshift, redshift, omega_m, omega_lambda,
                                    _HUBBLE_DISTANCE_MPC, w0, wa)),
        np.asarray(distance_modulus(redshift, redshift, omega_m, 1.0 - omega_m,
                                    _HUBBLE_DISTANCE_MPC, w0, wa)),
        rtol=1.0e-13,
    )


@pytest.mark.parametrize(
    "omega_m,omega_lambda,w0,wa,z",
    [
        (0.2, 0.65, -0.9, 0.2, 1.1),
        (0.4, 0.75, -1.1, -0.3, 0.8),
    ],
)
@pytest.mark.parametrize("radiation", [False, True])
def test_curved_cpl_distance(
    omega_m, omega_lambda, w0, wa, z, radiation,
):
    transverse = _curved_reference(z, omega_m, omega_lambda, w0, wa, radiation)
    expected_mu = 5.0 * np.log10(
        (1.0 + z) * (299792.458 / 70.0) * transverse
    ) + 25.0

    assert float(distance_modulus(
        z, z, omega_m, omega_lambda, _HUBBLE_DISTANCE_MPC, w0, wa, radiation=radiation,
    )) == pytest.approx(expected_mu, rel=1.0e-10)


def test_curved_cpl_distance_has_correct_flat_curvature_gradient():
    redshift = 0.8
    omega_m = 0.3
    omega_lambda = 0.7

    derivative = float(jax.grad(
        lambda value: distance_modulus(
            redshift, redshift, omega_m, value, _HUBBLE_DISTANCE_MPC, -1.0, 0.0,
        )
    )(omega_lambda))
    step = 1.0e-5
    finite_difference = (
        float(distance_modulus(
            redshift, redshift, omega_m, omega_lambda + step, _HUBBLE_DISTANCE_MPC,
            -1.0, 0.0,
        ))
        - float(distance_modulus(
            redshift, redshift, omega_m, omega_lambda - step, _HUBBLE_DISTANCE_MPC,
            -1.0, 0.0,
        ))
    ) / (2.0 * step)

    assert derivative == pytest.approx(finite_difference, rel=1.0e-7)


def test_radiation_preserves_flat_normalization_and_does_not_clamp():
    for omega_m in [0.0, 0.315, 1.0]:
        assert float(flat_cpl_e2(0.0, omega_m)) == pytest.approx(1.0)
    # Keep curvature from the original densities, before their rescaling.
    assert float(cpl_e2(0.0, 0.2, 0.65)) == pytest.approx(1.0 + 0.15 * 0.9e-4)
    # An invalid expansion rate remains invalid, rather than being floored.
    assert float(flat_cpl_e2(1.0, -1.0)) < 0.0


@pytest.mark.parametrize("radiation", [False, True])
def test_cmb_prior_uses_snana_radiation_free_convention(radiation):
    from jax_wfit.config import WFITConfig
    from jax_wfit.priors import cmb_prior_chi2, prepare_priors

    config = WFITConfig(cmb=True, cmb_sim=True, radiation=radiation)
    truth = dict(omega_m=.315, w0=-1., wa=0.)
    expected_target = np.sqrt(.315) * _reference(1089., .315, -1., 0., radiation=False)
    prepared = prepare_priors(config, [truth], None)
    assert float(prepared["cmb_r"][0]) == pytest.approx(expected_target, rel=1e-11)
    expected_r = np.sqrt(.28) * _reference(1089., .28, -.9, .2, radiation=False)
    expected_chi2 = ((expected_r - expected_target) / config.sigma_Rcmb)**2
    assert float(cmb_prior_chi2(.28, -.9, .2, expected_target, config)) == pytest.approx(
        expected_chi2, rel=1e-10)


@pytest.mark.parametrize("radiation", [False, True])
def test_bao_prior_matches_snana_distance_conventions(radiation):
    from jax_wfit.config import WFITConfig
    from jax_wfit.priors import bao_prior_chi2

    config = WFITConfig(bao="DESI_2024", radiation=radiation)
    config.rd_prior = config.bao_rd
    z, omega_m, w0, wa = .7, .315, -.9, .2
    dm = LIGHT_KM / config.bao_h0 * _reference(z, omega_m, w0, wa, radiation)
    # SNANA DH_bao_prior omits radiation even when DM_bao_prior includes it.
    de = (1 + z)**(3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / (1 + z))
    dh = LIGHT_KM / config.bao_h0 / np.sqrt(omega_m * (1 + z)**3 + (1 - omega_m) * de)
    expected = np.array([dm, dh, np.cbrt(z * dm**2 * dh)]) / config.bao_rd
    priors = dict(bao_redshift=np.full(3, z), bao_observable=np.array([1, 2, 3]),
                  bao_mean=expected, bao_precision=np.eye(3))
    assert float(bao_prior_chi2(omega_m, w0, wa, config, priors)) == pytest.approx(
        0., abs=1e-20)
