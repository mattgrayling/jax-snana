"""BBC reference cosmology with explicit SNANA-compatible radiation control."""

import numpy as np
import pytest
from scipy.integrate import quad

from jax_salt2mu.config import BBCConfig
from jax_salt2mu.config_parser import parse_bbc_input
from jax_salt2mu.distances import mu_at
from jax_snlc_core.physics.constants import LIGHT_KM


@pytest.mark.parametrize("radiation", [False, True])
def test_reference_distances_match_quadrature(radiation):
    config = parse_bbc_input(None, overrides=[f"radiation={int(radiation)}"])
    assert config.radiation is radiation
    omega_r = 0.9e-4 if radiation else 0.0
    redshift = np.array([0.01, 0.5, 1.2])
    expected = []
    for z in redshift:
        integral = quad(lambda x: 1 / np.sqrt(
            (1 - omega_r) * (0.3 * (1 + x)**3 + 0.7)
            + omega_r * (1 + x)**4), 0, z, epsabs=1e-12)[0]
        expected.append(5 * np.log10((1 + z) * LIGHT_KM / config.h0 * integral) + 25)
    np.testing.assert_allclose(mu_at(redshift, config), expected, atol=1e-10, rtol=0)


def test_radiation_is_enabled_by_default():
    assert BBCConfig().radiation is True
    assert parse_bbc_input(None).radiation is True


@pytest.mark.parametrize("radiation", [False, True])
def test_reference_polynomial_overrides_cosmology(radiation):
    config = BBCConfig(radiation=radiation, muref_zpoly=(35.0, 2.0))
    np.testing.assert_allclose(mu_at(np.array([0.1, 0.5]), config), [35.2, 36.0])
