"""Observed redshift source flags and reference-frame behavior."""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_snlc_core.cosmology import zhelio_to_zcmb
from jax_snlc_sim.selection import _observed_redshift
from jax_snlc_sim.simulation import Cosmology, Rates, SimulationConfig


@pytest.mark.parametrize("sigma", [-1.0, 1.0])
@pytest.mark.parametrize("dipole", [0.0, 369.0])
def test_host_photoz_source(sigma, dipole):
    # Negative sigma assigns both frames directly (GEN_SNHOST_ZPHOT).
    # Sigma > .999 takes the normal frame conversion (gen_zsmear).
    simulation = SimulationConfig(
        rates=Rates(gensigma_redshift=sigma, genbias_redshift=.01),
        cosmology=Cosmology(vel_cmbapex=dipole))
    photoz = jnp.array([.4, -9.])
    params = {'host_ZPHOT_EVENT': photoz, 'host_ZPHOT_EVENT_ERR': jnp.array([.05, -9.]),
              '_zsmear_normal': jnp.array([2., 2.])}
    params.update(ra=jnp.array([15., 30.]), dec=jnp.array([20., 40.]))
    helio, cmb, _, error = _observed_redshift(
        simulation, params, jnp.array([.2, .3]),
        jnp.ones(2, dtype=bool), jnp.ones(2, dtype=bool), False)
    np.testing.assert_allclose(helio, photoz)
    np.testing.assert_allclose(error, [.05, -9.])
    expected = (zhelio_to_zcmb(photoz, params['ra'], params['dec'])
                if sigma > 0 and dipole >= 1. else photoz)
    np.testing.assert_allclose(cmb, jnp.where(photoz >= 0, expected, -9.))
