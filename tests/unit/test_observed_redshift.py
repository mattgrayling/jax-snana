"""Observed redshift source flags and reference-frame behavior."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from jax_snlc_core.physics.cosmology import zhelio_to_zcmb
from jax_snlc_sim.selection import _observed_redshift


@pytest.mark.parametrize("sigma", [-1.0, 1.0])
@pytest.mark.parametrize("dipole", [0.0, 369.0])
def test_host_photoz_source(sigma, dipole):
    # Negative sigma assigns both frames directly (GEN_SNHOST_ZPHOT).
    # Sigma > .999 takes the normal frame conversion (gen_zsmear).
    simulation = SimpleNamespace(
        rates=SimpleNamespace(gensigma_redshift=sigma, genbias_redshift=.01),
        cosmology=SimpleNamespace(vel_cmbapex=dipole))
    photoz = jnp.array([.4, -9.])
    params = {'host_ZPHOT_EVENT': photoz, 'host_ZPHOT_EVENT_ERR': jnp.array([.05, -9.]),
              '_zsmear_normal': jnp.array([2., 2.])}
    results = {'parameters': {'ra': jnp.array([15., 30.]), 'dec': jnp.array([20., 40.])}}
    helio, cmb, _, error = _observed_redshift(
        simulation, results, params, jnp.array([.2, .3]),
        jnp.ones(2, dtype=bool), jnp.ones(2, dtype=bool), False)
    np.testing.assert_allclose(helio, photoz)
    np.testing.assert_allclose(error, [.05, -9.])
    expected = (zhelio_to_zcmb(photoz, results['parameters']['ra'], results['parameters']['dec'])
                if sigma > 0 and dipole >= 1. else photoz)
    np.testing.assert_allclose(cmb, jnp.where(photoz >= 0, expected, -9.))
