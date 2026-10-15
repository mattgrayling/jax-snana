"""Unit tests for the LENSING_PROBMAP_FILE reader and pre-inverted CDF table."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_snlc_sim.survey.lensing_probmap import read_lensing_probmap
from jax_snlc_sim.sampler import _sample_lensdmu


@pytest.fixture(scope="module")
def lensing_probmap_path(data_root):
    return data_root / "lensing" / "LENSING_PROBMAP_LogNormal+MICECATv1.DAT"


@pytest.mark.parametrize('last_z', [0.3, 0.4])
def test_redshift_interpolation(tmp_path, last_z):
    """Use the actual redshift spacing and scale the boundary distributions."""
    path = tmp_path / 'lensing.dat'
    path.write_text(''.join(
        f'{z} {offset - 0.1} 0\n{z} {offset + 0.1} 1\n'
        for z, offset in zip([0.1, 0.2, last_z], [0., 0.1, 0.3])))
    pm = read_lensing_probmap(str(path), dmuscale=1.)
    z = jnp.array([0.1, (0.2 + last_z)/2., 0.05, 2.*last_z])
    keys = jnp.tile(jax.random.PRNGKey(7), (len(z), 1))
    draws = np.asarray(_sample_lensdmu(pm.z_bins, jnp.asarray(pm.inv_cdf), z, keys))
    np.testing.assert_allclose(draws[1:],
                               [draws[0] + 0.2, draws[0]/2., 2.*(draws[0] + 0.3)],
                               atol=2e-8)


def test_inv_cdf_is_monotonic_per_redshift(lensing_probmap_path):
    """For each redshift bin, inv_cdf must be non-decreasing in quantile."""
    pm = read_lensing_probmap(str(lensing_probmap_path), dmuscale=1.0)
    deltas = np.diff(pm.inv_cdf, axis=1)
    n_violations = int((deltas < -1e-7).sum())
    assert n_violations == 0, (
        f"{n_violations} non-monotonic CDF entries — inversion broken"
    )


def test_dmuscale_scales_inv_cdf(lensing_probmap_path):
    """``dmuscale`` multiplies the inverted-CDF deltaMU values linearly."""
    pm1 = read_lensing_probmap(str(lensing_probmap_path), dmuscale=1.0)
    pm2 = read_lensing_probmap(str(lensing_probmap_path), dmuscale=0.5)
    assert np.allclose(pm2.inv_cdf, 0.5 * pm1.inv_cdf, atol=1e-7, rtol=1e-6)
