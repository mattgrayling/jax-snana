"""WEAKLENS_PROBMAP_FILE parity.

The magnification is a draw against the probability map, so jax's keyed PRNG
cannot reproduce SNANA's sequential stream event by event. Two things do not
depend on the stream. The reported error is a deterministic function of the
drawn offset, so SNANA's own SIM_LENSDMU pushed through jax reproduces its
LENSDMU_ERR exactly, per event. And the map itself is a distribution: jax's
sample must match SNANA's in the body, which pins the map read, the
WEAKLENS_DMUSCALE factor and the below-first-bin z/zMIN rule together.
"""

import numpy as np
import pytest
from astropy.io import fits

from jax_snlc_core.snana_files import load_snana_dump, resolve_fits_path
from jax_snlc_sim import run as sim_run
from jax_snlc_core.random import root_key
from jax_snlc_sim.lensing import apply_lensdmu_smear

# the map's first redshift bin; below it SNANA scales the bin-0 draw by
# z/zMIN (gen_lensDMU, sntools_weaklens.c:310-313)
_ZMIN_MAP = 0.405
_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def test_lensdmu_err_matches_snana_per_event(snana_fixture):
    """LENSDMU_ERR = |LENSDMU * WEAKLENS_DMUERR_FRAC| (snlc_sim.c:25423), so
    SNANA's own SIM_LENSDMU pushed through jax gives SNANA's LENSDMU_ERR."""
    fx = snana_fixture("weaklens_probmap")
    with fits.open(resolve_fits_path(fx.head_fits)) as hdul:
        sim_lensdmu = np.asarray(hdul[1].data['SIM_LENSDMU'], dtype=np.float64)
        lensdmu_err = np.asarray(hdul[1].data['LENSDMU_ERR'], dtype=np.float64)
    _prepared = sim_run.prepare(str(fx.input_file), **dict(n_sne=1))
    simulation = None if _prepared is None else _prepared.config
    assert simulation.lensing.dmuerr_frac > 0.0
    params = {'lensdmu': sim_lensdmu,
              'sn_global_idx': np.arange(len(sim_lensdmu))}
    apply_lensdmu_smear(simulation, params,
                        key=root_key(simulation.run.random_seed))
    np.testing.assert_allclose(np.asarray(params['lensdmu_err']), lensdmu_err,
                               rtol=0, atol=1e-6)


@pytest.fixture(scope="module")
def _lensdmu_samples(snana_fixture):
    """SNANA's drawn (z, LENSDMU) and jax's own over the same input."""
    fx = snana_fixture("weaklens_probmap")
    dump = load_snana_dump(str(fx.dump), dedup_cid=True).columns
    z_snana = np.asarray(dump['GENZ'], dtype=np.float64)
    dmu_snana = np.asarray(dump['LENSDMU'], dtype=np.float64)

    config = dict(n_sne=len(z_snana), random_seed=7)
    sim = sim_run.prepare(str(fx.input_file), **config)
    simulation = None if sim is None else sim.config
    results, _, _ = sim_run.run_simulation(sim)
    z_jax = np.asarray(results['parameters']['sim_redshift_cmb'],
                       dtype=np.float64)
    dmu_jax = np.asarray(results['parameters']['lensdmu'], dtype=np.float64)
    return z_snana, dmu_snana, z_jax, dmu_jax


def _assert_same_body(a, b, tol_frac):
    """Two independent samples of the same distribution, compared where the
    sample says something: the quartile spread and the 10-90 quantiles. The
    tails are a handful of events out of a few hundred and say nothing."""
    iqr_a = float(np.subtract(*np.quantile(a, [0.75, 0.25])))
    iqr_b = float(np.subtract(*np.quantile(b, [0.75, 0.25])))
    assert abs(iqr_b / iqr_a - 1.0) < tol_frac, (iqr_a, iqr_b)
    for q in _QUANTILES:
        qa, qb = float(np.quantile(a, q)), float(np.quantile(b, q))
        assert abs(qa - qb) < tol_frac * iqr_a, (q, qa, qb)


def test_in_range_lensing_distribution(_lensdmu_samples):
    """Inside the map's redshift coverage the draw is the inverse CDF of the
    interpolated probability row, scaled by WEAKLENS_DMUSCALE."""
    z_s, d_s, z_j, d_j = _lensdmu_samples
    a, b = d_s[z_s >= _ZMIN_MAP], d_j[z_j >= _ZMIN_MAP]
    assert len(a) > 1000 and len(b) > 1000
    _assert_same_body(a, b, 0.2)


def test_below_grid_lensing_scaling(_lensdmu_samples):
    """Below the map's first redshift SNANA draws from bin 0 and scales by
    z/zMIN (sntools_weaklens.c:310-313), so dividing that factor out leaves
    the bin-0 distribution in both."""
    z_s, d_s, z_j, d_j = _lensdmu_samples
    below_s, below_j = z_s < _ZMIN_MAP, z_j < _ZMIN_MAP
    a = d_s[below_s] / (z_s[below_s] / _ZMIN_MAP)
    b = d_j[below_j] / (z_j[below_j] / _ZMIN_MAP)
    assert len(a) > 250 and len(b) > 250
    _assert_same_body(a, b, 0.25)
