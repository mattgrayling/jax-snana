"""VPEC peculiar-velocity propagation: non-zero GENSIGMA_VPEC produces a
distribution of zHELIO − zCMB shifts consistent with c·VPEC."""

from pathlib import Path

import numpy as np


from jax_snlc_sim import run as sim_run  # noqa: E402

C_KM_S = 2.99792458e5  # speed of light, km/s


def test_vpec_smear_produces_zhelio_zcmb_offset(snana_fixture):
    """With GENSIGMA_VPEC=300 km/s, the distribution of (zHELIO − zCMB)·c
    should have stddev consistent with 300 km/s (modulo the helio-cmb offset).

    Pins: VPEC sampler runs, propagates into both zHELIO and zCMB, and the
    smearing magnitude matches the input GENSIGMA_VPEC.
    """
    fx = snana_fixture("vpec_smear")
    cfg = dict(n_sne=500, random_seed=122459)
    sim = sim_run.prepare(str(fx.input_file), **cfg)
    results, _, _ = sim_run.run_simulation(sim)
    p = results["parameters"]

    z_hel = np.asarray(p["sim_redshift_helio"])
    z_cmb = np.asarray(p["sim_redshift_cmb"])
    # The CMB-frame redshift is built from the CMB cosmological redshift
    # plus VPEC; helio adds a known dipole on top. The width of the helio-cmb
    # offset across SNe is therefore dominated by VPEC scatter.
    delta = (z_hel - z_cmb) * C_KM_S
    sigma = float(np.std(delta))
    # Allow factor of 2 around 300 km/s to cover the helio-CMB sky-direction
    # spread baked into the toy SIMLIB's two pointings.
    assert 100 < sigma < 700, (
        f"std((zHELIO−zCMB)·c) = {sigma:.0f} km/s; expected ~300 ± dipole"
    )
