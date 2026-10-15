"""True-SED spectrum generation for each supported model family."""

import numpy as np
import pytest

from jax_snlc_sim import run as sim_run
from jax_snlc_sim.random_streams import root_key


@pytest.mark.parametrize("fixture_name", [
    "salt3_k21_replay", "bayesn_w22", "non1ased_v19_replay", "simsed_91bg_replay",
], ids=["SALT3", "BayeSN", "NON1ASED", "SIMSED"])
def test_true_sed_spectra(snana_fixture, fixture_name):
    sim = sim_run.prepare(
        str(snana_fixture(fixture_name).input_file),
        n_sne=12, random_seed=7, batch_size=12,
        lambin_sed_true=20.0, lamrange_sed_true=(3000.0, 11500.0))
    results, selection, _ = sim_run.run_simulation(sim)
    mask = np.asarray(selection["passing_mask"])
    assert mask.any(), "no events passed selection"

    spectra, _ = sim_run.compute_spectra_for_mask(
        sim, results, mask, key=root_key(sim.config.run.random_seed))
    assert spectra, "no true-SED spectra generated"
    has_nonzero_flux = False
    for snid, per_sn in spectra.items():
        assert per_sn, f"event {snid} has no spectra"
        for spectrum in per_sn:
            lo = np.asarray(spectrum.lam_min)
            hi = np.asarray(spectrum.lam_max)
            flux = np.asarray(spectrum.flam)
            assert lo.ndim == 1 and lo.size > 1
            assert lo.shape == hi.shape == flux.shape
            assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
            assert np.all(lo > 0) and np.all(hi > lo)
            assert np.all(np.diff(lo) > 0) and np.all(np.diff(hi) > 0)
            assert np.all(np.isfinite(flux))
            has_nonzero_flux |= bool(np.any(flux != 0))
            assert np.isfinite(spectrum.mjd)
    # Individual template epochs can have zero flux.
    assert has_nonzero_flux, "all generated spectra are zero"
