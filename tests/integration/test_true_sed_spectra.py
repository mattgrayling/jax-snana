"""True-SED spectrum generation for each supported model family."""

import copy

import jax
import numpy as np
import pytest

from jax_snlc_sim import run as sim_run
from jax_snlc_sim.output import save_snana_outputs
from jax_snlc_sim.output.tables import build_columnar_data
from jax_snlc_core.random import root_key
from jax_snlc_sim.spectra import compute_spectra_for_mask


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

    spectra, _ = compute_spectra_for_mask(
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


@pytest.mark.parametrize('fixture_name', ['host_spectra_replay', 'atmosphere_replay'])
def test_run_completes_science_before_writing(snana_fixture, tmp_path, fixture_name):
    """Spectra and DCR exist before saving, and writing leaves results unchanged."""
    sim = sim_run.prepare(str(snana_fixture(fixture_name).input_file),
                          n_sne=12, random_seed=7)
    results, selection, spectra = sim_run.run_simulation(sim)
    assert selection['n_passed'] > 0
    assert spectra
    assert all(s.synmag for event in spectra.values() for s in event)
    if fixture_name == 'atmosphere_replay':
        observations = build_columnar_data(sim, results, selection)['observations']
        assert np.isfinite(observations['airmass']).all()
        assert np.any(np.abs(observations['sim_dcr_dmag']) > 0)

    before = copy.deepcopy(results)
    synmags = [dict(s.synmag) for event in spectra.values() for s in event]
    save_snana_outputs(sim, results, 'COMPLETED', selection,
                       spectra_by_sn=spectra, output_dir=str(tmp_path))
    assert jax.tree_util.tree_structure(results) == jax.tree_util.tree_structure(before)
    for actual, expected in zip(jax.tree_util.tree_leaves(results),
                                jax.tree_util.tree_leaves(before), strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert [s.synmag for event in spectra.values() for s in event] == synmags


def test_output_subset_excludes_spectra(snana_fixture, tmp_path):
    """Writing fewer events excludes their spectra from the FITS output too."""
    from astropy.io import fits
    from jax_snlc_core.snana_files import resolve_fits_path

    sim = sim_run.prepare(str(snana_fixture('host_spectra_replay').input_file),
                          n_sne=12, random_seed=7)
    results, selection, spectra = sim_run.run_simulation(sim)
    passing = np.asarray(selection['passing_mask']).copy()
    cids = np.asarray(results['parameters']['cid'])
    removed = next(i for i in np.flatnonzero(passing) if int(cids[i]) in spectra)
    passing[removed] = False
    assert passing.any()
    subset = dict(selection, passing_mask=passing, n_passed=int(passing.sum()),
                  n_failed=selection['n_failed'] + 1)
    original_ids = set(spectra)
    save_snana_outputs(sim, results, 'SUBSET', subset, spectra,
                       output_dir=str(tmp_path))

    folder = tmp_path / 'SUBSET'
    with fits.open(resolve_fits_path(folder / 'SUBSET_HEAD.FITS')) as hdus:
        head_ids = set(np.asarray(hdus[1].data['SNID'], dtype=int))
    with fits.open(resolve_fits_path(folder / 'SUBSET_SPEC.FITS')) as hdus:
        spec_ids = set(np.asarray(hdus[1].data['SNID'], dtype=int))
    assert int(cids[removed]) not in head_ids
    assert spec_ids == original_ids & head_ids
    assert set(spectra) == original_ids
