"""Compilation reuse across independently prepared survey realisations."""

import logging

import jax
import numpy as np
import pytest

from jax_snlc_sim import run as sim_run


@pytest.mark.parametrize("fixture", [
    "searcheff_snii_basic", "salt3_fit_select", "bayesn_w22", "simsed_91bg_replay",
])
def test_simulator_reuses_compilation_between_realisations(fixture_root, caplog, fixture):
    source = fixture_root / "snana" / fixture / "input.input"
    settings = dict(batch_size=4, wavelength_grid_points=32,
                    keep_rejected_events=True)
    first = sim_run.prepare(source, n_sne=6, random_seed=41, **settings)
    first_result, _, _ = sim_run.run_simulation(first)
    second = sim_run.prepare(source, n_sne=5, random_seed=42, **settings)

    with jax.log_compiles(), caplog.at_level(
            logging.DEBUG, logger="jax._src.interpreters.pxla"):
        result, selection, _ = sim_run.run_simulation(second)
    assert not any("Compiling jit(simulate_batch)" in record.getMessage()
                   for record in caplog.records)
    assert len(result["parameters"]["cid"]) == 5
    assert not np.array_equal(first_result["parameters"]["sim_redshift_cmb"][:5],
                              result["parameters"]["sim_redshift_cmb"])

    # A fresh batch compilation must reproduce the cached execution.
    sim_run.simulate_batch.clear_cache()
    fresh, fresh_selection, _ = sim_run.run_simulation(second)
    for name in ("sim_redshift_cmb", "peakmjd_true", "sn_global_idx"):
        np.testing.assert_array_equal(result["parameters"][name],
                                      fresh["parameters"][name])
    np.testing.assert_array_equal(result["fluxcals_noisy"], fresh["fluxcals_noisy"])
    np.testing.assert_array_equal(selection["passing_mask"],
                                  fresh_selection["passing_mask"])


def test_zeropoint_error_override_survives_compilation(fixture_root, caplog):
    source = fixture_root / 'snana/salt3_fit_select/input.input'
    settings = dict(n_sne=2, batch_size=2, random_seed=42,
                    wavelength_grid_points=32, keep_rejected_events=True,
                    smearflag_zeropt=1)
    first = sim_run.prepare(source, fudge_zpterr={'g': .02}, **settings)
    first_result, _, _ = sim_run.run_simulation(first)
    second = sim_run.prepare(source, fudge_zpterr={'g': .04}, **settings)
    with jax.log_compiles(), caplog.at_level(
            logging.DEBUG, logger='jax._src.interpreters.pxla'):
        second_result, _, _ = sim_run.run_simulation(second)
    assert not any('Compiling jit(simulate_batch)' in record.getMessage()
                   for record in caplog.records)
    np.testing.assert_array_equal(first_result['fluxcals'], second_result['fluxcals'])
    assert not np.array_equal(first_result['fluxcalerrs'], second_result['fluxcalerrs'])


def test_numerical_cut_and_efficiency_settings_reuse_compilation(fixture_root, caplog):
    source = fixture_root / 'snana/salt3_fit_select/input.input'
    settings = dict(n_sne=2, batch_size=2, random_seed=42,
                    wavelength_grid_points=32, keep_rejected_events=True,
                    apply_cutwin_opt=1)
    first = sim_run.prepare(source, peakmag_range=(-30., 100.),
                            searcheff_spec_scale=.5, **settings)
    first_result, first_selection, _ = sim_run.run_simulation(first)
    second = sim_run.prepare(source, peakmag_range=(90., 100.),
                             searcheff_spec_scale=1., **settings)
    with jax.log_compiles(), caplog.at_level(
            logging.DEBUG, logger='jax._src.interpreters.pxla'):
        second_result, second_selection, _ = sim_run.run_simulation(second)
    assert not any('Compiling jit(simulate_batch)' in record.getMessage()
                   for record in caplog.records)
    np.testing.assert_array_equal(first_result['fluxcals'], second_result['fluxcals'])
    assert not np.array_equal(first_selection['generated']['cutwin_fail_masks']['PEAKMAG'],
                              second_selection['generated']['cutwin_fail_masks']['PEAKMAG'])
