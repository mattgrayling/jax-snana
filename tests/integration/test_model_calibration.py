"""Model training calibration must reach simulated photometry."""

from jax_snlc_sim import run

from dataclasses import replace
import shutil

import numpy as np

from jax_snlc_sim.builder import build_simulation


def test_salt_training_shifts(data_root, tmp_path):
    config = dict(n_sne=1)
    _prepared = run.prepare('tests/fixtures/snana/salt3_k21_replay/input.input', **config)
    simulation = None if _prepared is None else _prepared.config
    baseline = _prepared
    path = tmp_path / 'SALT3.K21'
    shutil.copytree(data_root / 'models' / 'SALT3.K21', path)
    with (path / 'SALT3.INFO').open('a') as stream:
        stream.write('\nMAGSHIFT: LSST u 0.4\nWAVESHIFT: LSST u 100.0\n')
    base = baseline.photometry
    band = base.filter_names.index('LSST-u')
    others = np.arange(len(base.filter_names)) != band
    shifted = []
    for bits in (0, 4, 8, 12):
        changed = replace(simulation, model=replace(
            simulation.model, genmodel=str(path), genmodel_mskopt=bits,
            _model_input_values={**simulation.model._model_input_values,
                                'GENMODEL_MSKOPT': bits}))
        phot = build_simulation(changed, changed.numerical.wavelength_grid_points).photometry
        np.testing.assert_array_equal(phot.filter_ref_fluxes[others], base.filter_ref_fluxes[others])
        shifted.append(phot)
    both, wave, mag, neither = shifted
    np.testing.assert_array_equal(neither.filter_ref_fluxes, base.filter_ref_fluxes)
    np.testing.assert_array_equal(neither.common_trans_grid, base.common_trans_grid)
    np.testing.assert_allclose(mag.filter_ref_fluxes[band] / base.filter_ref_fluxes[band],
                               10**(.4 * .4), rtol=1e-12)
    np.testing.assert_allclose(both.filter_ref_fluxes[band] / wave.filter_ref_fluxes[band],
                               10**(.4 * .4), rtol=1e-12)
    assert not np.array_equal(wave.common_trans_grid[band], base.common_trans_grid[band])
    np.testing.assert_array_equal(mag.common_trans_grid, base.common_trans_grid)
