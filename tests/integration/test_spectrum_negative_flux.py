"""The negative-flux option reaches every spectrum scheduling path."""

import numpy as np
import pytest

from jax_snlc_sim import run as sim_run
from jax_snlc_core.random import root_key
from jax_snlc_sim.spectra import compute_spectra_for_mask


@pytest.mark.parametrize('source', [
    'spectro_optmask_replay', 'simlib_spectro_replay', 'sedtrue_replay',
    'simlib_take_spectrum',
])
def test_spectrum_drivers_clip_negative_flux(source, fixture_root, data_root, tmp_path):
    fixture = 'spectro_optmask_replay' if source == 'simlib_take_spectrum' else source
    text = (fixture_root / 'snana' / fixture / 'input.input').read_text()
    if source == 'simlib_take_spectrum':
        text = '\n'.join(line for line in text.splitlines()
                         if not line.startswith(('SIMLIB_FILE:', 'TAKE_SPECTRUM:')))
        text += (f'\nSIMLIB_FILE: {data_root / "simlib/JAX_TEST_LSST_TAKESPEC.SIMLIB"}'
                 '\nUSE_SIMLIB_TAKE_SPECTRUM: 1\n')
    path = tmp_path / 'spectra.input'
    path.write_text(text)
    sim = sim_run.prepare(path, n_sne=2, random_seed=5, keep_rejected_events=True)
    results, _, _ = sim_run.run_simulation(sim)
    # A negative wavelength interval exercises clipping alongside positive flux.
    negative = (sim.model.wavelengths > 3500.) & (sim.model.wavelengths < 4500.)
    sim.model.m0 = abs(sim.model.m0) * np.where(negative, -1., 1.)
    sim.model.m1 = sim.model.m1 * 0
    sim.model.resample_to_grid(sim.model.wavelengths)
    fluxes = []
    for flags in (0, 16):
        sim.config.model.genmodel_mskopt = flags
        spectra, _ = compute_spectra_for_mask(
            sim, results, np.ones(2, dtype=bool), key=root_key(5))
        assert spectra, source
        fluxes.append(np.concatenate([s.sim_flam_full for events in spectra.values()
                                      for s in events]))
    assert np.isfinite(fluxes[0]).all()
    assert (fluxes[0] < 0).any()
    assert np.isfinite(fluxes[1]).all()
    assert (fluxes[1] >= 0).all()
    assert (fluxes[1] > 0).any()
    assert (fluxes[1] == 0).any()
