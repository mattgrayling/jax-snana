"""The simulator's columns fit the same as the FITS files it wrote.

A pipeline hands the simulator's HEAD/PHOT columns straight to the fitter
rather than reading back what it just wrote. That is only safe if the two
routes give the same batch, so this compares them on one small simulation:
the same `_build_batch`, one route through `Table(columns)` and one through
astropy's FITS read.
"""

import numpy as np
import pytest

from jax_snlc_core.io.snana_table import SNANA_VERSION_STRING
from jax_snlc_fit.data_loader import (batch_from_columns,
                                      load_and_batch_snana_fits)
from jax_snlc_sim import run as sim_run
from jax_snlc_sim.output.tables import mw_option_keys

_BANDS = ["DES-g", "DES-r", "DES-i", "DES-z"]


def _sim_mw_colour_law(simulation):
    """The (SIMOPT_MWCOLORLAW, SIM_MWRV) the simulation's header carries."""
    keys = mw_option_keys(simulation)
    return keys['SIMOPT_MWCOLORLAW'], keys['SIM_MWRV']


@pytest.fixture(scope="module")
def simulated(tmp_path_factory, repo_root):
    """One small simulation, written and also kept in memory."""
    tmp = tmp_path_factory.mktemp("inmem")
    inp = repo_root / "tests/fixtures/snana/salt3_fit_select/input.input"
    cfg = dict(n_sne=60, random_seed=4321)
    sim = sim_run.prepare(str(inp), **cfg)
    simulation = None if sim is None else sim.config
    resources = sim
    results, selection, spectra = sim_run.run_simulation(sim)
    out = sim_run.save_snana_outputs(sim, results, "INMEM", selection, spectra_cache=spectra, output_dir=str(tmp))
    return tmp, simulation, resources, out


def test_columns_and_files_give_the_same_batch(simulated):
    tmp, simulation, resources, out = simulated
    assert out["header_columns"] and out["phot_columns"], (
        "save_snana_outputs did not hand back its columns")

    loader = dict(filter_map={b: i for i, b in enumerate(_BANDS)},
                  fit_bands=_BANDS, keep_obs_extras=True)
    version = tmp / "INMEM"
    from_file = load_and_batch_snana_fits(
        str(version / "INMEM_HEAD.FITS"), str(version / "INMEM_PHOT.FITS"),
        **loader)
    from_memory = batch_from_columns(
        out["header_columns"], out["phot_columns"], is_sim=True,
        survey=resources.survey_name, snana_version=SNANA_VERSION_STRING,
        zp_fluxcal=simulation.instrument.zp_fluxcal,
        sim_mw_colour_law=_sim_mw_colour_law(simulation), **loader)

    assert from_file.zp_fluxcal == simulation.instrument.zp_fluxcal, (
        "the sim's own zero point is not what its header carries")
    assert from_file.sim_mw_colour_law == from_memory.sim_mw_colour_law
    assert from_memory.n_sne == from_file.n_sne > 0
    assert (np.asarray(from_memory.snids).astype(str).tolist()
            == np.asarray(from_file.snids).astype(str).tolist())
    for name in ("redshifts", "mjd_offsets", "fluxcals", "fluxcalerrs",
                 "reference_mjds", "peakmjd_offsets"):
        a = getattr(from_memory, name)
        b = getattr(from_file, name)
        if a is None or b is None:
            assert a is b, f"{name}: one route has it, the other does not"
            continue
        a, b = np.asarray(a), np.asarray(b)
        assert a.dtype == b.dtype, f"{name}: {a.dtype} against {b.dtype}"
        np.testing.assert_allclose(a, b, rtol=0, atol=0, err_msg=name)
    for name in ("filter_indices", "observation_mask"):
        np.testing.assert_array_equal(
            np.asarray(getattr(from_memory, name)),
            np.asarray(getattr(from_file, name)), err_msg=name)


def test_sim_truth_survives_the_memory_route(simulated):
    """The truth columns are what a fit's SIM_ block is written from, and they
    are the ones a FITS round trip is most likely to reshape."""
    tmp, simulation, resources, out = simulated
    loader = dict(filter_map={b: i for i, b in enumerate(_BANDS)},
                  fit_bands=_BANDS)
    version = tmp / "INMEM"
    from_file = load_and_batch_snana_fits(
        str(version / "INMEM_HEAD.FITS"), str(version / "INMEM_PHOT.FITS"),
        **loader)
    from_memory = batch_from_columns(
        out["header_columns"], out["phot_columns"], is_sim=True,
        survey=resources.survey_name, snana_version=SNANA_VERSION_STRING,
        zp_fluxcal=simulation.instrument.zp_fluxcal,
        sim_mw_colour_law=_sim_mw_colour_law(simulation), **loader)

    assert (from_memory.sim_truth is None) == (from_file.sim_truth is None)
    assert from_file.sim_truth is not None, "the writer emitted no SIM_ truth columns"
    assert set(from_memory.sim_truth) == set(from_file.sim_truth)
    for col in from_file.sim_truth:
        a = np.asarray(from_memory.sim_truth[col])
        b = np.asarray(from_file.sim_truth[col])
        # byte order included: FITS is big-endian and the memory route is
        # native, so a route that skipped the read would differ silently
        assert a.dtype == b.dtype, f"{col}: {a.dtype} against {b.dtype}"
        np.testing.assert_allclose(a.astype(float), b.astype(float),
                                   rtol=0, atol=0, err_msg=col)
