"""MINOBS_SEARCH replay parity.

SNANA fails pipeline, SPECID and zHOST together when an event has fewer than
MINOBS epochs (sntools_trigger.c:2033 returns before evaluating any of them).
The count is GENLC.NEPOCH, not the reported NOBS: SNANA tacks an artificial
PEAKMJD epoch onto every band it generates for, plus one per band setting
MJD_TEMPLATE (snlc_sim.c:12979-12989).

The fixture is built so the test can fail three separate ways:
  - GENRANGE_TREST is widened so a band collects several epochs. With the
    nominal window every band gets exactly one, so n_obs == n_bands and the
    band term is indistinguishable from doubling n_obs.
  - MJD_TEMPLATE is set on two bands, and some events' SIMLIB entries lack
    them, so a global template count differs from a per-event one.
  - The pipeline efficiency is all-pass, so the trigger draws no randoms.
    With a real efficiency curve, marginal-SNR epochs flip independently in
    each code and swamp the gate under test.
"""

import numpy as np
from astropy.io import fits
from replay_snana_dump import build_override_parameters

from jax_snlc_core.snana_files import load_snana_dump
from jax_snlc_sim import run as sim_run
from jax_snlc_sim.selection import _simlib_band_mask

# jax and SNANA disagree on an epoch when it sits within a fraction of a day of
# the Trest window edge: the zHEL/zCMB convention moves
# the edge ~0.06 d, and float32 PEAKMJD quantises it to 0.0039 d.
MAX_BOUNDARY_FRACTION = 0.03


def test_minobs_search_replay(snana_fixture):
    fx = snana_fixture("minobs_search_replay")

    dump = load_snana_dump(str(fx.dump), dedup_cid=True)
    snana_nepoch = np.asarray(dump.columns['NEPOCH']).astype(int)
    snana_pass = np.asarray(dump.columns['FLAG_ACCEPT']).astype(float) > 0.5
    with fits.open(fx.head_fits) as hdul:
        head_cols = {n: hdul[1].data[n] for n in hdul[1].data.names}

    # the selection is re-run below on every event's epochs
    config = dict(n_sne=dump.size, keep_rejected_events=True)
    sim = sim_run.prepare(str(fx.input_file), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    assert simulation.survey.searcheff_minobs == 20
    assert simulation.survey.mjd_template.get('r') == 61000.0

    override = build_override_parameters(dump.columns, simulation, resources,
                                         head_columns=head_cols)
    results, sel, _ = sim_run.run_simulation(sim, override_parameters=override)
    sched = results['observation_schedule']
    n = len(snana_nepoch)

    nobs = np.asarray(sched['n_obs_per_sn'])[:n]
    nbands = np.asarray(
        _simlib_band_mask(resources, simulation.cuts, sched)
    ).sum(axis=1)[:n]
    # Without this the band term could be any multiple of n_obs.
    assert (nobs != nbands).sum() > n // 2, "fixture no longer discriminates"

    # The engine's own count, against SNANA's NEPOCH column.
    nepoch = np.asarray(sel['nepoch'])[:n]
    mismatch = nepoch != snana_nepoch
    assert mismatch.mean() < MAX_BOUNDARY_FRACTION, (
        f"nepoch differs for {mismatch.sum()}/{n}; "
        f"offsets {np.unique(nepoch[mismatch] - snana_nepoch[mismatch])}")
    # A boundary epoch moves nepoch by one, or by two if it was the only epoch
    # its band had. Anything larger is a counting error, not a boundary.
    assert np.abs(nepoch[mismatch] - snana_nepoch[mismatch]).max() <= 2

    key = 'passing_mask'
    jax_pass = np.asarray(sel[key]).astype(bool)[:n]
    # The gate must actually reject, or this proves nothing.
    assert 0.05 < (~snana_pass).mean() < 0.60, (
        f"SNANA rejected {(~snana_pass).sum()}/{n}; fixture lost its teeth")
    disagree = (jax_pass != snana_pass).mean()
    assert disagree < MAX_BOUNDARY_FRACTION, (
        f"gate disagrees on {(jax_pass != snana_pass).sum()}/{n} events")
