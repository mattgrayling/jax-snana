"""TAKE_SPECTRUM_PRESCALE parity.

The prescale accepts or rejects an event's spectra on a random draw, so jax's
keyed PRNG cannot reproduce SNANA's stream and per-SN parity is impossible. Two
things *are* checkable:

- the **decision rule** exactly, via `keep_spectrum` against a truth table
  (test_spectrum_prescale_rule.py in tests/integration);
- the **retained fraction per field**, distributionally, against SNANA's own
  output — which is what this file does.

The fixture prescales TOY_A by 4 and leaves TOY_B alone, so a correct
implementation thins one field and not the other. That asymmetry is the point:
a bug that dropped the field lookup entirely would thin both, and one that
ignored the prescale would thin neither.
"""

import glob

import numpy as np
from astropy.io import fits

from jax_snlc_core.io.fits_path import resolve_fits_path  # noqa: E402
from jax_snlc_sim.random_streams import root_key

PRESCALE = 4.0
N_LIBID_FIELDS = ('TOY_A', 'TOY_B')


def _snana_spectra_by_field(fx):
    """{field: (n_with_spectra, n_total)} from the bundled SNANA output."""
    with fits.open(resolve_fits_path(fx.head_fits)) as h:
        head = h[1].data
        snid = np.array([int(str(s).strip()) for s in head['SNID']])
        libid = np.array(head['SIM_LIBID'], dtype=int)
    spec_path = glob.glob(str(fx.root / "SPEC.FITS*"))[0]
    with fits.open(spec_path) as h:
        # the SPEC header table lists one row per spectrum, keyed by SNID
        tab = h[1].data
        col = 'SNID' if 'SNID' in tab.columns.names else tab.columns.names[0]
        spec_snids = set(int(str(s).strip()) for s in tab[col])
    # LIBID 0 -> TOY_A, LIBID 1 -> TOY_B in this SIMLIB
    out = {}
    for lib, field in enumerate(N_LIBID_FIELDS):
        sel = libid == lib
        out[field] = (int(np.isin(snid[sel], list(spec_snids)).sum()), int(sel.sum()))
    return out


def test_prescale_thins_only_the_named_field(snana_fixture):
    """SNANA keeps ~1/4 of TOY_A spectra and all of TOY_B."""
    fx = snana_fixture("spectrum_prescale_replay")
    by_field = _snana_spectra_by_field(fx)

    n_a, tot_a = by_field['TOY_A']
    n_b, tot_b = by_field['TOY_B']
    assert tot_a > 50 and tot_b > 50, f"fixture too small: {by_field}"

    # TOY_B is not in the prescale dict -> untouched
    assert n_b == tot_b, f"TOY_B should be untouched, got {n_b}/{tot_b}"

    # TOY_A thinned to ~1/PRESCALE; binomial sigma on tot_a at p=1/4
    frac = n_a / tot_a
    sigma = np.sqrt((1 / PRESCALE) * (1 - 1 / PRESCALE) / tot_a)
    assert abs(frac - 1 / PRESCALE) < 5 * sigma, (
        f"TOY_A retained {frac:.3f}, expected ~{1 / PRESCALE:.3f} "
        f"+/- {sigma:.3f} (n={tot_a})")


def test_jax_retained_fraction_matches_snana(snana_fixture, tmp_path):
    """jax thins the same field to the same fraction as SNANA (distributional:
    the accept/reject draw cannot be matched per-SN across the two RNGs)."""
    from jax_snlc_sim import run as sim_run

    fx = snana_fixture("spectrum_prescale_replay")
    by_field = _snana_spectra_by_field(fx)
    n_a, tot_a = by_field['TOY_A']
    snana_frac = n_a / tot_a

    n_sne = 300
    cfg = dict(
        n_sne=n_sne, random_seed=11)
    sim = sim_run.prepare(str(fx.input_file), **cfg)
    simulation = None if sim is None else sim.config
    resources = sim
    results, sel, _ = sim_run.run_simulation(sim)
    mask = np.asarray(sel['passing_mask'])
    spectra, _ = sim_run.compute_spectra_for_mask(
        sim, results, mask,
        key=root_key(simulation.run.random_seed))

    fields = np.array([resources.fields[int(i)].field_name
                       for i in np.asarray(results['parameters']['field_idx'])])
    cids = np.asarray(results['parameters']['cid'])
    with_spec = np.isin(cids, list(spectra.keys()))

    in_a = fields == 'TOY_A'
    in_b = fields == 'TOY_B'
    assert in_a.sum() > 50 and in_b.sum() > 50

    assert with_spec[in_b].all(), "TOY_B must be untouched by a TOY_A prescale"
    jax_frac = with_spec[in_a].sum() / in_a.sum()

    # both are binomial at p=1/PRESCALE; compare on the combined sigma
    sigma = np.sqrt((1 / PRESCALE) * (1 - 1 / PRESCALE)
                    * (1 / tot_a + 1 / int(in_a.sum())))
    assert abs(jax_frac - snana_frac) < 5 * sigma, (
        f"jax retained {jax_frac:.3f} vs SNANA {snana_frac:.3f} "
        f"(5 sigma = {5 * sigma:.3f})")


def test_prescale_draws_follow_the_sn_not_the_row(snana_fixture, tmp_path):
    """The prescale draw is keyed on each SN's global index, not its row number."""
    from jax_snlc_sim import run as sim_run
    from jax_snlc_sim.selection import keep_events

    fx = snana_fixture("spectrum_prescale_replay")
    cfg = dict(
        n_sne=120, random_seed=11)
    sim = sim_run.prepare(str(fx.input_file), **cfg)
    simulation = None if sim is None else sim.config
    resources = sim
    results, selection, _ = sim_run.run_simulation(sim)

    mask = np.ones(len(np.asarray(results['parameters']['cid'])), dtype=bool)
    key = root_key(simulation.run.random_seed)
    spectra_full, _ = sim_run.compute_spectra_for_mask(
        sim, results, mask, key=key)
    kept_full = set(spectra_full.keys())

    keep = np.arange(len(mask)) % 3 != 0
    thinned, _ = keep_events(results, selection, keep, written_only=False)
    spectra_thinned, _ = sim_run.compute_spectra_for_mask(
        sim, thinned,
        np.ones(int(keep.sum()), dtype=bool), key=key)

    fields = np.array([resources.fields[int(i)].field_name
                       for i in np.asarray(thinned['parameters']['field_idx'])])
    assert (fields == 'TOY_A').sum() > 20, "no prescaled events survived"

    surviving_cids = set(np.asarray(thinned['parameters']['cid']).tolist())
    assert set(spectra_thinned.keys()) == kept_full & surviving_cids


def test_prescale_keeps_entries_separate(snana_fixture):
    from dataclasses import replace
    from jax_snlc_sim import run

    fx = snana_fixture('spectrum_prescale_replay')
    sim = run.prepare(str(fx.input_file), **dict(n_sne=1))
    simulation = None if sim is None else sim.config
    resources = sim
    n = 256
    results = {'parameters': {'field_idx': np.zeros(n, dtype=int),
                             'sn_global_idx': np.arange(n),
                             'cid': np.arange(n)}}
    entry = simulation.spectra.take_spectrum[0]
    entries = [dict(entry, wgt=.5, prescale_key=0),
               dict(entry, wgt=.5, prescale_key=0),
               dict(entry, wgt=.5, prescale_key=2),
               dict(entry, epoch_frame='HOST', wgt=1., prescale_key=3)]
    simulation = replace(simulation, spectra=replace(
        simulation.spectra, take_spectrum=entries, take_spectrum_prescale={}))
    sim = run.prepare(simulation)
    keep = np.asarray(run.spectrum_prescales(
        sim, results, key=root_key(11)))
    # Expanded epochs share the input key's draw; different keys do not.
    np.testing.assert_array_equal(keep[:, 0], keep[:, 1])
    assert np.any(keep[:, 0] != keep[:, 2])
    assert keep[:, 3].all() and not keep[:, 0].all()
    simulation = replace(simulation, spectra=replace(
        simulation.spectra, take_spectrum_prescale={'TOY_A': 1e9}))
    sim = run.prepare(simulation)
    keep = np.asarray(run.spectrum_prescales(
        sim, results, key=root_key(11)))
    assert not keep[:, :3].any()
    assert keep[:, 3].all()
