"""Three simulator keys that had no SNANA reference until now.

Each fixture is a replay: SNANA's own events are pushed back through jax, so
the comparison is per event rather than statistical. The two peak-MJD and
SNR fixtures switch the flux and zero-point smear off, which makes both codes
see the same noiseless photometry and the quantity under test deterministic.

- OPT_SETPKMJD 16 sets the reported PEAKMJD from the Fmax-clump estimator
  (get_obs_atFLUXMAX, sntools.c:1595). Its window and both SNR cuts are
  hard-wired in SNANA, so this is the only knob the estimator has.
- EPCUTWIN_SNRMIN filters the epochs gen_cutwin counts (snlc_sim.c:24234),
  which moves the NEPOCH bit of CUTMASK.
- HOSTLIB_SBRADIUS is the aperture the host surface brightness integrates
  over (GEN_SNHOST_GALMAG, sntools_host.c:9343), which sets
  HOSTGAL_SB_FLUXCAL.
"""

import numpy as np
from _replay_helpers import select_replay
from astropy.io import fits
from replay_snana_dump import build_override_parameters

from jax_snlc_core.snana_files import load_snana_dump, resolve_fits_path
from jax_snlc_sim import run as sim_run


def _replay(fx):
    """Push SNANA's own written events back through jax."""
    with fits.open(resolve_fits_path(fx.head_fits)) as hdul:
        head = {c: np.asarray(hdul[1].data[c]) for c in hdul[1].data.names}
    snids = [int(s) for s in head['SNID']]
    dump = load_snana_dump(str(fx.dump), dedup_cid=True)
    dump_cols, head_cols = select_replay(dump.columns, dict(head), snids)

    config = dict(n_sne=len(dump_cols['CID']))
    sim = sim_run.prepare(str(fx.input_file), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    override = build_override_parameters(dump_cols, simulation, resources,
                                         head_columns=head_cols)
    results, selection, _ = sim_run.run_simulation(sim, override_parameters=override)
    return simulation, resources, results, selection, dump_cols, head_cols


def test_fmax_clump_peakmjd_matches_snana_per_event(snana_fixture):
    """With the same noiseless photometry the estimator is deterministic, so
    every event's reported PEAKMJD must be SNANA's."""
    fx = snana_fixture("setpkmjd_fmaxclump")
    simulation, _, results, _, _, head_cols = _replay(fx)
    assert simulation.rates.opt_setpkmjd == 16
    # the hard-wired parameters SNANA passes to get_obs_atFLUXMAX
    assert (simulation.rates.pkmjd_mjdwin,
            simulation.rates.pkmjd_snrcut,
            simulation.rates.pkmjd_snrcut_backup) == (60.0, 5.0, 3.0)

    jax_peak = np.asarray(results['parameters']['peakmjd'], dtype=np.float64)
    snana_peak = np.asarray(head_cols['PEAKMJD'], dtype=np.float64)
    assert len(snana_peak) > 250
    np.testing.assert_allclose(jax_peak, snana_peak, rtol=0, atol=1e-3)


def test_epcutwin_snrmin_matches_snanas_cutmask_per_event(snana_fixture):
    """The SNR floor filters the epochs gen_cutwin counts before it tests
    CUTWIN_NEPOCH, so the NEPOCH bit of CUTMASK moves with it. Both values
    appear in the fixture, so the comparison is not vacuous."""
    fx = snana_fixture("epcutwin_snrmin")
    simulation, _, _, selection, dump_cols, _ = _replay(fx)
    assert simulation.cuts.epcutwin_snrmin == (10.0, 1.0e9)

    snana_cutmask = np.asarray(dump_cols['CUTMASK'], dtype=np.float64).astype(int)
    jax_cutmask = np.asarray(selection['cutmask'], dtype=int)[:len(snana_cutmask)]
    assert len(np.unique(snana_cutmask)) > 1, "the cut must bite for some events"
    np.testing.assert_array_equal(jax_cutmask, snana_cutmask)


def test_hostlib_sbradius_matches_snanas_surface_brightness(snana_fixture):
    """HOSTGAL_SB_FLUXCAL integrates the host over an aperture of
    HOSTLIB_SBRADIUS, so the reported value scales as 1/radius^2: this
    fixture's 0.35 arcsec against SNANA's 1.2 default is a factor 12."""
    fx = snana_fixture("hostlib_sbradius")
    simulation, resources, results, _, _, head_cols = _replay(fx)
    assert simulation.host.sbradius == 0.35

    sb_jax = np.asarray(results['host_sb_fluxcal_stacked'], dtype=np.float64)
    n_compared = 0
    for i, name in enumerate(resources.filter_names):
        band = name.split('-')[-1]
        snana = np.asarray(head_cols[f'HOSTGAL_SB_FLUXCAL_{band}'],
                           dtype=np.float64)
        jax_sb = sb_jax[:, i]
        # A few per cent of SNe land on an aperture with no measurable
        # surface brightness in one code and not the other, most of them in
        # the faintest band; that is the replay's position match, not the
        # radius.
        both = (snana > 0.1) & (jax_sb > 0.1)
        assert both.mean() > 0.93, (band, float(both.mean()))
        np.testing.assert_allclose(jax_sb[both], snana[both], rtol=0.05)
        n_compared += int(both.sum())
    assert n_compared > 1500
