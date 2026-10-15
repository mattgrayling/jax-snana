"""MAGMONITOR_SNR per-epoch parity: SIM_SNRMAG<nn> against SNANA.

The monitor SNR is what a fixed-magnitude source would reach under each epoch's
observing conditions: SNANA swaps the SN's own shot noise for the monitor's
(snlc_sim.c:27887) and then rescales by whatever factor the reported error was
fudged by (:28193). Because it reuses the epoch's real sky, CCD, template and
zero-point terms, it is a direct check that jax's noise bookkeeping matches
SNANA component by component, independent of the SN flux itself.

The fixture carries no error fudges, so the rescale factor is 1 here; the
fudged path is exercised by the live test that appends FUDGESCALE_FLUXERR.
"""

import numpy as np
from astropy.io import fits

from jax_snlc_core.io.snana_dump_reader import load_snana_dump
from replay_snana_dump import build_override_parameters
from jax_snlc_sim import run as sim_run

_COL = 'SIM_SNRMAG22'
# Measured agreement is 6.6e-08 median / 8.4e-07 worst -- float32 round-off on
# a quantity built from float32 SIMLIB columns. The bound sits an order of
# magnitude above that, still far below any real formula error.
_REL_TOL = 1.0e-5


def _monitor_by_epoch(head_path, phot_path):
    """{(snid, band, rank-of-MJD-in-band): SIM_SNRMAG22}.

    Rank-keyed for the reason given in _replay_helpers: jax carries MJD through
    float32, so absolute MJDs cannot be joined on directly.
    """
    with fits.open(head_path) as h:
        hd = h[1].data
        snid = np.asarray(hd['SNID']).astype(int)
        lo = np.asarray(hd['PTROBS_MIN']).astype(int) - 1
        hi = np.asarray(hd['PTROBS_MAX']).astype(int)
    with fits.open(phot_path) as p:
        pd = p[1].data
        mjd = np.asarray(pd['MJD'])
        band = np.array([str(b).strip() for b in pd['BAND']])
        mon = np.asarray(pd[_COL])
    out = {}
    for k in range(len(snid)):
        sl = slice(lo[k], hi[k])
        m, b, v = mjd[sl], band[sl], mon[sl]
        good = m > 0
        for bb in set(b[good]):
            sel = good & (b == bb)
            for rank, idx in enumerate(np.where(sel)[0][np.argsort(m[sel])]):
                out[(int(snid[k]), bb, rank)] = float(v[idx])
    return out


def test_magmonitor_snr_parity(snana_fixture, tmp_path):
    fx = snana_fixture("magmonitor_snr")
    dump = load_snana_dump(str(fx.dump), dedup_cid=True)
    with fits.open(fx.head_fits) as hdul:
        head_cols = {n: np.asarray(hdul[1].data[n]) for n in hdul[1].data.names}

    config = dict(n_sne=dump.size)
    sim = sim_run.prepare(str(fx.input_file), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    override = build_override_parameters(dump.columns, simulation, resources,
                                         head_columns=head_cols)
    results, sel, spectra_cache = sim_run.run_simulation(sim, override_parameters=override)
    sim_run.save_snana_outputs(sim, results, "MM", sel, spectra_cache=spectra_cache, output_dir=str(tmp_path))

    snana = _monitor_by_epoch(fx.head_fits, fx.phot_fits)
    jax_dir = tmp_path / "MM"
    jax_out = _monitor_by_epoch(next(jax_dir.glob("MM*HEAD*")),
                                next(jax_dir.glob("MM*PHOT*")))

    common = sorted(set(snana) & set(jax_out))
    assert len(common) > 200, f"only {len(common)} epochs matched"

    a = np.array([snana[k] for k in common])
    b = np.array([jax_out[k] for k in common])
    # the column must carry real signal, or comparing zeros would pass
    assert (a > 1.0).sum() > 200, f"only {(a > 1.0).sum()} epochs with SNR > 1"
    assert a.max() / a.min() > 3.0, "monitor SNR spans too little dynamic range"

    rel = np.abs(b - a) / a
    assert rel.max() < _REL_TOL, (
        f"worst per-epoch SIM_SNRMAG22 disagreement {rel.max():.2e} "
        f"(median {np.median(rel):.2e}) over {len(common)} epochs")
