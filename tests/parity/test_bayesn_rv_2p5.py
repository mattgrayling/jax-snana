"""BayeSN with non-default host RV (2.5 vs nominal 3.1) — replay parity.

A different RV gates a distinct dust-law branch in the BayeSN model. Pin
that JAX reproduces SNANA's per-epoch SIM_MAGOBS in this configuration too.
"""

from pathlib import Path

import numpy as np
from _replay_helpers import slice_replay
from astropy.io import fits
from replay_snana_dump import build_override_parameters

from jax_snlc_core.snana_files import load_snana_dump
from jax_snlc_sim import run as sim_run
from jax_snlc_sim.output import save_snana_outputs


def _load_phot(head_path: Path, phot_path: Path):
    with fits.open(head_path) as h, fits.open(phot_path) as p:
        head, phot = h[1].data, p[1].data
        out = {}
        for row in head:
            snid = int(row["SNID"])
            pmin, pmax = int(row["PTROBS_MIN"]), int(row["PTROBS_MAX"])
            for i in range(pmin - 1, pmax):
                pr = phot[i]
                mjd = float(pr["MJD"])
                if mjd < 0:
                    continue
                band = str(pr["BAND"]).strip()
                out[(snid, round(mjd, 2), band)] = (
                    float(pr["SIM_MAGOBS"]),
                    float(pr["FLUXCAL"]),
                    float(pr["FLUXCALERR"]),
                )
    return out


def test_bayesn_rv_2p5_simmagobs_parity(snana_fixture, tmp_path):
    """RV mean 2.5 BayeSN replay matches SNANA SIM_MAGOBS sub-millimag."""
    fx = snana_fixture("bayesn_rv_2p5")

    dump = load_snana_dump(str(fx.dump), dedup_cid=True)
    with fits.open(fx.head_fits) as hdul:
        head_cols = {n: hdul[1].data[n] for n in hdul[1].data.names}

    # 300 of 1500 events: ~900 usable epochs, comfortably above the gate below
    dump_cols, head_cols = slice_replay(dump.columns, head_cols, 300)
    config = dict(n_sne=len(dump_cols['CID']))
    sim = sim_run.prepare(str(fx.input_file), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    override = build_override_parameters(dump_cols, simulation, resources, head_columns=head_cols)

    results, sel, spectra_cache = sim_run.run_simulation(sim, override_parameters=override)
    save_snana_outputs(sim, results, "BRV", sel, spectra_by_sn=spectra_cache, output_dir=str(tmp_path))

    snana_phot = _load_phot(fx.head_fits, fx.phot_fits)
    jax_dir = tmp_path / "BRV"
    jax_phot = _load_phot(
        next(jax_dir.glob("BRV*HEAD*")),
        next(jax_dir.glob("BRV*PHOT*")),
    )

    diffs = []
    common = set(snana_phot) & set(jax_phot)
    for key in common:
        sn_mag, sn_flux, sn_err = snana_phot[key]
        jx_mag, _, _ = jax_phot[key]
        if sn_err <= 0 or sn_flux / sn_err < 3.0:
            continue
        if sn_mag >= 50.0 or jx_mag >= 50.0:
            continue
        diffs.append(jx_mag - sn_mag)

    assert len(diffs) > 750
    diffs = np.asarray(diffs)
    abs_d = np.abs(diffs)
    median = float(np.median(abs_d))
    p95 = float(np.percentile(abs_d, 95))
    assert median < 0.005
    assert p95 < 0.05
