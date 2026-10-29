"""Multi-survey parity: DES SIMLIB + DES kcor + SALT3.K21 SIM_MAGOBS."""

from pathlib import Path

import numpy as np
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


def test_des_salt3_simmagobs_parity(snana_fixture, tmp_path):
    """SALT3.K21 on the DES survey config matches SNANA SIM_MAGOBS."""
    fx = snana_fixture("des_salt3_replay")

    dump = load_snana_dump(str(fx.dump), dedup_cid=True)
    with fits.open(fx.head_fits) as hdul:
        head_cols = {n: hdul[1].data[n] for n in hdul[1].data.names}

    config = dict(n_sne=dump.size)
    sim = sim_run.prepare(str(fx.input_file), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    override = build_override_parameters(dump.columns, simulation, resources, head_columns=head_cols)

    results, sel, spectra_cache = sim_run.run_simulation(sim, override_parameters=override)
    save_snana_outputs(sim, results, "DES", sel, spectra_by_sn=spectra_cache, output_dir=str(tmp_path))

    snana_phot = _load_phot(fx.head_fits, fx.phot_fits)
    jax_dir = tmp_path / "DES"
    jax_phot = _load_phot(
        next(jax_dir.glob("DES*HEAD*")),
        next(jax_dir.glob("DES*PHOT*")),
    )

    diffs = []
    common = set(snana_phot) & set(jax_phot)
    assert len(common) > 0
    for key in common:
        sn_mag, sn_flux, sn_err = snana_phot[key]
        jx_mag, _, _ = jax_phot[key]
        if sn_err <= 0 or sn_flux / sn_err < 5.0:
            continue
        if sn_mag >= 50.0 or jx_mag >= 50.0:
            continue
        diffs.append(jx_mag - sn_mag)

    assert len(diffs) > 1000, f"too few S/N≥5 epochs ({len(diffs)})"
    diffs = np.asarray(diffs)
    abs_d = np.abs(diffs)
    median = float(np.median(abs_d))
    p95 = float(np.percentile(abs_d, 95))
    assert median < 0.005, f"median |Δmag| = {median:.4f} > 0.005"
    assert p95 < 0.05, f"p95 |Δmag| = {p95:.4f} > 0.05"
