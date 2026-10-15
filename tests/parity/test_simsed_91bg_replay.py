"""SIMSED 91bg-like SNe Ia SED parity (PLASTICC model, 3×3 grid subset)."""

from pathlib import Path

import numpy as np
from astropy.io import fits


from jax_snlc_core.io.snana_dump_reader import load_snana_dump  # noqa: E402
from replay_snana_dump import build_override_parameters  # noqa: E402
from jax_snlc_sim import run as sim_run  # noqa: E402
from _replay_helpers import slice_replay  # noqa: E402


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


def test_simsed_91bg_simmagobs_parity(snana_fixture, tmp_path):
    """SIMSED bilinear (stretch × color) replay matches SNANA SIM_MAGOBS."""
    fx = snana_fixture("simsed_91bg_replay")

    dump = load_snana_dump(str(fx.dump), dedup_cid=True)
    with fits.open(fx.head_fits) as hdul:
        head_cols = {n: hdul[1].data[n] for n in hdul[1].data.names}

    # 300 of 1500 events: ~900 usable epochs, comfortably above the gate below
    dump_cols, head_cols = slice_replay(dump.columns, head_cols, 300)
    config = dict(n_sne=len(dump_cols['CID']))
    sim = sim_run.prepare(str(fx.input_file), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    # the grid stops at the reddest band's red edge (LSST-Y, 10990 A), not
    # the template's 12000 A, so its 500 points sample the observed range
    grid = np.asarray(resources.wavelength_grid)
    assert grid[-1] == 10990.0 and len(grid) == 500
    override = build_override_parameters(dump_cols, simulation, resources,
                                         head_columns=head_cols)

    results, sel, spectra_cache = sim_run.run_simulation(sim, override_parameters=override)
    sim_run.save_snana_outputs(sim, results, "S91", sel, spectra_cache=spectra_cache, output_dir=str(tmp_path))

    snana_phot = _load_phot(fx.head_fits, fx.phot_fits)
    jax_dir = tmp_path / "S91"
    jax_phot = _load_phot(
        next(jax_dir.glob("S91*HEAD*")),
        next(jax_dir.glob("S91*PHOT*")),
    )

    diffs = []
    common = set(snana_phot) & set(jax_phot)
    assert len(common) > 0, "no overlapping (SNID, MJD, BAND) keys"
    for key in common:
        sn_mag, sn_flux, sn_err = snana_phot[key]
        jx_mag, _, _ = jax_phot[key]
        if sn_err <= 0 or sn_flux / sn_err < 5.0:
            continue
        if sn_mag >= 50.0 or jx_mag >= 50.0:
            continue
        diffs.append(jx_mag - sn_mag)

    assert len(diffs) > 500, f"too few S/N≥5 epochs ({len(diffs)})"
    diffs = np.asarray(diffs)
    abs_d = np.abs(diffs)
    median = float(np.median(abs_d))
    p95 = float(np.percentile(abs_d, 95))
    assert median < 0.005, f"median |Δmag| = {median:.4f} > 0.005"
    assert p95 < 0.05, f"p95 |Δmag| = {p95:.4f} > 0.05"
