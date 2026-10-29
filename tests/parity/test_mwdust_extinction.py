"""MW dust extinction parity: JAX SIM_MAGOBS matches SNANA on a dusty sightline.

The bundled fixture has MWEBV=0.05 in the SIMLIB and ``OPT_MWEBV=3`` (apply
SF11 MW dust to the model). Pinning per-(SNID, MJD, BAND) SIM_MAGOBS against
SNANA catches a regression of the 2026-03-14 filter-weight-normalisation bug,
which produced a ~0.03 mag systematic in MW-extincted bands.
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


def _simmagobs_parity(fx, tmp_path, min_epochs=600, min_snr=5.0):
    """Replay the fixture and return per-band SIM_MAGOBS differences."""
    dump = load_snana_dump(str(fx.dump), dedup_cid=True)
    with fits.open(fx.head_fits) as hdul:
        head_cols = {n: hdul[1].data[n] for n in hdul[1].data.names}
        # Pin that the fixture really exercises MW dust — MWEBV must be set.
        mwebv = np.asarray(head_cols["MWEBV"])
        assert np.median(mwebv) > 0.04, (
            f"fixture MWEBV median = {np.median(mwebv):.4f}; expected ≈0.05"
        )

    # 300 events; guard below is ~80% of the resulting yield so a silent
    # loss of matched epochs fails instead of shrinking the sample
    dump_cols, head_cols = slice_replay(dump.columns, head_cols, 300)
    config = dict(n_sne=len(dump_cols['CID']))
    sim = sim_run.prepare(str(fx.input_file), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    override = build_override_parameters(dump_cols, simulation, resources,
                                         head_columns=head_cols)

    results, sel, spectra_cache = sim_run.run_simulation(sim, override_parameters=override)
    save_snana_outputs(sim, results, "MW", sel, spectra_by_sn=spectra_cache, output_dir=str(tmp_path))

    snana_phot = _load_phot(fx.head_fits, fx.phot_fits)
    jax_dir = tmp_path / "MW"
    jax_phot = _load_phot(
        next(jax_dir.glob("MW*HEAD*")),
        next(jax_dir.glob("MW*PHOT*")),
    )

    per_band: dict[str, list[float]] = {}
    common = set(snana_phot) & set(jax_phot)
    for key in common:
        sn_mag, sn_flux, sn_err = snana_phot[key]
        jx_mag, _, _ = jax_phot[key]
        if sn_err <= 0 or sn_flux / sn_err < min_snr:
            continue
        if sn_mag >= 50.0 or jx_mag >= 50.0:
            continue
        per_band.setdefault(key[2], []).append(jx_mag - sn_mag)

    assert sum(len(v) for v in per_band.values()) > min_epochs, (
        f"too few S/N≥5 epochs ({sum(len(v) for v in per_band.values())})"
    )
    return per_band


def test_mwdust_simmagobs_parity(snana_fixture, tmp_path):
    """Per-band SIM_MAGOBS matches SNANA on the MWEBV=0.05 sightline.

    The 2026-03-14 filter-weight-normalisation bug produced a per-band
    *median* shift of ~0.03 mag in MW-extincted bands. With the fix
    applied, every band should agree within < 0.005 mag at the median."""
    per_band = _simmagobs_parity(snana_fixture("mwdust_parity"), tmp_path)
    for band, diffs in per_band.items():
        median = float(np.median(np.abs(np.asarray(diffs))))
        assert median < 0.005, (
            f"band {band}: median |Δmag| = {median:.4f} > 0.005 "
            f"(MW dust filter-weight bug regression?)"
        )


def test_mwdust_somm25_per_event_curves(snana_fixture, tmp_path):
    """SIM_MAGOBS matches SNANA with OPT_MWCOLORLAW=225 (Sommovigo25).

    Somm25's c1/c3/c4 depend on AV, so SNANA rebuilds the MWXT table with
    each event's AV (fill_TABLE_MWXT_SEDMODEL, genmag_SEDtools.c:2644-2709).
    At the fixture's MWEBV=0.05 the AV=1 unit-curve shortcut is 0.019 mag
    off in the u band, so this median catches a regression to it."""
    # SIM_MAGOBS is noiseless truth, so no S/N floor is needed; keeping it
    # would drop the shallow u band, where the AV dependence is largest.
    per_band = _simmagobs_parity(snana_fixture("mwdust_somm25"), tmp_path,
                                 min_epochs=450, min_snr=0.0)
    assert any(band.endswith("u") for band in per_band), (
        "no u-band epochs; the AV dependence goes untested")
    for band, diffs in per_band.items():
        median = float(np.median(np.abs(np.asarray(diffs))))
        assert median < 0.005, (
            f"band {band}: median |Δmag| = {median:.4f} > 0.005 "
            f"(is the MW curve evaluated with each event's AV?)"
        )


def test_mwdust_somm26_per_event_curves(snana_fixture, tmp_path):
    """SIM_MAGOBS matches SNANA with OPT_MWCOLORLAW=226 (Sommovigo26), the
    four-parameter curve read from PARLIST_MWCOLORLAW (GALextinct_Somm26,
    MWgaldust.c:1213-1255). Its shape does not depend on AV, so the shared
    unit curve is the whole test; the u band carries the bump term."""
    per_band = _simmagobs_parity(snana_fixture("mwdust_somm26"), tmp_path,
                                 min_epochs=450, min_snr=0.0)
    assert any(band.endswith("u") for band in per_band), "no u-band epochs"
    for band, diffs in per_band.items():
        median = float(np.median(np.abs(np.asarray(diffs))))
        assert median < 0.005, f"band {band}: median |dmag| = {median:.4f} > 0.005"


def test_opt_mwcolorlaw_zero_simulates_no_extinction(snana_fixture, tmp_path):
    """OPT_MWCOLORLAW 0 turns Milky Way extinction off (MWEBV_FLAG = 0,
    snlc_sim.c:7179-7181), so gen_MWEBV returns before touching MWEBV
    (:13629): the true MWEBV and its error stay 0, and the reported MWEBV is
    the SIMLIB value under OPT_MWEBV 1 (:20525) and 0 otherwise.

    The photometry equals that of the same input with GENSCALE_MWEBV 0, as it
    does in SNANA. SNANA's own OPT_MWEBV 1 simulation of the same SIMLIB gives
    the reported MWEBV at each LIBID."""
    fx = snana_fixture("mwdust_opt1")
    source = fx.input_file.read_text()

    def run(name, law, opt_mwebv, extra=""):
        text = (source.replace("OPT_MWCOLORLAW:  99", f"OPT_MWCOLORLAW:  {law}")
                .replace("OPT_MWEBV:  1", f"OPT_MWEBV:  {opt_mwebv}"))
        inp = tmp_path / f"{name}.input"
        inp.write_text(text + extra)
        config = dict(n_sne=200, random_seed=122459)
        sim = sim_run.prepare(str(inp), **config)
        simulation = None if sim is None else sim.config
        resources = sim
        results, sel, spectra_cache = sim_run.run_simulation(sim)
        save_snana_outputs(sim, results, name, sel, spectra_by_sn=spectra_cache, output_dir=str(tmp_path))
        with fits.open(next((tmp_path / name).glob("*HEAD*"))) as h, \
                fits.open(next((tmp_path / name).glob("*PHOT*"))) as p:
            return (h[0].header['SIMOPT_MWCOLORLAW'],
                    {c: np.asarray(h[1].data[c]) for c in h[1].data.names},
                    {c: np.asarray(p[1].data[c]) for c in p[1].data.names})

    law, head, phot = run("OFF", 0, 1)
    assert law == 0
    assert len(head['SNID']) > 50
    np.testing.assert_array_equal(head['SIM_MWEBV'], 0.0)
    np.testing.assert_array_equal(head['MWEBV_ERR'], 0.0)
    with fits.open(fx.head_fits) as hdul:
        snana = hdul[1].data
        snana_mwebv = dict(zip(snana['SIM_LIBID'].astype(int),
                               snana['MWEBV'].astype(np.float32)))
    np.testing.assert_array_equal(
        head['MWEBV'], [snana_mwebv[int(i)] for i in head['SIM_LIBID']])
    assert np.all(head['MWEBV'] > 0.0)

    _, _, phot_unscaled = run("SCALE0", 99, 1, "GENSCALE_MWEBV:  0\n")
    np.testing.assert_array_equal(phot['SIM_MAGOBS'],
                                  phot_unscaled['SIM_MAGOBS'])
    np.testing.assert_array_equal(phot['FLUXCAL'], phot_unscaled['FLUXCAL'])

    _, head_sfd, _ = run("OFF_SFD", 0, 2)
    np.testing.assert_array_equal(head_sfd['MWEBV'], 0.0)
    np.testing.assert_array_equal(head_sfd['SIM_MWEBV'], 0.0)


def test_peakmag_null_filters_match_snana(snana_fixture, tmp_path):
    """SIM_PEAKMAG is -9 for filters with no kept epoch, as in SNANA.

    SNANA leaves peakmag_obs at NULLFLOAT unless the filter appears in the
    SIMLIB entry (snlc_sim.c:9639, :12971-12974). jax masks on the same
    predicate, so the two sentinel patterns agree except where the CMB-vs-
    heliocentric MJD window disagrees at its edge;
    that costs a handful of cells out of 9000, so the bound is on the count.
    """
    fx = snana_fixture("mwdust_parity")
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
    save_snana_outputs(sim, results, "PK", sel, spectra_by_sn=spectra_cache, output_dir=str(tmp_path))

    with fits.open(next((tmp_path / "PK").glob("PK*HEAD*"))) as hdul:
        jax_cols = {n: np.asarray(hdul[1].data[n]) for n in hdul[1].data.names}

    order = {int(c): i for i, c in enumerate(jax_cols['SNID'].astype(int))}
    rows = [(k, order[int(c)]) for k, c in enumerate(head_cols['SNID'].astype(int))
            if int(c) in order]
    assert len(rows) > 1400, f"only {len(rows)} SNe matched"

    bands = [n.split('_')[-1] for n in head_cols if n.startswith('SIM_PEAKMAG_')]
    n_cells = n_disagree = n_sentinel = 0
    for b in bands:
        col = f'SIM_PEAKMAG_{b}'
        s9 = np.array([head_cols[col][k] <= -8.9 for k, _ in rows])
        j9 = np.array([jax_cols[col][j] <= -8.9 for _, j in rows])
        n_cells += len(s9)
        n_disagree += int((s9 != j9).sum())
        n_sentinel += int(s9.sum())

    # the fixture must actually contain null filters, or an unmasked output
    # would pass this test
    assert n_sentinel > 2000, f"only {n_sentinel} SNANA sentinels; fixture too weak"
    assert n_disagree <= 10, (
        f"{n_disagree}/{n_cells} SIM_PEAKMAG sentinel cells disagree with SNANA")
