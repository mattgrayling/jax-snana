"""SALT3.K21 SIM_MAGOBS parity: JAX replay vs bundled SNANA PHOT.FITS."""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from jax_snlc_core.io.fits_path import resolve_fits_path
from jax_snlc_core.utils.error_handling import SNANAError


from jax_snlc_core.io.snana_dump_reader import load_snana_dump  # noqa: E402
from replay_snana_dump import build_override_parameters  # noqa: E402
from jax_snlc_sim import run as sim_run  # noqa: E402
from jax_snlc_sim.sampler import _population_weight  # noqa: E402
from _replay_helpers import slice_replay  # noqa: E402


def _load_phot(head_path: Path, phot_path: Path):
    with fits.open(head_path) as h, fits.open(phot_path) as p:
        head = h[1].data
        phot = p[1].data
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


def test_salt3_k21_simmagobs_parity(snana_fixture, tmp_path):
    """JAX replay reproduces SNANA's SIM_MAGOBS for SALT3.K21 (cropped grid)."""
    fx = snana_fixture("salt3_k21_replay")

    dump = load_snana_dump(str(fx.dump), dedup_cid=True)
    with fits.open(fx.head_fits) as hdul:
        head_cols = {n: hdul[1].data[n] for n in hdul[1].data.names}

    # 300 of 1500 events: ~900 usable epochs, comfortably above the gate below
    dump_cols, head_cols = slice_replay(dump.columns, head_cols, 300)
    config = dict(n_sne=len(dump_cols['CID']))
    sim = sim_run.prepare(str(fx.input_file), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    override = build_override_parameters(dump_cols, simulation, resources,
                                         head_columns=head_cols)

    results, sel, spectra_cache = sim_run.run_simulation(sim, override_parameters=override)
    sim_run.save_snana_outputs(sim, results, "SALT3_REPLAY", sel, spectra_cache=spectra_cache, output_dir=str(tmp_path))

    snana_phot = _load_phot(fx.head_fits, fx.phot_fits)
    jax_dir = tmp_path / "SALT3_REPLAY"
    jax_phot = _load_phot(
        next(jax_dir.glob("SALT3_REPLAY*HEAD*")),
        next(jax_dir.glob("SALT3_REPLAY*PHOT*")),
    )

    diffs = []
    common = set(snana_phot) & set(jax_phot)
    assert len(common) > 0, "no overlapping (SNID, MJD, BAND) keys"
    for key in common:
        sn_mag, sn_flux, sn_err = snana_phot[key]
        jx_mag, _, _ = jax_phot[key]
        if sn_err <= 0 or sn_flux / sn_err < 3.0:
            continue
        if sn_mag >= 50.0 or jx_mag >= 50.0:
            continue
        diffs.append(jx_mag - sn_mag)

    assert len(diffs) > 750, f"too few S/N≥3 epochs ({len(diffs)})"
    diffs = np.array(diffs)
    abs_diffs = np.abs(diffs)
    median = float(np.median(abs_diffs))
    p95 = float(np.percentile(abs_diffs, 95))

    assert median < 0.001, f"median |Δmag| = {median:.4f} > 0.001"
    assert p95 < 0.005, f"95th percentile |Δmag| = {p95:.4f} > 0.005"


def test_salt3_hostdust_simmagobs_parity(snana_fixture, tmp_path):
    """SALT3.K21 reddened by host dust matches SNANA per epoch.

    SNANA applies host extinction to SALT inside the filter integral, as
    HOSTXT_FRAC beside MWXT_FRAC (genmag_SALT2.c:2748-2752), on the colour
    law OPT_MWCOLORLAW selects (fill_TABLE_HOSTXT_SEDMODEL,
    genmag_SEDtools.c:2784). The fixture pins that at 99, FITZ99_EXACT, which
    is the law jax uses for host dust.

    The residual is also checked against AV itself: a term that is scaled
    wrongly rather than missing shows up as a trend across the AV range and
    not in the pooled median.
    """
    fx = snana_fixture("salt3_hostdust_replay")

    dump = load_snana_dump(str(fx.dump), dedup_cid=True)
    with fits.open(fx.head_fits) as hdul:
        head_cols = {n: hdul[1].data[n] for n in hdul[1].data.names}

    config = dict(n_sne=len(dump.columns['CID']))
    sim = sim_run.prepare(str(fx.input_file), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    override = build_override_parameters(dump.columns, simulation, resources,
                                         head_columns=head_cols)

    results, sel, spectra_cache = sim_run.run_simulation(sim, override_parameters=override)
    sim_run.save_snana_outputs(sim, results, "SALT3_HOSTDUST", sel, spectra_cache=spectra_cache, output_dir=str(tmp_path))

    snana_phot = _load_phot(fx.head_fits, fx.phot_fits)
    jax_dir = tmp_path / "SALT3_HOSTDUST"
    jax_phot = _load_phot(
        next(jax_dir.glob("SALT3_HOSTDUST*HEAD*")),
        next(jax_dir.glob("SALT3_HOSTDUST*PHOT*")),
    )

    av_by_snid = dict(zip((int(c) for c in head_cols['SNID']),
                          np.asarray(head_cols['SIM_AV'], dtype=float)))
    assert max(av_by_snid.values()) > 1.0, "fixture carries no reddened events"

    diffs, avs = [], []
    common = set(snana_phot) & set(jax_phot)
    assert len(common) > 0, "no overlapping (SNID, MJD, BAND) keys"
    for key in common:
        sn_mag, sn_flux, sn_err = snana_phot[key]
        jx_mag, _, _ = jax_phot[key]
        if sn_err <= 0 or sn_flux / sn_err < 3.0:
            continue
        if sn_mag >= 50.0 or jx_mag >= 50.0:
            continue
        diffs.append(jx_mag - sn_mag)
        avs.append(av_by_snid[key[0]])

    assert len(diffs) > 400, f"too few S/N≥3 epochs ({len(diffs)})"
    abs_diffs = np.abs(np.array(diffs))
    avs = np.array(avs)
    median = float(np.median(abs_diffs))
    p95 = float(np.percentile(abs_diffs, 95))

    assert median < 0.001, f"median |Δmag| = {median:.4f} > 0.001"
    assert p95 < 0.005, f"95th percentile |Δmag| = {p95:.4f} > 0.005"

    # No trend with reddening: the most-reddened quartile must agree as well
    # as the least. A dust term off by a scale factor fails here first.
    lo, hi = np.percentile(avs, [25, 75])
    faint, bright = abs_diffs[avs >= hi], abs_diffs[avs <= lo]
    assert faint.size > 50 and bright.size > 50, "AV quartiles too sparse"
    assert float(np.median(faint)) < 0.002, (
        f"median |Δmag| at AV>={hi:.2f} is {np.median(faint):.4f}")


def test_replay_takes_the_distance_from_head_when_dump_lacks_mu(snana_fixture):
    """DUMP MU and HEAD SIM_DLMU are both GENLC.DLMU (snlc_sim.c:16090,
    :25471), so a dump written without MU replays from the HEAD; the dump
    carries four decimals, hence the tolerance. A HEAD that does not cover
    every dump event cannot stand in."""
    fx = snana_fixture("salt3_k21_replay")
    dump = load_snana_dump(str(fx.dump), dedup_cid=True)
    with fits.open(fx.head_fits) as hdul:
        head_cols = {n: hdul[1].data[n] for n in hdul[1].data.names}
    dump_cols, head_cols = slice_replay(dump.columns, head_cols, 50)
    config = dict(n_sne=50)
    sim = sim_run.prepare(str(fx.input_file), **config)
    simulation = None if sim is None else sim.config
    resources = sim

    with_mu = build_override_parameters(dump_cols, simulation, resources,
                                        head_columns=head_cols)
    no_mu = {k: v for k, v in dump_cols.items() if k != 'MU'}
    from_head = build_override_parameters(no_mu, simulation, resources,
                                          head_columns=head_cols)
    np.testing.assert_allclose(from_head['mu'], with_mu['mu'], rtol=0, atol=1e-4)

    # equal length must not mean positional: a reordered HEAD joins by SNID
    perm = np.random.default_rng(3).permutation(len(head_cols['SNID']))
    shuffled = {k: np.asarray(v)[perm] for k, v in head_cols.items()}
    reordered = build_override_parameters(no_mu, simulation, resources,
                                          head_columns=shuffled)
    np.testing.assert_array_equal(reordered['mu'], from_head['mu'])

    short_head = {k: v[1:] for k, v in head_cols.items()}
    with pytest.raises(SNANAError, match="SIM_DLMU"):
        build_override_parameters(no_mu, simulation, resources, head_columns=short_head)


def test_population_weight_matches_snana(snana_fixture):
    """SIM_WGT_POPULATION under GENPDF_EXPON_REWGT is SNANA's
    prob(nominal)/prob(sharpened) at the drawn stretch and colour
    (rewgt_genPDF, snlc_sim.c:31492). The fixture's populations have
    asymmetric widths, a peakrange plateau and a second colour peak, so every
    branch of funVal_GENGAUSS_ASYM meets SNANA's own numbers."""
    fx = snana_fixture("population_rewgt")
    with fits.open(resolve_fits_path(fx.root / "HEAD.FITS")) as h:
        x1 = np.asarray(h[1].data["SIM_SALT2x1"], dtype=float)
        c = np.asarray(h[1].data["SIM_SALT2c"], dtype=float)
        snana = np.asarray(h[1].data["SIM_WGT_POPULATION"], dtype=float)
    simulator = sim_run.prepare(str(fx.root / "input.input"))

    jax_weight = (np.asarray(_population_weight(simulator, "x1", x1, None, None))
                  * np.asarray(_population_weight(simulator, "c", c, None, None)))

    assert snana.max() > 5.0, "the sharpening must have produced real weights"
    np.testing.assert_allclose(jax_weight, snana, rtol=2e-6)


def test_population_with_a_second_peak_simulates(snana_fixture):
    """The fixture's colour population has a second peak (GENPROB2_SALT2c),
    which the compiled simulation draws: a run reaches both peaks and records
    each event's weight at the values it drew."""
    fx = snana_fixture("population_rewgt")
    simulator = sim_run.prepare(str(fx.root / "input.input"), n_sne=200,
                                random_seed=7)
    results, _, _ = sim_run.run_simulation(simulator)
    x1 = np.asarray(results["parameters"]["x1"])
    c = np.asarray(results["parameters"]["c"])

    assert np.mean((c > 0.1) & (c < 0.2)) > 0.1, "the second peak is drawn"
    expected = (np.asarray(_population_weight(simulator, "x1", x1, None, None))
                * np.asarray(_population_weight(simulator, "c", c, None, None)))
    np.testing.assert_allclose(results["parameters"]["wgt_population"],
                               expected, rtol=1e-6)
