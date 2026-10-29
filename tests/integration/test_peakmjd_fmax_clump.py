"""OPT_SETPKMJD=16 (Fmax-clump) recovers peak_mjd from observed photometry.

The simulator generates SNe with truth peak_mjd_true and, when OPT_SETPKMJD
selects the Fmax-clump method, post-processes the observed light curves to
estimate peak_mjd. With dense cadence the recovered value should track truth
within a few days for the bulk of the sample.
"""


import numpy as np
from astropy.io import fits

from jax_snlc_sim import run as sim_run
from jax_snlc_sim.output import save_snana_outputs


def test_fmax_clump_recovers_peak_mjd(snana_fixture):
    """Recovered peak_mjd is within a few days of truth for the bulk of SNe."""
    fx = snana_fixture("peakmjd_fmax_clump")
    cfg = dict(n_sne=100, random_seed=122459)
    sim = sim_run.prepare(str(fx.input_file), **cfg)
    results, _, _ = sim_run.run_simulation(sim)
    p = results["parameters"]

    pk_true = np.asarray(p["peakmjd_true"])
    pk_reco = np.asarray(p["peakmjd"])
    sentinel = np.isclose(pk_reco, -9.0) | (pk_reco < 50000)
    valid = ~sentinel
    assert valid.sum() > 50, f"too few SNe with non-sentinel peak_mjd ({valid.sum()})"

    diff = pk_reco[valid] - pk_true[valid]
    median_abs = float(np.median(np.abs(diff)))
    p95_abs = float(np.percentile(np.abs(diff), 95))

    # Dense cadence (~3 d/band) lets Fmax-clump recover peak_mjd within ~1 d
    # at median. Anything > 5 d at median signals a regression in the
    # peak-recovery routine.
    assert median_abs < 5.0, f"median |Δpeak_mjd| = {median_abs:.2f} d > 5"
    assert p95_abs < 15.0, f"p95 |Δpeak_mjd| = {p95_abs:.2f} d > 15"


def test_gensigma_peakmjd_smears_reported_peak(snana_fixture, tmp_path):
    """GENSIGMA_PEAKMJD selects the legacy Gaussian smear (forcing
    OPT_SETPKMJD=0): the reported peak_mjd = peak_mjd_true + N(0, sigma),
    so the residual distribution has ~zero mean and std ≈ sigma."""
    fx = snana_fixture("peakmjd_fmax_clump")
    sigma = 2.0
    inp = tmp_path / "input.input"
    inp.write_text(open(fx.input_file).read() + f"\nGENSIGMA_PEAKMJD: {sigma}\n")
    cfg = dict(n_sne=150, random_seed=333)
    sim = sim_run.prepare(str(inp), **cfg)
    results, _, _ = sim_run.run_simulation(sim)
    p = results["parameters"]

    diff = np.asarray(p["peakmjd"]) - np.asarray(p["peakmjd_true"])
    assert diff.shape[0] > 100
    assert abs(float(np.mean(diff))) < 0.4 * sigma
    assert 0.7 * sigma < float(np.std(diff)) < 1.3 * sigma


def test_gensigma_peakmjd_smear_is_batch_invariant(snana_fixture, tmp_path):
    """The per-SN peak smear is keyed by global SN index, so a given SN gets
    the same smear regardless of batch_size (guards against re-seeding per
    batch, which would repeat smears across batches)."""
    fx = snana_fixture("peakmjd_fmax_clump")
    inp = tmp_path / "input.input"
    inp.write_text(open(fx.input_file).read() + "\nGENSIGMA_PEAKMJD: 2.0\n")

    def run(batch_size):
        cfg = dict(n_sne=120,
                                  random_seed=444, batch_size=batch_size)
        sim = sim_run.prepare(str(inp), **cfg)
        results, _, _ = sim_run.run_simulation(sim)
        p = results["parameters"]
        smear = np.asarray(p["peakmjd"]) - np.asarray(p["peakmjd_true"])
        return dict(zip(np.asarray(p["cid"]).tolist(), smear.tolist()))

    one_batch = run(1000)   # 250 SNe → single batch
    many_batches = run(40)   # 250 SNe → 7 batches
    common = set(one_batch) & set(many_batches)
    assert len(common) > 80
    for cid in common:
        assert abs(one_batch[cid] - many_batches[cid]) < 1e-9


def test_fmax_clump_resolves_clumps_beyond_1000_days():
    """The 10-day histogram is sized from the light curve's MJD span
    (MXWIN_SNRCUT, get_obs_atFLUXMAX): two clumps in a multi-season
    baseline stay in separate bins, and the denser one wins even when a
    brighter sparse clump sits nearby in time."""
    from jax_snlc_core.peak_time import fmax_clump_peakmjd

    mjd_a = np.arange(0.0, 50.0, 10.0)              # 5 bright early epochs
    mjd_c = np.array([1000.0, 1003.0, 1006.0])      # 3 brighter epochs
    mjd_b = np.arange(1400.0, 1450.0, 2.5)          # 20-epoch dense clump
    mjds = np.concatenate([mjd_a, mjd_c, mjd_b])[None, :]
    flux = np.concatenate([np.full(5, 100.0), np.full(3, 200.0),
                           np.full(20, 50.0)])[None, :]
    snr = np.full_like(flux, 10.0)
    peak = fmax_clump_peakmjd(
        mjds, flux, snr, np.array([mjds.shape[1]]),
        mjdwin=60.0, snrcut=5.0, snrcut_backup=3.0)
    assert 1400.0 <= float(peak[0]) <= 1450.0


def test_take_spectrum_epochs_follow_the_generated_peak(snana_fixture, tmp_path):
    """TAKE_SPECTRUM TREST epochs are laid out from SIM_PEAKMJD, not the
    OPT_SETPKMJD estimate the HEAD reports as PEAKMJD (GENSPEC_PICKMJD,
    snlc_sim.c:10867 uses GENLC.PEAKMJD)."""
    fx = snana_fixture("spectro_optmask_replay")
    text = open(fx.input_file).read()
    assert "OPT_SETPKMJD: 0" in text and "TAKE_SPECTRUM:  TREST(5:5)" in text
    inp = tmp_path / "input.input"
    inp.write_text(text.replace("OPT_SETPKMJD: 0", "OPT_SETPKMJD: 16")
                   .replace("GENSIGMA_PEAKMJD:  1.0000", "GENSIGMA_PEAKMJD: 0"))
    cfg = dict(n_sne=40, random_seed=122459)
    sim = sim_run.prepare(str(inp), **cfg)
    results, sel, spectra_cache = sim_run.run_simulation(sim)
    save_snana_outputs(sim, results, "SPECPK", sel, spectra_by_sn=spectra_cache, output_dir=str(tmp_path))

    with fits.open(tmp_path / "SPECPK" / "SPECPK_HEAD.FITS.gz") as h:
        head = {int(r["SNID"]): (float(r["SIM_PEAKMJD"]), float(r["PEAKMJD"]),
                                 float(r["REDSHIFT_HELIO"])) for r in h[1].data}
    with fits.open(tmp_path / "SPECPK" / "SPECPK_SPEC.FITS.gz") as h:
        spectra = [(int(r["SNID"]), float(r["MJD"])) for r in h[1].data]
    assert len(spectra) > 20

    trest = np.array([(mjd - head[snid][0]) / (1.0 + head[snid][2])
                      for snid, mjd in spectra])
    # the estimate differs from the generated peak here, so the two are told apart
    assert np.max(np.abs([head[s][1] - head[s][0] for s, _ in spectra])) > 1.0
    # MJD is written as float32
    np.testing.assert_allclose(trest, 5.0, atol=0.01)
