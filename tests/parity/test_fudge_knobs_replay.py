"""FUDGESHIFT_ZPT / FUDGESCALE_FLUXERR(2) / FUDGESHIFT_LAM replay parity.

The fixture sets one knob per band: ZPT+0.30 (r), fluxerr x1.30 (g).
Both knobs only touch errors, gated by per-band median FLUXCALERR ratios
(jax/snana ~ 1 when both sides apply the knob; a missing 30% scaling
separates cleanly from the ~1% noisy-Poisson replay floor). FUDGESHIFT_LAM
is dead in SNANA's sim; ours is validated against kcor.exe FILTER_LAMSHIFT
in the live tier (test_lamshift_kcor_ab). FUDGESCALE_FLUXERR2's input key
is dead in SNANA (strstr parse trap) so it cannot appear in the fixture;
ours works via both the input key and RANSYSTPAR_SIGSCALE_FLUXERR2.
"""

import numpy as np
from astropy.io import fits

from jax_snlc_core.io.snana_dump_reader import load_snana_dump  # noqa: E402
from replay_snana_dump import build_override_parameters  # noqa: E402
from jax_snlc_sim import run as sim_run  # noqa: E402
from _replay_helpers import load_snana_phot, load_jax_phot  # noqa: E402


def test_fudge_knobs_replay(snana_fixture, tmp_path):
    fx = snana_fixture("fudge_knobs_replay")

    dump = load_snana_dump(str(fx.dump), dedup_cid=True)
    with fits.open(fx.head_fits) as hdul:
        head_cols = {n: hdul[1].data[n] for n in hdul[1].data.names}

    config = dict(n_sne=dump.size)
    sim = sim_run.prepare(str(fx.input_file), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    assert simulation.instrument.fudgeshift_zpt.get('r') == 0.30
    assert simulation.instrument.fudgescale_fluxerr.get('g') == 1.30
    override = build_override_parameters(dump.columns, simulation, resources, head_columns=head_cols)

    results, sel, spectra_cache = sim_run.run_simulation(sim, override_parameters=override)
    sim_run.save_snana_outputs(sim, results, "FUDGE", sel, spectra_cache=spectra_cache, output_dir=str(tmp_path))

    snana_phot = load_snana_phot(fx.head_fits, fx.phot_fits)
    jax_phot = load_jax_phot(tmp_path / "FUDGE", "FUDGE")

    mag_diffs = {}
    err_ratios = {}
    for key in set(snana_phot) & set(jax_phot):
        _, band, _ = key
        sn_mag, sn_flux, sn_err, _ = snana_phot[key]
        jx_mag, _, jx_err, _ = jax_phot[key]
        short = band.split('-')[-1]
        if sn_mag < 50.0 and jx_mag < 50.0 and sn_err > 0 and sn_flux / sn_err >= 3.0:
            mag_diffs.setdefault(short, []).append(abs(jx_mag - sn_mag))
        if sn_err > 0 and np.isfinite(jx_err) and jx_err > 0:
            err_ratios.setdefault(short, []).append(jx_err / sn_err)

    # SIM_MAGOBS parity per band (validates the z-band lambda shift end to end)
    all_mags = np.concatenate([np.array(v) for v in mag_diffs.values()])
    assert len(all_mags) > 300
    assert float(np.median(all_mags)) < 0.005
    for band in ('r', 'g', 'i'):
        d = np.array(mag_diffs.get(band, []))
        assert len(d) > 30, f"too few {band} epochs ({len(d)})"
        assert float(np.median(d)) < 0.005, f"{band}: median |dmag| {np.median(d):.4f}"

    # FLUXCALERR ratio per band (validates ZPT shift + both fluxerr scales)
    for band in ('u', 'g', 'r', 'i', 'z', 'Y'):
        r = np.array(err_ratios.get(band, []))
        if len(r) < 30:
            continue
        med = float(np.median(r))
        assert abs(med - 1.0) < 0.03, f"{band}: median err ratio {med:.3f}"
