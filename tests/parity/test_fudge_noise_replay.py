"""FUDGESCALE_PSF / NOISE_SKY / NOISE_READ / NOISE_TEMPLATE replay parity.

The fixture sets all four global scales at distinct values (1.2 / 1.3 /
1.5 / 1.4), plus FUDGESHIFT_ZPT (r +0.30), on a SIMLIB variant with
per-field template noise and RDNOISE large enough (4.0) that a missing
READ scaling separates from the replay floor. The r-band error gate covers
the template-ZPT shift (snlc_sim.c:20835): the template zpt_scale term
mismatches if only the search ZPT is shifted. SNANA applies them per epoch at SIMLIB prep (snlc_sim.c:20777-20783,
template product at :20827-20832); a missing scale shows up in the per-band
FLUXCALERR ratio (SNANA-vs-control ratios are 1.4-1.7).
"""

import numpy as np
import pytest
from _replay_helpers import load_jax_phot, load_snana_phot
from astropy.io import fits
from replay_snana_dump import build_override_parameters

from jax_snlc_core.snana_files import load_snana_dump
from jax_snlc_sim import run as sim_run
from jax_snlc_sim.output import save_snana_outputs


@pytest.mark.parametrize("fixture_name",
                         ["fudge_noise_replay", "fudge_noise_nea_replay"])
def test_fudge_noise_replay(snana_fixture, tmp_path, fixture_name):
    fx = snana_fixture(fixture_name)

    dump = load_snana_dump(str(fx.dump), dedup_cid=True)
    with fits.open(fx.head_fits) as hdul:
        head_cols = {n: hdul[1].data[n] for n in hdul[1].data.names}

    config = dict(n_sne=dump.size)
    sim = sim_run.prepare(str(fx.input_file), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    assert simulation.instrument.fudgescale_psf == 1.2
    assert simulation.instrument.fudgescale_noise_sky == 1.3
    assert simulation.instrument.fudgescale_noise_read == 1.5
    assert simulation.instrument.fudgescale_noise_template == 1.4
    assert simulation.instrument.fudgeshift_zpt.get('r') == 0.30
    override = build_override_parameters(dump.columns, simulation, resources, head_columns=head_cols)

    results, sel, spectra_cache = sim_run.run_simulation(sim, override_parameters=override)
    save_snana_outputs(sim, results, "FNOISE", sel, spectra_by_sn=spectra_cache, output_dir=str(tmp_path))

    snana_phot = load_snana_phot(fx.head_fits, fx.phot_fits)
    jax_phot = load_jax_phot(tmp_path / "FNOISE", "FNOISE")

    mag_diffs = []
    err_ratios = {}
    for key in set(snana_phot) & set(jax_phot):
        _, band, _ = key
        sn_mag, sn_flux, sn_err, _ = snana_phot[key]
        jx_mag, _, jx_err, _ = jax_phot[key]
        short = band.split('-')[-1]
        if sn_mag < 50.0 and jx_mag < 50.0 and sn_err > 0 and sn_flux / sn_err >= 3.0:
            mag_diffs.append(abs(jx_mag - sn_mag))
        if sn_err > 0 and np.isfinite(jx_err) and jx_err > 0:
            err_ratios.setdefault(short, []).append(jx_err / sn_err)

    # noise scales must not perturb the model fluxes
    mag_diffs = np.array(mag_diffs)
    assert len(mag_diffs) > 300
    assert float(np.median(mag_diffs)) < 0.005

    # FLUXCALERR ratio per band gates all four scales
    for band in ('u', 'g', 'r', 'i', 'z', 'Y'):
        r = np.array(err_ratios.get(band, []))
        assert len(r) > 30, f"too few {band} epochs ({len(r)})"
        med = float(np.median(r))
        assert abs(med - 1.0) < 0.03, f"{band}: median err ratio {med:.3f}"
