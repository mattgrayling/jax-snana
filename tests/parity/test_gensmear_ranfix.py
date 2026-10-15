"""G10/C11 intrinsic-scatter parity vs SNANA, with the smear randoms pinned.

SNANA does not dump its smear randoms, so a replay cannot reproduce them. But both
codes accept GENSMEAR_RANGAUSS_FIX, which sets every Gaussian smear random to one
value (sntools_genSmear.c load_genSmear_randoms; jax sampler.py fills
_gensmear_randoms). With it set, both evaluate the same DeltaM(lam) and SIM_MAGOBS
compares directly.

That covers the node sigmas and their wavelength assignment, the interpolation
between nodes, the wavelength bounds, C11's covariance transform, and the whole
application path onto the SED and through each filter. It does not exercise the
draw itself: the +/-3 sigma clipping, and -- because every random holds the same
value -- which random feeds which node.

The two codes agree to 0.2-0.8 mmag per band, and that floor is set by SED
interpolation and float precision rather than by the smear, so the tolerance below
is absolute. What it means fractionally depends on how much smear a band carries:
the signal spans 1.2 mmag (G10 r) to 111 mmag (C11 u), so the same 1.5 mmag bound
catches a 0.6% error in C11 u and cannot constrain G10 r at all. The strong bands
are what this test rests on.
"""

from jax_snlc_core.utils.error_handling import SNANAError
import numpy as np
import pytest
from astropy.io import fits

from jax_snlc_core.io.snana_dump_reader import load_snana_dump
from replay_snana_dump import build_override_parameters
from jax_snlc_sim import run as sim_run
from _replay_helpers import load_snana_phot, load_jax_phot

_RANFIX = 1.0
_RMS_TOL = 0.0015     # mag per band; ~2x the 0.2-0.8 mmag floor
_MAX_TOL = 0.010      # mag, worst single epoch


def _run(simulation, resources, config, sim, override, fill, n_sne, name, out_dir):
    """Simulate with every smear random set to `fill`."""
    randoms = np.full((n_sne, resources.gensmear_n_randoms), fill,
                      dtype=np.float32)
    params = dict(override, _gensmear_randoms=randoms)
    results, selection, spectra_cache = sim_run.run_simulation(sim, override_parameters=params)
    sim_run.save_snana_outputs(sim, results, name, selection, spectra_cache=spectra_cache, output_dir=str(out_dir))
    return load_jax_phot(out_dir / name, name)


@pytest.mark.parametrize("fixture_name", ["g10_smear_ranfix", "c11_smear_ranfix"])
def test_gensmear_ranfix_parity(fixture_name, snana_fixture, tmp_path):
    fx = snana_fixture(fixture_name)
    dump = load_snana_dump(str(fx.dump), dedup_cid=True)
    with fits.open(fx.head_fits) as hdul:
        head_cols = {n: hdul[1].data[n] for n in hdul[1].data.names}

    config = dict(n_sne=dump.size)
    sim = sim_run.prepare(str(fx.input_file), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    override = build_override_parameters(dump.columns, simulation, resources, head_columns=head_cols)

    pinned = _run(simulation, resources, config, sim, override, _RANFIX,
                  dump.size, "PINNED", tmp_path)
    smearfree = _run(simulation, resources, config, sim, override, 0.0,
                     dump.size, "SMEARFREE", tmp_path)
    snana_phot = load_snana_phot(fx.head_fits, fx.phot_fits)

    diff, control, bands = [], [], []
    for key in set(snana_phot) & set(pinned) & set(smearfree):
        sn_mag, sn_flux, sn_err, _ = snana_phot[key]
        if sn_mag >= 50.0 or pinned[key][0] >= 50.0:
            continue
        if sn_err <= 0 or sn_flux / sn_err < 3.0:
            continue
        diff.append(sn_mag - pinned[key][0])
        control.append(sn_mag - smearfree[key][0])
        bands.append(key[1])
    diff, control, bands = np.array(diff), np.array(control), np.array(bands)
    assert len(diff) > 200, f"too few epochs ({len(diff)})"

    # the smear must actually be present, or matching smear-free output would pass
    assert control.std() > 20 * diff.std(), (
        f"smear signal {control.std():.5f} is not clearly above the residual "
        f"{diff.std():.5f}; the fixture may not carry intrinsic scatter")

    for band in sorted(set(bands)):
        in_band = bands == band
        rms = float(diff[in_band].std())
        assert rms < _RMS_TOL, f"{band}: rms {rms:.5f} mag over {in_band.sum()} epochs"
    assert np.abs(diff).max() < _MAX_TOL, \
        f"worst epoch {np.abs(diff).max():.5f} mag"


def test_unpinned_smear_refuses_replay(snana_fixture, tmp_path):
    """Without the pin the node draws are unknown, so the replay must not guess."""
    fx = snana_fixture("g10_smear_ranfix")
    src = "\n".join(line for line in fx.input_file.read_text().splitlines()
                    if not line.startswith("GENSMEAR_RANGAUSS_FIX"))
    unpinned = tmp_path / "input.input"
    unpinned.write_text(src)

    dump = load_snana_dump(str(fx.dump), dedup_cid=True)
    with fits.open(fx.head_fits) as hdul:
        head_cols = {n: hdul[1].data[n] for n in hdul[1].data.names}
    config = dict(n_sne=dump.size)
    sim = sim_run.prepare(str(unpinned), **config)
    simulation = None if sim is None else sim.config
    resources = sim

    with pytest.raises(SNANAError, match="G10"):
        build_override_parameters(dump.columns, simulation, resources, head_columns=head_cols)

    # the coherent term SNANA does record stays replayable
    override = build_override_parameters(dump.columns, simulation, resources, head_columns=head_cols,
                                         drop_node_smear=True)
    assert float(np.abs(override["_gensmear_randoms"]).max()) == 0.0
    assert np.allclose(override["magsmear"], 0.106, atol=1e-6)
