"""MJD_TEMPLATE replay parity.

The fixture puts the r and i reference images at MJD 61000, mid-way through
GENRANGE_PEAKMJD, so a good fraction of light curves are still bright when the
template was taken. Difference imaging then subtracts that flux from every
epoch in those bands (snlc_sim.c:28530).

The subtraction is compared as the systematic offset between FLUXCAL and the
flux SIM_MAGOBS implies: the noise realisations differ between the two codes,
but the offset does not, and SNANA leaves SIM_MAGOBS untouched by the
subtraction. The untouched bands pin the null.
"""

import numpy as np
from astropy.io import fits

from jax_snlc_core.io.fits_path import resolve_fits_path  # noqa: E402
from jax_snlc_core.io.snana_dump_reader import load_snana_dump  # noqa: E402
from replay_snana_dump import build_override_parameters  # noqa: E402
from jax_snlc_sim import run as sim_run  # noqa: E402
from _replay_helpers import load_snana_phot, load_jax_phot  # noqa: E402

TEMPLATE_BANDS = ('r', 'i')
CONTROL_BANDS = ('g', 'z')


def _offsets(phot, zp_fluxcal):
    """Median FLUXCAL - flux(SIM_MAGOBS) per band."""
    per_band = {}
    for (_, band, _), (mag, flux, err, *_) in phot.items():
        if not (0.0 < mag < 50.0) or not np.isfinite(flux):
            continue
        implied = 10.0 ** (-0.4 * (mag - zp_fluxcal))
        per_band.setdefault(band.split('-')[-1], []).append(flux - implied)
    return {b: float(np.median(v)) for b, v in per_band.items() if len(v) >= 30}


def test_mjd_template_replay(snana_fixture, tmp_path):
    fx = snana_fixture("mjd_template_replay")

    dump = load_snana_dump(str(fx.dump), dedup_cid=True)
    with fits.open(fx.head_fits) as hdul:
        head_cols = {n: hdul[1].data[n] for n in hdul[1].data.names}

    config = dict(n_sne=dump.size)
    sim = sim_run.prepare(str(fx.input_file), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    assert simulation.survey.mjd_template.get('r') == 61000.0
    assert simulation.survey.mjd_template.get('i') == 61000.0

    override = build_override_parameters(dump.columns, simulation, resources,
                                         head_columns=head_cols)
    results, sel, spectra_cache = sim_run.run_simulation(sim, override_parameters=override)
    sim_run.save_snana_outputs(sim, results, "MJDT", sel, spectra_cache=spectra_cache, output_dir=str(tmp_path))

    with fits.open(resolve_fits_path(fx.head_fits)) as h:
        zp_fluxcal = float(h[0].header["ZP_FLUXCAL"])
    assert zp_fluxcal == simulation.instrument.zp_fluxcal
    snana = _offsets(load_snana_phot(fx.head_fits, fx.phot_fits), zp_fluxcal)
    jax = _offsets(load_jax_phot(tmp_path / "MJDT", "MJDT"), zp_fluxcal)

    # Band-level medians of single-epoch residuals carry a few FLUXCAL of noise
    # (at ZP_FLUXCAL 27.5), since the two codes draw their own noise; the
    # untouched bands measure it.
    noise = 5.0 * 10.0 ** (0.4 * (zp_fluxcal - 27.5))
    for band in CONTROL_BANDS:
        assert band in snana and band in jax, f"no {band} epochs"
        assert abs(jax[band] - snana[band]) < noise, (
            f"{band}: untouched band drifted, "
            f"jax {jax[band]:.3f} vs SNANA {snana[band]:.3f}")

    for band in TEMPLATE_BANDS:
        assert band in snana and band in jax, f"no {band} epochs"
        # The fixture must actually be subtracting, well clear of that noise.
        assert snana[band] < -10.0 * noise, f"{band}: SNANA offset {snana[band]:.3f}"
        assert abs(jax[band] - snana[band]) / abs(snana[band]) < 0.03, (
            f"{band}: jax offset {jax[band]:.3f} vs SNANA {snana[band]:.3f}")


def _phot(sim_input, tmp_path, name, extra="", n_sne=60):
    """FLUXCAL per band from a plain (non-replay) run of a modified input."""
    path = tmp_path / f"{name}.input"
    path.write_text(sim_input + extra)
    out = tmp_path / name
    config = dict(n_sne=n_sne)
    sim = sim_run.prepare(str(path), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    results, sel, spectra_cache = sim_run.run_simulation(sim)
    sim_run.save_snana_outputs(sim, results, name, sel, spectra_cache=spectra_cache, output_dir=str(out))
    phot = next(iter(sorted((out / name).glob(f"{name}*PHOT*"))))
    with fits.open(resolve_fits_path(phot)) as hdul:
        band = np.char.strip(np.array(hdul[1].data['BAND'], dtype='U'))
        flux = np.asarray(hdul[1].data['FLUXCAL'], dtype=np.float64)
    return np.array([b.split('-')[-1] for b in band]), flux, sel


TEMPLATE_KEY = "MJD_TEMPLATE_FILTER:  r,i 61000.0,61000.0"


def test_mjd_template_only_touches_its_own_bands(snana_fixture, tmp_path):
    """The per-band gate: a band with no MJD_TEMPLATE is untouched, and a
    template epoch where the model has no flux subtracts nothing anywhere."""
    src = snana_fixture("mjd_template_replay").input_file.read_text()

    bands, on, _ = _phot(src, tmp_path, "ON")
    _, off, _ = _phot(src.replace(TEMPLATE_KEY, ""), tmp_path, "OFF")
    # Pre-explosion: below the model's phase range, so the template mag is null.
    _, pre, _ = _phot(src.replace(TEMPLATE_KEY,
                                  "MJD_TEMPLATE_FILTER:  r,i 60000.0,60000.0"),
                      tmp_path, "PRE")

    for band in CONTROL_BANDS + ('u', 'Y'):
        m = bands == band
        if m.sum():
            assert np.array_equal(on[m], off[m]), f"{band} was touched"
    for band in TEMPLATE_BANDS:
        m = bands == band
        assert np.max(np.abs(on[m] - off[m])) > 1.0, f"{band} was not subtracted"

    assert np.array_equal(pre, off), "a null template magnitude still subtracted"


def test_mjd_template_subtraction_reaches_saturation(snana_fixture, tmp_path):
    """Saturation is judged on the flux after the template is removed
    (snlc_sim.c:28548 passes the subtracted fluxTrue to npe_above_saturation)."""
    src = snana_fixture("mjd_template_replay").input_file.read_text()
    npe = "\nNPE_PIXEL_SATURATE: 100\n"

    _, _, sel_on = _phot(src, tmp_path, "SATON", extra=npe)
    _, _, sel_off = _phot(src.replace(TEMPLATE_KEY, ""), tmp_path, "SATOFF", extra=npe)

    n_on = int(np.asarray(sel_on['saturated_mask']).sum())
    n_off = int(np.asarray(sel_off['saturated_mask']).sum())
    assert n_off > 0, "threshold too high to exercise saturation"
    assert n_on < n_off, (
        f"removing template flux should desaturate epochs: {n_on} vs {n_off}")
