"""Per-filter primary correctness for ``load_photometry_resources``.

The test uses a real mixed-primary kcor fixture (one AB filter, one BD17
filter) and asserts that each filter's reference-flux integral uses its
own assigned primary, not a globally-hardcoded AB.
"""

import gzip
import os

import numpy as np
import jax.numpy as jnp
import pytest
from astropy.io import fits

from jax_snlc_core.io.kcor_reader import KcorReader
from jax_snlc_core.photometry.resources import (load_photometry_resources,
                                                resolve_filtlist)
from jax_snlc_core.photometry.filter_system import (
    band_window_width, compute_single_restframe_filter, fluxcal_from_sed,
)
from jax_snlc_core.utils.error_handling import SNANAError

# AB is flat in f_nu at 3631 Jy, so its f_lambda is closed form.
_AB_FNU_CGS = 3631.0e-23          # erg s-1 cm-2 Hz-1
_C_ANGSTROM_PER_S = 2.99792458e18


@pytest.mark.parametrize("z", [.5, 1., 2.])
def test_redshifted_band_window(z):
    wave = jnp.linspace(3000., 9000., 500)
    observed = jnp.linspace(9500., 14500., 501)
    transmission = ((observed >= 10000.) & (observed <= 14000.)).astype(float)
    width = band_window_width(wave, observed, transmission[None])
    weights = compute_single_restframe_filter(z, observed, transmission, wave)[None]
    args = (lambda phase: 1. + wave / 5000., jnp.array([0.]),
            jnp.array([True]), jnp.array([0]), weights, jnp.array([1.]), 0., 1.)
    full = fluxcal_from_sed(*args)
    windowed = fluxcal_from_sed(*args, width=width)
    np.testing.assert_allclose(windowed, full, rtol=1e-12)


def test_filter_ref_fluxes_use_per_filter_primary(mixed_primaries_kcor_path):
    """Each filter's reference flux integrates against its assigned primary.

    The reference flux is a photon-weighted band average of the primary's
    f_lambda, so over a narrow band it sits on the primary's value at the
    band's mean wavelength. AB gives the closed-form 3631e-23 c / lambda^2
    there; BD17 is four orders of magnitude fainter in the same units, so a
    filter that took the wrong primary cannot land near its own value.
    """
    kcor = KcorReader(str(mixed_primaries_kcor_path))
    assert kcor.get_filter_primary("SDSS-g") == "AB"
    assert kcor.get_filter_primary("Keplercam-B") == "BD17"

    wavelength_grid = jnp.linspace(2000.0, 12000.0, 500)
    photometry = load_photometry_resources(
        kcor, wavelength_grid, ["SDSS-g", "Keplercam-B"],
    )

    # The ZPoff/primary_mag scale factor is applied to the stored fluxes;
    # reverse it here so we can compare against the primary's own flux.
    scale = np.power(10.0, 0.4 * np.array([
        kcor.filter_primary_mag["SDSS-g"] - kcor.filter_zpoff_snpot.get("SDSS-g", 0.0),
        kcor.filter_primary_mag["Keplercam-B"] - kcor.filter_zpoff_snpot.get("Keplercam-B", 0.0),
    ]))
    stored = np.asarray(photometry.filter_ref_fluxes) / scale
    mean_wave = np.asarray(photometry.filter_mean_wavelengths)

    ab_flam = _AB_FNU_CGS * _C_ANGSTROM_PER_S / mean_wave[0] ** 2
    np.testing.assert_allclose(stored[0], ab_flam, rtol=0.05)

    bd17_waves, bd17_flux = kcor.get_primary_sed("BD17")
    bd17_flam = np.interp(mean_wave[1], np.asarray(bd17_waves),
                          np.asarray(bd17_flux))
    np.testing.assert_allclose(stored[1], bd17_flam, rtol=0.05)

    # Had Keplercam-B taken AB, it would sit near the AB flux instead.
    assert stored[1] < 1.0e-3 * stored[0]


@pytest.mark.parametrize('wavelength_shift', [0., 1.23456789])
def test_float32_filter_tables_use_double_precision(mixed_primaries_kcor_path,
                                                   wavelength_shift):
    kcor = KcorReader(str(mixed_primaries_kcor_path))
    names = ['SDSS-g', 'Keplercam-B']
    photometry = load_photometry_resources(
        kcor, jnp.linspace(2000., 12000., 500), names,
        lam_shift_map=dict.fromkeys(names, wavelength_shift))
    expected = []
    for name in names:
        wave, transmission = kcor.get_filter_transmission(name)
        assert wave.dtype == transmission.dtype == np.float32
        wave = np.asarray(wave, dtype=np.float64) + wavelength_shift
        transmission = np.asarray(transmission, dtype=np.float64)
        primary_wave, primary_flux = kcor.get_primary_sed(kcor.get_filter_primary(name))
        reference = np.interp(wave, primary_wave, primary_flux, left=0., right=0.)
        weight = transmission * wave
        scale = 10. ** (0.4 * (kcor.filter_primary_mag[name]
                              - kcor.filter_zpoff_snpot[name]))
        expected.append(np.trapezoid(reference * weight, wave)
                        / np.trapezoid(weight, wave) * scale)
    np.testing.assert_allclose(photometry.filter_ref_fluxes, expected,
                               rtol=2e-14, atol=0.)


def _kcor_header_file(path, cards):
    hdu = fits.PrimaryHDU()
    for key, value in cards.items():
        hdu.header[key] = value
    hdu.writeto(path)
    return str(path)


def test_resolve_filtlist_reads_filt_cards(tmp_path):
    path = _kcor_header_file(tmp_path / "kcor.fits", {
        "NFILTERS": 2, "FILT001": "LSST-g", "FILT002": "LSST-r"})
    assert resolve_filtlist("rg", KcorReader(path)) == ["LSST-r", "LSST-g"]


def test_resolve_filtlist_reads_a_gzipped_kcor(tmp_path):
    """CFITSIO opens ``<name>.gz`` when given ``<name>``, as SNANA does."""
    path = _kcor_header_file(tmp_path / "kcor.fits", {
        "NFILTERS": 1, "FILT001": "LSST-g"})
    with open(path, "rb") as raw, gzip.open(f"{path}.gz", "wb") as packed:
        packed.write(raw.read())
    os.remove(path)
    assert resolve_filtlist("g", KcorReader(path)) == ["LSST-g"]


@pytest.mark.parametrize("cards, missing", [
    ({"FILT001": "LSST-g"}, "NFILTERS"),
    ({"NFILTERS": 2, "FILT001": "LSST-g"}, "FILT002"),
])
def test_resolve_filtlist_aborts_on_missing_header_card(tmp_path, cards, missing):
    """SNANA read_calib_head aborts on a missing NFILTERS or FILTnnn card
    (sntools_calib.c:300-308)."""
    path = _kcor_header_file(tmp_path / "kcor.fits", cards)
    kcor = KcorReader(path)
    with pytest.raises(SNANAError, match=f"kcor.fits has no {missing} header card"):
        resolve_filtlist("g", kcor)


@pytest.mark.parametrize('bands', [('test-g',), ('test-g', 'test-r'), ('test-r', 'test-g')])
def test_catalogue_and_kcor_share_filter_preparation(tmp_path, bands):
    """The input file format must not change photometry or its derivatives."""
    import jax
    import yaml
    from jax_snlc_core.photometry.filter_system import band_weights_at_z

    observed = np.linspace(3000., 10000., 4001)
    names = ['test-g', 'test-r']
    curves = [np.exp(-.5 * ((observed - center) / 400.) ** 2)
              for center in (4800., 6500.)]
    offsets = [.02, -.01]
    catalogue = {'filters': {}}
    for name, transmission, offset in zip(names, curves, offsets):
        np.savetxt(tmp_path / f'{name}.dat', np.column_stack((observed, transmission)))
        catalogue['filters'][name] = {'path': f'{name}.dat', 'magsys': 'AB',
                                     'magzero': offset, 'lam_unit': 'AA'}
    path = tmp_path / 'filters.yaml'
    path.write_text(yaml.safe_dump(catalogue))
    yaml_reader = KcorReader(path)

    primary = fits.PrimaryHDU()
    primary.header['NFILTERS'] = len(names)
    for i, name in enumerate(names, 1):
        primary.header[f'FILT{i:03d}'] = name
    transmission_hdu = fits.BinTableHDU.from_columns([
        fits.Column(name='Wavelength', format='D', array=observed),
        *[fits.Column(name=name, format='D', array=curve)
          for name, curve in zip(names, curves)]])
    transmission_hdu.header['EXTNAME'] = 'FilterTrans'
    primary_hdu = fits.BinTableHDU.from_columns([
        fits.Column(name='Wavelength', format='D', array=observed),
        fits.Column(name='AB', format='D',
                    array=10.**(-0.4 * 48.6) * 2.99792458e18 / observed**2)])
    primary_hdu.header['EXTNAME'] = 'PrimarySED'
    offsets_hdu = fits.BinTableHDU.from_columns([
        fits.Column(name='Filter Name', format='20A', array=names),
        fits.Column(name='Primary Name', format='20A', array=['AB', 'AB']),
        fits.Column(name='Primary Mag', format='D', array=offsets),
        fits.Column(name='ZPoff(SNpot)', format='D', array=[0., 0.])])
    offsets_hdu.header['EXTNAME'] = 'ZPoff'
    fits_path = tmp_path / 'filters.fits'
    fits.HDUList([primary, transmission_hdu, primary_hdu, offsets_hdu]).writeto(fits_path)
    fits_reader = KcorReader(fits_path)

    wave = jnp.linspace(2000., 11000., 500)
    resources = [load_photometry_resources(reader, wave, list(bands))
                 for reader in (yaml_reader, fits_reader)]
    for field in ('common_wave_grid', 'common_trans_grid', 'filter_ref_fluxes',
                  'filter_ref_fluxes_snana', 'transmissions_stacked'):
        np.testing.assert_array_equal(getattr(resources[0], field), getattr(resources[1], field))
    assert resources[0].common_wave_grid.shape == wave.shape

    sed = 1. + (wave / 5000.)**2
    def flux(z, resource):
        weights = band_weights_at_z(z, resource.common_wave_grid,
                                    resource.common_trans_grid, wave)
        return weights @ sed

    for resource in resources:
        evaluate = lambda z: flux(z, resource)
        derivatives = jax.jit(jax.vmap(lambda z: (
            evaluate(z), jax.jacfwd(evaluate)(z), jax.jacfwd(jax.jacfwd(evaluate))(z))))
        actual = derivatives(jnp.array([0., .1, .4]))
        if resource is resources[0]:
            reference = actual
        else:
            for a, b in zip(actual, reference):
                np.testing.assert_array_equal(a, b)
        # Independent dense-grid quadrature of the redshifted spectrum.
        for i, z in enumerate((0., .1, .4)):
            expected = []
            for band in bands:
                trans = curves[names.index(band)]
                dense_sed = (1. + (observed / (1. + z) / 5000.)**2) / (1. + z)
                expected.append(np.trapezoid(dense_sed * trans * observed, observed)
                                / np.trapezoid(trans * observed, observed))
            np.testing.assert_allclose(actual[0][i], expected, rtol=1e-4)
