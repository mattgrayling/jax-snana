"""Unit tests for the SFD98 dust-map lookup, on small synthetic images and
on the bundled stand-in maps."""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from jax_snlc_core.dust import SFDDustMap, ebv_on_device
from jax_snlc_core.logging import SNANAError

_BUNDLED_SNDATA_ROOT = Path(__file__).parents[1] / 'fixtures' / 'data' / 'sndata_root'

# with LAM_NSGP = +-1 the pole projects onto pixel CRPIX - CRVAL - 1 (0-based)
_HEADER = {'LAM_SCAL': 2.0, 'CRPIX1': 3.0, 'CRPIX2': 3.0, 'CRVAL1': 0.0, 'CRVAL2': 0.0}
_NORTH = np.arange(25, dtype=np.float32).reshape(5, 5)
_SOUTH = 100.0 + _NORTH
_MAPS = ((_NORTH, {**_HEADER, 'LAM_NSGP': 1}), (_SOUTH, {**_HEADER, 'LAM_NSGP': -1}))


@pytest.fixture
def sfd_directory(tmp_path):
    """Use a private SNDATA_ROOT and restore the caller's environment on teardown."""
    directory = tmp_path / 'MWDUST'
    directory.mkdir()
    previous = os.environ.get('SNDATA_ROOT')
    os.environ['SNDATA_ROOT'] = str(tmp_path)
    try:
        yield directory
    finally:
        if previous is None:
            os.environ.pop('SNDATA_ROOT', None)
        else:
            os.environ['SNDATA_ROOT'] = previous


def test_galactic_poles_read_their_own_image_at_the_projection_centre():
    """The north galactic pole reads the north image and the south pole the
    south one, each at the pixel the Lambert projection puts the pole on
    (lambert_lb2fpix, MWgaldust.c)."""
    ebv = ebv_on_device(_MAPS, np.array([192.85948, 12.85948]),
                        np.array([27.12825, -27.12825]))
    np.testing.assert_allclose(ebv, [_NORTH[2, 2], _SOUTH[2, 2]], atol=1e-3)


def test_bundled_maps_match_snana_off_the_poles(sfd_directory):
    """Check SNANA's two DUSTY SIMLIB reference values, including interpolation
    away from the poles, using real files under SNDATA_ROOT."""
    for pole in ('ngp', 'sgp'):
        filename = f'SFD_dust_4096_{pole}.fits.gz'
        shutil.copyfile(_BUNDLED_SNDATA_ROOT / 'MWDUST' / filename, sfd_directory / filename)
    dust = SFDDustMap()
    np.testing.assert_allclose(dust.get_ebv_batch([10., 180.], [-30., -45.]),
                               [1.50000e-02, 4.70372e-01], rtol=3e-6)


def test_map_without_projection_cards_aborts(sfd_directory):
    """A map without its Lambert projection must not silently read a corner pixel."""
    with fits.open(_BUNDLED_SNDATA_ROOT / 'MWDUST' / 'SFD_dust_4096_ngp.fits.gz') as hdul:
        header = hdul[0].header.copy()
        del header['CTYPE1'], header['CTYPE2']
        fits.PrimaryHDU(hdul[0].data, header).writeto(sfd_directory / 'SFD_dust_4096_ngp.fits')
    with pytest.raises(SNANAError, match='not the LAMBERT--X/LAMBERT--Y projection'):
        SFDDustMap().get_ebv_batch([10.], [-30.])
