"""SFD98 dust map reader: reads Schlegel, Finkbeiner & Davis (1998) E(B-V)
maps from FITS files stored at
$SNDATA_ROOT/MWDUST/SFD_dust_4096_{ngp,sgp}.fits.

The maps use Lambert azimuthal equal-area projection in galactic coordinates.
Projection matches SNANA MWgaldust.c lambert_lb2xy / lambert_lb2fpix exactly.

Coordinate conversion uses SNANA's exact ICRS->Galactic rotation matrix
(from slaEqgal in sntools.c:4406-4414) for sub-micromag agreement.
"""

from pathlib import Path

import numpy as np
import jax.numpy as jnp

from astropy.io import fits

from jax_snlc_core.io.fits_path import resolve_fits_path
from jax_snlc_core.utils import sndata_root
from jax_snlc_core.utils.error_handling import SNANAError

# slaEqgal, sntools.c:4406-4414
_RMAT_EQ2GAL = np.array([
    [-0.054875539726, -0.873437108010, -0.483834985808],
    [ 0.494109453312, -0.444829589425,  0.746982251810],
    [-0.867666135858, -0.198076386122,  0.455983795705]
])


class SFDDustMap:
    """Read SFD98 E(B-V) dust maps and look up values by RA/DEC."""

    def __init__(self):
        self._map_ngp = None
        self._map_sgp = None
        # device copies, made only when a traced draw asks for them
        self._map_ngp_dev = None
        self._map_sgp_dev = None
        self._header_ngp = None
        self._header_sgp = None

    def _load_maps(self):
        """Lazy-load both NGP and SGP maps."""
        if self._map_ngp is not None:
            return

        mwdust_dir = sndata_root() / 'MWDUST'

        ngp_path = Path(resolve_fits_path(mwdust_dir / 'SFD_dust_4096_ngp.fits'))
        sgp_path = Path(resolve_fits_path(mwdust_dir / 'SFD_dust_4096_sgp.fits'))

        if not ngp_path.exists():
            raise SNANAError(f"SFD NGP map not found: {ngp_path}",
                             function_name="_load_maps")
        if not sgp_path.exists():
            raise SNANAError(f"SFD SGP map not found: {sgp_path}",
                             function_name="_load_maps")

        with fits.open(ngp_path) as hdul:
            self._map_ngp = hdul[0].data.astype(np.float32)
            self._header_ngp = dict(hdul[0].header)

        with fits.open(sgp_path) as hdul:
            self._map_sgp = hdul[0].data.astype(np.float32)
            self._header_sgp = dict(hdul[0].header)

    @staticmethod
    def _radec_to_galactic(ra_deg, dec_deg):
        """Convert RA/DEC (degrees) to galactic l,b (degrees).

        Uses SNANA's exact rotation matrix for sub-micromag agreement.
        Works for both scalars and arrays.
        """
        ra_rad = np.radians(ra_deg)
        dec_rad = np.radians(dec_deg)
        cos_dec = np.cos(dec_rad)
        v_eq = np.stack([cos_dec * np.cos(ra_rad),
                         cos_dec * np.sin(ra_rad),
                         np.sin(dec_rad)], axis=-1)
        v_gal = v_eq @ _RMAT_EQ2GAL.T
        l_deg = np.degrees(np.arctan2(v_gal[..., 1], v_gal[..., 0])) % 360.0
        b_deg = np.degrees(np.arcsin(np.clip(v_gal[..., 2], -1.0, 1.0)))
        return l_deg, b_deg

    @staticmethod
    def _lambert_lb2fpix(l_deg, b_deg, nsgp, scale, crpix1, crpix2, crval1, crval2):
        """Convert galactic (l,b) to 0-indexed pixel coords.

        Matches SNANA MWgaldust.c lambert_lb2fpix exactly.
        Works for both scalars and arrays.
        """
        dradeg = 180.0 / np.pi
        rho = np.sqrt(1.0 - nsgp * np.sin(b_deg / dradeg))
        xr = rho * np.cos(l_deg / dradeg) * scale
        yr = -nsgp * rho * np.sin(l_deg / dradeg) * scale
        px = xr + crpix1 - crval1 - 1.0
        py = yr + crpix2 - crval2 - 1.0
        return px, py

    @staticmethod
    def _bilinear_interp(data, px, py):
        """Bilinear interpolation on 2D array. Works for arrays of (px, py)."""
        ny, nx = data.shape
        ix = np.clip(np.floor(px).astype(int), 0, nx - 2)
        iy = np.clip(np.floor(py).astype(int), 0, ny - 2)
        fx = px - ix
        fy = py - iy
        return ((1 - fx) * (1 - fy) * data[iy, ix] +
                fx * (1 - fy) * data[iy, ix + 1] +
                (1 - fx) * fy * data[iy + 1, ix] +
                fx * fy * data[iy + 1, ix + 1])

    def device_data(self):
        """Prepare images and projection coefficients for device lookups."""
        self._load_maps()
        if self._map_ngp_dev is None:
            self._map_ngp_dev = jnp.asarray(self._map_ngp)
            self._map_sgp_dev = jnp.asarray(self._map_sgp)
        keys = ('LAM_NSGP', 'LAM_SCAL', 'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2')
        return tuple((image, {key: header[key] for key in keys})
                     for image, header in ((self._map_ngp_dev, self._header_ngp),
                                           (self._map_sgp_dev, self._header_sgp)))

    def ebv_on_device(self, ra_deg, dec_deg):
        """Look up E(B-V) at per-event coordinates using prepared device maps."""
        return ebv_on_device(self.device_data(), ra_deg, dec_deg)

    def get_ebv(self, ra_deg: float, dec_deg: float) -> float:
        """Look up E(B-V) from SFD98 maps for a given RA/DEC position."""
        return float(self.get_ebv_batch(np.array([ra_deg]), np.array([dec_deg]))[0])

    def get_ebv_batch(self, ra_arr, dec_arr):
        """Look up E(B-V) for arrays of RA/DEC positions.

        Fully vectorised: processes all coordinates in bulk via numpy.
        """
        self._load_maps()
        ra_arr = np.atleast_1d(np.asarray(ra_arr, dtype=np.float64))
        dec_arr = np.atleast_1d(np.asarray(dec_arr, dtype=np.float64))

        l, b = self._radec_to_galactic(ra_arr, dec_arr)
        result = np.zeros(len(ra_arr), dtype=np.float64)

        for mask, dust_map, hdr in [(b >= 0, self._map_ngp, self._header_ngp),
                                     (b < 0, self._map_sgp, self._header_sgp)]:
            if not np.any(mask):
                continue
            px, py = self._lambert_lb2fpix(
                l[mask], b[mask],
                hdr['LAM_NSGP'], hdr['LAM_SCAL'],
                hdr['CRPIX1'], hdr['CRPIX2'],
                hdr['CRVAL1'], hdr['CRVAL2'],
            )
            result[mask] = self._bilinear_interp(dust_map, px, py)

        return result


def ebv_on_device(maps, ra_deg, dec_deg):
    """Evaluate prepared SFD images without capturing their reader."""
    ra_r = jnp.radians(jnp.asarray(ra_deg))
    dec_r = jnp.radians(jnp.asarray(dec_deg))
    cos_dec = jnp.cos(dec_r)
    v_eq = jnp.stack([cos_dec * jnp.cos(ra_r),
                      cos_dec * jnp.sin(ra_r),
                      jnp.sin(dec_r)], axis=-1)
    v_gal = v_eq @ jnp.asarray(_RMAT_EQ2GAL).T
    l_deg = jnp.degrees(jnp.arctan2(v_gal[..., 1], v_gal[..., 0])) % 360.0
    b_deg = jnp.degrees(jnp.arcsin(jnp.clip(v_gal[..., 2], -1.0, 1.0)))

    north = b_deg >= 0
    out = None
    for dev_map, hdr in maps:
        dradeg = 180.0 / jnp.pi
        nsgp, scale = hdr['LAM_NSGP'], hdr['LAM_SCAL']
        rho = jnp.sqrt(1.0 - nsgp * jnp.sin(b_deg / dradeg))
        xr = rho * jnp.cos(l_deg / dradeg) * scale
        yr = -nsgp * rho * jnp.sin(l_deg / dradeg) * scale
        px = xr + hdr['CRPIX1'] - hdr['CRVAL1'] - 1.0
        py = yr + hdr['CRPIX2'] - hdr['CRVAL2'] - 1.0

        ny, nx = dev_map.shape
        ix = jnp.clip(jnp.floor(px).astype(jnp.int32), 0, nx - 2)
        iy = jnp.clip(jnp.floor(py).astype(jnp.int32), 0, ny - 2)
        fx, fy = px - ix, py - iy
        val = ((1 - fx) * (1 - fy) * dev_map[iy, ix] +
               fx * (1 - fy) * dev_map[iy, ix + 1] +
               (1 - fx) * fy * dev_map[iy + 1, ix] +
               fx * fy * dev_map[iy + 1, ix + 1])
        out = val if out is None else jnp.where(north, out, val)
    return out
