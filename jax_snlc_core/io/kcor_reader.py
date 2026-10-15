"""Shared reader for calibration FITS files, SNANA inputs and BayeSN catalogues."""

import gzip
import logging
import numpy as np
import jax.numpy as jnp
from astropy.io import fits
from pathlib import Path
import os
import yaml
from typing import Dict, List, Tuple

from jax_snlc_core.utils.error_handling import SNANAError

from jax_snlc_core.io.fits_path import open_text, resolve_fits_path
from jax_snlc_core.io.kcor_input import read_kcor_input
from jax_snlc_core.photometry.resources import filter_mean_wavelength
from jax_snlc_core.photometry.spectrograph import (
    read_spectrograph, read_spectrograph_fits, synthetic_filter_transmission,
)
from jax_snlc_core.utils import snana_log as slog

logger = logging.getLogger(__name__)


def _native_byteorder(arr):
    """Ensure native byte order for JAX consumption.

    FITS columns come back from astropy with the file's original byte order
    (typically big-endian, e.g. dtype '>f4'). Newer JAX versions reject
    non-native byte orders at ``jnp.array(...)`` with ``TypeError: not a
    valid JAX array``. Return a numpy view in native order with values
    preserved.
    """
    arr = np.asarray(arr)
    if arr.dtype.byteorder == '>':
        arr = arr.astype(arr.dtype.newbyteorder('='))
    return arr


class KcorReader:
    """Load filter transmissions, primary spectra and calibration metadata."""

    def __init__(self, kcor_file_path: str, leakage_cut: float = 0.0):
        """Load filter transmission curves, primary reference SEDs and
        per-filter calibration metadata from ``kcor_file_path`` (a kcor FITS
        file, SNANA calibration input or BayeSN YAML catalogue, which may contain
        environment variables). ``leakage_cut`` is
        LEAKAGE_CUT; 0, SNANA's default, keeps every filter as stored."""
        source = Path(os.path.expandvars(kcor_file_path)).expanduser()
        self.kcor_file_path = Path(resolve_fits_path(source, required=True))

        self.filter_data: Dict[str, Dict[str, jnp.ndarray]] = {}
        self.filter_names: List[str] = []
        # SYN_FILTER_SPECTROGRAPH bands: name -> (lammin, lammax) top-hat range.
        self.syn_filters: Dict[str, Tuple[float, float]] = {}
        self.spectrograph = None
        self.primary_seds: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.filter_primary: Dict[str, str] = {}
        # ZPoff HDU 'Primary Mag' and 'ZPoff(SNpot)' columns
        self.filter_primary_mag: Dict[str, float] = {}
        self.filter_zpoff_snpot: Dict[str, float] = {}
        # 'SURVEY=X' in the FILT{nnn} card comment; UNDEFINED if absent
        self.filter_survey: Dict[str, str] = {}

        suffixes = self.kcor_file_path.suffixes
        if suffixes and suffixes[-1].lower() == '.gz':
            suffixes = suffixes[:-1]
        suffix = suffixes[-1].lower() if suffixes else ''
        if suffix.lower() in ('.yaml', '.yml'):
            self.kcor_file_path = self.kcor_file_path.resolve()
            self._load_catalogue()
        elif self._is_fits(self.kcor_file_path):
            self._load_kcor_file()
        else:
            self.kcor_file_path = self.kcor_file_path.resolve()
            self._load_filter_definitions(read_kcor_input(self.kcor_file_path))
        if leakage_cut > 0.0:
            self._cut_leakage(leakage_cut)

    @staticmethod
    def _is_fits(path):
        opener = gzip.open if path.suffix.lower() == '.gz' else open
        with opener(path, 'rb') as stream:
            return stream.read(8) == b'SIMPLE  '

    def _load_catalogue(self):
        """Read a BayeSN filter catalogue, with its native magnitude systems.

        Catalogue magnitude updates and calibration corrections are applied;
        optional wavelength-systematic shifts are not. Relative resource
        paths are resolved against the catalogue, never the working directory.
        """
        path = self.kcor_file_path
        with open_text(path) as stream:
            catalogue = yaml.safe_load(stream)
        self._load_filter_definitions(catalogue)

    @staticmethod
    def _read_curve(source):
        """Read a standard spectrum or transmission curve without resampling."""
        if any(str(source).lower().endswith(ext)
               for ext in ('.fits', '.fit', '.fits.gz', '.fit.gz')):
            with fits.open(source) as hdus:
                wave = _native_byteorder(hdus[1].data['WAVELENGTH'])
                flux = _native_byteorder(hdus[1].data['FLUX'])
        else:
            wave, flux = np.loadtxt(source, usecols=(0, 1), unpack=True)
        order = np.argsort(wave, kind='stable')
        wave, flux = wave[order], flux[order]
        # Repeated wavelengths use the last value, as in np.interp.
        keep = np.r_[np.diff(wave) > 0, True]
        wave, flux = wave[keep], flux[keep]
        if (len(wave) < 2 or not np.all(np.isfinite(wave))
                or not np.all(np.isfinite(flux)) or np.any(wave <= 0)):
            raise SNANAError(f'{source}: expected finite values at positive wavelengths')
        return wave, flux

    def _load_filter_definitions(self, catalogue):
        """Load both text calibration formats into the same filter arrays."""
        self.filter_names = list(catalogue['filters'])
        if not self.filter_names:
            raise SNANAError(f'{self.kcor_file_path}: no filters defined')
        self.header = {'NFILTERS': len(self.filter_names)}
        self.filter_name_width = max(len(name) for name in self.filter_names)
        grid = catalogue.get('wavelength_grid')
        filter_bounds = []
        if 'spectrograph' in catalogue:
            self.spectrograph = read_spectrograph(**catalogue['spectrograph'])

        def resource(spec, kind):
            root = catalogue.get(kind + '_root', '')
            return self.kcor_file_path.parent / Path(
                os.path.expandvars(str(Path(root) / spec['path']))).expanduser()

        for name, spec in catalogue.get('standards', {}).items():
            self.primary_seds[name] = self._read_curve(resource(spec, 'standards'))
        for i, (name, spec) in enumerate(catalogue['filters'].items(), 1):
            if 'synthetic_range' in spec:
                bounds = spec['synthetic_range']
                wave, transmission = synthetic_filter_transmission(self.spectrograph, *bounds)
            else:
                wave, transmission = self._read_curve(resource(spec, 'filters'))
            unit = spec.get('lam_unit', 'AA').lower()
            factors = {'aa': 1., 'nm': 10., 'micron': 1.e4}
            if unit not in factors:
                raise SNANAError(f"Unknown wavelength unit {unit!r} for {name}")
            wave = wave * factors[unit]
            if 'out_of_band' in spec:
                wave, transmission = self._add_out_of_band(wave, transmission, spec['out_of_band'])
            if not np.any(transmission > 0):
                raise SNANAError(f'Filter {name} needs a positive transmission peak')
            cut = catalogue.get('leakage_cut', 0.)
            if cut > 0.:
                kept = transmission >= cut * transmission.max()
                wave, transmission = wave[kept], transmission[kept] / transmission.max()
                if len(wave) < 2:
                    raise SNANAError(f'LEAKAGE_CUT leaves fewer than two wavelengths for {name}')
            wave = wave + spec.get('wavelength_shift', 0.)
            if 'synthetic_range' in spec:
                self.syn_filters[name] = (float(wave[0]), float(wave[-1]))
            system = spec.get('filter_system', 'COUNT')
            if system == 'ENERGY':
                transmission = transmission * 1000. / wave
            elif system != 'COUNT':
                raise SNANAError(f'Unknown filter system {system!r} for {name}')
            filter_bounds.append((wave[0], wave[-1]))
            if grid is not None:
                if wave[0] < grid[0] or wave[-1] > grid[-1]:
                    raise SNANAError(f'Filter {name} lies outside LAMBDA_RANGE')
                transmission = np.interp(grid, wave, transmission, left=0., right=0.)
                wave = grid
            self.filter_data[name] = {'wavelength': jnp.asarray(wave),
                                      'transmission': jnp.asarray(transmission)}
            primary = '->'.join('AB' if p.lower() == 'ab' else p
                                for p in spec['magsys'].split('->'))
            self.filter_primary[name] = primary
            self.filter_primary_mag[name] = (float(spec['magzero'])
                + float(spec.get('magupdate', 0.)) + float(spec.get('magcal', 0.)))
            self.filter_zpoff_snpot[name] = float(spec.get('zpoff_snpot', 0.))
            self.filter_survey[name] = spec.get('survey', 'UNDEFINED')
            self.header[f'FILT{i:03d}'] = name
        wave = np.unique(np.concatenate([
            np.asarray(values['wavelength']) for values in self.filter_data.values()]))
        if 'AB' not in self.primary_seds:
            self.primary_seds['AB'] = (wave, 10.**(-0.4 * 48.6) * 2.99792458e18 / wave**2)
        if grid is not None:
            # SNANA rebins standards to multiples of the calibration step,
            # over the combined filter range, before writing PrimarySED.
            step = grid[1] - grid[0]
            lo = min(bounds[0] for bounds in filter_bounds)
            hi = max(bounds[1] for bounds in filter_bounds)
            if self.spectrograph is not None:
                observed = ~np.asarray(self.spectrograph.islam_extend)
                lo = min(lo, np.asarray(self.spectrograph.lam_min)[observed].min())
                hi = max(hi, np.asarray(self.spectrograph.lam_max)[observed].max())
            standard_grid = step * np.arange(np.ceil(lo / step), np.ceil(hi / step) + 1)
            for name, (waves, flux) in self.primary_seds.items():
                if waves[0] > standard_grid[0] or waves[-1] < standard_grid[-1]:
                    raise SNANAError(f'Primary {name} does not cover the filter wavelength range')
                rebinned = np.interp(standard_grid, waves, flux)
                self.primary_seds[name] = (grid, np.interp(grid, standard_grid, rebinned,
                                                         left=0., right=0.))
        self._convert_magnitude_systems()

    @staticmethod
    def _add_out_of_band(wave, transmission, specification):
        """Extend a curve with constant leakage relative to its peak."""
        lo, hi, ratio = specification
        if ratio == 0.:
            return wave, transmission
        blue_step, red_step = wave[1] - wave[0], wave[-1] - wave[-2]
        n_blue = max(int((wave[0] - lo) / blue_step), 0)
        n_red = max(int((hi - wave[-1]) / red_step), 0)
        extended = np.r_[wave[0] - blue_step * np.arange(n_blue, 0, -1),
                         wave, wave[-1] + red_step * np.arange(1, n_red + 1)]
        leakage = ratio * transmission.max()
        response = np.full(len(extended), leakage)
        response[n_blue:n_blue + len(wave)] += transmission
        return extended, response

    def _convert_magnitude_systems(self):
        """Translate primary magnitudes for MAGSYSTEM input->output entries."""
        for name, system in self.filter_primary.items():
            parts = system.split('->')
            if len(parts) not in (1, 2) or any(p not in self.primary_seds for p in parts):
                raise SNANAError(f'Filter {name} names missing or invalid standard {system!r}')
            if len(parts) == 1:
                continue
            target = parts[1]
            wave = np.asarray(self.filter_data[name]['wavelength'])
            trans = np.asarray(self.filter_data[name]['transmission'])
            step = np.median(np.diff(wave))
            # These are SNANA's primarymag_zp weights: dnu/(h*nu).
            weights = trans * (1. / (wave - .5 * step) - 1. / (wave + .5 * step)) * wave
            integrals = []
            for primary in parts:
                pwave, pflux = self.primary_seds[primary]
                fnu = np.interp(wave, pwave, pflux, left=0., right=0.) * wave**2
                integrals.append(np.sum(weights * fnu))
            if min(integrals) <= 0.:
                raise SNANAError(f'Primary standards have no flux in filter {name}')
            self.filter_primary_mag[name] += 2.5 * np.log10(integrals[0] / integrals[1])
            self.filter_primary[name] = target

    def _cut_leakage(self, leakage_cut):
        """Out-of-band leakage removed as kcor.exe's LEAKAGE_CUT does
        (cutOOBTrans_filter, kcor.c:2513): a wavelength bin whose
        transmission is below ``leakage_cut`` times the filter's peak is
        dropped, zeroed here since every filter shares the file's grid, so
        the filter's range becomes the span of the bins kept."""
        for name in self.filter_names:
            wave = np.asarray(self.filter_data[name]['wavelength'])
            trans = np.asarray(self.filter_data[name]['transmission'])
            kept = trans >= leakage_cut * trans.max()
            slog.line(logger, f"Remove {name} OOB leakage: WAVE-RANGE = "
                      f"[{wave[trans > 0].min():.1f},{wave[trans > 0].max():.1f}] -> "
                      f"[{wave[kept].min():.1f},{wave[kept].max():.1f}]")
            self.filter_data[name]['transmission'] = jnp.array(
                np.where(kept, trans, 0.0))

    @staticmethod
    def _fix_none_column_names(hdu):
        """Patch any None column names to placeholder strings.

        Some older kcor FITS files (e.g. DES 2013) have missing TTYPE keywords,
        which astropy reads as name=None.  numpy rejects None as a structured-
        array field name, so ``hdu.data`` would crash.  This assigns
        ``'UNNAMED_<idx>'`` to any such column before data is accessed.
        """
        for i, col in enumerate(hdu.columns):
            if col.name is None:
                hdu.columns[i].name = f'UNNAMED_{i}'

    def _load_kcor_file(self):
        """Load filter transmission curves and calibration data from kcor FITS file."""
        logger.info(f"Loading kcor file: {self.kcor_file_path}")

        with fits.open(self.kcor_file_path) as hdul:
            self.header = hdul[0].header.copy()
            self._extract_filter_surveys(self.header)
            self.spectrograph = read_spectrograph_fits(hdul, self.kcor_file_path)
            for hdu in hdul[1:]:
                self._fix_none_column_names(hdu)
                if hdu.name == 'FilterTrans':
                    self._extract_filter_data(hdu)
                elif hdu.name == 'PrimarySED':
                    self._extract_primary_seds(hdu)
                elif hdu.name == 'ZPoff':
                    self._extract_zpoff(hdu)
                elif hdu.name == 'SYN_FILTER_SPECTROGRAPH':
                    self._extract_syn_filters(hdu)

    def _extract_filter_surveys(self, header):
        """Per-filter survey from the FILT{nnn} card comments."""
        for card in header.cards:
            if not card.keyword.startswith('FILT') or not card.keyword[4:].isdigit():
                continue
            name = str(card.value).strip()
            comment = str(card.comment)
            survey = (comment.split('=', 1)[1].strip().strip("'\"")
                      if '=' in comment else 'UNDEFINED')
            self.filter_survey[name] = survey or 'UNDEFINED'

    def _extract_filter_data(self, hdu):
        """Extract filter transmission data from HDU."""
        col_names = [col.name for col in hdu.columns]
        logger.debug(f"Filter HDU columns: {col_names}")

        wave_col = None
        for col in col_names:
            if col is not None and any(w in col.lower() for w in ['lambda', 'wavelength', 'wave']):
                wave_col = col
                break

        if wave_col is None:
            raise SNANAError(f"No wavelength column in the FilterTrans HDU of "
                             f"{self.kcor_file_path}: {col_names}",
                             function_name="_extract_filter_data")

        data = hdu.data
        wavelengths = data[wave_col]
        logger.debug(f"Found wavelength column: {wave_col}")

        # Every non-wavelength column is a filter transmission
        for col_name in col_names:
            if col_name == wave_col:
                continue
            if col_name is None or col_name.startswith('UNNAMED_'):
                continue
            self.filter_data[col_name] = {
                'wavelength': jnp.array(_native_byteorder(wavelengths)),
                'transmission': jnp.array(_native_byteorder(data[col_name]))
            }
            self.filter_names.append(col_name)
            logger.debug(f"Loaded filter: {col_name}")
        # PHOT.FITS BAND width, before any GENFILTERS restriction
        self.filter_name_width = max(len(name) for name in self.filter_names)

    def _extract_syn_filters(self, hdu):
        """Record which FilterTrans bands are synthetic spectrograph filters.

        kcor.exe writes each synthetic band into FilterTrans like any filter, so
        the transmission is already registered by ``_extract_filter_data``; this
        only tags the band with its (lammin, lammax) range.
        """
        for row in hdu.data:
            name = row['NAME']
            name = (name.decode() if isinstance(name, bytes) else str(name)).strip()
            self.syn_filters[name] = (float(row['LAMMIN']), float(row['LAMMAX']))

    def _extract_primary_seds(self, hdu):
        """Extract all primary reference SEDs from the PrimarySED HDU.

        Stores each primary as ``self.primary_seds[name] = (waves, flux)``.
        The column names (e.g. 'AB', 'VEGA', 'BD17') serve as the lookup keys
        and match the 'Primary Name' column in the ZPoff HDU.
        """
        data = hdu.data
        col_names = [col.name for col in hdu.columns]

        wave_col = next(
            (c for c in col_names
             if c is not None and ('wave' in c.lower() or 'lambda' in c.lower())),
            None,
        )
        if wave_col is None:
            raise SNANAError(f"No wavelength column in the PrimarySED HDU of "
                             f"{self.kcor_file_path}: {col_names}",
                             function_name="_extract_primary_seds")

        waves = np.array(data[wave_col], dtype=np.float64)

        for col_name in col_names:
            if col_name == wave_col:
                continue
            if col_name is None or col_name.startswith('UNNAMED_'):
                continue
            flux = np.array(data[col_name], dtype=np.float64)
            self.primary_seds[col_name] = (waves, flux)
            logger.debug(f"  Loaded PrimarySED '{col_name}': {len(waves)} points, "
                         f"{waves.min():.0f}-{waves.max():.0f} A")

    def _extract_zpoff(self, hdu):
        """Extract per-filter primary reference mapping from the ZPoff HDU.

        Populates ``self.filter_primary`` mapping filter short names (e.g.
        'DES-g') to primary short names (e.g. 'AB'), plus the per-filter
        'Primary Mag' and 'ZPoff(SNpot)' calibration values.

        SNANA reads the same table positionally for every filter and treats
        a FITS read failure as fatal (read_calib_zpoff,
        sntools_calib.c:563-611), so a ZPoff table without the required
        columns or rows is a malformed calibration file.
        """
        col_names = [col.name for col in hdu.columns]
        required = ('Filter Name', 'Primary Name', 'Primary Mag',
                    'ZPoff(SNpot)')
        missing = [c for c in required if c not in col_names]
        if missing:
            raise SNANAError(
                f"KCOR ZPoff table is missing required column(s) {', '.join(missing)} "
                f"(has {col_names}); SNANA reads these positionally for every filter.",
                function_name="_extract_zpoff")

        for row in hdu.data:
            fname = str(row['Filter Name']).strip()
            pname = str(row['Primary Name']).strip()
            if not fname or not pname:
                raise SNANAError(
                    f"KCOR ZPoff table row has empty Filter Name/Primary Name ('{fname}', "
                    f"'{pname}'); SNANA aborts on an unrecognised PRIMARY_NAME.",
                    function_name="_extract_zpoff")
            self.filter_primary[fname] = pname
            self.filter_primary_mag[fname] = float(row['Primary Mag'])
            self.filter_zpoff_snpot[fname] = float(row['ZPoff(SNpot)'])

        slog.line(logger, f"ZPoff: {len(self.filter_primary)} filters, "
                  f"primaries used: {sorted(set(self.filter_primary.values()))}")
        if any(v != 0.0 for v in self.filter_primary_mag.values()):
            logger.debug(f"  Primary Mag (non-zero): "
                         f"{ {k: v for k, v in self.filter_primary_mag.items() if v != 0.0} }")
        if any(v != 0.0 for v in self.filter_zpoff_snpot.values()):
            logger.debug(f"  ZPoff(SNpot) (non-zero): "
                         f"{ {k: v for k, v in self.filter_zpoff_snpot.items() if v != 0.0} }")

    def get_primary_sed(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """The (wavelengths, flux_density) numpy arrays for the primary
        reference SED ``name`` (e.g. 'AB', 'VEGA', 'BD17'), which must match
        a column name in the PrimarySED HDU and the 'Primary Name' column in
        ZPoff."""
        if name not in self.primary_seds:
            raise SNANAError(
                f"Primary SED '{name}' not found. "
                f"Available: {sorted(self.primary_seds)}",
                function_name="get_primary_sed")
        return self.primary_seds[name]

    def get_filter_primary(self, filter_name: str) -> str:
        """The primary reference name (e.g. 'AB', 'VEGA') for ``filter_name``
        (e.g. 'DES-g', 'LSST-r', 'Bessell-V')."""
        if filter_name not in self.filter_primary:
            raise SNANAError(
                f"No primary mapping for filter '{filter_name}'. "
                f"Known filters: {sorted(self.filter_primary)}",
                function_name="get_filter_primary")
        return self.filter_primary[filter_name]

    def get_filter_names(self, *, include_duplicates=False) -> List[str]:
        """Available bands; include_duplicates adds the starred shifted curves."""
        return [name for name in self.filter_names
                if include_duplicates or not name.startswith('*')]

    def get_filter_transmission(self, filter_name: str) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """The (wavelength, transmission) arrays for ``filter_name`` (e.g.
        'LSST-u', 'LSST-g')."""
        if filter_name not in self.filter_data:
            available = list(self.filter_data.keys())
            raise SNANAError(f"Filter {filter_name} not found. Available: {available}",
                             function_name="get_filter_transmission")

        data = self.filter_data[filter_name]
        return data['wavelength'], data['transmission']

    def filter_lamavg(self, filter_names: List[str]) -> Dict[str, float]:
        """FILTOBS_LAMAVG of each of ``filter_names``, unshifted."""
        return {name: filter_mean_wavelength(*self.get_filter_transmission(name))
                for name in filter_names}
