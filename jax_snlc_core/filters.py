"""Filters: the calibration (kcor) file, the filters prepared for a model's
wavelength grid, and integrating an SED through them.

- The FLUXCAL zero points.
- :class:`KcorReader` reads the calibration file in its FITS, SNANA-input
  and BayeSN forms.
- :func:`filter_band` is the rule from a filter name to its band, and
  :func:`filter_values` puts per-band settings in filter order.
- :func:`compute_single_restframe_filter` builds band weights,
  :func:`band_window_width` sizes the wavelength window they run on and
  :func:`fluxcal_from_sed` integrates an SED through them.
- :func:`load_photometry_resources` prepares the active filters: their
  transmissions, primary-standard integrals with SNANA's zero-point offsets,
  and mean wavelengths.
- A model's MAGSHIFT/WAVESHIFT calibration shifts, read and matched to the
  active filters.
"""

import gzip
import logging
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Dict, List, Mapping, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from astropy.io import fits
from jax import lax

from jax_snlc_core import logging as log
from jax_snlc_core.interpolation import compute_catmullrom_slopes, hermite_interp_1d
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import (
    is_ignorefile, open_text, resolve_fits_path,
    read_two_column_file)
from jax_snlc_core.spectrograph import (parse_spectrograph_filename, read_spectrograph,
                                        read_spectrograph_fits)
from jax_snlc_core.utils import HC_ERG_A, LIGHT_A, float_dtype

logger = logging.getLogger(__name__)

# Default zero point of simulated and Python-input FLUXCAL photometry.
ZEROPOINT_FLUXCAL_SNANA = 31.4

# pre-nJy FLUXCAL zero point (sndata.h:46)
ZEROPOINT_FLUXCAL_SNANA_ORIG = 27.5


# Calibration-input keys, with the number of values each takes, that describe
# generated tables or diagnostic output rather than the filters an SED is
# integrated through.
_KCOR_INPUT_TABLE_KEYS = {
    'OUTFILE': 1, 'KCOR': 3, 'REDSHIFT_RANGE': 2, 'REDSHIFT_BINSIZE': 1,
    'TREST_RANGE': 2, 'AV_RANGE': 2, 'AV_BINSIZE': 1, 'AV_OPTION': 1,
    'RV': 1, 'RV_MWCOLORLAW': 1, 'OPT_MWCOLORLAW': 1,
    'SN_TYPE': 1, 'SN_MAGOFF': 1, 'TREF_EXPLODE': 1,
    'PLOTFLAG_SN': 1, 'PLOTFLAG_PRIMARY': 1, 'DUMP_SNMAG': 1,
    'SN_SED_FUDGE': 1, 'NLAMBIN_FT': 1,
}
_MAGNITUDE_TERM = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'


def magnitude_sum(value):
    """A primary magnitude may be written as a sum, e.g. 9.824+.011-.002."""
    terms = re.findall(_MAGNITUDE_TERM, value)
    if (not terms or ''.join(terms) != value
            or any(term[0] not in '+-' for term in terms[1:])):
        raise SNANAError(f'Invalid calibration magnitude {value!r}')
    return sum(float(term) for term in terms)


def calibration_path(value, directory, subdirectory, *, required=True):
    """Resolve explicit paths, then SNANA data-tree paths (including gzip)."""
    expanded = os.path.expandvars(value)
    if '$' in expanded:
        raise SNANAError(f'Unresolved environment variable in calibration path {value!r}')
    path = Path(expanded).expanduser()
    candidates = [path] if path.is_absolute() else [directory / path]
    root = os.environ.get('SNDATA_ROOT')
    if root and not path.is_absolute():
        candidates.append(Path(root) / subdirectory / path)
    for candidate in candidates:
        source = Path(resolve_fits_path(candidate))
        if source.is_file():
            return source.resolve()
    if required:
        raise SNANAError(f'Calibration resource not found: {value!r}; searched '
                         + ', '.join(str(p) for p in candidates))
    return None


def read_kcor_input(path):
    """Read a SNANA calibration input into its filter and primary-standard
    definitions; :class:`KcorReader` loads the spectra and transmissions.

    MAGSYSTEM, FILTSYSTEM, SURVEY and FILTPATH apply to subsequent FILTER
    entries. A new magnitude system resets SURVEY and ZPOFF_FILE, as in SNANA.
    """
    path = path.resolve()
    with open_text(path) as stream:
        contents = stream.read()
    lexer = shlex.shlex(contents, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = '#!'
    tokens = iter(lexer)
    catalogue = {'filters': {}, 'standards': {}}
    magsystem = filtsystem = None
    survey = 'UNDEFINED'
    filtpath = None
    zpoff_file = None
    offsets = {}
    read_zpoff = True
    wavelength_range = (2000., 12000.)  # SNANA's default LAMBDA_RANGE
    sn_sed = None
    leakage_cut = 0.
    duplicate_shift = 0.
    n_system_filters = 0

    def values(key, count):
        result = []
        for _ in range(count):
            value = next(tokens, None)
            if value is None or value.endswith(':'):
                raise SNANAError(f'{path}: {key} requires {count} values')
            result.append(value)
        return result

    def standard(name, filename):
        catalogue['standards'][name] = {'path': filename}

    def add_filter(name, magnitude, definition):
        if magsystem is None or filtsystem is None:
            raise SNANAError(f'{path}: FILTER needs MAGSYSTEM and FILTSYSTEM')
        if name in catalogue['filters']:
            raise SNANAError(f'{path}: duplicate filter {name!r}')
        catalogue['filters'][name] = {
            **definition, 'magsys': magsystem,
            'magzero': magnitude_sum(magnitude), 'lam_unit': 'AA',
            'filter_system': filtsystem, 'survey': survey,
            'zpoff_snpot': offsets.get(name[-1], 0.) if read_zpoff else 0.,
        }

    try:
        for token in tokens:
            if not token.endswith(':'):
                raise SNANAError(f'{path}: expected a calibration key, got {token!r}')
            key = token[:-1].upper()
            if key in _KCOR_INPUT_TABLE_KEYS:
                values(key, _KCOR_INPUT_TABLE_KEYS[key])
            elif key in ('AB_SED', 'VEGA_SED', 'BD17_SED'):
                standard(key.removesuffix('_SED'), values(key, 1)[0])
            elif key == 'PRIMARY_SED':
                name, filename = values(key, 2)
                standard(name, filename)
            elif key == 'MAGSYSTEM':
                magsystem = values(key, 1)[0]
                survey, zpoff_file = 'UNDEFINED', None
            elif key == 'FILTSYSTEM':
                filtsystem = values(key, 1)[0].upper()
                n_system_filters = 0
                if filtsystem not in ('COUNT', 'ENERGY'):
                    raise SNANAError(f'{path}: unknown FILTSYSTEM {filtsystem!r}')
            elif key in ('SURVEY', 'SURVEYS'):
                survey = values(key, 1)[0]
            elif key == 'READ_ZPOFF':
                flag = int(values(key, 1)[0])
                if flag not in (0, 1):
                    raise SNANAError(f'{path}: READ_ZPOFF must be 0 or 1')
                read_zpoff = bool(flag)
            elif key == 'ZPOFF_FILE':
                zpoff_file = values(key, 1)[0]
            elif key == 'FILTPATH':
                filtpath = values(key, 1)[0]
                explicit = not is_ignorefile(zpoff_file)
                filename = zpoff_file if explicit else 'ZPOFF.DAT'
                offset_path = (str(Path(filtpath) / filename)
                               if '/' not in filename else filename)
                source = calibration_path(offset_path, path.parent, 'filters',
                                          required=explicit)
                offsets = {}
                if source is not None and read_zpoff:
                    with open_text(source) as stream:
                        offset_tokens = shlex.shlex(stream, posix=True)
                        offset_tokens.whitespace_split = True
                        offset_tokens.commenters = '#!'
                        pairs = list(offset_tokens)
                    if len(pairs) % 2:
                        raise SNANAError(f'{source}: expected band and zero-point offset pairs')
                    offsets = {band: float(offset) for band, offset
                               in zip(pairs[::2], pairs[1::2])}
            elif key == 'FILTER':
                name, filename, magnitude = values(key, 3)
                if filtpath is None:
                    raise SNANAError(f'{path}: FILTER needs FILTPATH')
                source = calibration_path(str(Path(filtpath) / filename),
                                          path.parent, 'filters')
                add_filter(name, magnitude, {'path': str(source)})
                n_system_filters += 1
            elif key == 'SPECTROGRAPH':
                filename, rebin = parse_spectrograph_filename(values(key, 1)[0])
                source = calibration_path(str(Path(filtpath or '') / filename),
                                          path.parent, 'filters')
                catalogue['spectrograph'] = {'filepath': str(source), 'rebin': rebin}
            elif key == 'SYN_FILTER':
                name, lo, hi, magnitude = values(key, 4)
                if 'spectrograph' not in catalogue:
                    raise SNANAError(f'{path}: SYN_FILTER requires SPECTROGRAPH')
                bounds = (float(lo), float(hi))
                add_filter(name, magnitude, {'synthetic_range': bounds})
                n_system_filters += 1
            elif key == 'FILTER_OOB':
                bands, lo, hi, ratio = values(key, 4)
                bounds = (float(lo), float(hi))
                ratio = float(ratio)
                if not 0. < bounds[0] < bounds[1] or not np.isfinite(ratio) or ratio < 0.:
                    raise SNANAError(f'{path}: invalid FILTER_OOB range or transmission ratio')
                for band in bands:
                    matches = [spec for name, spec in catalogue['filters'].items()
                               if name[-1] == band]
                    if len(matches) != 1:
                        raise SNANAError(f'{path}: FILTER_OOB band {band!r} is missing or ambiguous')
                    matches[0]['out_of_band'] = (*bounds, ratio)
            elif key == 'DUPLICATE_LAMSHIFT_GLOBAL':
                duplicate_shift = float(values(key, 1)[0])
                if not np.isfinite(duplicate_shift):
                    raise SNANAError(f'{path}: wavelength shift must be finite')
            elif key == 'SNMAG0_CHECK':
                values(key, n_system_filters + 1)
            elif key == 'LAMBDA_RANGE':
                wavelength_range = tuple(float(v) for v in values(key, 2))
            elif key == 'SN_SED':
                sn_sed = values(key, 1)[0]
            elif key == 'LEAKAGE_CUT':
                leakage_cut = float(values(key, 1)[0])
                if not 0. <= leakage_cut < 1.:
                    raise SNANAError(f'{path}: LEAKAGE_CUT must be between 0 and 1')
            else:
                raise SNANAError(f'{path}: unsupported calibration setting {key!r}')
    except (ValueError, OSError) as exc:
        raise SNANAError(f'{path}: {exc}') from exc

    if not catalogue['filters']:
        raise SNANAError(f'{path}: no FILTER entries')
    lo, hi = wavelength_range
    if not 0. < lo < hi or not np.isfinite(hi):
        raise SNANAError(f'{path}: LAMBDA_RANGE needs positive, increasing bounds')
    # SNANA stores filter curves on the SN_SED wavelength grid, or a 10 A
    # grid when no SN_SED is supplied. Only its wavelengths are needed here.
    if not is_ignorefile(sn_sed):
        source = calibration_path(sn_sed, path.parent, 'snsed')
        data = np.loadtxt(source, usecols=(0, 1), ndmin=2)
        grid = np.unique(data[:, 1])
        grid = grid[(grid >= lo) & (grid <= hi)]
    else:
        grid = lo + 10. * np.arange(int((hi - lo) / 10.) + 1)
    if len(grid) < 2 or not np.allclose(np.diff(grid), grid[1] - grid[0]):
        raise SNANAError(f'{path}: calibration wavelength grid must be uniform with at least two bins')
    if abs(duplicate_shift) >= .0001:
        # SNANA duplicates the source passband, primary and offsets;
        # SURVEY and FILTER_OOB belong to the nominal definitions.
        for name, spec in list(catalogue['filters'].items()):
            duplicate = spec.copy()
            duplicate.pop('out_of_band', None)
            duplicate.update(wavelength_shift=duplicate_shift, survey='UNDEFINED',
                             magsys=spec['magsys'].split('->')[-1])
            catalogue['filters']['*' + name] = duplicate
    catalogue['wavelength_grid'] = grid
    catalogue['leakage_cut'] = leakage_cut
    # SNANA's default AB standard is flatnu.dat. An explicit AB_SED replaces
    # it; using the same tabulated spectrum preserves FITS calibration values.
    primaries = {p for spec in catalogue['filters'].values()
                 for p in spec['magsys'].split('->')}
    if 'AB' in primaries and 'AB' not in catalogue['standards']:
        standard('AB', 'flatnu.dat')
    catalogue['standards'] = {
        name: {'path': str(calibration_path(spec['path'], path.parent, 'standards'))}
        for name, spec in catalogue['standards'].items() if name in primaries
    }
    return catalogue


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


def synthetic_filter_transmission(calib, lammin, lammax):
    """A synthetic band's relative efficiency, from spectrograph zero points.

    Sample the band at roughly 5 A, with zero-transmission edge bins, as in
    SNANA get_FILTERtrans_spectrograph. Only the observed spectrograph bins
    contribute; edge bins added for spectral smearing are excluded.
    """
    if not 0. < lammin < lammax or not np.isfinite(lammax):
        raise SNANAError('SYN_FILTER needs positive, increasing wavelength bounds')
    n_bins = max(int((lammax - lammin) / 5.), 10)
    step = (lammax - lammin) / n_bins
    wave = lammin - step + step * (np.arange(n_bins + 2) + .5)
    observed = ~np.asarray(calib.islam_extend)
    centers = np.asarray(calib.lam_avg)[observed]
    zp = np.interp(wave, centers, np.asarray(calib.zp)[observed, 0])
    transmission = np.where(zp > .001, 10.**(.4 * (zp - float(calib.magref[0]))), 0.)
    transmission[[0, -1]] = 0.
    peak = transmission.max()
    if peak <= 0.:
        raise SNANAError('SYN_FILTER has no spectrograph response in its wavelength range')
    return wave, transmission / peak


class KcorReader:
    """Load filter transmissions, primary spectra and calibration metadata
    from a calibration FITS file, a SNANA calibration input or a BayeSN
    catalogue."""

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
            wave, flux = read_two_column_file(source)
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
            self.primary_seds['AB'] = (wave, 10.**(-0.4 * 48.6) * LIGHT_A / wave**2)
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
            log.line(logger, f"Remove {name} OOB leakage: WAVE-RANGE = "
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

        log.line(logger, f"ZPoff: {len(self.filter_primary)} filters, "
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

    def get_filter_names(self) -> List[str]:
        """Available bands, without the starred shifted curves."""
        return [name for name in self.filter_names if not name.startswith('*')]

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


def filter_band(name: str) -> str:
    """Band label of a kcor filter name: its last character, as INTFILTER
    takes it (sntools.c:2604)."""
    return name[-1:]


def filter_values(values, names, default):
    """Values in filter order, accepting full names or unambiguous SNANA labels.

    A full name takes precedence. The empty key supplies a value for all bands.
    Keys for inactive filters are retained for later fits but have no effect here.
    """
    labels = [filter_band(name) for name in names]
    result = []
    for name, label in zip(names, labels):
        if name in values:
            result.append(values[name])
        elif label in values:
            if labels.count(label) > 1:
                raise SNANAError(
                    f"Ambiguous filter label {label!r}; use full filter names")
            result.append(values[label])
        else:
            result.append(values.get('', default))
    return result


def mw_attenuation_rest(target_wave_grid, wave_rest, mw_att_obs):
    """Observer-frame Milky Way attenuation ``mw_att_obs``, tabulated on the
    common grid blue-shifted to ``wave_rest``, on the rest-frame grid
    ``target_wave_grid``: Catmull-Rom, and 1 beyond the tabulated range."""
    return jnp.where(
        (target_wave_grid < wave_rest[0]) | (target_wave_grid > wave_rest[-1]), 1.0,
        hermite_interp_1d(target_wave_grid, wave_rest, mw_att_obs,
                          compute_catmullrom_slopes(wave_rest, mw_att_obs)))


def band_weights_at_z(z, filter_wave, filter_trans, model_wave, mw_att=None):
    """Rest-frame weights for all filters at one redshift."""
    return jax.vmap(compute_single_restframe_filter,
                    in_axes=(None, None, 0, None, None))(
        z, filter_wave, filter_trans, model_wave, mw_att)


@jax.jit
def compute_single_restframe_filter(
    z: jnp.ndarray,
    filter_wave_common: jnp.ndarray,
    filter_trans_common: jnp.ndarray,
    target_wave_grid: jnp.ndarray,
    mw_att_obs: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Rest-frame band weights for one filter at one redshift.

    The filter is stored on a common observer-frame wavelength grid
    (``filter_wave_common``); this function blue-shifts to rest frame,
    interpolates onto the simulator's fixed rest-frame wavelength grid,
    normalises the weights to unit total λ·T integral, optionally folds in
    observer-frame MW extinction, and applies the ``1/(1+z)`` photon-energy
    scaling so the returned weights integrate an SED directly to observer-
    frame flux.

    ``z`` is a single redshift value; ``filter_wave_common`` and
    ``filter_trans_common`` are the observer-frame filter wavelengths and
    transmissions on the common grid, each shape ``(n_common,)``;
    ``target_wave_grid`` is the fixed rest-frame wavelength grid for
    interpolation; ``mw_att_obs``, when given, is the MW dust attenuation on
    the observer-frame common grid, shape ``(n_common,)``, applied after
    normalisation so the total flux reduction is preserved.

    Returns the band weights for this (filter, redshift), shape
    ``(len(target_wave_grid),)``.
    """
    wave_obs = filter_wave_common
    z_plus_1 = 1.0 + z

    # Hermite: weights C1 in z
    wave_rest = wave_obs / z_plus_1
    trans_rest = jnp.where(
        (target_wave_grid < wave_rest[0]) | (target_wave_grid > wave_rest[-1]), 0.0,
        hermite_interp_1d(target_wave_grid, wave_rest, filter_trans_common,
                          compute_catmullrom_slopes(wave_rest, filter_trans_common)))

    # normalised before MW extinction
    trans_wave = trans_rest * target_wave_grid
    trans_norm = jnp.maximum(jnp.trapezoid(trans_wave, x=target_wave_grid), 1e-30)

    dlambda = jnp.concatenate([
        jnp.array([target_wave_grid[1] - target_wave_grid[0]]),
        (target_wave_grid[2:] - target_wave_grid[:-2]) / 2.0,
        jnp.array([target_wave_grid[-1] - target_wave_grid[-2]]),
    ])
    band_weights = (trans_rest * target_wave_grid * dlambda) / trans_norm

    if mw_att_obs is not None:
        band_weights = band_weights * mw_attenuation_rest(
            target_wave_grid, wave_rest, mw_att_obs)

    # 1/(1+z) scaling converts integrated rest-frame flux to observer-frame.
    return band_weights / z_plus_1


def band_window_width(wave_rest, filter_wave_common, filter_trans_common,
                      margin: int = 6):
    """Grid points spanned by the widest filter at z = 0, with margin, or
    None where that is the whole grid.

    A filter spans ``[lam_lo, lam_hi] / (1 + z)`` of the rest grid, so its
    point count falls with redshift and z = 0 bounds every event.
    """
    wave_rest = np.asarray(wave_rest)
    wave_obs = np.asarray(filter_wave_common)
    nonzero = np.asarray(filter_trans_common) > 0.0
    covered = nonzero.any(axis=1)
    if not covered.any():
        return None
    # a cell wider each end: interp leaves weight on the zero-side neighbour
    first = np.maximum(nonzero.argmax(axis=1) - 1, 0)
    last = np.minimum(
        nonzero.shape[1] - 1 - nonzero[:, ::-1].argmax(axis=1) + 1,
        nonzero.shape[1] - 1)
    lo, hi = wave_obs[first], wave_obs[last]
    # A band outside the grid at z=0 can redshift fully into it.
    spans = np.ceil((hi - lo) / np.diff(wave_rest).min())
    width = int(spans[covered].max()) + margin
    return None if width >= len(wave_rest) else width


def band_window_starts(band_weights, filter_indices, width):
    """Window start per observation, two points before the filter's non-zero
    run and clipped to the grid: dynamic_slice clamps silently, so an
    out-of-range start would shift the window rather than fail.
    """
    n_wave = band_weights.shape[-1]
    return jnp.clip(jnp.argmax(band_weights != 0.0, axis=-1) - 2,
                    0, n_wave - width)[filter_indices]


def fluxcal_from_sed(sed_at_phase, phases, mask, filter_indices, band_weights,
                     ref_fluxes, zp, scale, width=None, sed_factor=None):
    """FLUXCAL at each (phase, filter), integrating over the filter's window.

    Only the window is read from ``sed_at_phase(phase)``, and every step of
    an SED model is elementwise in wavelength, so XLA never emits the
    arithmetic for the rest. A model that reduced across wavelength would
    lose that speedup.

    ``filter_indices`` must index ``band_weights`` on padded lanes too;
    ``width`` is None for the whole grid; ``sed_factor`` is an optional
    rest-frame ``[n_wavelengths]`` multiplier on the source.
    """
    n_wave = band_weights.shape[-1]
    if width is None:
        width = n_wave
    obs_lo = band_window_starts(band_weights, filter_indices, width)
    obs_band_weights = band_weights[filter_indices]
    obs_ref_fluxes = ref_fluxes[filter_indices]
    fluxcal_scale = jnp.power(10.0, 0.4 * zp) * scale

    def one_obs(phase, lo, obs_bw, obs_ref):
        cut = lambda a: lax.dynamic_slice(a, (lo,), (width,))
        sed = cut(sed_at_phase(phase))
        if sed_factor is not None:
            sed = sed * cut(sed_factor)
        return jnp.sum(sed * cut(obs_bw)) / obs_ref * fluxcal_scale

    return jnp.where(
        mask, jax.vmap(one_obs)(phases, obs_lo, obs_band_weights,
                                obs_ref_fluxes), 0.0)


def filter_mean_wavelength(wave, trans) -> float:
    """Transmission-weighted mean observer wavelength of one filter.

    SNANA's FILTOBS_LAMAVG, sum(dlam*T*lam)/sum(dlam*T) with dlam the central
    difference and one-sided at the ends (kcor.c:2340-2356), which is what
    np.gradient produces. An empty filter falls back to its grid centre.
    """
    wave = np.asarray(wave)
    trans = np.asarray(trans)
    dlam = np.gradient(wave)
    tsum = float(np.sum(dlam * trans))
    if tsum <= 0.0:
        return float(np.mean(wave))
    return float(np.sum(dlam * trans * wave) / tsum)


def _ref_mag_terms(ref_fluxes: jnp.ndarray) -> jnp.ndarray:
    """2.5*log10 of the reference flux, with SNANA's zero-reference branch:
    a filter whose reference integral is zero gets ZP = 0
    (genmag_SEDtools.c:338)."""
    nonzero = ref_fluxes != 0.0
    return jnp.where(
        nonzero, 2.5 * jnp.log10(jnp.where(nonzero, ref_fluxes, 1.0)), 0.0)


@dataclass
class PhotometryResources:
    """The active filters prepared for a model's wavelength grid: rest-frame
    transmissions, the observer-frame common grid, primary-standard reference
    fluxes and magnitude terms, and mean wavelengths."""

    filter_names: List[str]

    transmissions: Dict[str, jnp.ndarray]
    transmissions_stacked: jnp.ndarray                  # (n_filters, n_wave)
    common_wave_grid: jnp.ndarray
    common_trans_grid: jnp.ndarray                      # (n_filters, n_common)

    filter_ref_fluxes: jnp.ndarray                      # photon-weighted, normalised
    filter_ref_fluxes_snana: jnp.ndarray                # SNANA LAMSTEP/hc convention
    filter_ref_mag_terms: jnp.ndarray
    filter_ref_mag_terms_snana: jnp.ndarray

    filter_mean_wavelengths: jnp.ndarray                # ∫T·λ / ∫T, FILTOBS_LAMAVG


jax.tree_util.register_dataclass(
    PhotometryResources,
    data_fields=[
        'transmissions',
        'transmissions_stacked',
        'common_wave_grid',
        'common_trans_grid',
        'filter_ref_fluxes',
        'filter_ref_fluxes_snana',
        'filter_ref_mag_terms',
        'filter_ref_mag_terms_snana',
        'filter_mean_wavelengths',
    ],
    meta_fields=[
        'filter_names',
    ],
)


def load_photometry_resources(
    kcor,
    wavelength_grid: jnp.ndarray,
    filter_names: List[str],
    genmag_off_zp_map: Mapping[str, float] = MappingProxyType({}),
    lam_shift_map: Mapping[str, float] = MappingProxyType({}),
    primary_mag_shift_map: Mapping[str, float] = MappingProxyType({}),
) -> PhotometryResources:
    """Build :class:`PhotometryResources` from an already-loaded
    :class:`KcorReader` (``kcor``), a rest-frame working ``wavelength_grid``
    (Angstroms) the model was resampled to, and ``filter_names`` (the
    filters to process, in the desired index order).

    ``genmag_off_zp_map``, if given, is a ``{short_band: mag_offset}`` map
    from the sim's ``GENMAG_OFF_ZP`` input key, left empty by the fitter
    (which has no equivalent config). ``lam_shift_map``, if given, is
    a ``{short_band: wavelength_shift}`` map added to the filter's
    wavelength grid before use. ``primary_mag_shift_map``, if given, is
    a ``{short_band: mag_shift}`` map from the fitter's
    ``MAGOBS_SHIFT_PRIMARY``, added to the kcor primary magnitude. These
    last two may also key a full filter name, which wins over its band.
    """
    if not filter_names:
        raise SNANAError(
            "load_photometry_resources: filter_names is empty.",
            function_name="load_photometry_resources",
        )

    def _shift(m: Mapping[str, float], name: str) -> float:
        # full name (SALT2.INFO) first, then the band (namelist)
        if name in m:
            return float(m[name])
        return float(m.get(filter_band(name), 0.0))

    wavelength_grid = jnp.asarray(wavelength_grid)
    n_wave = int(wavelength_grid.shape[0])

    raw_filters = []
    for name in filter_names:
        w, t = kcor.get_filter_transmission(name)
        raw_filters.append((jnp.asarray(w, dtype=float_dtype()) + _shift(lam_shift_map, name),
                            jnp.asarray(t, dtype=float_dtype())))
    raw_filters_map = dict(zip(filter_names, raw_filters))

    transmissions: Dict[str, jnp.ndarray] = {
        name: jnp.interp(wavelength_grid, *raw_filters_map[name],
                         left=0.0, right=0.0)
        for name in filter_names
    }
    transmissions_stacked = jnp.stack(
        [transmissions[name] for name in filter_names], axis=0)

    min_wave = float(min(w.min() for w, _ in raw_filters))
    max_wave = float(max(w.max() for w, _ in raw_filters))
    common_wave_grid = jnp.linspace(min_wave, max_wave, n_wave)
    common_trans_list = []
    for name in filter_names:
        wave_obs, trans_obs = raw_filters_map[name]
        common_trans_list.append(jnp.interp(
            common_wave_grid, wave_obs, trans_obs, left=0.0, right=0.0))
    common_trans_grid = jnp.stack(common_trans_list, axis=0)

    primary_seds_cache: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]] = {}
    for name in filter_names:
        primary_name = kcor.get_filter_primary(name)
        if primary_name not in primary_seds_cache:
            pwaves, pflux = kcor.get_primary_sed(primary_name)
            primary_seds_cache[primary_name] = (
                jnp.asarray(pwaves), jnp.asarray(pflux))

    filter_ref_fluxes_list: List[float] = []
    filter_ref_fluxes_snana_list: List[float] = []
    mean_wavelengths: List[float] = []

    for name, (wave_obs_np, trans_obs_np) in zip(filter_names, raw_filters):
        wave_obs = jnp.asarray(wave_obs_np)
        trans_obs = jnp.asarray(trans_obs_np)

        primary_name = kcor.get_filter_primary(name)
        prim_waves, prim_flux = primary_seds_cache[primary_name]
        ref_obs = jnp.interp(
            wave_obs, prim_waves, prim_flux, left=0.0, right=0.0)

        transmission_wave_obs = trans_obs * wave_obs
        transmission_norm_obs = jnp.trapezoid(transmission_wave_obs, wave_obs)
        ref_integrand_obs = ref_obs * transmission_wave_obs
        ref_flux = (
            jnp.trapezoid(ref_integrand_obs, wave_obs) / transmission_norm_obs)
        filter_ref_fluxes_list.append(ref_flux)

        # SNANA-convention reference flux: rectangle rule × LAMSTEP / hc.
        lamstep_obs = jnp.median(jnp.diff(wave_obs))
        ref_integrand_snana = ref_obs * trans_obs * wave_obs
        ref_flux_snana = jnp.sum(ref_integrand_snana) * lamstep_obs / HC_ERG_A
        filter_ref_fluxes_snana_list.append(ref_flux_snana)

        mean_wavelengths.append(filter_mean_wavelength(
            np.asarray(wave_obs_np), np.asarray(trans_obs_np)))

    filter_ref_fluxes_array = jnp.asarray(filter_ref_fluxes_list)
    filter_ref_fluxes_snana_array = jnp.asarray(filter_ref_fluxes_snana_list)

    # delta = magprimary - ZPoff(SNpot) - user_zp (genmag_SEDtools.c:338, snlc_sim.c:8487, sntools_calib.c:1028)
    scale_per_filter: List[float] = []
    for name in filter_names:
        short_band = filter_band(name)
        missing = [label for label, table in
                   (('Primary Mag', kcor.filter_primary_mag),
                    ('ZPoff(SNpot)', kcor.filter_zpoff_snpot))
                   if name not in table]
        if missing:
            raise SNANAError(
                f"KCOR ZPoff table has no {', '.join(missing)} entry for active "
                f"filter '{name}'; SNANA reads these columns for every filter.",
                function_name="load_photometry_resources")
        primary_mag = (float(kcor.filter_primary_mag[name])
                       + _shift(primary_mag_shift_map, name))
        zpoff_snpot = float(kcor.filter_zpoff_snpot[name])
        user_zp = float(genmag_off_zp_map.get(short_band, 0.0))
        delta = primary_mag - zpoff_snpot - user_zp
        scale_per_filter.append(float(np.power(10.0, 0.4 * delta)))
    zp_scale = jnp.asarray(scale_per_filter)
    filter_ref_fluxes_array = filter_ref_fluxes_array * zp_scale
    filter_ref_fluxes_snana_array = filter_ref_fluxes_snana_array * zp_scale

    filter_ref_mag_terms = _ref_mag_terms(filter_ref_fluxes_array)
    filter_ref_mag_terms_snana = _ref_mag_terms(filter_ref_fluxes_snana_array)

    return PhotometryResources(
        filter_names=list(filter_names),
        transmissions=transmissions,
        transmissions_stacked=transmissions_stacked,
        common_wave_grid=common_wave_grid,
        common_trans_grid=common_trans_grid,
        filter_ref_fluxes=filter_ref_fluxes_array,
        filter_ref_fluxes_snana=filter_ref_fluxes_snana_array,
        filter_ref_mag_terms=filter_ref_mag_terms,
        filter_ref_mag_terms_snana=filter_ref_mag_terms_snana,
        filter_mean_wavelengths=jnp.asarray(mean_wavelengths),
    )


def read_calib_shifts(path):
    """MAGSHIFT/WAVESHIFT/LAMSHIFT entries of a model metadata file, as
    ('mag'|'lam', survey_csv, filter, shift) tuples (genmag_SALT2.c:1307)."""
    shifts = []
    with open(path) as fh:
        for line in fh:
            key, _, value = line.split('#')[0].partition(':')
            if key.strip() in ('MAGSHIFT', 'WAVESHIFT', 'LAMSHIFT'):
                survey, filt, shift = value.split()
                shifts.append(('mag' if key.strip() == 'MAGSHIFT' else 'lam',
                               survey, filt, float(shift)))
    return shifts


def resolve_calib_shifts(shifts, filter_names, filter_survey, enable_mag,
                         enable_wave):
    """Calibration shifts as {full_filter_name: shift} pairs.

    Mirrors match_SALT2train (genmag_SALT2.c:1915): the filter's own survey is
    itself a comma-separated list, names are truncated at '/', and a
    single-character calib filter collapses both sides to the trailing band.
    """
    mag_out, lam_out = {}, {}
    for which, survey_csv, filt_calib, shift in shifts:
        if which == 'mag' and not enable_mag:
            continue
        if which == 'lam' and not enable_wave:
            continue
        calib_base = filt_calib.split('/')[0]
        for name in filter_names:
            surveys = [x.strip() for x in
                       str(filter_survey.get(name, 'UNDEFINED')).split(',')]
            if not any(s in surveys for s in survey_csv.split(',')):
                continue
            genmag_base = name.split('/')[0]
            if len(filt_calib) == 1:
                genmag_base, calib_cmp = name[-1], filt_calib[-1]
            else:
                calib_cmp = calib_base
            if genmag_base != calib_cmp:
                continue
            out = mag_out if which == 'mag' else lam_out
            out[name] = out.get(name, 0.0) + shift
    return mag_out, lam_out

