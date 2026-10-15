"""Read SNANA calibration inputs into filter and primary-standard definitions.

This module handles the input format and resource paths. Loading spectra and
transmission curves belongs to KcorReader, as for BayeSN catalogues.
"""

import os
from pathlib import Path
import re
import shlex

import numpy as np

from jax_snlc_core.io import is_ignorefile
from jax_snlc_core.io.fits_path import open_text, resolve_fits_path
from jax_snlc_core.utils.error_handling import SNANAError
from jax_snlc_core.photometry.spectrograph import parse_spectrograph_filename


# These settings describe generated tables or diagnostic output, rather than
# the filter calibration used for direct SED integration.
_TABLE_SETTINGS = {
    'OUTFILE': 1, 'KCOR': 3, 'REDSHIFT_RANGE': 2, 'REDSHIFT_BINSIZE': 1,
    'TREST_RANGE': 2, 'AV_RANGE': 2, 'AV_BINSIZE': 1, 'AV_OPTION': 1,
    'RV': 1, 'RV_MWCOLORLAW': 1, 'OPT_MWCOLORLAW': 1,
    'SN_TYPE': 1, 'SN_MAGOFF': 1, 'TREF_EXPLODE': 1,
    'PLOTFLAG_SN': 1, 'PLOTFLAG_PRIMARY': 1, 'DUMP_SNMAG': 1,
    'SN_SED_FUDGE': 1, 'NLAMBIN_FT': 1,
}
_NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'


def magnitude_sum(value):
    """A primary magnitude may be written as a sum, e.g. 9.824+.011-.002."""
    terms = re.findall(_NUMBER, value)
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
    """Return definitions for the shared calibration loader.

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
            if key in _TABLE_SETTINGS:
                values(key, _TABLE_SETTINGS[key])
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
