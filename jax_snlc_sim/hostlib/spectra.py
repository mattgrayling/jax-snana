"""HOSTLIB_SPECBASIS_FILE / HOSTLIB_SPECDATA_FILE reader.

SNANA reads the header keys up to `VARNAMES:` and then the table with its
generic table reader (read_specTable_SNANA, sntools_host.c:1862). The one
wavelength column is the name containing wave, WAVE, lam or LAM, and the basis
columns are those starting with the SPECBASIS or SPECDATA prefix. The two
header keys FLAM_SCALE and FLAM_SCALE_POWZ1 scale the result as
FLAM = sum(coeff_i * basis_i) * FLAM_SCALE * (1+z)^POWZ1.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, NamedTuple, Union

import numpy as np

from jax_snlc_core import logging as log
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import (key_runs, leading_values, open_text, read_fitres,
                                       read_two_column_file)

logger = logging.getLogger(__name__)

# sntools_host.h:631-635
PREFIX_SPECBASIS = 'SPECBASIS'
PREFIX_SPECDATA = 'SPECDATA'
PREFIX_SPECBASIS_HOSTLIB = 'COEFF_'
VARNAME_SPECDATA_HOSTLIB = 'IDSPECDATA'

_WAVE_TOKENS = ('wave', 'WAVE', 'lam', 'LAM')

# basis grid truncation (genmag_SEDtools.h:41)
LAMMAX_SEDMODEL = 30000.0


@dataclass(frozen=True)
class HostSpecTable:
    """Rest-frame host spectral templates from one SPECBASIS/SPECDATA file."""

    mode: str                  # 'SPECBASIS' or 'SPECDATA'
    wave: np.ndarray           # (n_wave,) rest-frame bin centres
    wave_min: np.ndarray       # (n_wave,) rest-frame bin lower edges
    wave_max: np.ndarray       # (n_wave,) rest-frame bin upper edges
    flam: np.ndarray           # (n_spec, n_wave) template fluxes, file order
    names: List[str]           # each template's column name
    flam_scale: float
    flam_scale_powz1: float

    @property
    def n_spec(self) -> int:
        """Number of templates in the table."""
        return self.flam.shape[0]

    def scale(self, redshift):
        """FLAM_SCALE * (1+z)^POWZ1, the per-event normalisation."""
        return self.flam_scale * (1.0 + np.asarray(redshift)) ** self.flam_scale_powz1

    def event_seds(self, selector, redshift):
        """(n_event, n_wave) rest-frame FLAM on the basis grid.

        SPECBASIS combines every template by its per-galaxy coefficients;
        SPECDATA takes the single spectrum the galaxy's ID names
        (genSpec_HOSTLIB, sntools_host.c:2243-2273).
        """
        selector = np.asarray(selector)
        if self.mode == PREFIX_SPECDATA:
            flam = self.flam[self._specdata_index(selector[:, 0])]
        else:
            flam = selector @ self.flam
        return flam * np.asarray(self.scale(redshift))[:, None]

    def _specdata_index(self, ids):
        """IDSPECDATA -> row, rejecting the out-of-range value SNANA's own
        check lets through (its guard uses `> NSPECDATA` while its message
        says the last valid ID is NSPECDATA-1)."""
        idx = np.asarray(ids).astype(int)
        bad = (idx < 0) | (idx >= self.n_spec)
        if bad.any():
            raise SNANAError(
                f"Invalid IDSPECDATA = {idx[bad][0]}; valid range is 0 to "
                f"{self.n_spec - 1} for a SPECDATA table with {self.n_spec} "
                f"spectra.", function_name="event_seds")
        return idx

    def on_grid(self, redshift, lam_min_obs, lam_max_obs):
        """Which spectrograph bins carry host flux.

        SNANA bounds the basis grid by WAVE_MIN at *both* ends
        (sntools_host.c:2298-2299), so the last bin's upper half counts as
        off-grid; a bin whose rest-frame range leaves that span gets no flux.
        """
        z1 = 1.0 + redshift
        rest_lo = np.asarray(lam_min_obs, dtype=float) / z1
        rest_hi = np.asarray(lam_max_obs, dtype=float) / z1
        return (rest_lo >= self.wave_min[0]) & (rest_hi <= self.wave_min[-1])


def read_host_spec_table(path: str, mode: str) -> HostSpecTable:
    """Read a SPECBASIS or SPECDATA table: the header keys as a stream of
    words up to VARNAMES or EAZY_TEMPLATES_LIST_FILE, then the table or the
    templates that key lists (read_specTable_HOSTLIB, sntools_host.c:1765)."""
    if mode not in (PREFIX_SPECBASIS, PREFIX_SPECDATA):
        raise SNANAError(f"Unknown host-spectrum mode {mode!r}; expected "
                         f"{PREFIX_SPECBASIS} or {PREFIX_SPECDATA}.",
                         function_name="read_host_spec_table")

    scale, powz1 = 1.0, 0.0
    for key, values in key_runs(path):
        if key == 'FLAM_SCALE':
            scale = float(leading_values(path, key, values, 1)[0])
        elif key == 'FLAM_SCALE_POWZ1':
            powz1 = float(leading_values(path, key, values, 1)[0])
        elif key == 'EAZY_TEMPLATES_LIST_FILE':
            list_file = leading_values(path, key, values, 1)[0]
            wave, flam = _read_eazy_templates(Path(os.path.expandvars(list_file)))
            # named as a table's columns would be (sntools_host.c:2063)
            names = [f'{PREFIX_SPECBASIS}{i:02d}' for i in range(len(flam))]
            return _build_table(mode, wave, flam, names, scale, powz1)
        elif key == 'VARNAMES':
            break
        else:
            log.warn(logger, "HOSTLIB", f"{path}: {key}: is not read")
    else:
        raise SNANAError(f"{path} has neither VARNAMES nor "
                         f"EAZY_TEMPLATES_LIST_FILE.",
                         function_name="read_host_spec_table")

    table = read_fitres(path).df
    wave = table[_wavelength_column(list(table.columns), path)]
    names = _template_columns(list(table.columns), mode, path)
    return _build_table(mode, wave.to_numpy(dtype=float),
                        table[names].to_numpy(dtype=float).T, names, scale, powz1)


def _build_table(mode, wave, flam, names, scale, powz1) -> HostSpecTable:
    """Truncate the grid and derive bin edges (sntools_host.c:1803).

    Edges sit midway between neighbouring centres; the first and last bins
    mirror their one neighbour's half-width.
    """
    keep = wave <= LAMMAX_SEDMODEL
    wave, flam = wave[keep], flam[:, keep]
    half = np.diff(wave) / 2.0
    wave_min = np.concatenate([[wave[0] - half[0]], wave[1:] - half])
    wave_max = np.concatenate([wave[:-1] + half, [wave[-1] + half[-1]]])
    return HostSpecTable(mode=mode, wave=wave, wave_min=wave_min,
                         wave_max=wave_max, flam=flam, names=names,
                         flam_scale=scale, flam_scale_powz1=powz1)


class HostSpectra(NamedTuple):
    """The host-spectrum templates and each galaxy's coefficients on them
    (or its SPECDATA ID), in the catalogue's galaxy order."""
    table: HostSpecTable
    coeffs: np.ndarray              # (n_gal, n_spec)


def hostlib_column_names(table: HostSpecTable) -> List[str]:
    """HOSTLIB columns this table needs (match_specTable_HOSTVAR, :2090).

    SPECBASIS wants one coefficient per basis vector, named COEFF_ and the
    basis vector's column name (sntools_host.c:766-770); SPECDATA wants a
    single ID column.
    """
    if table.mode == PREFIX_SPECDATA:
        return [VARNAME_SPECDATA_HOSTLIB]
    return [f'{PREFIX_SPECBASIS_HOSTLIB}{name}' for name in table.names]


def hostlib_coefficients(table: HostSpecTable, column_data) -> np.ndarray:
    """(n_gal, n_spec) SPECBASIS coefficients from loaded HOSTLIB columns.

    SNANA aborts when a required column is absent rather than defaulting it to
    zero, which would silently yield a fainter host (sntools_host.c:2101).
    """
    names = hostlib_column_names(table)
    missing = [n for n in names if n not in column_data]
    if missing:
        raise SNANAError(
            f"HOSTLIB is missing required host-spectrum column(s) "
            f"{', '.join(missing)} for a {table.mode} table with "
            f"{table.n_spec} templates.",
            function_name="hostlib_coefficients")
    return np.stack([np.asarray(column_data[n]) for n in names], axis=1)


def _read_eazy_templates(list_file: Union[str, Path]):
    """Basis vectors named by an EAZY templates list (sntools_host.c:1996).

    Each list row is `index filename scale`. SNANA keeps only the filename's
    basename and resolves it against the list file's own directory, discarding
    both the row's relative path and its scale; templates are numbered by read
    order rather than by the index column.
    """
    directory = Path(list_file).parent
    wave, flam = None, []
    with open_text(list_file) as f:
        for line in f:
            words = line.split()
            if not words:
                continue
            wave, flux = read_two_column_file(directory / Path(words[1]).name)
            flam.append(flux)

    if not flam:
        raise SNANAError(f"No templates listed in {list_file}.",
                         function_name="read_host_spec_table")
    # ragged templates raise: SNANA shares one wavelength array
    return wave, np.asarray(flam, dtype=float)


def _wavelength_column(varnames: List[str], path: str) -> str:
    """The one name containing wave, WAVE, lam or LAM; SNANA aborts on none
    or several (sntools_host.c:1896-1943)."""
    found = [v for v in varnames if any(t in v for t in _WAVE_TOKENS)]
    if len(found) != 1:
        raise SNANAError(f"{path} has {len(found)} wavelength columns "
                         f"({' '.join(found)}); expected one, named with "
                         f"wave, WAVE, lam or LAM.",
                         function_name="read_host_spec_table")
    return found[0]


def _template_columns(varnames: List[str], mode: str, path: str) -> List[str]:
    """The names starting with the ``mode`` prefix in any case, in file
    order (sntools_host.c:1909-1913)."""
    names = [v for v in varnames if v[:len(mode)].upper() == mode]
    if not names:
        raise SNANAError(f"No {mode} columns in {path}.",
                         function_name="read_host_spec_table")
    return names


def load_host_spec_table(config):
    """The SPECBASIS/SPECDATA table this config asks for, or None.

    One table slot (HOSTSPEC.ITABLE): SNANA silently lets SPECDATA win when
    both are named, which jax rejects rather than picking for the user.
    """
    basis, data = config.host.specbasis_file, config.host.specdata_file
    if basis and data:
        raise SNANAError(
            "HOSTLIB_SPECBASIS_FILE and HOSTLIB_SPECDATA_FILE are mutually "
            "exclusive; SNANA stores one host-spectrum table.",
            function_name="load_host_spec_table")
    if not (basis or data):
        return None
    path, mode = ((basis, PREFIX_SPECBASIS) if basis
                  else (data, PREFIX_SPECDATA))
    return read_host_spec_table(Path(os.path.expandvars(path)), mode)
