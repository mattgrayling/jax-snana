"""HOSTLIB_SPECBASIS_FILE / HOSTLIB_SPECDATA_FILE reader.

SNANA reads these with its generic table reader (read_specTable_SNANA,
sntools_host.c:1880): a `VARNAMES:` header plus `ROW:` lines, where the
wavelength column is whichever name contains wave/lam (either case) and the
basis columns are those starting with the SPECBASIS/SPECDATA prefix followed
by an index. The two header keys FLAM_SCALE and FLAM_SCALE_POWZ1 scale the
result as FLAM = sum(coeff_i * basis_i) * FLAM_SCALE * (1+z)^POWZ1.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, NamedTuple, Union

import numpy as np

from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import open_text, read_two_column_file

# sntools_host.h:606-610
PREFIX_SPECBASIS = 'SPECBASIS'
PREFIX_SPECDATA = 'SPECDATA'
PREFIX_SPECBASIS_HOSTLIB = 'COEFF_'
VARNAME_SPECDATA_HOSTLIB = 'IDSPECDATA'

_WAVE_TOKENS = ('wave', 'lam')

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
    index: np.ndarray          # (n_spec,) the integer after each column prefix
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
        (genSpec_HOSTLIB, sntools_host.c:2232-2262).
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
        (sntools_host.c:2287-2288), so the last bin's upper half counts as
        off-grid; a bin whose rest-frame range leaves that span gets no flux.
        """
        z1 = 1.0 + redshift
        rest_lo = np.asarray(lam_min_obs, dtype=float) / z1
        rest_hi = np.asarray(lam_max_obs, dtype=float) / z1
        return (rest_lo >= self.wave_min[0]) & (rest_hi <= self.wave_min[-1])


def read_host_spec_table(path: str, mode: str) -> HostSpecTable:
    """Read a SPECBASIS or SPECDATA table."""
    if mode not in (PREFIX_SPECBASIS, PREFIX_SPECDATA):
        raise SNANAError(f"Unknown host-spectrum mode {mode!r}; expected "
                         f"{PREFIX_SPECBASIS} or {PREFIX_SPECDATA}.",
                         function_name="read_host_spec_table")

    scale, powz1, varnames, rows = 1.0, 0.0, None, []
    with open_text(path) as f:
        for line in f:
            words = line.split('#')[0].split()
            if not words:
                continue
            key = words[0]
            if key == 'FLAM_SCALE:':
                scale = float(words[1])
            elif key == 'FLAM_SCALE_POWZ1:':
                powz1 = float(words[1])
            elif key == 'VARNAMES:':
                varnames = words[1:]
            elif key == 'ROW:' and varnames is not None:
                rows.append([float(w) for w in words[1:]])
            elif key == 'EAZY_TEMPLATES_LIST_FILE:':
                wave, flam = _read_eazy_templates(Path(os.path.expandvars(words[1])))
                return _build_table(mode, wave, flam, np.arange(len(flam)),
                                    scale, powz1)

    if varnames is None or not rows:
        raise SNANAError(f"No VARNAMES/ROW table found in {path}.",
                         function_name="read_host_spec_table")

    table = np.asarray(rows, dtype=float)
    icol_wave = _wavelength_column(varnames, path)
    icols, index = _template_columns(varnames, mode, path)
    return _build_table(mode, table[:, icol_wave], table[:, icols].T,
                        index, scale, powz1)


def _build_table(mode, wave, flam, index, scale, powz1) -> HostSpecTable:
    """Truncate the grid and derive bin edges (sntools_host.c:1792).

    Edges sit midway between neighbouring centres; the first and last bins
    mirror their one neighbour's half-width.
    """
    keep = wave <= LAMMAX_SEDMODEL
    wave, flam = wave[keep], flam[:, keep]
    half = np.diff(wave) / 2.0
    wave_min = np.concatenate([[wave[0] - half[0]], wave[1:] - half])
    wave_max = np.concatenate([wave[:-1] + half, [wave[-1] + half[-1]]])
    return HostSpecTable(mode=mode, wave=wave, wave_min=wave_min,
                         wave_max=wave_max, flam=flam, index=index,
                         flam_scale=scale, flam_scale_powz1=powz1)


class HostSpectra(NamedTuple):
    """The host-spectrum templates and each galaxy's coefficients on them
    (or its SPECDATA ID), in the catalogue's galaxy order."""
    table: HostSpecTable
    coeffs: np.ndarray              # (n_gal, n_spec)


def hostlib_column_names(table: HostSpecTable) -> List[str]:
    """HOSTLIB columns this table needs (match_specTable_HOSTVAR, :2090).

    SPECBASIS wants one coefficient per basis vector, named COEFF_SPECBASIS<nn>
    by the basis file's read order; SPECDATA wants a single ID column.
    """
    if table.mode == PREFIX_SPECDATA:
        return [VARNAME_SPECDATA_HOSTLIB]
    return [f'{PREFIX_SPECBASIS_HOSTLIB}{PREFIX_SPECBASIS}{i:02d}'
            for i in range(table.n_spec)]


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
    """Basis vectors named by an EAZY templates list (sntools_host.c:1985).

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


def _wavelength_column(varnames: List[str], path: str) -> int:
    """The last name containing wave/lam, as SNANA's scan leaves it."""
    found = [i for i, v in enumerate(varnames)
             if any(t in v.lower() for t in _WAVE_TOKENS)]
    if not found:
        raise SNANAError(f"No wavelength column in {path}; expected a VARNAME "
                         f"containing 'wave' or 'lam'.",
                         function_name="read_host_spec_table")
    return found[-1]


def _template_columns(varnames: List[str], mode: str, path: str):
    """Columns named <prefix><index>, in file order (sntools_host.c:1902)."""
    icols, index = [], []
    for i, name in enumerate(varnames):
        match = re.fullmatch(rf'{mode}(\d+)', name, flags=re.IGNORECASE)
        if match:
            icols.append(i)
            index.append(int(match.group(1)))
    if not icols:
        raise SNANAError(f"No {mode}<nn> columns in {path}.",
                         function_name="read_host_spec_table")
    return icols, np.asarray(index, dtype=int)


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
