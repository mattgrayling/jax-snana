"""SIMSED parameterized SED grid model for peculiar Ia subtypes.

Supports two modes:

- **GRIDONLY** (e.g., Iax): discrete template selection by index,
  identical to NON1ASED at the SED level.
- **Continuous** (e.g., 91bg): N-dimensional multilinear interpolation
  over a regular parameter grid (2^N hypercube corners).

Matches SNANA's genmag_SIMSED.c:interp_flux_SIMSED().
"""

import logging
import os
import re
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core import logging as log
from jax_snlc_core.interpolation import linear_between, linear_bin
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import (interpolate_wgtmap, read_wgtmap, resolve_fits_path,
                                       skip_docana)

from .input_keys import SIM_ONLY
from .roles import SimulatedModel
from .sed_model import (SED_WAVELENGTH_RANGE, SNModel, compute_explosion_tshift,
                        late_time_decay, late_time_tables, read_sed_file, resample_surfaces,
                        uvlam_extrapflux_sedmodel)

logger = logging.getLogger(__name__)

# grid points of the tabulated marginal in _draw_correlated_truncated
N_GRID_SIMSED_REDCOR = 4096
# the rest-frame days a text SED is read within (genmag_SIMSED.c:234-235)
SIMSED_DAY_RANGE = (-500.0, 1500.0)

# SNANA's KEYWORD_SIMSED values (snlc_sim.c:6129, :6082, :23775).
KEYWORD_CONTINUOUS = 'SIMSED_PARAM'
KEYWORD_GRIDONLY = 'SIMSED_GRIDONLY'
# init_GENGAUSS_ASYM(0) then the wide-open range and flat sigma SNANA gives
# every SIMSED parameter (snlc_sim.c:945-950)
SIMSED_GENGAUSS_INIT = {'peak': 0.0, 'sigma': (1.0e8, 1.0e8),
                        'range': (-1.0e8, 1.0e8)}
FLAT_SIGMA = 1.0e7
KEYWORD_BAGGAGE = 'SIMSED_param'


def _is_index_simsed(parname):
    """IS_INDEX_SIMSED (genmag_SIMSED.c:1029-1037)."""
    return any(s in parname for s in ('INDEX', 'INDX', 'index', 'indx'))


def _redcor_conditional(x1, peak1, sig1, peak2, sig2, rho):
    """Mean and width of the conditional law of par2 given par1 = x1."""
    sig_cond = sig2 * np.sqrt(1.0 - rho * rho)
    mu_cond = peak2 + (rho * sig2 / sig1) * (x1 - peak1)
    return mu_cond, sig_cond


def _redcor_marginal_cdf(peak1, sig1, range1, peak2, sig2, range2, rho):
    """Tabulated unnormalised CDF of par1's rectangle-truncated marginal.

    Truncating par2 reweights par1, so par1's marginal is not a truncated
    univariate Gaussian but

        w(x1) = phi((x1-peak1)/sig1) * [Phi(b(x1)) - Phi(a(x1))]

    with a, b the range2 bounds standardised against the conditional
    N(peak2 + rho*(sig2/sig1)*(x1-peak1), sig2^2*(1-rho^2)). That has no
    closed form, so it is tabulated on N_GRID_SIMSED_REDCOR points.

    Returns (grid, cdf); cdf[-1] is the normalisation, which construction
    validates as finite and positive.
    """
    grid = jnp.linspace(float(range1[0]), float(range1[1]),
                        N_GRID_SIMSED_REDCOR)
    mu_cond, sig_cond = _redcor_conditional(grid, peak1, sig1, peak2, sig2, rho)
    lo_cdf = jax.scipy.special.ndtr((float(range2[0]) - mu_cond) / sig_cond)
    hi_cdf = jax.scipy.special.ndtr((float(range2[1]) - mu_cond) / sig_cond)
    weight = jnp.exp(-0.5 * ((grid - peak1) / sig1) ** 2) * (hi_cdf - lo_cdf)
    cdf = jnp.concatenate([
        jnp.zeros(1),
        jnp.cumsum(0.5 * (weight[1:] + weight[:-1])),
    ])
    return grid, cdf


def _draw_correlated_truncated(peak1, sig1, range1, peak2, sig2, range2,
                               rho, u1, u2):
    """Draw from a bivariate Gaussian truncated to range1 x range2.

    SNANA draws a Cholesky-correlated pair and redraws BOTH parameters whenever
    either lands outside its GENRANGE, up to 100 tries (snlc_sim.c:14165,
    PICK_RANCOV). That rejection defines a rectangle-truncated bivariate normal,
    which is sampled here by inverse CDF instead of by looping: par1 by
    inverting the tabulated marginal of _redcor_marginal_cdf, par2 in closed
    form from its truncated conditional.

    Construction validates sig1, sig2 > 0, |rho| < 1, ordered ranges, and a
    finite positive marginal normalisation (SIMSEDModel._parse_redcor), so no
    degenerate law reaches this kernel.
    """
    grid, cdf = _redcor_marginal_cdf(peak1, sig1, range1, peak2, sig2,
                                     range2, rho)
    x1 = jnp.interp(u1, cdf / cdf[-1], grid)

    mu_cond, sig_cond = _redcor_conditional(x1, peak1, sig1, peak2, sig2, rho)
    lo2, hi2 = float(range2[0]), float(range2[1])
    lo_cdf1 = jax.scipy.special.ndtr((lo2 - mu_cond) / sig_cond)
    hi_cdf1 = jax.scipy.special.ndtr((hi2 - mu_cond) / sig_cond)
    # clip keeps ndtri finite where the float32 tails coincide
    q = jnp.clip(lo_cdf1 + u2 * (hi_cdf1 - lo_cdf1), 1.0e-12, 1.0 - 1.0e-12)
    x2 = mu_cond + sig_cond * jax.scipy.special.ndtri(q)
    return x1, jnp.clip(x2, lo2, hi2)


def _sed_binary_records(raw: bytes):
    """Each SED's SEDBINARY floats, under either layout SNANA has written.

    Before FIX_SIZE_ARG_BINARY (genmag_SIMSED.h): IVERSION and MXDAY as
    8-byte ints (sizeof(int*), genmag_SIMSED.c:522, :1135), then per SED a
    300-byte path (MXPATHLEN, sndata.h:120), int NSEDBINARY and the floats.
    Since: 4-byte ints, then per SED int NCHAR, the path's NCHAR bytes, int
    NSEDBINARY and the floats. IVERSION is 4 in both. The old 8-byte writes
    come from 4-byte ints, so their high halves hold whatever the struct
    keeps next; bytes 12 to 16 tell the layout, being the start of the path
    only under the new one.
    """
    # bytes 12 to 16 start the path only under the 4-byte-int layout
    new_layout = all(0x20 <= b < 0x7f for b in raw[12:16])
    offset = 8 if new_layout else 16
    while offset < len(raw):
        if new_layout:
            nchar = struct.unpack_from('i', raw, offset)[0]
            offset += 4 + nchar
        else:
            offset += 300
        n_floats = struct.unpack_from('i', raw, offset)[0]
        offset += 4
        yield np.frombuffer(raw, dtype=np.float32, count=n_floats, offset=offset)
        offset += 4 * n_floats


def read_sed_binary(binary_path: str, n_seds: int
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a SNANA ``SED.BINARY`` file containing all grid-point SEDs.

    Phases in the binary already use the configured phase origin, and
    fluxes include the model's ``FLUXSCALE``.

    Returns ``(all_phases, wavelengths, all_flux)``:
        - all_phases: (n_seds, n_phases) per-template phase grids (days),
          using the configured phase origin
        - wavelengths: (n_wavelengths,) wavelength grid (Angstroms)
        - all_flux: (n_seds, n_phases, n_wavelengths)
    """
    with open(binary_path, 'rb') as f:
        raw = f.read()
    iversion = struct.unpack_from('i', raw, 0)[0]
    if iversion != 4:
        raise SNANAError(
            f"{binary_path} is SED.BINARY format version {iversion}; only version "
            f"4 is read. Delete it so SNANA rewrites it, or set SIMSED_USE_BINARY "
            f"0.",
            function_name="read_sed_binary")
    records = _sed_binary_records(raw)

    # pack_SEDBINARY layout; the last flux value is zero-padded
    data0 = next(records)
    nday = int(data0[2])
    padword = float(data0[5])
    if abs(padword - 77777.0) > 0.1:
        raise SNANAError(f"SED.BINARY PADWORD={padword}, expected 77777. "
                         f"Binary may be corrupt or incompatible version.",
                         function_name="read_sed_binary")

    all_phases = np.zeros((n_seds, nday), dtype=np.float64)
    all_phases[0] = data0[6:6 + nday].astype(np.float64)
    off2 = 6 + nday
    nlam = int(data0[off2])
    wavelengths = data0[off2 + 2:off2 + 2 + nlam].astype(np.float64)
    flux_start = off2 + 2 + nlam

    all_flux = np.zeros((n_seds, nday, nlam), dtype=np.float32)
    flux_avail = data0[flux_start:]
    all_flux[0].flat[:len(flux_avail)] = flux_avail

    for ised in range(1, n_seds):
        data = next(records)
        ised_nday = int(data[2])
        if ised_nday != nday:
            raise SNANAError(f"SED {ised} has NDAY={ised_nday}, expected {nday}",
                             function_name="read_sed_binary")
        all_phases[ised] = data[6:6 + nday].astype(np.float64)
        flux_avail = data[flux_start:]
        all_flux[ised].flat[:len(flux_avail)] = flux_avail

    return all_phases, wavelengths, all_flux


def read_sed_info(info_filepath: str) -> Dict:
    """Parse a SIMSED ``SED.INFO`` metadata file.

    The file is SNANA's index of a tabulated model directory, one ``KEY:``
    line each: ``NPAR:`` the number of parameters the grid spans,
    ``PARNAMES:`` their names, ``FLUX_SCALE:`` a factor applied to every
    tabulated flux, ``RESTLAMBDA_RANGE:`` the wavelengths in Angstroms the
    surfaces are read within, ``MAG_ERR:`` the model magnitude error, and
    ``OPTFLAG_T0SHIFT_PEAKMAG:`` 0 to keep each file's own phase zero rather
    than moving it to the bolometric peak; then one ``SED:`` line per
    surface giving the file name and that surface's value of each
    parameter, in ``PARNAMES`` order. Each surface file is three columns,
    phase in days, wavelength in Angstroms and flux density, as
    :func:`jax_snlc_core.models.sed_model.read_sed_file` reads them.

    Returns a dict with keys: ``npar`` (int), ``parnames`` (list[str]),
    ``flux_scale`` (float), ``restlambda_range`` (tuple[float, float]),
    ``mag_err`` (float), ``t0shift_peakmag`` (bool) and ``sed_entries``
    (list[{filename, par_values}]).
    """
    result: Dict = {
        'npar': 0,
        'parnames': [],
        'flux_scale': 1.0,
        'restlambda_range': SED_WAVELENGTH_RANGE,
        'mag_err': 0.0,
        't0shift_peakmag': True,
        'sed_entries': [],
    }

    with open(info_filepath, 'r') as f:
        for line in skip_docana(f):
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('$'):
                continue

            if line.startswith('NPAR:'):
                result['npar'] = int(line.split(':')[1].strip())
            elif line.startswith('PARNAMES:'):
                result['parnames'] = line.split(':')[1].strip().split()
            elif line.startswith('FLUX_SCALE:'):
                result['flux_scale'] = float(line.split(':')[1].strip())
            elif line.startswith('RESTLAMBDA_RANGE:'):
                parts = line.split(':')[1].strip().split()
                result['restlambda_range'] = (float(parts[0]), float(parts[1]))
            elif line.startswith('MAG_ERR:'):
                result['mag_err'] = float(line.split(':')[1].strip())
            elif line.startswith('OPTFLAG_T0SHIFT_PEAKMAG:'):
                # default on; 0 turns it off (genmag_SIMSED.c:848-853)
                result['t0shift_peakmag'] = \
                    int(line.split(':')[1].strip()) != 0
            elif line.startswith('SED:'):
                rest = line[4:].strip()
                parts = rest.split()
                filename = parts[0]
                par_values = [float(p) for p in parts[1:]]
                result['sed_entries'].append({
                    'filename': filename,
                    'par_values': par_values,
                })

    if result['npar'] > 0 and len(result['parnames']) != result['npar']:
        raise SNANAError(
            f"NPAR={result['npar']} but found {len(result['parnames'])} PARNAMES "
            f"in {info_filepath}",
            function_name="read_sed_info")
    for entry in result['sed_entries']:
        if len(entry['par_values']) != result['npar']:
            raise SNANAError(
                f"SED {entry['filename']} has {len(entry['par_values'])} parameter values "
                f"but NPAR={result['npar']} in {info_filepath}",
                function_name="read_sed_info")

    return result


@dataclass(eq=False, repr=False, kw_only=True)
class SIMSEDModel(SNModel, SimulatedModel):
    """SIMSED parameterized grid model.

    Loads all grid-point SEDs from a SIMSED model directory,
    builds the parameter grid axes, and provides vectorised SED
    generation via grid-point lookup or multilinear interpolation.
    """

    GENMODEL_ALIASES = ('simsed',)
    uses_nonia_rates = True
    mjd_explode_anchor = 'optmask'

    # read SED.BINARY instead of the text SEDs (snlc_sim.c:934)
    simsed_use_binary: Annotated[int, SIM_ONLY] = 0

    # SNANA's flux-table cache; jax keeps none
    simsed_path_binary: Annotated[str, SIM_ONLY] = field(
        default="", metadata={"static": True})

    # weights the choice of SED surface
    simsed_wgtmap_file: Annotated[str, SIM_ONLY] = field(
        default="", metadata={"static": True})

    # bright transients get extra headroom (check_crazyFlux, snlc_sim.c:28686)
    crazyflux_factor = 10.0

    wavelength_grid: jax.Array = None
    jax_template_seds: jax.Array = None
    jax_common_phases: jax.Array = None
    jax_template_phase_min: jax.Array = None
    jax_template_phase_max: jax.Array = None
    jax_late_time_slopes: jax.Array = None
    jax_late_boundary_seds: jax.Array = None
    sed_par_values: jax.Array = None
    grid_axes: jax.Array = None
    _corner_lookup: jax.Array = None
    _cont_axes: jax.Array = None
    _gridonly_lookup: jax.Array = None
    _gridonly_axes: jax.Array = None
    _gridonly_subset_seds: jax.Array = None
    _wgt_cumsum: jax.Array = None
    # The correlation's parameter names select the sampling calculation.
    _redcor: object = field(default=None, metadata={"static": True, "input": False})
    npar: int = field(default=0, metadata={"static": True, "input": False})
    parnames: tuple = field(default=(), metadata={"static": True, "input": False})
    par_keywords: tuple = field(default=(), metadata={"static": True, "input": False})
    sequential_gridonly: bool = field(default=False, metadata={"static": True, "input": False})
    index_parameter: int | None = field(default=None, metadata={"static": True, "input": False})

    @classmethod
    def load(cls, model_path, config, n_wave_points=500, wavelength_range=None):
        model = cls()
        model._load(model_path, config, n_wave_points, wavelength_range)
        return model

    def _load(self, model_path: str, config, n_wave_points=500,
                 wavelength_range=None):
        """Read SED.INFO and the SED payload that SIMSED_USE_BINARY selects,
        reanchor the phase grids where asked, extend into the UV, and resample
        onto the working grid. ``model_path`` is the SIMSED model directory
        containing SED.INFO and the .SED/.SED.gz template files.
        """
        super().__init__("simsed", config, n_wave_points,
                         wavelength_range)

        self._model_dir = Path(os.path.expandvars(model_path)).expanduser()
        info_file = self._model_dir / "SED.INFO"
        if not info_file.exists():
            raise SNANAError(f"SED.INFO not found in {self._model_dir}",
                             function_name="SIMSEDModel")

        logger.info(f"Loading SIMSED model from {self._model_dir.name}")

        self.sed_info = read_sed_info(str(info_file))
        self.npar = self.sed_info['npar']
        self.parnames = tuple(self.sed_info['parnames'])
        self._declared_parnames: List[str] = []
        self.flux_scale = self.sed_info['flux_scale']
        self.restlambda_range = self.sed_info['restlambda_range']
        self.mag_err = self.sed_info['mag_err']

        n_seds = len(self.sed_info['sed_entries'])
        log.line(logger, f"NPAR={self.npar}, PARNAMES={self.parnames}")
        log.line(logger, f"{n_seds} SED grid points, FLUX_SCALE={self.flux_scale}")

        self.sed_par_values = np.array(
            [e['par_values'] for e in self.sed_info['sed_entries']], dtype=np.float64
        )

        self.grid_axes = []
        for i in range(self.npar):
            unique_vals = np.unique(self.sed_par_values[:, i])
            self.grid_axes.append(unique_vals)

        # KEYWORD_SIMSED[ipar]; undeclared parameters stay baggage (:23410).
        self.par_keywords = (KEYWORD_BAGGAGE,) * self.npar
        self.sequential_gridonly = False
        # the last index-named parameter is the template index SNANA reports
        index_parameters = [i for i, name in enumerate(self.parnames)
                            if _is_index_simsed(name)]
        self.index_parameter = index_parameters[-1] if index_parameters else None

        self._corner_lookup = None
        self._cont_axes = None
        self._redcor = None
        self._gridonly_lookup = None
        self._gridonly_axes = None
        self._gridonly_subset_seds = None
        self.wgtmap = None
        self._wgt_cumsum = None

        if self.simsed_path_binary:
            log.line(logger, f"SIMSED_PATH_BINARY {self.simsed_path_binary} names "
                     f"SNANA's flux-table cache; jax keeps no such cache")
        self._load_sed_data(self.simsed_use_binary)

        # SNANA applies the explosion reanchor only on the text-SED read
        # (genmag_SIMSED.c:347-362); binaries keep their packed shift
        if self.optmask_t0shift_explode >= 1 and self._native_grid_mode == 'text':
            self._reanchor_to_explosion(self.optmask_t0shift_explode)
        del self._native_phases, self._native_flux

        if self.uvlam_extrapflux is not None:
            self._apply_uvlam_extrap(self.uvlam_extrapflux)
        self.resample_to_grid(self.choose_grid())

    def _load_sed_data(self, use_binary: int) -> None:
        """Read the SED payload. SNANA reads SED.BINARY only when
        SIMSED_USE_BINARY is set (genmag_SIMSED.c:248); otherwise the text
        SEDs are authoritative.
        """
        n_seds = len(self.sed_info['sed_entries'])
        binary_file = self._model_dir / "SED.BINARY"
        if use_binary and binary_file.exists():
            t0 = time.time()
            all_phases, wavelengths, all_flux = read_sed_binary(str(binary_file), n_seds)
            dt = time.time() - t0
            log.line(logger, f"Read SED.BINARY: {n_seds} SEDs, "
                     f"{all_phases.shape[1]} phases x {len(wavelengths)} wavelengths "
                     f"in {dt:.2f}s")
            self.native_wavelengths = wavelengths
            # FLUX_SCALE already applied in binary
            self.flux_scale = 1.0
            self._native_phases = [all_phases[i] for i in range(n_seds)]
            self._native_flux = [all_flux[i] for i in range(n_seds)]
            self._native_grid_step = float(all_phases[0, 1] - all_phases[0, 0])
            self._native_grid_mode = 'binary'
            self._resample_native_to_common()
        else:
            log.line(logger, f"Reading {n_seds} text SED files "
                     f"(SIMSED_USE_BINARY={use_binary})...")
            all_phases = []
            template_data = []

            for entry in self.sed_info['sed_entries']:
                sed_path = resolve_fits_path(
                    self._model_dir / entry['filename'], required=True)

                # SIMSED allows non-uniform day bins
                # (OPTMASK_DAYLIST_SEDMODEL, genmag_SIMSED.c:227-229)
                p, w, fg = read_sed_file(
                    sed_path, SIMSED_DAY_RANGE, self.sed_info['restlambda_range'],
                    t0shift_peakmag=self.sed_info['t0shift_peakmag'],
                    uniform_days=False)
                template_data.append((p, w, fg))
                all_phases.append(p)

            self.native_wavelengths = template_data[0][1]
            self._native_phases = list(all_phases)
            self._native_flux = [fg for (p, w, fg) in template_data]
            self._native_grid_step = 0.2
            self._native_grid_mode = 'text'
            self._resample_native_to_common()

        log.line(logger, f"Template library shape: {self.template_seds.shape}")

    def set_parameter_keywords(self, continuous_parnames: List[str],
                               _gridonly_parnames: List[str]) -> None:
        """Record which input key declared each SED.INFO parameter.

        Parameters named by neither key are baggage: SNANA generates nothing
        for them but still writes their grid values (snlc_sim.c:23769).
        """
        # SIMSED_GRIDONLY: SEQUENTIAL names no parameter and zeroes NPAR_SIMSED
        # (snlc_sim.c:6064), so no parameter columns are written.
        self.sequential_gridonly = 'SEQUENTIAL' in _gridonly_parnames
        # NGEN = NGENTOT = NSED, NGEN_LC = 0 (snlc_sim.c:23835-23843)
        self.forced_ngen = (len(self.sed_info['sed_entries'])
                            if self.sequential_gridonly else None)
        # input-declaration order, which the outputs follow (snlc_sim.c:6127);
        # SEQUENTIAL is a placeholder, not a parameter
        self._declared_parnames = list(continuous_parnames) + [
            n for n in _gridonly_parnames if n != 'SEQUENTIAL']
        keywords = []
        for name in self.parnames:
            if name in _gridonly_parnames:
                keywords.append(KEYWORD_GRIDONLY)
            elif name in continuous_parnames:
                keywords.append(KEYWORD_CONTINUOUS)
            else:
                keywords.append(KEYWORD_BAGGAGE)
        self.par_keywords = tuple(keywords)

    def _build_corner_lookup(self):
        """Dense map from continuous grid indices to flat SED index.

        SNANA aborts when a hypercube corner is filled twice or never
        (genmag_SIMSED.c:2057-2086); validating the full Cartesian grid here
        keeps the per-batch corner lookup a plain array index with no
        dictionary fallback.
        """
        names = self.continuous_parnames
        if not names:
            self._corner_lookup = None
            self._cont_axes = None
            return
        self._corner_lookup, self._cont_axes = self._build_grid_lookup(names)

    def _build_grid_lookup(self, names):
        """(dense index-to-SED lookup, axes) over the named parameters."""
        axis_ids = [self.parnames.index(n) for n in names]
        axes = [np.asarray(self.grid_axes[i]) for i in axis_ids]
        shape = tuple(len(ax) for ax in axes)
        lookup = np.full(shape, -1, dtype=np.int32)
        for sed_idx, par_vals in enumerate(self.sed_par_values):
            key = tuple(int(np.argmin(np.abs(ax - par_vals[ax_id])))
                        for ax, ax_id in zip(axes, axis_ids))
            if lookup[key] >= 0:
                raise SNANAError(
                    f"SIMSED grid point {key} filled twice (SED entries "
                    f"{int(lookup[key])} and {sed_idx}): two data points have the same "
                    f"coordinates.",
                    function_name="_build_grid_lookup")
            lookup[key] = sed_idx
        missing = np.argwhere(lookup < 0)
        if missing.size:
            raise SNANAError(
                f"SIMSED grid is missing {len(missing)} of {lookup.size} hypercube "
                f"corners (first missing grid index "
                f"{tuple(int(v) for v in missing[0])}); SNANA aborts on unfilled "
                f"corners.",
                function_name="_build_grid_lookup")
        return lookup, axes

    def _parse_redcor(self, simsed_cfg, snana_config):
        """Validate SIMSED_REDCOR/SIMSED_COV once and return the fixed draw
        parameters, or None when no non-zero correlation is configured.

        SNANA resolves both names against the declared SIMSED parameters and
        aborts unless exactly two match (snlc_sim.c:6321-6334), requires a
        symmetric Gaussian (snlc_sim.c:6355-6362), and aborts on an invalid
        reduced correlation (snlc_sim.c:6374-6380). The inverse-CDF sampler
        additionally needs positive widths, |rho| strictly below one, ordered
        truncation ranges, and a finite positive marginal normalisation; a
        violation of any of these is a degenerate law, so it is fatal here.
        """
        redcor = {pair: entry for pair, entry in simsed_cfg['redcor'].items()
                  if abs(entry[0]) > 0}
        if not redcor:
            return None
        continuous_parnames = self.continuous_parnames
        if len(continuous_parnames) != 2:
            raise SNANAError(
                f"SIMSED_REDCOR/SIMSED_COV needs exactly two continuous "
                f"SIMSED parameters; this model has "
                f"{len(continuous_parnames)} "
                f"({', '.join(continuous_parnames) or 'none'}).",
                function_name="_parse_redcor")
        p1, p2 = continuous_parnames
        for pair in redcor:
            unknown = [n for n in pair if n not in continuous_parnames]
            if unknown:
                raise SNANAError(
                    f"SIMSED_REDCOR/SIMSED_COV names "
                    f"{', '.join(repr(n) for n in unknown)}, which is not a "
                    f"continuous SIMSED parameter. Declared: "
                    f"{', '.join(continuous_parnames)}.",
                    function_name="_parse_redcor")
        value, is_cov = (redcor[(p1, p2)] if (p1, p2) in redcor
                         else redcor[(p2, p1)])
        populations = snana_config['populations']
        for name in (p1, p2):
            if len(populations[f'simsed_{name}'].distributions) > 1:
                raise SNANAError(
                    f"SIMSED_REDCOR/SIMSED_COV builds its correlated draw "
                    f"from the declared widths of '{p1}' and '{p2}', which "
                    f"SUBPOP classes give one set each. Drop the class "
                    f"setting for {name} or the correlation.",
                    function_name="_parse_redcor")
        d1, d2 = (populations[f'simsed_{p1}'].distributions[0],
                  populations[f'simsed_{p2}'].distributions[0])
        for name, dist in ((p1, d1), (p2, d2)):
            if dist.sigma[0] != dist.sigma[1]:
                raise SNANAError(
                    f"SIMSED_REDCOR/SIMSED_COV requires a symmetric "
                    f"Gaussian, but GENSIGMA_{name} is "
                    f"{dist.sigma[0]} {dist.sigma[1]}.",
                    function_name="_parse_redcor")
            if not dist.sigma[0] > 0:
                raise SNANAError(
                    f"SIMSED_REDCOR/SIMSED_COV requires a positive width, "
                    f"but GENSIGMA_{name} is {dist.sigma[0]}.",
                    function_name="_parse_redcor")
            if not dist.range[0] < dist.range[1]:
                raise SNANAError(
                    f"SIMSED_REDCOR/SIMSED_COV requires an ordered truncation "
                    f"range, but GENRANGE_{name} is "
                    f"{dist.range[0]} {dist.range[1]}.",
                    function_name="_parse_redcor")
        sig1, sig2 = float(d1.sigma[0]), float(d2.sigma[0])
        # SNANA keeps both forms as covariance and divides by the sigmas to
        # validate the reduced correlation (snlc_sim.c:6365-6380)
        rho = value / (sig1 * sig2) if is_cov else value
        if not abs(rho) < 1.0:
            raise SNANAError(
                f"Invalid reduced correlation {rho:.6f} between '{p1}' and '{p2}'; "
                f"the truncated conditional law needs |rho| < 1.",
                function_name="_parse_redcor")
        rc = {'p1': p1, 'p2': p2, 'rho': rho,
              'peak1': float(d1.peak), 'sig1': sig1,
              'range1': (float(d1.range[0]), float(d1.range[1])),
              'peak2': float(d2.peak), 'sig2': sig2,
              'range2': (float(d2.range[0]), float(d2.range[1]))}
        _, cdf = _redcor_marginal_cdf(
            rc['peak1'], rc['sig1'], rc['range1'],
            rc['peak2'], rc['sig2'], rc['range2'], rho)
        norm = float(cdf[-1])
        if not (np.isfinite(norm) and norm > 0.0):
            raise SNANAError(
                f"SIMSED_REDCOR/SIMSED_COV truncated marginal for '{p1}' has "
                f"normalisation {norm}: GENRANGE_{p1} x GENRANGE_{p2} "
                f"carries no probability mass under the configured "
                f"Gaussians.",
                function_name="_parse_redcor")
        return rc

    @classmethod
    def structured_input(cls, raw):
        """The SIMSED parameter declaration: which parameters are continuous,
        which select a grid surface, and how the continuous ones correlate."""
        continuous = [str(v) for v in raw.all(
            'SIMSED_PARAM', 'SIMSED_SHAPEPAR', 'SIMSED_COLORPAR',
            'SIMSED_COLORLAW')]
        gridonly = [str(v) for v in raw.all('SIMSED_GRIDONLY')]

        # REDCOR is reduced, COV absolute (snana_manual.tex:12350-12352)
        redcor = {}
        for key in list(raw):
            match = re.fullmatch(r'SIMSED_(REDCOR|COV)\((\w+),(\w+)\)', key)
            if match is None:
                continue
            redcor[(match.group(2), match.group(3))] = (
                float(raw[key]), match.group(1) == 'COV')

        simsed_config = {
            'parnames': continuous + gridonly,
            '_gridonly_parnames': gridonly,
            'continuous_parnames': continuous,
            'redcor': redcor,
        }
        # Each parameter is drawn under a simsed_ prefix, since its name comes
        # from the model's file and could clash with a shared one such as AV.
        # The input spells it without the prefix, and each carries SNANA's
        # SIMSED init, a flat sigma over an open range (snlc_sim.c:945-950).
        names = cls._declared_names({'simsed_config': simsed_config}, 'parnames')
        return {'simsed_config': simsed_config,
                'population_parameters': tuple(f'simsed_{n}' for n in names),
                'population_aliases': {f'simsed_{n}': (n,) for n in names},
                'population_defaults': {f'simsed_{n}': SIMSED_GENGAUSS_INIT
                                        for n in names}}

    @classmethod
    def _declared_names(cls, structured_config, key):
        """Parameter names from the declaration, without the INDEX(a,b,c)
        subset suffix and without the SEQUENTIAL placeholder, which names no
        parameter (snlc_sim.c:6064)."""
        cfg = structured_config.get('simsed_config') or {}
        names = [str(n).partition('(')[0] for n in cfg.get(key) or []]
        return tuple(n for n in names if n != 'SEQUENTIAL')

    @property
    def _output_order(self):
        """Parameter indices in SNANA's output order: the declared ones as
        declared, then the baggage as SED.INFO lists it (snlc_sim.c:23769)."""
        declared = [self.parnames.index(n) for n in self._declared_parnames]
        return declared + [i for i in range(self.npar) if i not in declared]

    def text_sim_header_lines(self, hcol, params, i):
        lines = []
        # SIMSED reuses the SALT2x0 slot as a distance-modulus flux scale
        # (gen_modelPar_SIMSED, snlc_sim.c:14144).
        if 'SIM_SALT2x0' not in hcol and 'SIM_DLMU' in hcol:
            lines.append("SIM_SALT2x0: "
                         f"{10.0 ** (-0.4 * float(hcol['SIM_DLMU'][i])):.4e}\n")
        lines.append(f"SIMSED_NPAR: {self.npar}\n")
        for k in self._output_order:
            value = float(params[f'simsed_{self.parnames[k]}'][i])
            lines.append(f"{self._header_column_name(k)}: {value:.6f}\n")
        return lines

    def _node_values(self, grid_idx):
        """Every SIMSED parameter at the selected grid nodes, keyed as the
        population declares them."""
        return {f'simsed_{name}': jnp.asarray(self.sed_par_values[:, i])[grid_idx]
                for i, name in enumerate(self.parnames)}

    def replay_params_from_dump(self, dump_columns, head_columns, *, mu,
                                distributions):
        selection_names = (self._gridonly_parnames if self._is_gridonly
                           else self.continuous_parnames)
        missing = [n for n in selection_names if n not in dump_columns]
        if missing:
            raise SNANAError(f"Missing SIMSED parameters in dump: {missing}",
                             function_name="replay_params_from_dump")
        selection_values = {n: dump_columns[n].astype(np.float32)
                            for n in selection_names}
        params = {}

        if self._is_gridonly:
            # recorded values sit on grid nodes; SNANA's evaluation refuses
            # one farther than 1e-4 of the parameter range from its node
            params['grid_idx'] = np.asarray(
                self.gridonly_index(selection_values, tolerance=1e-4),
                dtype=np.int32)
            params.update(self._node_values(params['grid_idx']))
        else:
            corner_indices, corner_weights = self.find_corners_and_weights(
                selection_values)
            params['corner_indices'] = corner_indices
            params['corner_weights'] = corner_weights
        for parname in self.parnames:
            if parname in dump_columns:
                params[f'simsed_{parname}'] = dump_columns[parname].astype(
                    np.float32)
        return params, None

    def configure_generation(self, snana_config, n_simulated, frac_pec1a):
        """Apply GRIDONLY flags, parameter keywords, SIMSED_REDCOR, and the
        SIMSED WGTMAP: the state per-SN sampling draws from."""
        simsed_cfg = snana_config["simsed_config"]
        _gridonly_parnames = [
            self._parse_gridonly_entry(entry)
            for entry in simsed_cfg["_gridonly_parnames"]]
        self.set_parameter_keywords(simsed_cfg["continuous_parnames"],
                                    _gridonly_parnames)
        if _gridonly_parnames:
            log.line(logger, f"SIMSED GRIDONLY parameters: {_gridonly_parnames}")
        self._build_corner_lookup()
        self._check_continuous_ranges(snana_config['populations'])
        if self._gridonly_parnames:
            self._gridonly_lookup, self._gridonly_axes = \
                self._build_grid_lookup(self._gridonly_parnames)
        self._redcor = self._parse_redcor(simsed_cfg, snana_config)

        if self.simsed_wgtmap_file:
            self._load_wgtmap(self.simsed_wgtmap_file)
        self._wgt_cumsum = self._surface_wgt_cumsum()
        if (self._gridonly_lookup is not None and self._wgt_cumsum is None
                and not self.sequential_gridonly and self._gridonly_subset_seds is None):
            self._warn_gridonly_ranges(snana_config['populations'])

    def _warn_gridonly_ranges(self, populations):
        """Report ignored flat-draw ranges while the grid bounds are concrete."""
        for name, axis in zip(self._gridonly_parnames, self._gridonly_axes):
            population = populations.get(f'simsed_{name}')
            if population is None or not population.distributions:
                continue
            dist = population.distributions[0]
            if (dist.sigma[0] <= FLAT_SIGMA
                    or tuple(dist.range) == SIMSED_GENGAUSS_INIT['range']):
                continue
            half_bin = 0.5 * (axis[-1] - axis[0]) / max(len(axis) - 1, 1)
            lo, hi = axis[0] - half_bin, axis[-1] + half_bin
            logger.warning(
                f"GENRANGE_{name} {dist.range[0]:g} {dist.range[1]:g} is ignored: "
                f"with no GENSIGMA_{name} the GRIDONLY draw is flat over the grid, "
                f"{lo:g} to {hi:g}")

    def _check_continuous_ranges(self, populations):
        """A continuous parameter's GENRANGE, in every SUBPOP class, or its
        GENPDF grid, must lie within the model's grid, widened by 1e-8 of
        its span for boundary values (checkpar_SIMSED,
        snlc_sim.c:23705-23725); with no GENRANGE the open init range fails
        the same check. The draw stays inside its declared range, so this
        is the whole of the check and the per-batch corners clamp."""
        if self._cont_axes is None:
            return
        for name, axis in zip(self.continuous_parnames, self._cont_axes):
            pop = populations.get(f'simsed_{name}')
            if pop is None:
                continue
            if pop.genpdf is not None:
                x = pop.genpdf.x_grid
                spans = [(float(x[0]), float(x[-1]))]
            else:
                spans = [d.range for d in pop.distributions]
            eps = 1.0e-8 * (axis[-1] - axis[0])
            for lo, hi in spans:
                if lo < axis[0] - eps:
                    raise SNANAError(
                        f"User {name}-min = {lo:.3f} < model "
                        f"min-range={axis[0]:.3f}; check sim-input keyword "
                        f"GENRANGE_{name}",
                        function_name="_check_continuous_ranges")
                if hi > axis[-1] + eps:
                    raise SNANAError(
                        f"User {name}-max = {hi:.3f} > model "
                        f"max-range={axis[-1]:.3f}; check sim-input keyword "
                        f"GENRANGE_{name}",
                        function_name="_check_continuous_ranges")

    def _parse_gridonly_entry(self, entry):
        """``NAME`` or ``NAME(a,b,c)``; the subset form is only valid for the
        template-index parameter (snlc_sim.c:6226) and restricts selection to
        the listed index values."""
        if '(' not in entry:
            return entry
        name, _, rest = entry.partition('(')
        if not _is_index_simsed(name):
            raise SNANAError(
                f"Cannot select GRIDONLY subset for {name}",
                function_name="_parse_gridonly_entry")
        values = [float(v) for v in rest.rstrip(')').split(',')]
        ipar = self.parnames.index(name)
        col = self.sed_par_values[:, ipar]
        sed_ids = []
        for v in values:
            match = np.flatnonzero(col == v)
            if match.size != 1:
                raise SNANAError(
                    f"GRIDONLY subset value {v:g} matches {match.size} "
                    f"entries of {name}",
                    function_name="_parse_gridonly_entry")
            sed_ids.append(int(match[0]))
        self._gridonly_subset_seds = np.array(sed_ids, dtype=np.int32)
        return name

    def _reanchor_to_explosion(self, optmask):
        """Shift each template's native DAY grid so T=0 is the explosion epoch
        (SNANA T0shiftExplode_SEDMODEL, genmag_SEDtools.c), then rebuild the
        common-phase grid. Explosion is the day bin preceding the first
        wavelength bin to exceed 1% (OPTMASK&1) or 0.1% (OPTMASK&2) of the
        template's global flux max."""
        for i in range(len(self._native_phases)):
            tshift = compute_explosion_tshift(
                self._native_flux[i], self._native_phases[i], optmask)
            self._native_phases[i] = self._native_phases[i] - tshift
        self._resample_native_to_common()

    def _resample_native_to_common(self):
        """Build common_phases, template_seds and the per-template phase bounds
        from the native per-template grids.

        The common grid follows the load path: the SED.BINARY daystep, or
        0.2 d anchored on phase 0 for the text SEDs, whose per-template grids
        differ wherever T0shiftPeak does. Binary templates that already share
        one grid keep it. FLUX_SCALE is a scalar, so applying it here commutes
        with the resampling; SNANA applies it when it computes the magnitude.
        """
        phases_list = self._native_phases
        flux_list = self._native_flux
        n_seds = len(phases_list)
        n_wl = flux_list[0].shape[1]
        phase_min = min(p[0] for p in phases_list)
        phase_max = max(p[-1] for p in phases_list)
        step = self._native_grid_step
        if self._native_grid_mode == 'text':
            neg = np.arange(0, phase_min - step, -step)[::-1]
            pos = np.arange(step, phase_max + step, step)
            common = np.concatenate([neg, pos])
            common = common[(common >= phase_min) & (common <= phase_max)]
            log.line(logger, f"Common phase grid: {phase_min:.1f} to {phase_max:.1f} d, "
                     f"{len(common)} points (step {step} d)")
        elif np.allclose(np.array(phases_list), phases_list[0][None, :]):
            self.common_phases = phases_list[0]
            self.template_seds = np.array(flux_list)
            self.template_phase_min = np.array([p[0] for p in phases_list])
            self.template_phase_max = np.array([p[-1] for p in phases_list])
            return
        else:
            common = np.arange(phase_min, phase_max + step * 0.5, step)
            n_differ = int(np.sum(~np.all(
                np.isclose(np.array(phases_list), phases_list[0][None, :]), axis=1)))
            log.line(logger, f"{n_differ}/{n_seds} templates have different phase grids, "
                     f"resampling to common grid: {phase_min:.0f} to {phase_max:.0f} d, "
                     f"{len(common)} points (step {step:.0f} d)")
        seds = np.zeros((n_seds, len(common), n_wl), dtype=np.float32)
        for i, (p, fg) in enumerate(zip(phases_list, flux_list)):
            idx = np.searchsorted(p, common, side='right') - 1
            idx = np.clip(idx, 0, len(p) - 2)
            denom = np.maximum(p[idx + 1] - p[idx], 1e-30)
            frac = np.clip((common - p[idx]) / denom, 0.0, 1.0)
            seds[i] = (1 - frac)[:, None] * fg[idx] + frac[:, None] * fg[idx + 1]
            outside = (common < p[0]) | (common > p[-1])
            seds[i, outside] = 0.0
        if self.flux_scale != 1.0:
            seds *= self.flux_scale
        self.common_phases = common
        self.template_seds = seds
        self.template_phase_min = np.array([p[0] for p in phases_list])
        self.template_phase_max = np.array([p[-1] for p in phases_list])

    def derived_parameters(self, params, keys_for, **kwargs):
        """SIMSED per-SN surfaces, in one of two modes. GRIDONLY picks one
        template per event by SNANA's precedence: the SIMSED weights, a
        sequential pass, a listed subset, the GRIDONLY parameter draws
        snapped to the grid, or else a uniform index; a SIMSED_WGTMAP_FILE
        selects this way for a continuous model too. Continuous takes each
        drawn SIMSED parameter and finds the 2^N interpolation corners and
        weights, drawing a SIMSED_REDCOR pair itself, correlated.
        """
        continuous_parnames = self.continuous_parnames
        if self._is_gridonly and continuous_parnames:
            raise SNANAError(
                f"SIMSED model has both GRIDONLY ({self._gridonly_parnames}) "
                f"and continuous ({continuous_parnames}) parameters; "
                f"mixed-mode SIMSED is not supported.",
                function_name="derived_parameters")

        surface = {}
        # a SIMSED_WGTMAP_FILE selects whole surfaces even for a continuous
        # model, bypassing interpolation (LSED_SELECT, genmag_SIMSED.c:1529)
        if not continuous_parnames or self.simsed_wgtmap_file:
            n_grid = self.jax_template_seds.shape[0]
            grid_uniforms = jax.vmap(lambda key: jax.random.uniform(key))(keys_for('SIMSED_GRID'))
            wgt_sum = self._wgt_cumsum
            # one surface selection per event, by SNANA's precedence
            # (gen_modelPar_SIMSED, snlc_sim.c:14248-14257)
            if wgt_sum is not None:
                # pick_SIMSED_BY_WGT (genmag_SIMSED.c:1269-1307)
                draw = grid_uniforms * wgt_sum[-1]
                grid_idx = jnp.searchsorted(wgt_sum, draw)
            elif self.sequential_gridonly:
                global_idx = params['sn_global_idx']
                grid_idx = global_idx[0] + jnp.arange(global_idx.shape[0])
            elif self._gridonly_subset_seds is not None:
                # pick_gridval_SIMSED (snlc_sim.c:14288-14299)
                subset = jnp.asarray(self._gridonly_subset_seds)
                pick = jnp.floor(grid_uniforms * subset.shape[0]).astype(jnp.int32)
                grid_idx = subset[jnp.clip(pick, 0, subset.shape[0] - 1)]
            elif self._gridonly_lookup is not None:
                grid_idx = self.gridonly_index({
                    name: self._gridonly_draw(name, axis, params, keys_for)
                    for name, axis in zip(self._gridonly_parnames,
                                          self._gridonly_axes)})
            else:
                grid_idx = jnp.floor(grid_uniforms * n_grid).astype(jnp.int32)
            surface['grid_idx'] = jnp.clip(grid_idx, 0, n_grid - 1).astype(jnp.int32)
            surface.update(self._node_values(surface['grid_idx']))
            return surface

        if self._redcor is not None:
            rc = self._redcor
            u1 = jax.vmap(jax.random.uniform)(keys_for('SIMSED_REDCOR1'))
            u2 = jax.vmap(jax.random.uniform)(keys_for('SIMSED_REDCOR2'))
            v1, v2 = _draw_correlated_truncated(
                rc['peak1'], rc['sig1'], rc['range1'],
                rc['peak2'], rc['sig2'], rc['range2'], rc['rho'], u1, u2)
            simsed_par_values = {rc['p1']: v1, rc['p2']: v2}
        else:
            simsed_par_values = {parname: params[f'simsed_{parname}']
                                 for parname in continuous_parnames}

        surface['corner_indices'], surface['corner_weights'] = \
            self.find_corners_and_weights(simsed_par_values)
        for parname in continuous_parnames:
            surface[f'simsed_{parname}'] = simsed_par_values[parname]
        return surface

    @staticmethod
    def _gridonly_draw(name, axis, params, keys_for):
        """A GRIDONLY parameter's value before the snap: its population
        draw, or without one (see :meth:`drawn_parameters`) a draw over the
        grid widened by half a bin each side, so every node is equally
        likely whatever GENRANGE says (snlc_sim.c:23679-23696)."""
        if f'simsed_{name}' in params:
            return params[f'simsed_{name}']
        half_bin = 0.5 * (axis[-1] - axis[0]) / max(len(axis) - 1, 1)
        lo, hi = axis[0] - half_bin, axis[-1] + half_bin
        u = jax.vmap(jax.random.uniform)(keys_for(name))
        return lo + u * (hi - lo)

    def _load_wgtmap(self, path):
        self.wgtmap = read_wgtmap(os.path.expandvars(str(path)),
                                  {n.lower(): n for n in self.parnames})
        unknown = [v for v in self.wgtmap.varnames if v not in self.parnames]
        if unknown:
            raise SNANAError(
                f"SIMSED_WGTMAP_FILE dimensions {', '.join(unknown)} are not "
                f"SIMSED parameters ({', '.join(self.parnames)}).",
                function_name="_load_wgtmap")
        log.line(logger, f"SIMSED WGTMAP over {self.wgtmap.varnames}")

    def _surface_wgt_cumsum(self):
        """Cumulative per-surface WGT, or None when selection is uniform.

        Weights come from a WGT column in SED.INFO, or from an external
        SIMSED_WGTMAP_FILE interpolated at each surface's parameter values
        (set_SIMSED_WGT_SUM, genmag_SIMSED.c:1178).
        """
        wgt = None
        # WGT column names by substring (IS_WGT_SIMSED, genmag_SIMSED.c:1040);
        # a WGTMAP file overrides the column (set_SIMSED_WGT_SUM,
        # genmag_SIMSED.c:1189-1190)
        wgt_cols = [i for i, n in enumerate(self.parnames)
                    if any(s in n for s in ('WGT', 'wgt', 'weight', 'WEIGHT'))]
        if wgt_cols:
            wgt = self.sed_par_values[:, wgt_cols[0]]
        if self.wgtmap is not None:
            wm = self.wgtmap
            cols = [self.parnames.index(v) for v in wm.varnames]
            points = jnp.asarray(self.sed_par_values[:, cols], dtype=jnp.float32)
            wgt = np.asarray(interpolate_wgtmap(
                wm.wgt, jnp.asarray(wm.ranges), jnp.asarray(wm.nbins), points))
        if wgt is None:
            return None
        # float64: the cumulative weight spans many decades
        return jnp.asarray(np.cumsum(wgt), dtype=jnp.float64)

    @property
    def _is_gridonly(self) -> bool:
        """True if any parameter is GRIDONLY (discrete template selection)."""
        return KEYWORD_GRIDONLY in self.par_keywords

    def drawn_parameters(self, populations):
        """The parameters the simulator draws for :meth:`derived_parameters`.
        A continuous model's, but a SIMSED_REDCOR pair, which it draws
        together. Under GRIDONLY, where the parameter draws pick the surface,
        those with a Gaussian or a GENPDF map: a flat sigma, the init or a
        given one, draws over the grid instead. None where the surface is
        picked another way."""
        if self.continuous_parnames and not self.simsed_wgtmap_file:
            redcor = ({self._redcor['p1'], self._redcor['p2']}
                      if self._redcor is not None else set())
            return tuple(f'simsed_{n}' for n in self.continuous_parnames
                         if n not in redcor)
        if (self._wgt_cumsum is not None or self.sequential_gridonly
                or self._gridonly_subset_seds is not None
                or self._gridonly_lookup is None):
            return ()
        drawn = []
        for name in self._gridonly_parnames:
            population = populations.get(f'simsed_{name}')
            if population is None:
                continue
            if (not population.distributions
                    or population.distributions[0].sigma[0] <= FLAT_SIGMA):
                drawn.append(f'simsed_{name}')
        return tuple(drawn)

    @property
    def continuous_parnames(self) -> List[str]:
        """Parameter names that require continuous interpolation."""
        return [name for name, keyword in zip(self.parnames, self.par_keywords)
                if keyword == KEYWORD_CONTINUOUS]

    @property
    def _gridonly_parnames(self) -> List[str]:
        """Parameter names that select a template surface."""
        return [name for name, keyword in zip(self.parnames, self.par_keywords)
                if keyword == KEYWORD_GRIDONLY]

    def gridonly_index(self, values: Dict[str, jnp.ndarray],
                       tolerance: Optional[float] = None) -> jnp.ndarray:
        """Surface index for per-event GRIDONLY parameter values.

        Each value snaps to a bracketing node the SNANA way: frac < 0.5 keeps
        the lower node, so an exact tie takes the upper
        (nearest_gridval_SIMSED, genmag_SEDtools.c:2489-2493). With
        ``tolerance``, a value farther from its node than that fraction of the
        parameter range aborts, as SNANA's all-GRIDONLY evaluation does
        (genmag_SIMSED.c:1844-1876).
        """
        idx_nd = []
        for name, axis in zip(self._gridonly_parnames, self._gridonly_axes):
            ax = jnp.asarray(axis)
            vals = jnp.asarray(values[name])
            hi = jnp.clip(jnp.searchsorted(ax, vals), 1, ax.shape[0] - 1)
            frac = (vals - ax[hi - 1]) / (ax[hi] - ax[hi - 1])
            idx = jnp.where(frac < 0.5, hi - 1, hi)
            if tolerance is not None:
                # written so a NaN value counts as a violation
                bad = ~(jnp.abs(vals - ax[idx]) / (ax[-1] - ax[0]) < tolerance)
                if bool(jnp.any(bad)):
                    raise SNANAError(
                        f"GRIDONLY parameter {name}: {int(jnp.sum(bad))} values sit farther "
                        f"than {tolerance:g} of the parameter range from any grid node.",
                        function_name="gridonly_index")
            idx_nd.append(idx)
        return jnp.asarray(self._gridonly_lookup)[tuple(idx_nd)]

    def _apply_uvlam_extrap(self, uvlam: float) -> None:
        """UVLAM_EXTRAPFLUX on every SIMSED template.

        One call per template, as SNANA's does (genmag_SIMSED.c:365-366). Each
        (template, phase) row is a DAY of that template's surface, so the
        stacked call preserves the per-template, per-day anchors.
        """
        waves = self.native_wavelengths
        lam_min_orig = waves[0]
        n_seds, n_phases, n_wl = self.template_seds.shape
        new_waves, new_flux = uvlam_extrapflux_sedmodel(
            uvlam, waves, self.template_seds.reshape(n_seds * n_phases, n_wl))
        n_uv = len(new_waves) - n_wl

        self.native_wavelengths = new_waves
        self.template_seds = new_flux.reshape(n_seds, n_phases, len(new_waves))

        if n_uv:
            log.line(logger, f"SIMSED UV extrapolation: extended from {lam_min_orig:.0f} to "
                     f"{new_waves[0]:.0f} A ({n_uv} new bins)")

    def resample_to_grid(self, wavelength_grid: jnp.ndarray) -> None:
        """Resample all templates from native wavelengths onto
        ``wavelength_grid``, the simulator's working grid."""
        n_seds = self.template_seds.shape[0]
        n_wl_new = len(wavelength_grid)
        wl_grid_np = np.asarray(wavelength_grid)

        log.line(logger, f"Resampling {n_seds} SIMSED templates from "
                 f"{len(self.native_wavelengths)} to {n_wl_new} wavelengths...")

        resampled = resample_surfaces(self.native_wavelengths, self.template_seds,
                                      wl_grid_np).astype(np.float32)

        self.jax_template_seds = jnp.array(resampled)
        self.jax_common_phases = jnp.array(self.common_phases)
        self.jax_template_phase_min = jnp.array(self.template_phase_min)
        self.jax_template_phase_max = jnp.array(self.template_phase_max)

        # each template's DAYMAX bounds its samples on the common grid
        n_valid = np.sum(self.common_phases[None, :]
                         <= np.asarray(self.template_phase_max)[:, None], axis=1)
        late_slopes, late_boundary_seds = late_time_tables(
            np.broadcast_to(self.common_phases, (n_seds, len(self.common_phases))),
            resampled, n_valid, self.minslope_extrapmag_late)

        self.jax_late_time_slopes = jnp.array(late_slopes)
        self.jax_late_boundary_seds = jnp.array(late_boundary_seds)

        log.line(logger, f"Resampled SIMSED library shape: {resampled.shape}")
        log.line(logger, f"Late-time slopes: {late_slopes.shape} "
                 f"(mean={late_slopes[late_slopes>0].mean():.3f} mag/d)"
                 if np.any(late_slopes > 0) else f"Late-time slopes: {late_slopes.shape}")

    def find_corners_and_weights(self, par_values_dict):
        """Hypercube corner indices and multilinear weights, (n_sne, 2^N)
        each, for N continuous parameters; the indices are flat indices into
        ``template_seds`` and each event's weights sum to one.

        Values clamp to the grid. Every declared range was checked against
        it at construction (`_check_continuous_ranges`), so a value lies
        outside only by the float32 rounding of a grid edge, which SNANA
        also treats as on the edge (checkpar_SIMSED, snlc_sim.c:23709).
        """
        names = self.continuous_parnames
        lookup = jnp.asarray(self._corner_lookup)
        lo_indices, fractions = [], []
        for name, axis in zip(names, self._cont_axes):
            axis = jnp.asarray(axis)
            vals = jnp.clip(jnp.asarray(par_values_dict[name]),
                            axis[0], axis[-1])
            idx = jnp.clip(jnp.searchsorted(axis, vals, side='right') - 1,
                           0, axis.shape[0] - 2)
            lo_indices.append(idx)
            fractions.append(jnp.clip(
                (vals - axis[idx]) / (axis[idx + 1] - axis[idx]), 0.0, 1.0))

        corner_indices, corner_weights = [], []
        for corner in range(2 ** len(names)):
            weight = jnp.ones_like(fractions[0], dtype=jnp.float32)
            grid_idx = []
            for d in range(len(names)):
                if (corner >> d) & 1:
                    grid_idx.append(lo_indices[d] + 1)
                    weight = weight * fractions[d].astype(jnp.float32)
                else:
                    grid_idx.append(lo_indices[d])
                    weight = weight * (1.0 - fractions[d]).astype(jnp.float32)
            corner_indices.append(lookup[tuple(grid_idx)])
            corner_weights.append(weight)
        return (jnp.stack(corner_indices, axis=1),
                jnp.stack(corner_weights, axis=1))

    def get_wavelength_range(self) -> Tuple[float, float]:
        return (float(self.native_wavelengths.min()),
                float(self.native_wavelengths.max()))

    def sed_param_names(self):
        return ('grid_idx', 'corner_indices', 'corner_weights', 'mu')

    def phase_range(self):
        """The common phase grid every template is resampled onto."""
        return float(self.common_phases[0]), float(self.common_phases[-1])

    def generate_sed(self, phase: float, params: Dict[str, float],
                     static_data: Dict) -> jnp.ndarray:
        """The rest-frame SED at ``phase`` (days relative to peak), shape
        (n_wavelengths,) in f_lambda units.

        GRIDONLY mode does a single template lookup (like NON1ASED), reading
        ``params['grid_idx']`` and ``params['mu']``; continuous mode does
        multilinear interpolation over 2^N corners, reading
        ``params['corner_indices']``, ``params['corner_weights']`` and
        ``params['mu']``.
        """
        template_seds = self.jax_template_seds
        common_phases = self.jax_common_phases
        distance_scale = jnp.power(10.0, -0.4 * params['mu'])

        # zero before DAYMIN, extrapolate past DAYMAX (genmag_SEDtools.c)
        phase_lo = common_phases[0]
        in_range = (phase >= phase_lo)
        idx, frac = linear_bin(phase, common_phases)

        if 'corner_indices' in params:
            corner_indices = params['corner_indices']
            corner_weights = params['corner_weights']

            corner_at_phase = linear_between(template_seds[corner_indices, idx, :],
                                             template_seds[corner_indices, idx + 1, :], frac)

            corner_phase_min = self.jax_template_phase_min[corner_indices]
            corner_in_range = (phase >= corner_phase_min)
            corner_at_phase = corner_at_phase * corner_in_range[:, None]

            # each corner extrapolates on its own (get_flux_SEDMODEL)
            corner_at_phase = late_time_decay(
                corner_at_phase, phase, self.jax_template_phase_max[corner_indices],
                self.jax_late_time_slopes[corner_indices],
                self.jax_late_boundary_seds[corner_indices])

            sed = jnp.sum(corner_weights[:, None] * corner_at_phase, axis=0)
        else:
            grid_idx = params['grid_idx']
            sed = linear_between(template_seds[grid_idx, idx, :],
                                 template_seds[grid_idx, idx + 1, :], frac)

            in_range = in_range & (phase >= self.jax_template_phase_min[grid_idx])

            sed = late_time_decay(sed, phase, self.jax_template_phase_max[grid_idx],
                                  self.jax_late_time_slopes[grid_idx],
                                  self.jax_late_boundary_seds[grid_idx])

        sed = jnp.where(in_range, sed, 0.0)

        # Distance scaling only: no /hc, since templates are in f_lambda units
        sed = sed * distance_scale
        return sed

    def _header_column_name(self, index: int) -> str:
        """SNANA's column name for parameter `index` (snlc_sim.c:25281): the
        declaring keyword wrapped around the name. SIMSED_PARAM / SHAPEPAR /
        COLORPAR / COLORLAW all record `SIMSED_PARAM` (:6129)."""
        return f'{self.par_keywords[index]}({self.parnames[index]})'

    def get_snana_header_params(self, parameters: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Map internal params to SNANA FITS column names.

        Matches the SIMSED block at sntools_dataformat_fits.c:486-493; the
        writer drops these when WRFLAG_MODELPAR is off.
        """
        result = {}
        # GENLC.TEMPLATE_INDEX is the index parameter's value, else -9
        # (genmag_SIMSED.c:1659-1666)
        values = {name: np.asarray(parameters[f'simsed_{name}'])
                  for name in self.parnames}
        if self.index_parameter is None:
            index = np.full(len(values[self.parnames[0]]), -9)
        else:
            index = values[self.parnames[self.index_parameter]] + 0.01
        result['SIM_TEMPLATE_INDEX'] = index.astype(np.int32)
        if 'mu' in parameters:
            # SIMSED reuses SALT2x0 purely as a flux scale (snlc_sim.c:14144).
            result['SIMSED_SALT2x0'] = np.power(
                10.0, -0.4 * np.asarray(parameters['mu'], dtype=np.float64)
            ).astype(np.float32)
        for i in self._output_order:
            result[self._header_column_name(i)] = values[self.parnames[i]].astype(
                np.float32)
        return result

    @property
    def dump_columns(self) -> tuple:
        """Every SIMSED parameter, the baggage included (snlc_sim.c:23769)."""
        return tuple(self._header_column_name(i) for i in self._output_order)

    sim_model_index = 7                        # MODEL_SIMSED
    sim_type_name_prefix = 'SIMSED.'           # SIMSED.SNIa-91bg -> SNIa-91bg
    writes_model_params = True
    fits_param_block = 'SIMSED'
