"""SNANA FUDGE_HOSTNOISE_FILE (sntools_fluxErrModels_legacy.c): legacy host
surface-brightness noise. Per band/field, an ERRSCALE curve vs host SB mag
inflates the flux error (``err *= ERRSCALE``); SBMAG = ZP - 2.5 log10(SB
flux).
"""

import os

import numpy as np
import pandas as pd

from jax_snlc_core.io.fits_path import open_text
from jax_snlc_core.utils.error_handling import SNANAError


def read_hostnoise_file(path):
    """Parse a FUDGE_HOSTNOISE_FILE (INIT_NOISEMODEL_HOST_LEGACY) into per-LIBID
    maps: {band, field, HOSTMAG grid, ERRSCALE grid}."""
    maps, cur = [], None
    with open_text(os.path.expandvars(path)) as fh:
        for line in fh:
            toks = line.split('#')[0].split()
            if not toks:
                continue
            key = toks[0].rstrip(':')
            if key == 'LIBID':
                cur = {'band': '', 'field': 'ALL', 'mag': [], 'noise': []}
                maps.append(cur)
            elif key == 'BAND':
                cur['band'] = toks[1]
            elif key == 'FIELD':
                cur['field'] = toks[1]
            elif key == 'HOSTNOISE':
                cur['mag'].append(float(toks[1]))
                cur['noise'].append(float(toks[2]))
    for m in maps:
        m['mag'], m['noise'] = np.array(m['mag']), np.array(m['noise'])
    return maps


def _errscale(m, sbmag):
    """ERRSCALE of one map at each host SB mag: bright-end linear
    extrapolation, dim-end clamp to the last bin, floored at 1.0
    (GEN_NOISEMODEL_HOST_LEGACY)."""
    g, n = m['mag'], m['noise']
    bright = n[0] + (sbmag - g[0]) * (n[1] - n[0]) / (g[1] - g[0])
    val = np.where(sbmag < g[0], bright, np.interp(sbmag, g, n))
    return np.maximum(val, 1.0)


def sbmag_from_fluxcal(sb_fluxcal, zp_fluxcal):
    """Host surface-brightness magnitude from its FLUXCAL (SET_SBMAG,
    snana.F90:18032). An undefined surface brightness reads as 99."""
    sb_fluxcal = np.asarray(sb_fluxcal, dtype=float)
    return np.where(sb_fluxcal <= -999.0, 99.0,
                    zp_fluxcal - 2.5 * np.log10(np.maximum(sb_fluxcal, 1.0)))


def hostnoise_errscale(maps, bands, fields, sb_fluxcal, zp_fluxcal):
    """Per-epoch ERRSCALE from the host SB flux (HOSTGAL_SB_FLUXCAL) in each
    epoch's band. Band and field match SNANA's strstr semantics (a map entry
    contains the epoch value); epochs are grouped by (band, field) so each map
    interpolates once."""
    sbmag = sbmag_from_fluxcal(sb_fluxcal, zp_fluxcal)
    bands = np.asarray(bands)
    fields = (np.asarray(fields) if fields is not None
              else np.full(len(bands), ''))
    out = np.ones(len(bands))
    groups = pd.DataFrame({'band': bands.astype(str), 'field': fields.astype(str)})
    for (b, f), rows in groups.groupby(['band', 'field']).indices.items():
        matches = [m for m in maps if b in m['band'] and f in m['field']]
        if len(matches) != 1:
            raise SNANAError(
                f"Expected one host-noise map for band '{b}', field '{f}'; "
                f"found {len(matches)}. Check FUDGE_HOSTNOISE_FILE.",
                function_name="hostnoise_errscale")
        out[rows] = _errscale(matches[0], sbmag[rows])
    return out
