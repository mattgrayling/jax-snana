"""Shared loaders for SIM_MAGOBS replay-parity tests.

Epochs are keyed by (snid, band, rank-of-MJD-within-band): the JAX scheduler
carries MJD through float32, so absolute MJDs differ from SNANA's float64
values at the ~1e-3 day level and cannot be joined on directly. Rank matching
is exact because both sides schedule the same SIMLIB epochs.
"""

from collections import defaultdict
from pathlib import Path

from astropy.io import fits


def load_snana_phot(head_path: Path, phot_path: Path):
    """Return ``{(snid, band, rank): (sim_magobs, flux, fluxerr, mjd)}``."""
    with fits.open(head_path) as h, fits.open(phot_path) as p:
        head = h[1].data
        phot = p[1].data
        epochs = defaultdict(list)
        for row in head:
            snid = int(row["SNID"])
            pmin, pmax = int(row["PTROBS_MIN"]), int(row["PTROBS_MAX"])
            for i in range(pmin - 1, pmax):
                pr = phot[i]
                mjd = float(pr["MJD"])
                if mjd < 0:
                    continue
                band = str(pr["BAND"]).strip()
                epochs[(snid, band)].append(
                    (mjd, float(pr["SIM_MAGOBS"]),
                     float(pr["FLUXCAL"]), float(pr["FLUXCALERR"])))
    out = {}
    for (snid, band), rows in epochs.items():
        rows.sort()
        for rank, (mjd, mag, flux, fluxerr) in enumerate(rows):
            out[(snid, band, rank)] = (mag, flux, fluxerr, mjd)
    return out


def load_jax_phot(jax_dir: Path, gv: str):
    """Same shape as ``load_snana_phot`` but for JAX's output."""
    head = next(jax_dir.glob(f"{gv}*HEAD*.FITS*"))
    phot = next(jax_dir.glob(f"{gv}*PHOT*.FITS*"))
    return load_snana_phot(head, phot)


def slice_replay(dump_columns, head_columns, n):
    """First ``n`` events of a replay, dump and HEAD kept aligned.

    build_override_parameters maps HEAD by SNID but zeroes SIM_SEARCHEFF_MASK
    when the two lengths disagree, so both must be cut to the same events.
    """
    import numpy as np
    keep = set(int(c) for c in np.asarray(dump_columns['CID'])[:n])
    dump_cut = {k: np.asarray(v)[:n] for k, v in dump_columns.items()}
    m = np.array([int(s) in keep for s in np.asarray(head_columns['SNID'])])
    return dump_cut, {k: np.asarray(v)[m] for k, v in head_columns.items()}


def select_replay(dump_columns, head_columns, cids):
    """Restrict a replay to ``cids``, dump and HEAD kept aligned."""
    import numpy as np
    keep = set(int(c) for c in cids)
    md = np.array([int(c) in keep for c in np.asarray(dump_columns['CID'])])
    mh = np.array([int(s) in keep for s in np.asarray(head_columns['SNID'])])
    return ({k: np.asarray(v)[md] for k, v in dump_columns.items()},
            {k: np.asarray(v)[mh] for k, v in head_columns.items()})
