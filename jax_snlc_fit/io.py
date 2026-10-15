"""Data discovery for the fit run: an SNANA version's FITS files or a Rubin
HATS catalogue. Tables are built in `sntable` and written through
`jax_snlc_core.io.snana_table.write_varnames_table`."""

import glob
import logging
from functools import partial
from pathlib import Path

from jax_snlc_core.io.fits_path import resolve_fits_path
from jax_snlc_core.utils.error_handling import SNANAError

from .data_loader import read_rubin_hats, read_snana_fits

logger = logging.getLogger(__name__)

# the file marking a directory as a HATS catalogue
HATS_PROPERTIES = 'hats.properties'


def version_sources(sim_dir, column_overrides):
    """What one data version is read from, as ``(path, read)`` pairs whose
    ``read()`` gives the HEAD and PHOT tables and the header values: the
    catalogue itself for Rubin HATS, else each of its HEAD/PHOT pairs.
    ``column_overrides`` is PHOTOMETRY_COLUMNS."""
    if (Path(sim_dir) / HATS_PROPERTIES).is_file():
        return [(str(sim_dir),
                 partial(read_rubin_hats, sim_dir, column_overrides))]
    return [(head, partial(read_snana_fits, head, phot))
            for head, phot in discover_fits_files(sim_dir)]


def discover_fits_files(sim_dir):
    """The HEAD/PHOT pairs of a data version, in the order its ``.LIST``
    names them, or the one pair a directory without a ``.LIST`` holds.

    A version holds one HEAD per job or model that went into it and the
    ``.LIST`` names them all (WR_SNFITSIO_INIT, snlc_sim.c:30643); every one
    is read, since each event lives in exactly one. Gzipped files are
    accepted as SNANA writes by default.

    Returns a list of (head_path, phot_path).
    """
    sim_dir = Path(sim_dir)
    lists = glob.glob(str(sim_dir / "*.LIST"))
    if len(lists) > 1:
        raise SNANAError(f"{sim_dir} holds more than one .LIST: {lists}",
                         function_name="discover_fits_files")
    if lists:
        with open(lists[0]) as f:
            names = [line.strip() for line in f
                     if line.strip() and not line.startswith('#')]
        odd = [n for n in names
               if not n.endswith(("_HEAD.FITS", "_HEAD.FITS.gz"))]
        if odd:
            raise SNANAError(f"{lists[0]} names files that are not HEAD "
                             f"FITS: {odd}", function_name="discover_fits_files")
        head_files = [resolve_fits_path(sim_dir / n) for n in names]
    else:
        head_files = sorted(glob.glob(str(sim_dir / "*_HEAD.FITS*")))
    if not head_files:
        raise SNANAError(f"No *_HEAD.FITS files found in {sim_dir}",
                         function_name="discover_fits_files")

    pairs = []
    for head_path in head_files:
        # the PHOT file shares the HEAD's prefix (strip .gz before deriving it)
        base = Path(head_path).name
        if base.endswith(".gz"):
            base = base[:-3]
        prefix = base.replace("_HEAD.FITS", "")
        phot_path = resolve_fits_path(sim_dir / f"{prefix}_PHOT.FITS")
        if not Path(phot_path).exists():
            raise SNANAError(f"{head_path} has no {prefix}_PHOT.FITS beside it",
                             function_name="discover_fits_files")
        pairs.append((head_path, phot_path))
    return pairs
