"""Resolve SNANA file paths and open SNANA files, accepting ``.gz``.

SNANA gzips its output by default but references the plain name; CFITSIO
auto-opens the ``.gz`` when given the base name. astropy does not, so we
replicate that fallback. Single source of truth for ``<name>`` ->
``<name>.gz`` resolution across the codebase, for the model, template and
map files as well as the FITS ones.
"""

import gzip
from pathlib import Path

import numpy as np
from astropy.io import fits

from jax_snlc_core.utils.error_handling import SNANAError


def resolve_fits_path(path, required=False):
    """Return the file that exists on disk for ``path``, plain or ``.gz``.

    Prefers the plain file when both exist. With neither present, ``required``
    aborts naming both spellings; otherwise the original path passes through
    so the caller's own open raises.
    """
    path = Path(path)
    if path.exists():
        return str(path)
    gz_path = path.with_name(path.name + ".gz")
    if gz_path.exists():
        return str(gz_path)
    if required:
        raise SNANAError(f"file not found: {path} (also checked {path}.gz)",
                         function_name="resolve_fits_path")
    return str(path)


def read_fits_table(path):
    """The primary header and the binary table of a SNANA FITS file, plain
    or ``.gz``, as a structured array in the machine's byte order.

    astropy parses the two headers; the rows stream from the file into one
    buffer that is byte-swapped in place, so the table is held once and
    never copied. SNANA writes an empty primary HDU and one unscaled table;
    a ``TSCALn``/``TZEROn`` column is refused rather than read raw.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        primary = fits.Header.fromfile(stream)
        header = fits.Header.fromfile(stream)
        fields = range(1, header["TFIELDS"] + 1)
        scaled = [header[f"TTYPE{i}"] for i in fields
                  if f"TSCAL{i}" in header or f"TZERO{i}" in header]
        if scaled:
            raise SNANAError(f"{path}: scaled columns {scaled} are not "
                             "SNANA's format", function_name="read_fits_table")
        columns = fits.ColDefs([fits.Column(name=header[f"TTYPE{i}"],
                                            format=header[f"TFORM{i}"])
                                for i in fields])
        stored = columns.dtype.newbyteorder(">")
        rows = bytearray(header["NAXIS2"] * stored.itemsize)
        _read_into(stream, rows)
    table = np.frombuffer(rows, stored)
    # Keep a block in cache while NumPy visits its individual columns.
    block_rows = max(1, (1 << 20) // stored.itemsize)
    for start in range(0, len(table), block_rows):
        table[start:start + block_rows].byteswap(inplace=True)
    return primary, table.view(stored.newbyteorder("="))


def _read_into(stream, buffer, piece=1 << 20):
    """Fill ``buffer`` from ``stream`` a piece at a time: a gzip stream's
    ``readinto`` inflates through an intermediate bytes object of the size
    asked for, so the piece bounds that transient."""
    view = memoryview(buffer)
    done = 0
    while done < len(buffer):
        got = stream.readinto(view[done:done + piece])
        if not got:
            raise SNANAError(f"{stream.name}: file ends {len(buffer) - done} "
                             "bytes short of its table",
                             function_name="read_fits_table")
        done += got


def open_text(path):
    """Open a SNANA text file, accepting its plain name or ``.gz`` spelling."""
    path = Path(resolve_fits_path(path))
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path, "rt")
