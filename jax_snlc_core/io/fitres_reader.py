"""Reader for SNANA FITRES / M0DIF text tables.

The generic counterpart to the writer in ``snana_table.py``: parses the
``# KEY: value`` header, the ``VARNAMES:`` line and the ``SN:``/``ROW:`` data
rows into a pandas DataFrame with **dtypes preserved**: string columns such as
``FIELD``, ``BANDLIST`` and non-numeric ``CID`` stay strings. (This is why
``snana_dump_reader.load_snana_dump`` is unsuitable for FITRES: it coerces every
column to float.) ``usecols`` restricts the read to the needed columns, which
matters for the large biasCor samples SALT2mu/BBC consumes.
"""

from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from jax_snlc_core.io.fits_path import open_text
from jax_snlc_core.utils.error_handling import SNANAError

_ROWKEY = "_ROWKEY"


@dataclass(frozen=True)
class FitresTable:
    """A parsed FITRES/M0DIF table.

    ``df`` holds the columns named by ``VARNAMES:`` (dtypes
    inferred per column, with CID/SNID kept as strings); ``meta`` holds the
    ``# KEY: value`` header block
    above ``VARNAMES:``, each value as the string the file wrote, so a number
    is converted by the caller. Lines below ``VARNAMES:`` written as
    ``KEY = value`` are not header lines and are not in ``meta``.
    """

    #: Table data with the columns named by ``VARNAMES:``.
    df: pd.DataFrame
    #: The column names, in the order ``VARNAMES:`` gave them.
    varnames: List[str]
    #: The ``# KEY: value`` header block, values as strings.
    meta: Dict[str, str]
    #: Each source file's own ``VARNAMES:``, in file order, for a table read
    #: from several files whose columns were unioned; a single-file read has
    #: one entry.
    varnames_by_file: List[List[str]] = dc_field(default_factory=list)
    #: The ``SN:``, ``ROW:`` or ``GAL:`` word opening each data row, kept
    #: only by a single-file ``as_text`` read so the rows can be written out
    #: again as they came; None otherwise.
    rowkey: Optional[np.ndarray] = None

    @property
    def size(self) -> int:
        """The number of rows."""
        return len(self.df)

    def has(self, col: str) -> bool:
        """Whether the table carries a column of this name."""
        return col in self.df.columns

    def get(self, col: str, default=None) -> np.ndarray:
        """The column as a numpy array, or ``default`` where the table has no
        such column, so a reader can take an optional column in one step."""
        if col in self.df.columns:
            return self.df[col].to_numpy()
        return default


def _parse_header_line(s: str, meta: Dict[str, str]) -> None:
    body = s[1:].strip()
    if ":" in body:
        key, val = body.split(":", 1)
        meta[key.strip()] = val.strip()


def read_fitres_header(path: str) -> Dict[str, str]:
    """The ``# KEY: value`` comment header of a FITRES/M0DIF file, read up to
    the ``VARNAMES:`` line without parsing the table."""
    meta: Dict[str, str] = {}
    with open_text(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("VARNAMES:"):
                break
            if s.startswith("#"):
                _parse_header_line(s, meta)
    return meta


def read_fitres(path: str, usecols: Optional[Sequence[str]] = None,
                as_text: bool = False) -> FitresTable:
    """Parse one FITRES/M0DIF file (``path``, optionally gzipped) into a
    :class:`FitresTable`.

    ``usecols``, if given, restricts the read to these columns (plus the
    rowkey, which is always dropped); a requested column absent from
    ``VARNAMES:`` is an error. ``as_text`` keeps every value as its original
    token and records the row key, so rows can be re-emitted byte-for-byte
    (``cat_only``).
    """
    meta: Dict[str, str] = {}
    varnames: Optional[List[str]] = None

    with open_text(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("VARNAMES:"):
                varnames = s.split()[1:]
                break
            if s.startswith("#"):
                _parse_header_line(s, meta)

        if varnames is None:
            raise SNANAError(
                f"No VARNAMES line found in {path}",
                function_name="read_fitres",
            )

        if usecols is not None:
            missing = [c for c in usecols if c not in varnames]
            if missing:
                raise SNANAError(
                    f"Columns {missing} requested via usecols are not in {path}",
                    function_name="read_fitres",
                )
            read_names = [c for c in varnames if c in set(usecols)]
        else:
            read_names = None

        # column 0 is the SN:/ROW: row key
        df = pd.read_csv(
            f,
            sep=r"\s+",
            comment="#",
            header=None,
            names=[_ROWKEY] + varnames,
            usecols=read_names,
            skip_blank_lines=True,
            dtype=str if as_text else None,
            # IDs are tokens: numeric inference loses leading zeros. Converters
            # also keep identifier strings such as "NA" out of missing-value parsing.
            converters=None if as_text else {
                name: str for name in ("CID", "SNID") if name in varnames},
            keep_default_na=not as_text,
        )

    rowkey = df[_ROWKEY].to_numpy() if as_text and _ROWKEY in df.columns else None
    df = df.drop(columns=[_ROWKEY], errors="ignore")
    return FitresTable(df=df, varnames=list(df.columns), meta=meta,
                       varnames_by_file=[varnames], rowkey=rowkey)


def read_fitres_multi(paths: Sequence[str],
                      usecols: Optional[Sequence[str]] = None,
                      as_text: bool = False) -> FitresTable:
    """Read several FITRES files and concatenate their rows (SNANA ``datafile=a,b``).

    Columns are unioned across files; a column absent from one file is filled
    with NaN (or None under ``as_text``). ``meta`` is taken from the first file.
    """
    return concat_tables([read_fitres(p, usecols=usecols, as_text=as_text)
                          for p in paths])


def concat_tables(tables: Sequence[FitresTable]) -> FitresTable:
    """One table from several, rows in order (SNANA ``datafile=a,b``).

    Columns are unioned; a column absent from one table is filled with NaN.
    ``meta`` is taken from the first table.
    """
    df = pd.concat([t.df for t in tables], ignore_index=True)
    return FitresTable(df=df, varnames=list(df.columns), meta=tables[0].meta,
                       varnames_by_file=[v for t in tables for v in t.varnames_by_file])
