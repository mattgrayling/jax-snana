"""Utility functions for parsing SNANA SIMGEN_DUMP files."""

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from jax_snlc_core.io.fits_path import open_text
from jax_snlc_core.utils.error_handling import SNANAError
from jax_snlc_core.utils import snana_log as slog

logger = logging.getLogger(__name__)


@dataclass
class SnanaDumpData:
    """Container for parsed SNANA SIMGEN_DUMP columns."""

    columns: Dict[str, np.ndarray]

    @property
    def size(self) -> int:
        """Number of entries in the dump."""
        if not self.columns:
            return 0
        first_key = next(iter(self.columns))
        return len(self.columns[first_key])

    def get(self, key: str, default=None):
        return self.columns.get(key, default)


def load_snana_dump(dump_path: str, dedup_cid: bool = False) -> SnanaDumpData:
    """Load a SNANA SIMGEN_DUMP file (``dump_path``) into a
    :class:`SnanaDumpData` of numpy arrays, one per column.

    SNANA reuses CIDs when a SN is prefiltered (NEPOCH < CUTWIN_NEPOCH) and
    regenerated, and SIMGEN_DUMPALL writes both entries; when ``dedup_cid``
    is True, only the entry with the highest CUTMASK per CID is kept.
    """
    varnames: List[str] = []
    data: Dict[str, List[float]] = {}

    with open_text(dump_path) as dump_file:
        for raw_line in dump_file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.upper().startswith("VARNAMES:"):
                _, varlist = line.split(":", 1)
                varnames = varlist.strip().split()
                data = {name: [] for name in varnames}
                continue

            if not line.startswith("SN:"):
                continue

            if not varnames:
                raise SNANAError("VARNAMES header not found in SNANA dump.",
                                 function_name="load_snana_dump")

            parts = line.split()
            values = parts[1:]
            if len(values) != len(varnames):
                raise SNANAError(
                    f"Expected {len(varnames)} values per row, found {len(values)} in line: {line}",
                    function_name="load_snana_dump")

            for name, value in zip(varnames, values):
                try:
                    data[name].append(float(value))
                except ValueError:
                    data[name].append(np.nan)

    columns = {name: np.array(vals) for name, vals in data.items()}

    if dedup_cid and 'CID' in columns and 'CUTMASK' in columns:
        cids = columns['CID'].astype(int)
        cutmasks = columns['CUTMASK'].astype(int)
        best_row, tied = {}, set()
        for i, (cid, cm) in enumerate(zip(cids, cutmasks)):
            if cid not in best_row or cm > cutmasks[best_row[cid]]:
                best_row[cid] = i
                tied.discard(cid)
            elif cm == cutmasks[best_row[cid]]:
                tied.add(cid)
        # rows tied at CUTMASK 0 are all rejects, so any one stands for them
        ambiguous = sorted(c for c in tied if cutmasks[best_row[c]] > 0)
        if ambiguous:
            cid = ambiguous[0]
            raise SNANAError(
                f"CID {cid} appears twice with CUTMASK {cutmasks[best_row[cid]]}, "
                "so neither row is the event: the sim reused the CID. Give the "
                "events distinct CIDs (CIDRAN_MIN/CIDRAN_MAX) and re-run the "
                "dump.", function_name='load_snana_dump')
        keep_indices = sorted(best_row.values())
        if len(keep_indices) < len(cids):
            n_removed = len(cids) - len(keep_indices)
            slog.line(logger, f"De-duplicated {n_removed} CID-reuse entries "
                      f"({len(cids)} -> {len(keep_indices)} rows)")
            columns = {name: arr[keep_indices] for name, arr in columns.items()}

    return SnanaDumpData(columns=columns)

