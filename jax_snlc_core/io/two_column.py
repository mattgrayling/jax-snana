"""SNANA 2-column ASCII table reader (sntools.c:7242 rd2columnFile).

SNANA reads HzFUN_FILE and DNDZ_FILE through this one routine, so both share
its comment and DOCUMENTATION conventions.
"""

from typing import Tuple

import numpy as np

from jax_snlc_core.io.docana import skip_docana
from jax_snlc_core.io.fits_path import open_text

# sntools.c:7341 commentchar()
_COMMENT_CHARS = '#!%@'


def read_two_column_file(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read the first two whitespace-separated columns of an ASCII table.

    Skips blank lines, rows whose first token starts with `#`, `!`, `%` or `@`,
    and `DOCUMENTATION:` ... `DOCUMENTATION_END:` blocks. Columns beyond the
    first two are ignored. Gzipped files are read transparently.
    """
    col1, col2 = [], []
    with open_text(path) as fh:
        for line in skip_docana(fh):
            tokens = line.split()
            if not tokens or tokens[0][0] in _COMMENT_CHARS:
                continue
            if len(tokens) < 2:
                continue
            col1.append(float(tokens[0]))
            col2.append(float(tokens[1]))
    return np.array(col1), np.array(col2)
