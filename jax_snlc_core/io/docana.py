"""DOCUMENTATION-block presence check (check_openFile_docana via
snana_openTextFile, sntools.c).

SNANA requires its data and config files to open with a ``DOCUMENTATION:`` block
so a published result can be traced back to its inputs.
"""

import logging
from typing import Iterable, Iterator

from jax_snlc_core.io.fits_path import open_text
from jax_snlc_core.utils import snana_log as slog
from jax_snlc_core.utils.error_handling import SNANAError

logger = logging.getLogger(__name__)

# sntools.h:45-46
KEYNAME_DOCANA = 'DOCUMENTATION:'
KEYNAME_DOCANA_END = 'DOCUMENTATION_END:'


def skip_docana(lines: Iterable[str]) -> Iterator[str]:
    """``lines`` with the DOCUMENTATION block removed.

    A line whose first whitespace-separated token is exactly
    ``DOCUMENTATION:`` opens the block and ``DOCUMENTATION_END:`` closes it;
    both keys and every line between them are dropped, as rd2columnFile does
    (sntools.c:7297-7303).
    """
    in_block = False
    for line in lines:
        tokens = line.split()
        if tokens and tokens[0] == KEYNAME_DOCANA:
            in_block = True
        elif tokens and tokens[0] == KEYNAME_DOCANA_END:
            in_block = False
        elif not in_block:
            yield line


def check_file_docana(path, require: bool = True, header_lines=None) -> None:
    """Abort (or warn) when ``path`` does not begin with ``DOCUMENTATION:``.

    ``header_lines`` selects the metadata check: the key may appear among
    the words of the first few lines. Otherwise it must be the first word.

    SNANA opens these files through snana_openTextFile, whose
    check_openFile_docana reads one fscanf("%s") token: the first
    whitespace-delimited word of the file, after any leading blank lines.
    """
    with open_text(path) as fh:
        if header_lines is None:
            first = next((words for words in map(str.split, fh) if words), [])
            found = bool(first) and first[0] == KEYNAME_DOCANA
        else:
            found = any(KEYNAME_DOCANA in fh.readline().split('#')[0].split()
                        for _ in range(header_lines))

    if found:
        return

    location = ('as its first word' if header_lines is None
                else f'in its first {header_lines} lines')
    detail = (f"{path} must contain '{KEYNAME_DOCANA}' {location}. See DOCANA examples "
              f"with: grep -R DOCUMENTATION_END $SNDATA_ROOT")
    if require:
        raise SNANAError(f"Missing DOCUMENTATION block. {detail}",
                         function_name="check_file_docana")
    slog.warn(logger, "DOCANA", detail)
