"""Readers for the file formats that more than one program, or a model class, reads.

Each submodule is imported directly by consumers. Model-specific parsers
(SALT file formats, NON1A.LIST, SIMSED.INFO/SED.BINARY) live inline with their
model classes in `jax_snlc_core.models`; a format only one program reads lives
in that program's package.
"""

from typing import Optional


def is_ignorefile(filename: Optional[str]) -> bool:
    """True when a configured file value means "no file".

    Mirrors SNANA's IGNOREFILE (sntools.c:4832-4853): None, the empty
    string, whitespace only, or NULL/NONE/BLANK in any case.
    """
    if filename is None:
        return True
    stripped = str(filename).strip()
    return stripped == '' or stripped.upper() in ('NONE', 'NULL', 'BLANK')
