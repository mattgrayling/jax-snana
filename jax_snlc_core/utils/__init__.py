"""Generic utilities used by more than one program.

Small helpers that don't fit a domain subpackage; cosmology, dust laws and the
mass step live in the sibling `jax_snlc_core.physics` subpackage.
"""

import os
from pathlib import Path

from .error_handling import SNANAError

# unset measured value (init_SNDATA_EVENT)
SNANA_NULLFLOAT = -9.0

# unset simulated magnitude
SIM_MAGOBS_UNSET = 99.0

# A host property no catalogue supplied (HOSTLIB_PROPERTY_UNDEFINED, sntools.h:131).
HOSTGAL_PROPERTY_NULL = -9999.0

# host slots HOSTGAL, HOSTGAL2, HOSTGAL3
MXHOSTGAL = 3
HOSTGAL_PREFIXES = ('HOSTGAL',) + tuple(f'HOSTGAL{k}' for k in range(2, MXHOSTGAL + 1))
# width of each slot's photo-z quantile vector in a data file (sndata.h:185)
MXBIN_HOSTGALz_QUANTILE = 20
# width of each slot's LOGMASS(z) grid in a data file (sndata.h:184)
MXBIN_HOSTGALz = 40

# MASK_REDSHIFT_SOURCE bits marking a spec-z: 1 host, 2 SN (sndata.h:106-107)
ZSPEC_SOURCE_MASK = 3


def sndata_root() -> Path:
    """$SNDATA_ROOT, which SURVEY.DEF, the models and the maps live under.
    SNANA reads it from the environment alone."""
    try:
        return Path(os.environ["SNDATA_ROOT"])
    except KeyError:
        raise SNANAError("$SNDATA_ROOT is not set", function_name="sndata_root") from None


__all__ = [
    "SNANAError",
    "SNANA_NULLFLOAT",
    "SIM_MAGOBS_UNSET",
    "sndata_root",
]


def group_equal(mapping):
    """The keys of ``mapping`` grouped by equal value, each group in the
    order of its first key. Values are compared rather than hashed, so they
    may be dicts or dataclasses holding them."""
    groups = []
    for key, value in mapping.items():
        group = next((keys for v, keys in groups if v == value), None)
        if group is None:
            groups.append((value, [key]))
        else:
            group.append(key)
    return [keys for _, keys in groups]
