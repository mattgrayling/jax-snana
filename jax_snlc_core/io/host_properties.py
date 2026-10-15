"""The host-galaxy column names SNANA treats as one quantity under several
spellings.

SNANA rewrites each property basename to ``<BASE>_TRUE`` wherever it reads a
name: the HOSTLIB header, a WGTMAP's variables and ``HOSTLIB_STOREPAR``
(checkAlternateVarNames_HOSTLIB, sntools_host.c:3029, applied at :2481,
:1333, :1098), so ``LOGSFR`` and ``LOGSFR_TRUE`` always resolve to the same
column; and it reads the host coordinates under ``RA``, ``RA_HOST`` or
``RA_GAL`` and likewise for DEC (sntools_host.c:2740-2750). Here one name is
kept in memory for each, and every name read from an input goes through
:func:`canonical_property_name` first.
"""

#: sndata.h:84-87
HOSTGAL_PROPERTY_BASENAMES = ('LOGMASS', 'LOGSFR', 'LOGsSFR', 'COLOR')

#: sntools_host.h:116-121, each to the spelling kept in memory
HOST_COORDINATE_NAMES = {'RA': 'RA_GAL', 'RA_HOST': 'RA_GAL', 'RA_GAL': 'RA_GAL',
                         'DEC': 'DEC_GAL', 'DEC_HOST': 'DEC_GAL',
                         'DEC_GAL': 'DEC_GAL'}


def canonical_property_name(name: str) -> str:
    """The in-memory spelling of a host column: ``LOGMASS_TRUE`` and
    ``LOGMASS`` are both ``LOGMASS``, ``RA`` and ``RA_HOST`` are ``RA_GAL``;
    any other name is itself."""
    if name.endswith('_TRUE') and name[:-5] in HOSTGAL_PROPERTY_BASENAMES:
        return name[:-5]
    return HOST_COORDINATE_NAMES.get(name, name)
