"""
HOSTLIB (Host Galaxy Library) module for JAX-SNANA.

This module provides functionality for:
- Parsing HOSTLIB files (galaxy catalogs)
- Parsing WGTMAP files (weight maps for preferential selection)
- Computing WGTSUM for host galaxy selection (with caching)
- Selecting host galaxies for simulated SNe using JAX (vectorised, JIT-compiled)
- Computing host properties and SN positions within galaxies
"""

from jax_snlc_core.io.wgtmap_reader import read_wgtmap, read_wgtmap_varnames, WgtMapData
from .hostlib_reader import (HostlibTable, essential_columns, hostlib_declared,
                             read_hostlib)
from .sersic_sampling import compute_sersic_integral_table
from .wgtsum_computer import compute_wgtsum
from .hostlib_selection import (HostCatalogue, Hostlib, HostSelector,
                                build_host_selection, compute_dztol)
from .wronghost import WrongHostTable, read_wronghost_file, gen_zhost
from .zphot_generator import (
    GenZphot,
    ZphotEfficiency,
    build_genzphot,
    read_zphoteff_file,
    zphoterr_asym,
    generate_zphot_from_calc,
    generate_zphot_quantiles_gaussian,
    validate_genzphot_fudgemap,
)

__all__ = [
    'HostlibTable',
    'essential_columns',
    'hostlib_declared',
    'read_hostlib',
    'compute_sersic_integral_table',
    'read_wgtmap',
    'read_wgtmap_varnames',
    'WgtMapData',
    'compute_wgtsum',
    'HostCatalogue',
    'Hostlib',
    'HostSelector',
    'build_host_selection',
    'compute_dztol',
    'WrongHostTable',
    'read_wronghost_file',
    'gen_zhost',
    'GenZphot',
    'ZphotEfficiency',
    'build_genzphot',
    'read_zphoteff_file',
    'zphoterr_asym',
    'generate_zphot_from_calc',
    'generate_zphot_quantiles_gaussian',
    'validate_genzphot_fudgemap',
]
