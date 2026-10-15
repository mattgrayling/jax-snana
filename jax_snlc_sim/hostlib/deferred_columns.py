"""Deferred HOSTLIB columns: the output-only columns, held in the
HostCatalogue and gathered at each event's drawn host inside JIT.
"""

import jax.numpy as jnp

from jax_snlc_sim.simulation import HOSTLIB_MSKOPT_ZPHOT_QGAUSS


def gather_deferred_hostlib_columns(host, hostlib, catalogue, params):
    """JIT-safe gather of the output-only HOSTLIB columns from ``catalogue``
    (the HostCatalogue passed into compiled code), indexed by galaxy_idx.
    ``host`` is the input's host keys and ``hostlib`` the built HOSTLIB,
    None without one.

    Returns a new dict of additional per-SN arrays (`host_<col>`, gathered
    at the drawn host only; ZPHOT columns the simulation computes are
    skipped). Caller merges into the sn_vars / params dict it uses for
    SEARCHEFF.
    """
    if hostlib is None or 'host_galaxy_index' not in params:
        return {}
    columns = {name: catalogue.extra_columns[name]
               for name in hostlib.output_only_columns
               if name in catalogue.extra_columns}
    gal_idx = jnp.asarray(params['host_galaxy_index'], dtype=jnp.int32)
    calc_zphot_keys = set()
    if hostlib.genzphot is not None:
        calc_zphot_keys = {'ZPHOT', 'ZPHOT_ERR'}
        if host.mskopt & HOSTLIB_MSKOPT_ZPHOT_QGAUSS:
            calc_zphot_keys.update(name for name, _ in hostlib.zphot_quantiles)
            calc_zphot_keys.update(['P50', 'IQR'])

    gathered = {}
    for col_name, col_full in columns.items():
        if col_name in calc_zphot_keys:
            continue
        gathered[f'host_{col_name}'] = col_full[gal_idx]
    return gathered
