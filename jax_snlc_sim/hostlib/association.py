"""Assemble host resources and associate galaxies with simulated events.

The batch association follows SNANA's GEN_SNHOST_DRIVER.

Everything between "a SN has a placement" and "a SN has a host": the
weighted HOSTLIB draw (with GROUPID filtering and the wrong-host model), the
SN placed on its galaxy, the DDLR sort that decides the reported host and its
neighbour, host-property smearing, the GENZPHOT photo-z and its quantiles,
GALID_UNIQUE, then the host's peculiar velocity, lensing and redshift passed
to the SN. The USESNPAR overrides run once the population is drawn
(:func:`apply_host_snpar_overrides`).

The catalogue, the weight map, the Sersic sampling and the photo-z formulae
live in :mod:`jax_snlc_sim.hostlib`; the SN's own draws live in
:mod:`jax_snlc_sim.sampler`, whose distance and MWEBV helpers this module
shares so a recomputed value keys off the same random streams."""


import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core.cosmology import zcmb_to_zhelio, zhelio_to_zcmb
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.random import batch_keys, event_keys
from jax_snlc_core.snana_files import (
    HOSTGAL_COORD_NULL, HOSTGAL_PREFIXES, HOSTGAL_PROPERTY_BASENAMES,
    SNANA_NULLFLOAT, is_ignorefile, read_wgtmap_varnames, resolve_fits_path)
from jax_snlc_core.utils import LIGHT_KM
from jax_snlc_sim.hostlib.catalogue import (
    HostCatalogue, discover_output_only_columns, essential_columns,
    hostlib_declared, read_hostlib)
from jax_snlc_sim.hostlib.photoz import (
    DECADE_QUANTILES, NRAND_ZPHOT, GenZphot, ZphotEfficiency, build_genzphot,
    catalogue_zphot_shift, generate_zphot_from_calc, generate_zphot_quantiles_gaussian,
    read_zphoteff_file, shifted_catalogue_zphot, zphot_columns, zphot_quantile_columns)
from jax_snlc_sim.hostlib.selection import (
    HostSelector, prepare_host_selection, sn_sky_position)
from jax_snlc_sim.hostlib.spectra import (
    HostSpectra, hostlib_coefficients, hostlib_column_names, load_host_spec_table)
from jax_snlc_sim.hostlib.wronghost import WrongHostTable, gen_zhost, read_wronghost_file
from jax_snlc_sim.lensing import lensing_offset
from jax_snlc_sim.population import HOST_DUST_PARAMETERS, model_spellings
from jax_snlc_sim.sampler import (
    PREFILTER_REASON_CODES, coord_shift_from_polar, distance_mag,
    distance_modulus_at, draw_mwebv)
from jax_snlc_sim.simulation import HOSTLIB_MSKOPT_ZPHOT_QGAUSS
from jax_snlc_sim.zvariation import REDSHIFT


logger = logging.getLogger(__name__)


DZ_SAFE_HOSTLIB = 0.002


def hostlib_snpar_columns(spellings, columns):
    """HOSTLIB columns USESNPAR may override (sntools_host.c:671-686): those
    of ``columns`` that spell one of the model's population parameters,
    whatever their case, and the host dust columns."""
    return tuple(column for column in columns
                 if column.lower() in spellings) + tuple(
        p.hostlib_column for p in HOST_DUST_PARAMETERS)


@dataclass(frozen=True, eq=False, kw_only=True)
class Hostlib:
    """A loaded HOSTLIB, everything a run reads of it: the draw's settings
    and the catalogue it draws from, the GALIDs that label its hosts, the
    photo-z model and quantile columns, and the optional wrong-host, photo-z
    efficiency and host-spectrum tables. None of it changes during a run."""
    selector: HostSelector
    catalogue: HostCatalogue
    galids: np.ndarray = None # int64; never in compiled code
    output_only_columns: List[str]         # the columns SEARCHEFF may name
    zphot_quantiles: List[Tuple[str, float]]   # (column, percentile)
    genzphot: Optional[GenZphot]           # None: the catalogue photo-z
    z_max: float                           # the file's own ZTRUE range top
    vpec_max: Optional[float]              # the largest |VPEC|
    vpec_err: Optional[float]              # the header's VPEC_ERR
    wronghost: Optional[WrongHostTable]
    zphoteff: Optional[ZphotEfficiency]
    spectra: Optional[HostSpectra] = None

    def galaxy_ids(self, rows):
        """The catalogue GALID of each row a batch recorded, as int64; an
        unmatched neighbour slot (row -1) keeps the event's initial OBJID of
        0, as SNANA copies the sorted list only up to NMATCH2
        (snlc_sim.c:26066, sntools.c:9163)."""
        rows = np.asarray(rows)
        return np.where(rows >= 0, self.galids[rows], np.int64(0))


jax.tree_util.register_dataclass(
    Hostlib,
    data_fields=[
        'selector',
        'catalogue',
        'genzphot',
        'wronghost',
        'zphoteff',
    ],
    meta_fields=[
        'output_only_columns',
        'zphot_quantiles',
        'z_max',
        'vpec_max',
        'vpec_err',
    ],
    drop_fields=['galids', 'spectra'],
)


def _check_hostlib_redshift_coverage(table, redshift_range) -> None:
    """Abort unless the HOSTLIB spans GENRANGE_REDSHIFT (check_redshift_HOSTLIB,
    sntools_host.c:3322). dz_safe converts the zCMB range to heliocentric."""
    ztrue = table.columns['ZTRUE']
    zmin_lib, zmax_lib = float(np.min(ztrue)), float(np.max(ztrue))
    zmin_gen = redshift_range[0] - DZ_SAFE_HOSTLIB
    zmax_gen = redshift_range[1] + DZ_SAFE_HOSTLIB
    if zmin_lib > zmin_gen or zmax_lib < zmax_gen:
        raise SNANAError(
            f"HOSTLIB z-HEL range does not contain user GENRANGE_REDSHIFT.\n"
            f"  HOSTLIB z-HEL range: {zmin_lib:.5f} - {zmax_lib:.5f}\n"
            f"  Gen     z-HEL range: {zmin_gen:.5f} - {zmax_gen:.5f} "
            f"(GENRANGE_REDSHIFT +- {DZ_SAFE_HOSTLIB})\n"
            f"  Gen     z-CMB range: {redshift_range[0]:.5f} - "
            f"{redshift_range[1]:.5f} (GENRANGE_REDSHIFT)",
            function_name="_check_hostlib_redshift_coverage")


def _check_snpar_conflict(config, sed_model, hostlib_varnames) -> None:
    """Refuse a population statement the HOSTLIB would then overwrite.

    Under USESNPAR the host's column replaces the drawn value
    (apply_host_snpar_overrides), so a class or a shift on that parameter
    would be drawn and discarded.
    """
    spellings = model_spellings(sed_model)
    columns = [(spellings[column.lower()], column)
               for column in hostlib_varnames if column.lower() in spellings]
    columns += [(p.key, p.hostlib_column) for p in HOST_DUST_PARAMETERS
                if p.hostlib_column in hostlib_varnames]
    clashing = sorted(
        f"{name} (HOSTLIB column {column})"
        for name, column in columns
        if name in config.model.populations
        and config.model.populations[name].is_shaped())
    if clashing:
        raise SNANAError(
            f"This run's population shapes {', '.join(clashing)} through a "
            f"SUBPOP class or a VARIATION, but HOSTLIB_MSKOPT bit 64 "
            f"(USESNPAR) makes the HOSTLIB column replace the drawn value, so "
            f"it would have no effect. Drop it or the USESNPAR bit.",
            function_name="_check_snpar_conflict")


def load_hostlib(config, simlib, sed_model, spec_zhost,
                 hz_map, frac_pec1a) -> Optional[Hostlib]:
    """The HOSTLIB the run draws its hosts from, built from the input's host
    keys, or None when HOSTLIB_FILE names none. The file is read once, into
    a table the build consumes and drops."""
    host = config.host
    if host.hostlib_file is None:
        logger.info("No HOSTLIB_FILE configured - skipping host galaxy selection")
        return None
    logger.info("Loading HOSTLIB for host galaxy selection...")
    hostlib_file = resolve_fits_path(os.path.expandvars(host.hostlib_file))
    wgtmap_file = (resolve_fits_path(os.path.expandvars(host.wgtmap_file))
                   if host.wgtmap_file else None)
    declared = hostlib_declared(hostlib_file)
    spec_table = load_host_spec_table(config)
    specid_maps, _, zhost_maps, _ = spec_zhost
    stacked, extra = _hostlib_columns(config, simlib, sed_model, declared,
                                      wgtmap_file, specid_maps + zhost_maps)
    build_only = ([] if spec_table is None
                  else hostlib_column_names(spec_table))
    table = read_hostlib(
        hostlib_file, list(dict.fromkeys(stacked + extra + build_only)),
        genrange_ra=host.hostlib_genrange_ra,
        genrange_dec=host.hostlib_genrange_dec,
        genrange_redshift=config.rates.redshift_range,
        maxread=host.hostlib_maxread,
        snr_store=(host.snr_store, config.survey.genfilters),
        varname_zphot=host.varname_zphot,
        swap_zphot=bool(host.mskopt & 128),
        fixsersic=host.fixsersic,
        scale_sersic_size=host.scale_sersic_size,
    )
    _check_hostlib(config, sed_model, table, declared)
    selector, catalogue = prepare_host_selection(
        config, simlib, sed_model, table, stacked, extra, wgtmap_file,
        hz_map, frac_pec1a)
    quantiles = zphot_quantile_columns(declared)
    if host.mskopt & HOSTLIB_MSKOPT_ZPHOT_QGAUSS and not quantiles:
        # the forced 11-decade set (sntools_host.c:4583-4585)
        quantiles = DECADE_QUANTILES
    return Hostlib(
        selector=selector,
        catalogue=catalogue,
        galids=table.columns['GALID'],
        output_only_columns=discover_output_only_columns(declared),
        zphot_quantiles=quantiles,
        genzphot=build_genzphot(host, config.rates.redshift_range,
                                (table.z_min, table.z_max)),
        z_max=table.z_max,
        vpec_max=table.vpec_max,
        vpec_err=table.vpec_err,
        wronghost=(None if is_ignorefile(host.wronghost_file)
                   else read_wronghost_file(
                       os.path.expandvars(host.wronghost_file),
                       *config.rates.redshift_range)),
        zphoteff=(None if is_ignorefile(host.zphoteff_file)
                  else read_zphoteff_file(
                      os.path.expandvars(host.zphoteff_file),
                      table.columns['ZTRUE'][[0, -1]])),
        spectra=(None if spec_table is None
                 else HostSpectra(spec_table,
                                  hostlib_coefficients(spec_table,
                                                       table.columns))),
    )


def _hostlib_columns(config, simlib, sed_model, declared, wgtmap_file,
                     searcheff_maps):
    """The HOSTLIB columns the run reads, as ``(stacked, extra)``: those
    the draw gathers into each host's properties, and those read by name
    (the photo-z, its quantiles, STOREPAR and the output-only columns a
    SEARCHEFF map may name). The variables of ``searcheff_maps``, the
    SEARCHEFF SPEC and zHOST maps, are among those gathered. A column the
    file lacks is simply not read."""
    host = config.host
    mskopt = host.mskopt
    stacked = essential_columns()
    if mskopt & 128:
        stacked.extend(['ZPHOT', 'ZPHOT_ERR'])
    if mskopt & 512:
        stacked.extend(['VPEC', 'VPEC_ERR'])
    if mskopt & 2:
        # GALMAG: host photometry for whatever [band]_obs columns exist
        # (sntools_host.c:2757-2766)
        stacked.extend(v for v in declared if v.endswith('_obs'))
    if (config.survey.use_simlib_groupid
            and any(f.groupid_hostlib for f in simlib.fields)):
        stacked.append('GROUPID')
    detect = (host.snr_detect is not None or host.mag_detect is not None
              or host.snr_store is not None)
    if detect:
        stacked.extend(v for v in declared
                       if v.endswith('_obs') and not v.endswith('_obs_err'))
        if host.snr_detect is not None or host.snr_store is not None:
            stacked.extend(v for v in declared if v.endswith('_obs_err'))
    stacked.extend(['WEAKLENS_DMU', 'TRUE'])
    # the variables a SEARCHEFF map or the WGTMAP reads
    stacked.extend(v for m in searcheff_maps for v in m.varnames)
    if wgtmap_file:
        stacked.extend(read_wgtmap_varnames(Path(wgtmap_file)))
    stacked.extend(host.storepar)
    # host properties a population depends on (sntools_host.c:951-969)
    generated = {name for name, _ in sed_model.configured_population} | {REDSHIFT}
    stacked.extend(name for name in config.model.quantities_depended_on()
                   if name not in generated)
    if mskopt & 64:
        # USESNPAR: the model's SN-parameter columns and SNMAGSHIFT
        # (sntools_host.c:671-686)
        stacked.extend([*hostlib_snpar_columns(model_spellings(sed_model),
                                               declared), 'SNMAGSHIFT'])
    extra = (['ZPHOT', 'ZPHOT_ERR']
             + [name for name, _ in zphot_quantile_columns(declared)]
             + host.storepar + discover_output_only_columns(declared))
    return list(dict.fromkeys(stacked)), list(dict.fromkeys(extra))


def _check_hostlib(config, sed_model, table, declared) -> None:
    """The HOSTLIB checks SNANA aborts on once the catalogue is read."""
    host = config.host
    mskopt = host.mskopt
    # a redshift taken from the SIMLIB is not drawn from GENRANGE_REDSHIFT,
    # so the catalogue need not span it (check_redshift_HOSTLIB, :3320)
    if not config.survey.use_simlib_redshift:
        _check_hostlib_redshift_coverage(table, config.rates.redshift_range)
    if mskopt & 64:
        _check_snpar_conflict(config, sed_model, declared)
        # USESNPAR with no SN-param column is fatal (sntools_host.c:2616-2624)
        snpar = (*hostlib_snpar_columns(model_spellings(sed_model),
                                        table.columns), 'SNMAGSHIFT')
        if not any(name in table.columns for name in snpar):
            names = [name for name, _ in sed_model.configured_population]
            raise SNANAError(
                f"Found zero SN params for HOSTLIB_MSKOPT ({mskopt}) & 64 "
                f"option. Expected a column for 1 or more of "
                f"'{' '.join(names + list(snpar))}'.",
                function_name="_check_hostlib")
    # USEVPEC needs the VPEC column and an error, as a column or the
    # header's VPEC_ERR (sntools_host.c:2797-2812)
    if mskopt & 512:
        if 'VPEC' not in table.columns:
            raise SNANAError(
                "Could not find required VPEC column. Add VPEC column, or "
                "remove 512-bit of HOSTLIB_MSKOPT", function_name="_check_hostlib")
        if 'VPEC_ERR' not in table.columns and table.vpec_err is None:
            raise SNANAError(
                "Found VPEC column, but VPEC_ERR not defined. Add VPEC_ERR "
                "column, or define VPEC_ERR in header",
                function_name="_check_hostlib")
    # Morphology-dependent MSKOPT bits are fatal without a Sersic profile
    if table.sersic_profiles is None:
        if mskopt & 2:
            raise SNANAError(
                "HOSTLIB_MSKOPT bit 2 (GALMAG): Galaxy (Sersic) profile NOT defined "
                "==> cannot compute galaxy mag at SN position.",
                function_name="_check_hostlib")
        if mskopt & 8:
            raise SNANAError(
                "HOSTLIB_MSKOPT bit 8 (SN2GAL_RADEC): must define host Sersic profile "
                "to move SN near host.",
                function_name="_check_hostlib")
    zphot_col, zphot_err_col = zphot_columns(host.varname_zphot, declared)
    # The forced Gaussian quantiles need catalogue ZPHOT/ZPHOT_ERR
    # ("Cannot compute forced Gaussian quantiles", sntools_host.c:4587-4607)
    if mskopt & HOSTLIB_MSKOPT_ZPHOT_QGAUSS and (
            zphot_col not in declared or zphot_err_col is None):
        raise SNANAError(
            f"HOSTLIB_MSKOPT bit {HOSTLIB_MSKOPT_ZPHOT_QGAUSS} (ZPHOT_QGAUSS) "
            "needs ZPHOT and ZPHOT_ERR columns in the HOSTLIB.",
            function_name="_check_hostlib")
    # init_HOSTLIB_ZPHOTEFF (sntools_host.c:4687-4692)
    if not is_ignorefile(host.zphoteff_file) and zphot_col not in declared:
        raise SNANAError(
            "Cannot implement HOSTLIB_ZPHOTEFF_FILE because ZPHOT is not "
            "defined in the HOSTLIB.", function_name="_check_hostlib")
    absent = [name for name in host.storepar if name not in table.columns]
    if absent:
        raise SNANAError(
            f"HOSTLIB_STOREPAR names {', '.join(absent)}, which the "
            f"HOSTLIB does not provide.", function_name="_check_hostlib")


def _repeats(simulator):
    """Whether HOSTLIB_NREPEAT_GALID_SNPOS blocks events on one host."""
    return simulator.host_settings.nrepeat_galid_snpos > 0


def _host_key_indices(simulator, params, n_sne: int, global_start_idx: int):
    """Indices to key host RNG streams from.

    Under HOSTLIB_NREPEAT_GALID_SNPOS every SN in a block keys off its block
    leader, so the galaxy, its Sersic offset and its smeared properties all
    repeat (sntools_host.c:5990). Without the key this is the SN's own index,
    matching a per-event stream key.
    """
    if _repeats(simulator):
        return params['host_leader_global_idx']
    return jnp.arange(n_sne) + global_start_idx


def wgtmap_sn_variables(simulator, params):
    """The WGTMAP's SN variables, which the reader names as the simulation
    does: a population parameter by name, or the host dust the SN draws
    (checkSNvar_HOSTLIB_WGTMAP, sntools_host.c:2868)."""
    if simulator.hostlib is None or not simulator.hostlib.selector.has_snvars:
        return {}
    selector = simulator.hostlib.selector
    named = {**simulator.model.sn_variables(params),
             # dust is drawn before the host only when the map reads it
             **{p.key: params[p.param] for p in HOST_DUST_PARAMETERS
                if p.param and p.key in selector.snvar_names}}
    return {name: named[name] for name in selector.snvar_names}


def _host_leader_value(simulator, params, name: str):
    """A host-selection input as the host is drawn: the block leader's under
    the repeat, the SN's own otherwise."""
    if _repeats(simulator):
        return params[f'host_leader_{name}']
    return params[name]


def _apply_nrepeat_raw_photoz(simulator, catalogue, host_properties, params,
                              galaxy_indices, n_sne: int,
                              global_start_idx: int):
    """Under the repeat, non-leader events report the raw catalogue ZPHOT.

    SORT_SNHOST_byDDLR reloads ZPHOT from the catalogue unconditionally
    (sntools_host.c:8892), then the gated GEN_SNHOST_ZPHOT applies the
    ZGEN-ZTRUE shift, the bias and any GENZPHOT formula, but only for block
    leaders. So followers keep the catalogue value whether or not
    HOSTLIB_GENZPHOT is in use.
    """
    resident = catalogue.extra_columns
    if not _repeats(simulator) or 'ZPHOT' not in resident:
        return
    is_leader = (params['host_leader_global_idx']
                 == jnp.arange(n_sne) + global_start_idx)
    for column in ('ZPHOT', 'ZPHOT_ERR'):
        host_properties[column] = jnp.where(
            is_leader, host_properties[column],
            jnp.take(resident[column], galaxy_indices))


def _parse_property_err_scales(host) -> Dict[str, float]:
    """Parse HOSTLIB_SCALE_PROPERTY_ERR / legacy HOSTLIB_SCALE_LOGMASS_ERR.

    SNANA uses VALUE(NAME) format: e.g., ``0.5(LOGMASS),2.0(LOGSFR)``.
    Specifying both keys is a fatal error.
    """
    legacy = host.scale_logmass_err
    scale_str = host.scale_property_err
    if legacy is not None and scale_str:
        raise SNANAError(
            "Cannot specify both HOSTLIB_SCALE_LOGMASS_ERR and "
            "HOSTLIB_SCALE_PROPERTY_ERR. Use HOSTLIB_SCALE_PROPERTY_ERR only.",
            function_name="_parse_property_err_scales",
        )
    scale_map: Dict[str, float] = {}
    if legacy is not None:
        scale_map['LOGMASS'] = float(legacy)
    if scale_str:
        for pair in scale_str.split(','):
            pair = pair.strip()
            m = re.match(r'([0-9.eE+-]+)\((\w+)\)', pair)
            if m is None:
                raise SNANAError(
                    f"HOSTLIB_SCALE_PROPERTY_ERR term '{pair}' is not "
                    f"VALUE(NAME)", function_name="_parse_property_err_scales")
            scale_map[m.group(2)] = float(m.group(1))
    return scale_map


def _smear_host_properties(host, host_properties: dict,
                           key_indices, prefix: str = '',
                           stream: str = 'HOSTPROP_SMEAR', *, key) -> None:
    """Apply measured-vs-true smearing to host galaxy properties, in place.

    Algorithm (GEN_SNHOST_PROPERTY, sntools_host.c:7495-7542): for each of
    the SNANA property basenames LOGMASS / LOGSFR / LOGsSFR / COLOR
    (sndata.h:85-88):
        1. If _OBS column is already populated, use as-is (still scale ERR)
        2. Else if bare/TRUE + _ERR exist → OBS = TRUE + G * ERR * SCALE,
           with G a truncated normal on [-3, 3]: SNANA's getRan_GaussClip
           redraws until the Gaussian lies inside the range
           (sntools.c:7870-7878), so there are no atoms at the bounds
        3. Else if only bare/TRUE → OBS = TRUE

    Each property comes from its own HOSTLIB columns (bare name or _TRUE
    suffix, check both); SNANA derives none of them from the others.
    """
    scale_map = _parse_property_err_scales(host)

    # getRan_GaussClip(1, -3, 3) redraws, so a truncated normal (sntools.c:7870-7878)
    all_gauss = jax.vmap(
        lambda k: jax.random.truncated_normal(
            k, -3.0, 3.0, shape=(len(HOSTGAL_PROPERTY_BASENAMES),)))(
        event_keys(key, stream, key_indices))

    for i, name in enumerate(HOSTGAL_PROPERTY_BASENAMES):
        column = f'{prefix}{name}'
        true_val = host_properties.get(column)
        if true_val is None:
            true_val = host_properties.get(f'{column}_TRUE')
        if true_val is None:
            continue
        obs_val = host_properties.get(f'{column}_OBS')
        err_val = host_properties.get(f'{column}_ERR')
        scale = scale_map.get(name, 1.0)
        if obs_val is not None:
            if err_val is not None:
                host_properties[f'{column}_ERR'] = err_val * scale
        elif err_val is not None:
            scaled_err = err_val * scale
            host_properties[f'{column}_OBS'] = (
                true_val + all_gauss[:, i] * scaled_err)
            host_properties[f'{column}_ERR'] = scaled_err
        else:
            host_properties[f'{column}_OBS'] = true_val


def _generate_calc_photoz(key, model, zgen, ztrue_host, n_sne: int,
                          global_start_idx: int, name: str):
    """Generate photo-z from the GENZPHOT ``model`` (a GenZphot), at the
    lookup redshift ``zgen``, with the host's ``ztrue_host`` as the
    polynomial-bias axis.

    ``name`` keys the per-SN random stream ('ZPHOT' primary host,
    'ZPHOT2' HOSTGAL2).
    """
    uniforms = jax.vmap(
        lambda k: jax.random.uniform(k, shape=(NRAND_ZPHOT,)))(
        batch_keys(key, name, n_sne, global_start_idx))
    return generate_zphot_from_calc(zgen, model, uniforms, ztrue_host)


def _generate_zphot_quantiles_gaussian(quantiles, catalogue,
                                       host_properties: dict,
                                       galaxy_indices, zratio) -> None:
    """Generate ZPHOT quantiles from Gaussian (HOSTLIB_MSKOPT ZPHOT_QGAUSS).

    SNANA computes zq = ZPHOT + ZPHOT_ERR*InverseNormal from the HOSTLIB
    catalogue values and rescales every quantile by ZRATIO = zSN/zGAL
    (GEN_SNHOST_ZPHOT_QUANTILE, sntools_host.c:9030-9052); the required
    columns are validated at build. ``quantiles`` are the (column,
    percentile) pairs, the HOSTLIB's declared set or the forced decades.
    Adds them to ``host_properties``."""
    resident = catalogue.extra_columns
    values = generate_zphot_quantiles_gaussian(
        jnp.take(resident['ZPHOT'], galaxy_indices),
        jnp.take(resident['ZPHOT_ERR'], galaxy_indices),
        quantiles)
    values = {name: zq * zratio for name, zq in values.items()}
    host_properties.update(values)
    at = {p: values[name] for name, p in quantiles}
    if 50 in at:
        host_properties['P50'] = at[50]
    if 20 in at and 80 in at:
        host_properties['IQR'] = at[80] - at[20]


def compute_host_dmucor(simulator, params, host_properties):
    """DMUCOR = mu(zSN) - mu(z_host), added to the reported host mags to
    account for the SN-vs-host redshift offset. SORT_SNHOST_byDDLR takes the
    SN's distance before the host moves its redshift, and the host's at
    vpec=0 and the SN's position (sntools_host.c:8795-8805). Returns 0 where
    there is no host."""
    host_zhel = host_properties['ZTRUE']
    valid = host_zhel > 1e-8
    host_zhel = jnp.where(valid, host_zhel, 0.1)

    if simulator.cosmology.vel_cmbapex != 0.0:
        host_zcmb = zhelio_to_zcmb(host_zhel, params['ra'], params['dec'])
    else:
        host_zcmb = host_zhel

    host_dlmu = distance_modulus_at(
        simulator, host_zcmb, host_zhel,
        jnp.zeros_like(host_zhel), params['ra'], params['dec'])
    return jnp.where(valid, params['mu'] - host_dlmu, 0.0)


def load_hostlib_zphot_quantiles(quantiles, catalogue, host_properties, zgen,
                                 galaxy_indices, hostlib_mskopt):
    """Catalogue photo-z quantiles (``quantiles``, (column, percentile)
    pairs) of the reported host and of each stored neighbour, every one
    rescaled by the lookup redshift ``zgen`` over the drawn host's ZTRUE, or
    by one under SN2GAL_Z (SNHOSTGAL.ZRATIO, sntools_host.c:6486;
    GEN_SNHOST_QUANTILE_ZPHOT, :9050)."""
    present = [name for name, _ in quantiles
               if name in catalogue.extra_columns]
    if not present:
        return
    if hostlib_mskopt & 16:
        zratio = 1.0
    else:
        zratio = zgen / jnp.take(catalogue.redshifts, galaxy_indices)
    for slot in _slots(host_properties):
        indices = host_properties[f'{slot}reported_galaxy_index']
        stored = indices >= 0
        for name in present:
            zq = jnp.where(stored,
                           jnp.take(catalogue.extra_columns[name],
                                    jnp.where(stored, indices, 0)),
                           SNANA_NULLFLOAT)
            host_properties[f'{slot}{name}'] = jnp.where(zq > 0.0,
                                                         zq * zratio, zq)


def _slots(host_properties):
    """The key prefix of every reported host slot the properties carry: ''
    for the host, then 'HOSTGAL2_' and on for its neighbours."""
    return [''] + [f'{prefix}_' for prefix in HOSTGAL_PREFIXES[1:]
                   if f'{prefix}_reported_galaxy_index' in host_properties]


def _apply_hostlib_storepar(storepar, catalogue, params, galaxy_indices):
    """HOSTLIB_STOREPAR columns, written as SIM_HOSTLIB(<name>)."""
    for name in storepar:
        out = (f'{name}_TRUE' if name in HOSTGAL_PROPERTY_BASENAMES
               else name)
        params[f'sim_hostlib_{out}'] = jnp.take(
            catalogue.extra_columns[name], galaxy_indices)


def _apply_wronghost(simulator, params, batch_n_sne,
                     global_start_idx, host_zhelio, host_idx, *, key):
    """WRONGHOST_FILE contamination. Returns the redshift the HOSTLIB lookup
    uses and, per event, whether a wrong host found no table row nearby,
    which SNANA aborts on (sntools_wronghost.c:324-329).

    gen_zHOST runs before GEN_SNHOST_DRIVER (snlc_sim.c:12885-12899), so the
    wrong-host draw stays per event under HOSTLIB_NREPEAT and only the galaxy
    lookup follows the block leader. SIM_REDSHIFT_HOST starts as that
    per-event value (GENLC.REDSHIFT_HOST, snlc_sim.c:12886): zSN for a
    correct match, shifted otherwise.
    """
    def _zhost(zhelio, indices):
        r = jax.vmap(lambda k: jax.random.uniform(k, shape=(2,)))(
            event_keys(key, 'WRONGHOST', indices))
        return gen_zhost(zhelio, r[:, 0], r[:, 1], simulator.hostlib.wronghost)

    own_idx = jnp.arange(batch_n_sne) + global_start_idx
    z_host, host_match, uncovered = _zhost(params['sim_redshift_helio'],
                                           own_idx)
    params['CORRECT_HOSTMATCH'] = host_match.astype(jnp.int32)
    params['sim_redshift_host'] = z_host
    if _repeats(simulator):
        z_host, _, _ = _zhost(host_zhelio, host_idx)
    return z_host, uncovered


def _apply_neighbour_hosts(simulator, catalogue, params,
                           host_properties, galaxy_indices, zgen,
                           hostlib_mskopt):
    """DDLR-sorted host and its neighbours (NBR_LIST), or the true-host fallback.

    Rewrites the primary host entries with the sorted values, so everything
    downstream sees the sorted host, and adds the HOSTGAL2_, HOSTGAL3_ copies
    of the next-ranked neighbours. Without a
    NBR_LIST the only candidate is the true host, which still has to pass the
    DDLR cuts: a SN too far from its galaxy is reported unmatched even though
    the host still drives generation (SORT_SNHOSTGAL_byDDLR,
    sntools_host.c:8830).
    """
    if simulator.hostlib.selector.has_neighbours:
        if hostlib_mskopt & 8:
            field_ra_arr = None
            field_dec_arr = None
        else:
            # hosts are reported at the SN's sky position (sntools_host.c:8882)
            field_ra_arr, field_dec_arr = params['ra'], params['dec']
        hostgal_sorted, neighbours, nmatch, nmatch2 =\
            simulator.hostlib.selector.compute_neighbor_info(
                catalogue, galaxy_indices, host_properties, field_ra_arr,
                field_dec_arr, params['peakmjd'], zgen)
        for prefix, props in zip(HOSTGAL_PREFIXES[1:], neighbours):
            for key, value in props.items():
                host_properties[f'{prefix}_{key}'] = value
        host_properties['NMATCH'] = nmatch
        host_properties['NMATCH2'] = nmatch2
        host_properties.update(hostgal_sorted)
        return

    hostlib_cfg = simulator.host_settings
    host_properties['reported_galaxy_index'] = galaxy_indices
    ddlr = host_properties['DDLR']
    host_properties['NMATCH'] = (ddlr < hostlib_cfg.maxddlr).astype(jnp.int32)
    host_properties['NMATCH2'] = (ddlr < hostlib_cfg.maxddlr2).astype(jnp.int32)
    host_properties['DDLR_TRUE'] = host_properties['DDLR']
    host_properties['SNSEP_TRUE'] = host_properties['SNSEP']
    # Host at SN sky position (GEN_SNHOST_POS, sntools_host.c:7796).
    if not hostlib_mskopt & 8:
        field_ra, field_dec = params['ra'], params['dec']
        ra_gal, dec_gal = host_properties['RA_GAL'], host_properties['DEC_GAL']
        cos_dec_gal = jnp.cos(jnp.radians(dec_gal))
        ra_sn, dec_sn = sn_sky_position(host_properties)
        cos_dec_field = jnp.cos(jnp.radians(field_dec))
        host_properties['RA_GAL'] = field_ra + (ra_gal - ra_sn) * (cos_dec_gal / cos_dec_field)
        host_properties['DEC_GAL'] = field_dec + (dec_gal - dec_sn)


def _apply_host_photoz(simulator, catalogue, params,
                       host_properties, galaxy_indices, zgen, batch_n_sne,
                       global_start_idx, hostlib_mskopt, *, key):
    """Host photo-z, either synthesised by GENZPHOT or taken from the HOSTLIB,
    at the lookup redshift ``zgen`` (SNHOSTGAL.ZGEN).

    Runs after the DDLR sort as SNANA does, skipping when NNBR_DDLRCUT2 is
    zero, meaning no host within MAXDDLR2. The HOSTLIB path shifts the stored
    ZPHOT by (ZGEN - host ZTRUE) (GEN_SNHOST_ZPHOT_from_HOSTLIB,
    sntools_host.c:7179-7192) instead of synthesising one; with a neighbour
    list the DDLR sort has already done so for every slot.
    """
    hostlib = simulator.hostlib
    if hostlib.genzphot is not None:
        ztrue_host = host_properties['ZTRUE']
        zphot, zphot_err = _generate_calc_photoz(
            key, hostlib.genzphot, zgen, ztrue_host, batch_n_sne,
            global_start_idx, 'ZPHOT')
        maxddlr2 = simulator.host_settings.maxddlr2
        no_host = host_properties['DDLR'] >= maxddlr2
        host_properties['ZPHOT'] = jnp.where(no_host, SNANA_NULLFLOAT, zphot)
        host_properties['ZPHOT_ERR'] = jnp.where(no_host, SNANA_NULLFLOAT,
                                                 zphot_err)

        if hostlib.selector.has_neighbours:
            # every neighbour under MAXDDLR2 gets its own draw
            # (GEN_SNHOST_ZPHOT, sntools_host.c:6798)
            for k, prefix in enumerate(HOSTGAL_PREFIXES[1:], start=2):
                zphot_k, zphot_err_k = _generate_calc_photoz(
                    key, hostlib.genzphot, zgen, ztrue_host, batch_n_sne,
                    global_start_idx, f'ZPHOT{k}')
                ddlr_k = host_properties[f'{prefix}_DDLR']
                no_host_k = (ddlr_k < 0) | (ddlr_k >= maxddlr2)
                host_properties[f'{prefix}_ZPHOT'] = jnp.where(
                    no_host_k, SNANA_NULLFLOAT, zphot_k)
                host_properties[f'{prefix}_ZPHOT_ERR'] = jnp.where(
                    no_host_k, SNANA_NULLFLOAT, zphot_err_k)

        if hostlib_mskopt & HOSTLIB_MSKOPT_ZPHOT_QGAUSS:
            zratio = jnp.where(ztrue_host > 0, zgen / ztrue_host, 1.0)
            _generate_zphot_quantiles_gaussian(hostlib.zphot_quantiles,
                                               catalogue, host_properties,
                                               galaxy_indices, zratio)
    elif (not hostlib.selector.has_neighbours
          and 'ZPHOT' in catalogue.extra_columns):
        host_properties['ZPHOT'], host_properties['ZPHOT_ERR'] =\
            shifted_catalogue_zphot(
                jnp.take(catalogue.extra_columns['ZPHOT'], galaxy_indices),
                jnp.take(catalogue.extra_columns['ZPHOT_ERR'], galaxy_indices),
                catalogue_zphot_shift(zgen, host_properties['ZTRUE'],
                                      hostlib_mskopt))
        host_properties['ZSPEC_TRUEMATCH'] = jnp.ones(batch_n_sne,
                                                      dtype=jnp.float32)

    if not any(name in host_properties for name, _ in hostlib.zphot_quantiles):
        load_hostlib_zphot_quantiles(hostlib.zphot_quantiles, catalogue,
                                     host_properties, zgen, galaxy_indices,
                                     hostlib_mskopt)
    _apply_nrepeat_raw_photoz(simulator, catalogue, host_properties, params,
                              galaxy_indices, batch_n_sne, global_start_idx)
    _apply_zphoteff(hostlib.zphoteff, host_properties, batch_n_sne,
                    global_start_idx, key=key)


def _apply_zphoteff(efficiency, host_properties, n_sne: int,
                    global_start_idx: int, *, key) -> None:
    """The event's own host photo-z, which HOSTLIB_ZPHOTEFF (``efficiency``,
    a ZphotEfficiency, None without the file) can remove.

    SNANA carries two photo-z values and masks only one of them. ZPHOT_EVENT
    here is SNHOSTGAL.ZPHOT: the reported host's photo-z, set to -9 when the
    ZPHOTEFF draw fails, read at the host's own true redshift
    (GEN_SNHOST_ZPHOT, sntools_host.c:6825-6833). CUTWIN_HOST_ZPHOT
    (snlc_sim.c:24172), gen_zsmear's use-the-host-photo-z branch (:17374), the
    GENSIGMA_REDSHIFT < 0 redshift source (sntools_host.c:6837),
    setz_unconfirmed (snlc_sim.c:24922) and the SIMGEN_DUMP GALZPHOT column
    (:16162) all read that masked scalar. ZPHOT stays the DDLR-sorted host's
    own value, which the mask never touches and which the data file reports
    as HOSTGAL_PHOTOZ (:26068).
    """
    if 'ZPHOT' not in host_properties:
        return
    zphot = host_properties['ZPHOT']
    zphot_err = host_properties['ZPHOT_ERR']
    if efficiency is None:
        host_properties['ZPHOT_EVENT'] = zphot
        host_properties['ZPHOT_EVENT_ERR'] = zphot_err
        return
    uniforms = jax.vmap(jax.random.uniform)(
        batch_keys(key, 'ZPHOTEFF', n_sne, global_start_idx))
    eff = jnp.interp(host_properties['ZTRUE'], jnp.asarray(efficiency.ztrue),
                     jnp.asarray(efficiency.eff))
    no_photoz = uniforms > eff
    host_properties['ZPHOT_EVENT'] = jnp.where(no_photoz, SNANA_NULLFLOAT,
                                               zphot)
    host_properties['ZPHOT_EVENT_ERR'] = jnp.where(no_photoz, SNANA_NULLFLOAT,
                                                   zphot_err)


def _apply_galid_unique(host, key, host_properties, batch_n_sne,
                        global_start_idx):
    """HOSTLIB_GALID_UNIQUE (set_GALID_UNIQUE, sntools_host.c:9060): each
    reported host slot draws an offset in [0, 57), distinct from the earlier
    slots', stored as ``galid_unique_offset`` (-1 on an empty slot). The
    label, CID * 57 + offset, is formed from the written CID at output."""
    if not host.galid_unique:
        return
    # SNANA's redraw-until-distinct is a uniform draw over the free offsets
    taken = []
    for slot, prefix in zip(_slots(host_properties), HOSTGAL_PREFIXES):
        ran = jax.vmap(lambda k: jax.random.uniform(k))(
            batch_keys(key, f'{prefix}_GALID_UNIQUE', batch_n_sne,
                       global_start_idx))
        off = jnp.floor(ran * (57 - len(taken))).astype(jnp.int32)
        if taken:
            stack = jnp.sort(jnp.stack(taken, axis=1), axis=1)
            for j in range(stack.shape[1]):
                off = jnp.where(off >= stack[:, j], off + 1, off)
        taken.append(off)
        listed = host_properties[f'{slot}reported_galaxy_index'] >= 0
        host_properties[f'{slot}galid_unique_offset'] = jnp.where(listed, off, -1)


def host_pec1a_flags(simulator, n_sne: int, global_start_idx: int,
                     template_budget):
    """Per-event PEC1A flag choosing the SN2GAL_Z host CDF, or None without a DNDZ_PEC1A one."""
    if simulator.hostlib.selector.pec1a_row_shift is None:
        return None
    return simulator.model.pec1a_flags(n_sne, global_start_idx,
                                           template_budget)


def select_hosts(simulator, catalogue,
                 params: Dict[str, jnp.ndarray], global_start_idx: int, *,
                 key, reuse, n_events, template_budget):
    """GEN_SNHOST_GALID for the batch: each event's galaxy from
    ``catalogue`` (:class:`HostCatalogue`), drawn from the weight map at its
    heliocentric redshift, or the WRONGHOST_FILE redshift
    (gen_zHOST, snlc_sim.c:12884), within its SIMLIB entry's GROUPID set
    (the block leader's entry under HOSTLIB_NREPEAT) and under the reuse
    constraints. Of the lanes only the first ``n_events`` are events.

    Returns ``(host_properties, galaxy_indices, zgen, reuse, report)``:
    ``zgen`` is the lookup redshift (SNHOSTGAL.ZGEN, sntools_host.c:6016),
    ``reuse`` the run's reuse record after this batch and ``report`` what
    the draw reports beside the hosts (:meth:`HostSelector.select`), with
    ``wronghost_uncovered`` under WRONGHOST_FILE. An event MINDAYSEP leaves
    without a galaxy is rejected in ``params``, and every other event
    reports its galaxy's redshift as SIM_REDSHIFT_HOST (GENLC.REDSHIFT_HOST,
    :6481).
    """
    selector = simulator.hostlib.selector
    batch_n_sne = len(params['sim_redshift_cmb'])
    host_idx = _host_key_indices(simulator, params, batch_n_sne,
                                 global_start_idx)
    pec1a = host_pec1a_flags(simulator, batch_n_sne,
                             global_start_idx, template_budget)
    if pec1a is not None and _repeats(simulator):
        pec1a = params['host_leader_pec1a']
    random_uniforms = jax.vmap(lambda k: jax.random.uniform(k))(
        event_keys(key, 'HOST_SELECT', host_idx))
    position_uniforms = jax.vmap(
        lambda k: jax.random.uniform(k, shape=(4,)))(
        event_keys(key, 'HOST_POSITION', host_idx))

    sn_properties = None
    if selector.has_snvars:
        sn_properties = wgtmap_sn_variables(simulator, params)
        if _repeats(simulator):
            sn_properties = {name: params[f'host_leader_wgtvar_{name}']
                             for name in sn_properties}

    # HOSTLIB_FRAME_ZTRUE_HEL: the lookup redshift is heliocentric
    zgen = _host_leader_value(simulator, params, 'sim_redshift_helio')
    wronghost_report = {}
    if simulator.hostlib.wronghost is not None:
        zgen, wronghost_report['wronghost_uncovered'] = _apply_wronghost(
            simulator, params, batch_n_sne, global_start_idx,
            zgen, host_idx, key=key)

    # repeat followers and padding never register for reuse (sntools_host.c:6138)
    leads = host_idx == jnp.arange(batch_n_sne) + global_start_idx
    galaxy_indices, host_properties, reuse, report = selector.select(
        catalogue, zgen, random_uniforms, position_uniforms, sn_properties,
        params['peakmjd'], _host_leader_value(simulator, params, 'field_idx'),
        pec1a, reuse=reuse,
        reselect_keys=event_keys(key, 'HOST_RESELECT', host_idx),
        reuse_lanes=leads & (jnp.arange(batch_n_sne) < n_events))
    report.update(wronghost_report)
    if selector.uses_mindaysep:
        # SNHOSTGAL.PEAKMJD (sntools_host.c:6018)
        report['peak_day'] = selector.peak_day(params['peakmjd'])

    drawn = leads
    if 'rejected' in report:
        rejected = report['rejected']
        drawn = drawn & ~rejected
        params['prefilter_pass'] = params['prefilter_pass'] & ~rejected
        params['prefilter_reason_code'] = jnp.where(
            rejected, jnp.int16(PREFILTER_REASON_CODES['HOSTLIB_NOHOST']),
            params['prefilter_reason_code'])
    params['sim_redshift_host'] = jnp.where(
        drawn, host_properties['ZTRUE'], params['sim_redshift_host'])
    return host_properties, galaxy_indices, zgen, reuse, report


def add_host_associations(simulator, catalogue,
                          params: Dict[str, jnp.ndarray],
                          global_start_idx: int, *, key, reuse, n_events,
                          template_budget):
    """GEN_SNHOST_DRIVER for the batch (sntools_host.c:5962-6080) over
    ``catalogue`` (:class:`HostCatalogue`), in its order: the galaxy, the
    SN placed on it under SN2GAL_RADEC with MWEBV at the new position, the
    DDLR sort with the host-mag distance correction, the photo-z and the
    reported host's properties, then the host's VPEC, lensing offset and
    redshift passed to the SN.

    Returns ``(params, host_properties, reuse, report)``: the parameters
    with the reported host's ``host_`` columns, the drawn galaxy's own
    properties, which the population and :func:`apply_host_snpar_overrides`
    read (SNANA reads IGAL, not the DDLR-sorted host), the run's reuse record
    after this batch, and what the draw reports beside the hosts
    (:func:`select_hosts`).
    """
    host_properties, galaxy_indices, zgen, reuse, report = select_hosts(
        simulator, catalogue, params, global_start_idx, key=key,
        reuse=reuse, n_events=n_events, template_budget=template_budget)

    hostlib_mskopt = simulator.host_settings.mskopt
    if hostlib_mskopt & 8:
        # load_SNHOST_GENLC_COORDS (sntools_host.c:7867)
        params['ra'], params['dec'] = sn_sky_position(host_properties)
        params.update(draw_mwebv(simulator, params, key=key))
    if not hostlib_mskopt & 16:
        params['host_dmucor'] = compute_host_dmucor(
            simulator, params, host_properties)

    drawn_host = dict(host_properties)
    params = _finish_host_associations(
        simulator, catalogue, params, host_properties, galaxy_indices, zgen,
        global_start_idx, key=key)
    transfer_host_redshift(simulator, params, drawn_host, key=key)
    return params, drawn_host, reuse, report


def transfer_host_redshift(simulator, params, host_properties,
                           *, key) -> None:
    """The host's say over the SN's velocity, lensing and redshift:
    GEN_SNHOST_VPEC, GEN_SNHOST_WEAKLENS_DMU and TRANSFER_SNHOST_REDSHIFT
    (sntools_host.c:7464, :7450, :9113).

    USEVPEC takes the true VPEC from the HOSTLIB and smears the measured one
    with its error (gen_zsmear, snlc_sim.c:17441-17460). Under SN2GAL_Z the
    SN takes the host's heliocentric ZTRUE, which already holds the host's
    velocity, and zCMB follows with that velocity removed; under
    SN2GAL_RADEC alone zCMB stays and zHEL is recomputed at the new
    position. Either recomputes the distance (gen_distanceMag). A wrong host
    transfers nothing, and without either bit the redshifts stay as drawn
    even under USEVPEC, while the lensing offset becomes the host's
    WEAKLENS_DMU where no map or DSIGMADZ overrides it. Mutates ``params``
    in place.
    """
    hostlib_mskopt = simulator.host_settings.mskopt
    if hostlib_mskopt & 512:
        vpec = host_properties['VPEC']
        # the VPEC_ERR column, else the header's (sntools_host.c:7480-7483)
        vpec_err = (host_properties['VPEC_ERR']
                    if 'VPEC_ERR' in host_properties
                    else jnp.full_like(
                        vpec, simulator.hostlib.vpec_err))
        gauss = jax.vmap(jax.random.normal)(
            event_keys(key, 'HOSTVPEC_SMEAR', params['sn_global_idx']))
        params['vpec_true'] = vpec
        params['vpec_err'] = vpec_err
        params['vpec'] = vpec + vpec_err * gauss
    if not hostlib_mskopt & (8 | 16):
        # the host's lensing offset, where it is the one that counts
        params['lensdmu'] = lensing_offset(
            simulator, params['sim_redshift_cmb'],
            event_keys(key, 'LENSDMU', params['sn_global_idx']),
            host_properties)
        return

    ra, dec = params['ra'], params['dec']
    apex = simulator.cosmology.vel_cmbapex != 0.0
    zpec = params['vpec_true'] / LIGHT_KM
    if hostlib_mskopt & 16:
        zhel = host_properties['ZTRUE']
        zcmb = zhelio_to_zcmb(zhel, ra, dec) if apex else zhel
        zcmb = (1.0 + zcmb) / (1.0 + zpec) - 1.0
    else:
        zcmb = params['sim_redshift_cmb']
        zhel = zcmb_to_zhelio(zcmb, ra, dec) if apex else zcmb
        zhel = (1.0 + zhel) * (1.0 + zpec) - 1.0
    moved = {'sim_redshift_cmb': zcmb, 'sim_redshift_helio': zhel}
    moved.update(distance_mag(simulator, {**params, **moved},
                              host_properties, key=key))
    correct = params['CORRECT_HOSTMATCH'] == 1
    for name, value in moved.items():
        params[name] = jnp.where(correct, value, params[name])


def apply_host_snpar_overrides(simulator, params,
                               host_properties):
    """override_modelPar_from_SNHOST (snlc_sim.c:13034), once the population
    is drawn: SNMAGSHIFT under bit 4 or bit 64, and under USESNPAR the
    model's parameters, PEAKMJD and host dust from the drawn host's columns.
    Returns the updated parameters; without a host, ``params`` unchanged."""
    if host_properties is None:
        return params
    hostlib_mskopt = simulator.host_settings.mskopt
    if (hostlib_mskopt & (4 | 64)) and 'SNMAGSHIFT' in host_properties:
        params['snmagshift'] = host_properties['SNMAGSHIFT']
    if hostlib_mskopt & 64:
        spellings = model_spellings(simulator.model)
        host_values = {spellings[column.lower()]: value
                       for column, value in host_properties.items()
                       if column.lower() in spellings}
        params = simulator.model.apply_host_snpar_overrides(
            params, host_values)
        _apply_host_general_snpar(simulator, params,
                                  host_properties)
    return params


def _apply_host_general_snpar(simulator, params, host_properties):
    """PEAKMJD/RV/AV/EBV host columns override the drawn values under
    USESNPAR, for columns that are deliberately stored (modelPar_from_SNHOST
    looks parameters up among the stored variables; snlc_sim.c:13068-13082,
    sntools_host.c:9801). An EBV column sets AV = RV * EBV, per event and
    only where RV > 0.001. The SIM_AV/SIM_RV truth follows the final values,
    as GENLC does."""
    stored = {p.hostlib_column for p in HOST_DUST_PARAMETERS}
    stored.update(str(v).strip() for v in simulator.host_settings.storepar)

    if 'PEAKMJD' in stored and 'PEAKMJD' in host_properties:
        params['peakmjd'] = host_properties['PEAKMJD']
    if 'RV' in stored and 'RV' in host_properties:
        params['rv'] = host_properties['RV']
        params['sim_rv'] = params['rv']
    if 'AV' in stored and 'AV' in host_properties:
        params['av'] = host_properties['AV']
        params['sim_av'] = params['av']
    if 'EBV' in stored and 'EBV' in host_properties:
        params['av'] = jnp.where(params['rv'] > 0.001,
                                 params['rv'] * host_properties['EBV'],
                                 params['av'])
        params['sim_av'] = params['av']


def _finish_host_associations(simulator, catalogue, params,
                              host_properties, galaxy_indices, zgen,
                              global_start_idx: int, *, key
                              ) -> Dict[str, jnp.ndarray]:
    """What follows host selection, in SNANA's order: the DDLR sort that
    decides the reported host and its neighbour, then that host's smeared
    properties, photo-z, GALID_UNIQUE and STOREPAR columns, prefixed
    ``host_`` into a copy of ``params``. The sort rewrites
    ``host_properties`` in place."""
    batch_n_sne = len(params['sim_redshift_cmb'])
    host_idx = _host_key_indices(simulator, params, batch_n_sne,
                                 global_start_idx)
    hostlib_mskopt = simulator.host_settings.mskopt

    _apply_neighbour_hosts(simulator, catalogue, params,
                           host_properties, galaxy_indices, zgen,
                           hostlib_mskopt)

    # smearing follows the DDLR sort, over every slot (sntools_host.c:7495-7542)
    _smear_host_properties(simulator.host_settings, host_properties, host_idx, key=key)
    for k, prefix in enumerate(HOSTGAL_PREFIXES[1:], start=2):
        _smear_host_properties(
            simulator.host_settings, host_properties, host_idx,
            prefix=f'{prefix}_', stream=f'HOSTPROP_SMEAR{k}', key=key)

    _apply_host_photoz(
        simulator, catalogue, params, host_properties,
        galaxy_indices, zgen, batch_n_sne, global_start_idx, hostlib_mskopt,
        key=key)

    _apply_galid_unique(simulator.host_settings, key, host_properties, batch_n_sne,
                        global_start_idx)

    params_with_hosts = params.copy()
    for column, value in host_properties.items():
        params_with_hosts[f'host_{column}'] = value
    # the drawn host; host_reported_galaxy_index is the one the DDLR sort ranks first
    params_with_hosts['host_galaxy_index'] = galaxy_indices
    _apply_hostlib_storepar(simulator.host_settings.storepar, catalogue,
                            params_with_hosts, galaxy_indices)

    return params_with_hosts


def galaxy_labels(simulator, params):
    """Each reported host slot's int64 GALID, from its galaxy index, and the
    drawn host's as ``drawn_GALID``; formed outside compiled code, where an
    int64 would narrow to int32 and wrap large catalogue ids."""
    labels = {f'host_{slot}GALID': simulator.hostlib.galaxy_ids(
                  params[f'host_{slot}reported_galaxy_index'])
              for slot in _slots({key[len('host_'):]: value
                                  for key, value in params.items()
                                  if key.startswith('host_')})}
    labels['drawn_GALID'] = simulator.hostlib.galaxy_ids(
        params['host_galaxy_index'])
    return labels


def random_coord_shift(host, key, n_sne: int, global_start_idx: int, dec):
    """Per-SN MXRADIUS_RANDOM_SHIFT offset, or None when the key is unset.

    r is drawn as MXRADIUS*sqrt(u) so it is uniform in area
    (snlc_sim.c:13145-13153).
    """
    mxradius = host.mxradius_random_shift
    if mxradius <= 1.0e-12:
        return None
    keys = batch_keys(key, 'COORD_SHIFT', n_sne, global_start_idx)
    rans = jax.vmap(lambda k: jax.random.uniform(k, (2,)))(keys)
    r = mxradius * jnp.sqrt(rans[:, 0])
    phi = 2.0 * jnp.pi * rans[:, 1]
    return coord_shift_from_polar(r, phi, dec)


def apply_random_coord_shift(simulator, params, n_sne,
                             global_start_idx, *, key) -> None:
    """MXRADIUS_RANDOM_SHIFT (gen_random_coord_shift, snlc_sim.c:13113): one
    random offset moves the SN and every host candidate, scaled by cos(DEC)
    at the SN's position. It follows the host's say over the SN, and MWEBV
    and the redshifts keep the unshifted position, an approximation SNANA
    documents as valid for MXRADIUS << 1 deg. Mutates ``params`` in place.
    """
    dec = params['dec']
    shift = random_coord_shift(simulator.host_settings, key, n_sne, global_start_idx,
                               dec)
    if shift is None:
        return

    ra_lo, ra_hi = simulator.survey.genrange_ra

    def shifted_ra(ra):
        # apply_RA_convention (snlc_sim.c:13208)
        ra = ra + shift['ra']
        return jnp.where(ra > ra_hi, ra - 360.0,
                         jnp.where(ra < ra_lo, ra + 360.0, ra))

    params['ra'] = shifted_ra(params['ra'])
    params['dec'] = dec + shift['dec']
    for prefix in HOSTGAL_PREFIXES:
        column = 'host_' if prefix == 'HOSTGAL' else f'host_{prefix}_'
        if f'{column}RA_GAL' not in params:
            continue
        ra_gal = params[f'{column}RA_GAL']
        dec_gal = params[f'{column}DEC_GAL']
        # an unfilled neighbour slot keeps its sentinel
        listed = ra_gal > HOSTGAL_COORD_NULL
        params[f'{column}RA_GAL'] = jnp.where(listed, shifted_ra(ra_gal), ra_gal)
        params[f'{column}DEC_GAL'] = jnp.where(listed, dec_gal + shift['dec'],
                                               dec_gal)
    params['random_shift_RA'] = shift['ra']
    params['random_shift_DEC'] = shift['dec']
    params['random_shift_RADIUS'] = shift['radius']
    params['random_shift_PHI'] = shift['phi']


def _nrepeat_leader_source_arrays(simulator, params, start_index, template_budget):
    """Per-SN inputs to host selection that the repeat takes from the leader."""
    arrays = {'sim_redshift_helio': params['sim_redshift_helio'],
              'field_idx': params['field_idx']}
    for name, values in wgtmap_sn_variables(simulator, params).items():
        arrays[f'wgtvar_{name}'] = values
    pec1a = host_pec1a_flags(simulator, params['sim_redshift_helio'].shape[0],
                             start_index, template_budget)
    if pec1a is not None:
        arrays['pec1a'] = pec1a
    return arrays


def nrepeat_host_leaders(simulator, params, start_index, n_events,
                         carried, template_budget):
    """HOSTLIB_NREPEAT_GALID_SNPOS: each SN's block leader, whose host it
    reuses, and the leader's host-selection inputs.

    SNANA's flag is `(NCALL_GEN_SNHOST_DRIVER % NREPEAT) > 0`
    (sntools_host.c:5990), and the counter advances only for events that
    reach GEN_SNHOST_DRIVER: those passing the NEPOCH gate at
    snlc_sim.c:12864. So blocks are counted over `prefilter_pass`, not over
    event number; events past ``n_events`` count for nothing. A block may open
    in an earlier batch, so ``carried`` holds the host-call count, the
    in-flight block's leader and its values. Returns the new columns and
    the updated ``carried``.
    """
    nrepeat = simulator.host_settings.nrepeat_galid_snpos
    n = params['prefilter_pass'].shape[0]
    lane = jnp.arange(n)
    passing = params['prefilter_pass'] & (lane < n_events)
    count = jnp.cumsum(passing.astype(jnp.int32))
    # the leader's rank among this batch's passing events; negative when the
    # block opened in an earlier batch
    leader_rank = ((carried['ncall'] + count - 1) // nrepeat) * nrepeat \
        - carried['ncall']
    leader_lane = jnp.searchsorted(count, leader_rank + 1)
    leader = jnp.where(passing,
                       jnp.where(leader_rank < 0, carried['leader'],
                                 start_index + leader_lane),
                       start_index + lane).astype(jnp.int32)
    n_passing = count[-1]
    last = jnp.max(jnp.where(passing, lane, 0))
    local = leader - start_index

    columns = {'host_leader_global_idx': leader}
    values = {}
    for name, source in _nrepeat_leader_source_arrays(
            simulator, params, start_index, template_budget).items():
        gathered = jnp.where(local >= 0, source[jnp.maximum(local, 0)],
                             carried['values'][name])
        columns[f'host_leader_{name}'] = gathered
        values[name] = jnp.where(n_passing > 0, gathered[last],
                                 carried['values'][name])
    return columns, {
        'ncall': carried['ncall'] + n_passing,
        'leader': jnp.where(n_passing > 0, leader[last], carried['leader']),
        'values': values}
