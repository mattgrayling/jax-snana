#!/usr/bin/env python3
"""
Replay SNANA SIMGEN_DUMP parameters through the JAX pipeline for debugging.

Usage:
    python replay_snana_dump.py --input INP_SIMGEN_SALT_HOSTLIB.input \
        --dump /path/to/MG_test_SNANA.DUMP --output MG_test_JAX_replay

What the SNANA run must have dumped for the replay to be exact:

- ``AV RV`` whenever the sim draws host dust per event (a P21 or GENPDF
  dust population with a HOSTLIB): without them the replay redraws the dust
  and is wrong by up to ~0.5 mag chromatically per SN.
- unique CIDs, i.e. ``CIDRAN_MIN``/``CIDRAN_MAX`` wide enough that no CID
  repeats: the dump reader keeps one row per CID (the highest CUTMASK), which
  is arbitrary when selection cuts are off and every row ties at 0.
- ``MU`` is optional; the HEAD's ``SIM_DLMU`` is used where it is absent.
"""

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict

import numpy as np
from astropy.io import fits

from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import SNANA_NULLFLOAT, load_snana_dump
from jax_snlc_sim import run as sim_run
from jax_snlc_sim.output import save_snana_outputs
from jax_snlc_sim.inputs import parse_snana_input_file
from jax_snlc_sim.hostlib.association import compute_host_dmucor, load_hostlib_zphot_quantiles
from jax_snlc_sim.output.dump import canonical_dump_columns
from jax_snlc_sim.sampler import PREFILTER_REASON_CODES, draw_grey_offset, field_coords

logger = logging.getLogger(__name__)


def _align_to_dump(values, head_snid, dump_cids, fill):
    """A HEAD column joined by SNID onto the dump's CID order, `fill` where
    absent. Always a join: the HEAD holds accepted events only, so equal
    lengths do not imply equal order."""
    vals = np.asarray(values, dtype=np.float32)
    snids = np.asarray(head_snid)
    if snids.dtype.kind not in 'iu':
        snids = np.char.strip(snids.astype(str)).astype(np.int64)
    cids = np.asarray(dump_cids).astype(np.int64)
    order = np.argsort(snids)
    pos = np.clip(np.searchsorted(snids, cids, sorter=order), 0, len(snids) - 1)
    idx = order[pos]
    return np.where(snids[idx] == cids, vals[idx],
                    np.float32(fill)).astype(np.float32)


def build_override_parameters(dump_columns: Dict[str, np.ndarray],
                              simulation, resources, head_columns: Dict[str, np.ndarray],
                              drop_node_smear: bool = False) -> Dict[str, np.ndarray]:
    """Map SNANA dump columns onto the simulator's parameter dictionary.

    drop_node_smear zeroes the per-node smear draws SNANA does not record,
    keeping the coherent term it does, for callers that measure the node
    scatter against a baseline carrying only the coherent part.
    """
    dump_columns = canonical_dump_columns(dump_columns, resources.model)
    required_shared = ['CID', 'GENZ', 'PEAKMJD', 'LIBID']
    missing_shared = [col for col in required_shared if col not in dump_columns]
    if missing_shared:
        raise SNANAError(f"Missing required columns in dump: {missing_shared}")

    libid_to_index = {field.libid: idx for idx, field in enumerate(resources.fields)}
    field_indices = []
    for libid in dump_columns['LIBID']:
        libid_int = int(libid)
        if libid_int not in libid_to_index:
            raise SNANAError(f"LIBID {libid_int} not found in SIMLIB file.")
        field_indices.append(libid_to_index[libid_int])

    # SNANA reports -9 for AV/RV when no dust was generated. The physics needs
    # 0 and 3.1; the reported columns keep the sentinel, so both are carried.
    av_column = dump_columns.get('AV')
    if av_column is None and simulation.host.dust_enabled:
        raise SNANAError(
            "the dump carries no AV, yet the sim generates host dust; add AV "
            "and RV to SIMGEN_DUMPALL in the SNANA input and re-run the dump.")
    if av_column is None:
        sim_av_values = np.full_like(
            dump_columns['CID'], SNANA_NULLFLOAT, dtype=np.float32)
        av_values = np.zeros_like(dump_columns['CID'], dtype=np.float32)
    else:
        sim_av_values = av_column.astype(np.float32)
        av_values = np.where(sim_av_values < -8.0, 0.0, sim_av_values)

    rv_column = dump_columns.get('RV')
    if rv_column is None:
        sim_rv_values = np.full_like(
            av_values, SNANA_NULLFLOAT, dtype=np.float32)
        rv_values = np.full_like(av_values, 3.1, dtype=np.float32)
    else:
        sim_rv_values = rv_column.astype(np.float32)
        rv_values = np.where(sim_rv_values < -8.0, 3.1, sim_rv_values)

    # Prefer heliocentric and CMB redshifts from HEAD table when available.
    # Map by SNID so we can use partial HEAD tables (common in targeted replays).
    head_snid = head_columns.get('SNID')
    head_sim_zhel = head_columns.get('SIM_REDSHIFT_HELIO')
    head_sim_zcmb = head_columns.get('SIM_REDSHIFT_CMB')
    head_zhel = head_columns.get('REDSHIFT_HELIO')
    head_zcmb = head_columns.get('REDSHIFT_CMB')

    head_zhel_source = head_sim_zhel if head_sim_zhel is not None else head_zhel
    head_zcmb_source = head_sim_zcmb if head_sim_zcmb is not None else head_zcmb

    redshift_helio = dump_columns['GENZ'].astype(np.float32)
    if head_snid is not None and head_zhel_source is not None:
        joined = _align_to_dump(head_zhel_source, head_snid,
                                dump_columns['CID'], np.nan)
        redshift_helio = np.where(np.isnan(joined), redshift_helio, joined)

    # the drawn galaxy's redshift (GENLC.REDSHIFT_HOST, sntools_host.c:6481),
    # the SN's own where the HEAD does not record it
    redshift_host = redshift_helio
    if head_snid is not None and 'SIM_REDSHIFT_HOST' in head_columns:
        joined = _align_to_dump(head_columns['SIM_REDSHIFT_HOST'], head_snid,
                                dump_columns['CID'], np.nan)
        redshift_host = np.where(np.isnan(joined), redshift_helio, joined)

    # Fallback: assume helio ~ CMB if not provided
    redshift_cmb = redshift_helio
    if head_snid is not None and head_zcmb_source is not None:
        joined = _align_to_dump(head_zcmb_source, head_snid,
                                dump_columns['CID'], np.nan)
        redshift_cmb = np.where(np.isnan(joined), redshift_helio, joined)

    # Milky Way extinction: use per-SN values from the dump when available,
    # otherwise default to zero (no MW extinction). Replay has no smearing
    # information, so the map reading and the "true" value are identical
    # and mwebv_err is zero.
    dump_mwebv = dump_columns.get('MWEBV')
    if dump_mwebv is not None:
        mwebv_values = dump_mwebv.astype(np.float32)
    else:
        mwebv_values = np.zeros_like(redshift_helio, dtype=np.float32)
    mwebv_err_values = np.zeros_like(mwebv_values, dtype=np.float32)

    # The generated distance modulus, GENLC.DLMU: the dump's MU column
    # (snlc_sim.c:16090) or the HEAD's SIM_DLMU (snlc_sim.c:25471).
    if 'MU' in dump_columns:
        mu = dump_columns['MU'].astype(np.float64)
    elif 'SIM_DLMU' in head_columns:
        mu = _align_to_dump(head_columns['SIM_DLMU'], head_snid,
                            dump_columns['CID'], np.nan).astype(np.float64)
        if np.any(np.isnan(mu)):
            raise SNANAError(
                "the HEAD file does not cover every dump event, so SIM_DLMU "
                "cannot stand in for MU; add MU to SIMGEN_DUMPALL in the SNANA "
                "input and re-run the dump.")
    else:
        raise SNANAError(
            "neither the dump (MU) nor the HEAD (SIM_DLMU) carries the "
            "generated distance modulus; add MU to SIMGEN_DUMPALL in the "
            "SNANA input and re-run the dump.")

    searcheff_mask = head_columns.get('SIM_SEARCHEFF_MASK')
    if searcheff_mask is None:
        searcheff_mask = np.zeros_like(dump_columns['CID'], dtype=np.int32)
    else:
        searcheff_mask = _align_to_dump(searcheff_mask, head_snid,
                                        dump_columns['CID'], 0.0).astype(np.int32)

    n_entries = len(dump_columns['CID'])
    prefilter_pass = np.ones(n_entries, dtype=bool)
    prefilter_reason_code = np.full(
        n_entries,
        PREFILTER_REASON_CODES['PASS'],
        dtype=np.int16
    )

    field_ra, field_dec = field_coords(
        resources, {'field_idx': np.asarray(field_indices, dtype=np.int32)})
    params = {
        'cid': dump_columns['CID'].astype(np.int32),
        # Convention: params['sim_redshift_cmb'] = zCMB (cosmological), params['sim_redshift_helio'] = zHEL (observed).
        'sim_redshift_cmb': redshift_cmb,
        'sim_redshift_helio': redshift_helio,
        'av': av_values,
        'rv': rv_values,
        'sim_av': sim_av_values,
        'sim_rv': sim_rv_values,
        'mwebv': mwebv_values,
        # SNANA head's MWEBV is the adopted value (no independent smearing in
        # replay), so truth = map here. Keeps downstream extinction readers
        # (`params['sim_mwebv']`) working in replay mode.
        'sim_mwebv': mwebv_values,
        'mwebv_err': mwebv_err_values,
        'mu': mu.astype(np.float32),
        'peakmjd': dump_columns['PEAKMJD'],
        'field_idx': np.array(field_indices, dtype=np.int32),
        # the SIMLIB pointing, where the sampler places the SN
        'ra': np.asarray(field_ra),
        'dec': np.asarray(field_dec),
        'sim_searcheff_mask': searcheff_mask,
        'prefilter_pass': prefilter_pass,
        'prefilter_reason_code': prefilter_reason_code,
        # what the sampler writes for every event and replay has no draw
        # for: the MAGOFF-sum terms (a template model fills magoff and
        # template_idx below), the host-lookup redshift and match flag
        'magsmear': np.zeros(n_entries, dtype=np.float32),
        'magoff': np.zeros(n_entries, dtype=np.float32),
        'template_idx': np.zeros(n_entries, dtype=np.int32),
        'mushift': np.zeros(n_entries, dtype=np.float32),
        'sim_redshift_host': redshift_host,
        'CORRECT_HOSTMATCH': np.ones(n_entries, dtype=np.int32),
    }
    # the population is not drawn under replay, so nothing is reweighted;
    # SNANA's own weight rides along where its header carries one
    params['wgt_population'] = (
        _align_to_dump(head_columns['SIM_WGT_POPULATION'], head_snid,
                       dump_columns['CID'], 1.0).astype(np.float32)
        if 'SIM_WGT_POPULATION' in head_columns
        else np.ones(n_entries, dtype=np.float32))

    # Weak lensing deltaMU: the dump column, else the HEAD's SIM_LENSDMU,
    # else zero as an unlensed sim carries.
    lensdmu_col = dump_columns.get('LENSDMU')
    if lensdmu_col is None and 'SIM_LENSDMU' in head_columns:
        lensdmu_col = _align_to_dump(head_columns['SIM_LENSDMU'], head_snid,
                                     dump_columns['CID'], 0.0)
    if lensdmu_col is None:
        lensdmu_col = np.zeros(n_entries, dtype=np.float32)
    params['lensdmu'] = lensdmu_col.astype(np.float32)

    # GENLC.VPEC: the dump's VPEC, else the HEAD's SIM_VPEC; a sim that draws
    # or forces peculiar velocities cannot be replayed without one.
    vpec_col = dump_columns.get('VPEC')
    if vpec_col is None and 'SIM_VPEC' in head_columns:
        vpec_col = _align_to_dump(head_columns['SIM_VPEC'], head_snid,
                                  dump_columns['CID'], np.nan)
        if np.any(np.isnan(vpec_col)):
            vpec_col = None
    if vpec_col is None:
        cosmology = simulation.cosmology
        if cosmology.gensigma_vpec > 0.0 or cosmology.vpec_force < 1.0e9:
            raise SNANAError(
                "neither the dump (VPEC) nor the HEAD (SIM_VPEC) carries the "
                "peculiar velocity the sim generates; add VPEC to "
                "SIMGEN_DUMPALL in the SNANA input and re-run the dump.")
        vpec_col = np.zeros(n_entries, dtype=np.float32)
    params['vpec_true'] = vpec_col.astype(np.float32)
    # no VPEC_ERR smear under replay, as for MWEBV above
    params['vpec'] = params['vpec_true']
    params['vpec_err'] = np.zeros(n_entries, dtype=np.float32)

    # SIM_HOSTLIB(xxx) columns: SNANA's per-SN HOSTLIB values. Pass through
    # as host_<xxx> so JAX SEARCHEFF/WGTMAP evaluation has them available.
    # Align by SNID (HEAD may carry fewer SNe than the dump); -9 sentinel
    # for dump SNe absent from HEAD.
    for col_name in head_columns:
        if col_name.startswith('SIM_HOSTLIB(') and col_name.endswith(')'):
            inner = col_name[len('SIM_HOSTLIB('):-1]
            params[f'host_{inner}'] = _align_to_dump(
                head_columns[col_name], head_snid, dump_columns['CID'], -9.0)

    # Host association does not run under replay, so the true-host DDLR/SNSEP
    # the writer reports come from SNANA's own header.
    for head_name, param_name in (('SIM_SNHOST_DDLR', 'host_DDLR_TRUE'),
                                  ('SIM_SNHOST_SEP', 'host_SNSEP_TRUE'),
                                  ('SIM_REDSHIFT_HOST_MATCH', 'host_ZTRUE')):
        if head_name in head_columns:
            params[param_name] = _align_to_dump(
                head_columns[head_name], head_snid, dump_columns['CID'], -9.0)

    # Host matching does not run under replay either, so the reported match
    # counts come from SNANA's header. They must be taken rather than inferred:
    # a no-HOSTLIB run stores HOSTGAL_OBJID = 0, which the writer's
    # "GALID != -9" fallback would read as a host.
    for head_name, param_name in (('HOSTGAL_NMATCH', 'host_NMATCH'),
                                  ('HOSTGAL_NMATCH2', 'host_NMATCH2')):
        if head_name in head_columns:
            params[param_name] = _align_to_dump(
                head_columns[head_name], head_snid,
                dump_columns['CID'], 0).astype(np.int32)
    # the galaxy that was drawn, which survives a failed DDLR match
    if 'SIM_HOSTLIB_GALID' in head_columns:
        params['host_GALID'] = _align_to_dump(
            head_columns['SIM_HOSTLIB_GALID'], head_snid,
            dump_columns['CID'], -9).astype(np.int64)
        params['drawn_GALID'] = params['host_GALID']

    # Host spectra need the HOSTLIB row, which the header does not carry;
    # recover it by looking the reported GALID back up in the library.
    hostlib = resources.hostlib
    if hostlib is not None and 'HOSTGAL_OBJID' in head_columns:
        row_of = {int(g): i for i, g in enumerate(hostlib.galids)}
        params['host_galaxy_index'] = np.array(
            [row_of.get(int(g), 0) for g in params['host_GALID']],
            dtype=np.int32)

    # Host selection is bypassed under replay, so the galaxy's own columns are
    # never gathered. Look them up from the index recovered above, which gives
    # the writer the same HOSTLIB row SNANA used.
    if 'host_galaxy_index' in params:
        gal_idx = np.asarray(params['host_galaxy_index'])
        zeros = np.zeros(len(gal_idx), dtype=np.int32)
        host_props = hostlib.selector.gather_host_properties(
            hostlib.catalogue, gal_idx, zeros)
        # Two different conditions. Host properties belong to the galaxy that
        # was DRAWN, which still contributes surface brightness after a failed
        # DDLR match. Positions are only meaningful when the match SUCCEEDED --
        # SNANA writes HOSTGAL_RA/DEC as -999 otherwise, and differencing
        # against that sentinel yields a separation of millions of arcsec.
        # SNANA derives an unmatched event's surface brightness from a SN-galaxy
        # position it never records (HOSTGAL_RA/DEC are -999 there), so it
        # cannot be reproduced: placing the SN at the galaxy centre overshoots
        # by ~1000x, and differencing the sentinel coordinates is nonsense.
        # Nulling the host properties leaves an error of at most a couple of
        # FLUXCAL against a range reaching 150.
        matched = has_position = np.asarray(params.get(
            'host_NMATCH2', np.ones(len(gal_idx), dtype=np.int32))) > 0
        for key, value in host_props.items():
            v = np.asarray(value)
            if v.dtype.kind == 'f':
                null = 999.0 if key.endswith('_obs') else -9999.0
                v = np.where(matched, v, null)
            params.setdefault(f'host_{key}', v)
        # The SN's offset from its host is drawn during selection, so replay has
        # none and would leave the SN at the galaxy centre, which peaks the
        # surface brightness. SNANA's header records both positions, so the
        # exact offset is recoverable rather than redrawn.
        if {'RA', 'DEC', 'HOSTGAL_RA', 'HOSTGAL_DEC'} <= set(head_columns):
            al = lambda name: _align_to_dump(
                head_columns[name], head_snid, dump_columns['CID'], np.nan)
            sn_ra, sn_dec = al('RA'), al('DEC')
            gal_ra, gal_dec = al('HOSTGAL_RA'), al('HOSTGAL_DEC')
            cosdec = np.cos(np.radians(sn_dec))
            ra_off = np.where(has_position, (sn_ra - gal_ra) * cosdec * 3600.0, -9.0)
            dec_off = np.where(has_position, (sn_dec - gal_dec) * 3600.0, -9.0)
            # Differencing two ~10-30 degree float32 coordinates to recover a
            # ~1 arcsec offset loses most of the precision: the storage step
            # scales with the coordinate, not the separation, leaving ~0.01
            # arcsec. SIM_SNHOST_SEP stores the separation itself, so its step
            # is ~1e-7 arcsec. Take the direction from the coordinates and the
            # magnitude from that column.
            if 'SIM_SNHOST_SEP' in head_columns:
                sep_exact = _align_to_dump(
                    head_columns['SIM_SNHOST_SEP'], head_snid,
                    dump_columns['CID'], -9.0)
                norm = np.hypot(ra_off, dec_off)
                scale = np.where((norm > 0) & (sep_exact > 0),
                                 sep_exact / np.where(norm > 0, norm, 1.0), 1.0)
                ra_off = ra_off * scale
                dec_off = dec_off * scale
            params.setdefault('host_RA_OFFSET', ra_off)
            params.setdefault('host_DEC_OFFSET', dec_off)
            # use the stored separation directly; recomputing it from the
            # rescaled components just adds rounding back in
            params.setdefault('host_SNSEP', np.where(
                has_position, sep_exact, -9.0)
                if 'SIM_SNHOST_SEP' in head_columns
                else np.where(has_position, np.hypot(ra_off, dec_off), -9.0))
            # GALFRAC is evaluated at the SN's position in the galaxy's own
            # frame; without it the SN sits at the centre and the surface
            # brightness peaks. Invert the sky-offset rotation from
            # hostlib.sersic to recover it.
            a_rot = np.radians(np.asarray(params.get(
                'host_a_rot', np.zeros(len(gal_idx)))))
            cos_r, sin_r = np.cos(a_rot), np.sin(a_rot)
            params.setdefault('host_a_SNGALSEP', np.where(
                has_position, ra_off * cos_r - dec_off * sin_r, 0.0))
            params.setdefault('host_b_SNGALSEP', np.where(
                has_position, ra_off * sin_r + dec_off * cos_r, 0.0))

        # Reported host mags are shifted to the SN distance (sntools_host.c:8805).
        params.setdefault('host_dmucor', np.asarray(
            compute_host_dmucor(resources, params, host_props)))

    # STOREPAR passthroughs are keyed sim_hostlib_<name> for the writer, while
    # the host_<name> copies above feed WGTMAP/SEARCHEFF evaluation.
    for col_name in head_columns:
        if col_name.startswith('SIM_HOSTLIB(') and col_name.endswith(')'):
            inner = col_name[len('SIM_HOSTLIB('):-1]
            params[f'sim_hostlib_{inner}'] = _align_to_dump(
                head_columns[col_name], head_snid, dump_columns['CID'], -9.0)

    # Observed host properties, which the HOSTLIB would otherwise supply.
    for col_name in head_columns:
        if col_name in (
                'HOSTGAL_LOGMASS', 'HOSTGAL_LOGMASS_ERR',
                'HOSTGAL_LOGSFR', 'HOSTGAL_LOGSFR_ERR',
                'HOSTGAL_LOGsSFR', 'HOSTGAL_LOGsSFR_ERR',
                'HOSTGAL_COLOR', 'HOSTGAL_COLOR_ERR'):
            params[f"host_{col_name[len('HOSTGAL_'):]}"] = _align_to_dump(
                head_columns[col_name], head_snid, dump_columns['CID'], -9.0)
    # The header stores the quantiles as an unlabelled vector, so the host's
    # own row supplies them, rescaled to the SN redshift as in a real run.
    if 'host_galaxy_index' in params:
        gal_idx = np.asarray(params['host_galaxy_index'])
        quantiles = {'reported_galaxy_index': gal_idx}
        load_hostlib_zphot_quantiles(hostlib.zphot_quantiles, hostlib.catalogue,
                                     quantiles, params['sim_redshift_helio'],
                                     gal_idx, simulation.host.mskopt)
        for key, value in quantiles.items():
            params[f'host_{key}'] = np.asarray(value)


    # SIM_MAGSMEAR_COH: per-SN coherent smear, SNID-aligned; NON1ASED
    # overrides it below. Under GENMAG_SMEAR_MODELNAME the column instead holds
    # the smear model's own coherent term (snlc_sim.c:30007): inject that only
    # when the model's randoms are zeroed, or it would be counted twice.
    head_magsmear = head_columns.get('SIM_MAGSMEAR_COH')
    if head_magsmear is not None and (
            resources.gensmear is None or drop_node_smear):
        params['magsmear'] = _align_to_dump(head_magsmear, head_snid,
                                            dump_columns['CID'], 0.0)

    # The per-node smear draws are absent from SNANA's output, so a run is only
    # replayable when GENSMEAR_RANGAUSS_FIX pins them (sntools_genSmear.c:307);
    # jax's sampler applies the same pin, so they need no override here.
    if resources.gensmear is not None and drop_node_smear:
        params['_gensmear_randoms'] = np.zeros(
            (len(dump_columns['CID']),
             resources.gensmear.n_randoms), dtype=np.float32)
    elif resources.gensmear is not None and \
            simulation.model.gensmear_rangauss_fix <= -99.0:
        raise SNANAError(
            "cannot replay GENMAG_SMEAR_MODELNAME "
            f"{simulation.model.genmag_smear_modelname}: SNANA does "
            "not record its per-node smear randoms, and this input does not "
            "pin them with GENSMEAR_RANGAUSS_FIX; pass drop_node_smear=True "
            "to replay with only the coherent term")

    model_params, invalid = resources.model.replay_params_from_dump(
        dump_columns, head_columns, mu=mu,
        distributions={name: pop.distributions[0]
                       for name, pop in simulation.model.populations.items()
                       if pop.distributions})
    params.update(model_params)
    if invalid is not None and np.any(invalid):
        prefilter_pass[invalid] = False
        prefilter_reason_code[invalid] = PREFILTER_REASON_CODES['NEPOCH']

    # the grey offset is drawn with the parameters it may depend on, so a
    # replay has to build it from the same function the sampler uses
    params['genmag_off'] = draw_grey_offset(
        resources, params,
        {name[len('host_'):]: value for name, value in params.items()
         if name.startswith('host_')})

    return params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay SNANA SIMGEN_DUMP through JAX pipeline.")
    parser.add_argument("--input", "-i", required=True,
                        help="Path to SNANA .input configuration file.")
    parser.add_argument("--dump", "-d", required=True,
                        help="Path to SNANA SIMGEN_DUMP file.")
    parser.add_argument("--head", "-H", default=None,
                        help="Path to SNANA *_HEAD.FITS file containing SIM_PEAKMJD. "
                             "If not provided, will infer from the dump path (required for exact peak alignment).")
    parser.add_argument("--output", "-o", required=True,
                        help="Output version name for JAX results.")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Chunk size for GPU processing.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional limit on number of dump entries to replay (for debug runs).")
    parser.add_argument("--wavelength-grid-points", type=int, default=500,
                        help="Number of wavelength grid points for SED integration (default: 500).")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable DEBUG logging.")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        format='%(message)s',
        level=logging.DEBUG if args.verbose else logging.INFO,
        stream=sys.stderr,
    )
    snana_input = Path(args.input).expanduser().resolve()
    dump_path = Path(args.dump).expanduser().resolve()
    head_path = Path(args.head).expanduser().resolve() if args.head else None

    snana_config = parse_snana_input_file(snana_input)
    if snana_config.model.subpops:
        raise SNANAError(
            "SUBPOP has no SNANA counterpart, so a SNANA dump carries no "
            "class for its events and replaying one would silently generate "
            "a single population. Replay the input without its SUBPOP keys.",
            function_name="replay_snana_dump")
    dump_data = load_snana_dump(dump_path, dedup_cid=True)
    if args.limit is not None and args.limit > 0:
        dump_columns_limited = {k: v[: args.limit] for k, v in dump_data.columns.items()}
        dump_data = type(dump_data)(columns=dump_columns_limited)
    n_entries = dump_data.size
    if n_entries == 0:
        raise SNANAError("No events found in SNANA dump.")

    logger.info(f"Loaded {n_entries} entries from {dump_path}")

    # Load true generated peak times from the SNANA HEAD file (required for exact parity).
    if head_path is None:
        inferred = dump_path.parent / f"{dump_path.stem}_HEAD.FITS"
        if inferred.exists():
            head_path = inferred
    if head_path is None or not head_path.exists():
        raise SNANAError(
            "SNANA HEAD file with SIM_PEAKMJD is required; pass via --head "
            "or ensure <dump>_HEAD.FITS exists.",
            function_name="main")
    with fits.open(head_path) as hdul:
        head_table = hdul[1].data
        head_columns = {name: head_table[name] for name in head_table.names}
        snids = head_table["SNID"]
        sim_peak = head_table["SIM_PEAKMJD"]
        sim_peak_mjd_map = {int(snid): float(pk) for snid, pk in zip(snids, sim_peak)}
    logger.info(f"Loaded SIM_PEAKMJD for {len(sim_peak_mjd_map)} SNe from {head_path}")

    config = dict(
        n_sne=n_entries,
        batch_size=args.batch_size,
        wavelength_grid_points=args.wavelength_grid_points,
    )
    snana_config.numerical = replace(snana_config.numerical, **config)
    sim = sim_run.prepare(snana_config)
    simulation = None if sim is None else sim.config
    resources = sim

    override_params = build_override_parameters(
        dump_data.columns, simulation, resources, head_columns=head_columns)

    # Replace peak_mjd with true simulated peaks to match SNANA exactly.
    sn_ids = override_params['cid']
    peak_mjd_true = []
    missing = []
    for snid in sn_ids:
        pk = sim_peak_mjd_map.get(int(snid))
        if pk is None:
            missing.append(int(snid))
            pk = override_params['peakmjd'][len(peak_mjd_true)]
        peak_mjd_true.append(pk)
    override_params['peakmjd'] = np.asarray(peak_mjd_true, dtype=np.float32)
    if missing:
        logger.warning(f"Missing SIM_PEAKMJD for {len(missing)} SNe; kept dump PEAKMJD for those.")

    results, selection, spectra = sim_run.run_simulation(sim, override_parameters=override_params)
    save_snana_outputs(sim, results, args.output, selection, spectra_by_sn=spectra)


if __name__ == "__main__":
    main()
