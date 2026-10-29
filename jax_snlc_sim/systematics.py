"""RANSYSTPAR: one random draw per job of systematic-parameter shifts.

Mirrors SNANA prep_RANSYSTPAR (snlc_sim.c:8135-8349): a synced phase seeded
by RANSEED_SYST (same draws for every survey) sets MWEBV scale, MW RV,
host-zphot bias, redshift bias and cosmology shifts; an unsynced phase
(seed + 137*IDSURVEY) sets flux-error scales and per-filter ZP and
wavelength shifts. Scale draws are 1 + sigma*G, shift draws sigma*G, with
G a +/-3 clipped standard normal (rejection sampled); RANGESHIFT draws are
uniform. Each drawn value lands in the corresponding deterministic config
knob and is logged, as SNANA does.

The GENMODEL/GENPDF_FILE wildcard picks live here too but are invoked by
the parser BEFORE post-processing (inputs
_resolve_ransystpar_wildcards): the picked files feed parse-time
derivations (model_type, GENPDF map load), so they cannot wait for
prep_ransystpar. Each pick uses its own sub-stream of the synced seed.

SIGSHIFT_LAMFILT shifts the filter wavelengths here through FUDGESHIFT_LAM;
SNANA's simulator does not apply it. Generation random numbers are not
re-seeded. RANSEED_GEN overrides the simulation's seed, as in SNANA.
"""

import glob
import logging
import os

import numpy as np

from jax_snlc_core import logging as log
from jax_snlc_core.logging import SNANAError
from jax_snlc_sim.simlib import survey_from_simlib_file

logger = logging.getLogger(__name__)

_CLIP = 3.0
_WILDCARD_STREAM = {'GENMODEL_WILDCARD': 0, 'GENPDF_FILE_WILDCARD': 1}


def pick_wildcard_file(wildcard, seed, which, jobid=0, njobtot=0):
    """SNANA pick_RANSYSTFILE_WILDCARD (snlc_sim.c:8352): glob the pattern
    and pick one file, sequentially by JOBID for batch jobs (NJOBTOT>0),
    else with a flat draw."""
    files = sorted(glob.glob(os.path.expandvars(wildcard)))
    if not files:
        raise SNANAError(f"no files match RANSYSTPAR_{which}: {wildcard}")
    if njobtot > 0:
        ifile = jobid - 1
        if not 0 <= ifile < len(files):
            raise SNANAError(
                f"invalid JOBID={jobid} for RANSYSTPAR_{which}: NJOBTOT="
                f"{njobtot} batch jobs but {len(files)} matched files")
    else:
        rng = np.random.default_rng([seed, _WILDCARD_STREAM[which]])
        ifile = int(rng.uniform() * len(files))
    logger.info(
        f"RANSYSTPAR: {which} pick {ifile} of {len(files)}: {files[ifile]}")
    return files[ifile]


# SNANA checkval_F bounds on SIGSCALE_* (snlc_sim.c:8156)
_SIGSCALE_MIN, _SIGSCALE_MAX = -1.0e-6, 0.2


def _checkval_sigscale(name, val):
    if not _SIGSCALE_MIN <= val <= _SIGSCALE_MAX:
        raise SNANAError(
            f"RANSYSTPAR_{name} = {val} outside SNANA bounds "
            f"[{_SIGSCALE_MIN}, {_SIGSCALE_MAX}]")


def _clip_normal(rng):
    while True:
        x = rng.standard_normal()
        if -_CLIP <= x <= _CLIP:
            return x


def _filter_groups(rs, base, genfilters):
    """Per-band sigma from KEY and KEY(group,group,...) entries.

    SNANA reads one value per comma-separated filter group in the
    parenthesised argument, and the bare key expands to one group per
    GENFILTERS filter, taking NFILT values; a missing value is fatal
    (parse_input_RANSYSTPAR, snlc_sim.c:4706-4738).
    """
    sigmas = {}
    for key, val in rs.items():
        if key == base:
            groups = list(genfilters)
        elif key.startswith(base + '(') and key.endswith(')'):
            groups = key[len(base) + 1:-1].split(',')
        else:
            continue
        values = val if isinstance(val, (list, tuple)) else str(val).split()
        if len(values) != len(groups):
            raise SNANAError(
                f"RANSYSTPAR_{key}: expected {len(groups)} values "
                f"(one per filter group), got {len(values)}",
                function_name="_filter_groups")
        for group, v in zip(groups, values):
            for c in group:
                sigmas[c] = float(v)
    return sigmas


def prep_ransystpar(config) -> None:
    """Draw and apply RANSYSTPAR shifts by mutating ``config`` in place."""
    rs = dict(config.run.ransystpar)
    if not rs:
        return
    genfilters = config.survey.genfilters

    def get(key, default=0.0):
        return float(rs.get(key, default))

    seed_syst = int(rs.get('RANSEED_SYST', 0))
    seed0 = seed_syst if seed_syst > 0 else config.run.random_seed
    rng = np.random.default_rng(seed0)
    logger.info(f"RANSYSTPAR: synced seed = {seed0}")

    sig = get('SIGSCALE_MWEBV')
    _checkval_sigscale('SIGSCALE_MWEBV', sig)
    if sig != 0.0:
        config.mw_dust.scale = 1.0 + sig * _clip_normal(rng)
        log.line(logger, f"MWEBV_SCALE = {config.mw_dust.scale:.4f}")

    sig = get('SIGSHIFT_MWRV')
    if sig != 0.0:
        config.mw_dust.rv += sig * _clip_normal(rng)
        log.line(logger, f"RV_MWCOLORLAW = {config.mw_dust.rv:.3f}")

    sig = get('SIGSHIFT_zPHOT_HOST')
    if sig != 0.0:
        # SNANA assigns only HOSTLIB_GENZPHOT_BIAS[0] (snlc_sim.c:8199-8205),
        # keeping any user-supplied higher-order bias coefficients.
        bias = list(config.host.genzphot_bias or [0.0])
        bias[0] = sig * _clip_normal(rng)
        config.host.genzphot_bias = bias
        log.line(logger, f"HOSTLIB_GENZPHOT_BIAS[0] = {bias[0]:.5f}")

    sig = get('SIGSHIFT_REDSHIFT')
    if sig != 0.0:
        config.rates.genbias_redshift = sig * _clip_normal(rng)
        log.line(logger, f"GENBIAS_REDSHIFT = {config.rates.genbias_redshift:.6f}")

    def shifted_cosmology_value(name, value):
        """Apply the Gaussian shift, then the optional uniform shift."""
        sig = get(f'SIGSHIFT_{name}')
        if sig != 0.0:
            value += sig * _clip_normal(rng)
            log.line(logger, f"{name} = {value:.4f}")
        rng_range = rs.get(f'RANGESHIFT_{name}')
        if rng_range is not None:
            lo, hi = (float(x) for x in (
                rng_range if isinstance(rng_range, (list, tuple))
                else str(rng_range).split()))
            if hi > lo:
                value = value + lo + (hi - lo) * rng.uniform()
                log.line(logger, f"{name} = {value:.4f}")
        return value

    config.cosmology.omega_matter = shifted_cosmology_value(
        'OMEGA_MATTER', config.cosmology.omega_matter)
    config.cosmology.w0 = shifted_cosmology_value('W0', config.cosmology.w0)

    idsurvey = int(rs.get('IDSURVEY', 0))
    if idsurvey <= 0:
        _, idsurvey = survey_from_simlib_file(
            config.survey.simlib_file, config.survey.survey_override)
    rng = np.random.default_rng(seed0 + 137 * idsurvey)
    logger.info(f"RANSYSTPAR: unsynced seed = {seed0 + 137 * idsurvey}")

    sig = get('SIGSCALE_FLUXERR')
    _checkval_sigscale('SIGSCALE_FLUXERR', sig)
    if sig != 0.0:
        val = 1.0 + sig * _clip_normal(rng)
        config.instrument.fudgescale_fluxerr = {c: val for c in genfilters}
        log.line(logger, f"FUDGESCALE_FLUXERR = {val:.4f}")

    sig = get('SIGSCALE_FLUXERR2')
    _checkval_sigscale('SIGSCALE_FLUXERR2', sig)
    if sig != 0.0:
        val = 1.0 + sig * _clip_normal(rng)
        config.instrument.fudgescale_fluxerr2 = {c: val for c in genfilters}
        log.line(logger, f"FUDGESCALE_FLUXERR2 = {val:.4f}")

    zp_sigmas = _filter_groups(rs, 'SIGSHIFT_ZP', genfilters)
    for c in genfilters:
        if zp_sigmas.get(c, 0.0) != 0.0:
            config.instrument.genmag_off_zp[c] = zp_sigmas[c] * _clip_normal(rng)
            log.line(logger, f"ZPerr({c}) = {config.instrument.genmag_off_zp[c]:+.5f}")

    lam_sigmas = _filter_groups(rs, 'SIGSHIFT_LAMFILT', genfilters)
    for c in genfilters:
        if lam_sigmas.get(c, 0.0) != 0.0:
            config.instrument.fudgeshift_lam[c] = lam_sigmas[c] * _clip_normal(rng)
            log.line(logger, f"LAMSHIFT({c}) = {config.instrument.fudgeshift_lam[c]:+.2f} A")

    seed_gen = int(rs.get('RANSEED_GEN', 0))
    if seed_gen > 0:
        config.run.random_seed = seed_gen
        log.line(logger, f"RANSEED_GEN override: random_seed = {seed_gen}")
