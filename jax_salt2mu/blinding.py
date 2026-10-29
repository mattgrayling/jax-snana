"""BBC blinding of the reference cosmology and binned distance offsets."""

import math
from dataclasses import replace

import numpy as np


# blindflag bits, mirroring SALT2mu.c:1619-1621.
BLINDMASK_MUz = 1        # add cos(10*z) to each M0DIF bin
BLINDMASK_FIXPAR = 2     # offset the fixed reference OL/w0
BLINDMASK_SIM = 64       # extend blinding to sims, which are otherwise unblinded


# blind offset = amplitude * cos(phase), per fit-parameter slot (SALT2mu.c:6749-6753).
BLINDPAR_DEFAULT = {9: (0.06, 23434.0), 11: (0.20, 8430.0)}
BLIND_COSINE_M0 = 10.0   # cos(BLIND_COSINE_M0 * z) for MUz (SALT2mu.c:25191)


def resolve_blinding(config, isdata_real: bool):
    """The config with the reference cosmology offset by ``amp * cos(phase)`` and
    ``blindflag`` reduced to what actually applies (apply_blindpar / ISBLIND_FIXPAR,
    SALT2mu.c:3154/25141).

    ``unblind`` keeps the pre-offset values, which SNANA holds in COSPAR_UNBLIND
    (:22195) and the CC prior reads. Called once the data read has established
    real-vs-sim, before any reference distance is computed (:6905).
    """
    blindflag = config.blindflag
    if not isdata_real and not (blindflag & BLINDMASK_SIM):
        blindflag = 0        # sims are unblinded unless asked for (:8844)
    if not (blindflag & BLINDMASK_FIXPAR):
        return replace(config, blindflag=blindflag)
    unblind, blinded = dict(config._unblind), {}
    for slot, attr, value in ((9, "omega_l", config.omega_l), (11, "w0", config.w0)):
        amp, phase = config.blindpar.get(slot, (0.0, 0.0))
        if amp == 0.0:
            continue
        base = unblind.setdefault(attr, value)
        blinded[attr] = base + amp * math.cos(phase)
    return replace(config, blindflag=blindflag, _unblind=unblind, **blinded)


def unblinded(config):
    """The config with the reference cosmology back at the values the user gave
    (COSPAR_UNBLIND, SALT2mu.c:22195)."""
    return replace(config, **config._unblind) if config._unblind else config


def blind_distance_offsets(config, offsets, redshifts):
    """Apply MUz blinding before un-floated bins are set to zero."""
    if config.blindflag & BLINDMASK_MUz:
        return offsets + np.cos(BLIND_COSINE_M0 * redshifts)
    return offsets
