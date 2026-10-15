"""SNMAGSHIFT (HOSTLIB_MSKOPT bit 4) parity vs SNANA.

The bundled WGTMAP ramps SNMAGSHIFT linearly with LOGMASS, so every host
carries its own shift and the interpolation is exercised rather than a single
constant. SNANA records the value it applied to each SN as ``GALSNDM`` in the
SIMGEN dump, which is what jax's per-galaxy shift is compared against.
"""

import numpy as np


from jax_snlc_sim import run as sim_run


def _dump_rows(path):
    names, rows = None, []
    with open(path) as fh:
        for line in fh:
            words = line.split()
            if line.startswith("VARNAMES:"):
                names = words[1:]
            elif line.startswith("SN:") and names:
                rows.append(dict(zip(names, words[1:])))
    return rows


def test_snmagshift_matches_snana(snana_fixture):
    fx = snana_fixture("hostlib_mskopt_alt")
    config = dict(n_sne=4, random_seed=42)
    sim = sim_run.prepare(str(fx.input_file), **config)
    resources = sim
    shift = np.asarray(resources.hostlib.catalogue.snmagshift)
    galid = resources.hostlib.galids
    jax_shift = dict(zip(galid, shift))

    rows = _dump_rows(fx.dump)
    snana = np.array([float(r["GALSNDM"]) for r in rows])
    ours = np.array([jax_shift[int(float(r["GALID"]))] for r in rows])

    # Guard against a vacuous pass: the reference must actually vary. Without
    # bit 4, or with a WGTMAP lacking a SNMAGSHIFT column, GALSNDM is all zero
    # and any comparison below would succeed while testing nothing.
    assert np.ptp(snana) > 0.05, f"GALSNDM range {np.ptp(snana):.4f} -- shift inactive"

    # GALSNDM is written to 4 decimals, so half a print step is the floor here.
    assert np.max(np.abs(ours - snana)) <= 5.1e-5
