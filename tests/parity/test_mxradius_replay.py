"""MXRADIUS_RANDOM_SHIFT geometry parity.

The shift is a random draw, so jax's keyed PRNG cannot reproduce SNANA's
sequential stream and per-SN parity on the *draw* is impossible. The geometry,
however, is exact and testable: SNANA dumps the drawn radius/phi alongside the
resulting RA/DEC offsets, so pushing SNANA's own r/phi/DEC through jax's formula
must reproduce SNANA's shift_RA/shift_DEC to print precision.

That pins the parts that could silently be wrong -- the 1/cos(DEC) factor, which
axis takes sin vs cos, and the sign convention -- independently of the RNG. The
draw itself is checked distributionally (uniform in area, phi over 0..2pi).
"""

import numpy as np

from jax_snlc_sim.sampler import coord_shift_from_polar  # noqa: E402


def _load_dump(path):
    """{column: array} from a SNANA SIMGEN_DUMP file."""
    hdr = None
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith('VARNAMES'):
                hdr = line.split()[1:]
            elif line.startswith('SN:'):
                rows.append(line.split()[1:])
    return {h: np.array([float(r[i]) for r in rows]) for i, h in enumerate(hdr)}


def test_mxradius_geometry_matches_snana(snana_fixture):
    """jax's shift formula reproduces SNANA's shift_RA/shift_DEC exactly when
    fed SNANA's own drawn radius/phi (gen_random_coord_shift, snlc_sim.c:13152).

    DEC in the dump is post-shift, so the original DEC -- the one SNANA took its
    single COSDEC from (:13139) -- is recovered as DEC - shift_DEC.
    """
    d = _load_dump(str(snana_fixture("mxradius_replay").dump))
    r, phi = d['shift_RADIUS'], d['shift_PHI']
    dec0 = d['DEC'] - d['shift_DEC']

    shift = coord_shift_from_polar(r, phi, dec0)

    assert len(r) > 300
    assert np.abs(shift['ra'] - d['shift_RA']).max() < 1e-5
    assert np.abs(shift['dec'] - d['shift_DEC']).max() < 1e-5


def test_mxradius_draw_is_uniform_in_area(snana_fixture):
    """SNANA draws r = MXRADIUS*sqrt(u), so r^2 is uniform on [0, R^2]; jax uses
    the same form. Guards the fixture against a degenerate radius distribution."""
    d = _load_dump(str(snana_fixture("mxradius_replay").dump))
    R = 0.05
    r, phi = d['shift_RADIUS'], d['shift_PHI']
    assert r.max() <= R + 1e-9
    u = (r / R) ** 2
    assert abs(u.mean() - 0.5) < 0.05
    assert phi.min() >= 0.0 and phi.max() <= 2 * np.pi
    assert abs(phi.mean() - np.pi) < 0.25
