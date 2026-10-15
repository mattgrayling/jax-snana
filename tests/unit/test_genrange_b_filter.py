"""Galactic-b SIMLIB-field-level filter."""

import numpy as np

from jax_snlc_sim.simulation import Survey
from jax_snlc_sim.builder import InputGenranges, _apply_simlib_filters
from jax_snlc_sim.survey.simlib_reader import SimlibField, _compute_seasons


def _make_field(libid=0, ra=0.0, dec=0.0):
    mjds = np.linspace(60000.0, 60050.0, 10).astype(np.float32)
    n = len(mjds)
    z = np.zeros(n, dtype=np.float32)
    starts, ends = _compute_seasons(mjds)
    return SimlibField(
        libid=libid, ra=ra, dec=dec, mwebv=0.0, pixsize=0.2,
        nobs=n, mjds=mjds,
        filter_indices=np.zeros(n, dtype=np.int8),
        skysigs=z, psf1s=z + 1.0, psf2s=z, psf_ratios=z, neas=z + 4.0,
        zptavgs=z + 27.0, zpterrs=z, gains=z + 1.0, readnoises=z,
        nexposes=z + 1.0,
        template_skysigs=z, template_readnoises=z, template_zpts=z + 27.0,
        append_photflags=np.zeros(n, dtype=np.int32),
        season_starts=starts, season_ends=ends,
    )


class _MockSimlib:
    def __init__(self, fields):
        self.fields = list(fields)
    def _rebuild_flat_structures(self):
        pass


def _wide_ranges(sim):
    """Input ranges wide enough to keep every entry, so only b filters."""
    return InputGenranges((0.01, 1.0), (59000.0, 62000.0), {},
                          np.full(len(sim.fields), np.nan))


def test_default_range_no_op():
    fields = [_make_field(libid=i, ra=ra, dec=dec)
              for i, (ra, dec) in enumerate([(120.0, 30.0), (0.0, 0.0)])]
    sim = _MockSimlib(fields)
    _apply_simlib_filters(sim, Survey(), _wide_ranges(sim))
    assert len(sim.fields) == 2


def test_narrow_b_drops_low_galactic_latitude():
    galactic_plane_ra, galactic_plane_dec = 266.4, -28.9
    high_b_ra, high_b_dec = 192.86, 27.13
    fields = [
        _make_field(libid=0, ra=galactic_plane_ra, dec=galactic_plane_dec),
        _make_field(libid=1, ra=high_b_ra, dec=high_b_dec),
    ]
    sim = _MockSimlib(fields)
    _apply_simlib_filters(sim, Survey(genrange_b=(20.0, 90.0)),
                          _wide_ranges(sim))
    assert [f.libid for f in sim.fields] == [1]


