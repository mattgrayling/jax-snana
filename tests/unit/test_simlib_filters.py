"""Field-level filter behavior + assignment formula tests."""

import numpy as np
import pytest

from jax_snlc_core.filters import filter_band
from jax_snlc_sim.simlib import (
    InputGenranges, SimlibData, SimlibField, _compute_seasons, apply_simlib_filters,
    simlib_field_index, spell_entry_headers)
from jax_snlc_sim.population import population_spellings
from jax_snlc_core.random import root_key
from jax_snlc_sim.simulation import Survey


def _make_field(libid=0, ra=0.0, dec=0.0, mwebv=0.0, mjds=None, nrest=10):
    """Build a minimal SimlibField for filter tests."""
    if mjds is None:
        mjds = np.linspace(60000.0, 60050.0, nrest).astype(np.float32)
    n = len(mjds)
    z = np.zeros(n, dtype=np.float32)
    starts, ends = _compute_seasons(mjds)
    return SimlibField(
        libid=libid, ra=ra, dec=dec, mwebv=mwebv, pixsize=0.2,
        nobs=n, mjds=mjds,
        filter_indices=np.zeros(n, dtype=np.int8),
        skysigs=z, psf1s=z + 1.0, psf2s=z, psf_ratios=z, neas=z + 4.0,
        zptavgs=z + 27.0, zpterrs=z, gains=z + 1.0, readnoises=z,
        nexposes=z + 1.0,
        template_skysigs=z, template_readnoises=z, template_zpts=z + 27.0,
        append_photflags=np.zeros(n, dtype=np.int32),
        season_starts=starts, season_ends=ends,
    )


def _ranges(sim, redshift=(0.01, 1.0), peakmjd=(59000.0, 62000.0), population=None,
            distance_z=None):
    """Input ranges wide enough to keep every entry the test builds."""
    n = len(sim.fields)
    return InputGenranges(redshift, peakmjd, population or {},
                          np.full(n, np.nan) if distance_z is None
                          else np.asarray(distance_z, dtype=float))


@pytest.mark.parametrize('native', [False, True])
def test_named_population_ranges_constrain_simlib_entries(native):
    from jax_snlc_sim.inputs import build_config
    from jax_snlc_sim.inputs import build
    from jax_snlc_sim.distributions import GenGaussAsym

    population = (GenGaussAsym(peak=0., sigma=(1., 1.), range=(-1., 1.)) if native else
                  {'distribution': 'normal', 'peak': 0., 'sigma': 1., 'range': [-1., 1.]})
    named = build_config(dict(genmodel='SALT3.K21', genfilters='gr', random_seed=1,
        ngentot_lc=1, simlib_file='survey.simlib', kcor_file='survey.fits', opt_mwebv=0,
        redshift_range=[.02, .4], trest_range=[-20, 50], peakmjd_range=[60000, 60100],
        use_simlib_salt2=True, populations={'SALT2x1': population}))
    snana = build(dict(GENMODEL='SALT3.K21', GENFILTERS='gr', RANSEED=1,
        NGENTOT_LC=1, SIMLIB_FILE='survey.simlib', KCOR_FILE='survey.fits', OPT_MWEBV=0,
        GENRANGE_REDSHIFT=[.02, .4], GENRANGE_TREST=[-20, 50], GENRANGE_PEAKMJD=[60000, 60100],
        USE_SIMLIB_SALT2=1, GENPEAK_SALT2x1=0., GENSIGMA_SALT2x1=[1., 1.], GENRANGE_SALT2x1=[-1., 1.]))
    for cfg in (named, snana):
        simlib = SimlibData(simlib_file='', fields=[_make_field(libid=i) for i in range(3)])
        for entry, value in zip(simlib.fields, [-2., 0., 2.]):
            entry.header_values['SALT2x1'] = str(value)
        spell_entry_headers(simlib, population_spellings(
            *cfg.model.population_declaration()[:2]))
        apply_simlib_filters(simlib, cfg.survey,
                             _ranges(simlib, population=cfg.model._input_genranges))
        assert [entry.libid for entry in simlib.fields] == [1]


def test_minseason_drops_the_season_not_its_epochs():
    """A short season leaves the entry's season bookkeeping but its epochs stay
    (remove_short_SIMLIB_SEASON, snlc_sim.c:21584). SNANA narrows
    MJDRANGE_SEASON/SURVEY and never touches the observations."""
    mjds = np.concatenate([
        np.linspace(60000.0, 60200.0, 20),
        np.linspace(60500.0, 60510.0, 5),
    ]).astype(np.float32)
    sim = SimlibData(simlib_file='', fields=[_make_field(mjds=mjds)])
    apply_simlib_filters(sim, Survey(simlib_minseason=30.0), _ranges(sim))
    f = sim.fields[0]
    assert f.nobs == 25 and f.mjds.max() == pytest.approx(60510.0)
    assert int(np.sum(np.isfinite(f.season_starts))) == 1
    assert np.max(f.season_ends) == pytest.approx(60200.0)


def test_minseason_keeps_a_libid_with_no_surviving_season():
    """SNANA leaves such a LIBID in the cycle with MJDRANGE_SURVEY = -9
    (snlc_sim.c:21649); dropping it would shift every later event's entry."""
    mjds = np.concatenate([
        np.linspace(60000.0, 60010.0, 5),
        np.linspace(60500.0, 60510.0, 5),
    ]).astype(np.float32)
    sim = SimlibData(simlib_file='', fields=[_make_field(mjds=mjds)])
    apply_simlib_filters(sim, Survey(simlib_minseason=30.0), _ranges(sim))
    assert len(sim.fields) == 1
    f = sim.fields[0]
    assert f.nobs == 10
    # No season survives, so the survey range inverts and rejects every event.
    assert not np.isfinite(f.season_starts).any()
    assert np.min(f.season_starts) > np.max(f.season_ends)


def test_minobs_drops_small_libids():
    fields = [_make_field(libid=1, nrest=3), _make_field(libid=2, nrest=20)]
    sim = SimlibData(simlib_file='', fields=fields)
    apply_simlib_filters(sim, Survey(simlib_minobs=10), _ranges(sim))
    assert [f.libid for f in sim.fields] == [2]


def test_idlock_keeps_only_locked():
    fields = [_make_field(libid=i) for i in (1, 2, 3)]
    sim = SimlibData(simlib_file='', fields=fields)
    apply_simlib_filters(sim, Survey(simlib_idlock=2), _ranges(sim))
    assert [f.libid for f in sim.fields] == [2]


def test_idskip_drops_listed():
    fields = [_make_field(libid=i) for i in (1, 2, 3, 4)]
    sim = SimlibData(simlib_file='', fields=fields)
    apply_simlib_filters(sim, Survey(simlib_idskip=(2, 4)), _ranges(sim))
    assert [f.libid for f in sim.fields] == [1, 3]


@pytest.mark.parametrize("attr,values,bounds,kept", [
    ("ra",  [10, 45, 70],   (30.0, 60.0),   [45.0]),
    ("dec", [-30, 0, 30],   (-10.0, 10.0),  [0.0]),
])
def test_genrange_narrow_filter(attr, values, bounds, kept):
    fields = [_make_field(libid=i, **{attr: v}) for i, v in enumerate(values)]
    sim = SimlibData(simlib_file='', fields=fields)
    cfg = Survey(**{f"genrange_{'decl' if attr == 'dec' else attr}": bounds})
    apply_simlib_filters(sim, cfg, _ranges(sim))
    assert [getattr(f, attr) for f in sim.fields] == kept


def test_genrange_ra_is_a_plain_window():
    """GENRANGE_RA compares the entry's RA as written, with no wrap, so the
    default -360 to 360 still rejects an entry that gives DEC but no RA
    (RA 9999, snlc_sim.c:20133-20134)."""
    fields = [_make_field(libid=0, ra=355.0), _make_field(libid=1, ra=9999.0),
              _make_field(libid=2, ra=-5.0)]
    sim = SimlibData(simlib_file='', fields=fields)
    apply_simlib_filters(sim, Survey(), _ranges(sim))
    assert [f.libid for f in sim.fields] == [0, 2]
    sim = SimlibData(simlib_file='', fields=fields)
    apply_simlib_filters(sim, Survey(genrange_ra=(-10.0, 10.0)), _ranges(sim))
    assert [f.libid for f in sim.fields] == [2]


def test_nrepeat_groups_consecutive_sne():
    indices = np.arange(9)
    out = np.asarray(simlib_field_index(root_key(0), indices, 100, nrepeat=3,
                                        maxranstart=0))
    assert list(out[:3]) == [0, 0, 0]
    assert list(out[3:6]) == [1, 1, 1]
    assert list(out[6:9]) == [2, 2, 2]


def test_maxranstart_deterministic_and_varies_with_seed():
    indices = np.arange(20)
    a1 = np.asarray(simlib_field_index(root_key(1), indices, 100, 1, maxranstart=50))
    a2 = np.asarray(simlib_field_index(root_key(1), indices, 100, 1, maxranstart=50))
    b = np.asarray(simlib_field_index(root_key(99), indices, 100, 1, maxranstart=50))
    assert list(a1) == list(a2)
    assert list(a1) != list(b)


def test_filter_band_takes_the_last_character_as_intfilter_does():
    """Real $SNDATA_ROOT kcor names; splitting on dashes gives 'B/B' and
    '4Shooter2_B', which match no single-character band key."""
    assert filter_band('LSST-r') == 'r'
    assert filter_band('g') == 'g'
    assert filter_band('Bessell-B/B') == 'B'
    assert filter_band('4Shooter2_B') == 'B'
    assert filter_band('Bessell-BX') == 'X'


def test_entry_peakmjd_outside_the_input_range_is_skipped():
    """USE_SIMLIB_PEAKMJD: an entry whose PEAKMJD value is not strictly
    inside GENRANGE_PEAKMJD is skipped, the edges included
    (check_SIMLIB_GENRANGE, snlc_sim.c:22288, :20167-20180); without the
    flag the value is not read and every entry stays."""
    values = [58999.0, 59000.0, 59500.0, 62000.0, 62500.0]
    def entries():
        fields = [_make_field(libid=i) for i in range(len(values))]
        for f, v in zip(fields, values):
            f.header_values = {'PEAKMJD': str(v)}
        return SimlibData(simlib_file='', fields=fields)
    sim = entries()
    apply_simlib_filters(sim, Survey(use_simlib_peakmjd=1), _ranges(sim))
    assert [f.libid for f in sim.fields] == [2]
    sim = entries()
    apply_simlib_filters(sim, Survey(), _ranges(sim))
    assert len(sim.fields) == len(values)


def test_entry_genranges_are_clipped_or_the_entry_skipped():
    """A header GENRANGE_PEAKMJD or GENRANGE_REDSHIFT overlapping the input
    range is clipped onto it; one with no overlap skips the entry, whatever
    the USE_SIMLIB flags. A REDSHIFT value under USE_SIMLIB_REDSHIFT and a
    DISTANCE under USE_SIMLIB_DISTANCE are checked as single values."""
    fields = [_make_field(libid=i) for i in range(5)]
    fields[0].gengauss = {'PEAKMJD': (58000.0, 59500.0, 5.0)}
    fields[1].gengauss = {'PEAKMJD': (62000.0, 63000.0, 5.0)}
    fields[2].genrange_redshift = (0.5, 2.0)
    fields[3].genrange_redshift = (1.0, 2.0)
    fields[2].redshift_keys = fields[3].redshift_keys = ('GENRANGE_REDSHIFT',)
    fields[4].header_values = {'REDSHIFT': '0.3'}
    fields[4].redshift_keys = ('REDSHIFT', 'DISTANCE')
    sim = SimlibData(simlib_file='', fields=fields)
    apply_simlib_filters(sim, Survey(use_simlib_redshift=1),
                          _ranges(sim, distance_z=[np.nan, np.nan, np.nan, np.nan, 1.5]))
    assert [f.libid for f in sim.fields] == [0, 2]
    assert sim.fields[0].gengauss['PEAKMJD'] == (59000.0, 59500.0, 5.0)
    assert sim.fields[1].genrange_redshift == (0.5, 1.0)


def test_entry_salt2_values_are_checked_under_use_simlib_salt2():
    """USE_SIMLIB_SALT2: an entry's SALT2c value or GENRANGE_SALT2c is checked
    against the input's range for the parameter, as PEAKMJD is
    (snlc_sim.c:20184-20205); without the flag the entry is not read."""
    def entries():
        fields = [_make_field(libid=i) for i in range(3)]
        fields[0].header_values = {'SALT2c': '0.1'}
        fields[1].header_values = {'SALT2c': '0.8'}
        fields[2].gengauss = {'SALT2c': (0.2, 0.9, 0.1)}
        return SimlibData(simlib_file='', fields=fields)
    population = {'SALT2c': (-0.3, 0.5), 'SALT2x1': (-3.0, 3.0)}
    sim = entries()
    apply_simlib_filters(sim, Survey(use_simlib_salt2=1),
                          _ranges(sim, population=population))
    assert [f.libid for f in sim.fields] == [0, 2]
    assert sim.fields[1].gengauss['SALT2c'] == (0.2, 0.5, 0.1)
    sim = entries()
    apply_simlib_filters(sim, Survey(), _ranges(sim, population=population))
    assert len(sim.fields) == 3
