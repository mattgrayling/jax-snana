"""SIMLIB_MSKOPT bit behavior tests: template noise + scheduler."""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_snlc_core.utils.error_handling import SNANAError
from jax_snlc_sim.builder import (
    _apply_ignore_template_noise, _validate_simlib_observations,
)
from jax_snlc_sim.survey.simlib_reader import (
    SimlibField, _compute_seasons, _rebuild_flat_structures,
    build_jax_field_cache, make_jax_scheduler, scheduler_tables,
)


def _make_field(libid=0, mjds=None):
    if mjds is None:
        mjds = np.linspace(60000.0, 60050.0, 10).astype(np.float32)
    n = len(mjds)
    z = np.zeros(n, dtype=np.float32)
    starts, ends = _compute_seasons(mjds)
    return SimlibField(
        libid=libid, ra=0.0, dec=0.0, mwebv=0.0, pixsize=0.2,
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
        self.fields = fields
        _rebuild_flat_structures(self)


# Scheduler — bit 256 (ENTIRE_SURVEY)


def test_bit_256_keeps_all_epochs():
    mjds = np.linspace(60000.0, 60050.0, 10).astype(np.float32)
    sim = _MockSimlib([_make_field(mjds=mjds)])
    fc = build_jax_field_cache(sim, filter_translation=np.array([0]))
    sched = make_jax_scheduler(fc, max_obs=10, mskopt=256, nskipmjd=(),
                               tobs_pad=0.1)
    # Narrow Trest window covering only the first epochs: the default branch
    # would mask the rest, bit 256 keeps the whole survey. The window has to
    # overlap the survey — SNANA's -9 return (snlc_sim.c:21411) applies whatever
    # the mskopt, so an event that misses the survey entirely gets nothing.
    peak = jnp.array([60005.0])
    out = sched(scheduler_tables(fc),
                jnp.array([0]), jnp.array([60000.0]), jnp.array([60010.0]),
                peak, jnp.array([0.0]))
    n_valid = int(out[5][0])
    assert n_valid == 10


def test_bit_128_masks_to_peak_season():
    mjds = np.concatenate([
        np.linspace(60000.0, 60050.0, 5),
        np.linspace(60500.0, 60550.0, 5),
    ]).astype(np.float32)
    sim = _MockSimlib([_make_field(mjds=mjds)])
    fc = build_jax_field_cache(sim, filter_translation=np.array([0]))
    sched = make_jax_scheduler(fc, max_obs=10, mskopt=128, nskipmjd=(),
                               tobs_pad=0.1)
    peak = jnp.array([60025.0])  # in first season
    out = sched(scheduler_tables(fc),
                jnp.array([0]), jnp.array([60020.0]), jnp.array([60030.0]),
                peak, jnp.array([0.0]))
    n_valid = int(out[5][0])
    assert n_valid == 5
    out_mjds = np.asarray(out[0][0])[:n_valid]
    assert out_mjds.max() < 60100.0


def test_bit_128_no_matching_season_returns_empty():
    mjds = np.concatenate([
        np.linspace(60000.0, 60050.0, 5),
        np.linspace(60500.0, 60550.0, 5),
    ]).astype(np.float32)
    sim = _MockSimlib([_make_field(mjds=mjds)])
    fc = build_jax_field_cache(sim, filter_translation=np.array([0]))
    sched = make_jax_scheduler(fc, max_obs=10, mskopt=128, nskipmjd=(),
                               tobs_pad=0.1)
    peak = jnp.array([60200.0])  # between seasons
    out = sched(scheduler_tables(fc),
                jnp.array([0]), jnp.array([0.0]), jnp.array([0.0]),
                peak, jnp.array([0.0]))
    assert int(out[5][0]) == 0


def test_bit_256_takes_precedence_over_128():
    mjds = np.concatenate([
        np.linspace(60000.0, 60050.0, 5),
        np.linspace(60500.0, 60550.0, 5),
    ]).astype(np.float32)
    sim = _MockSimlib([_make_field(mjds=mjds)])
    fc = build_jax_field_cache(sim, filter_translation=np.array([0]))
    sched = make_jax_scheduler(fc, max_obs=10, mskopt=128 | 256, nskipmjd=(),
                               tobs_pad=0.1)
    peak = jnp.array([60200.0])  # in the season gap → empty under bit 128 alone
    out = sched(scheduler_tables(fc),
                jnp.array([0]), jnp.array([60180.0]), jnp.array([60260.0]),
                peak, jnp.array([0.0]))
    assert int(out[5][0]) == 10


# Bit 16 (IGNORE_TEMPLATENOISE): template noise suppressed at the SIMLIB
# boundary, so no template variance reaches any error term
# (SNANA never stores the template keys under the bit, snlc_sim.c:19288-19289).


def test_bit_16_zeroes_template_arrays_at_boundary():
    f = _make_field()
    f.template_skysigs = f.template_skysigs + 5.0
    f.template_readnoises = f.template_readnoises + 2.0
    sim = _MockSimlib([f])
    assert np.any(sim.flat_obs_data['template_skysigs'] > 0)
    _apply_ignore_template_noise(sim)
    _rebuild_flat_structures(sim)
    assert not np.any(sim.flat_obs_data['template_skysigs'])
    assert not np.any(sim.flat_obs_data['template_readnoises'])
    assert not np.any(sim.flat_obs_data['template_zpts'])


# Construction-time SIMLIB observation validation
# (snlc_sim.c:20573-20580 reader ranges; :27753-27764 and :27808-27813
# fatal per-epoch bounds).


def _valid_mock():
    f = _make_field()
    f.skysigs = f.skysigs + 50.0
    return _MockSimlib([f])


def test_validate_simlib_observations_accepts_valid_entry():
    _validate_simlib_observations(_valid_mock())


def test_validate_simlib_observations_rejects_bad_values():
    cases = [
        ('zptavgs', 8.0, 'ZPTAVG'),          # < 10 (snlc_sim.c:27757)
        ('zptavgs', np.nan, 'ZPTAVG'),
        ('skysigs', 0.0, 'SKYSIG'),          # < 1e-4 (snlc_sim.c:27759)
        ('psf1s', 0.0, 'PSFSIG1'),           # < 1e-4 (snlc_sim.c:27758)
        ('psf_ratios', 1.5, 'PSFRATIO'),     # > 1 (snlc_sim.c:20578)
        ('gains', 0.0, 'GAIN'),
        ('nexposes', 0.0, 'NEXPOSE'),
        ('readnoises', -1.0, 'READNOISE'),
    ]
    for attr, value, name in cases:
        sim = _valid_mock()
        arr = getattr(sim.fields[0], attr).copy()
        arr[3] = value
        setattr(sim.fields[0], attr, arr)
        _rebuild_flat_structures(sim)
        with pytest.raises(SNANAError, match=name):
            _validate_simlib_observations(sim)


def test_validate_simlib_observations_rejects_bad_template_zpt():
    sim = _valid_mock()
    f = sim.fields[0]
    f.template_skysigs = f.template_skysigs + 5.0
    f.template_zpts = np.zeros_like(f.template_zpts) + 5.0  # < 10
    _rebuild_flat_structures(sim)
    with pytest.raises(SNANAError, match='TEMPLATE_ZPT'):
        _validate_simlib_observations(sim)
