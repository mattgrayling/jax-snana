"""JOBID / NJOBTOT: give each batch job its own slice of the SIMLIB.

submit_batch_jobs appends `JOBID k NJOBTOT n` to each split job's sim input
(submit_prog_sim.py:1676). Without it every job reads the same LIBIDs, so the
jobs share cadences and their union is not the sample one big run would give.

Exact SNANA parity is not available: SNANA draws a random offset inside the
slice, and jax slices the post-pruning pool where SNANA uses the raw NLIBID
header. What is pinned here is the partition property, which is what makes the
split sound.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from jax_snlc_core.utils.error_handling import SNANAError
from jax_snlc_sim.config_parser import parse_snana_input_file
from jax_snlc_sim.random_streams import root_key
from jax_snlc_sim.sampler import simlib_job_start, simlib_field_index


_A_FIXTURE_INPUT = str(Path(__file__).parent.parent / "fixtures" / "snana"
                       / "atmosphere_replay" / "input.input")
_POOL = 1000


@pytest.fixture(scope='module')
def cfg():
    return parse_snana_input_file(_A_FIXTURE_INPUT)


def _batched(cfg, jobid, njobtot, gentype=-9):
    return replace(
        cfg,
        run=replace(cfg.run, jobid=jobid, njobtot=njobtot, ngentot_lc=0,
                    random_seed=1),
        rates=replace(cfg.rates, gentype=gentype))


def test_unbatched_run_has_no_offset(cfg):
    assert simlib_job_start(_batched(cfg, 0, 0), _POOL) == 0


def test_each_job_starts_one_slice_further_in(cfg):
    slice_len = _POOL // 4
    starts = [simlib_job_start(_batched(cfg, j, 4), _POOL) for j in (1, 2, 3, 4)]
    assert starts == [0, slice_len, 2 * slice_len, 3 * slice_len]


def test_gentype_staggers_which_slice_a_job_reads(cfg):
    """GENTYPE rotates the job->slice mapping so different models do not all
    hammer the same part of the SIMLIB (snlc_sim.c:19108)."""
    slice_len = _POOL // 4
    starts = [simlib_job_start(_batched(cfg, j, 4, gentype=1), _POOL)
              for j in (1, 2, 3, 4)]
    # a permutation of the same slices, not the identity
    assert sorted(starts) == [0, slice_len, 2 * slice_len, 3 * slice_len]
    assert starts != [0, slice_len, 2 * slice_len, 3 * slice_len]


def test_small_pool_is_not_split(cfg):
    """SNANA only splits a pool of more than 100 entries (snlc_sim.c:19091)."""
    assert simlib_job_start(_batched(cfg, 2, 4), 100) == 0
    assert simlib_job_start(_batched(cfg, 2, 4), 101) > 0


def test_idstart_wins_over_the_batch_offset(cfg):
    """SIMLIB_IDSTART sets its own start, so the batch offset stands down."""
    c = _batched(cfg, 2, 4)
    c = replace(c, survey=replace(c.survey, simlib_idstart=5))
    assert simlib_job_start(c, _POOL) == 0


def test_jobs_draw_disjoint_libids_covering_the_pool(cfg):
    """The property that makes splitting sound: N jobs of NGENTOT/N events
    partition the SIMLIB instead of all reading the same entries."""
    njobtot, per_job = 4, _POOL // 4
    seen = []
    for jobid in (1, 2, 3, 4):
        start = simlib_job_start(_batched(cfg, jobid, njobtot), _POOL)
        idx = simlib_field_index(root_key(0), np.arange(per_job), _POOL, 1, 0,
                                 start)
        seen.append(set(np.asarray(idx).tolist()))

    for a in range(njobtot):
        for b in range(a + 1, njobtot):
            assert not seen[a] & seen[b], f"jobs {a+1} and {b+1} share LIBIDs"
    assert set().union(*seen) == set(range(_POOL))


def test_idlock_cannot_combine_with_batch_mode(tmp_path):
    """One pins a single LIBID, the other splits the library (snlc_sim.c:19150)."""
    inp = tmp_path / "in.input"
    inp.write_text(open(_A_FIXTURE_INPUT).read() + "\nSIMLIB_IDLOCK: 3\n")
    with pytest.raises(SNANAError, match="IDLOCK"):
        parse_snana_input_file(str(inp), ['NJOBTOT', '4', 'JOBID', '2'])
