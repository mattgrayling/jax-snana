"""Property-based tests for ``compute_dtseason_peak``.

SNANA's convention (store_SIMLIB_SEASONS, snlc_sim.c:21542-21559): for
each season the candidates are ``PEAKMJD - MJD_MIN`` and
``MJD_MAX - PEAKMJD``; the candidate with the smallest magnitude wins, so
DTSEASON_PEAK is positive when the peak is in season and negative when it
is out. Hypothesis generates random per-field season lists (padded with
inf/-inf) and SN positions; for each draw we assert:

1. The output value is exactly one of the per-season candidates for the
   SN's field.
2. ``|output|`` is the minimum of those candidate magnitudes.
3. Padded slots beyond the field's real seasons are never selected.
"""

import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings, strategies as st

from jax_snlc_sim.survey.searcheff_map import compute_dtseason_peak


def _pad_seasons(fields, max_s):
    n_fields = len(fields)
    starts = np.full((n_fields, max_s), np.inf, dtype=np.float32)
    ends = np.full((n_fields, max_s), -np.inf, dtype=np.float32)
    for i, seasons in enumerate(fields):
        for j, (s, e) in enumerate(seasons):
            starts[i, j] = s
            ends[i, j] = e
    return starts, ends


def _expected_candidates(seasons, peak):
    """SNANA's candidate list, in its loop order (start then end per season)."""
    cands = []
    for s, e in seasons:
        cands.append(peak - s)
        cands.append(e - peak)
    return np.asarray(cands, dtype=np.float32)


_season_list = st.lists(
    st.tuples(
        st.floats(min_value=58000.0, max_value=62000.0, allow_nan=False),
        st.floats(min_value=10.0, max_value=200.0, allow_nan=False),
    ).map(lambda t: (t[0], t[0] + t[1])),
    min_size=1, max_size=6,
).map(lambda seasons: sorted(seasons))


@given(
    fields=st.lists(_season_list, min_size=1, max_size=4),
    sn_specs=st.lists(
        st.tuples(
            st.floats(min_value=57500.0, max_value=62500.0, allow_nan=False),
            st.integers(min_value=0, max_value=3),
        ),
        min_size=1, max_size=20,
    ),
)
@settings(max_examples=50, deadline=None)
def test_dtseason_is_min_abs_season_edge_distance(fields, sn_specs):
    n_fields = len(fields)
    max_s = max(len(f) for f in fields)
    starts, ends = _pad_seasons(fields, max_s)

    sn_specs = [(p, fi % n_fields) for p, fi in sn_specs]
    peak_mjds = jnp.array([s[0] for s in sn_specs], dtype=jnp.float32)
    field_idx = jnp.array([s[1] for s in sn_specs], dtype=jnp.int32)

    dt = np.asarray(compute_dtseason_peak(
        peak_mjds, field_idx, jnp.asarray(starts), jnp.asarray(ends),
    ))

    for i, (peak, fi) in enumerate(sn_specs):
        cands = _expected_candidates(fields[fi], np.float32(peak))
        # Tolerance is float32-aware: values up to 1e4 carry ~1e-3 absolute
        # round-off after subtraction.
        tol = max(1e-3, 1e-7 * max(abs(peak), 1.0))
        assert np.min(np.abs(cands - dt[i])) < tol, (
            f"SN {i}: dt={dt[i]} not a season-edge candidate {cands.tolist()}"
        )
        assert abs(abs(dt[i]) - np.min(np.abs(cands))) < tol, (
            f"SN {i}: |dt|={abs(dt[i])} is not the minimum candidate "
            f"magnitude {np.min(np.abs(cands))}"
        )


@given(
    n_fields=st.integers(min_value=1, max_value=3),
    max_s=st.integers(min_value=2, max_value=5),
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(max_examples=30, deadline=None)
def test_padded_slots_never_selected(n_fields, max_s, seed):
    """A field's DTSEASON must not depend on how many padding slots trail
    its real seasons. Padding is inf/-inf, whose absolute value already
    loses every argmin, so comparing against the reference alone cannot
    fail; the tightly-sized array is what makes the padded width matter."""
    rng = np.random.default_rng(seed)
    starts = np.full((n_fields, max_s), np.inf, dtype=np.float32)
    ends = np.full((n_fields, max_s), -np.inf, dtype=np.float32)
    real_per_field = []
    for i in range(n_fields):
        n_real = rng.integers(1, max_s)
        s0 = np.sort(rng.uniform(58000, 61500, n_real)).astype(np.float32)
        e0 = (s0 + rng.uniform(10, 100, n_real)).astype(np.float32)
        starts[i, :n_real] = s0
        ends[i, :n_real] = e0
        real_per_field.append(list(zip(s0.tolist(), e0.tolist())))

    peaks = jnp.array([60000.0] * n_fields, dtype=jnp.float32)
    fields = jnp.array(list(range(n_fields)), dtype=jnp.int32)

    dt = np.asarray(compute_dtseason_peak(
        peaks, fields, jnp.asarray(starts), jnp.asarray(ends),
    ))
    for i, seasons in enumerate(real_per_field):
        cands = _expected_candidates(seasons, np.float32(60000.0))
        expected = cands[np.argmin(np.abs(cands))]
        # ~6e-3 absolute round-off for values up to 6e4 in float32.
        assert abs(dt[i] - expected) < 1e-2, (
            f"field {i}: padded slot affected output. seasons={seasons}, "
            f"got dt={dt[i]}, expected {expected}"
        )

        # Same field with no padding at all: any dependence on the padded
        # width, through the interleave, the reshape or the argmin, shows up
        # as a disagreement here.
        n_real = len(seasons)
        tight = np.array([[s for s, _ in seasons]], dtype=np.float32)
        tight_ends = np.array([[e for _, e in seasons]], dtype=np.float32)
        dt_tight = np.asarray(compute_dtseason_peak(
            jnp.array([60000.0], dtype=jnp.float32),
            jnp.array([0], dtype=jnp.int32),
            jnp.asarray(tight), jnp.asarray(tight_ends),
        ))[0]
        assert abs(dt[i] - dt_tight) < 1e-2, (
            f"field {i}: {max_s - n_real} padding slots changed the result, "
            f"padded {dt[i]} vs tight {dt_tight}"
        )


def test_no_surviving_season_keeps_snana_init():
    """A field whose seasons were all pruned keeps SNANA's DT_MIN init
    999999.9 (snlc_sim.c:21548)."""
    starts = jnp.array([[np.inf, np.inf]], dtype=jnp.float32)
    ends = jnp.array([[-np.inf, -np.inf]], dtype=jnp.float32)
    dt = np.asarray(compute_dtseason_peak(
        jnp.array([60000.0], dtype=jnp.float32),
        jnp.array([0], dtype=jnp.int32), starts, ends))
    assert np.allclose(dt, 999999.9)
