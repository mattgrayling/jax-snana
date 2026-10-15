"""Unit tests for the pure JAX host-selection helpers (no fixture I/O)."""

import jax.numpy as jnp
import numpy as np

from jax_snlc_sim.hostlib.hostlib_selection import (
    _build_marginal_cdf,
    _searchsorted_rows,
    empty_host_windows,
    select_hosts,
)


def _draw_window(weights, n_draws=20000):
    """Host shares for ``n_draws`` evenly spaced uniforms at z = 0.25 with a
    constant DZTOL of 0.06 over galaxies at z = 0.1, 0.2, 0.3 and 0.4."""
    galaxy_z = jnp.array([0.1, 0.2, 0.3, 0.4], dtype=jnp.float32)
    cumsum = jnp.cumsum(jnp.asarray(weights, dtype=jnp.float32))[None, :]
    uniforms = (jnp.arange(n_draws, dtype=jnp.float32) + 0.5) / n_draws
    chosen = select_hosts(jnp.full(n_draws, 0.25, dtype=jnp.float32),
                          uniforms, jnp.zeros(n_draws, dtype=jnp.int32),
                          galaxy_z, cumsum, jnp.array([0.06]))
    return np.bincount(np.asarray(chosen), minlength=4) / n_draws


def test_first_galaxy_of_the_window_is_drawn_at_its_weight():
    """The two galaxies inside DZTOL, weights 1 and 3, take a quarter and three
    quarters of the draws; the first is not passed over."""
    share = _draw_window([5.0, 1.0, 3.0, 7.0])
    np.testing.assert_allclose(share, [0.0, 0.25, 0.75, 0.0], atol=2e-4)


def test_top_of_the_window_is_drawn():
    """The last galaxy in the window holds 5% of its weight and takes 5% of
    the draws."""
    share = _draw_window([1.0, 19.0, 1.0, 1.0])
    np.testing.assert_allclose(share, [0.0, 0.95, 0.05, 0.0], atol=2e-4)


def test_a_window_of_zero_weight_gives_its_first_galaxy():
    """As in SNANA, a window whose galaxies all have zero weight gives its
    lowest-redshift galaxy."""
    share = _draw_window([1.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(share, [0.0, 1.0, 0.0, 0.0])


def test_the_bit_16_cdf_gives_a_zero_weight_window_to_its_first_galaxy():
    """Under SN2GAL_Z the zHEL cell at 0.25, whose window holds only
    zero-weight galaxies, gives its probability to the galaxy at 0.2, as the
    per-event draw does; the cell at 0.38 draws by weight."""
    galaxy_z = np.array([0.1, 0.2, 0.3, 0.4])
    cdf = _build_marginal_cdf(galaxy_z, (0.06,), np.ones(4),
                              np.array([0.25, 0.38]), np.array([0.5, 0.5]),
                              (0.0, 1.0), np.cumsum([[1.0, 0.0, 0.0, 1.0]],
                                                    axis=1))
    np.testing.assert_allclose(np.asarray(cdf), [[0.0, 0.5, 0.5, 1.0]])


def test_empty_windows_are_flagged():
    """A window with no galaxy is flagged, and on a GROUPID row (row 1 here)
    so is one whose galaxies all have zero weight."""
    galaxy_z = jnp.array([0.1, 0.2, 0.3, 0.4], dtype=jnp.float32)
    cumsum = jnp.cumsum(jnp.array([1.0, 0.0, 1.0, 1.0]))[None, :]
    sn_z = jnp.array([0.15, 0.2, 0.3, 0.35, 0.5], dtype=jnp.float32)
    for row, expected in [(0, [True, False, False, True, True]),
                          (1, [True, True, False, True, True])]:
        empty = empty_host_windows(sn_z, jnp.full(5, row, dtype=jnp.int32),
                                   galaxy_z, jnp.concatenate([cumsum] * 2),
                                   jnp.array([0.04]), 1)
        np.testing.assert_array_equal(np.asarray(empty), expected)


def test_select_hosts_distributes_across_bins():
    """Different SNe in different bins read different rows of the weights table."""
    n_gal = 30
    galaxy_z = jnp.linspace(0.0, 1.0, n_gal, dtype=jnp.float32)
    # bin 0 favours low-index galaxies, bin 2 favours high-index, bin 1 uniform.
    weights = jnp.stack([
        jnp.exp(-jnp.arange(n_gal, dtype=jnp.float32) / 5.0),
        jnp.ones(n_gal, dtype=jnp.float32),
        jnp.exp(-(n_gal - 1 - jnp.arange(n_gal, dtype=jnp.float32)) / 5.0),
    ], axis=0)
    # Pre-cumsum each bin (the function expects cumulative weights per row).
    cumsum = jnp.cumsum(weights, axis=1)

    sn_z = jnp.full(60, 0.5, dtype=jnp.float32)
    rng = np.random.default_rng(7)
    randoms = jnp.asarray(rng.uniform(size=60), dtype=jnp.float32)
    bin_idx = jnp.array([0] * 20 + [1] * 20 + [2] * 20, dtype=jnp.int32)
    dztol = jnp.array([0.0, 1.0])  # huge tol so the whole library is candidate

    selections = np.asarray(select_hosts(
        sn_z, randoms, bin_idx, galaxy_z, cumsum, dztol,
    ))
    # Bin 0 should choose lower-index galaxies on average than bin 2.
    mean_bin0 = float(np.mean(selections[:20]))
    mean_bin2 = float(np.mean(selections[40:]))
    assert mean_bin0 < mean_bin2 - 5.0


def test_select_hosts_mixed_bins_match_per_bin_batches():
    """A mixed-bin batch selects the same galaxies as per-bin sub-batches.

    Guards the row-indexed binary search: the kernel outcome for each event
    must not depend on which other bins share the batch.
    """
    n_gal = 40
    galaxy_z = jnp.linspace(0.0, 1.0, n_gal, dtype=jnp.float32)
    rng = np.random.default_rng(11)
    weights = jnp.asarray(rng.uniform(0.1, 2.0, size=(3, n_gal)),
                          dtype=jnp.float32)
    cumsum = jnp.cumsum(weights, axis=1)
    dztol = jnp.array([0.002, 0.04])

    n_sne = 90
    sn_z = jnp.asarray(rng.uniform(0.05, 0.95, size=n_sne), dtype=jnp.float32)
    randoms = jnp.asarray(rng.uniform(size=n_sne), dtype=jnp.float32)
    bin_idx = jnp.asarray(rng.integers(0, 3, size=n_sne), dtype=jnp.int32)

    mixed = np.asarray(select_hosts(
        sn_z, randoms, bin_idx, galaxy_z, cumsum, dztol))

    per_bin = np.empty(n_sne, dtype=np.int32)
    bin_idx_np = np.asarray(bin_idx)
    for ibin in range(3):
        sel = np.where(bin_idx_np == ibin)[0]
        per_bin[sel] = np.asarray(select_hosts(
            sn_z[sel], randoms[sel],
            jnp.full(len(sel), ibin, dtype=jnp.int32),
            galaxy_z, cumsum, dztol))

    np.testing.assert_array_equal(mixed, per_bin)


def test_searchsorted_rows_matches_jnp_searchsorted():
    """Row-indexed bisection agrees with jnp.searchsorted on every row,
    both sides, including exact ties and out-of-range targets."""
    rng = np.random.default_rng(3)
    table_np = np.sort(rng.uniform(0.0, 10.0, size=(4, 33)), axis=1)
    table_np[2, 10:14] = table_np[2, 10]  # plateau of equal values
    table = jnp.asarray(table_np, dtype=jnp.float32)

    values_np = np.concatenate([
        rng.uniform(-1.0, 11.0, size=40),
        table_np[2, 10:11],               # exact tie inside the plateau
        [table_np[0, 0], table_np[3, -1]]  # exact first/last elements
    ]).astype(np.float32)
    values = jnp.asarray(values_np)
    rows = jnp.asarray(rng.integers(0, 4, size=len(values_np)),
                       dtype=jnp.int32)

    for side in ('left', 'right'):
        got = np.asarray(_searchsorted_rows(table, rows, values, side))
        expected = np.array([
            int(jnp.searchsorted(table[int(r)], v, side=side))
            for r, v in zip(np.asarray(rows), values_np)])
        np.testing.assert_array_equal(got, expected)
