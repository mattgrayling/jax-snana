"""Unit tests for ``_compute_ddlr_and_sort`` — the JIT-compiled DDLR
computation that backs HOSTGAL/HOSTGAL2 selection over NBR_LIST.

Synthesises ``ddlr_columns`` + ``nbr_array`` directly rather than reading
a HOSTLIB; lets us assert the geometry of the SN-host association
math (Haversine separation, Sersic-derived DLR, NBR_LIST ordering).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_snlc_sim.hostlib.catalogue import neighbour_offsets
from jax_snlc_sim.hostlib.selection import _compute_ddlr_and_sort


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_columns(galaxies, dlr_ab=None):
    """galaxies: list of (ra, dec, a0, b0, w0, a1, b1, w1, a_rot) tuples.

    Returns the packed per-galaxy arrays the DDLR kernel takes: radec (2, n),
    per-component Sersic a/b/w (2, n), a_rot (n,), and a_DLR/b_DLR (2, n).
    """
    arr = np.array(galaxies, dtype=np.float32).T
    radec = jnp.asarray(arr[0:2])
    sersic_a = jnp.asarray(arr[[2, 5]])
    sersic_b = jnp.asarray(arr[[3, 6]])
    sersic_w = jnp.asarray(arr[[4, 7]])
    a_rot = jnp.asarray(arr[8])
    if dlr_ab is None:
        dlr = jnp.zeros((2, arr.shape[1]), dtype=jnp.float32)
    else:
        dlr = jnp.asarray(np.array(dlr_ab, dtype=np.float32))
    return (radec, sersic_a, sersic_b, sersic_w, a_rot, dlr)


def _run(cols, nbr_array, galaxy_indices, ra_sn, dec_sn, **kw):
    """Expand the packed column tuple into the kernel's positional inputs.

    The kernel works in offsets, so the neighbour offsets come from the same
    function the builder uses at load, and the SN's offset from its own host
    comes from the positions the test asked for.
    """
    radec, sersic_a, sersic_b, sersic_w, a_rot, dlr_ab = cols
    ra = np.asarray(radec[0], dtype=np.float64)
    dec = np.asarray(radec[1], dtype=np.float64)
    nbr_offsets = jnp.asarray(neighbour_offsets(ra, dec, nbr_array))
    gi = np.asarray(galaxy_indices)
    sn_dra = jnp.asarray(np.asarray(ra_sn, dtype=np.float64) - ra[gi])
    sn_ddec = jnp.asarray(np.asarray(dec_sn, dtype=np.float64) - dec[gi])
    return _compute_ddlr_and_sort(
        radec, sersic_a, sersic_b, sersic_w, a_rot, dlr_ab,
        nbr_array, nbr_offsets, galaxy_indices, ra_sn, dec_sn,
        sn_dra, sn_ddec, **kw)


def _empty_detect():
    """Empty (n_bands=0, n_thresh=0) tuple — disables detection filtering."""
    return (
        jnp.zeros((0, 0), dtype=jnp.float32),  # detect_columns
        jnp.zeros((0,), dtype=jnp.float32),    # detect_thresholds
        jnp.zeros((0, 0), dtype=jnp.float32),  # detect_raw_errs
    )


# ---------------------------------------------------------------------------
# Single-galaxy geometry
# ---------------------------------------------------------------------------

def test_sn_at_host_position_yields_zero_separation_and_ddlr():
    """SN coincident with host → snsep=0, ddlr=0."""
    cols = _make_columns([
        (10.0, -30.0, 1.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0),  # 1 Sersic
    ])
    nbr_array = jnp.array([[-1]], dtype=jnp.int32)  # no neighbours
    galaxy_indices = jnp.array([0], dtype=jnp.int32)
    ra_sn = jnp.array([10.0], dtype=jnp.float32)
    dec_sn = jnp.array([-30.0], dtype=jnp.float32)
    snr_err_scale = jnp.array([1.0], dtype=jnp.float32)
    detect_cols, detect_thresh, detect_raw = _empty_detect()

    sorted_ddlr, sorted_snsep, sorted_gal_idx, sorted_valid, nmatch, nmatch2, *_ = \
        _run(
            cols, nbr_array, galaxy_indices, ra_sn, dec_sn,
            maxddlr=4.0, maxddlr2=4.0,
            smear_sersic=0.0, use_dlr_columns=False,
            detect_columns=detect_cols, detect_thresholds=detect_thresh,
            snr_err_scale=snr_err_scale, detect_raw_errs=detect_raw,
        )
    assert sorted_gal_idx[0, 0] == 0       # primary host wins
    assert sorted_valid[0, 0]
    # SNSEP and DDLR are formally zero but float32 in degree-space ×3600″
    # introduces ~mas-scale noise; both values must be << any typical DDLR.
    assert float(sorted_snsep[0, 0]) < 1e-2  # arcsec
    assert float(sorted_ddlr[0, 0]) < 1e-2   # dimensionless
    assert int(nmatch[0]) == 1


def test_sn_offset_within_host_dlr_is_match():
    """SN offset by a small fraction of the half-light radius → ddlr<<1, match."""
    # Galaxy at (10°, -30°) with a_half=1.0 arcsec (0.000278°), b_half=0.5,
    # arot=0 (major axis along RA). Place SN ~0.3″ along major axis.
    a_half_deg = 1.0 / 3600.0
    cols = _make_columns([
        (10.0, -30.0, 1.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0),
    ])
    nbr_array = jnp.array([[-1]], dtype=jnp.int32)
    galaxy_indices = jnp.array([0], dtype=jnp.int32)
    ra_sn = jnp.array([10.0 + 0.3 * a_half_deg], dtype=jnp.float32)
    dec_sn = jnp.array([-30.0], dtype=jnp.float32)
    snr_err_scale = jnp.array([1.0], dtype=jnp.float32)
    detect_cols, detect_thresh, detect_raw = _empty_detect()

    sorted_ddlr, sorted_snsep, _, _, nmatch, *_ = _run(
        cols, nbr_array, galaxy_indices, ra_sn, dec_sn,
        maxddlr=4.0, maxddlr2=4.0,
        smear_sersic=0.0, use_dlr_columns=False,
        detect_columns=detect_cols, detect_thresholds=detect_thresh,
        snr_err_scale=snr_err_scale, detect_raw_errs=detect_raw,
    )
    # Along major axis at 0.3 a_half → SNSEP ≈ 0.3 * 1.0 arcsec, DLR = a_half = 1.0
    # → DDLR ≈ 0.3, well within maxddlr=4
    assert float(sorted_snsep[0, 0]) == pytest.approx(0.3, abs=0.05)
    assert float(sorted_ddlr[0, 0]) == pytest.approx(0.3, abs=0.05)
    assert int(nmatch[0]) == 1


def test_sn_far_from_host_has_no_match():
    """SN beyond maxddlr → nmatch=0."""
    cols = _make_columns([
        (10.0, -30.0, 1.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0),
    ])
    nbr_array = jnp.array([[-1]], dtype=jnp.int32)
    galaxy_indices = jnp.array([0], dtype=jnp.int32)
    a_half_deg = 1.0 / 3600.0
    # 100 a_half away → DDLR=100, way beyond maxddlr=4
    ra_sn = jnp.array([10.0 + 100.0 * a_half_deg], dtype=jnp.float32)
    dec_sn = jnp.array([-30.0], dtype=jnp.float32)
    snr_err_scale = jnp.array([1.0], dtype=jnp.float32)
    detect_cols, detect_thresh, detect_raw = _empty_detect()

    sorted_ddlr, _, _, _, nmatch, nmatch2, _, _ = _run(
        cols, nbr_array, galaxy_indices, ra_sn, dec_sn,
        maxddlr=4.0, maxddlr2=4.0,
        smear_sersic=0.0, use_dlr_columns=False,
        detect_columns=detect_cols, detect_thresholds=detect_thresh,
        snr_err_scale=snr_err_scale, detect_raw_errs=detect_raw,
    )
    assert int(nmatch[0]) == 0
    assert int(nmatch2[0]) == 0
    # Sorted DDLR for the host is far above the cut.
    assert float(sorted_ddlr[0, 0]) > 4.0


# ---------------------------------------------------------------------------
# NBR_LIST: SN closer to neighbour than to host
# ---------------------------------------------------------------------------

def test_nbr_list_neighbour_wins_when_closer():
    """SN closer to NBR_LIST neighbour than to nominal host → sort puts
    neighbour first (HOSTGAL = neighbour, HOSTGAL2 = host)."""
    a_half_deg = 1.0 / 3600.0
    # Galaxy 0: nominal host at (10.0, -30.0)
    # Galaxy 1: neighbour at (10.0 + 1.0 a_half, -30.0)  ← closer in RA
    cols = _make_columns([
        (10.0,                    -30.0, 1.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0),
        (10.0 + 1.0 * a_half_deg, -30.0, 1.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0),
    ])
    # Galaxy 0's NBR_LIST contains galaxy 1
    nbr_array = jnp.array([[1], [-1]], dtype=jnp.int32)
    galaxy_indices = jnp.array([0], dtype=jnp.int32)
    # SN positioned at the neighbour exactly
    ra_sn = jnp.array([10.0 + 1.0 * a_half_deg], dtype=jnp.float32)
    dec_sn = jnp.array([-30.0], dtype=jnp.float32)
    snr_err_scale = jnp.array([1.0], dtype=jnp.float32)
    detect_cols, detect_thresh, detect_raw = _empty_detect()

    sorted_ddlr, _, sorted_gal_idx, sorted_valid, nmatch, nmatch2, *_ = \
        _run(
            cols, nbr_array, galaxy_indices, ra_sn, dec_sn,
            maxddlr=4.0, maxddlr2=4.0,
            smear_sersic=0.0, use_dlr_columns=False,
            detect_columns=detect_cols, detect_thresholds=detect_thresh,
            snr_err_scale=snr_err_scale, detect_raw_errs=detect_raw,
        )
    assert int(sorted_gal_idx[0, 0]) == 1, "neighbour should rank first"
    assert int(sorted_gal_idx[0, 1]) == 0, "host should rank second"
    assert float(sorted_ddlr[0, 0]) < float(sorted_ddlr[0, 1])
    assert int(nmatch[0]) == 2
    assert int(nmatch2[0]) == 2


def test_nbr_padding_with_minus_one_is_invalid_and_skipped():
    """``-1`` entries in NBR_LIST are masked out — sort doesn't pick them."""
    a_half_deg = 1.0 / 3600.0
    cols = _make_columns([
        (10.0,                    -30.0, 1.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0),
        (10.0 + 1.0 * a_half_deg, -30.0, 1.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0),
    ])
    # Two padded slots, only one real neighbour
    nbr_array = jnp.array([[1, -1], [-1, -1]], dtype=jnp.int32)
    galaxy_indices = jnp.array([0], dtype=jnp.int32)
    ra_sn = jnp.array([10.0 + 0.5 * a_half_deg], dtype=jnp.float32)
    dec_sn = jnp.array([-30.0], dtype=jnp.float32)
    snr_err_scale = jnp.array([1.0], dtype=jnp.float32)
    detect_cols, detect_thresh, detect_raw = _empty_detect()

    sorted_ddlr, _, sorted_gal_idx, sorted_valid, nmatch, *_ = \
        _run(
            cols, nbr_array, galaxy_indices, ra_sn, dec_sn,
            maxddlr=4.0, maxddlr2=4.0,
            smear_sersic=0.0, use_dlr_columns=False,
            detect_columns=detect_cols, detect_thresholds=detect_thresh,
            snr_err_scale=snr_err_scale, detect_raw_errs=detect_raw,
        )
    # First two slots are valid (host=0, neighbour=1).
    assert sorted_valid[0, 0] and sorted_valid[0, 1]
    # The -1 padded slot must be marked invalid (sorted last, ddlr=inf).
    assert not sorted_valid[0, 2]
    assert float(sorted_ddlr[0, 2]) == float("inf")
    assert int(nmatch[0]) == 2


# ---------------------------------------------------------------------------
# DLR columns override Sersic average
# ---------------------------------------------------------------------------

def test_use_dlr_columns_overrides_weighted_sersic():
    """When use_dlr_columns=True the function reads a_DLR/b_DLR (cols 9, 10)
    instead of computing the Sersic-weighted half-light radii."""
    # Build full 11-col input. Sersic params would yield a_half=1, b_half=0.5,
    # but a_DLR/b_DLR override to a much larger 5″/2″ → DDLR shrinks 5x.
    base = [
        (10.0, -30.0, 1.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0),
    ]
    cols_full = _make_columns(base, dlr_ab=[[5.0], [2.0]])
    nbr_array = jnp.array([[-1]], dtype=jnp.int32)
    galaxy_indices = jnp.array([0], dtype=jnp.int32)
    a_half_deg = 1.0 / 3600.0
    ra_sn = jnp.array([10.0 + 1.0 * a_half_deg], dtype=jnp.float32)
    dec_sn = jnp.array([-30.0], dtype=jnp.float32)
    snr_err_scale = jnp.array([1.0], dtype=jnp.float32)
    detect_cols, detect_thresh, detect_raw = _empty_detect()

    sorted_ddlr_a, *_ = _run(
        cols_full, nbr_array, galaxy_indices, ra_sn, dec_sn,
        maxddlr=4.0, maxddlr2=4.0, smear_sersic=0.0,
        use_dlr_columns=False,
        detect_columns=detect_cols, detect_thresholds=detect_thresh,
        snr_err_scale=snr_err_scale, detect_raw_errs=detect_raw,
    )
    sorted_ddlr_b, *_ = _run(
        cols_full, nbr_array, galaxy_indices, ra_sn, dec_sn,
        maxddlr=4.0, maxddlr2=4.0, smear_sersic=0.0,
        use_dlr_columns=True,
        detect_columns=detect_cols, detect_thresholds=detect_thresh,
        snr_err_scale=snr_err_scale, detect_raw_errs=detect_raw,
    )
    # 5× larger a_DLR → DDLR shrinks ~5× (along major axis).
    ratio = float(sorted_ddlr_a[0, 0]) / float(sorted_ddlr_b[0, 0])
    assert ratio > 4.0, f"a_DLR override should shrink DDLR ~5x, got ratio {ratio:.2f}"


# ---------------------------------------------------------------------------
# SMEAR_SERSIC inflates a_half/b_half
# ---------------------------------------------------------------------------

def test_smear_sersic_inflates_half_light_radii():
    """A non-zero SMEAR_SERSIC FWHM enlarges effective half-light, shrinking DDLR."""
    cols = _make_columns([
        (10.0, -30.0, 1.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0),
    ])
    nbr_array = jnp.array([[-1]], dtype=jnp.int32)
    galaxy_indices = jnp.array([0], dtype=jnp.int32)
    a_half_deg = 1.0 / 3600.0
    ra_sn = jnp.array([10.0 + 1.0 * a_half_deg], dtype=jnp.float32)
    dec_sn = jnp.array([-30.0], dtype=jnp.float32)
    snr_err_scale = jnp.array([1.0], dtype=jnp.float32)
    detect_cols, detect_thresh, detect_raw = _empty_detect()

    sorted_ddlr_no_smear, *_ = _run(
        cols, nbr_array, galaxy_indices, ra_sn, dec_sn,
        maxddlr=4.0, maxddlr2=4.0, smear_sersic=0.0, use_dlr_columns=False,
        detect_columns=detect_cols, detect_thresholds=detect_thresh,
        snr_err_scale=snr_err_scale, detect_raw_errs=detect_raw,
    )
    sorted_ddlr_smear, *_ = _run(
        cols, nbr_array, galaxy_indices, ra_sn, dec_sn,
        maxddlr=4.0, maxddlr2=4.0, smear_sersic=2.0, use_dlr_columns=False,
        detect_columns=detect_cols, detect_thresholds=detect_thresh,
        snr_err_scale=snr_err_scale, detect_raw_errs=detect_raw,
    )
    # SMEAR_SERSIC=2.0 (FWHM, → ab_smear=1.0) ≥ original a_half=1.0 → noticeable shrink in DDLR.
    assert float(sorted_ddlr_smear[0, 0]) < float(sorted_ddlr_no_smear[0, 0])


# ---------------------------------------------------------------------------
# Detection filtering masks neighbours below threshold
# ---------------------------------------------------------------------------

def test_detection_threshold_excludes_undetected_neighbour():
    """A neighbour whose detect-columns value falls below the threshold is
    masked out and ranks behind valid candidates."""
    a_half_deg = 1.0 / 3600.0
    cols = _make_columns([
        (10.0,                    -30.0, 1.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0),
        (10.0 + 1.0 * a_half_deg, -30.0, 1.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0),
    ])
    nbr_array = jnp.array([[1], [-1]], dtype=jnp.int32)
    galaxy_indices = jnp.array([0], dtype=jnp.int32)
    ra_sn = jnp.array([10.0 + 1.0 * a_half_deg], dtype=jnp.float32)  # at neighbour
    dec_sn = jnp.array([-30.0], dtype=jnp.float32)
    snr_err_scale = jnp.array([1.0], dtype=jnp.float32)
    # 1 detect band, host has SNR=10, neighbour has SNR=1; threshold=5 keeps host only.
    detect_columns = jnp.array([[10.0, 1.0]], dtype=jnp.float32)  # (n_bands=1, n_galaxies)
    detect_thresholds = jnp.array([5.0], dtype=jnp.float32)
    detect_raw_errs = jnp.zeros((0, 0), dtype=jnp.float32)

    _, _, sorted_gal_idx, sorted_valid, nmatch, *_ = _run(
        cols, nbr_array, galaxy_indices, ra_sn, dec_sn,
        maxddlr=4.0, maxddlr2=4.0, smear_sersic=0.0, use_dlr_columns=False,
        detect_columns=detect_columns, detect_thresholds=detect_thresholds,
        snr_err_scale=snr_err_scale, detect_raw_errs=detect_raw_errs,
    )
    # Host (idx 0) is the only valid candidate; neighbour filtered out.
    assert int(sorted_gal_idx[0, 0]) == 0
    assert sorted_valid[0, 0]
    assert not sorted_valid[0, 1]
    assert int(nmatch[0]) == 1


def test_nmatch_counts_are_uncapped():
    """HOSTGAL_NMATCH/NMATCH2 carry the full neighbour counts; SNANA writes
    them uncapped since the Feb 9 2026 fix (snlc_sim.c:26045-26046)."""
    a_half_deg = 1.0 / 3600.0
    gals = [(10.0 + k * 0.2 * a_half_deg, -30.0,
             1.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0) for k in range(4)]
    cols = _make_columns(gals)
    nbr_array = jnp.array([[1, 2, 3], [-1, -1, -1],
                           [-1, -1, -1], [-1, -1, -1]], dtype=jnp.int32)
    galaxy_indices = jnp.array([0], dtype=jnp.int32)
    ra_sn = jnp.array([10.0], dtype=jnp.float32)
    dec_sn = jnp.array([-30.0], dtype=jnp.float32)
    snr_err_scale = jnp.array([1.0], dtype=jnp.float32)
    detect_cols, detect_thresh, detect_raw = _empty_detect()

    *_, nmatch, nmatch2, _, _ = _run(
        cols, nbr_array, galaxy_indices, ra_sn, dec_sn,
        maxddlr=4.0, maxddlr2=7.0, smear_sersic=0.0, use_dlr_columns=False,
        detect_columns=detect_cols, detect_thresholds=detect_thresh,
        snr_err_scale=snr_err_scale, detect_raw_errs=detect_raw,
    )
    assert int(nmatch[0]) == 4
    assert int(nmatch2[0]) == 4


def test_true_host_dlr_smear_applies_on_top_of_dlr_columns():
    """The true-host path follows SNANA's order: weighted Sersic average,
    THEN the a_DLR/b_DLR override, THEN SMEAR_SERSIC on top
    (GEN_SNHOST_DDLR, sntools_host.c:8559-8573). With a_DLR=5, b_DLR=2 and
    SMEAR_SERSIC=2 (ab_smear=1), the DLR along the major axis is
    sqrt(5^2 + 1^2)."""
    from types import SimpleNamespace
    from jax_snlc_sim.hostlib.selection import HostSelector
    from jax_snlc_sim.hostlib.sersic import compute_sersic_integral_table

    ngal = 1
    catalogue = SimpleNamespace(sersic={
            'a': jnp.full((1, ngal), 1.0),
            'b': jnp.full((1, ngal), 0.5),
            'n': jnp.full((1, ngal), 1.0),
            'w': jnp.full((1, ngal), 1.0),
            'a_rot': jnp.zeros(ngal),
        })
    sel = SimpleNamespace(
        use_dlr_columns=True,
        smear_sersic=2.0,
        fixran_wgt=-9.0,
        fixran_radius=-9.0,
        fixran_phi=0.0,          # place the SN along the major axis
        mnintflux_snpos=0.0,
        mxintflux_snpos=0.99,
        sersic_table=compute_sersic_integral_table(),
    )
    host_properties = {
        'DEC_GAL': jnp.zeros(1),
        'a_DLR': jnp.full(1, 5.0),
        'b_DLR': jnp.full(1, 2.0),
    }
    out = HostSelector._generate_sn_positions(
        sel, catalogue, host_properties, jnp.zeros(1, dtype=jnp.int32),
        jnp.array([[0.5, 0.5, 0.5, 0.5]]))
    expected = float(np.sqrt(5.0**2 + 1.0**2))
    assert float(out['DLR'][0]) == pytest.approx(expected, rel=1e-3)


def test_position_sampling_supports_three_sersic_components():
    """Component choice walks the cumulative weights over all packed
    components (GEN_SNHOST_POS, sntools_host.c:7662-7671); a third or later
    component is reachable, up to MXSERSIC_HOSTLIB=9."""
    from types import SimpleNamespace
    from jax_snlc_sim.hostlib.selection import HostSelector
    from jax_snlc_sim.hostlib.sersic import compute_sersic_integral_table

    ngal = 1
    a_comp = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    catalogue = SimpleNamespace(sersic={
            'a': jnp.asarray(a_comp[:, None]),
            'b': jnp.asarray((0.5 * a_comp)[:, None]),
            'n': jnp.full((3, ngal), 1.0),
            'w': jnp.asarray(np.array([0.3, 0.3, 0.4],
                                      dtype=np.float32)[:, None]),
            'a_rot': jnp.zeros(ngal),
        })
    sel = SimpleNamespace(
        use_dlr_columns=False,
        smear_sersic=0.0,
        fixran_wgt=-9.0,
        fixran_radius=0.5,
        fixran_phi=0.0,          # along the major axis
        mnintflux_snpos=0.0,
        mxintflux_snpos=0.99,
        sersic_table=compute_sersic_integral_table(),
    )
    host_properties = {'DEC_GAL': jnp.zeros(1)}
    offsets = []
    for u_comp in (0.1, 0.5, 0.9):   # cum weights 0.3, 0.6, 1.0
        out = HostSelector._generate_sn_positions(
            sel, catalogue, host_properties, jnp.zeros(1, dtype=jnp.int32),
            jnp.array([[u_comp, 0.5, 0.5, 0.5]]))
        offsets.append(float(out['RA_OFFSET'][0]))
    # Same Sersic index and radius draw for every component, so the offset
    # scales with the chosen component's a_half: 1 : 2 : 4
    np.testing.assert_allclose(
        [offsets[1] / offsets[0], offsets[2] / offsets[0]],
        [2.0, 4.0], rtol=1e-4)

    # HOSTLIB_FIXRAN_WGT replaces the component draw (sntools_host.c:6005),
    # so the same draw of 0.1 now lands in the third component
    sel.fixran_wgt = 0.9
    out = HostSelector._generate_sn_positions(
        sel, catalogue, host_properties, jnp.zeros(1, dtype=jnp.int32),
        jnp.array([[0.1, 0.5, 0.5, 0.5]]))
    assert float(out['RA_OFFSET'][0]) == pytest.approx(offsets[2], rel=1e-6)
