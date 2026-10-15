"""Unit tests for the wrong-host contamination model (HOSTLIB_WRONGHOST_FILE)."""

import numpy as np
import pytest

from jax_snlc_sim.hostlib.wronghost import (
    DZTRUEMAX,
    WrongHostTable,
    gen_zhost,
    read_wronghost_file,
)


@pytest.fixture(scope="module")
def wronghost_path(data_root):
    return data_root / "hostlib" / "JAX_TEST_WRONGHOST.DAT"


def test_read_wronghost_file_filters_to_z_range(wronghost_path):
    """ZTRUE/ZMATCH outside [zmin, zmax] are dropped."""
    full = read_wronghost_file(str(wronghost_path), zmin=0.0, zmax=2.0)
    narrow = read_wronghost_file(str(wronghost_path), zmin=0.3, zmax=0.5)
    assert narrow.ztrue.size <= full.ztrue.size
    assert narrow.ztrue.min() >= 0.3
    assert narrow.ztrue.max() <= 0.5
    assert narrow.zmatch.min() >= 0.3
    assert narrow.zmatch.max() <= 0.5


def test_read_wronghost_file_sorted_by_ztrue(wronghost_path):
    """Output table is sorted in ZTRUE — required for searchsorted."""
    data = read_wronghost_file(str(wronghost_path), zmin=0.0, zmax=2.0)
    assert np.all(np.diff(data.ztrue) >= 0)


def test_gen_zhost_full_contamination_when_random_below_prob():
    """SNe with random_prob < P(wrong) get remapped to a ZMATCH entry."""
    data = WrongHostTable(
        poly_coeffs=np.array([1.0]),  # P(wrong) = 1 always
        ztrue=np.array([0.10, 0.20, 0.30, 0.40]),
        zmatch=np.array([0.50, 0.55, 0.60, 0.65]),
        z_range=(0.0, 1.0),
    )
    z_sn = np.array([0.20, 0.30, 0.40])
    z_host, host_match, _ = gen_zhost(
        z_sn, np.zeros_like(z_sn), np.full_like(z_sn, 0.5), data,
    )
    # Every SN with nearby ZTRUE should be remapped (host_match False).
    assert not host_match.any()
    # Remapped z must be one of the offsets implied by the table.
    assert (z_host != z_sn).all()


def test_gen_zhost_random_prob_above_threshold_keeps_match():
    """SNe with random_prob > P(wrong) keep correct match."""
    data = WrongHostTable(
        poly_coeffs=np.array([0.1]),  # P(wrong) = 0.1
        ztrue=np.array([0.10, 0.20, 0.30]),
        zmatch=np.array([0.40, 0.45, 0.50]),
        z_range=(0.0, 1.0),
    )
    z_sn = np.array([0.20, 0.20, 0.20])
    # All random_probs above 0.1 → no wrong-host
    z_host, host_match, _ = gen_zhost(
        z_sn, np.array([0.5, 0.7, 0.9]), np.full(3, 0.5), data,
    )
    assert host_match.all()
    np.testing.assert_allclose(z_host, z_sn)


def test_reads_snana_native_commented_poly_key(tmp_path):
    """SNANA's own WRONGHOST files put PROB_WRONGHOST_POLY on a '#' line.

    read_WRONGHOST (sntools_wronghost.c:85-90) scans whitespace-separated
    tokens with fscanf and has no comment syntax, so '#' is just an ignored
    token and the key is still found. Every WRONGHOST_RANDOM.DAT shipped in
    $SNDATA_ROOT/sample_input_files is written that way; a line-based parser
    that drops '#' lines first cannot read any of them.
    """
    from jax_snlc_sim.hostlib.wronghost import read_wronghost_file

    body = "\n".join(f"{z:.3f}  {min(z + 0.15, 0.6):.3f}"
                     for z in [0.05, 0.10, 0.20, 0.30, 0.40])
    commented = tmp_path / "commented.dat"
    commented.write_text(f"# PROB_WRONGHOST_POLY: 0.026 0 0 0\n# ZTRUE ZMATCH\n{body}\n")
    plain = tmp_path / "plain.dat"
    plain.write_text(f"PROB_WRONGHOST_POLY: 0.026 0 0 0\nZTRUE ZMATCH\n{body}\n")

    a = read_wronghost_file(str(commented), 0.02, 0.60)
    b = read_wronghost_file(str(plain), 0.02, 0.60)

    np.testing.assert_allclose(a.poly_coeffs, [0.026, 0, 0, 0])
    np.testing.assert_allclose(a.poly_coeffs, b.poly_coeffs)
    np.testing.assert_allclose(a.ztrue, b.ztrue)
    np.testing.assert_allclose(a.zmatch, b.zmatch)
    assert len(a.ztrue) == 5


def test_gen_zhost_flags_a_wrong_host_with_no_entry_nearby():
    """A wrong host with no table entry within |dz| < 0.01 is flagged, as
    SNANA aborts on it (sntools_wronghost.c:324-329); one with an entry is
    not."""
    data = WrongHostTable(
        poly_coeffs=np.array([1.0]),  # always wrong host
        ztrue=np.array([0.10, 0.20]),
        zmatch=np.array([0.50, 0.55]),
        z_range=(0.0, 1.0),
    )
    _, _, uncovered = gen_zhost(np.array([0.90, 0.105]), np.zeros(2),
                                np.full(2, 0.5), data)
    np.testing.assert_array_equal(np.asarray(uncovered), [True, False])


def test_gen_zhost_dz_window_is_strict():
    """SNANA accepts strict DZ < DZTRUEMAX (sntools_wronghost.c:317); a table
    entry exactly at |dz| = 0.01 does not count."""
    z_sn = np.array([0.5])
    boundary = float(z_sn[0] + DZTRUEMAX)
    data = WrongHostTable(
        poly_coeffs=np.array([1.0]),
        ztrue=np.array([boundary]),
        zmatch=np.array([boundary + 0.1]),
        z_range=(0.0, 1.0),
    )
    _, _, uncovered = gen_zhost(z_sn, np.zeros(1), np.full(1, 0.5), data)
    assert np.asarray(uncovered).all()


def test_gen_zhost_clamp_applies_only_to_wrong_host():
    """The z-range clamp lives inside the wrong-host branch
    (sntools_wronghost.c:344-346); a correct match keeps zSN untouched even
    within 0.001 of the range edge."""
    data = WrongHostTable(
        poly_coeffs=np.array([0.0]),  # never wrong host
        ztrue=np.array([0.10, 0.9995]),
        zmatch=np.array([0.40, 0.50]),
        z_range=(0.0, 1.0),
    )
    z_sn = np.array([0.9995, 0.0002])
    z_host, host_match, _ = gen_zhost(
        z_sn, np.full(2, 0.99), np.full(2, 0.5), data)
    assert host_match.all()
    np.testing.assert_array_equal(z_host, z_sn)


def test_gen_zhost_wronghost_clamped_into_range():
    """A wrong-host remap past the user z range is set to the boundary
    minus/plus 0.001 (sntools_wronghost.c:344-346)."""
    data = WrongHostTable(
        poly_coeffs=np.array([1.0]),  # always wrong host
        ztrue=np.array([0.50]),
        zmatch=np.array([0.75]),      # remap +0.25
        z_range=(0.0, 0.7),
    )
    z_sn = np.array([0.495])
    z_host, host_match, _ = gen_zhost(
        z_sn, np.zeros(1), np.full(1, 0.5), data)
    assert not host_match[0]
    # 0.495 + 0.25 = 0.745 >= zmax → zmax - 0.001
    np.testing.assert_allclose(z_host, [0.699])
