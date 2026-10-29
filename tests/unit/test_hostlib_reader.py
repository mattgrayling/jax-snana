"""Unit tests for HOSTLIB reader helpers (path-handling, sentinel detection)."""

from pathlib import Path

import numpy as np
import pytest

from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import is_ignorefile, open_text
from jax_snlc_sim.hostlib.catalogue import (_apply_read_selection, _canonical_properties,
                                                 _remap_nbr_array_to_sorted,
                                                 discover_output_only_columns,
                                                 essential_columns, read_hostlib)
from jax_snlc_sim.hostlib.sersic import build_sersic_profiles
from jax_snlc_sim.hostlib.photoz import zphot_columns, zphot_quantile_columns


def test_is_ignorefile_none_and_empty():
    """``None``, empty, and SNANA's NONE/NULL/BLANK sentinels all ignored."""
    for v in (None, "", "NONE", "none", "NULL", "BLANK"):
        assert is_ignorefile(v) is True


def _sersic_columns(ngal=5, **cols):
    data = {"GALID": np.arange(ngal), "ZTRUE": np.linspace(0.1, 0.5, ngal)}
    for name, val in cols.items():
        data[name] = np.full(ngal, val, dtype=np.float32)
    return data, list(data.keys())


def test_build_sersic_profiles_none_when_no_morphology():
    """A HOSTLIB with no a/b columns has no profile; nothing is fabricated."""
    data, varnames = _sersic_columns()
    assert build_sersic_profiles(data, varnames, {}) is None


def test_build_sersic_profiles_partial_ab_pair_is_fatal():
    """a without b (or vice versa) aborts (sntools_host.c:5020-5030)."""
    data, varnames = _sersic_columns(a0_Sersic=1.5, n0_Sersic=2.0, a_rot=0.0)
    with pytest.raises(SNANAError, match="b0_Sersic"):
        build_sersic_profiles(data, varnames, {})


def test_build_sersic_profiles_missing_index_is_fatal():
    """A component without an n column or fixed header value aborts
    (sntools_host.c:5086-5110)."""
    data, varnames = _sersic_columns(a0_Sersic=1.5, b0_Sersic=1.0, a_rot=0.0)
    with pytest.raises(SNANAError, match="n0_Sersic"):
        build_sersic_profiles(data, varnames, {})


def test_build_sersic_profiles_fixed_header_index():
    """n{j}_Sersic given as a fixed header value fills the component index
    (parse_Sersic_n_fixed, sntools_host.c:3064-3093)."""
    data, varnames = _sersic_columns(a0_Sersic=1.5, b0_Sersic=1.0, a_rot=10.0)
    prof = build_sersic_profiles(data, varnames, {0: 4.0})
    assert prof['n_prof'] == 1
    np.testing.assert_allclose(prof['n'][:, 0], 4.0)
    # Single profile with no weight column gets w = 1 (sntools_host.c:5364)
    np.testing.assert_allclose(prof['w'][:, 0], 1.0)


def test_build_sersic_profiles_infers_missing_weight():
    """With NPROF-1 weights declared, the last is 1 - sum(others)
    (sntools_host.c:5362-5364); too few weights abort."""
    data, varnames = _sersic_columns(
        a0_Sersic=1.5, b0_Sersic=1.0, n0_Sersic=1.0, w0_Sersic=0.3,
        a1_Sersic=0.5, b1_Sersic=0.4, n1_Sersic=4.0,
        a2_Sersic=0.2, b2_Sersic=0.1, n2_Sersic=2.0, w2_Sersic=0.5,
        a_rot=0.0)
    prof = build_sersic_profiles(data, varnames, {})
    assert prof['n_prof'] == 3
    np.testing.assert_allclose(prof['w'][:, 1], 0.2, rtol=1e-6)

    data2, varnames2 = _sersic_columns(
        a0_Sersic=1.5, b0_Sersic=1.0, n0_Sersic=1.0,
        a1_Sersic=0.5, b1_Sersic=0.4, n1_Sersic=4.0,
        a2_Sersic=0.2, b2_Sersic=0.1, n2_Sersic=2.0, w2_Sersic=0.5,
        a_rot=0.0)
    with pytest.raises(SNANAError, match="Inadequate Sersic weights"):
        build_sersic_profiles(data2, varnames2, {})


def test_build_sersic_profiles_bad_weight_sum_is_fatal():
    """All-declared weights summing off unity beyond 1e-3 abort; small
    truncation slop is renormalised (sntools_host.c:5413-5434)."""
    data, varnames = _sersic_columns(
        a0_Sersic=1.5, b0_Sersic=1.0, n0_Sersic=1.0, w0_Sersic=0.4,
        a1_Sersic=0.5, b1_Sersic=0.4, n1_Sersic=4.0, w1_Sersic=0.4,
        a_rot=0.0)
    with pytest.raises(SNANAError, match="Sum of Sersic weights"):
        build_sersic_profiles(data, varnames, {})

    data2, varnames2 = _sersic_columns(
        a0_Sersic=1.5, b0_Sersic=1.0, n0_Sersic=1.0, w0_Sersic=0.6004,
        a1_Sersic=0.5, b1_Sersic=0.4, n1_Sersic=4.0, w1_Sersic=0.4,
        a_rot=0.0)
    prof = build_sersic_profiles(data2, varnames2, {})
    np.testing.assert_allclose(prof['w'].sum(axis=1), 1.0, rtol=1e-6)


def test_discovery_of_output_columns_and_quantiles():
    """Output-only columns and quantile percentiles come from the
    HOSTLIB's actual VARNAMES (sntools_host.c:2977-2986, 2757-2766)."""
    varnames = ['GALID', 'ZTRUE', 'ZPHOT', 'ZPHOT_ERR', 'ZPHOT_Q5',
                'ZPHOT_Q050', 'ZPHOT_Q95', 'J_obs', 'J_obs_err', 'K_obs',
                'LOGMASS', 'OBJID2']
    assert zphot_quantile_columns(varnames) == [
        ('ZPHOT_Q5', 5.0), ('ZPHOT_Q050', 50.0), ('ZPHOT_Q95', 95.0)]
    with pytest.raises(SNANAError, match="21 photo-z quantile columns .* exceed MXBIN_HOSTGALz_QUANTILE = 20"):
        zphot_quantile_columns([f'QZPHOT{i:02d}' for i in range(21)])
    cols = discover_output_only_columns(varnames)
    assert set(cols) == {'ZPHOT', 'ZPHOT_ERR', 'ZPHOT_Q5', 'ZPHOT_Q050',
                         'ZPHOT_Q95', 'J_obs', 'J_obs_err', 'K_obs', 'OBJID2'}


def test_zphot_error_column_found_under_every_spelling_snana_maps():
    """ZPHOT_ERR is the column HOSTLIB_VARNAME_ZPHOT names with ERR or _ERR
    appended, or ZPHOTERR (sntools_host.c:2988-2996)."""
    assert zphot_columns('ZPHOT', ['GALID', 'ZPHOT', 'ZPHOTERR']) == ('ZPHOT', 'ZPHOTERR')
    assert zphot_columns('ZPHOT', ['ZPHOT', 'ZPHOT_ERR']) == ('ZPHOT', 'ZPHOT_ERR')
    assert zphot_columns('PZ', ['PZ', 'PZ_ERR']) == ('PZ', 'PZ_ERR')
    assert zphot_columns('ZPHOT', ['ZPHOT']) == ('ZPHOT', None)


def test_indexed_quantile_columns_imply_uniform_percentiles():
    """QZPHOTnn columns are nn+1 quantiles at percentiles 100 i / nn (manual,
    Generating Host Galaxy Photometric Redshifts); mixing them with the
    ZPHOT_Qnnn convention is refused."""
    varnames = ['GALID', 'ZTRUE', 'QZPHOT03', 'QZPHOT00', 'QZPHOT02', 'QZPHOT01', 'QZPHOT04']
    assert zphot_quantile_columns(varnames) == [
        ('QZPHOT00', 0.0), ('QZPHOT01', 25.0), ('QZPHOT02', 50.0),
        ('QZPHOT03', 75.0), ('QZPHOT04', 100.0)]
    assert set(discover_output_only_columns(varnames)) == {f'QZPHOT0{i}' for i in range(5)}
    with pytest.raises(SNANAError, match="both as ZPHOT_Q\\* and as QZPHOT\\*"):
        zphot_quantile_columns(['GALID', 'ZPHOT_Q050', 'QZPHOT00'])


# ----------------------------------------------------------------------------
# _remap_nbr_array_to_sorted: row reorder + value remap on the same call
# ----------------------------------------------------------------------------


def test_remap_nbr_array_translates_indices_when_sorted():
    """When galaxies get reordered, neighbour indices follow the new ordering."""
    # 3 galaxies in original order.
    # Original neighbours: galaxy 0 → 1, galaxy 1 → 2, galaxy 2 → 0.
    nbr = np.array([
        [1, -1],
        [2, -1],
        [0, -1],
    ], dtype=np.int32)
    # Sort permutation: original [0, 1, 2] → sorted [2, 0, 1]
    # i.e. row 0 of sorted holds the data that was at original index 2.
    sorted_idx = np.array([2, 0, 1], dtype=np.int32)
    out = _remap_nbr_array_to_sorted(nbr, sorted_idx)
    # In sorted space, sorted-row 0 (== original 2) had nbr 0 (== sorted-row 1).
    assert int(out[0, 0]) == 1
    # Sorted-row 1 (== original 0) had nbr 1 (== sorted-row 2).
    assert int(out[1, 0]) == 2
    # Sorted-row 2 (== original 1) had nbr 2 (== sorted-row 0).
    assert int(out[2, 0]) == 0
    # Padding survives.
    assert (out[:, 1] == -1).all()


def test_fixsersic_applies_on_both_cold_and_warm_cache(tmp_path, jax_test_hostlib_path):
    """HOSTLIB_FIXSERSIC writes into the catalogue columns (VALUE_ZSORTED,
    sntools_host.c:5303-5330). The first read parses the source and reloads
    through the freshly written cache, so the override has to survive that
    round trip as well as a later warm read."""
    fix = (1.0, 0.5, 4.0, 30.0)
    cold = read_hostlib(str(jax_test_hostlib_path), essential_columns(),
                        cache_dir=str(tmp_path), fixsersic=fix)
    warm = read_hostlib(str(jax_test_hostlib_path), essential_columns(),
                        cache_dir=str(tmp_path), fixsersic=fix)
    for label, cat in (("cold", cold), ("warm", warm)):
        prof = cat.sersic_profiles
        assert np.allclose(np.asarray(prof["a"]), 1.0), label
        assert np.allclose(np.asarray(prof["b"]), 0.5), label
        assert np.allclose(np.asarray(prof["n"]), 4.0), label
        assert np.allclose(np.asarray(prof["a_rot"]), 30.0), label


def test_scale_sersic_size_reaches_the_catalogue_profiles(tmp_path,
                                                          jax_test_hostlib_path):
    """SNANA scales SERSIC->a/b on the same struct the position, DDLR and
    GALMAG paths all read (sntools_host.c:5328-5329), so the scale has to
    live on the catalogue profiles rather than on one consumer's copy."""
    plain = read_hostlib(str(jax_test_hostlib_path), essential_columns(),
                         cache_dir=str(tmp_path / "a"))
    scaled = read_hostlib(str(jax_test_hostlib_path), essential_columns(),
                          cache_dir=str(tmp_path / "b"), scale_sersic_size=2.0)
    a0 = np.asarray(plain.sersic_profiles["a"])
    a1 = np.asarray(scaled.sersic_profiles["a"])
    assert np.allclose(a1, 2.0 * a0)


def test_scale_sersic_size_multiplies_a_fixsersic_value(tmp_path,
                                                        jax_test_hostlib_path):
    """FIXSERSIC resolves the size first and the scale multiplies it, in
    SNANA's order (sntools_host.c:5303-5329)."""
    cat = read_hostlib(str(jax_test_hostlib_path), essential_columns(),
                       cache_dir=str(tmp_path),
                       fixsersic=(1.0, 0.5, 4.0, 30.0), scale_sersic_size=3.0)
    prof = cat.sersic_profiles
    assert np.allclose(np.asarray(prof["a"]), 3.0)
    assert np.allclose(np.asarray(prof["b"]), 1.5)
    assert np.allclose(np.asarray(prof["n"]), 4.0)


def test_a_property_column_is_kept_under_its_bare_name():
    """LOGSFR_TRUE in a HOSTLIB is the LOGSFR column (sntools_host.c:2481);
    a file spelling one property both ways is refused."""
    columns = _canonical_properties(
        {"GALID": np.arange(3), "LOGSFR_TRUE": np.ones(3), "LOGMASS": np.zeros(3),
         "LOGMASS_ERR": np.ones(3)}, "H.HOSTLIB")

    assert list(columns) == ["GALID", "LOGSFR", "LOGMASS", "LOGMASS_ERR"]
    with pytest.raises(SNANAError, match="both LOGMASS and LOGMASS_TRUE"):
        _canonical_properties({"LOGMASS": np.zeros(3), "LOGMASS_TRUE": np.zeros(3)},
                              "H.HOSTLIB")


def _hostlib_without(path, *columns):
    """The HOSTLIB text with ``columns`` removed from VARNAMES and every row."""
    with open_text(Path(path)) as stream:
        fixture_text = stream.read()
    lines = fixture_text.splitlines()
    ivar = next(i for i, l in enumerate(lines) if l.startswith("VARNAMES:"))
    names = lines[ivar].split()[1:]
    keep = [i for i, n in enumerate(names) if n not in columns]
    out = []
    for i, line in enumerate(lines):
        if i == ivar:
            out.append("VARNAMES: " + " ".join(names[j] for j in keep))
        elif line.startswith("GAL:"):
            toks = line.split()[1:]
            out.append("GAL: " + " ".join(toks[j] for j in keep))
        else:
            out.append(line)
    return "\n".join(out) + "\n"


@pytest.mark.parametrize("dropped", [("ZPHOT", "ZPHOT_ERR"), ("ZPHOT_ERR",)])
def test_swapzphot_without_its_columns_aborts_by_name(
        jax_test_hostlib_path, tmp_path, dropped):
    """SNANA refuses HOSTLIB_MSKOPT bit 128 without ZPHOT and ZPHOT_ERR
    (prep_head_HOSTLIB, sntools_host.c:2830-2836). The reader drops a
    requested column the file lacks, so the abort has to be here, at the
    substitution, rather than a bare KeyError."""
    stripped = tmp_path / "NOZPHOT.HOSTLIB"
    stripped.write_text(_hostlib_without(jax_test_hostlib_path, *dropped))
    with pytest.raises(SNANAError, match="SWAPZPHOT"):
        read_hostlib(str(stripped), essential_columns() + ["ZPHOT", "ZPHOT_ERR"],
                     swap_zphot=True, use_cache=False)


def test_swapzphot_sorts_the_catalogue_on_the_swapped_column(
        jax_test_hostlib_path, tmp_path):
    """SNANA swaps while reading and sorts afterwards (sntools_host.c:3228,
    :193), so the array the redshift window searches is ascending. The
    HOSTLIB z range stays the file's own ZTRUE, which the read loop
    accumulates before the swap (:3201-3204)."""
    columns = essential_columns() + ["ZPHOT", "ZPHOT_ERR"]
    plain = read_hostlib(str(jax_test_hostlib_path), columns,
                         cache_dir=str(tmp_path))
    swapped = read_hostlib(str(jax_test_hostlib_path), columns,
                           cache_dir=str(tmp_path), swap_zphot=True)

    z_swapped = swapped.columns["ZTRUE"]
    assert np.all(np.diff(z_swapped) >= 0.0)
    # it really is the photo-z column, not the true one
    assert not np.array_equal(
        z_swapped, plain.columns["ZTRUE"])
    assert (swapped.z_min, swapped.z_max) == (plain.z_min, plain.z_max)

def test_host_coordinates_are_read_under_any_of_their_three_names():
    """RA, RA_HOST and RA_GAL name the host RA, and likewise for DEC
    (sntools_host.c:2740-2750); each is kept as RA_GAL/DEC_GAL."""
    for ra, dec in (("RA", "DEC"), ("RA_HOST", "DEC_HOST"), ("RA_GAL", "DEC_GAL")):
        columns = _canonical_properties(
            {"GALID": np.arange(2), ra: np.ones(2), dec: np.zeros(2)}, "H.HOSTLIB")
        assert list(columns) == ["GALID", "RA_GAL", "DEC_GAL"]
    with pytest.raises(SNANAError, match="both RA and RA_HOST"):
        _canonical_properties({"RA": np.zeros(2), "RA_HOST": np.zeros(2)}, "H.HOSTLIB")


def test_the_read_time_redshift_window_drops_unreachable_galaxies():
    """A galaxy outside GENRANGE_REDSHIFT widened by 0.01 and three log-z
    bins is never stored (passCuts_HOSTLIB, sntools_host.c:3147-3156,
    :3369-3370); a lower edge at or below 0.01 cuts nothing below, as
    SNANA's log10 of a non-positive number compares false."""
    ztrue = np.array([0.003, 0.02, 0.10, 0.1177, 0.1179, 0.5])
    columns = {"ZTRUE": ztrue, "GALID": np.arange(6)}
    nbr = np.full((6, 1), -1, dtype=np.int32)

    order, kept, _ = _apply_read_selection(
        columns, np.arange(6), nbr, ztrue, (-999.0, 999.0), (-999.0, 999.0),
        (0.007, 0.0999), 80_000_000, (None, ''))
    np.testing.assert_array_equal(kept["GALID"], [0, 1, 2, 3])
    assert len(order) == 4

    order, kept, _ = _apply_read_selection(
        columns, np.arange(6), nbr, ztrue, (-999.0, 999.0), (-999.0, 999.0),
        (0.02, 0.0999), 80_000_000, (None, ''))
    np.testing.assert_array_equal(kept["GALID"], [1, 2, 3])

    order, *_ = _apply_read_selection(
        columns, np.arange(6), nbr, ztrue, (-999.0, 999.0), (-999.0, 999.0),
        None, 80_000_000, (None, ''))
    assert len(order) == 6


def test_the_redshift_window_reads_the_file_ztrue_under_swapzphot():
    """SNANA applies the window to the row as read (sntools_host.c:3179),
    before SWAPZPHOT overwrites ZTRUE with ZPHOT (:3228), so a galaxy is kept
    or dropped on its true redshift whatever it is later sorted on."""
    ztrue_file = np.array([0.05, 0.30, 0.60])
    zphot = np.array([0.60, 0.30, 0.05])
    columns = {"ZTRUE": zphot, "GALID": np.arange(3)}
    nbr = np.full((3, 1), -1, dtype=np.int32)
    order, kept, _ = _apply_read_selection(
        columns, np.arange(3), nbr, ztrue_file, (-999.0, 999.0), (-999.0, 999.0),
        (0.2, 0.4), 80_000_000, (None, ''))
    np.testing.assert_array_equal(kept["GALID"], [1])
    assert len(order) == 1


def test_snr_store_keeps_galaxies_whose_brightest_bands_reach_the_ladder():
    """HOSTLIB_SNR_STORE ranks each galaxy's bands by signal-to-noise and
    holds the i-th best to the i-th threshold (passCuts_snr_HOSTLIB,
    sntools_host.c:8290) over the GENFILTERS bands carrying both a mag and
    an error column; the ratio is (2.5/ln 10)/err with an error below 0 or
    above 9 read as 9 and an error of 0 giving no signal (snr_store_HOSTLIB,
    :8226). A ladder longer than the bands present aborts (:8310)."""
    snr = 2.5 / np.log(10.0)
    columns = {
        "ZTRUE": np.full(5, 0.1), "GALID": np.arange(5),
        "g_obs": np.full(5, 22.0), "r_obs": np.full(5, 21.0),
        "i_obs": np.full(5, 21.0),
        # (g, r) errors: both good; one good; error 0 and one good;
        # negative (read as 9) and one good; both too faint
        "g_obs_err": np.array([snr / 30, snr / 30, 0.0, -1.0, snr / 5]),
        "r_obs_err": np.array([snr / 12, snr / 8, snr / 12, snr / 12, snr / 4]),
    }
    nbr = np.full((5, 1), -1, dtype=np.int32)
    window = ((-999.0, 999.0), (-999.0, 999.0), None, 80_000_000)

    order, kept, _ = _apply_read_selection(
        columns, np.arange(5), nbr, columns["ZTRUE"], *window, ([10.0], 'gri'))
    np.testing.assert_array_equal(kept["GALID"], [0, 1, 2, 3])
    order, kept, _ = _apply_read_selection(
        columns, np.arange(5), nbr, columns["ZTRUE"], *window, ([20.0, 10.0], 'gri'))
    np.testing.assert_array_equal(kept["GALID"], [0])
    assert len(order) == 1
    # r alone is in GENFILTERS: only its own column counts
    order, kept, _ = _apply_read_selection(
        columns, np.arange(5), nbr, columns["ZTRUE"], *window, ([10.0], 'r'))
    np.testing.assert_array_equal(kept["GALID"], [0, 2, 3])
    with pytest.raises(SNANAError, match="NBAND = 2 but NBAND_SNR_CUT==3"):
        _apply_read_selection(
            columns, np.arange(5), nbr, columns["ZTRUE"], *window,
            ([20.0, 10.0, 5.0], 'gri'))


def test_shared_table_read_preserves_ids_neighbours_and_embedded_weights(tmp_path):
    """HOSTLIB reads keep int64 IDs and neighbour links, ignoring WGTMAP rows."""
    source = tmp_path / 'embedded.HOSTLIB'
    source.write_text(
        'VARNAMES: GALID ZTRUE LOGMASS NBR_LIST\n'
        'GAL: 9007199254740993 0.3 10.0 1,2\n'
        'GAL: 9007199254740994 0.1 9.0 -1\n'
        'GAL: 9007199254740995 0.2 11.0 0\n'
        'SN: 123 0.15 8.0 -1\n'
        'ROW: 124 0.25 8.0 -1\n'
        'OBS: ignored even without enough columns\n'
        'VARNAMES_WGTMAP: LOGMASS WGT SNMAGSHIFT\n'
        'WGT: 9.0 1.0 0.0\n'
        'WGT: 11.0 2.0 0.1\n')
    for cached in (False, True, True):
        table = read_hostlib(source, ['GALID', 'ZTRUE', 'LOGMASS'],
                             use_cache=cached, cache_dir=tmp_path / 'cache')
        np.testing.assert_array_equal(table.columns['GALID'],
                                      [9007199254740994, 9007199254740995, 9007199254740993])
        assert table.columns['GALID'].dtype == np.int64
        np.testing.assert_allclose(table.columns['ZTRUE'], [0.1, 0.2, 0.3])
        np.testing.assert_array_equal(table.neighbours, [[-1, -1], [2, -1], [0, 1]])
