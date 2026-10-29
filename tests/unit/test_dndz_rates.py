"""Unit tests for cosmic-rate calculations: POWERLAW, CC_S15, MD14."""

import numpy as np
import pytest

from jax_snlc_core.logging import SNANAError
from jax_snlc_sim.rate import (compute_rate_on_grid, _parse_dndz_multiplier,
                               _sfr_bg03, _SFRfun_MD14, parse_dndz_entries)


def _resolved_counts(cfg):
    from jax_snlc_sim.rate import resolve_counts
    return resolve_counts(cfg, cfg.hz_map())


def _resolved_budgets(cfg):
    """The per-template MXGEN the build would set, via the model's own hook."""
    from jax_snlc_core.models.non1a import NON1ASEDModel
    n_simulated, _, frac_pec1a, _ = _resolved_counts(cfg)
    model = object.__new__(NON1ASEDModel)
    model._non1a_config = cfg.model._structured_config['non1a_config']
    model._budget_templates(n_simulated, frac_pec1a)
    return model._mxgen_per_template


def test_md14_peaks_near_z2():
    """MD14 rate peaks near z≈2 — defining feature of the cosmic SFR history."""
    z = np.linspace(0.01, 6.0, 200)
    sfr = _SFRfun_MD14(z, A=0.015, B=2.9, C=2.7, D=5.6)
    z_peak = float(z[int(np.argmax(sfr))])
    assert 1.5 < z_peak < 2.5


def test_cc_s15_scales_with_user_scale():
    """CC_S15 rate scales linearly with user-provided ``scale``."""
    z = np.array([0.5])
    r1 = compute_rate_on_grid(z, {"model": "CC_S15", "scale": 1.0, "h0": 70.0})
    r2 = compute_rate_on_grid(z, {"model": "CC_S15", "scale": 2.0, "h0": 70.0})
    np.testing.assert_allclose(r2, 2.0 * r1, rtol=1e-6)


def test_parse_dndz_multiplier_prefix_and_suffix():
    """Both ``'0.7*CC_S15'`` and ``'CC_S15*0.7'`` are accepted."""
    s, name = _parse_dndz_multiplier("0.7*CC_S15")
    assert s == 0.7 and name == "CC_S15"
    s, name = _parse_dndz_multiplier("CC_S15*0.7")
    assert s == 0.7 and name == "CC_S15"


def test_parse_dndz_multiplier_no_multiplier():
    s, name = _parse_dndz_multiplier("POWERLAW")
    assert s == 1.0 and name == "POWERLAW"


# ----------------------------------------------------------------------------
# parse_dndz_entries: builds the per-entry DNDZ parameter dict list
# ----------------------------------------------------------------------------


def test_parse_dndz_entries_powerlaw():
    out = parse_dndz_entries(
        [["POWERLAW", "2.5e-5", "1.5"]], (0.05, 1.0), h0=70.0)
    assert len(out) == 1
    assert out[0]["model"] == "POWERLAW"
    assert abs(out[0]["A"] - 2.5e-5) < 1e-12
    assert abs(out[0]["B"] - 1.5) < 1e-9
    assert out[0]["zmin"] == 0.05 and out[0]["zmax"] == 1.0


def test_parse_dndz_entries_powerlaw2_uses_entry_z_range():
    out = parse_dndz_entries(
        [["POWERLAW2", "1.0e-5", "2.0", "0.0", "0.4"]], (0.05, 1.4), h0=70.0)
    assert out[0]["model"] == "POWERLAW2"
    assert out[0]["zmin"] == 0.0
    assert out[0]["zmax"] == 0.4


def test_parse_dndz_entries_md14():
    # MD14 scales with h^2, so h0 is load-bearing here
    out = parse_dndz_entries([["MD14", "2.5e-5"]], (0.01, 2.0), h0=70.0)
    assert out[0]["model"] == "MD14"
    assert abs(out[0]["R0"] - 2.5e-5) < 1e-12


def test_parse_dndz_entries_cc_s15_with_multiplier():
    # CC_S15 scales with h^2, so h0 is load-bearing here
    out = parse_dndz_entries([["0.7*CC_S15"]], (0.01, 1.5), h0=70.0)
    assert out[0]["model"] == "CC_S15"
    assert abs(out[0]["scale"] - 0.7) < 1e-9


def test_parse_dndz_entries_unknown_model_aborts():
    """A model SNANA supports but jax does not must not be dropped — that left
    dN/dz at zero for every redshift with only a log line."""
    with pytest.raises(SNANAError, match="Unknown DNDZ model"):
        parse_dndz_entries([["UNKNOWN_MODEL", "1.0"]], (0.01, 1.0), h0=70.0)


# ----------------------------------------------------------------------------
# parse_dndz_entries: invalid params are fatal (snlc_sim.c:18060-18066)
# ----------------------------------------------------------------------------


def test_parse_dndz_powerlaw2_invalid_params_fatal():
    """A/B above 10 in POWERLAW2 is fatal, matching SNANA's powerlaw bounds
    check (snlc_sim.c:18060-18066)."""
    with pytest.raises(SNANAError, match="powerlaw"):
        parse_dndz_entries(
            [["POWERLAW2", 100.0, 100.0, 0.0, 1.0]], (0.0, 2.0), h0=70.0)


def test_parse_dndz_powerlaw_invalid_params_fatal():
    """The same A/B bounds apply to plain POWERLAW (snlc_sim.c:18060-18066)."""
    with pytest.raises(SNANAError, match="powerlaw"):
        parse_dndz_entries([["POWERLAW", 100.0, 100.0]], (0.0, 2.0), h0=70.0)


def test_parse_dndz_powerlaw2_short_row_fatal():
    """A POWERLAW2 row missing its z range is a recognised-key error and
    must abort rather than be dropped."""
    with pytest.raises(SNANAError, match="POWERLAW2"):
        parse_dndz_entries([["POWERLAW2", 1.0e-5]], (0.0, 2.0), h0=70.0)


def test_parse_dndz_powerlaw_non_numeric_fatal():
    """Non-numeric POWERLAW parameters abort instead of warn-and-drop."""
    with pytest.raises(SNANAError, match="POWERLAW"):
        parse_dndz_entries([["POWERLAW", "abc", "2.0"]], (0.0, 2.0), h0=70.0)


# ----------------------------------------------------------------------------
# compute_rate_on_grid: unknown-model error
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# calculate_expected_sn_count
# ----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DNDZ_* rate modifiers (genz_wgt): scales and reweights
# ---------------------------------------------------------------------------

def _rate_mod_count(tmp_path, extra, non1ased=False, drop_dndz=False, subs=()):
    """Expected SN count for salt3_k21_replay plus `extra` config lines."""
    import pathlib
    from jax_snlc_sim.inputs import parse_snana_input_file

    root = pathlib.Path(__file__).resolve().parents[2]
    src = (root / "tests/fixtures/snana/salt3_k21_replay/input.input").read_text()
    if non1ased:
        src = src.replace("GENMODEL:  $JAX_TEST_FIXTURE_ROOT/models/SALT3.K21",
                          "GENMODEL:  $JAX_TEST_FIXTURE_ROOT/models/NON1ASED.V19_TEST")
    if drop_dndz:
        src = "".join(l + "\n" for l in src.splitlines()
                      if not l.startswith("DNDZ:"))
    for old, new in subs:
        src = src.replace(old, new)
    inp = tmp_path / "rate.input"
    inp.write_text(src + extra)
    return _expected_count(parse_snana_input_file(str(inp)))


def _expected_count(cfg):
    """The nominal-population rate integral for a parsed config."""
    from jax_snlc_sim.rate import calculate_expected_sn_count
    c = cfg.cosmology
    return calculate_expected_sn_count(
        cfg.dndz_params(), cfg.rates.solid_angle, cfg.rates.redshift_range,
        cfg.rates.peakmjd_range, cfg.rates.peakmjd_exclude, c.omega_matter,
        c.omega_lambda, c.h0, c.w0, c.wa, cfg.hz_map(),
        cfg.rates.dndz_rate_mods)


def test_dndz_scales_and_reweights(tmp_path):
    """DNDZ_* modifiers multiply the rate, in SNANA's genz_wgt order.

    Ratios against jax's own baseline, not SNANA's printed integers: SNANA
    reports `(int)(SEASON_COUNT + 0.5)`, so equality could be off by one.
    Reference values from stock SNANA with INIT_ONLY (base = 1320):
    ALLSCALE 2.0 -> 2641, ZEXP 1.0 -> 391, ZPOLY 0,1 -> 391, Z1POLY 0,1 -> 1711.
    """
    base = _rate_mod_count(tmp_path, "")

    assert _rate_mod_count(tmp_path, "DNDZ_ALLSCALE: 2.0\n") == pytest.approx(2 * base)

    # DNDZ_ZPOLY_REWGT 0,1 is w *= z, identical to DNDZ_ZEXP_REWGT 1.0 --
    # SNANA gives 391 for both, so they must agree here too.
    zexp = _rate_mod_count(tmp_path, "DNDZ_ZEXP_REWGT: 1.0\n")
    zpoly = _rate_mod_count(tmp_path, "DNDZ_ZPOLY_REWGT: 0,1\n")
    assert zexp == pytest.approx(zpoly)
    assert zexp / base == pytest.approx(391 / 1320, rel=2e-3)

    z1poly = _rate_mod_count(tmp_path, "DNDZ_Z1POLY_REWGT: 0,1\n")
    assert z1poly / base == pytest.approx(1711 / 1320, rel=2e-3)


def test_dndz_scale_slot_follows_model_type(tmp_path):
    """DNDZ_SCALE is <Ia> <NON1A>; SNANA picks [1] for NON1A/SIMSED models.

    Both arms matter: a SALT-only test passes with the selector inverted.
    Confirmed against SNANA in all four cells.
    """
    for non1ased, ia_factor, cc_factor in ((False, 2.0, 1.0), (True, 1.0, 2.0)):
        base = _rate_mod_count(tmp_path, "", non1ased)
        ia = _rate_mod_count(tmp_path, "DNDZ_SCALE: 2.0 1.0\n", non1ased)
        cc = _rate_mod_count(tmp_path, "DNDZ_SCALE: 1.0 2.0\n", non1ased)
        non1a_only = _rate_mod_count(tmp_path, "DNDZ_SCALE_NON1A: 2.0\n", non1ased)
        assert ia == pytest.approx(ia_factor * base)
        assert cc == pytest.approx(cc_factor * base)
        assert non1a_only == pytest.approx(cc_factor * base)


def test_dndz_non1a_is_an_alias(tmp_path):
    """DNDZ_NON1A / DNDZ_NONIA are NOMINAL-group primary keys, i.e. DNDZ."""
    base = _rate_mod_count(tmp_path, "")
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    src = (root / "tests/fixtures/snana/salt3_k21_replay/input.input").read_text()
    for alias in ("DNDZ_NON1A", "DNDZ_NONIA"):
        inp = tmp_path / f"{alias}.input"
        inp.write_text(src.replace("DNDZ:  POWERLAW", f"{alias}:  POWERLAW"))
        from jax_snlc_sim.inputs import parse_snana_input_file
        got = _expected_count(parse_snana_input_file(str(inp)))
        assert got == pytest.approx(base), f"{alias} did not alias DNDZ"


# ---------------------------------------------------------------------------
# DNDZ_FILE: tabulated R(z), read and interpolated as SNANA does
# ---------------------------------------------------------------------------


# A kinked table, so the count depends on the interpolation scheme: nearest-node
# or a flat read would not reproduce SNANA's number.
_KINK_TABLE = "0.0 1.0e-5\n0.2 3.0e-5\n0.4 2.0e-5\n0.6 5.0e-5\n"

# The same table behind every comment convention rd2columnFile skips.
_KINK_TABLE_COMMENTED = (
    "# hash comment\n"
    "! bang comment\n"
    "% percent comment\n"
    "@ at comment\n"
    "DOCUMENTATION:\n"
    "  PURPOSE: nonsense 999.0 999.0\n"
    "  NOTE: 111.0 222.0\n"
    "DOCUMENTATION_END:\n"
) + _KINK_TABLE


def _dndz_file_count(tmp_path, table, name="rate.dat", drop_dndz=True, extra="", **kwargs):
    path = tmp_path / name
    path.write_text(table)
    return _rate_mod_count(tmp_path, f"DNDZ_FILE: {path}\n" + extra,
                           drop_dndz=drop_dndz, **kwargs)


def test_dndz_file_matches_snana(tmp_path):
    """DNDZ_FILE reads R(z) from a table and interpolates it linearly.

    Reference values from stock SNANA with INIT_ONLY, off salt3_k21_replay
    (GENRANGE_REDSHIFT 0.02-0.40, baseline POWERLAW count 1320):
    a flat 2.5e-5 table gives 902, the kinked table gives 877.

    The kinked case is asserted at SOLID_ANGLE x100, where SNANA prints 87662
    -- the count scales linearly with solid angle, so this pins the agreement
    to 1e-5 instead of the +-0.5 that rounding to 877 would allow.
    """
    # Supply SNANA's radiation-free reference cosmology through the normal
    # HzFUN_FILE input. Production's analytic cosmology includes radiation.
    z = np.linspace(0., 0.6, 6001)
    hz = 70. * np.sqrt(0.315 * (1. + z)**3 + 0.685)
    hz_path = tmp_path / 'snana_hz.dat'
    np.savetxt(hz_path, np.column_stack([z, hz]))
    cosmology = f'HzFUN_FILE: {hz_path}\n'

    flat = "".join(f"{0.1*i:.3f} 2.500000e-05\n" for i in range(6))
    assert _dndz_file_count(tmp_path, flat, extra=cosmology) == pytest.approx(902, abs=0.5)

    kink = _dndz_file_count(tmp_path, _KINK_TABLE, extra=cosmology)
    assert kink == pytest.approx(877, abs=0.5)

    big = _dndz_file_count(
        tmp_path, _KINK_TABLE, extra=cosmology,
        subs=[("SOLID_ANGLE:  0.0150", "SOLID_ANGLE:  1.5000")])
    assert big == pytest.approx(87662, rel=1e-5)


def test_dndz_file_skips_comments_and_docana(tmp_path):
    """rd2columnFile skips #, !, %, @ rows and DOCUMENTATION blocks.

    The commented table carries numeric-looking DOCANA rows, so a reader that
    ignored the block would fold 999.0 and 111.0 into the rate.
    """
    clean = _dndz_file_count(tmp_path, _KINK_TABLE, name="clean.dat")
    commented = _dndz_file_count(tmp_path, _KINK_TABLE_COMMENTED, name="cmt.dat")
    assert commented == pytest.approx(clean)


def test_dndz_file_aborts_outside_table_range(tmp_path):
    """SNANA's quickBinSearch aborts when z leaves the table, rather than
    extrapolating: 'VAL = 3.005000e-01 outside SNrate_model range'."""
    short = "".join(f"{0.1*i:.3f} 2.500000e-05\n" for i in range(4))  # 0.0 - 0.3
    with pytest.raises(SNANAError, match="DNDZ_FILE"):
        _dndz_file_count(tmp_path, short)


def test_dndz_file_cannot_be_mixed_with_dndz(tmp_path):
    """SNANA: 'cannot mix multiple rate models with DNDZ_FILE ... (pick one)'."""
    with pytest.raises(SNANAError, match="cannot mix"):
        _dndz_file_count(tmp_path, _KINK_TABLE, drop_dndz=False)


# ---------------------------------------------------------------------------
# DNDZ_PEC1A: peculiar-Ia population with its own rate
# ---------------------------------------------------------------------------

def _pec1a_config(tmp_path, nominal, pec1a, extra=""):
    """non1ased_v19_replay with a PEC1A template and two rate models."""
    import pathlib
    from jax_snlc_sim.inputs import parse_snana_input_file
    root = pathlib.Path(__file__).resolve().parents[2]
    src = (root / "tests/fixtures/snana/non1ased_v19_replay/input.input").read_text()
    src = src.replace("NON1A: 834 1.0 0.00 0.00 21",
                      "NON1A: 834 1.0 0.00 0.00 21\nPEC1A: 501 1.0 0.00 0.00 21")
    src = src.replace("DNDZ:  0.7*CC_S15",
                      f"DNDZ:  {nominal}\nDNDZ_PEC1A: {pec1a}")
    src = src.replace("NGENTOT_LC:  1500", "NGEN_SEASON: 1.0")
    inp = tmp_path / "pec1a.input"
    inp.write_text(src + extra)
    return parse_snana_input_file(str(inp))


def test_dndz_pec1a_fraction_and_budgets(tmp_path):
    """FRAC_PEC1A from the two rate integrals, then the class-wise budgets.

    Stock SNANA with DNDZ POWERLAW 2.5E-5 1.5 and DNDZ_PEC1A POWERLAW 1.0E-5
    1.5: "Calculated Number of EVENTS per season = 1849 (0.286 PEC1A)".
    Equal beta makes the fraction z-independent, so it reduces to
    1.0/(2.5+1.0) = 0.2857.
    """
    cfg = _pec1a_config(tmp_path, "POWERLAW 2.5E-5 1.5", "POWERLAW 1.0E-5 1.5")

    n_simulated, _, frac_pec1a, _ = _resolved_counts(cfg)
    assert frac_pec1a == pytest.approx(1.0 / 3.5, rel=1e-3)
    assert n_simulated == 1848          # int(NGEN_SEASON x 1848.6)

    assert list(_resolved_budgets(cfg)) == [1320, 1848]  # the two rate integrals


def test_dndz_pec1a_fraction_is_z_dependent_for_differing_beta(tmp_path):
    """Differing beta makes the split z-dependent, not a ratio of amplitudes.

    Stock SNANA with DNDZ POWERLAW 2.5E-5 3.0 and DNDZ_PEC1A POWERLAW 1.0E-5
    0.0: "= 2311 (0.156 PEC1A)". A fraction taken from amplitudes alone would
    give 1.0/3.5 = 0.286 here.
    """
    cfg = _pec1a_config(tmp_path, "POWERLAW 2.5E-5 3.0", "POWERLAW 1.0E-5 0.0")
    n_simulated, _, frac_pec1a, _ = _resolved_counts(cfg)
    assert frac_pec1a == pytest.approx(0.156, abs=0.001)
    assert n_simulated == pytest.approx(2311, abs=1)


def test_dndz_pec1a_reweights_do_not_reach_the_pec1a_rate(tmp_path):
    """The DNDZ_* modifiers sit inside `if (IS_NOMINAL)` (snlc_sim.c:2722).

    Doubling the nominal rate must lower the PEC1A fraction, not leave it
    unchanged as it would if DNDZ_ALLSCALE also scaled the PEC1A rate.
    """
    base = _pec1a_config(tmp_path, "POWERLAW 2.5E-5 1.5", "POWERLAW 1.0E-5 1.5")
    scaled = _pec1a_config(tmp_path, "POWERLAW 2.5E-5 1.5", "POWERLAW 1.0E-5 1.5",
                           extra="\nDNDZ_ALLSCALE: 2.0\n")
    assert _resolved_counts(base)[2] == pytest.approx(1.0 / 3.5, rel=1e-3)
    assert _resolved_counts(scaled)[2] == pytest.approx(1.0 / 6.0, rel=1e-3)


def test_dndz_pec1a_requires_a_non1a_model(tmp_path):
    """DNDZ_PEC1A key allowed only for NON1A model (snlc_sim.c:2793)."""
    import pathlib
    from jax_snlc_sim.inputs import parse_snana_input_file
    root = pathlib.Path(__file__).resolve().parents[2]
    src = (root / "tests/fixtures/snana/salt3_k21_replay/input.input").read_text()
    inp = tmp_path / "salt_pec1a.input"
    inp.write_text(src + "\nDNDZ_PEC1A: POWERLAW 1.0E-5 1.5\n")
    with pytest.raises(SNANAError, match="only for NON1A"):
        parse_snana_input_file(str(inp))


def test_pec1a_templates_without_a_pec1a_rate_abort(tmp_path):
    """prep_NON1ASED aborts on PEC1A SEDs with PEC1A rate=0."""
    import pathlib
    from jax_snlc_sim.inputs import parse_snana_input_file
    root = pathlib.Path(__file__).resolve().parents[2]
    src = (root / "tests/fixtures/snana/non1ased_v19_replay/input.input").read_text()
    src = src.replace("NON1A: 834 1.0 0.00 0.00 21",
                      "NON1A: 834 1.0 0.00 0.00 21\nPEC1A: 501 1.0 0.00 0.00 21")
    inp = tmp_path / "no_pec1a_rate.input"
    inp.write_text(src)
    cfg = parse_snana_input_file(str(inp))
    with pytest.raises(SNANAError, match="PEC1A rate=0"):
        _resolved_budgets(cfg)


def test_dndz_pec1a_normalises_weights_within_each_class(tmp_path):
    """WGT is normalised inside a class, then budgeted by FRAC_PEC1A.

    Two NON1A at 3:1 plus one PEC1A over 1000 events. Stock SNANA reports
    "Will generate 536 (MXGEN= 536)", "179 (MXGEN= 715)", "285 (MXGEN=1000)":
    the NON1A budget 1000*(1-0.2857)=714.3 split 3:1, then the remainder.
    A single WGTSUM across all three would give 600/200/200 instead.
    """
    import pathlib
    from jax_snlc_sim.inputs import parse_snana_input_file
    root = pathlib.Path(__file__).resolve().parents[2]
    src = (root / "tests/fixtures/snana/non1ased_v19_replay/input.input").read_text()
    src = src.replace("NON1A: 834 1.0 0.00 0.00 21",
                      "NON1A: 834 3.0 0.00 0.00 21\nNON1A: 835 1.0 0.00 0.00 21\n"
                      "PEC1A: 501 1.0 0.00 0.00 21")
    src = src.replace("DNDZ:  0.7*CC_S15",
                      "DNDZ:  POWERLAW 2.5E-5 1.5\nDNDZ_PEC1A: POWERLAW 1.0E-5 1.5")
    src = src.replace("NGENTOT_LC:  1500", "NGENTOT_LC:  1000")
    inp = tmp_path / "pec1a_multi.input"
    inp.write_text(src)
    cfg = parse_snana_input_file(str(inp))

    non1a = cfg.model._structured_config['non1a_config']
    assert list(non1a['index_per_template']) == [834, 835, 501]
    assert list(non1a['ispec1a_per_template']) == [False, False, True]
    assert list(_resolved_budgets(cfg)) == [536, 715, 1000]


# ----------------------------------------------------------------------------
# Rate models added 2026-08-13. Each expected value is SNANA's own formula from
# SNrate_model (snlc_sim.c:17939-18092), not a copy of the implementation.
# ----------------------------------------------------------------------------


def _rate_of(tokens, z, cosmology=None):
    p = parse_dndz_entries([tokens], (0.01, 2.0), h0=70.0,
                           cosmology=cosmology)[0]
    return np.asarray(compute_rate_on_grid(np.asarray(z), p), dtype=float)


def test_hubble_and_flat_are_constants():
    """HUBBLE is a flat 3e-5 ("approx SNIa rate at z ~ 0"); FLAT is 1e-8, which
    SNANA calls meaningless — it exists only to avoid an abort."""
    z = np.array([0.02, 0.5, 1.9])
    # atol=0: these rates are ~1e-8, so np.allclose's default atol would
    # compare any two of them equal.
    assert np.allclose(_rate_of(['HUBBLE'], z), 3.0e-5, atol=0)
    assert np.allclose(_rate_of(['FLAT'], z), 1.0e-8, atol=0)


def test_pisn_polynomial_and_multiplier():
    z = np.array([0.0, 0.7])
    expected = (1.98 + 6.38 * z + 6.558 * z**2
                - 4.42 * z**3 + 0.8312 * z**4 - 0.0508 * z**5) / 1.0e9
    assert np.allclose(_rate_of(['PISN_PLK12'], z), expected, atol=0)
    assert np.allclose(_rate_of(['3*PISN_PLK12'], z), 3.0 * expected, atol=0)


def test_tde_decays_as_ten_to_the_minus_z():
    z = np.array([0.0, 0.6, 1.2])
    assert np.allclose(_rate_of(['TDE', '1e-6'], z),
                       1e-6 * 10.0 ** (-0.5 * z / 0.6), atol=0)


@pytest.mark.parametrize("spec", ['1e-5,2e-6,3e-7', None])
def test_zpoly_accepts_comma_and_legacy_spacing(spec):
    """read_genpoly takes 'a0,a1,a2' or three space-separated terms."""
    tokens = ['ZPOLY', spec] if spec else ['ZPOLY', '1e-5', '2e-6', '3e-7']
    z = np.array([0.0, 1.0, 2.0])
    assert np.allclose(_rate_of(tokens, z), 1e-5 + 2e-6 * z + 3e-7 * z**2,
                       atol=0)


def test_abmodel_prompt_term_contributes():
    """SNANA's prompt term is inert (its SFRfun_BG03 arguments are swapped); jax applies it, so B changes the rate."""
    from jax_snlc_sim.simulation import Cosmology
    cos = Cosmology()
    z = np.array([0.1, 1.0])
    a_only = _rate_of(['ABMODEL', '1e-14', '0.0'], z, cos)
    with_b = _rate_of(['ABMODEL', '1e-14', '1e-3'], z, cos)
    assert np.allclose(with_b - a_only, 1e-3 * _sfr_bg03(z, 70.0), atol=0)
    assert np.all(with_b > a_only * 2)


def test_abmodel_rejects_out_of_range_parameters():
    """SNANA aborts for A > 1e-13 or B > 0.01 (snlc_sim.c:18044)."""
    with pytest.raises(SNANAError, match="A\\+B rate model"):
        parse_dndz_entries([['ABMODEL', '1e-12', '1e-3']], (0.01, 1.0), h0=70.0)
