"""Integration scenarios for SNANA's SPECID + zHOST selection efficiency.

These exercise the full simulator with various SEARCHEFF_SPEC / SEARCHEFF_zHOST
configurations and check downstream effects (redshift_flag, observed redshift,
HEAD output columns, CUTWIN behaviour).

Pure-JAX unit tests of the underlying ``searcheff_map`` live in
``tests/unit/test_searcheff_map.py``.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest


from jax_snlc_sim import run as sim_run  # noqa: E402
from jax_snlc_sim.output.dump import _prepare_dump_data  # noqa: E402
from jax_snlc_sim.output.tables import (  # noqa: E402
    build_header_columns, compute_derived_values,
)
from jax_snlc_sim.selection import _build_columnar_data  # noqa: E402


def _derived(simulation, resources, sel, results):
    """The distance moduli and peak magnitudes the writer would compute."""
    cd = _build_columnar_data(resources, results, sel)
    return compute_derived_values(
        cd, results, resources.filter_names,
        len(cd["parameters"]["sim_redshift_cmb"]), sel)


def _peak_mags(simulation, resources, sel, results):
    return _derived(simulation, resources, sel, results)[1]


def _header_table(simulation, resources, sel, results):
    """The HEAD columns the writer would build for this run."""
    cd = _build_columnar_data(resources, results, sel)
    distance_moduli, peak_magnitudes = _derived(simulation, resources, sel, results)
    return build_header_columns(
        cd, simulation, resources, resources.filter_names,
        distance_moduli, resources.model,
        len(cd["parameters"]["sim_redshift_cmb"]),
        peak_magnitudes, _identity_cids)


# No CIDRAN in these scenarios, so the written CID is the CID.
_identity_cids = np.asarray


def _run_sim(input_file, n_sne=50, seed=50001):
    # the scenarios re-select the finished run on every event's epochs
    config = dict(
        n_sne=n_sne, batch_size=50, random_seed=seed, keep_rejected_events=True,
    )
    sim = sim_run.prepare(str(input_file), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    results, sel, _ = sim_run.run_simulation(sim)
    return simulation, resources, results, sel


def _patched_input(src: Path, dst: Path, replacements: list[tuple[str, str]]) -> Path:
    """Write a copy of ``src`` to ``dst`` with each (old, new) replacement applied."""
    text = src.read_text()
    for old, new in replacements:
        text = text.replace(old, new)
    dst.write_text(text)
    return dst


def test_peakmag_sentinel_output(snana_fixture):
    """SIM_PEAKMAG_<filter> populated, in [16, 32], and bluer-bands-fainter for low-z SNII."""

    fx = snana_fixture("searcheff_snii_basic")
    simulation, resources, results, sel = _run_sim(fx.input_file, n_sne=200)
    n = int(sel["n_passed"])
    assert n > 0, "searcheff_snii_basic passed no SNe"

    h = _header_table(simulation, resources, sel, results)

    band_mags = {b: np.asarray(h[f"SIM_PEAKMAG_{b}"]) for b in ["u", "g", "r", "i", "z", "Y"]}
    means = {}
    for band, m in band_mags.items():
        valid = m > 0
        assert np.any(valid), f"SIM_PEAKMAG_{band} has no valid entries"
        assert np.all((m[valid] > 16) & (m[valid] < 32)), (
            f"SIM_PEAKMAG_{band} out of range: [{m[valid].min():.2f}, {m[valid].max():.2f}]"
        )
        means[band] = float(np.mean(m[valid]))

    assert means["u"] > means["Y"] + 0.5, (
        f"u-Y colour wrong sign: mean(u)={means['u']:.2f}, mean(Y)={means['Y']:.2f}"
    )


@pytest.fixture(scope="module")
def specz_zero_run(snana_fixture, tmp_path_factory):
    """50 SNe with SPEC and zHOST efficiency zeroed. Both redshift-flag tests
    inject different host columns, so each copies the parameters before use."""
    fx = snana_fixture("searcheff_snii_basic")
    patched = _patched_input(
        fx.input_file, tmp_path_factory.mktemp("specz") / "specz_zero.input",
        [
            ("SEARCHEFF_SPEC_FILE:  NONE", "SEARCHEFF_SPEC_FILE:  ZERO"),
            ("SEARCHEFF_zHOST_FILE:  NONE", "SEARCHEFF_zHOST_FILE:  ZERO"),
        ],
    )
    config = dict(n_sne=50, random_seed=50001,
                                 keep_rejected_events=True)
    sim = sim_run.prepare(str(patched), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    results, _, _ = sim_run.run_simulation(sim)
    return simulation, resources, sim, results


def test_setz_unconfirmed_photoz(specz_zero_run):
    """SPEC=ZERO + zHOST=ZERO + photo-z available → use photo-z (FLAG=3)."""
    simulation, resources, sim, _base = specz_zero_run
    results = dict(_base, parameters=dict(_base["parameters"]))

    # setz_unconfirmed reads SNHOSTGAL.ZPHOT (snlc_sim.c:24922), the value
    # HOSTLIB_ZPHOTEFF masks, not the photo-z the data file reports.
    n_sne = len(results["parameters"]["sim_redshift_cmb"])
    true_z = np.array(results["parameters"]["sim_redshift_cmb"])
    results["parameters"]["host_ZPHOT_EVENT"] = true_z + 0.05
    results["parameters"]["host_ZPHOT_EVENT_ERR"] = np.full(n_sne, 0.03)

    sel = sim_run.apply_selection(sim, results)
    assert int(sel["n_passed"]) > 0, "searcheff_snii_basic passed no SNe"

    p = _build_columnar_data(resources, results, sel)["parameters"]
    assert not np.any(p["specid_pass"])
    assert np.all(p["redshift_flag"] == 3)
    passing = sel["passing_mask"]
    expected_z = (true_z + 0.05)[passing]
    assert np.allclose(p["redshift_helio"], expected_z, atol=1e-5)
    assert np.allclose(p["sim_redshift_cmb"], true_z[passing], atol=1e-5)


def test_wronghost_flag(specz_zero_run):
    """CORRECT_HOSTMATCH=0 with photo-z path → FLAG=4 (WRONGHOST)."""
    simulation, resources, sim, _base = specz_zero_run
    results = dict(_base, parameters=dict(_base["parameters"]))

    n_sne = len(results["parameters"]["sim_redshift_cmb"])
    true_z = np.array(results["parameters"]["sim_redshift_cmb"])
    results["parameters"]["CORRECT_HOSTMATCH"] = np.zeros(n_sne, dtype=np.int32)
    results["parameters"]["host_ZTRUE"] = true_z + 0.1
    results["parameters"]["host_ZPHOT_EVENT"] = true_z + 0.05
    results["parameters"]["host_ZPHOT_EVENT_ERR"] = np.full(n_sne, 0.03)

    sel = sim_run.apply_selection(sim, results)
    assert int(sel["n_passed"]) > 0, "searcheff_snii_basic passed no SNe"

    flag = _build_columnar_data(resources, results, sel)["parameters"]["redshift_flag"]
    # All CORRECT_HOSTMATCH=0 with photo-z available → all FLAG=4.
    assert np.all(flag == 4)


def test_cutwin_redshift_final(snana_fixture, tmp_path):
    """CUTWIN_REDSHIFT_FINAL: cuts on observed (modified) redshift."""
    fx = snana_fixture("searcheff_snii_basic")
    src = fx.input_file.read_text()
    src = src.replace("SEARCHEFF_SPEC_FILE:  NONE", "SEARCHEFF_SPEC_FILE:  ZERO")
    src += "\nCUTWIN_REDSHIFT_FINAL:  0.3  0.8\n"
    patched = tmp_path / "cutwin_zfinal.input"
    patched.write_text(src)

    config = dict(n_sne=200, random_seed=50001,
                                 keep_rejected_events=True)
    sim = sim_run.prepare(str(patched), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    results, _, _ = sim_run.run_simulation(sim)

    n_sne = len(results["parameters"]["sim_redshift_cmb"])
    true_z = np.array(results["parameters"]["sim_redshift_cmb"])
    results["parameters"]["host_ZPHOT_EVENT"] = true_z
    results["parameters"]["host_ZPHOT_EVENT_ERR"] = np.full(n_sne, 0.03)

    sel = sim_run.apply_selection(sim, results)
    n = int(sel["n_passed"])
    assert n > 0, "searcheff_snii_basic passed no SNe"

    z_final = _build_columnar_data(resources, results, sel)["parameters"]["sim_redshift_cmb"]
    in_range = (z_final >= 0.3) & (z_final <= 0.8)
    sentinel = z_final < 0
    assert np.all(in_range | sentinel)


# =============================================================================
# Scenario tests — vary SPEC / zHOST file directives with APPLY_SEARCHEFF_OPT=1
# =============================================================================


def _opt1_input(src: Path, dst: Path, *extra_replacements: tuple[str, str]) -> Path:
    """Patch the searcheff_snii_basic input to APPLY_SEARCHEFF_OPT=1 (gate
    SPECID/zHOST), then apply any caller-specified file/keyword swaps."""
    return _patched_input(
        src, dst,
        [("APPLY_SEARCHEFF_OPT:  0", "APPLY_SEARCHEFF_OPT:  1"), *extra_replacements],
    )


def test_scenario_spec_none(snana_fixture, tmp_path):
    """SPEC=NONE → all spec-confirmed: specid_pass=True, redshift_flag=1, mask & 2."""
    fx = snana_fixture("searcheff_snii_basic")
    patched = _opt1_input(fx.input_file, tmp_path / "spec_none.input")
    simulation, resources, results, sel = _run_sim(patched)
    assert int(sel["n_passed"]) > 0, "searcheff_snii_basic passed no SNe"
    p = _build_columnar_data(resources, results, sel)["parameters"]
    assert np.all(p["specid_pass"])
    assert np.all(p["redshift_flag"] == 1)
    assert np.all(p["sim_searcheff_mask"] & 2)


def test_scenario_spec_zero_zhost_none(snana_fixture, tmp_path):
    """SPEC=ZERO + zHOST=NONE → all phot, all get host spec-z (flag=2)."""
    fx = snana_fixture("searcheff_snii_basic")
    patched = _opt1_input(
        fx.input_file, tmp_path / "spec_zero.input",
        ("SEARCHEFF_SPEC_FILE:  NONE", "SEARCHEFF_SPEC_FILE:  ZERO"),
    )
    simulation, resources, results, sel = _run_sim(patched)
    assert int(sel["n_passed"]) > 0, "searcheff_snii_basic passed no SNe"
    p = _build_columnar_data(resources, results, sel)["parameters"]
    passing = sel["passing_mask"]
    true_zhel = np.array(results["parameters"]["sim_redshift_helio"])[passing]

    assert not np.any(p["specid_pass"])
    assert np.all(p["redshift_flag"] == 2)
    assert np.allclose(p["redshift_helio"], true_zhel, atol=0.01)
    assert np.all(p["sim_searcheff_mask"] & 4)
    assert not np.any(p["sim_searcheff_mask"] & 2)
    h = _header_table(simulation, resources, sel, results)
    assert np.all(h["SNTYPE"] >= 100)
    assert np.all(h["SIM_SEARCHEFF_MASK"] & 1)


def test_scenario_spec_zero_zhost_zero(snana_fixture, tmp_path):
    """SPEC=ZERO + zHOST=ZERO → no z source (flag=0, redshift_helio=-9)."""
    fx = snana_fixture("searcheff_snii_basic")
    patched = _opt1_input(
        fx.input_file, tmp_path / "all_zero.input",
        ("SEARCHEFF_SPEC_FILE:  NONE", "SEARCHEFF_SPEC_FILE:  ZERO"),
        ("SEARCHEFF_zHOST_FILE:  NONE", "SEARCHEFF_zHOST_FILE:  ZERO"),
    )
    simulation, resources, results, sel = _run_sim(patched)
    assert int(sel["n_passed"]) > 0, "searcheff_snii_basic passed no SNe"
    p = _build_columnar_data(resources, results, sel)["parameters"]
    assert not np.any(p["specid_pass"])
    assert np.all(p["redshift_flag"] == 0)
    assert np.all(p["redshift_helio"] == -9.0)


def test_scenario_pipeline_mask(snana_fixture, tmp_path):
    """Pipeline-failing SNe carry mask=0; OVP-failing SNe get SNTYPE=0 in DUMPALL."""

    fx = snana_fixture("searcheff_snii_basic")
    patched = _opt1_input(
        fx.input_file, tmp_path / "pipeline_mask.input",
        ("SEARCHEFF_SPEC_FILE:  NONE", "SEARCHEFF_SPEC_FILE:  ZERO"),
    )
    simulation, resources, results, sel = _run_sim(patched, n_sne=100)
    mask_full = np.asarray(sel["sim_searcheff_mask"]).astype(int)
    pipe_pass = (mask_full & 1).astype(bool)
    pipe_fail = ~pipe_pass

    assert np.all(mask_full[pipe_fail] == 0)
    if np.any(pipe_pass):
        assert np.all(mask_full[pipe_pass] & 1)

    dump = _prepare_dump_data(
        _build_columnar_data(resources, results, sel), simulation, resources, results,
        list(resources.photometry.transmissions.keys()),
        resources.model, _peak_mags(simulation, resources, sel, results), sel,
        include_all=True, to_written_cid=_identity_cids)
    sntypes = np.asarray(dump["sntype"]).astype(int)
    apply_opt = simulation.survey.apply_searcheff_opt
    if apply_opt > 0:
        ovp_pass = (mask_full & apply_opt) == apply_opt
        assert np.all(sntypes[~ovp_pass] == 0)


def test_speceff_map_on_salt2mb(snana_fixture, tmp_path):
    """SPECEFF map keyed on SALT2mB (a model-derived SEARCHEFF variable):
    with a step map at mB=23, bright SNe always spec-confirm and faint
    SNe never do. The map is uniform in 0.001 steps from 22 to 24, and
    OPT_EXTRAP: 1 carries its edge values beyond."""
    fx = snana_fixture("des_salt3_replay")
    spec_map = tmp_path / "SPEC_SALT2MB.DAT"
    knots = np.round(np.arange(22.0, 24.0005, 0.001), 3)
    spec_map.write_text(
        "DOCUMENTATION:\n    PURPOSE: test SEARCHEFF map\nDOCUMENTATION_END:\n\n"
        "OPT_EXTRAP: 1\n"
        "VARNAMES: SALT2mB SPECEFF\n"
        + "".join(f"SPECEFF: {x:.3f} {1.0 if x <= 23.0 else 0.0}\n"
                  for x in knots)
    )
    patched = _patched_input(fx.input_file, tmp_path / "input.input", [
        ("SEARCHEFF_SPEC_FILE:  NONE", f"SEARCHEFF_SPEC_FILE: {spec_map}"),
    ])
    simulation, resources, results, sel = _run_sim(patched, n_sne=120)
    assert int(sel["n_passed"]) > 0, "des_salt3_replay passed no SNe"
    p = _build_columnar_data(resources, results, sel)["parameters"]
    x0 = np.asarray(p["x0"])
    mb = -2.5 * np.log10(np.where(x0 > 0, x0, 1e-30)) + 10.635
    confirmed = np.asarray(p["specid_pass"])
    bright, faint = mb < 23.0, mb > 23.001
    assert np.any(bright) and np.any(faint), (
        "des_salt3_replay mB distribution no longer straddles the 23.0 step"
    )
    assert np.all(confirmed[bright])
    assert not np.any(confirmed[faint])


def test_speceff_map_missing_a_field_aborts_unless_told_not_to(snana_fixture, tmp_path):
    """A triggered event whose field matches no SPECEFF map aborts, as the
    refactored map reader does (gen_searcheff_map, sntools_trigger.c:2985);
    OPT_FIELDMATCH_REQUIRE: 0 in the map file gives it EFF=0 instead
    (:2990). The DES SIMLIB has deep and shallow fields; the map names the
    deep ones only."""
    from jax_snlc_core.utils.error_handling import SNANAError

    fx = snana_fixture("des_salt3_replay")
    body = ("VARNAMES: SALT2mB SPECEFF\n"
            "SPECEFF: 10.0   1.0\n"
            "SPECEFF: 99.0   1.0\n")
    strict = tmp_path / "SPEC_DEEP.DAT"
    strict.write_text("DOCUMENTATION:\n    PURPOSE: test SEARCHEFF map\n"
                      "DOCUMENTATION_END:\n\nFIELDLIST: deep1,deep2\n" + body)
    patched = _patched_input(fx.input_file, tmp_path / "strict.input", [
        ("SEARCHEFF_SPEC_FILE:  NONE", f"SEARCHEFF_SPEC_FILE: {strict}"),
    ])
    with pytest.raises(SNANAError, match="Invalid NMATCH=0 SPECEFF maps for FIELD = 'shallow"):
        _run_sim(patched, n_sne=120)

    lenient = tmp_path / "SPEC_DEEP_LENIENT.DAT"
    lenient.write_text("DOCUMENTATION:\n    PURPOSE: test SEARCHEFF map\n"
                       "DOCUMENTATION_END:\n\nOPT_FIELDMATCH_REQUIRE: 0\n"
                       "FIELDLIST: deep1,deep2\n" + body)
    patched = _patched_input(fx.input_file, tmp_path / "lenient.input", [
        ("SEARCHEFF_SPEC_FILE:  NONE", f"SEARCHEFF_SPEC_FILE: {lenient}"),
    ])
    simulation, resources, results, sel = _run_sim(patched, n_sne=120)
    field_of = np.array([f.field_name for f in resources.fields])
    fields = field_of[np.asarray(results["parameters"]["field_idx"])]
    confirmed = np.asarray(sel["specid_pass"])
    triggered = np.asarray(sel["triggered"])
    deep = np.char.startswith(fields, "deep")
    assert np.any(triggered & deep) and np.any(triggered & ~deep)
    assert np.all(confirmed[triggered & deep])
    assert not np.any(confirmed[~deep])
