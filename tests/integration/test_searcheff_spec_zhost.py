"""SPECID + zHOST selection efficiency: JAX-internal invariants.

- ``test_spec_scale_halves_pass_rate`` — JAX-only: SPEC_SCALE=0.5 noticeably
  lowers the SPECID-pass rate (small N, loose bound; catches wiring bugs
  like parameter ignored / applied to wrong stage).
- ``test_mixed_spec_zhost_masks`` — JAX-only invariants on the SEARCHEFF
  mask bit distribution when both SDSS SPEC and SDSS zHOST files are loaded.
- ``test_spec_file_with_no_map_*``: a SPEC file with no VARNAMES map passes
  every event, and APPLY_SEARCHEFF_OPT cannot require it.
- ``test_spec_map_*``: per event under NBR_LIST, a HOSTLIB-column map
  variable reads the drawn host, LOGMASS the DDLR-ranked one.
"""

from pathlib import Path

import numpy as np
import pytest

from jax_snlc_core.logging import SNANAError
from jax_snlc_sim import run as sim_run


def _patch_input(src: Path, dst: Path, replacements):
    text = src.read_text()
    for old, new in replacements:
        text = text.replace(old, new)
    dst.write_text(text)
    return dst


def _spec_pass_rate(input_file: Path, n_sne: int, seed: int) -> tuple[int, int]:
    """Return ``(n_pipe_pass, n_spec_pass)`` over ``n_sne`` JAX SNe."""
    config = dict(
        n_sne=n_sne, random_seed=seed,
    )
    sim = sim_run.prepare(str(input_file), **config)
    resources = sim
    results, sel, _ = sim_run.run_simulation(sim)
    mask = np.asarray(sel["sim_searcheff_mask"]).astype(int)
    n_pipe = int(((mask & 1) > 0).sum())
    n_spec = int((((mask & 1) > 0) & ((mask & 2) > 0)).sum())
    return n_pipe, n_spec


def test_spec_scale_halves_pass_rate(snana_fixture, tmp_path):
    """SPEC_SCALE=0.5 noticeably lowers the SPECID-pass rate.

    Single seed, small sample, generous bound around 0.5. Detects the
    real failure modes (parameter ignored → ratio ≈ 1; applied twice or
    to wrong field → ratio implausible; pipeline broken → no SNe pass).
    Does NOT pin the precise factor — the scaling math itself is
    sub-component-level and not worth a multi-thousand-SN integration
    test to verify statistically.
    """
    fx = snana_fixture("searcheff_snii_basic")
    spec_swap = (
        "SEARCHEFF_SPEC_FILE:  NONE",
        "SEARCHEFF_SPEC_FILE:  $JAX_TEST_FIXTURE_ROOT/searcheff/SEARCHEFF_SPEC_SDSS.DAT",
    )
    opt_swap = ("APPLY_SEARCHEFF_OPT:  0", "APPLY_SEARCHEFF_OPT:  1")
    z_swap = ("GENRANGE_REDSHIFT:  0.0200  1.4000",
              "GENRANGE_REDSHIFT:  0.0200  0.3000")  # bright SNe so SPECID curve registers

    noscale = _patch_input(fx.input_file, tmp_path / "noscale.input",
                           [spec_swap, opt_swap, z_swap])
    scale = _patch_input(fx.input_file, tmp_path / "scale.input",
                         [spec_swap, opt_swap, z_swap])
    scale.write_text(scale.read_text() + "\nSEARCHEFF_SPEC_SCALE:  0.5\n")

    n_pipe_no, n_spec_no = _spec_pass_rate(noscale, n_sne=200, seed=12345)
    n_pipe_sc, n_spec_sc = _spec_pass_rate(scale, n_sne=200, seed=12345)
    assert n_pipe_no > 0 and n_pipe_sc > 0, "fixture produced no pipeline-passing SNe"
    rate_no = n_spec_no / n_pipe_no
    rate_sc = n_spec_sc / n_pipe_sc
    assert rate_no > 0, "unscaled SPECID pass rate is zero"
    ratio = rate_sc / rate_no
    assert 0.3 < ratio < 0.7, f"SPEC_SCALE rate ratio {ratio:.3f} outside [0.3, 0.7]"


def _parse_dump_mask(path: Path) -> np.ndarray:
    varnames = None
    rows = []
    with path.open() as f:
        for line in f:
            if line.startswith("VARNAMES:"):
                varnames = line.split(":", 1)[1].split()
            elif line.startswith("SN:"):
                rows.append(line.split()[1:])
    if varnames is None or "SIM_SEARCHEFF_MASK" not in varnames:
        raise ValueError(f"SIM_SEARCHEFF_MASK column missing from {path}")
    idx = varnames.index("SIM_SEARCHEFF_MASK")
    return np.array([int(float(r[idx])) for r in rows])


def test_mixed_spec_zhost_masks(snana_fixture, tmp_path):
    """Mixed SPEC + zHOST: every passing SN has bit 0 (pipeline) set; the
    SPECID-passing subset is exactly the obs with bit 1; SPECID-only set is a
    subset of pipeline-pass; zHOST-only set is disjoint from SPECID-pass."""
    fx = snana_fixture("searcheff_snii_basic")
    patched = _patch_input(
        fx.input_file, tmp_path / "mixed.input", [
            ("APPLY_SEARCHEFF_OPT:  0", "APPLY_SEARCHEFF_OPT:  1"),
            ("SEARCHEFF_SPEC_FILE:  NONE",
             "SEARCHEFF_SPEC_FILE:  $JAX_TEST_FIXTURE_ROOT/searcheff/SEARCHEFF_SPEC_SDSS.DAT"),
            ("SEARCHEFF_zHOST_FILE:  NONE",
             "SEARCHEFF_zHOST_FILE:  $JAX_TEST_FIXTURE_ROOT/searcheff/SEARCHEFF_zHOST_SDSS.DAT"),
            ("GENRANGE_REDSHIFT:  0.0200  1.4000",
             "GENRANGE_REDSHIFT:  0.0200  0.5000"),
        ],
    )
    config = dict(n_sne=300, random_seed=50001)
    sim = sim_run.prepare(str(patched), **config)
    resources = sim
    results, sel, _ = sim_run.run_simulation(sim)

    mask = np.asarray(sel["sim_searcheff_mask"]).astype(int)
    pipe = (mask & 1) > 0
    spec = (mask & 2) > 0
    zhost = (mask & 4) > 0

    # SPECID and zHOST bits only meaningful when pipeline passed.
    assert not np.any(spec & ~pipe)
    assert not np.any(zhost & ~pipe)

    assert int(pipe.sum()) > 0, "fixture produced no pipeline-passing SNe"

    # bit 3 (zSPEC = SPECID OR zHOST) tracks the OR.
    z_or = (mask & 8) > 0
    assert np.array_equal(z_or, (spec | zhost) & pipe)


def test_spec_file_with_no_map_confirms_every_event_but_cannot_be_required(
        snana_fixture, tmp_path):
    """A SPEC file whose rows have no VARNAMES line holds no map: every
    pipeline-found event is confirmed (gen_searcheff_map,
    sntools_trigger.c:2898-2908), yet the file does not count as a map, so
    APPLY_SEARCHEFF_OPT may not require confirmation
    (check_APPLYMASK_SEARCHEFF, :133-194)."""
    fx = snana_fixture("searcheff_snii_basic")
    spec = tmp_path / "SEARCHEFF_SPEC_NOVARNAMES.DAT"
    spec.write_text("DOCUMENTATION:\n    PURPOSE: rows with no VARNAMES\n"
                    "DOCUMENTATION_END:\n\n"
                    "SPECEFF:  0.0  0.0\nSPECEFF:  2.0  0.0\n")
    spec_swap = ("SEARCHEFF_SPEC_FILE:  NONE", f"SEARCHEFF_SPEC_FILE:  {spec}")

    # the fixture input sets APPLY_SEARCHEFF_OPT: 1
    patched = _patch_input(fx.input_file, tmp_path / "opt1.input", [spec_swap])
    config = dict(n_sne=100, random_seed=50001)
    sim = sim_run.prepare(str(patched), **config)
    resources = sim
    assert resources.searcheff_specid_maps == []
    _, sel, _ = sim_run.run_simulation(sim)
    mask = np.asarray(sel["sim_searcheff_mask"]).astype(int)
    pipe = (mask & 1) > 0
    assert pipe.any(), "fixture produced no pipeline-passing SNe"
    assert np.all((mask[pipe] & 2) > 0)

    patched = _patch_input(fx.input_file, tmp_path / "opt3.input", [
        spec_swap, ("APPLY_SEARCHEFF_OPT:  1", "APPLY_SEARCHEFF_OPT:  3")])
    with pytest.raises(SNANAError, match="Invalid APPLY_SEARCHEFF_OPT: 3"):
        sim_run.prepare(str(patched), **config)


def _nbr_spec_pass(snana_fixture, tmp_path, varname, lo, hi, step,
                   high=1.0):
    """``(specid_pass, triggered, parameters)`` for hostlib_nbr under a SPEC
    map in ``varname`` that steps from 0 to ``high`` between ``step`` and
    ``step + 0.05``."""
    nodes = np.round(np.arange(lo, hi + 0.025, 0.05), 3)
    spec = tmp_path / f"SPEC_{varname}.DAT"
    spec.write_text(
        "DOCUMENTATION:\n    PURPOSE: step map\nDOCUMENTATION_END:\n\n"
        f"VARNAMES: {varname} SPECEFF\n"
        + "".join(f"SPECEFF: {x:.3f} {high if x > step else 0.0}\n"
                  for x in nodes))
    fx = snana_fixture("hostlib_nbr")
    patched = _patch_input(
        fx.input_file, tmp_path / f"{varname}.input",
        [("SEARCHEFF_SPEC_FILE:  NONE", f"SEARCHEFF_SPEC_FILE:  {spec}")])
    config = dict(n_sne=300, random_seed=4321)
    sim = sim_run.prepare(str(patched), **config)
    resources = sim
    results, sel, _ = sim_run.run_simulation(sim)
    return (np.asarray(sel["specid_pass"]), np.asarray(sel["triggered"]),
            results["parameters"])


def _assert_spec_follows(passed, triggered, value, step, other):
    """Among triggered events off the ramp, SPECID passes exactly where
    ``value`` is above the step, and some of them ``other`` would decide
    the other way."""
    used = triggered & ((value <= step) | (value >= step + 0.05))
    np.testing.assert_array_equal(passed[used], value[used] > step)
    assert np.any(used & ((value > step) != (other > step)))


def test_spec_map_hostlib_column_reads_drawn_host(snana_fixture, tmp_path):
    """A SPEC map in a plain HOSTLIB column (LOG_SFR) reads the drawn host,
    SIM_HOSTLIB(LOG_SFR), not the DDLR-ranked first host of NBR_LIST."""
    passed, triggered, p = _nbr_spec_pass(snana_fixture, tmp_path, "LOG_SFR",
                                          -3.0, 3.0, 0.0)
    _assert_spec_follows(passed, triggered,
                         np.asarray(p["sim_hostlib_LOG_SFR"]), 0.0,
                         np.asarray(p["host_LOG_SFR"]))


def test_spec_map_logmass_reads_ranked_host(snana_fixture, tmp_path):
    """LOGMASS is SNANA's own map variable and reads the true LOGMASS of
    the DDLR-ranked first host, not the drawn one."""
    passed, triggered, p = _nbr_spec_pass(snana_fixture, tmp_path, "LOGMASS",
                                          7.5, 13.5, 10.5)
    _assert_spec_follows(passed, triggered, np.asarray(p["host_LOGMASS"]),
                         10.5, np.asarray(p["sim_hostlib_LOGMASS_TRUE"]))


def test_spec_eff_outside_roundoff_window_aborts(snana_fixture, tmp_path):
    """A map node above SNANA's round-off window loads, and the run aborts
    once an evaluated event's EFF lands above it (gen_searcheff_map)."""
    with pytest.raises(SNANAError, match=r"Invalid EFF\(SPEC\) = 1\.2000"):
        _nbr_spec_pass(snana_fixture, tmp_path, "LOG_SFR", -3.0, 3.0, 0.0,
                       high=1.2)
