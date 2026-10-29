"""SNANA-vs-JAX cross-validation for NPE_SAT saturation.

When ``NPE_PIXEL_SATURATE`` is lowered so some epochs saturate:
- SEARCHEFF detections fall (saturated obs never detect — sntools_trigger.c:2496)
- ``PHOTFLAG_SATURATE`` is OR'd into PHOTFLAG for saturated epochs
- ``CUTWIN_NOBS_SATURATE`` / ``CUTWIN_NOBS_NOSATURATE`` apply as a CUTMASK bit

Two parity tests + one pure-JAX unit test:
- ``test_saturation_pipeline_pass_rate`` — pipeline-pass rate matches SNANA
  with saturation off/on; saturation actually triggers in the on case.
- ``test_saturation_final_pass_agreement`` — end-to-end final-pass agreement
  on the saturation-on config. Regression for the CUTWIN_SNRMAX bug where
  saturated obs needed flux/error/SNR sentinel-masking so CUTWIN_SNRMAX
  treats them as noise rather than bright peaks.
- ``test_cutwin_nobs_saturate_unit`` — pure-JAX unit test for the
  CUTWIN_NOBS_[NO]SATURATE branch of the CUTWIN kernel (lives here for
  proximity to the parity tests it complements).
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import open_text
from jax_snlc_sim import run as sim_run


def _parse_dump_column(path: Path, column: str) -> np.ndarray:
    varnames = None
    rows = []
    with open_text(path) as f:
        for line in f:
            if line.startswith("VARNAMES:"):
                varnames = line.split(":", 1)[1].split()
            elif line.startswith("SN:"):
                rows.append(line.split()[1:])
    if varnames is None or column not in varnames:
        raise ValueError(f"Column {column!r} not in {path}")
    idx = varnames.index(column)
    return np.array([float(r[idx]) for r in rows])


def _run_jax(input_file: Path, n_sne: int):
    config = dict(
        n_sne=n_sne, random_seed=122459,
    )
    sim = sim_run.prepare(str(input_file), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    results, sel, _ = sim_run.run_simulation(sim)
    mask = np.asarray(sel["sim_searcheff_mask"]).astype(int)
    sat = np.asarray(sel["saturated_mask"])
    pipe_pass = int(((mask & 1) > 0).sum())
    n_passed = int(np.asarray(sel["n_passed"]))
    return pipe_pass, len(mask), int(sat.sum()), n_passed


@pytest.fixture(scope="module")
def saturation_on_run(snana_fixture):
    """(pipe_pass, total, nsat, n_passed) for saturation_on, shared by both tests."""
    fx = snana_fixture("saturation_on")
    return _run_jax(fx.input_file, len(_parse_dump_column(fx.dump, "CID")))


def test_saturation_pipeline_pass_rate(snana_fixture, saturation_on_run):
    """OFF and ON pipeline-pass rates match SNANA; saturation triggers in ON."""
    fx_off = snana_fixture("saturation_off")
    fx_on = snana_fixture("saturation_on")

    off_mask = _parse_dump_column(fx_off.dump, "SIM_SEARCHEFF_MASK").astype(int)
    snana_off = int(((off_mask & 1) > 0).sum()) / len(off_mask)
    jax_off_pipe, jax_off_total, jax_off_nsat, _ = _run_jax(fx_off.input_file, len(off_mask))
    jax_off = jax_off_pipe / jax_off_total

    on_mask = _parse_dump_column(fx_on.dump, "SIM_SEARCHEFF_MASK").astype(int)
    snana_on = int(((on_mask & 1) > 0).sum()) / len(on_mask)
    jax_on_pipe, jax_on_total, jax_on_nsat, _ = saturation_on_run
    jax_on = jax_on_pipe / jax_on_total

    assert abs(snana_off - jax_off) < 0.025, (
        f"OFF rate diverges: SNANA={snana_off:.3f}, JAX={jax_off:.3f}"
    )
    assert abs(snana_on - jax_on) < 0.025, (
        f"ON rate diverges: SNANA={snana_on:.3f}, JAX={jax_on:.3f}"
    )
    assert jax_off_nsat == 0, (
        f"OFF: NPE_PIXEL_SATURATE=1e9 should produce zero saturated obs; got {jax_off_nsat}"
    )
    assert jax_on_nsat > 0, (
        f"ON: low NPE_PIXEL_SATURATE should produce some saturated obs; got {jax_on_nsat}"
    )


def test_saturation_final_pass_agreement(snana_fixture, saturation_on_run):
    """End-to-end final-pass agreement on the saturation-heavy config."""
    fx = snana_fixture("saturation_on")
    # Read SNANA's NGENLC_WRITE from the bundled README (post-cut pass count).
    snana_pass = _read_ngenlc_write(fx.root / "README.md", fx.root)
    n_total = len(_parse_dump_column(fx.dump, "CID"))

    _, jax_total, jax_nsat, jax_pass = saturation_on_run
    assert jax_total == n_total

    snana_rate = snana_pass / n_total
    jax_rate = jax_pass / n_total
    assert abs(snana_rate - jax_rate) < 0.02, (
        f"Final-pass rate diverges by {abs(snana_rate - jax_rate):.3f}: "
        f"SNANA={snana_pass}/{n_total} ({snana_rate:.1%}), "
        f"JAX={jax_pass}/{n_total} ({jax_rate:.1%})"
    )
    assert jax_nsat > 0, "Need saturated obs to exercise the CUTWIN path"


def test_saturation_flags_match_snana_per_epoch(snana_fixture):
    """Per-epoch saturation against snlc_sim.exe, on replayed events.

    Saturation is deterministic given the light curve, so unlike the trigger it
    can be compared event by event once SNANA's own parameters are replayed
    into jax. This is the only test here that pins the quantity the
    CUTWIN_NOBS_SATURATE keys actually cut on — SNANA's own counters for it are
    dead, so its cut cannot be compared.

    The residual tolerance covers epochs whose model magnitude differs from
    SNANA's by more than their margin to the threshold: SIM_MAGOBS agrees only
    to ~7 mmag (95th percentile) through NON1ASED template interpolation, which
    is several times the ~1 mmag needed to move a borderline epoch across.
    """
    from astropy.io import fits
    from replay_snana_dump import build_override_parameters

    fx = snana_fixture("saturation_on")
    varnames, rows = None, []
    for line in open_text(fx.dump):
        if line.startswith("VARNAMES:"):
            varnames = line.split(":", 1)[1].split()
        elif line.startswith("SN:"):
            rows.append(line.split()[1:])
    cols = {n: np.array([float(r[i]) for r in rows])
            for i, n in enumerate(varnames)}
    head = fits.open(fx.head_fits)[1].data
    phot = fits.open(fx.phot_fits)[1].data

    # per-epoch flags are compared for every replayed event
    config = dict(
        n_sne=len(rows), random_seed=122459, keep_rejected_events=True)
    sim = sim_run.prepare(str(fx.input_file), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    override = build_override_parameters(
        cols, simulation, resources, head_columns={"SNID": np.array([int(s) for s in head["SNID"]])})
    results, sel, _ = sim_run.run_simulation(sim, override_parameters=override)

    jsat = np.asarray(sel["saturated_mask"])
    jvalid = np.asarray(sel["valid_mask"])
    jmjd = np.asarray(results["observation_schedule"]["mjds"])
    by_cid = {int(c): i for i, c in enumerate(cols["CID"])}

    photflag = np.asarray(phot["PHOTFLAG"], dtype=np.int64)
    pmjd = np.asarray(phot["MJD"], dtype=float)
    lo = np.asarray(head["PTROBS_MIN"], dtype=int)
    hi = np.asarray(head["PTROBS_MAX"], dtype=int)
    saturate_bit = 16384  # PHOTFLAG_SATURATE; must not collide with DETECT

    matched = disagree = 0
    for k, snid in enumerate(head["SNID"]):
        i = by_cid.get(int(snid))
        if i is None:
            continue
        snana_sat = (photflag[lo[k] - 1:hi[k]] & saturate_bit) > 0
        jax_mjd, jax_sat = jmjd[i][jvalid[i]], jsat[i][jvalid[i]]
        for mjd, sat in zip(pmjd[lo[k] - 1:hi[k]], snana_sat):
            delta = np.abs(jax_mjd - mjd)
            if delta.size and delta.min() < 0.01:
                matched += 1
                if bool(jax_sat[int(delta.argmin())]) != bool(sat):
                    disagree += 1

    assert matched > 10000, f"too few epochs matched by MJD: {matched}"
    assert disagree <= 10, (
        f"{disagree} of {matched} epochs disagree on saturation; more than a "
        f"handful means the photoelectron threshold itself has drifted")


def _run_with_extra(input_file: Path, tmp_path: Path, extra: str, n_sne: int):
    patched = tmp_path / "patched.input"
    patched.write_text(input_file.read_text() + "\n" + extra)
    return _run_jax(patched, n_sne)


def test_cutwin_nobs_saturate_end_to_end(snana_fixture, tmp_path, saturation_on_run):
    """CUTWIN_NOBS_SATURATE selects on the saturated-epoch count end to end.

    SNANA cannot be compared here: its saturated-epoch counters are never
    populated, so the same cut leaves its accept
    count untouched. The reference is SNANA's own photometry — every event it
    writes for this fixture carries at least one PHOTFLAG_SATURATE epoch, so
    "reject any SN with a saturated epoch" must reject nearly all of them.
    """
    fx = snana_fixture("saturation_on")
    n = len(_parse_dump_column(fx.dump, "CID"))
    base = saturation_on_run[3]

    # 0 0: no saturated epoch allowed in any band.
    none_allowed = _run_with_extra(
        fx.input_file, tmp_path, "CUTWIN_NOBS_SATURATE: 0 0 ugrizY\n", n)[3]
    # 1 999: at least one saturated epoch required in every band.
    all_bands = _run_with_extra(
        fx.input_file, tmp_path, "CUTWIN_NOBS_SATURATE: 1 999 ugrizY\n", n)[3]

    assert none_allowed < base / 10, (
        f"expected the saturated majority to be cut: {none_allowed} of {base}")
    assert all_bands == 0, (
        f"no SN saturates in all six bands; got {all_bands}")


def test_cutwin_nobs_nosaturate_end_to_end(snana_fixture, tmp_path, saturation_on_run):
    """CUTWIN_NOBS_NOSATURATE selects on the unsaturated-epoch count.

    SNANA only tests a band whose total epoch count exceeds the triplet's
    minimum (NOBS_TOT > MINOBS, snlc_sim.c:24491-24506), so a minimum the
    cadence cannot reach skips the band instead of rejecting the event.
    """
    fx = snana_fixture("saturation_on")
    n = len(_parse_dump_column(fx.dump, "CID"))
    base = saturation_on_run[3]

    # Median NOBS is 8 per SN across six bands, so most gri bands carry
    # fewer than 4 epochs and the NOBS_TOT > 3 gate skips them, so the cut
    # must NOT reject everything.
    high_min = _run_with_extra(
        fx.input_file, tmp_path, "CUTWIN_NOBS_NOSATURATE: 3 999 gri\n", n)[3]
    # 0 0: no unsaturated epoch allowed in g, r or i (every observed band
    # passes the NOBS_TOT > 0 gate).
    none_allowed = _run_with_extra(
        fx.input_file, tmp_path, "CUTWIN_NOBS_NOSATURATE: 0 0 gri\n", n)[3]

    # Every gri band here carries fewer than 4 epochs, so the gate skips all
    # of them and the cut removes nothing: equality is the exact statement,
    # and any rejection would mean the gate had stopped skipping.
    assert high_min == base, (
        f"NOBS_TOT gate should skip sparse bands rather than reject any; "
        f"got {high_min} of {base}")
    assert 0 < none_allowed < base, (
        f"expected a partial cut on the unsaturated count, got {none_allowed}")


def _read_ngenlc_write(readme: Path, fixture_root: Path) -> int:
    """Return ``NGENLC_WRITE`` from the SNANA README accompanying the fixture.

    The bundled fixture ships only DUMP/HEAD/PHOT, not the SNANA README. We
    derive the equivalent count by inspecting the DUMP rows where the auto-
    appended FLAG_ACCEPT column is 1 (matches SNANA's accepted-event count).
    """
    dump = fixture_root / "DUMP"
    flag_accept = _parse_dump_column(dump, "FLAG_ACCEPT").astype(int)
    return int((flag_accept == 1).sum())


# =============================================================================
# Pure-JAX unit test for the CUTWIN_NOBS_[NO]SATURATE kernel branch
# =============================================================================


def _saturate_fail(filter_idx, valid, saturated, saturate_ranges,
                   nosaturate_ranges, filter_names):
    """The SATURATE fail mask from the CUTWIN kernel, with only these cuts set.

    Every other cut is left unconfigured, so the remaining kernel arguments are
    zeros it will not read.
    """
    from dataclasses import replace
    from jax_snlc_sim.simulation import Cuts
    from jax_snlc_sim.cutwin import cutwin_all, prepare_cutwin

    cutwin = replace(Cuts(), saturate_ranges=saturate_ranges,
                     nosaturate_ranges=nosaturate_ranges)
    arrays, options = prepare_cutwin(cutwin, filter_names, byfield_masks=())
    n_sne, max_obs = filter_idx.shape
    zeros = jnp.zeros((n_sne, max_obs), dtype=jnp.float32)
    _, fails, _ = cutwin_all(arrays, options, zeros, zeros, filter_idx, valid, zeros, valid,
                     jnp.zeros((n_sne, len(filter_names)), dtype=jnp.float32),
                     jnp.zeros(n_sne, dtype=jnp.int32), saturated)
    return fails['SATURATE']


def test_cutwin_nobs_saturate_unit():
    """Per-filter count + range-check logic against synthetic inputs."""

    # 3 SNe × 5 obs, 4 filters [u, g, r, i]
    filter_idx = jnp.array(
        [
            [1, 2, 1, 2, 2],
            [1, 1, 1, 2, 0],
            [1, 2, 2, 0, 3],
        ],
        dtype=jnp.int32,
    )
    valid = jnp.ones_like(filter_idx, dtype=bool)
    saturated = jnp.array(
        [
            [0, 0, 0, 1, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
        ],
        dtype=bool,
    )
    filter_names = ["LSST-u", "LSST-g", "LSST-r", "LSST-i"]

    # Case 1: '0 2 gr' — each of g, r must have ≤ 2 saturated obs
    fail = _saturate_fail(
        filter_idx, valid, saturated,
        saturate_ranges=[(0, 2, "gr")],
        nosaturate_ranges=[],
        filter_names=filter_names,
    )
    assert np.array_equal(np.asarray(fail), [False, True, False])

    # Case 2: '1 999 g' — each SN must have ≥ 1 g-saturated obs
    fail = _saturate_fail(
        filter_idx, valid, saturated,
        saturate_ranges=[(1, 999, "g")],
        nosaturate_ranges=[],
        filter_names=filter_names,
    )
    assert np.array_equal(np.asarray(fail), [True, False, False])

    # Case 2b: NOSATURATE '1 999 i': the NOBS_TOT > MINOBS gate skips
    # bands without enough observations (snlc_sim.c:24491-24506): SN0/SN1
    # have no i obs (NOBS_TOT=0) and SN2 has one (NOBS_TOT=1, not > 1),
    # so no SN is tested and none fails.
    fail = _saturate_fail(
        filter_idx, valid, saturated,
        saturate_ranges=[],
        nosaturate_ranges=[(1, 999, "i")],
        filter_names=filter_names,
    )
    assert not np.any(np.asarray(fail))

    # Case 2c: NOSATURATE '0 0 g': every SN has NOBS_TOT_g > 0 so all are
    # tested; the cut demands zero unsaturated g obs, so SN0 (two
    # unsaturated g obs) fails while SN1/SN2 (every g obs saturated) pass.
    fail = _saturate_fail(
        filter_idx, valid, saturated,
        saturate_ranges=[],
        nosaturate_ranges=[(0, 0, "g")],
        filter_names=filter_names,
    )
    assert np.array_equal(np.asarray(fail), [True, False, False])

    # Case 3: no ranges configured → the cut never fails an SN
    fail = _saturate_fail(
        filter_idx, valid, saturated,
        saturate_ranges=[],
        nosaturate_ranges=[],
        filter_names=filter_names,
    )
    assert not np.any(np.asarray(fail))

    # Case 4: invalid filter char aborts
    with pytest.raises(SNANAError):
        _saturate_fail(
            filter_idx, valid, saturated,
            saturate_ranges=[(0, 2, "x")],
            nosaturate_ranges=[],
            filter_names=filter_names,
        )
