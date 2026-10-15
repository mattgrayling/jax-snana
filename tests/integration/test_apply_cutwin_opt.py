"""APPLY_CUTWIN_OPT mode-0/1/3 integration tests.

SNANA defines three APPLY_CUTWIN_OPT modes:
  * 0: track CUTMASK but do not gate selection on it.
  * 1: gate selection on all CUTWIN cuts (default in production).
  * 3: gate selection as mode 1, and additionally route CUTWIN-rejected
       events into the SIMGEN dump (snlc_sim.c:16925-16926).
"""

from pathlib import Path

import numpy as np
import pytest


from jax_snlc_sim import run as sim_run  # noqa: E402


def _patched_input(src: Path, dst: Path, *, opt: int, extra: str = "") -> Path:
    text = src.read_text()
    if "APPLY_CUTWIN_OPT:" in text:
        # Replace existing line.
        new = []
        for line in text.splitlines():
            if line.startswith("APPLY_CUTWIN_OPT"):
                new.append(f"APPLY_CUTWIN_OPT:  {opt}")
            else:
                new.append(line)
        text = "\n".join(new) + "\n"
    else:
        text += f"\nAPPLY_CUTWIN_OPT:  {opt}\n"
    dst.write_text(text + extra)
    return dst


def _run(input_file: Path, n_sne: int = 200, seed: int = 122459):
    config = dict(
        n_sne=n_sne, random_seed=seed,
    )
    sim = sim_run.prepare(str(input_file), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    results, selection, spectra_cache = sim_run.run_simulation(sim)
    return (simulation, resources, config, sim), results, selection, spectra_cache


@pytest.fixture(scope="module")
def cutwin_opt_results(snana_fixture, tmp_path_factory):
    """Run the simulator once each at OPT=0 and OPT=1 — both opt0 tests
    below share the OPT=0 result and ``test_apply_cutwin_opt0_does_not_gate_on_cutwin``
    additionally consumes OPT=1."""
    tmp = tmp_path_factory.mktemp("cutwin_opt")
    fx = snana_fixture("cutwin_parity")
    inp_0 = _patched_input(fx.input_file, tmp / "opt0.input", opt=0)
    inp_1 = _patched_input(fx.input_file, tmp / "opt1.input", opt=1)
    return {
        0: _run(inp_0, n_sne=200, seed=4242),
        1: _run(inp_1, n_sne=200, seed=4242),
    }


def test_apply_cutwin_opt0_does_not_gate_on_cutwin(cutwin_opt_results):
    """With OPT=0, CUTWIN-failures still appear in the passing set —
    the cuts are tracked in CUTMASK but don't reduce ``n_passed``.
    """
    _, _, sel0, _ = cutwin_opt_results[0]
    _, _, sel1, _ = cutwin_opt_results[1]

    n_pass_opt0 = int(sel0["n_passed"])
    n_pass_opt1 = int(sel1["n_passed"])
    assert n_pass_opt0 >= n_pass_opt1, (
        f"OPT=0 should not reduce passes vs OPT=1; got {n_pass_opt0} vs {n_pass_opt1}"
    )
    assert n_pass_opt0 > n_pass_opt1, (
        f"CUTWIN cuts in this fixture aren't tight enough to demonstrate "
        f"OPT=0 vs OPT=1: both produced {n_pass_opt0} passes"
    )


def test_apply_cutwin_opt0_cutmask_still_records_failures(cutwin_opt_results):
    """Even under OPT=0, the CUTMASK column should still record per-bit
    CUTWIN failures (they're tracked, just not gated on)."""
    _, _, sel, _ = cutwin_opt_results[0]
    cutmask = np.asarray(sel["cutmask"]).astype(int)
    # CUTMASK is built per-SN; a clear bit indicates a cut.
    # The all-pass value is 4095 (all 12 bits set = no failures).
    n_with_failure = int((cutmask != 4095).sum())
    assert n_with_failure > 0, (
        "CUTWIN failures should still be tracked in CUTMASK under OPT=0; "
        "all SNe show CUTMASK=4095 (no failures recorded)"
    )


def test_apply_cutwin_opt3_gates_exactly_as_opt1(
        snana_fixture, tmp_path, cutwin_opt_results):
    """OPT=3 accepts the same events as OPT=1; only the dump differs.

    `doReject_SIMGEN_DUMP` (snlc_sim.c:16925-16926) turns OPT=3 into a dump
    flag on the CUTWIN reject stage, so the accepted set is untouched.
    `test_opt3_dump_rows_carry_snrmax` covers the dump side.
    """
    fx = snana_fixture("cutwin_parity")
    inp_3 = _patched_input(fx.input_file, tmp_path / "opt3.input", opt=3)
    _, _, sel3, _ = _run(inp_3, n_sne=200, seed=4242)
    _, _, sel0, _ = cutwin_opt_results[0]
    _, _, sel1, _ = cutwin_opt_results[1]
    assert int(sel0["n_passed"]) > int(sel1["n_passed"]), (
        "cutwin_parity rejects nothing on CUTWIN; the comparison would be vacuous"
    )
    np.testing.assert_array_equal(
        np.asarray(sel3["passing_mask"]), np.asarray(sel1["passing_mask"]),
        err_msg="OPT=3 accepted a different set of events from OPT=1")


# --- CUTWIN_TGAPMAX / T0GAPMAX / PEAKMAG (snlc_sim.c:24379-24417, :24558-24637) ---
#
# Each cut must change the accept count: before these were implemented every one
# of them parsed, applied nothing, and warned nothing.

@pytest.fixture(scope="module")
def gap_baseline(snana_fixture, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cutwin_new")
    fx = snana_fixture("cutwin_parity")
    inp = _patched_input(fx.input_file, tmp / "base.input", opt=1)
    return fx, tmp, _run(inp, n_sne=200, seed=4242)[2]


@pytest.mark.parametrize("key,window", [
    ("CUTWIN_TGAPMAX", "0 0.001"),
    ("CUTWIN_T0GAPMAX", "0 0.001"),
    ("CUTWIN_PEAKMAG", "0 1"),
    ("CUTWIN_PEAKMAG_ALL", "0 1"),
])
def test_new_cutwin_keys_actually_cut(gap_baseline, tmp_path, key, window):
    """An impossible window must reject events the baseline accepts."""
    fx, _, base = gap_baseline
    inp = _patched_input(fx.input_file, tmp_path / f"{key}.input",
                         opt=1, extra=f"\n{key}: {window}\n")
    _, _, sel, _ = _run(inp, n_sne=200, seed=4242)
    assert int(base["n_passed"]) > 0
    assert int(sel["n_passed"]) < int(base["n_passed"]), (
        f"{key}: {window} changed nothing — the cut is inert")


def test_tgapmax_sentinel_fails_a_bounded_window(gap_baseline, tmp_path):
    """SNANA leaves TGAPMAX at -99999 when under two epochs qualify
    (snlc_sim.c:9563), so such events fail any window with a finite lower
    bound. EPCUTWIN_LAMREST starves the epoch list to force that state.
    """
    fx, _, _ = gap_baseline
    extra = "\nEPCUTWIN_LAMREST: 5000 5600\nCUTWIN_TGAPMAX: 0 20\n"
    inp = _patched_input(fx.input_file, tmp_path / "sent.input", opt=1, extra=extra)
    _, _, sel, _ = _run(inp, n_sne=200, seed=4242)
    assert int(sel["n_passed"]) == 0


def test_opt3_dump_rows_carry_snrmax(snana_fixture, tmp_path):
    """CUTWIN-rejected dump rows keep their real SNRMAX: gen_cutwin fills
    SNRMAX_SORTED before the CUTWIN decision (snlc_sim.c:24300-24329), so
    only events rejected before gen_cutwin keep the -9 init (:9615)."""
    import re
    fx = snana_fixture("cutwin_parity")
    text = re.sub(r"SIMGEN_DUMPALL:.*",
                  "SIMGEN_DUMP: 4 CID SNRMAX CUTMASK FLAG_ACCEPT",
                  fx.input_file.read_text())
    src = tmp_path / "opt3snr_src.input"
    src.write_text(text)
    inp = _patched_input(src, tmp_path / "opt3snr.input", opt=3)
    (simulation, resources, config, sim), results, sel, spectra_cache = _run(
        inp, n_sne=200, seed=4242)
    sim_run.save_snana_outputs(sim, results, "O3", sel, spectra_cache=spectra_cache, output_dir=str(tmp_path))
    lines = (tmp_path / "O3" / "O3.DUMP").read_text().splitlines()
    varnames = next(l.split()[1:] for l in lines if l.startswith("VARNAMES:"))
    rows = [l.split()[1:] for l in lines if l.startswith("SN:")]
    snr = np.array([float(r[varnames.index("SNRMAX")]) for r in rows])
    acc = np.array([int(r[varnames.index("FLAG_ACCEPT")]) for r in rows])
    assert (acc == 0).any(), "no CUTWIN rejects in the dump; fixture drifted"
    assert (snr[acc == 0] > -9.0).all()
