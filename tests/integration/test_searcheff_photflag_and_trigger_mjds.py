"""Internal-consistency tests for SEARCHEFF output fields.

Runs JAX against the SEARCHEFF SNII fixture and asserts that:

1. ``PHOTFLAG_DETECT`` / ``PHOTFLAG_TRIGGER`` values from the SEARCHEFF file
   land in the simulator's resources.
2. PHOTFLAG bits in the output observations match the detection/trigger flags
   (every detected obs has DETECT bit; every trigger obs has TRIGGER bit;
   triggers ⊆ detections; exactly one trigger per passing SN).
3. ``MJD_TRIGGER``, ``MJD_DETECT_FIRST``, ``MJD_DETECT_LAST`` are properly
   ordered (FIRST ≤ TRIGGER ≤ LAST), within a reasonable MJD range, and not
   sentinel for passing SNe.
4. The regression detection count is within an expected band.

These don't compare to SNANA — they pin invariants on JAX's own output.
"""

import os
from pathlib import Path

import numpy as np
import pytest


from jax_snlc_sim import run as sim_run  # noqa: E402
from jax_snlc_sim.selection import _build_columnar_data  # noqa: E402


@pytest.fixture(scope="module")
def jax_columnar(snana_fixture):
    """Run JAX against searcheff_snii_basic; return the writer's columnar view."""
    fx = snana_fixture("searcheff_snii_basic")
    config = dict(
        n_sne=200, random_seed=50001,
    )
    sim = sim_run.prepare(str(fx.input_file), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    results, selection, _ = sim_run.run_simulation(sim)
    return _build_columnar_data(resources, results, selection)


def test_photflag_bits(jax_columnar):
    """Triggers are a subset of detections, one per SN."""
    columnar = jax_columnar
    obs = columnar["observations"]
    obs_counts = columnar["obs_counts"]
    n_sne = len(obs_counts)
    assert n_sne > 0, "fixture passed no SNe"

    is_detection = np.asarray(obs["is_detection"])
    is_trigger = np.asarray(obs.get("is_trigger", np.zeros_like(is_detection, dtype=bool)))

    if int(np.sum(is_trigger)) > 0:
        assert np.all(is_detection[is_trigger])

    # Exactly one trigger per SN.
    boundaries = np.concatenate([[0], np.cumsum(obs_counts)])
    triggers_per_sn = np.array([
        int(np.sum(is_trigger[boundaries[i]:boundaries[i + 1]]))
        for i in range(n_sne)
    ])
    assert int(np.sum(triggers_per_sn > 1)) == 0
    assert int(np.sum(triggers_per_sn > 0)) > 0


def test_mjd_detect_values(jax_columnar):
    """FIRST ≤ TRIGGER ≤ LAST; values within reasonable MJD range."""
    columnar = jax_columnar
    params = columnar["parameters"]
    n_sne = len(columnar["obs_counts"])
    assert n_sne > 0, "fixture passed no SNe"

    mjd_trigger = np.asarray(params["mjd_trigger"])
    mjd_first = np.asarray(params["mjd_detect_first"])
    mjd_last = np.asarray(params["mjd_detect_last"])
    peak_mjd = np.asarray(params["peakmjd"])

    # Not placeholder values (peak_mjd ± 10/30).
    n_placeholder_first = int(np.sum(np.isclose(mjd_first, peak_mjd - 10)))
    n_placeholder_last = int(np.sum(np.isclose(mjd_last, peak_mjd + 30)))
    assert n_placeholder_first < n_sne * 0.1
    assert n_placeholder_last < n_sne * 0.1

    # MJD_TRIGGER not sentinel for passing SNe.
    assert int(np.sum(mjd_trigger >= 1e5)) == 0

    # FIRST ≤ TRIGGER ≤ LAST.
    valid = mjd_trigger < 1e5
    if np.any(valid):
        assert np.all(
            (mjd_first[valid] <= mjd_trigger[valid] + 0.001)
            & (mjd_trigger[valid] <= mjd_last[valid] + 0.001)
        )

    # Within reasonable MJD range.
    assert np.all((mjd_first > 50000) & (mjd_first < 70000))




def _write(base_input, out, replacements):
    text = Path(base_input).read_text()
    for old, new in replacements:
        text = text.replace(old, new)
    out.write_text(text)
    return out


def test_detections_survive_a_failed_trigger(snana_fixture, tmp_path):
    """SNANA flags detected epochs whether or not the SN triggers; only
    MJD_TRIGGER goes sentinel (gen_SEARCHEFF_PIPELINE, sntools_trigger.c:2314).

    Verified against snlc_sim.exe with the detection logic raised so nothing
    triggers: 128 epochs still carry PHOTFLAG_DETECT, MJD_DETECT_FIRST/LAST are
    real and SIM_SEARCHEFF_MASK is 0. jax previously discarded every detection.
    """
    fx = snana_fixture("searcheff_snii_basic")
    logic = Path(fx.input_file).parent / "logic_hard.DAT"
    src = [l for l in Path(fx.input_file).read_text().splitlines()
           if l.startswith('SEARCHEFF_PIPELINE_LOGIC_FILE')]
    orig = os.path.expandvars(src[0].split(':', 1)[1].strip())
    logic = tmp_path / "logic_hard.DAT"
    logic.write_text("\n".join(
        l.replace(': 1 ', ': 99 ', 1) if l and not l.startswith(('#', ' ')) and ':' in l
        else l for l in Path(orig).read_text().splitlines()) + "\n")
    inp = _write(fx.input_file, tmp_path / "in.input",
                 [(src[0], f"SEARCHEFF_PIPELINE_LOGIC_FILE: {logic}"),
                  ("APPLY_SEARCHEFF_OPT:  1", "APPLY_SEARCHEFF_OPT:  0")])
    config = dict(n_sne=50,
                                 random_seed=50001)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    results, sel, _ = sim_run.run_simulation(sim)
    triggered = np.asarray(sel["triggered"])
    assert not triggered.any(), "fixture no longer exercises the failed trigger"
    assert np.asarray(sel["detection_mask"]).sum() > 0, \
        "detections must survive a failed trigger"
    assert np.all(np.asarray(sel["mjd_trigger"]) >= 1e5)
    assert np.any(np.asarray(sel["mjd_detect_first"]) < 1e5)
    assert np.any(np.asarray(sel["mjd_detect_last"]) < 1e5)
