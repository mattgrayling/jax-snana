"""End-to-end PHOTPROB integration test.

Runs the JAX simulator with a SEARCHEFF pipeline that has a PHOTPROB map
attached, and asserts the output PHOTPROB column is populated and within
[0, 1] for evaluated obs. Internal consistency only — no SNANA comparison.
"""

import os
from pathlib import Path

import numpy as np


from jax_snlc_sim import run as sim_run  # noqa: E402
from jax_snlc_sim.selection import _build_columnar_data  # noqa: E402


def test_photprob_pipeline_integration(snana_fixture, tmp_path):
    """Run JAX with PHOTPROB map; verify PHOTPROB column populated."""
    fx = snana_fixture("searcheff_snii_basic")
    # Build a variant input that swaps in the PHOTPROB pipeline DAT.
    src = fx.input_file.read_text()
    photprob_input = tmp_path / "photprob.input"
    photprob_input.write_text(
        src.replace(
            "$JAX_TEST_FIXTURE_ROOT/searcheff/SEARCHEFF_PIPELINE_LSST.DAT",
            "$JAX_TEST_FIXTURE_ROOT/searcheff/SEARCHEFF_PIPELINE_PHOTPROB_TEST.DAT",
        )
    )
    config = dict(n_sne=400, random_seed=50001)
    sim = sim_run.prepare(str(photprob_input), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    results, sel, _ = sim_run.run_simulation(sim)
    assert int(sel["n_passed"]) > 0, "photprob-patched fixture passed no SNe"

    columnar = _build_columnar_data(resources, results, sel)
    obs = columnar["observations"]
    pp = obs.get("photprob")
    snr = obs.get("snr")
    assert pp is not None, "photprob column missing from observations"
    assert snr is not None, "snr column missing from observations"

    pp = np.asarray(pp)
    snr = np.asarray(snr)
    evaluated = pp > -1
    n_evaluated = int(np.sum(evaluated))
    assert n_evaluated > 50, f"too few PHOTPROB-evaluated obs ({n_evaluated})"
    vals = pp[evaluated]
    assert np.all(vals >= 0) and np.all(vals <= 1)

    # The map's PHOTPROB_WGTLIST is sampled per observation, so the
    # evaluated values must span several of its four weight bins rather
    # than collapsing to one. The SNR dependence itself is not testable
    # here: this fixture yields ~50 evaluated epochs, all below the map's
    # second SNR row, and the per-epoch draw is noisy enough that the rank
    # correlation is not significant (r~0.2, p~0.1). It is established
    # instead by test_searcheff_photprob.py::test_snr_dependent, which
    # contrasts 5000 draws at SNR 2 against 5000 at SNR 30.
    assert len(np.unique(vals)) > 2, (
        f"PHOTPROB collapsed to {np.unique(vals)}; the weight list is not "
        f"being sampled")

    params = columnar["parameters"]
    assert "specid_pass" in params
    assert "sim_searcheff_mask" in params


def test_photprob_without_detection_curves_stays_null(snana_fixture,
                                                      tmp_path):
    """PHOTPROB is drawn only inside the detection pass, which a pipeline
    file with no curves never runs, so every PHOTPROB stays -9
    (gen_SEARCHEFF, sntools_trigger.c:2041; :2211)."""
    fx = snana_fixture("searcheff_snii_basic")
    src = Path(os.path.expandvars(
        "$JAX_TEST_FIXTURE_ROOT/searcheff/SEARCHEFF_PIPELINE_PHOTPROB_TEST.DAT"))
    kept, in_detect = [], False
    for line in src.read_text().splitlines():
        if line.startswith("MAPNAME_DETECT:"):
            in_detect = True
        elif in_detect and line.startswith("ENDMAP:"):
            in_detect = False
            continue
        if not in_detect:
            kept.append(line)
    photprob_only = tmp_path / "PHOTPROB_ONLY.DAT"
    photprob_only.write_text("\n".join(kept) + "\n")

    photprob_input = tmp_path / "photprob_only.input"
    photprob_input.write_text(fx.input_file.read_text().replace(
        "$JAX_TEST_FIXTURE_ROOT/searcheff/SEARCHEFF_PIPELINE_LSST.DAT",
        str(photprob_only)))
    config = dict(n_sne=50, random_seed=50001)
    sim = sim_run.prepare(str(photprob_input), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    assert not resources.searcheff_curves
    results, sel, _ = sim_run.run_simulation(sim)
    assert int(sel["n_passed"]) > 0

    columnar = _build_columnar_data(resources, results, sel)
    pp = np.asarray(columnar["observations"]["photprob"])
    assert pp.size > 0
    assert np.all(pp == -9.0)


def test_text_output_writes_photprob_only_with_photprob_maps(snana_fixture,
                                                             tmp_path):
    """TEXT writes a PHOTPROB column, after PHOTFLAG, only when the pipeline
    file holds PHOTPROB maps (WRFLAG_PHOTPROB, snlc_sim.c:25364;
    sntools_dataformat_text.c:959-960)."""
    fx = snana_fixture("searcheff_snii_basic")
    src = fx.input_file.read_text().replace("FORMAT_MASK:  32", "FORMAT_MASK:  2")

    def varlist(text, tag):
        inp = tmp_path / f"{tag}.input"
        inp.write_text(text)
        config = dict(n_sne=400, random_seed=50001)
        sim = sim_run.prepare(str(inp), **config)
        simulation = None if sim is None else sim.config
        resources = sim
        results, sel, spectra_cache = sim_run.run_simulation(sim)
        sim_run.save_snana_outputs(sim, results, "T", sel, spectra_cache=spectra_cache, output_dir=str(tmp_path / tag))
        dat = sorted((tmp_path / tag).rglob("*.DAT"))
        assert dat, "no TEXT light curves written"
        return next(line.split()[1:] for line in dat[0].read_text().splitlines()
                    if line.startswith("VARLIST:"))

    with_maps = varlist(src.replace(
        "$JAX_TEST_FIXTURE_ROOT/searcheff/SEARCHEFF_PIPELINE_LSST.DAT",
        "$JAX_TEST_FIXTURE_ROOT/searcheff/SEARCHEFF_PIPELINE_PHOTPROB_TEST.DAT"),
        "maps")
    assert with_maps[with_maps.index("PHOTFLAG") + 1] == "PHOTPROB"
    assert "PHOTPROB" not in varlist(src, "nomaps")
