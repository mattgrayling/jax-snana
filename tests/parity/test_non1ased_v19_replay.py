"""NON1ASED V19 SIM_MAGOBS parity: JAX replay vs bundled SNANA PHOT.FITS."""

import collections
from pathlib import Path

import numpy as np
import pytest
from _replay_helpers import slice_replay
from astropy.io import fits
from replay_snana_dump import build_override_parameters

from jax_snlc_core.snana_files import load_snana_dump
from jax_snlc_sim import run as sim_run
from jax_snlc_sim.output import save_snana_outputs


def _load_phot(head_path: Path, phot_path: Path):
    with fits.open(head_path) as h, fits.open(phot_path) as p:
        head, phot = h[1].data, p[1].data
        out = {}
        for row in head:
            snid = int(row["SNID"])
            pmin, pmax = int(row["PTROBS_MIN"]), int(row["PTROBS_MAX"])
            for i in range(pmin - 1, pmax):
                pr = phot[i]
                mjd = float(pr["MJD"])
                if mjd < 0:
                    continue
                band = str(pr["BAND"]).strip()
                out[(snid, round(mjd, 2), band)] = (
                    float(pr["SIM_MAGOBS"]),
                    float(pr["FLUXCAL"]),
                    float(pr["FLUXCALERR"]),
                )
    return out


def test_non1ased_v19_simmagobs_parity(snana_fixture, tmp_path):
    """JAX replay reproduces SNANA's per-(SNID, MJD, BAND) SIM_MAGOBS for NON1ASED V19."""
    fx = snana_fixture("non1ased_v19_replay")

    dump = load_snana_dump(str(fx.dump), dedup_cid=True)
    with fits.open(fx.head_fits) as hdul:
        head_cols = {n: hdul[1].data[n] for n in hdul[1].data.names}

    # 300 events; guard below is ~80% of the resulting yield so a silent
    # loss of matched epochs fails instead of shrinking the sample
    dump_cols, head_cols = slice_replay(dump.columns, head_cols, 300)
    config = dict(n_sne=len(dump_cols['CID']))
    sim = sim_run.prepare(str(fx.input_file), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    override = build_override_parameters(dump_cols, simulation, resources,
                                         head_columns=head_cols)

    results, sel, spectra_cache = sim_run.run_simulation(sim, override_parameters=override)
    save_snana_outputs(sim, results, "N19", sel, spectra_by_sn=spectra_cache, output_dir=str(tmp_path))

    snana_phot = _load_phot(fx.head_fits, fx.phot_fits)
    jax_dir = tmp_path / "N19"
    jax_phot = _load_phot(
        next(jax_dir.glob("N19*HEAD*")),
        next(jax_dir.glob("N19*PHOT*")),
    )

    diffs = []
    common = set(snana_phot) & set(jax_phot)
    assert len(common) > 0, "no overlapping (SNID, MJD, BAND) keys"
    for key in common:
        sn_mag, sn_flux, sn_err = snana_phot[key]
        jx_mag, _, _ = jax_phot[key]
        if sn_err <= 0 or sn_flux / sn_err < 5.0:
            continue
        if sn_mag >= 50.0 or jx_mag >= 50.0:
            continue
        diffs.append(jx_mag - sn_mag)

    assert len(diffs) > 550, f"too few S/N≥5 epochs ({len(diffs)})"
    diffs = np.asarray(diffs)
    abs_d = np.abs(diffs)
    median = float(np.median(abs_d))
    p95 = float(np.percentile(abs_d, 95))

    # NON1ASED reads pre-shifted SED templates and integrates against the same
    # filters JAX uses; agreement should be sub-millimag for converged epochs.
    assert median < 0.005, f"median |Δmag| = {median:.4f} > 0.005"
    assert p95 < 0.05, f"p95 |Δmag| = {p95:.4f} > 0.05"


def _two_template_sim(tmp_path, w834, w835, n_sne=100):
    """The NON1ASED fixture rewritten with two NON1A templates."""
    src = Path("tests/fixtures/snana/non1ased_v19_replay/input.input").read_text()
    src = src.replace("NGENTOT_LC:  1500", f"NGENTOT_LC:  {n_sne}")
    src = src.replace(
        "NON1A: 834 1.0 0.00 0.00 21",
        f"NON1A: 834 {w834} 0.00 0.00 21\nNON1A: 835 {w835} 0.00 0.00 21")
    inp = tmp_path / f"two_template_{w834}_{w835}.input"
    inp.write_text(src)
    config = dict(n_sne=n_sne,
                                 random_seed=1)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    results, _, _ = sim_run.run_simulation(sim)
    idx = np.asarray(results["parameters"]["non1a_index"])
    return simulation, resources, idx


def test_non1ased_template_budgets_match_snana(tmp_path):
    """Templates get integer event budgets and contiguous index blocks.

    prep_NON1ASED (genmag_NON1ASED.c:352-418) converts NON1A weights into whole
    event counts, and pick_NON1ASED (snlc_sim.c:14378) then assigns a template
    by generation index -- not by a random draw against the weights.

    Live stock SNANA with weights 3:1 and NGENTOT_LC=100 reports
    "Will generate 75 ... (MXGEN=  75)" then "25 ... (MXGEN= 100)", and its
    DUMP gives INDEX 834 for CID 1-75 and 835 for CID 76-100.
    """
    simulation, resources, idx = _two_template_sim(tmp_path, 3.0, 1.0)

    non1a = simulation.model._structured_config['non1a_config']
    assert list(non1a["index_per_template"]) == [834, 835]
    assert list(resources.model._mxgen_per_template) == [75, 100]

    # contiguous blocks, not a weighted random scatter
    assert set(idx[:75].tolist()) == {834}
    assert set(idx[75:].tolist()) == {835}


def test_non1ased_last_template_absorbs_rounding_remainder(tmp_path):
    """Per-template budgets are rounded, so they need not sum to NGENTOT.

    prep_NON1ASED gives the last template `NGENTOT - NGENSUM` rather than its
    own rounded share. Weights 0.5:99.5 over 100 events round to 1 and 100,
    which overshoots by one; stock SNANA reports "Will generate 1 (MXGEN= 1)"
    then "99 (MXGEN= 100)".

    Asserted on MXGEN rather than on the assignment: a too-large final budget
    is invisible in the drawn templates, because the trailing events clip onto
    the last template either way.
    """
    _, resources, _ = _two_template_sim(tmp_path, 0.5, 99.5)
    assert list(resources.model._mxgen_per_template) == [1, 100]


def _accept_n_sim(tmp_path, target=400):
    """Two templates 2 mag apart in MAGOFF, weights 3:1, NGEN_LC + trigger on."""
    src = Path("tests/fixtures/snana/non1ased_v19_replay/input.input").read_text()
    src = src.replace("NGENTOT_LC:  1500", f"NGEN_LC:  {target}")
    src = src.replace("APPLY_SEARCHEFF_OPT:  0", "APPLY_SEARCHEFF_OPT:  1")
    src = src.replace(
        "NON1A: 834 1.0 0.00 0.00 21",
        "NON1A: 834 3.0 0.00 0.00 21\nNON1A: 835 1.0 2.00 0.00 21")
    inp = tmp_path / "accept_n.input"
    inp.write_text(src)
    config = dict(n_sne=target,
                                 random_seed=3)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    results, sel, _ = sim_run.run_simulation(sim)
    passing = np.asarray(sel["passing_mask"]).astype(bool)
    return results, passing


@pytest.fixture(scope="module")
def accept_n_run(tmp_path_factory):
    """One NGEN_LC + trigger run, shared by the three acceptance tests."""
    return _accept_n_sim(tmp_path_factory.mktemp("accept_n"))


def test_ngen_lc_delivers_requested_template_mix(accept_n_run):
    """NGEN_LC accepts exactly the requested count and per-template split.

    Stock SNANA on this config: NGENLC_WRITE=400, 300/100 by template.
    """
    results, passing = accept_n_run
    idx = np.asarray(results["parameters"]["non1a_index"])

    assert int(passing.sum()) == 400
    assert dict(collections.Counter(idx[passing].tolist())) == {834: 300, 835: 100}


def test_ngen_lc_cids_follow_accepted_slot(accept_n_run):
    """CID is SNANA's `CIDOFF + ilc`, the accepted slot being filled."""
    results, passing = accept_n_run
    cid = np.asarray(results["parameters"]["cid"])

    np.testing.assert_array_equal(np.sort(cid[passing]), np.arange(1, 401))
    assert cid.max() == 400
    assert len(cid) > len(set(cid.tolist())), "retried attempts share a CID"


def _three_template_input(tmp_path, name, subs):
    src = Path("tests/fixtures/snana/non1ased_v19_replay/input.input").read_text()
    src = src.replace("NON1A: 834 1.0 0.00 0.00 21",
                      "NON1A: 834 1.0 0.00 0.00 21\n"
                      "NON1A: 835 1.0 0.00 0.00 21\n"
                      "NON1A: 836 1.0 0.00 0.00 21")
    for old, new in subs:
        src = src.replace(old, new)
    inp = tmp_path / name
    inp.write_text(src)
    return inp


def test_three_template_budgets_match_snana(tmp_path):
    """Equal weights over 100 give 33/33/34; the last takes the remainder.

    Stock SNANA: "Will generate 33 (MXGEN= 33)", "33 (MXGEN= 66)",
    "34 (MXGEN= 100)".
    """
    inp = _three_template_input(
        tmp_path, "three.input", [("NGENTOT_LC:  1500", "NGENTOT_LC:  100")])
    config = dict(n_sne=100,
                                 random_seed=1)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    assert list(resources.model._mxgen_per_template) == [33, 66, 100]


def test_ngen_lc_fills_every_quota_with_three_classes(tmp_path):
    """Three classes, batch_size below the class count, all quotas filled."""
    inp = _three_template_input(
        tmp_path, "three_accept.input",
        [("NGENTOT_LC:  1500", "NGEN_LC:  12"),
         ("APPLY_SEARCHEFF_OPT:  0", "APPLY_SEARCHEFF_OPT:  1")])
    config = dict(n_sne=12,
                                 random_seed=3, batch_size=4)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    results, sel, _ = sim_run.run_simulation(sim)
    passing = np.asarray(sel["passing_mask"]).astype(bool)
    idx = np.asarray(results["parameters"]["non1a_index"])

    assert int(passing.sum()) == 12
    assert dict(collections.Counter(idx[passing].tolist())) == {834: 4, 835: 4, 836: 4}


def test_ngen_lc_fills_a_class_whose_share_rounds_to_zero(tmp_path):
    """A tiny quota beside a large one is still filled exactly.

    Weights 19:1 over 20 give quotas 19 and 1 -- stock SNANA reports
    "Will generate 19 (MXGEN= 19)" then "1 (MXGEN= 20)".
    """
    src = Path("tests/fixtures/snana/non1ased_v19_replay/input.input").read_text()
    src = src.replace("NON1A: 834 1.0 0.00 0.00 21",
                      "NON1A: 834 19.0 0.00 0.00 21\nNON1A: 835 1.0 0.00 0.00 21")
    src = src.replace("NGENTOT_LC:  1500", "NGEN_LC:  20")
    src = src.replace("APPLY_SEARCHEFF_OPT:  0", "APPLY_SEARCHEFF_OPT:  1")
    inp = tmp_path / "lopsided.input"
    inp.write_text(src)
    config = dict(n_sne=20,
                                 random_seed=3, batch_size=10)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim

    assert list(resources.model._mxgen_per_template) == [19, 20]

    results, sel, _ = sim_run.run_simulation(sim)
    passing = np.asarray(sel["passing_mask"]).astype(bool)
    idx = np.asarray(results["parameters"]["non1a_index"])
    assert dict(collections.Counter(idx[passing].tolist())) == {834: 19, 835: 1}


def test_non1ased_sim_av_rv_are_the_drawn_dust(snana_fixture, tmp_path):
    """SNANA writes SIM_AV and SIM_RV for every model once host dust is drawn
    (gen_modelPar_dust runs whenever DOGEN_AV is set, snlc_sim.c:14325, and
    wr_snfitsio always writes SNDATA.SIM_AV, sntools_dataformat_fits.c:2152).
    The fixture dumps AV and RV, so the replay carries SNANA's own draws and
    the HEAD columns compare exactly, per event."""
    fx = snana_fixture("non1ased_hostdust")

    dump = load_snana_dump(str(fx.dump), dedup_cid=True)
    with fits.open(fx.head_fits) as hdul:
        head_cols = {n: hdul[1].data[n] for n in hdul[1].data.names}
    dump_cols, head_cols = slice_replay(dump.columns, head_cols, 300)
    config = dict(n_sne=len(dump_cols['CID']))
    sim = sim_run.prepare(str(fx.input_file), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    override = build_override_parameters(dump_cols, simulation, resources,
                                         head_columns=head_cols)
    results, sel, spectra_cache = sim_run.run_simulation(sim, override_parameters=override)
    save_snana_outputs(sim, results, "NAV", sel, spectra_by_sn=spectra_cache, output_dir=str(tmp_path))

    snana = fits.getdata(fx.head_fits, 1)
    jax = fits.getdata(next((tmp_path / "NAV").glob("NAV*HEAD*")), 1)
    snana_by_snid = {int(r["SNID"]): (float(r["SIM_AV"]), float(r["SIM_RV"]))
                     for r in snana}
    common = [int(s) for s in jax["SNID"] if int(s) in snana_by_snid]
    assert len(common) > 200, f"too few matched events ({len(common)})"
    jax_by_snid = {int(r["SNID"]): (float(r["SIM_AV"]), float(r["SIM_RV"]))
                   for r in jax}
    # the replay reads the DUMP's six-figure text, the HEAD holds the float
    for col, k in (("SIM_AV", 0), ("SIM_RV", 1)):
        np.testing.assert_allclose(
            [jax_by_snid[s][k] for s in common],
            [snana_by_snid[s][k] for s in common],
            rtol=0.0, atol=1e-5, err_msg=col)
