"""CUTWIN_MWEBV, the cut on the SN sightline's Milky Way E(B-V), which
selection applies as SNANA's gen_cutwin does (snlc_sim.c:24454-24456), and
the generic CUTWIN_<VAR> family.
"""

from pathlib import Path

import numpy as np
import pytest

from jax_snlc_core.utils.error_handling import SNANAError
from jax_snlc_sim import run as sim_run  # noqa: E402


def test_cutwin_mwebv_partial_pass_when_one_field_in_range(snana_fixture):
    """With OPT_MWEBV=1 reading per-pointing MWEBV from the bundled DUSTY
    SIMLIB (LIBID 0 = 0.02, LIBID 1 = 0.13), the CUTWIN_MWEBV range
    [0.0, 0.10] only admits the lower-MWEBV field, so ~half the SNe pass
    the cut and ~half fail it."""
    fx = snana_fixture("cutwin_mwebv")
    config = dict(n_sne=200, random_seed=122459)
    sim = sim_run.prepare(str(fx.input_file), **config)
    results, sel, _ = sim_run.run_simulation(sim)
    n_passed = int(sel["n_passed"])
    # Exactly one of two LIBIDs has MWEBV inside [0, 0.10]; passing fraction
    # should be ~50% (allow some random scatter in field assignment).
    frac = n_passed / 200
    assert 0.30 < frac < 0.70, f"unexpected pass rate {frac:.2f}"


def test_unmatched_generic_cutwin_aborts_before_photometry(snana_fixture, tmp_path):
    """CUTWIN_<VAR> takes any name, so a typo names a quantity nothing computes.

    The cut runs during selection, after every event exists, so the check is
    hoisted to just after parameter sampling — before the light curves that
    would otherwise be generated and thrown away.
    """
    fx = snana_fixture("cutwin_mwebv")
    patched = tmp_path / "typo.input"
    patched.write_text(fx.input_file.read_text() + "\nCUTWIN_REDSHIFT_FINEL: 0 1\n")
    config = dict(n_sne=100, random_seed=122459)
    sim = sim_run.prepare(str(patched), **config)
    with pytest.raises(SNANAError) as exc:
        sim_run.run_simulation(sim)
    assert "redshift_finel" in str(exc.value)
    # The reporting function pins where in the run the abort happened: the
    # selection-time cut would name apply_range_cuts instead.
    assert str(exc.value).startswith("check_generic_cut_names:")


def test_generic_cutwin_with_words_for_values_aborts_naming_the_key(snana_fixture, tmp_path):
    """A misspelt cut key whose values are not two numbers (a CUTWIN_SNRMAX
    typo carries a filter list) aborts naming the key, not with a bare
    ValueError from the number parse."""
    fx = snana_fixture("cutwin_mwebv")
    patched = tmp_path / "typo.input"
    patched.write_text(fx.input_file.read_text() + "\nCUTWIN_SNRMEX: 5 griz 2 -20 80\n")
    with pytest.raises(SNANAError, match="CUTWIN_SNRMEX expects two numbers"):
        sim_run.prepare(str(patched), **dict(n_sne=10, random_seed=1))


def test_engine_derived_quantities_are_all_cuttable(snana_fixture, tmp_path):
    """The keyword-only quantities of `range_cut_quantities` resolve end-to-end.

    Both the early name check and the engine's cut go through that one
    function, so the names cannot drift apart; this pins that each is actually
    supplied with a value by the time the cut runs.
    """
    import inspect
    from jax_snlc_sim.survey.cutwin import range_cut_quantities

    derived = [name for name, p in
               inspect.signature(range_cut_quantities).parameters.items()
               if p.kind is inspect.Parameter.KEYWORD_ONLY]
    fx = snana_fixture("cutwin_mwebv")
    wide = "".join(f"CUTWIN_{name.upper()}: -999 999\n" for name in derived)
    # The fixture already sets CUTWIN_MWEBV; replace it rather than repeat it.
    patched = tmp_path / "derived.input"
    patched.write_text(
        fx.input_file.read_text().replace("CUTWIN_MWEBV: 0.0 0.10", "") + "\n" + wide)
    config = dict(n_sne=50, random_seed=122459)
    sim = sim_run.prepare(str(patched), **config)
    results, sel, _ = sim_run.run_simulation(sim)
    assert int(sel["n_passed"]) > 0
