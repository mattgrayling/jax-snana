"""GENPDF + WGTMAP host-first integration: SN parameters drawn from a
GENPDF conditional on HOSTLIB-resident LOGMASS, with WGTMAP weighting
the HOSTLIB selection itself.

Exercises:
- the conditional-GENPDF branch of ``_sample_population_parameter`` in
  ``sampler.py`` (the one gated on ``genpdf.cond_varnames``)
- the WGTMAP-driven host selection path
- HOSTLIB selection *before* SN parameter sampling, SNANA's order, so the
  conditional GENPDF can read host LOGMASS.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import open_text
from jax_snlc_sim import run as sim_run
from jax_snlc_sim.output import save_snana_outputs


def _write_logmass_only_wgtmap(out: Path) -> None:
    """A WGTMAP with LOGMASS as the only conditioning variable (no SN vars).

    When WGTMAP only depends on HOSTLIB columns, host selection happens
    before SN parameter sampling, which is the ordering conditional GENPDFs
    (e.g. SALT2c|LOGMASS) require.
    Mass-favouring shape — high-mass galaxies get higher weight.
    """
    rows = [
        "VARNAMES_WGTMAP: LOGMASS WGT SNMAGSHIFT",
    ]
    for lm in (8.0, 9.0, 10.0, 11.0, 12.0):
        # weight grows steeply with LOGMASS (mass-step proxy)
        wgt = float(np.exp(0.6 * (lm - 9.0)))
        rows.append(f"WGT: {lm:.2f} {wgt:.4f} 0.0")
    out.write_text("\n".join(rows) + "\n")


def _write_genpdf_c_logmass(out: Path) -> None:
    """A 2-D GENPDF: SALT2c distribution shifts with host LOGMASS.

    Below LOGMASS=10: c peaks at -0.05 (bluer for low-mass).
    Above LOGMASS=10: c peaks at +0.05 (redder for high-mass).
    """
    rows = []
    for lm in (8.0, 9.0, 10.0, 11.0, 12.0):
        for c in (-0.3, -0.15, -0.05, 0.0, 0.05, 0.15, 0.3):
            # Gaussian centred at +0.10 * (lm - 10) with sigma 0.07 — a clean
            # mass-dependent colour distribution.
            mu = 0.10 * (lm - 10.0)
            prob = float(np.exp(-0.5 * ((c - mu) / 0.07) ** 2))
            rows.append(f"PDF: {c:+.3f} {lm:.1f} {prob:.6f}")
    body = "\n".join([
        "VARNAMES: SALT2c LOGMASS PROB",
        *rows,
    ])
    out.write_text(body + "\n")


def _write_input(out: Path, *, hostlib: Path, wgtmap: Path, genpdf: Path,
                  mskopt: int = 258, z_range: tuple[float, float] = (0.05, 0.40),
                  storepar: str = "LOGMASS,LOG_SFR,LOGsSFR") -> None:
    hostlib = hostlib or "$JAX_TEST_FIXTURE_ROOT/hostlib/JAX_TEST.HOSTLIB"
    out.write_text(f"""\
GENVERSION:  JAX_TEST_GENPDF_WGTMAP
NGENTOT_LC:  150
FORMAT_MASK:  32
GENTYPE:  1
SNTYPES_Ia:  1  101
CIDRAN_MIN:  0
CIDRAN_MAX:  3000000
GENMODEL:  $JAX_TEST_FIXTURE_ROOT/models/SALT3.K21
# x1: standard Gaussian — no GENPDF coupling for this param.
GENPEAK_SALT2x1:    0.0
GENRANGE_SALT2x1:  -2.0  2.0
GENSIGMA_SALT2x1:   1.0  1.0
# c: comes from GENPDF (file overrides Gaussian).
GENPEAK_SALT2c:     0.0
GENRANGE_SALT2c:   -0.3  0.3
GENSIGMA_SALT2c:    0.05  0.05
GENPEAK_SALT2ALPHA: 0.14
GENPEAK_SALT2BETA:  3.1
OMEGA_MATTER:  0.3150
OMEGA_LAMBDA:  0.6850
w0_LAMBDA:  -1.0000
wa_LAMBDA:  0
GENFILTERS:  ugrizY
KCOR_FILE:  $JAX_TEST_FIXTURE_ROOT/kcor/JAX_TEST_LSST.fits
SIMLIB_FILE:  $JAX_TEST_FIXTURE_ROOT/simlib/JAX_TEST_LSST.SIMLIB
SIMLIB_MINSEASON:  30
SMEARFLAG_FLUX:  1
SMEARFLAG_ZEROPT:  1
HOSTLIB_FILE:        {hostlib}
HOSTLIB_WGTMAP_FILE: {wgtmap}
GENPDF_FILE:         {genpdf}
HOSTLIB_MSKOPT:      {mskopt}
HOSTLIB_DZTOL:       0.0,0.05,0
HOSTLIB_STOREPAR:    {storepar}
GENRANGE_REDSHIFT:   {z_range[0]:.3f}  {z_range[1]:.3f}
GENRANGE_MJD:        60500.0  61500.0
GENRANGE_PEAKMJD:    60540.0  61460.0
GENSIGMA_REDSHIFT:   0.001
GENRANGE_TREST:      -20.0  +50.0
GENRANGE_RA:         -180.0 +180.0
GENRANGE_DECL:       -90.0  +90.0
SOLID_ANGLE:         0.0150
DNDZ:  POWERLAW  2.5E-5  1.5
GENSIGMA_MWEBV_RATIO:  0.16
RANSEED:  654321
""")


@pytest.fixture(scope="module")
def genpdf_wgtmap_input(tmp_path_factory):
    """GENPDF + LOGMASS-only WGTMAP input file (module-scoped so the
    sim_results fixture can run the simulator once for both tests)."""
    tmp = tmp_path_factory.mktemp("genpdf_wgtmap")
    genpdf_path = tmp / "GENPDF_C_LOGMASS.DAT"
    _write_genpdf_c_logmass(genpdf_path)
    wgtmap_path = tmp / "LOGMASS_ONLY.WGTMAP"
    _write_logmass_only_wgtmap(wgtmap_path)
    inp = tmp / "input.input"
    _write_input(inp, hostlib=None, wgtmap=wgtmap_path, genpdf=genpdf_path)
    return inp


@pytest.fixture(scope="module")
def genpdf_wgtmap_results(genpdf_wgtmap_input):
    cfg = dict(
        n_sne=300, random_seed=99001,
    )
    sim = sim_run.prepare(str(genpdf_wgtmap_input), **cfg)
    resources = sim
    results, _, _ = sim_run.run_simulation(sim)
    return results


def test_genpdf_conditional_on_logmass_shifts_color_with_mass(genpdf_wgtmap_results):
    """SALT2c distribution shifts with host LOGMASS via the conditional GENPDF.

    Mean(c) for high-mass hosts (LOGMASS>10) must exceed mean(c) for
    low-mass (LOGMASS<10) — the GENPDF mode goes from -0.05 to +0.05.
    """
    params = genpdf_wgtmap_results["parameters"]
    c = np.asarray(params["c"])
    lm = np.asarray(params["host_LOGMASS"])

    low = lm < 10.0
    high = lm >= 10.0
    assert low.sum() > 20 and high.sum() > 20, (
        f"too few SNe in each mass bin: low={low.sum()} high={high.sum()}"
    )
    mean_c_low = float(np.mean(c[low]))
    mean_c_high = float(np.mean(c[high]))
    assert mean_c_high > mean_c_low + 0.02, (
        f"GENPDF conditioning failed: mean(c|low LOGMASS)={mean_c_low:.3f} "
        f"vs mean(c|high LOGMASS)={mean_c_high:.3f}"
    )


def test_conditional_genpdf_with_bit_16_runs_and_stays_in_bounds(tmp_path):
    """Conditional GENPDF on host LOGMASS + HOSTLIB_MSKOPT bit 16: the
    marginal CDF must build and every SN's sim_redshift_cmb must stay inside
    GENRANGE_REDSHIFT."""
    z_min, z_max = 0.12, 0.32
    genpdf_path = tmp_path / "GENPDF_C_LOGMASS.DAT"
    _write_genpdf_c_logmass(genpdf_path)
    wgtmap_path = tmp_path / "LOGMASS_ONLY.WGTMAP"
    _write_logmass_only_wgtmap(wgtmap_path)
    inp = tmp_path / "input.input"
    _write_input(inp, hostlib=None, wgtmap=wgtmap_path, genpdf=genpdf_path,
                 mskopt=258 + 16, z_range=(z_min, z_max))
    cfg = dict(
        n_sne=200, random_seed=99002,
    )
    sim = sim_run.prepare(str(inp), **cfg)
    resources = sim
    assert resources.hostlib.catalogue.marginal_cdf is not None
    res, _, _ = sim_run.run_simulation(sim)
    sim_z = np.asarray(res["parameters"]["sim_redshift_cmb"])
    assert np.all((sim_z >= z_min) & (sim_z <= z_max))


def test_host_dependent_population_keeps_the_wronghost_draw(tmp_path):
    """The WRONGHOST_FILE draw is keyed on the event and precedes the
    population, so a population that reads the host (conditional GENPDF)
    reports the same CORRECT_HOSTMATCH and SIM_REDSHIFT_HOST as one that
    does not (plain Gaussian colour) under one seed. SIM_REDSHIFT_HOST is
    the drawn galaxy's redshift (GENLC.REDSHIFT_HOST, sntools_host.c:6481),
    the wrong one included."""
    genpdf_path = tmp_path / "GENPDF_C_LOGMASS.DAT"
    _write_genpdf_c_logmass(genpdf_path)
    wgtmap_path = tmp_path / "LOGMASS_ONLY.WGTMAP"
    _write_logmass_only_wgtmap(wgtmap_path)
    wronghost = ("WRONGHOST_FILE: "
                 "$JAX_TEST_FIXTURE_ROOT/hostlib/JAX_TEST_WRONGHOST.DAT\n")
    runs = {}
    for population in ("conditional", "plain"):
        inp = tmp_path / f"{population}.input"
        _write_input(inp, hostlib=None, wgtmap=wgtmap_path, genpdf=genpdf_path)
        text = inp.read_text() + wronghost
        if population == "plain":
            text = text.replace(f"GENPDF_FILE:         {genpdf_path}\n", "")
        inp.write_text(text)
        cfg = dict(n_sne=300, random_seed=99003)
        sim = sim_run.prepare(str(inp), **cfg)
        resources = sim
        res, _, _ = sim_run.run_simulation(sim)
        runs[population] = res["parameters"]
    match = np.asarray(runs["conditional"]["CORRECT_HOSTMATCH"])
    assert 0 < int((match == 0).sum()) < match.size
    np.testing.assert_array_equal(
        match, np.asarray(runs["plain"]["CORRECT_HOSTMATCH"]))
    zhost = np.asarray(runs["conditional"]["sim_redshift_host"])
    np.testing.assert_array_equal(
        zhost, np.asarray(runs["plain"]["sim_redshift_host"]))
    np.testing.assert_array_equal(
        zhost, np.asarray(runs["conditional"]["host_ZTRUE"]))


def test_genpdf_conditioning_column_is_loaded_without_storepar(tmp_path):
    """A GENPDF conditioning variable joins the HOSTLIB store list on its own
    (sntools_host.c:951-969), so a map on a column nothing else requests runs
    without the user naming it in HOSTLIB_STOREPAR."""
    genpdf_path = tmp_path / "GENPDF_C_LOGsSFR.DAT"
    rows = []
    for lsfr in (-13.0, -11.0, -9.0):
        for c in (-0.3, -0.15, -0.05, 0.0, 0.05, 0.15, 0.3):
            prob = float(np.exp(-0.5 * ((c - 0.05 * (lsfr + 11.0)) / 0.07) ** 2))
            rows.append(f"PDF: {c:+.3f} {lsfr:.1f} {prob:.6f}")
    genpdf_path.write_text(
        "\n".join(["VARNAMES: SALT2c LOGsSFR PROB", *rows]) + "\n")
    wgtmap_path = tmp_path / "LOGMASS_ONLY.WGTMAP"
    _write_logmass_only_wgtmap(wgtmap_path)
    inp = tmp_path / "input.input"
    _write_input(inp, hostlib=None, wgtmap=wgtmap_path, genpdf=genpdf_path,
                 storepar="LOGMASS")

    cfg = dict(n_sne=100, random_seed=99005)
    sim = sim_run.prepare(str(inp), **cfg)
    resources = sim
    assert "LOGsSFR" in resources.hostlib.selector.varnames
    res, _, _ = sim_run.run_simulation(sim)
    assert np.isfinite(np.asarray(res["parameters"]["c"])).all()


def _write_narrow_genpdf_c_logmass(out: Path) -> None:
    """GENPDF whose LOGMASS grid [9.5, 10.5] is narrower than the HOSTLIB
    LOGMASS range [8.5, 11.5], so many hosts fall outside the map."""
    rows = []
    for lm in (9.5, 10.0, 10.5):
        for c in (-0.3, -0.15, -0.05, 0.0, 0.05, 0.15, 0.3):
            prob = float(np.exp(-0.5 * ((c - 0.10 * (lm - 10.0)) / 0.07) ** 2))
            rows.append(f"PDF: {c:+.3f} {lm:.1f} {prob:.6f}")
    out.write_text("\n".join(["VARNAMES: SALT2c LOGMASS PROB", *rows]) + "\n")


def _narrow_genpdf_input(tmp_path: Path) -> Path:
    genpdf_path = tmp_path / "GENPDF_NARROW.DAT"
    _write_narrow_genpdf_c_logmass(genpdf_path)
    wgtmap_path = tmp_path / "LOGMASS_ONLY.WGTMAP"
    _write_logmass_only_wgtmap(wgtmap_path)
    inp = tmp_path / "input.input"
    _write_input(inp, hostlib=None, wgtmap=wgtmap_path, genpdf=genpdf_path)
    return inp


def test_conditional_genpdf_out_of_map_aborts_without_extrap(tmp_path):
    """No EXTRAP bit: a drawn host whose LOGMASS lies outside the GENPDF map
    aborts the run, matching SNANA's interp_GRIDMAP (sntools_gridmap.c:574-582);
    preparation accepts the catalogue, since a host is only checked once drawn."""
    inp = _narrow_genpdf_input(tmp_path)
    sim = sim_run.prepare(str(inp), n_sne=200, random_seed=99003)
    with pytest.raises(SNANAError, match="outside map range"):
        sim_run.run_simulation(sim)


def test_sbi_rejects_a_catalogue_outside_the_conditional_genpdf(tmp_path):
    """SBI cannot stop at an event whose host lies outside the map, so it
    rejects a catalogue that has such hosts when the pipeline is built."""
    inp = _narrow_genpdf_input(tmp_path)
    with pytest.raises(SNANAError, match="outside the map"):
        sim_run.build_sbi_pipeline(str(inp), n_sne=200, random_seed=99003)


def test_conditional_genpdf_out_of_map_clamps_with_extrap(tmp_path):
    """GENPDF_OPTMASK bit 1 (EXTRAP) clamps out-of-map conditioning to the
    edge instead of aborting, so the same setup runs to completion."""
    inp = _narrow_genpdf_input(tmp_path)
    with open(inp, "a") as fh:
        fh.write("GENPDF_OPTMASK: 1\n")
    cfg = dict(n_sne=200, random_seed=99004)
    sim = sim_run.prepare(str(inp), **cfg)
    resources = sim
    res, _, _ = sim_run.run_simulation(sim)
    assert np.asarray(res["parameters"]["c"]).shape[0] > 0


def test_wgtmap_skews_logmass_distribution(genpdf_wgtmap_results):
    """The mass-favouring WGTMAP weights high-mass galaxies higher.
    With WGTMAP active, the SN-host LOGMASS distribution should skew
    toward higher mass than the bare HOSTLIB underlying distribution
    (uniform on [8.5, 11.5] → mean 10.0)."""
    lm = np.asarray(genpdf_wgtmap_results["parameters"]["host_LOGMASS"])
    weighted_mean = float(np.mean(lm))
    assert weighted_mean > 10.05, (
        f"WGTMAP skew not detected: mean(LOGMASS)={weighted_mean:.2f}; "
        "expected significantly above 10.0 with mass-favouring WGTMAP"
    )




# ---------------------------------------------------------------------------
# VARIATION: a population component as a polynomial in a driving quantity.
# ---------------------------------------------------------------------------

def _write_variation_input(out: Path, *, lines: str, storepar: str = "LOGMASS",
                           x1_sigma: str = "0.001 0.001",
                           x1_range: str = "-9.0 9.0",
                           hostlib: bool = True) -> None:
    """A SALT run with a near-delta stretch population, so a drawn x1 is its
    peak and the declared relation is recovered without truncation."""
    host = ("HOSTLIB_FILE:     $JAX_TEST_FIXTURE_ROOT/hostlib/JAX_TEST.HOSTLIB\n"
            "HOSTLIB_MSKOPT:   258\n"
            "HOSTLIB_DZTOL:    0.0,0.05,0\n"
            f"HOSTLIB_STOREPAR: {storepar}\n") if hostlib else ""
    out.write_text(f"""\
GENVERSION:  JAX_TEST_VARIATION
NGENTOT_LC:  200
FORMAT_MASK:  32
GENTYPE:  1
SNTYPES_Ia:  1  101
CIDRAN_MIN:  0
CIDRAN_MAX:  3000000
GENMODEL:  $JAX_TEST_FIXTURE_ROOT/models/SALT3.K21
GENPEAK_SALT2x1:    0.0
GENRANGE_SALT2x1:  {x1_range}
GENSIGMA_SALT2x1:  {x1_sigma}
GENPEAK_SALT2c:     0.0
GENRANGE_SALT2c:   -0.3  0.3
GENSIGMA_SALT2c:    0.05  0.05
GENPEAK_SALT2ALPHA: 0.14
GENPEAK_SALT2BETA:  3.1
OMEGA_MATTER:  0.3150
OMEGA_LAMBDA:  0.6850
w0_LAMBDA:  -1.0000
wa_LAMBDA:  0
GENFILTERS:  ugrizY
KCOR_FILE:  $JAX_TEST_FIXTURE_ROOT/kcor/JAX_TEST_LSST.fits
SIMLIB_FILE:  $JAX_TEST_FIXTURE_ROOT/simlib/JAX_TEST_LSST.SIMLIB
SIMLIB_MINSEASON:  30
SMEARFLAG_FLUX:  1
SMEARFLAG_ZEROPT:  1
{host}GENRANGE_REDSHIFT:   0.050  0.400
GENRANGE_MJD:        60500.0  61500.0
GENRANGE_PEAKMJD:    60540.0  61460.0
GENSIGMA_REDSHIFT:   0.001
GENRANGE_TREST:      -20.0  +50.0
GENRANGE_RA:         -180.0 +180.0
GENRANGE_DECL:       -90.0  +90.0
SOLID_ANGLE:         0.0150
DNDZ:  POWERLAW  2.5E-5  1.5
GENSIGMA_MWEBV_RATIO:  0.16
RANSEED:  654321
{lines}""")


def _run_variation(tmp_path, **kwargs):
    inp = tmp_path / "variation.input"
    _write_variation_input(inp, **kwargs)
    cfg = dict(n_sne=300, random_seed=4242)
    sim = sim_run.prepare(str(inp), **cfg)
    resources = sim
    results, _, _ = sim_run.run_simulation(sim)
    return resources, results


def test_variation_recovers_the_declared_slope(tmp_path):
    """A stretch peak declared to slide with host mass does exactly that."""
    _, results = _run_variation(
        tmp_path,
        lines="VARIATION: GENPEAK_SALT2x1 LOGMASS POLY -0.30 PIVOT 10.0\n")
    x1 = np.asarray(results["parameters"]["x1"])
    logmass = np.asarray(results["parameters"]["host_LOGMASS"])
    slope = float(np.polyfit(logmass, x1, 1)[0])
    assert slope == pytest.approx(-0.30, abs=0.01), f"got slope {slope}"
    # the relation is anchored, so the pivot mass draws the unshifted peak
    np.testing.assert_allclose(x1 + 0.30 * (logmass - 10.0), 0.0, atol=0.01)


def test_variation_host_column_loads_without_storepar(tmp_path):
    """The host column a VARIATION depends on joins the HOSTLIB store list
    on its own, as a GENPDF's conditioning variable does
    (sntools_host.c:951-969). LOGsSFR is not an essential column, so only
    the scan can have loaded it."""
    resources, results = _run_variation(
        tmp_path, storepar="LOGMASS",
        lines="VARIATION: GENPEAK_SALT2x1 LOGsSFR POLY -0.40 PIVOT -10.0\n")
    assert "LOGsSFR" in resources.hostlib.selector.varnames
    x1 = np.asarray(results["parameters"]["x1"])
    lsfr = np.asarray(results["parameters"]["host_LOGsSFR"])
    assert float(np.polyfit(lsfr, x1, 1)[0]) == pytest.approx(-0.40, abs=0.01)


def test_variation_and_zvariation_shifts_sum(tmp_path):
    """Evolution and an environmental relation compose on one parameter."""
    _, results = _run_variation(
        tmp_path,
        lines=("VARIATION: GENPEAK_SALT2x1 LOGMASS POLY -0.30 PIVOT 10.0\n"
               "ZVARIATION_POLY: GENPEAK_SALT2x1 0.0,2.0\n"))
    x1 = np.asarray(results["parameters"]["x1"])
    logmass = np.asarray(results["parameters"]["host_LOGMASS"])
    z = np.asarray(results["parameters"]["sim_redshift_cmb"])
    np.testing.assert_allclose(
        x1 - (-0.30 * (logmass - 10.0) + 2.0 * z), 0.0, atol=0.01)


# ---------------------------------------------------------------------------
# SUBPOP: a class drawn once per event, with its own population.
# ---------------------------------------------------------------------------

_TWO_POPULATIONS = """
SUBPOP: YOUNG  WGT                 0.4
SUBPOP: YOUNG  GENPEAK_SALT2x1     0.4
SUBPOP: YOUNG  GENPEAK_SALT2ALPHA  0.12
SUBPOP: OLD    WGT                 0.6
SUBPOP: OLD    GENPEAK_SALT2x1    -1.0
SUBPOP: OLD    GENPEAK_SALT2ALPHA  0.16
SUBPOP: OLD    GENMAG_OFF_GLOBAL   0.06
"""


def test_subpop_correlates_every_parameter_through_one_class_draw(tmp_path):
    """The class is drawn once per event, so stretch and alpha agree on it
    for every SN. Two independent GENPROB2 mixtures would not."""
    _, results = _run_variation(tmp_path, lines=_TWO_POPULATIONS)
    params = results["parameters"]
    cls = np.asarray(params["subpop"])
    x1 = np.asarray(params["x1"])
    alpha = np.asarray(params["SALT2alpha"])

    assert set(np.unique(cls)) == {0, 1}
    for index, (peak, a) in enumerate(((0.4, 0.12), (-1.0, 0.16))):
        drawn = cls == index
        assert drawn.sum() > 20
        np.testing.assert_allclose(x1[drawn], peak, atol=0.01)
        np.testing.assert_allclose(alpha[drawn], a, atol=1e-6)
    # every event's alpha is the one belonging to its stretch population
    assert np.all((x1 > -0.3) == (alpha < 0.14))


def test_subpop_weights_are_relative(tmp_path):
    """0.4 and 0.6 draw the declared fractions; the weights need not sum to
    one, they set the proportion."""
    _, results = _run_variation(tmp_path, lines=_TWO_POPULATIONS.replace(
        "WGT                 0.4", "WGT                 4.0").replace(
        "WGT                 0.6", "WGT                 6.0"))
    cls = np.asarray(results["parameters"]["subpop"])
    assert float((cls == 0).mean()) == pytest.approx(0.4, abs=0.07)


def test_subpop_unqualified_keys_are_shared(tmp_path):
    """A key no class sets keeps the input's value for every class."""
    _, results = _run_variation(tmp_path, lines=_TWO_POPULATIONS)
    beta = np.asarray(results["parameters"]["SALT2beta"])
    np.testing.assert_allclose(beta, beta[0], atol=1e-6)


def test_subpop_weight_follows_a_host_column(tmp_path):
    """The class fraction is a number like any other, so a VARIATION on WGT
    makes the population mix depend on the environment."""
    _, results = _run_variation(tmp_path, lines=_TWO_POPULATIONS + (
        "SUBPOP: YOUNG VARIATION WGT LOGMASS POLY -0.20 PIVOT 10.0\n"))
    cls = np.asarray(results["parameters"]["subpop"])
    logmass = np.asarray(results["parameters"]["host_LOGMASS"])
    young_low = float((cls == 0)[logmass < 10.0].mean())
    young_high = float((cls == 0)[logmass >= 10.0].mean())
    assert young_low > young_high + 0.05, (
        f"young fraction should fall with mass: {young_low:.3f} -> {young_high:.3f}")


def test_subpop_refused_when_usesnpar_would_overwrite_the_parameter(tmp_path):
    """The HOSTLIB column would replace the drawn value, so the class would
    have no effect: that aborts rather than silently collapsing.

    Driven against the check directly, because the bundled HOSTLIB carries no
    x1 column, and without one USESNPAR aborts earlier for having nothing to
    use at all.
    """
    from jax_snlc_core.models import MODEL_REGISTRY
    from jax_snlc_sim.hostlib.association import _check_snpar_conflict
    from jax_snlc_sim.inputs import parse_snana_input_file

    inp = tmp_path / "clash.input"
    _write_variation_input(inp, lines=_TWO_POPULATIONS)
    config = parse_snana_input_file(str(inp))

    class _Model:
        configured_population = tuple(
            (name, MODEL_REGISTRY["salt"].population_aliases.get(name, ()))
            for name in MODEL_REGISTRY["salt"].population_parameters)

    with pytest.raises(SNANAError, match="USESNPAR"):
        _check_snpar_conflict(config, _Model(), ["GALID", "x1"])
    # no x1 column, nothing to overwrite, nothing to refuse
    _check_snpar_conflict(config, _Model(), ["GALID", "LOGMASS"])


def test_subpop_magnitude_offset_reaches_the_photometry(tmp_path):
    """A class declared fainter is fainter: GENMAG_OFF_GLOBAL is per class,
    scaling FLUXCAL after the integral as the grey offset does."""
    _, results = _run_variation(tmp_path, lines=(
        "SUBPOP: A  WGT 1.0\n"
        "SUBPOP: B  WGT 1.0\n"
        "SUBPOP: B  GENMAG_OFF_GLOBAL 0.50\n"))
    cls = np.asarray(results["parameters"]["subpop"])
    peakmags = np.asarray(results["peakmags"])
    finite = np.isfinite(peakmags)
    faint = np.median(peakmags[(cls == 1)[:, None] & finite])
    bright = np.median(peakmags[(cls == 0)[:, None] & finite])
    assert faint - bright == pytest.approx(0.50, abs=0.05)


@pytest.mark.parametrize("lines, present", [(_TWO_POPULATIONS, True),
                                            ("", False)])
def test_sim_subpop_column_is_written_only_when_classes_are_declared(
        tmp_path, lines, present):
    """The class reaches the output as truth, and only from a run that
    declares classes: a jax-only column, so every other run's columns are
    unchanged and the output contract against SNANA still holds."""
    from astropy.io import fits
    from jax_snlc_core.snana_files import resolve_fits_path

    inp = tmp_path / "columns.input"
    _write_variation_input(inp, lines=lines)
    cfg = dict(n_sne=60, random_seed=8)
    sim = sim_run.prepare(str(inp), **cfg)
    resources = sim
    results, selection, spectra = sim_run.run_simulation(sim)
    outdir = str(tmp_path / "out")
    save_snana_outputs(sim, results, "SUBPOP", selection, spectra_by_sn=spectra, output_dir=outdir)
    head = resolve_fits_path(str(Path(outdir) / "SUBPOP" / "SUBPOP_HEAD.FITS"))
    with fits.open(head) as hdus:
        columns = {c.name for c in hdus[1].columns}
    assert ("SIM_SUBPOP" in columns) is present
    if present:
        with fits.open(head) as hdus:
            assert set(np.unique(hdus[1].data["SIM_SUBPOP"])) <= {0, 1}
        # the column is an index, so the README has to say which class each
        # index is or the truth cannot be read back
        readme = (Path(outdir) / "SUBPOP" / "SUBPOP.README").read_text()
        assert "SUBPOP:                  0 YOUNG" in readme
        assert "SUBPOP:                  1 OLD" in readme


# ---------------------------------------------------------------------------
# A VARIATION depending on a drawn parameter: alpha non-linear in stretch.
# ---------------------------------------------------------------------------

def test_variation_depending_on_an_already_drawn_parameter(tmp_path):
    """Ginolin et al. 2024 ask whether alpha is linear in x1. Here it is not,
    and no HOSTLIB is involved: alpha depends on the stretch drawn moments
    earlier, the general form of SNANA's SALT2BETA_cPOLY."""
    _, results = _run_variation(
        tmp_path, hostlib=False, x1_sigma="1.0 1.0", x1_range="-3.0 3.0",
        lines="VARIATION: GENPEAK_SALT2ALPHA SALT2x1 POLY -0.010 PIVOT 0.0\n")
    x1 = np.asarray(results["parameters"]["x1"])
    alpha = np.asarray(results["parameters"]["SALT2alpha"])
    np.testing.assert_allclose(alpha, 0.14 - 0.010 * x1, atol=1e-6)


def test_a_parameter_with_no_distribution_is_refused(tmp_path):
    """Without SALT2BETA_cPOLY nothing supplies beta, so a run that gives it
    no distribution stops on the first batch, naming the keys to give."""

    inp = tmp_path / "variation.input"
    _write_variation_input(inp, hostlib=False, lines="")
    inp.write_text(inp.read_text().replace("GENPEAK_SALT2BETA:  3.1\n", ""))
    sim = sim_run.prepare(str(inp), n_sne=10, random_seed=1)
    with pytest.raises(SNANAError, match="SALT2beta has no distribution: "
                                         "give GENPEAK_SALT2beta"):
        sim_run.run_simulation(sim)


def test_a_zero_width_parameter_still_takes_its_variation(tmp_path):
    """SALT2ALPHA is declared with no width, and the zero-width shortcut
    reads the declared peak. A shifted parameter must take the sampler
    instead, or the relation would be silently dropped."""
    _, results = _run_variation(
        tmp_path, lines=(
            "VARIATION: GENPEAK_SALT2ALPHA LOGMASS POLY 0.02 PIVOT 10.0\n"))
    alpha = np.asarray(results["parameters"]["SALT2alpha"])
    logmass = np.asarray(results["parameters"]["host_LOGMASS"])
    np.testing.assert_allclose(alpha, 0.14 + 0.02 * (logmass - 10.0), atol=1e-6)


def test_variation_depending_on_a_parameter_drawn_later_aborts(tmp_path):
    """SALT draws x1, then c, then alpha, so a relation of x1 on alpha has
    no value to read and says so rather than using a default."""
    with pytest.raises(SNANAError, match="depends on SALT2alpha, which is not "
                                         "drawn before it"):
        _run_variation(
            tmp_path, hostlib=False,
            lines="VARIATION: GENPEAK_SALT2x1 SALT2ALPHA POLY 1.0 PIVOT 0.0\n")


def test_a_variation_on_a_drawn_parameter_leaves_the_host_ordering_alone(
        tmp_path):
    """Only a variation on a host column needs the host before the
    population. One depending on an already-drawn parameter reads no host,
    so it coexists with a WGTMAP that bins on SN variables, which draws the
    population first; one on a host column cannot."""

    wgtmap = tmp_path / "X1_LOGMASS.WGTMAP"
    rows = ["VARNAMES_WGTMAP: x1 LOGMASS WGT"]
    for x1 in (-3.0, 0.0, 3.0):
        for logmass in (8.0, 10.0, 12.0):
            rows.append(f"WGT: {x1} {logmass} {1.0 + 0.1 * logmass}")
    wgtmap.write_text("\n".join(rows) + "\n")
    host_lines = f"HOSTLIB_WGTMAP_FILE: {wgtmap}\n"

    _run_variation(tmp_path, x1_sigma="1.0 1.0", x1_range="-3.0 3.0",
                   lines=host_lines + ("VARIATION: GENPEAK_SALT2ALPHA "
                                       "SALT2x1 POLY -0.010 PIVOT 0.0\n"))
    with pytest.raises(SNANAError, match="WGTMAP SN variables"):
        _run_variation(tmp_path, lines=host_lines + (
            "VARIATION: GENPEAK_SALT2x1 LOGMASS POLY -0.30 PIVOT 10.0\n"))


def test_subpop_variation_on_an_unknown_target_aborts(tmp_path):
    """A class-scoped VARIATION is validated against the same registry a
    run-level one is, so a misspelt target cannot quietly do nothing."""
    with pytest.raises(SNANAError, match="not a recognised SIM-parameter or one of WGT"):
        _run_variation(tmp_path, lines=(
            "SUBPOP: YOUNG WGT 1.0\n"
            "SUBPOP: YOUNG VARIATION GENPEAK_NOSUCHTHING LOGMASS "
            "POLY 1.0 PIVOT 10.0\n"))


def test_a_host_property_resolves_under_either_spelling(tmp_path):
    """SNANA reads LOGMASS and LOGMASS_TRUE as one column wherever a name
    appears (checkAlternateVarNames_HOSTLIB, sntools_host.c:3029): a WGTMAP
    and a STOREPAR spelt _TRUE over a HOSTLIB spelt bare select the same
    hosts and store the same column as the bare spelling does."""
    drawn = {}
    for spelling in ("LOGMASS", "LOGMASS_TRUE"):
        wgtmap = tmp_path / f"{spelling}.WGT"
        _write_logmass_only_wgtmap(wgtmap)
        wgtmap.write_text(wgtmap.read_text().replace(
            "VARNAMES_WGTMAP: LOGMASS", f"VARNAMES_WGTMAP: {spelling}"))
        genpdf = tmp_path / f"{spelling}.DAT"
        _write_genpdf_c_logmass(genpdf)
        inp = tmp_path / f"{spelling}.input"
        _write_input(inp, hostlib=None, wgtmap=wgtmap, genpdf=genpdf,
                     storepar=spelling)
        cfg = dict(n_sne=60, random_seed=99001)
        sim = sim_run.prepare(str(inp), **cfg)
        resources = sim
        results, _, _ = sim_run.run_simulation(sim)
        drawn[spelling] = results["parameters"]

    for column in ("host_GALID", "host_LOGMASS", "sim_hostlib_LOGMASS_TRUE"):
        np.testing.assert_array_equal(np.asarray(drawn["LOGMASS_TRUE"][column]),
                                      np.asarray(drawn["LOGMASS"][column]), err_msg=column)


def test_host_coordinates_resolve_under_any_of_their_names(tmp_path):
    """A HOSTLIB spelling its coordinates RA/DEC or RA_HOST/DEC_HOST places
    the same hosts as one spelling them RA_GAL/DEC_GAL
    (sntools_host.c:2740-2750)."""
    source = Path(os.environ["JAX_TEST_FIXTURE_ROOT"]) / "hostlib" / "JAX_TEST.HOSTLIB"
    with open_text(source) as stream:
        fixture_text = stream.read()
    header = next(l for l in fixture_text.splitlines() if l.startswith("VARNAMES:"))
    drawn = {}
    for ra, dec in (("RA_GAL", "DEC_GAL"), ("RA", "DEC"), ("RA_HOST", "DEC_HOST")):
        hostlib = tmp_path / f"{ra}.HOSTLIB"
        hostlib.write_text(fixture_text.replace(
            header, header.replace(" RA_GAL ", f" {ra} ").replace(" DEC_GAL ", f" {dec} ")))
        wgtmap = tmp_path / f"{ra}.WGT"
        _write_logmass_only_wgtmap(wgtmap)
        genpdf = tmp_path / f"{ra}.DAT"
        _write_genpdf_c_logmass(genpdf)
        inp = tmp_path / f"{ra}.input"
        _write_input(inp, hostlib=hostlib, wgtmap=wgtmap, genpdf=genpdf)
        cfg = dict(n_sne=60, random_seed=99001)
        sim = sim_run.prepare(str(inp), **cfg)
        resources = sim
        results, _, _ = sim_run.run_simulation(sim)
        drawn[ra] = results["parameters"]

    for ra in ("RA", "RA_HOST"):
        for column in ("host_GALID", "host_RA_GAL", "host_DEC_GAL"):
            np.testing.assert_array_equal(np.asarray(drawn[ra][column]),
                                          np.asarray(drawn["RA_GAL"][column]),
                                          err_msg=f"{ra} {column}")


def test_the_wgtmap_is_evaluated_only_over_reachable_galaxies(tmp_path):
    """A galaxy beyond the read-time redshift window is never weighted, so a
    WGTMAP grid that does not cover it does not abort the run, as SNANA
    reads the DES-SN5YR low-z library (sntools_host.c:3147-3156)."""
    source = Path(os.environ["JAX_TEST_FIXTURE_ROOT"]) / "hostlib" / "JAX_TEST.HOSTLIB"
    with open_text(source) as stream:
        fixture_text = stream.read()
    lines = fixture_text.splitlines()
    header = next(l for l in lines if l.startswith("VARNAMES:")).split()[1:]
    iz, im = header.index("ZTRUE"), header.index("LOGMASS")
    far = next(i for i, l in enumerate(lines)
               if l.startswith("GAL:") and float(l.split()[1 + iz]) > 0.45)
    toks = lines[far].split()
    toks[1 + im] = "13.5"
    lines[far] = " ".join(toks)
    hostlib = tmp_path / "far.HOSTLIB"
    hostlib.write_text("\n".join(lines) + "\n")

    wgtmap = tmp_path / "far.WGT"
    _write_logmass_only_wgtmap(wgtmap)
    genpdf = tmp_path / "far.DAT"
    _write_genpdf_c_logmass(genpdf)
    inp = tmp_path / "far.input"
    _write_input(inp, hostlib=hostlib, wgtmap=wgtmap, genpdf=genpdf,
                 z_range=(0.05, 0.30))
    _prepared = sim_run.prepare(str(inp), **dict(n_sne=20, random_seed=99001))
    resources = _prepared
    assert float(np.max(resources.hostlib.catalogue.redshifts)) < 0.45

    _write_input(inp, hostlib=hostlib, wgtmap=wgtmap, genpdf=genpdf,
                 z_range=(0.05, 0.48))
    with pytest.raises(SNANAError, match="HOSTLIB_WGTMAP_EXTRAP"):
        sim_run.prepare(str(inp), **dict(n_sne=20, random_seed=99001))
