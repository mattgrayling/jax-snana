"""Unit tests for the GENPDF file parser."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_snlc_sim.genpdf import (load_all_genpdf, parse_genpdf_file, sample_genpdf,
                                        sample_genpdf_conditional)


def _write(tmp_path, content):
    p = tmp_path / "GENPDF.DAT"
    p.write_text(content)
    return str(p)


def test_parse_genpdf_basic_1d(tmp_path):
    path = _write(tmp_path, """
DOCUMENTATION:
    PURPOSE: test 1D
DOCUMENTATION_END:
VARNAMES: SALT2c PROB
PDF: -0.3 0.001
PDF: -0.2 0.05
PDF: -0.1 0.20
PDF:  0.0 1.00
PDF:  0.1 0.20
PDF:  0.2 0.05
PDF:  0.3 0.001
""")
    maps, embedded = parse_genpdf_file(path)
    assert "SALT2c" in maps
    data = maps["SALT2c"]["data"]
    assert data.shape == (7, 2)
    np.testing.assert_allclose(data[:, 0], [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
    assert embedded == {}


def test_parse_genpdf_collects_embedded_overrides(tmp_path):
    """``GENPEAK_*`` etc. lines populate the embedded-overrides dict."""
    path = _write(tmp_path, """
GENPEAK_SALT2ALPHA: 0.115
GENSIGMA_SALT2ALPHA: 0.003 0.003
GENRANGE_SALT2ALPHA: 0.09 0.13
MAG_OFFSET: 0.025

VARNAMES: SALT2c PROB
PDF: -0.3 0.001
PDF:  0.0 1.000
PDF:  0.3 0.001
""")
    maps, embedded = parse_genpdf_file(path)
    assert embedded["GENPEAK_SALT2ALPHA"] == "0.115"
    assert embedded["GENSIGMA_SALT2ALPHA"] == "0.003 0.003"
    assert embedded["GENRANGE_SALT2ALPHA"] == "0.09 0.13"
    assert embedded["MAG_OFFSET"] == "0.025"
    assert "SALT2c" in maps


def test_parse_genpdf_2d_conditional(tmp_path):
    path = _write(tmp_path, """
VARNAMES: SALT2c LOGMASS PROB
PDF: -0.3 9.0 0.01
PDF:  0.0 9.0 1.00
PDF:  0.3 9.0 0.01
PDF: -0.3 11.0 0.05
PDF:  0.0 11.0 0.50
PDF:  0.3 11.0 0.05
""")
    maps, embedded = parse_genpdf_file(path)
    assert "SALT2c" in maps
    data = maps["SALT2c"]["data"]
    assert data.shape == (6, 3)
    assert maps["SALT2c"]["varnames"] == ["SALT2c", "LOGMASS", "PROB"]


def test_parse_genpdf_skips_doc_blocks(tmp_path):
    path = _write(tmp_path, """
DOCUMENTATION:
  PDF: -100 100
  VARNAMES: should_be_ignored
DOCUMENTATION_END:

VARNAMES: x PROB
PDF: 0.0 1.0
""")
    maps, _ = parse_genpdf_file(path)
    assert "x" in maps
    assert "should_be_ignored" not in maps


# ----------------------------------------------------------------------------
# load_all_genpdf + sample_genpdf — high-level round-trip
# ----------------------------------------------------------------------------


def test_sample_genpdf_conditional_concentrates_near_conditional_peak(tmp_path):
    """A 1D conditional GENPDF whose mode shifts with the conditioner: samples
    drawn at each conditioning value concentrate near that value's peak."""
    # Two conditioning bins: cond=0 → mode at x=-0.1, cond=1 → mode at x=+0.1
    p = tmp_path / "GENPDF.DAT"
    p.write_text("""
VARNAMES: SALT2c LOGMASS PROB
PDF: -0.3  0.0 0.001
PDF: -0.1  0.0 1.0
PDF:  0.1  0.0 0.05
PDF:  0.3  0.0 0.001
PDF: -0.3  1.0 0.001
PDF: -0.1  1.0 0.05
PDF:  0.1  1.0 1.0
PDF:  0.3  1.0 0.001
""")
    pdfs, _ = load_all_genpdf(str(p), prob_expon_rewgt=1.0)
    pdf = pdfs["SALT2c"]
    keys = jax.random.split(jax.random.PRNGKey(2), 4000)

    samples_low = jax.vmap(
        lambda k: sample_genpdf_conditional(k, pdf, jnp.float32(0.0))
    )(keys)
    samples_high = jax.vmap(
        lambda k: sample_genpdf_conditional(k, pdf, jnp.float32(1.0))
    )(keys)

    mean_low = float(np.mean(np.asarray(samples_low)))
    mean_high = float(np.mean(np.asarray(samples_high)))
    # Means must shift in the direction of the conditional mode.
    assert mean_low < 0.0, f"cond=0 expected negative mean, got {mean_low:.3f}"
    assert mean_high > 0.0, f"cond=1 expected positive mean, got {mean_high:.3f}"


def test_genpdf_log_axis_returns_linear_value(tmp_path):
    """``LOG_<var>`` PROB axis: returned values are 10^x, not log10(x)."""
    p = tmp_path / "GENPDF.DAT"
    p.write_text("""
VARNAMES: LOG_AV PROB
PDF: -2.0 0.001
PDF: -1.0 1.0
PDF:  0.0 0.001
""")
    pdfs, _ = load_all_genpdf(str(p), prob_expon_rewgt=1.0)
    pdf = pdfs["LOG_AV"]
    assert pdf.is_log
    keys = jax.random.split(jax.random.PRNGKey(3), 200)
    s = np.asarray(jax.vmap(lambda k: sample_genpdf(k, pdf))(keys))
    assert np.all(s >= 0.0), "log axis should return positive linear AV values"
    assert s.min() >= 10**(-2.0) - 1e-6
    assert s.max() <= 10**(0.0) + 1e-6


# ---------------------------------------------------------------------------
# Exact piecewise-linear inversion and SNANA's interpolated-raw-PDF mixture
# (getRan_genPDF, sntools_genPDF.c:691-865)
# ---------------------------------------------------------------------------


def test_sample_genpdf_inverts_linear_pdf_exactly(tmp_path):
    """A triangular PDF p(x) = 2x on [0,1] has CDF x^2, so samples follow
    sqrt(uniform). The exact quadratic interval inversion reproduces that
    with only two grid nodes; a linear interpolation between the endpoint
    CDF values would return the uniform itself (mean 0.5, P(X<0.5)=0.5)."""
    p = tmp_path / "GENPDF.DAT"
    p.write_text("""
VARNAMES: SALT2c PROB
PDF: 0.0 0.0
PDF: 1.0 2.0
""")
    pdfs, _ = load_all_genpdf(str(p), prob_expon_rewgt=1.0)
    pdf = pdfs["SALT2c"]
    keys = jax.random.split(jax.random.PRNGKey(7), 20000)
    s = np.asarray(jax.vmap(lambda k: sample_genpdf(k, pdf))(keys))
    assert abs(float(np.mean(s)) - 2.0 / 3.0) < 0.01
    assert abs(float(np.mean(s < 0.5)) - 0.25) < 0.02


def test_expon_rewgt_raises_the_interpolated_density(tmp_path):
    """GENPDF_EXPON_REWGT applies to the interpolated probability, after
    interpolation (funVal_genPDF, sntools_genPDF.c:670-672): p(x) = 2x on
    [0,1] with e=2 gives density x^2, so samples follow u^(1/3) with mean
    3/4 and P(X<0.5) = 1/8; exponentiating the two grid nodes instead would
    leave the density linear (mean 2/3)."""
    p = tmp_path / "GENPDF.DAT"
    p.write_text("""
VARNAMES: SALT2c PROB
PDF: 0.0 0.0
PDF: 1.0 2.0
""")
    pdfs, _ = load_all_genpdf(str(p), prob_expon_rewgt=2.0)
    pdf = pdfs["SALT2c"]
    keys = jax.random.split(jax.random.PRNGKey(7), 20000)
    s = np.asarray(jax.vmap(lambda k: sample_genpdf(k, pdf))(keys))
    assert abs(float(np.mean(s)) - 0.75) < 0.01
    assert abs(float(np.mean(s < 0.5)) - 0.125) < 0.01


def test_conditional_mixture_weights_carry_raw_integrals(tmp_path):
    """At a conditioning coordinate midway between two corners, SNANA samples
    the normalised interpolated raw PDF: corner i enters with probability
    a_i*I_i/sum(a_j*I_j), NOT with the bare geometric weight a_i. Corner 0 is
    a rising triangle with raw integral 10, corner 1 a falling triangle with
    raw integral 1; at frac=0.5 the interpolated density has mean 7/11. A
    corner-normalised mixture (a_i alone) would give mean 0.5."""
    p = tmp_path / "GENPDF.DAT"
    p.write_text("""
VARNAMES: SALT2c LOGMASS PROB
PDF: 0.0  0.0  0.0
PDF: 1.0  0.0  20.0
PDF: 0.0  1.0  2.0
PDF: 1.0  1.0  0.0
""")
    pdfs, _ = load_all_genpdf(str(p), prob_expon_rewgt=1.0)
    pdf = pdfs["SALT2c"]
    np.testing.assert_allclose(np.asarray(pdf.raw_integral), [10.0, 1.0])
    keys = jax.random.split(jax.random.PRNGKey(11), 20000)
    s = np.asarray(jax.vmap(
        lambda k: sample_genpdf_conditional(k, pdf, jnp.float32(0.5)))(keys))
    assert abs(float(np.mean(s)) - 7.0 / 11.0) < 0.01


def test_genpdf_construction_rejects_malformed_maps(tmp_path):
    """Negative PROB, duplicate cells, missing Cartesian cells, and zero-mass
    slices are recognised-input errors and abort at construction."""
    from jax_snlc_core.logging import SNANAError

    cases = {
        "negative": """
VARNAMES: SALT2c PROB
PDF: 0.0 -0.5
PDF: 1.0 1.0
""",
        "duplicate": """
VARNAMES: SALT2c PROB
PDF: 0.0 1.0
PDF: 0.0 1.0
PDF: 1.0 1.0
""",
        "missing": """
VARNAMES: SALT2c LOGMASS PROB
PDF: 0.0 0.0 1.0
PDF: 1.0 0.0 1.0
PDF: 0.0 1.0 1.0
""",
        "zero-mass": """
VARNAMES: SALT2c LOGMASS PROB
PDF: 0.0 0.0 1.0
PDF: 1.0 0.0 1.0
PDF: 0.0 1.0 0.0
PDF: 1.0 1.0 0.0
""",
    }
    for name, text in cases.items():
        f = tmp_path / f"GENPDF_{name}.DAT"
        f.write_text(text)
        with pytest.raises(SNANAError):
            load_all_genpdf(str(f), prob_expon_rewgt=1.0)


def test_conditional_expon_applies_after_mixing(tmp_path):
    """With EXPON_REWGT the corner mixture is exponentiated after
    interpolation (getRan_genPDF, sntools_genPDF.c:773-774): opposite
    triangles mix to a flat density at frac=0.5, so any exponent leaves the
    samples uniform. Exponentiating the corner slices before mixing would
    concentrate them at both edges (P(X<0.25) = 0.30, var = 0.10)."""
    p = tmp_path / "GENPDF.DAT"
    p.write_text("""
VARNAMES: SALT2c LOGMASS PROB
PDF: 0.0  0.0  0.0
PDF: 1.0  0.0  2.0
PDF: 0.0  1.0  2.0
PDF: 1.0  1.0  0.0
""")
    pdfs, _ = load_all_genpdf(str(p), prob_expon_rewgt=2.0)
    pdf = pdfs["SALT2c"]
    keys = jax.random.split(jax.random.PRNGKey(3), 20000)
    s = np.asarray(jax.vmap(
        lambda k: sample_genpdf_conditional(k, pdf, jnp.float32(0.5)))(keys))
    assert abs(float(np.mean(s < 0.25)) - 0.25) < 0.01
    assert abs(float(np.var(s)) - 1.0 / 12.0) < 0.004
