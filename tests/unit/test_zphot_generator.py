"""Unit tests for ``generate_zphot_from_calc``, the Gaussian-quantile
helpers and the HOSTLIB_ZPHOTEFF draw."""

import numpy as np
import pytest

from jax_snlc_core.logging import SNANAError
from jax_snlc_sim.hostlib.association import _apply_zphoteff
from jax_snlc_sim.hostlib.photoz import DECADE_QUANTILES
from jax_snlc_sim.hostlib.photoz import (NRAND_ZPHOT, ZPHOT_MIN, GenZphot,
                                                  ZphotEfficiency, build_genzphot,
                                                  generate_zphot_from_calc,
                                                  generate_zphot_quantiles_gaussian,
                                                  read_zphoteff_file,
                                                  validate_genzphot_fudgemap)
from jax_snlc_core.random import root_key
from jax_snlc_sim.simulation import Host


def _model(fudgepar, outlier_range=(0.0, 1.0)):
    """The FUDGEPAR-only photo-z model."""
    return GenZphot(fudgepar=np.asarray(fudgepar, dtype=float),
                    fudgemap_z=None, fudgemap_rms=None, bias=None,
                    outlier_range=outlier_range)


def test_zphoteff_masks_the_events_photoz_and_not_the_reported_one():
    """HOSTLIB_ZPHOTEFF is a per-host success probability read at the host's
    own ZTRUE, so half the hosts fail a flat 50% efficiency. SNANA masks
    SNHOSTGAL.ZPHOT, which the CUTWIN and the redshift source read, and
    leaves the DDLR-sorted photo-z the data file reports alone
    (sntools_host.c:6831, snlc_sim.c:26068)."""
    n = 200
    efficiency = ZphotEfficiency(np.array([0.0, 1.0]), np.array([0.5, 0.5]))
    props = {'ZTRUE': np.full(n, 0.3), 'ZPHOT': np.full(n, 0.31),
             'ZPHOT_ERR': np.full(n, 0.02)}
    _apply_zphoteff(efficiency, props, n, 0, key=root_key(3))
    failed = np.asarray(props['ZPHOT_EVENT']) < 0.0
    assert 70 < int(failed.sum()) < 130
    np.testing.assert_array_equal(
        np.asarray(props['ZPHOT_EVENT'])[failed], np.full(int(failed.sum()), -9.0))
    np.testing.assert_array_equal(
        np.asarray(props['ZPHOT_EVENT_ERR'])[failed],
        np.full(int(failed.sum()), -9.0))
    np.testing.assert_array_equal(
        np.asarray(props['ZPHOT_EVENT'])[~failed],
        np.full(int((~failed).sum()), 0.31))
    np.testing.assert_array_equal(props['ZPHOT'], np.full(n, 0.31))
    np.testing.assert_array_equal(props['ZPHOT_ERR'], np.full(n, 0.02))


def test_zphoteff_is_read_at_the_host_redshift():
    """The efficiency is interpolated at SNHOSTGAL.ZTRUE, so a map that is
    1 below z=0.5 and 0 above it passes every low-z host and fails every
    high-z one (sntools_host.c:6829)."""
    efficiency = ZphotEfficiency(np.array([0.0, 0.49, 0.51, 1.0]),
                                 np.array([1.0, 1.0, 0.0, 0.0]))
    ztrue = np.array([0.1, 0.2, 0.8, 0.9])
    props = {'ZTRUE': ztrue, 'ZPHOT': ztrue + 0.01,
             'ZPHOT_ERR': np.full(4, 0.02)}
    _apply_zphoteff(efficiency, props, len(ztrue), 0, key=root_key(3))
    np.testing.assert_array_equal(
        np.asarray(props['ZPHOT_EVENT']) >= 0.0,
        np.array([True, True, False, False]))


def test_zphoteff_file_missing_or_without_two_columns_aborts(tmp_path):
    """A missing HOSTLIB_ZPHOTEFF_FILE aborts, as SNANA's rd2columnFile does
    (sntools_host.c:4679); one without an EFF column aborts too, where SNANA
    crashes. A good file reads as its two columns."""
    hosts = (0.1, 0.5)
    with pytest.raises(SNANAError, match="not found"):
        read_zphoteff_file(tmp_path / "NOSUCH.DAT", hosts)
    onecol = tmp_path / "ONECOL.DAT"
    onecol.write_text("0.1\n0.3\n")
    with pytest.raises(SNANAError, match="two numbers"):
        read_zphoteff_file(onecol, hosts)
    good = tmp_path / "EFF.DAT"
    good.write_text("DOCUMENTATION:\nPURPOSE: photo-z efficiency\nDOCUMENTATION_END:\n"
                    "! comment\n% comment\n@ comment\n# ZTRUE EFF\n0.0 1.0\n1.0 0.5\n")
    efficiency = read_zphoteff_file(good, hosts)
    np.testing.assert_array_equal(efficiency.ztrue, [0.0, 1.0])
    np.testing.assert_array_equal(efficiency.eff, [1.0, 0.5])


def test_generate_zphot_quantiles_gaussian_monotonic():
    """ZPHOT_Q values are monotonically increasing per row (from CDF)."""
    n = 20
    zphot = np.linspace(0.05, 0.5, n)
    zphot_err = np.full(n, 0.05)
    out = generate_zphot_quantiles_gaussian(zphot, zphot_err, DECADE_QUANTILES)
    sorted_cols = sorted([k for k in out if k.startswith("ZPHOT_Q")])
    grid = np.stack([out[k] for k in sorted_cols], axis=1)
    diffs = np.diff(grid, axis=1)
    assert np.all(diffs >= 0)


def test_generate_zphot_truncated_law_has_no_atom_at_lower_bound():
    """SNANA redraws component and value while ZPHOT < 0.001 (PICKRAN loop,
    sntools_host.c:6916-6952); the accepted law is a lower-truncated mixture
    with no point mass at exactly 0.001."""
    n = 20000
    rng = np.random.default_rng(23)
    ztrue = np.full(n, 0.02)
    # Large core sigma so an untruncated draw would often fall below 0.001
    fudgepar = np.array([0.2, 0.0, 0.0, 0.0, 0.0])
    uniforms = rng.uniform(size=(n, NRAND_ZPHOT))
    zphot, _ = generate_zphot_from_calc(ztrue, _model(fudgepar), uniforms, ztrue)
    assert float(zphot.min()) >= ZPHOT_MIN
    assert int((zphot == ZPHOT_MIN).sum()) == 0


def test_generate_zphot_flat_outlier_truncated_at_lower_bound():
    """Flat outlier draws come from [max(zmin, 0.001), zmax], the accepted
    (truncated) flat law (sntools_host.c:6929-6952)."""
    n = 5000
    rng = np.random.default_rng(29)
    ztrue = np.full(n, 0.3)
    fudgepar = np.array([0.0, 0.03, 0.0, 1.0, 10.0])
    uniforms = rng.uniform(size=(n, NRAND_ZPHOT))
    zphot, _ = generate_zphot_from_calc(
        ztrue, _model(fudgepar, (0.0, 0.6)), uniforms, ztrue)
    assert float(zphot.min()) >= ZPHOT_MIN
    # Truncated-uniform mean over [0.001, 0.6]
    assert abs(float(np.mean(zphot)) - 0.5 * (ZPHOT_MIN + 0.6)) < 0.01


def test_malformed_genzphot_keys_are_fatal():
    """A FUDGEPAR without its five values and a FUDGEMAP entry that is not
    z:rms abort. SNANA misreads both: the short FUDGEPAR takes the next key
    as its fifth value and crashes (snlc_sim.c:4013-4020), and the entry
    without a colon takes the previous entry's rms (snlc_sim.c:3377-3381)."""
    ranges = ((0.1, 0.5), (0.0, 1.0))
    with pytest.raises(SNANAError, match="five values"):
        build_genzphot(Host(genzphot_fudgepar="0.0 0.03 0.0"), *ranges)
    with pytest.raises(SNANAError, match="z:rms"):
        build_genzphot(Host(genzphot_fudgemap="0.0:0.01,0.3,1.0:0.05"), *ranges)


def test_validate_genzphot_fudgemap_coverage_is_fatal():
    """A FUDGEMAP not covering GENRANGE_REDSHIFT aborts at build
    (snlc_sim.c:7568-7588)."""
    z = np.array([0.1, 0.3, 0.5])
    rms = np.array([0.01, 0.02, 0.03])
    validate_genzphot_fudgemap(z, rms, (0.1, 0.5))
    with pytest.raises(SNANAError, match="GENRANGE_REDSHIFT"):
        validate_genzphot_fudgemap(z, rms, (0.05, 0.5))
    with pytest.raises(SNANAError, match="GENRANGE_REDSHIFT"):
        validate_genzphot_fudgemap(z, rms, (0.1, 0.6))
    with pytest.raises(SNANAError, match="increasing"):
        validate_genzphot_fudgemap(z[::-1], rms, (0.2, 0.4))


def test_generate_zphot_quantiles_gaussian_custom_percentiles():
    """Quantile percentiles follow the HOSTLIB-declared set
    (sntools_host.c:2977-2986), not a fixed 11-decade list."""
    n = 10
    zphot = np.linspace(0.05, 0.5, n)
    zphot_err = np.full(n, 0.05)
    out = generate_zphot_quantiles_gaussian(
        zphot, zphot_err, [(f'ZPHOT_Q{p:03d}', float(p)) for p in (5, 25, 50, 75, 95)])
    assert set(out.keys()) == {
        'ZPHOT_Q005', 'ZPHOT_Q025', 'ZPHOT_Q050', 'ZPHOT_Q075', 'ZPHOT_Q095'}
    med = out['ZPHOT_Q050']
    np.testing.assert_allclose(med, zphot, atol=1e-6)


def test_fudgemap_validation_keeps_input_precision():
    """Host-side validation preserves ordered nodes in single-precision runs."""
    import jax

    with jax.enable_x64(False):
        validate_genzphot_fudgemap(
            np.array([0.1, 0.100000001, 0.5]), np.array([0.01, 0.02, 0.03]),
            (0.1, 0.5))
