"""ZVARIATION parity vs SNANA.

Two fixtures, two input routes, two evaluation modes:

- zvariation_ebvigm: ZVARIATION_POLY key, EBV_IGM = 0.05*z, deterministic
  MWEBV (GENSIGMA_MWEBV_RATIO 0). SNANA's dumped MWEBV matches
  simlib_MWEBV + 0.05*zCMB to 5e-7 (dump rounding), so both SNANA and jax
  are gated EXACTLY against the same closed form per SN.

- zvariation_popshift: ZVARIATION_FILE with ZBIN rows drifting the SALT2c
  peak (0.2*z) and width (both sigmas +0.1*z), the SALT2x1 peak
  (piecewise), the AV exponential tau (+0.8*z) and GENMAG_OFF_GLOBAL
  (0.3*z, gated via SIM_PEAKMAG_r). Draws are not replayable, so gating is
  statistical: per-z-bin means (and stds for the width drift), SNANA vs
  jax, within 3*sigma of the binned sampling error (pre-registered before
  jax numbers were seen), plus slope-presence guards on both sides so a
  both-sides-inactive regression cannot pass.

The in-file ZPOLY form is dead in SNANA (FLAG never set -> init abort
"Invalid FLAG=-9"); the file route is therefore exercised with ZBIN only.
"""

import numpy as np
import pytest

from jax_snlc_core.io.snana_dump_reader import load_snana_dump  # noqa: E402
from jax_snlc_sim import run as sim_run  # noqa: E402

X1_ZBINS = ([-0.01, 0.0, 0.2, 0.5], [0.0, 0.0, 0.5, 1.0])
Z_BIN_EDGES = np.linspace(0.02, 0.4, 5)


def _jax_run(fx, n_sne):
    config = dict(n_sne=n_sne)
    sim = sim_run.prepare(str(fx.input_file), **config)
    resources = sim
    results, _, _ = sim_run.run_simulation(sim)
    return resources, results


def test_zvariation_ebvigm(snana_fixture):
    fx = snana_fixture("zvariation_ebvigm")
    dump = load_snana_dump(str(fx.dump), dedup_cid=True)

    # SNANA side: dumped MWEBV is exactly simlib MWEBV + 0.05*z
    z = np.asarray(dump.columns['GENZ'])
    mw = np.asarray(dump.columns['MWEBV'])
    libid = np.asarray(dump.columns['LIBID']).astype(int)

    resources, results = _jax_run(fx, dump.size)
    params = results['parameters']
    field_mw = np.array([f.mwebv for f in resources.fields])
    libid_to_idx = {f.libid: i for i, f in enumerate(resources.fields)}

    base = field_mw[[libid_to_idx[l] for l in libid]]
    np.testing.assert_allclose(mw, base + 0.05 * z, atol=1e-6)

    # jax side: same closed form on its own draws (SIM_MWEBV == MWEBV here,
    # err is configured to zero)
    zj = np.asarray(params['sim_redshift_cmb'])
    mwj = np.asarray(params['sim_mwebv'])
    basej = field_mw[np.asarray(params['field_idx'], dtype=int)]
    np.testing.assert_allclose(mwj, basej + 0.05 * zj, atol=1e-6)
    assert float(np.max(np.abs(np.asarray(params['mwebv']) - mwj))) < 1e-6


def _binned_gate(name, stat, z_s, val_s, z_j, val_j):
    """Per-z-bin SNANA-vs-jax comparison of a statistic at 3x sampling err."""
    for lo, hi in zip(Z_BIN_EDGES[:-1], Z_BIN_EDGES[1:]):
        m_s = (z_s >= lo) & (z_s < hi)
        m_j = (z_j >= lo) & (z_j < hi)
        assert m_s.sum() > 10 and m_j.sum() > 10, (name, lo, hi)
        a, b = val_s[m_s], val_j[m_j]
        if stat == 'mean':
            diff = b.mean() - a.mean()
            tol = 3.0 * np.sqrt(a.var() / len(a) + b.var() / len(b))
        else:  # std: sampling error of std ~ std/sqrt(2N)
            diff = b.std() - a.std()
            tol = 3.0 * np.sqrt(a.var() / (2 * len(a)) + b.var() / (2 * len(b)))
        assert abs(diff) < tol, (
            f"{name} {stat} z[{lo:.2f},{hi:.2f}): jax-snana diff "
            f"{diff:+.4f} exceeds {tol:.4f}")


def _slope(z, val):
    lo = z < Z_BIN_EDGES[1]
    hi = z >= Z_BIN_EDGES[-2]
    return val[hi].mean() - val[lo].mean()


def test_zvariation_popshift(snana_fixture):
    fx = snana_fixture("zvariation_popshift")
    dump = load_snana_dump(str(fx.dump), dedup_cid=True)

    z_s = np.asarray(dump.columns['GENZ'])
    c_s = np.asarray(dump.columns['SALT2c'])
    x1_s = np.asarray(dump.columns['SALT2x1'])
    av_s = np.asarray(dump.columns['AV'])

    resources, results = _jax_run(fx, dump.size)
    params = results['parameters']
    z_j = np.asarray(params['sim_redshift_cmb'])
    c_j = np.asarray(params['c'])
    x1_j = np.asarray(params['x1'])
    av_j = np.asarray(params['av'])

    _binned_gate('SALT2c', 'mean', z_s, c_s, z_j, c_j)
    _binned_gate('SALT2c', 'std', z_s, c_s, z_j, c_j)   # width drift
    _binned_gate('SALT2x1', 'mean', z_s, x1_s, z_j, x1_j)
    _binned_gate('AV', 'mean', z_s, av_s, z_j, av_j)    # exp tau drift

    # GENMAG_OFF_GLOBAL via r-band peak mag; the z-slope is dominated by
    # distance modulus, so only the SNANA-vs-jax diff carries the gate
    # (a missing 0.3*z term is ~0.12 mag in the top bin).
    from astropy.io import fits
    with fits.open(fx.head_fits) as h:
        cid_s = np.array([int(str(c).strip()) for c in h[1].data['SNID']])
        pkr_s = np.asarray(h[1].data['SIM_PEAKMAG_r'], dtype=float)
    cids = np.asarray(dump.columns['CID']).astype(int)
    z_by_cid = dict(zip(cids, z_s))
    zh_s = np.array([z_by_cid[c] for c in cid_s])
    r_idx = list(resources.filter_names).index('LSST-r')
    pkr_j = np.asarray(results['peakmags'])[:, r_idx]
    # SNANA writes -9 when the filter has no epoch; both sides also use
    # >50 null sentinels
    ok_j = np.isfinite(pkr_j) & (pkr_j > 0) & (pkr_j < 50)
    ok_s = np.isfinite(pkr_s) & (pkr_s > 0) & (pkr_s < 50)
    _binned_gate('PEAKMAG_r', 'mean', zh_s[ok_s], pkr_s[ok_s],
                 z_j[ok_j], pkr_j[ok_j])

    # slope-presence guards: every drift must be visible on BOTH sides
    for z, c, x1, av in ((z_s, c_s, x1_s, av_s), (z_j, c_j, x1_j, av_j)):
        assert _slope(z, x1) > 0.3
        assert _slope(z, c) > 0.02
        assert _slope(z, av) > 0.1
        hi = z >= Z_BIN_EDGES[-2]
        lo = z < Z_BIN_EDGES[1]
        assert c[hi].std() > c[lo].std() * 1.2
