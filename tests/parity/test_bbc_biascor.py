"""Parity vs SALT2mu.exe on the 5D biasCor map + data-event application.

See tests/fixtures/snana/bbc_des_beams/README.md. The biasCor map is built from an
11294-event 2x2-(alpha,beta)-grid sample; the per-event biasCor columns the fit
produces are compared to the SALT2mu.exe reference.
"""


import numpy as np
import pytest

from jax_snlc_core.io.fitres_reader import read_fitres
from jax_snlc_core.models import BayeSNModel, SALTModel
from jax_snlc_core.physics.mass_step import gammadm
from jax_salt2mu.biascor_grid import biascor_gammadm_offset, build_coeff_bins
from jax_salt2mu.config import BBCConfig, resolve
from jax_salt2mu.distances import mu_at_sim
from jax_salt2mu.run import prepare_cell, run_bbc_fit
from jax_salt2mu.tablevar import EVENT_TYPE_BIASCOR, read_tablevar
from jax_snlc_core.utils.error_handling import SNANAError

_DIR = "tests/fixtures/snana/bbc_des_beams"
_REF = f"{_DIR}/SALT2mu.FITRES"

# tolerances are the FITRES output precision; x1 and muCOVSCALE are the coarse ones
_COLUMNS = (("biasCor_mB", 2e-4), ("biasCor_x1", 6e-4), ("biasCor_c", 2e-4),
            ("biasCor_mu", 2e-4), ("biasCor_muCOVSCALE", 2e-3))


def test_biascor_matches_salt2mu():
    cfg = BBCConfig(
        radiation=False,
        datafile=[f"{_DIR}/DATA.FITRES"],
        nzbin=10, zmin=0.1, zmax=1.0,
        simfile_biascor=[f"{_DIR}/BIASCOR.FITRES"], opt_biascor=112,
        min_per_cell_biascor=3, snrmin_sigint_biascor=60.0,
        surveygroup_biascor="DES(zbin=0.15:x1bin=2.0:cbin=0.2)")
    cfg.nuisance_float["alpha0"] = True
    cfg.nuisance_float["beta0"] = True
    cfg.nuisance_onebin.update(alpha0=False, beta0=False)
    _, output, _, tv, accept = run_bbc_fit(cfg)
    cids = [str(c) for c in tv.name[accept]]

    ref = read_fitres(_REF)
    ref_cid = [str(c) for c in ref.get("CID")]

    # every data event agrees on whether it gets a valid biasCor
    assert set(cids) == set(ref_cid)

    for col, tol in _COLUMNS:
        rmap = dict(zip(ref_cid, np.asarray(ref.get(col))))
        got = output.biascor[col]
        d = np.array([got[i] - rmap[c] for i, c in enumerate(cids)])
        assert np.max(np.abs(d)) < tol, f"{col} max|d|={np.max(np.abs(d)):.2e}"


def _biascor_with_gammadm(bbc_biascor_variants, tmp_path, values):
    """BIASCOR_MASS.FITRES with SIM_gammaDM rewritten to cycle ``values``.

    That sim is the one with a real host-mass spread ([8, 12] about a
    logmass_cen of 10), which the mass step needs to be sensitive to p7 and p8.

    Every bundled biasCor has SIM_gammaDM identically 0, so its axis is a point
    and GAMMADM_OFFSET is 0. A sim generated with a one-sided mass step is what
    makes the offset bite, and none is bundled.
    """
    src = (bbc_biascor_variants / "BIASCOR_MASS.FITRES").read_text().splitlines()
    out, col, n = [], None, 0
    for line in src:
        words = line.split()
        if line.startswith("VARNAMES:"):
            col = words.index("SIM_gammaDM")
        elif line.startswith("SN:") and col is not None:
            words[col] = f"{values[n % len(values)]:.5f}"
            n += 1
            line = " ".join(words)
        out.append(line)
    path = tmp_path / "BIASCOR_GAMMADM.FITRES"
    path.write_text("\n".join(out) + "\n")
    return str(path)


def _gammadm_setup(bbc_biascor_variants, tmp_path, values, float_gamma0=True, **parval):
    """The coefficient axes and the biasCor TableVar for that doctored sim.

    ``logmass_tau`` is SNANA's ``p8``; the reference config for a floated gamma0
    passes ``p7=10 p8=0.01`` and SALT2mu aborts on a tau below 0.001 with the gate
    open (:21300-21307), so the tests pass it too.
    """
    cfg = BBCConfig(
        radiation=False,
        datafile=[f"{_DIR}/DATA_MASS.FITRES"],
        nzbin=10, zmin=0.1, zmax=1.0,
        simfile_biascor=[_biascor_with_gammadm(bbc_biascor_variants, tmp_path, values)],
        opt_biascor=112, min_per_cell_biascor=3, snrmin_sigint_biascor=60.0,
        surveygroup_biascor="DES(zbin=0.15:x1bin=2.0:cbin=0.2)")
    cfg.nuisance_float.update(alpha0=True, beta0=True, gamma0=float_gamma0)
    cfg.nuisance_onebin.update(alpha0=False, beta0=False, gamma0=not float_gamma0)
    cfg.nuisance_val.update({"logmass_cen": 10.0, "logmass_tau": 0.01,
                             **parval})
    cfg = resolve(cfg, SALTModel)
    tv = read_tablevar(cfg.simfile_biascor, EVENT_TYPE_BIASCOR, cfg, SALTModel)
    return cfg, tv, build_coeff_bins(cfg, SALTModel, tv.sim_truth)


def test_gammadm_offset_is_the_biascor_axis_midpoint(bbc_biascor_variants, tmp_path):
    """GAMMADM_OFFSET is (VAL_MIN + VAL_MAX)/2 of the SIM_gammaDM biasCor axis
    (setup_CELLINFO_biasCor, SALT2mu.c:16225-16228), where VAL_MIN/VAL_MAX are the
    distinct values widened by half a bin either side (get_BININFO_biasCor_abg,
    :16406-16410). SIM_gammaDM in {0, 0.1} gives a bin of 0.1, a range of -0.05 to
    0.15, and a midpoint of 0.05.

    SNANA's USE_GAMMA0 gate returns from get_gammadm_host before subtracting the
    offset (:16776), so a gamma0 neither floated nor started off zero
    (prep_input_gamma, :22820-22828) leaves the offset out.
    """
    cfg, _, bins = _gammadm_setup(bbc_biascor_variants, tmp_path, [0.0, 0.1])
    assert biascor_gammadm_offset(SALTModel, bins, cfg) == pytest.approx(0.05)

    # symmetric about zero: the same bin, midpoint 0
    cfg_sym, _, bins_sym = _gammadm_setup(bbc_biascor_variants, tmp_path, [-0.05, 0.05])
    assert biascor_gammadm_offset(SALTModel, bins_sym, cfg_sym) == pytest.approx(0.0)

    # gamma0 held at its parval of 0: USE_GAMMA0 is off and so is the offset
    cfg_off, _, bins_off = _gammadm_setup(
        bbc_biascor_variants, tmp_path, [0.0, 0.1], float_gamma0=False)
    assert biascor_gammadm_offset(SALTModel, bins_off, cfg_off) == 0.0
    # ... but a non-zero p5 opens the gate without floating it (:22823-22826)
    cfg_p5, _, bins_p5 = _gammadm_setup(bbc_biascor_variants, tmp_path, [0.0, 0.1], float_gamma0=False,
                                        gamma0=0.02)
    assert biascor_gammadm_offset(SALTModel, bins_p5, cfg_p5) == pytest.approx(0.05)

    # a model with no mass step has no axis to take a midpoint of
    assert BayeSNModel.gammadm_coeff is None
    assert biascor_gammadm_offset(BayeSNModel, bins, cfg) == 0.0


def test_gammadm_offset_shifts_every_mu(bbc_biascor_variants, tmp_path):
    """get_gammadm_host subtracts the offset from the step (SALT2mu.c:16782) and
    muresid_biasCor adds the step to muFit (:13676), so every biasCor event's mu
    moves by exactly -GAMMADM_OFFSET. The one-sided sim above puts -0.05 mag on
    all of them.
    """
    cfg, tv, bins = _gammadm_setup(bbc_biascor_variants, tmp_path, [0.0, 0.1], gamma0=0.06)
    offset = biascor_gammadm_offset(SALTModel, bins, cfg)
    assert offset == pytest.approx(0.05)

    coeffs = SALTModel.biascor_coeffs
    with_offset = mu_at_sim(SALTModel, tv, tv.sim_truth, coeffs, cfg, offset)
    without = mu_at_sim(SALTModel, tv, tv.sim_truth, coeffs, cfg, 0.0)
    np.testing.assert_allclose(with_offset - without, -offset, rtol=0, atol=1e-12)


def test_mu_at_sim_reproduces_muresid_biascor(bbc_biascor_variants, tmp_path):
    """muresid_biasCor's muFit (SALT2mu.c:13596-13676) differs from the fit's mu on
    four counts, each pinned here.

    The event's own SIM_alpha and SIM_beta standardise the LC parameters
    (get_abg_biasCor, :13665), but the host mass step comes from the input parval,
    not from the event's SIM_gammaDM: the per-event line is commented out at
    :13622 and the hostPar block at :13628-13634 replaced it in 2019. The step is
    absent unless USE_GAMMA0 (:16776), and muFit adds it where fcn subtracts it
    (:13676 against :5477).
    """
    coeffs = SALTModel.biascor_coeffs
    # SIM_gammaDM alternates 0 and 0.1 in the sim; p5 is 0.06, a different number,
    # so the two sources cannot be confused
    cfg, tv, _ = _gammadm_setup(bbc_biascor_variants, tmp_path, [0.0, 0.1], gamma0=0.06)
    mu = mu_at_sim(SALTModel, tv, tv.sim_truth, coeffs, cfg, 0.0)

    lcpar = {n: tv.lcpar[n] for n in SALTModel.lcpar_names}
    host = {"z": tv.zhd, "gamma_var": tv.logmass}
    nuis = {n: v[0] for n, v in SALTModel.nuisance_spec.items()}
    nuis.update(alpha0=tv.sim_truth["SIM_alpha"], beta0=tv.sim_truth["SIM_beta"],
                logmass_cen=10.0, logmass_tau=0.01)
    lc_only = np.asarray(SALTModel.compute_mu(lcpar, nuis, host, cfg._u_flags,
                                              apply_mass_step=False))

    # the step is built from p5, and added
    step = gammadm(0.06, 0.0, 10.0, 0.01, tv.zhd, tv.logmass, 0.0, True)
    np.testing.assert_allclose(mu, lc_only + np.asarray(step), rtol=0, atol=1e-10)
    assert np.abs(np.asarray(step)).max() > 0.0     # the term is live

    # not from the event's SIM_gammaDM, which is a different number
    from_sim = gammadm(np.asarray(tv.sim_truth["SIM_gammaDM"]), 0.0, 10.0, 0.01,
                       tv.zhd, tv.logmass, 0.0, True)
    assert np.abs(np.asarray(step) - np.asarray(from_sim)).max() > 1.0e-3

    # p6/p7/p8 reach the step: moving logmass_cen off 10 moves mu
    cfg_cen, tv_cen, _ = _gammadm_setup(bbc_biascor_variants, tmp_path, [0.0, 0.1], gamma0=0.06,
                                        logmass_cen=8.5)
    mu_cen = mu_at_sim(SALTModel, tv_cen, tv_cen.sim_truth, coeffs, cfg_cen, 0.0)
    assert np.abs(mu_cen - mu).max() > 1.0e-3

    # USE_GAMMA0 shut: no step at all, so mu is the LC part alone. The gate reads
    # IPAR_GAMMA0 only (:21885-21889), so a p6 with no p5 and no u5 leaves the
    # whole term out where an ungated jax would apply dgamma_dz*z*(0.5 - Fermi).
    cfg_off, tv_off, _ = _gammadm_setup(
        bbc_biascor_variants, tmp_path, [0.0, 0.1], float_gamma0=False, dgamma_dz=0.05)
    mu_off = mu_at_sim(SALTModel, tv_off, tv_off.sim_truth, coeffs, cfg_off, 0.0)
    nuis_g1 = dict(nuis, dgamma_dz=0.05)
    lc_g1 = np.asarray(SALTModel.compute_mu(lcpar, nuis_g1, host, cfg._u_flags,
                                            apply_mass_step=False))
    np.testing.assert_allclose(mu_off, lc_g1, rtol=0, atol=1e-10)
    # the term the gate suppresses is not itself zero
    assert np.abs(np.asarray(
        gammadm(0.0, 0.05, 10.0, 0.01, tv.zhd, tv.logmass, 0.0,
                True))).max() > 1.0e-3


def _des_cfg():
    cfg = BBCConfig(
        radiation=False,
        datafile=[f"{_DIR}/DATA.FITRES"],
        nzbin=10, zmin=0.1, zmax=1.0,
        simfile_biascor=[f"{_DIR}/BIASCOR.FITRES"], opt_biascor=112,
        min_per_cell_biascor=3, snrmin_sigint_biascor=60.0,
        surveygroup_biascor="DES(zbin=0.15:x1bin=2.0:cbin=0.2)")
    cfg.nuisance_float["alpha0"] = True
    cfg.nuisance_float["beta0"] = True
    cfg.nuisance_onebin.update(alpha0=False, beta0=False)
    return cfg


def test_a_dust_model_biascor_takes_the_input_alpha_and_beta(caplog):
    """A biasCor whose first 20 rows hold a Ia with SIM_AV > 0 is a BS21
    dust model, whose per-event SIM_alpha and SIM_beta are intrinsic draws:
    SNANA overwrites every event's with the input's p1 and p2 before the
    maps are built (set_DUST_FLAG_biasCor, SALT2mu.c:11930-11997), and the
    fit says so. The same sample with SIM_AV = 0 keeps its own values."""
    import logging

    from jax_snlc_core.io.fitres_reader import FitresTable

    biascor = read_fitres(f"{_DIR}/BIASCOR.FITRES")
    rng = np.random.default_rng(3)
    dusty = biascor.df.copy()
    dusty["SIM_beta"] = rng.uniform(1.5, 4.5, len(dusty)).astype(np.float32)
    dusty["SIM_AV"] = rng.uniform(0.0, 0.5, len(dusty)).astype(np.float32)
    clean = dusty.copy()
    clean["SIM_AV"] = 0.0

    def coefficients(df, p1, p2):
        table = FitresTable(df=df, varnames=list(df.columns), meta=dict(biascor.meta),
                            varnames_by_file=[list(df.columns)])
        cfg = _des_cfg()
        cfg.simfile_biascor = []
        cfg.nuisance_val.update(alpha0=p1, beta0=p2)
        with caplog.at_level(logging.WARNING, logger="jax_salt2mu.run"):
            cell = prepare_cell(cfg, {EVENT_TYPE_BIASCOR: table})
        maps = cell.summary_maps
        return maps.dust_flag, maps.maps[0].bininfo[0].nbin, maps.maps[0].bininfo[1].nbin

    caplog.clear()
    dust, nalpha, nbeta = coefficients(dusty, 0.15, 3.0)
    assert dust and (nalpha, nbeta) == (1, 1)
    assert any("Detected BS21 dust model" in m and "SIM_beta = 3 (p2)" in m
               for m in caplog.messages)

    caplog.clear()
    with pytest.raises(SNANAError, match="exceed the 2-value grid bound"):
        coefficients(clean, 0.15, 3.0)
