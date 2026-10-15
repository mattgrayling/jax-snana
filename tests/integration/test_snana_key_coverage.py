"""SNANA input-key coverage: generation/HOSTLIB knobs graduated out of the
unrecognized-key warning (VPEC_FORCE, GENMAG_SMEAR_*, exposure time, GENPERFECT,
explosion anchoring, HOSTLIB position/magnitude knobs, ...).

Each runs a small sim and checks the affected truth quantity behaves as SNANA's
snlc_sim.c does (referenced per test)."""

import collections
import glob
import gzip
import io
import logging
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from astropy.io import fits

from jax_snlc_sim import run as sim_run
from replay_snana_dump import build_override_parameters
from jax_snlc_sim.builder import (
    _build_murewgt, _load_simlib, build_simulation, resolve_counts,
)
from jax_snlc_sim.config_parser import (
    _parse_simlib_nskipmjd, _parse_string_prescales, build,
    parse_snana_input_file, parse_take_spectrum_entry, prepare_spectrum_requests, read_keys,
)
from jax_snlc_sim.hostlib.hostlib_reader import essential_columns, read_hostlib
from jax_snlc_sim.population import Population
from jax_snlc_sim.spectrograph.genspec import keep_spectrum, nearpeak_mask
from jax_snlc_core.io.fits_path import open_text, resolve_fits_path
from jax_snlc_core.io.sfd_reader import SFDDustMap
from jax_snlc_core.io.snana_table import SNANA_VERSION_STRING
from jax_snlc_core.models import build_model
from jax_snlc_core.models.bayesn_model import BayeSNModel
from jax_snlc_core.models.fitted import PHOTOZ_PARAMETER
from jax_snlc_core.models.non1ased_model import NON1ASEDModel
from jax_snlc_core.models.salt_model import SALTModel
from jax_snlc_core.models.simsed_model import (
    SIMSEDModel, _draw_correlated_truncated, read_sed_binary,
)
from jax_snlc_core.physics.cosmology import (
    hubble_parameter, distance_modulus,
)
from jax_snlc_sim.rate import dndz_density_on_grid
from jax_snlc_sim.simulation import neutral_rate_mods
from jax_snlc_sim.distributions import GenGaussAsym
from jax_snlc_core.utils.error_handling import SNANAError
from jax_snlc_sim.survey.survey_def import (
    parse_survey_def, survey_from_simlib_file, survey_id,
)
from jax_snlc_fit import priors
from jax_snlc_fit import run as fit_run
from jax_snlc_fit.config import FitConfig, Fit, Model
from jax_snlc_fit.data_loader import (
    load_and_batch_snana_fits, snana_version_float, vpec_sign_is_correct,
)
from jax_snlc_fit.fit_results import BatchFitResult
from jax_snlc_core.io.fitres_reader import read_fitres
from _fit_helpers import fit_config

_SALT3 = "tests/fixtures/snana/salt3_k21_replay/input.input"
_NON1A = "tests/fixtures/snana/non1ased_v19_replay/input.input"


def _merge(base_text, extra):
    """Base text plus `extra`, with any key `extra` sets dropped from the base.
    SNANA allows each key once per file (keyMatchSim MXKEY=1)."""
    keys = {l.split(':', 1)[0] for l in extra.splitlines() if ':' in l}
    kept = [l for l in base_text.splitlines()
            if ':' not in l or l.split(':', 1)[0] not in keys]
    return "\n".join(kept) + "\n" + extra + "\n"


def _run(base_input, extra, tmp_path, n_sne=20, seed=5):
    inp = tmp_path / "in.input"
    inp.write_text(_merge(Path(base_input).read_text(), extra))
    config = dict(
        n_sne=n_sne, random_seed=seed)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    results, _, _ = sim_run.run_simulation(sim)
    return results


def _phot_by_cid(base_text, extra, tmp_path, col, tag, n_sne=20, seed=5):
    """Save a sim and return {(cid, round(mjd,2), band): col_value} for a PHOT column."""
    inp = tmp_path / f"{tag}.input"
    inp.write_text(base_text + "\n" + extra + "\n")
    config = dict(
        n_sne=n_sne, random_seed=seed)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    res, sel, spectra_cache = sim_run.run_simulation(sim)
    d = str(tmp_path / f"{tag}_out")
    os.makedirs(d, exist_ok=True)
    sim_run.save_snana_outputs(sim, res, "B", sel, spectra_cache=spectra_cache, output_dir=d)
    with fits.open(resolve_fits_path(f"{d}/B/B_HEAD.FITS")) as h:
        pmin, pmax, cid = h[1].data['PTROBS_MIN'], h[1].data['PTROBS_MAX'], h[1].data['SNID']
    with fits.open(resolve_fits_path(f"{d}/B/B_PHOT.FITS")) as h:
        v, bd, mj = h[1].data[col], h[1].data['BAND'], h[1].data['MJD']
    out = {}
    for i in range(len(pmin)):
        s = slice(int(pmin[i]) - 1, int(pmax[i]))
        for a, b, m in zip(v[s], bd[s], mj[s]):
            out[(int(str(cid[i]).strip()), round(float(m), 2), str(b).strip())] = float(a)
    return out


def test_vpec_force(tmp_path, salt3_baseline):
    """VPEC_FORCE pins every SN's vpec to the given value (snlc_sim.c:16948)."""
    r = _run(_SALT3, "VPEC_FORCE: 250.0", tmp_path)
    vpec = np.asarray(r['parameters']['vpec_true'])
    assert np.allclose(vpec, 250.0)
    # helio redshift is boosted by the forced peculiar velocity
    r0 = salt3_baseline
    assert not np.allclose(np.asarray(r['parameters']['sim_redshift_helio']),
                           np.asarray(r0['parameters']['sim_redshift_helio']))


def _run_smear(extra, tmp_path, seed=9):
    return _run(_SALT3, "GENMAG_SMEAR: 0.1\nGENMAG_SMEAR_MODELNAME: G10\n" + extra,
                tmp_path, n_sne=25, seed=seed)


@pytest.fixture(scope="module")
def salt3_baseline(tmp_path_factory):
    """_run(_SALT3, "") — the unmodified SALT3 reference. Read-only."""
    return _run(_SALT3, "", tmp_path_factory.mktemp("base"))


@pytest.fixture(scope="module")
def smear_scale_refs(tmp_path_factory):
    """GENMAG_SMEAR_SCALE 0.0 and 1.0 peakmags, shared by the two scale tests."""
    d = tmp_path_factory.mktemp("smear")
    return (np.asarray(_run_smear("GENMAG_SMEAR_SCALE: 0.0", d)['peakmags']),
            _run_smear("GENMAG_SMEAR_SCALE: 1.0", d))


def test_genmag_smear_scale_global(tmp_path, smear_scale_refs):
    """GENMAG_SMEAR_SCALE scales ONLY the GENMAG_SMEAR_MODELNAME (G10/C11) smear
    (sntools_genSmear.c:415-418) — linearly — and leaves the coherent GENMAG_SMEAR
    untouched (verified against SNANA: SIM_MAGSMEAR_COH does not scale)."""
    pk0, r1 = smear_scale_refs
    r3 = _run_smear("GENMAG_SMEAR_SCALE: 3.0", tmp_path)
    pk1, pk3 = np.asarray(r1['peakmags']), np.asarray(r3['peakmags'])
    m = (pk0 > -2) & (pk0 < 50) & (np.abs(pk1 - pk0) > 1e-3)
    assert m.any()
    # band-integrated wavelength-dependent smear is mildly nonlinear per SN, so
    # check the aggregate scale (ratio of mean |delta|) is 3x.
    ratio = np.mean(np.abs((pk3 - pk0)[m])) / np.mean(np.abs((pk1 - pk0)[m]))
    assert abs(ratio - 3.0) < 0.05
    # coherent magsmear is identical across scales (not scaled)
    assert np.allclose(np.asarray(r1['parameters']['magsmear']),
                       np.asarray(r3['parameters']['magsmear']))


def test_genmag_smear_scale_poly(tmp_path, smear_scale_refs):
    """GENMAG_SMEAR_SCALE(SALT2c) c0,c1 makes the model-smear scale = c0 + c1*c per
    SN (sntools_genSmear.c:742); poly(c)=1+c gives per-SN ratio (1+c) vs scale=1."""
    pk0, _r1 = smear_scale_refs
    pk1 = np.asarray(_r1['peakmags'])
    rp = _run_smear("GENMAG_SMEAR_SCALE(SALT2c): 1.0,1.0", tmp_path)
    pkp = np.asarray(rp['peakmags'])
    c = np.repeat(np.asarray(rp['parameters']['c'])[:, None], pkp.shape[1], axis=1)
    m = (pk0 > -2) & (pk0 < 50) & (np.abs(pk1 - pk0) > 1e-3)
    assert m.any()
    assert np.allclose((pkp - pk0)[m] / (pk1 - pk0)[m], (1.0 + c)[m], atol=0.02)
    # config carries the parsed poly
    inp = tmp_path / "p.input"
    inp.write_text(Path(_SALT3).read_text() + "\nGENMAG_SMEAR_SCALE(SALT2x1): 0.9,0.3\n")
    assert parse_snana_input_file(str(inp)).model.genmag_smear_scale_poly == ('x1', (0.9, 0.3))


def test_genmag_off_non1a(tmp_path, salt3_baseline):
    """GENMAG_OFF_NON1A shifts NON1A peak mags by the offset (snlc_sim.c:14182);
    SALT models are unaffected (the offset is NON1A-only)."""
    n0 = _run(_NON1A, "", tmp_path)
    n1 = _run(_NON1A, "GENMAG_OFF_NON1A: 0.5", tmp_path)
    a, b = np.asarray(n0['peakmags']), np.asarray(n1['peakmags'])
    m = (a > -2) & (a < 50) & (b > -2) & (b < 50)
    assert m.any()
    assert np.allclose((b - a)[m], 0.5, atol=1e-4)
    # SALT is not NON1A → unaffected
    s0 = salt3_baseline
    s1 = _run(_SALT3, "GENMAG_OFF_NON1A: 0.5", tmp_path)
    sa, sb = np.asarray(s0['peakmags']), np.asarray(s1['peakmags'])
    ms = (sa > -2) & (sa < 50) & (sb > -2) & (sb < 50)
    assert np.allclose(sa[ms], sb[ms], atol=1e-4)


def _wv07_pdf(grid, rewgt=None):
    tau, ss = 0.4, 0.01
    aexp, bexp = 1.0 / tau, 1.0 / np.sqrt(ss * 2.0 * np.pi)
    if rewgt is not None:
        aexp *= rewgt
    pdf = aexp * np.exp(-grid / tau) + bexp * np.exp(-0.5 * grid * grid / ss)
    return pdf / np.trapezoid(pdf, grid)


def test_genav_wv07(tmp_path):
    """GENAV_WV07 draws host A_V from the WV07 exponential+half-Gaussian PDF over
    GENRANGE_AV (snlc_sim.c:17953); the drawn distribution matches the analytic PDF,
    and WV07_REWGT_EXPAV up-weights the exponential tail."""
    av = np.asarray(_run(_SALT3, "GENAV_WV07: 1\nGENRANGE_AV: 0.0 3.0",
                         tmp_path, n_sne=4000, seed=11)['parameters']['av'])
    g = np.linspace(0, 3, 20001)
    pdf = _wv07_pdf(g)
    assert abs(av.mean() - np.trapezoid(g * pdf, g)) < 0.03
    assert abs(np.mean(av < 0.1) - np.trapezoid(pdf[g < 0.1], g[g < 0.1])) < 0.03
    # reweighting the exponential (x3) fattens the tail -> larger mean
    av2 = np.asarray(_run(_SALT3, "GENAV_WV07: 1\nWV07_REWGT_EXPAV: 3.0\n"
                          "GENRANGE_AV: 0.0 3.0", tmp_path, n_sne=4000, seed=11
                          )['parameters']['av'])
    assert av2.mean() > av.mean() + 0.05


def test_sim_av_rv_report_nullfloat_without_host_dust(tmp_path):
    """SIM_AV/SIM_RV are -9 unless dust was generated (snlc_sim.c:9447-9448).

    gen_modelPar_dust returns early on !DOGEN_AV (:14057), so both stay at their
    NULLFLOAT initialiser and reach SNDATA.SIM_AV/SIM_RV unchanged (:25133). The
    extinction maths still needs AV=0 and RV=3.1, so only the reported pair
    carries the sentinel. The bundled mwdust_parity HEAD.FITS, generated with no
    dust keys, holds -9 for all 1500 of its events.
    """
    no_dust = _run(_SALT3, "", tmp_path, n_sne=20, seed=7)['parameters']
    assert np.all(np.asarray(no_dust['sim_av']) == -9.0)
    assert np.all(np.asarray(no_dust['sim_rv']) == -9.0)
    # the values the extinction maths uses are untouched
    assert np.all(np.asarray(no_dust['av']) == 0.0)
    assert np.all(np.asarray(no_dust['rv']) == 3.1)

    dusty = _run(_SALT3, "GENRANGE_AV: 0.0 0.5\nGENTAU_AV: 0.4\n"
                 "GENRANGE_RV: 2.5 3.5\nGENMEAN_RV: 3.1",
                 tmp_path, n_sne=20, seed=7)['parameters']
    assert np.array_equal(np.asarray(dusty['sim_av']), np.asarray(dusty['av']))
    assert np.array_equal(np.asarray(dusty['sim_rv']), np.asarray(dusty['rv']))
    assert np.asarray(dusty['sim_av']).max() > 0.0


def test_wv07_rewgt_activates_wv07_on_its_own(tmp_path):
    """WV07_REWGT_EXPAV alone selects the WV07 A_V model (snlc_sim.c:7795).

    Supplying the reweight without GENAV_WV07 must match an explicit
    GENAV_WV07: 1 run, and must differ from the default exponential model.
    """
    rng = "GENRANGE_AV: 0.0 3.0\n"
    alone = np.asarray(_run(_SALT3, rng + "WV07_REWGT_EXPAV: 3.0",
                            tmp_path, n_sne=2000, seed=11)['parameters']['av'])
    explicit = np.asarray(_run(_SALT3, rng + "GENAV_WV07: 1\nWV07_REWGT_EXPAV: 3.0",
                               tmp_path, n_sne=2000, seed=11)['parameters']['av'])
    assert np.allclose(alone, explicit)
    # WV07's half-Gaussian core (sigma^2=0.01) piles up near AV=0, which a plain
    # exponential of the same tau does not.
    default = np.asarray(_run(_SALT3, rng + "GENTAU_AV: 0.4",
                              tmp_path, n_sne=2000, seed=11)['parameters']['av'])
    assert np.mean(alone < 0.1) > np.mean(default < 0.1) + 0.05


def _simsed_gridonly_input(tmp_path, extra, tag):
    src = Path(_SIMSED).read_text().splitlines()
    drop = ('SIMSED_PARAM', 'GENPEAK_stretch', 'GENSIGMA_stretch',
            'GENRANGE_stretch', 'GENPEAK_color', 'GENSIGMA_color', 'GENRANGE_color')
    body = "\n".join(l for l in src if not l.startswith(drop))
    inp = tmp_path / f"{tag}.input"
    inp.write_text(body + "\nSIMSED_GRIDONLY: stretch\nSIMSED_GRIDONLY: color\n"
                   "GENPEAK_stretch: 0.95\nGENSIGMA_stretch: 1.0E8 1.0E8\n"
                   "GENRANGE_stretch: 0.65 1.25\n"
                   "GENPEAK_color: 0.5\nGENSIGMA_color: 1.0E8 1.0E8\n"
                   "GENRANGE_color: 0.0 1.0\n" + extra)
    return inp


def _simsed_grid_indices(tmp_path, extra, tag, n_sne=400):
    inp = _simsed_gridonly_input(tmp_path, extra, tag)
    config = dict(
        n_sne=n_sne, random_seed=5)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    results, _, _ = sim_run.run_simulation(sim)
    gi = np.asarray(results['parameters']['grid_idx'])
    return gi, resources.model


def test_simsed_wgtmap_file_weights_surface_selection(tmp_path):
    """SIMSED_WGTMAP_FILE selects the SED surface by cumulative WGT
    (pick_SIMSED_BY_WGT, genmag_SIMSED.c:1236) instead of by the
    grid-snapped parameter draws."""
    wgt = tmp_path / "simsed.wgtmap"
    rows = ["VARNAMES_WGTMAP: stretch color WGT"]
    for st in (0.65, 0.95, 1.25):
        for c in (0.0, 0.5, 1.0):
            rows.append(f"WGT: {st:.6f} {c:.6f} {10.0 if st < 0.7 else 1.0}")
    wgt.write_text("\n".join(rows) + "\n")

    def low_stretch_fraction(extra, tag):
        gi, sed_model = _simsed_grid_indices(tmp_path, extra, tag)
        return float(np.mean(sed_model.sed_par_values[gi, 0] < 0.7))

    # a flat GENGAUSS gives the 3 stretch nodes equal weight (checkpar_SIMSED
    # widens the range by half a bin, snlc_sim.c:23314-23331); the WGTMAP
    # overrides that entirely
    snapped = low_stretch_fraction("", "sw_off")
    weighted = low_stretch_fraction(f"SIMSED_WGTMAP_FILE: {wgt}\n", "sw_on")
    assert abs(snapped - 1.0 / 3.0) < 0.08
    assert abs(weighted - 30.0 / 36.0) < 0.08


def test_simsed_wgtmap_selects_surfaces_for_continuous_model(tmp_path):
    """A SIMSED_WGTMAP_FILE on a continuous (SIMSED_PARAM) model selects
    whole surfaces by weight and bypasses interpolation (LSED_SELECT,
    genmag_SIMSED.c:1496); output parameters are the surfaces' grid values."""
    wgt = tmp_path / "simsed.wgtmap"
    rows = ["VARNAMES_WGTMAP: stretch color WGT"]
    for st in (0.65, 0.95, 1.25):
        for c in (0.0, 0.5, 1.0):
            rows.append(f"WGT: {st:.6f} {c:.6f} {10.0 if st < 0.7 else 1.0}")
    wgt.write_text("\n".join(rows) + "\n")

    inp = tmp_path / "cw.input"
    inp.write_text(Path(_SIMSED).read_text()
                   + f"\nSIMSED_WGTMAP_FILE: {wgt}\n")
    config = dict(
        n_sne=400, random_seed=5)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    results, _, _ = sim_run.run_simulation(sim)
    params = results['parameters']
    assert 'corner_weights' not in params
    gi = np.asarray(params['grid_idx'])
    sed_model = resources.model
    stretch = sed_model.sed_par_values[gi, 0]
    assert set(np.unique(stretch)) <= {0.65, 0.95, 1.25}
    assert abs(float(np.mean(stretch < 0.7)) - 30.0 / 36.0) < 0.08


def test_simsed_boundary_value_interpolates_and_outside_range_aborts(tmp_path):
    """A float32 draw of a double-precision grid edge can land one ulp
    outside the axis; SNANA treats boundary values as legal (checkpar_SIMSED
    widens its range checks, snlc_sim.c:23344-23346), so within-ulp values
    clamp to the corner, while a GENRANGE beyond the grid aborts at setup
    (snlc_sim.c:23340-23360) before any draw."""
    config = dict(
        n_sne=4, random_seed=5)
    sim = sim_run.prepare(_SIMSED, **config)
    simulation = None if sim is None else sim.config
    resources = sim
    model = resources.model
    names, axes = model.continuous_parnames, model._cont_axes
    hi = float(axes[0][-1])
    ulp = float(np.spacing(np.float32(hi)))
    vals = {n: np.full(2, float(ax[0])) for n, ax in zip(names, axes)}
    vals[names[0]] = np.array([hi + 0.5 * ulp, float(axes[0][0])])
    _, wgt = model.find_corners_and_weights(vals)
    np.testing.assert_allclose(np.asarray(wgt).sum(axis=1), 1.0, rtol=1e-5)

    span = hi - float(axes[0][0])
    lines = Path(_SIMSED).read_text().splitlines()
    key = f"GENRANGE_{names[0]}:"
    inp = tmp_path / "wide.input"
    inp.write_text("\n".join(
        f"{key} {float(axes[0][0])} {hi + 1e-3 * span}"
        if l.startswith(key) else l for l in lines) + "\n")
    with pytest.raises(SNANAError, match=f"check sim-input keyword {key[:-1]}"):
        sim_run.prepare(str(inp), **config)


def test_simsed_gridonly_sequential_processes_each_sed_once(tmp_path):
    """SIMSED_GRIDONLY: SEQUENTIAL forces NGEN to the SED count and walks
    the library in order (snlc_sim.c:13970, 23470-23478)."""
    src = Path(_SIMSED).read_text().splitlines()
    body = "\n".join(l for l in src if not l.startswith('SIMSED_PARAM'))
    inp = tmp_path / "seq.input"
    inp.write_text(body + "\nSIMSED_GRIDONLY: SEQUENTIAL\n")
    config = dict(
        n_sne=400, batch_size=4, wavelength_grid_points=32, random_seed=5)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    results, _, _ = sim_run.run_simulation(sim)
    gi = np.asarray(results['parameters']['grid_idx'])
    n_sed = len(resources.model.sed_info['sed_entries'])
    np.testing.assert_array_equal(gi, np.arange(n_sed))


def test_simsed_continuous_range_must_lie_within_the_grid(tmp_path):
    """A continuous SIMSED parameter's GENRANGE outside the model grid, or
    missing so the open init range stands, is refused at setup
    (checkpar_SIMSED, snlc_sim.c:23349-23360)."""
    src = Path(_SIMSED).read_text()
    for extra, drop, match in (
            ("GENRANGE_stretch: 0.5 1.25\n", "GENRANGE_stretch", "stretch-min = 0.500 <"),
            ("GENRANGE_color: 0.0 1.5\n", "GENRANGE_color", "color-max = 1.500 >"),
            ("", "GENRANGE_stretch", "stretch-min = -100000000.000 <")):
        inp = tmp_path / f"{drop}.input"
        inp.write_text("\n".join(l for l in src.splitlines()
                                 if not l.startswith(drop)) + "\n" + extra)
        with pytest.raises(SNANAError, match=match):
            sim_run.prepare(str(inp), **dict(n_sne=4, random_seed=5))


def test_simsed_gridonly_without_distribution_keys_is_flat_over_the_grid(tmp_path, caplog):
    """A GRIDONLY parameter with no GENGAUSS keys keeps SNANA's flat init,
    which draws every grid node with equal probability (checkpar_SIMSED,
    snlc_sim.c:23314-23331); a GENRANGE without a GENSIGMA is overridden
    the same way, and the run says so."""
    src = Path(_SIMSED).read_text().splitlines()
    drop = ('SIMSED_PARAM', 'GENPEAK_', 'GENSIGMA_', 'GENRANGE_')
    keep = [l for l in src if not (l.startswith(drop)
                                   and ('stretch' in l or 'color' in l))]
    inp = tmp_path / "bare.input"
    inp.write_text("\n".join(keep) + "\nSIMSED_GRIDONLY: stretch\n"
                   "SIMSED_GRIDONLY: color\nGENRANGE_color: 0.4 0.6\n")
    config = dict(n_sne=900, random_seed=5)
    with caplog.at_level(logging.WARNING, logger="jax_snlc_core.models.simsed_model"):
        sim = sim_run.prepare(str(inp), **config)
        simulation = None if sim is None else sim.config
        resources = sim
        results, _, _ = sim_run.run_simulation(sim)
    gi = np.asarray(results['parameters']['grid_idx'])
    counts = np.bincount(gi, minlength=9) / gi.size
    np.testing.assert_allclose(counts, 1.0 / 9.0, atol=0.04)
    assert any("GENRANGE_color 0.4 0.6 is ignored" in m for m in caplog.messages)


def test_simsed_gridonly_snaps_gengauss_draws_to_grid_nodes(tmp_path):
    """A narrow GENGAUSS on a GRIDONLY parameter lands every event on the
    nearest grid node (pick_gridval_SIMSED, snlc_sim.c:14024-14028)."""
    gi, sed_model = _simsed_grid_indices(
        tmp_path, "GENPEAK_stretch: 1.20\nGENSIGMA_stretch: 0.01 0.01\n",
        "snap", n_sne=50)
    np.testing.assert_array_equal(sed_model.sed_par_values[gi, 0], 1.25)


def test_simsed_gridonly_replay_selects_the_recorded_surface(tmp_path):
    """A GRIDONLY dump replays through the generator's own snap rule, so each
    surface's recorded parameter values map back to that surface within
    SNANA's 1e-4-of-range match window, and beyond it (or unparsable, hence
    NaN) the replay aborts (genmag_SIMSED.c:1811-1843)."""
    inp = _simsed_gridonly_input(tmp_path, "", "replay")
    config = dict(
        n_sne=4, random_seed=5)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    sed_model = resources.model
    n_sed = len(sed_model.sed_par_values)
    dump = {'CID': np.arange(1, n_sed + 1), 'GENZ': np.full(n_sed, 0.05),
            'PEAKMJD': np.full(n_sed, 60300.0), 'MU': np.full(n_sed, 36.7),
            'LIBID': np.full(n_sed, resources.fields[0].libid)}
    for i, name in enumerate(sed_model.parnames):
        # 2e-5 offset: inside the match window on both axes (ranges 0.6, 1.0)
        dump[name] = sed_model.sed_par_values[:, i] + 2e-5
    override = build_override_parameters(dump, simulation, resources, head_columns={})
    np.testing.assert_array_equal(override['grid_idx'], np.arange(n_sed))

    dump['stretch'] = dump['stretch'].copy()
    dump['stretch'][0] = 0.80
    with pytest.raises(SNANAError, match="grid node"):
        build_override_parameters(dump, simulation, resources, head_columns={})

    dump['stretch'][0] = np.nan
    with pytest.raises(SNANAError, match="grid node"):
        build_override_parameters(dump, simulation, resources, head_columns={})


def test_simsed_param_header_order_follows_the_declaration(tmp_path):
    """SIMSED_PARAM(...) header lines come in input-declaration order, not
    SED.INFO order (snlc_sim.c:5975; sntools_dataformat_text.c:452-458)."""
    src = Path(_SIMSED).read_text().replace(
        "SIMSED_PARAM:  stretch", "SIMSED_PARAM:  color").replace(
        "SIMSED_PARAM:  color\nGENPEAK_color", "SIMSED_PARAM:  stretch\nGENPEAK_color")
    config = dict(
        n_sne=4, random_seed=5)
    sim = sim_run.prepare(str(_write(tmp_path, src)), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    lines = resources.model.text_sim_header_lines(
        {}, {'simsed_stretch': np.array([1.0]), 'simsed_color': np.array([0.5])}, 0)
    named = [line for line in lines if line.startswith('SIMSED_PARAM(')]
    assert [line.split('(')[1].split(')')[0] for line in named] == ['color', 'stretch']


def test_simsed_gridonly_tie_snaps_to_the_upper_node(tmp_path):
    """An exact midpoint takes the upper node: nearest_gridval_SIMSED keeps
    the lower one only for frac < 0.5 (genmag_SEDtools.c:2425-2429)."""
    inp = _simsed_gridonly_input(tmp_path, "", "tie")
    config = dict(
        n_sne=4, random_seed=5)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    sed_model = resources.model
    idx = int(sed_model.gridonly_index(
        {'stretch': np.array([0.65], np.float32),
         'color': np.array([0.25], np.float32)})[0])
    assert sed_model.sed_par_values[idx, 0] == 0.65
    assert sed_model.sed_par_values[idx, 1] == 0.5


def test_simsed_wgtmap_rejects_non_simsed_dimension(tmp_path):
    """A WGTMAP dimension that is not a SIMSED parameter is fatal."""
    wgt = tmp_path / "bad.wgtmap"
    wgt.write_text("VARNAMES_WGTMAP: LOGMASS WGT\nWGT: 1.0 1.0\nWGT: 2.0 1.0\n")
    inp = _simsed_gridonly_input(tmp_path, f"SIMSED_WGTMAP_FILE: {wgt}\n", "sw_bad")
    with pytest.raises(SNANAError, match="not"):
        sim_run.prepare(str(inp), **dict(
            n_sne=4, random_seed=5))


def test_genmag_off_global_shifts_every_peakmag(tmp_path, salt3_baseline):
    """GENMAG_OFF_GLOBAL is a uniform gray offset added to every SN
    (snlc_sim.c:12673); it sums with the ZVARIATION term of the same name."""
    off = 0.3
    a = np.asarray(salt3_baseline['peakmags'])
    b = np.asarray(_run(_SALT3, f"GENMAG_OFF_GLOBAL: {off}", tmp_path)['peakmags'])
    m = (a > -2) & (a < 50) & (b > -2) & (b < 50)
    assert m.sum() > 10
    assert np.allclose(b[m] - a[m], off, atol=1e-3)


def test_crazy_flux_aborts_and_skip_key_allows_it(tmp_path):
    """A model driven far too bright aborts (check_crazyFlux, snlc_sim.c:28282);
    SKIP_CHECK_CRAZYFLUX generates anyway (:28304)."""

    too_bright = "GENMAG_OFF_GLOBAL: -12"
    with pytest.raises(SNANAError, match="Crazy flux"):
        _run(_SALT3, too_bright, tmp_path)

    res = _run(_SALT3, too_bright + "\nSKIP_CHECK_CRAZYFLUX: 1", tmp_path)
    assert len(np.asarray(res['parameters']['sim_redshift_cmb'])) == 20


def test_genrange_mjd_exclude_drops_observations(tmp_path):
    """Epochs inside the window never enter the cadence (snlc_sim.c:19501)."""
    def mjds(extra, tag):
        inp = tmp_path / f"{tag}.input"
        inp.write_text(Path(_SALT3).read_text() + "\n" + extra + "\n")
        sim = sim_run.prepare(str(inp), **dict(
            n_sne=3, random_seed=3))
        simulation = None if sim is None else sim.config
        resources = sim
        return np.concatenate([f.mjds for f in resources.fields])
    base = mjds("", "mjx_off")
    lo, hi = float(np.percentile(base, 40)), float(np.percentile(base, 60))
    assert ((base > lo) & (base < hi)).sum() > 0
    cut = mjds(f"GENRANGE_MJD_EXCLUDE: {lo} {hi}", "mjx_on")
    assert ((cut > lo) & (cut < hi)).sum() == 0
    assert len(cut) < len(base)


@pytest.mark.parametrize("value", ["SJPAR", "FITZ99"])
def test_extinc_hostgal_rejects_unavailable_laws(tmp_path, value):
    """CCM89 is the only usable OPT_SNXT: SNANA rejects SJPAR itself
    (genmag_extinction.c:44) and aborts on anything else (snlc_sim.c:6683)."""
    inp = tmp_path / f"x_{value}.input"
    inp.write_text(Path(_SALT3).read_text() + f"\nEXTINC_HOSTGAL: {value}\n")
    with pytest.raises(SNANAError, match="EXTINC_HOSTGAL"):
        parse_snana_input_file(str(inp))
    ok = tmp_path / "x_ok.input"
    ok.write_text(Path(_SALT3).read_text() + "\nCOLORLAW_HOSTGAL: CCM89\n")
    assert parse_snana_input_file(str(ok)) is not None


def _written_sntypes(tmp_path, extra, tag, n_sne=10):
    """SNTYPE column as written to HEAD.FITS."""
    inp = tmp_path / f"{tag}.input"
    inp.write_text(Path(_SALT3).read_text() + "\n" + extra + "\n")
    config = dict(
        n_sne=n_sne, random_seed=5)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    res, sel, spectra_cache = sim_run.run_simulation(sim)
    d = str(tmp_path / f"{tag}_out")
    os.makedirs(d, exist_ok=True)
    sim_run.save_snana_outputs(sim, res, "B", sel, spectra_cache=spectra_cache, output_dir=d)
    with fits.open(resolve_fits_path(f"{d}/B/B_HEAD.FITS")) as h:
        return set(int(t) for t in h[1].data['SNTYPE'])


def test_sntype_ia_reaches_the_output_column(tmp_path):
    """SNTYPE_Ia sets GENLC.SNTYPE (snlc_sim.c:24507), so it must reach the
    written SNTYPE column — not just the README. GENTYPE overrides it (:24515)."""
    without_pair = "\n".join(
        ln for ln in Path(_SALT3).read_text().splitlines()
        if not ln.startswith("SNTYPES_Ia:"))
    inp = tmp_path / "t.input"
    inp.write_text(without_pair + "\nSNTYPE_Ia: 120\n")
    cfg = parse_snana_input_file(str(inp))
    assert cfg.rates.sntype_spec == 120 and cfg.rates.sntype_phot == 220

    # GENTYPE: 0 disables the override (SNANA tests GENTYPE_SPEC > 0)
    assert _written_sntypes(
        tmp_path, "GENTYPE: 0\nSNTYPES_Ia: 120 220", "pair") <= {120, 220}
    # a positive GENTYPE wins over SNTYPE_Ia, matching SNANA
    assert _written_sntypes(
        tmp_path, "GENTYPE: 7\nSNTYPES_Ia: 120 220", "over") <= {7, 107}


def _written_cids(tmp_path, extra, tag, n_sne=20):
    """SNIDs as written to HEAD.FITS — CIDRAN remapping happens on save, not in
    the results dict."""
    inp = tmp_path / f"{tag}.input"
    inp.write_text(Path(_SALT3).read_text() + "\n" + extra + "\n")
    config = dict(
        n_sne=n_sne, random_seed=5)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    res, sel, spectra_cache = sim_run.run_simulation(sim)
    d = str(tmp_path / f"{tag}_out")
    os.makedirs(d, exist_ok=True)
    sim_run.save_snana_outputs(sim, res, "B", sel, spectra_cache=spectra_cache, output_dir=d)
    with fits.open(resolve_fits_path(f"{d}/B/B_HEAD.FITS")) as h:
        return set(int(c) for c in h[1].data['SNID'])


def test_cidran_skiplist_excludes_listed_cids(tmp_path):
    """CIDRAN_SKIPLIST CIDs never appear in the drawn pool (snlc_sim.c:23007)."""
    skip = [7, 11, 13, 17, 19]
    pool = "CIDRAN_MIN: 0\nCIDRAN_MAX: 29\nFORMAT_MASK: 48\n"
    with_skip = _written_cids(
        tmp_path, pool + "CIDRAN_SKIPLIST: " + ",".join(str(c) for c in skip),
        "skip")
    assert not with_skip & set(skip), \
        f"skiplist CIDs drawn: {sorted(with_skip & set(skip))}"
    # Without the skiplist the same seed draws from the full pool, so at least
    # one listed CID appears — otherwise the assertion above proves nothing.
    without = _written_cids(tmp_path, pool, "noskip")
    assert without & set(skip)


def test_cidran_never_draws_either_end_of_its_range(tmp_path):
    """init_CIDRAN redraws CIDRAN_MIN and CIDRAN_MAX (snlc_sim.c:23430-23431),
    and aborts once CIDOFF-CIDRAN_MIN plus the event count reaches CIDRAN_MAX
    (:23366). Ten events on (0, 11) can only be the CIDs 1 to 10."""
    pool = "CIDRAN_MIN: 0\nCIDRAN_MAX: 11\nFORMAT_MASK: 48\n"
    assert _written_cids(tmp_path, pool, "ends", n_sne=10) <= set(range(1, 11))
    with pytest.raises(SNANAError, match="reaches CIDRAN_MAX"):
        _written_cids(tmp_path, pool, "full", n_sne=11)


def test_simlib_fakeid_does_not_replace_the_cid(tmp_path, lsst_simlib_path):
    """A SIMLIB FAKEID never reaches the CID: SNANA writes SNID 1..10 for
    this SIMLIB, including on the entry that regenerates SALT2x1."""
    text = lsst_simlib_path.read_text()
    text = text.replace(
        "FIELD: TOY_A\n", "FIELD: TOY_A\nPEAKMJD: 61000.0  FAKEID: 777\n")
    text = text.replace(
        "FIELD: TOY_B\n", "FIELD: TOY_B\nGENRANGE_SALT2x1: -1.0 1.0  FAKEID: 888\n")
    simlib = tmp_path / "FAKEID.SIMLIB"
    simlib.write_text(text)
    results = _run(_SALT3, f"SIMLIB_FILE: {simlib}\nUSE_SIMLIB_PEAKMJD: 1\n",
                   tmp_path, n_sne=10, seed=7)
    np.testing.assert_array_equal(
        np.asarray(results['parameters']['cid']), np.arange(1, 11))


@pytest.mark.parametrize("vel_cmbapex", ["0", "0.5"])
def test_vel_cmbapex_below_one_leaves_every_redshift_heliocentric(
        tmp_path, vel_cmbapex):
    """VEL_CMBAPEX below 1 switches the dipole off, so with no redshift smear
    SNANA writes REDSHIFT_FINAL == REDSHIFT_HELIO and SIM_REDSHIFT_CMB ==
    SIM_REDSHIFT_HELIO for every event."""
    inp = tmp_path / "vcmb.input"
    inp.write_text(_merge(Path(_NON1A).read_text(),
                          f"VEL_CMBAPEX: {vel_cmbapex}\nGENSIGMA_REDSHIFT: 0\n"))
    config = dict(n_sne=40, random_seed=5)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    res, sel, spectra_cache = sim_run.run_simulation(sim)
    d = str(tmp_path / "out")
    os.makedirs(d, exist_ok=True)
    sim_run.save_snana_outputs(sim, res, "B", sel, spectra_cache=spectra_cache, output_dir=d)
    with fits.open(resolve_fits_path(f"{d}/B/B_HEAD.FITS")) as h:
        head = h[1].data
        assert len(head) > 0
        np.testing.assert_array_equal(
            head['REDSHIFT_FINAL'], head['REDSHIFT_HELIO'])
        np.testing.assert_array_equal(
            head['SIM_REDSHIFT_CMB'], head['SIM_REDSHIFT_HELIO'])


def _photflag_values(tmp_path, extra, tag, eff_file=None):
    """Distinct PHOTFLAG values written for a run, optionally with a patched
    SEARCHEFF_PIPELINE_EFF_FILE."""
    text = Path(_SALT3).read_text()
    if eff_file is not None:
        text = "\n".join(
            f"SEARCHEFF_PIPELINE_EFF_FILE:  {eff_file}"
            if ln.startswith("SEARCHEFF_PIPELINE_EFF_FILE:") else ln
            for ln in text.splitlines())
    vals = _phot_by_cid(text, extra, tmp_path, 'PHOTFLAG', tag)
    return set(int(v) for v in vals.values())


def test_photflag_input_key_overrides_searcheff_file(tmp_path):
    """PHOTFLAG_DETECT/TRIGGER: the sim-input key wins; the SEARCHEFF file fills
    in only while the value is still 0 (sntools_trigger.c:341-347); with neither
    set, SNANA OR's no bit and PHOTFLAG stays 0 (verified against snlc_sim.exe).
    """
    from_file = _photflag_values(tmp_path, "", "file")
    assert from_file & {4096}, f"expected the file's 4096 bit, got {from_file}"

    overridden = _photflag_values(
        tmp_path, "PHOTFLAG_DETECT: 8192\nPHOTFLAG_TRIGGER: 16384\n", "key")
    assert overridden & {8192}, f"input key did not win: {overridden}"
    assert not overridden & {4096}

    stripped = tmp_path / "NOPF.DAT"
    src = Path(os.path.expandvars(
        "$JAX_TEST_FIXTURE_ROOT/searcheff/SEARCHEFF_PIPELINE_LSST.DAT"))
    stripped.write_text("\n".join(
        ln for ln in src.read_text().splitlines()
        if not ln.startswith(("PHOTFLAG_DETECT:", "PHOTFLAG_TRIGGER:"))))
    assert _photflag_values(tmp_path, "", "none", eff_file=str(stripped)) == {0}


def test_pipeline_eff_file_bare_name_found_under_path_user_input(tmp_path):
    """A bare SEARCHEFF_PIPELINE_EFF_FILE name is found in PATH_USER_INPUT, as
    snana_openTextFile finds it; the file's PHOTFLAG bit reaches the output."""
    found = _photflag_values(
        tmp_path, "PATH_USER_INPUT: $JAX_TEST_FIXTURE_ROOT/searcheff\n",
        "bare", eff_file="SEARCHEFF_PIPELINE_LSST.DAT")
    assert found & {4096}, f"expected the file's 4096 bit, got {found}"


def test_pipeline_eff_file_default_is_no_file(tmp_path):
    """SEARCHEFF_PIPELINE_EFF_FILE: DEFAULT opens no file, as NONE
    (init_SEARCHEFF_PIPELINE, sntools_trigger.c:278-298): no detection is
    flagged."""
    assert _photflag_values(tmp_path, "", "default", eff_file="DEFAULT") == {0}


def _subsample_index(extra, tmp_path):
    inp = tmp_path / "in.input"
    inp.write_text(Path(_SALT3).read_text() + "\n" + extra + "\n")
    config = dict(
        n_sne=30, random_seed=1)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    res, sel, spectra_cache = sim_run.run_simulation(sim)
    sim_run.save_snana_outputs(sim, res, "NS", sel, spectra_cache=spectra_cache, output_dir=str(tmp_path))
    with fits.open(resolve_fits_path(str(tmp_path / "NS" / "NS_HEAD.FITS"))) as h:
        return np.asarray(h[1].data['SIM_SUBSAMPLE_INDEX'])


def test_nsubsample_mark(tmp_path):
    """NSUBSAMPLE_MARK: SIM_SUBSAMPLE_INDEX = NGENLC_WRITE % N over written SNe, with
    NGENLC_WRITE incremented before assignment (snlc_sim.c:30369) so it cycles
    1,2,..,N-1,0; -9 when the key is off. Byte-exact vs SNANA."""
    si = _subsample_index("NSUBSAMPLE_MARK: 4", tmp_path)
    assert np.array_equal(si, (np.arange(len(si)) + 1) % 4)
    off = _subsample_index("", tmp_path)
    assert np.all(off == -9)


_GENPOP = """MODEL_NAME: OTHER
GENPEAK_SALT2x1: -1.0
GENSIGMA_SALT2x1: 0.10 0.10
GENRANGE_SALT2x1: -3.0 3.0
GENPEAK_SALT2c: 0.0
GENSIGMA_SALT2c: 0.05 0.05
GENRANGE_SALT2c: -0.3 0.5
MODEL_END:

MODEL_NAME: TESTPOP
GENPEAK_SALT2x1: 1.5
GENSIGMA_SALT2x1: 0.10 0.10
GENRANGE_SALT2x1: -3.0 3.0
GENPEAK_SALT2c: 0.20
GENSIGMA_SALT2c: 0.05 0.05
GENRANGE_SALT2c: -0.3 0.5
MODEL_END:
"""


def test_genpop_asymgauss(tmp_path):
    """GENPOP_ASYMGAUSS: <file> <model> overrides the SALT2 x1/c population with the
    named model block's asym-Gaussian params (snlc_sim.c:6285)."""
    popf = tmp_path / "genpop.dat"
    popf.write_text(_GENPOP)
    r = _run(_SALT3, f"GENPOP_ASYMGAUSS: {popf} TESTPOP", tmp_path, n_sne=300, seed=7)
    x1, c = np.asarray(r['parameters']['x1']), np.asarray(r['parameters']['c'])
    assert abs(x1.mean() - 1.5) < 0.05 and abs(x1.std() - 0.1) < 0.03
    assert abs(c.mean() - 0.2) < 0.02 and abs(c.std() - 0.05) < 0.02
    # a different model name in the same file selects different params
    r2 = _run(_SALT3, f"GENPOP_ASYMGAUSS: {popf} OTHER", tmp_path, n_sne=300, seed=7)
    assert abs(np.asarray(r2['parameters']['x1']).mean() + 1.0) < 0.05


def test_genpop_asymgauss_accepted_for_any_genmodel(tmp_path):
    """SNANA applies GENPOP_ASYMGAUSS whatever the GENMODEL
    (parse_input_GENPOP_ASYMGAUSS has no model gate, snlc_sim.c:2267), so a
    shared include works for SALT and non-SALT versions alike."""
    popf = tmp_path / "genpop.dat"
    popf.write_text(_GENPOP)
    src = Path("tests/fixtures/snana/bayesn_w22/input.input").read_text()
    cfg = parse_snana_input_file(
        str(_write(tmp_path, src + f"\nGENPOP_ASYMGAUSS: {popf} TESTPOP\n")))
    # a SALT key in a BayeSN run is another model's, kept as spelt
    assert cfg.model.populations['SALT2x1'].distributions[0].peak == 1.5


_USESIMLIB = "$JAX_TEST_FIXTURE_ROOT/simlib/JAX_TEST_LSST_USESIMLIB.SIMLIB"


@pytest.mark.parametrize("layout", ["one key per line", "keys sharing lines"])
def test_use_simlib_truth_overrides(tmp_path, layout):
    """USE_SIMLIB_REDSHIFT/PEAKMJD/SALT2 pin zCMB, peakMJD and SALT2 x1/c to the
    per-LIBID SIMLIB header values (snlc_sim.c:21977). The fixture has two LIBIDs
    with distinct truth values. SNANA reads the header as a token stream, so
    the keys reach the run whether each has its own line or shares one, as in
    the DES-SN5YR low-z library's 'NOBS: 310 PIXSIZE: 0.5 REDSHIFT: 0.031
    PEAKMJD: 53265.094' (snlc_sim.c:19380-19445)."""
    simlib = Path(os.path.expandvars(_USESIMLIB)).read_text()
    if layout == "keys sharing lines":
        simlib = re.sub(r"NOBS: (\S+)  MWEBV: (\S+)  PIXSIZE: (\S+)  FIELD: (\S+)\n"
                        r"REDSHIFT: (\S+)\nPEAKMJD: (\S+)\nSALT2x1: (\S+)\nSALT2c: (\S+)\n",
                        r"MWEBV: \2  FIELD: \4\nNOBS: \1  PIXSIZE: \3  REDSHIFT: \5  "
                        r"PEAKMJD: \6\nSALT2x1: \7  SALT2c: \8\n", simlib)
        assert simlib.count("PIXSIZE: 0.200  REDSHIFT:") == 2
    (tmp_path / "T.SIMLIB").write_text(simlib)
    src = Path(_SALT3).read_text()
    src = re.sub(r"SIMLIB_FILE:.*", f"SIMLIB_FILE: {tmp_path / 'T.SIMLIB'}", src)
    src += ("\nUSE_SIMLIB_REDSHIFT: 1\nUSE_SIMLIB_PEAKMJD: 1\nUSE_SIMLIB_SALT2: 1\n")
    inp = tmp_path / "in.input"
    inp.write_text(src)
    config = dict(
        n_sne=40, random_seed=3)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    results, _, _ = sim_run.run_simulation(sim)
    p = results['parameters']
    z = set(np.round(np.asarray(p['sim_redshift_cmb']), 3).tolist())
    pk = set(np.round(np.asarray(p['peakmjd_true']), 1).tolist())
    x1 = set(np.round(np.asarray(p['x1']), 2).tolist())
    c = set(np.round(np.asarray(p['c']), 2).tolist())
    assert z == {0.15, 0.35}
    assert pk == {60600.0, 60900.0}
    assert x1 == {1.2, -1.0}
    assert c == {0.05, 0.15}


def test_hostlib_need_not_span_a_simlib_redshift(tmp_path):
    """The HOSTLIB must span GENRANGE_REDSHIFT (check_redshift_HOSTLIB,
    sntools_host.c:3322) unless USE_SIMLIB_REDSHIFT pins the redshift to
    the SIMLIB header instead (:3320): the bundled catalogue stops at z=0.5,
    and a range to 0.8 aborts only while the redshift is drawn from it."""
    src = Path(_HOSTLIB).read_text()
    src = re.sub(r"SIMLIB_FILE:.*",
                 f"SIMLIB_FILE: {os.path.expandvars(_USESIMLIB)}", src)
    src = re.sub(r"GENRANGE_REDSHIFT:.*", "GENRANGE_REDSHIFT: 0.05 0.80", src)
    src = "".join(l for l in src.splitlines(keepends=True)
                  if not l.startswith("SIMGEN_DUMP"))
    inp = _write(tmp_path, src)
    with pytest.raises(SNANAError, match="does not contain user GENRANGE_REDSHIFT"):
        sim_run.prepare(str(inp), **dict(n_sne=10, random_seed=3))
    inp = _write(tmp_path, src + "\nUSE_SIMLIB_REDSHIFT: 1\n")
    sim = sim_run.prepare(str(inp), **dict(n_sne=10, random_seed=3))
    simulation = None if sim is None else sim.config
    resources = sim
    assert simulation.survey.use_simlib_redshift == 1


def test_simlib_entry_header_keys_match_snana(tmp_path):
    """Per-LIBID header keys, against the bundled SNANA run on the same SIMLIB:
    CUTWIN_REDSHIFT keeps an entry for the redshifts inside its window
    (keep_SIMLIB_HEADER, snlc_sim.c:19808), GENRANGE_REDSHIFT redraws z inside
    the entry's window (:21983), GENRANGE/GENSIGMA_PEAKMJD is a Gaussian about
    the midpoint whatever USE_SIMLIB_PEAKMJD says (:21797, :22000), SUBSURVEY
    reaches the HEAD column and the TEXT survey line, and APPEND's flag reaches
    PHOTFLAG (:25279). SNANA walks the entries and jax draws them, so the
    split between LIBIDs is compared as a count, everything else exactly."""
    fixture = Path("tests/fixtures/snana/simlib_windows")
    ref = fits.getdata(resolve_fits_path(fixture / "HEAD.FITS"), 1)
    ref_lib = ref['SIM_LIBID']
    config = dict(n_sne=600, random_seed=3)
    sim = sim_run.prepare(str(fixture / "input.input"), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    results, selection, spectra_cache = sim_run.run_simulation(sim)
    sim_run.save_snana_outputs(sim, results, "J", selection, spectra_cache=spectra_cache, output_dir=str(tmp_path))
    head = fits.getdata(resolve_fits_path(next((tmp_path / "J").glob("*HEAD.FITS*"))), 1)
    phot = fits.getdata(resolve_fits_path(next((tmp_path / "J").glob("*PHOT.FITS*"))), 1)
    lib, z, pk = head['SIM_LIBID'], head['SIM_REDSHIFT_CMB'], head['SIM_PEAKMJD']

    n0 = int((lib == 0).sum())
    assert abs(n0 - int((ref_lib == 0).sum())) < 3 * np.sqrt(n0)
    assert z[lib == 0].max() <= 0.20
    assert 0.25 <= z[lib == 1].min() and z[lib == 1].max() <= 0.35
    assert 61300 <= pk[lib == 0].min() and pk[lib == 0].max() <= 61400
    ref_std = ref['SIM_PEAKMJD'][ref_lib == 0].std()
    assert abs(pk[lib == 0].std() - ref_std) < 0.3 * ref_std
    assert set(head['SUBSURVEY'][lib == 0]) == {'SDSS'}
    assert set(head['SUBSURVEY'][lib == 1]) == {''}
    rows = np.concatenate([np.arange(a - 1, b)
                           for a, b in zip(head['PTROBS_MIN'], head['PTROBS_MAX'])])
    photflag = phot['PHOTFLAG'][rows]
    assert (photflag & 16384 > 0).sum() > 0
    appended = np.repeat(lib == 0, head['NOBS'])
    assert not (photflag[~appended] & 16384).any()

    # TEXT: the survey line names the subsurvey
    config = dict(n_sne=40, random_seed=3)
    src = (fixture / "input.input").read_text().replace("FORMAT_MASK:  32",
                                                        "FORMAT_MASK:  2")
    inp = tmp_path / "text.input"
    inp.write_text(src)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    results, selection, spectra_cache = sim_run.run_simulation(sim)
    sim_run.save_snana_outputs(sim, results, "T", selection, spectra_cache=spectra_cache, output_dir=str(tmp_path))
    surveys = {}
    for dat in (tmp_path / "T").glob("*.DAT"):
        text = dat.read_text()
        libid = re.search(r"^SIM_LIBID:\s+(\d+)", text, re.M).group(1)
        surveys[libid] = re.search(r"^SURVEY:\s+(\S+)", text, re.M).group(1)
    assert surveys == {'0': 'LSST(SDSS)', '1': 'LSST'}


def test_searcheff_pipeline_file_is_optional_under_the_pipeline_bit(tmp_path):
    """APPLY_SEARCHEFF_OPT with the pipeline bit but no
    SEARCHEFF_PIPELINE_EFF_FILE loads no maps and marks every event
    pipeline-found, so selection is the spec efficiency alone
    (init_SEARCHEFF_PIPELINE, sntools_trigger.c:296-330; gen_SEARCHEFF,
    :2113-2120), as the DES-SN5YR low-z input relies on. A named file that
    does not exist still aborts."""
    base = "\n".join(
        l for l in Path(_SEARCHEFF_MIXED).read_text().splitlines()
        if not l.startswith(("SEARCHEFF_PIPELINE_EFF_FILE", "APPLY_SEARCHEFF_OPT",
                             "SEARCHEFF_zHOST_FILE")))
    masks = {}
    for opt in (3, 2):
        inp = tmp_path / f"opt{opt}.input"
        inp.write_text(base + f"\nAPPLY_SEARCHEFF_OPT: {opt}\n")
        config = dict(n_sne=200, random_seed=5)
        sim = sim_run.prepare(str(inp), **config)
        simulation = None if sim is None else sim.config
        resources = sim
        _, sel, _ = sim_run.run_simulation(sim)
        masks[opt] = np.asarray(sel['sim_searcheff_mask'])
        assert 0 < int(sel['n_passed']) < 200
    np.testing.assert_array_equal(masks[3], masks[2])
    assert np.all(masks[3] & 1), "every event is pipeline-found without maps"

    inp = tmp_path / "missing.input"
    inp.write_text(base + "\nAPPLY_SEARCHEFF_OPT: 3\n"
                   "SEARCHEFF_PIPELINE_EFF_FILE: /nowhere/SEARCHEFF_PIPELINE.DAT\n")
    with pytest.raises(SNANAError, match="SEARCHEFF_PIPELINE_EFF_FILE "
                       "/nowhere/SEARCHEFF_PIPELINE.DAT not found"):
        sim_run.prepare(str(inp), **dict(n_sne=4, random_seed=5))


def test_searcheff_magshift_primary_keynames(tmp_path):
    """SEARCHEFF_SPEC_MAGSHIFT / SEARCHEFF_zHOST_MAGSHIFT are accepted as the
    primary spellings of MAGSHIFT_SPECEFF / MAGSHIFT_zHOSTEFF (snlc_sim.c:3176)."""
    inp = tmp_path / "s.input"
    inp.write_text(Path(_SALT3).read_text()
                   + "\nSEARCHEFF_SPEC_MAGSHIFT: 0.3\nSEARCHEFF_zHOST_MAGSHIFT: -0.2\n")
    cfg = parse_snana_input_file(str(inp))
    assert cfg.survey.searcheff_spec_magshift == 0.3
    assert cfg.survey.searcheff_zhost_magshift == -0.2
    # legacy alias still works
    inp2 = tmp_path / "s2.input"
    inp2.write_text(Path(_SALT3).read_text() + "\nMAGSHIFT_SPECEFF: 0.15\n")
    assert parse_snana_input_file(str(inp2)).survey.searcheff_spec_magshift == 0.15


def test_ngen_scale_non1a(tmp_path):
    """NGEN_SCALE_NON1A scales the generated count for NON1A/SIMSED only
    (snlc_sim.c:6974); accepts the NGEN_SCALE_NONIA alias."""
    def nsim(fixture, extra):
        inp = tmp_path / "n.input"
        inp.write_text(Path(f"tests/fixtures/snana/{fixture}/input.input").read_text()
                       + "\n" + extra + "\n")
        cfg = parse_snana_input_file(str(inp))
        return resolve_counts(cfg, cfg.hz_map())[0]
    base = nsim("non1ased_v19_replay", "")
    assert nsim("non1ased_v19_replay", "NGEN_SCALE_NON1A: 2.0") == 2 * base
    assert nsim("non1ased_v19_replay", "NGEN_SCALE_NONIA: 3.0") == 3 * base
    # SALT is unaffected
    salt = nsim("salt3_k21_replay", "")
    assert nsim("salt3_k21_replay", "NGEN_SCALE_NON1A: 2.0") == salt


def test_use_simlib_distance(tmp_path):
    """USE_SIMLIB_DISTANCE: header DISTANCE (Mpc) -> MU -> zCMB (snlc_sim.c:21765)."""
    om, ol, hd = 0.315, 0.685, 299792.458 / 70.0
    dl = {0: 10 ** ((float(distance_modulus(0.20, 0.20, om, ol, hd)) - 25) / 5),
          1: 10 ** ((float(distance_modulus(0.35, 0.35, om, ol, hd)) - 25) / 5)}
    src = Path("tests/fixtures/data/simlib/JAX_TEST_LSST.SIMLIB").read_text()
    out, lib = [], None
    for line in src.splitlines():
        out.append(line)
        if line.startswith("LIBID:"):
            lib = int(line.split()[1])
        if line.startswith("RA:") and lib in dl:
            out.append(f"DISTANCE: {dl[lib]:.4f}")
    simlib = tmp_path / "dist.SIMLIB"
    simlib.write_text("\n".join(out) + "\n")
    base = re.sub(r"SIMLIB_FILE:.*", f"SIMLIB_FILE: {simlib}", Path(_SALT3).read_text())
    inp = tmp_path / "d.input"
    inp.write_text(base + "\nUSE_SIMLIB_DISTANCE: 1\n")
    config = dict(
        n_sne=40, random_seed=3)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    results, _, _ = sim_run.run_simulation(sim)
    z = np.asarray(results['parameters']['sim_redshift_cmb'])
    assert set(np.round(z.astype(np.float64), 2).tolist()) == {0.2, 0.35}


def test_genmag_off_model(tmp_path):
    """GENMAG_OFF_MODEL shifts observed mags per filter (snlc_sim.c:8344); the
    per-band SIM_MAGOBS shift equals the configured offset."""
    base = Path(_SALT3).read_text()  # GENFILTERS ugrizY
    m0 = _phot_by_cid(base, "", tmp_path, 'SIM_MAGOBS', "a")
    m1 = _phot_by_cid(base, "GENMAG_OFF_MODEL: 0.1 0.2 0.3 0.4 0.5 0.6", tmp_path, 'SIM_MAGOBS', "b")
    exp = dict(zip("ugrizY", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    common = [k for k in set(m0) & set(m1) if -2 < m0[k] < 50 and -2 < m1[k] < 50]
    assert common
    by_band = {}
    for k in common:
        by_band.setdefault(k[2].split('-')[-1], []).append(m1[k] - m0[k])
    for b, shifts in by_band.items():
        assert abs(np.mean(shifts) - exp[b]) < 1e-3, b


def test_applyflag_mwebv(tmp_path):
    """APPLYFLAG_MWEBV de-reddens the written FLUXCAL for MW extinction
    (snlc_sim.c:25506); the per-band correction factor is >1 and wavelength-ordered
    (bluer bands corrected more). Byte-exact vs SNANA (see live test)."""
    base = re.sub(r"OPT_MWEBV:.*", "OPT_MWEBV: 1\nGENSIGMA_MWEBV_RATIO: 0.16", Path(_SALT3).read_text())  # SIMLIB MWEBV=0.02
    f0 = _phot_by_cid(base, "", tmp_path, 'FLUXCAL', "a")
    f1 = _phot_by_cid(base, "APPLYFLAG_MWEBV: 1", tmp_path, 'FLUXCAL', "b")
    common = [k for k in set(f0) & set(f1) if abs(f0[k]) > 50]
    by_band = {}
    for k in common:
        by_band.setdefault(k[2].split('-')[-1], []).append(f1[k] / f0[k])
    ratios = {b: float(np.mean(v)) for b, v in by_band.items()}
    assert all(r > 1.0 for r in ratios.values())          # de-reddened (brighter)
    assert ratios['u'] > ratios['g'] > ratios['r'] > ratios['i']  # bluer corrected more
    assert parse_snana_input_file(_SALT3).mw_dust.applyflag_mwebv == 0   # off by default


def test_dump_mwebv_columns(tmp_path):
    """DUMP MWEBV is the applied (smeared) value GENLC.MWEBV_SMEAR; MWEBVMAP
    carries the map value and MWEBVERR its error (snlc_sim.c:15785-15803).
    Cross-checked against the HEAD columns, whose MWEBV is the map value and
    SIM_MWEBV the applied one (snlc_sim.c:25128, :25200).

    SNANA also registers MWEBV_TRUE for the map value; jax takes one spelling
    per column and rejects the other."""
    base = re.sub(r"OPT_MWEBV:.*", "OPT_MWEBV: 1\nGENSIGMA_MWEBV_RATIO: 0.16", Path(_SALT3).read_text())
    base = "\n".join(l for l in base.splitlines()
                     if not l.startswith("SIMGEN_DUMPALL"))
    inp = tmp_path / "mwdump.input"
    inp.write_text(_merge(
        base, "SIMGEN_DUMP: 4 CID MWEBV MWEBVMAP MWEBVERR"))
    config = dict(
        n_sne=40, random_seed=5)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    res, sel, spectra_cache = sim_run.run_simulation(sim)
    out = str(tmp_path / "out")
    os.makedirs(out, exist_ok=True)
    sim_run.save_snana_outputs(sim, res, "MW", sel, spectra_cache=spectra_cache, output_dir=out)
    lines = Path(out, "MW", "MW.DUMP").read_text().splitlines()
    varnames = next(l.split()[1:] for l in lines if l.startswith("VARNAMES:"))
    rows = [l.split()[1:] for l in lines if l.startswith("SN:")]
    col = {n: np.array([float(r[i]) for r in rows])
           for i, n in enumerate(varnames)}
    with fits.open(resolve_fits_path(
            os.path.join(out, "MW", "MW_HEAD.FITS"))) as h:
        head = {n: np.asarray(h[1].data[n])
                for n in ("SNID", "MWEBV", "MWEBV_ERR", "SIM_MWEBV")}
    hid = {int(v): i for i, v in enumerate(head['SNID'])}
    idx = np.array([hid[int(c)] for c in col['CID']])
    assert np.allclose(col['MWEBV'], head['SIM_MWEBV'][idx], atol=1e-6)
    assert np.allclose(col['MWEBVMAP'], head['MWEBV'][idx], atol=1e-6)
    assert np.allclose(col['MWEBVERR'], head['MWEBV_ERR'][idx], atol=1e-6)
    assert not np.allclose(col['MWEBV'], col['MWEBVMAP'])  # smear applied


def test_dump_string_column(tmp_path):
    """NON1ASED's SIM_TYPE_NAME column is a string; the dump writes it as
    text, as the HEAD file carries it, with the format following the dtype."""
    base = "\n".join(l for l in Path(_NON1A).read_text().splitlines()
                     if not l.startswith("SIMGEN_DUMP"))
    inp = tmp_path / "typename.input"
    inp.write_text(_merge(base, "SIMGEN_DUMP: 2 CID SIM_TYPE_NAME"))
    config = dict(n_sne=20, random_seed=5)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    res, sel, spectra_cache = sim_run.run_simulation(sim)
    out = str(tmp_path / "out")
    os.makedirs(out, exist_ok=True)
    sim_run.save_snana_outputs(sim, res, "TN", sel, spectra_cache=spectra_cache, output_dir=out)
    lines = Path(out, "TN", "TN.DUMP").read_text().splitlines()
    varnames = next(l.split()[1:] for l in lines if l.startswith("VARNAMES:"))
    rows = [l.split()[1:] for l in lines if l.startswith("SN:")]
    names = [r[varnames.index("SIM_TYPE_NAME")] for r in rows]
    with fits.open(resolve_fits_path(
            os.path.join(out, "TN", "TN_HEAD.FITS"))) as h:
        head_names = {int(s): str(n).strip() for s, n in
                      zip(h[1].data["SNID"], h[1].data["SIM_TYPE_NAME"])}
    assert names == [head_names[int(r[varnames.index("CID")])] for r in rows]


def test_genpdf_flat(tmp_path):
    """GENPDF_FLAT: VAR(min:max) forces a uniform population over the range
    (prep_GENPDF_FLAT, snlc_sim.c:7866; SIGMA_FLAT=1e6). Here SALT2c(-0.3:0.3)."""
    c = np.asarray(_run(_SALT3, "GENPDF_FLAT: SALT2c(-0.3:0.3)",
                        tmp_path, n_sne=400)['parameters']['c'])
    assert c.min() >= -0.3 - 1e-4 and c.max() <= 0.3 + 1e-4      # bounded to range
    assert (c.max() - c.min()) > 0.5                             # spans the range
    assert 0.14 < c.std() < 0.20                                 # uniform std = 0.6/sqrt(12)
    assert abs(c.mean()) < 0.04                                  # symmetric about midpoint


def test_genpdf_ignore(tmp_path):
    """GENPDF_IGNORE drops the named GENPDF map so the variable falls back to its
    GENGAUSS population (sntools_genPDF.c). GENPDF_FLAT implies IGNORE."""
    pdf = tmp_path / "c.genpdf"
    rows = ["VARNAMES: SALT2c PROB"]
    rows += [f"PDF: {v:+.3f} 1.0" for v in np.linspace(-0.3, 0.5, 17)]
    pdf.write_text("\n".join(rows) + "\n")

    def maps(extra):
        inp = tmp_path / "in.input"
        inp.write_text(Path(_SALT3).read_text() + f"\nGENPDF_FILE: {pdf}\n" + extra + "\n")
        return {n for n, p in parse_snana_input_file(
            str(inp)).model.populations.items() if p.genpdf is not None}

    assert 'c' in maps("")                    # loaded from GENPDF_FILE
    assert 'c' not in maps("GENPDF_IGNORE: SALT2c")


def _simlib_band(extra, tmp_path, tag, band):
    """Load the fixture SIMLIB with `extra` and return the field-0 per-obs
    (skysig, readnoise, zptavg) arrays for one band."""
    inp = tmp_path / f"{tag}.input"
    inp.write_text(Path(_SALT3).read_text() + "\n" + extra + "\n")
    sl = _load_simlib(parse_snana_input_file(str(inp)))
    f = sl.fields[0]
    m = f.filter_indices == sl.filter_names.index(band)
    return f.skysigs[m], f.readnoises[m], f.zptavgs[m]


def test_exposure_time_scaling(tmp_path):
    """EXPOSURE_TIME_FILTER scales per-band noise/ZPT: SCALE_NOISE=sqrt(ratio),
    SHIFT_ZPT=2.5*log10(ratio) (get_SIMLIB_SCALES, snlc_sim.c:7040/22254)."""
    sky0, read0, zpt0 = _simlib_band("", tmp_path, "a", "g")
    r_sky0, _, r_zpt0 = _simlib_band("", tmp_path, "a", "r")
    sky4, read4, zpt4 = _simlib_band("EXPOSURE_TIME_FILTER: g 4.0", tmp_path, "b", "g")
    r_sky4, _, r_zpt4 = _simlib_band("EXPOSURE_TIME_FILTER: g 4.0", tmp_path, "b", "r")
    assert np.allclose(sky4, sky0 * 2.0)                       # sqrt(4)
    assert np.allclose(read4, read0 * 2.0)
    assert np.allclose(zpt4, zpt0 + 2.5 * np.log10(4.0))
    assert np.allclose(r_sky4, r_sky0) and np.allclose(r_zpt4, r_zpt0)  # other bands untouched


def test_exposure_time_global(tmp_path):
    """Bare EXPOSURE_TIME broadcasts the ratio to all bands (KEY_PLUS_FILTER)."""
    sky0, _, zpt0 = _simlib_band("", tmp_path, "a", "r")
    sky4, _, zpt4 = _simlib_band("EXPOSURE_TIME: 4.0", tmp_path, "g", "r")
    assert np.allclose(sky4, sky0 * 2.0)
    assert np.allclose(zpt4, zpt0 + 2.5 * np.log10(4.0))


def test_exposure_time_is_written_as_sim_exposure(tmp_path):
    """SIM_EXPOSURE_<band> carries the band's EXPOSURE_TIME_FILTER ratio
    (snlc_sim.c:25536), 1 for a band it does not name."""
    inp = tmp_path / "x.input"
    inp.write_text(Path(_SALT3).read_text() + "\nEXPOSURE_TIME_FILTER: g 4.0\n")
    config = dict(n_sne=20, random_seed=5)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    res, sel, spectra_cache = sim_run.run_simulation(sim)
    d = str(tmp_path / "x_out")
    os.makedirs(d, exist_ok=True)
    sim_run.save_snana_outputs(sim, res, "B", sel, spectra_cache=spectra_cache, output_dir=d)
    with fits.open(resolve_fits_path(f"{d}/B/B_HEAD.FITS")) as h:
        head = h[1].data
        assert len(head) > 0
        np.testing.assert_array_equal(head['SIM_EXPOSURE_g'], 4.0)
        np.testing.assert_array_equal(head['SIM_EXPOSURE_r'], 1.0)


def test_exposure_time_mskopt_gating(tmp_path):
    """EXPOSURE_TIME_MSKOPT bits independently gate ZPT(1)/SKYSIG(2)/READNOISE(4)."""
    sky0, read0, zpt0 = _simlib_band("", tmp_path, "a", "g")
    # bit 0 only: ZPT shifts, sky/read unchanged
    sky_z, read_z, zpt_z = _simlib_band(
        "EXPOSURE_TIME_FILTER: g 4.0\nEXPOSURE_TIME_MSKOPT: 1", tmp_path, "z", "g")
    assert np.allclose(sky_z, sky0) and np.allclose(read_z, read0)
    assert np.allclose(zpt_z, zpt0 + 2.5 * np.log10(4.0))
    # bit 1 only: SKYSIG scales, read/ZPT unchanged
    sky_s, read_s, zpt_s = _simlib_band(
        "EXPOSURE_TIME_FILTER: g 4.0\nEXPOSURE_TIME_MSKOPT: 2", tmp_path, "s", "g")
    assert np.allclose(sky_s, sky0 * 2.0) and np.allclose(read_s, read0)
    assert np.allclose(zpt_s, zpt0)


# GENPERFECT: a baseline with every override target turned ON, so each ideal-mode
# override is observable.
_GP_ON = ("OPT_MWEBV: 1\nGENSIGMA_MWEBV_RATIO: 0.16\nGENMAG_SMEAR: 0.1\nGENRANGE_AV: 0.0 3.0\nGENTAU_AV: 0.4\n"
          "GENSIGMA_VPEC: 300\nHOSTLIB_FILE: /fake/host.HOSTLIB")


def _gp_cfg(extra, tmp_path, tag):
    inp = tmp_path / f"{tag}.input"
    inp.write_text(Path(_SALT3).read_text() + "\n" + _GP_ON + "\n" + extra + "\n")
    return parse_snana_input_file(str(inp))


def test_genperfect_all(tmp_path):
    """GENPERFECT: 1 forces every ideal-mode override (genperfect_override,
    snlc_sim.c:8422): no noise/smear/dust/HOSTLIB, pinned z/vpec/peakmjd."""
    c0 = _gp_cfg("", tmp_path, "on")
    assert c0.mw_dust.opt_mwebv == 1 and c0.model.genmag_smear == 0.1  # baseline is ON
    assert c0.host.av_range == (0.0, 3.0) and c0.host.hostlib_file
    c = _gp_cfg("GENPERFECT: 1", tmp_path, "gp1")
    assert c.mw_dust.opt_mwebv == 0                              # no MW dust
    assert c.host.av_range == (0.0, 0.0) and c.host.tau_av == 0.0  # AV=0
    assert c.model.genmag_smear == 0.0 and c.model.genmag_smear_modelname is None
    assert set(c.survey.exposure_time.values()) == {1.0e5}             # noise -> 0
    assert c.cosmology.gensigma_vpec == 0.0 and c.rates.gensigma_redshift == 1e-5
    assert c.rates.gensigma_peakmjd == 0.1 and c.rates.opt_setpkmjd == 0
    assert c.host.hostlib_file is None                               # no HOSTLIB


def test_genperfect_bit_isolation(tmp_path):
    """Each GENPERFECT bit gates only its own overrides (mandatory z/vpec pins
    still apply for any GENPERFECT>0)."""
    base = _gp_cfg("", tmp_path, "gp8base")                     # no GENPERFECT
    x = _gp_cfg("GENPERFECT: 8", tmp_path, "gp8")               # XTMW only
    assert x.mw_dust.opt_mwebv == 0                             # MW off
    assert x.model.genmag_smear == 0.1 and x.host.av_range == (0.0, 3.0)  # smear/AV untouched
    assert x.survey.exposure_time == base.survey.exposure_time               # exposure untouched
    s = _gp_cfg("GENPERFECT: 4", tmp_path, "gp4")               # MAGSMEAR only
    assert s.model.genmag_smear == 0.0 and s.mw_dust.opt_mwebv == 1   # smear off, MW untouched


def test_genperfect_off(tmp_path):
    """GENPERFECT: 0 changes nothing."""
    c = _gp_cfg("GENPERFECT: 0", tmp_path, "gp0")
    assert c.mw_dust.opt_mwebv == 1 and c.model.genmag_smear == 0.1
    assert c.host.av_range == (0.0, 3.0) and c.host.hostlib_file


def test_genperfect_no_warning(tmp_path, caplog):
    """GENPERFECT is consumed and its overrides keep the overridden keys 'used':
    no unrecognized-key warning."""
    inp = tmp_path / "in.input"
    inp.write_text(Path(_SALT3).read_text() + "\n" + _GP_ON + "\nGENPERFECT: 1\n")
    with caplog.at_level(logging.WARNING, logger="jax_snlc_sim.config_parser"):
        parse_snana_input_file(str(inp))
    assert "Unrecognized SNANA input key" not in caplog.text


def test_gensmear_rangauss_fix(tmp_path):
    """GENSMEAR_RANGauss_FIX pins the G10/C11 gensmear randoms to a constant
    (sntools_genSmear.c:307) — seed-independent."""
    def gsr(extra, seed):
        r = _run(_SALT3, "GENMAG_SMEAR: 0.1\nGENMAG_SMEAR_MODELNAME: G10\n" + extra,
                 tmp_path, n_sne=30, seed=seed)
        return np.asarray(r['parameters']['_gensmear_randoms'])
    assert np.allclose(gsr("GENSMEAR_RANGauss_FIX: 0.7", 3), 0.7)
    assert np.allclose(gsr("GENSMEAR_RANGauss_FIX: 0.7", 99), 0.7)   # different seed, same
    assert not np.allclose(gsr("", 3), 0.7)                          # unpinned = random


def test_salt2beta_cpoly(tmp_path):
    """SALT2BETA_cPOLY: beta = base + poly(c) per SN (gen_modelPar_SALT2,
    snlc_sim.c:13791-13798); base beta peak defaults to 0 (snlc_sim.c:898)."""
    def x0c(extra, drop_beta=False):
        text = Path(_SALT3).read_text()
        if drop_beta:  # cPOLY-alone: no GENPEAK_SALT2BETA -> base beta = 0
            text = "\n".join(l for l in text.splitlines()
                             if not re.match(r"\s*GEN(PEAK|SIGMA|RANGE)_SALT2BETA", l))
        inp = tmp_path / "n.input"
        inp.write_text(text + "\n" + extra + "\n")
        config = dict(
            n_sne=150, random_seed=5)
        sim = sim_run.prepare(str(inp), **config)
        simulation = None if sim is None else sim.config
        resources = sim
        results, _, _ = sim_run.run_simulation(sim)
        p = results['parameters']
        return np.asarray(p['x0']), np.asarray(p['c'])
    # (a) with a constant beta configured, the poly adds: dmB = a0*c.
    x0_0, c = x0c("")
    x0_1, _ = x0c("SALT2BETA_cPOLY: 0.5")
    assert np.allclose(-2.5 * np.log10(x0_1) + 2.5 * np.log10(x0_0), 0.5 * c, atol=1e-4)
    # (b) override: cPOLY-alone (no GENPEAK) supplies beta with base=0, so
    # `SALT2BETA_cPOLY: 3.1` is identical to `GENPEAK_SALT2BETA: 3.1`.
    x0_ovr, _ = x0c("SALT2BETA_cPOLY: 3.1", drop_beta=True)
    assert np.allclose(x0_ovr, x0_0, rtol=1e-6)
    # (c) a range term draws its coefficient per SN, uniform in [lo, hi] (getRan_Flat).
    x0_r, cr = x0c("SALT2BETA_cPOLY: 0.9:1.1")
    a0 = (-2.5 * np.log10(x0_r) + 2.5 * np.log10(x0_0)) / cr
    assert a0.min() >= 0.9 - 1e-3 and a0.max() <= 1.1 + 1e-3
    assert 0.03 < a0.std() < 0.08


def test_genmag_smear_filter(tmp_path):
    """GENMAG_SMEAR_FILTER adds per-filter coherent scatter (snlc_sim.c:29551):
    one sigma per (SN, band), constant across that band's epochs; repeatable."""
    base = Path(_SALT3).read_text()
    m0 = _phot_by_cid(base, "", tmp_path, 'SIM_MAGOBS', "a", n_sne=300)
    m1 = _phot_by_cid(base, "GENMAG_SMEAR_FILTER: g 0.3\nGENMAG_SMEAR_FILTER: r 0.15",
                      tmp_path, 'SIM_MAGOBS', "b", n_sne=300)
    common = [k for k in set(m0) & set(m1) if -2 < m0[k] < 50 and -2 < m1[k] < 50]
    by_band, per_sn_g = {}, {}
    for k in common:
        b = k[2].split('-')[-1]
        by_band.setdefault(b, []).append(m1[k] - m0[k])
        if b == 'g':
            per_sn_g.setdefault(k[0], []).append(m1[k] - m0[k])
    assert 0.2 < np.std(by_band['g']) < 0.35          # g sigma ~0.3
    assert 0.1 < np.std(by_band['r']) < 0.2           # r sigma ~0.15
    assert np.allclose(by_band.get('i', [0.0]), 0.0)  # unlisted band untouched
    assert all(np.ptp(v) < 1e-4 for v in per_sn_g.values() if len(v) > 1)  # coherent per SN


# --- MJD_EXPLODE + OPTMASK_T0SHIFT_EXPLODE: explosion-time anchoring ---
_SIMSED = "tests/fixtures/snana/simsed_91bg_replay/input.input"
_SEARCHEFF_MIXED = "tests/fixtures/snana/searcheff_mixed_spec_zhost/input.input"
_NON1A_MODEL = "tests/fixtures/data/models/NON1ASED.V19_TEST"
_SIMSED_MODEL = "tests/fixtures/data/models/SIMSED.SNIa-91bg"


def _first_cross_day(day, flux):
    """Earliest phase where any wavelength bin exceeds 1% of the template's global
    flux max — the explosion epoch under OPTMASK_T0SHIFT_EXPLODE:1."""
    thr = 0.01 * flux.max()
    return float(day[np.argmax(flux.max(axis=1) > thr)])


def test_mjd_explode_reanchor(tmp_path, caplog):
    """OPTMASK_T0SHIFT_EXPLODE:1 shifts each NON1ASED/SIMSED template's DAY grid
    so the explosion (1% flux crossing) moves from pre-peak to ~day 0
    (T0shiftExplode_SEDMODEL); MJD_EXPLODE pins the reference MJD to it
    (GENRANGE_PEAKMJD=[M,M], snlc_sim.c:2056); no unrecognized-key warning."""

    # NON1ASED: per-template padded grids
    def non1a_template(**config):
        m = NON1ASEDModel.load(_NON1A_MODEL, config)
        n = m._template_n_phases[0]
        return (np.asarray(m.jax_template_phases[0])[:n],
                np.asarray(m.jax_template_seds[0])[:n])

    day, flux = non1a_template()
    step = float(day[1] - day[0])
    assert _first_cross_day(day, flux) < 0.0                          # pre-peak
    day, flux = non1a_template(OPTMASK_T0SHIFT_EXPLODE=1)
    assert 0.0 <= _first_cross_day(day, flux) <= step + 1e-4          # explosion at T=0

    # SIMSED: shared common grid across templates
    s = SIMSEDModel.load(_SIMSED_MODEL, {})
    assert _first_cross_day(s.common_phases, s.template_seds[0]) < 0.0
    s = SIMSEDModel.load(_SIMSED_MODEL, {'OPTMASK_T0SHIFT_EXPLODE': 1})
    for i in range(s.template_seds.shape[0]):
        assert 0.0 <= _first_cross_day(s.common_phases, s.template_seds[i]) <= 1.0 + 1e-4

    # Pin the reference MJD; the two keys are consumed (not flagged unrecognized).
    with caplog.at_level(logging.WARNING, logger="jax_snlc_sim.config_parser"):
        for base in (_NON1A, _SIMSED):
            r = _run(base, "MJD_EXPLODE: 61000.0\nOPTMASK_T0SHIFT_EXPLODE: 1", tmp_path)
            assert np.allclose(np.asarray(r['parameters']['peakmjd_true']), 61000.0)
    unrecognized = "\n".join(m for m in caplog.messages if "Unrecognized SNANA input key" in m)
    assert "MJD_EXPLODE" not in unrecognized and "OPTMASK_T0SHIFT_EXPLODE" not in unrecognized


# --- HOSTLIB position / magnitude knobs ---
_HOSTLIB = "tests/fixtures/snana/hostlib_e2e/input.input"


def _hostgal(extra, tmp_path, tag, cols, n_sne=80, seed=99):
    base = "".join(l for l in Path(_HOSTLIB).read_text().splitlines(keepends=True)
                   if not l.startswith("SIMGEN_DUMP"))
    inp = tmp_path / f"{tag}.input"
    inp.write_text(base + "\n" + extra + "\n")
    config = dict(
        n_sne=n_sne, random_seed=seed)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    res, sel, spectra_cache = sim_run.run_simulation(sim)
    d = str(tmp_path / f"{tag}_out")
    os.makedirs(d, exist_ok=True)
    sim_run.save_snana_outputs(sim, res, "H", sel, spectra_cache=spectra_cache, output_dir=d)
    with fits.open(resolve_fits_path(f"{d}/H/H_HEAD.FITS")) as h:
        return {c: np.asarray(h[1].data[c]) for c in cols}


def _valid(a, lo=0.0, hi=1e4):
    return a[np.isfinite(a) & (a > lo) & (a < hi)]


def test_hostlib_position_knobs(tmp_path):
    """SCALE_SERSIC_SIZE scales SNSEP; FIXSERSIC+FIXRAN_* make it deterministic;
    MXINTFLUX_SNPOS shrinks the SN-position window."""
    fix = "HOSTLIB_FIXSERSIC: 1.0 0.5 4.0 30.0\nHOSTLIB_FIXRAN_RADIUS: 0.6\nHOSTLIB_FIXRAN_PHI: 0.25"
    base = _valid(_hostgal("", tmp_path, "hb", ['HOSTGAL_SNSEP'])['HOSTGAL_SNSEP'])
    scaled = _valid(_hostgal("HOSTLIB_SCALE_SERSIC_SIZE: 2.0", tmp_path, "hs",
                             ['HOSTGAL_SNSEP'])['HOSTGAL_SNSEP'])
    assert abs(np.median(scaled) / np.median(base) - 2.0) < 0.05
    det = _valid(_hostgal(fix, tmp_path, "hd", ['HOSTGAL_SNSEP'])['HOSTGAL_SNSEP'])
    assert np.std(det) < 1e-3                                    # constant across SNe
    win = _valid(_hostgal(fix + "\nHOSTLIB_MXINTFLUX_SNPOS: 0.5", tmp_path, "hm",
                          ['HOSTGAL_SNSEP'])['HOSTGAL_SNSEP'])
    assert np.median(win) < np.median(det)

    # FIXSERSIC writes back into the catalogue columns, so stored/dumped
    # values report the override (VALUE_ZSORTED, sntools_host.c:5690-5716)
    base = "".join(l for l in Path(_HOSTLIB).read_text().splitlines(keepends=True)
                   if not l.startswith("SIMGEN_DUMP"))
    inp = tmp_path / "hfix_store.input"
    inp.write_text(base + "\n" + fix + "\nHOSTLIB_STOREPAR: a0_Sersic\n")
    config = dict(
        n_sne=40, random_seed=99)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    results, _, _ = sim_run.run_simulation(sim)
    p = results['parameters']
    stored = np.asarray(p['sim_hostlib_a0_Sersic'])
    has_host = np.asarray(p['host_ZTRUE']) > 0
    assert np.all(stored[has_host] == np.float32(1.0))


def test_hostlib_abmag(tmp_path):
    """ABMAG_OFFSET shifts every galaxy mag; ABMAG_FORCE pins the catalog mag.
    The reported HOSTGAL_MAG = catalog + DMUCOR (mu(zSN)-mu(zhost)), so FORCE:22
    centres on 22 with the (zero-mean) DMUCOR spread; the offset shift cancels it."""
    base = _hostgal("", tmp_path, "ab0", ['HOSTGAL_MAG_r'])['HOSTGAL_MAG_r']
    off = _hostgal("HOSTLIB_ABMAG_OFFSET: 0.5", tmp_path, "abo", ['HOSTGAL_MAG_r'])['HOSTGAL_MAG_r']
    frc = _hostgal("HOSTLIB_ABMAG_FORCE: 22.0", tmp_path, "abf", ['HOSTGAL_MAG_r'])['HOSTGAL_MAG_r']
    v = lambda a: a[(a > 0) & (a < 50)]
    assert abs(np.median(v(off)) - np.median(v(base)) - 0.5) < 1e-3
    assert abs(np.mean(v(frc)) - 22.0) < 0.05          # centred on forced value
    assert 0.02 < np.std(v(frc)) < 0.20                # DMUCOR distance-modulus spread


def test_hostlib_dmucor(tmp_path):
    """host_dmucor = mu(zSN) - mu(z_host) (sntools_host.c:9012): zero-mean, non-zero
    spread, and tracks the SN-vs-host redshift offset (positive when zSN > zhost)."""
    base = "".join(l for l in Path(_HOSTLIB).read_text().splitlines(keepends=True)
                   if not l.startswith("SIMGEN_DUMP"))
    inp = tmp_path / "dmucor.input"
    inp.write_text(base + "\n")
    config = dict(
        n_sne=200, random_seed=99)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    results, _, _ = sim_run.run_simulation(sim)
    p = results['parameters']
    dmucor = np.asarray(p['host_dmucor'])
    dz = np.asarray(p['sim_redshift_cmb']) - np.asarray(p['host_ZTRUE'])
    v = np.asarray(p['host_ZTRUE']) > 0
    assert abs(dmucor[v].mean()) < 0.03 and dmucor[v].std() > 0.01
    assert np.corrcoef(dmucor[v], dz[v])[0, 1] > 0.9


def test_hostlib_knobs_guard_and_warning(tmp_path, caplog):
    """Knobs require HOSTLIB_FILE (raise otherwise); with a HOSTLIB they don't warn."""
    with pytest.raises(SNANAError):
        parse_snana_input_file(str(_write(
            tmp_path, Path(_SALT3).read_text() + "\nHOSTLIB_SCALE_SERSIC_SIZE: 2.0\n")))
    keys = ("\nHOSTLIB_ABMAG_OFFSET: 0.5\nHOSTLIB_ABMAG_FORCE: 22\nHOSTLIB_FIXRAN_RADIUS: 0.5\n"
            "HOSTLIB_FIXRAN_WGT: 0.5\n"
            "HOSTLIB_FIXRAN_PHI: 0.25\nHOSTLIB_MNINTFLUX_SNPOS: 0.1\nHOSTLIB_MXINTFLUX_SNPOS: 0.9\n"
            "HOSTLIB_SCALE_SERSIC_SIZE: 2.0\nHOSTLIB_FIXSERSIC: 1.0 0.5 4.0 30.0\n")
    inp = _write(tmp_path, Path(_HOSTLIB).read_text() + keys)
    with caplog.at_level(logging.WARNING, logger="jax_snlc_sim.config_parser"):
        parse_snana_input_file(str(inp))
    unrec = "\n".join(m for m in caplog.messages if "Unrecognized SNANA input key" in m)
    assert not any(k in unrec for k in ("ABMAG", "FIXRAN", "INTFLUX", "SCALE_SERSIC", "FIXSERSIC"))


def _write(tmp_path, text):
    inp = tmp_path / "hl.input"
    inp.write_text(text)
    return inp


# --- HOSTLIB galaxy-selection / read-time knobs ---
# The bundled JAX_TEST.HOSTLIB has galaxies at two (RA_GAL, DEC_GAL) points:
# (10, -30) for file rows 0-... and (180, -45), GALID = 100000 + file_row.


_HOSTLIB_FILE = "tests/fixtures/data/hostlib/JAX_TEST.HOSTLIB"


def _catalog_col(col):
    """{GALID: value} for a column of the bundled HOSTLIB (VARNAMES: GALID ZTRUE
    ZPHOT ZPHOT_ERR RA_GAL DEC_GAL ...)."""
    idx = {'ZTRUE': 2, 'ZPHOT': 3, 'RA_GAL': 5, 'DEC_GAL': 6}[col]
    out = {}
    with open_text(Path(_HOSTLIB_FILE)) as stream:
        fixture_text = stream.read()
    for line in fixture_text.splitlines():
        if line.startswith("GAL:"):
            p = line.split()
            out[int(p[1])] = float(p[idx])
    return out


def _used_catalog(objid, col):
    """Catalog RA_GAL/DEC_GAL of the assigned hosts (OBJID = GALID)."""
    cat = _catalog_col(col)
    return np.array([cat[int(o)] for o in objid if o > 0])


def test_hostlib_genrange_ra(tmp_path):
    """HOSTLIB_GENRANGE_RA restricts the read-time galaxy pool to RA_GAL in the box
    (passCuts_HOSTLIB, sntools_host.c:3764); every assigned host's catalog RA_GAL
    lands in the box. The bundled HOSTLIB clusters at RA_GAL 10 and 180."""
    base = _used_catalog(_hostgal("", tmp_path, "gra0", ['HOSTGAL_OBJID'])['HOSTGAL_OBJID'],
                         'RA_GAL')
    assert {round(x) for x in base} == {10, 180}                    # both clusters used
    cut = _used_catalog(_hostgal("HOSTLIB_GENRANGE_RA: 100.0 200.0", tmp_path, "gra1",
                                 ['HOSTGAL_OBJID'])['HOSTGAL_OBJID'], 'RA_GAL')
    assert len(cut) > 0 and np.all(np.abs(cut - 180.0) < 1e-2)      # only RA=180 cluster


def test_hostlib_genrange_dec(tmp_path):
    """HOSTLIB_GENRANGE_DEC restricts the pool to DEC_GAL in the box; every assigned
    host's catalog DEC_GAL lands in the box (independent of the RA cut)."""
    cut = _used_catalog(_hostgal("HOSTLIB_GENRANGE_DEC: -35.0 -25.0", tmp_path, "gdec",
                                 ['HOSTGAL_OBJID'])['HOSTGAL_OBJID'], 'DEC_GAL')
    assert len(cut) > 0 and np.all(np.abs(cut - (-30.0)) < 1e-2)    # only DEC=-30 cluster


def test_hostlib_hostgal_pos_at_sn(tmp_path):
    """HOSTGAL_RA/DEC report the host at the SN sky position minus the SN-galaxy
    offset (GEN_SNHOST_POS, sntools_host.c:8156), not the raw catalog RA_GAL."""
    d = _hostgal("", tmp_path, "pos",
                 ['HOSTGAL_RA', 'HOSTGAL_DEC', 'SIM_RA', 'SIM_DEC', 'HOSTGAL_OBJID'])
    ra, dec, sra, sdec = d['HOSTGAL_RA'], d['HOSTGAL_DEC'], d['SIM_RA'], d['SIM_DEC']
    ok = d['HOSTGAL_OBJID'] > 0
    assert ok.any()
    assert np.all(np.abs(ra[ok] - sra[ok]) < 0.01)      # within arcsec of the SN, not
    assert np.all(np.abs(dec[ok] - sdec[ok]) < 0.01)    # a distant catalog cluster


def test_hostlib_hostgal_specz(tmp_path):
    """HOSTGAL_SPECZ is the measured SN helio redshift (snlc_sim.c:17040/17154)."""
    d = _hostgal("", tmp_path, "sz",
                 ['HOSTGAL_OBJID', 'HOSTGAL_SPECZ', 'REDSHIFT_HELIO'])
    ok = d['HOSTGAL_OBJID'] > 0
    assert ok.any() and np.allclose(d['HOSTGAL_SPECZ'][ok], d['REDSHIFT_HELIO'][ok], atol=1e-5)


def test_hostlib_hostgal_photoz_formula(tmp_path):
    """HOSTGAL_PHOTOZ = catalog ZPHOT + (ZGEN - host ZTRUE), the SN-vs-host redshift
    shift (GEN_SNHOST_ZPHOT_from_HOSTLIB, sntools_host.c:7489)."""
    d = _hostgal("", tmp_path, "pzf",
                 ['HOSTGAL_OBJID', 'HOSTGAL_PHOTOZ', 'SIM_REDSHIFT_HELIO'])
    objid, pz, zhel = d['HOSTGAL_OBJID'], d['HOSTGAL_PHOTOZ'], d['SIM_REDSHIFT_HELIO']
    ok = objid > 0
    exp = _used_catalog(objid, 'ZPHOT') + (zhel[ok] - _used_catalog(objid, 'ZTRUE'))
    assert ok.any() and np.allclose(pz[ok], exp, atol=1e-4)


def test_hostlib_genzphot_fudgepar_flat(tmp_path):
    """FUDGEPAR's 5th element FLAT(z1:z2) sets flat-outlier mode (sigma_outlier=99) and
    the outlier z-range (parse_input_GENZPHOT_OUTLIER, snlc_sim.c:3345)."""
    cfg = parse_snana_input_file(str(_write(tmp_path, Path(_HOSTLIB).read_text()
        + "\nHOSTLIB_GENZPHOT_FUDGEPAR: 0.02 0 0 0.05 FLAT(0.1:0.5)\n")))
    genzphot = build_simulation(cfg, 500).hostlib.genzphot
    assert genzphot.outlier_range == (0.1, 0.5)
    assert genzphot.fudgepar[4] == 99.0


def test_hostlib_genzphot_bias_triggers_calc(tmp_path):
    """Nonzero HOSTLIB_GENZPHOT_BIAS routes photo-z through GENZPHOT synthesis
    (USE_HOSTLIB_GENZPHOT, snlc_sim.c:6696); zero bias does not."""
    base = Path(_HOSTLIB).read_text()
    cfg = parse_snana_input_file(
        str(_write(tmp_path, base + "\nHOSTLIB_GENZPHOT_BIAS: 0.02 0.1 0 0\n")))
    assert build_simulation(cfg, 500).hostlib.genzphot is not None
    cfg0 = parse_snana_input_file(str(_write(tmp_path, base)))
    assert build_simulation(cfg0, 500).hostlib.genzphot is None


def test_hostlib_maxread(tmp_path):
    """HOSTLIB_MAXREAD caps the pool to the first N file rows (sntools_host.c:3591);
    GALID = 100000 + file_row, so every assigned host has OBJID < 100000+N."""
    base = _hostgal("", tmp_path, "mr0", ['HOSTGAL_OBJID'])['HOSTGAL_OBJID']
    assert np.max(base[base > 0]) >= 100500                          # base reaches beyond
    cut = _hostgal("HOSTLIB_MAXREAD: 500", tmp_path, "mr1",
                   ['HOSTGAL_OBJID'])['HOSTGAL_OBJID']
    used = cut[cut > 0]
    assert len(used) > 0 and np.all(used < 100500)                   # only first 500 rows


def test_hostlib_galid_unique(tmp_path):
    """HOSTLIB_GALID_UNIQUE: HOSTGAL_OBJID_UNIQUE = CID*57 + floor(ran*57)
    (set_GALID_UNIQUE, sntools_host.c:9242); -9 sentinel when unset.

    A SN whose host fails the DDLR cut keeps SNDATA's own 0 rather than the
    -9 the DDLR sort writes, so both values are legitimate here."""
    unset = _hostgal("", tmp_path, "gu0",
                     ['HOSTGAL_OBJID_UNIQUE', 'HOSTGAL_NMATCH2'])
    off = unset['HOSTGAL_OBJID_UNIQUE']
    matched = unset['HOSTGAL_NMATCH2'] > 0
    assert np.all(off[matched] == -9)                    # sentinel when unset
    assert np.all(off[~matched] == 0)                    # no host matched
    d = _hostgal("HOSTLIB_GALID_UNIQUE: 1", tmp_path, "gu1",
                 ['HOSTGAL_OBJID_UNIQUE', 'HOSTGAL_OBJID', 'SNID'])
    uniq, objid = d['HOSTGAL_OBJID_UNIQUE'], d['HOSTGAL_OBJID']
    cid = np.array([int(str(s).strip()) for s in d['SNID']], dtype=np.int64)
    # SNANA's no-host HOSTGAL_OBJID is 0, not -9 (sntools.c:8523)
    has_host = objid > 0
    assert has_host.any()
    off_from_cid = uniq[has_host] - cid[has_host] * 57
    assert np.all(off_from_cid >= 0) and np.all(off_from_cid <= 56)  # CID*57 + [0,56]
    assert np.all(uniq[~has_host] == 0)                              # hostless -> 0


def test_hostlib_galid_unique_survives_ngen_lc_renumbering(tmp_path):
    """GALID_UNIQUE tracks the written CID after NGEN_LC renumbers it.

    NGEN_LC sets CID from the accepted slot being filled, so galid_unique --
    built from the generated index during sampling -- must be remapped with it.
    """
    base = "".join(l for l in Path(_HOSTLIB).read_text().splitlines(keepends=True)
                   if not l.startswith("SIMGEN_DUMP"))
    base = base.replace("NGENTOT_LC:  300", "NGEN_LC:  40")
    base = base.replace("APPLY_SEARCHEFF_OPT:  0", "APPLY_SEARCHEFF_OPT:  1")
    inp = tmp_path / "gu_ngenlc.input"
    inp.write_text(base + "\nHOSTLIB_GALID_UNIQUE: 1\n")

    config = dict(
        n_sne=40, random_seed=99)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    res, sel, spectra_cache = sim_run.run_simulation(sim)
    out = str(tmp_path / "gu_ngenlc_out")
    os.makedirs(out, exist_ok=True)
    sim_run.save_snana_outputs(sim, res, "H", sel, spectra_cache=spectra_cache, output_dir=out)
    with fits.open(resolve_fits_path(f"{out}/H/H_HEAD.FITS")) as h:
        uniq = np.asarray(h[1].data['HOSTGAL_OBJID_UNIQUE'])
        objid = np.asarray(h[1].data['HOSTGAL_OBJID'])
        cid = np.array([int(str(x).strip()) for x in h[1].data['SNID']], dtype=np.int64)

    has_host = objid != -9
    assert has_host.sum() > 10, "need hosts to check the relation"
    # galid_unique = CID*57 + off with off in [0, 56], so // 57 recovers the CID
    np.testing.assert_array_equal(uniq[has_host] // 57, cid[has_host])


def _write_zphot_hostlib(path, zphot_name):
    """Small HOSTLIB whose photo-z column is named `zphot_name` (+ _ERR), with
    ZPHOT = ZTRUE + 0.02 so an alias is verifiable by value."""
    hdr = ("VARNAMES: GALID ZTRUE RA_GAL DEC_GAL a0_Sersic b0_Sersic n0_Sersic "
           f"w0_Sersic a_rot {zphot_name} {zphot_name}_ERR")
    rows = ["DOCUMENTATION:", "DOCUMENTATION_END:", hdr]
    for i in range(40):
        z = 0.05 + i * 0.01
        rows.append(f"GAL: {1000 + i} {z:.4f} {10.0 + i * 0.1:.4f} -30.0 "
                    f"1.0 0.8 1.0 1.0 45.0 {z + 0.02:.4f} 0.01")
    Path(path).write_text("\n".join(rows) + "\n")


def test_hostlib_varname_zphot(tmp_path):
    """HOSTLIB_VARNAME_ZPHOT aliases a differently-named photo-z column (and its
    _ERR) to ZPHOT/ZPHOT_ERR at read time (checkAlternateVarNames_HOSTLIB,
    sntools_host.c:3395), so ZPHOT is read from the aliased column; without
    the alias the reader finds no ZPHOT column."""
    hl = tmp_path / "zp.HOSTLIB"
    _write_zphot_hostlib(hl, "ZPHOT_GAUSS")
    columns = essential_columns() + ['ZPHOT', 'ZPHOT_ERR']

    aliased = read_hostlib(str(hl), columns, varname_zphot="ZPHOT_GAUSS",
                           cache_dir=str(tmp_path / "cache"))
    np.testing.assert_allclose(aliased.columns['ZPHOT'],
                               aliased.columns['ZTRUE'] + 0.02, atol=1e-4)
    assert np.allclose(aliased.columns['ZPHOT_ERR'], 0.01)

    # default (no alias) finds no literal ZPHOT column
    ctrl = read_hostlib(str(hl), columns, cache_dir=str(tmp_path / "cache2"))
    assert 'ZPHOT' not in ctrl.columns and 'ZPHOT_ERR' not in ctrl.columns


def test_hostlib_varname_zphot_err_nounderscore(tmp_path):
    """The photo-z error alias also resolves the SNANA `<name>ERR` spelling (no
    underscore), not just `<name>_ERR` (checkAlternateVarNames_HOSTLIB)."""
    hdr = ("VARNAMES: GALID ZTRUE RA_GAL DEC_GAL a0_Sersic b0_Sersic n0_Sersic "
           "w0_Sersic a_rot ZPHOT_GAUSS ZPHOT_GAUSSERR")
    rows = ["DOCUMENTATION:", "DOCUMENTATION_END:", hdr]
    for i in range(30):
        z = 0.05 + i * 0.01
        rows.append(f"GAL: {1000 + i} {z:.4f} {10.0 + i * 0.1:.4f} -30.0 1.0 0.8 1.0 1.0 45.0 "
                    f"{z + 0.02:.4f} 0.02")
    hl = tmp_path / "zpe.HOSTLIB"
    hl.write_text("\n".join(rows) + "\n")
    r = read_hostlib(str(hl), essential_columns() + ['ZPHOT', 'ZPHOT_ERR'],
                     varname_zphot="ZPHOT_GAUSS", cache_dir=str(tmp_path / "c"))
    assert np.allclose(r.columns['ZPHOT_ERR'], 0.02)


def test_hostlib_swapzphot_draws_on_the_aliased_photoz(tmp_path):
    """SWAPZPHOT swaps in the column HOSTLIB_VARNAME_ZPHOT names: SNANA
    replaces ZPHOT with the alias when it reads the header
    (checkAlternateVarNames_HOSTLIB, sntools_host.c:2988-2996) and swaps
    afterwards (:3643). Every host's aliased photo-z is then within
    HOSTLIB_DZTOL (0.05 z here) of the SN's redshift, which its ZTRUE is not;
    SNANA's own run of this input gives the same."""
    hostlib = tmp_path / "ALIAS.HOSTLIB"
    with open_text(Path(_HOSTLIB_FILE)) as stream:
        fixture_text = stream.read()
    hostlib.write_text(fixture_text.replace(
        "VARNAMES: GALID ZTRUE ZPHOT ZPHOT_ERR ",
        "VARNAMES: GALID ZTRUE ZPHOT_GAUSS ZPHOT_GAUSS_ERR ", 1))
    text = re.sub(r"(?m)^HOSTLIB_FILE:.*$",
                  f"HOSTLIB_FILE: {hostlib}\nHOSTLIB_VARNAME_ZPHOT: ZPHOT_GAUSS",
                  Path(_HOSTLIB).read_text())
    text = re.sub(r"(?m)^HOSTLIB_MSKOPT:.*$", "HOSTLIB_MSKOPT: 386", text)
    config = dict(n_sne=80, random_seed=99)
    sim = sim_run.prepare(str(_write(tmp_path, text)), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    results, _, _ = sim_run.run_simulation(sim)
    galid = np.asarray(results['parameters']['host_GALID'])
    z = np.asarray(results['parameters']['sim_redshift_helio'])[galid > 0]
    zphot = _used_catalog(galid, 'ZPHOT')           # the aliased column's values
    ztrue = _used_catalog(galid, 'ZTRUE')
    assert len(z) > 50
    assert np.all(np.abs(zphot - z) <= 0.05 * z + 1e-4)
    assert np.any(np.abs(ztrue - z) > 0.05 * z)


def test_hostlib_genrange_ra_wrap(tmp_path):
    """HOSTLIB_GENRANGE_RA keeps galaxies via the RA+360 wrap and the SNANA
    ±0.0001° margin (passCuts_HOSTLIB, sntools_host.c:3784)."""
    ras = [-5.0, -2.0, 5.0, 355.0, 350.00005]     # wrap: -5->355, -2->358; 355 in-box
    hdr = ("VARNAMES: GALID ZTRUE RA_GAL DEC_GAL a0_Sersic b0_Sersic "
           "n0_Sersic w0_Sersic a_rot")
    rows = ["DOCUMENTATION:", "DOCUMENTATION_END:", hdr]
    for i, ra in enumerate(ras):
        rows.append(
            f"GAL: {1000 + i} {0.1 + 0.01 * i:.4f} {ra:.5f} -30.0 "
            f"1.0 0.8 1.0 1.0 45.0")
    hl = tmp_path / "wrap.HOSTLIB"
    hl.write_text("\n".join(rows) + "\n")
    d = read_hostlib(str(hl), essential_columns(), genrange_ra=(350.0, 360.0),
                     cache_dir=str(tmp_path / "c"))
    kept = set(d.columns['GALID'].tolist())
    # -5->355, -2->358, 355 in-box, 350.00005 within +margin; +5 excluded
    assert kept == {1000, 1001, 1003, 1004}


def test_simsed_param_aliases_not_warned(tmp_path, caplog):
    """SIMSED_SHAPEPAR/COLORPAR/COLORLAW share SIMSED_PARAM's parser
    (snlc_sim.c:5892-5903); the IPAR tags are GRID-mode only."""
    inp = _write(tmp_path, Path(_SALT3).read_text() + "\nFLUXERRMODEL_OPTMASK: 1152\n")
    simsed = tmp_path / "simsed.input"
    simsed.write_text(Path(_SIMSED).read_text() + "\nSIMSED_USE_BINARY: 1\n")
    with caplog.at_level(logging.WARNING, logger="jax_snlc_sim.config_parser"):
        parse_snana_input_file(str(inp))
        parse_snana_input_file(str(simsed))
    unrec = "\n".join(m for m in caplog.messages if "Unrecognized SNANA input key" in m)
    assert not any(k in unrec for k in ("SIMSED_USE_BINARY", "FLUXERRMODEL_OPTMASK"))

    # SIMSED_PATH_BINARY names SNANA's flux-table cache, which jax has no
    # analogue for: accepted so the official SIMSED inputs run, and logged
    simsed.write_text(Path(_SIMSED).read_text() + "\nSIMSED_PATH_BINARY: flux_cache\n")
    with caplog.at_level(logging.INFO, logger="jax_snlc_core.models.simsed_model"):
        build_model(_SIMSED_MODEL, parse_snana_input_file(str(simsed)).model_config())
    assert any("SIMSED_PATH_BINARY flux_cache names SNANA's flux-table cache" in m
               for m in caplog.messages)


def test_sed_binary_reads_either_snana_layout(tmp_path):
    """SNANA wrote SED.BINARY with 8-byte header ints and a 300-byte path
    field until FIX_SIZE_ARG_BINARY, and with 4-byte ints and a
    length-prefixed path since; IVERSION is 4 in both. Both layouts of the
    same grid must read back identically, the last flux zero-padded as
    pack_SEDBINARY leaves it. The old 8-byte ints were written from 4-byte
    fields, so their high halves carry the neighbouring struct member (a
    version-2 library on disk has 6 there), and a version other than 4 is
    refused."""

    nday, nlam = 3, 4
    days = [-2.0, 0.0, 2.419]
    lams = [4000.0, 4100.0, 4200.0, 4300.0]
    seds = []
    for ised in range(2):
        flux = [float(10 * ised + i) for i in range(nday * nlam)]
        # pack_SEDBINARY fills from index 1 and fwrite writes N floats from
        # index 0, so index 0 is unused and the last flux is never written
        packed = [0.0, 4.0, nday, 2.419, 1.0, 77777.0, *days, nlam, 100.0, *lams, *flux]
        seds.append((f"models/SIMSED.TEST/sed_{ised:03d}.dat".encode(),
                     np.asarray(packed[:-1], dtype=np.float32)))

    old = struct.pack("<ii", 4, 6) + struct.pack("<ii", nday, 6)
    new = struct.pack("<ii", 4, nday)
    for path, floats in seds:
        old += path.ljust(300, b"\0") + struct.pack("<i", len(floats)) + floats.tobytes()
        new += struct.pack("<i", len(path)) + path + struct.pack("<i", len(floats)) + floats.tobytes()
    (tmp_path / "old.BINARY").write_bytes(old)
    (tmp_path / "new.BINARY").write_bytes(new)

    phases_old, lam_old, flux_old = read_sed_binary(str(tmp_path / "old.BINARY"), 2)
    phases_new, lam_new, flux_new = read_sed_binary(str(tmp_path / "new.BINARY"), 2)

    np.testing.assert_array_equal(phases_old, phases_new)
    np.testing.assert_array_equal(lam_old, lam_new)
    np.testing.assert_array_equal(flux_old, flux_new)
    np.testing.assert_allclose(phases_new[1], days, rtol=1e-6)   # float32 file
    np.testing.assert_array_equal(lam_new, lams)

    (tmp_path / "v2.BINARY").write_bytes(struct.pack("<ii", 2, 0) + old[8:])
    with pytest.raises(SNANAError, match="format version 2"):
        read_sed_binary(str(tmp_path / "v2.BINARY"), 2)
    np.testing.assert_array_equal(flux_new, [
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 0]],
        [[10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 0]],
    ])  # each record's last flux is unwritten and must be zero-padded



def test_simsed_shapepar_feeds_simsed_params(tmp_path):
    """All four spellings pool into one declaration, in file order.

    Read through the model, so dropping a spelling from what SIMSED declares
    fails here rather than leaving the test naming the aliases to itself.
    """
    keys = ("\nSIMSED_PARAM: ONE\nSIMSED_SHAPEPAR: TWO\n"
            "SIMSED_COLORPAR: THREE\nSIMSED_COLORLAW: FOUR\n")
    raw = read_keys(str(_write(tmp_path, Path(_SALT3).read_text() + keys)))
    declared = SIMSEDModel.structured_input(raw)['simsed_config']
    assert declared['continuous_parnames'] == ['ONE', 'TWO', 'THREE', 'FOUR']


def test_gengauss_init_values(tmp_path):
    """Any family key activates a declared population, with SNANA's
    per-parameter init values filling the rest (prepIndex_GENGAUSS;
    SALT2x1 inits at snlc_sim.c:887-893)."""
    src = "\n".join(l for l in Path(_SALT3).read_text().splitlines()
                    if 'SALT2x1' not in l)
    cfg = parse_snana_input_file(
        str(_write(tmp_path, src + "\nGENSIGMA_SALT2x1: 0.1 0.1\n")))
    dist = cfg.model.populations['x1'].distributions[0]
    assert dist.peak == -9.0
    assert tuple(dist.range) == (-5.0, 5.0)


def test_fluxerrmodel_optmask_unknown_bit_aborts(tmp_path):
    """Only MONITORCOV(128)/REQUIRE_DOCANA(1024) exist
    (sntools_fluxErrModels.h:27,29)."""
    inp = _write(tmp_path, Path(_SALT3).read_text() + "\nFLUXERRMODEL_OPTMASK: 5\n")
    with pytest.raises(SNANAError):
        parse_snana_input_file(str(inp))
    # bit 256 dumps the map in SNANA (MASK_DUMP_MAP_FLUXERRMODEL), which
    # segfaults there; the abort names it
    inp = _write(tmp_path, Path(_SALT3).read_text() + "\nFLUXERRMODEL_OPTMASK: 256\n")
    with pytest.raises(SNANAError, match="Bit 256"):
        parse_snana_input_file(str(inp))


def test_simlib_nskipmjd_thins_epochs(tmp_path):
    """SIMLIB_NSKIPMJD keeps every Nth unique MJD (keep_SIMLIB_MJD,
    snlc_sim.c:20907); the TRESTMIN form thins only late epochs."""
    def nobs(extra):
        return np.asarray(_run(_SALT3, extra, tmp_path, n_sne=20)
                          ['observation_schedule']['n_obs_per_sn'])
    base = nobs("")
    ps2 = nobs("SIMLIB_NSKIPMJD: 2")
    ps3 = nobs("SIMLIB_NSKIPMJD: 3")
    late = nobs("SIMLIB_NSKIPMJD: 2(20)")
    assert ps2.sum() < base.sum() and ps3.sum() < ps2.sum()
    assert ps2.sum() <= late.sum() <= base.sum()


def test_simlib_nskipmjd_parse_and_aborts(tmp_path):
    """PRESCALE(TRESTMIN) parsing; TRESTMIN must increase (snlc_sim.c:6028)."""
    assert _parse_simlib_nskipmjd("2(50),4(100)") == ((2, 50), (4, 100))
    assert _parse_simlib_nskipmjd("3") == ((3, -999999),)
    assert _parse_simlib_nskipmjd(None) == ()
    for bad in ("4(100),2(50)", "2(50),3(50)", "x(1)"):
        with pytest.raises(SNANAError):
            _parse_simlib_nskipmjd(bad)
    cfg = parse_snana_input_file(str(_write(
        tmp_path, Path(_SALT3).read_text() + "\nSIMLIB_NSKIPMJD: 2(50),4(100)\n")))
    assert cfg.survey.simlib_nskipmjd == ((2, 50), (4, 100))


def test_mxradius_random_shift_moves_sn_and_hosts(tmp_path):
    """MXRADIUS_RANDOM_SHIFT displaces the SN within the radius and moves every
    host by the same offset (gen_random_coord_shift, snlc_sim.c:13113)."""
    hostlib = Path(_HOSTLIB).read_text()
    base = _run(_HOSTLIB, "", tmp_path, n_sne=200, seed=3)
    inp = _write(tmp_path, hostlib + "\nMXRADIUS_RANDOM_SHIFT: 0.05\n")
    config = dict(
        n_sne=200, random_seed=3)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    shifted, _, _ = sim_run.run_simulation(sim)

    p0, p1 = base['parameters'], shifted['parameters']
    ra0, dec0 = np.asarray(p0['ra'], float), np.asarray(p0['dec'], float)
    ra1, dec1 = np.asarray(p1['ra'], float), np.asarray(p1['dec'], float)
    sep = np.hypot((ra1 - ra0) * np.cos(np.radians(dec0)), dec1 - dec0)
    assert sep.max() <= 0.05 + 1e-6
    assert abs(sep.mean() - 2.0 / 3.0 * 0.05) < 0.005   # uniform in area

    d_sn_ra = np.asarray(p1['random_shift_RA'], float)
    d_host_ra = np.asarray(p1['host_RA_GAL'], float) - np.asarray(p0['host_RA_GAL'], float)
    assert np.abs(d_host_ra - d_sn_ra).max() < 1e-4      # hosts move rigidly
    # SNANA leaves MWEBV and the redshifts on the pre-shift coords (:12849-12853)
    assert np.allclose(np.asarray(p0['mwebv'], float),
                       np.asarray(p1['mwebv'], float))
    assert np.allclose(np.asarray(p0['sim_redshift_cmb'], float),
                       np.asarray(p1['sim_redshift_cmb'], float))


def test_take_spectrum_prescale_parsing(tmp_path):
    """FIELD/prescale pairs, '+' or ',' separated (sntools.c
    parse_string_prescales); a bare name means prescale 1 (:988)."""
    assert _parse_string_prescales("C3/4.3+X3/2.2") == {'C3': 4.3, 'X3': 2.2}
    assert _parse_string_prescales("C3/4,X3/2") == {'C3': 4.0, 'X3': 2.0}
    assert _parse_string_prescales(None) == {}
    assert _parse_string_prescales("C3") == {'C3': 1.0}
    cfg = parse_snana_input_file(str(_write(
        tmp_path, Path(_SALT3).read_text()
        + "\nTAKE_SPECTRUM_PRESCALE: TOY_A/4.0\nTAKE_SPECTRUM_DUMPCID: 17\n")))
    assert cfg.spectra.take_spectrum_prescale == {'TOY_A': 4.0}
    assert cfg.spectra.take_spectrum_dumpcid == 17


def test_keep_spectrum_rule():
    """DO_GENSPEC gate order (snlc_sim.c:10401): no-field bypasses everything,
    then per-key WGT, then exact FIELD_REQUIRE, then the partial-match dict
    prescale with host spectra exempt."""
    assert keep_spectrum('', '', 0.1, {'C3': 100.0}, 0.99, 0.99) is True      # no field
    assert keep_spectrum('FIELD_NONAME', '', 0.1, {'C3': 100.0}, 0.99, 0.99) is True
    assert keep_spectrum('C3', '', 0.5, {}, 0.4, 0.0) is True                 # wgt keep
    assert keep_spectrum('C3', '', 0.5, {}, 0.6, 0.0) is False                # wgt drop
    assert keep_spectrum('C3', '', 1.0, {}, 0.99, 0.0) is True                # wgt=1 inert
    assert keep_spectrum('C3', 'C3', 1.0, {}, 0.0, 0.0) is True               # field match
    assert keep_spectrum('C3', 'X3', 1.0, {}, 0.0, 0.0) is False              # exact strcmp
    assert keep_spectrum('C3', '', 1.0, {'C3': 4.0}, 0.0, 0.2) is True        # ps keep
    assert keep_spectrum('C3', '', 1.0, {'C3': 4.0}, 0.0, 0.3) is False       # ps drop
    assert keep_spectrum('X3', '', 1.0, {'C3': 4.0}, 0.0, 0.99) is True       # absent -> 1.0
    assert keep_spectrum('C3', '', 1.0, {'C3': 4.0}, 0.0, 0.99, True) is True  # host exempt
    assert keep_spectrum('C3', '', 1.0, {'C': 4.0}, 0.0, 0.3) is False        # partial match
    assert keep_spectrum('C3', '', 1.0, {'C': 1.0, 'C3': 4.0}, 0.0, 0.9) is True  # first wins
    # array form must agree with the scalar form elementwise
    f = np.array(['C3', 'X3', '', 'C3'], dtype=object)
    r = np.array([0.1, 0.9, 0.9, 0.9])
    arr = keep_spectrum(f, '', 0.5, {'C3': 4.0}, r, r)
    assert list(arr) == [keep_spectrum(f[i], '', 0.5, {'C3': 4.0}, r[i], r[i]) for i in range(4)]


def _spec_by_cid(base_text, extra, tmp_path, tag, simlib=None, n_sne=30, seed=5):
    """Save a sim and return {cid: (texpose, mjd, sim_flam)} from the SPEC table."""
    text = base_text
    if simlib is not None:
        text = "\n".join(f"SIMLIB_FILE:  {simlib}" if l.startswith("SIMLIB_FILE:") else l
                         for l in text.split("\n"))
    inp = tmp_path / f"{tag}.input"
    inp.write_text(text + "\n" + extra + "\n")
    config = dict(
        n_sne=n_sne, random_seed=seed)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    res, sel, spectra_cache = sim_run.run_simulation(sim)
    d = str(tmp_path / f"{tag}_out")
    os.makedirs(d, exist_ok=True)
    sim_run.save_snana_outputs(sim, res, tag, sel, spectra_cache=spectra_cache, output_dir=d)
    out = {}
    with fits.open(resolve_fits_path(f"{d}/{tag}/{tag}_SPEC.FITS")) as h:
        hdr, flx = h[1].data, h[2].data
        for r in hdr:
            lo, hi = int(r['PTRSPEC_MIN']) - 1, int(r['PTRSPEC_MAX'])
            out[int(str(r['SNID']).strip())] = (
                float(r['Texpose']), float(r['MJD']),
                np.asarray(flx['SIM_FLAM'][lo:hi], dtype=float))
    return out


def test_use_simlib_take_spectrum_matches_input_file_schedule(tmp_path):
    """USE_SIMLIB_TAKE_SPECTRUM: the same schedule delivered per-LIBID in the
    SIMLIB header must produce the same spectra as the input-file key.

    SNANA parses both through parse_input_TAKE_SPECTRUM (snlc_sim.c:21894), so
    the two are the same schedule by construction. This pins the per-LIBID
    driver against the already-validated input-file path; SNANA itself cannot
    run this key (see SNANA_KEY_COVERAGE.md), so there is no live fixture.
    """
    take = "TAKE_SPECTRUM:  TREST(5:5) TEXPOSE_ZPOLY(2000,0)"
    base = "\n".join(l for l in Path(
        "tests/fixtures/snana/spectrum_prescale_replay/input.input").read_text().split("\n")
        if not l.startswith("TAKE_SPECTRUM"))
    simlib = "$JAX_TEST_FIXTURE_ROOT/simlib/JAX_TEST_LSST_TAKESPEC.SIMLIB"

    a = _spec_by_cid(base, take, tmp_path, "IF")
    b = _spec_by_cid(base, "USE_SIMLIB_TAKE_SPECTRUM: 1", tmp_path, "SL", simlib=simlib)

    assert a and set(a) == set(b)
    for cid in a:
        assert a[cid][0] == b[cid][0]                      # TEXPOSE
        assert a[cid][1] == b[cid][1]                      # MJD
        np.testing.assert_array_equal(a[cid][2], b[cid][2])  # SIM_FLAM (truth)


def test_use_simlib_take_spectrum_aborts(tmp_path):
    """The six guards around the key, five of which SNANA also enforces."""

    base = Path("tests/fixtures/snana/spectrum_prescale_replay/input.input").read_text()
    no_take = "\n".join(l for l in base.split("\n") if not l.startswith("TAKE_SPECTRUM"))

    # both spellings of the same key (USE_SIMLIB_SPECTRA is the pre-Aug-2025 one)
    with pytest.raises(SNANAError, match="aliases"):
        parse_snana_input_file(str(_write(
            tmp_path, no_take + "\nUSE_SIMLIB_TAKE_SPECTRUM: 1\nUSE_SIMLIB_SPECTRA: 1\n")))
    # schedule given in both the SIMLIB and the input file
    with pytest.raises(SNANAError, match="not both"):
        parse_snana_input_file(str(_write(
            tmp_path, base + "\nUSE_SIMLIB_TAKE_SPECTRUM: 1\n")))
    # both SIMLIB spectra modes at once
    with pytest.raises(SNANAError, match="cannot"):
        parse_snana_input_file(str(_write(
            tmp_path, no_take + "\nUSE_SIMLIB_TAKE_SPECTRUM: 1\nUSE_SIMLIB_SPECTROGRAPH: 1\n")))

    # entry-level guards, which apply to the input-file path too
    with pytest.raises(SNANAError, match="SNR_ZPOLY or TEXPOSE_ZPOLY"):
        prepare_spectrum_requests([parse_take_spectrum_entry("TAKE_SPECTRUM", "TREST(5:5)")])
    with pytest.raises(SNANAError, match="MJD"):
        parse_take_spectrum_entry("TAKE_SPECTRUM", "MJD(60000:60100) TEXPOSE_ZPOLY(100,0)",
                                  from_simlib=True)
    # ... but MJD() stays legal from an input file
    entry = parse_take_spectrum_entry(
        "TAKE_SPECTRUM", "MJD(60000:60100) TEXPOSE_ZPOLY(100,0)")
    assert entry['epoch_frame'] == 'MJD'
    # MJD(min:max:step) is one spectrum per MJD, with TEXPOSE_ZPOLY only
    # (expand_TAKE_SPECTRUM_MJD, snlc_sim.c:5819-5847)
    entries = prepare_spectrum_requests([parse_take_spectrum_entry(
        "TAKE_SPECTRUM", "MJD(60000:60100:25) TEXPOSE_ZPOLY(100,0)")])
    assert [e['epoch_range'] for e in entries] == [
        [m, m] for m in (60000.0, 60025.0, 60050.0, 60075.0, 60100.0)]
    with pytest.raises(SNANAError, match="TEXPOSE_ZPOLY"):
        prepare_spectrum_requests([parse_take_spectrum_entry(
            "TAKE_SPECTRUM", "MJD(60000:60100:25) SNR_ZPOLY(30,0)")])


def test_nearpeak_uses_realised_trest():
    """GENSPEC_MJD_ORDER (snlc_sim.c:10676-10687) picks each event's near-peak
    spectrum from its own realised |TREST|, so the choice varies per SN.

    Selecting from the config's nominal epoch centre instead gives every SN the
    same answer, and additionally compares OBS-frame centres against REST-frame
    ones. Both are what this guards. The flat (SN, epoch) form is shared by the
    global and per-LIBID schedules, so a ragged case is covered too."""

    # two SNe, two epochs each: by config centre (|-13| < |20|) the first epoch
    # would always win, but per SN the realised draw can favour the second
    sn = np.array([0, 1, 0, 1])
    trest = np.array([-18.0, -18.0, 25.0, 10.0])
    got = nearpeak_mask(sn, trest, np.zeros(4, bool))
    assert list(got) == [True, False, False, True]

    # ragged: SN 0 has three epochs, SN 1 has one
    sn = np.array([0, 0, 0, 1])
    trest = np.array([30.0, -4.0, 12.0, 99.0])
    assert list(nearpeak_mask(sn, trest, np.zeros(4, bool))) == [False, True, False, True]

    # HOST epochs are never near-peak even at phase 0
    got = nearpeak_mask(np.array([0, 0]), np.array([0.0, 40.0]), np.array([True, False]))
    assert list(got) == [False, True]

    # ties keep the first row (SNANA's scan is strictly decreasing)
    got = nearpeak_mask(np.array([0, 0]), np.array([5.0, -5.0]), np.zeros(2, bool))
    assert list(got) == [True, False]


def test_use_simlib_take_spectrum_simlib_entry_aborts(tmp_path):
    """HOST entries need host SEDs and WARP_SPECTRUM is unimplemented, so both
    must abort rather than be silently dropped from a SIMLIB header."""

    src = Path("tests/fixtures/data/simlib/JAX_TEST_LSST_SPECPS.SIMLIB").read_text()
    base = "\n".join(l for l in Path(
        "tests/fixtures/snana/spectrum_prescale_replay/input.input").read_text().split("\n")
        if not l.startswith("TAKE_SPECTRUM"))

    def run(header_line):
        lines = []
        for l in src.split("\n"):
            lines.append(l)
            if l.startswith("LIBID:"):
                lines.append(header_line)
        sl = tmp_path / f"S{abs(hash(header_line))}.SIMLIB"
        sl.write_text("\n".join(lines))
        inp = tmp_path / f"i{abs(hash(header_line))}.input"
        inp.write_text("\n".join(
            f"SIMLIB_FILE:  {sl}" if l.startswith("SIMLIB_FILE:") else l
            for l in base.split("\n")) + "\nUSE_SIMLIB_TAKE_SPECTRUM: 1\n")
        config = dict(
            n_sne=5, random_seed=5)
        sim = sim_run.prepare(str(inp), **config)
        simulation = None if sim is None else sim.config
        resources = sim
        from jax_snlc_sim.builder import simlib_take_spectrum_entries
        simlib_take_spectrum_entries(sim.fields)

    with pytest.raises(SNANAError, match="host SEDs"):
        run("TAKE_SPECTRUM:  HOST TEXPOSE_ZPOLY(1800,0)")
    with pytest.raises(SNANAError, match="WARP_SPECTRUM"):
        run("WARP_SPECTRUM:  LAMPOLY(1,0,0)")


def test_gensource_only_random(tmp_path):
    """GENSOURCE selects the generation mode; jax implements RANDOM only.

    SNANA branches three ways (snlc_sim.c:117-137): RANDOM simulates a survey,
    GRID calls init_GRIDsource to write a model grid for PSNID photometric
    templates (NGRID_* / GRID_FORMAT keys) rather than light curves, and any
    other value is a fatal abort. GRID is deferred, so an unhandled value must
    abort rather than quietly hand back a normal simulated survey.
    """

    base = "\n".join(l for l in Path(_SALT3).read_text().split("\n")
                     if not l.startswith("GENSOURCE"))

    # the supported mode, explicit or omitted, and case-insensitively
    for value in ("GENSOURCE: RANDOM", "GENSOURCE: random", ""):
        parse_snana_input_file(str(_write(tmp_path, base + "\n" + value + "\n")))

    with pytest.raises(SNANAError, match="GRID"):
        parse_snana_input_file(str(_write(tmp_path, base + "\nGENSOURCE: GRID\n")))
    with pytest.raises(SNANAError, match="RANDOM only"):
        parse_snana_input_file(str(_write(tmp_path, base + "\nGENSOURCE: BOGUS\n")))


def test_simsed_param_grammar(tmp_path):
    """A SIMSED distribution is declared by naming the parameter in the key.

    Canonical form (manual sec. 12116) is "SIMSED_PARAM: stretch" followed by
    GENPEAK_stretch / GENSIGMA_stretch / GENRANGE_stretch. Writing the literal
    words SIMSED_PARAM in place of the name -- GENRANGE_SIMSED_PARAM(stretch)
    -- reads as a range but declares nothing, and SNANA drops it silently
    (snlc_sim.c:2222 routes it into parse_input_SIMSED, whose branches all
    match exactly, so the else-if chain never reaches keyContains_SIMSED_PARAM).
    Every event then lands on grid entry 0, so this must abort rather than
    quietly collapse the sample onto one template.
    """

    simsed = Path("tests/fixtures/snana/simsed_91bg_replay/input.input").read_text()

    # canonical spelling binds each GEN* key to the parameter it names
    cfg = parse_snana_input_file(str(_write(tmp_path, simsed)))
    for name, peak, lo, hi in (("stretch", 0.975, 0.65, 1.25),
                               ("color", 0.557, 0.0, 1.0)):
        dist = cfg.model.populations[f'simsed_{name}'].distributions[0]
        assert dist.peak == peak
        assert dist.range == (lo, hi)

    bad = simsed.replace("SIMSED_PARAM:  stretch",
                         "GENRANGE_SIMSED_PARAM(stretch): 0.65 1.25\n#")
    with pytest.raises(SNANAError, match="name no parameter"):
        parse_snana_input_file(str(_write(tmp_path, bad)))


def test_simsed_redcor_matches_rejection_sampling():
    """SIMSED_REDCOR draws the rectangle-truncated bivariate Gaussian.

    SNANA correlates the pair through a Cholesky decomposition and redraws BOTH
    parameters whenever either falls outside its GENRANGE, up to 100 tries
    (snlc_sim.c:13885, PICK_RANCOV). That rejection defines the target
    distribution; jax reaches it by inverse CDF so the draw stays vectorised.
    Compared here against a brute-force rejection sampler -- the two must be
    statistically indistinguishable, and in particular the truncation must NOT
    pile probability onto the range boundaries the way clipping would.
    """

    peak1, sig1, range1 = 0.975, 0.096, (0.65, 1.25)
    peak2, sig2, range2 = 0.557, 0.175, (0.0, 1.0)
    rho, n = -0.656, 200_000

    key = jax.random.PRNGKey(0)
    x1, x2 = _draw_correlated_truncated(
        peak1, sig1, range1, peak2, sig2, range2, rho,
        jax.random.uniform(jax.random.fold_in(key, 1), (n,)),
        jax.random.uniform(jax.random.fold_in(key, 2), (n,)))
    x1, x2 = np.asarray(x1), np.asarray(x2)

    rng = np.random.default_rng(7)
    g = rng.standard_normal((6 * n, 2))
    a = peak1 + sig1 * g[:, 0]
    b = peak2 + sig2 * (rho * g[:, 0] + np.sqrt(1.0 - rho ** 2) * g[:, 1])
    keep = ((a >= range1[0]) & (a <= range1[1])
            & (b >= range2[0]) & (b <= range2[1]))
    y1, y2 = a[keep][:n], b[keep][:n]
    assert len(y1) == n, "rejection reference undersampled"

    # Sharpest discriminator first: clipping stacks ~0.6% of colours onto
    # 0.0/1.0, which rejection sampling never does.
    on_edge = (x2 <= range2[0] + 1e-6) | (x2 >= range2[1] - 1e-6)
    assert on_edge.mean() < 1e-4, f"boundary pile-up {on_edge.mean():.4%}"

    assert stats.ks_2samp(x1, y1).pvalue > 1e-3
    assert stats.ks_2samp(x2, y2).pvalue > 1e-3
    assert abs(np.corrcoef(x1, x2)[0, 1] - np.corrcoef(y1, y2)[0, 1]) < 0.01


def test_simsed_cov_equals_redcor(tmp_path):
    """SIMSED_COV is the same correlation expressed as an absolute covariance.

    The manual documents both spellings (sec. 12130); SNANA keeps each as a
    covariance and divides by sigma1*sigma2 to recover the reduced correlation
    (snlc_sim.c:6213-6220). So SIMSED_COV(a,b) = rho*sigma_a*sigma_b must
    generate exactly what SIMSED_REDCOR(a,b) = rho does.
    """
    simsed = Path("tests/fixtures/snana/simsed_91bg_replay/input.input").read_text()
    rho, sig_stretch, sig_color = -0.656, 0.096, 0.175


    def parsed(line):
        return parse_snana_input_file(
            str(_write(tmp_path, simsed + line + "\n"))
        ).model._structured_config['simsed_config']["redcor"]

    redcor = parsed(f"SIMSED_REDCOR(stretch,color): {rho}")
    cov = parsed(f"SIMSED_COV(stretch,color): {rho * sig_stretch * sig_color}")

    assert redcor[("stretch", "color")] == (rho, False)
    value, is_cov = cov[("stretch", "color")]
    assert is_cov is True
    # both reduce to the same rho once divided by the sigmas
    assert abs(value / (sig_stretch * sig_color) - rho) < 1e-9


def test_simsed_redcor_aborts(tmp_path):
    """Malformed or unresolvable correlation keys must abort, not fall through.

    SNANA resolves both names against the declared SIMSED params and aborts
    unless exactly two match (snlc_sim.c:6178). Silently applying rho=0 -- or
    silently dropping a key whose form is slightly off -- gives an uncorrelated
    sample under a config that plainly asked for correlation.
    """

    simsed = Path("tests/fixtures/snana/simsed_91bg_replay/input.input").read_text()

    def parse(line):
        return parse_snana_input_file(str(_write(tmp_path, simsed + line + "\n")))

    # form the parser cannot resolve to a parameter pair
    for line in ("SIMSED_COVARIANCE(stretch,color): -0.01",
                 "SIMSED_REDCOR(stretch, color): -0.656",
                 "SIMSED_REDCOR: -0.656"):
        with pytest.raises(SNANAError, match="parameter pair"):
            parse(line)

    # well-formed, but names a parameter the model does not have: fatal at
    # model construction, as SNANA aborts while parsing the input
    # (parse_input_SIMSED_COV, snlc_sim.c:6169-6182)
    path = _write(tmp_path, simsed + "SIMSED_REDCOR(bogus,color): -0.656\n")
    with pytest.raises(SNANAError, match="not a continuous SIMSED parameter"):
        sim_run.prepare(str(path), **dict(n_sne=20))


def test_weaklens_preserves_zeroflux_sentinels(tmp_path):
    """Magnitude offsets must not move the 99.0 zero-flux flag.

    SNANA sums LENSDMU, MUSHIFT, GENMAG_OFF_* and SALT2gammaDM into one MAGOFF
    and applies it only when `genmag < 50.0 && genmag > -2.0` (genmag_offsets,
    snlc_sim.c:8325-8337), so epochs flagged as zero flux stay at exactly 99.
    Verified live at WEAKLENS_DSIGMADZ: 1.0 -- SNANA keeps all 29 such epochs at
    99.0, while applying the offset ungated smears them to 98.67, 98.71, ...
    which silently breaks any downstream `== 99` test for zero flux.
    """

    src = Path(_SALT3).read_text()
    # large scatter so any leak is unmistakable (sigma = z * DSIGMADZ)
    src = re.sub(r"^OMEGA_MATTER:", "WEAKLENS_DSIGMADZ: 1.0\nOMEGA_MATTER:",
                 src, count=1, flags=re.M)

    config = dict(n_sne=200)
    sim = sim_run.prepare(str(_write(tmp_path, src)), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    results, _, _ = sim_run.run_simulation(sim)

    lensdmu = np.asarray(results['parameters']['lensdmu'])
    assert np.abs(lensdmu).max() > 0.1, "weak lensing produced no scatter to test"

    mags = np.asarray(results['magnitudes'])
    sentinels = mags[mags > 50.0]
    assert sentinels.size > 0, "no zero-flux epochs generated"
    assert np.all(sentinels == 99.0), (
        f"zero-flux sentinels shifted: {np.unique(sentinels)[:5]}")


def test_mushift(tmp_path):
    """MUSHIFT offsets observed mags without touching the truth distance.

    A scalar shifts every SN by that value; <dmu0>:<dmu1> draws per SN. SNANA
    puts it in the gated MAGOFF sum, so zero-flux epochs stay at 99, and the
    manual notes SIM_DLMAG is not corrected for it.
    """

    base = Path(_SALT3).read_text()

    def run(extra):
        src = re.sub(r"^OMEGA_MATTER:", extra + "OMEGA_MATTER:", base,
                     count=1, flags=re.M)
        config = dict(n_sne=300)
        sim = sim_run.prepare(str(_write(tmp_path, src)), **config)
        simulation = None if sim is None else sim.config
        resources = sim
        results, _, _ = sim_run.run_simulation(sim)
        return results

    off, scalar, spread = run(""), run("MUSHIFT: 0.1\n"), run("MUSHIFT: -0.1:0.3\n")

    m_off = np.asarray(off['magnitudes'])
    m_scalar = np.asarray(scalar['magnitudes'])
    in_range = (m_off > -2) & (m_off < 50)
    assert np.allclose(m_scalar[in_range] - m_off[in_range], 0.1, atol=1e-5)
    assert np.all(m_scalar[m_off >= 50] == 99.0)
    assert np.allclose(np.asarray(off['parameters']['mu']),
                       np.asarray(scalar['parameters']['mu']))
    assert np.unique(np.asarray(scalar['parameters']['mushift'])) == [0.1]

    drawn = np.asarray(spread['parameters']['mushift'])
    assert len(np.unique(drawn)) == drawn.size
    assert drawn.min() >= -0.1 and drawn.max() <= 0.3
    assert abs(drawn.mean() - 0.1) < 0.02


def _write_hz_map(path, distort=None, *, radiation=True):
    """Two-column z H(z) map; `distort` multiplies the wCDM H(z)."""
    om, ol, h0 = 0.315, 0.685, 70.0
    z = np.concatenate([[0.0], np.geomspace(1e-4, 3.0, 600)])
    hz = h0 * np.asarray(hubble_parameter(
        jnp.asarray(z), om, 0.0, ol, radiation=radiation))
    if distort is not None:
        hz = hz * distort(z)
    Path(path).write_text(
        "\n".join(f"{a:.8f} {b:.6f}" for a, b in zip(z, hz)) + "\n")
    return str(path)


def test_hzfun_file(tmp_path):
    """HzFUN_FILE replaces the analytic H(z) for distances AND comoving volume.

    SNANA interpolates the map linearly (Hzfun_interp, OPT_INTERP=1) and uses it
    in both dLmag and dVdz, so a map changes the distance modulus and the
    generated redshift distribution. Scaling H by a constant is the one case
    that leaves GENZ alone -- dV/dz ~ D_M^2/H picks up a constant factor that
    cancels in the normalised CDF -- so it pins the distance effect exactly
    while a shape change is needed to see the volume effect.
    """

    base = Path(_SALT3).read_text()

    def run(extra, n=200):
        src = re.sub(r"^OMEGA_MATTER:", extra + "OMEGA_MATTER:", base,
                     count=1, flags=re.M)
        config = dict(n_sne=n)
        sim = sim_run.prepare(str(_write(tmp_path, src)), **config)
        simulation = None if sim is None else sim.config
        resources = sim
        results, _, _ = sim_run.run_simulation(sim)
        return results

    scaled = _write_hz_map(tmp_path / "hz_scaled.dat", lambda z: 1.10)
    off, mapped = run(""), run(f"HzFUN_FILE: {scaled}\n")

    assert np.allclose(np.asarray(off['parameters']['sim_redshift_cmb']),
                       np.asarray(mapped['parameters']['sim_redshift_cmb']))
    dmu = (np.asarray(mapped['parameters']['mu'])
           - np.asarray(off['parameters']['mu']))
    assert np.allclose(dmu, -5.0 * np.log10(1.10), atol=1e-4)


def test_hzfun_file_drives_comoving_volume():
    """The map must reach dV/dz, not only the distance modulus."""

    om, ol, h0 = 0.315, 0.685, 70.0
    hd = 2997.92458 * (100.0 / h0)
    ok = 1.0 - om - ol
    params = [{'model': 'POWERLAW', 'A': 2.5e-5, 'B': 1.5,
               'zmin': 0.01, 'zmax': 1.0}]
    z_grid = np.linspace(0.01, 1.0, 200)

    z = np.concatenate([[0.0], np.geomspace(1e-4, 3.0, 600)])
    e_shape = (np.sqrt(om * (1 + z) ** 3 + ol) * (1.0 + 0.60 * z / (1.0 + z)))
    hz_map = (jnp.asarray(z), jnp.asarray(e_shape))

    plain = dndz_density_on_grid(params, z_grid, om, ok, ol, hd, -1.0, 0.0,
                                 None, neutral_rate_mods())
    mapped = dndz_density_on_grid(params, z_grid, om, ok, ol, hd, -1.0, 0.0,
                                  hz_map, neutral_rate_mods())

    # normalise away the overall rate: the CDF *shape* must differ
    assert not np.allclose(plain / plain.sum(), mapped / mapped.sum(), atol=1e-4)


def test_anisotropy_modelname_v04(tmp_path):
    """ANISOTROPY_MODELNAME: V04 replaces the distance modulus with the Visser
    2004 dipole (dLmag_anisotropic, sntools_cosmology.c:583), leaving redshifts
    alone — it acts in dLmag only, never in the rate integral. Unset must leave
    mu bit-identical, and SNANA implements no other model name (:591)."""

    off = _run(_SALT3, "", tmp_path, n_sne=20)
    on = _run(_SALT3, "ANISOTROPY_MODELNAME: V04", tmp_path, n_sne=20)

    z_off = np.asarray(off['parameters']['sim_redshift_helio'])
    z_on = np.asarray(on['parameters']['sim_redshift_helio'])
    assert np.array_equal(z_off, z_on), "anisotropy must not move redshifts"

    mu_off = np.asarray(off['parameters']['mu'])
    mu_on = np.asarray(on['parameters']['mu'])
    assert not np.allclose(mu_off, mu_on), "V04 must change the distance modulus"
    # a low-z series against wCDM: same ballpark, not the same number
    assert np.abs(mu_on - mu_off).max() < 1.0

    with pytest.raises(SNANAError, match="ANISOTROPY_MODELNAME"):
        _run(_SALT3, "ANISOTROPY_MODELNAME: V99", tmp_path, n_sne=5)

    # With a HOSTLIB the host distance modulus is computed too (DMUCOR); SNANA
    # passes the SN's own galactic coords there (sntools_host.c:9009), so that
    # site takes the dipole as well.
    host = ("HOSTLIB_FILE: $JAX_TEST_FIXTURE_ROOT/hostlib/JAX_TEST.HOSTLIB\n"
            "GENRANGE_REDSHIFT: 0.0500 0.4000\n"
            "HOSTLIB_MSKOPT: 2\n")
    h_off = _run(_SALT3, host, tmp_path, n_sne=20)
    h_on = _run(_SALT3, host + "ANISOTROPY_MODELNAME: V04", tmp_path, n_sne=20)
    mu_h_off = np.asarray(h_off['parameters']['mu'])
    mu_h_on = np.asarray(h_on['parameters']['mu'])
    assert not np.allclose(mu_h_off, mu_h_on)
    assert np.all(np.isfinite(mu_h_on))


def _simlib_kept_libids(tmp_path, fieldlist, name, n_sne=64):
    """LIBIDs of the SNe that survive SIMLIB_FIELDLIST, and how many survive."""
    inp = tmp_path / name
    inp.write_text(Path(_SALT3).read_text() + f"\nSIMLIB_FIELDLIST: {fieldlist}\n")
    config = dict(
        n_sne=n_sne, random_seed=7)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    res, sel, _ = sim_run.run_simulation(sim)
    keep = np.asarray(sel['passing_mask']).astype(bool)
    lib = np.asarray(res['observation_schedule']['libid'])
    if lib.ndim > 1:
        lib = lib[:, 0]
    return sorted(set(lib[keep].tolist())), int(keep.sum())


def test_simlib_fieldlist_accepts_prescale_syntax(tmp_path):
    """`FIELD/PRESCALE` entries, '+' or ',' separated, must be parsed as fields.

    SNANA parses SIMLIB_FIELDLIST with parse_string_prescales (sntools.c:950).
    jax previously split on commas only and matched the whole string as one
    literal field name, so this form dropped every LIBID and left an empty
    SIMLIB.
    """
    for fl, name in (("TOY_A/1+TOY_B/1", "ps_both"), ("TOY_A,TOY_B", "ps_bare"),
                     ("ALL", "ps_all")):
        libs, _ = _simlib_kept_libids(tmp_path, fl, f"{name}.input")
        assert libs == [0, 1], f"{fl}: kept {libs}"
    # a key is a partial name match (get_string_dict OPT_DICT=1), unlike
    # FIELD_REQUIRE's exact strcmp
    assert _simlib_kept_libids(tmp_path, "TOY/1", "ps_partial.input")[0] == [0, 1]
    assert _simlib_kept_libids(tmp_path, "TOY_A/1", "ps_a.input")[0] == [0]


def _wide_simlib(tmp_path, n_libid=8):
    """SIMLIB with `n_libid` entries all in one field, so a /N prescale bites."""
    src = Path("tests/fixtures/data/simlib/JAX_TEST_LSST.SIMLIB").read_text()
    head, _, rest = src.partition("LIBID: 0")
    block = ("LIBID: 0" + rest).split("LIBID: 1")[0]
    out = [head]
    for i in range(n_libid):
        b = re.sub(r"^LIBID:\s*0", f"LIBID: {i}", block, count=1, flags=re.M)
        out.append(re.sub(r"END_LIBID:\s*0", f"END_LIBID: {i}", b))
    out.append("END_OF_SIMLIB:\n")
    path = tmp_path / "WIDE.SIMLIB"
    path.write_text("".join(out))
    return path


def test_simlib_fieldlist_prescale_selects_every_nth_entry(tmp_path):
    """`FIELD/N` keeps entries where LIBID % N == 0.

    Measured against stock SNANA on an 8-entry single-field library: /2 keeps
    LIBIDs {0,2,4,6} and /4 keeps {0,4}. SNANA additionally drops LIBID 0 after
    its first appearance -- a defect, since
    (LIBID+NWRAP)%N cannot treat 0 differently from 2/4/6 -- which jax does not
    reproduce, so the selected set matches while entry 0's weight does not.
    """
    lib = _wide_simlib(tmp_path)
    base = re.sub(r"^SIMLIB_FILE:.*$", f"SIMLIB_FILE: {lib}",
                   Path(_SALT3).read_text(), count=1, flags=re.M)

    def kept(fieldlist, tag):
        inp = tmp_path / f"{tag}.input"
        inp.write_text(base + f"\nSIMLIB_FIELDLIST: {fieldlist}\n")
        config = dict(
            n_sne=64, random_seed=7)
        sim = sim_run.prepare(str(inp), **config)
        simulation = None if sim is None else sim.config
        resources = sim
        res, sel, _ = sim_run.run_simulation(sim)
        keep = np.asarray(sel['passing_mask']).astype(bool)
        ids = np.asarray(res['observation_schedule']['libid'])
        if ids.ndim > 1:
            ids = ids[:, 0]
        return sorted(set(ids[keep].tolist()))

    assert kept("ALL", "w_all") == [0, 1, 2, 3, 4, 5, 6, 7]
    assert kept("TOY_A/2", "w_2") == [0, 2, 4, 6]
    assert kept("TOY_A/4", "w_4") == [0, 4]


def test_simlib_fieldlist_rejects_events_not_entries(tmp_path):
    """SIMLIB_FIELDLIST discards the *event* whose LIBID is excluded; the LIBID
    cycle itself is untouched (SKIP_SIMLIB_FIELD, snlc_sim.c:22212).

    Confirmed live: on a one-field 8-LIBID library SNANA keeps the full
    0..7 cycle and zeroes FLAG_ACCEPT for excluded entries, unlike
    SIMLIB_PRESCALE which removes entries from the cycle outright.
    """
    _, n_all = _simlib_kept_libids(tmp_path, "ALL", "rej_all.input")
    _, n_one = _simlib_kept_libids(tmp_path, "TOY_A", "rej_one.input")
    assert n_all == 64
    # half the events land on TOY_B and are discarded, not reassigned to TOY_A
    assert 0.4 * n_all < n_one < 0.6 * n_all, f"{n_one} of {n_all}"


def test_genmag_smear_mskopt_accepts_only_bit32(tmp_path):
    """Bit 32 is the only bit SNANA defines (sntools_genSmear.c:458) and it is
    output-neutral: it disables a recompute-skip that only triggers when the
    recomputation would be identical. Verified live — SNANA HEAD/PHOT are
    byte-identical with and without it. Other bits abort rather than being
    silently swallowed."""

    cfg = parse_snana_input_file(
        _write(tmp_path, Path(_SALT3).read_text() + "\nGENMAG_SMEAR_MSKOPT: 32\n"))
    assert cfg is not None

    for bad in (1, 3, 64):
        with pytest.raises(SNANAError, match="GENMAG_SMEAR_MSKOPT"):
            parse_snana_input_file(_write(
                tmp_path, Path(_SALT3).read_text() + f"\nGENMAG_SMEAR_MSKOPT: {bad}\n"))


def test_hzfun_file_drives_expected_count(tmp_path):
    """The map must also reach the NGEN_SEASON rate integral.

    SNcount_model integrates over dVdz(z, &INPUTS.HzFUN_INFO)
    (snlc_sim.c:17662), so H(z) sets how many events a season contains, not
    just where they sit in redshift. Counts are stock SNANA v12_01d-24-g563a1bc7
    "NGEN_SEASON=1.00 --> NGENTOT_LC" on this fixture; the no-map and flat-map
    entries are controls that a fix must leave at 1320.
    """

    base = re.sub(r"^NGENTOT_LC:.*$", "NGEN_SEASON: 1.0",
                  Path(_SALT3).read_text(), count=1, flags=re.M)

    def n_simulated(extra, name):
        src = re.sub(r"^OMEGA_MATTER:", extra + "OMEGA_MATTER:", base,
                     count=1, flags=re.M)
        path = tmp_path / f"{name}.input"
        path.write_text(src)
        cfg = parse_snana_input_file(str(path))
        return resolve_counts(cfg, cfg.hz_map())[0]

    # These SNANA reference counts use H(z) tables without radiation.
    cases = {
        'off':    ("", 1320),
        'flat':   (_write_hz_map(tmp_path / "c_flat.dat", radiation=False), 1320),
        'scaled': (_write_hz_map(tmp_path / "c_scaled.dat", lambda z: 1.10,
                                 radiation=False), 992),
        'shape':  (_write_hz_map(tmp_path / "c_shape.dat",
                                lambda z: 1.0 + 0.60 * z / (1.0 + z),
                                radiation=False), 1017),
    }
    for name, (hz, expected) in cases.items():
        got = n_simulated(f"HzFUN_FILE: {hz}\n" if hz else "", name)
        assert got == expected, f"{name}: {got} != SNANA {expected}"


def test_hzfun_file_requires_z0(tmp_path):
    """SNANA aborts unless the map's first bin is z=0 (sntools_cosmology.c:74)."""

    path = tmp_path / "hz_bad.dat"
    _write_hz_map(path)
    rows = path.read_text().splitlines()[1:]
    path.write_text("\n".join(rows) + "\n")

    src = re.sub(r"^OMEGA_MATTER:", f"HzFUN_FILE: {path}\nOMEGA_MATTER:",
                 Path(_SALT3).read_text(), count=1, flags=re.M)
    with pytest.raises(SNANAError, match="must be z=0"):
        sim_run.prepare(str(_write(tmp_path, src)), **dict(n_sne=10))


def test_hzfun_map_interpolates_linearly():
    """SNANA reads the map with interp_1DFUN OPT_INTERP=1, i.e. linear.

    A coarse map separates linear from nearest-node: between nodes the two
    disagree by O(node spacing), which a finely sampled map hides.
    """

    hz_map = (jnp.array([0.0, 1.0, 2.0]), jnp.array([1.0, 5.0, 9.0]))
    for z, expected in ((0.25, 2.0), (0.5, 3.0), (1.5, 7.0)):
        got = float(hubble_parameter(z, 0.315, 0.0, 0.685,
                                             hz_map=hz_map))
        assert abs(got - expected) < 1e-6, f"z={z}: {got} != {expected}"


def test_mushift_range_is_reweighted_only_under_a_z_dependent_rewgt(tmp_path):
    """A MUSHIFT range is drawn from the DNDZ reweight's CDF against MU when
    DNDZ_ZPOLY_REWGT or DNDZ_Z1POLY_REWGT has order above zero, and flat
    otherwise (prep_user_murewgt gate, snlc_sim.c:7650). The CDF grid runs
    from the lowest generated redshift in steps of 0.001; its distribution
    against SNANA's is tests/live_snana."""
    base = Path(_SALT3).read_text()

    def table(extra):
        cfg = parse_snana_input_file(str(_write(tmp_path, base + extra)))
        return _build_murewgt(cfg, None)

    for key in ("DNDZ_ZPOLY_REWGT", "DNDZ_Z1POLY_REWGT"):
        mu, cdf = table(f"MUSHIFT: -0.1:0.3\n{key}: 0,1\n")
        assert np.all(np.diff(mu) > 0) and np.all(np.diff(cdf) > 0)
    assert table("MUSHIFT: -0.1:0.3\n") is None
    assert table("MUSHIFT: 0.1\nDNDZ_Z1POLY_REWGT: 0,1\n") is None     # constant MUSHIFT
    assert table("MUSHIFT: -0.1:0.3\nDNDZ_ZEXP_REWGT: 1.0\n") is None   # not in SNANA's gate
    assert table("MUSHIFT: -0.1:0.3\nDNDZ_ZPOLY_REWGT: 1\n") is None    # order 0


def test_dndz_modifier_keys_are_not_silent(tmp_path):
    """DNDZ_* modifiers are applied, so they neither warn nor enter the rate list.

    `'DNDZ'` was a blanket entry in _OPEN_KEY_PREFIXES, exempting every DNDZ*
    sibling from the unrecognized-key warning — so `DNDZ_ALLSCALE: 2.0`, which
    doubles the SN count in SNANA, ran silently in jax. The same prefix also
    routed modifier keys into dndz_entries, the rate-model list, where they
    appeared as malformed entries beside the real rate model.
    """

    base = Path(_SALT3).read_text()


    def parse(extra):
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        log = logging.getLogger('jax_snlc_sim.config_parser')
        log.addHandler(handler)
        try:
            cfg = parse_snana_input_file(str(_write(tmp_path, base + extra)))
        finally:
            log.removeHandler(handler)
        return cfg.rates.dndz_entries, buf.getvalue()

    rate_only, quiet = parse("")
    assert rate_only == [['POWERLAW', 2.5e-05, 1.5]]
    assert 'DNDZ' not in quiet, "bare DNDZ should not warn"

    # implemented modifiers: applied, so they neither warn nor enter the list
    for key, extra in (('DNDZ_ALLSCALE', "DNDZ_ALLSCALE: 3.0\n"),
                       ('DNDZ_SCALE', "DNDZ_SCALE: 2.0 1.0\n"),
                       ('DNDZ_ZPOLY_REWGT', "DNDZ_ZPOLY_REWGT: 0,1\n")):
        entries, warned = parse(extra)
        assert entries == rate_only, f"{key} polluted dndz_entries: {entries}"
        assert key not in warned, f"{key} is implemented but warned"

    # DNDZ_FILE and DNDZ_PEC1A are primary rate keys, covered by their own
    # tests: neither may reach dndz_entries as a malformed modifier entry.


def test_dndz_rewgt_reshapes_the_redshift_cdf(tmp_path):
    """The REWGT polynomials must reach the drawn redshifts, not just the count.

    `genz_wgt` feeds both SNANA's expected-count integral and its z-CDF, so a
    z-dependent reweight has to change the sampled redshift distribution too.
    DNDZ_ALLSCALE is the control: a constant cancels in the normalised CDF, so
    it must leave the median exactly where it was.
    """
    base = Path(_SALT3).read_text()

    def cdf_median(extra, name):
        inp = tmp_path / name
        inp.write_text(base + extra)
        config = dict(n_sne=10, random_seed=7)
        sim = sim_run.prepare(str(inp), **config)
        simulation = None if sim is None else sim.config
        resources = sim
        z = np.asarray(resources.redshift_cdf_z)
        cdf = np.asarray(resources.redshift_cdf_vals)
        return float(np.interp(0.5, cdf, z))

    plain = cdf_median("", "plain.input")
    rewgt = cdf_median("DNDZ_ZPOLY_REWGT: 0,1\n", "rewgt.input")
    scaled = cdf_median("DNDZ_ALLSCALE: 2.0\n", "scaled.input")

    assert rewgt > plain + 0.01, (
        f"DNDZ_ZPOLY_REWGT left the redshift CDF unchanged "
        f"(median {rewgt:.4f} vs {plain:.4f})")
    assert abs(scaled - plain) < 1e-9, (
        f"DNDZ_ALLSCALE moved the redshift CDF median: {scaled} vs {plain}")


def _pec1a_sim(tmp_path, n_sne=600):
    """NON1ASED + one PEC1A template, flat nominal rate vs steeply rising PEC1A.

    Wide GENRANGE_REDSHIFT so (1+z)^beta dominates dV/dz and the two rate
    models give visibly different redshift distributions.
    """
    src = Path(_NON1A).read_text()
    src = src.replace("NON1A: 834 1.0 0.00 0.00 21",
                      "NON1A: 834 1.0 0.00 0.00 21\nPEC1A: 501 1.0 0.00 0.00 21")
    src = src.replace("DNDZ:  0.7*CC_S15",
                      "DNDZ:  POWERLAW 2.5E-5 0.0\nDNDZ_PEC1A: POWERLAW 1.0E-5 10.0")
    src = src.replace("GENRANGE_REDSHIFT:  0.0200  0.4000",
                      "GENRANGE_REDSHIFT:  0.0500  1.0000")
    src = src.replace("NGENTOT_LC:  1500", f"NGENTOT_LC:  {n_sne}")
    inp = tmp_path / "pec1a.input"
    inp.write_text(src)
    config = dict(n_sne=n_sne, random_seed=5)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    return simulation, resources, config, sim


def _cdf_median(resources, cdf_vals):
    return float(np.interp(0.5, np.asarray(cdf_vals),
                           np.asarray(resources.redshift_cdf_z)))


def test_dndz_pec1a_builds_a_second_redshift_cdf(tmp_path):
    """DNDZ_PEC1A gives the peculiar-Ia population its own dN/dz.

    set_RATEPAR (snlc_sim.c:14247) points the redshift draw at RATEPAR_PEC1A
    for PEC1A templates, so a steeper PEC1A rate must push its CDF to higher z.
    """
    _, resources, _, _ = _pec1a_sim(tmp_path)
    assert resources.redshift_cdf_vals_pec1a is not None

    nominal = _cdf_median(resources, resources.redshift_cdf_vals)
    pec1a = _cdf_median(resources, resources.redshift_cdf_vals_pec1a)
    assert pec1a > nominal + 0.1, (
        f"PEC1A CDF median {pec1a:.4f} vs nominal {nominal:.4f}")


def test_dndz_pec1a_events_draw_from_the_pec1a_rate(tmp_path):
    """PEC1A events' redshifts follow the PEC1A CDF, not the nominal one."""
    simulation, resources, config, sim = _pec1a_sim(tmp_path)
    results, _, _ = sim_run.run_simulation(sim)
    z = np.asarray(results['parameters']['sim_redshift_cmb'])
    is_pec1a = np.asarray(results['parameters']['non1a_index']) == 501
    assert is_pec1a.sum() > 100, "need a decent PEC1A sample"

    nominal = _cdf_median(resources, resources.redshift_cdf_vals)
    pec1a = _cdf_median(resources, resources.redshift_cdf_vals_pec1a)
    realised = float(np.median(z[is_pec1a]))
    assert abs(realised - pec1a) < abs(realised - nominal) / 5, (
        f"PEC1A median {realised:.4f}: PEC1A CDF {pec1a:.4f}, nominal {nominal:.4f}")


def _pec1a_multi_input(tmp_path, name, subs=()):
    src = Path(_NON1A).read_text()
    src = src.replace("NON1A: 834 1.0 0.00 0.00 21",
                      "NON1A: 834 3.0 0.00 0.00 21\nNON1A: 835 1.0 0.00 0.00 21\n"
                      "PEC1A: 501 1.0 0.00 0.00 21")
    src = src.replace("DNDZ:  0.7*CC_S15",
                      "DNDZ:  POWERLAW 2.5E-5 1.5\nDNDZ_PEC1A: POWERLAW 1.0E-5 1.5")
    for old, new in subs:
        src = src.replace(old, new)
    inp = tmp_path / name
    inp.write_text(src)
    return inp


def test_dndz_pec1a_generates_the_budgeted_counts(tmp_path):
    """The generated sample carries exactly the per-template budgets.

    Stock SNANA over 1000 events: 536 of INDEX 834, 179 of 835, 285 of PEC1A
    501.
    """
    inp = _pec1a_multi_input(tmp_path, "multi.input",
                             [("NGENTOT_LC:  1500", "NGENTOT_LC:  1000")])
    config = dict(n_sne=1000, random_seed=4)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    results, _, _ = sim_run.run_simulation(sim)
    idx = np.asarray(results['parameters']['non1a_index'])
    assert dict(collections.Counter(idx.tolist())) == {834: 536, 835: 179, 501: 285}


def test_dndz_pec1a_under_ngen_lc(tmp_path):
    """PEC1A quotas fill under NGEN_LC, where the batch budget is per-batch."""
    inp = _pec1a_multi_input(
        tmp_path, "multi_ngenlc.input",
        [("NGENTOT_LC:  1500", "NGEN_LC:  200"),
         ("APPLY_SEARCHEFF_OPT:  0", "APPLY_SEARCHEFF_OPT:  1")])
    config = dict(n_sne=200, random_seed=4)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    results, sel, _ = sim_run.run_simulation(sim)
    passing = np.asarray(sel['passing_mask']).astype(bool)
    idx = np.asarray(results['parameters']['non1a_index'])

    quotas = resources.model.generation_quotas(200)
    expected = dict(zip([834, 835, 501], quotas.tolist()))
    assert int(passing.sum()) == 200
    assert dict(collections.Counter(idx[passing].tolist())) == expected


# ---------------------------------------------------------------------------
# HOSTLIB_NREPEAT_GALID_SNPOS (sntools_host.c:6364)
# ---------------------------------------------------------------------------

_HOSTLIB_PATH = "$JAX_TEST_FIXTURE_ROOT/hostlib/JAX_TEST.HOSTLIB"
_HOSTLIB_LINE = (f"HOSTLIB_FILE: {_HOSTLIB_PATH}\n"
                 "GENRANGE_REDSHIFT: 0.0500 0.4000")


def _leader_of(params):
    """(leader global index array, per-SN is-leader mask)."""
    leader = np.asarray(params['host_leader_global_idx'])
    return leader, leader == np.arange(len(leader))


_NREPEAT_ON = f"{_HOSTLIB_LINE}\nHOSTLIB_NREPEAT_GALID_SNPOS: 3"


@pytest.fixture(scope="module")
def nrepeat_off(tmp_path_factory):
    """24 SNe with NREPEAT unset. Read-only, shared across the NREPEAT tests."""
    return _run(_SALT3, _HOSTLIB_LINE,
                tmp_path_factory.mktemp("nrep_off"), n_sne=24)['parameters']


@pytest.fixture(scope="module")
def nrepeat_on(tmp_path_factory):
    """The same 24 SNe with NREPEAT: 3."""
    return _run(_SALT3, _NREPEAT_ON,
                tmp_path_factory.mktemp("nrep_on"), n_sne=24)['parameters']


def test_nrepeat_reuses_the_block_leaders_host(nrepeat_off, nrepeat_on):
    """Blocks of N consecutive events share one galaxy and one SN-host offset.

    GEN_SNHOST_GALID, GEN_SNHOST_DDLR and GEN_SNHOST_POS all early-return when
    (NCALL_GEN_SNHOST_DRIVER % NREPEAT) > 0, so the galaxy and the drawn offset
    persist from the block leader. The four randoms burned at sntools_host.c:
    6369-6375 keep the stream synced, so a leader's host is exactly the host it
    would have drawn with the key off.
    """
    n, nrep = 24, 3
    off, on = nrepeat_off, nrepeat_on
    leader, _ = _leader_of(on)

    assert list(leader[:6]) == [0, 0, 0, 3, 3, 3]
    for key in ('host_GALID', 'host_SNSEP', 'host_DDLR', 'host_LOGMASS'):
        got = np.asarray(on[key], float)
        # the whole block carries the leader's value ...
        blocks = got.reshape(-1, nrep)
        assert np.all(blocks == blocks[:, :1]), key
        # ... and that value is what the leader drew with the key off
        assert np.array_equal(got, np.asarray(off[key], float)[leader]), key


def test_nrepeat_host_coords_still_track_each_event(nrepeat_on):
    """Host sky coords are NOT repeated: SORT_SNHOST_byDDLR is ungated and sets
    them from the current event's position (RA_SN + RA_SHIFT, sntools_host.c:
    9075), so they follow that event's field even inside a block."""
    ra = np.asarray(nrepeat_on['host_RA_GAL'], float).reshape(-1, 3)
    assert not np.all(ra == ra[:, :1])


def test_nrepeat_followers_report_raw_catalog_photoz(nrepeat_on):
    """Only block leaders get the ZGEN-ZTRUE shift.

    SORT_SNHOST_byDDLR reloads ZPHOT from the catalog unconditionally
    (sntools_host.c:9085); the gated GEN_SNHOST_ZPHOT then applies the shift
    (:7163) for leaders only. Measured against SNANA: followers within a block
    agree with each other and differ from their leader.
    """
    z = np.asarray(nrepeat_on['host_ZPHOT'], float).reshape(-1, 3)
    assert np.all(z[:, 1] == z[:, 2])
    assert np.all(z[:, 0] != z[:, 1])


def test_nrepeat_photoz_rule_is_the_same_under_genzphot(tmp_path, nrepeat_on):
    """The follower value is the raw catalog ZPHOT whether or not GENZPHOT is
    in use — GEN_SNHOST_ZPHOT is skipped wholesale, so the formula never runs
    for them. Verified against SNANA, where follower ZPHOT is byte-identical
    across both configurations while leaders differ."""
    plain = nrepeat_on
    calc = _run(_SALT3, _NREPEAT_ON + "\nHOSTLIB_GENZPHOT_FUDGEPAR: 0.02 0.0 0.0 0.0 0.0",
                tmp_path, n_sne=24)['parameters']
    zp = np.asarray(plain['host_ZPHOT'], float).reshape(-1, 3)
    zc = np.asarray(calc['host_ZPHOT'], float).reshape(-1, 3)
    assert np.allclose(zp[:, 1:], zc[:, 1:])       # followers: catalog either way
    assert not np.allclose(zp[:, 0], zc[:, 0])     # leaders: formula applies


def test_nrepeat_off_is_unchanged(tmp_path, nrepeat_off):
    """NREPEAT: 0 must leave every generated quantity untouched."""
    a = nrepeat_off
    b = _run(_SALT3, f"{_HOSTLIB_LINE}\nHOSTLIB_NREPEAT_GALID_SNPOS: 0",
             tmp_path, n_sne=24)['parameters']
    for key, va in a.items():
        if hasattr(va, 'dtype') and va.dtype.kind in 'fiu':
            assert np.array_equal(np.asarray(va), np.asarray(b[key])), key


def test_nrepeat_blocks_count_only_events_reaching_host_selection(tmp_path):
    """SNANA's counter advances in GEN_SNHOST_DRIVER, which sits behind
    `if (GENLC.NEPOCH < NEPMIN) return` (snlc_sim.c:12597). Events dropped by
    that gate are not part of any block, so blocks are indexed over
    prefilter_pass rather than over event number.

    CUTWIN_NEPOCH is raised so the gate actually bites — at the fixture's
    default of 1 nothing is dropped and block-by-event-number would be
    indistinguishable.
    """
    src = Path(_SALT3).read_text().replace(
        'CUTWIN_NEPOCH:   1 -10', 'CUTWIN_NEPOCH:   5 -10')
    inp = tmp_path / "nepgate.input"
    inp.write_text(_merge(src, f"{_HOSTLIB_LINE}\nHOSTLIB_NREPEAT_GALID_SNPOS: 3\n"))
    config = dict(
        n_sne=24, random_seed=5)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    results, _, _ = sim_run.run_simulation(sim)
    params = results['parameters']

    leader, is_leader = _leader_of(params)
    passing = np.asarray(params['prefilter_pass'], bool)
    assert 0 < (~passing).sum() < len(passing), "gate must drop some but not all"

    # every passing event's leader is itself a passing event ...
    assert passing[leader[passing]].all()
    # ... leaders are every NREPEAT-th passing event ...
    assert (is_leader & passing).sum() == -(-passing.sum() // 3)
    # ... and blocks straddle the dropped events, so at least one leader sits
    # further back than NREPEAT-1 places (which event-number indexing forbids)
    gaps = np.arange(len(leader))[passing] - leader[passing]
    assert gaps.max() > 2


# ---------------------------------------------------------------------------
# HOSTLIB_MSKOPT bit 8 (SN2GAL_RADEC) — sntools_host.c load_SNHOST_GENLC_COORDS
# ---------------------------------------------------------------------------

def test_sn2gal_radec_puts_the_sn_at_the_drawn_offset_not_the_centre(tmp_path):
    """SN2GAL_RADEC moves the SN onto its host at the position drawn *within*
    the galaxy, not at the galaxy centre.

    load_SNHOST_GENLC_COORDS sets GENLC.RA = SNHOSTGAL.RA_SN_DEG, which is the
    centre plus the Sersic offset. Assigning RA_GAL instead leaves the SN on the
    host centre while HOSTGAL_SNSEP still reports a non-zero separation — the
    written coords and the reported separation then disagree.
    """
    p = _run(_SALT3, f"{_HOSTLIB_LINE}\nHOSTLIB_MSKOPT: 8", tmp_path,
             n_sne=20)['parameters']
    ra, dec = np.asarray(p['ra'], float), np.asarray(p['dec'], float)
    gra, gdec = np.asarray(p['host_RA_GAL'], float), np.asarray(p['host_DEC_GAL'], float)
    implied = np.hypot((ra - gra) * np.cos(np.radians(gdec)) * 3600.0,
                       (dec - gdec) * 3600.0)
    snsep = np.asarray(p['host_SNSEP'], float)
    assert np.all(snsep > 0)
    assert np.allclose(implied, snsep, atol=0.02)


def test_sn2gal_radec_recomputes_mwebv_at_the_sn_position(tmp_path):
    """SNANA defers the MWEBV lookup for this case.

    gen_MWEBV is skipped in the event loop when SN2GAL_RADEC is set
    (snlc_sim.c:12611-12612) and run from load_SNHOST_GENLC_COORDS once the SN
    sits on its host, so extinction is sampled at the host position rather than
    the SIMLIB pointing. OPT_MWEBV=2 selects the SFD98 path, where position
    matters.

    The only test that reads the SFD maps -- every other dust test replays
    SNANA's own MWEBV from the dump. Both sides read the same map (real SFD98,
    or the synthetic stand-in in the fixture stub), so what is checked is the
    position the lookup uses, not the extinction values.
    """

    src = Path(_SALT3).read_text().replace('OPT_MWEBV:  0', 'OPT_MWEBV:  2')
    inp = tmp_path / "s2gmw.input"
    inp.write_text(_merge(src, f"{_HOSTLIB_LINE}\nHOSTLIB_MSKOPT: 8\n"))
    config = dict(
        n_sne=20, random_seed=5)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    results, _, _ = sim_run.run_simulation(sim)
    p = results['parameters']

    dustmap = SFDDustMap()
    at_sn = dustmap.get_ebv_batch(np.asarray(p['ra'], float),
                                  np.asarray(p['dec'], float))
    fields = resources.fields
    fi = np.asarray(p['field_idx'], int)
    at_field = dustmap.get_ebv_batch(
        np.array([fields[i].ra for i in fi]),
        np.array([fields[i].dec for i in fi]))

    assert np.allclose(np.asarray(p['mwebv'], float), at_sn, atol=1e-6)
    assert not np.allclose(at_sn, at_field, atol=1e-4)


def test_nrepeat_blocks_span_batches_under_ngen_lc(tmp_path):
    """NGEN_LC generates in several batches, so a block can straddle a batch
    boundary. The host-call count and the in-flight block's leader carry across
    `_generate_events`, so the block keeps its leader rather than restarting.
    """
    src = Path(_SALT3).read_text().replace("NGENTOT_LC:  1500", "NGEN_LC:  30")
    assert "NGEN_LC" in src, "fixture generation-count line changed"
    inp = tmp_path / "ngenlc.input"
    inp.write_text(_merge(src, f"{_HOSTLIB_LINE}\nHOSTLIB_NREPEAT_GALID_SNPOS: 3\n"))
    # batch size is batch_size (_ngen_lc_batch_alloc), so a small batch forces
    # several batches and makes blocks straddle the boundaries
    config = dict(
        n_sne=30, random_seed=5, batch_size=8)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    results, _, _ = sim_run.run_simulation(sim)
    params = results['parameters']

    leader = np.asarray(params['host_leader_global_idx'])
    # NGEN_LC instalments are capped at batch_size, so generating more than
    # one instalment's worth guarantees blocks straddled a batch boundary
    assert len(leader) > 8, f"needs multiple batches, generated {len(leader)}"
    galid = np.asarray(params['host_GALID'], float)
    passing = np.asarray(params['prefilter_pass'], bool)

    # every event shares its leader's galaxy, whichever batch the leader fell in
    assert np.array_equal(galid[passing], galid[leader][passing])
    # blocks of NREPEAT actually formed (not every event its own leader)
    n_leaders = int((leader[passing] == np.arange(len(leader))[passing]).sum())
    assert n_leaders == -(-int(passing.sum()) // 3)


def test_nrepeat_with_conditional_genpdf_forms_blocks(tmp_path):
    """NREPEAT blocks form over the events past the NEPOCH gate, with a
    conditional GENPDF reading each block's galaxy."""

    genpdf = tmp_path / "genpdf.dat"
    rows = [f"PDF: {c:+.3f} {lm:.1f} {np.exp(-0.5 * ((c - 0.1 * (lm - 10.0)) / 0.07) ** 2):.6f}"
            for lm in (8.0, 9.0, 10.0, 11.0, 12.0) for c in (-0.3, 0.0, 0.3)]
    genpdf.write_text("\n".join(["VARNAMES: SALT2c LOGMASS PROB", *rows]) + "\n")

    inp = tmp_path / "gp.input"
    inp.write_text(f"{Path(_SALT3).read_text()}\n{_HOSTLIB_LINE}\n"
                   f"GENPDF_FILE: {genpdf}\nHOSTLIB_STOREPAR: LOGMASS\n"
                   f"HOSTLIB_NREPEAT_GALID_SNPOS: 3\n")
    config = dict(
        n_sne=12, random_seed=5)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    results, _, _ = sim_run.run_simulation(sim)
    params = results['parameters']

    leader = np.asarray(params['host_leader_global_idx'])
    passing = np.asarray(params['prefilter_pass'], bool)
    galid = np.asarray(params['host_GALID'], float)
    assert np.array_equal(galid[passing], galid[leader][passing])
    n_leaders = int((leader[passing] == np.arange(len(leader))[passing]).sum())
    assert n_leaders == -(-int(passing.sum()) // 3)
    assert n_leaders < int(passing.sum())


def test_nrepeat_repeats_the_sn_position_under_sn2gal_radec(tmp_path, nrepeat_on):
    """The SNPOS half of the key. With MSKOPT bit 8 the SN is placed on its
    host, so a repeated galaxy plus a repeated offset means the SN's own sky
    position repeats too — measured in SNANA, where RA/DEC/SIM_RA/SIM_DEC all
    become block-constant with bit 8 set and vary without it."""
    on = _run(_SALT3, f"{_NREPEAT_ON}\nHOSTLIB_MSKOPT: 8",
              tmp_path, n_sne=24)['parameters']
    off = nrepeat_on

    for key in ('ra', 'dec'):
        blocks = np.asarray(on[key], float).reshape(-1, 3)
        assert np.all(blocks == blocks[:, :1]), key

    # The repeat is not simply making everything identical: events inside a
    # block still cycle through different SIMLIB pointings, and it is the move
    # onto the shared host that collapses their sky position.
    fields = np.asarray(on['field_idx']).reshape(-1, 3)
    assert not np.all(fields == fields[:, :1])
    # without bit 8 the SN keeps its field's position, which cycles within a block
    off_ra = np.asarray(off['ra'], float).reshape(-1, 3)
    assert not np.all(off_ra == off_ra[:, :1])


def test_nrepeat_followers_do_not_consume_the_galaxy_reuse_window(tmp_path):
    """MINDAYSEP_SAMEGAL blocks a galaxy from being reused within a time window.
    Repeat followers never reach GEN_SNHOST_GALID, so SNANA never registers
    them against that window and the two features coexist — measured live at
    MINDAYSEP_SAMEGAL 9999 with NREPEAT 3, which still gave clean blocks."""
    p = _run(_SALT3,
             f"{_HOSTLIB_LINE}\nHOSTLIB_NREPEAT_GALID_SNPOS: 3"
             f"\nHOSTLIB_MINDAYSEP_SAMEGAL: 9999",
             tmp_path, n_sne=24)['parameters']
    galid = np.asarray(p['host_GALID'], float).reshape(-1, 3)
    assert np.all(galid == galid[:, :1])
    # distinct galaxies across blocks — the reuse window still applies to leaders
    assert len(set(galid[:, 0].tolist())) == galid.shape[0]


def test_dumpall_keeps_host_columns_when_events_are_rejected(tmp_path):
    """SIMGEN_DUMPALL emits every generated event, host columns included.

    The row count comes from the full generated set while the columnar data
    holds only passing events, so a host column read from the latter is short
    the moment anything is rejected before generation and the writer indexes
    off the end. MINDAYSEP_SAMEGAL over a narrow redshift slice is the cheapest
    way to leave collisions unresolved, and no fixture sets it.
    """
    n_sne = 120
    inp = tmp_path / "in.input"
    inp.write_text(_merge(Path(_SALT3).read_text(),
                          f"HOSTLIB_FILE: {_HOSTLIB_PATH}\n"
                          "GENRANGE_REDSHIFT: 0.0500 0.0510\n"
                          "HOSTLIB_MINDAYSEP_SAMEGAL: 9000\n"
                          "SIMGEN_DUMPALL: 4 CID CUTMASK GALID GALZTRUE"))
    config = dict(
        n_sne=n_sne, random_seed=5)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    res, sel, spectra_cache = sim_run.run_simulation(sim)
    out = str(tmp_path / "out")
    os.makedirs(out, exist_ok=True)
    sim_run.save_snana_outputs(sim, res, "D", sel, spectra_cache=spectra_cache, output_dir=out)

    rows = [l.split()[1:] for l in open(f"{out}/D/D.DUMP") if l.startswith("SN:")]
    assert len(rows) == n_sne, "DUMPALL emits one row per generated event"
    cutmask = [r[1] for r in rows]
    assert cutmask.count('0') > 0, "config must actually reject events"
    # GALID/GALZTRUE resolve for rejected rows too; a short column would have
    # taken the writer past the end long before here.
    assert all(int(r[2]) > 0 for r in rows)
    assert len({r[2] for r in rows}) < n_sne, "rejects share the colliding hosts"


_SPECTRO = "tests/fixtures/snana/simlib_spectro_replay/input.input"


def _write_out(simulation, resources, config, sim, tmp_path, tag, extra_res=None):
    """Run, select and write; return the output directory. Selecting the
    finished run again needs every event (keep_rejected_events)."""
    spectra_cache = None
    res = extra_res
    if res is None:
        res, _, spectra_cache = sim_run.run_simulation(sim)
    sel = sim_run.apply_selection(sim, res)
    d = str(tmp_path / tag)
    os.makedirs(d, exist_ok=True)
    sim_run.save_snana_outputs(sim, res, "V", sel, spectra_cache=spectra_cache, output_dir=d)
    return d


def test_cidran_randomises_the_written_cid_but_not_the_derived_ones(tmp_path):
    """FORMAT_MASK CIDRAN(16) is SNANA's GENLC.CID_FINAL (snlc_sim.c:12739): it
    changes the CID written to the data files and nothing else. GALID_UNIQUE is
    built from GENLC.CID (set_GALID_UNIQUE, sntools_host.c:9257), so it keeps
    tracking the internal sequential CID, which is what SNANA's own output does.
    """
    inp = tmp_path / "in.input"
    inp.write_text(_merge(Path(_SALT3).read_text(),
                          f"HOSTLIB_FILE: {_HOSTLIB_PATH}\n"
                          "GENRANGE_REDSHIFT: 0.0500 0.4000\n"
                          "HOSTLIB_GALID_UNIQUE: 1\n"
                          "FORMAT_MASK: 48"))
    config = dict(
        n_sne=40, random_seed=5, keep_rejected_events=True)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    d = _write_out(simulation, resources, config, sim, tmp_path, "cidran")

    with fits.open(resolve_fits_path(f"{d}/V/V_HEAD.FITS")) as h:
        snid = np.asarray(h[1].data['SNID']).astype(np.int64)
        objid = np.asarray(h[1].data['HOSTGAL_OBJID_UNIQUE']).astype(np.int64)
    assert snid.max() > 1000, "CIDRAN should not leave sequential CIDs"
    real = objid > 0
    assert real.any(), "fixture must actually carry GALID_UNIQUE"
    # the internal CID is sequential, so this tracks it rather than SNID
    assert not np.array_equal(objid[real] // 57, snid[real])
    assert set((objid[real] // 57).tolist()) <= set(range(1, 41))


def test_spectra_prescale_is_unaffected_by_cidran(tmp_path):
    """WRSPEC_PRESCALE tests the internal CID (apply_prescale_GENSPEC,
    snlc_sim.c:21514), so which SNe keep spectra cannot depend on whether the
    written CID was randomised. Reading the prescale off the written CID drops
    a different, smaller subset.
    """
    counts = []
    for fmt in (32, 48):
        inp = tmp_path / f"in{fmt}.input"
        inp.write_text(_merge(Path(_SPECTRO).read_text(),
                              f"WRSPEC_PRESCALE: 3\nFORMAT_MASK: {fmt}"))
        config = dict(
            n_sne=60, random_seed=5, keep_rejected_events=True)
        sim = sim_run.prepare(str(inp), **config)
        simulation = None if sim is None else sim.config
        resources = sim
        d = _write_out(simulation, resources, config, sim, tmp_path, f"ps{fmt}")
        with fits.open(resolve_fits_path(f"{d}/V/V_SPEC.FITS")) as h:
            counts.append(len(h[1].data))
    assert counts[0] > 0, "fixture must produce spectra"
    assert counts[0] == counts[1], (
        f"CIDRAN changed the prescaled spectra set: {counts[0]} vs {counts[1]}")


def test_nepoch_taking_every_event_aborts_like_snana(tmp_path):
    """SNANA aborts when NEPOCH rejects every generated event, naming the likely
    cause (simEnd, snlc_sim.c:555), because that usually means the peak-MJD
    range misses the SIMLIB rather than that the cuts are strict. A wipeout by
    any other cut is not an error there, so the abort is keyed on the rejection
    code rather than on the survivor count.
    """
    inp = tmp_path / "in.input"
    inp.write_text(_merge(Path(_SALT3).read_text(),
                          "CUTWIN_NEPOCH: 9999 -10"))
    config = dict(
        n_sne=20, random_seed=5)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    res, sel, spectra_cache = sim_run.run_simulation(sim)
    out = str(tmp_path / "out")
    os.makedirs(out, exist_ok=True)
    with pytest.raises(SNANAError, match="NEPOCH"):
        sim_run.save_snana_outputs(sim, res, "N", sel, spectra_cache=spectra_cache, output_dir=out)


def test_an_event_with_no_model_flux_on_any_epoch_is_nepoch(tmp_path):
    """SNANA rejects as NEPOCH a second time after the fluxes, when no epoch
    carries a model flux (NOBS_MODELFLUX == 0, snlc_sim.c:331), before the
    trigger. SALT3 starts at 2000 A, above LSST-u's blue edge from z = 0.33
    and g's from 0.72, so a ug sim beyond that has epochs on every event and
    a flux on none.
    """
    inp = tmp_path / "in.input"
    inp.write_text(_merge(Path(_SALT3).read_text(),
                          "GENFILTERS: ug\nGENRANGE_REDSHIFT: 0.75 0.80\n"
                          "APPLY_SEARCHEFF_OPT: 1"))
    config = dict(n_sne=20, random_seed=5)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    res, sel, spectra_cache = sim_run.run_simulation(sim)
    assert int(sel['n_passed']) == 0
    assert np.all(sel['generated']['rejection_codes'] == 2)
    # the magnitudes were generated, so the epoch statistics are theirs
    assert np.all(np.asarray(sel['reached_flux']))


def test_wipeout_by_another_cut_is_not_an_abort(tmp_path):
    """The abort is keyed on the NEPOCH rejection code, not on the survivor
    count: SNANA lets a wipeout by any other cut write its output and finish.
    CUTWIN_MWEBV excludes every generated MWEBV, so all events fail CUTWIN
    and none fail NEPOCH.
    """
    inp = tmp_path / "in.input"
    inp.write_text(_merge(Path(_SALT3).read_text(),
                          "CUTWIN_MWEBV: 1.0 2.0\nAPPLY_CUTWIN_OPT: 1"))
    config = dict(
        n_sne=20, random_seed=5)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    res, sel, spectra_cache = sim_run.run_simulation(sim)
    assert int(sel['n_passed']) == 0, "fixture must reject every event"
    out = str(tmp_path / "out")
    os.makedirs(out, exist_ok=True)
    summary = sim_run.save_snana_outputs(sim, res, "W", sel, spectra_cache=spectra_cache, output_dir=out)
    assert summary['n_passed'] == 0


def test_plain_dump_selection_columns_track_the_rows_it_emits(tmp_path):
    """Selection columns are per generated event, the plain DUMP is per passing
    event, so they need the same subsetting the parameters get. Emitting only
    accepted events means every FLAG_ACCEPT must read 1; misalignment shows up
    as zeros against SNANA's own invariant (its DUMP has FLAG_ACCEPT set iff
    CUTMASK is all-pass, with no exceptions).
    """
    inp = tmp_path / "in.input"
    text = _merge(Path(_SALT3).read_text(),
                  f"HOSTLIB_FILE: {_HOSTLIB_PATH}\n"
                  "GENRANGE_REDSHIFT: 0.0500 0.0510\n"
                  "HOSTLIB_MINDAYSEP_SAMEGAL: 9000\n"
                  "SIMGEN_DUMP: 2 CID CUTMASK")
    inp.write_text("\n".join(l for l in text.splitlines()
                             if not l.startswith("SIMGEN_DUMPALL")))
    config = dict(
        n_sne=120, random_seed=5)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    res, sel, spectra_cache = sim_run.run_simulation(sim)
    out = str(tmp_path / "out")
    os.makedirs(out, exist_ok=True)
    sim_run.save_snana_outputs(sim, res, "P", sel, spectra_cache=spectra_cache, output_dir=out)

    hdr, rows = None, []
    for line in open(f"{out}/P/P.DUMP"):
        t = line.split()
        if t[:1] == ['VARNAMES:']:
            hdr = t[1:]
        elif t[:1] == ['SN:']:
            rows.append(t[1:])
    assert len(rows) < 120, "config must actually reject events"
    flag = [r[hdr.index('FLAG_ACCEPT')] for r in rows]
    assert set(flag) == {'1'}, "a passing-only DUMP cannot carry FLAG_ACCEPT=0"


_HOSTLIB_NBR_LINE = "HOSTLIB_FILE: $JAX_TEST_FIXTURE_ROOT/hostlib/JAX_TEST_NBR.HOSTLIB"


def test_neighbour_hosts_are_reported_at_the_sn_sky_position(tmp_path):
    """Host coords follow the SN even when the HOSTLIB carries NBR_LIST.

    SORT_SNHOST_byDDLR sets DDLR_SORT[i].RA = RA_SN + RA_SHIFT unconditionally
    (sntools_host.c:9075), so reported hosts sit beside the SN rather than at
    their catalog coordinates. JAX_TEST_NBR.HOSTLIB spreads galaxies ~0.6 deg
    around the two pointings, so a missing shift shows up as host coords
    degrees away from the SN instead of arcseconds.
    """
    p = _run(_SALT3, _HOSTLIB_NBR_LINE, tmp_path, n_sne=40)['parameters']
    assert 'host_HOSTGAL2_RA_GAL' in p, "fixture must actually carry neighbours"

    ra = np.asarray(p['host_RA_GAL'], float)
    dec = np.asarray(p['host_DEC_GAL'], float)
    sep = np.full(len(ra), np.inf)
    for cra, cdec in ((10.0, -30.0), (180.0, -45.0)):
        d = np.hypot((ra - cra) * np.cos(np.radians(dec)) * 3600.0,
                     (dec - cdec) * 3600.0)
        sep = np.minimum(sep, d)
    # catalog coords would sit ~1000 arcsec out; SN-relative ones are arcsec-scale
    assert np.median(sep) < 30.0, f"median host offset {np.median(sep):.1f} arcsec"


def test_neighbour_and_host_are_distinct_positions(tmp_path):
    """HOSTGAL and HOSTGAL2 are different galaxies and must not be written at
    one point — the failure mode of SNANA bug 23, which jax must not copy."""
    p = _run(_SALT3, _HOSTLIB_NBR_LINE, tmp_path, n_sne=40)['parameters']
    ra1 = np.asarray(p['host_RA_GAL'], float)
    ra2 = np.asarray(p['host_HOSTGAL2_RA_GAL'], float)
    gid1 = np.asarray(p['host_GALID'], float)
    gid2 = np.asarray(p['host_HOSTGAL2_GALID'], float)

    real = gid2 > 0
    assert real.sum() > 0, "fixture must yield real neighbours"
    assert np.all(gid1[real] != gid2[real])
    assert np.any(ra1[real] != ra2[real]), "distinct galaxies share a position"


def test_sim_redshift_host_match_is_the_matched_host_redshift(tmp_path):
    """SIM_REDSHIFT_HOST_MATCH is the DDLR-matched host's redshift
    (SNHOSTGAL_DDLR_SORT[0].REDSHIFT, snlc_sim.c:25113), not a flag. Live SNANA
    on a comparable fixture writes a float spanning 0.019-0.394; jax wrote a
    hardcoded 1.

    SIM_REDSHIFT_HOST is a different quantity: the drawn galaxy's own
    redshift (GENLC.REDSHIFT_HOST, set by GEN_SNHOST_GALID,
    sntools_host.c:6481), which is the matched one wherever the DDLR sort
    reports the drawn galaxy.
    """
    inp = tmp_path / "zhost.input"
    inp.write_text(f"{Path(_SALT3).read_text()}\n{_HOSTLIB_NBR_LINE}\n")
    config = dict(
        n_sne=40, random_seed=5)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    res, sel, spectra_cache = sim_run.run_simulation(sim)
    sim_run.save_snana_outputs(sim, res, "JAX_ZHOST", sel, spectra_cache=spectra_cache, output_dir=str(tmp_path))

    with fits.open(glob.glob(f"{tmp_path}/**/*HEAD.FITS*", recursive=True)[0]) as f:
        d = f[1].data
        zhel = np.asarray(d['SIM_REDSHIFT_HELIO'], float)
        zhost = np.asarray(d['SIM_REDSHIFT_HOST'], float)
        zmatch = np.asarray(d['SIM_REDSHIFT_HOST_MATCH'], float)
        drawn_matched = (np.asarray(d['HOSTGAL_OBJID'])
                         == np.asarray(d['SIM_HOSTLIB_GALID']))

    # a spread of real redshifts, not the constant it used to be
    assert len(set(np.round(zmatch, 6))) > 1
    assert zmatch.min() > 0.0 and zmatch.max() < 1.0
    # a host redshift tracks the SN's within HOSTLIB_DZTOL without matching it
    assert np.median(np.abs(zmatch - zhel)) < 0.1
    assert not np.allclose(zmatch, zhel, atol=1e-6)
    assert drawn_matched.any()
    np.testing.assert_array_equal(zhost[drawn_matched], zmatch[drawn_matched])
    assert not np.allclose(zhost, zhel, atol=1e-6)


def test_storepar_columns_and_always_present_host_vars(tmp_path):
    """HOSTLIB_STOREPAR writes SIM_HOSTLIB(<name>), and SNANA renames the four
    host-property basenames to <name>_TRUE on read (sndata.h:77-80,
    checkAlternateVarNames_HOSTLIB), so LOGMASS becomes LOGMASS_TRUE while
    LOG_SFR is untouched. Live SNANA on this HOSTLIB writes exactly
    SIM_HOSTLIB(LOGMASS_TRUE), SIM_HOSTLIB(LOG_SFR), SIM_HOSTLIB(ZTRUE).

    GALID and GALZTRUE are always read from a HOSTLIB, so SIMGEN_DUMP can name
    them with no STOREPAR entry.
    """
    base = [l for l in Path(_SALT3).read_text().splitlines()
            if not l.startswith('SIMGEN_DUMP')]
    inp = tmp_path / "sp.input"
    inp.write_text(_merge("\n".join(base),
                          f"{_HOSTLIB_LINE}\n"
                          "HOSTLIB_STOREPAR: LOGMASS,LOG_SFR,ZTRUE\n"
                          "SIMGEN_DUMP: 4 CID GALID GALZTRUE LOGMASS\n"))
    config = dict(
        n_sne=20, random_seed=5)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    res, sel, spectra_cache = sim_run.run_simulation(sim)
    sim_run.save_snana_outputs(sim, res, "JAX_SP", sel, spectra_cache=spectra_cache, output_dir=str(tmp_path))

    with fits.open(glob.glob(f"{tmp_path}/**/*HEAD.FITS*", recursive=True)[0]) as f:
        d = f[1].data
        names = [c for c in d.names if c.startswith('SIM_HOSTLIB(')]
        galid = np.asarray(d['SIM_HOSTLIB_GALID'])
        stored = {n: np.asarray(d[n], float) for n in names}

    assert names == ['SIM_HOSTLIB(LOGMASS_TRUE)', 'SIM_HOSTLIB(LOG_SFR)',
                     'SIM_HOSTLIB(ZTRUE)']

    # each row carries its own galaxy's HOSTLIB values
    hostlib = os.path.expandvars(_HOSTLIB_PATH)
    cols, rows = None, []
    with open_text(hostlib) as stream:
        lines = stream.read().splitlines()
    for line in lines:
        if line.startswith('VARNAMES:'):
            cols = line.split()[1:]
        elif line.startswith('GAL:'):
            rows.append(line.split()[1:])
    table = {c: np.array([float(r[i]) for r in rows]) for i, c in enumerate(cols)}
    where = {int(g): i for i, g in enumerate(table['GALID'].astype(np.int64))}
    idx = np.array([where[int(g)] for g in galid])
    for name, src in (('SIM_HOSTLIB(LOGMASS_TRUE)', 'LOGMASS'),
                      ('SIM_HOSTLIB(LOG_SFR)', 'LOG_SFR'),
                      ('SIM_HOSTLIB(ZTRUE)', 'ZTRUE')):
        assert np.allclose(stored[name], table[src][idx], atol=1e-5), name

    dump = Path(glob.glob(f"{tmp_path}/**/*.DUMP", recursive=True)[0]).read_text()
    header = next(l for l in dump.splitlines() if l.startswith('VARNAMES:')).split()
    assert 'GALID' in header and 'GALZTRUE' in header
    # a DUMP request for LOGMASS is renamed too, so the written column is
    # LOGMASS_TRUE — live SNANA writes exactly that for these inputs
    assert 'LOGMASS_TRUE' in header and 'LOGMASS' not in header


def test_zp_fluxcal_header_key_and_flux_scaling(tmp_path):
    """SNANA writes ZP_FLUXCAL into the FITS primary header
    (sntools_dataformat_fits.c:1042) and reads it back with a 27.5 fallback
    (:4203); without the key a non-default zeropoint is silently misread.

    Measured live: SNANA at ZP_FLUXCAL 30 vs 27.5 scales both FLUXCAL and
    FLUXCALERR by exactly 10.0, and writes 30.0 / 27.5 into the header. The
    default is 31.4 since v12_02 (ZEROPOINT_FLUXCAL_DEFAULT, sndata.h:49),
    so the same test now runs 30.0 against 31.4.
    """
    def run(extra, tag):
        inp = tmp_path / f"{tag}.input"
        inp.write_text(Path(_SALT3).read_text() + "\n" + extra + "\n")
        config = dict(
            n_sne=20, random_seed=5)
        sim = sim_run.prepare(str(inp), **config)
        simulation = None if sim is None else sim.config
        resources = sim
        res, sel, spectra_cache = sim_run.run_simulation(sim)
        sim_run.save_snana_outputs(sim, res, f"JAX_{tag}", sel, spectra_cache=spectra_cache, output_dir=str(tmp_path / tag))
        head = glob.glob(f"{tmp_path}/{tag}/**/*HEAD.FITS*", recursive=True)[0]
        phot = glob.glob(f"{tmp_path}/{tag}/**/*PHOT.FITS*", recursive=True)[0]
        with fits.open(head) as h, fits.open(phot) as p:
            return (h[0].header.get('ZP_FLUXCAL'), p[0].header.get('ZP_FLUXCAL'),
                    np.asarray(p[1].data['FLUXCAL'], float),
                    np.asarray(p[1].data['FLUXCALERR'], float),
                    np.asarray(p[1].data['MJD'], float))

    zh0, zp0, f0, e0, mjd = run("", "base")
    zh1, zp1, f1, e1, _ = run("ZP_FLUXCAL: 30.0", "zp30")

    assert (zh0, zp0) == (31.4, 31.4), "default zeropoint must be written as 31.4"
    assert (zh1, zp1) == (30.0, 30.0)

    keep = (f0 != 0) & (mjd > 0)
    assert keep.sum() > 50
    ratio = 10.0 ** (0.4 * (30.0 - 31.4))
    assert np.allclose(f1[keep] / f0[keep], ratio, rtol=1e-4)
    assert np.allclose(e1[keep] / e0[keep], ratio, rtol=1e-4)


def test_no_fabricated_output_name(tmp_path):
    """GENVERSION has no SNANA default, so jax must not invent one. It used to
    fall back to 'JAX_SIM', writing a run to a directory nobody asked for."""
    base = [l for l in Path(_SALT3).read_text().splitlines()
            if not l.startswith('GENVERSION')]
    inp = tmp_path / "nogv.input"
    inp.write_text("\n".join(base) + "\n")
    assert parse_snana_input_file(str(inp)).output.genversion == ""

    inp2 = tmp_path / "gv.input"
    inp2.write_text("\n".join(base) + "\nGENVERSION: MY_RUN\n")
    assert parse_snana_input_file(str(inp2)).output.genversion == "MY_RUN"


def test_malformed_genrange_redshift_aborts(tmp_path):
    """A GENRANGE_REDSHIFT that is present but unparseable used to become an
    invented (0.01, 1.2); SNANA's own default is 0.0 0.0 (snlc_sim.c:809-810)."""
    base = [l for l in Path(_SALT3).read_text().splitlines()
            if not l.startswith('GENRANGE_REDSHIFT')]
    inp = tmp_path / "badz.input"
    inp.write_text("\n".join(base) + "\nGENRANGE_REDSHIFT: not_a_range\n")
    with pytest.raises(SNANAError, match="GENRANGE_REDSHIFT"):
        parse_snana_input_file(str(inp))


def test_absent_genrange_redshift_aborts(tmp_path):
    """Absent behaves like malformed. The malformed case was fixed earlier, but
    an absent key still fell through to the field default and simulated
    0.01 < z < 1.2 silently. SNANA leaves 0,0 and aborts (snlc_sim.c:26647):
    'Invalid zmin=-1.233587e-03 (must be > 0)'."""
    base = [l for l in Path(_SALT3).read_text().splitlines()
            if not l.startswith('GENRANGE_REDSHIFT')]
    inp = tmp_path / "noz.input"
    inp.write_text("\n".join(base) + "\n")
    with pytest.raises(SNANAError, match="zmin"):
        parse_snana_input_file(str(inp))


def test_mindaysep_samegal_default_matches_snana(tmp_path):
    """SNANA inits HOSTLIB_MINDAYSEP_SAMEGAL to 9999999 (snlc_sim.c:1152); jax
    carried 99999 in two of its three sites, 100x short."""
    inp = tmp_path / "h.input"
    inp.write_text(f"{Path(_SALT3).read_text()}\n{_HOSTLIB_LINE}\n")
    assert parse_snana_input_file(str(inp)).host.mindaysep_samegal == 9_999_999


def test_simsed_keys_are_recorded_as_used():
    """The parser pools SIMSED_PARAM/SHAPEPAR/COLORPAR/COLORLAW and
    SIMSED_GRIDONLY into one list each. Pool under the canonical key name, as
    DNDZ already does, so `used` records what the parser consumed — pooling
    under an invented '_simsed_params' left those keys looking untouched.

    Snapshot `used` first: `k in config` is itself a tracked lookup.
    """
    raw = read_keys("tests/fixtures/snana/simsed_91bg_replay/input.input")
    build(raw)
    used = set(raw.used)
    assert 'SIMSED_PARAM' in used
    assert 'SIMSED_GRIDONLY' in used


def test_magmonitor_snr_column_gating_and_text(tmp_path):
    """MAGMONITOR_SNR: column presence, the >10 value gate, and TEXT VARLIST.

    SNANA creates the SIM_SNRMAG<nn> column whenever the key is non-zero
    (snlc_sim.c:7286-7289) but only fills it above mag 10 (:27438), so a small
    value yields a column of zeros rather than no column. The name is
    zero-padded to two digits, and the same column appears in the TEXT format
    (sntools_dataformat_text.c:955-957).
    """
    def run(extra, tag, fmt=32):
        inp = tmp_path / f"{tag}.input"
        inp.write_text(_merge(Path(_SALT3).read_text(),
                              f"FORMAT_MASK: {fmt}\n" + extra))
        config = dict(
            n_sne=15, random_seed=5)
        sim = sim_run.prepare(str(inp), **config)
        simulation = None if sim is None else sim.config
        resources = sim
        res, sel, spectra_cache = sim_run.run_simulation(sim)
        d = tmp_path / tag
        sim_run.save_snana_outputs(sim, res, "B", sel, spectra_cache=spectra_cache, output_dir=str(d))
        return d

    d_on = run("MAGMONITOR_SNR: 22", "on")
    d_low = run("MAGMONITOR_SNR: 5", "low")
    d_off = run("", "off")

    def phot_cols(d):
        phot = glob.glob(f"{d}/**/*PHOT.FITS*", recursive=True)[0]
        with fits.open(phot) as p:
            return list(p[1].data.names), p[1].data
    def head_key(d):
        head = glob.glob(f"{d}/**/*HEAD.FITS*", recursive=True)[0]
        with fits.open(head) as h:
            return h[0].header.get('SIM_VARNAME_SNRMON')

    cols_on, data_on = phot_cols(d_on)
    assert 'SIM_SNRMAG22' in cols_on
    assert head_key(d_on) == 'SIM_SNRMAG22'
    v = np.asarray(data_on['SIM_SNRMAG22'], float)
    assert (v > 0).sum() > 20, "monitor SNR never evaluated"

    # value <= 10: column exists, values stay zero (SNANA's asymmetric gate)
    cols_low, data_low = phot_cols(d_low)
    assert 'SIM_SNRMAG05' in cols_low, "zero-padded name expected for mag 5"
    assert head_key(d_low) == 'SIM_SNRMAG05'
    assert np.all(np.asarray(data_low['SIM_SNRMAG05'], float) == 0.0)

    # key absent: no column, no header key
    cols_off, _ = phot_cols(d_off)
    assert not [c for c in cols_off if 'SNRMAG' in c]
    # SNANA writes the key with an empty value when the monitor is off
    # (sntools_dataformat_fits.c:1286, sntools.c:8406)
    assert head_key(d_off) == ''

    # TEXT format carries the same column in VARLIST
    d_text = run("MAGMONITOR_SNR: 22", "text", fmt=2)
    dat = sorted(Path(d_text).rglob("*.DAT"))
    assert dat, "no TEXT light curves written"
    varlist = [l for l in dat[0].read_text().splitlines()
               if l.startswith("VARLIST:")][0]
    assert 'SIM_SNRMAG22' in varlist, varlist


def test_minseason_keeps_the_entry_but_yields_no_events(tmp_path):
    """SIMLIB_MINSEASON drops short seasons from an entry's bookkeeping, not the
    entry: SNANA leaves the LIBID in the cycle with an empty survey range
    (remove_short_SIMLIB_SEASON, snlc_sim.c:21241, :21306) so events still land
    on it and are rejected for want of epochs. Dropping the entry instead would
    shift which LIBID every later event draws.

    JAX_TEST_LSST_SEASON.SIMLIB has LIBID 0 = one 920 d season plus a 9 d one,
    LIBID 1 = two 9 d seasons.
    """
    res = _run(_SALT3,
               "SIMLIB_FILE: $JAX_TEST_FIXTURE_ROOT/simlib/JAX_TEST_LSST_SEASON.SIMLIB\n"
               "SIMLIB_MINSEASON: 30\n"
               "GENRANGE_PEAKMJD: 60560.0 61440.0",
               tmp_path, n_sne=60)
    libid = np.asarray(res['observation_schedule']['libid'])
    nobs = np.asarray(res['observation_schedule']['n_obs_per_sn'])

    assert set(np.unique(libid)) == {0, 1}, "the all-short LIBID left the cycle"
    assert (nobs[libid == 1] == 0).all(), "LIBID 1 has no surviving season"
    assert (nobs[libid == 0] > 0).any(), "LIBID 0 keeps its long season"


def test_simlib_survey_override_drives_both_the_name_and_the_id():
    """SIMLIB_SURVEY overwrites the SIMLIB header SURVEY while the header is
    read (snlc_sim.c:18167), and IDSURVEY is then taken from that overridden
    name (:18315). Both must move together: SNANA uses the same GENLC.IDSURVEY
    for the written header and for the RANSYSTPAR unsynced seed (:8113), so a
    name/ID pair drawn from different surveys shifts the systematic draws.

    CSP carries no `geo:` token in SURVEY.DEF, so this also covers the surveys
    `parse_survey_def` excludes.
    """
    simlib = "$JAX_TEST_FIXTURE_ROOT/simlib/JAX_TEST_LSST.SIMLIB"

    native_name, native_id = survey_from_simlib_file(simlib, "")
    assert native_name == "LSST"
    assert native_id == survey_id("LSST")

    over_name, over_id = survey_from_simlib_file(simlib, "CSP")
    assert over_name == "CSP"
    assert over_id == survey_id("CSP"), "IDSURVEY ignored SIMLIB_SURVEY"
    assert over_id != native_id, "fixture no longer discriminates"

    assert "CSP" not in parse_survey_def(), "CSP gained a geo: entry"

def _restage_head(src_dir, dst_dir, version, snana_version, vpec=None):
    """Copy a fit fixture, rewriting the HEAD's SNANA_VERSION and VPEC."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    with fits.open(resolve_fits_path(src_dir / "HEAD.FITS")) as h:
        h[0].header['SNANA_VERSION'] = snana_version
        if vpec is not None:
            h[1].data['VPEC'] = vpec[:len(h[1].data)]
        h.writeto(dst_dir / f"{version}_HEAD.FITS", overwrite=True)
    (dst_dir / f"{version}_PHOT.FITS").write_bytes(
        gzip.decompress((src_dir / "PHOT.FITS.gz").read_bytes()))
    (dst_dir / f"{version}.LIST").write_text(f"{version}_HEAD.FITS\n")


def test_restore_wrong_vpec_flips_the_sign_for_old_data(tmp_path):
    """SNANA wrote VPEC with the opposite sign below v11.02 and flips it on read
    unless RESTORE_WRONG_VPEC keeps the file's convention
    (DOFIX_WRONG_SIGN_VPEC snana.F90:8475, correct_sign_vpec_data sntools.c:3328).
    """

    # the version parse, against SNANA's own documented examples
    assert abs(snana_version_float('v10_78c') - 10.78) < 1e-5
    assert abs(snana_version_float('v11_04b-3-dstewtf') - 11.04) < 1e-5
    assert not vpec_sign_is_correct('v10_78c')
    assert vpec_sign_is_correct('v11_04b-3-dstewtf')
    assert vpec_sign_is_correct('UNKNOWN') and vpec_sign_is_correct('')

    # the version the sim writes into FITS headers must parse to the
    # current VPEC convention, or every reader flips the sign
    assert snana_version_float(SNANA_VERSION_STRING) >= 11.02
    assert vpec_sign_is_correct(SNANA_VERSION_STRING)

    src = Path(__file__).resolve().parents[1] / "fixtures/snana/salt3_fit_dense"
    version = "JAX_VPEC"

    # the fixture's VPEC column is all zeros, so give it a signal to flip
    peculiar = np.linspace(-400.0, 400.0, 100).astype(np.float32)

    def vpec(snana_version, restore):
        d = tmp_path / f"{snana_version}_{restore}".replace('.', '_') / version
        _restage_head(src, d, version, snana_version, vpec=peculiar)
        batch = load_and_batch_snana_fits(
            str(d / f"{version}_HEAD.FITS"), str(d / f"{version}_PHOT.FITS"),
            restore_wrong_vpec=restore)
        return np.asarray(batch.head_extras['VPEC'], dtype=float)

    named = vpec('v12_01d-24-g563a1bc7', False)
    old = vpec('v10_78c', False)
    kept = vpec('v10_78c', True)

    assert np.any(named != 0.0)
    # data written before v11.02 has its sign corrected on read ...
    np.testing.assert_allclose(old, -named)
    # ... unless the key asks for the file's own convention back
    np.testing.assert_allclose(kept, named)


def test_lfixpar_all_pins_every_parameter_and_reports_no_error(tmp_path):
    """LFIXPAR_ALL zeroes INISTP for every parameter (snlc_fit.F90:1979), so
    FLOATPAR is false throughout: the ERRMAX_BAD substitution is gated off
    (snana.F90:27878) and each parameter is reported at INIVAL with no error.
    Values checked against snlc_fit.exe on this fixture."""

    src = Path(__file__).resolve().parents[1] / "fixtures/snana/salt3_fit_dense"
    version = "JAX_TEST_SALT3_FIT_DENSE"
    vdir = tmp_path / version
    vdir.mkdir()
    for name in ("HEAD", "PHOT"):
        (vdir / f"{version}_{name}.FITS").write_bytes(
            gzip.decompress((src / f"{name}.FITS.gz").read_bytes()))
    (vdir / f"{version}.LIST").write_text(f"{version}_HEAD.FITS\n")

    env = os.environ.copy()
    env["JAX_TEST_FIXTURE_ROOT"] = str(
        Path(__file__).resolve().parents[1] / "fixtures/data")
    env["JAX_TEST_FIT_SIM_DIR"] = str(tmp_path)
    repo_root = Path(__file__).resolve().parents[2]

    def fit(name, *overrides):
        out = tmp_path / name
        r = subprocess.run(
            [sys.executable, str(repo_root / "snlc_fit.py"),
             str(src / "fit.nml"), "BATCH_SIZE", "4", *overrides, "MAX_FIT", "6",
             "OUTPUT_NAME", str(out)],
            capture_output=True, text=True, env=env, timeout=1800)
        assert r.returncode == 0, r.stdout[-1200:] + r.stderr[-1200:]
        names, rows = None, []
        for line in out.read_text().splitlines():
            if line.startswith("VARNAMES:"):
                names = line.split()[1:]
            elif line.startswith("SN:") and names:
                v = line.split()[1:]
                rows.append((float(v[names.index("x1")]),
                             float(v[names.index("c")]),
                             float(v[names.index("x1ERR")]),
                             float(v[names.index("cERR")]),
                             int(v[names.index("NDOF")])))
        return rows

    base = fit("EBBASE.FITRES.TEXT")
    assert len(base) == 6
    # a floated fit reports real errors on both
    assert all(x > 0.0 and c > 0.0 for _, _, x, c, _ in base)

    pinned = fit("EBPIN.FITRES.TEXT", "LFIXPAR_ALL", "T")
    # every event survives -- nothing is rejected for being held
    assert len(pinned) == 6
    # INIVAL_SHAPE 0.001 and INIVAL_COLOR 0.01, neither moved, neither errored
    assert all(abs(x1 - 0.001) < 1e-9 and abs(c - 0.01) < 1e-9
               and x1e == 0.0 and ce == 0.0 for x1, c, x1e, ce, _ in pinned)
    # nothing floats, so NDOF loses no parameters at all: NFITPAR_MN - NFIXPAR
    # is zero and NDOF is the epoch count (snlc_fit.F90:13102, :15678).
    # Values are snlc_fit.exe's on this fixture.
    assert [n for *_, n in pinned] == [22, 23, 24, 24, 24, 24]

    # the key only zeroes the ERRMAX_BAD floors (snlc_fit.F90:1481), which
    # changes nothing while every parameter is held
    off = fit("EBOFF.FITRES.TEXT", "LFIXPAR_ALL", "T", "MINOS_ERRMAX_BAD", "F")
    assert off == pinned


def test_errmax_bad_substitutes_the_declared_inistp(data_root):
    """A floated parameter whose error is unusable is reported at +/-INISTP and
    the event is kept (snana.F90:27878). The floors and steps are SNANA's
    FITPAR_INI values, carried by the model that owns them."""

    model = SALTModel.load(str(data_root / "models" / "SALT3.K21"), {})
    declared = model.get_fit_parameters()
    # ERRMAX_BAD (snlc_fit.F90:1242-1262) and INISTP (:6841-6846, :8139-8171)
    assert declared['x1']['errmax_bad'] == 0.001
    assert declared['c']['errmax_bad'] == 0.001
    assert declared['PKMJD']['errmax_bad'] == 1.0e-4
    assert declared['x0']['errmax_bad'] == 0.0
    assert declared['x1']['inistp'] == 0.3
    assert declared['c']['inistp'] == 0.03
    assert declared['PKMJD']['inistp'] == 2.0
    assert declared['x0']['inistp_relative_to_init'] == 0.1

    init = {'x1': np.array([0.5, 0.001, 0.001]),
            'c': np.array([0.01, 0.01, 0.01])}

    def results():
        return BatchFitResult(
            snids=np.array(['1', '2', '3']),
            params={'PKMJD': np.full(3, 55000.0), 'x0': np.full(3, 1e-5),
                    'x1': np.array([0.5, 1.0, -0.2]),
                    'c': np.array([0.0, 0.1, 0.2])},
            param_errors={'PKMJD': np.full(3, 0.5), 'x0': np.full(3, 1e-7),
                          'x1': np.array([0.11, 0.12, 0.13]),
                          'c': np.array([0.05, 1e-5, 0.05])},
            param_covariances=np.zeros((3, 4, 4)), cov_output_params=[],
            chi_squared=np.zeros(3), ndof=np.full(3, 4), prior_chi2=np.zeros(3),
            observation_mask=np.ones((3, 1), dtype=bool),
            hessian_cond=np.ones(3),
            hessian_positive=np.ones(3, dtype=bool),
            edm=np.zeros(3),
            z_prior=None, iter1_verdicts={})

    cfg = FitConfig()
    specs = model.get_fit_parameters()
    r = fit_run.errmax_bad(cfg, specs, (), results(), init)
    # SN 0's x1 never left its start; SN 1's colour error is under the floor
    np.testing.assert_allclose(r.param_errors['x1'], [0.3, 0.12, 0.13])
    np.testing.assert_allclose(r.param_errors['c'], [0.05, 0.03, 0.05])
    assert r.n_sne == 3, "the error is flagged, the fit is not rejected"

    # FLOATPAR gates the substitution, so a held parameter is untouched
    r = fit_run.errmax_bad(cfg, specs, ('x1', 'c'), results(), init)
    np.testing.assert_allclose(r.param_errors['x1'], [0.11, 0.12, 0.13])
    np.testing.assert_allclose(r.param_errors['c'], [0.05, 1e-5, 0.05])

    # MINOS_ERRMAX_BAD = F zeroes the floors alone, leaving the second
    # condition live: x1 is still substituted, the tiny colour error survives
    r = fit_run.errmax_bad(FitConfig(fit=Fit(minos_errmax_bad=False)), specs,
                           (), results(), init)
    np.testing.assert_allclose(r.param_errors['x1'], [0.3, 0.12, 0.13])
    np.testing.assert_allclose(r.param_errors['c'], [0.05, 1e-5, 0.05])

    # an explicit INISTP_<p> overrides the declared default, under the name
    # resolve_parameter_keys leaves on the config
    r = fit_run.errmax_bad(FitConfig(model=Model(inistp={'c': 0.5})), specs,
                           (), results(), init)
    np.testing.assert_allclose(r.param_errors['c'], [0.05, 0.5, 0.05])


def test_fitwin_chi2red_ini_abandons_events_before_the_fit(tmp_path):
    """FITWIN_CHI2RED_INI windows chi2 per epoch at the initial parameters and
    abandons the event outside it (snlc_fit.F90:9597). The 55 edge falls in a
    gap in this fixture's distribution -- nothing sits between 50 and 60 -- so
    the surviving set is stable rather than knife-edge. snlc_fit.exe drops
    CID 50 alone at that window, and every event with the window wide."""

    src = Path(__file__).resolve().parents[1] / "fixtures/snana/salt3_fit_dense"
    version = "JAX_TEST_SALT3_FIT_DENSE"
    vdir = tmp_path / version
    vdir.mkdir()
    for name in ("HEAD", "PHOT"):
        (vdir / f"{version}_{name}.FITS").write_bytes(
            gzip.decompress((src / f"{name}.FITS.gz").read_bytes()))
    (vdir / f"{version}.LIST").write_text(f"{version}_HEAD.FITS\n")

    env = os.environ.copy()
    env["JAX_TEST_FIXTURE_ROOT"] = str(
        Path(__file__).resolve().parents[1] / "fixtures/data")
    env["JAX_TEST_FIT_SIM_DIR"] = str(tmp_path)
    repo_root = Path(__file__).resolve().parents[2]

    def cids(name, window):
        out = tmp_path / name
        r = subprocess.run(
            [sys.executable, str(repo_root / "snlc_fit.py"), str(src / "fit.nml"), "BATCH_SIZE", "4",
             "FITWIN_CHI2RED_INI", window, "OUTPUT_NAME", str(out)],
            capture_output=True, text=True, env=env, timeout=1800)
        assert r.returncode == 0, r.stdout[-1200:] + r.stderr[-1200:]
        return {line.split()[1] for line in out.read_text().splitlines()
                if line.startswith("SN:")}

    wide = cids("CRWIDE.FITRES.TEXT", "0.0 1.0E6")
    assert len(wide) == 100, "a window this wide may not reject anything"

    cut = cids("CRCUT.FITRES.TEXT", "0.0 55.0")
    assert wide - cut == {"50"}, "snlc_fit.exe abandons CID 50 and no other"


def test_opt_sncid_list_bit0_skips_the_post_fit_cuts(tmp_path):
    """OPT_SNCID_LIST bit 0 returns from FITANA_CUTS before every post-fit cut
    (snlc_fit.F90:12801), so a cut that would reject events stops doing so.
    snlc_fit.exe on this fixture: 55 events survive CUTWIN_NFILT_TRESTMIN=6,
    and all 100 once the bit is set."""

    src = Path(__file__).resolve().parents[1] / "fixtures/snana/salt3_fit_dense"
    version = "JAX_TEST_SALT3_FIT_DENSE"
    vdir = tmp_path / version
    vdir.mkdir()
    for name in ("HEAD", "PHOT"):
        (vdir / f"{version}_{name}.FITS").write_bytes(
            gzip.decompress((src / f"{name}.FITS.gz").read_bytes()))
    (vdir / f"{version}.LIST").write_text(f"{version}_HEAD.FITS\n")
    cid_file = tmp_path / "cidlist.txt"
    cid_file.write_text("\n".join(str(i) for i in range(1, 101)) + "\n")

    env = os.environ.copy()
    env["JAX_TEST_FIXTURE_ROOT"] = str(
        Path(__file__).resolve().parents[1] / "fixtures/data")
    env["JAX_TEST_FIT_SIM_DIR"] = str(tmp_path)
    repo_root = Path(__file__).resolve().parents[2]

    def n_rows(name, *overrides):
        out = tmp_path / name
        r = subprocess.run(
            [sys.executable, str(repo_root / "snlc_fit.py"), str(src / "fit.nml"), "BATCH_SIZE", "4",
             "CUTWIN_NFILT_TRESTMIN", "6 100", *overrides, "OUTPUT_NAME", str(out)],
            capture_output=True, text=True, env=env, timeout=1800)
        assert r.returncode == 0, r.stdout[-1200:] + r.stderr[-1200:]
        return sum(1 for line in out.read_text().splitlines()
                   if line.startswith("SN:"))

    assert n_rows("SCBASE.FITRES.TEXT") == 55, "the cut must reject events"
    assert n_rows("SCBIT0.FITRES.TEXT",
                  "SNCID_LIST_FILE", str(cid_file),
                  "OPT_SNCID_LIST", "1") == 100


def test_opt_sncid_list_bit1_starts_each_event_at_the_list_files_fit(tmp_path):
    """OPT_SNCID_LIST bit 1 seeds each listed event's fit with the parameters
    SNCID_LIST_FILE holds for it (FITINI_LIST_FILE, snlc_fit.F90:9997): with
    snlc_fit.exe's own FITRES as the list, every event starts at SNANA's
    answer."""
    src = Path(__file__).resolve().parents[1] / "fixtures/snana/salt3_fit_dense"
    version = "JAX_TEST_SALT3_FIT_DENSE"
    vdir = tmp_path / version
    vdir.mkdir()
    for name in ("HEAD", "PHOT"):
        (vdir / f"{version}_{name}.FITS").write_bytes(
            gzip.decompress((src / f"{name}.FITS.gz").read_bytes()))
    (vdir / f"{version}.LIST").write_text(f"{version}_HEAD.FITS\n")
    config = fit_config(src / "fit.nml", private_data_path=str(tmp_path),
                        opt_sncid_list=2,
                        sncid_list_file=str(src / "FITRES.TEXT"))

    sim_dirs, _, kcor, model = fit_run._inputs(config)
    batch, _, _, _ = fit_run.load(config, model, kcor, sim_dirs,
                                  fit_run.read_shared_files(config))
    fitter = fit_run.prepare(config, model, kcor, batch)
    start = fit_run.start_values(fitter, fitter.config, batch, {})

    snana = read_fitres(str(src / "FITRES.TEXT")).df.set_index("CID")
    rows = snana.loc[np.asarray(batch.snids).astype(str)]
    assert batch.n_sne == len(snana)
    for name in ("PKMJD", "x0", "x1", "c"):
        np.testing.assert_array_equal(np.asarray(start[name]), rows[name])


def test_subsurvey_follows_the_entry_whatever_its_libid(tmp_path):
    """SUBSURVEY names sit per SIMLIB entry in file order, so an entry's
    name reaches its events by entry position, not by LIBID value, in the
    HEAD column and the README count (sntools_dataformat_fits.c:1326,
    sntools_sim_readme.c:640); and the fitter writes each event's SUBSURVEY
    id under IDSURVEY, the survey's own id where there is none
    (snana.F90:3839-3843, :23294-23297)."""
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1] / "fixtures"
    simlib = (root / "data/simlib/JAX_TEST_LSST_WINDOWS.SIMLIB").read_text()
    for old, new in (("LIBID: 0", "LIBID: 2008050"), ("LIBID: 1", "LIBID: 7")):
        assert simlib.count(old) == 2
        simlib = simlib.replace(old, new)
    (tmp_path / "W.SIMLIB").write_text(simlib)
    inp = tmp_path / "w.input"
    inp.write_text(re.sub(r"SIMLIB_FILE:.*", f"SIMLIB_FILE: {tmp_path / 'W.SIMLIB'}",
                          (root / "snana/simlib_windows/input.input").read_text()))

    config = dict(n_sne=200, random_seed=3)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    results, selection, spectra_cache = sim_run.run_simulation(sim)
    version = "JAX_TEST_SUBSURVEY"
    sim_run.save_snana_outputs(sim, results, version, selection, spectra_cache=spectra_cache, output_dir=str(tmp_path))
    vdir = tmp_path / version
    head = fits.getdata(resolve_fits_path(next(vdir.glob("*HEAD.FITS*"))), 1)
    lib = head['SIM_LIBID']
    assert set(lib) == {2008050, 7}
    assert set(head['SUBSURVEY'][lib == 2008050]) == {'SDSS'}
    assert set(head['SUBSURVEY'][lib == 7]) == {''}
    readme = (vdir / f"{version}.README").read_text()
    n_sdss = int((lib == 2008050).sum())
    assert re.search(rf"SDSS:\s+\d+\s+{n_sdss}\s+# NTOT NWRITE", readme)

    env = os.environ.copy()
    env["JAX_TEST_FIXTURE_ROOT"] = str(root / "data")
    env["JAX_TEST_FIT_SIM_DIR"] = str(tmp_path)
    out = tmp_path / "SUB.FITRES.TEXT"
    r = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[2] / "snlc_fit.py"),
         str(root / "snana/salt3_fit_dense/fit.nml"), "BATCH_SIZE", "4",
         "VERSION_PHOTOMETRY", version, "OUTPUT_NAME", str(out)],
        capture_output=True, text=True, env=env, timeout=1800)
    assert r.returncode == 0, r.stdout[-1200:] + r.stderr[-1200:]
    rows = [line.split() for line in out.read_text().splitlines()
            if line.startswith(("VARNAMES:", "SN:"))]
    cols = rows[0][1:]
    idsurvey = {int(row[1 + cols.index("CID")]): int(row[1 + cols.index("IDSURVEY")])
                for row in rows[1:]}
    by_cid = {int(snid): libid for snid, libid in zip(head['SNID'], lib)}
    assert idsurvey, "the fit wrote no events"
    for cid, sid in idsurvey.items():
        assert sid == (1 if by_cid[cid] == 2008050 else 12), (cid, sid)


def test_cutwin_trestrange_matches_snana(tmp_path):
    """CUTWIN_TRESTRANGE cuts on TrestMax - TrestMin before the fit
    (snana.F90:21343) and, against its lower edge, on the fitted range after
    it (snlc_fit.F90:12905-12909). snlc_fit.exe on this fixture: 30 drops
    CIDs 62 and 70 before the fit and 85 after; 45 drops 52, 62, 64, 70 and
    80 before and 85 after."""
    import subprocess
    import sys

    src = Path(__file__).resolve().parents[1] / "fixtures/snana/salt3_fit_dense"
    version = "JAX_TEST_SALT3_FIT_DENSE"
    vdir = tmp_path / version
    vdir.mkdir()
    for name in ("HEAD", "PHOT"):
        (vdir / f"{version}_{name}.FITS").write_bytes(
            gzip.decompress((src / f"{name}.FITS.gz").read_bytes()))
    (vdir / f"{version}.LIST").write_text(f"{version}_HEAD.FITS\n")

    env = os.environ.copy()
    env["JAX_TEST_FIXTURE_ROOT"] = str(
        Path(__file__).resolve().parents[1] / "fixtures/data")
    env["JAX_TEST_FIT_SIM_DIR"] = str(tmp_path)
    repo_root = Path(__file__).resolve().parents[2]

    def cids(threshold):
        out = tmp_path / f"TR{threshold}.FITRES.TEXT"
        r = subprocess.run(
            [sys.executable, str(repo_root / "snlc_fit.py"), str(src / "fit.nml"), "BATCH_SIZE", "4",
             "CUTWIN_TRESTRANGE", f"{threshold} 999", "OUTPUT_NAME", str(out)],
            capture_output=True, text=True, env=env, timeout=1800)
        assert r.returncode == 0, r.stdout[-1200:] + r.stderr[-1200:]
        return {line.split()[1] for line in out.read_text().splitlines()
                if line.startswith("SN:")}

    every = {str(i) for i in range(1, 101)}
    assert every - cids(30) == {"62", "70", "85"}
    assert every - cids(45) == {"52", "62", "64", "70", "80", "85"}


def test_fitwin_mb_err_cuts_when_unset(data_root):
    """SNANA always applies FITWIN_MB_ERR with an upper edge of 1.0, whether or
    not the input names it (snlc_fit.F90:6835, tested at :12958). No fixture
    event comes near that bound -- mBERR peaks at 0.06 -- so the cut is driven
    here rather than run end to end."""

    model = SALTModel.load(str(data_root / "models" / "SALT3.K21"), {})
    n = 3
    z = np.full(n, 0.3)
    mu = np.full(n, 41.0)
    params = {'PKMJD': np.zeros(n), 'x0': np.full(n, 1e-5),
              'x1': np.zeros(n), 'c': np.zeros(n)}
    # SN 2's amplitude error carries mBERR to 5.4, far above the 1.0 bound
    errors = {'PKMJD': np.full(n, 0.5), 'x0': np.array([1e-7, 5e-5, 1e-7]),
              'x1': np.full(n, 0.2), 'c': np.full(n, 0.02)}
    results = BatchFitResult(
        snids=np.array(['1', '2', '3']),
        params=params, param_errors=errors,
        param_covariances=np.zeros((n, 4, 4)), cov_output_params=[],
        chi_squared=np.full(n, 10.0), ndof=np.full(n, 20), prior_chi2=np.zeros(n),
        observation_mask=np.ones((n, 1), dtype=bool),
        hessian_cond=np.ones(n), hessian_positive=np.ones(n, dtype=bool), edm=np.zeros(n),
        z_prior=None, iter1_verdicts={})

    keep = fit_run._fitwin_mask({}, model, results, z, mu, None, FitConfig())
    assert list(keep) == [True, False, True]


def test_error_bounds_are_per_parameter(data_root):
    """CRAZYFITERR bounds the fitted error of five of SNANA's parameters
    (snlc_fit.F90:2436). Each is a property of that parameter, so each model
    declares its own; a model is then checked on exactly what it declares
    rather than on whatever a central table happened to name.

    Driven directly: no fixture event has an error anywhere near these edges,
    and the previous role-keyed table silently skipped every non-SALT model's
    shape and colour.
    """

    spec = SALTModel.load(str(data_root / "models" / "SALT3.K21"),
                     {}).get_fit_parameters()
    assert spec['x1']['error_bounds'] == (1.0e-8, 3.0)
    assert spec['c']['error_bounds'] == (1.0e-8, 3.0)
    assert spec['x0']['error_bounds'] == (1.0e-15, 3.0)
    assert spec['PKMJD']['error_bounds'] == (1.0e-8, 10.0)
    # zPHOT is the fitter's, so it is declared once for every model
    assert PHOTOZ_PARAMETER['zPHOT']['error_bounds'] == (1.0e-4, 2.0)

    n = 5
    def result(errs):
        return BatchFitResult(
            snids=np.arange(n).astype(str),
            params={k: np.zeros(n) for k in ('PKMJD', 'x1', 'c', 'x0')},
            param_errors=errs,
            param_covariances=np.zeros((n, 4, 4)), cov_output_params=[],
            chi_squared=np.full(n, 10.0), ndof=np.full(n, 20),
            prior_chi2=np.zeros(n), observation_mask=np.ones((n, 1), dtype=bool),
            hessian_cond=np.ones(n),
            hessian_positive=np.ones(n, dtype=bool),
            edm=np.zeros(n),
            z_prior=None, iter1_verdicts={})

    # SN1 x1 error above its bound, SN2 colour below its, SN3 PKMJD above the
    # 10.0 that only PKMJD gets, SN4 NaN. SN0 is within bounds throughout.
    errs = {'PKMJD': np.array([0.5, 0.5, 0.5, 11.0, 0.5]),
            'x0': np.full(n, 1e-7),
            'x1': np.array([0.2, 3.5, 0.2, 0.2, 0.2]),
            'c': np.array([0.02, 0.02, 1e-9, 0.02, np.nan])}
    keep = fit_run._fit_error_bounds_mask(spec, {}, result(errs))
    assert list(keep) == [True, False, False, False, False]

    # a held parameter is only checked for NaN (SNANA tests IF(FLOAT_x))
    keep = fit_run._fit_error_bounds_mask(spec, {'x1'}, result(errs))
    assert list(keep) == [True, True, False, False, False]

    # PKMJD's 10.0 is its own: 3.5 breaks x1's bound but is fine for PKMJD
    within = {'PKMJD': np.full(n, 3.5), 'x0': np.full(n, 1e-7),
              'x1': np.full(n, 0.2), 'c': np.full(n, 0.02)}
    assert fit_run._fit_error_bounds_mask(spec, {}, result(within)).all()


def test_peg_margins_are_per_parameter(data_root):
    """OPT_SNCID_LIST bit 0 forces the listed events through every post-fit cut
    but still rejects a fit pegged against a flat-prior edge
    (snlc_fit.F90:12801).

    The 0.05 margin is a width, not a fraction, so it belongs to the parameter
    it was chosen for. SALT declares it on x1 alone, which is the parameter
    SNANA tests; colour is left alone even when it pegs.
    """

    model = SALTModel.load(str(data_root / "models" / "SALT3.K21"), {})
    assert model.get_fit_parameters()['x1']['peg_margin'] == 0.05
    assert 'peg_margin' not in model.get_fit_parameters()['c']

    n = 5
    # x1 at the low edge, inside it, mid-range, inside the high edge, at it
    x1 = np.array([-3.0, -2.96, 0.0, 2.96, 3.0])
    results = BatchFitResult(
        snids=np.arange(n).astype(str),
        params={'PKMJD': np.zeros(n), 'x0': np.full(n, 1e-5),
                'x1': x1, 'c': np.full(n, -0.3)},
        param_errors={},
        param_covariances=np.zeros((n, 4, 4)), cov_output_params=[],
        chi_squared=np.full(n, 10.0), ndof=np.full(n, 20), prior_chi2=np.zeros(n),
        observation_mask=np.ones((n, 1), dtype=bool),
        hessian_cond=np.ones(n), hessian_positive=np.ones(n, dtype=bool), edm=np.zeros(n),
        z_prior=None, iter1_verdicts={})

    config = FitConfig(model=Model(priors={
        'x1': {'flat_rolloff': {'lo': -3.0, 'hi': 3.0}},
        'c': {'flat_rolloff': {'lo': -0.3, 'hi': 0.3}}}))
    keep = fit_run._sncid_pegged_mask(config, model, results, {})
    assert list(keep) == [False, False, True, False, False]
    # c sits exactly on its own prior edge and survives, because it declares
    # no margin -- 0.05 would cut a sixth off a range that runs over tenths
    assert keep[2]

    # the default x1 range is wider than x1's own bounds, so nothing pegs
    assert fit_run._sncid_pegged_mask(FitConfig(), model, results, {}).all()


def test_prior_range_needs_a_roll_off_width(data_root):
    """A flat prior needs both edges and a roll-off width. A model declares
    both for the parameters SNANA gives a prior (snlc_fit.F90:12006); asking
    for one on any other parameter is incomplete input rather than a cue to
    invent a width."""

    model = BayeSNModel.load(str(data_root / "models" / "BAYESN.W22"), {})
    spec = model.get_fit_parameters()
    assert not any(form == 'flat_rolloff'
                   for form, _ in spec['AV'].get('priors', ()))

    def av(**coefficients):
        return FitConfig(model=Model(
            priors={'AV': {'flat_rolloff': coefficients}}))

    with pytest.raises(SNANAError, match="no roll-off width"):
        priors.flat_window(av(lo=0.0, hi=2.0), 'AV', spec['AV'])

    # supplying the width too is a complete request and is honoured
    assert priors.flat_window(
        av(lo=0.0, hi=2.0, sigma=0.05), 'AV', spec['AV']) == (0.0, 2.0, 0.05)

    # and a parameter the model does give a prior needs no input at all
    assert priors.flat_window(
        FitConfig(), 'THETA', spec['THETA']) == (-9.0, 9.0, 0.1)


def test_simsed_redcor_degenerate_laws_abort():
    """SIMSED_REDCOR construction validates the whole law once: asymmetric or
    zero widths, unordered truncation ranges, |rho| >= 1, and a truncation
    rectangle carrying no probability mass are all fatal (parse bounds:
    snlc_sim.c:6203-6228; the rest are degenerate inputs to the truncated
    conditional law)."""

    model = SIMSEDModel.load(_SIMSED_MODEL, {})
    model.set_parameter_keywords(['stretch', 'color'], [])

    def cfg(rho=-0.5, s1=(0.096, 0.096), r1=(0.65, 1.25),
            s2=(0.175, 0.175), r2=(0.0, 1.0), peak1=0.975, is_cov=False):
        simsed_cfg = {'redcor': {('stretch', 'color'): (rho, is_cov)}}
        dists = {'stretch': GenGaussAsym(peak=peak1, sigma=s1, range=r1),
                 'color': GenGaussAsym(peak=0.557, sigma=s2, range=r2)}
        return simsed_cfg, {'populations': {
            f'simsed_{n}': Population(distributions=(d,), shifts=((),))
            for n, d in dists.items()}}

    # valid law parses and records the reduced correlation
    rc = model._parse_redcor(*cfg())
    assert rc['rho'] == -0.5 and rc['p1'] == 'stretch'

    for kwargs, msg in (
            (dict(s1=(0.1, 0.2)), "symmetric"),
            (dict(s2=(0.0, 0.0)), "positive width"),
            (dict(r1=(1.25, 0.65)), "ordered"),
            (dict(rho=1.2), "reduced correlation"),
            (dict(peak1=500.0, s1=(1e-3, 1e-3), r1=(0.0, 0.5)),
             "no probability mass")):
        with pytest.raises(SNANAError, match=msg):
            model._parse_redcor(*cfg(**kwargs))


def test_simsed_corner_lookup_validates_grid():
    """A SIMSED grid with duplicate coordinates or a missing Cartesian corner
    is fatal at construction, matching SNANA's corner checks
    (genmag_SIMSED.c:2024-2053); the complete fixture grid passes."""

    model = SIMSEDModel.load(_SIMSED_MODEL, {})
    model.set_parameter_keywords(['stretch', 'color'], [])
    model._build_corner_lookup()
    assert model._corner_lookup is not None
    assert (model._corner_lookup >= 0).all()

    dup = SIMSEDModel.load(_SIMSED_MODEL, {})
    dup.set_parameter_keywords(['stretch', 'color'], [])
    dup.sed_par_values[1] = dup.sed_par_values[0]
    with pytest.raises(SNANAError, match="same coordinates"):
        dup._build_corner_lookup()

    short = SIMSEDModel.load(_SIMSED_MODEL, {})
    short.set_parameter_keywords(['stretch', 'color'], [])
    short.sed_par_values = short.sed_par_values[:-1]
    with pytest.raises(SNANAError, match="missing"):
        short._build_corner_lookup()


_A_NON1A = "tests/fixtures/snana/non1ased_v19_replay/input.input"


def test_non1a_keys_columns_match_by_name(tmp_path):
    """Columns beyond INDEX and WGT follow the names NON1A_KEYS declares
    (snlc_sim.c:3508-3534), not fixed positions."""
    src = Path(_A_NON1A).read_text().replace(
        "NON1A_KEYS:  5 INDEX WGT MAGOFF MAGSMEAR SNTYPE",
        "NON1A_KEYS:  5 INDEX WGT MAGSMEAR MAGOFF SNTYPE").replace(
        "NON1A: 834 1.0 0.00 0.00 21", "NON1A: 834 1.0 0.30 0.10 21")
    cfg = parse_snana_input_file(str(_write(tmp_path, src)))
    non1a = cfg.model._structured_config['non1a_config']
    assert float(non1a['magsmear_per_template'][0]) == pytest.approx(0.30)
    assert float(non1a['magoff_per_template'][0]) == pytest.approx(0.10)


def test_non1a_keys_defaults_to_index_and_wgt(tmp_path):
    """Without a NON1A_KEYS line NKEY defaults to 2 (snlc_sim.c:1367), so a
    bare 'INDEX WGT' row simulates with zero MAGOFF/MAGSMEAR/SNTYPE."""
    src = "\n".join(
        line for line in Path(_A_NON1A).read_text().splitlines()
        if not line.startswith('NON1A_KEYS')).replace(
        "NON1A: 834 1.0 0.00 0.00 21", "NON1A: 834 1.0")
    cfg = parse_snana_input_file(str(_write(tmp_path, src)))
    non1a = cfg.model._structured_config['non1a_config']
    assert list(non1a['index_per_template']) == [834]
    assert float(non1a['magoff_per_template'][0]) == 0.0
    assert int(non1a['sntype_per_template'][0]) == 0


def test_simsed_sequential_writes_every_parameter_as_baggage(tmp_path):
    """SEQUENTIAL declares no parameter, so every SED.INFO parameter is
    baggage: written as SIMSED_param(<name>) in the TEXT header, the FITS
    header and the DUMP, and the template index is -9 without a parameter
    named like one (snlc_sim.c:5912, :23401-23420; genmag_SIMSED.c:1626-1633)."""
    inp = _simsed_gridonly_input(tmp_path, "", "seqtext")
    src = Path(inp).read_text().replace("SIMSED_GRIDONLY: stretch\n", "").replace(
        "SIMSED_GRIDONLY: color\n", "SIMSED_GRIDONLY: SEQUENTIAL\n")
    inp = _write(tmp_path, src)
    config = dict(
        n_sne=4, random_seed=5)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    sed_model = resources.model
    grid_idx = np.array([3], dtype=np.int32)
    params = {'grid_idx': grid_idx, **sed_model._node_values(grid_idx)}
    assert sed_model.dump_columns == ("SIMSED_param(stretch)", "SIMSED_param(color)")
    lines = sed_model.text_sim_header_lines({}, params, 0)
    assert "SIMSED_NPAR: 2\n" in lines
    assert f"SIMSED_param(stretch): {float(sed_model.sed_par_values[3, 0]):.6f}\n" in lines
    header = sed_model.get_snana_header_params(params)
    assert header["SIMSED_param(color)"] == pytest.approx(sed_model.sed_par_values[3, 1])
    assert header["SIM_TEMPLATE_INDEX"].tolist() == [-9]


def test_simsed_gridonly_text_header_reads_the_grid(tmp_path):
    """GRIDONLY sampling records the surface index and the parameter values
    at that node, which the TEXT header writes."""
    inp = _simsed_gridonly_input(tmp_path, "", "gridtext")
    config = dict(
        n_sne=4, random_seed=5)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    sed_model = resources.model
    grid_idx = np.array([3], dtype=np.int32)
    lines = sed_model.text_sim_header_lines(
        {}, {'grid_idx': grid_idx, **sed_model._node_values(grid_idx)}, 0)
    # the line carries the declaring keyword, as the FITS column does
    # (SNDATA.SIMSED_KEYWORD, sntools_dataformat_text.c:457; snlc_sim.c:5930)
    assert f"SIMSED_GRIDONLY(stretch): {float(sed_model.sed_par_values[3, 0]):.6f}\n" in lines
    assert f"SIMSED_GRIDONLY(color): {float(sed_model.sed_par_values[3, 1]):.6f}\n" in lines


# ---------------------------------------------------------------------------
# HOSTLIB_ZPHOTEFF (GEN_SNHOST_ZPHOT, sntools_host.c:7178-7194)
# ---------------------------------------------------------------------------

def test_zphoteff_masks_the_cut_and_the_redshift_source_only(tmp_path):
    """SNANA carries two photo-z values and HOSTLIB_ZPHOTEFF masks one.

    SNHOSTGAL.ZPHOT goes to -9 on a failed draw, and CUTWIN_HOST_ZPHOT
    (snlc_sim.c:23833), setz_unconfirmed (:24574) and the GENSIGMA_REDSHIFT<0
    redshift source (sntools_host.c:7190) all read it. The data file reports
    SNHOSTGAL_DDLR_SORT[m].ZPHOT (snlc_sim.c:25723), which the mask never
    touches, so the two must differ for the events that failed.
    """
    eff_file = "$JAX_TEST_FIXTURE_ROOT/hostlib/JAX_TEST_ZPHOTEFF.DAT"
    p = _run(_SALT3,
             f"{_HOSTLIB_LINE}\nHOSTLIB_ZPHOTEFF_FILE: {eff_file}",
             tmp_path, n_sne=200)['parameters']

    reported = np.asarray(p['host_ZPHOT'], float)
    event = np.asarray(p['host_ZPHOT_EVENT'], float)
    failed = event < 0.0
    # a flat 50% efficiency over 200 events
    assert 60 < int(failed.sum()) < 140, int(failed.sum())
    # the reported photo-z is untouched where the draw failed
    assert np.all(reported[failed] > 0.0)
    np.testing.assert_array_equal(event[~failed], reported[~failed])
    np.testing.assert_array_equal(np.asarray(p['host_ZPHOT_EVENT_ERR'])[failed],
                                  np.full(int(failed.sum()), -9.0))


def test_cutwin_host_zphot_cuts_on_the_masked_photoz(tmp_path):
    """A ZPHOTEFF failure fails CUTWIN_HOST_ZPHOT in SNANA, because the cut
    reads the masked scalar (gen_cutwin, snlc_sim.c:23833)."""
    eff_file = "$JAX_TEST_FIXTURE_ROOT/hostlib/JAX_TEST_ZPHOTEFF.DAT"
    extra = (f"{_HOSTLIB_LINE}\nHOSTLIB_ZPHOTEFF_FILE: {eff_file}\n"
             "CUTWIN_HOST_ZPHOT: 0.0 1.0\nAPPLY_CUTWIN_OPT: 1")
    inp = tmp_path / "in.input"
    inp.write_text(_merge(Path(_SALT3).read_text(), extra))
    config = dict(n_sne=200, random_seed=5)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    res, sel, _ = sim_run.run_simulation(sim)

    event = np.asarray(res['parameters']['host_ZPHOT_EVENT'], float)
    passing = np.asarray(sel['passing_mask'], bool)
    failed = event < 0.0
    assert failed.any() and (~failed).any()
    assert not passing[failed].any(), (
        "a masked photo-z is outside CUTWIN_HOST_ZPHOT 0..1 and must fail")


def test_zphoteff_file_short_of_the_hostlib_redshifts_aborts(tmp_path):
    """SNANA aborts on a host outside the file's ZTRUE range
    (sntools.c:8234); the build refuses the file up front."""
    eff_file = tmp_path / "NARROW.DAT"
    eff_file.write_text("0.20 0.5\n0.25 0.5\n")
    inp = tmp_path / "in.input"
    inp.write_text(_merge(Path(_SALT3).read_text(),
                          f"{_HOSTLIB_LINE}\nHOSTLIB_ZPHOTEFF_FILE: {eff_file}"))
    with pytest.raises(SNANAError, match="does not cover"):
        sim_run.prepare(str(inp), **dict(n_sne=20, random_seed=5))
