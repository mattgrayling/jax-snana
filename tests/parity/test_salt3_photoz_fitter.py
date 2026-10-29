"""SALT3 photo-z fitter tests.

Two tests:

* ``test_salt3_photoz_fitter_recovers_z`` — JAX-only: recovered z within 3σ
  of HOSTGAL_PHOTOZ for ≥80% of SNe; pins prior wiring + free-z minimisation.
* ``test_salt3_photoz_fitter_vs_snana`` — JAX vs SNANA's snlc_fit.exe (in
  ``DOFIT_PHOTOZ=T`` mode): per-SN ``|z_JAX - z_SNANA|`` within
  ``3 √(σ_JAX² + σ_SNANA²)``. Catches divergence between the two photo-z
  optimisers.
"""

import gzip
import os
import subprocess
import sys

import numpy as np
import pytest
from _fit_helpers import (build_fit_model, build_fitter, compare_lcplots, fit_config, for_fit,
                          load_batch, override, parse_fitres, peak_start, regenerate_lcplot,
                          run_fit)

from jax_snlc_core.snana_files import read_fitres
from jax_snlc_fit.batch import (ERRFLAG_FIT_COLOR_ITER1, ERRFLAG_FIT_CUTS,
                                ERRFLAG_FIT_FITPROB_ITER1, ERRFLAG_FIT_SHAPE_ITER1,
                                ERRFLAG_FITPREP_ADJUST, ERRFLAG_NO_FIT)
from jax_snlc_fit.cuts import build_selection, mark_errflags, select, select_events, sncid_list
from jax_snlc_fit.fitter import BatchFitResult
from jax_snlc_fit.photoz import z_prior_sigma
from jax_snlc_fit.loader import model_sim_columns
from _fit_helpers import load_and_batch_snana_fits
from jax_snlc_fit.output import _snana_flags


_LSST_BANDS = ["LSST-u", "LSST-g", "LSST-r", "LSST-i", "LSST-z", "LSST-Y"]


def _photoz_fit(fx, data=None):
    """The photo-z fit of a bundle's first 40 events, at its own settings."""
    config = fit_config(fx.root / "fit.nml", batch_size=40)
    data = fx if data is None else data
    batch = load_batch(config, data.head_fits, data.phot_fits, max_fit=40,
                       keep_obs_extras=True)
    assert batch.n_sne >= 20, f"too few fittable SNe in fixture ({batch.n_sne})"
    fitter = build_fitter(config, build_fit_model(config), batch)
    batch = for_fit(fitter, batch)
    results, _ = fitter.fit_sample(batch, peak_start(fitter, batch))
    return results, batch, fitter


@pytest.fixture(scope="module")
def salt3_photoz_fit_results(snana_fixture):
    """Run the SALT3 photo-z fit on salt3_fit_photoz once, share across both
    parity tests. Each fit is ~16-27s of JIT + optimisation."""
    fx = snana_fixture("salt3_fit_photoz")
    results, batch, fitter = _photoz_fit(fx)
    return fx, results, batch, fitter


def test_photoz_bound_bounds_the_fit(snana_fixture):
    """PHOTOZ_BOUND is the MINUIT bound on zPHOT (snlc_fit.F90:8459-8467), so
    it holds the start point and a pinned prior as well as every step. On a
    bound tighter than the sample's redshifts every fit lies inside it and
    the events SNANA leaves at the upper bound sit there in jax too."""
    fx = snana_fixture("salt3_fit_photoz_bound")
    results, _, fitter = _photoz_fit(fx, snana_fixture("salt3_fit_photoz"))
    lo, hi = fitter.config.photoz.photoz_bound
    z = np.asarray(results.params["zPHOT"])
    assert z.min() >= lo and z.max() <= hi, (z.min(), z.max())

    snana_rows = parse_fitres(fx.root / "FITRES.TEXT")
    cids = [int(s) for s in np.asarray(results.snids)]
    both = [c for c in cids if c in snana_rows
            and snana_rows[c].get("ERRFLAG_FIT", 0) == 0]
    assert len(both) >= 15, f"too few cross-matched fits ({len(both)})"
    # SNANA writes the fitted redshift as zHEL (SKIPTABLE_FITPAR drops zPHOT)
    at_bound_snana = {c for c in both if snana_rows[c]["zHEL"] >= hi - 1e-4}
    at_bound_jax = {c for c, zj in zip(cids, z) if c in both and zj >= hi - 1e-4}
    assert len(at_bound_snana) >= 10, "the bound must actually bite"
    differ = at_bound_snana ^ at_bound_jax
    assert len(differ) <= 2, f"events at the bound differ: {sorted(differ)}"


@pytest.mark.parametrize("name", ["salt3_fit_photoz_muprior",
                                  "salt3_fit_photoz_dmuprior"])
def test_distance_prior_matches_snana(snana_fixture, name):
    """PRIOR_MUERRSCALE and PRIOR_DMU_RANGE tie the fitted distance to the
    reference cosmology's at the fitted redshift (CHI2PRIOR(IPAR_DLMAG),
    snlc_fit.F90:4910). With no other redshift information either prior
    moves many fitted redshifts by more than the tolerance here, so the match
    shows the prior applied as SNANA applies it. Fits on the light curve alone
    can settle in different local minima, so two events may differ."""
    fx = snana_fixture(name)
    results, _, _ = _photoz_fit(fx, snana_fixture("salt3_fit_photoz"))
    snana_rows = parse_fitres(fx.root / "FITRES.TEXT")
    both = [(cid, z) for cid, z in zip(np.asarray(results.snids).astype(int),
                                       np.asarray(results.params["zPHOT"]))
            if cid in snana_rows and snana_rows[cid].get("ERRFLAG_FIT", 0) == 0]
    assert len(both) >= 35, f"too few cross-matched fits ({len(both)})"
    # SNANA writes the fitted redshift as zHEL (SKIPTABLE_FITPAR drops zPHOT)
    differ = {cid: (round(float(z), 4), snana_rows[cid]["zHEL"]) for cid, z in both
              if abs(z - snana_rows[cid]["zHEL"]) > 0.002}
    assert len(differ) <= 2, f"fitted z differs from SNANA's (jax, SNANA): {differ}"


def test_photoz_batch_independence(snana_fixture):
    """A band-set repeat for one event must not affect its batch companion."""
    from jax_snlc_fit import photoz

    fx = snana_fixture("salt3_fit_photoz")
    config = fit_config(fx.root / "fit.nml", n_fit_iterations=1, batch_size=2)
    batch = load_batch(config, fx.head_fits, fx.phot_fits, max_fit=40)
    batch = batch.take(np.flatnonzero(np.isin(batch.snids, ["1", "26"])))
    assert batch.n_sne == 2
    model = build_fit_model(config)
    fitter = build_fitter(config, model, batch)
    together = run_fit(fitter, batch)
    loaded = for_fit(fitter, batch)
    _, repeat = photoz.round1_band_set(
        config, fitter.restlambda_usefit,
        loaded.photometry['filter_mean_wavelengths'], batch.filter_indices,
        batch.observation_mask, fitter.stack(together.params, batch.reference_mjds),
        loaded.z_priors)
    assert np.any(repeat) and not np.all(repeat)

    separately = run_fit(
        build_fitter(override(config, batch_size=1), model, batch), batch)
    for name in together.params:
        np.testing.assert_allclose(together.params[name], separately.params[name],
                                   rtol=1e-6, atol=1e-8)
    for name in ('param_covariances', 'prior_chi2', 'chi_squared', 'edm'):
        np.testing.assert_allclose(getattr(together, name), getattr(separately, name),
                                   rtol=1e-6, atol=1e-10)


def test_fixed_redshift_dof(snana_fixture):
    fx = snana_fixture("salt3_fit_photoz")
    config = fit_config(fx.root / "fit.nml", batch_size=4)
    config.model.inistp['zPHOT'] = 0.
    batch = load_batch(config, fx.head_fits, fx.phot_fits, max_fit=4)
    fitter = build_fitter(config, build_fit_model(config), batch)
    results = run_fit(fitter, batch)
    np.testing.assert_array_equal(results.param_errors['zPHOT'], 0.)
    np.testing.assert_array_equal(
        results.ndof, np.asarray(results.observation_mask).sum(axis=1) - 4)


def test_photoz_initialisation_preserves_fixed_amplitude_and_redshift(snana_fixture):
    """Held inputs survive both the coarse search and the final optimiser."""
    import jax.numpy as jnp

    fx = snana_fixture('salt3_fit_photoz')
    config = fit_config(fx.root / 'fit.nml', batch_size=1, n_fit_iterations=1)
    config.model.fixed = {'x0': 0.00018905, 'zPHOT': 0.21925}
    batch = load_batch(config, fx.head_fits, fx.phot_fits, max_fit=1)
    fitter = build_fitter(config, build_fit_model(config), batch)
    batch = for_fit(fitter, batch)
    start = fitter.start_values(fitter.config, batch, fitter.usesim_columns)
    results, _ = fitter.fit_sample(batch, start)
    for name, value in config.model.fixed.items():
        np.testing.assert_allclose(results.params[name], value, rtol=1e-12)
        np.testing.assert_array_equal(results.param_errors[name], jnp.zeros(1))


def test_zero_redshift_error(snana_fixture):
    """A zero measured uncertainty retains its finite protected prior."""
    fx = snana_fixture('salt3_fit_photoz')
    config = fit_config(fx.root / 'fit.nml', batch_size=1)
    batch = load_batch(config, fx.head_fits, fx.phot_fits, max_fit=1)
    batch.redshift_errs = np.zeros(1)
    fitter = build_fitter(config, build_fit_model(config), batch)
    batch = for_fit(fitter, batch)
    np.testing.assert_array_equal(batch.z_priors.z_prior_err, 0.)
    results, _ = fitter.fit_sample(batch, peak_start(fitter, batch))
    assert np.isfinite(results.chi_squared).all()
    for name in results.params:
        assert np.isfinite(results.params[name]).all()
        assert np.isfinite(results.param_errors[name]).all()
    np.testing.assert_allclose(results.params['zPHOT'], batch.redshifts,
                               atol=0.005)


def test_missing_redshift_is_fitted_without_a_prior(snana_fixture):
    """An event without a REDSHIFT_FINAL is fitted from its light curve
    alone, with zPRIORERR = 9999, rather than pulled to the missing value."""
    fx = snana_fixture('salt3_fit_photoz')
    config = fit_config(fx.root / 'fit.nml', batch_size=2)
    batch = load_batch(config, fx.head_fits, fx.phot_fits, max_fit=2)
    true_z = np.array(batch.redshifts)
    batch.redshifts = np.array([true_z[0], -999.0])
    batch.redshift_errs = np.array([batch.redshift_errs[0], -999.0])
    fitter = build_fitter(config, build_fit_model(config), batch)
    batch = for_fit(fitter, batch)
    results, _ = fitter.fit_sample(batch, peak_start(fitter, batch))
    np.testing.assert_array_equal(np.asarray(results.z_prior[1]) == 9999.0,
                                  [False, True])
    np.testing.assert_allclose(results.params['zPHOT'], true_z, atol=0.01)


def test_photoz_external_start(snana_fixture):
    """External fit values seed every parameter, bypassing the coarse grid."""
    from jax_snlc_fit.loader import start_values_from_sncid_list

    fx = snana_fixture('salt3_fit_photoz')
    config = fit_config(fx.root / 'fit.nml', batch_size=1,
                        n_fit_iterations=1, lfixpar_all=True,
                        maxiter_round1=0, maxiter_round2=0)
    config.selection.opt_sncid_list = 2
    batch = load_batch(config, fx.head_fits, fx.phot_fits, max_fit=1)
    fitter = build_fitter(config, build_fit_model(config), batch)
    batch = for_fit(fitter, batch)
    batch.sncid_inival = start_values_from_sncid_list(
        read_fitres(fx.root / 'FITRES.TEXT'), batch.snids, fitter.param_names)
    start = fitter.start_values(fitter.config, batch, fitter.usesim_columns)
    for name, values in batch.sncid_inival.items():
        np.testing.assert_array_equal(start[name], values)
    assert not config.fit.skip_grid_search
    assert np.any(start['zPHOT'] != batch.z_priors.z_init)
    results, _ = fitter.fit_sample(batch, start)
    for name, values in start.items():
        np.testing.assert_allclose(results.params[name], values, rtol=1e-12)


def test_the_coarse_grid_spans_the_window_the_fit_uses(snana_fixture):
    """The grid's length is sized from the priors the fit reads. With the
    host prior off (PRIOR_ZERRSCALE past the DOFIT_PHOTOZ_HOST gate,
    snlc_fit.F90:13961) the window is the whole PHOTOZ_BOUND, and a host
    error wider than the rest of the sample's widens its own window; sized
    from a constant width, the grid stopped short of both and the fit
    started from wherever it ended."""
    from jax_snlc_fit import photoz

    fx = snana_fixture("salt3_fit_photoz")
    config = fit_config(fx.root / "fit.nml")
    batch = load_batch(config, fx.head_fits, fx.phot_fits, max_fit=40,
                       keep_obs_extras=True)
    config.photoz.opt_photoz = 1
    lo, hi = config.photoz.photoz_bound

    def n_over(zot_width):
        return int(photoz.PHOTOZ_GRID_DENSITY * zot_width) + 1

    def scan_size():
        return photoz.scan_size(config, photoz.redshift_priors(config, batch))

    config.photoz.prior_zerrscale = 100.0
    assert scan_size() == n_over(
        hi / (1 + hi) - lo / (1 + lo))

    # every event with a host (the fixture's one hostless event spans the
    # whole bound), the first with an error five times the others'
    config.photoz.prior_zerrscale = 1.0
    z_host = np.where(np.asarray(batch.photoz) > 0, np.asarray(batch.photoz), 0.2)
    err = np.full_like(z_host, 0.03)
    err[0] = 0.15
    batch.photoz, batch.photoz_err = z_host, err
    z_lo, z_hi = max(z_host[0] - 0.3, lo), min(z_host[0] + 0.3, hi)
    assert scan_size() == n_over(
        z_hi / (1 + z_hi) - z_lo / (1 + z_lo))


def test_the_coarse_grid_starts_at_its_best_peakmjd(snana_fixture):
    """NBINT_COURSE_PHOTOZ adds a PEAKMJD axis of offsets from each event's
    estimate, and the fit starts from the best grid point's PEAKMJD as well as
    its other parameters; SNANA keeps the last bin's instead. Offered a bin a
    week before the estimate and a last bin three weeks after it, the grid
    starts every event from the nearer one."""
    fx = snana_fixture("salt3_fit_photoz")
    config = fit_config(fx.root / "fit.nml", batch_size=8, nbint_course=2,
                        ranget_course=(-7.5, 22.5))
    batch = load_batch(config, fx.head_fits, fx.phot_fits, max_fit=8)
    fitter = build_fitter(config, build_fit_model(config), batch)
    batch = for_fit(fitter, batch)
    start = {**peak_start(fitter, batch), 'zPHOT': batch.z_priors.z_init}
    scales = fitter._scales(batch, batch.z_priors)
    physical = fitter.stack(start, batch.reference_mjds)
    grid = fitter._grid_start(batch, physical / scales, scales, batch.z_priors) * scales
    np.testing.assert_allclose(np.asarray(grid[:, 0] - physical[:, 0]), -7.5)


def test_salt3_photoz_fitter_recovers_z(salt3_photoz_fit_results):
    """Photo-z fit recovers z within 3σ of HOSTGAL_PHOTOZ for ≥80% of SNe."""
    fx, results, _, _ = salt3_photoz_fit_results
    snids = np.asarray(results.snids)
    z_jax = np.asarray(results.params["zPHOT"])

    from astropy.io import fits
    with fits.open(fx.head_fits) as hdul:
        head = hdul[1].data
        priors = {
            int(row["SNID"]): (
                float(row["HOSTGAL_PHOTOZ"]),
                float(row["HOSTGAL_PHOTOZ_ERR"]),
                float(row["SIM_REDSHIFT_CMB"]),
            )
            for row in head
        }

    pulls = []
    truth_diff = []
    for i, snid in enumerate(snids):
        cid = int(snid)
        if cid not in priors:
            continue
        zphot, zphot_err, ztrue = priors[cid]
        if zphot < 0:
            continue
        sigma = max(zphot_err, 1e-3)
        pulls.append((z_jax[i] - zphot) / sigma)
        truth_diff.append(z_jax[i] - ztrue)

    pulls = np.asarray(pulls)
    truth_diff = np.asarray(truth_diff)
    assert len(pulls) >= 15, f"too few photo-z fits to compare ({len(pulls)})"

    n_within_3sigma = int((np.abs(pulls) < 3.0).sum())
    frac = n_within_3sigma / len(pulls)
    assert frac >= 0.8, (
        f"{n_within_3sigma}/{len(pulls)} ({frac:.0%}) within 3σ of HOSTGAL_PHOTOZ"
    )

    # Recovered z should also be close to truth (HOSTGAL_PHOTOZ is unbiased
    # by construction, scatter 0.03). Median |z - ZTRUE| < 0.05 leaves
    # plenty of headroom over the 0.03 prior scatter.
    abs_truth = np.abs(truth_diff)
    median_truth_err = float(np.median(abs_truth))
    p95_truth_err = float(np.percentile(abs_truth, 95))
    # Dense SIMLIB + multi-band SALT3 fit + 0.03 photo-z prior → recovered
    # z is within 1e-3 of ZTRUE for the median, ~2e-3 at p95. Thresholds
    # below allow ~5x headroom while flagging real bias drift.
    assert median_truth_err < 0.005, (
        f"median |z_fit - ZTRUE| = {median_truth_err:.4f} > 0.005"
    )
    assert p95_truth_err < 0.02, (
        f"p95 |z_fit - ZTRUE| = {p95_truth_err:.4f} > 0.02"
    )


def test_salt3_photoz_fitter_vs_snana(salt3_photoz_fit_results):
    """JAX recovers the same photo-z as SNANA's snlc_fit.exe within fit error."""
    fx, results, _, _ = salt3_photoz_fit_results
    fitres_path = fx.root / "FITRES.TEXT"
    assert fitres_path.is_file(), (
        "salt3_fit_photoz fixture missing FITRES.TEXT — regenerate with "
        "`python tests/fixtures/regenerate.py --only salt3_fit_photoz`"
    )
    snana_rows = parse_fitres(fitres_path)

    snids = np.asarray(results.snids)
    z_jax = np.asarray(results.params["zPHOT"])
    z_jax_err = np.asarray(results.param_errors["zPHOT"])

    pulls = []
    for i, sid in enumerate(snids):
        cid = int(sid)
        if cid not in snana_rows:
            continue
        s = snana_rows[cid]
        if s.get("ERRFLAG_FIT", 0) != 0:
            continue
        sv = s.get("zHEL")
        se = s.get("zHELERR")
        if sv is None or se is None:
            continue
        je = float(z_jax_err[i])
        if not np.isfinite(je) or je <= 0 or not np.isfinite(z_jax[i]):
            continue
        sigma = float(np.sqrt(je ** 2 + se ** 2))
        pulls.append((float(z_jax[i]) - sv) / sigma)

    pulls_arr = np.asarray(pulls)
    assert len(pulls_arr) >= 15, f"too few cross-matched fits ({len(pulls_arr)})"
    n_outliers = int((np.abs(pulls_arr) > 3.0).sum())
    assert n_outliers <= 2, (
        f"{n_outliers}/{len(pulls_arr)} SNe with |pull(zPHOT)| > 3 (max 2)"
    )


def test_photoz_holds_a_pinned_parameter(snana_fixture):
    """INISTP_<p> = 0 pins the parameter in a photo-z fit as in a spec-z one:
    FLOATPAR is a property of the parameter, not of the fit mode
    (snana.F90:27850), and SET_COURSEBIN_PHOTOZ collapses the pinned axis of
    the coarse grid to one bin (snlc_fit.F90:9402)."""
    fx = snana_fixture("salt3_fit_photoz")
    config = fit_config(fx.root / "fit.nml")
    config = override(config, inistp={**config.model.inistp, "x1": 0.0})
    batch = load_batch(config, fx.root / "HEAD.FITS", fx.root / "PHOT.FITS",
                       max_fit=8)
    fitter = build_fitter(config, build_fit_model(config), batch)
    assert "x1" in fitter.held_params
    results = run_fit(fitter, batch)

    x1_init = fitter.model.get_fit_parameters()["x1"]["init"]
    assert np.allclose(np.asarray(results.params["x1"]), x1_init)
    # a held parameter consumes no degree of freedom either: PKMJD, c, x0 and
    # the fitted redshift are left (XNFITDOF(0), snlc_fit.F90:13132-13140)
    n_obs = np.asarray(results.observation_mask).sum(axis=1)
    assert np.array_equal(np.asarray(results.ndof), n_obs - 4)


def test_photoz_evaluates_the_chi2red_ini_windows(snana_fixture):
    """FITINI_ADJUST is common to both fit modes: only
    DOFIT_PHOTOZ_MCMC_ADJUST returns early (snlc_fit.F90:9544), so
    FITWIN_CHI2RED_INI and _INI2 window the chi2 either side of it in a
    photo-z fit as in a spec-z one (:9596, :9803)."""
    fx = snana_fixture("salt3_fit_photoz")
    config = fit_config(fx.root / "fit.nml")
    batch = load_batch(config, fx.root / "HEAD.FITS", fx.root / "PHOT.FITS",
                       max_fit=6)
    model = build_fit_model(config)
    assert run_fit(build_fitter(config, model, batch),
                   batch).iter1_verdicts == {}

    narrow = override(config, fitwin_chi2red_ini=(0.0, 1e-12),
                      fitwin_chi2red_ini2=(0.0, 1e-12))
    verdicts = run_fit(build_fitter(narrow, model, batch),
                       batch).iter1_verdicts
    assert set(verdicts) == {"CHI2RED_INI", "CHI2RED_INI2"}
    assert not np.any(np.asarray(verdicts["CHI2RED_INI"]))
    assert not np.any(np.asarray(verdicts["CHI2RED_INI2"]))


def test_photoz_fitres_columns_vs_snana(salt3_photoz_fit_results):
    """The FITRES parameter and covariance columns of a photo-z fit, in
    SNANA's order. TABLE_VARLIST_COV pairs every floated parameter with the
    fitted redshift last (snlc_fit.F90:19536), so COV_x1_zPHOT, COV_c_zPHOT
    and COV_x0_zPHOT belong in the text table beside the spec-z pairs; the
    fitted redshift itself is zHEL, and zPHOT/zPHOTERR are not written
    (SKIPTABLE_FITPAR, :19042). zPRIOR and zPRIORERR are the prior the fit
    applied, the start redshift and its step scaled by PRIOR_ZERRSCALE
    (snlc_fit.F90:8561); under the best-redshift Gaussian, the data redshift
    and its error."""
    from jax_snlc_fit.output import build_fitres_table

    fx, results, batch, fitter = salt3_photoz_fit_results
    redshifts = np.asarray(results.params["zPHOT"])
    table, _, _ = build_fitres_table(
        results, redshifts, fitter.config.fit.distance_modulus(redshifts),
        model=fitter.model)

    with open(fx.root / "FITRES.TEXT") as f:
        snana_varnames = next(line.split()[1:] for line in f
                              if line.startswith("VARNAMES:"))

    block = ["PKMJD", "PKMJDERR", "x1", "x1ERR", "c", "cERR", "mB", "mBERR",
             "x0", "x0ERR",
             "COV_x1_c", "COV_x1_x0", "COV_x1_zPHOT", "COV_c_x0",
             "COV_c_zPHOT", "COV_x0_zPHOT", "zPRIOR", "zPRIORERR",
             "NDOF", "FITCHI2", "FITPROB"]
    start = snana_varnames.index("PKMJD")
    assert snana_varnames[start:start + len(block)] == block
    start = table.varnames.index("PKMJD")
    assert table.varnames[start:start + len(block)] == block

    snana = parse_fitres(fx.root / "FITRES.TEXT")
    for cid, zp, zpe in zip(np.asarray(results.snids).astype(int),
                            *map(np.asarray, results.z_prior)):
        if cid in snana:
            assert abs(zp - snana[cid]["zPRIOR"]) < 1e-4, cid
            assert abs(zpe - snana[cid]["zPRIORERR"]) < 1e-4, cid


def test_photoz_fitchi2_is_snanas(salt3_photoz_fit_results):
    """FITCHI2 under photo-z with the full flux covariance. Photo-z turns
    OPT_CHI2_SIGMA on (snlc_fit.F90:7130), so the chi2 carries
    2*ln(sigma/sigma_last) per epoch, where sigma_last is the previous
    iteration's full diagonal error (JEP_FLUX_ERRTOT, :5100), not the
    covariance's diagonal with the colour dispersion taken out, which would
    add a positive amount to every event at the same minimum."""
    fx, results, _, _ = salt3_photoz_fit_results
    snana = parse_fitres(fx.root / "FITRES.TEXT")
    chi2 = np.asarray(results.chi_squared)
    diffs = [chi2[i] - snana[int(snid)]["FITCHI2"]
             for i, snid in enumerate(np.asarray(results.snids))
             if snana.get(int(snid), {}).get("ERRFLAG_FIT", 1) == 0]
    assert len(diffs) >= 30
    assert abs(np.median(diffs)) < 0.05, (
        f"FITCHI2 is {np.median(diffs):+.3f} from SNANA's in the median")


def test_photoz_error_stays_within_the_prior_width(salt3_photoz_fit_results):
    """A Gaussian prior of width s on z bounds its marginal error at s, since
    the data's curvature can only add to the prior's. An error above the
    bound means the Hessian's residual-times-curvature term went negative,
    which a model with kinks in z produces as noise of either sign; the
    allowance is the genuine curvature of a smooth model.
    """
    _, results, batch, fitter = salt3_photoz_fit_results
    bound = np.asarray(z_prior_sigma(np.asarray(batch.redshift_errs),
                                     fitter.config.photoz.prior_zerrscale))
    zerr = np.asarray(results.param_errors["zPHOT"])
    excess = zerr / bound - 1.0
    assert np.all(excess <= 0.01), (
        f"{int((excess > 0.01).sum())} of {len(zerr)} zPHOT errors exceed the "
        f"prior width by more than 1%, the worst by {excess.max():.1%}")


def test_photoz_fit_matches_snana_without_a_covariance_matrix(snana_fixture):
    """Photo-z under USE_DIAG_ONLY, where rounds 2+ recompute sigma rather than
    freezing it (snlc_fit.F90:10572). Photo-z defaults OPT_CHI2_SIGMA on
    (:7184), so this is the one path carrying the live error surface and the
    2*ln(sigma/sigma_last) term together."""
    fx = snana_fixture("salt3_fit_photoz_diagcov")
    config = fit_config(fx.root / "fit.nml")
    batch = load_batch(config, fx.head_fits, fx.phot_fits,
                       keep_obs_extras=True)
    fitter = build_fitter(config, build_fit_model(config), batch)
    results = run_fit(fitter, batch)

    snana_rows = parse_fitres(fx.root / "FITRES.TEXT")
    pulls = {"zPHOT": [], "x1": [], "c": []}
    for i, sid in enumerate(np.asarray(results.snids)):
        s = snana_rows.get(int(sid))
        if s is None or s.get("ERRFLAG_FIT", 0) != 0:
            continue
        for name, col, err_col in (("zPHOT", "zHEL", "zHELERR"),
                                   ("x1", "x1", "x1ERR"), ("c", "c", "cERR")):
            jx = float(np.asarray(results.params[name])[i])
            je = float(np.asarray(results.param_errors[name])[i])
            sigma = float(np.sqrt(je ** 2 + s[err_col] ** 2))
            if np.isfinite(sigma) and sigma > 0 and np.isfinite(jx):
                pulls[name].append((jx - s[col]) / sigma)

    assert len(pulls["x1"]) >= 15, f"too few cross-matched fits ({len(pulls['x1'])})"
    for name, vals in pulls.items():
        n_outliers = int((np.abs(np.asarray(vals)) > 3.0).sum())
        assert n_outliers <= 2, (
            f"{n_outliers}/{len(vals)} SNe with |pull({name})| > 3 (max 2)")


def test_host_columns_vs_snana(snana_fixture):
    """FITRES HOST_* columns parity: every host column SNANA wrote with
    'FITRES(text:host)' (the fixture nml) matches build_host_columns per CID.
    Host columns are data copies (+ computed SBMAG/CONFUSION/QZPHOT), so no
    fit is needed and the match is exact-tolerance."""
    from jax_snlc_fit.output import build_host_columns

    fx = snana_fixture("salt3_fit_photoz")
    batch = load_and_batch_snana_fits(
        str(fx.head_fits), str(fx.phot_fits), fit_bands=_LSST_BANDS,
        filter_map={b: i for i, b in enumerate(_LSST_BANDS)})
    cols = build_host_columns(batch.host_data, batch.zp_fluxcal, host_text=True,
                              zpdf_quantiles=batch.zpdf_quantiles,
                              zpdf_percentiles=batch.zpdf_percentiles,
                              quantile_spline='cubic')

    hdr, rows = None, {}
    with open(fx.root / "FITRES.TEXT") as f:
        for line in f:
            if line.startswith("VARNAMES:"):
                hdr = line.split()[1:]
            elif line.startswith("SN:"):
                v = line.split()[1:]
                rows[int(v[hdr.index("CID")])] = dict(zip(hdr, v))
    snana_host = [c for c in hdr if c.startswith("HOST")]
    assert sorted(snana_host) == sorted(cols), (
        f"missing {set(snana_host) - set(cols)}, "
        f"extra {set(cols) - set(snana_host)}")

    cids = np.asarray(batch.snids).astype(int)
    for c in snana_host:
        jax_vals = np.asarray(cols[c], dtype=float)
        snana_vals = np.array([float(rows[cid][c]) for cid in cids])
        # QZPHOT: documented quadrature difference (fine grid vs 100 bins)
        atol = 1e-3 if c.startswith('HOST_QZPHOT') else 2e-4
        assert np.allclose(jax_vals, snana_vals, rtol=1e-4, atol=atol), (
            f"{c}: max dev {np.abs(jax_vals - snana_vals).max()}")

    # without text:host only the property/mag columns appear (ITEXT_LOCAL=1)
    default_cols = build_host_columns(batch.host_data, batch.zp_fluxcal,
                                      quantile_spline='cubic')
    expected = [c for c in snana_host
                if c.split('_', 1)[1].split('_')[0] in
                ('LOGMASS', 'LOGSFR', 'LOGsSFR', 'COLOR', 'MAG', 'MAGERR')]
    assert sorted(default_cols) == sorted(expected)


def _overridden_batch(snana_fixture, data_root, files, head="salt3_fit_photoz",
                      dofit_photoz=False):
    """The salt3_fit_photoz light curves loaded under a SIM_HEADER_OVERRIDE_FILE
    list from the header_override data directory, with ``head``'s HEAD."""
    from jax_snlc_fit.loader import read_header_override_file

    override = read_header_override_file(
        [",".join(str(data_root / "header_override" / f) for f in files)])
    phot = snana_fixture("salt3_fit_photoz")
    return load_and_batch_snana_fits(
        str(snana_fixture(head).root / "HEAD.FITS"), str(phot.root / "PHOT.FITS"),
        fit_bands=_LSST_BANDS, sim_header_override=override,
        filter_map={b: i for i, b in enumerate(_LSST_BANDS)},
        dofit_photoz=dofit_photoz)


def test_header_override_vs_snana(snana_fixture, data_root):
    """HEADER_OVERRIDE_FILE: overriding REDSHIFT_HELIO + VPEC per CID matches
    SNANA: overridden CIDs get the new zHEL/VPEC with SNANA-recomputed
    zCMB/zHD, untouched CIDs are unchanged. Data-derived columns, no fit
    (the reference is a spec-z fit with NFIT_ITERATION = 1)."""
    from jax_snlc_fit.output import build_event_columns

    batch = _overridden_batch(snana_fixture, data_root, ["redshift_vpec.dat"])
    pre, _, _ = build_event_columns(batch)
    built = {**pre}

    rows = _parse_text_rows(snana_fixture("salt3_fit_headover").root / "FITRES.TEXT")
    idx = {int(c): i for i, c in enumerate(np.asarray(batch.snids))}
    for cid, r in rows.items():
        i = idx[cid]
        for c in ("zHEL", "zHELERR", "zCMB", "zCMBERR", "zHD", "zHDERR",
                  "VPEC", "VPECERR"):
            assert np.isclose(float(built[c][i]), float(r[c]),
                              rtol=1e-4, atol=2e-4), (
                f"{c} CID {cid}: jax {built[c][i]} snana {r[c]}")


def _assert_host_photoz_columns(batch, rows, prefixes, host2_text=False):
    """HOST[2]_ZPHOT[ERR] (the override's grid statistics, exact) and
    HOST[2]_QZPHOT[STD] (the splined PDF, within the documented quadrature)
    per CID against an SNANA FITRES."""
    from jax_snlc_fit.output import build_host_columns

    cols = build_host_columns(batch.host_data, batch.zp_fluxcal, host_text=True,
                              host2_text=host2_text,
                              zpdf_quantiles=batch.zpdf_quantiles,
                              zpdf_percentiles=batch.zpdf_percentiles,
                              quantile_spline='cubic')
    idx = {int(c): i for i, c in enumerate(np.asarray(batch.snids))}
    for cid, r in rows.items():
        i = idx[cid]
        for prefix in prefixes:
            for c, atol in ((f"{prefix}_ZPHOT", 1e-5), (f"{prefix}_ZPHOTERR", 1e-5),
                            (f"{prefix}_QZPHOT", 1e-3), (f"{prefix}_QZPHOTSTD", 1e-3)):
                assert np.isclose(float(cols[c][i]), float(r[c]), atol=atol), (
                    f"{c} CID {cid}: jax {cols[c][i]} snana {r[c]}")


def test_header_override_quantiles_by_cid_vs_snana(snana_fixture, data_root):
    """QZPHOTnn override columns keyed by CID replace every event's host
    quantiles, rescaled by the sim's zSN/zHOST, so HOST_QZPHOT[STD] move; a
    point photo-z override in the same set replaces HOST_ZPHOT[ERR] for the
    CIDs it lists (1 to 20) and, being present at all, stops the quantile
    grid from supplying HOST_ZPHOT[ERR] for the rest."""
    batch = _overridden_batch(snana_fixture, data_root,
                              ["qzphot_cid.dat", "zphot_cid.dat"])
    rows = _parse_text_rows(snana_fixture("salt3_fit_headover_qz").root / "FITRES.TEXT")
    _assert_host_photoz_columns(batch, rows, ["HOST"])


def test_header_override_quantiles_by_galid_vs_snana(snana_fixture, data_root):
    """One-row-per-quantile overrides keyed by GALID: the reported host from
    HOSTGAL_OBJID (rescaled by the sim), the second host from
    HOSTGAL2_OBJID (unscaled), `*` rows repeating the previous GALID. With
    no point photo-z in the override, HOST_ZPHOT[ERR] and HOST2_ZPHOT[ERR]
    are the grids' means and widths."""
    batch = _overridden_batch(snana_fixture, data_root, ["qzphot_galid.dat"],
                              head="salt3_fit_host2")
    rows = _parse_text_rows(
        snana_fixture("salt3_fit_host2").root / "FITRES_headover_qgalid.TEXT")
    _assert_host_photoz_columns(batch, rows, ["HOST", "HOST2"], host2_text=True)


@pytest.mark.parametrize("fixture, dofit_photoz", [
    ("salt3_fit_headover_logmass", False),
    ("salt3_fit_headover_logmass_pz", True)])
def test_header_override_logmass_grid_vs_snana(snana_fixture, data_root,
                                               fixture, dofit_photoz):
    """A LOGMASS(z) grid in the override replaces HOST_LOGMASS and its error by
    the grid's spline at the fit's redshift, and in a photo-z fit again at the
    host's photo-z (SET_LOGMASS, snana.F90:17831; FITANA_LOGMASS,
    snlc_fit.F90:13047). The override covers CIDs 1 to 40; the rest keep
    their catalogue mass. HOST_LOGMASS_ZPHOT stays the null SNANA writes."""
    from jax_snlc_fit.output import build_host_columns

    batch = _overridden_batch(snana_fixture, data_root, ["logmass_cid.dat"],
                              dofit_photoz=dofit_photoz)
    cols = build_host_columns(batch.host_data, batch.zp_fluxcal,
                              quantile_spline="cubic")
    rows = _parse_text_rows(snana_fixture(fixture).root / "FITRES.TEXT")
    idx = {int(c): i for i, c in enumerate(np.asarray(batch.snids))}
    changed = 0
    for cid, r in rows.items():
        i = idx[cid]
        for c in ("HOST_LOGMASS", "HOST_LOGMASS_ERR", "HOST_LOGMASS_ZPHOT"):
            assert np.isclose(float(cols[c][i]), float(r[c]), atol=2e-5), (
                f"{c} CID {cid}: jax {cols[c][i]} snana {r[c]}")
        changed += cid <= 40
    assert changed >= 35


def test_event_columns_vs_snana(snana_fixture):
    """Data-derived FITRES event columns match SNANA per CID with no fit.
    The doctored host2 fixture (nonzero VPEC, spec-z, NFIT_ITERATION=1)
    additionally anchors helio sourcing, the Davis zHD correction, and the
    SNR/BANDLIST family over all epochs; the photoz fixture's SNR columns
    are fit-windowed (see host2 README) so only the pure data copies are
    compared there."""
    from jax_snlc_fit.output import build_event_columns

    data_cols = ["IDSURVEY", "TYPE", "FIELD", "CUTFLAG_SNANA", "ERRFLAG_FIT",
                 "RA", "DEC", "VPEC", "VPECERR", "LENSDMU", "LENSDMUERR",
                 "MWEBV", "PKMJDINI"]
    host2_cols = ["zHEL", "zHELERR", "zCMB", "zCMBERR", "zHD", "zHDERR",
                  "SNRMAX1", "SNRMAX2", "SNRMAX3", "SNRSUM", "BANDLIST"]

    photoz = snana_fixture("salt3_fit_photoz")
    host2 = snana_fixture("salt3_fit_host2")
    for root, cols in ((photoz.root, data_cols),
                       (host2.root, data_cols + host2_cols)):
        batch = load_and_batch_snana_fits(
            str(root / "HEAD.FITS"), str(photoz.phot_fits),
            fit_bands=_LSST_BANDS,
            filter_map={b: i for i, b in enumerate(_LSST_BANDS)})
        pre, post, _ = build_event_columns(batch)
        built = {**pre, **post}

        rows = _parse_text_rows(root / "FITRES.TEXT")
        idx = {int(c): i for i, c in enumerate(np.asarray(batch.snids))}
        for c in cols:
            for cid, r in rows.items():
                jv = built[c][idx[cid]]
                if c in ("FIELD", "BANDLIST"):
                    assert str(jv) == r[c], f"{c} CID {cid}: {jv} vs {r[c]}"
                else:
                    assert np.isclose(float(jv), float(r[c]),
                                      rtol=1e-4, atol=2e-4), (
                        f"{c} CID {cid}: jax {jv} snana {r[c]}")


def _parse_text_rows(path):
    hdr, rows = None, {}
    with open(path) as f:
        for line in f:
            if line.startswith("VARNAMES:"):
                hdr = line.split()[1:]
            elif line.startswith("SN:"):
                v = line.split()[1:]
                rows[int(v[hdr.index("CID")])] = dict(zip(hdr, v))
    return rows


def test_snana_table_vs_snana(snana_fixture):
    """SNANA-table parity on both fixtures, no fit: rows for ALL events,
    CUTFLAG/ERRFLAG gating, sentinel SNR for prefiltered events, FIELD from
    PHOT epochs, host/SIM blocks. fitted_snids come from the reference
    FITRES (SNANA's CUTFLAG=3 set), so fitted-z columns (photo-z mode) and
    the 1-vs-3 split (SNANA fit rejects) are excluded from value checks."""
    from jax_snlc_fit.output import build_snana_table

    for name, zfitted in (("salt3_fit_photoz", True),
                          ("salt3_fit_select", False)):
        fx = snana_fixture(name)
        config = fit_config(fx.root / "fit.nml")
        model = build_fit_model(config)
        batch = load_batch(
            config, fx.root / "HEAD.FITS", fx.root / "PHOT.FITS",
            photflag_mskrej=config.selection.photflag_mskrej,
            snfield_list=config.selection.snfield_list,
            model_sim_columns=model_sim_columns(model))
        fitted = list(_parse_text_rows(fx.root / "FITRES.TEXT"))
        sntable = config.output
        # only the fitted set is reproducible here; the 1-vs-3 split is
        # excluded from the checks below
        cutflag, errflag = _snana_flags(mark_errflags(batch, (fitted, 0)))
        varnames, cols, _ = build_snana_table(
            batch, sntable, cutflag=cutflag, errflag=errflag,
            nepoch_outlier={} if 'outlier' in sntable.tables else None,
            quantile_spline=config.photoz.quantile_spline())

        rows = _parse_text_rows(fx.root / "SNANA.TEXT")
        hdr = None
        with open(fx.root / "SNANA.TEXT") as f:
            for line in f:
                if line.startswith("VARNAMES:"):
                    hdr = line.split()[1:]
                    break
        assert varnames == hdr
        assert len(rows) == batch.n_sne
        idx = {int(c): i for i, c in enumerate(np.asarray(batch.snids))}

        skip = {"CID", "CUTFLAG_SNANA", "ERRFLAG_FIT", "NEPOCH_OUTLIER"}
        if zfitted:
            skip |= {"zHEL", "zHELERR", "zCMB", "zCMBERR", "zHD", "zHDERR"}
        for cid, r in rows.items():
            i = idx[cid]
            ref_cut = int(r["CUTFLAG_SNANA"])
            jax_cut = int(cols["CUTFLAG_SNANA"][i])
            # SNANA's fit rejects (CUTFLAG 1) are not reproducible; the
            # fitted set is
            assert (jax_cut == 3) == (ref_cut == 3), f"CID {cid}"
            if ref_cut != 1:
                assert int(cols["ERRFLAG_FIT"][i]) == int(r["ERRFLAG_FIT"]), \
                    f"CID {cid}"
            for c in hdr:
                if c in skip:
                    continue
                jv = cols[c][i]
                try:
                    ok = np.isclose(float(jv), float(r[c]),
                                    rtol=1e-4, atol=1e-3 if "QZPHOT" in c
                                    else 2e-4)
                except ValueError:
                    ok = (str(jv) == r[c])
                assert ok, f"{name} {c} CID {cid}: jax {jv} snana {r[c]}"

        if name == "salt3_fit_select":
            # the cut boundary is deterministic: jax's own cut decisions
            # must reproduce CUTFLAG 0 vs {1,3} per event, exactly
            selected = select_events(
                batch, snana_selection=build_selection(
                    config, dict(config.selection.cutwin),
                    sncid_list(config)), ladder=[],
                max_process=config.selection.mxevt_process,
                max_fit=config.selection.mxevt_cuts)
            jax_pass = set(np.asarray(selected.snids).astype(int).tolist())
            ref_pass = {cid for cid, r in rows.items()
                        if int(r["CUTFLAG_SNANA"]) in (1, 3)}
            assert jax_pass == ref_pass, (
                f"jax-only {sorted(jax_pass - ref_pass)}, "
                f"snana-only {sorted(ref_pass - jax_pass)}")


def test_outlier_table_vs_snana(salt3_photoz_fit_results):
    """OUTLIER table parity: data-copy columns exact on rows both codes
    select; row membership and the fit-derived columns (FLUXCAL_FIT/ERR_FIT/
    ERRTOT/NSIG/ERR_CALC) are fit-dependent — SNANA's NSIG even uses an
    earlier-iteration fit flux — so membership is gated as an overlap
    fraction. TREST carries SNANA's float32-MJD truncation
    (SNGL(MJD8), snana.F90:17280): gated at 5e-3."""
    from jax_snlc_fit.output import build_outlier_table

    fx, results, batch, fitter = salt3_photoz_fit_results
    mask = np.asarray(results.observation_mask)[:batch.n_sne]
    zeval = results.params.get("zPHOT", batch.redshifts)
    mf, mfe = fitter.evaluate_flux(
        results.params,
        np.asarray(batch.reference_mjds)[:, None] +
        np.asarray(batch.mjd_offsets),
        zeval, batch.filter_indices, mask,
        reference_mjds=batch.reference_mjds, mw_att_obs=batch.mw_att_obs,
        photometry=batch.photometry)
    _, cols, _, _ = build_outlier_table(batch, fitter.config.output, mf, mfe)

    hdr, ref = None, {}
    with open(fx.root / "OUTLIER.TEXT") as f:
        for line in f:
            if line.startswith("VARNAMES:"):
                hdr = line.split()[1:]
            elif line.startswith("SN:"):
                d = dict(zip(hdr, line.split()[1:]))
                ref[(d["CID"], round(float(d["MJD"]), 3), d["BAND"])] = d

    fit_snids = set(np.char.strip(np.asarray(batch.snids).astype(str)))
    ref_here = {k: v for k, v in ref.items() if k[0] in fit_snids}
    jax_keys = [(str(cols["CID"][i]).strip(), round(float(cols["MJD"][i]), 3),
                 str(cols["BAND"][i])) for i in range(len(cols["CID"]))]
    shared = [k for k in jax_keys if k in ref_here]
    assert len(shared) >= 0.75 * len(jax_keys), (
        f"only {len(shared)}/{len(jax_keys)} jax outlier rows in reference")
    assert len(shared) >= 50

    jmap = {k: i for i, k in enumerate(jax_keys)}
    for k in shared:
        i, r = jmap[k], ref_here[k]
        for c in ("FIELD", "IFILTOBS", "DETNUM", "IXPIX", "IYPIX",
                  "AREAFRAC", "PHOTFLAG", "FLUXCAL_DATA", "FLUXCAL_ERR_DATA",
                  "LOGSNR", "SBMAG", "ZP", "PSF", "SKYSIG", "z"):
            jv = cols[c][i]
            try:
                ok = np.isclose(float(jv), float(r[c]), rtol=1e-4, atol=2e-4)
            except ValueError:
                ok = (str(jv) == r[c])
            assert ok, f"{c} {k}: jax {jv} snana {r[c]}"
        assert np.isclose(float(cols["TREST"][i]), float(r["TREST"]),
                          atol=5e-3), k


def test_lcplot_vs_snana(salt3_photoz_fit_results):
    """LCPLOT parity for the reference's plotted CIDs: data rows exact
    (fluxes, SIM_FLUXCAL, DATAFLAG, MJD at 1dp — SNANA prints float32
    MJDs); the fit-curve grid has SNANA's exact Tobs points; CHI2/Tobs
    and curve fluxes are fit-derived and not value-pinned."""
    from jax_snlc_fit.output import (LCPLOT_TOBS_GRID, build_lcplot_table)

    fx, results, batch, fitter = salt3_photoz_fit_results
    hdr, ref = None, []
    with gzip.open(fx.root / "LCPLOT.TEXT.gz", "rt") as f:
        for line in f:
            if line.startswith("VARNAMES:"):
                hdr = line.split()[1:]
            elif line.startswith("OBS:"):
                ref.append(dict(zip(hdr, line.split()[1:])))
    ref_cids = list(dict.fromkeys(r["CID"] for r in ref))

    snids = np.char.strip(np.asarray(batch.snids).astype(str))
    plot_idx = np.array([int(np.nonzero(snids == c)[0][0])
                         for c in ref_cids])
    mask = np.asarray(results.observation_mask)[:batch.n_sne]
    zeval = np.asarray(results.params.get("zPHOT", batch.redshifts))
    mjds = (np.asarray(batch.reference_mjds)[:, None] +
            np.asarray(batch.mjd_offsets))
    mf, mfe = fitter.evaluate_flux(
        results.params, mjds, zeval, batch.filter_indices,
        batch.observation_mask, reference_mjds=batch.reference_mjds,
        mw_att_obs=batch.mw_att_obs, photometry=batch.photometry)
    pkmjd = np.asarray(results.params["PKMJD"])
    n_bands = len(fitter.filters)
    gm = pkmjd[plot_idx, None] + np.tile(LCPLOT_TOBS_GRID, n_bands)[None, :]
    gf = np.repeat(np.arange(n_bands),
                   len(LCPLOT_TOBS_GRID))[None, :].repeat(len(plot_idx),
                                                          axis=0)
    gflux, gferr = fitter.evaluate_flux(
        {k: np.asarray(v)[plot_idx] for k, v in results.params.items()},
        gm, zeval[plot_idx], gf, np.ones_like(gm, dtype=bool),
        mw_att_obs=np.asarray(batch.mw_att_obs)[plot_idx],
        photometry=batch.photometry,
        reference_mjds=np.asarray(batch.reference_mjds)[plot_idx])
    shape = (len(plot_idx), n_bands, len(LCPLOT_TOBS_GRID))
    varnames, cols, _ = build_lcplot_table(
        batch, mask, np.asarray(mf), np.asarray(mfe), pkmjd, plot_idx,
        np.asarray(gflux).reshape(shape), np.asarray(gferr).reshape(shape))
    assert varnames == hdr

    # match by nearest MJD per (CID, band): SNANA prints float32 MJDs
    # (up to ~4e-3 off the float64 value)
    jrows = {}
    for i in range(len(cols["CID"])):
        if int(cols["DATAFLAG"][i]) == 0:
            continue
        jrows.setdefault((str(cols["CID"][i]).strip(),
                          str(cols["BAND"][i])), []).append(i)
    n_data = 0
    trest_lo, trest_hi = fitter.config.fit.fitwin_trest
    for r in ref:
        if r["DATAFLAG"] == "0":
            continue
        n_data += 1
        cand = jrows[(r["CID"], r["BAND"])]
        dm = [abs(float(cols["MJD"][i]) - float(r["MJD"])) for i in cand]
        i = cand[int(np.argmin(dm))]
        assert min(dm) < 0.01, (r["CID"], r["BAND"], r["MJD"])
        # The codes fix the FITWIN_TREST window at slightly different PKMJD
        # and zPHOT, so an epoch a few hundredths of a day from its edge can
        # be used by one only.
        ei = plot_idx[ref_cids.index(r["CID"])]
        trest = (float(r["MJD"]) - pkmjd[ei]) / (1.0 + zeval[ei])
        if min(abs(trest - trest_lo), abs(trest - trest_hi)) > 0.2:
            assert int(cols["DATAFLAG"][i]) == int(r["DATAFLAG"]), r["MJD"]
        for c in ("SIM_FLUXCAL", "FLUXCAL", "FLUXCAL_ERR"):
            assert np.isclose(float(cols[c][i]), float(r[c]),
                              rtol=1e-4, atol=2e-4), (r["MJD"], c)
    assert n_data == sum(len(v) for v in jrows.values())

    # a band's fit curve only where the fit used an epoch of it
    # (EXIST_FILT, snana.F90:17454, :22250)
    jax_curves = {(str(cols["CID"][i]).strip(), str(cols["BAND"][i]))
                  for i in range(len(cols["CID"])) if int(cols["DATAFLAG"][i]) == 0}
    ref_curves = {(r["CID"], r["BAND"]) for r in ref if r["DATAFLAG"] == "0"}
    assert jax_curves == ref_curves

    ref_grid = sorted({float(r["Tobs"]) for r in ref if r["DATAFLAG"] == "0"
                       and r["CID"] == ref_cids[0] and r["BAND"] == "u"})
    assert np.allclose(ref_grid, LCPLOT_TOBS_GRID)


def test_event_columns_fit_stage_vs_snana(salt3_photoz_fit_results):
    """In photo-z mode SNANA fills zHEL/zCMB/zHD from the fitted zPHOT;
    the zfit override reproduces that within combined fit errors."""
    from jax_snlc_fit.output import build_event_columns

    fx, results, batch, _ = salt3_photoz_fit_results
    pre, _, _ = build_event_columns(
        batch,
        zfit=results.params["zPHOT"], zfit_err=results.param_errors["zPHOT"])

    rows = _parse_text_rows(fx.root / "FITRES.TEXT")
    idx = {int(c): i for i, c in enumerate(np.asarray(results.snids))}
    zerr = np.asarray(results.param_errors["zPHOT"])
    shared = [cid for cid in rows if cid in idx
              and np.isfinite(zerr[idx[cid]]) and zerr[idx[cid]] > 0]
    assert len(shared) >= 15, f"too few cross-matched fits ({len(shared)})"

    for cid in shared:
        r, i = rows[cid], idx[cid]
        for c in ("zHEL", "zCMB", "zHD"):
            sigma = np.sqrt(float(r[c + "ERR"]) ** 2 +
                            float(pre[c + "ERR"][i]) ** 2)
            pull = (float(pre[c][i]) - float(r[c])) / sigma
            assert abs(pull) < 3.0, f"{c} CID {cid}: pull {pull:.2f}"


def test_host2_confusion_vs_snana(snana_fixture):
    """Doctored-HEAD anchor (salt3_fit_host2): HOST_CONFUSION Gupta-Eq-3
    branch + HOST2_* propagation vs SNANA text:host2. HOST2_MAG/MAGERR are
    gated against the HEAD values (SNANA's reader writes 99)."""
    from jax_snlc_fit.output import build_host_columns

    fx = snana_fixture("salt3_fit_host2")
    phot = snana_fixture("salt3_fit_photoz")
    batch = load_and_batch_snana_fits(
        str(fx.root / "HEAD.FITS"), str(phot.root / "PHOT.FITS"),
        fit_bands=_LSST_BANDS,
        filter_map={b: i for i, b in enumerate(_LSST_BANDS)})
    cols = build_host_columns(batch.host_data, batch.zp_fluxcal, host2_text=True,
                              zpdf_quantiles=batch.zpdf_quantiles,
                              zpdf_percentiles=batch.zpdf_percentiles,
                              quantile_spline='cubic')

    hdr, rows = None, {}
    with open(fx.root / "FITRES.TEXT") as f:
        for line in f:
            if line.startswith("VARNAMES:"):
                hdr = line.split()[1:]
            elif line.startswith("SN:"):
                v = line.split()[1:]
                rows[int(v[hdr.index("CID")])] = dict(zip(hdr, v))
    snana_host = [c for c in hdr if c.startswith("HOST")]
    assert sorted(snana_host) == sorted(cols)

    cids = np.asarray(batch.snids).astype(int)
    hc = np.asarray(cols['HOST_CONFUSION'])
    assert (hc[:8] > -9).all() and len(set(np.round(hc[:8], 3))) == 8

    for c in snana_host:
        jax_vals = np.asarray(cols[c], dtype=float)
        if c.startswith('HOST2_MAG'):
            ref = np.asarray(batch.host_data['HOSTGAL2' + c[5:]], dtype=float)
        else:
            ref = np.array([float(rows[cid][c]) for cid in cids])
        atol = 1e-3 if 'QZPHOT' in c else 2e-4
        assert np.allclose(jax_vals, ref, rtol=1e-4, atol=atol), (
            f"{c}: max dev {np.abs(jax_vals - ref).max()}")


def test_photoz_trest_cut_matches_snana(snana_fixture, repo_root, data_root,
                                        tmp_path):
    """A photo-z fit relaxes the Trest-family cuts and re-tests after fitting.

    SNANA widens each window by (1+z) against PHOTOZ_BOUND(2) before fitting
    (TRESTzCUT, snlc_fit.F90:3289) and then counts the filters still holding an
    epoch inside the *nominal* window, cutting on that count rather than on the
    window itself (GET_TRESTVAR :13234, CUTWIN_NFILT_TRESTMIN default 1).

    Anchor: snlc_fit.exe run on this fixture with CUTWIN_TRESTMIN = -20, 0 keeps
    all 100 events, exactly as it does with no cut at all. Applying the nominal
    window directly as a per-SN cut instead drops five.
    """
    fx = snana_fixture("salt3_fit_photoz")
    version = "JAX_TEST_SALT3_FIT_PHOTOZ"
    vdir = tmp_path / version
    vdir.mkdir()
    (vdir / f"{version}_HEAD.FITS").write_bytes(
        gzip.decompress((fx.root / "HEAD.FITS.gz").read_bytes()))
    (vdir / f"{version}_PHOT.FITS").write_bytes(
        gzip.decompress((fx.root / "PHOT.FITS.gz").read_bytes()))
    (vdir / f"{version}.LIST").write_text(f"{version}_HEAD.FITS\n")

    env = os.environ.copy()
    env["JAX_TEST_FIXTURE_ROOT"] = str(data_root)
    env["JAX_TEST_FIT_SIM_DIR"] = str(tmp_path)

    def rows(name, *overrides):
        out = tmp_path / name
        r = subprocess.run(
            [sys.executable, str(repo_root / "snlc_fit.py"),
             str(fx.root / "fit.nml"), "BATCH_SIZE", "4", *overrides, "OUTPUT_NAME", str(out)],
            capture_output=True, text=True, env=env, timeout=1800)
        assert r.returncode == 0, r.stdout[-1200:] + r.stderr[-1200:]
        return sum(1 for l in out.read_text().splitlines() if l.startswith("SN:"))

    # snlc_fit.exe anchors on this fixture, one run per cut
    assert rows("PZBASE.FITRES.TEXT") == 100
    assert rows("PZMIN.FITRES.TEXT", "CUTWIN_TRESTMIN", "-20,0") == 100
    assert rows("PZTREST.FITRES.TEXT", "CUTWIN_TREST", "-15.0,30.0") == 100
    assert rows("PZGAP.FITRES.TEXT", "CUTWIN_TGAPMAX", "0.0,10.0") == 100
    # the discriminating case: this one really does cut, 100 -> 6, so a
    # regression in the widen-then-recount path shows up here
    assert rows("PZMAX.FITRES.TEXT", "CUTWIN_TRESTMAX", "0.0,20.0") == 6

    # T0GAPMAX counts only gaps touching the peak window, and that window is
    # (CUTWIN_TRESTMIN(2), CUTWIN_TRESTMAX(1)) rather than a key of its own
    # (snana.F90:21105-21106). Left open it inverts and collapses to TGAPMAX; bounded
    # it is a different cut, so both regimes are anchored.
    assert rows("PZT0.FITRES.TEXT", "CUTWIN_T0GAPMAX", "0.0,12.0") == 100
    assert rows("PZT0WIN.FITRES.TEXT", "CUTWIN_TRESTMIN", "-20.0,-2.0",
                "CUTWIN_TRESTMAX", "8.0,60.0",
                "CUTWIN_T0GAPMAX", "0.0,6.0") == 99


def test_photoz_fit_in_fewer_bands_than_the_data(snana_fixture, repo_root,
                                                 data_root, tmp_path):
    """A photo-z fit whose FILTLIST_FIT leaves out a band of the data, here u
    of ugrizY, tests ZFIT_RANGE on its own fit bands, each against that
    band's rest-frame range (LTEST_ZPHFILTER, snlc_fit.F90:3569-3572); the
    band left out is never fitted."""
    fx = snana_fixture("salt3_fit_photoz")
    version = "JAX_TEST_SALT3_FIT_PHOTOZ"
    vdir = tmp_path / version
    vdir.mkdir()
    for tail in ("HEAD.FITS", "PHOT.FITS"):
        (vdir / f"{version}_{tail}").write_bytes(
            gzip.decompress((fx.root / (tail + ".gz")).read_bytes()))
    (vdir / f"{version}.LIST").write_text(f"{version}_HEAD.FITS\n")
    env = {**os.environ, "JAX_TEST_FIXTURE_ROOT": str(data_root),
           "JAX_TEST_FIT_SIM_DIR": str(tmp_path)}
    out = tmp_path / "GRIZY.FITRES.TEXT"

    r = subprocess.run(
        [sys.executable, str(repo_root / "snlc_fit.py"), str(fx.root / "fit.nml"), "BATCH_SIZE", "4",
         "FILTLIST_FIT", "grizY", "OUTPUT_NAME", str(out)],
        capture_output=True, text=True, env=env, timeout=1800)

    assert r.returncode == 0, r.stdout[-1200:] + r.stderr[-1200:]
    bandlists = read_fitres(str(out)).df["BANDLIST"].astype(str)
    assert len(bandlists) > 0
    assert all(set(b) <= set("grizY") for b in bandlists)


def test_snana_flags_iter1_codes(snana_fixture):
    """The round-1 verdicts reach the SNANA table with their own codes
    (snlc_fit.F90:99-112) rather than the generic FIT_CUTS: 5 for the
    FITINI_ADJUST chi2 window, 15-17 for FITPROB/SHAPE/COLOR_ITER1. An
    event dropped by any other post-fit cut keeps FIT_CUTS, and one the
    selection dropped keeps the no-fit code."""
    fx = snana_fixture("salt3_fit_photoz")
    config = fit_config(fx.root / "fit.nml")
    loaded = load_batch(config, fx.head_fits, fx.phot_fits, max_fit=7)
    precut = mark_errflags(loaded, (loaded.snids[:6], ERRFLAG_FIT_CUTS))
    fitted = loaded.take(np.arange(6))
    zero = np.zeros(6)
    results = BatchFitResult(
        snids=fitted.snids, params={'x1': zero}, param_errors={'x1': zero},
        param_covariances=np.zeros((6, 1, 1)), cov_output_params=['x1'],
        chi_squared=zero, ndof=zero, prior_chi2=zero,
        observation_mask=np.asarray(fitted.observation_mask),
        hessian_cond=zero, hessian_positive=np.ones_like(zero, dtype=bool), edm=zero,
        z_prior=None, iter1_verdicts={})
    keep = np.array([True, False, False, False, False, False])
    codes = np.array([0, ERRFLAG_FITPREP_ADJUST, ERRFLAG_FIT_FITPROB_ITER1,
                      ERRFLAG_FIT_SHAPE_ITER1, ERRFLAG_FIT_COLOR_ITER1, 0])
    _, written, precut = select(fitted, results, precut, keep, codes)
    assert written.n_sne == 1
    cutflag, errflag = _snana_flags(precut)
    assert errflag.tolist() == [0, 5, 15, 16, 17, ERRFLAG_FIT_CUTS,
                                ERRFLAG_NO_FIT]
    assert cutflag.tolist() == [3, 1, 1, 1, 1, 1, 0]


def test_lcplot_rebuilt_from_fitres(snana_fixture, tmp_path):
    """lcplot_from_fitres rebuilds the fit's own LCPLOT from its FITRES alone:
    data rows exact, the curve and CHI2 to the FITRES print precision, and
    the flags exact but near a FITWIN_TREST edge. The fit sets that window at
    its round-1 PKMJD and zPHOT, which the FITRES does not hold; these fits
    barely move after round 1, so only an epoch within 0.2 rest-frame days
    of an edge can be flagged either way."""
    fx = snana_fixture("salt3_fit_photoz")
    fit_rows, rebuilt, fitres = regenerate_lcplot(fx.root, tmp_path)
    diff = compare_lcplots(fit_rows, rebuilt)
    assert diff["missing"] == 0 and diff["data_changed"] == 0, diff
    assert diff["curve"] < 2e-4 and diff["chi2"] < 0.011, diff

    trest_lo, trest_hi = fit_config(fx.root / "fit.nml").fit.fitwin_trest
    for cid, mjd in diff["flags"]:
        row = fitres[int(cid)]
        trest = (mjd - row["PKMJD"]) / (1.0 + row["zHEL"])
        assert min(abs(trest - trest_lo), abs(trest - trest_hi)) < 0.2, (
            cid, mjd, trest)
