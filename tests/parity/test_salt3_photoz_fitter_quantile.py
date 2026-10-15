"""SALT3 photo-z fitter with quantile-PDF prior (OPT_PHOTOZ=4 in NML).

The bundled HOSTLIB has HOSTGAL_ZPHOT_Q000..Q100 quantile columns. The fitter
should accept the quantile prior path (different code branch from the
Gaussian OPT_PHOTOZ=2) and recover z values close to truth.
"""

import numpy as np

from _fit_helpers import (build_fit_model, build_fitter, fit_config,
                          load_batch, parse_fitres, run_fit)


def test_salt3_photoz_quantile_runs(snana_fixture):
    """Quantile-PDF prior fits run cleanly and recover z near truth."""
    fx = snana_fixture("salt3_fit_photoz_quantile")
    config = fit_config(fx.root / "fit.nml")

    data = snana_fixture("salt3_fit_photoz")
    batch = load_batch(config, data.head_fits, data.phot_fits, max_fit=30)
    assert batch.n_sne >= 15, f"too few fittable SNe ({batch.n_sne})"
    assert batch.zpdf_quantiles is not None, (
        "the loader stopped surfacing zpdf_quantiles from HEAD.FITS"
    )
    # Drop SNe whose host wasn't matched (HOSTGAL_ZPHOT_Q* all = -9 sentinel).
    q = np.asarray(batch.zpdf_quantiles)
    has_host = (q > 0).all(axis=1)
    if not has_host.all():
        batch = batch.filter_by_mask(has_host)
    assert batch.n_sne >= 15, (
        f"too few host-matched SNe for the quantile fit ({batch.n_sne})"
    )

    results = run_fit(build_fitter(config, build_fit_model(config), batch),
                      batch)

    snids = np.asarray(results.snids)
    z_jax = np.asarray(results.params["zPHOT"])

    from astropy.io import fits
    with fits.open(data.head_fits) as hdul:
        head = hdul[1].data
        truth_z = {int(r["SNID"]): float(r["SIM_REDSHIFT_CMB"]) for r in head}

    diffs = []
    for i, sid in enumerate(snids):
        cid = int(sid)
        if cid in truth_z:
            diffs.append(z_jax[i] - truth_z[cid])

    assert len(diffs) >= 10
    diffs = np.abs(np.asarray(diffs))
    median = float(np.median(diffs))
    p95 = float(np.percentile(diffs, 95))

    # The quantile prior carries the same Gaussian-derived information as
    # OPT_PHOTOZ=2; recovery should be similar (median <0.01, p95 <0.05).
    assert median < 0.01, f"median |z_fit - ZTRUE| = {median:.4f} > 0.01"
    assert p95 < 0.05, f"p95 |z_fit - ZTRUE| = {p95:.4f} > 0.05"
    # zPRIOR/zPRIORERR are the PDF's mean and width (snlc_fit.F90:8561),
    # SNANA's from 100 bins of the spline (init_spline, sntools_spline.h:70)
    _assert_prior_columns(results, parse_fitres(fx.root / "FITRES.TEXT"), 0.001)


def test_steffen_quantile_prior_vs_snana(snana_fixture):
    """OPT_PHOTOZ = 12 reads the quantile PDF as the derivative of Steffen's
    monotone cubic through the CDF, GSL's steffen (init_spline,
    sntools_spline.c), rather than the natural spline's. The fixture is
    the quantile fixture's light curves refitted by SNANA under 12; each
    event's fitted redshift must be SNANA's to the level the cubic case
    reaches."""
    fx = snana_fixture("salt3_fit_photoz_steffen")
    data = snana_fixture("salt3_fit_photoz")
    config = fit_config(fx.root / "fit.nml")
    assert config.photoz.quantile_spline() == "steffen"
    batch = load_batch(config, data.head_fits, data.phot_fits, keep_obs_extras=True)
    # the hostless events carry the -9 sentinel in every quantile
    batch = batch.filter_by_mask((np.asarray(batch.zpdf_quantiles) > 0).all(axis=1))
    results = run_fit(build_fitter(config, build_fit_model(config), batch), batch)
    snana = parse_fitres(fx.root / "FITRES.TEXT")

    z_jax = {int(s): float(z) for s, z in
             zip(np.asarray(results.snids), np.asarray(results.params["zPHOT"]))}
    both = sorted(set(z_jax) & set(snana))
    assert len(both) >= 80, f"too few cross-matched fits ({len(both)})"
    # the fitted redshift is zHEL (SKIPTABLE_FITPAR, snlc_fit.F90:19042)
    dz = np.array([z_jax[c] - snana[c]["zHEL"] for c in both])
    assert np.median(np.abs(dz)) < 0.001, np.median(np.abs(dz))
    assert np.abs(dz).max() < 0.01, [both[k] for k in np.flatnonzero(np.abs(dz) >= 0.01)]
    # zPRIOR/zPRIORERR are the PDF's mean and width; SNANA's are from 100
    # bins of the spline (init_spline), which the corners of Steffen's curve
    # put up to 0.002 from the fine grid's
    _assert_prior_columns(results, snana, 0.003)


def _assert_prior_columns(results, snana, tol):
    for cid, zp, zpe in zip(np.asarray(results.snids).astype(int),
                            *map(np.asarray, results.z_prior)):
        if cid in snana:
            assert abs(zp - snana[cid]["zPRIOR"]) < tol, (cid, zp, snana[cid]["zPRIOR"])
            assert abs(zpe - snana[cid]["zPRIORERR"]) < tol, (cid, zpe, snana[cid]["zPRIORERR"])
