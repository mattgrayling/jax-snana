"""BayeSN W22 fitter recovers truth params from bundled simulated data.

Pins JAX BayeSN fitting end-to-end: the fixture is generated with a
known truth (theta, AV, RV) and no DELTAM/EPSILON scatter, and recovered
theta + AV must agree with truth within a generous tolerance set by the
prior widths.
"""

import gzip
import jax.numpy as jnp
import numpy as np
import pytest

from _fit_helpers import (build_fit_model, build_fitter, compare_lcplots,
                          fit_config, fitter_photometry, load_batch,
                          parse_fitres, rebuild_lcplot, regenerate_lcplot,
                          run_fit)
from test_salt3_fitter import (assert_analytic_derivatives,
                               one_sn_flux)

from jax_snlc_core.models import SNModel
from jax_snlc_core.utils.error_handling import SNANAError
from jax_snlc_fit.fit_spec import init_params
from jax_snlc_fit.photoz import coarse_grid_axes
from jax_snlc_fit.config import FitConfig, Model, PhotoZ


# Two SNe per fit call keeps the vmapped batch CPU-sized (CI is 2-core).
_BATCH_SIZE = 2


@pytest.fixture(scope="module")
def bayesn_setup(snana_fixture):
    """The fixture's light curves and a fitter built over them.

    The grid derives from W22's declared 3000-18500 A knot support, so every
    evaluated point lies inside the model and its NIR coverage is complete.
    """
    fx = snana_fixture("bayesn_fit_dense")
    config = fit_config(fx.root / "fit.nml", batch_size=_BATCH_SIZE)
    batch = load_batch(config, fx.head_fits, fx.phot_fits, max_fit=20)
    assert batch.n_sne >= 8, f"too few fittable SNe ({batch.n_sne})"

    model = build_fit_model(config)
    return fx, batch, model, build_fitter(config, model, batch), config


@pytest.fixture(scope="module")
def bayesn_fit_results(bayesn_setup):
    """Run the BayeSN fit on the bayesn_fit_dense fixture once for both
    parity tests. Each fit is ~12-18s of JIT + optimisation."""
    fx, batch, _, fitter, _ = bayesn_setup
    return fx, run_fit(fitter, batch), batch


def test_deferred_parameter_errors(snana_fixture):
    fx = snana_fixture("bayesn_fit_dense")
    config = fit_config(fx.root / "fit.nml", n_fit_iterations=1, batch_size=1)
    batch = load_batch(config, fx.head_fits, fx.phot_fits, max_fit=1)
    model = build_fit_model(config)
    deferred = [name for name, spec in model.get_fit_parameters().items()
                if spec.get('deferred')]
    result = run_fit(build_fitter(config, model, batch), batch)
    assert deferred
    for name in deferred:
        np.testing.assert_array_equal(result.params[name], 0.)
        np.testing.assert_array_equal(result.param_errors[name], 0.)

    # Explicitly holding the same parameters must give the same uncertainties.
    config.model.inistp.update({name: 0. for name in deferred})
    held = run_fit(build_fitter(config, model, batch), batch)
    np.testing.assert_allclose(result.param_covariances, held.param_covariances)


def test_bayesn_fitter_truth_recovery(bayesn_fit_results):
    """JAX BayeSN fitter recovers theta and AV within prior-derived tolerances."""
    fx, results, batch = bayesn_fit_results

    from astropy.io import fits
    with fits.open(fx.head_fits) as hdul:
        head = hdul[1].data
        truth = {int(row["SNID"]): {
            "theta": float(row["SIM_THETA"]),
            "av":    float(row["SIM_AV"]),
        } for row in head}

    snids = np.asarray(results.snids)
    theta_jax = np.asarray(results.params["THETA"])
    av_jax = np.asarray(results.params["AV"])

    theta_diff = []
    av_diff = []
    for i, sid in enumerate(snids):
        cid = int(sid)
        if cid not in truth:
            continue
        theta_diff.append(theta_jax[i] - truth[cid]["theta"])
        av_diff.append(av_jax[i] - truth[cid]["av"])

    assert len(theta_diff) >= 5, f"too few cross-matched ({len(theta_diff)})"
    theta_arr = np.asarray(theta_diff)
    av_arr = np.asarray(av_diff)

    # GENSIGMA_THETA=0.7 in the fixture, so prior-driven recovery noise is
    # σ_theta ≈ 0.5 once data constrains it. Median |Δtheta| < 0.5 leaves a
    # comfortable margin while still flagging a fitter that's off by >1σ.
    assert float(np.median(np.abs(theta_arr))) < 0.5, (
        f"median |Δtheta| = {np.median(np.abs(theta_arr)):.3f} > 0.5"
    )
    # GENRANGE_AV uniform [0, 0.5]; recovery uncertainty per-SN is ~0.1 mag.
    assert float(np.median(np.abs(av_arr))) < 0.2, (
        f"median |ΔAV| = {np.median(np.abs(av_arr)):.3f} > 0.2"
    )


def test_bayesn_fitter_vs_snana(bayesn_fit_results):
    """JAX recovers theta and AV consistent with SNANA's snlc_fit.exe BayeSN fits."""
    fx, results, _ = bayesn_fit_results
    fitres_path = fx.root / "FITRES.TEXT"
    assert fitres_path.is_file(), (
        "bayesn_fit_dense fixture missing FITRES.TEXT — regenerate with "
        "`python tests/fixtures/regenerate.py --only bayesn_fit_dense`"
    )
    snana_rows = parse_fitres(fitres_path)

    snids = np.asarray(results.snids)
    theta_jax = np.asarray(results.params["THETA"])
    av_jax = np.asarray(results.params["AV"])
    theta_err = np.asarray(results.param_errors["THETA"])
    av_err = np.asarray(results.param_errors["AV"])

    # Per-SN deviations against SNANA, normalised by combined fit error.
    pulls_theta = []
    pulls_av = []
    for i, sid in enumerate(snids):
        cid = int(sid)
        if cid not in snana_rows:
            continue
        s = snana_rows[cid]
        if s.get("ERRFLAG_FIT", 0) != 0:
            continue
        sv_t, se_t = s.get("THETA"), s.get("THETAERR")
        sv_a, se_a = s.get("AV"), s.get("AVERR")
        if sv_t is None or sv_a is None:
            continue
        sig_t = float(np.sqrt(theta_err[i] ** 2 + (se_t or 0) ** 2))
        sig_a = float(np.sqrt(av_err[i] ** 2 + (se_a or 0) ** 2))
        if sig_t > 0 and np.isfinite(theta_jax[i]):
            pulls_theta.append((theta_jax[i] - sv_t) / sig_t)
        if sig_a > 0 and np.isfinite(av_jax[i]):
            pulls_av.append((av_jax[i] - sv_a) / sig_a)

    assert len(pulls_theta) >= 5
    assert len(pulls_av) >= 5

    # 3σ outlier gate. SNANA's BayeSN fit uses MINUIT; JAX uses LM/L-BFGS.
    # Different optimisers may land in slightly different optima for SNe with
    # weak constraints, so allow up to 2 outliers per parameter.
    n_out_t = int(np.sum(np.abs(pulls_theta) > 3))
    n_out_a = int(np.sum(np.abs(pulls_av) > 3))
    assert n_out_t <= 2, f"theta: {n_out_t} > 2 SNe with |pull| > 3"
    assert n_out_a <= 2, f"av: {n_out_a} > 2 SNe with |pull| > 3"


def test_opt_prior_av_gates_the_av_prior(snana_fixture, repo_root, data_root,
                                         tmp_path):
    """USE_AVPRIOR needs OPT_PRIOR_AV > 0 as well as a positive PRIOR_AVEXP
    (snlc_fit.F90:7165); jax honoured only the second half, so OPT_PRIOR_AV = 0
    silently left the prior on."""
    import os
    import subprocess
    import sys
    fx = snana_fixture("bayesn_fit_dense")
    version = "JAX_TEST_BAYESN_FIT_DENSE"
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

    def av(name, *overrides):
        out = tmp_path / name
        r = subprocess.run(
            [sys.executable, str(repo_root / "snlc_fit.py"),
             str(fx.root / "fit.nml"), "BATCH_SIZE", "4", *overrides, "MAX_FIT", "8",
             "OUTPUT_NAME", str(out)],
            capture_output=True, text=True, env=env, timeout=1800)
        assert r.returncode == 0, r.stdout[-1200:] + r.stderr[-1200:]
        names, vals = None, []
        for line in out.read_text().splitlines():
            if line.startswith("VARNAMES:"):
                names = line.split()[1:]
            elif line.startswith("SN:") and names:
                vals.append(float(line.split()[1:][names.index("AV")]))
        return vals

    on = av("AVON.FITRES.TEXT")
    off = av("AVOFF.FITRES.TEXT", "OPT_PRIOR_AV", "0")
    assert len(on) == len(off) == 8
    # dropping the prior must move the fitted AV
    assert max(abs(a - b) for a, b in zip(on, off)) > 1e-6


def test_bayesn_photoz_fit_recovers_the_simulated_redshift(snana_fixture):
    """Photo-z fitting is not a SNANA capability for BayeSN: snlc_fit.exe
    hard-stops in INIT_COURSEBIN_PHOTOZ for this model, so there is no
    reference to replay against and recovery of the simulated redshift is what
    can be checked. The fixture carries no HOSTGAL_PHOTOZ, so the fit runs with
    no host prior and the redshift comes from the light curve alone.

    This is the path the coarse grid takes when a model declares its own axes
    rather than SALT's: BayeSN searches THETA over RANGEs_COURSE_PHOTOZ and has
    no colour axis at all.
    """
    fx = snana_fixture("bayesn_fit_dense")
    config = fit_config(fx.root / "fit.nml", dofit_photoz=True, opt_photoz=2,
                        prior_zerrscale=1.0, photoz_bound=(0.01, 0.4),
                        batch_size=_BATCH_SIZE)
    batch = load_batch(config, fx.head_fits, fx.phot_fits, max_fit=12,
                       keep_obs_extras=True)
    model = build_fit_model(config)

    # The grid the model asks for: shape only, no colour. NBINs/RANGEs were
    # inert for every non-SALT model until the grid began asking the model what
    # to search, so pin that they now reach THETA, SCALE_NBIN included.
    specs = model.get_fit_parameters()
    axes = coarse_grid_axes(config, specs)
    assert list(axes) == ["THETA"]
    np.testing.assert_allclose(axes["THETA"], np.linspace(-3.0, 1.0, 3))
    wider = FitConfig(
        model=Model(coarse_grid={'THETA': {'range': (-2.0, 2.0), 'nbin': 5}}),
        photoz=PhotoZ(scale_nbin_course=2))
    np.testing.assert_allclose(coarse_grid_axes(wider, specs)["THETA"],
                               np.linspace(-2.0, 2.0, 10))

    results = run_fit(build_fitter(config, model, batch), batch)

    from astropy.io import fits
    with fits.open(fx.head_fits) as hdul:
        truth = {int(row["SNID"]): float(row["SIM_REDSHIFT_CMB"])
                 for row in hdul[1].data}

    z_fit = np.asarray(results.params["zPHOT"])
    z_true = np.array([truth[int(s)] for s in np.asarray(results.snids)])
    assert np.all(np.isfinite(z_fit))
    for name, values in results.params.items():
        assert np.all(np.isfinite(np.asarray(values))), name

    pull = np.abs(z_fit - z_true) / (1.0 + z_true)
    assert np.max(pull) < 0.01, (
        f"worst dz/(1+z) = {np.max(pull):.4f}; "
        f"z_fit={z_fit.tolist()} z_true={z_true.tolist()}")


def test_bayesn_closed_form_derivatives_match_the_flux(bayesn_setup):
    """BayeSN's analytic band-flux derivatives differentiate its own flux.

    The closed form is reached only with the band window on and the late-time
    extrapolation off, which is how the fitter runs it, so the test asserts
    both before differentiating anything.
    """
    fx, batch, model, fitter, config = bayesn_setup
    assert fitter.static_data['band_window'] is not None
    assert fitter.static_data["latetime_extrap"] is False

    t0 = (float(batch.peakmjd_offsets[0])
          if batch.peakmjd_offsets is not None else 0.0)
    z = jnp.asarray([float(batch.redshifts[0])])
    init = init_params(model, 1, jnp.asarray([t0]),
                       config.fit.distance_modulus(z))
    params = jnp.stack([jnp.asarray(init[n])[0]
                        for n in fitter.param_names]).astype(jnp.float64)
    assert_analytic_derivatives(one_sn_flux(model, fitter, batch, 0),
                                params, "BayeSN band flux")


def test_bayesn_closed_form_redshift_derivative(bayesn_setup):
    """The redshift direction of the analytic rule differentiates the flux.

    Redshift enters through the band weights as well as the phase, and the
    weights' own tangent is supplied to the rule rather than differentiated
    through, so this direction is carried by a separate term. A rule that
    dropped it would return a zero redshift gradient, which no fit result
    reports and no parity test sees.
    """
    fx, batch, model, fitter, config = bayesn_setup
    pz_fitter = build_fitter(
        fit_config(fx.root / "fit.nml", dofit_photoz=True), model, batch)
    assert fitter.static_data['band_window'] is not None

    t0 = (float(batch.peakmjd_offsets[0])
          if batch.peakmjd_offsets is not None else 0.0)
    z = jnp.asarray([float(batch.redshifts[0])])
    init = init_params(model, 1, jnp.asarray([t0]),
                       config.fit.distance_modulus(z))
    params = jnp.stack([jnp.asarray(init[n])[0]
                        for n in fitter.param_names]).astype(jnp.float64)
    v = jnp.concatenate([params, z.astype(jnp.float64)])
    assert_analytic_derivatives(
        one_sn_flux(model, pz_fitter, batch, 0, probe_z=True), v,
        "BayeSN band flux with redshift")


@pytest.mark.parametrize('host_law', ['FITZ99_EXACT', 'CCM89', 'SOMM25'])
def test_host_dust_flux(bayesn_setup, host_law):
    """BayeSN overrides ``evaluate`` with a closed-form band integral; the base
    implementation generates the SED per epoch and integrates it. Running both
    on one model instance checks the closed form's algebra, including the
    phase extrapolation the fitter uses before the first knot and after the
    last, where the warping is held at its edge value.
    """
    fx, batch, model, fitter, config = bayesn_setup
    photometry = fitter_photometry(fitter, batch)
    from jax_snlc_core.physics.dust_laws import load_dust_law
    law = load_dust_law(host_law)
    law.prepare_wavelength_grid(np.asarray(model.wavelength_grid))
    static_data = {**fitter.static_data, **photometry, 'host_dust_law': law}

    z = jnp.asarray(batch.redshifts)[:1]
    t0 = float(batch.peakmjd_offsets[0])
    init = init_params(model, 1, jnp.asarray([t0]),
                       config.fit.distance_modulus(z))
    params = jnp.stack([jnp.asarray(init[n])[0]
                        for n in fitter.param_names]).astype(jnp.float64)
    params = params.at[fitter.param_names.index('THETA')].set(1.2)

    # rest-frame phases before the first knot (-10 d), inside the knots,
    # between the last knot (40 d) and the end of its Hermite span (50 d),
    # and beyond it; the bands cycle through those the event was observed in
    phases = jnp.array([-18.0, -14.0, -11.0, 0.0, 20.0, 38.0,
                        43.0, 47.0, 55.0, 70.0, 80.0])
    mjds = t0 + phases * (1.0 + z[0])
    observed = jnp.asarray(batch.observation_mask[0])
    fi = jnp.resize(jnp.asarray(batch.filter_indices[0])[observed],
                    phases.shape)
    mask = jnp.ones(phases.shape, dtype=bool)

    bw = fitter.band_weights(
        photometry, z[0],
        jnp.ones(len(photometry['common_filter_wave_grid'])))[0]
    sed_params = model.sed_params_from_fit(params, z[0])
    flux_fast = model.evaluate(sed_params, mjds, fi, mask, bw, static_data)
    flux_slow = SNModel.evaluate(model, sed_params, mjds, fi, mask, bw,
                                 static_data)
    np.testing.assert_allclose(flux_fast, flux_slow, rtol=1e-12)
    import jax
    def flux(av):
        return model.evaluate({**sed_params, 'av': av}, mjds, fi, mask,
                              bw, static_data)
    av = jnp.asarray(.3)
    gradient = jax.jacfwd(flux)(av)
    finite_difference = (flux(av + 1e-5) - flux(av - 1e-5)) / 2e-5
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-7, atol=1e-9)



def test_lcplot_rebuilt_from_fitres(snana_fixture, tmp_path):
    """lcplot_from_fitres rebuilds the fit's own LCPLOT from its FITRES alone,
    Ds taken as MU_LCFIT. With one fit round the FITRES values are the
    round-1 values the fit's window was set at, so every flag matches.

    SNCID_LIST picks the events, and an event the FITRES does not hold is
    an error that names it."""
    from astropy.io import fits

    fx = snana_fixture("bayesn_fit_dense")
    overrides = ("SNTABLE_LIST", "FITRES LCPLOT", "MXLC_PLOT", "8",
                 "MAX_FIT", "8")
    fit_rows, rebuilt, fitres = regenerate_lcplot(fx.root, tmp_path, *overrides)
    assert all("MUERR_LCFIT" in row and "MU_LCFITERR" not in row
               for row in fitres.values())
    diff = compare_lcplots(fit_rows, rebuilt)
    assert diff["missing"] == 0 and diff["data_changed"] == 0, diff
    assert diff["curve"] < 2e-4 and diff["chi2"] < 0.011, diff
    assert diff["flags"] == [], diff

    cids = sorted({r["CID"] for r in fit_rows})[:2]
    picked = rebuild_lcplot(fx.root, tmp_path, "PICKED", *overrides,
                            "SNCID_LIST", *cids)
    assert picked == {k: r for k, r in rebuilt.items() if k[0] in cids}

    with fits.open(fx.head_fits) as hdul:
        absent = next(str(int(s)) for s in hdul[1].data["SNID"]
                      if int(s) not in fitres)
    with pytest.raises((SNANAError, KeyError), match=absent):
        rebuild_lcplot(fx.root, tmp_path, "ABSENT", *overrides,
                       "SNCID_LIST", absent)


def test_fit_band_support(bayesn_setup):
    _, _, model, fitter, _ = bayesn_setup
    # 3500 A lies inside W22's SED but outside its allowed band centres;
    # 9000 A is allowed by W22 but outside this fit's requested window.
    assert model.get_wavelength_range()[0] < 3500.
    photometry = {'filter_mean_wavelengths': jnp.array([3500., 4500., 9000.]) * 1.1}
    mask = fitter.restlambda_mask(photometry, jnp.array([[0, 1, 2]]), jnp.array([.1]))
    np.testing.assert_array_equal(mask, [[False, True, False]])
