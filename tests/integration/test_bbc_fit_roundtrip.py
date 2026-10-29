"""The fit with no SNANA reference: synthetic round-trip, then joint vs secant.

The first half injects alpha/beta/M0/sigma_int into clean events and recovers
them, validating the standardiser mu, the mu-error projection, the likelihood and
the Newton-LM fit with no biasCor, muCOVscale or CC prior.

The second half runs both sigint modes through the full driver on the DES fixture.
jax's joint mode has no SNANA equivalent, so agreement with the ported secant is
the only evidence available for it -- there is no reference file to diff against.
"""


import jax.numpy as jnp
import numpy as np
import pytest

from jax_salt2mu import fitter
from jax_salt2mu.config import BBCConfig
from jax_salt2mu.fitter import FitSpec, run_fit
from jax_salt2mu.likelihood import LikelihoodInputs
from jax_salt2mu.run import run_bbc_fit, run_bbc_grid
from jax_snlc_core.models import SALTModel

_ALPHA, _BETA, _SIGINT = 0.14, 3.1, 0.10
_NZBIN = 5


def _synthetic(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.05, 0.5, n)
    x1 = rng.normal(0.0, 1.0, n)
    c = rng.normal(0.0, 0.1, n)
    izbin = np.clip((z / 0.5 * _NZBIN).astype(int), 0, _NZBIN - 1)
    m0_true = np.array([-0.02, 0.0, 0.03, -0.01, 0.02])
    mumodel = 35.0 + 10.0 * z

    mBerr, x1err, cerr = 0.02, 0.1, 0.02
    muerrsq_meas = mBerr**2 + (_ALPHA * x1err)**2 + (_BETA * cerr)**2
    scatter = rng.normal(0.0, np.sqrt(_SIGINT**2 + muerrsq_meas), n)
    mu_obs = m0_true[izbin] + mumodel + scatter
    mB = mu_obs - _ALPHA * x1 + _BETA * c   # so mu = mB + a*x1 - b*c = mu_obs

    n_ev = n
    cov = np.zeros((n_ev, 3, 3))
    cov[:, 0, 0] = mBerr**2
    cov[:, 1, 1] = x1err**2
    cov[:, 2, 2] = cerr**2

    inputs = LikelihoodInputs(
        lcpar={"mB": jnp.array(mB), "x1": jnp.array(x1), "c": jnp.array(c)},
        z=jnp.array(z), logmass=jnp.full(n, 10.0),
        covmat_fit=jnp.array(cov), mumodel=jnp.array(mumodel),
        izbin=jnp.array(izbin), muerr_vpec_sq=jnp.zeros(n), valid=jnp.ones(n))
    return inputs, m0_true


def _salt_defaults():
    return {"alpha0": _ALPHA, "dalpha_dz": 0.0, "beta0": _BETA, "dbeta_dz": 0.0,
            "gamma0": 0.0, "dgamma_dz": 0.0, "logmass_cen": 10.0, "logmass_tau": 0.01,
            "dalpha_dlogm": 0.0, "dbeta_dlogm": 0.0}


def test_joint_mode_recovers_truth(tmp_path):
    inputs, m0_true = _synthetic()
    # start displaced from truth so recovery is meaningful
    defaults = dict(_salt_defaults(), alpha0=0.10, beta0=2.8)
    spec = FitSpec(SALTModel, ("alpha0", "beta0"), tuple(sorted(defaults.items())),
                   _NZBIN, "joint", False)
    res = run_fit(spec, inputs, m0_reference=0.0,
                  n_pad=inputs.z.shape[0], sigint_init=0.15)
    assert res.nuisance["alpha0"] == pytest.approx(_ALPHA, abs=0.01)
    assert res.nuisance["beta0"] == pytest.approx(_BETA, abs=0.05)
    assert res.sigint == pytest.approx(_SIGINT, abs=0.02)
    np.testing.assert_allclose(res.m0, m0_true, atol=0.02)
    from jax_salt2mu.output import write_chi2grid, write_yaml
    from jax_salt2mu.output import YamlSummary
    from jax_snlc_core.snana_files import read_fitres
    import yaml

    fitted = jnp.array([res.nuisance[n] for n in spec.floated]
                      + list(res.m0) + [res.sigint])
    assert res.ndof_ia == len(inputs.z) - len(spec.floated) - _NZBIN - 1
    assert res.chi2sum_ia > 0
    assert res.chi2 == pytest.approx(float(spec.loss(inputs)(fitted)), rel=1e-10)
    # Retaining the gradient from the Hessian pass must preserve the covariance.
    import jax
    gradient, hessian = fitter._fit_derivatives(spec, fitted, inputs, 0., None)
    reference = jax.hessian(spec.loss(inputs))(fitted)
    np.testing.assert_allclose(hessian, reference, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(gradient, jax.grad(spec.loss(inputs))(fitted),
                               rtol=1e-12, atol=1e-10)
    assert res.max_gradient == pytest.approx(float(np.max(np.abs(gradient))))
    summary_path = tmp_path / "fit.YAML"
    write_yaml(summary_path, YamlSummary(), res, BBCConfig())
    summary = yaml.safe_load(summary_path.read_text())
    np.testing.assert_allclose(summary['FIT_DIAGNOSTICS']['max_gradient'],
                               res.max_gradient, rtol=1e-7)
    assert np.isfinite(res.max_projected_gradient)
    assert 0 <= res.max_projected_gradient <= res.max_gradient
    reduced = next(row["CHI2RED_IA"] for row in summary["BBCFIT_RESULTS"]
                   if "CHI2RED_IA" in row)
    assert float(str(reduced).split()[0]) == pytest.approx(
        res.chi2sum_ia / res.ndof_ia, abs=.0005)

    grid_path = tmp_path / "fit.CHI2GRID"
    write_chi2grid(grid_path, spec, inputs, res, 2)
    grid = read_fitres(str(grid_path)).df
    for label, name, _ in SALTModel.chi2grid_axes:
        for _, row in grid.iterrows():
            point = fitted.at[spec.floated.index(name)].set(row[label])
            assert row[f"chi2_{label}"] == pytest.approx(
                float(spec.loss(inputs)(point)), rel=1e-6)


def test_secant_mode_recovers_sigint():
    inputs, _ = _synthetic(seed=1)
    defaults = dict(_salt_defaults(), alpha0=0.12, beta0=3.0)
    spec = FitSpec(SALTModel, ("alpha0", "beta0"), tuple(sorted(defaults.items())),
                   _NZBIN, "snana_secant", False)
    res = run_fit(spec, inputs, m0_reference=0.0,
                  n_pad=inputs.z.shape[0], sigint_init=0.15)
    # secant drives chi2/dof -> 1, recovering the injected scatter
    assert res.sigint == pytest.approx(_SIGINT, abs=0.03)
    assert res.nuisance["alpha0"] == pytest.approx(_ALPHA, abs=0.015)


_DES = "tests/fixtures/snana/bbc_des_beams"
_DES_GROUP = "DES(zbin=0.15:x1bin=2.0:cbin=0.2)"


def _des_cfg(sigint_mode):
    cfg = BBCConfig(
        datafile=[f"{_DES}/DATA.FITRES"], nzbin=10, zmin=0.1, zmax=1.0,
        simfile_biascor=[f"{_DES}/BIASCOR.FITRES"], opt_biascor=112,
        min_per_cell_biascor=3, snrmin_sigint_biascor=60.0,
        surveygroup_biascor=_DES_GROUP, fitflag_sigmb=1,
        # this sample's secant oscillates into the sigint=0 floor unless the slope
        # is fixed, which is what SNANA offers dchi2red_dsigint for
        dchi2red_dsigint=-5.0, redchi2_tol=0.002)
    cfg.sigint_mode = sigint_mode
    cfg.nuisance_float["alpha0"] = True
    cfg.nuisance_float["beta0"] = True
    cfg.nuisance_onebin.update(alpha0=False, beta0=False)
    return cfg


def _run_both():
    out = {}
    for mode in ("snana_secant", "joint"):
        result, output, _, _, _ = run_bbc_fit(_des_cfg(mode))
        out[mode] = (result, output)
    return out["snana_secant"], out["joint"]


def test_joint_and_secant_agree_on_untruncated_data():
    """Driving chi2/dof to 1 and maximising the Gaussian likelihood are different
    estimators, so they agree only approximately -- 7% here -- but should not drift.
    Untruncated is the qualifier: under chi2max they split by 37%, which resolve()
    warns about."""
    (sec, sec_out), (joint, joint_out) = _run_both()
    assert sec_out.config.radiation and joint_out.config.radiation
    assert sec.sigint == pytest.approx(joint.sigint, rel=0.15)
    assert sec.nuisance["alpha0"] == pytest.approx(joint.nuisance["alpha0"], abs=0.005)
    assert sec.nuisance["beta0"] == pytest.approx(joint.nuisance["beta0"], abs=0.05)
    assert np.max(np.abs(sec_out.m0dif - joint_out.m0dif)) < 2e-3


def test_grid_matches_independent_fits_and_compiles_once():
    """The cells of a systematics grid differ in event count. Fitted as a
    grid they are padded to the largest and share one compiled fit, and each
    gets what its lone fit gives, to the float32 reduction order over the
    padded sum."""
    narrow, wide = _des_cfg("snana_secant"), _des_cfg("snana_secant")
    narrow.cutwin = [("c", -0.25, 0.25)]
    wide.cutwin = [("x1", -2.5, 2.5)]
    alone = {key: run_bbc_fit(cfg) for key, cfg in (("narrow", narrow), ("wide", wide))}
    assert alone["narrow"][0].nsnfit != alone["wide"][0].nsnfit

    fitter._fit.clear_cache()
    grid = run_bbc_grid({"narrow": (narrow, None), "wide": (wide, None)})
    assert fitter._fit._cache_size() == 1

    for key, (result, output, *_) in grid.items():
        lone_result, lone_output, *_ = alone[key]
        assert result.nsnfit == lone_result.nsnfit
        for name in ("alpha0", "beta0"):
            assert result.nuisance[name] == pytest.approx(lone_result.nuisance[name], abs=1e-5)
        assert result.sigint == pytest.approx(lone_result.sigint, abs=1e-5)
        np.testing.assert_allclose(result.m0, lone_result.m0, atol=1e-4)
        np.testing.assert_allclose(output.mu, lone_output.mu, atol=1e-4)
        np.testing.assert_allclose(output.muerr, lone_output.muerr, atol=1e-5)


@pytest.mark.parametrize("prior", ["cc", "h11"])
def test_prior_inputs_reuse_compilation(prior):
    from dataclasses import replace

    import jax

    from jax_salt2mu.fitter import padded
    from _bbc_helpers import prepare_cell

    cfg = _des_cfg("snana_secant")
    cfg.datafile = [f"{_DES}/CONTAM_DATA.FITRES"]
    cfg.varname_pia, cfg.typeIa_ccprior = "PROBIA", 999
    cfg.blindflag = 0
    cfg.nuisance_val["scalePCC"] = 1.0
    if prior == "cc":
        cfg.simfile_ccprior = [f"{_DES}/CCPRIOR.FITRES"]
    else:
        cfg.use_h11 = True
    cell = prepare_cell(cfg)
    spec, inputs = cell.spec, cell.inputs
    defaults = dict(spec.defaults)
    p = jnp.array([defaults[n] for n in spec.floated]
                  + [spec.model.m0_reference] * spec.nzbin)

    def evaluate(spec, inputs):
        return jax.value_and_grad(spec.closed(inputs, cfg.sigint, inputs.muerrsq_last))(p)

    compiled = jax.jit(evaluate, static_argnames="spec")
    original = compiled(spec, inputs)
    inp = padded(inputs, len(inputs.z) + 13)
    value, gradient = compiled(spec, inp)
    np.testing.assert_allclose(value, original[0], rtol=1e-12)
    np.testing.assert_allclose(gradient, original[1], rtol=1e-10, atol=1e-8)
    if prior == "cc":
        np.testing.assert_array_equal(inp.ccprior.cc_z, inputs.ccprior.cc_z)
        assert isinstance(inp.ccprior.cc_z, jax.Array)
        changed = inp._replace(ccprior=inp.ccprior._replace(
            cc_mumodel=inp.ccprior.cc_mumodel + .1))
    else:
        changed = inp._replace(h11prior=inp.h11prior._replace(
            muerrsq_raw=inp.h11prior.muerrsq_raw * 1.2))
    changed = changed._replace(lensing_zpar=inp.lensing_zpar + .01)
    cache_size = compiled._cache_size()
    updated = compiled(spec, changed)
    assert compiled._cache_size() == cache_size
    assert not np.isclose(updated[0], value)
    for actual, expected in zip(updated, evaluate(spec, changed)):
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-8)

    other = replace(spec, with_log=not spec.with_log)
    for actual, expected in zip(compiled(other, changed), evaluate(other, changed)):
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-8)
    assert compiled._cache_size() == cache_size + 1
    _, hessian = fitter._fit_derivatives(spec, p, inp, cfg.sigint, inp.muerrsq_last)
    assert np.all(np.isfinite(hessian))


@pytest.mark.parametrize("mode, stop", [("joint", False),
                                        ("snana_secant", True),
                                        ("snana_secant", False)])
def test_final_variance_reference(mode, stop):
    """Stored variance references reproduce the fitted likelihood and final errors."""
    from _bbc_helpers import prepare_cell

    cell = prepare_cell(_des_cfg(mode))
    inputs = cell.inputs
    initial = np.array(inputs.muerrsq_last)
    n = len(inputs.z)
    result = run_fit(cell.spec, inputs, cell.model.m0_reference, n + 13,
                     sigint_init=cell.config.sigint, stop_covfix=stop,
                     dchi2red_dsigint=cell.config.dchi2red_dsigint,
                     redchi2_tol=cell.config.redchi2_tol)
    np.testing.assert_array_equal(inputs.muerrsq_last, initial)
    if mode == "joint" or stop:
        np.testing.assert_array_equal(result.muerrsq_last, initial)
        assert result.niter == 1
    else:
        assert result.niter > 1
        assert not np.array_equal(result.muerrsq_last, result.muerrsq_final)
    p = jnp.array([result.nuisance[name] for name in cell.spec.floated]
                  + list(result.m0))
    final = fitter._muerrsq(cell.spec, p, inputs, result.sigint)
    np.testing.assert_allclose(result.muerrsq_final, final, rtol=1e-12)
    fitted = jnp.concatenate((p, jnp.array([result.sigint]))) if mode == "joint" else p
    objective = cell.spec.closed(inputs, result.sigint, result.muerrsq_last)(fitted)
    assert result.chi2 == pytest.approx(float(objective), rel=1e-10)


def test_cc_padding_preserves_likelihood_derivatives_and_reuses_compilation():
    from jax_salt2mu.fitter import _fit_derivatives, padded
    from jax_salt2mu.ccprior import pad_ccprior
    from _bbc_helpers import prepare_cell

    cfg = _des_cfg('snana_secant')
    cfg.datafile = [f'{_DES}/CONTAM_DATA.FITRES']
    cfg.simfile_ccprior = [f'{_DES}/CCPRIOR.FITRES']
    cfg.varname_pia, cfg.typeIa_ccprior = 'PROBIA', 999
    cfg.blindflag = 0
    cfg.nuisance_val['scalePCC'] = 1.0
    cell = prepare_cell(cfg)
    inp = padded(cell.inputs, len(cell.inputs.z))
    prior = inp.ccprior
    defaults = dict(cell.spec.defaults)
    p = jnp.array([defaults[n] for n in cell.spec.floated]
                  + [cell.model.m0_reference] * cell.spec.nzbin)

    def derivatives(inputs):
        return _fit_derivatives(cell.spec, p, inputs, cfg.sigint, inputs.muerrsq_last)

    width = len(prior.cc_z) + 3
    for removed in (0, 2):
        n = len(prior.cc_z) - removed
        trimmed = prior._replace(
            cc_lcpar={k: v[:n] for k, v in prior.cc_lcpar.items()},
            **{name: getattr(prior, name)[:n] for name in
               ('cc_z', 'cc_logmass', 'cc_mumodel', 'cc_fitparbias_nodes',
                'cc_cell', 'cc_valid')})
        original = inp._replace(ccprior=trimmed)
        expected = derivatives(original)
        compiled = _fit_derivatives._cache_size()
        padded_prior = pad_ccprior(trimmed, width)
        result = derivatives(inp._replace(ccprior=padded_prior))
        if removed:
            assert _fit_derivatives._cache_size() == compiled
        for a, b in zip(result, expected):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-8)
        np.testing.assert_array_equal(padded_prior.cc_valid[n:], 0.)
        assert padded_prior.data_cell is prior.data_cell
        assert padded_prior.pIa is prior.pIa
        assert len(padded_prior.cc_z) == width
