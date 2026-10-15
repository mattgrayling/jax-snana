"""The StandardisedModel role on the model classes: what BBC asks of a model."""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_snlc_core.models import (BayeSNModel, SALTModel, StandardisedModel,
                                  model_input_keys, standardised_model)
from jax_snlc_core.utils.error_handling import SNANAError

_NUIS = {"alpha0": 0.14, "dalpha_dz": 0.0, "beta0": 3.2, "dbeta_dz": 0.0,
         "gamma0": 0.0, "dgamma_dz": 0.0, "logmass_cen": 10.0, "logmass_tau": 0.01,
         "dalpha_dlogm": 0.0, "dbeta_dlogm": 0.0}


def test_model_lcfit_resolves_to_a_model_playing_the_role():
    assert standardised_model("SALT2") is SALTModel
    assert standardised_model("BAYESN") is BayeSNModel
    assert issubclass(SALTModel, StandardisedModel)
    with pytest.raises(SNANAError):
        standardised_model("REDBACK")


class _Own(StandardisedModel):
    """A model whose LC parameters are its own fit parameters declares no
    FITRES aliases and no reader: both follow from lcpar_names."""
    m0_reference = 0.0
    lcpar_names = ("MAG", "TAU")
    nuisance_spec = {}
    biascor_binning = {"MAG": (-1.0, 1.0, 0.5), "TAU": (0.0, 2.0, 0.5)}

    @classmethod
    def compute_mu(cls, lcpar, nuis, host, u_flags, apply_mass_step=True):
        return lcpar["MAG"]

    @classmethod
    def sim_lcpar(cls, tv):
        return {}


def test_a_model_without_aliases_reads_the_fitter_s_own_columns():
    assert _Own.fitres_aliases == {
        "MAG": ("MAG",), "MAGERR": ("MAGERR",),
        "TAU": ("TAU",), "TAUERR": ("TAUERR",),
        "COV_MAG_TAU": ("COV_MAG_TAU", "COV_TAU_MAG")}
    # a declared alias table is left alone
    assert "x0" in SALTModel.fitres_aliases and "MU" in BayeSNModel.fitres_aliases
    cols = {"MAG": jnp.array([15.0, 16.0]), "MAGERR": jnp.array([0.1, 0.2]),
            "TAU": jnp.array([1.0, 1.5]), "TAUERR": jnp.array([0.3, 0.4]),
            "COV_MAG_TAU": jnp.array([0.01, -0.02])}
    lcpar, cov, bad = _Own.read_lcpar(cols, {})
    np.testing.assert_array_equal(np.asarray(lcpar["TAU"]), [1.0, 1.5])
    np.testing.assert_allclose(np.asarray(cov),
                               [[[0.01, 0.01], [0.01, 0.09]],
                                [[0.04, -0.02], [-0.02, 0.16]]])
    assert bad is None


def test_compute_mu_salt_and_bayesn():
    lcpar = {"mB": jnp.array([15.0]), "x1": jnp.array([1.0]), "c": jnp.array([0.1])}
    host = {"z": jnp.array([0.1]), "gamma_var": jnp.array([11.0]),   # high mass
            "gammadm_offset": 0.0, "use_gamma0": True}

    # No mass step (gamma0=0): mu = 15 + 0.14 - 0.32 = 14.82.
    mu = SALTModel.compute_mu(lcpar, dict(_NUIS), host, {})
    np.testing.assert_allclose(np.asarray(mu), [14.82], rtol=1e-6)

    # With gamma0=0.06 the high-mass step brightens by +0.03; apply_mass_step=False
    # drops it, and gamma0=0 makes the two paths exactly equal (parity safety).
    with_step = SALTModel.compute_mu(lcpar, {**_NUIS, "gamma0": 0.06}, host, {})
    without_step = SALTModel.compute_mu(lcpar, {**_NUIS, "gamma0": 0.06}, host, {},
                                        apply_mass_step=False)
    np.testing.assert_allclose(np.asarray(with_step), [14.85], atol=1e-4)
    np.testing.assert_allclose(np.asarray(without_step), [14.82], atol=1e-6)
    zero_on = SALTModel.compute_mu(lcpar, dict(_NUIS), host, {})
    zero_off = SALTModel.compute_mu(lcpar, dict(_NUIS), host, {}, apply_mass_step=False)
    np.testing.assert_array_equal(np.asarray(zero_on), np.asarray(zero_off))

    # GAMMADM_OFFSET, the midpoint of the biasCor SIM_gammaDM range, is
    # subtracted from the step (SALT2mu.c:16782), so mu gains it back.
    offset_host = {**host, "gammadm_offset": 0.02}
    shifted = SALTModel.compute_mu(lcpar, {**_NUIS, "gamma0": 0.06},
                                   offset_host, {})
    np.testing.assert_allclose(np.asarray(shifted),
                               np.asarray(with_step) + 0.02, atol=1e-6)

    # BayeSN reads mu directly.
    mu_b = BayeSNModel.compute_mu({"MU": jnp.array([39.5])}, {}, {}, {})
    np.testing.assert_array_equal(np.asarray(mu_b), [39.5])


def test_use_gamma0_shuts_the_mass_step_before_it_divides():
    """get_gammadm_host returns zero before reading dgamma_dz or logmass_tau when
    USE_GAMMA0 is off (SALT2mu.c:16776). SNANA's parval holds logmass_tau at 0
    unless p8 is given, so an ungated step divides by zero there."""
    lcpar = {"mB": jnp.array([15.0]), "x1": jnp.array([1.0]), "c": jnp.array([0.1])}
    nuis = {**_NUIS, "gamma0": 0.0, "dgamma_dz": 0.05, "logmass_tau": 0.0}
    base = {"z": np.array([0.1]), "gamma_var": np.array([11.0]),
            "gammadm_offset": 0.02}

    with np.errstate(divide="raise", invalid="raise"):
        mu = SALTModel.compute_mu(lcpar, nuis, {**base, "use_gamma0": False}, {})
        _, _, gdm = SALTModel.biascor_coeff_values(
            nuis, {**base, "use_gamma0": False}, {})

    lc_only = SALTModel.compute_mu(lcpar, nuis, {**base, "use_gamma0": False},
                                   {}, apply_mass_step=False)
    np.testing.assert_array_equal(np.asarray(mu), np.asarray(lc_only))
    assert gdm == 0.0

    # the term the gate suppresses is not itself zero
    open_gate = {**base, "use_gamma0": True, "gamma_var": jnp.array([11.0])}
    _, _, live = SALTModel.biascor_coeff_values({**nuis, "logmass_tau": 0.01},
                                                open_gate, {})
    assert abs(float(np.asarray(live)[0])) > 1.0e-3


def test_salt_host_mass_flags():
    """SNANA's independent-if quirk: u15=2 turns the split on while u16=0 keeps
    the slope on, so both terms fire (fcnFetch_AlphaBetaGamma)."""
    lcpar = {"mB": jnp.array([15.0, 15.0]), "x1": jnp.array([1.0, 1.0]),
             "c": jnp.array([0.0, 0.0])}
    host = {"z": jnp.array([0.1, 0.1]), "gamma_var": jnp.array([11.0, 9.0]),
            "gammadm_offset": 0.0, "use_gamma0": True}
    nuis = {**_NUIS, "dalpha_dlogm": 0.02}
    plain = np.asarray(SALTModel.compute_mu(lcpar, nuis, host, {}))
    np.testing.assert_allclose(plain, 15.14 + 0.02 * np.array([1.0, -1.0]))   # slope
    both = np.asarray(SALTModel.compute_mu(lcpar, nuis, host, {"u15": 2}))
    np.testing.assert_allclose(both, plain + 0.02 * np.array([0.5, -0.5]))   # + split


def test_model_declarations():
    assert (SALTModel.m0_reference, BayeSNModel.m0_reference) == (-30.0, 0.0)
    # Cut shorthands are model-specific and disjoint.
    assert SALTModel.cut_aliases["x1min"] == ("x1", "lo")
    assert BayeSNModel.cut_aliases["thetamax"] == ("THETA", "hi")
    assert "x1min" not in BayeSNModel.cut_aliases
    # the nuisance slots are SNANA's p<N> numbers
    assert SALTModel.nuisance_spec["alpha0"][3] == 1
    assert SALTModel.nuisance_spec["dbeta_dlogm"][3] == 16
    # and the bounds are set_fitPar's (SALT2mu.c:22354-22369)
    assert {name: spec[1:3] for name, spec in SALTModel.nuisance_spec.items()} == {
        "alpha0": (0.02, 0.30), "beta0": (1.0, 6.0),
        "dalpha_dz": (-0.5, 0.5), "dbeta_dz": (-3.0, 3.0),
        "gamma0": (-0.5, 0.5), "dgamma_dz": (-2.0, 2.0),
        "logmass_cen": (9.0, 11.0), "logmass_tau": (8.0e-4, 1.0),
        "dalpha_dlogm": (-0.5, 0.5), "dbeta_dlogm": (-3.0, 3.0)}
    # the model's own BBC keys, offered to BBC's input alone
    assert sorted(model_input_keys(SALTModel, consumer='bbc')) == [
        'MAXERR_ABORT_C', 'MAXERR_ABORT_X0', 'MAXERR_ABORT_X1']
    assert 'MAXERR_ABORT_X0' not in model_input_keys(SALTModel, consumer='fit')
    assert model_input_keys(BayeSNModel, consumer='bbc') == {}


def test_read_lcpar_and_covariance():
    cols = {"x0": jnp.array([1e-5, 2e-5]), "x0err": jnp.array([1e-7, 2e-7]),
            "x1": jnp.array([0.5, -1.0]), "x1err": jnp.array([0.2, 0.3]),
            "c": jnp.array([0.1, -0.05]), "cerr": jnp.array([0.02, 0.03]),
            "COV_x1_x0": jnp.array([1e-9, 2e-9]), "COV_x1_c": jnp.array([1e-4, 2e-4]),
            "COV_c_x0": jnp.array([1e-10, 2e-10])}
    inputs = {"maxerr_abort_x0": 0.0, "maxerr_abort_x1": 0.25, "maxerr_abort_c": 0.0}
    lcpar, cov, bad = SALTModel.read_lcpar(cols, inputs)
    # mB = -2.5*log10(x0), offset-free; x1/c read directly.
    np.testing.assert_allclose(np.asarray(lcpar["mB"]), -2.5 * np.log10([1e-5, 2e-5]))
    np.testing.assert_array_equal(np.asarray(lcpar["x1"]), [0.5, -1.0])

    cov = np.asarray(cov)
    assert cov.shape == (2, 3, 3)
    np.testing.assert_allclose(cov, np.transpose(cov, (0, 2, 1)))   # symmetric
    np.testing.assert_allclose(cov[:, 1, 1], [0.04, 0.09])          # x1,x1 = x1err^2
    # an error at or below its threshold is bad (SALT2mu.c:18885)
    np.testing.assert_array_equal(np.asarray(bad), [True, False])

    # read_lcpar returns the distance MU plus the THETA/AV biasCor cell dimensions.
    lc_b, cov_b, bad_b = BayeSNModel.read_lcpar(
        {"MU": jnp.array([39.5, 40.0]), "MUERR": jnp.array([0.1, 0.2]),
         "THETA": jnp.array([0.3, 0.0]), "AV": jnp.array([0.1, 0.2])}, {})
    np.testing.assert_array_equal(np.asarray(lc_b["MU"]), [39.5, 40.0])
    np.testing.assert_array_equal(np.asarray(lc_b["THETA"]), [0.3, 0.0])
    cov_b = np.asarray(cov_b)
    assert cov_b.shape == (2, 1, 1)
    np.testing.assert_allclose(cov_b[:, 0, 0], [0.01, 0.04])
    assert bad_b is None


def test_compute_mu_refuses_a_host_without_the_mass_step_settings():
    """BBC supplies GAMMADM_OFFSET and USE_GAMMA0 with the host properties, so
    a caller that has not been updated for them aborts here rather than
    reaching the mass step with the wrong zero point or an unwanted step
    (SALT2mu.c:16228, :16776, :16782)."""
    lcpar = {"mB": jnp.array([15.0]), "x1": jnp.array([1.0]), "c": jnp.array([0.1])}
    host = {"z": jnp.array([0.1]), "gamma_var": jnp.array([11.0])}
    for call in (lambda: SALTModel.compute_mu(lcpar, dict(_NUIS), host, {}),
                 lambda: SALTModel.biascor_coeff_values(dict(_NUIS), host, {})):
        with pytest.raises(SNANAError, match="gammadm_offset.*use_gamma0"):
            call()
    # apply_mass_step=False never reaches the term, so it needs neither
    SALTModel.compute_mu(lcpar, dict(_NUIS), host, {}, apply_mass_step=False)
    # SALT is the model with a mass step; BayeSN declares none
    assert SALTModel.gammadm_coeff == "gamma0"
    assert BayeSNModel.gammadm_coeff is None
