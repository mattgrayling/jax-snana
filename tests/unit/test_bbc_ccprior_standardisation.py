"""The CC map uses global SALT coefficients; data retain their local ones."""

from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_salt2mu.ccprior import CCPriorInputs, PIFAC
from jax_salt2mu.fitter import FitSpec
from jax_salt2mu.likelihood import _ccprior_dPdmu
from jax_snlc_core.models import SALTModel


def _prior(on_mu):
    n = 16
    axes = ([.1, .2], [2., 4.], [-.2, .2])
    nodes = np.array(list(product(*axes)))
    a, b, g = nodes.T
    bias = ((.02 + .2*a + .01*b + .3*g)[:, None] if on_mu else
            np.column_stack((.02 + .2*a, .04 + .01*b, .01 + .3*g)))
    cc = CCPriorInputs(
        cc_lcpar={'mB': jnp.linspace(-.1, .6, n),
                  'x1': jnp.sin(jnp.arange(n)),
                  'c': jnp.linspace(-.08, .1, n)},
        cc_z=jnp.linspace(.1, .9, n), cc_logmass=jnp.linspace(9., 11., n),
        cc_mumodel=jnp.zeros(n), cc_fitparbias_nodes=jnp.tile(bias[None], (n, 1, 1)),
        cc_cell=jnp.zeros(n, dtype=int), cc_valid=jnp.ones(n),
        data_cell=jnp.zeros(2, dtype=int), pIa=jnp.full(2, .8),
        coeff_avg=tuple(jnp.array(axis) for axis in axes),
        coeff_binsize=(.1, 2., .4), m0_default=0.)
    nuisance = {name: values[0] for name, values in SALTModel.nuisance_spec.items()}
    nuisance.update(alpha0=.14, beta0=3.1, dalpha_dz=.1, dbeta_dz=.4,
                    dalpha_dlogm=.05, dbeta_dlogm=.2, gamma0=.08,
                    dgamma_dz=.03, logmass_cen=10., logmass_tau=.1)
    return cc, nuisance


@pytest.mark.parametrize('on_mu', [False, True])
@pytest.mark.parametrize('flags', [{}, {'u15': 1, 'u16': 1}, {'u15': 2, 'u16': 2},
                                   {'u15': 2, 'u16': 1}])
def test_cc_density_uses_alpha0_beta0_and_zero_gamma(on_mu, flags):
    cc, nuisance = _prior(on_mu)
    spec = FitSpec(SALTModel, (), (), 1, 'snana_secant', False,
                   u_flags=tuple(flags.items()), cc_ncell=1,
                   cc_bias_names=('mu',) if on_mu else SALTModel.lcpar_names)
    residual = jnp.array([.05, .25])
    actual = _ccprior_dPdmu(spec, cc, nuisance, residual)

    # Independent analytic calculation: linearly interpolated node biases at
    # alpha0=.14, beta0=3.1, gammaDM=0, then projected with (1, alpha0, -beta0).
    a, b = .14, 3.1
    bias = .02 + .2*a + .01*b if on_mu else .02 + .2*a + a*(.04 + .01*b) - b*.01
    distances = np.asarray(cc.cc_lcpar['mB'] + a*cc.cc_lcpar['x1'] - b*cc.cc_lcpar['c']) - bias
    mean, sigma = distances.mean(), distances.std()
    expected = PIFAC/sigma * np.exp(-.5*((np.asarray(residual)-mean)/sigma)**2)
    np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def density(parameters):
        return _ccprior_dPdmu(spec, cc, parameters, residual).sum()

    gradient = jax.grad(density)(nuisance)
    for name in ('dalpha_dz', 'dbeta_dz', 'dalpha_dlogm', 'dbeta_dlogm', 'gamma0', 'dgamma_dz'):
        assert gradient[name] == 0.
    for name in ('alpha0', 'beta0'):
        step = 1e-5
        finite = (density({**nuisance, name: nuisance[name]+step})
                  - density({**nuisance, name: nuisance[name]-step}))/(2*step)
        assert abs(float(gradient[name])) > 1e-5
        np.testing.assert_allclose(gradient[name], finite, rtol=1e-7)

    # The CC restriction must not mutate the parameters used by data events.
    host = {'z': cc.cc_z, 'gamma_var': cc.cc_logmass,
            'gammadm_offset': .03, 'use_gamma0': True}
    def data_distance(parameters):
        return SALTModel.compute_mu(cc.cc_lcpar, parameters, host, flags)

    data_gradient = jax.jacfwd(data_distance)(nuisance)
    for name in ('dalpha_dz', 'dbeta_dz', 'dalpha_dlogm', 'dbeta_dlogm', 'gamma0', 'dgamma_dz'):
        assert np.any(np.abs(data_gradient[name]) > 1e-5)
