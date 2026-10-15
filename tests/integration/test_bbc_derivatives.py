"""Compare structured curvature with dense AD on the real BBC likelihoods."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_salt2mu.config import BBCConfig
from jax_salt2mu.derivatives import loss_derivatives
from jax_salt2mu.fitter import padded
from jax_salt2mu.run import prepare_cell
from jax_snlc_core.utils.newton_lm import _grad_and_hess


@pytest.fixture(scope='module', params=['salt', 'sim_prior', 'h11', 'bayesn'])
def cell(request):
    root = 'tests/fixtures/snana/bbc_des_beams'
    if request.param == 'bayesn':
        cfg = BBCConfig(
            datafile=['tests/fixtures/snana/bbc_des_bayesn_data/FITRES.TEXT'],
            simfile_biascor=['tests/fixtures/snana/bbc_des_bayesn_biascor/FITRES.TEXT.gz'],
            model_lcfit='BAYESN', nzbin=4, zmin=.05, zmax=.21,
            opt_biascor=176, sigint_biascor=.1, nbin_logmass=1, snrmin_sigint_biascor=6.)
    else:
        cfg = BBCConfig(
            datafile=[f'{root}/CONTAM_DATA.FITRES'],
            simfile_biascor=[f'{root}/BIASCOR.FITRES'],
            nzbin=10, zmin=.1, zmax=1., opt_biascor=112,
            min_per_cell_biascor=3, snrmin_sigint_biascor=60.,
            surveygroup_biascor='DES(zbin=0.15:x1bin=2.0:cbin=0.2)',
            blindflag=0, nuisance_float={'alpha0': True, 'beta0': True},
            nuisance_onebin={'alpha0': False, 'beta0': False})
        if request.param != 'salt':
            cfg.varname_pia, cfg.typeIa_ccprior = 'PROBIA', 999
            cfg.nuisance_val['scalePCC'] = 1.
            if request.param == 'sim_prior':
                cfg.simfile_ccprior = [f'{root}/CCPRIOR.FITRES']
            else:
                cfg.use_h11 = True
    return prepare_cell(cfg)


@pytest.mark.parametrize('joint', [False, True])
@pytest.mark.parametrize('interpolate', [False, True])
def test_dense_and_structured_derivatives(cell, joint, interpolate):
    inp = padded(cell.inputs, len(cell.inputs.z) + 7)
    spec = replace(cell.spec, mode='joint' if joint else 'snana_secant')
    if interpolate:
        neighbour = jnp.clip(inp.izbin + jnp.where(jnp.arange(len(inp.z)) % 2, 1, -1),
                             0, spec.nzbin - 1)
        inp = inp._replace(izbin_nbr=neighbour, m0_zfrac=jnp.linspace(-.1, 1.1, len(inp.z)))
    nf = len(spec.floated)
    p = jnp.array([dict(spec.defaults)[n] for n in spec.floated]
                  + [cell.model.m0_reference] * spec.nzbin + ([.1] if joint else []))

    @jax.jit
    def compare(point, inputs):
        loss = spec.closed(inputs, .1, inputs.muerrsq_last)
        return (_grad_and_hess(point, loss),
                loss_derivatives(point, loss, n_nuisance=nf, nzbin=spec.nzbin,
                                 interpolate=interpolate))

    for offset in (0., .003, -.002):
        expected, actual = compare(p + offset * jnp.linspace(0., 1., len(p)), inp)
        for a, b in zip(actual[:2], expected[:2]):
            assert np.isfinite(a).all() and np.isfinite(b).all()
            np.testing.assert_allclose(a, b, rtol=5e-11, atol=2e-8)
        np.testing.assert_allclose(actual[1], actual[1].T, rtol=5e-11, atol=2e-8)
