"""SIMSED sampling modes and configuration diagnostics across the JIT boundary."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_snlc_core.models.simsed import KEYWORD_BAGGAGE, KEYWORD_CONTINUOUS, SIMSEDModel
from jax_snlc_sim.distributions import GenGaussAsym
from jax_snlc_sim.population import Population


@pytest.mark.parametrize('external_map', [False, True])
def test_surface_weights_preserve_continuous_sampling_mode(external_map):
    model = SIMSEDModel(
        npar=2, parnames=('SHAPE', 'WGT'),
        par_keywords=(KEYWORD_CONTINUOUS, KEYWORD_BAGGAGE),
        sed_par_values=np.array([[0., 1.], [1., 2.]]),
        grid_axes=[np.array([0., 1.]), np.array([1., 2.])],
        jax_template_seds=jnp.ones((2, 2, 2)),
        simsed_wgtmap_file='weights.dat' if external_map else '')
    model.wgtmap = None
    model._build_corner_lookup()
    model._wgt_cumsum = model._surface_wgt_cumsum()
    assert model._wgt_cumsum is not None
    populations = {'simsed_SHAPE': None}
    drawn = model.drawn_parameters(populations)
    assert drawn == (() if external_map else ('simsed_SHAPE',))

    @jax.jit
    def draw(model, keys):
        params = {'mu': jnp.zeros(2), 'sn_global_idx': jnp.arange(2),
                  **{name: jnp.array([.25, .75]) for name in drawn}}
        return model.derived_parameters(params, lambda name: keys)

    result = draw(model, jax.random.split(jax.random.PRNGKey(0), 2))
    if external_map:
        assert 'grid_idx' in result
        assert 'corner_weights' not in result
        assert np.all(np.isin(result['simsed_SHAPE'], [0., 1.]))
    else:
        assert 'grid_idx' not in result
        np.testing.assert_array_equal(result['simsed_SHAPE'], [.25, .75])
        np.testing.assert_array_equal(result['corner_weights'], [[.75, .25], [.25, .75]])


def test_gridonly_ignored_range_warns_before_tracing(caplog):
    model = SIMSEDModel(
        npar=1, parnames=('SHAPE',),
        sed_par_values=np.array([[0.], [1.]]),
        grid_axes=[np.array([0., 1.])], jax_template_seds=jnp.ones((2, 2, 2)))
    model.wgtmap = None
    population = Population(distributions=(GenGaussAsym(
        sigma=(1.e8, 1.e8), range=(.2, .8)),))
    config = {'populations': {'simsed_SHAPE': population},
              'simsed_config': {'continuous_parnames': [],
                                '_gridonly_parnames': ['SHAPE'], 'redcor': {}}}
    model.configure_generation(config, n_simulated=2, frac_pec1a=0.)
    assert any('GENRANGE_SHAPE 0.2 0.8 is ignored' in text for text in caplog.messages)
    caplog.clear()

    # a flat GRIDONLY population draws over the grid, not through the sampler
    assert model.drawn_parameters(config['populations']) == ()

    @jax.jit
    def draw(model, keys):
        return model.derived_parameters(
            {'mu': jnp.zeros(2), 'sn_global_idx': jnp.arange(2)},
            lambda name: keys)

    result = draw(model, jax.random.split(jax.random.PRNGKey(0), 2))
    assert np.all(np.isin(result['simsed_SHAPE'], [0., 1.]))
    assert not caplog.messages
