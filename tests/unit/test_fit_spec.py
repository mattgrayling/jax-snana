"""Model declarations fail at setup, before tracing the fitter."""

import pytest

from jax_snlc_core.logging import SNANAError
from jax_snlc_core.models import build_model
from jax_snlc_fit.parameters import validate_fit_parameters


@pytest.fixture(scope='module')
def model(data_root):
    return build_model(str(data_root / 'models' / 'SALT3.K21'), {}, model_type='salt')


def test_model_parameters(model):
    validate_fit_parameters(model, model.get_fit_parameters())


@pytest.mark.parametrize('change, message', [
    ({'bounds_relative_to_init': (0., 1.)}, 'exactly one'),
    ({'bounds': (5., -5.)}, 'lower < upper'),
    ({'scale': 0.}, 'positive'),
])
def test_invalid_parameter(model, change, message):
    parameters = model.get_fit_parameters()
    parameters['x1'].update(change)
    with pytest.raises(SNANAError, match=f'SALTModel.*x1.*{message}'):
        validate_fit_parameters(model, parameters)


def test_required_parameters(model):
    parameters = model.get_fit_parameters()
    del parameters['x1']['init']
    with pytest.raises(SNANAError, match='x1.*init is required'):
        validate_fit_parameters(model, parameters)
    parameters = model.get_fit_parameters()
    parameters['zPHOT'] = {'bounds': (0., 1.), 'init': .1}
    with pytest.raises(SNANAError, match='zPHOT.*supplied by the fitter'):
        validate_fit_parameters(model, parameters)
