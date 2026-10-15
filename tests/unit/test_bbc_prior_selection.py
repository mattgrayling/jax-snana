"""BBC source overrides and the analytic H11 prior at the preparation boundary."""

import logging

import pytest

from jax_pipeline import BBC, Fit
from jax_salt2mu.config import BBCConfig, H11_PARAMS
from jax_salt2mu.config_parser import parse_bbc_input
from jax_salt2mu.run import floated_and_defaults, prepare_cell
from jax_salt2mu.tablevar import EVENT_TYPE_DATA, EVENT_TYPE_BIASCOR, EVENT_TYPE_CCPRIOR
from jax_snlc_core.io.fitres_reader import read_fitres
from jax_snlc_core.models import SALTModel
from jax_snlc_core.utils.error_handling import SNANAError


@pytest.mark.parametrize('layer', ['muopts', 'muopt_settings'])
@pytest.mark.parametrize('role,key,event_type', [
    ('data', 'datafile', EVENT_TYPE_DATA),
    ('biascor', 'simfile_biascor', EVENT_TYPE_BIASCOR),
    ('ccprior', 'simfile_ccprior', EVENT_TYPE_CCPRIOR),
])
def test_explicit_source_overrides_replace_inherited_tables(layer, role, key, event_type):
    data = Fit('data', fitres='data.FITRES')
    own = data if role == 'data' else Fit(role, fitres=f'{role}.FITRES')
    other = Fit('other', fitres='other.FITRES')
    sources = {'data': data, role: own}
    replacement = ['replacement.FITRES'] if layer == 'muopt_settings' else 'replacement.FITRES'
    stage = BBC('bbc', **sources, **{layer: {
        'file': {key: replacement},
        'stage': {key: other},
    }})

    def selected(label):
        return [s for t, _, s in stage._sources(label) if t == event_type]

    assert selected(None) == [own]
    assert selected('file') == []
    assert selected('stage') == [other]
    table = read_fitres('tests/fixtures/snana/bbc_des_beams/DATA.FITRES')
    cfg, tables = stage._cell({'data:data': {'nominal': table}}, (0, 1), {})
    assert getattr(cfg, key) == ['replacement.FITRES']
    assert event_type not in tables


@pytest.mark.parametrize('layer', ['muopts', 'muopt_settings'])
@pytest.mark.parametrize('spelling', ['H11', 'h11'])
def test_h11_muopt_reaches_likelihood_without_nominal_cc_table(layer, spelling):
    data = Fit('data', fitres='unused')
    cc = Fit('cc', fitres='unused')
    prior = [spelling] if layer == 'muopt_settings' else spelling
    stage = BBC('bbc', data=data, ccprior=cc,
                replace_ccprior={'DEFAULT': 'must-not-read.FITRES'},
                overrides={'nzbin': 5, 'zmin': .1, 'zmax': 1., 'chi2max': 1e7,
                           'varname_pIa': 'PROBIA', 'typeIa_ccprior': 999,
                           'p1': .14, 'p2': 3.1, 'u1': 1, 'u2': 1, 'u13': 1},
                **{layer: {'h11': {'simfile_ccprior': prior}}})
    table = read_fitres('tests/fixtures/snana/bbc_des_beams/CONTAM_DATA.FITRES')
    cfg, tables = stage._cell({'data:data': {'nominal': table}}, (0, 1), {})
    assert set(tables) == {EVENT_TYPE_DATA}
    cell = prepare_cell(cfg, tables)
    assert cell.inputs.h11prior is not None
    assert cell.inputs.ccprior is None
    assert set(H11_PARAMS) <= set(cell.spec.floated)
    assert 'scalePCC' in cell.spec.floated


@pytest.mark.parametrize('flags', [[], ['u17=1', 'u20=1'], ['u17=0', 'u20=0'],
                                   [f'u{i}=1' for i in range(17, 23)]])
def test_h11_always_floats_six_coefficients(flags, caplog):
    cfg = parse_bbc_input(None, ['simfile_ccprior=H11', 'u13=1', *flags])
    with caplog.at_level(logging.WARNING):
        floated, _ = floated_and_defaults(SALTModel, cfg)
    assert set(H11_PARAMS) <= set(floated)
    assert 'scalePCC' in floated
    assert len(floated) == len(set(floated))
    assert ('H11 floats all six' in caplog.text) == (bool(flags) and len(flags) < 6)


@pytest.mark.parametrize('in_memory', [False, True])
def test_conflicting_cc_priors_fail_before_reading_data(in_memory):
    cfg = BBCConfig(use_h11=True, nuisance_float={'scalePCC': True})
    tables = {EVENT_TYPE_CCPRIOR: object()} if in_memory else None
    if not in_memory:
        cfg.simfile_ccprior = ['not-read.FITRES']
    with pytest.raises(SNANAError, match='H11 and a simulated CC-prior'):
        prepare_cell(cfg, tables)
