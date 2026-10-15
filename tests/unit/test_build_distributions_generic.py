"""Generic GENPEAK_/GENMEAN_ distribution building (no per-model name lists).

Any GENPEAK_<suffix>/GENMEAN_<suffix> key yields a GenGaussAsym for <suffix>;
the dust profile (AV / EBV_HOST) is exp + half-Gaussian and owns its keys.
"""

import pytest

from jax_snlc_sim.distributions import GenExpHalfGauss, GenGaussAsym
from jax_snlc_core.utils.error_handling import SNANAError
from jax_snlc_core.models import MODEL_REGISTRY
from jax_snlc_sim.config_parser import (_build_distributions, _spell_population_keys,
                                        _spellings)
from jax_snlc_sim.population import declared_population, population_spellings


def spelled(raw):
    """A SALT run's input as the parser hands it on, its population keys
    spelled by parameter name."""
    _spell_population_keys(raw, _spellings(MODEL_REGISTRY['salt'], {}))
    return raw


@pytest.mark.parametrize('named', [False, True])
def test_subpop_coefficients_override_shared_population(named):
    from dataclasses import asdict, replace
    from jax_snlc_sim.config_parser import _build_subpops

    inherited = GenGaussAsym(
        peak=0.2, sigma=(0.8, 1.1), range=(-3.0, 3.0),
        peakrange=(-0.1, 0.4), skew=(0.01, 0.02), prob2=0.3,
        peak2=1.2, sigma2=(0.3, 0.5), ngrid=5)
    raw = {
        'GENPEAK_SALT2x1': inherited.peak,
        'GENSIGMA_SALT2x1': list(inherited.sigma),
        'GENRANGE_SALT2x1': list(inherited.range),
        'GENPEAKRANGE_SALT2x1': list(inherited.peakrange),
        'GENSKEW_SALT2x1': list(inherited.skew),
        'GENPROB2_SALT2x1': inherited.prob2,
        'GENPEAK2_SALT2x1': inherited.peak2,
        'GENSIGMA2_SALT2x1': list(inherited.sigma2),
        'GENGRID_SALT2x1': inherited.ngrid,
    }
    if named:
        raw = {'POPULATIONS': {'x1': inherited}}
    raw['SUBPOP'] = [
        'wide WGT 1', 'wide GENSIGMA_SALT2x1 1.5 2.0',
        'second WGT 1', 'second GENPEAK2_SALT2x1 -1.2',
        'unchanged WGT 1',
    ]
    raw = spelled(raw)
    distributions, maps = _build_distributions(raw, MODEL_REGISTRY['salt'], {})
    classes = _build_subpops(raw, MODEL_REGISTRY['salt'], {}, maps, distributions)
    expected = [replace(inherited, sigma=(1.5, 2.0)),
                replace(inherited, peak2=-1.2), inherited]
    assert [asdict(dists['x1']) for _, dists, _ in classes] == [
        asdict(dist) for dist in expected]
    assert distributions['x1'] == inherited


def test_subpop_peak_updates_named_population_peakrange():
    from dataclasses import replace
    from jax_snlc_sim.config_parser import _build_subpops

    inherited = GenGaussAsym(peak=0.2, sigma=(0.8, 1.1), range=(-3., 3.),
                            peakrange=(-0.1, 0.4), prob_expon_rewgt=2.)
    raw = {'POPULATIONS': {'x1': inherited},
           'SUBPOP': ['young WGT 1', 'young GENPEAK_SALT2x1 1',
                      'old WGT 1', 'old GENPEAK_SALT2x1 -1']}
    distributions, maps = _build_distributions(raw, MODEL_REGISTRY['salt'], {})
    classes = _build_subpops(raw, MODEL_REGISTRY['salt'], {}, maps, distributions)
    assert [dists['x1'] for _, dists, _ in classes] == [
        replace(inherited, peak=1., peakrange=(1., 1.)),
        replace(inherited, peak=-1., peakrange=(-1., -1.))]


@pytest.mark.parametrize('parameter, spelling', [('AV', 'AV_HOST'), ('EBV_HOST', 'EBV')])
def test_subpop_dust_coefficient_inherits_named_population(parameter, spelling):
    from dataclasses import replace
    from jax_snlc_sim.config_parser import _build_subpops

    inherited = GenExpHalfGauss(exp_tau=0.3, peak=0.1, sigma=0.4, ratio=0.2,
                                range=(0., 2.), prob_expon_rewgt=1.5)
    raw = {'POPULATIONS': {parameter: inherited},
           'SUBPOP': ['dusty WGT 1', f'dusty GENTAU_{spelling} 0.8',
                      'unchanged WGT 1']}
    distributions, maps = _build_distributions(raw, MODEL_REGISTRY['salt'], {})
    classes = _build_subpops(raw, MODEL_REGISTRY['salt'], {}, maps, distributions)
    assert [dists[parameter] for _, dists, _ in classes] == [
        replace(inherited, exp_tau=0.8), inherited]
    assert distributions[parameter] == inherited


def build(config):
    distributions, _ = _build_distributions(spelled(dict(config)),
                                            MODEL_REGISTRY['salt'], {})
    return distributions


def test_salt_params_built_generically():
    dists = build({
        'GENPEAK_SALT2x1': 0.0, 'GENSIGMA_SALT2x1': [1.0, 1.0], 'GENRANGE_SALT2x1': [-3.0, 3.0],
        'GENPEAK_SALT2c': -0.05, 'GENSIGMA_SALT2c': [0.05, 0.1], 'GENRANGE_SALT2c': [-0.3, 0.5],
        'GENPEAK_SALT2ALPHA': 0.14,
        'GENPEAK_SALT2BETA': 3.1,
    })
    assert set(dists) == {'x1', 'c', 'SALT2alpha', 'SALT2beta'}
    assert isinstance(dists['x1'], GenGaussAsym)
    assert dists['c'].peak == -0.05
    assert dists['c'].sigma == (0.05, 0.1)
    assert dists['c'].range == (-0.3, 0.5)
    assert dists['SALT2alpha'].sigma == (0.0, 0.0)


def test_bayesn_and_simsed_suffixes():
    dists = build({
        'GENMEAN_THETA': 0.5, 'GENSIGMA_THETA': [1.0, 1.0], 'GENRANGE_THETA': [-2.0, 3.0],
        'GENPEAK_DELTAM': 0.0, 'GENSIGMA_DELTAM': [0.1, 0.1], 'GENRANGE_DELTAM': [-1.0, 1.0],
        'GENPEAK_stretch': 1.0, 'GENSIGMA_stretch': [0.2, 0.2], 'GENRANGE_stretch': [0.5, 1.5],
    })
    assert set(dists) == {'THETA', 'DELTAM', 'stretch'}
    assert dists['THETA'].peak == 0.5


def test_any_gengauss_key_activates_snana_parameter():
    """prepIndex_GENGAUSS sets USE on any family key
    (sntools_genGauss_asym.c:146-165); missing keys keep the SNANA init
    values (snlc_sim.c:872-917)."""
    dists = build({
        'GENPEAK_SALT2x1': 0.0, 'GENSIGMA_SALT2x1': [1.0, 1.0],
        'GENPEAK2_SALT2x1': 2.0, 'GENPROB2_SALT2x1': 0.1, 'GENSIGMA2_SALT2x1': [0.5, 0.5],
        'GENPEAKRANGE_SALT2c': [-0.1, 0.1],
    })
    assert set(dists) == {'x1', 'c'}
    assert dists['x1'].prob2 == 0.1
    # x1 without GENRANGE keeps SNANA's init range, not a delta
    assert dists['x1'].range == (-5.0, 5.0)
    # c activated by GENPEAKRANGE alone, init values fill the rest
    assert dists['c'].peak == -9.0
    assert dists['c'].sigma == (1.0, 1.0)
    assert dists['c'].range == (-0.5, 1.0)
    assert dists['c'].peakrange == (-0.1, 0.1)


def test_gensigma_rv_alone_is_a_live_population():
    """GENSIGMA_RV without GENPEAK/GENRANGE uses SNANA's RV init
    (peak 3.1, range (1, 5); snlc_sim.c:884-887)."""
    dists = build({'GENSIGMA_RV': [0.5, 0.5]})
    assert dists['RV'].peak == 3.1
    assert dists['RV'].range == (1.0, 5.0)
    assert dists['RV'].sigma == (0.5, 0.5)


def test_gengauss_negative_sigma_aborts():
    """checkVal_GENGAUSS (sntools_genGauss_asym.c:607-639)."""
    with pytest.raises(SNANAError, match="non-negative"):
        build({'GENPEAK_SALT2x1': 0.0, 'GENSIGMA_SALT2x1': [-1.0, 1.0]})


def test_gengauss_inverted_range_aborts():
    with pytest.raises(SNANAError, match="inverted"):
        build({'GENPEAK_SALT2x1': 0.0, 'GENSIGMA_SALT2x1': [1.0, 1.0],
               'GENRANGE_SALT2x1': [2.0, -2.0]})


def test_genpeak_av_alone_builds_no_av_distribution():
    """GENPEAK_AV is not an SNANA dust key (the Gaussian-core peak key is
    GENGAUPEAK_AV, sntools_genExpHalfGauss.c:364-369)."""
    dists = build({'GENPEAK_AV': 0.0})
    assert 'AV' not in dists


def test_av_halfgauss_gengaupeak_grammar():
    dists = build({
        'GENGAUPEAK_AV': 0.3, 'GENTAU_AV': 0.4, 'GENSIG_AV': 1.0,
        'GENRATIO_AV0': 4.0, 'GENRANGE_AV': [0.001, 3.0],
    })
    assert isinstance(dists['AV'], GenExpHalfGauss)
    assert dists['AV'].exp_tau == 0.4
    assert dists['AV'].peak == 0.3
    assert dists['AV'].sigma == 1.0


def test_av_halfgauss_ignores_legacy_genpeak_av():
    """A GENPEAK_AV line is inert in SNANA; the Gaussian core keeps the -9
    init peak, which samples the positive half-normal."""
    dists = build({
        'GENPEAK_AV': 0.0, 'GENTAU_AV': 0.4, 'GENSIGMA_AV': 1.0,
        'GENRATIO_AV0': 4.0, 'GENRANGE_AV': [0.001, 3.0],
    })
    assert dists['AV'].peak == -9.0


def test_av_halfgauss_without_range_is_no_dust():
    """Without GENRANGE the (-9, -9) init range collapses every draw to the
    AV = -9 no-dust sentinel (getRan_GEN_EXP_HALFGAUSS delta branch), so the
    profile reads as absent - the GENTAU_AV: 0.0 idiom real inputs use."""
    assert 'AV' not in build({'GENTAU_AV': 0.0})
    assert 'AV' not in build({'GENTAU_AV': 0.4})


def test_av_and_ebv_host_mutually_exclusive():
    from jax_snlc_sim.config_parser import _build_populations

    distributions = build({
        'GENTAU_AV': 0.4, 'GENRANGE_AV': [0.0, 3.0],
        'GENTAU_EBV_HOST': 0.2, 'GENRANGE_EBV_HOST': [0.0, 1.0],
    })
    with pytest.raises(SNANAError, match="one or the other"):
        _build_populations(distributions, {}, {}, {}, [])


def test_genpdf_embedded_keys_become_distributions(tmp_path):
    genpdf = tmp_path / "test.GENPDF"
    genpdf.write_text(
        "GENPEAK_SALT2ALPHA: 0.146\n"
        "GENPEAK_THETA: 0.7\n"
        "VARNAMES: SALT2x1 PROB\n"
        "PDF: -2.0 0.1\n"
        "PDF: 0.0 1.0\n"
        "PDF: 2.0 0.1\n"
    )
    dists = build({'GENPDF_FILE': str(genpdf)})
    assert dists['SALT2alpha'].peak == 0.146
    assert dists['THETA'].peak == 0.7


@pytest.mark.parametrize("varname", ["SALT2colour", "SIM_SALT2c"])
def test_genpdf_map_naming_no_parameter_aborts(tmp_path, varname):
    """Nothing draws from a map whose variable no model declares, so the
    typo aborts instead of leaving the population it asked for unused;
    a SIM_ prefix, the name of the output column rather than of the
    parameter, is the mistake SNANA names (checkAbort_VARNAME_GENPDF,
    sntools_genPDF.c:561)."""
    genpdf = tmp_path / "typo.GENPDF"
    genpdf.write_text(
        f"VARNAMES: {varname} PROB\n"
        "PDF: -0.1 0.5\n"
        "PDF:  0.1 0.5\n"
    )
    with pytest.raises(SNANAError, match="names no population parameter"):
        build({'GENPDF_FILE': str(genpdf)})


def test_genpdf_map_takes_either_dust_spelling(tmp_path):
    """SNANA names host dust EBV (PARNAME_EBV, sntools.h:174); the map is
    stored under the key the sampler draws."""
    genpdf = tmp_path / "ebv.GENPDF"
    genpdf.write_text(
        "VARNAMES: EBV PROB\n"
        "PDF: 0.0 1.0\n"
        "PDF: 0.5 0.5\n"
    )
    _, maps = _build_distributions({'GENPDF_FILE': str(genpdf)},
                                   MODEL_REGISTRY['salt'], {})
    assert set(maps) == {'EBV_HOST'}


@pytest.mark.parametrize('field,legacy,value', [
    ('range', 'GENRANGE', [2., -2.]), ('range', 'GENRANGE', [0., float('inf')]),
    ('sigma', 'GENSIGMA', [-1., 1.]), ('sigma2', 'GENSIGMA2', [float('nan'), 1.]),
    ('prob2', 'GENPROB2', 1.1), ('prob2', 'GENPROB2', float('nan')),
    ('ngrid', 'GENGRID', -1), ('ngrid', 'GENGRID', 101),
])
def test_gaussian_constraints_are_shared_by_input_formats(field, legacy, value):
    from jax_snlc_sim.config_parser import _parse_gengauss_asym
    from jax_snlc_sim.population import distribution_from_settings
    messages = []
    for read in (
        lambda: distribution_from_settings({'range': [-3., 3.], field: value}),
        lambda: _parse_gengauss_asym({f'{legacy}_X': value}, 'X', {'range': (-3., 3.)}),
    ):
        with pytest.raises(SNANAError) as exc:
            read()
        messages.append(str(exc.value))
    assert messages[0] == messages[1]


def test_population_keys_take_any_spelling_in_any_case():
    """SNANA's spelling and the model's own name, in any case, set the one
    population, stored under the name."""
    dists = build({'GENPEAK_SALT2X1': 0.5, 'GENSIGMA_x1': [1.0, 1.0],
                   'GENRANGE_X1': [-3.0, 3.0]})
    assert set(dists) == {'x1'}
    assert (dists['x1'].peak, dists['x1'].sigma, dists['x1'].range) == (
        0.5, (1.0, 1.0), (-3.0, 3.0))


def test_one_key_given_under_two_spellings_aborts():
    with pytest.raises(SNANAError, match="same population key"):
        build({'GENPEAK_SALT2x1': 0.5, 'GENPEAK_x1': 0.4})


def test_a_spelling_two_parameters_claim_aborts():
    """Names match whatever their case, so a model whose input declares
    two names differing only in case, or one that is also a host-dust name,
    cannot be read."""
    with pytest.raises(SNANAError, match="names both"):
        population_spellings(('stretch', 'STRETCH'), {})
    with pytest.raises(SNANAError, match="names both"):
        population_spellings(('av',), {})


def test_zvariation_may_shift_any_population_parameter():
    """BayeSN's theta, under SNANA's spelling THETA, takes a ZVARIATION
    like any other population parameter."""
    from jax_snlc_sim.zvariation import load_zvariation

    names, aliases, _ = declared_population(MODEL_REGISTRY['bayesn'], {})
    entries = load_zvariation([('GENPEAK_THETA', '0,0.5')], '',
                              population_spellings(names, aliases), False,
                              (0.01, 1.0))
    assert list(entries) == ['GENPEAK_theta']


def test_genpdf_flat_takes_any_population_parameter():
    """SNANA's GENPDF_FLAT handles a fixed list (prep_GENPDF_FLAT,
    snlc_sim.c:8102); flattening works alike for any parameter, so here it
    takes BayeSN's theta."""
    dists, _ = _build_distributions({'GENPDF_FLAT': 'THETA(-1:2)'},
                                    MODEL_REGISTRY['bayesn'], {})
    assert dists['theta'].range == (-1.0, 2.0)
