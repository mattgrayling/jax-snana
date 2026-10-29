"""Exact covariance/derivative parity and compact storage across the pipeline."""
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import yaml

from jax_create_cov import create_covariance, load_config, parse_covopt
from jax_create_cov.covariance import Contribution, covsys_from_covopt
from jax_pipeline import Covariance
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import (dense_factorized_covariance, read_factorized_covariance,
                                       validate_factorized_covariance,
                                       write_factorized_covariance)
from jax_wfit.config import WFITConfig
from jax_wfit.hubble import (HubbleDiagram, cut_supplied_diagram,
                             hubble_diagram_from_covariance, load_hubble_diagrams,
                             prepare_precision, stack_arrays)
from jax_wfit.likelihood import offset_marginalised_chi2
from jax_wfit.run import run_wfit


def factors(n=24, k=3):
    rng = np.random.default_rng(823)
    return {"diag": rng.uniform(.01, .03, n), "U": rng.normal(0, .025, (n, k)), "vec_labels": ()}


def diagram(cov):
    from jax_snlc_core.utils import LIGHT_KM
    from jax_snlc_core.cosmology import distance_modulus
    z = jnp.linspace(.02, 1., cov["diag"].size)
    mu = distance_modulus(z, z, .3, .7, LIGHT_KM / 70., -1., 0.)
    return HubbleDiagram(cid=np.arange(z.size).astype(str), redshift=z,
        mu=mu + .015 * jnp.sin(jnp.arange(z.size)), muerr=jnp.sqrt(cov["diag"]),
        precision=prepare_precision(cov["diag"], cov["U"]), covariance_kind='factorized', covariance=cov,
        meta={'ISDATA_REAL': '0'})


@pytest.mark.parametrize('k', [0, 3, 30])
def test_chi2_gradient_hessian_match_dense(k):
    cov = factors(k=k)
    precision = prepare_precision(cov["diag"], cov["U"])
    dense = jnp.asarray(np.linalg.inv(dense_factorized_covariance(cov)))
    rng = np.random.default_rng(34)
    design = jnp.asarray(rng.normal(size=(24, 3)))
    def chi(p, weights):
        residual = jnp.sin(design @ p) + .07
        return offset_marginalised_chi2(residual, weights, jnp.ones(24, bool), 3.)[0]
    p = jnp.array([.1, -.2, .3])
    for fn in (lambda f: f, jax.grad, jax.hessian):
        actual = jax.jit(fn(lambda p: chi(p, precision)))(p)
        expected = fn(lambda p: chi(p, dense))(p)
        np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-10)


def test_padding_different_row_counts_and_ranks():
    diagrams = [diagram(factors(19, 0)), diagram(factors(24, 4))]
    _, mu, weights, valid = stack_arrays(diagrams)
    actual = jax.jit(offset_marginalised_chi2)(mu, weights, valid, 3.)
    for i, d in enumerate(diagrams):
        expected = offset_marginalised_chi2(d.mu, np.linalg.inv(dense_factorized_covariance(d.covariance)),
                                           jnp.ones(d.n_rows, bool), 3.)
        np.testing.assert_allclose([a[i] for a in actual], expected, atol=1e-8)


def test_cuts_and_scatter_rebuild_small_operator():
    d = diagram(factors())
    config = WFITConfig(zmin=.2, zmax=.8, snrms=.03)
    cut = cut_supplied_diagram(config, d)
    keep = (d.redshift >= .2) & (d.redshift <= .8)
    expected = dense_factorized_covariance(d.covariance)[np.ix_(keep, keep)] + np.eye(cut.n_rows)*.03**2
    np.testing.assert_allclose(dense_factorized_covariance(cut.covariance), expected)
    reset = cut_supplied_diagram(replace(config, snrms=.01), cut)
    expected += np.eye(cut.n_rows) * (.01**2 - .03**2)
    np.testing.assert_allclose(dense_factorized_covariance(reset.covariance), expected)
    from jax_wfit.output import _final_precision
    np.testing.assert_allclose(_final_precision(reset), np.linalg.inv(expected), atol=1e-11)


def test_vector_selection_scales_and_calibrators():
    """Compact factors reproduce scaled dense covariances and calibrator exclusions."""
    base = pd.DataFrame({'CID': ['A', 'B', 'C'], 'MUERR': [.1, .2, .1]})
    contributions = [Contribution("DEFAULT", "DEFAULT", diff=np.zeros(3)),
                     Contribution("VPEC", "DEFAULT", diff=np.array([.1, .2, .3])),
                     Contribution("CAL", "DEFAULT", diff=np.array([.3, .1, .2]))]
    opt = parse_covopt('[ALL] [,,-1.5]')
    expected = covsys_from_covopt(opt, contributions, base, ['A'])
    result = covsys_from_covopt(opt, contributions, base, ['A'], factorized=True)
    np.testing.assert_allclose(result["U"] @ result["U"].T, expected)
    assert result["U"].shape == (3, 2)
    assert result["vec_labels"] == ('VPEC|DEFAULT', 'CAL|DEFAULT')


@pytest.mark.parametrize('mask', [8, 10])
def test_snana_file_roundtrip_and_pipeline_preference(tmp_path, mask):
    cfg = load_config('tests/fixtures/snana/create_cov/input.yml')
    cfg.outdir, cfg.write_mask_cov = tmp_path, mask
    result = create_covariance(cfg)
    stage = Covariance('cov', SimpleNamespace(name='distances'))
    loaded = stage.load({}, tmp_path)
    assert set(loaded['covariances']) == set(result.covariances)
    for label, entry in result.covariances.items():
        # Factors are preferred even when both files exist.
        assert entry['kind'] == loaded['covariances'][label]['kind'] == 'factorized'
        np.testing.assert_allclose(dense_factorized_covariance(loaded['covariances'][label]),
                                   dense_factorized_covariance(entry))
    info = yaml.safe_load((tmp_path / 'INFO.YML').read_text())
    label, _, inverse, filename = info['COVOPTS'][0].split()
    config = WFITConfig(hubble_diagram=str(tmp_path / info['HD']),
                        covariance=str(tmp_path / filename), covariance_kind='factorized',
                        zmin=.04, snrms=.03)
    d = load_hubble_diagrams(config)[0]
    supplied = hubble_diagram_from_covariance(loaded['hubble_diagram'],
        loaded['covariances'][label], loaded['isdata_real'], source='pipeline')
    supplied = cut_supplied_diagram(config, supplied)
    np.testing.assert_allclose(supplied.mu, d.mu)
    np.testing.assert_allclose(dense_factorized_covariance(supplied.covariance), dense_factorized_covariance(d.covariance))
    assert d.covariance_kind == 'factorized'
    assert d.precision["design"].shape[0] == d.n_rows
    assert (inverse != 'None') == (mask == 10)


def test_external_covariance_falls_back_only_for_matching_covopts(tmp_path):
    from jax_create_cov.config import build_config
    cfg = build_config(yaml.safe_load(open('tests/fixtures/snana/create_cov/input_extracov.yml')),
                       {'mode': 'unbinned'})
    cfg.outdir, cfg.write_mask_cov = tmp_path, 8
    result = create_covariance(cfg)
    loaded = Covariance('cov', SimpleNamespace(name='distances')).load({}, tmp_path)
    assert {entry['kind'] for entry in loaded['covariances'].values()} == {
        'total_inverse', 'factorized'}
    for label, entry in result.covariances.items():
        if entry['kind'] == 'systematics':
            np.testing.assert_allclose(loaded['covariances'][label]['matrix'],
                                       result.covtot_inv(label), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('bad_diag,bad_u', [([0., 1.], [[1.], [2.]]),
    ([1., np.nan], [[1.], [2.]]), ([1., 1.], [[np.inf], [2.]]), ([1., 1.], [[1.]])])
def test_reject_invalid_factors(bad_diag, bad_u):
    with pytest.raises(SNANAError):
        validate_factorized_covariance({"diag": bad_diag, "U": bad_u, "vec_labels": ()})


def test_reader_size_validation_and_zero_rank(tmp_path):
    path = tmp_path / 'factorized.npz'
    write_factorized_covariance(path, factors(12, 0))
    assert read_factorized_covariance(path, size=12)["U"].shape == (12, 0)
    with pytest.raises(SNANAError, match='size'):
        read_factorized_covariance(path, size=11)


@pytest.mark.parametrize('method', ['laplace', 'grid'])
def test_full_fit_matches_dense_and_writes_residuals(tmp_path, method):
    d = diagram(factors())
    dense = replace(d, precision=jnp.asarray(np.linalg.inv(dense_factorized_covariance(d.covariance))),
                    covariance_kind='total', covariance=jnp.asarray(dense_factorized_covariance(d.covariance)))
    config = WFITConfig(model='lcdm', method=method, ommin=.1, ommax=.5,
                        omsteps=31, blind_mode='none',
                        cospar_yaml=str(tmp_path/'fit.yaml'),
                        outfile_cospar=str(tmp_path/'fit.cospar'),
                        outfile_resid=str(tmp_path/'resid.txt'))
    actual = run_wfit(config, diagrams=[d], write=True)[0]
    expected = run_wfit(config, diagrams=[dense], write=False)[0]
    np.testing.assert_allclose(actual['covariance'], expected['covariance'], rtol=2e-9)
    for key in actual['likelihood']:
        np.testing.assert_allclose(actual['likelihood'][key], expected['likelihood'][key], atol=1e-9)
    assert (tmp_path/'resid.txt').is_file()


def test_nuts_accepts_factorized_pytree():
    config = WFITConfig(model="lcdm", method="nuts", ommin=.1, ommax=.5,
                        nuts_warmup=30, nuts_samples=30, nuts_chains=1,
                        blind_mode="none")
    result = run_wfit(config, diagrams=[diagram(factors())], write=False)[0]
    assert np.isfinite(result["likelihood"]["total_chi2"])
    assert result["samples"]["omega_m"].size == 30


def test_mixed_dense_fallback_and_factorized_fit_keep_order():
    d = diagram(factors())
    dense = replace(d, precision=jnp.asarray(np.linalg.inv(dense_factorized_covariance(d.covariance))),
                    covariance_kind="total", covariance=jnp.asarray(dense_factorized_covariance(d.covariance)))
    config = WFITConfig(model="lcdm", method="grid", ommin=.1, ommax=.5,
                        omsteps=21, blind_mode="none")
    result = run_wfit(config, diagrams=[d, dense, diagram(factors(19, 0))], write=False)
    np.testing.assert_allclose(result[0]["covariance"], result[1]["covariance"], rtol=1e-10)
    assert result[2]["ndof_hd"] == 17


def test_large_covopt_keeps_compact_storage():
    """A large diagram needs only one stored column per systematic."""
    n = 100_000
    base = pd.DataFrame({"MUERR": np.full(n, .1)})
    contributions = [Contribution("CAL", "DEFAULT", diff=np.linspace(0., .03, n))]
    cov = covsys_from_covopt(parse_covopt("[ALL] [,]"), contributions, base, factorized=True)
    p = prepare_precision(cov["diag"], cov["U"])
    assert p["design"].shape == (n, 1)
    result = offset_marginalised_chi2(jnp.zeros(n), p, jnp.ones(n, bool), 3.)
    assert float(result[0]) == 0.


def test_single_precision_with_large_offset():
    cov = factors(1000, 6)
    precision = prepare_precision(cov["diag"], cov["U"])
    residual = jnp.asarray(40. + np.random.default_rng(1).normal(0., .1, 1000))
    valid = jnp.ones(1000, bool)
    expected = offset_marginalised_chi2(residual, precision, valid, 3.)
    actual = offset_marginalised_chi2(residual.astype(jnp.float32),
        jax.tree.map(lambda a: a.astype(jnp.float32), precision), valid, 3.)
    np.testing.assert_allclose(actual, expected, rtol=5e-6)


def test_snana_four_field_legacy_inverse_metadata(tmp_path):
    cfg = load_config('tests/fixtures/snana/create_cov/input.yml')
    cfg.outdir = tmp_path
    create_covariance(cfg)
    path = tmp_path / 'INFO.YML'
    info = yaml.safe_load(path.read_text())
    info['COVOPTS'] = {i: entry + ' None' for i, entry in info['COVOPTS'].items()}
    path.write_text(yaml.safe_dump(info))
    loaded = Covariance('cov', SimpleNamespace(name='distances')).load({}, tmp_path)
    assert loaded['covariances']
    assert all(entry['kind'] == 'total_inverse' for entry in loaded['covariances'].values())


def test_factorized_cli_and_error_override_rejection():
    from jax_wfit.config import build_parser, cli_keys, build
    cfg = build(cli_keys(build_parser().parse_args(
        ['hd.txt', '-mucovtot_factorized_file', 'cov.npz'])))
    assert cfg.covariance_kind == 'factorized'
    with pytest.raises(SNANAError, match='muerr_force'):
        replace(cfg, muerr_force=.1).validate()


@pytest.mark.parametrize("scale", [.02, .1])
def test_float32_coherent_systematic_at_large_n(scale):
    # Along a systematic mode, the old Woodbury subtraction loses significant
    # digits in both chi2 and derivatives even at realistic systematic scales.
    n = 100_000
    x = np.linspace(-1., 1., n)
    cov = {"diag": np.full(n, .01), "U": (scale * x)[:, None], "vec_labels": ()}
    p = jax.tree.map(lambda a: a.astype(jnp.float32), prepare_precision(cov["diag"], cov["U"]))
    direction = jnp.asarray(x, dtype=jnp.float32)

    def chi(amplitude):
        return offset_marginalised_chi2(
            amplitude * direction, p, jnp.ones(n, bool), 5.)[0]

    coefficient = np.sum(x*x) / (.01 + scale**2 * np.sum(x*x))
    expected = [.01 * coefficient, .2 * coefficient, 2 * coefficient]
    actual = [float(jax.jit(fn)(jnp.float32(.1)))
              for fn in (chi, jax.grad(chi), jax.hessian(chi))]
    np.testing.assert_allclose(actual, expected, rtol=5e-5)


@pytest.mark.parametrize("k", [0, 3, 30])
def test_float32_degenerate_systematics_and_offset_match_dense(k):
    rng = np.random.default_rng(731)
    n = 150
    U = rng.normal(0, .03, (n, k))
    if k:
        U[:, 0] = .05  # exactly degenerate with the magnitude offset
    if k > 1:
        U[:, 1] = U[:, 0] * (1 + 1e-8)
    cov = {"diag": rng.uniform(.005, .03, n), "U": U, "vec_labels": ()}
    dense = jnp.asarray(np.linalg.inv(dense_factorized_covariance(cov)))
    p = jax.tree.map(lambda a: a.astype(jnp.float32), prepare_precision(cov["diag"], cov["U"]))
    residual = 40. + rng.normal(0, .1, n)
    design = rng.normal(size=(n, 3))

    def reference(t):
        r = jnp.asarray(residual) + jnp.asarray(design) @ t
        b = jnp.sum(dense @ r) / (jnp.sum(dense) + 1 / 25.)
        e = r - b
        return e @ dense @ e + b*b / 25.

    def compact(t):
        r = jnp.asarray(residual, jnp.float32) + jnp.asarray(design, jnp.float32) @ t
        return offset_marginalised_chi2(r, p, jnp.ones(n, bool), 5.)[0]

    for fn in (lambda f: f, jax.grad, jax.hessian):
        expected = fn(reference)(jnp.array([.01, .02, -.03]))
        actual = jax.jit(fn(compact))(jnp.array([.01, .02, -.03], jnp.float32))
        np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=.005)


def test_nuts_streams_follow_original_indices_across_format_groups():
    """Grouping by covariance format preserves each dataset's sampling stream."""
    d = diagram(factors())
    dense = replace(d, precision=jnp.asarray(np.linalg.inv(dense_factorized_covariance(d.covariance))),
                    covariance_kind="total", covariance=jnp.asarray(dense_factorized_covariance(d.covariance)))
    config = WFITConfig(model="lcdm", method="nuts", blind_mode="none", seed=73,
                        nuts_warmup=30, nuts_samples=30, nuts_chains=1)
    mixed = run_wfit(config, diagrams=[d, dense, d, dense], write=False)
    regrouped = run_wfit(config, diagrams=[dense, dense, d, d], write=False)
    # Dataset 1 stays dense and dataset 2 stays factorized, but both change
    # position within their format group. Their random streams must stay put.
    for index in (1, 2):
        np.testing.assert_allclose(mixed[index]["samples"]["omega_m"],
                                   regrouped[index]["samples"]["omega_m"], atol=1e-8, rtol=1e-8)
    assert not np.array_equal(mixed[0]["samples"]["omega_m"],
                              mixed[2]["samples"]["omega_m"])


@pytest.mark.parametrize("method", ["laplace", "grid", "nuts"])
def test_single_precision_backend_with_padded_factorized_diagrams(method):
    # Exercise the actual default precision, not just casts under an x64 context.
    with jax.enable_x64(False):
        config = WFITConfig(model="lcdm", method=method, ommin=.1, ommax=.5,
                            omsteps=31, nuts_warmup=30, nuts_samples=30,
                            nuts_chains=1, blind_mode="none")
        results = run_wfit(config, diagrams=[diagram(factors(24, 3)),
                                             diagram(factors(19, 0))], write=False)
        assert len(results) == 2
        for result in results:
            assert np.isfinite(result["likelihood"]["total_chi2"])
            assert np.isfinite(result["covariance"]).all()


@pytest.mark.parametrize("rank", [0, 3])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_explicit_precision_output_uses_host_covariance_in_float32(rank, dtype):
    from jax_wfit.output import _final_precision
    cov = factors(100, rank)
    cov = {**cov, "diag": cov["diag"].astype(dtype), "U": cov["U"].astype(dtype)}
    with jax.enable_x64(False):
        d = hubble_diagram_from_covariance(pd.DataFrame({
            "CID": np.arange(100).astype(str), "zHD": np.linspace(.02, 1., 100),
            "MU": np.full(100, 40.), "MUERR": np.sqrt(cov["diag"])}),
            cov, 0, source="array input")
        actual = _final_precision(d)
    assert actual.dtype == np.float64
    expected = np.linalg.inv(dense_factorized_covariance(validate_factorized_covariance(cov)))
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
