"""Numerical contracts for BBC maps with independent named coordinates."""

from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from jax_salt2mu.biascor_build import _build_one_map, _apply_biascor_grid_cut
from jax_salt2mu.biascor_map import fitparbias_events
from jax_salt2mu.cellgrid import CellGrid, uniform_bininfo
from jax_salt2mu.ccprior import build_ccprior_bins, cell_index
from jax_salt2mu.config import BBCConfig, map_bins, validate_coordinates
from jax_salt2mu.distances import bias_values
from jax_salt2mu.config_parser import parse_bbc_input
from jax_salt2mu.mucovscale import build_mucovscale_map, mucovscale_events
from jax_salt2mu.samples import BiasCorSample
from jax_salt2mu.tablevar import CUTBIT_BIASCOR, CUTBIT_TRUESNIa
from jax_snlc_core.models import StandardisedModel, SALTModel
from jax_snlc_core.utils.error_handling import SNANAError


class DistanceModel(StandardisedModel):
    m0_reference = 0.0
    lcpar_names = ("MU",)
    nuisance_spec = {}

    @classmethod
    def compute_mu(cls, lcpar, nuis, host, u_flags, apply_mass_step=True):
        return lcpar["MU"]

    @classmethod
    def sim_lcpar(cls, tv):
        return {"MU": tv.mumodel}


def test_mu_bias_without_mass_step():
    class Model(DistanceModel):
        lcpar_names = ("MU", "width")

        @classmethod
        def compute_mu(cls, lcpar, nuis, host, u_flags, apply_mass_step=True):
            return lcpar["MU"] + .1 * lcpar["width"]

    tv = SimpleNamespace(lcpar={"MU": np.array([40.]), "width": np.array([2.])},
                         sim_truth={}, zhd=np.array([.2]), logmass=np.array([10.]),
                         mumodel=np.array([39.8]))
    fitted, truth = bias_values(Model, tv, BBCConfig(opt_biascor=128))
    np.testing.assert_allclose(fitted["mu"], [40.2])
    np.testing.assert_allclose(truth["mu"], [39.8])


def sample(names):
    """Twenty events per cell with bias .2 and residual pulls +/-2."""
    class Model(DistanceModel):
        biascor_binning = {name: (-1.5, 1.5, 1.) for name in names}
        default_cuts = {**{name: (-1.5, 1.5) for name in names}, "aux": (-1.5, 1.5)}
        mucovscale_bins = {"aux": 3}
        ccprior_bins = {"aux": 3}

    bins = (uniform_bininfo(.1, .7, 3), uniform_bininfo(-40, 40, 1),
            *(uniform_bininfo(-1.5, 1.5, 3) for _ in names))
    centres = np.array(list(product(*(b.avg for b in bins))))
    values = np.repeat(centres, 20, axis=0)
    n = len(values)
    lcpar = {name: values[:, j + 2] for j, name in enumerate(names)}
    lcpar.update(MU=np.full(n, 40.2), aux=np.zeros(n))
    tv = SimpleNamespace(zhd=values[:, 0], logmass=values[:, 1], lcpar=lcpar,
                         covmat_fit=np.full((n, 1, 1), .01), zmuerr=np.zeros(n),
                         sim_truth={"SIM_WGT_POP": np.ones(n)})
    events = SimpleNamespace(coeff_idx=[], node=np.zeros(n, dtype=int),
                             v=np.ones((n, 1)), mucov_fit=np.full(n, .01),
                             mures=.2 + np.tile([-.2, .2], n // 2),
                             biasval=np.full((n, 1), .2))
    cfg = BBCConfig(lensing_zpar=0., interp_biascor_logmass=0)
    mask = np.ones(n, dtype=bool)
    fitmap = _build_one_map(tv, events, Model, CellGrid(bins), bins, np.array([0.]),
                            mask, .7, cfg)
    return Model, cfg, tv, events, mask, fitmap


@pytest.mark.parametrize("names", [(), ("a",), ("a", "b", "d")])
def test_named_maps(names):
    model, cfg, tv, events, mask, fitmap = sample(names)
    point = {name: np.array([.2]) for name in names}
    result = fitparbias_events(fitmap, (), [.35], [0.], point, 3)
    assert result[-1][0] == 1
    np.testing.assert_allclose(result[0], .2)
    error_map = build_mucovscale_map(tv, events, fitmap, (), np.array([0.]), mask,
                                    *fitmap.bininfo[:2], model, cfg, 3)
    assert error_map.axis_names == ("aux",)
    scale, _, ok = mucovscale_events(error_map, (), [.4], [0.], {"aux": np.array([0.])})
    assert ok[0]
    np.testing.assert_allclose(scale, 4., atol=1e-12)
    assert all(name not in model.lcpar_names for name in (*names, "aux"))


def test_linear_bias_and_order():
    outputs = []
    for names in (("a", "b", "d"), ("d", "a", "b")):
        _, _, _, _, _, m = sample(names)
        slopes = {"a": .1, "b": .2, "d": .3}
        m.fitparbias.val[:, 0] = 2 * m.avg[0] + sum(
            slopes[name] * m.avg[j + 2] for j, name in enumerate(names))
        point = {"a": np.array([.2]), "b": np.array([.3]), "d": np.array([.4])}
        value, _, _, _, status = fitparbias_events(m, (), [.35], [0.], point, 3)
        assert status[0] == 1
        outputs.append(value)
    np.testing.assert_allclose(outputs, .7 + .02 + .06 + .12)


def test_lower_dimensional_coverage():
    _, _, _, _, _, m = sample(())
    assert fitparbias_events(m, (), [.12], [0.], {}, 3)[-1][0] == -9
    m.fitparbias.npercell[0] = 0
    assert fitparbias_events(m, (), [.35], [0.], {}, 3)[-1][0] == -9
    m.fitparbias.npercell[:] = 3
    assert fitparbias_events(m, (), [.35], [0.], {}, 3)[-1][0] == -12


def test_error_population_requires_bias():
    model, cfg, tv, events, mask, m = sample(("a",))
    mask[:] = False
    empty = build_mucovscale_map(tv, events, m, (), np.array([0.]), mask,
                                *m.bininfo[:2], model, cfg, 3)
    assert not empty.use.any()
    mask[:] = True
    m.fitparbias.npercell[:] = 0
    invalid = build_mucovscale_map(tv, events, m, (), np.array([0.]), mask,
                                  *m.bininfo[:2], model, cfg, 3)
    assert not invalid.use.any()


def test_omitted_axis_differs_from_one_bin():
    model, cfg, tv, events, mask, m = sample(())
    cfg.mucovscale_bins = {"aux": 1}
    included = build_mucovscale_map(tv, events, m, (), np.array([0.]), mask,
                                   *m.bininfo[:2], model, cfg, 3)
    cfg.mucovscale_bins = {"aux": 0}
    omitted = build_mucovscale_map(tv, events, m, (), np.array([0.]), mask,
                                  *m.bininfo[:2], model, cfg, 3)
    assert not mucovscale_events(included, (), [.4], [0.], {"aux": np.array([4.])})[-1][0]
    assert mucovscale_events(omitted, (), [.4], [0.], {})[-1][0]


def test_prior_coordinates():
    model, cfg, *_ = sample(("a", "b", "d"))
    cfg.zmin, cfg.zmax, cfg.nzbin_ccprior = .1, .7, 3
    bins = build_ccprior_bins(cfg, model)
    np.testing.assert_array_equal(cell_index(bins, ([.2, .4], [-1., 1.])), [0, 7])
    cfg.ccprior_bins = {"aux": 0}
    bins = build_ccprior_bins(cfg, model)
    np.testing.assert_array_equal(cell_index(bins, ([0., 1.],)), [0, 2])


def test_named_alias_precedence(tmp_path):
    path = tmp_path / "bbc.input"
    path.write_text("nbinc_mucovscale=4\nmucovscale_bins.c=2\nx1min=-2\n")
    cfg = parse_bbc_input(str(path), ["nbinc_mucovscale=5", "lcpar_range.x1=-1,2"])
    assert map_bins(SALTModel, cfg, "mucovscale_bins") == {"c": 5}
    assert cfg.lcpar_range["x1"] == (-1., 2.)
    cfg = parse_bbc_input(str(path), ["mucovscale_bins.c=0", "x1max=1"])
    assert not map_bins(SALTModel, cfg, "mucovscale_bins")
    assert cfg.lcpar_range["x1"] == (-2., 1.)


@pytest.mark.parametrize("override", [-1, 1.5, True])
def test_invalid_bin_count(override):
    with pytest.raises(SNANAError, match="mucovscale_bins.c"):
        validate_coordinates(SALTModel, BBCConfig(mucovscale_bins={"c": override}))


def test_auxiliary_loading_and_grid_cut():
    import pandas as pd
    from jax_snlc_core.io.fitres_reader import FitresTable
    from jax_salt2mu.tablevar import EVENT_TYPE_BIASCOR, tablevar_from

    class Model(DistanceModel):
        biascor_binning = {name: (-1.5, 1.5, 1.) for name in ("a", "b", "d")}
        default_cuts = {name: (-3., 3.) for name in ("a", "b", "d", "aux")}
        fitres_aliases = {**DistanceModel.fitres_aliases, "aux": ("MEASURED_AUX",)}

    df = pd.DataFrame({"CID": ["one", "two"], "zHD": [.3, .3],
                       "MU": [40., 40.], "MUERR": [.1, .1],
                       "a": [0., 0.], "b": [0., 0.], "d": [0., 2.],
                       "MEASURED_AUX": [.1, .2], "SIM_NONIA_INDEX": [0, 0],
                       "SIM_ZCMB": [.3, .3], "SIM_DLMAG": [40., 40.]})
    table = FitresTable(df, list(df), {}, varnames_by_file=[list(df)])
    cfg = BBCConfig(nzbin=2, zmin=.1, zmax=.7)
    tv = tablevar_from(table, EVENT_TYPE_BIASCOR, cfg, Model)
    np.testing.assert_allclose(tv.lcpar["aux"], [.1, .2])
    assert tv.covmat_fit.shape == (2, 1, 1)
    group = BiasCorSample(0, "test", "test", "", False, "", (.1, .7))
    _apply_biascor_grid_cut(tv, np.zeros(2, dtype=int), [group], Model, cfg)
    np.testing.assert_array_equal(tv.cutbits, [0, 1 << CUTBIT_BIASCOR])
    assert tv.n_fail(CUTBIT_TRUESNIa) == 0


def test_prior_gradient():
    import jax
    import jax.numpy as jnp
    from jax_salt2mu.ccprior import CCPriorInputs, PIFAC
    from jax_salt2mu.likelihood import _ccprior_dPdmu
    from jax_salt2mu.fitter import FitSpec

    class Model(DistanceModel):
        @classmethod
        def compute_mu(cls, lcpar, nuis, host, u_flags, apply_mass_step=True):
            return lcpar["MU"] + nuis["offset"]

    residuals = np.concatenate([np.tile([-.3, .1], 6), np.tile([.2, .6], 6)])
    cc = CCPriorInputs(
        cc_lcpar={"MU": jnp.asarray(residuals)}, cc_z=jnp.full(24, .3),
        cc_logmass=jnp.zeros(24), cc_mumodel=jnp.zeros(24),
        cc_fitparbias_nodes=jnp.zeros((24, 1, 1)),
        cc_cell=jnp.repeat(jnp.arange(2), 12), cc_valid=jnp.ones(24),
        data_cell=jnp.arange(2), pIa=jnp.ones(2),
        coeff_avg=(), coeff_binsize=(),
        m0_default=0.)
    spec = FitSpec(Model, (), (), 1, "snana_secant", False,
                   cc_bias_names=("MU",), cc_ncell=2)
    obs = jnp.array([.2, -.1])
    def density(offset):
        return _ccprior_dPdmu(spec, cc, {"offset": offset}, obs)
    means = np.array([-.1, .4])
    expected = PIFAC / .2 * np.exp(-.5 * ((np.asarray(obs) - means) / .2) ** 2)
    np.testing.assert_allclose(density(0.), expected, rtol=1e-12)
    derivative = jax.jacfwd(density)(0.)
    finite = (density(1e-5) - density(-1e-5)) / 2e-5
    np.testing.assert_allclose(derivative, finite, rtol=1e-8)


@pytest.mark.parametrize("names", [(), ("a",), ("a", "b", "d")])
def test_registered_model_preparation(names, monkeypatch):
    import pandas as pd
    from jax_snlc_core.models import MODEL_REGISTRY
    from jax_snlc_core.io.fitres_reader import FitresTable
    from jax_salt2mu.run import prepare_cell
    from jax_salt2mu.tablevar import EVENT_TYPE_DATA, EVENT_TYPE_BIASCOR

    model, cfg, tv, _, _, _ = sample(names)
    model.GENMODEL_ALIASES = ("testdistance",)
    monkeypatch.setitem(MODEL_REGISTRY, "TESTDISTANCE", model)
    cfg.model_lcfit = "TESTDISTANCE"
    cfg.nzbin, cfg.zmin, cfg.zmax = 3, .1, .7
    cfg.opt_biascor = 16 | 32 | 128
    cfg.sigint_biascor = 0.
    cfg.blindflag = 0
    n = len(tv.zhd)
    df = pd.DataFrame({**tv.lcpar, "MUERR": np.full(n, .1),
                       "CID": np.arange(n).astype(str), "zHD": tv.zhd,
                       "IDSURVEY": np.full(n, 10), "SIM_ZCMB": tv.zhd,
                       "SIM_DLMAG": np.full(n, 40.), "SIM_NONIA_INDEX": np.zeros(n)})
    # Symmetric residuals avoid a zero pull variance in the error map.
    df["MU"] += np.tile([-.2, .2], n // 2)
    table = FitresTable(df, list(df), {}, varnames_by_file=[list(df)])
    cell = prepare_cell(cfg, tables={EVENT_TYPE_DATA: table, EVENT_TYPE_BIASCOR: table})
    assert cell.accept.any()
    assert cell.maps.maps[0].axis_names == names
    assert cell.maps.mucov_maps[0].axis_names == ("aux",)
    assert np.all(np.isfinite(cell.inputs.fitparbias_nodes))


@pytest.mark.parametrize("names", [(), ("aux",), ("d", "a", "b")])
def test_error_map_dimensions(names):
    model, cfg, tv, events, mask, fitmap = sample(("a", "b", "d"))
    cfg.mucovscale_bins = {"aux": 0, **dict.fromkeys(names, 3)}
    error_map = build_mucovscale_map(tv, events, fitmap, (), np.array([0.]), mask,
                                    *fitmap.bininfo[:2], model, cfg, 3)
    scale, _, ok = mucovscale_events(
        error_map, (), [.4], [0.], {name: np.array([0.]) for name in names})
    assert ok[0]
    np.testing.assert_allclose(scale, 4., atol=1e-12)


def test_single_cell_bias():
    from jax_salt2mu.biascor_map import BiasCorMap, FitParBias

    bins = (uniform_bininfo(.1, .7, 1), uniform_bininfo(-40, 40, 1))
    m = BiasCorMap(CellGrid(bins), bins,
                   FitParBias(np.array([[.2]]), np.array([[.1]]),
                              np.array([[.1]]), np.array([20])),
                   tuple(b.avg for b in bins), interp_logmass=False)
    value, _, _, count, status = fitparbias_events(m, (), [.4], [0.], {}, 3)
    assert status[0] == 1 and count[0] == 20
    np.testing.assert_allclose(value, .2)
