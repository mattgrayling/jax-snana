"""Unit tests for the BayeSN model's unified generate_sed fit path.

Golden values were captured from the pre-unification `_generate_sed_direct`
(branch point of sed-unify, main @ 966b029) on macOS/ARM. All points allow
ulp-level drift (rtol 1e-12): XLA codegen differs by 1 ulp across platforms
(Linux CI vs macOS), and the av>0 points additionally carry the deliberate
dust-multiply reorder (extinction applied caller-side, after m0_scale).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_snlc_core.models import build_model
from jax_snlc_fit.fitter import fit_static_data


@pytest.fixture(scope="module")
def bayesn_fit_setup(data_root):
    model_path = str(data_root / "models" / "BAYESN.W22")
    model = build_model(model_path, {}, model_type="bayesn")
    return fit_static_data(model, None), model


def test_shift_amplitude_shifts_ds(bayesn_fit_setup):
    """Post-sampling distance updates shift Ds (the SED's distance carrier);
    deltam (header truth) is untouched."""
    _, model = bayesn_fit_setup
    params = {'Ds': jnp.array([34.0, 35.0]), 'deltam': jnp.array([0.1, -0.1])}
    got = model.shift_amplitude(params, jnp.array([0.25, -0.5]))
    np.testing.assert_allclose(np.asarray(got['Ds']), [34.25, 34.5], rtol=0)
    np.testing.assert_allclose(np.asarray(got['deltam']), [0.1, -0.1], rtol=0)


def test_generate_sed_fit_path_gradient(bayesn_fit_setup):
    """Gradients flow through the dict-built fit path (theta, Ds, av)."""
    import jax
    static, model = bayesn_fit_setup
    names = list(model.get_fit_parameters())[1:]
    n_eps = len(names) - 3
    eps = 0.05 * np.random.default_rng(1).standard_normal(n_eps)

    def total_flux(arr):
        params = model.sed_params_from_fit(
            jnp.concatenate([jnp.zeros(1), arr]), 0.05)
        sed = model.generate_sed(10.0, params, static)
        sed = static['host_dust_law'].apply_extinction(
            sed, static['wavelength_grid'], params['av'],
            params.get('rv', static['host_rv']))
        return jnp.sum(sed)

    grad = jax.grad(total_flux)(jnp.array([0.6, 34.5, 0.2, *eps]))
    assert np.all(np.isfinite(np.asarray(grad)))
    assert float(jnp.abs(grad).max()) > 0.0


def test_warping_is_held_at_the_last_knot_past_the_hermite_span(
        bayesn_fit_setup):
    """Without bit 32 the warping past the Hermite span (50 d for W22) is its
    value at the last phase knot (40 d), so the ratio of two SEDs differing
    only in theta is the same at all three phases. Bit 32 takes it to zero
    instead, leaving no theta dependence."""
    static, model = bayesn_fit_setup
    eps = jnp.zeros(model.n_sig_knots)

    def theta_ratio(phase):
        sed = lambda theta: model.generate_sed(
            phase, {'theta': theta, 'Ds': 0.0, 'eps_tform': eps}, static)
        ratio = np.asarray(sed(1.0) / sed(-1.0))
        assert np.all(np.isfinite(ratio))
        return ratio

    at_knot = theta_ratio(40.0)
    assert not np.allclose(at_knot, 1.0)
    for phase in (60.0, 70.0):
        np.testing.assert_allclose(theta_ratio(phase), at_knot, rtol=1e-10)
    try:
        model.set_mskopt(32)
        np.testing.assert_allclose(theta_ratio(60.0), 1.0, rtol=1e-10)
    finally:
        model.set_mskopt(0)


# ---------------------------------------------------------------------------
# GENMODEL_MSKOPT scatter bits and the model-owned DELTAM law
# (genmag_BAYESN.c:458-500 and genSCATTER_BAYESN, :1369-1380)
# ---------------------------------------------------------------------------


def _sample_scatter(model, n_sne=64):
    import jax
    keys_for = lambda name: jax.random.split(
        jax.random.PRNGKey(hash(name) % (2**31)), n_sne)
    out = model.derived_parameters(
        {'mu': jnp.full(n_sne, 38.0), 'theta': jnp.zeros(n_sne)}, keys_for)
    return np.asarray(out['deltam']), np.asarray(out['eps_tform'])


def test_mskopt_scatter_bit_combinations(bayesn_fit_setup):
    """bit1 = epsilon+DELTAM, bit2 = epsilon only, bit4 = DELTAM only,
    no scatter bit = no intrinsic scatter (genmag_BAYESN.c:458-477);
    disabled draws are deterministic zeros with stable shapes."""
    _, model = bayesn_fit_setup
    try:
        model.set_mskopt(0)
        deltam, eps = _sample_scatter(model)
        assert np.all(deltam == 0.0)
        assert np.all(eps == 0.0)

        model.set_mskopt(2)
        deltam2, eps2 = _sample_scatter(model)
        assert np.all(deltam2 == 0.0)
        assert np.any(eps2 != 0.0)

        model.set_mskopt(4)
        deltam4, eps4 = _sample_scatter(model)
        assert np.any(deltam4 != 0.0)
        assert np.all(eps4 == 0.0)
        assert eps4.shape == eps2.shape

        model.set_mskopt(1)
        deltam1, eps1 = _sample_scatter(model)
        assert np.any(deltam1 != 0.0)
        assert np.any(eps1 != 0.0)
    finally:
        model.set_mskopt(0)


def test_mskopt_ambiguous_bits_fatal(bayesn_fit_setup):
    """bit1 with bit2/4, both time modes, or both wavelength modes abort
    (genmag_BAYESN.c:461-468, :487-500)."""
    from jax_snlc_core.logging import SNANAError
    _, model = bayesn_fit_setup
    try:
        for mskopt in (1 | 2, 1 | 4, 1 | 6, 32 | 64, 512 | 1024):
            with pytest.raises(SNANAError, match="Ambiguous"):
                model.set_mskopt(mskopt)
    finally:
        model.set_mskopt(0)


def test_deltam_is_model_sigma0_gaussian(bayesn_fit_setup):
    """DELTAM scales with the model YAML's SIGMA0 (genSCATTER_BAYESN,
    genmag_BAYESN.c:1378-1379), not with any generic GENGAUSS_DELTAM input."""
    _, model = bayesn_fit_setup
    sigma0 = model.sigma0
    try:
        model.set_mskopt(4)
        deltam_a, _ = _sample_scatter(model)
        model.sigma0 = 2.0 * sigma0
        deltam_b, _ = _sample_scatter(model)
        np.testing.assert_allclose(deltam_b, 2.0 * deltam_a, rtol=1e-6)
    finally:
        model.sigma0 = sigma0
        model.set_mskopt(0)


@pytest.mark.parametrize("mskopt", [0, 512, 1024])
def test_model_wavelength_support(data_root, mskopt):
    """Declared coverage must reach the generated spectrum."""
    model = build_model(str(data_root / "models" / "BAYESN.W22"),
                        {'GENMODEL_MSKOPT': mskopt}, model_type="bayesn")
    grid = model.wavelength_grid
    lo, hi = (float(model.wavelength_knots.min()),
              float(model.wavelength_knots.max()))
    knots = (lo, hi)
    if mskopt:
        lo, hi = model.hsiao_wavelength_range
    assert hi > 11000.0
    assert model.get_wavelength_range() == (lo, hi)
    assert float(grid[0]) == pytest.approx(lo)
    assert float(grid[-1]) == pytest.approx(hi)
    assert model.wavelength_spline_coeffs.shape[0] == len(grid)
    if mskopt:
        sed = np.asarray(model.generate_sed(
            0., {'theta': 0., 'Ds': 0., 'eps_tform': jnp.zeros(model.n_sig_knots)},
            fit_static_data(model, None)))
        assert np.isfinite(sed).all()
        assert np.any(sed[np.asarray(grid) < knots[0]] > 0.)
        assert np.any(sed[np.asarray(grid) > knots[1]] > 0.)


def test_phase_range_is_the_hsiao_grid(bayesn_fit_setup):
    """BayeSN's SED is the Hsiao template warped by the model, so its phase
    support is the template's grid. SNANA's PySEDMODEL route imposes no
    phase gate of its own (genSpec_PySEDMODEL)."""
    _, model = bayesn_fit_setup
    lo, hi = model.phase_range()
    assert (lo, hi) == (float(model.hsiao_phases[0]),
                        float(model.hsiao_phases[-1]))
    assert lo < 0.0 < hi


@pytest.mark.parametrize("key, window", [("L_FILTER_CEN_MIN", (0.0, 16500.0)),
                                         ("L_FILTER_CEN_MAX", (4000.0, 0.0))])
def test_filter_centre_limits_are_optional(data_root, tmp_path, key, window):
    """SNANA reads the filter-centre window optionally (genmag_BAYESN.c:203-214):
    without wavelength extrapolation an absent key leaves that limit at zero
    (:573-574); with it the window is the Hsiao edges, 1000 to 25000 A, moved
    in by 1000 and 1500 A, and the YAML is not read (:568-571)."""
    source = data_root / "models" / "BAYESN.W22" / "BAYESN.YAML"
    lines = source.read_text().splitlines(keepends=True)
    trimmed = tmp_path / "BAYESN.YAML"
    trimmed.write_text("".join(line for line in lines if not line.startswith(key)))
    model = build_model(str(trimmed), {}, model_type="bayesn")
    assert model.filter_center_range == window
    for mskopt in (512, 1024):
        model.set_mskopt(mskopt)
        assert model.filter_center_range == (2000.0, 23500.0)


def test_filter_centre_limits_come_from_the_yaml(bayesn_fit_setup):
    """Without wavelength extrapolation the window is the YAML's (:573-574)."""
    _, model = bayesn_fit_setup
    assert model.filter_center_range == (4000.0, 16500.0)


def test_host_law_configuration(data_root, tmp_path):
    from jax_snlc_fit.config_parser import build
    from jax_snlc_sim.inputs import parse_snana_input_file

    path = str(data_root / 'models' / 'BAYESN.W22')
    config = build({'FITMODEL_NAME': path, 'HOST_COLORLAW': 'CCM89'})
    model = build_model(path, config.model._model_key_values)
    law = model.build_host_dust_law()
    assert law.name == 'CCM89'
    source = data_root.parent / 'snana' / 'bayesn_w22' / 'input.input'
    input_file = tmp_path / 'dust.input'
    input_file.write_text(source.read_text() + '\nHOST_COLORLAW: CCM89\n')
    simulation = parse_snana_input_file(input_file)
    simulated = build_model(path, simulation.model_config())
    np.testing.assert_array_equal(
        law.extinction_curve(model.wavelength_grid, 3.1),
        simulated.build_host_dust_law().extinction_curve(model.wavelength_grid, 3.1))


def test_unavailable_bayesn_options(bayesn_fit_setup, data_root):
    from jax_snlc_core.logging import SNANAError

    with pytest.raises(SNANAError, match='UVLAM_EXTRAPFLUX'):
        build_model(str(data_root / 'models' / 'BAYESN.W22'),
                    {'UVLAM_EXTRAPFLUX': 1500.})
    _, model = bayesn_fit_setup
    try:
        model.set_mskopt(2)
        with pytest.raises(SNANAError, match='epsilon realisation'):
            model.replay_params_from_dump({'THETA': np.zeros(1)}, {},
                                         mu=np.zeros(1), distributions={})
    finally:
        model.set_mskopt(0)


@pytest.mark.parametrize("model_name", ["BAYESN.W22", "BAYESN.G26"])
def test_dimensions_are_optional(repo_root, tmp_path, model_name):
    import yaml

    source = repo_root / "jax_snlc_core/data/models" / model_name / "BAYESN.YAML"
    data = yaml.safe_load(source.read_text())
    for key in ("N_LAM", "N_TAU", "N_SIG"):
        data.pop(key, None)
    implicit_path = tmp_path / "implicit.YAML"
    implicit_path.write_text(yaml.safe_dump(data))
    implicit = build_model(str(implicit_path), {}, model_type="bayesn")

    data.update(N_LAM=len(data["L_KNOTS"]), N_TAU=len(data["TAU_KNOTS"]),
                N_SIG=len(data["L_SIGMA_EPSILON"]))
    explicit_path = tmp_path / "explicit.YAML"
    explicit_path.write_text(yaml.safe_dump(data))
    explicit = build_model(str(explicit_path), {}, model_type="bayesn")
    assert implicit.get_fit_parameters() == explicit.get_fit_parameters()
    params = {"theta": 0.4, "Ds": 35.0,
              "eps_tform": jnp.full(implicit.n_sig_knots, 0.1)}
    for phase in (-5.0, 0.0, 20.0):
        np.testing.assert_array_equal(
            implicit.generate_sed(phase, params, fit_static_data(implicit, None)),
            explicit.generate_sed(phase, params, fit_static_data(explicit, None)))


@pytest.mark.parametrize("key", ["N_LAM", "N_TAU", "N_SIG", "W0", "W1",
                                  "L_SIGMA_EPSILON"])
def test_inconsistent_dimensions_are_rejected(data_root, tmp_path, key):
    import yaml
    from jax_snlc_core.logging import SNANAError

    data = yaml.safe_load((data_root / "models/BAYESN.W22/BAYESN.YAML").read_text())
    for size_key in ("N_LAM", "N_TAU", "N_SIG"):
        data.pop(size_key, None)
    if key.startswith("N_"):
        data[key] = 999
    else:
        data[key] = data[key][:-1]
    path = tmp_path / "invalid.YAML"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(SNANAError, match="doesn't match"):
        build_model(str(path), {}, model_type="bayesn")
