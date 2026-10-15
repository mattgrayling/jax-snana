"""Unit tests for non-trivial default implementations on ``SNModel``.

The tests in this file each exercise a default that has real semantic
content — a t0 prior with a specific shape, a chi2-survival formula,
parameter-name-aware logic, etc. Defaults that are just ``pass`` /
``return None`` / ``return {}`` are intentionally NOT tested; they
have no failure mode worth pinning.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_snlc_core.io.fits_path import open_text
from jax_snlc_core.models import FittedModel, SNModel, SimulatedModel
from jax_snlc_core.models.salt_model import compute_expected_x0_scale
from jax_snlc_core.utils.error_handling import SNANAError


class TinyModel(SNModel, SimulatedModel, FittedModel):
    """Minimal concrete model playing every role, for default-method testing."""

    def __init__(self):
        super().__init__("tiny", {})

    def get_wavelength_range(self):
        return (3000.0, 9000.0)

    def phase_range(self):
        return (-20.0, 50.0)

    def sed_param_names(self):
        return ()

    def generate_sed(self, phase, params, static_data):
        return jnp.zeros_like(static_data["wavelength_grid"])

    def get_snana_header_params(self, parameters):
        return {}

    def get_fit_parameters(self):
        return {"t0": {"bounds": (-30.0, 30.0), "scale": 1.0},
                "amp": {"bounds": (0.0, 1.0), "scale": 1.0}}


# ---------------------------------------------------------------------------
# The t0 prior's shape, which every model gets and no model declares
# ---------------------------------------------------------------------------

def test_t0_prior_is_gaussian_then_flares_then_saturates():
    """Gaussian on t0 with sigma=20d, blowing up beyond ±28d, and stopping
    once the offset leaves the ±30d grid — SNANA tabulates the prior over that
    range and CHI2_PRIOR clamps into its ends, so the flare cannot run away."""
    from jax_snlc_fit.priors import FORMS
    from jax_snlc_core.models.fitted import (PKMJD_PRIOR_FLARE,
                                             PKMJD_PRIOR_GRID)

    def chi2(dt):
        return float(FORMS['gauss_flare'](jnp.array(dt), {
            'centre': 0.0, 'sigma': 20.0,
            'grid': PKMJD_PRIOR_GRID, 'flare': PKMJD_PRIOR_FLARE}))

    assert chi2(0.0) == pytest.approx(0.0, abs=1e-6)
    assert chi2(20.0) == pytest.approx(1.0, abs=1e-3)      # one sigma
    assert chi2(-20.0) == pytest.approx(1.0, abs=1e-3)     # and symmetric

    far = chi2(35.0)
    assert far > 1.5 * (35.0 / 20.0) ** 2, "the flare must beat a quadratic"
    # past the grid the penalty is flat: 40d and 100d cost what 30d costs
    assert chi2(40.0) == pytest.approx(far, rel=1e-12)
    assert chi2(100.0) == pytest.approx(far, rel=1e-12)


def test_shift_amplitude_carries_a_distance_update_to_a_flux_scale():
    """Post-sampling distance updates (SN2GAL_Z / HOSTLIB VPEC) must reach the
    flux. The simulator and the fitter ask for this the same way, so a model
    whose brightness is a flux multiplier scales it and leaves the rest."""
    class FluxScale(TinyModel):
        distance_param = 'distance_scale'

        def _amplitude_shifted(self, value, dmu):
            return value * jnp.power(10.0, -0.4 * dmu)

    params = {'distance_scale': jnp.array([1e-14, 2e-14]), 'other': jnp.ones(2)}
    got = FluxScale().shift_amplitude(params, jnp.array([1.0, -0.5]))
    np.testing.assert_allclose(
        np.asarray(got['distance_scale']),
        [1e-14 * 10 ** (-0.4), 2e-14 * 10 ** (0.2)], rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(got['other']), np.ones(2))


def test_snana_output_identity_defaults():
    """A model with no SNANA counterpart: no model index, no type-name prefix
    so SIM_TYPE_NAME keeps the whole GENMODEL name, and header params that
    WRFLAG_MODELPAR does not gate."""
    m = TinyModel()
    assert m.sim_model_index == 0
    assert m.sim_type_name_prefix == ''
    assert m.writes_model_params is False


def test_compute_fitprob_chi2_survival():
    """FITPROB is the survival probability of the data chi-square at NDOF."""
    from scipy.stats import chi2 as chi2_dist
    m = TinyModel()
    for chi2, ndof in ((5.0, 4), (12.0, 7)):
        fp = float(m.compute_fitprob(chi_squared=np.array([chi2]),
                                     ndof=np.array([ndof]))[0])
        assert fp == pytest.approx(float(chi2_dist.sf(chi2, ndof)), rel=1e-3)


def test_flat_prior_chi2_shape():
    """PRIOR_<p>_RANGE/_SIGMA: flat inside the range, rolling off outside as the
    fourth power of the offset in sigma -- SNANA forms (dif/sigma)**2 and stores
    its square (FITINI_FLATPRIOR, snlc_fit.F90:12062)."""
    from jax_snlc_fit.priors import FORMS

    def chi2(value):
        return float(FORMS['flat_rolloff'](jnp.array(value),
                                           {'lo': -0.3, 'hi': 0.3,
                                            'sigma': 0.05}))

    assert chi2(0.1) == 0.0, "flat region must contribute nothing"
    below, above = chi2(-0.4), chi2(0.4)
    assert np.isclose(below, (0.1 / 0.05) ** 4) and np.isclose(above, below)
    # the fourth power is the point: it suppresses the boundary far harder
    # than a Gaussian would, so twice the offset costs sixteen times the chi2
    assert np.isclose(chi2(0.5), 16.0 * below)




class _Magnitudes(TinyModel):
    """A model whose distance parameter is a magnitude, so a shift adds."""
    distance_param = "amp"


def test_the_working_grid_is_cut_to_the_callers_range():
    """The grid covers the support cut to what the caller will evaluate, at
    the caller's resolution, so the points go where the bands are; without a
    range it covers the whole support."""
    whole = TinyModel()
    whole.n_wave_points = 100
    grid = np.asarray(whole.choose_grid())
    assert grid[0] == 3000.0 and grid[-1] == 9000.0 and len(grid) == 100

    cut = TinyModel()
    cut.n_wave_points, cut.wavelength_range = 100, (0.0, 7000.0)
    grid = np.asarray(cut.choose_grid())
    assert grid[0] == 3000.0 and grid[-1] == 7000.0 and len(grid) == 100


def test_shift_amplitude_leaves_a_model_holding_no_distance_alone():
    """NON1ASED, SIMSED and redback read the simulator's own mu, which the
    caller updates before asking. They keep no copy, so there is nothing for
    the shift to move and the parameters come back untouched."""
    params = {"amp": np.array([1.0])}
    assert TinyModel().shift_amplitude(params, 0.5) is params


def test_shift_amplitude_accepts_dicts_and_fit_vectors():
    """The base class owns the container handling so each model implements only
    the arithmetic, and the fitter never names the amplitude. Callers pass a
    name-keyed dict of physical values (vmax) or a scaled fit vector (the grid
    search), and `scales` converts the latter and back."""
    m = _Magnitudes()
    got = m.shift_amplitude({"t0": np.array([5.0]), "amp": np.array([20.0])}, 0.5)
    assert np.allclose(got["amp"], 20.5)
    assert np.allclose(got["t0"], 5.0), "other parameters must be untouched"

    # scaled fit vector: the shift is physical, so it divides back by the scale
    params = jnp.array([[5.0, 2.0]])                 # amp scaled by 10 -> 20 physical
    scales = jnp.array([[1.0, 10.0]])
    got = m.shift_amplitude(params, 0.5, scales, names=["t0", "amp"])
    assert np.allclose(np.asarray(got)[:, 1], 2.05)
    assert np.allclose(np.asarray(got)[:, 0], 5.0)


def test_a_cut_grid_leaves_the_reported_support_alone(data_root):
    """`get_wavelength_range` is the support the grid was cut from, whatever
    the caller cut it to; SALT once reported the grid, which coincided with
    the support only while nothing cut it."""
    from jax_snlc_core.models.salt_model import SALTModel

    whole = SALTModel.load(str(data_root / "models" / "SALT3.K21"), {})
    cut = SALTModel.load(str(data_root / "models" / "SALT3.K21"), {},
                    wavelength_range=(0.0, 8000.0))
    assert cut.get_wavelength_range() == whole.get_wavelength_range()
    assert float(cut.wavelength_grid[-1]) == 8000.0
    assert whole.get_wavelength_range()[1] > 8000.0


def test_absolute_and_relative_brightness_agree(data_root):
    """Setting the brightness for a distance and shifting it are one operation
    seen twice, so evaluating at mu+dmu must equal setting mu then shifting by
    dmu. The check runs on the real models because it is their encodings that
    could disagree: SALT's x0 is a flux scale (log10 of one under
    OPT_SALT2FIT=1) and BayeSN's Ds is a magnitude.

    This pins the two against each other, not the arithmetic of either. An
    error inside _amplitude_at_distance moves both sides equally and passes
    here; SNANA parity is what guards that.
    """
    from jax_snlc_core.models.salt_model import SALTModel
    from jax_snlc_core.models.bayesn_model import BayeSNModel

    salt = SALTModel.load(str(data_root / "models" / "SALT3.K21"), {})
    bayesn = BayeSNModel.load(str(data_root / "models" / "BAYESN.W22"), {})

    mu, dmu = 38.0, 0.37
    for model in (salt, bayesn):
        names = list(model.get_fit_parameters())
        # a shape and colour that are not zero, so SALT's alpha*x1 - beta*c
        # term is exercised rather than cancelling
        physical = {'PKMJD': 0.0, 'x1': 0.8, 'c': -0.12,
                    'THETA': 0.8, 'AV': 0.3, 'RV': 3.1}
        vec = jnp.array([physical.get(n, 0.0) for n in names])

        direct = model.params_at_distance(vec, mu + dmu, names=names)
        stepped = model.shift_amplitude(
            model.params_at_distance(vec, mu, names=names), dmu, names=names)
        np.testing.assert_allclose(np.asarray(direct), np.asarray(stepped),
                                   rtol=1e-10, err_msg=type(model).__name__)

        # and through the name-keyed dict container, which vmax and the
        # simulator pass
        as_dict = {n: physical.get(n, 0.0) for n in names}
        amp = model.distance_param
        direct = model.params_at_distance(as_dict, mu + dmu)[amp]
        stepped = model.shift_amplitude(
            model.params_at_distance(as_dict, mu), dmu)[amp]
        np.testing.assert_allclose(direct, stepped, rtol=1e-10,
                                   err_msg=type(model).__name__)

        # and again through the scaled fit vector, where the amplitude carries
        # a scale and the parameters it depends on must still arrive physical
        i = names.index(model.distance_param)
        scales = np.ones(len(names))
        scales[i] = 1e-5 if model is salt else 2.0
        scales = jnp.asarray(scales)
        scaled = vec / scales
        direct = model.params_at_distance(scaled, mu + dmu, scales, names)
        stepped = model.shift_amplitude(
            model.params_at_distance(scaled, mu, scales, names), dmu, scales,
            names)
        np.testing.assert_allclose(np.asarray(direct), np.asarray(stepped),
                                   rtol=1e-10, err_msg=type(model).__name__)


# ---------------------------------------------------------------------------
# uvlam_extrapflux_sedmodel: SNANA's per-surface UV extrapolation
# (UVLAM_EXTRAPFLUX_SEDMODEL, genmag_SEDtools.c:3210-3357)
# ---------------------------------------------------------------------------


def _uv_surface():
    """Two-surface fixture with different blue edges, as in SALT M0/M1.

    Grid 2000-2090 A, step 10. Surface A (M0-like) has flux above 1e-12 from
    bin 2 (2020 A). Surface B (M1-like) is negative out to bin 5 and first
    exceeds 1e-12 at bin 6 (2060 A), so its anchor sits redward of A's.
    """
    from jax_snlc_core.models.base import uvlam_extrapflux_sedmodel
    waves = 2000.0 + 10.0 * np.arange(10)
    surf_a = np.array([[0.0, 0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])
    surf_b = np.array([[-1.0, -1.0, -0.5, -0.2, 0.0, 1e-13, 2.0, 3.0, 4.0, 5.0]])
    return uvlam_extrapflux_sedmodel, waves, surf_a, surf_b


def test_uvlam_extrap_surfaces_anchor_independently():
    """Each surface finds its own per-day anchor (signed flux > 1e-12); the
    ramp overwrites original bins blueward of that anchor, so a surface with
    negative trained blue values loses them (genmag_SEDtools.c:3295-3357)."""
    fn, waves, surf_a, surf_b = _uv_surface()
    uvlam = 1950.0

    new_waves, new_a = fn(uvlam, waves, surf_a)
    _, new_b = fn(uvlam, waves, surf_b)

    # Native-grid stepping: new minimum steps down by LAMSTEP to <= uvlam.
    assert new_waves[0] == pytest.approx(1950.0)
    n_add = len(new_waves) - len(waves)
    assert n_add == 5

    # Surface A: anchor at original bin 2 (flux 3.0); ramp covers the added
    # bins plus original bins 0-1; anchor bin and redward bins unchanged.
    np.testing.assert_allclose(new_a[0, n_add + 2:], surf_a[0, 2:])
    assert new_a[0, 0] == 0.0  # ramp is zero at the new grid minimum
    assert np.all(np.diff(new_a[0, :n_add + 3]) > 0)  # monotonic rise to anchor
    # linear ramp: constant second difference up to the anchor bin
    second = np.diff(new_a[0, :n_add + 3], n=2)
    np.testing.assert_allclose(second, 0.0, atol=1e-12)
    assert new_a[0, n_add + 2] == pytest.approx(3.0)

    # Surface B: anchor at original bin 6 (2.0; the 1e-13 bin fails the
    # signed > 1e-12 test); every blueward original bin, including the
    # negative trained values, is overwritten by the non-negative ramp.
    np.testing.assert_allclose(new_b[0, n_add + 6:], surf_b[0, 6:])
    assert np.all(new_b[0, :n_add + 6] >= 0.0)
    assert new_b[0, n_add + 6] == pytest.approx(2.0)
    second_b = np.diff(new_b[0, :n_add + 7], n=2)
    np.testing.assert_allclose(second_b, 0.0, atol=1e-12)

    # The two surfaces extrapolate to different anchors: B's ramp is still
    # rising where A already holds trained values.
    assert not np.allclose(new_a[0, :n_add + 6], surf_a[0, 2] *
                           (new_waves[:n_add + 6] - new_waves[0]) /
                           (new_waves[n_add + 6] - new_waves[0]))


def test_uvlam_extrap_no_anchor_day_ramps_from_bin_zero():
    """A day whose flux never exceeds 1e-12 anchors at bin 0 with that bin's
    flux (genmag_SEDtools.c:3312-3315), filling only the added bins."""
    fn, waves, _, _ = _uv_surface()
    flat = np.full((1, len(waves)), 1e-13)
    new_waves, new_flat = fn(1950.0, waves, flat)
    n_add = len(new_waves) - len(waves)
    np.testing.assert_allclose(new_flat[0, n_add:], flat[0])
    assert new_flat[0, 0] == 0.0
    assert np.all(new_flat[0, :n_add] <= 1e-13)


def test_uvlam_extrap_inactive_inputs_unchanged():
    """uvlam < 0 and uvlam > waves[0] leave the surface untouched
    (genmag_SEDtools.c:3244-3245)."""
    fn, waves, surf_a, _ = _uv_surface()
    for uvlam in (-9.0, 2005.0):
        w, f = fn(uvlam, waves, surf_a)
        np.testing.assert_array_equal(w, waves)
        np.testing.assert_array_equal(f, surf_a)


def test_extrap_phase_method_defaults_to_sedflux():
    """SNANA's default extrapolation beyond DAYMAX is SEDFLUX (=1), which
    suppresses SED-true spectra past the support (genmag_SALT2.c:3595)."""
    assert TinyModel().extrap_phase_method == 1


def test_salt_colour_law_comes_from_the_info_file(data_root, tmp_path):
    """SNANA reads the colour law from SALT?.INFO alone, under either spelling
    of the key (genmag_SALT2.c:1253); the color_correction file is not read.
    SALT2.P18_UV2IR is the installed model whose two sources disagree."""
    from jax_snlc_core.models.salt_model import SALTModel

    src = data_root / "models" / "SALT3.K21"
    info = (src / "SALT3.INFO").read_text()
    model = SALTModel.load(str(src), {})
    np.testing.assert_allclose(
        model.colorlaw_coeffs,
        [-5.6774105513e-01, 6.8597705612e-01, -4.3325071358e-01,
         1.1666954200e-01, -1.1344238031e-02])    # SALT3.INFO:26

    def copy(name, info_text):
        # a directory per case: the loader keys its cache on the path and reads
        # the version off the name
        d = tmp_path / f"SALT3.{name}"
        d.mkdir()
        for f in src.iterdir():
            if f.name not in ("salt3_color_correction.dat", "SALT3.INFO"):
                (d / f.name).symlink_to(f.resolve())
        (d / "SALT3.INFO").write_text(info_text)
        return str(d)

    respelt = SALTModel.load(copy("RESPELT", info.replace("COLORCOR_PARAMS:",
                                                     "COLORLAW_PARAMS:")), {})
    np.testing.assert_array_equal(respelt.colorlaw_coeffs, model.colorlaw_coeffs)
    with pytest.raises(SNANAError, match="COLORLAW_PARAMS"):
        SALTModel.load(copy("NOLAW", "\n".join(l for l in info.splitlines()
                                          if not l.startswith("COLORCOR_PARAMS"))), {})


def _salt3_copy(src, tmp_path, name, info_extra="", replace=None, drop=()):
    """A SALT3.K21 copy named SALT3.<name>, with INFO lines appended, files
    replaced by ``replace`` {name: text} and files in ``drop`` left out."""
    d = tmp_path / f"SALT3.{name}"
    d.mkdir()
    for f in src.iterdir():
        filename = f.name.removesuffix(".gz")
        if filename in drop or filename in (replace or {}):
            continue
        if f.name == "SALT3.INFO":
            (d / f.name).write_text(f.read_text() + info_extra)
        else:
            (d / f.name).symlink_to(f.resolve())
    for fname, text in (replace or {}).items():
        (d / fname).write_text(text)
    return str(d)


@pytest.mark.parametrize("mu, snana_x0", [
    (34.0, 2.335608246625e-02), (38.5, 3.701689610332e-04),
    (42.0, 1.473669177037e-05)])
def test_expected_x0_matches_snana_salt2x0calc(mu, snana_x0):
    """x0 from a distance modulus is SNANA's SALT2x0calc (genmag_SALT2.c),
    compiled and run at alpha=0.16, beta=2.9, x1=1.5, c=0.11."""
    x0 = float(compute_expected_x0_scale(mu, 0.16, 2.9, 1.5, 0.11))
    assert x0 == pytest.approx(snana_x0, rel=1e-6)


def test_salt_errmap_kcor_opt_0_leaves_out_the_colour_dispersion(data_root,
                                                                 tmp_path):
    """ERRMAP_KCOR_OPT: 0 in SALT?.INFO leaves the colour dispersion unread
    (read_SALT2colorDisp, genmag_SALT2.c:1047-1050), so it is in no error
    and the fit has no off-diagonal covariance."""
    from jax_snlc_core.models.salt_model import SALTModel

    src = data_root / "models" / "SALT3.K21"
    default = SALTModel.load(str(src), {})
    off = SALTModel.load(_salt3_copy(src, tmp_path, "KCOROFF",
                                "\nERRMAP_KCOR_OPT: 0\n"), {})
    assert default.has_offdiag_covariance
    assert off.error_model['color_dispersion'] is None
    assert not off.has_offdiag_covariance


def test_salt_errmap_bad_values_abort_unless_switched_off(data_root, tmp_path):
    """A variance value outside [-10, 500] more than 9 days inside the map's
    phase range aborts (check_BADVAL_SALT2errmap, genmag_SALT2.c:1602-1760)
    unless ERRMAP_BADVAL_ABORT: 0; at the phase edge it is not judged."""
    from jax_snlc_core.models.salt_model import SALTModel

    src = data_root / "models" / "SALT3.K21"
    with open_text(src / "salt3_lc_variance_0.dat") as stream:
        rows = stream.read().splitlines()

    def with_value(phase, value):
        out = list(rows)
        i = next(k for k, r in enumerate(out) if float(r.split()[0]) == phase)
        day, lam, _ = out[i].split()
        out[i] = f"{day} {lam} {value}"
        return "\n".join(out) + "\n"

    crazy = {"salt3_lc_variance_0.dat": with_value(0.0, 900.0)}
    with pytest.raises(SNANAError, match="VAR0"):
        SALTModel.load(_salt3_copy(src, tmp_path, "CRAZY", replace=crazy), {})
    SALTModel.load(_salt3_copy(src, tmp_path, "CRAZYOK", "\nERRMAP_BADVAL_ABORT: 0\n",
                          replace=crazy), {})
    SALTModel.load(_salt3_copy(src, tmp_path, "EDGE", replace={
        "salt3_lc_variance_0.dat": with_value(-20.0, 900.0)}), {})


def test_salt_errmap_single_phase_map_is_always_judged():
    """A map with one phase is judged whatever its phase, with no day-edge
    window (check_BADVAL_SALT2errmap, genmag_SALT2.c:1733-1735)."""
    from jax_snlc_core.models.salt_model import (
        ERRMAP_VALID_ERRSCALE, ERRMAP_VALID_VARIANCE, _check_errmap_values)

    phases = np.arange(-20.0, 51.0)
    variance = ("VAR0", "var.dat", phases, np.zeros((phases.size, 2)),
                ERRMAP_VALID_VARIANCE)
    edge_scale = ("ERRSCALE", "scale.dat", np.array([-20.0]),
                  np.array([[900.0, 1.0]]), ERRMAP_VALID_ERRSCALE)
    with pytest.raises(SNANAError, match="ERRSCALE"):
        _check_errmap_values([variance, edge_scale], abort=True)


@pytest.mark.parametrize("line", ["5000.0\n", "5000.0 high\n"])
def test_salt_colour_dispersion_line_without_two_numbers_aborts(tmp_path,
                                                                line):
    """A colour-dispersion data line must hold two numbers; rd2columnFile has
    no defined reading of one that does not (sntools.c:7280-7312)."""
    from jax_snlc_core.models.salt_model import _parse_color_dispersion_file

    p = tmp_path / "salt3_color_dispersion.dat"
    p.write_text("# lam disp\n2000.0 0.1\n" + line)
    with pytest.raises(SNANAError, match="is not two numbers"):
        _parse_color_dispersion_file(p)


def test_salt_errmap_nan_is_read_as_zero(data_root, tmp_path):
    """rd_sedFlux reads a NaN value as zero (sntools.c:8520), so a NaN in an
    error map never reaches the bad-value check."""
    from jax_snlc_core.models.salt_model import SALTModel

    src = data_root / "models" / "SALT3.K21"
    with open_text(src / "salt3_lc_variance_0.dat") as stream:
        rows = stream.read().splitlines()
    i = next(k for k, r in enumerate(rows) if float(r.split()[0]) == 0.0)
    day, lam, _ = rows[i].split()
    rows[i] = f"{day} {lam} nan"
    model = SALTModel.load(_salt3_copy(src, tmp_path, "NAN", replace={
        "salt3_lc_variance_0.dat": "\n".join(rows) + "\n"}), {})
    assert np.isfinite(np.asarray(model.error_model['variance_0'])).all()


def test_salt_error_maps_are_required(data_root, tmp_path):
    """Every SALT error map is required, as rd_sedFlux and rd2columnFile
    abort on a missing file; a colour dispersion that is empty or all zero
    aborts too (genmag_SALT2.c:1072-1085)."""
    from jax_snlc_core.models.salt_model import SALTModel

    src = data_root / "models" / "SALT3.K21"
    with pytest.raises(SNANAError, match="salt3_lc_covariance_01.dat"):
        SALTModel.load(_salt3_copy(src, tmp_path, "NOCOV",
                              drop=("salt3_lc_covariance_01.dat",)), {})
    with pytest.raises(SNANAError, match="All colorDisp elements are zero"):
        SALTModel.load(_salt3_copy(src, tmp_path, "OLDDISP", replace={
            "salt3_color_dispersion.dat": "@kU 0.0218\n@kR 0.0179\n"}), {})


def test_salt_colour_dispersion_nan_aborts_and_third_surface_refused(
        data_root, tmp_path):
    """rd2columnFile keeps a NaN, which the bad-value check then counts
    (genmag_SALT2.c:1716-1727); a template_2 surface, which SNANA would read
    as x2*M2 (NSURFACE_SALT2), is refused."""
    from jax_snlc_core.models.salt_model import SALTModel

    src = data_root / "models" / "SALT3.K21"
    with open_text(src / "salt3_color_dispersion.dat") as stream:
        rows = stream.read().splitlines()
    lam = rows[5].split()[0]
    rows[5] = f"{lam} nan"
    with pytest.raises(SNANAError, match="COLOR-DISP"):
        SALTModel.load(_salt3_copy(src, tmp_path, "DISPNAN", replace={
            "salt3_color_dispersion.dat": "\n".join(rows) + "\n"}), {})
    with open_text(src / "salt3_template_1.dat") as stream:
        third_surface = stream.read()
    with pytest.raises(SNANAError, match="three-surface"):
        SALTModel.load(_salt3_copy(src, tmp_path, "THREE", replace={
            "salt3_template_2.dat": third_surface}), {})
