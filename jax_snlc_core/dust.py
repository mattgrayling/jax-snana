"""Dust: the extinction laws of SNANA's MWgaldust.c, and the Milky Way
E(B-V) map of Schlegel, Finkbeiner & Davis (1998).

Every law coefficient is taken from MWgaldust.c and MWgaldust.h. The SFD maps
($SNDATA_ROOT/MWDUST/SFD_dust_4096_{ngp,sgp}.fits) use a Lambert azimuthal
equal-area projection in galactic coordinates, evaluated as MWgaldust.c's
lambert_lb2xy and lambert_lb2fpix do, after rotating equatorial coordinates
with slaEqgal's matrix (sntools.c:4464-4472).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
from astropy.io import fits
from jax import jit

from jax_snlc_core.cosmology import eq_to_gal
from jax_snlc_core.interpolation import invkd_irr, spline_coefficients
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import resolve_fits_path, sndata_root


def _linear_coeffs_irr(x_int: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Build matrix J for linear interpolation: J @ y gives linearly interpolated y(x_int)."""
    n_x_int = len(x_int)
    n_x = len(x)
    X = np.zeros((n_x_int, n_x))

    for i in range(n_x_int):
        x_now = x_int[i]
        if x_now >= x[-1]:
            X[i, -1] = 1.0
        elif x_now <= x[0]:
            X[i, 0] = 1.0
        else:
            q = np.where(x[0:-1] <= x_now)[0][-1]
            h = x[q + 1] - x[q]
            a = (x[q + 1] - x_now) / h
            b = 1.0 - a
            X[i, q] = a
            X[i, q + 1] = b

    return X


# MWgaldust.h
WAVE_V_ANGSTROM = 5495.0

WAVEMIN_FITZ99_EXACT = 912.0
WAVEMAX_FITZ99_EXACT = 35000.0
WAVEMAX_FITZ99 = 25000.0       # approx version
WAVEMIN_GOOB08 = 2000.0
WAVEMAX_GOOB08 = 22000.0
WAVEMIN_MAIZ14 = 3000.0
WAVEMAX_MAIZ14 = 35000.0
WAVEMIN_FITZ19 = 1150.0
WAVEMAX_FITZ19 = 35000.0
WAVEMIN_GORD23 = 912.0
WAVEMAX_GORD23 = 320000.0
WAVEMIN_SOMM25 = 700.0
WAVEMAX_SOMM25 = 9134.0
WAVEMIN_SOMM26 = 1000.0
WAVEMAX_SOMM26 = 10000.0

# SNANA OPT_MWCOLORLAW integer -> DUST_LAWS name (shared by sim + fit config)
OPT_MWCOLORLAW_NAMES = {
    0: "OFF",
    89: "CCM89",
    94: "ODONNELL94",
    -99: "FITZ99_APPROX",
    99: "FITZ99_EXACT",
    203: "GORD03",
    204: "FITZ04",
    208: "GOOBAR08",
    214: "MAIZ14",
    216: "GORD16",
    -219: "FITZ19_LIN",
    219: "FITZ19",
    223: "GORD23",
    225: "SOMM25",
    226: "SOMM26",
}


def _dust_spline_matrix(x_eval, knots, invkd):
    """The natural cubic spline matrix from ``knots`` to the inverse-micron
    points ``x_eval``, linear beyond the knots (:func:`spline_coefficients`
    with mode 0). Built in double precision whatever frame the run is in,
    like the knots' :func:`invkd_irr`, and rounded once to the run's
    precision."""
    with jax.enable_x64(True):
        M = np.asarray(spline_coefficients(jnp.asarray(x_eval, dtype=jnp.float64),
                                           jnp.asarray(knots, dtype=jnp.float64),
                                           jnp.asarray(invkd, dtype=jnp.float64), 0))
    return jnp.asarray(M)


def _compute_fm90(x, c1, c2, c3, c4, c5, x02, g2):
    """FM90 parameterisation: E(x-V)/E(B-V)
    (GALextinct_FM90, MWgaldust.c:1263-1296)."""
    x2 = x * x
    denom = (x2 - x02) ** 2 + x2 * g2
    bump = x2 / denom
    k = c1 + c2 * x + c3 * bump
    y = x - c5
    y2 = y * y
    k = jnp.where(x >= c5, k + c4 * (0.5392 * y2 + 0.05644 * y2 * y), k)
    return k


class DustLaw(ABC):
    """Abstract base class for dust law implementations."""

    # Override when A(lambda)/A(V) itself depends on AV.
    linear_in_av = True

    def __init__(self, name: str):
        self.name = name

    # Only coefficients used during evaluation cross the JIT boundary.
    # Reader tables and spline preparation stay on the Python side.
    numerical_fields = ()

    def tree_flatten(self):
        return (tuple(getattr(self, name) for name in self.numerical_fields),
                (self.name,))

    @classmethod
    def tree_unflatten(cls, metadata, coefficients):
        law = cls.__new__(cls)
        (law.name,) = metadata
        for name, value in zip(cls.numerical_fields, coefficients):
            setattr(law, name, value)
        return law

    @abstractmethod
    def extinction_curve(self, wavelength: jnp.ndarray, rv: float, **kwargs) -> jnp.ndarray:
        """Compute A(λ)/A(V) extinction curve."""
        pass

    def set_params(self, parlist):
        """PARLIST_MWCOLORLAW, which only the parametrised laws read."""

    # (min, max) validity in Angstroms; SNANA fatals outside it
    # (e.g. GALextinct_Maiz14, MWgaldust.c:868-873). None means unrestricted.
    wave_range = None

    def check_rv(self, rv: float) -> None:
        """Validate a configured R_V before entering compiled evaluation."""

    def check_wave_range(self, wave_min: float, wave_max: float) -> None:
        """Abort when [wave_min, wave_max] leaves the law's validity range."""
        if self.wave_range is None:
            return
        lo, hi = self.wave_range
        if wave_min < lo or wave_max > hi:
            raise SNANAError(
                f"{self.name} is only valid from {lo:.0f} to {hi:.0f} A; "
                f"requested {wave_min:.0f}-{wave_max:.0f} A.",
                function_name="check_wave_range")

    def apply_extinction(self, flux: jnp.ndarray, wavelength: jnp.ndarray,
                        av: float, rv: float, **kwargs) -> jnp.ndarray:
        """Apply dust extinction to flux array.

        ``av`` reaches the law, for the laws whose curve shape depends on it
        (GALextinct_Somm25, MWgaldust.c:1188-1193)."""
        extinction_curve = self.extinction_curve(wavelength, rv, av=av,
                                                 **kwargs)
        extinction_mag = av * extinction_curve
        extinction_factor = jnp.power(10.0, -0.4 * extinction_mag)
        return flux * extinction_factor

    def prepare_wavelength_grid(self, wavelength_grid: np.ndarray):
        """Optional precomputation hook."""


def _ccm_uv_fuv(x):
    """CCM89 UV and FUV a, b curves, with the fa/fb bump terms and the FUV
    polynomial (GALextinct, MWgaldust.c:611-640)."""
    xx59 = x - 5.9
    fa = jnp.where(x >= 5.9, -0.04473 * xx59**2 - 0.009779 * xx59**3, 0.0)
    fb = jnp.where(x >= 5.9,  0.21300 * xx59**2 + 0.120700 * xx59**3, 0.0)

    uv_a = 1.752 - 0.316 * x - 0.104 / ((x - 4.67)**2 + 0.341) + fa
    uv_b = -3.090 + 1.825 * x + 1.206 / ((x - 4.62)**2 + 0.263) + fb

    xx8 = x - 8.0
    fuv_a = -1.073 - 0.628 * xx8 + 0.137 * xx8**2 - 0.070 * xx8**3
    fuv_b = 13.670 + 4.257 * xx8 - 0.420 * xx8**2 + 0.374 * xx8**3

    # FUV takes over at x = 8.0 (MWgaldust.c:630)
    a = jnp.where(x >= 8.0, fuv_a, uv_a)
    b = jnp.where(x >= 8.0, fuv_b, uv_b)
    return a, b


@jax.tree_util.register_pytree_node_class
class NoExtinctionDustLaw(DustLaw):
    """No Milky Way extinction at any wavelength."""

    numerical_fields = ()

    def __init__(self):
        super().__init__("NoExtinction")

    def extinction_curve(self, wavelength: jnp.ndarray, rv: float, **kwargs) -> jnp.ndarray:
        return jnp.zeros(jnp.shape(wavelength))


@jax.tree_util.register_pytree_node_class
class CCM89DustLaw(DustLaw):
    """Cardelli, Clayton, & Mathis (1989) extinction law."""

    numerical_fields = ()

    def __init__(self):
        super().__init__("CCM89")

    def extinction_curve(self, wavelength: jnp.ndarray, rv: float, **kwargs) -> jnp.ndarray:
        x = 10000.0 / wavelength
        a = jnp.zeros_like(x)
        b = jnp.zeros_like(x)

        ir_mask = (x >= 0.3) & (x < 1.1)
        a = jnp.where(ir_mask, 0.574 * jnp.power(x, 1.61), a)
        b = jnp.where(ir_mask, -0.527 * jnp.power(x, 1.61), b)

        opt_mask = (x >= 1.1) & (x < 3.3)
        y = x - 1.82
        opt_a = (1.0 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4
                 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7)
        opt_b = (1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4
                 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7)
        a = jnp.where(opt_mask, opt_a, a)
        b = jnp.where(opt_mask, opt_b, b)

        uv_fuv_mask = (x >= 3.3) & (x <= 10.0)
        uv_a, uv_b = _ccm_uv_fuv(x)
        a = jnp.where(uv_fuv_mask, uv_a, a)
        b = jnp.where(uv_fuv_mask, uv_b, b)

        return a + b / rv


@jax.tree_util.register_pytree_node_class
class ODonnell94DustLaw(DustLaw):
    """O'Donnell (1994) update to CCM89 (updated optical coefficients)."""

    numerical_fields = ()

    def __init__(self):
        super().__init__("ODonnell94")

    def extinction_curve(self, wavelength: jnp.ndarray, rv: float, **kwargs) -> jnp.ndarray:
        x = 10000.0 / wavelength
        a = jnp.zeros_like(x)
        b = jnp.zeros_like(x)

        ir_mask = (x >= 0.3) & (x < 1.1)
        a = jnp.where(ir_mask, 0.574 * jnp.power(x, 1.61), a)
        b = jnp.where(ir_mask, -0.527 * jnp.power(x, 1.61), b)

        opt_mask = (x >= 1.1) & (x < 3.3)
        y = x - 1.82
        opt_a = (1.0 + 0.104*y - 0.609*y**2 + 0.701*y**3 + 1.137*y**4
                 - 1.718*y**5 - 0.827*y**6 + 1.647*y**7 - 0.505*y**8)
        opt_b = (1.952*y + 2.908*y**2 - 3.989*y**3 - 7.985*y**4
                 + 11.102*y**5 + 5.491*y**6 - 10.805*y**7 + 3.347*y**8)
        a = jnp.where(opt_mask, opt_a, a)
        b = jnp.where(opt_mask, opt_b, b)

        uv_fuv_mask = (x >= 3.3) & (x <= 10.0)
        uv_a, uv_b = _ccm_uv_fuv(x)
        a = jnp.where(uv_fuv_mask, uv_a, a)
        b = jnp.where(uv_fuv_mask, uv_b, b)

        return a + b / rv


@jax.tree_util.register_pytree_node_class
class Fitzpatrick99ExactDustLaw(DustLaw):
    """Fitzpatrick (1999) exact implementation using cubic splines
    (GALextinct_Fitz99_exact, MWgaldust.c:689-850)."""

    numerical_fields = ('_prep',)

    wave_range = (WAVEMIN_FITZ99_EXACT, WAVEMAX_FITZ99_EXACT)

    def __init__(self):
        super().__init__("Fitzpatrick99_Exact")
        self._xk_np = np.array([
            0.0, 1e4/26500.0, 1e4/12200.0, 1e4/6000.0, 1e4/5470.0,
            1e4/4670.0, 1e4/4110.0, 1e4/2700.0, 1e4/2600.0
        ])
        self._invkd_np = invkd_irr(self._xk_np)
        self._f99_x0 = 4.596
        self._f99_gamma = 0.99
        self._f99_c3 = 3.23
        self._f99_c4 = 0.41
        self._f99_c5 = 5.9
        self._prep = None

    def prepare_wavelength_grid(self, wavelength_grid: np.ndarray):
        wave_np = np.array(wavelength_grid, dtype=np.float64)
        M = _dust_spline_matrix(1e4 / wave_np, self._xk_np, self._invkd_np)
        uv_mask = wave_np < 2700.0
        uv_mask_fuv = uv_mask & ((1e4 / wave_np) >= self._f99_c5)
        fuv_in_uv = (1e4 / wave_np[uv_mask]) >= self._f99_c5
        self._prep = {
            'M': M,
            # scatter targets on the full grid; FUV gather positions within the UV subset
            'uv_indices': jnp.array(np.where(uv_mask)[0], dtype=jnp.int32),
            'uv_fuv_indices': jnp.array(np.where(uv_mask_fuv)[0], dtype=jnp.int32),
            'uv_fuv_in_uv': jnp.array(np.where(fuv_in_uv)[0], dtype=jnp.int32),
            'uv_ind3': jnp.array(fuv_in_uv, dtype=bool),
            'uv_x': jnp.array(1e4 / wave_np[uv_mask]),
            'xk': jnp.array(self._xk_np),
            'x0': float(self._f99_x0),
            'gamma': float(self._f99_gamma),
            'c3': float(self._f99_c3),
            'c4': float(self._f99_c4),
            'c5': float(self._f99_c5),
        }

    def extinction_curve(self, wavelength: jnp.ndarray, rv: jnp.ndarray, **kwargs) -> jnp.ndarray:
        rv_arr = jnp.atleast_1d(rv)
        axav = _compute_fitz99_from_prepared_internal(rv_arr, self._prep)
        return axav[0] if rv_arr.shape == (1,) else axav


@jit
def _compute_fitz99_from_prepared_internal(rv: jnp.ndarray, prep: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Compute F99 A(λ)/A(V) using precomputed spline matrix and UV masks."""
    rv = jnp.atleast_1d(rv)
    rv_shape = rv.shape
    rv_exp = rv[..., None]

    M = prep['M']
    uv_idx = prep['uv_indices']
    uv_fuv_idx = prep['uv_fuv_indices']
    uv_x = prep['uv_x']
    xk = prep['xk']
    x0 = prep['x0']
    gamma = prep['gamma']
    c3 = prep['c3']
    c4 = prep['c4']
    c5 = prep['c5']

    c2 = -0.824 + 4.717 / rv_exp
    c1 = 2.030 - 3.007 * c2
    d1 = xk[7] ** 2 / ((xk[7] ** 2 - x0 ** 2) ** 2 + (gamma * xk[7]) ** 2)
    d2 = xk[8] ** 2 / ((xk[8] ** 2 - x0 ** 2) ** 2 + (gamma * xk[8]) ** 2)

    yk = jnp.zeros(rv.shape + (9,))
    yk = yk.at[..., 0].set(-rv)
    yk = yk.at[..., 1].set(0.26469 * rv / 3.1 - rv)
    yk = yk.at[..., 2].set(0.82925 * rv / 3.1 - rv)
    yk = yk.at[..., 3].set(-0.422809 + 1.00270 * rv + 2.13572e-4 * rv ** 2 - rv)
    yk = yk.at[..., 4].set(-5.13540e-2 + 1.00216 * rv - 7.35778e-5 * rv ** 2 - rv)
    yk = yk.at[..., 5].set(0.700127 + 1.00184 * rv - 3.32598e-5 * rv ** 2 - rv)
    yk = yk.at[..., 6].set(
        1.19456 + 1.01707 * rv - 5.46959e-3 * rv ** 2 + 7.97809e-4 * rv ** 3 - 4.45636e-5 * rv ** 4 - rv
    )
    yk = yk.at[..., 7].set(c1[..., 0] + c2[..., 0] * xk[7] + c3 * d1)
    yk = yk.at[..., 8].set(c1[..., 0] + c2[..., 0] * xk[8] + c3 * d2)

    A_norm = 1.0 + jnp.einsum('wj,...j->...w', M, yk) / rv_exp

    if uv_idx.size > 0:
        x2 = uv_x * uv_x
        y = x2 - x0 * x0
        d = x2 / (y * y + x2 * gamma * gamma)
        k = c1 + c2 * uv_x + c3 * d

        A_uv = 1.0 + k / rv_exp
        A_norm = A_norm.at[..., uv_idx].set(A_uv)

        if uv_fuv_idx.size > 0:
            y_fuv = uv_x - c5
            y2_fuv = y_fuv * y_fuv
            k_fuv = k + c4 * (0.5392 * y2_fuv + 0.05644 * y2_fuv * y_fuv)
            A_fuv_all = 1.0 + k_fuv / rv_exp
            A_fuv_sel = jnp.take(A_fuv_all, prep['uv_fuv_in_uv'], axis=-1)
            A_norm = A_norm.at[..., uv_fuv_idx].set(A_fuv_sel)

    return A_norm.reshape(rv_shape + (M.shape[0],))


@jax.tree_util.register_pytree_node_class
class Fitzpatrick99ApproxDustLaw(DustLaw):
    """Deprecated F99 approximation: O'Donnell94 * polynomial correction
    (MWgaldust.c:646-681)."""

    numerical_fields = ('_o94', '_coeffs')

    _F99_OVER_O94 = np.array([
        8.55929205e-02, 1.91547833e+00, -1.65101945e+00, 7.50611119e-01,
        -2.00041118e-01, 3.30155576e-02, -3.46344458e-03, 2.30741420e-04,
        -9.43018242e-06, 2.14917977e-07, -2.08276810e-09
    ])

    def __init__(self):
        super().__init__("Fitzpatrick99_Approx")
        self._o94 = ODonnell94DustLaw()
        self._coeffs = jnp.array(self._F99_OVER_O94)

    def extinction_curve(self, wavelength: jnp.ndarray, rv: float, **kwargs) -> jnp.ndarray:
        # SNANA aborts above WAVEMAX_FITZ99 (MWgaldust.c:655-661); checked on concrete grids
        if not isinstance(wavelength, jax.core.Tracer):
            wmax = float(np.max(np.asarray(wavelength)))
            if wmax > WAVEMAX_FITZ99:
                raise SNANAError(
                    f"Invalid WAVE={wmax:.1f} A for Fitzpatrick 99 colour law; SNANA "
                    f"aborts above {WAVEMAX_FITZ99:.1f} A.",
                    function_name="Fitzpatrick99ApproxDustLaw.extinction_curve")
        xt_o94 = self._o94.extinction_curve(wavelength, rv)
        wk = wavelength / 1000.0
        wpow = jnp.stack([jnp.power(wk, i) for i in range(11)], axis=-1)
        correction = wpow @ self._coeffs
        return xt_o94 * correction


@jax.tree_util.register_pytree_node_class
class Gordon03DustLaw(DustLaw):
    """Gordon et al. (2003) SMC bar dust law, fixed RV=2.74
    (GALextinct_Fitz99_exact, MWgaldust.c:750-757)."""

    numerical_fields = ('_prep', '_rv', '_c1', '_c2', '_c3', '_c4', '_c5', '_x02', '_g2')

    wave_range = (WAVEMIN_FITZ99_EXACT, WAVEMAX_FITZ99_EXACT)

    def __init__(self):
        super().__init__("Gordon03")
        self._rv = 2.74
        self._xk_np = np.array([
            0.0, 1.0/2.198, 1.0/1.65, 1.0/1.25, 1.0/0.81,
            1.0/0.65, 1.0/0.55, 1.0/0.44, 1.0/0.37, 1.0/0.270, 1.0/0.260
        ])
        self._c1 = -4.959
        self._c2 = 2.264
        self._c3 = 0.389
        self._c4 = 0.461
        self._c5 = 5.90
        self._x02 = 21.16      # 4.6^2
        self._g2 = 1.0
        rv = self._rv
        fm9 = lambda xx: float(_compute_fm90(
            jnp.array([xx]), self._c1, self._c2, self._c3, self._c4,
            self._c5, self._x02, self._g2)[0])
        self._yk_np = np.array([
            -rv, 0.11*rv - rv, 0.169*rv - rv, 0.25*rv - rv,
            0.567*rv - rv, 0.801*rv - rv, 0.0, 1.374*rv - rv,
            1.672*rv - rv, fm9(self._xk_np[9]), fm9(self._xk_np[10])
        ])
        self._invkd_np = invkd_irr(self._xk_np)
        self._prep = None

    def prepare_wavelength_grid(self, wavelength_grid: np.ndarray):
        wave_np = np.array(wavelength_grid, dtype=np.float64)
        x_eval = 1e4 / wave_np
        M = _dust_spline_matrix(x_eval, self._xk_np, self._invkd_np)
        uv_mask = wave_np < 2700.0
        self._prep = {
            'M': M,
            'yk': jnp.array(self._yk_np),
            'uv_mask': jnp.array(uv_mask),
            'uv_x': jnp.array(x_eval[uv_mask]) if uv_mask.any() else jnp.zeros(0),
            'uv_indices': jnp.array(np.where(uv_mask)[0], dtype=jnp.int32),
        }

    def check_rv(self, rv: float) -> None:
        if abs(float(rv) - 2.74) > 1e-9:
            raise SNANAError(
                f"Gordon et al. 2003 is only valid for RV=2.74; got RV={rv:.3f}.",
                function_name="check_rv")

    def extinction_curve(self, wavelength: jnp.ndarray, rv: float, **kwargs) -> jnp.ndarray:
        if not isinstance(rv, jax.core.Tracer):
            self.check_rv(rv)
        rv_use = self._rv
        M = self._prep['M']
        yk = self._prep['yk']
        uv_idx = self._prep['uv_indices']
        uv_x = self._prep['uv_x']

        y_spline = M @ yk
        result = 1.0 + y_spline / rv_use

        if uv_idx.size > 0:
            k_uv = _compute_fm90(uv_x, self._c1, self._c2, self._c3, self._c4,
                                 self._c5, self._x02, self._g2)
            result = result.at[uv_idx].set(1.0 + k_uv / rv_use)

        return result


@jax.tree_util.register_pytree_node_class
class Fitzpatrick04DustLaw(DustLaw):
    """Fitzpatrick (2004) with FM07 IR knots and updated UV constants
    (GALextinct_Fitz99_exact, MWgaldust.c:743-749)."""

    numerical_fields = ('_prep',)

    wave_range = (WAVEMIN_FITZ99_EXACT, WAVEMAX_FITZ99_EXACT)

    def __init__(self):
        super().__init__("Fitzpatrick04")
        self._xk_np = np.array([
            0.0, 0.5, 0.75, 1.0,
            1.0/0.60, 1.0/0.547, 1.0/0.467, 1.0/0.411,
            1.0/0.270, 1.0/0.260
        ])
        self._invkd_np = invkd_irr(self._xk_np)
        self._c3 = 2.991
        self._c4 = 0.319
        self._c5 = 5.90
        self._x02 = 21.086464   # 4.592^2
        self._g2 = 0.850084     # 0.922^2
        self._prep = None

    def prepare_wavelength_grid(self, wavelength_grid: np.ndarray):
        wave_np = np.array(wavelength_grid, dtype=np.float64)
        x_eval = 1e4 / wave_np
        M = _dust_spline_matrix(x_eval, self._xk_np, self._invkd_np)
        uv_mask = wave_np < 2700.0
        self._prep = {
            'M': M,
            'xk': jnp.array(self._xk_np),
            'uv_x': jnp.array(x_eval[uv_mask]) if uv_mask.any() else jnp.zeros(0),
            'uv_indices': jnp.array(np.where(uv_mask)[0], dtype=jnp.int32),
            'c3': float(self._c3),
            'c4': float(self._c4),
            'c5': float(self._c5),
            'x02': float(self._x02),
            'g2': float(self._g2),
        }

    def extinction_curve(self, wavelength: jnp.ndarray, rv: float, **kwargs) -> jnp.ndarray:

        M = self._prep['M']
        xk = self._prep['xk']
        uv_idx = self._prep['uv_indices']
        uv_x = self._prep['uv_x']
        c3 = self._prep['c3']
        c4 = self._prep['c4']
        c5 = self._prep['c5']
        x02 = self._prep['x02']
        g2 = self._prep['g2']

        c2 = -0.824 + 4.717 / rv
        c1 = 2.18 - 2.91 * c2

        d8 = xk[8]**2 / ((xk[8]**2 - x02)**2 + xk[8]**2 * g2)
        d9 = xk[9]**2 / ((xk[9]**2 - x02)**2 + xk[9]**2 * g2)
        k8 = c1 + c2 * xk[8] + c3 * d8
        k9 = c1 + c2 * xk[9] + c3 * d9

        yFNIR = 0.63 * rv - 0.84
        yk = jnp.array([
            -rv,
            yFNIR * 0.5**1.84 - rv,
            yFNIR * 0.75**1.84 - rv,
            yFNIR * 1.0**1.84 - rv,
            -0.422809 + 1.00270 * rv + 2.13572e-4 * rv**2 - rv,
            -5.13540e-2 + 1.00216 * rv - 7.35778e-5 * rv**2 - rv,
            7.00127e-1 + 1.00184 * rv - 3.32598e-5 * rv**2 - rv,
            1.19456 + 1.01707 * rv - 5.46959e-3 * rv**2 + 7.97809e-4 * rv**3 - 4.45636e-5 * rv**4 - rv,
            k8,
            k9,
        ])

        y_spline = M @ yk
        result = 1.0 + y_spline / rv

        if uv_idx.size > 0:
            k_uv = _compute_fm90(uv_x, c1, c2, c3, c4, c5, x02, g2)
            result = result.at[uv_idx].set(1.0 + k_uv / rv)

        return result


@jax.tree_util.register_pytree_node_class
class Goobar08DustLaw(DustLaw):
    """Goobar (2008) power law for circumstellar dust (MWgaldust.c:485-518)."""

    numerical_fields = ('_p', '_a')

    wave_range = (WAVEMIN_GOOB08, WAVEMAX_GOOB08)

    def __init__(self):
        super().__init__("Goobar08")
        self._p = None
        self._a = None

    def set_params(self, parlist):
        """P and A from PARLIST_MWCOLORLAW (MWgaldust.c:487-507)."""
        if len(parlist) < 2:
            raise SNANAError(
                "Goobar (2008) requires two values in PARLIST_MWCOLORLAW: P,A.",
                function_name="set_params")
        p, a = float(parlist[0]), float(parlist[1])
        if not -2.5 <= p <= -0.5:
            raise SNANAError(f"insane P = {p:.3f}; valid range is [-2.5, -0.5].",
                             function_name="set_params")
        if not 0.0 < a <= 1.0:
            raise SNANAError(f"insane A = {a:.3f}; valid range is (0, 1].",
                             function_name="set_params")
        self._p, self._a = p, a

    def extinction_curve(self, wavelength: jnp.ndarray, rv: float, **kwargs) -> jnp.ndarray:
        """Returns A(λ)/A(V) = 1 - A + A*(λ/5495)^P."""
        if self._p is None:
            raise SNANAError(
                "Goobar (2008) requires PARLIST_MWCOLORLAW: P,A.",
                function_name="extinction_curve")
        return 1.0 - self._a + self._a * jnp.power(
            wavelength / WAVE_V_ANGSTROM, self._p)


@jax.tree_util.register_pytree_node_class
class MaizApellaniz14DustLaw(DustLaw):
    """Maiz Apellaniz et al. (2014) piecewise cubic extinction law
    (GALextinct_Maiz14, MWgaldust.c:852-940)."""

    numerical_fields = ('_xk', '_a3', '_a2', '_a1', '_a0', '_b3', '_b2', '_b1', '_b0')

    wave_range = (WAVEMIN_MAIZ14, WAVEMAX_MAIZ14)

    _XK = np.array([1.0, 1.15, 1.81984, 2.1, 2.27015, 2.7, 3.5, 3.9, 4.0, 4.1, 4.2])

    _A3 = np.array([-3.09348541, 2.28902153e-1, 5.41605406e-1, -6.37404842e-1,
                    3.52950213e-1, -5.91231605e-2, -5.56727269, 48.1384135,
                    -11.6556097, -12.6892172])
    _A2 = np.array([5.57088021e-1, -8.34980412e-1, -3.74996957e-1, 8.02115549e-2,
                    -2.45151747e-1, 2.09995201e-1, 6.80996157e-2, -6.61262761,
                    7.82889643, 4.33221353])
    _A1 = np.array([9.24140000e-1, 8.82456141e-1, 7.19649009e-2, -1.06221772e-2,
                    -3.86867508e-2, -5.37987921e-2, 1.68677061e-1, -2.44913414,
                    -2.32750725, -1.11139626])
    _A0 = np.array([5.74000000e-1, 7.14714967e-1, 9.99971669e-1, 1.00260970,
                    9.99984676e-1, 9.66090893e-1, 1.02717773, 7.49239041e-1,
                    4.86337764e-1, 3.20220393e-1])

    _B3 = np.array([6.11543973, -4.71924979e-1, -3.75700076, 3.30710701,
                    -6.80610047e-1, 4.81511488e-1, 17.8352808, -124.325934,
                    12.0120271, 48.1516935])
    _B2 = np.array([-2.49479124e-1, 2.50246875, 1.55412607, -1.60355793,
                    8.45548471e-2, -7.93125839e-1, 3.62501733e-1, 21.7648387,
                    -15.5329415, -11.9293334])
    _B1 = np.array([-8.48470000e-1, -5.10521556e-1, 2.20674792, 2.19289909,
                    1.93444072, 1.62986148, 1.28536219, 10.1362984,
                    10.7594881, 8.01326059])
    _B0 = np.array([-5.27000000e-1, -6.39244171e-1, -2.26082358e-4, 6.57384043e-1,
                    1.00037205, 1.79345802, 2.83628055, 4.54988367,
                    5.65683596, 6.58946738])

    def __init__(self):
        super().__init__("MaizApellaniz14")
        self._xk = jnp.array(self._XK)
        self._a3 = jnp.array(self._A3)
        self._a2 = jnp.array(self._A2)
        self._a1 = jnp.array(self._A1)
        self._a0 = jnp.array(self._A0)
        self._b3 = jnp.array(self._B3)
        self._b2 = jnp.array(self._B2)
        self._b1 = jnp.array(self._B1)
        self._b0 = jnp.array(self._B0)

    def extinction_curve(self, wavelength: jnp.ndarray, rv: float, **kwargs) -> jnp.ndarray:
        valid_mask = (wavelength >= WAVEMIN_MAIZ14) & (wavelength <= WAVEMAX_MAIZ14)
        x = 10000.0 / wavelength

        ir_mask = x < 1.0
        a_ir = 0.574 * jnp.power(x, 1.61)
        b_ir = -0.527 * jnp.power(x, 1.61)

        q = jnp.searchsorted(self._xk, x) - 1
        q = jnp.clip(q, 0, 9)
        dx = x - self._xk[q]
        dx2 = dx * dx
        dx3 = dx2 * dx

        a_spline = self._a3[q] * dx3 + self._a2[q] * dx2 + self._a1[q] * dx + self._a0[q]
        b_spline = self._b3[q] * dx3 + self._b2[q] * dx2 + self._b1[q] * dx + self._b0[q]

        a = jnp.where(ir_mask, a_ir, a_spline)
        b = jnp.where(ir_mask, b_ir, b_spline)

        return jnp.where(valid_mask, a + b / rv, 0.0)


# GALextinct_Fitz19, MWgaldust.c:975-1008
_FITZ19_XK = np.array([
    0.000, 0.455, 0.606, 0.800, 1.000, 1.100, 1.200, 1.250,
    1.300, 1.350, 1.400, 1.450, 1.500, 1.550, 1.600, 1.650, 1.700, 1.750,
    1.800, 1.818, 1.850, 1.900, 1.950, 2.000, 2.050, 2.100, 2.150, 2.200,
    2.250, 2.273, 2.300, 2.350, 2.400, 2.450, 2.500, 2.550, 2.600, 2.650,
    2.700, 2.750, 2.800, 2.850, 2.900, 2.950, 3.000, 3.100, 3.200, 3.300,
    3.400, 3.500, 3.600, 3.700, 3.800, 3.900, 4.000, 4.100, 4.200, 4.300,
    4.400, 4.500, 4.600, 4.700, 4.800, 4.900, 5.000, 5.100, 5.200, 5.300,
    5.400, 5.500, 5.600, 5.700, 5.800, 5.900, 6.000, 6.100, 6.200, 6.300,
    6.400, 6.500, 6.600, 6.700, 6.800, 6.900, 7.000, 7.100, 7.200, 7.300,
    7.400, 7.500, 7.600, 7.700, 7.800, 7.900, 8.000, 8.100, 8.200, 8.300,
    8.400, 8.500, 8.600, 8.700
])

_FITZ19_K302K = np.array([
    -3.020, -2.747, -2.528, -2.222, -1.757, -1.567, -1.300,
    -1.216, -1.070, -0.973, -0.868, -0.750, -0.629, -0.509, -0.407, -0.320,
    -0.221, -0.133, -0.048, 0.000, 0.071, 0.188, 0.319, 0.438, 0.575, 0.665,
    0.744, 0.838, 0.951, 1.000, 1.044, 1.113, 1.181, 1.269, 1.346, 1.405,
    1.476, 1.558, 1.632, 1.723, 1.791, 1.869, 1.948, 2.009, 2.090, 2.253,
    2.408, 2.565, 2.746, 2.933, 3.124, 3.328, 3.550, 3.815, 4.139, 4.534,
    5.012, 5.560, 6.118, 6.565, 6.767, 6.681, 6.394, 6.038, 5.704, 5.432,
    5.226, 5.078, 4.978, 4.913, 4.877, 4.862, 4.864, 4.879, 4.904, 4.938,
    4.982, 5.038, 5.105, 5.181, 5.266, 5.359, 5.460, 5.569, 5.684, 5.805,
    5.933, 6.067, 6.207, 6.352, 6.502, 6.657, 6.817, 6.981, 7.150, 7.323,
    7.500, 7.681, 7.866, 8.054, 8.246, 8.441
])

_FITZ19_SK = np.array([
    -1.000, -0.842, -0.728, -0.531, -0.360, -0.284, -0.223,
    -0.198, -0.173, -0.150, -0.130, -0.110, -0.096, -0.081, -0.063, -0.048,
    -0.032, -0.017, -0.005, 0.000, 0.007, 0.013, 0.012, 0.010, 0.004, 0.003,
    0.000, 0.002, 0.001, 0.000, -0.000, 0.001, 0.001, -0.002, 0.000, -0.002,
    -0.002, -0.006, -0.009, -0.011, -0.017, -0.025, -0.029, -0.037, -0.043,
    -0.064, -0.092, -0.122, -0.161, -0.201, -0.249, -0.303, -0.366, -0.437,
    -0.517, -0.603, -0.692, -0.774, -0.843, -0.888, -0.908, -0.903, -0.880,
    -0.849, -0.816, -0.785, -0.760, -0.741, -0.729, -0.722, -0.722, -0.726,
    -0.734, -0.745, -0.760, -0.778, -0.798, -0.820, -0.845, -0.870, -0.898,
    -0.926, -0.956, -0.988, -1.020, -1.053, -1.087, -1.122, -1.158, -1.195,
    -1.232, -1.270, -1.309, -1.349, -1.389, -1.429, -1.471, -1.513, -1.555,
    -1.598, -1.641, -1.685
])


@jax.tree_util.register_pytree_node_class
class Fitzpatrick19DustLaw(DustLaw):
    """Fitzpatrick et al. (2019) from 102 tabulated values with RV correction. Cubic spline."""

    numerical_fields = ('_prep',)

    wave_range = (WAVEMIN_FITZ19, WAVEMAX_FITZ19)

    def __init__(self):
        super().__init__("Fitzpatrick19")
        self._invkd_np = invkd_irr(_FITZ19_XK)
        self._prep = None

    def prepare_wavelength_grid(self, wavelength_grid: np.ndarray):
        wave_np = np.array(wavelength_grid, dtype=np.float64)
        x_eval = 1e4 / wave_np
        M = _dust_spline_matrix(x_eval, _FITZ19_XK, self._invkd_np)
        self._prep = {
            'M': M,
            'k302k': jnp.array(_FITZ19_K302K),
            'sk': jnp.array(_FITZ19_SK),
        }

    def extinction_curve(self, wavelength: jnp.ndarray, rv: float, **kwargs) -> jnp.ndarray:
        M = self._prep['M']
        k302k = self._prep['k302k']
        sk = self._prep['sk']
        kRVk = k302k + sk * (rv - 3.10) * 0.99
        y = M @ kRVk
        return 1.0 + y / rv


@jax.tree_util.register_pytree_node_class
class Fitzpatrick19LinearDustLaw(DustLaw):
    """Fitzpatrick et al. (2019) with linear interpolation (OPT=-219)."""

    numerical_fields = ('_prep',)

    wave_range = (WAVEMIN_FITZ19, WAVEMAX_FITZ19)

    def __init__(self):
        super().__init__("Fitzpatrick19_Linear")
        self._prep = None

    def prepare_wavelength_grid(self, wavelength_grid: np.ndarray):
        wave_np = np.array(wavelength_grid, dtype=np.float64)
        x_eval = 1e4 / wave_np
        M = jnp.array(_linear_coeffs_irr(x_eval, _FITZ19_XK))
        self._prep = {
            'M': M,
            'k302k': jnp.array(_FITZ19_K302K),
            'sk': jnp.array(_FITZ19_SK),
        }

    def extinction_curve(self, wavelength: jnp.ndarray, rv: float, **kwargs) -> jnp.ndarray:
        M = self._prep['M']
        k302k = self._prep['k302k']
        sk = self._prep['sk']
        kRVk = k302k + sk * (rv - 3.10) * 0.99
        y = M @ kRVk
        return 1.0 + y / rv


def _smoothstep(f):
    """Cubic smoothstep: (3 - 2f)*f^2, clamped to [0,1]."""
    f = jnp.clip(f, 0.0, 1.0)
    return (3.0 - 2.0 * f) * f * f


@jax.tree_util.register_pytree_node_class
class Gordon23DustLaw(DustLaw):
    """Gordon et al. (2023) UV/optical/N-MIR extinction law
    (GALextinct_Gord23, MWgaldust.c:1020-1168)."""

    numerical_fields = ()

    wave_range = (WAVEMIN_GORD23, WAVEMAX_GORD23)

    def __init__(self):
        super().__init__("Gordon23")

    def extinction_curve(self, wavelength: jnp.ndarray, rv: float, **kwargs) -> jnp.ndarray:
        valid_mask = (wavelength >= WAVEMIN_GORD23) & (wavelength <= WAVEMAX_GORD23)
        x = 10000.0 / wavelength
        mwave = wavelength / 10000.0  # microns

        a = jnp.zeros_like(x)
        b = jnp.zeros_like(x)

        uv_region = (x >= 1.0/0.33) & (x <= 1.0/0.09)
        w_uv = jnp.where(x > 1.0/0.30, 1.0,
                         1.0 - _smoothstep((mwave - 0.30) / 0.03))
        w_uv = jnp.where(uv_region, w_uv, 0.0)

        a_uv = _compute_fm90(x, 0.81297, 0.2775, 1.06295, 0.11303, 5.90, 21.16, 0.9801)
        b_uv = _compute_fm90(x, -2.97868, 1.89808, 3.10334, 0.65484, 5.90, 21.16, 0.9801)
        a = a + w_uv * a_uv
        b = b + w_uv * b_uv

        opt_region = (x >= 1.0/1.1) & (x < 1.0/0.3)
        w_opt = jnp.ones_like(x)
        w_opt = jnp.where(x >= 1.0/0.33, _smoothstep((mwave - 0.30) / 0.03), w_opt)
        w_opt = jnp.where(x <= 1.0/0.9, 1.0 - _smoothstep((mwave - 0.9) / 0.2), w_opt)
        w_opt = jnp.where(opt_region, w_opt, 0.0)

        x2 = x * x
        x3 = x2 * x
        x4 = x2 * x2
        a_opt = -0.35848 + 0.7122*x + 0.08746*x2 - 0.05403*x3 + 0.00674*x4
        b_opt = 0.12354 - 2.68335*x + 2.01901*x2 - 0.39299*x3 + 0.03355*x4

        x01, x02_d, x03 = 2.288, 2.054, 1.587
        FW1, FW2 = 0.243, 0.179
        FX1 = FW1**2 / x01**2
        FX2 = FW2**2 / x02_d**2
        FX3 = FW1**2 / x03**2   # FW3 = FW1
        XX1 = x/x01 - x01/x
        XX2 = x/x02_d - x02_d/x
        XX3 = x/x03 - x03/x
        D1 = FX1 / (XX1**2 + FX1)
        D2 = FX2 / (XX2**2 + FX2)
        D3 = FX3 / (XX3**2 + FX3)

        a_opt = a_opt + 0.03893*D1 + 0.02965*D2 + 0.01747*D3
        b_opt = b_opt + 0.18453*D1 + 0.19728*D2 + 0.17130*D3

        a = a + w_opt * a_opt
        b = b + w_opt * b_opt

        ir_region = (x >= 1.0/35.0) & (x < 1.0/0.9)
        w_ir = jnp.where(x < 1.0/1.1, 1.0,
                         _smoothstep((mwave - 0.9) / 0.2))
        w_ir = jnp.where(ir_region, w_ir, 0.0)

        scale = 0.38526
        alpha = 1.68467
        alpha2 = 0.78791
        swave = 4.30578
        swidth = 4.78338

        fweight = (mwave - (swave - 0.5*swidth)) / swidth
        pweight = _smoothstep(fweight)
        ratio = swave**(-alpha) / swave**(-alpha2)
        a_ir = scale * (1.0 - pweight) * jnp.power(jnp.maximum(mwave, 1e-10), -alpha)
        a_ir = a_ir + scale * ratio * pweight * jnp.power(jnp.maximum(mwave, 1e-10), -alpha2)

        sil1_amp, sil1_cen, sil1_fwhm, sil1_asym = 0.06652, 9.8434, 2.21205, -0.24703
        sil2_amp, sil2_cen, sil2_fwhm, sil2_asym = 0.0267, 19.58294, 17.0, -0.27

        sil1_gamma = 2.0 * sil1_fwhm / (1.0 + jnp.exp(sil1_asym * (mwave - sil1_cen)))
        sil2_gamma = 2.0 * sil2_fwhm / (1.0 + jnp.exp(sil2_asym * (mwave - sil2_cen)))
        sil1_gx2 = sil1_gamma**2 / sil1_cen**2
        sil2_gx2 = sil2_gamma**2 / sil2_cen**2
        sil1_xx = mwave/sil1_cen - sil1_cen/mwave
        sil2_xx = mwave/sil2_cen - sil2_cen/mwave
        a_ir = a_ir + sil1_amp * sil1_gx2 / (sil1_xx**2 + sil1_gx2)
        a_ir = a_ir + sil2_amp * sil2_gx2 / (sil2_xx**2 + sil2_gx2)

        b_ir = -1.01251 * jnp.power(jnp.maximum(x, 1e-10), 1.06099)

        a = a + w_ir * a_ir
        b = b + w_ir * b_ir

        extinction = a + b * (1.0/rv - 1.0/3.1)
        return jnp.where(valid_mask, extinction, 0.0)


@jax.tree_util.register_pytree_node_class
class Sommovigo25DustLaw(DustLaw):
    """Sommovigo et al. (2025) four-parameter Pei model
    (GALextinct_Somm25, MWgaldust.c:1170-1209)."""

    numerical_fields = ()

    wave_range = (WAVEMIN_SOMM25, WAVEMAX_SOMM25)
    linear_in_av = False

    def __init__(self):
        super().__init__("Sommovigo25")

    def extinction_curve(self, wavelength: jnp.ndarray, rv: float, av: float = 1.0, **kwargs) -> jnp.ndarray:
        valid_mask = (wavelength >= WAVEMIN_SOMM25) & (wavelength <= WAVEMAX_SOMM25)
        x = 10000.0 / wavelength

        log_av = jnp.log10(jnp.maximum(av, 1e-10))
        log_c1 = -0.37 * log_av + 0.75
        c1 = jnp.power(10.0, log_c1)
        c2 = 1.88
        c3 = 1.21 * log_c1 - 1.33
        log_c4 = -0.59 * log_av - 1.42
        c4 = jnp.power(10.0, log_c4)

        # Pei4 formula (GALextinct_Pei4, MWgaldust.c:1297-1332)
        x08 = x * 0.08
        x046 = x * 0.046
        x2175 = x * 0.2175
        y08 = jnp.power(x08, c2)
        y046 = x046 * x046
        y2175 = x2175 * x2175
        b_val = jnp.power(0.145, c2)

        k = c1 / (y08 + 1.0/y08 + c3)
        k = k + 233.0 * (1.0 - c4/4.60 - c1/(b_val + 1.0/b_val + c3)) / (y046 + 1.0/y046 + 90.0)
        k = k + c4 / (y2175 + 1.0/y2175 - 1.95)

        return jnp.where(valid_mask, k, 0.0)


@jax.tree_util.register_pytree_node_class
class Gordon16DustLaw(DustLaw):
    """Gordon et al. (2016) mixture: FA*F99(RVA) + (1-FA)*G03(2.74)
    (MWgaldust.c:522-552)."""

    numerical_fields = ('_rva', '_fa', '_f99', '_g03')

    wave_range = (WAVEMIN_FITZ99_EXACT, WAVEMAX_FITZ99_EXACT)

    def __init__(self):
        super().__init__("Gordon16")
        self._rva = None
        self._fa = None
        self._f99 = Fitzpatrick99ExactDustLaw()
        self._g03 = Gordon03DustLaw()

    def set_params(self, parlist):
        """RVA and FA from PARLIST_MWCOLORLAW (MWgaldust.c:525-546)."""
        if len(parlist) < 2:
            raise SNANAError(
                "Gordon (2016) requires two values in PARLIST_MWCOLORLAW: RVA,FA.",
                function_name="set_params")
        rva, fa = float(parlist[0]), float(parlist[1])
        if not 2.0 <= rva <= 6.0:
            raise SNANAError(f"insane RVA = {rva:.3f}; valid range is [2, 6].",
                             function_name="set_params")
        if not 0.0 <= fa <= 1.0:
            raise SNANAError(f"insane FA = {fa:.3f}; valid range is [0, 1].",
                             function_name="set_params")
        self._rva, self._fa = rva, fa

    def prepare_wavelength_grid(self, wavelength_grid: np.ndarray):
        self._f99.prepare_wavelength_grid(wavelength_grid)
        self._g03.prepare_wavelength_grid(wavelength_grid)

    def extinction_curve(self, wavelength: jnp.ndarray, rv: float, **kwargs) -> jnp.ndarray:
        if self._rva is None:
            raise SNANAError(
                "Gordon (2016) requires PARLIST_MWCOLORLAW: RVA,FA.",
                function_name="extinction_curve")
        xt_f99 = self._f99.extinction_curve(wavelength, self._rva)
        xt_g03 = self._g03.extinction_curve(wavelength, 2.74)
        return self._fa * xt_f99 + (1.0 - self._fa) * xt_g03


@jax.tree_util.register_pytree_node_class
class Sommovigo26DustLaw(DustLaw):
    """Sommovigo et al. (2026) four-parameter attenuation curve
    (GALextinct_Somm26, MWgaldust.c:1211-1260): the bump strength B0, FUV
    slope B1, UV-optical curvature B2 and UV-optical slope B3 from
    PARLIST_MWCOLORLAW, about a V band at 5542 A where the curve is one; B0
    and B3 must not be negative (BMIN_SOMM26)."""

    numerical_fields = ('_b',)

    wave_range = (WAVEMIN_SOMM26, WAVEMAX_SOMM26)
    # the population-level constants c0..c4 of the symbolic-regression form
    _c0, _c1, _c2, _c3, _c4 = 0.4002, 285.6, 0.2092, 9.223, 1.016
    _wave_v = 5542.0

    def __init__(self):
        super().__init__("Sommovigo26")
        self._b = None

    def set_params(self, parlist):
        """B0, B1, B2, B3 from PARLIST_MWCOLORLAW (MWgaldust.c:562-568, :1240)."""
        if len(parlist) < 4:
            raise SNANAError(
                "Sommovigo et al. (2026) requires four values in "
                "PARLIST_MWCOLORLAW: B0,B1,B2,B3.", function_name="set_params")
        b = tuple(float(v) for v in parlist[:4])
        if b[0] < 0.0 or b[3] < 0.0:
            raise SNANAError(
                f"Read invalid B0={b[0]:.1f} or B3={b[3]:.1f} from "
                f"PARLIST_MWCOLORLAW! Sommovigo et al. (2026) only recommended "
                f"for positive B0 and B3.", function_name="set_params")
        self._b = b

    def extinction_curve(self, wavelength: jnp.ndarray, rv: float, **kwargs) -> jnp.ndarray:
        """A(lambda)/A(V): the bump, the UV-optical polynomial and the slope
        term of the symbolic-regression form."""
        if self._b is None:
            raise SNANAError(
                "Sommovigo et al. (2026) requires PARLIST_MWCOLORLAW: B0,B1,B2,B3.",
                function_name="extinction_curve")
        b0, b1, b2, b3 = self._b
        c0, c1, c2, c3, c4 = self._c0, self._c1, self._c2, self._c3, self._c4
        x = wavelength / self._wave_v
        k = b0 * (jnp.exp(-c1 * (x - c0) ** 2) - jnp.exp(-c1 * (1.0 - c0) ** 2))
        k = k + (b1 + b2 * (x - c2)) * (x - c2) * (jnp.exp(-c3 * x) - jnp.exp(-c3))
        k = k + jnp.exp(b3 * (jnp.tanh(c4) - jnp.tanh(c4 * x)))
        valid = (wavelength >= WAVEMIN_SOMM26) & (wavelength <= WAVEMAX_SOMM26)
        return jnp.where(valid, k, 0.0)


# Each law under its OPT_MWCOLORLAW_NAMES name
DUST_LAWS = {
    "OFF": NoExtinctionDustLaw,
    "CCM89": CCM89DustLaw,
    "ODONNELL94": ODonnell94DustLaw,
    "FITZ99_EXACT": Fitzpatrick99ExactDustLaw,
    "FITZ99_APPROX": Fitzpatrick99ApproxDustLaw,
    "GORD03": Gordon03DustLaw,
    "FITZ04": Fitzpatrick04DustLaw,
    "GOOBAR08": Goobar08DustLaw,
    "MAIZ14": MaizApellaniz14DustLaw,
    "GORD16": Gordon16DustLaw,
    "FITZ19": Fitzpatrick19DustLaw,
    "FITZ19_LIN": Fitzpatrick19LinearDustLaw,
    "GORD23": Gordon23DustLaw,
    "SOMM25": Sommovigo25DustLaw,
    "SOMM26": Sommovigo26DustLaw,
}


def load_dust_law(name: str) -> DustLaw:
    """A new instance of the dust law named ``name`` in :data:`DUST_LAWS`."""
    if name not in DUST_LAWS:
        raise SNANAError(f"Unknown dust law '{name}'. Available: {list(DUST_LAWS)}",
                         function_name="load_dust_law")
    return DUST_LAWS[name]()


class SFDDustMap:
    """The SFD98 E(B-V) maps, read on the first lookup, and lookups by RA/DEC."""

    def __init__(self):
        self._images = None

    def device_data(self):
        """The north and south galactic pole images on the device, each with
        its Lambert projection's header coefficients, loaded from
        $SNDATA_ROOT/MWDUST on the first call."""
        if self._images is None:
            directory = sndata_root() / 'MWDUST'
            keys = ('LAM_NSGP', 'LAM_SCAL', 'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2')
            images = []
            for pole in ('ngp', 'sgp'):
                path = Path(resolve_fits_path(directory / f'SFD_dust_4096_{pole}.fits'))
                if not path.exists():
                    raise SNANAError(f"SFD {pole.upper()} map not found: {path}",
                                     function_name="device_data")
                with fits.open(path) as hdul:
                    header = hdul[0].header
                    images.append((jnp.asarray(hdul[0].data.astype(np.float32)),
                                   {key: header[key] for key in keys}))
            self._images = tuple(images)
        return self._images

    def get_ebv(self, ra_deg: float, dec_deg: float) -> float:
        """Look up E(B-V) from SFD98 maps for a given RA/DEC position."""
        return float(self.get_ebv_batch(np.array([ra_deg]), np.array([dec_deg]))[0])

    def get_ebv_batch(self, ra_arr, dec_arr):
        """E(B-V) at arrays of RA/DEC positions (degrees), by
        :func:`ebv_on_device` in double precision whatever frame the run is
        in, since these are looked up once at setup."""
        maps = self.device_data()
        with jax.enable_x64(True):
            return np.asarray(ebv_on_device(
                maps, jnp.asarray(np.atleast_1d(ra_arr), dtype=jnp.float64),
                jnp.asarray(np.atleast_1d(dec_arr), dtype=jnp.float64)))


def ebv_on_device(maps, ra_deg, dec_deg):
    """E(B-V) from the SFD images :meth:`SFDDustMap.device_data` prepares:
    the Lambert projection of MWgaldust.c's lambert_lb2fpix, then bilinear
    interpolation in the north or south image by galactic latitude. Takes
    the images rather than the reader so that compiled code can use it."""
    l_deg, b_deg = eq_to_gal(ra_deg, dec_deg)

    north = b_deg >= 0
    out = None
    for dev_map, hdr in maps:
        dradeg = 180.0 / jnp.pi
        nsgp, scale = hdr['LAM_NSGP'], hdr['LAM_SCAL']
        rho = jnp.sqrt(1.0 - nsgp * jnp.sin(b_deg / dradeg))
        xr = rho * jnp.cos(l_deg / dradeg) * scale
        yr = -nsgp * rho * jnp.sin(l_deg / dradeg) * scale
        px = xr + hdr['CRPIX1'] - hdr['CRVAL1'] - 1.0
        py = yr + hdr['CRPIX2'] - hdr['CRVAL2'] - 1.0

        ny, nx = dev_map.shape
        ix = jnp.clip(jnp.floor(px).astype(jnp.int32), 0, nx - 2)
        iy = jnp.clip(jnp.floor(py).astype(jnp.int32), 0, ny - 2)
        fx, fy = px - ix, py - iy
        val = ((1 - fx) * (1 - fy) * dev_map[iy, ix] +
               fx * (1 - fy) * dev_map[iy, ix + 1] +
               (1 - fx) * fy * dev_map[iy + 1, ix] +
               fx * fy * dev_map[iy + 1, ix + 1])
        out = val if out is None else jnp.where(north, out, val)
    return out
