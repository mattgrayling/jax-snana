"""The BayeSN model (Mandel et al. 2022, MNRAS 510, 3939), following SNANA's
genmag_BAYESN.c.

:func:`generate_bayesn_sed` builds the rest-frame SED: the Hsiao template at
the phase, warped by ``10^(-0.4 (W0 + theta W1 + epsilon))`` through the
spline coefficients of spline_coeffs_irr and its extrapolation modes, and
dimmed by the grey DELTAM. :class:`BayeSNModel` loads the model's YAML and H5
data and serves the simulator, the light-curve fit and BBC.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from jax import lax, vmap

from jax_snlc_core import logging as log
from jax_snlc_core.filters import band_window_starts
from jax_snlc_core.interpolation import (compute_catmullrom_slopes, hermite_interp_1d,
                                         invkd_irr, spline_coefficients)
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.utils import NEG_0_4_LN10

from .roles import MU_REF, FittedModel, SimulatedModel, StandardisedModel
from .sed_model import LatetimeExtrapIa, SNModel, modelflux_extrap, resample_surfaces

logger = logging.getLogger(__name__)


@jax.jit
def apply_grey_offset(sed: jnp.ndarray, deltam: float) -> jnp.ndarray:
    """Apply BayeSN's grey (wavelength-independent) DELTAM offset:
    ``SED * 10^(-0.4 · DELTAM)``."""
    grey_factor = jnp.exp(NEG_0_4_LN10 * deltam)
    return sed * grey_factor


@jax.jit
def generate_bayesn_sed(phase: float, theta: float, deltam: float,
                        hsiao_phases: jnp.ndarray,
                        hsiao_template_grid: jnp.ndarray,
                        hsiao_slopes: jnp.ndarray,
                        epsilon_matrix: jnp.ndarray,
                        w0_matrix: jnp.ndarray,
                        w1_matrix: jnp.ndarray,
                        phase_knots: jnp.ndarray,
                        phase_invKD: jnp.ndarray,
                        wavelength_spline_coeffs: jnp.ndarray,
                        phase_extrap_mode: jnp.ndarray) -> jnp.ndarray:
    """Complete BayeSN rest-frame SED (before dust/cosmology).

    Steps:
        1. Hsiao-template interpolation at ``phase`` (cubic Hermite).
        2. BayeSN warping ``10^(-0.4 · (W0 + θ·W1 + ε))`` via
           wavelength-spline coefficients and Hermite phase-extrapolation
           modes (0=linear, 1=smooth→zero, 2=smooth→constant).
        3. Grey offset ``10^(-0.4 · DELTAM)``.
    """
    hsiao_sed = hermite_interp_1d(phase, hsiao_phases, hsiao_template_grid, hsiao_slopes)

    w_combined = w0_matrix + theta * w1_matrix + epsilon_matrix
    phase_array = jnp.array([phase])
    J_phase = spline_coefficients(
        phase_array, phase_knots, phase_invKD, phase_extrap_mode)
    J_phase_vec = J_phase[0, :]
    matrix_at_phase = jnp.dot(w_combined, J_phase_vec)
    warping_interpolated = jnp.dot(wavelength_spline_coeffs, matrix_at_phase)
    warping_factor = jnp.exp(NEG_0_4_LN10 * warping_interpolated)

    sed = hsiao_sed * warping_factor
    sed = apply_grey_offset(sed, deltam)
    return sed


@dataclass(eq=False, repr=False, kw_only=True)
class BayeSNModel(SNModel, SimulatedModel, FittedModel, StandardisedModel):
    """The BayeSN model for the simulator, the light-curve fit and BBC.

    BayeSN model: Mandel et al. 2022, MNRAS 510, 3939
    - Uses Hsiao template as base
    - Applies warping: W = W0 + θ*W1 + ε
    - Includes grey offset (DELTAM)

    Reference implementation: SNANA's genmag_BAYESN.c
    """

    wavelength_grid: jax.Array = None
    wavelength_knots: jax.Array = None
    phase_knots: jax.Array = None
    w0: jax.Array = None
    w1: jax.Array = None
    m0_abs: jax.Array = None
    sigma0: jax.Array = None
    _Ds_err: jax.Array = None
    tauA: jax.Array = None
    mu_R: jax.Array = None
    sigma_R: jax.Array = None
    RV: jax.Array = None
    host_rv: jax.Array = None
    L_Sigma_epsilon: jax.Array = None
    phase_invKD: jax.Array = None
    wavelength_invKD: jax.Array = None
    hsiao_phases: jax.Array = None
    hsiao_wavelengths: jax.Array = None
    hsiao_flux_grid: jax.Array = None
    hsiao_phase_slopes: jax.Array = None
    wavelength_spline_coeffs: jax.Array = None
    hsiao_at_daymin: jax.Array = None
    j_phase_daymin: jax.Array = None
    latetime_tau1: jax.Array = None
    latetime_tau2: jax.Array = None
    latetime_expratio: jax.Array = None
    latetime_daymin: jax.Array = None
    _band_basis: jax.Array = None
    _m0_scale: jax.Array = None
    rv_prior_type: str = field(default='fixed', metadata={"static": True, "input": False})
    n_lam_knots: int = field(default=0, metadata={"static": True, "input": False})
    n_tau_knots: int = field(default=0, metadata={"static": True, "input": False})
    n_sig_knots: int = field(default=0, metadata={"static": True, "input": False})
    scatter_epsilon: bool = field(default=False, metadata={"static": True, "input": False})
    scatter_deltam: bool = field(default=False, metadata={"static": True, "input": False})
    time_extrap_mode: int = field(default=2, metadata={"static": True, "input": False})
    wave_extrap_mode: int = field(default=0, metadata={"static": True, "input": False})
    model_name: object = field(init=False, default=None, repr=False)
    yaml_filter_center_range: object = field(init=False, default=None, repr=False)
    hsiao_wavelength_range: object = field(init=False, default=None, repr=False)
    filter_center_range: object = field(init=False, default=None, repr=False)
    extrap_latetime: object = field(init=False, default=None, repr=False)

    GENMODEL_ALIASES = ('bayesn',)

    # SNANA spells it THETA, and SIM_THETA is BayeSN's only header column
    # (sntools_dataformat_fits.c:546). DELTAM is not a population: the model
    # draws it as scatter under GENMODEL_MSKOPT, so its GEN keys have no effect.
    population_parameters = ('theta',)
    population_aliases = {'theta': ('THETA',)}

    def sn_variables(self, params: Dict) -> Dict:
        """theta, plus the grey offset DELTAM the model draws itself."""
        return {**super().sn_variables(params),
                'DELTAM': jnp.asarray(params['deltam'], dtype=jnp.float32)}

    def replay_params_from_dump(self, dump_columns, head_columns, *, mu,
                                distributions):
        if self.scatter_epsilon:
            raise SNANAError(
                "BayeSN replay with epsilon scatter requires the original "
                "epsilon realisation, which SIMGEN_DUMP does not store. "
                "Use a simulation with epsilon scatter disabled for replay.",
                function_name="replay_params_from_dump")
        if 'THETA' not in dump_columns:
            raise SNANAError("Missing required BayeSN column in dump: THETA",
                             function_name="replay_params_from_dump")

        theta_values = dump_columns['THETA'].astype(np.float32)
        deltam_values = dump_columns.get('DELTAM')
        if deltam_values is None:
            deltam_values = np.zeros_like(theta_values, dtype=np.float32)
        else:
            deltam_values = deltam_values.astype(np.float32)
            # SNANA uses -9 sentinel for "no DELTAM" (e.g., MSKOPT=0)
            deltam_values = np.where(deltam_values < -8.0, 0.0, deltam_values)

        n_knots = (int(self.n_lam_knots) - 2) * int(self.n_tau_knots)
        return {
            'theta': theta_values,
            'deltam': deltam_values,
            'Ds': mu.astype(np.float32) + deltam_values,
            # Epsilon scatter is disabled for supported dump replays.
            'eps_tform': np.zeros((len(theta_values), n_knots),
                                  dtype=np.float32),
        }, None

    @classmethod
    def load(cls, model_path: str, config, n_wave_points=500,
             wavelength_range=None):
        """Load the model YAML at ``model_path``
        (e.g. '$SNDATA_ROOT/models/BayeSN/W22/W22.yaml'), apply
        GENMODEL_MSKOPT and any late-time file, and resample onto the working
        grid."""
        model = cls()
        SNModel.__init__(model, "bayesn", config, n_wave_points,
                         wavelength_range)
        if model.uvlam_extrapflux is not None:
            raise SNANAError("UVLAM_EXTRAPFLUX is not supported by BayeSN.",
                             function_name="BayeSNModel")

        yaml_file = Path(model_path).expanduser()
        if not yaml_file.name.endswith(('.yaml', '.YAML')):
            yaml_file = yaml_file / "BAYESN.YAML"

        if not yaml_file.exists():
            raise SNANAError(f"BayeSN model file not found: {yaml_file}",
                             function_name="BayeSNModel")

        logger.info(f"Loading BayeSN model: {yaml_file.name}")
        with open(yaml_file, 'r') as f:
            bayesn_data = yaml.safe_load(f)

        model.wavelength_knots = jnp.array(bayesn_data['L_KNOTS'])
        model.phase_knots = jnp.array(bayesn_data['TAU_KNOTS'])
        model.w0 = jnp.array(bayesn_data['W0'])
        model.w1 = jnp.array(bayesn_data['W1'])
        model.m0_abs = float(bayesn_data['M0'])

        model.sigma0 = float(bayesn_data['SIGMA0'])
        # Weak cosmological distance prior, widened by grey scatter.
        model._Ds_err = float(np.sqrt(5.0 ** 2 + model.sigma0 ** 2))
        model.tauA = float(bayesn_data['TAUA'])

        # SNANA reads the window optionally and leaves an absent key at zero
        model.yaml_filter_center_range = (float(bayesn_data.get('L_FILTER_CEN_MIN', 0.0)),
                                         float(bayesn_data.get('L_FILTER_CEN_MAX', 0.0)))

        if 'MUR' in bayesn_data and 'SIGMAR' in bayesn_data:
            model.rv_prior_type = 'gaussian'
            model.mu_R = float(bayesn_data['MUR'])
            model.sigma_R = float(bayesn_data['SIGMAR'])
            model.RV = model.mu_R
        else:
            model.rv_prior_type = 'fixed'
            model.RV = float(bayesn_data['RV'])
        model.host_rv = model.RV

        if model.wavelength_knots.ndim != 1 or model.phase_knots.ndim != 1:
            raise SNANAError("L_KNOTS and TAU_KNOTS must be one-dimensional",
                             function_name="BayeSNModel")
        n_lam = len(model.wavelength_knots)
        n_tau = len(model.phase_knots)
        n_sig = (n_lam - 2) * n_tau

        for key, size in (("N_LAM", n_lam), ("N_TAU", n_tau), ("N_SIG", n_sig)):
            if key in bayesn_data and bayesn_data[key] != size:
                raise SNANAError(
                    f"{key}={bayesn_data[key]} doesn't match the knot arrays "
                    f"(expected {size})", function_name="BayeSNModel")

        for name, matrix in (("W0", model.w0), ("W1", model.w1)):
            if matrix.shape != (n_lam, n_tau):
                raise SNANAError(
                    f"{name} shape {matrix.shape} doesn't match expected "
                    f"{(n_lam, n_tau)}", function_name="BayeSNModel")

        model.L_Sigma_epsilon = jnp.array(
            np.array(bayesn_data['L_SIGMA_EPSILON'], dtype=np.float32))
        expected_shape = (n_sig, n_sig)
        if model.L_Sigma_epsilon.shape != expected_shape:
            raise SNANAError(
                f"L_Sigma_epsilon shape {model.L_Sigma_epsilon.shape} doesn't "
                f"match expected {expected_shape}", function_name="BayeSNModel")

        model.n_lam_knots = n_lam
        model.n_tau_knots = n_tau
        model.n_sig_knots = n_sig

        model.phase_invKD = jnp.asarray(invkd_irr(model.phase_knots))
        model.wavelength_invKD = jnp.asarray(invkd_irr(model.wavelength_knots))

        hsiao_file = Path(__file__).parent.parent / "data" / "hsiao.h5"
        with h5py.File(hsiao_file, 'r') as f:
            model.hsiao_phases = jnp.array(f['default/phase'][...])
            model.hsiao_wavelengths = jnp.array(f['default/wave'][...])
            model.hsiao_flux_grid = jnp.array(f['default/flux'][...])
        model.hsiao_wavelength_range = (float(model.hsiao_wavelengths[0]),
                                       float(model.hsiao_wavelengths[-1]))

        model.hsiao_phase_slopes = compute_catmullrom_slopes(
            model.hsiao_phases, model.hsiao_flux_grid)

        # Intrinsic scatter flags (set by set_mskopt). GENMODEL_MSKOPT=0
        # means no intrinsic BayeSN scatter (genmag_BAYESN.c:458-461).
        model.scatter_epsilon = False
        model.scatter_deltam = False

        model.extrap_latetime = None

        model.set_mskopt(model.genmodel_mskopt)
        if model.genmodel_extrap_latetime:
            model.set_extrap_latetime(model.genmodel_extrap_latetime)
        model.resample_to_grid(model.choose_grid())

        log.line(logger, "BayeSN model ready")
        log.line(logger, f"Wavelength knots: {len(model.wavelength_knots)} ({model.wavelength_knots.min():.0f}-{model.wavelength_knots.max():.0f} Å)", indent=2)
        log.line(logger, f"Phase knots: {len(model.phase_knots)} ({model.phase_knots.min():.0f}-{model.phase_knots.max():.0f} days)", indent=2)
        log.line(logger, f"L_Sigma_epsilon: {model.L_Sigma_epsilon.shape}", indent=2)
        log.line(logger, f"Hsiao template: {len(model.hsiao_phases)} phases, {len(model.hsiao_wavelengths)} wavelengths", indent=2)
        log.line(logger, f"M0={model.m0_abs}, SIGMA0={model.sigma0}, TAUA={model.tauA}",
                 indent=2)
        rv_info = f"RV={model.RV} (fixed)" if model.rv_prior_type == 'fixed' else f"RV~N({model.mu_R}, {model.sigma_R})"
        log.line(logger, rv_info, indent=2)
        log.line(logger, f"N_SIG={model.n_sig_knots} epsilon parameters for fitting", indent=2)

        return model

    def set_mskopt(self, mskopt: int) -> None:
        """Parse GENMODEL_MSKOPT bits for BayeSN scatter and extrapolation.

        SNANA bit definitions (genmag_BAYESN.h:103-112)::

          1    = default scatter (epsilon + DELTAM); combining it with bits
                 2/4 is fatal (genmag_BAYESN.c:445-457)
          2    = epsilon scatter only
          4    = DELTAM scatter only
          32   = TIME_EXTRAP_0  (phase: Hermite smooth to zero)
          64   = TIME_EXTRAP_W  (phase: Hermite smooth to constant)
          512  = WAVE_EXTRAP_0  (wavelength: Hermite smooth to zero)
          1024 = WAVE_EXTRAP_W  (wavelength: Hermite smooth to constant)

        No scatter bit means no intrinsic BayeSN scatter; selecting both time
        modes or both wavelength modes is fatal (genmag_BAYESN.c:471-484).

        Without bit 32 the phase extrapolation is smooth to constant, where
        SNANA's default is linear.

        A wavelength mode sets the filter-centre window to the Hsiao
        template's range less 1000 A at the blue end and 1500 A at the red;
        without one the window is the YAML's L_FILTER_CEN_MIN and
        L_FILTER_CEN_MAX, zero where a key is absent.
        """
        if mskopt & 1:
            if mskopt & 6:
                raise SNANAError(
                    f"Ambiguous scatter configuration requested with "
                    f"GENMODEL_MSKOPT={mskopt}: if bit 1 is set (default scatter), no "
                    f"other scatter bits can be set.",
                    function_name="set_mskopt")
            self.scatter_epsilon = True
            self.scatter_deltam = True
        else:
            self.scatter_epsilon = bool(mskopt & 2)
            self.scatter_deltam = bool(mskopt & 4)

        if (mskopt & 32) and (mskopt & 64):
            raise SNANAError(
                f"Ambiguous time extrapolation requested with "
                f"GENMODEL_MSKOPT={mskopt}: only one of bits 32 and 64 can be set.",
                function_name="set_mskopt")
        if (mskopt & 512) and (mskopt & 1024):
            raise SNANAError(
                f"Ambiguous wavelength extrapolation requested with "
                f"GENMODEL_MSKOPT={mskopt}: only one of bits 512 and 1024 can be set.",
                function_name="set_mskopt")

        self.time_extrap_mode = 1 if mskopt & 32 else 2

        self.wave_extrap_mode = 0
        if mskopt & 512:
            self.wave_extrap_mode = 1
        if mskopt & 1024:
            self.wave_extrap_mode = 2

        if self.wave_extrap_mode > 0:
            # inside the full Hsiao template, whatever the YAML says
            lo, hi = self.hsiao_wavelength_range
            self.filter_center_range = (lo + 1000.0, hi - 1500.0)
        else:
            self.filter_center_range = self.yaml_filter_center_range

        log.line(logger, f"BayeSN scatter: epsilon={self.scatter_epsilon}, "
                 f"DELTAM={self.scatter_deltam}")
        log.line(logger, f"BayeSN extrap: time_mode={self.time_extrap_mode}, "
                 f"wave_mode={self.wave_extrap_mode}")

    def set_extrap_latetime(self, filename: str) -> None:
        """Load late-time Ia double-exponential decay model for BayeSN.

        When set, phases beyond the file's EXTRAP_DAYMIN use per-wavelength
        FLAM decay instead of the default linear-in-magnitude tail from the
        Hsiao boundary (modelflux_extrap). ``filename`` is the
        GENMODEL_EXTRAP_LATETIME path.
        """
        self.extrap_latetime = LatetimeExtrapIa(filename)
        log.line(logger, f"BayeSN late-time extrap: FLAM method, "
                 f"daymin={self.extrap_latetime.daymin:.0f}")

    def get_fit_parameters(self) -> Dict[str, Dict[str, Any]]:
        """BayeSN fit parameters.

        ``[PKMJD, THETA, Ds, AV, (RV,) EPS_TFORM_0..{n_sig-1}]``, where
        ``Ds = mu + deltam`` (distance modulus plus intrinsic grey offset) and
        ``EPS_TFORM`` are whitened epsilons, ``~ N(0, I)``. ``RV`` is present
        only under a Gaussian RV prior; the epsilon count comes from the trained
        file, so the set varies with what was loaded.

        All are O(1) in their natural/whitened space, so scales are 1.0 except
        ``AV``, whose typical range is smaller.
        """
        params = {
            **super().get_fit_parameters(),
            'THETA': {'bounds': (-5.0, 5.0), 'scale': 1.0, 'init': 0.0,
                      'sim_truth': 'SIM_THETA', 'inistp': 0.1,
                      'errmax_bad': 0.001, 'error_bounds': (1.0e-8, 3.0),
                      'priors': (('flat_rolloff',
                                  {'lo': -9.0, 'hi': 9.0, 'sigma': 0.1}),
                                 ('gauss', {'centre': 0.0, 'sigma': 1.0})),
                      'peg_margin': 0.05,
                      'snana_keys': ('SHAPE', 'LUMIPAR'),
                      'grid_search': (-1.0, 0.0, 1.0),
                      'coarse_grid': {'range': (-3.0, 1.0), 'nbin': 3}},
            'Ds': {'bounds': (10.0, 60.0), 'scale': 1.0,
                   'sim_truth': 'SIM_DLMU', 'inistp': 0.5,
                   'errmax_bad': 0.0, 'error_bounds': (1.0e-15, 3.0),
                   'priors': (('gauss', {'centre': MU_REF,
                                         'sigma': self._Ds_err}),),
                   'snana_keys': ('DLMAG',)},
            # no flat prior: IS_MODEL_FLATCOLOR is SALT2/SNOOPY only (snlc_fit.F90:12023)
            'AV': {'bounds': (0.0, 3.0), 'scale': 0.2, 'sim_truth': 'SIM_AV',
                   'init': 0.1, 'inistp': 0.2, 'errmax_bad': 0.001,
                   'priors': (('exp_convolved', {'tau': [self.tauA, 1.0],
                                                 'wgt': [1.0, 0.0],
                                                 'sigma': 0.0}),),
                   'error_bounds': (1.0e-8, 3.0), 'snana_keys': ('COLOR',)},
        }
        if self.rv_prior_type == 'gaussian':
            params['RV'] = {'bounds': (1.2, 6.0), 'scale': 1.0,
                            'init': self.mu_R,
                            'priors': (('gauss', {'centre': self.mu_R,
                                                  'sigma': self.sigma_R}),),
                            'sim_truth': 'SIM_RV', 'errmax_bad': 0.001}
        for i in range(self.n_sig_knots):
            params[f'EPS_TFORM_{i}'] = {'bounds': (-5.0, 5.0), 'scale': 1.0,
                                        'init': 0.0,
                                        'priors': (('gauss', {'centre': 0.0,
                                                              'sigma': 1.0}),),
                                        'ndof': False, 'deferred': True}
        return params

    def get_wavelength_range(self) -> Tuple[float, float]:
        """The knot range, or the Hsiao range with wavelength extrapolation,
        as (min_wavelength, max_wavelength) in Angstroms."""
        if self.wave_extrap_mode:
            return self.hsiao_wavelength_range
        return (float(self.wavelength_knots.min()), float(self.wavelength_knots.max()))

    def resample_to_grid(self, wavelength_grid: jnp.ndarray) -> None:
        """Resample the Hsiao template onto ``wavelength_grid``, called once
        during setup, then precompute the wavelength spline coefficients for
        BayeSN warping on that grid."""
        native_wavelengths = self.hsiao_wavelengths

        log.line(logger, f"Resampling Hsiao from {len(native_wavelengths)} to {len(wavelength_grid)} wavelengths...")

        self.hsiao_flux_grid = jnp.array(resample_surfaces(
            native_wavelengths, self.hsiao_flux_grid, wavelength_grid))
        self.hsiao_wavelengths = wavelength_grid

        self.hsiao_phase_slopes = compute_catmullrom_slopes(
            self.hsiao_phases, self.hsiao_flux_grid)

        log.line(logger, f"Hsiao template resampled to {len(wavelength_grid)} wavelengths")

        self._precompute_wavelength_spline_coeffs(wavelength_grid)
        self._m0_scale = jnp.exp(NEG_0_4_LN10 * self.m0_abs)
        self._band_basis = jnp.concatenate([
            self.wavelength_spline_coeffs,
            jnp.ones((len(wavelength_grid), 1)),
        ], axis=1)

    def _precompute_wavelength_spline_coeffs(self, wavelength_grid: jnp.ndarray):
        """Precompute the wavelength spline coefficients for ``wavelength_grid``
        (Å), once, rather than recomputing them for every SED generation,
        extrapolated by ``wave_extrap_mode``.
        """
        self.wavelength_spline_coeffs = spline_coefficients(
            wavelength_grid, self.wavelength_knots, self.wavelength_invKD,
            jnp.int32(self.wave_extrap_mode))
        logger.debug(f"  Pre-computed wavelength spline coefficients: {self.wavelength_spline_coeffs.shape}"
                     f" (wave_extrap_mode={self.wave_extrap_mode})")

        if self.extrap_latetime is not None:
            daymin = float(self.extrap_latetime.daymin)
            self.latetime_daymin = daymin
            daymin_hsiao = float(jnp.clip(daymin, self.hsiao_phases[0], self.hsiao_phases[-1]))
            self.hsiao_at_daymin = hermite_interp_1d(
                daymin_hsiao, self.hsiao_phases, self.hsiao_flux_grid, self.hsiao_phase_slopes)
            phase_extrap_mode = jnp.int32(self.time_extrap_mode)
            j_phase = spline_coefficients(
                jnp.array([daymin]), self.phase_knots, self.phase_invKD, phase_extrap_mode)
            self.j_phase_daymin = j_phase[0, :]
            self.latetime_tau1, self.latetime_tau2, self.latetime_expratio = (
                self.extrap_latetime.precompute_decay_params(wavelength_grid))

    def sed_param_names(self):
        return ('theta', 'Ds', 'eps_tform')

    def phase_range(self):
        """The Hsiao template's phase grid, which bounds where the warped
        SED is defined. SNANA's PySEDMODEL route imposes no phase gate of
        its own (genSpec_PySEDMODEL), leaving it to the model.
        """
        return float(self.hsiao_phases[0]), float(self.hsiao_phases[-1])

    def generate_sed(self, phase: float, params: Dict[str, float],
                     static_data: Dict) -> jnp.ndarray:
        """The BayeSN SED at ``phase`` (days relative to peak), via
        :func:`generate_bayesn_sed` on the model's own data and the
        wavelength grid from ``static_data``.

        ``params`` holds 'theta', 'Ds' (= mu + deltam, the combined
        grey/distance offset) and 'eps_tform' (whitened epsilon vector,
        n_sig components; the fitter's numbered eps_tform_* fit parameters
        arrive grouped under this key). Host dust is applied by the caller.
        Returns the SED (erg/s/cm²/Å).
        """
        theta = params['theta']
        Ds = params['Ds']
        epsilon_matrix = self._eps_matrix(params['eps_tform'])

        phase_extrap_mode = jnp.int32(self.time_extrap_mode)
        wl_coeffs = self.wavelength_spline_coeffs

        def _eval_sed(ph):
            return generate_bayesn_sed(
                phase=ph, theta=theta, deltam=Ds,
                hsiao_phases=self.hsiao_phases,
                hsiao_template_grid=self.hsiao_flux_grid,
                hsiao_slopes=self.hsiao_phase_slopes,
                epsilon_matrix=epsilon_matrix,
                w0_matrix=self.w0, w1_matrix=self.w1,
                phase_knots=self.phase_knots,
                phase_invKD=self.phase_invKD,
                wavelength_spline_coeffs=wl_coeffs,
                phase_extrap_mode=phase_extrap_mode,
            )

        sed = _eval_sed(phase)

        m0_scale = jnp.exp(NEG_0_4_LN10 * self.m0_abs)
        sed = sed * m0_scale

        # Late-time extrapolation beyond the Hsiao boundary; the fitter
        # excludes it, the simulator applies it.
        if not static_data['latetime_extrap']:
            return sed

        hsiao_phase_max = self.hsiao_phases[-1]  # ~85d
        hsiao_phase_min = self.hsiao_phases[0]

        if self.latetime_tau1 is not None:
            daymin = self.latetime_daymin
            use_flam = phase > daymin

            w_combined = self.w0 + theta * self.w1 + epsilon_matrix
            matrix_at_daymin = jnp.dot(w_combined, self.j_phase_daymin)
            warp_at_daymin = jnp.dot(wl_coeffs, matrix_at_daymin)
            warp_factor_daymin = jnp.exp(NEG_0_4_LN10 * warp_at_daymin)
            sed_daymin_raw = self.hsiao_at_daymin * warp_factor_daymin
            sed_daymin_raw = apply_grey_offset(sed_daymin_raw, Ds)
            sed_daymin = sed_daymin_raw * m0_scale
            decay = LatetimeExtrapIa.decay_factor(
                phase, daymin, self.latetime_tau1, self.latetime_tau2,
                self.latetime_expratio
            )
            sed = jnp.where(use_flam, sed_daymin * decay, sed)
        else:
            beyond_hsiao = phase > hsiao_phase_max
            delta_ph = 3.0
            inner_phase = jnp.clip(hsiao_phase_max - delta_ph,
                                   hsiao_phase_min, hsiao_phase_max)
            sed_inner = _eval_sed(inner_phase) * m0_scale
            sed_boundary = sed
            flux_slope = jnp.where(beyond_hsiao,
                                   (sed_boundary - sed_inner) / delta_ph, 0.0)
            safe_edge_phase = jnp.where(beyond_hsiao, hsiao_phase_max, phase)
            safe_edge_flux = jnp.where(beyond_hsiao, sed_boundary, sed)
            extrap_sed = modelflux_extrap(phase, safe_edge_phase,
                                         safe_edge_flux, flux_slope)
            sed = jnp.where(beyond_hsiao, extrap_sed, sed)

        return sed

    def evaluate(self, sed_params, mjds, filter_indices, mask, band_weights,
                 static_data, sed_factor=None, band_weights_dz=None,
                 band_weights_dz2=None):
        """FLUXCAL with the band integral's derivatives in closed form.

        Every parameter but the peak time enters one exponent linearly against
        a fixed wavelength basis, so each derivative is a moment of the same
        band-weighted spectrum and the wavelength axis closes before the
        parameter axis opens. The base implementation would instead let AD
        carry one spectrum-shaped tangent per parameter.

        The moments are taken against that fixed basis, which the late-time
        decay is not part of, so the simulator's configuration goes to the
        base implementation; so does a caller with no window, and one
        supplying its own wavelength factor or an AV-dependent dust curve.
        """
        if (static_data['band_window'] is None or static_data['latetime_extrap']
                or sed_factor is not None
                or not static_data['host_dust_law'].linear_in_av):
            return super().evaluate(sed_params, mjds, filter_indices, mask,
                                    band_weights, static_data,
                                    sed_factor=sed_factor,
                                    band_weights_dz=band_weights_dz,
                                    band_weights_dz2=band_weights_dz2)
        band_flux = self._build_band_flux(
            static_data, fit_z=band_weights_dz is not None)

        t0, redshift = sed_params['t0'], sed_params['redshift']
        theta, Ds, av = sed_params['theta'], sed_params['Ds'], sed_params['av']
        rv = (sed_params['rv'] if 'rv' in sed_params
              else static_data['host_rv'])
        eps = sed_params['eps_tform']
        width = static_data['band_window']
        obs_lo = band_window_starts(band_weights, filter_indices, width)
        obs_bw = band_weights[filter_indices]
        obs_ref = static_data['filter_ref_fluxes'][filter_indices]
        zp = static_data['zp_fluxcal']
        scale = static_data['fluxcal_scale_mag_offset']
        mjds_masked = jnp.where(mask, mjds, t0)

        if band_weights_dz is not None:
            obs_bw_dz = band_weights_dz[filter_indices]
            obs_bw_dz2 = band_weights_dz2[filter_indices]
        else:
            obs_bw_dz = obs_bw_dz2 = jnp.zeros_like(obs_bw)

        def one(mjd, lo, bw, bwz, bwzz, ref):
            s = band_flux(t0, theta, Ds, av, rv, eps, mjd, redshift,
                          bw, bwz, bwzz, lo)
            return s / ref * jnp.power(10.0, 0.4 * zp) * scale

        return jnp.where(mask, vmap(one)(mjds_masked, obs_lo, obs_bw,
                                         obs_bw_dz, obs_bw_dz2, obs_ref), 0.0)

    def _eps_matrix(self, eps_tform):
        """The epsilon knot matrix; linear in the whitened parameters."""
        interior = (self.L_Sigma_epsilon @ eps_tform).reshape(
            self.n_lam_knots - 2, self.n_tau_knots, order='F')
        return jnp.zeros((self.n_lam_knots, self.n_tau_knots)).at[1:-1, :].set(interior)

    def _build_band_flux(self, static_data, fit_z=False):
        """Band-integrated flux with its derivatives in closed form.

        Every fitted parameter but the peak time enters one exponent,
        linearly, against a fixed wavelength basis: the spline coefficients
        for theta and the epsilons, a constant column for the grey offset,
        and the host dust curve for AV. So each derivative is a moment of the
        band-weighted spectrum against that basis, and the wavelength axis
        closes before the parameter axis opens.

        Two copies of one rule: the outer takes its moments from the inner,
        so differentiating it for the Hessian reaches the inner rule's
        analytic tangents rather than a spectrum-shaped tangent.

        The moment names below are S (integrated flux), G (flux times each
        exponent basis column), and T (the Hsiao phase derivative times the
        warped band weights). K, Th and U are the corresponding second
        moments; a z suffix replaces the weights by their redshift derivative.
        """
        C = NEG_0_4_LN10
        HP, HG, HS = self.hsiao_phases, self.hsiao_flux_grid, self.hsiao_phase_slopes
        W0, W1 = self.w0, self.w1
        KN, IK = self.phase_knots, self.phase_invKD
        WLc = self.wavelength_spline_coeffs
        mode = self.time_extrap_mode
        fit_rv = self.rv_prior_type == 'gaussian'
        grid = static_data['wavelength_grid']
        law = static_data['host_dust_law']
        _curve = lambda rv: law.extinction_curve(grid, rv)
        ONES = jnp.ones((WLc.shape[0], 1))
        m0s = self._m0_scale
        # The spline/grey basis is model-only and prepared during loading.
        # Dust belongs to the supplied evaluation context, which can override
        # the model's default law even when RV is held fixed.
        if not fit_rv:
            BASIS = jnp.concatenate([
                self._band_basis, _curve(static_data['host_rv'])[:, None],
            ], axis=1)

        def _basis(av, rv):
            """The wavelength basis of the exponent.

            With RV fixed the dust column is a constant. With RV fitted the
            curve moves with it, so the basis is rebuilt and gains a fourth
            group: the exponent's RV derivative, whose tangent coefficient is
            dRV. Both extra columns are [n_wave] and carry no parameter axis.
            """
            if not fit_rv:
                return BASIS
            c0, c1 = jax.jvp(_curve, (rv,), (1.0,))
            return jnp.concatenate(
                [WLc, ONES, c0[:, None], (av * c1)[:, None]], axis=1)
        _hs = lambda ph: hermite_interp_1d(ph, HP, HG, HS)
        _Jf = lambda ph: spline_coefficients(
            jnp.array([ph]), KN, IK, mode)[0]

        width = static_data['band_window']   # static: a traced width cannot slice

        def _small(t0, theta, eps_tform, mjd, z):
            ph = (mjd - t0) / (1.0 + z)
            Wc = W0 + theta * W1 + self._eps_matrix(eps_tform)
            J, dJ = jax.jvp(_Jf, (ph,), (1.0,))
            return ph, Wc, J, dJ

        def _spectra(t0, theta, Ds, av, rv, eps_tform, mjd, z, bw, lo):
            """Band-weighted spectrum and its phase derivatives, on the window."""
            ph, Wc, J, _ = _small(t0, theta, eps_tform, mjd, z)
            h, dh = jax.jvp(_hs, (ph,), (1.0,))
            _, d2h = jax.jvp(lambda q: jax.jvp(_hs, (q,), (1.0,))[1], (ph,), (1.0,))
            sl = lambda a: lax.dynamic_slice(a, (lo,), (width,))
            full = _basis(av, rv)
            B = lax.dynamic_slice(full, (lo, 0), (width, full.shape[1]))
            # the RV column contributes nothing to the exponent itself
            tail = jnp.array([Ds, av, 0.0]) if fit_rv else jnp.array([Ds, av])
            coef = jnp.concatenate([Wc @ J, tail])
            fac = jnp.exp(C * (B @ coef)) * m0s          # without the weights
            bws = sl(bw)
            return (B, sl(h) * fac * bws, sl(dh) * fac * bws,
                    sl(d2h) * fac * bws, sl(h) * fac, sl(dh) * fac)

        def _hi(pr):
            """Third-order moments: values only, never differentiated.

            Under photo-z it also returns the moments against d(bw)/dz and
            d2(bw)/dz2, which carry the redshift through G, T and Sz."""
            sg = jax.lax.stop_gradient
            t0, theta, Ds, av, rv, eps, mjd, z, bw, bwz, bwzz, lo = pr
            B, hw, dhw, d2hw, hwo, dhwo = _spectra(
                sg(t0), sg(theta), sg(Ds), sg(av), sg(rv), sg(eps), mjd, z, bw,
                lo)
            sl = lambda a: lax.dynamic_slice(a, (lo,), (width,))
            K = jnp.einsum('x,xa,xc->ac', hw, B, B)
            zmom = ()
            if fit_z:
                bzw = sl(bwz)
                zmom = ((hwo * bzw) @ B, jnp.dot(dhwo, bzw),
                        jnp.dot(hwo, sl(bwzz)))
            if not fit_rv:
                return (K, dhw @ B, jnp.sum(d2hw), 0.0, 0.0) + zmom
            c1, c2 = jax.jvp(lambda q: jax.jvp(_curve, (q,), (1.0,))[1],
                             (sg(rv),), (1.0,))
            Pd = jnp.dot(hw, sl(c1))
            Qd = sg(av) * jnp.dot(hw, sl(c2))
            return (K, dhw @ B, jnp.sum(d2hw), Pd, Qd) + zmom

        def _tangent(pr, dp, S, G, T, dz=0.0, Sz=0.0):
            """Under photo-z the redshift is fitted, so it reaches the phase
            and the band weights; both are carried here. The weights enter
            through d(bw)/dz and d2(bw)/dz2, which do not vary with the
            tangent direction, rather than through their own tangents, which
            do: the moments against them are formed once and scaled by dz.

            ``Sz``, the flux against d(bw)/dz, is an output of the inner rule,
            so the outer rule's redshift term is differentiated through
            moments rather than through the spectrum."""
            t0, theta, Ds, av, rv, eps_tform, mjd, z, bw, bwz, bwzz, lo = pr
            ph, Wc, J, dJ = _small(t0, theta, eps_tform, mjd, z)
            K, Th, U, Pd, Qd, *zmom = _hi(pr)
            dt0, dth, dDs, dav, drv, deps = dp
            head = (dth * W1 + self._eps_matrix(deps)) @ J
            n_pad = 3 if fit_rv else 2
            tail = jnp.array([dDs, dav, drv]) if fit_rv else jnp.array([dDs, dav])
            u = jnp.concatenate([head, tail])
            v = jnp.concatenate([Wc @ dJ, jnp.zeros(n_pad)])
            # phase = (mjd - t0) / (1 + z), so the redshift moves it too
            dph = (-dt0 - ph * dz) / (1.0 + z)
            dS = dph * (T + C * jnp.dot(G, v)) + C * jnp.dot(G, u)
            dG = dph * (Th + C * (K @ v))      + C * (K @ u)
            dT = dph * (U + C * jnp.dot(Th, v)) + C * jnp.dot(Th, u)
            if fit_z:
                Gz, Tz, Szz = zmom
                dS = dS + dz * Sz
                dG = dG + dz * Gz
                dT = dT + dz * Tz
            if fit_rv:
                # d(basis) reaches G only: the dust column moves with RV and
                # the RV column with both AV and RV.
                nl_ = head.shape[0]
                dB = jnp.zeros_like(G).at[nl_ + 1].add(drv * Pd) \
                                     .at[nl_ + 2].add(dav * Pd + drv * Qd)
                dG = dG + dB
            if not fit_z:
                return dS, dG, dT
            # Sz has S's form with d(bw)/dz for the weights
            dSz = (dph * (Tz + C * jnp.dot(Gz, v)) + C * jnp.dot(Gz, u)
                   + dz * Szz)
            return dS, dG, dT, dSz

        def _raw(t0, theta, Ds, av, rv, eps_tform, mjd, z, bw, bwz, bwzz, lo):
            B, hw, dhw, _, hwo, _ = _spectra(t0, theta, Ds, av, rv, eps_tform,
                                             mjd, z, bw, lo)
            out = (jnp.sum(hw), hw @ B, jnp.sum(dhw))
            if fit_z:
                bzw = lax.dynamic_slice(bwz, (lo,), (width,))
                out = out + (jnp.dot(hwo, bzw),)
            return out

        @jax.custom_jvp
        def inner(t0, theta, Ds, av, rv, eps_tform, mjd, z, bw, bwz, bwzz, lo):
            return _raw(t0, theta, Ds, av, rv, eps_tform, mjd, z, bw, bwz,
                        bwzz, lo)

        @inner.defjvp
        def inner_jvp(primals, tangents):
            sg = jax.lax.stop_gradient
            out = _raw(*[sg(x) for x in primals[:6]], *primals[6:])
            return out, _tangent(primals, tangents[:6], *out[:3], tangents[7],
                                 *out[3:])

        @jax.custom_jvp
        def outer(t0, theta, Ds, av, rv, eps_tform, mjd, z, bw, bwz, bwzz, lo):
            return inner(t0, theta, Ds, av, rv, eps_tform, mjd, z, bw, bwz,
                         bwzz, lo)[0]

        @outer.defjvp
        def outer_jvp(primals, tangents):
            out = inner(*primals)
            return out[0], _tangent(primals, tangents[:6], *out[:3],
                                    tangents[7], *out[3:])[0]

        return outer

    sim_model_index = 13                       # MODEL_BAYESN, the C-code one

    def sed_params_from_fit(self, params_arr, redshift):
        """Fit-vector layout: the eps_tform block passes as one array."""
        arr = params_arr[1:]
        params = {'theta': arr[0], 'Ds': arr[1], 'av': arr[2]}
        if self.rv_prior_type == 'gaussian':
            params['rv'] = arr[3]
            params['eps_tform'] = arr[4:]
        else:
            params['eps_tform'] = arr[3:]
        params['t0'] = params_arr[0]
        params['redshift'] = redshift
        return params

    distance_param = 'Ds'                      # mu + grey scatter

    m0_reference = 0.0                        # MU is already a distance modulus
    lcpar_names = ("MU",)

    fitres_aliases = {
        "MU": ("DLMAG", "MU_LCFIT"),
        "MUERR": ("DLMAGERR", "MU_LCFITERR", "MUERR_LCFIT"),
        "THETA": ("THETA",),
        "AV": ("AV",),
    }
    nuisance_spec = {}
    mucovscale_bins = {"AV": 3}
    ccprior_bins = {"AV": 1, "THETA": 1}
    bbc_input_aliases = {
        "nbinc_mucovscale": ("mucovscale_bins", "AV"),
        "nsbin_ccprior": ("ccprior_bins", "THETA"),
        "ncbin_ccprior": ("ccprior_bins", "AV"),
    }
    bbc_cut_bits = {"THETA": (1, 14), "AV": (2, 15)}
    biascor_binning = {"THETA": (-4.0, 4.0, 0.5),
                       "AV": (0.0, 1.0, 0.10)}            # SALT2mu.c:486-489
    cut_aliases = {
        "thetamin": ("THETA", "lo"), "thetamax": ("THETA", "hi"),
        "avmin": ("AV", "lo"), "avmax": ("AV", "hi"),
        "logmass_min": ("logmass", "lo"), "logmass_max": ("logmass", "hi"),
    }
    # thetamin/thetamax and avmin/avmax (SALT2mu.c:5739-5742)
    default_cuts = {"THETA": (-3.0, 3.0), "AV": (0.0, 1.0)}

    @classmethod
    def compute_mu(cls, lcpar, nuis, host, u_flags, apply_mass_step=True):
        return lcpar["MU"]

    @classmethod
    def read_lcpar(cls, cols, inputs):
        lcpar = {"MU": cols["MU"], "THETA": cols["THETA"], "AV": cols["AV"]}
        return lcpar, (cols["MUERR"] ** 2)[:, None, None], None

    @classmethod
    def sim_lcpar(cls, tv):
        # true mu at the observed zHD: SIM_MUz = tv.mumodel (SALT2mu.c:11283)
        return {"MU": np.asarray(tv.mumodel)}

    def derived_parameters(self, params, keys_for, **kwargs):
        """BayeSN's own per-SN draws beside the drawn theta: deltam (grey
        offset) and whitened epsilon draws (generate_sed applies
        L_Sigma_epsilon).

        DELTAM is the model's own grey scatter, SIGMA0 * Gaussian
        (genSCATTER_BAYESN, genmag_BAYESN.c:1361-1372), not the generic
        GENGAUSS_DELTAM population. Each component is drawn only when its
        GENMODEL_MSKOPT scatter bit enables it (set_mskopt); disabled draws
        are deterministic zeros so the SED kernel stays shape-stable.

        The SED-facing grey/distance parameter is the combined Ds = mu + deltam
        (the fit's parameterisation); deltam is kept for the output headers.
        """
        mu = params['mu']
        n_sne = mu.shape[0]
        if self.scatter_deltam:
            deltam = self.sigma0 * jax.vmap(jax.random.normal)(
                keys_for('DELTAM'))
        else:
            deltam = jnp.zeros(n_sne)
        n_knots = (self.n_lam_knots - 2) * self.n_tau_knots
        if self.scatter_epsilon:
            nu = jax.vmap(lambda key: jax.random.normal(key, shape=(n_knots,)))(
                keys_for('EPSILON'))
        else:
            nu = jnp.zeros((n_sne, n_knots))
        return {
            'deltam': deltam,
            'Ds': mu + deltam,
            'eps_tform': nu,
        }

    def fitres_param_columns(self, param_names, physical_only=False):
        """Ds is written as MU_LCFIT, the distance spelling SALT2mu accepts
        beside DLMAG (SALT2mu.c:7786). The N_SIG epsilon spline coefficients
        are written by default (available for downstream jax analyses);
        ``physical_only`` drops them for a SALT2mu-readable FITRES (SNANA
        omits them, and 54+ columns overflow SALT2mu's table writer)."""
        if physical_only:
            keep = ['PKMJD', 'THETA', 'Ds', 'AV']
            if self.rv_prior_type == 'gaussian':
                keep.append('RV')
            names = [n for n in keep if n in param_names]
        else:
            names = list(param_names)
        return [(n, 'MU_LCFIT' if n == 'Ds' else n) for n in names]

    def get_fitres_param_columns(self, params, param_errors, redshifts, mu_ref,
                                 physical_only=False):
        """The columns of :meth:`fitres_param_columns`; MU_LCFIT is the
        posterior-mean distance extracted from Ds."""
        cols = []
        for name, column in self.fitres_param_columns(list(params), physical_only):
            if column == 'MU_LCFIT':
                mu, mu_err = self._extract_mu(params[name], mu_ref,
                                              param_errors[name])
                cols.append((column, mu, mu_err))
            else:
                cols.append((column, params[name], param_errors[name]))
        return cols

    def _extract_mu(self, Ds, mu_ref, Ds_err):
        """Posterior-mean mu and its error from Ds under a mu ~ N(mu_ref, 5^2)
        prior."""
        muhat = np.asarray(mu_ref)
        Ds = np.asarray(Ds)
        muhat_err = 5.0
        denom = muhat_err**2 + self.sigma0**2
        mu = (Ds * muhat_err**2 + muhat * self.sigma0**2) / denom
        mu_err = np.sqrt(np.asarray(Ds_err)**2 + self.sigma0**2)
        return mu, mu_err
