"""JAX-based Supernova Light Curve Fitting Package.

GPU-accelerated light-curve fitting for the models that declare fit
parameters (SALT2/SALT3, BayeSN, REDBACK). Models are constructed through
the shared `jax_snlc_core.models.build_model`; a model that declares no fit
parameters (NON1ASED, SIMSED) is rejected explicitly by `snlc_fit`. Works
alongside `jax_snlc_sim` for simulation but keeps fitter and simulator
concerns separate.

Key public exports:
    - LightCurveFitter, FitConfig: main fitting interface
    - fit_static_data, fit_photometry: what the fitter evaluates a model
      against, and the calibrated filters of a data version
"""

from .config import FitConfig
from .fitter import LightCurveFitter, fit_photometry, fit_static_data
from .run import run_fitting

__all__ = [
    "FitConfig",
    "LightCurveFitter",
    "fit_photometry",
    "fit_static_data",
    "run_fitting",
]

from jax_snlc_core import __version__
