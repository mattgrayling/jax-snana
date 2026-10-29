"""JAX reimplementation of SNANA's SALT2mu / BBC distance-fit stage.

Consumes FITRES tables from the fitter, applies simulation-derived bias
corrections, and fits the global nuisance parameters (alpha, beta, gamma
mass-step, sigma_int) plus a distance offset per redshift bin.

The input key `double_precision` chooses the frame and defaults to single
(`jax_snlc_core.utils`); importing this package fixes nothing.
"""

__version__ = "0.1.0"
__author__ = "Matthew Grayling"

from .config import BBCConfig, build_config, parse_bbc_input
from .output import BBCOutput
from .run import DistanceFitResult, fit_distances, run_bbc, run_bbc_fit, run_bbc_grid
from .binning import ZBinGrid

__all__ = ["BBCConfig", "BBCOutput", "ZBinGrid", "parse_bbc_input",
           "fit_distances", "DistanceFitResult", "run_bbc", "run_bbc_fit", "run_bbc_grid", "build_config"]
