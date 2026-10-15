"""JAX rewrite of SNANA WFIT.

The package consumes a binned or unbinned Hubble diagram and exposes a shared
likelihood to Laplace, NUTS, and grid inference backends.

The precision frame is chosen by the `-double_precision` option alone, not by
the YAML configuration file, and defaults to single
(`jax_snlc_core.utils.precision`); importing this package fixes nothing.
"""

__version__ = "0.1.0"

from .api import fit_cosmology
from .config import WFITConfig
from .data import HubbleDiagram, hubble_diagram_from_covariance
from .run import run_wfit

__all__ = ["fit_cosmology", "WFITConfig", "HubbleDiagram",
           "hubble_diagram_from_covariance", "run_wfit"]
