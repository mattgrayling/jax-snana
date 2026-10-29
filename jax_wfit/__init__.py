"""JAX rewrite of SNANA WFIT.

The package consumes a binned or unbinned Hubble diagram and exposes a shared
likelihood to Laplace, NUTS, and grid inference backends.

Precision follows the double_precision setting. The CLI defaults to single;
Python calls retain the active precision when it is omitted. Importing this
package does not change precision.
"""

__version__ = "0.1.0"

from .config import WFITConfig
from .hubble import HubbleDiagram, hubble_diagram_from_covariance
from .run import fit_cosmology, run_wfit

__all__ = ["fit_cosmology", "WFITConfig", "HubbleDiagram",
           "hubble_diagram_from_covariance", "run_wfit"]
