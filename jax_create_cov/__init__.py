"""JAX reimplementation of SNANA's create_covariance.py.

Aggregates a FITOPT x MUOPT grid of BBC Hubble diagrams into systematic
covariance matrices, either from a submit_batch output tree or from diagrams
already in memory.

Not carried over from SNANA: the CosmoMC writers and their template/dataset
keys, the csv covariance format, the JLA-vs-BBC ``METHOD`` switch, ``RESTORE_DES``,
``--yaml_file`` (a status file for submit_batch_jobs, which jax has no
equivalent of), and the tracemalloc/debug instrumentation. An input file naming
any of them is warned about rather than silently accepted.

The input YAML's `DOUBLE_PRECISION` key chooses the frame and defaults to single
(`jax_snlc_core.utils.precision`); importing this package fixes nothing.
"""

from .config import Cosmology, Covopt, CreateCovConfig
from .config_parser import apply_filter, load_config, parse_covopt
from .run import CovResult, build_covariance, create_covariance, write_outputs
from .api import compute_covariance

__all__ = [
    "compute_covariance",
    "Cosmology",
    "CovResult",
    "Covopt",
    "CreateCovConfig",
    "apply_filter",
    "build_covariance",
    "create_covariance",
    "load_config",
    "parse_covopt",
    "write_outputs",
]
