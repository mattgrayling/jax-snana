"""JAX-based SNANA light curve simulator.

A vectorised implementation of SNANA's `snlc_sim.exe` for
GPU-accelerated supernova simulation. Shared domain code (models,
physics, photometry, io, utilities) lives in the sibling
`jax_snlc_core` package.

``prepare`` returns one LightCurveSimulator holding the prepared survey.
Shared numerical kernels compile on first use.
"""

from jax_snlc_core import __version__
from .run import (
    LightCurveSimulator, SimulationConfig, prepare, simulate,
)
from .inputs import build_config, parse_input

__all__ = ["LightCurveSimulator", "SimulationConfig",
           "prepare", "simulate", "build_config", "parse_input", "__version__"]

__author__ = "Matthew Grayling"
