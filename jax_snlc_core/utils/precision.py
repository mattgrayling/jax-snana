"""The floating-point frame a run works in.

Every input carries one boolean key, ``DOUBLE_PRECISION``, spelt the way that
input spells its other logicals: ``DOUBLE_PRECISION: 1`` in a sim input,
``DOUBLE_PRECISION = T`` in a fitter namelist, ``double_precision=1`` for
SALT2mu, ``DOUBLE_PRECISION: true`` in a create_covariance YAML, and
``-double_precision 1`` on wfit's command line, which has no input file.
An omitted key is recorded as None, leaving a Python caller's active precision
unchanged. :data:`DEFAULT_DOUBLE_PRECISION` supplies the command-line default.

:func:`configure_precision` is the only place ``jax_enable_x64`` is set. Each
stage's ``main`` calls it once its input is parsed and before the first array
is built; the parsers record the key and touch nothing global, so parsing an
input inside another process, a test's for one, leaves that process's frame
alone. Importing the packages sets nothing.

The frame is the width of JAX's default float: unannotated array constructors,
reductions, and the uniform and normal draws that carry the per-event random
paths. It does not move the explicit ``numpy`` and ``jax.numpy`` ``float32``
dtypes in the simulator, which are there to match the single-precision storage
SNANA reads and writes (SIMLIB columns, HOSTLIB columns, the TABLEVAR
truncation in :mod:`jax_salt2mu.tablevar`, the HEAD and PHOT output tables) and
so must stay 4-byte in either frame. Code that wants the chosen width, rather
than a fixed one, asks :func:`float_dtype` for it.
"""

import logging

import jax
import jax.numpy as jnp

from .error_handling import SNANAError

logger = logging.getLogger(__name__)

#: The precision a command-line run uses when its input omits the key.
DEFAULT_DOUBLE_PRECISION = False


def configure_precision(double: bool | None) -> None:
    """Run in double precision, or in single."""
    if double is None:
        double = DEFAULT_DOUBLE_PRECISION
    jax.config.update("jax_enable_x64", double)
    if double:
        logger.info("Precision: double (float64)")
    else:
        logger.info("Precision: single (float32); DOUBLE_PRECISION selects double")


def float_dtype():
    """The float dtype of the chosen frame."""
    return jnp.result_type(float)


def require_precision(double: bool | None) -> None:
    """Check an explicit request without changing a Python process's precision."""
    if double is not None and double != jax.config.jax_enable_x64:
        raise SNANAError(
            'double_precision conflicts with the active JAX precision. '
            'Call configure_precision before preparing arrays or models, '
            'or omit double_precision to use the active precision.')
