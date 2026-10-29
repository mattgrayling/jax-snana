"""CLI for the JAX rewrite of SNANA WFIT.

Configuration accepts YAML, named options, and SNANA command-line flags.
Precision defaults to single unless double_precision is supplied.
"""

import logging
import sys

from jax_snlc_core import logging as log
from jax_snlc_core.logging import aborting
from jax_snlc_core.utils import configure_precision
from jax_wfit.blinding import parameter_offsets
from jax_wfit.config import parse_command
from jax_wfit.run import run_wfit

logger = logging.getLogger("jax_wfit")


def _report(results, config) -> None:
    """The fitted parameters, one summary block per realisation."""
    offsets = parameter_offsets(config)
    for index, result in enumerate(results):
        title = f"WFIT realisation {index}" if len(results) > 1 else "WFIT"
        rows = [
            ("method", result["method"]),
            ("success", result["diagnostics"]["success"]),
            ("chi2", f"{float(result['likelihood']['total_chi2']):.3f}"),
            ("ndof_hd", result["ndof_hd"]),
        ]
        for name, value in result["summaries"].items():
            shift = offsets[name] if result["blinded"] else 0.0
            rows.append((name,
                         f"{value['value'] + shift:.6f} +/- {value['sigma']:.6f}"))
        log.summary(logger, title, rows)


@aborting
def main(argv: list[str] | None = None) -> int:
    config, verbose = parse_command(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )
    # The frame before the first array is built.
    configure_precision(config.double_precision)
    if verbose:
        logger.setLevel(logging.DEBUG)
    log.startup(logger, "wfit",
                     sys.argv if argv is None else ["wfit"] + argv)
    _report(run_wfit(config), config)
    log.done(logger, "wfit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
