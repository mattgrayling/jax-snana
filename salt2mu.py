"""JAX reimplementation of SNANA's SALT2mu / BBC distance-fit stage.

Usage:
    salt2mu <input_file> [key=value ...]
    salt2mu --input <input_file> [key=value ...] [-v]

The input file and trailing ``key=value`` tokens follow SNANA's SALT2mu syntax;
command-line tokens override the file.

`double_precision=1` runs the fit in double precision; it defaults to single.
"""

import argparse
import logging
import sys

from jax_salt2mu.config import bootstrap_settings, parse_bbc_input
from jax_salt2mu.run import run_bbc
from jax_snlc_core import logging as log
from jax_snlc_core.logging import aborting
from jax_snlc_core.utils import configure_precision

logger = logging.getLogger(__name__)


def _parse_cli_args(argv):
    settings, remaining = bootstrap_settings(argv)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False,
        description="JAX SALT2mu/BBC: reads FITRES, fits distances, writes "
                    "<prefix>.M0DIF, .FITRES and .COV, prefix= being an input "
                    "key. Trailing key=value pairs override input-file keys.",
    )
    parser.add_argument("input", nargs="?", default=None,
                        help="SALT2mu input file (SNANA syntax)")
    parser.add_argument("--input", dest="input_opt", default=None,
                        help="SALT2mu input file (alternative to positional)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable DEBUG logging.")
    args, overrides = parser.parse_known_args(remaining)
    input_file = args.input
    if args.input_opt is not None:
        # argparse put the first override token in the positional
        if args.input is not None:
            overrides.insert(0, args.input)
        input_file = args.input_opt
    cat_prescale = _parse_cat_only(input_file)
    # key=value, NULL/BLANK (SALT2mu.c:19321) and cat_only are not filenames
    if input_file is not None:
        if "=" in input_file:
            overrides.insert(0, input_file)
            input_file = None
        elif input_file in ("NULL", "null", "BLANK") or cat_prescale is not None:
            input_file = None
    return input_file, overrides, args.verbose, cat_prescale, settings


def _parse_cat_only(token):
    """Prescale from a ``cat_only`` / ``cat_only/<N>`` argument, else None.

    SNANA matches the substring anywhere in the parFile argument
    (parse_parFile :19314, parse_cat_only :20154).
    """
    if token is None or "cat_only" not in token:
        return None
    _, _, prescale = token.partition("/")
    return int(prescale) if prescale else 1


@aborting
def main():
    input_file, overrides, verbose, cat_prescale, settings = _parse_cli_args(sys.argv[1:])
    logging.basicConfig(
        format="%(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
        stream=sys.stdout,
    )
    log.startup(logger, "salt2mu", sys.argv)
    config = parse_bbc_input(input_file, overrides, settings=settings)
    configure_precision(config.double_precision)
    run_bbc(config, cat_prescale)


if __name__ == "__main__":
    main()
