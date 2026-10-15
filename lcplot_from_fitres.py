#!/usr/bin/env python3
"""Write the LCPLOT a fit describes, from its namelist and its FITRES, without
refitting.

The events are named as for a fit, by SNCID_LIST or SNCID_LIST_FILE, and
without either every event of the FITRES is plotted. The light curves load as
the fit loaded them, the model is evaluated at the FITRES parameters, and the
output is the fitter's own LCPLOT and LCLIST:

    python lcplot_from_fitres.py fit.nml FIT.FITRES.TEXT --prefix plots/FIT
    python lcplot_from_fitres.py fit.nml FIT.FITRES.TEXT --prefix SEL SNCID_LIST 123 456
    python lcplot_from_fitres.py fit.nml FIT.FITRES.TEXT --prefix SEL SNCID_LIST_FILE ids.txt

Trailing 'KEY value' pairs override namelist keys, as for snlc_fit.py.
"""

import argparse
import logging
import sys

from jax_snlc_core.utils import snana_log as slog
from jax_snlc_core.utils.error_handling import aborting
from jax_snlc_core.utils.precision import configure_precision

from jax_snlc_fit.config_parser import parse_fit_input_file
from jax_snlc_fit.run import lcplot_from_fitres

logger = logging.getLogger(__name__)


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Writes <prefix>.LCPLOT.TEXT and <prefix>.LCLIST.TEXT for "
                    "the events of a FITRES table, from the namelist that "
                    "fitted them, without refitting: those SNCID_LIST or "
                    "SNCID_LIST_FILE names, or all of them.",
        epilog="Any trailing 'KEY value' pairs override input-file keys, "
               "as snlc_fit.exe accepts them.",
    )
    parser.add_argument("input_file",
                        help="the SNANA-format NML (or YAML) input of the fit")
    parser.add_argument("fitres", help="the fit's FITRES table")
    parser.add_argument("--prefix", required=True,
                        help="output path prefix for the LCPLOT and LCLIST")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable DEBUG logging.")
    return parser


@aborting
def main():
    parser = _build_parser()
    args, overrides = parser.parse_known_args()
    logging.basicConfig(
        format='%(message)s',
        level=logging.DEBUG if args.verbose else logging.INFO,
        stream=sys.stdout,
    )
    slog.startup(logger, "lcplot_from_fitres", sys.argv)
    config = parse_fit_input_file(args.input_file, overrides)
    configure_precision(config.fit.double_precision)
    lcplot_from_fitres(config, args.fitres, args.prefix)
    slog.done(logger, "lcplot_from_fitres")


if __name__ == "__main__":
    main()
