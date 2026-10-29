#!/usr/bin/env python3
"""Command-line entry point for the JAX SNANA light curve fitter.

Analogous to SNANA's snlc_fit.exe: reads simulated (or real) light curves
from SNANA FITS data or a Rubin HATS catalogue, fits them with a specified
model, and writes SNANA-format FITRES text output.

Every setting is a key in the input file, and any of them can be overridden
on the command line, as snlc_fit.exe accepts them or as the named options
``--help`` lists, in any mix:

    python snlc_fit.py fit.nml
    python snlc_fit.py fit.nml NFIT_ITERATION 4 FITWIN_TREST -15 45
    python snlc_fit.py fit.nml BATCH_SIZE 500 --output-name my_results.FITRES.TEXT

`DOUBLE_PRECISION = T` in the namelist runs the fit in double precision; it
defaults to single.
"""

import logging
import sys
import time

from jax_snlc_core import logging as log
from jax_snlc_core.logging import aborting
from jax_snlc_core.utils import configure_precision
from jax_snlc_fit.config_parser import command_line_input, parse_fit_input_file
from jax_snlc_fit.run import run_fitting

_T0 = time.perf_counter()

logger = logging.getLogger(__name__)


@aborting
def main(argv=None):
    timings = {'startup': time.perf_counter() - _T0}
    argv = sys.argv[1:] if argv is None else argv
    input_file, verbose, overrides = command_line_input(argv)
    logging.basicConfig(
        format='%(message)s',
        level=logging.DEBUG if verbose else logging.INFO,
        stream=sys.stdout,
    )
    log.startup(logger, "snlc_fit", sys.argv)

    start = time.perf_counter()
    config = parse_fit_input_file(input_file, overrides)
    configure_precision(config.fit.double_precision)
    timings['config'] = time.perf_counter() - start

    log.banner(logger, "Fit configuration")
    log.kv(logger, "input", input_file)
    log.kv(logger, "model", config.model.fitmodel_name)
    log.kv(logger, "kcor", config.data.kcor_file)
    log.kv(logger, "iterations", config.fit.n_fit_iterations)

    run_fitting(config, timings=timings, process_start=_T0)
    log.done(logger, "snlc_fit")


if __name__ == "__main__":
    main()
