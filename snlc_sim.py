"""Command-line entry point for the JAX vectorised SNANA simulator.

`DOUBLE_PRECISION: 1` in the input file, or on the command line, runs the
simulation in double precision; it defaults to single. The output tables are
single precision either way, because SNANA stores them that way.
"""

import logging
import sys
import time

from jax_snlc_core import logging as log
from jax_snlc_core.logging import aborting
from jax_snlc_core.utils import configure_precision
from jax_snlc_sim import run as sim_run
from jax_snlc_sim.output import save_snana_outputs
from jax_snlc_sim.inputs import parse_cli_args, command_line_config

logger = logging.getLogger(__name__)


@aborting
def main(argv=None) -> None:
    args = parse_cli_args(argv)
    logging.basicConfig(
        format="%(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
        stream=sys.stdout,
    )
    log.startup(logger, "snlc_sim", sys.argv)

    # Include configuration parsing in the reported initialization time.
    timings = {}
    start = time.perf_counter()
    snana_config = command_line_config(args)
    configure_precision(snana_config.run.double_precision)

    sim = sim_run.prepare(snana_config)
    timings['init'] = time.perf_counter() - start
    # as set_TIMERS(1) prints it at the end of setup (snlc_sim.c:14727)
    log.line(logger, f"CPUTIME_INITIALIZE   = {timings['init']:.3f} sec")

    # SIMLIB_DUMP, DASHBOARD, INIT_ONLY and SIMGEN_DUMP: 0 stop after init
    if sim is None:
        return

    output_name = snana_config.output.genversion
    if not output_name:
        logger.error("No output name: set GENVERSION in the input file or pass --output.")
        sys.exit(1)

    log.banner(logger, "Generate lightcurves")
    start = time.perf_counter()
    results, selection, spectra = sim_run.run_simulation(sim)
    timings['generate'] = time.perf_counter() - start

    log.banner(logger, "Write SNANA output")
    save_snana_outputs(
        sim, results, output_name, selection, spectra_by_sn=spectra,
        output_dir=snana_config.output.path_sndata_sim or None, timings=timings)


if __name__ == "__main__":
    main()
