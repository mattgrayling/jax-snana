"""Command-line entry point for the JAX vectorised SNANA simulator.

`DOUBLE_PRECISION: 1` in the input file, or on the command line, runs the
simulation in double precision; it defaults to single. The output tables are
single precision either way, because SNANA stores them that way.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from jax_snlc_sim import run as sim_run
from jax_snlc_sim.inputs import parse_input, read_input, command_settings, bootstrap_settings
from jax_snlc_core.utils.input_settings import merge_settings
from jax_snlc_core.utils import snana_log as slog
from jax_snlc_core.utils.error_handling import SNANAError, aborting
from jax_snlc_core.utils.precision import configure_precision

logger = logging.getLogger(__name__)


def _resolve_input_and_overrides(args: argparse.Namespace):
    """The input file and the override tokens, whichever CLI form was used.

    With --input, argparse puts the first override token in the positional slot.
    """
    if args.input:
        overrides = list(args.overrides)
        if args.input_file:
            overrides.insert(0, args.input_file)
        return args.input, overrides
    return args.input_file, list(args.overrides)


def _parse_cli_args(argv=None) -> argparse.Namespace:
    settings, remaining = bootstrap_settings(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False,
        description="Run the JAX vectorised SNANA simulation pipeline. Needs "
                    "SNDATA_ROOT (kcor files, models, SIMLIBs) from the "
                    "environment; writes "
                    "the data version to ./<GENVERSION>/ in the working "
                    "directory."
    )
    parser.add_argument(
        "input_file", nargs="?", default=None,
        help="Path to the SNANA .input configuration file (e.g., INP_SIMGEN_*.input).",
    )
    parser.add_argument(
        "--input", "-i", default=None,
        help="The input file, as a flag instead of a positional argument.",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="GENVERSION name for the outputs (defaults to GENVERSION from the input file).",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=None,
        help="Alias for --batch-size.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed override (defaults to RANSEED in the input file).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging.",
    )
    args, args.overrides = parser.parse_known_args(remaining)
    args.settings = settings
    return args


@aborting
def main(argv=None) -> None:
    args = _parse_cli_args(argv)
    logging.basicConfig(
        format="%(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
        stream=sys.stdout,
    )
    slog.startup(logger, "snlc_sim", sys.argv)

    input_arg, overrides = _resolve_input_and_overrides(args)
    if not input_arg and any(flag in overrides for flag in ("-h", "--help")):
        command_settings(overrides, args.settings)
    if not input_arg:
        logger.error("No input file: pass it as the first argument, or with --input.")
        sys.exit(1)
    snana_input = Path(input_arg).expanduser().resolve()
    if not snana_input.is_file():
        logger.error(f"Input file not found: {snana_input}")
        sys.exit(1)

    # seconds per phase for the OUTPUT_SUMMARY; init is SNANA's t_start to t_end_init
    timings = {}
    start = time.perf_counter()
    file_settings = read_input(snana_input)
    if file_settings is None:
        from jax_snlc_sim.config_parser import read_keys
        selectors = {'genmodel': read_keys(snana_input).get('GENMODEL', '')}
    else:
        selectors = file_settings
    named_values, overrides = command_settings(overrides, merge_settings(selectors, args.settings))
    named_values = merge_settings(named_values, args.settings)
    aliases = {key: value for key, value in {
        'random_seed': args.seed, 'batch_size': args.chunk_size,
        'genversion': args.output}.items() if value is not None}
    named_values = merge_settings(named_values, aliases)
    if file_settings is not None:
        if overrides:
            raise SNANAError('YAML simulation inputs take named --options')
        snana_config = parse_input(snana_input, named_values)
    else:
        from jax_snlc_sim.config_parser import parse_snana_input_file
        snana_config = parse_snana_input_file(snana_input, overrides, settings=named_values)
    configure_precision(snana_config.run.double_precision)

    sim = sim_run.prepare(snana_config)
    timings['init'] = time.perf_counter() - start
    # as set_TIMERS(1) prints it at the end of setup (snlc_sim.c:14727)
    slog.line(logger, f"CPUTIME_INITIALIZE   = {timings['init']:.3f} sec")

    # SIMLIB_DUMP, DASHBOARD, INIT_ONLY and SIMGEN_DUMP: 0 stop after init
    if sim is None:
        return

    output_name = snana_config.output.genversion
    if not output_name:
        logger.error("No output name: set GENVERSION in the input file or pass --output.")
        sys.exit(1)

    slog.banner(logger, "Generate lightcurves")
    start = time.perf_counter()
    results, selection, spectra_cache = sim_run.run_simulation(sim)
    timings['generate'] = time.perf_counter() - start

    slog.banner(logger, "Write SNANA output")
    sim_run.save_snana_outputs(sim, results,
                               output_name, selection,
                               spectra_cache=spectra_cache,
                               output_dir=snana_config.output.path_sndata_sim or None,
                               timings=timings)


if __name__ == "__main__":
    main()
