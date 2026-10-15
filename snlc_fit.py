#!/usr/bin/env python3
"""Command-line entry point for the JAX SNANA light curve fitter.

Analogous to SNANA's snlc_fit.exe: reads simulated (or real) light curves
from SNANA FITS format, fits them with a specified model, and writes
SNANA-format FITRES text output.

Every setting is a key in the input file, and any of them can be overridden
on the command line as snlc_fit.exe accepts them, so there is one place a
setting can come from:

    python snlc_fit.py fit.nml
    python snlc_fit.py fit.nml NFIT_ITERATION 4 FITWIN_TREST -15 45
    python snlc_fit.py fit.nml BATCH_SIZE 500 OUTPUT_NAME my_results.FITRES.TEXT

`DOUBLE_PRECISION = T` in the namelist runs the fit in double precision; it
defaults to single.
"""

import logging
import sys
import time

from jax_snlc_core.utils import snana_log as slog
from jax_snlc_core.utils.error_handling import aborting
from jax_snlc_core.models import model_input_declarations
from jax_snlc_core.utils.input_settings import parsed_settings, bootstrap_settings
from jax_snlc_core.utils.precision import configure_precision
from jax_snlc_core.utils.config_tracking import UsedTrackingDict
from jax_snlc_fit.config_parser import (apply_overrides, build_file_config,
                                       group_override_tokens, known_keys, read_fit_input)
from jax_snlc_fit.inputs import (argument_parser, declarations,
                                selected_model)
from jax_snlc_fit.run import run_fitting

_T0 = time.perf_counter()

logger = logging.getLogger(__name__)


@aborting
def main(argv=None):
    timings = {'startup': time.perf_counter() - _T0}
    argv = sys.argv[1:] if argv is None else argv
    types, owners, _ = declarations()
    selectors, remaining = bootstrap_settings(
        argv, types, owners, *model_input_declarations('fit'))
    locator = argument_parser(input_file=True, add_help=False)
    initial, legacy_overrides = locator.parse_known_args(remaining)
    input_format, settings = ('yaml', {})
    if initial.input_file is not None:
        input_format, settings = read_fit_input(initial.input_file)
    selection = UsedTrackingDict(settings)
    if input_format == 'snana' and legacy_overrides:
        apply_overrides(selection, group_override_tokens(legacy_overrides, known_keys()))
    model_key = 'FITMODEL_NAME' if input_format == 'snana' else 'fitmodel_name'
    type_key = 'MODEL_TYPE' if input_format == 'snana' else 'model_type'
    model_type = selectors.get('model_type') or selected_model({
        'fitmodel_name': selectors.get('fitmodel_name') or selection.get(model_key),
        'model_type': None if selectors.get('fitmodel_name') else selection.get(type_key)})
    parser = argument_parser(model_type, input_file=True)
    args, overrides = parser.parse_known_args(argv)
    if args.input_file is None:
        parser.error('an input file is required')
    if any(value.startswith('--') for value in overrides):
        parser.error('unrecognized arguments: ' + ' '.join(overrides))
    logging.basicConfig(
        format='%(message)s',
        level=logging.DEBUG if args.verbose else logging.INFO,
        stream=sys.stdout,
    )
    slog.startup(logger, "snlc_fit", sys.argv)

    start = time.perf_counter()
    types, owners, _ = declarations(model_type)
    config = build_file_config(settings, input_format, overrides,
                               named_settings=parsed_settings(args, types, owners))
    configure_precision(config.fit.double_precision)
    timings['config'] = time.perf_counter() - start

    slog.banner(logger, "Fit configuration")
    slog.kv(logger, "input", args.input_file)
    slog.kv(logger, "model", config.model.fitmodel_name)
    slog.kv(logger, "kcor", config.data.kcor_file)
    slog.kv(logger, "iterations", config.fit.n_fit_iterations)

    run_fitting(config, timings=timings, process_start=_T0)
    slog.done(logger, "snlc_fit")


if __name__ == "__main__":
    main()
