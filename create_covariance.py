"""CLI for the JAX rewrite of SNANA create_covariance.

`DOUBLE_PRECISION: true` in the input YAML runs the build in double precision;
it defaults to single.
"""

import argparse
import logging
import sys

from jax_create_cov.config import CreateCovConfig
from jax_create_cov.config import build_config
from jax_create_cov.config import read_yaml
from jax_create_cov.covariance import create_covariance
from jax_snlc_core import logging as log
from jax_snlc_core.input_settings import (add_settings, dataclass_settings, merge_settings,
                                          parsed_settings, settings_parser)
from jax_snlc_core.logging import SNANAError, aborting
from jax_snlc_core.utils import configure_precision


logger = logging.getLogger(__name__)


def config_from_args(argv=None) -> CreateCovConfig:
    """Parse the command line into a config. Separate from main so the flag
    mapping can be tested without running a fit."""
    settings_argparser = settings_parser(CreateCovConfig, add_help=False)
    named, argv = settings_argparser.parse_known_args(argv)
    parser = argparse.ArgumentParser(
        prog="create_covariance",
        description="Build systematic covariance from a BBC FITOPTxMUOPT grid",
        conflict_handler="resolve", allow_abbrev=False)
    parser.add_argument("config", help="input YAML")
    parser.add_argument("-o", "--outdir", help="override OUTDIR")
    parser.add_argument("-i", "--input_dir", help="override INPUT_DIR")
    parser.add_argument("-v", "--version", help="override VERSION")
    parser.add_argument("--sys_scale_file", help="override SYS_SCALE_FILE")
    parser.add_argument("--mxsize_test_posdef", type=int,
                        help="skip the conditioning check above this HD size")
    parser.add_argument("--write_mask_cov", type=int,
                        help="1 covsys | 2 covtot_inv | 4 covtot | 8 factorized (default 2)")
    parser.add_argument("--write_format_cov", choices=("text", "npz"),
                        help="covariance file format (default npz)")
    parser.add_argument("-u", "--unbinned", action="store_true")
    parser.add_argument("-s", "--subtract_vpec", action="store_true",
                        help="subtract MUERR(VPEC); forces unbinned")
    parser.add_argument("--nbin_x1", type=int, help="rebin: x1 bins")
    parser.add_argument("--nbin_c", type=int, help="rebin: c bins")
    parser.add_argument("--rebin", nargs="*", metavar="COLUMN=COUNT",
                        help="rebin by named FITRES columns; no columns uses IZBIN only")
    parser.add_argument("--nosys", action="store_true", help="restrict the default covariance to the unmodified MUOPT")
    parser.add_argument("-m", "--muopt", type=int, help="use only this MUOPT")
    parser.add_argument("--systematic_HD", action="store_true",
                        help="a Hubble diagram for every systematic")
    parser.add_argument("--label_cov_rows", action="store_true",
                        help="label covariance rows (forces text format)")
    types, owners = dataclass_settings(CreateCovConfig)
    # Named options were consumed above; include their descriptions in full help.
    add_settings(parser, types, owners)
    args = parser.parse_args(argv)

    values = {name: value for name, value in vars(args).items()
              if name not in ('config', 'rebin', 'unbinned', 'systematic_HD')
              and value is not None and value is not False}
    if args.systematic_HD:
        values['systematic_hd'] = True
    if args.rebin is not None:
        bins = {}
        for item in args.rebin:
            name, sep, count = item.partition('=')
            if not sep or not count.isdecimal() or name in bins:
                raise SNANAError(f"expected distinct COLUMN=COUNT entries, got {item!r}")
            bins[name] = int(count)
        values['rebin_bins'] = bins
        values['mode'] = 'rebin'
    elif args.nbin_x1 is not None or args.nbin_c is not None:
        values['mode'] = 'rebin'
    elif args.unbinned:
        values.update(mode='unbinned', rebin_bins=None, nbin_x1=0, nbin_c=0)
    named_values = parsed_settings(named, types, readers={})
    config = build_config(read_yaml(args.config), merge_settings(values, named_values))
    for value, key, flag in ((config.input_dir, "INPUT_DIR", "-i"),
                             (config.version, "VERSION", "-v"),
                             (config.outdir, "OUTDIR", "-o")):
        if not value:
            raise SNANAError(f"no {key}, and none given with {flag}",
                             function_name="config_from_args")
    return config


@aborting
def main(argv=None) -> int:
    logging.basicConfig(format="%(message)s", level=logging.INFO,
                        stream=sys.stdout)
    log.startup(logger, "create_covariance",
                     sys.argv if argv is None else ["create_covariance"] + argv)
    config = config_from_args(argv)
    configure_precision(config.double_precision)
    result = create_covariance(config)
    logger.info(f"wrote {len(result.covariances)} covariance matrices "
                f"to {config.outdir}")
    log.done(logger, "create_covariance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
