"""Shared application logging, console formatting and fatal-error reporting.

Every fatal error raises :class:`SNANAError`, and :func:`aborting` turns one
into SNANA's abort banner and exit status. The console helpers reproduce
SNANA's layout (print_banner, sntools.c): 66-``*`` banners titled with the
section, tab-indented ``key : value`` blocks, and an ``OUTPUT_SUMMARY`` /
``DONE with`` block. They emit through the caller's own logger, so an entry
point's ``basicConfig(format="%(message)s")`` is unchanged, and colour the
text only when stdout is a terminal and ``NO_COLOR`` is unset.
"""

import functools
import logging
import os
import subprocess
import sys
from pathlib import Path


DR_EVIL_ABORT_FACE = r"""

   `|```````|`
   <| o\ /o |>
    | ' ; ' |
    |  ___  |     ABORT program on Fatal Error.
    | |' '| |
    | `---' |
    \_______/

"""


class SNANAError(Exception):
    """The exception every fatal error in jax_SNANA raises, formatting its
    message with the Dr Evil Abort Face."""

    def __init__(self, message: str, function_name: str = None):
        """``function_name`` names the function the error is raised from, if
        any."""
        self.message = message
        self.function_name = function_name
        super().__init__(f"{function_name}: {message}" if function_name else message)

    def _format_error(self) -> str:
        """Format error message with Dr Evil Abort Face."""
        lines = [DR_EVIL_ABORT_FACE]

        if self.function_name:
            lines.append(f" FATAL ERROR ABORT called by {self.function_name}")
        else:
            lines.append(" FATAL ERROR ABORT")

        lines.append(f"   {self.message}")
        lines.append("")

        return "\n".join(lines)


def aborting(main):
    """A program's main: input errors print a banner and exit 1, as
    SNANA's abort does, instead of a Python traceback."""
    @functools.wraps(main)
    def run(*args, **kwargs):
        try:
            return main(*args, **kwargs)
        except (SNANAError, FileNotFoundError) as err:
            if isinstance(err, FileNotFoundError):
                err = SNANAError(str(err))
            print(err._format_error(), flush=True)
            raise SystemExit(1) from None
    return run


_BANNER_WIDTH = 66

_CODES = {
    'cyan': '\033[36m',
    'yellow': '\033[33m',
    'green': '\033[32m',
    'bold': '\033[1m',
    'reset': '\033[0m',
}

_use_color = sys.stdout.isatty() and 'NO_COLOR' not in os.environ


def _color(text: str, name: str) -> str:
    if not _use_color:
        return text
    return f"{_CODES[name]}{text}{_CODES['reset']}"


def banner(logger: logging.Logger, title: str) -> None:
    """Blank line, 66-`*` rule, indented title, 66-`*` rule (print_banner)."""
    rule = ' ' + '*' * _BANNER_WIDTH
    logger.info('')
    logger.info(rule)
    logger.info('   ' + _color(title, 'cyan'))
    logger.info(rule)


def kv(logger: logging.Logger, key: str, value, comment: str | None = None,
       width: int = 20) -> None:
    """Tab-indented ``key : value`` init-block row, with optional ``# comment``."""
    row = f"\t {key:<{width}}: {value}"
    if comment:
        row += f"   # {comment}"
    logger.info(row)


def line(logger: logging.Logger, msg: str, indent: int = 1) -> None:
    """Tab-indented plain line."""
    logger.info('\t' * indent + ' ' + msg)


def warn(logger: logging.Logger, subsystem: str, msg: str) -> None:
    """``<SUBSYSTEM> WARNING: <msg>`` (yellow), at WARNING level."""
    logger.warning(_color(f" {subsystem} WARNING: {msg}", 'yellow'))


def summary(logger: logging.Logger, title: str, rows) -> None:
    """Banner + right-aligned ``KEY: value  # comment`` block (OUTPUT_SUMMARY).

    ``rows`` is an iterable of ``(key, value)`` or ``(key, value, comment)``.
    """
    banner(logger, title)
    items = [(r[0], r[1], r[2] if len(r) > 2 else None) for r in rows]
    kw = max((len(k) for k, _, _ in items), default=0)
    for key, value, comment in items:
        row = f"    {key + ':':<{kw + 1}} {value}"
        if comment:
            row += f"   # {comment}"
        logger.info(row)


def done(logger: logging.Logger, name: str) -> None:
    """`` DONE with <name>.`` (green)."""
    logger.info(_color(f" DONE with {name}.", 'green'))


def version_string() -> str:
    """``jax-<git short-SHA>``; ``jax`` if the working tree is not a git clone."""
    try:
        out = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'], capture_output=True,
            text=True, cwd=Path(__file__).parent, timeout=2)
        sha = out.stdout.strip()
        return f"jax-{sha}" if out.returncode == 0 and sha else "jax"
    except (OSError, subprocess.TimeoutExpired):
        return "jax"


def startup(logger: logging.Logger, prog: str, argv) -> None:
    """SNANA-style startup block: command, version, banner, SNDATA_ROOT."""
    # Optional backend probes are useful in verbose logs, but a missing TPU
    # library on a CPU machine is not a failed run. Warnings still pass through.
    logging.getLogger("jax._src.xla_bridge").setLevel(
        logging.DEBUG if logger.isEnabledFor(logging.DEBUG) else logging.WARNING)
    logger.info(f"\n Full command: {' '.join(argv)}")
    logger.info(f" VERSION: {version_string()}")
    banner(logger, f"Begin execution of {prog} (jax)")
    logger.info(f" SNDATA_ROOT = {os.environ.get('SNDATA_ROOT', '(unset)')}")
