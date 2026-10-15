"""SNANA's fatal-error abort message: :class:`SNANAError`, the exception
every abort in jax_SNANA raises, displaying SNANA's Dr Evil Abort Face.
"""

import functools

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
