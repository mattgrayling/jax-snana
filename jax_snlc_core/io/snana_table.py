"""SNANA text-table writer, shared across the fitter and SALT2mu/BBC outputs.

Format-generic serialiser for SNANA's ``VARNAMES:`` + rowkey text tables
(``sntools_output_text.c:600-761``): FITRES (``SN:``), M0DIF (``ROW:``),
SNANA/OUTLIER/LCPLOT, and BBC diagnostic dumps all go through it.
"""

import logging

import numpy as np

from jax_snlc_core import __version__
from jax_snlc_core.utils.snana_log import version_string
from jax_snlc_core.utils import snana_log as slog

logger = logging.getLogger(__name__)


DISTRIBUTION = "jax-snana"

# must parse as >= 11.02 for the VPEC sign (correct_sign_vpec_data)
SNANA_VERSION_STRING = "v12_02d-12-g74dfb2c5-JAX"


def snana_version_comment():
    """The SNANA_VERSION header line (sntools_output_text.c): the one value
    every jax output carries, which readers parse as a release."""
    return f"SNANA_VERSION: {SNANA_VERSION_STRING}"


def program_comment(program):
    """The PROGRAM header line SNANA's tables carry after SNANA_VERSION
    (sntools_output_text.c:721, SALT2mu.c:23555), naming the script that
    wrote the file."""
    return f"PROGRAM:       {program}"


def jax_version_comment():
    """The line that says which jax wrote the file, beside SNANA_VERSION."""
    return f"JAX_VERSION: {DISTRIBUTION} {__version__} {version_string()}"


def snana_value_strings(name, values):
    """Format a float column as SNANA's text writer does.

    The choice is per value, not per column, so one column can hold both
    ``-9`` and ``2.29797e-05`` (sntools_output_text.c:467-495).
    """
    v = np.asarray(values, dtype=float)
    if "MJD" in name:
        return np.char.mod("%.4f", v)
    if name[:1] == "z":
        return np.char.mod("%.5f", v)

    finite = np.isfinite(v)
    truncated = np.where(finite, v, 0.0).astype(np.int64)
    fmt = np.full(v.shape, "%.5e", dtype=object)
    fmt[v > 1.0e10] = "%.4e"
    fmt[(v > 0.1) & (v <= 1.0e10)] = "%.5f"
    # an exactly integral value prints as an integer, whatever its magnitude
    integral = finite & (v - truncated == 0.0)
    fmt[integral] = "%d"
    # coordinates outrank the tests above, but only in their own range
    if "RA" in name or "DEC" in name:
        fmt[np.abs(v) < 400] = "%.6f"
    out = np.empty(v.shape, dtype=object)
    for spec in np.unique(fmt):
        rows = fmt == spec
        src = truncated if spec == "%d" else v
        out[rows] = [spec % x for x in src[rows]]
    return out.astype(str)


def write_varnames_table(output_path, varnames, columns, fmt, style='key',
                         rowkey='SN:', header_comments=(),
                         blank_separator=True, post_varnames_comments=()):
    """Write an SNANA text table (sntools_output_text.c:600-761 formats).

    ``columns`` maps varname -> array; ``fmt`` maps varname -> printf format.
    A column it leaves out, or maps to None, is written as SNANA's own text
    writer would: strings as they are, numbers by the per-value rule.
    Styles: key (# comments + VARNAMES: + rowkey rows), csv, col, none.
    """
    if style == 'none':
        return
    str_cols = []
    for name in varnames:
        data = np.asarray(columns[name])
        spec = fmt.get(name)
        if spec == '%d':
            str_cols.append(data.astype(np.int64).astype(str))
        elif spec == '%s' or (spec is None and data.dtype.kind in 'OUS'):
            str_cols.append(data.astype(str))
        elif spec is None:
            str_cols.append(snana_value_strings(name, data))
        else:
            str_cols.append(np.char.mod(spec, data))

    sep = ',' if style == 'csv' else ' '
    # explicit dtype: an empty table must stay a str array
    lines = np.array([sep.join(row) for row in zip(*str_cols)],
                     dtype=np.str_) if str_cols else np.empty(0, dtype=np.str_)

    with open(output_path, 'w') as f:
        if style == 'key':
            for line in header_comments:
                # None is a truly blank separator line; "" is a bare "# "
                f.write("\n" if line is None else f"# {line}\n")
            if header_comments and blank_separator:
                f.write("#\n")
            f.write(f"VARNAMES: {' '.join(varnames)}\n")
            # after VARNAMES, as write_fitres_driver
            for line in post_varnames_comments:
                f.write("\n" if line is None else f"# {line}\n")
            lines = np.char.add(f'{rowkey} ', lines)
        elif style == 'csv':
            f.write(','.join(varnames) + '\n')
        f.write('\n'.join(lines))
        f.write('\n')
    slog.line(logger, f"Wrote {len(lines)} rows to {output_path}")
