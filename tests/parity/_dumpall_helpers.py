"""Shared helpers for SNANA-vs-JAX SIMGEN_DUMPALL parity tests.

Used by ``test_dumpall.py`` and ``test_searcheff_spec_scale_replay.py``.
Lives in a non-test module so both files import from a real location
rather than reaching across test modules.
"""

from pathlib import Path

import numpy as np


# Per-column float tolerance. Integer columns use exact equality.
# GENZ/LIBID/PEAKMJD are overrides (should match to float precision); MU is derived.
FLOAT_TOLERANCES = {
    "GENZ":               (1e-5, 1e-6),
    "SIM_REDSHIFT_CMB":   (1e-5, 1e-6),
    "ZHELIO":             (1e-5, 1e-6),
    "RA":                 (1e-4, 1e-4),
    "DEC":                (1e-4, 1e-4),
    "PEAKMJD":            (1e-4, 1e-3),
    "MU":                 (1e-3, 1e-2),
    "MWEBV":              (1e-3, 1e-4),
    "SIM_MWEBV":          (1e-3, 1e-4),
    "MWEBVERR":           (1e-3, 1e-4),
    # rest-frame epochs from the replayed peak and redshift, in days
    "TRESTMIN":           (1e-4, 1e-3),
    "TRESTMAX":           (1e-4, 1e-3),
    "TGAPMAX":            (1e-4, 1e-3),
}
# Columns by prefix. The surface brightness sits at the SN's offset from its
# host, which replay reconstructs from float32 coordinates. An SNR maximum
# that differs by less than 0.05 is a flux difference well inside the flux
# error, which is where the faint-UV model fluxes of the two codes part.
FLOAT_TOLERANCE_PREFIXES = {
    "SBFLUX_": (0.02, 0.05),
    "SBMAG_":  (1e-3, 0.03),
    "SNRMAX":  (2e-3, 0.05),
}


def float_tolerance(name):
    if name in FLOAT_TOLERANCES:
        return FLOAT_TOLERANCES[name]
    for prefix, tolerance in FLOAT_TOLERANCE_PREFIXES.items():
        if name.startswith(prefix):
            return tolerance
    return 1e-4, 1e-6

# Selection/trigger columns are stochastic (independent SNANA vs JAX RNGs);
# allow up to 5% row-level disagreement on these. The epoch counts join
# them: the replayed peak (three decimals in the dump) can move an epoch
# across the GENRANGE_TREST edge.
STOCHASTIC_INT_COLUMNS = {
    "CUTMASK", "SIM_SEARCHEFF_MASK", "SIM_EFFMASK", "FLAG_ACCEPT", "SNTYPE",
    "NEPOCH", "NOBS",
}
STOCHASTIC_DIFF_FRAC = 0.05


def parse_dump(path: Path):
    """Parse a SNANA .DUMP file. Returns ``(varnames, {col: np.ndarray})``."""
    varnames = None
    rows = []
    with path.open() as f:
        for line in f:
            if line.startswith("VARNAMES:"):
                varnames = line.split(":", 1)[1].split()
            elif line.startswith("SN:"):
                rows.append(line.split()[1:])
    if varnames is None:
        return None, {}
    cols = {name: [] for name in varnames}
    for row in rows:
        for i, name in enumerate(varnames):
            cols[name].append(row[i])
    out = {}
    for name, vals in cols.items():
        try:
            out[name] = np.array(vals, dtype=np.int64)
        except ValueError:
            try:
                out[name] = np.array(vals, dtype=np.float64)
            except ValueError:
                out[name] = np.array(vals)
    return varnames, out


def compare_dumps(snana_varnames, snana_cols, jax_varnames, jax_cols):
    """Raise AssertionError on any disagreement, with a verbose summary."""
    assert snana_varnames == jax_varnames, (
        f"VARNAMES mismatch.\n  SNANA: {snana_varnames}\n  JAX:   {jax_varnames}"
    )

    def _unique_first(cids):
        seen = {}
        for i, c in enumerate(cids):
            k = int(c)
            if k not in seen:
                seen[k] = i
        return seen

    s_idx = _unique_first(snana_cols["CID"])
    j_idx = _unique_first(jax_cols["CID"])
    shared = sorted(set(s_idx) & set(j_idx))
    assert len(shared) == len(s_idx), (
        f"only {len(shared)} of SNANA's {len(s_idx)} CIDs appear in jax's dump; "
        f"the column comparison would cover a fraction of the rows"
    )
    assert len(j_idx) == len(s_idx), (
        f"jax wrote {len(j_idx)} CIDs where SNANA wrote {len(s_idx)}"
    )
    s_order = np.array([s_idx[c] for c in shared])
    j_order = np.array([j_idx[c] for c in shared])

    failures: list[str] = []
    for name in snana_varnames:
        s = snana_cols[name][s_order]
        j = jax_cols[name][j_order]
        if np.issubdtype(s.dtype, np.integer) and np.issubdtype(j.dtype, np.integer):
            n_diff = int((s != j).sum())
            if name in STOCHASTIC_INT_COLUMNS:
                frac = n_diff / len(s)
                if frac > STOCHASTIC_DIFF_FRAC:
                    failures.append(
                        f"{name} (int, stochastic): {n_diff}/{len(s)} differ "
                        f"({frac:.1%} > {STOCHASTIC_DIFF_FRAC:.0%} tol)"
                    )
            elif n_diff > 0:
                failures.append(f"{name} (int): {n_diff}/{len(s)} differ")
        elif s.dtype.kind in 'OUS' or j.dtype.kind in 'OUS':
            # FIELD is the SIMLIB entry's name, compared as written
            n_diff = int((s.astype(str) != j.astype(str)).sum())
            if n_diff > 0:
                failures.append(f"{name} (text): {n_diff}/{len(s)} differ")
        else:
            s_f = s.astype(np.float64, copy=False)
            j_f = j.astype(np.float64, copy=False)
            rtol, atol = float_tolerance(name)
            close = np.isclose(s_f, j_f, rtol=rtol, atol=atol, equal_nan=True)
            if not np.all(close):
                n_diff = int((~close).sum())
                max_abs = float(np.max(np.abs(s_f - j_f)))
                failures.append(
                    f"{name} (float): {n_diff}/{len(s)} outside "
                    f"(rtol={rtol}, atol={atol}); max|Δ|={max_abs:.3e}"
                )

    assert not failures, "Per-column disagreements:\n  " + "\n  ".join(failures)
