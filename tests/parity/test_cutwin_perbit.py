"""SNANA-vs-JAX CUTWIN parity, per-cut.

Bundled fixture has APPLY_CUTWIN_OPT=1 with CUTWIN_NEPOCH/SNRMAX/TRESTMIN/
TRESTMAX active. SNANA writes the per-cut bits into the DUMP CUTMASK column.
This test builds the same CUTMASK from JAX and compares per-bit pass counts.

The cuts are deterministic given truth params + observation noise; with
independent RNGs, per-SN agreement is bounded by SNR-borderline coincidences,
so the assertion is on aggregate pass count per bit.
"""

from pathlib import Path

import numpy as np

from jax_snlc_core.snana_files import open_text
from jax_snlc_sim import run as sim_run
from jax_snlc_sim import cutwin


def _parse_dump_cutmask(path: Path) -> np.ndarray:
    varnames: list[str] | None = None
    rows: list[list[str]] = []
    with open_text(path) as f:
        for line in f:
            if line.startswith("VARNAMES:"):
                varnames = line.split(":", 1)[1].split()
            elif line.startswith("SN:"):
                rows.append(line.split()[1:])
    assert varnames is not None and "CUTMASK" in varnames
    idx = varnames.index("CUTMASK")
    return np.array([int(float(r[idx])) for r in rows])


def test_cutwin_perbit_aggregate_rates(snana_fixture):
    """JAX reproduces SNANA's per-cut pass count for every active CUTBIT."""
    fx = snana_fixture("cutwin_parity")
    snana_mask = _parse_dump_cutmask(fx.dump)
    n_total = len(snana_mask)

    config = dict(
        n_sne=n_total, random_seed=122459,
    )
    sim = sim_run.prepare(str(fx.input_file), **config)
    results, sel, _ = sim_run.run_simulation(sim)
    jax_mask = np.asarray(sel["cutmask"]).astype(int)

    active_bits = {
        "TRESTMIN":  cutwin.CUTBIT_TRESTMIN,
        "TRESTMAX":  cutwin.CUTBIT_TRESTMAX,
        "SNRMAX":    cutwin.CUTBIT_SNRMAX,
        "NOBS_SNR":  cutwin.CUTBIT_NOBS_SNR,
    }

    failures = []
    for name, bit in active_bits.items():
        s_pass = int(((snana_mask >> bit) & 1).sum())
        j_pass = int(((jax_mask  >> bit) & 1).sum())
        diff = abs(s_pass - j_pass) / n_total
        # TRESTMIN/MAX classify SNe at trest boundaries; small differences
        # in peak_mjd treatment between SNANA and JAX flip ~1% of borderline
        # SNe. SNRMAX adds further per-obs noise variance.
        tol = 0.03 if name == "SNRMAX" else 0.02
        if diff > tol:
            failures.append(
                f"{name}: SNANA={s_pass}/{n_total}, JAX={j_pass}/{n_total} "
                f"(|Δ|={diff:.3f} > {tol:.2f})"
            )

    assert not failures, "Per-bit CUTWIN divergence:\n  " + "\n  ".join(failures)


# =============================================================================
# Pure-JAX unit tests for the CUTWIN kernel's SNANA epoch semantics
# =============================================================================


def _run_kernel(cuts, snrs, mjds, epoch_mask, filter_names=("LSST-g", "LSST-r")):
    import jax.numpy as jnp
    from jax_snlc_sim.cutwin import cutwin_all, prepare_cutwin

    arrays, options = prepare_cutwin(cuts, list(filter_names), byfield_masks=())
    snrs = jnp.asarray(snrs, dtype=jnp.float32)
    n_sne, max_obs = snrs.shape
    zeros = jnp.zeros_like(snrs)
    _, fails, _ = cutwin_all(
        arrays, options,
        snrs, zeros, jnp.zeros_like(snrs, dtype=jnp.int32),
        jnp.ones_like(snrs, dtype=bool), jnp.asarray(mjds, dtype=jnp.float32),
        jnp.asarray(epoch_mask, dtype=bool),
        jnp.zeros((n_sne, len(filter_names)), dtype=jnp.float32),
        jnp.zeros(n_sne, dtype=jnp.int32),
        jnp.zeros_like(snrs, dtype=bool))
    return {k: np.asarray(v) for k, v in fails.items()}


def test_cutwin_epochs_selects_fast_transients():
    """CUTWIN_EPOCHS passes when TIME_ABOVE_SNRMIN < TRANGE[1]
    (snlc_sim.c:24295-24297), with strict SNR > (:24277) and pass when no
    epoch qualifies (TIME_ABOVE=0)."""
    from dataclasses import replace
    from jax_snlc_sim.simulation import Cuts

    cuts = replace(Cuts(), epochs_time_cut={
        'snr_min': 5.0, 'time_above_max': 10.0, 'filters': 'gr'})
    mjds = np.array([[100.0, 104.0, 115.0]])
    ones = np.ones((1, 3), dtype=bool)

    # Span 15 d above SNRMIN: a slow transient, FAILS the fast cut.
    fails = _run_kernel(cuts, np.full((1, 3), 6.0), mjds, ones)
    assert fails['TIME_ABOVE'][0]
    # Span 4 d (last epoch at exactly SNR=5 is excluded by strict >).
    fails = _run_kernel(cuts, np.array([[6.0, 6.0, 5.0]]), mjds, ones)
    assert not fails['TIME_ABOVE'][0]
    # No qualifying epoch: TIME_ABOVE=0 < 10 passes.
    fails = _run_kernel(cuts, np.full((1, 3), 5.0), mjds, ones)
    assert not fails['TIME_ABOVE'][0]


def test_nobs_mjddif_counts_like_snana():
    """MJD_last starts at -99999 so the first accepted epoch counts, and
    diffs skip rejected epochs (snlc_sim.c:24194, 24272-24273)."""
    from dataclasses import replace
    from jax_snlc_sim.simulation import Cuts

    cuts = replace(Cuts(), nobsdiff_min=3, mjddif_min=2.0)
    mjds = np.array([[100.0, 100.5, 103.0, 104.0, 107.0]])
    snrs = np.zeros((1, 5))

    # All accepted: first epoch + 103 (2.5 d) + 107 (3 d) = 3 counts.
    fails = _run_kernel(cuts, snrs, mjds, np.ones((1, 5), dtype=bool))
    assert not fails['NOBS_MJDDIF'][0]
    # Reject 103: the 104 epoch now diffs 3.5 d against 100.5 = 3 counts.
    fails = _run_kernel(
        cuts, snrs, mjds, np.array([[1, 1, 0, 1, 1]], dtype=bool))
    assert not fails['NOBS_MJDDIF'][0]
    # Reject 103 and 107: only 2 counts remain.
    fails = _run_kernel(
        cuts, snrs, mjds, np.array([[1, 1, 0, 1, 0]], dtype=bool))
    assert fails['NOBS_MJDDIF'][0]


def test_nobs_snr_counts_filtered_epochs_inclusive():
    """The CUTWIN_NEPOCH SNR count uses SNR >= threshold over the filtered
    epoch set (snlc_sim.c:24251): an epoch at exactly the threshold counts,
    and epochs outside the mask never do."""
    from dataclasses import replace
    from jax_snlc_sim.simulation import Cuts

    cuts = replace(Cuts(), nepoch_min=2, nepoch_snr_threshold=5.0)
    mjds = np.array([[100.0, 101.0, 102.0]])
    snrs = np.array([[5.0, 5.0, 4.9]])

    fails = _run_kernel(cuts, snrs, mjds, np.ones((1, 3), dtype=bool))
    assert not fails['NOBS_SNR'][0]
    # Masking one boundary epoch (e.g. EPCUTWIN_LAMREST failure) drops the
    # count below the minimum.
    fails = _run_kernel(
        cuts, snrs, mjds, np.array([[1, 0, 1]], dtype=bool))
    assert fails['NOBS_SNR'][0]
