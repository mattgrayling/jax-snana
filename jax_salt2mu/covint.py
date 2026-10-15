"""biasCor intrinsic-scatter matrix COVINT (SALT2mu.c init_sigInt_biasCor_SNRCUT,
init_COVINT_biasCor, sntools.c sigint_muresid_list).

For the default ``opt_biascor`` (empirical COVINT bit 256 off) the intrinsic
matrix is diagonal with a single element, ``COVINT[0][0] = sigInt(node)**2``.
``sigInt`` per coefficient node is the scatter that brings the SNR>snrmin
biasCor subset's Hubble-residual pull to unit RMS.
"""

import numpy as np

from .map_kernels import in_map_precision

from jax_snlc_core.utils.error_handling import SNANAError

from .tablevar import CUTBIT_zBIASCOR

MINEVT_SIGINT_COMPUTE = 50   # fewest high-SNR events a node may have (SALT2mu.c:14220)


def _check_scatter(diagnostics, abort):
    invalid, narrow, floor, unbracketed = (np.asarray(x) for x in diagnostics)
    for condition, message in (
            (invalid, "Invalid MUCOV in intrinsic-scatter sample"),
            (narrow & abort, "Cannot compute sigint because STD < AVG_MUERR"),
            (floor & abort, "Cannot compute sigint because sig trial < -0.300000"),
            (unbracketed, "ONE NOT CONTAINED by rmsPull array")):
        if np.any(condition):
            raise SNANAError(message, function_name="sigint_muresid_list")


@in_map_precision
def sigint_muresid_list(mures, mucov, wgt, abort=True, use_mad=False) -> float:
    """Intrinsic scatter on the reference trial grid, evaluated by JAX.

    ``abort=False`` permits negative intrinsic scatter down to -0.3.
    MAD uses an unweighted median and a population-weighted residual centre.
    """
    from .scatter import estimate
    n = len(mures)
    if not n:
        raise SNANAError("Empty intrinsic-scatter sample", function_name="sigint_muresid_list")
    result, diagnostics, _ = estimate(
        np.asarray(mures), np.asarray(mucov), np.asarray(wgt),
        np.zeros(n, dtype=np.int64), np.ones(n, dtype=bool),
        ncell=1, use_mad=use_mad)
    _check_scatter(diagnostics, abort)
    return float(result[0])


@in_map_precision
def sigint_coeff_grid(biascor_tv, events, coeff_bins, config,
                      event_mask=None) -> np.ndarray:
    """sigInt per coefficient node from the SNR>snrmin subset, flat ``[N_node]``. A
    fixed ``sigint_biascor`` (>=0) overrides the computation. ``event_mask`` narrows
    the biasCor events considered, e.g. to the selected IDSAMPLEs."""
    n_node = int(np.prod([b.nbin for b in coeff_bins])) if coeff_bins else 1
    if config.sigint_biascor >= 0.0:
        return np.full(n_node, config.sigint_biascor)

    tv = biascor_tv
    # selection without the biasCor z cut, keeping the global z cut, so
    # high-z-only fits still find SNR>snrmin events (SALT2mu.c:14326-14335)
    passing = (tv.cutbits & ~(1 << CUTBIT_zBIASCOR)) == 0
    # SNR membership depends on the correction catalogue and configuration,
    # not the data realisation. Project this fixed population once; keep cuts
    # as a mask so the iterative kernel does not scan millions of low-SNR rows.
    rows = np.flatnonzero(tv.snrmax > config.snrmin_sigint_biascor)
    sel = passing[rows]
    if event_mask is not None:
        sel = sel & event_mask[rows]
    if not len(rows):
        raise SNANAError("No biasCor events above snrmin_sigint_biascor; "
                         f"sigInt needs at least {MINEVT_SIGINT_COMPUTE}",
                         function_name="sigint_coeff_grid")

    from .scatter import estimate
    result, diagnostics, counts = estimate(
        events.mures[rows], events.mucov_fit[rows], tv.sim_truth["SIM_WGT_POP"][rows],
        events.node[rows], sel, ncell=n_node, minimum=MINEVT_SIGINT_COMPUTE)
    counts = np.asarray(counts)
    for i in np.flatnonzero(counts < MINEVT_SIGINT_COMPUTE):
        raise SNANAError(
            f"coefficient node {i} has {counts[i]} biasCor events above "
            f"snrmin_sigint_biascor={config.snrmin_sigint_biascor}; "
            f"sigInt needs at least {MINEVT_SIGINT_COMPUTE}",
            function_name="sigint_coeff_grid")
    _check_scatter(diagnostics, True)
    return np.array(result)
