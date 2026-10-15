"""Apply the biasCor map to data events (SALT2mu storeBias_data:14150,
get_muBias:11778).

``storeBias_data`` precomputes each data event's fit-parameter bias at every
coefficient node and rejects the event unless all nodes are valid. The
interpolation over the coefficients happens in the likelihood, at the values the
fit is currently at (``mu_and_parts``).
"""

from itertools import product

import numpy as np

from .biascor_map import fitparbias_events
from .mucovscale import mucovscale_events


def _coeff_nodes(bcmap):
    return list(product(*[range(bcmap.bininfo[j].nbin) for j in range(bcmap.n_coeff)]))


def precompute_fitparbias_nodes(data_tv, maps, model, min_per_cell: int,
                                idsample=None):
    """Per event, the LC-parameter bias at every coefficient node and an all-nodes
    valid mask (storeBias_data), ``[n, N_node, k]``. ``idsample`` overrides
    ``data_tv.idsample`` (e.g. for CC-prior events)."""
    m0 = maps.maps[0]
    nodes = _coeff_nodes(m0)
    N, k = len(nodes), m0.fitparbias.val.shape[1]
    n = data_tv.size
    if idsample is None:
        idsample = data_tv.idsample
    z, mass = data_tv.zhd, data_tv.logmass

    val = np.zeros((n, N, k))
    err = np.zeros((n, N, k))
    nevt = np.zeros((n, N), dtype=np.int64)
    ok = np.zeros(n, dtype=bool)
    for isamp in np.unique(idsample):
        if isamp < 0:                    # excluded from every IDSAMPLE
            continue
        sel = np.nonzero(idsample == isamp)[0]
        bcmap = maps.maps[isamp]
        allok = np.ones(sel.shape, dtype=bool)
        for inode, node in enumerate(nodes):
            v, e, _, ns, code = fitparbias_events(
                bcmap, node, z[sel], mass[sel],
                {name: data_tv.lcpar[name][sel] for name in bcmap.axis_names}, min_per_cell)
            good = code == 1
            rows = sel[good]
            val[rows, inode] = v[good]
            err[rows, inode] = e[good]
            nevt[rows, inode] = ns[good]
            allok &= good
        ok[sel] = allok
    return val, nevt, err, ok


def precompute_mucovscale(data_tv, maps, model):
    """Per data event, muCOVscale (and muCOVADD, if built) at every coefficient node
    (get_muCOVcorr) and an all-nodes valid mask, ``[n, N_node]``."""
    m0 = maps.mucov_maps[0]
    nodes = _coeff_nodes(m0)
    N, n = len(nodes), data_tv.size
    z, mass = data_tv.zhd, data_tv.logmass
    has_covadd = m0.covadd is not None
    val = np.ones((n, N))
    cadd = np.zeros((n, N)) if has_covadd else None
    ok = np.zeros(n, dtype=bool)
    for isamp in np.unique(data_tv.idsample):
        if isamp < 0:                    # excluded from every IDSAMPLE
            continue
        sel = np.nonzero(data_tv.idsample == isamp)[0]
        mc = maps.mucov_maps[isamp]
        allok = np.ones(sel.shape, dtype=bool)
        for inode, node in enumerate(nodes):
            scale, ca, good = mucovscale_events(
                mc, node, z[sel], mass[sel],
                {name: data_tv.lcpar[name][sel] for name in mc.axis_names})
            rows = sel[good]
            val[rows, inode] = scale[good]
            if has_covadd:
                cadd[rows, inode] = ca[good]
            allok &= good
        ok[sel] = allok
    return val, ok, cadd


def precompute_biascor(data_tv, maps, model, min_per_cell):
    """All cell-map corrections before data-specific exemptions and cuts."""
    val, nevt, err, ok = precompute_fitparbias_nodes(data_tv, maps, model, min_per_cell)
    if maps.mucov_maps[0] is None:
        scale, added = np.ones(val.shape[:2]), None
    else:
        scale, cov_ok, added = precompute_mucovscale(data_tv, maps, model)
        ok = ok & cov_ok
    return val, nevt, err, scale, added, ok
