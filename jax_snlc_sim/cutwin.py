"""
JIT-compiled CUTWIN per-observation selection cuts.

Every per-observation CUTWIN lives in the one vmapped kernel: NEPOCH,
SNRMAX, TRESTMIN, TRESTMAX, NOBSDIFF, EPOCHS_TIME, TGAPMAX, T0GAPMAX,
PEAKMAG and NOBS_[NO]SATURATE. Per-SN scalar range cuts (MWEBV, redshift,
the open CUTWIN_<VAR> family) are handled by apply_generic_range_cuts.
"""

from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core.filters import filter_band
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.utils import apply_range_cuts, in_window
from jax_snlc_sim.simulation import Cuts


# Bit positions, SNANA snlc_sim.h:1786-1797.
CUTBIT_TRESTMAX    = 0   # (1)    max Trest coverage
CUTBIT_TRESTMIN    = 1   # (2)    min Trest coverage (pre-max)
CUTBIT_SNRMAX      = 2   # (4)    max SNR per filter requirement
CUTBIT_TGAPMAX     = 3   # (8)    max time gap
CUTBIT_T0GAPMAX    = 4   # (16)   max time gap near peak
CUTBIT_NOBS_SNR    = 5   # (32)   NOBS with SNR threshold (NEPOCH)
CUTBIT_NOBS_MJDDIF = 6   # (64)   NOBS with MJD difference
CUTBIT_MWEBV       = 7   # (128)  Milky Way extinction
CUTBIT_REDSHIFT    = 8   # (256)  redshift range
CUTBIT_PEAKMAG     = 9   # (512)  peak magnitude
CUTBIT_TIME_ABOVE  = 10  # (1024) time above SNR threshold
CUTBIT_SATURATE    = 11  # (2048) saturation cuts

NUM_CUTBITS = 12

CUTMASK_ALL_PASS = (1 << NUM_CUTBITS) - 1

CUTBIT_MAP = {
    'TRESTMAX': CUTBIT_TRESTMAX,
    'TRESTMIN': CUTBIT_TRESTMIN,
    'SNRMAX': CUTBIT_SNRMAX,
    'TGAPMAX': CUTBIT_TGAPMAX,
    'T0GAPMAX': CUTBIT_T0GAPMAX,
    'NOBS_SNR': CUTBIT_NOBS_SNR,
    'NOBS_MJDDIF': CUTBIT_NOBS_MJDDIF,
    'MWEBV': CUTBIT_MWEBV,
    # SNANA shares CUTBIT_REDSHIFT across the redshift triad (snlc_sim.c:24182)
    'REDSHIFT': CUTBIT_REDSHIFT,
    'REDSHIFT_TRUE': CUTBIT_REDSHIFT,
    'REDSHIFT_FINAL': CUTBIT_REDSHIFT,
    'HOST_ZPHOT': CUTBIT_REDSHIFT,
    'PEAKMAG': CUTBIT_PEAKMAG,
    'TIME_ABOVE': CUTBIT_TIME_ABOVE,
    'SATURATE': CUTBIT_SATURATE,
}


# TGAPMAX/T0GAPMAX before any gap is measured (snlc_sim.c:9563).
GAP_UNSET = -99999.0


def prepare_cutwin(cutwin: Cuts, filter_names: list, byfield_masks):
    """Prepare numerical cut thresholds and structural options for ``cutwin_all``.

    ``filter_names`` gives the canonical filter order. ``byfield_masks``
    holds a per-SIMLIB-entry mask for each CUTWIN_PEAKMAG_BYFIELD row.
    The returned arrays remain runtime inputs to the shared compiled kernel.
    """
    nepoch_min = cutwin.nepoch_min
    nepoch_snr_threshold = cutwin.nepoch_snr_threshold

    trestmin_lo, trestmin_hi = cutwin.trestmin_range
    trestmax_lo, trestmax_hi = cutwin.trestmax_range

    nobsdiff_min = cutwin.nobsdiff_min
    mjddif_threshold = cutwin.mjddif_min

    char_to_idx = {}
    for idx, fname in enumerate(filter_names):
        char = filter_band(fname)
        char_to_idx[char] = idx

    epochs_snr_min = 0.0
    epochs_time_above_max = 0.0
    epochs_filter_mask = None
    epochs_cfg = cutwin.epochs_time_cut
    has_epochs_time = epochs_cfg is not None
    if has_epochs_time:
        epochs_snr_min = float(epochs_cfg['snr_min'])
        epochs_time_above_max = float(epochs_cfg['time_above_max'])
        epochs_filters_str = epochs_cfg['filters'].strip()
        epochs_filter_mask = np.ones(len(filter_names), dtype=bool)
        if epochs_filters_str:
            epochs_filter_mask = np.zeros(len(filter_names), dtype=bool)
            for char in epochs_filters_str:
                if char in char_to_idx:
                    epochs_filter_mask[char_to_idx[char]] = True
        epochs_filter_mask = jnp.array(epochs_filter_mask)

    # CUTWIN_SNRMAX entries, AND-combined (snlc_sim.c:24339-24369)
    snrmax_cuts = []
    for snrmax, filters, nfilt, tlo, thi in cutwin.snrmax_cuts:
        unknown = [c for c in filters if c not in char_to_idx]
        if unknown:
            raise SNANAError(
                f"Invalid filter {''.join(unknown)!r} in CUTWIN_SNRMAX: "
                f"{snrmax} {filters}; the survey defines "
                f"{''.join(sorted(char_to_idx))}.",
                function_name="prepare_cutwin")
        idx = [char_to_idx[c] for c in filters]
        snrmax_cuts.append((float(snrmax),
                            jnp.array(idx, dtype=jnp.int32),
                            int(nfilt), float(tlo), float(thi)))

    # Gap endpoint bounds and the near-peak window (snlc_sim.c:24394-24408).
    gap_lo, gap_hi = trestmin_lo, trestmax_hi
    t0_lo, t0_hi = trestmin_hi, trestmax_lo
    peakmag_lo, peakmag_hi = cutwin.peakmag_range or (-30.0, 999.0)
    peakmag_all_lo, peakmag_all_hi = cutwin.peakmag_all_range or (-30.0, 999.0)
    byfield = [(lo, hi, mask) for (lo, hi, _), mask
               in zip(cutwin.peakmag_byfield, byfield_masks)]
    has_peakmag_cuts = bool(cutwin.peakmag_range or cutwin.peakmag_all_range
                            or byfield)
    tgapmax_range = cutwin.tgapmax_range
    t0gapmax_range = cutwin.t0gapmax_range
    has_gap_cuts = tgapmax_range is not None or t0gapmax_range is not None
    tgap_lo, tgap_hi = tgapmax_range or (-999999.0, 999999.0)
    t0gap_lo, t0gap_hi = t0gapmax_range or (-999999.0, 999999.0)
    sat_prep = _prepare_saturate_ranges(cutwin.saturate_ranges, filter_names)
    nosat_prep = _prepare_saturate_ranges(cutwin.nosaturate_ranges, filter_names)
    has_saturate_cuts = sat_prep is not None or nosat_prep is not None
    n_filters = len(filter_names)

    use_nepoch = nepoch_min > 0
    use_premax = trestmin_lo > -900
    use_postmax = trestmax_lo > -900
    use_nobsdiff = nobsdiff_min > 0
    arrays = {
        "nepoch_min": nepoch_min,
        "nepoch_snr_threshold": nepoch_snr_threshold,
        "trestmin_lo": trestmin_lo,
        "trestmin_hi": trestmin_hi,
        "trestmax_lo": trestmax_lo,
        "trestmax_hi": trestmax_hi,
        "nobsdiff_min": nobsdiff_min,
        "mjddif_threshold": mjddif_threshold,
        "epochs_snr_min": epochs_snr_min,
        "epochs_time_above_max": epochs_time_above_max,
        "epochs_filter_mask": epochs_filter_mask,
        "snrmax_cuts": snrmax_cuts,
        "gap_lo": gap_lo,
        "gap_hi": gap_hi,
        "t0_lo": t0_lo,
        "t0_hi": t0_hi,
        "peakmag_lo": peakmag_lo,
        "peakmag_hi": peakmag_hi,
        "peakmag_all_lo": peakmag_all_lo,
        "peakmag_all_hi": peakmag_all_hi,
        "byfield": byfield,
        "tgap_lo": tgap_lo,
        "tgap_hi": tgap_hi,
        "t0gap_lo": t0gap_lo,
        "t0gap_hi": t0gap_hi,
        "sat_prep": sat_prep,
        "nosat_prep": nosat_prep,
    }
    options = (
        ("has_epochs_time", has_epochs_time),
        ("has_peakmag_cuts", has_peakmag_cuts),
        ("has_gap_cuts", has_gap_cuts),
        ("has_saturate_cuts", has_saturate_cuts),
        ("n_filters", n_filters),
        ("use_nepoch", use_nepoch),
        ("use_premax", use_premax),
        ("use_postmax", use_postmax),
        ("use_nobsdiff", use_nobsdiff),
    )
    return arrays, options


def cutwin_one_sn(arrays, options, snrs, phases, filter_indices, valid, mjds, epoch_mask,
                  peakmags, field_idx, saturated):
    options = dict(options)
    # counters range over the filtered epochs (snlc_sim.c:24203-24235);
    # SATURATE counts over `valid`, filled outside gen_cutwin's loop
    snr_ok = epoch_mask & (snrs >= arrays["nepoch_snr_threshold"])
    pass_nepoch = (jnp.sum(snr_ok) >= arrays["nepoch_min"]
                   if options["use_nepoch"] else True)

    # strict SNR > snrmax per filter
    pass_snrmax = True
    for snrmax, filt_idx, nfilt, tlo, thi in arrays["snrmax_cuts"]:
        phase_ok = in_window(phases, tlo, thi)
        def max_snr_in_filter(f):
            fmask = epoch_mask & (filter_indices == f) & phase_ok
            return jnp.where(jnp.any(fmask),
                             jnp.max(jnp.where(fmask, snrs, -jnp.inf)),
                             -jnp.inf)
        filter_maxsnr = jax.vmap(max_snr_in_filter)(filt_idx)
        pass_snrmax = pass_snrmax & (
            jnp.sum(filter_maxsnr > snrmax) >= nfilt)

    if options["use_premax"]:
        has_premax = jnp.any(epoch_mask & in_window(
            phases, arrays["trestmin_lo"], arrays["trestmin_hi"]))
    else:
        has_premax = True

    if options["use_postmax"]:
        has_postmax = jnp.any(epoch_mask & in_window(
            phases, arrays["trestmax_lo"], arrays["trestmax_hi"]))
    else:
        has_postmax = True

    if options["use_nobsdiff"]:
        # against the previous accepted epoch (snlc_sim.c:24194, 24272-24273)
        masked_mjds = jnp.where(epoch_mask, mjds, GAP_UNSET)
        prev_accepted = jnp.concatenate([
            jnp.array([GAP_UNSET], dtype=mjds.dtype),
            jax.lax.cummax(masked_mjds)[:-1]])
        gap_counts = jnp.sum(
            epoch_mask & (mjds - prev_accepted >= arrays["mjddif_threshold"]))
        pass_nobsdiff = gap_counts >= arrays["nobsdiff_min"]
    else:
        pass_nobsdiff = True

    if options["has_epochs_time"]:
        # passes when TIME_ABOVE < CUTWIN_EPOCHS_TRANGE[1] (snlc_sim.c:24280-24297)
        filt_ok = arrays["epochs_filter_mask"][filter_indices]
        time_mask = epoch_mask & (snrs > arrays["epochs_snr_min"]) & filt_ok
        has_any = jnp.any(time_mask)
        mjd_min_t = jnp.min(jnp.where(time_mask, mjds, jnp.inf))
        mjd_max_t = jnp.max(jnp.where(time_mask, mjds, -jnp.inf))
        time_above = jnp.where(has_any, mjd_max_t - mjd_min_t, 0.0)
        pass_epochs_time = time_above < arrays["epochs_time_above_max"]
    else:
        pass_epochs_time = True

    # measured with or without a gap cut, for the SIMGEN dump
    sorted_t = jnp.sort(jnp.where(epoch_mask, phases, jnp.inf))
    t_last, t_rest = sorted_t[:-1], sorted_t[1:]
    pair_ok = jnp.isfinite(t_last) & jnp.isfinite(t_rest)
    in_range = ((t_rest >= arrays["gap_lo"]) & (t_rest <= arrays["gap_hi"])
                & (t_last >= arrays["gap_lo"]) & (t_last <= arrays["gap_hi"]))
    pair_ok = pair_ok & in_range
    gaps = t_rest - t_last

    use_t0 = (((t_rest >= arrays["t0_lo"]) & (t_rest <= arrays["t0_hi"]))
              | ((t_last >= arrays["t0_lo"]) & (t_last <= arrays["t0_hi"]))
              | ((t_rest >= arrays["t0_hi"]) & (t_last <= arrays["t0_lo"])))

    tgapmax = jnp.max(jnp.where(pair_ok, gaps, GAP_UNSET))
    t0gapmax = jnp.max(jnp.where(pair_ok & use_t0, gaps, GAP_UNSET))
    if options["has_gap_cuts"]:
        pass_tgapmax = (tgapmax >= arrays["tgap_lo"]) & (tgapmax <= arrays["tgap_hi"])
        pass_t0gapmax = (t0gapmax >= arrays["t0gap_lo"]) & (t0gapmax <= arrays["t0gap_hi"])
    else:
        pass_tgapmax = True
        pass_t0gapmax = True

    if options["has_peakmag_cuts"]:
        in_or = (peakmags >= arrays["peakmag_lo"]) & (peakmags <= arrays["peakmag_hi"])
        for blo, bhi, mask in arrays["byfield"]:
            in_or = in_or & jnp.where(
                jnp.asarray(mask)[field_idx],
                (peakmags >= blo) & (peakmags <= bhi), True)
        in_all = (peakmags >= arrays["peakmag_all_lo"]) & (peakmags <= arrays["peakmag_all_hi"])
        pass_peakmag = jnp.any(in_or) & jnp.all(in_all)
    else:
        pass_peakmag = True

    # CUTWIN_NOBS_[NO]SATURATE (snlc_sim.c:24467-24477)
    if options["has_saturate_cuts"]:
        is_filter = filter_indices[:, None] == jnp.arange(options["n_filters"])[None, :]

        def _saturate_fail(prep, obs_mask, nobs_tot_gate):
            lo, hi, active = prep
            counts = jnp.sum(obs_mask[:, None] & is_filter, axis=0)
            out_of_range = ((counts[None, :] < lo[:, None])
                            | (counts[None, :] > hi[:, None]))
            if nobs_tot_gate:
                # only bands with NOBS_TOT > MINOBS (snlc_sim.c:24491-24506)
                nobs_tot = jnp.sum(valid[:, None] & is_filter, axis=0)
                out_of_range = out_of_range & (
                    nobs_tot[None, :] > lo[:, None])
            return (out_of_range & active).any()

        fail_sat = jnp.bool_(False)
        if arrays["sat_prep"] is not None:
            fail_sat = fail_sat | _saturate_fail(arrays["sat_prep"], saturated, False)
        if arrays["nosat_prep"] is not None:
            fail_sat = fail_sat | _saturate_fail(
                arrays["nosat_prep"], valid & ~saturated, True)
        pass_saturate = ~fail_sat
    else:
        pass_saturate = True

    pass_nepoch = jnp.bool_(pass_nepoch)
    pass_snrmax = jnp.bool_(pass_snrmax)
    has_premax = jnp.bool_(has_premax)
    has_postmax = jnp.bool_(has_postmax)
    pass_nobsdiff = jnp.bool_(pass_nobsdiff)
    pass_epochs_time = jnp.bool_(pass_epochs_time)
    pass_peakmag = jnp.bool_(pass_peakmag)
    pass_tgapmax = jnp.bool_(pass_tgapmax)
    pass_t0gapmax = jnp.bool_(pass_t0gapmax)
    pass_saturate = jnp.bool_(pass_saturate)

    combined = (pass_nepoch & pass_snrmax & has_premax & has_postmax
                & pass_nobsdiff & pass_epochs_time
                & pass_tgapmax & pass_t0gapmax & pass_peakmag
                & pass_saturate)

    # per-event statistics for the dump; TRESTMIN/MAX span every epoch,
    # heliocentric (snlc_sim.c:20766-20771)
    present = filter_indices >= 0
    in_band = epoch_mask[None, :] & (
        filter_indices[None, :] == jnp.arange(options["n_filters"])[:, None])
    stats = {
        'trestmin': jnp.min(jnp.where(present, phases, jnp.inf)),
        'trestmax': jnp.max(jnp.where(present, phases, -jnp.inf)),
        'tgapmax': tgapmax,
        'snrmax_band': jnp.where(
            in_band.any(axis=1),
            jnp.max(jnp.where(in_band, snrs[None, :], -jnp.inf), axis=1),
            -9.0),
        'nobs': jnp.sum(valid),
    }
    return (combined,
            ~pass_nepoch, ~pass_snrmax, ~has_premax, ~has_postmax,
            ~pass_nobsdiff, ~pass_epochs_time,
            ~pass_tgapmax, ~pass_t0gapmax, ~pass_peakmag, ~pass_saturate,
            stats)


@jax.jit(static_argnames=("options",))
def cutwin_all(arrays, options, snrs, phases, filter_indices, valid, mjds, epoch_mask,
               peakmags, field_idx, saturated):
    (combined, fail_nobs, fail_snrmax, fail_trestmin, fail_trestmax,
     fail_nobsdiff, fail_epochs_time, fail_tgapmax, fail_t0gapmax,
     fail_peakmag, fail_saturate, stats) = jax.vmap(cutwin_one_sn, in_axes=(None, None, 0, 0, 0, 0, 0, 0, 0, 0, 0))(
        arrays, options,
        snrs, phases, filter_indices, valid, mjds, epoch_mask,
        peakmags, field_idx, saturated)
    return combined, {
        'NOBS_SNR': fail_nobs,
        'SNRMAX': fail_snrmax,
        'TRESTMIN': fail_trestmin,
        'TRESTMAX': fail_trestmax,
        'NOBS_MJDDIF': fail_nobsdiff,
        'TIME_ABOVE': fail_epochs_time,
        'TGAPMAX': fail_tgapmax,
        'T0GAPMAX': fail_t0gapmax,
        'PEAKMAG': fail_peakmag,
        'SATURATE': fail_saturate,
    }, stats


def per_sn_windows(cutwin: Cuts) -> Dict[str, tuple]:
    """The per-SN scalar cut windows, keyed by parameter name.

    `generic_ranges` is the open CUTWIN_<VAR> family; MWEBV has a named field
    but is cut the same way. The per-observation cuts are separate fields, so
    no name filtering is needed to tell the two apart.
    """
    windows = dict(cutwin.generic_ranges)
    if cutwin.mwebv_range is not None:
        windows['mwebv'] = cutwin.mwebv_range
    return windows


def range_cut_quantities(params: Dict, *, mwebv=None, redshift_true=None,
                         redshift_final=None, host_zphot=None) -> Dict:
    """The per-SN quantities a CUTWIN_<VAR> may name.

    The four keywords have no counterpart in `params`: SNANA's CUTWIN names for
    the redshift triad differ from the sampled ones, and MWEBV comes from the
    observation schedule. Omitting them yields the names alone, which is what
    the pre-generation check needs: it validates spelling, not values.
    """
    quantities = {
        'mwebv': mwebv,
        'redshift_true': redshift_true,
        'redshift_final': redshift_final,
        'host_zphot': host_zphot,
    }
    quantities.update(
        (key, value) for key, value in params.items()
        if hasattr(value, 'shape') and value.ndim == 1)
    return quantities


def check_generic_cut_names(cutwin: Cuts, params: Dict) -> None:
    """Abort on a CUTWIN_<VAR> naming a quantity the sim never computes.

    The cut itself runs during selection, after every event has been generated,
    so a mistyped name would otherwise cost a whole run before reporting.
    """
    known = set(range_cut_quantities(params))
    known |= {n.replace('_', '') for n in known}
    missing = sorted(w for w in per_sn_windows(cutwin) if w not in known)
    if missing:
        raise SNANAError(
            f"CUTWIN {', '.join(missing)} does not match any selection "
            f"quantity; known: {sorted(known)}",
            function_name='check_generic_cut_names')


def apply_generic_range_cuts(cutwin: Cuts, params: Dict) -> jnp.ndarray:
    """Apply per-SN scalar range cuts (MWEBV, redshift, etc.).

    ``cutwin`` is the input's CUTWIN_* section; ``params`` is the dict of
    per-SN parameter arrays ``{name: (n_sne,)}``. Returns the (n_sne,)
    bool mask of SNe passing all range cuts.
    """
    param_map = {}
    for key, value in params.items():
        if hasattr(value, 'shape') and value.ndim == 1:
            param_map[key] = value
            compact = key.replace('_', '')
            if compact != key:
                param_map[compact] = value

    n_sne = next(v.shape[0] for v in params.values() if hasattr(v, 'shape') and v.ndim == 1)
    passing = jnp.ones(n_sne, dtype=bool)

    windows = {k: (v[0], v[1]) for k, v in per_sn_windows(cutwin).items()}
    return passing & jnp.asarray(apply_range_cuts(windows, param_map))


def _prepare_saturate_ranges(ranges, filter_names):
    """Pad a list of (lo, hi, filter_chars) triplets into JAX arrays.

    Returns None when the input is empty. Otherwise returns
    ``(lo, hi, active)``: ``lo``/``hi`` are (n_ranges,) int32; ``active``
    is (n_ranges, n_filters) bool, True at (range, filter) slots named by
    the corresponding triplet's char string.

    Aborts on a char the survey's filter set does not carry: fails loud
    rather than silently dropping a misconfigured filter.
    """
    if not ranges:
        return None
    short_to_idx = {
        filter_band(fn): i
        for i, fn in enumerate(filter_names)
    }
    n_filters = len(filter_names)
    n_r = len(ranges)
    lo = np.empty(n_r, dtype=np.int32)
    hi = np.empty(n_r, dtype=np.int32)
    active = np.zeros((n_r, n_filters), dtype=bool)
    for i, (l, h, chars) in enumerate(ranges):
        lo[i] = l
        hi[i] = h
        for ch in chars:
            if ch not in short_to_idx:
                raise SNANAError(
                    f"CUTWIN_NOBS_[NO]SATURATE filter '{ch}' not in "
                    f"survey filter set {sorted(short_to_idx)}")
            active[i, short_to_idx[ch]] = True
    return jnp.asarray(lo), jnp.asarray(hi), jnp.asarray(active)
