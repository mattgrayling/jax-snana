"""Event and epoch selection: which observations are fitted and which fits
are kept. The loader applies epoch cuts at their required stages before
and after peak-time estimation.

Before the fit: the header, SNTYPE and CID-list selection, the duplicate
checks, the per-event CUTWIN_ windows and the epoch-count cuts, which mark
each event dropped with the reason SNANA would give. After the fit: the
FITWIN windows on the fitted parameters, the error bounds, the rest-frame
and PEAKMJD windows, and the filter and S/N requirements. A photo-z fit
widens the rest-frame windows for the fit and applies them as given after
it (:func:`cut_windows`).

Cuts take the fitter, model and results objects they need as arguments.
"""

import dataclasses
import logging
import os
from dataclasses import replace
from typing import Optional, Tuple

import numpy as np

from jax_snlc_core import logging as log
from jax_snlc_core.filters import filter_band, filter_values
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.noise import fluxcal_errcalc
from jax_snlc_core.snana_files import SNANA_NULLFLOAT, read_fitres
from jax_snlc_core.utils import (CUTVAL_OPEN, apply_range_cuts, in_window, narrow,
                                 range_cut_masks)

from . import parameters, photoz
from .batch import ERRFLAG_ZFIT_RANGE, ITER1_ERRFLAG, LightCurveBatch, trest_z1
from .calibration import FWHM_PER_SIGMA

logger = logging.getLogger(__name__)


# MODEL_ indices SNANA counts as true SNIa (ISMODEL_SNIa, sntools.c:8315):
# MLCS2k2, SNOOPY, SALT2, SALT3/SNEMO (both 9), BayeSN.
SNANA_SNIA_MODEL_INDICES = (3, 4, 6, 9, 13)


MXCID_CHECK = 99999999      # snana.F90:21


def band_maxima(values, band_index, accepted, n_bands):
    """Per-band maxima of ``values`` over the ``accepted`` epochs, -9 for a
    band without one, and whether each band has an epoch."""
    band_max = np.full((values.shape[0], n_bands), -9.0)
    has_epoch = np.zeros((values.shape[0], n_bands), dtype=bool)
    for b in range(n_bands):
        in_band = accepted & (band_index == b)
        band_max[:, b] = np.where(in_band, values, -9.0).max(axis=1)
        has_epoch[:, b] = in_band.any(axis=1)
    return band_max, has_epoch


def gap_maxima(times, accepted, t0_lo, t0_hi):
    """Largest gap between consecutive accepted epochs, and the largest one
    touching the peak window (GET_TRESTVAR, snana.F90:21105-21106, :21167-21181;
    gen_cutwin, snlc_sim.c:24390-24417).

    The window is bounded by where the TRESTMIN and TRESTMAX cuts meet, so
    left open it inverts and every gap qualifies. Both start at -999 and only
    a real gap moves them (snana.F90:21092-21093).
    """
    ordered = np.sort(np.where(accepted, times, np.inf), axis=1)
    with np.errstate(invalid='ignore'):     # inf-inf on padding, masked below
        gaps = np.diff(ordered, axis=1)
    real = np.isfinite(gaps)
    before, after = ordered[:, :-1], ordered[:, 1:]
    touches = (((after > t0_lo) & (after < t0_hi))
               | ((before > t0_lo) & (before < t0_hi))
               | ((after > t0_hi) & (before < t0_lo)))
    # initial: an epoch axis of length one leaves no gaps to reduce over
    return (np.where(real, gaps, -999.0).max(axis=1, initial=-999.0),
            np.where(real & touches, gaps, -999.0).max(axis=1, initial=-999.0))


def band_snr_maxima(batch, mask=None):
    """Accepted-epoch SNR, per-band SNR maxima, and per-band epoch presence."""
    obs_ok = np.asarray(mask if mask is not None else batch.observation_mask)
    flux = np.asarray(batch.fluxcals)
    err = np.asarray(batch.fluxcalerrs)
    snr = np.where(obs_ok & (err > 0), flux / np.maximum(err, 1e-30), SNANA_NULLFLOAT)
    band_max, has_epoch = band_maxima(
        snr, np.asarray(batch.filter_indices), obs_ok, len(batch.filter_names))
    return snr, band_max, has_epoch


def snrmax_band_mask(filter_names, snrmax_filters):
    """Per-band flag for SNANA's LFILTDEF_SNRMAX (snana.F90:8446).

    Blank selects every band. A band the survey does not define aborts, as it
    does upstream. Gates the SNRMAX quantities only: SNANA sets EXIST_FILT
    outside the same test, so band presence itself is never filtered.
    """
    names = [filter_names[b] for b in range(len(filter_names))]
    shorts = [filter_band(name) for name in names]
    if not snrmax_filters:
        return np.ones(len(names), dtype=bool)
    selected = ([snrmax_filters] if isinstance(snrmax_filters, str)
                and snrmax_filters in names else list(snrmax_filters))
    unknown = sorted(set(selected) - set(names) - set(shorts))
    if unknown:
        raise SNANAError(
            f"SNRMAX_FILTERS='{snrmax_filters}' names band(s) "
            f"{''.join(unknown)} not defined by the survey "
            f"({''.join(shorts)})")
    return np.asarray(filter_values(dict.fromkeys(selected, True), names, False))


def apply_snana_selection(
    batch: 'LightCurveBatch',
    head_sntypes: Optional[np.ndarray] = None,
    nepoch_store: Optional[np.ndarray] = None,
    *,
    windows: Optional[dict] = None,
    sntype_list: Optional[list] = None,
    sncid_list: Optional[list] = None,
    snccid_names: Optional[list] = None,
    sncid_ignore: Optional[list] = None,
    sntype_ignore: Optional[list] = None,
    sim_prescale: float = 1.0,
    abort_on_duplcid: bool = True,
    sim_model_index: Optional[np.ndarray] = None,
    usesim_snia: bool = True,
    usesim_nonia: bool = True,
    snrmax_filters: str = '',
    simvar_windows: Optional[dict] = None,
    nobs_trest: Optional[list] = None,
    private_windows: Optional[dict] = None,
    private_veto: Optional[dict] = None,
    require_epochs: Optional[dict] = None,
    ladder: list,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pre-fit SN selection mask, mirroring SNANA's SNLC cut evaluation
    (snana.F90:21430-21460).

    ``windows`` maps cut names to inclusive (lo, hi) ranges, evaluated by
    the shared jax_snlc_core.utils machinery against per-SN derived
    quantities: redshift, mwebv, snrmax, snrmax2 (largest/2nd-largest
    per-band SNR maxima), trestmin/trestmax/tgapmax (rest-frame times from
    the header peak-MJD estimate), and per-band snrmax_<band>, all computed
    from the ACCEPTED epochs (post band/PHOTFLAG/field masks). The nepoch
    quantity is the exception: SNANA's CUTWIN_NEPOCH tests the stored-epoch
    count from BEFORE those cuts (ISNLC_NEPOCH_STORE), so it is supplied by
    the loader via ``nepoch_store`` rather than derived from the batch.
    An unknown window name raises. List cuts accept membership. SNFIELD is
    an epoch-level cut and lives in the loader, not here. ``snrmax_filters``
    restricts which bands the SNRMAX quantities see (snana.F90:17442).

    Returns the boolean [n_sne] keep mask and the 1-based N_SNLC_PROC index,
    the event's position among those passing the header cuts (zero where
    it fails them).
    Combine the mask with batch.filter_by_mask. ``ladder`` collects the
    surviving count after each cut, in order.
    """
    n = batch.n_sne
    windows = dict(windows or {})
    evaluated = {}
    trest = (None if batch.peakmjd_offsets is None
             else (np.asarray(batch.mjd_offsets)
                   - np.asarray(batch.peakmjd_offsets)[:, None])
             / trest_z1(batch.redshifts)[:, None])

    # RDHEAD_DRIVER's CID/SNTYPE/model-type cuts (snana.F90:4114-4203)
    keep = narrow(np.ones(n, dtype=bool), 'CID/SNTYPE', _header_selection(
        batch, head_sntypes, sntype_list=sntype_list, sncid_list=sncid_list,
        snccid_names=snccid_names,
        sncid_ignore=sncid_ignore, sntype_ignore=sntype_ignore,
        sim_model_index=sim_model_index, usesim_snia=usesim_snia,
        usesim_nonia=usesim_nonia), ladder)
    keep = narrow(keep, 'DUPLICATE_CID',
                  duplicate_cid_mask(batch, abort_on_duplcid), ladder)
    if sim_prescale > 1.000001:
        # REJECT_PRESCALE (snana.F90:4083, :9990)
        keep = narrow(keep, 'SIM_PRESCALE',
                      np.mod(np.arange(n, dtype=float), sim_prescale) == 0.0,
                      ladder)
    # N_SNLC_PROC counts the events that survive to processing
    snana_index = np.where(keep, np.cumsum(keep), 0)

    if windows:
        obs_ok = np.asarray(batch.observation_mask)
        snr, band_max, has_epoch = band_snr_maxima(batch)
        use_band = snrmax_band_mask(batch.filter_names, snrmax_filters)
        band_max = np.where(use_band, band_max, SNANA_NULLFLOAT)
        has_epoch = has_epoch & use_band

        quantities = {
            'redshift': np.asarray(batch.redshifts),
            'redshift_err': np.asarray(batch.redshift_errs),
            'mwebv': np.asarray(batch.mwebvs),
        }
        # CUTWIN_PEAKMJD cuts on the header search PEAKMJD (snana.F90:21274)
        extras = batch.head_extras
        if 'PEAKMJD' in extras:
            quantities['peakmjd'] = np.asarray(extras['PEAKMJD'])
        if 'nepoch' in windows:
            if nepoch_store is None:
                raise SNANAError(
                    "CUTWIN_NEPOCH configured but the batch carries no "
                    "nepoch_store, so the pre-cut epoch count is unknown",
                    function_name="apply_snana_selection")
            quantities['nepoch'] = np.asarray(nepoch_store)
        host = batch.host_data or {}
        for b, short in enumerate(batch.band_names()):
            quantities[f'snrmax_{short.lower()}'] = band_max[:, b]
            sb = f'HOSTGAL_SB_FLUXCAL_{short}'
            if sb in host:
                quantities[f'sbflux_{short.lower()}'] = np.asarray(
                    host[sb], dtype=float)
        ranked = np.sort(band_max, axis=1)[:, ::-1]
        quantities['snrmax'] = ranked[:, 0]
        quantities['snrmax2'] = (ranked[:, 1] if ranked.shape[1] > 1
                                 else np.full(batch.n_sne, SNANA_NULLFLOAT))
        # NFILT_SNRMAX counts bands whose SNRMAX clears the *lower edge* of
        # the matching CUTWIN_SNRMAX window (snana.F90:17458; unset -> -1e12)
        for nf_name, snr_name in (('nfilt_snrmax', 'snrmax'),
                                  ('nfilt_snrmax2', 'snrmax2')):
            if nf_name in windows:
                lo = (windows[snr_name][0] if snr_name in windows
                      else -CUTVAL_OPEN)
                quantities[nf_name] = (
                    (band_max >= lo) & has_epoch).sum(axis=1).astype(float)

        if trest is not None:
            quantities['trestmin'] = _extreme(np.min, trest, obs_ok, np.inf)
            quantities['trestmax'] = _extreme(np.max, trest, obs_ok, -np.inf)
            # SNLC_TrestRange (snana.F90:21343)
            quantities['trestrange'] = quantities['trestmax'] - quantities['trestmin']
            quantities['tgapmax'], quantities['t0gapmax'] = gap_maxima(
                trest, obs_ok,
                windows.get('trestmin', (0.0, CUTVAL_OPEN))[1],
                windows.get('trestmax', (-CUTVAL_OPEN, 0.0))[0])
            # NFTMIN/NFTMAX: bands holding an epoch inside the matching
            # CUTWIN_TREST{MIN,MAX} window (snana.F90:20880, :21388)
            fidx = np.asarray(batch.filter_indices)
            for nf_name, win in (('nfilt_trestmin', 'trestmin'),
                                 ('nfilt_trestmax', 'trestmax')):
                inside = obs_ok & in_window(
                    trest, *windows.get(win, (-CUTVAL_OPEN, CUTVAL_OPEN)))
                quantities[nf_name] = _bands_with_epochs(
                    inside, fidx, len(batch.filter_names)).astype(float)
        elif any(k in windows for k in ('trestmin', 'trestmax', 'tgapmax',
                                        't0gapmax', 'nfilt_trestmin',
                                        'nfilt_trestmax')):
            raise SNANAError(
                "TREST-family CUTWIN configured but the batch has no "
                "peak-MJD estimate (PEAKMJD header column)",
                function_name="apply_snana_selection")

        for name, window_mask in range_cut_masks(windows, quantities):
            keep = narrow(keep, name.upper(), window_mask, ladder)
        evaluated = quantities

    if nobs_trest:
        # every triplet required (sntools.c:1892)
        if batch.peakmjd_offsets is None:
            raise SNANAError(
                "SNCUT_NOBS_TREST needs a peak-MJD estimate (PEAKMJD column)",
                function_name="apply_snana_selection")
        obs_ok = np.asarray(batch.observation_mask)
        in_all = np.ones(n, dtype=bool)
        for nobs, tmin, tmax in nobs_trest:
            # endpoints included, as sntools.c:1972 tests them
            in_all &= (obs_ok & in_window(trest, tmin, tmax)).sum(axis=1) >= nobs
        keep = narrow(keep, 'NOBS_TREST', in_all, ladder)

    if require_epochs:
        # SNR test spans every band; SNANA never reads IFILTLIST_REQEP (snana.F90:16379)
        t1, t2, t3 = require_epochs['tranges']
        snrmin = require_epochs['snrmin']
        z = np.asarray(batch.redshifts)
        if require_epochs['rest_frame'] and np.any(z < 0.0):
            raise SNANAError(
                f"REQUIRE_EPOCHS in TREST needs a redshift; "
                f"{int(np.sum(z < 0.0))} events have none",
                function_name="ISTAT_REQUIRE_EPOCHS")
        z1 = 1.0 + z if require_epochs['rest_frame'] else np.ones(batch.n_sne)
        obs_ok = np.asarray(batch.observation_mask)
        err = np.asarray(batch.fluxcalerrs)
        snr = np.where(obs_ok & (err > 1e-5),
                       np.asarray(batch.fluxcals) / np.where(err > 1e-5, err, 1.0),
                       0.0)
        det = obs_ok & (snr > snrmin)
        mjd = np.asarray(batch.mjd_offsets) + np.asarray(
            batch.reference_mjds)[:, None]
        big = np.where(det, mjd, np.inf)
        small = np.where(det, mjd, -np.inf)
        first, last = big.min(axis=1), small.max(axis=1)
        n_det = det.sum(axis=1)
        ok = n_det > 0
        with np.errstate(invalid='ignore'):
            ok &= ((last - first) / z1) <= t2
        # at least one epoch in the leading / trailing baseline window
        for tgap, lo, hi in ((t1, first - t1 * z1, first - 0.001),
                             (t3, last + 0.001, last + t3 * z1)):
            if tgap >= 1.0e-9:
                ok &= (obs_ok & (mjd > lo[:, None])
                       & (mjd < hi[:, None])).any(axis=1)
        keep = narrow(keep, 'REQUIRE_EPOCHS', ok, ladder)

    if private_windows:
        extras = batch.head_extras
        missing = [n for n in private_windows if n not in extras]
        if missing:
            raise SNANAError(
                f"PRIVATE_CUTWIN_STRING names {', '.join(sorted(missing))}, "
                f"which the HEAD file does not carry (or which "
                f"PRIVATE_VARNAME_READLIST did not request). Available: "
                f"{', '.join(sorted(k for k in extras if k.startswith('PRIVATE')))}")
        keep = narrow(keep, 'PRIVATE', apply_range_cuts(
            private_windows,
            {n: np.asarray(extras[n], dtype=float) for n in private_windows}),
            ladder)

    if private_veto:
        # the 'VARNAME != VALUE' spelling rejects on equality rather than
        # windowing (USE_PRIVATE_CUTWIN = -1, snana.F90:5095)
        extras = batch.head_extras
        vetoed = np.ones(n, dtype=bool)
        for name, value in private_veto.items():
            if name not in extras:
                raise SNANAError(
                    f"PRIVATE_CUTWIN_STRING vetoes on {name}, which the HEAD "
                    f"file does not carry.",
                    function_name="PRIVATE_CUTWIN_STRING")
            vetoed &= np.asarray(extras[name], dtype=float) != value
        keep = narrow(keep, 'PRIVATE_VETO', vetoed, ladder)

    if simvar_windows:
        truth = batch.sim_truth or {}
        missing = [n for n in simvar_windows if n not in truth]
        if missing:
            raise SNANAError(
                f"SIMVAR_CUTWIN_STRING names {', '.join(sorted(missing))}, "
                f"absent from the sim truth. Available: "
                f"{', '.join(sorted(truth))}")
        keep = narrow(keep, 'SIMVAR', apply_range_cuts(
            simvar_windows,
            {n: np.asarray(truth[n], dtype=float) for n in simvar_windows}),
            ladder)

    # Name the cuts rejecting everything, as SNANA does (snana.F90:8775).
    if n and not keep.any():
        for name, (lo, hi) in windows.items():
            if not np.asarray(in_window(evaluated[name], lo, hi)).any():
                logger.warning("every SN fails CUTWIN_%s = %g %g",
                               name.upper(), lo, hi)

    return keep, snana_index


def select_events(batch, snana_selection, ladder, max_process, max_fit):
    """The SNLC cuts and the event caps, narrowing a loaded batch to the events
    that reach the fit.

    Always runs: the duplicate-CID check inside ``apply_snana_selection`` is
    unconditional in SNANA (PARSE_CID), not gated on a cut being configured.
    ``max_process`` bounds N_SNLC_PROC, the events passing the header cuts
    (snana.F90:4955); ``max_fit`` bounds N_SNLC_CUTS, those passing every SNLC
    cut too (:4960). ``ladder`` collects the survivors after each cut.
    """
    extras = batch.head_extras
    warn_duplicate_mjd(batch)
    ladder.append(('READ', batch.n_sne))
    keep, snana_index = apply_snana_selection(
        batch, head_sntypes=extras.get('SNTYPE'),
        nepoch_store=batch.nepoch_store,
        sim_model_index=extras.get('SIM_MODEL_INDEX'),
        ladder=ladder, **snana_selection)
    batch = replace(batch, snana_index=snana_index,
                    n_proc=int(snana_index.max(initial=0)))
    if not keep.all():
        batch = batch.filter_by_mask(np.asarray(keep))
    within = np.asarray(batch.snana_index) <= max_process
    if not within.all():
        batch = batch.filter_by_mask(within)
    ladder.append(('MAX_PROCESS', batch.n_sne))
    if batch.n_sne > max_fit:
        batch = batch.slice(0, max_fit)
        ladder.append(('MAX_FIT', batch.n_sne))
    return batch


def warn_duplicate_mjd(batch) -> int:
    """Count and report epochs repeating the previous MJD in the same band.

    SNANA compares consecutive epochs and warns once per distinct MJD; its
    ABORT_ON_DUPLMJD gate is commented out, so this always runs and never
    rejects (snana.F90:22414-22447).
    """
    mjd = np.asarray(batch.mjd_offsets)
    fidx = np.asarray(batch.filter_indices)
    ok = np.asarray(batch.observation_mask)
    repeat = ((mjd[:, 1:] == mjd[:, :-1]) & (fidx[:, 1:] == fidx[:, :-1])
              & ok[:, 1:] & ok[:, :-1])
    n = int(repeat.sum())
    if n:
        sn, ep = np.nonzero(repeat)
        names = batch.filter_names
        for i in range(min(len(sn), 5)):
            mjd_v = float(batch.reference_mjds[sn[i]] + mjd[sn[i], ep[i] + 1])
            band = names[int(fidx[sn[i], ep[i] + 1])]
            log.warn(logger, "DUPLICATE MJD",
                     f"CID={batch.snids[sn[i]]} MJD={mjd_v:.4f} BAND={band}")
        if n > 5:
            log.warn(logger, "DUPLICATE MJD", f"{n - 5} more")
    return n


def _integer_cids(cids):
    """The stripped CIDs, which of them are integers, and their integer
    values (0 for the others)."""
    stripped = np.char.strip(np.asarray(cids).astype(str))
    numeric = np.char.isdigit(stripped)
    return stripped, numeric, np.where(numeric, stripped, '0').astype(np.int64)


def cid_keys(cids) -> np.ndarray:
    """CIDs as the strings a CID list is matched on: stripped, and a numeric
    CID without leading zeros, as SNANA reads it into its integer CID."""
    stripped, numeric, as_int = _integer_cids(cids)
    return np.where(numeric, as_int.astype(str), stripped)


def duplicate_cid_mask(batch, abort) -> np.ndarray:
    """PARSE_CID's duplicate check (snana.F90:14190). ABORT_ON_DUPLCID defaults
    true; false keeps the first occurrence and rejects the rest."""
    # only integer CIDs below MXCID_CHECK are checked (snana.F90:14144, :14179)
    _, numeric, as_int = _integer_cids(batch.snids)
    checkable = numeric & (as_int < MXCID_CHECK)
    uniq, first, counts = np.unique(as_int[checkable], return_index=True,
                                    return_counts=True)
    if not (counts > 1).any():
        return np.ones(batch.n_sne, dtype=bool)
    if abort:
        dup = sorted(str(c) for c in uniq[counts > 1])
        raise SNANAError(
            f"CID {dup[0]} appears more than once; each CID can appear only "
            f"once. {len(dup)} duplicated: {dup[:10]}",
            function_name="PARSE_CID")
    keep = ~checkable
    keep[np.flatnonzero(checkable)[first]] = True
    log.warn(logger, "DUPLICATE", f"rejected {int((~keep).sum())} repeat CID(s)")
    return keep


def _header_selection(batch, head_sntypes, *, sntype_list, sncid_list,
                      snccid_names, sncid_ignore, sntype_ignore,
                      sim_model_index, usesim_snia, usesim_nonia) -> np.ndarray:
    """The cuts RDHEAD_DRIVER applies before the event is counted."""
    keep = np.ones(batch.n_sne, dtype=bool)
    if sntype_list or sntype_ignore:
        if head_sntypes is None:
            raise SNANAError("SNTYPE_LIST/SNTYPE_IGNORE configured but no "
                             "SNTYPE column",
                             function_name="_header_selection")
        # an unset SNTYPE reads as 0 before the cuts see it (snana.F90:3901)
        sntypes = np.asarray(head_sntypes, dtype=int)
        sntypes = np.where(sntypes == -9, 0, sntypes)
        if sntype_list:
            keep &= np.isin(sntypes, [int(t) for t in sntype_list])
        if sntype_ignore:
            keep &= ~np.isin(sntypes, [int(t) for t in sntype_ignore])
    if sncid_list or snccid_names:
        listed = np.isin(cid_keys(batch.snids), cid_keys(sncid_list))
        # a name in SNCCID_LIST or the list file selects on the event's
        # transient or IAUC name as well as its SNID (LCIDSELECT,
        # snana.F90:11902)
        extras = batch.head_extras
        for col in ('NAME_TRANSIENT', 'NAME_IAUC'):
            if col in extras:
                listed |= np.isin(np.char.strip(np.asarray(extras[col]).astype(str)),
                                  list(snccid_names or []))
        keep &= listed
    if sncid_ignore:
        keep &= ~np.isin(cid_keys(batch.snids), cid_keys(sncid_ignore))

    if not (usesim_snia and usesim_nonia):
        if sim_model_index is None:
            raise SNANAError(
                "USESIM_SNIA/USESIM_NONIA select on the true model type, but "
                "this data version carries no SIM_MODEL_INDEX column",
                function_name="_header_selection")
        is_snia = np.isin(np.asarray(sim_model_index, dtype=int),
                          list(SNANA_SNIA_MODEL_INDICES))
        if not usesim_snia:
            keep &= ~is_snia
        if not usesim_nonia:
            keep &= is_snia
    return keep


def _read_cid_file(path):
    """CIDs from a SNCID list file, plus its table when it is in FITRES format.

    match_cidlist_init (sntools.c:259) takes either format; only the table one
    carries the fit parameters OPT_SNCID_LIST bit 1 seeds from.
    """
    path = os.path.expandvars(path)
    with open(path) as fh:
        text = fh.read()
    if 'VARNAMES:' not in text:
        return [tok for line in text.splitlines()
                for tok in line.split("#")[0].split()], None
    table = read_fitres(path)
    return [str(c) for c in table.df['CID']], table


def _cid_list(numeric, strings, path):
    """One CID list from its numeric, string and file spellings, the names
    among the strings and the file, and the list-file table.

    SNANA keeps SNCID_ and SNCCID_ apart because Fortran cannot hold both in one
    array; here every CID is a string, matched by :func:`cid_keys`, so they
    merge. The string list ends at its first blank (snana.F90:11897). Every
    string and every file entry is also a transient or IAUC name to match, a
    numeric one included, since a Rubin object id is a number (LCIDSELECT,
    snana.F90:11902).
    """
    cids = [str(c) for c in (numeric or [])]
    blank = next((i for i, c in enumerate(strings or [])
                  if not str(c).strip()), None)
    names = [str(c).strip() for c in (strings or [])[:blank]]
    table = None
    if path:
        from_file, table = _read_cid_file(path)
        names += from_file
    return cids + names, names, table


def drop_unfittable(batch, nepoch_window, *, abort_on_noepochs,
                    quantile_prior, abort_on_missing_qzphot):
    """Events SNANA skips at fit prep rather than at the cuts: nothing to fit
    (NFITDATA <= 0, snlc_fit.F90:10610), fewer epochs than CUTWIN_NEPOCH's
    lower bound (:10630), or a host z-PDF the spline cannot take
    (:8540-8543). They passed the SNLC cuts, so they keep
    CUTFLAG_SNANA = 1. An empty event aborts under ABORT_ON_NOEPOCHS, and
    one missing its z-PDF quantiles under ABORT_ON_MISSING_QZPHOT.

    Returns (fittable batch, empty SNIDs, too-thin SNIDs, bad-quantile
    SNIDs); each carries its own ERRFLAG_FIT code."""
    nobs = np.asarray(batch.n_obs_per_sn)
    snids = np.asarray(batch.snids)
    empty = nobs <= 0
    if abort_on_noepochs and empty.any():
        raise SNANAError(
            f"Nothing to fit for CID {snids[empty][0]}; to avoid abort, set "
            f"ABORT_ON_NOEPOCHS = F.", function_name="FITINI_COV")
    floor = max(1, int(nepoch_window[0])) if nepoch_window else 1
    epoch_ok = nobs >= floor
    if not epoch_ok.all():
        log.line(logger, f"Skipped {int((~epoch_ok).sum())} SNe with fewer than "
                 f"{floor} epochs to fit")
    badq = np.zeros(batch.n_sne, dtype=bool)
    needs_quantiles = epoch_ok & np.asarray(quantile_prior, dtype=bool)
    if needs_quantiles.any():
        # FITINI_PHOTOZ, snlc_fit.F90:8468-8478, :8540-8543
        missing = (needs_quantiles if batch.zpdf_quantiles is None else
                   needs_quantiles & (np.asarray(batch.zpdf_nbin) == 0))
        if missing.any() and abort_on_missing_qzphot:
            raise SNANAError(
                f"Missing zPDF quantiles for photo-z fit to CID="
                f"{snids[missing][0]}. To skip such events, set &SNLCINP "
                f"ABORT_ON_MISSING_QZPHOT=F", function_name="FITINI_PHOTOZ")
        badq = (missing if batch.zpdf_quantiles is None else
                needs_quantiles & ~photoz.zpdf_rows_usable(
                    batch.zpdf_quantiles, batch.zpdf_nbin))
        if badq.any():
            log.line(logger, f"Skipped {int(badq.sum())} SNe whose host z-PDF "
                     f"quantiles are missing or do not increase")
    fittable = epoch_ok & ~badq
    return (batch.filter_by_mask(fittable) if not fittable.all() else batch,
            snids[empty], snids[~epoch_ok & ~empty], snids[badq])


def sncid_list(config):
    """SNCID_LIST, SNCCID_LIST and SNCID_LIST_FILE as one CID list, with
    the names and the list file's table (:func:`_cid_list`)."""
    return _cid_list(config.selection.sncid_list,
                     config.selection.snccid_list,
                     config.selection.sncid_list_file)


def build_selection(config, cutwin, listed):
    """The per-SN selection for the loader; ``listed`` is the config's
    :func:`sncid_list`."""
    selection = {"abort_on_duplcid": config.selection.abort_on_duplcid}
    cids, names, _ = listed
    prescale = config.selection.sim_prescale
    if cids or names:
        # any CID list zeroes the prescale (snana.F90:11872)
        prescale = 1.0
        selection["sncid_list"] = cids
        selection["snccid_names"] = names
    ignored, _, _ = _cid_list(config.selection.sncid_ignore,
                              config.selection.snccid_ignore,
                              config.selection.sncid_ignore_file)
    if ignored:
        selection["sncid_ignore"] = ignored
    if config.selection.sntype_ignore:
        selection["sntype_ignore"] = list(config.selection.sntype_ignore)
    # OPT_SNCID_LIST bit 0: CID cuts only, ignore lists still apply (snana.F90:15444)
    if (cids or names) and config.selection.opt_sncid_list & 1:
        return selection
    if cutwin:
        selection["windows"] = cutwin
    if config.selection.sntype_list:
        selection["sntype_list"] = list(config.selection.sntype_list)
    if config.selection.require_epochs:
        selection["require_epochs"] = dict(config.selection.require_epochs)
    if config.selection.private_cutwin:
        selection["private_windows"] = dict(config.selection.private_cutwin)
    if config.selection.private_veto:
        selection["private_veto"] = dict(config.selection.private_veto)
    if config.selection.sncut_nobs_trest:
        selection["nobs_trest"] = list(config.selection.sncut_nobs_trest)
    if config.selection.simvar_cutwin:
        selection["simvar_windows"] = dict(config.selection.simvar_cutwin)
    if prescale > 1.0:
        selection["sim_prescale"] = prescale
    if config.selection.snrmax_filters:
        selection["snrmax_filters"] = config.selection.snrmax_filters
    # USESIM_SNIA/NONIA select on the simulated true type (snana.F90:3952).
    if not config.selection.usesim_snia:
        selection["usesim_snia"] = False
    if not config.selection.usesim_nonia:
        selection["usesim_nonia"] = False
    return selection


# The Trest-family cuts a photo-z fit relaxes before fitting and re-applies
# after (snlc_fit.F90:3289-3312). 'trest' is an epoch cut, the rest per-SN.
PHOTOZ_TREST_CUTS = ('trest', 'trestmin', 'trestmax', 't0gapmax', 'tgapmax')


def trest_zcut(edge: int, z: float, trest: float) -> float:
    """Loosen one edge of a rest-frame cut for a trial redshift: edge 1 goes
    below ``trest``, edge 2 above (TRESTzCUT, snlc_fit.F90:3661)."""
    z1 = 1.0 + z
    if edge == 1:
        return trest * (1.0 / z1 if trest >= 0.0 else z1)
    return trest * (z1 if trest >= 0.0 else 1.0 / z1)


def relax_photoz_trest_cuts(config, cutwin, epoch_cutwin):
    """``cutwin`` and ``epoch_cutwin`` with the Trest-family cuts widened for
    the fit, and the nominal ones, which are re-tested after it
    (PHOTOZ_SAVECUT, snlc_fit.F90:3703)."""
    z = config.photoz.photoz_bound[1]
    nominal = {name: window for store in (cutwin, epoch_cutwin)
               for name, window in store.items() if name in PHOTOZ_TREST_CUTS}

    def widened(store):
        return {name: ((trest_zcut(1, z, lo), trest_zcut(2, z, hi))
                       if name in PHOTOZ_TREST_CUTS else (lo, hi))
                for name, (lo, hi) in store.items()}

    return widened(cutwin), widened(epoch_cutwin), nominal


def cut_windows(config):
    """``(cutwin, epoch_cutwin, nominal)``: the ``CUTWIN_`` windows as the
    run applies them. A photo-z fit widens the Trest ones for the fit and
    re-tests ``nominal``, the windows as given, after it (snlc_fit.F90:3289,
    PHOTOZ_SAVECUT :3703)."""
    selection = config.selection
    if not config.photoz.dofit_photoz:
        return selection.cutwin, selection.epoch_cutwin, {}
    return relax_photoz_trest_cuts(config, selection.cutwin,
                                   selection.epoch_cutwin)


def mark_errflags(batch, *marks):
    """``batch`` with the ERRFLAG_FIT of each ``(snids, code)`` mark set, a
    later mark winning over an earlier one."""
    sid = np.char.strip(np.asarray(batch.snids).astype(str))
    errflag = np.array(batch.errflag_fit)
    for snids, code in marks:
        errflag[np.isin(sid, np.char.strip(np.asarray(snids).astype(str)))] = code
    return dataclasses.replace(batch, errflag_fit=errflag)


def _fitwin_mask(windows, model, results, redshifts, mu_ref, batch, config):
    """Per-SN mask for the FITWIN_ windows on the finished fit.

    Quantities come from the model's own FITRES columns, so derived ones (SALT's
    mB) and their errors are covered without naming any parameter here.
    """
    z1 = 1.0 + np.asarray(redshifts)
    quantities = {}
    for column, values, errs in model.get_fitres_param_columns(
            results.params, results.param_errors, redshifts, mu_ref):
        quantities[column] = np.asarray(values)
        quantities[f"{column}_ERR"] = np.asarray(errs)
        quantities[f"{column}_ERRZ1"] = np.asarray(errs) / z1

    quantities['PROB'] = results.fitprob(model)

    if 'ZPULL' in windows:
        if 'zPHOT' not in results.params or batch.photoz is None:
            raise SNANAError(
                "FITWIN_ZPULL cuts on (z_fit - z_prior)/z_prior_err, which "
                "needs a fitted redshift and a host photo-z prior; this fit "
                "has neither.", function_name="FITWIN_ZPULL")
        zfit = np.asarray(results.params['zPHOT'])
        zfiterr = np.asarray(results.param_errors['zPHOT'])
        dz = 0.01
        mu = config.fit.distance_modulus(zfit)
        mu2 = config.fit.distance_modulus(zfit + dz)
        dzdmu = dz / (np.asarray(mu) - np.asarray(mu2))
        zmagerr = config.data.snmagrms_for_photoz * np.abs(dzdmu)
        zerr = np.maximum(
            np.sqrt(zfiterr ** 2 + np.asarray(batch.photoz_err) ** 2
                    + zmagerr ** 2), 0.00001)
        quantities['ZPULL'] = (zfit - np.asarray(batch.photoz)) / zerr

    named = {}
    for column, window in windows.items():
        # SNANA keeps only the upper edge of _ERR windows (snlc_fit.F90:1403-1417)
        lo, hi = window
        if column.endswith(('_ERR', '_ERRZ1')):
            lo = -CUTVAL_OPEN
        named[column] = (lo, hi)

    # two loose error cuts SNANA hard-wires and always applies
    # (snlc_fit.F90:6835, tested at :12958)
    named.setdefault('COV_OFFDIAG', (-8.0, 8.0))
    if 'mB_ERR' in quantities:
        named.setdefault('mB_ERR', (-CUTVAL_OPEN, 1.0))
    windows = named

    cov = np.asarray(results.param_covariances)
    off = ~np.eye(cov.shape[1], dtype=bool)
    terms = cov[:, off]
    # a term left at -9 means the pair was never fitted, and SNANA skips it
    # rather than testing it (snlc_fit.F90:12963)
    unfitted = np.abs(terms + 9.0) < 1.0e-6
    lo_edge, hi_edge = windows['COV_OFFDIAG']
    # the window must hold for every off-diagonal term, so test the extremes
    quantities['COV_OFFDIAG'] = np.where(unfitted, hi_edge, terms).min(axis=1)
    extra = apply_range_cuts(
        {'COV_OFFDIAG': (lo_edge, hi_edge)},
        {'COV_OFFDIAG': np.where(unfitted, lo_edge, terms).max(axis=1)})

    return apply_range_cuts(windows, quantities) & extra


def _fit_error_bounds_mask(fit_parameters, held_params, results):
    """Drop events whose fitted errors are NaN or outside SNANA's fixed ranges
    (CRAZYFITERR, snlc_fit.F90:2436). SNANA sets ERRFLAG_FIT_CRAZYERR and
    returns, so the event never reaches the FITRES.

    Each parameter carries its own bounds, so a model is checked on exactly the
    parameters it declares them for.
    """
    bounds = {name: spec['error_bounds']
              for name, spec in fit_parameters.items()
              if 'error_bounds' in spec}
    keep = np.ones(results.n_sne, dtype=bool)
    for name, (lo, hi) in bounds.items():
        err = np.asarray(results.param_errors[name])
        keep &= ~np.isnan(err)
        if name not in held_params:
            keep &= (err >= lo) & (err <= hi)
    return keep


def _extreme(reduce, values, live, initial):
    """A row's extreme over its live epochs, NaN where it has none."""
    return np.where(live.any(axis=1),
                    reduce(values, axis=1, initial=initial, where=live), np.nan)


def _bands_with_epochs(inside, filter_indices, n_filters):
    """Each event's number of bands with an epoch in ``inside``."""
    return np.stack([(inside & (filter_indices == b)).any(axis=1)
                     for b in range(n_filters)], axis=1).sum(axis=1)


def _sncid_pegged_mask(config, model, results, fitwin_windows):
    """OPT_SNCID_LIST bit 0 keeps every event but those whose fit pegged against
    a flat-prior edge, and those whose FITPROB fell under a tenth of
    FITWIN_PROB's lower bound (snlc_fit.F90:12700-12708).

    The margin is the parameter's own, because 0.05 is a width rather than a
    fraction: it means one thing on a shape running over units and quite
    another on a colour running over tenths. SNANA tests its shape slot alone,
    so declaring the margin only there reproduces it.
    """
    keep = np.ones(results.n_sne, dtype=bool)
    for name, spec in model.get_fit_parameters().items():
        margin = spec.get('peg_margin')
        window = parameters.flat_window(config, name, spec)
        if margin is None or window is None or name not in results.params:
            continue
        lo, hi, _ = window
        value = np.asarray(results.params[name])
        keep &= (value >= lo + margin) & (value <= hi - margin)
    if 'PROB' in fitwin_windows:
        keep &= results.fitprob(model) >= fitwin_windows['PROB'][0] / 10.0
    return keep


def _post_fit_trest_mask(config, saved_nominal, effective_cutwin, results,
                         batch, fit_mask):
    """The Trest block of FITANA_CUTS, on the finished fit.

    FITANA_STORE recomputes the quantities through GET_TRESTVAR with the
    *nominal* windows (FITANA_CUTWIN_*, snlc_fit.F90:13236), which under a
    photo-z fit are not the widened ones PHOTOZ_SAVECUT left in cutwin_var,
    and cutwin_var is what the cuts then test against (:12851-12862).
    """
    fitted_z = photoz.fitted_z(config, results.params, batch.redshifts)
    t0_offset = (np.asarray(results.params['PKMJD'])
                 - np.asarray(batch.reference_mjds))
    trest = ((np.asarray(batch.mjd_offsets) - t0_offset[:, None])
             / (1.0 + fitted_z)[:, None])
    fidx = np.asarray(batch.filter_indices)
    n_filt = len(batch.filter_names)
    effective = effective_cutwin
    nominal = {**effective_cutwin, **saved_nominal}

    # fitted epochs only (snlc_fit.F90:13221, snana.F90:20872)
    obs_ok = np.asarray(fit_mask)
    if 'trest' in nominal:
        obs_ok = obs_ok & in_window(trest, *nominal['trest'])
    keep = np.ones(batch.n_sne, dtype=bool)
    fitted = {'trestmin': _extreme(np.min, trest, obs_ok, np.inf),
              'trestmax': _extreme(np.max, trest, obs_ok, -np.inf)}
    # T0GAPWIN is built from the nominal edges (snana.F90:21105-21106)
    fitted['tgapmax'], fitted['t0gapmax'] = gap_maxima(
        trest, obs_ok,
        nominal.get('trestmin', (0.0, CUTVAL_OPEN))[1],
        nominal.get('trestmax', (-CUTVAL_OPEN, 0.0))[0])

    for name, value in fitted.items():
        if name in effective:
            keep &= in_window(value, *effective[name]) | np.isnan(value)
    # TrestRange_FIT against the lower edge only (snlc_fit.F90:12905-12909)
    if 'trestrange' in effective:
        span = fitted['trestmax'] - fitted['trestmin']
        keep &= (span >= effective['trestrange'][0]) | np.isnan(span)

    # NFILT_TRESTMIN/MAX (snana.F90:20880, snlc_fit.F90:13065)
    for name in ('trestmin', 'trestmax'):
        threshold = effective.get(f'nfilt_{name}', (1.0, 100.0))[0]
        inside = obs_ok & in_window(
            trest, *nominal.get(name, (-CUTVAL_OPEN, CUTVAL_OPEN)))
        keep &= _bands_with_epochs(inside, fidx, n_filt) >= threshold

    if not keep.all():
        log.line(logger, f"Post-fit Trest cut: dropped {int((~keep).sum())} SNe")
    return keep


def _peakmjd_window_mask(config, results):
    """PEAKMJD_SANITY, the post-fit window on the fitted PKMJD
    (snlc_fit.F90:12813-12828).

    A zero lower edge means unset, and SNANA substitutes CUTWIN_PEAKMJD +/- 20.
    Both edges are strict here, unlike the inclusive FITWIN_ windows.
    """
    lo, hi = config.data.peakmjd_sanity
    if lo == 0.0:
        lo, hi = config.selection.cutwin.get('peakmjd', (-CUTVAL_OPEN, CUTVAL_OPEN))
        lo, hi = lo - 20.0, hi + 20.0
    pkmjd = np.asarray(results.params['PKMJD'])
    return (pkmjd > lo) & (pkmjd < hi)


def _nfilt_snrmax_mask(windows, batch, fit_mask, filter_names, fiterr):
    """NFILT_SNRMAX recounted over the epochs that survived the fit
    (snlc_fit.F90:13038), or None when neither window is set.

    The SNR is SNR_FIT, over the error carrying the MAXFRAC fudge, so a
    deweighted passband stops counting (snlc_fit.F90:4215, :13277). Unlike the
    pre-fit count, SNANA starts each band's maximum at -99 (:2042) and counts
    a band only strictly above the SNRMAX cut (:13219).
    """
    cuts = [(windows[w][0], windows.get(s, (-CUTVAL_OPEN,))[0])
            for w, s in (('nfilt_snrmax', 'snrmax'),
                         ('nfilt_snrmax2', 'snrmax2'))
            if w in windows]
    if not cuts:
        return None
    err = np.asarray(fiterr)
    fidx = np.asarray(batch.filter_indices)
    snr = np.where(fit_mask & (err > 0), np.asarray(batch.fluxcals) / err,
                   -99.0)
    band_max = np.stack(
        [np.where(fit_mask & (fidx == b), snr, -99.0).max(axis=1)
         for b in range(len(filter_names))], axis=1)
    keep = np.ones(batch.n_sne, dtype=bool)
    for nfilt_lo, snrmax_lo in cuts:
        keep &= (band_max > snrmax_lo).sum(axis=1) >= nfilt_lo
    return keep


def post_fit_cuts(fitter, config, fitwins, batch, results):
    """FITANA_CUTS on the finished fit of a data version ``config`` names;
    ``fitwins`` are its ``FITWIN_`` windows (:meth:`LightCurveFitter.fitwin_windows`).
    Returns the events kept, the ERRFLAG_FIT of each event the fit itself
    abandoned (a round-1 window or ZFIT_RANGE), 0 elsewhere, and one CUTSTAT
    row per cut.
    """
    model = fitter.model
    rows = []
    lam = batch.photometry['filter_mean_wavelengths']
    n_total = results.n_sne
    fit_mask = np.asarray(results.observation_mask)
    keep = np.ones(n_total, dtype=bool)
    codes = np.zeros(n_total, dtype=int)
    # OPT_SNCID_LIST bit 0 skips the other post-fit cuts (snlc_fit.F90:12808)
    if config.selection.opt_sncid_list & 1:
        keep = narrow(keep, 'SNCID_PEGGED',
                      _sncid_pegged_mask(config, model, results, fitwins),
                      rows)
    else:
        if results.iter1_verdicts:
            # the last rejecting window's code wins, as in SNANA
            iter1_keep = np.ones(n_total, dtype=bool)
            iter1_codes = np.zeros(n_total, dtype=int)
            for name, errflag in ITER1_ERRFLAG:
                verdict = results.iter1_verdicts.get(name)
                if verdict is None:
                    continue
                verdict = np.asarray(verdict)
                iter1_codes = np.where(verdict, iter1_codes, errflag)
                iter1_keep &= verdict
            codes = np.where(iter1_keep, codes, iter1_codes)
            keep = narrow(keep, 'FITWIN_ITER1', iter1_keep, rows)

        # LTEST_ZPHFILTER (snlc_fit.F90:3569-3572, :3616-3630)
        if config.photoz.dofit_photoz and config.fit.n_fit_iterations > 1:
            z_min, z_max = map(np.asarray, photoz.band_z_ranges(
                fitter.restlambda_usefit, lam))
            zphot = np.asarray(results.params['zPHOT'])[:, None]
            zerr = np.asarray(results.param_errors['zPHOT'])[:, None]
            # the fit bands are the first indices, in z_min's order; a band
            # the data has beyond them is numbered after
            fitted_band = band_snr_maxima(batch, mask=fit_mask)[2][
                :, :fitter.n_filters]
            outside = fitted_band & ((zphot - zerr < z_min) | (zphot + zerr > z_max))
            zfit_ok = fitted_band.any(axis=1) & ~outside.any(axis=1)
            codes = np.where(keep & ~zfit_ok, ERRFLAG_ZFIT_RANGE, codes)
            keep = narrow(keep, 'ZFIT_RANGE', zfit_ok, rows)

        cutwin, _, photoz_trest_nominal = cut_windows(config)
        fitted_z = photoz.fitted_z(config, results.params, batch.redshifts)
        # the last round's errors, as SNANA accumulates SNRMAX on its final
        # iteration (snlc_fit.F90:4411)
        nfilt_keep = _nfilt_snrmax_mask(
            cutwin, batch, fit_mask, fitter.filters,
            fitter.fudged_fluxcalerrs(
                lam, config.fit.fudgeall_maxfrac,
                np.asarray(results.params['PKMJD']) - batch.reference_mjds,
                np.asarray(fitted_z), batch.mjd_offsets,
                batch.filter_indices, fit_mask, batch.fluxcals,
                batch.fluxcalerrs, config.fit.n_fit_iterations))
        if nfilt_keep is not None:
            keep = narrow(keep, 'NFILT_SNRMAX_FIT', nfilt_keep, rows)
        keep = narrow(keep, 'TREST', _post_fit_trest_mask(
            config, photoz_trest_nominal, cutwin, results, batch, fit_mask),
            rows)
        # always run: two of the windows are hard-wired defaults in SNANA
        keep = narrow(keep, 'FITWIN', _fitwin_mask(
            fitwins, model, results,
            np.asarray(fitted_z),
            results.reference_mu(config, batch), batch, config), rows)
        keep = narrow(keep, 'PEAKMJD',
                      _peakmjd_window_mask(config, results), rows)
    if config.fit.check_crazyfiterr:
        keep = narrow(keep, 'CRAZYFITERR', _fit_error_bounds_mask(
            fitter.fit_parameters, fitter.held_params, results), rows)

    # MXLC_FIT caps fitted events (N_SNLC_FITCUTS, snana.F90:2895)
    if keep.sum() > config.selection.mxlc_fit:
        cap = np.flatnonzero(keep)[config.selection.mxlc_fit:]
        keep[cap] = False
        rows.append(('MXLC_FIT', int(np.count_nonzero(keep))))
    return keep, codes, rows


def select(batch, results, precut_batch, keep, codes):
    """Select output rows using the shared fit verdict and record its codes."""
    written = keep
    if precut_batch is not None:
        snids = np.asarray(results.snids)
        precut_batch = mark_errflags(
            precut_batch,
            *[(snids[codes == code], code)
              for code in np.unique(codes[codes != 0])],
            (snids[written], 0))
    return (batch.filter_by_mask(written), results.filter_by_mask(written),
            precut_batch)


# largest EPCUT_SNRMIN floor that still reads as a cut (snana.F90:13506)
EPCUT_SNRMIN_OFF = 99.9999


# the cuts SELECT_EPOCH_DRIVER(2) applies after SET_PEAKMJD (snana.F90:17221)
_POST_EPOCH_CUTS = {'tobs', 'trest', 'lamobs', 'lamrest'}


# the CUTWIN epoch cuts applied as the light curve is read (PASS_EPOCH_CUTS,
# snana.F90:13243); the rest wait for LSIG
_READ_EPOCH_CUTS = {'mjd', 'mjd_exclude', 'zpadu', 'zpnpe', 'psf',
                    'snr_nodetect'}


# same night within 0.4 day (DT_SAMENIGHT, snana.F90:528)
_DT_SAMENIGHT = 0.400


def early_lightcurve_mask(cfg, sn_idx, n_sne, mjds, bands, fluxcal, fluxcalerr,
                          phot_data, phot_indices):
    """EARLYLC_STRING: keep epochs until MAXOBS obs or MAXNIGHT nights pass.

    SELECT_EARLYLC (snana.F90:16050) walks each SN's epochs in file order,
    carrying counters that no vectorised form reproduces, so this is a per-SN
    scan. It is called for every epoch regardless of earlier rejections, so the
    counters here run over the full epoch list too.
    """
    photmask = int(cfg["photmask"])
    photmask_start = int(cfg["photmask_start"])
    need_flag = photmask or photmask_start
    if need_flag and 'PHOTFLAG' not in phot_data.colnames:
        raise SNANAError(
            "EARLYLC_STRING PHOTMASK configured but PHOT file has no "
            "PHOTFLAG column", function_name="early_lightcurve_mask")
    flags = (np.array(phot_data['PHOTFLAG'], dtype=np.int64)[phot_indices]
             if need_flag else np.zeros(len(mjds), dtype=np.int64))
    if 'PHOTPROB' in phot_data.colnames:
        probs = np.array(phot_data['PHOTPROB'], dtype=np.float64)[phot_indices]
    elif cfg["photprobmin"] > -900.0:
        raise SNANAError(
            "EARLYLC_STRING PHOTPROBMIN configured but PHOT file has no "
            "PHOTPROB column", function_name="early_lightcurve_mask")
    else:
        probs = np.full(len(mjds), 9.0)

    short = np.array([filter_band(b) for b in bands])
    in_filt = (np.ones(len(mjds), dtype=bool) if cfg["filters"] is None
               else np.isin(short, list(cfg["filters"])))
    snr = np.divide(fluxcal, fluxcalerr, out=np.zeros_like(fluxcal),
                    where=fluxcalerr > 0)

    keep = np.ones(len(mjds), dtype=bool)
    order = np.argsort(sn_idx, kind='stable')
    starts = np.searchsorted(sn_idx[order], np.arange(n_sne))
    ends = np.searchsorted(sn_idx[order], np.arange(n_sne), side='right')
    for lo, hi in zip(starts, ends):
        n_obs = n_night = n_snr_start = n_mask_start = 0
        mjd_last = mjd_last_select = -999.0
        for ep in order[lo:hi]:
            if fluxcalerr[ep] > 0 and snr[ep] > cfg["snr_start"]:
                n_snr_start += 1
            if not photmask_start or (flags[ep] & photmask_start):
                n_mask_start += 1
            passes = (in_filt[ep]
                      and snr[ep] >= cfg["snrmin"] and n_snr_start > 0
                      and probs[ep] >= cfg["photprobmin"]
                      and (not photmask or (flags[ep] & photmask))
                      and n_mask_start > 0)
            if passes:
                n_obs += 1
                if (mjds[ep] - mjd_last) >= _DT_SAMENIGHT:
                    n_night += 1
                    mjd_last = mjds[ep]
            select = n_obs <= cfg["maxobs"] and n_night <= cfg["maxnight"]
            if select:
                mjd_last_select = mjds[ep]
            elif cfg["ndayadd"] > 0 and mjd_last_select > 10000:
                select = (int(mjds[ep] - mjd_last_select + 0.5)
                          < cfg["ndayadd"])
            keep[ep] = select
    return keep


def epoch_quality_cuts(obs_keep_mask, epoch_windows, epoch_snrmin, head_data,
                        phot_data, phot_mjd, phot_fluxcal, phot_fluxcalerr,
                        valid_phot_indices, sn_idx_all, band_codes,
                        unique_bands, photflag_detect, fluxerrcalc_zpterr,
                        zp_fluxcal, *, at_read):
    """Epoch CUTWIN cuts and EPCUT_SNRMIN, ANDed into the keep mask: with
    ``at_read`` those PASS_EPOCH_CUTS applies (snana.F90:13243), otherwise
    the LSIG ones (:17421)."""
    if epoch_windows:
        mjds = phot_mjd[valid_phot_indices]

        def _phot_col(col, key):
            if col not in phot_data.colnames:
                raise SNANAError(f"CUTWIN_{key.upper()} configured but PHOT "
                                 f"file has no {col} column",
                                 function_name="epoch_quality_cuts")
            return np.asarray(phot_data[col], dtype=np.float64)[valid_phot_indices]

        for key, (lo, hi) in epoch_windows.items():
            if (key in _READ_EPOCH_CUTS) != at_read:
                continue
            if key == 'mjd':
                obs_keep_mask &= in_window(mjds, lo, hi)
            elif key == 'mjd_exclude':
                obs_keep_mask &= ~in_window(mjds, lo, hi)
            elif key == 'zperr':
                obs_keep_mask &= in_window(_phot_col('ZEROPT_ERR', key), lo, hi)
            elif key == 'zpadu':  # SNLC_ZEROPT_forCUT in ADU (snana.F90:13934)
                obs_keep_mask &= in_window(_phot_col('ZEROPT', key), lo, hi)
            elif key == 'zpnpe':  # the same zeropoint in photoelectrons (:13944)
                gain = _phot_col('GAIN', key)
                zp_pe = (_phot_col('ZEROPT', key)
                         + 2.5 * np.log10(np.where(gain < 0.01, 0.001, gain)))
                obs_keep_mask &= in_window(zp_pe, lo, hi)
            elif key == 'photprob':
                pp = _phot_col('PHOTPROB', key)
                obs_keep_mask &= (pp > lo) | (pp < -0.001)
            elif key == 'psf':  # LPSF on SNLC_PSF_FWHM_ARCSEC (snana.F90:13963)
                if 'PIXSIZE' not in head_data.colnames:
                    raise SNANAError(
                        "CUTWIN_PSF configured but HEAD file has no PIXSIZE "
                        "column", function_name="epoch_quality_cuts")
                pixsize = np.asarray(head_data['PIXSIZE'],
                                     dtype=np.float64)[sn_idx_all]
                fwhm = _phot_col('PSF_SIG1', key) * pixsize * FWHM_PER_SIGMA
                obs_keep_mask &= in_window(fwhm, lo, hi)
            elif key == 'errtest':
                ferr = phot_fluxcalerr[valid_phot_indices]
                total_obs = obs_keep_mask.size
                valid_bands = unique_bands[band_codes]
                sb_ep = np.zeros(total_obs)
                if 'PIXSIZE' in head_data.colnames:
                    for b, ch in ((b, filter_band(b))
                                  for b in np.unique(valid_bands)):
                        k = f'HOSTGAL_SB_FLUXCAL_{ch}'
                        if k in head_data.colnames:
                            sb_sn = np.asarray(head_data[k],
                                               dtype=np.float64)
                            sb_ep[valid_bands == b] = \
                                sb_sn[sn_idx_all][valid_bands == b]
                    pixsize_ep = np.asarray(
                        head_data['PIXSIZE'],
                        dtype=np.float64)[sn_idx_all]
                else:
                    pixsize_ep = np.zeros(total_obs)
                errcalc = fluxcal_errcalc(
                    phot_fluxcal[valid_phot_indices], ferr,
                    _phot_col('ZEROPT', key),
                    (np.full(total_obs, fluxerrcalc_zpterr)
                     if fluxerrcalc_zpterr >= 0.0
                     else _phot_col('ZEROPT_ERR', key)),
                    _phot_col('SKY_SIG', key), _phot_col('SKY_SIG_T', key),
                    _phot_col('PSF_SIG1', key), _phot_col('PSF_SIG2', key),
                    _phot_col('PSF_RATIO', key), _phot_col('GAIN', key),
                    zp_fluxcal, sb_fluxcal=sb_ep, pixsize=pixsize_ep)
                errtest = np.where(ferr > 0, errcalc / np.where(ferr > 0, ferr, 1.0),
                                   SNANA_NULLFLOAT)
                obs_keep_mask &= in_window(errtest, lo, hi)
            elif key == 'snr_nodetect':
                # only non-detections are tested (snana.F90:13514)
                if not photflag_detect:
                    raise SNANAError(
                        "CUTWIN_SNR_NODETECT configured but PHOTFLAG_DETECT "
                        "is unset, so no epoch can be called a non-detection",
                        function_name="epoch_quality_cuts")
                pf = _phot_col('PHOTFLAG', key).astype(np.int64)
                ferr = phot_fluxcalerr[valid_phot_indices]
                snr = np.where(ferr > 0,
                               phot_fluxcal[valid_phot_indices]
                               / np.where(ferr > 0, ferr, 1.0), 0.0)
                detected = (pf & int(photflag_detect)) > 0
                obs_keep_mask &= detected | in_window(snr, lo, hi)
            elif key not in _POST_EPOCH_CUTS:
                raise SNANAError(
                    f"CUTWIN_{key.upper()} is not a supported epoch cut",
                    function_name="epoch_quality_cuts")

    if epoch_snrmin:
        floors = np.array(filter_values(epoch_snrmin, unique_bands, -np.inf))[band_codes]
        if at_read:
            # a floor above 99.9999 drops the band outright, which is how
            # 'u 999' turns a band off (snana.F90:13355)
            return obs_keep_mask & (floors <= EPCUT_SNRMIN_OFF)
        # the per-filter S/N floor itself (snana.F90:17390, LSIG >= floor)
        snr_ep = np.where(
            phot_fluxcalerr[valid_phot_indices] > 0,
            phot_fluxcal[valid_phot_indices]
            / np.where(phot_fluxcalerr[valid_phot_indices] > 0,
                       phot_fluxcalerr[valid_phot_indices], 1.0),
            SNANA_NULLFLOAT)
        obs_keep_mask &= snr_ep >= floors
    return obs_keep_mask


def post_peak_epoch_cuts(batch, epoch_windows, filter_lamavg):
    """Peak/wavelength-dependent epoch cuts (SNANA LSIG: LTobs/LTrest/LLAM),
    applied after the final peak is known. ANDed into observation_mask so they
    feed the SNRMAX/TREST cut variables and the fit, exactly as the column-only
    epoch cuts do at load time."""
    post_cuts = {k: v for k, v in (epoch_windows or {}).items()
                 if k in _POST_EPOCH_CUTS}
    if not post_cuts:
        return batch
    obs_ok = np.asarray(batch.observation_mask)
    z = np.asarray(batch.redshifts)
    z1 = trest_z1(z)[:, None]
    keep_ep = np.ones_like(obs_ok)
    if 'tobs' in post_cuts or 'trest' in post_cuts:
        if batch.peakmjd_offsets is None:
            raise SNANAError("CUTWIN_TOBS/TREST need a peak-MJD estimate",
                             function_name="post_peak_epoch_cuts")
        tobs = (np.asarray(batch.mjd_offsets)
                - np.asarray(batch.peakmjd_offsets)[:, None])
        if 'tobs' in post_cuts:
            keep_ep &= in_window(tobs, *post_cuts['tobs'])
        if 'trest' in post_cuts:
            keep_ep &= in_window(tobs / z1, *post_cuts['trest'])
    if 'lamobs' in post_cuts or 'lamrest' in post_cuts:
        if not filter_lamavg:
            raise SNANAError("CUTWIN_LAMOBS/LAMREST need filter_lamavg",
                             function_name="post_peak_epoch_cuts")
        lam_by_idx = np.array([filter_lamavg[batch.filter_names[i]]
                               for i in range(len(batch.filter_names))])
        lamobs = lam_by_idx[np.asarray(batch.filter_indices)]
        # both wavelength cuts wait for a redshift (snana.F90:17309)
        has_z = ((z > 0.0) & (z < 998.0))[:, None]
        if 'lamobs' in post_cuts:
            keep_ep &= in_window(lamobs, *post_cuts['lamobs']) | ~has_z
        if 'lamrest' in post_cuts:
            keep_ep &= in_window(lamobs / z1, *post_cuts['lamrest']) | ~has_z
    return _compact_epochs(batch, obs_ok & keep_ep)


def _compact_epochs(batch, keep):
    """Shift surviving epochs down so each row stays a contiguous prefix, as
    SELECT_EPOCH_DRIVER does via MOVE_SNLC_ARRAYS (snana.F90:13392).

    Masking in place would leave holes, while observation_mask is built as
    ``arange(max_obs) < n_obs_per_sn``; code assuming that prefix, as
    fmax_clump_peakmjd does, would then read the wrong epochs."""
    keep = np.asarray(keep)
    n_obs = keep.sum(axis=1)
    order = np.argsort(~keep, axis=1, kind='stable')
    rows = np.arange(keep.shape[0])[:, None]

    def gather(arr):
        return np.asarray(arr)[rows, order]

    extras = batch.obs_extras
    if extras is not None:
        extras = {k: gather(v) for k, v in extras.items()}
    return replace(
        batch,
        mjd_offsets=gather(batch.mjd_offsets),
        fluxcals=gather(batch.fluxcals),
        fluxcalerrs=gather(batch.fluxcalerrs),
        filter_indices=gather(batch.filter_indices),
        obs_extras=extras,
        observation_mask=np.arange(keep.shape[1])[None, :] < n_obs[:, None],
        n_obs_per_sn=n_obs)


def read_epoch_mask(obs_keep_mask, snfield_list, unique_fields, field_codes,
                    epoch_ignore, head_data, sn_idx_all, phot_mjd,
                    valid_phot_indices, unique_bands, band_codes,
                    photflag_mskrej, phot_data):
    """Apply SNFIELD, EPOCH_IGNORE and PHOTFLAG rejection before peak finding."""
    # SNFIELD_LIST is an epoch cut (snana.F90:14798, :13510)
    if snfield_list and 'ALL' not in snfield_list:
        if unique_fields is None:
            raise SNANAError(
                "SNFIELD_LIST configured but PHOT file has no FIELD "
                "column", function_name="read_epoch_mask")
        obs_keep_mask &= np.isin(unique_fields, list(snfield_list))[field_codes]

    # EPOCH_IGNORE_FILE drops named (CID, MJD, band) epochs; the band is
    # a single character and the MJD match allows 0.002 d (snana.F90:13221).
    if epoch_ignore:
        head_cids = np.char.strip(np.array(head_data['SNID'], dtype='U'))
        # Only scan epochs belonging to SNe named in the ignore file.
        ignored_cids = [cid for cid, _, _ in epoch_ignore]
        rows = np.flatnonzero(np.isin(head_cids, ignored_cids)[sn_idx_all])
        ep_cids = head_cids[sn_idx_all[rows]]
        ep_mjds = np.asarray(phot_mjd[valid_phot_indices[rows]], dtype=float)
        ep_bands = np.array([filter_band(b) for b in unique_bands])[band_codes[rows]]
        drop = np.zeros(len(rows), dtype=bool)
        for cid, mjd, band in epoch_ignore:
            drop |= ((ep_cids == cid) & (ep_bands == band)
                     & (np.abs(ep_mjds - mjd) < 0.002))
        obs_keep_mask[rows[drop]] = False

    # mask 1 ORs, masks 2-5 AND (snana.F90:13461)
    mskrej = tuple(int(m) for m in photflag_mskrej)
    if any(mskrej):
        if 'PHOTFLAG' not in phot_data.colnames:
            raise SNANAError(
                "PHOTFLAG_MSKREJ configured but PHOT file has no "
                "PHOTFLAG column", function_name="read_epoch_mask")
        photflags = np.array(phot_data['PHOTFLAG'],
                             dtype=np.int64)[valid_phot_indices]
        reject = np.zeros(len(obs_keep_mask), dtype=bool)
        if mskrej[0]:
            reject |= (photflags & mskrej[0]) > 0
        for m in mskrej[1:5]:
            if m:
                reject |= (photflags & m) == m
        obs_keep_mask &= ~reject

    return obs_keep_mask


# RDOBS_DRIVER's MJD_SAFETEY around the read window (snana.F90:4858)
_READ_MARGIN_DAYS = 10.0


def read_epoch_window(mjds, sn_idx_all, peakmjds, read_redshifts, trest, zmax):
    """The epochs SNANA reads: within ``trest`` of the file's PEAKMJD plus the
    margin, where it has one (RDOBS_DRIVER, snana.F90:4878-4889), in its
    single precision. ``zmax`` stands in for a missing redshift."""
    peak = peakmjds.astype(np.float32)
    z = read_redshifts.astype(np.float32)
    z1 = np.float32(1.0) + np.where(z < 0.0, np.float32(zmax), z)
    lo = (peak + np.float32(trest[0]) * z1).astype(np.float64) - _READ_MARGIN_DAYS
    hi = (peak + np.float32(trest[1]) * z1).astype(np.float64) + _READ_MARGIN_DAYS
    windowed = (peak > 40000.0)[sn_idx_all]
    return ~windowed | ((mjds >= lo[sn_idx_all]) & (mjds <= hi[sn_idx_all]))
