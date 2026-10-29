"""The batched light curve. `LightCurveBatch` is the padded form every fit
and every output table consumes; the ERRFLAG_FIT codes live with it, as do
the grouping of light curves by length into padded fit batches
(`choose_epoch_lengths`, `plan_fit_batches`), which the fitter sizes its
compiled shape from. Kept apart from the loader so that reading files and
holding the result are separate concerns.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core.filters import filter_band
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import MAG_ZEROFLUX, SNANA_NULLFLOAT

from .photoz import RedshiftPriors

logger = logging.getLogger(__name__)


# ERRFLAG_FIT codes (snlc_fit.F90:99, snana.F90:181); -9 before a fit
ERRFLAG_NO_FIT = -9
ERRFLAG_FITPREP_NOEPOCHS = 2
ERRFLAG_FITPREP_NEPOCH = 3
ERRFLAG_FITPREP_QUANTILES = 4
ERRFLAG_FITPREP_ADJUST = 5
ERRFLAG_FIT_CUTS = 12
ERRFLAG_ZFIT_RANGE = 14
ERRFLAG_FIT_FITPROB_ITER1 = 15
ERRFLAG_FIT_SHAPE_ITER1 = 16
ERRFLAG_FIT_COLOR_ITER1 = 17
ERRFLAG_MNFIT_NAN = 93

FIT_REASON_DESCRIPTIONS = {
    0: 'Passed fit criteria',
    ERRFLAG_NO_FIT: 'Not fitted',
    ERRFLAG_FITPREP_NOEPOCHS: 'No usable observations',
    ERRFLAG_FITPREP_NEPOCH: 'Too few usable observations',
    ERRFLAG_FITPREP_QUANTILES: 'Missing or unusable photometric-redshift quantiles',
    ERRFLAG_FITPREP_ADJUST: 'Failed initial chi-squared cut',
    ERRFLAG_FIT_CUTS: 'Failed selection or fit criteria',
    ERRFLAG_ZFIT_RANGE: 'Fitted redshift outside allowed range',
    ERRFLAG_FIT_FITPROB_ITER1: 'Failed first-round fit-probability cut',
    ERRFLAG_FIT_SHAPE_ITER1: 'Failed first-round shape cut',
    ERRFLAG_FIT_COLOR_ITER1: 'Failed first-round colour cut',
    ERRFLAG_MNFIT_NAN: 'Non-finite fitted parameters',
}

# round-1 window codes, in FITPAR_ANA's precedence (snlc_fit.F90:2139-2155)
ITER1_ERRFLAG = (('PROB', ERRFLAG_FIT_FITPROB_ITER1),
                 ('SHAPE', ERRFLAG_FIT_SHAPE_ITER1),
                 ('COLOR', ERRFLAG_FIT_COLOR_ITER1),
                 ('CHI2RED_INI', ERRFLAG_FITPREP_ADJUST),
                 ('CHI2RED_INI2', ERRFLAG_FITPREP_ADJUST))


def absent_value(arrays):
    """What a column holds for an event whose HEAD file never had it: SNANA's
    null, -9, which its reader initialises every field to before reading
    what the file has (NULLFLOAT, sntools.c:8604); NULLSTRING for a string
    (sntools.h:108)."""
    dtype = np.result_type(*arrays)
    kind = dtype.kind
    if kind in 'US' or (kind == 'O' and any(
            isinstance(value, (str, bytes)) for array in arrays for value in array.flat)):
        return 'NULL'
    if kind in 'fiu':
        return -9
    raise SNANAError(f"no absent value for columns of dtype {dtype}",
                     function_name="absent_value")


def trest_z1(redshifts):
    """1+z for rest-frame times, per SN: 1 where the redshift is undefined,
    so Trest is Tobs (snana.F90:17166)."""
    z = np.asarray(redshifts)
    return np.where(z > 0.0, 1.0 + z, 1.0)


def pad_epochs(arr, width):
    """``arr`` [n_sne, n_obs] padded out to ``width`` epochs: zeros, or for a
    string column its last value (blank if there are no epochs to repeat)."""
    a = np.asarray(arr)
    if a.shape[1] == width:
        return a
    padding = ((0, 0), (0, width - a.shape[1]))
    if a.dtype.kind in 'US' and a.shape[1]:
        return np.pad(a, padding, mode='edge')
    return np.pad(a, padding, constant_values='' if a.dtype.kind in 'US' else 0)


@dataclass
class LightCurveBatch:
    """
    Batched light curve data for vectorised GPU fitting.

    All arrays are padded to uniform shape for efficient JAX operations.
    Invalid/padded observations are masked via observation_mask.

    The per-SN and per-epoch arrays live on the host. The fit is handed one
    chunk at a time and jax moves that chunk across as it enters compiled
    code, so the sample never occupies the accelerator.

    Main attributes (every field is described where it is declared):
        n_sne: Number of supernovae in batch
        max_obs: Maximum observations per SN (padding size)

        snids: SN identifiers [n_sne]
        redshifts: Redshifts [n_sne]
        redshift_errs: Redshift uncertainties [n_sne]
        mwebvs: MW E(B-V) values [n_sne]
        n_obs_per_sn: Actual observation count per SN [n_sne]
        reference_mjds: Per-SN reference MJDs for offset computation [n_sne]

        mjd_offsets: Observation time offsets from reference [n_sne, max_obs]
        filter_indices: Filter indices (0-indexed) [n_sne, max_obs]
        fluxcals: Calibrated fluxes [n_sne, max_obs]
        fluxcalerrs: Flux uncertainties [n_sne, max_obs]
        observation_mask: True for valid observations [n_sne, max_obs]

        filter_names: Mapping from index to filter name
        peakmjd_offsets: Peak MJD guess offsets from reference [n_sne] (optional)
    """
    # [n_sne]
    snids: np.ndarray
    redshifts: np.ndarray
    redshift_errs: np.ndarray
    mwebvs: np.ndarray
    n_obs_per_sn: np.ndarray
    reference_mjds: np.ndarray

    # [n_sne, max_obs]
    mjd_offsets: np.ndarray
    filter_indices: np.ndarray
    fluxcals: np.ndarray
    fluxcalerrs: np.ndarray
    observation_mask: np.ndarray
    zp_fluxcal: float             # ZP_FLUXCAL
    # a simulation's (SIMOPT_MWCOLORLAW, SIM_MWRV); None for real data
    sim_mw_colour_law: Optional[Tuple[int, float]]
    # [n_sne] degrees; NaN when absent
    ras: np.ndarray
    decs: np.ndarray
    # [n_sne] ERRFLAG_FIT: ERRFLAG_NO_FIT as loaded, then what the run
    # records against the event
    errflag_fit: np.ndarray

    filter_names: Dict[int, str] = field(default_factory=dict)

    peakmjd_offsets: Optional[np.ndarray] = None
    sim_truth: Optional[Dict[str, np.ndarray]] = None
    host_data: Optional[Dict[str, np.ndarray]] = None
    head_extras: Dict[str, np.ndarray] = field(default_factory=dict)
    obs_extras: Optional[Dict[str, np.ndarray]] = None  # [n_sne, max_obs] raw PHOT columns
    # FLUXCAL_BASELINE_CORRECT: per fit band, the [n_sne] baseline subtracted
    flux_baselines: Optional[Dict[str, np.ndarray]] = None
    survey: str = ''

    photoz: Optional[np.ndarray] = None          # [n_sne] HOSTGAL_PHOTOZ
    photoz_err: Optional[np.ndarray] = None      # [n_sne] HOSTGAL_PHOTOZ_ERR
    zpdf_quantiles: Optional[np.ndarray] = None   # [n_sne, n_q] host photo-z quantiles
    zpdf_percentiles: Optional[np.ndarray] = None  # [n_sne, n_q] their percentiles, 0 to 100
    zpdf_nbin: Optional[np.ndarray] = None         # [n_sne] the event's quantile count, 0 for none
    redshift_source_mask: Optional[np.ndarray] = None  # [n_sne] MASK_REDSHIFT_SOURCE

    # the fit bands' calibrated filters (fit_photometry), set by loader.load
    photometry: Optional[Dict[str, jnp.ndarray]] = None
    # a photo-z fit's redshift priors, one row per SN, set by loader.prepare_batch
    z_priors: Optional[RedshiftPriors] = None
    # OPT_SNCID_LIST bit 1: each fit parameter's [n_sne] starting value from
    # SNCID_LIST_FILE, NaN for an event it does not list; set by
    # loader.prepare_batch
    sncid_inival: Optional[Dict[str, np.ndarray]] = None
    # [n_sne, n_common], set by attach_mw_attenuation
    mw_att_obs: Optional[jnp.ndarray] = None
    mwebv_errs: Optional[np.ndarray] = None
    # SNLC_MWXT_MAGERR (snana.F90:15742) [n_sne, n_filters]
    mwxt_magerrs: Optional[jnp.ndarray] = None
    # ISNLC_NEPOCH_STORE, counted before the epoch cuts (snana.F90:21285)
    nepoch_store: Optional[np.ndarray] = None
    # SNLC_ZCMB, which SET_zSHIFT increments rather than re-deriving (:17890)
    zcmb: Optional[np.ndarray] = None
    # 1-based N_SNLC_PROC
    snana_index: Optional[np.ndarray] = None
    # N_SNLC_PROC total; resumes across versions (snana.F90:11666)
    n_proc: int = 0

    @property
    def n_sne(self):
        return len(self.snids)

    @property
    def max_obs(self):
        return self.mjd_offsets.shape[1]

    def take(self, rows, max_obs: Optional[int] = None) -> 'LightCurveBatch':
        """The SNe in ``rows``, in that order, at ``max_obs`` epochs.

        ``rows`` is a non-negative index array or a ``slice``; an index may
        repeat. ``max_obs`` keeps the first ``max_obs`` columns of the
        per-epoch arrays, padded as :meth:`concatenate` pads them where it is
        wider; it must be no shorter than the longest light curve selected,
        since real observations occupy the prefix of every row.

        Only the epoch axis is resized. ``zpdf_quantiles``, ``mw_att_obs`` and
        ``mwxt_magerrs`` are two-dimensional as well, but their second axis is
        quantiles, common wavelengths and filters.
        """
        if not isinstance(rows, slice):
            rows = np.asarray(rows)
            if rows.size and rows.min() < 0:
                raise SNANAError(
                    "take needs non-negative indices; a negative one reads as "
                    "a slice endpoint and selects nothing",
                    function_name="LightCurveBatch.take")
        length = self.max_obs if max_obs is None else max_obs
        longest = int(np.max(np.asarray(self.n_obs_per_sn)[rows], initial=0))
        if length < longest:
            raise SNANAError(
                f"epoch length {length} is below the longest light curve "
                f"selected ({longest}); real observations would be dropped",
                function_name="LightCurveBatch.take")

        def per_sn(arr):
            return None if arr is None else arr[rows]

        def per_epoch(arr):
            # rows first: JAX arrays have no views
            return (None if arr is None
                    else pad_epochs(arr[rows][:, :length], length))

        def per_sn_map(d):
            return None if d is None else {k: v[rows] for k, v in d.items()}

        return LightCurveBatch(
            snids=per_sn(self.snids),
            redshifts=per_sn(self.redshifts),
            redshift_errs=per_sn(self.redshift_errs),
            mwebvs=per_sn(self.mwebvs),
            n_obs_per_sn=per_sn(self.n_obs_per_sn),
            reference_mjds=per_sn(self.reference_mjds),
            mjd_offsets=per_epoch(self.mjd_offsets),
            filter_indices=per_epoch(self.filter_indices),
            fluxcals=per_epoch(self.fluxcals),
            fluxcalerrs=per_epoch(self.fluxcalerrs),
            observation_mask=per_epoch(self.observation_mask),
            filter_names=self.filter_names,
            peakmjd_offsets=per_sn(self.peakmjd_offsets),
            sim_truth=per_sn_map(self.sim_truth),
            host_data=per_sn_map(self.host_data),
            head_extras=per_sn_map(self.head_extras),
            obs_extras=(None if self.obs_extras is None else
                        {k: per_epoch(v) for k, v in self.obs_extras.items()}),
            flux_baselines=per_sn_map(self.flux_baselines),
            survey=self.survey,
            zp_fluxcal=self.zp_fluxcal,
            sim_mw_colour_law=self.sim_mw_colour_law,
            photoz=per_sn(self.photoz),
            photoz_err=per_sn(self.photoz_err),
            zpdf_quantiles=per_sn(self.zpdf_quantiles),
            zpdf_percentiles=per_sn(self.zpdf_percentiles),
            zpdf_nbin=per_sn(self.zpdf_nbin),
            redshift_source_mask=per_sn(self.redshift_source_mask),
            ras=per_sn(self.ras),
            decs=per_sn(self.decs),
            errflag_fit=per_sn(self.errflag_fit),
            photometry=self.photometry,
            z_priors=(None if self.z_priors is None
                      else jax.tree.map(per_sn, self.z_priors)),
            sncid_inival=per_sn_map(self.sncid_inival),
            mw_att_obs=per_sn(self.mw_att_obs),
            mwebv_errs=per_sn(self.mwebv_errs),
            mwxt_magerrs=per_sn(self.mwxt_magerrs),
            nepoch_store=per_sn(self.nepoch_store),
            zcmb=per_sn(self.zcmb),
            snana_index=per_sn(self.snana_index),
            n_proc=self.n_proc,
        )

    def slice(self, start: int, end: int) -> 'LightCurveBatch':
        """The SNe in ``[start, end)``."""
        return self.take(slice(start, end))

    @classmethod
    def concatenate(cls, batches: list) -> 'LightCurveBatch':
        """One batch from several, each holding events, padding each to the
        widest max_obs.

        SNANA processes one data version per pass and accumulates into a single
        output table (PROCESS_DATA_VERSION, snana.F90:5193), so versions may
        differ in epoch count, survey and filter set.
        """
        if len(batches) == 1:
            return batches[0]
        width = max(b.max_obs for b in batches)

        def cat(attr, two_d=False):
            vals = [getattr(b, attr) for b in batches]
            if any(v is None for v in vals):
                return None
            if two_d:
                return np.concatenate([pad_epochs(v, width) for v in vals])
            return np.concatenate([np.asarray(v) for v in vals])

        def cat_optional(attr, missing, columns=None):
            """Preserve available metadata and mark only absent rows as missing."""
            vals = [getattr(b, attr) for b in batches]
            if all(v is None for v in vals):
                return None
            parts = []
            for b, value in zip(batches, vals):
                shape = (b.n_sne,) if columns is None else (b.n_sne, columns)
                if value is None:
                    value = np.full(shape, missing)
                elif columns is not None:
                    value = np.pad(value, ((0, 0), (0, columns - value.shape[1])),
                                   constant_values=missing)
                parts.append(value)
            return np.concatenate(parts)

        quantile_width = max((b.zpdf_quantiles.shape[1] for b in batches
                              if b.zpdf_quantiles is not None), default=0)

        def cat_snana_index():
            """N_SNLC_PROC resumes across versions rather than restarting."""
            vals = [b.snana_index for b in batches]
            if any(v is None for v in vals):
                return None
            offsets = np.cumsum([0] + [b.n_proc for b in batches[:-1]])
            return np.concatenate(
                [np.asarray(v) + off for v, off in zip(vals, offsets)])

        def cat_dict(attr, two_d=False):
            """The union of the columns; a column one batch lacks is filled
            with SNANA's absent value for its type."""
            vals = [getattr(b, attr) for b in batches]
            if all(v is None for v in vals):
                return None
            vals = [v if v is not None else {} for v in vals]
            keys = set().union(*(set(v) for v in vals))

            result = {}
            for key in sorted(keys):
                arrays = [np.asarray(v[key]) for v in vals if key in v]
                dtype = np.result_type(*arrays)
                # HEAD vectors may differ in width between files.
                columns = None
                if two_d:
                    columns = width
                elif arrays[0].ndim == 2:
                    columns = max(a.shape[1] for a in arrays)
                needs_fill = (len(arrays) < len(batches)
                              or (columns is not None and not two_d
                                  and any(a.shape[1] < columns for a in arrays)))
                fill = 0 if attr == 'obs_extras' and key == 'PHOTFLAG' else None
                if needs_fill:
                    if fill is None:
                        fill = absent_value(arrays)
                    dtype = np.result_type(dtype, np.min_scalar_type(fill))
                parts = []
                for v, batch in zip(vals, batches):
                    if key not in v:
                        shape = ((batch.n_sne, columns) if columns is not None
                                 else (batch.n_sne,))
                        value = np.full(shape, fill, dtype=dtype)
                    else:
                        value = np.asarray(v[key], dtype=dtype)
                        if two_d:
                            value = pad_epochs(value, width)
                        elif columns is not None and value.shape[1] < columns:
                            value = np.pad(value, ((0, 0), (0, columns - value.shape[1])),
                                           constant_values=fill)
                    parts.append(value)
                result[key] = np.concatenate(parts)
            return result

        # merge filter maps, then remap each batch's indices
        names, slot_of = {}, {}
        for b in batches:
            for idx in sorted(b.filter_names):
                band = b.filter_names[idx]
                if band in slot_of:
                    continue
                slot = idx if idx not in names else max(names) + 1
                names[slot] = band
                slot_of[band] = slot

        def cat_filter_indices():
            out = []
            for b in batches:
                idx = np.asarray(b.filter_indices)
                lut = np.arange(max(int(idx.max(initial=0)),
                                    max(b.filter_names, default=0)) + 1)
                for i, band in b.filter_names.items():
                    lut[i] = slot_of[band]
                out.append(pad_epochs(lut[idx], width))
            return np.concatenate(out)

        return cls(
            snids=np.concatenate([np.asarray(b.snids) for b in batches]),
            redshifts=cat('redshifts'), redshift_errs=cat('redshift_errs'),
            mwebvs=cat('mwebvs'), mwebv_errs=cat('mwebv_errs'),
            # mwxt_magerrs is recomputed per event (snana.F90:15742)
            mw_att_obs=cat('mw_att_obs'),
            n_obs_per_sn=cat('n_obs_per_sn'),
            reference_mjds=cat('reference_mjds'),
            mjd_offsets=cat('mjd_offsets', True),
            filter_indices=cat_filter_indices(),
            fluxcals=cat('fluxcals', True),
            fluxcalerrs=cat('fluxcalerrs', True),
            observation_mask=cat('observation_mask', True),
            filter_names=names,
            peakmjd_offsets=cat('peakmjd_offsets'),
            sim_truth=cat_dict('sim_truth'),
            host_data=cat_dict('host_data'),
            head_extras=cat_dict('head_extras'),
            obs_extras=cat_dict('obs_extras', True),
            flux_baselines=cat_dict('flux_baselines'),
            survey=batches[0].survey,
            zp_fluxcal=batches[0].zp_fluxcal,
            sim_mw_colour_law=batches[0].sim_mw_colour_law,
            photoz=cat_optional('photoz', SNANA_NULLFLOAT),
            photoz_err=cat_optional('photoz_err', SNANA_NULLFLOAT),
            zpdf_quantiles=cat_optional('zpdf_quantiles', SNANA_NULLFLOAT, quantile_width),
            zpdf_percentiles=cat_optional('zpdf_percentiles', SNANA_NULLFLOAT, quantile_width),
            zpdf_nbin=cat_optional('zpdf_nbin', 0),
            redshift_source_mask=cat_optional('redshift_source_mask', 0),
            ras=cat('ras'), decs=cat('decs'),
            errflag_fit=cat('errflag_fit'),
            nepoch_store=cat('nepoch_store'),
            zcmb=cat('zcmb'),
            snana_index=cat_snana_index(),
            n_proc=sum(b.n_proc for b in batches),
        )

    def band_names(self):
        """Each filter index's short band name, in index order."""
        return [filter_band(self.filter_names[b]) for b in range(len(self.filter_names))]

    def true_fluxcals(self):
        """Each epoch's simulated true FLUXCAL, 10^(-0.4*(SIM_MAGOBS -
        ZP_FLUXCAL)), and zero where SIM_MAGOBS is undefined; None for data
        without SIM_MAGOBS."""
        simmag = (self.obs_extras or {}).get('SIM_MAGOBS')
        if simmag is None:
            return None
        simmag = np.asarray(simmag)
        return np.where(simmag < MAG_ZEROFLUX,
                        10.0 ** (-0.4 * (simmag - self.zp_fluxcal)), 0.0)

    def filter_by_mask(self, mask: np.ndarray) -> 'LightCurveBatch':
        """The SNe where ``mask`` is True, in batch order."""
        mask = np.asarray(mask)
        if mask.shape != (self.n_sne,):
            raise SNANAError(
                f"mask of {mask.shape} does not match n_sne={self.n_sne}; "
                f"a shorter mask would silently drop the tail of the sample",
                function_name="LightCurveBatch.filter_by_mask")
        return self.take(np.flatnonzero(mask))


def choose_epoch_lengths(n_obs_per_sn, max_bins: int, min_gain: float):
    """The length each epoch bin pads to, one per bin, at most ``max_bins``.

    The lengths minimise the padded work ``sum(count * length)`` exactly, by a
    dynamic programme over the distinct epoch counts. A bin is added only while
    it cuts that total by at least ``min_gain`` relative, since each costs a
    recompilation at its own length.

    Returns them ascending, the last always the largest epoch count so every SN
    has a bin to fall in, and 1 for a sample with no events, whose arrays then
    keep one empty epoch. ``max_bins <= 1`` returns that one length alone.
    Choose these once for a whole run: a caller that re-chooses per batch gets
    different lengths per batch, and a compilation for each.
    """
    n_obs = np.asarray(n_obs_per_sn).astype(int)
    if not len(n_obs):
        return [1]
    distinct, counts = np.unique(n_obs, return_counts=True)   # ascending
    n_distinct = len(distinct)
    n_wanted = int(min(max(max_bins, 1), n_distinct))
    if n_wanted <= 1:
        return [int(n_obs.max())]

    # cost[m, j]: least padded work for the first j distinct counts in m bins
    cumulative = np.cumsum(counts)
    before = np.concatenate([[0], cumulative[:-1]])
    cost = np.full((n_wanted + 1, n_distinct + 1), np.inf)
    split = np.zeros((n_wanted + 1, n_distinct + 1), dtype=int)
    cost[0, 0] = 0.0
    for m in range(1, n_wanted + 1):
        for j in range(m, n_distinct + 1):
            starts = np.arange(m - 1, j)
            bin_cost = (cumulative[j - 1] - before[starts]) * distinct[j - 1]
            best = int(np.argmin(cost[m - 1, starts] + bin_cost))
            cost[m, j] = cost[m - 1, starts[best]] + bin_cost[best]
            split[m, j] = starts[best]

    n_chosen = 1
    for m in range(2, n_wanted + 1):
        previous = cost[m - 1, n_distinct]
        if previous > 0 and (previous - cost[m, n_distinct]) / previous >= min_gain:
            n_chosen = m
        else:
            break

    chosen, j = [], n_distinct
    for m in range(n_chosen, 0, -1):
        chosen.append(int(distinct[j - 1]))
        j = split[m, j]
    return sorted(chosen)


def group_by_epoch_length(n_obs_per_sn, lengths):
    """``[(length, sn_indices), ...]``, each SN in the shortest length it fits.

    Bins with no SN are dropped, so the result never asks for a compilation
    that fits nothing. Every SN appears exactly once.
    """
    n_obs = np.asarray(n_obs_per_sn).astype(int)
    edges = np.asarray(sorted(lengths))
    if not len(edges):
        raise SNANAError("no epoch lengths to bin into",
                         function_name="group_by_epoch_length")
    if edges[-1] < n_obs.max(initial=0):
        raise SNANAError(
            f"longest epoch length {edges[-1]} is below the largest epoch "
            f"count {int(n_obs.max())}; real observations would be dropped",
            function_name="group_by_epoch_length")
    which = np.searchsorted(edges, n_obs, side='left')
    bins = [(int(edges[b]), np.flatnonzero(which == b)) for b in range(len(edges))]
    return [(n, idx) for n, idx in bins if len(idx)]


def padded_batches(rows, batch_size: int):
    """``rows`` in batches of ``batch_size``: ``[(rows, padded), ...]``, a
    short last batch padded up to ``batch_size`` by repeating its own rows,
    so every batch has one shape. A caller computes on ``padded`` and keeps
    the first ``len(rows)`` results."""
    return [(rows[lo:lo + batch_size],
             np.resize(rows[lo:lo + batch_size], batch_size))
            for lo in range(0, len(rows), batch_size)]


def plan_fit_batches(n_obs_per_sn, epoch_lengths, batch_size: int):
    """The fit calls to make: ``[(epoch_length, rows, padded), ...]``.

    The sample is binned by epoch count into ``epoch_lengths``
    (:func:`choose_epoch_lengths`), and each bin is split into
    :func:`padded_batches` of ``batch_size`` events, so the compiled shapes
    number one per length.
    """
    return [(length, rows, padded)
            for length, sn_idx in group_by_epoch_length(n_obs_per_sn,
                                                        epoch_lengths)
            for rows, padded in padded_batches(sn_idx, batch_size)]
