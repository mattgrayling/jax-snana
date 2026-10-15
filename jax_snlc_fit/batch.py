"""The batched light curve and the quantities derived from it.

`LightCurveBatch` is the padded form every fit and every output table
consumes; `band_snr_maxima` and `apply_snana_selection` are the per-SN quantities
and cuts derived from one. Kept apart from the loader so that reading files and
holding the result are separate concerns, and so the table writers can depend on
the batch without depending on the loader.
"""

import logging
from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core.utils.cutwin import (CUTVAL_OPEN, apply_range_cuts,
                                        band_maxima, gap_maxima, in_window,
                                        narrow, range_cut_masks)
from jax_snlc_core.photometry.filter_system import filter_band, filter_values
from jax_snlc_core.utils import SNANA_NULLFLOAT
from jax_snlc_core.utils import snana_log as slog
from jax_snlc_core.utils.error_handling import SNANAError
from .photoz import RedshiftPriors

logger = logging.getLogger(__name__)

# MODEL_ indices SNANA counts as true SNIa (ISMODEL_SNIa, sntools.c:8315):
# MLCS2k2, SNOOPY, SALT2, SALT3/SNEMO (both 9), BayeSN.
SNANA_SNIA_MODEL_INDICES = (3, 4, 6, 9, 13)

MXCID_CHECK = 99999999      # snana.F90:21

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


def absent_value(dtype):
    """What a column holds for an event whose HEAD file never had it: SNANA's
    null, -9, which its reader initialises every field to before reading
    what the file has (NULLFLOAT, sntools.c:8604); NULLSTRING for a string
    (sntools.h:108)."""
    kind = np.dtype(dtype).kind
    if kind in 'fiu':
        return -9
    if kind in 'US':
        return 'NULL'
    raise SNANAError(f"no absent value for columns of dtype {dtype}",
                     function_name="absent_value")


def trest_z1(redshifts):
    """1+z for rest-frame times, per SN: 1 where the redshift is undefined,
    so Trest is Tobs (snana.F90:17166)."""
    z = np.asarray(redshifts)
    return np.where(z > 0.0, 1.0 + z, 1.0)


def pad_epochs(arr, width):
    """``arr`` [n_sne, n_obs] padded out to ``width`` epochs: zeros, or for a
    string column its last value, so a label column stays a valid label."""
    a = np.asarray(arr)
    if a.shape[1] == width:
        return a
    return np.pad(a, ((0, 0), (0, width - a.shape[1])),
                  mode='edge' if a.dtype.kind in 'US' else 'constant')


@dataclass
class LightCurveBatch:
    """
    Batched light curve data for vectorised GPU fitting.

    All arrays are padded to uniform shape for efficient JAX operations.
    Invalid/padded observations are masked via observation_mask.

    The arrays live on the host. The fit is handed one chunk at a time and
    jax moves that chunk across as it enters compiled code, so the sample
    never occupies the accelerator.

    Attributes:
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

    # the fit bands' calibrated filters (fit_photometry), set by run.load
    photometry: Optional[Dict[str, jnp.ndarray]] = None
    # a photo-z fit's redshift priors, one row per SN, set by run.load
    z_priors: Optional[RedshiftPriors] = None
    # OPT_SNCID_LIST bit 1: each fit parameter's [n_sne] starting value from
    # SNCID_LIST_FILE, NaN for an event it does not list; set by run.load
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

            def part(v, b, k):
                arrays = [np.asarray(w[k]) for w in vals if k in w]
                dtype = np.result_type(*arrays)
                # HEAD vector columns (including host quantiles) may differ
                # in width between files, just as epoch columns do.
                columns = None
                if two_d:
                    columns = width
                elif arrays[0].ndim == 2:
                    columns = max(a.shape[1] for a in arrays)
                if k in v:
                    value = np.asarray(v[k])
                    if two_d:
                        return pad_epochs(value, width)
                    if columns is None:
                        return value
                    return np.pad(value, ((0, 0), (0, columns - value.shape[1])),
                                  constant_values=absent_value(dtype))
                shape = (b.n_sne, columns) if columns is not None else (b.n_sne,)
                return np.full(shape, absent_value(dtype), dtype=dtype)

            return {k: np.concatenate([part(v, b, k) for v, b in zip(vals, batches)])
                    for k in sorted(keys)}

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

    def filter_by_mask(self, mask: np.ndarray) -> 'LightCurveBatch':
        """The SNe where ``mask`` is True, in batch order."""
        mask = np.asarray(mask)
        if mask.shape != (self.n_sne,):
            raise SNANAError(
                f"mask of {mask.shape} does not match n_sne={self.n_sne}; "
                f"a shorter mask would silently drop the tail of the sample",
                function_name="LightCurveBatch.filter_by_mask")
        return self.take(np.flatnonzero(mask))


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


# per-SN CUTWIN_ quantities
#: the per-SN quantities a CUTWIN_<name> can window, as apply_snana_selection
#: evaluates them; a band suffix on the prefixed ones names one filter
SN_CUT_NAMES = frozenset({
    'redshift', 'redshift_err', 'mwebv', 'peakmjd', 'nepoch', 'snrmax',
    'snrmax2', 'nfilt_snrmax', 'nfilt_snrmax2', 'trestmin', 'trestmax',
    'trestrange', 'nfilt_trestmin', 'nfilt_trestmax', 'tgapmax', 't0gapmax',
    'searcheff_mask',
})
SN_CUT_BAND_PREFIXES = ('snrmax_', 'sbflux_')


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
    the shared jax_snlc_core.utils.cutwin machinery against per-SN derived
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
        for b in range(len(batch.filter_names)):
            short = filter_band(batch.filter_names[b])
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

        if batch.peakmjd_offsets is not None:
            z1 = trest_z1(batch.redshifts)[:, None]
            trest = (np.asarray(batch.mjd_offsets)
                     - np.asarray(batch.peakmjd_offsets)[:, None]) / z1
            # a SN with no epochs has no extreme, so its row is NaN
            any_obs = obs_ok.any(axis=1)
            quantities['trestmin'] = np.where(
                any_obs, np.min(trest, axis=1, initial=np.inf, where=obs_ok), np.nan)
            quantities['trestmax'] = np.where(
                any_obs, np.max(trest, axis=1, initial=-np.inf, where=obs_ok), np.nan)
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
                quantities[nf_name] = np.stack(
                    [(inside & (fidx == b)).any(axis=1)
                     for b in range(len(batch.filter_names))],
                    axis=1).sum(axis=1).astype(float)
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
        z1 = trest_z1(batch.redshifts)[:, None]
        tr = (np.asarray(batch.mjd_offsets)
              - np.asarray(batch.peakmjd_offsets)[:, None]) / z1
        in_all = np.ones(n, dtype=bool)
        for nobs, tmin, tmax in nobs_trest:
            # endpoints included, as sntools.c:1972 tests them
            in_all &= (obs_ok & in_window(tr, tmin, tmax)).sum(axis=1) >= nobs
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
            slog.warn(logger, "DUPLICATE MJD",
                      f"CID={batch.snids[sn[i]]} MJD={mjd_v:.4f} BAND={band}")
        if n > 5:
            slog.warn(logger, "DUPLICATE MJD", f"{n - 5} more")
    return n


def duplicate_cid_mask(batch, abort) -> np.ndarray:
    """PARSE_CID's duplicate check (snana.F90:14190). ABORT_ON_DUPLCID defaults
    true; false keeps the first occurrence and rejects the rest."""
    # only integer CIDs below MXCID_CHECK are checked (snana.F90:14144, :14179)
    stripped = np.char.strip(np.asarray(batch.snids).astype(str))
    numeric = np.char.isdigit(stripped)
    as_int = np.where(numeric, stripped, '0').astype(np.int64)
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
    slog.warn(logger, "DUPLICATE", f"rejected {int((~keep).sum())} repeat CID(s)")
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
        listed = np.isin(np.asarray(batch.snids).astype(int),
                         [int(c) for c in sncid_list])
        # a name in SNCCID_LIST selects on the event's transient or IAUC
        # name as well as its SNID (LCIDSELECT, snana.F90:11902)
        extras = batch.head_extras
        for col in ('NAME_TRANSIENT', 'NAME_IAUC'):
            if col in extras:
                listed |= np.isin(np.char.strip(np.asarray(extras[col]).astype(str)),
                                  list(snccid_names or []))
        keep &= listed
    if sncid_ignore:
        keep &= ~np.isin(np.asarray(batch.snids).astype(int),
                         [int(c) for c in sncid_ignore])

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
