"""The fitter's typed configuration: six records under one aggregate, and
two (`EarlyLC`, `RequireEpochs`) that check the selection's EARLYLC_STRING
and REQUIRE_EPOCHS settings.

Public fields declare the settings accepted by Python, YAML and the CLI.
Private fields hold parser or runtime bookkeeping. Defaults live on the
fields; constructing a fitter copies the configuration so preparation does
not change the caller's settings. Fields are named after their SNANA
key, lowercased; a field naming a file holds the path, which the run
reads. Input settings are retained separately from linked-rule changes,
so a configuration can be copied and reused with different settings.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from jax_snlc_core.cosmology import reference_distance_modulus
from jax_snlc_core.peak_time import (OPTMASK_SETPKMJD_BAZIN, OPTMASK_SETPKMJD_FLUXMAX,
                                     OPTMASK_SETPKMJD_FLUXMAX2, OPTMASK_SETPKMJD_FLUXMAX3,
                                     OPTMASK_SETPKMJD_TRIGGER)

# SNANA FUDGE_FITERR window defaults (snlc_fit.F90:7072-7084)
FUDGE_FITERR_MAXITER_DEFAULT = 9
FUDGE_FITERR_TREST_DEFAULT = (999.0, 999.0, 999.0, 999.0)
FUDGE_FITERR_RESTLAM_DEFAULT = (1000.0, 1000.0, 90000.0, 90000.0)

# PRIOR_ZERRSCALE's unset value (snlc_fit.F90:7003)
PRIOR_ZERRSCALE_UNSET = 1000.01


@dataclass
class EarlyLC:
    """Early light-curve selection (PARSE_EARLYLC_STRING, snana.F90)."""

    maxobs: int = 999
    maxnight: int = 999
    ndayadd: int = 0
    filters: Optional[str] = None
    snrmin: float = -999.0
    photprobmin: float = -999.0
    photmask: int = 0
    photmask_start: int = 0
    snr_start: float = -9999.0


@dataclass
class RequireEpochs:
    """Epoch intervals and threshold (PARSE_REQUIRE_EPOCHS, snana.F90)."""

    tranges: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    snrmin: float = 0.0
    rest_frame: bool = False


@dataclass
class Data:
    """Photometry inputs, calibration and pre-fit corrections."""

    input_names = {'kcor_file': ['CALIB_FILE'],
                   'hostgal_zphot_shift': ['HOSTGAL_PHOTOZ_SHIFT'],
                   'hostgal_zspec_shift': ['HOSTGAL_SPECZ_SHIFT'],
                   'enable_magshift': ['ENABLE_MAGSHIFT_SALT2'],
                   'enable_waveshift': ['ENABLE_WAVESHIFT_SALT2']}

    # Photometry and calibration inputs
    #: SNANA VERSION_PHOTOMETRY; data version to fit, without a path. Identified
    #: by <VERSION>.LIST or a Rubin HATS catalogue's hats.properties. The
    #: version must be unique across the input search directories.
    version_photometry: Optional[str] = None
    #: SNANA VERSION_PHOTOMETRY_WILDCARD; a glob matching several data
    #: versions, all of which are read, for a sample split across directories.
    version_photometry_wildcard: Optional[str] = None
    #: SNANA PRIVATE_DATA_PATH; a directory searched for the version in place
    #: of the lcmerge tree of SNDATA_ROOT, both on its own and with the
    #: version name appended. The simulation paths are still searched after
    #: it. NULL, NONE or BLANK keeps the standard tree.
    private_data_path: Optional[str] = None
    #: SNANA KCOR_FILE; a calibration input or FITS file describing filter transmissions,
    #: the primary reference spectrum and the zero points. Required.
    kcor_file: Optional[str] = None
    #: SNANA LEAKAGE_CUT; suppress filter transmission below this fraction of
    #: its peak to remove out-of-band leakage. Zero retains the full
    #: transmission curve.
    leakage_cut: float = 0.0
    #: SNANA FILTER_REPLACE; written 'LIST1 -> LIST2', it renames each band of
    #: the first list to the band at the same position in the second, before
    #: the light curve is used.
    filter_replace: Dict[str, str] = field(default_factory=dict)
    #: SNANA PHOTOMETRY_COLUMNS; for a Rubin HATS catalogue, 'NAME column'
    #: pairs naming the Rubin column that supplies an SNANA one in place of
    #: the standard choice, e.g. 'FLUXCALERR psfDiffFluxErr_corrected'.
    #: Python/YAML takes {FLUXCALERR: psfDiffFluxErr_corrected}.
    photometry_columns: Dict[str, str] = field(default_factory=dict)
    #: SNANA PRIVATE_VARNAME_READLIST; the PRIVATE_* header columns to read
    #: from the data files, which PRIVATE_CUTWIN_STRING then cuts on.
    private_varname_readlist: Tuple[str, ...] = ()
    #: SNANA HEADER_OVERRIDE_FILE; files whose rows replace header values of
    #: real data, a redshift, a peculiar velocity, a host's photo-z quantiles
    #: or its LOGMASS(z) grid, without rewriting the data.
    header_override_file: Optional[List[str]] = None
    #: SNANA HEADER_OVERRIDE_DIR; the directory the HEADER_OVERRIDE_FILE files
    #: are looked for in.
    header_override_dir: Optional[str] = None
    #: SNANA SIM_HEADER_OVERRIDE_FILE; the override files read instead of
    #: HEADER_OVERRIDE_FILE when the data version is a simulation.
    sim_header_override_file: Optional[List[str]] = None
    #: SNANA SIM_HEADER_OVERRIDE_DIR; the directory the simulated-truth
    #: override files are looked for in.
    sim_header_override_dir: Optional[str] = None
    #: SNANA PHOTFLAG_DETECT; the PHOTFLAG bit marking an epoch as a detection.
    #: CUTWIN_SNR_NODETECT acts on the epochs without that bit only when this
    #: is set.
    photflag_detect: int = 0
    #: SNANA PHOTFLAG_TRIGGER; the PHOTFLAG bit marking the epoch that fired
    #: the trigger, needed for OPT_SETPKMJD to take the peak from the trigger
    #: MJD.
    photflag_trigger: int = 0

    # Calibration and photometric adjustments
    #: SNANA MAGCOR_FILE; a table of per-epoch magnitude corrections, keyed by
    #: CID, MJD and band, added to the measured magnitudes. A leading minus on
    #: the path subtracts the correction instead.
    magcor_file: Optional[str] = None
    #: SNANA SIM_MAGCOR_FILE; a table of per-epoch magnitude corrections in
    #: the same form as MAGCOR_FILE, applied to simulated data alone.
    sim_magcor_file: Optional[str] = None
    #: SNANA NONLINEARITY_FILE; a detector non-linearity model, one map per
    #: block of bands, applied to the measured fluxes as they are read.
    nonlinearity_file: Optional[str] = None
    #: SNANA ENABLE_MAGSHIFT; apply the MAGSHIFT entries in a model's
    #: calibration metadata. On by default; the _SALT2 spelling is SNANA's.
    enable_magshift: bool = True
    #: SNANA ENABLE_WAVESHIFT; apply the WAVESHIFT entries in a model's
    #: calibration metadata. On by default.
    enable_waveshift: bool = True
    #: SNANA MAGOBS_SHIFT_ZP; a per-band zero-point shift in magnitudes,
    #: written as SNANA's filter string, ``'gri 0.003'``, or as a
    #: Python/YAML mapping, {g: 0.003, r: 0.003, i: 0.003}.
    magobs_shift_zp: Dict[str, float] = field(default_factory=dict)
    #: SNANA MAGOBS_SHIFT_ZP_PARAMS; a quadratic A0 + A1 lam + A2 lam^2 in the
    #: band's mean wavelength in microns, added to the MAGOBS_SHIFT_ZP shift
    #: of every band.
    magobs_shift_zp_params: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: SNANA FILTER_LAMSHIFT; wavelength shift in Angstroms applied to each
    #: named filter transmission.
    filter_lamshift: Dict[str, float] = field(default_factory=dict)
    #: SNANA MAGOBS_SHIFT_PRIMARY; shift the primary reference magnitude per
    #: band, which moves the synthetic zero point rather than the data.
    magobs_shift_primary: Dict[str, float] = field(default_factory=dict)
    #: SNANA MAGOBS_SHIFT_PRIMARY_PARAMS; a quadratic in the band's mean
    #: wavelength in microns, added to that primary shift.
    magobs_shift_primary_params: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: SNANA FUDGE_FLUXCAL_OFFSET, FUDGE_FLUXCAL_ERROR, FUDGE_FLUXCAL_ERRPIX,
    #: FUDGE_FLUXERR_SCALE, FUDGE_MAG_ERROR, SIM_FUDGE_MAG_ERROR; systematic
    #: adjustments to the calibrated photometry. In order: an added flux offset,
    #: an error added in quadrature, a per-pixel error weighted by the
    #: noise-equivalent aperture, a scale on the measured error, a fractional
    #: error in magnitudes, and that last term for simulated data alone. Each
    #: is written as SNANA's filter string, ``'griz 0.05'``, or as one number
    #: for every band. Python/YAML uses mappings named offset, err_add,
    #: err_pix, err_scale, mag_err and sim_mag_err, respectively; for example,
    #: {err_scale: {g: 1.1, r: 1.1}}.
    fudge_fluxcal: Dict[str, Dict[str, float]] = field(default_factory=dict)
    #: SNANA FLUXCAL_BASELINE_CORRECT; subtract from each band the median flux
    #: of its epochs below -3 sigma, where there are at least four, and write
    #: it to the FITRES as FLUXCAL_BASELINE_<band>.
    fluxcal_baseline_correct: bool = False
    #: SNANA USESIM_TRUEFLUX; fit the noiseless simulated flux in place of the
    #: measured one, leaving the errors unchanged.
    usesim_trueflux: bool = False

    # Measurement errors
    #: SNANA FLUXERRMODEL_FILE; a map scaling the reported flux errors by
    #: observing conditions, band and field.
    fluxerrmodel_file: Optional[str] = None
    #: SNANA SIM_FLUXERRMODEL_FILE; a map in the same form as
    #: FLUXERRMODEL_FILE, applied to simulated data alone.
    sim_fluxerrmodel_file: Optional[str] = None
    #: SNANA FLUXERRMODEL_OPTMASK; options for reading the flux-error map, as
    #: a bit mask. In the fitter only 256 (dump each map as it is read) and
    #: 1024 (require the file to have a DOCUMENTATION block) have any effect.
    fluxerrmodel_optmask: int = 0
    #: SNANA FUDGE_HOSTNOISE_FILE; maps, one per SIMLIB entry, of the extra
    #: noise from the host surface brightness, which scale the flux errors by
    #: host magnitude.
    fudge_hostnoise_file: Optional[str] = None
    #: SNANA FUDGE_DATAERR_SCALE; multiply every measured flux error by this.
    fudge_dataerr_scale: float = 1.0
    #: SNANA FLUXERRCALC_ZPTERR; override the per-epoch zero-point error inside
    #: the recomputed error used by the CUTWIN_ERRTEST epoch cut. Negative,
    #: the default, uses the value in the data.
    fluxerrcalc_zpterr: float = -9.9

    # Milky Way extinction
    #: SNANA OPT_MWEBV; source of Galactic E(B-V) for each event: 0
    #: no extinction, 1 the value in the data header, 2 the SFD98 map, 3 that
    #: map scaled by 0.86.
    opt_mwebv: int = 1
    #: SNANA OPT_MWCOLORLAW; Milky Way extinction law: 0 no
    #: extinction, 89 CCM89, 94 ODONNELL94, 99 the exact Fitzpatrick 1999 law
    #: and -99 its approximation, 203 GORD03, 204 FITZ04, 208 GOOBAR08, 214
    #: MAIZ14, 216 GORD16, 219 the cubic Fitzpatrick 2019 law and -219 its
    #: linear form, 223 GORD23, 225 SOMM25 and 226 SOMM26. Unset, a
    #: simulation is fitted with the law it was made with, read from its
    #: header, and real data with 99.
    opt_mwcolorlaw: Optional[int] = None
    #: SNANA RV_MWCOLORLAW; Milky Way total-to-selective extinction ratio.
    #: Defaults to the value in the simulation header for simulated data,
    #: or 3.1 for observations.
    rv_mwcolorlaw: Optional[float] = None
    #: SNANA PARLIST_MWCOLORLAW; the free parameters of the parametrised laws,
    #: Goobar 08 (P, A), Gordon 16 (RVA, FA) and Sommovigo 26 (B0, B1, B2,
    #: B3). Empty leaves each law with its own defaults.
    parlist_mwcolorlaw: Tuple[float, ...] = ()
    #: SNANA MWEBV_SCALE; scale factor applied to E(B-V) and its uncertainty.
    mwebv_scale: float = 1.0
    #: SNANA MWEBV_SHIFT; additive shift in E(B-V), with no change to its
    #: uncertainty.
    mwebv_shift: float = 0.0
    #: SNANA MWEBV_FORCE; give every event this E(B-V), ignoring the header and
    #: the map. None, the default, leaves it off; a namelist value turns it on
    #: only above -0.001, as SNANA applies it (snana.F90:17209).
    mwebv_force: Optional[float] = None

    # Input redshifts
    #: SNANA REDSHIFT_FINAL_SHIFT; additive shift applied after selecting
    #: each event's fit redshift.
    redshift_final_shift: float = 0.0
    #: SNANA HOSTGAL_ZPHOT_SHIFT; shift the host photometric redshift. It
    #: affects the fit only where the event is fitted at that redshift, to
    #: within a rounding tolerance.
    hostgal_zphot_shift: float = 0.0
    #: SNANA HOSTGAL_ZSPEC_SHIFT; shift the host spectroscopic redshift. It
    #: affects the fit only where the event is fitted at that redshift.
    hostgal_zspec_shift: float = 0.0
    #: SNANA VPEC_ERR_OVERRIDE; replace the peculiar-velocity error in the data
    #: with this value, in km/s.
    vpec_err_override: float = 0.0
    #: SNANA OPT_VPEC_COR; 0 zeroes the peculiar velocity and its error, so no
    #: peculiar-velocity correction is made to the Hubble-diagram redshift.
    opt_vpec_cor: int = 1
    #: SNANA ZTOL_HELIO2CMB; maximum permitted difference between the CMB-frame
    #: redshift in the data and the value recomputed from the heliocentric
    #: redshift.
    ztol_helio2cmb: float = 0.02
    #: SNANA USE_SNHOST_ZPHOT; fit at the host photometric redshift rather than
    #: the event's own.
    use_snhost_zphot: bool = False
    #: SNANA USE_SNHOST_QZPHOT; fit at the mean of the host redshift's
    #: quantiles rather than the event's own redshift, where the data supply
    #: them.
    use_snhost_qzphot: bool = False
    #: SNANA SNMAGRMS_FOR_PHOTOZ; the intrinsic magnitude dispersion in
    #: magnitudes, converted to a redshift error through the derivative of
    #: redshift with respect to distance modulus and added in quadrature to the
    #: denominator of the reported redshift pull.
    snmagrms_for_photoz: float = 0.12
    #: SNANA RESTORE_WRONG_VPEC; keep the peculiar-velocity sign as written
    #: in the file, for data written before SNANA v11.02, whose sign is
    #: otherwise flipped on reading.
    restore_wrong_vpec: bool = False

    # Peak-time estimation
    #: SNANA OPT_SETPKMJD; initial peak-time estimator,
    #: as a bit mask. Zero, the default, uses the peak time in the data header;
    #: 1 fits a Bazin 09 form to each band; 16 takes the clump of high
    #: signal-to-noise epochs about the brightest one; 64 takes the MJD of the
    #: trigger; 4 lets the run continue when an event gives no estimate.
    opt_setpkmjd: int = 0
    #: SNANA MJDWIN_SETPKMJD; the width in days of the window the clump of
    #: epochs is collected in under OPT_SETPKMJD = 16.
    mjdwin_setpkmjd: float = 60.0
    #: SNANA SNRCUT_SETPKMJD; the signal-to-noise an epoch must reach to join
    #: the clump under OPT_SETPKMJD = 16.
    snrcut_setpkmjd: float = 5.0
    #: SNANA SHIFT_SETPKMJD; shift in days applied to the initial peak-MJD
    #: estimate.
    shift_setpkmjd: float = 0.0
    #: SNANA PEAKMJD_SANITY; allowed MJD range of the fitted peak. If unset,
    #: CUTWIN_PEAKMJD is widened by 20 days on either side.
    peakmjd_sanity: Tuple[float, float] = (0.0, 1.0e6)

    def has_setpkmjd_bazin(self) -> bool:
        return bool(self.opt_setpkmjd & OPTMASK_SETPKMJD_BAZIN)

    def has_setpkmjd_fmax_clump(self) -> bool:
        # FLUXMAX2 or FLUXMAX3, the latter only weighting by log10(SNR)
        return bool(self.opt_setpkmjd & (OPTMASK_SETPKMJD_FLUXMAX2 | OPTMASK_SETPKMJD_FLUXMAX3))

    def has_setpkmjd_fluxmax(self) -> bool:
        return bool(self.opt_setpkmjd & OPTMASK_SETPKMJD_FLUXMAX)

    def has_setpkmjd_trigger(self) -> bool:
        return bool(self.opt_setpkmjd & OPTMASK_SETPKMJD_TRIGGER)


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


@dataclass
class Selection:
    """Event and epoch selection, including post-fit cuts."""

    # Event identifiers and exclusion lists
    #: SNANA SNTYPE_LIST; fit only the events whose SNTYPE appears here.
    sntype_list: Optional[List[int]] = None
    #: SNANA SNCID_LIST; fit only these CIDs, written as integers.
    sncid_list: Optional[List[int]] = None
    #: SNANA SNCCID_LIST; fit only these events, written as strings. Each
    #: name matches a CID, a NAME_TRANSIENT or a NAME_IAUC.
    snccid_list: Optional[List[str]] = None
    #: SNANA SNCID_LIST_FILE; read the list of CIDs to fit from a file.
    sncid_list_file: Optional[str] = None
    #: SNANA OPT_SNCID_LIST; event-list selection and initialisation bit
    #: mask: 1 skips every per-event cut but the CID one, 2 seeds the starting
    #: values from the list file. The value 4 is refused. A positive
    #: FLAG_USE_SAME_EVENTS replaces this with its own value.
    opt_sncid_list: int = 0
    #: SNANA SNCID_IGNORE; drop these CIDs. The list ends at its first entry
    #: below 1, and later entries are ignored.
    sncid_ignore: Optional[List[int]] = None
    #: SNANA SNCCID_IGNORE; drop these CIDs, written as strings.
    snccid_ignore: Optional[List[str]] = None
    #: SNANA SNCID_IGNORE_FILE; read the CIDs to drop from a file.
    sncid_ignore_file: Optional[str] = None
    #: SNANA SNTYPE_IGNORE; drop the events of these types, the list ending at
    #: its first negative entry.
    sntype_ignore: Optional[List[int]] = None
    #: SNANA SIM_TEMPLATE_INDEX_LIST; accepted for compatibility; does not
    #: filter events.
    sim_template_index_list: Optional[List[int]] = None
    #: SNANA SNFIELD_LIST; fit only the events observed in these survey fields.
    snfield_list: Optional[List[str]] = None
    #: SNANA USESIM_SNIA; fit the simulated type Ia events. False fits only
    #: the non-Ia events.
    usesim_snia: bool = True
    #: SNANA USESIM_NONIA; fit the simulated non-Ia events. False fits only
    #: the type Ia events.
    usesim_nonia: bool = True
    #: SNANA ABORT_ON_DUPLCID; abort on a repeated CID. False keeps the first
    #: occurrence and rejects the rest.
    abort_on_duplcid: bool = True

    # Sample size and subsampling
    #: SNANA SIM_PRESCALE; keep one event in this many, chosen by index so that
    #: the same subset is selected every run.
    sim_prescale: float = 1.0
    #: SNANA MAX_FIT; stop once this many events have been fitted.
    max_fit: Optional[int] = None
    #: SNANA JOBSPLIT; written 'IJOB NJOBTOT', it processes every NJOBTOT-th
    #: event starting from IJOB, to divide one sample between batch jobs
    #: without splitting the data.
    jobsplit: Tuple[int, int] = (1, 1)
    #: SNANA MXEVT_PROCESS; maximum number of events processed after the header
    #: cuts on CID and type, before the remaining event cuts and fitting.
    mxevt_process: int = 999888777
    #: SNANA MXEVT_CUTS; maximum number of events retained after event-level
    #: selection (N_SNLC_CUTS). Selection is evaluated for every event read.
    mxevt_cuts: int = 999888777
    #: SNANA MXLC_FIT; write at most this many rows. It counts the events whose
    #: final iteration succeeded, and limits only the output, not the reading
    #: or the fitting.
    mxlc_fit: int = 999888777

    # Event selection
    #: SNANA CUTWIN_<QUANTITY>, SNCUT_SNRMAX, SNCUT_HOST_SBFLUX; the windows on
    #: whole events, one key per quantity: REDSHIFT, REDSHIFT_ERR, MWEBV,
    #: SNRMAX, SNRMAX2 and their per-band forms, NFILT_SNRMAX and NFILT_SNRMAX2
    #: (the number of bands clearing the matching window), the phase coverage
    #: (TRESTMIN, TRESTMAX, TGAPMAX, T0GAPMAX), PEAKMJD, NEPOCH, SEARCHEFF_MASK
    #: and the host surface brightness SBFLUX. A window written with one edge
    #: leaves the other open. ``SNCUT_SNRMAX`` and
    #: ``SNCUT_HOST_SBFLUX``, written as filter strings, set a per-band lower
    #: edge on the SNRMAX and SBFLUX windows. NEPOCH counts every stored epoch
    #: with a non-negative flux error, in all the survey's bands, before any
    #: epoch cut removes one. Quantity names are case-insensitive.
    cutwin: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    #: SNANA SIMVAR_CUTWIN_STRING; 'NAME LO HI' windows on the simulated-truth
    #: columns.
    simvar_cutwin: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    #: SNANA PRIVATE_CUTWIN_STRING; 'VARNAME CUTMIN CUTMAX' windows on the
    #: PRIVATE_* header columns that PRIVATE_VARNAME_READLIST asked for.
    private_cutwin: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    #: SNANA PRIVATE_CUTWIN_STRING; written 'VARNAME != VALUE' instead, it
    #: rejects an event when that column equals the value.
    private_veto: Dict[str, float] = field(default_factory=dict)
    #: SNANA SNCUT_NOBS_TREST; (NOBS, TMIN, TMAX) triplets, each requiring at
    #: least NOBS epochs inside that rest-frame phase range.
    sncut_nobs_trest: Tuple[Tuple[int, float, float], ...] = ()
    #: SNANA REQUIRE_EPOCHS_STRING; limit the duration between first and last
    #: detections above SNRMIN and require baseline observations before and
    #: after them. TOBS_RANGES or TREST_RANGES gives three durations in days:
    #: the preceding baseline window, maximum detection span and following
    #: baseline window. Python/YAML takes, for example,
    #: {tranges: [10, 50, 20], snrmin: 5, rest_frame: true}.
    require_epochs: Optional[Dict[str, Any]] = None
    #: SNANA EARLYLC_STRING; restrict the fit to the first MAXOBS observations
    #: or MAXNIGHT nights passing the specified signal-to-noise and flag cuts,
    #: for early-time analysis. Python/YAML takes a mapping such as
    #: {maxobs: 5, snrmin: 3.0}; other fields are maxnight, ndayadd, filters,
    #: photprobmin, photmask, photmask_start and snr_start.
    earlylc: Optional[Dict[str, Any]] = None
    #: SNANA SNRMAX_FILTERS; the bands SNRMAX, SNRMAX2 and the NFILT_SNRMAX
    #: counts are computed over, as one string of band characters. Blank, the
    #: default, uses every band.
    snrmax_filters: Union[str, List[str]] = ''

    # Epoch selection
    #: SNANA CUTWIN_<EPOCH_QUANTITY>; the ``CUTWIN_`` keys that mask single
    #: epochs, one key per quantity: MJD, MJD_EXCLUDE, TOBS, TREST, LAMOBS,
    #: LAMREST, PSF, ZPADU, ZPNPE, ZPERR, PHOTPROB, ERRTEST and SNR_NODETECT.
    #: ZPADU and ZPNPE cut the same zero point in different units, so only one
    #: of the two may be given. Quantity names are case-insensitive.
    epoch_cutwin: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    #: SNANA EPCUT_SNRMIN; a per-band signal-to-noise an epoch must reach
    #: before it is fitted.
    epcut_snrmin: Optional[Dict[str, float]] = None
    #: SNANA EPOCH_IGNORE_FILE; a file naming single epochs to drop by CID, MJD
    #: and band. DEFAULT reads the version's own .IGNORE file beside the data
    #: where one exists, and NONE or NULL reads nothing.
    epoch_ignore_file: Optional[str] = 'DEFAULT'
    #: SNANA PHOTFLAG_MSKREJ, PHOTFLAG_BITLIST_REJECT; PHOTFLAG bits that
    #: reject an epoch. The first value is a mask; the bit-list spelling names
    #: the bits individually and replaces that first value, leaving any later
    #: entries in place.
    photflag_mskrej: Tuple[int, ...] = ()

    # Post-fit selection
    #: SNANA FITWIN_<COLUMN>; an inclusive window on the fitted result. The
    #: suffix is one of a closed list: a fitted parameter under its FITRES
    #: name, its ``_ERR`` or ``_ERRz1`` form (``x1_ERR``, not ``x1ERR``),
    #: ``PROB``, ``ZPULL`` or ``COV_OFFDIAG``; the generic ``SHAPE``,
    #: ``COLOR``, ``DLMAG`` and ``PEAKMJD`` are resolved as for ``INIVAL_``.
    #: ``COV_OFFDIAG`` and ``mB_ERR`` are always applied.
    fitwin: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class Model:
    """Source model, initial parameter values, bounds and priors, the axes
    of the photo-z coarse grid, and the model-error settings."""

    input_flags = {'priors': 'prior'}
    # Source model
    #: SNANA FITMODEL_NAME; the path of the model to fit. Unless MODEL_TYPE
    #: is set, the model is chosen from the directory name. Required.
    fitmodel_name: Optional[str] = None
    #: SNANA MODEL_TYPE; the model class, for a directory whose name does not
    #: identify the model.
    model_type: Optional[str] = None
    # Set by the keys the model declares
    #: The values of the input keys the selected model class declares as its
    #: own, typed by the parser and set on the instance by its constructor.
    _model_key_values: Dict[str, Any] = field(default_factory=dict)

    # Initial values and fixed parameters
    #: SNANA INIVAL_<PARAM>; the value a fitted parameter starts at. The suffix
    #: names one of the model's own parameters, or a shared name such
    #: as SHAPE, COLOR or PEAKMJD, resolved once the model is known.
    inival: Dict[str, float] = field(default_factory=dict)
    #: Physical starting values, keyed by fitted parameter name.
    initial: Dict[str, float] = field(default_factory=dict)
    #: Fixed physical values; null releases a model-default held parameter.
    fixed: Dict[str, Optional[float]] = field(default_factory=dict)
    #: Physical lower and upper bounds within the model's valid range.
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    #: SNANA INISTP_<PARAM>; the initial step for a fitted parameter, the
    #: suffix resolved as an ``INIVAL_`` suffix is. Zero holds the parameter
    #: at its starting value.
    inistp: Dict[str, float] = field(default_factory=dict)
    #: SNANA INIVAL_GRIDSEARCH_<PARAM>; 'LO HI STEP' for the pre-fit grid
    #: search over that parameter, used to find a starting point when the
    #: chi2 has more than one minimum.
    inival_gridsearch: Dict[str, tuple] = field(default_factory=dict)
    #: SNANA LFIXPAR_ALL; hold every parameter, including redshift in a
    #: photo-z fit, at its initial value. Skip the grid search and the pre-fit
    #: adjustment, evaluating the model without fitting it.
    lfixpar_all: bool = False
    #: SNANA LFIXPAR_ADJUST; hold every parameter as LFIXPAR_ALL does, except
    #: that the pre-fit adjustment runs first and the parameters are then held
    #: at its result.
    lfixpar_adjust: bool = False
    #: SNANA USESIM_<PARAM>; start the named parameter at its simulated truth
    #: and hold it there. PEAKMJD, SHAPE, LUMIPAR, COLOR, AV, RV, DLMAG and
    #: REDSHIFT are the spellings accepted, each resolved against the model as
    #: the ``INIVAL_`` suffixes are. Python/YAML uses a list, such as
    #: [SHAPE, COLOR].
    usesim_params: List[str] = field(default_factory=list)
    #: SNANA USESIM_INIVAL; start each parameter with a recorded simulated value
    #: at that value while leaving all parameters free.
    usesim_inival: bool = False

    # Priors
    #: SNANA PRIOR_<PARAM>_RANGE, PRIOR_<PARAM>_SIGMA, PRIOR_<PARAM>_EXP,
    #: PRIOR_<PARAM>_WGT, PRIOR_<PARAM>_RES; the prior on a fitted parameter.
    #: RANGE and SIGMA give a flat interval with a quartic penalty outside
    #: its limits, (excess/SIGMA)^4; EXP, WGT and RES give a weighted sum of
    #: exponentials convolved with a Gaussian for a host-extinction prior. These
    #: replace the model's default prior for that parameter.
    priors: Dict[str, Any] = field(default_factory=dict)
    #: SNANA RANGEC_COURSE_PHOTOZ, NBINC_COURSE_PHOTOZ, RANGES_COURSE_PHOTOZ,
    #: NBINS_COURSE_PHOTOZ; the colour and shape axes of the photo-z coarse
    #: grid, a range and a bin count each, replacing the model's defaults.
    #: Python/YAML uses COLOR and SHAPE mappings, for example
    #: {COLOR: {range: [-0.3, 0.5], nbin: 5}}.
    coarse_grid: Dict[str, dict] = field(default_factory=dict)
    #: SNANA OPT_PRIOR; 0 disables every prior, the redshift prior of a photo-z
    #: fit included.
    opt_prior: int = 1
    #: SNANA OPT_PRIOR_AV; 0 drops the input's own host-extinction prior keys,
    #: leaving the parameter with the model's default prior.
    opt_prior_av: int = 1

    # Model uncertainty
    #: SNANA USE_MODEL_MAGERR, SCALE_COVAR; USE_MODEL_MAGERR = F zeroes the
    #: model magnitude error and switches the off-diagonal covariance off with
    #: it. SCALE_COVAR, a number, does the same at or below zero and nothing
    #: above it.
    use_model_magerr: bool = True
    #: SNANA FUDGE_MODELERR_SCALE; scale factor applied to the model magnitude
    #: uncertainty.
    fudge_modelerr_scale: float = 1.0
    #: SNANA FUDGE_MAGERR_MODEL; replace the model magnitude error with this
    #: constant in magnitudes, converted to a fractional flux error as
    #: 1 - 10^(-0.4 mag). Negative, the default, leaves the model's own
    #: error in place, and zero removes it.
    fudge_magerr_model: float = -9.0


@dataclass
class Fit:
    """Fit configuration, numerical settings and reference cosmology."""

    # Original and resolved values for settings changed by linked rules.
    _derived_settings: Dict[str, Tuple[Any, Any]] = field(default_factory=dict, repr=False)

    # Filters and phase range
    #: SNANA FILTLIST_FIT; the bands to fit, given as a list of full calibration
    #: names in YAML or Python, or as a string of single-character filter labels.
    filtlist_fit: Optional[Union[str, List[str]]] = None
    #: SNANA RESTLAMBDA_FITRANGE; allowed rest-frame range in Angstroms
    #: for filter mean wavelengths: the observer-frame mean divided by
    #: (1+z) must lie within it. Intersected with the model's filter-centre
    #: range before selecting observations.
    restlambda_fitrange: Tuple[float, float] = (0.0, 25000.0)

    # Windows applied during the fit
    #: SNANA FITWIN_SHAPE_ITER1, FITWIN_COLOR_ITER1; windows on the shape and
    #: the colour after the first round. An event outside them is dropped
    #: after round 1 and left out of the output.
    fitwin_iter1: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    #: SNANA FITWIN_PROB_ITER1; a window on the fit probability after round 1.
    fitwin_prob_iter1: Optional[Tuple[float, float]] = None
    #: SNANA FITWIN_CHI2RED_INI; a window on chi2 per epoch at the initial
    #: parameter values, tested before the starting values are adjusted.
    fitwin_chi2red_ini: Optional[Tuple[float, float]] = None
    #: SNANA FITWIN_CHI2RED_INI2; a window on chi2 per epoch at the starting
    #: values, tested once the pre-fit adjustment has run and before the fit
    #: itself begins.
    fitwin_chi2red_ini2: Optional[Tuple[float, float]] = None
    #: SNANA FITWIN_TREST; the rest-frame phase range in days an epoch must
    #: fall in, applied after round 1 against the time of peak the fit has just
    #: found.
    fitwin_trest: Tuple[float, float] = (-999.0, 999.0)
    #: SNANA FITWIN_TREST_FILTER; 'gr -10 50 iz 0 50' triplets replacing the
    #: FITWIN_TREST phase range for the bands each triplet names.
    fitwin_trest_filter: Dict[str, Tuple[float, float]] = field(
        default_factory=dict)
    #: SNANA FITWIN_MJD; the observer-frame MJD range an epoch must fall in,
    #: applied after round 1 as FITWIN_TREST is.
    fitwin_mjd: Tuple[float, float] = (20000.0, 90000.0)   # snlc_fit.F90:6784

    # Running the fit
    #: Profile redshift ('z') or every free parameter ('all') after the final
    #: round. None retains Hessian-only errors; unprofiled parameters also
    #: retain Hessian errors.
    minos: Optional[str] = None
    #: SNANA BATCH_SIZE; number of events per batch for fitting and initial
    #: peak-MJD estimation.
    batch_size: int = 1000
    #: SNANA NUM_NOBS_BINS; group the sample into this many bins by epoch
    #: count, each padded to its own longest light curve and compiled
    #: separately, which saves padded work when light-curve lengths vary
    #: widely. 1, the default, pads the whole sample to its longest light
    #: curve.
    num_nobs_bins: int = 1
    #: SNANA MIN_BIN_GAIN; a NUM_NOBS_BINS bin is kept only where it cuts
    #: the padded work by at least this fraction.
    min_bin_gain: float = 0.05

    # Input validation
    #: SNANA REQUIRE_DOCANA; check that the data version's README and the
    #: model's own metadata files open with the DOCUMENTATION key, as a bit
    #: mask. Bit 1 aborts on a file which does not, a non-zero value without
    #: that bit reduces the abort to a warning, and zero, the default, skips
    #: the check.
    require_docana: int = 0

    # Precision
    #: SNANA DOUBLE_PRECISION; work in 64-bit floats instead of 32-bit.
    double_precision: Optional[bool] = None

    # Aborts and rejected events
    #: SNANA ABORT_ON_BADQZPHOT; abort where a host redshift's quantiles are
    #: not monotonic; false downgrades that to a warning.
    abort_on_badqzphot: bool = True
    #: SNANA ABORT_ON_MISSING_QZPHOT; abort where a quantile photo-z fit meets
    #: an event with no quantiles; false skips the event.
    abort_on_missing_qzphot: bool = True
    #: SNANA ABORT_ON_TRESTCUT; abort where a photo-z fit that floats the
    #: redshift with a unit step is given a rest-frame cut on the input
    #: redshift, which is not a meaningful cut.
    abort_on_trestcut: bool = True
    #: SNANA ABORT_ON_NOPKMJD; stop if the Bazin pre-fit cannot estimate the
    #: peak time for an event. False allows the run to continue, as does
    #: OPT_SETPKMJD bit 4.
    abort_on_nopkmjd: bool = True
    #: SNANA ABORT_ON_NOEPOCHS; abort on an event with no epoch left to fit.
    #: Off by default, in which case such events are recorded as unfittable and
    #: the run continues.
    abort_on_noepochs: bool = False
    #: SNANA CHECK_CRAZYFITERR; drop an event whose fitted errors are NaN or
    #: outside SNANA's fixed ranges. On by default.
    check_crazyfiterr: bool = True
    #: SNANA MINOS_ERRMAX_BAD; treat a fitted parameter's error as unusable,
    #: and flag the event, where the error is below that parameter's floor or
    #: the value did not move from its starting value.
    minos_errmax_bad: bool = True

    # Numerical controls
    #: Minimum total per-epoch flux uncertainty used in the fit.
    lm_sigma_floor: float = 1e-10

    # Minimisation
    #: SNANA NFIT_ITERATION; how many outer rounds the fit runs. Round 1
    #: recomputes the model error at every step, and each later round refits
    #: with the covariance fixed at the previous round's model. Epochs are
    #: rejected between rounds.
    n_fit_iterations: int = 3
    #: SNANA MAXITER_ROUND1; the optimiser iterations allowed in round 1.
    maxiter_round1: int = 25
    #: SNANA MAXITER_ROUND2; the iterations allowed in each later round.
    maxiter_round2: int = 15
    # Initial search
    #: Skip the initial peak-time grid search. Set by LFIXPAR_ALL;
    #: initialising from simulated truth skips the search as well.
    skip_grid_search: bool = False

    # Fit statistic and flux covariance
    #: SNANA OPT_CHI2_SIGMA, DOCHI2_SIGMA; 1, or DOCHI2_SIGMA = T, adds the
    #: logarithmic sigma term to the chi2, so the fit accounts for the model
    #: error changing with the parameters. On by default for a photo-z fit. The
    #: other declared bits are refused.
    opt_chi2_sigma: int = 0
    #: SNANA OPT_COVAR_FLUX; 1 includes the model's off-diagonal epoch
    #: covariance in rounds 2 and later, and 0 fits on a diagonal error alone.
    #: On by default.
    use_offdiag_covariance: bool = True
    #: SNANA FUDGE_MAG_COVERR; a per-band magnitude error fully correlated
    #: across every epoch in that band, which enters as an off-diagonal
    #: covariance term rather than as extra error on each epoch.
    fudge_mag_coverr: Dict[str, float] = field(default_factory=dict)
    #: SNANA OPT_COVAR_MWXTERR; how the Milky Way extinction error enters the
    #: fit covariance: 1 the full rank-one term correlating every epoch of
    #: every band, 2 its diagonal alone, 0 not at all. The term comes from each
    #: event's E(B-V) uncertainty, so it vanishes under OPT_MWEBV = 0 and under
    #: MWEBV_FORCE. Both fixed-redshift and photometric-redshift fits support
    #: these settings.
    opt_covar_mwxterr: int = 1
    #: SNANA OPT_COVAR_LCFIT; any non-zero value repairs the reported
    #: covariance of amplitude, shape and colour through its eigenvalues where
    #: it is not positive definite, and bit 2 rebuilds it for every event.
    opt_covar_lcfit: int = 0
    #: SNANA SIMFIT_IDEAL_PRESCALE; refit one event in this many on the
    #: noise-free simulated fluxes and write the result as the _IDEAL columns,
    #: which separates fitting error from measurement noise.
    simfit_ideal_prescale: int = 0
    # Not set by an input key
    #: True inside that second, noise-free pass alone. Because the pass starts
    #: from simulated truth, the pre-fit adjustment leaves the initial values
    #: as they are.
    _dofit_ideal: bool = False

    # Uncertainty adjustments
    #: SNANA FUDGEALL_ITER1_MAXFRAC; add this fraction of the event's peak
    #: measured flux, over every band, in quadrature to every epoch's error,
    #: in round 1 alone.
    fudgeall_iter1_maxfrac: float = 0.0
    #: SNANA FUDGEALL_MAXFRAC; add this fraction of the event's peak measured
    #: flux, over every band, in quadrature to every epoch's error, in every
    #: round.
    fudgeall_maxfrac: float = 0.0
    #: SNANA FUDGE_FITERR_MAXFRAC; add this fraction of the event's peak
    #: measured flux in quadrature to the error of the epochs the
    #: ``FUDGE_FITERR_`` phase, wavelength, band and iteration windows select.
    #: Zero, the default, adds nothing.
    fudge_fiterr_maxfrac: float = 0.0
    #: SNANA FUDGE_FITERR_MAXITER; the last optimiser iteration the
    #: FUDGE_FITERR_MAXFRAC term is applied on.
    fudge_fiterr_maxiter: int = FUDGE_FITERR_MAXITER_DEFAULT
    #: SNANA FUDGE_FITERR_TREST; four rest-frame phases in days, read as two
    #: windows, an epoch taking the FUDGE_FITERR_MAXFRAC term when it falls
    #: inside either. The default places both windows at a phase no epoch
    #: reaches.
    fudge_fiterr_trest: Tuple[float, float, float, float] = FUDGE_FITERR_TREST_DEFAULT
    #: SNANA FUDGE_FITERR_RESTLAM; four rest-frame wavelengths in Angstroms,
    #: read as two windows on the same terms as FUDGE_FITERR_TREST.
    fudge_fiterr_restlam: Tuple[float, float, float, float] = FUDGE_FITERR_RESTLAM_DEFAULT
    #: SNANA FUDGE_FITERR_PASSBANDS; the bands the FUDGE_FITERR_MAXFRAC term
    #: acts on, as band characters. NULL, the default, selects none.
    fudge_fiterr_passbands: Union[str, List[str]] = "NULL"
    #: SNANA DELCHI2_REJECT; reject an epoch between rounds where it
    #: contributes more than this to the chi2. The default is high enough that
    #: nothing is rejected.
    delchi2_reject: float = 1e9
    #: SNANA SIGNOISE_REJECT; reject an epoch between rounds where its
    #: signal-to-noise falls below this. Negative, the default, rejects
    #: nothing.
    signoise_reject: float = -9.0

    # Reference cosmology
    #: SNANA H0_REF; the Hubble constant in km/s/Mpc, and optionally its
    #: uncertainty, that the reference distance modulus is computed at. Its
    #: uncertainty does not affect the fit.
    h0_ref: Tuple[float, float] = (70.0, 7.0)
    #: SNANA OMAT_REF; the matter density today, and optionally its
    #: uncertainty, which with those of W0_REF and WA_REF sets the width of
    #: the PRIOR_MUERRSCALE distance prior.
    omat_ref: Tuple[float, float] = (0.315, 0.03)
    #: SNANA OLAM_REF; the dark-energy density today, and optionally its
    #: uncertainty.
    olam_ref: Tuple[float, float] = (0.685, 0.03)
    #: SNANA W0_REF; the dark-energy equation of state today, and optionally
    #: its uncertainty.
    w0_ref: Tuple[float, float] = (-1.0, 0.1)
    #: SNANA WA_REF; the evolution of the dark-energy equation of state, as
    #: w(a) = w0 + wa (1 - a), and optionally its uncertainty.
    wa_ref: Tuple[float, float] = (0.0, 0.0)

    def distance_modulus(self, redshifts):
        """Reference distance modulus for each redshift.

        At the REF cosmology; models are handed the result so none of them
        carries a cosmology.
        """
        return reference_distance_modulus(
            redshifts, self.h0_ref[0], self.omat_ref[0], self.olam_ref[0],
            self.w0_ref[0], self.wa_ref[0])

    def epoch_rejection_active(self) -> bool:
        return (self.delchi2_reject != Fit.delchi2_reject
                or self.signoise_reject != Fit.signoise_reject
                or tuple(self.fitwin_mjd) != Fit.fitwin_mjd)


@dataclass
class PhotoZ:
    """Photometric-redshift fitting (snlc_fit.F90 DOFIT_PHOTOZ, §8.16)."""

    # Redshift priors and bounds
    #: SNANA DOFIT_PHOTOZ; fit the redshift as a free parameter alongside the
    #: model's own. Setting it with no OPT_PHOTOZ implies OPT_PHOTOZ = 1, and
    #: any positive OPT_PHOTOZ implies this in turn.
    dofit_photoz: bool = False
    #: SNANA OPT_PHOTOZ; what constrains the fitted redshift, as a bit mask. 1
    #: is a Gaussian on the host photometric redshift, and 2 a Gaussian on the
    #: redshift the data file marks as best. 4 reads each event's host
    #: quantiles as a probability distribution, the derivative of a natural
    #: cubic spline through them. 8 uses Steffen's monotone cubic instead,
    #: whose distribution cannot dip below zero, and implies 4, so it is
    #: written 12. The host Gaussian is applied only when PRIOR_ZERRSCALE is
    #: below 30, so set PRIOR_ZERRSCALE = 1 to enable it. Events without a
    #: usable host photometric redshift start at the middle of PHOTOZ_BOUND
    #: with an effectively flat prior.
    opt_photoz: int = 0
    #: SNANA PRIOR_ZERRSCALE; multiply the width of the Gaussian redshift
    #: prior and the zPRIORERR column. The Gaussian includes 0.001 added in quadrature to
    #: the quoted error before scaling, and remains active at small errors.
    #: The default means unset. With bit 1 of OPT_PHOTOZ the
    #: default turns the host Gaussian off, since that prior is applied only
    #: below 30, so set this to 1 for a fit on host redshifts. With bit 2 an
    #: unset value becomes 1, so the data file's best redshift is used at its
    #: own width.
    prior_zerrscale: float = PRIOR_ZERRSCALE_UNSET
    #: SNANA PHOTOZ_BOUND; lower and upper bounds on the fitted redshift.
    photoz_bound: Tuple[float, float] = (1e-6, 1.4)

    # Distance priors
    #: SNANA PRIOR_MUERRSCALE; a Gaussian prior tying the fitted distance to
    #: the reference cosmology's at the fitted redshift. Its width is how far
    #: the reference distance moves under the OMAT_REF, W0_REF and WA_REF
    #: uncertainties, multiplied by this scale. None applies no such prior.
    prior_muerrscale: Optional[float] = None
    #: SNANA PRIOR_DMU_RANGE; a flat prior on the fitted distance less the
    #: reference cosmology's at the fitted redshift, within this range. None
    #: applies no such prior.
    prior_dmu_range: Optional[Tuple[float, float]] = None
    #: SNANA PRIOR_DMU_SIGMA; the width of the Gaussian roll-off outside
    #: PRIOR_DMU_RANGE.
    prior_dmu_sigma: float = 0.1
    #: SNANA SKIP_PHOTOZ_INIT; skip the coarse redshift grid for the events
    #: with no usable host redshift, starting those at the middle of
    #: PHOTOZ_BOUND instead.
    skip_photoz_init: bool = False
    #: SNANA QUANTILE_ZERRMIN; a redshift error below this counts the event as
    #: spectroscopic, which where bit 2 of OPT_PHOTOZ is also set gives it the
    #: Gaussian prior in place of its quantiles. Zero, the default, marks no
    #: event.
    quantile_zerrmin: float = 0.0
    # Numerical controls
    #: Search the whole of PHOTOZ_BOUND on the coarse grid rather than the
    #: window the prior implies.
    photoz_grid_fullrange: bool = False
    #: Number of redshift grid points used to interpolate the filter integration
    #: weights during photometric-redshift fitting.
    n_zgrid: int = 1000
    #: Number of events per batch in the coarse grid search.
    photoz_grid_batch: int = 100

    # Coarse redshift grid
    #: SNANA NBINZ_COURSE_PHOTOZ; bins on the coarse grid's redshift axis,
    #: which is spaced in z/(1+z). A positive value overrides the default of 42
    #: bins per unit of z/(1+z) across the prior window.
    nbinz_course: int = -9
    #: SNANA RANGEZ_COURSE_PHOTOZ; the redshift range of the coarse grid's
    #: redshift axis. A positive lower edge overrides the default window,
    #: two prior widths either side of each event's prior centre.
    rangez_course: Tuple[float, float] = (-99.0, -99.0)
    #: SNANA NBINT_COURSE_PHOTOZ; bins on a coarse-grid PEAKMJD axis around
    #: each event's PEAKMJD estimate. The default 1 holds PEAKMJD at the
    #: estimate.
    nbint_course: int = 1
    #: SNANA RANGET_COURSE_PHOTOZ; the PEAKMJD axis's range, in days relative to
    #: the estimate, used when NBINT_COURSE_PHOTOZ is above 1.
    ranget_course: Tuple[float, float] = (-15.0, 15.0)
    #: SNANA SCALE_NBIN_COURSE_PHOTOZ; multiply the bin count of every
    #: coarse-grid axis. It must be positive.
    scale_nbin_course: int = 1
    # Filter and epoch selection for photo-z fits
    #: SNANA PHOTODZ_REJECT; drop a band if the fitted redshift is within this
    #: margin of either end of the band's usable redshift range. When using a
    #: Gaussian prior on the best available redshift, its uncertainty replaces
    #: this margin if it is below 0.05. This replacement does not apply with
    #: LFIXPAR_ALL.
    photodz_reject: float = 0.05
    #: SNANA PHOTODZ1Z_REJECT; the PHOTODZ_REJECT margin written as dz/(1+z)
    #: instead. Negative, the default, applies no such cut.
    photodz1z_reject: float = -99.0
    #: SNANA PHOTOZ_ITER1_LAMRANGE; the observer-frame wavelength range in
    #: Angstroms a band must fall in to be fitted in round 1.
    photoz_iter1_lamrange: Tuple[float, float] = (4000.0, 26000.0)

    # OPT_PHOTOZ bits (snlc_fit.F90:121)
    def has_photoz_host_gaussian(self):
        return bool(self.opt_photoz & 1)

    def has_photoz_bestz_gaussian(self):
        return bool(self.opt_photoz & 2)

    def has_photoz_quantiles(self):
        return bool(self.opt_photoz & 4)

    def quantile_spline(self):
        """The CDF interpolant the quantile prior's PDF is the derivative of
        (METHOD_SPLINE_QUANTILES, snlc_fit.F90:11677-11680)."""
        return 'steffen' if self.opt_photoz & 8 else 'cubic'


@dataclass
class Output:
    """Output tables, formats and paths."""

    # Output tables
    #: SNANA SNTABLE_LIST; mapping from table names to layouts, e.g.
    #: ``{snana: key, fitres: key}``. Tables are SNANA (events read), FITRES
    #: (fitted events), OUTLIER (discrepant epochs) and LCPLOT (photometry
    #: and model curves). Layouts are key, csv or col. In SNANA inputs, use
    #: a space-separated string such as ``SNANA(text:key) FITRES(text:csv)``.
    tables: Dict[str, str] = field(default_factory=lambda: {'snana': 'key', 'fitres': 'key'})
    #: SNANA SNTABLE_LIST; include the first host's columns in every table;
    #: selected with ``text:host`` in the input string.
    host_text: bool = False
    #: SNANA SNTABLE_LIST; include the second host's columns as well;
    #: selected with ``text:host2`` in the input string.
    host2_text: bool = False
    #: SNANA SNTABLE_LIST; omit simulated-truth columns (``nosimvar``).
    nosimvar: bool = False
    #: SNANA SNTABLE_LIST; accept ``nozphot`` without changing the output.
    nozphot: bool = False
    #: SNANA SNTABLE_LIST; include each event's IAU name beside its CID,
    #: or UNKNOWN when absent (``iauc``).
    iauc: bool = False
    #: SNANA SNTABLE_LIST; select SNANA table rows: 63 includes every row,
    #: 0 only events passing every cut, and other values include events
    #: failing a cut named by the mask. Written as ``SNANA(cutmask:N)``.
    cutmask: int = 63
    #: SNANA MXLC_PLOT; maximum number of events in the LCPLOT table.
    #: ``LCPLOT(mxlc:N)`` in SNTABLE_LIST overrides this setting. No light
    #: curves are written unless LCPLOT is requested.
    mxlc: int = 5
    #: SNANA SNTABLE_LIST; ``OUTLIER(nsig:N)``, the minimum absolute flux
    #: residual for inclusion in the OUTLIER table, in units of the combined
    #: measurement and model uncertainty.
    outlier_nsig: float = 3.0
    #: SNANA SNTABLE_LIST; ``OUTLIER(ftrue:N)``, a threshold on the simulated
    #: true flux, used only in a simulation job and ignored in a fit.
    outlier_ftrue: float = 0.01
    #: SNANA SNTABLE_LIST; ``OUTLIER(sbmag:N)``, a threshold on the host
    #: surface brightness, likewise ignored in a fit.
    outlier_sbmag: float = 999.0
    # Not set by an input key
    #: The SNTABLE_LIST options recognised but without effect here, each with
    #: its reason, which the run reports once as it starts.
    _unsupported: List[str] = field(default_factory=list)

    # Output paths
    #: SNANA TEXTFILE_PREFIX; the start of every output file name, so
    #: ``MY_FIT`` writes ``MY_FIT.FITRES.TEXT``. If unset, the prefix is the
    #: name of the version's HEAD file with its ``_HEAD.FITS`` ending
    #: removed.
    textfile_prefix: Optional[str] = None
    #: The path of the FITRES file to write, in place of the
    #: name TEXTFILE_PREFIX would give it; the other tables are written into
    #: its directory with the prefix TEXTFILE_PREFIX gives them.
    output_name: Optional[str] = None
    #: SNANA SNTABLE_LIST; ``nonuisance`` omits nuisance parameters and their
    #: uncertainties from FITRES. For BayeSN, these are the intrinsic residual
    #: coefficients. It has no effect for SALT.
    nonuisance: bool = False
    #: SNANA MAGLIM_VMAX; written 'r 21.5 i 22.0', the per-band limiting
    #: magnitude the 1/Vmax volume of each event is computed to.
    maglim_vmax: Dict[str, float] = field(default_factory=dict)
    #: SNANA OPT_VMAX; options for the 1/Vmax calculation, as a bit mask. Bit
    #: 0 (value 1), selecting model peak magnitudes, is refused. Use 0 for the
    #: calculation from observed peak magnitudes; the other bits are accepted
    #: for input compatibility and change nothing.
    opt_vmax: int = 0


@dataclass
class FitConfig:
    """Configuration for a light-curve fit."""

    data: Data = field(default_factory=Data)
    selection: Selection = field(default_factory=Selection)
    model: Model = field(default_factory=Model)
    fit: Fit = field(default_factory=Fit)
    photoz: PhotoZ = field(default_factory=PhotoZ)
    output: Output = field(default_factory=Output)
