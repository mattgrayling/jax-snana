"""The simulation's concept objects: one class per concept.

`SimulationConfig` holds the parsed settings. Preparation snapshots those
settings before loading the model and survey.
Everything read from external files or derived from them lives on `LightCurveSimulator`,
the prepared object `build_simulation` returns.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import jax
import numpy as np

from jax_snlc_core.models import MODEL_REGISTRY, infer_model_type_from_path
from jax_snlc_core.random import root_key
from jax_snlc_core.cosmology import load_hz_map
from jax_snlc_core.dust import load_dust_law
from jax_snlc_core.filters import ZEROPOINT_FLUXCAL_SNANA
from jax_snlc_core.snana_files import is_ignorefile
from jax_snlc_sim.population import declared_population
from jax_snlc_sim.rate import neutral_rate_mods, parse_dndz_entries


@dataclass
class EpochsTimeCut:
    """CUTWIN_EPOCHS_SNRMIN threshold, duration and filters."""

    snr_min: float = 0.0
    time_above_max: float = 99999.0
    filters: str = ''


@dataclass
class Run:
    """Event counts, random seed, job splitting and diagnostics."""

    input_names = {
        'ngen_scale_non1a': ['NGEN_SCALE_NON1A', 'NGEN_SCALE_NONIA'],
        'random_seed': ['RANSEED'],
    }

    # Set by the program rather than by an input key
    #: Path of the input file this run was parsed from.
    _input_file: str = ""
    #: Every (key, line) of the input as read, for the README's echo of
    #: the keys the summary block does not write.
    _input_lines: List[Tuple[str, str]] = field(default_factory=list)

    # Event counts
    #: Keep generating until this many events pass the trigger and the cuts.
    ngen_lc: int = 0
    #: Generate this many events in total, however many of them are accepted.
    ngentot_lc: int = 0
    #: Generate the count the rate model expects in this many survey seasons.
    ngen_season: float = 0.0
    #: SNANA README; a key with no value, written as the word alone. With it a
    #: run may set none of the three counting keys, to write a README without
    #: generating anything. Every run writes its .README whether or not this
    #: key is given.
    readme_dumpflag: bool = False
    #: Multiply the resolved event count by this factor.
    ngen_scale: float = 1.0
    #: Scale the count for the NON1ASED and SIMSED models alone.
    ngen_scale_non1a: float = 1.0
    #: Maximum number of generated events, as a multiple of NGEN_LC.
    ngen_lc_maxgen_factor: int = 100

    # Random seed and precision
    #: SNANA RANSEED; random seed for the simulation. Required.
    random_seed: int = 12345
    #: SNANA RANSYSTPAR_<NAME>; Systematic shifts drawn once for the run, keyed
    #: by the name after the prefix.
    ransystpar: Dict[str, Any] = field(default_factory=dict)
    #: Double or single precision. Omit to use the active precision in Python;
    #: command-line runs default to single. A jax-only key.
    double_precision: bool | None = None

    # Diagnostics, batch splitting and the input search path
    #: Stop after initialisation, before any event is generated.
    init_only: int = 0
    #: SNANA DASHBOARD; Summarise the inputs and quit.
    dashboard: bool = False
    #: Disable the abort on a physically impossible flux.
    skip_check_crazyflux: int = 0
    #: Whether the SIMLIB must open with a DOCUMENTATION block: 1 aborts a
    #: SIMLIB without one and 0 reduces that to a warning. The input file
    #: itself is not checked. SEARCHEFF files always require the block.
    #: Accepted on the command line only.
    require_docana: int = 1
    #: Index of this batch job, from 1 to NJOBTOT.
    jobid: int = 0
    #: Number of batch jobs the run is split across; 0 means it is not split.
    njobtot: int = 0
    #: Extra directory searched for input files.
    path_user_input: str = ""


@dataclass
class Rates:
    """Volumetric event rates and simulation ranges."""

    input_names = {
        'redshift_range': ['GENRANGE_REDSHIFT'],
        'peakmjd_range': ['GENRANGE_PEAKMJD'],
        'peakmjd_exclude': ['GENRANGE_PEAKMJD_EXCLUDE'],
        'trest_range': ['GENRANGE_TREST'],
    }

    #: Cosmological redshift range of the simulated events. Required, with a
    #: positive lower bound. Heliocentric redshifts can lie slightly outside
    #: this range.
    redshift_range: Tuple[float, float] = field(default=(0.0, 0.0), metadata={'static': True})
    #: MJD range the true peak is drawn from.
    peakmjd_range: Tuple[float, float] = field(default=(0.0, 0.0), metadata={'static': True})
    #: MJD window no peak falls in, active when its upper bound is positive.
    #: Its span is also subtracted from the season length used for the expected
    #: count.
    peakmjd_exclude: Tuple[float, float] = field(default=(0.0, 0.0), metadata={'static': True})
    #: Rest-frame phase range about peak over which epochs are generated.
    trest_range: Tuple[float, float] = field(default=(0.0, 0.0), metadata={'static': True})
    #: Observation window SIMLIB_DUMP reports as its MJD_RANGE cut. Unset takes
    #: the range from the SIMLIB itself.
    genrange_mjd: Optional[Tuple[float, float]] = field(default=None, metadata={'static': True})
    #: Survey solid angle in steradians, which with the rate model fixes the
    #: expected event count.
    solid_angle: float = field(default=0.0, metadata={'host_only': True})
    #: SNANA DNDZ, DNDZ_FILE; volumetric rate in events per comoving Mpc cubed
    #: per rest-frame year, using the model arguments listed above. Python/YAML
    #: takes a list of rows, e.g. [[POWERLAW, 2.6e-5, 1.5]]. A tabulated rate
    #: uses [[FILE, path/to/rate.dat]]. Repeat POWERLAW2 rows for a piecewise rate.
    dndz_entries: List[Any] = field(default_factory=list, metadata={'host_only': True})
    #: SNANA DNDZ_PEC1A/DNDZ_PECIA; The peculiar-Ia rate model, separate from
    #: the Ia one.
    dndz_entries_pec1a: List[Any] = field(default_factory=list, metadata={'host_only': True})
    #: SNANA DNDZ_SCALE, DNDZ_SCALE_NON1A, DNDZ_ALLSCALE, DNDZ_ZEXP_REWGT,
    #: DNDZ_ZPOLY_REWGT, DNDZ_Z1POLY_REWGT; Six keys that reweight the DNDZ
    #: rate model. DNDZ_SCALE multiplies the rate by its first value for a type
    #: Ia model and by its second for a template or SIMSED model,
    #: DNDZ_SCALE_NON1A sets that second value alone, and DNDZ_ALLSCALE
    #: multiplies every model alike. DNDZ_ZEXP_REWGT multiplies the rate by z
    #: to the power it gives, and DNDZ_ZPOLY_REWGT and DNDZ_Z1POLY_REWGT by a
    #: polynomial in z and in 1+z respectively, written as comma-separated
    #: coefficients starting at the constant term.
    dndz_rate_mods: Dict[str, Any] = field(default_factory=neutral_rate_mods, metadata={'host_only': True})

    # Reported redshifts and type codes
    #: Gaussian smearing of the reported heliocentric redshift, which is also
    #: reported as its error.
    gensigma_redshift: float = field(default=0.0, metadata={'static': True})
    #: Shift added to the reported heliocentric redshift.
    genbias_redshift: float = field(default=0.0, metadata={'static': True})
    #: SNANA GENTYPE/GENTYPES; The spectroscopic type code, which on its own
    #: implies the photometric code 100 above it.
    gentype: int = field(default=-9, metadata={'static': True})
    #: SNANA GENTYPE/GENTYPES; GENTYPES gives the two codes explicitly, its
    #: second value being the photometric one.
    gentype_phot: Optional[int] = field(default=None, metadata={'static': True})
    #: SNANA SNTYPE_Ia/SNTYPES_Ia; The SNTYPE written for a spectroscopically
    #: confirmed event.
    sntype_spec: int = field(default=1, metadata={'static': True})
    #: SNANA SNTYPE_Ia/SNTYPES_Ia; SNTYPES_Ia gives the two codes as a pair,
    #: its second value being the one written for a photometric-only event,
    #: where SNTYPE_Ia on its own puts that code 100 above the confirmed one.
    sntype_phot: int = field(default=101, metadata={'static': True})

    # Estimating the observed peak MJD
    #: Observed peak-time estimator: 8 uses the brightest epoch; 16 averages
    #: high signal-to-noise epochs near it; 32 weights that average by log10 SNR.
    opt_setpkmjd: int = field(default=16, metadata={'static': True})
    #: Gaussian smearing of the estimated peak. A positive value selects
    #: it instead of the clump, forcing OPT_SETPKMJD to 0.
    gensigma_peakmjd: float = field(default=0.0, metadata={'static': True})
    # Fixed, with no input key of its own
    #: Width in days of the window the Fmax clump is collected in, fixed in
    #: SNANA with the two SNR cuts below.
    pkmjd_mjdwin: float = field(default=60.0, metadata={'static': True})
    #: SNR an epoch must reach to join the clump.
    pkmjd_snrcut: float = field(default=5.0, metadata={'static': True})
    #: Fallback SNR threshold when no epoch reaches the clump's own.
    pkmjd_snrcut_backup: float = field(default=3.0, metadata={'static': True})


@dataclass
class Cosmology:
    """Cosmology, peculiar velocities and redshift-frame conversion."""

    input_names = {
        'w0': ['w0_LAMBDA'],
        'wa': ['wa_LAMBDA'],
        'hzfun_file': ['HzFUN_FILE'],
    }

    #: Matter density today.
    omega_matter: float = 0.315
    #: Dark-energy density today; the curvature follows from the two.
    omega_lambda: float = 0.685
    #: Hubble constant in km/s/Mpc.
    h0: float = 70.0
    #: Dark-energy equation of state today.
    w0: float = -1.0
    #: Its evolution, as w(a) = w0 + wa (1 - a).
    wa: float = 0.0
    #: Tabulated (z, H/H0) replacing the analytic expansion history.
    hzfun_file: str = field(default="", metadata={'host_only': True})
    #: Anisotropic-distance model. Empty is isotropic, and V04 (Visser 2004) is
    #: the only model.
    anisotropy_modelname: str = field(default="", metadata={'static': True})
    #: Dispersion in km/s of the true peculiar velocity drawn per event.
    gensigma_vpec: float = field(default=0.0, metadata={'static': True})
    #: Error on the reported peculiar velocity, which smears the reported value
    #: when it is smaller than GENSIGMA_VPEC.
    vpec_err: float = field(default=0.0, metadata={'static': True})
    #: Switches the conversion between zCMB and heliocentric redshift. The
    #: supernova's redshift is converted when the value is 1 or more, a wrong
    #: host's when it is above 1, and the true host's unless it is 0. Every
    #: conversion uses an apex speed of 369.82 km/s.
    vel_cmbapex: float = field(default=369.82, metadata={'static': True})
    #: Give every event this peculiar velocity in km/s rather than drawing one.
    #: The default value above 1e8 leaves this option disabled.
    vpec_force: float = field(default=1.0e9, metadata={'static': True})


@dataclass(frozen=True)
class SubPopulation:
    """One named class of SNe, drawn once per event.

    The class is an index: what it changes lives on each parameter's own
    record. That single draw is what correlates the quantities it changes,
    where per-parameter mixtures would leave them independent.
    """

    name: str = field(metadata={'static': True})
    #: relative, so the class is drawn in proportion to it; a VARIATION on
    #: WGT makes it per event and no constraint between classes is needed
    weight: float
    genmag_off_global: Optional[float] = None
    #: Per-parameter distributions for this class; omitted parameters use the shared population.
    populations: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Model:
    """Source model, parameter distributions and intrinsic scatter."""

    input_names = {
        'gensmear_rangauss_fix': ['GENSMEAR_RANGauss_FIX', 'GENSMEAR_RANGAUSS_FIX'],
    }

    #: SNANA GENMODEL; The SED model the light curves are generated from, given
    #: as a path or a supported model name, such as a SALT model
    #: directory or BAYESN.<version>. Required, and it sets which of the
    #: model's own keys the input may use.
    genmodel: str = field(default="", metadata={'host_only': True})

    @property
    def model_type(self) -> str:
        """Registered model family, determined by GENMODEL."""
        return infer_model_type_from_path(self.genmodel) or ''

    def population_declaration(self):
        """The model's population parameter names, their other input
        spellings and SNANA's defaults, as its class or its input declares
        them."""
        return declared_population(MODEL_REGISTRY[self.model_type],
                                   self._structured_config)

    # Generation options
    #: SNANA GENMODEL_MSKOPT; Model-specific scatter and extrapolation options.
    #: See :ref:`sim-model-options`.
    genmodel_mskopt: int = field(default=0, metadata={'static': True})

    # Set by the model's own keys
    #: The model's own structured input, which only the model reads.
    _structured_config: Dict[str, Any] = field(default_factory=dict, metadata={'host_only': True})

    #: The supplied values of the keys the chosen model declares, typed by the
    #: parser and set on the model instance at construction.
    _model_input_values: Dict[str, Any] = field(default_factory=dict, metadata={'static': True})

    # Magnitude offsets
    #: Magnitude added to every event, brightening it for a negative value.
    #: A GENPDF_FILE may set it instead.
    mag_offset: float = field(default=0.0, metadata={})
    #: Shift of the distance modulus in magnitudes, drawn per event from a
    #: range: uniformly, or under a z-dependent
    #: DNDZ_ZPOLY_REWGT or DNDZ_Z1POLY_REWGT so that the shifted distances keep
    #: the reweighted redshift distribution. SNANA inputs take 'lo:hi' or a
    #: scalar for a fixed shift. Python/YAML takes [lo, hi]; equal limits
    #: specify a fixed shift.
    mushift: Tuple[float, float] = field(default=(0.0, 0.0), metadata={'static': True})
    #: Magnitude added to every event before the band integral. Shift it with a
    #: ZVARIATION or a SUBPOP class for a redshift-dependent or class-dependent
    #: brightness.
    genmag_off_global: float = field(default=0.0, metadata={})
    #: A further magnitude added to the events of a NON1A template library
    #: alone, leaving any Ia population in the same run untouched.
    genmag_off_non1a: float = field(default=0.0, metadata={})
    #: SNANA GENMAG_OFF_MODEL; magnitude offsets added to the model in each
    #: band. SNANA inputs give one value per GENFILTERS band, in that order;
    #: Python/YAML takes a mapping such as {g: 0.003, r: 0.0}.
    genmag_off_model: Dict[str, float] = field(default_factory=dict)

    # Intrinsic scatter
    #: Width in magnitudes of a Gaussian drawn once per event and applied to
    #: every band alike, the simplest form of intrinsic scatter. A
    #: GENPDF_FILE may set it instead; specifying it in both inputs is rejected.
    genmag_smear: float = field(default=0.0, metadata={'static': True})
    #: Scale applied to the scatter GENMAG_SMEAR_MODELNAME produces, and to
    #: that alone.
    genmag_smear_scale: float = field(default=1.0, metadata={})
    #: SNANA GENMAG_SMEAR_SCALE(<PARAMETER>); the same scale as a polynomial in
    #: one of the model's own parameters, evaluated per event, which lets the
    #: scatter grow with colour or with shape.
    genmag_smear_scale_poly: Any = field(default=None, metadata={'static': True})
    #: Wavelength-dependent intrinsic-scatter prescription, such as G10 or C11.
    #: Only G10 needs the SALT directory, for its colour-dispersion file.
    genmag_smear_modelname: Optional[str] = field(default=None, metadata={'host_only': True})
    #: SNANA GENMAG_SMEAR_FILTER; a repeatable '<bands> <sigma>' line giving
    #: those bands their own Gaussian scatter, drawn once per event and band.
    genmag_smear_filter: Dict[str, float] = field(default_factory=dict)
    #: A phase-correlated scatter, given as 'sigma tau': a Gaussian of width
    #: sigma magnitudes whose correlation between two epochs falls off as
    #: exp(-dT/tau) in the rest-frame separation dT. A sigma at or below zero
    #: disables it.
    genmag_smear_addphasecor: Any = field(default=None, metadata={'static': True})
    #: Range in standard deviations every magnitude-scatter Gaussian is
    #: truncated to.
    sigmaclip_magsmear: Tuple[float, float] = field(default=(-3.0, 3.0), metadata={'static': True})
    #: Give every scatter-model random this fixed value instead of drawing one,
    #: to inspect the effect of a particular draw. The default value of -999
    #: retains random sampling.
    gensmear_rangauss_fix: float = field(default=-999.0, metadata={'static': True})
    #: SNANA SALT2BETA_cPOLY; beta as a polynomial in colour instead of a
    #: single value. It is accepted with any GENMODEL, and only the SALT model
    #: uses it.
    salt2beta_cpoly: List[Any] = field(default_factory=list, metadata={'static': True})

    # Parameter distributions
    #: SNANA GENPEAK_<PAR>, GENSIGMA_<PAR>, GENRANGE_<PAR>, GENPDF_FILE;
    #: parameter distributions. Python/YAML uses the populations mapping
    #: described above. SNANA parameter suffixes are case-insensitive, such
    #: as SALT2x1 for SALT and THETA for BayeSN. Required model populations
    #: must be supplied through these settings or a GENPDF map. R_V defaults
    #: to 3.1 when no R_V population is specified.
    populations: Dict[str, Any] = field(default_factory=dict)
    #: Population-map boundary options. See :ref:`sim-population-map-options`.
    genpdf_optmask: int = field(default=0, metadata={'static': True})

    # Explosion time
    #: Put every event's explosion at this MJD rather than drawing a time of
    #: peak, which pins GENRANGE_PEAKMJD to that single value. Only the models
    #: that define an explosion epoch accept it, and some of them need
    #: OPTMASK_T0SHIFT_EXPLODE as well.
    mjd_explode: Optional[float] = field(default=None, metadata={'static': True})

    # Redshift dependence and sub-populations
    #: A file of ZVARIATION entries, each naming a population parameter and
    #: giving its shift against redshift as a polynomial or as interpolated
    #: bins. Every shift must vanish at redshift zero and the bins must cover
    #: GENRANGE_REDSHIFT. It cannot be combined with ZVARIATION_POLY.
    zvariation_file: str = field(default='', metadata={'host_only': True})
    #: A shift of a population parameter against redshift, written in the
    #: input file itself as one '<parameter> <poly>' line per parameter, the
    #: polynomial's coefficients starting at the constant term so that the
    #: shift is anchored at redshift zero. It cannot be combined with
    #: ZVARIATION_FILE.
    zvariation_poly: List[Tuple[str, str]] = field(default_factory=list, metadata={'host_only': True})
    #: SNANA VARIATION; a shift of a population distribution component or
    #: an applied offset, written as '<target> <quantity> POLY <coefficients>
    #: PIVOT <value>'. The shift depends on <quantity>, a drawn parameter or a
    #: HOSTLIB column, and the coefficients start at the linear term, the
    #: shift vanishing at the pivot. The key is jax only.
    _offset_shifts: Dict[str, Any] = field(default_factory=dict)
    #: SNANA ZVARIATION_FILE; an entry naming EBV_IGM, in the file or in the
    #: input file, reddens for intergalactic dust rather than shifting a
    #: population parameter, and is applied alongside the Milky Way's
    #: extinction.
    _ebv_igm_shift: Optional[Any] = None
    #: SNANA GENRANGE_<PAR>; each population parameter's range as the input
    #: gives it, else the model's default range for it; a SIMLIB entry's own
    #: value or range for the parameter must fall inside it under
    #: USE_SIMLIB_SALT2.
    _input_genranges: Dict[str, Tuple[float, float]] = field(default_factory=dict, metadata={'static': True})
    #: SNANA SUBPOP; a named class of events, written as '<class> <key>
    #: <values>' with one line per setting and a WGT line giving its share.
    #: A class is drawn once per event and may override any population key,
    #: GENMAG_OFF_GLOBAL among them, which correlates the quantities it
    #: changes in a way per-parameter mixtures cannot. The key is jax only.
    subpops: Tuple['SubPopulation', ...] = ()

    def offsets_for(self, target: str, index: int) -> Tuple[Any, ...]:
        """The shifts on one applied target for one class."""
        per_class = self._offset_shifts.get(target, ())
        return per_class[index] if per_class else ()

    def quantities_depended_on(self):
        """Every quantity a shift or a GENPDF map depends on.

        The parameters carry their own shifts; the targets the run applies
        rather than draws carry theirs in `offset_shifts`.
        """
        for pop in self.populations.values():
            yield from pop.depends_on()
        for per_class in self._offset_shifts.values():
            for entries in per_class:
                yield from (entry.depends_on for entry in entries)

    def parameter_range(self, param_name: str):
        """The widest range the run draws a parameter over, across every
        class, or None where nothing draws it."""
        pop = self.populations.get(param_name)
        if pop is None or not pop.distributions:
            return None
        spans = [d.range for d in pop.distributions]
        return min(s[0] for s in spans), max(s[1] for s in spans)


    def model_config(self, trest_range):
        """The simulation settings read by this model's generation methods."""
        m = self
        return {
            'genrange_trest': trest_range,
            **m._model_input_values,
            'populations': m.populations,
            'SIGMACLIP_MAGSMEAR': m.sigmaclip_magsmear,
            'salt2beta_cpoly': m.salt2beta_cpoly,
            **m._structured_config,
        }


@dataclass
class Survey:
    """Survey observations, calibration and detection efficiencies."""

    input_names = {
        'survey_override': ['SIMLIB_SURVEY'],
        'genrange_b': ['GENRANGE_b'],
        'use_simlib_take_spectrum': ['USE_SIMLIB_TAKE_SPECTRUM', 'USE_SIMLIB_SPECTRA'],
        'exposure_time_global': ['EXPOSURE_TIME'],
        'exposure_time': ['EXPOSURE_TIME_FILTER'],
        'mjd_template': ['MJD_TEMPLATE_FILTER'],
        'searcheff_apply_detect_single': ['APPLY_DETECT_SINGLE'],
        'searcheff_minobs': ['MINOBS_SEARCH'],
        'searcheff_spec_eff_file': ['SEARCHEFF_SPEC_FILE'],
        'searcheff_zhost_eff_file': ['SEARCHEFF_zHOST_FILE'],
        'searcheff_spec_magshift': ['SEARCHEFF_SPEC_MAGSHIFT', 'MAGSHIFT_SPECEFF'],
        'searcheff_zhost_magshift': ['SEARCHEFF_zHOST_MAGSHIFT', 'MAGSHIFT_zHOSTEFF'],
        'searcheff_photflag_detect': ['PHOTFLAG_DETECT'],
        'searcheff_photflag_trigger': ['PHOTFLAG_TRIGGER'],
        'searcheff_map_refac': ['REFAC_SEARCHEFF_MAP'],
    }

    # Survey observations
    #: The SIMLIB giving the survey's cadence, depth and conditions, one LIBID
    #: per sightline. Required.
    simlib_file: str = field(default="", metadata={'host_only': True})
    #: Drop a season shorter than this many days before any LIBID is used.
    simlib_minseason: float = field(default=0.0, metadata={'host_only': True})
    #: Skip a LIBID with fewer observations than this.
    simlib_minobs: int = field(default=1, metadata={'host_only': True})
    #: Skip a LIBID with more observations than this.
    simlib_maxobs: int = field(default=999999, metadata={'host_only': True})
    #: Use only this SIMLIB entry for all events.
    simlib_idlock: int = field(default=-9, metadata={'host_only': True})
    #: Skip the LIBIDs below this one.
    simlib_idstart: int = field(default=0, metadata={'static': True})
    #: Skip these LIBIDs by number.
    simlib_idskip: Tuple[int, ...] = field(default=(), metadata={'host_only': True})
    #: Keep one LIBID in this many, those whose number divides by it.
    simlib_prescale: int = field(default=1, metadata={'host_only': True})
    #: Add this to every observation MJD, which moves the survey in time
    #: without editing the file.
    simlib_mjd_shift: float = field(default=0.0, metadata={'host_only': True})
    #: Drop the observations falling inside this MJD range.
    genrange_mjd_exclude: Optional[Tuple[float, float]] = field(default=None, metadata={'host_only': True})
    #: Use only the named fields, as a comma-separated list. ALL uses all.
    simlib_fieldlist: str = field(default="ALL", metadata={'host_only': True})
    #: Draw this many consecutive events from each LIBID before moving to the
    #: next. With SIMLIB_MSKOPT bit 4, the run is capped at the number of
    #: LIBIDs times this repeat count.
    simlib_nrepeat: int = field(default=1, metadata={'static': True})
    #: Begin the cycle at a LIBID drawn from the first this many, rather than
    #: at the first.
    simlib_maxranstart: int = field(default=0, metadata={'static': True})
    #: Observing-log options. See :ref:`sim-simlib-options`.
    simlib_mskopt: int = field(default=0, metadata={'static': True})
    #: Restrict the sky positions events are drawn at to this right-ascension
    #: range, in degrees.
    genrange_ra: Tuple[float, float] = field(default=(-360.0, 360.0), metadata={'static': True})
    #: Restrict the sky positions events are drawn at to this declination
    #: range, in degrees.
    genrange_decl: Tuple[float, float] = field(default=(-90.0, 90.0), metadata={'static': True})
    #: Restrict the sky positions events are drawn at to this Galactic latitude
    #: range, in degrees, for example to avoid the plane.
    genrange_b: Tuple[float, float] = field(default=(-90.0, 90.0), metadata={'static': True})
    #: SNANA SIMLIB_SURVEY; the survey name, overriding the one in the SIMLIB
    #: header.
    survey_override: str = field(default="", metadata={'host_only': True})
    #: SNANA SIMLIB_NSKIPMJD; keep one night of observations in every this
    #: many, written as comma-separated PRESCALE(TRESTMIN) terms such as
    #: '2(50),4(100)': each prescale applies from the rest-frame phase in its
    #: parentheses onward, so the cadence is thinned progressively at late
    #: phases, and a term without parentheses applies from the first epoch.
    simlib_nskipmjd: Tuple[Tuple[int, int], ...] = field(default=(), metadata={'host_only': True})
    #: Summarise the SIMLIB's cadence and quit without generating anything. A
    #: negative value leaves the simulation to run.
    simlib_dump: int = field(default=-1, metadata={'host_only': True})

    # Bands, calibration and cadence
    #: SNANA GENFILTERS; the bands to generate, as the single characters naming
    #: them in the kcor file.
    genfilters: str = field(default="", metadata={'host_only': True})
    #: SNANA KCOR_FILE; a calibration input or FITS file describing filter transmissions,
    #: the primary reference spectrum and the zero points. Required.
    kcor_file: str = field(default="", metadata={'host_only': True})
    #: SNANA LEAKAGE_CUT; suppress filter transmission below this fraction of
    #: its peak to remove out-of-band leakage. Zero retains the full
    #: transmission curve.
    leakage_cut: float = field(default=0.0, metadata={'host_only': True})
    #: Observations closer together than this many days count as one epoch for
    #: the trigger, rather than as two detections.
    newmjd_dif: float = field(default=0.007, metadata={'host_only': True})
    #: SNANA MJD_TEMPLATE, MJD_TEMPLATE_FILTER; the MJD of the template image,
    #: before which an observation has no template noise. The bare key sets
    #: every band, and the _FILTER form names the bands it sets.
    mjd_template: Dict[str, float] = field(default_factory=dict, metadata={'static': True})
    #: SNANA EXPOSURE_TIME, EXPOSURE_TIME_FILTER; the exposure time as a ratio
    #: to the SIMLIB's own, which shifts the zero point and scales the noise.
    #: The bare key sets every band and the _FILTER form, written '<bands>
    #: <ratio>', the bands it names.
    exposure_time: Dict[str, float] = field(default_factory=dict, metadata={'host_only': True})
    #: SNANA EXPOSURE_TIME, EXPOSURE_TIME_FILTER; a bare EXPOSURE_TIME above 10
    #: also switches saturation off.
    exposure_time_global: float = field(default=1.0, metadata={'static': True})
    #: Quantities scaled with exposure time. See :ref:`sim-exposure-options`.
    exposure_time_mskopt: int = field(default=7, metadata={'host_only': True})
    #: SNANA FORCEVAL_PSF; replace the first Gaussian PSF sigma with this
    #: fixed value in pixels, after SIMLIB unit conversion. The default
    #: retains the SIMLIB widths.
    forceval_psf: float = field(default=-9.0, metadata={'host_only': True})
    # Fixed, with no input key of its own
    #: Days of padding either side of the observer-frame window an event is
    #: generated over. SNANA's own constant.
    genrange_tobs_pad: float = field(default=0.1, metadata={'static': True})

    # Event parameters from the SIMLIB
    #: Take each event's redshift from its LIBID instead of drawing one, to
    #: replay a SIMLIB of real events.
    use_simlib_redshift: int = field(default=0, metadata={'static': True})
    #: Take the time of peak from the LIBID in the same way.
    use_simlib_peakmjd: int = field(default=0, metadata={'static': True})
    #: Take the SALT2 shape and colour from the LIBID.
    use_simlib_salt2: int = field(default=0, metadata={'static': True})
    #: Take the distance modulus from the LIBID.
    use_simlib_distance: int = field(default=0, metadata={'static': True})
    #: Take the spectrograph exposures from the LIBID.
    use_simlib_spectrograph: bool = field(default=False, metadata={'static': True})
    #: SNANA USE_SIMLIB_TAKE_SPECTRUM/USE_SIMLIB_SPECTRA; take the spectrum
    #: epochs from the LIBID rather than from the TAKE_SPECTRUM keys.
    use_simlib_take_spectrum: bool = field(default=False, metadata={'static': True})
    #: Match each event's host to the LIBID's GROUPID, so the host comes from
    #: the same group as the sightline.
    use_simlib_groupid: bool = field(default=False, metadata={'static': True})

    # Detection and follow-up selection
    #: SNANA APPLY_SEARCHEFF_OPT; Required detection stages. See :ref:`sim-detection-options`.
    _searcheff_apply: bool = field(default=False, metadata={'static': True})
    #: SNANA APPLY_SEARCHEFF_OPT; An event must pass every selected stage.
    apply_searcheff_opt: int = field(default=0, metadata={'static': True})
    #: SNANA APPLY_SEARCHEFF_OPT; A stage whose efficiency file is not given
    #: passes every event.
    _searcheff_apply_pipeline: bool = field(default=False, metadata={'static': True})
    #: SNANA APPLY_SEARCHEFF_OPT; The stages each event passed are written to
    #: SIM_SEARCHEFF_MASK under the same bit values, with 8 added where
    #: either redshift stage passed.
    _searcheff_apply_spec: bool = field(default=False, metadata={'static': True})
    #: SNANA APPLY_DETECT_SINGLE; count an event as found on a single
    #: detection, instead of applying the pipeline logic.
    searcheff_apply_detect_single: bool = field(default=False, metadata={'static': True})
    #: SNANA MINOBS_SEARCH; the fewest epochs an event must have before the
    #: trigger is evaluated at all.
    searcheff_minobs: int = field(default=2, metadata={'static': True})
    #: Per-band detection efficiency against signal-to-noise, giving each
    #: observation a detection probability. Without it, or with DEFAULT,
    #: every event counts as found by the pipeline.
    searcheff_pipeline_eff_file: Optional[str] = field(default=None, metadata={'host_only': True})
    #: The logic those detections must satisfy, as a file holding one
    #: '<SURVEY>: <nights> <bands>' line per survey, such as 'DES: 2 g+r+i+z'
    #: for detections on two separate nights in any one of those bands;
    #: letters run together, as in 'gr+iz', mean both bands on the same
    #: night. The line read is the one naming the SIMLIB's SURVEY, and a file
    #: without it aborts the run. DEFAULT reads SNANA's
    #: SEARCHEFF_PIPELINE_LOGIC.DAT; the key is required whenever
    #: SEARCHEFF_PIPELINE_EFF_FILE holds a curve.
    searcheff_pipeline_logic_file: Optional[str] = field(default=None, metadata={'host_only': True})
    #: SNANA SEARCHEFF_SPEC_FILE; the probability of spectroscopic
    #: confirmation, against magnitude, redshift or colour. An event whose
    #: field no map covers aborts the run, unless the file sets
    #: OPT_FIELDMATCH_REQUIRE: 0, which gives it zero efficiency.
    searcheff_spec_eff_file: Optional[str] = field(default=None, metadata={'host_only': True})
    #: SNANA SEARCHEFF_zHOST_FILE; the probability of obtaining a host
    #: redshift for an event the spectroscopy missed. When a HOSTLIB is
    #: used, an event with no matched host cannot receive a host redshift.
    #: Field matching is as for SEARCHEFF_SPEC_FILE.
    searcheff_zhost_eff_file: Optional[str] = field(default=None, metadata={'host_only': True})
    #: SNANA SEARCHEFF_SPEC_SHIFT(<VARIABLE>); shifts applied to the variables
    #: the spectroscopic efficiency is read against, for systematics.
    searcheff_spec_shifts: Dict[str, float] = field(default_factory=dict, metadata={'static': True})
    #: SNANA SEARCHEFF_zHOST_SHIFT(<VARIABLE>); shifts applied to the
    #: variables the host-redshift efficiency is read against, for
    #: systematics.
    searcheff_zhost_shifts: Dict[str, float] = field(default_factory=dict, metadata={'static': True})
    #: SNANA SEARCHEFF_SPEC_MAGSHIFT/MAGSHIFT_SPECEFF; a magnitude shift
    #: applied before the spectroscopic efficiency is read, which moves its
    #: threshold.
    searcheff_spec_magshift: float = field(default=0.0, metadata={'static': True})
    #: SNANA SEARCHEFF_zHOST_MAGSHIFT/MAGSHIFT_zHOSTEFF; a magnitude shift
    #: applied before the host-redshift efficiency is read, which moves its
    #: threshold.
    searcheff_zhost_magshift: float = field(default=0.0, metadata={'static': True})
    #: PHOTFLAG bit marking a detected epoch in the written photometry. Zero
    #: takes the value the SEARCHEFF_PIPELINE_EFF_FILE header declares, and any
    #: other value overrides it.
    searcheff_photflag_detect: int = field(default=0, metadata={'static': True})
    #: PHOTFLAG bit marking the epoch that triggered the search, taken from
    #: the SEARCHEFF_PIPELINE_EFF_FILE header in the same way when zero.
    searcheff_photflag_trigger: int = field(default=0, metadata={'static': True})
    #: Which SPECID and zHOST map logic is used: 1, where a zHOST REQUIRE map
    #: reuses the SPECID random of the same map index, or 2, with independent
    #: zHOST randoms. 0 is refused.
    searcheff_map_refac: int = field(default=1, metadata={'static': True})
    #: Scale factor applied to the spectroscopic confirmation efficiency.
    searcheff_spec_scale: float = field(default=1.0, metadata={'host_only': True})


# HOSTLIB_MSKOPT bits with no generation meaning here: +HOSTAPPEND, +HOSTMAGS,
# +HOSTNBR and +HOSTSELECT rewrite the catalogue instead (sntools_host.h:48-51)
HOSTLIB_MSKOPT_REWRITE_BITS = {4096: "+HOSTAPPEND", 8192: "+HOSTMAGS",
                               16384: "+HOSTNBR", 32768: "+HOSTSELECT"}
# Gaussian photo-z quantiles from ZPHOT and ZPHOT_ERR (sntools_host.h:52)
HOSTLIB_MSKOPT_ZPHOT_QGAUSS = 65536


@dataclass
class Host:
    """Host-galaxy catalogue, association and extinction settings."""

    input_names = {
        'wgtmap_file': ['HOSTLIB_WGTMAP_FILE'],
        'wgtmap_extrap': ['HOSTLIB_WGTMAP_EXTRAP'],
        'dztol': ['HOSTLIB_DZTOL'],
        'mskopt': ['HOSTLIB_MSKOPT'],
        'sbradius': ['HOSTLIB_SBRADIUS'],
        'scale_property_err': ['HOSTLIB_SCALE_PROPERTY_ERR'],
        'scale_logmass_err': ['HOSTLIB_SCALE_LOGMASS_ERR'],
        'maxddlr': ['HOSTLIB_MAXDDLR'],
        'maxddlr2': ['HOSTLIB_MAXDDLR2'],
        'mindaysep_samegal': ['HOSTLIB_MINDAYSEP_SAMEGAL'],
        'galid_force': ['HOSTLIB_GALID_FORCE'],
        'smear_sersic': ['HOSTLIB_SMEAR_SERSIC'],
        'galid_priority': ['HOSTLIB_GALID_PRIORITY'],
        'abmag_offset': ['HOSTLIB_ABMAG_OFFSET'],
        'abmag_force': ['HOSTLIB_ABMAG_FORCE'],
        'fixran_wgt': ['HOSTLIB_FIXRAN_WGT'],
        'fixran_radius': ['HOSTLIB_FIXRAN_RADIUS'],
        'fixran_phi': ['HOSTLIB_FIXRAN_PHI'],
        'mnintflux_snpos': ['HOSTLIB_MNINTFLUX_SNPOS'],
        'mxintflux_snpos': ['HOSTLIB_MXINTFLUX_SNPOS'],
        'scale_sersic_size': ['HOSTLIB_SCALE_SERSIC_SIZE'],
        'fixsersic': ['HOSTLIB_FIXSERSIC'],
        'varname_zphot': ['HOSTLIB_VARNAME_ZPHOT'],
        'galid_null': ['HOSTLIB_GALID_NULL'],
        'specbasis_file': ['HOSTLIB_SPECBASIS_FILE'],
        'specdata_file': ['HOSTLIB_SPECDATA_FILE'],
        'galid_unique': ['HOSTLIB_GALID_UNIQUE'],
        'nrepeat_galid_snpos': ['HOSTLIB_NREPEAT_GALID_SNPOS'],
        'snr_scale': ['HOSTLIB_SNR_SCALE'],
        'genzphot_fudgepar': ['HOSTLIB_GENZPHOT_FUDGEPAR'],
        'genzphot_fudgemap': ['HOSTLIB_GENZPHOT_FUDGEMAP'],
        'genzphot_bias': ['HOSTLIB_GENZPHOT_BIAS'],
        'zphoteff_file': ['HOSTLIB_ZPHOTEFF_FILE'],
        'storepar': ['HOSTLIB_STOREPAR'],
        'av_range': ['GENRANGE_AV'],
        'tau_av': ['GENTAU_AV'],
    }

    # Host catalogue and probability weights
    #: Host-galaxy catalogue containing redshifts, sky positions, photometry and
    #: light profiles. Photo-z quantiles use ZPHOT_Qnnn columns specifying
    #: percentiles, or QZPHOT00 to QZPHOTnn at uniformly spaced percentiles. Its
    #: redshift coverage must include GENRANGE_REDSHIFT to within 0.002. Without
    #: a catalogue, events are simulated without hosts.
    hostlib_file: Optional[str] = field(default=None, metadata={'host_only': True})
    #: SNANA HOSTLIB_WGTMAP_FILE; a map of the relative probability of hosting
    #: an event against galaxy properties, and optionally against the event's
    #: own parameters, for simulating a mass step or a stretch-mass
    #: correlation. The map may instead be included in the HOSTLIB itself.
    #: Without one every galaxy in the redshift window is equally likely.
    #: If all candidate weights are zero, the lowest-redshift candidate is used.
    wgtmap_file: Optional[str] = field(default=None, metadata={'host_only': True})
    #: Whether a galaxy falling outside the weight map is given the nearest
    #: edge value rather than aborting the run.
    wgtmap_extrap: int = field(default=0, metadata={'static': True})
    #: Polynomial in redshift, comma separated, giving the half-width of the
    #: redshift window a host is looked for in. The default 0.002 + 0.04z
    #: widens with redshift as the peculiar-velocity and photometric errors
    #: do. No candidate within this window stops the run.
    dztol: str = field(default="0.002,0.04", metadata={'static': True})
    #: Host-library options. See :ref:`sim-host-options`.
    mskopt: int = field(default=0, metadata={'static': True})
    #: Radius in arcseconds of the aperture the host surface brightness is
    #: measured in, which sets how much galaxy light enters the noise of each
    #: observation.
    sbradius: float = field(default=1.2, metadata={'static': True})
    #: SNANA HOSTLIB_SCALE_PROPERTY_ERR; scales on the catalogue errors of the
    #: host properties, written as 'VALUE(NAME)' terms separated by commas,
    #: such as '0.5(LOGMASS),2.0(LOGSFR)'. The measured property is scattered
    #: by the scaled error.
    scale_property_err: str = field(default="", metadata={'static': True})
    #: SNANA HOSTLIB_SCALE_LOGMASS_ERR; scales the host-mass error alone.
    #: Set either this key or HOSTLIB_SCALE_PROPERTY_ERR, not both.
    scale_logmass_err: Any = field(default=None, metadata={'static': True})

    # Host selection and association
    #: Maximum separation in units of the galaxy's directional light
    #: radius, at which a galaxy still
    #: counts as the event's host. An event whose nearest galaxy is further
    #: than this is written as hostless, with HOSTGAL_NMATCH 0 and its other
    #: host columns marked as missing, although SIM_HOSTLIB_GALID still
    #: names the galaxy it was generated in.
    maxddlr: float = field(default=4.0, metadata={'static': True})
    #: Maximum separation in directional light radii at which a neighbouring galaxy is
    #: written in the event's further hosts, HOSTGAL2 and HOSTGAL3, and
    #: counted in HOSTGAL_NMATCH2. A photometric redshift generated for a
    #: galaxy beyond it is set to -9, marking it as unavailable.
    maxddlr2: float = field(default=7.0, metadata={'static': True})
    #: Fewest days apart two events sharing a galaxy must peak, which stops a
    #: galaxy hosting two events the same season. The default imposes no limit,
    #: and it cannot be combined with MSKOPT bit 32. Rejected events free
    #: their hosts at batch boundaries; use batch_size=1 if each host must
    #: become available immediately after a rejection.
    mindaysep_samegal: int = field(default=9_999_999, metadata={'static': True})
    #: Give every event this one galaxy, named by its GALID, to study a single
    #: sightline. A negative value retains random sampling.
    galid_force: int = field(default=-9, metadata={'static': True})
    #: Extra size in arcseconds added in quadrature to each galaxy's half-light
    #: axes before the host match, to represent the seeing that blurs a real
    #: measurement of them.
    smear_sersic: float = field(default=0.0, metadata={'static': True})
    #: A 'min,max' GALID range whose galaxies are preferred over the rest when
    #: a host is chosen. An empty or inverted range prefers nothing.
    galid_priority: str = field(default="0,0", metadata={'static': True})
    #: A table of the probability that the host chosen is the wrong galaxy, and
    #: of the redshift the wrong one would have, for simulating host
    #: misassociation.
    wronghost_file: str = field(default="", metadata={'host_only': True})
    #: SNANA BIASCOR_SALT2GAMMA_GRID; a 'min,max' pair replacing the
    #: catalogue's SNMAGSHIFT with one of the two values by the parity of the
    #: GALID, so that a bias-correction simulation spans a range of the mass
    #: step rather than assuming one.
    biascor_salt2gamma_grid: str = field(default="0,0", metadata={'static': True})
    #: SNANA HOSTLIB_SNR_STORE; the signal-to-noise a galaxy must reach in its
    #: bands to be included in the catalogue. Supply comma-separated
    #: thresholds in descending band signal-to-noise order, as for
    #: HOSTLIB_SNR_DETECT. A galaxy that fails is neither a host nor a
    #: neighbour.
    snr_store: Any = field(default=None, metadata={'static': True})
    #: SNANA HOSTLIB_SNR_DETECT; the signal-to-noise a galaxy must reach in
    #: its bands for host detection. Supply comma-separated thresholds in
    #: descending band signal-to-noise order. A galaxy that fails is not
    #: available as a host, so the event appears hostless.
    snr_detect: Any = field(default=None, metadata={'static': True})
    #: SNANA HOSTLIB_MAG_DETECT; the magnitudes a galaxy must reach in its
    #: brightest bands for host detection, as comma-separated limits from the
    #: brightest band down, the limiting-magnitude form of HOSTLIB_SNR_DETECT.
    mag_detect: Any = field(default=None, metadata={'static': True})
    #: SNANA HOSTLIB_SNR_SCALE(<MJDMIN>:<MJDMAX>); a scale on the catalogue
    #: magnitude errors before the HOSTLIB_SNR_DETECT test, one occurrence per
    #: epoch range named in the parentheses, so the survey's depth can change
    #: with time.
    snr_scale: Any = field(default=None, metadata={'static': True})

    # Photometric redshifts and stored columns
    #: SNANA HOSTLIB_GENZPHOT_FUDGEPAR; five values 'a0 a1 a2 prob_outlier
    #: sig_outlier' generating a photometric redshift instead of reading one
    #: from the catalogue. The core width is a0 + a1(1+z) + a2(1+z)^2, a
    #: fraction prob_outlier of events instead come from a wider Gaussian of
    #: width sig_outlier(1+z), and writing the fifth value as FLAT(z1:z2) makes
    #: those outliers uniform over that range.
    genzphot_fudgepar: Any = field(default=None, metadata={'static': True})
    #: SNANA HOSTLIB_GENZPHOT_FUDGEMAP; the core width tabulated against
    #: redshift as 'z:rms' pairs instead of given by the polynomial. It must
    #: cover GENRANGE_REDSHIFT.
    genzphot_fudgemap: Any = field(default=None, metadata={'static': True})
    #: SNANA HOSTLIB_GENZPHOT_BIAS; polynomial coefficients in the host's true
    #: redshift giving a shift added to the generated photometric redshift, for
    #: a photo-z systematic.
    genzphot_bias: Optional[List[float]] = field(default=None, metadata={'static': True})
    #: SNANA HOSTLIB_ZPHOTEFF_FILE; the probability of obtaining a photometric
    #: redshift at all, against redshift. An event that fails is written with
    #: -9 to mark the missing redshift. The file must cover the redshifts of the HOSTLIB's
    #: galaxies.
    zphoteff_file: Any = field(default=None, metadata={'host_only': True})
    #: SNANA HOSTLIB_STOREPAR; further catalogue columns, comma separated, to
    #: write to the output alongside the ones the simulation needs, as
    #: SIM_HOSTLIB(<column>) in the headers. A column named here is also
    #: available to a VARIATION and to the SIMGEN_DUMP. LOGMASS, LOGSFR,
    #: LOGsSFR and COLOR are written with _TRUE appended, so LOGMASS appears as
    #: LOGMASS_TRUE.
    storepar: List[str] = field(default_factory=list, metadata={'static': True})

    # Host photometry and supernova positions
    #: Magnitude offset applied to every galaxy in the catalogue.
    abmag_offset: float = field(default=0.0, metadata={'static': True})
    #: Give every galaxy this magnitude in every band instead of the
    #: catalogue's own. The negative default retains the catalogue magnitudes.
    abmag_force: float = field(default=-9.0, metadata={'static': True})
    #: Fix the draw that picks which Sersic component of the host an event is
    #: placed in, as a fraction of the cumulative component weight. A negative
    #: value retains random sampling.
    fixran_wgt: float = field(default=-9.0, metadata={'static': True})
    #: Put every event at this fixed fraction of the way through its host's
    #: enclosed light instead of drawing one, so they all lie at the same
    #: radius. A negative value retains random sampling.
    fixran_radius: float = field(default=-9.0, metadata={'static': True})
    #: Put every event at this fixed azimuth about its host, as a fraction of a
    #: full turn.
    fixran_phi: float = field(default=-9.0, metadata={'static': True})
    #: Least enclosed light fraction an event may be drawn at, which keeps
    #: events out of the very centre of their hosts.
    mnintflux_snpos: float = field(default=0.0, metadata={'static': True})
    #: Maximum enclosed light fraction used when drawing the event position.
    mxintflux_snpos: float = field(default=0.99, metadata={'static': True})
    #: SNANA HOSTLIB_SCALE_SERSIC_SIZE; a scale on every galaxy's half-light
    #: axes, applied when the catalogue is read, so it affects the position,
    #: the host match and the galaxy magnitude.
    scale_sersic_size: float = field(default=1.0, metadata={'static': True})
    #: SNANA HOSTLIB_FIXSERSIC; four values 'a b n angle' replacing every
    #: galaxy's light profile with one shape, to isolate the effect of
    #: morphology. A value at or below zero, or below -998 for the index and
    #: the angle, leaves that quantity as the catalogue has it.
    fixsersic: Optional[Tuple[float, float, float, float]] = field(default=None, metadata={'static': True})

    # Sky area, reading limits and column names
    #: SNANA HOSTLIB_GENRANGE_RA; the right-ascension range galaxies may be
    #: drawn from, which restricts the catalogue to the survey's own footprint.
    hostlib_genrange_ra: Tuple[float, float] = field(default=(-999.0, 999.0), metadata={'static': True})
    #: SNANA HOSTLIB_GENRANGE_DEC; the declination range galaxies may be
    #: drawn from, which restricts the catalogue to the survey's own
    #: footprint.
    hostlib_genrange_dec: Tuple[float, float] = field(default=(-999.0, 999.0), metadata={'static': True})
    #: Maximum number of galaxies read from the host catalogue.
    hostlib_maxread: int = field(default=80_000_000, metadata={'static': True})
    #: Name of the catalogue column holding the photometric redshift, for a
    #: catalogue that spells it something other than ZPHOT. Its error column is
    #: looked for under the matching spellings.
    varname_zphot: str = field(default="ZPHOT", metadata={'static': True})
    #: GALID written for an event that ended up with no host.
    galid_null: int = field(default=-9, metadata={'static': True})
    #: A basis of host spectra the catalogue's own coefficients combine into
    #: one spectrum per galaxy, used to contaminate the simulated spectra of
    #: the event.
    specbasis_file: str = field(default="", metadata={'host_only': True})
    #: A table of complete host spectra indexed by the catalogue instead, one
    #: per galaxy. It cannot be combined with the basis file.
    specdata_file: str = field(default="", metadata={'host_only': True})
    #: Replace the catalogue GALID written out with one derived from the
    #: event's own CID, so that no two events appear to share a galaxy.
    galid_unique: int = field(default=0, metadata={'static': True})
    #: Reuse one galaxy, and the position within it, for this many consecutive
    #: events, which generates several events in the same place on the same
    #: host.
    nrepeat_galid_snpos: int = field(default=0, metadata={'static': True})
    #: Displace the event and every candidate host by the same random offset,
    #: drawn uniformly in area out to this radius in degrees, to simulate an
    #: astrometric error common to both.
    mxradius_random_shift: float = field(default=0.0, metadata={'static': True})

    # Turned on by naming any host-dust key
    #: Whether the host reddens its event at all, which no key sets directly:
    #: naming GENRANGE_AV, GENTAU_AV, GENSIGMA_AV, GENRANGE_RV, GENMEAN_RV,
    #: GENSIGMA_RV or either WV07 key turns it on, and otherwise A_V and R_V
    #: retain values that leave the event unreddened by its host.
    dust_enabled: bool = field(default=False, metadata={'static': True})

    # Host extinction
    #: Range in magnitudes the host extinction A_V is drawn over, which is
    #: the key that switches host dust on. The profile over that range is an
    #: exponential of scale GENTAU_AV, a half-Gaussian of width GENSIGMA_AV,
    #: or both where GENRATIO_AV gives the Gaussian's height relative to the
    #: exponential's at zero; a width left out or set to zero leaves that
    #: component out.
    av_range: Tuple[float, float] = field(default=(0.0, 0.0), metadata={'static': True})
    #: Scale length in magnitudes of the exponential part of the host A_V
    #: profile, which is the tail of dusty hosts. A value above 50 with no
    #: GENRATIO_AV draws A_V uniformly over GENRANGE_AV instead.
    tau_av: float = field(default=0.0, metadata={'static': True})
    #: SNANA GENAV_WV07; use the Wood-Vasey 2007
    #: parameterisation of the A_V distribution in place of the
    #: exponential-plus-Gaussian one.
    genav_wv07: int = field(default=0, metadata={'static': True})
    #: SNANA WV07_REWGT_EXPAV; the weight given to that parameterisation's
    #: exponential component, which on its own also selects the Wood-Vasey
    #: profile, even when it is zero.
    wv07_rewgt_expav: float = field(default=-9.0, metadata={'static': True})


@dataclass
class MWDust:
    """Milky Way extinction law and reddening distribution."""

    input_names = {
        'genrange': ['GENRANGE_MWEBV'],
        'rv': ['RV_MWCOLORLAW'],
        'parlist': ['PARLIST_MWCOLORLAW'],
        'sigma_ratio': ['GENSIGMA_MWEBV_RATIO'],
        'sigma': ['GENSIGMA_MWEBV'],
        'shift': ['GENSHIFT_MWEBV', 'FUDGESHIFT_MWEBV'],
        'scale': ['GENSCALE_MWEBV', 'FUDGESCALE_MWEBV'],
    }

    #: SNANA OPT_MWCOLORLAW; Milky Way extinction law. See :ref:`sim-mwlaw-options`.
    dust_law: str = field(default="FITZ99_EXACT", metadata={'static': True})
    #: R_V of the Milky Way colour law OPT_MWCOLORLAW selects.
    rv: float = field(default=3.1, metadata={})
    #: Further parameters of the Milky Way colour law, for the laws that take
    #: any.
    parlist: List[float] = field(default_factory=list, metadata={'static': True})
    #: Source of Milky Way reddening. See :ref:`sim-mwebv-options`.
    opt_mwebv: int = field(default=1, metadata={'static': True})
    #: Error on MWEBV as a fraction of the value, added in quadrature with
    #: GENSIGMA_MWEBV. The -9 default means unset, and OPT_MWEBV 1, the default
    #: source, aborts without it.
    sigma_ratio: float = field(default=-9.0, metadata={'static': True})
    #: Error on MWEBV in magnitudes, which for the non-SIMLIB sources replaces
    #: the fractional one rather than joining it.
    sigma: float = field(default=0.0, metadata={'static': True})
    #: Shift added to MWEBV after it is smeared.
    shift: float = field(default=0.0, metadata={})
    #: Scale applied to MWEBV.
    scale: float = field(default=1.0, metadata={})
    #: Draw MWEBV uniformly from this range instead of looking it up. A
    #: negative lower bound leaves the lookup in place.
    genrange: Tuple[float, float] = field(default=(-9.0, -9.0), metadata={'static': True})
    #: Correct the reported photometry for Milky Way extinction, as an analysis
    #: would, so the cuts see the corrected magnitudes.
    applyflag_mwebv: int = field(default=0, metadata={'static': True})

    @property
    def extinction_applied(self) -> bool:
        """Whether Milky Way extinction is simulated; OPT_MWCOLORLAW 0 or
        OPT_MWEBV 0 turns it off."""
        return self.dust_law != "OFF" and self.opt_mwebv != 0

    def build_law(self):
        """A fresh law instance carrying this configuration (PARLIST included);
        callers prepare it on the grid they evaluate on."""
        law = load_dust_law(self.dust_law)
        if self.parlist:
            law.set_params(self.parlist)
        law.check_rv(self.rv)
        return law


@dataclass
class Instrument:
    """Photometric calibration, noise, saturation and atmospheric effects."""

    input_names = {
        'fudgeshift_zpt': ['FUDGESHIFT_ZPT_FILTER'],
        'fudgescale_fluxerr': ['FUDGESCALE_FLUXERR_FILTER'],
        'fudgescale_fluxerr2': ['FUDGESCALE_FLUXERR2_FILTER'],
        'fudgeshift_lam': ['FUDGESHIFT_LAM_FILTER'],
        'fudge_magerr': ['FUDGE_MAGERR_FILTER'],
        'fudge_zpterr': ['FUDGE_ZPTERR_FILTER'],
        'atmos_optmask': ['ATMOSPHERE_OPTMASK'],
        'atmos_sedstar_file': ['ATMOSPHERE_SEDSTAR_FILE'],
        'coordres_floor': ['ATMOSPHERE_COORDRES_FLOOR'],
        'dcr_coordres_poly': ['ATMOSPHERE_DCR_COORDRES_POLY'],
        'dcr_magshift_poly': ['ATMOSPHERE_DCR_MAGSHIFT_POLY'],
        'fudgescale_noise_sky': ['FUDGESCALE_NOISE_SKY', 'FUDGESCALE_SKYNOISE'],
        'fudgescale_noise_read': ['FUDGESCALE_NOISE_READ', 'FUDGESCALE_READNOISE'],
        'fluxerr_file': ['FLUXERRMODEL_FILE'],
        'fluxerr_redcov': ['FLUXERRMODEL_REDCOV'],
        'fluxerr_snrmin_redcov': ['FLUXERRMODEL_SNRMIN_REDCOV'],
        'fluxerr_ignore_dataerr': ['FLUXERRMAP_IGNORE_DATAERR'],
    }

    # Calibration and noise adjustments
    #: Zero point of the reported FLUXCAL scale, defined by
    #: m = ZP_FLUXCAL - 2.5 log10(FLUXCAL). The magnitude system is set by
    #: the calibration.
    zp_fluxcal: float = field(default=ZEROPOINT_FLUXCAL_SNANA, metadata={'static': True})
    #: Whether the reported flux is scattered about the true one. Zero writes
    #: the noiseless flux with its error; bit 2
    #: also removes the source and host shot noise from the reported error, as
    #: scene-modelling photometry does.
    smearflag_flux: int = field(default=1, metadata={'static': True})
    #: Whether the zero-point jitter of each observation enters the error, as a
    #: bit mask: 1 puts it in the noise actually drawn, 2 puts it in the error
    #: reported alongside the flux.
    smearflag_zeropt: int = field(default=0, metadata={'static': True})
    #: SNANA GENMAG_OFF_ZP; Per-band shifts of the zero point used to turn the
    #: model spectrum into a magnitude. SNANA inputs give one value per
    #: GENFILTERS band, in that order; Python/YAML takes a mapping such as
    #: {g: 0.003, r: 0.0}. This is the survey calibration systematic, and
    #: RANSYSTPAR_SIGSHIFT_ZP draws this shift randomly for each run.
    genmag_off_zp: Dict[str, float] = field(default_factory=dict, metadata={'host_only': True})
    #: SNANA FUDGESHIFT_ZPT, FUDGESHIFT_ZPT_FILTER; a shift in magnitudes of
    #: the SIMLIB's own zero point, which changes the depth of the observations
    #: rather than the calibration of the model. The bare key sets every band
    #: and the _FILTER form, written '<bands> <value>', sets the bands it
    #: names.
    fudgeshift_zpt: Dict[str, float] = field(default_factory=dict, metadata={'host_only': True})
    #: SNANA FUDGESCALE_FLUXERR, FUDGESCALE_FLUXERR_FILTER; a scale on the flux
    #: error, applied to both the noise drawn and the reported error.
    fudgescale_fluxerr: Dict[str, float] = field(default_factory=dict, metadata={'host_only': True})
    #: SNANA FUDGESCALE_FLUXERR2, FUDGESCALE_FLUXERR2_FILTER; scale factor
    #: applied to the reported flux uncertainty, without changing the simulated
    #: photometric noise.
    fudgescale_fluxerr2: Dict[str, float] = field(default_factory=dict, metadata={'host_only': True})
    #: SNANA FUDGESHIFT_LAM, FUDGESHIFT_LAM_FILTER; a shift in Angstroms of the
    #: filter transmission, which is the wavelength calibration systematic.
    #: RANSYSTPAR_SIGSHIFT_LAMFILT draws this shift randomly for each run.
    fudgeshift_lam: Dict[str, float] = field(default_factory=dict, metadata={'host_only': True})
    #: SNANA FUDGE_MAGERR, FUDGE_MAGERR_FILTER; an error in magnitudes added in
    #: quadrature to the flux error, setting a floor on the photometric
    #: precision.
    fudge_magerr: Dict[str, float] = field(default_factory=dict, metadata={'host_only': True})
    #: SNANA FUDGE_ZPTERR, FUDGE_ZPTERR_FILTER; the zero-point uncertainty in
    #: magnitudes, replacing the SIMLIB's own value.
    fudge_zpterr: Dict[str, float] = field(default_factory=dict, metadata={'host_only': True})
    #: Scale factor applied to the PSF width of every observation.
    fudgescale_psf: float = field(default=1.0, metadata={'host_only': True})
    #: SNANA FUDGESCALE_NOISE_SKY/FUDGESCALE_SKYNOISE; a scale on the sky
    #: noise of every observation.
    fudgescale_noise_sky: float = field(default=1.0, metadata={'host_only': True})
    #: SNANA FUDGESCALE_NOISE_READ/FUDGESCALE_READNOISE; a scale on the
    #: detector read noise.
    fudgescale_noise_read: float = field(default=1.0, metadata={'host_only': True})
    #: Scale on the noise contributed by the template image subtracted from
    #: each observation.
    fudgescale_noise_template: float = field(default=1.0, metadata={'host_only': True})
    #: PHOTFLAG bit set on the epoch of highest signal-to-noise.
    photflag_snrmax: int = field(default=0, metadata={'static': True})
    #: PHOTFLAG bit set on the epoch nearest the time of peak.
    photflag_nearpeak: int = field(default=0, metadata={'static': True})

    # Flux-error model
    #: A map of the extra flux error a real survey shows beyond the analytic
    #: noise, read against observing conditions and host surface brightness.
    #: Supplying it sets SIMLIB_MSKOPT bit 32, which ignores any FLUXERR_COR
    #: map in the SIMLIB header.
    fluxerr_file: Optional[str] = field(default=None, metadata={'host_only': True})
    #: Correlation between the bands of one epoch, overriding the map file's
    #: REDCOV entries. If omitted, the file's correlations are retained.
    fluxerr_redcov: Any = field(default=None, metadata={'host_only': True})
    #: Signal-to-noise an observation must reach to join the correlated draw;
    #: observations below this threshold receive independent noise.
    fluxerr_snrmin_redcov: float = 2.0
    #: SNANA FLUXERRMAP_IGNORE_DATAERR; the named maps, comma separated, whose
    #: extra error is drawn into the photometry but left out of the reported
    #: error.
    fluxerr_ignore_dataerr: Any = field(default=None, metadata={'host_only': True})

    # Saturation
    #: Photoelectrons in the brightest pixel above which an observation counts
    #: as saturated and its flux is discarded. The default is high enough that
    #: nothing saturates, and a SIMLIB header may override it.
    npe_pixel_saturate: int = field(default=1_000_000_000, metadata={'static': True})
    #: PHOTFLAG bit set on a saturated epoch.
    photflag_saturate: int = field(default=0, metadata={'static': True})

    # Atmospheric effects and differential chromatic refraction
    #: Atmospheric calculation and output options. See :ref:`sim-atmosphere-options`.
    atmos_optmask: int = field(default=0, metadata={'static': True})
    #: SNANA ATMOSPHERE_SEDSTAR_FILE; the spectrum of the calibration star the
    #: refraction of each event is measured against, as two columns of
    #: wavelength and flux.
    atmos_sedstar_file: Optional[str] = field(default=None, metadata={'host_only': True})
    #: SNANA ATMOSPHERE_DCR_COORDRES_POLY; polynomial coefficients giving the
    #: centroiding error in arcseconds as a function of the PSF width divided
    #: by the signal-to-noise.
    dcr_coordres_poly: List[float] = field(default_factory=list, metadata={'static': True})
    #: SNANA ATMOSPHERE_DCR_MAGSHIFT_POLY; polynomial coefficients giving the
    #: magnitude lost to the aperture as a function of the refraction shift in
    #: units of the PSF width.
    dcr_magshift_poly: List[float] = field(default_factory=list, metadata={'static': True})
    #: SNANA ATMOSPHERE_COORDRES_FLOOR; a floor in arcseconds added in
    #: quadrature to that centroiding error, for the systematic part that does
    #: not fall with signal-to-noise.
    coordres_floor: float = field(default=0.0, metadata={'static': True})
    #: SNANA ATMOSPHERE_SITE_SIGMAS; dispersion of site temperature in Celsius.
    #: The SNANA key takes temperature, pressure and water-vapour dispersions
    #: together, in that order.
    sigma_site_temp: float = field(default=0.0, metadata={'static': True})
    #: SNANA ATMOSPHERE_SITE_SIGMAS; dispersion of site barometric pressure
    #: in mmHg.
    sigma_site_bp: float = field(default=0.0, metadata={'static': True})
    #: SNANA ATMOSPHERE_SITE_SIGMAS; dispersion of water-vapour pressure
    #: in mmHg. Site conditions are read from the survey’s SURVEY.DEF entry.
    sigma_site_pwv: float = field(default=0.0, metadata={'static': True})

    # Fixed, with no input key of its own
    #: First point in rest-frame days of the phase grid the ADDPHASECOR scatter
    #: is drawn on. SNANA's own constant.
    phasecor_phase_min: float = field(default=-18.0, metadata={'static': True})
    #: Spacing in days of that grid.
    phasecor_phase_step: float = field(default=1.0, metadata={'static': True})
    #: Number of points in it.
    phasecor_n_grid: int = field(default=60, metadata={'static': True})


@dataclass
class Lensing:
    """Weak-lensing magnitude offsets."""

    input_names = {
        'probmap_file': ['WEAKLENS_PROBMAP_FILE', 'LENSING_PROBMAP_FILE'],
        'dmuscale': ['WEAKLENS_DMUSCALE', 'LENSING_DMUSCALE'],
        'dmuerr_frac': ['WEAKLENS_DMUERR_FRAC'],
        'dsigmadz': ['WEAKLENS_DSIGMADZ', 'LENSING_DSIGMADZ'],
    }

    #: Probability distribution of weak-lensing magnitude offsets at each
    #: redshift. WEAKLENS_DSIGMADZ takes precedence when enabled.
    probmap_file: Any = field(default=None, metadata={'host_only': True})
    #: Scale applied to the magnitude offsets drawn from the map.
    dmuscale: float = field(default=1.0, metadata={'static': True})
    #: Fractional uncertainty on the reported weak-lensing magnitude offset.
    dmuerr_frac: float = field(default=0.0, metadata={'static': True})
    #: Gaussian width of the weak-lensing magnitude offset per unit redshift.
    #: When enabled, this replaces the probability map.
    dsigmadz: float = field(default=0.0, metadata={'static': True})


@dataclass
class Spectra:
    """Spectrograph configuration and exposure requests."""

    input_names = {
        'warp_spectrum_lampoly': ['WARP_SPECTRUM'],
    }

    #: The spectrograph definition, giving the wavelength bins and the
    #: signal-to-noise the exposure times deliver.
    spectrograph_file: str = ""
    #: Spectrum calculation and output options. See :ref:`sim-spectrograph-options`.
    spectrograph_optmask: int = 0
    #: Scale applied to every spectroscopic exposure time.
    spectrograph_scale_texpose: float = 1.0
    #: Scale applied to the spectroscopic signal-to-noise in each wavelength
    #: bin, written as comma-separated polynomial coefficients in the bin's
    #: wavelength starting at the constant term.
    spectrograph_scale_snr: str = ''
    #: The spectra to take, one entry per TAKE_SPECTRUM line, each giving an
    #: epoch and either an exposure time or a signal-to-noise to reach.
    take_spectrum: List[Dict[str, Any]] = field(default_factory=list)
    #: Scale applied to the template exposure time behind each spectrum.
    take_spectrum_template_texpose_scale: float = 1.0
    #: Fraction of the host galaxy's spectrum added to every simulated spectrum
    #: of the event, as contamination of the slit; the host spectrum comes from
    #: HOSTLIB_SPECBASIS_FILE or HOSTLIB_SPECDATA_FILE.
    take_spectrum_hostfrac: float = 0.0
    #: Host contamination of the simulated spectra given instead as the ratio
    #: of the host's integrated flux to the event's own at peak, from which
    #: the run solves the fraction of the host spectrum to add.
    take_spectrum_hostsnfrac: float = 0.0
    #: A multiplicative warp of every measured spectrum, written as
    #: LAMPOLY(a0,a1,...) for a polynomial in wavelength in SNANA inputs,
    #: or [a0, a1, ...] in Python/YAML, to simulate a
    #: flux-calibration error in the spectrograph; the true spectrum and the
    #: errors written beside it are left alone.
    warp_spectrum_lampoly: List[float] = field(default_factory=list)
    #: Keep the spectra of roughly one event in every this many, per field,
    #: written as '<field>/<prescale>' terms joined by '+', such as
    #: 'C3/4.3+X3/2.2'; a field the list does not name keeps every spectrum,
    #: and host spectra are exempt.
    take_spectrum_prescale: Dict[str, float] = field(default_factory=dict)
    #: Write a diagnostic log of the spectra taken for this CID alone.
    take_spectrum_dumpcid: int = -9
    #: Wavelength bin width in Angstroms for the true-SED spectra. The default
    #: is SNANA's unset value.
    lambin_sed_true: float = -9.0
    #: Wavelength range those spectra span.
    lamrange_sed_true: Optional[Tuple[float, float]] = None
    #: Write a diagnostic table comparing broadband magnitudes with those
    #: reintegrated from the true spectra.
    verify_sed_true: int = 0


@dataclass
class Cuts:
    """Selection-cut (CUTWIN_*) and per-epoch (EPCUTWIN_*) windows.

    `generic_ranges` maps a parameter name to its (lo, hi) window, one entry
    per `CUTWIN_<X>` input line the named fields below do not cover: the
    family is open-ended per SNANA semantics, so it is a typed dict rather
    than a fixed set of fields.
    """

    input_names = {
    }

    #: SNANA APPLY_CUTWIN_OPT; Choose whether cuts reject events. See :ref:`sim-cut-options`.
    apply_cutwin_opt: int = field(default=0, metadata={'static': True})
    #: SNANA CUTWIN_NEPOCH; minimum number of epochs passing the per-epoch
    #: windows and signal-to-noise threshold. During generation, this also
    #: sets the minimum number of scheduled observations in the rest-frame
    #: window, independently of APPLY_CUTWIN_OPT. The SNANA key takes
    #: '<nepoch> <snrmin>', giving the count and threshold together.
    nepoch_min: int = field(default=1, metadata={'static': True})
    #: SNANA CUTWIN_NEPOCH; minimum signal-to-noise for an epoch to count
    #: towards nepoch_min when applying selection cuts.
    nepoch_snr_threshold: float = field(default=float("-inf"), metadata={'host_only': True})
    #: SNANA CUTWIN_SNRMAX; A cut of the form 'snrmax filters logic trest_lo
    #: trest_hi', requiring the peak signal-to-noise in the named bands to
    #: exceed snrmax within that rest-frame window. The logic is AND for every
    #: band named, OR for any one of them, or an integer giving how many must
    #: pass. The key repeats up to five times, and every entry must pass.
    snrmax_cuts: List[Tuple[float, str, int, float, float]] = \
        field(default_factory=list, metadata={'host_only': True})
    #: SNANA CUTWIN_MJDDIF; The separation in days two epochs must have before
    #: the later one counts as a distinct visit under CUTWIN_NOBSDIF.
    mjddif_min: float = field(default=0.0, metadata={'host_only': True})
    #: SNANA CUTWIN_NOBSDIF; The fewest such distinct visits an event must
    #: have, as a cadence requirement.
    nobsdiff_min: int = field(default=0, metadata={'host_only': True})
    #: SNANA CUTWIN_EPOCHS_SNRMIN; A cut on how long an event stays bright,
    #: given as 'snrmin time_max filters': the event passes when the span
    #: between its first and last epoch above snrmin in those bands is shorter
    #: than time_max days, which selects fast transients. An unset time_max is
    #: 99999 days, so the cut passes everything.
    epochs_time_cut: Optional[Dict[str, Any]] = field(default=None, metadata={'host_only': True})
    #: SNANA CUTWIN_TRESTMIN; Rest-frame window in days at least one epoch must
    #: fall in, to require a pre-maximum observation. Its lower bound also
    #: trims the epochs every other cut is counted over.
    trestmin_range: Tuple[float, float] = field(default=(-9999.0, 9999.0))
    #: SNANA CUTWIN_TRESTMAX; Rest-frame window in days at least one epoch must
    #: fall in, to require a post-maximum observation. Its upper bound also
    #: trims the epochs every other cut is counted over.
    trestmax_range: Tuple[float, float] = field(default=(-9999.0, 9999.0))
    #: SNANA CUTWIN_TGAPMAX; Window in rest-frame days the largest gap between
    #: consecutive epochs must fall in, which rejects a light curve broken by a
    #: long interruption.
    tgapmax_range: Optional[Tuple[float, float]] = field(default=None, metadata={'host_only': True})
    #: SNANA CUTWIN_T0GAPMAX; Window in rest-frame days the largest gap
    #: spanning maximum light must fall in, so an event observed either side
    #: of a poorly sampled peak is rejected.
    t0gapmax_range: Optional[Tuple[float, float]] = field(default=None, metadata={'host_only': True})
    #: SNANA CUTWIN_PEAKMAG; Magnitude window at least one band's peak
    #: magnitude must fall in.
    peakmag_range: Optional[Tuple[float, float]] = field(default=None, metadata={'host_only': True})
    #: SNANA CUTWIN_PEAKMAG_ALL; Magnitude window every band's peak magnitude
    #: must fall in, rather than one of them.
    peakmag_all_range: Optional[Tuple[float, float]] = field(default=None, metadata={'host_only': True})
    #: SNANA CUTWIN_PEAKMAG_BYFIELD; A 'lo hi fields' triplet applying the
    #: peak-magnitude window to the named fields alone, so deep and shallow
    #: fields can be cut differently in one run. The key repeats.
    peakmag_byfield: List[Tuple[float, float, str]] = field(default_factory=list, metadata={'host_only': True})
    #: SNANA CUTWIN_NOBS_SATURATE; A 'lo hi filters' triplet requiring the
    #: number of saturated observations in each named band to fall in that
    #: range, for selecting saturated events. The key repeats.
    saturate_ranges: List[Tuple[int, int, str]] = field(default_factory=list, metadata={'host_only': True})
    #: SNANA CUTWIN_NOBS_NOSATURATE; A 'lo hi filters' triplet requiring the
    #: number of unsaturated observations in each named band to fall in that
    #: range. A band with no more observations than the lower bound is skipped
    #: rather than failed, and the key repeats.
    nosaturate_ranges: List[Tuple[int, int, str]] = field(default_factory=list, metadata={'host_only': True})
    #: SNANA CUTWIN_MWEBV; Window the Milky Way E(B-V) of the sightline must
    #: fall in, applied before the event is generated rather than after.
    mwebv_range: Optional[Tuple[float, float]] = field(default=None)
    #: SNANA CUTWIN_<VAR>; A window on any other per-event quantity, named by
    #: the text after the prefix: CUTWIN_REDSHIFT_TRUE, CUTWIN_HOST_ZPHOT and
    #: the parameters the source model draws are all cut this way. The suffix
    #: must name a generated quantity; input validation lists the available
    #: names when needed. CUTWIN_REDSHIFT is read as
    #: CUTWIN_REDSHIFT_TRUE.
    generic_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    #: Rest-frame wavelength range in Angstroms an epoch must sit in to be
    #: counted by any cut, which drops the bands that fall off the model at the
    #: event's redshift.
    epcutwin_lamrest: Tuple[float, float] = field(default=(2000.0, 22000.0))
    #: Signal-to-noise an epoch must reach to be counted by any cut. Only the
    #: lower bound is used.
    epcutwin_snrmin: Tuple[float, float] = field(default=(float("-inf"), float("inf")), metadata={'static': True})


@dataclass
class Output:
    """Output formats, event identifiers and simulation-dump columns."""

    input_names = {
        'prescale_simgen_dump': ['PRESCALE_SIMGEN_DUMP', 'SIMGEN_DUMP_PRESCALE'],
    }

    # Output files
    #: Name of the run, which names the output directory and the files in it.
    #: Required.
    genversion: str = ""
    #: SNANA PATH_SNDATA_SIM; the directory the version is written under;
    #: unset, it is the working directory.
    path_sndata_sim: str = ""
    #: Light-curve format and output options. See :ref:`sim-output-options`.
    format_mask: int = 32
    #: SNANA SIMGEN_DUMP, SIMGEN_DUMPALL, SIMGEN_DUMPADD; The one-row-per-event
    #: dump of the truth. SIMGEN_DUMPALL puts every generated event in it, the
    #: rejected ones included, for a selection-efficiency study.
    simgen_dumpall: bool = False
    #: SNANA SIMGEN_DUMP, SIMGEN_DUMPALL, SIMGEN_DUMPADD; Either key takes the
    #: columns, written as a count followed by that many names or as one
    #: comma-separated list. Model parameters must be requested explicitly,
    #: except for SIMSED, which adds its parameter columns automatically.
    #: FLAG_ACCEPT is always included. SIMGEN_DUMPADD, accepted on the command
    #: line alone, extends a list instead of replacing it.
    simgen_dumpall_columns: List[str] = field(default_factory=list)
    #: SNANA SIMGEN_DUMP, SIMGEN_DUMPALL, SIMGEN_DUMPADD; SIMGEN_DUMP writes
    #: the same table holding the accepted events alone, unless
    #: APPLY_CUTWIN_OPT is 3, which adds the ones the cuts rejected.
    simgen_dump: bool = False
    #: Compress the FITS files once the run has finished.
    gzip_data_files: bool = True
    #: Prefix for the data files inside the output directory, which the
    #: directory's own name supplies when this is unset.
    genprefix: str = ""
    #: Subsampling factor for simulation-dump rows: write one in every N events.
    prescale_simgen_dump: int = 1
    #: Write the spectra of one event in every this many, tested against the
    #: internal CID rather than the written one.
    wrspec_prescale: int = 1
    #: Write the .NOISE dump, breaking each observation's error down into its
    #: photoelectron contributions: 1 covers the epoch nearest peak in each
    #: band, 2 every epoch. The rows cover every event that reached flux
    #: generation, so most rejected events appear.
    simgen_dump_noise: int = 0
    #: Write the .RATE dump, tabulating the volumetric rate against redshift on
    #: a 0.01 grid across GENRANGE_REDSHIFT after the DNDZ reweighting has been
    #: applied.
    simgen_dump_rate: int = 0
    #: Write the .MWCL dump, tabulating Milky Way extinction against wavelength
    #: and R_V for the five colour laws, for comparing laws without running a
    #: simulation.
    simgen_dump_mwcl: int = 0
    #: Write the .TRAINSALT dump, one row per event giving the estimated and the
    #: true time of peak, the input a SALT training expects.
    simgen_dump_trainsalt: int = 0
    #: Write the .YAML job summary a batch pipeline reads to learn how many
    #: events the run produced and how long it took.
    wrflag_yaml_file: int = 0
    #: SNANA WRFLAG_MODELPAR; write SIMSED model parameters to the data files
    #: and the dump.
    wrflag_modelpar: int = 1
    #: Timestamp written into the README instead of the current time, so a run
    #: can reproduce an earlier one byte for byte. The SNANA_TIME_START
    #: environment variable takes precedence.
    time_start: str = ""
    #: Label the written events cyclically from 1 to this number, so a single
    #: run can be split into that many independent subsamples. Values of 1 or
    #: below set the label to -9, meaning no subsample is assigned.
    nsubsample_mark: int = 0
    #: Write the SED template index into the event name, as CID-TEMPLATE_INDEX
    #: rather than the CID alone, which distinguishes the templates of a
    #: multi-template model.
    append_snid_sedindex: int = 0
    #: Write a per-epoch column giving the signal-to-noise a source of this
    #: fixed AB magnitude would have had, to monitor the depth of each
    #: observation. Zero writes no such column.
    magmonitor_snr: int = 0

    # Random CIDs
    #: Lowest CID the random draw may return, for FORMAT_MASK bit 16.
    cidran_min: int = 0
    #: Highest CID the random draw may return. The range must be wide enough
    #: for the run, counting CIDOFF and the skip list, or the run aborts.
    cidran_max: int = 3_000_000
    #: Offset added to every sequential CID, or the number of drawn CIDs
    #: discarded before the run's own, so two runs can have non-overlapping
    #: identifiers. With random CIDs a nonzero offset also changes the run's
    #: other random numbers, so runs sharing a seed simulate different events.
    cidoff: int = 0
    #: CIDs the random draw must not return, as a comma-separated list.
    cidran_skiplist: Tuple[int, ...] = ()


def field_default(cls: type, name: str) -> Any:
    """The default declared on a dataclass field.

    A key's default lives on its field and nowhere else. Where the build must
    compute with a value the input did not supply, it reads it from here rather
    than restating it, including for `default_factory` fields, which a bare
    `cls.name` would return as a Field object.
    """
    f = next(f for f in dataclasses.fields(cls) if f.name == name)
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    return f.default


# FORMAT_MASK bits (snlc_sim.h:113-122).
FORMAT_MASK_VBOSE = 1
FORMAT_MASK_TEXT = 2
FORMAT_MASK_MODEL = 4
FORMAT_MASK_BLINDTEST = 8
FORMAT_MASK_CIDRAN = 16
FORMAT_MASK_FITS = 32
FORMAT_MASK_COMPACT = 64
FORMAT_MASK_ATMOS = 128
FORMAT_MASK_FILTERS = 256
FORMAT_MASK_noSPEC = 2048


def parse_format_mask(mask: int) -> Dict[str, bool]:
    """Decode FORMAT_MASK into per-bit flags (snlc_sim.c:7415-7424)."""
    m = int(mask)
    return {
        'vbose': bool(m & FORMAT_MASK_VBOSE),
        'text': bool(m & FORMAT_MASK_TEXT),
        'model': bool(m & FORMAT_MASK_MODEL),
        'blindtest': bool(m & FORMAT_MASK_BLINDTEST),
        'cidran': bool(m & FORMAT_MASK_CIDRAN),
        'fits': bool(m & FORMAT_MASK_FITS),
        'compact': bool(m & FORMAT_MASK_COMPACT),
        'atmos': bool(m & FORMAT_MASK_ATMOS),
        'filters': bool(m & FORMAT_MASK_FILTERS),
        'nospec': bool(m & FORMAT_MASK_noSPEC),
    }


@dataclass(frozen=True)
class NumericalSettings:
    """Execution size, integration resolution and diagnostic retention."""
    n_sne: Optional[int] = None
    wavelength_grid_points: int = 500
    batch_size: Optional[int] = 1000
    keep_rejected_events: bool = False


@dataclass
class SimulationConfig:
    """Simulation configuration, grouped into model, survey, population,
    calibration and execution settings."""

    numerical: NumericalSettings = field(default_factory=NumericalSettings)
    run: Run = field(default_factory=Run)
    rates: Rates = field(default_factory=Rates)
    cosmology: Cosmology = field(default_factory=Cosmology)
    model: Model = field(default_factory=Model)
    survey: Survey = field(default_factory=Survey)
    host: Host = field(default_factory=Host)
    mw_dust: MWDust = field(default_factory=MWDust)
    instrument: Instrument = field(default_factory=Instrument)
    lensing: Lensing = field(default_factory=Lensing)
    spectra: Spectra = field(default_factory=Spectra)
    cuts: Cuts = field(default_factory=Cuts)
    output: Output = field(default_factory=Output)

    def __post_init__(self):
        # Atmosphere/DCR auto-enables the ATMOS output format and the SED-true
        # spectrograph path that drives the DCR integral (snlc_sim.c:7389).
        if self.instrument.atmos_optmask > 0:
            self.output.format_mask |= 128
            # ZERO_NEGFLAM, in both copies of GENMODEL_MSKOPT
            self.model.genmodel_mskopt |= 16
            self.model._model_input_values['GENMODEL_MSKOPT'] = \
                self.model.genmodel_mskopt
            if self.spectra.lambin_sed_true <= 0.0:
                self.spectra.lambin_sed_true = 10.0

    @property
    def effective_spectrograph_optmask(self) -> int:
        """Spectrograph calculation options, including true SEDs required
        by LAMBIN_SED_TRUE or atmospheric calculations. The configured
        spectrograph_optmask separately controls which spectra are written.
        """
        mask = self.spectra.spectrograph_optmask
        lambin = self.spectra.lambin_sed_true
        if lambin > 0.001 or self.instrument.atmos_optmask > 0:
            return mask | 128
        if lambin >= 0.0:
            return mask & ~128
        return mask

    def model_config(self):
        """The values the source model reads from the simulation settings."""
        return self.model.model_config(self.rates.trest_range)

    def dndz_params(self, entries: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        """Rate-model parameters for the nominal DNDZ population.

        `entries` selects a different population over the same redshift range:
        PEC1A uses this to integrate its separate rate model.
        """
        return parse_dndz_entries(
            self.rates.dndz_entries if entries is None else entries,
            self.rates.redshift_range, self.cosmology.h0,
            self.cosmology)

    def hz_map(self):
        """The HzFUN_FILE H(z) table, or None when the input names no file.

        A value of NONE selects the configured cosmological model instead."""
        c = self.cosmology
        if is_ignorefile(c.hzfun_file):
            return None
        return load_hz_map(c.hzfun_file, c.h0,
                           z_max=self.rates.redshift_range[1])


@dataclass(kw_only=True)
class LightCurveSimulator:
    """Prepared model and survey inputs for the shared numerical kernels.

    The builder fills this object before returning it. Running a simulation
    leaves its inputs unchanged and keeps all run state in local records.
    """

    # generation counts (rate integrals over the resolved config)
    n_simulated: int = field(default=None, metadata={'host_only': True}) # events to generate
    ngen_lc: int = field(default=None, metadata={'host_only': True}) # NGEN_SCALE-scaled accept target; 0 = fixed-count mode
    season_count: float = field(default=None, metadata={'host_only': True}) # expected events per season in GENRANGE(z, MJD, dOmega)

    # the source model on its working grids
    _model: Any
    _static_model: Any = field(default=None, metadata={'static': True})
    wavelength_grid: Any
    phase_range: Tuple[float, float] = field(metadata={'static': True})
    model_wave_range: Tuple[float, float] = field(metadata={'static': True})
    filter_center_range: Tuple[float, float] = field(metadata={'static': True})
    gensmear: Optional[Any]   # prepared wavelength-dependent intrinsic scatter

    # survey identity (SIMLIB header SURVEY after SIMLIB_SURVEY)
    survey_name: str = field(default=None, metadata={'static': True})
    survey_id: int = field(default=None, metadata={'host_only': True})
    site: Optional[Any] = field(default=None, metadata={'host_only': True}) # SURVEY.DEF geo entry, for DCR; often None

    # the SIMLIB, unpacked from the reader
    fields: List[Any] = field(default=None, metadata={'host_only': True})
    flat_field_index: Dict[str, Any] = None
    flat_obs_data: Dict[str, Any] = field(default=None, metadata={'host_only': True})
    field_prescales: np.ndarray = field(default=None, metadata={'host_only': True})
    nea_psf_unit: bool = field(metadata={'static': True})
    simlib_filter_names: List[str] = field(metadata={'static': True})   # the SIMLIB's band chars
    simlib_fluxerr_cor: Optional[Any] = field(default=None, metadata={'host_only': True})
    simlib_fluxerr_add: Optional[Any] = field(default=None, metadata={'host_only': True})
    simlib_template_texpose: float = field(default=None, metadata={'host_only': True})
    simlib_subsurvey_list: tuple = field(metadata={'static': True})     # global-header SUBSURVEY_LIST
    # the entries grouped by CUTWIN_REDSHIFT window, for the entry draw;
    # None when no entry has one
    simlib_windows: Optional[Any]
    field_mwebv: np.ndarray          # per SIMLIB field, for indexing

    # calibration
    kcor: Any = field(default=None, metadata={'host_only': True})
    photometry: Any
    calibration: Optional[Any] = field(default=None, metadata={'host_only': True})

    # filter lookup and model coverage
    filter_names: List[str] = field(metadata={'static': True})          # canonical, in kcor order
    filter_name_width: int = field(metadata={'static': True})           # longest kcor filter name, the BAND column width
    filter_name_to_index: Dict[str, int] = field(metadata={'static': True})
    filter_z_min: Any                # per-filter redshift coverage
    filter_z_max: Any

    # SEARCHEFF maps + effective pipeline knobs
    searcheff_curves: List[Any] = field(default=None, metadata={'host_only': True})
    searcheff_filter_masks: List[int] = field(default=None, metadata={'host_only': True})
    searcheff_photprob_maps: List[Any] = field(default=None, metadata={'host_only': True})
    searcheff_specid_maps: List[Any]
    searcheff_specid_flag: int = field(metadata={'static': True})
    evaluate_redshift_efficiency: bool = field(metadata={'static': True})
    searcheff_zhost_maps: List[Any]
    searcheff_zhost_flag: int = field(metadata={'static': True})
    searcheff_n_trigger_detections: int = field(default=None, metadata={'host_only': True})
    searcheff_trigger_logic: str = field(default=None, metadata={'host_only': True})
    searcheff_photflag_detect: int = field(metadata={'static': True})
    searcheff_photflag_trigger: int = field(metadata={'static': True})
    searcheff_spec_scale: float

    # the loaded HOSTLIB (hostlib.Hostlib), None without one, and the host
    # dust law, which applies whether or not there is one
    hostlib: Optional[Any]
    host_dust_law: Any

    # cosmology tables
    hz_map: Optional[Any]            # HzFUN_FILE (z, E=H/H0) or None
    redshift_cdf_z: Any              # dN/dz CDF grid
    redshift_cdf_vals: Any
    redshift_cdf_vals_pec1a: Optional[Any]  # DNDZ_PEC1A rate, or None
    murewgt: Optional[Any]           # (MU, REWGT CDF) for a reweighted MUSHIFT

    # MW extinction, prepared per wavelength grid
    mw_law: Any                      # model grid
    mw_law_obs: Any                  # common filter grid
    mw_law_at_mean: Any              # filter mean wavelengths
    host_mw_law_at_mean: Any         # ODONNELL94 at R_V = 3.1, hard-wired in SNANA
    sfd_map: Optional[Any] = field(default=None, metadata={'host_only': True}) # None if unreachable
    field_sfd_ebv: Optional[Any]     # SFD98 at each field, or None

    # instrument state: effective saturation values (SIMLIB-header override),
    # the ADDPHASECOR Cholesky factor, and the compiled FLUXERRMODEL
    npe_pixel_saturate: int
    photflag_saturate: int = field(metadata={'static': True})
    phasecor_cholesky_L: Optional[Any]  # None = GENMAG_SMEAR_ADDPHASECOR off
    fluxerr_enabled: bool = field(metadata={'static': True})
    fluxerr_active_vars: Tuple[str, ...] = field(metadata={'static': True})
    fluxerr_is_errscale: bool = field(metadata={'static': True})
    fluxerr_grids: Optional[Any]
    fluxerr_grid_origins: Tuple[float, ...]
    fluxerr_grid_steps: Tuple[float, ...]
    fluxerr_scale_true: Optional[Any]
    fluxerr_scale_data: Optional[Any]
    fluxerr_applies_true: Optional[Any]
    fluxerr_applies_data: Optional[Any]
    fluxerr_map_index_per_field: Optional[Any]
    fluxerr_needs_host_vars: bool = field(metadata={'static': True})
    fluxerr_has_redcov: bool = field(metadata={'static': True})
    fluxerr_redcov_factors: Optional[Any]
    fluxerr_n_factors: int = field(metadata={'static': True})
    # Match counts per field and band, checked only for generated epochs.
    fluxerr_map_counts: Optional[Any] = field(default=None, metadata={'host_only': True})
    fluxerr_redcov_counts: Optional[Any] = field(default=None, metadata={'host_only': True})

    # weak lensing
    lensing_probmap: Optional[Any]

    rates: Rates | None = None
    cosmology: Cosmology | None = None
    model_settings: Model | None = None
    survey: Survey | None = None
    host_settings: Host | None = None
    mw_dust: MWDust | None = None
    instrument: Instrument | None = None
    lensing: Lensing | None = None
    cuts: Cuts | None = None
    magmonitor_snr: int = field(default=0, metadata={'static': True})
    sfd_data: tuple | None = None
    field_data: dict | None = None
    field_start_index: int = 0
    calibration_filter_names: tuple = field(default=(), metadata={'static': True})
    config: SimulationConfig | None = field(default=None, compare=False, repr=False, metadata={'host_only': True})
    photometry_data: Dict = None
    noise: dict | None = None
    zeropoint_error_override: Any = None
    noise_options: tuple = field(default=(), metadata={'static': True})
    band_window: int | None = field(default=None, metadata={'static': True})
    max_obs: int = field(default=0, metadata={'static': True})
    scheduler_tables: Dict = None
    search_efficiency: Any = None
    cutwin: Any = None
    cutwin_options: tuple = field(default=(), metadata={'static': True})
    season_boundaries: Any = None
    photprob_prepared: Any = None
    survey_mjd_span: float = field(default=0.0, metadata={'static': True})
    model_before_host: bool = field(default=False, metadata={'static': True})
    dust_before_host: bool = field(default=False, metadata={'static': True})
    specid_uses_snrsum_rest_v: bool = field(default=False, metadata={'static': True})
    makes_spectra: bool = field(default=False, metadata={'static': True})
    simlib_take_entries: Optional[list] = field(default=None, metadata={'host_only': True})
    atmosphere_static: Optional[tuple] = field(default=None, metadata={'host_only': True})
    n_sne: int = field(default=0, metadata={'host_only': True})
    batch_size: int = field(default=0, metadata={'static': True})
    time_end_init: float = field(default=0.0, metadata={'host_only': True})

    @property
    def model(self):
        """The loaded model; ordinary custom models keep instance-specific JIT."""
        return self._model if self._static_model is None else self._static_model

    def model_config(self):
        """The model's numerical sampling inputs."""
        if self._static_model is not None:
            return self._static_model.config
        return self.model_settings.model_config(self.rates.trest_range)

    @property
    def output(self):
        return self.config.output

    @property
    def run(self):
        return self.config.run


# Numerical records passed to the shared simulator kernels. Setup-only fields
# retain their defaults when JAX reconstructs these records during tracing.

# JAX sees numerical coefficients and structural options, while file readers,
# provenance and output bookkeeping stay on the Python simulator.
for _record in (
    Rates,
    Cosmology,
    Model,
    Survey,
    Host,
    MWDust,
    Instrument,
    Lensing,
    Cuts,
    LightCurveSimulator,
    SubPopulation,
):
    _fields = dataclasses.fields(_record)
    jax.tree_util.register_dataclass(
        _record,
        data_fields=[f.name for f in _fields
                     if not f.metadata.get("static") and not f.metadata.get("host_only")],
        meta_fields=[f.name for f in _fields if f.metadata.get("static")],
        drop_fields=[f.name for f in _fields if f.metadata.get("host_only")],
    )


def run_key(simulation):
    """The root key of an ordinary run's events: RANSEED's, with CIDOFF folded
    in under random CIDs (FORMAT_MASK 16), where SNANA's CID draw takes
    CIDOFF and more randoms off the stream before the first event. The CID
    draw itself stays on `root_key`, so every job sharing a seed draws the
    same CID list."""
    key = root_key(simulation.run.random_seed)
    if simulation.output.format_mask & FORMAT_MASK_CIDRAN and simulation.output.cidoff:
        key = jax.random.fold_in(key, simulation.output.cidoff)
    return key
