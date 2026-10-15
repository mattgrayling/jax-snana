"""The interface BBC (SALT2mu) uses.

A model takes this role by listing :class:`StandardisedModel` among its
bases. BBC works on a FITRES table and never builds a model, so the role is
class-level: declarations and classmethods, resolved from MODEL_LCFIT by
:func:`jax_snlc_core.models.standardised_model`. The model's own BBC input
keys are annotated attributes marked ``BBC_ONLY`` and reach the classmethods
as the ``inputs`` dict the BBC parser collects.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import jax.numpy as jnp


def fitres_aliases_for(names) -> Dict[str, Tuple[str, ...]]:
    """The fitter's own FITRES spellings for LC parameters that are its fit
    parameters: ``<name>``, ``<name>ERR`` and ``COV_<i>_<j>``, the latter
    written in fit-vector order and accepted in either."""
    aliases = {}
    for i, a in enumerate(names):
        aliases[a] = (a,)
        aliases[f"{a}ERR"] = (f"{a}ERR",)
        for b in names[i + 1:]:
            aliases[f"COV_{a}_{b}"] = (f"COV_{a}_{b}", f"COV_{b}_{a}")
    return aliases


class StandardisedModel(ABC):
    """The distance a model's fitted parameters standardise to, the nuisance
    parameters of that standardisation, and how BBC reads the model's FITRES
    columns and bins its biasCor.

    Declarations are plain class attributes (an annotated lowercase attribute
    would be an input key). A model playing the role declares:

    ``m0_reference``
        M0 zero-point, subtracted from ``compute_mu`` to form muFit.
    ``lcpar_names``
        The LC-parameter vector, in order; the first carries sigma_int.
    ``fitres_aliases``
        Each raw input BBC reads -> its FITRES column candidates, first
        present wins, mirroring SNANA's "DLMAG:F MU_LCFIT:F" fallback. A
        model that declares none gets :func:`fitres_aliases_for` its
        ``lcpar_names``, which the default :meth:`read_lcpar` reads.
    ``nuisance_spec``
        Fitted nuisance parameters as name -> (default, lower bound, upper
        bound, parameter number). The number identifies the corresponding
        p<N>/u<N> input keys.
    ``biascor_binning``
        {parameter: (min, max, default_binsize)} for any number of model
        coordinates. Redshift and host mass are supplied by BBC separately.
    ``mucovscale_bins``, ``ccprior_bins``
        Independent {parameter: bin_count} maps. Their ranges are the model's
        default cuts, overridden by named BBC parameter windows. Coordinates
        may be auxiliary quantities outside ``lcpar_names``.
    ``gammadm_coeff``
        The ``biascor_coeffs`` name whose axis midpoint is GAMMADM_OFFSET,
        the value :meth:`compute_mu` subtracts from the mass step. None for a model with no mass step.

    A model whose fitted distance is already a distance modulus declares
    ``m0_reference = 0.0`` and its distance column alone in ``lcpar_names``
    (BayeSN reads ``MU`` from the FITRES column ``DLMAG`` or ``MU_LCFIT``
    through ``fitres_aliases``). It declares an empty ``nuisance_spec``,
    since nothing is fitted globally, and a ``compute_mu`` returning that
    column. It keeps the default ``chi2grid_axes``, so the scan SNANA writes
    for every model comes out flat, and gives ``biascor_binning`` the
    parameters its map is binned in, THETA and AV for BayeSN.
    """

    biascor_binning = {}
    mucovscale_bins = {}
    ccprior_bins = {}

    # Input spelling -> (named configuration field, parameter).
    bbc_input_aliases = {}
    # Parameter -> legacy (selection, bias-grid) CUTMASK bits.
    bbc_cut_bits = {}

    #: The ``biascor_coeffs`` name whose axis midpoint is the GAMMADM_OFFSET
    #: :meth:`compute_mu` subtracts from the mass step; None for a model
    #: with no mass step.
    gammadm_coeff = None

    #: The nuisance parameters the bias-correction map is also gridded in,
    #: one ``(nuisance name, SIM truth column, continuous)`` each: the map
    #: is interpolated at every event's own value of them, the truth column
    #: giving the value the biasCor simulation generated with, on a
    #: two-point grid unless ``continuous`` says the simulation drew it per
    #: event, in which case the axis is a single bin. Empty for a model
    #: standardised without global coefficients.
    biascor_coeffs = ()

    #: Nuisance values held fixed only when constructing the simulated CC prior.
    #: The data likelihood continues to use the fitted values. SALT holds its
    #: redshift and host-mass slopes at zero, as in SNANA's load_DMU_CCprior.
    ccprior_fixed_nuisance = {}

    #: The ``biascor_coeffs`` names a dust-model biasCor is read at the input's
    #: ``p<N>`` values of, rather than at its own per-event truth, since its
    #: own are intrinsic draws.
    dust_forced_coeffs = ()

    #: The SIM truth columns the bias-correction map reads beyond BBC's own,
    #: each mapped to the FITRES spellings accepted for it, first present
    #: winning; :meth:`sim_lcpar` reads them from ``tv.sim_truth``.
    biascor_truth_aliases = {}

    #: The CHI2GRID scan written after the fit, one ``(column label, nuisance
    #: name, (lo, hi))`` per axis. SNANA scans alpha and beta for every model,
    #: so a model without them keeps this default and writes a flat scan, as
    #: SALT2mu does.
    chi2grid_axes = (("alpha", "alpha0", (0.10, 0.20)),
                     ("beta", "beta0", (2.00, 4.00)))

    #: The model's shorthand cut keys, ``{key: (parameter, 'lo' | 'hi')}``,
    #: such as SALT's ``x1min`` and ``cmax``, which the BBC parser resolves
    #: to one bound of that parameter's window.
    cut_aliases = {}

    #: The selection windows ``{parameter: (lo, hi)}`` applied to the fitted
    #: parameters with no CUTWIN given, which the shorthand keys move; the
    #: same windows bound matching error-map and contaminant-prior axes.
    default_cuts = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if 'lcpar_names' in cls.__dict__ and 'fitres_aliases' not in cls.__dict__:
            cls.fitres_aliases = fitres_aliases_for(cls.lcpar_names)

    @classmethod
    @abstractmethod
    def compute_mu(cls, lcpar: Dict, nuis: Dict, host: Dict, u_flags: Dict,
                   apply_mass_step: bool = True):
        """Distance modulus from LC-fit params ``lcpar`` and nuisance params
        ``nuis``. ``host`` carries per-event ``z`` and the mass-step
        variable; ``u_flags`` the input's raw ``u<N>`` flags, whose overloaded
        values the model interprets. ``apply_mass_step=False`` on the
        CC-prior path (no host mass step)."""

    @classmethod
    def biascor_coeff_values(cls, nuis: Dict, host: Dict, u_flags: Dict) -> Tuple:
        """Per-event effective values of the ``biascor_coeffs`` axes; SNANA
        interpolates the biasCor at each event's alpha/beta/gammaDM."""
        return ()

    @classmethod
    def read_lcpar(cls, cols: Dict, inputs: Dict):
        """The LC parameters from the read columns, as ``(lcpar, covariance,
        bad_error)``: ``{name: array}`` per ``lcpar_names``, the per-event
        ``[n, n_lcpar, n_lcpar]`` LC-fit covariance (intrinsic scatter is
        added later), and the events whose LC-fit errors fail the model's
        thresholds, or None. Extra dictionary entries can supply derived map
        coordinates without adding covariance dimensions. BBC loads other
        required coordinates through ``fitres_aliases`` or exact column names.
        ``inputs`` holds the model's BBC keys.

        Default: the LC parameters are the fitter's own, read through
        :func:`fitres_aliases_for`, with no error threshold. SALT overrides
        to derive mB from x0 and apply its thresholds."""
        names = cls.lcpar_names
        n = cols[names[0]].shape[0]
        cov = jnp.zeros((n, len(names), len(names)))
        for i, a in enumerate(names):
            cov = cov.at[:, i, i].set(cols[f"{a}ERR"] ** 2)
            for j, b in enumerate(names[i + 1:], i + 1):
                cov = cov.at[:, i, j].set(cols[f"COV_{a}_{b}"])
                cov = cov.at[:, j, i].set(cols[f"COV_{a}_{b}"])
        return {a: cols[a] for a in names}, cov, None

    @classmethod
    def bbc_bad_covariance(cls, cols, covariance):
        """Events failing covariance-quality cuts, or None for no such cut.

        ``covariance`` is the repaired LC-fit covariance. These cuts remain
        active when BBC refits a selected event list with CUTWIN NONE.
        """
        return None

    @classmethod
    @abstractmethod
    def sim_lcpar(cls, tv) -> Dict:
        """Return true parameter arrays for calculating fitted-minus-true biases.

        The dictionary keys correspond to ``lcpar_names`` and its values
        are NumPy arrays for the BIASCOR sample.

        ``tv`` is the BIASCOR sample read into a
        :class:`~jax_salt2mu.tablevar.TableVar`: its ``sim_truth`` dict holds
        the ``SIM_`` columns named by :attr:`biascor_truth_aliases` and BBC's
        own, its ``lcpar`` the fitted parameters, and its ``mumodel`` the
        reference distance modulus at each event's redshift, which is the
        truth for a model whose fitted parameter is the distance itself.
        """
