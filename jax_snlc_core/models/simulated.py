"""The interface the simulator uses.

A model takes this role by listing :class:`SimulatedModel` among its bases.
The simulator checks for the role once, when it builds the model, and uses
nothing the role does not declare.
"""

from abc import ABC
from dataclasses import dataclass, field
from typing import Annotated, Any, ClassVar, Dict, List, Tuple

import jax.numpy as jnp
import numpy as np

from jax_snlc_core.utils.error_handling import SNANAError

from .base import SIM_ONLY


@dataclass(init=False, eq=False, repr=False, kw_only=True)
class SimulatedModel(ABC):
    """Per-event sampling, the truth columns, and the declarations the
    simulator's parser, builder and writers read. Reads ``self.config``,
    which :class:`SNModel` sets."""

    # an event count the model dictates (SIMSED_GRIDONLY: SEQUENTIAL)
    forced_ngen = None

    #: Scale factor for the maximum permitted generated flux. Models of
    #: intrinsically brighter transients can override this value.
    crazyflux_factor = 1.0

    #: Counts events in the non-Ia entry of DNDZ_SCALE and NGEN_SCALE_NON1A, so
    #: the non-Ia rate scalings apply to them rather than the Ia ones.
    uses_nonia_rates = False

    # events drawn from a NON1A template catalogue (NON1A_MODELFLAG,
    # snlc_sim.c:4806)
    non1a_templates = False

    #: Whether MJD_EXPLODE can be used with this model. Under 'none' it
    #: aborts, 'optmask' needs OPTMASK_T0SHIFT_EXPLODE >= 1 to re-anchor the
    #: template day grid, and 'native' is already anchored on the explosion.
    #: Any other value aborts every simulation with the model.
    mjd_explode_anchor = 'none'

    #: The parameters the simulation input may give a population, by name.
    #: ``GENPEAK_<name>``, ``GENSIGMA_<name>`` and ``GENRANGE_<name>`` set one,
    #: whatever the case of the name, and the drawn value is stored under the
    #: name. A model that reads its parameters from its own input file returns
    #: them from :meth:`structured_input` under this key instead.
    population_parameters = ()

    #: Other spellings the input may use for a population parameter, as
    #: ``{name: (spelling, ...)}``. SALT reads SNANA's ``SALT2x1`` as ``x1``.
    population_aliases = {}

    #: SNANA's values for the population keys an input leaves out, as
    #: ``{name: {'peak': ..., 'sigma': (lo, hi), 'range': (lo, hi)}}``.
    population_defaults = {}

    # (canonical, *alternates) key spellings, merged into the alias table
    INPUT_KEY_ALIASES: ClassVar[Tuple[Tuple[str, ...], ...]] = ()

    # keys SNANA takes from a sim-input file only (check_argv, snlc_sim.c:3475)
    FILE_ONLY_INPUT_KEYS: ClassVar[Tuple[str, ...]] = ()

    #: The model's own columns appended to every SIMGEN_DUMP and, where
    #: ``fits_param_block`` names one, listed in the FITS header, spelled
    #: as :meth:`get_snana_header_params` names them; empty for a model whose
    #: truth columns are asked for by name alone.
    dump_columns = ()

    #: SIMGEN_DUMP names for the model's truth columns, as (name, column) pairs
    #: whose column is a :meth:`get_snana_header_params` key. Each name
    #: makes that column available in the simulation dump.
    SNANA_DUMP_SIM_COLUMNS: ClassVar[Tuple[Tuple[str, str], ...]] = ()

    # per-template (wl_min, wl_max), or None
    per_template_wavelength_bounds = None

    #: SNANA's numerical code for this model, written as SIM_MODEL_INDEX in
    #: every output, where the fitter's USESIM_SNIA and USESIM_NONIA selection
    #: reads it back; 0 for a model SNANA does not have.
    sim_model_index = 0

    # prefix stripped from GENMODEL for SIM_TYPE_NAME (snlc_sim.c:29450)
    sim_type_name_prefix = ''

    #: Whether WRFLAG_MODELPAR controls the model's parameter columns in the
    #: data files and the dump: True lets an input set to 0 drop them, as
    #: SNANA does for SIMSED and LCLIB, whose column sets vary with the model
    #: file, so that a version merging several models keeps one schema.
    #: False writes them whatever the key says.
    writes_model_params = False

    # The primary FITS header names those columns under this prefix, as
    # SNANA does for SIMSED and LCLIB alone (sntools_dataformat_fits.c:1316-1346).
    fits_param_block = ''

    #: File of double-exponential decay parameters continuing the SED beyond
    #: the last phase the model covers, in place of the model's own late-time
    #: behaviour. Read by the simulator only.
    genmodel_extrap_latetime: Annotated[str, SIM_ONLY] = field(
        default="", metadata={"static": True})

    #: Option bits whose meaning each model defines. Read by the simulator
    #: only.
    genmodel_mskopt: Annotated[int, SIM_ONLY] = field(
        default=0, metadata={"static": True})

    #: Directory searched, by file name, for a template SED file that is not
    #: in the model directory. The empty default searches the model directory
    #: only.
    path_non1ased: Annotated[str, SIM_ONLY] = field(default="", metadata={"static": True})

    #: Minimum, in magnitudes per day, of the decay rate fitted to a
    #: template's last phase samples and used to continue its flux past its
    #: end. The default 0.0 sets no minimum.
    minslope_extrapmag_late: Annotated[float, SIM_ONLY] = field(
        default=0.0, metadata={"static": True})

    #: Select the template's phase origin. 1 takes the day before the flux
    #: first exceeds one per cent of the template's peak flux, 2 the same
    #: at 0.1 per cent, and 32 the bolometric peak plus 32 days.
    #: The default 0 keeps the file's own
    #: day zero.
    optmask_t0shift_explode: Annotated[int, SIM_ONLY] = field(
        default=0, metadata={"static": True})

    # (name, aliases) for each population parameter this instance's input
    # declares, which the builder sets; a field so that compiled code
    # rebuilding the model keeps it
    _population: tuple = field(
        default=(), metadata={"static": True, "input": False})

    @classmethod
    def structured_input(cls, raw) -> Dict[str, Any]:
        """Read model-specific tables or groups of input settings.

        ``raw`` is a mapping of parsed input keys to values. Use
        ``raw.all('KEY')`` to read every occurrence of a repeated key in
        input order, or pass several spellings to combine their rows.
        Multi-value rows are lists. Ordinary mapping access reads the last
        value supplied for a key.

        Return a dictionary of model-specific settings. Its entries are
        included in the ``config`` passed to model loading and in the
        ``snana_config`` passed to ``configure_generation``. The default
        returns an empty dictionary.
        """
        return {}

    def configure_generation(self, snana_config: Dict, n_simulated: int,
                             frac_pec1a: float) -> None:
        """Configure parameter sampling from distributions or GENPDF files,
        template selection and WGTMAP weights. For models that assign an
        event count to each template, use the total event count and the
        fraction of peculiar type Ia supernovae to set those counts.

        Called by the simulator only, on a model that is already built.
        Overrides must not mutate template wavelength or phase state.

        Default: no-op.
        """

    def derived_parameters(self, params, keys_for, **kwargs):
        """The model's per-event parameters beyond its population draws:
        values it computes from them, and draws of its own.

        Called once per batch by the simulator's model draw (``draw_model``),
        after it has drawn every population parameter the input gives a
        distribution, in declaration order.

        - ``params``: the batch so far, by name: the redshift, the distance
          modulus ``mu``, each event's ``sn_global_idx`` and the population
          draws. A population parameter the input gives no distribution is
          absent: the model may supply it, and reading it stops the run with
          the keys to give.
        - ``keys_for``: ``(name) -> (n_sne, 2)``, per-SN RNG keys for a named
          random quantity, for the model's own draws (a NON1ASED template
          pick, BayeSN's epsilon).
        - ``template_budget``, by keyword: the per-template event quota of
          this batch, for a model that limits each template's events.

        Returns ``{name: array}``, merged into the event's parameters; with
        the draws it holds every population parameter, which
        :meth:`sn_variables` and the header read, plus whatever
        :meth:`SNModel.generate_sed` reads. Default: nothing.
        """
        return {}

    def get_snana_header_params(self, parameters: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """The model's per-event truth as SNANA's ``SIM_`` columns: the sampled
        params in (by name, numpy arrays of shape (n_sne,)), the FITS header
        columns out. The default writes each population parameter as
        ``SIM_<NAME>``, its name in upper case; a model with other truth to
        record, or whose columns SNANA spells otherwise, overrides it.

        A model that also fits names the column the fitter compares each
        parameter to as ``sim_truth`` on its fit spec, and the fitter copies
        those columns to the FITRES itself; the optional declarations are
        :attr:`FittedModel.SIM_TRUTH_COLUMNS` (columns beyond the spec's),
        :attr:`FittedModel.SNANA_TABLE_SIM_COLUMNS` (SNTABLE renames) and
        :meth:`replay_params_from_dump` (the DUMP columns a replay rebuilds
        params from).
        """
        return {f'SIM_{name.upper()}': parameters[name]
                for name, _ in self._population}

    def pec1a_flags(self, n_sne, global_start_idx, template_budget=None):
        """Per-event PEC1A flag, which a model declaring non1a_templates
        supplies from its catalogue."""
        raise SNANAError(
            f"{type(self).__name__} has no PEC1A templates to flag.",
            function_name="pec1a_flags")

    def apply_host_snpar_overrides(self, params, host_values):
        """USESNPAR: the host's values, by parameter name, replace the drawn
        ones; a model with derived parameters overrides to keep them
        consistent."""
        params.update(host_values)
        return params

    def drawn_parameters(self, populations) -> Tuple[str, ...]:
        """The population parameters the simulator draws, in declaration
        order: those the input gives a distribution, ``populations`` being
        the run's records by name. A model that draws some of them itself
        overrides this."""
        return tuple(name for name, _ in self._population
                     if name in populations)

    def generation_quotas(self, target: int) -> np.ndarray:
        """Per-template event quotas for NGEN_LC accounting; one open quota
        for a model without template budgets."""
        return np.array([target], dtype=np.int64)

    def text_sim_header_lines(self, hcol, params, i) -> List[str]:
        """Extra ``SIM_`` header lines the TEXT writer emits for this model."""
        return []

    def replay_params_from_dump(self, dump_columns, head_columns, *, mu,
                                distributions):
        """Map recorded SIMGEN_DUMP/HEAD columns onto this model's per-event
        sampling parameters, as ``(params, invalid)`` where ``invalid`` (or
        None) marks events with missing recorded parameter values. A dump
        column written under any name :attr:`SNANA_DUMP_SIM_COLUMNS` declares
        arrives under its column as well."""
        raise SNANAError(
            f"Replay override not implemented for {type(self).__name__}.",
            function_name="replay_params_from_dump")

    def sn_variables(self, params: Dict) -> Dict:
        """Per-SN variables a SEARCHEFF map may cut on or a WGTMAP bin on.
        Every population parameter is one, under its name; a model with
        quantities derived from its parameters rather than sampled adds
        them."""
        return {name: jnp.asarray(params[name], dtype=jnp.float32)
                for name, _ in self._population}
