"""NON1ASED (non-Ia SED) template model for core-collapse SNe.

Simulation-only model: per-SN template selection by index, phase
interpolation on native grids, distance scaling. The templates the input
selects are loaded from the .SED/.SED.gz files NON1A.LIST names.

Matches SNANA's genmag_NON1ASED.c: get_flux_SEDMODEL() → distance scaling.
Each template keeps its native phase grid (padded to a common max length)
so phase interpolation uses the original data with no resampling loss.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jax_snlc_core import logging as log
from jax_snlc_core.interpolation import linear_between, linear_bin
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import resolve_fits_path

from .roles import SimulatedModel
from .sed_model import (SED_WAVELENGTH_RANGE, SNModel, compute_explosion_tshift,
                        late_time_decay, late_time_tables, read_sed_file, resample_surfaces,
                        uvlam_extrapflux_sedmodel)

logger = logging.getLogger(__name__)

# the rest-frame days a template is read within (genmag_NON1ASED.c:115-116)
NON1ASED_DAY_RANGE = (-150.0, 500.0)


def read_non1a_list(list_filepath: str
                    ) -> Tuple[List[Dict], Optional[Tuple[float, float]]]:
    """Parse a NON1A.LIST template catalogue file (read_NON1A_LIST,
    genmag_NON1ASED.c:554-698).

    Reads ``NON1A:`` and ``PEC1A:`` entries. Both name their SED the same
    way: SNANA resolves either class through getName_SED_FILE_NON1ASED(PATH,
    name), relative to the model directory. ``#``/``!``/``%`` start a
    comment. ``FLUX_SCALE:`` is a running value: each entry keeps the scale
    in force where it appears, so template groups can differ.

    Returns ``(templates, restlambda_range)``: dicts with keys index (int),
    type (str), sed_file (str), is_pec1a (bool), flux_scale (float), and the
    optional ``RESTLAMBDA_RANGE:`` pair (None when absent).
    """
    templates: List[Dict] = []
    restlambda_range = None
    flux_scale = 1.0
    with open(list_filepath, 'r') as f:
        for line in f:
            tokens = line.split()
            for i, token in enumerate(tokens):
                if token[0] in '#!%':
                    tokens = tokens[:i]
                    break
            if not tokens:
                continue
            key = tokens[0]

            if key == 'FLUX_SCALE:':
                flux_scale = float(tokens[1])
                continue
            if key == 'RESTLAMBDA_RANGE:':
                restlambda_range = (float(tokens[1]), float(tokens[2]))
                continue
            if key in ('NON1A:', 'NONIA:', 'NON1ASED:', 'NONIASED:'):
                is_pec1a = False
            elif key in ('PEC1A:', 'PECIA:'):
                is_pec1a = True
            else:
                continue
            if len(tokens) < 4:
                continue

            templates.append({
                'index': int(tokens[1]),
                'type': tokens[2],
                'sed_file': os.path.expandvars(tokens[3]),
                'is_pec1a': is_pec1a,
                'flux_scale': flux_scale,
            })
    return templates, restlambda_range


def template_slots(mxgen, one_based_index):
    """Catalogue slot per event: the first whose cumulative budget MXGEN covers
    its index (pick_NON1ASED, snlc_sim.c:14378)."""
    slots = jnp.searchsorted(jnp.asarray(mxgen), one_based_index, side='left')
    return jnp.clip(slots, 0, len(mxgen) - 1)


@dataclass(eq=False, repr=False, kw_only=True)
class NON1ASEDModel(SNModel, SimulatedModel):
    """NON1ASED template library model.

    Loads the SED templates the input selects from a NON1A.LIST catalogue,
    stores each on its native phase grid (zero-padded to a common max
    length), and provides vectorised SED generation via template index
    lookup + phase interpolation.
    """

    GENMODEL_ALIASES = ('nonia', 'non1a', 'noniased', 'non1ased')
    uses_nonia_rates = True
    non1a_templates = True
    mjd_explode_anchor = 'optmask'

    per_template_wavelength_bounds: jax.Array = None
    wavelength_grid: jax.Array = None
    jax_template_seds: jax.Array = None
    jax_template_phases: jax.Array = None
    jax_template_phase_min: jax.Array = None
    jax_template_phase_max: jax.Array = None
    jax_late_time_slopes: jax.Array = None
    jax_late_boundary_seds: jax.Array = None
    _stack_pos_per_template: jax.Array = None
    _mxgen_per_template: jax.Array = None
    _non1a_config: dict | None = None
    # the SIGMACLIP_MAGSMEAR clip of the per-SN magnitude smear
    _sigmaclip_magsmear: tuple | None = None

    @classmethod
    def load(cls, model_path, config, n_wave_points=500, wavelength_range=None):
        model = cls()
        model._load(model_path, config, n_wave_points, wavelength_range)
        return model

    def _load(self, model_path: str, config, n_wave_points=500,
                 wavelength_range=None):
        """Load the NON1ASED template library from ``model_path``, which holds
        NON1A.LIST and the .SED/.SED.gz template files, reanchor the phase
        grids where asked, extend into the UV, and resample onto the working
        grid. PATH_NON1ASED is the fallback directory searched for a template
        SED file that is not in ``model_path``.
        """
        super().__init__("non1ased", config, n_wave_points,
                         wavelength_range)

        model_dir = Path(os.path.expandvars(model_path)).expanduser()
        self._path_non1ased = (
            Path(os.path.expandvars(self.path_non1ased)).expanduser()
            if self.path_non1ased else None)
        list_file = model_dir / "NON1A.LIST"
        if not list_file.exists():
            raise SNANAError(f"NON1A.LIST not found in {model_dir}",
                             function_name="NON1ASEDModel")

        logger.info(f"Loading NON1ASED templates from {model_dir.name}")

        # RESTLAMBDA_RANGE bounds the read and gates filter centres
        # (genmag_NON1ASED.c:75-76, :293-294)
        self.template_catalog, restlambda_range = read_non1a_list(str(list_file))
        self.filter_center_range = restlambda_range or SED_WAVELENGTH_RANGE
        log.line(logger, f"Found {len(self.template_catalog)} templates in NON1A.LIST")

        # only selected templates are read (genmag_NON1ASED.c:122)
        non1a_cfg = config.get('non1a_config')
        if non1a_cfg is None:
            loaded = list(self.template_catalog)
        else:
            selected = self._listed_indices(non1a_cfg['index_per_template'])
            loaded = [entry for entry in self.template_catalog
                      if entry['index'] in selected]
        n_templates = len(loaded)

        all_phases = []
        all_wavelengths = []
        template_data = []

        for entry in loaded:
            sed_file = Path(entry['sed_file'])
            named = sed_file if sed_file.is_absolute() else model_dir / sed_file
            sed_path = Path(resolve_fits_path(named))
            if not sed_path.exists() and self._path_non1ased:
                # SNANA PATH_NON1ASED: the extra search directory, by basename
                sed_path = Path(resolve_fits_path(
                    self._path_non1ased / sed_file.name))
            if not sed_path.exists():
                raise SNANAError(f"SED file not found: {named}",
                                 function_name="NON1ASEDModel")

            # NON1ASED allows non-uniform day bins (OPTMASK_DAYLIST_SEDMODEL,
            # genmag_NON1ASED.c:79-81)
            phases, wavelengths, flux_grid = read_sed_file(
                sed_path, NON1ASED_DAY_RANGE, self.filter_center_range,
                uniform_days=False)
            template_data.append((phases, wavelengths,
                                  flux_grid * entry['flux_scale']))
            all_phases.append(phases)
            all_wavelengths.append(wavelengths)

        wl_mins = np.array([w.min() for w in all_wavelengths])
        wl_maxs = np.array([w.max() for w in all_wavelengths])
        self._wl_range = (float(wl_mins.min()), float(wl_maxs.max()))
        # each template keeps its own blue edge, so the z-limit is per template
        self.per_template_wavelength_bounds = (wl_mins, wl_maxs)

        grid_sizes = np.array([len(w) for w in all_wavelengths])
        if grid_sizes.min() != grid_sizes.max():
            log.line(logger, f"Found {len(np.unique(grid_sizes))} distinct wavelength grids "
                     f"(sizes {grid_sizes.min()}-{grid_sizes.max()})")
        else:
            log.line(logger, f"Wavelength grid: {grid_sizes[0]} points "
                     f"({self._wl_range[0]:.0f}-{self._wl_range[1]:.0f} A)")

        max_n_phases = max(len(p) for p in all_phases)
        self._max_n_phases = max_n_phases
        n_phase_counts = np.array([len(p) for p in all_phases])
        log.line(logger, f"Native phase grids: {n_phase_counts.min()}-{n_phase_counts.max()} points "
                 f"(padding to {max_n_phases})")

        self._template_native_wl = all_wavelengths
        self._template_phases_padded = []
        self._template_flux_padded = []
        self._template_n_phases = []
        self._template_phase_min = []
        self._template_phase_max = []

        for i, (phases, wavelengths, flux_grid) in enumerate(template_data):
            n_ph = len(phases)
            self._template_phase_min.append(float(phases[0]))
            self._template_phase_max.append(float(phases[-1]))
            self._template_n_phases.append(n_ph)

            # Pad phases with large sentinel (searchsorted stays in bounds)
            padded_phases = np.full(max_n_phases, 1e10, dtype=np.float32)
            padded_phases[:n_ph] = phases

            padded_flux = np.zeros((max_n_phases, len(wavelengths)), dtype=np.float32)
            padded_flux[:n_ph, :] = flux_grid

            self._template_phases_padded.append(padded_phases)
            self._template_flux_padded.append(padded_flux)

        # NON1A numeric INDEX -> stack position, for the loaded templates
        self.index_map = {entry['index']: i for i, entry in enumerate(loaded)}

        log.line(logger, f"Template library: {n_templates} of "
                 f"{len(self.template_catalog)} templates loaded, "
                 f"native phase grids preserved (max {max_n_phases} pts)")

        if self.optmask_t0shift_explode >= 1:
            self._reanchor_to_explosion(self.optmask_t0shift_explode)

        if self.uvlam_extrapflux is not None:
            self._apply_uvlam_extrap(self.uvlam_extrapflux)
        self.resample_to_grid(self.choose_grid())

    # NON1A input-key spellings (parse_input_NON1ASED, snlc_sim.c:3528); the
    # table declares the row columns, the row keys carry one template each.
    NON1A_TABLE_KEYS = ('NON1A_KEYS', 'NONIA_KEYS',
                        'NON1ASED_KEYS', 'NONIASED_KEYS')
    NON1A_ROW_KEYS = ('NON1A', 'NONIA', 'NON1ASED', 'NONIASED')
    PEC1A_ROW_KEYS = ('PEC1A', 'PECIA', 'PEC1ASED', 'PECIASED')
    FILE_ONLY_INPUT_KEYS = NON1A_TABLE_KEYS + NON1A_ROW_KEYS + PEC1A_ROW_KEYS

    @classmethod
    def structured_input(cls, raw):
        """The NON1A_KEYS table: every declared column read wherever
        NON1A_KEYS names it (snlc_sim.c:3559-3591; NKEY defaults to 2, INDEX
        and WGT, :1394), in the order template budgets are assigned."""
        # (field, dtype, default or None if required), snlc_sim.c:3568-3591
        columns = {
            'INDEX':    ('index_per_template',    np.int32,   None),
            'WGT':      ('weight_per_template',   np.float64, None),
            'MAGOFF':   ('magoff_per_template',   np.float32, 0.0),
            'MAGSMEAR': ('magsmear_per_template', np.float32, 0.0),
            'SNTYPE':   ('sntype_per_template',   np.int32,   0),
            'SNTAG':    ('sntype_per_template',   np.int32,   0),
        }
        keys_rows = raw.all(*cls.NON1A_TABLE_KEYS)
        keys_row = keys_rows[-1] if keys_rows else [2, 'INDEX', 'WGT']
        keys_row = keys_row if isinstance(keys_row, list) else [keys_row]
        n_keys = int(keys_row[0])
        # SNANA reads KEYLIST[1..NKEY], so names beyond the count are not read
        names = [str(c) for c in keys_row[1:n_keys + 1]]
        unknown = [c for c in names if c not in columns]
        if unknown:
            raise SNANAError(
                f"Invalid NON1A key(s) {', '.join(unknown)} in NON1A_KEYS. "
                f"Valid keys are {' '.join(columns)}.",
                function_name="structured_input")

        rows = []
        for is_pec1a, spellings in ((False, cls.NON1A_ROW_KEYS),
                                    (True, cls.PEC1A_ROW_KEYS)):
            for values in raw.all(*spellings):
                values = values if isinstance(values, list) else [values]
                if len(values) >= n_keys:
                    rows.append({'is_pec1a': is_pec1a, 'values': values})

        # NON1A_INDEX_FORCE keeps one template, at WGT=1 (snlc_sim.c:3607-3625)
        force = next((raw[k] for k in ('NON1A_INDEX_FORCE', 'NONIA_INDEX_FORCE')
                      if k in raw), None)
        if not rows:
            if force is not None:
                raise SNANAError(
                    f"NON1A_INDEX_FORCE={int(force)} is not among the NON1A "
                    f"templates.", function_name="structured_input")
            return {}

        # sort_NON1ASED (genmag_NON1ASED.c:330): PEC1A after NON1A, stable
        order = np.argsort([e['is_pec1a'] for e in rows], kind='stable')

        declared = {name: names.index(name) for name in columns
                    if name in names}
        index_at, weight_at = declared['INDEX'], declared['WGT']

        if force is not None:
            # INDEX 0 matches any force; last match wins (snlc_sim.c:3610-3623)
            kept = [e for e in rows
                    if int(float(e['values'][index_at])) in (int(force), 0)]
            if not kept:
                raise SNANAError(
                    f"NON1A_INDEX_FORCE={int(force)} is not among the NON1A "
                    f"templates.", function_name="structured_input")
            forced = dict(kept[-1], values=list(kept[-1]['values']),
                          is_pec1a=False)
            forced['values'][index_at] = int(force)
            forced['values'][weight_at] = 1.0
            rows = [forced]
            order = np.array([0])

        def _value(row, name, position):
            value = (row['values'][position] if position < len(row['values'])
                     else columns[name][2])
            if name != 'MAGSMEAR':
                return value
            # MAGSMEAR alone is a sigma pair (split2floats, :3524-3527)
            lo, _, hi = str(value).partition(',')
            if hi and float(hi) != float(lo):
                raise SNANAError(
                    f"asymmetric per-template MAGSMEAR {value} is not "
                    f"implemented.", function_name="structured_input")
            return float(lo)

        def _col(values, dtype):
            return np.array(values, dtype=dtype)[order]

        config = {'ispec1a_per_template': _col([e['is_pec1a'] for e in rows],
                                               bool),
                  'n_templates': len(rows)}
        # SNTYPE and SNTAG share an array, so read every named column first
        for name, (field, dtype, _) in columns.items():
            position = declared.get(name)
            if position is not None and field not in config:
                config[field] = _col([_value(e, name, position) for e in rows],
                                     dtype)
        for name, (field, dtype, default) in columns.items():
            if field not in config:
                config[field] = _col([default] * len(rows), dtype)
        return {'non1a_config': config}

    def _listed_indices(self, indices):
        """The configured NON1A indices as a set; one absent from NON1A.LIST
        is fatal (INDEXVALID, genmag_NON1ASED.c:408-412)."""
        selected = {int(i) for i in indices}
        listed = {entry['index'] for entry in self.template_catalog}
        unknown = sorted(selected - listed)
        if unknown:
            raise SNANAError(
                f"Invalid NON1A index = {unknown[0]} in sim-input file; compare with "
                f"the model's NON1A.LIST.",
                function_name="NON1ASEDModel")
        return selected

    def _budget_templates(self, n_simulated, frac_pec1a):
        """Turn NON1A weights into integer per-template event budgets.

        prep_NON1ASED (genmag_NON1ASED.c:352-418) hands each template a whole
        number of events rather than a sampling probability, and pick_NON1ASED
        then assigns template by event index: the first template whose
        cumulative budget MXGEN reaches that index. Needs the generated count,
        so it runs once n_simulated is resolved, and the budgets live on the
        instance.
        """
        non1a_cfg = self._non1a_config
        if non1a_cfg is None:
            return

        ngentot = int(n_simulated)
        weights = non1a_cfg['weight_per_template']
        ispec1a = non1a_cfg['ispec1a_per_template']
        n_index = len(weights)

        if ispec1a.any() and frac_pec1a <= 0.0:
            raise SNANAError(
                "Generating PEC1A SEDs but PEC1A rate=0. Check that DNDZ_PEC1A "
                "is defined, GENRANGE_REDSHIFT and GENRANGE_PEAKMJD are not "
                "delta functions, and SOLID_ANGLE > 0.",
                function_name="_budget_templates")

        # prep_NON1ASED, genmag_NON1ASED.c:339-366
        if ngentot < n_index:
            ngen = (np.arange(n_index) < ngentot).astype(np.int64)
        else:
            xn = np.zeros(n_index, dtype=np.float64)
            for is_pec, budget in ((False, ngentot * (1.0 - frac_pec1a)),
                                   (True, ngentot * frac_pec1a)):
                m = ispec1a == is_pec
                if m.any() and weights[m].sum() > 0:
                    xn[m] = weights[m] / weights[m].sum() * budget
            ngen = (xn + 0.5).astype(np.int64)

        # The last template takes whatever the rounded budgets left over.
        ngen[-1] = max(ngentot - int(ngen[:-1].sum()), 0)

        self._mxgen_per_template = np.cumsum(ngen)

    def _budget_slots(self, n_sne, global_start_idx, template_budget):
        """Template slot per event: the cumulative-budget search over MXGEN,
        or over the caller's per-batch budget (pick_NON1ASED,
        snlc_sim.c:14378). Redshift class and template assignment must use
        the same slots, so both go through here."""
        if template_budget is None:
            mxgen = self._mxgen_per_template
            ilc = jnp.arange(n_sne) + global_start_idx + 1
        else:
            mxgen, ilc = template_budget, jnp.arange(n_sne) + 1
        return template_slots(mxgen, ilc)

    def pec1a_flags(self, n_sne, global_start_idx, template_budget=None):
        """Per-event PEC1A flag, from the template its index is budgeted to."""
        slots = self._budget_slots(n_sne, global_start_idx, template_budget)
        return jnp.asarray(self._non1a_config['ispec1a_per_template'])[slots]

    def generation_quotas(self, target):
        if self._mxgen_per_template is None:
            return super().generation_quotas(target)
        mxgen = np.asarray(self._mxgen_per_template, dtype=np.int64)
        return np.diff(np.concatenate([[0], mxgen]))

    def replay_params_from_dump(self, dump_columns, head_columns, *, mu,
                                distributions):
        if 'NON1A_INDEX' not in dump_columns:
            raise SNANAError(
                "Missing required NON1ASED column in dump: NON1A_INDEX",
                function_name="replay_params_from_dump")

        non1a_indices = dump_columns['NON1A_INDEX'].astype(int)
        unloaded = sorted(set(non1a_indices.tolist()) - set(self.index_map))
        if unloaded:
            raise SNANAError(
                f"NON1A_INDEX {unloaded[0]} in the dump is not among the "
                f"templates the sim-input file selects.",
                function_name="replay_params_from_dump")
        template_idx = np.array([self.index_map[idx] for idx in non1a_indices],
                                dtype=np.int32)

        # Per-SN MAGOFF from the template table (by NON1A_INDEX);
        # MAGSMEAR is always 0 for replay (random draw not recorded in DUMP)
        n_entries = len(non1a_indices)
        magoff_values = np.zeros(n_entries, dtype=np.float32)
        if self._non1a_config is not None:
            idx_to_magoff = dict(zip(self._non1a_config['index_per_template'],
                                     self._non1a_config['magoff_per_template']))
            magoff_values = np.array(
                [idx_to_magoff.get(int(idx), 0.0) for idx in non1a_indices],
                dtype=np.float32)

        sntype = dump_columns.get('SNTYPE', np.zeros(n_entries)).astype(np.int32)
        return {
            'template_idx': template_idx,
            'magoff': magoff_values,
            'magsmear': np.zeros(n_entries, dtype=np.float32),
            'sntype': sntype,
            'non1a_index': non1a_indices.astype(np.int32),
        }, None

    def configure_generation(self, snana_config, n_simulated, frac_pec1a):
        """Resolve each configured NON1A catalogue INDEX to its SED-stack
        position (an index absent from NON1A.LIST is fatal, matching SNANA's
        INDEXVALID check, genmag_NON1ASED.c:408-412) and budget the
        templates over the generated count."""
        non1a_cfg = snana_config.get('non1a_config')
        self._non1a_config = non1a_cfg
        self._sigmaclip_magsmear = tuple(snana_config['SIGMACLIP_MAGSMEAR'])
        if non1a_cfg is not None:
            selected = self._listed_indices(non1a_cfg['index_per_template'])
            unloaded = sorted(selected - set(self.index_map))
            if unloaded:
                raise SNANAError(
                    f"NON1A index {unloaded[0]} was not among the templates "
                    f"selected when the model was built.",
                    function_name="configure_generation")
            self._stack_pos_per_template = np.array(
                [self.index_map[int(i)]
                 for i in non1a_cfg['index_per_template']], dtype=np.int32)
        self._budget_templates(n_simulated, frac_pec1a)

    def _reanchor_to_explosion(self, optmask):
        """Shift each template's DAY grid so T=0 is the explosion epoch
        (SNANA T0shiftExplode_SEDMODEL, genmag_SEDtools.c). Explosion is the
        earliest day whose previous bin precedes the first wavelength bin to
        exceed 1% (OPTMASK&1) or 0.1% (OPTMASK&2) of the template's global
        flux max. Runs before resample_to_grid, on native-wavelength flux."""
        for i in range(len(self._template_flux_padded)):
            n_ph = self._template_n_phases[i]
            day = self._template_phases_padded[i]
            tshift = compute_explosion_tshift(
                self._template_flux_padded[i][:n_ph], day[:n_ph], optmask)
            day[:n_ph] -= tshift
            self._template_phase_min[i] -= tshift
            self._template_phase_max[i] -= tshift

    def derived_parameters(self, params, keys_for, template_budget=None,
                           **kwargs):
        """NON1ASED per-SN sampling: pick a template by event index, look up
        MAGOFF / MAGSMEAR / SNTYPE for that template, draw a per-SN coherent
        MAGSMEAR offset.
        """
        non1a_cfg = self._non1a_config
        if non1a_cfg is None:
            raise SNANAError(
                "NON1ASED model requires non1a_config (from NON1A_KEYS in input file)",
                function_name="derived_parameters")

        index_arr = jnp.array(non1a_cfg['index_per_template'])
        magoff_arr = jnp.array(non1a_cfg['magoff_per_template'])
        magsmear_arr = jnp.array(non1a_cfg['magsmear_per_template'])
        sntype_arr = jnp.array(non1a_cfg['sntype_per_template'])

        # the batch's events are contiguous
        global_idx = params['sn_global_idx']
        template_sel_idx = self._budget_slots(global_idx.shape[0], global_idx[0],
                                              template_budget)

        stack_pos_arr = jnp.array(self._stack_pos_per_template)
        non1a_indices = index_arr[template_sel_idx]
        template_idx = stack_pos_arr[template_sel_idx]

        magoff = magoff_arr[template_sel_idx]
        magsmear_sigma = magsmear_arr[template_sel_idx]
        sntype = sntype_arr[template_sel_idx]

        # Coherent MAGSMEAR: one truncated normal per SN (SNANA
        # SIGMACLIP_MAGSMEAR), scaled by template sigma
        clip_lo, clip_hi = self._sigmaclip_magsmear
        magsmear_normals = jax.vmap(
            lambda key: jax.random.truncated_normal(key, clip_lo, clip_hi))(
            keys_for('NON1A_MAGSMEAR'))
        magsmear = magsmear_sigma * magsmear_normals

        return {
            'template_idx': template_idx,
            'magoff': magoff,
            'magsmear': magsmear,
            'sntype': sntype,
            'non1a_index': non1a_indices,
            # Catalogue slot, not SED-stack position: entries sharing an SED
            # file collapse in template_idx but need separate NGEN_LC quotas.
            'template_slot': template_sel_idx,
        }

    def _apply_uvlam_extrap(self, uvlam: float) -> None:
        """UVLAM_EXTRAPFLUX on every NON1ASED template.

        One call per template, as SNANA's does (genmag_NON1ASED.c:130). The
        templates have their own native grids, so a value below one template's
        blue edge can still be above another's, and the call is the thing that
        decides.
        """
        n_templates = len(self._template_flux_padded)
        extended_count = 0

        for i in range(n_templates):
            waves = self._template_native_wl[i]
            new_waves, new_flux = uvlam_extrapflux_sedmodel(
                uvlam, waves, self._template_flux_padded[i])
            if len(new_waves) == len(waves):
                continue

            self._template_native_wl[i] = new_waves
            self._template_flux_padded[i] = new_flux
            self.per_template_wavelength_bounds[0][i] = new_waves[0]
            extended_count += 1

        if extended_count > 0:
            wl_mins = self.per_template_wavelength_bounds[0]
            self._wl_range = (min(float(wl_mins.min()), self._wl_range[0]),
                              self._wl_range[1])
            log.line(logger, f"NON1ASED UV extrapolation: extended "
                     f"{extended_count}/{n_templates} templates to {uvlam:.0f} A")

    def resample_to_grid(self, wavelength_grid: jnp.ndarray) -> None:
        """Resample all templates from native wavelengths onto
        ``wavelength_grid``, the simulator's working grid. Only wavelength
        is resampled; phase grids are kept native.
        """
        n_templates = len(self._template_flux_padded)
        n_wl_new = len(wavelength_grid)
        wl_grid_np = np.asarray(wavelength_grid)
        max_n_phases = self._max_n_phases

        log.line(logger, f"Resampling {n_templates} NON1ASED templates to "
                 f"{n_wl_new} wavelengths (native phase grids preserved)...")

        resampled = np.zeros((n_templates, max_n_phases, n_wl_new), dtype=np.float32)
        for i in range(n_templates):
            n_ph = self._template_n_phases[i]
            resampled[i, :n_ph] = resample_surfaces(
                self._template_native_wl[i], self._template_flux_padded[i][:n_ph],
                wl_grid_np)

        phases_stacked = np.stack(self._template_phases_padded)

        self.jax_template_seds = jnp.array(resampled)
        self.jax_template_phases = jnp.array(phases_stacked)
        self.jax_template_phase_min = jnp.array(self._template_phase_min)
        self.jax_template_phase_max = jnp.array(self._template_phase_max)

        late_slopes, late_boundary_seds = late_time_tables(
            phases_stacked, resampled, self._template_n_phases,
            self.minslope_extrapmag_late)

        self.jax_late_time_slopes = jnp.array(late_slopes)
        self.jax_late_boundary_seds = jnp.array(late_boundary_seds)

        self._template_flux_padded = None
        self._template_phases_padded = None
        self._template_native_wl = None

        log.line(logger, f"Resampled template library shape: {resampled.shape}")

    def get_wavelength_range(self) -> Tuple[float, float]:
        return self._wl_range

    def sed_param_names(self):
        return ('template_idx', 'mu')

    def phase_range(self):
        """The union across templates. generate_sed still zeroes each event
        below its own template's DAYMIN and extrapolates above its DAYMAX,
        matching getFluxLam_SEDMODEL's per-template bounds; this wider range
        only decides which epochs are offered a spectrum at all.
        """
        return (float(min(self._template_phase_min)),
                float(max(self._template_phase_max)))

    def generate_sed(self, phase: float, params: Dict[str, float],
                     static_data: Dict) -> jnp.ndarray:
        """The rest-frame SED at ``phase`` (days relative to peak) via
        template lookup, shape (n_wavelengths,) in f_lambda units.
        ``params`` holds 'template_idx' (int) and 'mu' (float). Runs under
        nested vmap (outer over SNe, inner over observations), interpolating
        directly on the template's native phase grid.
        """
        template_seds = self.jax_template_seds
        template_phases = self.jax_template_phases
        template_phase_min = self.jax_template_phase_min
        template_phase_max = self.jax_template_phase_max
        template_idx = params['template_idx']
        distance_scale = jnp.power(10.0, -0.4 * params['mu'])

        phases = template_phases[template_idx]

        phase_lo = template_phase_min[template_idx]
        phase_hi = template_phase_max[template_idx]
        in_range = (phase >= phase_lo)
        # the padding phases past a template's last are 1e10
        idx, frac = linear_bin(jnp.clip(phase, phase_lo, phase_hi), phases)
        sed = linear_between(template_seds[template_idx, idx, :],
                             template_seds[template_idx, idx + 1, :], frac)

        sed = jnp.where(in_range, sed, 0.0)

        sed = late_time_decay(sed, phase, phase_hi, self.jax_late_time_slopes[template_idx],
                              self.jax_late_boundary_seds[template_idx])

        # templates are f_lambda: no /hc
        sed = sed * distance_scale
        return sed

    def get_snana_header_params(self, parameters: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Map internal params to SNANA FITS column names.

        SIM_TEMPLATE_INDEX is the NON1A catalogue index, not the SED-stack
        position (several catalogue entries can share one stack slot)."""
        index = np.asarray(parameters['non1a_index'], dtype=np.int32)
        # the template's type, PEC1A- prefixed (genmag_NON1ASED.c:672)
        type_name = {
            entry['index']: (f"PEC1A-{entry['type']}" if entry['is_pec1a']
                             else entry['type'])
            for entry in self.template_catalog}
        result = {
            'SIM_TEMPLATE_INDEX': index,
            'SIM_TYPE_NAME': np.array([type_name[int(i)] for i in index],
                                      dtype='U12'),
        }
        if 'sntype' in parameters:
            result['SIM_TYPE_INDEX'] = np.asarray(parameters['sntype'], dtype=np.int16)
        return result

    sim_model_index = 10                       # MODEL_NON1ASED
