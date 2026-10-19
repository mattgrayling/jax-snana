"""Fit settings, shared by scripts and the command line.

The SNANA reader keeps its own file and override syntax. Both routes produce
the existing FitConfig; model construction and scientific validation do not
depend on which syntax supplied it.
"""

import argparse
import copy
import dataclasses
import logging
import os
from collections.abc import Mapping
from typing import get_type_hints

from jax_snlc_core.models import (MODEL_REGISTRY, infer_model_type_from_path,
                                  model_input_keys)
from jax_snlc_core.utils.error_handling import SNANAError
from jax_snlc_core.utils.input_settings import (
    add_settings, build_dataclass, convert, dataclass_settings, derived_changes,
    merge_settings, parsed_settings, supplied_values)
from .config import EarlyLC, RequireEpochs, FitConfig, PRIOR_ZERRSCALE_UNSET


def declarations(model_type=None):
    """Find configurable fields on their owners, including the selected model."""
    types, owners, groups = {}, {}, {}
    for group, owner in get_type_hints(FitConfig).items():
        for name, annotation in dataclass_settings(owner)[0].items():
            if name.startswith('_'):
                continue
            if name in types:
                raise SNANAError(f"Input {name!r} is declared by more than one component")
            types[name], owners[name], groups[name] = annotation, owner, group
    if model_type:
        if model_type not in MODEL_REGISTRY:
            raise SNANAError(f"Unknown model type {model_type!r}; choose from {tuple(MODEL_REGISTRY)}")
        owner = MODEL_REGISTRY[model_type]
        for spec in model_input_keys(owner, 'fit').values():
            name = spec.attribute
            if name in types:
                # Existing shared model settings have one configuration owner.
                if types[name] != spec.kind:
                    raise SNANAError(f"Conflicting declarations for {name}")
                continue
            types[name] = get_type_hints(owner, include_extras=True)[name]
            owners[name], groups[name] = owner, '_model_key_values'
    return types, owners, groups


def selected_model(settings):
    """Select a model class without opening its trained assets."""
    return settings.get('model_type') or infer_model_type_from_path(
        os.path.expandvars(settings.get('fitmodel_name') or ''))


def configuration_values(config, *, supplied=True):
    """Return detached flat inputs, including model-owned settings."""
    values = {}
    for field in dataclasses.fields(config):
        record = getattr(config, field.name)
        values.update({name: copy.deepcopy(getattr(record, name))
                       for name in dataclass_settings(type(record))[0]})
    model_values = config.model._model_key_values
    for name, value in model_values.items():
        values[name.lower()] = value
    if supplied:
        values = supplied_values(values, config.fit._derived_settings)
    return values


def supplied_configuration(config):
    """Copy a config with linked-rule changes restored to their supplied values."""
    result = copy.deepcopy(config)
    for group in dataclasses.fields(result):
        record = getattr(result, group.name)
        values = supplied_values(dataclasses.asdict(record), result.fit._derived_settings)
        for name, value in values.items():
            setattr(record, name, value)
    result.fit._derived_settings = {}
    return result


def merge_fit_settings(*sources):
    """Merge fit inputs, matching cut quantities regardless of letter case.

    Normalize each source before merging so later overrides replace the same
    quantity even when its spelling differs between Python, YAML and SNANA.
    """
    normalized = []
    for source in sources:
        values = dict(source)
        for name in ('cutwin', 'epoch_cutwin'):
            if isinstance(values.get(name), Mapping):
                values[name] = {str(key).lower(): value
                                for key, value in values[name].items()}
        normalized.append(values)
    return merge_settings(*normalized)


def build_config(settings=None, *, config=None):
    """Build a typed configuration from supplied inputs without loading files."""
    if settings is not None and not isinstance(settings, Mapping):
        raise SNANAError('Fit settings must be a mapping')
    values = merge_fit_settings(configuration_values(config) if config else {}, settings or {})
    types, _, groups = declarations(selected_model(values))
    unknown = values.keys() - types.keys()
    if unknown:
        raise SNANAError(f"Unknown fit settings: {', '.join(sorted(unknown))}")
    records = {name: {} for name in get_type_hints(FitConfig)}
    if config is not None:
        for group, owner in get_type_hints(FitConfig).items():
            record = getattr(config, group)
            records[group].update({field.name: copy.deepcopy(getattr(record, field.name))
                                   for field in dataclasses.fields(owner)
                                   if field.name.startswith('_')})
    model_values = {}
    for name, value in values.items():
        value = convert(value, types[name], name)
        if groups[name] == '_model_key_values':
            model_values[name.upper()] = value
        else:
            records[groups[name]][name] = value
    if model_values:
        records['model']['_model_key_values'] = model_values
    result = FitConfig(**{group: owner(**records[group])
                          for group, owner in get_type_hints(FitConfig).items()})
    original = configuration_values(result, supplied=False)
    resolve_settings(result)
    resolved = configuration_values(result, supplied=False)
    result.fit._derived_settings = derived_changes(original, resolved)
    return result


def resolve_settings(config):
    """Apply the fitting rules shared by all input formats to a new config."""
    for name, owner in (('earlylc', EarlyLC), ('require_epochs', RequireEpochs)):
        values = getattr(config.selection, name)
        if values is not None:
            setattr(config.selection, name, dataclasses.asdict(build_dataclass(owner, values)))
    fit, model, photoz = config.fit, config.model, config.photoz
    if fit.minos is not None:
        fit.minos = fit.minos.lower()
        if fit.minos in ('none', 'null'):
            fit.minos = None
    if fit.minos not in (None, 'z', 'all'):
        raise SNANAError("minos must be None, 'z', or 'all'")
    if "lcplot" not in config.output.tables:
        config.output.mxlc = 0
    for name in ('batch_size', 'num_nobs_bins', 'n_fit_iterations'):
        if getattr(fit, name) < 1:
            raise SNANAError(f'{name} must be positive')
    if photoz.scale_nbin_course <= 0:
        raise SNANAError('scale_nbin_course must be positive')
    if photoz.dofit_photoz and photoz.opt_photoz == 0:
        photoz.opt_photoz = 1
    if photoz.opt_photoz > 0:
        photoz.dofit_photoz = True
    if photoz.opt_photoz & 8:
        photoz.opt_photoz |= 4
    if photoz.opt_photoz & 2 and photoz.prior_zerrscale == PRIOR_ZERRSCALE_UNSET:
        photoz.prior_zerrscale = 1.0
    gaussian_mu_prior = photoz.prior_muerrscale is not None
    flat_mu_prior = photoz.prior_dmu_range is not None
    if gaussian_mu_prior and flat_mu_prior:
        raise SNANAError('PRIOR_MUERRSCALE and PRIOR_DMU_RANGE are both distance '
                         'priors; set one of them')
    if (gaussian_mu_prior or flat_mu_prior) and not photoz.dofit_photoz:
        raise SNANAError('the distance priors PRIOR_MUERRSCALE and PRIOR_DMU_RANGE '
                         'apply only to photometric-redshift fits')
    if fit.opt_chi2_sigma not in (0, 1):
        raise SNANAError('opt_chi2_sigma must be 0 or 1')
    if photoz.dofit_photoz:
        fit.opt_chi2_sigma = 1
    if not model.use_model_magerr:
        model.fudge_modelerr_scale = 0.0
        fit.use_offdiag_covariance = False
    if model.lfixpar_all or model.lfixpar_adjust:
        fit.maxiter_round1 = fit.maxiter_round2 = 0
        if model.lfixpar_all:
            fit.skip_grid_search = True
    unknown_bits = config.data.opt_setpkmjd & ~(1 | 4 | 16 | 64)
    if unknown_bits:
        logging.getLogger(__name__).warning(
            "OPT_SETPKMJD=%s selects unimplemented bits %s; implemented: "
            "1 (Bazin09), 4 (no abort), 16 (Fmax clump), 64 (MJD_TRIGGER)",
            config.data.opt_setpkmjd, unknown_bits)
    if config.data.has_setpkmjd_trigger() and config.data.photflag_trigger == 0:
        raise SNANAError('photflag_trigger is required when opt_setpkmjd selects MJD_TRIGGER')


def argument_parser(model_type=None, *, input_file=False, add_help=True):
    """Build argparse from component/model declarations, without model I/O."""
    parser = argparse.ArgumentParser(description='Fit supernova light curves.',
                                     allow_abbrev=False, add_help=add_help)
    if input_file:
        parser.add_argument('input_file', nargs='?')
    parser.add_argument('--verbose', '-v', action='store_true')
    types, owners, _ = declarations(model_type)
    add_settings(parser, types, owners)
    return parser


def command_settings(tokens, file_settings):
    """Resolve selectors first, then parse every selected-model option."""
    selectors = argument_parser(add_help=False)
    selected, _ = selectors.parse_known_args(tokens)
    model_type = selected_model(merge_settings(file_settings, vars(selected)))
    parser = argument_parser(model_type)
    args = parser.parse_args(tokens)
    types, owners, _ = declarations(model_type)
    return parsed_settings(args, types, owners)
