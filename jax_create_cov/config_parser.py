"""Reads a create_covariance input YAML into a CreateCovConfig."""

import logging
import os
import re
from pathlib import Path
from typing import Mapping

from jax_snlc_core.utils import SNANAError, snana_log as slog
from jax_snlc_core.io.sim_cosmology import parse_mushift
from jax_snlc_core.utils.config_tracking import UsedTrackingDict, refused

from .config import Covopt, CreateCovConfig
from .inputs import read_yaml
from jax_snlc_core.utils.input_settings import (
    build_dataclass, configuration_settings, derived_changes, merge_settings)

logger = logging.getLogger(__name__)

#: Keys refused with a reason rather than ignored, because ignoring them would
#: silently change the covariance.
_REJECTED_KEYS = {
    **refused("define MUOPT systematic scales in "
              "SYS_SCALE_FILE alongside the FITOPT ones", "MUOPT_SCALES"),
}


# split on whitespace first, as SNANA (:1418)
_BRACKET = re.compile(r"\[(.*)\]")


def parse_covopt(text: str) -> Covopt:
    if isinstance(text, Mapping):
        return build_dataclass(Covopt, text)
    terms = text.split()
    if len(terms) != 2:
        raise SNANAError(
            f"COVOPT must be two bracketed terms, got {text!r}",
            function_name="parse_covopt",
        )
    try:
        label = _BRACKET.findall(terms[0])[0]
        inner = _BRACKET.findall(terms[1])[0]
    except IndexError:
        raise SNANAError(f"COVOPT has unbracketed term: {text!r}",
                         function_name="parse_covopt") from None

    parts = inner.split(",")
    if len(parts) not in (2, 3):
        raise SNANAError(
            f"COVOPT second term takes 2 or 3 comma-separated items, got {text!r}",
            function_name="parse_covopt",
        )
    scale = 1.0
    if len(parts) == 3:
        try:
            scale = float(parts[2])
        except ValueError:
            raise SNANAError(f"COVOPT scale is not a number: {text!r}",
                             function_name="parse_covopt") from None
    return Covopt(label=label.strip(), fitopt_filter=parts[0].strip(),
                  muopt_filter=parts[1].strip(), scale=scale)


def apply_filter(string: str, pattern: str) -> bool:
    """Match a FITOPT/MUOPT label against one covopt filter term (:1392).

    ``=X`` exact, ``+X`` contains, ``-X`` excludes, empty matches everything.
    """
    if isinstance(pattern, Mapping):
        unknown = set(pattern) - {'exact', 'include', 'exclude'}
        if unknown:
            raise SNANAError(f'Unknown systematic selector(s): {sorted(unknown)}')
        label = string.upper()
        return all((label == value.upper() if key == 'exact' else
                    value.upper() in label if key == 'include' else
                    value.upper() not in label) for key, value in pattern.items())
    string, pattern = string.upper(), pattern.upper()
    if pattern.startswith("="):
        return string == pattern[1:]
    if pattern.startswith("+"):
        return pattern[1:] in string
    if pattern.startswith("-"):
        return pattern[1:] not in string
    if pattern == "":
        return True
    raise SNANAError(f"cannot parse COVOPT matching pattern {pattern!r}",
                     function_name="apply_filter")


def load_sys_scales(path: Path) -> dict[str, float]:
    """label -> systematic scale, from a SYS_SCALE_FILE."""
    if not Path(path).is_file():
        raise SNANAError(f"SYS_SCALE_FILE not found: {path}",
                         function_name="load_sys_scales")
    return {str(k): float(v) for k, v in read_yaml(path).items()}


def _resolve(value) -> Path:
    """Expand $VARS and ~; relative paths stay relative to the cwd.

    SNANA expandvars these paths (prep_config, :2792): real configs lean on it,
    Pippin's libraries being written in terms of $PS1_ROOT and $DES_ROOT. It then
    uses them as given, so a relative path means relative to where you ran it.
    """
    return Path(os.path.expandvars(str(value))).expanduser()


def load_config(path: str | Path) -> CreateCovConfig:
    """Read a create_covariance input YAML."""
    return build_config(read_yaml(path))


def build_config(keys: Mapping | CreateCovConfig, overrides: Mapping | None = None) -> CreateCovConfig:
    """Translate file syntax, merge overrides, then validate and load resources."""
    sources = []
    if isinstance(keys, CreateCovConfig):
        keys = configuration_settings(keys)
    for source in (keys, overrides or {}):
        named = {key: value for key, value in source.items()
                 if isinstance(key, str) and key == key.lower()}
        snana = {key: value for key, value in source.items() if key not in named}
        sources.append(merge_settings(snana_settings(snana), named))
    values = merge_settings(*sources)
    bins = values.get('rebin_bins')
    if isinstance(bins, Mapping) and any(isinstance(count, bool) for count in bins.values()):
        raise SNANAError('Rebin counts must be positive integers, not booleans')
    cfg = build_dataclass(CreateCovConfig, values)
    if cfg.mode != 'unbinned' and (cfg.calibrators or cfg.extra_covs):
        raise SNANAError("CALIBRATORS and EXTRA_COVS require mode='unbinned'")
    if not 0 <= cfg.write_mask_cov <= 15:
        raise SNANAError("WRITE_MASK_COV must use only bits 1, 2, 4 and 8")
    if cfg.mxsize_test_posdef < 0:
        raise SNANAError("mxsize_test_posdef must be a non-negative integer")
    for name in ('input_dir', 'outdir', 'sys_scale_file'):
        value = getattr(cfg, name)
        if value is not None:
            setattr(cfg, name, _resolve(value))
    if cfg.sys_scale_file is not None:
        original = cfg.sys_scales.copy()
        cfg.sys_scales = {**load_sys_scales(cfg.sys_scale_file), **cfg.sys_scales}
        cfg._derived_settings.update(derived_changes(
            {'sys_scales': original}, {'sys_scales': cfg.sys_scales}))
    return cfg


def snana_settings(keys: Mapping) -> dict:
    """Translate SNANA YAML syntax without resolving defaults or opening files."""
    raw = UsedTrackingDict(dict(keys))
    found = sorted(set(raw) & set(_REJECTED_KEYS))
    if found:
        raise SNANAError("\n".join(f"{key}: {_REJECTED_KEYS[key]}" for key in found),
                         function_name="build_config")
    values = {}
    for key in ('INPUT_DIR', 'VERSION', 'OUTDIR', 'CALIBRATORS', 'SUBTRACT_VPEC',
                'WRITE_MASK_COV', 'WRITE_FORMAT_COV', 'DOUBLE_PRECISION', 'REBIN_BINS',
                'RADIATION'):
        if key in raw:
            values[key.lower()] = raw[key]
    if values.get('calibrators') is None and 'calibrators' in values:
        values['calibrators'] = []
    if 'COVOPTS' in raw:
        values['covopts'] = [parse_covopt(text) for text in raw['COVOPTS'] or []]
    cosmo = raw.get('COSMOLOGY')
    if cosmo:
        names = {'OMEGA_MATTER': 'omega_matter', 'OMEGA_LAMBDA': 'omega_lambda',
                 'w0_LAMBDA': 'w0', 'wa_LAMBDA': 'wa', 'MUSHIFT': 'mushift'}
        values['cosmology'] = {names.get(key, key): parse_mushift(value) if key == 'MUSHIFT'
                               else value for key, value in cosmo.items()}
    scale_file = raw.get('SYS_SCALE_FILE') or raw.get('SYSFILE')
    if scale_file:
        values['sys_scale_file'] = scale_file
    if 'EXTRA_COVS' in raw:
        extra = []
        for entry in raw['EXTRA_COVS'] or []:
            words = str(entry).split()
            if len(words) != 3:
                raise SNANAError(f"EXTRA_COVS entry needs 'label weight file', got {entry!r}")
            extra.append((words[0], words[1], str(_resolve(words[2]))))
        values['extra_covs'] = extra
    if raw.get('UNBINNED') and 'rebin_bins' not in values:
        values['mode'] = 'unbinned'
    unread = sorted(key for key in raw if key not in raw.used)
    if unread:
        slog.warn(logger, "CREATE_COV", f"unrecognised key(s) ignored: {', '.join(unread)}")
    return values
