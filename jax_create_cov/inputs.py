"""Discovery of a submit_batch BBC output tree: the grid files and their scales."""

from pathlib import Path

import yaml

from jax_snlc_core.io.sim_cosmology import SIM_COSMOLOGY_KEYS, parse_mushift
from jax_snlc_core.utils import SNANAError

from .config import Cosmology, CreateCovConfig


# create_covariance skips these while globbing the grid (:695)
_SKIP = ("~", "wfit_", "cospar")

# FITOPT and MUOPT rows put the label in different columns (get_sys_scales, :1024)
_LABEL_COLUMN = {"FITOPT": 2, "MUOPT": 1}

REF = 0  # FITOPT000/MUOPT000 is the reference every contribution differences against


def grid_files(config: CreateCovConfig) -> dict[str, Path]:
    """Map ``FITOPT<i>_MUOPT<j>`` to its Hubble-diagram file.

    Binned reads ``.M0DIF``; unbinned and rebin both read ``.FITRES`` (:713-717).
    """
    suffix = "M0DIF" if config.mode == "binned" else "FITRES"
    folder = config.data_dir
    found = {}
    for name in sorted(p.name for p in folder.glob("FITOPT*MUOPT*")):
        if any(s in name for s in _SKIP):
            continue
        stem = name
        for drop in (".gz", f".{suffix}"):
            stem = stem.replace(drop, "")
        if f".{suffix}" not in name:
            continue
        found[stem] = folder / name
    if not found:
        raise SNANAError(
            f"no FITOPT*MUOPT*.{suffix} files under {folder}",
            function_name="grid_files",
        )
    return found


def split_name(name: str) -> tuple[int, int]:
    """``FITOPT003_MUOPT002`` -> (3, 2)."""
    fit, _, mu = name.partition("_")
    return int(fit.replace("FITOPT", "")), int(mu.replace("MUOPT", ""))


def join_name(fitopt: int, muopt: int) -> str:
    return f"FITOPT{fitopt:03d}_MUOPT{muopt:03d}"


def read_yaml(path) -> dict:
    """A YAML file's contents. An empty file aborts, where SNANA's read_yaml
    returns None and fails at its first use (:312)."""
    values = yaml.safe_load(Path(path).read_text())
    if values is None:
        raise SNANAError(f"{path} is empty", function_name="read_yaml")
    return values


def read_submit_info(input_dir: Path) -> dict:
    path = Path(input_dir) / "SUBMIT.INFO"
    if not path.is_file():
        raise SNANAError(f"no SUBMIT.INFO in {input_dir}",
                         function_name="read_submit_info")
    return read_yaml(path)


def sys_scales(which_opt: str, submit_info: dict,
               scales: dict[str, float]) -> dict[int, tuple[str, float]]:
    """Map each FITOPT/MUOPT number to its ``(label, scale)`` (:1011).

    Index 0 is always ``('DEFAULT', 1.0)``: it is the reference the others are
    differenced against, so scaling it would be meaningless.
    """
    column = _LABEL_COLUMN[which_opt]
    rows = submit_info.get(f"{which_opt}_OUT_LIST")
    if rows is None:
        return {REF: ("DEFAULT", 1.0)}

    result = {}
    for row in rows:
        number = int(str(row[0]).replace(which_opt, ""))
        label = str(row[column])
        result[number] = (label, 1.0)
    return apply_sys_scales(result, scales)


def apply_sys_scales(options, scales):
    """Apply label overrides; the reference always has unit scale."""
    result = {number: (label, float(scales.get(label, scale)))
              for number, (label, scale) in options.items() if number != REF}
    result[REF] = ("DEFAULT", 1.0)
    return result


def version_photometry(header: dict[str, str]) -> dict[str, str]:
    """The VERSION_PHOTOMETRY keys of a grid FITRES's comment header (:2408).

    SNANA reads the value as ``wdlist[2]`` and raises IndexError when a key has
    none, which SALT2mu writes whenever its input carried no version. jax
    records the empty string instead.
    """
    return {key: value.split()[0] if value else ""
            for key, value in header.items() if "VERSION_PHOTOMETRY" in key}


def isdata_real(header: dict[str, str]) -> int:
    """ISDATA_REAL from a grid file's comment header, -1 when absent (:531)."""
    return int(header["ISDATA_REAL"].split()[0]) if "ISDATA_REAL" in header else -1


def reference_cosmology(given: Cosmology | None,
                        header: dict[str, str]) -> Cosmology | None:
    """The biasCor simulation's cosmology: as BBC records it in its output
    header, or the input's ``COSMOLOGY`` for an output that records none.
    Given both, they must agree."""
    if "SIM_OMEGA_MATTER(BIASCOR)" not in header:
        return given
    value = {key: header[f"SIM_{key}(BIASCOR)"] for key in SIM_COSMOLOGY_KEYS}
    recorded = Cosmology(omega_matter=float(value["OMEGA_MATTER"]),
                         omega_lambda=float(value["OMEGA_LAMBDA"]),
                         w0=float(value["w0_LAMBDA"]), wa=float(value["wa_LAMBDA"]),
                         mushift=parse_mushift(value["MUSHIFT"]))
    if given is not None and given != recorded:
        raise SNANAError(
            f"COSMOLOGY {given} disagrees with the biasCor simulation's, "
            f"{recorded}, recorded in the BBC output",
            function_name="reference_cosmology")
    return recorded
