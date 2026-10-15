"""The cosmology a simulation was generated with.

A simulation's README records it among its input keys. The light-curve fit
copies it from there into its FITRES header as ``SIM_<key>`` lines, and BBC
copies the bias-correction simulation's into its outputs as
``SIM_<key>(BIASCOR)``, so create_covariance reads it with the distances
instead of looking the simulation up by name (get_cospar_sim,
create_covariance.py:2514).
"""

from pathlib import Path

import yaml

from jax_snlc_core.io.fits_path import open_text

# KEYLIST_COSPAR_SIM (create_covariance.py:287)
SIM_COSMOLOGY_KEYS = ("OMEGA_MATTER", "OMEGA_LAMBDA", "w0_LAMBDA", "wa_LAMBDA",
                      "MUSHIFT")

# a batch job's merged README labels each part's keys; a single run's does not
_INPUT_KEY_BLOCKS = ("INPUT_KEYS_SNIaMODEL00", "INPUT_KEYS")


def parse_mushift(text: str) -> tuple[float, float]:
    """MUSHIFT: ``<value>`` or ``<lo>:<hi>``; a bare value gives lo == hi."""
    lo, _, hi = str(text).strip().partition(':')
    return (float(lo), float(hi or lo))


def read_simulation_cosmology(readme: Path) -> dict[str, str]:
    """The README's cosmology keys as written, or {} for a version that is not
    a simulation.

    MUSHIFT appears only when the simulation set it (sntools_sim_readme.c:848),
    so its absence means no shift.
    """
    with open_text(str(readme)) as stream:
        # BaseLoader keeps every value as its text; the default loader reads
        # some lo:hi ranges as sexagesimal numbers
        docana = yaml.load(stream, Loader=yaml.BaseLoader)["DOCUMENTATION"]
    keys = next((docana[name] for name in _INPUT_KEY_BLOCKS if name in docana), {})
    if "OMEGA_MATTER" not in keys:
        return {}
    return {key: keys.get(key, "0") if key == "MUSHIFT" else keys[key]
            for key in SIM_COSMOLOGY_KEYS}
