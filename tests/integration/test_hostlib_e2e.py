"""HOSTLIB end-to-end: JAX simulator attaches synthetic host galaxies.

Runs JAX with a HOSTLIB input and asserts:

* every SN gets a non-sentinel host assignment;
* the assigned HOSTGAL_LOGMASS matches the HOSTLIB row for the chosen GALID;
* SIM_REDSHIFT_HOST is consistent with the chosen galaxy's ZTRUE.

Pins HOSTLIB plumbing — config parsing, galaxy matching by redshift, host
property pass-through to columnar parameters.
"""

from pathlib import Path

import numpy as np

from jax_snlc_core.snana_files import open_text
from jax_snlc_sim import run as sim_run


def _load_hostlib(path: Path) -> dict[int, dict[str, float]]:
    """Return ``{galid: {col: value}}`` for the synthetic HOSTLIB."""
    varnames: list[str] | None = None
    rows: dict[int, dict[str, float]] = {}
    with open_text(path) as stream:
        lines = stream.read().splitlines()
    for line in lines:
        if line.startswith("VARNAMES:"):
            varnames = line.split(":", 1)[1].split()
        elif line.startswith("GAL:"):
            assert varnames is not None
            parts = line.split()[1:]
            row = {}
            for name, val in zip(varnames, parts):
                try:
                    row[name] = float(val)
                except ValueError:
                    row[name] = val
            rows[int(row["GALID"])] = row
    return rows


def test_hostlib_jax_e2e(snana_fixture, jax_test_hostlib_path):
    """Every JAX SN in the fixture gets a non-sentinel host assignment with consistent properties."""
    fx = snana_fixture("hostlib_e2e")

    config = dict(
        n_sne=300, random_seed=122459,
    )
    sim = sim_run.prepare(str(fx.input_file), **config)
    results, _, _ = sim_run.run_simulation(sim)
    params = results["parameters"]

    n_sne = len(params["sim_redshift_cmb"])
    assert n_sne == 300

    galid = np.asarray(params.get("host_GALID", params.get("host_galid")))
    logmass = np.asarray(
        params.get("host_LOGMASS", params.get("LOGMASS"))
    )
    z_host = np.asarray(
        params.get("host_ZTRUE", params.get("sim_redshift_host"))
    )
    z_sn = np.asarray(params["sim_redshift_cmb"])

    assigned = galid > 0
    assert int(assigned.sum()) == n_sne, (
        f"missing host assignments: {n_sne - int(assigned.sum())}/{n_sne}"
    )

    hostlib = _load_hostlib(jax_test_hostlib_path)
    valid_ids = set(hostlib.keys())
    n_unknown = int(sum(int(g) not in valid_ids for g in galid))
    assert n_unknown == 0, f"{n_unknown} GALIDs not in HOSTLIB"

    # LOGMASS must match the HOSTLIB row exactly for every SN.
    expected_logmass = np.array([hostlib[int(g)]["LOGMASS"] for g in galid])
    diff = np.abs(logmass - expected_logmass)
    n_bad = int((diff >= 1e-3).sum())
    assert n_bad == 0, (
        f"{n_bad}/{n_sne} SNe have LOGMASS not matching HOSTLIB row "
        f"(max |diff|={diff.max():.4f})"
    )

    # SN redshift within DZTOL=0.05 of host ZTRUE.
    dz = np.abs(z_sn - z_host)
    assert float(np.max(dz)) < 0.05 + 5e-3, (
        f"max |zSN - zHOST| = {float(np.max(dz)):.4f} > DZTOL"
    )

    # Hosts span a non-trivial subset of the library — pins that matching
    # isn't degenerate (single galaxy used for everything).
    n_unique_hosts = len(set(int(g) for g in galid))
    assert n_unique_hosts >= 50, f"only {n_unique_hosts} unique hosts assigned"
