"""The host's say over the SN's velocity and redshift:
``transfer_host_redshift`` against SNANA's GEN_SNHOST_VPEC and
TRANSFER_SNHOST_REDSHIFT (sntools_host.c:7464, :9113).

Loads a simulator built from the bundled ``hostlib_e2e`` fixture (HOSTLIB +
SALT3 + LSST SIMLIB + LSST kcor) and drives the transfer with synthetic
params under USEVPEC and SN2GAL_Z.
"""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from jax_snlc_core.cosmology import cosmo_params, distance_modulus, zcmb_to_zhelio
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import open_text
from jax_snlc_sim.hostlib.association import transfer_host_redshift
from jax_snlc_core.random import root_key


C_KM = 2.99792458e5


@pytest.fixture(scope="module")
def sim_setup(fixture_root):
    """Build a small SALT3+HOSTLIB simulator from the bundled hostlib_e2e fixture."""
    from jax_snlc_sim import run as sim_run

    input_path = fixture_root / "snana" / "hostlib_e2e" / "input.input"
    cfg = dict(n_sne=4, random_seed=42)
    _prepared = sim_run.prepare(str(input_path), **cfg)
    simulation = None if _prepared is None else _prepared.config
    resources = _prepared
    return simulation, resources, cfg


def _with_mskopt(simulation, mskopt):
    return dataclasses.replace(
        simulation, host=dataclasses.replace(simulation.host, mskopt=mskopt))


def _build_synthetic_params(simulation, resources, n=4):
    """Synthetic per-SN params at known redshifts, velocities and positions,
    the last event on a wrong host."""
    field_indices = jnp.arange(n, dtype=jnp.int32) % len(resources.fields)
    z_cmb = jnp.array([0.05, 0.20, 0.30, 0.35], dtype=jnp.float32)
    ra = jnp.array([resources.fields[int(i)].ra for i in field_indices],
                   dtype=jnp.float32)
    dec = jnp.array([resources.fields[int(i)].dec for i in field_indices],
                    dtype=jnp.float32)
    vpec = jnp.array([+200.0, -100.0, +400.0, +300.0], dtype=jnp.float32)
    z_hel = (1.0 + zcmb_to_zhelio(z_cmb, ra, dec)) * (1.0 + vpec / C_KM) - 1.0

    return {
        "sim_redshift_cmb": z_cmb,
        "sim_redshift_helio": z_hel,
        "vpec": vpec,
        "vpec_true": vpec,
        "vpec_err": jnp.zeros(n, dtype=jnp.float32),
        "mu": jnp.asarray(_mu(simulation, z_cmb, z_hel, vpec), dtype=jnp.float32),
        "lensdmu": jnp.zeros(n, dtype=jnp.float32),
        "mushift": jnp.zeros(n, dtype=jnp.float32),
        "field_idx": field_indices,
        "ra": ra,
        "dec": dec,
        "CORRECT_HOSTMATCH": jnp.array([1, 1, 1, 0], dtype=jnp.int32),
        "sn_global_idx": jnp.arange(n, dtype=jnp.int32),
    }


def _host_properties():
    return {
        "ZTRUE": jnp.array([0.10, 0.25, 0.30, 0.32], dtype=jnp.float32),
        "VPEC": jnp.array([+500.0, +50.0, -800.0, 0.0], dtype=jnp.float32),
        "VPEC_ERR": jnp.array([250.0, 100.0, 300.0, 50.0], dtype=jnp.float32),
    }


def _mu(simulation, zcmb, zhel, vpec):
    omega_m, _omega_k, omega_l, hubble_distance, w0, wa = cosmo_params(simulation.cosmology)
    return np.array([float(distance_modulus(
        float(zcmb[i]), float(zhel[i]), omega_m, omega_l, hubble_distance,
        w0, wa, float(vpec[i]))) for i in range(len(zcmb))])


def test_usevpec_alone_moves_only_the_velocity(sim_setup):
    """USEVPEC takes the true VPEC and its error from the host and smears
    the measured one, but without SN2GAL_Z or SN2GAL_RADEC the transfer
    returns before touching the redshift or the distance
    (sntools_host.c:9146)."""
    simulation, resources, _ = sim_setup
    simulation = _with_mskopt(simulation, 512)
    params = _build_synthetic_params(simulation, resources)
    before = {k: np.array(params[k]) for k in
              ("sim_redshift_cmb", "sim_redshift_helio", "mu")}
    host = _host_properties()

    simulator = dataclasses.replace(resources, host_settings=simulation.host)
    transfer_host_redshift(simulator, params, host,
                           key=root_key(simulation.run.random_seed))

    np.testing.assert_array_equal(np.array(params["vpec_true"]), np.array(host["VPEC"]))
    np.testing.assert_array_equal(np.array(params["vpec_err"]), np.array(host["VPEC_ERR"]))
    smear = np.array(params["vpec"]) - np.array(host["VPEC"])
    assert not np.allclose(smear, 0.0)
    assert np.all(np.abs(smear) < 6.0 * np.array(host["VPEC_ERR"]))
    for k, v in before.items():
        np.testing.assert_array_equal(np.array(params[k]), v, err_msg=k)


def test_a_wrong_host_transfers_nothing(sim_setup):
    """Under SN2GAL_Z the SNe on their true hosts take the host's redshift,
    while the one on a wrong host keeps its own redshifts and distance."""
    simulation, resources, _ = sim_setup
    simulation = _with_mskopt(simulation, 16)
    params = _build_synthetic_params(simulation, resources)
    before = {k: np.array(v) for k, v in params.items()}
    host = _host_properties()

    simulator = dataclasses.replace(resources, host_settings=simulation.host)
    transfer_host_redshift(simulator, params, host,
                           key=root_key(simulation.run.random_seed))

    ok = before["CORRECT_HOSTMATCH"] == 1
    np.testing.assert_array_equal(np.array(params["sim_redshift_helio"])[ok],
                                  np.array(host["ZTRUE"])[ok])
    for k in ("sim_redshift_cmb", "sim_redshift_helio", "mu"):
        np.testing.assert_array_equal(np.array(params[k])[~ok], before[k][~ok],
                                      err_msg=k)


def _hostlib_with_vpec(fixture_root, out_dir, header_vpec_err=None,
                       vpec_err_column=False):
    """The e2e HOSTLIB with a VPEC column, and the error as a column or as a
    header key."""
    src = fixture_root / "data" / "hostlib" / "JAX_TEST.HOSTLIB"
    lines = []
    with open_text(src) as stream:
        source_lines = stream.read().splitlines()
    for line in source_lines:
        if line.startswith("VARNAMES:"):
            if header_vpec_err is not None:
                lines.append(f"VPEC_ERR: {header_vpec_err}")
            line += " VPEC" + (" VPEC_ERR" if vpec_err_column else "")
        elif line.startswith("GAL:"):
            line += " 300.0" + (" 150.0" if vpec_err_column else "")
        lines.append(line)
    path = out_dir / "VPEC.HOSTLIB"
    path.write_text("\n".join(lines) + "\n")
    return path


def _usevpec_input(fixture_root, out_dir, hostlib):
    """The e2e input with USEVPEC (HOSTLIB_MSKOPT bit 512) over ``hostlib``."""
    text = (fixture_root / "snana" / "hostlib_e2e" / "input.input").read_text()
    kept = [l for l in text.splitlines()
            if not l.startswith(("HOSTLIB_FILE:", "HOSTLIB_MSKOPT:"))]
    kept += [f"HOSTLIB_FILE: {hostlib}", "HOSTLIB_MSKOPT: 770"]
    path = out_dir / "usevpec.input"
    path.write_text("\n".join(kept) + "\n")
    return path


def test_usevpec_without_a_vpec_column_aborts(fixture_root, tmp_path):
    """USEVPEC needs a VPEC column (sntools_host.c:2799-2805)."""
    from jax_snlc_sim import run as sim_run

    hostlib = fixture_root / "data" / "hostlib" / "JAX_TEST.HOSTLIB"
    inp = _usevpec_input(fixture_root, tmp_path, hostlib)
    with pytest.raises(SNANAError, match="required VPEC column"):
        sim_run.prepare(str(inp), **dict(n_sne=4, random_seed=42))


def test_usevpec_without_a_vpec_err_aborts(fixture_root, tmp_path):
    """USEVPEC needs the error as a column or a header key
    (sntools_host.c:2807-2812)."""
    from jax_snlc_sim import run as sim_run

    hostlib = _hostlib_with_vpec(fixture_root, tmp_path)
    inp = _usevpec_input(fixture_root, tmp_path, hostlib)
    with pytest.raises(SNANAError, match="VPEC_ERR not defined"):
        sim_run.prepare(str(inp), **dict(n_sne=4, random_seed=42))


def test_usevpec_takes_the_header_vpec_err(fixture_root, tmp_path):
    """Without a VPEC_ERR column every host carries the header's value
    (GEN_SNHOST_VPEC, sntools_host.c:7477-7483)."""
    from jax_snlc_sim import run as sim_run

    hostlib = _hostlib_with_vpec(fixture_root, tmp_path, header_vpec_err=250.0)
    inp = _usevpec_input(fixture_root, tmp_path, hostlib)
    config = dict(n_sne=20, random_seed=42)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    assert resources.hostlib.vpec_err == 250.0
    results, _, _ = sim_run.run_simulation(sim)
    p = results['parameters']
    assert np.all(np.asarray(p['vpec_true']) == 300.0)
    assert np.all(np.asarray(p['vpec_err']) == 250.0)
    assert not np.allclose(np.asarray(p['vpec']), 300.0)
