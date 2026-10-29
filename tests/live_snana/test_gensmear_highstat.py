"""High-statistics G10/C11 scatter parity against live SNANA.

The statistical form of the G10/C11 check, at 10k SNe (~33k (SN, band) draws per
model), resolving the scatter amplitude to ~0.4% — the level at which the
SIGMACLIP_MAGSMEAR truncation bug was found. Requires --live-snana.

Run this when SNANA is upgraded, or to re-derive the scatter agreement; it is not
routine coverage. tests/parity/test_gensmear_ranfix.py guards day to day by pinning
the smear randoms, which is deterministic and takes seconds, but cannot see the
+/-3 sigma truncation or which random feeds which node. Those two are what this
test exists for.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.io import fits

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for extra in ("tests/parity", "."):
    p = str(PROJECT_ROOT / extra)
    if p not in sys.path:
        sys.path.insert(0, p)

from jax_snlc_core.snana_files import load_snana_dump  # noqa: E402
from replay_snana_dump import build_override_parameters  # noqa: E402
from jax_snlc_sim import run as sim_run  # noqa: E402
from jax_snlc_sim.output import save_snana_outputs
from _replay_helpers import load_snana_phot, load_jax_phot  # noqa: E402

_N_MC = 250
_CHUNK = 50
_MIN_BIN = 100


def _entry_moments(n_randoms, delta_mag, weights, lam_rest, zero_coherent):
    """Predicted (mean, std) of the band-integrated smear offset per entry.

    Each entry draws its own randoms. Sharing one draw-set across entries makes the
    MC error common-mode, so it survives into the aggregate pull statistics instead
    of cancelling; measured on 2000 entries, independent draws at 250 per entry gave
    7x less seed-to-seed movement than 10000 shared, and ran 34x faster.
    """
    def one_entry(w, lam, key):
        # SNANA SIGMACLIP_MAGSMEAR default: smear randoms are +/-3 sigma truncated
        randoms = jax.random.truncated_normal(
            key, -3.0, 3.0, shape=(_N_MC, n_randoms))
        if zero_coherent:
            randoms = randoms.at[:, n_randoms - 1].set(0.0)
        dm = jax.vmap(delta_mag, in_axes=(0, None))(randoms, lam)
        ratio = jnp.sum(w * jnp.power(10.0, -0.4 * dm), axis=1) / jnp.sum(w)
        mags = -2.5 * jnp.log10(ratio)
        return jnp.mean(mags), jnp.std(mags)

    one_entry = jax.jit(jax.vmap(one_entry))
    keys = jax.random.split(jax.random.PRNGKey(0), len(weights))
    means, stds = [], []
    for i in range(0, len(weights), _CHUNK):
        m, s = one_entry(jnp.asarray(weights[i:i + _CHUNK]),
                         jnp.asarray(lam_rest[i:i + _CHUNK]),
                         keys[i:i + _CHUNK])
        means.append(np.asarray(m))
        stds.append(np.asarray(s))
    return np.concatenate(means), np.concatenate(stds)

_N_SNE = 10000


def _generate_snana(fixture_name, genversion, workdir):
    src = (PROJECT_ROOT / "tests" / "fixtures" / "snana" / fixture_name
           / "input.input").read_text()
    s = re.sub(r"GENVERSION:\s+\S+", f"GENVERSION:  {genversion}", src)
    s = re.sub(r"NGENTOT_LC:\s+\d+", f"NGENTOT_LC:  {_N_SNE}", s)
    s = re.sub(r"CIDRAN_MAX:\s+\d+", "CIDRAN_MAX:  30000000", s)
    inp = workdir / "input.input"
    inp.write_text(s)
    env = os.environ.copy()
    r = subprocess.run(["snlc_sim.exe", str(inp)], cwd=workdir, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]
    for base in (Path(env.get("SNANA_LSST_SIM", "")) / genversion,
                 Path(env.get("SNDATA_ROOT", "")) / "SIM" / genversion):
        if base.is_dir():
            return inp, base
    raise FileNotFoundError(f"no sim output for {genversion}")


@pytest.mark.parametrize("fixture_name,genversion", [
    ("g10_smear_stat", "JAX_LIVE_G10_HISTAT"),
    ("c11_smear_stat", "JAX_LIVE_C11_HISTAT"),
])
def test_gensmear_highstat(fixture_name, genversion, tmp_path):
    inp, sim_out = _generate_snana(fixture_name, genversion, tmp_path)
    head = next(sim_out.glob("*HEAD.FITS*"))
    phot = next(sim_out.glob("*PHOT.FITS*"))
    dump_file = next(sim_out.glob("*.DUMP"))

    dump = load_snana_dump(str(dump_file), dedup_cid=True)
    with fits.open(head) as h:
        head_cols = {n: h[1].data[n] for n in h[1].data.names}
    config = dict(n_sne=dump.size)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    override = build_override_parameters(dump.columns, simulation, resources,
                                         head_columns=head_cols,
                                         drop_node_smear=True)
    assert float(np.abs(override['_gensmear_randoms']).max()) == 0.0
    # the replay carries the coherent term SNANA records, so the prediction
    # below must cover only what it left out
    zero_coherent = 'magsmear' in override

    results, sel, spectra_cache = sim_run.run_simulation(sim, override_parameters=override)
    save_snana_outputs(sim, results, "HISTAT", sel, spectra_by_sn=spectra_cache, output_dir=str(tmp_path))

    snana_phot = load_snana_phot(head, phot)
    jax_phot = load_jax_phot(tmp_path / "HISTAT", "HISTAT")

    res = resources
    band_idx = {n: i for i, n in enumerate(res.filter_names)}
    mean_wl = np.asarray(res.photometry.filter_mean_wavelengths)
    wave_obs = np.asarray(res.photometry.common_wave_grid)
    trans_obs = np.asarray(res.photometry.common_trans_grid)
    rest_grid = np.asarray(res.wavelength_grid)
    model = res.model
    cids = np.asarray(dump.columns['CID']).astype(int)
    genz = {int(c): float(z) for c, z in zip(cids, dump.columns['GENZ'])}
    peakmjd = {int(c): float(p) for c, p in zip(cids, dump.columns['PEAKMJD'])}
    x1 = {int(c): float(v) for c, v in zip(cids, dump.columns['SALT2x1'])}
    cc = {int(c): float(v) for c, v in zip(cids, dump.columns['SALT2c'])}

    per_snband = {}
    for key in set(snana_phot) & set(jax_phot):
        snid, band, _ = key
        sn_mag, sn_flux, sn_err, mjd = snana_phot[key]
        jx_mag = jax_phot[key][0]
        if sn_mag >= 50.0 or jx_mag >= 50.0:
            continue
        if sn_err <= 0 or sn_flux / sn_err < 3.0:
            continue
        dt = abs(mjd - peakmjd[snid])
        prev = per_snband.get((snid, band))
        if prev is None or dt < prev[0]:
            phase = (mjd - peakmjd[snid]) / (1.0 + genz[snid])
            per_snband[(snid, band)] = (dt, sn_mag - jx_mag, band, snid, phase)

    entries = [v[1:] for v in per_snband.values()]
    assert len(entries) > 20000, f"too few (SN, band) entries ({len(entries)})"
    diff = np.array([e[0] for e in entries])
    weights = np.empty((len(entries), len(wave_obs)))
    lam_rest_grid = np.empty_like(weights)
    for j, (_, band, snid, phase) in enumerate(entries):
        z = genz[snid]
        lam_rest_grid[j] = wave_obs / (1.0 + z)
        sed = np.asarray(model.generate_sed(
            phase, {'x0': 1e-5, 'x1': x1[snid], 'c': cc[snid], 'redshift': z},
            sim.photometry_data))
        weights[j] = (trans_obs[band_idx[band]] * wave_obs
                      * np.interp(lam_rest_grid[j], rest_grid, sed))

    pred_mean, pred_std = _entry_moments(
        resources.gensmear.n_randoms,
        resources.gensmear.delta_mag,
        weights, lam_rest_grid, zero_coherent)
    n_nan = int(np.isnan(pred_std).sum())
    assert n_nan <= 50, f"{n_nan} NaN predictions"
    usable = np.nan_to_num(pred_std) >= 0.01
    pulls = (diff[usable] - pred_mean[usable]) / pred_std[usable]

    n = len(pulls)
    mean, std = float(np.mean(pulls)), float(np.std(pulls))
    assert abs(mean) < 4.0 / np.sqrt(n), f"global pull mean {mean:+.4f} (n={n})"
    assert abs(std - 1.0) < 4.0 / np.sqrt(2.0 * n), \
        f"global pull std {std:.4f} (n={n})"
