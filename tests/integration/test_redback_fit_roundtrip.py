"""Simulate redback Arnett light curves with the JAX simulator, then recover
the truth parameters with the JAX fitter.

Pins the redback fitting path end to end: the explosion-time t0 convention,
the pinned-by-default distance modulus, and (second fit) ``INIVAL_``/``INISTP_``
flowing from a config through the parameter resolver into the held set.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("redback_jax")

from astropy.io import fits

from _fit_helpers import (build_fit_model, build_fitter, fit_config,
                          load_batch, override, run_fit)

from jax_snlc_sim import run as sim_run
from jax_snlc_core.models import build_model
from jax_snlc_fit.config_parser import build, read_keys
from jax_snlc_fit.io import discover_fits_files

_F = "$JAX_TEST_FIXTURE_ROOT"
_MODEL = "REDBACK.arnett_bolometric"
_GENVERSION = "REDBACK_FIT_ROUNDTRIP"


def _write_sim_input(out: Path) -> None:
    # Truth offset from the fitter's table inits (mej 2.0, f_nickel 0.1) with
    # tight GENSIGMAs; the simulation uses H0=67.4.
    out.write_text(f"""\
GENVERSION: {_GENVERSION}
NGENTOT_LC: 200
FORMAT_MASK: 32
GENTYPE: 10
SNTYPES_Ia: 10 110
GENMODEL: {_MODEL}
GENPEAK_f_nickel: 0.15
GENSIGMA_f_nickel: 0.01 0.01
GENRANGE_f_nickel: 0.05 0.4
GENPEAK_mej: 3.0
GENSIGMA_mej: 0.2 0.2
GENRANGE_mej: 1.0 8.0
GENPEAK_vej: 9000.0
GENSIGMA_vej: 500 500
GENRANGE_vej: 5000 15000
GENPEAK_kappa: 0.1
GENPEAK_kappa_gamma: 0.03
GENPEAK_temperature_floor: 5000.0
H0: 67.4
OMEGA_MATTER: 0.315
OMEGA_LAMBDA: 0.685
w0_LAMBDA: -1.0
wa_LAMBDA: 0
GENFILTERS: ugrizY
KCOR_FILE: {_F}/kcor/JAX_TEST_LSST.fits
SIMLIB_FILE: {_F}/simlib/JAX_TEST_LSST_DENSE.SIMLIB
GENRANGE_REDSHIFT: 0.02 0.12
GENRANGE_PEAKMJD: 60540 61460
GENSIGMA_REDSHIFT: 0.001
GENRANGE_TREST: -20 60
SOLID_ANGLE: 0.015
OPT_MWEBV: 0
DNDZ: POWERLAW 2.5E-5 1.5
RANSEED: 12345
""")


def _write_fit_nml(out: Path, tmp: Path) -> None:
    out.write_text(f"""\
 &SNLCINP
  VERSION_PHOTOMETRY = '{_GENVERSION}'
  PRIVATE_DATA_PATH  = '{tmp}'
  KCOR_FILE          = '{_F}/kcor/JAX_TEST_LSST.fits'
  NFIT_ITERATION     = 3
 &END

 &FITINP
  FITMODEL_NAME = '{_MODEL}'
  FILTLIST_FIT  = 'ugrizY'
 &END
""")


@pytest.fixture(scope="module")
def roundtrip(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("redback_roundtrip")
    inp = tmp / "sim.input"
    _write_sim_input(inp)
    _write_fit_nml(tmp / "fit.nml", tmp)

    cfg = dict(n_sne=200, batch_size=4,
                              random_seed=12345)
    sim = sim_run.prepare(str(inp), **cfg)
    results, selection, spectra_cache = sim_run.run_simulation(sim)
    sim_run.save_snana_outputs(sim, results, _GENVERSION, selection, spectra_cache=spectra_cache, output_dir=str(tmp))

    [(head_path, phot_path)] = discover_fits_files(str(tmp / _GENVERSION))
    return tmp, head_path, phot_path


def _truth_by_cid(head_path):
    with fits.open(head_path) as hdul:
        head = hdul[1].data
        cols = head.columns.names
        rows = [{
            "cid": int(r["SNID"]),
            "z": float(r["SIM_REDSHIFT_CMB"]),
            "peakmjd": float(r["SIM_PEAKMJD"]),
            "f_nickel": float(r["SIM_REDBACK_f_nickel"]),
            "mej": float(r["SIM_REDBACK_mej"]),
            "vej": float(r["SIM_REDBACK_vej"]),
            "kappa": float(r["SIM_REDBACK_kappa"]),
            "kappa_gamma": float(r["SIM_REDBACK_kappa_gamma"]),
        } for r in head]
    return {row["cid"]: row for row in rows}, cols


def _fit(roundtrip, config=None):
    """Fit the simulated bundle at the roundtrip's own fit.nml settings."""
    tmp, head_path, phot_path = roundtrip
    if config is None:
        config = fit_config(tmp / "fit.nml")
    config = override(config, batch_size=1)
    batch = load_batch(config, head_path, phot_path, max_fit=25)
    assert batch.n_sne >= 6, f"too few fittable SNe ({batch.n_sne})"
    fitter = build_fitter(config, build_fit_model(config), batch)
    return batch, run_fit(fitter, batch)


def test_default_fit_recovers_truth(roundtrip):
    _, head_path, _ = roundtrip
    truth, cols = _truth_by_cid(head_path)
    batch, results = _fit(roundtrip)

    snids = np.asarray(results.snids)
    t0 = np.asarray(results.params["PKMJD"])
    fni = np.asarray(results.params["f_nickel"])
    mej = np.asarray(results.params["mej"])

    # t0 truth is the explosion epoch = the sim's (explosion-anchored) PEAKMJD.
    dt0, lum_ratio = [], []
    for i, sid in enumerate(snids):
        tr = truth[int(sid)]
        dt0.append(t0[i] - tr["peakmjd"])
        # f_nickel * mej sets the peak luminosity and is the well-constrained combo.
        lum_ratio.append((fni[i] * mej[i]) / (tr["f_nickel"] * tr["mej"]))

    assert np.all(np.isfinite(t0))
    assert np.median(np.abs(dt0)) < 2.0
    assert np.median(np.abs(np.array(lum_ratio) - 1.0)) < 0.30


def test_yaml_pinned_kappa_fit(roundtrip, tmp_path):
    """INIVAL_/INISTP_ for kappa and kappa_gamma flow from a config through the
    resolver into the held set. A pinned parameter keeps its column and is
    reported at its pinned value, and the fit still recovers f_nickel*mej."""
    _, head_path, _ = roundtrip
    truth, _ = _truth_by_cid(head_path)

    yml = tmp_path / "fit.yaml"
    yml.write_text(
        f"SNLCINP:\n"
        f"  KCOR_FILE: {_F}/kcor/JAX_TEST_LSST.fits\n"
        f"  NFIT_ITERATION: 3\n"
        f"FITINP:\n"
        f"  FITMODEL_NAME: {_MODEL}\n"
        f"  FILTLIST_FIT: ugrizY\n"
        f"  INIVAL_kappa: 0.1\n"
        f"  INISTP_kappa: 0\n"
        f"  INIVAL_kappa_gamma: 0.03\n"
        f"  INISTP_kappa_gamma: 0\n"
    )

    batch, results = _fit(roundtrip, config=fit_config(yml))

    # Pinned, not removed: reported at exactly the supplied value.
    assert np.allclose(np.asarray(results.params["kappa"]), 0.1)
    assert np.allclose(np.asarray(results.params["kappa_gamma"]), 0.03)

    snids = np.asarray(results.snids)
    fni = np.asarray(results.params["f_nickel"])
    mej = np.asarray(results.params["mej"])
    lum_ratio = []
    for i, sid in enumerate(snids):
        tr = truth[int(sid)]
        lum_ratio.append((fni[i] * mej[i]) / (tr["f_nickel"] * tr["mej"]))
    assert np.median(np.abs(np.array(lum_ratio) - 1.0)) < 0.30


def test_inistp_mu_frees_the_distance(roundtrip):
    """mu is pinned by default and floats when INISTP_mu is non-zero. Nothing
    else distinguishes the two fits, so this is the declared default step being
    overridden by the input."""
    tmp, _, _ = roundtrip
    base = fit_config(tmp / "fit.nml")

    batch, pinned = _fit(roundtrip, config=base)
    _, freed = _fit(roundtrip,
                    config=fit_config(tmp / "fit.nml", inistp={'mu': 1.0}))

    # Pinned: the fit never moves it off the cosmological seed.
    seed = np.asarray(base.fit.distance_modulus(batch.redshifts))
    assert np.allclose(np.asarray(pinned.params["mu"]), seed, rtol=0, atol=1e-9)
    # Freed: it moves, degenerate with the luminosity scale but not held.
    assert np.median(np.abs(np.asarray(freed.params["mu"]) - seed)) > 0.05


def test_redback_tgrid_nml_keys_reach_the_model(tmp_path):
    """REDBACK_TGRID_* in the fit namelist are REDBACK's declared keys: the
    parser types them into model_key_values without knowing them by name, and
    build_model sets them on the instance before the grid is built."""
    nml = tmp_path / "fit.nml"
    nml.write_text(
        "&FITINP\n"
        "  FITMODEL_NAME = 'REDBACK.arnett_bolometric'\n"
        "  REDBACK_TGRID_TMIN = 0.05\n"
        "  REDBACK_TGRID_TMAX = 300.0\n"
        "  REDBACK_TGRID_NBIN = 256\n"
        "&END\n")
    cfg = build(read_keys(nml))
    assert cfg.model._model_key_values == {'REDBACK_TGRID_TMIN': 0.05,
                                    'REDBACK_TGRID_TMAX': 300.0,
                                    'REDBACK_TGRID_NBIN': 256}
    model = build_model('REDBACK.arnett_bolometric',
                        cfg.model._model_key_values)
    assert len(model.lbol_time_grid) == 256
    assert model.lbol_time_grid[0] == pytest.approx(0.05)
    assert model.lbol_time_grid[-1] == pytest.approx(300.0)
