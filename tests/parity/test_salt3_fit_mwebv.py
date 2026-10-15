"""SALT3 fit parity with MW extinction on (OPT_MWEBV=1).

Same gates as test_salt3_fitter_vs_snana, on a fixture whose header carries
nonzero MWEBV consumed by both fitters. An unmodelled E(B-V)=0.02 would bias
the fitted colour by ~0.02 (a coherent multi-sigma mean pull over the batch),
so the mean-pull gate is the sensitive one here.

The OUTLIER test runs the whole fitter rather than calling the flux routine,
because the attenuation reaches the reported model flux through the driver.
"""

import numpy as np

from _fit_helpers import fit_fixture, fixture_data_path, parse_fitres

from jax_snlc_fit.config_parser import parse_fit_input_file
from jax_snlc_fit.run import run_fitting


def _read_outlier(path):
    """OUTLIER rows keyed by the epoch each one describes."""
    hdr, rows = None, {}
    with open(path) as f:
        for line in f:
            if line.startswith("VARNAMES:"):
                hdr = line.split()[1:]
            elif line.startswith("SN:"):
                d = dict(zip(hdr, line.split()[1:]))
                rows[(d["CID"], round(float(d["MJD"]), 2), d["BAND"])] = d
    return rows


def test_salt3_fit_mwebv_vs_snana(snana_fixture):
    fx = snana_fixture("salt3_fit_mwebv")
    results, batch = fit_fixture(fx.root, max_fit=80)
    assert float(np.max(np.asarray(batch.mwebvs))) > 0.01

    fitres_path = fx.root / "FITRES.TEXT"
    assert fitres_path.is_file()
    snana_rows = parse_fitres(fitres_path)

    snids = np.asarray(results.snids)
    pulls = {"PKMJD": [], "x1": [], "c": []}
    for i, snid in enumerate(snids):
        cid = int(snid)
        if cid not in snana_rows:
            continue
        s = snana_rows[cid]
        if s.get("ERRFLAG_FIT", 0) != 0:
            continue
        row_pulls = {}
        for jax_name, val_col, err_col in [
            ("PKMJD", "PKMJD", "PKMJDERR"),
            ("x1", "x1", "x1ERR"),
            ("c", "c", "cERR"),
        ]:
            jax_val = float(results.params[jax_name][i])
            jax_err = float(results.param_errors[jax_name][i])
            err = np.sqrt(s[err_col] ** 2 + jax_err ** 2)
            if not np.isfinite(err) or err <= 0 or not np.isfinite(jax_val):
                break
            row_pulls[jax_name] = (jax_val - s[val_col]) / err
        else:
            for k, v in row_pulls.items():
                pulls[k].append(v)

    n = len(pulls["c"])
    assert n > 40, f"too few compared SNe ({n})"
    for name, p in pulls.items():
        p = np.array(p)
        frac_bad = float(np.mean(np.abs(p) > 3.0))
        assert frac_bad < 0.05, f"{name}: {frac_bad:.1%} of pulls beyond 3 sigma"
        mean_pull = float(np.mean(p))
        assert abs(mean_pull) < 0.5, f"{name}: mean pull {mean_pull:+.2f}"


def test_outlier_model_flux_carries_mw_extinction(snana_fixture, tmp_path):
    """The model flux reported per epoch carries Galactic extinction.

    E(B-V)=0.02 attenuates the model by 9% in u falling to 2% in Y, a
    coherent offset ordered by wavelength. The two minimisers settle close
    enough that the per-band median of the flux ratio sits within a few parts
    in ten thousand of unity, so that median separates the extinction from
    the fit difference by more than an order of magnitude; individual epochs
    scatter by up to 2% and are not a usable gate.

    The whole fitter runs here: SNLC_MWXT is applied where the tables are
    built, not inside the flux routine, so a test that called the routine
    itself would pass with the attenuation dropped.
    """
    fx = snana_fixture("salt3_fit_mwebv")
    config = parse_fit_input_file(
        fx.root / "fit.nml",
        ["BATCH_SIZE", "4", "PRIVATE_DATA_PATH", str(fixture_data_path(fx.root, tmp_path)),
         "OUTPUT_NAME", str(tmp_path / "jax.FITRES.TEXT")])
    run_fitting(config)

    jax_rows = _read_outlier(
        tmp_path / f"{config.output.textfile_prefix}.OUTLIER.TEXT")
    ref_rows = _read_outlier(fx.root / "OUTLIER.TEXT")
    shared = [k for k in jax_rows if k in ref_rows]

    for band in config.fit.filtlist_fit:
        # an epoch whose model flux is consistent with zero carries no
        # information about a multiplicative attenuation
        ratios = [float(jax_rows[k]["FLUXCAL_FIT"])
                  / float(ref_rows[k]["FLUXCAL_FIT"])
                  for k in shared if k[2] == band
                  and abs(float(ref_rows[k]["FLUXCAL_FIT"])) > 1.0]
        assert len(ratios) >= 20, f"{band}: only {len(ratios)} usable epochs"
        offset = float(np.median(ratios)) - 1.0
        assert abs(offset) < 5.0e-3, (
            f"{band}: model flux runs {offset:+.1%} against SNANA, the sign "
            f"and size of dropped Galactic extinction")

    assert len(shared) >= 0.9 * len(ref_rows), (
        f"{len(shared)} of {len(ref_rows)} SNANA outlier epochs matched; "
        f"the significance that selects them follows the model flux")
