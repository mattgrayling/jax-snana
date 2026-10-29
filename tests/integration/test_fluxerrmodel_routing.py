"""FLUXERRMODEL routing: per-band ERRSCALE applied multiplicatively.

The bundled flat FLUXERRMODEL DAT prescribes one ERRSCALE constant per band
(u=1.5, g=2.0, r=2.5, i=3.0, z=3.5, Y=4.0). Comparing FLUXCALERR from a JAX
run with FLUXERRMODEL on against a JAX run without it should yield exactly
those band-specific ratios — provided base noise (skysig + flux Poisson)
cancels out, which it does because both runs use the same SIMLIB epoch by
SIMLIB epoch.

Pins the ROUTING + per-band table dispatch in ``flux_smearing.py``.
"""

from pathlib import Path

import numpy as np
from astropy.io import fits


from jax_snlc_sim import run as sim_run  # noqa: E402
from jax_snlc_sim.output import save_snana_outputs


_EXPECTED_SCALE = {"u": 1.5, "g": 2.0, "r": 2.5, "i": 3.0, "z": 3.5, "Y": 4.0}


def _run_jax(input_file: Path, n_sne: int = 150) -> dict[tuple[int, float, str], float]:
    """Run JAX, save FITS, return ``{(snid, mjd, band): fluxcalerr}``."""
    import os, tempfile  # noqa: E401
    config = dict(
        n_sne=n_sne, random_seed=122459,
    )
    sim = sim_run.prepare(str(input_file), **config)
    results, sel, spectra_cache = sim_run.run_simulation(sim)
    with tempfile.TemporaryDirectory() as td:
        cwd = os.getcwd()
        try:
            os.chdir(td)
            save_snana_outputs(sim, results, "FE", sel, spectra_by_sn=spectra_cache)
            return _load_fluxerr(Path(td) / "FE")
        finally:
            os.chdir(cwd)


def _load_fluxerr(out_dir: Path) -> dict[tuple[int, float, str], float]:
    head_files = list(out_dir.glob("*HEAD*FITS*"))
    phot_files = list(out_dir.glob("*PHOT*FITS*"))
    out = {}
    with fits.open(head_files[0]) as h, fits.open(phot_files[0]) as p:
        head, phot = h[1].data, p[1].data
        for r in head:
            snid = int(r["SNID"])
            pmin, pmax = int(r["PTROBS_MIN"]), int(r["PTROBS_MAX"])
            for i in range(pmin - 1, pmax):
                pr = phot[i]
                mjd = float(pr["MJD"])
                if mjd < 0:
                    continue
                band = str(pr["BAND"]).strip()
                short = band.split("-")[-1] if "-" in band else band
                out[(snid, round(mjd, 2), short)] = float(pr["FLUXCALERR"])
    return out


def test_fluxerrmodel_routing_per_band(snana_fixture):
    """Per-band ERRSCALE applied as a constant multiplier on FLUXCALERR."""
    fx_nom = snana_fixture("fluxerr_nominal")
    fx_fm = snana_fixture("fluxerr_flat")

    nom = _run_jax(fx_nom.input_file)
    fm = _run_jax(fx_fm.input_file)
    common = set(nom) & set(fm)
    assert len(common) > 500, f"too few common epochs ({len(common)})"

    per_band: dict[str, list[float]] = {}
    for key in common:
        n = nom[key]
        f = fm[key]
        if n <= 0:
            continue
        per_band.setdefault(key[2], []).append(f / n)

    # Every band in the FLUXERRMODEL must show the prescribed ratio.
    for band, expected in _EXPECTED_SCALE.items():
        ratios = np.asarray(per_band.get(band, []))
        assert len(ratios) > 80, f"band {band}: too few ratios ({len(ratios)})"
        median = float(np.median(ratios))
        assert abs(median - expected) / expected < 0.01, (
            f"band {band}: median(FLUXCALERR_fm/nom)={median:.3f} but "
            f"FLUXERRMODEL prescribes {expected}"
        )
