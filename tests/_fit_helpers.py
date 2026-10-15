"""Fitter construction driven by a fixture's own fit namelist.

A test that compares against a recorded SNANA table has to fit at the
settings that table was produced with, so the configuration comes from the
namelist beside it rather than from literals in the test. Tests use small
numerical batches; scientific settings come from the namelist and any
explicit overrides.
"""

import dataclasses
import os
import re
from pathlib import Path

import jax.numpy as jnp

from jax_snlc_core.models import infer_model_type_from_path
from jax_snlc_fit.config_parser import FIELD_RECORD, parse_fit_input_file
from jax_snlc_fit.data_loader import load_and_batch_snana_fits
from jax_snlc_fit.fit_spec import init_params
from jax_snlc_fit.fitter import attach_mw_attenuation, fit_photometry
from jax_snlc_fit.photoz import redshift_priors
from jax_snlc_fit.run import (lcplot_from_fitres, mwebv_options, prepare,
                              read_kcor, resolved_filters, run_fitting)
from jax_snlc_fit.run import fit_model


def override(config, **flat):
    """``config`` with flat FitConfig field names replaced."""
    groups = {}
    for name, value in flat.items():
        groups.setdefault(FIELD_RECORD[name], {})[name] = value
    return dataclasses.replace(config, **{
        rec: dataclasses.replace(getattr(config, rec), **kw)
        for rec, kw in groups.items()})


def fit_config(nml_path, **overrides):
    """The namelist's own configuration, with flat ``overrides`` applied."""
    return override(parse_fit_input_file(nml_path),
                    **{'batch_size': 4, 'maxiter_round1': 50, 'maxiter_round2': 30,
                       **overrides})


def load_batch(config, head_path, phot_path, max_fit=None, **loader_kwargs):
    """Light curves in the fit bands ``config`` names, cut to ``max_fit``."""
    filters = resolved_filters(config, read_kcor(config))
    batch = load_and_batch_snana_fits(
        str(head_path), str(phot_path), fit_bands=filters,
        filter_map={b: i for i, b in enumerate(filters)},
        mwebv=mwebv_options(config), **loader_kwargs)
    if max_fit is not None and batch.n_sne > max_fit:
        batch = batch.slice(0, max_fit)
    return batch


def build_fit_model(config):
    """The model ``config`` names, of the type the driver picks for it."""
    return fit_model(dataclasses.replace(config, model=dataclasses.replace(
        config.model, model_type=config.model.model_type
        or infer_model_type_from_path(
            os.path.expandvars(config.model.fitmodel_name)))))


def build_fitter(config, model, batch):
    """The fitter the driver builds from ``config`` over ``batch``, which
    carries the photometry and redshift priors ``run.load`` gives it."""
    return prepare(config, model, read_kcor(config),
                   _as_loaded(config, model, batch))


def _as_loaded(config, model, batch):
    """``batch`` with the photometry and, under photo-z, the redshift priors
    ``run.load`` gives it under ``config`` for ``model``."""
    return dataclasses.replace(
        batch, photometry=_photometry(config, model, batch),
        z_priors=(redshift_priors(config, batch)
                  if config.photoz.dofit_photoz else None))


def _photometry(config, model, batch):
    """The fit bands as ``config`` calibrates them, and the batch's zero
    point."""
    return {**fit_photometry(config, model, read_kcor(config)),
            'zp_fluxcal': batch.zp_fluxcal}


def fitter_photometry(fitter, batch):
    """The photometry ``run.load`` gives ``batch`` under ``fitter``'s own
    namelist."""
    return _photometry(fitter.config, fitter.model, batch)


def for_fit(fitter, batch):
    """``batch`` as the run fits it: with the photometry and, under
    photo-z, the redshift priors ``run.load`` gives it, and its Milky Way
    attenuation."""
    return attach_mw_attenuation(
        fitter.config, _as_loaded(fitter.config, fitter.model, batch))


def peak_start(fitter, batch):
    """Starting values for ``batch`` seeded at its own PEAKMJD."""
    t0_init = jnp.array(batch.reference_mjds if batch.peakmjd_offsets is None
                        else batch.peakmjd_offsets + batch.reference_mjds)
    return init_params(fitter.model, batch.n_sne, t0_init,
                       fitter.config.fit.start_distance_modulus(batch.redshifts))


def run_fit(fitter, batch):
    """Fit ``batch``, as loaded, seeded at its own PEAKMJD."""
    return fitter.fit_sample(for_fit(fitter, batch),
                             peak_start(fitter, batch))[0]


def fit_fixture(fixture_dir: Path, max_fit=None, epoch_ignore=None, data_dir=None,
                **overrides) -> tuple:
    """Fit a fixture with its own fit.nml plus flat ``overrides``."""
    config = fit_config(fixture_dir / "fit.nml", **overrides)
    data_dir = fixture_dir if data_dir is None else data_dir
    batch = load_batch(config, data_dir / "HEAD.FITS",
                       data_dir / "PHOT.FITS", max_fit,
                       epoch_ignore=epoch_ignore)
    assert batch.n_sne > 0
    fitter = build_fitter(config, build_fit_model(config), batch)
    return run_fit(fitter, batch), batch


def parse_fitres(path: Path) -> dict[int, dict[str, float]]:
    """Return ``{cid: {col: float}}`` for an SNANA FITRES.TEXT file."""
    varnames: list[str] | None = None
    rows: dict[int, dict[str, float]] = {}
    for line in path.read_text().splitlines():
        if line.startswith("VARNAMES:"):
            varnames = line.split(":", 1)[1].split()
        elif line.startswith("SN:"):
            assert varnames is not None
            parts = line.split()[1:]
            row: dict[str, float] = {}
            for name, val in zip(varnames, parts):
                try:
                    row[name] = float(val)
                except ValueError:
                    pass
            rows[int(row["CID"])] = row
    return rows


def _lcplot_rows(path: Path) -> list[dict[str, str]]:
    """The ``OBS:`` rows of an LCPLOT table, as strings."""
    names, rows = None, []
    for line in path.read_text().splitlines():
        if line.startswith("VARNAMES:"):
            names = line.split()[1:]
        elif line.startswith("OBS:"):
            rows.append(dict(zip(names, line.split()[1:])))
    return rows


def fixture_data_path(fixture_dir: Path, work: Path) -> Path:
    """A PRIVATE_DATA_PATH under ``work`` holding the fixture's HEAD/PHOT as
    the data version its fit.nml names, for an in-process fit. The nml names
    the path as $JAX_TEST_FIT_SIM_DIR for the subprocess cases; in-process,
    PRIVATE_DATA_PATH is overridden with this rather than a variable set that
    would outlive the test."""
    version = re.search(r"VERSION_PHOTOMETRY\s*=\s*'([^']+)'",
                        (fixture_dir / "fit.nml").read_text()).group(1)
    version_dir = work / "sim" / version
    version_dir.mkdir(parents=True)
    for kind in ("HEAD", "PHOT"):
        (version_dir / f"{version}_{kind}.FITS.gz").symlink_to(
            fixture_dir / f"{kind}.FITS.gz")
    (version_dir / f"{version}.LIST").write_text(f"{version}_HEAD.FITS\n")
    return work / "sim"


def _in_process_config(fixture_dir: Path, work: Path, overrides):
    """The fixture's fit.nml reading the data :func:`fixture_data_path` put
    under ``work`` and writing its tables there."""
    return parse_fit_input_file(
        fixture_dir / "fit.nml",
        ["BATCH_SIZE", "4", "PRIVATE_DATA_PATH", str(work / "sim"),
         "TEXTFILE_PREFIX", str(work / "FIT"), *overrides])


def regenerate_lcplot(fixture_dir: Path, work: Path, *overrides) -> tuple:
    """Fit a fixture with its own fit.nml and ``overrides``, then rebuild its
    LCPLOT from the FITRES with :func:`rebuild_lcplot`.

    Returns the fit's LCPLOT rows, the rebuilt rows, and the FITRES rows by
    CID.
    """
    fixture_data_path(fixture_dir, work)
    run_fitting(_in_process_config(fixture_dir, work, overrides))
    return (_lcplot_rows(work / "FIT.LCPLOT.TEXT"),
            rebuild_lcplot(fixture_dir, work, "REBUILT", *overrides),
            parse_fitres(work / "FIT.FITRES.TEXT"))


def rebuild_lcplot(fixture_dir: Path, work: Path, name: str,
                   *overrides) -> dict:
    """Rebuild the LCPLOT of the fit :func:`regenerate_lcplot` left in
    ``work`` as ``<work>/<name>`` with :func:`lcplot_from_fitres`, returning
    its rows keyed by (CID, BAND, whether a curve row, MJD)."""
    lcplot_from_fitres(_in_process_config(fixture_dir, work, overrides),
                       work / "FIT.FITRES.TEXT", work / name)
    return {(r["CID"], r["BAND"], r["DATAFLAG"] == "0", r["MJD"]): r
            for r in _lcplot_rows(work / f"{name}.LCPLOT.TEXT")}


def compare_lcplots(fit_rows, rebuilt) -> dict:
    """How a rebuilt LCPLOT differs from the fit's own: rows it lacks, data
    rows whose flux or error changed, the largest curve difference as a
    fraction of that band's peak, the largest CHI2 difference, and the
    (CID, MJD) of each data row whose DATAFLAG differs."""
    def key(r):
        return (r["CID"], r["BAND"], r["DATAFLAG"] == "0", r["MJD"])
    peak = {}
    for r in fit_rows:
        if r["DATAFLAG"] == "0":
            band = (r["CID"], r["BAND"])
            peak[band] = max(peak.get(band, 0.0), abs(float(r["FLUXCAL"])))
    out = {"missing": 0, "data_changed": 0, "curve": 0.0, "chi2": 0.0,
           "flags": []}
    for r in fit_rows:
        s = rebuilt.get(key(r))
        if s is None:
            out["missing"] += 1
        elif r["DATAFLAG"] == "0":
            out["curve"] = max(out["curve"], max(
                abs(float(r[c]) - float(s[c])) for c in ("FLUXCAL", "FLUXCAL_ERR"))
                / peak[(r["CID"], r["BAND"])])
        else:
            out["data_changed"] += (r["FLUXCAL"], r["FLUXCAL_ERR"]) != (
                s["FLUXCAL"], s["FLUXCAL_ERR"])
            out["chi2"] = max(out["chi2"],
                              abs(float(r["CHI2"]) - float(s["CHI2"])))
            if r["DATAFLAG"] != s["DATAFLAG"]:
                out["flags"].append((r["CID"], float(r["MJD"])))
    return out
