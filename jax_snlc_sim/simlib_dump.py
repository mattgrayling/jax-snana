"""SNANA SIMLIB_DUMP diagnostic mode (SIMLIB_DUMP_DRIVER, snlc_sim.c:31710).

Reads the SIMLIB and writes per-LIBID cadence averages (bit 1) and/or per-obs
tables (bit 2), then the job exits without generating any SNe. Runs off the
read_simlib alone, no GENMODEL required.
"""

import os
from pathlib import Path

import numpy as np

from jax_snlc_sim.photometry.maglimit import maglimit_calculator
from jax_snlc_sim import builder
from jax_snlc_sim.survey.simlib_reader import FWHM_SIGMA_RATIO

SIMLIB_DUMPMASK_AVG = 1
SIMLIB_DUMPMASK_OBS = 2
ZPTERR_MAX = 0.05
MJDGAP_IGNORE = 50.0


def _mjdgap(mjds):
    """SNANA MJDGAP: GAPMAX = max consecutive gap (sorted MJDs); GAPAVG = mean
    gap shorter than MJDGAP_IGNORE (seasonal gaps excluded from the average)."""
    if len(mjds) <= 1:
        return 0.0, 0.0
    gaps = np.diff(np.sort(np.asarray(mjds, dtype=np.float64)))
    gapmax = float(gaps.max())
    within = gaps[(gaps > 0.001) & (gaps < MJDGAP_IGNORE)]
    gapavg = float(within.mean()) if within.size else 0.0
    return gapmax, gapavg


def _obs_photometry(field, i):
    """Per-obs SNANA conversions: PSF(arcsec FWHM), ZPT(pe), SKYMAG, M5SIG."""
    gain = float(field.gains[i])
    pixsize = float(field.pixsize)
    psf = float(field.psf1s[i]) * pixsize * FWHM_SIGMA_RATIO
    zpt_pe = float(field.zptavgs[i]) + 2.5 * np.log10(gain)
    skysig_pe = float(field.skysigs[i]) * gain
    fsky_pe = skysig_pe ** 2 / pixsize ** 2
    skymag = zpt_pe - 2.5 * np.log10(fsky_pe) if fsky_pe > 1e-9 else 0.0
    m5sig = maglimit_calculator(zpt_pe, psf, skymag, 5.0)
    return psf, zpt_pe, skymag, m5sig


def _prefix(simlib_file):
    return Path(simlib_file).name.split('.')[0]


def _keep_obs(field, mjd_range, peakmjd_range):
    """Per-obs keep mask: GENRANGE_MJD always and GENRANGE_PEAKMJD in dump
    mode (keep_SIMLIB_OBS, snlc_sim.c:21200-21233), plus the dump's ZPTERR
    skip; NOBS_TOT counts what survives all three (snlc_sim.c:32015-32019)."""
    mjds = np.asarray(field.mjds, dtype=np.float64)
    keep = np.asarray(field.zpterrs) <= ZPTERR_MAX
    keep &= (mjds >= mjd_range[0]) & (mjds <= mjd_range[1])
    keep &= (mjds >= peakmjd_range[0]) & (mjds <= peakmjd_range[1])
    return keep


def _write_docana(fp, meta, is_obs):
    """SNANA SIMLIB_DUMP DOCUMENTATION block (+ COLUMN_NOTES for the OBS table)."""
    path = meta['simlib_file']
    fp.write("DOCUMENTATION:\n")
    fp.write(f"  PURPOSE: table dump for SIMLIB_FILE {path}\n")
    fp.write(f"  SIMLIB_FILE: {path} \n")
    fp.write(f"  SURVEY:      {meta['survey']} \n")
    fp.write(f"  FILTERS:     {meta['filters']} \n")
    fp.write(f"  FIELDLIST:   {meta['fieldlist']} \n")
    fp.write(f"  MJD_RANGE:   {meta['mjd_range']} \n")
    fp.write(f"  PRESCALE:    {meta['prescale']} \n")
    fp.write(f"  NLIBID_TOT:  {meta['nlibid']} \n")
    fp.write(f"  NOBS_TOT:    {meta['nobs']} \n")
    if is_obs:
        fp.write("\n  COLUMN_NOTES:\n")
        fp.write(f"    LIBID:    Cadence ID in {path} \n")
        fp.write("    SKYMAG:   ADU/pixel\n")
        fp.write("    PSF:      effective Gauss sigma (pixels) = sqrt[ NEA/(4*PI) ]\n")
        fp.write("    MJD_DIF:       time since last observation (days)\n")
        fp.write("    MJD_DIF_BAND:  time since last observation in same band (days)\n")
    fp.write("DOCUMENTATION_END:\n\n")


def _write_avg(path, simlib, meta, keeps):
    bands = list(simlib.filter_names)
    with open(path, 'w') as fp:
        _write_docana(fp, meta, is_obs=False)
        fp.write("VARNAMES: ROW LIBID RA DEC FIELD MWEBV GAPMAX GAPAVG "
                 "NOBS MJDMIN MJDMAX NSEASON  ")
        for b in bands:
            fp.write(f"NOBS_{b} ZPT_{b} PSF_{b} M5SIG_{b} ")
        fp.write("\n\n")
        for row, (field, keep) in enumerate(zip(simlib.fields, keeps), start=1):
            mjd = np.asarray(field.mjds, dtype=np.float64)[keep]
            fidx = np.asarray(field.filter_indices)[keep]
            nseason = int(np.sum(np.isfinite(field.season_starts)))
            gapmax, gapavg = _mjdgap(mjd)
            mjdmin = float(mjd.min()) if mjd.size else 99999.0
            mjdmax = float(mjd.max()) if mjd.size else -99999.0
            fp.write(
                f"ROW: {row:4d} {int(field.libid):4d} {float(field.ra):7.3f} "
                f"{float(field.dec):7.3f} {field.field_name} {float(field.mwebv):6.3f} "
                f"{gapmax:5.0f} {gapavg:3.1f} {mjd.size:3d} {mjdmin:.2f} {mjdmax:.2f}  "
                f"{nseason}  ")
            for bi, b in enumerate(bands):
                sel = fidx == bi
                nobs_b = int(np.sum(sel))
                if nobs_b:
                    idx = np.where(keep)[0][sel]
                    phot = np.array([_obs_photometry(field, i) for i in idx])
                    zpt_b, psf_b, m5_b = phot[:, 1].mean(), phot[:, 0].mean(), phot[:, 3].mean()
                else:
                    zpt_b = psf_b = m5_b = 0.0
                fp.write(f"{nobs_b:2d} {zpt_b:6.2f} {psf_b:5.2f} {m5_b:5.2f} ")
            fp.write("\n")


def _write_obs(path, simlib, meta, keeps):
    with open(path, 'w') as fp:
        _write_docana(fp, meta, is_obs=True)
        fp.write("VARNAMES: ROW LIBID FIELD  RA DEC MJD BAND ZP_pe SKYMAG PSF "
                 "M5SIG   MJD_DIF MJD_DIF_BAND\n")
        row = 0
        for field, keep in zip(simlib.fields, keeps):
            last_mjd = -9.0
            last_band = {}
            for i in range(len(field.mjds)):
                if not keep[i]:
                    continue
                mjd = float(field.mjds[i])
                bi = int(field.filter_indices[i])
                band = simlib.filter_names[bi]
                psf, zpt_pe, skymag, m5sig = _obs_photometry(field, i)
                mjd_dif = mjd - last_mjd
                if mjd_dif > 40000.0:
                    mjd_dif = 9999.0
                mjd_dif_band = mjd - last_band.get(bi, -9.0)
                if mjd_dif_band > 40000.0:
                    mjd_dif_band = 9999.0
                row += 1
                fp.write(
                    f"ROW: {row:3d} {int(field.libid):4d} {field.field_name} "
                    f"{float(field.ra):.4f} {float(field.dec):.4f} {mjd:.4f}  "
                    f"{band}  {zpt_pe:.3f}  {skymag:.3f}  {psf:.3f} {m5sig:.3f}   "
                    f"{mjd_dif:.4f} {mjd_dif_band:.4f}\n")
                last_mjd = mjd
                last_band[bi] = mjd


def run_simlib_dump(config, output_dir=None):
    """Write the SIMLIB_DUMP file(s) selected by the SIMLIB_DUMP bit mask.
    Returns the list of written file paths."""
    simlib = builder._load_simlib(config)
    mode = config.survey.simlib_dump
    outdir = output_dir or os.getcwd()
    prefix = _prefix(simlib.simlib_file)

    # GENRANGE_MJD becomes the MJD_RANGE cut (filename suffix + DOCANA line);
    # unset means SNANA's wide-open init 20000-80000 (snlc_sim.c:856-857).
    grm = config.rates.genrange_mjd
    if grm:
        cut = f"_{int(grm[0])}-{int(grm[1])}"
    else:
        cut = ""
        grm = (20000.0, 80000.0)
    mjd_range = f"{int(grm[0])} {int(grm[1])}"
    # PEAKMJD window as an obs cut, unset -> GENRANGE_MJD (snlc_sim.c:21220-21223, :32568-32571)
    pkr = config.rates.peakmjd_range
    if pkr[0] == 0.0:
        pkr = grm
    keeps = [_keep_obs(f, grm, pkr) for f in simlib.fields]
    meta = {
        'simlib_file': os.path.expandvars(simlib.simlib_file),
        'survey': simlib.survey_name,
        'filters': "".join(simlib.filter_names),
        'fieldlist': config.survey.simlib_fieldlist,
        'mjd_range': mjd_range,
        'prescale': config.survey.simlib_prescale,
        'nlibid': len(simlib.fields),
        'nobs': int(sum(k.sum() for k in keeps)),
    }
    written = []
    if mode & SIMLIB_DUMPMASK_AVG:
        path = str(Path(outdir) / f"SIMLIB_DUMP_AVG_{prefix}{cut}.TEXT")
        _write_avg(path, simlib, meta, keeps)
        written.append(path)
    if mode & SIMLIB_DUMPMASK_OBS:
        path = str(Path(outdir) / f"SIMLIB_DUMP_OBS_{prefix}{cut}.TEXT")
        _write_obs(path, simlib, meta, keeps)
        written.append(path)
    return written
