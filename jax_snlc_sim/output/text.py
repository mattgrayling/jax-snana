"""SNANA TEXT (.DAT) light-curve serialisation."""

import logging
from pathlib import Path

import numpy as np

from jax_snlc_core.snana_files import HOSTGAL_PREFIXES, MXHOSTGAL
from jax_snlc_sim.output.tables import (HOSTGAL_SB_NULL, SNANA_NULLFLOAT, filters_string,
                                        mw_option_keys)
from jax_snlc_sim.simulation import parse_format_mask

logger = logging.getLogger(__name__)


_TEXT_SIM_FILTERPAR = [
    ('SIM_GALFRAC', "F_gal/F_SNpeak for PSF=1''", '{:6.3f}', 14,
     SNANA_NULLFLOAT),
    ('SIM_PEAKMAG', '', '{:7.4f}', 14, None),
    ('SIM_EXPOSURE', '', '{:6.1f}', 14, None),
]

_TEXT_HOST_FILTERPAR = [
    ('HOSTGAL_SB_FLUXCAL', '{bands}/asec^2', '{:.2f}', 10, HOSTGAL_SB_NULL),
    ('HOSTGAL_SB_FLUXCAL_ERR', '{bands}/asec^2', '{:6.2f}', 10, HOSTGAL_SB_NULL),
]

_TEXT_OBS_COLS = [
    ('MJD', 'MJD', '{:.4f}'), ('BAND', 'BAND', '{}'),
    ('FIELD', 'FIELD', '{}'), ('FLUXCAL', 'FLUXCAL', '{:.4e}'),
    ('FLUXCALERR', 'FLUXCALERR', '{:.4e}'), ('PHOTFLAG', 'PHOTFLAG', '{:d}'),
    ('PHOTPROB', 'PHOTPROB', '{:.3f}'),
    ('GAIN', 'GAIN', '{:.3f}'), ('ZPT', 'ZEROPT', '{:.3f}'),
    ('PSF', 'PSF_SIG1', '{:.3f}'), ('SKY_SIG', 'SKY_SIG', '{:.3e}'),
    ('SKY_SIG_T', 'SKY_SIG_T', '{:.3e}'),
    ('dRA', 'dRA', '{:.4f}'), ('dDEC', 'dDEC', '{:.4f}'),
    ('AIRMASS', 'AIRMASS', '{:.2f}'),
    ('SIM_DCR_dRA', 'SIM_DCR_dRA', '{:.4f}'),
    ('SIM_DCR_dDEC', 'SIM_DCR_dDEC', '{:.4f}'),
    ('SIM_DCR_dMAG', 'SIM_DCR_dMAG', '{:.5f}'),
    ('SIM_MAGOBS', 'SIM_MAGOBS', '{:.4f}'),
]

_TEXT_HOSTGAL_KEYS = [
    ('OBJID', '{:d}', '', None),
    ('FLAG', '{:d}', '', None),
    ('PHOTOZ', '{:.5f}', '', None),
    ('SPECZ', '{:.5f}', '', None),
    ('RA', '{:.6f}', 'deg', None),
    ('DEC', '{:.6f}', 'deg', None),
    ('SNSEP', '{:.3f}', 'arcsec', None),
    ('DDLR', '{:.3f}', 'SNSEP/DLR', None),
    ('LOGMASS', '{:.3f}', 'log10(Mgal/Msolar)', -1.0e9),
    ('LOGSFR', '{:.3e}', 'SFR', -9000.0),
    ('LOGsSFR', '{:.3e}', 'specific SFR', -9000.0),
    ('COLOR', '{:.3e}', 'COLOR (e.g. U-R)', -9000.0),
    ('OBJID2', '{:d}', 'alternate ID', 0),
    ('OBJID_UNIQUE', '{:d}', 'unique ID', 0),
    ('ELLIPTICITY', '{:.3f}', '', 0),
    ('SQRADIUS', '{:.3f}', 'Ixx + Iyy', 0),
]

_TEXT_HOSTGAL_PAIRED = {'PHOTOZ', 'SPECZ', 'LOGMASS', 'LOGSFR',
                        'LOGsSFR', 'COLOR'}

_TEXT_SIM_FITS_ONLY = {'SIM_TYPE_INDEX'}


def write_text_lightcurves(columnar_data, config, resources, kcor_filter_names,
                           file_prefix, snrmon_colname, zp_fluxcal, output_dir,
                           header_cols, phot_cols, to_written_cid,
                           trest_ranges=None, spectra_by_sn=None, *,
                           blindtest=False):
    """Write per-SN SNANA TEXT light curves (<GENPREFIX>_SN<CID>.DAT),
    serialising the same HEAD/PHOT columns the FITS path writes.

    ``trest_ranges`` is the per-SN (SIM_TRESTMIN, SIM_TRESTMAX) pair over the
    EPCUTWIN-filtered epochs (gen_cutwin, snlc_sim.c:24199-24241)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    n_sne = len(next(iter(header_cols.values())))
    cidran = parse_format_mask(config.output.format_mask)['cidran']
    cid_w = 8 if (cidran or n_sne > 1000000) else 6

    # bytes columns to str
    def printable(cols):
        return {c: (np.asarray(v).astype('U') if v.dtype.kind == 'S'
                    else np.asarray(v))
                for c, v in cols.items()}

    hcol = printable(header_cols)
    pcol = printable(phot_cols)

    sim_cols = [] if blindtest else sorted(c for c in hcol if c.startswith('SIM_'))
    host_cols = sorted(c for c in hcol if c.startswith('HOSTGAL'))
    # SKY_SIG_T only when template noise is present (WRFLAG_SKYSIG_T)
    skysigt_on = 'SKY_SIG_T' in pcol and bool(np.any(pcol['SKY_SIG_T'] != 0))
    # PHOTPROB only with PHOTPROB maps (WRFLAG_PHOTPROB, snlc_sim.c:25364)
    photprob_on = bool(resources.searcheff_photprob_maps)
    obs_cols = [(n, c, f) for (n, c, f) in _TEXT_OBS_COLS if c in pcol
                and not (n.startswith('SIM_') and blindtest)
                and not (n == 'SKY_SIG_T' and not skysigt_on)
                and not (n == 'PHOTPROB' and not photprob_on)]
    # last, after SIM_MAGOBS (sntools_dataformat_text.c:988-993)
    if snrmon_colname and snrmon_colname in pcol and not blindtest:
        obs_cols.append(
            (snrmon_colname, snrmon_colname, '{:.1f}'))
    obs_arrays = [(n, pcol[c], f) for n, c, f in obs_cols]
    mjd_arr = pcol['MJD']

    dat_names = []
    # integer CID (SNID may carry a -SEDINDEX suffix)
    written = to_written_cid(columnar_data['parameters']['cid'])
    for i in range(n_sne):
        cid = int(written[i])
        fname = f'{file_prefix}_SN{cid:0{cid_w}d}.DAT'
        with open(Path(output_dir) / fname, 'w') as fp:
            _write_text_header_block(columnar_data, config, resources,
                                     kcor_filter_names, zp_fluxcal, fp, hcol,
                                     i, sim_cols, host_cols, blindtest,
                                     trest_ranges)
            lo = int(hcol['PTROBS_MIN'][i]) - 1
            hi = int(hcol['PTROBS_MAX'][i])
            fp.write(f"\nNOBS: {hi - lo}\nNVAR: {len(obs_cols)}\n")
            fp.write("VARLIST: " + " ".join(n for n, _, _ in obs_cols) + "\n")
            mjd_trigger = hcol['MJD_TRIGGER'][i] if 'MJD_TRIGGER' in hcol else None
            pending_trigger = (mjd_trigger is not None
                               and 1000.0 < float(mjd_trigger) < 0.99e6)
            for r in range(lo, hi):
                # Marks the first epoch past the trigger, between the OBS
                # rows rather than in the header (text.c:1145).
                if pending_trigger and \
                        float(mjd_arr[r]) > float(mjd_trigger) + 1.0e-5:
                    fp.write("TRIGGER:  trigger logic satisfied at "
                             f"MJD_TRIGGER={float(mjd_trigger):.4f}\n")
                    pending_trigger = False
                vals = [f.format(arr[r]) for _, arr, f in obs_arrays]
                fp.write("OBS: " + " ".join(vals) + "\n")
            fp.write("END_PHOTOMETRY:\n")
            specs = (spectra_by_sn or {}).get(cid)
            if specs:
                _write_text_spec_block(kcor_filter_names, fp, specs, sim=not blindtest)
        dat_names.append(fname)
    return dat_names


def _write_text_header_block(columnar_data, config, resources,
                             kcor_filter_names, zp_fluxcal, fp, hcol, i,
                             sim_cols, host_cols, blindtest, trest_ranges):
    """Emit the `KEY: value` header for one SN row of the HEAD table."""
    def hv(col):
        return hcol[col][i] if col in hcol else None

    def pair(key, col, ecol, fmt='{:.6f}'):
        v = hv(col)
        if v is None:
            return
        e = hv(ecol)
        line = f"{key}: {fmt.format(float(v))}"
        if e is not None:
            line += f" +- {fmt.format(float(e))}"
        fp.write(line + "\n")

    bands = filters_string(kcor_filter_names)

    def filterpar(key, comment, fmt, col_of, wrap=14, absent=None,
                  default=None):
        """One key holding every band (wr_dataformat_text_FILTERPAR, :566).

        ``absent`` is the sentinel standing in for a quantity the HOSTLIB
        never supplied, and the key is skipped when every band holds it;
        SNANA gates those keys on HOSTGAL_USEMASK instead.
        """
        vals = [col_of(b) for b in bands]
        if any(v is None for v in vals):
            if default is None:
                return
            vals = [default] * len(bands)
        if absent is not None and all(float(v) == absent for v in vals):
            return
        fp.write(f"{key}: ")
        for n, v in enumerate(vals, start=1):
            fp.write(" " + fmt.format(float(v)))
            if n % wrap == 0 and n != len(vals):
                fp.write("\n    ")
        fp.write(f"  # {comment.format(bands=bands)}\n")

    # SURVEY(SUBSURVEY) where the entry named one (sntools_dataformat_text.c:115)
    sub = hv('SUBSURVEY')
    sub = sub.decode() if isinstance(sub, bytes) else sub
    survey = (f"{resources.survey_name}({sub})"
              if sub and sub != resources.survey_name else resources.survey_name)
    fp.write(f"SURVEY: {survey}\n")
    # a string: APPEND_SNID_SEDINDEX writes "CID-INDEX" (snlc_sim.c:25393-25398)
    fp.write(f"SNID: {str(hv('SNID')).strip()}\n")
    fp.write(f"ZP_FLUXCAL: {zp_fluxcal:.2f}\n")
    if hv('SNTYPE') is not None:
        fp.write(f"SNTYPE: {int(hv('SNTYPE'))}\n")
    fp.write(f"RA: {float(hv('RA')):.6f}\n")
    fp.write(f"DEC: {float(hv('DEC')):.6f}\n")
    # Atmosphere/DCR per-band averaged coords (WRFLAG_ATMOS, text.c:152)
    if bands and hv(f'RA_AVG_{bands[0]}') is not None:
        filterpar('RA_AVG', 'avg per band', '{:.5f}',
                  lambda b: hv(f'RA_AVG_{b}'))
        filterpar('DEC_AVG', 'avg per band', '{:.5f}',
                  lambda b: hv(f'DEC_AVG_{b}'))
    fp.write(f"FILTERS: {bands}\n")
    if hv('PIXSIZE') is not None and float(hv('PIXSIZE')) > 0:
        fp.write(f"PIXSIZE: {float(hv('PIXSIZE')):.4f}\n")
    fp.write(f"FAKE: {3 if blindtest else 2}\n")
    pair('MWEBV', 'MWEBV', 'MWEBV_ERR', '{:.4f}')
    mwxt = columnar_data['parameters'].get('mwxt_mag')
    if mwxt is not None:
        filterpar('MWXT_MAG', 'MW mag-XT per band', '{:.3f}',
                  lambda b: mwxt[i][bands.index(b)])
    fp.write(f"PEAKMJD: {float(hv('PEAKMJD')):.4f}\n")
    for key in ('MJD_TRIGGER', 'MJD_DETECT_FIRST', 'MJD_DETECT_LAST'):
        v = hv(key)
        if v is not None and 0.0 < float(v) < 1.0e6:
            fp.write(f"{key}: {float(v):.4f}\n")
    fp.write("\n")
    pair('REDSHIFT_HELIO', 'REDSHIFT_HELIO', 'REDSHIFT_HELIO_ERR')
    pair('REDSHIFT_FINAL', 'REDSHIFT_FINAL', 'REDSHIFT_FINAL_ERR')
    if hv('MASK_REDSHIFT_SOURCE') is not None:
        fp.write(f"MASK_REDSHIFT_SOURCE: {int(hv('MASK_REDSHIFT_SOURCE'))}\n")
    pair('VPEC', 'VPEC', 'VPEC_ERR', '{:.2f}')
    pair('LENSDMU', 'LENSDMU', 'LENSDMU_ERR', '{:.3f}')

    def block(cols, families):
        per_band = {f"{fam}_{b}" for fam, _, _, _, _ in families for b in bands}
        for c in cols:
            if c not in per_band and not _text_sim_suppressed(hv, c):
                fp.write(f"{c}: {_fmt_hdr_val(hv(c))}\n")
        for fam, comment, fmt, wrap, default in families:
            filterpar(fam, comment, fmt,
                      lambda b, f=fam: hv(f"{f}_{b}"), wrap, default=default)

    if host_cols:
        fp.write("\n")
        _write_text_hostgal(fp, hv, filterpar)
    if sim_cols:
        fp.write("\n")
        block(sim_cols, _TEXT_SIM_FILTERPAR)
        _write_text_simpar_extras(columnar_data, config, resources, fp, hcol,
                                  i, trest_ranges)


def _write_text_hostgal(fp, hv, filterpar):
    """Emit the host block for however many galaxies matched.

    SNANA loops over HOSTGAL_NMATCH2 galaxies, so the block length varies
    per event and is empty when nothing matched; the FITS columns are fixed
    width regardless (wr_dataformat_text_HOSTGAL, :616).
    """
    fp.write(f"HOSTGAL_NMATCH: {int(hv('HOSTGAL_NMATCH'))}\n")
    fp.write(f"HOSTGAL_NMATCH2: {int(hv('HOSTGAL_NMATCH2'))}\n")

    ngal = min(int(hv('HOSTGAL_NMATCH2')), MXHOSTGAL)
    for igal in range(ngal):
        prefix = HOSTGAL_PREFIXES[igal]
        for suffix, fmt, comment, sentinel in _TEXT_HOSTGAL_KEYS:
            v = hv(f'{prefix}_{suffix}')
            if v is None or (sentinel is not None and float(v) <= sentinel):
                continue
            cast = int if fmt == '{:d}' else float
            line = f"{prefix}_{suffix}: {fmt.format(cast(v))}"
            if suffix in _TEXT_HOSTGAL_PAIRED:
                e = hv(f'{prefix}_{suffix}_ERR')
                if e is not None:
                    line += f" +- {fmt.format(cast(e))}"
            fp.write(line + (f"  # {comment}\n" if comment else "\n"))
            if suffix == 'DDLR' and igal == 0:
                fp.write("HOSTGAL_CONFUSION: "
                         f"{float(hv('HOSTGAL_CONFUSION')):6.3f}\n")
        filterpar(f'{prefix}_MAG', '{bands}', '{:7.3f}',
                  lambda b, p=prefix: hv(f'{p}_MAG_{b}'), 10)
        filterpar(f'{prefix}_MAGERR', '{bands}', '{:7.3f}',
                  lambda b, p=prefix: hv(f'{p}_MAGERR_{b}'), 10)
        fp.write("\n")
        # the slot's photo-z quantile and LOGMASS(z) grids
        # (wr_dataformat_text_HOSTGALz: NBIN, then the lists when NBIN > 0)
        for grid, lists in (('QUANTILE_ZPHOT', ('QUANTILE_ZPHOT', 'QUANTILE_PERCENT')),
                            ('LOGMASS_ZGRID', ('LOGMASS_ZGRID', 'LOGMASS_VALGRID',
                                               'LOGMASS_ERRGRID'))):
            nbin = int(hv(f'{prefix}z_NBIN_{grid}'))
            fp.write(f"{prefix}z_NBIN_{grid}:  {nbin} \n")
            if nbin > 0:
                for j, name in enumerate(lists):
                    vals = np.asarray(hv(f'{prefix}z_{name}'))[:nbin]
                    fmt = '{:.4f}' if j == 0 else '{:.2f}'
                    fp.write(f"{prefix}z_{name}: "
                             + " ".join(fmt.format(float(v)) for v in vals) + " \n")
            fp.write("\n")

    for fam, comment, fmt, wrap, absent in _TEXT_HOST_FILTERPAR:
        filterpar(fam, comment, fmt,
                  lambda b, f=fam: hv(f"{f}_{b}"), wrap, absent)


def _text_sim_suppressed(hv, col):
    """Whether SNANA would omit this SIM key for this SN.

    The host-extinction trio is gated as a group on SIM_AV
    (sntools_dataformat_text.c:365); SIM_SUBSAMPLE_INDEX on a
    non-negative index (:273).
    """
    if col in _TEXT_SIM_FITS_ONLY:
        return True
    if col in ('SIM_AV', 'SIM_RV'):
        av = hv('SIM_AV')
        return av is not None and float(av) == SNANA_NULLFLOAT
    if col == 'SIM_SUBSAMPLE_INDEX':
        return int(hv(col)) < 0
    return False


def _write_text_simpar_extras(columnar_data, config, resources, fp, hcol, i,
                              trest_ranges):
    """SIMPAR keys SNANA writes to TEXT but not to FITS, so they have no
    header-table column to carry them (sntools_dataformat_text.c:285-446)."""
    mw = mw_option_keys(config)
    fp.write(f"SIMLIB_MSKOPT: {config.survey.simlib_mskopt}\n")
    fp.write(f"SIM_MWRV: {mw['SIM_MWRV']:.3f}  # MilkyWay RV\n")
    fp.write(f"SIMOPT_MWCOLORLAW: {mw['SIMOPT_MWCOLORLAW']}  # MW color-law option\n")
    fp.write(f"SIMOPT_MWEBV: {mw['SIMOPT_MWEBV']}  # MWEBV option\n")

    if 'SIM_AV' in hcol and float(hcol['SIM_AV'][i]) != SNANA_NULLFLOAT:
        fp.write(f"SIM_AVTAU: {config.host.tau_av:.3f}"
                 "     # dN/dAV = exp(-AV/AVTAU)\n")

    eff = columnar_data['parameters'].get('sim_searcheff_spec')
    if eff is not None:
        fp.write(f"SIM_SEARCHEFF_SPEC: {float(eff[i]):.4f}"
                 "  # spectro-search efficiency\n")

    for line in resources.model.text_sim_header_lines(
            hcol, columnar_data['parameters'], i):
        fp.write(line)

    fp.write(f"SIM_HOSTLIB_NPAR: {len(config.host.storepar)}\n")

    # over the EPCUTWIN-filtered epochs (gen_cutwin, snlc_sim.c:24199-24241)
    if trest_ranges is not None:
        fp.write(f"SIM_TRESTMIN: {float(trest_ranges[0][i]):.2f}  # days\n")
        fp.write(f"SIM_TRESTMAX: {float(trest_ranges[1][i]):.2f}  # days\n")

    fp.write("SIM_RISETIME_SHIFT: 0.0   # days\n")
    fp.write("SIM_FALLTIME_SHIFT: 0.0   # days\n")


def _fmt_hdr_val(x):
    """Format a scalar header value (int/float/str) for TEXT output."""
    if isinstance(x, (bool, np.bool_)):
        return str(int(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.6g}"
    return str(x).strip()


def _write_text_spec_block(kcor_filter_names, fp, specs, *, sim=True):
    """Emit the SNSPEC section (wr_dataformat_text_SNSPEC, text.c:1170)."""
    varnames = ["LAMMIN", "LAMMAX", "FLAM", "FLAMERR"]
    # GENSPEC.USE_WARP (sntools_dataformat_text.c:1209-1210)
    use_warp = bool(specs) and all(s.warp is not None for s in specs)
    if sim:
        varnames += ["SIM_GENFLAM", "SIM_GENMAG"]
        if use_warp:
            varnames += ["SIM_WARP"]
        varnames += ["SNR"]
    fp.write(f"\nNSPECTRA: {len(specs)}\n")
    fp.write(f"NVAR_SPEC: {len(varnames)}\n")
    fp.write("VARNAMES_SPEC: " + " ".join(varnames) + "\n")
    for sid, s in enumerate(specs, start=1):
        fp.write(f"SPECTRUM_ID: {sid}\n")
        fp.write(f"SPECTRUM_MJD: {s.mjd:.3f}\n")
        # sntools_dataformat_text.c:1238-1240
        if s.scale_host_contam > 0.0:
            fp.write(f"SPECTRUM_SCALE_HOST_CONTAM: {s.scale_host_contam:.3f}\n")
        fp.write(f"SPECTRUM_TEXPOSE: {s.texpose:.1f}\n")
        fp.write(f"SPECTRUM_INSTRUMENT: {s.instrument}\n")
        fp.write(f"SPECTRUM_SNR_COMPUTE: {s.snr_compute:.3f}\n")
        fp.write(f"SPECTRUM_LAMOBS_SNR: {s.lammin_snr:.1f} {s.lammax_snr:.1f}\n")
        centres = 0.5 * (np.asarray(s.lam_min) + np.asarray(s.lam_max))
        fp.write(f"SPECTRUM_LAMRANGE: {centres.min():7.1f} {centres.max():7.1f}"
                 "  # valid wave range for spectrum\n")
        fp.write(f"SPECTRUM_NLAM: {len(s.lam_min):4d}  {s.n_lam_total:4d}"
                 "         # Number of wave bins: VALID  TOTAL\n")
        if sim and s.synmag is not None:
            vals = [s.synmag[b] for b in filters_string(kcor_filter_names)]
            fp.write("SIM_SYNMAG: "
                     + "".join(f" {v:8.4f}" for v in vals)
                     + f"  # {filters_string(kcor_filter_names)}\n")
        for k in range(len(s.lam_min)):
            row = [f"{float(s.lam_min[k]):.2f}", f"{float(s.lam_max[k]):.2f}",
                   f"{float(s.flam[k]):.3e}", f"{float(s.flamerr[k]):.3e}"]
            if sim:
                # SNR is the true-model GENFLAM/FLAMERR, not the noisy flux
                # ratio (sntools_dataformat_text.c:1317-1322).
                snr = float(s.sim_flam[k]) / float(s.flamerr[k]) \
                    if s.flamerr[k] > 0 else 0.0
                row += [f"{float(s.sim_flam[k]):.3e}", f"{float(s.sim_genmag[k]):.2f}"]
                if use_warp:
                    row += [f"{float(s.warp[k]):6.3f}"]
                row += [f"{snr:.2f}"]
            fp.write("SPEC: " + " ".join(row) + "\n")
        fp.write("SPECTRUM_END:  \n\n")
