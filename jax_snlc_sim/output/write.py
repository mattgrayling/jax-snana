"""Write simulation light curves, spectra and companion files in SNANA formats."""

import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from jax_snlc_core import logging as log
from jax_snlc_core.filters import filter_band
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import SNANA_VERSION_STRING
from jax_snlc_sim.distributions import GenGaussAsym
from jax_snlc_sim.output.dump import (cutwin_reject_rows, resolve_dump_content,
                                      write_dump_dcr, write_dump_file,
                                      write_dump_mwcl, write_dump_noise,
                                      write_dump_rate, write_dump_trainsalt)
from jax_snlc_sim.output.fits import (gzip_output_fits, primary_keywords, write_head_fits,
                                      write_phot_fits, write_spec_fits)
from jax_snlc_sim.output.tables import (build_columnar_data, build_header_columns, build_phot_columns,
                                        check_output_schema, compute_derived_values,
                                        effective_gentype, generates_snia, gentype_to_name)
from jax_snlc_sim.output.text import write_text_lightcurves
from jax_snlc_core.random import root_key, stream_key
from jax_snlc_sim.observations import (row_slice, simlib_epochs)
from jax_snlc_sim.simulation import LightCurveSimulator, SimulationConfig, parse_format_mask

logger = logging.getLogger(__name__)

# where a combined version keeps each run's README (submit_prog_sim.py)
COMBINED_README_DIR = 'misc'


def write_snana_outputs(output_dir: str, version_name: str, columnar_data: Dict,
                        snana_input_file: str, config: SimulationConfig,
                        resources: LightCurveSimulator, kcor_filter_names: List[str],
                        model, full_results: Optional[Dict] = None,
                        selection_result: Optional[Dict] = None,
                        spectra_by_sn: Optional[Dict] = None,
                        to_written_cid=None, *, makes_spectra: bool,
                        seconds_since_init: float, combined: bool = False) -> Dict:
    """Write every SNANA output file serialised from a simulation: HEAD/PHOT
    (or TEXT ``.DAT``), SPEC.FITS, and the ``.DCR``/``.DUMP``/``.LIST``/
    ``.README`` companions, then gzip the FITS if configured.

    Resource-dependent outputs (VERIFY_SED_TRUE, FILTERS transmission curves)
    are prepared by save_snana_outputs before and after this dataset write.

    ``columnar_data`` holds the written events; ``full_results`` and
    ``selection_result`` hold every generated one, which the DUMPALL columns
    and the README statistics need. ``seconds_since_init`` is the wall time
    since setup ended, SNANA's ``t_end - t_end_init``.

    ``combined`` says this run is one of several written into the version, as
    SNANA's batch layer combines jobs or models into one: the directory is
    taken as found, the ``.LIST`` gains this run's HEAD, the dumps are named
    for GENPREFIX so each run keeps its own, and the README goes under
    ``misc/`` with that name, for :func:`write_combined_readme` to merge.

    Returns the per-phase timings and the HEAD and PHOT columns as written,
    less the FITS files' marker rows, which PTROBS steps over.
    """
    start_time = time.time()

    snana_output_dir = Path(output_dir) / version_name
    # clr_VERSION, sntools.c:8784
    if not combined and (snana_output_dir / f'{version_name}.LIST').exists():
        log.line(logger, f"removing previous {snana_output_dir}")
        shutil.rmtree(snana_output_dir)
    snana_output_dir.mkdir(parents=True, exist_ok=True)
    log.line(logger, f"Writing SNANA output to {snana_output_dir}")

    file_prefix = config.output.genprefix or version_name
    aux_name = file_prefix if combined else version_name
    n_sne = len(columnar_data['parameters']['sim_redshift_cmb'])
    total_observations = np.sum(columnar_data['obs_counts'])
    check_output_schema(columnar_data, n_sne, int(total_observations))
    zp_fluxcal = config.instrument.zp_fluxcal
    # SIM_SNRMAG%2.2d, snlc_sim.c:7441
    _magmon = config.output.magmonitor_snr
    snrmon_colname = f'SIM_SNRMAG{_magmon:02d}' if _magmon else None
    gentype = effective_gentype(config)
    dumpall_requested = config.output.simgen_dumpall

    timing = {}

    phase_start = time.time()
    distance_moduli, peak_magnitudes = compute_derived_values(
        columnar_data, full_results, kcor_filter_names, n_sne,
        selection_result)
    timing['derived_values'] = time.time() - phase_start
    log.line(logger, f"Computed derived values ({timing['derived_values']:.2f}s)",
             indent=2)

    phase_start = time.time()
    header_cols = build_header_columns(columnar_data, config, resources,
                                       kcor_filter_names, distance_moduli,
                                       model, n_sne,
                                       peak_magnitudes, to_written_cid)
    timing['header_table'] = time.time() - phase_start
    log.line(logger, f"Built HEAD table: {n_sne} rows, "
             f"{len(header_cols)} cols ({timing['header_table']:.2f}s)",
             indent=2)

    phase_start = time.time()
    phot_cols = build_phot_columns(columnar_data, resources, snrmon_colname,
                                   total_observations)
    timing['phot_table'] = time.time() - phase_start
    log.line(logger, f"Built PHOT table: {total_observations} rows, "
             f"{len(phot_cols)} cols ({timing['phot_table']:.2f}s)",
             indent=2)

    flags = parse_format_mask(config.output.format_mask)
    writes_spectra, writes_sed_true = _spectra_write_flags(
        config, flags, makes_spectra)
    if not writes_spectra:
        spectra_by_sn = {}

    # BLINDTEST(8): data-like output, drop SIM_* truth, FAKE=3 (snlc_sim.c:7428)
    if flags['blindtest']:
        for c in [c for c in header_cols if c.startswith('SIM_')]:
            del header_cols[c]
        if 'FAKE' in header_cols:
            header_cols['FAKE'][:] = 3
        for c in [c for c in phot_cols if c.startswith('SIM_')]:
            del phot_cols[c]
    # COMPACT(64): only the WRFULL columns go; SIM_MAGOBS and the SNR monitor
    # are added outside that block (sntools_dataformat_fits.c:820-906)
    if flags['compact'] and flags['fits']:
        keep = {'MJD', 'BAND', 'FLUXCAL', 'FLUXCALERR', 'SIM_MAGOBS'}
        if snrmon_colname:
            keep.add(snrmon_colname)
        if config.instrument.smearflag_flux == 0:
            keep -= {'FLUXCAL', 'FLUXCALERR'}
        for c in [c for c in phot_cols if c not in keep]:
            del phot_cols[c]

    # resolved before any data file is written (wr_SIMGEN_DUMP, snlc_sim.c:14833)
    phase_start = time.time()
    dump_content = None
    if config.output.simgen_dump or config.output.simgen_dumpall:
        include_all_dump = bool(
            dumpall_requested and full_results and selection_result)
        dump_rows = (None if include_all_dump
                     else cutwin_reject_rows(config, full_results, selection_result))
        if dump_rows is not None:
            include_all_dump = True
        dump_content = resolve_dump_content(
            columnar_data, config, resources, full_results, kcor_filter_names,
            model, peak_magnitudes, selection_result, include_all_dump,
            to_written_cid, dump_rows)
    timing['dump'] = time.time() - phase_start

    phase_start = time.time()
    dat_names = None
    if flags['fits']:
        def keywords(head):
            return primary_keywords(
                config, resources, kcor_filter_names, file_prefix, model,
                snrmon_colname, version_name, zp_fluxcal, header_cols, flags,
                writes_spectra, writes_sed_true, head)
        write_head_fits(keywords(head=True), file_prefix, snana_output_dir,
                        header_cols)
        write_phot_fits(keywords(head=False), file_prefix, snana_output_dir,
                        phot_cols, columnar_data['obs_counts'])
        if writes_spectra:
            write_spec_fits(config, keywords(head=False), kcor_filter_names,
                            file_prefix, snana_output_dir, spectra_by_sn,
                            to_written_cid)
    elif flags['text']:
        passing = np.asarray(selection_result['passing_mask'], dtype=bool)
        trest_ranges = (
            np.asarray(selection_result['trestmin'])[passing],
            np.asarray(selection_result['trestmax'])[passing])
        dat_names = write_text_lightcurves(
            columnar_data, config, resources, kcor_filter_names, file_prefix,
            snrmon_colname, zp_fluxcal,
            snana_output_dir, header_cols, phot_cols, to_written_cid,
            trest_ranges, spectra_by_sn, blindtest=flags['blindtest'])
    timing['data_files'] = time.time() - phase_start
    log.line(logger, f"Wrote data files ({timing['data_files']:.2f}s)", indent=2)

    if config.instrument.atmos_optmask & 512:
        write_dump_dcr(columnar_data, aux_name, snana_output_dir, to_written_cid)

    if config.output.simgen_dump_rate > 0:
        write_dump_rate(config, aux_name, snana_output_dir)
    if config.output.simgen_dump_mwcl > 0:
        write_dump_mwcl(aux_name, snana_output_dir)
    if config.output.simgen_dump_trainsalt > 0:
        write_dump_trainsalt(columnar_data, aux_name, snana_output_dir, to_written_cid)
    if config.output.simgen_dump_noise > 0:
        write_dump_noise(full_results, config, resources, aux_name,
                         snana_output_dir)

    phase_start = time.time()
    if dump_content is not None:
        write_dump_file(dump_content, aux_name, snana_output_dir)
    if flags['text']:
        _write_list_file(version_name, snana_output_dir,
                         dat_names or [], suffix='', append=combined)
    else:
        # The FITS-mode LIST carries the actual HEAD file name
        # (WR_SNFITSIO_INIT returns it, snlc_sim.c:31000-31008).
        _write_list_file(version_name, snana_output_dir,
                         [f'{file_prefix}_HEAD'], append=combined)
    n_spectra = sum(len(v) for v in spectra_by_sn.values())
    # NGENTOT_LC as resolved at setup (sntools_sim_readme.c:1332)
    # a run that is one of several keeps its README under misc/, as SNANA's
    # batch layer moves each job's there (submit_prog_sim.py, merge_job_wrapup)
    readme_dir = snana_output_dir / COMBINED_README_DIR if combined else snana_output_dir
    readme_dir.mkdir(exist_ok=True)
    _write_readme_file(config, resources, gentype, generates_snia(model),
                       event_counts(selection_result)[0],
                       snana_input_file, version_name, aux_name,
                       readme_dir,
                       _gentype_map(config, model, selection_result['generated'])
                       + _output_summary(config, resources, full_results,
                                         selection_result, n_spectra,
                                         seconds_since_init))
    timing['auxiliary'] = time.time() - phase_start
    log.line(logger, f"Wrote auxiliary files ({timing['auxiliary']:.2f}s)",
             indent=2)

    # gzip all FITS (HEAD/PHOT/SPEC) at the end, like SNANA `gzip *.FITS`
    phase_start = time.time()
    if config.output.gzip_data_files:
        gzip_output_fits(snana_output_dir)
    timing['compress'] = time.time() - phase_start

    elapsed = time.time() - start_time
    timing['total'] = elapsed
    rate = n_sne / elapsed if elapsed else 0.0
    log.line(logger, f"Wrote output in {elapsed:.2f}s ({rate:.1f} SNe/sec)")
    log.line(logger, "Output phases: " + "  ".join(
        f"{k} {v:.2f}s" for k, v in timing.items() if v >= 0.005), indent=2)

    return timing, header_cols, phot_cols


def _spectra_write_flags(config, flags, makes_spectra):
    """Whether spectra, and the true SED, are written (snlc_sim.c:30958-30985).
    Under DCR the spectrograph runs for the atmosphere alone, so both follow
    the user's own SPECTROGRAPH_OPTMASK rather than the effective one. The
    mask's NOSPEC bit skips the spectra (snlc_sim.c:10294), so nothing is
    written either."""
    if config.instrument.atmos_optmask > 0:
        asked = bool(config.spectra.spectrograph_optmask & 128)
        return asked, asked
    skipped = flags['nospec'] or bool(config.spectra.spectrograph_optmask & 2048)
    return (makes_spectra and not skipped,
            config.spectra.lambin_sed_true > 0.0)


def _write_list_file(version_name, output_dir: str, file_list: List[str],
                     suffix: str = '.FITS', append: bool = False) -> Path:
    """Write LIST file (one data-file name per line); ``append`` adds to the
    names a version already lists."""
    output_file = Path(output_dir) / f'{version_name}.LIST'

    content = '\n'.join([f'{fname}{suffix}' for fname in file_list]) + '\n'

    with open(output_file, 'a' if append else 'w') as f:
        f.write(content)

    log.line(logger, f"wrote {output_file}", indent=2)
    return output_file


def event_counts(selection_result) -> Tuple[int, int, int]:
    """SNANA's NGENEV_TOT, NGENLC_TOT and NGENLC_WRITE: every generated
    event, those past the GENRANGE cuts (snlc_sim.c:255, :292) and those
    written."""
    codes = selection_result['generated']['rejection_codes']
    return codes.size, int((codes != 1).sum()), int(selection_result['n_passed'])


# input keys a combined README leaves out of each run's, since they belong to
# one job (merge_write_input_keys, submit_prog_sim.py)
_COMBINED_IGNORED_KEYS = ('GENVERSION', 'NGENTOT_LC', 'NGEN_LC', 'CIDRAN_MIN', 'CIDRAN_MAX')


def write_combined_readme(version_dir, version_name: str, parts) -> Path:
    """The README of a version several runs were written into, as SNANA's
    batch layer writes it (merge_write_readme, submit_prog_sim.py): the summed
    statistics, the input files, every run's GENTYPE_TO_NAME lines labelled
    by its model, and every run's input keys, its FILTERS block among them.

    ``parts`` is each run's (input file, README name), its README under
    ``misc/``. SNANA lists the Type Ia models first and numbers the models
    in that order, SNIaMODEL00, NONIaMODEL01 and so on.
    """
    version_dir = Path(version_dir)
    runs = []
    for input_file, name in parts:
        readme = version_dir / COMBINED_README_DIR / f'{name}.README'
        runs.append((input_file, readme,
                     yaml.safe_load(readme.read_text())['DOCUMENTATION']))
    runs.sort(key=lambda run: not run[2]['OVERVIEW']['SNIA'])
    kinds = ['SNIa' if docana['OVERVIEW']['SNIA'] else 'NONIa' for _, _, docana in runs]
    labels = [f'{kind}MODEL{i:02d}' for i, kind in enumerate(kinds)]

    def total(key):
        return sum(docana['OUTPUT_SUMMARY'][key] for _, _, docana in runs)

    lines = ['DOCUMENTATION:', f'  GENVERSION: {version_name} ', '',
             '#                   NLC_GEN   NLC_WRITE  NSPEC_WRITE  CPU(minutes)',
             '  STAT_SUMMARY: ',
             f"  - {'TOTAL':<12}   {total('NGENLC_TOT'):8}   {total('NGENLC_WRITE'):8}  "
             f"{total('NGENSPEC_WRITE'):8}        {round(total('CPU_MINUTES'), 2)}",
             '', '  INPUT_FILES:']
    # SNANA's first line names its own master input file, which a jax run has none of
    lines += [f'  - {kind:<8}  {input_file} ' for kind, (input_file, _, _) in zip(kinds, runs)]
    lines += ['', f'  SUBMIT_DIR:  {os.getcwd()}', f'  SNANA_VERSION:  {SNANA_VERSION_STRING}',
              f"  TIME_STAMP_SUBMIT:  {datetime.now().strftime('%Y-%m-%d_%H:%M')} "]

    names = {}
    for label, (_, _, docana) in zip(labels, runs):
        for gentype, name in docana['GENTYPE_TO_NAME'].items():
            names[gentype] = f'{name:<30} {label}'
    lines += ['', '  GENTYPE_TO_NAME:   # GENTYPE-integer    (non)Ia   transient-Name     FITS-prefix']
    lines += [f'    {gentype}:  {name}  ' for gentype, name in names.items()]

    # each run's lines from INPUT_KEYS to the next section, which in SNANA's
    # README is INPUT_NOTES and in this one is GENTYPE_TO_NAME
    for label, (_, readme, _) in zip(labels, runs):
        lines += ['', '# - - - - - - - - - - - - - - - - - - - - - - - - - - - - ', '',
                  f'  INPUT_KEYS_{label}: ']
        copying = False
        for line in readme.read_text().splitlines():
            if line.startswith(('  INPUT_NOTES:', '  GENTYPE_TO_NAME:')):
                copying = False
            if copying and line.strip().partition(':')[0] not in _COMBINED_IGNORED_KEYS:
                lines.append(line)
            if line.startswith('  INPUT_KEYS:'):
                copying = True
    lines.append('DOCUMENTATION_END:')

    output_file = version_dir / f'{version_name}.README'
    output_file.write_text('\n'.join(lines) + '\n')
    log.line(logger, f"wrote {output_file}", indent=2)
    return output_file


def _filters_block(config, resources) -> str:
    """SNANA's FILTERS block (README_DOCANA_FILTERS, sntools_sim_readme.c:303):
    the mean wavelength of each GENFILTERS band, which SCONE reads."""
    mean = dict(zip((filter_band(name) for name in resources.photometry.filter_names),
                    np.asarray(resources.photometry.filter_mean_wavelengths)))
    lines = ['# ', '  FILTERS:   # mean wavelength (A)']
    lines += [f'    {band}:  {mean[band]:.1f}' for band in config.survey.genfilters]
    return '\n'.join(lines) + '\n'


def _gentype_map(config, model, generated) -> str:
    """SNANA's GENTYPE_TO_NAME block (README_DOCANA_GENTYPE_MAP,
    sntools_sim_readme.c:466), which classifiers read to tell the Type Ia
    GENTYPEs from the rest. Ia or nonIa is the model's, for every line."""
    kind = 'Ia' if generates_snia(model) else 'nonIa'
    lines = ['#', '  GENTYPE_TO_NAME:  # GENTYPE:  [non]Ia   type  ']
    lines += [f'    {gentype}:  {kind:<6}   {name}'
              for gentype, name in gentype_to_name(config, model, generated).items()]
    return '\n'.join(lines) + '\n'


def _output_summary(config, resources, full_results, selection_result,
                    n_spectra, seconds) -> str:
    """SNANA's OUTPUT_SUMMARY statistics (README_DOCANA_OUTPUT_SUMMARY,
    sntools_sim_readme.c:550-685). Lines SNANA writes only for a non-zero
    count are omitted the same way; the implausible-flux count is always
    zero here because jax aborts on one instead. RANDOM_SYNC describes
    SNANA's random lists and has no counterpart."""
    n_event, n_lc, n_write = event_counts(selection_result)
    generated = selection_result['generated']
    codes = generated['rejection_codes']
    passing = np.asarray(selection_result['passing_mask'], dtype=bool)
    specid = passing & np.asarray(selection_result['specid_pass'], dtype=bool)
    zhost = passing & np.asarray(selection_result['zhost_pass'], dtype=bool)
    params = full_results['parameters']

    # gen_event_stats, snlc_sim.c:24018-24047
    if n_lc > 0:
        eff = n_write / n_lc
        eff_err = np.sqrt(n_write * (n_lc - n_write) / n_lc**3)
        if eff_err == 0.0:
            eff_err = 1.0 / n_lc
    else:
        eff, eff_err = 0.0, 1.0
    n_season = resources.season_count
    n_accept = n_season * eff
    accept_err = n_accept / np.sqrt(n_write) if n_write > 0 else 0.0

    def per_second(n):
        return n / seconds if seconds > 0 else 0.0

    pad = '    '
    lines = [
        '# ',
        '  OUTPUT_SUMMARY:',
        f'{pad}CPU_MINUTES:       {seconds / 60:.2f}  ',
        f'{pad}NGENEV_TOT:        {n_event}    '
        f'# ({per_second(n_event):.0f}/sec, total events)',
        f'{pad}NGENLC_TOT:        {n_lc}    # ({per_second(n_lc):.0f}/sec, total LC)',
        f'{pad}NGENLC_WRITE:      {n_write}    '
        f'# ({per_second(n_write):.0f}/sec, LC passing trigger)',
    ]
    if resources.simlib_subsurvey_list:
        names = np.array([f.subsurvey for f in resources.fields])
        by_event = names[generated['field_idx']]
        lines.append(f'{pad}NGENLC_SUBSURVEY:')
        for j, name in enumerate(resources.simlib_subsurvey_list):
            mine = by_event == name
            lines.append(
                f'{pad}  {name + ":":<12}  {int((mine & (codes != 1)).sum()):5d} '
                f'{int((mine & generated["passing"]).sum()):5d}   '
                f'{"# NTOT NWRITE" if j == 0 else ""}')
    lines += [
        f'{pad}NGENSPEC_WRITE:    {n_spectra}  ',
        f'{pad}EFF(SEARCH+CUTS): {eff:7.4f} +- {eff_err:7.4f}',
        f'{pad}NACCEPT:  [ {int(specid.sum())}, {int((passing & ~specid).sum())}, '
        f'{int(zhost.sum())} ]   # NSN(ACCEPT) for [ SpecID, noSpecID, zHOST]',
        f'{pad}NGEN_PER_SEASON:   {n_season:.0f}       '
        '# NSN(GENLC) in GENRANGE(z,MJD,dOmega)',
        f'{pad}NACC_PER_SEASON:   {n_accept:.0f} +_ {accept_err:.0f}  '
        '# NSN(ACCEPT) per season  after trigger+cuts',
        f'{pad}NREJECT:  [{int((codes == 1).sum())},   {int((codes == 2).sum())}, '
        f'{int(((codes == 3) | (codes == 4)).sum())}, {int((codes == 5).sum())}]   '
        f'# [GENRANGE,   NEP<{config.cuts.nepoch_min}, SEARCHEFF, CUTWIN] ',
    ]
    hostlib = resources.hostlib
    if hostlib is not None and hostlib.wronghost is not None:
        n_wrong = int((passing & ~specid
                       & (np.asarray(params['CORRECT_HOSTMATCH']) == 0)).sum())
        frac = n_wrong / n_write if n_write > 0 else 0.0
        lines.append(f'{pad}NWRONGHOST_WRITE:   {n_wrong}    # frac = {frac:.4f}')
    if hostlib is not None and n_write > 0:
        lines += _hostmatch_lines(config, params, passing, pad)
    return '\n'.join(lines) + '\n'


def _hostmatch_lines(config, params, passing, pad) -> List[str]:
    """Hostless and multi-host counts of the written events in three redshift
    bins over GENRANGE_REDSHIFT (update_hostmatch_counters, snlc_sim.c:30788;
    readme_docana_hostmatch, sntools_sim_readme.c:692)."""
    zmin, zmax = config.rates.redshift_range
    zbin = (zmax - zmin) / 3
    z = np.asarray(params['sim_redshift_cmb'])[passing]
    # a redshift on a bin edge belongs to the later bin, the top edge included
    iz = np.minimum(((z - zmin) / zbin).astype(int), 2)
    n = np.bincount(iz, minlength=3)
    n0 = np.bincount(iz, np.asarray(params['host_NMATCH'])[passing] == 0,
                     minlength=3).astype(int)
    n2 = np.bincount(iz, np.asarray(params['host_NMATCH2'])[passing] > 1,
                     minlength=3).astype(int)
    ranges = ','.join(f' {zmin + zbin * i:.2f}-{zmin + zbin * (i + 1):.2f}'
                      for i in range(3))

    def counts(values):
        return ','.join(f' {v:3d}' for v in values)

    def fractions(values):
        return ','.join(f' {v / m if m else 0.0:.3f}' for v, m in zip(values, n))

    return [
        f'{pad}HOST_NMATCH:   # accepted',
        f'{pad}- 0 : [ {counts(n0)} ]  # z-ranges: {ranges}',
        f'{pad}- 2 : [ {counts(n2)} ] ',
        f'{pad}HOST_NMATCH_FRACTIONS:  # accepted ',
        f'{pad}- 0 : [ {fractions(n0)} ]  # z-ranges: {ranges}',
        f'{pad}- 2 : [ {fractions(n2)} ] ',
    ]


def _write_readme_file(config, resources, effective_gentype, snia: bool,
                       n_generated, snana_input_file, version_name, file_name,
                       output_dir: str, closing: str) -> Path:
    """Write README file (DOCUMENTATION block with generation settings),
    named ``file_name``, which is the version unless a run is being added to
    one already written. ``closing`` is the text after the input keys: the
    GENTYPE_TO_NAME block and the OUTPUT_SUMMARY."""
    output_file = Path(output_dir) / f'{file_name}.README'

    # sntools_sim_readme.c:227
    current_time = (os.environ.get('SNANA_TIME_START')
                    or config.output.time_start
                    or datetime.now().strftime("%Y-%m-%d  %H:%M"))

    cosmology = config.cosmology
    omega_m = cosmology.omega_matter
    omega_l = cosmology.omega_lambda
    w0 = cosmology.w0
    wa = cosmology.wa

    redshift_range = config.rates.redshift_range
    genfilters = config.survey.genfilters
    kcor_file = config.survey.kcor_file
    simlib_file = config.survey.simlib_file

    # SIM_SUBPOP carries the class index, so the names are listed in that order
    classes = [sp.name for sp in config.model.subpops]

    model_input_lines = [
        line
        for name, pop in sorted(config.model.populations.items())
        for d, tag in zip(pop.distributions,
                          [f" ({c})" for c in classes]
                          if len(pop.distributions) > 1 else [''])
        if isinstance(d, GenGaussAsym)
        for line in ((f"GENPEAK_{name}{tag}", f"{d.peak}"),
                     (f"GENSIGMA_{name}{tag}", f"{d.sigma[0]} {d.sigma[1]}"),
                     (f"GENRANGE_{name}{tag}", f"{d.range[0]}  {d.range[1]}"))]

    host = config.host
    if host.dust_enabled:
        av_range = host.av_range
        tau_av = host.tau_av
    else:
        av_range = (0.0, 0.0)
        tau_av = 0.0

    # the values setup resolved, written whatever the input said
    # (VERSION_INFO_load, sntools_sim_readme.c); a bare string is a comment
    resolved = [
        "# Output data",
        ("GENVERSION", version_name),
        ("NGENTOT_LC", n_generated),
        ("RANSEED", config.run.random_seed),
        ("FORMAT_MASK", f"{config.output.format_mask}  #  += 2,32,16 -> TEXT, FITS, randomCID"),
        ("WRFLAG_MODELPAR", config.output.wrflag_modelpar),
        ("GENTYPE", f"{effective_gentype}  # true type"),
        ("SNTYPE", f"{config.rates.sntype_spec}  {config.rates.sntype_phot}  "
                   f"# spec Type, photID type"),
        "# Source model",
        ("GENMODEL", config.model.genmodel),
        ("GENMODEL_MSKOPT", config.model.genmodel_mskopt),
        ("GENMAG_SMEAR", config.model.genmag_smear),
        ("GENMAG_SMEAR_MODELNAME", config.model.genmag_smear_modelname or 'NONE'),
        "# Cosmology inputs",
        ("OMEGA_MATTER", f"{omega_m:.4f}"),
        ("OMEGA_LAMBDA", f"{omega_l:.4f}"),
        ("w0_LAMBDA", f"{w0:.4f}"),
        ("wa_LAMBDA", wa),
        "# Instrumental inputs",
        ("GENFILTERS", genfilters),
        ("CALIB_FILE", kcor_file),
        ("SIMLIB_FILE", simlib_file),
        "# SEARCHEFF/detections",
        ("APPLY_SEARCHEFF_OPT", f"{config.survey.apply_searcheff_opt}  "
                               f"# += 1,2,4 => pipe,spec,zhost"),
        "# Redshift inputs",
        ("GENRANGE_REDSHIFT", f"{redshift_range[0]:.4e}  {redshift_range[1]:.4f}"),
        ("GENSIGMA_REDSHIFT", f"{config.rates.gensigma_redshift:.4e}"),
        "# SN model inputs",
        *[("SUBPOP", f"{index} {sp.name}  WGT {sp.weight}")
          for index, sp in enumerate(config.model.subpops)],
        *model_input_lines,
        "# Host extinction inputs",
        ("GENRANGE_AV", f"{av_range[0]:.2f}   {av_range[1]:.1f}"),
        ("GENTAU_AV", f"{tau_av:.3f}"),
    ]
    written = {entry[0] for entry in resolved if not isinstance(entry, str)}
    # every other key the input gave, as it gave it (README_KEYPLUSARGS_load)
    echoed = [text for key, text in config.run._input_lines if key not in written]
    key_lines = [entry if isinstance(entry, str)
                 else f"    {entry[0] + ':':<24} {entry[1]}" for entry in resolved]
    if echoed:
        key_lines += ["# Remaining inputs, as given"] + [f"    {line}" for text in echoed for line in text.splitlines()]

    content = (
        f"DOCUMENTATION:\n"
        f"  OVERVIEW:\n"
        f"    TIME_START:   {current_time}  # UT\n"
        f"    SURVEY:       {resources.survey_name}\n"
        f"    GENMODEL:     {config.model.genmodel}\n"
        f"    SNIA:         {snia}\n"
        f"    HOST_MACHINE: JAX-SNANA\n"
        f"    USERNAME:     jax_user\n"
        f"    SNANA_VERSION: {SNANA_VERSION_STRING}\n"
        f"    CWD:   {os.getcwd()}\n"
        f"    INPUT_FILE:\n"
        f"    - {Path(snana_input_file).name}\n"
        f"# \n"
        f"  INPUT_KEYS:\n"
        + "\n".join(key_lines) + "\n"
        + _filters_block(config, resources)
        + closing
        + "DOCUMENTATION_END:\n"
    )

    with open(output_file, 'w') as f:
        f.write(content)

    log.line(logger, f"wrote {output_file}", indent=2)
    return output_file


def draw_cidran(simulation, cids):
    """FORMAT_MASK CIDRAN(16): unique random CIDs strictly between
    CIDRAN_MIN and CIDRAN_MAX, less CIDRAN_SKIPLIST, one per event
    (init_CIDRAN). The list is keyed by RANSEED alone, so jobs sharing a
    seed take disjoint slices of one list.

    As SNANA does, the first CIDOFF - CIDRAN_MIN draws are another job's,
    and a draw that repeats an earlier one or falls on the skip list is
    redrawn, all such draws at once per round; not byte-matched to SNANA.
    SNANA never draws either end of the range (snlc_sim.c:23430-23431) and
    aborts once CIDOFF - CIDRAN_MIN plus the event count reaches CIDRAN_MAX
    (:23366), and so does this.
    """
    cids = np.asarray(cids)
    n = len(cids)
    cfg = simulation
    cmin, cmax, coff = cfg.output.cidran_min, cfg.output.cidran_max, cfg.output.cidoff
    if coff < cmin:
        raise SNANAError(f"CIDOFF={coff} must be >= CIDRAN_MIN={cmin}")
    skip = jnp.asarray(sorted(c for c in set(cfg.output.cidran_skiplist)
                              if cmin < c < cmax), dtype=jnp.int32)
    off = coff - cmin
    m = off + n
    if m >= cmax:
        # init_CIDRAN's NSTORE_ALL bound (snlc_sim.c:23366)
        raise SNANAError(f"CIDOFF-CIDRAN_MIN + {n} events = {m} reaches "
                         f"CIDRAN_MAX={cmax}")
    # SNANA never draws either end of the range (snlc_sim.c:23430-23431)
    pool = cmax - cmin - 1
    if m + skip.shape[0] > pool:
        raise SNANAError(
            f"CIDRAN pool ({cmin},{cmax}) too small for {n} events (CIDOFF={coff})")
    key = stream_key(root_key(simulation.run.random_seed), 'CIDRAN')

    def draw(round_):
        return cmin + 1 + jax.random.randint(jax.random.fold_in(key, round_),
                                             (m,), 0, pool, dtype=jnp.int32)

    def refused(values):
        # a repeat of an earlier draw, the earlier kept, or a skipped CID
        order = jnp.argsort(values, stable=True)
        ranked = values[order]
        repeat = jnp.zeros(m, dtype=bool).at[order[1:]].set(
            ranked[1:] == ranked[:-1])
        return repeat | jnp.isin(values, skip)

    values, round_ = draw(0), 0
    pending = refused(values)
    while bool(jnp.any(pending)):
        round_ += 1
        values = jnp.where(pending, draw(round_), values)
        pending = refused(values)
    return np.asarray(values[off:]).astype(cids.dtype)


def to_written_cid(simulation, cids, fmt):
    """Return the map from event CID to the CID written to file, SNANA's
    GENLC.CID_FINAL (snlc_sim.c:13007): CIDOFF added (GENLC.CID = CIDOFF +
    ilc, :12863) unless FORMAT_MASK CIDRAN(16) draws them. `cid` itself stays
    the working identity that GALID_UNIQUE and the spectra keys derive from,
    as GENLC.CID does."""
    if not fmt['cidran']:
        return lambda c: np.asarray(c) + simulation.output.cidoff
    drawn = dict(zip(np.asarray(cids).tolist(),
                     draw_cidran(simulation, cids).tolist()))
    return lambda c: np.array([drawn[int(x)] for x in np.asarray(c)])


def write_filter_transmission_files(simulator, base_dir, version_name):
    """FORMAT_MASK FILTERS(256): write per-band transmission curves
    (wr_SIMGEN_FILTERS, snlc_sim.c:14636)."""
    kcor = simulator.kcor
    out = Path(base_dir) / version_name / f'{version_name}.FILTERS'
    out.mkdir(parents=True, exist_ok=True)
    for name in kcor.get_filter_names():
        wave, trans = (np.asarray(a) for a in kcor.get_filter_transmission(name))
        char = filter_band(name)
        with open(out / f'{char}.dat', 'w') as f:
            for w, t in zip(wave, trans):
                f.write(f"{float(w):7.1f}  {float(t):.6f}\n")


def write_verify_sed_true(simulator, base_dir, version_name, spectra_by_sn,
                          results):
    """Write the VERIFY_SED_TRUE table (wr_VERIFY_SED_TRUE, snlc_sim.c:11538):
    one row per spectrum epoch with broadband MAG and synthetic MAG_VERIFY."""
    params = results['parameters']
    obs = results['observation_schedule']
    z_cmb = np.asarray(params['sim_redshift_cmb'])
    peak = np.asarray(params['peakmjd'])
    mjds = np.asarray(simlib_epochs(simulator, results, 'mjds'))
    fidx = np.asarray(obs['filter_indices'])
    mags = np.asarray(results['magnitudes'])
    obs_bands = [filter_band(n)
                 for n in simulator.photometry.filter_names]
    mean_wl = np.asarray(simulator.photometry.filter_mean_wavelengths)
    cid_to_idx = {int(c): i for i, c in enumerate(np.asarray(params['cid']))}
    mjds, fidx, mags = (np.asarray(a).reshape(-1) for a in (mjds, fidx, mags))

    lines = ["VARNAMES: CID z MJD TREST LAMREST BAND MAG MAG_VERIFY MAG_DIF\n"]
    for cid in sorted(spectra_by_sn):
        si = cid_to_idx[cid]
        span = row_slice(results, si)
        z = float(z_cmb[si]); z1 = 1.0 + z
        for spec in spectra_by_sn[cid]:
            dt = np.abs(mjds[span] - spec.mjd)
            oi = int(np.argmin(dt))
            if dt[oi] > 0.06:
                continue
            bi = int(fidx[span][oi])
            band = obs_bands[bi]
            genmag = float(mags[span][oi])
            if genmag < 1.0 or genmag > 60.0:
                continue
            synmag = (spec.synmag or {}).get(band, 99.0)
            trest = (spec.mjd - float(peak[si])) / z1
            lines.append(
                f"SN: {cid:8d}  {z:.3f}  {spec.mjd:.3f}  {trest:6.2f}  "
                f"{mean_wl[bi] / z1:6.0f}  {band}   "
                f"{genmag:.5f} {synmag:.5f} {synmag - genmag:8.5f}\n")
    # SNANA's own VERFIY spelling (snlc_sim.c:11569)
    path = Path(base_dir) / f"{version_name}_VERFIY_SED_TRUE.DAT"
    with open(path, 'w') as f:
        f.writelines(lines)


def validate_format_mask(simulation) -> None:
    """Enforce SNANA FORMAT_MASK rules (snlc_sim.c:7451-7468, :30986-30994)."""
    m = int(simulation.output.format_mask)
    f = parse_format_mask(m)
    if f['vbose']:
        raise SNANAError(
            "FORMAT_MASK bit 1 (VBOSE) is no longer supported")
    if m > 0 and not f['text'] and not f['fits']:
        raise SNANAError(
            f"Invalid FORMAT_MASK={m}: must specify TEXT (2) or FITS (32)")
    if f['text'] and f['fits']:
        raise SNANAError(
            f"Cannot mix TEXT and FITS format; FORMAT_MASK={m}")


def write_yaml_summary(simulator, version_name, counts, n_spec, seconds,
                       output_dir=None):
    """SNANA WRFLAG_YAML_FILE (wr_SIMGEN_YAML_SUMMARY, snlc_sim.c:14736):
    `{GENVERSION}.YAML` job summary consumed by batch/pipeline scripts.
    SNANA writes it with a bare GENVERSION name (snlc_sim.c:30917), i.e. to
    the process working directory, so batch scripts find it locally;
    ``output_dir`` is how a caller owning its own run directory says
    otherwise. ``counts`` is `event_counts`; ``seconds`` is the time since
    setup ended."""
    n_event, n_lc, n_write = counts
    path = Path(output_dir or Path.cwd()) / f'{version_name}.YAML'
    with open(path, 'w') as fp:
        fp.write(f"SURVEY:          {simulator.survey_name}\n")
        fp.write(f"IDSURVEY:        {simulator.survey_id}\n")
        fp.write(f"NGENEV_TOT:      {n_event}\n")
        fp.write(f"NGENLC_TOT:      {n_lc}\n")
        fp.write(f"NGENLC_WRITE:    {n_write}\n")
        fp.write(f"NGENSPEC_WRITE:  {n_spec}\n")
        fp.write(f"CPU_MINUTES:     {seconds / 60.0:.2f}\n")
        fp.write(f"ABORT_IF_ZERO:   {n_write}\n")


def log_output_summary(counts, base_dir, version_name, timings):
    """SNANA-style OUTPUT_SUMMARY + AUXILIARY FILES + DONE block.

    ``counts`` is `event_counts`. ``timings`` holds seconds per phase of the
    run: ``init`` and ``generate`` from the caller that ran those phases,
    ``write`` from `save_snana_outputs`. A phase the caller did not time has
    no row.

    The time after init is SNANA's CPUTIME_PROCESS_ALL (simEnd,
    snlc_sim.c:530), the number its YAML and README carry as CPU_MINUTES;
    WALLTIME_MIN adds the init to it. Every rate names its denominator,
    because SNANA reports one per event count (:532-539): GENLC is per
    generated light curve, ACCEP per written one.
    """
    n_event, n_lc, n_write = counts
    p = n_write / n_lc if n_lc else 0.0
    eff_err = float(np.sqrt(p * (1.0 - p) / n_lc)) if n_lc else 0.0

    def minutes(seconds):
        return f"{seconds / 60.0:.3f}"

    def per_lc(seconds):
        return f"{n_lc / seconds if seconds else 0.0:.1f} SNe/sec (GENLC)"

    def per_accept(seconds):
        return f"{n_write / seconds if seconds else 0.0:.1f} SNe/sec (ACCEP)"

    rows = [
        ("NGENEV_TOT", n_event),
        ("NGENLC_TOT", n_lc),
        ("NGENLC_WRITE", n_write, "pass trigger+cuts"),
        ("NREJECT", n_lc - n_write, "see selection breakdown above"),
        ("EFF(SEARCH+CUTS)", f"{p:.4f} +- {eff_err:.4f}"),
    ]
    if 'generate' in timings:
        rows.append(("GENTIME_MIN", minutes(timings['generate']),
                     f"{per_lc(timings['generate'])}, "
                     f"{per_accept(timings['generate'])}"))
    rows.append(("WRITETIME_MIN", minutes(timings['write']),
                 per_accept(timings['write'])))
    if 'generate' in timings:
        process = timings['generate'] + timings['write']
        rows.append(("CPUTIME_PROCESS_ALL", minutes(process),
                     f"generate + write, the YAML's CPU_MINUTES; "
                     f"{per_lc(process)}, {per_accept(process)}"))
    if {'init', 'generate'} <= timings.keys():
        total = timings['init'] + timings['generate'] + timings['write']
        rows.append(("WALLTIME_MIN", minutes(total),
                     f"init + generate + write; {per_lc(total)}, "
                     f"{per_accept(total)}"))
    log.summary(logger, "OUTPUT_SUMMARY", rows)
    out = Path(base_dir) / version_name
    aux = sorted(f.name for f in out.iterdir()
                 if f.name.startswith(version_name)) if out.is_dir() else []
    if aux:
        logger.info("\n AUXILIARY FILES:")
        for f in aux:
            log.line(logger, f"{out / f}")
    log.done(logger, "snlc_sim")


def save_snana_outputs(simulator, results,
                       version_name, selection_result, spectra_by_sn,
                       output_dir=None, timings=None, combined=False):
    """Write every SNANA output for a finished run: the writer's dataset
    plus the resource-dependent diagnostics (spectra, DCR, VERIFY_SED_TRUE,
    FILTERS, the YAML summary).

    ``spectra_by_sn`` contains the completed spectra returned by
    `run_simulation`, which also completes atmospheric effects.

    ``timings`` is an optional dict of seconds per phase the caller has
    already run (``init``, ``generate``); this adds ``write`` and the
    OUTPUT_SUMMARY reports each phase it holds.
    """
    simulation = simulator.config

    if timings is None:
        timings = {}
    start_time = time.time()
    fmt = parse_format_mask(simulation.output.format_mask)

    columnar_data = build_columnar_data(simulator, results, selection_result)
    write_timing = {'columnar_data': time.time() - start_time}

    n_passing = selection_result['n_passed']
    n_sne = n_passing + selection_result['n_failed']

    if n_passing == 0:
        # SNANA aborts only when NEPOCH took every event (simEnd, snlc_sim.c:566).
        codes = selection_result['generated']['rejection_codes']
        if codes.size and np.all(codes == 2):   # 2 = NEPOCH
            raise SNANAError(
                f"Every generated event fails NEPOCH>="
                f"{simulation.cuts.nepoch_min}. "
                f"GENRANGE_PEAKMJD and SIMLIB probably don't overlap.",
                function_name="save_snana_outputs")
        logger.warning("No supernovae passed selection cuts. Skipping SNANA output files.")
        return {
            'n_passed': 0,
            'n_failed': int(n_sne)
        }

    passing = np.asarray(selection_result['passing_mask'], dtype=bool)
    written_cids = set(np.asarray(results['parameters']['cid'])[passing])
    spectra_by_sn = {cid: spectra for cid, spectra in spectra_by_sn.items()
                     if cid in written_cids}

    # Atmosphere may need spectra even when no spectrum output was requested.
    if fmt['nospec'] or (simulation.instrument.atmos_optmask > 0
                         and not simulation.spectra.spectrograph_optmask & 128):
        spectra_by_sn = {}

    base_dir = output_dir or os.getcwd()
    phases, header_cols, phot_cols = write_snana_outputs(
        base_dir,
        version_name,
        columnar_data,
        simulation.run._input_file,
        simulation,
        simulator,
        simulator.filter_names,
        model=simulator.model,
        full_results=results,
        selection_result=selection_result,
        spectra_by_sn=spectra_by_sn,
        to_written_cid=to_written_cid(simulation, results['parameters']['cid'], fmt),
        makes_spectra=simulator.makes_spectra,
        seconds_since_init=time.time() - simulator.time_end_init,
        combined=combined,
    )
    write_timing.update(phases)

    # independent of FORMAT_MASK (snlc_sim.c:11295-11298)
    if (spectra_by_sn
            and simulation.effective_spectrograph_optmask & 128
            and simulation.spectra.verify_sed_true):
        write_verify_sed_true(simulator, base_dir, version_name,
                              spectra_by_sn, results)
    if fmt['filters']:
        write_filter_transmission_files(simulator, base_dir, version_name)

    timings['write'] = time.time() - start_time
    counts = event_counts(selection_result)
    if simulation.output.wrflag_yaml_file:
        n_spec = sum(len(v) for v in spectra_by_sn.values()) if spectra_by_sn else 0
        write_yaml_summary(simulator, version_name, counts, n_spec,
                           time.time() - simulator.time_end_init)
    log_output_summary(counts, base_dir, version_name, timings)

    return {
        'n_passed': int(n_passing),
        'n_failed': int(n_sne - n_passing),
        'timings': timings,
        'write_timing': write_timing,
        'header_columns': header_cols,
        'phot_columns': phot_cols,
    }
