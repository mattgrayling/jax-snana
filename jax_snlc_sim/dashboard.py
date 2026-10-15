"""SNANA DASHBOARD diagnostic mode (DASHBOARD_DRIVER, snlc_sim.c:31627).

Summarises the input files and the maps an initialised run resolved, then the
job exits without generating any SNe. Reports the same quantities as SNANA,
over the config and simulation jax already holds after init.
"""

import logging

import numpy as np

from jax_snlc_core.utils import snana_log as slog

logger = logging.getLogger(__name__)

WIDTH = 30


def _row(key, value):
    """One file row. NONE is SNANA's own unset sentinel, as are '' and None."""
    unset = value is None or str(value).strip() in ('', 'NONE')
    slog.kv(logger, key, '(none)' if unset else value, width=WIDTH)


def run_dashboard(simulation, resources):
    """Print the dashboard, in SNANA's order."""
    slog.banner(logger, "DASHBOARD_DRIVER")

    _row('GENMODEL', simulation.model.genmodel)

    _row('SIMLIB_FILE', simulation.survey.simlib_file)
    all_mjds = np.concatenate([f.mjds for f in resources.fields])
    slog.line(logger, f"{len(resources.fields)} observation sequences  "
                      f"{int(np.min(all_mjds))} < MJD < {int(np.max(all_mjds))}", indent=2)

    _row('CALIB_FILE', simulation.survey.kcor_file)

    _row('HOSTLIB_FILE', simulation.host.hostlib_file)
    if resources.hostlib is not None:
        ztrue = np.asarray(resources.hostlib.catalogue.redshifts)
        slog.line(logger, f"{resources.hostlib.selector.n_galaxies} galaxies, "
                          f"{np.min(ztrue):.2f} < z < {np.max(ztrue):.2f}",
                  indent=2)
    _row('HOSTLIB_WGTMAP_FILE', simulation.host.wgtmap_file)
    _row('HOSTLIB_ZPHOTEFF_FILE', simulation.host.zphoteff_file)
    _row('HOSTLIB_SPECBASIS_FILE', simulation.host.specbasis_file)
    _row('HOSTLIB_SPECDATA_FILE', simulation.host.specdata_file)
    _row('WRONGHOST_FILE', simulation.host.wronghost_file)

    _row('FLUXERRMODEL_FILE', simulation.instrument.fluxerr_file)
    # declined keys, printed unset as SNANA does (snlc_sim.c:31677, :31688)
    _row('NONLINEARITY_FILE', None)
    _row('ZVARIATION_FILE', simulation.model.zvariation_file)
    _row('WEAKLENS_PROBMAP_FILE', simulation.lensing.probmap_file)
    _row('STRONGLENS_FILE', None)

    searcheff = simulation.survey
    _row('SEARCHEFF_PIPELINE_LOGIC_FILE', searcheff.searcheff_pipeline_logic_file)
    # SNANA prints TIME_SINGLE_DETECT here (snlc_sim.c:31691-31692), which is
    # NEWMJD_DIF (snlc_sim.c:7523).
    slog.line(logger, f"Logic: {resources.searcheff_trigger_logic or '(none)'}  "
                      f"(epoch-sep > {float(simulation.survey.newmjd_dif):.3f} days)",
              indent=2)
    _row('SEARCHEFF_PIPELINE_EFF_FILE', searcheff.searcheff_pipeline_eff_file)
    slog.line(logger, f"NMAP_EFFDETECT={len(resources.searcheff_curves)}, "
                      f"NMAP_PHOTPROB={len(resources.searcheff_photprob_maps)}",
              indent=2)
    _row('SEARCHEFF_SPEC_FILE', searcheff.searcheff_spec_eff_file)
    slog.line(logger, f"NMAP_SPECEFF = {len(resources.searcheff_specid_maps)}",
              indent=2)
    _row('SEARCHEFF_zHOST_FILE', searcheff.searcheff_zhost_eff_file)
    slog.line(logger, f"NMAP_zHOST = {len(resources.searcheff_zhost_maps)}",
              indent=2)
