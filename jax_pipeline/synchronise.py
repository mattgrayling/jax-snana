"""BBC's two-pass event synchronisation across systematic variations."""

import logging
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from jax_salt2mu.preparation import (format_data, interpolate_corrections,
                                     prepare_data, select_data)
from jax_salt2mu.run import finish_grid, prepare_cells
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import fitopt_muopt_name
from jax_snlc_core.utils import require_precision

logger = logging.getLogger(__name__)


def _events(cell):
    """Accepted identities and redshift bins, in the prepared table's order."""
    tv, accept = cell.tv, cell.accept
    fields = np.asarray(tv.field) if tv.field is not None else np.full(len(tv.name), "NULL")
    rows = [(str(cid), int(survey), str(field)) for cid, survey, field in
            zip(np.asarray(tv.name)[accept], np.asarray(tv.idsurvey)[accept], fields[accept])]
    if len(set(rows)) != len(rows):
        raise SNANAError("event synchronisation requires unique CID/IDSURVEY/FIELD combinations",
                         function_name="BBC")
    return dict(zip(rows, map(int, cell.inputs.izbin)))


def _write_selection(path, events):
    with Path(path).open("w") as stream:
        stream.write("VARNAMES: CID IDSURVEY FIELD IZBIN\n")
        for (cid, survey, field), izbin in events.items():
            stream.write(f"SN: {cid} {survey} {field} {izbin}\n")


def _selected(cell, path):
    config, tables, text = cell
    # SNANA appends these overrides to the original BBC input. Biascor and
    # CC-prior tables are unchanged; CUTWIN NONE applies inside BBC itself.
    return replace(config, cid_select_file=[str(path)], cutwin_disable=True), tables, text


def synchronised_row_counts(grid, data, fitopts):
    """Upper bounds for pass-2 rows, before bias validity, z-bin and chi2 cuts."""
    if (0, 0) not in grid:
        raise SNANAError("event synchronisation requires FITOPT000/MUOPT000", function_name="BBC")
    config, sources, _ = grid[0, 0]
    reference = select_data(config, sources, data[0, 0], None)
    fields = reference.field if reference.field is not None else np.full(reference.size, "NULL")
    identities = {(str(cid), int(survey), str(field)) for cid, survey, field in
                  zip(reference.name[reference.cutmask], reference.idsurvey[reference.cutmask],
                      fields[reference.cutmask])}
    counts = {}
    for cell, (config, sources, _) in grid.items():
        independent = cell[0] > 0 and "NOREJECT" in fitopts[cell[0] - 1]
        if not independent:
            config = replace(config, cid_select_file=[], cutwin_disable=True)
        tv = select_data(config, sources, data[cell], None)
        keep = tv.cutmask
        if not independent:
            fields = tv.field if tv.field is not None else np.full(tv.size, "NULL")
            keep = keep & np.array([(str(cid), int(survey), str(field)) in identities
                                    for cid, survey, field in zip(tv.name, tv.idsurvey, fields)])
        counts[cell] = int(keep.sum())
    return counts


def fit_prepared_common_sample(grid, maps, priors, fitopts, muopts, outdir, *, t_start=None,
                               fit_widths=None, cosmology_widths=None):
    """Prepare each cell's acceptance, then fit the common sample once.

    All supported cuts, including chi2max, are decided by prepare_cells.
    Converged-fit cuts (opt_chi2max >= 2) would require fitting this pass;
    BBC configuration currently refuses that key.

    FITOPT NOREJECT variations bypass event selection in both passes. FITOPT
    and MUOPT NOREJECT variations are excluded when finding the common sample,
    matching submit_prog_bbc's prep_JOB_INFO_bbc and get_fflist_accept_summary.
    """
    if t_start is None:
        t_start = time.time()
    if (0, 0) not in grid:
        raise SNANAError("event synchronisation requires FITOPT000/MUOPT000", function_name="BBC")
    for config, _, _ in grid.values():
        require_precision(config.double_precision)
    initial_dir = Path(outdir) / "initial"
    initial_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_data(grid, cosmology_widths=cosmology_widths)
    format_data(grid, data)
    biascor = interpolate_corrections(grid, data, maps)
    logger.info("BBC %s: preparing acceptance for the reference and systematic variations",
                Path(outdir).name)
    reference = _events(prepare_cells({(0, 0): grid[(0, 0)]}, data, maps, biascor, priors)[0, 0])
    initial = {(0, 0): reference}
    reference_file = initial_dir / "REFERENCE.LIST"
    _write_selection(reference_file, reference)

    def independent_fit(cell):
        return cell[0] > 0 and "NOREJECT" in fitopts[cell[0] - 1]

    others = {cell: (_selected(value, reference_file) if cell[0] and not independent_fit(cell)
                     else value) for cell, value in grid.items() if cell != (0, 0)}
    if others:
        for cell, value in others.items():
            prepared = prepare_cells({cell: value}, data, maps, biascor, priors)
            initial[cell] = _events(prepared[cell])
        del prepared
    common = set(reference)
    for cell, events in initial.items():
        cell_dir = initial_dir / fitopt_muopt_name(cell)
        cell_dir.mkdir()
        _write_selection(cell_dir / "ACCEPT.LIST", events)
        muopt_exempt = cell[1] > 0 and "NOREJECT" in muopts[cell[1] - 1]
        if not independent_fit(cell) and not muopt_exempt:
            common.intersection_update(events)
    if not common:
        raise SNANAError("no events survive all BBC variations used for event synchronisation",
                         function_name="BBC")
    logger.info("BBC %s: fitting with %d common events from %d reference events",
                Path(outdir).name, len(common), len(reference))
    selected_file = Path(outdir) / "BBC_ACCEPT_SUMMARY.LIST"
    _write_selection(selected_file, {event: zbin for event, zbin in reference.items() if event in common})
    selected_grid = {cell: value if independent_fit(cell) else _selected(value, selected_file)
                     for cell, value in grid.items()}
    return finish_grid(prepare_cells(selected_grid, data, maps, biascor, priors),
                       t_start=t_start, fit_widths=fit_widths)
