"""APPLY_CUTWIN_OPT=3 replay parity.

SNANA adds the CUTWIN-rejected events to the SIMGEN_DUMP alongside the accepted
ones (doReject_SIMGEN_DUMP, snlc_sim.c:16927), so a user tuning cuts can see the
population being discarded. Events rejected at any other stage stay out, and the
header keeps its accepted-set label despite the extra rows (:14897).

Scope is the dump alone: the flag is consumed at one place, wr_SIMGEN_DUMP
(:13564), and the caller rejects the event regardless — so HEAD/PHOT are
untouched.

The fixture uses epoch-coverage cuts, deterministic given the schedule, and an
all-pass pipeline efficiency, so dump membership turns on CUTWIN alone rather
than on the noise draw or the efficiency roll.
"""

import re

from astropy.io import fits
from replay_snana_dump import build_override_parameters

from jax_snlc_core.snana_files import load_snana_dump
from jax_snlc_sim import run as sim_run
from jax_snlc_sim.output import save_snana_outputs

# Events whose epochs sit within a fraction of a day of the Trest window edge
# land differently in each code (zHEL/zCMB, float32 PEAKMJD), which can move
# them across an epoch-coverage cut.
MAX_BOUNDARY_FRACTION = 0.03


def _dump_cids(path):
    """CIDs in a SIMGEN_DUMP, by column name rather than position."""
    col = None
    cids = []
    for line in path.read_text().splitlines():
        if line.startswith('VARNAMES:'):
            col = line.split().index('CID')
        elif line.startswith('SN:') and col is not None:
            cids.append(int(float(line.split()[col])))
    return set(cids)


def test_cutwin_opt3_replay(snana_fixture, tmp_path):
    fx = snana_fixture("cutwin_opt3_replay")

    dump = load_snana_dump(str(fx.dump), dedup_cid=True)
    with fits.open(fx.head_fits) as hdul:
        head_cols = {n: hdul[1].data[n] for n in hdul[1].data.names}
        snana_accepted = {int(s) for s in hdul[1].data['SNID']}
    snana_dumped = _dump_cids(fx.dump)

    # The fixture must exercise the feature: extra rows beyond the accepted set,
    # and not simply every generated event.
    assert len(snana_dumped) > len(snana_accepted), "no CUTWIN rejects dumped"
    extra = len(snana_dumped) - len(snana_accepted)
    assert extra > 20, f"only {extra} CUTWIN rejects; fixture lost its teeth"

    config = dict(n_sne=dump.size)
    sim = sim_run.prepare(str(fx.input_file), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    assert simulation.cuts.apply_cutwin_opt == 3

    override = build_override_parameters(dump.columns, simulation, resources, head_columns=head_cols)
    results, sel, spectra_cache = sim_run.run_simulation(sim, override_parameters=override)
    save_snana_outputs(sim, results, "OPT3", sel, spectra_by_sn=spectra_cache, output_dir=str(tmp_path))

    out = tmp_path / "OPT3"
    jax_dumped = _dump_cids(out / "OPT3.DUMP")
    with fits.open(next(iter(sorted(out.glob("OPT3*HEAD*"))))) as hdul:
        jax_accepted = {int(s) for s in hdul[1].data['SNID']}

    # The dump carries the CUTWIN rejects in jax too.
    assert len(jax_dumped) > len(jax_accepted), "jax dumped no CUTWIN rejects"

    disagree = len(jax_dumped ^ snana_dumped) / max(len(snana_dumped), 1)
    assert disagree < MAX_BOUNDARY_FRACTION, (
        f"dump membership differs for {len(jax_dumped ^ snana_dumped)} of "
        f"{len(snana_dumped)} events")

    # The feature must not leak past the dump: the accepted set is unchanged.
    leaked = len(jax_accepted ^ snana_accepted) / max(len(snana_accepted), 1)
    assert leaked < MAX_BOUNDARY_FRACTION, "OPT=3 changed the accepted set"

    # SNANA keeps the accepted-set label despite the extra rows.
    header = (out / "OPT3.DUMP").read_text()
    assert re.search(r'^#\s+SELECTION:.*(PASS|Pass)', header, re.M), (
        "dump header lost its selection label")
    assert 'every generated event' not in header, (
        "OPT=3 wrongly labelled as a write-everything dump")
