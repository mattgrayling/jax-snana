"""SNANA-vs-JAX parity test for the SIMGEN_DUMPALL writer.

For each bundled fixture, asserts:
1. VARNAMES header matches exactly (same columns, same order).
2. Per-SN integer columns agree exactly.
3. Per-SN float columns agree within a per-column tolerance.

`dumpall_listed_cols` asks for every name jax's `SIMGEN_DUMP: 0` listing
offers that a run with no HOSTLIB, lensing map or coordinate shift can
supply. SNANA aborts on a name it does not know
(MATCH_INDEX_SIMGEN_DUMP, snlc_sim.c:15833), so the bundled reference
existing at all is what says those names are SNANA's.

JAX runs in **replay mode**: the bundled SNANA DUMP+HEAD.FITS supply
truth params so both codes score the same SNe. No live SNANA call.
"""

import os
import re
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from parity._dumpall_helpers import compare_dumps, parse_dump
from replay_snana_dump import build_override_parameters

from jax_snlc_core.models import SALTModel
from jax_snlc_core.snana_files import load_snana_dump
from jax_snlc_sim import run as sim_run
from jax_snlc_sim.output import save_snana_outputs
from jax_snlc_sim.output.dump import _STATIC_DUMP_VARIABLES


def _run_jax_replay(input_file: Path, gv: str, snana_dump: Path, snana_head: Path,
                    work_root: Path):
    """Run JAX in replay mode and return ``(varnames, cols)`` of its DUMP."""
    dump_data = load_snana_dump(str(snana_dump), dedup_cid=True)
    if dump_data.size == 0:
        return None, {}

    with fits.open(snana_head) as hdul:
        head_table = hdul[1].data
        head_cols = {name: head_table[name] for name in head_table.names}
        peak_map = {
            int(s): float(p)
            for s, p in zip(head_table["SNID"], head_table["SIM_PEAKMJD"])
        }

    # Toy SIMLIB has 50 obs/LIBID; padding must not exceed that.
    config = dict(
        n_sne=dump_data.size,
    )
    sim = sim_run.prepare(str(input_file), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    override = build_override_parameters(dump_data.columns, simulation, resources,
                                         head_columns=head_cols)
    peaks = [
        peak_map.get(int(cid), float(override["peakmjd"][i]))
        for i, cid in enumerate(override["cid"])
    ]
    override["peakmjd"] = np.asarray(peaks, dtype=np.float32)

    # save_snana_outputs writes into cwd/<gv>/, so chdir into work_root first.
    cwd = os.getcwd()
    try:
        os.chdir(work_root)
        results, selection, spectra_cache = sim_run.run_simulation(sim, override_parameters=override)
        save_snana_outputs(sim, results, gv, selection, spectra_by_sn=spectra_cache)
    finally:
        os.chdir(cwd)

    jax_dump = work_root / gv / f"{gv}.DUMP"
    if not jax_dump.exists():
        return None, {}
    return parse_dump(jax_dump)


@pytest.mark.parametrize(
    ("fixture_name", "genversion"),
    [
        ("dumpall_minimal_cols", "JAX_TEST_DUMPALL_MIN"),
        ("dumpall_extended_cols", "JAX_TEST_DUMPALL_EXT"),
        ("dumpall_wide_cols", "JAX_TEST_DUMPALL_WIDE"),
        ("dumpall_wide_hostlib", "JAX_TEST_DUMPALL_HOSTLIB"),
    ],
)
def test_dumpall_parity(snana_fixture, fixture_name, genversion, tmp_path):
    """JAX in replay mode reproduces SNANA's SIMGEN_DUMPALL within tolerances."""
    fx = snana_fixture(fixture_name)
    snana_vn, snana_cols = parse_dump(fx.dump)
    assert snana_vn is not None, f"bundled DUMP for {fixture_name} unparseable"

    jax_vn, jax_cols = _run_jax_replay(
        fx.input_file, genversion, fx.dump, fx.head_fits, tmp_path
    )
    assert jax_vn is not None, "JAX failed to write a DUMP"

    compare_dumps(snana_vn, snana_cols, jax_vn, jax_cols)


# Columns whose value follows a draw replay does not reproduce: the redshift
# smear (GENSIGMA_REDSHIFT) and the SEARCHEFF detection draw behind the trigger
# and detection MJDs. The rest-frame epoch range is here for a different
# reason: it agrees with SNANA to a few hundredths of a day, not to the
# tolerance the other float columns hold to, and
# test_the_rest_frame_range_covers_the_whole_cadence is what guards it.
_REPLAY_STOCHASTIC = {"REDSHIFT_FINAL", "MJD_TRIGGER", "MJD_DETECT_FIRST",
                      "MJD_DETECT_LAST", "TRESTMIN", "TMIN", "TRESTMAX"}


def test_every_listed_dump_variable_is_a_snana_variable(snana_fixture, tmp_path):
    """Every column the SIMGEN_DUMP: 0 listing offers is one SNANA knows.

    `dumpall_listed_cols` asks for all of them that a run with no HOSTLIB,
    lensing map or coordinate shift can supply. SNANA aborts on a name it does
    not know and prints its own list instead (MATCH_INDEX_SIMGEN_DUMP,
    snlc_sim.c:15833), so the reference existing is what says the listing names
    SNANA's columns; the assertion below says the listing still offers them.
    """
    fx = snana_fixture("dumpall_listed_cols")
    snana_vn, snana_cols = parse_dump(fx.dump)
    jax_vn, jax_cols = _run_jax_replay(
        fx.input_file, "JAX_TEST_DUMPALL_LISTED", fx.dump, fx.head_fits, tmp_path
    )
    assert jax_vn == snana_vn

    # the SALT names are the model's, not the static listing's
    salt_names = {name for name, _ in SALTModel.SNANA_DUMP_SIM_COLUMNS}
    assert set(snana_vn) - salt_names <= set(_STATIC_DUMP_VARIABLES)
    assert {"AV", "RV"} <= set(snana_vn)

    deterministic = [v for v in snana_vn if v not in _REPLAY_STOCHASTIC]
    compare_dumps(deterministic, snana_cols, deterministic, jax_cols)


def test_replay_needs_av_when_the_sim_generates_dust(snana_fixture):
    """A dump without AV cannot replay a sim that draws host dust: the
    physics would run dust-free while the sim did not, so the replay aborts
    as it does without MU."""
    from jax_snlc_core.logging import SNANAError

    fx = snana_fixture("bayesn_w22")
    dump = load_snana_dump(str(fx.dump), dedup_cid=True)
    columns = {k: v for k, v in dump.columns.items() if k != 'AV'}
    config = dict(n_sne=dump.size)
    _prepared = sim_run.prepare(str(fx.input_file), **config)
    simulation = None if _prepared is None else _prepared.config
    resources = _prepared
    with fits.open(fx.head_fits) as hdul:
        head_cols = {n: hdul[1].data[n] for n in hdul[1].data.names}
    with pytest.raises(SNANAError, match="host dust"):
        build_override_parameters(columns, simulation, resources, head_columns=head_cols)


def test_replay_reads_any_name_the_dump_used(snana_fixture):
    """A dump written with ZCMB, S2x1 and S2c replays as one written with GENZ,
    SALT2x1 and SALT2c: SNANA registers each pair as one variable
    (snlc_sim.c:16050, :16083, :16486-16500), and the Foundation input of
    DES-SN5YR writes the first spellings."""
    from jax_snlc_core.logging import SNANAError

    fx = snana_fixture("dumpall_listed_cols")
    dump = load_snana_dump(str(fx.dump), dedup_cid=True)
    config = dict(n_sne=dump.size)
    _prepared = sim_run.prepare(str(fx.input_file), **config)
    simulation = None if _prepared is None else _prepared.config
    resources = _prepared
    with fits.open(fx.head_fits) as hdul:
        head_cols = {n: hdul[1].data[n] for n in hdul[1].data.names}

    def override(columns):
        return build_override_parameters(columns, simulation, resources,
                                         head_columns=head_cols)

    dropped = {'GENZ', 'SALT2x1', 'SIM_x1', 'SALT2c', 'SIM_c'}
    full = override(dump.columns)
    aliased = override({k: v for k, v in dump.columns.items()
                        if k not in dropped})
    assert aliased.keys() == full.keys()
    for key in full:
        np.testing.assert_array_equal(aliased[key], full[key], err_msg=key)

    # with none of them, the abort names what the dump could have used
    no_x1 = {k: v for k, v in dump.columns.items()
             if k not in {'S2x1', 'SALT2x1', 'SIM_x1'}}
    with pytest.raises(SNANAError, match=r"\['S2x1', 'SALT2x1', 'SIM_x1'\]"):
        override(no_x1)


def test_dump_column_resolution_by_source():
    """Every source a requested DUMP column can come from, resolved by name.

    The bundled fixtures ask only for pool columns, so the rest of the
    resolver goes untested by the parity run above: a model's own parameters,
    the bare alias SNANA also accepts for them, and the PEAKMAG_<filter>
    pattern. Each is pinned to the array it must return and the format the
    writer prints it with, which differs by source.
    """
    from jax_snlc_core.logging import SNANAError
    from jax_snlc_sim.output.dump import _resolve_dump_column

    pool = {'CID': np.array([1, 2]), 'PEAKMJD': np.array([55000.5, 55001.5])}
    model_params = {'SIM_SALT2x1': np.array([0.4, -0.2]),
                    'SIM_AV': np.array([0.1, 0.3])}

    band_tables = {'PEAKMAG_': {'LSST-r': np.array([22.0, 23.0])},
                   'SNRMAX_': {'LSST-r': np.array([15.0, 4.0])}}

    def resolve(name):
        return _resolve_dump_column(name, pool, model_params, band_tables)

    arr, _ = resolve('CID')
    np.testing.assert_array_equal(arr, [1, 2])

    # a pool column with a format override rather than the default
    assert resolve('PEAKMJD')[1] == '%.3f'

    # a model parameter under its full SNANA column name
    arr, fmt = resolve('SIM_SALT2x1')
    np.testing.assert_allclose(arr, [0.4, -0.2])
    assert fmt == '%.5e'

    # and under the bare name SNANA also accepts for it
    np.testing.assert_allclose(resolve('AV')[0], [0.1, 0.3])

    # <prefix><filter> matches the band character, not the kcor filter name
    np.testing.assert_allclose(resolve('PEAKMAG_r')[0], [22.0, 23.0])
    np.testing.assert_allclose(resolve('SNRMAX_r')[0], [15.0, 4.0])
    # a per-band prefix the run does not offer (no HOSTLIB) is unknown
    with pytest.raises(SNANAError, match="HOSTMAG_r"):
        resolve('HOSTMAG_r')

    with pytest.raises(SNANAError, match='Unknown DUMP column'):
        resolve('NOT_A_COLUMN')

    # a legacy name SNANA registers only to refuse (snlc_sim.c:16658-16663)
    with pytest.raises(SNANAError, match="legacy name; use 'TRESTMAX'"):
        resolve('TMAX')


def test_the_rest_frame_range_covers_the_whole_cadence(snana_fixture, tmp_path):
    """TRESTMIN/TRESTMAX span every epoch the SIMLIB kept, not the ones
    gen_cutwin goes on to accept.

    SNANA fills them in the cadence loop (snlc_sim.c:20766-20771) and
    gen_cutwin's own min/max (:24238-24240) runs over a subset of the same
    epochs, so it never narrows them. Taking the accepted subset instead put
    the range up to 13.9 days inside SNANA's on 61 of these 80 events. What
    is left is a few hundredths of a day: the cadence loop divides by
    1 + zCMB where the rest of the simulation uses 1 + zHELIO
    (snlc_sim.c:20499 against :24132), which on this fixture is a difference
    of about 1e-3 in redshift and so up to 0.05 days at Trest = 50.
    """
    fx = snana_fixture("dumpall_listed_cols")
    _, snana_cols = parse_dump(fx.dump)
    _, jax_cols = _run_jax_replay(
        fx.input_file, "JAX_TEST_DUMPALL_TREST", fx.dump, fx.head_fits, tmp_path)

    order = {int(c): i for i, c in enumerate(jax_cols["CID"])}
    rows = [order[int(c)] for c in snana_cols["CID"]]
    for name in ("TRESTMIN", "TRESTMAX"):
        snana = np.asarray(snana_cols[name], dtype=float)
        jax = np.asarray(jax_cols[name], dtype=float)[rows]
        assert np.abs(snana - jax).max() < 0.1, name


def test_peakmjd_exclude_window_and_season(snana_fixture):
    """GENRANGE_PEAKMJD_EXCLUDE takes a window out of the peak-MJD draw and
    out of the season the expected count is scaled by.

    SNANA redraws any peak inside the window (gen_peakmjd, snlc_sim.c:17030)
    and shortens the season by the window's span (SNcount_model, :17926).
    The fixture cuts 400 days from a 920-day range, so what is left splits
    260 days either side of the hole: the peaks jax draws must all fall
    outside it, the two sides must be populated evenly, and the README's
    NGEN_PER_SEASON must equal SNANA's.
    """
    fx = snana_fixture("peakmjd_exclude")
    _, snana_cols = parse_dump(fx.dump)
    snana_peaks = np.asarray(snana_cols["PEAKMJD"], dtype=float)
    lo, hi = 60800.0, 61200.0
    assert not np.any((snana_peaks > lo) & (snana_peaks < hi))
    readme = fx.root.joinpath("SIM.README").read_text()
    snana_season = int(re.search(r"NGEN_PER_SEASON:\s+(\d+)", readme).group(1))

    config = dict(n_sne=2000, random_seed=7)
    sim = sim_run.prepare(str(fx.input_file), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    assert simulation.rates.peakmjd_exclude == (lo, hi)
    results, _, _ = sim_run.run_simulation(sim)
    peaks = np.asarray(results["parameters"]["peakmjd_true"], dtype=float)
    assert not np.any((peaks > lo) & (peaks < hi))
    assert np.all((peaks >= 60540.0) & (peaks <= 61460.0))
    # 2000 draws split by a fair coin: 1000 +- 22
    below = int((peaks <= lo).sum())
    assert abs(below - 1000) < 4 * 22.4
    assert round(resources.season_count) == snana_season


def test_dump_coordinates_and_aliases():
    from astropy.coordinates import SkyCoord
    from jax_snlc_sim.output.dump import _build_dump_column_pool, _DUMP_EVENT_COLUMNS

    data = {name: np.zeros(2) for name in _DUMP_EVENT_COLUMNS}
    data.update(ra=np.array([10., 250.]), dec=np.array([-30., 60.]),
                sntype=np.array([1, 101]), mwebv=np.array([.1, .2]),
                mwebvmap=np.array([.15, .25]), snrmax_sorted=np.zeros((2, 5)),
                params={'redshift_final': np.array([.12, .23]),
                        'redshift_smear_err': np.array([.001, .002])})
    pool = _build_dump_column_pool(data)
    for alias, source in [('TYPE', 'SNTYPE'), ('MWEBV_TRUE', 'MWEBVMAP'),
                          ('ZCMB_SMEAR', 'REDSHIFT_FINAL'),
                          ('ZHELIO_ERR', 'REDSHIFT_FINAL_ERR')]:
        np.testing.assert_array_equal(pool[alias], pool[source])
    galactic = SkyCoord(data['ra'], data['dec'], unit='deg').galactic
    np.testing.assert_allclose(pool['GLON'], galactic.l.deg, atol=1e-3)
    np.testing.assert_allclose(pool['GLAT'], galactic.b.deg, atol=1e-3)
