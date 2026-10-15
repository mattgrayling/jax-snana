"""Every column jax's sim writes matches SNANA's, across every output mode.

The sim writers produced plausible-looking HEAD/PHOT/SPEC files that had never
been diffed against snlc_sim.exe's. That hid ~15 defects: twenty columns whose
FITS type was inferred rather than declared, eight columns SNANA does not write,
seven it writes that jax omitted, two per-band families taken from the wrong
filter set, a renamed column, quantiles written unconditionally, three tables in
the wrong column order, and no TUNIT keys anywhere.

Most were invisible to a single-config check: they only appear under NON1ASED,
SIMSED, BayeSN, a HOSTLIB, spectra, DCR, or photo-z quantiles. So this is
parametrised over the modes rather than run on the SALT3 case alone.

The values inside the columns are the parity suites' business; this fixes the
files those values go into.
"""

import os
import re
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml
from astropy.io import fits

from jax_snlc_core.io.fits_path import open_text, resolve_fits_path
from jax_snlc_core.io.snana_dump_reader import load_snana_dump
from jax_snlc_sim import run as sim_run
from jax_snlc_sim.config_parser import parse_snana_input_file
from jax_snlc_sim.output.tables import mw_option_keys
from replay_snana_dump import build_override_parameters

from _replay_helpers import select_replay

_FIXTURES = Path("tests/fixtures/snana")

# Each spans an axis the others do not: model, host, spectra, atmosphere,
# photo-z, PSF unit, photo-z quantile grid, FORMAT_MASK COMPACT, weak lensing.
MODES = [
    "salt3_k21_replay",
    "fudge_noise_nea_replay",
    "non1ased_v19_replay",
    "simsed_91bg_replay",
    "bayesn_w22",
    "hostlib_e2e",
    "host_spectra_replay",
    "atmosphere_replay",
    "salt3_fit_photoz",
    "hostlib_mskopt_alt",
    "simlib_windows",
    "hostlib_nbr",
    "hostlib_nbr3",
    "simsed_91bg_sequential",
    "hostlib_zphot_q",
    "magmonitor_compact",
    "weaklens_probmap",
]

# Every mode bundles HEAD and PHOT; only the spectra fixture bundles SPEC.
TABLE_CASES = ([(m, k) for m in MODES for k in ("HEAD", "PHOT")]
               + [("host_spectra_replay", "SPEC")])


def _dump_varnames(path):
    with open_text(Path(path)) as stream:
        fixture_text = stream.read()
    for line in fixture_text.splitlines():
        if line.startswith("VARNAMES:"):
            return line.split()[1:]
    return None


def _tform(fmt):
    """FITS TFORM as (repeat, code); an absent repeat count means 1."""
    m = re.match(r"^(\d*)([A-Z])$", fmt)
    return (int(m.group(1) or 1), m.group(2)) if m else (None, fmt)


def _columns(path):
    with fits.open(resolve_fits_path(path)) as hdul:
        return [(c.name, _tform(c.format)) for c in hdul[1].columns], dict(hdul[1].header)


def _simulate(mode, n_sne):
    """A fresh jax run of a fixture's input; returns its version directory."""
    out = Path(tempfile.mkdtemp(prefix=f"contract_{mode}_"))
    config = dict(n_sne=n_sne, random_seed=3)
    sim = sim_run.prepare(str(_FIXTURES / mode / "input.input"), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results, selection, spectra_cache = sim_run.run_simulation(sim)
        sim_run.save_snana_outputs(sim, results, "J", selection, spectra_cache=spectra_cache, output_dir=str(out))
    return out / "J"


@pytest.fixture(scope="module")
def written():
    """Run each mode once and keep the output directory for the whole module."""
    cache = {}

    def run(mode):
        if mode not in cache:
            cache[mode] = _simulate(mode, 25)
        return cache[mode]

    return run


@pytest.mark.parametrize("mode,kind", TABLE_CASES)
def test_columns_match_snana(written, mode, kind):
    """Column names, order and FITS types are SNANA's, table by table."""
    reference = _FIXTURES / mode / f"{kind}.FITS"
    jax_file = next(Path(written(mode)).glob(f"J*{kind}*"), None)
    assert jax_file is not None, f"jax wrote no {kind} where SNANA does"

    snana_cols, _ = _columns(reference)
    jax_cols, _ = _columns(jax_file)
    assert jax_cols == snana_cols


def _marker_values(head_path, table_path, ext, lo, hi, columns=None):
    """The distinct values of ``columns`` (all, where None) on the marker rows
    of a pointer-indexed table, after checking that one closes each event's
    rows and no other exists."""
    head = fits.getdata(resolve_fits_path(head_path), ext[0])
    table = fits.getdata(resolve_fits_path(table_path), ext[1])
    first, last = np.asarray(head[lo]), np.asarray(head[hi])
    assert len(table) == int((last - first + 1).sum()) + len(head)
    assert np.all(first[1:] == last[:-1] + 2)
    is_marker = np.zeros(len(table), dtype=bool)
    is_marker[last] = True        # the row after each one-based last row
    assert not np.any(np.asarray(table[table.columns.names[0]])[~is_marker] == -777.0)
    rows = table[is_marker]
    return {name: {str(v).strip() for v in rows[name]}
            for name in columns or table.columns.names}


@pytest.mark.parametrize("mode", MODES)
def test_phot_marker_rows_match_snana(written, mode):
    """A marker row closes each light curve, as SNANA's writer adds one after
    every event (sntools_dataformat_fits.c:1509): -777 in MJD and the flux
    columns, '-' and 'XXXX' in BAND and FIELD, and PTROBS steps over it.
    SuperNNova splits the PHOT table at these rows, so without them it reads
    a whole file as one light curve. SNANA's other columns on the row hold
    what its per-epoch arrays held, nulls or zeros depending on the epochs of
    the events before, so only the five it sets are compared."""
    jax_dir = Path(written(mode))
    named = [c for c in ("MJD", "BAND", "FIELD", "FLUXCAL", "FLUXCALERR")
             if c in [name for name, _ in _columns(_FIXTURES / mode / "PHOT.FITS")[0]]]
    assert (_marker_values(next(jax_dir.glob("J*HEAD*")), next(jax_dir.glob("J*PHOT*")),
                           (1, 1), "PTROBS_MIN", "PTROBS_MAX", named)
            == _marker_values(_FIXTURES / mode / "HEAD.FITS", _FIXTURES / mode / "PHOT.FITS",
                              (1, 1), "PTROBS_MIN", "PTROBS_MAX", named))


def test_spec_marker_rows_match_snana(written):
    """A marker row of -777 in every flux column closes each spectrum, and
    PTRSPEC steps over it (sntools_dataformat_fits.c:2895)."""
    spec = next(Path(written("host_spectra_replay")).glob("J*SPEC*"))
    reference = _FIXTURES / "host_spectra_replay" / "SPEC.FITS"
    assert (_marker_values(spec, spec, (1, 2), "PTRSPEC_MIN", "PTRSPEC_MAX")
            == _marker_values(reference, reference, (1, 2), "PTRSPEC_MIN", "PTRSPEC_MAX"))


# Named by the run rather than the configuration.
_HEADER_RUN_KEYS = {'SNANA_PATH', 'SNANA_VERSION', 'VERSION', 'PHOTFILE',
                    'SPECFILE'}
_HEADER_PATH_KEYS = {'SIMLIB_FILE', 'CALIB_FILE', 'HOSTLIB_FILE'}


@pytest.mark.parametrize("mode", MODES)
def test_primary_header_matches_snana(written, mode):
    """The primary header carries SNANA's keys, in SNANA's order, with
    SNANA's values. The fitter reads a dozen of them and aborts on a missing
    one (sntools_dataformat_fits.c:4597-4667)."""
    reference = _FIXTURES / mode / "HEAD.FITS"

    def cards(path):
        with fits.open(resolve_fits_path(path)) as hdul:
            h = hdul[0].header
            return [(k, h[k]) for k in h if k not in (
                'SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'EXTEND', 'COMMENT')]

    snana = cards(reference)
    jax = cards(next(Path(written(mode)).glob("J*HEAD*")))
    assert [k for k, _ in jax] == [k for k, _ in snana]
    for (key, s), (_, j) in zip(snana, jax):
        if key in _HEADER_RUN_KEYS:
            continue
        if key in _HEADER_PATH_KEYS:
            # cfitsio cuts a HIERARCH string at the card, so SNANA's
            # HOSTLIB_FILE keeps 55 characters at most
            if len(s) < 55:
                assert Path(j).name == Path(s).name, key
        elif isinstance(s, float):
            assert j == pytest.approx(s, rel=1e-6), key
        elif key == 'DETINFO_FLAG':
            # SNANA hands cfitsio a bool as a 4-byte int
            # (sntools_dataformat_fits.c:1155), so the flag arrives in the
            # low byte with the next flag's byte above it
            assert j == s & 0xFF, key
        else:
            assert j == s, key


# One fixture per Milky Way colour law the bundle exercises, with the
# OPT_MWCOLORLAW code SNANA stamped into each HEAD.
_MWDUST_MODES = ["mwdust_opt1", "mwdust_parity", "mwdust_somm25",
                 "hostlib_mwebv"]


@pytest.mark.parametrize("mode", _MWDUST_MODES)
def test_mw_option_keys_match_snana(mode):
    """SIMOPT_MWCOLORLAW, SIM_MWRV and SIMOPT_MWEBV are the run's own.

    One fixture per colour law the bundle exercises; MODES above holds only
    FITZ99 runs, so nothing there separates one law's code from another's.
    """
    config = parse_snana_input_file(_FIXTURES / mode / "input.input")
    with fits.open(resolve_fits_path(_FIXTURES / mode / "HEAD.FITS")) as hdul:
        header = hdul[0].header
    keys = mw_option_keys(config)
    assert keys == {k: header[k] for k in keys}


@pytest.mark.parametrize("mode", MODES)
def test_dump_columns_match_snana(written, mode):
    """The SIMGEN dump names and orders its columns as SNANA does, except for
    a model's own parameters.

    SNANA writes those bare in the DUMP and decorated in the data files —
    `stretch` against `SIMSED_PARAM(stretch)` — where jax writes the decorated
    name in both, so one quantity has one name across its outputs. Order
    and every other column still match exactly.
    """
    reference = _FIXTURES / mode / "DUMP"
    jax_dump = Path(written(mode)) / "J.DUMP"
    assert jax_dump.exists(), "jax wrote no DUMP where SNANA does"

    jax_names = _dump_varnames(jax_dump)
    bare = [n.split("(", 1)[1].rstrip(")") if n.endswith(")") else n
            for n in jax_names]
    assert bare == _dump_varnames(reference)
    # the decoration is the declaring keyword, as the data files write it;
    # an undeclared parameter is baggage (snlc_sim.c:23775)
    assert all(n.startswith(("SIMSED_PARAM(", "SIMSED_GRIDONLY(",
                             "SIMSED_param(", "SHAPEPAR(", "COLORPAR(",
                             "COLORLAW("))
               for n in jax_names if n.endswith(")"))


@pytest.mark.parametrize("mode,kind", TABLE_CASES)
def test_tunit_written(written, mode, kind):
    """jax's TUNIT per column is SNANA's; readers that index by it need them."""
    reference = _FIXTURES / mode / f"{kind}.FITS"
    snana_cols, snana_header = _columns(reference)
    jax_file = next(Path(written(mode)).glob(f"J*{kind}*"), None)
    jax_cols, jax_header = _columns(jax_file)

    def units(header, n):
        return [header.get(f"TUNIT{i}") for i in range(1, n + 1)]

    assert units(jax_header, len(jax_cols)) == units(snana_header, len(snana_cols))


# ---------------------------------------------------------------------------
# TEXT (.DAT) contract
# ---------------------------------------------------------------------------

# Modes bundling a SNANA text_ref whose key set jax reproduces exactly. The
# HOSTLIB modes are excluded: jax cannot yield a hostless SN without a
# NBR_LIST, so it writes a galaxy block where SNANA writes none (see the
# host-association gap notes), and replay cannot reproduce which neighbour
# SNANA's DDLR sort put second, so hostlib_nbr_zphot_q is out too. Their
# spectra block is still checked below.
# keys SNANA writes for every model that belong to one: SIM_THETA is BayeSN's
_SNANA_ALWAYS = {"SIM_THETA"}
TEXT_MODES = ["salt3_k21_replay", "bayesn_w22", "non1ased_v19_replay",
              "hostlib_e2e", "host_spectra_replay", "hostlib_zphot_q",
              # the only fixture with MW extinction on, so the only one where
              # MWXT_MAG is non-zero (its values are checked separately below)
              "mwdust_opt1"]


def _dat_keys(path):
    """(header keys, OBS varlist, per-spectrum SPECTRUM_* keys)."""
    header, varlist, spec = set(), None, set()
    for raw in open_text(path):
        line = raw.split('#', 1)[0].strip()
        if not line or line.startswith(('OBS:', 'SPEC:', 'END')):
            continue
        if line.startswith('VARLIST:'):
            varlist = line.split()[1:]
        elif ':' in line:
            key = line.split(':', 1)[0].strip()
            (spec if key.startswith('SPECTRUM_') else header).add(key)
    return header, varlist, spec


@pytest.fixture(scope="module")
def text_written():
    """Run each mode once with FORMAT_MASK=2 and keep the .DAT output."""
    cache = {}

    def run(mode):
        """``mode`` names a fixture directory, or is a directory holding an
        input.input of its own."""
        if mode not in cache:
            out = Path(tempfile.mkdtemp(prefix="text_"))
            config = dict(n_sne=25,
                                         random_seed=3)
            input_dir = Path(mode) if Path(mode).is_absolute() else _FIXTURES / mode
            sim = sim_run.prepare(str(input_dir / "input.input"), **config)
            simulation = None if sim is None else sim.config
            resources = sim
            simulation.output.format_mask = 2
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results, selection, spectra_cache = sim_run.run_simulation(sim)
                sim_run.save_snana_outputs(sim, results, "T", selection, spectra_cache=spectra_cache, output_dir=str(out))
            cache[mode] = out / "T"
        return cache[mode]

    return run


def _union(paths, index):
    """Union of one _dat_keys element over several files."""
    out = set()
    for p in paths:
        part = _dat_keys(p)[index]
        out |= set(part or ())
    return out


@pytest.mark.parametrize("mode", TEXT_MODES)
def test_text_header_keys_match_snana(text_written, mode):
    """Every KEY: jax writes is one SNANA writes, and none is missing.

    Unioned over events because SNANA gates several keys per-SN (TRIGGER,
    MJD_DETECT_*), so a single file understates the key set.
    """
    ref_dir = _FIXTURES / mode / "text_ref"
    assert ref_dir.is_dir(), f"{mode} bundles no text_ref"

    snana = _union(sorted(ref_dir.glob("SN*.DAT")), 0)
    jax = _union(sorted(Path(text_written(mode)).glob("T_*.DAT")), 0)
    # SNANA writes SIM_THETA for every model, never having initialised the
    # BayeSN parameter (snlc_sim.c:25499); jax writes it where the model has it
    assert jax - _SNANA_ALWAYS == snana - _SNANA_ALWAYS
    assert ("SIM_THETA" in jax) == ("bayesn" in mode)


@pytest.mark.parametrize("mode", TEXT_MODES)
def test_text_obs_varlist_matches_snana(text_written, mode):
    """The OBS VARLIST names and orders its columns as SNANA does."""
    ref_dir = _FIXTURES / mode / "text_ref"
    assert ref_dir.is_dir(), f"{mode} bundles no text_ref"

    snana = _dat_keys(sorted(ref_dir.glob("SN*.DAT"))[0])[1]
    jax = _dat_keys(sorted(Path(text_written(mode)).glob("T_*.DAT"))[0])[1]
    assert jax == snana


def test_text_spectra_block_matches_snana(text_written):
    """The per-spectrum SPECTRUM_* keys match SNANA's.

    Only host_spectra_replay bundles a spectra-bearing reference, and this is
    the sole check on that block.
    """
    mode = "host_spectra_replay"
    ref_dir = _FIXTURES / mode / "text_ref"
    assert ref_dir.is_dir(), f"{mode} bundles no text_ref"

    snana = _union(sorted(ref_dir.glob("SN*.DAT")), 2)
    assert snana, "reference bundles no spectra — the check would be vacuous"
    jax = _union(sorted(Path(text_written(mode)).glob("T_*.DAT")), 2)
    assert jax == snana


def _quantile_block(path, prefix='HOSTGAL'):
    """One host slot's photo-z quantile lines of a .DAT: NBIN, the z values
    and the percentile labels, or None when the slot is not written."""
    keys = {f'{prefix}z_NBIN_QUANTILE_ZPHOT': 'NBIN',
            f'{prefix}z_QUANTILE_ZPHOT': 'Z',
            f'{prefix}z_QUANTILE_PERCENT': 'PERCENT'}
    out = {}
    for line in open_text(path):
        key = line.split(':', 1)[0]
        if key in keys:
            out[keys[key]] = line.split(':', 1)[1].split()
    if not out:
        return None
    out['NBIN'] = int(out['NBIN'][0])
    return out


@pytest.mark.parametrize("mode", ["hostlib_e2e", "hostlib_zphot_q"])
def test_text_zphot_q_block_matches_snana(text_written, mode):
    """The quantile block names the percentiles and holds one z per bin.

    SNANA labels the quantiles uniformly from their count
    (compute_implicit_percentiles, sntools.c:9513); jax reads the label from
    the HOSTLIB column name. Both catalogues declare the eleven decades, so
    the two blocks agree exactly.
    """
    ref_dir = _FIXTURES / mode / "text_ref"

    snana = [_quantile_block(p) for p in sorted(ref_dir.glob("SN*.DAT"))]
    snana = [b for b in snana if b is not None]
    assert len(snana) > 10
    assert {b['NBIN'] for b in snana} == {11}
    percent = snana[0]['PERCENT']

    written = sorted(Path(text_written(mode)).glob("T_*.DAT"))
    jax = [_quantile_block(p) for p in written]
    jax = [b for b in jax if b is not None]
    assert len(jax) > 10, "no event wrote a host block"
    for block in jax:
        assert block['NBIN'] == 11
        assert block['PERCENT'] == percent
        assert len(block['Z']) == 11


def test_indexed_quantile_columns_write_the_same_block(text_written, tmp_path):
    """A HOSTLIB naming its quantiles QZPHOT00..QZPHOT10 writes the block the
    same catalogue writes under ZPHOT_Q000..Q100: the implied uniform
    percentiles are the decades (manual 5030-5047). jax-only: the pinned
    snlc_sim counts the QZPHOTnn columns but reads none of them
    (IVAR_Q0ZPHOT is found by the ZPHOT_Q prefix alone, sntools_host.c:2694),
    writing every quantile as -9."""
    mode = "hostlib_zphot_q"
    src = _FIXTURES / mode / "input.input"
    hostlib_line = next(ln for ln in src.read_text().splitlines()
                        if ln.startswith("HOSTLIB_FILE:"))
    hostlib = Path(os.path.expandvars(hostlib_line.split()[1]))
    renamed = tmp_path / "QZ.HOSTLIB"
    with open_text(hostlib) as stream:
        fixture_text = stream.read()
    renamed.write_text(re.sub(r"ZPHOT_Q(\d{3})", lambda m: f"QZPHOT{int(m.group(1)) // 10:02d}",
                              fixture_text))
    (tmp_path / "input.input").write_text(
        src.read_text().replace(hostlib_line, f"HOSTLIB_FILE: {renamed}"))

    def blocks(paths):
        return [b for b in (_quantile_block(p) for p in sorted(paths)) if b is not None]

    decade = blocks(Path(text_written(mode)).glob("T_*.DAT"))
    indexed = blocks(Path(text_written(str(tmp_path))).glob("T_*.DAT"))
    assert len(decade) > 10 and len(indexed) == len(decade)
    for a, b in zip(decade, indexed):
        assert a == b


def test_text_third_host_block_is_the_third_neighbour(text_written):
    """A third matched host fills the HOSTGAL3 block, as SNANA writes for
    the third galaxy by DDLR (MXHOSTGAL = 3, sndata.h:42): its quantile
    block is its own, and the three galaxies are one NBR_LIST triplet in
    DDLR order. hostlib_nbr3's catalogue groups galaxies in threes, so
    most of SNANA's reference events carry all three blocks."""
    mode = "hostlib_nbr3"
    ref_dir = _FIXTURES / mode / "text_ref"

    def three_host_events(paths):
        out = []
        for p in paths:
            keys = {}
            for line in open_text(p):
                key = line.split(':', 1)[0]
                if key in ('HOSTGAL_NMATCH2', 'HOSTGAL_OBJID', 'HOSTGAL2_OBJID',
                           'HOSTGAL3_OBJID', 'HOSTGAL_DDLR', 'HOSTGAL2_DDLR',
                           'HOSTGAL3_DDLR'):
                    keys[key] = float(line.split(':', 1)[1].split()[0])
            if keys.get('HOSTGAL_NMATCH2') == 3:
                out.append((keys, _quantile_block(p, 'HOSTGAL2'),
                            _quantile_block(p, 'HOSTGAL3')))
        return out

    def check(events):
        assert len(events) > 10, "too few three-host events"
        for keys, second, third in events:
            objids = sorted(int(keys[f'{p}_OBJID'])
                            for p in ('HOSTGAL', 'HOSTGAL2', 'HOSTGAL3'))
            # a triplet is rows 2k, 2k+1 (GALID 200000 + row) and 200600 + k
            k = (objids[0] - 200000) // 2
            assert objids == [200000 + 2 * k, 200001 + 2 * k, 200600 + k]
            assert keys['HOSTGAL_DDLR'] <= keys['HOSTGAL2_DDLR'] <= keys['HOSTGAL3_DDLR']
            assert third['NBIN'] == second['NBIN'] == 11
            assert third['PERCENT'] == second['PERCENT']
            assert third['Z'] != second['Z']

    check(three_host_events(sorted(ref_dir.glob("SN*.DAT"))))
    check(three_host_events(sorted(Path(text_written(mode)).glob("T_*.DAT"))))


def test_text_hostgal2_quantiles_are_the_second_galaxy(text_written):
    """Each galaxy's quantile block carries its own numbers.

    SNANA prints SNHOSTGAL_DDLR_SORT[igal].QZPHOT under the igal prefix
    (wr_dataformat_text_HOSTGALz), reading each galaxy's own HOSTLIB row.
    hostlib_nbr_zphot_q's catalogue gives every neighbour a photo-z of its
    own, so in SNANA's reference the two blocks differ on every event with
    two matched hosts; jax once printed host 1's numbers under both labels.
    """
    mode = "hostlib_nbr_zphot_q"
    ref_dir = _FIXTURES / mode / "text_ref"

    def two_host_blocks(paths):
        out = [(_quantile_block(p), _quantile_block(p, 'HOSTGAL2'))
               for p in paths]
        return [(a, b) for a, b in out if b is not None]

    snana = two_host_blocks(sorted(ref_dir.glob("SN*.DAT")))
    assert len(snana) > 10, "reference has too few two-host events"
    for first, second in snana:
        assert second['NBIN'] == first['NBIN'] == 11
        assert second['PERCENT'] == first['PERCENT']
        assert first['Z'] != second['Z']
    percent = snana[0][0]['PERCENT']

    jax = two_host_blocks(sorted(Path(text_written(mode)).glob("T_*.DAT")))
    assert len(jax) > 10, "jax wrote too few two-host events"
    for first, second in jax:
        assert second['NBIN'] == first['NBIN'] == 11
        assert second['PERCENT'] == first['PERCENT'] == percent
        assert first['Z'] != second['Z']


# Keys SNANA gates on a detection that is drawn against the SEARCHEFF
# efficiency curve, so they differ per realisation even under replay.
_DETECTION_GATED = {'TRIGGER', 'MJD_TRIGGER',
                    'MJD_DETECT_FIRST', 'MJD_DETECT_LAST'}


@pytest.mark.parametrize("mode", TEXT_MODES)
def test_text_keys_match_snana_per_event(mode):
    """Every SN's key set matches SNANA's for the same SN.

    The union check above compares vocabulary, so it cannot see a key emitted
    for the wrong events. Replaying SNANA's own parameters pairs the events,
    which is what makes per-SN gating checkable.
    """
    fx = _FIXTURES / mode
    ref_dir = fx / "text_ref"
    assert ref_dir.is_dir(), f"{mode} bundles no text_ref"

    ref = sorted(ref_dir.glob("SN*.DAT"))
    snids = [int(p.stem[2:]) for p in ref]
    dump = load_snana_dump(str(fx / "DUMP"), dedup_cid=True)
    with fits.open(resolve_fits_path(fx / "HEAD.FITS")) as hdul:
        head = {n: hdul[1].data[n] for n in hdul[1].data.names}
    dump_cols, head = select_replay(dump.columns, head, snids)

    out = Path(tempfile.mkdtemp(prefix=f"perevent_{mode}_"))
    config = dict(n_sne=len(dump_cols['CID']))
    sim = sim_run.prepare(str(fx / "input.input"), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    simulation.output.format_mask = 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        override = build_override_parameters(dump_cols, simulation, resources,
                                             head_columns=head)
        results, selection, spectra_cache = sim_run.run_simulation(sim, override_parameters=override)
        sim_run.save_snana_outputs(sim, results, "P", selection, spectra_cache=spectra_cache, output_dir=str(out))

    compared = 0
    for snid in snids:
        jax_dat = out / "P" / f"P_SN{snid:06d}.DAT"
        if not jax_dat.exists():
            continue
        compared += 1
        snana_keys = _dat_keys(ref_dir / f"SN{snid:06d}.DAT")[0] - _DETECTION_GATED - _SNANA_ALWAYS
        jax_keys = _dat_keys(jax_dat)[0] - _DETECTION_GATED - _SNANA_ALWAYS
        assert jax_keys == snana_keys, f"SN{snid:06d}"
    assert compared > 10, f"only {compared} events replayed"


def test_text_mwxt_mag_values_match_snana():
    """MW extinction per band, checked against SNANA's own numbers.

    Every other fixture sets OPT_MWEBV: 0, so MWXT_MAG is all zeros there and
    a wrong RV factor, extinction curve or band order would still match. This
    is the only place those are exercised.
    """
    mode = "mwdust_opt1"
    fx = _FIXTURES / mode
    ref_dir = fx / "text_ref"
    assert ref_dir.is_dir(), f"{mode} bundles no text_ref"

    ref = sorted(ref_dir.glob("SN*.DAT"))
    snids = [int(p.stem[2:]) for p in ref]
    dump = load_snana_dump(str(fx / "DUMP"), dedup_cid=True)
    with fits.open(resolve_fits_path(fx / "HEAD.FITS")) as hdul:
        head = {n: hdul[1].data[n] for n in hdul[1].data.names}
    dump_cols, head = select_replay(dump.columns, head, snids)

    out = Path(tempfile.mkdtemp(prefix="mwxt_"))
    config = dict(n_sne=len(dump_cols['CID']))
    sim = sim_run.prepare(str(fx / "input.input"), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    simulation.output.format_mask = 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        override = build_override_parameters(dump_cols, simulation, resources,
                                             head_columns=head)
        results, selection, spectra_cache = sim_run.run_simulation(sim, override_parameters=override)
        sim_run.save_snana_outputs(sim, results, "X", selection, spectra_cache=spectra_cache, output_dir=str(out))

    def mwxt(path):
        for line in open_text(path):
            if line.startswith('MWXT_MAG:'):
                return [float(v) for v in
                        line.split(':', 1)[1].split('#')[0].split()]
        return None

    checked = 0
    for snid in snids:
        jax_dat = out / "X" / f"X_SN{snid:06d}.DAT"
        if not jax_dat.exists():
            continue
        snana, jax = mwxt(ref_dir / f"SN{snid:06d}.DAT"), mwxt(jax_dat)
        assert jax is not None and snana is not None
        assert any(v > 0 for v in snana), "reference is all zeros — check is vacuous"
        # SNANA writes three decimals, so a half-digit is the achievable floor.
        assert jax == pytest.approx(snana, abs=1e-3), f"SN{snid:06d}"
        checked += 1
    assert checked > 10, f"only {checked} events compared"


# Columns whose values are fixed by config or by SNANA's per-field "no data"
# sentinels, so replay must reproduce them exactly. jax previously used a
# single -9 for all of them, wrote the SN's own coordinates where there is no
# host, and hardcoded SEARCH_TYPE, SIM_EXPOSURE and SIM_MJD_EXPLODE.
_EXACT_VALUE_COLUMNS = [
    'HOSTGAL_RA', 'HOSTGAL_DEC', 'HOSTGAL_CONFUSION',
    'HOSTGAL_SPECZ', 'HOSTGAL_SPECZ_ERR', 'HOSTGAL_PHOTOZ', 'HOSTGAL_PHOTOZ_ERR',
    'HOSTGAL_SNSEP', 'HOSTGAL_DDLR',
    'HOSTGAL_OBJID', 'HOSTGAL_FLAG', 'HOSTGAL_OBJID2', 'HOSTGAL_OBJID_UNIQUE',
    'HOSTGAL_LOGMASS', 'HOSTGAL_LOGSFR', 'HOSTGAL_LOGsSFR', 'HOSTGAL_COLOR',
    'HOSTGAL_ELLIPTICITY', 'HOSTGAL_SQRADIUS',
    'HOSTGAL2_OBJID', 'HOSTGAL2_FLAG', 'HOSTGAL2_RA', 'HOSTGAL2_DEC',
    'HOSTGAL2_LOGMASS', 'HOSTGAL2_LOGSFR', 'HOSTGAL2_COLOR',
    'HOSTGAL2_ELLIPTICITY', 'HOSTGAL2_SQRADIUS',
    'SIM_HOSTLIB_GALID', 'SEARCH_TYPE', 'SIM_MJD_EXPLODE',
    'SIM_REDSHIFT_HOST_MATCH', 'SIM_NGEN_LIBID', 'SIM_NOBS_UNDEFINED',
    'SIM_LENSDMU', 'LENSDMU_ERR',
]

# LENSDMU is the truth plus a Gaussian of width LENSDMU_ERR
# (gen_lensDMU_smear, snlc_sim.c:12938), an independent draw in each code.
# Its width is checked through LENSDMU_ERR above.
_INDEPENDENT_DRAW = {'LENSDMU'}


# Computed while a host is being selected rather than read from its HOSTLIB
# row, so replay -- which overrides selection -- cannot reproduce them. The
# surface brightness and the reported galaxy coordinates depend on where the
# SN landed relative to its host, which selection decides.
_SELECTION_DERIVED = {'HOSTGAL_DDLR',
                      'HOSTGAL_PHOTOZ', 'HOSTGAL_PHOTOZ_ERR',
                      'HOSTGAL_RA', 'HOSTGAL_DEC',
                      # carries the observed-redshift smear, not the catalogue value
                      'HOSTGAL_SPECZ', 'HOSTGAL_SPECZ_ERR',
                      # SNANA's unmatched-with-HOSTLIB null (-9, -99) differs
                      # from its no-HOSTLIB null (-9999); rule not established
                      'HOSTGAL_LOGMASS', 'HOSTGAL_LOGsSFR'}
_SELECTION_DERIVED_PREFIXES = ()

# The SN's offset from its host is reconstructed from HOSTGAL_RA/DEC, which the
# header stores as float32, so these two carry that rounding rather than an
# exact value. They are checked because they were badly wrong before the
# reconstruction: SB_FLUXCAL ran a median 1.94x SNANA with a tail to 273x.
# (rtol, atol). The SN-host offset is reconstructed from the header: its
# direction from HOSTGAL_RA/DEC and its magnitude from SIM_SNHOST_SEP. Taking
# the magnitude from the separation column rather than differencing two ~30
# degree coordinates is what makes these tight -- coordinate differencing
# leaves ~0.01 arcsec, since the float32 step scales with the coordinate and
# not with the separation. Only the direction still carries that error.
_POSITION_DERIVED_TOL = {'HOSTGAL_SNSEP': (1e-5, 1e-5),
                         'HOSTGAL_SB_FLUXCAL': (0.02, 0.05)}


def _replay_written_events(fx, tag):
    """Replay every event SNANA wrote for a fixture and write jax's outputs.
    Returns the output directory, SNANA's HEAD columns and the resources."""
    with fits.open(resolve_fits_path(fx / "HEAD.FITS")) as hdul:
        snana = {n: np.asarray(hdul[1].data[n]) for n in hdul[1].data.names}
    snids = [int(s) for s in snana['SNID']]
    dump = load_snana_dump(str(fx / "DUMP"), dedup_cid=True)
    dump_cols, head = select_replay(dump.columns, dict(snana), snids)

    out = Path(tempfile.mkdtemp(prefix=tag))
    config = dict(n_sne=len(dump_cols['CID']))
    sim = sim_run.prepare(str(fx / "input.input"), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        override = build_override_parameters(dump_cols, simulation, resources,
                                             head_columns=head)
        results, selection, spectra_cache = sim_run.run_simulation(sim, override_parameters=override)
        sim_run.save_snana_outputs(sim, results, "V", selection, spectra_cache=spectra_cache, output_dir=str(out))
    return out / "V", snana, resources


def _output_summary(path):
    """The README's OUTPUT_SUMMARY lines without their comments, SNANA's
    random-list sync or the CPU time, none of which has a counterpart."""
    lines = Path(path).read_text().splitlines()
    kept = []
    for line in lines[lines.index('  OUTPUT_SUMMARY:') + 1:]:
        if line.startswith('DOCUMENTATION_END'):
            break
        text = line.split('#')[0].strip()
        if text and not text.startswith(('RANDOM_SYNC', '- List=', 'CPU_MINUTES')):
            kept.append(' '.join(text.split()))
    return kept


def test_readme_summary_matches_snana():
    """The README's OUTPUT_SUMMARY statistics equal SNANA's for the same
    events: the counts, the efficiency and its error, the accepted events by
    typing, the per-season numbers, the rejections and the host-match table.
    hostlib_e2e is the fixture that bundles SNANA's README."""
    fx = _FIXTURES / "hostlib_e2e"
    out, _, _ = _replay_written_events(fx, "readme_")
    assert _output_summary(out / "V.README") == _output_summary(fx / "SIM.README")


def _readme_types_and_filters(path):
    """The README's SNIA line, its GENTYPE_TO_NAME lines and its FILTERS
    block, read as YAML as SCONE and SNANA's batch layer read it. Each
    joined type name is a set: SNANA joins the names in the order its events
    first meet them, which a run on another random stream cannot reproduce."""
    docana = yaml.safe_load(Path(path).read_text())["DOCUMENTATION"]
    types = {gentype: (line.split()[0], set(line.split()[1].split("+")))
             for gentype, line in docana["GENTYPE_TO_NAME"].items()}
    return docana["OVERVIEW"]["SNIA"], types, docana["FILTERS"]


@pytest.mark.parametrize("mode,n_sne", [("hostlib_e2e", 25), ("gentype_map_k10", 200)])
def test_readme_types_and_filters_match_snana(mode, n_sne):
    """The README's GENTYPE_TO_NAME block, which SuperNNova and SCONE read to
    tell the Type Ia GENTYPEs from the rest, its SNIA line, and its FILTERS
    block of mean wavelengths, which SCONE reads, equal SNANA's: a SALT3
    sim, and K10 core-collapse templates with five SNTYPEs, one of them
    shared by the Ibc and Ic templates."""
    jax_dir = _simulate(mode, n_sne)
    assert (_readme_types_and_filters(jax_dir / "J.README")
            == _readme_types_and_filters(_FIXTURES / mode / "SIM.README"))


@pytest.mark.parametrize("mode", ["salt3_k21_replay", "hostlib_e2e",
                                  "hostlib_mwebv", "weaklens_probmap"])
def test_head_sentinel_values_match_snana(mode):
    """Sentinel and config-derived HEAD values match SNANA's, event by event.

    The contract tests above check which columns exist, not what is in them, so
    nothing else would notice a sentinel reverting to -9.

    Both a no-HOSTLIB and a HOSTLIB fixture are needed: several host columns
    carry one sentinel when nothing matched and a different one once a host
    did (MAGERR 999 vs -9, ELLIPTICITY -9999 vs 999, OBJID2 0 vs -9), so a
    single fixture pins only half of each pair. hostlib_mwebv is the only one
    with OPT_MWEBV on, so it is the only place HOSTGAL_MAG carries the Milky
    Way extinction SNANA adds to every reported host (sntools_host.c:9287),
    and weaklens_probmap the only one with a lensing map, so it is the only
    place SIM_LENSDMU and LENSDMU_ERR are anything but zero.
    """
    fx = _FIXTURES / mode
    out, snana, resources = _replay_written_events(fx, "sentinel_")

    with fits.open(resolve_fits_path(next(out.glob("V*HEAD*")))) as hdul:
        jax = {n: np.asarray(hdul[1].data[n]) for n in hdul[1].data.names}

    s_id = {int(v): i for i, v in enumerate(snana['SNID'])}
    j_id = {int(v): i for i, v in enumerate(jax['SNID'])}
    both = sorted(set(s_id) & set(j_id))
    assert len(both) > 100, f"only {len(both)} events replayed"

    bands = ''.join(n.rsplit('-', 1)[-1]
                    for n in resources.filter_names)
    columns = list(_EXACT_VALUE_COLUMNS) + [
        f'{fam}_{b}' for fam in ('HOSTGAL_MAG', 'HOSTGAL_MAGERR',
                                 'HOSTGAL2_MAG', 'HOSTGAL2_MAGERR',
                                 'HOSTGAL_SB_FLUXCAL', 'SIM_EXPOSURE')
        for b in bands]

    matched_host = np.array(
        [snana['HOSTGAL_NMATCH2'][s_id[c]] for c in both]) > 0
    checked, mismatched = 0, []
    for col in columns:
        if (col not in snana or col not in jax
                or col in _SELECTION_DERIVED or col in _INDEPENDENT_DRAW
                or col.startswith(_SELECTION_DERIVED_PREFIXES)):
            continue
        a = np.array([snana[col][s_id[c]] for c in both], dtype=np.float64)
        b = np.array([jax[col][j_id[c]] for c in both], dtype=np.float64)
        checked += 1
        # DMUCOR makes the host mags a computed quantity, not a stored one
        rtol, atol, subset = 1e-5, 0.0, slice(None)
        for prefix, (r, at) in _POSITION_DERIVED_TOL.items():
            if col.startswith(prefix):
                # SNANA still derives these from the drawn galaxy when the DDLR
                # match fails, while replay nulls its host properties there, so
                # compare only where SNANA reports a match.
                rtol, atol, subset = r, at, matched_host
        close = np.isclose(a[subset], b[subset], rtol=rtol, atol=atol)
        if not close.all():
            n = int((~close).sum())
            mismatched.append(f"{col}: {n}/{len(both)} events differ, "
                              f"SNANA {a[subset][~close][0]} vs "
                              f"jax {b[subset][~close][0]}")
    assert checked > 30, f"only {checked} columns compared — check the names"
    assert not mismatched, "\n".join(mismatched)


def test_hostgal2_values_by_match_state_match_snana():
    """Second-host columns by match state, against SNANA's own HEAD.

    Replay cannot reproduce SNANA's DDLR-sorted host list: the header records
    the reported hosts, not the SN's offset from the drawn one. So the
    neighbour block is compared as value sets per match state. Without a
    neighbour every column carries one sentinel; with one, a column the
    HOSTLIB lacks carries the sorted list's reset value, and a column it has
    never carries the no-neighbour sentinel.
    """
    fx = _FIXTURES / "hostlib_nbr"
    with fits.open(resolve_fits_path(fx / "HEAD.FITS")) as hdul:
        snana = {n: np.asarray(hdul[1].data[n]) for n in hdul[1].data.names}

    out = Path(tempfile.mkdtemp(prefix="nbr_"))
    config = dict(n_sne=len(snana['SNID']), random_seed=7)
    sim = sim_run.prepare(str(fx / "input.input"), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results, selection, spectra_cache = sim_run.run_simulation(sim)
        sim_run.save_snana_outputs(sim, results, "N", selection, spectra_cache=spectra_cache, output_dir=str(out))
    with fits.open(resolve_fits_path(next((out / "N").glob("N*HEAD*")))) as hdul:
        jax = {n: np.asarray(hdul[1].data[n]) for n in hdul[1].data.names}

    matched_s = snana['HOSTGAL2_OBJID'] > 0
    matched_j = jax['HOSTGAL2_OBJID'] > 0
    assert matched_s.sum() > 100 and (~matched_s).sum() > 10
    f_s, f_j = matched_s.mean(), matched_j.mean()
    scatter = np.sqrt(f_s * (1 - f_s) * (1 / len(matched_s) + 1 / len(matched_j)))
    assert abs(f_j - f_s) < 3 * scatter, f"neighbour fraction {f_j} vs SNANA {f_s}"

    values = lambda table, col, sel: set(np.unique(table[col][sel]).tolist())
    mismatched = []
    for col in [c for c in snana if c.startswith('HOSTGAL2_')]:
        s_none, j_none = values(snana, col, ~matched_s), values(jax, col, ~matched_j)
        if j_none != s_none:
            mismatched.append(f"{col} without a neighbour: SNANA {s_none} "
                              f"vs jax {j_none}")
        s_some, j_some = values(snana, col, matched_s), values(jax, col, matched_j)
        if len(s_some) == 1 and j_some != s_some:
            mismatched.append(f"{col} with a neighbour: SNANA {s_some} "
                              f"vs jax {j_some}")
        if len(s_some) > 1 and (j_some & s_none):
            mismatched.append(f"{col} with a neighbour carries {j_some & s_none}")
    assert not mismatched, "\n".join(mismatched)


# PHOT columns copied verbatim from the SIMLIB S: row.
_SIMLIB_PHOT_COLUMNS = ['RDNOISE', 'ZEROPT_ERR', 'ZEROPT', 'GAIN', 'TEXPOSE',
                        'PSF_SIG1', 'PSF_SIG2', 'PSF_RATIO', 'SKY_SIG']


def test_phot_simlib_values_match_snana():
    """Per-epoch SIMLIB quantities in PHOT match SNANA's, epoch by epoch.

    Paired on (SNID, MJD) rather than row order, so a differing epoch count
    cannot silently misalign every subsequent row.
    """
    mode = "salt3_k21_replay"
    fx = _FIXTURES / mode
    with fits.open(resolve_fits_path(fx / "HEAD.FITS")) as h:
        sh = {n: np.asarray(h[1].data[n]) for n in h[1].data.names}
    with fits.open(resolve_fits_path(fx / "PHOT.FITS")) as h:
        sp = {n: np.asarray(h[1].data[n]) for n in h[1].data.names}

    dump = load_snana_dump(str(fx / "DUMP"), dedup_cid=True)
    dump_cols, head = select_replay(dump.columns, dict(sh),
                                    [int(s) for s in sh['SNID']])
    out = Path(tempfile.mkdtemp(prefix="photval_"))
    config = dict(n_sne=len(dump_cols['CID']))
    sim = sim_run.prepare(str(fx / "input.input"), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        override = build_override_parameters(dump_cols, simulation, resources,
                                             head_columns=head)
        results, selection, spectra_cache = sim_run.run_simulation(sim, override_parameters=override)
        sim_run.save_snana_outputs(sim, results, "Q", selection, spectra_cache=spectra_cache, output_dir=str(out))

    with fits.open(resolve_fits_path(next((out / "Q").glob("Q*HEAD*")))) as h:
        jh = {n: np.asarray(h[1].data[n]) for n in h[1].data.names}
    with fits.open(resolve_fits_path(next((out / "Q").glob("Q*PHOT*")))) as h:
        jp = {n: np.asarray(h[1].data[n]) for n in h[1].data.names}

    def epoch_rows(head, phot):
        rows = {}
        for i, sid in enumerate(head['SNID']):
            lo = int(head['PTROBS_MIN'][i]) - 1
            hi = int(head['PTROBS_MAX'][i])
            for r in range(lo, hi):
                rows[(int(sid), round(float(phot['MJD'][r]) / 0.01))] = r
        return rows

    s_rows, j_rows = epoch_rows(sh, sp), epoch_rows(jh, jp)
    shared = sorted(set(s_rows) & set(j_rows))
    assert len(shared) > 1000, f"only {len(shared)} epochs paired"

    mismatched = []
    for col in _SIMLIB_PHOT_COLUMNS:
        if col not in sp or col not in jp:
            continue
        a = np.array([sp[col][s_rows[k]] for k in shared], dtype=np.float64)
        b = np.array([jp[col][j_rows[k]] for k in shared], dtype=np.float64)
        # float32 storage of SIMLIB values; a few ULP is the achievable floor
        tol = 4 * np.spacing(np.maximum(np.abs(a), 1.0).astype(np.float32))
        bad = np.abs(a - b) > tol
        if bad.any():
            mismatched.append(f"{col}: {int(bad.sum())}/{len(shared)} epochs "
                              f"differ, SNANA {a[bad][0]} vs jax {b[bad][0]}")
    # FIELD carries the per-epoch SIMLIB field name
    a = np.array([str(sp['FIELD'][s_rows[k]]).strip() for k in shared])
    b = np.array([str(jp['FIELD'][j_rows[k]]).strip() for k in shared])
    assert set(a) != {'VOID'}, "reference FIELD is all VOID; check is vacuous"
    bad = a != b
    if bad.any():
        mismatched.append(f"FIELD: {int(bad.sum())}/{len(shared)} epochs "
                          f"differ, SNANA {a[bad][0]} vs jax {b[bad][0]}")
    assert not mismatched, "\n".join(mismatched)
