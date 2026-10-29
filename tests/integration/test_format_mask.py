"""FORMAT_MASK: output-format selection (TEXT/FITS) + modifier bits.

TEXT and FITS are written from the same HEAD/PHOT tables, so a TEXT run and a
FITS run of the same seed must load to the same light curves (round-trip via
the jax FITS and TEXT loaders). Modifier bits (BLINDTEST/COMPACT/FILTERS/CIDRAN)
and the SNANA validation aborts are checked directly.
"""

import os
import re
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import SNANA_VERSION_STRING, resolve_fits_path
from _fit_helpers import (load_and_batch_snana_fits, load_and_batch_snana_text)
from jax_snlc_sim import run as sim_run
from jax_snlc_sim.output import save_snana_outputs
from jax_snlc_sim.output.fits import _write_bintable, write_head_fits

_ROOT = "$JAX_TEST_FIXTURE_ROOT"
_BANDS = ['LSST-g', 'LSST-r', 'LSST-i']


def _write_input(out: Path, mask) -> None:
    out.write_text(f"""GENVERSION: JAX_FMASK
FORMAT_MASK: {mask}
GENTYPE: 1
SNTYPES_Ia: 1 101
GENMODEL: {_ROOT}/models/SALT3.K21
GENPEAK_SALT2x1: 0.0
GENRANGE_SALT2x1: -2.0 2.0
GENSIGMA_SALT2x1: 1.0 1.0
GENPEAK_SALT2c: 0.0
GENRANGE_SALT2c: -0.2 0.3
GENSIGMA_SALT2c: 0.05 0.05
GENPEAK_SALT2ALPHA: 0.14
GENPEAK_SALT2BETA: 3.1
OMEGA_MATTER: 0.315
OMEGA_LAMBDA: 0.685
w0_LAMBDA: -1.0
GENFILTERS: ugrizY
KCOR_FILE: {_ROOT}/kcor/JAX_TEST_LSST.fits
SIMLIB_FILE: {_ROOT}/simlib/JAX_TEST_LSST.SIMLIB
GENRANGE_REDSHIFT: 0.05 0.40
GENRANGE_PEAKMJD: 60540 61460
GENSIGMA_REDSHIFT: 0.001
GENRANGE_TREST: -20 50
SOLID_ANGLE: 0.015
OPT_MWEBV: 0
DNDZ: POWERLAW 2.5E-5 1.5
RANSEED: 12345
NGENTOT_LC: 40
""")


def _run(mask, outdir) -> str:
    os.makedirs(outdir, exist_ok=True)
    inp = Path(outdir) / "in.input"
    _write_input(inp, mask)
    cfg = dict(n_sne=40, random_seed=12345)
    sim = sim_run.prepare(str(inp), **cfg)
    results, sel, spectra_cache = sim_run.run_simulation(sim)
    save_snana_outputs(sim, results, "JAX_FMASK", sel, spectra_by_sn=spectra_cache, output_dir=outdir)
    return os.path.join(outdir, "JAX_FMASK")


@pytest.fixture(scope="module")
def fits_baseline(tmp_path_factory):
    """Plain FORMAT_MASK=32 output, shared by the tests that only read it."""
    return _run(32, str(tmp_path_factory.mktemp("fits_base")))


def test_text_fits_roundtrip(tmp_path, fits_baseline):
    """A TEXT run and a FITS run of the same seed load to identical light curves."""
    fdir = fits_baseline
    tdir = _run(2, str(tmp_path / "text"))
    assert os.path.exists(resolve_fits_path(os.path.join(fdir, "JAX_FMASK_HEAD.FITS")))
    assert os.path.exists(resolve_fits_path(os.path.join(fdir, "JAX_FMASK_PHOT.FITS")))
    assert sum(f.endswith('.DAT') for f in os.listdir(tdir)) == 40
    assert Path(tdir, "JAX_FMASK_SN000001.DAT").exists()

    bf = load_and_batch_snana_fits(
        os.path.join(fdir, "JAX_FMASK_HEAD.FITS"),
        os.path.join(fdir, "JAX_FMASK_PHOT.FITS"), fit_bands=_BANDS)
    bt = load_and_batch_snana_text(
        tdir, fit_bands=_BANDS)

    sf = {int(s): i for i, s in enumerate(np.asarray(bf.snids))}
    st = {int(s): i for i, s in enumerate(np.asarray(bt.snids))}
    assert set(sf) == set(st) and len(sf) == 40
    fcf, fct = np.asarray(bf.fluxcals), np.asarray(bt.fluxcals)
    # same obs per SN and matching fluxes (TEXT is fprintf-truncated, ~1e-3 rel)
    for cid in sf:
        a, b = np.sort(fcf[sf[cid]]), np.sort(fct[st[cid]])
        assert len(a) == len(b)
        scale = max(1.0, float(np.max(np.abs(a))))
        assert float(np.max(np.abs(a - b))) / scale < 2e-3


def test_blindtest_text(tmp_path):
    """BLINDTEST(8): FAKE=3, no SIM_* truth keys, no SIM_MAGOBS column."""
    tdir = _run(2 | 8, str(tmp_path / "blind"))
    dat = next(f for f in os.listdir(tdir) if f.endswith('.DAT'))
    txt = Path(tdir, dat).read_text()
    assert "FAKE: 3" in txt
    assert "SIM_" not in txt


def test_compact_fits(tmp_path):
    """COMPACT(64): FITS PHOT table keeps only essential columns, still readable."""
    fdir = _run(32 | 64, str(tmp_path / "compact"))
    with fits.open(resolve_fits_path(os.path.join(fdir, "JAX_FMASK_PHOT.FITS"))) as h:
        cols = {c.name for c in h[1].columns}
    assert cols == {'MJD', 'BAND', 'FLUXCAL', 'FLUXCALERR', 'SIM_MAGOBS'}
    b = load_and_batch_snana_fits(
        os.path.join(fdir, "JAX_FMASK_HEAD.FITS"),
        os.path.join(fdir, "JAX_FMASK_PHOT.FITS"), fit_bands=_BANDS)
    assert b.n_sne == 40


def test_blindtest_fits(tmp_path):
    """BLINDTEST(8) FITS: SIM_* columns dropped from HEAD, still readable."""
    fdir = _run(32 | 8, str(tmp_path / "blindf"))
    with fits.open(resolve_fits_path(os.path.join(fdir, "JAX_FMASK_HEAD.FITS"))) as h:
        assert not any(c.name.startswith('SIM_') for c in h[1].columns)
    b = load_and_batch_snana_fits(
        os.path.join(fdir, "JAX_FMASK_HEAD.FITS"),
        os.path.join(fdir, "JAX_FMASK_PHOT.FITS"), fit_bands=_BANDS)
    assert b.n_sne == 40


def test_filters(tmp_path):
    """FILTERS(256): per-band transmission files."""
    fdir = _run(32 | 256, str(tmp_path / "filt"))
    fpath = Path(fdir, "JAX_FMASK.FILTERS")
    assert fpath.is_dir()
    assert {p.name for p in fpath.glob("*.dat")} == {f"{b}.dat" for b in "ugrizY"}
    first = next(fpath.glob("*.dat")).read_text().splitlines()[0].split()
    assert len(first) == 2  # wavelength  transmission


def test_cidran(tmp_path):
    """CIDRAN(16): unique random CIDs in [CIDRAN_MIN, CIDRAN_MAX], not sequential."""
    fdir = _run(32 | 16, str(tmp_path / "cidran"))
    with fits.open(resolve_fits_path(os.path.join(fdir, "JAX_FMASK_HEAD.FITS"))) as h:
        cids = [int(str(s).strip()) for s in h[1].data['SNID']]
    assert len(set(cids)) == len(cids) == 40
    assert all(0 <= c <= 3000000 for c in cids)
    assert sorted(cids)[:5] != [1, 2, 3, 4, 5]


def _run_extra(mask, outdir, extra) -> str:
    """Like _run, but append extra input lines (for run-admin key tests)."""
    os.makedirs(outdir, exist_ok=True)
    inp = Path(outdir) / "in.input"
    _write_input(inp, mask)
    inp.write_text(inp.read_text() + extra)
    cfg = dict(n_sne=40, random_seed=12345)
    sim = sim_run.prepare(str(inp), **cfg)
    results, sel, spectra_cache = sim_run.run_simulation(sim)
    save_snana_outputs(sim, results, "JAX_FMASK", sel, spectra_by_sn=spectra_cache, output_dir=outdir)
    return os.path.join(outdir, "JAX_FMASK")


def test_genprefix_and_simlib_survey(tmp_path):
    """GENPREFIX names the data files (default GENVERSION); the directory + aux
    files keep GENVERSION. SIMLIB_SURVEY overrides the HEAD SURVEY name.

    The override has to name a real SURVEY.DEF survey: SNANA resolves IDSURVEY
    from it while reading the SIMLIB header and aborts otherwise
    (SIMLIB_prepGlobalHeader, snlc_sim.c:18577). CSP differs from the fixture's
    native LSST.
    """
    fdir = _run_extra(32, str(tmp_path / "gp"),
                      "GENPREFIX: FOO\nSIMLIB_SURVEY: CSP\n")
    # data files use GENPREFIX
    assert os.path.exists(os.path.join(fdir, "FOO_HEAD.FITS.gz"))
    assert os.path.exists(os.path.join(fdir, "FOO_PHOT.FITS.gz"))
    assert not os.path.exists(os.path.join(fdir, "JAX_FMASK_HEAD.FITS.gz"))
    # directory + aux files keep GENVERSION
    assert os.path.basename(fdir) == "JAX_FMASK"
    assert os.path.exists(os.path.join(fdir, "JAX_FMASK.LIST"))
    assert os.path.exists(os.path.join(fdir, "JAX_FMASK.README"))
    # SIMLIB_SURVEY override lands in the HEAD SURVEY keyword; loader still reads it
    with fits.open(resolve_fits_path(os.path.join(fdir, "FOO_HEAD.FITS"))) as h:
        assert h[0].header['SURVEY'].strip() == "CSP"
    b = load_and_batch_snana_fits(
        os.path.join(fdir, "FOO_HEAD.FITS"),
        os.path.join(fdir, "FOO_PHOT.FITS"), fit_bands=_BANDS)
    assert b.n_sne == 40


def test_prescale_simgen_dump(tmp_path):
    """PRESCALE_SIMGEN_DUMP: N writes 1 of every N SNe (NEVT % N == 0) to the .DUMP."""
    def dump_rows(fdir):
        txt = Path(fdir, "JAX_FMASK.DUMP").read_text()
        return sum(l.startswith("SN:") for l in txt.splitlines())

    dump_key = "SIMGEN_DUMP: 5 CID GENZ PEAKMJD SNRMAX CUTMASK\n"
    full = dump_rows(_run_extra(32, str(tmp_path / "full"), dump_key))
    half = dump_rows(_run_extra(32, str(tmp_path / "ps2"),
                                dump_key + "PRESCALE_SIMGEN_DUMP: 2\n"))
    assert full > 0
    assert half == full // 2


def test_no_dump_without_simgen_dump(fits_baseline):
    """No SIMGEN dump is written without a SIMGEN_DUMP/SIMGEN_DUMPALL key
    (wr_SIMGEN_DUMP returns when NVAR_SIMGEN_DUMP < 0, snlc_sim.c:14833)."""
    assert not os.path.exists(os.path.join(fits_baseline, "JAX_FMASK.DUMP"))


def test_simgen_dump_zero_lists_and_stops(tmp_path, capsys):
    """SIMGEN_DUMP: 0 prints the available dump variables and stops before
    generating (PREP_SIMGEN_DUMP(0) + debugexit, snlc_sim.c:14860-14862).
    `prepare` returns no compiled simulator, so every caller stops, not only
    the command line."""
    outdir = str(tmp_path / "d0")
    os.makedirs(outdir, exist_ok=True)
    inp = Path(outdir) / "in.input"
    _write_input(inp, 32)
    inp.write_text(inp.read_text() + "SIMGEN_DUMP: 0\n")
    cfg = dict(n_sne=4, random_seed=1)
    sim = sim_run.prepare(str(inp), **cfg)
    assert sim is None
    printed = capsys.readouterr().out
    assert "list variables and quit" in printed
    assert "  CID" in printed


def test_fits_list_names_head_file(fits_baseline):
    """The FITS-mode .LIST carries the actual HEAD file name
    (WR_SNFITSIO_INIT returns it, snlc_sim.c:31000-31008)."""
    listtxt = Path(fits_baseline, "JAX_FMASK.LIST").read_text()
    assert listtxt == "JAX_FMASK_HEAD.FITS\n"


def test_readme_reports_resolved_config(tmp_path):
    """README INPUT_KEYS come from the parsed configuration, not from
    hard-coded example values."""
    fdir = _run_extra(32, str(tmp_path / "rp"),
                      "GENSIGMA_REDSHIFT: 0.005\nGENMAG_SMEAR: 0.1\n")
    txt = Path(fdir, "JAX_FMASK.README").read_text()
    assert "GENSIGMA_REDSHIFT:       5.0000e-03" in txt
    assert "GENMAG_SMEAR:            0.1" in txt
    assert "GENMAG_SMEAR_MODELNAME:  NONE" in txt
    assert re.search(r"RANSEED:\s+\d+", txt)
    assert "APPLY_SEARCHEFF_OPT:     0" in txt


def test_readme_sntype_defaults_without_sntype_keys(tmp_path):
    """Without SNTYPE_Ia/SNTYPES_Ia the README writes SNANA's defaults, 1 and
    101 (snlc_sim.c:811-812), whatever GENTYPE is."""
    outdir = tmp_path / "st"
    os.makedirs(outdir)
    inp = outdir / "in.input"
    _write_input(inp, 32)
    inp.write_text("\n".join(
        "GENTYPE: 10" if line.startswith("GENTYPE:") else line
        for line in inp.read_text().splitlines()
        if not line.startswith("SNTYPES_Ia:")) + "\n")
    cfg = dict(n_sne=40, random_seed=12345)
    sim = sim_run.prepare(str(inp), **cfg)
    results, sel, spectra_cache = sim_run.run_simulation(sim)
    save_snana_outputs(sim, results, "JAX_FMASK", sel, spectra_by_sn=spectra_cache, output_dir=str(outdir))
    txt = Path(outdir, "JAX_FMASK", "JAX_FMASK.README").read_text()
    assert re.search(r"^\s+SNTYPE:\s+1  101  #", txt, re.M)


def test_wgt_population_undoes_the_exponent_reweighting(tmp_path):
    """GENPDF_EXPON_REWGT draws stretch and colour from prob^e; the written
    weight is prob(nominal)/prob(drawn), as SNANA's rewgt_genPDF computes it
    (snlc_sim.c:31492), so reweighting the output recovers the population."""
    fdir = _run_extra(32, str(tmp_path / "wp"), "GENPDF_EXPON_REWGT: 2.0\n")
    with fits.open(resolve_fits_path(os.path.join(fdir, "JAX_FMASK_HEAD.FITS"))) as h:
        x1 = np.asarray(h[1].data["SIM_SALT2x1"], dtype=float)
        c = np.asarray(h[1].data["SIM_SALT2c"], dtype=float)
        wgt = np.asarray(h[1].data["SIM_WGT_POPULATION"], dtype=float)
    # the fixture's populations: x1 ~ N(0, 1), c ~ N(0, 0.05), each with e - 1 = 1
    expected = np.exp(0.5 * x1 ** 2) * np.exp(0.5 * (c / 0.05) ** 2)

    assert wgt.min() > 1.0
    np.testing.assert_allclose(wgt, expected, rtol=2e-4)


def test_readme_ngentot_is_the_count_to_generate(tmp_path):
    """NGENTOT_LC is an input key: the number of light curves to generate, as
    SNANA prints it once setup has resolved it (sntools_sim_readme.c:1332).
    Cut hard enough that fewer are written than generated, which is the only
    way the written count masquerading as the input is visible."""
    fdir = _run_extra(32, str(tmp_path / "ng"),
                      "APPLY_CUTWIN_OPT: 1\nCUTWIN_SNRMAX: 30 griz 2 -20. 80.\n")
    txt = Path(fdir, "JAX_FMASK.README").read_text()
    generated = int(re.search(r"NGENLC_TOT:\s+(\d+)", txt).group(1))
    written = int(re.search(r"NGENLC_WRITE:\s+(\d+)", txt).group(1))

    assert written < generated, "the cut kept everything, so this proves nothing"
    assert f"NGENTOT_LC:              {generated}" in txt


def test_readme_echoes_the_inputs_the_resolved_block_does_not(tmp_path):
    """After the values setup resolved, the README carries every other input
    line as given (README_KEYPLUSARGS_load), so a cut, a template table or a
    model's own key is on record without the writer naming it."""
    fdir = _run_extra(32, str(tmp_path / "ek"),
                      "APPLY_CUTWIN_OPT: 1\nCUTWIN_SNRMAX: 30 griz 2 -20. 80.\n")
    txt = Path(fdir, "JAX_FMASK.README").read_text()
    keys = txt.split("INPUT_KEYS:")[1].split("OUTPUT_SUMMARY:")[0]
    assert "    CUTWIN_SNRMAX: 30 griz 2 -20. 80." in keys
    assert "    APPLY_CUTWIN_OPT: 1" in keys
    # a key the resolved block writes is not repeated from the input
    assert keys.count("GENVERSION:") == 1


def test_gzip_default_and_off(tmp_path, fits_baseline):
    """GZIP_DATA_FILES defaults ON (match SNANA): FITS output is gzipped, the
    .LIST still names the logical .FITS, and the loader reads it back via the
    resolver. GZIP_DATA_FILES: 0 leaves the FITS uncompressed."""
    fdir = fits_baseline
    assert os.path.exists(os.path.join(fdir, "JAX_FMASK_HEAD.FITS.gz"))
    assert os.path.exists(os.path.join(fdir, "JAX_FMASK_PHOT.FITS.gz"))
    assert not os.path.exists(os.path.join(fdir, "JAX_FMASK_HEAD.FITS"))
    listtxt = Path(fdir, "JAX_FMASK.LIST").read_text()
    assert ".FITS" in listtxt and ".FITS.gz" not in listtxt  # logical name only
    b = load_and_batch_snana_fits(
        os.path.join(fdir, "JAX_FMASK_HEAD.FITS"),
        os.path.join(fdir, "JAX_FMASK_PHOT.FITS"), fit_bands=_BANDS)
    assert b.n_sne == 40

    odir = str(tmp_path / "plain")
    os.makedirs(odir, exist_ok=True)
    inp = Path(odir) / "in.input"
    _write_input(inp, 32)
    inp.write_text(inp.read_text() + "GZIP_DATA_FILES: 0\n")
    cfg = dict(n_sne=40, random_seed=12345)
    sim = sim_run.prepare(str(inp), **cfg)
    res, sel, spectra_cache = sim_run.run_simulation(sim)
    save_snana_outputs(sim, res, "JAX_FMASK", sel, spectra_by_sn=spectra_cache, output_dir=odir)
    pdir = os.path.join(odir, "JAX_FMASK")
    assert os.path.exists(os.path.join(pdir, "JAX_FMASK_HEAD.FITS"))
    assert not os.path.exists(os.path.join(pdir, "JAX_FMASK_HEAD.FITS.gz"))


@pytest.mark.parametrize("mask", [1, 2 | 32, 128, 8])
def test_format_mask_aborts(tmp_path, mask):
    """VBOSE(1), TEXT|FITS(34), ATMOS(128), and format-less(8) abort at init."""
    inp = tmp_path / "in.input"
    _write_input(inp, mask)
    cfg = dict(n_sne=4, random_seed=1)
    with pytest.raises(SNANAError):
        sim_run.prepare(str(inp), **cfg)


def test_bintable_writer_matches_astropy(tmp_path):
    """The FITS binary table is written directly rather than through astropy,
    which was spending its time converting column objects rather than moving
    bytes. astropy is the oracle: an independent implementation of the same
    format, so the data block must agree byte for byte and the header must
    agree keyword for keyword.

    Comments are deliberately not compared. They are free-form in FITS and
    astropy's wording varies between its own versions, so pinning them would
    pin the test to a release rather than to the format.
    """
    columns = {
        'MJD': np.linspace(60000.0, 60100.0, 64, dtype=np.float64),
        'BAND': np.full(64, b'LSST-r', dtype='S20'),
        'DETNUM': np.arange(64, dtype=np.int16),
        'PHOTFLAG': np.arange(64, dtype=np.int32),
        'FLUXCAL': np.linspace(-5.0, 500.0, 64, dtype=np.float32),
        'NOBS': np.arange(64, dtype=np.int64),
    }
    mine = tmp_path / 'mine.FITS'
    with open(mine, 'wb') as fh:
        fits.PrimaryHDU().writeto(fh)
        _write_bintable(fh, columns, 'Photometry')

    theirs = tmp_path / 'theirs.FITS'
    hdu = fits.BinTableHDU.from_columns(
        [fits.Column(name=n, array=v, format=f) for n, v, f in (
            ('MJD', columns['MJD'], 'D'), ('BAND', columns['BAND'], '20A'),
            ('DETNUM', columns['DETNUM'], 'I'),
            ('PHOTFLAG', columns['PHOTFLAG'], 'J'),
            ('FLUXCAL', columns['FLUXCAL'], 'E'),
            ('NOBS', columns['NOBS'], 'K'))], name='Photometry')
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(theirs)

    with fits.open(mine) as a, fits.open(theirs) as b:
        assert a[1].data.dtype == b[1].data.dtype
        assert a[1].data.tobytes() == b[1].data.tobytes(), \
            "the data block differs from astropy's"
        for key in ('XTENSION', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2',
                    'PCOUNT', 'GCOUNT', 'TFIELDS'):
            assert a[1].header[key] == b[1].header[key], key
        # astropy upper-cases the extension name where SNANA's own files
        # carry it as given, and FITS keyword values are not case-sensitive
        assert (a[1].header['EXTNAME'].upper()
                == b[1].header['EXTNAME'].upper())
        for i, name in enumerate(columns, start=1):
            assert a[1].header[f'TTYPE{i}'] == name
            assert (a[1].header[f'TFORM{i}'].lstrip('1')
                    == b[1].header[f'TFORM{i}'].lstrip('1')), f'TFORM{i}'
        # every value survives the round trip, not just the layout; astropy
        # decodes character columns on read, so both sides are compared as text
        for name, want in columns.items():
            got = a[1].data[name]
            if want.dtype.kind == 'S':
                got = np.char.encode(np.asarray(got, dtype=str), 'ascii')
            np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("simlib_path", [
    '/' + 'a' * 120 + '/X.SIMLIB',
    "/data/observer's/X.SIMLIB",
    "/data/observer''s/X.SIMLIB",
    '/' + "observer's/" * 15 + 'X.SIMLIB',
    '/' + 'a' * 66,
])
def test_long_header_keywords_use_hierarch(tmp_path, simlib_path):
    """SNANA's own headers carry HIERARCH cards for the keywords beyond the
    eight characters FITS allows, WRITE_MASK_HEAD and CODE_IVERSION among them
    (sntools_dataformat_fits.c). Writing them as plain cards puts the equals
    sign past column nine and readers silently truncate the name. A file path
    beyond one card continues over CONTINUE cards, as SNANA's do. Quoted
    paths and values near a card boundary must also survive unchanged.
    """
    head = {'SNID': np.array([b'000001'], dtype='S16'),
            'PTROBS_MIN': np.array([1], dtype=np.int32),
            'PTROBS_MAX': np.array([1], dtype=np.int32)}
    keywords = {'CODE_IVERSION': 27, 'ZP_FLUXCAL': 27.5,
                'SNANA_VERSION': SNANA_VERSION_STRING,
                'SIM_VARNAME_SNRMON': 'SIM_SNRMAG21',
                'SIMLIB_FILE': simlib_path,
                'WRITE_MASK_HEAD': 36}
    write_head_fits(keywords, 'T', str(tmp_path), head)
    hdr = fits.getheader(str(tmp_path / 'T_HEAD.FITS'), 0)
    for key, value in keywords.items():
        assert hdr[key] == value, key
