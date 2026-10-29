"""Fitter configuration parsing and FITS/TEXT photometry loading."""

import dataclasses
import os

import numpy as np
import pytest

from jax_snlc_core.filters import KcorReader
from jax_snlc_core.logging import SNANAError
from jax_snlc_fit import LightCurveFitter
from jax_snlc_fit.config import FitConfig, Model
from jax_snlc_fit.config_parser import build, read_keys
from jax_snlc_fit.loader import (_host_photoz_columns, read_snana_fits,
                                 read_snana_text)


def test_yaml_and_nml_equivalent(tmp_path):
    os.environ["MY_ROOT"] = "run_root"

    nml = tmp_path / "fit.nml"
    nml.write_text(
        "&SNLCINP\n"
        "  KCOR_FILE = '$MY_ROOT/kcor.fits'\n"
        "  VERSION_PHOTOMETRY = 'MY_SIM'\n"
        "&END\n"
        "&FITINP\n"
        "  FITMODEL_NAME = 'REDBACK.arnett_bolometric'\n"
        "&END\n"
    )
    yml = tmp_path / "fit.yaml"
    yml.write_text(
        "SNLCINP:\n"
        "  kcor_file: $MY_ROOT/kcor.fits\n"
        "  version_photometry: MY_SIM\n"
        "FITINP:\n"
        "  fitmodel_name: REDBACK.arnett_bolometric\n"
    )

    assert dict(read_keys(nml)) == dict(read_keys(yml))


def test_sections_flatten_with_keys_uppercased(tmp_path):
    os.environ["KDIR"] = "/k"
    yml = tmp_path / "fit.yml"
    yml.write_text(
        "snlcinp:\n"
        "  kcor_file: $KDIR/lsst.fits\n"
        "fitinp:\n"
        "  fitmodel_name: REDBACK.arnett_bolometric\n"
    )
    raw = read_keys(yml)
    # the section level is parsing syntax only; it does not survive
    assert "SNLCINP" not in raw and "FITINP" not in raw
    assert raw["KCOR_FILE"] == "/k/lsst.fits"
    # Values are not uppercased, only keys.
    assert raw["FITMODEL_NAME"] == "REDBACK.arnett_bolometric"


def test_inival_inistp_reach_the_config(tmp_path):
    """Pinning a model parameter uses SNANA's own keys — there is no jax-only
    fit-control vocabulary. Suffixes stay unresolved until a model exists."""
    yml = tmp_path / "fit.yaml"
    yml.write_text(
        "FITINP:\n"
        "  FITMODEL_NAME: REDBACK.arnett_bolometric\n"
        "  INIVAL_kappa: 0.07\n"
        "  INISTP_kappa: 0\n"
    )
    cfg = build(read_keys(yml))
    # Keys are uppercased; the model matches case-insensitively.
    assert cfg.model.inival == {"KAPPA": 0.07}
    assert cfg.model.inistp == {"KAPPA": 0.0}


def test_opt_photoz_steffen_bit_implies_the_quantiles(tmp_path):
    """OPT_PHOTOZ 8 alone is the Steffen interpolant with no prior to apply
    it to; SNANA sets the quantile bit for the user (snlc_fit.F90:11663),
    so 8 reads as 12."""
    yml = tmp_path / "fit.yaml"
    yml.write_text("FITINP:\n  FITMODEL_NAME: SALT3.K21\n  OPT_PHOTOZ: 8\n")
    cfg = build(read_keys(yml))
    assert cfg.photoz.opt_photoz == 12 and cfg.photoz.dofit_photoz
    assert cfg.photoz.has_photoz_quantiles()
    assert cfg.photoz.quantile_spline() == "steffen"


def test_key_outside_a_section_is_not_read(tmp_path):
    """Namelist keys only count inside &SECTION ... &END."""
    nml = tmp_path / "fit.nml"
    nml.write_text(
        "&SNLCINP\n  KCOR_FILE = '/k.fits'\n&END\n"
        "  USE_MINOS = T\n"
        "&FITINP\n  FITMODEL_NAME = 'SALT3.K21'\n&END\n"
    )
    raw = read_keys(nml)
    assert "USE_MINOS" not in raw
    assert raw["KCOR_FILE"] == "/k.fits"


def test_alias_resolves_to_the_canonical_key(tmp_path):
    """CALIB_FILE is SNANA's current spelling of KCOR_FILE (snana.F90:65)."""
    nml = tmp_path / "fit.nml"
    nml.write_text("&SNLCINP\n  CALIB_FILE = '/k.fits'\n&END\n")
    raw = read_keys(nml)
    assert raw["KCOR_FILE"] == "/k.fits"
    assert "CALIB_FILE" not in raw


def test_parse_sntable_list():
    """SNTABLE_LIST grammar (snana.F90:6540-6740, empirically anchored)."""
    from jax_snlc_fit.config_parser import parse_sntable_list

    s = parse_sntable_list("DEFAULT")
    assert s['tables'] == {'snana': 'key', 'fitres': 'key'}

    assert parse_sntable_list("NONE")['tables'] == {}
    assert parse_sntable_list("")['tables'] == {}

    s = parse_sntable_list("FITRES(text:host)")
    assert s['tables'] == {'fitres': 'key'} and s['host_text']
    # text:host2 also enables the HOST_ identity set (snana.F90:23440).
    # The parse returns only the fields the string set, so an option the
    # string never named is absent and the Output record's default stands.
    s = parse_sntable_list("FITRES(text:host2)")
    assert s['host2_text'] and 'host_text' not in s

    s = parse_sntable_list("FITRES(text:csv) NOSIMVAR NOZPHOT")
    assert s['tables']['fitres'] == 'csv' and s['nosimvar'] and s['nozphot']

    s = parse_sntable_list("SNANA(iauc,cutmask:0) OUTLIER(nsig:1.5) LCPLOT(mxlc:5)")
    assert s['iauc'] and s['cutmask'] == 0
    assert s['outlier_nsig'] == 1.5 and s['mxlc'] == 5

    # TEXT no-ops collect warnings but parse (verified vs SNANA)
    s = parse_sntable_list("FITRES+RESIDUAL(ps:4,text:errcalc) SNANA+EPOCH PRIOR")
    assert set(s['tables']) == {'fitres', 'snana'}
    assert len(s['unsupported']) == 5

    with pytest.raises(SNANAError):
        parse_sntable_list("BOGUS_TABLE")
    with pytest.raises(SNANAError):
        parse_sntable_list("FITRES(text:xml)")


def test_write_varnames_table_styles(tmp_path):
    """key/csv/col/none output shapes (sntools_output_text.c:578-732)."""
    from jax_snlc_core.snana_files import write_varnames_table

    cols = {'CID': [1, 2], 'x1': [0.5, -1.25]}
    fmt = {'CID': '%d', 'x1': '%.2f'}
    p = tmp_path / "t.TEXT"

    write_varnames_table(p, ['CID', 'x1'], cols, fmt, style='key',
                         header_comments=['TABLE_NAME:    FITRES'])
    lines = p.read_text().splitlines()
    assert lines[0] == "# TABLE_NAME:    FITRES"
    assert lines[2] == "VARNAMES: CID x1"
    assert lines[3] == "SN: 1 0.50"

    write_varnames_table(p, ['CID', 'x1'], cols, fmt, style='csv')
    lines = p.read_text().splitlines()
    assert lines[0] == "CID,x1" and lines[1] == "1,0.50"

    write_varnames_table(p, ['CID', 'x1'], cols, fmt, style='col')
    assert p.read_text().splitlines()[0] == "1 0.50"

    p2 = tmp_path / "none.TEXT"
    write_varnames_table(p2, ['CID', 'x1'], cols, fmt, style='none')
    assert not p2.exists()


def test_write_varnames_table_formats_unlisted_columns_as_snana_does(tmp_path):
    """A caller need only pin the formats it wants; the rest print as SNANA's
    text writer prints them (sntools_output_text.c:445)."""
    from jax_snlc_core.snana_files import write_varnames_table

    cols = {'CID': np.array([7, 8]), 'FIELD': np.array(['X1', 'C3'], dtype=object),
            'x1': np.array([0.5, -1.25]), 'mB': np.array([23.4, 100.0])}
    p = tmp_path / "t.TEXT"

    write_varnames_table(p, list(cols), cols, {'CID': '%d'})

    assert p.read_text().splitlines()[1:] == ["SN: 7 X1 0.50000 23.40000",
                                              "SN: 8 C3 -1.25000e+00 100"]


def test_write_fitres_photoz_columns(tmp_path):
    """A photo-z fit writes OPT_PHOTOZ and keeps its covariances and prior,
    but zPHOT/zPHOTERR reach the table only as zHD (SKIPTABLE_FITPAR,
    snlc_fit.F90:19042; OPT_PHOTOZ always written, :18813)."""
    from jax_snlc_core.snana_files import write_varnames_table as _wvt
    from jax_snlc_fit.fitter import BatchFitResult
    from jax_snlc_fit.output import build_fitres_table

    class _Model:
        def compute_fitprob(self, chi2, ndof):
            return np.zeros(len(chi2))

        def get_fitres_param_columns(self, params, param_errors, redshifts,
                                     mu_ref, physical_only=False):
            return [(n, params[n], param_errors[n]) for n in params]

        def get_fitres_derived_columns(self, params):
            return []

    two = np.ones(2)
    results = BatchFitResult(
        snids=np.array([1, 2]),
        params={'x0': two, 'zPHOT': two}, param_errors={'x0': two, 'zPHOT': two},
        param_covariances=np.zeros((2, 2, 2)), cov_output_params=['x0', 'zPHOT'],
        chi_squared=two, ndof=two.astype(int), prior_chi2=two * 0,
        observation_mask=np.ones((2, 1), dtype=bool), hessian_cond=two,
        hessian_positive=np.ones_like(two, dtype=bool),
        edm=two,
        z_prior=(two, two),
        iter1_verdicts={})

    def varnames(**kw):
        p = tmp_path / "f.TEXT"
        table, fmt, comments = build_fitres_table(
            results, two, two * 36.0, model=_Model(), **kw)
        _wvt(p, table.varnames,
             {c: table.df[c].values for c in table.varnames}, fmt,
             style='key', header_comments=comments)
        return [ln.split()[1:] for ln in p.read_text().splitlines()
                if ln.startswith("VARNAMES:")][0]

    v = varnames(opt_photoz=2)
    assert 'zPHOT' not in v and 'zPHOTERR' not in v
    assert 'COV_x0_zPHOT' in v and 'zPRIOR' in v
    assert v.index('OPT_PHOTOZ') == v.index('x0') - 1

    results.minos_errors = {n: np.array([[.1, .3], [.1, np.nan]])
                           for n in results.param_names}
    results.minos_status = {n: np.array([[0, 0], [0, 2]])
                           for n in results.param_names}
    from jax_snlc_fit.uncertainties import reported_errors
    results.param_errors = {n: reported_errors(results.param_errors[n],
                            results.minos_errors[n], results.minos_status[n])
                            for n in results.param_names}
    # both sides profiled, then one side failed: what errmax_bad flags
    results.error_flags = {n: np.array([1, 2]) for n in results.param_names}
    table, fmt, comments = build_fitres_table(results, two, two*36., model=_Model())
    np.testing.assert_allclose(table.df['x0ERR'], [.2, 1.])
    np.testing.assert_array_equal(table.df['ERRFLAG_x0'], [1, 2])
    assert fmt['ERRFLAG_x0'] == '%d'
    assert fmt['x0_MINOS_STATUS_HI'] == '%d'
    assert any('2=Hessian' in comment for comment in comments)
    assert 'ERRFLAG_zPHOT' in varnames(opt_photoz=2)
    results.error_flags = {n: np.array([1, 6]) for n in results.param_names}
    table, _, _ = build_fitres_table(results, two, two*36., model=_Model())
    np.testing.assert_array_equal(table.df['ERRFLAG_x0'], [1, 6])
    np.testing.assert_allclose(table.df['x0ERR'], [.2, 1.])


def test_header_override_reader(tmp_path):
    """read_header_override_file: DOCANA skip, CID/GALID key detection,
    comma-list + dir merge, REDSHIFT_CMB alias, key-mix abort."""
    from jax_snlc_fit.loader import read_header_override_file

    cdir = tmp_path / "cid"
    cdir.mkdir()
    a = cdir / "a.dat"
    a.write_text("DOCUMENTATION:\n  PURPOSE: x\nDOCUMENTATION_END:\n\n"
                 "VARNAMES: CID REDSHIFT_HELIO VPEC\n"
                 "SN: 5 0.3 100.0\nSN: 7 0.4 -50.0\n")
    b = cdir / "b.dat"
    b.write_text("VARNAMES: SNID REDSHIFT_CMB\nROW: 9 0.25\n")
    match_col, values = read_header_override_file(
        [f"{a},{b}"])  # comma-separated list
    assert match_col == 'SNID'
    assert values['REDSHIFT_HELIO'] == {'5': 0.3, '7': 0.4}
    assert values['VPEC'] == {'5': 100.0, '7': -50.0}
    # REDSHIFT_CMB folds onto REDSHIFT_FINAL
    assert values['REDSHIFT_FINAL'] == {'9': 0.25}

    # dir scan (both files in cdir are CID-keyed)
    assert read_header_override_file(None, str(cdir))[0] == 'SNID'

    # GALID keying
    g = tmp_path / "g.dat"
    g.write_text("VARNAMES: GALID HOSTGAL_LOGMASS\nSN: 12345 10.5\n")
    assert read_header_override_file([str(g)])[0] == 'HOSTGAL_OBJID'

    # mixing CID and GALID keys aborts
    with pytest.raises(SNANAError):
        read_header_override_file([str(a), str(g)])


def test_header_override_apply(tmp_path):
    """apply_header_override: overwrite existing column, ignore absent,
    zHEL->zCMB recompute, and an overridden HOSTGAL_PHOTOZ becoming the
    redshift of an event with no spec-z (rd_override_zphot,
    sntools_data.c:2206-2218)."""
    from astropy.table import Table
    from jax_snlc_fit.loader import apply_header_override

    # RA=0,DEC=0 so the CMB dipole gives a known nonzero shift
    t = Table({'SNID': ['1', '2', '3'], 'RA': [0.0, 0.0, 0.0],
               'DEC': [0.0, 0.0, 0.0],
               'REDSHIFT_HELIO': [0.2, 0.31, 0.5],
               'REDSHIFT_HELIO_ERR': [0.01, 0.05, 0.01],
               'REDSHIFT_FINAL': [0.2, 0.31, 0.5],
               'REDSHIFT_FINAL_ERR': [0.01, 0.05, 0.01],
               'HOSTGAL_PHOTOZ': [0.0, 0.3, 0.0],
               'HOSTGAL_PHOTOZ_ERR': [0.0, 0.05, 0.0],
               'HOSTGAL_SPECZ': [0.2, -9.0, 0.5],
               'MASK_REDSHIFT_SOURCE': [1, 4, 2],
               'VPEC': [0.0, 0.0, 0.0]})
    from jax_snlc_core.cosmology import zhelio_to_zcmb

    # override REDSHIFT_HELIO on CID 1 (recompute zCMB) + an absent column
    ov = ('SNID', {'REDSHIFT_HELIO': {'1': 0.35}, 'NOSUCH': {'1': 1.0}})
    out = apply_header_override(t, ov)
    assert out is not t  # copy, original untouched
    assert out['REDSHIFT_HELIO'][0] == 0.35
    exp = float(zhelio_to_zcmb(np.array([0.35]), np.array([0.0]),
                                   np.array([0.0]))[0])
    assert np.isclose(out['REDSHIFT_FINAL'][0], exp)
    assert out['REDSHIFT_HELIO'][2] == 0.5  # untouched

    # HOSTGAL_PHOTOZ overridden on CID 2 (photo-z only) and CID 3 (spec-z)
    ov = ('SNID', {'HOSTGAL_PHOTOZ': {'2': 0.28, '3': 0.45}})
    out = apply_header_override(t, ov)
    assert np.isclose(out['REDSHIFT_HELIO'][1], 0.28)
    assert np.isclose(out['REDSHIFT_HELIO_ERR'][1], 0.05)
    exp = float(zhelio_to_zcmb(np.array([0.28]), np.array([0.0]),
                                   np.array([0.0]))[0])
    assert np.isclose(out['REDSHIFT_FINAL'][1], exp)
    assert out['HOSTGAL_PHOTOZ'][2] == 0.45 and out['REDSHIFT_HELIO'][2] == 0.5


def _quantile_head(n_hosts=2):
    from astropy.table import Table
    z = np.full((3, 20), -9.0)
    z[:, :3] = [[0.1, 0.2, 0.3]] * 3
    pct = np.full((3, 20), -9.0)
    pct[:, :3] = [0.0, 50.0, 100.0]
    cols = {'SNID': ['1', '2', '3'], 'HOSTGAL_OBJID': [11, 22, 22],
            'HOSTGAL_PHOTOZ': [0.2, 0.2, 0.2], 'HOSTGAL_PHOTOZ_ERR': [0.1, 0.1, 0.1],
            'HOSTGALz_NBIN_QUANTILE_ZPHOT': [3, 3, 3],
            'HOSTGALz_QUANTILE_ZPHOT': z.copy(), 'HOSTGALz_QUANTILE_PERCENT': pct.copy(),
            'HOSTGAL2_OBJID': [33, 0, 0],
            'HOSTGAL2_PHOTOZ': [0.4, -9.0, -9.0], 'HOSTGAL2_PHOTOZ_ERR': [0.1, -9.0, -9.0],
            'HOSTGAL2z_NBIN_QUANTILE_ZPHOT': [0, 0, 0],
            'HOSTGAL2z_QUANTILE_ZPHOT': np.full((3, 20), -9.0),
            'HOSTGAL2z_QUANTILE_PERCENT': np.full((3, 20), -9.0),
            'SIM_REDSHIFT_HELIO': [0.22, 0.2, 0.2], 'SIM_REDSHIFT_HOST': [0.2, 0.2, 0.2]}
    return Table(cols)


def test_header_override_quantiles_implicit(tmp_path):
    """QZPHOTnn columns: uniform percentiles over the values >= 0, every
    event's reported host replaced (one absent from the file keeps none),
    a sim's zSN/zHOST rescale, and HOSTGAL_PHOTOZ[_ERR] from the grid's mean
    and width (rd_override_qzphot_implicit, rd_override_zphot); the old
    HOSTGAL_ZPHOT_Q* spelling aborts."""
    from jax_snlc_fit.loader import (apply_header_override,
                                          read_header_override_file)

    f = tmp_path / "qz.dat"
    f.write_text("VARNAMES: CID QZPHOT00 QZPHOT01 QZPHOT02 QZPHOT03\n"
                 "SN: 1 0.10 0.15 0.20 0.30\n"
                 "SN: 2 0.30 0.40 0.50 -9\n")
    out = apply_header_override(_quantile_head(), read_header_override_file([str(f)]))
    nbin = np.asarray(out['HOSTGALz_NBIN_QUANTILE_ZPHOT'])
    z = np.asarray(out['HOSTGALz_QUANTILE_ZPHOT'])
    pct = np.asarray(out['HOSTGALz_QUANTILE_PERCENT'])
    assert nbin.tolist() == [4, 3, 0]
    np.testing.assert_allclose(z[0, :4], 1.1 * np.array([0.10, 0.15, 0.20, 0.30]))
    np.testing.assert_allclose(pct[0, :5], [0, 100 / 3, 200 / 3, 100, -9])
    np.testing.assert_allclose(z[1, :4], [0.30, 0.40, 0.50, -9])
    np.testing.assert_allclose(pct[1, :4], [0, 50, 100, -9])
    assert (z[2] == -9).all() and (pct[2] == -9).all()
    # the grid: weights 0, 50, 50 over the three of CID 2
    np.testing.assert_allclose(out['HOSTGAL_PHOTOZ'][1], 0.45)
    np.testing.assert_allclose(out['HOSTGAL_PHOTOZ_ERR'][1], 0.05)
    assert out['HOSTGAL_PHOTOZ'][2] == -9.0 and out['HOSTGAL_PHOTOZ_ERR'][2] == -9.0

    # a point photo-z column in the override set switches the grid off
    g = tmp_path / "zp.dat"
    g.write_text("VARNAMES: CID HOSTGAL_PHOTOZ HOSTGAL_PHOTOZ_ERR\nSN: 1 0.25 0.02\n")
    out = apply_header_override(_quantile_head(),
                                read_header_override_file([f"{f},{g}"]))
    assert list(out['HOSTGAL_PHOTOZ']) == [0.25, 0.2, 0.2]

    old = tmp_path / "old.dat"
    old.write_text("VARNAMES: CID HOSTGAL_ZPHOT_Q000 HOSTGAL_ZPHOT_Q100\nSN: 1 0.1 0.3\n")
    with pytest.raises(SNANAError, match="obsolete OVERRIDE columns .* QZPHOT00 QZPHOT01"):
        read_header_override_file([str(old)])


def test_header_override_logmass_grid_needs_its_errors_and_increasing_z(tmp_path):
    """A LOGMASS(z) grid in the override is refused without its error grid,
    which SNANA would read as the -9 fill, and the fit refuses a grid whose
    redshifts do not increase (init_spline, sntools_spline.c)."""
    from jax_snlc_fit.loader import (_set_logmass, apply_header_override,
                                          read_header_override_file)

    head = _quantile_head()
    head['HOSTGAL_LOGMASS'] = [10.0, 10.0, 10.0]
    head['HOSTGAL_LOGMASS_ERR'] = [0.1, 0.1, 0.1]
    no_err = tmp_path / "lm.dat"
    no_err.write_text("VARNAMES: CID HOSTGALz_LOGMASS_ZGRID HOSTGALz_LOGMASS_VALGRID\n"
                      "SN: 1 0.1 9.5\nSN: 1 0.3 10.0\n")
    with pytest.raises(SNANAError, match="without HOSTGALz_LOGMASS_ERRGRID"):
        apply_header_override(head, read_header_override_file([str(no_err)]))

    flat = tmp_path / "flat.dat"
    flat.write_text("VARNAMES: CID HOSTGALz_LOGMASS_ZGRID HOSTGALz_LOGMASS_VALGRID "
                    "HOSTGALz_LOGMASS_ERRGRID\n"
                    "SN: 1 0.3 9.5 0.1\nSN: 1 0.1 10.0 0.1\n")
    out = apply_header_override(head, read_header_override_file([str(flat)]))
    host = {c: np.array(out[c]) for c in out.colnames if c.startswith('HOSTGAL')}
    with pytest.raises(SNANAError, match="must increase strictly"):
        _set_logmass(host, np.full(3, 0.2), np.full(3, 0.01), False)


def test_header_override_quantiles_by_galid(tmp_path):
    """One row per quantile keyed by GALID, `*` repeating the key: the
    reported host's rows land on every event sharing that GALID (rescaled
    by the sim), a second host's on its slot unscaled, events absent from
    the file keep their quantiles, and every host of an overridden event
    takes the grid's mean and width, an absent second host's grid being
    empty (rd_override_hostgal2z, rd_override_zphot)."""
    from jax_snlc_fit.loader import (apply_header_override,
                                          read_header_override_file)

    f = tmp_path / "galid.dat"
    f.write_text("VARNAMES: GALID HOSTGALz_QUANTILE_ZPHOT HOSTGALz_QUANTILE_PERCENT\n"
                 "GAL: 11 0.10 0\nGAL: * 0.20 50\nGAL: * 0.30 100\n"
                 "GAL: 33 0.40 0\nGAL: 33 0.44 100\n")
    match_col, values = read_header_override_file([str(f)])
    assert match_col == 'HOSTGAL_OBJID'
    assert values['HOSTGALz_QUANTILE_ZPHOT'] == {'11': [0.1, 0.2, 0.3], '33': [0.4, 0.44]}

    out = apply_header_override(_quantile_head(), (match_col, values))
    assert list(out['HOSTGALz_NBIN_QUANTILE_ZPHOT']) == [3, 3, 3]
    np.testing.assert_allclose(out['HOSTGALz_QUANTILE_ZPHOT'][0, :3], [0.11, 0.22, 0.33])
    np.testing.assert_allclose(out['HOSTGALz_QUANTILE_ZPHOT'][1, :3], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(out['HOSTGAL_PHOTOZ'][0], 0.275)
    assert out['HOSTGAL_PHOTOZ'][1] == 0.2       # GALID 22 is not in the file
    assert list(out['HOSTGAL2z_NBIN_QUANTILE_ZPHOT']) == [2, 0, 0]
    np.testing.assert_allclose(out['HOSTGAL2z_QUANTILE_ZPHOT'][0, :2], [0.40, 0.44])
    np.testing.assert_allclose(out['HOSTGAL2_PHOTOZ'][0], 0.44)
    assert out['HOSTGAL2_PHOTOZ'][1] == -9.0


def test_a_cutwin_must_name_a_cut_quantity_with_a_numeric_window(tmp_path):
    """SNANA's namelist refuses an unknown variable at read time; so does
    this, rather than at cut time after the light curves are loaded."""
    nml = tmp_path / "fit.nml"
    nml.write_text("&SNLCINP\n  CUTWIN_SNRMX = 5, 100\n&END\n")
    with pytest.raises(SNANAError, match="CUTWIN_SNRMX names no quantity"):
        build(read_keys(nml))
    nml.write_text("&SNLCINP\n  CUTWIN_SNRMAX = 'g', 5\n&END\n")
    with pytest.raises(SNANAError, match="numeric window"):
        build(read_keys(nml))
    nml.write_text("&SNLCINP\n  CUTWIN_SNRMAX_G = 5\n  CUTWIN_TREST = -15, 45\n&END\n")
    cfg = build(read_keys(nml))
    assert cfg.selection.cutwin["snrmax_g"][0] == 5.0
    assert cfg.selection.epoch_cutwin["trest"] == (-15.0, 45.0)


def test_nml_value_shapes(tmp_path):
    """Quoted lists split on commas; commas inside quotes don't split."""
    nml = tmp_path / "fit.nml"
    nml.write_text(
        "&SNLCINP\n"
        "  SNFIELD_LIST = 'C3', 'E1', 'S1'\n"
        "  CUTWIN_REDSHIFT = 0.05, 0.40\n"
        "  HOSTLIB_DZTOL = '0.002,0.04'\n"
        "  TEXTFILE_PREFIX = 'OUT'\n"
        "&END\n"
    )
    raw = read_keys(nml)
    assert raw["SNFIELD_LIST"] == ["C3", "E1", "S1"]
    assert raw["CUTWIN_REDSHIFT"] == ["0.05", "0.40"]
    assert raw["HOSTLIB_DZTOL"] == "0.002,0.04"
    assert raw["TEXTFILE_PREFIX"] == "OUT"


# ---------------------------------------------------------------------------
# INIVAL_* / INISTP_*
# ---------------------------------------------------------------------------

_SALT = ["PKMJD", "x1", "c", "x0"]
_BAYESN = ["PKMJD", "THETA", "Ds", "AV"]


@pytest.fixture(scope="module")
def salt_model(data_root):
    from jax_snlc_core.models.salt import SALTModel
    return SALTModel.load(str(data_root / "models" / "SALT3.K21"), {})


@pytest.fixture(scope="module")
def salt_aliases(salt_model):
    from jax_snlc_fit.parameters import snana_parameter_aliases
    return snana_parameter_aliases(salt_model)


@pytest.fixture(scope="module")
def bayesn_model(data_root):
    from jax_snlc_core.models.bayesn import BayeSNModel
    return BayeSNModel.load(str(data_root / "models" / "BAYESN.W22"), {})


@pytest.fixture(scope="module")
def bayesn_aliases(bayesn_model):
    from jax_snlc_fit.parameters import snana_parameter_aliases
    return snana_parameter_aliases(bayesn_model)


def test_inival_family_splits_settings_from_parameters(tmp_path):
    """Named settings are claimed first; the rest are carried as suffixes."""
    nml = tmp_path / "fit.nml"
    nml.write_text(
        "&FITINP\n"
        "  INIVAL_x1 = 0.5\n"
        "  INISTP_x1 = 0\n"
        "  INIVAL_GRIDSEARCH_DPEAKMJD = -4.0, 4.0, 2.0\n"
        "&END\n"
    )
    cfg = build(read_keys(nml))
    # keys upper-case on read; resolution against the model is case-insensitive
    assert cfg.model.inival == {"X1": 0.5}
    assert cfg.model.inistp == {"X1": 0.0}
    assert cfg.model.inival_gridsearch == {"DPEAKMJD": (-4.0, 4.0, 2.0)}


@pytest.mark.parametrize("suffix, expected", [
    ("x1", "x1"), ("X1", "x1"),          # the model's own name, any case
    ("SHAPE", "x1"), ("COLOR", "c"),     # SNANA's SALT-era spellings
    ("PEAKMJD", "PKMJD"),                # universal
    ("DLMAG", "x0"), ("X0", "x0"),       # one slot, named x0 for SALT
    ("DPEAKMJD", "PKMJD"),               # grid axis is t0 offsets
])
def test_parameter_key_resolution(suffix, expected, salt_aliases):
    from jax_snlc_fit.parameters import resolve_parameter_key
    assert resolve_parameter_key(suffix, _SALT, salt_aliases) == expected


@pytest.mark.parametrize("suffix, expected", [
    ("SHAPE", "THETA"),
    ("COLOR", "AV"),        # IPAR_COLOR = IPAR_AV (snlc_fit.F90:1135)
    ("DLMAG", "Ds"),        # BayeSN fits distance directly
])
def test_generic_names_resolve_per_model(suffix, expected, bayesn_aliases):
    """SNANA's generic names address one slot whatever model is loaded, so they
    land on the parameter that model declares them on."""
    from jax_snlc_fit.parameters import resolve_parameter_key
    assert resolve_parameter_key(suffix, _BAYESN, bayesn_aliases) == expected


def test_parameters_resolve_by_name_without_aliases():
    """The generic names are backwards compatibility, not the interface. A
    model declaring none does not answer to them, and says so."""
    from jax_snlc_fit.parameters import resolve_parameter_key
    names = ["PKMJD", "brightness", "stretch"]
    assert resolve_parameter_key("stretch", names, {}) == "stretch"
    assert resolve_parameter_key("PEAKMJD", names, {}) == "PKMJD"
    with pytest.raises(SNANAError, match="not a parameter of this model"):
        resolve_parameter_key("SHAPE", names, {})
    assert resolve_parameter_key(
        "SHAPE", names, {"SHAPE": "stretch"}) == "stretch"


@pytest.mark.parametrize("suffix, phrase", [
    ("AVRV", "AV/RV"),          # a parameterisation switch, not a parameter
    ("SHAPE2", "second light-curve shape"),
])
def test_unsupported_keys_say_what_they_are(suffix, phrase):
    """Not 'unknown parameter' — these name features jax lacks."""
    from jax_snlc_fit.parameters import resolve_parameter_key
    with pytest.raises(SNANAError) as exc:
        resolve_parameter_key(suffix, _SALT, {})
    assert phrase in str(exc.value)


def test_inival_outside_bounds_aborts():
    """Otherwise the optimiser silently pulls it in, or pins it out of range."""
    from jax_snlc_fit.parameters import validate_inival_bounds

    declared = {"x1": {"bounds": (-5.0, 5.0), "scale": 1.0}}
    validate_inival_bounds(FitConfig(model=Model(inival={"x1": 2.0})), declared)   # in range
    with pytest.raises(SNANAError, match="outside its bounds"):
        validate_inival_bounds(FitConfig(model=Model(inival={"x1": 99.0})), declared)


def test_relative_bounds_are_not_range_checked():
    """x0's bounds move with its value, so it cannot be out of range."""
    from jax_snlc_fit.parameters import validate_inival_bounds

    declared = {"x0": {"bounds_relative_to_init": (0.01, 100.0), "scale": 1e-5}}
    validate_inival_bounds(FitConfig(model=Model(inival={"x0": 1e-3})), declared)


def test_reference_cosmology_keys_reach_the_distance_modulus(tmp_path):
    """H0_REF/OMAT_REF/OLAM_REF/W0_REF/WA_REF set the fitter's one cosmology.

    Defaults are SNANA's (snana.F90:220-224); jax previously hardcoded four
    different cosmologies, one of them H0=67.4 against SNANA's 70.
    """

    assert FitConfig().fit.h0_ref == (70.0, 7.0)
    assert (FitConfig().fit.omat_ref[0] == 0.315
            and FitConfig().fit.olam_ref[0] == 0.685)
    assert FitConfig().fit.w0_ref[0] == -1.0 and FitConfig().fit.wa_ref[0] == 0.0

    nml = tmp_path / "fit.nml"
    nml.write_text("&SNLCINP\n  H0_REF = 60.0, 5.0\n  OMAT_REF = 0.30\n"
                   "  W0_REF = -0.9\n&END\n")
    cfg = build(read_keys(nml))
    assert cfg.fit.h0_ref == (60.0, 5.0)
    # a single value keeps the default uncertainty
    assert cfg.fit.omat_ref == (0.30, 0.03) and cfg.fit.w0_ref == (-0.9, 0.1)

    # H0 alone rescales D_L by exactly 70/60, so mu shifts analytically at every z
    h0_only = tmp_path / "h0.nml"
    h0_only.write_text("&SNLCINP\n  H0_REF = 60.0\n&END\n")
    z = np.array([0.05, 0.3, 1.0])
    shifted = np.asarray(build(read_keys(h0_only)).fit.distance_modulus(z))
    default = np.asarray(FitConfig().fit.distance_modulus(z))
    assert np.allclose(shifted - default, 5.0 * np.log10(70.0 / 60.0), atol=1e-9)

    # Omega_m and w0 must also move it, or they are inert
    assert not np.allclose(np.asarray(cfg.fit.distance_modulus(z)), shifted)

def test_mwebv_force_turns_on_above_snanas_threshold(tmp_path):
    """SNANA forces every event's E(B-V) only where MWEBV_FORCE > -0.001
    (snana.F90:17209), so its default, -9, and a value just below the
    threshold leave each event's own E(B-V); zero forces it."""

    forced = {}
    for value in ("-9", "-0.005", "0", "0.02"):
        nml = tmp_path / "fit.nml"
        nml.write_text(f"&SNLCINP\n  MWEBV_FORCE = {value}\n&END\n")
        forced[value] = build(read_keys(nml)).data.mwebv_force
    assert forced == {"-9": None, "-0.005": None, "0": 0.0, "0.02": 0.02}


def test_an_array_value_may_be_blank_or_comma_separated(tmp_path):
    """A Fortran namelist separates an array's values by blanks or commas, as
    SNANA reads them, so both spellings of PRIOR_DMU_RANGE give one window."""

    windows = []
    for spelling in ("-0.5 0.5", "-0.5, 0.5"):
        nml = tmp_path / "fit.nml"
        nml.write_text("&SNLCINP\n  DOFIT_PHOTOZ = T\n  OPT_PHOTOZ = 1\n"
                       f"  PRIOR_DMU_RANGE = {spelling}\n&END\n")
        windows.append(build(read_keys(nml)).photoz.prior_dmu_range)
    assert windows == [(-0.5, 0.5), (-0.5, 0.5)]



def test_salt2_standardisation_keys(tmp_path):
    """SALT2ALPHA/SALT2BETA are SALT's declared keys: the parser types them
    into model_key_values and build_model sets them on the instance."""

    nml = tmp_path / "fit.nml"
    nml.write_text("&FITINP\n  SALT2ALPHA = 0.15\n  SALT2BETA = 3.1\n&END\n")
    cfg = build(read_keys(nml))
    assert cfg.model._model_key_values == {'SALT2ALPHA': 0.15, 'SALT2BETA': 3.1}


def test_usesim_keys_resolve_to_truth_columns():
    """USESIM_<p> is INIVAL=<sim truth> + INISTP=0 (snlc_fit.F90:8341-8353), so
    it resolves to a parameter and the truth column that parameter declares.
    USESIM_INIVAL seeds every such parameter but pins none (:8305)."""
    from jax_snlc_fit.parameters import resolve_usesim_columns

    declared = {
        'PKMJD': {'bounds': (-1.0, 1.0), 'scale': 1.0, 'sim_truth': 'SIM_PEAKMJD'},
        'x0': {'bounds': (0.0, 1.0), 'scale': 1e-5, 'sim_truth': 'SIM_SALT2x0'},
        'x1': {'bounds': (-5.0, 5.0), 'scale': 1.0, 'sim_truth': 'SIM_SALT2x1',
               'snana_keys': ('SHAPE',)},
        'c': {'bounds': (-1.0, 1.0), 'scale': 1.0, 'sim_truth': 'SIM_SALT2c',
              'snana_keys': ('COLOR',)},
    }
    names = list(declared)
    aliases = {a: n for n, s in declared.items()
               for a in s.get('snana_keys', ())}

    cfg = FitConfig(model=Model(usesim_params=['SHAPE', 'COLOR']))
    cols, pinned = resolve_usesim_columns(cfg, declared, names, aliases)
    assert cols == {'x1': 'SIM_SALT2x1', 'c': 'SIM_SALT2c'}
    assert sorted(pinned) == ['c', 'x1']

    # USESIM_INIVAL covers everything with a truth column, pinning nothing
    cfg = FitConfig(model=Model(usesim_inival=True))
    cols, pinned = resolve_usesim_columns(cfg, declared, names, aliases)
    assert set(cols) == set(names) and pinned == []

    # a parameter with no truth column cannot be fitted against truth
    cfg = FitConfig(model=Model(usesim_params=['x1']))
    with pytest.raises(SNANAError, match="no.*simulated-truth column"):
        resolve_usesim_columns(cfg, {'x1': {'bounds': (0.0, 1.0), 'scale': 1.0}},
                               ['x1'], {})


def test_usesim_flags_parse(tmp_path):
    """USESIM_SNIA/NONIA default true (snana.F90:5723); TRUEFLUX false."""

    assert (FitConfig().selection.usesim_snia
            and FitConfig().selection.usesim_nonia)
    assert (not FitConfig().data.usesim_trueflux
            and not FitConfig().model.usesim_inival)

    nml = tmp_path / "fit.nml"
    nml.write_text("&SNLCINP\n  USESIM_SNIA = F\n  USESIM_TRUEFLUX = T\n&END\n"
                   "&FITINP\n  USESIM_PEAKMJD = T\n  USESIM_DLMAG = T\n&END\n")
    cfg = build(read_keys(nml))
    assert cfg.selection.usesim_snia is False and cfg.data.usesim_trueflux is True
    assert sorted(cfg.model.usesim_params) == ['DLMAG', 'PEAKMJD']


def test_fitwin_postfit_windows_resolve(tmp_path, salt_model, salt_aliases):
    """FITWIN_<quantity> cuts the finished fit. The base name resolves like any
    generic SNANA parameter name, and the _ERR / _ERRz1 form rides along, so
    FITWIN_SHAPE_ERR becomes x1_ERR under SALT."""
    from jax_snlc_fit.parameters import resolve_fitwin_windows

    nml = tmp_path / "fit.nml"
    nml.write_text("&FITINP\n"
                   "  FITWIN_SHAPE = -0.5, 0.5\n"
                   "  FITWIN_SHAPE_ERR = 0.0, 0.15\n"
                   "  FITWIN_COLOR_ERRz1 = 0.0, 0.05\n"
                   "  FITWIN_PROB = 0.01, 1.0\n"
                   "  FITWIN_TREST = -15, 45\n"          # epoch cut, not post-fit
                   "  FITWIN_PROB_ITER1 = 0.0, 1.0\n"    # mid-fit, not post-fit
                   "&END\n")
    cfg = build(read_keys(nml))
    assert set(cfg.selection.fitwin) == {'SHAPE', 'SHAPE_ERR', 'COLOR_ERRZ1', 'PROB'}
    assert cfg.fit.fitwin_trest == (-15.0, 45.0)

    resolved = resolve_fitwin_windows(cfg, salt_model, _SALT, salt_aliases)
    assert resolved == {'x1': (-0.5, 0.5), 'x1_ERR': (0.0, 0.15),
                        'c_ERRZ1': (0.0, 0.05), 'PROB': (0.01, 1.0)}


def _fitwin_config(tmp_path, *keys):
    nml = tmp_path / "fitwin.nml"
    nml.write_text("&FITINP\n" + "".join(f"  {k} = 0, 1\n" for k in keys) + "&END\n")
    return build(read_keys(nml))


def test_fitwin_suffixes_are_checked_before_the_fit(tmp_path, salt_model,
                                                     salt_aliases, bayesn_model,
                                                     bayesn_aliases):
    """A FITWIN_ suffix must name a column the model writes, a derived
    quantity, or their _ERR / _ERRz1 forms; anything else aborts at set-up,
    naming the key, rather than after the fit has run. A column no parameter
    owns (SALT's mB) resolves by name, and a generic spelling resolves to the
    column its parameter is written as (BayeSN's DLMAG is Ds, written as
    MU_LCFIT)."""
    from jax_snlc_fit.parameters import resolve_fitwin_windows

    with pytest.raises(SNANAError, match="FITWIN_SHAPPE names no fitted quantity"):
        resolve_fitwin_windows(_fitwin_config(tmp_path, "FITWIN_SHAPPE"),
                               salt_model, _SALT, salt_aliases)
    resolved = resolve_fitwin_windows(_fitwin_config(tmp_path, "FITWIN_mB_ERR"),
                                      salt_model, _SALT, salt_aliases)
    assert resolved == {"mB_ERR": (0.0, 1.0)}
    resolved = resolve_fitwin_windows(_fitwin_config(tmp_path, "FITWIN_DLMAG"),
                                      bayesn_model, _BAYESN, bayesn_aliases)
    assert resolved == {"MU_LCFIT": (0.0, 1.0)}


@pytest.mark.parametrize('cid_list', ["'5', '9'", '5 9'])
def test_selection_ignore_lists_and_caps(tmp_path, cid_list):
    """The rejection lists, event caps and reproducible simulation prescale."""

    assert FitConfig().selection.mxevt_process == 999888777
    assert FitConfig().selection.sim_prescale == 1.0

    nml = tmp_path / "fit.nml"
    nml.write_text("&SNLCINP\n"
                   "  SNCID_IGNORE = 3, 7, 11\n"
                   "  SNTYPE_IGNORE = 120, 121\n"
                   f"  SNCCID_LIST = {cid_list}\n"
                   "  MXEVT_PROCESS = 50\n"
                   "  SIM_PRESCALE = 3.0\n"
                   "&END\n")
    cfg = build(read_keys(nml))
    assert cfg.selection.sncid_ignore == [3, 7, 11]
    assert cfg.selection.sntype_ignore == [120, 121]
    assert cfg.selection.snccid_list == ['5', '9']
    assert cfg.selection.mxevt_process == 50 and cfg.selection.sim_prescale == 3.0

    # the prescale's effect on the surviving set is asserted against
    # apply_snana_selection in tests/unit/test_lightcurve_batch.py


def test_flag_use_same_events_overrides_opt_sncid_list(tmp_path):
    """FLAG_USE_SAME_EVENTS is the newer spelling of OPT_SNCID_LIST and wins
    when positive (snana.F90:6290)."""

    nml = tmp_path / "fit.nml"
    nml.write_text("&SNLCINP\n  OPT_SNCID_LIST = 3\n  FLAG_USE_SAME_EVENTS = 1\n&END\n")
    assert build(read_keys(nml)).selection.opt_sncid_list == 1
    nml.write_text("&SNLCINP\n  OPT_SNCID_LIST = 3\n  FLAG_USE_SAME_EVENTS = 0\n&END\n")
    assert build(read_keys(nml)).selection.opt_sncid_list == 3


def test_varlist_pdf_is_refused_with_the_marginalisation_family(tmp_path):
    """VARLIST_PDF picks which parameters MINUIT's marginalisation grid runs
    over (snana.F90:14370); its default 'ALL' asks for nothing, any other
    value is refused with the rest of that family."""
    from jax_snlc_fit.config_parser import _check_rejected_keys, read_keys

    nml = tmp_path / "fit.nml"
    nml.write_text("&FITINP\n  VARLIST_PDF = 'ALL'\n&END\n")
    _check_rejected_keys(read_keys(nml))
    nml.write_text("&FITINP\n  VARLIST_PDF = 'x1,c'\n&END\n")
    with pytest.raises(SNANAError, match="VARLIST_PDF"):
        _check_rejected_keys(read_keys(nml))


def test_redshift_shift_systematics():
    """SET_zSHIFT (snana.F90:17848): REDSHIFT_FINAL_SHIFT moves every redshift,
    while a host shift applies only when that host measurement IS the SN
    redshift — shifting an unused host-z would corrupt the reported column
    without changing the fit."""
    from jax_snlc_fit.loader import apply_redshift_shifts

    z = np.array([0.2, 0.3, 0.4])
    extras = {'REDSHIFT_HELIO': z.copy()}
    # SNLC_ZCMB is incremented like the others rather than re-derived from the
    # shifted helio value; offset here so the two are distinguishable.
    zcmb = z + 0.001
    out, _, _, zc = apply_redshift_shifts(
        {'final': 0.01}, z, extras, None, None, None, zcmb)
    assert np.allclose(out, [0.21, 0.31, 0.41])
    assert np.allclose(extras['REDSHIFT_HELIO'], [0.21, 0.31, 0.41])
    assert np.allclose(zc, [0.211, 0.311, 0.411])

    # host photo-z equal to the SN redshift: shift applies to both
    photoz = np.array([0.2, 0.3, 0.4])
    out, ph, _, _ = apply_redshift_shifts(
        {'final': 0.0, 'zphot': 0.02}, z.copy(), {}, photoz, None, None,
        z.copy())
    assert np.allclose(out, [0.22, 0.32, 0.42])
    assert np.allclose(np.asarray(ph), [0.22, 0.32, 0.42])

    # host photo-z that is not the SN redshift: nothing moves
    other = np.array([0.9, 0.9, 0.9])
    out, ph, _, zc = apply_redshift_shifts(
        {'final': 0.0, 'zphot': 0.02}, z.copy(), {}, other, None, None,
        z.copy())
    assert np.allclose(out, z) and np.allclose(np.asarray(ph), other)
    assert np.allclose(zc, z)


def test_magobs_shift_zp_parses_as_a_filter_string(tmp_path):
    """MAGOBS_SHIFT_ZP is a SNANA filter string naming the SHORT band, while
    PHOT carries the survey-qualified name — the loader matches on the short
    form, and getting that wrong makes the key silently inert."""

    nml = tmp_path / "fit.nml"
    nml.write_text("&SNLCINP\n  MAGOBS_SHIFT_ZP = 'r 0.1 i -0.05'\n&END\n")
    cfg = build(read_keys(nml))
    assert cfg.data.magobs_shift_zp == {'r': 0.1, 'i': -0.05}


def test_photflag_bitlist_reject(tmp_path):
    """PHOTFLAG_BITLIST_REJECT is the bit-list spelling of PHOTFLAG_MSKREJ:
    SNANA sets each listed bit and REPLACES the mask (snana.f:60664), with the
    list terminating at a negative value."""

    nml = tmp_path / "fit.nml"
    nml.write_text("&SNLCINP\n  PHOTFLAG_BITLIST_REJECT = 1, 3, 5, -1\n&END\n")
    assert build(read_keys(nml)).selection.photflag_mskrej == (2 | 8 | 32,)

    # entries past the sentinel are not part of the mask
    nml.write_text("&SNLCINP\n  PHOTFLAG_BITLIST_REJECT = 2, -1, 9\n&END\n")
    assert build(read_keys(nml)).selection.photflag_mskrej == (4,)


def test_use_model_magerr_and_shift_setpkmjd(tmp_path):
    """USE_MODEL_MAGERR = F zeroes the model error and forces OPT_COVAR_FLUX = 0
    (snlc_fit.f:464029). SHIFT_SETPKMJD displaces the peak-MJD estimate itself,
    so the fit and the reported PKMJDINI move together rather than diverging."""

    assert (FitConfig().model.use_model_magerr
            and FitConfig().data.shift_setpkmjd == 0.0)

    nml = tmp_path / "fit.nml"
    nml.write_text("&SNLCINP\n  USE_MODEL_MAGERR = F\n  SHIFT_SETPKMJD = 2.0\n&END\n")
    cfg = build(read_keys(nml))
    assert cfg.model.fudge_modelerr_scale == 0.0
    assert cfg.fit.use_offdiag_covariance is False
    assert cfg.data.shift_setpkmjd == 2.0

    # the default leaves both mechanisms alone
    nml.write_text("&SNLCINP\n&END\n")
    cfg = build(read_keys(nml))
    assert cfg.model.fudge_modelerr_scale == 1.0 and cfg.fit.use_offdiag_covariance is True


def test_bang_inside_quotes_is_not_a_comment(tmp_path):
    """A '!' opens a namelist comment, but not inside a quoted value. SNANA's
    own PRIVATE_CUTWIN_STRING veto spelling is 'VARNAME != VALUE', so stripping
    on the bare character truncates a real input."""
    nml = tmp_path / "fit.nml"
    nml.write_text(
        "&SNLCINP\n"
        "  PRIVATE_CUTWIN_STRING(1) = 'PRIVATE(FOO) 0.0 1.0'\n"
        "  PRIVATE_CUTWIN_STRING(2) = 'PRIVATE(BAR) != 3'\n"
        "  TEXTFILE_PREFIX = 'a!b'\n"
        "  KCOR_FILE = '/k.fits'   ! a genuine trailing comment\n"
        "&END\n")
    raw = read_keys(nml)
    assert raw["TEXTFILE_PREFIX"] == "a!b"
    assert raw["KCOR_FILE"] == "/k.fits"
    assert raw["PRIVATE_CUTWIN_STRING(2)"] == "PRIVATE(BAR) != 3"


def test_private_cutwin_window_and_veto_spellings(tmp_path):
    """SNANA accepts 'VARNAME LO HI' and 'VARNAME != VALUE' (snana.F90:5085),
    reading only the first three words of each array element."""
    nml = tmp_path / "fit.nml"
    nml.write_text(
        "&SNLCINP\n"
        "  VERSION_PHOTOMETRY = 'V'\n"
        "  KCOR_FILE = 'k.fits'\n"
        "  PRIVATE_CUTWIN_STRING(1) = 'PRIVATE(FOO) 0.0 1.0'\n"
        "  PRIVATE_CUTWIN_STRING(2) = 'PRIVATE(BAR) != 3'\n"
        "&END\n"
        "&FITINP\n  FITMODEL_NAME = 'M'\n  FILTLIST_FIT = 'griz'\n&END\n")
    cfg = build(read_keys(nml))
    assert cfg.selection.private_cutwin == {'PRIVATE(FOO)': (0.0, 1.0)}
    assert cfg.selection.private_veto == {'PRIVATE(BAR)': 3.0}


def test_abort_policy_keys_are_live(tmp_path):
    """The four ABORT_ON_* keys jax used to refuse now reach the config."""
    d = FitConfig()
    assert (d.fit.abort_on_badqzphot, d.fit.abort_on_trestcut,
            d.fit.abort_on_nopkmjd, d.data.restore_wrong_vpec,
            d.fit.minos_errmax_bad) == (True, True, True, False, True)

    nml = tmp_path / "fit.nml"
    nml.write_text(
        "&SNLCINP\n  VERSION_PHOTOMETRY = 'V'\n  KCOR_FILE = 'k.fits'\n"
        "  ABORT_ON_BADQZPHOT = F\n  ABORT_ON_TRESTCUT = F\n"
        "  ABORT_ON_NOPKMJD = F\n  RESTORE_WRONG_VPEC = T\n"
        "  MINOS_ERRMAX_BAD = F\n&END\n"
        "&FITINP\n  FITMODEL_NAME = 'M'\n  FILTLIST_FIT = 'griz'\n&END\n")
    cfg = build(read_keys(nml))
    assert not cfg.fit.abort_on_badqzphot and not cfg.fit.abort_on_trestcut
    assert not cfg.fit.abort_on_nopkmjd and cfg.data.restore_wrong_vpec
    assert not cfg.fit.minos_errmax_bad


def test_scale_covar_only_switches_the_model_error(tmp_path):
    """SNANA declares SCALE_COVAR a covariance scale but never reads it as one;
    its single live effect is USE_MODEL_MAGERR (snlc_fit.F90:7161)."""
    def cfg(value):
        nml = tmp_path / f"s{value}.nml"
        nml.write_text(
            "&SNLCINP\n  VERSION_PHOTOMETRY = 'V'\n  KCOR_FILE = 'k.fits'\n&END\n"
            "&FITINP\n  FITMODEL_NAME = 'M'\n  FILTLIST_FIT = 'griz'\n"
            f"  SCALE_COVAR = {value}\n&END\n")
        return build(read_keys(nml))
    assert cfg(4.1).model.use_model_magerr      # positive: inert, as upstream
    assert not cfg(0.0).model.use_model_magerr  # not positive: model error off


def test_file_keys_read_ignorefile_values_as_unset(tmp_path):
    """SNANA gates each file key through IGNOREFILE (snana.F90:18558-18626,
    :2816-2831), and EPOCH_IGNORE_FILE skips on NONE or NULL (:12876), so
    these values name no file rather than a file called NONE."""
    def cfg(snlcinp):
        nml = tmp_path / "f.nml"
        nml.write_text(
            "&SNLCINP\n  VERSION_PHOTOMETRY = 'V'\n  KCOR_FILE = 'k.fits'\n"
            f"{snlcinp}&END\n"
            "&FITINP\n  FITMODEL_NAME = 'M'\n  FILTLIST_FIT = 'griz'\n&END\n")
        return build(read_keys(nml))
    c = cfg("  MAGCOR_FILE = 'NONE'\n  NONLINEARITY_FILE = 'null'\n"
            "  FUDGE_HOSTNOISE_FILE = 'NULL'\n  FLUXERRMODEL_FILE = 'BLANK'\n"
            "  HEADER_OVERRIDE_FILE = 'NONE'\n  EPOCH_IGNORE_FILE = 'NONE'\n")
    assert c.data.magcor_file is None and c.data.nonlinearity_file is None
    assert c.data.fudge_hostnoise_file is None
    assert c.data.fluxerrmodel_file is None
    assert c.data.header_override_file is None
    assert c.selection.epoch_ignore_file is None
    c = cfg("  MAGCOR_FILE = 'mag.dat'\n  EPOCH_IGNORE_FILE = 'ep.ignore'\n")
    assert c.data.magcor_file == 'mag.dat'
    assert c.selection.epoch_ignore_file == 'ep.ignore'
    assert cfg("").selection.epoch_ignore_file == 'DEFAULT'


def test_fits_header_and_survey_def_are_required(tmp_path):
    """SNANA reads DATATYPE and SURVEY through snfitsio_errorCheck, which
    aborts (sntools_dataformat_fits.c:4491-4495, :4537-4541), and
    read_SURVEYDEF aborts on a missing SURVEY.DEF (sntools.c:3586-3588); an
    unmatched survey name is still -9 (:3722)."""
    from astropy.io import fits
    from jax_snlc_fit.output import survey_id

    head = tmp_path / "X_HEAD.FITS"
    primary = fits.PrimaryHDU()
    primary.header['DATATYPE'] = 'SIM_SNANA'
    fits.HDUList([primary, fits.BinTableHDU.from_columns(
        [fits.Column(name='SNID', format='8A', array=np.array(['1']))])]
                 ).writeto(head)
    with pytest.raises(SNANAError, match="lacks SURVEY"):
        read_snana_fits(str(head), str(tmp_path / "X_PHOT.FITS"))

    assert survey_id('NOT_A_SURVEY') == -9
    saved = os.environ['SNDATA_ROOT']
    os.environ['SNDATA_ROOT'] = str(tmp_path)
    try:
        with pytest.raises(SNANAError, match="Unable to open file"):
            survey_id('LSST')
    finally:
        os.environ['SNDATA_ROOT'] = saved


def test_unknown_opt_mwcolorlaw_aborts(tmp_path):
    """An unknown colour-law code is fatal in SNANA (text_MWoption,
    MWgaldust.c:218), not a silent Fitzpatrick 99."""
    def cfg(value):
        nml = tmp_path / "mw.nml"
        nml.write_text(
            "&SNLCINP\n  VERSION_PHOTOMETRY = 'V'\n  KCOR_FILE = 'k.fits'\n"
            f"  OPT_MWCOLORLAW = {value}\n&END\n"
            "&FITINP\n  FITMODEL_NAME = 'M'\n  FILTLIST_FIT = 'griz'\n&END\n")
        return build(read_keys(nml))
    assert cfg(94).data.opt_mwcolorlaw == 94
    # no extinction
    assert cfg(0).data.opt_mwcolorlaw == 0
    # SNANA's own unset value leaves the law to the data
    assert cfg(-999).data.opt_mwcolorlaw is None
    with pytest.raises(SNANAError, match="Invalid OPT_MWCOLORLAW = 98"):
        cfg(98)


@pytest.mark.parametrize("key,value,override", [
    ("OPT_MWCOLORLAW", -999, 94),
    ("OPT_MWCOLORLAW", -998, 94),
    ("RV_MWCOLORLAW", -1.0, 3.1),
    ("RV_MWCOLORLAW", -0.01, 3.1),
])
def test_mw_unset_values_survive_generic_field_parsing(key, value, override):
    raw = {"VERSION_PHOTOMETRY": "V", "KCOR_FILE": "k.fits",
           "FITMODEL_NAME": "M", "FILTLIST_FIT": "griz", key: value}
    assert getattr(build(raw).data, key.lower()) is None
    config = build(raw, settings={key.lower(): override})
    assert getattr(config.data, key.lower()) == override


def test_mxlc_plot_is_the_lcplot_bin_count(tmp_path):
    """MXLC_PLOT and SNTABLE_LIST's LCPLOT(mxlc:N) set the same variable
    (snana.F90:6692), and it is zeroed with no light-curve table (:6744)."""
    def cfg(tables, extra=""):
        nml = tmp_path / "m.nml"
        nml.write_text(
            "&SNLCINP\n  VERSION_PHOTOMETRY = 'V'\n  KCOR_FILE = 'k.fits'\n"
            f"  SNTABLE_LIST = '{tables}'\n{extra}&END\n"
            "&FITINP\n  FITMODEL_NAME = 'M'\n  FILTLIST_FIT = 'griz'\n&END\n")
        return build(read_keys(nml))
    assert cfg("FITRES LCPLOT", "  MXLC_PLOT = 7\n").output.mxlc == 7
    assert cfg("FITRES LCPLOT(mxlc:3)").output.mxlc == 3
    # MXLC_PLOT defaults to 5 (snana.F90:5978), and an explicit 0 plots none
    assert cfg("FITRES LCPLOT").output.mxlc == 5
    assert cfg("FITRES LCPLOT(mxlc:0)").output.mxlc == 0
    # no light-curve table requested, so nothing is plotted
    assert cfg("FITRES", "  MXLC_PLOT = 7\n").output.mxlc == 0


def test_photoz_trest_cuts_are_widened_for_the_fit():
    """A photo-z fit cannot cut on rest frame at the input redshift, so SNANA
    widens the window by (1+z) either way (TRESTzCUT, snlc_fit.F90:3661)."""
    from jax_snlc_fit.cuts import trest_zcut
    # a negative edge widens by multiplying, a positive one by dividing
    assert trest_zcut(1, 1.0, -15.0) == -30.0
    assert trest_zcut(2, 1.0, 45.0) == 90.0
    assert trest_zcut(1, 1.0, 5.0) == 2.5
    # every widened window contains the nominal one
    for lo, hi, z in ((-15.0, 45.0, 0.6), (5.0, 30.0, 1.4), (-20.0, 0.0, 0.3)):
        assert trest_zcut(1, z, lo) <= lo and trest_zcut(2, z, hi) >= hi


def test_opt_chi2_sigma_accepts_only_the_bit_that_does_something():
    """Bit 1 is the only value that changes a SNANA fit; 2, 128 and 1|128 give a
    FITRES byte-identical to 0, and 4 aborts snlc_fit.exe. Accepting any of them
    would hand back silence for a request that asked for a term."""
    base = {'VERSION_PHOTOMETRY': 'V', 'KCOR_FILE': 'k.fits',
            'FITMODEL_NAME': 'M', 'FILTLIST_FIT': 'griz'}

    assert build({**base, 'OPT_CHI2_SIGMA': '0'}).fit.opt_chi2_sigma == 0
    assert build({**base, 'OPT_CHI2_SIGMA': '1'}).fit.opt_chi2_sigma == 1
    # DOCHI2_SIGMA overwrites the mask before it is judged (snlc_fit.F90:7182),
    # so a refused mask alongside it is still a valid request for bit 1
    assert build({**base, 'OPT_CHI2_SIGMA': '128',
                  'DOCHI2_SIGMA': 'T'}).fit.opt_chi2_sigma == 1
    for opt in (2, 4, 8, 128, 129):
        with pytest.raises(SNANAError):
            build({**base, 'OPT_CHI2_SIGMA': str(opt)})


@pytest.mark.parametrize("keys, expected", [
    ({}, True),                                          # both switches default on
    ({'OPT_COVAR_FLUX': '0'}, True),                     # MWXTERR still on
    ({'OPT_COVAR_MWXTERR': '0'}, True),                  # COVAR_FLUX still on
    ({'OPT_COVAR_FLUX': '0', 'OPT_COVAR_MWXTERR': '0'}, False),
])
def test_use_fitcov_decides_frozen_versus_live_rounds(keys, expected,
                                                    salt_model, data_root):
    """USE_FITCOV (snlc_fit.F90:10566) is what tells rounds 2+ to freeze the
    error surface; false means USE_DIAG_ONLY, where SNANA keeps it live.
    OPT_COVAR_FLUX brings in SALT's colour dispersion and OPT_COVAR_MWXTERR
    the Milky Way extinction error, either of which puts the fit on the full
    epoch covariance."""
    base = {'VERSION_PHOTOMETRY': 'V', 'KCOR_FILE': 'k.fits',
            'FITMODEL_NAME': str(data_root / "models" / "SALT3.K21"),
            'FILTLIST_FIT': 'ugrizY'}
    shape = {'epoch_lengths': [4], 'batch_size': 1, 'band_window': None,
             'n_z': None}
    fitter = LightCurveFitter._from_prepared(
        build({**base, **keys}), salt_model,
        KcorReader(str(data_root / "kcor" / "JAX_TEST_LSST.fits")), shape)
    assert (fitter.rounds2_strategy != 'live_reweight') is expected


def test_prior_windows_resolve_to_parameter_names(
        tmp_path, salt_aliases, bayesn_aliases):
    """PRIOR_<p>_RANGE / _SIGMA name a parameter, so a model's own spelling
    works and SNANA's generic one is only a legacy alias resolved at load time
    (the same table INIVAL_ and FITWIN_ go through). Nothing downstream of the
    parser sees a generic name.
    """
    from jax_snlc_fit.parameters import resolve_parameter_keys

    def load(body, param_names, aliases):
        nml = tmp_path / "fit.nml"
        nml.write_text("&FITINP\n" + body + "&END\n")
        cfg = build(read_keys(nml))
        resolved = resolve_parameter_keys(cfg.model, param_names, aliases)
        return dataclasses.replace(
            cfg, model=dataclasses.replace(cfg.model, **resolved))

    salt = ["PKMJD", "x1", "c", "x0"]
    bayesn = ["PKMJD", "Ds", "THETA", "AV"]

    # the parameter's own name works, which it did not before
    cfg = load("  PRIOR_x1_RANGE = -3.0, 3.0\n  PRIOR_x1_SIGMA = 0.2\n",
               salt, salt_aliases)
    assert cfg.model.priors == {
        "x1": {"flat_rolloff": {"lo": -3.0, "hi": 3.0, "sigma": 0.2}}}

    # SNANA's generic spelling and its legacy alias reach the same parameter
    for key in ("PRIOR_SHAPE_RANGE", "PRIOR_LUMIPAR_RANGE"):
        assert load(f"  {key} = -3.0, 3.0\n", salt, salt_aliases).model.priors \
            == {"x1": {"flat_rolloff": {"lo": -3.0, "hi": 3.0}}}

    # and the same key reaches a different parameter under a different model
    assert load("  PRIOR_SHAPE_RANGE = -2.0, 2.0\n", bayesn,
                bayesn_aliases).model.priors == {
                    "THETA": {"flat_rolloff": {"lo": -2.0, "hi": 2.0}}}
    assert load("  PRIOR_COLOR_RANGE = -0.3, 0.3\n", salt,
                salt_aliases).model.priors == {
                    "c": {"flat_rolloff": {"lo": -0.3, "hi": 0.3}}}

    # the AV prior's keys, written without a separator, reach AV by the same
    # rule and land under the form they configure
    assert load("  PRIOR_AVEXP = 0.194, 1e9\n  PRIOR_AVWGT = 1.0, 0.0\n"
                "  PRIOR_AVRES = 0.01\n", bayesn,
                bayesn_aliases).model.priors == {
                    "AV": {"exp_convolved": {"tau": [0.194, 1e9],
                                             "wgt": [1.0, 0.0],
                                             "sigma": 0.01}}}

    # a parameter this model does not fit is refused rather than ignored
    with pytest.raises(SNANAError, match="not a parameter of this model"):
        load("  PRIOR_x1_RANGE = -3.0, 3.0\n", bayesn, bayesn_aliases)

    # PRIOR_ keys that name something other than a parameter are untouched
    cfg = load("  PRIOR_MJDSIG = 5.0\n  PRIOR_ZERRSCALE = 2.0\n",
               salt, salt_aliases)
    assert cfg.model.priors == {}
    assert cfg.model._model_key_values['PRIOR_MJDSIG'] == 5.0
    assert cfg.photoz.prior_zerrscale == 2.0


def test_the_keys_jax_adds_reach_their_fields(tmp_path):
    """Settings snlc_fit.exe has no counterpart for are keys like any other,
    so the input file is the only place a fit is configured from."""
    nml = tmp_path / "fit.nml"
    nml.write_text(
        "&SNLCINP\n"
        "  MAX_FIT = 7\n"
        "  DOUBLE_PRECISION = F\n"
        "&END\n"
        "&FITINP\n"
        "  BATCH_SIZE = 50\n"
        "  NUM_NOBS_BINS = 3\n"
        "  MIN_BIN_GAIN = 0.1\n"
        "  MAXITER_ROUND1 = 11\n"
        "  MAXITER_ROUND2 = 9\n"
        "  MODEL_TYPE = 'salt'\n"
        "  OUTPUT_NAME = 'x.FITRES.TEXT'\n"
        "&END\n")
    cfg = build(read_keys(nml))
    assert cfg.fit.double_precision is False
    assert cfg.fit.batch_size == 50
    assert cfg.fit.num_nobs_bins == 3
    assert cfg.fit.min_bin_gain == 0.1
    assert cfg.fit.maxiter_round1 == 11
    assert cfg.fit.maxiter_round2 == 9
    assert cfg.selection.max_fit == 7
    assert cfg.model.model_type == 'salt'
    assert cfg.output.output_name == 'x.FITRES.TEXT'

    # MODEL_TYPE names a registered model or aborts naming them
    with pytest.raises(SNANAError, match="Unknown model type"):
        build({'MODEL_TYPE': 'salt2'})


def test_nonuisance_drops_the_model_s_nuisance_parameters():
    """SNTABLE_LIST's jax option, beside SNANA's own NOSIMVAR and NOZPHOT."""
    assert build({'SNTABLE_LIST': 'SNANA FITRES NONUISANCE'}).output.nonuisance
    assert not build({'SNTABLE_LIST': 'SNANA FITRES'}).output.nonuisance


def test_every_recognised_key_reaches_its_field():
    """Keys the parser reads but nothing else in the suite sets. A key that
    stops reaching its field is a knob the input file can no longer turn."""
    cfg = build({
        'VERSION_PHOTOMETRY_WILDCARD': 'DES*',
        'SNCCID_IGNORE': ['sn1', 'sn2'],
        'SNCID_IGNORE_FILE': 'ignore.list',
        'MXEVT_CUTS': '12',
        'MAGOBS_SHIFT_ZP_PARAMS': '0.1 0.2',
        'MAGOBS_SHIFT_PRIMARY_PARAMS': '0.3 0.4',
        'SNRMAX_FILTERS': 'gri',
        'FILTER_REPLACE': 'gr -> ri',
        'PEAKMJD_SANITY': '-5 5',
        'OPT_VPEC_COR': '2',
        'USE_SNHOST_ZPHOT': 'T',
        'USE_SNHOST_QZPHOT': 'T',
        'LFIXPAR_ADJUST': 'T',
        'PRIVATE_VARNAME_READLIST': 'HOST_LOGMASS',
        'SIM_TEMPLATE_INDEX_LIST': '3 4',
        'ENABLE_MAGSHIFT': 'T',
        'ENABLE_WAVESHIFT': 'T',
        'SKIP_PHOTOZ_INIT': 'T',
        'FITWIN_MJD': '50000 60000',
        'PHOTOZ_ITER1_LAMRANGE': '3000 9000',
        'NBINZ_COURSE_PHOTOZ': '7',
        'RANGEZ_COURSE_PHOTOZ': '0.05 1.2',
        'NBINT_COURSE_PHOTOZ': '4',
        'RANGET_COURSE_PHOTOZ': '-10 10',
        'OPT_VMAX': '2',
        'FLUXCAL_BASELINE_CORRECT': 'T',
    })
    assert cfg.data.version_photometry_wildcard == 'DES*'
    assert cfg.data.fluxcal_baseline_correct
    assert cfg.data.filter_replace == {'g': 'r', 'r': 'i'}
    assert cfg.data.private_varname_readlist == ('HOST_LOGMASS',)
    assert cfg.data.enable_magshift and cfg.data.enable_waveshift
    assert cfg.data.magobs_shift_zp_params == (0.1, 0.2, 0.0)
    assert cfg.data.opt_vpec_cor == 2
    assert cfg.data.use_snhost_zphot and cfg.data.use_snhost_qzphot
    assert cfg.data.peakmjd_sanity == (-5.0, 5.0)
    assert cfg.selection.snccid_ignore == ['sn1', 'sn2']
    assert cfg.selection.sncid_ignore_file == 'ignore.list'
    assert cfg.selection.sim_template_index_list == [3, 4]
    assert cfg.selection.mxevt_cuts == 12
    assert cfg.selection.snrmax_filters == 'gri'
    assert cfg.fit.fitwin_mjd == (50000.0, 60000.0)
    assert cfg.data.magobs_shift_primary_params == (0.3, 0.4, 0.0)
    assert cfg.model.lfixpar_adjust
    assert cfg.photoz.skip_photoz_init
    assert cfg.photoz.nbinz_course == 7
    assert cfg.photoz.rangez_course == (0.05, 1.2)
    assert cfg.photoz.nbint_course == 4
    assert cfg.photoz.ranget_course == (-10.0, 10.0)
    assert cfg.photoz.photoz_iter1_lamrange == (3000.0, 9000.0)
    assert cfg.output.opt_vmax == 2

    # LFIXPAR_ADJUST fixes every parameter, so no Newton step is taken
    # (snlc_fit.F90:1979)
    assert cfg.fit.maxiter_round1 == 0 and cfg.fit.maxiter_round2 == 0

    # OPT_VMAX bit 0 indexes PEAKMAG_OBS_MODEL by Vmax list position where
    # SNANA fills it by survey-filter index (snlc_fit.F90:15240 against :14738)
    with pytest.raises(SNANAError, match="OPT_VMAX bit 0"):
        build({'OPT_VMAX': '1'})


def test_photoz_constrained_amplitude(data_root):
    from jax_snlc_core.models import build_model
    from jax_snlc_fit.config import PhotoZ
    from jax_snlc_fit.config_parser import build_config
    from jax_snlc_fit.fitter import resolved_config

    path = str(data_root / 'models' / 'SALT3.K21')
    model = build_model(path, {})
    config = FitConfig(model=Model(fitmodel_name=path, inistp={'DLMAG': 0}),
                       photoz=PhotoZ(dofit_photoz=True))
    with pytest.raises(SNANAError, match='cosmology-constrained'):
        build_config(config=config)
    # An explicit fixed x0 still means holding that amplitude constant.
    config = dataclasses.replace(config, model=dataclasses.replace(
        config.model, inistp={'x0': 0}))
    assert resolved_config(config, model).model.inistp['x0'] == 0


@pytest.mark.parametrize('fixture', ['hostlib_zphot_q', 'hostlib_nbr_zphot_q'])
def test_text_host_metadata_matches_fits(snana_fixture, fixture):
    """Error pairs, band magnitudes and quantiles survive normal TEXT loading."""
    root = snana_fixture(fixture).root
    text, _, metadata = read_snana_text(sorted((root / 'text_ref').glob('*.DAT')))
    fits, _, fits_metadata = read_snana_fits(root / 'HEAD.FITS', root / 'PHOT.FITS')
    rows = {str(cid).strip(): i for i, cid in enumerate(fits['SNID'])}
    fits = fits[[rows[cid] for cid in text['SNID']]]
    assert metadata['survey'] == fits_metadata['survey']
    for column in ('HOSTGAL_OBJID', 'HOSTGAL_NMATCH', 'HOSTGAL_NMATCH2',
                   'MASK_REDSHIFT_SOURCE'):
        np.testing.assert_array_equal(text[column], fits[column])
    for column, tolerance in (
        ('HOSTGAL_PHOTOZ', 6e-6), ('HOSTGAL_PHOTOZ_ERR', 6e-6),
        ('HOSTGAL_LOGMASS', 6e-4), ('HOSTGAL_LOGMASS_ERR', 6e-4),
        ('REDSHIFT_HELIO_ERR', 6e-7), ('VPEC', 0.006), ('VPEC_ERR', 0.006),
        ('HOSTGAL_MAG_u', 6e-4), ('HOSTGAL_MAG_r', 6e-4),
        ('HOSTGALz_QUANTILE_ZPHOT', 6e-5),
        ('HOSTGALz_QUANTILE_PERCENT', 0.006),
    ):
        np.testing.assert_allclose(text[column], fits[column], atol=tolerance, rtol=0)
    # These are the arrays used by the Gaussian and quantile photo-z priors.
    text_prior = _host_photoz_columns(text, True)
    fits_prior = _host_photoz_columns(fits, True)
    for actual, expected in zip(text_prior, fits_prior):
        np.testing.assert_allclose(actual, expected, atol=6e-5, rtol=0)
    if fixture == 'hostlib_nbr_zphot_q':
        for column in ('HOSTGAL2_OBJID', 'HOSTGAL2_PHOTOZ_ERR', 'HOSTGAL2_MAG_r',
                       'HOSTGAL2z_QUANTILE_ZPHOT'):
            np.testing.assert_allclose(text[column], fits[column], atol=6e-4, rtol=0)


def test_text_preserves_large_ids_and_variable_host_grids(tmp_path):
    """Large IDs remain integers; shorter or absent host grids get null padding."""
    first = tmp_path / 'one.DAT'
    first.write_text('''SURVEY: DES(SHALLOW)
SNID: event-one
FILTERS: gr
FAKE: 0
REDSHIFT_HELIO: 0.2 +- 0.003
HOSTGAL_OBJID: 9007199254740993
HOSTGAL_MAG: 21.1
    22.2
HOSTGAL_PHOTOZ: 0.21 +- 0.04
HOSTGALz_NBIN_LOGMASS_ZGRID: 2
HOSTGALz_LOGMASS_ZGRID: 0.1 0.3
HOSTGALz_LOGMASS_VALGRID: 9.1 9.3
HOSTGALz_LOGMASS_ERRGRID: 0.1 0.2
VARLIST: MJD BAND FLUXCAL FLUXCALERR
OBS: 60000 g 10 1
''')
    second = tmp_path / 'two.DAT'
    second.write_text('''SURVEY: DES(DEEP)
SNID: event-two
FILTERS: gr
FAKE: 0
HOSTGAL_NMATCH: 0
VARLIST: MJD BAND FLUXCAL FLUXCALERR
OBS: 60001 r 20 2
''')
    head, phot, metadata = read_snana_text([first, second])
    assert metadata['survey'] == 'DES'
    assert list(head['SUBSURVEY']) == ['SHALLOW', 'DEEP']
    assert head['HOSTGAL_OBJID'].dtype.kind in 'iu'
    assert head['HOSTGAL_OBJID'][0] == 9007199254740993
    assert head['HOSTGAL_OBJID'][1] == 0
    np.testing.assert_allclose(head['HOSTGAL_MAG_r'], [22.2, 999.0])
    np.testing.assert_allclose(head['HOSTGALz_LOGMASS_VALGRID'][0, :2], [9.1, 9.3])
    assert np.all(head['HOSTGALz_LOGMASS_VALGRID'][0, 2:] == -9)
    assert np.all(head['HOSTGALz_LOGMASS_VALGRID'][1] == -9)
    assert head['HOSTGALz_NBIN_LOGMASS_ZGRID'][1] == 0
    np.testing.assert_array_equal(head['PTROBS_MIN'], [1, 2])
    np.testing.assert_array_equal(phot['FLUXCAL'], [10, 20])


@pytest.mark.parametrize('value', ['invalid', '0.1 +- invalid'])
@pytest.mark.parametrize('field', ['MWEBV', 'REDSHIFT_HELIO', 'HOSTGAL_PHOTOZ',
                                   'HOSTGAL_MAG', 'HOSTGALz_LOGMASS_ZGRID'])
def test_text_rejects_invalid_numeric_metadata(tmp_path, field, value):
    """Malformed scientific values must not disappear or become defaults."""
    from jax_snlc_core.logging import SNANAError

    path = tmp_path / 'bad.DAT'
    path.write_text(f"SURVEY: DES\nSNID: 1\nFILTERS: g\n{field}: {value}\n"
                    "VARLIST: MJD BAND FLUXCAL FLUXCALERR\nOBS: 60000 g 10 1\n")
    with pytest.raises(SNANAError, match=field):
        read_snana_text([path])


def test_text_empty_host_grids_support_overrides(tmp_path):
    """Zero-bin TEXT grids have the same null arrays as FITS, ready for overrides."""
    from jax_snlc_fit.loader import apply_header_override

    path = tmp_path / 'empty.DAT'
    path.write_text("SURVEY: DES\nSNID: 1\nFILTERS: g\n"
                    "HOSTGAL_OBJID: 10\nHOSTGAL2_OBJID: 20\n"
                    "HOSTGAL_PHOTOZ: -9 +- -9\nHOSTGAL2_PHOTOZ: -9 +- -9\n"
                    "HOSTGALz_NBIN_LOGMASS_ZGRID: 0\n"
                    "HOSTGALz_NBIN_QUANTILE_ZPHOT: 0\n"
                    "HOSTGAL2z_NBIN_QUANTILE_ZPHOT: 0\n"
                    "VARLIST: MJD BAND FLUXCAL FLUXCALERR\nOBS: 60000 g 10 1\n")
    head, _, _ = read_snana_text([path])
    for prefix in ('HOSTGALz', 'HOSTGAL2z'):
        for suffix in ('QUANTILE_ZPHOT', 'QUANTILE_PERCENT'):
            assert head[f'{prefix}_{suffix}'].shape == (1, 20)
            assert np.all(head[f'{prefix}_{suffix}'] == -9)
    values = {f'HOSTGALz_{name}': {'1': row} for name, row in (
        ('LOGMASS_ZGRID', [0.1, 0.3]), ('LOGMASS_VALGRID', [9., 10.]),
        ('LOGMASS_ERRGRID', [0.1, 0.1]))}
    updated = apply_header_override(head, ('SNID', values))
    assert updated['HOSTGALz_NBIN_LOGMASS_ZGRID'][0] == 2
    np.testing.assert_allclose(updated['HOSTGALz_LOGMASS_VALGRID'][0, :2], [9., 10.])
    assert np.all(updated['HOSTGALz_LOGMASS_VALGRID'][0, 2:] == -9)

    quantiles = {'HOSTGALz_QUANTILE_ZPHOT': {'10': [0.1, 0.2], '20': [0.3, 0.4]},
                 'HOSTGALz_QUANTILE_PERCENT': {'10': [0., 100.], '20': [0., 100.]}}
    updated = apply_header_override(head, ('HOSTGAL_OBJID', quantiles))
    for prefix, expected in (('HOSTGALz', [0.1, 0.2]), ('HOSTGAL2z', [0.3, 0.4])):
        assert updated[f'{prefix}_NBIN_QUANTILE_ZPHOT'][0] == 2
        np.testing.assert_allclose(updated[f'{prefix}_QUANTILE_ZPHOT'][0, :2], expected)


@pytest.mark.parametrize('reverse', [False, True])
def test_text_version_and_optional_epoch_columns(tmp_path, reverse):
    """Version strings and per-event optional columns retain their meanings."""
    first = tmp_path / 'one.DAT'
    second = tmp_path / 'two.DAT'
    header = 'SURVEY: DES\nFILTERS: g\nSNANA_VERSION: v12_00\n'
    first.write_text(header + 'SNID: 1\n'
                     'VARLIST: MJD FLT FLUXCAL FLUXCALERR PHOTFLAG FIELD ZPT\n'
                     'OBS: 60000 g 10 1 8 C3 27.5\n'
                     'OBS: 60001 g 11 1 0 C3 27.5\n')
    second.write_text(header + 'SNID: 2\n'
                      'VARLIST: MJD BAND FLUXCAL FLUXCALERR\n'
                      'OBS: 60002 g 20 2\n')
    paths = [second, first] if reverse else [first, second]
    head, phot, metadata = read_snana_text(paths)
    assert metadata['snana_version'] == 'v12_00'
    assert 'SNANA_VERSION' not in head.colnames
    expected = [2, 0, 1] if reverse else [0, 1, 2]
    np.testing.assert_array_equal(phot['MJD'], np.array([60000, 60001, 60002])[expected])
    np.testing.assert_array_equal(phot['PHOTFLAG'], np.array([8, 0, 0])[expected])
    np.testing.assert_array_equal(phot['FIELD'], np.array(['C3', 'C3', ''])[expected])
    np.testing.assert_allclose(phot['ZEROPT'], np.array([27.5, 27.5, np.nan])[expected])
    np.testing.assert_array_equal(phot['BAND'], ['g', 'g', 'g'])
    np.testing.assert_array_equal(head['PTROBS_MIN'], [1, 2] if reverse else [1, 3])
    np.testing.assert_array_equal(head['PTROBS_MAX'], [1, 3] if reverse else [2, 3])


@pytest.mark.parametrize('reverse', [False, True])
def test_missing_header_values_keep_loader_defaults(tmp_path, reverse):
    """Reading events together preserves redshift fallbacks and trigger defaults."""
    from jax_snlc_fit.loader import _head_metadata, batch_from_columns

    paths = [tmp_path / 'first.DAT', tmp_path / 'second.DAT']
    headers = [
        'REDSHIFT_HELIO: 0.2 +- 0.001\nREDSHIFT_FINAL: 0.21 +- 0.002\n'
        'MWEBV: 0.05 +- 0.003\nVPEC: 100\nMJD_TRIGGER: 60001\nPRIVATE: 7\n',
        'REDSHIFT_CMB: 0.25 +- 0.004\n',
    ]
    for i, (path, header) in enumerate(zip(paths, headers)):
        path.write_text(f'SURVEY: DES\nSNID: {i}\nFILTERS: g\nFAKE: 0\n'
                        + header + 'VARLIST: MJD BAND FLUXCAL FLUXCALERR\n'
                        'OBS: 60000 g 10 1\n')
    if reverse:
        paths.reverse()
    head, phot, metadata = read_snana_text(paths)
    combined = _head_metadata(head)
    for i, path in enumerate(paths):
        single, _, _ = read_snana_text([path])
        expected = _head_metadata(single)
        for name in ('redshifts', 'redshift_errs', 'mwebvs', 'mwebv_errs',
                     'ras', 'decs', 'peakmjds'):
            np.testing.assert_allclose(combined[name][i], expected[name][0], equal_nan=True)
    absent = list(head['SNID']).index('1')
    assert head['PRIVATE'][absent] == -9
    assert head['VPEC'][absent] == 0
    assert head['MJD_TRIGGER'][absent] == -99
    batch = batch_from_columns(head, phot, **metadata, setpkmjd={
        'trigger': True, 'fmax_clump': False, 'fluxmax': False,
        'bazin': False, 'batch_size': 2})
    expected_peak = [-99., 60001.] if reverse else [60001., -99.]
    np.testing.assert_array_equal(batch.head_extras['PEAKMJD'], expected_peak)
