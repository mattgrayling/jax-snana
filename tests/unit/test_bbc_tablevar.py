"""Unit tests for the BBC tablevar reader, z-bins, and IDSAMPLE assignment."""


import numpy as np
import pytest

from jax_salt2mu.config import (BBCConfig,
                                MASK_BIASCOR_1DZAVG, MASK_BIASCOR_CELLMAP,
                                MASK_BIASCOR_DEFAULT, MASK_BIASCOR_MAD,
                                MASK_BIASCOR_MUCOVADD, MASK_BIASCOR_SAMPLE, resolve)
from jax_salt2mu.config_parser import parse_bbc_input
from jax_salt2mu.run import floated_and_defaults
from jax_salt2mu.tablevar import (CUTBIT_SIMPS, EVENT_TYPE_BIASCOR, EVENT_TYPE_DATA,
                                  accept_zrate_scale, read_tablevar)
from jax_salt2mu.zbins import make_zbins
from jax_snlc_core.io.fits_path import open_text
from jax_snlc_core.models import SALTModel, standardised_model
from jax_snlc_core.physics.constants import LIGHT_KM
from jax_snlc_core.utils.error_handling import SNANAError


_SALT_FITRES = "tests/fixtures/snana/salt3_fit_dense/FITRES.TEXT"
_BEAMS_BIASCOR = "tests/fixtures/snana/bbc_des_beams/BIASCOR.FITRES"
_BEAMS_REAL = "tests/fixtures/snana/bbc_des_beams/DATA_REAL.FITRES"


def _salt_config(**zkw):
    # resolved, as run_bbc_fit reads it: the model's BBC keys at their defaults
    return _resolve(BBCConfig(model_lcfit="SALT2", zmin=0.01, zmax=2.0, **zkw))


def test_read_salt_data(tmp_path):
    cfg = _salt_config(nzbin=10)
    tv = read_tablevar([_SALT_FITRES], EVENT_TYPE_DATA, cfg, SALTModel)
    assert tv.size > 0
    # mB derived from x0 (offset-free), finite; 3x3 covariance; finite mumodel.
    assert set(tv.lcpar) == {"mB", "x1", "c"}
    assert np.all(np.isfinite(tv.lcpar["mB"]))
    assert tv.covmat_fit.shape == (tv.size, 3, 3)
    assert np.all(np.isfinite(tv.mumodel))
    # single-sample default
    assert np.all(tv.idsample == 0)
    # covariance symmetric
    np.testing.assert_allclose(tv.covmat_fit, np.transpose(tv.covmat_fit, (0, 2, 1)))


def test_cutwin_reduces_passing():
    base = read_tablevar([_SALT_FITRES], EVENT_TYPE_DATA, _salt_config(nzbin=10),
                         SALTModel)
    cut_cfg = _salt_config(nzbin=10)
    cut_cfg.cutwin = [("x1", -0.5, 0.5)]
    cut = read_tablevar([_SALT_FITRES], EVENT_TYPE_DATA, cut_cfg, SALTModel)
    assert cut.n_pass < base.n_pass
    assert np.all(np.abs(cut.lcpar["x1"][cut.cutmask]) <= 0.5)


@pytest.mark.parametrize("disable", [False, True])
@pytest.mark.parametrize("event_type", [EVENT_TYPE_DATA, EVENT_TYPE_BIASCOR])
def test_covariance_quality_cut_survives_cutwin_none(disable, event_type):
    from dataclasses import replace
    from jax_snlc_core.io.fitres_reader import read_fitres
    from jax_salt2mu.tablevar import CUTBIT_BADCOV, tablevar_from

    table = read_fitres(_BEAMS_BIASCOR)
    # Positive-definite examples isolate each of the three covariance cuts
    # from covariance repair, plus the independent mB uncertainty cut.
    table = replace(table, df=table.df.iloc[:6].copy())
    sf = -2.5 / np.log(10.0)
    table.df["x0"] = 1.0
    table.df["x0ERR"] = 0.1 / abs(sf)
    table.df["x1ERR"] = 1.0
    table.df["cERR"] = 0.1
    for name in ["COV_x1_x0", "COV_c_x0", "COV_x1_c"]:
        table.df[name] = 0.0
    table.df.iloc[1, table.df.columns.get_loc("x0ERR")] = 1.01 / abs(sf)
    table.df.iloc[2, table.df.columns.get_loc("x1ERR")] = 100.0
    table.df.iloc[2, table.df.columns.get_loc("COV_x1_x0")] = 8.1 / sf
    table.df.iloc[3, table.df.columns.get_loc("cERR")] = 100.0
    table.df.iloc[3, table.df.columns.get_loc("COV_c_x0")] = -8.1 / sf
    table.df.iloc[4, table.df.columns.get_loc("cERR")] = 10.0
    table.df.iloc[4, table.df.columns.get_loc("COV_x1_c")] = 8.1
    # An undefined covariance is cleared by repair before the quality cut.
    table.df.iloc[5, table.df.columns.get_loc("COV_x1_c")] = -9.0
    cfg = _salt_config(nzbin=10, cutwin_disable=disable)
    tv = tablevar_from(table, event_type, cfg, SALTModel)
    np.testing.assert_array_equal((tv.cutbits & (1 << CUTBIT_BADCOV)) != 0,
                                  [False, True, True, True, True, False])


def test_cutwin_on_an_absent_column_aborts():
    """SNANA reads any named column for a CUTWIN and aborts when the table
    has no such column (set_DOFLAG_SELECT_VAR, SALT2mu.c:21380)."""
    cfg = _salt_config(nzbin=10)
    cfg.cutwin = [("NOSUCHVAR", 0.0, 1.0)]
    with pytest.raises(SNANAError, match="NOSUCHVAR"):
        read_tablevar([_SALT_FITRES], EVENT_TYPE_DATA, cfg, SALTModel)


def test_absent_idsurvey_reads_as_minus_9(tmp_path):
    """SNTABLE_READPREP_TABLEVAR leaves IDSURVEY at -9 when the column is
    absent (SALT2mu.c:8533)."""
    out, ivar = [], None
    for line in open(_SALT_FITRES):
        w = line.split()
        if line.startswith("VARNAMES:"):
            ivar = w.index("IDSURVEY")
        if ivar is not None and w[:1] in (["VARNAMES:"], ["SN:"]):
            line = " ".join(w[:ivar] + w[ivar + 1:]) + "\n"
        out.append(line)
    path = tmp_path / "NO_IDSURVEY.TEXT"
    path.write_text("".join(out))
    tv = read_tablevar([str(path)], EVENT_TYPE_DATA, _salt_config(nzbin=10), SALTModel)
    assert np.all(tv.idsurvey == -9)


def test_vpec_err_is_read_before_vpecerr(tmp_path):
    """With both columns present, VPEC_ERR is read: SNTABLE_READPREP_VARDEF
    takes the first name of 'VPEC_ERR:F VPECERR:F' the table has
    (SALT2mu.c:8651)."""
    out, ivar = [], None
    for line in open(_SALT_FITRES):
        w = line.split()
        if line.startswith("VARNAMES:"):
            ivar = w.index("VPECERR")
            w = w + ["VPEC_ERR"]
        elif ivar is not None and w[:1] == ["SN:"]:
            assert float(w[ivar]) != 111.0
            w = w + ["111.0"]
        out.append(" ".join(w) + "\n")
    path = tmp_path / "BOTH_VPECERR.TEXT"
    path.write_text("".join(out))
    tv = read_tablevar([str(path)], EVENT_TYPE_DATA, _salt_config(nzbin=10), SALTModel)
    assert tv.size > 0
    np.testing.assert_allclose(tv.zmuerr * LIGHT_KM, 111.0, rtol=1e-6)

def test_zrate_scale_outside_the_table_aborts(tmp_path):
    """interp_1DFUN aborts when z lies outside the table (sntools.c:7962), where
    np.interp would clamp."""
    table = tmp_path / "zrate.txt"
    table.write_text("0.1 0.5\n0.5 0.5\n")
    names = np.array(["1001", "1002"])
    assert accept_zrate_scale(names, np.array([0.2, 0.4]), str(table)).all()
    with pytest.raises(SNANAError, match="outside the zrate_scale_file range"):
        accept_zrate_scale(names, np.array([0.2, 0.6]), str(table))


def _zrate_inputs(tmp_path, cc_zcmb="5.0", sim_zcmb_name="SIM_ZCMB"):
    """Thirty biasCor rows with zHD moved outside a zero-scale table on [0, 2],
    and every third row made a CC event at SIM_ZCMB ``cc_zcmb``, by default
    also outside it. The SIM_ZCMB column is named ``sim_zcmb_name``."""
    out, names, rows = [], None, 0
    with open_text(_BEAMS_BIASCOR) as stream:
        lines = stream.read().splitlines()
    for line in lines:
        w = line.split()
        if w[:1] == ["VARNAMES:"]:
            names = list(w)
            w[names.index("SIM_ZCMB")] = sim_zcmb_name
            line = " ".join(w)
        elif w[:1] == ["SN:"]:
            if rows == 30:
                continue
            w[names.index("zHD")] = "3.0"
            if rows % 3 == 0:
                w[names.index("SIM_TEMPLATE_INDEX")] = "2"
                w[names.index("SIM_ZCMB")] = cc_zcmb
            rows += 1
            line = " ".join(w)
        out.append(line)
    fitres = tmp_path / "ZRATE.FITRES"
    fitres.write_text("\n".join(out) + "\n")
    table = tmp_path / "zrate.txt"
    table.write_text("0.0 0.0\n2.0 0.0\n")
    cfg = _salt_config(nzbin=10)
    cfg.zrate_scale_file = str(table)
    return fitres, cfg


def _simps_matches_cid_draw(tv):
    """A zero scale keeps only the SNIa whose CID ends in 000; CC events pass."""
    simps = (tv.cutbits >> CUTBIT_SIMPS) & 1 == 1
    is_ia = tv.sim_truth["SIM_TEMPLATE_INDEX"] == 0
    assert is_ia.any() and not is_ia.all()
    assert not simps[~is_ia].any()
    np.testing.assert_array_equal(simps[is_ia],
                                  [not name.endswith("000") for name in tv.name[is_ia]])


def test_zrate_scale_cuts_simulated_snia_at_their_true_redshift(tmp_path):
    """prescale_reject_simData evaluates the scale only for a simulated SNIa,
    at SIM_ZCMB (SALT2mu.c:19185-19189): a CC event, or a zHD outside the
    table, neither aborts nor is cut."""
    fitres, cfg = _zrate_inputs(tmp_path)
    _simps_matches_cid_draw(read_tablevar([str(fitres)], EVENT_TYPE_DATA, cfg, SALTModel))


def test_zrate_scale_ignores_real_data(tmp_path):
    """Real data return before any prescale (SALT2mu.c:19162)."""
    table = tmp_path / "zrate.txt"
    table.write_text("0.0 0.0\n0.01 0.0\n")
    cfg = _salt_config(nzbin=10)
    cfg.zrate_scale_file = str(table)
    tv = read_tablevar([_BEAMS_REAL], EVENT_TYPE_DATA, cfg, SALTModel)
    assert not ((tv.cutbits >> CUTBIT_SIMPS) & 1).any()


def test_zrate_scale_on_biascor_looks_up_every_event(tmp_path):
    """prescale_reject_biasCor looks up every event at SIM_ZCMB and rejects
    only SNIa (SALT2mu.c:19223-19231), so a CC event outside the table aborts,
    and with every event inside it only the SNIa are cut."""
    fitres, cfg = _zrate_inputs(tmp_path)
    with pytest.raises(SNANAError, match="outside the zrate_scale_file range"):
        read_tablevar([str(fitres)], EVENT_TYPE_BIASCOR, cfg, SALTModel)

    fitres, cfg = _zrate_inputs(tmp_path, cc_zcmb="1.0")
    _simps_matches_cid_draw(read_tablevar([str(fitres)], EVENT_TYPE_BIASCOR, cfg, SALTModel))


@pytest.mark.parametrize("event_type", [EVENT_TYPE_DATA, EVENT_TYPE_BIASCOR])
def test_simulated_input_without_sim_zcmb_aborts(tmp_path, event_type):
    """SALT2mu counts SIM_ZCMB as a required variable of any simulated input,
    one with SIM_TEMPLATE_INDEX, and aborts at the read when it is missing
    (SALT2mu.c:8918-8921, :8954-8957)."""
    fitres, cfg = _zrate_inputs(tmp_path, sim_zcmb_name="SIM_ZCMB_RENAMED")
    with pytest.raises(SNANAError, match="SIM_ZCMB"):
        read_tablevar([str(fitres)], event_type, cfg, SALTModel)


def _renamed_template_index(tmp_path, name):
    """The SALT FITRES with SIM_TEMPLATE_INDEX renamed to ``name``."""
    path = tmp_path / f"{name or 'NONE'}.FITRES"
    lines = []
    for line in open(_SALT_FITRES):
        w = line.split()
        if w[:1] == ["VARNAMES:"]:
            i = w.index("SIM_TEMPLATE_INDEX")
            if name:
                w[i] = name
            else:
                del w[i]
            line = " ".join(w) + "\n"
        elif name is None and w[:1] == ["SN:"]:
            del w[i]
            line = " ".join(w) + "\n"
        lines.append(line)
    path.write_text("".join(lines))
    return str(path)


def test_sim_nonia_index_is_the_template_index(tmp_path):
    """SALT2mu reads SIM_NONIA_INDEX or SIM_TEMPLATE_INDEX into the template
    index (SALT2mu.c:8822)."""
    path = _renamed_template_index(tmp_path, "SIM_NONIA_INDEX")
    tv = read_tablevar([path], EVENT_TYPE_BIASCOR, _salt_config(nzbin=10), SALTModel)
    ref = read_tablevar([_SALT_FITRES], EVENT_TYPE_BIASCOR, _salt_config(nzbin=10),
                        SALTModel)
    np.testing.assert_array_equal(tv.sim_truth["SIM_TEMPLATE_INDEX"],
                                  ref.sim_truth["SIM_TEMPLATE_INDEX"])


def test_input_without_a_template_index_is_real_data(tmp_path):
    """Without a template-index column SALT2mu treats the input as real data
    and reads no SIM column (SALT2mu.c:8822-8832)."""
    path = _renamed_template_index(tmp_path, None)
    tv = read_tablevar([path], EVENT_TYPE_DATA, _salt_config(nzbin=10), SALTModel)
    assert tv.sim_truth == {}


def test_make_zbins_uniform_log_user():
    uni = make_zbins(BBCConfig(nzbin=4, zmin=0.01, zmax=0.8))
    assert uni.nbin == 4
    np.testing.assert_allclose(uni.hi[-1], 0.8)

    log = make_zbins(BBCConfig(nlogzbin=3, zmin=0.01, zmax=1.0))
    assert log.nbin == 3

    user = make_zbins(BBCConfig(zbinuser=[0.01, 0.05, 0.2, 0.6]))
    assert user.nbin == 3
    idx = user.index(np.array([0.03, 0.1, 0.5, 5.0]))
    np.testing.assert_array_equal(idx, [0, 1, 2, -1])
    # SNANA IBINFUN takes the first inclusive match at a shared boundary.
    edges = make_zbins(BBCConfig(zbinuser=[0.125, 0.25, 0.5]))
    np.testing.assert_array_equal(
        edges.index([0.125, 0.25, 0.5, 0.50001]), [0, 0, 1, -1])


def _resolve(cfg):
    return resolve(cfg, standardised_model(cfg.model_lcfit))


def _resolved(**zbins):
    return _resolve(BBCConfig(**zbins))


def test_resolve_normalises_nzbin_and_range():
    """nzbin/zmin/zmax end up describing the bins actually built, whichever key
    supplied them (SALT2mu.c:2896, :21965) -- the biasCor z cut reads them."""
    z = _resolved(zbinuser=[0.01, 0.05, 0.2, 0.6])
    assert (z.nzbin, z.zmin, z.zmax) == (3, 0.01, 0.6)
    assert _resolved(nlogzbin=3, zmin=0.01, zmax=1.0).nzbin == 3


def test_resolve_is_pure_and_repeatable():
    """It returns a new config rather than editing the caller's, and resolving an
    already-resolved config changes nothing."""
    cfg = BBCConfig(nlogzbin=3, zmin=0.01, zmax=1.0)
    once = _resolve(cfg)
    assert cfg.nzbin == 0                      # caller's config untouched
    assert _resolve(once) == once


@pytest.mark.parametrize("zbins,match", [
    # conflict_check requires zmin>0 and zmax<9 unless zbinuser gives the edges
    (dict(nzbin=4, zmin=0.0, zmax=0.8), "zmin and zmax"),
    (dict(nlogzbin=10, zmin=0.0, zmax=1.0), "zmin and zmax"),
    # setup_BININFO_userz aborts on Nsplit<=1
    (dict(zbinuser=[0.1]), "two bin edges"),
    (dict(), "no redshift binning"),
])
def test_resolve_rejects_what_salt2mu_rejects(zbins, match):
    with pytest.raises(SNANAError, match=match):
        _resolved(**zbins)


def _biascor_cfg(**biascor):
    return BBCConfig(nzbin=10, zmin=0.1, zmax=1.0,
                     simfile_biascor=["x.FITRES"], **biascor)


@pytest.mark.parametrize("cfg,opt_biascor,fitflag", [
    # no biasCor file: SNANA leaves opt_biascor alone (ISFILE_BIASCOR, :21193), and
    # downstream code reads a non-zero value as "there is a biasCor"
    (BBCConfig(nzbin=10, zmin=0.1, zmax=1.0,
               fitflag_sigmb=1), 0, 1),
    # default mask; a 5D cell map forces fitflag_sigmb to 2 (:10634)
    (_biascor_cfg(), MASK_BIASCOR_DEFAULT, 0),
    # surveygroup forces SAMPLE (:21198); muCOVADD forces MAD (:10519)
    (_biascor_cfg(opt_biascor=MASK_BIASCOR_CELLMAP, surveygroup_biascor="DES"),
     MASK_BIASCOR_CELLMAP | MASK_BIASCOR_SAMPLE, 0),
    (_biascor_cfg(opt_biascor=MASK_BIASCOR_CELLMAP | MASK_BIASCOR_MUCOVADD),
     MASK_BIASCOR_CELLMAP | MASK_BIASCOR_MUCOVADD | MASK_BIASCOR_MAD, 0),
    # a 1D map rewrites a requested 2 back down to 1
    (BBCConfig(nzbin=10, zmin=0.1, zmax=1.0,
               simfile_biascor=["x.FITRES"], opt_biascor=MASK_BIASCOR_1DZAVG,
               fitflag_sigmb=2), MASK_BIASCOR_1DZAVG, 1),
])
def test_resolve_effective_values(cfg, opt_biascor, fitflag):
    """The rewrites everything downstream reads, pinned at the point they happen
    rather than through their effect on a fit."""
    r = _resolve(cfg)
    assert r.opt_biascor == opt_biascor
    assert r.fitflag_sigmb == fitflag


def test_resolve_applies_fixpar_all():
    """fixpar_all forces one z bin and no sigint iteration (SALT2mu.c:21939). It used
    to be applied while parsing, so a config built in Python never got it."""
    cfg = BBCConfig(nlogzbin=8, zmin=0.1, zmax=1.0,
                    fitflag_sigmb=1)
    cfg.fixpar_all = True
    r = _resolve(cfg)
    assert (r.nzbin, r.nlogzbin, r.zbinuser) == (1, 0, [])
    assert r.fitflag_sigmb == 0


def test_resolve_agrees_between_parsed_and_built(tmp_path):
    """The two ways a config reaches the fit must normalise identically -- they did
    not, which is what made the zbinuser range silently wrong for built configs."""
    path = tmp_path / "in.txt"
    path.write_text("zbinuser=0.2,0.4,0.6\n")
    parsed = _resolve(parse_bbc_input(str(path)))
    built = _resolved(zbinuser=[0.2, 0.4, 0.6])
    assert (parsed.nzbin, parsed.zmin, parsed.zmax) == (built.nzbin, built.zmin, built.zmax)



@pytest.mark.parametrize("floats,expect_floated,expect_alpha", [
    ({}, [], 0.0),                            # no u<N>: nothing floats, parval holds
    ({"alpha0": True}, ["alpha0"], 0.14),     # u1=1: alpha floats from its own default
])
def test_floated_and_defaults_follows_snana(floats, expect_floated, expect_alpha):
    """SNANA inits every float flag to 0 (SALT2mu.c:6701), so a config with no u<N>
    floats nothing and its nuisances hold parval -- 0 unless p<N> was given (:22354).
    SALT2mu.exe on such a config prints `alpha0 0.0000 fixed`; jax used to fall back to
    floating alpha0/beta0, fitting them to 0.158/3.487 and moving MUDIF by 0.13."""
    cfg = BBCConfig(nzbin=10, zmin=0.1, zmax=1.0)
    cfg.nuisance_float.update(floats)
    floated, defaults = floated_and_defaults(SALTModel, cfg)
    assert floated == expect_floated
    assert defaults["alpha0"] == expect_alpha
    assert defaults["beta0"] == 0.0           # fixed either way here
    assert defaults["logmass_cen"] == 10.0    # parval[IPAR_LOGMASS_CEN] (:6708)


def test_fixpar_all_freezes_the_h11_coefficients_at_their_parval():
    """fixpar_all holds every fitted parameter at SNANA's parval (SALT2mu.c:21939),
    and set_defaults leaves parval[18..20] at 0.1 rather than 0 (:6710). BBC's own
    parameters take those values from the same table their bounds come from, so
    halving that table cannot silently move them."""
    cfg = BBCConfig(nzbin=10, zmin=0.1, zmax=1.0, use_h11=True, fixpar_all=True)
    _, frozen = floated_and_defaults(SALTModel, cfg)
    assert frozen["H11mucc1"] == 0.1
    assert frozen["H11mucc2"] == 0.1
    assert frozen["H11sigcc0"] == 0.1
    assert frozen["H11mucc0"] == 0.0 and frozen["H11sigcc2"] == 0.0
    assert frozen["scalePCC"] == 0.0          # parval[IPAR_scalePCC] (:6701)


@pytest.mark.parametrize("error_column", ["DLMAGERR", "MU_LCFITERR", "MUERR_LCFIT"])
def test_bayesn_distance_error_spellings(tmp_path, error_column):
    """BBC preserves the fitted distance covariance for each supported format."""
    from pathlib import Path
    from jax_snlc_core.models import BayeSNModel

    source = Path("tests/fixtures/snana/bbc_des_bayesn_data/FITRES.TEXT")
    cfg = _resolve(BBCConfig(model_lcfit="BAYESN", zmin=0.05, zmax=0.21, nzbin=4))
    reference = read_tablevar([str(source)], EVENT_TYPE_DATA, cfg, BayeSNModel)
    with open_text(source) as stream:
        lines = stream.read().splitlines()
    for i, line in enumerate(lines):
        if line.startswith("VARNAMES:"):
            names = line.split()
            names = [error_column if n == "MUERR_LCFIT" else n for n in names]
            lines[i] = " ".join(names)
    path = tmp_path / "bayesn.FITRES"
    path.write_text("\n".join(lines) + "\n")
    actual = read_tablevar([str(path)], EVENT_TYPE_DATA, cfg, BayeSNModel)
    np.testing.assert_array_equal(actual.lcpar["MU"], reference.lcpar["MU"])
    np.testing.assert_array_equal(actual.covmat_fit, reference.covmat_fit)
    assert np.all(actual.covmat_fit[:, 0, 0] > 0)
