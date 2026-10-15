"""HOSTLIB GROUPID-matched selection: SNe forced to galaxies in their
SIMLIB field's allowed GROUPID set.

Closes the largest remaining production-feature gap. Constructs a
GROUPID-tagged HOSTLIB + a GROUPID-tagged SIMLIB at test time, runs the
simulator, and verifies every SN's selected galaxy has the GROUPID
permitted by its field — exercising
``_select_with_groupid`` (~120 lines that no other test
reaches).
"""

from pathlib import Path

import numpy as np
import pytest

from jax_snlc_sim import run as sim_run
from jax_snlc_sim.survey.simlib_reader import read_simlib


def _write_groupid_hostlib(out: Path, *, n_per_group: int = 60,
                            seed: int = 91) -> None:
    """3 GROUPIDs (1, 2, 3) × n_per_group galaxies. Photometry/Sersic
    follow the same toy distributions as JAX_TEST.HOSTLIB."""
    rng = np.random.default_rng(seed)
    rows = []
    galid = 300_000
    centres = [(10.0, -30.0), (180.0, -45.0)]
    for groupid in (1, 2, 3):
        for _ in range(n_per_group):
            z = rng.uniform(0.04, 0.45)
            logmass = rng.uniform(8.5, 11.5)
            log_sfr = -1.0 + 0.3 * (logmass - 10.0) + rng.normal(0, 0.4)
            log_ssfr = log_sfr - logmass
            base_r = 17.0 + 5.0 * np.log10(z / 0.05)
            colour = rng.normal(0.5, 0.2) - 0.3 * (log_ssfr + 10.0)
            ra, dec = centres[rng.integers(0, 2)]
            zphot = z + rng.normal(0, 0.03)
            row = " ".join([
                f"{galid}",
                f"{z:.4f}", f"{zphot:.4f}", "0.0300",
                f"{ra:.4f}", f"{dec:.4f}",
                f"{base_r + 1.5 + colour:.3f}",
                f"{base_r + 0.7 + 0.5 * colour:.3f}",
                f"{base_r:.3f}",
                f"{base_r - 0.3:.3f}",
                f"{base_r - 0.5:.3f}",
                f"{base_r - 0.7:.3f}",
                f"{rng.uniform(0.3, 1.5):.3f}",
                f"{rng.uniform(0.2, 0.8):.3f}",
                f"{rng.uniform(0.8, 4.0):.3f}",
                f"{rng.uniform(0.0, 180.0):.2f}",
                f"{groupid}",
                f"{logmass:.3f}", f"{logmass:.3f}", "0.100",
                f"{log_sfr:.3f}", f"{log_ssfr:.3f}",
            ])
            rows.append("GAL: " + row)
            galid += 1

    header = [
        "DOCUMENTATION:",
        "    PURPOSE: HOSTLIB with GROUPID column for jax-snana parity tests",
        "    USAGE_KEY: HOSTLIB_FILE",
        "DOCUMENTATION_END:",
        "",
        ("VARNAMES: GALID ZTRUE ZPHOT ZPHOT_ERR RA_GAL DEC_GAL "
         "u_obs g_obs r_obs i_obs z_obs Y_obs "
         "a0_Sersic b0_Sersic n0_Sersic a_rot GROUPID "
         "LOGMASS LOGMASS_OBS LOGMASS_ERR LOG_SFR LOGsSFR"),
    ]
    out.write_text("\n".join(header + rows) + "\n")


def _write_groupid_simlib(out: Path, src_simlib: Path,
                          key: str = "GROUPID_HOSTLIB") -> None:
    """Take the bundled SIMLIB and inject GROUPID_HOSTLIB onto its two
    LIBIDs: LIBID 0 → group {1, 2}, LIBID 1 → group {3}."""
    src = src_simlib.read_text().splitlines()
    out_lines = []
    for line in src:
        out_lines.append(line)
        if line.startswith("LIBID: 0"):
            out_lines.append(f"{key}: 1,2")
        elif line.startswith("LIBID: 1"):
            out_lines.append(f"{key}: 3")
    out.write_text("\n".join(out_lines) + "\n")


@pytest.mark.parametrize("key", ["GROUPID_HOSTLIB", "HOSTLIB_GROUPID"])
def test_groupid_key_spellings(tmp_path, lsst_simlib_path, key):
    """Both spellings SNANA accepts (snlc_sim.c:22249) reach the entry."""
    simlib = tmp_path / "groupid.SIMLIB"
    _write_groupid_simlib(simlib, lsst_simlib_path, key=key)
    data = read_simlib(str(simlib), cache_dir=tmp_path / "cache")
    assert [f.groupid_hostlib for f in data.fields] == [(1, 2), (3,)]


def _write_logmass_only_wgtmap(out: Path) -> None:
    """LOGMASS-only WGTMAP — no SN vars, host-first compatible."""
    rows = ["VARNAMES_WGTMAP: LOGMASS WGT SNMAGSHIFT"]
    for lm in (8.0, 9.0, 10.0, 11.0, 12.0):
        rows.append(f"WGT: {lm:.2f} 1.0 0.0")
    out.write_text("\n".join(rows) + "\n")


def _write_input(out: Path, *, hostlib: Path, simlib: Path,
                  wgtmap: Path, use_groupid: bool = True,
                  mskopt: int = 258) -> None:
    """DZTOL stays above half the largest redshift gap within a GROUPID,
    as an empty window aborts."""
    groupid_key = "USE_SIMLIB_GROUPID:  1\n" if use_groupid else ""
    out.write_text(groupid_key + f"""\
GENVERSION:  JAX_TEST_GROUPID
NGENTOT_LC:  300
FORMAT_MASK:  32
GENTYPE:  1
SNTYPES_Ia:  1  101
CIDRAN_MIN:  0
CIDRAN_MAX:  3000000
GENMODEL:  $JAX_TEST_FIXTURE_ROOT/models/SALT3.K21
GENPEAK_SALT2x1:    0.0
GENRANGE_SALT2x1:  -2.0  2.0
GENSIGMA_SALT2x1:   1.0  1.0
GENPEAK_SALT2c:     0.0
GENRANGE_SALT2c:   -0.2  0.3
GENSIGMA_SALT2c:    0.05  0.05
GENPEAK_SALT2ALPHA: 0.14
GENPEAK_SALT2BETA:  3.1
OMEGA_MATTER:  0.3150
OMEGA_LAMBDA:  0.6850
w0_LAMBDA:  -1.0000
wa_LAMBDA:  0
GENFILTERS:  ugrizY
KCOR_FILE:  $JAX_TEST_FIXTURE_ROOT/kcor/JAX_TEST_LSST.fits
SIMLIB_FILE:  {simlib}
SIMLIB_MINSEASON:  30
SMEARFLAG_FLUX:  1
SMEARFLAG_ZEROPT:  1
HOSTLIB_FILE:        {hostlib}
HOSTLIB_WGTMAP_FILE: {wgtmap}
HOSTLIB_MSKOPT:      {mskopt}
HOSTLIB_DZTOL:       0.02,0.05,0
HOSTLIB_STOREPAR:    GROUPID,LOGMASS,LOG_SFR,LOGsSFR
GENRANGE_REDSHIFT:   0.05  0.40
GENRANGE_MJD:        60500.0  61500.0
GENRANGE_PEAKMJD:    60540.0  61460.0
GENSIGMA_REDSHIFT:   0.001
GENRANGE_TREST:      -20.0  +50.0
GENRANGE_RA:         -180.0 +180.0
GENRANGE_DECL:       -90.0  +90.0
SOLID_ANGLE:         0.0150
OPT_MWEBV:           0
DNDZ:  POWERLAW  2.5E-5  1.5
RANSEED:  333333
""")


@pytest.fixture
def groupid_fixture(tmp_path, lsst_simlib_path):
    hostlib = tmp_path / "JAX_TEST_GROUPID.HOSTLIB"
    _write_groupid_hostlib(hostlib)
    simlib = tmp_path / "JAX_TEST_GROUPID.SIMLIB"
    _write_groupid_simlib(simlib, lsst_simlib_path)
    wgtmap = tmp_path / "LOGMASS_FLAT.WGTMAP"
    _write_logmass_only_wgtmap(wgtmap)
    inp = tmp_path / "input.input"
    _write_input(inp, hostlib=hostlib, simlib=simlib, wgtmap=wgtmap)
    return inp


@pytest.fixture
def groupid_fixture_no_key(tmp_path, lsst_simlib_path):
    """Same GROUPID-tagged HOSTLIB and SIMLIB, without USE_SIMLIB_GROUPID."""
    hostlib = tmp_path / "JAX_TEST_GROUPID.HOSTLIB"
    _write_groupid_hostlib(hostlib)
    simlib = tmp_path / "JAX_TEST_GROUPID.SIMLIB"
    _write_groupid_simlib(simlib, lsst_simlib_path)
    wgtmap = tmp_path / "LOGMASS_FLAT.WGTMAP"
    _write_logmass_only_wgtmap(wgtmap)
    inp = tmp_path / "input_nokey.input"
    _write_input(inp, hostlib=hostlib, simlib=simlib, wgtmap=wgtmap,
                 use_groupid=False)
    return inp


def test_groupid_selection_restricts_hosts_to_field_set(groupid_fixture):
    """Every selected SN host's GROUPID is in its field's allowed set.

    LIBID 0 (FIELD=TOY_A) only matches GROUPIDs {1, 2}.
    LIBID 1 (FIELD=TOY_B) only matches GROUPID {3}.
    """
    cfg = dict(
        n_sne=300, random_seed=12321,
    )
    sim = sim_run.prepare(str(groupid_fixture), **cfg)
    resources = sim
    results, _, _ = sim_run.run_simulation(sim)
    params = results["parameters"]
    field_idx = np.asarray(params["field_idx"]).astype(int)
    host_groupid = np.asarray(params["host_GROUPID"]).astype(int)

    # Sanity: both fields drew SNe.
    n_libid0 = int((field_idx == 0).sum())
    n_libid1 = int((field_idx == 1).sum())
    assert n_libid0 > 30 and n_libid1 > 30, (
        f"unbalanced field draws: LIBID0={n_libid0}, LIBID1={n_libid1}"
    )

    # a window with no galaxy of the set aborts, so no host can leak
    assert np.isin(host_groupid[field_idx == 0], [1, 2]).all()
    assert (host_groupid[field_idx == 1] == 3).all()


# Note: behaviour when a SIMLIB field demands a GROUPID set with NO
# matching HOSTLIB galaxies is undefined in SNANA (it falls back to the
# default cumsum), so we don't assert anything specific for that scenario.


def test_groupid_headers_inert_without_the_key(groupid_fixture_no_key):
    """SNANA reads the SIMLIB's GROUPID header only when USE_SIMLIB_GROUPID is
    set (RDFLAG_GROUPID, snlc_sim.c:22115); without it the header is inert and
    hosts are drawn from the whole catalogue.

    jax previously applied GROUPID matching whenever the header was present.
    """
    cfg = dict(
        n_sne=300, random_seed=12321,
    )
    sim = sim_run.prepare(str(groupid_fixture_no_key), **cfg)
    resources = sim
    assert resources.hostlib.selector.field_row_block is None
    results, _, _ = sim_run.run_simulation(sim)
    groupids = np.unique(np.asarray(results["parameters"]["sim_hostlib_GROUPID"]))
    assert set(groupids) == {1, 2, 3}


def test_groupid_without_hostlib_column_aborts(tmp_path, lsst_simlib_path,
                                                jax_test_hostlib_path):
    """SIMLIB demands a GROUPID the HOSTLIB has no column for -> fatal, as in
    MATCH_GROUPID_HOSTLIB (sntools_host.c:6718)."""
    from jax_snlc_core.utils.error_handling import SNANAError

    simlib = tmp_path / "JAX_TEST_GROUPID.SIMLIB"
    _write_groupid_simlib(simlib, lsst_simlib_path)
    wgtmap = tmp_path / "LOGMASS_FLAT.WGTMAP"
    _write_logmass_only_wgtmap(wgtmap)
    inp = tmp_path / "input_nocol.input"
    _write_input(inp, hostlib=jax_test_hostlib_path, simlib=simlib,
                 wgtmap=wgtmap)
    with pytest.raises(SNANAError, match="GROUPID"):
        sim_run.prepare(str(inp), **dict(n_sne=10))


@pytest.fixture
def groupid_fixture_marginal(tmp_path, lsst_simlib_path):
    """MSKOPT bit 16 (SN2GAL_Z) switches selection to the marginal-CDF
    mode; GROUPID must be enforced there too."""
    hostlib = tmp_path / "JAX_TEST_GROUPID.HOSTLIB"
    _write_groupid_hostlib(hostlib)
    simlib = tmp_path / "JAX_TEST_GROUPID.SIMLIB"
    _write_groupid_simlib(simlib, lsst_simlib_path)
    wgtmap = tmp_path / "LOGMASS_FLAT.WGTMAP"
    _write_logmass_only_wgtmap(wgtmap)
    inp = tmp_path / "input_marginal.input"
    _write_input(inp, hostlib=hostlib, simlib=simlib, wgtmap=wgtmap,
                 mskopt=258 + 16)
    return inp


def test_groupid_enforced_in_marginal_cdf_mode(groupid_fixture_marginal):
    """With MSKOPT bit 16 the selector samples from per-GROUPID-set marginal
    CDFs, so field GROUPID constraints hold exactly as in the binned mode
    (SNANA enforces GROUPID in the weight search unconditionally,
    sntools_host.c:6379-6399)."""
    cfg = dict(
        n_sne=300, random_seed=12321,
    )
    sim = sim_run.prepare(str(groupid_fixture_marginal), **cfg)
    resources = sim
    results, _, _ = sim_run.run_simulation(sim)
    params = results["parameters"]
    field_idx = np.asarray(params["field_idx"]).astype(int)
    host_groupid = np.asarray(params["host_GROUPID"]).astype(int)

    libid0_groups = host_groupid[field_idx == 0]
    libid1_groups = host_groupid[field_idx == 1]
    assert len(libid0_groups) > 0 and len(libid1_groups) > 0
    # Marginal-CDF selection draws from the filtered weights directly, so
    # conformance is exact.
    assert np.isin(libid0_groups, [1, 2]).all()
    assert (libid1_groups == 3).all()
