"""HOSTLIB with NBR_LIST: integration test exercising the
``_compute_ddlr_and_sort`` path through the full simulator.

A separate small HOSTLIB is constructed at test time with proper RA/DEC
scatter and a NBR_LIST column listing each galaxy's 5 nearest by-sky
neighbours. Running the simulator with that HOSTLIB activates the
HOSTGAL2 / NMATCH machinery in ``HostSelector.compute_neighbor_info``
plus all of ``_compute_ddlr_and_sort``, which had no integration-test
coverage before.
"""

from pathlib import Path

import numpy as np
import pytest

from jax_snlc_sim import run as sim_run


def _write_nbr_hostlib(out: Path, *, n_gal: int = 80, k_nbr: int = 5,
                        seed: int = 31) -> None:
    """Write a HOSTLIB with per-galaxy spatial scatter + NBR_LIST.

    For each galaxy, compute its k nearest by-sky neighbours (excluding
    self) and write them as a comma-separated NBR_LIST entry. Galaxies
    are spread around two SIMLIB pointings with ~30″ RMS scatter.
    """
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.04, 0.45, n_gal)

    # Two field centres mirroring the SIMLIB used downstream.
    centres = np.array([(10.0, -30.0), (180.0, -45.0)])
    field_idx = rng.integers(0, 2, n_gal)
    sigma_arcsec = 30.0  # per-axis RMS scatter
    sigma_deg = sigma_arcsec / 3600.0
    ra = centres[field_idx, 0] + rng.normal(0, sigma_deg, n_gal)
    dec = centres[field_idx, 1] + rng.normal(0, sigma_deg, n_gal)

    # Compute k nearest by-sky neighbour indices for each galaxy.
    nbr_indices = np.full((n_gal, k_nbr), -1, dtype=int)
    for i in range(n_gal):
        same_field = (field_idx == field_idx[i])
        same_field[i] = False  # exclude self
        candidates = np.where(same_field)[0]
        if len(candidates) == 0:
            continue
        d_ra = ra[candidates] - ra[i]
        d_dec = dec[candidates] - dec[i]
        dist = np.sqrt(d_ra**2 + d_dec**2)
        order = candidates[np.argsort(dist)]
        nbr_indices[i, :min(k_nbr, len(order))] = order[:k_nbr]

    a_sersic = rng.uniform(1.0, 3.0, n_gal)
    b_over_a = rng.uniform(0.5, 1.0, n_gal)
    a_rot = rng.uniform(0.0, 180.0, n_gal)
    logmass = rng.uniform(8.5, 11.5, n_gal)
    log_sfr = -1.0 + 0.3 * (logmass - 10.0) + rng.normal(0, 0.4, n_gal)
    log_ssfr = log_sfr - logmass

    # Match column ordering of bundled HOSTLIB so engine outputs flow.
    base_r = 17.0 + 5.0 * np.log10(z / 0.05)
    colour = rng.normal(0.5, 0.2, n_gal) - 0.3 * (log_ssfr + 10.0)

    rows = []
    for i in range(n_gal):
        nbr_str = ",".join(
            str(int(j)) for j in nbr_indices[i] if j >= 0
        ) or "-1"
        gid = 200_000 + i
        zphot = z[i] + rng.normal(0, 0.03)
        zphot_err = 0.03
        row = " ".join([
            f"{gid}",
            f"{z[i]:.4f}", f"{zphot:.4f}", f"{zphot_err:.4f}",
            f"{ra[i]:.6f}", f"{dec[i]:.6f}",
            f"{base_r[i] + 1.5 + colour[i]:.3f}",  # u
            f"{base_r[i] + 0.7 + 0.5 * colour[i]:.3f}",  # g
            f"{base_r[i]:.3f}",  # r
            f"{base_r[i] - 0.3:.3f}",  # i
            f"{base_r[i] - 0.5:.3f}",  # z
            f"{base_r[i] - 0.7:.3f}",  # Y
            f"{a_sersic[i]:.3f}",
            f"{a_sersic[i] * b_over_a[i]:.3f}",
            "1.0",  # n_sersic
            f"{a_rot[i]:.2f}",
            f"{logmass[i]:.3f}", f"{logmass[i]:.3f}", "0.100",
            f"{log_sfr[i]:.3f}", f"{log_ssfr[i]:.3f}",
            nbr_str,
        ])
        rows.append("GAL: " + row)

    header = [
        "DOCUMENTATION:",
        "    PURPOSE: Synthetic HOSTLIB with NBR_LIST for jax-snana e2e tests",
        "    USAGE_KEY: HOSTLIB_FILE",
        "DOCUMENTATION_END:",
        "",
        ("VARNAMES: GALID ZTRUE ZPHOT ZPHOT_ERR RA_GAL DEC_GAL "
         "u_obs g_obs r_obs i_obs z_obs Y_obs "
         "a0_Sersic b0_Sersic n0_Sersic a_rot "
         "LOGMASS LOGMASS_OBS LOGMASS_ERR LOG_SFR LOGsSFR NBR_LIST"),
    ]
    out.write_text("\n".join(header + rows) + "\n")


def _write_input(out: Path, hostlib_path: Path) -> None:
    """Write a SNANA input file pointing at the NBR_LIST HOSTLIB. DZTOL
    stays above half the largest redshift gap between its 80 galaxies, as an
    empty window aborts."""
    out.write_text(f"""\
# HOSTLIB-NBR end-to-end fixture: exercises HOSTGAL2 / NBR_LIST processing.
GENVERSION:  JAX_TEST_HOSTLIB_NBR
NGENTOT_LC:  120
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
SIMLIB_FILE:  $JAX_TEST_FIXTURE_ROOT/simlib/JAX_TEST_LSST.SIMLIB
SIMLIB_MINSEASON:  30
SMEARFLAG_FLUX:  1
SMEARFLAG_ZEROPT:  1
HOSTLIB_FILE:        {hostlib_path}
HOSTLIB_MSKOPT:      258
HOSTLIB_DZTOL:       0.02,0.05,0
HOSTLIB_MAXDDLR:     4.0
HOSTLIB_MAXDDLR2:    7.0
HOSTLIB_STOREPAR:    LOGMASS,LOG_SFR,LOGsSFR
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
RANSEED:  122459
""")


@pytest.fixture
def nbr_fixture(tmp_path):
    hostlib = tmp_path / "JAX_TEST_NBR.HOSTLIB"
    _write_nbr_hostlib(hostlib)
    inp = tmp_path / "input.input"
    _write_input(inp, hostlib)
    return inp


@pytest.fixture(scope="module")
def nbr_head(tmp_path_factory):
    """(HEAD columns, hostlib path) from one 300-SN run with GALID_UNIQUE set.

    GALID_UNIQUE only adds the *_OBJID_UNIQUE columns and draws from its own key
    stream (sampler.py:1746), so every other column matches a run without it and
    the two column tests can share this one.
    """
    import os
    from astropy.io import fits
    from jax_snlc_core.io.fits_path import resolve_fits_path

    d = tmp_path_factory.mktemp("nbr")
    hostlib = d / "JAX_TEST_NBR.HOSTLIB"
    _write_nbr_hostlib(hostlib)
    inp = d / "input.input"
    _write_input(inp, hostlib)
    inp.write_text(inp.read_text() + "\nHOSTLIB_GALID_UNIQUE: 1\n")

    cfg = dict(
        n_sne=300, random_seed=12345)
    sim = sim_run.prepare(str(inp), **cfg)
    simulation = None if sim is None else sim.config
    res, sel, spectra_cache = sim_run.run_simulation(sim)
    out = str(d / "out")
    os.makedirs(out, exist_ok=True)
    sim_run.save_snana_outputs(sim, res, "N", sel, spectra_cache=spectra_cache, output_dir=out)
    with fits.open(resolve_fits_path(f"{out}/N/N_HEAD.FITS")) as h:
        cols = {n: np.asarray(h[1].data[n]) for n in h[1].data.names}
    return cols, hostlib


def test_host_first_reports_the_same_neighbours_as_host_after(nbr_fixture, tmp_path):
    """A VARIATION following a host quantity selects the host before the SN
    parameters are drawn. The same galaxy is drawn from the same streams in
    either flow, so the DDLR sort, its counters and the second host must
    come out identical, and the written outputs carry the counters."""
    import os
    first_input = tmp_path / "first.input"
    first_input.write_text(nbr_fixture.read_text()
                           + "VARIATION: GENPEAK_SALT2x1 LOGMASS POLY -0.30 PIVOT 10.0\n")
    hosts = {}
    for flow, inp in (("after", nbr_fixture), ("first", first_input)):
        cfg = dict(n_sne=120, random_seed=12345)
        sim = sim_run.prepare(str(inp), **cfg)
        simulation = None if sim is None else sim.config
        res, sel, spectra_cache = sim_run.run_simulation(sim)
        hosts[flow] = res["parameters"]
        out = tmp_path / flow
        os.makedirs(out)
        sim_run.save_snana_outputs(sim, res, "N", sel, spectra_cache=spectra_cache, output_dir=str(out))

    for column in ("host_GALID", "host_NMATCH", "host_NMATCH2", "host_DDLR",
                   "host_HOSTGAL2_GALID", "host_HOSTGAL2_DDLR"):
        np.testing.assert_array_equal(np.asarray(hosts["first"][column]),
                                      np.asarray(hosts["after"][column]), err_msg=column)
    assert (np.asarray(hosts["first"]["host_HOSTGAL2_GALID"]) > 0).any()
    readme = next((tmp_path / "first" / "N").glob("*.README")).read_text()
    assert "HOST_NMATCH:" in readme


def test_nbr_hostlib_hostgal_and_hostgal2_invariants(nbr_fixture):
    """One sim run, three NBR_LIST invariants:

    1. NMATCH ≤ NMATCH2 by definition (NMATCH2 has a larger DDLR cut).
    2. NMATCH ≥ 1 for every SN (the fixture places SNe near a host).
    3. At least some SNe have a valid HOSTGAL2 within MAXDDLR2 (proves
       the second-closest neighbour selection actually fires).
    4. Where both DDLRs are valid, host_DDLR ≤ host_HOSTGAL2_DDLR (the
       JIT sort produced ascending order).
    """
    cfg = dict(
        n_sne=120, random_seed=12345,
    )
    sim = sim_run.prepare(str(nbr_fixture), **cfg)
    simulation = None if sim is None else sim.config
    results, _, _ = sim_run.run_simulation(sim)
    params = results["parameters"]

    # 1+2: NMATCH/NMATCH2 invariants.
    assert "host_NMATCH" in params, f"missing host_NMATCH; keys: {sorted(params)}"
    assert "host_NMATCH2" in params
    nmatch = np.asarray(params["host_NMATCH"])
    nmatch2 = np.asarray(params["host_NMATCH2"])
    assert np.all(nmatch <= nmatch2)
    assert int(nmatch.min()) >= 1

    assert "host_DDLR" in params and "host_HOSTGAL2_DDLR" in params, (
        f"missing DDLR columns; keys: {sorted(params)}"
    )

    ddlr1 = np.asarray(params["host_DDLR"])
    ddlr2 = np.asarray(params["host_HOSTGAL2_DDLR"])

    # 3: at least some SNe have a valid HOSTGAL2 within MAXDDLR2 (= 7.0).
    valid_2 = (ddlr2 > 0) & (ddlr2 < 7.0)
    assert valid_2.sum() > 0, (
        f"no valid HOSTGAL2 found across {len(ddlr2)} SNe; "
        f"DDLR2 stats: min={ddlr2.min():.2f}, max={ddlr2.max():.2f}"
    )

    # 4: DDLR sort ascending wherever both are valid.
    valid_both = valid_2 & (ddlr1 > 0) & (ddlr1 < 7.0)
    if valid_both.sum() == 0:
        return  # nothing to check beyond #1-#3
    assert np.all(ddlr1[valid_both] <= ddlr2[valid_both] + 1e-3), (
        f"DDLR sort violated: "
        f"{(ddlr1[valid_both] > ddlr2[valid_both]).sum()} SNe with "
        f"HOSTGAL_DDLR > HOSTGAL2_DDLR"
    )


def test_nbr_hostgal2_specz_photoz(nbr_head):
    """HOSTGAL/HOSTGAL2 photo-z carry the same ZDIF shift and each spec-z is the
    measured SN z (true-match host) or that host's catalog ZTRUE
    (GEN_SNHOST_ZPHOT / snlc_sim.c:26071-26078)."""
    c, hostlib = nbr_head
    cat = {}
    for line in hostlib.read_text().splitlines():
        if line.startswith("GAL:"):
            p = line.split()
            cat[int(p[1])] = (float(p[2]), float(p[3]))    # ZTRUE, ZPHOT
    o1, o2 = c['HOSTGAL_OBJID'], c['HOSTGAL2_OBJID']
    pz1, pz2 = c['HOSTGAL_PHOTOZ'], c['HOSTGAL2_PHOTOZ']
    sz1, sz2 = c['HOSTGAL_SPECZ'], c['HOSTGAL2_SPECZ']
    zhel = c['REDSHIFT_HELIO']
    czt = lambda o: np.array([cat[int(x)][0] for x in o])
    czp = lambda o: np.array([cat[int(x)][1] for x in o])
    has2 = o2 > 0
    # HOSTGAL2 photo-z populated unless the shifted catalogue value is negative
    zdif = c['SIM_REDSHIFT_HELIO'] - czt(c['SIM_HOSTLIB_GALID'])
    assert has2.sum() > 20
    np.testing.assert_array_equal(pz2[has2] > -8,
                                  czp(o2[has2]) + zdif[has2] >= 0)
    both = has2 & (pz1 > -8) & (pz2 > -8)
    zdif0, zdif1 = pz1[both] - czp(o1[both]), pz2[both] - czp(o2[both])
    assert np.max(np.abs(zdif0 - zdif1)) < 1e-4                       # same ZDIF for primary + secondary
    for sz, o in ((sz1[has2], o1[has2]), (sz2[has2], o2[has2])):
        is_snz = np.isclose(sz, zhel[has2], atol=1e-4)
        is_cat = np.isclose(sz, czt(o), atol=1e-4)
        assert np.all(is_snz | is_cat)                               # spec-z = measured SN z or catalog ZTRUE


def test_nbr_hostgal2_objid_unique(nbr_head):
    """HOSTGAL2_OBJID_UNIQUE = CID*57 + [0,56], distinct from HOSTGAL_OBJID_UNIQUE
    (set_GALID_UNIQUE collision avoidance, sntools_host.c:9060)."""
    c, _ = nbr_head
    u1, u2 = c['HOSTGAL_OBJID_UNIQUE'], c['HOSTGAL2_OBJID_UNIQUE']
    o2 = c['HOSTGAL2_OBJID']
    cid = np.array([int(str(s).strip()) for s in c['SNID']], dtype=np.int64)
    has2 = o2 > 0
    off2 = u2[has2] - cid[has2] * 57
    assert has2.sum() > 20 and np.all(u2[has2] >= 0)                 # populated
    assert np.all(off2 >= 0) and np.all(off2 <= 56)                  # CID*57 + [0,56]
    assert np.all(u2[has2] != u1[has2])                             # distinct from primary


def test_sim_hostlib_galid_names_the_drawn_host(nbr_head):
    """SIM_HOSTLIB_GALID is the drawn galaxy (SNHOSTGAL.GALID, snlc_sim.c:26128),
    so its HOSTLIB row gives every SIM_HOSTLIB(...) value, while HOSTGAL_OBJID
    names the DDLR-ranked host and differs for some events."""
    c, hostlib = nbr_head
    varnames = None
    rows = {}
    for line in hostlib.read_text().splitlines():
        if line.startswith("VARNAMES:"):
            varnames = line.split()[1:]
        elif line.startswith("GAL:"):
            p = line.split()[1:]
            rows[int(p[0])] = dict(zip(varnames, p))
    galid = c['SIM_HOSTLIB_GALID']
    for name in ("LOGMASS", "LOG_SFR", "LOGsSFR"):
        column = next(k for k in c if k in (f"SIM_HOSTLIB({name})",
                                            f"SIM_HOSTLIB({name}_TRUE)"))
        np.testing.assert_allclose(
            c[column], [float(rows[int(g)][name]) for g in galid],
            rtol=0, atol=1e-6, err_msg=column)
    assert np.any((c['HOSTGAL_OBJID'] > 0) & (c['HOSTGAL_OBJID'] != galid))


def test_no_nbr_list_applies_ddlr_cuts(tmp_path):
    """Without a NBR_LIST the single candidate still faces the DDLR cuts.

    SNANA counts a match only when DDLR clears HOSTLIB_MAXDDLR (NMATCH) or
    MAXDDLR2 (NMATCH2) (SORT_SNHOSTGAL_byDDLR, sntools_host.c:8830), so a SN
    far from its galaxy is reported hostless even though the galaxy still
    drives generation. jax used to report a match unconditionally here.
    """
    fixture = Path("tests/fixtures/snana/hostlib_e2e/input.input")
    config = dict(n_sne=200)
    sim = sim_run.prepare(str(fixture), **config)
    simulation = None if sim is None else sim.config
    cfg = simulation.host
    assert not cfg.hostlib_file.endswith("NBR.HOSTLIB"), "fixture must lack a NBR_LIST"

    results, _, _ = sim_run.run_simulation(sim)
    params = results['parameters']
    ddlr = np.asarray(params['host_DDLR'])
    nmatch = np.asarray(params['host_NMATCH'])
    nmatch2 = np.asarray(params['host_NMATCH2'])

    np.testing.assert_array_equal(nmatch, (ddlr < cfg.maxddlr).astype(nmatch.dtype))
    np.testing.assert_array_equal(nmatch2, (ddlr < cfg.maxddlr2).astype(nmatch2.dtype))
    # The cuts must actually bite, or the assertions above are vacuous.
    assert (ddlr >= cfg.maxddlr2).any(), "no SN beyond MAXDDLR2 — test is vacuous"
