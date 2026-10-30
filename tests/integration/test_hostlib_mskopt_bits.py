"""HOSTLIB_MSKOPT bit-level integration tests.

Targets the bits that lacked any dedicated test coverage:
- bit 8 (SN2GAL_RADEC): override SN RA/DEC with host's
- bit 32 (USEONCE): every galaxy used at most once across the simulation
- bit 64 (USESNPAR): host SN params (x1, c) override the sampled values

- bits 8/16/512 (SN2GAL_RADEC, SN2GAL_Z, USEVPEC) and WRONGHOST_FILE: the
  host's say over the SN's redshift and velocity, against SNANA's output

(Bit 4 SNMAGSHIFT is exercised by ``parity/test_hostlib_snmagshift``.)
"""

import os
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from scipy.special import gammaincinv

from jax_snlc_core.cosmology import zcmb_to_zhelio, zhelio_to_zcmb
from jax_snlc_core.models.salt import compute_expected_x0_scale
from jax_snlc_core.snana_files import open_text
from jax_snlc_sim import run as sim_run
from jax_snlc_sim.output import save_snana_outputs
from jax_snlc_core.random import root_key


_HOSTLIB = "$JAX_TEST_FIXTURE_ROOT/hostlib/JAX_TEST.HOSTLIB"
_KCOR = "$JAX_TEST_FIXTURE_ROOT/kcor/JAX_TEST_LSST.fits"
_SIMLIB = "$JAX_TEST_FIXTURE_ROOT/simlib/JAX_TEST_LSST.SIMLIB"
_MODEL = "$JAX_TEST_FIXTURE_ROOT/models/SALT3.K21"


def _write_input(out: Path, *, mskopt: int, n_sne: int = 200,
                  redshift_max: float = 0.40, seed: int = 122459,
                  hostlib: str = _HOSTLIB) -> None:
    """Write a SALT3+HOSTLIB input file with the given HOSTLIB_MSKOPT."""
    out.write_text(f"""\
GENVERSION:  JAX_TEST_MSKOPT_BITS
NGENTOT_LC:  {n_sne}
FORMAT_MASK:  32
GENTYPE:  1
SNTYPES_Ia:  1  101
CIDRAN_MIN:  0
CIDRAN_MAX:  3000000
GENMODEL:  {_MODEL}
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
KCOR_FILE:  {_KCOR}
SIMLIB_FILE:  {_SIMLIB}
SIMLIB_MINSEASON:  30
SMEARFLAG_FLUX:  1
SMEARFLAG_ZEROPT:  1
HOSTLIB_FILE:        {hostlib}
HOSTLIB_MSKOPT:      {mskopt}
HOSTLIB_DZTOL:       0.0,0.05,0
HOSTLIB_STOREPAR:    LOGMASS,LOG_SFR,LOGsSFR
GENRANGE_REDSHIFT:   0.05  {redshift_max:.2f}
GENRANGE_MJD:        60500.0  61500.0
GENRANGE_PEAKMJD:    60540.0  61460.0
GENSIGMA_REDSHIFT:   0.001
GENRANGE_TREST:      -20.0  +50.0
GENRANGE_RA:         -180.0 +180.0
GENRANGE_DECL:       -90.0  +90.0
SOLID_ANGLE:         0.0150
OPT_MWEBV:           0
DNDZ:  POWERLAW  2.5E-5  1.5
RANSEED:  {seed}
""")


def _run(input_path: Path, n_sne: int = 200, seed: int = 122459):
    cfg = dict(
        n_sne=n_sne, random_seed=seed,
    )
    sim = sim_run.prepare(str(input_path), **cfg)
    simulation = None if sim is None else sim.config
    resources = sim
    results, _, _ = sim_run.run_simulation(sim)
    return sim, results


# ---------------------------------------------------------------------------
# Bit 8 — SN2GAL_RADEC
# ---------------------------------------------------------------------------

def test_sn2gal_radec_moves_the_sn_onto_its_host(tmp_path):
    """With bit 8 set, every SN sits on its host — at the position drawn within
    the galaxy, which is the centre plus the Sersic offset.

    load_SNHOST_GENLC_COORDS assigns GENLC.RA = SNHOSTGAL.RA_SN_DEG, not
    RA_GAL, so the SN is displaced from the centre by exactly HOSTGAL_SNSEP.
    Confirmed against a live SNANA run: SIM_RA - HOSTGAL_RA implies 0.7307"
    against a reported HOSTGAL_SNSEP of 0.7326".

    Without bit 8, SN RA/DEC are sampled within GENRANGE_RA/DECL — they almost
    never land near host values.
    """
    inp = tmp_path / "input.input"
    # 256 (verbose) + 2 (HOSTNOISE off) + 8 (SN2GAL_RADEC) = 266
    _write_input(inp, mskopt=266)
    _, results = _run(inp)
    params = results["parameters"]
    sn_ra = np.asarray(params["ra"])
    sn_dec = np.asarray(params["dec"])
    host_ra = np.asarray(params["host_RA_GAL"])
    host_dec = np.asarray(params["host_DEC_GAL"])
    snsep = np.asarray(params["host_SNSEP"])

    implied = np.hypot((sn_ra - host_ra) * np.cos(np.radians(host_dec)) * 3600.0,
                       (sn_dec - host_dec) * 3600.0)
    np.testing.assert_allclose(implied, snsep, atol=0.02)
    # the SN is on its host: within the major axis times the radius that
    # encloses MXINTFLUX_SNPOS = 0.99 of the Sersic flux
    with open_text(os.path.expandvars(_HOSTLIB)) as stream:
        rows = {int(line.split()[1]): line.split() for line in stream
                if line.startswith("GAL:")}
    a0, n0 = np.array([[float(rows[int(g)][13]), float(rows[int(g)][15])]
                       for g in np.asarray(params["host_GALID"])]).T
    r99 = (gammaincinv(2 * n0, 0.99) / gammaincinv(2 * n0, 0.5)) ** n0
    assert np.all(implied < a0 * r99)


# ---------------------------------------------------------------------------
# Bit 32 — USEONCE
# ---------------------------------------------------------------------------

def test_useonce_assigns_each_galaxy_at_most_once(tmp_path):
    """With USEONCE (bit 32) and a HOSTLIB much larger than the SN sample
    (5000 galaxies, 150 SNe), every galaxy is assigned to at most one SN.
    """
    inp = tmp_path / "input.input"
    _write_input(inp, mskopt=290, n_sne=150)
    _, results = _run(inp, n_sne=150)
    galid = np.asarray(results["parameters"]["host_GALID"]).astype(int)
    valid = galid > 0
    _, counts = np.unique(galid[valid], return_counts=True)
    duplicates = int((counts > 1).sum())
    assert duplicates == 0, (
        f"USEONCE violated: {duplicates} galaxies assigned to >1 SN "
        f"(max count {int(counts.max())})"
    )


def test_useonce_holds_across_batch_boundaries(tmp_path):
    """USEONCE must dedupe across batches, not just within one. With
    batch_size < n_sne the selector processes several batches; a galaxy
    taken in an earlier batch must not reappear in a later one."""
    inp = tmp_path / "input.input"
    _write_input(inp, mskopt=290, n_sne=120)
    cfg = dict(
        n_sne=120, random_seed=122459,
        batch_size=20)
    sim = sim_run.prepare(str(inp), **cfg)
    simulation = None if sim is None else sim.config
    resources = sim
    results, _, _ = sim_run.run_simulation(sim)
    galid = np.asarray(results["parameters"]["host_GALID"]).astype(int)
    valid = galid > 0
    _, counts = np.unique(galid[valid], return_counts=True)
    duplicates = int((counts > 1).sum())
    assert duplicates == 0, (
        f"USEONCE violated across batches: {duplicates} galaxies reused "
        f"(max count {int(counts.max())})"
    )


def test_useonce_takes_one_galaxy_per_event_with_a_partial_last_batch(tmp_path):
    """A batch count that does not divide the sample leaves a partial last
    batch, which the generation stages run at the full batch shape so the
    run compiles one shape. The events past the sample's end draw hosts of
    their own, so the run must still consume exactly one galaxy per reported
    event: if those extra draws were counted, they would take galaxies the
    reported events need and register as collisions against them.
    """
    inp = tmp_path / "input.input"
    n_sne = 210
    _write_input(inp, mskopt=290, n_sne=n_sne)
    cfg = dict(
        n_sne=n_sne, random_seed=122459,
        batch_size=50)          # 4 full batches and a tail of 10
    sim = sim_run.prepare(str(inp), **cfg)
    simulation = None if sim is None else sim.config
    resources = sim

    _, _, state = sim_run.generate_events(
        sim, n_sne, 0,
        key=root_key(simulation.run.random_seed),
        host_state=sim_run.initial_batch_state(resources, n_sne))

    n_taken = int(np.sum(state['reuse']['taken']))
    assert n_taken == n_sne, (
        f"USEONCE consumed {n_taken} galaxies for {n_sne} events; the events "
        f"past the sample's end took {n_taken - n_sne} of them"
    )


def test_useonce_off_baseline_allows_galaxy_reuse(tmp_path):
    """Sanity baseline: without USEONCE, the same HOSTLIB does produce
    duplicates given enough SNe — confirms the previous test is
    actually exercising the suppression rather than coincidentally
    drawing unique galaxies."""
    inp = tmp_path / "input.input"
    # Force enough SNe that pigeonhole kicks in even on a large HOSTLIB.
    _write_input(inp, mskopt=258, n_sne=600)
    _, results = _run(inp, n_sne=600)
    galid = np.asarray(results["parameters"]["host_GALID"]).astype(int)
    valid = galid > 0
    _, counts = np.unique(galid[valid], return_counts=True)
    assert (counts > 1).any(), (
        "Without USEONCE, expected at least one galaxy to be reused"
    )


# ---------------------------------------------------------------------------
# Bit 64 — USESNPAR
# ---------------------------------------------------------------------------

def test_usesnpar_override(tmp_path):
    """USESNPAR (bit 64): host SN-param columns replace the sampled values,
    and x0 is recomputed from the overridden values (sntools_host.c:671-686;
    override_modelPar_from_SNHOST, snlc_sim.c:13053-13107)."""
    with open_text(os.path.expandvars(_HOSTLIB)) as stream:
        lines = stream.read().splitlines()
    rewritten = []
    for line in lines:
        if line.startswith('VARNAMES:'):
            line = line.rstrip() + ' x1 c alpha beta RV AV'
        elif line.startswith('GAL:'):
            line = line.rstrip() + ' 1.5000 0.1100 0.1600 2.9000 2.7000 0.2000'
        rewritten.append(line)
    hostlib = tmp_path / "snpar.HOSTLIB"
    hostlib.write_text("\n".join(rewritten) + "\n")

    inp = tmp_path / "input.input"
    _write_input(inp, mskopt=322, hostlib=str(hostlib))
    sim, results = _run(inp)
    params = results["parameters"]
    for name, value in (("x1", 1.5), ("c", 0.11), ("SALT2alpha", 0.16),
                        ("SALT2beta", 2.9), ("rv", 2.7), ("av", 0.2),
                        ("sim_rv", 2.7), ("sim_av", 0.2)):
        got = np.asarray(params[name])
        # hosted events carry the column values verbatim (the SIM_ truth
        # follows the final values); the tolerance is only their
        # decimal-to-float storage
        assert np.isclose(got, value, atol=1e-6).mean() > 0.9, name

    x0 = np.asarray(params["x0"])
    expected = np.asarray(compute_expected_x0_scale(
        np.asarray(params["mu"]), np.asarray(params["SALT2alpha"]),
        np.asarray(params["SALT2beta"]), np.asarray(params["x1"]),
        np.asarray(params["c"])))
    np.testing.assert_allclose(x0, expected, rtol=1e-5)


def test_usesnpar_without_snpar_columns_aborts(tmp_path):
    """Bit 64 with a HOSTLIB carrying no SN-param column is fatal
    (sntools_host.c:2616-2624)."""
    from jax_snlc_core.logging import SNANAError
    inp = tmp_path / "input.input"
    _write_input(inp, mskopt=322)
    with pytest.raises(SNANAError, match="zero SN params"):
        _run(inp)


def test_hostlib_must_span_the_generation_redshift_range(tmp_path):
    """SNANA aborts when the HOSTLIB does not cover GENRANGE_REDSHIFT
    (check_redshift_HOSTLIB, sntools_host.c:3322). Without the check jax
    assigned the nearest galaxy instead, breaching HOSTLIB_DZTOL for most
    of the sample: verified against snlc_sim.exe, which aborts here."""
    from jax_snlc_core.logging import SNANAError
    inp = tmp_path / "input.input"
    # JAX_TEST.HOSTLIB spans z 0.02-0.50; ask for SNe out to z=0.70.
    _write_input(inp, mskopt=258, redshift_max=0.70)
    with pytest.raises(SNANAError) as exc:
        sim_run.prepare(str(inp), **dict(n_sne=5))
    msg = str(exc.value)
    assert "GENRANGE_REDSHIFT" in msg
    assert "0.50000" in msg          # the library ceiling it fell short of


def test_hostlib_spanning_the_range_does_not_abort(tmp_path):
    """The same input within the library's z range still initialises."""
    inp = tmp_path / "input.input"
    _write_input(inp, mskopt=258, redshift_max=0.40)
    sim_run.prepare(str(inp), **dict(n_sne=5))


def test_usesnpar_snmagshift_only_column(tmp_path):
    """A HOSTLIB whose only SN-param column is SNMAGSHIFT satisfies bit 64
    and lands as the SNMAGSHIFT (sntools_host.c:680; snlc_sim.c:13060)."""
    with open_text(os.path.expandvars(_HOSTLIB)) as stream:
        lines = stream.read().splitlines()
    rewritten = []
    for line in lines:
        if line.startswith('VARNAMES:'):
            line = line.rstrip() + ' SNMAGSHIFT'
        elif line.startswith('GAL:'):
            line = line.rstrip() + ' 0.0400'
        rewritten.append(line)
    hostlib = tmp_path / "snmag.HOSTLIB"
    hostlib.write_text("\n".join(rewritten) + "\n")

    inp = tmp_path / "input.input"
    _write_input(inp, mskopt=322, hostlib=str(hostlib))
    _, results = _run(inp)
    gamma = np.asarray(results["parameters"]["snmagshift"])
    assert np.isclose(gamma, 0.04, atol=1e-6).mean() > 0.9


# ------------------------------------------------------------------
# The host's say over the SN's redshift and velocity, against SNANA: each
# relation is asserted on SNANA's bundled HEAD and on jax's HEAD from the
# same input, event by event.
# ------------------------------------------------------------------

_C_KM = 2.99792458e5


def _heads(snana_fixture, name, tmp_path):
    """SNANA's HEAD for the fixture and jax's for the same input."""
    fx = snana_fixture(name)
    config = dict(n_sne=300, random_seed=122459)
    sim = sim_run.prepare(str(fx.input_file), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    results, selection, spectra = sim_run.run_simulation(sim)
    save_snana_outputs(sim, results, name.upper(), selection, spectra_by_sn=spectra, output_dir=str(tmp_path))
    heads = {}
    for side, path in (("snana", fx.head_fits),
                       ("jax", next(tmp_path.rglob("*HEAD.FITS*")))):
        with fits.open(path) as f:
            heads[side] = {n: np.asarray(f[1].data[n])
                           for n in f[1].data.columns.names}
    return heads


def _f(head, name):
    return np.asarray(head[name], dtype=np.float64)


def _zcmb(zhel, head):
    return np.asarray(zhelio_to_zcmb(zhel.astype(np.float32),
                                     _f(head, "SIM_RA").astype(np.float32),
                                     _f(head, "SIM_DEC").astype(np.float32)),
                      dtype=np.float64)


def _catalogue(path):
    """A HOSTLIB's columns, and each GALID's row."""
    with open_text(os.path.expandvars(path)) as stream:
        lines = stream.read().splitlines()
    names = next(l for l in lines if l.startswith("VARNAMES:")).split()[1:]
    rows = np.array([[float(x) for x in l.split()[1:]]
                     for l in lines if l.startswith("GAL:")])
    table = {n: rows[:, i] for i, n in enumerate(names)}
    return table, {int(g): i for i, g in enumerate(table["GALID"])}


@pytest.mark.parametrize("name", ["hostlib_sn2galz_vpec",
                                  "hostlib_sn2galz_wgtmap"])
def test_sn2gal_z_takes_the_hosts_redshift(snana_fixture, tmp_path, name):
    """SN2GAL_Z: zHEL is the host's ZTRUE, which already holds its velocity,
    and zCMB is its CMB-frame value with the SN's VPEC removed
    (TRANSFER_SNHOST_REDSHIFT, sntools_host.c:9160-9172). Dust is drawn after
    the host in either order, so an RV with a redshift variation and no width
    reads the final zCMB (snlc_sim.c:12912); so does x1 when the model follows
    the host too, which it does unless a weight map reads it."""
    for side, h in _heads(snana_fixture, name, tmp_path).items():
        zhel, zcmb = _f(h, "SIM_REDSHIFT_HELIO"), _f(h, "SIM_REDSHIFT_CMB")
        np.testing.assert_array_equal(zhel, _f(h, "SIM_REDSHIFT_HOST"),
                                      err_msg=side)
        np.testing.assert_allclose(
            zcmb, (1.0 + _zcmb(zhel, h)) / (1.0 + _f(h, "SIM_VPEC") / _C_KM) - 1.0,
            rtol=0, atol=1e-6, err_msg=side)
        np.testing.assert_allclose(_f(h, "SIM_RV"), 3.1 + 5.0 * zcmb,
                                   rtol=0, atol=1e-6, err_msg=side)
        if name == "hostlib_sn2galz_vpec":
            np.testing.assert_allclose(_f(h, "SIM_SALT2x1"), 10.0 * zcmb,
                                       rtol=0, atol=1e-5, err_msg=side)


def test_sn2gal_radec_recomputes_zhel_at_the_host(snana_fixture, tmp_path):
    """SN2GAL_RADEC alone: the SN is written at its host, zCMB stays and zHEL
    is its heliocentric value at the new position with the SN's VPEC
    (sntools_host.c:9176-9189, :7867)."""
    for side, h in _heads(snana_fixture, "hostlib_sn2galradec",
                          tmp_path).items():
        zcmb, ra, dec = (_f(h, "SIM_REDSHIFT_CMB"), _f(h, "SIM_RA"),
                         _f(h, "SIM_DEC"))
        zhel = np.asarray(zcmb_to_zhelio(zcmb.astype(np.float32),
                                         ra.astype(np.float32),
                                         dec.astype(np.float32)), float)
        np.testing.assert_allclose(
            _f(h, "SIM_REDSHIFT_HELIO"),
            (1.0 + zhel) * (1.0 + _f(h, "SIM_VPEC") / _C_KM) - 1.0,
            rtol=0, atol=1e-6, err_msg=side)
        matched = _f(h, "HOSTGAL_RA") > -900.0
        separation = 3600.0 * np.hypot(
            (ra - _f(h, "HOSTGAL_RA")) * np.cos(np.radians(dec)),
            dec - _f(h, "HOSTGAL_DEC"))
        assert matched.any() and separation[matched].max() < 60.0, side


def test_usevpec_with_sn2gal_z_uses_the_hosts_velocity(snana_fixture,
                                                       tmp_path):
    """USEVPEC with SN2GAL_Z: the SN's VPEC is its host's, and it is the one
    removed from zCMB (GEN_SNHOST_VPEC, sntools_host.c:7464)."""
    table, row = _catalogue(
        "$JAX_TEST_FIXTURE_ROOT/hostlib/JAX_TEST_VPEC.HOSTLIB")
    for side, h in _heads(snana_fixture, "hostlib_usevpec_sn2galz",
                          tmp_path).items():
        gal = np.array([row[int(g)] for g in h["SIM_HOSTLIB_GALID"]])
        vpec = table["VPEC"][gal]
        np.testing.assert_allclose(_f(h, "SIM_VPEC"), vpec, rtol=0,
                                   atol=1e-3, err_msg=side)
        zhel = _f(h, "SIM_REDSHIFT_HELIO")
        np.testing.assert_allclose(zhel, table["ZTRUE"][gal], rtol=0,
                                   atol=1e-6, err_msg=side)
        np.testing.assert_allclose(
            _f(h, "SIM_REDSHIFT_CMB"),
            (1.0 + _zcmb(zhel, h)) / (1.0 + vpec / _C_KM) - 1.0,
            rtol=0, atol=1e-6, err_msg=side)


def test_wrong_host_photoz_follows_the_lookup_redshift(snana_fixture,
                                                       tmp_path):
    """WRONGHOST_FILE: SIM_REDSHIFT_HOST is the drawn galaxy's ZTRUE
    (GEN_SNHOST_GALID, sntools_host.c:6481), and the host photo-z is the
    catalogue's shifted by the lookup redshift ZGEN, not the SN's
    (GEN_SNHOST_ZPHOT_from_HOSTLIB, :7185). So on a wrong host the implied
    ZGEN sits by the galaxy, far from the SN."""
    table, row = _catalogue(_HOSTLIB)
    for side, h in _heads(snana_fixture, "hostlib_wronghost",
                          tmp_path).items():
        gal = np.array([row[int(g)] for g in h["SIM_HOSTLIB_GALID"]])
        ztrue = table["ZTRUE"][gal]
        zhost = _f(h, "SIM_REDSHIFT_HOST")
        np.testing.assert_allclose(zhost, ztrue, rtol=0, atol=1e-6,
                                   err_msg=side)
        zhel = _f(h, "SIM_REDSHIFT_HELIO")
        reported = (_f(h, "HOSTGAL_OBJID") == _f(h, "SIM_HOSTLIB_GALID")) \
            & (_f(h, "HOSTGAL_PHOTOZ") > 0.0)
        wrong = reported & (np.abs(zhost - zhel) > 0.06)
        assert wrong.any(), side
        zgen = _f(h, "HOSTGAL_PHOTOZ") - table["ZPHOT"][gal] + ztrue
        assert np.all(np.abs(zgen - ztrue)[wrong] < 0.05), side
        assert np.all(np.abs(zgen - zhel)[wrong] > 0.06), side


@pytest.mark.parametrize('drawn_rv', [0., 3.1])
@pytest.mark.parametrize('host_dust', ['RV', 'RV AV', 'RV EBV'])
def test_host_rv_preserves_drawn_av(tmp_path, data_root, host_dust, drawn_rv):
    """SNANA preserves drawn AV when RV changes; host AV/EBV can replace it."""
    with open_text(str(data_root / 'hostlib/JAX_TEST.HOSTLIB')) as stream:
        lines = stream.read().splitlines()
    values = {'RV': '2.7', 'AV': '.4', 'EBV': '.2'}
    for i, line in enumerate(lines):
        if line.startswith('VARNAMES:'):
            lines[i] += ' ' + host_dust
        elif line.startswith('GAL:'):
            lines[i] += ' ' + ' '.join(values[name] for name in host_dust.split())
    hostlib = tmp_path / 'dust.HOSTLIB'
    hostlib.write_text('\n'.join(lines) + '\n')
    path = tmp_path / 'sim.input'
    _write_input(path, mskopt=322, hostlib=str(hostlib), n_sne=4)
    with path.open('a') as stream:
        stream.write('\nGENRANGE_EBV_HOST: .1 .1\n'
                     f'GENPEAK_RV: {drawn_rv}\nGENRANGE_RV: {drawn_rv} {drawn_rv}\n')
    simulator = sim_run.prepare(str(path), n_sne=4, batch_size=2)
    results, _, _ = sim_run.run_simulation(simulator)
    parameters = results['parameters']
    for name in ('rv', 'sim_rv'):
        np.testing.assert_allclose(parameters[name], 2.7)
    # snlc_sim.c:13072-13082 preserves AV unless a host AV/EBV column replaces it.
    expected_av = {'RV': .1 * drawn_rv, 'RV AV': .4, 'RV EBV': .54}[host_dust]
    for name in ('av', 'sim_av'):
        np.testing.assert_allclose(parameters[name], expected_av)
