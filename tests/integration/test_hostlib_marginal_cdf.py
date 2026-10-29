"""HOSTLIB marginal-CDF sampling (bit 16) keeps SN z in GENRANGE bounds."""

import re
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate, stats

from jax_snlc_core.cosmology import zcmb_to_zhelio
from jax_snlc_core.logging import SNANAError
from jax_snlc_core.snana_files import open_text
from jax_snlc_sim import run as sim_run
from jax_snlc_sim.hostlib.selection import refuse_empty_windows


_HOSTLIB = "$JAX_TEST_FIXTURE_ROOT/hostlib/JAX_TEST.HOSTLIB"
_KCOR = "$JAX_TEST_FIXTURE_ROOT/kcor/JAX_TEST_LSST.fits"
_SIMLIB = "$JAX_TEST_FIXTURE_ROOT/simlib/JAX_TEST_LSST.SIMLIB"
_MODEL = "$JAX_TEST_FIXTURE_ROOT/models/SALT3.K21"
_WGTMAP_X1 = "$JAX_TEST_FIXTURE_ROOT/wgtmap/JAX_TEST_X1_LOGMASS.WGTMAP"
_NON1A = "tests/fixtures/snana/non1ased_v19_replay/input.input"


def _write_input(out: Path, *, mskopt: int, z_min: float, z_max: float,
                  n_sne: int = 200, seed: int = 12345,
                  ra_range=None, decl_range=None,
                  wgtmap: str | None = None,
                  galid_priority: tuple[int, int] | None = None,
                  mindaysep: int | None = None,
                  dztol: str = "0.002,0.04") -> None:
    extra = ""
    if ra_range is not None:
        extra += f"GENRANGE_RA:  {ra_range[0]}  {ra_range[1]}\n"
    if decl_range is not None:
        extra += f"GENRANGE_DECL:  {decl_range[0]}  {decl_range[1]}\n"
    if wgtmap is not None:
        extra += f"HOSTLIB_WGTMAP_FILE:  {wgtmap}\n"
    if galid_priority is not None:
        extra += f"HOSTLIB_GALID_PRIORITY:  {galid_priority[0]},{galid_priority[1]}\n"
    if mindaysep is not None:
        extra += f"HOSTLIB_MINDAYSEP_SAMEGAL:  {mindaysep}\n"
    out.write_text(f"""\
GENVERSION:  JAX_TEST_HOSTLIB_MARGINAL
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
HOSTLIB_FILE:  {_HOSTLIB}
HOSTLIB_MSKOPT:  {mskopt}
HOSTLIB_DZTOL:  {dztol}
GENRANGE_REDSHIFT:  {z_min}  {z_max}
GENRANGE_MJD:  60500.0  61500.0
GENRANGE_PEAKMJD:  60540.0  61460.0
GENSIGMA_REDSHIFT:  0.001
GENRANGE_TREST: -20.0 50.0
SOLID_ANGLE: 0.015
OPT_MWEBV: 0
DNDZ: POWERLAW 2.5E-5 1.5
RANSEED:  {seed}
{extra}""")


def test_bit_16_follows_dndz_reweight(tmp_path):
    """Under bit 16 the host draw follows the reweighted dN/dz, as the
    redshift draw without bit 16 does: DNDZ_ZEXP_REWGT = -2 pulls the mean
    z down from about 0.29 to the bit-0 value near 0.21."""
    def mean_z(mskopt, zexp):
        inp = tmp_path / f"m{mskopt}_z{zexp}.input"
        _write_input(inp, mskopt=mskopt, z_min=0.05, z_max=0.40, n_sne=300,
                     seed=4321)
        if zexp:
            inp.write_text(inp.read_text() + f"DNDZ_ZEXP_REWGT:  {zexp}\n")
        cfg = dict(n_sne=300, random_seed=4321)
        sim = sim_run.prepare(str(inp), **cfg)
        resources = sim
        results, _, _ = sim_run.run_simulation(sim)
        return float(np.mean(results['parameters']['sim_redshift_cmb']))

    plain, reweighted = mean_z(16, 0), mean_z(16, -2.0)
    assert plain - reweighted > 0.04
    assert abs(reweighted - mean_z(0, -2.0)) < 0.02


def _write_simlib(path, headers):
    """The fixture SIMLIB with a header line added to each entry ``headers`` names."""
    simlib = Path("tests/fixtures/data/simlib/JAX_TEST_LSST.SIMLIB").read_text()
    for field, line in headers.items():
        simlib = re.sub(rf"(RA:.*FIELD: {field}\n)",
                        lambda m: f"{m.group(1)}{line}\n", simlib)
    path.write_text(simlib)


def _run_pec1a(tmp_path, mskopt, windows=(), window="0.15 0.35"):
    """The NON1A input with a PEC1A template over z 0.05-0.45, 600 events;
    ``windows`` gives GENRANGE_REDSHIFT ``window`` to the SIMLIB entries it
    names.
    Returns each event's zCMB, PEC1A flag and SIMLIB entry."""
    src = Path(_NON1A).read_text()
    src = src.replace("NON1A: 834 1.0 0.00 0.00 21",
                      "NON1A: 834 1.0 0.00 0.00 21\n"
                      "PEC1A: 501 1.0 0.00 0.00 21")
    src = src.replace("DNDZ:  0.7*CC_S15",
                      "DNDZ:  POWERLAW 2.5E-5 0.0\n"
                      "DNDZ_PEC1A: POWERLAW 1.0E-4 -12.0")
    src = src.replace("GENRANGE_REDSHIFT:  0.0200  0.4000",
                      "GENRANGE_REDSHIFT:  0.0500  0.4500")
    src = src.replace("NGENTOT_LC:  1500", "NGENTOT_LC:  600")
    src += (f"HOSTLIB_FILE:  {_HOSTLIB}\nHOSTLIB_MSKOPT:  {mskopt}\n"
            "HOSTLIB_DZTOL:  0.002,0.04\n")
    tag = f"pec1a_m{mskopt}_w{''.join(windows)}_{window.replace(' ', '_')}"
    if windows:
        _write_simlib(tmp_path / f"{tag}.SIMLIB",
                      {field: f"GENRANGE_REDSHIFT: {window}" for field in windows})
        src = src.replace(_SIMLIB, str(tmp_path / f"{tag}.SIMLIB"))
    inp = tmp_path / f"{tag}.input"
    inp.write_text(src)
    cfg = dict(n_sne=600, random_seed=11)
    sim = sim_run.prepare(str(inp), **cfg)
    resources = sim
    results, _, _ = sim_run.run_simulation(sim)
    params = results['parameters']
    pec1a = np.asarray(params['non1a_index']) == 501
    assert pec1a.sum() > 80
    return (np.asarray(params['sim_redshift_cmb']), pec1a,
            np.asarray(params['field_idx']))


def test_bit_16_pec1a_hosts_follow_the_pec1a_rate(tmp_path):
    """Under bit 16 each event's host follows its own rate: DNDZ_PEC1A for
    the peculiar events, DNDZ for the rest, as without bit 16.

    Stock SNANA on this input with 2000 events: PEC1A mean z 0.217 with bit
    16 and 0.223 without; the others 0.321 and 0.324.
    """
    def mean_z(mskopt):
        z, pec1a, _ = _run_pec1a(tmp_path, mskopt)
        return float(z[pec1a].mean()), float(z[~pec1a].mean())

    pec1a_16, normal_16 = mean_z(16)
    pec1a_0, normal_0 = mean_z(0)
    assert normal_16 - pec1a_16 > 0.06
    assert abs(pec1a_16 - pec1a_0) < 0.03
    assert abs(normal_16 - normal_0) < 0.02


_WINDOW_WITH_DZTOL = (0.15 - (0.002 + 0.04 * 0.15), 0.35 + (0.002 + 0.04 * 0.35))


def test_bit_16_hosts_stay_in_the_simlib_entry_window(tmp_path):
    """Under bit 16 an entry's GENRANGE_REDSHIFT redraws z inside its window
    at the DNDZ rate, peculiar or not (regen_SIMLIB_GENRANGES,
    snlc_sim.c:22316-22371), and the host is then picked within DZTOL of
    that z, so the event's z stays inside the window give or take DZTOL.

    Stock SNANA on this input with 2000 events and both entries windowed to
    0.15-0.35: z spans 0.144-0.363, PEC1A mean z 0.271, the others 0.269.
    """
    z, pec1a, _ = _run_pec1a(tmp_path, 16, windows=("TOY_A", "TOY_B"))
    lo, hi = _WINDOW_WITH_DZTOL
    assert np.all((z >= lo) & (z <= hi))
    assert abs(z[pec1a].mean() - z[~pec1a].mean()) < 0.02
    assert abs(z[pec1a].mean() - 0.27) < 0.02
    assert abs(z[~pec1a].mean() - 0.27) < 0.02


def test_matching_entry_window_uses_dndz(
        tmp_path):
    """An entry's GENRANGE_REDSHIFT redraws z whenever its lower edge is
    above 1e-5 (regen_SIMLIB_GENRANGES, snlc_sim.c:22361-22362), even when
    it equals the input's range, so PEC1A events there follow DNDZ, not
    DNDZ_PEC1A. Without the window their mean z is about 0.10 lower."""
    z, pec1a, _ = _run_pec1a(tmp_path, 16, windows=("TOY_A", "TOY_B"),
                             window="0.05 0.45")
    assert abs(z[pec1a].mean() - z[~pec1a].mean()) < 0.03


def test_bit_16_unwindowed_entry_keeps_the_global_range(tmp_path):
    """With a GENRANGE_REDSHIFT on one of two SIMLIB entries, that entry's
    events stay in its window while the other entry's span the input's
    GENRANGE_REDSHIFT, 0.05-0.45."""
    z, _, entry = _run_pec1a(tmp_path, 16, windows=("TOY_A",))
    lo, hi = _WINDOW_WITH_DZTOL
    assert np.all((z[entry == 0] >= lo) & (z[entry == 0] <= hi))
    assert z[entry == 1].min() < 0.10 and z[entry == 1].max() > 0.40


def test_bit_16_entries_with_one_window_share_its_host_draw(tmp_path):
    """Two SIMLIB entries with the same GENRANGE_REDSHIFT, a zCMB window, share
    one host draw over the zHEL range common to both entries' dipole terms.
    Every galaxy inside that common range can be drawn, and none farther
    than DZTOL outside it."""
    dztol = 0.001
    simlib = tmp_path / "shared_window.SIMLIB"
    _write_simlib(simlib, {field: "GENRANGE_REDSHIFT: 0.20 0.21"
                           for field in ("TOY_A", "TOY_B")})
    inp = tmp_path / "shared_window.input"
    _write_input(inp, mskopt=16, z_min=0.05, z_max=0.45, dztol=f"{dztol},0")
    inp.write_text(inp.read_text().replace(_SIMLIB, str(simlib)))
    _prepared = sim_run.prepare(str(inp), **dict(n_sne=10))
    resources = _prepared
    z_gal, share = _marginal_share(resources, 0)
    np.testing.assert_array_equal(share, _marginal_share(resources, 1)[1])
    drawn = share > 0
    windows = [[float(zcmb_to_zhelio(z, *_ENTRY_RADEC[entry])) for z in (0.20, 0.21)]
               for entry in (0, 1)]
    lo, hi = max(w[0] for w in windows), min(w[1] for w in windows)
    assert drawn[(z_gal >= lo) & (z_gal <= hi)].all()
    assert np.all((z_gal[drawn] > lo - dztol) & (z_gal[drawn] < hi + dztol))


def test_bit_16_nrepeat_follower_keeps_its_leaders_host(tmp_path):
    """Under HOSTLIB_NREPEAT_GALID_SNPOS a follower takes its leader's host
    without a draw (GEN_SNHOST_GALID returns early, sntools_host.c:6137),
    even from a SIMLIB entry with another redshift window."""
    simlib = tmp_path / "nrepeat.SIMLIB"
    _write_simlib(simlib, {"TOY_A": "GENRANGE_REDSHIFT: 0.15 0.20",
                           "TOY_B": "GENRANGE_REDSHIFT: 0.30 0.35"})
    inp = tmp_path / "nrepeat.input"
    _write_input(inp, mskopt=16, z_min=0.05, z_max=0.45, n_sne=60, seed=7)
    inp.write_text(inp.read_text().replace(_SIMLIB, str(simlib))
                   + "HOSTLIB_NREPEAT_GALID_SNPOS:  3\n")
    cfg = dict(n_sne=60, random_seed=7)
    sim = sim_run.prepare(str(inp), **cfg)
    resources = sim
    results, _, _ = sim_run.run_simulation(sim)
    params = results['parameters']
    passing = np.asarray(params['prefilter_pass'], bool)
    leader = np.asarray(params['host_leader_global_idx'])[passing]
    entry = np.asarray(params['field_idx'])
    galid = np.asarray(params['host_GALID'])
    assert (entry[passing] != entry[leader]).sum() > 5
    np.testing.assert_array_equal(galid[passing], galid[leader])


def _run_simlib_redshift(tmp_path, tag, headers):
    """The SALT input under bit 16 and USE_SIMLIB_REDSHIFT over z 0.05-0.45,
    400 events, with ``headers`` on the SIMLIB entries. Returns each event's
    zHEL, zCMB and SIMLIB entry."""
    simlib = tmp_path / f"{tag}.SIMLIB"
    _write_simlib(simlib, headers)
    inp = tmp_path / f"{tag}.input"
    _write_input(inp, mskopt=16, z_min=0.05, z_max=0.45, n_sne=400, seed=7)
    inp.write_text(inp.read_text().replace(_SIMLIB, str(simlib))
                   + "USE_SIMLIB_REDSHIFT:  1\n")
    cfg = dict(n_sne=400, random_seed=7)
    sim = sim_run.prepare(str(inp), **cfg)
    resources = sim
    results, _, _ = sim_run.run_simulation(sim)
    params = results['parameters']
    return (np.asarray(params['sim_redshift_helio']),
            np.asarray(params['sim_redshift_cmb']),
            np.asarray(params['field_idx']))


_ENTRY_RADEC = ((10.0, -30.0), (180.0, -45.0))


def _assert_hosts_at_simlib_redshift(zhel, entry, i, z_simlib):
    """Every event of entry ``i`` lies within DZTOL of the entry's zHEL."""
    z_point = float(zcmb_to_zhelio(z_simlib, *_ENTRY_RADEC[i]))
    z = zhel[entry == i]
    assert z.size > 0
    assert np.all(np.abs(z - z_point) < 0.002 + 0.04 * z_point)


def _hostlib_ztrue():
    """ZTRUE of every galaxy in the toy HOSTLIB, read from the file."""
    path = Path(_HOSTLIB.replace("$JAX_TEST_FIXTURE_ROOT",
                                 "tests/fixtures/data"))
    with open_text(path) as stream:
        return np.sort([float(line.split()[2]) for line in stream
                        if line.startswith("GAL:")])


def test_bit_16_fixed_simlib_redshift_hosts(tmp_path):
    """Under bit 16 an entry's REDSHIFT sets zCMB (regen_SIMLIB_GENRANGES,
    snlc_sim.c:22360-22370), the host is picked within DZTOL of the entry's
    zHEL (GEN_SNHOST_GALID, sntools_host.c:6144-6290) and the event takes
    the host's ZTRUE. Without a weight map every galaxy strictly inside the
    window is equally likely.

    Stock SNANA on this input with 2000 events: zHEL spans 0.1913-0.2101 at
    REDSHIFT 0.20 and 0.2851-0.3117 at 0.30. The DZTOL windows reach 0.2111
    and 0.3127; SNANA never draws the first galaxy of a window nor the top 5%
    of its weight, so its hosts stop at 0.2101 and 0.3117 while these reach
    the window tops. The KS bound is the 0.1% critical value for the sample.
    """
    zhel, _, entry = _run_simlib_redshift(
        tmp_path, "fixed", {"TOY_A": "REDSHIFT: 0.20", "TOY_B": "REDSHIFT: 0.30"})
    z_gal = _hostlib_ztrue()
    for i, (z_simlib, snana_max) in enumerate(((0.20, 0.2101), (0.30, 0.3117))):
        _assert_hosts_at_simlib_redshift(zhel, entry, i, z_simlib)
        z = zhel[entry == i]
        z_point = float(zcmb_to_zhelio(z_simlib, *_ENTRY_RADEC[i]))
        dztol = 0.002 + 0.04 * z_point
        window = z_gal[np.abs(z_gal - z_point) < dztol]
        # the HOSTLIB gives ZTRUE to four decimals
        ks = stats.kstest(np.round(z, 4),
                          lambda x: np.searchsorted(window, x, 'right')
                          / window.size)
        assert ks.statistic < 1.95 / np.sqrt(z.size)
        assert z.max() > snana_max


def test_bit_16_fixed_and_unfixed_simlib_entries(tmp_path):
    """REDSHIFT on one entry fixes only that entry's events; the other's span
    GENRANGE_REDSHIFT. Stock SNANA with 2000 events: the unfixed entry's
    zCMB spans 0.071-0.449."""
    zhel, zcmb, entry = _run_simlib_redshift(
        tmp_path, "mixed", {"TOY_A": "REDSHIFT: 0.20"})
    _assert_hosts_at_simlib_redshift(zhel, entry, 0, 0.20)
    assert zcmb[entry == 1].min() < 0.15 and zcmb[entry == 1].max() > 0.40


def test_bit_16_last_simlib_redshift_key_wins(tmp_path):
    """REDSHIFT and GENRANGE_REDSHIFT on one entry: the one read last sets
    the window (parse_SIMLIB_GENRANGES, snlc_sim.c:22138-22165). Stock
    SNANA with 2000 events: zCMB 0.1902-0.2090 where REDSHIFT: 0.20 comes
    last, and 0.2875-0.4142 where GENRANGE_REDSHIFT: 0.30 0.40 does."""
    zhel, zcmb, entry = _run_simlib_redshift(tmp_path, "order", {
        "TOY_A": "GENRANGE_REDSHIFT: 0.30 0.40  REDSHIFT: 0.20",
        "TOY_B": "REDSHIFT: 0.20  GENRANGE_REDSHIFT: 0.30 0.40"})
    _assert_hosts_at_simlib_redshift(zhel, entry, 0, 0.20)
    z = zcmb[entry == 1]
    assert z.min() > 0.30 - (0.002 + 0.04 * 0.30)
    assert z.max() < 0.40 + (0.002 + 0.04 * 0.40)
    assert z.max() - z.min() > 0.08


def test_bit_16_with_3d_wgtmap_keeps_z_in_bounds(tmp_path):
    """3D WGTMAP (x1, LOGMASS): per-bin marginal CDF dispatched correctly.

    With ``HOSTLIB_WGTMAP_FILE`` adding x1 as an SN variable, the per-
    galaxy weight depends on the SN's x1 bin. The marginal CDF must be
    (n_snvar_bins, n_galaxies) and each SN must sample from its own bin.
    """
    z_min, z_max = 0.10, 0.30
    inp = tmp_path / "input.input"
    _write_input(inp, mskopt=16, z_min=z_min, z_max=z_max,
                 n_sne=200, wgtmap=_WGTMAP_X1)
    cfg = dict(n_sne=200, random_seed=12345)
    sim = sim_run.prepare(str(inp), **cfg)
    resources = sim
    cdf = resources.hostlib.catalogue.marginal_cdf
    assert cdf is not None
    assert cdf.ndim == 2
    assert cdf.shape[0] > 1, (
        "3D WGTMAP should produce > 1 SN-variable bin"
    )
    assert resources.hostlib.selector.has_snvars
    results, _, _ = sim_run.run_simulation(sim)
    sim_z = np.asarray(results['parameters']['sim_redshift_cmb'])
    assert np.all(sim_z >= z_min)
    assert np.all(sim_z <= z_max)


def _window_draw_model(z_gal, z_hel, prob, dztol, z_range):
    """Each galaxy's share of hosts when zHEL takes the values ``z_hel`` with
    probabilities ``prob`` and every galaxy strictly within ``dztol(zHEL)``
    is equally likely, kept to the galaxies inside ``z_range``."""
    width = dztol(z_hel)
    first = np.searchsorted(z_gal, z_hel - width, side='right')
    end = np.searchsorted(z_gal, z_hel + width, side='left')
    share = np.zeros(z_gal.size + 1)
    np.add.at(share, first, prob / (end - first))
    np.add.at(share, end, -prob / (end - first))
    share = np.cumsum(share)[:-1] * ((z_gal >= z_range[0])
                                     & (z_gal <= z_range[1]))
    return share / share.sum()


def _marginal_share(resources, entry=None):
    """``(ZTRUE, share)`` of each loaded galaxy in the SN2GAL_Z host draw for
    a SIMLIB entry, or for the input's redshift range."""
    selector = resources.hostlib.selector
    catalogue = resources.hostlib.catalogue
    row = 0 if entry is None else int(selector.field_row_block[entry]) * selector.n_bins
    cdf = np.asarray(catalogue.marginal_cdf[row], dtype=np.float64)
    return (np.asarray(catalogue.redshifts, dtype=np.float64),
            np.diff(np.concatenate([[0.0], cdf])))


def test_bit_16_host_draw_with_a_redshift_dependent_dztol(tmp_path):
    """Under bit 16 each galaxy's share of hosts is the DZTOL-window draw
    integrated over zHEL, with DZTOL evaluated at the supernova's zHEL
    (GEN_SNHOST_GALID, sntools_host.c:6145) and the window strict on both
    sides (:6182-6223). DZTOL = 0.02 + 0.2z over GENRANGE_REDSHIFT 0.20-0.30,
    with no dipole and no peculiar velocity, so zHEL follows dN/dz for the
    POWERLAW rate in flat LCDM.

    SNANA's 2000 events on this input agree with the model at KS p = 0.21,
    though SNANA differs from it by its two draw quirks: the first galaxy of
    each window is never drawn and the draw stops at 95% of the window's
    weight.
    """
    z_min, z_max = 0.20, 0.30
    inp = tmp_path / "wide.input"
    _write_input(inp, mskopt=16, z_min=z_min, z_max=z_max, dztol="0.02,0.2")
    inp.write_text(inp.read_text() + "VEL_CMBAPEX:  0\n")
    _prepared = sim_run.prepare(str(inp), **dict(n_sne=10))
    resources = _prepared

    omega_m = 0.315
    z = np.linspace(z_min, z_max, 400001)
    z_fine = np.linspace(0.0, z_max, 20001)
    hubble = np.sqrt(omega_m * (1 + z_fine) ** 3 + 1 - omega_m)
    chi = np.interp(z, z_fine, integrate.cumulative_trapezoid(
        1 / hubble, z_fine, initial=0))
    rate = chi ** 2 / np.interp(z, z_fine, hubble) * (1 + z) ** 0.5
    z_gal, share = _marginal_share(resources)
    expected = _window_draw_model(z_gal, z, rate / rate.sum(),
                                  lambda zz: 0.02 + 0.2 * zz, (z_min, z_max))
    assert 0.5 * np.abs(share - expected).sum() < 2e-3
    assert np.abs(np.cumsum(share) - np.cumsum(expected)).max() < 2e-4


def _fixed_redshift_resources(tmp_path, gensigma_vpec):
    """Both SIMLIB entries at fixed REDSHIFT 0.20 and 0.30 under bit 16 with a
    constant DZTOL of 0.002 and the given GENSIGMA_VPEC."""
    simlib = tmp_path / "fixedz.SIMLIB"
    _write_simlib(simlib, {"TOY_A": "REDSHIFT: 0.20", "TOY_B": "REDSHIFT: 0.30"})
    inp = tmp_path / f"vpec{gensigma_vpec}.input"
    _write_input(inp, mskopt=16, z_min=0.05, z_max=0.45, dztol="0.002,0")
    inp.write_text(inp.read_text().replace(_SIMLIB, str(simlib))
                   + f"USE_SIMLIB_REDSHIFT:  1\nGENSIGMA_VPEC:  {gensigma_vpec}\n")
    return sim_run.prepare(str(inp), n_sne=10)


def test_bit_16_peculiar_velocity_widens_a_fixed_redshift_host(tmp_path):
    """A fixed SIMLIB redshift reaches the host draw as zHEL times
    1 + vpec/c (gen_redshift_helio, snlc_sim.c:17235-17243), so with
    GENSIGMA_VPEC the host follows the window draw averaged over a Gaussian
    zHEL rather than at one zHEL.

    SNANA with 1000 events per entry: host-z std 0.00099 at REDSHIFT 0.20
    without VPEC and 0.00159 with GENSIGMA_VPEC 300; both carry its two draw
    quirks. The model here gives 0.00108 and 0.00168.
    """
    def std(share):
        mean = (share * z_gal).sum()
        return np.sqrt((share * (z_gal - mean) ** 2).sum())

    stds = []
    for gensigma_vpec in (0, 300):
        resources = _fixed_redshift_resources(tmp_path, gensigma_vpec)
        for entry, z_simlib in enumerate((0.20, 0.30)):
            z_gal, share = _marginal_share(resources, entry)
            z_point = float(zcmb_to_zhelio(z_simlib, *_ENTRY_RADEC[entry]))
            sigma = (1 + z_point) * gensigma_vpec / 2.99792458e5
            if sigma > 0:
                z = np.linspace(z_point - 8 * sigma, z_point + 8 * sigma, 200001)
                prob = stats.norm.pdf(z, z_point, sigma)
            else:
                z, prob = np.array([z_point]), np.array([1.0])
            expected = _window_draw_model(z_gal, z, prob / prob.sum(),
                                          lambda zz: 0.002 + 0 * zz, (0.0, 1.0))
            assert 0.5 * np.abs(share - expected).sum() < 2e-3
            assert abs(std(share) - std(expected)) < 0.01 * std(expected)
            stds.append(std(share))
    assert stds[2] > 1.4 * stds[0] and stds[3] > 1.4 * stds[1]


_EMPTY_WINDOW = "no HOSTLIB galaxy to draw within DZTOL"


def _empty_window_input(tmp_path, mskopt, dztol):
    """A constant DZTOL below half of some gaps in the HOSTLIB's redshifts,
    so some windows hold no galaxy."""
    inp = tmp_path / f"empty{mskopt}.input"
    _write_input(inp, mskopt=mskopt, z_min=0.10, z_max=0.30, n_sne=50,
                 dztol=f"{dztol},0")
    return inp


def test_empty_dztol_window_aborts_the_run(tmp_path):
    """SNANA aborts when the host lies beyond DZTOL (sntools_host.c:6497),
    which an empty window forces; the run aborts after the batch."""
    cfg = dict(n_sne=50, random_seed=3)
    sim = sim_run.prepare(str(_empty_window_input(tmp_path, 0, 0.000001)), **cfg)
    resources = sim
    with pytest.raises(SNANAError, match=_EMPTY_WINDOW):
        sim_run.run_simulation(sim)


@pytest.mark.parametrize("z_range, dztol, aborts",
                         [((0.199, 0.201), 0.02, False),
                          ((0.205, 0.210), 0.002, True)])
def test_galid_force_outside_dztol_aborts(tmp_path, z_range, dztol, aborts):
    """SNANA's DZTOL check also covers a HOSTLIB_GALID_FORCE host: GALID
    100222 lies at ZTRUE=0.2002, within DZTOL=0.02 of SNe at z 0.199-0.201
    and beyond DZTOL=0.002 of SNe at z 0.205-0.210."""
    inp = tmp_path / "force.input"
    _write_input(inp, mskopt=0, z_min=z_range[0], z_max=z_range[1], n_sne=50,
                 dztol=f"{dztol},0")
    inp.write_text(inp.read_text() + "HOSTLIB_GALID_FORCE:  100222\n")
    cfg = dict(n_sne=50, random_seed=3)
    sim = sim_run.prepare(str(inp), **cfg)
    resources = sim
    if aborts:
        with pytest.raises(SNANAError, match="exceeds dztol=0.002000"):
            sim_run.run_simulation(sim)
    else:
        results, _, _ = sim_run.run_simulation(sim)
        np.testing.assert_allclose(
            results['parameters']['sim_redshift_host'], 0.2002, rtol=1e-6)


def test_galid_priority_wins_over_galid_force(tmp_path):
    """SNANA takes a GALID_PRIORITY galaxy in the DZTOL window before the
    forced one (sntools_host.c:6302-6317): GALID 104800 at ZTRUE=0.2005 is
    within DZTOL=0.02 of every SN at z 0.199-0.201."""
    inp = tmp_path / "force.input"
    _write_input(inp, mskopt=0, z_min=0.199, z_max=0.201, n_sne=50,
                 dztol="0.02,0", galid_priority=(104800, 104800))
    inp.write_text(inp.read_text() + "HOSTLIB_GALID_FORCE:  100222\n")
    cfg = dict(n_sne=50, random_seed=3)
    sim = sim_run.prepare(str(inp), **cfg)
    resources = sim
    results, _, _ = sim_run.run_simulation(sim)
    assert np.all(np.asarray(results['parameters']['host_GALID']) == 104800)


def test_empty_dztol_window_refuses_the_bit_16_host_cdf(tmp_path):
    """Under bit 16 SNANA zeroes ZDIF before its DZTOL check
    (sntools_host.c:6488, :6497), so an empty window silently gives a host
    outside DZTOL. Here a zHEL the draw can reach with an empty window
    refuses the host CDF at initialisation. The fixture HOSTLIB has gaps of
    up to 0.0008 in redshift between 0.1 and 0.3."""
    with pytest.raises(SNANAError, match=f"(?i){_EMPTY_WINDOW}"):
        sim_run.prepare(str(_empty_window_input(tmp_path, 16, 0.0003)), **dict(n_sne=50, random_seed=3))


def test_a_steep_dztol_refuses_the_bit_16_host_cdf(tmp_path):
    """The empty-window check at initialisation holds only while DZTOL's
    slope in z lies between -1 and 1 wherever the draw can reach, so a
    steeper DZTOL aborts: here 0.002 + 1.5 z, and a cubic whose slope
    reaches 1.2 only at z = 0.2, inside the reach."""
    inp = tmp_path / "steep.input"
    _write_input(inp, mskopt=16, z_min=0.10, z_max=0.30, n_sne=50,
                 dztol="0.002,1.5")
    with pytest.raises(SNANAError, match="HOSTLIB_DZTOL has slope 1.5000"):
        sim_run.prepare(str(inp), **dict(n_sne=50, random_seed=3))

    galaxy_z = np.linspace(0.1, 0.3, 2001)
    drawable = np.ones((1, galaxy_z.size), dtype=bool)
    cubic = (0.05, -2.8, 20.0, -100.0 / 3.0)
    with pytest.raises(SNANAError, match=r"slope 1\.2000 at zHEL=0\.20000"):
        refuse_empty_windows(galaxy_z, cubic, (0.1, 0.3), drawable)


def test_useonce_with_bit_16_enforces_uniqueness_and_bounds(tmp_path):
    """USEONCE (bit 32) + bit 16: each host used at most once, every SN's
    final z still in GENRANGE_REDSHIFT.

    Prior to the dispatch refactor, the marginal-CDF early-return bypassed
    the reuse-constraint pass entirely — useonce was silently ignored."""
    z_min, z_max = 0.10, 0.30
    n_sne = 80
    inp = tmp_path / "input.input"
    _write_input(inp, mskopt=16 + 32, z_min=z_min, z_max=z_max, n_sne=n_sne)
    cfg = dict(n_sne=n_sne,
                              random_seed=2026)
    sim = sim_run.prepare(str(inp), **cfg)
    resources = sim
    assert resources.hostlib.selector.useonce
    assert resources.hostlib.catalogue.marginal_cdf is not None
    res, _, _ = sim_run.run_simulation(sim)
    sim_z = np.asarray(res['parameters']['sim_redshift_cmb'])
    assert np.all(sim_z >= z_min)
    assert np.all(sim_z <= z_max)
    galids = np.asarray(res['parameters']['host_GALID'])
    assert len(np.unique(galids)) == len(galids), (
        f"USEONCE violated: {len(galids) - len(np.unique(galids))} "
        f"galaxies selected more than once"
    )


def test_bit_8_with_bit_16_keeps_radec_in_bounds(tmp_path):
    """bit 8 + bit 16 + narrow GENRANGE_RA/DECL: every SN's final RA and
    DEC land within the configured bounds (host RA/DEC replaces the
    SIMLIB-field RA/DEC). Marginal-CDF in_range pre-filtering must zero
    out hosts whose coordinates fall outside the GENRANGE."""
    z_min, z_max = 0.10, 0.40
    inp = tmp_path / "input.input"
    _write_input(inp, mskopt=8 + 16, z_min=z_min, z_max=z_max, n_sne=200,
                 ra_range=(160.0, 200.0), decl_range=(-60.0, -30.0))
    cfg = dict(n_sne=200,
                              random_seed=2027)
    sim = sim_run.prepare(str(inp), **cfg)
    resources = sim
    assert resources.hostlib.catalogue.marginal_cdf is not None
    res, _, _ = sim_run.run_simulation(sim)
    sn_ra = np.asarray(res['parameters']['ra'])
    sn_dec = np.asarray(res['parameters']['dec'])
    host_ra = np.asarray(res['parameters']['host_RA_GAL'])
    host_dec = np.asarray(res['parameters']['host_DEC_GAL'])
    # bit 8 puts the SN on its host, offset by the drawn SN-galaxy separation
    # (load_SNHOST_GENLC_COORDS), so allow that arcsec-scale displacement.
    implied = np.hypot((sn_ra - host_ra) * np.cos(np.radians(host_dec)) * 3600.0,
                       (sn_dec - host_dec) * 3600.0)
    np.testing.assert_allclose(implied, np.asarray(res['parameters']['host_SNSEP']),
                               atol=0.02)
    assert np.all(sn_ra >= 160.0) and np.all(sn_ra <= 200.0)
    assert np.all(sn_dec >= -60.0) and np.all(sn_dec <= -30.0)


def test_galid_priority_with_bit_16_overrides_and_stays_in_bounds(tmp_path):
    """HOSTLIB_GALID_PRIORITY + bit 16: priority galaxies override the
    marginal-CDF pick but only if their ZTRUE survives the shrunken-helio
    bounds filter, so sim_redshift_cmb cannot leave GENRANGE_REDSHIFT."""
    z_min, z_max = 0.10, 0.30
    n_sne = 200
    inp = tmp_path / "input.input"
    _write_input(inp, mskopt=16, z_min=z_min, z_max=z_max, n_sne=n_sne,
                 galid_priority=(100000, 100050))
    cfg = dict(n_sne=n_sne,
                              random_seed=2028)
    sim = sim_run.prepare(str(inp), **cfg)
    resources = sim
    pri = resources.hostlib.selector.galid_priority
    assert pri is not None and len(pri['igals']) > 0
    assert pri['redshifts'].min() >= z_min - 1e-3
    assert pri['redshifts'].max() <= z_max + 1e-3
    res, _, _ = sim_run.run_simulation(sim)
    sim_z = np.asarray(res['parameters']['sim_redshift_cmb'])
    galids = np.asarray(res['parameters']['host_GALID'])
    assert np.all((sim_z >= z_min) & (sim_z <= z_max))
    assert np.any((galids >= 100000) & (galids <= 100050)), (
        "Expected at least one SN to be overridden by a priority galaxy"
    )


def _assert_mindaysep_spacing(res, mindaysep):
    galids = np.asarray(res['parameters']['host_GALID'])
    peak_mjds = np.asarray(res['parameters']['peakmjd'])
    for gid in np.unique(galids):
        days = sorted(peak_mjds[galids == gid].tolist())
        for a, b in zip(days, days[1:]):
            assert b - a >= mindaysep, (
                f"Galaxy {gid} reused at MJDs {a:.1f} and {b:.1f} "
                f"(separation {b - a:.1f} < {mindaysep})"
            )


def test_mindaysep_with_bit_16_enforces_spacing(tmp_path):
    """HOSTLIB_MINDAYSEP_SAMEGAL + bit 16: same host can be reused only if
    peak MJDs are separated by ≥ MINDAYSEP days. Marginal-CDF dispatch
    must route through the reuse-constraint pass."""
    z_min, z_max = 0.10, 0.30
    n_sne = 300
    mindaysep = 30
    inp = tmp_path / "input.input"
    _write_input(inp, mskopt=16, z_min=z_min, z_max=z_max, n_sne=n_sne,
                 mindaysep=mindaysep)
    cfg = dict(n_sne=n_sne,
                              random_seed=2029)
    sim = sim_run.prepare(str(inp), **cfg)
    resources = sim
    assert resources.hostlib.selector.mindaysep == mindaysep
    res, _, _ = sim_run.run_simulation(sim)
    sim_z = np.asarray(res['parameters']['sim_redshift_cmb'])
    assert np.all((sim_z >= z_min) & (sim_z <= z_max))
    _assert_mindaysep_spacing(res, mindaysep)


def test_mindaysep_unplaceable_host_rejects(tmp_path):
    """MINDAYSEP no-host matches SNANA (sntools_host.c:6429): the SN is
    rejected under the HOSTLIB reason code (4), prefilter_pass=False —
    not aborted, not kept on a colliding host."""
    n_sne = 500
    inp = tmp_path / "input.input"
    _write_input(inp, mskopt=2, z_min=0.195, z_max=0.205, n_sne=n_sne,
                 mindaysep=9000)
    cfg = dict(n_sne=n_sne, random_seed=12345, batch_size=n_sne,
                              keep_rejected_events=True)
    sim = sim_run.prepare(str(inp), **cfg)
    resources = sim
    res, _, _ = sim_run.run_simulation(sim)
    code = np.asarray(res['parameters']['prefilter_reason_code'])
    pf = np.asarray(res['parameters']['prefilter_pass'])
    nohost = code == 4
    assert nohost.sum() > 0, "expected some SNe to exhaust the host pool"
    assert np.all(~pf[nohost]), "HOSTLIB-rejected SNe must have prefilter_pass=False"


def test_useonce_exhaustion_aborts(tmp_path):
    """USEONCE no-host matches SNANA (sntools_host.c:6432): when the host
    pool is exhausted, SNANA aborts — so jax raises SNANAError rather than
    rejecting or reusing."""
    n_sne = 500
    inp = tmp_path / "input.input"
    _write_input(inp, mskopt=32, z_min=0.195, z_max=0.205, n_sne=n_sne)
    cfg = dict(n_sne=n_sne,
                              random_seed=12345, batch_size=n_sne)
    sim = sim_run.prepare(str(inp), **cfg)
    resources = sim
    with pytest.raises(SNANAError, match="HOSTLIB exhausted"):
        sim_run.run_simulation(sim)


def test_mindaysep_spacing_holds_across_batches(tmp_path):
    """MINDAYSEP spacing among output events must hold when batch_size <
    n_sne (cross-batch reuse), and at batch_size=1 (fully sequential)."""
    z_min, z_max = 0.10, 0.30
    n_sne = 300
    mindaysep = 30
    inp = tmp_path / "input.input"
    _write_input(inp, mskopt=16, z_min=z_min, z_max=z_max, n_sne=n_sne,
                 mindaysep=mindaysep)
    # batch_size=1 dispatches once per event, so it runs a smaller sample:
    # the invariant is cross-batch reuse bookkeeping, not a rate
    for batch_size, n in ((50, n_sne), (1, 60)):
        cfg = dict(n_sne=n,
                                  random_seed=2029, batch_size=batch_size)
        sim = sim_run.prepare(str(inp), **cfg)
        resources = sim
        res, _, _ = sim_run.run_simulation(sim)
        _assert_mindaysep_spacing(res, mindaysep)
