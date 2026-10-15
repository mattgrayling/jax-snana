"""Host-galaxy spectra: TAKE_SPECTRUM: HOST and GENSPEC_HOST_CONTAMINATION.

The three tests marked `live_snana` compare against a generated snlc_sim.exe
run; the rest run on bundled data.
"""

import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from astropy.io import fits

from jax_snlc_core.physics.dust_laws import DustLaw
from jax_snlc_sim.hostlib.hostspec_reader import HostSpectra
from jax_snlc_sim.random_streams import root_key

_FIXTURES = Path(__file__).parent.parent / 'fixtures' / 'snana'
_INPUT = str(_FIXTURES / 'host_spectra_replay' / 'input.input')
_SPECDATA_INPUT = str(_FIXTURES / 'host_spectra_specdata' / 'input.input')
_GENVERSION = 'JAX_TEST_HOST_SPECTRA'


class _FlatUnitLaw(DustLaw):
    """A(λ)/A(V) == 1 everywhere, so the expected dimming is closed-form."""

    def __init__(self):
        super().__init__('flat', -1)

    def extinction_curve(self, wavelength, rv, **kwargs):
        return jnp.ones_like(jnp.asarray(wavelength))


def _snana_spec(genversion, fixture):
    """(spec header rows, per-row flux segments) from a generated GENVERSION."""
    d = os.path.expandvars(f'$SNDATA_ROOT/SIM/{genversion}/')
    if not os.path.isdir(d):
        pytest.skip(f"{genversion} not generated; run regenerate.py --only "
                    f"{fixture}")
    head = fits.getdata(d + f'{genversion}_HEAD.FITS', ext=1)
    spec = fits.getdata(d + f'{genversion}_SPEC.FITS', ext=1)
    flux = fits.getdata(d + f'{genversion}_SPEC.FITS', ext=2)
    info = {s: (int(g), float(z)) for s, g, z in
            zip(head['SNID'], head['HOSTGAL_OBJID'], head['SIM_REDSHIFT_HELIO'])}
    return info, spec, [flux[r['PTRSPEC_MIN'] - 1:r['PTRSPEC_MAX']] for r in spec]


def _snana_host_rows(genversion=_GENVERSION, fixture='host_spectra_replay'):
    info, spec, segs = _snana_spec(genversion, fixture)
    return [(info[r['SNID']], seg) for r, seg in zip(spec, segs)
            if r['MJD'] < -8]


def _jax_host_flam(input_path=_INPUT):
    """Host SIM_FLAM per galaxy, through the shipping pipeline."""
    from jax_snlc_sim import run as sim_run
    from jax_snlc_sim.spectrograph.genspec import (
        _host_bins_to_smeared, _parse_spectro_options, observed_bins)
    _prepared = sim_run.prepare(input_path, **dict(
        n_sne=5, random_seed=1))
    resources = _prepared
    galid = resources.hostlib.galids
    calib = resources.calibration
    opt = _parse_spectro_options({}, calib)
    bundle = resources.hostlib.spectra

    def one(gal, z):
        row = np.where(galid == gal)[0]
        if not len(row):
            return None
        sim_flam, _, _ = _host_bins_to_smeared(
            bundle, np.array([int(row[0])]), np.array([z]), np.array([0.0]),
            np.array([1.0 + z]), calib, opt, {})
        return np.asarray(sim_flam)[0][observed_bins(calib)]
    return one


@pytest.mark.live_snana
def test_host_sim_flam_matches_snana_within_the_quadrature_difference():
    """jax integrates the basis SED with a trapezoid where SNANA uses a
    piecewise-constant overlap sum, so the per-bin difference is a quadrature
    residual: it must average to ~0 rather than showing a scale error.
    A normalisation mistake anywhere in the chain would bias the mean instead.
    """
    jax_flam = _jax_host_flam()
    residuals = []
    for (gal, z), seg in _snana_host_rows():
        got = jax_flam(gal, z)
        if got is None:
            continue
        valid = seg['SIM_FLAM'] > 0
        residuals.append(got[valid] / seg['SIM_FLAM'][valid] - 1.0)
    assert residuals, "no host spectra matched by GALID"
    res = np.concatenate(residuals)
    assert np.abs(res).max() < 1e-2      # quadrature-level, not a scale error
    assert abs(res.mean()) < 1e-4        # unbiased: no normalisation mistake


@pytest.mark.live_snana
def test_specdata_host_sim_flam_matches_snana():
    """SPECDATA gives each galaxy one whole spectrum chosen by IDSPECDATA,
    rather than SPECBASIS's coefficient-weighted sum (genSpec_HOSTLIB,
    snlc_sim.c/sntools_host.c:2232). The binning downstream is shared, so the
    residual is the same quadrature difference as the SPECBASIS fixture.
    """
    jax_flam = _jax_host_flam(_SPECDATA_INPUT)
    residuals = []
    for (gal, z), seg in _snana_host_rows('JAX_TEST_HOST_SPECDATA',
                                          'host_spectra_specdata'):
        got = jax_flam(gal, z)
        if got is None:
            continue
        valid = seg['SIM_FLAM'] > 0
        residuals.append(got[valid] / seg['SIM_FLAM'][valid] - 1.0)
    assert residuals, "no host spectra matched by GALID"
    res = np.concatenate(residuals)
    assert np.abs(res).max() < 1e-2
    assert abs(res.mean()) < 1e-4


def test_specdata_ids_select_distinct_spectra():
    """The fixture must actually exercise the ID lookup: if every galaxy
    resolved to the same spectrum, swapping the index for a constant would
    still pass the parity test above."""
    from jax_snlc_sim import run as sim_run
    _prepared = sim_run.prepare(_SPECDATA_INPUT, **dict(n_sne=5))
    resources = _prepared
    ids = np.asarray(resources.hostlib.spectra.coeffs)[:, 0].astype(int)
    assert set(ids.tolist()) == {0, 1, 2, 3}
    table = resources.hostlib.spectra.table
    assert table.mode == 'SPECDATA' and table.n_spec == 4
    seds = table.event_seds(np.array([[0], [1], [2], [3]]), np.zeros(4))
    assert len({tuple(np.round(s[:5], 30)) for s in seds}) == 4


def test_host_rows_are_flagged_and_written_at_the_mjd_sentinel():
    """SNANA writes host spectra with MJD = -9 (manual sss:SPEC_EPOCH)."""
    from jax_snlc_sim import run as sim_run
    cfg = dict(
        n_sne=20, random_seed=122459)
    sim = sim_run.prepare(_INPUT, **cfg)
    simulation = None if sim is None else sim.config
    resources = sim
    res, _, _ = sim_run.run_simulation(sim)
    n = len(np.asarray(res['parameters']['sim_redshift_cmb']))
    spectra, _ = sim_run.compute_spectra_for_mask(
        sim, res, np.ones(n, dtype=bool),
        key=root_key(simulation.run.random_seed))
    hosts = [s for v in spectra.values() for s in v if s.is_host]
    others = [s for v in spectra.values() for s in v if not s.is_host]
    assert hosts and len(hosts) == len(others)   # one HOST config, one SN config
    assert all(s.mjd == -9.0 for s in hosts)
    assert not any(s.mjd == -9.0 for s in others)

    simulation.spectra.take_spectrum_prescale = {
        field.field_name: 1e9 for field in resources.fields}
    prescaled, _ = sim_run.compute_spectra_for_mask(
        sim, res, np.ones(n, dtype=bool),
        key=root_key(simulation.run.random_seed))
    kept = [s for rows in prescaled.values() for s in rows]
    assert len(kept) == len(hosts)
    assert all(s.is_host for s in kept)



_CONTAM_INPUT = str(_FIXTURES / 'host_spectra_contam' / 'input.input')
_CONTAM_GENVERSION = 'JAX_TEST_HOST_CONTAM'
_HOSTSNFRAC = 0.3


def _contam_spectra(hostsnfrac):
    """{snid: (host, sn)} SpectrumResults from the contamination fixture."""
    from jax_snlc_sim import run as sim_run
    cfg = dict(
        n_sne=20, random_seed=122459)
    sim = sim_run.prepare(_CONTAM_INPUT, **cfg)
    simulation = None if sim is None else sim.config
    resources = sim
    simulation.spectra.take_spectrum_hostsnfrac = hostsnfrac
    res, _, _ = sim_run.run_simulation(sim)
    n = len(np.asarray(res['parameters']['sim_redshift_cmb']))
    spectra, _ = sim_run.compute_spectra_for_mask(
        sim, res, np.ones(n, dtype=bool),
        key=root_key(simulation.run.random_seed))
    return {snid: (next(s for s in v if s.is_host),
                   next(s for s in v if not s.is_host))
            for snid, v in spectra.items() if len(v) == 2}


def _aligned(clean, contam):
    """Bins present in both spectra, matched on their lower edge.

    Contamination lifts empty SN bins above zero, so the two spectra do not
    carry the same bin set.
    """
    keep = np.isin(contam.lam_min, clean.lam_min)
    keep_c = np.isin(clean.lam_min, contam.lam_min)
    return keep_c, keep


def test_host_contamination_is_additive():
    """GENSPEC_HOST_CONTAMINATION (snlc_sim.c:11748) adds SCALE x the host to
    every SN spectrum and leaves host rows alone. Running the same fixture with
    and without the key isolates exactly that addition.
    """
    clean = _contam_spectra(0.0)
    contam = _contam_spectra(_HOSTSNFRAC)
    assert set(clean) == set(contam) and len(clean) > 5
    for snid, (host_c, sn_c) in contam.items():
        host_0, sn_0 = clean[snid]
        assert host_c.scale_host_contam == 0.0     # host rows are never scaled
        assert np.array_equal(host_0.sim_flam, host_c.sim_flam)
        k0, kc = _aligned(sn_0, sn_c)
        expected = sn_0.sim_flam[k0] + sn_c.scale_host_contam * host_c.sim_flam
        np.testing.assert_allclose(sn_c.sim_flam[kc], expected[k0], rtol=1e-6)


def test_hostsnfrac_uses_bin_widths_and_shared_support():
    """HOSTSNFRAC is defined as SCALE*FSUM_HOST/FSUM_PEAK (snlc_sim.c:11730),
    the sums running only over bins where both spectra carry flux -- so an
    empty SN-peak bin must drop the host flux there from the ratio too.
    """
    from jax_snlc_sim.spectrograph.genspec import host_contam_scale
    # Deliberately unequal bin widths: SNANA sums GENFLUX*LAMBIN and GENFLUX is
    # itself FLAM*LAMBIN, so the weight is LAMBIN^2 -- which only shows up when
    # the bins differ. Equal widths cancel out of the ratio entirely.
    calib = _tiny_calib([4000.0, 5000.0, 6000.0], [4100.0, 5200.0, 6400.0])
    host = np.array([[2.0, 3.0, 5.0], [2.0, 3.0, 5.0]])
    peak = np.array([[7.0, 11.0, 13.0], [7.0, 11.0, 0.0]])
    scale = np.asarray(host_contam_scale(0.0, _HOSTSNFRAC, host, peak, calib))

    lam_bin = np.asarray(calib.lam_bin)
    for j in range(2):
        both = (host[j] != 0) & (peak[j] != 0)
        fsum = lambda f: np.sum(np.where(both, f * lam_bin * lam_bin, 0.0))
        assert abs(scale[j] * fsum(host[j]) / fsum(peak[j]) - _HOSTSNFRAC) < 1e-9
    assert scale[0] != scale[1]        # the empty peak bin changed the ratio
    # 0.3*(7*100^2 + 11*200^2 + 13*400^2)/(2*100^2 + 3*200^2 + 5*400^2), and the
    # same with the third bin dropped. Only the LAMBIN^2 weight gives these.
    np.testing.assert_allclose(scale, [0.82659574, 1.09285714], rtol=1e-7)


def test_hostsnfrac_scale_is_epoch_independent():
    """FSUM_PEAK comes from GENSPEC_TRUE(ISPEC_PEAK), the SN at TOBS=0
    (snlc_sim.c:10578), not from the epoch being contaminated. Moving that
    epoch must therefore leave the scale untouched.
    """
    from jax_snlc_sim import run as sim_run

    def scales_at(trest):
        cfg = dict(
            n_sne=20, random_seed=122459)
        sim = sim_run.prepare(_CONTAM_INPUT, **cfg)
        simulation = None if sim is None else sim.config
        resources = sim
        for spec_cfg in simulation.spectra.take_spectrum:
            if spec_cfg['epoch_frame'] == 'REST':
                spec_cfg['epoch_range'] = (trest, trest)
        res, _, _ = sim_run.run_simulation(sim)
        n = len(np.asarray(res['parameters']['sim_redshift_cmb']))
        spectra, _ = sim_run.compute_spectra_for_mask(
            sim, res, np.ones(n, dtype=bool),
            key=root_key(simulation.run.random_seed))
        return {snid: next(s.scale_host_contam for s in v if not s.is_host)
                for snid, v in spectra.items()}

    at5, at12 = scales_at(5.0), scales_at(12.0)
    common = sorted(set(at5) & set(at12))
    assert len(common) > 5
    for snid in common:
        assert abs(at5[snid] - at12[snid]) < 1e-9, snid
    assert len(set(at5.values())) > 1     # the scale does vary per SN


@pytest.mark.live_snana
def test_jax_and_snana_agree_on_the_hostsnfrac_scale_range():
    """The scale is a host-to-SN brightness ratio, so it is set by the HOSTLIB
    and the SN population rather than by any one event. jax's spread must sit
    on SNANA's; a mis-scaled peak or host spectrum would displace it.
    """
    _, spec, _ = _snana_spec(_CONTAM_GENVERSION, 'host_spectra_contam')
    snana = spec[spec['MJD'] > 0]['SCALE_HOST_CONTAM']
    jax_scales = np.array([sn.scale_host_contam
                           for _, sn in _contam_spectra(_HOSTSNFRAC).values()])
    assert (spec[spec['MJD'] < -8]['SCALE_HOST_CONTAM'] == 0).all()
    assert abs(np.median(jax_scales) / np.median(snana) - 1.0) < 0.25, (
        f"jax median {np.median(jax_scales):.4f} vs "
        f"SNANA {np.median(snana):.4f}")


def _eazy_basis():
    """$SNDATA_ROOT's EAZY SPECBASIS table; the bundled tree carries a copy."""
    path = os.path.expandvars('$SNDATA_ROOT/models/EAZY/SNANA_SPECBASIS_EAZY.DAT')
    assert os.path.exists(path), f"EAZY basis missing from $SNDATA_ROOT: {path}"
    return path


def _tiny_calib(lam_min, lam_max):
    """Minimal spectrograph calibration for exercising the host binning."""
    from types import SimpleNamespace
    lam_min, lam_max = np.asarray(lam_min, float), np.asarray(lam_max, float)
    n = len(lam_min)
    return SimpleNamespace(
        lam_min=lam_min, lam_max=lam_max, lam_avg=0.5 * (lam_min + lam_max),
        lam_bin=lam_max - lam_min, n_lam=n,
        zp_sedmodel=np.full(n, 30.0), instrument_name='TEST')


def _host_flam(table, coeffs, z, calib, mwebv=0.0, static_data=None):
    from jax_snlc_sim.spectrograph.genspec import _host_bins_to_smeared
    opt = {'lamspike_idx': -1, 'smearing_matrix': np.eye(calib.n_lam)}
    sim_flam, _, _ = _host_bins_to_smeared(
        HostSpectra(table, np.asarray(coeffs)[None, :]),
        np.array([0]), np.array([z]), np.array([mwebv]), np.array([1.0 + z]),
        calib, opt, static_data or {})
    return np.asarray(sim_flam)[0]


def test_host_bins_apply_mw_extinction():
    """MWEBV > 0 must dim the stored host flux by the transmission factor;
    the fixture runs with OPT_MWEBV: 0, so nothing else covers this.
    """
    from jax_snlc_sim.hostlib.hostspec_reader import read_host_spec_table
    table = read_host_spec_table(_eazy_basis(), 'SPECBASIS')
    coeffs = np.full(table.n_spec, 0.3)
    calib = _tiny_calib([4000.0, 6000.0], [4100.0, 6100.0])
    static = {'mw_dust_law': _FlatUnitLaw(), 'rv_mw': 3.1}
    off = _host_flam(table, coeffs, 0.2, calib)
    on = _host_flam(table, coeffs, 0.2, calib, mwebv=0.1, static_data=static)
    expected = 10.0 ** (-0.4 * 3.1 * 0.1)     # flat A/Av == 1 curve
    assert np.allclose(on / off, expected, rtol=1e-6)


def test_host_bins_outside_the_basis_grid_are_zeroed():
    """A spectrograph bin whose rest range leaves the basis span carries no
    flux (sntools_host.c:2310-2311). The fixture's bins all land on-grid.
    """
    from jax_snlc_sim.hostlib.hostspec_reader import read_host_spec_table
    text = ("FLAM_SCALE: 1.0\nVARNAMES: ROW WAVELENGTH SPECBASIS00\n"
            "ROW: 1 4000.0 1.0\nROW: 2 4010.0 1.0\nROW: 3 4020.0 1.0\n")
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.dat', delete=False) as fh:
        fh.write(text); path = fh.name
    table = read_host_spec_table(path, 'SPECBASIS')
    # wave_min = [3995, 4005, 4015], wave_max = [.., .., 4025]. Bin 1 straddles
    # wave_min[-1], so the trapezoid alone would give it flux; SNANA does not.
    calib = _tiny_calib([4002.0, 4010.0], [4008.0, 4020.0])
    flam = _host_flam(table, [1.0], 0.0, calib)
    assert flam[0] > 0.0
    assert flam[1] == 0.0


def test_host_magnitude_ignores_mw_extinction():
    """SNANA builds the host mag from the un-extincted flux (genSpec_HOSTLIB):
    the mag is the galaxy's true brightness while the spectrum is what is
    observed. jax reproduces that, so genmag must not move with MWEBV.
    """
    from jax_snlc_sim.hostlib.hostspec_reader import read_host_spec_table
    from jax_snlc_sim.spectrograph.genspec import _host_bins_to_smeared
    table = read_host_spec_table(_eazy_basis(), 'SPECBASIS')
    calib = _tiny_calib([4000.0, 6000.0], [4100.0, 6100.0])
    static = {'mw_dust_law': _FlatUnitLaw(), 'rv_mw': 3.1}
    opt = {'lamspike_idx': -1, 'smearing_matrix': np.eye(calib.n_lam)}
    bundle = HostSpectra(table, np.full((1, table.n_spec), 0.3))

    def run(mwebv, sd):
        flam, genmag, _ = _host_bins_to_smeared(
            bundle, np.array([0]), np.array([0.2]), np.array([mwebv]),
            np.array([1.2]), calib, opt, sd)
        return np.asarray(flam)[0], np.asarray(genmag)[0]
    flam0, mag0 = run(0.0, {})
    flam1, mag1 = run(0.1, static)
    assert np.all(flam1 < flam0)                    # the spectrum is extincted
    assert np.allclose(mag1, mag0, rtol=1e-9)       # the magnitude is not
