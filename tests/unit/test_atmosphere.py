"""Unit tests for the atmosphere/DCR physics (jax_snlc_sim/atmosphere.py).

Structural properties of the refraction and DCR integral (dispersion direction,
tan-zenith scaling, null/edge handling, config/survey parsing) that the bundled
SNANA fixture does not exercise. Byte-level agreement with SNANA — AIRMASS and
SIM_DCR — is covered by tests/parity/test_atmosphere_replay.py.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jax_snlc_sim.atmosphere import (
    index_refrac_minus1, dcr_weighted_arcsec, _parallactic_shift,
    airmass_from_mjd, compute_atmosphere, site_refrac_draws,
    DCR_BATCH, DCR_NULL_ARCSEC)
from jax_snlc_sim.inputs import parse_snana_input_file
from jax_snlc_core.random import event_keys, root_key


def test_index_refrac_dispersion_and_pressure():
    """n-1 is positive, decreases with wavelength (normal dispersion), and scales
    with site pressure (lower pressure -> less refraction)."""
    nm1 = index_refrac_minus1(np.array([3000.0, 5000.0, 10000.0]), 15.0, 760.0, 0.0)
    assert nm1[0] > nm1[1] > nm1[2] > 0.0
    high = index_refrac_minus1(np.array([5000.0]), 15.0, 760.0, 0.0)[0]
    low = index_refrac_minus1(np.array([5000.0]), 15.0, 550.0, 0.0)[0]
    assert low / high == pytest.approx(550.0 / 760.0, rel=0.05)


def test_index_refrac_per_obs_site_jitter():
    """SITE_SIGMAS path: per-obs T/BP/PWV (column vectors) yield per-obs n(lambda)
    that varies across obs and reduces to the shared-site value when unjittered."""
    wave = np.array([4000.0, 6000.0, 8000.0])
    temp = np.array([[10.0], [12.0], [8.0]])          # (n_obs, 1) jittered site
    bp = np.array([[600.0], [610.0], [590.0]])
    pwv = np.array([[1.0], [1.2], [0.8]])
    nm1 = index_refrac_minus1(wave[None, :], temp, bp, pwv)   # (n_obs, n_wave)
    assert nm1.shape == (3, 3)
    assert np.std(nm1[:, 0]) > 0.0                    # varies across obs
    single = index_refrac_minus1(wave, 10.0, 600.0, 1.0)
    np.testing.assert_allclose(nm1[0], single, rtol=1e-12)


def _flat_grid():
    wave = np.linspace(4000.0, 7000.0, 300)
    trans = np.ones_like(wave)
    return wave, trans


def test_dcr_scales_with_tan_zenith():
    """Net DCR is linear in tan(zenith)."""
    wave, trans = _flat_grid()
    sim = (wave / 5000.0) ** -1.0                # blue-ish transient
    cal = np.ones_like(wave)                     # flat reference star
    sim2d = sim[None, :].repeat(2, 0)
    cal2d = cal[None, :].repeat(2, 0)
    tr2d = trans[None, :].repeat(2, 0)
    nm1 = index_refrac_minus1(wave, 10.0, 600.0, 1.0)
    net, _ = dcr_weighted_arcsec(sim2d, cal2d, tr2d, wave, nm1,
                                 np.array([1.0, 2.0]))
    assert net[1] == pytest.approx(2.0 * net[0], rel=1e-6)
    assert abs(net[0]) > 0.0


def test_dcr_zero_for_matching_sed():
    """A transient SED identical to the reference star gives zero net DCR."""
    wave, trans = _flat_grid()
    flat = np.ones_like(wave)[None, :]
    nm1 = index_refrac_minus1(wave, 10.0, 600.0, 1.0)
    net, _ = dcr_weighted_arcsec(flat, flat, trans[None, :], wave, nm1,
                                 np.array([1.5]))
    assert net[0] == pytest.approx(0.0, abs=1e-9)


def test_dcr_null_when_no_sed():
    """Zero SED flux (obs outside the model phase range) -> DCR null sentinel."""
    wave, trans = _flat_grid()
    zero = np.zeros_like(wave)[None, :]
    cal = np.ones_like(wave)[None, :]
    nm1 = index_refrac_minus1(wave, 10.0, 600.0, 1.0)
    net, _ = dcr_weighted_arcsec(zero, cal, trans[None, :], wave, nm1,
                                 np.array([1.5]))
    assert net[0] == DCR_NULL_ARCSEC


def test_parallactic_null_passthrough():
    """A null DCR angle projects to the null coordinate-shift sentinel."""
    geom = {k: np.array([0.3]) for k in
            ('sin_h', 'sin_alt', 'cos_alt', 'sin_dec', 'cos_dec')}
    _, ra_shift, dec_shift = _parallactic_shift(
        np.array([DCR_NULL_ARCSEC]), geom, np.sin(-0.5), np.cos(-0.5))
    assert ra_shift[0] == pytest.approx(99.0 / 3600.0)
    assert dec_shift[0] == pytest.approx(99.0 / 3600.0)


def test_atmosphere_optmask_auto_enables_sed_true():
    """optmask>0 auto-enables the ATMOS format and the SED-true spectrograph
    path, without claiming the user asked for spectra output."""
    from jax_snlc_sim.simulation import SimulationConfig
    from jax_snlc_sim.simulation import Instrument
    c = SimulationConfig(instrument=Instrument(
        atmos_optmask=513, atmos_sedstar_file='star.sed',
        dcr_coordres_poly=[0.0, 1.0], dcr_magshift_poly=[0.0, 0.5],
        coordres_floor=0.01))
    assert c.output.format_mask & 128
    assert c.effective_spectrograph_optmask & 128
    assert c.spectra.spectrograph_optmask == 0  # user did not request spectra output
    assert c.spectra.lambin_sed_true == 10.0


def test_atmosphere_keys_reach_the_config(tmp_path):
    """Every ATMOSPHERE_* key the input carries lands on Instrument."""
    fixture = (Path(__file__).parent.parent / "fixtures" / "snana"
               / "atmosphere_replay" / "input.input")
    inp = tmp_path / 'atm.input'
    inp.write_text(open(fixture).read() + "\n"
                   "ATMOSPHERE_OPTMASK: 513\n"
                   "ATMOSPHERE_SEDSTAR_FILE: star.sed\n"
                   "ATMOSPHERE_DCR_COORDRES_POLY: 0.0,1.0\n"
                   "ATMOSPHERE_COORDRES_FLOOR: 0.01\n"
                   "ATMOSPHERE_SITE_SIGMAS: 1.0,2.0,3.0\n")
    a = parse_snana_input_file(str(inp)).instrument
    assert a.atmos_optmask == 513
    assert a.atmos_sedstar_file == 'star.sed'
    assert a.dcr_coordres_poly == [0.0, 1.0]
    assert a.coordres_floor == 0.01
    assert (a.sigma_site_temp, a.sigma_site_bp, a.sigma_site_pwv) == (1.0, 2.0, 3.0)


def test_survey_def_site_params():
    """SURVEY.DEF geo parsing derives site T/BP/PWV from altitude."""
    import os
    from jax_snlc_sim.atmosphere import read_observatory_sites
    sites = read_observatory_sites(os.path.join(os.environ['SNDATA_ROOT'], 'SURVEY.DEF'))
    lsst = sites['LSST']
    assert lsst.geo_lat == pytest.approx(-30.244639)
    assert lsst.geo_alt == pytest.approx(2647.0)
    assert lsst.temperature == pytest.approx(20.0 - 0.006 * 2647.0)


def test_site_refrac_draws_per_bin_and_stable():
    """ATMOSPHERE_SIGMA_SITE redraws T/BP/PWV for every wavelength bin of the
    DCR integral (sntools_sim_atmosphere.c:752-755, 867-877): draws must vary
    across bins and obs, stay clipped at 3 sigma, and be stable per
    (event, obs, bin) under any batching."""
    site = SimpleNamespace(temperature=10.0, pressure=600.0, pwv=1.0)
    sn_keys = event_keys(root_key(42), 'DCR_SITE', np.array([0, 1]))
    sigma = (2.0, 5.0, 0.3)
    temp, bp, pwv = site_refrac_draws(sn_keys, np.array([0, 3]), 50, site, sigma)
    assert temp.shape == (2, 50)
    assert np.std(temp[0]) > 0.0                 # per-bin, not per-obs
    assert not np.allclose(temp[0], temp[1])     # per-obs distinct
    assert np.max(np.abs(temp - 10.0)) <= 3.0 * 2.0 + 1e-9
    temp_again, _, _ = site_refrac_draws(
        sn_keys[1:2], np.array([3]), 50, site, sigma)
    np.testing.assert_array_equal(temp_again[0], temp[1])


def test_compute_atmosphere_batched_matches_direct():
    """The flattened, DCR_BATCH-batched integral must reproduce the direct
    single-obs weighted-DCR computation, including with more valid obs than
    one batch holds."""
    n_sne, max_obs, n_wave = 30, 200, 40
    assert n_sne * max_obs > DCR_BATCH
    wave = np.linspace(4000.0, 7000.0, n_wave)
    sed = (wave / 5000.0) ** -1.0
    cal = np.ones_like(wave)
    trans = np.ones((1, n_wave))
    site = SimpleNamespace(temperature=10.0, pressure=600.0, pwv=1.0,
                           geo_lat=-30.24, geo_lon=-70.75)
    mjd = 60000.0 + np.linspace(0.0, 3.0, n_sne * max_obs).reshape(
        n_sne, max_obs)
    ra = np.full(n_sne, 30.0)
    dec = np.full(n_sne, -10.0)
    out = compute_atmosphere(
        mjd=mjd, filter_idx=np.zeros((n_sne, max_obs), dtype=np.int32),
        valid_mask=np.ones((n_sne, max_obs), dtype=bool),
        ra_deg=ra, dec_deg=dec, global_indices=np.arange(n_sne),
        psf_fwhm=np.full((n_sne, max_obs), 1.0),
        snr_true=np.full((n_sne, max_obs), 10.0),
        snr_obs=np.full((n_sne, max_obs), 10.0),
        spec_flam=sed[None, :],
        spec_row=np.zeros((n_sne, max_obs), dtype=np.int32),
        wave=wave, trans_by_filter=trans, cal_flux=cal, site=site,
        coordres_poly=[0.0, 1.0], magshift_poly=[0.0, 0.5],
        coordres_floor=0.01, sigma_site=None, key=root_key(7))

    geom = airmass_from_mjd(mjd, ra[:, None], dec[:, None],
                            site.geo_lat, site.geo_lon)
    nm1 = index_refrac_minus1(wave, site.temperature, site.pressure, site.pwv)
    for i, j in [(0, 0), (17, 150), (29, 199)]:
        net, _ = dcr_weighted_arcsec(
            sed[None, :], cal[None, :], trans, wave, nm1,
            np.array([geom['tan_zenith'][i, j]]))
        assert out['sim_dcr'][i, j] == pytest.approx(net[0], rel=1e-12)
