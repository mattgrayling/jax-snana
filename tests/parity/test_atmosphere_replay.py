"""Atmosphere / DCR (ATMOSPHERE_OPTMASK) replay parity.

ATMOSPHERE_OPTMASK>0 makes SNANA append per-obs airmass and differential
chromatic refraction (DCR) columns to the photometry table, driven by the
auto-enabled SED-true spectrograph path. This is output-only — it never feeds
back into flux/noise — so jax computes it as a post-pass.

Two parity checks:

- **AIRMASS** is pure geometry (MJD, RA, DEC, telescope geo), so jax's
  ``airmass_from_mjd`` reproduces SNANA's AIRMASS byte-for-byte when fed SNANA's
  own MJD/RA/DEC (float32-output rounding only).
- **SIM_DCR_dMAG / SIM_DCR_dRA / SIM_DCR_dDEC** are the SED-weighted DCR truth
  shifts (the RA/DEC ones band-average subtracted). The dense-cadence fixture
  gives several obs per band, so the band average is non-trivial; replaying the
  SNANA truth reproduces them to the SED-integral quadrature level (median well
  below a milliarcsec — the averaging weights carry only weak SNR noise).
- **Measured dRA/dDEC** additionally carry the per-epoch Gaussian coord-smear,
  drawn from jax's own RNG, so they match SNANA only in scatter amplitude.
"""

import numpy as np
from astropy.io import fits
from replay_snana_dump import build_override_parameters

from jax_snlc_core.snana_files import load_snana_dump, resolve_fits_path
from jax_snlc_sim import run as sim_run
from jax_snlc_sim.output import save_snana_outputs
from jax_snlc_sim.atmosphere import airmass_from_mjd
from jax_snlc_sim.atmosphere import read_observatory_sites


def _phot(path, cols):
    with fits.open(resolve_fits_path(path)) as h:
        return {n: np.array(h[1].data[n]) for n in cols}


def _head(path):
    with fits.open(resolve_fits_path(path)) as h:
        return {n: np.array(h[1].data[n]) for n in h[1].data.names}


def _obs_key(head, phot):
    """{(cid, band): [(mjd, row), ...]} from HEAD PTROBS ranges."""
    snid = np.array([int(str(s).strip()) for s in head['SNID']])
    pmin = np.array(head['PTROBS_MIN'], int)
    pmax = np.array(head['PTROBS_MAX'], int)
    out = {}
    for i in range(len(snid)):
        for r in range(pmin[i] - 1, pmax[i]):
            b = str(phot['BAND'][r]).strip()
            out.setdefault((snid[i], b), []).append((float(phot['MJD'][r]), r))
    return out


def test_airmass_byte_exact(snana_fixture):
    """jax airmass reproduces SNANA's AIRMASS on SNANA's own MJD/RA/DEC."""
    fx = snana_fixture("atmosphere_replay")
    head = _head(fx.head_fits)
    phot = _phot(fx.phot_fits, ('MJD', 'AIRMASS'))
    site = read_observatory_sites()['LSST']
    ra = np.array(head['RA'], float)
    dec = np.array(head['DEC'], float)
    pmin = np.array(head['PTROBS_MIN'], int)
    pmax = np.array(head['PTROBS_MAX'], int)
    mjd = np.array(phot['MJD'], float)
    air = np.array(phot['AIRMASS'], float)
    maxerr, n = 0.0, 0
    for i in range(len(ra)):
        sl = slice(pmin[i] - 1, pmax[i])
        g = airmass_from_mjd(mjd[sl], np.full(pmax[i] - pmin[i] + 1, ra[i]),
                             np.full(pmax[i] - pmin[i] + 1, dec[i]),
                             site.geo_lat, site.geo_lon)
        ja = np.array(g['airmass'])
        ok = np.isfinite(air[sl]) & (air[sl] > -8.9)
        if ok.sum():
            maxerr = max(maxerr, float(np.abs(ja[ok] - air[sl][ok]).max()))
            n += int(ok.sum())
    assert n > 50
    assert maxerr < 1e-4, f"AIRMASS max|jax-SNANA|={maxerr:.2e} over {n} obs"


def test_dcr_magshift_replay(snana_fixture, tmp_path):
    """Replaying SNANA SNe reproduces SIM_DCR_dMAG to the SED-integral level."""
    fx = snana_fixture("atmosphere_replay")
    dump = load_snana_dump(str(fx.dump), dedup_cid=True)
    head_cols = _head(fx.head_fits)
    cols = ('MJD', 'BAND', 'dRA', 'dDEC', 'SIM_DCR_dRA', 'SIM_DCR_dDEC', 'SIM_DCR_dMAG')
    s_phot = _phot(fx.phot_fits, cols)

    cfg = dict(n_sne=dump.size)
    sim = sim_run.prepare(str(fx.input_file), **cfg)
    simulation = None if sim is None else sim.config
    resources = sim
    assert simulation.instrument.atmos_optmask == 513
    assert simulation.effective_spectrograph_optmask & 128

    override = build_override_parameters(dump.columns, simulation, resources, head_columns=head_cols)
    results, sel, spectra_cache = sim_run.run_simulation(sim, override_parameters=override)
    save_snana_outputs(sim, results, "JAXATM", sel, spectra_by_sn=spectra_cache, output_dir=str(tmp_path))
    jdir = tmp_path / "JAXATM"
    j_head = _head(jdir / "JAXATM_HEAD.FITS")
    j_phot = _phot(jdir / "JAXATM_PHOT.FITS", cols)

    # the 6 SNANA atmosphere columns are all present
    with fits.open(resolve_fits_path(jdir / "JAXATM_PHOT.FITS")) as h:
        assert {'dRA', 'dDEC', 'AIRMASS', 'SIM_DCR_dRA', 'SIM_DCR_dDEC',
                'SIM_DCR_dMAG'} <= set(h[1].columns.names)

    # join obs by (CID, BAND, nearest MJD) and collect matched (SNANA, jax) pairs
    s_idx, j_idx = _obs_key(head_cols, s_phot), _obs_key(j_head, j_phot)
    pairs = {c: [] for c in ('dRA', 'dDEC', 'SIM_DCR_dRA', 'SIM_DCR_dDEC', 'SIM_DCR_dMAG')}
    for key, s_list in s_idx.items():
        j_list = j_idx.get(key)
        if not j_list:
            continue
        jm = np.array([m for m, _ in j_list])
        for mjd_s, rs in s_list:
            k = int(np.argmin(np.abs(jm - mjd_s)))
            if abs(jm[k] - mjd_s) > 0.02:
                continue
            rj = j_list[k][1]
            for c in pairs:
                sv, jv = s_phot[c][rs], j_phot[c][rj]
                if abs(sv) < 90 and abs(jv) < 90:
                    pairs[c].append((sv, jv))

    # truth DCR (band-average subtracted) is near-deterministic: the coord-smear
    # only perturbs the SNR-based averaging weights, so jax matches SNANA per obs.
    for c in ('SIM_DCR_dRA', 'SIM_DCR_dDEC', 'SIM_DCR_dMAG'):
        v = np.array(pairs[c])
        assert len(v) > 500, f"{c}: only {len(v)} matched obs"
        med = np.median(np.abs(v[:, 0] - v[:, 1]))
        assert med < 1e-3, f"{c} median|Δ|={med:.2e} over {len(v)} obs"
    # the fixture actually exercises band-averaging (dense cadence, >1 obs/band):
    # SIM_DCR_dRA spans a real range, not a vacuous all-zero comparison.
    sim_dra = np.array(pairs['SIM_DCR_dRA'])[:, 0]
    assert sim_dra.max() - sim_dra.min() > 0.5, "band-average not exercised"

    # the per-epoch Gaussian coord-smear is (measured dRA - truth SIM_DCR_dRA).
    # Its RNG is independent of SNANA's, so it matches only in scatter amplitude;
    # compare each code's own population std (no cross-match) to avoid the DCR
    # truth (which dominates dRA) leaking into a per-obs comparison.
    def smear_std(p, meas, truth):
        d, t = p[meas], p[truth]
        ok = (np.abs(d) < 90) & (np.abs(t) < 90)
        return np.std(d[ok] - t[ok])
    for meas, truth in (('dRA', 'SIM_DCR_dRA'), ('dDEC', 'SIM_DCR_dDEC')):
        ratio = smear_std(j_phot, meas, truth) / smear_std(s_phot, meas, truth)
        assert 0.5 < ratio < 2.0, f"{meas} coord-smear scatter ratio jax/SNANA={ratio:.2f}"


def test_atmosphere_text_format(snana_fixture, tmp_path):
    """TEXT output carries the atmosphere columns: the 6 per-obs PHOT columns in
    the VARLIST and per-band RA_AVG/DEC_AVG in the header (text.c:152, 980)."""
    fx = snana_fixture("atmosphere_replay")
    cfg = dict(n_sne=5, random_seed=1)
    sim = sim_run.prepare(str(fx.input_file), **cfg)
    simulation = None if sim is None else sim.config
    resources = sim
    simulation.output.format_mask = 130          # TEXT(2) + ATMOS(128)
    res, sel, spectra_cache = sim_run.run_simulation(sim)
    save_snana_outputs(sim, res, "ATXT", sel, spectra_by_sn=spectra_cache, output_dir=str(tmp_path))
    dats = sorted((tmp_path / "ATXT").glob("*.DAT"))
    assert dats, "no TEXT light curves written"
    txt = dats[0].read_text()
    assert "RA_AVG:" in txt and "DEC_AVG:" in txt
    varlist = next(l for l in txt.splitlines() if l.startswith("VARLIST:"))
    for c in ('dRA', 'dDEC', 'AIRMASS', 'SIM_DCR_dRA', 'SIM_DCR_dDEC', 'SIM_DCR_dMAG'):
        assert c in varlist, f"{c} missing from TEXT VARLIST"


def test_atmosphere_format_combos(snana_fixture, tmp_path):
    """ATMOS composes with the FORMAT_MASK modifier bits: COMPACT drops the
    per-obs atmosphere columns (SNANA gates them behind WRFULL, fits.c:888), and
    CIDRAN keeps them while remapping every CID (HEAD, PHOT, .DCR) consistently."""
    fx = snana_fixture("atmosphere_replay")

    def run(fmask, tag):
        cfg = dict(n_sne=8, random_seed=3)
        sim = sim_run.prepare(str(fx.input_file), **cfg)
        simulation = None if sim is None else sim.config
        resources = sim
        simulation.output.format_mask = fmask
        res, sel, spectra_cache = sim_run.run_simulation(sim)
        out = tmp_path / tag
        save_snana_outputs(sim, res, tag, sel, spectra_by_sn=spectra_cache, output_dir=str(out))
        return out / tag

    atmos_cols = ('dRA', 'dDEC', 'AIRMASS', 'SIM_DCR_dRA')
    # COMPACT(64) + FITS(32) + ATMOS(128): atmosphere columns dropped
    d = run(64 | 32 | 128, "CMP")
    with fits.open(resolve_fits_path(d / "CMP_PHOT.FITS")) as h:
        assert not (set(atmos_cols) & set(h[1].columns.names))

    # CIDRAN(16) + FITS(32) + ATMOS(128): columns kept, CIDs remapped everywhere
    d = run(16 | 32 | 128, "CDR")
    with fits.open(resolve_fits_path(d / "CDR_HEAD.FITS")) as h:
        hcids = {int(str(s).strip()) for s in h[1].data['SNID']}
    with fits.open(resolve_fits_path(d / "CDR_PHOT.FITS")) as h:
        assert set(atmos_cols) <= set(h[1].columns.names)
    dcr = list((tmp_path / "CDR" / "CDR").glob("*.DCR"))
    assert dcr, ".DCR dump not written under CIDRAN"
    dcids = {int(l.split()[1]) for l in open(dcr[0]) if l.startswith('ROW:')}
    assert dcids and dcids <= hcids, "DUMP_DCR CIDs inconsistent with HEAD"


def test_atmosphere_batch_invariant(snana_fixture, tmp_path):
    """The DCR columns (incl. the per-epoch coord-smear RNG) are identical whether
    the population is generated in one batch or many: the smear is keyed on the
    global SN index and the post-pass runs on the concatenated results."""
    fx = snana_fixture("atmosphere_replay")
    cols = ('MJD', 'BAND', 'AIRMASS', 'dRA', 'dDEC', 'SIM_DCR_dRA',
            'SIM_DCR_dDEC', 'SIM_DCR_dMAG')

    def run(batch):
        cfg = dict(n_sne=40,
                                  random_seed=4242, batch_size=batch)
        sim = sim_run.prepare(str(fx.input_file), **cfg)
        simulation = None if sim is None else sim.config
        resources = sim
        res, sel, spectra_cache = sim_run.run_simulation(sim)
        out = tmp_path / f"c{batch}"
        save_snana_outputs(sim, res, "C", sel, spectra_by_sn=spectra_cache, output_dir=str(out))
        return _head(out / "C" / "C_HEAD.FITS"), _phot(out / "C" / "C_PHOT.FITS", cols)

    def per_obs(h, p):
        sn = np.array([int(str(s).strip()) for s in h['SNID']])
        lo, hi = np.array(h['PTROBS_MIN'], int), np.array(h['PTROBS_MAX'], int)
        return {(sn[i], round(float(p['MJD'][r]), 3), str(p['BAND'][r]).strip()): r
                for i in range(len(sn)) for r in range(lo[i] - 1, hi[i])}

    h1, p1 = run(1000)          # single batch
    h2, p2 = run(7)             # ~6 batches
    i1, i2 = per_obs(h1, p1), per_obs(h2, p2)
    common = set(i1) & set(i2)
    assert len(common) > 500
    for c in ('AIRMASS', 'dRA', 'dDEC', 'SIM_DCR_dRA', 'SIM_DCR_dDEC', 'SIM_DCR_dMAG'):
        d = max(abs(float(p1[c][i1[k]]) - float(p2[c][i2[k]])) for k in common)
        assert d == 0.0, f"{c} not batch-invariant: max|Δ|={d}"
