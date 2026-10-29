"""SEDMODEL true-SED spectra mode (SPECTROGRAPH_OPTMASK bit 128) replay parity.

LAMBIN_SED_TRUE sets bit 128: SNANA builds an IDEAL spectrograph and writes a
noiseless true-SED spectrum at every photometry epoch using the reduced SPEC
format (uniform grid as header scalars LAMMIN/LAMMAX/LAMBIN/NBIN_LAM, SIM_FLAM
the only flux column, -777 end-of-spectrum markers). Replaying the SNANA truth
reproduces the same epochs and the same true SIM_FLAM per bin.

SIM_FLAM carries the documented trapezoid-vs-left-Riemann bin-integration
systematic: median ~0.5%, correlated with SED slope.
Bins below 1e-4 of the peak flux are SED tails where the relative comparison is
meaningless (both sides ~0), so the gate is on significant bins.
"""

import os
from dataclasses import replace

import numpy as np
from astropy.io import fits
from replay_snana_dump import build_override_parameters

from jax_snlc_core.snana_files import load_snana_dump, resolve_fits_path
from jax_snlc_sim import run as sim_run
from jax_snlc_sim.output import save_snana_outputs
from jax_snlc_sim.inputs import parse_snana_input_file


def _load_sedtrue_spec(path):
    """Reduced SED-true SPEC: grid from HDU1 scalars, SIM_FLAM from HDU2.
    Returns {snid: [(mjd, lam, sim_flam, instrument, synmag), ...]} by MJD."""
    out = {}
    with fits.open(resolve_fits_path(path)) as h:
        hdr, flx = h[1].data, h[2].data
        sf = flx['SIM_FLAM']
        bands = [c.name.split('SIM_SYNMAG_')[1] for c in h[1].columns
                 if c.name.startswith('SIM_SYNMAG_')]
        for row in hdr:
            lo, hi = int(row['PTRSPEC_MIN']) - 1, int(row['PTRSPEC_MAX'])
            lammin, lambin, nb = (float(row['LAMMIN']), float(row['LAMBIN']),
                                  int(row['NBIN_LAM']))
            lam = lammin + (np.arange(nb) + 0.5) * lambin
            synmag = {b: float(row[f'SIM_SYNMAG_{b}']) for b in bands}
            out.setdefault(int(str(row['SNID']).strip()), []).append(
                (float(row['MJD']), lam, np.asarray(sf[lo:hi], dtype=float),
                 str(row['INSTRUMENT']).strip(), synmag))
    for v in out.values():
        v.sort(key=lambda t: t[0])
    return out


def _load_verify(path):
    """Return {(cid, round(mjd,1), band): (mag, mag_verify)}."""
    out = {}
    with open(path) as f:
        for line in f:
            if not line.startswith('SN:'):
                continue
            p = line.split()
            out[(int(p[1]), round(float(p[3]), 1), p[6])] = (float(p[7]), float(p[8]))
    return out


def test_sedtrue_replay(snana_fixture, tmp_path):
    fx = snana_fixture("sedtrue_replay")
    dump = load_snana_dump(str(fx.dump), dedup_cid=True)
    with fits.open(fx.head_fits) as hdul:
        head_cols = {n: hdul[1].data[n] for n in hdul[1].data.names}
    config = dict(n_sne=dump.size)
    sim = sim_run.prepare(str(fx.input_file), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    # LAMBIN_SED_TRUE sets the SED-true bit on the working mask, not the
    # user's own (snlc_sim.c:2619-2624)
    assert simulation.spectra.spectrograph_optmask == 0
    assert simulation.effective_spectrograph_optmask == 128

    override = build_override_parameters(dump.columns, simulation, resources,
                                         head_columns=head_cols)
    results, sel, spectra_cache = sim_run.run_simulation(sim, override_parameters=override)
    save_snana_outputs(sim, results, "SEDT", sel, spectra_by_sn=spectra_cache, output_dir=str(tmp_path))

    snana = _load_sedtrue_spec(fx.root / "SPEC.FITS")
    jax_spec = _load_sedtrue_spec(tmp_path / "SEDT" / "SEDT_SPEC.FITS")
    common = sorted(set(snana) & set(jax_spec))
    assert len(common) > 40

    peakflux = max(float(np.max(sf)) for c in common for _, _, sf, _, _ in snana[c])
    reldiff, synmag_absdiff, n_pairs, n_snana, n_unmatched = [], [], 0, 0, 0
    for cid in common:
        for (m_s, lam_s, sf_s, inst_s, syn_s) in snana[cid]:
            n_snana += 1
            # match SNANA's epoch to the nearest jax epoch (dedup is 0.05 d);
            # jax can emit one extra epoch at the TREST-window edge where the
            # z_helio schedule window differs from SNANA's z_cmb window.
            cand = [x for x in jax_spec[cid] if abs(x[0] - m_s) < 0.06]
            if not cand:
                n_unmatched += 1
                continue
            m_j, lam_j, sf_j, inst_j, syn_j = cand[0]
            assert inst_s == inst_j == 'IDEAL'
            assert len(lam_s) == len(lam_j)
            assert np.allclose(lam_s, lam_j, atol=0.5)
            n_pairs += 1
            ok = (sf_s > 1e-4 * peakflux) & (sf_j > 0)
            reldiff.extend(np.abs(sf_j[ok] / sf_s[ok] - 1.0))
            # per-band synthetic mag (SIM_SYNMAG); skip zero-flux sentinels
            for b, ms in syn_s.items():
                if ms < 90.0 and syn_j.get(b, 99.0) < 90.0:
                    synmag_absdiff.append(abs(syn_j[b] - ms))

    reldiff = np.asarray(reldiff)
    synmag_absdiff = np.asarray(synmag_absdiff)
    assert n_pairs > 100
    assert len(reldiff) > 20000
    # exact epoch parity: every SNANA spectrum reproduced and no jax extras.
    # genSpec_SALT2 (genmag_SALT2.c:3591) zeroes spectra outside the model
    # phase range; jax drops the same epochs (Trest_helio in [DAYMIN, DAYMAX]).
    assert n_unmatched == 0
    n_jax = sum(len(v) for v in jax_spec.values())
    assert n_jax == n_snana == n_pairs
    # Two systematics, neither of them jax being wrong: SNANA integrates each
    # spectrograph bin as a left-Riemann sum where jax uses a trapezoid, and
    # jax evaluates the SED on the run's wavelength grid rather than the
    # model's native one, which wavelength_grid_points raises.
    assert float(np.median(reldiff)) < 0.01, \
        f"SIM_FLAM median rel diff {np.median(reldiff):.4f}"
    assert float(np.percentile(reldiff, 95)) < 0.05, \
        f"SIM_FLAM p95 rel diff {np.percentile(reldiff, 95):.4f}"
    # SIM_SYNMAG: synthetic photometry from the true SED, same bin-error level
    assert len(synmag_absdiff) > 300
    assert float(np.median(synmag_absdiff)) < 0.005, \
        f"SIM_SYNMAG median abs diff {np.median(synmag_absdiff):.5f}"

    # VERIFY_SED_TRUE table: broadband MAG and synthetic MAG_VERIFY per epoch
    snana_v = _load_verify(fx.root / "VERIFY_SED_TRUE.DAT")
    # written at the output base with no directory prefix, like SNANA
    # (snlc_sim.c:11569)
    jax_v = _load_verify(tmp_path / "SEDT_VERFIY_SED_TRUE.DAT")
    keys = set(snana_v) & set(jax_v)
    assert len(keys) > 100
    mag_d = np.array([abs(jax_v[k][0] - snana_v[k][0]) for k in keys])
    ver_d = np.array([abs(jax_v[k][1] - snana_v[k][1]) for k in keys
                      if snana_v[k][1] < 90.0 and jax_v[k][1] < 90.0])
    assert float(np.median(mag_d)) < 0.005, f"VERIFY MAG median {np.median(mag_d):.5f}"
    assert float(np.median(ver_d)) < 0.005, f"VERIFY MAG_VERIFY median {np.median(ver_d):.5f}"


def _load_dat_spectra(path):
    """Parse the SNSPEC block of a TEXT .DAT: {mjd: (varnames, [rows])}."""
    out, cur, varn, rows = {}, None, None, []
    for line in open(path):
        t = line.split()
        if not t:
            continue
        if t[0] == 'VARNAMES_SPEC:':
            varn = t[1:]
        elif t[0] == 'SPECTRUM_MJD:':
            cur = round(float(t[1]), 2)
            rows = []
        elif t[0] == 'SPEC:':
            rows.append([float(x) for x in t[1:]])
        elif t[0] == 'SPECTRUM_END:':
            out[cur] = (varn, rows)
    return out


def test_sedtrue_text_spectra(snana_fixture, tmp_path):
    """FORMAT_MASK=2 SED-true run: the TEXT SNSPEC block matches SNANA's format
    and its SIM_GENFLAM equals the FITS SIM_FLAM (same quantity, both formats)."""
    fx = snana_fixture("sedtrue_replay")
    dump = load_snana_dump(str(fx.dump), dedup_cid=True)
    with fits.open(fx.head_fits) as hdul:
        head_cols = {n: hdul[1].data[n] for n in hdul[1].data.names}
    config = dict(n_sne=dump.size)
    parsed = parse_snana_input_file(str(fx.input_file))
    parsed.output.format_mask = 2
    parsed.numerical = replace(parsed.numerical, n_sne=config["n_sne"])
    sim = sim_run.prepare(parsed)
    simulation = None if sim is None else sim.config
    resources = sim
    override = build_override_parameters(dump.columns, simulation, resources,
                                         head_columns=head_cols)
    results, sel, spectra_cache = sim_run.run_simulation(sim, override_parameters=override)
    save_snana_outputs(sim, results, "SEDT2", sel, spectra_by_sn=spectra_cache, output_dir=str(tmp_path))

    snana = _load_sedtrue_spec(fx.root / "SPEC.FITS")
    dats = sorted((tmp_path / "SEDT2").glob("SEDT2_SN*.DAT"))
    assert len(dats) > 10
    reldiff, checked, genmags, genflams = [], 0, [], []
    for dat in dats:
        cid = int(dat.stem.split("_SN")[1])
        if cid not in snana:
            continue
        spec = _load_dat_spectra(dat)
        for (m_s, lam_s, sf_s, _, _) in snana[cid]:
            block = spec.get(round(m_s, 2))
            if block is None:
                continue
            varn, rows = block
            assert varn == ['LAMMIN', 'LAMMAX', 'FLAM', 'FLAMERR',
                            'SIM_GENFLAM', 'SIM_GENMAG', 'SNR']
            assert len(rows) == len(sf_s)
            genflam = np.array([r[varn.index('SIM_GENFLAM')] for r in rows])
            genmag = np.array([r[varn.index('SIM_GENMAG')] for r in rows])
            ok = (sf_s > 1e-4 * sf_s.max()) & (genflam > 0)
            reldiff.extend(np.abs(genflam[ok] / sf_s[ok] - 1.0))
            genmags.extend(genmag[ok])
            genflams.extend(genflam[ok])
            checked += 1
    assert checked > 20
    # jax's TEXT SIM_GENFLAM against SNANA's SIM_FLAM, so the same two
    # systematics bound it as the FITS comparison above
    assert float(np.median(reldiff)) < 0.01
    assert float(np.percentile(reldiff, 95)) < 0.05
    # SIM_GENMAG: finite, sensible, and anti-correlated with SIM_GENFLAM
    genmags, genflams = np.asarray(genmags), np.asarray(genflams)
    assert np.all(np.isfinite(genmags)) and genmags.min() > 5 and genmags.max() < 45
    assert float(np.corrcoef(genmags, np.log10(genflams))[0, 1]) < -0.8


def test_sedtrue_cidran_and_nospec(snana_fixture, tmp_path):
    """CIDRAN(16): HEAD and SPEC share the remapped CIDs. noSPEC(2048): no SPEC.FITS."""
    fx = snana_fixture("sedtrue_replay")
    dump = load_snana_dump(str(fx.dump), dedup_cid=True)
    with fits.open(fx.head_fits) as hdul:
        head_cols = {n: hdul[1].data[n] for n in hdul[1].data.names}
    config = dict(n_sne=dump.size)
    sim = sim_run.prepare(str(fx.input_file), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    override = build_override_parameters(dump.columns, simulation, resources,
                                         head_columns=head_cols)
    results, sel, spectra_cache = sim_run.run_simulation(sim, override_parameters=override)

    def save(mask, ver):
        """format_mask is read only when writing, so one run serves both masks."""
        simulation.output.format_mask = mask
        save_snana_outputs(sim, results, ver, sel, spectra_by_sn=spectra_cache, output_dir=str(tmp_path))
        return tmp_path / ver

    # noSPEC first: the CIDRAN write renumbers results['parameters']['cid'] in place
    d2 = save(32 | 2048, "NS")
    assert os.path.exists(resolve_fits_path(d2 / "NS_HEAD.FITS"))
    assert not os.path.exists(resolve_fits_path(d2 / "NS_SPEC.FITS"))

    d = save(32 | 16, "CR")  # CIDRAN + FITS + spectra
    with fits.open(resolve_fits_path(d / "CR_HEAD.FITS")) as h:
        head = {int(str(s).strip()) for s in h[1].data['SNID']}
    with fits.open(resolve_fits_path(d / "CR_SPEC.FITS")) as h:
        spec = {int(str(s).strip()) for s in h[1].data['SNID']}
    assert head == spec and len(head) > 40
    assert sorted(head)[:5] != [1, 2, 3, 4, 5]  # CIDs randomized
