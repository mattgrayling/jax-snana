"""SPECTROGRAPH_OPTMASK / SCALE keys replay parity.

The fixture takes one spectrum per SN at a pinned epoch (TREST 5:5, so
both sides evaluate the same phase) with OPTMASK bit 2 (double wavelength
smearing), a wavelength-dependent SPECTROGRAPH_SCALE_SNR (1.0 + 2e-4*lam,
so the per-bin FLAMERR gate constrains the polynomial's lambda shape) and
SPECTROGRAPH_SCALE_TEXPOSE 1.3.
Verified SNANA semantics baked into the gates: SCALE_TEXPOSE scales only
SIMLIB-sourced exposures, never the TAKE_SPECTRUM value (Texpose stays
2000); the SIMLIB's TEMPLATE_TEXPOSE_SPECTROGRAPH key feeds only
SIMLIB-driven spectra, so template noise is OFF here — TAKE_SPECTRUM
grows a template only in SNR-solver mode (snlc_sim.c:11954). The reader
still parses the key (asserted) for future SIMLIB-driven support.

SIM_FLAM and FLAMERR are deterministic given the replayed parameters, so
both are gated per wavelength bin; the noisy FLAM is gated as pulls
against FLAMERR. A missing smear scale or SNR scale, or a spurious
template term, shifts FLAMERR by tens of percent — far above the gates.
"""

import os

import numpy as np
from _replay_helpers import load_jax_phot, load_snana_phot
from astropy.io import fits
from replay_snana_dump import build_override_parameters

from jax_snlc_core.snana_files import load_snana_dump, resolve_fits_path
from jax_snlc_sim import run as sim_run
from jax_snlc_sim.output import save_snana_outputs


def _load_spec(path):
    """Return {snid: (texpose, lammin, flam, flamerr, sim_flam)}."""
    out = {}
    with fits.open(resolve_fits_path(path)) as h:
        hdr = h[1].data
        flx = h[2].data
        for row in hdr:
            lo = int(row['PTRSPEC_MIN']) - 1
            hi = int(row['PTRSPEC_MAX'])
            out[int(str(row['SNID']).strip())] = (
                float(row['Texpose']),
                np.asarray(flx['LAMMIN'][lo:hi], dtype=float),
                np.asarray(flx['FLAM'][lo:hi], dtype=float),
                np.asarray(flx['FLAMERR'][lo:hi], dtype=float),
                np.asarray(flx['SIM_FLAM'][lo:hi], dtype=float),
            )
    return out


def _replay(fx, tmp_path, version):
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
    save_snana_outputs(sim, results, version, sel, spectra_by_sn=spectra_cache, output_dir=str(tmp_path))
    return simulation, resources


def _load_spec_multi(path):
    """Return {snid: [(mjd, texpose, lammin, flam, flamerr, sim_flam), ...]}
    sorted by MJD (for fixtures with several spectra per SN)."""
    out = {}
    with fits.open(resolve_fits_path(path)) as h:
        hdr = h[1].data
        flx = h[2].data
        for row in hdr:
            lo = int(row['PTRSPEC_MIN']) - 1
            hi = int(row['PTRSPEC_MAX'])
            out.setdefault(int(str(row['SNID']).strip()), []).append((
                float(row['MJD']), float(row['Texpose']),
                np.asarray(flx['LAMMIN'][lo:hi], dtype=float),
                np.asarray(flx['FLAM'][lo:hi], dtype=float),
                np.asarray(flx['FLAMERR'][lo:hi], dtype=float),
                np.asarray(flx['SIM_FLAM'][lo:hi], dtype=float),
            ))
    for v in out.values():
        v.sort(key=lambda t: t[0])
    return out


def test_spectro_snrmode_replay(snana_fixture, tmp_path):
    """SNR_ZPOLY exposure solving + per-SN template noise + TEXTRAP.

    Two TAKE_SPECTRUM configs: near-peak SNR_ZPOLY(25) whose solved
    exposure fixes the SN's template exposure (TEMPLATE_TEXPOSE_SCALE
    default 1.0, snlc_sim.c:11954), and a fixed 1500 s epoch that reuses
    it. OPTMASK 64 lets exposures extrapolate outside the calibration
    table (SNANA solved 47 s to 1.05e5 s here). Solver convergence is 3%
    in SNR on both sides, so exposures are gated at 15% per spectrum / 5%
    median and errors at 5%; a missing template term or sqrt(T) boost is
    a factor-level miss.
    """
    fx = snana_fixture("spectro_snrmode_replay")
    simulation, _ = _replay(fx, tmp_path, "SNRX")
    assert simulation.spectra.spectrograph_optmask == 64

    snana = _load_spec_multi(fx.root / "SPEC.FITS")
    jax_spec = _load_spec_multi(tmp_path / "SNRX" / "SNRX_SPEC.FITS")
    common = sorted(set(snana) & set(jax_spec))
    assert len(common) > 100

    # SNe whose near-peak solve stayed inside the calibration table have
    # well-defined template semantics on both sides; extrapolated solves
    # depend on SNANA's cross-event template state (documented divergence)
    # and are gated only directionally.
    def regime(t):
        return 'in' if 600.0 <= t <= 5400.0 else 'out'

    t_ratios, err_ratios, simflam_reldiff = [], [], []
    extrap_pairs, boundary_pairs = [], []
    for snid in common:
        assert len(snana[snid]) == len(jax_spec[snid]) == 2, snid
        t_near_s = snana[snid][0][1]
        t_near_j = jax_spec[snid][0][1]
        both_in = regime(t_near_s) == regime(t_near_j) == 'in'
        for (m_s, t_s, lam_s, _, err_s, sf_s), \
                (m_j, t_j, lam_j, _, err_j, sf_j) in zip(snana[snid],
                                                         jax_spec[snid]):
            both = np.intersect1d(lam_s, lam_j)
            i_s = np.isin(lam_s, both)
            i_j = np.isin(lam_j, both)
            ok = (sf_s[i_s] > 0) & (sf_j[i_j] > 0) & (err_s[i_s] > 0)
            simflam_reldiff.extend(np.abs(sf_j[i_j][ok] / sf_s[i_s][ok] - 1.0))
            if t_s == 1500.0:      # fixed epoch: exposure matches trivially,
                if both_in:        # errors gate the reused template
                    err_ratios.extend(err_j[i_j][ok] / err_s[i_s][ok])
                continue
            if regime(t_s) == regime(t_j) == 'in':
                t_ratios.append(t_j / t_s)
                err_ratios.extend(err_j[i_j][ok] / err_s[i_s][ok])
            elif regime(t_s) == regime(t_j) == 'out':
                extrap_pairs.append((t_s, t_j))
            else:
                boundary_pairs.append((t_s, t_j))

    # both in-range: deterministic replay, gates hug the realized values
    # (median 0.019 / p95 0.057 measured; structural bounds ~0.02 from the
    # <=1% per-bin SNR agreement and ~0.06 from one extra stopping-rule
    # iteration at SNANA's 3% SNR tolerance)
    t_ratios = np.asarray(t_ratios)
    assert len(t_ratios) > 30
    assert float(np.median(np.abs(t_ratios - 1.0))) < 0.025, \
        f"texpose median |ratio-1| {np.median(np.abs(t_ratios - 1.0)):.3f}"
    assert float(np.percentile(np.abs(t_ratios - 1.0), 95)) < 0.08
    assert float(np.max(np.abs(t_ratios - 1.0))) < 0.12
    assert abs(float(np.median(err_ratios)) - 1.0) < 0.05
    assert float(np.median(simflam_reldiff)) < 0.01

    # both extrapolated (TEXTRAP live on both sides): same side of the
    # table, within the stale-template envelope. SNANA's bound SNR carries
    # template noise from a PREVIOUS event (a deliberate divergence); a short
    # stale template is zp-rescaled by
    # 10^(0.8 dZP) = (T_max/T_t)^2, adding up to (T_max/T_min) extra
    # variance: SNR down sqrt(10), texpose up 10x after the quadratic.
    assert len(extrap_pairs) > 50
    for t_s, t_j in extrap_pairs:
        assert (t_s < 600.0) == (t_j < 600.0), (t_s, t_j)
        assert abs(np.log10(t_j / t_s)) < 1.0, (t_s, t_j)

    # regime flips: SNANA's stale-template bounds SNR shifts its in/out
    # decision. The in-range side must sit within the solver band of the
    # edge; the extrapolated side within the stale-template envelope of
    # the edge (same sqrt(10)-in-SNR bound as above).
    assert len(boundary_pairs) < 0.25 * len(common)
    for t_s, t_j in boundary_pairs:
        t_in, t_out = (t_s, t_j) if 600.0 <= t_s <= 5400.0 else (t_j, t_s)
        edge = 600.0 if t_out < 600.0 else 5400.0
        assert abs(np.log10(t_in / edge)) < 1.0, (t_s, t_j)
        assert abs(np.log10(t_out / edge)) < 1.0, (t_s, t_j)


def test_spectro_bits_replay(snana_fixture, tmp_path):
    """OPTMASK 13 = no smearing (bit 1) + LAMSPIKE (bit 4) + SNR x100
    (bit 8): flux confined to exactly the centre bin, errors /100."""
    fx = snana_fixture("spectro_bits_replay")
    simulation, _ = _replay(fx, tmp_path, "BITSX")
    assert simulation.spectra.spectrograph_optmask == 13

    snana = _load_spec(fx.root / "SPEC.FITS")
    jax_spec = _load_spec(tmp_path / "BITSX" / "BITSX_SPEC.FITS")
    common = sorted(set(snana) & set(jax_spec))
    assert len(common) > 100

    err_ratios, simflam_reldiff = [], []
    for snid in common:
        _, lam_s, _, err_s, sf_s = snana[snid]
        _, lam_j, _, err_j, sf_j = jax_spec[snid]
        # LAMSPIKE + no smear: exactly one surviving bin on both sides
        assert len(lam_s) == 1 and len(lam_j) == 1, snid
        assert lam_s[0] == lam_j[0]
        simflam_reldiff.append(abs(sf_j[0] / sf_s[0] - 1.0))
        if err_s[0] > 0:
            err_ratios.append(err_j[0] / err_s[0])

    assert float(np.median(simflam_reldiff)) < 0.01
    assert abs(float(np.median(err_ratios)) - 1.0) < 0.03


def test_spectro_nospec(snana_fixture, tmp_path):
    """OPTMASK bit 2048 (NOSPEC): jax writes no spectra."""
    import re
    fx = snana_fixture("spectro_bits_replay")
    inp = tmp_path / "nospec.input"
    inp.write_text(re.sub(r"SPECTROGRAPH_OPTMASK:\s+\d+",
                          "SPECTROGRAPH_OPTMASK:  2048",
                          fx.input_file.read_text()))
    # nothing about the replayed truth matters here — only that no SPEC file
    # is written, so generate a handful of events instead of the full dump
    config = dict(n_sne=5)
    sim = sim_run.prepare(str(inp), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    results, sel, spectra_cache = sim_run.run_simulation(sim)
    save_snana_outputs(sim, results, "NOSPEC", sel, spectra_by_sn=spectra_cache, output_dir=str(tmp_path))
    assert not os.path.exists(
        resolve_fits_path(tmp_path / "NOSPEC" / "NOSPEC_SPEC.FITS"))


def test_spectro_optmask_replay(snana_fixture, tmp_path):
    fx = snana_fixture("spectro_optmask_replay")
    simulation, resources = _replay(fx, tmp_path, "SPECX")
    assert simulation.spectra.spectrograph_optmask == 2
    assert simulation.spectra.spectrograph_scale_texpose == 1.3
    assert simulation.spectra.spectrograph_scale_snr == '1.0,0.0002'
    assert resources.simlib_template_texpose == 600.0

    snana = _load_spec(fx.root / "SPEC.FITS")
    jax_spec = _load_spec(tmp_path / "SPECX" / "SPECX_SPEC.FITS")

    common = sorted(set(snana) & set(jax_spec))
    assert len(common) > 100, f"only {len(common)} common spectra"

    simflam_reldiff, err_ratios, pulls = [], [], []
    for snid in common:
        t_s, lam_s, flam_s, err_s, sf_s = snana[snid]
        t_j, lam_j, flam_j, err_j, sf_j = jax_spec[snid]
        assert t_j == t_s, (snid, t_j, t_s)
        # align on common wavelength bins (each side drops zero-flux bins)
        both = np.intersect1d(lam_s, lam_j)
        i_s = np.isin(lam_s, both)
        i_j = np.isin(lam_j, both)
        ok = (sf_s[i_s] > 0) & (sf_j[i_j] > 0) & (err_s[i_s] > 0)
        simflam_reldiff.extend(
            np.abs(sf_j[i_j][ok] / sf_s[i_s][ok] - 1.0))
        err_ratios.extend(err_j[i_j][ok] / err_s[i_s][ok])
        pulls.extend((flam_j[i_j][ok] - sf_j[i_j][ok]) / err_j[i_j][ok])

    simflam_reldiff = np.asarray(simflam_reldiff)
    err_ratios = np.asarray(err_ratios)
    pulls = np.asarray(pulls)
    assert len(simflam_reldiff) > 2000

    # deterministic gates: true spectrum and its error
    assert float(np.median(simflam_reldiff)) < 0.01, \
        f"SIM_FLAM median rel diff {np.median(simflam_reldiff):.4f}"
    med_ratio = float(np.median(err_ratios))
    assert abs(med_ratio - 1.0) < 0.03, f"FLAMERR median ratio {med_ratio:.3f}"

    # noisy FLAM: pulls vs FLAMERR must be ~N(0,1)
    assert abs(float(np.mean(pulls))) < 4.0 / np.sqrt(len(pulls))
    assert abs(float(np.std(pulls)) - 1.0) < 0.05


def test_spectro_mjdstep_replay(snana_fixture, tmp_path):
    """TAKE_SPECTRUM MJD(min:max:step) is one spectrum per MJD
    (expand_TAKE_SPECTRUM_MJD, snlc_sim.c:5967-5999), each kept only inside
    the SN's GENRANGE_TREST window, so every SN here gets at most one of the
    three. The set of (SN, MJD) spectra, the true spectrum and its error per
    bin must match SNANA's."""
    fx = snana_fixture("spectro_mjdstep_replay")
    simulation, _ = _replay(fx, tmp_path, "MJDX")
    assert [e['epoch_range'] for e in simulation.spectra.take_spectrum] == [
        [m, m] for m in (60600.0, 60950.0, 61300.0)]

    snana = _load_spec_multi(fx.root / "SPEC.FITS")
    jax_spec = _load_spec_multi(tmp_path / "MJDX" / "MJDX_SPEC.FITS")
    assert len(snana) > 30
    assert ({s: [t[0] for t in v] for s, v in jax_spec.items()}
            == {s: [t[0] for t in v] for s, v in snana.items()})

    simflam_reldiff, err_ratios = [], []
    for snid, rows in snana.items():
        for (_, t_s, lam_s, _, err_s, sf_s), (_, t_j, lam_j, _, err_j, sf_j) \
                in zip(rows, jax_spec[snid]):
            assert t_j == t_s == 2000.0
            both = np.intersect1d(lam_s, lam_j)
            i_s, i_j = np.isin(lam_s, both), np.isin(lam_j, both)
            ok = (sf_s[i_s] > 0) & (sf_j[i_j] > 0) & (err_s[i_s] > 0)
            simflam_reldiff.extend(np.abs(sf_j[i_j][ok] / sf_s[i_s][ok] - 1.0))
            err_ratios.extend(err_j[i_j][ok] / err_s[i_s][ok])
    assert float(np.median(simflam_reldiff)) < 0.01
    assert abs(float(np.median(err_ratios)) - 1.0) < 0.03


def test_simlib_spectro_replay(snana_fixture, tmp_path):
    """SIMLIB SPECTROGRAPH-driven spectra: epochs + raw exposures come from the
    SIMLIB (USE_SIMLIB_SPECTROGRAPH), not TAKE_SPECTRUM. SIM_FLAM is deterministic."""
    fx = snana_fixture("simlib_spectro_replay")
    simulation, _ = _replay(fx, tmp_path, "SIMSPEC")
    assert simulation.survey.use_simlib_spectrograph

    snana = _load_spec_multi(fx.root / "SPEC.FITS")
    jax_spec = _load_spec_multi(tmp_path / "SIMSPEC" / "SIMSPEC_SPEC.FITS")
    common = sorted(set(snana) & set(jax_spec))
    assert len(common) > 15

    simflam_reldiff, err_ratios, texpose_ok, edge_ratios = [], [], [], []
    for snid in common:
        assert len(snana[snid]) == len(jax_spec[snid]), snid
        for (m_s, t_s, lam_s, _, err_s, sf_s), (m_j, t_j, lam_j, _, err_j, sf_j) in zip(
                snana[snid], jax_spec[snid]):
            texpose_ok.append(abs(m_s - m_j) < 0.01 and abs(t_s - t_j) < 0.01)
            both = np.intersect1d(lam_s, lam_j)
            i_s, i_j = np.isin(lam_s, both), np.isin(lam_j, both)
            ok = (sf_s[i_s] != 0) & (err_s[i_s] > 0)
            simflam_reldiff.extend(np.abs(sf_j[i_j][ok] / sf_s[i_s][ok] - 1.0))
            r = err_j[i_j][ok] / err_s[i_s][ok]
            err_ratios.extend(r)
            if len(r) > 4:
                edge_ratios.extend([r[0], r[-1]])
    assert all(texpose_ok)
    # Deterministic replay: SIM_FLAM and the (template-free) FLAMERR match tightly.
    assert float(np.median(simflam_reldiff)) < 0.01
    assert float(np.percentile(simflam_reldiff, 95)) < 0.03
    assert abs(float(np.median(err_ratios)) - 1.0) < 0.02
    # The outermost bins are the only ones the extended edge bins smear into
    # (extend_spectrograph_lambins), so they are where a missing extension or a
    # renormalised kernel shows up; the medians above average it away.
    assert len(edge_ratios) > 20
    assert abs(float(np.median(edge_ratios)) - 1.0) < 0.02

    # Each SPECTROGRAPH epoch also yields synthetic-band photometry (pqst).
    # SIM_MAGOBS is deterministic; the bluest band (p) inherits the far-UV
    # SED-integration systematic, so gate on the median.
    snana_phot = load_snana_phot(fx.head_fits, fx.phot_fits)
    jax_phot = load_jax_phot(tmp_path / "SIMSPEC", "SIMSPEC")
    mag_absdiff = []
    for key, (mag_s, _, _, _) in snana_phot.items():
        if key[1] not in "pqst" or mag_s >= 90.0:
            continue
        assert key in jax_phot, key
        mag_j = jax_phot[key][0]
        if mag_j >= 90.0:
            continue
        mag_absdiff.append(abs(mag_j - mag_s))
    assert len(mag_absdiff) > 100
    assert float(np.median(mag_absdiff)) < 0.005
    assert float(np.percentile(mag_absdiff, 95)) < 0.01


def test_spectro_snrscale_replay(snana_fixture, tmp_path):
    """SCALE_SNR must be applied when SOLVING for the exposure, not only when
    the spectrum is written.

    SPECTROGRAPH_OPTMASK bit 8 sets a fixed SNR x100 (SPECTROGRAPH_OPTMASK_SNRx100,
    snlc_sim.h:346). SNANA folds it into GENPOLY_SCALE_SNR (snlc_sim.c:10348) and
    applies it inside GENSPEC_SMEAR (:12184) -- which is the very function its
    exposure solver calls for SNR_MIN/SNR_MAX, so its solve and its output agree
    by construction.

    No other bundled fixture combines a SCALE_SNR with SNR_ZPOLY: the SNR-mode
    fixture uses OPTMASK 64 (no scale), and the two bit-8 fixtures use
    TEXPOSE_ZPOLY so no solve runs. With the scale missing from the solve, the
    search sees an SNR 100x too small, never reaches the requested 25, and rails
    to the top of the exposure table (5400 s) where SNANA takes the floor (600 s).
    """
    fx = snana_fixture("spectro_snrscale_replay")
    simulation, _ = _replay(fx, tmp_path, "SNRSC")
    assert simulation.spectra.spectrograph_optmask & 8, "fixture must set the x100 bit"

    snana = _load_spec_multi(fx.root / "SPEC.FITS")
    jax_spec = _load_spec_multi(tmp_path / "SNRSC" / "SNRSC_SPEC.FITS")
    common = sorted(set(snana) & set(jax_spec))
    assert len(common) > 100

    t_s = np.array([snana[c][0][1] for c in common])
    t_j = np.array([jax_spec[c][0][1] for c in common])
    # the request (25) is far below what even the shortest exposure delivers once
    # the x100 is applied, so both engines must select the table minimum
    assert np.allclose(t_s, 600.0), "SNANA reference should sit at the table floor"
    np.testing.assert_allclose(t_j, t_s, rtol=1e-6)
