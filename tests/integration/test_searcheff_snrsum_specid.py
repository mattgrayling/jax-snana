"""SEARCHEFF SPECID gated on the simulated spectrum's SNRSUM_REST_V.

SNANA generates the spectrum before gen_SEARCHEFF, so the spectroscopic-typing
efficiency map can be a function of the spectrum's rest-V SNR (brighter spectrum
-> higher typing efficiency). jax computes SNRSUM_REST_V for all candidates
before selection and feeds it to SPECID. snrsum reflects SN brightness (lower z
-> brighter -> higher snrsum), so a snrsum-gated map preferentially types the
brighter (lower-z) SNe.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from jax_snlc_sim import run as sim_run
from jax_snlc_sim.config_parser import parse_snana_input_file

_F = "$JAX_TEST_FIXTURE_ROOT"


SNRSUM_GATE = 20.0


_DOCANA = "DOCUMENTATION:\n    PURPOSE: test SEARCHEFF map\nDOCUMENTATION_END:\n\n"


def _write_specmap(out: Path, *, gated: bool) -> None:
    if gated:
        # Knife-edge step at the gate (1e-4 interp window) so typing is
        # deterministic: eff=0 below, eff=1 above — types only the brighter
        # SNe. The map is uniform in 1e-4 steps about the gate, and
        # OPT_EXTRAP: 1 carries its edge values beyond.
        knots = np.round(np.arange(SNRSUM_GATE - 0.1, SNRSUM_GATE + 0.10005,
                                   1e-4), 4)
        out.write_text(_DOCANA +
                       "OPT_EXTRAP: 1\n"
                       "VARNAMES: SNRSUM_REST_V SPECEFF\n"
                       + "".join(f"SPECEFF: {x:.4f} "
                                 f"{1.0 if x > SNRSUM_GATE else 0.0}\n"
                                 for x in knots))
    else:
        out.write_text(_DOCANA +
                       "VARNAMES: SNRSUM_REST_V SPECEFF\n"
                       "SPECEFF: 0.0     1.0\n"
                       "SPECEFF: 1.0e6   1.0\n")


_SPECTRUM_SOURCES = {
    # TAKE_SPECTRUM with an explicit spectrograph
    'take_spectrum': (
        f"SPECTROGRAPH_FILE: {_F}/spectrograph/SPECTROGRAPH_TEST.DAT\n"
        f"TAKE_SPECTRUM: TREST(-1:1) SNR_ZPOLY(30,0) SNR_LAMREST(5000:6000)\n"
        f"KCOR_FILE: {_F}/kcor/JAX_TEST_LSST.fits\n"
        f"SIMLIB_FILE: {_F}/simlib/JAX_TEST_LSST.SIMLIB\n"),
    # the SIMLIB's own SPECTROGRAPH rows
    'simlib': (
        "USE_SIMLIB_SPECTROGRAPH: 1\n"
        f"KCOR_FILE: {_F}/kcor/JAX_TEST_LSST_SPECTRO.fits\n"
        f"SIMLIB_FILE: {_F}/simlib/JAX_TEST_LSST_SIMSPEC.SIMLIB\n"),
    None: (
        f"KCOR_FILE: {_F}/kcor/JAX_TEST_LSST.fits\n"
        f"SIMLIB_FILE: {_F}/simlib/JAX_TEST_LSST.SIMLIB\n"),
}


def _write_input(out: Path, *, specmap: Path, source='take_spectrum',
                 n_sne: int = 120) -> None:
    spec_lines = _SPECTRUM_SOURCES[source]
    out.write_text(f"""\
GENVERSION: T
NGENTOT_LC: {n_sne}
FORMAT_MASK: 32
GENTYPE: 1
SNTYPES_Ia: 1 101
GENMODEL: {_F}/models/SALT3.K21
GENPEAK_SALT2x1: 0.0
GENRANGE_SALT2x1: -2 2
GENSIGMA_SALT2x1: 1 1
GENPEAK_SALT2c: 0.0
GENRANGE_SALT2c: -0.2 0.3
GENSIGMA_SALT2c: 0.05 0.05
GENPEAK_SALT2ALPHA: 0.14
GENPEAK_SALT2BETA: 3.1
OMEGA_MATTER: 0.315
OMEGA_LAMBDA: 0.685
w0_LAMBDA: -1.0
wa_LAMBDA: 0
GENFILTERS: ugrizY
{spec_lines}SEARCHEFF_SPEC_FILE: {specmap}
APPLY_SEARCHEFF_OPT: 2
GENRANGE_REDSHIFT: 0.08 0.45
GENRANGE_PEAKMJD: 60540 61460
GENSIGMA_REDSHIFT: 0.001
GENRANGE_TREST: -20 50
SOLID_ANGLE: 0.015
OPT_MWEBV: 0
DNDZ: POWERLAW 2.5E-5 1.5
RANSEED: 12345
""")


def _run(inp: Path):
    cfg = dict(n_sne=120, random_seed=12345,
                              keep_rejected_events=True)
    sim = sim_run.prepare(str(inp), **cfg)
    resources = sim
    results, selection, spectra_cache = sim_run.run_simulation(sim)
    return sim, results, selection, spectra_cache, resources


@pytest.mark.parametrize("source", ["take_spectrum", "simlib"])
def test_specid_gates_on_spectrum_snr(tmp_path, source):
    """A snrsum-gated SPEC map types exactly the SNe an all-pass map types
    that also clear the gate, whichever source requests the spectra.
    Requires SNRSUM_REST_V populated before selection for every candidate
    with a spectrum (SNANA generates spectra before gen_SEARCHEFF): all of
    them under TAKE_SPECTRUM, and under the SIMLIB's SPECTROGRAPH rows those
    with an epoch inside GENRANGE_TREST."""
    inp_gated = tmp_path / "gated.input"
    _write_input(inp_gated, specmap=tmp_path / "m_gated.DAT", source=source)
    _write_specmap(tmp_path / "m_gated.DAT", gated=True)
    sim_g, r_g, s_g, cache_g, resources = _run(inp_gated)

    inp_flat = tmp_path / "flat.input"
    _write_input(inp_flat, specmap=tmp_path / "m_flat.DAT", source=source)
    _write_specmap(tmp_path / "m_flat.DAT", gated=False)
    _, r_f, s_f, _, _ = _run(inp_flat)

    assert sim_g.specid_uses_snrsum_rest_v
    p_g = r_g['parameters']
    snr_g = np.asarray(p_g['snrsum_rest_v'])
    pf = np.asarray(p_g['prefilter_pass'])
    has_spectrum = np.ones_like(pf)
    if source == 'simlib':
        trest_lo, trest_hi = -20.0, 50.0
        peak = np.asarray(p_g['peakmjd_true'])
        z1 = 1.0 + np.asarray(p_g['sim_redshift_helio'])
        has_spectrum = np.array([
            bool(np.any((trest_lo <= (resources.fields[f].spec_mjds - m) / z)
                        & ((resources.fields[f].spec_mjds - m) / z <= trest_hi)))
            for f, m, z in zip(np.asarray(p_g['field_idx']), peak, z1)])
        assert 0 < has_spectrum[pf].sum() < pf.sum()
        # a spectrum whose rest-frame V bins all carry zero flux (a SIMLIB
        # epoch well before peak) sums to zero, as in SNANA's GENSPEC_SNR
        z1_cmb = 1.0 + np.asarray(p_g['sim_redshift_cmb'])
        cid = np.asarray(p_g['cid'])
        has_spectrum &= np.array([
            any(np.any((0.5 * (s.lam_min + s.lam_max) >= 5000.0 * z)
                       & (0.5 * (s.lam_min + s.lam_max) <= 6000.0 * z)
                       & (np.asarray(s.sim_flam) > 0))
                for s in cache_g.get(int(c), []))
            for c, z in zip(cid, z1_cmb)])
    assert np.array_equal(snr_g[pf] > 0, has_spectrum[pf]), (
        "snrsum must be populated for exactly the candidates with a spectrum")

    pass_g = np.asarray(s_g['passing_mask'])
    pass_f = np.asarray(s_f['passing_mask'])
    n_g, n_f = int(pass_g.sum()), int(pass_f.sum())
    assert n_g > 0, "gated map should still type the bright SNe"
    assert n_g < n_f, f"snrsum gate should type fewer ({n_g}) than all-pass ({n_f})"
    # Same seed → identical SNe in both runs, and the knife-edge map is
    # deterministic, so the typed sets must agree exactly.
    assert np.array_equal(pass_g, pass_f & (snr_g > SNRSUM_GATE)), (
        "gated typed set must equal the all-pass typed set above the gate"
    )


def test_no_spectrograph_snrsum_zero(tmp_path):
    """A SNRSUM_REST_V SPEC map with no spectrograph configured: snrsum stays
    0, so the gated map types nobody and nothing crashes."""
    specmap = tmp_path / "spec.DAT"
    _write_specmap(specmap, gated=True)
    inp = tmp_path / "input.input"
    _write_input(inp, specmap=specmap, source=None)
    sim, _, selection, _, _ = _run(inp)
    assert not sim.specid_uses_snrsum_rest_v
    assert int(np.asarray(selection['passing_mask']).sum()) == 0, (
        "snrsum=0 for every SN → SPECEFF=0 → nothing passes SPECID"
    )


def test_cidran_with_cached_spectra(tmp_path):
    """FORMAT_MASK CIDRAN(16) with the SNRSUM_REST_V cached-spectra path: HEAD and
    SPEC share the remapped CIDs (exercises _remap_spectra_cids on the spectra
    cache)."""
    from astropy.io import fits
    from jax_snlc_core.io.fits_path import resolve_fits_path
    inp = tmp_path / "cr.input"
    _write_input(inp, specmap=tmp_path / "m.DAT")
    _write_specmap(tmp_path / "m.DAT", gated=True)
    cfg = dict(n_sne=120, random_seed=12345)
    parsed = parse_snana_input_file(str(inp))
    parsed.output.format_mask = 32 | 16  # FITS + CIDRAN
    parsed.numerical = replace(parsed.numerical, n_sne=cfg["n_sne"])
    sim = sim_run.prepare(parsed)
    resources = sim
    results, sel, spectra_cache = sim_run.run_simulation(sim)
    assert sim.specid_uses_snrsum_rest_v and spectra_cache is not None
    sim_run.save_snana_outputs(sim, results, "CR", sel, spectra_cache=spectra_cache, output_dir=str(tmp_path))

    d = tmp_path / "CR"
    with fits.open(resolve_fits_path(d / "CR_HEAD.FITS")) as h:
        head = {int(str(s).strip()) for s in h[1].data['SNID']}
    with fits.open(resolve_fits_path(d / "CR_SPEC.FITS")) as h:
        spec = {int(str(s).strip()) for s in h[1].data['SNID']}
    assert head == spec and len(head) > 3
    assert sorted(head)[:3] != [1, 2, 3]  # CIDs randomized


def test_snrsum_specid_under_ngen_lc(tmp_path):
    """NGEN_LC with the SNRSUM_REST_V cached-spectra path.

    The loop generates past the target and drops the surplus, so snrsum must
    stay aligned with the surviving rows and the spectra cache must not keep
    entries for dropped SNe -- their CIDs can collide with the renumbered ones.
    """
    # gated map so only the brighter SNe type: the rejected attempts share
    # their accepted slot's CID, which is where mis-keying shows up
    specmap = tmp_path / "m_ngenlc.DAT"
    _write_specmap(specmap, gated=True)
    inp = tmp_path / "in_ngenlc.input"
    _write_input(inp, specmap=specmap)
    inp.write_text(inp.read_text().replace("NGENTOT_LC: 120", "NGEN_LC: 20"))

    cfg = dict(n_sne=20, random_seed=12345,
                              keep_rejected_events=True)
    sim = sim_run.prepare(str(inp), **cfg)
    resources = sim
    results, selection, spectra_cache = sim_run.run_simulation(sim)

    cid = np.asarray(results['parameters']['cid'])
    snrsum = np.asarray(results['parameters']['snrsum_rest_v'])
    passing = np.asarray(selection['passing_mask']).astype(bool)

    assert int(passing.sum()) == 20
    assert len(snrsum) == len(cid), "snrsum desynchronised from the kept rows"

    cached = spectra_cache or {}
    stale = set(cached) - set(cid.tolist())
    assert not stale, f"spectra cached for CIDs no longer present: {sorted(stale)[:8]}"
    assert int((~passing).sum()) > 5, "config must produce rejected attempts"
    shared = set(cid[~passing].tolist()) & set(cid[passing].tolist())
    assert len(shared) > 3, "rejected attempts must share an accepted CID"

    # TREST(-1:1) pins each spectrum to its own SN's PEAKMJD, and PEAKMJDs span
    # ~920 days, so a spectrum keyed to the wrong SN shows up as a huge offset.
    peak = {int(c): float(m) for c, m in
            zip(cid, np.asarray(results['parameters']['peakmjd_true']))}
    worst = max(abs(s.mjd - peak[int(c)])
                for c, specs in cached.items() for s in specs)
    assert worst < 30.0, f"spectrum attached to the wrong SN: |mjd-peak| = {worst:.1f}d"
