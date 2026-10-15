"""FORMAT_MASK=2 (TEXT) output parity vs live SNANA.

Replays the salt3_k21 events in jax with FORMAT_MASK=2 and compares the emitted
per-SN <SN>.DAT against SNANA's own TEXT output of the same seed (bundled in
text_ref/). Confirms the TEXT serialization — header KEY:value fields and the
OBS photometry rows — matches SNANA.
"""

import numpy as np
from astropy.io import fits

from jax_snlc_core.io.snana_dump_reader import load_snana_dump  # noqa: E402
from jax_snlc_fit.data_loader import parse_snana_dat  # noqa: E402
from replay_snana_dump import build_override_parameters  # noqa: E402
from jax_snlc_sim import run as sim_run  # noqa: E402
from _replay_helpers import select_replay  # noqa: E402


def _hfloat(hdr, key, idx=0):
    toks = hdr.get(key, '').split()
    return float(toks[idx]) if idx < len(toks) else float('nan')


def test_format_mask_text_replay(snana_fixture, tmp_path):
    fx = snana_fixture("salt3_k21_replay")
    ref_dir = fx.root / "text_ref"
    ref_snids = sorted(int(p.stem[2:]) for p in ref_dir.glob("SN*.DAT"))

    dump = load_snana_dump(str(fx.dump), dedup_cid=True)
    with fits.open(fx.head_fits) as hdul:
        head_cols = {n: hdul[1].data[n] for n in hdul[1].data.names}
    # only the events with a bundled TEXT reference are compared, so replay those
    dump_cols, head_cols = select_replay(dump.columns, head_cols, ref_snids)
    config = dict(n_sne=len(dump_cols['CID']))
    sim = sim_run.prepare(str(fx.input_file), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    simulation.output.format_mask = 2  # TEXT output
    override = build_override_parameters(dump_cols, simulation, resources, head_columns=head_cols)
    results, sel, spectra_cache = sim_run.run_simulation(sim, override_parameters=override)
    sim_run.save_snana_outputs(sim, results, "FMT", sel, spectra_cache=spectra_cache, output_dir=str(tmp_path))
    jax_dir = tmp_path / "FMT"

    common = [s for s in ref_snids if (jax_dir / f"FMT_SN{s:06d}.DAT").exists()]
    assert len(common) > 22

    hdr_absdiff = {k: [] for k in ('RA', 'DEC',
                                   'REDSHIFT_HELIO', 'REDSHIFT_FINAL')}
    peakmjd_absdiff = []
    simmag_absdiff, fluxcalerr_reldiff = [], []
    for sid in common:
        sh, sv, so = parse_snana_dat(ref_dir / f"SN{sid:06d}.DAT")
        jh, jv, jo = parse_snana_dat(jax_dir / f"FMT_SN{sid:06d}.DAT")
        assert int(sh['SNID']) == int(jh['SNID']) == sid
        assert int(float(sh['SNTYPE'].split()[0])) == int(float(jh['SNTYPE'].split()[0]))
        for k in hdr_absdiff:
            hdr_absdiff[k].append(abs(_hfloat(sh, k) - _hfloat(jh, k)))
        peakmjd_absdiff.append(abs(_hfloat(sh, 'PEAKMJD') - _hfloat(jh, 'PEAKMJD')))
        # OBS rows aligned by (BAND, MJD)
        def rows(v, o):
            return {(b, round(float(m), 2)): i for i, (b, m) in
                    enumerate(zip(o['BAND'], o['MJD']))}
        sr, jr = rows(sv, so), rows(jv, jo)
        for key in set(sr) & set(jr):
            si, ji = sr[key], jr[key]
            ms = float(so['SIM_MAGOBS'][si])
            mj = float(jo['SIM_MAGOBS'][ji])
            if ms < 90 and mj < 90:
                simmag_absdiff.append(abs(ms - mj))
            es, ej = float(so['FLUXCALERR'][si]), float(jo['FLUXCALERR'][ji])
            if es > 1.0:
                fluxcalerr_reldiff.append(abs(ej - es) / es)

    # header truth fields are exact (replay reproduces the events); PEAKMJD is
    # SNANA's fit-estimate (GENLC.PEAKMJD != SIM t0), so gated loosely
    assert max(np.median(v) for v in hdr_absdiff.values()) < 1e-3
    assert float(np.median(peakmjd_absdiff)) < 2.0
    # SIM_MAGOBS truth to mmag; FLUXCALERR is the deterministic noise model
    # (the noisy FLUXCAL itself is RNG-divergent between the codes)
    assert len(simmag_absdiff) > 80
    assert float(np.median(simmag_absdiff)) < 0.002
    assert float(np.median(fluxcalerr_reldiff)) < 0.02
