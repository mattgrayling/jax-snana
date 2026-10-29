"""SIMLIB_NSKIPMJD replay parity.

SIMLIB_NSKIPMJD thins SIMLIB epochs by a per-TREST-bin prescale
(``keep_SIMLIB_MJD``, snlc_sim.c:21250). It is fully deterministic given the
SIMLIB and the SN's PEAKMJD/zCMB, so replaying SNANA's own SNe must reproduce
its kept-epoch set exactly rather than only statistically.

The fixture uses ``1(0),4(10)`` -- keep every MJD before Trest=10, every 4th
after. That form is deliberately discriminating: against a single ``1(0)`` bin
it changes 59/60 SNe (265 -> 167 epochs), so the second bin, the
last-matching-TRESTMIN selection and the unique-MJD counter are all exercised.
A single-bin form such as ``2`` would not be -- with no TRESTMIN, TREST is never
consulted at all.

jax computes TREST heliocentrically, like every other Trest in both codes; SNANA
uses zCMB here alone (snlc_sim.c:21259), a deliberate divergence. This fixture
cannot see the difference: max|zCMB - zHELIO| = 0.0013 moves TREST by ~0.01 d
and no epoch lies within 0.5 d of a bin edge, so
the parity result below holds under either frame.
"""

import numpy as np
from astropy.io import fits
from replay_snana_dump import build_override_parameters

from jax_snlc_core.snana_files import load_snana_dump, resolve_fits_path
from jax_snlc_sim import run as sim_run
from jax_snlc_sim.output import save_snana_outputs


def _mjd_sets(head_path, phot_path):
    """{CID: sorted MJDs} from a SNANA-format HEAD/PHOT pair."""
    with fits.open(resolve_fits_path(head_path)) as h:
        head = h[1].data
        ptr_min = np.array(head['PTROBS_MIN'])
        ptr_max = np.array(head['PTROBS_MAX'])
        snid = np.array(head['SNID'])
        head_cols = {n: np.array(head[n]) for n in head.names}
    with fits.open(resolve_fits_path(phot_path)) as h:
        mjd = np.array(h[1].data['MJD'], dtype=float)
    out = {}
    for i in range(len(snid)):
        s = slice(int(ptr_min[i]) - 1, int(ptr_max[i]))
        out[int(str(snid[i]).strip())] = np.sort(mjd[s])
    return out, head_cols


def test_nskipmjd_kept_epochs_replay(snana_fixture, tmp_path):
    """Replaying SNANA's SNe reproduces its kept-epoch MJD set exactly."""
    fx = snana_fixture("nskipmjd_replay")
    s_mjds, head_cols = _mjd_sets(fx.head_fits, fx.phot_fits)
    dump = load_snana_dump(str(fx.dump), dedup_cid=True)

    config = dict(n_sne=dump.size)
    sim = sim_run.prepare(str(fx.input_file), **config)
    simulation = None if sim is None else sim.config
    resources = sim
    assert simulation.survey.simlib_nskipmjd == ((1, 0), (4, 10))

    override = build_override_parameters(dump.columns, simulation, resources,
                                         head_columns=head_cols)
    results, sel, spectra_cache = sim_run.run_simulation(sim, override_parameters=override)
    save_snana_outputs(sim, results, "JAXNSKIP", sel, spectra_by_sn=spectra_cache, output_dir=str(tmp_path))
    jdir = tmp_path / "JAXNSKIP"
    j_mjds, _ = _mjd_sets(jdir / "JAXNSKIP_HEAD.FITS",
                          jdir / "JAXNSKIP_PHOT.FITS")

    common = sorted(set(s_mjds) & set(j_mjds))
    assert len(common) > 50, f"only {len(common)} SNe matched by CID"
    mismatched = [c for c in common
                  if len(s_mjds[c]) != len(j_mjds[c])
                  or not np.allclose(s_mjds[c], j_mjds[c], atol=1e-3)]
    assert not mismatched, (
        f"{len(mismatched)}/{len(common)} SNe differ; first CID {mismatched[0]}: "
        f"SNANA {s_mjds[mismatched[0]]} vs jax {j_mjds[mismatched[0]]}")


def test_nskipmjd_second_bin_actually_bites(snana_fixture):
    """Guard the fixture's discriminating power: with 1(0),4(10) a sizeable
    fraction of epochs must sit past Trest=10, else the test proves nothing."""
    fx = snana_fixture("nskipmjd_replay")
    with fits.open(resolve_fits_path(fx.head_fits)) as h:
        head = h[1].data
        ptr_min, ptr_max = np.array(head['PTROBS_MIN']), np.array(head['PTROBS_MAX'])
        zcmb = np.array(head['SIM_REDSHIFT_CMB'], dtype=float)
        peak = np.array(head['SIM_PEAKMJD'], dtype=float)
    with fits.open(resolve_fits_path(fx.phot_fits)) as h:
        mjd = np.array(h[1].data['MJD'], dtype=float)
    late = total = 0
    for i in range(len(zcmb)):
        s = slice(int(ptr_min[i]) - 1, int(ptr_max[i]))
        trest = (mjd[s] - peak[i]) / (1.0 + zcmb[i])
        late += int((trest > 10.0).sum())
        total += len(trest)
    assert total > 100 and late > 0.15 * total
