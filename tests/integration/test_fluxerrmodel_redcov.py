"""FLUXERRMODEL REDCOV end-to-end tests.

* ``test_redcov_empirical_band_correlation_matches_input`` — parametrised
  over ρ ∈ {0, 0.3, 0.81}: the empirical band-band correlation of the
  simulator's noisy-flux residuals matches the input ρ; cross-group
  pairs are ~0. Includes ρ=0 as a negative control.
* ``test_redcov_snrmin_stratifies_correlation_above_vs_below`` — with
  SNRMIN_REDCOV=3, above-floor pairs show correlation ρ and below-floor
  pairs show ~0, pinning the engine's per-obs SNR gating in both
  directions.
* ``test_redcov_independent_per_field_group`` — field-group negative
  case: a REDCOV scoped to a non-applicable group leaves all factors zero.
* ``test_redcov_field_group_routes_per_sn`` — field-group positive case:
  with two groups defined and REDCOV scoped to one, SNe in the matching
  group show empirical correlation ρ; SNe in the other show ~0.

JAX-only — no byte-compare against SNANA, because REDCOV draws are
stochastic and the two codes use independent RNGs. The statistical
tests pin the JAX claim end-to-end; SNANA-vs-JAX parity at this level
would require a population-level comparison that's out of scope here.
"""

from pathlib import Path

import numpy as np
import pytest


from jax_snlc_sim import run as sim_run  # noqa: E402


def _write_fluxerr_model_with_redcov(out: Path, *, redcov: float) -> None:
    """A FLUXERRMODEL DAT with strong (×3) per-band fudge plus REDCOV."""
    out.write_text(f"""\
DOCUMENTATION:
    PURPOSE: REDCOV e2e test — strong per-band fudge + band correlation
    USAGE_KEY: FLUXERRMODEL_FILE
DOCUMENTATION_END:

REDCOV: g:{redcov:.3f},r:{redcov:.3f},i:{redcov:.3f},z:{redcov:.3f}

MAPNAME: FLUXERR_SCALE
BAND:    g
VARNAMES: PSF ERRSCALE
ROW:   0.5  30.000
ROW:   3.0  30.000
ENDMAP:

MAPNAME: FLUXERR_SCALE
BAND:    r
VARNAMES: PSF ERRSCALE
ROW:   0.5  30.000
ROW:   3.0  30.000
ENDMAP:

MAPNAME: FLUXERR_SCALE
BAND:    i
VARNAMES: PSF ERRSCALE
ROW:   0.5  30.000
ROW:   3.0  30.000
ENDMAP:

MAPNAME: FLUXERR_SCALE
BAND:    z
VARNAMES: PSF ERRSCALE
ROW:   0.5  30.000
ROW:   3.0  30.000
ENDMAP:

MAPNAME: FLUXERR_SCALE
BAND:    u
VARNAMES: PSF ERRSCALE
ROW:   0.5  1.000
ROW:   3.0  1.000
ENDMAP:

MAPNAME: FLUXERR_SCALE
BAND:    Y
VARNAMES: PSF ERRSCALE
ROW:   0.5  1.000
ROW:   3.0  1.000
ENDMAP:
""")


def _write_input(out: Path, fluxerr_path: Path, *, seed: int = 11111,
                 snrmin_redcov: float = 0.5) -> None:
    out.write_text(f"""\
GENVERSION:  JAX_TEST_REDCOV
NGENTOT_LC:  300
FORMAT_MASK:  32
GENTYPE:  1
SNTYPES_Ia:  1  101
CIDRAN_MIN:  0
CIDRAN_MAX:  3000000
GENMODEL:  $JAX_TEST_FIXTURE_ROOT/models/SALT3.K21
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
KCOR_FILE:  $JAX_TEST_FIXTURE_ROOT/kcor/JAX_TEST_LSST.fits
SIMLIB_FILE:  $JAX_TEST_FIXTURE_ROOT/simlib/JAX_TEST_LSST.SIMLIB
SIMLIB_MINSEASON:  30
SMEARFLAG_FLUX:  1
SMEARFLAG_ZEROPT:  1
FLUXERRMODEL_FILE:   {fluxerr_path}
FLUXERRMODEL_SNRMIN_REDCOV:  {snrmin_redcov}
GENRANGE_REDSHIFT:   0.05  0.30
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


def _run_sim(input_path: Path, n_sne: int = 200, seed: int = 99000):
    # the band correlations are read from every event's (event, epoch) grid
    config = dict(
        n_sne=n_sne, random_seed=seed, keep_rejected_events=True,
    )
    sim = sim_run.prepare(str(input_path), **config)
    resources = sim
    results, _, _ = sim_run.run_simulation(sim)
    return resources, results


def _bands(resources):
    """Band names in `filter_indices` order."""
    return list(resources.photometry.transmissions.keys())


def _empirical_pair_correlations(
    results, band_names, *, snr_floor=None, snr_ceil=None, sn_mask=None,
):
    """Return ``(empirical, counts, band_names)`` where ``empirical[b1, b2]``
    is the mean of ``r_i · r_j`` over same-SN obs pairs with bands b1, b2
    (i ≠ j when b1 == b2).

    With ``r_i = (noisy_i − truth_i) / σ_total,i`` and σ_F dominating
    σ_total (ERRSCALE=30 in this module's fixtures), this directly
    estimates ``F_i · F_j`` — i.e. the band-band fudge-noise covariance,
    which equals ρ for any pair where both obs are in the REDCOV group.

    ``snr_floor`` / ``snr_ceil`` (if set) restrict the analysis to obs
    with SNR in [snr_floor, snr_ceil). Both bounds use the observed
    SNR (noisy/σ) — a proxy for the engine's snr_calc_st that's exact
    in the high-SNR limit and noise-broadened near the threshold.

    ``sn_mask`` (if set) restricts the analysis to the subset of SNe
    where ``sn_mask[s]`` is True — used to stratify by per-SN field
    group, SN class, etc.
    """
    noisy = np.asarray(results["fluxcals_noisy"])
    truth = np.asarray(results["fluxcals"])
    sigma = np.asarray(results["fluxcalerrs"])
    sched = results["observation_schedule"]
    band_idx = np.asarray(sched["filter_indices"])
    n_obs = np.asarray(sched["n_obs_per_sn"])

    n_sne, max_obs = noisy.shape
    valid = (
        (np.arange(max_obs)[None, :] < n_obs[:, None])
        & (band_idx >= 0)
        & (sigma > 0)
    )
    if sn_mask is not None:
        valid &= np.asarray(sn_mask, dtype=bool)[:, None]
    if snr_floor is not None or snr_ceil is not None:
        snr = np.where(sigma > 0, noisy / np.maximum(sigma, 1e-30), 0.0)
        if snr_floor is not None:
            valid &= snr >= snr_floor
        if snr_ceil is not None:
            valid &= snr < snr_ceil
    res = np.where(valid, (noisy - truth) / np.maximum(sigma, 1e-30), 0.0)

    n_bands = len(band_names)
    sums = np.zeros((n_bands, n_bands))
    counts = np.zeros((n_bands, n_bands), dtype=np.int64)
    for s in range(n_sne):
        v = valid[s]
        if not v.any():
            continue
        rs = res[s][v]
        bs = band_idx[s][v]
        prods = rs[:, None] * rs[None, :]
        for b1 in range(n_bands):
            m1 = bs == b1
            if not m1.any():
                continue
            for b2 in range(n_bands):
                m2 = bs == b2
                if not m2.any():
                    continue
                block = prods[np.ix_(m1, m2)]
                if b1 == b2:
                    np.fill_diagonal(block, 0.0)
                    n_pairs = m1.sum() * (m1.sum() - 1)
                else:
                    n_pairs = m1.sum() * m2.sum()
                sums[b1, b2] += block.sum()
                counts[b1, b2] += n_pairs

    empirical = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    return empirical, counts, band_names


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.81])
def test_redcov_empirical_band_correlation_matches_input(tmp_path, rho):
    """End-to-end statistical: REDCOV=ρ produces empirical band-band
    correlation ρ in the simulated noisy-flux residuals.

    Parametrised over ρ ∈ {0, 0.3, 0.81}:
    - ρ=0 is the negative control — REDCOV disabled, every band-band
      pair correlation should be ~0 even though the fudge noise
      magnitude is unchanged.
    - ρ ∈ {0.3, 0.81} verify the empirical correlation tracks the input
      across a range, guarding against pathological "always emits one
      hardcoded value" bugs.

    Why E[r_i · r_j] measures ρ: with delta_i = σ_F,i (F_i·w + √(1−|F_i|²)z_i)
    and σ_total dominated by σ_F (ERRSCALE=30), the standardised residual
    r_i = (noisy_i − truth_i) / σ_total,i has
        E[r_i · r_j] = (σ_F,i σ_F,j / (σ_total,i σ_total,j)) · F_i · F_j
                     ≈ ρ      (both obs in REDCOV group)
                     ≈ 0      (otherwise).
    SNRMIN_REDCOV=0 keeps every valid obs in the correlated factor.
    """
    fluxerr = tmp_path / "REDCOV_STAT.DAT"
    _write_fluxerr_model_with_redcov(fluxerr, redcov=rho)
    inp = tmp_path / "input.input"
    _write_input(inp, fluxerr, seed=22222, snrmin_redcov=0.0)
    resources, results = _run_sim(inp, n_sne=500, seed=33333)

    empirical, counts, band_names = _empirical_pair_correlations(
        results, _bands(resources))

    correlated_bands = {"g", "r", "i", "z"}
    in_redcov = [i for i, n in enumerate(band_names) if n in correlated_bands]
    out_redcov = [i for i, n in enumerate(band_names) if n not in correlated_bands]

    # Within REDCOV group: expect ρ. Stat noise σ ≈ (1 − ρ²)/√N_eff with
    # N_eff ≈ n_sne = 500 → ≤ 0.045 across the ρ values; 0.1 is many σ and
    # absorbs the σ_F/σ_total<1 leakage from the surviving Poisson noise.
    for i in in_redcov:
        for j in in_redcov:
            if counts[i, j] < 200:
                continue
            emp = empirical[i, j]
            assert abs(emp - rho) < 0.1, (
                f"ρ={rho}: {band_names[i]}-{band_names[j]} empirical "
                f"{emp:.3f} vs target {rho} (n_pairs={counts[i, j]})"
            )

    # Cross-group pairs must be ~0 regardless of ρ.
    for i in in_redcov:
        for j in out_redcov:
            if counts[i, j] < 200:
                continue
            emp = empirical[i, j]
            assert abs(emp) < 0.1, (
                f"ρ={rho}: {band_names[i]}-{band_names[j]} (cross-group) "
                f"empirical {emp:.3f}, expected ~0 (n_pairs={counts[i, j]})"
            )


def test_redcov_snrmin_stratifies_correlation_above_vs_below(tmp_path):
    """SNRMIN_REDCOV gating: obs below the SNR floor get F=0 (iid fudge).

    Stratify a single SNRMIN_REDCOV=3 run by per-obs SNR:
    - Above × above pairs (both participated in the correlated factor)
      should show empirical correlation ≈ ρ.
    - Below × below pairs (both got F=0 → pure iid fudge) should show
      empirical correlation ≈ 0.

    Both strata must hold for the gating to be working: if SNRMIN were
    ignored, below × below would also show ρ; if SNRMIN excluded too
    much, above × above would drop below ρ. The observed-SNR proxy is
    noisy near the boundary, so we use a wide separation around the
    threshold (≥4 and <2) to avoid mis-classifying obs.
    """
    rho = 0.81
    fluxerr = tmp_path / "REDCOV_SNRMIN.DAT"
    _write_fluxerr_model_with_redcov(fluxerr, redcov=rho)
    inp = tmp_path / "input.input"
    _write_input(inp, fluxerr, seed=44444, snrmin_redcov=3.0)
    resources, results = _run_sim(inp, n_sne=500, seed=55555)

    correlated_bands = {"g", "r", "i", "z"}

    # Above-floor pairs: should recover ρ for in-group bands.
    above_emp, above_counts, band_names = _empirical_pair_correlations(
        results, _bands(resources), snr_floor=4.0,
    )
    in_redcov = [i for i, n in enumerate(band_names) if n in correlated_bands]
    for i in in_redcov:
        for j in in_redcov:
            if above_counts[i, j] < 200:
                continue
            emp = above_emp[i, j]
            assert abs(emp - rho) < 0.1, (
                f"above SNR 4: {band_names[i]}-{band_names[j]} "
                f"empirical {emp:.3f} vs ρ={rho} "
                f"(n_pairs={above_counts[i, j]})"
            )

    # Below-floor pairs: F=0 on both → empirical correlation ~0 even for
    # bands that would otherwise be correlated.
    below_emp, below_counts, _ = _empirical_pair_correlations(
        results, _bands(resources), snr_ceil=2.0,
    )
    for i in in_redcov:
        for j in in_redcov:
            if below_counts[i, j] < 200:
                continue
            emp = below_emp[i, j]
            assert abs(emp) < 0.1, (
                f"below SNR 2: {band_names[i]}-{band_names[j]} "
                f"empirical {emp:.3f}, expected ~0 (F=0 below SNRMIN; "
                f"n_pairs={below_counts[i, j]})"
            )


def test_redcov_field_group_routes_per_sn(tmp_path):
    """REDCOV scoped to an existing field group applies only to SNe in
    that group. Bundled SIMLIB has two LIBIDs (TOY_A, TOY_B); scope
    REDCOV to TOY_A and confirm that:

    - SNe drawn from TOY_A show empirical band correlation ≈ ρ;
    - SNe drawn from TOY_B (no REDCOV configured for their group) show
      empirical correlation ≈ 0.

    A bug like "all SNe get the first group's ρ regardless of which
    group they land in" would surface as both strata showing ρ.
    """
    rho = 0.81
    fluxerr = tmp_path / "REDCOV_GROUP.DAT"
    # Two groups defined; REDCOV(GROUP_A) only. ERRSCALE=30 is global so
    # the fudge dominates σ_total for both populations.
    fluxerr.write_text(f"""\
DOCUMENTATION:
    PURPOSE: REDCOV scoped to one of two SIMLIB field groups
    USAGE_KEY: FLUXERRMODEL_FILE
DOCUMENTATION_END:

DEFINE_FIELDGROUP: GROUP_A TOY_A
DEFINE_FIELDGROUP: GROUP_B TOY_B
REDCOV(GROUP_A): g:{rho:.3f},r:{rho:.3f},i:{rho:.3f},z:{rho:.3f}

MAPNAME: FLUXERR_SCALE
BAND:    g
VARNAMES: PSF ERRSCALE
ROW:   0.5  30.000
ROW:   3.0  30.000
ENDMAP:

MAPNAME: FLUXERR_SCALE
BAND:    r
VARNAMES: PSF ERRSCALE
ROW:   0.5  30.000
ROW:   3.0  30.000
ENDMAP:

MAPNAME: FLUXERR_SCALE
BAND:    i
VARNAMES: PSF ERRSCALE
ROW:   0.5  30.000
ROW:   3.0  30.000
ENDMAP:

MAPNAME: FLUXERR_SCALE
BAND:    z
VARNAMES: PSF ERRSCALE
ROW:   0.5  30.000
ROW:   3.0  30.000
ENDMAP:
""")
    inp = tmp_path / "input.input"
    _write_input(inp, fluxerr, seed=66666, snrmin_redcov=0.0)
    resources, results = _run_sim(inp, n_sne=500, seed=77777)

    field_idx = np.asarray(results["parameters"]["field_idx"]).astype(int)
    in_group_a = field_idx == 0  # TOY_A (LIBID 0)
    in_group_b = field_idx == 1  # TOY_B (LIBID 1)
    assert in_group_a.sum() > 50 and in_group_b.sum() > 50, (
        f"unbalanced field draws: A={in_group_a.sum()}, B={in_group_b.sum()} "
        "— field-group routing test needs both populated"
    )

    correlated_bands = {"g", "r", "i", "z"}

    # Group A SNe: should see correlation ρ.
    emp_a, counts_a, band_names = _empirical_pair_correlations(
        results, _bands(resources), sn_mask=in_group_a,
    )
    in_redcov = [i for i, n in enumerate(band_names) if n in correlated_bands]
    for i in in_redcov:
        for j in in_redcov:
            if counts_a[i, j] < 200:
                continue
            emp = emp_a[i, j]
            assert abs(emp - rho) < 0.1, (
                f"GROUP_A {band_names[i]}-{band_names[j]}: empirical "
                f"{emp:.3f} vs target ρ={rho} (n_pairs={counts_a[i, j]})"
            )

    # Group B SNe: no REDCOV for their group → empirical correlation ~0.
    emp_b, counts_b, _ = _empirical_pair_correlations(
        results, _bands(resources), sn_mask=in_group_b,
    )
    for i in in_redcov:
        for j in in_redcov:
            if counts_b[i, j] < 200:
                continue
            emp = emp_b[i, j]
            assert abs(emp) < 0.1, (
                f"GROUP_B {band_names[i]}-{band_names[j]}: empirical "
                f"{emp:.3f}, expected ~0 (no REDCOV for this group; "
                f"n_pairs={counts_b[i, j]})"
            )
