"""Chunk-size reproducibility: outputs are independent of batch_size. Truth
parameters and observation schedule are bit-identical (sampled once, then
sliced into batches); noisy fluxes match to float tolerance, since per-SN RNG keys
are batch-invariant via ``sn_global_idx`` + ``jax.random.fold_in``, but the
float math runs at a different batch dimension per batch.
"""

from pathlib import Path

import numpy as np
import pytest


from jax_snlc_sim import run as sim_run  # noqa: E402


N_SNE = 20


def _run(input_path: Path, *, batch_size: int, seed: int = 42):
    # every event's light curve is compared across batch sizes
    config = dict(
        n_sne=N_SNE,
        random_seed=seed, batch_size=batch_size, keep_rejected_events=True,
    )
    sim = sim_run.prepare(str(input_path), **config)
    results, _, _ = sim_run.run_simulation(sim)
    return results


@pytest.fixture(scope="module")
def batch_results(snana_fixture):
    """Run the simulator with two batch_size values and cache the results
    (both tests below share these runs). ``big`` is single-batch; ``small``
    forces multiple batches, which is what triggers the regression we
    guard against. The check is bit-identity, so N=20 is plenty."""
    fx = snana_fixture("searcheff_snii_basic")
    return {
        "big":   _run(fx.input_file, batch_size=N_SNE),     # 1 batch
        "small": _run(fx.input_file, batch_size=N_SNE // 2), # 2 batches
    }


def test_truth_params_invariant_under_batch_size(batch_results):
    """Per-SN truth parameters (z, mu, peak_mjd_true, AV, RV) are
    bit-identical regardless of batch_size, with no per-batch RNG bookkeeping
    leaks into the simulated truth."""
    for key in ("sim_redshift_cmb", "sim_redshift_helio", "mu",
                 "peakmjd_true", "av", "rv"):
        np.testing.assert_array_equal(
            np.asarray(batch_results["small"]["parameters"][key]),
            np.asarray(batch_results["big"]["parameters"][key]),
            err_msg=f"small vs big: parameters[{key!r}] depends on batch_size",
        )


def test_noise_invariant_under_batch_size(batch_results):
    for key in ("fluxcals_noisy", "fluxcalerrs", "snrs"):
        np.testing.assert_allclose(
            np.asarray(batch_results["small"][key]),
            np.asarray(batch_results["big"][key]),
            rtol=1e-5, atol=1e-6,
            err_msg=f"small vs big: {key} depends on batch_size",
        )


def test_results_are_per_sn(batch_results):
    """Every leaf of the results dict is indexed by SN, so operations on that
    axis need no exceptions. Run-level facts belong to the resources, not here."""
    def leaves(d, prefix=""):
        for k, v in d.items():
            path = f"{prefix}{k}"
            if isinstance(v, dict):
                yield from leaves(v, path + ".")
            elif v is not None:
                yield path, v

    offenders = [
        (path, getattr(v, "shape", type(v).__name__))
        for path, v in leaves(batch_results["small"])
        if getattr(v, "shape", None) is None or v.shape[:1] != (N_SNE,)
    ]
    assert not offenders, f"not indexed by SN: {offenders}"


def test_observation_schedule_invariant_under_batch_size(batch_results):
    """Observation schedule (per-SN MJDs, filter indices, libid, n_obs)
    is bit-identical regardless of batch_size in the valid-obs region.
    Padding-region MJDs are unspecified and may differ harmlessly."""
    sched_big = batch_results["big"]["observation_schedule"]
    sched_small = batch_results["small"]["observation_schedule"]
    n_obs = np.asarray(sched_big["n_obs_per_sn"])

    for key in ("filter_indices", "libid", "n_obs_per_sn"):
        np.testing.assert_array_equal(
            np.asarray(sched_small[key]),
            np.asarray(sched_big[key]),
            err_msg=f"sched[{key!r}] depends on batch_size",
        )

    mjds_big = np.asarray(sched_big["mjds"])
    mjds_small = np.asarray(sched_small["mjds"])
    for i in range(N_SNE):
        n = int(n_obs[i])
        np.testing.assert_array_equal(
            mjds_small[i, :n], mjds_big[i, :n],
            err_msg=f"SN {i}: valid-obs MJDs depend on batch_size",
        )
