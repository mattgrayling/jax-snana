# Tests

Project test suite, run with `pytest`. Default invocation is hermetic —
no SNANA install or `$SNDATA_ROOT` required.

## Layout

- `unit/` — pure-function tests on synthetic inputs. No simulator
  instantiation, no fixture I/O. Includes `hypothesis` property-based
  tests for core helpers.
- `integration/` — full simulator (or fitter) runs against bundled
  fixtures, asserting **JAX-internal** invariants. No comparison to
  SNANA outputs.
- `parity/` — full simulator/fitter runs **compared against bundled
  SNANA outputs** (DUMP, HEAD/PHOT FITS, FITRES). The load-bearing
  cross-check that JAX matches SNANA.
- `smoke/` — import sanity + CLI subprocess end-to-end run.
- `perf/` — hardware-independent perf invariants (JIT cache hit ratio,
  recompile count).
- `live_snana/` — opt-in tier. Regenerates each fixture by running real
  SNANA and byte-compares the DUMP file against the bundled copy. Skipped
  by default; enable with `pytest --live-snana` (or `RUN_LIVE_SNANA=1`).
  Requires `snlc_sim.exe` on PATH plus `$SNANA_LSST_ROOT`/`$SNDATA_ROOT`
  set.
- `fixtures/` — bundled minimal test data + recorded SNANA outputs (see
  `fixtures/README.md`).

## Where to put a new test

| If your change | Add a test in |
|---|---|
| edits a pure helper / numerical kernel | `unit/` |
| changes simulator output for a given input.input | `integration/` (no SNANA comparison) **or** `parity/` (vs bundled SNANA) |
| changes how a config key parses | `unit/` plus an `integration/` test exercising it end-to-end |
| changes the FITS / DUMP / FITRES writer schema | `parity/` (the writer's parity test must still pass) |
| changes the fitter's recovered parameters | `parity/test_*_fitter.py` regression freeze must be updated |
| adds a new SNANA model implementation | bundle a small fixture in `fixtures/data/models/`, register a `parity/test_<model>_replay.py` SIM_MAGOBS comparison |
| adds a CLI script | `smoke/test_imports_and_cli.py` — add an end-to-end subprocess case |
| adds a hot path | optional `perf/` ratio or compile-count test |

## Running tests

```bash
# Install package + test deps (one-time)
pip install -e .[test]

# Default suite: unit, integration, parity, smoke and perf
pytest

# Just the units
pytest tests/unit

# Live-SNANA tier (needs SNANA + $SNDATA_ROOT)
pytest --live-snana

# In parallel. The suite is parallel-safe, but the default is serial so a run
# does not take over a working machine — choose a worker count, or set
# PYTEST_ADDOPTS="-n 4" in your shell profile. CI pins -n 1: two JAX workers
# exhaust a GitHub Actions runner.
pytest -n 4
```

Compiled XLA kernels persist under `platformdirs.user_cache_dir("jax_snana")`, so
runs after the first skip compilation (~37% faster). `JAX_COMPILATION_CACHE_DIR`
relocates it; deleting it forces a cold run.

## Regenerating fixtures

After upgrading SNANA, refresh recorded outputs:

```bash
python tests/fixtures/regenerate.py --list           # list registered fixtures
python tests/fixtures/regenerate.py --only NAME      # one fixture
python tests/fixtures/regenerate.py --all            # everything
python tests/fixtures/regenerate.py --diff           # regen to tmp + diff
```

The script pins the SNANA version in `tests/fixtures/snana_version.txt`
and fails loudly on a mismatch. Bump the pin with `--force-version-bump`
after auditing the diff.

## Adding a new parity fixture

1. Build an `input.input` under `tests/fixtures/snana/<name>/` that
   references bundled data via `$JAX_TEST_FIXTURE_ROOT`. The top-level
   `tests/conftest.py` sets that env var to `tests/fixtures/data` at
   session start.
2. Add `Fixture(name=..., genversion=...)` to the FIXTURES list in
   `tests/fixtures/regenerate.py`.
3. Run `python tests/fixtures/regenerate.py --only <name>` to capture the
   bundled SNANA outputs.
4. Add the parity test under `tests/parity/`. Use the `snana_fixture`
   pytest fixture from `tests/conftest.py` to load the bundle.

## Adding new bundled data

If a test needs new minimal data (extra HOSTLIB columns, a different
filter set, a new model), edit `tests/fixtures/build_toy_data.py` and
re-run it. Existing assets:

- Toy LSST SIMLIB (hand-crafted, 2 LIBIDs × ~50 obs cycling NEXPOSE)
- Cropped real LSST kcor (~475 KB)
- Cropped real NON1ASED V19 template (1 SED, INDEX 834)
- Real BayeSN.W22 (small enough to bundle as-is)
- LSST SEARCHEFF default DATs + a synthetic MAG-curve PHOTPROB DAT
