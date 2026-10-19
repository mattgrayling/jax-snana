"""Top-level pytest conftest.

Sets `JAX_TEST_FIXTURE_ROOT` so that input files in `tests/fixtures/snana/`
can reference data via `$JAX_TEST_FIXTURE_ROOT/...`. Both SNANA (env-var
substitution in input files) and JAX (`os.path.expandvars` in
`builder.py`) honour the variable.

Also registers the `--live-snana` flag and the auto-skip behaviour for
`live_snana`-marked tests, and provides the `snana_fixture` factory that the
integration and parity tiers load their recorded SNANA bundles through.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import jax
import pytest

from jax_snlc_core.io.fits_path import resolve_fits_path
from jax_snlc_core.utils.precision import configure_precision

# Persist XLA compilation across runs. The 1.0s default threshold would store
# almost nothing from this suite; 0.1s keeps the fit and simulation kernels,
# which take seconds, and skips the single operations, which compile in
# milliseconds and would otherwise be nine entries in ten. The cap evicts the
# least recently used entries so the cache cannot grow without bound.
if not jax.config.jax_compilation_cache_dir:
    from platformdirs import user_cache_dir
    jax.config.update("jax_compilation_cache_dir",
                      str(Path(user_cache_dir("jax_snana")) / "compilation"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.1)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
jax.config.update("jax_compilation_cache_max_size", 2 * 1024 ** 3)

REPO_ROOT = Path(__file__).parent.parent
FIXTURE_ROOT = Path(__file__).parent / "fixtures"
DATA_ROOT = FIXTURE_ROOT / "data"
FIXTURE_SNDATA_ROOT = DATA_ROOT / "sndata_root"

# SNANA's $SNDATA_ROOT is a hard runtime requirement (production code reads
# reference tables from it at module-import time, e.g. Sersic_bn.dat). If a
# real SNANA install is present in the developer's environment, use that; on
# CI runners without SNANA, point at a minimal bundled SNANA-shaped tree
# containing just the reference files our test surface needs. Set here at
# conftest module-import time — must run before any test module is collected
# (which triggers the import-time reads). Empty-string and unset both treated
# as "not configured".
if not os.environ.get("SNDATA_ROOT"):
    os.environ["SNDATA_ROOT"] = str(FIXTURE_SNDATA_ROOT)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--live-snana",
        action="store_true",
        default=False,
        help="Run tests marked `live_snana` (requires SNANA + $SNDATA_ROOT).",
    )


def pytest_configure(config: pytest.Config) -> None:
    # The suite runs in double precision whatever the default: the derivative
    # checks need float64's rounding. Set before any test module is imported.
    configure_precision(True)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--live-snana") or os.environ.get("RUN_LIVE_SNANA") == "1":
        return
    skip_live = pytest.mark.skip(
        reason="live_snana suite requires --live-snana (or RUN_LIVE_SNANA=1)"
    )
    for item in items:
        if "live_snana" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture(scope="session", autouse=True)
def _set_jax_test_fixture_root() -> None:
    """Point input files at the bundled minimal data tree."""
    os.environ["JAX_TEST_FIXTURE_ROOT"] = str(DATA_ROOT)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def fixture_root() -> Path:
    return FIXTURE_ROOT


@pytest.fixture(scope="session")
def data_root() -> Path:
    return DATA_ROOT


@dataclass(frozen=True)
class SnanaFixture:
    """A bundle of SNANA outputs recorded for one parity scenario."""

    name: str
    root: Path

    @property
    def input_file(self) -> Path:
        return self.root / "input.input"

    @property
    def dump(self) -> Path:
        return Path(resolve_fits_path(self.root / "DUMP"))

    @property
    def head_fits(self) -> Path:
        return Path(resolve_fits_path(self.root / "HEAD.FITS"))

    @property
    def phot_fits(self) -> Path:
        return Path(resolve_fits_path(self.root / "PHOT.FITS"))

    @property
    def readme(self) -> Path:
        return self.root / "README.md"


@pytest.fixture(scope="session")
def snana_fixture(fixture_root: Path):
    """Return a loader function: `fx = snana_fixture("dumpall_basic")`."""
    snana_root = fixture_root / "snana"

    def _load(name: str) -> SnanaFixture:
        root = snana_root / name
        if not root.is_dir():
            raise FileNotFoundError(
                f"SNANA fixture {name!r} not found at {root}. "
                f"Run `python tests/fixtures/regenerate.py --only {name}` "
                f"to create it (requires SNANA installed)."
            )
        return SnanaFixture(name=name, root=root)

    return _load


# Per-resource fixtures — for paths used directly by multiple test
# modules. Tests should consume these instead of constructing the path
# from data_root each time.

@pytest.fixture(scope="session")
def lsst_kcor_path(data_root) -> Path:
    return data_root / "kcor" / "JAX_TEST_LSST.fits"


@pytest.fixture(scope="session")
def lsst_simlib_path(data_root) -> Path:
    return data_root / "simlib" / "JAX_TEST_LSST.SIMLIB"


@pytest.fixture(scope="session")
def jax_test_hostlib_path(data_root) -> Path:
    return data_root / "hostlib" / "JAX_TEST.HOSTLIB"


@pytest.fixture(scope="session")
def mixed_primaries_kcor_path(data_root) -> Path:
    return data_root / "kcor" / "JAX_TEST_MIXED_PRIMARIES.fits"


@pytest.fixture(scope="session")
def searcheff_data_dir(data_root) -> Path:
    return data_root / "searcheff"


@pytest.fixture(scope="session")
def spectrograph_data_dir(data_root) -> Path:
    return data_root / "spectrograph"


@pytest.fixture(scope="module", autouse=True)
def _clear_jax_caches_after_module():
    """Drop JAX's compiled-XLA cache between test modules.

    Each test module exercises the simulator with its own input shapes;
    JAX caches every compiled kernel by `(function, input signature)` and
    never evicts. On a 380+ test suite that's gigabytes of accumulated
    XLA programs by the time we hit the parity tier. Clearing the cache
    between modules pays a small recompilation cost on the first test of
    each new module in exchange for bounded memory growth.
    """
    yield
    jax.clear_caches()


@pytest.fixture(scope="session")
def bbc_biascor_variants(tmp_path_factory):
    """Derived BBC inputs; independent SNANA reference outputs stay in the bundle."""
    from fixtures.snana.bbc_des_beams.build import BIASCOR_VARIANTS, build

    directory = tmp_path_factory.mktemp("bbc_biascor_variants")
    tables, _ = build()
    for name in BIASCOR_VARIANTS:
        (directory / name).write_text(tables[name])
    return directory
