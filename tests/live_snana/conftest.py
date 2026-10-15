"""Conftest for the opt-in live-SNANA tier.

Every test in this directory is automatically marked `live_snana`, so the
top-level conftest's auto-skip applies unless `--live-snana` (or
`RUN_LIVE_SNANA=1`) is set. The actual fixture-regeneration helpers are
added in plan phase 6.
"""

import os
from pathlib import Path

import pytest

_LIVE_DIR = Path(__file__).parent.resolve()


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    live = pytest.mark.live_snana
    for item in items:
        # Hook fires for the full item list; only mark items that actually
        # live under tests/live_snana/.
        if _LIVE_DIR in Path(item.fspath).resolve().parents:
            item.add_marker(live)


@pytest.fixture(scope="session")
def data_root(tmp_path_factory):
    """Give live SNANA runs plain assets; ordinary tests read the gzip files."""
    from fixtures.regenerate import prepare_snana_assets

    return prepare_snana_assets(tmp_path_factory.mktemp("snana-assets") / "fixtures")


@pytest.fixture(scope="session", autouse=True)
def _set_jax_test_fixture_root(data_root):
    previous = os.environ.get("JAX_TEST_FIXTURE_ROOT")
    os.environ["JAX_TEST_FIXTURE_ROOT"] = str(data_root)
    yield
    if previous is None:
        os.environ.pop("JAX_TEST_FIXTURE_ROOT", None)
    else:
        os.environ["JAX_TEST_FIXTURE_ROOT"] = previous
